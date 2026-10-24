import numpy as np
import pytest
import threading
from scipy.spatial import cKDTree
from types import SimpleNamespace

from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilter,
    RM65CapturedRobotObservationFilterConfig,
    RM65CapturedRobotObservationFilterResult,
    captured_robot_observation_mask,
    _surface_samples,
    _asymmetric_surface_membership,
)


def test_capture_robot_mask_uses_bounded_silhouette_and_depth_consistency():
    depth = np.ones((5, 6), dtype=float)
    depth[2, 4] = 0.95
    target = np.zeros(depth.shape, dtype=bool)
    target[2, 1] = True
    rendered = np.full(depth.shape, np.nan, dtype=float)
    rendered[2, 2] = 1.005

    removed = captured_robot_observation_mask(
        depth_m=depth,
        target_mask=target,
        rendered_depth_maps_m=(rendered,),
        silhouette_radius_px=2.0,
        depth_tolerance_m=0.015,
    )

    assert removed[2, 2]
    assert removed[1, 2]
    assert not removed[2, 1]
    assert not removed[2, 4]
    assert not removed[0, 0]


def test_capture_robot_mask_unions_rendered_finger_states_without_removing_target():
    depth = np.ones((3, 5), dtype=float)
    target = np.zeros(depth.shape, dtype=bool)
    target[1, 3] = True
    first = np.full(depth.shape, np.nan, dtype=float)
    second = np.full(depth.shape, np.nan, dtype=float)
    first[1, 1] = 1.0
    second[1, 3] = 1.0

    removed = captured_robot_observation_mask(
        depth_m=depth,
        target_mask=target,
        rendered_depth_maps_m=(first, second),
        silhouette_radius_px=0.0,
        depth_tolerance_m=0.001,
    )

    np.testing.assert_array_equal(
        removed,
        np.asarray(
            [
                [False, False, False, False, False],
                [False, True, False, False, False],
                [False, False, False, False, False],
            ]
        ),
    )


@pytest.mark.parametrize(
    "fields",
    (
        {"silhouette_radius_px": -1},
        {"silhouette_radius_px": 16.1},
        {"depth_tolerance_m": 0},
        {"depth_tolerance_m": 0.0151},
        {"finger_sweep_q_rad": (0.0, 0.5)},
        {"finger_sweep_q_rad": (0.0, 0.864, 0.432)},
    ),
)
def test_capture_robot_filter_config_is_bounded(fields):
    with pytest.raises(ValueError):
        RM65CapturedRobotObservationFilterConfig(**fields)


def test_capture_robot_filter_result_copies_arrays_and_freezes_evidence():
    depth = np.ones((2, 2), dtype=float)
    removed = np.zeros((2, 2), dtype=bool)
    result = RM65CapturedRobotObservationFilterResult(
        depth_m=depth,
        removed_mask=removed,
        evidence={"removed_background_points": 0, "removed_target_points": 0},
    )
    depth[0, 0] = 2
    removed[0, 0] = True

    assert result.depth_m[0, 0] == 1
    assert not result.removed_mask[0, 0]
    with pytest.raises(ValueError):
        result.depth_m[0, 0] = 3
    with pytest.raises(TypeError):
        result.evidence["removed_background_points"] = 1


def _mesh_cache_renderer(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="test">
      <link name="world"/>
      <link name="left_sensor"><visual/></link><link name="left_optical"/>
      <link name="right_sensor"><visual/></link><link name="right_optical"/>
      <link name="head_optical"><visual/></link>
      <joint name="left_optical_joint" type="fixed"><parent link="left_sensor"/><child link="left_optical"/></joint>
      <joint name="right_optical_joint" type="fixed"><parent link="right_sensor"/><child link="right_optical"/></joint>
    </robot>""")
    renderer = RM65CapturedRobotObservationFilter(urdf_path=path)
    # Exercise real mesh assembly and filtering with deterministic planar ray
    # hits; actual CAD/Embree replay is a separate offline acceptance check.
    def mesh(name, frame, depth):
        vertices = np.array([[-1., -1., depth], [1., -1., depth], [0., 1., depth]])
        return SimpleNamespace(name=name, parentFrame=frame, geometry=SimpleNamespace(
            vertices=lambda: vertices, num_tris=1, tri_indices=lambda _: (0, 1, 2)))
    items = (
        mesh("left_sensor_0", 1, 2.3e-7),
        mesh("right_sensor_0", 2, .25),
        mesh("left_mount_0", 3, .35),
        mesh("left_eg2_4b2_cad_0", 4, .45),
    )
    model = SimpleNamespace(getFrameId=lambda name: {"left_sensor": 1, "right_sensor": 2, "head_optical": 5}[name],
                            createData=SimpleNamespace)
    visual = SimpleNamespace(geometryObjects=items)
    pin = SimpleNamespace(
        GeometryData=lambda _: SimpleNamespace(oMg=[SimpleNamespace(rotation=np.eye(3), translation=np.zeros(3)) for _ in items]),
        forwardKinematics=lambda *_: None, updateFramePlacements=lambda *_: None,
        updateGeometryPlacements=lambda *_: None)
    trimesh = SimpleNamespace(Trimesh=lambda vertices, faces, process: SimpleNamespace(vertices=vertices, faces=faces))

    class Intersector:
        def __init__(self, mesh):
            self.mesh = mesh

        def intersects_location(self, origins, directions, multiple_hits):
            depth = self.mesh.vertices[:, 2].min()
            locations = directions * (depth / directions[:, 2])[:, None]
            return locations, np.arange(len(directions)), np.zeros(len(directions), dtype=int)

    renderer._runtime = (pin, trimesh, Intersector, model, visual, (1, 2, 3))
    return renderer



def test_static_mesh_buffers_are_read_once_but_camera_pose_is_live(tmp_path):
    renderer = _mesh_cache_renderer(tmp_path)
    items = renderer._runtime[4].geometryObjects
    reads = {index: {"vertices": 0, "faces": 0} for index in range(len(items))}
    for index, item in enumerate(items):
        old_vertices, old_faces = item.geometry.vertices, item.geometry.tri_indices

        def vertices(index=index, original=old_vertices):
            reads[index]["vertices"] += 1
            return original()

        def faces(face, index=index, original=old_faces):
            reads[index]["faces"] += 1
            return original(face)

        item.geometry.vertices = vertices
        item.geometry.tri_indices = faces
    inputs = {
        "depth_m": np.ones((1, 1)),
        "intrinsics": {"fx": 10., "fy": 10., "cx": 0., "cy": 0.},
        "source_from_camera": np.eye(4),
        "capture_qpos_by_side": {s: np.zeros(6) for s in ("left", "right")},
    }
    first, evidence = renderer._render_depth_maps(**inputs)
    again, other = renderer._render_depth_maps(**inputs)
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    assert evidence == other
    assert reads[0] == {"vertices": 0, "faces": 0}  # Only selected meshes are decoded.
    assert all(reads[index] == {"vertices": 1, "faces": 1} for index in (1, 2, 3))

    moved = np.eye(4)
    moved[2, 3] = .1
    shifted, _ = renderer._render_depth_maps(**{**inputs, "source_from_camera": moved})
    np.testing.assert_allclose(shifted[0], first[0]-.1)
    np.testing.assert_allclose(shifted[1], first[1]-.1)
    # Another render reuses only mesh buffers, not camera poses or masks.
    renderer._render_depth_maps(**inputs)
    assert reads[0] == {"vertices": 0, "faces": 0}
    assert all(reads[index] == {"vertices": 1, "faces": 1} for index in (1, 2, 3))



def test_static_mesh_cache_owns_readonly_arrays(tmp_path):
    renderer = _mesh_cache_renderer(tmp_path)
    shape = renderer._runtime[4].geometryObjects[3].geometry
    source_vertices = shape.vertices()
    vertices, faces = renderer._mesh_buffers_for(3, shape)
    expected = vertices.copy()
    source_vertices[:] = 10
    np.testing.assert_array_equal(vertices, expected)
    with pytest.raises(ValueError):
        vertices[0, 0] = 1
    with pytest.raises(ValueError):
        faces[0, 0] = 1
    second = _mesh_cache_renderer(tmp_path)
    other_vertices, other_faces = second._mesh_buffers_for(3, second._runtime[4].geometryObjects[3].geometry)
    assert other_vertices is not vertices and other_faces is not faces



def test_static_mesh_cache_keeps_both_arm_fk_and_finger_states_live(tmp_path):
    renderer = _mesh_cache_renderer(tmp_path)
    pin, _, _, model, _, _ = renderer._runtime
    model.createData = lambda: SimpleNamespace(qpos=None)

    def forward_kinematics(_model, data, qpos):
        data.qpos = qpos.copy()

    def update_geometry(_model, data, _visual, placements, _qpos):
        placements.oMg[1].translation[2] = data.qpos[12]
        placements.oMg[2].translation[2] = data.qpos[0]
        placements.oMg[3].translation[2] = data.qpos[0] + .01 * data.qpos[6]

    pin.forwardKinematics = forward_kinematics
    pin.updateGeometryPlacements = update_geometry
    inputs = {
        "depth_m": np.ones((1, 1)),
        "intrinsics": {"fx": 10., "fy": 10., "cx": 0., "cy": 0.},
        "source_from_camera": np.eye(4),
    }
    first, _ = renderer._render_depth_maps(**inputs,
        capture_qpos_by_side={s: np.zeros(6) for s in ("left", "right")})
    qpos = {"left": np.array([.03, 0., 0., 0., 0., 0.]),
            "right": np.array([.02, 0., 0., 0., 0., 0.])}
    changed, _ = renderer._render_depth_maps(**inputs, capture_qpos_by_side=qpos)
    assert first[0][0, 0] == pytest.approx(.25)
    assert changed[0][0, 0] == pytest.approx(.27)
    for index, finger_q in enumerate(renderer.config.finger_sweep_q_rad[1:], start=1):
        assert first[index][0, 0] == pytest.approx(.45 + .01 * finger_q)
        assert changed[index][0, 0] == pytest.approx(.48 + .01 * finger_q)


def test_observation_topology_comes_from_complete_urdf_mesh_not_native_indices(tmp_path):
    import trimesh

    renderer = _mesh_cache_renderer(tmp_path)
    resource = tmp_path / "complete.obj"
    resource.write_text("v 0 0 0.8\nv 1 0 0.8\nv 0 1 0.8\nf 1 2 3\n")
    pin, _, intersector, model, visual, selected = renderer._runtime
    item = visual.geometryObjects[3]
    item.meshPath, item.meshScale = str(resource), np.array([2., 3., 1.])
    # This native shape's indexed surface is deliberately different; equal
    # resource/model labels cannot justify using it as the observed CAD.
    renderer._runtime = (pin, trimesh, intersector, model, visual, selected)
    vertices, faces = renderer._mesh_buffers_for(3, item.geometry)
    np.testing.assert_array_equal(vertices, [[0., 0., .8], [2., 0., .8], [0., 3., .8]])
    np.testing.assert_array_equal(faces, [[0, 1, 2]])
    assert renderer._mesh_buffers_for(3, item.geometry)[0] is vertices
    resource.unlink()
    assert renderer._mesh_buffers_for(3, item.geometry)[0] is vertices
    with pytest.raises(ValueError):
        vertices[0] = 0.


def test_static_preparation_needs_no_pose_or_camera_cut(tmp_path):
    renderer = _mesh_cache_renderer(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Static mesh preparation must not read FK or camera state")
    renderer._runtime[0].forwardKinematics = forbidden
    renderer._runtime[0].updateGeometryPlacements = forbidden
    renderer.prepare_geometry()
    buffers = dict(renderer._mesh_buffers)
    renderer.prepare_geometry()
    assert set(buffers) == {1, 2, 3}
    assert all(renderer._mesh_buffers[key] is value for key, value in buffers.items())


@pytest.mark.integration
def test_complete_body_surface_query_uses_same_topology_and_original_tolerance(tmp_path):
    import trimesh
    from robot_motion.rm65.captured_robot_observations import _native_body_surface_membership

    renderer = _mesh_cache_renderer(tmp_path)
    resource = tmp_path / "complete.obj"
    resource.write_text("v 0 0 0.8\nv 1 0 0.8\nv 0 1 0.8\nf 1 2 3\n")
    pin, _, intersector, model, visual, selected = renderer._runtime
    item = visual.geometryObjects[3]
    item.name, item.meshPath, item.meshScale = "left_arm_link2_0", str(resource), np.ones(3)
    renderer._runtime = (pin, trimesh, intersector, model, visual, selected)
    geometry, lower, upper = renderer._body_surface_references()[item.name]
    points = np.array([[.1, .1, .8], [.1, .1, .82]])
    matched, _ = _native_body_surface_membership(points, geometry, lower, upper, .015)
    np.testing.assert_array_equal(matched, [True, False])
    assert renderer._body_surface_references()[item.name][0] is geometry


def test_sample_cell_keys_keep_exact_original_first_samples():
    rng = np.random.default_rng(942)
    vertices = rng.uniform(-.1, .1, (4000, 3))
    vertices = np.r_[vertices, vertices[:1000], vertices[:1000]+1e-8]
    _, first = np.unique(np.floor(vertices/.001).astype(np.int64), axis=0, return_index=True)
    fake_trimesh = SimpleNamespace(remesh=SimpleNamespace(
        subdivide_to_size=lambda *_args, **_kw: (vertices, None)))
    np.testing.assert_array_equal(_surface_samples(fake_trimesh, None, None),
                                  vertices[np.sort(first)])


@pytest.mark.parametrize('limit', [1., 1.05, 1.1])
def test_symmetric_surface_nearest_metric_matches_exhaustive_ellipsoid(limit):
    rng = np.random.default_rng(486)
    surface = rng.uniform(-.1, .1, (350, 3))
    bounds = np.array([[-.045, .045], [-.05, .05], [-.04, .04]])
    points = np.vstack((rng.uniform(-.18, .18, (600, 3)),
                       surface[0] + np.array([bounds[0, 1]*np.sqrt(limit), 0, 0])))
    delta = (points[:, None] - surface) / bounds[:, 1]
    expected = (np.sum(delta**2, axis=2) <= limit + 1e-12).any(axis=1)
    actual = _asymmetric_surface_membership(points, surface, bounds,
                                           normalized_squared_limit=limit)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('limit', [1., 1.05, 1.1])
def test_symmetric_cached_tree_matches_single_surface_signed_boundaries(limit):
    surface = np.array([[.03, -.02, .01]])
    bounds = np.array([[-.045, .045], [-.05, .05], [-.04, .04]])
    points = np.array([surface[0] + sign*np.eye(3)[axis]*bounds[axis, 1]*factor
        for axis in range(3) for sign in (-1, 1)
        for factor in (np.sqrt(limit)-1e-9, np.sqrt(limit), np.sqrt(limit)+1e-9)])
    delta = (points-surface[0])/bounds[:, 1]
    expected = np.sum(delta**2, axis=1) <= limit+1e-12
    assert expected.reshape(-1, 3).tolist() == [[True, True, False]]*6
    tree = cKDTree(surface/bounds[:, 1])
    np.testing.assert_array_equal(_asymmetric_surface_membership(points, surface, bounds,
        tree=tree, normalized_squared_limit=limit), expected)


@pytest.mark.parametrize("mode", ["legacy", "cad_surface"])
def test_static_query_prepare_respects_disabled_tool_surface(mode):
    renderer = object.__new__(RM65CapturedRobotObservationFilter)
    renderer.config = RM65CapturedRobotObservationFilterConfig(geometry_mode=mode)
    renderer._lock = threading.Lock()
    renderer._flange_from_tool = {side: np.eye(4) for side in ("left", "right")}
    calls = []
    def runtime():
        calls.append("model")
        return (None, None, None, None, SimpleNamespace(geometryObjects=[]), ())
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled tool bounds must not prepare extended tool queries")
    renderer._runtime_components = runtime
    renderer._tool_surface_references = forbidden
    renderer._tool_query_references = forbidden
    phases = renderer.prepare()
    assert calls == ["model"]
    assert set(phases) == {"visual_model", "local_queries"}
    assert renderer.config.depth_tolerance_m == .015
    assert renderer.config.tool_surface_bounds_m is None


def test_warmed_ray_queries_keep_complete_scaled_mesh_snapshot(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    ray = pytest.importorskip("trimesh.ray.ray_pyembree").RayMeshIntersector
    renderer = _mesh_cache_renderer(tmp_path)
    resource = tmp_path / "complete-ray.obj"
    resource.write_text("v 0 0 0.8\nv 1 0 0.8\nv 0 1 0.8\nf 1 2 3\n")
    pin, _, _, model, visual, selected = renderer._runtime
    item = visual.geometryObjects[3]
    item.meshPath, item.meshScale = str(resource), np.array([2., 3., 1.])
    renderer._runtime = (pin, trimesh, ray, model, visual, selected)
    renderer._ray_meshes = {}
    query, _, _ = renderer._ray_mesh(3, item.geometry)
    resource.unlink()
    assert renderer._ray_mesh(3, item.geometry)[0] is query
    hits, indices, _ = query.intersects_location(
        np.array([[.4, .6, 0.], [1.8, 2.7, 0.]]),
        np.array([[0., 0., 1.], [0., 0., 1.]]), multiple_hits=False)
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_allclose(hits, [[.4, .6, .8]], rtol=0, atol=1e-6)
