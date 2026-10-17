import numpy as np
import pytest
from types import SimpleNamespace

from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilter,
    RM65CapturedRobotObservationFilterConfig,
    RM65CapturedRobotObservationFilterResult,
    captured_robot_observation_mask,
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
