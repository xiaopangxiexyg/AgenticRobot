import numpy as np
import pytest
import threading
from types import SimpleNamespace

from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilter,
    _asymmetric_surface_membership,
    _surface_samples,
    _camera_mesh_contains_optical_origin,
    _is_arm_body_surface,
    _native_body_surface_membership,
    _self_proxy_membership,
    RM65CapturedRobotObservationFilterConfig,
    captured_robot_observation_mask,
)


def _extended_config(**kwargs):
    values = dict(geometry_mode="cad_surface", depth_tolerance_m=.025,
        tool_surface_bounds_m=((-0.045, .045), (-.050, .050), (-.050, .050)),
        body_surface_tolerance_m=.050)
    values.update(kwargs)
    return RM65CapturedRobotObservationFilterConfig(**values)


def _filter_without_runtime():
    filt = object.__new__(RM65CapturedRobotObservationFilter)
    filt._mesh_buffers = {}
    filt._input_binding_hash = "test:extended-inputs"
    filt._tool_queries = {}
    # Primitive-only fixtures have no external URDF mesh resource.
    filt._runtime = (None, None, None, None,
        SimpleNamespace(geometryObjects=[SimpleNamespace(meshPath="")]), (0,))
    return filt


def test_local_mesh_arrays_load_once_are_owned_and_do_not_cross_instances():
    calls = []
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    shape = SimpleNamespace(vertices=lambda: vertices, num_tris=1,
        tri_indices=lambda index: calls.append(index) or (0, 1, 2))
    filt = _filter_without_runtime()
    a = filt._geometry_mesh(0, shape)
    assert filt._geometry_mesh(0, shape) is a
    assert calls == [0]
    vertices[:] = 5.
    np.testing.assert_array_equal(a[0][0], [0., 0., 0.])
    assert not a[0].flags.writeable and not a[1].flags.writeable
    other = _filter_without_runtime()._geometry_mesh(0, shape)
    assert calls == [0, 0]
    np.testing.assert_array_equal(other[0][0], [5., 5., 5.])


def test_cached_mesh_supports_native_fcl_triangle_indices():
    fcl = pytest.importorskip('hppfcl')
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    mesh = SimpleNamespace(vertices=lambda: vertices, num_tris=1,
        tri_indices=lambda index: fcl.Triangle(0, 1, 2))
    filt = _filter_without_runtime()
    actual, triangles = filt._geometry_mesh(0, mesh)
    np.testing.assert_array_equal(actual, vertices)
    np.testing.assert_array_equal(triangles, [[0, 1, 2]])


def test_cached_surface_tree_preserves_exact_membership_and_camera_exclusion():
    filt = _filter_without_runtime()
    filt.config = _extended_config()
    rng = np.random.default_rng(27)
    geometries = {'left': {'hand': rng.uniform(-.03, .03, (50, 3)),
                           'camera': np.array([[.3, 0, 0]])}}
    query = rng.uniform(-.12, .12, (80, 3))
    for excluded in (set(), {'camera'}, set()):
        surface, tree = filt._tool_query_references(geometries, excluded)['left']
        assert filt._tool_query_references(geometries, excluded)['left'][1] is tree
        expected = _asymmetric_surface_membership(query, surface,
            np.asarray(filt.config.tool_surface_bounds_m), fast_neighbors=1,
            normalized_squared_limit=filt.config.tool_surface_normalized_squared_limit)
        actual = _asymmetric_surface_membership(query, surface,
            np.asarray(filt.config.tool_surface_bounds_m), fast_neighbors=1,
            normalized_squared_limit=filt.config.tool_surface_normalized_squared_limit, tree=tree)
        np.testing.assert_array_equal(actual, expected)
    assert len(filt._tool_queries) == 2


@pytest.mark.parametrize("value", [-.001, .050001, np.nan, np.inf, True])
def test_body_surface_tolerance_is_bounded(value):
    with pytest.raises(ValueError, match="body_surface_tolerance_m"):
        _extended_config(body_surface_tolerance_m=value)


@pytest.mark.parametrize("side", ["left", "right"])
def test_body_surface_rule_excludes_tools_and_optional_accessories(side):
    assert _is_arm_body_surface(side + "_arm_link4_0")
    assert not _is_arm_body_surface(side + "_arm_link6_0")
    assert not _is_arm_body_surface(side + "_self_observation_cable_0")
    assert not _is_arm_body_surface(side + "_realsense_d405_0")


@pytest.mark.parametrize("side, angle", [("left", 0.), ("right", .7), ("left", -1.2)])
@pytest.mark.parametrize("tolerance", [0., .020, .035, .050])
def test_full_body_surface_matches_clipped_cad_in_capture_geometry_frame(monkeypatch, side, angle, tolerance):
    filt = _filter_without_runtime()
    filt._self_proxy_boxes = []
    filt.config = _extended_config(body_surface_tolerance_m=tolerance)
    filt.urdf_sha256 = "fixture"
    filt._lock = threading.Lock()
    filt._flange_from_tool = None
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = [.2, -.3, .4]
    name = side + "_arm_link4_0"
    # CAD surface projects outside the image. A first-hit depth map cannot
    # cover it, but the full native surface is only 15 mm from the observation.
    camera_surface = np.array([[-.015, 0., 1.]])
    local_surface = camera_surface @ rotation.T + pose[:3, 3]
    monkeypatch.setattr(filt, "_body_surface_references", lambda: {name: (None, None, None)})
    def membership(points, geometry, lower, upper, bound):
        return np.linalg.norm(points-local_surface[0], axis=1) <= bound, len(points)
    monkeypatch.setattr('robot_motion.rm65.captured_robot_observations._native_body_surface_membership', membership)
    evidence = {"rendered_depth_map_count": 1, "rendered_geometry_count_by_state": (1,),
        "rendered_triangle_count_by_state": (1,), "excluded_capture_origin_camera_meshes": [],
        "tool_from_camera_by_side": {}, "body_geometry_from_camera": {name: pose.tolist()}}
    monkeypatch.setattr(filt, "_render_depth_maps", lambda **kwargs: ((np.full((1, 4), np.nan),), evidence))
    result = filt(depth_m=np.array([[1., 1., 1.07, 1.1]]),
        target_mask=np.array([[False, True, False, False]]),
        intrinsics={"fx": 1e6, "fy": 1e6, "cx": 0., "cy": 0.}, source_from_camera=np.eye(4),
        capture_qpos_by_side={side: np.zeros(6) for side in ("left", "right")})
    np.testing.assert_array_equal(result.removed_mask, [[bool(tolerance), False, False, False]])
    assert result.evidence["body_surface_additional_removed_points"] == int(bool(tolerance))
    assert result.evidence["target_depth_points_before"] == result.evidence["target_depth_points_after"] == 1


def test_native_body_distance_uses_surface_not_expanded_aabb():
    fcl = pytest.importorskip("hppfcl")
    geometry = fcl.Box(.1, .1, .1)
    points = np.array([[.07 - 1e-8, 0, 0], [.07 + 1e-8, 0, 0],
                       [.065, .065, 0], [.06, .06, 0], [.1, .1, 0]])
    matched, queries = _native_body_surface_membership(points, geometry,
        np.full(3, -.05), np.full(3, .05), .020)
    np.testing.assert_array_equal(matched, [True, False, False, True, False])
    assert queries == 3


def test_arm_body_margin_covers_confirmed_shell_returns_but_keeps_nearby_background():
    fcl = pytest.importorskip("hppfcl")
    geometry = fcl.Box(.1, .1, .1)
    points = np.array([[.0824, 0, 0], [.0947, 0, 0], [.1-1e-8, 0, 0],
                       [.1+1e-8, 0, 0], [.09, .09, 0], [.11, 0, 0]])
    lower, upper = np.full(3, -.05), np.full(3, .05)
    old, _ = _native_body_surface_membership(points, geometry, lower, upper, .020)
    new, _ = _native_body_surface_membership(points, geometry, lower, upper,
        _extended_config().body_surface_tolerance_m)
    np.testing.assert_array_equal(old, [False]*6)
    # A corner inside the expanded AABB is still outside the 50 mm CAD shell.
    np.testing.assert_array_equal(new, [True, True, True, False, False, False])


def test_self_proxy_contains_rotated_volume_without_second_padding():
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    half = np.array([.02, .03, .04])
    points = np.array([[0, 0, 0], [.029, 0, 0], [.031, 0, 0],
        [0, .019, 0], [0, .021, 0], [.074, .069, 0]])
    actual = _self_proxy_membership(points, pose, half)
    np.testing.assert_array_equal(actual, [True, True, False, True, False, False])


@pytest.mark.parametrize('side', ['left', 'right'])
def test_capture_proxy_is_separate_from_cad_and_target_pixels_are_never_removed(monkeypatch, side):
    from robot_motion.rm65.scene_profiles import HandProxyBoxSpec
    filt = _filter_without_runtime()
    filt.config = _extended_config()
    filt.urdf_sha256 = 'fixture'
    filt._lock = threading.Lock()
    filt._flange_from_tool = {s: np.eye(4) for s in ('left', 'right')}
    box = HandProxyBoxSpec('self', side, side+'_hand_proxy_1',
        'arm_link6' if side=='left' else 'right_arm_link6', (.04, .04, .04),
        (0.,0.,0.), (0.,0.,0.), (0.,0.,1.), (0.,0.,0.), (1.,1.,1.,1.))
    pose = np.eye(4); pose[2,3] = 1.
    filt._self_proxy_boxes = [(box, pose, np.full(3, .02))]
    evidence = {'rendered_depth_map_count':1, 'rendered_geometry_count_by_state':(1,),
        'rendered_triangle_count_by_state':(1,), 'excluded_capture_origin_camera_meshes':[],
        'tool_from_camera_by_side':{s:np.eye(4).tolist() for s in ('left','right')}}
    monkeypatch.setattr(filt, '_render_depth_maps', lambda **kw: ((np.full((1,4),np.nan),),evidence))
    monkeypatch.setattr(filt, '_tool_surface_references', lambda: {
        s:{s+'_arm_link6_0':np.array([[1.,1.,1.]])} for s in ('left','right')})
    result = filt(depth_m=np.array([[1.,1.,1.01,1.12]]), target_mask=np.array([[False,True,False,False]]),
        intrinsics={'fx':1e6,'fy':1e6,'cx':0.,'cy':0.}, source_from_camera=np.eye(4),
        capture_qpos_by_side={s:np.zeros(6) for s in ('left','right')})
    np.testing.assert_array_equal(result.removed_mask, [[True,False,True,False]])
    assert result.evidence['self_proxy_additional_removed_points_by_geometry'][box.geometry_name] == 2
    assert result.evidence['method'] == 'urdf_cad_and_capture_self_proxy_v6'
    assert result.evidence['target_depth_points_before'] == result.evidence['target_depth_points_after'] == 1


def test_local_surface_signed_bounds_are_ellipsoidal_not_box_clearing():
    bounds = np.asarray(_extended_config().tool_surface_bounds_m)
    query = np.array([[-.045, 0, 0], [.045, 0, 0], [-.0451, 0, 0], [.0451, 0, 0],
        [0, .05, 0], [0, -.05, 0], [0, 0, .05], [0, 0, -.05], [.045, .05, .05]])
    actual = _asymmetric_surface_membership(query, np.zeros((1, 3)), bounds)
    np.testing.assert_array_equal(actual, [True, True, False, False, True, True, True, True, False])
    assert not _asymmetric_surface_membership(query, np.zeros((0, 3)), bounds).any()


def test_local_surface_exact_fallback_and_translation_invariance():
    bounds = np.array([[-.04, .03], [-.05, .05], [-.04, .04]])
    assert _asymmetric_surface_membership(np.zeros((1, 3)),
        np.array([[-.031, 0, 0], [.039, 0, 0]]), bounds, fast_neighbors=1)[0]
    rng = np.random.default_rng(17)
    surface, query = rng.uniform(-.07, .07, (45, 3)), rng.uniform(-.15, .15, (60, 3))
    delta = query[:, None] - surface
    radii = np.where(delta >= 0, bounds[:, 1], -bounds[:, 0])
    brute = np.any(np.sum((delta / radii) ** 2, axis=2) <= 1 + 1e-12, axis=1)
    for shift in (np.zeros(3), np.array([2, -3, 4])):
        actual = _asymmetric_surface_membership(query + shift, surface + shift, bounds, fast_neighbors=1)
        np.testing.assert_array_equal(actual, brute)


@pytest.mark.parametrize("limit", [1.0, 1.05, 1.10])
def test_surface_norm_slack_matches_brute_force_and_searches_expanded_sphere(limit):
    bounds = np.asarray(_extended_config().tool_surface_bounds_m)
    surface = np.zeros((1, 3))
    # Y-axis samples outside the original query sphere must remain discoverable.
    query = np.array([[0, .05 * np.sqrt(1.033), 0],
                      [0, .05 * np.sqrt(limit) - 1e-7, 0],
                      [0, .05 * np.sqrt(limit) + 1e-7, 0]])
    actual = _asymmetric_surface_membership(query, surface, bounds, normalized_squared_limit=limit)
    np.testing.assert_array_equal(actual, [limit >= 1.033, True, False])
    rng = np.random.default_rng(39)
    surface, query = rng.uniform(-.07, .07, (45, 3)), rng.uniform(-.12, .12, (60, 3))
    delta = query[:, None] - surface
    radii = np.where(delta >= 0, bounds[:, 1], -bounds[:, 0])
    brute = np.any(np.sum((delta / radii) ** 2, axis=2) <= limit + 1e-12, axis=1)
    actual = _asymmetric_surface_membership(query, surface, bounds,
        fast_neighbors=1, normalized_squared_limit=limit)
    np.testing.assert_array_equal(actual, brute)


@pytest.mark.parametrize("limit", [.99, 1.10001, np.nan, np.inf, True])
def test_surface_norm_slack_config_is_bounded(limit):
    with pytest.raises(ValueError, match="tool_surface_normalized_squared_limit"):
        _extended_config(tool_surface_normalized_squared_limit=limit)


def test_default_surface_norm_slack_is_five_percent_of_squared_norm():
    assert _extended_config().tool_surface_normalized_squared_limit == 1.05


def test_local_membership_uses_rotated_tool_axes_not_camera_axes():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    camera = np.array([[0, -.047, 0], [.047, 0, 0]])
    bounds = np.asarray(_extended_config().tool_surface_bounds_m)
    np.testing.assert_array_equal(_asymmetric_surface_membership(
        camera @ rotation.T, np.zeros((1, 3)), bounds), [False, True])


@pytest.mark.parametrize("bounds", [((0, .03), (-.05, .05), (-.04, .04)),
    ((-.04, .03), (-.061, .05), (-.04, .04)), ((-.04, .03),)])
def test_invalid_tool_surface_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="tool_surface_bounds_m"):
        _extended_config(tool_surface_bounds_m=bounds)


@pytest.mark.parametrize("limit, removed", [(1.0, False), (1.05, True)])
def test_full_tool_reference_handles_depth_hole_and_offscreen_surface_and_protects_target(monkeypatch, limit, removed):
    # The reference projects outside the image and has no captured depth.
    # Small norm slack is wired through the filter; target labels still win.
    filt = _filter_without_runtime()
    filt._self_proxy_boxes = []
    filt.config = _extended_config(tool_surface_normalized_squared_limit=limit)
    filt.urdf_sha256 = "fixture"
    filt._lock = threading.Lock()
    filt._flange_from_tool = {side: np.eye(4) for side in ("left", "right")}
    pose = np.eye(4)
    # Camera -Y maps to tool +X, which has the tighter 45 mm radius.
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    maps = (np.full((1, 2), np.nan),)
    evidence = {"rendered_depth_map_count": 1, "rendered_geometry_count_by_state": (1,),
        "rendered_triangle_count_by_state": (1,), "excluded_capture_origin_camera_meshes": [],
        "tool_from_camera_by_side": {side: pose.tolist() for side in ("left", "right")}}
    monkeypatch.setattr(filt, "_render_depth_maps", lambda **kwargs: (maps, evidence))
    surface = np.array([[0, -.05 * np.sqrt(1.033), 1.]])
    monkeypatch.setattr(filt, "_tool_surface_references", lambda: {
        side: {side + "_arm_link6_0": surface} for side in ("left", "right")})
    depth = np.ones((1, 2))
    target = np.array([[False, True]])
    result = filt(depth_m=depth, target_mask=target,
        intrinsics={"fx": 1e6, "fy": 1e6, "cx": 0., "cy": 0.},
        source_from_camera=np.eye(4), capture_qpos_by_side={side: np.zeros(6) for side in ("left", "right")})
    np.testing.assert_array_equal(result.removed_mask, [[removed, False]])
    assert result.depth_m[0, 1] == 1
    assert result.evidence["removed_target_points"] == 0
    assert result.evidence["tool_surface_normalized_squared_limit"] == limit


def test_tool_surface_expansion_is_bounded_and_keeps_corner_rejection():
    config = _extended_config()
    old = np.array([[-.04, .03], [-.05, .05], [-.04, .04]])
    new = np.asarray(config.tool_surface_bounds_m)
    np.testing.assert_array_equal(new, [[-.045, .045], [-.05, .05], [-.05, .05]])
    assert config.tool_surface_normalized_squared_limit == 1.05
    assert config.depth_tolerance_m == .025 and config.body_surface_tolerance_m == .050
    # Representative geometric deltas, not an image/pose-specific ignore list.
    points = np.array([[.02753, -.02283, .00035], [.02454, -.03199, .00147],
                       [.03136, -.03118, -.01895], [.05, 0, 0], [-.05, 0, 0],
                       [0, .055, 0], [0, 0, .055], [.045, .05, .05]])
    for bounds, expected in ((old, [False] * 8),
                             (new, [True, True, True, False, False, False, False, False])):
        np.testing.assert_array_equal(_asymmetric_surface_membership(points,
            np.zeros((1, 3)), bounds, normalized_squared_limit=1.05), expected)


def test_surface_reference_samples_stay_on_original_triangles_and_are_deterministic():
    import trimesh
    vertices = np.array([[0, 0, 0], [.02, 0, 0], [0, .02, 0]])
    faces = np.array([[0, 1, 2]])
    points = _surface_samples(trimesh, vertices, faces)
    assert len(points) > 50
    assert np.all(points[:, 2] == 0)
    assert np.all(points[:, :2] >= 0)
    assert np.all(points[:, 0] + points[:, 1] <= .02 + 1e-12)
    np.testing.assert_array_equal(points, _surface_samples(trimesh, vertices, faces))


@pytest.mark.parametrize('calibrated', [False, True])
def test_cad_rays_cover_zero_depth_pixels(monkeypatch, calibrated):
    filt = _filter_without_runtime()
    filt._self_proxy_boxes = []
    filt.config = _extended_config(tool_surface_bounds_m=None)
    filt._flange_from_tool = {s: np.eye(4) for s in ('left', 'right')} if calibrated else None
    shape = SimpleNamespace(vertices=lambda: np.array([[-1, -1, 1], [1, -1, 1], [0, 1, 1]]),
        num_tris=1, tri_indices=lambda index: (0, 1, 2))
    visual = SimpleNamespace(geometryObjects=[SimpleNamespace(name='left_eg2_4b2_cad_l_1_0', geometry=shape)])
    pose = SimpleNamespace(rotation=np.eye(3), translation=np.zeros(3))
    pin = SimpleNamespace(GeometryData=lambda model: SimpleNamespace(oMg=[pose]),
        forwardKinematics=lambda *args: None, updateFramePlacements=lambda *args: None,
        updateGeometryPlacements=lambda *args: None)
    ray_counts = []
    class Intersector:
        def __init__(self, mesh):
            pass
        def intersects_location(self, origins, directions, multiple_hits):
            ray_counts.append(len(directions))
            return np.array([[0., 0., 1.]]), np.array([0]), np.array([0])
    fake_trimesh = SimpleNamespace(Trimesh=lambda **kwargs: None)
    monkeypatch.setattr(filt, '_runtime_components', lambda: (
        pin, fake_trimesh, Intersector, SimpleNamespace(
            createData=lambda: SimpleNamespace(oMf=[SimpleNamespace(homogeneous=np.eye(4))]),
            getFrameId=lambda name: 0, frames=[None]), visual, (0,)))
    depth = np.array([[0., 1., 0.], [0., 0., 0.]])
    maps, _ = filt._render_depth_maps(depth_m=depth,
        intrinsics={'fx': 1., 'fy': 1., 'cx': 0., 'cy': 0.}, source_from_camera=np.eye(4),
        capture_qpos_by_side={side: np.zeros(6) for side in ('left', 'right')})
    assert ray_counts == [6] * 5
    assert maps[0][0, 0] == 1
    assert depth[0, 0] == 0


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("moving", [False, True])
def test_explicit_observation_attachment_has_independent_depth_layer(monkeypatch, calibrated, moving):
    filt = _filter_without_runtime()
    filt._self_proxy_boxes = []
    filt.config = _extended_config(tool_surface_bounds_m=(
        ((-.045, .045), (-.05, .05), (-.05, .05)) if calibrated else None))
    filt._flange_from_tool = ({side: np.eye(4) for side in ('left', 'right')} if calibrated else None)
    def shape(z):
        return SimpleNamespace(vertices=lambda: np.array([[-1, -1, z], [1, -1, z], [0, 1, z]]),
            num_tris=1, tri_indices=lambda index: (0, 1, 2))
    visual = SimpleNamespace(geometryObjects=[
        SimpleNamespace(name='right_eg2_4b2_cad_body_0', geometry=shape(.408)),
        SimpleNamespace(name='right_self_observation_wrist_cable_0', geometry=shape(.986))])
    pose = SimpleNamespace(rotation=np.eye(3), translation=np.zeros(3))
    def update_placements(model, data, visual, visual_data, q):
        visual_data.oMg[1] = SimpleNamespace(rotation=np.eye(3),
            translation=np.array([0., 0., float(q[18]) * .1 if moving else 0.]))
    pin = SimpleNamespace(GeometryData=lambda model: SimpleNamespace(oMg=[pose, pose]),
        forwardKinematics=lambda *args: None, updateFramePlacements=lambda *args: None,
        updateGeometryPlacements=update_placements)
    class Intersector:
        def __init__(self, mesh):
            self.z = np.min(mesh.vertices[:, 2])
        def intersects_location(self, origins, directions, multiple_hits):
            return np.array([[0., 0., self.z]]), np.array([0]), np.array([0])
    fake_trimesh = SimpleNamespace(Trimesh=lambda **kwargs: SimpleNamespace(**kwargs))
    model = SimpleNamespace(createData=lambda: SimpleNamespace(oMf=[SimpleNamespace(homogeneous=np.eye(4))]),
        getFrameId=lambda name: 0, frames=[None])
    monkeypatch.setattr(filt, '_runtime_components', lambda: (
        pin, fake_trimesh, Intersector, model, visual, (0, 1)))
    maps, evidence = filt._render_depth_maps(depth_m=np.ones((1, 1)),
        intrinsics={'fx': 1., 'fy': 1., 'cx': 0., 'cy': 0.}, source_from_camera=np.eye(4),
        capture_qpos_by_side={side: np.zeros(6) for side in ('left', 'right')})
    extra_count = 5 if moving else 1
    assert evidence['rendered_depth_map_count'] == (1 if calibrated else 5) + extra_count
    assert evidence['self_observation_attachment_layer_count'] == extra_count
    assert evidence['self_observation_attachment_geometry_count'] == 1
    if calibrated:
        assert np.isnan(maps[0][0, 0])
    else:
        assert maps[0][0, 0] == .408
    assert maps[1][0, 0] == .986
    if moving:
        assert maps[-1][0, 0] == pytest.approx(.986 + .0864)


@pytest.mark.parametrize('tools', [{'left': np.zeros(6)},
    {'left': [0] * 5, 'right': np.zeros(6)}, {'left': [np.nan] * 6, 'right': np.zeros(6)}])
def test_filter_rejects_incomplete_or_invalid_tcp_calibration(tmp_path, tools):
    path = tmp_path / 'robot.urdf'
    path.write_text('<robot/>')
    with pytest.raises(ValueError, match='tool TCP calibration'):
        RM65CapturedRobotObservationFilter(urdf_path=path, tool_tcp_pose_flange_m_rad_by_side=tools)


def test_only_capture_origin_camera_housing_is_excluded_from_self_rendering():
    vertices = np.array([[-.03, -.02, -.01], [.03, .02, .01]])
    assert _camera_mesh_contains_optical_origin("right_realsense_d405_0", vertices)
    assert not _camera_mesh_contains_optical_origin("left_realsense_d405_0", vertices + [.2, 0, 0])
    assert not _camera_mesh_contains_optical_origin("right_gripper_cad_body", vertices)


def test_finite_neighbor_fast_path_cannot_hide_a_valid_depth_consistent_pixel():
    depth = np.zeros((7, 7))
    depth[3, 3] = 1
    rendered = np.full((7, 7), np.nan)
    rendered[3, 3] = 1.016  # Closest normalized point, outside the 15 mm depth bound.
    rendered[3, 5] = 1.011  # Within both original bounds but not in k=1.
    removed = captured_robot_observation_mask(depth_m=depth, target_mask=np.zeros_like(depth, bool),
        rendered_depth_maps_m=(rendered,), silhouette_radius_px=2., depth_tolerance_m=.015,
        depth_consistent_neighbor_candidates=1, extended_policy=True)
    assert removed[3, 3]








@pytest.mark.parametrize("tolerance", (.015, .020, .025, .030))
def test_depth_tolerance_preserves_target_and_rejects_more_distant_depth(tolerance):
    config = _extended_config(depth_tolerance_m=tolerance)
    depth = np.ones((1, 5))
    target = np.array([[False, False, False, True, False]])
    rendered = np.array([[1 + tolerance - .001, 1 + tolerance + .001,
        1 - tolerance + .001, 1 - tolerance + .001, np.nan]])
    removed = captured_robot_observation_mask(depth_m=depth, target_mask=target,
        rendered_depth_maps_m=(rendered,), silhouette_radius_px=0,
        depth_tolerance_m=config.depth_tolerance_m)
    np.testing.assert_array_equal(removed, [[True, False, True, False, False]])


def test_explicit_extended_depth_tolerance_is_twenty_five_mm():
    config = _extended_config()
    assert config.depth_tolerance_m == .025
    assert config.silhouette_radius_px == 12.0
