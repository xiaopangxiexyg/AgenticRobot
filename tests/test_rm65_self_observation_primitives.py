import json

import numpy as np
import pytest
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilter,
    RM65CapturedRobotObservationFilterConfig,
    _projected_box_roi,
)
from robot_motion.rm65.scene_profiles import HandProxyBoxSpec


def _primitive_config(**kwargs):
    values = dict(geometry_mode="link_primitives", depth_tolerance_m=.025,
        tool_surface_bounds_m=((-0.045, .045), (-.050, .050), (-.050, .050)),
        body_surface_tolerance_m=.050)
    values.update(kwargs)
    return RM65CapturedRobotObservationFilterConfig(**values)


def _boxes():
    return (HandProxyBoxSpec('fixture', 'left', 'left_hand_proxy_1', 'arm_link6',
        (.04, .04, .04), (0., 0., 0.), (0., 0., 0.), (0., 0., 1.),
        (0., 0., 0.), (1., 1., 1., 1.)),)


def test_profile_selection_keeps_layered_cad_for_explicit_or_unknown_attachments(tmp_path):
    urdf = tmp_path / 'fixture.urdf'
    def link(name):
        return f'<link name="{name}"><visual><geometry><mesh filename="unused.stl"/></geometry></visual></link>'
    for accessory, expected in (('', 'link_primitives'), ('right_self_observation_wrist_cable', 'cad_surface'),
                                ('left_custom_camera_bracket', 'cad_surface')):
        urdf.write_text('<robot name="fixture">' + link('left_arm_link2')
                        + (link(accessory) if accessory else '') + '</robot>')
        assert RM65CapturedRobotObservationFilterConfig.for_urdf(
            urdf, has_self_envelope=True).geometry_mode == expected
        assert RM65CapturedRobotObservationFilterConfig.for_urdf(
            urdf, has_self_envelope=False).geometry_mode == 'cad_surface'


def test_projected_proxy_roi_contains_random_projections():
    rng = np.random.default_rng(237)
    intr = {'fx': 605., 'fy': 612., 'cx': 318., 'cy': 240.}
    for _ in range(50):
        pose = np.eye(4)
        angle = rng.uniform(-3., 3.)
        pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                       [-np.sin(angle), 0, np.cos(angle)]]
        pose[:3, 3] = rng.uniform([-.6, -.4, .5], [.6, .4, 2.])
        half = rng.uniform(.01, .2, 3)
        rows, columns = _projected_box_roi(pose, -half, half, intr, (480, 640))
        points = rng.uniform(-half, half, (200, 3)) @ pose[:3, :3].T + pose[:3, 3]
        uv = points[:, :2] / points[:, 2, None] * [605, 612] + [318, 240]
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)]
        assert np.all((uv[:, 0] >= columns.start) & (uv[:, 0] < columns.stop))
        assert np.all((uv[:, 1] >= rows.start) & (uv[:, 1] < rows.stop))


def test_frustum_rejects_offscreen_box_even_when_it_crosses_camera_plane():
    pose = np.eye(4); pose[0, 3] = 2.
    rows, columns = _projected_box_roi(pose, np.full(3, -.05), np.full(3, .05),
        {'fx': 600., 'fy': 600., 'cx': 320., 'cy': 240.}, (480, 640))
    assert rows.start == rows.stop and columns.start == columns.stop


@pytest.mark.parametrize('side', ['left', 'right'])
def test_primitive_path_uses_measured_joint_and_preserves_targets_without_rays(tmp_path, monkeypatch, side):
    urdf = tmp_path / 'fixture.urdf'
    urdf.write_text('<robot name="fixture"/>')
    filt = RM65CapturedRobotObservationFilter(urdf_path=urdf,
        config=_primitive_config(),
        self_observation_proxy_boxes=_boxes(),
        tool_tcp_pose_flange_m_rad_by_side={s: np.zeros(6) for s in ('left', 'right')})
    def geometry(arm, source_from_camera):
        pose = np.eye(4)
        pose[2, 3] = arm[0 if side == 'left' else 1][0]
        return [(side, pose, np.full(3, -.06), np.full(3, .06), 'contains', None)], set()
    monkeypatch.setattr(filt, '_capture_primitive_geometry', geometry)
    kwargs = {'depth_m': np.ones((1, 3)), 'target_mask': np.array([[False, True, False]]),
        'intrinsics': {'fx': 10., 'fy': 10., 'cx': 1., 'cy': 0.}, 'source_from_camera': np.eye(4)}
    qpos = {s: np.zeros(6) for s in ('left', 'right')}
    qpos[side][0] = 1.
    result = filt(**kwargs, capture_qpos_by_side=qpos)
    np.testing.assert_array_equal(result.removed_mask, [[False, False, False]])
    kwargs['target_mask'][:] = False
    result = filt(**kwargs, capture_qpos_by_side=qpos)
    np.testing.assert_array_equal(result.removed_mask, [[False, True, False]])
    qpos[side][0] = 1.2
    result = filt(**kwargs, capture_qpos_by_side=qpos)
    assert not result.removed_mask.any()
    assert result.evidence['rendered_depth_map_count'] == 0


def test_tool_broad_box_never_erases_empty_gap(tmp_path, monkeypatch):
    urdf = tmp_path / 'fixture.urdf'
    urdf.write_text('<robot name="fixture"/>')
    filt = RM65CapturedRobotObservationFilter(urdf_path=urdf,
        config=_primitive_config(),
        self_observation_proxy_boxes=_boxes(),
        tool_tcp_pose_flange_m_rad_by_side={s: np.zeros(6) for s in ('left', 'right')})
    surface = np.array([[-.15, 0, 1.], [.15, 0, 1.]])
    extent = np.max(np.abs(filt.config.tool_surface_bounds_m), axis=1)
    monkeypatch.setattr(filt, '_capture_primitive_geometry', lambda *args:
        ([('left', np.eye(4), surface.min(axis=0), surface.max(axis=0), 'tool',
           (surface, cKDTree(surface / extent)))], set()))
    result = filt(depth_m=np.ones((1, 5)), target_mask=np.zeros((1, 5), bool),
        intrinsics={'fx': 1/.15, 'fy': 10., 'cx': 2., 'cy': 0.}, source_from_camera=np.eye(4),
        capture_qpos_by_side={s: np.zeros(6) for s in ('left', 'right')})
    # The central point is inside the broad box, but 150 mm from either finger.
    assert not result.removed_mask[0, 2]
    assert result.removed_mask[0, 1] and result.removed_mask[0, 3]


def test_body_reference_uncertainty_uses_native_distance_without_corner_inflation(tmp_path, monkeypatch):
    fcl = pytest.importorskip('hppfcl')
    urdf = tmp_path / 'fixture.urdf'
    urdf.write_text('<robot name="fixture"/>')
    filt = RM65CapturedRobotObservationFilter(urdf_path=urdf,
        config=_primitive_config(
            body_surface_tolerance_m=.020),
        self_observation_proxy_boxes=_boxes(),
        tool_tcp_pose_flange_m_rad_by_side={s: np.zeros(6) for s in ('left', 'right')})
    pose = np.eye(4); pose[2, 3] = 1.
    vertices = np.array([[x, y, z] for x in (-.05, .05)
                         for y in (-.05, .05) for z in (-.05, .05)])
    monkeypatch.setattr(filt, '_capture_primitive_geometry', lambda *args:
        ([('left_arm_link2_0', pose, np.full(3, -.05), np.full(3, .05), 'body',
           (fcl.Box(.1, .1, .1), cKDTree(vertices), .18))], set()))
    result = filt(depth_m=np.full((1, 3), 1.065), target_mask=np.zeros((1, 3), bool),
        intrinsics={'fx': 1.065/.065, 'fy': 10., 'cx': 1., 'cy': 0.}, source_from_camera=np.eye(4),
        capture_qpos_by_side={s: np.zeros(6) for s in ('left', 'right')})
    np.testing.assert_array_equal(result.removed_mask, [[False, True, False]])
    assert result.evidence['native_boundary_query_counts_by_geometry']['left_arm_link2_0'] == 3


def test_base_roi_rays_include_zero_depth_pixel_and_keep_depth_mismatch(tmp_path, monkeypatch):
    urdf = tmp_path / 'fixture.urdf'; urdf.write_text('<robot name="fixture"/>')
    filt = RM65CapturedRobotObservationFilter(urdf_path=urdf,
        config=_primitive_config(),
        self_observation_proxy_boxes=_boxes(),
        tool_tcp_pose_flange_m_rad_by_side={s: np.zeros(6) for s in ('left', 'right')})
    def render(**kwargs):
        assert kwargs['depth_m'].shape == (3, 3)
        assert kwargs['depth_m'][1, 1] == 0
        rendered = np.full((3, 3), np.nan); rendered[1, 1] = 1.
        return (rendered,), {}
    monkeypatch.setattr(filt, '_render_primitive_base', render)
    monkeypatch.setattr(filt, '_capture_primitive_geometry', lambda *args:
        ([('left_arm_base_link_0', np.eye(4), np.array([-.01, -.01, .99]),
           np.array([.01, .01, 1.01]), 'base', None)], set()))
    depth = np.array([[1., 1., 1.1], [1., 0., 1.], [1., 1., 1.]])
    target = np.zeros((3, 3), bool); target[0, 0] = True
    result = filt(depth_m=depth, target_mask=target,
        intrinsics={'fx': 100., 'fy': 100., 'cx': 1., 'cy': 1.}, source_from_camera=np.eye(4),
        capture_qpos_by_side={s: np.zeros(6) for s in ('left', 'right')})
    np.testing.assert_array_equal(result.removed_mask,
        [[False, True, False], [True, False, True], [True, True, True]])


def test_local_ray_mesh_reuses_bvh_but_updates_link_pose_and_occlusion(tmp_path, monkeypatch):
    trimesh = pytest.importorskip('trimesh')
    ray = pytest.importorskip('trimesh.ray.ray_pyembree').RayMeshIntersector
    urdf = tmp_path / 'fixture.urdf'; urdf.write_text('<robot name="fixture"/>')
    filt = RM65CapturedRobotObservationFilter(urdf_path=urdf)
    vertices = np.array([[-.15, -.15, 0], [.15, -.15, 0], [.15, .15, 0], [-.15, .15, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    shape = SimpleNamespace(vertices=lambda: vertices, num_tris=2, tri_indices=lambda i: faces[i])
    visual = SimpleNamespace(geometryObjects=[
        SimpleNamespace(name='left_arm_base_link_0', geometry=shape),
        SimpleNamespace(name='right_arm_link6_0', geometry=shape)])
    poses = [np.eye(4), np.eye(4)]
    poses[0][2, 3], poses[1][2, 3] = 1., .8
    def update(_model, _data, geom, geom_data, q):
        poses[1][0, 3] = q[12]
    pin = SimpleNamespace(GeometryData=lambda _: SimpleNamespace(oMg=[
        SimpleNamespace(homogeneous=p) for p in poses]), forwardKinematics=lambda *_: None,
        updateGeometryPlacements=update)
    model = SimpleNamespace(createData=lambda: None)
    monkeypatch.setattr(filt, '_runtime_components', lambda: (pin, trimesh, ray, model, visual, (0, 1)))
    qpos = {s: np.zeros(6) for s in ('left', 'right')}
    kw = dict(depth_m=np.zeros((3, 3)), intrinsics={'fx': 10., 'fy': 10., 'cx': 1., 'cy': 1.},
        source_from_camera=np.eye(4), capture_qpos_by_side=qpos)
    blocked = filt._render_primitive_base(**kw)[0][0]
    assert np.isnan(blocked).all()  # Tool occludes the body; it is not a body hit.
    cached = dict(filt._ray_meshes)
    qpos['right'][0] = 2.
    visible = filt._render_primitive_base(**kw)[0][0]
    np.testing.assert_allclose(visible, 1.)
    assert all(filt._ray_meshes[i][0] is cached[i][0] for i in cached)
    qpos['right'][0] = 0.
    assert np.isnan(filt._render_primitive_base(**kw)[0][0]).all()
    # Compare independent combined-mesh queries under changing camera and
    # link rotations. Every zero-depth pixel must still cast a ray.
    rng = np.random.default_rng(128)
    for _ in range(12):
        poses[0][:3, :3] = Rotation.from_rotvec(rng.uniform(-.7, .7, 3)).as_matrix()
        poses[1][:3, :3] = Rotation.from_rotvec(rng.uniform(-.7, .7, 3)).as_matrix()
        qpos['right'][0] = rng.uniform(-.2, .2)
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_rotvec(rng.uniform(-.15, .15, 3)).as_matrix()
        camera[:3, 3] = rng.uniform(-.08, .08, 3)
        kw.update(depth_m=np.zeros((11, 13)), source_from_camera=camera,
                  intrinsics={'fx': 20., 'fy': 20., 'cx': 6., 'cy': 5.})
        actual = filt._render_primitive_base(**kw)[0][0]
        local = [np.linalg.inv(camera) @ p for p in poses]
        meshes = [vertices @ p[:3, :3].T + p[:3, 3] for p in local]
        combined = trimesh.Trimesh(vertices=np.concatenate(meshes),
            faces=np.r_[faces, faces+len(vertices)], process=False)
        r, c = np.indices(actual.shape).reshape(2, -1)
        directions = np.column_stack(((c-6)/20, (r-5)/20, np.ones(len(r))))
        hits, rays, triangles = ray(combined).intersects_location(
            np.zeros_like(directions), directions, multiple_hits=False)
        expected = np.full(actual.shape, np.nan)
        body = triangles < len(faces)
        expected[r[rays[body]], c[rays[body]]] = hits[body, 2]
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    poses[0][:] = np.eye(4); poses[0][2, 3] = 1.
    poses[1][:] = poses[0]
    qpos['right'][0] = 0.
    kw.update(source_from_camera=np.eye(4))
    for gap in (0., -1e-7, 1e-7, -1e-5, 1e-5):
        poses[1][2, 3] = 1. + gap
        actual = filt._render_primitive_base(**kw)[0][0]
        combined = trimesh.Trimesh(vertices=np.r_[vertices+[0, 0, 1.],
            vertices+[0, 0, 1.+gap]], faces=np.r_[faces, faces+len(vertices)], process=False)
        r, c = np.indices(actual.shape).reshape(2, -1)
        directions = np.column_stack(((c-6)/20, (r-5)/20, np.ones(len(r))))
        hits, rays, triangles = ray(combined).intersects_location(
            np.zeros_like(directions), directions, multiple_hits=False)
        expected = np.full(actual.shape, np.nan)
        body = triangles < len(faces)
        expected[r[rays[body]], c[rays[body]]] = hits[body, 2]
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected), err_msg=f'gap={gap}')


def test_warmed_ray_caches_exclude_only_current_camera_across_a_b_a(tmp_path, monkeypatch):
    trimesh = pytest.importorskip('trimesh')
    ray = pytest.importorskip('trimesh.ray.ray_pyembree').RayMeshIntersector
    urdf = tmp_path / 'fixture.urdf'; urdf.write_text('<robot name="fixture"/>')
    filt = RM65CapturedRobotObservationFilter(urdf_path=urdf)
    plane = trimesh.Trimesh(vertices=[[-.2, -.2, 1.], [.2, -.2, 1.], [.2, .2, 1.], [-.2, .2, 1.]],
                            faces=[[0, 1, 2], [0, 2, 3]], process=False)
    housing = trimesh.creation.box(extents=[.4, .4, .4])
    shapes = [SimpleNamespace(vertices=lambda m=m: np.asarray(m.vertices), num_tris=len(m.faces),
        tri_indices=lambda i, m=m: m.faces[i]) for m in (plane, housing, housing)]
    visual = SimpleNamespace(geometryObjects=[SimpleNamespace(name=n, geometry=s) for n, s in zip(
        ('left_arm_base_link_0', 'left_realsense_d405_0', 'right_realsense_d405_0'), shapes)])
    poses = [np.eye(4) for _ in shapes]; poses[2][2, 3] = .6
    pin = SimpleNamespace(GeometryData=lambda _: SimpleNamespace(oMg=[
        SimpleNamespace(homogeneous=p) for p in poses]), forwardKinematics=lambda *_: None,
        updateGeometryPlacements=lambda *_: None)
    model = SimpleNamespace(createData=lambda: None)
    monkeypatch.setattr(filt, '_runtime_components', lambda: (pin, trimesh, ray, model, visual, (0, 1, 2)))
    for i, shape in enumerate(shapes):
        filt._ray_mesh(i, shape)
    cached = dict(filt._ray_meshes)
    for z, expected in ((0., np.nan), (.6, .4), (0., np.nan)):
        camera = np.eye(4); camera[2, 3] = z
        actual = filt._render_primitive_base(depth_m=np.zeros((1, 1)),
            intrinsics={'fx': 100., 'fy': 100., 'cx': 0., 'cy': 0.}, source_from_camera=camera,
            capture_qpos_by_side={s: np.zeros(6) for s in ('left', 'right')})[0][0]
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert all(filt._ray_meshes[i][0] is cached[i][0] for i in cached)
