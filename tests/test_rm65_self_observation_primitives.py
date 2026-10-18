import json

import numpy as np
import pytest
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
    monkeypatch.setattr(filt, '_render_depth_maps', render)
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
