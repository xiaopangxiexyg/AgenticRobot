from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real.realsense_perception import RealAlignedRgbdCapture
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.rm65.captured_robot_observations import RM65CapturedRobotObservationFilterResult
from scripts.probes import inspect_safe_move_snapshot as probe

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def capture(role='main'):
    return RealAlignedRgbdCapture(source_id=probe.SOURCES[role], role=role, acquired_at=10.,
        station_revision=0, source_generation=1, frame_index=2, capture_group_id='test:cut',
        source_binding_hash='test:binding', width=8, height=8,
        intrinsics={'fx': 100., 'fy': 100., 'cx': 4., 'cy': 4.},
        source_from_camera=tuple(np.eye(4).ravel()),
        rgb=np.full((8, 8, 3), [20, 70, 130], np.uint8), depth_m=np.full((8, 8), .5),
        rgb_content_hash='test:rgb', depth_content_hash='test:depth',
        capture_qpos_by_side={side: np.zeros(6) for side in ('left', 'right')},
        pose_measured_at=10., pose_available_at=10.001)


def test_rgbd_backprojection_preserves_meter_depth_and_rgb_pixel_indices():
    cap = capture()
    pose = np.eye(4)
    pose[:3, 3] = [1., 2., 3.]
    mask = np.zeros((8, 8), bool)
    mask[4, 4] = True
    xyz, rows, columns = probe.backproject(cap.depth_m, cap.intrinsics, pose, mask)
    np.testing.assert_allclose(xyz, [[1., 2., 3.5]])
    assert rows.tolist() == columns.tolist() == [4]
    projected = probe.project(xyz, replace(cap, source_from_camera=tuple(pose.ravel())))
    np.testing.assert_allclose(projected['uv'], [[4., 4.]])
    assert projected['depth_consistent'].tolist() == [True]


def test_projection_distinguishes_occlusion_missing_depth_outside_and_behind():
    cap = capture()
    depth = cap.depth_m.copy()
    depth[4, 4] = 0
    missing = probe.project([[0, 0, .5]], replace(cap, depth_m=depth))
    assert missing['in_image'][0] and not missing['depth_consistent'][0]
    result = probe.project([[0, 0, .5], [0, 0, .7], [10, 0, .5], [0, 0, -.5]], cap)
    assert result['in_image'].tolist() == [True, True, False, False]
    assert result['depth_consistent'].tolist() == [True, False, False, False]


@pytest.mark.parametrize('problem', ['missing_view', 'skew', 'uncaptured'])
def test_select_frames_never_replaces_missing_or_incoherent_views(problem):
    report = {'status': 'captured', 'stationarity': {'left': {}}, 'views': {
        role: {'frames': [{'received_at': 10+i*.01, 'index': i} for i in range(3)]}
        for role in probe.ROLES}}
    if problem == 'missing_view':
        report['views'].pop('wrist_left')
    elif problem == 'skew':
        report['views']['wrist_right']['frames'] = [{'received_at': 11.}]
    else:
        report['status'] = 'failed'
    with pytest.raises(ValueError):
        probe.select_frames(report)


def test_exact_frame_selection_uses_camera_time_not_list_index():
    report = {'status': 'captured', 'stationarity': {'left': {}}, 'views': {
        role: {'frames': [{'received_at': 10+i*.01, 'index': i} for i in range(3)]}
        for role in probe.ROLES}}
    report['views']['wrist_left']['frames'].reverse()
    selected = probe.select_frames(report)
    assert all(frame['index'] == 2 for frame in selected.values())


def test_load_arrays_rejects_escape_and_preserves_exact_rgbd(tmp_path):
    rgb = np.full((8, 8, 3), 70, np.uint8)
    depth = np.full((8, 8), .42, np.float32)
    np.savez_compressed(tmp_path / 'frame.npz', rgb=rgb, depth_m=depth)
    saved_rgb, saved_depth = probe.load_arrays(tmp_path, {'file': 'frame.npz'})
    np.testing.assert_array_equal(saved_rgb, rgb)
    np.testing.assert_array_equal(saved_depth, depth)
    nested = tmp_path / 'nested'
    nested.mkdir()
    with pytest.raises(ValueError, match='escapes'):
        probe.load_arrays(nested, {'file': '../frame.npz'})


def test_snapshot_loader_uses_frame_near_qpos_and_closed_lifecycle(tmp_path):
    views = {}
    for role in probe.ROLES:
        cap = capture(role)
        np.savez_compressed(tmp_path / f'{role}.npz', rgb=cap.rgb, depth_m=cap.depth_m)
        views[role] = {'frames': [{'received_at': 10.+i*.01, 'file': f'{role}.npz',
            'source_generation': 2, 'source_frame_index': i, 'intrinsics': dict(cap.intrinsics)} for i in range(3)]}
    samples = [{side: {'started_at': t, 'finished_at': t+.001,
        'qpos_rad': [t*.00001]*6} for side in ('left', 'right')} for t in np.arange(9.8, 10.31, .05)]
    report = {'schema': 'agenticrobot.readonly_rgbd_qpos.v1', 'status': 'captured',
        'cleanup_complete': True, 'stationarity': {'left': {}}, 'qpos_samples': samples,
        'views': views, 'capture_started_at': 9.95, 'capture_finished_at': 10.1}
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps(report))
    calls = []

    class Geometry:
        def publish_measured_qpos(self, **kwargs):
            calls.append(kwargs)

        def pose_cut(self, **kwargs):
            assert kwargs['frame_acquired_at'] == 10.02
            assert abs(calls[-1]['measured_at']-10.0) < .00001
            return SimpleNamespace(source_from_camera=tuple(np.eye(4).ravel()), evidence_refs=('native-fk',))

    cuts = probe.load_captures(tmp_path, object(), SimpleNamespace(semantic_content_hash='binding'), Geometry())
    assert len(cuts) == 3 and len(calls) == 3
    assert all(cut.acquired_at == 10.02 and cut.rgb.shape == (8, 8, 3) for cut in cuts)
    report['cleanup_complete'] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='closed'):
        probe.load_captures(tmp_path, object(), SimpleNamespace(semantic_content_hash='binding'), Geometry())


def test_inspection_saves_inputs_masks_and_actual_projections_without_path_search(tmp_path, monkeypatch):
    events = []
    pose = np.eye(4)
    pose[2, 3] = .50

    class Oracle:
        def check(self, qpos):
            events.append('check')
            return JointConfigurationCheck(safe=False, collision_check_available=True,
                reason='observed_obstacle_clearance_blocked', minimum_distance_m=-.01)

        def world_obstacle_proxy_boxes(self, qpos):
            return [('test_box', pose, np.full(3, .03))]

    class Planner:
        def __init__(self, **kwargs):
            events.append('construct')
            self.oracle = Oracle()
            self.scene = SimpleNamespace(minimum_clearance_m=.02)
            assert kwargs['observed_obstacles'].reference_frame_id == 'world'

        def __call__(self, **kwargs):
            pytest.fail('A read-only scene inspection cannot invoke path search')

    monkeypatch.setattr(probe, 'RM65GraspJointTrajectoryPlanner', Planner)

    def clean(**kwargs):
        depth = kwargs['depth_m'].copy()
        mask = np.zeros(depth.shape, bool)
        mask[0] = True
        depth[mask] = 0
        return RM65CapturedRobotObservationFilterResult(depth, mask, {'method': 'test'})

    captures = [capture(role) for role in probe.ROLES]
    output = tmp_path / 'inspection'
    report = probe.inspect_captures(captures, cleaner=clean,
        config=probe.SafeMotionObservationConfig(), profile=SimpleNamespace(station_world_frame_id='world'),
        builder=object(), output=output, maximum_display_points=12)
    assert events == ['construct', 'check']
    assert report['device_connections'] is False and report['path_search_invoked'] is False
    assert report['motion_commands'] == []
    assert (output / 'inspection.html').is_file()
    assert '__INSPECTION_DATA__' not in (output / 'inspection.html').read_text()
    assert '__RGB_PREVIEW__' not in (output / 'inspection.html').read_text()
    assert '<img alt="main 原始 RGB" src="data:image/jpeg;base64,' in (output / 'inspection.html').read_text()
    for cap in captures:
        with np.load(output / f'{cap.role}.npz', allow_pickle=False) as data:
            np.testing.assert_array_equal(data['rgb'], cap.rgb)
            np.testing.assert_array_equal(data['depth_m'], cap.depth_m)
            assert data['self_removed_mask'][0].all()
            assert not data['retained_obstacle_mask'][0].any()
        view = next(view for view in report['views'] if view['role'] == cap.role)
        assert view['nearest_projection']['in_image'] is True
        assert view['nearest_projection']['depth_consistent'] is True
        assert (output / f'{cap.role}-nearest.jpg').is_file()
    with pytest.raises(FileExistsError):
        probe.inspect_captures(captures, cleaner=clean, config=probe.SafeMotionObservationConfig(),
            profile=SimpleNamespace(station_world_frame_id='world'), builder=object(), output=output)


def test_failed_persistence_does_not_emit_a_complete_inspection(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.np, 'savez_compressed', lambda *_a, **_kw: (_ for _ in ()).throw(OSError('disk full')))
    def clean(**kwargs):
        return RM65CapturedRobotObservationFilterResult(kwargs['depth_m'], np.zeros((8, 8), bool), {})
    output = tmp_path / 'failed'
    with pytest.raises(OSError, match='disk full'):
        probe.inspect_captures([capture(role) for role in probe.ROLES], cleaner=clean,
            config=probe.SafeMotionObservationConfig(), profile=SimpleNamespace(station_world_frame_id='world'),
            builder=object(), output=output)
    assert not (output / 'inspection.html').exists()
