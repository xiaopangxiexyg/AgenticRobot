"""Offline benchmark consumers use explicit shared policy and fresh output."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.rm65.captured_robot_observations import RM65CapturedRobotObservationFilterResult
from scripts.probes import benchmark_self_observation as probe


def inputs(tmp_path):
    session = tmp_path / 'session'; session.mkdir()
    snapshot = session / 'state_0001/capture/snapshot.json'
    snapshot.parent.mkdir(parents=True); snapshot.write_text('{}')
    entries = [dict(status='captured', state='state_0001', snapshot='state_0001/capture/snapshot.json')]
    (session / 'samples.jsonl').write_text(json.dumps(entries[0])+'\n')
    config = tmp_path / 'filter.json'
    config.write_text(json.dumps(dict(geometry_mode='cad_surface', depth_tolerance_m=.025,
        tool_surface_bounds_m=[[-.045, .045], [-.05, .05], [-.05, .05]])))
    return dict(session=session, urdf=tmp_path / 'physical.urdf', envelope_profile=None,
        output=tmp_path / 'output', captured_robot_filter_config=config,
        captured_robot_self_envelope_urdf=tmp_path / 'stepped.urdf')


def test_benchmark_reuses_explicit_policies_protects_targets_and_persists_exact_inputs(tmp_path, monkeypatch):
    args = inputs(tmp_path); calls = []
    profile = SimpleNamespace(tool_tcp_pose_flange_m_rad_by_side={s: np.zeros(6) for s in ('left','right')})
    def filt(**kwargs):
        remove = ~kwargs['target_mask']
        depth = kwargs['depth_m'].copy(); depth[remove] = 0.
        calls.append(kwargs)
        return RM65CapturedRobotObservationFilterResult(depth, remove, {'method':'test'})
    bound = []
    def resolve(context, urdf, **kwargs):
        bound.append(kwargs)
        return profile, None, None, None, filt
    monkeypatch.setattr(probe, 'resolve_inputs', resolve)
    primitive = []
    def create(**kwargs):
        primitive.append(kwargs)
        return filt
    monkeypatch.setattr(probe.RM65CapturedRobotObservationFilter, 'for_binding', create)
    capture = SimpleNamespace(role='main', depth_m=np.full((2, 3), .8),
        rgb=np.full((2, 3, 3), 90, np.uint8), intrinsics={'fx':2.,'fy':2.,'cx':1.,'cy':1.},
        source_from_camera=np.eye(4), capture_qpos_by_side={s:np.zeros(6) for s in ('left','right')})
    monkeypatch.setattr(probe, 'load_captures', lambda *args: [capture])
    report = probe.compare_session(**args)
    assert len(report['views']) == 1 and len(calls) == 5
    assert calls[-1]['target_mask'].all()
    assert bound[0]['filter_config'].depth_tolerance_m == primitive[0]['config'].depth_tolerance_m == .025
    assert bound[0]['filter_config'].geometry_mode == 'cad_surface'
    assert primitive[0]['config'].geometry_mode == 'link_primitives'
    assert primitive[0]['self_observation_proxy_urdf_path'] == args['captured_robot_self_envelope_urdf']
    assert report['device_connections'] is False and report['motion_commands'] == []
    assert report['reference_is_pixel_ground_truth'] is False
    with np.load(args['output'] / 'state_0001-main.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['depth_m'], capture.depth_m)
        np.testing.assert_array_equal(data['rgb'], capture.rgb)
        assert data['cad_removed'].all() and data['primitive_removed'].all()
    assert json.loads((args['output'] / 'result.json').read_text())['views'][0]['primitive_only'] == 0


@pytest.mark.parametrize('problem', ['state_escape', 'snapshot_escape', 'duplicate', 'uncaptured', 'empty'])
def test_benchmark_rejects_ambiguous_archives_before_any_output(tmp_path, monkeypatch, problem):
    args = inputs(tmp_path)
    entry = dict(status='captured', state='state_0001', snapshot='state_0001/capture/snapshot.json')
    if problem == 'state_escape':entry['state'] = '../escape'
    elif problem == 'snapshot_escape':
        other = tmp_path / 'snapshot.json'; other.write_text('{}')
        entry['snapshot'] = '../snapshot.json'
    elif problem == 'uncaptured':entry['status'] = 'failed'
    entries = [] if problem == 'empty' else [entry, entry] if problem == 'duplicate' else [entry]
    (args['session'] / 'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in entries))
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid archive must fail before profile construction')
    monkeypatch.setattr(probe, 'resolve_inputs', forbidden)
    with pytest.raises(ValueError):probe.compare_session(**args)
    assert not args['output'].exists()


def test_benchmark_cannot_implicitly_select_extended_policy(tmp_path, monkeypatch):
    args = inputs(tmp_path); args['captured_robot_filter_config'].write_text('{}')
    with pytest.raises(ValueError, match='explicit cad_surface'):
        probe.compare_session(**args)
    assert not args['output'].exists()
