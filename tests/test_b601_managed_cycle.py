"""Single-cycle intents and independent completion receipts; no real devices."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_b601_automatic_calibration import AcquisitionReplay
from test_b601_cruise_velocity import controller_from_profile
from test_b601_workspace_calibration import LIMITS

from control_station.b601_torque_console.managed_cycle import (
    B601ManagedCycle,
    assess_managed_result,
)
from control_station.b601_torque_console.server import B601TorqueConsoleServer
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
from scripts.hardware import rebot_b601_rs_owner as owner

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT/'docs/hardware/profiles/b601_j2_j3_near_zero_command_field_v1.json'
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def managed_replay(tmp_path):
    controller = controller_from_profile(tmp_path, profile=PROFILE)
    r = AcquisitionReplay(controller._workspace.settings, controller=controller, position_limits=LIMITS)
    manager = B601ManagedCycle(controller, now_sec=r.now)
    return r, manager


@pytest.mark.parametrize('terminal', ['complete', 'soft_timeout', 'intervention', 'page_unload', 'hardware_fault'])
def test_managed_cycle_uses_existing_admission_return_and_disable(tmp_path, terminal):
    r, manager = managed_replay(tmp_path)
    assert manager.prepare_timeout_sec == r.controller.workspace_configuration.prepare_timeout_sec
    triggered = False
    for n in range(22000):
        q = list(r.q)
        goal = r.controller._workspace._coordinated.pose.goal
        if goal is not None:
            for i in (1, 2):
                q[i] += max(-.008*.025, min(.008*.025, goal[i]-q[i]))
        r.tick(q, dt_sec=.025, permit=False)
        assert manager.tick(now_sec=r.now+.0008, owner_active=True) is None
        batch = r.controller.status()['automatic_calibration']
        if manager.start_count and batch['state'] == 'sweep' and not triggered and terminal != 'complete':
            triggered = True
            native = r.controller._workspace._coordinated
            if terminal == 'soft_timeout':
                native._leg_started = r.now-native.config.max_sweep_leg_sec-.1
            elif terminal == 'intervention':
                r.controller.set_human_intervention(True)
            elif terminal == 'page_unload':
                r.controller.submit('withdraw', now_sec=r.now, withdraw_source='page_unload')
            else:
                native.recording_failed('joint_velocity_exceeded')
        if r.controller.automatic_disable_ready or batch['state'] in {'failed', 'cancelled'}:
            break
    else:
        pytest.fail(str(r.controller.status()))
    result = r.controller.result_record()
    assert manager.start_count == 1
    if terminal in {'complete', 'soft_timeout'}:
        assert r.controller.automatic_disable_ready
        assert result['workspace']['supported_return']['completed']
        stop = {'requested': False, 'full_disable_requested': False}
        confirmation = {'confirmed': False}
        assert probe._request_supported_return_disable(stop, r.controller, confirmation)
        assert stop['full_disable_requested'] and confirmation['confirmed']
        if terminal == 'complete':
            assert len(result['motion']['completed_legs']) == 28
        else:
            assert result['reason'] == 'leg_timeout'
    else:
        assert not r.controller.automatic_disable_ready
        assert not result['workspace']['supported_return']['completed']
    before = manager.heartbeat_count
    for n in range(20):
        manager.tick(now_sec=r.now+.1+n*.1, owner_active=True)
    assert manager.start_count == 1 and manager.heartbeat_count == before


def test_managed_start_waits_for_consumed_mode_and_supported_origin(tmp_path):
    r, manager = managed_replay(tmp_path)
    assert not r.controller.status()['mode_selection']['ready']
    assert manager.start_count == 0  # No post-selection zero receipt yet.
    for _ in range(20):
        r.tick(permit=False)
        manager.tick(now_sec=r.now+.0008, owner_active=False)
    assert manager.start_count == 0
    manager.tick(now_sec=r.now+.11, owner_active=True)
    assert manager.start_count == 1


def test_managed_context_is_declared_once_before_connection_and_does_not_override_human(tmp_path):
    controller = controller_from_profile(tmp_path, profile=PROFILE)
    path = tmp_path/'contexts.jsonl'
    server = B601TorqueConsoleServer(controller, assessment_path=path, automatic_recording=True,
        human_intervention_handler=controller.set_human_intervention)
    try:
        server.declare_managed_context()
        assert controller.status()['automatic_calibration']['human_intervention'] is False
        row = json.loads(path.read_text().splitlines()[0])
        assert row['declaration_source'] == 'managed_cycle' and row['human_intervention'] is False
        controller.owner_ready()
        with pytest.raises(ValueError, match='unopened'):
            server.declare_managed_context()
        controller.set_human_intervention(True)
        assert controller.status()['automatic_calibration']['human_intervention'] is True
    finally:
        server.close()


def test_managed_preparation_timeout_requests_only_existing_protective_hold():
    controller = SimpleNamespace(workspace_configuration=SimpleNamespace(coordinated_holding=True, prepare_timeout_sec=1),
        config=SimpleNamespace(prepare_timeout_sec=1), submit=lambda *a, **k: None,
        status=lambda: {'owner_ready': True, 'automatic_calibration': {
            'human_intervention': False, 'state': 'idle', 'workspace': {}},
            'mode_selection': {'ready': False, 'locked': False}})
    manager = B601ManagedCycle(controller, now_sec=0)
    assert manager.tick(now_sec=.2, owner_active=True) is None
    assert manager.tick(now_sec=1.3, owner_active=True) == 'protective_hold'
    assert manager.phase == 'operator_required' and manager.start_count == 0
    assert manager.tick(now_sec=1.5, owner_active=True) is None


def test_managed_assessment_separates_cleanup_acquisition_and_model_readiness():
    payload = {'status': 'COMPLETED_WITH_WARNINGS', 'cycle_disable_confirmation': {'confirmed': True},
        'automatic_calibration': {'state': 'failed', 'reason': 'leg_timeout', 'outcome': {'action': 'retain_partial_data_stop_acquisition'},
            'workspace': {'supported_return': {'completed': True}}},
        'cleanup': {k: True for k in ('device_cleanup_confirmed', 'motor_disable_confirmed',
            'active_report_disable_confirmed', 'driver_limits_restore_confirmed')}}
    payload['cleanup'].update(transport_release={'completed': True}, errors=[])
    report = {'acquisition_complete': False, 'model_readiness': {'status': 'INSUFFICIENT_DATA'}}
    result = assess_managed_result(payload, report)
    assert result['status'] == 'PARTIAL_ACQUISITION' and result['lifecycle_complete']
    assert not result['acquisition_complete']
    broken = copy.deepcopy(payload)
    broken['cleanup']['motor_disable_confirmed'] = False
    assert assess_managed_result(broken, report)['status'] == 'OPERATOR_REQUIRED'
    payload['automatic_calibration'].update(state='complete', outcome={'action': 'complete'})
    assert not assess_managed_result(payload, report)['acquisition_complete']
    report['acquisition_complete'] = True
    assert assess_managed_result(payload, report)['status'] == 'COMPLETE'
    payload['automatic_calibration_report_status'] = 'ERROR'
    assert assess_managed_result(payload, report)['status'] == 'REPORT_ERROR'
    payload['status'] = 'ERROR'
    assert assess_managed_result(payload, report)['status'] == 'EXECUTION_ERROR'


def test_managed_flag_is_explicit_and_forwarded_to_the_existing_owner(tmp_path):
    controller = controller_from_profile(tmp_path, profile=PROFILE, extra=('--torque-console-managed-cycle',))
    assert controller.workspace_configuration.coordinated_holding is not None
    args = owner._parser().parse_args(['--allow-motion', '--torque-console-managed-cycle'])
    with pytest.raises(ValueError, match='workspace profile'):
        owner.build_probe_command(args, run_dir=tmp_path)
    parsed = probe.build_parser().parse_args(['--config', str(tmp_path/'config.yaml'),
        '--rebot-source-root', str(tmp_path), '--torque-console-managed-cycle'])
    with pytest.raises(probe.LowGainProbeError, match='workspace profile'):
        probe._validate_cli(parsed)
