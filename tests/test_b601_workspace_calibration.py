"""Scripted encoder/receipt contracts, never a plant or hardware acceptance."""

from __future__ import annotations

import gzip
import io
import json
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_b601_automatic_calibration import AcquisitionReplay
from test_b601_calibration_session import settings as shared_settings

from control_station.b601_torque_console.automatic_calibration import (
    AutomaticCalibrationConfig,
    B601AutomaticCalibrationController,
)
from control_station.b601_torque_console.automatic_report import (
    _workspace_motion_windows,
    build_automatic_report,
)
from control_station.b601_torque_console.calibration_cycle import (
    B601AutomaticCalibrationCycleController,
    CalibrationCycleConfig,
)
from control_station.b601_torque_console.mode_controller import (
    B601CalibrationModeController,
)
from control_station.b601_torque_console.workspace_calibration import (
    B601WorkspaceCalibrationController,
    WorkspaceCalibrationConfig,
)
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
from scripts.hardware import rebot_b601_rs_owner as owner

pytestmark = [pytest.mark.unit, pytest.mark.contract]
ZERO = (0.0,) * 6
LIMITS = ((-1.0, 1.0), (0.0, 2.2), (0.0, 1.6), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))


@pytest.fixture
def settings(tmp_path):
    return shared_settings.__wrapped__(tmp_path)


def replay_workspace(settings, *, mode_wrapper=False, initial_q=None, **overrides):
    config = AutomaticCalibrationConfig(motion_plan="local_identification",
        motion_acquisition_arrival_tolerance_rad=.0039,
        motion_position_resolution_rad=.0004, motion_responsive_tracking_brake=True)
    workspace = WorkspaceCalibrationConfig(**{
        "joint2_centres_deg": (15,), "joint3_centres_deg": (12,),
        "joint2_acquisition_range_deg": (0, 30), "joint3_acquisition_range_deg": (0, 20),
        "maximum_start_angle_deg": 6,
        "gravity_observation_sec": .1, "positioning_torque_fall_nm_s": 2.0,
        **overrides})
    resolved = B601AutomaticCalibrationController(settings, config).settings
    controller = B601WorkspaceCalibrationController(resolved, config, workspace)
    if mode_wrapper:
        supported = B601AutomaticCalibrationCycleController(
            B601AutomaticCalibrationController(settings, config), CalibrationCycleConfig())
        controller = B601CalibrationModeController(supported, workspace=controller)
        controller.submit("select_mode", now_sec=1.0, operation_mode="workspace")
    origin = initial_q or (0.0, math.radians(5), math.radians(5), 0.0, 0.0, 0.0)
    return AcquisitionReplay(settings, controller=controller, position_limits=LIMITS, initial_q=origin)


def advance(replay, *, until=None, ticks=60000):
    for _ in range(ticks):
        batch = replay.controller.status()["automatic_calibration"]
        if until is not None and until(batch):
            return
        if batch["state"] in {"complete", "failed", "cancelled"}:
            return
        q = list(replay.q)
        j = int(replay.audit.get("joint_name", "joint3")[-1]) - 1
        motion = replay.audit.get("motion_control")
        if motion and motion["phase"] == "track":
            q[j] = motion["q_reference_rad"]
        elif replay.audit.get("target_offset_rad") and (
            replay.audit.get("phase") == "breakaway"
            or replay.audit.get("phase") == "releasing"
            and not replay.audit.get("release_terminal")
            and abs(q[j] - replay.audit["anchor_rad"]) < .001
        ):
            q[j] += math.copysign(.0001, replay.audit["target_offset_rad"])
        replay.tick(q)
    pytest.fail(str(replay.controller.result_record()))


def test_workspace_first_plan_stays_near_zero_and_unfolds_j3_before_moving_j2():
    config = WorkspaceCalibrationConfig()
    poses = config.pose_centres()
    assert len(poses) == 4
    assert {round(math.degrees(p[0])) for p in poses} == {3, 6}
    assert {round(math.degrees(p[1])) for p in poses} == {3, 6}
    assert all(0 < j2 < math.radians(10) and 0 < j3 < math.radians(10) for j2, j3 in poses)
    assert poses[1][1] == poses[2][1]
    tasks = config.tasks(ZERO)
    assert [item["joint"] for item in tasks[:2]] == ["joint3", "joint2"]
    assert [item["joint"] for item in tasks[-3:-1]] == ["joint2", "joint3"]
    assert [item["target"] for item in tasks[-3:-1]] == [0, 0]


def test_workspace_near_origin_native_batch_completes_and_never_targets_wide_grid(settings):
    replay = replay_workspace(settings, initial_q=ZERO, joint2_centres_deg=(3, 6),
        joint3_centres_deg=(3, 6), joint2_acquisition_range_deg=(0, 10),
        joint3_acquisition_range_deg=(0, 10), maximum_start_angle_deg=2)
    replay.begin()
    advance(replay)
    result = replay.controller.result_record()
    assert result["state"] == "complete", result["reason"]
    assert len(result["motion"]["completed_legs"]) == 96
    assert result["workspace"]["completed_joint_acquisitions"] == 8
    assert result["workspace"]["moves"][0]["task"]["joint"] == "joint3"
    assert all(0 <= frame["qpos_rad"][j] <= math.radians(10)
               for frame in replay.audits for j in (1, 2))
    assert replay.controller.cycle_disable_confirmation_ready


def test_workspace_rejects_start_away_from_zero_and_retains_narrow_feedback_envelope(settings):
    replay = replay_workspace(settings, maximum_start_angle_deg=2)
    with pytest.raises(ValueError, match="within 2 degrees"):
        replay.begin()
    assert replay.delta == ZERO
    replay = replay_workspace(settings, initial_q=ZERO, joint2_centres_deg=(3,),
        joint3_centres_deg=(3,), joint2_acquisition_range_deg=(0, 10),
        joint3_acquisition_range_deg=(0, 10), maximum_start_angle_deg=2)
    replay.begin()
    replay.tick()
    q = list(replay.q)
    q[1] = math.radians(10.1)
    replay.tick(q)
    assert replay.audit["fault"] == "workspace_position_envelope"
    assert replay.delta == ZERO


@pytest.mark.parametrize("values", [
    {"joint2_centres_deg": (-.3, 60)}, {"joint2_centres_deg": (120,)},
    {"joint3_centres_deg": (91,)}, {"joint3_centres_deg": (40, 10)},
    {"joint2_drive_kd": (0,) * 5}, {"max_command_exposure_nm_sec": float("inf")},
    {"positioning_support_soft_speed_recovery": 1},
    {"positioning_support_soft_speed_recovery": None},
])
def test_workspace_config_rejects_unbounded_or_outside_targets(values):
    with pytest.raises((ValueError, TypeError)):
        WorkspaceCalibrationConfig(**values)


def test_workspace_native_both_axis_sweeps_segmented_positioning_return_and_trace(settings, tmp_path):
    from control_station.b601_torque_console.automatic_report import (
        save_automatic_report,
    )
    replay = replay_workspace(settings)
    origin = replay.q
    replay.begin()
    advance(replay)
    result = replay.controller.result_record()
    assert result["state"] == "complete", result["reason"]
    workspace = result["workspace"]
    assert workspace["completed_joint_acquisitions"] == 2
    assert len(result["motion"]["completed_legs"]) == 24
    assert {item["task"]["joint"] for item in workspace["acquisitions"]} == {"joint2", "joint3"}
    assert len(workspace["gravity_observations"]) == 2
    assert len(workspace["moves"]) > 4
    assert all(abs(a - b) <= .0039 for a, b in zip(origin, replay.q))
    assert replay.delta == replay.last_kd == ZERO
    assert replay.controller.cycle_disable_confirmation_ready
    assert workspace["auto_apply"] is False
    assert result["onset_trials"] == []
    for audit in replay.audits:
        j = int(audit["joint_name"][-1]) - 1
        assert j in {1, 2}
        if (audit.get("motion_control") or {}).get("phase") == "track":
            assert all(0 <= audit["qpos_rad"][i] <= LIMITS[i][1] for i in (1, 2))
    stream = io.StringIO()
    queued = SimpleNamespace(drain_records=lambda: replay.records)
    probe._flush_automatic_calibration_records(queued, probe._DynamicsTrace(stream))
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert sum(row["kind"] == "automatic_calibration_started" for row in rows) == 1
    assert any(row["kind"] == "workspace_gravity_observation" for row in rows)
    (tmp_path / "b601_owner.debug.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in [*rows, result]))
    (tmp_path / "b601_owner.transparency.jsonl").write_text("")
    report = build_automatic_report(tmp_path)
    assert report["acquisition_complete"] is True
    assert report["workspace"]["completed_joint_acquisitions"] == 2
    assert report["onset_data_assessment"]["status"] == "NOT_REQUESTED_GRAVITY_FIRST"
    assert report["gravity_identification"]["auto_apply"] is False
    # Missing final serialization must not erase already committed stage data.
    (tmp_path / "b601_owner.debug.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in [*rows,
            {"kind": "torque_console_guard", "reason": "coordinated_reference_lag_stop",
             "joint_name": "joint2", "time_sec": 10., "motion_fault": {"brake_trigger_lag_rad": -.019647}},
            {"kind": "control_observation_complete", "boundary": "after_control_cleanup_and_backend_close_attempt",
             "recorded_monotonic_sec": 11., "cleanup_errors": []}]))
    recovered = build_automatic_report(tmp_path)
    assert recovered["acquisition_complete"] is False
    assert recovered["acquisition_outcome"]["reason"] == "terminal_record_missing"
    assert recovered["workspace"]["recovered_from_stage_records"] is True
    assert len(recovered["workspace"]["moves"]) == len(workspace["moves"])
    assert len(recovered["workspace"]["gravity_observations"]) == 2
    assert len(recovered["workspace"]["acquisitions"]) == 2
    assert recovered["gravity_identification"]["auto_apply"] is False
    assert recovered["recovery"]["first_guard"]["reason"] == "coordinated_reference_lag_stop"
    assert recovered["recovery"]["first_guard"]["motion_fault"]["brake_trigger_lag_rad"] == -.019647
    assert recovered["recovery"]["control_cleanup_boundary"]["cleanup_errors"] == []
    # Recovery output must not mutate or write into the retained source run.
    source_before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    destination = tmp_path / "fresh-recovery"
    destination.mkdir()
    report_path = save_automatic_report(tmp_path, output_dir=destination)
    exported = json.loads(report_path.read_text())
    assert exported == json.loads(json.dumps(recovered))
    assert exported["source_trace"] == str(tmp_path / "b601_owner.debug.jsonl")
    assert exported["acquisition_complete"] is False
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == source_before
    with pytest.raises(FileExistsError):
        save_automatic_report(tmp_path, output_dir=destination)
    assert json.loads(report_path.read_text()) == exported


def test_workspace_native_gui_wrapper_advances_between_pose_centres(settings):
    replay = replay_workspace(settings, mode_wrapper=True, joint2_centres_deg=(12, 15))
    replay.controller.submit("arm", now_sec=replay.now + .001, operation_mode="workspace")
    advance(replay)
    result = replay.controller.result_record()
    assert result["state"] == "complete", result["reason"]
    assert result["mode_selection"]["selected"] == "workspace"
    assert result["mode_selection"]["locked"]
    assert result["workspace"]["completed_joint_acquisitions"] == 4
    assert len(result["motion"]["completed_legs"]) == 48
    assert replay.controller.cycle_disable_confirmation_ready


def test_workspace_idle_mode_reentry_rebases_feedback_without_relaxing_active_gap(settings):
    replay = replay_workspace(settings)
    replay.tick(dt_sec=1.0)
    replay.begin()
    replay.tick()
    with pytest.raises(ValueError, match="feedback_clock_or_gap"):
        replay.tick(dt_sec=.05)
    assert replay.controller.result_record()["state"] == "failed"


def test_workspace_cancelled_position_stage_stays_diagnostic_and_serializable(settings):
    replay = replay_workspace(settings)
    replay.begin()
    advance(replay, until=lambda batch: any(abs(v) > .05 for v in replay.delta))
    # An interrupted positioning child must not be misclassified as a sweep.
    replay.controller._session.cancel(now_sec=replay.now, reason="scripted_positioning_stop")
    advance(replay)
    result = replay.controller.result_record()
    assert result["state"] == "cancelled"
    assert result["workspace"]["moves"][-1]["result"]["state"] == "cancelled"
    assert result["workspace"]["acquisitions"] == []
    assert result["motion"]["completed_legs"] == []


@pytest.mark.parametrize("stage", ["position", "sweep"])
def test_workspace_short_target_retains_terminal_receipt_and_remains_passive(settings, stage):
    replay = replay_workspace(settings, initial_q=ZERO, joint2_centres_deg=(3,),
        joint3_centres_deg=(3,), joint2_acquisition_range_deg=(0, 10),
        joint3_acquisition_range_deg=(0, 10), maximum_start_angle_deg=2)
    replay.begin()
    advance(replay, until=lambda batch: batch["state"] == stage and replay.audit.get("phase") == "track")
    replay.controller._delegate.submit("stop_and_hold", now_sec=replay.now + .001)
    for _ in range(300):
        replay.tick()
        if replay.controller.status()["automatic_calibration"]["state"] == "failed":
            break
    result = replay.controller.result_record()
    assert result["reason"] == "stopped_short_of_target"
    assert result["outcome"]["severity"] == "warning"
    assert not replay.controller.cycle_disable_confirmation_ready
    archive = "moves" if stage == "position" else "acquisitions"
    retained = result["workspace"][archive]
    assert retained[-1]["result"]["state"] == "failed"
    assert retained[-1]["result"]["pending_dispatch"] is False
    assert any(row["kind"] == "workspace_calibration_stage_result"
        and row["result"]["reason"] == "stopped_short_of_target" for row in replay.records)
    for _ in range(10):
        replay.tick()
        assert replay.audit["phase"] != "fault"
        assert replay.delta == replay.last_kd == ZERO
        assert replay.controller.result_record()["workspace"][archive] == retained
    replay.controller.owner_stopped(now_sec=replay.now + .001, reason="operator_stop")
    assert replay.controller.result_record()["workspace"][archive] == retained


@pytest.mark.parametrize("stage", ["position", "sweep"])
def test_workspace_owner_stop_preserves_unconfirmed_child_without_fake_receipt(settings, stage):
    replay = replay_workspace(settings, initial_q=ZERO, joint2_centres_deg=(3,),
        joint3_centres_deg=(3,), joint2_acquisition_range_deg=(0, 10),
        joint3_acquisition_range_deg=(0, 10), maximum_start_angle_deg=2)
    replay.begin()
    advance(replay, until=lambda batch: batch["state"] == stage and replay.audit.get("phase") == "track")
    previous_receipt = dict(replay.receipt)
    replay.tick(ack=False)
    replay.controller.owner_stopped(now_sec=replay.now + .001, reason="dispatch_interrupted")
    result = replay.controller.result_record()
    retained = result["workspace"]["moves" if stage == "position" else "acquisitions"][-1]
    assert result["state"] == "failed"
    assert retained["partial_stage"] is True
    assert retained["result"]["pending_dispatch"] is True
    assert retained["result"]["state"] == "moving"
    assert result["last_complete_dispatch"]["sequence"] == previous_receipt["sequence"]
    assert result["pending_dispatch"]["feedback_sequence"] == replay.sequence
    assert result["pending_dispatch"]["receipt_confirmed"] is False
    assert not replay.controller.status()["owner_ready"]
    assert not replay.controller.cycle_disable_confirmation_ready


@pytest.mark.parametrize("reason", ["human", "page_unload", "budget"])
def test_workspace_withdraws_without_return_or_rearming(settings, reason):
    replay = replay_workspace(settings, **({"max_command_exposure_nm_sec": .1} if reason == "budget" else {}))
    replay.begin()
    advance(replay, until=lambda batch: any(abs(v) > .05 for v in replay.delta))
    if reason == "human":
        replay.controller.set_human_intervention(True)
    elif reason == "page_unload":
        replay.controller.submit("withdraw", now_sec=replay.now, withdraw_source="page_unload")
    advance(replay)
    result = replay.controller.result_record()
    assert result["state"] in {"cancelled", "failed"}
    assert replay.delta == ZERO
    assert not replay.controller.cycle_disable_confirmation_ready
    assert all(item["task"]["kind"] != "return" for item in result["workspace"]["moves"])
    with pytest.raises(ValueError):
        replay.begin()


def test_workspace_rejects_incomplete_receipt_and_changed_support_before_advancing(settings):
    replay = replay_workspace(settings)
    replay.begin()
    with pytest.raises(ValueError, match="incomplete_six_axis_dispatch"):
        replay.tick(receipt_updates={"sent_joint_names": ["joint2"]})
    assert replay.controller.result_record()["state"] == "failed"
    assert replay.controller.result_record()["workspace"]["completed_joint_acquisitions"] == 0


@pytest.mark.parametrize("fault", ["support", "speed", "position", "missing_ack"])
def test_workspace_retains_existing_guards_and_never_claims_completion(settings, fault):
    replay = replay_workspace(settings)
    replay.begin()
    replay.tick()
    if fault == "missing_ack":
        replay.tick(ack=False)
        with pytest.raises(ValueError, match="not confirmed"):
            replay.tick()
    else:
        q = list(replay.q)
        if fault == "support":
            q[3] += replay.controller.limits.max_support_drift_rad + .001
        elif fault == "speed":
            advance(replay, until=lambda batch: any(abs(v) > .05 for v in replay.delta))
            q = list(replay.q)
            q[1] += replay.controller.limits.max_joint_velocity_rad_s * .011
        else:
            q[1] = math.radians(121)
        replay.tick(q)
        assert replay.audit["phase"] == "fault"
        assert replay.delta == ZERO
    assert not replay.controller.cycle_disable_confirmation_ready
    assert replay.controller.result_record()["workspace"]["completed_joint_acquisitions"] == 0


def test_workspace_dispatch_start_deadline_and_elastic_feedback_boundary(settings):
    replay = replay_workspace(settings)
    replay.begin()
    replay.tick(ack=False)
    with pytest.raises(ValueError, match="deadline"):
        replay.controller.validate_dispatch_start(feedback_sequence=replay.sequence,
            feedback_timestamp_sec=replay.now, now_sec=replay.now + .040001)
    q, changed = probe.normalize_feedback_boundary_qpos(
        (0.0, math.radians(-.3), .1, 0.0, 0.0, 0.0), name="elastic_feedback")
    assert q[1] == 0.0
    assert "joint2" in changed


def test_workspace_preserves_completed_legs_when_later_cancelled(settings):
    replay = replay_workspace(settings)
    replay.begin()
    advance(replay, until=lambda batch: batch["workspace"]["completed_joint_acquisitions"] == 1)
    replay.controller.submit("withdraw", now_sec=replay.now)
    advance(replay)
    result = replay.controller.result_record()
    assert result["state"] == "cancelled"
    assert len(result["motion"]["completed_legs"]) == 12
    assert result["workspace"]["acquisitions"][0]["task"]["joint"] == "joint2"


def test_workspace_changing_command_windows_remain_available_without_claiming_friction(tmp_path):
    names = [f"joint{i}" for i in range(1, 7)]
    rows = []
    for direction in (-1, 1):
        for n in range(50):
            sequence = len(rows) + 1
            t = sequence * .01
            q = [0.0, .3 + direction * .01 * n * .01, .3, 0.0, 0.0, 0.0]
            rows.append({"kind": "active_dispatch", "sequence": sequence,
                "feedback_timestamp_sec": t, "recorded_monotonic_sec": t + .001,
                "qpos_rad": q, "gravity_tau_nm": [0, 2, 3, 0, 0, 0],
                "conditioned_tau_nm": [0, 2 + direction * (.1 + .002 * n), 3, 0, 0, 0],
                "vendor_kd_nm_s_per_rad": [0.4, .6, .4, .6, .2, .1],
                "host_observation": {"host_receive_freshness_verified": True},
                "dispatch_receipt": {"accepted": True, "requested_joint_names": names,
                    "sent_joint_names": names, "errors": [], "failed_joint_names": [],
                    "mode": "mit", "frame_id": "b601_rs/joint_mit_reference",
                    "source_sequence": sequence, "sequence": sequence + 500,
                    "requested_at_sec": t + .0001, "completed_at_sec": t + .0008},
                "torque_console": {"phase": "track", "calibration_trial": {
                    "trial_id": str(direction), "state": "moving", "role": "training"},
                    "automatic_calibration": {"human_intervention": False, "workspace": {
                        "current_task": {"kind": "sweep", "pose": 0, "joint": "joint2"}}}}})
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = _workspace_motion_windows(trace, {"acquisitions": [{"result": {
        "completed_legs": [{"trial_id": "1"}]}}]})
    assert report["window_count"] >= 2
    assert any(row["command_residual_span_nm"] > .02 for row in report["windows"])
    assert {math.copysign(1, row["velocity_rad_s"]) for row in report["windows"]} == {-1, 1}
    assert {row["source_leg_completed"] for row in report["windows"]} == {False, True}
    assert all(row["measured_output_torque_nm"] is None for row in report["windows"])
    assert report["auto_apply"] is False


def test_workspace_samples_stream_to_lossless_archive_without_bulk_expansion(tmp_path, monkeypatch):
    samples = probe._ProbeSampleArchive()
    expected = [{"sequence": 1, "qpos_rad": [0, .3, .4, 0, 0, 0]},
                {"sequence": 2, "qpos_rad": [0, .31, .4, 0, 0, 0]}]
    for sample in expected:
        samples.append(sample)
    monkeypatch.setattr(samples, "materialize", lambda: pytest.fail("bulk expansion"))
    output = tmp_path / "owner.json"
    payload = {"status": "OK", "automatic_calibration": {"operation_mode": "workspace"}, "samples": samples}
    probe._output_payload(payload, output)
    result = json.loads(output.read_text())
    assert result["samples"] == []
    assert result["sample_storage"]["sample_count"] == 2
    with gzip.open(result["sample_storage"]["sample_archive_path"], "rt") as source:
        assert [json.loads(line) for line in source] == expected
    assert len(payload["samples"]) == 2
    with pytest.raises(FileExistsError):
        probe._output_payload(payload, output)
    assert json.loads(output.read_text()) == result


def test_workspace_archive_interrupt_preserves_primary_and_partial_samples(tmp_path):
    class InterruptedSamples:
        def __len__(self):
            return 2

        def __iter__(self):
            yield {"sequence": 1}
            raise KeyboardInterrupt

    output = tmp_path / "owner.json"
    with pytest.raises(KeyboardInterrupt):
        probe._output_payload({"status": "FAIL", "stop_reason": "leg_timeout",
            "automatic_calibration": {"operation_mode": "workspace"},
            "samples": InterruptedSamples()}, output)
    result = json.loads(output.read_text())
    assert result["stop_reason"] == "leg_timeout"
    assert result["sample_storage"]["sample_archive_status"] == "INTERRUPTED"
    assert result["sample_storage"]["sample_count"] == 2
    with gzip.open(result["sample_storage"]["sample_archive_path"], "rt") as archive:
        assert [json.loads(line) for line in archive] == [{"sequence": 1}]


def test_primary_result_serialization_failure_does_not_damage_previous_checkpoint(tmp_path, monkeypatch):
    output = tmp_path / "owner.json"
    probe._write_result_json({"status": "ERROR", "stop_reason": "original"}, output)

    def interrupted_dump(_payload, stream, **kwargs):
        stream.write('{"incomplete":')
        raise ValueError("synthetic serialization failure")

    monkeypatch.setattr(probe.json, "dump", interrupted_dump)
    with pytest.raises(ValueError, match="synthetic serialization failure"):
        probe._write_result_json({"status": "OK"}, output)
    assert json.loads(output.read_text()) == {"status": "ERROR", "stop_reason": "original"}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["owner.json"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_automatic_report_publication_never_overwrites_or_exposes_partial_json(tmp_path, monkeypatch, interrupted):
    from control_station.b601_torque_console import automatic_report

    monkeypatch.setattr(automatic_report, "build_automatic_report", lambda run: {"status": "INCOMPLETE"})
    output = tmp_path / "automatic_calibration_report.json"
    if interrupted:
        def interrupted_dump(_payload, stream, **kwargs):
            stream.write('{"partial":')
            raise KeyboardInterrupt

        monkeypatch.setattr(automatic_report.json, "dump", interrupted_dump)
        with pytest.raises(KeyboardInterrupt):
            automatic_report.save_automatic_report(tmp_path)
        assert list(tmp_path.iterdir()) == []
    else:
        assert automatic_report.save_automatic_report(tmp_path) == output
        expected = output.read_bytes()
        with pytest.raises(FileExistsError):
            automatic_report.save_automatic_report(tmp_path)
        assert output.read_bytes() == expected
        assert [p.name for p in tmp_path.iterdir()] == [output.name]


@pytest.fixture
def production_controller(tmp_path):
    profile = Path(__file__).resolve().parents[1] / "docs/hardware/profiles/b601_j2_j3_workspace_calibration_v1.json"
    args = owner._parser().parse_args([
        "--allow-motion", "--torque-console-auto-calibration-profile", str(profile),
        "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE", "--rate-hz", "100",
        "--torque-console-joint", "joint3", "--torque-console-max-assist-nm", "1",
        "--torque-console-max-joint-velocity-rad-s", ".195", "--torque-console-max-excursion-rad", ".1",
        "--torque-console-staged-target-offset-limit-rad", ".09",
        "--torque-console-staged-breakaway-torque-nm", "1",
        "--torque-console-staged-breakaway-rise-slew-nm-s", "1",
        "--torque-console-staged-breakaway-duration-sec", "3",
        "--torque-console-staged-breakaway-impulse-nm-sec", "1.5",
        "--torque-console-staged-tracking-overlay-cap-nm", "1",
    ])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
    probe._validate_cli(parsed)
    return probe._automatic_calibration_controller_from_args(parsed)


def test_workspace_profile_is_available_without_changing_legacy_default_mode(production_controller):
    controller = production_controller
    selection = controller.status()["mode_selection"]
    assert selection["selected"] == "fixed_position"
    assert selection["available_modes"] == ["fixed_position", "supported_cycle", "workspace"]
    controller.validate_position_limits(LIMITS)
    with pytest.raises(ValueError, match="pinned owner"):
        controller.validate_position_limits(((0, 1),) * 6)


@pytest.mark.parametrize("joint", ["joint2", "joint3"])
@pytest.mark.parametrize("positioning", [False, True])
def test_workspace_production_profile_constructs_every_child_before_motion(
    production_controller, joint, positioning,
):
    workspace = production_controller._workspace
    child = workspace._make_controller(joint, positioning=positioning)
    staged = child.settings.staged_assist
    assert staged.joint_name == joint
    assert child.limits.joint_name == joint
    assert staged.breakaway_probe is not None
    assert staged.motion_control.position_resolution_rad == .0004
    assert staged.tracking_recovery_gain_nm_per_rad == 20
    assert staged.onset_handoff == "track"
    assert staged.max_total_overlay_nm == 1
    assert child.limits.max_joint_velocity_rad_s == .195
    assert staged.motion_control.max_reference_lag_rad == .0195
    assert staged.motion_control.soft_velocity_rad_s == .0585
    assert staged.motion_control.continuous_speed_limit_rad_s == .0585
    assert staged.motion_control.max_control_gap_sec == .04
    assert staged.settle_error_rad == .01352
    # The wrapper's cycle section does not configure workspace children.
    # Recovery must be explicit for positioning, never silently for sweeps.
    assert staged.motion_control.support_soft_speed_recovery is positioning


def test_workspace_positioning_support_brake_resumes_same_leg_only_after_zero_and_still(production_controller):
    controller = production_controller
    controller.submit("select_mode", now_sec=1., operation_mode="workspace")
    replay = AcquisitionReplay(controller.settings, controller=controller,
        position_limits=LIMITS, initial_q=ZERO)
    controller.submit("arm", now_sec=replay.now+.001, operation_mode="workspace")
    advance(replay, until=lambda b: b["state"] == "position" and replay.audit.get("phase") == "track")
    target = replay.audit["motion_control"]["target_position_rad"]
    session = controller._workspace._session
    batch_id = controller._workspace._batch_id
    # Prescribed support motion is below the raw hard limit, but crosses the
    # existing causal-window soft limit. It must withdraw before any resumption.
    for _ in range(9):
        q = list(replay.q)
        q[3] += .0006
        replay.tick(q)
    assert replay.audit["motion_control"]["reason"] == "support_velocity_braking"
    for _ in range(8):
        replay.tick()
        assert replay.audit["phase"] != "track"
    for _ in range(200):
        replay.tick()
        if replay.audit["phase"] == "track":
            break
    assert replay.audit["phase"] == "track", replay.audit
    assert replay.audit["motion_control"]["speed_governor_resume_count"] == 1
    assert replay.audit["motion_control"]["target_position_rad"] == target
    assert controller._workspace._session is session
    assert controller._workspace._batch_id == batch_id
    assert controller.status()["automatic_calibration"]["workspace"]["task_index"] == 0
    # An explicit operator cancel cannot be undone by the next quiet window.
    controller.submit("withdraw", now_sec=replay.now+.001)
    for _ in range(150):
        replay.tick()
        assert replay.audit["phase"] != "track"
    assert controller.result_record()["state"] == "cancelled"


def test_workspace_production_profile_checks_both_axes_at_construction(production_controller, monkeypatch):
    workspace = production_controller._workspace
    original = B601WorkspaceCalibrationController._make_controller
    checked = []

    def build(self, joint, *, positioning):
        checked.append((joint, positioning))
        return original(self, joint, positioning=positioning)

    monkeypatch.setattr(B601WorkspaceCalibrationController, "_make_controller", build)
    candidate = B601WorkspaceCalibrationController(workspace.settings, workspace.config, workspace.workspace)
    assert set(checked) == {(j, p) for j in ("joint2", "joint3") for p in (False, True)}
    assert candidate.status()["automatic_calibration"]["state"] == "idle"


@pytest.mark.parametrize("joint", ["joint1", "joint4", "joint5", "joint6"])
def test_workspace_held_probe_does_not_open_other_axes(production_controller, joint):
    with pytest.raises(ValueError, match="explicit resolved J2/J3"):
        replace(production_controller._workspace.settings.staged_assist, joint_name=joint)


def test_workspace_production_profile_scripted_feedback_crosses_j3_j2_and_sweeps(production_controller):
    # This prescribes encoder motion; it proves lifecycle wiring, not plant dynamics.
    controller = production_controller
    controller.submit("select_mode", now_sec=1.0, operation_mode="workspace")
    replay = AcquisitionReplay(controller.settings, controller=controller,
        position_limits=LIMITS, initial_q=ZERO)
    controller.submit("arm", now_sec=replay.now + .001, operation_mode="workspace")
    advance(replay)
    result = controller.result_record()
    assert result["state"] == "complete", result["reason"]
    assert len(result["motion"]["completed_legs"]) == 96
    assert result["workspace"]["completed_joint_acquisitions"] == 8
    assert {item["task"]["joint"] for item in result["workspace"]["moves"]} == {"joint2", "joint3"}
    assert controller.cycle_disable_confirmation_ready


def test_gui_supported_return_requires_an_explicit_single_round_grant():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for GUI contracts")
    source = Path("control_station/b601_torque_console/static/app.js").read_text()
    handler = source[source.index('el("arm").onclick ='):source.index('el("mode-fixed-position").onclick =')]
    script = """
const vm=require('node:vm');const nodes={arm:{},'supported-return':{checked:false}};
const calls=[];let confirms=0;let accepted=false;
const context={el:id=>nodes[id],state:{mode_selection:{selected:'workspace',supported_return_available:true}},
  intent:(action,fields)=>calls.push({action,...fields}),window:{confirm:()=>{confirms++;return accepted;}}};
vm.runInNewContext(JSON.parse(process.argv[1]),context);
nodes.arm.onclick();
nodes['supported-return'].checked=true;nodes.arm.onclick();
accepted=true;nodes.arm.onclick();
context.state.mode_selection.supported_return_available=false;nodes.arm.onclick();
console.log(JSON.stringify({calls,confirms}));
"""
    completed = subprocess.run([node, "-e", script, json.dumps(handler)],
        check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert result["confirms"] == 2
    assert result["calls"] == [
        {"action": "arm", "operation_mode": "workspace"},
        {"action": "arm", "operation_mode": "workspace", "return_and_disable": True},
        {"action": "arm", "operation_mode": "workspace"},
    ]


def test_gui_manual_return_confirmation_uses_owner_readiness_not_acquisition_success():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for GUI contracts")
    source = Path("control_station/b601_torque_console/static/app.js").read_text()
    handler = source[source.index('el("full-disable").onclick ='):source.index('document.addEventListener("visibilitychange"')]
    script = """
const vm=require('node:vm');const nodes={'full-disable':{}};const calls=[];
const cycle={phase:'failed',disable_confirmation_available:true,zero_assist_base_damping_confirmed:true};
const c={el:id=>nodes[id],state:{automatic_calibration:{cycle}},window:{confirm:()=>true},
  post:(path,body)=>calls.push({path,...body})};
vm.runInNewContext(JSON.parse(process.argv[1]),c);
nodes['full-disable'].onclick();cycle.disable_confirmation_available=false;nodes['full-disable'].onclick();
console.log(JSON.stringify(calls));
"""
    result = subprocess.run([node, "-e", script, json.dumps(handler)],
        check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [
        {"path": "/api/stop", "action": "confirm_cycle_full_disable"},
        {"path": "/api/stop", "action": "full_disable"},
    ]


def test_workspace_gui_start_stop_and_background_semantics():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for GUI contracts")
    source = Path("control_station/b601_torque_console/static/app.js").read_text()
    functions = source[source.index("function number("):source.index("function post(")]
    script = """
const vm = require('node:vm'); const elements = {};
const el = id => elements[id] ||= {value:'20',setAttribute(k,v){this[k]=v;}};
const selection = {selected:'workspace',available_modes:['fixed_position','supported_cycle','workspace'],ready:true,locked:false,
  supported_return_available:true,supported_return_configured:true};
const batch = {state:'idle',active:false,start_requested:false,human_intervention:false,
  workspace:{phase:'idle',reason:'waiting',task_count:48,task_index:0,completed_joint_acquisitions:0,
    expected_joint_acquisitions:18,pose_centres_rad:[[.1,.1]],setup_label:'test',supported_return:{manual_available:true,manual_gate_reason:'ready',supported_origin_qpos_rad:[0,0,0,0,0,0]}},cycle:{}};
const c = {el,state:{controller_mode:'automatic_calibration',mode_selection:selection,
  owner_ready:true,feedback_age_sec:0,stop_controls_available:true,transparency_assessment:{available:true},automatic_calibration:batch},
  cycleAngleInitialized:false};
vm.runInNewContext(JSON.parse(process.argv[1]),c);
const render = () => {c.renderAutomatic(); return {arm:el('arm').disabled,joint:el('joint-select').disabled,
  targetHidden:el('target').hidden,authorizeHidden:el('authorize-breakaway').hidden,
  confirm:el('cycle-disable-hint').hidden,background:c.isSupportedCycle()};};
const idle = render();
const returnIdle={panelHidden:el('supported-return-panel').hidden,buttonDisabled:el('return-to-origin').disabled};
selection.selected='fixed_position';selection.supported_return_available=false;render();
const returnFixed={panelHidden:el('supported-return-panel').hidden,buttonDisabled:el('return-to-origin').disabled};
selection.selected='workspace';selection.supported_return_available=true;
batch.state='sweep';batch.active=true;batch.workspace.phase='sweep';selection.locked=true;
const active = render();
batch.workspace.coordinated_holding_enabled=true;
batch.workspace.leg_command_exposure_nm_sec=16.1;
batch.workspace.max_sweep_leg_command_exposure_nm_sec=24;
batch.workspace.max_sweep_leg_sec=58.5;
batch.workspace.arrival={reason:'controlled_target_error',tolerance_rad:.0015,controlled_error_rad:[0,.002]};
render();const arrivalWaiting=el('automatic-progress').textContent;
batch.workspace.arrival.reason='ready';render();const arrivalReady=el('automatic-progress').textContent;
batch.workspace.phase='observe';
batch.workspace.static_observation={active:true,elapsed_sec:2.45,required_dwell_sec:5,
  position_span_rad:[.0004,.0004,.0008,0,.0004,0],allowed_position_span_rad:.0045,reason:'collecting_window'};
batch.workspace.arrival.reason='six_axis_not_stationary';
render();const staticWindow=el('automatic-progress').textContent;
batch.workspace.phase='sweep';delete batch.workspace.static_observation;
batch.workspace.holding_calibration={enabled:true,context_reason:'support_pose_changed'};
render();const contextChanged=el('automatic-progress').textContent;
batch.workspace.holding_calibration={enabled:true,nominal_target_rad:[0,.04235987756,.05235987756,0,0,0],
  search:{state:'searching',interior_recheck_start_observation:6,rejected_stationary_commands_nm:[-.14209]}};
render();const recheck=el('automatic-progress').textContent;const rejected=el('automatic-trials').textContent;
batch.workspace.holding_calibration.search.state='complete';render();const rechecked=el('automatic-progress').textContent;
batch.workspace.coordinated_holding_enabled=false;delete batch.workspace.holding_calibration;
batch.state='complete';batch.active=false;batch.workspace.phase='complete';batch.cycle.disable_confirmation_available=true;
const complete = render();
batch.state='failed';batch.workspace.phase='failed';batch.cycle.disable_confirmation_available=false;
batch.outcome={severity:'warning'};render();const warning=el('phase').textContent;
const warningInstructions=el('drive-instructions').textContent;
batch.outcome={severity:'error'};render();const fault=el('phase').textContent;
const faultInstructions=el('drive-instructions').textContent;
batch.state='cancelled';batch.workspace.phase='cancelled';
batch.workspace.supported_return.acquisition_stop={state:'cancelled',reason:'operator_return'};
batch.cycle.disable_confirmation_available=true;batch.cycle.zero_assist_base_damping_confirmed=true;
render();const returned=el('phase').textContent;const returnInstructions=el('drive-instructions').textContent;
batch.state='idle';batch.workspace.phase='idle';batch.human_intervention=true;
console.log(JSON.stringify({idle,returnIdle,returnFixed,active,complete,warning,fault,warningInstructions,faultInstructions,returned,returnInstructions,arrivalWaiting,arrivalReady,staticWindow,contextChanged,recheck,rejected,rechecked,human:render(),stopLabel:el('full-disable').textContent}));
"""
    result = subprocess.run([node, "-e", script, json.dumps(functions)],
        check=True, capture_output=True, text=True)
    states = json.loads(result.stdout)
    assert states["returnIdle"] == {"panelHidden": False, "buttonDisabled": False}
    assert states["returnFixed"] == {"panelHidden": False, "buttonDisabled": True}
    assert "原预算内重新复核" in states["recheck"] and "2.427°" in states["recheck"]
    assert "已排除的旧维持候选" in states["rejected"]
    assert "已完成内部维持命令重新复核" in states["rechecked"]
    assert states["idle"] == {"arm": False, "joint": True, "targetHidden": True,
        "authorizeHidden": True, "confirm": True, "background": True}
    assert states["active"]["arm"] is True
    assert states["complete"]["confirm"] is False
    assert states["human"]["arm"] is True
    assert "告警，数据保留" in states["warning"]
    assert "采集故障" in states["fault"]
    assert "当前仅重力补偿" in states["warningInstructions"]
    assert "不会自动重试" in states["warningInstructions"]
    assert "停止采集不等于全失能" in states["faultInstructions"]
    assert states["returned"] == "回位完成 · 等待人工确认失能"
    assert "回位完成不等于采集完成" in states["returnInstructions"]
    assert "等待 J2/J3 到位" in states["arrivalWaiting"]
    assert "16.10 / 24.00 N·m·s" in states["arrivalWaiting"]
    assert "单段时限 58.5 秒" in states["arrivalWaiting"]
    assert "不重置本段预算" in states["arrivalWaiting"]
    assert "J2/J3 已到位，六轴已停稳" in states["arrivalReady"]
    assert "静态观察 2.45 / 5.00 秒" in states["staticWindow"]
    assert "0.00080 / 0.00450 rad" in states["staticWindow"]
    assert "带内微小抖动不重置计时" in states["staticWindow"]
    assert "等待六轴停稳" not in states["staticWindow"]
    assert "不自动重锚或重试" in states["contextChanged"]
