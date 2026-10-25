"""Frozen source and native command composition; no device or measured force."""

import copy
import hashlib
import json
import math
from dataclasses import asdict, replace

import pytest
from test_b601_automatic_calibration import AcquisitionReplay
from test_b601_coordinated_holding import scripted_coordinated_tick
from test_b601_cruise_velocity import (
    FOUR_CENTRES,
    controller_from_profile,
    simulate_motion,
    warm_core,
)
from test_b601_workspace_calibration import LIMITS
from test_b601_workspace_identification import field_observations

from robot_motion.rebot_b601_rs.command_field_source import CommandFieldSource
from robot_motion.rebot_b601_rs.coordinated_holding import (
    AXES,
    ZERO,
    CoordinatedPoseHolding,
)
from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionFrame

pytestmark = [pytest.mark.unit, pytest.mark.contract]
KD = (.4, 4.5, 4.5, 1.2, .2, .1)
NEAR_ZERO = FOUR_CENTRES.with_name("b601_j2_j3_near_zero_command_field_v1.json")


def source_report():
    training, validation = field_observations()
    windows = []
    for role, rows in (("training", training), ("validation", validation)):
        for sample in rows:
            sample = replace(sample, joint_name="joint3", qpos_rad=(0, .03, sample.qpos_rad[1], 0, 0, 0),
                active_driver_kd_nm_s_per_rad=KD)
            windows.append({"pose_index": 0, "role": role, "model_eligible": True, "native_phase_verified": True,
                "observation": {k: v for k, v in asdict(sample).items() if k != "condition"}})
    return {"schema": "b601.workspace-command-identification.v1", "reduction_method": "closed_loop_average",
        "auto_apply": False, "source_batch_state": "failed", "source_trace_sha256": "c"*64,
        "source_report_sha256": "d"*64, "condition": asdict(training[0].condition),
        "command_field_context": {"setup_label": "synthetic", "pose_centres_rad": [[.03, .03]]}, "windows": windows}


def source():
    return CommandFieldSource(source_report(), source_sha256="e"*64, source_path="synthetic.json", total_assist_cap_nm=1)


def test_source_is_frozen_training_only_and_requires_native_provenance(tmp_path):
    report = source_report()
    path = tmp_path/"source.json"
    raw = json.dumps(report).encode()
    path.write_bytes(raw)
    bank = CommandFieldSource.load(path, total_assist_cap_nm=1)
    assert bank.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert bank.metadata()["regions"] == 1
    assert bank.validate_context(replace(bank.condition, run_id="new-run"), setup_label="synthetic",
        pose_centres_rad=((.03, .03),)) == (0,)
    request = {"pose_index": 0, "joint_name": "joint3", "qpos_rad": (0, .03, .02, 0, 0, 0),
        "velocity_rad_s": .01, "active_driver_kd_nm_s_per_rad": KD}
    assert bank.predict(**request)["predicted_command_residual_nm"] == pytest.approx(.4)
    path.write_text("{}")
    assert bank.predict(**request)["predicted_command_residual_nm"] == pytest.approx(.4)
    assert not bank.predict(**{**request, "velocity_rad_s": .1})["predicted"]
    assert not bank.predict(**{**request, "active_driver_kd_nm_s_per_rad": (.4, .6, .4, .6, .2, .1)})["predicted"]
    legacy = copy.deepcopy(report)
    for window in legacy["windows"]:
        window.pop("native_phase_verified")
    with pytest.raises(ValueError, match="native-track"):
        CommandFieldSource(legacy, source_sha256="e"*64, source_path="legacy", total_assist_cap_nm=1)
    assert report["source_batch_state"] == "failed"  # Later stopping does not erase good intervals.


@pytest.mark.parametrize("fault", ["label", "gravity", "centres", "overcap", "leak", "window"])
def test_source_rejects_changed_conditions_bad_data_and_leaked_trials(fault):
    bank = source()
    report = source_report()
    if fault == "label":
        with pytest.raises(ValueError, match="setup label"):
            bank.validate_context(bank.condition, setup_label="different", pose_centres_rad=((.03, .03),))
        return
    if fault == "gravity":
        with pytest.raises(ValueError, match="condition mismatch"):
            bank.validate_context(replace(bank.condition, gravity_scale=.9), setup_label="synthetic", pose_centres_rad=((.03, .03),))
        return
    if fault == "centres":
        with pytest.raises(ValueError, match="pose centre"):
            bank.validate_context(bank.condition, setup_label="synthetic", pose_centres_rad=((.03, .09),))
        return
    if fault == "overcap":
        report["windows"][0]["observation"]["command_residual_nm"] = 1.1
    elif fault == "leak":
        report["windows"][-1]["observation"]["trial_id"] = report["windows"][0]["observation"]["trial_id"]
    else:
        report["windows"] = [None]
    with pytest.raises(TypeError if fault == "window" else ValueError):
        CommandFieldSource(report, source_sha256="e"*64, source_path="bad", total_assist_cap_nm=1)


def core_fixture(tmp_path, monkeypatch, mode):
    original = controller_from_profile(tmp_path)._workspace._coordinated.pose
    core = CoordinatedPoseHolding(original.settings, replace(original.config, command_field_mode=mode),
        command_field_source=source() if mode != "off" else None)
    warm_core(core, q=(0, .03, .02, 0, 0, 0))
    for tracker, i in zip(core.trackers, AXES):
        monkeypatch.setattr(tracker, "update", lambda i=i, **kwargs: B601JointMotionFrame(
            phase="track", reason="bounded_reference_tracking", q_reference_rad=core.q[i],
            qvel_reference_rad_s=.01 if i == 2 else 0, control_qvel_rad_s=0,
            position_torque_nm=0, velocity_torque_nm=0, requested_torque_nm=0))
    monkeypatch.setattr(core, "_approach_bias", lambda i, frame, dt, moving, anchor, **kwargs:
        (core.bias[i], True, anchor, "test_retains_support"))
    monkeypatch.setattr(core, "_cruise_feedback", lambda i, frame, dt:
        (0, core.bias[i], {"active": True, "reason": "test_resolved_window", "aligned_reference_velocity_rad_s": .01}))
    core.adopt_hold(1, .2)
    core.adopt_hold(2, .3)
    for n in range(35):
        dt = core.observe(core.last_time+.01, core.q, ZERO)
        cmd, _ = core.step(sequence=100+n, dt=dt, permitted=True, point_search=False, cruise_joint=2)
        core.record_dispatch(100+n, cmd)
    assert core.last_delta[1:3] == pytest.approx((.2, .3))
    return core


def tick(core, sequence):
    dt = core.observe(core.last_time+.01, core.q, ZERO)
    core.cruise_velocity = (0, 0, .01, 0, 0, 0)
    return core.step(sequence=sequence, dt=dt, permitted=True, point_search=False, cruise_joint=2,
        field_pose_index=0, field_driver_kd=KD)


def test_shadow_is_command_equivalent_and_candidate_engagement_never_adds_absolute_model(tmp_path, monkeypatch):
    baseline = core_fixture(tmp_path, monkeypatch, "off")
    shadow = core_fixture(tmp_path, monkeypatch, "shadow")
    candidate = core_fixture(tmp_path, monkeypatch, "candidate")
    for n in range(10):
        before = candidate.bias
        a, _ = tick(baseline, 200+n)
        b, _ = tick(shadow, 200+n)
        c, audit = tick(candidate, 200+n)
        assert a == b == c
        assert candidate.bias == before
        assert audit["command_field"]["requested_increment_nm"] == 0
        assert audit["command_field"]["prediction_nm"] == pytest.approx(.4)
        for core, command in ((baseline, a), (shadow, b), (candidate, c)):
            core.record_dispatch(200+n, command)
    assert candidate.field_component[2] == pytest.approx(.4)
    assert candidate.bias[2]-candidate.field_component[2] == pytest.approx(-.1)
    assert shadow.field_component == ZERO


@pytest.mark.parametrize("matching", [False, True])
def test_model_increments_are_slew_bounded_and_only_exact_receipts_commit(tmp_path, monkeypatch, matching):
    core = core_fixture(tmp_path, monkeypatch, "candidate")
    cmd, _ = tick(core, 200)
    core.record_dispatch(200, cmd)
    before = core.bias
    monkeypatch.setattr(core.command_field_source, "predict", lambda **kwargs:
        {"predicted": True, "reason": "test_field_change", "predicted_command_residual_nm": .7})
    command, audit = tick(core, 201)
    assert audit["command_field"]["requested_increment_nm"] == pytest.approx(.0225)
    assert core.bias == before and core.field_component[2] == pytest.approx(.4)
    actual = command if matching else tuple(v*.99 for v in command)
    core.record_dispatch(201, actual)
    if matching:
        assert core.field_component[2] == pytest.approx(.4225)
        assert core.bias[2] == pytest.approx(before[2]+.0225)
    else:
        assert core.bias == before and core.field_component == ZERO
    assert sum(abs(v) for v in command) <= 1


def test_field_queries_reference_speed_and_handles_zero_model_and_axis_handoffs(tmp_path, monkeypatch):
    core = core_fixture(tmp_path, monkeypatch, "candidate")
    seen = []
    target = 0.0

    def predict(**kwargs):
        seen.append(kwargs["velocity_rad_s"])
        return {"predicted": True, "reason": "test_zero_field", "predicted_command_residual_nm": target}

    monkeypatch.setattr(core.command_field_source, "predict", predict)
    command, _ = tick(core, 200)
    core.record_dispatch(200, command)
    assert core._field_joint == 2 and core.field_component == ZERO
    target = .4
    command, audit = tick(core, 201)
    assert audit["command_field"]["requested_increment_nm"] == pytest.approx(.0225)
    core.record_dispatch(201, command)
    core.cruise_velocity = (0, .02, .02, 0, 0, 0)
    command, audit = core.step(sequence=202, dt=.01, permitted=True, point_search=False, cruise_joint=1,
        field_pose_index=0, field_driver_kd=KD)
    assert audit["command_field"]["requested_increment_nm"] == 0
    core.record_dispatch(202, command)
    assert seen == [.01, .01, .01]  # Actual overspeed is never queried as desired speed.
    assert core._field_joint == 1 and core.field_component[2] == 0


def test_coverage_exit_goal_handoff_and_revocation_preserve_or_clear_the_right_state(tmp_path, monkeypatch):
    core = core_fixture(tmp_path, monkeypatch, "candidate")
    command, _ = tick(core, 200)
    core.record_dispatch(200, command)
    before = core.bias
    monkeypatch.setattr(core.command_field_source, "predict", lambda **kwargs:
        {"predicted": False, "reason": "outside_validated_query_neighbourhood"})
    command, audit = tick(core, 201)
    assert audit["command_field"]["reason"] == "outside_validated_query_neighbourhood"
    core.record_dispatch(201, command)
    assert core.bias == before and core.field_component == ZERO
    core.set_goal((0, .03, .04, 0, 0, 0))
    assert core.bias == before and core.field_component == ZERO
    command, _ = core.step(sequence=202, dt=.01, permitted=False, point_search=False)
    core.record_dispatch(202, command)
    assert core.bias == core.last_delta == core.field_component == ZERO


def test_genuine_native_braking_never_queries_the_field(tmp_path, monkeypatch):
    core = core_fixture(tmp_path, monkeypatch, "candidate")
    command, _ = tick(core, 200)
    core.record_dispatch(200, command)

    def forbidden(**kwargs):
        pytest.fail("native braking queried the command model")

    monkeypatch.setattr(core.command_field_source, "predict", forbidden)
    monkeypatch.setattr(core.trackers[1], "update", lambda **kwargs: B601JointMotionFrame(
        phase="brake", reason="test_native_braking", q_reference_rad=core.q[2], qvel_reference_rad_s=0,
        control_qvel_rad_s=0, position_torque_nm=0, velocity_torque_nm=0, requested_torque_nm=0))
    command, audit = tick(core, 201)
    assert audit["braking"] and not audit["command_field"]["predicted"]
    core.record_dispatch(201, command)
    assert core.field_component == ZERO and core._field_joint is None


def test_withdrawal_invalidates_model_even_with_unmatched_conditioner_tail(tmp_path, monkeypatch):
    core = core_fixture(tmp_path, monkeypatch, "candidate")
    command, _ = tick(core, 200)
    core.record_dispatch(200, command)
    command, _ = core.step(sequence=201, dt=.01, permitted=False, point_search=False)
    core.record_dispatch(201, (0, 0, .001, 0, 0, 0))
    assert core.field_component == ZERO and core._field_joint is None
    command, audit = core.step(sequence=202, dt=.01, permitted=False, point_search=False)
    assert command == ZERO and not audit["command_field"]["predicted"]
    core.record_dispatch(202, command)
    assert core.bias == ZERO


def test_source_mode_and_report_are_both_required_before_backend_creation(tmp_path):
    for extra in (("--torque-console-command-field-mode", "shadow"),
                  ("--torque-console-command-field-report", str(tmp_path/"absent.json"))):
        with pytest.raises(ValueError, match="both an explicit mode"):
            controller_from_profile(tmp_path, extra=extra)


def test_candidate_preserves_raw_velocity_watchdog_and_combined_caps(tmp_path, monkeypatch):
    core = core_fixture(tmp_path, monkeypatch, "candidate")
    core.raw = (0, core.settings.limits.max_joint_velocity_rad_s*1.01, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="raw joint velocity"):
        core.step(sequence=200, dt=.01, permitted=True, point_search=False, cruise_joint=2,
            field_pose_index=0, field_driver_kd=KD)
    core.raw = ZERO
    with pytest.raises(ValueError, match="watchdog"):
        core.observe(core.last_time+.041, core.q, ZERO)
    targets = iter([-.8]+[.8]*100)
    monkeypatch.setattr(core.command_field_source, "predict", lambda **kwargs:
        {"predicted": True, "reason": "test_cap", "predicted_command_residual_nm": next(targets)})
    core.adopt_hold(2, .8)
    for n in range(100):
        command, _audit = tick(core, 300+n)
        assert sum(abs(v) for v in command) <= 1+1e-12
        core.record_dispatch(300+n, command)
        assert max(abs(v) for v in core.bias) <= core.config.bias_cap_nm
        assert max(abs(v) for v in core.field_component) <= 1


def test_small_acceptance_profile_changes_only_centres_and_uses_existing_owner_flags(tmp_path):
    old, new = [json.loads(path.read_text()) for path in (FOUR_CENTRES, NEAR_ZERO)]
    for name in ("joint2_centres_deg", "joint3_centres_deg"):
        assert new["workspace"][name] == [3]
        new["workspace"][name] = old["workspace"][name]
    assert new == old
    report = source_report()
    report["command_field_context"] = {
        "setup_label": old["workspace"]["setup_label"], "pose_centres_rad": [[math.radians(3)]*2]}
    path = tmp_path/"source.json"
    path.write_text(json.dumps(report))
    controller = controller_from_profile(tmp_path, profile=NEAR_ZERO, extra=(
        "--torque-console-command-field-report", str(path), "--torque-console-command-field-mode", "shadow"))
    controller.configure_context(source().condition, LIMITS)
    core = controller._workspace._coordinated
    assert core.pose.config.command_field_mode == "shadow"
    assert core._field_pose_indices == (0,)
    assert controller.command_field_source_metadata["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert core.workspace.coordinated_holding.total_assist_cap_nm == 1
    assert core.pose.motion.max_control_gap_sec == .04


def test_owner_workspace_source_reaches_all_28_legs_and_retains_disable_confirmation(tmp_path):
    report = source_report()
    profile = json.loads(NEAR_ZERO.read_text())
    report["command_field_context"] = {"setup_label": profile["workspace"]["setup_label"],
        "pose_centres_rad": [[math.radians(3)]*2]}
    path = tmp_path/"source.json"
    path.write_text(json.dumps(report))
    whole = controller_from_profile(tmp_path, profile=NEAR_ZERO, extra=(
        "--torque-console-command-field-report", str(path), "--torque-console-command-field-mode", "shadow"))
    workspace = whole._workspace
    replay = AcquisitionReplay(workspace.settings, controller=workspace, position_limits=LIMITS)
    replay.controller.submit("arm", now_sec=replay.now)
    seen_field = False
    for _ in range(20000):
        scripted_coordinated_tick(replay, passive_target=0)
        field = replay.audit.get("coordinated_holding", {}).get("command_field")
        if field:
            seen_field = True
            assert field["mode"] == "shadow" and field["requested_increment_nm"] == 0
        if workspace.result_record()["state"] in {"complete", "failed", "cancelled"}:
            break
    result = workspace.result_record()
    assert result["state"] == "complete", result["reason"]
    assert len(result["motion"]["completed_legs"]) == 28
    assert seen_field and workspace.cycle_disable_confirmation_ready
    assert not workspace.automatic_disable_ready
    assert result["workspace"]["supported_return"]["route_completed"]


def plant_source():
    """Known command field of the existing toy plant, not B601 observations."""
    report = source_report()
    base = report["windows"][0]["observation"]
    centre = math.radians(5)
    report["command_field_context"]["pose_centres_rad"] = [[centre, centre]]
    windows = []
    for k in range(19):
        q = (0, centre, .04+k*.005, 0, 0, 0)
        for speed in (.0075, .015):
            for direction in (-1, 1):
                for repeat in range(3):
                    observed = {**base, "trial_id": f"toy:{speed}:{direction}:{repeat}", "qpos_rad": q,
                        "velocity_rad_s": speed*direction,
                        "command_residual_nm": .08+direction*(.10+.02*math.cos(45*q[2]))+4.5*speed*direction}
                    windows.append({"pose_index": 0, "role": "validation" if repeat == 2 else "training",
                        "model_eligible": True, "native_phase_verified": True, "observation": observed})
    report["windows"] = windows
    return CommandFieldSource(report, source_sha256="e"*64, source_path="declared-toy", total_assist_cap_nm=1)


@pytest.mark.parametrize("direction,inertia,speed", [(1, .08, .0075), (-1, .2, .0075), (1, .2, .015), (-1, .6, .015)])
def test_native_closed_loop_field_comparison(tmp_path, direction, inertia, speed):
    bank = plant_source()
    options = {"candidate": True, "direction": direction, "inertia": inertia, "speed": speed, "arrival": True}
    baseline = simulate_motion(tmp_path, **options)
    shadow = simulate_motion(tmp_path, **options, field_mode="shadow", field_source=bank)
    candidate = simulate_motion(tmp_path, **options, field_mode="candidate", field_source=bank)
    print(json.dumps({"declared_plant_only": True, "baseline": baseline, "shadow": shadow, "candidate": candidate}))
    for result in (baseline, shadow, candidate):
        assert result["outcome"] == "arrived", result
        assert result["peak_speed_rad_s"] < .195
        assert abs(result["final_error_rad"]) <= .003375
    for key in ("duration_sec", "final_error_rad", "rms_speed_error_rad_s", "peak_speed_rad_s"):
        assert shadow[key] == baseline[key]
    assert candidate["field_predicted_cycles"] > 0 and candidate["field_increment_cycles"] > 0
    assert candidate["rms_speed_error_rad_s"] < baseline["rms_speed_error_rad_s"]
