from __future__ import annotations

import os
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_URDF_RELATIVE_PATH,
    B601PolicyAssistCommand,
    B601SharedControlConfig,
    B601StagedAssistConfig,
    B601StagedAssistPhase,
    B601StagedSingleJointAssistRuntime,
)
from robot_motion.rebot_b601_rs.joint_motion import (
    B601JointMotionConfig,
    B601JointMotionFrame,
    B601WindowStopEvidence,
    B601WindowVelocityEstimator,
)
from robot_motion.rebot_b601_rs.torque_experiment import B601OnsetProbeConfig

ZERO = (0.0,) * 6


def _shared_config() -> B601SharedControlConfig:
    return B601SharedControlConfig(
        position_kp_nm_per_rad=(0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        velocity_kd_nm_s_per_rad=ZERO,
        max_target_error_rad=(0.2,) * 6,
        max_abs_assist_torque_nm=(0.3,) * 6,
        max_assist_torque_rate_nm_s=(10.0,) * 6,
        authority_rise_per_sec=100.0,
        policy_stale_timeout_sec=1.0,
    )


def _staged_config(**overrides: object) -> B601StagedAssistConfig:
    values: dict[str, object] = {
        "joint_name": "joint3",
        "cue_torque_nm": 0.04,
        "cue_duration_sec": 0.04,
        "cue_rise_slew_nm_s": 2.0,
        "breakaway_torque_nm": 0.12,
        "breakaway_rise_slew_nm_s": 4.0,
        "breakaway_max_duration_sec": 0.2,
        "breakaway_max_impulse_nm_sec": 0.015,
        "minimum_breakaway_error_rad": 0.02,
        "stuck_max_velocity_rad_s": 0.01,
        "motion_onset_displacement_rad": 0.001,
        "motion_onset_velocity_rad_s": 0.02,
        "motion_onset_confirm_samples": 2,
        "new_intent_delta_rad": 0.03,
        "tracking_overlay_cap_nm": 0.2,
        "tracking_slew_nm_s": 2.0,
        "settle_error_rad": 0.004,
        "settle_capture_error_rad": 0.02,
        "settle_velocity_rad_s": 0.01,
        "settle_dwell_sec": 0.06,
        "settle_integral_gain_nm_per_rad_sec": 1.0,
        "settle_integral_leak_per_sec": 0.1,
        "settle_integral_cap_nm": 0.02,
        "settle_overlay_cap_nm": 0.05,
        "max_total_overlay_nm": 0.2,
    }
    values.update(overrides)
    return B601StagedAssistConfig(**values)  # type: ignore[arg-type]


def _command(sequence: int, timestamp: float, target: float = 0.1) -> B601PolicyAssistCommand:
    qpos = (0.0, 0.0, target, 0.0, 0.0, 0.0)
    return B601PolicyAssistCommand(
        sequence=sequence,
        source_timestamp_sec=timestamp,
        receive_timestamp_sec=timestamp,
        qpos_target_rad=qpos,
    )


def _step(
    runtime: B601StagedSingleJointAssistRuntime,
    *,
    sequence: int,
    timestamp: float,
    q3: float = 0.0,
    qvel3: float = 0.0,
    dt: float = 0.02,
    human_override: bool = False,
):
    qpos = (0.0, 0.0, q3, 0.0, 0.0, 0.0)
    qvel = (0.0, 0.0, qvel3, 0.0, 0.0, 0.0)
    motion = runtime.config.motion_control
    frame = runtime.step(
        control_sequence=sequence,
        control_timestamp_sec=timestamp,
        measured_qpos_rad=qpos,
        measured_qvel_rad_s=qvel,
        dt_sec=dt,
        human_override_detected=human_override,
        control_qvel_rad_s=qvel if motion is not None and motion.windowed_soft_velocity else None,
        control_qvel_window_span_sec=motion.velocity_window_sec if motion is not None and motion.windowed_soft_velocity else None,
    )
    _ack_runtime(runtime, frame, timestamp)
    return frame


def _ack_runtime(runtime, frame, timestamp, final=None):
    if runtime.config.motion_onset_confirm_displacement_rad is not None:
        runtime.record_dispatch(
            frame.reference.sequence, conditioned_torque_nm=frame.overlay_torque_nm if final is None else final,
            passive_torque_nm=ZERO, driver_kd_nm_s_per_rad=frame.drive_driver_kd_nm_s_per_rad,
            torque_rate_nm_s=(28.8,)*6, completed_at_sec=timestamp,
        )


def _active_runtime(
    staged_config: B601StagedAssistConfig | None = None,
) -> B601StagedSingleJointAssistRuntime:
    runtime = B601StagedSingleJointAssistRuntime(
        _shared_config(), staged_config or _staged_config()
    )
    assert runtime.publish_policy(_command(1, 1.0)).accepted
    return runtime


def _held_runtime(direction=1, *, recovery=False, target=.1, feedback_gain=0., arrival_margin=0., pacing=False,
                  withdrawal_confirmation=False, observation_timeout=None, handoff="stop_and_zero", governor=False,
                  arrival_stabilization=False, velocity_headroom=0., continuous_limit=None):
    motion = B601JointMotionConfig(windowed_soft_velocity=True,
        position_resolution_rad=.0004, responsive_tracking_brake=True,
        max_control_gap_sec=.04, max_reference_velocity_rad_s=.0075 if recovery else .015,
        arrival_margin_rad=arrival_margin, reference_pacing=pacing, speed_governor=governor,
        arrival_stabilization=arrival_stabilization, velocity_headroom_ratio=velocity_headroom,
        continuous_speed_limit_rad_s=continuous_limit)
    config = _staged_config(motion_control=motion,
        breakaway_torque_nm=1., breakaway_rise_slew_nm_s=1.,
        breakaway_max_duration_sec=1.2, breakaway_max_impulse_nm_sec=.6,
        max_total_overlay_nm=1., motion_onset_displacement_rad=.0004,
        motion_onset_confirm_displacement_rad=.0007,
        breakaway_probe=B601OnsetProbeConfig(hold_sec=.12, max_trial_sec=1.2,
            max_command_exposure_nm_sec=.6, max_displacement_rad=.003,
            observation_timeout_sec=observation_timeout),
        tracking_recovery_gain_nm_per_rad=20. if recovery else None,
        tracking_recovery_velocity_feedback_nm_s_per_rad=feedback_gain,
        tracking_overlay_cap_nm=1. if recovery else .2,
        tracking_slew_nm_s=.15 if recovery else 2.,
        minimum_breakaway_error_rad=.002 if recovery else .02,
        settle_error_rad=.003 if recovery else .004,
        breakaway_soft_withdrawal_confirmation=withdrawal_confirmation, onset_handoff=handoff)
    runtime = B601StagedSingleJointAssistRuntime(replace(_shared_config(),
        max_abs_assist_torque_nm=(1.,)*6,
        position_kp_nm_per_rad=(0.,0.,12.,0.,0.,0.) if recovery else _shared_config().position_kp_nm_per_rad), config)
    runtime.publish_policy(_command(1, 1., direction*target))
    _held_tick(runtime, 1.01, 0., direction=direction, target=target)
    runtime.authorize_breakaway(sequence=1010)
    _held_tick(runtime, 1.06, 0., direction=direction, target=target)
    for n in range(1, 11):
        frame = _held_tick(runtime, 1.06+n*.01, 0., direction=direction, target=target)
    assert frame.phase is B601StagedAssistPhase.BREAKAWAY
    return runtime


def _held_tick(runtime, now, q, *, direction=1, speed=0., raw=0., span=.000384,
               ack=True, final=None, human=False, ack_delay=.001,
               permission=None, target=.1, raw_limit=.05, dt_sec=.01):
    seq = round(now*1000)
    runtime.publish_policy(_command(seq, now, direction*target))
    positions = (0.,0.,direction*q,0.,0.,0.)
    velocities = (0.,0.,direction*raw,0.,0.,0.)
    soft = (0.,0.,direction*speed,0.,0.,0.)
    frame = runtime.step(control_sequence=seq, control_timestamp_sec=now,
        measured_qpos_rad=positions, measured_qvel_rad_s=velocities,
        dt_sec=dt_sec, human_override_detected=human,
        control_qvel_rad_s=soft, control_qvel_window_span_sec=.08,
        tracking_recovery_permitted=(runtime.config.tracking_recovery_gain_nm_per_rad is not None if permission is None else permission),
        stop_evidence=B601WindowStopEvidence(now, positions, velocities,
            soft, (0.,0.,span,0.,0.,0.), .08,raw_limit))
    if ack:
        _ack_runtime(runtime, frame, now+ack_delay, final=final)
    return frame


def _recovery_track(direction=1, *, feedback_gain=0., target=.008, arrival_margin=0., pacing=False, handoff="stop_and_zero", governor=False, arrival_stabilization=False, continuous_limit=None):
    runtime = _held_runtime(direction, recovery=True, target=target, feedback_gain=feedback_gain, arrival_margin=arrival_margin, pacing=pacing, handoff=handoff, governor=governor, arrival_stabilization=arrival_stabilization, continuous_limit=continuous_limit)
    for now, q in ((1.17,.000768),(1.18,.001152),(1.19,.001536)):
        _held_tick(runtime,now,q,direction=direction,target=target,speed=.008,span=q)
    for n in range(1,50):
        now = 1.19+n*.01
        frame = _held_tick(runtime,now,.001536,direction=direction,target=target)
        if frame.phase is B601StagedAssistPhase.TRACK:
            return runtime, now, frame
    pytest.fail(f"native stop/zero handoff did not reach TRACK: {frame}")


@pytest.mark.parametrize("direction", [-1, 1])
def test_onset_burst_keeps_warm_feedback_and_removes_drive_before_raw_hard_guard(direction):
    runtime = _held_runtime(direction, recovery=True, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05, feedback_gain=10.)
    runtime.config = replace(runtime.config, tracking_release_slew_nm_s=1.)
    now = 1.16
    while abs(runtime._command_delta[2]) < .62:
        now += .01
        _held_tick(runtime, now, 0., direction=direction, target=.1, raw_limit=.1)
    for q in [.000384, .000768, .001152]:
        now += .01
        frame = _held_tick(runtime, now, q, direction=direction, target=.1,
            speed=.005, raw=.0384, span=q, raw_limit=.1)
        assert frame.phase is B601StagedAssistPhase.BREAKAWAY
    previous = abs(runtime._command_delta[2])
    now += .01
    frame = _held_tick(runtime, now, .001536, direction=direction, target=.1,
        speed=.016, raw=.0384, span=.001536, raw_limit=.1)
    assert frame.phase is B601StagedAssistPhase.TRACK
    assert frame.motion_onset_confirm_count == 2
    assert abs(frame.motion.tracking_recent_velocity_rad_s) > .0195
    assert direction*frame.tracking_recovery["tracking_cut_velocity_rad_s"] > 0
    assert frame.motion.torque_transition_reason == "fast_resolved_drive_removal"
    assert frame.motion.torque_transition_rate_nm_s == 25.
    assert 0 <= direction*frame.overlay_torque_nm[2] <= previous-.15
    # A multi-count sample can advance withdrawal, not grant new authority.
    # If the supplied trajectory still crosses the raw hard limit, it MUST
    # fail closed; a prescribed replay is not a counterfactual plant forecast.
    now += .01
    frame = _held_tick(runtime, now, .002304, direction=direction, target=.1,
        speed=.025, raw=.0768, span=.002304, raw_limit=.1)
    assert abs(frame.overlay_torque_nm[2]) <= previous-.15
    frame = _held_tick(runtime, now+.01, .003456, direction=direction, target=.1,
        speed=.0376, raw=.1152, span=.003456, raw_limit=.1)
    assert frame.release_terminal
    assert frame.overlay_torque_nm == ZERO


@pytest.mark.parametrize("direction", [-1, 1])
def test_fast_drive_removal_never_restores_a_tail_already_cut_by_final_conditioning(direction):
    runtime, now, _ = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05)
    for n in range(1, 301):
        _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
    frame = _held_tick(runtime, now+3.01, .001536, direction=direction, target=.1,
        final=(0., 0., direction*.1, 0., 0., 0.))
    assert direction*runtime._previous_overlay_nm > .4
    # Boundary-level unit input: resolved recent forward motion and a drive
    # reduction proposal still ABOVE the most recent native conditioned value.
    runtime._motion_frame = replace(frame.motion, tracking_recent_velocity_rad_s=direction*.025,
        tracking_recent_position_span_rad=.002)
    output = runtime._responsive_tracking_output(direction*.4, direction*.025, .01)
    assert 0 <= direction*output <= .1
    assert runtime._motion_frame.torque_transition_reason == "fast_resolved_drive_removal"


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("feedback_gain", [0., 10.])
def test_transfer_regulation_does_not_recursively_erase_maintenance_from_exact_receipts(direction, feedback_gain):
    runtime, now, frame = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05, feedback_gain=feedback_gain)
    for n in range(1, 351):
        frame = _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
    before = abs(runtime._tracking_bias_nm)
    assert before > .4
    for n in range(1, 11):
        frame = _held_tick(runtime, now+3.5+n*.01, .001536+n*.00025,
            direction=direction, target=.1, speed=.025, raw=.025, span=.002)
        assert frame.phase is B601StagedAssistPhase.TRACK
        assert 0 < frame.motion.continuous_speed_scale < 1
        assert frame.tracking_recovery["dispatch_tracking_cut_nm"] == pytest.approx(0.)
        assert frame.tracking_recovery["tracking_velocity_correction_nm"] >= 0.
    # Ordinary velocity-error feedback still drains maintenance, once. The
    # same intentional scaled receipt must not geometrically erase it again.
    assert abs(runtime._tracking_bias_nm) > before-.06
    if feedback_gain:
        assert frame.tracking_recovery["tracking_velocity_correction_nm"] > .1
    assert frame.tracking_recovery["dispatch_credit_mode"] == "receipt_shortfall_from_previous_owner_request"
    runtime.request_withdrawal("operator_withdraw")
    _held_tick(runtime, now+3.62, .004036, direction=direction, target=.1)
    assert runtime._tracking_bias_nm == 0.


@pytest.mark.parametrize("direction", [-1, 1])
def test_transfer_maintenance_still_charges_unexpected_downstream_shortfall(direction):
    runtime, now, _ = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05)
    for n in range(1, 301):
        _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
    _held_tick(runtime, now+3.01, .001536, direction=direction, target=.1,
        final=(0., 0., direction*.05, 0., 0., 0.))
    shortfall = max(0., direction*(runtime._previous_overlay_nm-direction*.05))
    previous = abs(runtime._tracking_bias_nm)
    frame = _held_tick(runtime, now+3.02, .001536, direction=direction, target=.1)
    assert frame.tracking_recovery["dispatch_tracking_cut_nm"] == pytest.approx(min(previous, shortfall))
    assert abs(runtime._tracking_bias_nm) <= max(0., previous-shortfall)+.0015+1e-9


@pytest.mark.parametrize("direction", [-1, 1])
def test_transfer_normal_reduction_is_slower_but_reversal_keeps_fast_zero_gate(direction):
    runtime, now, frame = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05)
    runtime.config = replace(runtime.config, tracking_release_slew_nm_s=1.)
    for n in range(1, 301):
        frame = _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
    previous = frame.overlay_torque_nm[2]
    frame = _held_tick(runtime, now+3.01, .0018, direction=direction, target=.1,
        speed=.025, raw=.025, span=.002)
    # PD-only speed scaling need not reduce a maintenance-dominated total.
    # Exercise an actual normal reduction proposal at the existing slew port.
    previous = frame.overlay_torque_nm[2]
    reduced = runtime._responsive_tracking_output(previous-direction*.05, direction*.025, .01)
    assert runtime._motion_frame.torque_transition_reason == "normal_tracking_drive_reduction"
    assert runtime._motion_frame.torque_transition_rate_nm_s == 1.
    assert abs(previous-reduced) <= .01+1e-9
    # Directly exercise the existing native reversal gate with a real pending
    # same-direction receipt. Slower normal reduction must never slow reversal.
    runtime._responsive_tracking_output(-direction*.1, direction*.025, .01)
    assert runtime._motion_frame.torque_transition_reason == "native_zero_pending_before_torque_reversal"
    assert runtime._motion_frame.torque_transition_rate_nm_s == runtime.config.motion_control.braking_slew_nm_s


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_transfer_regulates_propulsion_without_rearming(direction):
    runtime, now, _ = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05)
    frame = _held_tick(runtime, now+.01, .0018, direction=direction, target=.1,
        speed=.025, raw=.038, span=.002)
    assert frame.phase is B601StagedAssistPhase.TRACK
    assert frame.motion.speed_governor_resume_count == 0
    assert 0 < frame.motion.continuous_speed_scale < 1
    recovery = frame.tracking_recovery
    before = recovery["pd_request_nm"] + frame.base_frame.decision.state.authority*recovery["bias_request_nm"]
    expected_cut = max(0., direction*recovery["pd_request_nm"])*(1-frame.motion.continuous_speed_scale)
    assert recovery["continuous_speed_drive_cut_nm"] == pytest.approx(expected_cut)
    assert recovery["velocity_feedback_cut_nm"] == 0.
    assert recovery["velocity_feedback_baseline_mode"] == "continuous_pd_scaling_with_net_velocity_feedback"
    assert recovery["combined_request_nm"] == pytest.approx(before-direction*expected_cut)
    assert abs(recovery["target_velocity_rad_s"]) < runtime.config.motion_control.max_reference_velocity_rad_s
    assert abs(frame.overlay_torque_nm[2]) <= 1.
    # Explicit withdrawal retains native zero/release behavior, not automatic
    # recovery of the old target when observation becomes quiet again.
    runtime.request_withdrawal("operator_withdraw")
    frame = _held_tick(runtime, now+.02, .0018, direction=direction, target=.1)
    assert frame.release_terminal


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_negative_feedback_can_reduce_a_maintenance_dominated_request(direction):
    runtime, now, _ = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05, feedback_gain=10.)
    for n in range(1, 301):
        _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
    frame = _held_tick(runtime, now+3.01, .0018, direction=direction, target=.1,
        speed=.0252, raw=.0384, span=.002, raw_limit=.13)
    r = frame.tracking_recovery
    full = r["pd_request_nm"] + frame.base_frame.decision.state.authority*r["bias_request_nm"]
    error_cut = 10.*max(0., direction*(direction*.0252-r["target_velocity_rad_s"]))
    assert direction*r["bias_request_nm"] > .1
    assert r["tracking_velocity_correction_nm"] == pytest.approx(min(max(0., direction*full), error_cut))
    assert r["tracking_velocity_correction_nm"] > max(0., direction*r["pd_request_nm"])
    assert 0 <= direction*r["combined_request_nm"] < direction*r["bias_request_nm"]
    # The output correction is not recursively subtracted from integral memory.
    assert r["velocity_feedback_cut_nm"] == 0.
    next_frame = _held_tick(runtime, now+3.02, .0018, direction=direction, target=.1,
        speed=.0252, raw=0., span=.002, raw_limit=.13)
    assert next_frame.tracking_recovery["dispatch_tracking_cut_nm"] == pytest.approx(0.)


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_regulation_preserves_integral_memory_not_a_drive_floor(direction):
    runtime, now, _ = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05, feedback_gain=10.)
    for n in range(1, 301):
        _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
    frame = _held_tick(runtime, now+3.01, .0018, direction=direction, target=.1,
        speed=.025, raw=.0384, span=.002, raw_limit=.13)
    assert frame.phase is B601StagedAssistPhase.TRACK
    recovery = frame.tracking_recovery
    maintenance = frame.base_frame.decision.state.authority*recovery["bias_request_nm"]
    pd = recovery["pd_request_nm"]
    assert direction*maintenance > .1
    assert 0 < frame.motion.continuous_speed_scale < 1
    # Scaling remains PD-only; velocity error can reduce net drive below I.
    # The integral remains separately bounded, not an output lower bound.
    scale_cut = max(0., direction*pd)*(1-frame.motion.continuous_speed_scale)
    velocity_cut = min(max(0., direction*(pd+maintenance)),
        10.*max(0., direction*(direction*.025-recovery["target_velocity_rad_s"])))
    assert recovery["applied_drive_cut_nm"] == pytest.approx(
        min(max(0., direction*(pd+maintenance)), max(scale_cut, velocity_cut)))
    assert 0 <= direction*recovery["combined_request_nm"] < direction*maintenance
    assert abs(frame.overlay_torque_nm[2]) <= runtime.config.max_total_overlay_nm


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("terminal", [None, "withdraw", "human", "permission", "raw_guard"])
def test_speed_brake_keeps_only_receipt_bounded_memory_never_a_residual_output(direction, terminal):
    runtime, now, frame = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05, feedback_gain=10.)
    for n in range(1, 81):
        frame = _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
    sent, memory = abs(frame.overlay_torque_nm[2]), abs(runtime._tracking_bias_nm)
    frame = _held_tick(runtime, now+.81, .0018, direction=direction, target=.1,
        speed=.055, raw=.06, span=.002, raw_limit=.13)
    assert frame.phase is B601StagedAssistPhase.BRAKE
    assert frame.motion.speed_governor_active
    frozen = abs(runtime._tracking_bias_nm)
    assert 0 < frozen <= min(sent, memory)+1e-12
    assert frame.tracking_recovery["bias_request_nm"] == 0.
    # The saved memory must not be sent during braking. Output is solely
    # the original dissipative brake, followed by the native zero receipt.
    assert frame.tracking_recovery["combined_request_nm"] == frame.motion.requested_torque_nm
    if terminal == "withdraw":
        runtime.request_withdrawal("operator_withdraw")
    now += .81
    for n in range(1, 41):
        frame = _held_tick(runtime, now+n*.01, .0018, direction=direction, target=.1,
            human=terminal == "human", permission=terminal != "permission",
            raw=.131 if terminal == "raw_guard" else 0., raw_limit=.13)
        if terminal is None and frame.phase is B601StagedAssistPhase.TRACK:
            assert frame.motion.speed_governor_resume_count == 1
            assert abs(runtime._tracking_bias_nm) > 0.
            assert abs(frame.overlay_torque_nm[2]) <= runtime.config.tracking_slew_nm_s*.01+1e-9
            break
    if terminal is None:
        assert frame.phase is B601StagedAssistPhase.TRACK
    else:
        assert runtime._tracking_bias_nm == 0.
        assert frame.phase is not B601StagedAssistPhase.TRACK


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_velocity_correction_uses_same_window_as_maintenance(direction):
    frames = []
    for raw in (0., .0384, .0768):
        runtime, now, _ = _recovery_track(direction, target=.1, pacing=True,
            handoff="track", governor=True, continuous_limit=.05, feedback_gain=10.)
        for n in range(1, 351):
            _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
        frames.append(_held_tick(runtime, now+3.51, .002736,
            direction=direction, target=.1, speed=.015, raw=raw, span=.002,
            raw_limit=.13))
    assert all(f.phase is B601StagedAssistPhase.TRACK for f in frames)
    expected = frames[0].tracking_recovery
    # An identical window/reference must produce one identical PI correction,
    # despite quantized single-frame speed. The short-window ceiling remains
    # separately effective through the existing continuous speed scale.
    for frame in frames:
        recovery = frame.tracking_recovery
        assert recovery["tracking_cut_velocity_rad_s"] == pytest.approx(direction*.015)
        assert recovery["tracking_velocity_correction_nm"] == pytest.approx(
            min(max(0., direction*(recovery["pd_request_nm"]
                +frame.base_frame.decision.state.authority*recovery["bias_request_nm"])),
                10.*max(0., direction*(direction*.015-recovery["target_velocity_rad_s"]))))
        assert recovery["combined_request_nm"] == pytest.approx(expected["combined_request_nm"])
        assert frame.overlay_torque_nm == pytest.approx(frames[0].overlay_torque_nm)
    hard = _held_tick(runtime, now+3.52, .002736,
        direction=direction, target=.1, speed=.015, raw=.131, span=.002, raw_limit=.13)
    assert hard.motion.reason == "tracking_motion_evidence_unavailable_or_unsafe"
    assert hard.motion.motion_evidence_block_reason == "raw_velocity_hard_guard"
    assert hard.release_terminal


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_velocity_and_scale_bounds_are_applied_once_to_net_request(direction):
    runtime, now, _ = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05, feedback_gain=10.)
    for n in range(1, 351):
        _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
    frame = _held_tick(runtime, now+3.51, .0018, direction=direction, target=.1,
        speed=.025, raw=.038, span=.002)
    assert frame.phase is B601StagedAssistPhase.TRACK
    recovery = frame.tracking_recovery
    full = recovery["pd_request_nm"] + frame.base_frame.decision.state.authority*recovery["bias_request_nm"]
    drive = max(0., direction*recovery["pd_request_nm"])
    scale = frame.motion.continuous_speed_scale
    velocity_cut = recovery["tracking_velocity_correction_nm"]
    assert direction*full >= velocity_cut > drive
    assert 0 < scale < 1
    scale_cut = drive*(1-scale)
    assert recovery["continuous_speed_bound_cut_nm"] == pytest.approx(scale_cut)
    assert recovery["applied_drive_cut_nm"] == pytest.approx(max(velocity_cut, scale_cut))
    assert recovery["continuous_speed_drive_cut_nm"] == pytest.approx(
        recovery["applied_drive_cut_nm"] - velocity_cut)
    combined = direction*recovery["combined_request_nm"]
    assert combined == pytest.approx(direction*full-max(velocity_cut, scale_cut))
    assert combined >= 0.
    assert abs(frame.overlay_torque_nm[2]) <= runtime.config.max_total_overlay_nm
    runtime.request_withdrawal("operator_withdraw")
    withdrawn = _held_tick(runtime, now+3.52, .0018, direction=direction, target=.1)
    assert withdrawn.release_terminal
    assert withdrawn.tracking_recovery is None


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("forward_history", [False, True])
def test_reverse_feedback_never_scales_away_dissipative_corrective_request(direction, forward_history):
    runtime, now, _ = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05)
    if forward_history:
        for n in range(1, 5):
            _held_tick(runtime, now+n*.01, .001536+n*.0003,
                direction=direction, target=.1, speed=.03, raw=.03, span=.002)
        now += .04
    else:
        # Normal onset history is now retained. Observe a genuinely quiet
        # window before testing reversal WITHOUT preceding forward motion.
        for n in range(1, 9):
            _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
        now += .08
    frame = _held_tick(runtime, now+.01, .00265 if forward_history else .0013,
        direction=direction, target=.1, speed=-.025, raw=-.038, span=.002)
    assert frame.phase is B601StagedAssistPhase.TRACK
    recovery = frame.tracking_recovery
    assert recovery["continuous_speed_drive_cut_nm"] == 0.
    assert direction*recovery["combined_request_nm"] > 0.
    assert abs(recovery["target_velocity_rad_s"]) == runtime.config.motion_control.max_reference_velocity_rad_s
    if forward_history:
        assert frame.motion.continuous_speed_scale < 1.
    else:
        assert frame.motion.continuous_speed_scale == 1.


@pytest.mark.parametrize("direction", [-1, 1])
def test_warm_forward_history_never_removes_force_opposing_current_reverse_motion(direction):
    runtime, now, _ = _recovery_track(direction, target=.1, pacing=True,
        handoff="track", governor=True, continuous_limit=.05, feedback_gain=10.)
    # Warm normal maintenance independently of the now explicitly dropped
    # one-shot startup pulse, before introducing the short forward history.
    for n in range(1, 81):
        _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.1)
    now += .8
    for n in range(1, 5):
        _held_tick(runtime, now+n*.01, .001536+n*.0003,
            direction=direction, target=.1, speed=.03, raw=.03, span=.002)
    frame = _held_tick(runtime, now+.05, .00265, direction=direction, target=.1,
        speed=-.025, raw=-.038, span=.002)
    assert frame.phase is B601StagedAssistPhase.TRACK
    assert direction*frame.motion.tracking_recent_velocity_rad_s > 0.
    assert frame.tracking_recovery["tracking_velocity_correction_nm"] == 0.
    assert frame.tracking_recovery["continuous_speed_drive_cut_nm"] == 0.
    assert direction*frame.tracking_recovery["combined_request_nm"] > 0.
    assert frame.motion.torque_transition_reason != "fast_resolved_drive_removal"


@pytest.mark.parametrize("direction", [-1, 1])
def test_transfer_arrival_recoil_corrects_current_error_but_requires_zero_before_reversal(direction):
    runtime, now, frame = _recovery_track(direction, target=.02, pacing=True,
        handoff="track", governor=True, continuous_limit=.05, arrival_stabilization=True)
    for _ in range(600):
        now += .01
        q = direction*frame.motion.q_reference_rad
        frame = _held_tick(runtime, now, q,
            direction=direction, target=.02, speed=abs(frame.motion.qvel_reference_rad_s))
        if frame.motion.arrival_stabilization_active:
            break
    assert frame.motion.arrival_stabilization_active
    while q < .0236-1e-9:
        q = min(.0236, q+.0001)
        now += .01
        zero_was_confirmed = runtime._final_zero_confirmed
        previous_sent = runtime._command_delta[2]
        frame = _held_tick(runtime, now, q, direction=direction, target=.02,
            speed=.01, raw=.01, span=.001)
        if direction*frame.overlay_torque_nm[2] < 0 and direction*previous_sent >= 0:
            assert zero_was_confirmed
    saw_corrective_output = False
    for _ in range(12):
        now += .01
        zero_was_confirmed = runtime._final_zero_confirmed
        previous_sent = runtime._command_delta[2]
        frame = _held_tick(runtime, now, .0236, direction=direction, target=.02)
        assert frame.motion.arrival_stabilization_active
        assert direction*frame.tracking_recovery["bias_request_nm"] < 0.
        if direction*frame.overlay_torque_nm[2] < 0:
            if direction*previous_sent >= 0:
                assert zero_was_confirmed
            saw_corrective_output = True
        assert abs(frame.overlay_torque_nm[2]) <= 1.
    assert saw_corrective_output


@pytest.mark.parametrize("direction", [-1, 1])
def test_confirmed_held_onset_transfers_accepted_command_without_intermediate_stop(direction):
    runtime = _held_runtime(direction, recovery=True, target=.02, pacing=True, handoff="track")
    sent_before = runtime._command_delta[2]
    frames = [_held_tick(runtime, now, q, direction=direction, target=.02, speed=.008, span=q)
              for now, q in ((1.17, .000768), (1.18, .001152), (1.19, .001536))]
    frame = frames[-1]
    assert frame.phase is B601StagedAssistPhase.TRACK
    assert frame.reason == "confirmed_onset_transferred_to_tracking"
    assert frame.tracking_recovery["handoff_seed_command_nm"] == sent_before
    assert 0 < direction*frame.overlay_torque_nm[2] <= abs(sent_before)
    assert frame.motion.target_position_rad == direction*.02
    assert frame.motion.q_reference_rad == direction*.001536
    assert not frame.release_terminal
    assert all(f.phase is not B601StagedAssistPhase.RELEASING for f in frames)
    assert frame.breakaway_probe["stage"] == "tracking_handoff"
    assert runtime._breakaway_started_sec == 1.06
    assert not runtime._command_accounting


@pytest.mark.parametrize("cause", ["human", "permit", "hard_speed", "withdraw", "lost_receipt"])
def test_tracking_handoff_preserves_terminal_withdrawal_and_dispatch_requirements(cause):
    runtime = _held_runtime(recovery=True, target=.02, pacing=True, handoff="track")
    _held_tick(runtime, 1.17, .000768, target=.02, speed=.008, span=.000768)
    _held_tick(runtime, 1.18, .001152, target=.02, speed=.008, span=.001152,
               ack=cause != "lost_receipt")
    options = {"target": .02, "speed": .008, "span": .001536}
    if cause == "lost_receipt":
        with pytest.raises(ValueError, match="lacks a complete conditioned dispatch"):
            _held_tick(runtime, 1.19, .001536, **options)
        return
    if cause == "withdraw":
        runtime.request_withdrawal("operator_withdraw")
    elif cause == "human":
        options["human"] = True
    elif cause == "permit":
        options["permission"] = False
    else:
        options["raw"] = .051
    frame = _held_tick(runtime, 1.19, .001536, **options)
    assert frame.phase is B601StagedAssistPhase.RELEASING
    assert frame.release_terminal
    assert runtime._tracking_bias_nm == 0.
    assert frame.tracking_recovery is None


def test_tracking_handoff_does_not_promote_repeated_or_single_count_feedback():
    runtime = _held_runtime(recovery=True, target=.02, pacing=True, handoff="track")
    for i in range(1, 15):
        frame = _held_tick(runtime, 1.16+i*.01, .000768 if i % 2 else .001152,
                           target=.02, speed=0., span=.000384)
        assert frame.phase is B601StagedAssistPhase.BREAKAWAY
        assert frame.breakaway_probe["distinct_progress_samples"] <= 1


@pytest.mark.parametrize("cause", ["human", "permit", "withdraw", "deadline", "support_tail"])
def test_normal_handoff_cannot_preserve_bias_after_stop_or_exhaustion(cause):
    runtime = _held_runtime(recovery=True, target=.02, pacing=True, handoff="track")
    for now, q in ((1.17, .000768), (1.18, .001152), (1.19, .001536)):
        if cause == "deadline" and now == 1.19:
            runtime._breakaway_started_sec = 0.
            now = 1.21
        final = (0., 0., .1, .001, 0., 0.) if cause == "support_tail" and now == 1.18 else None
        frame = _held_tick(runtime, now, q, target=.02, speed=.008, span=q, final=final)
    if cause not in {"deadline", "support_tail"}:
        assert frame.phase is B601StagedAssistPhase.TRACK
        if cause == "withdraw":
            runtime.request_withdrawal("operator_withdraw")
        frame = _held_tick(runtime, 1.20, .001536, target=.02,
            human=cause == "human", permission=cause != "permit")
    assert frame.phase is B601StagedAssistPhase.RELEASING
    assert frame.release_terminal
    assert runtime._tracking_bias_nm == 0.


@pytest.mark.parametrize("direction", [-1, 1])
def test_unload_return_remains_a_diagnostic_failure_but_normal_motion_hands_off(direction):
    diagnostic = _held_runtime(direction, recovery=True, target=.02, pacing=True)
    execution = _held_runtime(direction, recovery=True, target=.02, pacing=True, handoff="track")
    for now, q in ((1.17, .000768), (1.18, .001152), (1.19, .001536)):
        for runtime in (diagnostic, execution):
            _held_tick(runtime, now, q, direction=direction, target=.02, speed=.008, span=q)
    assert execution.phase is B601StagedAssistPhase.TRACK
    assert diagnostic.phase is B601StagedAssistPhase.RELEASING
    for i in range(1, 45):
        now = 1.19+i*.01
        q = max(.000384, .001536-i*.000384)
        frame = _held_tick(diagnostic, now, q, direction=direction, target=.02, ack=False)
        tail = (0., 0., direction*.001, 0., 0., 0.)
        _ack_runtime(diagnostic, frame, now+.001,
                     final=ZERO if frame.motion.phase == "soft_hold" else tail)
        if diagnostic._release_terminal:
            break
    assert diagnostic._release_terminal
    assert diagnostic._release_reason == "held_probe_progress_returned_after_withdrawal"


@pytest.mark.parametrize("direction", [-1, 1])
def test_paced_recovery_retains_bounded_stall_demand_instead_of_reference_lag_exit(direction):
    """Scripted feedback reproduces post-onset arrest, not physical friction."""
    baseline, old_now, _ = _recovery_track(direction, target=.02, feedback_gain=40.)
    paced, now, _ = _recovery_track(direction, target=.02, feedback_gain=40., pacing=True)
    onset_clock = paced._breakaway_started_sec
    old_braked = False
    for n in range(1, 501):
        old = _held_tick(baseline, old_now+n*.01, .001536, direction=direction, target=.02)
        old_braked |= old.reason == "reference_lag_braking"
        frame = _held_tick(paced, now+n*.01, .001536, direction=direction, target=.02)
        assert frame.phase is B601StagedAssistPhase.TRACK
        assert abs(frame.motion.q_reference_rad-direction*.001536) < .013
        assert frame.motion.target_position_rad == direction*.02
        assert frame.tracking_recovery["target_velocity_rad_s"] == pytest.approx(direction*.0075)
        assert abs(frame.overlay_torque_nm[2]) <= 1.
    assert old_braked
    assert abs(frame.overlay_torque_nm[2]) > .6
    assert frame.motion.reference_speed_scale < .01
    assert paced._breakaway_started_sec == onset_clock
    # Resume gradual measured progress without resetting target/reference or
    # authorizing a second pulse. Arrival still uses the original tracker.
    q = .001536
    for n in range(1, 401):
        q = min(.018, q+.00006)
        frame = _held_tick(paced, now+5+n*.01, q, direction=direction, target=.02,
                           speed=.006 if q < .018 else 0., raw=.006 if q < .018 else 0.,
                           span=.00048 if q < .018 else .000384)
        assert abs(frame.overlay_torque_nm[2]) <= 1.
        if frame.phase is B601StagedAssistPhase.BRAKE:
            assert frame.reason == "target_reached_braking"
            assert paced._tracking_bias_nm == 0.
            break
    else:
        pytest.fail(str(frame))


@pytest.mark.parametrize("direction", [-1,1])
def test_tracking_recovery_generates_demand_for_recorded_stall_without_new_pulse(direction):
    runtime, now, frame = _recovery_track(direction)
    onset_clock = runtime._breakaway_started_sec
    for n in range(1,501):
        frame = _held_tick(runtime,now+n*.01,.001536,direction=direction,target=.008)
        assert frame.phase is B601StagedAssistPhase.TRACK
        assert abs(frame.overlay_torque_nm[2]) <= 1.
    assert abs(frame.tracking_recovery["pd_request_nm"]) < .1
    assert abs(frame.overlay_torque_nm[2]) > .5
    assert frame.tracking_recovery["target_velocity_rad_s"] == pytest.approx(direction*.0075)
    assert runtime._breakaway_started_sec == onset_clock
    # A resolved speed above the reference decreases the bias, rather than
    # continuing its stationary ramp. The existing soft brake remains final.
    bias = abs(frame.tracking_recovery["bias_request_nm"])
    frame = _held_tick(runtime,now+5.01,.00162,direction=direction,target=.008,speed=.009,raw=.0084,span=.00072)
    assert abs(frame.tracking_recovery["bias_request_nm"]) < bias


@pytest.mark.parametrize("direction", [-1,1])
def test_recovery_velocity_feedback_cuts_demand_before_soft_guard_without_washing_out_maintenance(direction):
    runtime, now, frame = _recovery_track(direction,feedback_gain=40.)
    for n in range(1,351):
        frame = _held_tick(runtime,now+n*.01,.001536,direction=direction,target=.008)
    before = abs(frame.overlay_torque_nm[2])
    state_before = abs(runtime._tracking_bias_nm)
    frame = _held_tick(runtime,now+3.51,.00162,direction=direction,target=.008,
        speed=.0112,raw=.0084,span=.000768)
    r = frame.tracking_recovery
    assert frame.phase is B601StagedAssistPhase.TRACK
    assert r["velocity_feedback_cut_nm"] == pytest.approx(40.*.0112)
    assert abs(r["bias_request_nm"]) < state_before-.25
    assert before-abs(frame.overlay_torque_nm[2]) == pytest.approx(.02)
    assert abs(runtime._tracking_bias_nm) == pytest.approx(state_before+20.*(.0075-.0112)*.01)
    assert direction * r["bias_request_nm"] >= 0
    retained = abs(runtime._tracking_bias_nm)
    frame = _held_tick(runtime,now+3.52,.00162,direction=direction,target=.008)
    assert abs(runtime._tracking_bias_nm) <= retained+.0015+1e-9


def test_recovery_velocity_feedback_single_count_or_opposing_velocity_cannot_cut_bias():
    runtime, now, frame = _recovery_track(feedback_gain=40.)
    for n in range(1,101):
        frame = _held_tick(runtime,now+n*.01,.001536,target=.008)
    frame = _held_tick(runtime,now+1.01,.00162,target=.008,speed=.009,span=.000384)
    assert frame.tracking_recovery["velocity_feedback_cut_nm"] == 0.
    frame = _held_tick(runtime,now+1.02,.00162,target=.008,speed=-.009,span=.000768)
    assert frame.tracking_recovery["velocity_feedback_cut_nm"] == 0.


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_tracking_does_not_recharge_resolved_speed_after_quantized_gaps(direction):
    runtime, now, _frame = _recovery_track(
        direction, feedback_gain=10., pacing=True, handoff="track", governor=True)
    for n in range(1, 101):
        _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.008)
    frame = _held_tick(runtime, now+1.01, .00162, direction=direction,
                       target=.008, speed=.010, span=.001152)
    assert frame.tracking_recovery["velocity_feedback_cut_nm"] == pytest.approx(10.*(.010-.008))
    # Resolved progress interspersed with one-count windows is normal at low
    # speed. It must not repeatedly subtract K*v from the maintenance state.
    for n, (speed, span) in enumerate((
        (.003, .000768), (.001, .000384), (.004, .000768),
        (.002, .000384), (.006, .001152), (.0075, .001152),
    ), 102):
        frame = _held_tick(runtime, now+n*.01, .00162, direction=direction,
                           target=.008, speed=speed, span=span)
        assert frame.phase is B601StagedAssistPhase.TRACK
        assert frame.tracking_recovery["velocity_feedback_cut_nm"] == 0.


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_tracking_unresolved_window_preserves_speed_feedback_baseline(direction):
    runtime, now, _frame = _recovery_track(
        direction, feedback_gain=40., pacing=True, handoff="track", governor=True)
    for n in range(1, 351):
        _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.008)
    frame = _held_tick(runtime, now+3.51, .00162, direction=direction,
                       target=.008, speed=.0112, span=.000768)
    assert frame.tracking_recovery["velocity_feedback_cut_nm"] == pytest.approx(40.*(.0112-.008))
    frame = _held_tick(runtime, now+3.52, .00162, direction=direction,
                       target=.008, speed=.001, span=.000384)
    assert frame.tracking_recovery["velocity_feedback_cut_nm"] == 0.
    assert frame.tracking_recovery["feedback_velocity_rad_s"] == pytest.approx(direction*.0112)
    frame = _held_tick(runtime, now+3.53, .00162, direction=direction,
                       target=.008, speed=.0112, span=.000768)
    assert frame.tracking_recovery["velocity_feedback_cut_nm"] == 0.
    # A genuine further increase above the old resolved speed still cuts
    # promptly, before the unchanged soft/hard guards take ownership.
    frame = _held_tick(runtime, now+3.54, .00162, direction=direction,
                       target=.008, speed=.012, span=.001152)
    assert frame.tracking_recovery["velocity_feedback_cut_nm"] == pytest.approx(40.*(.012-.0112))


@pytest.mark.parametrize("cause", ["permit", "human", "hard_speed", "withdraw", "soft_speed"])
def test_governed_tracking_clears_speed_frontier_and_bias_on_exit(cause):
    runtime, now, _frame = _recovery_track(
        feedback_gain=10., pacing=True, handoff="track", governor=True)
    assert runtime._tracking_recovery["feedback_velocity_rad_s"] > 0.
    args = {"target": .008}
    if cause == "permit":
        args["permission"] = False
    elif cause == "human":
        args["human"] = True
    elif cause == "hard_speed":
        args["raw"] = .051
    elif cause == "withdraw":
        runtime.request_withdrawal("operator_withdraw")
    else:
        args.update(speed=.025, span=.001152)
    frame = _held_tick(runtime, now+.01, .00162, **args)
    assert frame.phase is not B601StagedAssistPhase.TRACK
    assert runtime._tracking_bias_nm == 0.
    if frame.tracking_recovery is not None:
        assert frame.tracking_recovery["feedback_velocity_rad_s"] == 0.


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_tracking_keeps_net_bias_after_motion_instead_of_restoring_hidden_reserve(direction):
    runtime, now, frame = _recovery_track(direction, feedback_gain=40., pacing=True, handoff="track")
    for n in range(1, 351):
        frame = _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.008)
    before = abs(runtime._tracking_bias_nm)
    frame = _held_tick(runtime, now+3.51, .00162, direction=direction, target=.008,
                       speed=.0075, span=.000768)
    recovery = frame.tracking_recovery
    assert recovery["maintenance_state_mode"] == "dispatch_tracked_net_bias"
    assert recovery["velocity_feedback_cut_nm"] == pytest.approx(.3)
    assert abs(runtime._tracking_bias_nm) <= before-.29
    reduced = abs(runtime._tracking_bias_nm)
    # Constant resolved speed does not repeatedly subtract K*v from memory.
    frame = _held_tick(runtime, now+3.52, .00162, direction=direction, target=.008,
                       speed=.0075, span=.000768)
    assert frame.tracking_recovery["velocity_feedback_cut_nm"] == 0.
    assert abs(runtime._tracking_bias_nm) == pytest.approx(reduced)
    # A quiet window may resume only the original slow rise, not a saved peak.
    frame = _held_tick(runtime, now+3.53, .00162, direction=direction, target=.008)
    assert abs(runtime._tracking_bias_nm) <= reduced+.0015+1e-9


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_tracking_backcalculates_conditioned_dispatch_not_requested_torque(direction):
    runtime, now, frame = _recovery_track(direction, pacing=True, handoff="track")
    for n in range(1, 201):
        frame = _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.008)
    previous_pd = frame.tracking_recovery["pd_request_nm"]
    assert abs(runtime._tracking_bias_nm) > .2
    delivered = direction*.05
    _held_tick(runtime, now+2.01, .001536, direction=direction, target=.008,
               final=(0., 0., delivered, 0., 0., 0.))
    frame = _held_tick(runtime, now+2.02, .001536, direction=direction, target=.008)
    assert frame.tracking_recovery["dispatch_tracking_cut_nm"] > .1
    assert abs(runtime._tracking_bias_nm) <= max(0., direction*(delivered-previous_pd))+.0015+1e-9


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_feedback_does_not_treat_reduced_opposing_velocity_as_aligned_acceleration(direction):
    runtime, now, _frame = _recovery_track(direction, feedback_gain=10., pacing=True, handoff="track")
    for n, speed in enumerate((-.006, -.003), 1):
        frame = _held_tick(runtime, now+n*.01, .001536, direction=direction,
                           target=.008, speed=speed, span=.000768)
        assert frame.phase is B601StagedAssistPhase.TRACK
        assert frame.tracking_recovery["velocity_feedback_cut_nm"] == 0.


@pytest.mark.parametrize("direction", [-1, 1])
def test_governed_tracking_single_count_pd_changes_do_not_erase_stall_maintenance(direction):
    runtime, now, frame = _recovery_track(
        direction, feedback_gain=10., pacing=True, handoff="track", governor=True)
    estimator = B601WindowVelocityEstimator(.08)
    anchor = .001536
    for n in range(10):
        estimator.update(timestamp_sec=now-.09+n*.01, qpos_rad=anchor, fallback_rad_s=0.)
    previous_q = anchor
    previous_sent = abs(runtime._command_delta[2])
    onset_clock = runtime._breakaway_started_sec
    for n in range(1, 301):
        # Declared stationary single-count toggle, not actual net progress.
        q = anchor+(n % 2)*.000383496
        raw = (q-previous_q)/.01
        speed = estimator.update(timestamp_sec=now+n*.01, qpos_rad=q, fallback_rad_s=raw)
        frame = _held_tick(runtime, now+n*.01, q, direction=direction,
            target=.008, speed=speed, raw=raw, span=estimator.position_span_rad)
        assert frame.phase is B601StagedAssistPhase.TRACK
        # Only the original bounded velocity-error increment can add state.
        assert abs(runtime._tracking_bias_nm) <= previous_sent+.0015+1e-9
        previous_sent = abs(frame.overlay_torque_nm[2])
        previous_q = q
    assert previous_sent > .25
    assert runtime._tracking_bias_nm*direction > .2
    assert runtime._breakaway_started_sec == onset_clock
    assert frame.tracking_recovery["dispatch_credit_mode"] == "maintenance_first_bounded_by_sent_total"


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("delivered", [0., .02])
def test_governed_maintenance_credit_still_cuts_state_to_actual_final_output(direction, delivered):
    runtime, now, frame = _recovery_track(
        direction, pacing=True, handoff="track", governor=True)
    for n in range(1, 201):
        frame = _held_tick(runtime, now+n*.01, .001536, direction=direction, target=.008)
    assert abs(runtime._tracking_bias_nm) > .2
    _held_tick(runtime, now+2.01, .001536, direction=direction, target=.008,
        final=(0., 0., direction*delivered, 0., 0., 0.))
    frame = _held_tick(runtime, now+2.02, .001536, direction=direction, target=.008)
    assert frame.tracking_recovery["dispatch_tracking_cut_nm"] > .1
    assert abs(runtime._tracking_bias_nm) <= delivered+.0015+1e-9


@pytest.mark.parametrize("direction", [-1, 1])
def test_governed_nonzero_pd_does_not_create_maintenance_without_velocity_error(direction):
    runtime, now, frame = _recovery_track(
        direction, pacing=True, handoff="track", governor=True)
    runtime._tracking_bias_nm = 0.
    for n in range(1, 101):
        frame = _held_tick(runtime, now+n*.01, .001536, direction=direction,
            target=.008, speed=.0075, span=.000768)
        assert frame.phase is B601StagedAssistPhase.TRACK
        assert runtime._tracking_bias_nm == 0.
    assert abs(frame.tracking_recovery["pd_request_nm"]) > .05


@pytest.mark.parametrize("direction", [-1, 1])
def test_governed_opposing_pd_cannot_hide_maintenance_above_delivered_total(direction):
    runtime, _, frame = _recovery_track(
        direction, pacing=True, handoff="track", governor=True)
    # Only 0.1 was delivered; an opposing 0.7 PD request must not justify a
    # latent 0.8 maintenance state that could reappear when PD disappears.
    runtime._tracking_bias_nm = direction*.8
    runtime._tracking_recovery = {"pd_request_nm": -direction*.7}
    runtime._command_delta = (0., 0., direction*.1, 0., 0., 0.)
    motion = replace(frame.motion, requested_torque_nm=-direction*.7)
    requested = runtime._tracking_recovery_output(motion, frame.base_frame, .01, 0., 1.)
    assert abs(runtime._tracking_bias_nm) <= .1015+1e-9
    runtime._previous_overlay_nm = direction*.1
    runtime._final_zero_confirmed = False
    # This fixture retains the original 2 Nm/s withdrawal rate: reaching zero
    # takes five ticks, and reversal must still await a native zero receipt.
    for _ in range(5):
        previous = runtime._previous_overlay_nm
        output = runtime._responsive_tracking_output(requested, 0., .01)
        assert direction*output >= -1e-9
        assert abs(output-previous) <= .02+1e-9
        runtime._previous_overlay_nm = output
    assert output == pytest.approx(0., abs=1e-9)
    assert runtime._responsive_tracking_output(requested, 0., .01) == pytest.approx(0., abs=1e-9)


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_tracking_faster_unload_still_waits_for_native_zero_before_reversal(direction):
    runtime, _now, frame = _recovery_track(direction, pacing=True, handoff="track")
    runtime.config = replace(runtime.config, motion_control=replace(
        runtime.config.motion_control, braking_slew_nm_s=10.))
    runtime._previous_overlay_nm = direction*.8
    runtime._command_delta = (0., 0., direction*.8, 0., 0., 0.)
    runtime._phase = B601StagedAssistPhase.BRAKE
    frame = replace(frame.motion, phase="brake")
    runtime._motion_frame = frame
    for n in range(8):
        result = runtime._responsive_tracking_output(-direction*.1, direction*.01, .01)
        assert direction*result >= -1e-9
        runtime._previous_overlay_nm = result
    assert result == pytest.approx(0., abs=1e-9)
    # A zero request with an old nonzero receipt still cannot reverse.
    zero = runtime._responsive_tracking_output(-direction*.1, direction*.01, .01)
    assert zero == 0.
    runtime._previous_overlay_nm = zero
    runtime._command_delta = ZERO
    runtime._last_output_zero = runtime._final_zero_confirmed = True
    assert runtime._responsive_tracking_output(-direction*.1, direction*.01, .01) == pytest.approx(-direction*.1)


@pytest.mark.parametrize("feedback_gain", [0.,40.])
@pytest.mark.parametrize("cause", ["permit", "human", "hard_speed", "target", "withdraw"])
@pytest.mark.parametrize("pacing", [False, True])
def test_tracking_recovery_never_carries_stall_bias_into_exit(cause,feedback_gain,pacing):
    runtime, now, frame = _recovery_track(feedback_gain=feedback_gain,pacing=pacing)
    for n in range(1,101):
        frame = _held_tick(runtime,now+n*.01,.001536,target=.008)
    assert frame.tracking_recovery["bias_request_nm"] > .1
    args = {"target": .008}
    if cause == "permit":
        args["permission"] = False
    elif cause == "human":
        args["human"] = True
    elif cause == "hard_speed":
        args["raw"] = .051
    elif cause == "withdraw":
        runtime.request_withdrawal("operator_withdraw")
    q = .0051 if cause == "target" else .001536
    frame = _held_tick(runtime,now+1.01,q,**args)
    assert runtime._tracking_bias_nm == 0.
    assert frame.phase is not B601StagedAssistPhase.TRACK


def _arrival_runtime(direction=1, *, continuous_limit=None):
    runtime, now, _ = _recovery_track(direction, feedback_gain=10., target=.02,
        arrival_margin=.0008, pacing=True, handoff="track", governor=True,
        arrival_stabilization=True, continuous_limit=continuous_limit)
    for _ in range(200):
        now += .01
        _held_tick(runtime, now, .001536, direction=direction, target=.02)
    q = .001536
    while q < .01785:
        now += .01
        q += .00005
        frame = _held_tick(runtime, now, q, direction=direction,
            target=.02, speed=.005, raw=.005)
    assert frame.motion.arrival_stabilization_active
    return runtime, now, q, frame


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("interval", [.01, .025])
def test_continuous_arrival_does_not_recharge_planned_velocity_cut_to_memory(direction, interval):
    runtime, now, q, _ = _arrival_runtime(direction, continuous_limit=.05)
    runtime._tracking_bias_nm = direction*.3
    runtime._previous_overlay_nm = direction*.3
    runtime._command_delta = (0., 0., direction*.3, 0., 0., 0.)
    runtime._last_output_zero = runtime._final_zero_confirmed = False
    # A prescribed constant-speed window isolates the receipt algebra. This
    # is not a plant forecast: each receipt matches its preceding request.
    for n in range(1, 9):
        frame = _held_tick(runtime, now+n*interval, q, direction=direction,
            target=.02, speed=.008, raw=.008, span=.0012, raw_limit=.13, dt_sec=interval)
        assert frame.motion.arrival_stabilization_active
        assert not frame.motion.arrival_unloading
        assert frame.tracking_recovery["tracking_velocity_correction_nm"] > .05
        assert frame.tracking_recovery["dispatch_tracking_cut_nm"] == pytest.approx(0.)
        assert abs(runtime._tracking_bias_nm) == pytest.approx(.3-n*20.*.008*interval)
    previous = abs(runtime._tracking_bias_nm)
    runtime._command_delta = (0., 0., runtime._previous_overlay_nm-direction*.1, 0., 0., 0.)
    frame = _held_tick(runtime, now+9*interval, q, direction=direction,
        target=.02, speed=.008, raw=.008, span=.0012, raw_limit=.13, dt_sec=interval)
    assert frame.tracking_recovery["dispatch_tracking_cut_nm"] == pytest.approx(.1)
    assert abs(runtime._tracking_bias_nm) == pytest.approx(previous-.1-20.*.008*interval)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("sent", [.589663066342947, .4])
def test_continuous_arrival_cannot_restore_latent_bias_or_conditioned_shortfall(direction, sent):
    runtime, now, frame = _recovery_track(direction, feedback_gain=10., target=.02,
        arrival_margin=.0008, pacing=True, handoff="track", governor=True,
        arrival_stabilization=True, continuous_limit=.05)
    q = .001536
    while q < .0177-1e-12:
        q = min(.0177, q+.00005)
        now += .01
        frame = _held_tick(runtime, now, q, direction=direction, target=.02,
            speed=.005, raw=.005, span=.0012, raw_limit=.13)
        assert frame.phase is B601StagedAssistPhase.TRACK
    # Recorded failure structure, reflected in both directions: internal
    # maintenance exceeds the requested and actually dispatched overlays.
    # These are declared controller states, not inferred physical friction.
    runtime.config = replace(runtime.config, motion_control=replace(
        runtime.config.motion_control, braking_slew_nm_s=10.))
    runtime._tracking_bias_nm = direction*.8600604716862357
    runtime._previous_overlay_nm = direction*.589663066342947
    runtime._command_delta = (0., 0., direction*sent, 0., 0., 0.)
    runtime._last_output_zero = runtime._final_zero_confirmed = False
    now += .01
    q = .0179
    frame = _held_tick(runtime, now, q, direction=direction, target=.02,
        speed=.0084812277, raw=.038, span=.00076699, raw_limit=.13)
    assert frame.motion.arrival_stabilization_active
    assert direction*frame.overlay_torque_nm[2] <= sent+1e-10
    assert abs(runtime._tracking_bias_nm) <= sent+1e-10
    assert frame.tracking_recovery["tracking_velocity_correction_nm"] > 0.
    assert frame.tracking_recovery["dispatch_tracking_cut_nm"] > .2
    previous = abs(frame.overlay_torque_nm[2])
    for _ in range(250):
        now += .01
        frame = _held_tick(runtime, now, q, direction=direction, target=.02, raw_limit=.13)
        # Quieting the velocity correction is not final unloading. A bounded
        # stabilization rise is permitted, but cannot restore the latent .86
        # state or bypass the original slow-rise and final-unload contracts.
        assert abs(runtime._tracking_bias_nm) <= sent+1e-10
        if frame.motion.arrival_unloading:
            assert abs(frame.overlay_torque_nm[2]) <= previous+1e-10
        else:
            assert abs(frame.overlay_torque_nm[2])-previous <= runtime.config.tracking_slew_nm_s*.01+1e-10
        previous = abs(frame.overlay_torque_nm[2])
        if frame.phase is B601StagedAssistPhase.SOFT_HOLD:
            break
    assert frame.phase is B601StagedAssistPhase.SOFT_HOLD
    assert frame.overlay_torque_nm == ZERO and runtime._final_zero_confirmed


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("velocity", [0., .008])
def test_continuous_arrival_fast_brake_rate_cannot_amplify_same_direction_drive(direction, velocity):
    runtime, _, _, frame = _arrival_runtime(direction, continuous_limit=.05)
    runtime.config = replace(runtime.config, motion_control=replace(
        runtime.config.motion_control, braking_slew_nm_s=10.))
    runtime._previous_overlay_nm = direction*.2
    runtime._command_delta = (0., 0., direction*.1, 0., 0., 0.)
    runtime._last_output_zero = runtime._final_zero_confirmed = False
    assert frame.motion.arrival_stabilization_active and not frame.motion.arrival_unloading
    output = runtime._responsive_tracking_output(direction*.6, direction*velocity, .01)
    assert output == pytest.approx(direction*(.1+.15*.01))
    assert runtime._motion_frame.torque_transition_reason == "bounded_arrival_correction_rise"


@pytest.mark.parametrize("direction", [-1, 1])
def test_arrival_retains_receipt_bounded_maintenance_then_gradually_unloads(direction):
    runtime, now, q, frame = _arrival_runtime(direction)
    assert direction*frame.overlay_torque_nm[2] > .05
    assert frame.tracking_recovery["reason"] == "arrival_bounded_stabilization"
    unloading = []
    for _ in range(250):
        now += .01
        frame = _held_tick(runtime, now, q, direction=direction, target=.02)
        assert abs(frame.overlay_torque_nm[2]) <= 1.
        if frame.motion.arrival_unloading:
            unloading.append(abs(frame.overlay_torque_nm[2]))
            assert frame.motion.requested_torque_nm == 0.
        if frame.phase is B601StagedAssistPhase.SOFT_HOLD:
            break
    assert len(unloading) > 5
    for previous, current in pairwise(unloading):
        assert 0 <= previous-current <= .5*.01+1e-10
    assert frame.phase is B601StagedAssistPhase.SOFT_HOLD
    assert runtime._final_zero_confirmed
    assert frame.overlay_torque_nm == ZERO


@pytest.mark.parametrize("direction", [-1, 1])
def test_arrival_one_count_hysteresis_keeps_final_command_unloading(direction):
    runtime, now, q, frame = _arrival_runtime(direction)
    for _ in range(23):
        now += .01
        frame = _held_tick(runtime, now, q, direction=direction, target=.02)
    assert frame.motion.arrival_unloading
    count = .00038348883390426636
    previous = abs(frame.overlay_torque_nm[2])
    for n in range(1, 180):
        now += .01
        frame = _held_tick(runtime, now, q if n%2 else q-count,
            direction=direction, target=.02, span=count)
        current = abs(frame.overlay_torque_nm[2])
        assert current <= previous+1e-10
        assert previous-current <= .5*.01+1e-10
        previous = current
        if frame.phase is B601StagedAssistPhase.SOFT_HOLD:
            break
        assert frame.motion.arrival_unloading
        assert frame.motion.requested_torque_nm == 0.
    assert frame.phase is B601StagedAssistPhase.SOFT_HOLD
    assert runtime._final_zero_confirmed
    assert frame.overlay_torque_nm == ZERO
    assert abs(frame.motion.target_error_rad) <= .003


@pytest.mark.parametrize("during_unloading", [False, True])
@pytest.mark.parametrize("cause", ["human", "permit", "withdraw", "hard_speed", "soft_speed"])
@pytest.mark.parametrize("continuous_limit", [None, .05])
def test_arrival_maintenance_cannot_survive_stop_or_permission_loss(cause, during_unloading, continuous_limit):
    runtime, now, q, _ = _arrival_runtime(continuous_limit=continuous_limit)
    if during_unloading:
        for _ in range(23):
            now += .01
            frame = _held_tick(runtime, now, q, target=.02)
        assert frame.motion.arrival_unloading
    options = {"target": .02}
    if cause == "human":
        options["human"] = True
    elif cause == "permit":
        options["permission"] = False
    elif cause == "withdraw":
        runtime.request_withdrawal("operator_withdraw")
    elif cause == "hard_speed":
        options["raw"] = .051
    else:
        options.update(speed=(continuous_limit or .03)+.001, span=.003)
    frame = _held_tick(runtime, now+.01, q, **options)
    assert not frame.motion.arrival_stabilization_active
    assert runtime._tracking_bias_nm == 0.
    for n in range(2, 15):
        frame = _held_tick(runtime, now+n*.01, q, target=.02)
        assert frame.motion is None or not frame.motion.arrival_stabilization_active
        assert runtime._tracking_bias_nm == 0.


def test_tracking_recovery_saturation_has_no_hidden_integral_reserve():
    runtime, now, frame = _recovery_track()
    for n in range(1,1001):
        frame = _held_tick(runtime,now+n*.01,.001536,target=.008)
    recovery = frame.tracking_recovery
    assert recovery["bias_request_nm"] + recovery["pd_request_nm"] == pytest.approx(1.)
    assert runtime._tracking_bias_nm < 1.
    assert frame.overlay_torque_nm[2] <= 1.


def test_tracking_recovery_missing_same_tick_evidence_fails_without_retaining_bias():
    runtime, now, frame = _recovery_track()
    for n in range(1,101):
        frame = _held_tick(runtime,now+n*.01,.001536,target=.008)
    assert runtime._tracking_bias_nm > .1
    t = now+1.01
    runtime.publish_policy(_command(round(t*1000),t,.008))
    frame = runtime.step(control_sequence=round(t*1000),control_timestamp_sec=t,
        measured_qpos_rad=(0.,0.,.001536,0.,0.,0.),measured_qvel_rad_s=ZERO,
        dt_sec=.01,control_qvel_rad_s=ZERO,control_qvel_window_span_sec=.08,
        tracking_recovery_permitted=True,stop_evidence=None)
    assert frame.overlay_torque_nm == ZERO
    assert frame.release_terminal and frame.release_failed
    assert not frame.release_completed
    assert frame.tracking_recovery is None
    assert runtime._tracking_bias_nm == 0.
    _ack_runtime(runtime,frame,t+.001)
    assert not runtime.release_completed  # Zero receipt is not stop evidence.
    for n in range(1,31):
        frame = _held_tick(runtime,t+n*.01,.001536,target=.008)
        assert frame.phase is B601StagedAssistPhase.RELEASING
        assert frame.release_terminal and frame.release_failed
        assert not frame.release_completed
        assert frame.overlay_torque_nm == ZERO
        assert frame.tracking_recovery is None
        assert runtime._tracking_bias_nm == 0.
        assert runtime._motion._brake_start is None  # No replacement clock.


@pytest.mark.parametrize("feedback_gain", [0.,40.])
@pytest.mark.parametrize("kinetic,inertia", [(.45,1.),(.05,.1)])
@pytest.mark.parametrize("target,arrival_margin", [(.008,0.),(.010,.0008)])
def test_tracking_recovery_declared_dry_friction_plant_retains_guards(kinetic,inertia,feedback_gain,target,arrival_margin):
    """Synthetic 1-DOF dynamics, not measured B601 friction or hardware proof."""
    import math

    from scripts.hardware.rebot_b601_rs_low_gain_probe import condition_torque

    runtime, now, frame = _recovery_track(feedback_gain=feedback_gain,target=target,arrival_margin=arrival_margin)
    estimator = B601WindowVelocityEstimator(.08)
    q, v, quantized, sent = .001536, 0., .001536, ZERO
    for n in range(10):
        estimator.update(timestamp_sec=now-.09+n*.01,qpos_rad=q,fallback_rad_s=0.)
    maximum_bias = 0.
    observed_exit = False
    first_brake_reason = None
    for n in range(1,1101):
        t = now+n*.01
        measured = round(q/.000384)*.000384
        raw = (measured-quantized)/.01
        speed = estimator.update(timestamp_sec=t,qpos_rad=measured,fallback_rad_s=raw)
        quantized = measured
        frame = _held_tick(runtime,t,measured,target=target,speed=speed,raw=raw,
            span=estimator.position_span_rad,ack=False)
        sent, _, _ = condition_torque(frame.overlay_torque_nm,sent,
            absolute_limits_nm=(1.,)*6,max_rate_nm_s=(28.8,)*6,dt_sec=.01)
        _ack_runtime(runtime,frame,t+.001,final=sent)
        maximum_bias = max(maximum_bias,abs(runtime._tracking_bias_nm))
        assert abs(sent[2]) <= 1.
        if frame.phase in {B601StagedAssistPhase.BRAKE,B601StagedAssistPhase.RELEASING}:
            if not observed_exit:
                first_brake_reason = frame.reason
            observed_exit = True
            assert runtime._tracking_bias_nm == 0.
        # Explicit static/kinetic friction with no hidden q-ref attraction.
        # Physics truth is never fed into the online velocity estimator.
        for _ in range(10):
            torque = sent[2] - .4*v
            if abs(v) < 1e-8 and abs(torque) <= .5:
                v = 0.
            else:
                friction = math.copysign(kinetic,v if abs(v)>1e-8 else torque)
                next_v = v+(torque-friction)/inertia*.001
                v = 0. if next_v*v<0 and abs(torque)<.5 else next_v
            q += v*.001
        if observed_exit and frame.phase in {B601StagedAssistPhase.SOFT_HOLD,B601StagedAssistPhase.PASSIVE}:
            break
    assert maximum_bias > .3  # Higher resistance is genuinely encountered.
    assert observed_exit, {"q":q,"window":speed,"velocity":v,
        "command":sent[2],"recovery":frame.tracking_recovery}
    assert runtime._tracking_bias_nm == 0.
    if kinetic == .45:
        if arrival_margin and feedback_gain == 0.:
            # Same-input negative control: enlarging the target WITHOUT the
            # existing explicit velocity feedback is not an accepted profile.
            assert abs(target-q) > .003
            assert first_brake_reason == "resolved_tracking_speed_braking"
        else:
            assert abs(target-q) <= .003, {"phase":frame.phase.value,"reason":frame.reason,
                "time":t-now,"q":q,"raw":raw,"window":speed,"velocity":v}


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("braking_rate,feedback_gain,inertia,kinetic,expect_arrival,governor,disturbance,continuous_limit", [
    (2., 40., 1., .45, False, False, False, None), (2., 40., 5., .49, True, False, False, None),
    (10., 10., 1., .45, True, False, False, None), (10., 10., 5., .49, True, False, False, None),
    (10., 10., 1., .45, False, False, True, None), (10., 10., 1., .45, True, True, True, None),
    (10., 10., 1., .45, True, True, False, None), (10., 10., 5., .49, True, True, False, None),
    (10., 10., 1., .45, True, True, False, .05), (10., 10., 5., .49, True, True, False, .05),
])
def test_normal_onset_tracking_and_fast_plant_counterexample(direction,inertia,kinetic,expect_arrival,braking_rate,feedback_gain,governor,disturbance,continuous_limit):
    """A quantized closed loop with declared friction, not an identified B601."""
    import math

    from scripts.hardware.rebot_b601_rs_low_gain_probe import condition_torque

    runtime = _held_runtime(direction, recovery=True, target=.02, pacing=True,
                            feedback_gain=feedback_gain, handoff="track", governor=governor,
                            continuous_limit=continuous_limit)
    runtime.config = replace(runtime.config, settle_error_rad=.00845,
        breakaway_max_duration_sec=3.,
        motion_control=replace(runtime.config.motion_control, max_reference_velocity_rad_s=.015,
                               braking_slew_nm_s=braking_rate, speed_governor=governor),
        breakaway_max_impulse_nm_sec=1.5, breakaway_probe=replace(
            runtime.config.breakaway_probe, max_trial_sec=3., max_command_exposure_nm_sec=1.5))
    runtime._motion = type(runtime._motion)(runtime.config.motion_control)
    estimator = B601WindowVelocityEstimator(.08)
    q, v, last_q = 0., 0., 0.
    sent = runtime._command_delta
    for i in range(10):
        estimator.update(timestamp_sec=1.07+i*.01, qpos_rad=0., fallback_rad_s=0.)
    handoffs, track_distance, handoff_q, first_brake = 0, 0., None, None
    disturbance_applied, final_brake, resume_count = False, None, 0
    for i in range(1, 2001):
        now = 1.16+i*.01
        measured = round(q/.000384)*.000384
        raw = (measured-last_q)/.01
        velocity = estimator.update(timestamp_sec=now, qpos_rad=measured, fallback_rad_s=raw)
        last_q = measured
        frame = _held_tick(runtime, now, direction*measured, direction=direction, target=.02,
            speed=direction*velocity, raw=direction*raw, span=estimator.position_span_rad, ack=False)
        sent, _, _ = condition_torque(frame.overlay_torque_nm, sent,
            absolute_limits_nm=(1.,)*6, max_rate_nm_s=(28.8,)*6, dt_sec=.01)
        _ack_runtime(runtime, frame, now+.001, final=sent)
        assert max(abs(x) for x in sent) <= 1.
        if frame.reason == "confirmed_onset_transferred_to_tracking":
            handoffs += 1
            handoff_q = q
            if disturbance:
                # A declared one-time plant velocity disturbance, never fed
                # directly to the controller or claimed as measured friction.
                v = direction*.024
                disturbance_applied = True
        if frame.motion is not None:
            resume_count = frame.motion.speed_governor_resume_count
        if frame.phase is B601StagedAssistPhase.BRAKE:
            final_brake = frame.reason
        if frame.phase is B601StagedAssistPhase.TRACK:
            track_distance = max(track_distance, direction*(q-handoff_q))
        if frame.phase in {B601StagedAssistPhase.BRAKE, B601StagedAssistPhase.RELEASING}:
            first_brake = first_brake or frame.reason
            if continuous_limit is not None and frame.motion.speed_governor_active:
                assert abs(runtime._tracking_bias_nm) <= 1.
                assert frame.tracking_recovery["bias_request_nm"] == 0.
            else:
                assert runtime._tracking_bias_nm == 0.
        for _ in range(10):
            effort = sent[2]-.4*v
            if abs(v) < 1e-8 and abs(effort) <= .5:
                v = 0.
            else:
                resistance = math.copysign(kinetic, v if abs(v)>1e-8 else effort)
                next_v = v+(effort-resistance)/inertia*.001
                v = 0. if next_v*v<0 and abs(effort)<.5 else next_v
            q += v*.001
        if first_brake and frame.phase in {B601StagedAssistPhase.SOFT_HOLD, B601StagedAssistPhase.PASSIVE}:
            break
    assert handoffs == 1, {"phase": frame.phase.value, "reason": frame.reason,
        "brake": first_brake, "q": q, "v": v, "probe": frame.breakaway_probe}
    if expect_arrival:
        assert track_distance > .005, {"progress": track_distance, "q":q, "phase":frame.phase.value}
        assert abs(direction*.02-q) <= .00845
        assert final_brake == "target_reached_braking"
        if governor and disturbance:
            assert resume_count >= 1
    else:
        # Removing the forced intermediate stop cannot guarantee arrival for
        # an unknown high-acceleration plant. Preserve this rejected case.
        assert first_brake == "resolved_tracking_speed_braking"
        assert abs(direction*.02-q) > .00845
        assert frame.phase is B601StagedAssistPhase.SOFT_HOLD
    assert disturbance_applied is disturbance


@pytest.mark.parametrize("direction", [-1,1])
def test_held_breakaway_quiet_plateau_resumes_without_reusing_position_or_budget(direction):
    runtime = _held_runtime(direction)
    frame = _held_tick(runtime, 1.17,.000768,direction=direction,speed=.008,span=.000768)
    held = frame.overlay_torque_nm[2]
    assert frame.reason == "held_probe_pending_actual_dispatch"
    started = runtime._breakaway_started_sec
    impulse = frame.breakaway_impulse_nm_sec
    for n in range(1, 14):
        frame = _held_tick(runtime, 1.17+n*.01,.000768,direction=direction)
        if n < 13:
            assert frame.overlay_torque_nm[2] == pytest.approx(held)
            assert frame.motion_onset_confirm_count == 0
    assert frame.reason == "held_probe_quiet_resume_same_budget"
    assert abs(frame.overlay_torque_nm[2]) > abs(held)
    assert frame.breakaway_impulse_nm_sec > impulse
    assert runtime._breakaway_started_sec == started
    frame = _held_tick(runtime,1.31,.000768,direction=direction)
    assert frame.reason == "bounded_breakaway_pulse"
    assert frame.breakaway_probe["stage"] == "quiet_resume"


@pytest.mark.parametrize("direction", [-1,1])
def test_held_breakaway_distinct_post_receipt_progress_withdraws_then_hands_off(direction):
    runtime = _held_runtime(direction)
    _held_tick(runtime,1.17,.000768,direction=direction,speed=.008,span=.000768)
    frame = _held_tick(runtime,1.18,.001152,direction=direction,speed=.01,raw=.0384,span=.001152)
    assert frame.motion_onset_confirm_count == 1
    frame = _held_tick(runtime,1.19,.001536,direction=direction,speed=.012,raw=.0384,span=.001536)
    assert frame.phase is B601StagedAssistPhase.RELEASING
    assert frame.reason == "held_probe_distinct_progress_confirmed"
    assert frame.overlay_torque_nm == ZERO
    assert frame.motion_onset_confirm_count == 2
    for n in range(1, 30):
        frame = _held_tick(runtime,1.19+n*.01,.001536,direction=direction)
        if frame.phase is B601StagedAssistPhase.TRACK:
            break
    assert frame.phase is B601StagedAssistPhase.TRACK
    assert frame.release_terminal is False


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("timeout", [None, .6])
def test_held_probe_late_plateau_has_explicit_observation_margin_without_budget_reset(direction, timeout):
    runtime = _held_runtime(direction, observation_timeout=timeout)
    _held_tick(runtime, 1.17, .000768, direction=direction)
    started = runtime._breakaway_started_sec
    impulse = runtime._command_impulse
    for n in range(1, 24):
        q = .000768 if n < 12 else .001152
        frame = _held_tick(runtime, 1.17+n*.01, q, direction=direction)
        if frame.release_terminal or frame.reason == "held_probe_quiet_resume_same_budget":
            break
    assert runtime._breakaway_started_sec == started
    assert runtime._command_impulse >= impulse
    assert frame.phase is not B601StagedAssistPhase.TRACK
    if timeout is None:
        assert frame.release_terminal and frame.reason == "held_probe_observation_timeout"
    else:
        assert not frame.release_terminal
        assert frame.reason == "held_probe_quiet_resume_same_budget"
        assert frame.motion_onset_confirm_count == 0


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("speed,expected", [(.01255, "breakaway"), (.0163, "releasing")])
def test_held_headroom_retains_real_progress_confirmation_and_speed_withdrawal(direction, speed, expected):
    runtime = _held_runtime(direction, recovery=True, target=.02, pacing=True,
        withdrawal_confirmation=True, handoff="track", velocity_headroom=.30)
    _held_tick(runtime, 1.17, .000768, direction=direction, target=.02)
    frame = _held_tick(runtime, 1.18, .001152, direction=direction, target=.02,
        speed=speed, raw=.0381, span=.001150)
    assert frame.phase.value == expected
    assert frame.motion_onset_confirm_count == 1
    if expected == "breakaway":
        # Repeated positions cannot become a second onset frontier.
        frame = _held_tick(runtime, 1.19, .001152, direction=direction, target=.02,
            speed=.01255, span=.001150)
        assert frame.motion_onset_confirm_count == 1
        frame = _held_tick(runtime, 1.20, .001536, direction=direction, target=.02,
            speed=.014, raw=.0381, span=.001536)
        assert frame.phase is B601StagedAssistPhase.TRACK
        assert frame.motion_onset_confirm_count == 2
    else:
        assert frame.overlay_torque_nm == ZERO
        assert frame.reason == "held_probe_soft_speed_confirmation_pending"


@pytest.mark.parametrize("cause", ["raw_speed", "human", "displacement"])
def test_held_speed_headroom_does_not_bypass_hard_or_human_gates(cause):
    runtime = _held_runtime(recovery=True, target=.02, pacing=True,
        withdrawal_confirmation=True, handoff="track", velocity_headroom=.30)
    frame = _held_tick(runtime, 1.17, .0031 if cause == "displacement" else 0.,
        target=.02, raw=.051 if cause == "raw_speed" else 0., human=cause == "human")
    assert frame.phase is not B601StagedAssistPhase.TRACK
    assert frame.overlay_torque_nm == ZERO


def _soft_pending_runtime(direction=1):
    runtime = _held_runtime(direction, recovery=True, target=.02, pacing=True,
                            withdrawal_confirmation=True)
    _held_tick(runtime, 1.17, .000768, direction=direction, target=.02)
    frame = _held_tick(runtime, 1.18, .001152, direction=direction, target=.02,
                       speed=.012604, raw=.0381, span=.001150)
    assert frame.reason == "held_probe_soft_speed_confirmation_pending"
    assert not frame.release_terminal
    assert frame.overlay_torque_nm == ZERO
    assert frame.motion_onset_confirm_count == 1
    return runtime


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("tail_axis", [2, 4])
def test_soft_withdrawal_new_frontier_confirms_only_after_six_axis_zero_and_stop(direction, tail_axis):
    runtime = _soft_pending_runtime(direction)
    clock = runtime._breakaway_started_sec
    tail = tuple(.001 if i == tail_axis else 0. for i in range(6))
    frame = _held_tick(runtime, 1.19, .001536, direction=direction, target=.02,
                       speed=.0162, raw=.0381, span=.001534, final=tail)
    assert frame.reason == "held_probe_post_withdrawal_progress_confirmed"
    assert frame.motion_onset_confirm_count == 2
    assert frame.phase is B601StagedAssistPhase.RELEASING
    for n in range(1, 40):
        now = 1.19+n*.01
        frame = _held_tick(runtime, now, .001536, direction=direction, target=.02,
                           ack=False)
        assert frame.phase is B601StagedAssistPhase.RELEASING
        assert runtime._phase is not B601StagedAssistPhase.DROP
        if frame.motion.phase == "soft_hold":
            _ack_runtime(runtime, frame, now+.001, final=ZERO)
            break
        _ack_runtime(runtime, frame, now+.001, final=tail)
    else:
        pytest.fail(str(frame))
    assert runtime._phase is B601StagedAssistPhase.DROP
    frame = _held_tick(runtime, now+.01, .001536, direction=direction, target=.02)
    assert frame.phase is B601StagedAssistPhase.TRACK
    assert frame.motion.target_position_rad == pytest.approx(direction*.02)
    assert runtime._breakaway_started_sec == clock


@pytest.mark.parametrize("case", ["plateau", "dither", "return", "raw_speed", "human", "permission", "target_change", "cumulative", "expiry"])
def test_soft_withdrawal_confirmation_never_grants_track_without_safe_new_progress(case):
    runtime = _soft_pending_runtime()
    started = runtime._breakaway_started_sec
    q = .001152
    terminal = False
    for n in range(1, 150):
        now = 1.18+n*.01
        if case == "dither":
            q = .000768 if n % 2 else .001152
        elif case == "return":
            q = .0001
        elif case in {"permission", "target_change"}:
            q = .001536
        elif case == "cumulative":
            q = .0031
        args = {"target": .02}
        if n == 1:
            if case == "raw_speed": args["raw"] = .051
            if case == "human": args["human"] = True
            if case == "permission": args["permission"] = False
            if case == "target_change": args["target"] = .06
        if case == "expiry":
            # Retain nonzero actual tails until the ORIGINAL pulse deadline.
            args["final"] = (0.,0.,.001,0.,0.,0.)
        frame = _held_tick(runtime, now, q, **args)
        assert runtime._breakaway_started_sec in {started, None}
        if runtime._breakaway_started_sec is None:
            assert runtime.release_completed
        assert frame.phase not in {B601StagedAssistPhase.DROP, B601StagedAssistPhase.TRACK}
        assert runtime._phase not in {B601StagedAssistPhase.DROP, B601StagedAssistPhase.TRACK}
        terminal |= frame.release_terminal
        if terminal:
            break
    assert terminal


def test_soft_withdrawal_buffered_frontier_is_not_recounted_when_it_becomes_fresh():
    runtime = _soft_pending_runtime()
    # A receipt completed after the next feedback timestamp cannot make that
    # already-buffered position new post-dispatch progress.
    _held_tick(runtime, 1.19, .001152, target=.02, ack_delay=.011)
    frame = _held_tick(runtime, 1.20, .001536, target=.02, ack_delay=.004)
    assert frame.motion_onset_confirm_count == 1
    frame = _held_tick(runtime, 1.21, .001536, target=.02)
    assert frame.motion_onset_confirm_count == 1
    frame = _held_tick(runtime, 1.22, .00192, target=.02)
    assert frame.motion_onset_confirm_count == 2


@pytest.mark.parametrize("stop", ["withdraw", "clutch", "target"])
def test_soft_withdrawal_last_zero_receipt_cannot_override_concurrent_cancel(stop):
    runtime = _soft_pending_runtime()
    tail = (0.,0.,.001,0.,0.,0.)
    _held_tick(runtime, 1.19, .001536, target=.02, final=tail)
    for n in range(1, 40):
        now = 1.19+n*.01
        frame = _held_tick(runtime, now, .001536, target=.02, ack=False)
        if frame.motion.phase == "soft_hold":
            if stop == "withdraw":
                runtime.request_withdrawal("operator_withdraw")
            elif stop == "clutch":
                runtime.set_policy_clutch(True)
            else:
                runtime.publish_policy(_command(round(now*1000)+1,now,.06))
            _ack_runtime(runtime, frame, now+.001, final=ZERO)
            break
        _ack_runtime(runtime, frame, now+.001, final=tail)
    else:
        pytest.fail(str(frame))
    assert runtime._release_terminal
    assert runtime._phase is B601StagedAssistPhase.RELEASING
    assert not runtime._release_transition_ready


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_soft_withdrawal_confirmation_rejects_non_boolean_values(value):
    with pytest.raises(TypeError, match="soft withdrawal"):
        replace(_held_runtime().config, breakaway_soft_withdrawal_confirmation=value)


def test_soft_withdrawal_confirmation_requires_existing_paced_recovery():
    with pytest.raises(ValueError, match="paced tracking recovery"):
        replace(_held_runtime().config, breakaway_soft_withdrawal_confirmation=True)


@pytest.mark.parametrize("case", ["raw_speed", "soft_speed", "opposing", "cumulative", "command_tail", "human", "missing_ack"])
def test_held_breakaway_keeps_guard_and_receipt_boundaries(case):
    runtime = _held_runtime()
    _held_tick(runtime,1.17,.000768)
    if case == "missing_ack":
        _held_tick(runtime,1.18,.000768,ack=False)
        with pytest.raises(ValueError,match="dispatch receipt"):
            _held_tick(runtime,1.19,.000768)
        return
    args = {
        "raw_speed": {"raw": .051},
        "soft_speed": {"speed": .025, "span": .001},
        "opposing": {},
        "cumulative": {},
        "command_tail": {"final": (0.,0.,.12,0.,0.,0.)},
        "human": {"human": True},
    }[case]
    q = -.0001 if case == "opposing" else .0031 if case == "cumulative" else .000768
    frame = _held_tick(runtime,1.18,q,**args)
    if case in {"command_tail", "human"}:
        frame = _held_tick(runtime,1.19,q)
    assert frame.release_terminal is True
    assert frame.overlay_torque_nm[2] <= 0


def test_held_breakaway_repeated_position_dither_cannot_confirm_or_renew_duration():
    runtime = _held_runtime()
    _held_tick(runtime,1.17,.000768)
    for n in range(1, 25):
        # Revisit the same frontier, never add a second distinct one.
        q = .001152 if n%2 else .000768
        frame = _held_tick(runtime,1.17+n*.01,q,span=.000384)
        if frame.release_terminal:
            break
    assert not frame.release_terminal
    assert frame.breakaway_probe["stage"] == "quiet_resume"
    assert frame.motion_onset_confirm_count == 0
    started = runtime._breakaway_started_sec
    for n in range(25, 140):
        frame = _held_tick(runtime,1.17+n*.01,.000768)
        if frame.release_terminal:
            break
    assert frame.release_terminal
    assert runtime._breakaway_started_sec == started
    assert frame.phase is not B601StagedAssistPhase.TRACK


def test_held_breakaway_buffered_feedback_never_counts_as_new_post_dispatch_progress():
    runtime = _held_runtime()
    _held_tick(runtime,1.17,.000768,ack_delay=.011)
    frame = _held_tick(runtime,1.18,.001152,ack_delay=.004)
    assert frame.motion_onset_confirm_count == 0
    frame = _held_tick(runtime,1.19,.001152)
    assert frame.motion_onset_confirm_count == 0
    frame = _held_tick(runtime,1.20,.001536)
    assert frame.motion_onset_confirm_count == 1


@pytest.mark.parametrize("direction", [-1,1])
@pytest.mark.parametrize("tail_axis", [2,4])
def test_held_breakaway_return_to_anchor_at_first_final_zero_ack_vetoes_handoff(direction,tail_axis):
    runtime = _held_runtime(direction)
    _held_tick(runtime,1.17,.000768,direction=direction)
    _held_tick(runtime,1.18,.001152,direction=direction,speed=.01)
    _held_tick(runtime,1.19,.001536,direction=direction,speed=.012)
    assert runtime._breakaway_probe_confirmed
    tail = tuple(direction*.001 if i==tail_axis else 0. for i in range(6))
    for n in range(1,50):
        now=1.19+n*.01
        q=max(.000384,.001536-n*.000384)
        frame=_held_tick(runtime,now,q,direction=direction,ack=False)
        if frame.motion and frame.motion.phase == "soft_hold":
            assert runtime._final_zero_confirmed is False
            assert frame.motion_onset_confirm_count >= 2
            # Final conditioner has only just reached zero. The veto must
            # run in this receipt, not depend on a prior zero receipt.
            _ack_runtime(runtime,frame,now+.001,final=ZERO)
            break
        _ack_runtime(runtime,frame,now+.001,final=tail)
    else:
        pytest.fail("withdrawal never reached the native stop dwell")
    assert runtime._release_terminal is True
    assert runtime._phase is not B601StagedAssistPhase.DROP
    assert runtime._release_reason == "held_probe_progress_returned_after_withdrawal"


@pytest.mark.parametrize("invalid", ["duration", "impulse", "resolution", "joint", "handoff"])
def test_held_breakaway_configuration_does_not_expand_existing_envelope(invalid):
    runtime = _held_runtime()
    updates = {
        "duration": {"breakaway_max_duration_sec": .5},
        "impulse": {"breakaway_max_impulse_nm_sec": .2},
        "resolution": {"motion_control": replace(runtime.config.motion_control,position_resolution_rad=None,responsive_tracking_brake=False)},
        # Workspace acquisition now explicitly supports J2 and J3, not other axes.
        "joint": {"joint_name": "joint1"},
        "handoff": {"continuous_motion_handoff": True},
    }[invalid]
    with pytest.raises(ValueError):
        replace(runtime.config,**updates)


@pytest.mark.parametrize("direction", [-1,1])
def test_responsive_torque_transition_preserves_slow_rise_but_quickly_withdraws(
    direction,
):
    runtime = _active_runtime(
        _staged_config(
            motion_control=B601JointMotionConfig(
                windowed_soft_velocity=True,
                position_resolution_rad=0.0004,
                responsive_tracking_brake=True,
                max_control_gap_sec=0.04,
            ),
            motion_onset_displacement_rad=0.0004,
            motion_onset_confirm_displacement_rad=0.0007,
            tracking_slew_nm_s=0.15,
        )
    )
    runtime._phase = B601StagedAssistPhase.TRACK
    runtime._motion_frame = B601JointMotionFrame(
        "track", "bounded_reference_tracking", 0, 0, direction * 0.01, 0, 0, 0
    )
    runtime._command_delta = ZERO
    runtime._final_zero_confirmed = True
    runtime._previous_overlay_nm = 0.0
    rise = runtime._responsive_tracking_output(direction * 0.04, direction * 0.01, 0.01)
    assert rise == pytest.approx(direction * 0.0015)
    runtime._previous_overlay_nm = direction * 0.0355974
    runtime._command_delta = (0.0, 0.0, direction * 0.0355974, 0.0, 0.0, 0.0)
    runtime._final_zero_confirmed = False
    runtime._last_output_zero = False
    reduced = runtime._responsive_tracking_output(
        direction * 0.0111399, direction * 0.0144844, 0.01
    )
    assert reduced == pytest.approx(direction * 0.0155974)
    assert abs(reduced - runtime._previous_overlay_nm) <= 2 * 0.01 + 1e-12
    assert runtime._motion_frame.torque_transition_rate_nm_s == 2.0


@pytest.mark.parametrize("direction", [-1, 1])
def test_responsive_reversal_waits_for_complete_native_zero_before_fast_braking(
    direction,
):
    runtime = _active_runtime(
        _staged_config(
            motion_control=B601JointMotionConfig(
                windowed_soft_velocity=True,
                position_resolution_rad=0.0004,
                responsive_tracking_brake=True,
                max_control_gap_sec=0.04,
            ),
            motion_onset_displacement_rad=0.0004,
            motion_onset_confirm_displacement_rad=0.0007,
            tracking_slew_nm_s=0.15,
        )
    )
    runtime._phase = B601StagedAssistPhase.TRACK
    runtime._motion_frame = B601JointMotionFrame(
        "track", "bounded_reference_tracking", 0, 0, direction * 0.023253, 0, 0, 0
    )
    runtime._previous_overlay_nm = direction * 0.0216059
    runtime._command_delta = (0.0, 0.0, direction * 0.0216059, 0.0, 0.0, 0.0)
    runtime._final_zero_confirmed = False
    runtime._last_output_zero = False
    runtime._stop_evidence = B601WindowStopEvidence(
        1.0,
        ZERO,
        ZERO,
        (0.0, 0.0, direction * 0.023253, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.001534, 0.0, 0.0, 0.0),
        0.08,
        0.05,
    )
    first = runtime._responsive_tracking_output(
        -direction * 0.0041173, direction * 0.023253, 0.01
    )
    assert first == pytest.approx(direction * 0.0016059)
    runtime._previous_overlay_nm = first
    second = runtime._responsive_tracking_output(
        -direction * 0.0115245, direction * 0.0269644, 0.01
    )
    assert second == 0
    runtime._previous_overlay_nm = 0.0
    runtime._last_output_zero = True
    # A final conditioner can retain torque even after the proposed zero.
    assert (
        runtime._responsive_tracking_output(
            -direction * 0.0115245, direction * 0.0269644, 0.01
        )
        == 0
    )
    runtime._command_delta = ZERO
    runtime._final_zero_confirmed = True
    braking = runtime._responsive_tracking_output(
        -direction * 0.0115245, direction * 0.0269644, 0.01
    )
    assert braking == pytest.approx(-direction * 0.0115245)
    assert (
        runtime._motion_frame.torque_transition_reason == "fast_resolved_motion_braking"
    )


def test_responsive_tracking_refuses_withdrawal_slower_than_guidance_rise():
    with pytest.raises(ValueError, match="withdrawal rate"):
        _staged_config(
            motion_control=B601JointMotionConfig(
                windowed_soft_velocity=True,
                position_resolution_rad=0.0004,
                responsive_tracking_brake=True,
                max_control_gap_sec=0.04,
            ),
            motion_onset_displacement_rad=0.0004,
            motion_onset_confirm_displacement_rad=0.0007,
            tracking_slew_nm_s=2.1,
        )


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("resolution", [None, 0.0004])
def test_acquisition_single_count_return_does_not_end_pulse_or_claim_onset(
    direction, resolution
):
    config = _staged_config(
        motion_control=B601JointMotionConfig(
            windowed_soft_velocity=True,
            position_resolution_rad=resolution,
            max_control_gap_sec=0.04,
        ),
        motion_onset_displacement_rad=0.0004,
        motion_onset_confirm_displacement_rad=0.0007,
    )
    runtime = B601StagedSingleJointAssistRuntime(_shared_config(), config)
    runtime.publish_policy(_command(1, 1.0, target=direction * 0.1))
    _step(runtime, sequence=1, timestamp=1.01, dt=0.01)
    runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06, dt=0.01)
    for seq, q, raw in [
        (3, -direction * 0.000383496, -direction * 0.0383496),
        (4, 0.0, direction * 0.0383496),
    ]:
        now = 1.06 + (seq - 2) * 0.01
        positions = (0, 0, q, 0, 0, 0)
        velocities = (0, 0, raw, 0, 0, 0)
        soft = (0, 0, direction * 0.0006, 0, 0, 0)
        evidence = B601WindowStopEvidence(
            now, positions, velocities, soft, (0, 0, 0.000383496, 0, 0, 0), 0.08, 0.05
        )
        frame = runtime.step(
            control_sequence=seq,
            control_timestamp_sec=now,
            measured_qpos_rad=positions,
            measured_qvel_rad_s=velocities,
            dt_sec=0.01,
            control_qvel_rad_s=soft,
            control_qvel_window_span_sec=0.08,
            stop_evidence=evidence,
        )
        _ack_runtime(runtime, frame, now)
    assert frame.motion_onset_confirm_count == 0
    if resolution is None:
        assert frame.phase is B601StagedAssistPhase.RELEASING
    else:
        assert frame.phase is B601StagedAssistPhase.BREAKAWAY
        assert frame.overlay_torque_nm[2] * direction > 0
        # Resolved progress still withdraws at the original displacement gate.
        now += 0.01
        positions = (0, 0, direction * 0.0008, 0, 0, 0)
        frame = runtime.step(
            control_sequence=5,
            control_timestamp_sec=now,
            measured_qpos_rad=positions,
            measured_qvel_rad_s=ZERO,
            dt_sec=0.01,
            control_qvel_rad_s=soft,
            control_qvel_window_span_sec=0.08,
            stop_evidence=B601WindowStopEvidence(
                now, positions, ZERO, soft, (0, 0, 0.0008, 0, 0, 0), 0.08, 0.05
            ),
        )
        assert frame.phase is B601StagedAssistPhase.RELEASING
        assert frame.motion_onset_confirm_count == 1
        assert frame.overlay_torque_nm == ZERO


@pytest.mark.parametrize("invalid", ["missing", "cold", "raw_fast", "stale"])
def test_acquisition_onset_requires_same_tick_window_and_keeps_raw_guard(invalid):
    runtime = _active_runtime(
        _staged_config(
            motion_control=B601JointMotionConfig(
                windowed_soft_velocity=True,
                position_resolution_rad=0.0004,
                max_control_gap_sec=0.04,
            ),
            motion_onset_displacement_rad=0.0004,
            motion_onset_confirm_displacement_rad=0.0007,
        )
    )
    _step(runtime, sequence=1, timestamp=1.01, dt=0.01)
    runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06, dt=0.01)
    velocities = (0, 0, 0.051, 0, 0, 0) if invalid == "raw_fast" else ZERO
    evidence = (
        None
        if invalid == "missing"
        else B601WindowStopEvidence(
            1.06 if invalid == "stale" else 1.07,
            ZERO,
            velocities,
            ZERO,
            ZERO,
            0.02 if invalid == "cold" else 0.08,
            0.05,
        )
    )
    kwargs = {
        "control_sequence": 3,
        "control_timestamp_sec": 1.07,
        "measured_qpos_rad": ZERO,
        "measured_qvel_rad_s": velocities,
        "dt_sec": 0.01,
        "control_qvel_rad_s": ZERO,
        "control_qvel_window_span_sec": 0.08,
        "stop_evidence": evidence,
    }
    if invalid == "stale":
        with pytest.raises(ValueError, match="match this six-axis"):
            runtime.step(**kwargs)
    else:
        frame = runtime.step(**kwargs)
        assert frame.phase is B601StagedAssistPhase.RELEASING
        assert frame.release_terminal
        assert frame.overlay_torque_nm==ZERO


def _continuous_runtime(**overrides):
    values = {
        "motion_control": B601JointMotionConfig(
            windowed_soft_velocity=True, max_control_gap_sec=.04,
            drive_driver_kd_nm_s_per_rad=(.4,.6,.4,.6,.2,.1),
        ),
        "motion_onset_displacement_rad": .0004,
        "motion_onset_confirm_displacement_rad": .0007,
        "continuous_motion_handoff": True,
    }
    values.update(overrides)
    runtime = _active_runtime(_staged_config(**values))
    _step(runtime, sequence=1, timestamp=1.01, dt=.01)
    assert runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06, dt=.01)
    return runtime


def _continuous_tick(runtime, seq, now, *, q=.0008, raw=.01, soft=.01, final=ZERO, ack=True):
    frame = runtime.step(
        control_sequence=seq, control_timestamp_sec=now,
        measured_qpos_rad=(0.,0.,q,0.,0.,0.), measured_qvel_rad_s=(0.,0.,raw,0.,0.,0.),
        dt_sec=.01, control_qvel_rad_s=(0.,0.,soft,0.,0.,0.), control_qvel_window_span_sec=.08,
    )
    if ack:
        _ack_runtime(runtime, frame, now, final=final)
    return frame


def _capturing_runtime(**overrides):
    runtime = _continuous_runtime(capture_overlay_cap_nm=.10, **overrides)
    frame = _continuous_tick(runtime, 3, 1.07, q=.0004, raw=.04, soft=.006)
    assert not frame.motion_onset_confirm_count
    assert runtime.phase is B601StagedAssistPhase.CAPTURE
    return runtime


def _motion_timeout_candidate(*, ack_timeout=True, timeout_final=ZERO):
    from robot_motion.rebot_b601_rs.joint_motion import B601WindowVelocityEstimator

    runtime = _continuous_runtime()
    _continuous_tick(runtime, 3, 1.07, q=.0004, raw=.04, soft=.006)
    _continuous_tick(runtime, 4, 1.08, q=.0008, raw=.04, soft=.007)
    _continuous_tick(runtime, 5, 1.09, q=.0012, raw=.04, soft=.008)
    assert runtime.phase is B601StagedAssistPhase.DROP
    estimator = B601WindowVelocityEstimator(.08)
    for i in range(9):
        estimator.update(timestamp_sec=1.01+i*.01, qpos_rad=.0012, fallback_rad_s=0.)
    previous_q, brake_at = .0012, None
    for i in range(1, 701):
        now, seq = 1.09+i*.01, i+5
        q = .0012+(i%2)*.000384
        raw = (q-previous_q)/.01
        soft = estimator.update(timestamp_sec=now, qpos_rad=q, fallback_rad_s=0.)
        assert estimator.ready
        assert runtime.publish_policy(_command(i+1, now)).accepted
        frame = _continuous_tick(runtime, seq, now, q=q, raw=raw, soft=soft, ack=False)
        if frame.motion is not None and frame.motion.phase == "brake" and brake_at is None:
            brake_at = now
        expired = frame.motion is not None and frame.motion.reason == "braking_timeout_assist_withdrawn"
        if not expired or ack_timeout:
            _ack_runtime(runtime, frame, now, final=timeout_final if expired else ZERO)
        if expired:
            assert brake_at is not None
            assert now-brake_at == pytest.approx(
                runtime.config.motion_control.braking_timeout_sec + .01
            )
            return runtime, frame, now, seq, brake_at
        previous_q = q
    pytest.fail("causal single-count feedback must expose the original motion timeout")


@pytest.mark.parametrize("tail", [ZERO, (0., .001, .02, 0., 0., 0.)])
def test_offline_motion_timeout_enters_failed_release_without_reset_or_gain_drop(tail):
    runtime, frame, _, _, _ = _motion_timeout_candidate(timeout_final=tail)
    assert frame.phase is B601StagedAssistPhase.RELEASING
    assert frame.release_failed and frame.release_terminal
    assert not frame.release_completed and not frame.release_restoring_base_damping
    assert frame.overlay_torque_nm == ZERO
    assert frame.drive_driver_kd_nm_s_per_rad == runtime.config.motion_control.drive_driver_kd_nm_s_per_rad


@pytest.mark.parametrize("action", ["fresh_target", "new_target", "clutch", "withdraw", "stop"])
def test_offline_motion_timeout_cannot_rearm_or_restart_deadline_after_stationary_receipts(action):
    runtime, first, now, seq, brake_at = _motion_timeout_candidate()
    assert first.release_failed
    for i in range(1, 61):
        t = now+i*.01
        assert runtime.publish_policy(_command(seq+i, t, target=-.1 if action == "new_target" else .1)).accepted
        if action == "clutch":
            runtime.set_policy_clutch(True)
        elif action == "withdraw":
            runtime.request_withdrawal("operator_withdrawal")
        elif action == "stop":
            runtime.request_stop_and_hold()
        frame = _continuous_tick(runtime, seq+i, t, q=.0012, raw=0., soft=0.)
        assert frame.release_failed and frame.release_terminal
        assert frame.phase is B601StagedAssistPhase.RELEASING
        assert frame.overlay_torque_nm == ZERO
        assert not frame.release_completed and not frame.release_restoring_base_damping
        assert frame.motion.brake_elapsed_sec == pytest.approx(t-brake_at)
        assert frame.drive_driver_kd_nm_s_per_rad == runtime.config.motion_control.drive_driver_kd_nm_s_per_rad
        assert not runtime.authorize_breakaway(sequence=seq+i)
    assert runtime.release_completed is False


def test_offline_failed_motion_still_requires_complete_timely_owner_dispatch():
    runtime, frame, now, seq, _ = _motion_timeout_candidate(ack_timeout=False)
    assert frame.release_failed
    with pytest.raises(ValueError, match="watchdog"):
        _ack_runtime(runtime, frame, now+runtime.config.motion_control.max_control_gap_sec+.001)
    with pytest.raises(ValueError, match="conditioned dispatch receipt"):
        _continuous_tick(runtime, seq+1, now+.01, q=.0012, raw=0., soft=0., ack=False)


@pytest.mark.parametrize("cap", [True, -.1, .100001, float("nan"), float("inf")])
def test_capture_cap_is_explicit_bounded_and_default_off(cap):
    with pytest.raises((TypeError, ValueError)):
        _continuous_runtime(capture_overlay_cap_nm=cap)
    assert _staged_config().capture_overlay_cap_nm == 0
    with pytest.raises(ValueError, match="offline continuous"):
        _staged_config(capture_overlay_cap_nm=.10)


@pytest.mark.parametrize("tail", [(0.,0.,.01,0.,0.,0.), (0.,.001,0.,0.,0.,0.)])
def test_capture_requires_complete_final_zero_before_any_low_force(tail):
    runtime = _continuous_runtime(capture_overlay_cap_nm=.10)
    for seq in (3, 4):
        frame = _continuous_tick(runtime, seq, 1.04+seq*.01, q=.0004, raw=.04, soft=.006, final=tail)
        assert frame.overlay_torque_nm == ZERO
        assert runtime.phase is B601StagedAssistPhase.RELEASING
    _continuous_tick(runtime, 5, 1.09, q=.0004)
    assert runtime.phase is B601StagedAssistPhase.CAPTURE


def test_capture_retains_transactional_reference_at_confirmed_track_transition():
    runtime = _capturing_runtime()
    first = _continuous_tick(runtime, 4, 1.08, q=.0008, raw=.04, soft=.006)
    assert first.phase is B601StagedAssistPhase.CAPTURE
    assert first.motion_onset_confirm_count == 1
    second = _continuous_tick(runtime, 5, 1.09, q=.0012, raw=.04, soft=.007, ack=False)
    assert second.motion_onset_confirm_count == 2
    assert runtime.phase is B601StagedAssistPhase.CAPTURE
    _ack_runtime(runtime, second, 1.09, final=second.overlay_torque_nm)
    assert runtime.phase is B601StagedAssistPhase.TRACK
    tracking = _continuous_tick(runtime, 6, 1.10, q=.0012)
    assert tracking.motion.q_reference_rad < .0012  # No reset to current q.
    assert tracking.motion.qvel_reference_rad_s > second.motion.qvel_reference_rad_s
    assert abs(tracking.motion.qvel_reference_rad_s-second.motion.qvel_reference_rad_s) <= .03*.01+1e-9
    assert all(v > 0 for v in tracking.drive_driver_kd_nm_s_per_rad)


def test_capture_count_jitter_does_not_grant_tracking_or_restart_high_pulse():
    runtime = _capturing_runtime(breakaway_max_duration_sec=.5)
    previous = 0.
    for seq in range(4, 20):
        frame = _continuous_tick(runtime, seq, 1.04+seq*.01, q=.000384*(seq%2), raw=0., soft=0.)
        assert frame.phase is B601StagedAssistPhase.CAPTURE
        assert frame.motion_onset_confirm_count == 0
        assert abs(frame.overlay_torque_nm[2]) <= .10
        assert abs(frame.overlay_torque_nm[2]-previous) <= 2*.01+1e-9
        previous = frame.overlay_torque_nm[2]


@pytest.mark.parametrize("when", ["before_tick", "before_receipt"])
def test_capture_stop_vetoes_confirmed_tracking_and_preserves_terminal_brake(when):
    runtime = _capturing_runtime()
    _continuous_tick(runtime, 4, 1.08, q=.0008)
    if when == "before_tick":
        runtime.request_stop_and_hold()
    frame = _continuous_tick(runtime, 5, 1.09, q=.0012, ack=False)
    if when == "before_receipt":
        assert frame.motion_onset_confirm_count == 2
        runtime.request_stop_and_hold()
    _ack_runtime(runtime, frame, 1.09, final=frame.overlay_torque_nm)
    assert runtime.phase is not B601StagedAssistPhase.TRACK
    withdrawn = _continuous_tick(runtime, 6, 1.10, q=.0012, raw=0., soft=0.)
    assert withdrawn.phase is B601StagedAssistPhase.RELEASING
    assert withdrawn.release_terminal
    assert withdrawn.reason == "operator_stop_during_low_force_capture"


def test_capture_ack_after_original_deadline_cannot_grant_tracking():
    runtime = _capturing_runtime(breakaway_max_duration_sec=.04)
    _continuous_tick(runtime, 4, 1.08, q=.0008)
    frame = _continuous_tick(runtime, 5, 1.09, q=.0012, ack=False)
    assert frame.motion_onset_confirm_count == 2
    _ack_runtime(runtime, frame, 1.101, final=frame.overlay_torque_nm)
    assert runtime.phase is not B601StagedAssistPhase.TRACK
    withdrawn = _continuous_tick(runtime, 6, 1.11, q=.0012, raw=0., soft=0.)
    assert withdrawn.release_terminal
    assert "duration_exhausted" in withdrawn.reason


def test_capture_preserves_conditioned_budget_reserve():
    runtime = _capturing_runtime(breakaway_max_impulse_nm_sec=.003)
    frame = _continuous_tick(runtime, 4, 1.08, q=.0004, raw=0., soft=0.)
    assert frame.phase is B601StagedAssistPhase.RELEASING
    assert frame.release_terminal
    assert frame.reason == "capture_conditioned_budget_withdrawal"
    assert frame.overlay_torque_nm == ZERO


def test_capture_out_of_cap_final_receipt_cannot_grant_tracking():
    runtime = _capturing_runtime()
    _continuous_tick(runtime, 4, 1.08, q=.0008)
    frame = _continuous_tick(runtime, 5, 1.09, q=.0012, ack=False)
    _ack_runtime(runtime, frame, 1.09, final=(0.,0.,.100001,0.,0.,0.))
    assert runtime.phase is not B601StagedAssistPhase.TRACK
    withdrawn = _continuous_tick(runtime, 6, 1.10, q=.0012, raw=0., soft=0.)
    assert withdrawn.release_terminal
    assert withdrawn.reason == "capture_conditioned_overlay_cap_exceeded"


def test_new_target_during_capture_withdraws_old_intent_instead_of_rearming():
    runtime = _capturing_runtime()
    assert runtime.publish_policy(_command(2, 1.08, target=-.1)).accepted
    frame = _continuous_tick(runtime, 4, 1.08, q=.0004)
    assert frame.phase is B601StagedAssistPhase.RELEASING
    assert frame.release_terminal
    assert frame.reason == "capture_target_changed_withdrawal"
    assert not runtime.authorize_breakaway(sequence=2)


@pytest.mark.parametrize("event", ["clutch", "withdrawal", "new_target", "clear", "stale", "future"])
def test_capture_receipt_cannot_grant_track_after_intent_withdrawal(event):
    runtime = _capturing_runtime(breakaway_max_duration_sec=2)
    _continuous_tick(runtime, 4, 1.08, q=.0008)
    if event == "stale":
        shared = replace(runtime.shared_config, policy_stale_timeout_sec=.10)
        runtime = B601StagedSingleJointAssistRuntime(shared, runtime.config)
        runtime.publish_policy(_command(1, 1.0))
        _continuous_tick(runtime, 1, 1.01, q=0.)
        assert runtime.authorize_breakaway(sequence=1)
        _continuous_tick(runtime, 2, 1.06, q=0.)
        _continuous_tick(runtime, 3, 1.07, q=.0004, raw=.04, soft=.006)
        _continuous_tick(runtime, 4, 1.08, q=.0008)
    frame = _continuous_tick(runtime, 5, 1.09, q=.0012, ack=False)
    assert frame.motion_onset_confirm_count == 2
    if event == "clutch":
        runtime.set_policy_clutch(True)
    elif event == "withdrawal":
        runtime.request_withdrawal("operator_withdrawal")
    elif event == "new_target":
        runtime.publish_policy(_command(2, 1.09, target=-.1))
    elif event == "clear":
        runtime.clear_policy()
    elif event == "future":
        runtime.publish_policy(_command(2, 10., target=.1))
    _ack_runtime(runtime, frame, 1.101 if event == "stale" else 1.09, final=frame.overlay_torque_nm)
    assert runtime.phase is not B601StagedAssistPhase.TRACK
    withdrawn = _continuous_tick(runtime, 6, 1.11, q=.0012, raw=0., soft=0.)
    assert withdrawn.release_terminal
    assert withdrawn.phase is B601StagedAssistPhase.RELEASING
    if event == "future":
        assert withdrawn.reason == "policy_command_from_future"


@pytest.mark.parametrize("invalid", ["gap", "window"])
def test_capture_does_not_bypass_existing_feedback_and_gap_gates(invalid):
    runtime = _capturing_runtime()
    frame = runtime.step(
        control_sequence=4, control_timestamp_sec=1.08,
        measured_qpos_rad=(0.,0.,.0008,0.,0.,0.), measured_qvel_rad_s=ZERO,
        dt_sec=.041 if invalid == "gap" else .01,
        control_qvel_rad_s=ZERO, control_qvel_window_span_sec=.01 if invalid == "window" else .08,
    )
    assert frame.release_failed
    assert frame.release_terminal
    assert frame.overlay_torque_nm == ZERO
    _ack_runtime(runtime, frame, 1.08)
    assert runtime.phase is B601StagedAssistPhase.RELEASING


def _withdrawal_trace(completion_index=19):
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import _withdrawal_evaluation

    motion = B601JointMotionConfig(windowed_soft_velocity=True, max_control_gap_sec=.04,
        drive_driver_kd_nm_s_per_rad=(.4,.6,.4,.6,.2,.1))
    staged = _staged_config(motion_control=motion, motion_onset_displacement_rad=.0004,
        motion_onset_confirm_displacement_rad=.0007, settle_velocity_rad_s=.005, settle_dwell_sec=.15)
    rows = []
    for i in range(completion_index+13):
        now = 4.99+i*.01
        tail = .06 if i == 0 else .04 if i == 1 else .02 if i == 2 else 0.
        speed = .01 if i <= 2 else 0.
        completed = i == completion_index
        active = i < completion_index
        rows.append({
            "time_sec":now, "phase":"releasing" if active or completed else "human_override",
            "qpos_rad":ZERO, "raw_qvel_rad_s":(0.,0.,speed,0.,0.,0.),
            "control_qvel_rad_s":(0.,0.,speed,0.,0.,0.), "overlay_nm":-2*speed,
            "release_completed":completed,
            "motion":None if not active else {"phase":"brake", "position_torque_nm":0.,
                "qvel_reference_rad_s":0., "requested_torque_nm":-2*speed, "control_qvel_rad_s":speed},
            "dispatch":{"sequence":i+1,"feedback_time_sec":now+.001,
                "host_command_torque_nm":(0.,0.,.3+tail,0.,0.,0.),
                "conditioned_overlay_nm":(0.,0.,tail,0.,0.,0.),
                "torque_breakdown":{"gravity_torque_nm":(0.,0.,.3,0.,0.,0.)},
                "driver_kd_nm_s_per_rad":motion.drive_driver_kd_nm_s_per_rad if active else ZERO,
                "command_applied":True,"plant_advanced":True},
        })
    def evaluate():
        return _withdrawal_evaluation(rows, withdraw_sec=5., staged=staged,
            base_driver_kd=ZERO, torque_limits=(1.,)*6, torque_rates=(10.,)*6)
    return rows, evaluate


@pytest.mark.parametrize("withdrawal,duration,scenario", [
    (True, 6., "target-arrival"), (float("nan"), 6., "target-arrival"),
    (float("inf"), 6., "target-arrival"), (.6, 6., "target-arrival"),
    (6., 6., "target-arrival"), (7., 6., "target-arrival"),
    (4., 6., "lifecycle"),
])
def test_observation_timing_rejects_invalid_input_before_model_or_plant(withdrawal, duration, scenario, monkeypatch):
    from sim.mujoco.rebot_b601_rs import joint_motion_demo as demo

    def forbidden_model(*args, **kwargs):
        pytest.fail("invalid timing must not initialize a model or plant")
    monkeypatch.setattr(demo.B601PinocchioModel, "from_rebot_source_root", forbidden_model)
    with pytest.raises(ValueError, match="withdrawal|observation timing"):
        demo.run_b601_joint_motion_demo(source_root="unused", duration_sec=duration,
            scenario=scenario, scripted_withdrawal_sec=withdrawal)


def test_motion_diagnostics_separates_demand_slew_and_final_command_without_authorizing_motion():
    from copy import deepcopy

    from sim.mujoco.rebot_b601_rs.joint_motion_demo import _motion_diagnostics

    rows, _ = _withdrawal_trace()
    rows = deepcopy(rows[:3])
    for i, row in enumerate(rows):
        row.update(time_sec=1+i*.01, phase="track", target_error_rad=.002,
            q3_rad=i*.000384, raw_qvel_rad_s=(0,0,.0384,0,0,0),
            control_qvel_rad_s=(0,0,.001,0,0,0), physics_qvel_at_host_rad_s=(0,0,.0002,0,0,0),
            overlay_nm=.09)
        row["motion"].update(phase="track", requested_torque_nm=.12,
            stop_capture_block_reason=None)
        row["dispatch"]["conditioned_overlay_nm"] = (0,0,.07,0,0,0)
    staged = _staged_config(tracking_overlay_cap_nm=.10)
    before = deepcopy(rows)
    report = _motion_diagnostics(rows, staged=staged, withdraw_sec=2., position_quantum_rad=.000384)
    assert rows == before
    track = report["tracking"]
    assert track["demand_peak_nm"] == .12
    assert track["stage_request_peak_nm"] == .09
    assert track["final_command_delta_peak_nm"] == .07
    assert track["demand_cap_exceeded_samples"] == 3
    assert track["demand_cap_exceeded_fraction"] == 1
    assert track["stage_request_differs_from_capped_demand_samples"] == 3
    assert track["final_command_differs_from_stage_request_samples"] == 3
    stop = report["arrival_stop"]
    assert stop["position_tolerance_host_samples"] == 3
    assert stop["selected_raw_only_velocity_block_samples"] == 3
    assert stop["selected_window_velocity_block_samples"] == 0
    assert stop["maximum_physics_velocity_at_raw_only_block_rad_s"] == .0002
    assert report.get("passed") is None  # Diagnostics never add an acceptance path.
    empty = _motion_diagnostics([], staged=staged, withdraw_sec=2., position_quantum_rad=0.)
    assert empty["tracking"]["host_samples"] == 0
    assert empty["tracking"]["demand_peak_nm"] is None
    assert empty["tracking"]["demand_cap_exceeded_fraction"] is None


def test_protected_withdrawal_acceptance_requires_tail_damping_stop_dwell_and_final_receipt():
    rows, evaluate = _withdrawal_trace()
    result = evaluate()
    assert result["passed"], result
    assert result["completion_sec"] == pytest.approx(5.181)
    assert result["deadline_sec"] == pytest.approx(8.94)
    assert rows[1]["overlay_nm"] != 0 and any(rows[1]["dispatch"]["driver_kd_nm_s_per_rad"])


@pytest.mark.parametrize("reengaged", [False, True])
def test_already_completed_trial_needs_no_second_release_but_cannot_reuse_receipt_after_reengage(reengaged):
    rows, evaluate = _withdrawal_trace()
    for i, row in enumerate(rows):
        row.update(phase="human_override", overlay_nm=0., motion=None, release_completed=i == 0)
        row["raw_qvel_rad_s"] = row["control_qvel_rad_s"] = ZERO
        row["dispatch"].update(host_command_torque_nm=(0.,0.,.3,0.,0.,0.),
            conditioned_overlay_nm=ZERO, driver_kd_nm_s_per_rad=ZERO)
    if reengaged:
        rows[1]["phase"] = "cue"
    result = evaluate()
    assert result["passed"] is not reengaged
    assert result["checks"]["withdrawal_completed_within_deadline"] is not reengaged


@pytest.mark.parametrize("fault,check", [
    ("old_target", "withdrawal_requests_only_bounded_braking"),
    ("tail_gains", "withdrawal_damping_retained_during_tail"),
    ("zero_receipt", "withdrawal_completed_within_deadline"),
    ("stationary", "withdrawal_completed_within_deadline"),
    ("late", "withdrawal_completed_within_deadline"),
    ("resume", "withdrawal_zero_and_base_gains_retained"),
    ("slew", "withdrawal_conditioned_torque_bounds"),
    ("absolute", "withdrawal_conditioned_torque_bounds"),
    ("partial", "withdrawal_receipts_valid"),
    ("nan", "withdrawal_receipts_valid"),
    ("duplicate", "withdrawal_receipts_valid"),
    ("forged_zero", "withdrawal_receipts_valid"),
    ("opposite_brake_request", "withdrawal_requests_only_bounded_braking"),
    ("mismatched_brake_velocity", "withdrawal_requests_only_bounded_braking"),
    ("missing_boundary", "withdrawal_receipts_valid"),
])
def test_protected_withdrawal_rejects_real_faults_instead_of_only_relaxing_assertions(fault, check):
    rows, evaluate = _withdrawal_trace(completion_index=406 if fault == "late" else 19)
    if fault == "old_target":
        rows[1]["motion"]["position_torque_nm"] = .01
    elif fault == "tail_gains":
        rows[1]["dispatch"]["driver_kd_nm_s_per_rad"] = ZERO
    elif fault == "zero_receipt":
        for row in rows:
            row["release_completed"] = False
    elif fault == "stationary":
        for row in rows:
            row["raw_qvel_rad_s"] = (.01,)*6
    elif fault == "resume":
        rows[-1]["dispatch"]["conditioned_overlay_nm"] = (0.,0.,.001,0.,0.,0.)
        rows[-1]["dispatch"]["host_command_torque_nm"] = (0.,0.,.301,0.,0.,0.)
    elif fault == "slew":
        rows[1]["dispatch"]["host_command_torque_nm"] = (.5,)*6
        rows[1]["dispatch"]["conditioned_overlay_nm"] = (.5,.5,.2,.5,.5,.5)
    elif fault == "absolute":
        rows[1]["dispatch"]["host_command_torque_nm"] = (2.,)*6
        rows[1]["dispatch"]["conditioned_overlay_nm"] = (2.,2.,1.7,2.,2.,2.)
    elif fault == "partial":
        rows[1]["dispatch"]["host_command_torque_nm"] = (0.,)*5
    elif fault == "nan":
        rows[1]["dispatch"]["conditioned_overlay_nm"] = (float("nan"),)*6
    elif fault == "duplicate":
        rows[1]["dispatch"]["sequence"] = rows[0]["dispatch"]["sequence"]
    elif fault == "forged_zero":
        rows[1]["dispatch"]["conditioned_overlay_nm"] = ZERO
    elif fault == "opposite_brake_request":
        rows[1]["overlay_nm"] *= -1
    elif fault == "mismatched_brake_velocity":
        rows[1]["motion"]["control_qvel_rad_s"] *= -1
        rows[1]["motion"]["requested_torque_nm"] *= -1
        rows[1]["overlay_nm"] *= -1
    elif fault == "missing_boundary":
        del rows[:3]
    result = evaluate()
    assert not result["passed"]
    assert not result["checks"][check], result


@pytest.mark.parametrize("value", [True, 1, "yes"])
def test_continuous_handoff_requires_explicit_confirmed_boolean_contract(value):
    with pytest.raises((TypeError, ValueError)):
        _staged_config(continuous_motion_handoff=value)
    assert not _staged_config().continuous_motion_handoff


def test_continuous_handoff_confirmed_motion_does_not_need_full_stop():
    runtime = _continuous_runtime()
    first = _continuous_tick(runtime, 3, 1.07, q=.0004, raw=.04, soft=.006)
    assert first.overlay_torque_nm == ZERO
    second = _continuous_tick(runtime, 4, 1.08, q=.0008, raw=.04, soft=.007)
    assert runtime.phase is B601StagedAssistPhase.RELEASING
    third = _continuous_tick(runtime, 5, 1.09, q=.0012, raw=.04, soft=.008)
    assert third.motion_onset_confirm_count == 2
    assert third.motion is None
    assert runtime.phase is B601StagedAssistPhase.DROP
    assert all(v > 0. for v in third.drive_driver_kd_nm_s_per_rad)
    assert second.overlay_torque_nm == third.overlay_torque_nm == ZERO
    tracking = _continuous_tick(runtime, 6, 1.10, q=.0013, raw=.01, soft=.009)
    assert tracking.phase is B601StagedAssistPhase.TRACK
    assert tracking.motion.q_reference_rad == pytest.approx(.0013)


@pytest.mark.parametrize("tail", [(0.,0.,.01,0.,0.,0.), (0.,.001,0.,0.,0.,0.)])
def test_continuous_handoff_requires_final_zero_on_all_six_joints(tail):
    runtime = _continuous_runtime()
    _continuous_tick(runtime, 3, 1.07, final=tail)
    frame = _continuous_tick(runtime, 4, 1.08, final=tail)
    assert frame.motion_onset_confirm_count == 2
    assert runtime.phase is B601StagedAssistPhase.RELEASING
    _continuous_tick(runtime, 5, 1.09)
    assert runtime.phase is B601StagedAssistPhase.DROP


@pytest.mark.parametrize("when", ["before_tick", "before_receipt"])
def test_stop_during_continuous_handoff_brakes_and_cannot_commit_drop(when):
    runtime = _continuous_runtime()
    _continuous_tick(runtime, 3, 1.07, q=.0004, raw=.04, soft=.006)
    if when == "before_tick":
        runtime.request_stop_and_hold()
        seq, now, q = 4, 1.08, .0008
    else:
        _continuous_tick(runtime, 4, 1.08, q=.0008, raw=.04, soft=.007)
        seq, now, q = 5, 1.09, .0012
    frame = _continuous_tick(runtime, seq, now, q=q, raw=.04, soft=.008, ack=False)
    if when == "before_receipt":
        assert frame.motion_onset_confirm_count == 2
        runtime.request_stop_and_hold()
    _ack_runtime(runtime, frame, now)
    assert runtime.phase is B601StagedAssistPhase.RELEASING
    brake = _continuous_tick(runtime, seq+1, now+.01, q=q)
    assert brake.release_terminal
    assert brake.motion.phase == "brake"
    assert brake.reason == "operator_stop_during_continuous_handoff"
    assert brake.overlay_torque_nm[2] <= 0.
    for i in range(2, 30):
        frame = _continuous_tick(runtime, seq+i, now+i*.01, q=q, raw=0., soft=0.)
        assert frame.phase not in {B601StagedAssistPhase.DROP, B601StagedAssistPhase.TRACK}
        if runtime.release_completed:
            break
    assert runtime.release_completed


@pytest.mark.parametrize("cap", [True, 0., -.1, float("nan"), float("inf"), .31])
def test_sim_tracking_cap_is_validated_before_resource_loading(tmp_path, cap):
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    with pytest.raises((TypeError, ValueError)):
        run_b601_joint_motion_demo(source_root=tmp_path/"missing-source", tracking_overlay_cap_nm=cap)


def test_continuous_handoff_soft_speed_starts_existing_terminal_brake():
    runtime = _continuous_runtime()
    _continuous_tick(runtime, 3, 1.07, q=.0005)
    frame = _continuous_tick(runtime, 4, 1.08, soft=.03)
    assert frame.release_terminal
    assert frame.motion.phase == "brake"
    assert frame.overlay_torque_nm == ZERO
    assert frame.reason == "continuous_handoff_soft_velocity_braking"


def test_unconfirmed_continuous_handoff_expires_then_uses_fresh_brake_clock():
    runtime = _continuous_runtime(breakaway_max_duration_sec=.04)
    _continuous_tick(runtime, 3, 1.07, q=.0005)
    _continuous_tick(runtime, 4, 1.08, q=.0005)
    frame = _continuous_tick(runtime, 5, 1.10, q=.0005, raw=0., soft=0.)
    assert frame.release_terminal
    assert frame.motion.phase == "brake"
    assert frame.motion.brake_elapsed_sec == 0.
    for seq in range(6, 30):
        frame = _continuous_tick(runtime, seq, 1.10+(seq-5)*.01, q=.0005, raw=0., soft=0.)
        assert frame.phase not in {B601StagedAssistPhase.DROP, B601StagedAssistPhase.TRACK}
        if runtime.release_completed:
            break
    assert runtime.release_completed


@pytest.mark.parametrize("peak,slew,duration,impulse,ceiling,reasons", [
    (.2, 1, .2, .04, .19, ["duration_limited"]),
    (.2, 1, .20000000005, .04, .19, ["duration_limited"]),
    (.8, 2, .2, .04, .38, ["duration_limited", "impulse_limited"]),
    (.8, 10, .2, .034, .7, ["impulse_limited"]),
    (.8, 10, .2, .036, .8, []),
    (1.0, 10, .2, .1, 1.0, []),
])
def test_breakaway_envelope_matches_discrete_no_motion_requests(peak, slew, duration, impulse, ceiling, reasons):
    config = _staged_config(
        breakaway_torque_nm=peak, breakaway_rise_slew_nm_s=slew,
        breakaway_max_duration_sec=duration, breakaway_max_impulse_nm_sec=impulse,
        max_total_overlay_nm=max(.2, peak), motion_onset_confirm_samples=1,
    )
    envelope = config.breakaway_envelope(control_period_sec=.01)
    assert envelope["optimistic_command_peak_ceiling_nm"] == pytest.approx(ceiling)
    assert envelope["limiting_reasons"] == reasons
    shared = replace(_shared_config(), max_abs_assist_torque_nm=(max(.3, peak),) * 6)
    runtime = B601StagedSingleJointAssistRuntime(shared, config)
    pulse_peak, used_impulse = 0.0, 0.0
    for i in range(60):
        timestamp = 1 + i * .01
        runtime.publish_policy(_command(i + 1, timestamp))
        frame = _step(runtime, sequence=i + 1, timestamp=timestamp, dt=.01)
        if i == 0:
            assert runtime.authorize_breakaway(sequence=1)
        if frame.phase is B601StagedAssistPhase.BREAKAWAY:
            pulse_peak = max(pulse_peak, abs(frame.overlay_torque_nm[2]))
            used_impulse = max(used_impulse, frame.breakaway_impulse_nm_sec)
    assert pulse_peak == pytest.approx(ceiling)
    assert used_impulse <= impulse + 1e-12


def test_opt_in_motion_drops_pulse_at_reference_velocity_not_late_onset_threshold():
    runtime = _active_runtime(_staged_config(motion_control=B601JointMotionConfig()))
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06)
    pulse = _step(runtime, sequence=3, timestamp=1.08)
    assert pulse.overlay_torque_nm[2] > 0
    onset = _step(runtime, sequence=4, timestamp=1.10, q3=.0004, qvel3=.016)
    assert onset.overlay_torque_nm == ZERO
    assert onset.phase is B601StagedAssistPhase.BREAKAWAY
    dropped = _step(runtime, sequence=5, timestamp=1.12, q3=.0005, qvel3=.016)
    assert dropped.phase is B601StagedAssistPhase.DROP
    assert dropped.overlay_torque_nm == ZERO


def test_cue_motion_brakes_without_implicitly_authorizing_tracking():
    runtime = _active_runtime(_staged_config(motion_control=B601JointMotionConfig()))
    _step(runtime, sequence=1, timestamp=1.01)
    moved = _step(runtime, sequence=2, timestamp=1.03, q3=.0012, qvel3=.02)
    assert moved.phase is B601StagedAssistPhase.BRAKE
    assert moved.reason == "cue_motion_braking"
    assert moved.overlay_torque_nm[2] < 0
    assert moved.motion.position_torque_nm == 0
    assert not runtime.authorize_breakaway(sequence=1)


def test_motion_gap_withdraws_without_resuming_old_reference():
    runtime = _active_runtime(_staged_config(motion_control=B601JointMotionConfig()))
    _step(runtime, sequence=1, timestamp=1.01)
    gap = _step(runtime, sequence=2, timestamp=1.21, dt=.2)
    assert gap.reason == "control_gap_assist_withdrawn"
    assert gap.overlay_torque_nm == ZERO
    assert _step(runtime, sequence=3, timestamp=1.23).phase is B601StagedAssistPhase.AUTHORITY_EXHAUSTED


def test_legacy_runtime_does_not_accept_new_stop_hold_action():
    with pytest.raises(ValueError, match="motion"):
        _active_runtime().request_stop_and_hold()


@pytest.mark.parametrize("withdrawal", ["clear", "clutch", "override", "stale", "gap"])
def test_phase_gated_native_damping_rises_only_during_drive_and_withdraws(withdrawal):
    motion = B601JointMotionConfig(drive_driver_kd_nm_s_per_rad=(.4,) * 6)
    runtime = _active_runtime(_staged_config(motion_control=motion))
    first = _step(runtime, sequence=1, timestamp=1.01)
    assert first.drive_driver_kd_nm_s_per_rad == pytest.approx((.04,) * 6)
    assert first.overlay_torque_nm[0] == 0
    if withdrawal == "clear":
        runtime.clear_policy()
    elif withdrawal == "clutch":
        runtime.set_policy_clutch(True)
    frame = _step(runtime, sequence=2,
                  timestamp=2.2 if withdrawal == "stale" else 1.03,
                  human_override=withdrawal == "override",
                  dt=.2 if withdrawal == "gap" else .02)
    assert frame.drive_driver_kd_nm_s_per_rad == ZERO
    assert frame.overlay_torque_nm == ZERO


def test_support_motion_brakes_pretracking_and_revokes_breakaway():
    runtime = _active_runtime(_staged_config(motion_control=B601JointMotionConfig(
        drive_driver_kd_nm_s_per_rad=(.1,) * 6,
    )))
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    frame = runtime.step(
        control_sequence=2, control_timestamp_sec=1.03,
        measured_qpos_rad=ZERO, measured_qvel_rad_s=(0,0,0,-.04,0,0), dt_sec=.02,
    )
    assert frame.phase is B601StagedAssistPhase.BRAKE
    assert frame.reason == "support_velocity_braking"
    assert frame.overlay_torque_nm == ZERO
    assert frame.motion.position_torque_nm == 0
    assert not runtime.authorize_breakaway(sequence=1)
    assert frame.support_max_velocity_rad_s == .04
    assert any(frame.drive_driver_kd_nm_s_per_rad)


@pytest.mark.optional
@pytest.mark.parametrize("host_hz,direction", [(20, 1), (50, 1), (100, 1), (50, -1)])
def test_motion_fixture_brakes_and_captures_actual_stop_on_urdf_plant(host_hz, direction):
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    code, payload = run_b601_joint_motion_demo(source_root=source, host_hz=host_hz,
                                              target_offset_rad=direction * .02)
    acceptance = payload["simulation_acceptance"]
    # Host-period differences hid J4 peaks at 20/50 Hz. Lifecycle events can
    # remain correct without qualifying the full motion as speed-safe.
    assert code == (0 if host_hz == 100 else 3), acceptance
    assert acceptance["checks"]["physics_all_joint_velocity_within_limit"] == (host_hz == 100)
    assert acceptance["checks"]["dynamics_force_balance_consistent"] is True
    assert acceptance["physics_velocity_sample_count"] == 6000
    assert acceptance["physics_velocity_sample_hz"] == 1000
    assert acceptance["hardware_claim"] is False
    assert payload["target_arrival_claim"] is False
    assert acceptance["kind"] == "stop_lifecycle"
    assert acceptance["maximum_hold_drift_rad"] <= .003
    assert acceptance["peak_all_joint_raw_velocity_rad_s"] < .05
    assert {"drop", "track", "brake", "soft_hold"} <= set(acceptance["phases"])
    row = next(row for row in payload["samples"] if row["phase"] == "track")
    assert len(row["qpos_rad"]) == len(row["raw_qvel_rad_s"]) == 6
    dispatch = row["dispatch"]
    assert dispatch["command_applied"] and dispatch["plant_advanced"]
    assert len(dispatch["applied_sim_torque_nm"]) == 6
    breakdown = dispatch["torque_breakdown"]
    assert all(kp_tau == 0 for kp_tau in breakdown["position_torque_nm"])
    assert all(kd_tau == 0 for kd_tau in breakdown["damping_torque_nm"])
    assert breakdown["requested_torque_nm"][2] == pytest.approx(
        breakdown["gravity_torque_nm"][2] + breakdown["overlay_effective_nm"][2]
    )


@pytest.mark.optional
def test_motion_target_arrival_does_not_count_another_captured_stop_point_as_success():
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    code, payload = run_b601_joint_motion_demo(source_root=source, scenario="target-arrival")
    acceptance = payload["simulation_acceptance"]
    assert code == 3
    assert acceptance["kind"] == "target_arrival"
    assert acceptance["stop_lifecycle_passed"] is False
    assert acceptance["passed"] is False
    assert acceptance["failure_reason"] == "lifecycle_acceptance_failed"
    assert acceptance["failed_checks"] == [
        "physics_all_joint_velocity_within_limit", "stationary_at_original_target",
    ]
    evaluation = payload["target_evaluation"]
    assert evaluation["arrived"] is False
    assert evaluation["final_error_before_withdraw_rad"] > evaluation["position_tolerance_rad"]
    brake = evaluation["first_brake"]
    assert brake["time_sec"] < 3
    assert brake["reason"] == "reference_lag_braking"
    assert brake["motion"]["brake_trigger_lag_rad"] > .01


@pytest.mark.optional
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("host_hz", [50, 100])
def test_motion_small_target_needs_stationary_dwell_and_physics_speed_safety(direction, host_hz):
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    code, payload = run_b601_joint_motion_demo(
        source_root=source, scenario="target-arrival", target_offset_rad=direction * .005,
        host_hz=host_hz,
    )
    acceptance = payload["simulation_acceptance"]
    assert code == (0 if host_hz == 100 else 3), acceptance
    assert acceptance["hardware_claim"] is False
    assert acceptance["checks"]["stationary_at_original_target"] is True
    assert acceptance["failed_checks"] == (
        [] if host_hz == 100 else ["physics_all_joint_velocity_within_limit"]
    )
    assert payload["target_arrival_claim"] == (host_hz == 100)
    evaluation = payload["target_evaluation"]
    assert evaluation["arrival_confirmed_time_sec"] is not None
    assert abs(evaluation["final_error_before_withdraw_rad"]) <= evaluation["position_tolerance_rad"]


def test_joint_motion_scenario_cannot_leak_into_other_fixture(capsys, tmp_path):
    from sim.mujoco.rebot_b601_rs import haptic_demo

    with pytest.raises(SystemExit) as error:
        haptic_demo.main([
            "--rebot-source-root", str(tmp_path), "--fixture", "policy-assist",
            "--joint-motion-scenario", "target-arrival",
        ])
    assert error.value.code == 2
    assert "requires --fixture joint-motion" in capsys.readouterr().err


@pytest.mark.parametrize("option,value", [
    ("--joint-motion-profile", "offline.json"),
    ("--joint-motion-friction-nm", ".06"),
    ("--joint-motion-target-offset-rad", "-.02"),
    ("--joint-motion-breakaway-rise-slew-nm-s", "1.0"),
    ("--joint-motion-driver-kd", "0"),
    ("--joint-motion-static-friction-nm", ".75"),
    ("--joint-motion-friction-transition-rad-s", ".002"),
    ("--joint-motion-breakaway-torque-nm", ".8"),
    ("--joint-motion-breakaway-duration-sec", ".2"),
    ("--joint-motion-breakaway-impulse-nm-sec", ".08"),
    ("--joint-motion-overlay-cap-nm", ".8"),
    ("--joint-motion-feedback-quantum-rad", ".000384"),
    ("--joint-motion-onset-rad", ".0004"),
])
def test_joint_motion_counterfactual_tuning_is_sim_fixture_only(option, value, capsys, tmp_path):
    from sim.mujoco.rebot_b601_rs import haptic_demo

    with pytest.raises(SystemExit) as error:
        args = [
            "--rebot-source-root", str(tmp_path), "--fixture", "policy-assist", option, value,
        ]
        if option == "--joint-motion-driver-kd":
            args.extend(["0"] * 5)
        haptic_demo.main(args)
    assert error.value.code == 2
    assert "requires --fixture joint-motion" in capsys.readouterr().err


@pytest.mark.parametrize("profile_text", ['[]', '{"unknown":1}', '{"tracking_kp_nm_per_rad":0}', '{"tracking_kp_nm_per_rad":NaN}'])
def test_joint_motion_cli_rejects_invalid_profile_before_plant(profile_text, capsys, tmp_path):
    from sim.mujoco.rebot_b601_rs import haptic_demo

    profile = tmp_path / "offline.json"
    profile.write_text(profile_text)
    with pytest.raises(SystemExit) as error:
        haptic_demo.main([
            "--rebot-source-root", str(tmp_path), "--fixture", "joint-motion",
            "--joint-motion-profile", str(profile),
        ])
    assert error.value.code == 2
    assert "invalid offline joint-motion profile" in capsys.readouterr().err


@pytest.mark.parametrize("kwargs", [
    {"synthetic_friction_nm": -1}, {"synthetic_friction_nm": float("nan")},
    {"synthetic_friction_nm": float("inf")}, {"synthetic_friction_nm": True},
    {"breakaway_rise_slew_nm_s": 0}, {"breakaway_rise_slew_nm_s": -1},
    {"breakaway_rise_slew_nm_s": float("nan")},
    {"breakaway_rise_slew_nm_s": float("inf")},
    {"breakaway_rise_slew_nm_s": True},
    {"motion_config": {}},
    {"duration_sec": 3},
    {"motion_config": B601JointMotionConfig(soft_velocity_rad_s=.05)},
    {"driver_kd_nm_s_per_rad": (0,) * 5},
    {"driver_kd_nm_s_per_rad": (0, 0, -1, 0, 0, 0)},
    {"driver_kd_nm_s_per_rad": (0, 0, float("nan"), 0, 0, 0)},
    {"driver_kd_nm_s_per_rad": (0, 0, float("inf"), 0, 0, 0)},
    {"feedback_position_quantum_rad": -1},
    {"feedback_position_quantum_rad": float("nan")},
    {"feedback_position_quantum_rad": True},
    {"motion_onset_displacement_rad": 0},
])
def test_joint_motion_counterfactual_inputs_fail_before_plant(kwargs, tmp_path):
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    with pytest.raises((ValueError, TypeError)):
        run_b601_joint_motion_demo(source_root=tmp_path, **kwargs)


@pytest.mark.optional
@pytest.mark.parametrize("direction", [1, -1])
def test_v2_zero_passive_damping_quantized_lifecycle_and_restore(direction):
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo
    profile = Path(__file__).resolve().parents[1] / "docs/hardware/profiles/b601_j3_interaction_v2.json"
    import json
    config = B601JointMotionConfig(**json.loads(profile.read_text()))
    code, payload = run_b601_joint_motion_demo(
        source_root=source, host_hz=100, motion_config=config,
        target_offset_rad=direction * .02, synthetic_friction_nm=.06,
        driver_kd_nm_s_per_rad=ZERO, breakaway_rise_slew_nm_s=1,
        feedback_position_quantum_rad=.000384, motion_onset_displacement_rad=.0004,
    )
    acceptance = payload["simulation_acceptance"]
    assert code == 0, acceptance
    assert acceptance["checks"]["physics_all_joint_velocity_within_limit"]
    assert acceptance["withdrawal_passed"]
    assert payload["withdrawal_evaluation"]["completion_sec"] < payload["withdrawal_evaluation"]["deadline_sec"]
    assert payload["target_arrival_claim"] is False
    assert payload["feedback_position_quantum_rad"] == .000384
    assert payload["declared_model_diagnostics"]["driver_kd_nm_s_per_rad"] == ZERO
    assert payload["declared_model_diagnostics"]["motion_onset_displacement_rad"] == .0004
    samples = payload["samples"]
    assert any(any(row["drive_driver_kd_nm_s_per_rad"]) for row in samples if row["time_sec"] < 5)
    assert all(not any(row["drive_driver_kd_nm_s_per_rad"]) for row in samples if row["time_sec"] >= 5)


@pytest.mark.optional
@pytest.mark.parametrize("transition", [.002, .02])
def test_large_breakaway_is_not_promoted_by_motion_or_lifecycle_pass(transition):
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    import json

    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo
    profile = Path(__file__).resolve().parents[1] / "docs/hardware/profiles/b601_j3_interaction_v2.json"
    config = B601JointMotionConfig(**json.loads(profile.read_text()))
    code, payload = run_b601_joint_motion_demo(
        source_root=source, host_hz=100, motion_config=config, scenario="target-arrival",
        driver_kd_nm_s_per_rad=ZERO, feedback_position_quantum_rad=.000384,
        motion_onset_displacement_rad=.0004, synthetic_friction_nm=.06,
        synthetic_static_friction_nm=.75, friction_transition_velocity_rad_s=transition,
        breakaway_torque_nm=.8, breakaway_rise_slew_nm_s=10,
        breakaway_max_duration_sec=.2, breakaway_max_impulse_nm_sec=.08,
        total_overlay_cap_nm=.8,
    )
    assert code == 3
    assert payload["target_arrival_claim"] is False
    assert payload["declared_model_diagnostics"]["parameter_source"] == "simulation_only_not_hardware_identified"
    if transition == .002:
        assert not payload["simulation_acceptance"]["checks"]["physics_all_joint_velocity_within_limit"]
        assert payload["simulation_acceptance"]["first_physics_velocity_violation"]["limit_rad_s"] == .05
        assert payload["simulation_acceptance"]["first_physics_velocity_violation"]["joint_name"] == "joint4"
        window = payload["physics_velocity_event_window"]
        assert len(window["samples"]) == window["maximum_samples"] == 41
        assert window["post_window_complete"]
        assert window["host_violation_delay_sec"] == pytest.approx(.006)
        assert window["parameter_source"] == "simulation_ground_truth_not_hardware_latency_or_force"
        event = next(row for row in window["samples"] if row["measured_time_sec"] == window["event_time_sec"])
        assert abs(event["physics_qvel_rad_s"][3]) > .05
        assert event["host_raw_qvel_rad_s"] == ZERO
        assert event["host_time_sec"] < event["measured_time_sec"]
    else:
        assert payload["simulation_acceptance"]["stop_lifecycle_passed"]
        assert not payload["target_evaluation"]["arrived"]
        assert payload["physics_velocity_event_window"]["event_time_sec"] is None
        assert payload["physics_velocity_event_window"]["post_window_complete"] is None
        assert payload["physics_velocity_event_window"]["samples"] == []


@pytest.mark.parametrize("kwargs", [
    {"synthetic_static_friction_nm": .01}, {"synthetic_static_friction_nm": float("nan")},
    {"friction_transition_velocity_rad_s": 0}, {"breakaway_torque_nm": -1},
    {"total_overlay_cap_nm": .01},
])
def test_joint_motion_envelope_inputs_fail_before_plant(kwargs, tmp_path):
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo
    with pytest.raises(ValueError):
        run_b601_joint_motion_demo(source_root=tmp_path, **kwargs)


def test_joint_motion_cli_exposes_the_whole_simulation_pulse_and_friction_shape(monkeypatch, tmp_path):
    from sim.mujoco.rebot_b601_rs import haptic_demo
    arguments = {}
    def run(**kwargs):
        arguments.update(kwargs)
        return 0, {}
    monkeypatch.setattr(haptic_demo, "run_b601_joint_motion_demo", run)
    monkeypatch.setattr(haptic_demo, "_emit_payload", lambda **kwargs: kwargs["code"])
    assert haptic_demo.main([
        "--rebot-source-root", str(tmp_path), "--fixture", "joint-motion",
        "--joint-motion-static-friction-nm", ".75", "--joint-motion-friction-nm", ".06",
        "--joint-motion-friction-transition-rad-s", ".002",
        "--joint-motion-breakaway-torque-nm", ".8", "--joint-motion-breakaway-duration-sec", ".2",
        "--joint-motion-breakaway-impulse-nm-sec", ".08", "--joint-motion-overlay-cap-nm", ".8",
        "--joint-motion-feedback-quantum-rad", ".000384", "--joint-motion-onset-rad", ".0004",
        "--joint-motion-initial-qpos-rad", "0", ".15", ".18", ".05", ".04", ".008",
    ]) == 0
    assert arguments["synthetic_static_friction_nm"] == .75
    assert arguments["friction_transition_velocity_rad_s"] == .002
    assert arguments["breakaway_torque_nm"] == arguments["total_overlay_cap_nm"] == .8
    assert arguments["breakaway_max_duration_sec"] == .2
    assert arguments["breakaway_max_impulse_nm_sec"] == .08
    assert arguments["feedback_position_quantum_rad"] == .000384
    assert arguments["motion_onset_displacement_rad"] == .0004
    assert arguments["initial_arm_qpos_rad"] == (0, .15, .18, .05, .04, .008)


@pytest.mark.optional
def test_recorded_pose_counterfactual_catches_physics_speed_missed_by_host_difference():
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    import json

    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo
    profile = Path(__file__).resolve().parents[1] / "docs/hardware/profiles/b601_j3_interaction_v2.json"
    config = B601JointMotionConfig(**json.loads(profile.read_text()))
    # Declared pose from a prior run; other plant terms are not hardware identified.
    pose = (-.012656090781092644, .14458218216896057, .18408368527889252, .04908888414502144, .043719951063394547, .008053399622440338)
    code, payload = run_b601_joint_motion_demo(
        source_root=source, host_hz=100, motion_config=config, scenario="target-arrival",
        initial_arm_qpos_rad=pose, driver_kd_nm_s_per_rad=ZERO,
        feedback_position_quantum_rad=.000384, motion_onset_displacement_rad=.0004,
        synthetic_friction_nm=.06, synthetic_static_friction_nm=.75,
        friction_transition_velocity_rad_s=.002, breakaway_torque_nm=.8,
        breakaway_rise_slew_nm_s=10, breakaway_max_duration_sec=.2,
        breakaway_max_impulse_nm_sec=.08, total_overlay_cap_nm=.8,
    )
    assert code == 3
    assert payload["initial_arm_qpos_rad"] == pose
    checks = payload["simulation_acceptance"]["checks"]
    assert checks["all_joint_velocity_within_limit"]
    assert not checks["physics_all_joint_velocity_within_limit"]
    assert not payload["target_arrival_claim"]
    window = payload["physics_velocity_event_window"]
    assert len(window["samples"]) == 41
    assert window["host_violation_delay_sec"] is None
    assert window["post_window_complete"]


def test_joint_motion_initial_pose_cannot_leak_to_other_fixtures(capsys, tmp_path):
    from sim.mujoco.rebot_b601_rs import haptic_demo
    with pytest.raises(SystemExit) as error:
        haptic_demo.main([
            "--fixture", "policy-assist", "--rebot-source-root", str(tmp_path),
            "--joint-motion-initial-qpos-rad", "0", "1", "1", "0", "0", "0",
        ])
    assert error.value.code == 2
    assert "requires --fixture joint-motion" in capsys.readouterr().err


def test_joint_motion_cli_consumes_explicit_sim_profile_and_signed_target(monkeypatch, tmp_path):
    from sim.mujoco.rebot_b601_rs import haptic_demo

    profile = tmp_path / "offline.json"
    profile.write_text('{"tracking_kp_nm_per_rad":12}')
    arguments = {}

    def run(**kwargs):
        arguments.update(kwargs)
        return 0, {"simulation_acceptance": {"passed": True}}

    monkeypatch.setattr(haptic_demo, "run_b601_joint_motion_demo", run)
    assert haptic_demo.main([
        "--rebot-source-root", str(tmp_path), "--fixture", "joint-motion",
        "--joint-motion-profile", str(profile), "--joint-motion-friction-nm", ".08",
        "--joint-motion-target-offset-rad", "-.02",
        "--joint-motion-breakaway-rise-slew-nm-s", "1.0",
        "--joint-motion-driver-kd", "0", "0", "0", "0", "0", "0",
    ]) == 0
    assert arguments["motion_config"] == replace(B601JointMotionConfig(), tracking_kp_nm_per_rad=12)
    assert arguments["synthetic_friction_nm"] == .08
    assert arguments["target_offset_rad"] == -.02
    assert arguments["breakaway_rise_slew_nm_s"] == 1.0
    assert arguments["driver_kd_nm_s_per_rad"] == ZERO
    assert arguments["duration_sec"] == 6


@pytest.mark.parametrize("fixture,duration,expected", [
    ("joint-motion", None, 6), ("joint-motion", 8, 8), ("joint-motion", 3, 3),
    ("policy-assist", None, None), ("policy-assist", 8, 8),
])
def test_joint_motion_cli_default_duration_does_not_change_other_fixtures_or_explicit_input(
    fixture, duration, expected, monkeypatch, tmp_path,
):
    from sim.mujoco.rebot_b601_rs import haptic_demo

    arguments = {}

    def run(**kwargs):
        arguments.update(kwargs)
        return 0, {}

    monkeypatch.setattr(haptic_demo, "run_b601_joint_motion_demo", run)
    monkeypatch.setattr(haptic_demo, "run_b601_shared_control_demo", run)
    monkeypatch.setattr(haptic_demo, "run_haptic_fixture_demo", run)
    monkeypatch.setattr(haptic_demo, "_emit_payload", lambda **kwargs: kwargs["code"])
    args = ["--rebot-source-root", str(tmp_path), "--fixture", fixture]
    if duration is not None:
        args.extend(["--duration-sec", str(duration)])
    assert haptic_demo.main(args) == 0
    assert arguments["duration_sec"] == (
        haptic_demo.DEFAULT_HAPTIC_DURATION_SEC if expected is None else expected
    )


@pytest.mark.optional
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("host_hz", [20, 50, 100])
def test_offline_tracking_arrival_does_not_hide_physics_velocity_peaks(direction, host_hz):
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    code, payload = run_b601_joint_motion_demo(
        source_root=source, scenario="target-arrival", target_offset_rad=direction * .02,
        host_hz=host_hz,
        motion_config=replace(B601JointMotionConfig(), tracking_kp_nm_per_rad=12),
    )
    acceptance = payload["simulation_acceptance"]
    assert code == (0 if host_hz == 100 else 3), acceptance
    assert acceptance["hardware_claim"] is False
    assert acceptance["first_velocity_violation"] is None
    assert acceptance["checks"]["all_joint_velocity_within_limit"] is True
    assert acceptance["checks"]["physics_all_joint_velocity_within_limit"] == (host_hz == 100)
    if host_hz != 100:
        assert acceptance["failed_checks"] == ["physics_all_joint_velocity_within_limit"]
        assert acceptance["first_physics_velocity_violation"]["joint_name"] == "joint4"
    assert payload["target_arrival_claim"] == (host_hz == 100)
    assert payload["target_evaluation"]["arrived"] is True
    model = payload["declared_model_diagnostics"]
    assert model["motion_config_source"] == "explicit_offline_counterfactual"
    assert model["tracking_overlay_cap_nm"] == .1
    assert model["breakaway_peak_nm"] == .2
    assert model["breakaway_rise_slew_nm_s"] == 2.0
    assert model["breakaway_max_duration_sec"] == .2
    assert model["breakaway_max_impulse_nm_sec"] == .04
    assert model["hard_velocity_acceptance_rad_s"] == .05
    brake = payload["target_evaluation"]["first_brake"]
    assert brake["reason"] == "target_reached_braking"
    assert abs(brake["target_error_rad"]) <= .003


@pytest.mark.optional
@pytest.mark.parametrize("kp,friction,failed_check,reason", [
    (24, .06, "stationary_at_original_target", "soft_velocity_braking"),
    (12, .05, "all_joint_velocity_within_limit", "target_reached_braking"),
    (12, .0, "stationary_at_original_target", "cue_motion_braking"),
])
def test_offline_counterfactual_does_not_hide_failure_by_raising_gain(kp, friction, failed_check, reason):
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    code, payload = run_b601_joint_motion_demo(
        source_root=source, scenario="target-arrival", synthetic_friction_nm=friction,
        motion_config=replace(B601JointMotionConfig(), tracking_kp_nm_per_rad=kp),
    )
    assert code == 3
    assert failed_check in payload["simulation_acceptance"]["failed_checks"]
    assert payload["target_arrival_claim"] is False
    assert payload["target_evaluation"]["first_brake"]["reason"] == reason
    if friction == .05:
        violation = payload["simulation_acceptance"]["first_velocity_violation"]
        assert violation["joint_name"] == "joint4"
        assert violation["phase"] == "drop"
        assert violation["time_sec"] < .7
        assert abs(violation["raw_velocity_rad_s"]) > violation["limit_rad_s"]


@pytest.mark.optional
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("friction", [.05, .06, .08])
@pytest.mark.parametrize("host_hz", [20, 50, 100])
def test_gentler_offline_breakaway_reduces_coupling_without_hiding_arrival_failure(direction, friction, host_hz):
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    code, payload = run_b601_joint_motion_demo(
        source_root=source, scenario="target-arrival", synthetic_friction_nm=friction,
        target_offset_rad=direction * .02, breakaway_rise_slew_nm_s=1.0, host_hz=host_hz,
        motion_config=replace(B601JointMotionConfig(), tracking_kp_nm_per_rad=12),
    )
    acceptance = payload["simulation_acceptance"]
    assert acceptance["hardware_claim"] is False
    if friction == .08 and host_hz == 20:
        assert acceptance["checks"]["all_joint_velocity_within_limit"] is False
        assert acceptance["first_velocity_violation"]["joint_name"] == "joint4"
        assert acceptance["first_velocity_violation"]["phase"] == "brake"
    else:
        assert acceptance["first_velocity_violation"] is None
        assert acceptance["checks"]["all_joint_velocity_within_limit"] is True
    model = payload["declared_model_diagnostics"]
    assert model["breakaway_rise_slew_nm_s"] == 1.0
    assert model["breakaway_peak_nm"] == .2
    assert model["breakaway_max_duration_sec"] == .2
    assert model["breakaway_max_impulse_nm_sec"] == .04
    assert model["tracking_overlay_cap_nm"] == .1
    assert model["hard_velocity_acceptance_rad_s"] == .05
    if friction == .08:
        assert code == 3
        assert "stationary_at_original_target" in acceptance["failed_checks"]
        assert payload["target_arrival_claim"] is False
        assert abs(payload["target_evaluation"]["final_error_before_withdraw_rad"]) > .003
    else:
        physics_safe = host_hz == 100 or (host_hz == 50 and friction == .06)
        assert code == (0 if physics_safe else 3), acceptance
        assert acceptance["checks"]["physics_all_joint_velocity_within_limit"] == physics_safe
        assert payload["target_arrival_claim"] == physics_safe
        if not physics_safe:
            assert acceptance["failed_checks"] == ["physics_all_joint_velocity_within_limit"]
            assert acceptance["first_physics_velocity_violation"]["joint_name"] == "joint4"
        assert payload["target_evaluation"]["first_brake"]["reason"] == "target_reached_braking"


@pytest.mark.optional
def test_slower_reference_alone_does_not_fix_existing_low_gain_arrival_failure():
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    code, payload = run_b601_joint_motion_demo(
        source_root=source, scenario="target-arrival",
        motion_config=replace(B601JointMotionConfig(), max_reference_velocity_rad_s=.005),
    )
    assert code == 3
    assert payload["target_evaluation"]["first_brake"]["reason"] == "reference_lag_braking"
    assert payload["target_evaluation"]["first_brake"]["time_sec"] > 3
    assert payload["target_evaluation"]["final_error_before_withdraw_rad"] > .015


@pytest.mark.optional
def test_physics_diagnostics_are_read_only_and_keep_force_and_integration_clocks(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs import joint_motion_demo

    original = joint_motion_demo._physics_dynamics_snapshot
    observed = []

    def inspect(scene, model, tick):
        if observed or tick.physics_time_sec < .65:
            return original(scene, model, tick)
        names = (
            "qpos", "qvel", "ctrl", "qacc", "qM", "qfrc_actuator",
            "qfrc_applied", "qfrc_passive", "qfrc_constraint", "qfrc_bias",
        )
        before = {name: getattr(scene.data, name).copy() for name in names}
        clock = scene.data.time
        gains = scene.model.actuator_gainprm.copy()
        report = original(scene, model, tick)
        assert scene.data.time == clock
        np.testing.assert_array_equal(scene.model.actuator_gainprm, gains)
        for name in names:
            np.testing.assert_array_equal(getattr(scene.data, name), before[name])
        assert report["force_state_time_sec"] == tick.physics_time_sec
        assert report["integrated_state_time_sec"] == tick.measured_time_sec
        assert tick.measured_time_sec - tick.physics_time_sec == pytest.approx(.001)
        components = np.asarray(list(report["acceleration_terms_rad_s2"].values()))
        np.testing.assert_allclose(
            components.sum(axis=0), report["pre_integration_qacc_rad_s2"], atol=1e-8,
        )
        assert report["force_balance_max_residual_nm"] < 1e-6
        assert report["parameter_source"] == "simulation_ground_truth_not_hardware_force"
        assert max(abs(v) for v in report["force_terms_nm"]["constraints"]) > 0
        observed.append(report)
        return report

    monkeypatch.setattr(joint_motion_demo, "_physics_dynamics_snapshot", inspect)
    code, payload = joint_motion_demo.run_b601_joint_motion_demo(
        source_root=source, host_hz=100, scenario="target-arrival",
        motion_config=replace(B601JointMotionConfig(), tracking_kp_nm_per_rad=12),
    )
    assert observed
    assert code == 0, payload["simulation_acceptance"]
    assert payload["declared_model_diagnostics"]["driver_damping_source"] == "raw_model_baseline"
    assert payload["declared_model_diagnostics"]["driver_kd_nm_s_per_rad"] == (.2, .3, .2, .15, .1, .05)
    for row in payload["samples"]:
        # Host truth and controller feedback share a pre-step clock. A
        # dispatch's integrated position must not stand in for either one.
        assert row["physics_qpos_at_host_rad"] == row["qpos_rad"]
        assert row["physics_qpos_at_host_rad"] == row["dynamics"]["force_state_qpos_rad"]
        assert row["physics_qvel_at_host_rad_s"] == row["dynamics"]["force_state_qvel_rad_s"]
        assert row["time_sec"] == row["dynamics"]["force_state_time_sec"]
        assert 0 <= row["motion_onset_confirm_count"]
        assert row["motion_onset_required_samples"] >= 1
        assert row["breakaway_impulse_nm_sec"] >= 0


@pytest.mark.optional
@pytest.mark.parametrize("quantum", [0., .000384])
def test_offline_low_force_capture_confirms_arrival_and_protected_exit_without_hardware_claim(quantum):
    import json

    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j3_interaction_v2.json"
    code, payload = run_b601_joint_motion_demo(
        source_root=source, host_hz=100, scenario="target-arrival",
        motion_config=B601JointMotionConfig(**json.loads(profile.read_text())),
        driver_kd_nm_s_per_rad=ZERO, feedback_position_quantum_rad=quantum,
        motion_onset_displacement_rad=.0004, motion_onset_confirm_displacement_rad=.0007,
        motion_onset_confirm_samples=2, breakaway_torque_nm=1., breakaway_rise_slew_nm_s=1.,
        breakaway_max_duration_sec=1.2, breakaway_max_impulse_nm_sec=.6, total_overlay_cap_nm=1.,
        synthetic_static_friction_nm=.06, friction_transition_velocity_rad_s=.002,
        continuous_motion_handoff=True, capture_overlay_cap_nm=.10,
    )
    assert payload["target_evaluation"]["arrived"]
    acceptance = payload["simulation_acceptance"]
    assert acceptance["checks"]["physics_all_joint_velocity_within_limit"]
    assert acceptance["checks"]["all_joint_velocity_within_limit"]
    assert acceptance["maximum_support_drift_rad"] < .008727
    assert code == 0 and acceptance["passed"] and payload["target_arrival_claim"]
    assert acceptance["hardware_claim"] is False
    assert acceptance["withdrawal_passed"] and acceptance["safety_passed"]
    withdrawal = payload["withdrawal_evaluation"]
    assert 5 < withdrawal["completion_sec"] < withdrawal["deadline_sec"]
    assert payload["human_interaction_evaluation"]["assessed"] is False
    assert payload["scripted_withdrawal_time_sec"] == 5.0
    diagnostics = payload["motion_diagnostics"]
    assert diagnostics["available"] is True
    assert diagnostics["tracking"]["demand_cap_exceeded_samples"] == 0
    assert diagnostics["tracking"]["cap_nm"] == .10
    capture_rows = [r for r in payload["samples"] if r["phase"] == "capture"]
    assert capture_rows
    entry = next(r for r in payload["samples"] if r.get("phase_after_dispatch") == "capture")
    assert max(abs(x) for x in entry["dispatch"]["conditioned_overlay_nm"]) <= 1e-9
    assert all(abs(r["overlay_nm"]) <= .10 for r in capture_rows)
    assert all(r["breakaway_impulse_nm_sec"] <= .6 for r in capture_rows)
    assert all(r["conditioned_command_impulse_nm_sec"] <= .6 for r in capture_rows)
    assert capture_rows[-1]["time_sec"] < 1.8  # Original one-shot deadline.
    for index, row in enumerate(payload["samples"]):
        if row["phase"] == "track" and payload["samples"][index-1]["phase"] == "capture":
            assert payload["samples"][index-1]["motion_onset_confirm_count"] >= 2
            assert row["motion"]["qvel_reference_rad_s"] > 0


@pytest.mark.optional
@pytest.mark.parametrize("direction", [-1, 1])
def test_offline_reference_pacing_improves_high_resistance_motion_but_does_not_claim_arrival(direction):
    import json

    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j3_interaction_v2.json"
    motion = B601JointMotionConfig(**json.loads(profile.read_text()))
    results = []
    for pacing in (False, True):
        _, payload = run_b601_joint_motion_demo(
            source_root=source, host_hz=100, scenario="target-arrival", target_offset_rad=direction*.02,
            motion_config=replace(motion, reference_pacing=pacing),
            driver_kd_nm_s_per_rad=ZERO, feedback_position_quantum_rad=.000384,
            motion_onset_displacement_rad=.0004, motion_onset_confirm_displacement_rad=.0007,
            motion_onset_confirm_samples=2, breakaway_torque_nm=1., breakaway_rise_slew_nm_s=1.,
            breakaway_max_duration_sec=1.2, breakaway_max_impulse_nm_sec=.6, total_overlay_cap_nm=1.,
            synthetic_static_friction_nm=.75, friction_transition_velocity_rad_s=.002,
            continuous_motion_handoff=True, capture_overlay_cap_nm=.10,
        )
        assert payload["simulation_acceptance"]["safety_passed"]
        assert payload["simulation_acceptance"]["withdrawal_passed"]
        assert payload["simulation_acceptance"]["hardware_claim"] is False
        assert payload["target_evaluation"]["arrived"] is False
        assert payload["simulation_acceptance"]["passed"] is False
        for row in payload["samples"]:
            if row["phase"] in {"capture", "track"}:
                assert abs(row["overlay_nm"]) <= .10 + 1e-9
                assert abs(row["motion"]["q_reference_rad"]-row["q3_rad"]) <= .01 + 1e-9
        results.append(payload)
    baseline, candidate = results
    assert baseline["target_evaluation"]["first_brake"]["reason"] == "reference_lag_braking"
    assert abs(candidate["target_evaluation"]["final_error_before_withdraw_rad"]) + .005 < abs(baseline["target_evaluation"]["final_error_before_withdraw_rad"])


@pytest.mark.optional
def test_long_quantized_candidate_latches_failed_exit_and_keeps_actual_damping_after_timeout():
    import json

    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j3_interaction_v2.json"
    motion = replace(B601JointMotionConfig(**json.loads(profile.read_text())), reference_pacing=True)
    code, payload = run_b601_joint_motion_demo(
        source_root=source, host_hz=100, scenario="target-arrival", duration_sec=14., scripted_withdrawal_sec=10.,
        motion_config=motion, driver_kd_nm_s_per_rad=ZERO, feedback_position_quantum_rad=.000384,
        motion_onset_displacement_rad=.0004, motion_onset_confirm_displacement_rad=.0007,
        motion_onset_confirm_samples=2, breakaway_torque_nm=1., breakaway_rise_slew_nm_s=1.,
        breakaway_max_duration_sec=1.2, breakaway_max_impulse_nm_sec=.6, total_overlay_cap_nm=1.,
        synthetic_static_friction_nm=.06, friction_transition_velocity_rad_s=.002,
        continuous_motion_handoff=True, capture_overlay_cap_nm=.10,
    )
    assert code == 3
    acceptance = payload["simulation_acceptance"]
    assert not acceptance["passed"]
    assert acceptance["failure_reason"] == "staged_exit_failed:braking_timeout_assist_withdrawn"
    assert acceptance["checks"]["no_fixture_fault"] is False
    assert acceptance["checks"]["physics_all_joint_velocity_within_limit"]
    assert acceptance["checks"]["all_joint_velocity_within_limit"]
    assert acceptance["withdrawal_passed"] is False
    assert payload["withdrawal_evaluation"]["completion_sec"] is None
    failed = [r for r in payload["samples"] if r["release_failed"]]
    assert failed and failed[0]["time_sec"] == pytest.approx(7.47)
    first_index = payload["samples"].index(failed[0])
    for row in payload["samples"][first_index:]:
        assert row["release_failed"] and row["release_terminal"]
        assert row["phase"] == "releasing"
        assert row["overlay_nm"] == 0 and not row["release_completed"]
        assert not row["release_restoring_base_damping"]
        assert row["dispatch"]["driver_kd_nm_s_per_rad"] == motion.drive_driver_kd_nm_s_per_rad
        assert row["motion"]["brake_elapsed_sec"] == pytest.approx(row["time_sec"]-4.47)


@pytest.mark.optional
def test_zero_motor_damping_counterfactual_is_explicit_and_not_qualified_for_hardware():
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    code, payload = run_b601_joint_motion_demo(
        source_root=source, host_hz=100, scenario="target-arrival",
        driver_kd_nm_s_per_rad=ZERO, breakaway_rise_slew_nm_s=1,
        motion_config=replace(B601JointMotionConfig(), tracking_kp_nm_per_rad=12),
    )
    assert code == 3
    acceptance = payload["simulation_acceptance"]
    assert acceptance["hardware_claim"] is False
    assert "physics_all_joint_velocity_within_limit" in acceptance["failed_checks"]
    assert payload["target_arrival_claim"] is False
    diagnostics = payload["declared_model_diagnostics"]
    assert diagnostics["driver_damping_source"] == "explicit_offline_counterfactual"
    assert diagnostics["driver_kd_nm_s_per_rad"] == ZERO
    assert diagnostics["controller_profile"]["driver_kd_nm_s_per_rad"] == ZERO
    assert diagnostics["human_force_measured"] is False
    assert all(
        all(t == 0 for t in row["dynamics"]["force_terms_nm"]["motor_local_damping"])
        for row in payload["samples"]
    )


def test_target_arrival_requires_continuous_dwell_and_no_later_drift():
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import _target_arrival_time

    def row(t, error=.001, velocity=0):
        return {
            "time_sec": t, "phase": "soft_hold", "target_error_rad": error,
            "raw_v3_rad_s": velocity, "control_qvel_rad_s": (0, 0, velocity, 0, 0, 0),
        }

    def evaluate(rows):
        return _target_arrival_time(
            rows, tolerance_rad=.003, velocity_rad_s=.005,
            dwell_sec=.15, withdraw_sec=5,
        )

    assert evaluate([row(.1), row(.26), row(.5), row(5, error=.02)]) == .26
    assert evaluate([row(.1), row(.26), row(.5, error=.02)]) is None
    assert evaluate([row(.1), row(.2, velocity=.01), row(.3), row(.4)]) is None
    assert evaluate([row(.1), row(.2, velocity=.01), row(.3), row(.46)]) == .46


@pytest.mark.optional
@pytest.mark.parametrize("missing_receipt", ["dispatch", "off_cadence_physics"])
def test_motion_qualification_rejects_missing_dispatch_or_physics_receipt(monkeypatch, missing_receipt):
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs import joint_motion_demo

    original = joint_motion_demo.B601MujocoEffortDriver.step_physics
    missing = False

    def drop_one_receipt(driver, reference, *, restore_driver_kd_nm_s_per_rad=None):
        nonlocal missing
        tick = original(driver, reference, restore_driver_kd_nm_s_per_rad=restore_driver_kd_nm_s_per_rad)
        if reference is not None and tick.physics_time_sec >= .70 and not missing and missing_receipt == "dispatch":
            missing = True
            return replace(tick, command_applied=False, rejection_reason="synthetic_missing_receipt")
        if reference is None and tick.physics_time_sec >= .70 and not missing and missing_receipt == "off_cadence_physics":
            missing = True
            return replace(tick, plant_advanced=False)
        return tick

    monkeypatch.setattr(joint_motion_demo.B601MujocoEffortDriver, "step_physics", drop_one_receipt)
    code, payload = joint_motion_demo.run_b601_joint_motion_demo(source_root=source)
    assert missing
    assert code == 3
    assert "complete_dispatch_and_physics" in payload["simulation_acceptance"]["failed_checks"]


@pytest.mark.optional
def test_motion_fixture_does_not_claim_success_when_synthetic_stiction_blocks_motion():
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs.joint_motion_demo import run_b601_joint_motion_demo

    code, payload = run_b601_joint_motion_demo(source_root=source, synthetic_friction_nm=.8)
    assert code == 3
    assert payload["simulation_acceptance"]["passed"] is False
    rows = payload["samples"]
    exhausted = [row for row in rows if row["phase"] == "authority_exhausted"]
    assert exhausted
    assert all(row["overlay_nm"] == 0 for row in exhausted)
    assert "track" not in payload["simulation_acceptance"]["phases"]


@pytest.mark.optional
def test_motion_fixture_preserves_interrupt_result_even_if_viewer_teardown_times_out(monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    source = Path(os.environ.get("B601_REBOT_SOURCE_ROOT", "hardware/reBotArm_control_py"))
    if not (source / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("declare B601_REBOT_SOURCE_ROOT for offline plant validation")
    from sim.mujoco.rebot_b601_rs import joint_motion_demo

    class InterruptedViewer:
        def __init__(self, scene):
            pass

        def start(self):
            pass

        def sync(self):
            raise KeyboardInterrupt

        def close(self):
            raise TimeoutError("synthetic teardown timeout")

    monkeypatch.setattr(joint_motion_demo, "MujocoPassiveViewerSession", InterruptedViewer)
    code, payload = joint_motion_demo.run_b601_joint_motion_demo(source_root=source, viewer=True)
    assert code == 130
    assert payload["status"] == "INTERRUPTED"
    assert payload["simulation_acceptance"]["passed"] is False
    assert payload["simulation_acceptance"]["driver_fault"] is None
    assert payload["simulation_acceptance"]["failure_reason"] == "operator_interrupted"
    assert payload["viewer_teardown_error"] == "synthetic teardown timeout"


def test_cue_is_single_joint_bounded_and_policy_sequence_updates_do_not_restart_it() -> None:
    runtime = _active_runtime()
    first = _step(runtime, sequence=1, timestamp=1.01)
    generation = first.intent_generation
    assert first.phase is B601StagedAssistPhase.CUE
    assert first.overlay_torque_nm == pytest.approx((0.0, 0.0, 0.04, 0.0, 0.0, 0.0))

    assert runtime.publish_policy(_command(2, 1.02)).accepted
    next_frame = _step(runtime, sequence=2, timestamp=1.03)
    assert next_frame.phase is B601StagedAssistPhase.CUE
    assert next_frame.intent_generation == generation


def test_breakaway_requires_explicit_one_shot_grant_and_drops_on_confirmed_motion() -> None:
    runtime = _active_runtime()
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert not runtime.authorize_breakaway(sequence=999)
    assert runtime.publish_policy(_command(2, 1.02)).accepted

    start = _step(runtime, sequence=2, timestamp=1.06)
    assert start.phase is B601StagedAssistPhase.BREAKAWAY
    assert start.overlay_torque_nm == ZERO

    pulse = _step(runtime, sequence=3, timestamp=1.08)
    assert pulse.phase is B601StagedAssistPhase.BREAKAWAY
    assert pulse.overlay_torque_nm[2] == pytest.approx(0.08)
    assert pulse.breakaway_impulse_nm_sec == pytest.approx(0.0016)

    onset_one = _step(runtime, sequence=4, timestamp=1.10, q3=0.0012, qvel3=0.0)
    assert onset_one.phase is B601StagedAssistPhase.BREAKAWAY
    assert onset_one.reason == "motion_onset_candidate_pulse_paused"
    assert onset_one.overlay_torque_nm == ZERO
    onset_two = _step(runtime, sequence=5, timestamp=1.12, q3=0.0013, qvel3=0.0)
    assert onset_two.phase is B601StagedAssistPhase.DROP
    assert onset_two.overlay_torque_nm == ZERO

    tracking = _step(runtime, sequence=6, timestamp=1.14, q3=0.0013, qvel3=0.0)
    assert tracking.phase is B601StagedAssistPhase.TRACK
    assert 0.0 < tracking.overlay_torque_nm[2] <= 0.2


@pytest.mark.parametrize("dither", [False, True])
@pytest.mark.parametrize("windowed", [False, True])
def test_one_count_dither_keeps_fast_onset_withdrawal_and_budget_in_v2(dither, windowed):
    motion = B601JointMotionConfig(windowed_soft_velocity=True) if windowed else None
    runtime = _active_runtime(_staged_config(motion_control=motion))
    _step(runtime, sequence=1, timestamp=1.01, dt=.01)
    assert runtime.authorize_breakaway(sequence=1)
    start = _step(runtime, sequence=2, timestamp=1.06, dt=.01)
    assert start.phase is B601StagedAssistPhase.BREAKAWAY
    previous_q, peak, paused = 0., 0., 0
    for i in range(1, 23):
        q = .000384 if dither and i % 2 else 0.
        v = (q - previous_q) / .01
        frame = runtime.step(
            control_sequence=2+i, control_timestamp_sec=1.06+i*.01,
            measured_qpos_rad=(0.,0.,q,0.,0.,0.),
            measured_qvel_rad_s=(0.,0.,v,0.,0.,0.), dt_sec=.01,
            control_qvel_rad_s=ZERO,
            control_qvel_window_span_sec=.08 if windowed else None,
        )
        previous_q = q
        peak = max(peak,abs(frame.overlay_torque_nm[2]))
        if frame.reason == "motion_onset_candidate_pulse_paused":
            paused += 1
            assert frame.overlay_torque_nm == ZERO
        assert frame.phase not in {B601StagedAssistPhase.DROP,B601StagedAssistPhase.TRACK}
        assert frame.breakaway_impulse_nm_sec <= .015 + 1e-12
    assert frame.phase is B601StagedAssistPhase.AUTHORITY_EXHAUSTED
    # Known remaining tradeoff: do not delay pulse withdrawal to cure dither.
    assert paused > 0 if dither else paused == 0
    assert peak == pytest.approx(.04 if dither else .12)


@pytest.mark.parametrize("span", [None, .02])
def test_v2_runtime_refuses_missing_or_incomplete_window(span):
    runtime = _active_runtime(_staged_config(motion_control=B601JointMotionConfig(windowed_soft_velocity=True)))
    frame = runtime.step(
        control_sequence=1, control_timestamp_sec=1.01,
        measured_qpos_rad=ZERO, measured_qvel_rad_s=ZERO, dt_sec=.01,
        control_qvel_rad_s=ZERO, control_qvel_window_span_sec=span,
    )
    assert frame.phase is B601StagedAssistPhase.AUTHORITY_EXHAUSTED
    assert frame.reason == "control_velocity_window_unavailable_assist_withdrawn"
    assert frame.overlay_torque_nm == ZERO
    assert frame.drive_driver_kd_nm_s_per_rad == ZERO


def test_v2_onset_keeps_fast_raw_withdrawal_even_when_window_is_stationary():
    runtime = _active_runtime(_staged_config(motion_control=B601JointMotionConfig(windowed_soft_velocity=True)))
    _step(runtime, sequence=1, timestamp=1.01, dt=.01)
    assert runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06, dt=.01)
    for seq, q in [(3,0.),(4,.000384),(5,.000768)]:
        frame = runtime.step(
            control_sequence=seq, control_timestamp_sec=1.06+(seq-2)*.01,
            measured_qpos_rad=(0.,0.,q,0.,0.,0.),
            measured_qvel_rad_s=(0.,0.,.0384 if seq>3 else 0.,0.,0.,0.),
            dt_sec=.01, control_qvel_rad_s=ZERO, control_qvel_window_span_sec=.08,
        )
        if seq == 3:
            assert frame.overlay_torque_nm[2] > 0
        elif seq == 4:
            assert frame.reason == "motion_onset_candidate_pulse_paused"
            assert frame.phase is B601StagedAssistPhase.BREAKAWAY
        else:
            assert frame.phase is B601StagedAssistPhase.DROP
            assert frame.overlay_torque_nm == ZERO


@pytest.mark.parametrize("direction", [1, -1])
def test_v2_pauses_raw_candidate_before_confirming_sustained_window_motion(direction):
    runtime = B601StagedSingleJointAssistRuntime(_shared_config(), _staged_config(
        motion_control=B601JointMotionConfig(windowed_soft_velocity=True, max_control_gap_sec=.04),
        motion_onset_displacement_rad=.0004,
        motion_onset_confirm_displacement_rad=.0007,
        motion_onset_confirm_samples=2,
    ))
    assert runtime.publish_policy(_command(1, 1.0, target=direction*.1)).accepted
    _step(runtime, sequence=1, timestamp=1.01, dt=.01)
    assert runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06, dt=.01)
    for seq, q, raw, window in [
        (3, 0., 0., 0.),
        (4, .000384, .0384, .003),
        (5, .000768, .0384, .008),
        (6, .000768, 0., .009),
    ]:
        frame = runtime.step(
            control_sequence=seq, control_timestamp_sec=1.06+(seq-2)*.01,
            measured_qpos_rad=(0.,0.,direction*q,0.,0.,0.),
            measured_qvel_rad_s=(0.,0.,direction*raw,0.,0.,0.),
            dt_sec=.01, control_qvel_rad_s=(0.,0.,direction*window,0.,0.,0.),
            control_qvel_window_span_sec=.08,
        )
        _ack_runtime(runtime, frame, 1.06+(seq-2)*.01)
        if seq == 3:
            assert direction * frame.overlay_torque_nm[2] > 0
        else:
            if seq == 4:
                assert frame.overlay_torque_nm == ZERO
            else:
                assert direction*frame.overlay_torque_nm[2] <= 0.0
                assert abs(frame.overlay_torque_nm[2]) <= runtime.config.tracking_overlay_cap_nm
            assert frame.motion_onset_required_samples == 2
        if seq == 4:
            assert frame.motion_onset_confirm_count == 0
        elif seq == 5:
            assert frame.phase is B601StagedAssistPhase.RELEASING
            assert frame.motion_onset_confirm_count == 1
        elif seq == 6:
            assert frame.phase is B601StagedAssistPhase.RELEASING
            assert frame.motion_onset_confirm_count == 2


def test_v2_unconfirmed_persistent_raw_candidate_cannot_extend_pulse_duration():
    runtime = _active_runtime(_staged_config(
        motion_control=B601JointMotionConfig(windowed_soft_velocity=True, max_control_gap_sec=.04),
        motion_onset_displacement_rad=.0004,
        motion_onset_confirm_displacement_rad=.0007,
    ))
    _step(runtime, sequence=1, timestamp=1.01, dt=.01)
    assert runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06, dt=.01)
    for seq in range(3, 24):
        frame = runtime.step(
            control_sequence=seq, control_timestamp_sec=1.06+(seq-2)*.01,
            measured_qpos_rad=(0.,0.,.000384,0.,0.,0.),
            measured_qvel_rad_s=(0.,0.,.0384,0.,0.,0.), dt_sec=.01,
            control_qvel_rad_s=ZERO, control_qvel_window_span_sec=.08,
        )
        _ack_runtime(runtime, frame, 1.06+(seq-2)*.01)
        assert frame.overlay_torque_nm == ZERO
        assert frame.phase not in {B601StagedAssistPhase.DROP,B601StagedAssistPhase.TRACK}
    assert frame.phase in {B601StagedAssistPhase.RELEASING, B601StagedAssistPhase.PASSIVE}


@pytest.mark.parametrize("threshold,samples,motion", [
    (0., 2, B601JointMotionConfig(windowed_soft_velocity=True)),
    (float("nan"), 2, B601JointMotionConfig(windowed_soft_velocity=True)),
    (.0007, 1, B601JointMotionConfig(windowed_soft_velocity=True)),
    (.0003, 2, B601JointMotionConfig(windowed_soft_velocity=True)),
    (.0007, 2, B601JointMotionConfig()),
    (.0007, 2, None),
])
def test_independent_onset_confirmation_rejects_invalid_contract(threshold, samples, motion):
    with pytest.raises(ValueError):
        _staged_config(
            motion_control=motion, motion_onset_displacement_rad=.0004,
            motion_onset_confirm_displacement_rad=threshold,
            motion_onset_confirm_samples=samples,
        )


@pytest.mark.parametrize("q_sequence,window_velocity", [
    ([.000384, 0., .000384, 0.], .01),
    ([.000768, .000384, .000768, .000384], .01),
    ([.000768] * 4, -.001),
])
def test_independent_onset_confirmation_rejects_dither_or_window_reversal(q_sequence, window_velocity):
    runtime = _active_runtime(_staged_config(
        motion_control=B601JointMotionConfig(windowed_soft_velocity=True, max_control_gap_sec=.04),
        motion_onset_displacement_rad=.0004,
        motion_onset_confirm_displacement_rad=.0007,
    ))
    _step(runtime, sequence=1, timestamp=1.01, dt=.01)
    assert runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06, dt=.01)
    previous = 0.
    for seq, q in enumerate(q_sequence, start=3):
        frame = runtime.step(
            control_sequence=seq, control_timestamp_sec=1.06+(seq-2)*.01,
            measured_qpos_rad=(0.,0.,q,0.,0.,0.),
            measured_qvel_rad_s=(0.,0.,(q-previous)/.01,0.,0.,0.), dt_sec=.01,
            control_qvel_rad_s=(0.,0.,window_velocity,0.,0.,0.),
            control_qvel_window_span_sec=.08,
        )
        _ack_runtime(runtime, frame, 1.06+(seq-2)*.01)
        assert frame.phase is B601StagedAssistPhase.RELEASING
        previous = q


def test_late_motion_confirmation_cannot_reuse_expired_pulse_budget():
    runtime = _active_runtime(_staged_config(
        motion_control=B601JointMotionConfig(windowed_soft_velocity=True, max_control_gap_sec=.04),
        motion_onset_displacement_rad=.0004,
        motion_onset_confirm_displacement_rad=.0007,
        breakaway_max_duration_sec=.04,
    ))
    _step(runtime, sequence=1, timestamp=1.01, dt=.01)
    assert runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06, dt=.01)
    _step(runtime, sequence=3, timestamp=1.08, q3=.000768, qvel3=.01, dt=.02)
    frame = _step(runtime, sequence=4, timestamp=1.10, q3=.001152, qvel3=.01, dt=.02)
    assert frame.phase is B601StagedAssistPhase.RELEASING
    assert abs(frame.overlay_torque_nm[2]) <= runtime.config.tracking_overlay_cap_nm
    assert frame.release_terminal
    for seq in range(5, 30):
        frame = _step(runtime, sequence=seq, timestamp=1.10+(seq-4)*.01, q3=.001152, dt=.01)
        assert frame.phase not in {B601StagedAssistPhase.DROP, B601StagedAssistPhase.TRACK}
        if runtime.release_completed:
            break
    assert runtime.release_completed
    frame = _step(runtime, sequence=seq+1, timestamp=1.10+(seq-3)*.01, q3=.001152, dt=.01)
    assert frame.phase is B601StagedAssistPhase.PASSIVE
    assert frame.overlay_torque_nm == ZERO


def test_receipt_completed_after_pulse_deadline_cannot_commit_track():
    runtime = _active_runtime(_staged_config(
        motion_control=B601JointMotionConfig(windowed_soft_velocity=True, max_control_gap_sec=.04),
        motion_onset_displacement_rad=.0004, motion_onset_confirm_displacement_rad=.0007,
        breakaway_max_duration_sec=.08, settle_dwell_sec=.04,
    ))
    _step(runtime, sequence=1, timestamp=1.01, dt=.01)
    assert runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06, dt=.01)
    _step(runtime, sequence=3, timestamp=1.08, q3=.000768, dt=.02)
    _step(runtime, sequence=4, timestamp=1.11, q3=.000768, dt=.03)
    # Stop dwell is complete before 1.14. The receipt crosses that original
    # deadline while otherwise fitting the unchanged 40 ms owner watchdog.
    frame = runtime.step(
        control_sequence=5, control_timestamp_sec=1.13,
        measured_qpos_rad=(0.,0.,.000768,0.,0.,0.), measured_qvel_rad_s=ZERO,
        dt_sec=.02, control_qvel_rad_s=ZERO, control_qvel_window_span_sec=.08,
    )
    assert not frame.release_terminal
    _ack_runtime(runtime, frame, 1.141)
    assert runtime.phase is B601StagedAssistPhase.RELEASING
    for seq in range(6, 30):
        frame = _step(runtime, sequence=seq, timestamp=1.15+(seq-6)*.01, q3=.000768, dt=.01)
        assert frame.phase not in {B601StagedAssistPhase.DROP, B601StagedAssistPhase.TRACK}
        if runtime.release_completed:
            break
    assert runtime.release_completed


@pytest.mark.parametrize("direction", [1, -1])
def test_return_from_negative_counts_pauses_pulse_before_crossing_anchor(direction):
    runtime = B601StagedSingleJointAssistRuntime(_shared_config(), _staged_config(
        motion_control=B601JointMotionConfig(windowed_soft_velocity=True, max_control_gap_sec=.04),
        motion_onset_displacement_rad=.0004,
        motion_onset_confirm_displacement_rad=.0007,
    ))
    assert runtime.publish_policy(_command(1, 1.0, target=direction*.1)).accepted
    _step(runtime, sequence=1, timestamp=1.01, dt=.01)
    assert runtime.authorize_breakaway(sequence=1)
    _step(runtime, sequence=2, timestamp=1.06, dt=.01)
    previous = 0.
    for seq, q in enumerate([-.000384, -.000768, -.001152, -.000768, -.000384, 0.], start=3):
        frame = runtime.step(
            control_sequence=seq, control_timestamp_sec=1.06+(seq-2)*.01,
            measured_qpos_rad=(0.,0.,direction*q,0.,0.,0.),
            measured_qvel_rad_s=(0.,0.,direction*(q-previous)/.01,0.,0.,0.), dt_sec=.01,
            control_qvel_rad_s=ZERO, control_qvel_window_span_sec=.08,
        )
        _ack_runtime(runtime, frame, 1.06+(seq-2)*.01)
        assert frame.phase is (B601StagedAssistPhase.BREAKAWAY if seq < 6 else B601StagedAssistPhase.RELEASING)
        assert frame.motion_onset_confirm_count == 0
        if seq < 6:
            assert 0. < direction*frame.overlay_torque_nm[2] <= runtime.config.breakaway_torque_nm
        else:
            assert frame.overlay_torque_nm == ZERO
        previous = q


def test_console_resolution_threshold_catches_two_count_motion_before_speed_guard() -> None:
    runtime = _active_runtime(
        _staged_config(
            motion_onset_displacement_rad=0.0007,
            motion_onset_velocity_rad_s=0.025,
            motion_onset_confirm_samples=1,
        )
    )
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert runtime.publish_policy(_command(2, 1.02)).accepted
    start = _step(runtime, sequence=2, timestamp=1.06)
    assert start.phase is B601StagedAssistPhase.BREAKAWAY
    pulse = _step(runtime, sequence=3, timestamp=1.08)
    assert pulse.overlay_torque_nm[2] > 0.0

    # Field trace resolution was 0.0003835 rad/count. Two counts at a 20 ms
    # feedback interval are 0.000767 rad and about 0.038 rad/s: below the
    # 0.05 rad/s hard guard, but enough to drop the pulse before the next
    # observed three-count jump crossed that guard.
    onset = _step(
        runtime,
        sequence=4,
        timestamp=1.10,
        q3=0.000767,
        qvel3=0.0192,
    )
    assert onset.phase is B601StagedAssistPhase.DROP
    assert onset.reason == "motion_onset_confirmed_drop_breakaway"
    assert onset.overlay_torque_nm == ZERO

    tracking = _step(
        runtime,
        sequence=5,
        timestamp=1.12,
        q3=0.000767,
        qvel3=0.0192,
    )
    assert tracking.phase is B601StagedAssistPhase.TRACK


def test_cue_latches_at_zero_overlay_until_operator_authorizes_breakaway() -> None:
    runtime = _active_runtime()
    first_cue = _step(runtime, sequence=1, timestamp=1.01)
    assert first_cue.phase is B601StagedAssistPhase.CUE
    assert first_cue.overlay_torque_nm[2] > 0.0

    assert runtime.publish_policy(_command(2, 1.02)).accepted
    waiting = _step(runtime, sequence=2, timestamp=1.06)
    assert waiting.phase is B601StagedAssistPhase.CUE
    assert waiting.reason == "awaiting_breakaway_authorization"
    assert waiting.overlay_torque_nm == ZERO

    assert runtime.authorize_breakaway(sequence=2)
    assert runtime.publish_policy(_command(3, 1.07)).accepted
    starting = _step(runtime, sequence=3, timestamp=1.08)
    assert starting.phase is B601StagedAssistPhase.BREAKAWAY
    assert starting.reason == "authorized_breakaway_started"
    assert starting.overlay_torque_nm == ZERO


def test_breakaway_budget_exhaustion_fails_passive_without_retry() -> None:
    runtime = _active_runtime(_staged_config(breakaway_max_impulse_nm_sec=0.001))
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert runtime.publish_policy(_command(2, 1.02)).accepted
    _step(runtime, sequence=2, timestamp=1.06)

    pulse = _step(runtime, sequence=3, timestamp=1.08)
    assert pulse.overlay_torque_nm[2] == pytest.approx(0.001 / 0.02)
    exhausted = _step(runtime, sequence=4, timestamp=1.10)
    assert exhausted.phase is B601StagedAssistPhase.AUTHORITY_EXHAUSTED
    assert exhausted.reason == "breakaway_impulse_budget_exhausted"
    assert exhausted.overlay_torque_nm == ZERO
    still_exhausted = _step(runtime, sequence=5, timestamp=1.12)
    assert still_exhausted.phase is B601StagedAssistPhase.AUTHORITY_EXHAUSTED
    assert still_exhausted.overlay_torque_nm == ZERO


def test_breakaway_rise_slew_is_independent_of_the_ordinary_small_slew() -> None:
    runtime = _active_runtime(
        _staged_config(
            breakaway_torque_nm=0.20,
            breakaway_rise_slew_nm_s=2.0,
            breakaway_max_duration_sec=0.20,
            breakaway_max_impulse_nm_sec=0.04,
        )
    )
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert runtime.publish_policy(_command(2, 1.02)).accepted
    _step(runtime, sequence=2, timestamp=1.06)

    pulse = _step(runtime, sequence=3, timestamp=1.08, dt=0.02)
    assert pulse.phase is B601StagedAssistPhase.BREAKAWAY
    assert pulse.overlay_torque_nm[2] == pytest.approx(0.04)
    assert pulse.breakaway_impulse_nm_sec == pytest.approx(0.0008)


def test_breakaway_reports_duration_exhaustion_before_impulse_budget() -> None:
    runtime = _active_runtime(
        _staged_config(
            breakaway_max_duration_sec=0.04,
            breakaway_max_impulse_nm_sec=0.04,
        )
    )
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert runtime.publish_policy(_command(2, 1.02)).accepted
    _step(runtime, sequence=2, timestamp=1.06)
    _step(runtime, sequence=3, timestamp=1.08)

    exhausted = _step(runtime, sequence=4, timestamp=1.10)
    assert exhausted.phase is B601StagedAssistPhase.AUTHORITY_EXHAUSTED
    assert exhausted.reason == "breakaway_duration_exhausted_before_impulse_budget"
    assert exhausted.breakaway_impulse_nm_sec < runtime.config.breakaway_max_impulse_nm_sec


def test_explicit_breakaway_revocation_cancels_pending_cue_grant() -> None:
    runtime = _active_runtime()
    cue = _step(runtime, sequence=1, timestamp=1.01)
    assert cue.phase is B601StagedAssistPhase.CUE
    assert runtime.authorize_breakaway(sequence=1)
    runtime.revoke_breakaway_authorization()
    assert runtime.publish_policy(_command(2, 1.02)).accepted

    after_cue = _step(runtime, sequence=2, timestamp=1.05)
    assert after_cue.phase is B601StagedAssistPhase.CUE
    assert after_cue.reason == "awaiting_breakaway_authorization"
    assert after_cue.overlay_torque_nm == ZERO


def test_human_override_drops_an_active_breakaway_and_its_accounting() -> None:
    runtime = _active_runtime(
        _staged_config(
            motion_onset_displacement_rad=0.1,
            motion_onset_velocity_rad_s=1.0,
        )
    )
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert runtime.publish_policy(_command(2, 1.02)).accepted
    started = _step(runtime, sequence=2, timestamp=1.06)
    assert started.phase is B601StagedAssistPhase.BREAKAWAY
    pulse = _step(runtime, sequence=3, timestamp=1.08)
    assert pulse.overlay_torque_nm[2] > 0.0
    assert pulse.breakaway_impulse_nm_sec > 0.0

    overridden = _step(
        runtime,
        sequence=4,
        timestamp=1.10,
        human_override=True,
    )
    assert overridden.phase is B601StagedAssistPhase.HUMAN_OVERRIDE
    assert overridden.overlay_torque_nm == ZERO
    assert overridden.breakaway_impulse_nm_sec == 0.0


def test_breakaway_cancels_immediately_when_target_enters_deadband() -> None:
    runtime = _active_runtime()
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert runtime.publish_policy(_command(2, 1.02)).accepted
    _step(runtime, sequence=2, timestamp=1.06)
    _step(runtime, sequence=3, timestamp=1.08)

    cancelled = _step(runtime, sequence=4, timestamp=1.10, q3=0.09)
    assert cancelled.phase is B601StagedAssistPhase.DROP
    assert cancelled.reason == "breakaway_cancelled_target_within_deadband"
    assert cancelled.overlay_torque_nm == ZERO


def test_breakaway_cancels_if_the_target_moves_behind_the_measured_joint() -> None:
    runtime = _active_runtime(
        _staged_config(
            motion_onset_displacement_rad=0.5,
            motion_onset_velocity_rad_s=1.0,
        )
    )
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert runtime.publish_policy(_command(2, 1.02)).accepted
    _step(runtime, sequence=2, timestamp=1.06)
    pulse = _step(runtime, sequence=3, timestamp=1.08)
    assert pulse.overlay_torque_nm[2] > 0.0

    cancelled = _step(runtime, sequence=4, timestamp=1.10, q3=0.13)
    assert cancelled.phase is B601StagedAssistPhase.DROP
    assert cancelled.reason == "breakaway_cancelled_target_direction_changed"
    assert cancelled.overlay_torque_nm == ZERO


def test_new_intent_invalidates_a_pending_breakaway_grant() -> None:
    runtime = _active_runtime()
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert runtime.publish_policy(_command(2, 1.02, target=0.2)).accepted

    new_cue = _step(runtime, sequence=2, timestamp=1.03)
    assert new_cue.phase is B601StagedAssistPhase.CUE
    assert new_cue.intent_generation == 2
    assert runtime.publish_policy(_command(3, 1.04, target=0.2)).accepted
    after_cue = _step(runtime, sequence=3, timestamp=1.08)
    assert after_cue.phase is B601StagedAssistPhase.CUE
    assert after_cue.reason == "awaiting_breakaway_authorization"


def test_tracking_enters_low_cap_settle_after_position_and_velocity_dwell() -> None:
    runtime = B601StagedSingleJointAssistRuntime(
        _shared_config(), _staged_config(minimum_breakaway_error_rad=0.001)
    )
    assert runtime.publish_policy(_command(1, 1.0, target=0.004)).accepted
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert runtime.publish_policy(_command(2, 1.02, target=0.004)).accepted

    starting = _step(runtime, sequence=2, timestamp=1.06)
    assert starting.phase is B601StagedAssistPhase.BREAKAWAY
    _step(runtime, sequence=3, timestamp=1.08)
    _step(runtime, sequence=4, timestamp=1.10, q3=0.0012)
    dropped = _step(runtime, sequence=5, timestamp=1.12, q3=0.0013)
    assert dropped.phase is B601StagedAssistPhase.DROP
    tracking = _step(runtime, sequence=6, timestamp=1.14, q3=0.0013)
    assert tracking.phase is B601StagedAssistPhase.TRACK
    _step(runtime, sequence=7, timestamp=1.16, q3=0.0013)
    settled = _step(runtime, sequence=8, timestamp=1.23, q3=0.0013)
    assert settled.phase is B601StagedAssistPhase.SETTLE
    assert abs(settled.overlay_torque_nm[2]) <= 0.05


def test_settle_uses_bounded_integral_without_windup_and_override_clears_it() -> None:
    config = _staged_config(
        settle_capture_error_rad=0.05,
        settle_integral_gain_nm_per_rad_sec=10.0,
        settle_integral_cap_nm=0.06,
        settle_overlay_cap_nm=0.1,
    )
    runtime = B601StagedSingleJointAssistRuntime(_shared_config(), config)
    assert runtime.publish_policy(_command(1, 1.0, target=0.04)).accepted
    _step(runtime, sequence=1, timestamp=1.01)
    assert runtime.authorize_breakaway(sequence=1)
    assert runtime.publish_policy(_command(2, 1.02, target=0.04)).accepted
    starting = _step(runtime, sequence=2, timestamp=1.05)
    assert starting.phase is B601StagedAssistPhase.BREAKAWAY

    sequence = 3
    frame = None
    for timestamp, q3 in (
        (1.07, 0.0),
        (1.09, 0.0012),
        (1.11, 0.0013),
        (1.13, 0.0013),
    ):
        assert runtime.publish_policy(_command(sequence, timestamp, target=0.04)).accepted
        frame = _step(runtime, sequence=sequence, timestamp=timestamp, q3=q3)
        sequence += 1
    assert frame is not None and frame.phase is B601StagedAssistPhase.TRACK

    for timestamp in (1.15, 1.17, 1.19, 1.21, 1.23, 1.25, 1.27, 1.29, 1.31):
        assert runtime.publish_policy(_command(sequence, timestamp, target=0.04)).accepted
        frame = _step(runtime, sequence=sequence, timestamp=timestamp, q3=0.0013)
        sequence += 1

    assert frame.phase is B601StagedAssistPhase.SETTLE
    assert 0.0 < frame.settle_integral_torque_nm <= 0.06
    assert abs(frame.overlay_torque_nm[2]) <= 0.1

    overridden = _step(
        runtime,
        sequence=sequence,
        timestamp=1.40,
        human_override=True,
    )
    assert overridden.phase is B601StagedAssistPhase.HUMAN_OVERRIDE
    assert overridden.settle_integral_torque_nm == 0.0
    assert overridden.overlay_torque_nm == ZERO


def test_stale_command_and_human_override_immediately_remove_staged_overlay() -> None:
    runtime = _active_runtime()
    cue = _step(runtime, sequence=1, timestamp=1.01)
    assert cue.overlay_torque_nm[2] > 0.0

    stale = _step(runtime, sequence=2, timestamp=2.2)
    assert stale.phase is B601StagedAssistPhase.PASSIVE
    assert stale.overlay_torque_nm == ZERO

    assert runtime.publish_policy(_command(2, 2.21)).accepted
    _step(runtime, sequence=3, timestamp=2.22)
    overridden = _step(
        runtime,
        sequence=4,
        timestamp=2.23,
        human_override=True,
    )
    assert overridden.phase is B601StagedAssistPhase.HUMAN_OVERRIDE
    assert overridden.overlay_torque_nm == ZERO


def test_non_target_shared_assist_is_rejected() -> None:
    invalid = B601SharedControlConfig(
        position_kp_nm_per_rad=(1.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        velocity_kd_nm_s_per_rad=ZERO,
        max_target_error_rad=(0.2,) * 6,
        max_abs_assist_torque_nm=(0.3,) * 6,
        max_assist_torque_rate_nm_s=(10.0,) * 6,
        authority_rise_per_sec=100.0,
        policy_stale_timeout_sec=1.0,
    )
    with pytest.raises(ValueError, match="non-target joint"):
        B601StagedSingleJointAssistRuntime(invalid, _staged_config())


def test_phase_torque_caps_cannot_exceed_existing_shared_control_cap() -> None:
    with pytest.raises(ValueError, match="max_total_overlay_nm"):
        replace(_staged_config(), max_total_overlay_nm=0.1)


@pytest.mark.parametrize("direction", (1.0, -1.0), ids=("positive", "negative"))
def test_staged_assist_releases_synthetic_dry_friction_and_settles_in_mujoco(
    direction: float,
) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    repo_root = Path(__file__).resolve().parents[1]
    source_root = Path(
        os.environ.get(
            "B601_REBOT_SOURCE_ROOT",
            repo_root / "hardware" / "reBotArm_control_py",
        )
    )
    if not (source_root / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("set B601_REBOT_SOURCE_ROOT to the pinned ReBot source checkout")

    from robot_motion.rebot_b601_rs import B601PinocchioModel
    from sim.contracts import TimingSpec
    from sim.mujoco.rebot_b601_rs.mit_driver import (
        B601MujocoEffortDriver,
        default_b601_mit_profile,
    )
    from sim.mujoco.rebot_b601_rs.real2sim import (
        B601Real2SimControllerProfile,
        B601Real2SimPlantProfile,
    )
    from sim.mujoco.rebot_b601_rs.scene import (
        B601_RS_DEFAULT_ARM_QPOS,
        B601_RS_PHYSICS_HZ,
        build_b601_mujoco_scene,
    )

    plant = B601Real2SimPlantProfile(
        name="staged_assist_synthetic_joint3_friction",
        parameter_source="test_only_not_hardware_identified",
        frictionloss_nm=(0.0, 0.0, 0.06, 0.0, 0.0, 0.0),
    )
    scene = build_b601_mujoco_scene(
        source_root,
        initial_arm_qpos=B601_RS_DEFAULT_ARM_QPOS,
        gravity_mps2=(0.0, 0.0, -9.81),
        actuation_mode="effort",
        plant_profile=plant,
    )
    model = B601PinocchioModel.from_rebot_source_root(source_root)
    shared_config = B601SharedControlConfig(
        position_kp_nm_per_rad=(0.0, 0.0, 0.8, 0.0, 0.0, 0.0),
        velocity_kd_nm_s_per_rad=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        max_target_error_rad=(0.15,) * 6,
        max_abs_assist_torque_nm=(0.12,) * 6,
        max_assist_torque_rate_nm_s=(20.0,) * 6,
        authority_rise_per_sec=100.0,
        policy_stale_timeout_sec=0.08,
    )
    staged_config = _staged_config(
        cue_torque_nm=0.02,
        cue_duration_sec=0.10,
        cue_rise_slew_nm_s=1.0,
        breakaway_torque_nm=0.12,
        breakaway_rise_slew_nm_s=8.0,
        breakaway_max_duration_sec=0.2,
        breakaway_max_impulse_nm_sec=0.01,
        minimum_breakaway_error_rad=0.01,
        stuck_max_velocity_rad_s=0.01,
        motion_onset_displacement_rad=0.0002,
        motion_onset_velocity_rad_s=0.01,
        motion_onset_confirm_samples=2,
        tracking_overlay_cap_nm=0.08,
        tracking_slew_nm_s=1.0,
        settle_error_rad=0.003,
        settle_capture_error_rad=0.05,
        settle_velocity_rad_s=0.01,
        settle_dwell_sec=0.05,
        settle_integral_gain_nm_per_rad_sec=1.0,
        settle_integral_leak_per_sec=0.2,
        settle_integral_cap_nm=0.07,
        settle_overlay_cap_nm=0.08,
        max_total_overlay_nm=0.12,
    )
    runtime = B601StagedSingleJointAssistRuntime(shared_config, staged_config)
    profile = replace(
        default_b601_mit_profile(scene),
        damping_kd_nm_s_per_rad=ZERO,
        max_abs_overlay_torque_nm=shared_config.max_abs_assist_torque_nm,
        max_abs_state_acceleration_rad_s2=(250.0,) * 6,
    )
    driver = B601MujocoEffortDriver(
        scene,
        model,
        timing=TimingSpec(
            command_hz=100.0,
            physics_hz=B601_RS_PHYSICS_HZ,
            parameter_source="staged_assist_synthetic_friction_test",
        ),
        profile=profile,
        enable_gravity_compensation=True,
        plant_profile=plant,
        real2sim_profile=B601Real2SimControllerProfile.raw_model_baseline(),
    )

    initial_qpos = tuple(B601_RS_DEFAULT_ARM_QPOS)
    target_q3 = initial_qpos[2] + direction * 0.04
    control_sequence = 0
    breakaway_authorized = False
    cue_displacement_rad = 0.0
    max_overlay_nm = 0.0
    max_breakaway_impulse_nm_sec = 0.0
    max_abs_qvel_rad_s = 0.0
    max_target_displacement_rad = 0.0
    max_joint_drift_rad = [0.0] * 6
    phases: set[B601StagedAssistPhase] = set()
    drop_frames = []
    final_frame = None
    final_tick = None

    for _ in range(round(6 * B601_RS_PHYSICS_HZ)):
        reference = None
        if driver.command_due:
            now = driver.physics_time_sec
            qpos = tuple(
                float(scene.data.qpos[scene.joint_qpos_addr(name)])
                for name in B601_RS_ARM_JOINT_NAMES
            )
            qvel = tuple(
                float(scene.data.qvel[scene.joint_dof_addr(name)])
                for name in B601_RS_ARM_JOINT_NAMES
            )
            control_sequence += 1
            if now >= 0.1:
                runtime.publish_policy(
                    B601PolicyAssistCommand(
                        sequence=control_sequence,
                        source_timestamp_sec=now,
                        receive_timestamp_sec=now,
                        qpos_target_rad=(
                            qpos[0],
                            qpos[1],
                            target_q3,
                            qpos[3],
                            qpos[4],
                            qpos[5],
                        ),
                    )
                )
            final_frame = runtime.step(
                control_sequence=control_sequence,
                control_timestamp_sec=now,
                measured_qpos_rad=qpos,
                measured_qvel_rad_s=qvel,
                dt_sec=0.01,
            )
            if (
                final_frame.phase is B601StagedAssistPhase.CUE
                and not breakaway_authorized
            ):
                breakaway_authorized = runtime.authorize_breakaway(
                    sequence=control_sequence
                )
            phases.add(final_frame.phase)
            cue_displacement_rad = max(
                cue_displacement_rad,
                abs(qpos[2] - initial_qpos[2])
                if final_frame.phase is B601StagedAssistPhase.CUE
                else 0.0,
            )
            max_overlay_nm = max(
                max_overlay_nm,
                *(abs(value) for value in final_frame.overlay_torque_nm),
            )
            max_breakaway_impulse_nm_sec = max(
                max_breakaway_impulse_nm_sec,
                final_frame.breakaway_impulse_nm_sec,
            )
            if final_frame.phase is B601StagedAssistPhase.DROP:
                drop_frames.append(final_frame)
            reference = final_frame.reference

        final_tick = driver.step_physics(reference)
        max_abs_qvel_rad_s = max(
            max_abs_qvel_rad_s,
            *(abs(value) for value in final_tick.measured_qvel),
        )
        max_target_displacement_rad = max(
            max_target_displacement_rad,
            abs(final_tick.measured_qpos[2] - initial_qpos[2]),
        )
        for index in range(6):
            max_joint_drift_rad[index] = max(
                max_joint_drift_rad[index],
                abs(final_tick.measured_qpos[index] - initial_qpos[index]),
            )
        if final_tick.driver_faulted:
            break

    assert final_frame is not None
    assert final_tick is not None
    final_error_rad = target_q3 - final_tick.measured_qpos[2]
    assert breakaway_authorized is True
    assert B601StagedAssistPhase.CUE in phases
    assert B601StagedAssistPhase.BREAKAWAY in phases
    assert B601StagedAssistPhase.DROP in phases
    assert B601StagedAssistPhase.TRACK in phases
    assert B601StagedAssistPhase.SETTLE in phases
    assert cue_displacement_rad < 1e-4
    assert len(drop_frames) == 1
    assert drop_frames[0].reason == "motion_onset_confirmed_drop_breakaway"
    assert drop_frames[0].overlay_torque_nm == ZERO
    assert 0.0 < max_breakaway_impulse_nm_sec <= staged_config.breakaway_max_impulse_nm_sec
    assert max_overlay_nm <= staged_config.max_total_overlay_nm
    assert max_abs_qvel_rad_s < 0.05
    assert max_target_displacement_rad <= 0.045
    assert abs(final_error_rad) <= staged_config.settle_error_rad
    assert abs(final_tick.measured_qvel[2]) <= staged_config.settle_velocity_rad_s
    # This is an unanchored gravity-compensated plant: small passive coupled
    # drift on other axes is expected and is not an assist-output assertion.
    assert max(max_joint_drift_rad[index] for index in (0, 1, 3, 4, 5)) <= 0.01, (
        f"max per-joint drift={max_joint_drift_rad!r}; "
        f"peak velocity={max_abs_qvel_rad_s:.6f} rad/s; "
        f"target displacement={max_target_displacement_rad:.6f} rad; "
        f"final target error={final_error_rad:.6f} rad"
    )
    assert final_tick.driver_faulted is False
