"""Offline tests for staged-assist composition and loopback console intents."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from control_station.b601_torque_console import B601TorqueConsoleServer
from control_station.b601_torque_console.staged_controller import (
    B601StagedTorqueConsoleController,
    B601StagedTorqueConsoleSettings,
)
from robot_motion.rebot_b601_rs import B601StagedAssistConfig
from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionConfig
from robot_motion.rebot_b601_rs.model import B601_RS_ARM_JOINT_NAMES
from robot_motion.rebot_b601_rs.torque_experiment import B601TorqueExperimentLimits
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
from scripts.hardware import rebot_b601_rs_owner as owner

ZERO = (0.0,) * 6


def test_direct_velocity_debug_gui_distinguishes_request_zero_confirmation_and_manual_rearm():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the pure GUI message contract")
    source = Path("control_station/b601_torque_console/static/app.js").read_text()
    functions = source[source.index("function number("):source.index("function post(")]
    script = """
const vm = require('node:vm');
const context = {};
vm.runInNewContext(JSON.parse(process.argv[1]), context);
const state = {controller_mode:'direct_torque',phase:'velocity_check',owner_ready:true,
  feedback_age_sec:0.01,control_qvel_rad_s:[0,0,-0.0013,0,0,0],
  quantized_velocity_debug:{zero_dispatch_confirmed:false,confirmation_window_sec:0.08}};
console.log(JSON.stringify([
  context.gateMessage(state),
  context.gateMessage({...state,quantized_velocity_debug:{...state.quantized_velocity_debug,zero_dispatch_confirmed:true}}),
  context.gateMessage({...state,phase:'inactive',quantized_velocity_debug:{reason:'zero_confirmed_manual_rearm_required'}}),
  context.gateMessage({...state,phase:'fault'}),
  context.gateMessage({...state,phase:'inactive',quantized_velocity_debug:null}),
  context.gateMessage({...state,quantized_velocity_debug:{...state.quantized_velocity_debug,
    zero_dispatch_confirmed:true,confirmation_first_feedback_at_sec:1.2,confirmation_position_span_rad:0.000383496}})
]));
"""
    result = subprocess.run([node, "-e", script, json.dumps(functions)], check=True, capture_output=True, text=True)
    messages = json.loads(result.stdout)
    assert "等待最终六轴零助力派发" in messages[0]
    assert "此时不能调力或重新授权" in messages[0]
    assert "六轴零助力派发已确认" in messages[1]
    assert "等待零派发后的首帧实测反馈" in messages[1]
    assert "不是实测速度安全证明" in messages[2] and "不会自动恢复" in messages[2]
    assert "最终发送状态请查看 owner 回执和保护原因" in messages[3]
    assert messages[4] == "当前选择仅重力补偿模式，最终发送状态见回执。添加驱动力前，请让六轴静止。"
    assert "零派发后位置范围 0.000383 rad" in messages[5]
    assert "撤力后观察窗口尚未完整" in messages[5]


def test_explicit_debug_gui_reports_rejected_preparation_not_endless_waiting():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the pure GUI message contract")
    source = Path("control_station/b601_torque_console/static/app.js").read_text()
    functions = source[source.index("function number("):source.index("function post(")]
    functions += source[source.index("function lastOperatorEvent("):source.index("async function refresh(")]
    script = """
const vm = require('node:vm'); const context = {};
vm.runInNewContext(JSON.parse(process.argv[1]), context);
const state = {controller_mode:'direct_torque',phase:'inactive',owner_ready:true,feedback_age_sec:0.01,
  joint_name:'joint3',dispatch_age_sec:.01,
  last_dispatch:{joint_name:'joint3',feedback_timestamp_sec:1.1,overlay_after_profile_nm:0,
    gravity_nm:6.2,conditioned_total_nm:6.2,
    joint_commands:{gravity_nm:[0,0,6.2,0,0,0],overlay_nm:[0,0,0,0,0,0],total_nm:[0,0,6.2,0,0,0]}},
  fault:'arm_rejected_raw_velocity_hard_guard', events:[{action:'arm',time_sec:1}],
  quantized_velocity_debug:{preparation_gate_reason:'ready'}};
console.log(JSON.stringify([context.commandState(state),context.gateMessage(state),
  context.commandState({...state,phase:'armed',fault:null}),
  context.commandState({...state,phase:'arm_requested',fault:null}),
  context.commandState({...state,quantized_velocity_debug:null}),
  context.commandState({...state,phase:'armed',fault:null,last_dispatch:null})]));
"""
    result = subprocess.run([node, "-e", script, json.dumps(functions)], check=True, capture_output=True, text=True)
    messages = json.loads(result.stdout)
    assert "准备已拒绝" in messages[0] and "不会继续等待或自动授权" in messages[0]
    assert "原始差分速度超过硬门限" in messages[0]
    assert "当前准备门：六轴窗口速度、位移范围和零助力派发已通过" in messages[1]
    assert "最终六轴零助力命令已确认" in messages[2]
    assert "等待 owner 下一周期判定" in messages[3]
    assert messages[4] == "添加驱动力请求已接收；等待六轴静止检查。"
    assert "最终零助力发送尚未确认" in messages[5]


def test_stop_capture_gui_reports_gate_and_never_keeps_stale_success_visible():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the pure GUI message contract")
    source = Path("control_station/b601_torque_console/static/app.js").read_text()
    functions = source[source.index("function number("):source.index("function isStaged(")]
    script = """
const vm = require('node:vm');
const context = {};
vm.runInNewContext(JSON.parse(process.argv[1]), context);
const state = {motion_control_enabled:true,owner_ready:true,operator_armed:true,
  feedback_age_sec:0.01,phase:'brake',motion_control:{phase:'brake',
    stop_capture_block_reason:'selected_raw_velocity',stop_capture_dwell_elapsed_sec:0,
    stop_capture_required_dwell_sec:0.15,stop_capture_velocity_limit_rad_s:0.005}};
const raw = context.stopCaptureMessage(state);
const dwell = context.stopCaptureMessage({...state,motion_control:{...state.motion_control,
  stop_capture_block_reason:'stationary_dwell_incomplete',stop_capture_dwell_elapsed_sec:0.05}});
const held = context.stopCaptureMessage({...state,phase:'soft_hold',
  motion_control:{...state.motion_control,phase:'soft_hold'}});
const hidden = [{...state,owner_ready:false},{...state,operator_armed:false},
  {...state,feedback_age_sec:0.2},{...state,feedback_age_sec:null},{...state,phase:'fault'},
  {...state,phase:'inactive'},{...state,motion_control_enabled:false},null,
  {...state,feedback_age_sec:undefined},{...state,feedback_age_sec:-0.1},
  {...state,phase:'soft_hold'}]
  .map(value => context.stopCaptureMessage(value));
console.log(JSON.stringify({raw,dwell,held,hidden}));
"""
    result = subprocess.run([node, "-e", script, json.dumps(functions)], check=True, capture_output=True, text=True)
    messages = json.loads(result.stdout)
    assert "瞬时位置差分速度" in messages["raw"]
    assert "不是硬保护限速" in messages["raw"]
    assert "0.050 / 0.150 s" in messages["dwell"]
    assert "停点保持不等于到位" in messages["held"]
    assert messages["hidden"] == [None] * 11


def _controller(
    *,
    joint_name: str = "joint3",
    max_abs_assist_nm: float = 0.30,
    max_support_drift_rad: float = 0.02,
    command_timeout_sec: float = 0.35,
    breakaway_impulse_nm_sec: float = 0.015,
    breakaway_torque_nm: float = 0.12,
    breakaway_rise_slew_nm_s: float = 0.15,
    authority_rise_per_sec: float = 2.0,
    cue_duration_sec: float = 0.06,
    nominal_control_period_sec: float = 0.02,
    motion_control: B601JointMotionConfig | None = None,
) -> B601StagedTorqueConsoleController:
    limits = B601TorqueExperimentLimits(
        joint_name=joint_name,
        max_abs_assist_nm=max_abs_assist_nm,
        rise_slew_nm_s=0.15,
        release_slew_nm_s=2.0,
        command_timeout_sec=command_timeout_sec,
        stationarity_limit_rad_s=0.02,
        max_joint_velocity_rad_s=0.50,
        max_excursion_rad=0.15,
        max_support_drift_rad=max_support_drift_rad,
    )
    staged = B601StagedAssistConfig(
        joint_name=joint_name,
        cue_torque_nm=0.04,
        cue_duration_sec=cue_duration_sec,
        cue_rise_slew_nm_s=0.15,
        breakaway_torque_nm=breakaway_torque_nm,
        breakaway_rise_slew_nm_s=breakaway_rise_slew_nm_s,
        breakaway_max_duration_sec=0.20,
        breakaway_max_impulse_nm_sec=breakaway_impulse_nm_sec,
        minimum_breakaway_error_rad=0.002,
        stuck_max_velocity_rad_s=0.02,
        motion_onset_displacement_rad=0.001,
        motion_onset_velocity_rad_s=0.02,
        motion_onset_confirm_samples=2,
        new_intent_delta_rad=0.01,
        tracking_overlay_cap_nm=0.10,
        tracking_slew_nm_s=0.15,
        settle_error_rad=0.002,
        settle_capture_error_rad=0.01,
        settle_velocity_rad_s=0.005,
        settle_dwell_sec=0.06,
        settle_integral_gain_nm_per_rad_sec=0.5,
        settle_integral_leak_per_sec=1.0,
        settle_integral_cap_nm=0.02,
        settle_overlay_cap_nm=0.05,
        max_total_overlay_nm=max_abs_assist_nm,
        motion_control=motion_control,
    )
    return B601StagedTorqueConsoleController(
        B601StagedTorqueConsoleSettings(
            limits=limits,
            staged_assist=staged,
            maximum_target_offset_rad=0.10,
            tracking_kp_nm_per_rad=0.6,
            tracking_kd_nm_s_per_rad=0.0,
            authority_rise_per_sec=authority_rise_per_sec,
            nominal_control_period_sec=nominal_control_period_sec,
        )
    )


def test_motion_profile_launcher_roundtrip_does_not_change_gravity_or_hard_guards(tmp_path):
    profile = Path(__file__).resolve().parents[1] / "docs/hardware/profiles/b601_j3_interaction_v1.json"
    args = owner._parser().parse_args([
        "--allow-motion", "--torque-console", "--gravity-debug-mode", "--debug-dynamics-trace",
        "--torque-console-motion-profile", str(profile),
        "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE",
    ])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
    probe._validate_torque_console_cli(parsed)
    settings = probe._staged_torque_console_settings_from_args(parsed)
    assert settings.staged_assist.motion_control == B601JointMotionConfig()
    assert settings.tracking_kp_nm_per_rad == 6.0
    assert settings.tracking_kd_nm_s_per_rad == 1.0
    assert parsed.kp == parsed.kd == 0.0
    assert tuple(parsed.driver_kd) == ZERO
    assert tuple(parsed.gravity_joint_scales) == (1, .96, 1, 1, 1, 1)
    assert tuple(parsed.gravity_joint_bias_nm) == (0, -.22, -.11, -.08, 0, 0)
    assert parsed.debug_dynamics_trace is True
    assert settings.limits.max_joint_velocity_rad_s == .05
    assert settings.limits.max_support_drift_rad == pytest.approx(.008726646259971648)
    assert settings.staged_assist.breakaway_torque_nm == .2
    assert owner._parser().parse_args([]).torque_console_motion_profile is None


def test_motion_profile_rejects_unknown_parameters_and_non_j2_j3_before_connection(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"not_a_parameter": 1}))
    args = owner._parser().parse_args([
        "--allow-motion", "--torque-console", "--gravity-debug-mode",
        "--torque-console-motion-profile", str(profile),
        "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE",
    ])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
    with pytest.raises(TypeError, match="not_a_parameter"):
        probe._staged_torque_console_settings_from_args(parsed)
    profile.write_text("{}")
    parsed.torque_console_joint = "joint4"
    with pytest.raises(ValueError, match="J2/J3 only"):
        probe._staged_torque_console_settings_from_args(parsed)


def test_v2_profile_is_explicit_and_requires_declared_owner_cadence(tmp_path):
    profile = Path(__file__).resolve().parents[1] / "docs/hardware/profiles/b601_j3_interaction_v2.json"
    args = owner._parser().parse_args([
        "--allow-motion", "--torque-console", "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE",
        "--torque-console-motion-profile", str(profile), "--rate-hz", "100",
        "--torque-console-staged-motion-onset-rad", ".0004",
    ])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
    probe._validate_cli(parsed)
    settings = probe._staged_torque_console_settings_from_args(parsed)
    assert settings.staged_assist.motion_control.drive_driver_kd_nm_s_per_rad == (.4,.6,.4,.6,.2,.1)
    assert settings.staged_assist.motion_control.windowed_soft_velocity is True
    assert tuple(parsed.driver_kd) == ZERO
    assert parsed.kp == parsed.kd == 0
    parsed.rate_hz = 50
    with pytest.raises(ValueError, match="rate"):
        probe._staged_torque_console_settings_from_args(parsed)


@pytest.mark.parametrize("peak,slew,impulse,accepted", [
    (.2, 1, .04, False), (.8, 2, .04, False),
    (.8, 10, .034, False), (.8, 10, .08, True),
])
def test_live_motion_cli_checks_coupled_breakaway_envelope_before_connection(tmp_path, peak, slew, impulse, accepted):
    profile = Path(__file__).resolve().parents[1] / "docs/hardware/profiles/b601_j3_interaction_v2.json"
    args = owner._parser().parse_args([
        "--allow-motion", "--torque-console", "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE",
        "--torque-console-motion-profile", str(profile), "--rate-hz", "100",
        "--torque-console-staged-motion-onset-rad", ".0004",
        "--torque-console-max-assist-nm", str(max(.3, peak)),
        "--torque-console-staged-breakaway-torque-nm", str(peak),
        "--torque-console-staged-breakaway-rise-slew-nm-s", str(slew),
        "--torque-console-staged-breakaway-impulse-nm-sec", str(impulse),
    ])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
    if not accepted:
        with pytest.raises(ValueError, match="breakaway peak is unreachable"):
            probe._validate_cli(parsed)
    else:
        probe._validate_cli(parsed)
        settings = probe._staged_torque_console_settings_from_args(parsed)
        assert settings.staged_assist.breakaway_envelope(control_period_sec=.01)["nominal_peak_within_optimistic_ceiling"]
        assert parsed.kp == parsed.kd == 0
        assert tuple(parsed.driver_kd) == ZERO
        assert tuple(parsed.gravity_joint_bias_nm) == (0, -.22, -.11, -.08, 0, 0)


def test_owner_native_damping_is_a_phase_output_not_gui_input():
    controller = _controller(motion_control=B601JointMotionConfig(drive_driver_kd_nm_s_per_rad=(.4,) * 6))
    _arm_and_set_target(controller)
    _step(controller, now=1.02)
    base = (.01,) * 6
    assert probe._motion_console_driver_kd(controller, base) == pytest.approx((.04,) * 6)
    controller.submit("withdraw", now_sec=1.025)
    assert probe._motion_console_driver_kd(controller, base) == base
    _step(controller, now=1.04)
    assert controller.status()["drive_driver_kd_nm_s_per_rad"] == ZERO
    assert probe._motion_console_driver_kd(object(), base) == base


@pytest.mark.parametrize("joint_index", [0, 1, 3, 4, 5])
@pytest.mark.parametrize("resolution", [None, .0004])
def test_drive_profile_support_raw_hard_guard_withdraws_before_damping_dispatch(joint_index, resolution):
    base = _controller(motion_control=B601JointMotionConfig(
        drive_driver_kd_nm_s_per_rad=(.4,) * 6, windowed_soft_velocity=True,
    ))
    controller = B601StagedTorqueConsoleController(replace(
        base.settings,
        limits=replace(base.limits, max_joint_velocity_rad_s=.05),
        staged_assist=replace(
            base.settings.staged_assist, motion_onset_displacement_rad=.0004,
            motion_onset_confirm_displacement_rad=.0007 if resolution is not None else None,
            motion_control=replace(
                base.settings.staged_assist.motion_control,
                position_resolution_rad=resolution,
                responsive_tracking_brake=resolution is not None,
            ),
        ),
        nominal_control_period_sec=.01,
    ))
    _arm_and_set_target(controller)
    _step(controller, now=1.02)
    if resolution is not None:
        _record_confirmed_dispatch(controller, 1.02)
    velocity = list(ZERO)
    velocity[joint_index] = .051
    overlay, audit = _step(controller, now=1.04, qvel=tuple(velocity))
    assert overlay == ZERO
    assert audit["phase"] == "fault"
    assert audit["fault"] == (
        f"support_joint_velocity_exceeded: joint{joint_index+1} "
        "abs_qvel=0.051000rad/s limit=0.050000rad/s"
    )
    assert audit["control_qvel_rad_s"][joint_index] == 0
    assert probe._motion_console_driver_kd(controller, ZERO) == ZERO
    assert probe._motion_protective_driver_kd(controller.settings, ZERO) == (.4,) * 6
    assert probe._motion_protective_driver_kd(controller.settings, (.7,) * 6) == (.7,) * 6


def test_quantized_support_soft_feedback_uses_window_but_reports_raw():
    controller = _controller(motion_control=B601JointMotionConfig(
        drive_driver_kd_nm_s_per_rad=(.4,) * 6, windowed_soft_velocity=True,
    ))
    _arm_and_set_target(controller)
    for i in range(1, 10):
        _step(controller, now=1.0 + i * .01)
    q = (0, 0, 0, .000384, 0, 0)
    raw = (0, 0, 0, .0384, 0, 0)
    _, audit = _step(controller, now=1.1, qpos=q, qvel=raw)
    assert audit["phase"] == "cue"
    assert audit["support_raw_max_velocity_rad_s"] == pytest.approx(.0384)
    assert audit["support_max_velocity_rad_s"] < .01
    assert controller.status()["velocity_window_span_sec"] > .05


def _v2_controller():
    base = _controller(
        motion_control=B601JointMotionConfig(
            drive_driver_kd_nm_s_per_rad=(.4,) * 6, windowed_soft_velocity=True,
            max_control_gap_sec=.04, minimum_owner_rate_hz=100.,
        ),
        nominal_control_period_sec=.01, breakaway_rise_slew_nm_s=4.,
    )
    return B601StagedTorqueConsoleController(replace(
        base.settings, limits=replace(base.limits,max_joint_velocity_rad_s=.05),
        staged_assist=replace(base.settings.staged_assist,motion_onset_displacement_rad=.0004),
    ))


@pytest.mark.parametrize("capture_cap", [0., .10])
def test_live_console_rejects_offline_continuous_handoff_candidate(capture_cap):
    base = _v2_controller()
    with pytest.raises(ValueError, match="offline simulation candidate"):
        replace(base.settings, staged_assist=replace(
            base.settings.staged_assist, motion_onset_confirm_displacement_rad=.0007,
            motion_onset_confirm_samples=2, continuous_motion_handoff=True,
            capture_overlay_cap_nm=capture_cap,
        ))


def test_live_console_rejects_reference_pacing_without_explicit_tracking_recovery():
    base = _v2_controller()
    with pytest.raises(ValueError, match="reference pacing requires explicit"):
        replace(base.settings, staged_assist=replace(
            base.settings.staged_assist, motion_control=replace(
                base.settings.staged_assist.motion_control, reference_pacing=True,
            ),
        ))


def test_confirmed_motion_cli_forwards_larger_reachable_envelope_without_weakening_guards(tmp_path):
    profile = Path(__file__).resolve().parents[1] / "docs/hardware/profiles/b601_j3_interaction_v2.json"
    args = owner._parser().parse_args([
        "--allow-motion", "--torque-console", "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE",
        "--torque-console-motion-profile", str(profile), "--rate-hz", "100",
        "--torque-console-staged-motion-onset-rad", ".0004",
        "--torque-console-staged-motion-confirm-rad", ".0007",
        "--torque-console-staged-motion-confirm-samples", "2",
        "--torque-console-max-assist-nm", "1.0",
        "--torque-console-staged-breakaway-torque-nm", "1.0",
        "--torque-console-staged-breakaway-rise-slew-nm-s", "1.0",
        "--torque-console-staged-breakaway-duration-sec", "1.2",
        "--torque-console-staged-breakaway-impulse-nm-sec", ".6",
    ])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
    probe._validate_cli(parsed)
    settings = probe._staged_torque_console_settings_from_args(parsed)
    assert settings.staged_assist.motion_onset_confirm_displacement_rad == .0007
    assert settings.staged_assist.motion_onset_confirm_samples == 2
    assert settings.staged_assist.breakaway_envelope(control_period_sec=.01)["nominal_peak_within_optimistic_ceiling"]
    assert settings.limits.max_joint_velocity_rad_s == .05
    assert settings.staged_assist.motion_control.max_control_gap_sec == .04
    parsed.torque_console_staged_motion_confirm_samples = 1
    with pytest.raises(ValueError, match="at least two"):
        probe._validate_cli(parsed)
    parsed.torque_console_staged_motion_confirm_samples = 2
    parsed.driver_kd = [.01]*6
    with pytest.raises(probe.LowGainProbeError, match="zero base"):
        probe._validate_cli(parsed)


def test_budget_completed_confirmed_motion_attempt_returns_to_gravity_only_without_auto_retry():
    base = _v2_controller()
    controller = B601StagedTorqueConsoleController(replace(
        base.settings, staged_assist=replace(
            base.settings.staged_assist,
            motion_onset_confirm_displacement_rad=.0007,
            motion_onset_confirm_samples=2,
        ),
    ))
    controller.owner_ready()
    for i in range(11):
        _step(controller, now=1.+i*.01)
        _record_confirmed_dispatch(controller, 1.+i*.01)
    controller.submit("arm", now_sec=1.105)
    _step(controller, now=1.11)
    _record_confirmed_dispatch(controller, 1.11)
    controller.submit("set_target", now_sec=1.115, target_offset_rad=.04)
    _step(controller, now=1.12)
    _record_confirmed_dispatch(controller, 1.12)
    controller.submit("authorize_breakaway", now_sec=1.121)
    for i in range(1,81):
        now = 1.12+i*.01
        if controller.status()["operator_armed"]:
            controller.submit("heartbeat", now_sec=now)
        overlay, audit = _step(controller, now=now)
        _record_confirmed_dispatch(controller, now)
        if controller.status()["phase"] == "inactive":
            break
    assert controller.status()["phase"] == "inactive"
    assert audit["target_offset_rad"] is None
    assert not audit["operator_armed"]
    assert overlay == ZERO
    assert audit["drive_driver_kd_nm_s_per_rad"] == ZERO
    assert any(e.get("reason") == "conditioned_withdrawal_and_stop_confirmed" for e in controller.status()["events"])
    # The one-count spike from the field trace arrives after the action ended.
    # This is no longer an authorized drive; the hardware gravity-only owner
    # still independently applies its original velocity/acceleration guards.
    overlay, audit = _step(controller, now=now+.007612826, qpos=(0.,-.000383496,0.,.000383496,0.,0.),
                           qvel=(0.,-.050375,0.,.050375,0.,0.))
    assert audit["phase"] == "inactive"
    assert overlay == ZERO
    assert audit["fault"] is None
    with pytest.raises(ValueError):
        controller.submit("authorize_breakaway", now_sec=now+.009)


def _record_confirmed_dispatch(controller, now, final=None):
    audit = controller.status()["history"][-1]
    overlay = audit["overlay_torque_nm"]
    kd = audit["drive_driver_kd_nm_s_per_rad"]
    controller.record_dispatch(
        joint_name=controller.limits.joint_name, receipt_sequence=round(now*10000),
        feedback_timestamp_sec=now, overlay_after_profile_nm=overlay[2], gravity_nm=0.,
        requested_total_nm=overlay[2], conditioned_total_nm=overlay[2] if final is None else final[2],
        joint_commands={"gravity_nm":ZERO,"overlay_nm":overlay,"total_nm":overlay if final is None else final,
                        "driver_kd_nm_s_per_rad":kd},
        completed_at_sec=now, torque_rate_nm_s=(28.8,)*6,
    )


def _conditioned_trial():
    """A real final conditioner with no hardware, not an ideal overlay ACK."""
    base = _v2_controller()
    controller = B601StagedTorqueConsoleController(replace(
        base.settings,
        limits=replace(base.limits, max_abs_assist_nm=1.),
        staged_assist=replace(
            base.settings.staged_assist, max_total_overlay_nm=1.,
            breakaway_torque_nm=1., breakaway_rise_slew_nm_s=1.,
            breakaway_max_duration_sec=1.2, breakaway_max_impulse_nm_sec=.6,
            motion_onset_confirm_displacement_rad=.0007, motion_onset_confirm_samples=2,
        ),
    ))
    controller.owner_ready()
    gravity = (0., 0., 6.9, 0., 0., 0.)
    previous = gravity
    receipt = 0

    def tick(now, qpos=ZERO, qvel=ZERO, passive=gravity, accept=True, heartbeat=True):
        nonlocal previous, receipt
        if heartbeat and controller.status()["operator_armed"]:
            controller.submit("heartbeat", now_sec=now)
        overlay, audit = _step(controller, now=now, qpos=qpos, qvel=qvel)
        if callable(passive):
            passive = passive(audit)
        requested = tuple(g+u for g, u in zip(passive, overlay))
        previous, _, _ = probe.condition_torque(
            requested, previous, absolute_limits_nm=(14.4,)*6,
            max_rate_nm_s=(28.8,)*6, dt_sec=.01,
        )
        actual_kd = probe._motion_console_driver_kd(
            controller, ZERO, final_command_delta_nm=tuple(a-b for a,b in zip(previous,passive)),
        )
        receipt += 1
        kwargs = {
            "joint_name":"joint3", "receipt_sequence":receipt, "feedback_timestamp_sec":now,
            "overlay_after_profile_nm":overlay[2], "gravity_nm":passive[2],
            "requested_total_nm":requested[2], "conditioned_total_nm":previous[2],
            "joint_commands":{"gravity_nm":passive, "overlay_nm":overlay, "total_nm":previous,
                              "driver_kd_nm_s_per_rad":actual_kd},
            "completed_at_sec":now, "torque_rate_nm_s":(28.8,)*6,
        }
        if accept and audit["phase"] != "fault":
            controller.record_dispatch(**kwargs)
        return overlay, audit, tuple(a-b for a,b in zip(previous,passive)), kwargs

    for i in range(11):
        tick(1.+i*.01)
    controller.submit("arm", now_sec=1.105)
    tick(1.11)
    controller.submit("set_target", now_sec=1.115, target_offset_rad=.04)
    tick(1.12)
    controller.submit("authorize_breakaway", now_sec=1.121)
    for i in range(1, 130):
        now = 1.12+i*.01
        _, audit, delta, _ = tick(now)
        if abs(delta[2]-1.) < 1e-9:
            assert audit["stage_phase"] == "breakaway"
            return controller, tick, now
    pytest.fail("declared no-motion pulse never reached the 1 Nm boundary")


@pytest.mark.parametrize("cause", ["zero", "withdraw", "onset", "gui_timeout"])
def test_final_slew_tail_retains_damping_guards_and_requires_stop_then_base_ack(cause):
    controller, tick, now = _conditioned_trial()
    if cause in {"zero", "withdraw"}:
        controller.submit(cause, now_sec=now+.001)
    elif cause == "gui_timeout":
        # Emulate producer death without changing control cadence or feedback.
        controller._last_intent_sec = now-controller.limits.command_timeout_sec-.01
    tail = []
    for i in range(1, 4):
        q = (0., 0., .000384, 0., 0., 0.) if cause == "onset" else ZERO
        v = (0., 0., .0384, 0., 0., 0.) if cause == "onset" and i == 1 else ZERO
        overlay, audit, delta, _ = tick(now+i*.01, q, v, heartbeat=cause != "gui_timeout")
        tail.append(delta[2])
        assert audit["stage_phase"] == "releasing"
        assert controller.status()["phase"] != "inactive"
        assert any(controller.active_driver_damping(ZERO))
        assert audit["drive_driver_kd_nm_s_per_rad"][2] == pytest.approx(.4)
        if i == 1:
            assert overlay == ZERO
    assert tail == pytest.approx([.712, .424, .136])
    assert audit["conditioned_command_impulse_nm_sec"] >= .01*(1.+.712+.424)
    # A zero packet alone is insufficient: all six raw/window velocities and
    # existing continuous stop dwell, then a distinct base-damping ACK, apply.
    stopped = False
    for i in range(4, 60):
        q = (0., 0., .000384, 0., 0., 0.) if cause == "onset" else ZERO
        _, audit, delta, _ = tick(now+i*.01, q)
        if controller.status()["phase"] == "inactive":
            assert audit["release_restoring_base_damping"]
            assert delta == pytest.approx(ZERO)
            assert audit["drive_driver_kd_nm_s_per_rad"] == ZERO
            stopped = True
            break
    assert stopped


@pytest.mark.parametrize("fault", ["support_drift", "support_speed", "selected_speed"])
def test_tail_release_keeps_original_drive_guards(fault):
    controller, tick, now = _conditioned_trial()
    controller.submit("withdraw", now_sec=now+.001)
    _, audit, delta, _ = tick(now+.01)
    assert delta[2] == pytest.approx(.712)
    q, v = list(ZERO), list(ZERO)
    if fault == "support_drift":
        q[3] = controller.limits.max_support_drift_rad+.0001
        v[3] = .0384
    else:
        v[3 if fault == "support_speed" else 2] = .050001
    _, audit, _, _ = tick(now+.02, tuple(q), tuple(v))
    assert audit["phase"] == "fault"
    assert "exceeded" in audit["fault"]
    assert not controller.status()["operator_armed"]


@pytest.mark.parametrize("invalid", ["partial", "wrong_joint", "stale_feedback", "early_clock", "late_clock", "duplicate_sequence"])
def test_bad_final_receipt_cannot_commit_withdrawal(invalid):
    controller, tick, now = _conditioned_trial()
    controller.submit("withdraw", now_sec=now+.001)
    _, _, _, kwargs = tick(now+.01, accept=False)
    valid = dict(kwargs)
    if invalid == "partial":
        kwargs["joint_commands"] = {**kwargs["joint_commands"], "total_nm":(0.,)*5}
    elif invalid == "wrong_joint":
        kwargs["joint_name"] = "joint2"
    elif invalid == "stale_feedback":
        kwargs["feedback_timestamp_sec"] = now
    elif invalid == "early_clock":
        kwargs["completed_at_sec"] = now
    elif invalid == "late_clock":
        kwargs["completed_at_sec"] = now+.06
    else:
        kwargs["receipt_sequence"] -= 1
    last = controller.status()["last_dispatch"]
    with pytest.raises(ValueError):
        controller.record_dispatch(**kwargs)
    assert controller.status()["last_dispatch"] == last
    assert controller.status()["phase"] == "releasing"
    assert any(controller.active_driver_damping(ZERO))
    controller.record_dispatch(**valid)
    with pytest.raises(ValueError):
        controller.record_dispatch(**valid)


def test_missing_final_receipt_fails_closed_instead_of_inventing_zero():
    controller, tick, now = _conditioned_trial()
    controller.submit("withdraw", now_sec=now+.001)
    tick(now+.01, accept=False)
    with pytest.raises(ValueError, match="lacks a complete"):
        tick(now+.02)
    assert controller.status()["phase"] == "releasing"


def test_base_damping_restore_is_gated_by_this_packets_final_conditioner():
    controller, tick, now = _conditioned_trial()
    controller.submit("withdraw", now_sec=now+.001)
    for i in range(1, 50):
        _, audit, delta, receipt = tick(
            now+i*.01,
            passive=lambda frame: (0.,0.,6.1 if frame["release_restoring_base_damping"] else 6.9,0.,0.,0.),
        )
        if audit["release_restoring_base_damping"]:
            break
    # A previous zero ACK permits requesting base Kd. Gravity changes before
    # this packet is conditioned; the authoritative owner must KEEP damping.
    assert audit["release_restoring_base_damping"]
    assert delta[2] == pytest.approx(.512)
    assert receipt["joint_commands"]["driver_kd_nm_s_per_rad"][2] == pytest.approx(.4)
    assert controller.status()["phase"] == "releasing"
    for j in range(1, 50):
        _, audit, delta, receipt = tick(now+(i+j)*.01, passive=(0.,0.,6.1,0.,0.,0.))
        if controller.status()["phase"] == "inactive":
            break
    assert controller.status()["phase"] == "inactive"
    assert delta == pytest.approx(ZERO)
    assert receipt["joint_commands"]["driver_kd_nm_s_per_rad"] == ZERO


def test_changing_gravity_tail_overrun_is_reported_not_silently_refunded():
    controller, tick, now = _conditioned_trial()
    controller.submit("withdraw", now_sec=now+.001)
    # Force a model-base change faster than the final slew; the old constant-
    # passive reserve cannot certify this and must not disguise it as zero.
    for i in range(1, 90):
        _, audit, _, _ = tick(now+i*.01, passive=(0.,0.,-14.,0.,0.,0.))
        if audit["fault"]:
            break
    assert audit["fault"] == "conditioned_command_impulse_budget_exceeded"
    assert audit["conditioned_command_impulse_nm_sec"] > .6
    assert controller.status()["phase"] == "fault"


def test_v2_prepare_and_authorize_accept_one_count_only_after_full_stationary_window():
    controller = _v2_controller()
    controller.owner_ready()
    controller.submit("arm",now_sec=1.)
    overlay,audit = _step(controller,now=1.)
    assert overlay == ZERO
    assert audit["fault"] == "arm_rejected_velocity_window_not_ready"
    assert not audit["velocity_window"]["ready"]
    for i in range(1,11):
        _step(controller,now=1.+i*.01)
    controller.submit("arm",now_sec=1.105)
    q, raw = (0.,0.,.000384,0.,0.,0.), (0.,0.,.0384,0.,0.,0.)
    _,armed = _step(controller,now=1.11,qpos=q,qvel=raw)
    assert armed["phase"] == "armed"
    controller.submit("set_target",now_sec=1.115,target_offset_rad=.01)
    _,cue = _step(controller,now=1.12,qpos=ZERO,qvel=(0.,0.,-.0384,0.,0.,0.))
    assert cue["breakaway_eligible"]
    assert cue["velocity_window"]["soft_decision_source"] == "causal_window_velocity_and_displacement"
    controller.submit("authorize_breakaway",now_sec=1.121)
    for i in range(1,9):
        _,audit = _step(controller,now=1.12+i*.01,qpos=q if i%2 else ZERO,
                       qvel=(0.,0.,.0384 if i%2 else -.0384,0.,0.,0.))
    assert audit["phase"] == "breakaway"
    assert audit["effective_assist_nm"] > 0


def test_v2_prepare_rejects_sustained_motion_and_raw_hard_speed():
    controller = _v2_controller()
    controller.owner_ready()
    for i in range(10):
        _step(controller,now=1.+i*.01,qpos=(0.,0.,i*.0003,0.,0.,0.),qvel=(0.,0.,.03,0.,0.,0.))
    controller.submit("arm",now_sec=1.095)
    _,rejected = _step(controller,now=1.1,qpos=(0.,0.,.003,0.,0.,0.),qvel=(0.,0.,.03,0.,0.,0.))
    assert rejected["fault"] == "arm_rejected_not_stationary"
    controller = _v2_controller()
    controller.owner_ready()
    for i in range(10):
        _step(controller,now=1.+i*.01)
    controller.submit("arm",now_sec=1.095)
    _,rejected = _step(controller,now=1.1,qvel=(0.,0.,.051,0.,0.,0.))
    assert rejected["fault"] == "arm_rejected_raw_velocity_hard_guard"


def test_v2_control_gap_requires_fresh_window_and_never_resumes_exhausted_target():
    controller = _v2_controller()
    _arm_and_set_target(controller,target_offset_rad=.01)
    _step(controller,now=1.02)
    overlay,gap = _step(controller,now=1.08)
    assert gap["phase"] == "authority_exhausted"
    assert gap["reason"] == "control_gap_assist_withdrawn"
    assert not gap["velocity_window"]["ready"]
    assert overlay == ZERO
    for i in range(1,10):
        overlay,audit = _step(controller,now=1.08+i*.01)
        assert audit["phase"] == "authority_exhausted"
        assert overlay == ZERO
        assert controller.active_driver_damping(ZERO) == ZERO
    assert audit["velocity_window"]["ready"]
    assert not audit["breakaway_eligible"]


def test_native_gain_display_is_confirmed_dispatch_not_phase_request():
    controller = _controller(motion_control=B601JointMotionConfig(drive_driver_kd_nm_s_per_rad=(.4,) * 6))
    _arm_and_set_target(controller)
    _step(controller, now=1.02)
    assert controller.status()["last_dispatch"] is None
    commands = {"gravity_nm": ZERO, "overlay_nm": ZERO, "total_nm": ZERO,
                "driver_kd_nm_s_per_rad": (.025,) * 6}
    controller.record_dispatch(joint_name="joint3", receipt_sequence=1, feedback_timestamp_sec=1.02,
                               overlay_after_profile_nm=0, gravity_nm=0, requested_total_nm=0,
                               conditioned_total_nm=0, joint_commands=commands)
    status = controller.status()
    assert status["last_dispatch"]["driver_kd_nm_s_per_rad"] == (.025,) * 6
    assert status["drive_driver_kd_nm_s_per_rad"] == pytest.approx((.04,) * 6)
    assert "driver_kd_nm_s_per_rad" in commands


@pytest.mark.parametrize("action", ["withdraw", "owner_stopped"])
def test_gain_request_clears_without_a_further_step_but_dispatch_stays_historical(action):
    controller = _controller(motion_control=B601JointMotionConfig(drive_driver_kd_nm_s_per_rad=(.4,) * 6))
    _arm_and_set_target(controller)
    _step(controller, now=1.02)
    controller.record_dispatch(joint_name="joint3", receipt_sequence=1, feedback_timestamp_sec=1.02,
                               overlay_after_profile_nm=0, gravity_nm=0, requested_total_nm=0,
                               conditioned_total_nm=0, joint_commands={
                                   "gravity_nm": ZERO, "overlay_nm": ZERO, "total_nm": ZERO,
                                   "driver_kd_nm_s_per_rad": (.04,) * 6,
                               })
    if action == "withdraw":
        controller.submit("withdraw", now_sec=1.025)
    else:
        controller.owner_stopped(now_sec=1.025, reason="test_protective_hold")
    status = controller.status()
    assert status["drive_driver_kd_nm_s_per_rad"] == ZERO
    assert probe._motion_console_driver_kd(controller, ZERO) == ZERO
    assert status["last_dispatch"]["driver_kd_nm_s_per_rad"] == (.04,) * 6


@pytest.mark.parametrize("capability_present", [True, False])
@pytest.mark.parametrize("omit_driver_kd", [False, True])
def test_preflight_cannot_construct_backend_or_open_console(tmp_path, monkeypatch, capability_present, omit_driver_kd):
    args = owner._parser().parse_args(["--allow-motion", "--preflight-only"])
    command = owner.build_probe_command(args, run_dir=tmp_path)[2:]
    if omit_driver_kd:
        index = command.index("--driver-kd")
        del command[index:index+7]
    parsed = probe.build_parser().parse_args(command)
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight attempted hardware or console construction")
    monkeypatch.setattr(probe, "B601RSBackend", forbidden)
    monkeypatch.setattr("control_station.b601_torque_console.B601TorqueConsoleServer", forbidden)
    monkeypatch.setattr(probe, "_resolve_model_and_limits", lambda source: (SimpleNamespace(urdf_path=tmp_path / "model.urdf"), ((-1, 1),) * 6))
    monkeypatch.setattr(probe, "_resolve_effort_limits", lambda path, **kwargs: (1,) * 6)
    def load_config(*args, **kwargs):
        assert kwargs["allow_motion"] is False
        return SimpleNamespace(channel="can0", joints=[SimpleNamespace(name=name) for name in B601_RS_ARM_JOINT_NAMES])
    monkeypatch.setattr(probe.B601RSBackendConfig, "from_rs_yaml", load_config)
    features = ["parameter_reply_correlation_v1", "state_observation_metadata_v1"] if capability_present else []
    monkeypatch.setitem(sys.modules, "motorbridge", SimpleNamespace(abi_capabilities=lambda: {"features": {"robstride": features}}))
    assert probe.main(command) == (0 if capability_present else 2)
    result = json.loads(parsed.output.read_text())
    assert result["can_opened"] is False
    assert result["motor_command_sent"] is False
    assert result["status"] == ("PASS" if capability_present else "ERROR")
    if capability_present:
        assert result["protective_hold_driver_kd_nm_s_per_rad"] == [0.0] * 6


def test_motion_control_stop_hold_withdraw_and_reanchor_are_owner_tick_actions():
    controller = _controller(motion_control=B601JointMotionConfig(), command_timeout_sec=2)
    _arm_and_set_target(controller)
    _step(controller, now=1.02)
    controller.submit("stop_and_hold", now_sec=1.025)
    q = (0, 0, .004, 0, 0, 0)
    _, braking = _step(controller, now=1.04, qpos=q)
    assert braking["phase"] == "brake"
    assert braking["motion_control"]["position_torque_nm"] == 0
    assert braking["motion_control"]["target_position_rad"] != .004
    assert braking["motion_control"]["target_error_rad"] != 0
    for t in (1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22):
        _, stopped = _step(controller, now=t, qpos=q)
    assert stopped["phase"] == "soft_hold"
    assert stopped["motion_control"]["hold_position_rad"] == .004
    controller.submit("set_target", now_sec=1.225, target_offset_rad=.02)
    _step(controller, now=1.24, qpos=q)
    _, new_cue = _step(controller, now=1.26, qpos=q)
    assert new_cue["phase"] == "cue"
    assert controller.status()["anchor_rad"] == .004
    controller.submit("withdraw", now_sec=1.27)
    overlay, withdrawn = _step(controller, now=1.28, qpos=q)
    assert overlay == ZERO
    assert withdrawn["motion_control"] is None
    assert controller.status()["phase"] == "inactive"


def test_motion_control_does_not_filter_raw_hard_velocity_guard():
    controller = _controller(motion_control=B601JointMotionConfig())
    _arm_and_set_target(controller)
    _step(controller, now=1.02)
    overlay, guard = _step(controller, now=1.04, qvel=(0, 0, .51, 0, 0, 0))
    assert overlay == ZERO
    assert controller.status()["phase"] == "fault"
    assert "joint_velocity_exceeded" in guard["fault"]
    assert controller.status()["control_qvel_rad_s"][2] == 0


def _arm_and_set_target(
    controller: B601StagedTorqueConsoleController,
    *,
    target_offset_rad: float = 0.08,
) -> tuple[float, ...]:
    controller.owner_ready()
    if controller.settings.staged_assist.motion_control is not None and controller.settings.staged_assist.motion_control.windowed_soft_velocity:
        # v2 preparation now requires an actually populated feedback window.
        for i in range(10):
            controller.step(now_sec=.90+i*.01, qpos_rad=ZERO, qvel_rad_s=ZERO)
    controller.submit("arm", now_sec=1.0)
    _, armed = controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
    assert armed["phase"] == "armed"
    controller.submit("set_target", now_sec=1.01, target_offset_rad=target_offset_rad)
    return ZERO


def _step(
    controller: B601StagedTorqueConsoleController,
    *,
    now: float,
    qpos: tuple[float, ...] = ZERO,
    qvel: tuple[float, ...] = ZERO,
) -> tuple[tuple[float, ...], dict[str, object]]:
    return controller.step(now_sec=now, qpos_rad=qpos, qvel_rad_s=qvel)


def test_staged_controller_captures_anchor_and_only_assists_selected_joint() -> None:
    controller = _controller()
    _arm_and_set_target(controller)

    cue_overlay, cue = _step(controller, now=1.02)
    assert cue["phase"] == "cue"
    assert cue["anchor_rad"] == 0.0
    assert cue["target_offset_rad"] == pytest.approx(0.08)
    assert cue_overlay[2] > 0.0
    assert sum(abs(value) for index, value in enumerate(cue_overlay) if index != 2) == 0.0
    assert cue["overlay_torque_nm"] == cue_overlay


def test_staged_breakaway_requires_one_shot_gui_authorization_and_drops_on_motion() -> None:
    controller = _controller()
    _arm_and_set_target(controller)

    with pytest.raises(ValueError, match="owner target admission"):
        controller.submit("authorize_breakaway", now_sec=1.015)
    _, cue = _step(controller, now=1.02)
    assert cue["phase"] == "cue"
    controller.submit("authorize_breakaway", now_sec=1.025)
    assert controller.status()["breakaway_authorized"] is True

    _, cue = _step(controller, now=1.04)
    assert cue["phase"] == "cue"
    assert controller.status()["breakaway_authorized"] is True
    _, armed_pulse = _step(controller, now=1.08)
    assert armed_pulse["phase"] == "breakaway"
    assert armed_pulse["effective_assist_nm"] == 0.0
    assert controller.status()["breakaway_authorized"] is False

    pulse, pulse_audit = _step(controller, now=1.10)
    assert pulse_audit["phase"] == "breakaway"
    assert pulse[2] > 0.0
    assert pulse_audit["breakaway_impulse_nm_sec"] > 0.0
    onset_one, first_onset = _step(
        controller,
        now=1.12,
        qpos=(0.0, 0.0, 0.0012, 0.0, 0.0, 0.0),
    )
    assert first_onset["phase"] == "breakaway"
    assert onset_one == ZERO
    onset_two, confirmed = _step(
        controller,
        now=1.14,
        qpos=(0.0, 0.0, 0.0013, 0.0, 0.0, 0.0),
    )
    assert confirmed["phase"] == "drop"
    assert onset_two == ZERO
    tracked, track_audit = _step(
        controller,
        now=1.16,
        qpos=(0.0, 0.0, 0.0013, 0.0, 0.0, 0.0),
    )
    assert track_audit["phase"] == "track"
    assert 0.0 < tracked[2] <= 0.10


def test_cue_waits_for_late_gui_authorization_and_accepts_small_target_after_encoder_motion() -> None:
    controller = _controller()
    _arm_and_set_target(controller, target_offset_rad=0.01)

    cue, first = _step(controller, now=1.02)
    assert first["phase"] == "cue"
    assert first["breakaway_eligible"] is True
    assert cue[2] > 0.0

    quiet, waiting = _step(
        controller,
        now=1.08,
        qpos=(0.0, 0.0, 0.00038, 0.0, 0.0, 0.0),
        qvel=(0.0, 0.0, 0.0077, 0.0, 0.0, 0.0),
    )
    assert waiting["phase"] == "cue"
    assert waiting["reason"] == "awaiting_breakaway_authorization"
    assert quiet == ZERO
    assert waiting["breakaway_eligible"] is True
    assert waiting["breakaway_gate_reason"] == "ready"

    controller.submit("authorize_breakaway", now_sec=1.081)
    assert controller.status()["breakaway_authorized"] is True
    starting_overlay, starting = _step(
        controller,
        now=1.10,
        qpos=(0.0, 0.0, 0.00038, 0.0, 0.0, 0.0),
        qvel=(0.0, 0.0, 0.0077, 0.0, 0.0, 0.0),
    )
    assert starting["phase"] == "breakaway"
    assert starting_overlay == ZERO


def test_breakaway_gate_reports_target_error_and_motion_reasons() -> None:
    controller = _controller()
    _arm_and_set_target(controller, target_offset_rad=0.01)
    _, cue = _step(controller, now=1.02)
    assert cue["breakaway_eligible"] is True

    _, moving = _step(
        controller,
        now=1.04,
        qvel=(0.0, 0.0, 0.03, 0.0, 0.0, 0.0),
    )
    assert moving["breakaway_eligible"] is False
    assert moving["breakaway_gate_reason"] == "joint_not_stationary_for_breakaway"


def test_new_gui_target_clears_breakaway_grant_before_new_cue() -> None:
    controller = _controller()
    _arm_and_set_target(controller)
    _step(controller, now=1.02)
    _step(controller, now=1.04)
    controller.submit("authorize_breakaway", now_sec=1.05)
    assert controller.status()["breakaway_authorized"] is True

    controller.submit("set_target", now_sec=1.055, target_offset_rad=0.079)
    assert controller.status()["target_update_pending"] is True
    with pytest.raises(ValueError, match="owner target admission"):
        controller.submit("authorize_breakaway", now_sec=1.056)
    overlay, reset = _step(controller, now=1.06)
    assert reset["phase"] == "passive"
    assert overlay == ZERO
    assert controller.status()["breakaway_authorized"] is False

    _, new_cue = _step(controller, now=1.08)
    assert new_cue["phase"] == "cue"
    assert controller.status()["target_update_pending"] is False
    # The old one-shot authorization cannot skip the new cue interval.
    _, still_cue = _step(controller, now=1.12)
    assert still_cue["phase"] == "cue"


def test_same_target_does_not_reset_exhausted_breakaway_budget() -> None:
    controller = _controller(breakaway_impulse_nm_sec=0.00002)
    _arm_and_set_target(controller)
    _step(controller, now=1.02)
    controller.submit("authorize_breakaway", now_sec=1.025)
    _step(controller, now=1.04)
    _step(controller, now=1.08)
    _step(controller, now=1.10)
    _, exhausted = _step(controller, now=1.12)
    assert exhausted["phase"] == "authority_exhausted"

    controller.submit("set_target", now_sec=1.125, target_offset_rad=0.08)
    _, same_intent = _step(controller, now=1.14)
    assert same_intent["phase"] == "authority_exhausted"
    assert same_intent["effective_assist_nm"] == 0.0

    controller.submit("set_target", now_sec=1.145, target_offset_rad=0.095)
    _, reset = _step(controller, now=1.16)
    assert reset["phase"] == "passive"
    _, new_cue = _step(controller, now=1.18)
    assert new_cue["phase"] == "cue"


def test_staged_target_guard_withdraws_overlay_without_widening_limits() -> None:
    controller = _controller(max_support_drift_rad=0.005)
    _arm_and_set_target(controller)
    drift = (0.006, 0.0, 0.0, 0.0, 0.0, 0.0)
    overlay, audit = _step(controller, now=1.02, qpos=drift)
    assert audit["phase"] == "fault"
    assert "support_joint_drift_exceeded: joint1" in str(audit["fault"])
    assert overlay == ZERO
    assert controller.status()["limits"]["max_joint_velocity_rad_s"] == 0.50
    assert controller.status()["limits"]["max_support_drift_rad"] == 0.005


def test_staged_heartbeat_timeout_withdraws_without_old_target() -> None:
    controller = _controller(command_timeout_sec=0.10)
    _arm_and_set_target(controller)
    _step(controller, now=1.02)
    _step(controller, now=1.04)
    overlay, audit = _step(controller, now=1.20)
    assert overlay == ZERO
    assert audit["phase"] == "inactive"
    assert audit["target_offset_rad"] is None
    assert audit["fault"] == "gui_intent_stale"
    assert controller.status()["target_offset_rad"] is None


def test_staged_joint_selection_and_owner_stop_are_fail_closed() -> None:
    controller = _controller()
    controller.owner_ready()
    controller.submit("select_joint", now_sec=1.0, joint_name="joint6")
    assert controller.status()["joint_name"] == "joint6"
    assert controller.limits.joint_name in B601_RS_ARM_JOINT_NAMES

    controller.submit("arm", now_sec=1.01)
    _step(controller, now=1.01)
    controller.owner_stopped(now_sec=1.02, reason="operator_interrupt")
    status = controller.status()
    assert status["phase"] == "fault"
    assert status["owner_ready"] is False
    assert status["effective_assist_nm"] == 0.0
    with pytest.raises(ValueError, match="active gravity owner"):
        controller.submit("arm", now_sec=1.03)


def test_staged_console_http_accepts_target_and_rejects_extra_fields() -> None:
    controller = _controller()
    server = B601TorqueConsoleServer(controller)
    server.start()
    try:
        with urlopen(server.url + "/", timeout=2) as response:
            page = response.read().decode()
        token = page.split('name="csrf-token" content="', 1)[1].split('"', 1)[0]
        controller.owner_ready()

        def post(payload: dict[str, object]):
            request = Request(
                server.url + "/api/intent",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Origin": server.url,
                    "X-B601-Console-Token": token,
                },
            )
            return urlopen(request, timeout=2)

        with post({"action": "arm"}) as response:
            assert json.load(response)["status"] == "accepted"
        _step(controller, now=1.0)
        with post({"action": "set_target", "target_offset_rad": 0.025}) as response:
            assert json.load(response)["status"] == "accepted"
        with pytest.raises(HTTPError) as invalid:
            post(
                {
                    "action": "set_target",
                    "target_offset_rad": 0.025,
                    "torque_nm": 0.1,
                }
            )
        assert invalid.value.code == 400

        with urlopen(server.url + "/api/status", timeout=2) as response:
            status = json.load(response)
        assert status["controller_mode"] == "staged_assist"
        assert status["operator_armed"] is True
        assert status["target_offset_rad"] == pytest.approx(0.025)
    finally:
        server.close()


def test_owner_launcher_forwards_staged_options_to_offline_probe(tmp_path) -> None:
    owner_defaults = owner._parser().parse_args([])
    assert owner_defaults.torque_console_staged_breakaway_rise_slew_nm_s == 2.0
    assert owner_defaults.torque_console_staged_motion_onset_rad == 0.0007
    assert owner_defaults.torque_console_staged_motion_onset_velocity_rad_s == 0.025
    args = owner._parser().parse_args(
        [
            "--allow-motion",
            "--debug-dynamics-trace",
            "--torque-console",
            "--torque-console-staged-assist",
            "--torque-console-joint",
            "joint4",
            "--torque-console-max-assist-nm",
            "0.30",
            "--torque-console-staged-target-offset-limit-rad",
            "0.08",
            "--torque-console-staged-breakaway-torque-nm",
            "0.18",
            "--torque-console-staged-authority-rise-per-sec",
            "3.0",
            "--torque-console-rise-slew-nm-s",
            "0.15",
            "--torque-console-staged-breakaway-rise-slew-nm-s",
            "2.0",
            "--torque-console-confirmation",
            "B601_RS_TORQUE_CONSOLE",
        ]
    )
    command = owner.build_probe_command(args, run_dir=tmp_path)
    parsed = probe.build_parser().parse_args(command[2:])
    assert parsed.torque_console is True
    assert parsed.torque_console_staged_assist is True
    assert parsed.torque_console_joint == "joint4"
    assert parsed.torque_console_staged_target_offset_limit_rad == 0.08
    assert parsed.torque_console_staged_breakaway_torque_nm == 0.18
    assert parsed.torque_console_staged_authority_rise_per_sec == 3.0
    assert parsed.torque_console_rise_slew_nm_s == 0.15
    assert parsed.torque_console_staged_breakaway_rise_slew_nm_s == 2.0
    assert parsed.debug_dynamics_trace is True
    probe._validate_torque_console_cli(parsed)
    settings = probe._staged_torque_console_settings_from_args(parsed)
    assert settings.limits.joint_name == "joint4"
    assert settings.staged_assist.breakaway_torque_nm == 0.18
    assert settings.staged_assist.breakaway_rise_slew_nm_s == 2.0
    assert settings.authority_rise_per_sec == 3.0
    assert settings.staged_assist.minimum_breakaway_error_rad == pytest.approx(0.002)
    assert settings.staged_assist.motion_onset_confirm_samples == 1
    assert settings.maximum_target_offset_rad == 0.08


@pytest.mark.parametrize(
    ("displacement", "velocity", "message"),
    [
        (0.003, 0.025, "motion-onset displacement"),
        (0.0007, 0.04, "motion-onset velocity"),
    ],
)
def test_staged_settings_reject_onset_threshold_without_guard_margin(
    displacement: float,
    velocity: float,
    message: str,
) -> None:
    base = _controller()
    unsafe_staged = replace(
        base.settings.staged_assist,
        motion_onset_displacement_rad=displacement,
        motion_onset_velocity_rad_s=velocity,
    )
    unsafe_limits = replace(
        base.settings.limits,
        max_joint_velocity_rad_s=0.05,
    )
    with pytest.raises(ValueError, match=message):
        replace(
            base.settings,
            limits=unsafe_limits,
            staged_assist=unsafe_staged,
        )


def test_staged_status_reports_requested_and_owner_accepted_breakaway_peaks() -> None:
    controller = _controller(
        breakaway_torque_nm=0.20,
        breakaway_rise_slew_nm_s=2.0,
        authority_rise_per_sec=100.0,
    )
    _arm_and_set_target(controller)
    _step(controller, now=1.02)
    controller.submit("authorize_breakaway", now_sec=1.025)
    _step(controller, now=1.08)
    overlay, pulse = _step(controller, now=1.10)
    assert overlay[2] == pytest.approx(0.04)
    assert pulse["effective_assist_nm"] == overlay[2]

    controller.record_dispatch(
        joint_name="joint3",
        receipt_sequence=10,
        feedback_timestamp_sec=1.10,
        overlay_after_profile_nm=0.035,
        gravity_nm=1.0,
        requested_total_nm=1.035,
        conditioned_total_nm=1.033,
    )
    status = controller.status()
    assert status["breakaway_peak_overlay_nm"] == pytest.approx(0.04)
    assert status["breakaway_peak_dispatched_nm"] == pytest.approx(0.035)
    assert status["breakaway_peak_conditioned_delta_nm"] == pytest.approx(0.033)
    assert status["limits"]["breakaway_rise_slew_nm_s"] == pytest.approx(2.0)
    assert status["policy_authority"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "period_sec, expected_impulse_nm_sec",
    [(0.05, 0.025), (0.02, 0.028)],
)
def test_default_staged_authority_and_breakaway_ramps_reach_cap_on_owner_schedule(
    period_sec: float,
    expected_impulse_nm_sec: float,
) -> None:
    controller = _controller(
        breakaway_torque_nm=0.20,
        breakaway_rise_slew_nm_s=2.0,
        authority_rise_per_sec=3.0,
        cue_duration_sec=0.30,
        breakaway_impulse_nm_sec=0.04,
        nominal_control_period_sec=period_sec,
    )
    _arm_and_set_target(controller, target_offset_rad=0.04)
    _, cue = _step(controller, now=1.02)
    assert cue["phase"] == "cue"
    controller.submit("authorize_breakaway", now_sec=1.025)

    for index in range(1, round(0.30 / period_sec) + 1):
        now = 1.02 + index * period_sec
        controller.submit("heartbeat", now_sec=now)
        _, frame = _step(controller, now=now)
    assert frame["phase"] == "breakaway"
    assert frame["policy_authority"] == pytest.approx(0.96)

    peak_overlay = 0.0
    for index in range(1, round(0.20 / period_sec) + 1):
        now = 1.32 + index * period_sec
        controller.submit("heartbeat", now_sec=now)
        overlay, frame = _step(controller, now=now)
        peak_overlay = max(peak_overlay, abs(overlay[2]))
        if frame["phase"] != "breakaway":
            break
    assert peak_overlay == pytest.approx(0.20)
    assert frame["phase"] == "authority_exhausted"
    assert frame["reason"] == "breakaway_duration_exhausted_before_impulse_budget"
    assert frame["breakaway_impulse_nm_sec"] == pytest.approx(expected_impulse_nm_sec)


def test_higher_breakaway_peak_remains_bounded_by_one_shot_impulse() -> None:
    controller = _controller(
        max_abs_assist_nm=0.80,
        breakaway_torque_nm=0.80,
        breakaway_rise_slew_nm_s=10.0,
        authority_rise_per_sec=3.0,
        cue_duration_sec=0.30,
        breakaway_impulse_nm_sec=0.08,
        nominal_control_period_sec=0.02,
    )
    _arm_and_set_target(controller, target_offset_rad=0.04)
    _step(controller, now=1.02)
    controller.submit("authorize_breakaway", now_sec=1.025)

    for index in range(1, 16):
        now = 1.02 + index * 0.02
        controller.submit("heartbeat", now_sec=now)
        _, frame = _step(controller, now=now)
    assert frame["phase"] == "breakaway"

    peak_overlay = 0.0
    for index in range(1, 10):
        now = 1.32 + index * 0.02
        controller.submit("heartbeat", now_sec=now)
        overlay, frame = _step(controller, now=now)
        peak_overlay = max(peak_overlay, abs(overlay[2]))
        if frame["phase"] != "breakaway":
            break

    assert peak_overlay == pytest.approx(0.80)
    assert frame["phase"] == "authority_exhausted"
    assert frame["reason"] == "breakaway_impulse_budget_exhausted"
    assert frame["breakaway_impulse_nm_sec"] == pytest.approx(0.08)


def test_staged_mode_requires_explicit_torque_console_opt_in() -> None:
    args = owner._parser().parse_args(
        ["--allow-motion", "--torque-console-staged-assist"]
    )
    assert args.torque_console_staged_assist is True
    with pytest.raises(SystemExit) as error:
        owner.main(["--allow-motion", "--torque-console-staged-assist"])
    assert error.value.code == 2
