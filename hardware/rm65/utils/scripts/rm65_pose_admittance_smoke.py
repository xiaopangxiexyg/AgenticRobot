from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.force_compensation import (  # noqa: E402
    PoseAdmittance,
    WrenchProcessor,
    angle_delta,
    apply_pose_offset,
    finite_list,
    mean_wrench,
    norm,
    normalize_pose,
    parse_axes,
    parse_signs,
)
from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65  # noqa: E402


CONFIRM_TEXT = "TEST_RM65_POSE_ADMITTANCE"


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _connect_robot(side: str, ip: str) -> Arm:
    robot = Arm(RM65, ip)
    if int(robot.Arm_Socket_State()) != 0:
        robot.Arm_Socket_Close()
        raise RuntimeError(f"{side}: failed to connect ip={ip}")
    print(f"{side}: connected ip={ip} api={robot.API_Version()}")
    return robot


def _read_state(robot: Arm, side: str) -> dict[str, list[float]]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")
    return {
        "joint_deg": finite_list(joint_deg, expected_len=6, label="joint_deg"),
        "pose": finite_list(pose, expected_len=6, label="pose"),
    }


def _read_force(robot: Arm, side: str) -> dict[str, list[float]]:
    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    return {
        "raw": finite_list(raw, expected_len=6, label="raw"),
        "zero": finite_list(zero, expected_len=6, label="zero"),
        "work": finite_list(work, expected_len=6, label="work"),
        "tool": finite_list(tool, expected_len=6, label="tool"),
    }


def _print_force(prefix: str, readings: dict[str, list[float]]) -> None:
    for source in ("raw", "zero", "work", "tool"):
        wrench = readings[source]
        print(
            f"{prefix} {source}={_format(wrench)} "
            f"|F|={norm(wrench[:3]):.4f}N |T|={norm(wrench[3:]):.4f}Nm"
        )


def _current_tool_name(frame_result: object) -> str:
    if isinstance(frame_result, tuple) and len(frame_result) >= 2:
        frame = frame_result[1]
    else:
        frame = frame_result
    if isinstance(frame, dict):
        value = frame.get("name") or frame.get("frame_name") or frame.get("tool_name")
        if value is not None:
            return str(value).strip("\x00")
    return str(frame_result).strip("\x00")


def _check_expected_tool(robot: Arm, args) -> None:
    if not args.expected_tool:
        return
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    current_name = _current_tool_name((code, frame))
    print(f"current_tool={current_name}")
    if args.expected_tool not in current_name:
        raise RuntimeError(
            f"expected active tool containing '{args.expected_tool}', got '{current_name}'"
        )


def _joint_delta_deg(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _pose_delta(start_pose: list[float], current_pose: list[float]) -> tuple[float, float]:
    linear = norm(current_pose[i] - start_pose[i] for i in range(3))
    angular = norm(angle_delta(current_pose[i], start_pose[i]) for i in range(3, 6))
    return linear, angular


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _stats_ms(values_sec: list[float]) -> dict[str, float]:
    values_ms = [value * 1000.0 for value in values_sec]
    return {
        "mean_ms": statistics.fmean(values_ms) if values_ms else math.nan,
        "p50_ms": _percentile(values_ms, 0.50),
        "p95_ms": _percentile(values_ms, 0.95),
        "p99_ms": _percentile(values_ms, 0.99),
        "max_ms": max(values_ms) if values_ms else math.nan,
    }


def _remember_event(events: dict[str, dict[str, float]], name: str, step: int, time_sec: float, value: float) -> None:
    if name not in events:
        events[name] = {
            "step": float(step),
            "time_sec": float(time_sec),
            "value": float(value),
        }


def _latency_ms(events: dict[str, dict[str, float]], later: str, earlier: str) -> float | None:
    if later not in events or earlier not in events:
        return None
    return (events[later]["time_sec"] - events[earlier]["time_sec"]) * 1000.0


def _sample_tare(robot: Arm, side: str, args) -> list[float]:
    samples = max(1, int(round(args.tare_sec * args.control_hz)))
    interval = 1.0 / args.control_hz
    print(
        f"{side}: sampling tare source={args.force_source} "
        f"samples={samples} interval={interval:.4f}s"
    )
    wrenches = []
    for index in range(samples):
        readings = _read_force(robot, side)
        wrenches.append(readings[args.force_source])
        if index + 1 < samples:
            time.sleep(interval)
    tare = mean_wrench(wrenches)
    print(f"{side}: tare_{args.force_source}={_format(tare)} |F|={norm(tare[:3]):.4f}N |T|={norm(tare[3:]):.4f}Nm")
    return tare


def _check_start_force(readings: dict[str, list[float]], args) -> None:
    wrench = readings[args.force_source]
    force_norm = norm(wrench[:3])
    torque_norm = norm(wrench[3:])
    print(
        f"precheck force_source={args.force_source} "
        f"|F|={force_norm:.4f}N |T|={torque_norm:.4f}Nm"
    )
    if force_norm > args.max_start_force_n:
        raise RuntimeError(
            f"start force {force_norm:.4f}N > --max-start-force-n {args.max_start_force_n:.4f}N"
        )
    if torque_norm > args.max_start_torque_nm:
        raise RuntimeError(
            f"start torque {torque_norm:.4f}Nm > --max-start-torque-nm {args.max_start_torque_nm:.4f}Nm"
        )


def _check_runtime_limits(
    *,
    side: str,
    start_state: dict[str, list[float]],
    state: dict[str, list[float]],
    tare_corrected: list[float],
    offset: list[float],
    args,
) -> tuple[float, float, float, float, float]:
    joint_delta = _joint_delta_deg(start_state["joint_deg"], state["joint_deg"])
    pose_linear, pose_angular = _pose_delta(start_state["pose"], state["pose"])
    force_norm = norm(tare_corrected[:3])
    torque_norm = norm(tare_corrected[3:])
    offset_norm = norm(offset[:3])
    angular_offset_norm = norm(offset[3:])
    if force_norm > args.hard_stop_force_n:
        raise RuntimeError(f"{side}: force {force_norm:.4f}N > --hard-stop-force-n {args.hard_stop_force_n:.4f}N")
    if torque_norm > args.hard_stop_torque_nm:
        raise RuntimeError(f"{side}: torque {torque_norm:.4f}Nm > --hard-stop-torque-nm {args.hard_stop_torque_nm:.4f}Nm")
    if offset_norm > args.max_linear_offset_m * 1.05:
        raise RuntimeError(f"{side}: offset {offset_norm:.5f}m exceeded configured limit")
    if args.max_angular_offset_rad > 0.0 and angular_offset_norm > args.max_angular_offset_rad * 1.05:
        raise RuntimeError(f"{side}: angular offset {angular_offset_norm:.5f}rad exceeded configured limit")
    if pose_linear > args.max_observed_pose_linear_m:
        raise RuntimeError(
            f"{side}: observed pose linear delta {pose_linear:.5f}m > "
            f"--max-observed-pose-linear-m {args.max_observed_pose_linear_m:.5f}m"
        )
    if pose_angular > args.max_observed_pose_angular_rad:
        raise RuntimeError(
            f"{side}: observed pose angular delta {pose_angular:.5f}rad > "
            f"--max-observed-pose-angular-rad {args.max_observed_pose_angular_rad:.5f}rad"
        )
    if joint_delta > args.max_joint_delta_deg:
        raise RuntimeError(
            f"{side}: joint delta {joint_delta:.4f}deg > --max-joint-delta-deg {args.max_joint_delta_deg:.4f}deg"
        )
    return joint_delta, pose_linear, pose_angular, force_norm, torque_norm


def _uses_rotation(enabled_axes: list[bool]) -> bool:
    return any(enabled_axes[3:])


def _stimulus_norm(tare_corrected: list[float], enabled_axes: list[bool]) -> float:
    if _uses_rotation(enabled_axes) and not any(enabled_axes[:3]):
        return norm(tare_corrected[3:])
    return norm(tare_corrected[:3])


def _active_norm(active_wrench: list[float], enabled_axes: list[bool]) -> float:
    if _uses_rotation(enabled_axes) and not any(enabled_axes[:3]):
        return norm(active_wrench[3:])
    return norm(active_wrench[:3])


def _offset_norm(offset: list[float], enabled_axes: list[bool]) -> float:
    if _uses_rotation(enabled_axes) and not any(enabled_axes[:3]):
        return norm(offset[3:])
    return norm(offset[:3])


def _run_admittance(robot: Arm, side: str, start_state: dict[str, list[float]], tare: list[float], args) -> dict[str, object]:
    period = 1.0 / args.control_hz
    loops = max(1, int(round(args.duration_sec * args.control_hz)))
    processor = WrenchProcessor(
        tare=tare,
        lowpass_alpha=args.lowpass_alpha,
        force_deadband_n=args.force_deadband_n,
        torque_deadband_nm=args.torque_deadband_nm,
    )
    admittance = PoseAdmittance(
        enabled_axes=args.enabled_axes,
        force_sign=args.force_sign,
        angular_force_sign=args.angular_force_sign,
        angular_axis_signs=args.angular_axis_signs,
        linear_gain_m_s_per_n=args.linear_gain_m_s_per_n,
        angular_gain_rad_s_per_nm=args.angular_gain_rad_s_per_nm,
        linear_compliance_m_per_n=args.linear_compliance_m_per_n,
        angular_compliance_rad_per_nm=args.angular_compliance_rad_per_nm,
        max_linear_velocity_m_s=args.max_linear_velocity_m_s,
        max_angular_velocity_rad_s=args.max_angular_velocity_rad_s,
        max_linear_offset_m=args.max_linear_offset_m,
        max_angular_offset_rad=args.max_angular_offset_rad,
        return_rate_s=args.return_rate_s,
    )

    records = []
    max_force_norm = 0.0
    max_torque_norm = 0.0
    max_offset_norm = 0.0
    max_angular_offset_norm = 0.0
    max_pose_linear = 0.0
    max_pose_angular = 0.0
    max_joint_delta = 0.0
    missed = 0
    motion_started = False
    loop_durations = []
    loop_intervals = []
    state_read_durations = []
    force_read_durations = []
    compute_durations = []
    command_durations = []
    latency_events: dict[str, dict[str, float]] = {}
    run_start = time.perf_counter()
    previous_loop_start = run_start
    next_deadline = run_start
    last_state = start_state
    print(
        f"{side}: starting pose admittance hz={args.control_hz:.2f} "
        f"duration={args.duration_sec:.2f}s source={args.force_source} "
        f"axes={args.axes} sign={args.force_sign:+.1f} "
        f"angular_sign={args.angular_force_sign:+.1f} angular_axis_signs={_format(args.angular_axis_signs, precision=0)}"
    )

    try:
        for index in range(loops):
            loop_start = time.perf_counter()
            dt = loop_start - previous_loop_start if index > 0 else period
            if index > 0:
                loop_intervals.append(dt)
            previous_loop_start = loop_start
            time_sec = loop_start - run_start

            force_start = time.perf_counter()
            readings = _read_force(robot, side)
            force_read_sec = time.perf_counter() - force_start
            force_read_durations.append(force_read_sec)

            compute_start = time.perf_counter()
            processed = processor.process(readings[args.force_source])
            update = admittance.update(processed["deadbanded"], dt)
            pose_cmd = normalize_pose(apply_pose_offset(start_state["pose"], update["offset"]))
            compute_sec = time.perf_counter() - compute_start
            compute_durations.append(compute_sec)

            command_start = time.perf_counter()
            _require_zero(
                robot.Movep_CANFD(
                    pose_cmd,
                    follow=False,
                    trajectory_mode=args.trajectory_mode,
                    radio=args.radio,
                ),
                f"{side} Movep_CANFD",
            )
            command_sec = time.perf_counter() - command_start
            command_durations.append(command_sec)
            motion_started = True

            state_sampled = index % args.state_read_every == 0 or index + 1 == loops
            if state_sampled:
                state_start = time.perf_counter()
                state = _read_state(robot, side)
                state_read_sec = time.perf_counter() - state_start
                state_read_durations.append(state_read_sec)
                last_state = state
            else:
                state = last_state
                state_read_sec = 0.0

            joint_delta, pose_linear, pose_angular, force_norm, torque_norm = _check_runtime_limits(
                side=side,
                start_state=start_state,
                state=state,
                tare_corrected=processed["tare_corrected"],
                offset=update["offset"],
                args=args,
            )
            offset_norm = float(update["linear_offset_norm_m"])
            angular_offset_norm = float(update["angular_offset_norm_rad"])
            event_active_norm = _active_norm(update["active_wrench"], args.enabled_axes)
            event_offset_norm = _offset_norm(update["offset"], args.enabled_axes)
            event_stimulus_norm = _stimulus_norm(processed["tare_corrected"], args.enabled_axes)
            max_force_norm = max(max_force_norm, force_norm)
            max_torque_norm = max(max_torque_norm, torque_norm)
            max_offset_norm = max(max_offset_norm, offset_norm)
            max_angular_offset_norm = max(max_angular_offset_norm, angular_offset_norm)
            max_pose_linear = max(max_pose_linear, pose_linear)
            max_pose_angular = max(max_pose_angular, pose_angular)
            max_joint_delta = max(max_joint_delta, joint_delta)
            if event_stimulus_norm >= args.latency_stimulus_threshold:
                _remember_event(latency_events, "stimulus_threshold", index + 1, time_sec, event_stimulus_norm)
            if event_active_norm >= args.latency_active_threshold:
                _remember_event(latency_events, "active_threshold", index + 1, time_sec, event_active_norm)
            if event_offset_norm >= args.latency_offset_threshold:
                _remember_event(latency_events, "offset_threshold", index + 1, time_sec, event_offset_norm)
            observed_event_norm = pose_angular if _uses_rotation(args.enabled_axes) and not any(args.enabled_axes[:3]) else pose_linear
            if observed_event_norm >= args.latency_pose_threshold:
                _remember_event(latency_events, "pose_threshold", index + 1, time_sec, observed_event_norm)

            if index % args.print_every == 0 or index + 1 == loops:
                print(
                    f"{side}:adm step={index + 1}/{loops} "
                    f"force={_format(processed['tare_corrected'][:3])} "
                    f"torque={_format(processed['tare_corrected'][3:])} "
                    f"filtered={_format(processed['filtered'][:3])} "
                    f"active={_format(update['active_wrench'], precision=4)} "
                    f"target={_format(update['target_offset'], precision=5)} "
                    f"offset={_format(update['offset'], precision=5)} "
                    f"|offset|={offset_norm:.5f}m "
                    f"|angular_offset|={angular_offset_norm:.5f}rad "
                    f"pose_delta={pose_linear:.5f}m/{pose_angular:.5f}rad "
                    f"joint_delta={joint_delta:.4f}deg "
                    f"loop_ms={(time.perf_counter() - loop_start) * 1000.0:.2f}"
                )

            elapsed = time.perf_counter() - loop_start
            missed_deadline = elapsed > period
            if missed_deadline:
                missed += 1
            next_deadline += period
            sleep_sec = next_deadline - time.perf_counter()
            loop_durations.append(elapsed)

            records.append(
                {
                    "step": index + 1,
                    "time_sec": time_sec,
                    "dt": dt,
                    "pose_cmd": pose_cmd,
                    "observed_pose": state["pose"],
                    "observed_joint_deg": state["joint_deg"],
                    "measured_wrench": processed["measured"],
                    "tare_corrected_wrench": processed["tare_corrected"],
                    "filtered_wrench": processed["filtered"],
                    "deadbanded_wrench": processed["deadbanded"],
                    "active_wrench": update["active_wrench"],
                    "target_offset": update["target_offset"],
                    "velocity": update["velocity"],
                    "offset": update["offset"],
                    "force_norm_n": force_norm,
                    "torque_norm_nm": torque_norm,
                    "pose_linear_delta_m": pose_linear,
                    "pose_angular_delta_rad": pose_angular,
                    "joint_delta_deg": joint_delta,
                    "timing": {
                        "state_sampled": state_sampled,
                        "state_read_sec": state_read_sec,
                        "force_read_sec": force_read_sec,
                        "compute_sec": compute_sec,
                        "command_sec": command_sec,
                        "loop_elapsed_sec": elapsed,
                        "sleep_sec": max(0.0, sleep_sec),
                        "missed_deadline": missed_deadline,
                    },
                }
            )

            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
    finally:
        if motion_started:
            slow_stop_code = robot.Arm_Slow_Stop()
            print(f"{side}: Arm_Slow_Stop -> {slow_stop_code}")
            time.sleep(args.stop_settle_sec)

    print(
        f"{side}: pose admittance complete max_force={max_force_norm:.4f}N "
        f"max_torque={max_torque_norm:.4f}Nm max_offset={max_offset_norm:.5f}m "
        f"max_angular_offset={max_angular_offset_norm:.5f}rad "
        f"max_pose_delta={max_pose_linear:.5f}m/{max_pose_angular:.5f}rad "
        f"max_joint_delta={max_joint_delta:.4f}deg missed={missed}/{loops}"
    )
    timing_summary = {
        "period_ms": period * 1000.0,
        "loop_elapsed": _stats_ms(loop_durations),
        "loop_interval": _stats_ms(loop_intervals),
        "state_read": _stats_ms(state_read_durations),
        "force_read": _stats_ms(force_read_durations),
        "compute": _stats_ms(compute_durations),
        "command": _stats_ms(command_durations),
    }
    latency_estimates_ms = {
        "stimulus_to_active": _latency_ms(latency_events, "active_threshold", "stimulus_threshold"),
        "stimulus_to_offset": _latency_ms(latency_events, "offset_threshold", "stimulus_threshold"),
        "stimulus_to_pose": _latency_ms(latency_events, "pose_threshold", "stimulus_threshold"),
        "active_to_offset": _latency_ms(latency_events, "offset_threshold", "active_threshold"),
        "offset_to_pose": _latency_ms(latency_events, "pose_threshold", "offset_threshold"),
    }
    print(
        f"{side}: timing loop_ms mean/p95/p99/max="
        f"{timing_summary['loop_elapsed']['mean_ms']:.3f}/"
        f"{timing_summary['loop_elapsed']['p95_ms']:.3f}/"
        f"{timing_summary['loop_elapsed']['p99_ms']:.3f}/"
        f"{timing_summary['loop_elapsed']['max_ms']:.3f} "
        f"command_p99={timing_summary['command']['p99_ms']:.3f}ms "
        f"force_read_p99={timing_summary['force_read']['p99_ms']:.3f}ms"
    )
    print(f"{side}: latency_events={latency_events}")
    print(f"{side}: latency_estimates_ms={latency_estimates_ms}")
    return {
        "records": records,
        "max_force_norm_n": max_force_norm,
        "max_torque_norm_nm": max_torque_norm,
        "max_linear_offset_m": max_offset_norm,
        "max_angular_offset_rad": max_angular_offset_norm,
        "max_pose_linear_delta_m": max_pose_linear,
        "max_pose_angular_delta_rad": max_pose_angular,
        "max_joint_delta_deg": max_joint_delta,
        "missed_deadlines": missed,
        "loops": loops,
        "timing_summary": timing_summary,
        "latency_thresholds": {
            "stimulus_threshold": args.latency_stimulus_threshold,
            "active_threshold": args.latency_active_threshold,
            "offset_threshold": args.latency_offset_threshold,
            "pose_threshold": args.latency_pose_threshold,
        },
        "latency_events": latency_events,
        "latency_estimates_ms": latency_estimates_ms,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe RM65 outer-loop pose admittance smoke test. It holds the startup "
            "TCP orientation, computes a small translational compliance offset from "
            "external force, and sends absolute Movep_CANFD targets."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="inspire")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for motion.")
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--control-hz", type=float, default=40.0)
    parser.add_argument("--force-source", choices=("zero", "work", "tool"), default="work")
    parser.add_argument("--axes", default="xyz", help="Enabled axes, e.g. x, z, xyz, rx, rz, or rx,ry,rz.")
    parser.add_argument("--force-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--angular-force-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument(
        "--angular-axis-signs",
        default="1,1,-1",
        help="Per-axis rotational signs for rx,ry,rz, applied after --angular-force-sign.",
    )
    parser.add_argument("--tare-sec", type=float, default=1.0)
    parser.add_argument("--lowpass-alpha", type=float, default=0.85)
    parser.add_argument("--force-deadband-n", type=float, default=0.30)
    parser.add_argument("--torque-deadband-nm", type=float, default=0.03)
    parser.add_argument("--linear-gain-m-s-per-n", type=float, default=0.010)
    parser.add_argument("--angular-gain-rad-s-per-nm", type=float, default=0.0)
    parser.add_argument(
        "--linear-compliance-m-per-n",
        type=float,
        default=0.0,
        help="Optional spring-like direct compliance. Example: 0.004 means 4mm target offset per 1N.",
    )
    parser.add_argument(
        "--angular-compliance-rad-per-nm",
        type=float,
        default=0.0,
        help="Optional rotational compliance. Example: 0.02 means 0.02rad target RPY offset per 1Nm.",
    )
    parser.add_argument("--max-linear-velocity-m-s", type=float, default=0.05)
    parser.add_argument("--max-angular-velocity-rad-s", type=float, default=0.40)
    parser.add_argument("--max-linear-offset-m", type=float, default=0.10)
    parser.add_argument("--max-angular-offset-rad", type=float, default=0.50)
    parser.add_argument("--return-rate-s", type=float, default=0.35)
    parser.add_argument("--trajectory-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--radio", type=int, default=0)
    parser.add_argument("--max-start-force-n", type=float, default=5.0)
    parser.add_argument("--max-start-torque-nm", type=float, default=0.30)
    parser.add_argument("--hard-stop-force-n", type=float, default=50.0)
    parser.add_argument("--hard-stop-torque-nm", type=float, default=10.0)
    parser.add_argument("--max-observed-pose-linear-m", type=float, default=0.20)
    parser.add_argument("--max-observed-pose-angular-rad", type=float, default=1.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=30.0)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--state-read-every", type=int, default=1)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--latency-stimulus-threshold", type=float, default=0.50)
    parser.add_argument("--latency-active-threshold", type=float, default=0.05)
    parser.add_argument("--latency-offset-threshold", type=float, default=0.001)
    parser.add_argument("--latency-pose-threshold", type=float, default=0.001)
    return parser


def _validate_args(args) -> None:
    args.enabled_axes = parse_axes(args.axes, allow_rotation=True)
    args.angular_axis_signs = parse_signs(args.angular_axis_signs, expected_len=3, label="angular-axis-signs")
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"pose admittance smoke requires --confirm {CONFIRM_TEXT}")
    if args.duration_sec <= 0.0:
        raise RuntimeError("--duration-sec must be positive")
    if args.control_hz <= 0.0 or args.control_hz > 40.0:
        raise RuntimeError("--control-hz must be in (0, 40] for this smoke test")
    if args.tare_sec <= 0.0:
        raise RuntimeError("--tare-sec must be positive")
    if not 0.0 < args.lowpass_alpha <= 1.0:
        raise RuntimeError("--lowpass-alpha must be in (0, 1]")
    if args.print_every < 1:
        raise RuntimeError("--print-every must be >= 1")
    if args.state_read_every < 1:
        raise RuntimeError("--state-read-every must be >= 1")
    if args.stop_settle_sec < 0.0:
        raise RuntimeError("--stop-settle-sec must be >= 0")
    for name in (
        "force_deadband_n",
        "torque_deadband_nm",
        "linear_gain_m_s_per_n",
        "angular_gain_rad_s_per_nm",
        "linear_compliance_m_per_n",
        "angular_compliance_rad_per_nm",
        "max_linear_velocity_m_s",
        "max_angular_velocity_rad_s",
        "max_linear_offset_m",
        "max_angular_offset_rad",
        "return_rate_s",
        "max_start_force_n",
        "max_start_torque_nm",
        "hard_stop_force_n",
        "hard_stop_torque_nm",
        "max_observed_pose_linear_m",
        "max_observed_pose_angular_rad",
        "max_joint_delta_deg",
        "latency_stimulus_threshold",
        "latency_active_threshold",
        "latency_offset_threshold",
        "latency_pose_threshold",
    ):
        if getattr(args, name) < 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be non-negative")
    if args.max_linear_offset_m > args.max_observed_pose_linear_m:
        raise RuntimeError("--max-linear-offset-m must be <= --max-observed-pose-linear-m")
    if args.hard_stop_force_n <= args.force_deadband_n:
        raise RuntimeError("--hard-stop-force-n must exceed --force-deadband-n")


def _write_report(path_text: str, report: dict[str, object]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote report: {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    robot = None
    try:
        robot = _connect_robot(side, ip)
        _check_expected_tool(robot, args)
        start_state = _read_state(robot, side)
        start_force = _read_force(robot, side)
        print(f"{side}: start_joint_deg={_format(start_state['joint_deg'])}")
        print(f"{side}: start_pose_xyz_rpy={_format(start_state['pose'])}")
        _print_force(f"{side}: start_force", start_force)
        _check_start_force(start_force, args)
        tare = _sample_tare(robot, side, args)
        report: dict[str, object] = {
            "schema_version": 1,
            "generated_by": "hardware/rm65/utils/scripts/rm65_pose_admittance_smoke.py",
            "arm": side,
            "ip": ip,
            "executed": bool(args.execute),
            "force_source": args.force_source,
            "axes": args.axes,
            "enabled_axes": args.enabled_axes,
            "control_hz": args.control_hz,
            "duration_sec": args.duration_sec,
            "force_sign": args.force_sign,
            "angular_force_sign": args.angular_force_sign,
            "angular_axis_signs": args.angular_axis_signs,
            "start_state": start_state,
            "start_force": start_force,
            "tare": tare,
            "parameters": {
                "lowpass_alpha": args.lowpass_alpha,
                "force_deadband_n": args.force_deadband_n,
                "linear_gain_m_s_per_n": args.linear_gain_m_s_per_n,
                "angular_gain_rad_s_per_nm": args.angular_gain_rad_s_per_nm,
                "linear_compliance_m_per_n": args.linear_compliance_m_per_n,
                "angular_compliance_rad_per_nm": args.angular_compliance_rad_per_nm,
                "max_linear_velocity_m_s": args.max_linear_velocity_m_s,
                "max_angular_velocity_rad_s": args.max_angular_velocity_rad_s,
                "max_linear_offset_m": args.max_linear_offset_m,
                "max_angular_offset_rad": args.max_angular_offset_rad,
                "return_rate_s": args.return_rate_s,
                "trajectory_mode": args.trajectory_mode,
                "radio": args.radio,
                "state_read_every": args.state_read_every,
            },
        }
        if not args.execute:
            print("DRY-RUN: pose admittance commands will not be sent.")
            _write_report(args.output_json, report)
            return 0
        result = _run_admittance(robot, side, start_state, tare, args)
        final_state = _read_state(robot, side)
        final_force = _read_force(robot, side)
        linear, angular = _pose_delta(start_state["pose"], final_state["pose"])
        print(f"{side}: final_joint_delta={_joint_delta_deg(start_state['joint_deg'], final_state['joint_deg']):.4f}deg")
        print(f"{side}: final_pose_delta={linear:.5f}m/{angular:.5f}rad")
        _print_force(f"{side}: final_force", final_force)
        report.update(result)
        report["final_state"] = final_state
        report["final_force"] = final_force
        _write_report(args.output_json, report)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            robot.Arm_Socket_Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
