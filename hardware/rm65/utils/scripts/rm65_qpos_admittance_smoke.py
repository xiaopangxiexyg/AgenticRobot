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


CONFIRM_TEXT = "TEST_RM65_QPOS_ADMITTANCE"


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _selected_ip(args) -> tuple[str, str]:
    return ("left", args.left_ip) if args.arm == "left" else ("right", args.right_ip)


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


def _check_expected_tool(robot: Arm, expected_tool: str) -> str:
    if not expected_tool:
        return ""
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    current_name = _current_tool_name((code, frame))
    print(f"current_tool={current_name}")
    if expected_tool not in current_name:
        raise RuntimeError(f"expected active tool containing '{expected_tool}', got '{current_name}'")
    return current_name


def _print_force(prefix: str, readings: dict[str, list[float]]) -> None:
    for source in ("raw", "zero", "work", "tool"):
        wrench = readings[source]
        print(
            f"{prefix} {source}={_format(wrench)} "
            f"|F|={norm(wrench[:3]):.4f}N |T|={norm(wrench[3:]):.4f}Nm"
        )


def _joint_delta_deg(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _pose_delta(a: list[float], b: list[float]) -> tuple[float, float]:
    linear = norm(a[index] - b[index] for index in range(3))
    angular = norm(angle_delta(a[index], b[index]) for index in range(3, 6))
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


def _sample_tare(robot: Arm, side: str, args) -> list[float]:
    samples = max(1, int(round(args.tare_sec * args.control_hz)))
    interval = 1.0 / args.control_hz
    print(f"{side}: sampling tare source={args.force_source} samples={samples}")
    wrenches = []
    for index in range(samples):
        readings = _read_force(robot, side)
        wrenches.append(readings[args.force_source])
        if index + 1 < samples:
            time.sleep(interval)
    tare = mean_wrench(wrenches)
    print(f"{side}: tare={_format(tare)} |F|={norm(tare[:3]):.4f}N |T|={norm(tare[3:]):.4f}Nm")
    return tare


def _check_start_force(readings: dict[str, list[float]], args) -> None:
    wrench = readings[args.force_source]
    force_norm = norm(wrench[:3])
    torque_norm = norm(wrench[3:])
    print(f"precheck force_source={args.force_source} |F|={force_norm:.4f}N |T|={torque_norm:.4f}Nm")
    if force_norm > args.max_start_force_n:
        raise RuntimeError(f"start force {force_norm:.4f}N > --max-start-force-n")
    if torque_norm > args.max_start_torque_nm:
        raise RuntimeError(f"start torque {torque_norm:.4f}Nm > --max-start-torque-nm")


def _solve_ik(robot: Arm, seed_joint: list[float], target_pose: list[float], side: str) -> list[float]:
    code, joint = robot.Algo_Inverse_Kinematics(seed_joint, target_pose, flag=1)
    if int(code) != 0:
        raise RuntimeError(f"{side}: IK failed with return code {code} for target_pose={_format(target_pose)}")
    return finite_list(joint[:6], expected_len=6, label="ik_joint_deg")


def _check_runtime_limits(
    *,
    side: str,
    start_state: dict[str, list[float]],
    state: dict[str, list[float]],
    command_joint: list[float],
    previous_command_joint: list[float],
    tare_corrected: list[float],
    offset: list[float],
    args,
) -> tuple[float, float, float, float, float, float, float]:
    observed_joint_delta = _joint_delta_deg(start_state["joint_deg"], state["joint_deg"])
    command_joint_delta = _joint_delta_deg(start_state["joint_deg"], command_joint)
    command_joint_step = _joint_delta_deg(previous_command_joint, command_joint)
    pose_linear, pose_angular = _pose_delta(start_state["pose"], state["pose"])
    force_norm = norm(tare_corrected[:3])
    torque_norm = norm(tare_corrected[3:])
    linear_offset = norm(offset[:3])
    angular_offset = norm(offset[3:])
    if force_norm > args.hard_stop_force_n:
        raise RuntimeError(f"{side}: force {force_norm:.4f}N > --hard-stop-force-n")
    if torque_norm > args.hard_stop_torque_nm:
        raise RuntimeError(f"{side}: torque {torque_norm:.4f}Nm > --hard-stop-torque-nm")
    if linear_offset > args.max_linear_offset_m * 1.05:
        raise RuntimeError(f"{side}: linear offset {linear_offset:.5f}m exceeded configured limit")
    if args.max_angular_offset_rad > 0.0 and angular_offset > args.max_angular_offset_rad * 1.05:
        raise RuntimeError(f"{side}: angular offset {angular_offset:.5f}rad exceeded configured limit")
    if observed_joint_delta > args.max_observed_joint_delta_deg:
        raise RuntimeError(f"{side}: observed joint delta {observed_joint_delta:.4f}deg > --max-observed-joint-delta-deg")
    if command_joint_delta > args.max_command_joint_delta_deg:
        raise RuntimeError(f"{side}: command joint delta {command_joint_delta:.4f}deg > --max-command-joint-delta-deg")
    if command_joint_step > args.max_command_joint_step_deg:
        raise RuntimeError(f"{side}: command joint step {command_joint_step:.4f}deg > --max-command-joint-step-deg")
    if pose_linear > args.max_observed_pose_linear_m:
        raise RuntimeError(f"{side}: observed pose linear delta {pose_linear:.5f}m > --max-observed-pose-linear-m")
    if pose_angular > args.max_observed_pose_angular_rad:
        raise RuntimeError(f"{side}: observed pose angular delta {pose_angular:.5f}rad > --max-observed-pose-angular-rad")
    return (
        observed_joint_delta,
        command_joint_delta,
        command_joint_step,
        pose_linear,
        pose_angular,
        force_norm,
        torque_norm,
    )


def _run_qpos_admittance(robot: Arm, side: str, start_state: dict[str, list[float]], tare: list[float], args) -> dict[str, object]:
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

    print(
        f"{side}: starting QPOS admittance hz={args.control_hz:.2f} duration={args.duration_sec:.2f}s "
        f"source={args.force_source} axes={args.axes} angular_axis_signs={_format(args.angular_axis_signs, precision=0)}"
    )
    records = []
    timing = {"force_read": [], "state_read": [], "compute": [], "ik": [], "command": [], "loop_elapsed": [], "loop_interval": []}
    max_force_norm = 0.0
    max_torque_norm = 0.0
    max_observed_joint_delta = 0.0
    max_command_joint_delta = 0.0
    max_command_joint_step = 0.0
    max_pose_linear = 0.0
    max_pose_angular = 0.0
    max_linear_offset = 0.0
    max_angular_offset = 0.0
    missed = 0
    last_state = start_state
    previous_command_joint = list(start_state["joint_deg"])
    ik_seed = list(start_state["joint_deg"])
    previous_loop_start = None
    run_start = time.perf_counter()
    next_deadline = run_start
    motion_started = False

    try:
        for index in range(loops):
            loop_start = time.perf_counter()
            if previous_loop_start is not None:
                timing["loop_interval"].append(loop_start - previous_loop_start)
            dt = loop_start - previous_loop_start if previous_loop_start is not None else period
            previous_loop_start = loop_start
            time_sec = loop_start - run_start

            force_start = time.perf_counter()
            readings = _read_force(robot, side)
            timing["force_read"].append(time.perf_counter() - force_start)

            compute_start = time.perf_counter()
            processed = processor.process(readings[args.force_source])
            update = admittance.update(processed["deadbanded"], dt)
            target_pose = normalize_pose(apply_pose_offset(start_state["pose"], update["offset"]))
            timing["compute"].append(time.perf_counter() - compute_start)

            ik_start = time.perf_counter()
            command_joint = _solve_ik(robot, ik_seed, target_pose, side)
            timing["ik"].append(time.perf_counter() - ik_start)
            ik_seed = list(command_joint)

            command_start = time.perf_counter()
            _require_zero(
                robot.Movej_CANFD(
                    command_joint,
                    follow=False,
                    trajectory_mode=args.trajectory_mode,
                    radio=args.radio,
                ),
                f"{side} Movej_CANFD",
            )
            timing["command"].append(time.perf_counter() - command_start)
            motion_started = True

            state_sampled = index % args.state_read_every == 0 or index + 1 == loops
            state_read_sec = 0.0
            if state_sampled:
                state_start = time.perf_counter()
                last_state = _read_state(robot, side)
                state_read_sec = time.perf_counter() - state_start
                timing["state_read"].append(state_read_sec)
            state = last_state

            (
                observed_joint_delta,
                command_joint_delta,
                command_joint_step,
                pose_linear,
                pose_angular,
                force_norm,
                torque_norm,
            ) = _check_runtime_limits(
                side=side,
                start_state=start_state,
                state=state,
                command_joint=command_joint,
                previous_command_joint=previous_command_joint,
                tare_corrected=processed["tare_corrected"],
                offset=update["offset"],
                args=args,
            )
            previous_command_joint = list(command_joint)
            linear_offset = float(update["linear_offset_norm_m"])
            angular_offset = float(update["angular_offset_norm_rad"])
            max_force_norm = max(max_force_norm, force_norm)
            max_torque_norm = max(max_torque_norm, torque_norm)
            max_observed_joint_delta = max(max_observed_joint_delta, observed_joint_delta)
            max_command_joint_delta = max(max_command_joint_delta, command_joint_delta)
            max_command_joint_step = max(max_command_joint_step, command_joint_step)
            max_pose_linear = max(max_pose_linear, pose_linear)
            max_pose_angular = max(max_pose_angular, pose_angular)
            max_linear_offset = max(max_linear_offset, linear_offset)
            max_angular_offset = max(max_angular_offset, angular_offset)

            loop_elapsed = time.perf_counter() - loop_start
            timing["loop_elapsed"].append(loop_elapsed)
            missed_deadline = loop_elapsed > period
            if missed_deadline:
                missed += 1
            if index % args.print_every == 0 or index + 1 == loops:
                print(
                    f"{side}:qpos_adm step={index + 1}/{loops} "
                    f"offset={_format(update['offset'], precision=5)} "
                    f"cmd_joint={_format(command_joint, precision=3)} "
                    f"cmd_delta={command_joint_delta:.3f}deg step={command_joint_step:.3f}deg "
                    f"obs_joint_delta={observed_joint_delta:.3f}deg "
                    f"pose_delta={pose_linear:.5f}m/{pose_angular:.5f}rad "
                    f"|F|={force_norm:.3f}N |T|={torque_norm:.3f}Nm "
                    f"loop_ms={loop_elapsed * 1000.0:.2f}"
                )

            records.append(
                {
                    "step": index + 1,
                    "time_sec": time_sec,
                    "target_pose": target_pose,
                    "command_joint_deg": command_joint,
                    "observed_joint_deg": state["joint_deg"],
                    "observed_pose": state["pose"],
                    "tare_corrected_wrench": processed["tare_corrected"],
                    "filtered_wrench": processed["filtered"],
                    "active_wrench": update["active_wrench"],
                    "target_offset": update["target_offset"],
                    "offset": update["offset"],
                    "command_joint_delta_deg": command_joint_delta,
                    "command_joint_step_deg": command_joint_step,
                    "observed_joint_delta_deg": observed_joint_delta,
                    "pose_linear_delta_m": pose_linear,
                    "pose_angular_delta_rad": pose_angular,
                    "force_norm_n": force_norm,
                    "torque_norm_nm": torque_norm,
                    "timing": {
                        "state_sampled": state_sampled,
                        "state_read_sec": state_read_sec,
                        "loop_elapsed_sec": loop_elapsed,
                        "missed_deadline": missed_deadline,
                    },
                }
            )
            next_deadline += period
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
    finally:
        if motion_started:
            code = robot.Arm_Slow_Stop()
            print(f"{side}: Arm_Slow_Stop -> {code}")
            time.sleep(args.stop_settle_sec)

    summary = {
        "loops": loops,
        "missed_deadlines": missed,
        "max_force_norm_n": max_force_norm,
        "max_torque_norm_nm": max_torque_norm,
        "max_linear_offset_m": max_linear_offset,
        "max_angular_offset_rad": max_angular_offset,
        "max_observed_joint_delta_deg": max_observed_joint_delta,
        "max_command_joint_delta_deg": max_command_joint_delta,
        "max_command_joint_step_deg": max_command_joint_step,
        "max_pose_linear_delta_m": max_pose_linear,
        "max_pose_angular_delta_rad": max_pose_angular,
        "timing": {key: _stats_ms(values) for key, values in timing.items() if values},
    }
    print(f"{side}: QPOS admittance summary={summary}")
    return {"records": records, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe RM65 QPOS/CANFD outer-loop admittance smoke test.")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="inspire")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for motion.")
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--control-hz", type=float, default=40.0)
    parser.add_argument("--force-source", choices=("zero", "work", "tool"), default="work")
    parser.add_argument("--axes", default="all")
    parser.add_argument("--force-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--angular-force-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--angular-axis-signs", default="1,1,-1")
    parser.add_argument("--tare-sec", type=float, default=1.0)
    parser.add_argument("--lowpass-alpha", type=float, default=0.85)
    parser.add_argument("--force-deadband-n", type=float, default=0.30)
    parser.add_argument("--torque-deadband-nm", type=float, default=0.03)
    parser.add_argument("--linear-gain-m-s-per-n", type=float, default=0.0)
    parser.add_argument("--angular-gain-rad-s-per-nm", type=float, default=0.0)
    parser.add_argument("--linear-compliance-m-per-n", type=float, default=0.004)
    parser.add_argument("--angular-compliance-rad-per-nm", type=float, default=0.04)
    parser.add_argument("--max-linear-velocity-m-s", type=float, default=0.04)
    parser.add_argument("--max-angular-velocity-rad-s", type=float, default=0.15)
    parser.add_argument("--max-linear-offset-m", type=float, default=0.2)
    parser.add_argument("--max-angular-offset-rad", type=float, default=0.4)
    parser.add_argument("--return-rate-s", type=float, default=0.35)
    parser.add_argument("--trajectory-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--radio", type=int, default=0)
    parser.add_argument("--max-start-force-n", type=float, default=5.0)
    parser.add_argument("--max-start-torque-nm", type=float, default=0.30)
    parser.add_argument("--hard-stop-force-n", type=float, default=50.0)
    parser.add_argument("--hard-stop-torque-nm", type=float, default=10.0)
    parser.add_argument("--max-observed-pose-linear-m", type=float, default=0.20)
    parser.add_argument("--max-observed-pose-angular-rad", type=float, default=1.0)
    parser.add_argument("--max-observed-joint-delta-deg", type=float, default=30.0)
    parser.add_argument("--max-command-joint-delta-deg", type=float, default=30.0)
    parser.add_argument("--max-command-joint-step-deg", type=float, default=5.0)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--state-read-every", type=int, default=1)
    parser.add_argument("--output-json", default="")
    return parser


def _validate_args(args) -> None:
    args.enabled_axes = parse_axes(args.axes, allow_rotation=True)
    args.angular_axis_signs = parse_signs(args.angular_axis_signs, expected_len=3, label="angular-axis-signs")
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"QPOS admittance smoke requires --confirm {CONFIRM_TEXT}")
    if args.duration_sec <= 0.0:
        raise RuntimeError("--duration-sec must be positive")
    if args.control_hz <= 0.0 or args.control_hz > 40.0:
        raise RuntimeError("--control-hz must be in (0, 40]")
    if args.tare_sec <= 0.0:
        raise RuntimeError("--tare-sec must be positive")
    if args.print_every < 1 or args.state_read_every < 1:
        raise RuntimeError("--print-every and --state-read-every must be >= 1")
    for name in (
        "lowpass_alpha",
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
        "max_observed_joint_delta_deg",
        "max_command_joint_delta_deg",
        "max_command_joint_step_deg",
        "stop_settle_sec",
    ):
        if getattr(args, name) < 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be non-negative")
    if not 0.0 < args.lowpass_alpha <= 1.0:
        raise RuntimeError("--lowpass-alpha must be in (0, 1]")


def _write_report(path_text: str, report: dict[str, object]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
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
        active_tool = _check_expected_tool(robot, args.expected_tool)
        start_state = _read_state(robot, side)
        start_force = _read_force(robot, side)
        print(f"{side}: start_joint_deg={_format(start_state['joint_deg'])}")
        print(f"{side}: start_pose_xyz_rpy={_format(start_state['pose'])}")
        _print_force(f"{side}: start_force", start_force)
        _check_start_force(start_force, args)
        tare = _sample_tare(robot, side, args)
        report: dict[str, object] = {
            "schema_version": 1,
            "generated_by": "hardware/rm65/utils/scripts/rm65_qpos_admittance_smoke.py",
            "arm": side,
            "ip": ip,
            "active_tool": active_tool,
            "executed": bool(args.execute),
            "force_source": args.force_source,
            "axes": args.axes,
            "enabled_axes": args.enabled_axes,
            "control_hz": args.control_hz,
            "duration_sec": args.duration_sec,
            "parameters": vars(args),
            "start_state": start_state,
            "start_force": start_force,
            "tare": tare,
        }
        if not args.execute:
            print("DRY-RUN: QPOS admittance commands will not be sent.")
            _write_report(args.output_json, report)
            return 0
        result = _run_qpos_admittance(robot, side, start_state, tare, args)
        report.update(result)
        report["final_state"] = _read_state(robot, side)
        report["final_force"] = _read_force(robot, side)
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
