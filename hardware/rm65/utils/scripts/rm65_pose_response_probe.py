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

from hardware.rm65.force_compensation import finite_list, norm  # noqa: E402
from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65  # noqa: E402


CONFIRM_TEXT = "TEST_RM65_POSE_RESPONSE"
AXES = ("x", "y", "z", "rx", "ry", "rz")


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _axis_index(axis: str) -> int:
    if axis not in AXES:
        raise RuntimeError(f"unsupported axis '{axis}', expected one of {AXES}")
    return AXES.index(axis)


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


def _check_expected_tool(robot: Arm, expected_tool: str) -> None:
    if not expected_tool:
        return
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    current_name = _current_tool_name((code, frame))
    print(f"current_tool={current_name}")
    if expected_tool not in current_name:
        raise RuntimeError(f"expected active tool containing '{expected_tool}', got '{current_name}'")


def _angle_delta(current: float, previous: float) -> float:
    delta = current - previous
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _axis_delta(current_pose: list[float], start_pose: list[float], axis_index: int) -> float:
    if axis_index < 3:
        return current_pose[axis_index] - start_pose[axis_index]
    return _angle_delta(current_pose[axis_index], start_pose[axis_index])


def _pose_delta(start_pose: list[float], current_pose: list[float]) -> tuple[float, float]:
    linear = norm(current_pose[index] - start_pose[index] for index in range(3))
    angular = norm(_angle_delta(current_pose[index], start_pose[index]) for index in range(3, 6))
    return linear, angular


def _joint_delta_deg(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


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


def _command_offset_at(t_sec: float, args) -> float:
    if t_sec < args.pre_sec:
        return 0.0
    if t_sec < args.pre_sec + args.hold_sec:
        return args.offset
    return 0.0


def _command_pose(start_pose: list[float], axis_index: int, offset: float) -> list[float]:
    pose = list(start_pose)
    pose[axis_index] += offset
    return pose


def _check_force(force: dict[str, list[float]], args) -> tuple[float, float]:
    wrench = force[args.monitor_force_source]
    force_norm = norm(wrench[:3])
    torque_norm = norm(wrench[3:])
    if force_norm > args.max_monitor_force_n:
        raise RuntimeError(f"{args.monitor_force_source} force {force_norm:.4f}N > --max-monitor-force-n")
    if torque_norm > args.max_monitor_torque_nm:
        raise RuntimeError(f"{args.monitor_force_source} torque {torque_norm:.4f}Nm > --max-monitor-torque-nm")
    return force_norm, torque_norm


def _check_state_limits(start_state: dict[str, list[float]], state: dict[str, list[float]], args) -> tuple[float, float, float]:
    joint_delta = _joint_delta_deg(start_state["joint_deg"], state["joint_deg"])
    pose_linear, pose_angular = _pose_delta(start_state["pose"], state["pose"])
    if joint_delta > args.max_joint_delta_deg:
        raise RuntimeError(f"joint delta {joint_delta:.4f}deg > --max-joint-delta-deg")
    if pose_linear > args.max_pose_linear_m:
        raise RuntimeError(f"pose linear delta {pose_linear:.5f}m > --max-pose-linear-m")
    if pose_angular > args.max_pose_angular_rad:
        raise RuntimeError(f"pose angular delta {pose_angular:.5f}rad > --max-pose-angular-rad")
    return joint_delta, pose_linear, pose_angular


def _response_events(records: list[dict[str, object]], offset: float) -> dict[str, object]:
    sign = 1.0 if offset >= 0.0 else -1.0
    magnitude = abs(offset)
    events: dict[str, object] = {}
    step_time = None
    return_time = None
    last_cmd = 0.0
    for record in records:
        cmd = float(record["command_axis_offset"])
        t_sec = float(record["time_sec"])
        if step_time is None and abs(cmd) >= magnitude * 0.95 and abs(last_cmd) < magnitude * 0.5:
            step_time = t_sec
            events["command_step"] = {"time_sec": t_sec, "step": record["step"], "value": cmd}
        if step_time is not None and return_time is None and abs(cmd) < magnitude * 0.1 and abs(last_cmd) >= magnitude * 0.5:
            return_time = t_sec
            events["command_return"] = {"time_sec": t_sec, "step": record["step"], "value": cmd}
        last_cmd = cmd
    if step_time is None:
        return events
    for fraction in (0.10, 0.50, 0.90):
        key = f"rise_{int(fraction * 100)}"
        threshold = magnitude * fraction
        for record in records:
            if not record.get("state_sampled"):
                continue
            t_sec = float(record["time_sec"])
            if t_sec < step_time:
                continue
            observed = sign * float(record["observed_axis_offset"])
            if observed >= threshold:
                events[key] = {
                    "time_sec": t_sec,
                    "step": record["step"],
                    "value": float(record["observed_axis_offset"]),
                    "latency_ms": (t_sec - step_time) * 1000.0,
                }
                break
    return events


def _run_probe(robot: Arm, side: str, start_state: dict[str, list[float]], args) -> dict[str, object]:
    axis_index = _axis_index(args.axis)
    total_sec = args.pre_sec + args.hold_sec + args.return_sec
    period = 1.0 / args.control_hz
    loops = max(1, int(math.ceil(total_sec * args.control_hz)))
    records = []
    timing = {
        "command": [],
        "state_read": [],
        "force_read": [],
        "loop_elapsed": [],
        "loop_interval": [],
    }
    missed = 0
    last_state = start_state
    previous_loop_start = None
    run_start = time.perf_counter()
    next_deadline = run_start
    motion_started = False
    max_joint_delta = 0.0
    max_pose_linear = 0.0
    max_pose_angular = 0.0
    max_force_norm = 0.0
    max_torque_norm = 0.0
    print(
        f"{side}: response probe axis={args.axis} offset={args.offset:+.5f} "
        f"hz={args.control_hz:.2f} state_every={args.state_read_every} "
        f"force_every={args.force_read_every} trajectory_mode={args.trajectory_mode} radio={args.radio}"
    )
    try:
        for index in range(loops):
            loop_start = time.perf_counter()
            if previous_loop_start is not None:
                timing["loop_interval"].append(loop_start - previous_loop_start)
            previous_loop_start = loop_start
            time_sec = loop_start - run_start
            command_axis_offset = _command_offset_at(time_sec, args)
            pose_cmd = _command_pose(start_state["pose"], axis_index, command_axis_offset)

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
            timing["command"].append(command_sec)
            motion_started = True

            state_sampled = index % args.state_read_every == 0 or index + 1 == loops
            state_read_sec = 0.0
            if state_sampled:
                state_start = time.perf_counter()
                last_state = _read_state(robot, side)
                state_read_sec = time.perf_counter() - state_start
                timing["state_read"].append(state_read_sec)
            state = last_state
            joint_delta, pose_linear, pose_angular = _check_state_limits(start_state, state, args)
            max_joint_delta = max(max_joint_delta, joint_delta)
            max_pose_linear = max(max_pose_linear, pose_linear)
            max_pose_angular = max(max_pose_angular, pose_angular)

            force_sampled = args.force_read_every > 0 and (index % args.force_read_every == 0 or index + 1 == loops)
            force_read_sec = 0.0
            force_norm = 0.0
            torque_norm = 0.0
            if force_sampled:
                force_start = time.perf_counter()
                force = _read_force(robot, side)
                force_read_sec = time.perf_counter() - force_start
                timing["force_read"].append(force_read_sec)
                force_norm, torque_norm = _check_force(force, args)
                max_force_norm = max(max_force_norm, force_norm)
                max_torque_norm = max(max_torque_norm, torque_norm)

            observed_axis_offset = _axis_delta(state["pose"], start_state["pose"], axis_index)
            elapsed = time.perf_counter() - loop_start
            missed_deadline = elapsed > period
            if missed_deadline:
                missed += 1
            timing["loop_elapsed"].append(elapsed)
            if index % args.print_every == 0 or index + 1 == loops:
                print(
                    f"{side}:resp step={index + 1}/{loops} t={time_sec:.3f}s "
                    f"cmd={command_axis_offset:+.5f} obs={observed_axis_offset:+.5f} "
                    f"pose_delta={pose_linear:.5f}m/{pose_angular:.5f}rad "
                    f"loop_ms={elapsed * 1000.0:.2f}"
                )
            records.append(
                {
                    "step": index + 1,
                    "time_sec": time_sec,
                    "command_pose": pose_cmd,
                    "command_axis_offset": command_axis_offset,
                    "observed_pose": state["pose"],
                    "observed_joint_deg": state["joint_deg"],
                    "observed_axis_offset": observed_axis_offset,
                    "state_sampled": state_sampled,
                    "force_sampled": force_sampled,
                    "joint_delta_deg": joint_delta,
                    "pose_linear_delta_m": pose_linear,
                    "pose_angular_delta_rad": pose_angular,
                    "force_norm_n": force_norm,
                    "torque_norm_nm": torque_norm,
                    "timing": {
                        "command_sec": command_sec,
                        "state_read_sec": state_read_sec,
                        "force_read_sec": force_read_sec,
                        "loop_elapsed_sec": elapsed,
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
        "max_joint_delta_deg": max_joint_delta,
        "max_pose_linear_delta_m": max_pose_linear,
        "max_pose_angular_delta_rad": max_pose_angular,
        "max_force_norm_n": max_force_norm,
        "max_torque_norm_nm": max_torque_norm,
        "response_events": _response_events(records, args.offset),
        "timing": {
            key: _stats_ms(values)
            for key, values in timing.items()
            if values
        },
    }
    print(
        f"{side}: response summary events={summary['response_events']} "
        f"loop_p99={summary['timing']['loop_elapsed']['p99_ms']:.3f}ms "
        f"cmd_p99={summary['timing']['command']['p99_ms']:.3f}ms"
    )
    return {
        "records": records,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe RM65 Movep_CANFD command-to-observed-pose response latency.")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="inspire")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for motion.")
    parser.add_argument("--axis", choices=AXES, default="z")
    parser.add_argument("--offset", type=float, default=0.010)
    parser.add_argument("--control-hz", type=float, default=40.0)
    parser.add_argument("--pre-sec", type=float, default=1.0)
    parser.add_argument("--hold-sec", type=float, default=1.5)
    parser.add_argument("--return-sec", type=float, default=1.0)
    parser.add_argument("--state-read-every", type=int, default=1)
    parser.add_argument("--force-read-every", type=int, default=0, help="0 disables force reads.")
    parser.add_argument("--trajectory-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--radio", type=int, default=0)
    parser.add_argument("--monitor-force-source", choices=("zero", "work", "tool"), default="work")
    parser.add_argument("--max-monitor-force-n", type=float, default=20.0)
    parser.add_argument("--max-monitor-torque-nm", type=float, default=2.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=15.0)
    parser.add_argument("--max-pose-linear-m", type=float, default=0.05)
    parser.add_argument("--max-pose-angular-rad", type=float, default=0.10)
    parser.add_argument("--max-command-offset-m", type=float, default=0.05)
    parser.add_argument("--max-command-offset-rad", type=float, default=0.05)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    parser.add_argument("--output-json", default="")
    return parser


def _validate_args(args) -> None:
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"response probe requires --confirm {CONFIRM_TEXT}")
    if args.control_hz <= 0.0 or args.control_hz > 40.0:
        raise RuntimeError("--control-hz must be in (0, 40]")
    if args.pre_sec < 0.0 or args.hold_sec <= 0.0 or args.return_sec < 0.0:
        raise RuntimeError("--pre-sec/--return-sec must be >=0 and --hold-sec must be positive")
    if args.state_read_every < 1:
        raise RuntimeError("--state-read-every must be >= 1")
    if args.force_read_every < 0:
        raise RuntimeError("--force-read-every must be >= 0")
    if args.print_every < 1:
        raise RuntimeError("--print-every must be >= 1")
    axis_index = _axis_index(args.axis)
    if axis_index < 3 and abs(args.offset) > args.max_command_offset_m:
        raise RuntimeError("--offset exceeds --max-command-offset-m")
    if axis_index >= 3 and abs(args.offset) > args.max_command_offset_rad:
        raise RuntimeError("--offset exceeds --max-command-offset-rad")
    for name in (
        "max_monitor_force_n",
        "max_monitor_torque_nm",
        "max_joint_delta_deg",
        "max_pose_linear_m",
        "max_pose_angular_rad",
        "max_command_offset_m",
        "max_command_offset_rad",
    ):
        if getattr(args, name) <= 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be positive")


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
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    side, ip = _selected_ip(args)
    robot = None
    try:
        robot = _connect_robot(side, ip)
        _check_expected_tool(robot, args.expected_tool)
        start_state = _read_state(robot, side)
        print(f"{side}: start_joint_deg={_format(start_state['joint_deg'])}")
        print(f"{side}: start_pose_xyz_rpy={_format(start_state['pose'])}")
        report: dict[str, object] = {
            "schema_version": 1,
            "generated_by": "hardware/rm65/utils/scripts/rm65_pose_response_probe.py",
            "arm": side,
            "ip": ip,
            "executed": bool(args.execute),
            "axis": args.axis,
            "offset": args.offset,
            "control_hz": args.control_hz,
            "pre_sec": args.pre_sec,
            "hold_sec": args.hold_sec,
            "return_sec": args.return_sec,
            "state_read_every": args.state_read_every,
            "force_read_every": args.force_read_every,
            "trajectory_mode": args.trajectory_mode,
            "radio": args.radio,
            "start_state": start_state,
        }
        if not args.execute:
            print("DRY-RUN: response probe commands will not be sent.")
            _write_report(args.output_json, report)
            return 0
        result = _run_probe(robot, side, start_state, args)
        report.update(result)
        final_state = _read_state(robot, side)
        report["final_state"] = final_state
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
