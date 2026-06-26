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

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65


CONFIRM_TEXT = "TEST_RM65_FREQUENCY"
READ_MODES = ("state", "force", "state-force")
POSE_MODES = ("pose-hold", "pose-hold-state-force")
ALL_MODES = READ_MODES + POSE_MODES


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _parse_hz_list(text: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise RuntimeError(f"invalid --hz-list '{text}'") from exc
    if not values:
        raise RuntimeError("--hz-list must contain at least one value")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise RuntimeError("--hz-list values must be positive finite numbers")
    return values


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


def _connect_robot(side: str, ip: str) -> Arm:
    robot = Arm(RM65, ip)
    if int(robot.Arm_Socket_State()) != 0:
        robot.Arm_Socket_Close()
        raise RuntimeError(f"{side}: failed to connect ip={ip}")
    print(f"{side}: connected ip={ip} api={robot.API_Version()}")
    return robot


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


def _read_state(robot: Arm, side: str) -> dict[str, object]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")
    return {
        "joint_deg": [float(value) for value in joint_deg],
        "pose": [float(value) for value in pose],
    }


def _read_force(robot: Arm, side: str, args) -> dict[str, list[float]]:
    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    readings = {
        "raw": [float(value) for value in raw],
        "zero": [float(value) for value in zero],
        "work": [float(value) for value in work],
        "tool": [float(value) for value in tool],
    }
    source = readings[args.monitor_force_source]
    force_norm = _norm(source[:3])
    torque_norm = _norm(source[3:])
    if force_norm > args.max_monitor_force_n:
        raise RuntimeError(
            f"{args.monitor_force_source} force {force_norm:.4f}N > "
            f"--max-monitor-force-n {args.max_monitor_force_n:.4f}N"
        )
    if torque_norm > args.max_monitor_torque_nm:
        raise RuntimeError(
            f"{args.monitor_force_source} torque {torque_norm:.4f}Nm > "
            f"--max-monitor-torque-nm {args.max_monitor_torque_nm:.4f}Nm"
        )
    return readings


def _joint_delta_deg(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _pose_delta(start_pose: list[float], current_pose: list[float]) -> tuple[float, float]:
    linear = _norm(current_pose[i] - start_pose[i] for i in range(3))
    angular = _norm(current_pose[i] - start_pose[i] for i in range(3, 6))
    return linear, angular


def _run_one_rate(robot: Arm, side: str, hz: float, start_state: dict[str, object], args) -> dict[str, object]:
    period = 1.0 / hz
    loops = max(1, int(round(args.duration_sec * hz)))
    loop_durations = []
    intervals = []
    missed = 0
    command_count = 0
    state_count = 0
    force_count = 0
    previous_loop_start = None
    start_perf = time.perf_counter()
    next_deadline = start_perf
    max_joint_delta = 0.0
    max_pose_linear = 0.0
    max_pose_angular = 0.0
    max_force_norm = 0.0
    max_torque_norm = 0.0

    for index in range(loops):
        loop_start = time.perf_counter()
        if previous_loop_start is not None:
            intervals.append(loop_start - previous_loop_start)
        previous_loop_start = loop_start

        state = None
        force = None
        if args.mode in POSE_MODES:
            _require_zero(
                robot.Movep_CANFD(
                    start_state["pose"],
                    follow=False,
                    trajectory_mode=args.trajectory_mode,
                    radio=args.radio,
                ),
                f"{side} Movep_CANFD",
            )
            command_count += 1
        if args.mode in ("state", "state-force", "pose-hold-state-force"):
            state = _read_state(robot, side)
            state_count += 1
            joint_delta = _joint_delta_deg(start_state["joint_deg"], state["joint_deg"])
            pose_linear, pose_angular = _pose_delta(start_state["pose"], state["pose"])
            max_joint_delta = max(max_joint_delta, joint_delta)
            max_pose_linear = max(max_pose_linear, pose_linear)
            max_pose_angular = max(max_pose_angular, pose_angular)
            if joint_delta > args.max_joint_delta_deg:
                raise RuntimeError(
                    f"joint delta {joint_delta:.4f}deg > "
                    f"--max-joint-delta-deg {args.max_joint_delta_deg:.4f}deg"
                )
            if pose_linear > args.max_pose_linear_m:
                raise RuntimeError(
                    f"pose linear delta {pose_linear:.5f}m > "
                    f"--max-pose-linear-m {args.max_pose_linear_m:.5f}m"
                )
            if pose_angular > args.max_pose_angular_rad:
                raise RuntimeError(
                    f"pose angular delta {pose_angular:.5f}rad > "
                    f"--max-pose-angular-rad {args.max_pose_angular_rad:.5f}rad"
                )
        if args.mode in ("force", "state-force", "pose-hold-state-force"):
            force = _read_force(robot, side, args)
            force_count += 1
            source = force[args.monitor_force_source]
            max_force_norm = max(max_force_norm, _norm(source[:3]))
            max_torque_norm = max(max_torque_norm, _norm(source[3:]))

        elapsed = time.perf_counter() - loop_start
        loop_durations.append(elapsed)
        if elapsed > period:
            missed += 1
        next_deadline += period
        sleep_sec = next_deadline - time.perf_counter()
        if sleep_sec > 0.0:
            time.sleep(sleep_sec)

    total_sec = time.perf_counter() - start_perf
    loop_stats = _stats_ms(loop_durations)
    interval_stats = _stats_ms(intervals)
    missed_fraction = missed / loops
    period_ms = period * 1000.0
    actual_hz = loops / total_sec if total_sec > 0 else math.nan
    p99_ratio = loop_stats["p99_ms"] / period_ms if period_ms > 0 else math.inf
    passed = (
        missed_fraction <= args.max_missed_fraction
        and p99_ratio <= args.max_p99_period_ratio
    )
    report = {
        "hz": hz,
        "mode": args.mode,
        "period_ms": period_ms,
        "loops": loops,
        "total_sec": total_sec,
        "actual_hz": actual_hz,
        "command_count": command_count,
        "state_count": state_count,
        "force_count": force_count,
        "missed_deadlines": missed,
        "missed_fraction": missed_fraction,
        "loop_duration": loop_stats,
        "loop_interval": interval_stats,
        "p99_period_ratio": p99_ratio,
        "max_joint_delta_deg": max_joint_delta,
        "max_pose_linear_delta_m": max_pose_linear,
        "max_pose_angular_delta_rad": max_pose_angular,
        "max_force_norm_n": max_force_norm,
        "max_torque_norm_nm": max_torque_norm,
        "passed": passed,
    }
    print(
        f"{side}: hz={hz:.1f} mode={args.mode} actual={actual_hz:.2f}Hz "
        f"loop_ms mean/p95/p99/max="
        f"{loop_stats['mean_ms']:.3f}/{loop_stats['p95_ms']:.3f}/"
        f"{loop_stats['p99_ms']:.3f}/{loop_stats['max_ms']:.3f} "
        f"missed={missed}/{loops} p99_ratio={p99_ratio:.3f} "
        f"passed={passed}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RM65 communication frequency probe. It can sweep read-only state/force "
            "loops or, with explicit confirmation, send current-pose Movep_CANFD hold "
            "commands while reading state/force."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="")
    parser.add_argument("--mode", choices=ALL_MODES, default="state-force")
    parser.add_argument("--hz-list", default="20,33,40,50,75,100")
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--settle-sec", type=float, default=0.5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for pose-hold modes.")
    parser.add_argument("--trajectory-mode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--radio", type=int, default=50)
    parser.add_argument("--monitor-force-source", choices=("zero", "work", "tool"), default="tool")
    parser.add_argument("--max-monitor-force-n", type=float, default=8.0)
    parser.add_argument("--max-monitor-torque-nm", type=float, default=0.50)
    parser.add_argument("--max-joint-delta-deg", type=float, default=3.0)
    parser.add_argument("--max-pose-linear-m", type=float, default=0.010)
    parser.add_argument("--max-pose-angular-rad", type=float, default=0.05)
    parser.add_argument("--max-missed-fraction", type=float, default=0.05)
    parser.add_argument("--max-p99-period-ratio", type=float, default=1.0)
    parser.add_argument("--fail-on-threshold", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser


def _validate_args(args) -> None:
    args.hz_values = _parse_hz_list(args.hz_list)
    if args.duration_sec <= 0.0:
        raise RuntimeError("--duration-sec must be positive")
    if args.settle_sec < 0.0:
        raise RuntimeError("--settle-sec must be >= 0")
    if args.mode in POSE_MODES and (not args.execute or args.confirm != CONFIRM_TEXT):
        raise RuntimeError(f"{args.mode} requires --execute --confirm {CONFIRM_TEXT}")
    if args.mode in READ_MODES and args.execute:
        raise RuntimeError("--execute is only meaningful for pose-hold modes")
    for name in (
        "max_monitor_force_n",
        "max_monitor_torque_nm",
        "max_joint_delta_deg",
        "max_pose_linear_m",
        "max_pose_angular_rad",
        "max_p99_period_ratio",
    ):
        if getattr(args, name) <= 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be positive")
    if not (0.0 <= args.max_missed_fraction <= 1.0):
        raise RuntimeError("--max-missed-fraction must be in [0, 1]")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    robot = None
    reports = []
    try:
        robot = _connect_robot(side, ip)
        _check_expected_tool(robot, args.expected_tool)
        start_state = _read_state(robot, side)
        print(f"{side}: start_joint_deg={_format(start_state['joint_deg'])}")
        print(f"{side}: start_pose_xyz_rpy={_format(start_state['pose'])}")
        print(
            f"{side}: frequency sweep mode={args.mode} hz={args.hz_values} "
            f"duration={args.duration_sec:.2f}s"
        )
        if args.mode in POSE_MODES:
            print("POSE HOLD: sending the startup pose every cycle; no trajectory offset is commanded.")
        for hz in args.hz_values:
            if reports and args.settle_sec > 0.0:
                time.sleep(args.settle_sec)
            reports.append(_run_one_rate(robot, side, hz, start_state, args))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            if args.mode in POSE_MODES:
                try:
                    code = robot.Arm_Slow_Stop()
                    print(f"{side}: Arm_Slow_Stop -> {code}")
                except Exception as exc:  # noqa: BLE001
                    print(f"{side}: Arm_Slow_Stop failed: {exc}")
            robot.Arm_Socket_Close()

    all_passed = all(item["passed"] for item in reports)
    print(f"\nRESULT: frequency probe all_passed={all_passed}")
    for item in reports:
        print(
            f"  {item['hz']:.1f}Hz: actual={item['actual_hz']:.2f}Hz "
            f"p99={item['loop_duration']['p99_ms']:.3f}ms "
            f"missed={item['missed_deadlines']}/{item['loops']} passed={item['passed']}"
        )

    payload = {
        "schema_version": 1,
        "generated_by": "hardware/rm65/utils/scripts/rm65_comm_frequency_probe.py",
        "arm": side,
        "ip": ip,
        "mode": args.mode,
        "duration_sec": args.duration_sec,
        "hz_values": args.hz_values,
        "thresholds": {
            "max_missed_fraction": args.max_missed_fraction,
            "max_p99_period_ratio": args.max_p99_period_ratio,
        },
        "reports": reports,
        "all_passed": all_passed,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {output_path}")

    return 0 if all_passed or not args.fail_on_threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
