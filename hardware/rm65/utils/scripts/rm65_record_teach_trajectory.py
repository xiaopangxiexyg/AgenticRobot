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
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    current_name = _current_tool_name((code, frame))
    print(f"current_tool={current_name}")
    if expected_tool and expected_tool not in current_name:
        raise RuntimeError(f"expected active tool containing '{expected_tool}', got '{current_name}'")
    return current_name


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


def _angle_delta(current: float, previous: float) -> float:
    delta = current - previous
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _joint_velocity(current: list[float], previous: list[float] | None, dt: float) -> list[float]:
    if previous is None or dt <= 0.0:
        return [0.0] * 6
    return [(current[index] - previous[index]) / dt for index in range(6)]


def _pose_velocity(current: list[float], previous: list[float] | None, dt: float) -> list[float]:
    if previous is None or dt <= 0.0:
        return [0.0] * 6
    linear = [(current[index] - previous[index]) / dt for index in range(3)]
    angular = [_angle_delta(current[index], previous[index]) / dt for index in range(3, 6)]
    return linear + angular


def _pose_step(current: list[float], previous: list[float] | None) -> tuple[float, float]:
    if previous is None:
        return 0.0, 0.0
    linear = norm(current[index] - previous[index] for index in range(3))
    angular = norm(_angle_delta(current[index], previous[index]) for index in range(3, 6))
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


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else math.nan,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else math.nan,
    }


def _stats_ms(values_sec: list[float]) -> dict[str, float]:
    stats = _stats([value * 1000.0 for value in values_sec])
    return {f"{key}_ms": value for key, value in stats.items()}


def _range_by_axis(rows: list[list[float]]) -> list[float]:
    if not rows:
        return [0.0] * 6
    return [max(row[index] for row in rows) - min(row[index] for row in rows) for index in range(6)]


def _default_output_path(arm: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"hardware/rm65/utils/data/trajectories/rm65_{arm}_teach_trajectory_{stamp}.json"


def _write_json(path_text: str, payload: dict[str, object]) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote trajectory: {path}")


def _check_force_limits(force: dict[str, list[float]], args) -> None:
    wrench = force[args.monitor_force_source]
    force_norm = norm(wrench[:3])
    torque_norm = norm(wrench[3:])
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


def _summarize(samples: list[dict[str, object]], timing: dict[str, list[float]], missed: int) -> dict[str, object]:
    if not samples:
        return {
            "samples": 0,
            "missed_deadlines": missed,
        }
    duration = float(samples[-1]["t_sec"]) - float(samples[0]["t_sec"])
    actual_hz = (len(samples) - 1) / duration if duration > 0.0 and len(samples) > 1 else math.nan
    joint_rows = [sample["joint_deg"] for sample in samples]
    pose_rows = [sample["pose"] for sample in samples]
    linear_speeds = [float(sample["linear_speed_m_s"]) for sample in samples]
    angular_speeds = [float(sample["angular_speed_rad_s"]) for sample in samples]
    joint_speeds = [float(sample["max_joint_speed_deg_s"]) for sample in samples]
    path_length = sum(float(sample["linear_step_m"]) for sample in samples)
    angular_path = sum(float(sample["angular_step_rad"]) for sample in samples)
    return {
        "samples": len(samples),
        "duration_sec": duration,
        "actual_hz": actual_hz,
        "missed_deadlines": missed,
        "path_length_m": path_length,
        "angular_path_rad": angular_path,
        "joint_range_deg": _range_by_axis(joint_rows),
        "pose_range": _range_by_axis(pose_rows),
        "linear_speed_m_s": _stats(linear_speeds),
        "angular_speed_rad_s": _stats(angular_speeds),
        "max_joint_speed_deg_s": _stats(joint_speeds),
        "timing": {
            "loop_elapsed": _stats_ms(timing["loop_elapsed"]),
            "loop_interval": _stats_ms(timing["loop_interval"]),
            "state_read": _stats_ms(timing["state_read"]),
            "force_read": _stats_ms(timing["force_read"]),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only RM65 teaching trajectory recorder. It samples joint pose, "
            "TCP pose, finite-difference velocity, and optional force while the "
            "operator moves the robot externally."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="inspire")
    parser.add_argument("--hz", type=float, default=40.0)
    parser.add_argument("--duration-sec", type=float, default=20.0, help="Use 0 to record until Ctrl-C.")
    parser.add_argument("--start-delay-sec", type=float, default=3.0)
    parser.add_argument("--record-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-every", type=int, default=1, help="Read force every N samples when --record-force is enabled.")
    parser.add_argument("--monitor-force-source", choices=("zero", "work", "tool"), default="work")
    parser.add_argument("--max-monitor-force-n", type=float, default=100.0)
    parser.add_argument("--max-monitor-torque-nm", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--output-json", default="")
    return parser


def _validate_args(args) -> None:
    if args.hz <= 0.0:
        raise RuntimeError("--hz must be positive")
    if args.duration_sec < 0.0:
        raise RuntimeError("--duration-sec must be >= 0")
    if args.start_delay_sec < 0.0:
        raise RuntimeError("--start-delay-sec must be >= 0")
    if args.force_every < 1:
        raise RuntimeError("--force-every must be >= 1")
    if args.print_every < 1:
        raise RuntimeError("--print-every must be >= 1")
    if args.max_monitor_force_n <= 0.0 or args.max_monitor_torque_nm <= 0.0:
        raise RuntimeError("--max-monitor-force-n and --max-monitor-torque-nm must be positive")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    output_json = args.output_json or _default_output_path(side)
    robot = None
    samples: list[dict[str, object]] = []
    timing = {
        "loop_elapsed": [],
        "loop_interval": [],
        "state_read": [],
        "force_read": [],
    }
    missed = 0
    interrupted = False
    error_text = ""
    period = 1.0 / args.hz
    max_samples = None if args.duration_sec == 0.0 else max(1, int(round(args.duration_sec * args.hz)))
    current_tool = ""
    start_wall = time.time()
    try:
        robot = _connect_robot(side, ip)
        current_tool = _check_expected_tool(robot, args.expected_tool)
        print("READ-ONLY: this script does not command robot motion, drag mode, or force clearing.")
        if args.start_delay_sec > 0.0:
            print(f"{side}: recording starts in {args.start_delay_sec:.1f}s")
            time.sleep(args.start_delay_sec)
        print(
            f"{side}: recording trajectory hz={args.hz:.2f} duration="
            f"{'until Ctrl-C' if max_samples is None else f'{args.duration_sec:.2f}s'} "
            f"force={'on' if args.record_force else 'off'} output={output_json}"
        )

        start_perf = time.perf_counter()
        previous_loop_start = start_perf
        previous_joint = None
        previous_pose = None
        next_deadline = start_perf
        index = 0
        while max_samples is None or index < max_samples:
            loop_start = time.perf_counter()
            if index > 0:
                timing["loop_interval"].append(loop_start - previous_loop_start)
            dt = loop_start - previous_loop_start if index > 0 else period
            previous_loop_start = loop_start
            t_sec = loop_start - start_perf

            state_start = time.perf_counter()
            state = _read_state(robot, side)
            state_read_sec = time.perf_counter() - state_start
            timing["state_read"].append(state_read_sec)

            force = None
            force_read_sec = 0.0
            force_sampled = args.record_force and index % args.force_every == 0
            if force_sampled:
                force_start = time.perf_counter()
                force = _read_force(robot, side)
                force_read_sec = time.perf_counter() - force_start
                timing["force_read"].append(force_read_sec)
                _check_force_limits(force, args)

            joint_vel = _joint_velocity(state["joint_deg"], previous_joint, dt)
            pose_vel = _pose_velocity(state["pose"], previous_pose, dt)
            linear_step, angular_step = _pose_step(state["pose"], previous_pose)
            previous_joint = state["joint_deg"]
            previous_pose = state["pose"]
            sample = {
                "index": index + 1,
                "t_sec": t_sec,
                "wall_time": start_wall + t_sec,
                "dt_sec": dt,
                "joint_deg": state["joint_deg"],
                "pose": state["pose"],
                "joint_velocity_deg_s": joint_vel,
                "pose_velocity": pose_vel,
                "linear_velocity_m_s": pose_vel[:3],
                "angular_velocity_rad_s": pose_vel[3:],
                "linear_speed_m_s": norm(pose_vel[:3]),
                "angular_speed_rad_s": norm(pose_vel[3:]),
                "max_joint_speed_deg_s": max(abs(value) for value in joint_vel),
                "linear_step_m": linear_step,
                "angular_step_rad": angular_step,
                "force_sampled": force_sampled,
                "timing": {
                    "state_read_sec": state_read_sec,
                    "force_read_sec": force_read_sec,
                },
            }
            if force is not None:
                sample["force"] = force
            samples.append(sample)

            loop_elapsed = time.perf_counter() - loop_start
            timing["loop_elapsed"].append(loop_elapsed)
            if loop_elapsed > period:
                missed += 1
            if index % args.print_every == 0:
                print(
                    f"{side}:rec step={index + 1} t={t_sec:.3f}s "
                    f"pose={_format(state['pose'], precision=4)} "
                    f"|v|={sample['linear_speed_m_s']:.4f}m/s "
                    f"|w|={sample['angular_speed_rad_s']:.4f}rad/s "
                    f"max_joint_v={sample['max_joint_speed_deg_s']:.2f}deg/s "
                    f"loop_ms={loop_elapsed * 1000.0:.2f}"
                )
            index += 1
            next_deadline += period
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
    except KeyboardInterrupt:
        interrupted = True
        print("\nrecording interrupted by Ctrl-C; writing collected samples.")
    except RuntimeError as exc:
        error_text = str(exc)
        print(f"ERROR: {error_text}", file=sys.stderr)
    finally:
        if robot is not None:
            robot.Arm_Socket_Close()

    payload = {
        "schema_version": 1,
        "data_type": "rm65_teach_trajectory",
        "generated_by": "hardware/rm65/utils/scripts/rm65_record_teach_trajectory.py",
        "arm": side,
        "ip": ip,
        "active_tool": current_tool,
        "expected_tool": args.expected_tool,
        "recorded_at": start_wall,
        "read_only": True,
        "interrupted": interrupted,
        "error": error_text,
        "config": {
            "hz": args.hz,
            "duration_sec": args.duration_sec,
            "start_delay_sec": args.start_delay_sec,
            "record_force": args.record_force,
            "force_every": args.force_every,
            "monitor_force_source": args.monitor_force_source,
            "max_monitor_force_n": args.max_monitor_force_n,
            "max_monitor_torque_nm": args.max_monitor_torque_nm,
        },
        "summary": _summarize(samples, timing, missed),
        "samples": samples,
    }
    _write_json(output_json, payload)
    if error_text:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
