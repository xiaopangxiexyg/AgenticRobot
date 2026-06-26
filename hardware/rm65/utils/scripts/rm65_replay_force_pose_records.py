from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent
DATA_DIR = UTILS_DIR / "data"
POSE_DIR = DATA_DIR / "poses"
MEASUREMENTS_DIR = DATA_DIR / "measurements"
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65
from hardware.rm65.standard_paths import (
    force_sensor_calib4_measurements_path,
    force_sensor_calib4_pose_sequence_path,
)


CONFIRM_TEXT = "REPLAY_RM65_FORCE_POSES"
FORCE_SOURCES = ("raw", "zero", "work", "tool")


class ReplayInterrupted(RuntimeError):
    pass


def _default_input_path(side: str) -> Path:
    return POSE_DIR / f"rm65_{side}_pose_sequence.json"


def _default_output_path(side: str, label: str) -> Path:
    suffix = f"_{label}" if label else "_unlabeled_tool"
    return MEASUREMENTS_DIR / f"rm65_{side}{suffix}_force_measurements.json"


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _finite_float_list(values: Iterable[float], *, expected_len: int) -> list[float]:
    out = [float(value) for value in values]
    if len(out) != expected_len:
        raise RuntimeError(f"expected {expected_len} values, got {len(out)}")
    if not all(math.isfinite(value) for value in out):
        raise RuntimeError(f"non-finite values: {out}")
    return out


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    return [statistics.fmean(vector[index] for vector in vectors) for index in range(len(vectors[0]))]


def _std_vector(vectors: list[list[float]]) -> list[float]:
    if len(vectors) <= 1:
        return [0.0] * len(vectors[0])
    return [statistics.pstdev(vector[index] for vector in vectors) for index in range(len(vectors[0]))]


def _range_vector(vectors: list[list[float]]) -> list[float]:
    return [
        max(vector[index] for vector in vectors) - min(vector[index] for vector in vectors)
        for index in range(len(vectors[0]))
    ]


def _joint_distance_deg(a: list[float], b: list[float]) -> dict[str, float]:
    deltas = [abs(x - y) for x, y in zip(a, b)]
    return {
        "max_abs": max(deltas),
        "rms": math.sqrt(statistics.fmean(delta * delta for delta in deltas)),
        "per_joint": deltas,
    }


def _interpolation_segments(
    start: list[float],
    target: list[float],
    max_joint_step_deg: float,
) -> int:
    distance = _joint_distance_deg(start, target)
    return max(1, math.ceil(distance["max_abs"] / max_joint_step_deg))


def _interpolated_waypoints(
    start: list[float],
    target: list[float],
    max_joint_step_deg: float,
) -> list[list[float]]:
    segments = _interpolation_segments(start, target, max_joint_step_deg)
    waypoints = []
    for segment_index in range(1, segments + 1):
        alpha = segment_index / segments
        waypoints.append(
            [
                start[joint_index]
                + (target[joint_index] - start[joint_index]) * alpha
                for joint_index in range(6)
            ]
        )
    return waypoints


def _waypoint_plan_summary(
    start: list[float],
    target: list[float],
    args,
) -> dict[str, object]:
    if args.interpolate_jumps:
        waypoints = _interpolated_waypoints(
            start,
            target,
            args.interp_max_joint_step_deg,
        )
    else:
        waypoints = [target]
    last = start
    max_segment_delta = 0.0
    for waypoint in waypoints:
        max_segment_delta = max(
            max_segment_delta,
            _joint_distance_deg(last, waypoint)["max_abs"],
        )
        last = waypoint
    return {
        "segments": len(waypoints),
        "intermediate_waypoints": max(0, len(waypoints) - 1),
        "max_segment_joint_delta_deg": max_segment_delta,
    }


def _interp_movej_speed(args) -> int:
    if args.interp_movej_speed > 0:
        return args.interp_movej_speed
    return int(math.ceil(args.movej_speed * args.interp_speed_multiplier))


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _load_targets(path: Path, side: str) -> tuple[dict[str, object], list[list[float]]]:
    if not path.exists():
        raise RuntimeError(
            f"missing pose sequence file: {path}. "
            "Record this arm first with: "
            "conda run --no-capture-output -n default python "
            "hardware/rm65/utils/scripts/rm65_auto_record_force_poses.py "
            f"--arm {side}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("arm") not in (None, side):
        raise RuntimeError(f"{path} belongs to arm={payload.get('arm')}, not {side}")
    targets = payload.get("joint_targets_deg")
    if not targets:
        if "poses" in payload:
            targets = [pose["joint_deg"] for pose in payload.get("poses", [])]
        else:
            records = payload.get("records", [])
            targets = [record["joint_deg_mean"] for record in records]
    if not isinstance(targets, list) or not targets:
        raise RuntimeError(f"{path} has no joint targets")
    checked_targets = [
        _finite_float_list(target, expected_len=6)
        for target in targets
    ]
    return payload, checked_targets


def _read_once(robot: Arm, side: str) -> dict[str, object]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")

    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    return {
        "timestamp": time.time(),
        "joint_deg": _finite_float_list(joint_deg, expected_len=6),
        "pose": _finite_float_list(pose, expected_len=6),
        "force": {
            "raw": _finite_float_list(raw, expected_len=6),
            "zero": _finite_float_list(zero, expected_len=6),
            "work": _finite_float_list(work, expected_len=6),
            "tool": _finite_float_list(tool, expected_len=6),
        },
    }


def _read_joint_deg(robot: Arm, side: str) -> list[float]:
    code, joint_deg, _pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")
    return _finite_float_list(joint_deg, expected_len=6)


def _stop_motion(robot: Arm | None, side: str, args, reason: str) -> None:
    if robot is None:
        return
    try:
        if args.interrupt_stop_mode == "fast":
            code = robot.Arm_Stop()
            print(f"{side}: Arm_Stop({reason}) -> {code}", file=sys.stderr)
        else:
            code = robot.Arm_Slow_Stop()
            print(f"{side}: Arm_Slow_Stop({reason}) -> {code}", file=sys.stderr)
    except Exception as exc:
        print(f"{side}: stop failed during {reason}: {exc}", file=sys.stderr)


def _wait_for_joint_target(
    robot: Arm,
    side: str,
    target: list[float],
    args,
    label: str,
    should_stop,
) -> list[float]:
    deadline = time.monotonic() + args.move_timeout_sec
    while True:
        if should_stop():
            raise ReplayInterrupted(f"{label} interrupted")
        joint_deg = _read_joint_deg(robot, side)
        distance = _joint_distance_deg(joint_deg, target)
        if distance["max_abs"] <= args.move_reached_tolerance_deg:
            return joint_deg
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"{label} did not reach target within {args.move_timeout_sec:.1f}s; "
                f"current max error={distance['max_abs']:.3f}deg"
            )
        time.sleep(args.motion_poll_sec)


def _summarize_samples(readings: list[dict[str, object]]) -> dict[str, object]:
    joints = [reading["joint_deg"] for reading in readings]
    poses = [reading["pose"] for reading in readings]
    force_stats = {}
    for source in FORCE_SOURCES:
        vectors = [reading["force"][source] for reading in readings]
        mean = _mean_vector(vectors)
        std = _std_vector(vectors)
        ranges = _range_vector(vectors)
        force_stats[source] = {
            "mean": mean,
            "std": std,
            "range": ranges,
            "force_norm": _norm(mean[:3]),
            "torque_norm": _norm(mean[3:]),
            "max_force_std": max(std[:3]),
            "max_torque_std": max(std[3:]),
            "max_force_range": max(ranges[:3]),
            "max_torque_range": max(ranges[3:]),
        }
    return {
        "samples": len(readings),
        "duration_sec": readings[-1]["timestamp"] - readings[0]["timestamp"],
        "joint_deg_mean": _mean_vector(joints),
        "joint_deg_std": _std_vector(joints),
        "joint_deg_range": _range_vector(joints),
        "pose_mean": _mean_vector(poses),
        "pose_range": _range_vector(poses),
        "force": force_stats,
    }


def _sample_at_current_pose(robot: Arm, side: str, args) -> dict[str, object]:
    readings = []
    for sample_index in range(args.samples):
        readings.append(_read_once(robot, side))
        if sample_index + 1 < args.samples:
            time.sleep(args.interval_sec)
    return _summarize_samples(readings)


def _validate_targets(targets: list[list[float]], args) -> tuple[list[str], list[str]]:
    warnings = []
    errors = []
    for index in range(1, len(targets)):
        distance = _joint_distance_deg(targets[index - 1], targets[index])
        plan = _waypoint_plan_summary(targets[index - 1], targets[index], args)
        if distance["max_abs"] > args.warn_adjacent_joint_delta_deg:
            if args.interpolate_jumps:
                warnings.append(
                    f"target {index}->{index + 1} max joint delta "
                    f"{distance['max_abs']:.3f}deg exceeds warning threshold "
                    f"{args.warn_adjacent_joint_delta_deg:.3f}deg; "
                    f"will use {plan['segments']} MoveJ segments "
                    f"({plan['intermediate_waypoints']} interpolated waypoints)"
                )
            else:
                warnings.append(
                    f"target {index}->{index + 1} max joint delta "
                    f"{distance['max_abs']:.3f}deg exceeds "
                    f"warning threshold {args.warn_adjacent_joint_delta_deg:.3f}deg"
                )
        if distance["max_abs"] > args.max_adjacent_joint_delta_deg and not args.interpolate_jumps:
            errors.append(
                f"target {index}->{index + 1} max joint delta "
                f"{distance['max_abs']:.3f}deg exceeds safety limit "
                f"{args.max_adjacent_joint_delta_deg:.3f}deg"
            )
        if args.interpolate_jumps and plan["segments"] > args.max_interpolation_segments:
            errors.append(
                f"target {index}->{index + 1} requires {plan['segments']} "
                f"interpolation segments, exceeding --max-interpolation-segments "
                f"{args.max_interpolation_segments}"
            )
    return warnings, errors


def _print_targets(targets: list[list[float]], args) -> None:
    print(f"loaded {len(targets)} targets")
    for index, target in enumerate(targets, start=1):
        if index == 1:
            print(f"  #{index:02d}: {_format(target, precision=3)}")
            continue
        distance = _joint_distance_deg(targets[index - 2], target)
        plan = _waypoint_plan_summary(targets[index - 2], target, args)
        print(
            f"  #{index:02d}: {_format(target, precision=3)} "
            f"adjacent_max_delta={distance['max_abs']:.3f}deg "
            f"move_segments={plan['segments']}"
        )


def _check_start_path(
    current_joint_deg: list[float],
    first_target: list[float],
    args,
) -> dict[str, object]:
    start_distance = _joint_distance_deg(current_joint_deg, first_target)
    start_plan = _waypoint_plan_summary(current_joint_deg, first_target, args)
    print(f"current_joint_deg={_format(current_joint_deg, precision=3)}")
    print(
        f"distance_to_first_target max={start_distance['max_abs']:.3f}deg "
        f"rms={start_distance['rms']:.3f}deg"
    )
    print(
        f"start_motion segments={start_plan['segments']} "
        f"intermediate={start_plan['intermediate_waypoints']} "
        f"max_segment_delta={start_plan['max_segment_joint_delta_deg']:.3f}deg"
    )
    if start_distance["max_abs"] > args.max_start_joint_delta_deg:
        if args.interpolate_jumps:
            print(
                "WARNING: current pose is far from the first target; "
                f"will interpolate the start path using {start_plan['segments']} "
                "MoveJ segments"
            )
        elif not args.allow_far_start:
            raise RuntimeError(
                "current pose is far from first target; move near the first recorded pose "
                "or add --allow-far-start after checking the path"
            )
    if args.interpolate_jumps and start_plan["segments"] > args.max_interpolation_segments:
        raise RuntimeError(
            "current pose to first target requires "
            f"{start_plan['segments']} interpolation segments, exceeding "
            f"--max-interpolation-segments {args.max_interpolation_segments}"
        )
    return {
        "distance_to_first_target_deg": start_distance,
        "motion_plan": start_plan,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely replay recorded RM65 joint poses in sequence and record force "
            "statistics at each point. Defaults to dry-run; real motion requires "
            f"--execute --confirm {CONFIRM_TEXT}."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--input-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--tool-label", default="")
    parser.add_argument("--expected-tool", default="", help="Alias for --tool-label in standard calibration commands.")
    parser.add_argument("--calib4", action="store_true", help="Use standard force_sensor_calib4 input/output paths.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--check-current-start",
        action="store_true",
        help="In dry-run mode, connect read-only and print the current-to-first-target plan.",
    )
    parser.add_argument("--movej-speed", type=int, default=2)
    parser.add_argument("--interp-speed-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--interp-movej-speed",
        type=int,
        default=0,
        help="Absolute speed for intermediate waypoints. 0 means movej-speed * multiplier.",
    )
    parser.add_argument("--max-movej-speed", type=int, default=10)
    parser.add_argument("--settle-sec", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--interval-sec", type=float, default=0.05)
    parser.add_argument("--max-start-joint-delta-deg", type=float, default=8.0)
    parser.add_argument("--allow-far-start", action="store_true")
    parser.add_argument("--warn-adjacent-joint-delta-deg", type=float, default=45.0)
    parser.add_argument("--max-adjacent-joint-delta-deg", type=float, default=65.0)
    parser.add_argument(
        "--interpolate-jumps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert intermediate joint-space MoveJ waypoints between recorded poses.",
    )
    parser.add_argument("--interp-max-joint-step-deg", type=float, default=10.0)
    parser.add_argument("--max-interpolation-segments", type=int, default=50)
    parser.add_argument(
        "--interp-trajectory-connect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Queue intermediate waypoints with trajectory_connect=1 so they do not stop.",
    )
    parser.add_argument(
        "--interp-blend-r",
        type=int,
        default=20,
        help="MoveJ blend radius percentage for intermediate waypoints when connected.",
    )
    parser.add_argument(
        "--interrupt-stop-mode",
        choices=("slow", "fast"),
        default="slow",
        help="Motion stop command sent on Ctrl-C/SIGTERM or motion errors.",
    )
    parser.add_argument(
        "--motion-poll-sec",
        type=float,
        default=0.05,
        help="Polling interval while waiting for non-blocking MoveJ completion.",
    )
    parser.add_argument(
        "--move-timeout-sec",
        type=float,
        default=120.0,
        help="Maximum wait for each MoveJ waypoint or connected waypoint group.",
    )
    parser.add_argument(
        "--move-reached-tolerance-deg",
        type=float,
        default=1.0,
        help="Joint-space tolerance used to consider a non-blocking MoveJ complete.",
    )
    parser.add_argument("--max-tracking-error-deg", type=float, default=1.0)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=0)
    return parser


def _validate_args(args) -> None:
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"real replay requires --confirm {CONFIRM_TEXT}")
    if args.movej_speed < 1:
        raise RuntimeError("--movej-speed must be >= 1")
    if args.movej_speed > args.max_movej_speed:
        raise RuntimeError("--movej-speed exceeds --max-movej-speed")
    if args.interp_speed_multiplier <= 0.0:
        raise RuntimeError("--interp-speed-multiplier must be positive")
    if args.interp_movej_speed < 0:
        raise RuntimeError("--interp-movej-speed must be >= 0")
    if _interp_movej_speed(args) < 1:
        raise RuntimeError("computed interpolation MoveJ speed must be >= 1")
    if _interp_movej_speed(args) > args.max_movej_speed:
        raise RuntimeError(
            "computed interpolation MoveJ speed exceeds --max-movej-speed; "
            "lower --interp-speed-multiplier/--interp-movej-speed or raise the cap"
        )
    if args.max_movej_speed > 10:
        raise RuntimeError("--max-movej-speed must be <= 10")
    if args.settle_sec < 0.0:
        raise RuntimeError("--settle-sec must be >= 0")
    if args.samples < 1:
        raise RuntimeError("--samples must be >= 1")
    if args.interval_sec < 0.0:
        raise RuntimeError("--interval-sec must be >= 0")
    if args.max_start_joint_delta_deg <= 0.0:
        raise RuntimeError("--max-start-joint-delta-deg must be positive")
    if args.warn_adjacent_joint_delta_deg <= 0.0:
        raise RuntimeError("--warn-adjacent-joint-delta-deg must be positive")
    if args.max_adjacent_joint_delta_deg <= 0.0:
        raise RuntimeError("--max-adjacent-joint-delta-deg must be positive")
    if args.warn_adjacent_joint_delta_deg > args.max_adjacent_joint_delta_deg:
        raise RuntimeError("--warn-adjacent-joint-delta-deg must be <= max threshold")
    if args.interp_max_joint_step_deg <= 0.0:
        raise RuntimeError("--interp-max-joint-step-deg must be positive")
    if args.max_interpolation_segments < 1:
        raise RuntimeError("--max-interpolation-segments must be >= 1")
    if args.interp_blend_r < 0 or args.interp_blend_r > 100:
        raise RuntimeError("--interp-blend-r must be in [0, 100]")
    if args.motion_poll_sec <= 0.0:
        raise RuntimeError("--motion-poll-sec must be positive")
    if args.move_timeout_sec <= 0.0:
        raise RuntimeError("--move-timeout-sec must be positive")
    if args.move_reached_tolerance_deg <= 0.0:
        raise RuntimeError("--move-reached-tolerance-deg must be positive")
    if args.max_tracking_error_deg <= 0.0:
        raise RuntimeError("--max-tracking-error-deg must be positive")
    if args.start_index < 1:
        raise RuntimeError("--start-index must be >= 1")
    if args.end_index < 0:
        raise RuntimeError("--end-index must be >= 0")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    tool_label = args.tool_label or args.expected_tool
    input_path = (
        Path(args.input_json)
        if args.input_json
        else force_sensor_calib4_pose_sequence_path(side)
        if args.calib4
        else _default_input_path(side)
    )
    output_path = (
        Path(args.output_json)
        if args.output_json
        else force_sensor_calib4_measurements_path(side, tool_label)
        if args.calib4 and tool_label
        else _default_output_path(side, tool_label)
    )
    try:
        input_payload, targets = _load_targets(input_path, side)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    end_index = args.end_index if args.end_index else len(targets)
    targets = targets[args.start_index - 1 : end_index]
    if not targets:
        print("ERROR: selected target range is empty", file=sys.stderr)
        return 2

    print(f"input_json={input_path}")
    print(f"output_json={output_path}")
    print(f"execute={args.execute} arm={side} ip={ip}")
    print(
        f"interpolate_jumps={args.interpolate_jumps} "
        f"interp_max_joint_step_deg={args.interp_max_joint_step_deg:.3f} "
        f"movej_speed={args.movej_speed} "
        f"interp_movej_speed={_interp_movej_speed(args)} "
        f"interp_connect={args.interp_trajectory_connect} "
        f"interp_blend_r={args.interp_blend_r}"
    )
    _print_targets(targets, args)
    warnings, target_errors = _validate_targets(targets, args)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in target_errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if target_errors:
        return 2
    if not args.execute:
        if args.check_current_start:
            robot = None
            try:
                robot = Arm(RM65, ip)
                current = _read_once(robot, side)
                _check_start_path(current["joint_deg"], targets[0], args)
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            finally:
                if robot is not None:
                    robot.Arm_Socket_Close()
        print("DRY-RUN: no robot motion or force sampling will be executed.")
        return 0

    robot = None
    replay_records = []
    start_path = None
    stop_requested = False
    stop_sent = False

    def _stop_once(reason: str) -> None:
        nonlocal stop_sent
        if stop_sent:
            return
        stop_sent = True
        _stop_motion(robot, side, args, reason)

    def _request_stop(signum, frame) -> None:
        del frame
        nonlocal stop_requested
        stop_requested = True
        print(f"\n{side}: received signal {signum}; stopping motion...", file=sys.stderr)
        _stop_once(f"signal {signum}")
        raise KeyboardInterrupt

    def _should_stop() -> bool:
        return stop_requested

    old_sigint = signal.signal(signal.SIGINT, _request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, _request_stop)
    try:
        robot = Arm(RM65, ip)
        current = _read_once(robot, side)
        start_path = _check_start_path(current["joint_deg"], targets[0], args)

        last_joint = current["joint_deg"]
        for local_index, target in enumerate(targets, start=1):
            global_index = args.start_index + local_index - 1
            waypoints = (
                _interpolated_waypoints(
                    last_joint,
                    target,
                    args.interp_max_joint_step_deg,
                )
                if args.interpolate_jumps
                else [target]
            )
            motion_plan = _waypoint_plan_summary(last_joint, target, args)
            print(
                f"\nmove target #{global_index}: {_format(target, precision=3)} "
                f"speed={args.movej_speed} "
                f"interp_speed={_interp_movej_speed(args)} "
                f"segments={motion_plan['segments']} "
                f"intermediate={motion_plan['intermediate_waypoints']} "
                f"max_segment_delta={motion_plan['max_segment_joint_delta_deg']:.3f}deg "
                f"interp_connect={args.interp_trajectory_connect}"
            )
            for waypoint_index, waypoint in enumerate(waypoints, start=1):
                final_waypoint = waypoint_index == len(waypoints)
                if not final_waypoint:
                    print(
                        f"  interp {waypoint_index}/{len(waypoints) - 1}: "
                        f"{_format(waypoint, precision=3)} "
                        f"speed={_interp_movej_speed(args)} "
                        f"connect={int(args.interp_trajectory_connect)} "
                        f"r={args.interp_blend_r if args.interp_trajectory_connect else 0}"
                    )
                speed = args.movej_speed if final_waypoint else _interp_movej_speed(args)
                connect = (
                    int(args.interp_trajectory_connect)
                    if not final_waypoint
                    else 0
                )
                blend_r = (
                    args.interp_blend_r
                    if not final_waypoint and args.interp_trajectory_connect
                    else 0
                )
                _require_zero(
                    robot.Movej_Cmd(
                        waypoint,
                        speed,
                        r=blend_r,
                        connect=connect,
                        block=False,
                    ),
                    (
                        f"{side} Movej_Cmd target #{global_index}"
                        if final_waypoint
                        else (
                            f"{side} Movej_Cmd target #{global_index} "
                            f"interp #{waypoint_index}"
                        )
                    ),
                )
                should_wait = final_waypoint or not args.interp_trajectory_connect
                if should_wait:
                    _wait_for_joint_target(
                        robot,
                        side,
                        waypoint,
                        args,
                        (
                            f"target #{global_index}"
                            if final_waypoint
                            else f"target #{global_index} interp #{waypoint_index}"
                        ),
                        _should_stop,
                    )
            if _should_stop():
                raise ReplayInterrupted("replay interrupted")
            if args.settle_sec:
                time.sleep(args.settle_sec)
            if _should_stop():
                raise ReplayInterrupted("replay interrupted")
            summary = _sample_at_current_pose(robot, side, args)
            tracking = _joint_distance_deg(summary["joint_deg_mean"], target)
            if tracking["max_abs"] > args.max_tracking_error_deg:
                raise RuntimeError(
                    f"target #{global_index} tracking error {tracking['max_abs']:.3f}deg > "
                    f"{args.max_tracking_error_deg:.3f}deg"
                )
            replay_record = {
                "pose_index": global_index,
                "target_joint_deg": target,
                "tracking_error_deg": tracking,
                "motion_plan": motion_plan,
                "sample_summary": summary,
            }
            replay_records.append(replay_record)
            last_joint = summary["joint_deg_mean"]
            source = summary["force"]["tool"]
            print(
                f"sampled #{global_index}: tracking_max={tracking['max_abs']:.4f}deg "
                f"tool|F|={source['force_norm']:.4f}N "
                f"tool|T|={source['torque_norm']:.4f}Nm"
            )

    except KeyboardInterrupt:
        stop_requested = True
        _stop_once("KeyboardInterrupt")
        print("ERROR: replay interrupted by user", file=sys.stderr)
        return 130
    except ReplayInterrupted as exc:
        _stop_once("interrupt")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        _stop_once("runtime error")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if robot is not None:
            robot.Arm_Socket_Close()
        if replay_records:
            output = {
                "schema_version": 2,
                "data_type": "force_measurement_dataset",
                "generated_by": "hardware/rm65/utils/scripts/rm65_replay_force_pose_records.py",
                "arm": side,
                "ip": ip,
                "tool_label": tool_label,
                "pose_sequence_json": str(input_path),
                "pose_sequence_schema_version": input_payload.get("schema_version"),
                "pose_count": len(input_payload.get("poses", input_payload.get("records", []))),
                "movej_speed": args.movej_speed,
                "interp_speed_multiplier": args.interp_speed_multiplier,
                "interp_movej_speed": _interp_movej_speed(args),
                "max_movej_speed": args.max_movej_speed,
                "interpolate_jumps": args.interpolate_jumps,
                "interp_max_joint_step_deg": args.interp_max_joint_step_deg,
                "max_interpolation_segments": args.max_interpolation_segments,
                "interp_trajectory_connect": args.interp_trajectory_connect,
                "interp_blend_r": args.interp_blend_r,
                "interrupt_stop_mode": args.interrupt_stop_mode,
                "motion_poll_sec": args.motion_poll_sec,
                "move_timeout_sec": args.move_timeout_sec,
                "move_reached_tolerance_deg": args.move_reached_tolerance_deg,
                "start_path": start_path,
                "settle_sec": args.settle_sec,
                "samples": args.samples,
                "interval_sec": args.interval_sec,
                "created_at": time.time(),
                "measurements": replay_records,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"\nwrote {len(replay_records)} replay samples to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
