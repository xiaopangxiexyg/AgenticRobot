from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent
DATA_DIR = UTILS_DIR / "data"
POSE_DIR = DATA_DIR / "poses"
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65
from hardware.rm65.standard_paths import (
    force_sensor_calib4_pose_sequence_path,
    official_force_sensor_report_path,
)


CONFIRM_TEXT = "CALIBRATE_RM65_FORCE_SENSOR"
FORCE_SOURCES = ("raw", "zero", "work", "tool")
VENDOR_RM65_FORCE_SENSOR_POINTS_DEG = [
    [0.0, 0.0, -60.0, 0.0, 60.0, 0.0],
    [0.0, 0.0, -60.0, 0.0, -30.0, 0.0],
    [0.0, 0.0, -60.0, 0.0, -30.0, 180.0],
    [0.0, 0.0, -60.0, 0.0, -120.0, 0.0],
]


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


def _joint_distance_deg(a: list[float], b: list[float]) -> dict[str, object]:
    deltas = [abs(x - y) for x, y in zip(a, b)]
    return {
        "max_abs": max(deltas),
        "rms": math.sqrt(statistics.fmean(delta * delta for delta in deltas)),
        "per_joint": deltas,
    }


def _interpolation_segment_count(
    start: list[float],
    target: list[float],
    max_joint_step_deg: float,
) -> int:
    distance = _joint_distance_deg(start, target)
    return max(1, math.ceil(float(distance["max_abs"]) / max_joint_step_deg))


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _default_pose_path(side: str) -> Path:
    calib4 = force_sensor_calib4_pose_sequence_path(side)
    if calib4.exists():
        return calib4
    return POSE_DIR / f"rm65_{side}_pose_sequence.json"


def _default_report_path(side: str, tool_label: str) -> Path:
    return official_force_sensor_report_path(side, tool_label or "unlabeled_tool")


def _load_pose_sequence(path: Path, side: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not path.exists():
        raise RuntimeError(f"missing pose sequence file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("arm") not in (None, side):
        raise RuntimeError(f"{path} belongs to arm={payload.get('arm')}, not {side}")

    raw_poses = payload.get("poses")
    if not raw_poses and payload.get("joint_targets_deg"):
        raw_poses = [
            {"index": index, "joint_deg": joint, "pose": []}
            for index, joint in enumerate(payload["joint_targets_deg"], start=1)
        ]
    if not isinstance(raw_poses, list) or len(raw_poses) < 4:
        raise RuntimeError(f"{path} must contain at least 4 poses")

    poses = []
    for fallback_index, pose in enumerate(raw_poses, start=1):
        joint = _finite_float_list(pose.get("joint_deg", []), expected_len=6)
        pose_value = pose.get("pose", [])
        if pose_value:
            pose_value = _finite_float_list(pose_value, expected_len=6)
        poses.append(
            {
                "index": int(pose.get("index", fallback_index)),
                "joint_deg": joint,
                "pose": pose_value,
            }
        )
    return payload, poses


def _parse_index_list(text: str) -> list[int]:
    if not text.strip():
        return []
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def _poses_by_indices(poses: list[dict[str, object]], indices: list[int]) -> list[dict[str, object]]:
    by_index = {int(pose["index"]): pose for pose in poses}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise RuntimeError(f"pose indices not found: {missing}")
    selected = [by_index[index] for index in indices]
    if len(selected) != 4:
        raise RuntimeError("--pose-indices must contain exactly 4 pose indices")
    if len({int(pose["index"]) for pose in selected}) != 4:
        raise RuntimeError("--pose-indices must contain 4 distinct pose indices")
    return selected


def _vendor_rm65_force_sensor_poses() -> list[dict[str, object]]:
    return [
        {
            "index": f"vendor_rm65_{index}",
            "joint_deg": joint,
            "pose": [],
        }
        for index, joint in enumerate(VENDOR_RM65_FORCE_SENSOR_POINTS_DEG, start=1)
    ]


def _score_combo(combo: tuple[dict[str, object], ...]) -> tuple[float, float, float]:
    pairwise = [
        _joint_distance_deg(a["joint_deg"], b["joint_deg"])
        for a, b in itertools.combinations(combo, 2)
    ]
    adjacent = [
        _joint_distance_deg(combo[index]["joint_deg"], combo[index + 1]["joint_deg"])
        for index in range(len(combo) - 1)
    ]
    min_pair_rms = min(float(distance["rms"]) for distance in pairwise)
    mean_pair_rms = statistics.fmean(float(distance["rms"]) for distance in pairwise)
    max_adjacent = max(float(distance["max_abs"]) for distance in adjacent)
    return min_pair_rms, mean_pair_rms, -max_adjacent


def _auto_select_poses(poses: list[dict[str, object]], args) -> list[dict[str, object]]:
    candidate_indices = _parse_index_list(args.candidate_indices)
    candidates = _poses_by_indices(poses, candidate_indices) if candidate_indices else poses
    if len(candidates) < 4:
        raise RuntimeError("at least 4 candidate poses are required")

    best_safe = None
    best_safe_score = None
    best_any = None
    best_any_score = None
    for combo in itertools.permutations(candidates, 4):
        adjacent = [
            _joint_distance_deg(combo[index]["joint_deg"], combo[index + 1]["joint_deg"])
            for index in range(3)
        ]
        max_adjacent = max(float(distance["max_abs"]) for distance in adjacent)
        score = _score_combo(combo)
        if best_any_score is None or score > best_any_score:
            best_any = combo
            best_any_score = score
        if max_adjacent <= args.max_adjacent_joint_delta_deg:
            if best_safe_score is None or score > best_safe_score:
                best_safe = combo
                best_safe_score = score

    selected = best_safe or best_any
    if selected is None:
        raise RuntimeError("could not select calibration poses")
    return list(selected)


def _read_state(robot: Arm, side: str) -> dict[str, object]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")
    return {
        "joint_deg": _finite_float_list(joint_deg, expected_len=6),
        "pose": _finite_float_list(pose, expected_len=6),
    }


def _read_force(robot: Arm, side: str) -> dict[str, list[float]]:
    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    return {
        "raw": _finite_float_list(raw, expected_len=6),
        "zero": _finite_float_list(zero, expected_len=6),
        "work": _finite_float_list(work, expected_len=6),
        "tool": _finite_float_list(tool, expected_len=6),
    }


def _sample_force(robot: Arm, side: str, args) -> dict[str, object]:
    states = []
    readings = []
    for index in range(args.samples):
        states.append(_read_state(robot, side))
        readings.append(_read_force(robot, side))
        if index + 1 < args.samples and args.interval_sec:
            time.sleep(args.interval_sec)
    force_stats = {}
    for source in FORCE_SOURCES:
        vectors = [reading[source] for reading in readings]
        mean = _mean_vector(vectors)
        std = _std_vector(vectors)
        force_stats[source] = {
            "mean": mean,
            "std": std,
            "force_norm": _norm(mean[:3]),
            "torque_norm": _norm(mean[3:]),
            "max_force_std": max(std[:3]),
            "max_torque_std": max(std[3:]),
        }
    return {
        "timestamp": time.time(),
        "samples": args.samples,
        "joint_deg_mean": _mean_vector([state["joint_deg"] for state in states]),
        "pose_mean": _mean_vector([state["pose"] for state in states]),
        "force": force_stats,
    }


def _print_force_summary(label: str, summary: dict[str, object]) -> None:
    print(label)
    print(f"  joint_deg={_format(summary['joint_deg_mean'], precision=3)}")
    print(f"  pose={_format(summary['pose_mean'], precision=4)}")
    for source in ("raw", "zero", "work", "tool"):
        stats = summary["force"][source]
        print(
            f"  {source}: mean={_format(stats['mean'])} "
            f"|F|={stats['force_norm']:.4f}N "
            f"|T|={stats['torque_norm']:.4f}Nm "
            f"max_std={max(stats['max_force_std'], stats['max_torque_std']):.4f}"
        )


def _check_expected_tool(robot: Arm, args) -> str:
    if not args.expected_tool:
        return ""
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    current = str(frame).strip("\x00")
    print(f"current_tool={current}")
    if args.expected_tool not in current:
        raise RuntimeError(
            f"expected active tool containing '{args.expected_tool}', got '{current}'"
        )
    return current


def _print_selected(selected: list[dict[str, object]], args) -> None:
    print("selected official force calibration poses:")
    for point_num, pose in enumerate(selected, start=1):
        print(
            f"  point {point_num}: pose_index={pose['index']} "
            f"joint_deg={_format(pose['joint_deg'], precision=3)} "
            f"tcp_pose={_format(pose.get('pose', []), precision=4) if pose.get('pose') else '[]'}"
        )
    for index in range(3):
        distance = _joint_distance_deg(
            selected[index]["joint_deg"],
            selected[index + 1]["joint_deg"],
        )
        segments = _interpolation_segment_count(
            selected[index]["joint_deg"],
            selected[index + 1]["joint_deg"],
            args.interp_max_joint_step_deg,
        )
        print(
            f"  adjacent point {index + 1}->{index + 2}: "
            f"max_delta={distance['max_abs']:.3f}deg "
            f"rms={distance['rms']:.3f}deg "
            f"interp_segments={segments} "
            f"per_joint={_format(distance['per_joint'], precision=3)}"
        )


def _validate_selected(selected: list[dict[str, object]], args) -> None:
    if args.use_vendor_rm65_points and args.allow_vendor_large_jumps:
        return
    for index in range(3):
        distance = _joint_distance_deg(
            selected[index]["joint_deg"],
            selected[index + 1]["joint_deg"],
        )
        if float(distance["max_abs"]) > args.max_adjacent_joint_delta_deg:
            segments = _interpolation_segment_count(
                selected[index]["joint_deg"],
                selected[index + 1]["joint_deg"],
                args.interp_max_joint_step_deg,
            )
            if not args.smooth_between_points:
                raise RuntimeError(
                    f"selected point {index + 1}->{index + 2} max joint jump "
                    f"{distance['max_abs']:.3f}deg > "
                    f"--max-adjacent-joint-delta-deg {args.max_adjacent_joint_delta_deg:.3f}; "
                    "choose closer poses, use --pose-indices/--candidate-indices, or raise "
                    "--max-adjacent-joint-delta-deg only after reviewing the official "
                    "controller calibration path"
                )
            if segments > args.max_interpolation_segments:
                raise RuntimeError(
                    f"selected point {index + 1}->{index + 2} max joint jump "
                    f"{distance['max_abs']:.3f}deg needs {segments} interpolation segments > "
                    f"--max-interpolation-segments {args.max_interpolation_segments}"
                )
            print(
                f"WARNING: selected point {index + 1}->{index + 2} max joint jump "
                f"{distance['max_abs']:.3f}deg exceeds "
                f"--max-adjacent-joint-delta-deg {args.max_adjacent_joint_delta_deg:.3f}; "
                f"--smooth-between-points will add a {segments}-segment pre-submission "
                "MoveJ pass, but the controller may still run its own calibration motion "
                "after point 4"
            )


def _interpolated_waypoints(
    start: list[float],
    target: list[float],
    max_joint_step_deg: float,
) -> list[list[float]]:
    segments = _interpolation_segment_count(start, target, max_joint_step_deg)
    return [
        [
            start[joint_index]
            + (target[joint_index] - start[joint_index]) * segment / segments
            for joint_index in range(6)
        ]
        for segment in range(1, segments + 1)
    ]


def _move_to_joint(
    robot: Arm,
    side: str,
    current_joint: list[float],
    target: list[float],
    args,
    *,
    label: str,
) -> dict[str, object]:
    waypoints = _interpolated_waypoints(current_joint, target, args.interp_max_joint_step_deg)
    if len(waypoints) > args.max_interpolation_segments:
        raise RuntimeError(
            f"{label} requires {len(waypoints)} interpolation segments > "
            f"--max-interpolation-segments {args.max_interpolation_segments}"
        )
    distance = _joint_distance_deg(current_joint, target)
    print(
        f"{side}: moving {label} with {len(waypoints)} MoveJ segment(s), "
        f"max_delta={distance['max_abs']:.3f}deg speed={args.movej_speed}"
    )
    for index, waypoint in enumerate(waypoints, start=1):
        print(f"  {label} interp {index}/{len(waypoints)}: {_format(waypoint, precision=3)}")
        _require_zero(
            robot.Movej_Cmd(waypoint, args.movej_speed, r=0, connect=0, block=True),
            f"{side} Movej_Cmd {label} interp {index}",
        )
    return {
        "label": label,
        "segments": len(waypoints),
        "max_joint_delta_deg": distance["max_abs"],
        "rms_joint_delta_deg": distance["rms"],
    }


def _move_to_first(robot: Arm, side: str, current_joint: list[float], target: list[float], args) -> None:
    _move_to_joint(robot, side, current_joint, target, args, label="to first calibration pose")
    if args.settle_sec:
        time.sleep(args.settle_sec)


def _check_current_to_first(current_joint: list[float], first_joint: list[float], args) -> None:
    distance = _joint_distance_deg(current_joint, first_joint)
    segments = len(_interpolated_waypoints(current_joint, first_joint, args.interp_max_joint_step_deg))
    print(f"current_joint_deg={_format(current_joint, precision=3)}")
    print(
        f"distance_to_first max={distance['max_abs']:.3f}deg "
        f"rms={distance['rms']:.3f}deg "
        f"interp_segments_if_moved={segments}"
    )
    if (
        float(distance["max_abs"]) > args.max_start_joint_delta_deg
        and not args.move_to_first
        and not args.allow_far_start
    ):
        raise RuntimeError(
            "current pose is far from the first calibration pose; move near it, "
            "or use --move-to-first after dry-run path inspection"
        )


def _run_official_manual_calibration(
    robot: Arm,
    side: str,
    selected: list[dict[str, object]],
    args,
) -> list[dict[str, object]]:
    results = []
    print(
        "\nStarting official rm_manual_set_force sequence. "
        "The controller is expected to run its own calibration motion after point 4."
    )
    if args.smooth_between_points:
        print(
            "WARNING: --smooth-between-points is enabled; this adds an extra "
            "MoveJ pre-submission pass before Manual_Set_Force calls."
        )
    try:
        for point_num, pose in enumerate(selected, start=1):
            joint = pose["joint_deg"]
            pre_move = None
            if args.smooth_between_points:
                current = _read_state(robot, side)
                distance = _joint_distance_deg(current["joint_deg"], joint)
                if float(distance["max_abs"]) > args.move_reached_tolerance_deg:
                    pre_move = _move_to_joint(
                        robot,
                        side,
                        current["joint_deg"],
                        joint,
                        args,
                        label=f"to calibration point {point_num}",
                    )
                    if args.point_move_settle_sec:
                        time.sleep(args.point_move_settle_sec)
                else:
                    pre_move = {
                        "label": f"to calibration point {point_num}",
                        "segments": 0,
                        "max_joint_delta_deg": distance["max_abs"],
                        "rms_joint_delta_deg": distance["rms"],
                    }
            print(
                f"{side}: rm_manual_set_force point={point_num} "
                f"pose_index={pose['index']} joint={_format(joint, precision=3)}"
            )
            code = robot.Manual_Set_Force(point_num, joint, True)
            _require_zero(code, f"{side} Manual_Set_Force point {point_num}")
            results.append(
                {
                    "point_num": point_num,
                    "pose_index": pose["index"],
                    "joint_deg": joint,
                    "pre_move": pre_move,
                    "return_code": int(code),
                    "timestamp": time.time(),
                }
            )
            if point_num < 4 and args.point_interval_sec:
                time.sleep(args.point_interval_sec)
    except Exception:
        try:
            stop_code = robot.Stop_Set_Force_Sensor()
            print(f"{side}: Stop_Set_Force_Sensor after failure -> {stop_code}")
        except Exception as stop_exc:
            print(f"{side}: Stop_Set_Force_Sensor failed after error: {stop_exc}")
        raise
    return results


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run RealMan official six-axis force sensor gravity calibration using "
            "four safe recorded RM65 poses. Defaults to dry-run; real calibration "
            f"requires --execute --confirm {CONFIRM_TEXT}."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--input-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--tool-label", default="")
    parser.add_argument("--expected-tool", default="")
    parser.add_argument(
        "--use-vendor-rm65-points",
        action="store_true",
        help="Use RealMan documented RM65 force-sensor QPOS points instead of recorded safe poses.",
    )
    parser.add_argument(
        "--allow-vendor-large-jumps",
        action="store_true",
        help="Allow the large adjacent joint deltas in the documented RM65 force-sensor points.",
    )
    parser.add_argument("--pose-indices", default="", help="Four pose indices, e.g. 1,6,12,18.")
    parser.add_argument("--candidate-indices", default="", help="Optional candidate pose index list for auto-selection.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--check-current", action="store_true")
    parser.add_argument("--move-to-first", action="store_true")
    parser.add_argument("--allow-far-start", action="store_true")
    parser.add_argument(
        "--smooth-between-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compatibility option: MoveJ through interpolated waypoints before submitting "
            "each Manual_Set_Force point. On controllers that auto-run calibration after "
            "point 4, this creates an extra pre-pass and should normally stay disabled."
        ),
    )
    parser.add_argument("--movej-speed", type=int, default=2)
    parser.add_argument("--settle-sec", type=float, default=2.0)
    parser.add_argument("--point-move-settle-sec", type=float, default=0.2)
    parser.add_argument("--point-interval-sec", type=float, default=1.0)
    parser.add_argument("--max-start-joint-delta-deg", type=float, default=8.0)
    parser.add_argument("--max-adjacent-joint-delta-deg", type=float, default=45.0)
    parser.add_argument("--interp-max-joint-step-deg", type=float, default=8.0)
    parser.add_argument("--max-interpolation-segments", type=int, default=50)
    parser.add_argument("--move-reached-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval-sec", type=float, default=0.05)
    parser.add_argument("--max-post-force-n", type=float, default=1.0)
    parser.add_argument("--max-post-torque-nm", type=float, default=0.10)
    return parser


def _validate_args(args) -> None:
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"official calibration requires --confirm {CONFIRM_TEXT}")
    if args.movej_speed < 1 or args.movej_speed > 10:
        raise RuntimeError("--movej-speed must be in [1, 10]")
    if args.settle_sec < 0.0:
        raise RuntimeError("--settle-sec must be >= 0")
    if args.point_move_settle_sec < 0.0:
        raise RuntimeError("--point-move-settle-sec must be >= 0")
    if args.point_interval_sec < 0.0:
        raise RuntimeError("--point-interval-sec must be >= 0")
    if args.max_start_joint_delta_deg <= 0.0:
        raise RuntimeError("--max-start-joint-delta-deg must be positive")
    if args.max_adjacent_joint_delta_deg <= 0.0:
        raise RuntimeError("--max-adjacent-joint-delta-deg must be positive")
    if args.interp_max_joint_step_deg <= 0.0:
        raise RuntimeError("--interp-max-joint-step-deg must be positive")
    if args.max_interpolation_segments < 1:
        raise RuntimeError("--max-interpolation-segments must be >= 1")
    if args.move_reached_tolerance_deg <= 0.0:
        raise RuntimeError("--move-reached-tolerance-deg must be positive")
    if args.samples < 1:
        raise RuntimeError("--samples must be >= 1")
    if args.interval_sec < 0.0:
        raise RuntimeError("--interval-sec must be >= 0")
    if args.max_post_force_n <= 0.0:
        raise RuntimeError("--max-post-force-n must be positive")
    if args.max_post_torque_nm <= 0.0:
        raise RuntimeError("--max-post-torque-nm must be positive")
    if args.pose_indices and len(_parse_index_list(args.pose_indices)) != 4:
        raise RuntimeError("--pose-indices must contain exactly 4 indices")
    if args.use_vendor_rm65_points and args.pose_indices:
        raise RuntimeError("--use-vendor-rm65-points cannot be combined with --pose-indices")
    if args.use_vendor_rm65_points and args.candidate_indices:
        raise RuntimeError("--use-vendor-rm65-points cannot be combined with --candidate-indices")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    input_path = Path(args.input_json) if args.input_json else _default_pose_path(side)
    tool_label = args.tool_label or args.expected_tool
    output_path = (
        Path(args.output_json)
        if args.output_json
        else _default_report_path(side, tool_label)
    )
    try:
        if args.use_vendor_rm65_points:
            input_payload = {
                "schema_version": None,
                "data_type": "vendor_rm65_force_sensor_points",
                "source": "RealMan documented rm_set_force_sensor RM65 joint-angle targets",
            }
            selected = _vendor_rm65_force_sensor_poses()
        else:
            input_payload, poses = _load_pose_sequence(input_path, side)
            if args.pose_indices:
                selected = _poses_by_indices(poses, _parse_index_list(args.pose_indices))
            else:
                selected = _auto_select_poses(poses, args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"input_json={input_path}")
    print(f"output_json={output_path}")
    print(f"execute={args.execute} arm={side} ip={ip}")
    print(f"tool_label={tool_label or '(not set)'} expected_tool={args.expected_tool or '(not checked)'}")
    print(
        "WARNING: official force sensor calibration writes controller calibration data "
        "and is expected to move the robot after point 4."
    )
    print(
        f"preposition_between_points={args.smooth_between_points} "
        f"interp_max_joint_step_deg={args.interp_max_joint_step_deg:.3f} "
        f"max_interpolation_segments={args.max_interpolation_segments}"
    )
    print("WARNING: do not run while the arm is in contact with the environment.")
    _print_selected(selected, args)
    try:
        _validate_selected(selected, args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    robot = None
    pre_summary = None
    post_summary = None
    manual_results = []
    try:
        if args.check_current or args.execute:
            robot = Arm(RM65, ip)
            if int(robot.Arm_Socket_State()) != 0:
                raise RuntimeError(f"{side}: failed to connect ip={ip}")
            print(f"{side}: connected ip={ip}")
            _check_expected_tool(robot, args)
            current = _read_state(robot, side)
            _check_current_to_first(current["joint_deg"], selected[0]["joint_deg"], args)
            pre_summary = _sample_force(robot, side, args)
            _print_force_summary("pre-calibration force summary", pre_summary)

        if not args.execute:
            print("DRY-RUN: no official force calibration command was sent.")
            return 0

        current = _read_state(robot, side)
        if args.move_to_first:
            _move_to_first(robot, side, current["joint_deg"], selected[0]["joint_deg"], args)
        manual_results = _run_official_manual_calibration(robot, side, selected, args)
        if args.settle_sec:
            time.sleep(args.settle_sec)
        post_summary = _sample_force(robot, side, args)
        _print_force_summary("post-calibration force summary", post_summary)
        tool_force = post_summary["force"]["tool"]["force_norm"]
        tool_torque = post_summary["force"]["tool"]["torque_norm"]
        if tool_force > args.max_post_force_n or tool_torque > args.max_post_torque_nm:
            print(
                "WARNING: post-calibration tool residual exceeds target: "
                f"|F|={tool_force:.4f}N limit={args.max_post_force_n:.4f}N, "
                f"|T|={tool_torque:.4f}Nm limit={args.max_post_torque_nm:.4f}Nm"
            )
        else:
            print(
                "RESULT: official force sensor calibration residual is within target "
                f"tool|F|={tool_force:.4f}N tool|T|={tool_torque:.4f}Nm"
            )
        report = {
            "schema_version": 1,
            "data_type": "official_force_sensor_calibration_report",
            "generated_by": "hardware/rm65/utils/scripts/rm65_official_force_sensor_calibrate.py",
            "arm": side,
            "ip": ip,
            "tool_label": tool_label,
            "expected_tool": args.expected_tool,
            "pose_sequence_json": str(input_path),
            "pose_sequence_schema_version": input_payload.get("schema_version"),
            "selected_pose_indices": [pose["index"] for pose in selected],
            "selected_joint_deg": [pose["joint_deg"] for pose in selected],
            "smooth_between_points": args.smooth_between_points,
            "preposition_between_points": args.smooth_between_points,
            "interp_max_joint_step_deg": args.interp_max_joint_step_deg,
            "max_interpolation_segments": args.max_interpolation_segments,
            "move_reached_tolerance_deg": args.move_reached_tolerance_deg,
            "manual_set_force_results": manual_results,
            "pre_summary": pre_summary,
            "post_summary": post_summary,
            "created_at": time.time(),
        }
        _write_report(output_path, report)
        print(f"wrote {output_path}")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            robot.Arm_Socket_Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
