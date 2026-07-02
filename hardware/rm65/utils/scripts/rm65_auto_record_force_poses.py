from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65
from hardware.rm65.standard_paths import force_sensor_calib4_pose_sequence_path


FORCE_SOURCES = ("raw", "zero", "work", "tool")
GRAVITY_BASE = [0.0, 0.0, -1.0]


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


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _default_output_path(side: str) -> Path:
    return force_sensor_calib4_pose_sequence_path(side)


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


def _summarize_window(readings: list[dict[str, object]]) -> dict[str, object]:
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
            "max_force_range": max(ranges[:3]),
            "max_torque_range": max(ranges[3:]),
            "max_force_std": max(std[:3]),
            "max_torque_std": max(std[3:]),
            "max_std": max(std),
        }

    joint_mean = _mean_vector(joints)
    joint_range = _range_vector(joints)
    pose_mean = _mean_vector(poses)
    pose_range = _range_vector(poses)
    return {
        "samples": len(readings),
        "duration_sec": readings[-1]["timestamp"] - readings[0]["timestamp"],
        "start_timestamp": readings[0]["timestamp"],
        "end_timestamp": readings[-1]["timestamp"],
        "joint_deg_mean": joint_mean,
        "joint_deg_std": _std_vector(joints),
        "joint_deg_range": joint_range,
        "joint_max_range_deg": max(joint_range),
        "pose_mean": pose_mean,
        "pose_range": pose_range,
        "force": force_stats,
    }


def _is_stable(summary: dict[str, object], args) -> tuple[bool, list[str]]:
    source_stats = summary["force"][args.force_source]
    failures = []
    if summary["duration_sec"] < args.stable_sec:
        failures.append(
            f"window {summary['duration_sec']:.2f}s < stable_sec {args.stable_sec:.2f}s"
        )
    if summary["joint_max_range_deg"] > args.joint_stable_deg:
        failures.append(
            f"joint range {summary['joint_max_range_deg']:.4f}deg > "
            f"{args.joint_stable_deg:.4f}deg"
        )
    if args.force_stability_metric != "off":
        if args.force_stability_metric == "range":
            force_value = source_stats["max_force_range"]
            torque_value = source_stats["max_torque_range"]
            metric_label = "range"
        else:
            force_value = source_stats["max_force_std"]
            torque_value = source_stats["max_torque_std"]
            metric_label = "std"
        if force_value > args.force_stable_n:
            failures.append(
                f"{args.force_source} force {metric_label} {force_value:.4f}N > "
                f"{args.force_stable_n:.4f}N"
            )
        if torque_value > args.torque_stable_nm:
            failures.append(
                f"{args.force_source} torque {metric_label} {torque_value:.4f}Nm > "
                f"{args.torque_stable_nm:.4f}Nm"
            )
    return not failures, failures


def _joint_distance_deg(a: list[float], b: list[float]) -> dict[str, float]:
    deltas = [abs(x - y) for x, y in zip(a, b)]
    return {
        "max_abs": max(deltas),
        "rms": math.sqrt(statistics.fmean(delta * delta for delta in deltas)),
    }


def _is_new_joint_state(
    joint_deg: list[float],
    poses: list[dict[str, object]],
    distinct_joint_deg: float,
) -> tuple[bool, int | None, dict[str, float] | None]:
    nearest_index = None
    nearest_distance = None
    for index, pose in enumerate(poses, start=1):
        distance = _joint_distance_deg(joint_deg, pose["joint_deg"])
        if nearest_distance is None or distance["max_abs"] < nearest_distance["max_abs"]:
            nearest_index = index
            nearest_distance = distance
    if nearest_distance is None:
        return True, None, None
    return nearest_distance["max_abs"] > distinct_joint_deg, nearest_index, nearest_distance


def _pose_from_summary(
    summary: dict[str, object],
    *,
    side: str,
    ip: str,
    index: int,
    args,
) -> dict[str, object]:
    return {
        "index": index,
        "arm": side,
        "ip": ip,
        "recorded_at": time.time(),
        "gravity_base_dir": GRAVITY_BASE,
        "joint_deg": summary["joint_deg_mean"],
        "joint_deg_std": summary["joint_deg_std"],
        "joint_deg_range": summary["joint_deg_range"],
        "pose": summary["pose_mean"],
        "pose_range": summary["pose_range"],
        "diagnostic_force": {
            "note": (
                "Snapshot from the stable window used to accept this pose. "
                "Use rm65_replay_force_pose_records.py for end-effector-specific "
                "force measurement datasets."
            ),
            "summary": summary["force"],
        },
        "stability": {
            "force_source": args.force_source,
            "force_stability_metric": args.force_stability_metric,
            "stable_sec_required": args.stable_sec,
            "joint_stable_deg": args.joint_stable_deg,
            "force_stable_n": args.force_stable_n,
            "torque_stable_nm": args.torque_stable_nm,
            "distinct_joint_deg": args.distinct_joint_deg,
            "samples": summary["samples"],
            "duration_sec": summary["duration_sec"],
            "force_norm": summary["force"][args.force_source]["force_norm"],
            "torque_norm": summary["force"][args.force_source]["torque_norm"],
            "max_force_std": summary["force"][args.force_source]["max_force_std"],
            "max_torque_std": summary["force"][args.force_source]["max_torque_std"],
            "max_force_range": summary["force"][args.force_source]["max_force_range"],
            "max_torque_range": summary["force"][args.force_source]["max_torque_range"],
        },
    }


def _payload(side: str, ip: str, poses: list[dict[str, object]], args) -> dict[str, object]:
    return {
        "schema_version": 2,
        "data_type": "pose_sequence",
        "generated_by": "hardware/rm65/utils/scripts/rm65_auto_record_force_poses.py",
        "read_only": True,
        "arm": side,
        "ip": ip,
        "gravity_base_dir": GRAVITY_BASE,
        "sample_rate_hz": args.rate_hz,
        "stable_sec": args.stable_sec,
        "force_source": args.force_source,
        "force_stability_metric": args.force_stability_metric,
        "joint_stable_deg": args.joint_stable_deg,
        "force_stable_n": args.force_stable_n,
        "torque_stable_nm": args.torque_stable_nm,
        "distinct_joint_deg": args.distinct_joint_deg,
        "updated_at": time.time(),
        "poses": poses,
        "joint_targets_deg": [pose["joint_deg"] for pose in poses],
    }


def _pose_from_legacy_record(record: dict[str, object]) -> dict[str, object]:
    force_source = record.get("force_source_used_for_stability", "raw")
    force_stats = (record.get("force", {}) or {}).get(force_source, {}) or {}
    return {
        "index": int(record["index"]),
        "arm": record.get("arm"),
        "ip": record.get("ip"),
        "recorded_at": record.get("recorded_at"),
        "gravity_base_dir": record.get("gravity_base_dir", GRAVITY_BASE),
        "joint_deg": record["joint_deg_mean"],
        "joint_deg_std": record.get("joint_deg_std", [0.0] * 6),
        "joint_deg_range": record.get("joint_deg_range", [0.0] * 6),
        "pose": record["pose_mean"],
        "pose_range": record.get("pose_range", [0.0] * 6),
        "stability": {
            "force_source": force_source,
            "force_stability_metric": record.get("force_stability_metric", "std"),
            "stable_sec_required": record.get("stable_sec_required"),
            "joint_stable_deg": record.get("joint_stable_deg"),
            "force_stable_n": record.get("force_stable_n"),
            "torque_stable_nm": record.get("torque_stable_nm"),
            "distinct_joint_deg": record.get("distinct_joint_deg"),
            "samples": record.get("samples"),
            "duration_sec": record.get("duration_sec"),
            "force_norm": force_stats.get("force_norm"),
            "torque_norm": force_stats.get("torque_norm"),
            "max_force_std": force_stats.get("max_force_std"),
            "max_torque_std": force_stats.get("max_torque_std"),
            "max_force_range": force_stats.get("max_force_range"),
            "max_torque_range": force_stats.get("max_torque_range"),
        },
    }


def _load_existing(path: Path, side: str, ip: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if "poses" in data:
        poses = data.get("poses", [])
    else:
        poses = [_pose_from_legacy_record(record) for record in data.get("records", [])]
    if not isinstance(poses, list):
        raise RuntimeError(f"{path} does not contain a poses list")
    if data.get("arm") not in (None, side):
        raise RuntimeError(f"{path} belongs to arm={data.get('arm')}, not {side}")
    if data.get("ip") not in (None, ip):
        print(f"WARNING: existing file ip={data.get('ip')} differs from current ip={ip}")
    return poses


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _print_pose(pose: dict[str, object], args) -> None:
    stability = pose["stability"]
    print(
        f"\nRECORDED POSE #{pose['index']}: "
        f"joint_deg={_format(pose['joint_deg'], precision=3)} "
        f"pose={_format(pose['pose'], precision=4)}"
    )
    print(
        f"  stability {args.force_source}: "
        f"|F|={stability['force_norm']:.4f}N |T|={stability['torque_norm']:.4f}Nm "
        f"max_force_range={stability['max_force_range']:.4f}N "
        f"max_torque_range={stability['max_torque_range']:.4f}Nm "
        f"max_force_std={stability['max_force_std']:.4f}N "
        f"max_torque_std={stability['max_torque_std']:.4f}Nm"
    )


def _print_status(
    summary: dict[str, object],
    stable: bool,
    already_recorded: bool,
    failures: list[str],
    poses: list[dict[str, object]],
    args,
) -> None:
    source = summary["force"][args.force_source]
    if stable and already_recorded:
        state = "already_recorded"
    elif stable:
        state = "stable"
    else:
        state = "waiting"
    reason = "" if not failures else f" ({failures[-1]})"
    if args.force_stability_metric == "range":
        force_metric = source["max_force_range"]
        force_metric_label = "force_range"
    elif args.force_stability_metric == "std":
        force_metric = source["max_force_std"]
        force_metric_label = "force_std"
    else:
        force_metric = source["max_force_range"]
        force_metric_label = "force_range"
    print(
        f"{state:>12} poses={len(poses)} "
        f"window={summary['duration_sec']:.2f}s "
        f"joint_range={summary['joint_max_range_deg']:.4f}deg "
        f"{args.force_source}|F|={source['force_norm']:.3f}N "
        f"{force_metric_label}={force_metric:.3f}N "
        f"force_range={source['max_force_range']:.3f}N{reason}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only RM65 automatic pose-sequence recorder. It samples joint angles "
            "and force stability, then records a new pose when the current window is stable."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--stable-sec", type=float, default=5.0)
    parser.add_argument("--joint-stable-deg", type=float, default=0.05)
    parser.add_argument("--force-source", choices=FORCE_SOURCES, default="raw")
    parser.add_argument(
        "--force-stability-metric",
        choices=("std", "range", "off"),
        default="std",
        help=(
            "Use max component standard deviation by default. Range is stricter "
            "and can reject a still robot because of sensor spikes."
        ),
    )
    parser.add_argument("--force-stable-n", type=float, default=0.25)
    parser.add_argument("--torque-stable-nm", type=float, default=0.02)
    parser.add_argument("--distinct-joint-deg", type=float, default=2.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--status-interval-sec", type=float, default=1.0)
    parser.add_argument(
        "--wait-enter-to-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for Enter before collecting samples. Use --no-wait-enter-to-start for automation.",
    )
    return parser


def _validate_args(args) -> None:
    if args.rate_hz <= 0.0:
        raise RuntimeError("--rate-hz must be positive")
    if args.stable_sec <= 0.0:
        raise RuntimeError("--stable-sec must be positive")
    if args.joint_stable_deg <= 0.0:
        raise RuntimeError("--joint-stable-deg must be positive")
    if args.force_stable_n <= 0.0:
        raise RuntimeError("--force-stable-n must be positive")
    if args.torque_stable_nm <= 0.0:
        raise RuntimeError("--torque-stable-nm must be positive")
    if args.distinct_joint_deg <= 0.0:
        raise RuntimeError("--distinct-joint-deg must be positive")
    if args.max_records < 0:
        raise RuntimeError("--max-records must be >= 0")
    if args.status_interval_sec <= 0.0:
        raise RuntimeError("--status-interval-sec must be positive")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    output_path = Path(args.output_json) if args.output_json else _default_output_path(side)
    try:
        poses = _load_existing(output_path, side, ip) if args.resume else []
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.max_records and len(poses) >= args.max_records:
        print(
            f"{output_path} already has {len(poses)} poses, meeting "
            f"--max-records {args.max_records}; nothing to record. "
            "Use --no-resume to replace the file or a larger --max-records to append."
        )
        return 0

    robot = None
    stop_requested = False

    def _request_stop(signum, frame) -> None:
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, _request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, _request_stop)

    try:
        robot = Arm(RM65, ip)
        print(f"{side}: connected ip={ip}")
        print("READ-ONLY: this script does not command robot motion or clear force data.")
        print(f"output_json={output_path}")
        if poses:
            print(f"loaded {len(poses)} existing poses")
        print(
            "data_type=pose_sequence; force is used for stability and stored only "
            "as a diagnostic snapshot, not as a tool measurement dataset."
        )
        print(
            f"record rule: stable {args.stable_sec:.1f}s, "
            f"joint_range<={args.joint_stable_deg:.4f}deg, "
            f"force_metric={args.force_stability_metric}, "
            f"{args.force_source} force<={args.force_stable_n:.3f}N, "
            f"torque<={args.torque_stable_nm:.3f}Nm, "
            f"new_pose>{args.distinct_joint_deg:.3f}deg"
        )
        print("Move the arm through safe poses; press Ctrl-C to stop and keep the JSON file.")
        if args.wait_enter_to_start:
            input("Press Enter when the arm is at the first candidate pose and clear of contact...")

        window: deque[dict[str, object]] = deque()
        sample_interval = 1.0 / args.rate_hz
        last_status = 0.0

        while not stop_requested:
            loop_start = time.monotonic()
            reading = _read_once(robot, side)
            window.append(reading)
            while (
                window
                and reading["timestamp"] - window[0]["timestamp"]
                > args.stable_sec + sample_interval
            ):
                window.popleft()

            if len(window) >= 2:
                summary = _summarize_window(list(window))
                stable, failures = _is_stable(summary, args)
                is_new, nearest_index, nearest_distance = _is_new_joint_state(
                    summary["joint_deg_mean"],
                    poses,
                    args.distinct_joint_deg,
                )
                if stable and is_new:
                    pose = _pose_from_summary(
                        summary,
                        side=side,
                        ip=ip,
                        index=len(poses) + 1,
                        args=args,
                    )
                    poses.append(pose)
                    _write_payload(output_path, _payload(side, ip, poses, args))
                    _print_pose(pose, args)
                    if args.max_records and len(poses) >= args.max_records:
                        print(f"reached --max-records {args.max_records}")
                        stop_requested = True
                elif time.monotonic() - last_status >= args.status_interval_sec:
                    already_recorded = stable and not is_new
                    if stable and nearest_distance is not None:
                        failures = [
                            f"near record #{nearest_index}, "
                            f"max_joint_delta={nearest_distance['max_abs']:.3f}deg"
                        ]
                    _print_status(summary, stable, already_recorded, failures, poses, args)
                    last_status = time.monotonic()

            elapsed = time.monotonic() - loop_start
            if elapsed < sample_interval:
                time.sleep(sample_interval - elapsed)

    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if robot is not None:
            robot.Arm_Socket_Close()
        if poses:
            _write_payload(output_path, _payload(side, ip, poses, args))
            print(f"\nwrote {len(poses)} poses to {output_path}")
        else:
            print("\nno poses captured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
