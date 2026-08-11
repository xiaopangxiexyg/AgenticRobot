from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate the DexTeleop repository root")


REPO_ROOT = _repository_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65


FORCE_SOURCES = ("raw", "zero", "work", "tool")
EXTERNAL_FORCE_SOURCES = ("zero", "work", "tool")


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    return [statistics.fmean(vector[index] for vector in vectors) for index in range(6)]


def _std_vector(vectors: list[list[float]]) -> list[float]:
    if len(vectors) <= 1:
        return [0.0] * 6
    return [statistics.pstdev(vector[index] for vector in vectors) for index in range(6)]


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _read_once(robot: Arm, side: str) -> dict[str, object]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")

    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    return {
        "timestamp": time.time(),
        "joint_deg": [float(value) for value in joint_deg],
        "pose": [float(value) for value in pose],
        "force": {
            "raw": [float(value) for value in raw],
            "zero": [float(value) for value in zero],
            "work": [float(value) for value in work],
            "tool": [float(value) for value in tool],
        },
    }


def _sample_checkpoint(robot: Arm, side: str, args, index: int) -> dict[str, object]:
    readings = []
    for sample_index in range(args.samples):
        readings.append(_read_once(robot, side))
        if sample_index + 1 < args.samples:
            time.sleep(args.interval_sec)

    joint_deg = _mean_vector([reading["joint_deg"] for reading in readings])
    pose = _mean_vector([reading["pose"] for reading in readings])
    force_stats = {}
    for source in FORCE_SOURCES:
        vectors = [reading["force"][source] for reading in readings]
        mean = _mean_vector(vectors)
        std = _std_vector(vectors)
        force_stats[source] = {
            "mean": mean,
            "std": std,
            "force_norm": _norm(mean[:3]),
            "torque_norm": _norm(mean[3:]),
            "max_std": max(std),
        }

    return {
        "index": index,
        "samples": args.samples,
        "joint_deg_mean": joint_deg,
        "pose_mean": pose,
        "force": force_stats,
    }


def _print_checkpoint(checkpoint: dict[str, object], baseline: dict[str, object] | None) -> None:
    print(f"\ncheckpoint {checkpoint['index']}")
    print(f"  joint_deg_mean={_format(checkpoint['joint_deg_mean'], precision=3)}")
    print(f"  pose_xyz_rpy_mean={_format(checkpoint['pose_mean'], precision=4)}")
    for source in FORCE_SOURCES:
        stats = checkpoint["force"][source]
        line = (
            f"  {source:>4}: mean={_format(stats['mean'])} "
            f"|F|={stats['force_norm']:.4f}N |T|={stats['torque_norm']:.4f}Nm "
            f"max_std={stats['max_std']:.4f}"
        )
        if baseline is not None:
            base_stats = baseline["force"][source]
            delta = [a - b for a, b in zip(stats["mean"], base_stats["mean"])]
            line += f" delta={_format(delta)} delta|F|={_norm(delta[:3]):.4f}N"
        print(line)


def _evaluate(checkpoints: list[dict[str, object]], args) -> int:
    if len(checkpoints) <= 1:
        return 0
    sources = EXTERNAL_FORCE_SOURCES if args.evaluate_source == "external" else (args.evaluate_source,)
    baseline = checkpoints[0]
    failures = []
    for checkpoint in checkpoints[1:]:
        for source in sources:
            stats = checkpoint["force"][source]
            base_stats = baseline["force"][source]
            delta = [a - b for a, b in zip(stats["mean"], base_stats["mean"])]
            force_delta = _norm(delta[:3])
            torque_delta = _norm(delta[3:])
            force_norm_delta = abs(stats["force_norm"] - base_stats["force_norm"])
            torque_norm_delta = abs(stats["torque_norm"] - base_stats["torque_norm"])
            if args.evaluate_metric == "vector":
                force_metric = force_delta
                torque_metric = torque_delta
                metric_name = "vector delta"
            else:
                force_metric = force_norm_delta
                torque_metric = torque_norm_delta
                metric_name = "norm delta"
            if force_metric > args.max_pose_force_delta_n:
                failures.append(
                    f"checkpoint {checkpoint['index']} {source} force {metric_name} "
                    f"{force_metric:.3f}N > {args.max_pose_force_delta_n:.3f}N"
                )
            if torque_metric > args.max_pose_torque_delta_nm:
                failures.append(
                    f"checkpoint {checkpoint['index']} {source} torque {metric_name} "
                    f"{torque_metric:.3f}Nm > {args.max_pose_torque_delta_nm:.3f}Nm"
                )
    if failures:
        print("\nRESULT: pose-dependent external wrench detected")
        for failure in failures:
            print(f"  - {failure}")
        if args.fail_on_threshold:
            return 1
        print("  diagnostic only: use --fail-on-threshold to return nonzero")
        return 0
    print("\nRESULT: external wrench stayed within pose-variation thresholds")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only RM65 force-vs-pose survey. The script never moves the robot; "
            "use pendant/teach mode to place the arm at each checkpoint, then press Enter "
            "or use --checkpoint-wait-sec for conda-run captured stdin."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--checkpoints", type=int, default=1)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval-sec", type=float, default=0.05)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument(
        "--checkpoint-wait-sec",
        type=float,
        default=None,
        help="Wait this many seconds before each checkpoint instead of reading Enter.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--evaluate-source",
        choices=("external", "zero", "work", "tool", "raw"),
        default="external",
    )
    parser.add_argument(
        "--evaluate-metric",
        choices=("norm", "vector"),
        default="norm",
        help=(
            "Use force/torque norm drift by default. Vector delta is stricter but can "
            "over-report when comparing rotated coordinate-frame components."
        ),
    )
    parser.add_argument("--max-pose-force-delta-n", type=float, default=1.0)
    parser.add_argument("--max-pose-torque-delta-nm", type=float, default=0.1)
    parser.add_argument("--fail-on-threshold", action="store_true")
    return parser


def _wait_for_checkpoint(side: str, args, index: int) -> None:
    if args.non_interactive:
        return
    if args.checkpoint_wait_sec is not None:
        wait_sec = float(args.checkpoint_wait_sec)
        print(
            f"\nPlace {side} arm at checkpoint {index}/{args.checkpoints} "
            f"with no external contact. Sampling in {wait_sec:.1f}s..."
        )
        time.sleep(wait_sec)
        return
    try:
        input(
            f"\nPlace {side} arm at checkpoint {index}/{args.checkpoints} "
            "with no external contact, then press Enter to sample..."
        )
    except EOFError as exc:
        raise RuntimeError(
            "stdin is not interactive. Use either "
            "`conda run --no-capture-output -n default python ...` for Enter prompts, "
            "or add `--checkpoint-wait-sec SECONDS` for countdown sampling."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoints < 1:
        print("ERROR: --checkpoints must be >= 1", file=sys.stderr)
        return 2
    if args.samples < 1:
        print("ERROR: --samples must be >= 1", file=sys.stderr)
        return 2
    if args.interval_sec < 0.0:
        print("ERROR: --interval-sec must be >= 0", file=sys.stderr)
        return 2
    if args.checkpoint_wait_sec is not None and args.checkpoint_wait_sec < 0.0:
        print("ERROR: --checkpoint-wait-sec must be >= 0", file=sys.stderr)
        return 2
    if args.non_interactive and args.checkpoints != 1:
        print("ERROR: --non-interactive only supports one checkpoint", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    robot = None
    checkpoints = []
    try:
        robot = Arm(RM65, ip)
        print(f"{side}: connected ip={ip}")
        print("READ-ONLY: this script does not command robot motion or clear force data.")
        for index in range(1, args.checkpoints + 1):
            _wait_for_checkpoint(side, args, index)
            checkpoint = _sample_checkpoint(robot, side, args, index)
            checkpoints.append(checkpoint)
            _print_checkpoint(checkpoint, checkpoints[0] if index > 1 else None)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            robot.Arm_Socket_Close()

    payload = {
        "arm": side,
        "ip": ip,
        "samples": args.samples,
        "interval_sec": args.interval_sec,
        "checkpoints": checkpoints,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {output_path}")

    return _evaluate(checkpoints, args)


if __name__ == "__main__":
    raise SystemExit(main())
