from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent
RESULTS_DIR = UTILS_DIR / "results"
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65
from hardware.rm65.standard_paths import resolve_fit_path, tool_payload_validation_path


FORCE_SOURCES = ("raw", "zero", "work", "tool")
EXTERNAL_FORCE_SOURCES = ("zero", "work", "tool")


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _finite_float_list(values: Iterable[float], *, expected_len: int) -> list[float]:
    out = [float(value) for value in values]
    if len(out) != expected_len:
        raise RuntimeError(f"expected {expected_len} values, got {len(out)}")
    if not all(math.isfinite(value) for value in out):
        raise RuntimeError(f"non-finite values: {out}")
    return out


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    return [statistics.fmean(vector[index] for vector in vectors) for index in range(len(vectors[0]))]


def _std_vector(vectors: list[list[float]]) -> list[float]:
    if len(vectors) <= 1:
        return [0.0] * len(vectors[0])
    return [statistics.pstdev(vector[index] for vector in vectors) for index in range(len(vectors[0]))]


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _default_fit_path(arm: str, tool_label: str | None) -> Path:
    return resolve_fit_path(arm, tool_label, require_existing=True)


def _load_fit(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        raise RuntimeError(f"{path} does not contain a fit object")
    if fit.get("mode") != "tool_gravity_model":
        raise RuntimeError(f"{path} is not a tool gravity model fit")
    if fit.get("source") != "tool":
        raise RuntimeError(f"{path} was fitted from source={fit.get('source')}, expected tool")
    return payload, fit


def _current_tool_name(frame: dict[str, object]) -> str:
    value = frame.get("name") or frame.get("frame_name") or frame.get("tool_name")
    return str(value or "").strip("\x00")


def _tool_payload_com(frame: dict[str, object]) -> tuple[float, list[float]]:
    payload = float(frame.get("payload", math.nan))
    com = [float(frame.get(axis, math.nan)) for axis in ("x", "y", "z")]
    if not math.isfinite(payload) or not all(math.isfinite(value) for value in com):
        raise RuntimeError(f"current tool frame lacks finite payload/CoM: {frame}")
    return payload, com


def _infer_controller_com_unit(com_raw: list[float], fit_com_m: list[float]) -> tuple[str, list[float], list[float]]:
    com_error_as_m = [abs(com_raw[index] - fit_com_m[index]) for index in range(3)]
    com_m_from_mm = [value / 1000.0 for value in com_raw]
    com_error_as_mm_readback = [abs(com_m_from_mm[index] - fit_com_m[index]) for index in range(3)]
    if max(com_error_as_mm_readback) < max(com_error_as_m):
        return "mm_readback", com_m_from_mm, com_error_as_mm_readback
    return "m", com_raw, com_error_as_m


def _check_tool_frame(robot: Arm, expected_tool: str, fit: dict[str, object], args) -> dict[str, object]:
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    if not isinstance(frame, dict):
        raise RuntimeError(f"unexpected current tool frame payload: {frame}")

    name = _current_tool_name(frame)
    payload, com_raw = _tool_payload_com(frame)
    fit_payload = float(fit["mass_kg_abs"])
    fit_com = _finite_float_list(fit["com_tool_m"], expected_len=3)
    payload_error = abs(payload - fit_payload)
    com_unit, com_m, com_error = _infer_controller_com_unit(com_raw, fit_com)

    print(f"current_tool={name}")
    print(f"controller_payload_kg={payload:.6f} fit_payload_kg={fit_payload:.6f} err={payload_error:.6f}")
    print(
        f"controller_com_raw={_format(com_raw, precision=6)} "
        f"inferred_unit={com_unit} normalized_com_m={_format(com_m, precision=6)}"
    )
    print(f"fit_com_m={_format(fit_com, precision=6)} normalized_com_abs_err_m={_format(com_error, precision=6)}")
    if expected_tool and expected_tool not in name:
        raise RuntimeError(f"expected current tool containing '{expected_tool}', got '{name}'")
    if payload_error > args.max_payload_error_kg:
        raise RuntimeError(
            f"current tool payload error {payload_error:.6f}kg > "
            f"--max-payload-error-kg {args.max_payload_error_kg:.6f}kg"
        )
    if max(com_error) > args.max_com_error_m:
        raise RuntimeError(
            f"current tool CoM max error {max(com_error):.6f}m > "
            f"--max-com-error-m {args.max_com_error_m:.6f}m"
        )
    return {
        "current_tool_frame": frame,
        "fit_payload_kg": fit_payload,
        "fit_com_m": fit_com,
        "controller_com_raw": com_raw,
        "controller_com_inferred_unit": com_unit,
        "controller_com_m": com_m,
        "payload_error_kg": payload_error,
        "com_error_m": com_error,
    }


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
            "max_force_std": max(std[:3]),
            "max_torque_std": max(std[3:]),
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
            f"stdFmax={stats['max_force_std']:.4f} stdTmax={stats['max_torque_std']:.4f}"
        )
        if baseline is not None:
            base_stats = baseline["force"][source]
            delta = [a - b for a, b in zip(stats["mean"], base_stats["mean"])]
            line += f" delta={_format(delta)} delta|F|={_norm(delta[:3]):.4f}N"
        print(line)


def _evaluate(checkpoints: list[dict[str, object]], args) -> tuple[bool, list[str]]:
    failures = []
    sources = EXTERNAL_FORCE_SOURCES if args.evaluate_source == "external" else (args.evaluate_source,)
    for checkpoint in checkpoints:
        for source in sources:
            stats = checkpoint["force"][source]
            if stats["force_norm"] > args.max_force_norm_n:
                failures.append(
                    f"checkpoint {checkpoint['index']} {source} |F| "
                    f"{stats['force_norm']:.3f}N > {args.max_force_norm_n:.3f}N"
                )
            if stats["torque_norm"] > args.max_torque_norm_nm:
                failures.append(
                    f"checkpoint {checkpoint['index']} {source} |T| "
                    f"{stats['torque_norm']:.3f}Nm > {args.max_torque_norm_nm:.3f}Nm"
                )
    if len(checkpoints) > 1:
        baseline = checkpoints[0]
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
    return not failures, failures


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
            "stdin is not interactive. Use `conda run --no-capture-output ...` "
            "or add `--checkpoint-wait-sec SECONDS`."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only RM65 official tool payload/CoM validation. It checks the "
            "active RealMan tool frame against a fit JSON, then samples no-contact "
            "force across one or more manually placed poses."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--fit-json", default="")
    parser.add_argument("--tool-label", default="")
    parser.add_argument("--expected-tool", default="")
    parser.add_argument("--checkpoints", type=int, default=4)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval-sec", type=float, default=0.05)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--checkpoint-wait-sec", type=float, default=None)
    parser.add_argument("--evaluate-source", choices=("external", "zero", "work", "tool", "raw"), default="tool")
    parser.add_argument("--evaluate-metric", choices=("norm", "vector"), default="norm")
    parser.add_argument("--max-payload-error-kg", type=float, default=0.002)
    parser.add_argument("--max-com-error-m", type=float, default=0.002)
    parser.add_argument("--max-force-norm-n", type=float, default=1.0)
    parser.add_argument("--max-torque-norm-nm", type=float, default=0.12)
    parser.add_argument("--max-pose-force-delta-n", type=float, default=0.6)
    parser.add_argument("--max-pose-torque-delta-nm", type=float, default=0.08)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser


def _validate_args(args) -> None:
    if args.checkpoints < 1:
        raise RuntimeError("--checkpoints must be >= 1")
    if args.samples < 1:
        raise RuntimeError("--samples must be >= 1")
    if args.interval_sec < 0.0:
        raise RuntimeError("--interval-sec must be >= 0")
    if args.checkpoint_wait_sec is not None and args.checkpoint_wait_sec < 0.0:
        raise RuntimeError("--checkpoint-wait-sec must be >= 0")
    if args.non_interactive and args.checkpoints != 1:
        raise RuntimeError("--non-interactive only supports one checkpoint")
    for name in (
        "max_payload_error_kg",
        "max_com_error_m",
        "max_force_norm_n",
        "max_torque_norm_nm",
        "max_pose_force_delta_n",
        "max_pose_torque_delta_nm",
    ):
        if getattr(args, name) <= 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be positive")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        tool_label = args.tool_label or args.expected_tool or None
        fit_path = Path(args.fit_json) if args.fit_json else _default_fit_path(args.arm, tool_label)
        fit_payload, fit = _load_fit(fit_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    print(f"fit_json={fit_path}")
    fit_tool_label = str(fit_payload.get("tool_label") or args.expected_tool or args.tool_label or "")
    print(f"fit_arm={fit_payload.get('arm')} fit_tool_label={fit_tool_label}")
    print(
        "fit quality corrected_force_norm mean/rms/max="
        f"{float(fit.get('corrected_force_norm_mean_n', math.nan)):.4f}/"
        f"{float(fit.get('corrected_force_norm_rms_n', math.nan)):.4f}/"
        f"{float(fit.get('corrected_force_norm_max_n', math.nan)):.4f}N"
    )

    robot = None
    checkpoints = []
    tool_report = {}
    try:
        robot = Arm(RM65, ip)
        print(f"{side}: connected ip={ip} api={robot.API_Version()}")
        print("READ-ONLY: this script does not command robot motion or clear force data.")
        tool_report = _check_tool_frame(robot, args.expected_tool, fit, args)
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

    passed, failures = _evaluate(checkpoints, args)
    if passed:
        print("\nRESULT: official tool payload/CoM validation passed thresholds")
    else:
        print("\nRESULT: official tool payload/CoM validation failed thresholds")
        for failure in failures:
            print(f"  - {failure}")
        if args.diagnostic_only:
            print("  diagnostic only: returning success because --diagnostic-only was set")

    payload = {
        "schema_version": 1,
        "generated_by": "hardware/rm65/utils/scripts/rm65_validate_tool_payload.py",
        "fit_json": str(fit_path),
        "arm": side,
        "ip": ip,
        "tool_report": tool_report,
        "evaluate_source": args.evaluate_source,
        "evaluate_metric": args.evaluate_metric,
        "thresholds": {
            "max_force_norm_n": args.max_force_norm_n,
            "max_torque_norm_nm": args.max_torque_norm_nm,
            "max_pose_force_delta_n": args.max_pose_force_delta_n,
            "max_pose_torque_delta_nm": args.max_pose_torque_delta_nm,
        },
        "checkpoints": checkpoints,
        "passed": passed,
        "failures": failures,
    }
    output_path = Path(args.output_json) if args.output_json else (
        tool_payload_validation_path(side, args.expected_tool or fit_tool_label)
        if (args.expected_tool or fit_tool_label)
        else None
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {output_path}")

    return 0 if passed or args.diagnostic_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
