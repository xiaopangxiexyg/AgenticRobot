"""Foreground, J7-only RobStride MIT hold for commissioning.

This command is intentionally separate from ``rebot_b601_rs_owner.py``.  The
normal B601 owner controls J1--J6 and must not be widened implicitly when a
gripper gain is changed.  This command registers and disables only motor 7;
it never calls a controller-wide enable/disable operation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rebot_b601_rs import (
    B601RSGripperControlConfig,
    B601RSGripperController,
)

DEFAULT_REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
DEFAULT_CONFIG = DEFAULT_REBOT_SOURCE_ROOT / "config" / "rebotarm_rs.yaml"
DEFAULT_OUTPUT_ROOT = Path("/tmp/b601-rs-commissioning")
CONFIRMATION = "B601_RS_J7_MIT_GRIPPER"


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Foreground J7-only RobStride MIT hold. Runtime --gripper-kp and "
            "--gripper-kd override the YAML values for this run only."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_env_path("REBOT_RS_CONFIG", DEFAULT_CONFIG),
        help="rebotarm_rs.yaml (default: pinned hardware/reBotArm_control_py submodule)",
    )
    parser.add_argument("--channel", default=None, help="optional SocketCAN channel override")
    parser.add_argument(
        "--gripper-kp",
        "--kp",
        dest="gripper_kp",
        type=float,
        default=None,
        help="runtime J7 MIT Kp; defaults to joints.gripper.MIT.kp",
    )
    parser.add_argument(
        "--gripper-kd",
        "--kd",
        dest="gripper_kd",
        type=float,
        default=None,
        help="runtime J7 MIT Kd; defaults to joints.gripper.MIT.kd",
    )
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--max-position-error-rad", type=float, default=0.25)
    parser.add_argument("--timeout-ms", type=int, default=500)
    parser.add_argument(
        "--allow-motion",
        action="store_true",
        help="required acknowledgement that J7 may enable and hold",
    )
    parser.add_argument(
        "--auto-disable",
        action="store_true",
        help="required cleanup acknowledgement; J7 is disabled on exit",
    )
    parser.add_argument("--confirmation", default=CONFIRMATION)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("B601_RM65_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    return parser


def _emit(stream: TextIO, message: str) -> None:
    print(message, flush=True)
    stream.write(message + "\n")
    stream.flush()


def _write_result(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure_local_motorbridge_runtime() -> None:
    """Prefer the repository-pinned Python binding/ABI when it is built."""

    binding_root = REPO_ROOT / "hardware" / "MotorBridge" / "bindings" / "python" / "src"
    abi_path = REPO_ROOT / "hardware" / "MotorBridge" / "target" / "release" / "libmotor_abi.so"
    if binding_root.is_dir() and str(binding_root) not in sys.path:
        sys.path.insert(0, str(binding_root))
    if abi_path.is_file() and not os.environ.get("MOTORBRIDGE_LIB"):
        os.environ["MOTORBRIDGE_LIB"] = str(abi_path)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.allow_motion:
        parser.error("J7 owner is motion-capable; rerun with explicit --allow-motion")
    if not args.auto_disable:
        parser.error("J7 owner requires explicit --auto-disable")
    if args.confirmation != CONFIRMATION:
        parser.error(f"J7 owner requires --confirmation {CONFIRMATION}")

    args.output_root.expanduser().mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="b601-j7-owner-", dir=args.output_root.expanduser()))
    output = args.output.expanduser() if args.output is not None else run_dir / "b601_j7_owner.json"
    log_path = args.log.expanduser() if args.log is not None else run_dir / "b601_j7_owner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "status": "ERROR",
        "mode": "j7-mit-hold",
        "config": str(args.config.expanduser()),
        "output": str(output),
        "log": str(log_path),
        "motion_commands_sent": 0,
        "motor_enabled_by_probe": False,
        "cleanup_completed": False,
    }
    status = 2
    controller: B601RSGripperController | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            _emit(log, "[B601 J7] foreground gripper owner starts")
            _emit(log, "[B601 J7] ONLY motor 7 may enable; J1-J6 are not registered")
            _emit(log, "[B601 J7] Ctrl-C requests J7-only disable; physical E-stop remains primary")
            config = B601RSGripperControlConfig.from_rs_yaml(
                args.config,
                kp=args.gripper_kp,
                kd=args.gripper_kd,
                rate_hz=args.rate_hz,
                duration_sec=args.duration_sec,
                max_position_error_rad=args.max_position_error_rad,
                timeout_ms=args.timeout_ms,
            )
            if args.channel is not None:
                config = replace(config, reader=replace(config.reader, channel=args.channel))
            result.update(
                {
                    "channel": config.reader.channel,
                    "motor_id": config.reader.motor_id,
                    "feedback_id": config.reader.feedback_id,
                    "model": config.reader.model,
                    "kp": config.kp,
                    "kd": config.kd,
                    "rate_hz": config.rate_hz,
                    "duration_sec": config.duration_sec,
                    "max_position_error_rad": config.max_position_error_rad,
                }
            )
            _configure_local_motorbridge_runtime()
            controller = B601RSGripperController(config)
            controller.connect()
            target = controller.prepare_hold()
            result["target_position_rad"] = target
            _emit(log, f"[B601 J7] target={target:.6f} rad; Kp={config.kp:g}; Kd={config.kd:g}")
            _emit(log, "[B601 J7] 3")
            time.sleep(1.0)
            _emit(log, "[B601 J7] 2")
            time.sleep(1.0)
            _emit(log, "[B601 J7] 1")
            time.sleep(1.0)
            _emit(log, "[B601 J7] ENABLE")
            commands = controller.run_hold(target)
            result["motion_commands_sent"] = commands
            result["motor_enabled_by_probe"] = True
            result["status"] = "OK"
            status = 0
            _emit(log, "[B601 J7] duration complete; J7 disabled")
    except KeyboardInterrupt:
        result["status"] = "INTERRUPTED"
        result["stop_reason"] = "operator_interrupt"
        status = 130
    except Exception as exc:  # noqa: BLE001 - serialize the operator-facing receipt
        result["error"] = f"{type(exc).__name__}: {exc}"
        status = 2
    finally:
        if controller is not None:
            try:
                controller.close()
                result["cleanup_completed"] = True
            except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                status = 1 if status == 0 else status
        _write_result(output, result)
    print(f"[B601 J7] status={status}")
    print(f"[B601 J7] result={output}")
    print(f"[B601 J7] log={log_path}")
    print("[B601 J7] parent shell remains active")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
