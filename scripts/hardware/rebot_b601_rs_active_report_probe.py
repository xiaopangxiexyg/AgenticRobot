#!/usr/bin/env python3
"""Foreground, no-motion probe for RobStride active-report receive cadence.

The probe sends only the vendor active-report configuration request.  It never
changes a motor mode, enables a motor, or sends position/MIT references.  Raw
SocketCAN timing is delegated to the existing passive frequency probe so this
file does not duplicate RobStride ID parsing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.rebot_b601_rs import (
    B601ActiveReportReceipt,
    B601HardwareError,
    B601RSBackend,
    B601RSBackendConfig,
    evaluate_active_report_capture,
)
from scripts.hardware import rebot_b601_rs_frequency_probe as frequency_probe

_CONFIRMATION = "B601_RS_ACTIVE_REPORT_TELEMETRY"


class ActiveReportProbeError(RuntimeError):
    """A probe precondition, transport, or cleanup operation failed."""


def _receipt_payload(receipt: B601ActiveReportReceipt) -> dict[str, object]:
    return {
        "operation": receipt.operation,
        "requested_joint_names": receipt.requested_joint_names,
        "completed_joint_names": receipt.completed_joint_names,
        "failed_joint_names": receipt.failed_joint_names,
        "errors": receipt.errors,
        "completed": receipt.completed,
        "rollback_attempted": receipt.rollback_attempted,
        "rollback_completed": receipt.rollback_completed,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not args.allow_active_report:
        raise ActiveReportProbeError("active reporting requires --allow-active-report")
    if args.confirmation != _CONFIRMATION:
        raise ActiveReportProbeError(
            f"active reporting requires --confirmation {_CONFIRMATION}"
        )
    if not args.auto_disable_active_report:
        raise ActiveReportProbeError(
            "active reporting requires explicit --auto-disable-active-report"
        )
    for name in (
        "duration_sec",
        "target_hz",
        "minimum_rate_ratio",
        "maximum_p99_factor",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if args.minimum_rate_ratio > 1.0:
        raise ValueError("minimum_rate_ratio must be at most 1")
    if args.minimum_sample_count < 2:
        raise ValueError("minimum_sample_count must be at least 2")


def _run_active_report_probe(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    config_path = Path(args.config).expanduser().resolve(strict=True)
    if not config_path.is_file():
        raise ActiveReportProbeError(
            f"B601-RS config is not a regular file: {config_path}"
        )
    selected = frequency_probe._parse_joint_names(args.joint)
    config = B601RSBackendConfig.from_rs_yaml(
        config_path,
        allow_motion=False,
        allow_active_reporting=True,
    )
    if args.channel is not None:
        config = replace(config, channel=args.channel)

    backend = B601RSBackend(config)
    enable_attempted = False
    interrupted = False
    capture: dict[str, object] | None = None
    enable_receipt: B601ActiveReportReceipt | None = None
    cleanup_receipt: B601ActiveReportReceipt | None = None
    errors: list[str] = []
    cleanup_errors: list[str] = []
    try:
        backend.connect()
        print(
            "[REPORT] active telemetry only; no motor enable or motion command",
            file=sys.stderr,
            flush=True,
        )
        for remaining in (3, 2, 1):
            print(f"[REPORT] {remaining}", file=sys.stderr, flush=True)
            time.sleep(1.0)
        print("[REPORT] ENABLE STREAM", file=sys.stderr, flush=True)
        enable_attempted = True
        enable_receipt = backend.set_active_reporting(True, joint_names=selected)
        if not enable_receipt.completed:
            raise ActiveReportProbeError(
                "active-report enable failed: "
                + ("; ".join(enable_receipt.errors) or "incomplete receipt")
            )
        capture = frequency_probe._read_only_capture(config.channel, args.duration_sec)
    except KeyboardInterrupt:
        interrupted = True
        errors.append("operator interrupted the foreground probe")
    except Exception as exc:  # noqa: BLE001 - cleanup still must run
        errors.append(str(exc))
    finally:
        if backend.connected and enable_attempted:
            try:
                cleanup_receipt = backend.set_active_reporting(
                    False,
                    joint_names=selected,
                )
                if not cleanup_receipt.completed:
                    cleanup_errors.extend(cleanup_receipt.errors)
                    if not cleanup_receipt.errors:
                        cleanup_errors.append("active-report disable receipt incomplete")
            except Exception as exc:  # noqa: BLE001 - cleanup is reported, never hidden
                cleanup_errors.append(f"disable active reporting: {exc}")
            except KeyboardInterrupt:
                cleanup_errors.append(
                    "disable active reporting was interrupted; device state is uncertain"
                )
        if backend.connected:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - unresolved state must stay visible
                cleanup_errors.append(f"close: {exc}")

    acceptance = None
    if capture is not None and not errors:
        try:
            acceptance = evaluate_active_report_capture(
                capture,
                config.joints,
                selected,
                target_hz=args.target_hz,
                minimum_rate_ratio=args.minimum_rate_ratio,
                maximum_p99_factor=args.maximum_p99_factor,
                minimum_sample_count=args.minimum_sample_count,
            )
        except Exception as exc:  # noqa: BLE001 - malformed capture is probe evidence
            errors.append(f"capture acceptance: {exc}")

    if cleanup_errors:
        errors.extend(f"cleanup: {error}" for error in cleanup_errors)
    execution_status = "ERROR" if errors else "COMPLETED"
    transport_status = (
        "PASS"
        if acceptance is not None and acceptance.passed and not cleanup_errors
        else "FAIL"
    )
    result: dict[str, object] = {
        "mode": "active-report-telemetry",
        "execution_status": execution_status,
        "stream_transport_acceptance": transport_status,
        "control_feedback_admission": "NOT_ADMITTED",
        "hardware_acceptance": "NOT_CLAIMED",
        "channel": config.channel,
        "duration_sec": args.duration_sec,
        "target_hz": args.target_hz,
        "selected_joint_names": selected,
        "selected_motor_ids": tuple(
            joint.motor_id for joint in config.joints if joint.name in selected
        ),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "motion_commands_sent": False,
        "motors_enabled_by_probe": False,
        "mode_transition_called": False,
        "active_report_commands_sent": enable_attempted,
        "motorbridge_state_observation_metadata_available": (
            backend.state_observation_metadata_verified
        ),
        "freshness_verified": False,
        "freshness_semantics": (
            "socket_receive_time_only; decoded_state_observation_not_sampled; "
            "device_source_timestamp_unavailable"
        ),
        "interrupted": interrupted,
        "errors": tuple(errors),
        "cleanup_errors": tuple(cleanup_errors),
        "notes": (
            "This foreground probe configures RobStride active reporting and passively times complete type-2 operation-status frames.",
            "It does not enable motors, change control mode, or send position/MIT references.",
            "A transport PASS does not admit q/dq/tau into the control loop and is not haptic hardware acceptance.",
            "MotorBridge observation metadata availability is reported separately; this raw transport probe does not consume it.",
        ),
    }
    if enable_receipt is not None:
        result["enable_receipt"] = _receipt_payload(enable_receipt)
    if capture is not None:
        result["capture"] = capture
    if acceptance is not None:
        result["acceptance"] = acceptance.as_dict()
    if cleanup_receipt is not None:
        result["cleanup_receipt"] = _receipt_payload(cleanup_receipt)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "No-motion B601-RS RobStride active-report transport timing probe."
        )
    )
    parser.add_argument("--config", required=True, help="explicit rebotarm_rs.yaml")
    parser.add_argument(
        "--channel",
        default=None,
        help="optional SocketCAN override; otherwise use the YAML channel",
    )
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--target-hz", type=float, default=125.0)
    parser.add_argument("--minimum-rate-ratio", type=float, default=0.90)
    parser.add_argument("--maximum-p99-factor", type=float, default=2.0)
    parser.add_argument("--minimum-sample-count", type=int, default=10)
    parser.add_argument(
        "--joint",
        action="append",
        help="joint name or comma-separated names; default is all six arm joints",
    )
    parser.add_argument("--allow-active-report", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--auto-disable-active-report", action="store_true")
    parser.add_argument("--output", type=Path, help="write JSON outside the worktree")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = _run_active_report_probe(args)
    except (ActiveReportProbeError, B601HardwareError, OSError, ValueError) as exc:
        payload = {
            "mode": "active-report-telemetry",
            "execution_status": "ERROR",
            "stream_transport_acceptance": "FAIL",
            "control_feedback_admission": "NOT_ADMITTED",
            "hardware_acceptance": "NOT_CLAIMED",
            "motion_commands_sent": False,
            "error": str(exc),
        }
    status = "OK"
    exit_code = 0
    if payload.get("interrupted"):
        status = "ERROR"
        exit_code = 130
    elif payload.get("execution_status") == "ERROR":
        status = "ERROR"
        exit_code = 2
    elif payload.get("stream_transport_acceptance") != "PASS":
        status = "FAIL"
        exit_code = 1
    encoded = json.dumps({"status": status, **payload}, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.expanduser().resolve().write_text(encoded + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
