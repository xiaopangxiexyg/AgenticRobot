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
    B601HardwareState,
    B601RSBackend,
    B601RSBackendConfig,
    evaluate_active_report_capture,
)
from hardware.rebot_b601_rs.active_report import (
    evaluate_decoded_active_report_observations,
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
        "decoded_warmup_sec",
        "decoded_target_hz",
        "decoded_minimum_rate_ratio",
        "decoded_maximum_p99_factor",
        "decoded_minimum_raw_ratio",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if args.minimum_rate_ratio > 1.0:
        raise ValueError("minimum_rate_ratio must be at most 1")
    if args.decoded_minimum_rate_ratio > 1.0:
        raise ValueError("decoded_minimum_rate_ratio must be at most 1")
    if args.decoded_minimum_raw_ratio > 1.0:
        raise ValueError("decoded_minimum_raw_ratio must be at most 1")
    if args.minimum_sample_count < 2:
        raise ValueError("minimum_sample_count must be at least 2")
    if args.decoded_minimum_sample_count < 2:
        raise ValueError("decoded_minimum_sample_count must be at least 2")
    if args.decoded_max_receive_age_sec is not None and (
        not math.isfinite(args.decoded_max_receive_age_sec)
        or args.decoded_max_receive_age_sec <= 0.0
    ):
        raise ValueError("decoded_max_receive_age_sec must be positive")


def _selected_sequences_available(
    state: B601HardwareState,
    selected: tuple[str, ...],
) -> bool:
    joints = {joint.name: joint for joint in state.joints}
    return all(
        name in joints and joints[name].status_sequence is not None for name in selected
    )


def _wait_for_host_observation_baseline(
    backend: B601RSBackend,
    selected: tuple[str, ...],
    *,
    timeout_sec: float,
) -> B601HardwareState:
    deadline = time.monotonic() + timeout_sec
    last_state = None
    while time.monotonic() < deadline:
        last_state = backend.poll_feedback(require_host_freshness=True)
        if _selected_sequences_available(last_state, selected):
            return last_state
        time.sleep(0.01)
    if last_state is None:
        raise ActiveReportProbeError("decoded observation polling produced no snapshot")
    return last_state


def _state_payload(
    state: B601HardwareState,
    selected: tuple[str, ...],
) -> dict[str, object]:
    return {
        "sequence": state.sequence,
        "receive_timestamp_sec": state.receive_timestamp_sec,
        "valid": state.valid,
        "stale": state.stale,
        "host_receive_freshness_verified": state.host_receive_freshness_verified,
        "freshness_semantics": state.freshness_semantics,
        "fault_reason": state.fault_reason,
        "joints": tuple(
            {
                "name": joint.name,
                "motor_id": joint.motor_id,
                "position_rad": joint.position_rad,
                "velocity_reported": joint.velocity_reported,
                "torque_reported_nm": joint.torque_reported_nm,
                "status_sequence": joint.status_sequence,
                "receive_age_sec": joint.receive_age_sec,
                "valid": joint.valid,
                "stale": joint.stale,
                "host_receive_freshness_verified": (
                    joint.host_receive_freshness_verified
                ),
                "error": joint.error,
            }
            for joint in state.joints
            if joint.name in selected
        ),
    }


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
    baseline_state = None
    final_state = None
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
        print(
            "[REPORT] WAIT DECODED TYPE-24 BASELINE",
            file=sys.stderr,
            flush=True,
        )
        baseline_state = _wait_for_host_observation_baseline(
            backend,
            selected,
            timeout_sec=args.decoded_warmup_sec,
        )
        print(
            "[REPORT] CAPTURE RAW TYPE-24 STREAM",
            file=sys.stderr,
            flush=True,
        )
        capture = frequency_probe._read_only_capture(config.channel, args.duration_sec)
        print(
            "[REPORT] READ DECODED TYPE-24 FINAL SNAPSHOT",
            file=sys.stderr,
            flush=True,
        )
        final_state = backend.poll_feedback(require_host_freshness=True)
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
                        cleanup_errors.append(
                            "active-report disable receipt incomplete"
                        )
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
    decoded_acceptance = None
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
        if baseline_state is not None and final_state is not None:
            try:
                decoded_acceptance = evaluate_decoded_active_report_observations(
                    capture,
                    baseline_state,
                    final_state,
                    config.joints,
                    selected,
                    target_hz=args.decoded_target_hz,
                    maximum_receive_age_sec=(
                        config.feedback_timeout_sec
                        if args.decoded_max_receive_age_sec is None
                        else args.decoded_max_receive_age_sec
                    ),
                    minimum_rate_ratio=args.decoded_minimum_rate_ratio,
                    maximum_p99_factor=args.decoded_maximum_p99_factor,
                    minimum_sample_count=args.decoded_minimum_sample_count,
                    minimum_decoded_raw_ratio=args.decoded_minimum_raw_ratio,
                )
            except Exception as exc:  # noqa: BLE001 - evidence remains fail-closed
                errors.append(f"decoded active-report acceptance: {exc}")

    if cleanup_errors:
        errors.extend(f"cleanup: {error}" for error in cleanup_errors)
    execution_status = "ERROR" if errors else "COMPLETED"
    transport_status = (
        "PASS"
        if acceptance is not None and acceptance.passed and not cleanup_errors
        else "FAIL"
    )
    decoded_status = (
        "PASS"
        if decoded_acceptance is not None
        and decoded_acceptance.passed
        and not cleanup_errors
        else "FAIL"
    )
    result: dict[str, object] = {
        "mode": "active-report-telemetry",
        "execution_status": execution_status,
        "stream_transport_acceptance": transport_status,
        "decoded_host_observation_acceptance": decoded_status,
        "control_feedback_admission": "NOT_ADMITTED",
        "hardware_acceptance": "NOT_CLAIMED",
        "channel": config.channel,
        "duration_sec": args.duration_sec,
        "target_hz": args.target_hz,
        "decoded_target_hz": args.decoded_target_hz,
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
        "host_receive_freshness_verified": decoded_status == "PASS",
        "freshness_semantics": (
            "host_receive_sequence_and_age_cross_checked_against_raw_type_24; "
            "device_source_timestamp_unavailable"
            if decoded_status == "PASS"
            else "decoded_host_receive_freshness_not_accepted; "
            "device_source_timestamp_unavailable"
        ),
        "interrupted": interrupted,
        "errors": tuple(errors),
        "cleanup_errors": tuple(cleanup_errors),
        "notes": (
            "This foreground probe configures RobStride active reporting and passively times complete type-24 active-report frames.",
            "It does not enable motors, change control mode, or send position/MIT references.",
            "A transport PASS does not admit q/dq/tau into the control loop and is not haptic hardware acceptance.",
            "Decoded status sequence and receive age are cross-checked against raw type-24 frames from the bracketed foreground interval.",
            "The decoded/raw ratio prevents sparse type-2 traffic from satisfying the type-24 gate.",
            "A decoded PASS proves host receive freshness only; device acquisition time remains unavailable and control feedback is not admitted.",
        ),
    }
    if enable_receipt is not None:
        result["enable_receipt"] = _receipt_payload(enable_receipt)
    if capture is not None:
        result["capture"] = capture
    if acceptance is not None:
        result["acceptance"] = acceptance.as_dict()
    if baseline_state is not None:
        result["decoded_baseline"] = _state_payload(baseline_state, selected)
    if final_state is not None:
        result["decoded_final"] = _state_payload(final_state, selected)
    if decoded_acceptance is not None:
        result["decoded_acceptance"] = decoded_acceptance.as_dict()
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
        "--decoded-warmup-sec",
        type=float,
        default=2.0,
        help="maximum wait for a host-fresh decoded baseline snapshot",
    )
    parser.add_argument(
        "--decoded-target-hz",
        type=float,
        default=40.0,
        help="expected decoded type-24 observation rate",
    )
    parser.add_argument("--decoded-minimum-rate-ratio", type=float, default=0.80)
    parser.add_argument("--decoded-maximum-p99-factor", type=float, default=2.0)
    parser.add_argument("--decoded-minimum-sample-count", type=int, default=10)
    parser.add_argument(
        "--decoded-minimum-raw-ratio",
        type=float,
        default=0.80,
        help="minimum decoded status-sequence delta divided by raw type-24 count",
    )
    parser.add_argument(
        "--decoded-max-receive-age-sec",
        type=float,
        default=None,
        help="override backend feedback timeout for the final receive-age gate",
    )
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
            "decoded_host_observation_acceptance": "FAIL",
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
    elif (
        payload.get("stream_transport_acceptance") != "PASS"
        or payload.get("decoded_host_observation_acceptance") != "PASS"
    ):
        status = "FAIL"
        exit_code = 1
    encoded = json.dumps({"status": status, **payload}, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.expanduser().resolve().write_text(encoded + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
