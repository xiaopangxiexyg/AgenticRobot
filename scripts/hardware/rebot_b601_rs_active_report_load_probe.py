#!/usr/bin/env python3
"""Foreground Type-24 receive and zero-effort MIT transmit load probe.

This is a commissioning-only transport test.  It enables the RobStride
active-report stream, then sends zero-gain/zero-torque MIT frames at a chosen
host rate while every motor remains explicitly disabled.  The receive socket
and transmit loop live in one foreground process so the two measurements cover
the same bus interval without starting two control owners.

The probe deliberately does not enable a motor or admit feedback into a
control loop.  ``motion_commands_sent`` therefore remains false even though
the caller must explicitly opt in to the backend's command boundary with
``--allow-motion``.  A successful result is transport evidence only: it does
not establish state freshness, torque behavior, haptic stability, or hardware
acceptance.
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
from threading import Event, Thread

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.rebot_b601_rs import (
    B601ActiveReportReceipt,
    B601HardwareError,
    B601ModeReceipt,
    B601RSBackend,
    B601RSBackendConfig,
    FrequencyLedger,
    evaluate_active_report_capture,
    evaluate_frequency_target,
)
from robot_motion.rebot_b601_rs import B601MitReference
from scripts.hardware import rebot_b601_rs_frequency_probe as frequency_probe

_CONFIRMATION = "B601_RS_ACTIVE_REPORT_TX_LOAD"
_MAX_TARGET_HZ = 500.0
_MAX_DURATION_SEC = 300.0
_CAPTURE_JOIN_TIMEOUT_SEC = 2.0


class ActiveReportLoadProbeError(RuntimeError):
    """A probe precondition, transport, or cleanup operation failed."""


def _receipt_payload(receipt: object) -> dict[str, object]:
    """Serialize either a mode, active-report, or command receipt."""

    fields = (
        "operation",
        "requested_joint_names",
        "completed_joint_names",
        "sent_joint_names",
        "failed_joint_names",
        "errors",
        "completed",
        "accepted",
        "rollback_attempted",
        "rollback_completed",
        "sequence",
        "source_sequence",
    )
    return {
        field: getattr(receipt, field) for field in fields if hasattr(receipt, field)
    }


def _validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.allow_active_report:
        raise ActiveReportLoadProbeError(
            "active-report load requires --allow-active-report"
        )
    if not args.allow_motion:
        raise ActiveReportLoadProbeError(
            "active-report load requires explicit --allow-motion"
        )
    if args.confirmation != _CONFIRMATION:
        raise ActiveReportLoadProbeError(
            f"active-report load requires --confirmation {_CONFIRMATION}"
        )
    if not args.auto_disable:
        raise ActiveReportLoadProbeError(
            "active-report load requires explicit --auto-disable"
        )
    if not args.auto_disable_active_report:
        raise ActiveReportLoadProbeError(
            "active-report load requires explicit --auto-disable-active-report"
        )

    for name in ("duration_sec", "warmup_sec", "target_tx_hz", "target_report_hz"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if args.duration_sec > _MAX_DURATION_SEC:
        raise ValueError(f"duration_sec must be at most {_MAX_DURATION_SEC:g}")
    for name in ("target_tx_hz", "target_report_hz"):
        if getattr(args, name) > _MAX_TARGET_HZ:
            raise ValueError(f"{name} must be at most {_MAX_TARGET_HZ:g}")
    if args.warmup_sec > _MAX_DURATION_SEC:
        raise ValueError(f"warmup_sec must be at most {_MAX_DURATION_SEC:g}")

    for name in (
        "minimum_tx_rate_ratio",
        "minimum_report_rate_ratio",
        "maximum_tx_deadline_miss_ratio",
        "maximum_report_p99_factor",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0.0 < args.minimum_tx_rate_ratio <= 1.0:
        raise ValueError("minimum_tx_rate_ratio must be in (0, 1]")
    if not 0.0 < args.minimum_report_rate_ratio <= 1.0:
        raise ValueError("minimum_report_rate_ratio must be in (0, 1]")
    if args.maximum_tx_deadline_miss_ratio > 1.0:
        raise ValueError("maximum_tx_deadline_miss_ratio must be at most 1")
    if args.maximum_report_p99_factor <= 0.0:
        raise ValueError("maximum_report_p99_factor must be positive")
    if (
        isinstance(args.minimum_tx_sample_count, bool)
        or args.minimum_tx_sample_count < 2
    ):
        raise ValueError("minimum_tx_sample_count must be at least 2")
    if (
        isinstance(args.minimum_report_sample_count, bool)
        or args.minimum_report_sample_count < 2
    ):
        raise ValueError("minimum_report_sample_count must be at least 2")
    selected = frequency_probe._parse_joint_names(args.joint)
    if (
        args.output is not None
        and args.raw_output is not None
        and args.output.expanduser().resolve() == args.raw_output.expanduser().resolve()
    ):
        raise ValueError("--output and --raw-output must be distinct paths")
    return selected


def _capture_worker(
    channel: str,
    duration_sec: float,
    raw_output: Path | None,
    ready: Event,
    stop: Event,
    capture_holder: dict[str, object],
    capture_errors: list[Exception],
) -> None:
    """Run the existing passive receiver in a joined, non-daemon thread."""

    try:
        capture_holder["capture"] = frequency_probe._read_only_capture(
            channel,
            duration_sec,
            raw_output=raw_output,
            ready=ready,
            stop=stop,
        )
    except Exception as exc:  # noqa: BLE001 - foreground owner reports thread faults
        capture_errors.append(exc)
    finally:
        ready.set()


def _join_capture_worker(
    worker: Thread | None,
    stop: Event,
    cleanup_errors: list[str],
) -> None:
    if worker is None:
        return
    stop.set()
    deadline = time.monotonic() + _CAPTURE_JOIN_TIMEOUT_SEC
    while worker.is_alive() and time.monotonic() < deadline:
        try:
            worker.join(timeout=0.1)
        except KeyboardInterrupt:
            cleanup_errors.append(
                "passive receiver join interrupted; capture thread state is uncertain"
            )
            break
    if worker.is_alive():
        cleanup_errors.append("passive receiver thread did not stop")


def _evaluate_load(
    *,
    capture: dict[str, object] | None,
    ledger: FrequencyLedger,
    config: B601RSBackendConfig,
    selected: tuple[str, ...],
    target_tx_hz: float,
    target_report_hz: float,
    minimum_tx_rate_ratio: float,
    minimum_report_rate_ratio: float,
    maximum_tx_deadline_miss_ratio: float,
    maximum_report_p99_factor: float,
    minimum_tx_sample_count: int,
    minimum_report_sample_count: int,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object] | None,
    bool,
]:
    stats = ledger.summarize(target_hz=target_tx_hz)
    tx_acceptance = {
        event: evaluate_frequency_target(
            stats[event],
            minimum_rate_ratio=minimum_tx_rate_ratio,
            maximum_deadline_miss_ratio=maximum_tx_deadline_miss_ratio,
            minimum_sample_count=minimum_tx_sample_count,
        )
        for event in ("compute", "tx")
    }
    tx_payload = {
        "passed": False,
        "required_events": ("compute", "tx"),
        "minimum_rate_ratio": minimum_tx_rate_ratio,
        "maximum_deadline_miss_ratio": maximum_tx_deadline_miss_ratio,
        "events": {
            event: acceptance.as_dict() for event, acceptance in tx_acceptance.items()
        },
    }
    report_acceptance = None
    if capture is not None:
        report_acceptance = evaluate_active_report_capture(
            capture,
            config.joints,
            selected,
            target_hz=target_report_hz,
            minimum_rate_ratio=minimum_report_rate_ratio,
            maximum_p99_factor=maximum_report_p99_factor,
            minimum_sample_count=minimum_report_sample_count,
        ).as_dict()
    tx_passed = all(acceptance.passed for acceptance in tx_acceptance.values())
    tx_payload["passed"] = tx_passed
    report_passed = bool(report_acceptance and report_acceptance["passed"])
    return (
        {
            "compute": stats["compute"].as_dict(),
            "tx": stats["tx"].as_dict(),
        },
        tx_payload,
        report_acceptance,
        tx_passed and report_passed,
    )


def _run_active_report_load_probe(args: argparse.Namespace) -> dict[str, object]:
    selected = _validate_args(args)
    config_path = Path(args.config).expanduser().resolve(strict=True)
    if not config_path.is_file():
        raise ActiveReportLoadProbeError(
            f"B601-RS config is not a regular file: {config_path}"
        )
    if args.raw_output is not None:
        raw_output = args.raw_output.expanduser().resolve()
        if raw_output.exists():
            raise ActiveReportLoadProbeError(
                f"raw capture output already exists: {raw_output}"
            )
        raw_output.parent.mkdir(parents=True, exist_ok=True)
    else:
        raw_output = None
    position_limits = frequency_probe._resolve_position_limits(
        config_path,
        args.rebot_source_root,
    )
    config = B601RSBackendConfig.from_rs_yaml(
        config_path,
        allow_motion=True,
        allow_active_reporting=True,
        position_limits_rad=position_limits,
    )
    if args.channel is not None:
        config = replace(config, channel=args.channel)

    backend = B601RSBackend(config)
    ledger = FrequencyLedger()
    capture_holder: dict[str, object] = {}
    capture_errors: list[Exception] = []
    capture_ready = Event()
    capture_stop = Event()
    worker: Thread | None = None
    cleanup_required = False
    enable_attempted = False
    interrupted = False
    commands_sent = 0
    schedule_overruns = 0
    initial_positions: tuple[float, ...] | None = None
    initial_sequence: int | None = None
    mit_reference: B601MitReference | None = None
    stop_reason: str | None = None
    primary_errors: list[str] = []
    cleanup_errors: list[str] = []
    enable_receipt: B601ActiveReportReceipt | None = None
    disable_active_report_receipt: B601ActiveReportReceipt | None = None
    preflight_disable_receipt: B601ModeReceipt | None = None
    prepare_mit_receipt: B601ModeReceipt | None = None
    post_mode_disable_receipt: B601ModeReceipt | None = None
    cleanup_receipt: B601ModeReceipt | None = None

    try:
        backend.connect()
        cleanup_required = True
        print(
            "[LOAD] no motor enable; zero-effort MIT TX only; Type-24 RX will be measured",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[LOAD] WARMUP: reading one complete six-joint snapshot (no motion)",
            file=sys.stderr,
            flush=True,
        )
        warmup_deadline = time.monotonic() + args.warmup_sec
        state = None
        while time.monotonic() < warmup_deadline:
            state = backend.poll_feedback(request_feedback=True)
            if state.valid and state.positions_rad is not None:
                break
            time.sleep(0.005)
        if state is None or not state.valid or state.positions_rad is None:
            reason = (
                "no feedback observation"
                if state is None
                else state.fault_reason
                or "snapshot did not contain six valid positions"
            )
            raise ActiveReportLoadProbeError(
                "no complete valid feedback snapshot arrived during warmup; "
                f"last_fault={reason}"
            )
        initial_positions = tuple(state.positions_rad)
        initial_sequence = int(state.sequence)
        mit_reference = B601MitReference(
            sequence=initial_sequence,
            source_timestamp_sec=state.receive_timestamp_sec,
            receive_timestamp_sec=state.receive_timestamp_sec,
            qpos_rad=initial_positions,
        )

        print("[LOAD] PREFLIGHT DISABLE", file=sys.stderr, flush=True)
        preflight_disable_receipt = backend.safe_stop(joint_names=selected)
        if not preflight_disable_receipt.completed:
            raise ActiveReportLoadProbeError(
                "preflight disable failed: "
                + ("; ".join(preflight_disable_receipt.errors) or "incomplete receipt")
            )
        print(
            "[LOAD] MIT MODE SETUP (motors remain disabled)",
            file=sys.stderr,
            flush=True,
        )
        prepare_mit_receipt = backend.prepare_mit_mode(joint_names=selected)
        if not prepare_mit_receipt.completed:
            raise ActiveReportLoadProbeError(
                "MIT mode setup failed: "
                + ("; ".join(prepare_mit_receipt.errors) or "incomplete receipt")
            )
        print("[LOAD] POST-SETUP DISABLE", file=sys.stderr, flush=True)
        post_mode_disable_receipt = backend.safe_stop(joint_names=selected)
        if not post_mode_disable_receipt.completed:
            raise ActiveReportLoadProbeError(
                "post-setup disable failed: "
                + ("; ".join(post_mode_disable_receipt.errors) or "incomplete receipt")
            )

        print("[LOAD] 3", file=sys.stderr, flush=True)
        time.sleep(1.0)
        print("[LOAD] 2", file=sys.stderr, flush=True)
        time.sleep(1.0)
        print("[LOAD] 1", file=sys.stderr, flush=True)
        time.sleep(1.0)
        print("[LOAD] ENABLE TYPE-24 STREAM", file=sys.stderr, flush=True)
        enable_attempted = True
        enable_receipt = backend.set_active_reporting(True, joint_names=selected)
        if not enable_receipt.completed:
            raise ActiveReportLoadProbeError(
                "active-report enable failed: "
                + ("; ".join(enable_receipt.errors) or "incomplete receipt")
            )

        raw_capture_duration = args.duration_sec + 1.0
        worker = Thread(
            target=_capture_worker,
            args=(
                config.channel,
                raw_capture_duration,
                raw_output,
                capture_ready,
                capture_stop,
                capture_holder,
                capture_errors,
            ),
            name="b601-active-report-rx",
            daemon=False,
        )
        worker.start()
        if not capture_ready.wait(2.0):
            raise ActiveReportLoadProbeError(
                "active-report receiver did not become ready"
            )
        if capture_errors:
            raise ActiveReportLoadProbeError(
                f"active-report receiver failed: {capture_errors[0]}"
            )
        print(
            f"[LOAD] TX START: zero-effort MIT at {args.target_tx_hz:g} Hz; motors disabled",
            file=sys.stderr,
            flush=True,
        )
        period = 1.0 / args.target_tx_hz
        started = time.monotonic()
        next_deadline = started
        while time.monotonic() - started < args.duration_sec:
            now = time.monotonic()
            if now < next_deadline:
                time.sleep(next_deadline - now)
            if capture_errors:
                stop_reason = f"active-report receiver failed: {capture_errors[0]}"
                break
            if worker is not None and not worker.is_alive():
                stop_reason = "active-report receiver stopped during TX load"
                break
            if mit_reference is None:
                raise ActiveReportLoadProbeError("MIT reference was not initialized")
            ledger.mark("compute")
            receipt = backend.send_mit(
                mit_reference,
                kp=(0.0,) * 6,
                kd=(0.0,) * 6,
                tau_nm=(0.0,) * 6,
                torque_limits_nm=(0.0,) * 6,
                joint_names=selected,
            )
            if not receipt.accepted:
                stop_reason = "; ".join(receipt.errors) or "partial MIT dispatch"
                break
            ledger.mark("tx", receipt.completed_at_sec)
            commands_sent += 1
            next_deadline += period
            completed_at = time.monotonic()
            if completed_at > next_deadline:
                schedule_overruns += 1
                next_deadline = completed_at
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "operator_interrupt"
    except Exception as exc:  # noqa: BLE001 - cleanup must always run
        primary_errors.append(str(exc))
    finally:
        _join_capture_worker(worker, capture_stop, cleanup_errors)
        if backend.connected and enable_attempted:
            try:
                disable_active_report_receipt = backend.set_active_reporting(
                    False,
                    joint_names=selected,
                )
                if not disable_active_report_receipt.completed:
                    cleanup_errors.extend(disable_active_report_receipt.errors)
                    if not disable_active_report_receipt.errors:
                        cleanup_errors.append(
                            "active-report disable receipt incomplete"
                        )
            except Exception as exc:  # noqa: BLE001 - report cleanup failure
                cleanup_errors.append(f"disable active reporting: {exc}")
        if backend.connected and cleanup_required:
            try:
                cleanup_receipt = backend.safe_stop(joint_names=selected)
                if not cleanup_receipt.completed:
                    cleanup_errors.extend(
                        f"safe_stop: {error}" for error in cleanup_receipt.errors
                    )
                    if not cleanup_receipt.errors:
                        cleanup_errors.append("safe_stop receipt incomplete")
            except Exception as exc:  # noqa: BLE001 - report cleanup failure
                cleanup_errors.append(f"safe_stop: {exc}")
        if backend.connected:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - report cleanup failure
                cleanup_errors.append(f"close: {exc}")

    if capture_errors:
        primary_errors.extend(f"receiver: {error}" for error in capture_errors)
    capture = capture_holder.get("capture")
    capture_payload = capture if isinstance(capture, dict) else None
    frequency, tx_acceptance, report_acceptance, clocks_passed = _evaluate_load(
        capture=capture_payload,
        ledger=ledger,
        config=config,
        selected=selected,
        target_tx_hz=args.target_tx_hz,
        target_report_hz=args.target_report_hz,
        minimum_tx_rate_ratio=args.minimum_tx_rate_ratio,
        minimum_report_rate_ratio=args.minimum_report_rate_ratio,
        maximum_tx_deadline_miss_ratio=args.maximum_tx_deadline_miss_ratio,
        maximum_report_p99_factor=args.maximum_report_p99_factor,
        minimum_tx_sample_count=args.minimum_tx_sample_count,
        minimum_report_sample_count=args.minimum_report_sample_count,
    )
    if cleanup_errors:
        primary_errors.extend(f"cleanup: {error}" for error in cleanup_errors)
    execution_status = (
        "INTERRUPTED" if interrupted else "ERROR" if primary_errors else "COMPLETED"
    )
    acceptance_passed = (
        execution_status == "COMPLETED"
        and stop_reason is None
        and not cleanup_errors
        and clocks_passed
    )
    result: dict[str, object] = {
        "mode": "active-report-tx-load",
        "execution_status": execution_status,
        "acceptance_status": "PASS" if acceptance_passed else "FAIL",
        "stream_transport_acceptance": (
            "PASS"
            if report_acceptance is not None and report_acceptance["passed"]
            else "FAIL"
        ),
        "tx_acceptance": "PASS" if tx_acceptance["passed"] else "FAIL",
        "tx_frequency_acceptance": tx_acceptance,
        "control_feedback_admission": "NOT_ADMITTED",
        "hardware_acceptance": "NOT_CLAIMED",
        "channel": config.channel,
        "duration_sec": args.duration_sec,
        "target_tx_hz": args.target_tx_hz,
        "target_report_hz": args.target_report_hz,
        "selected_joint_names": selected,
        "selected_motor_ids": tuple(
            joint.motor_id for joint in config.joints if joint.name in selected
        ),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "rebot_source_root": str(
            config_path.parent.parent
            if args.rebot_source_root is None
            else args.rebot_source_root.expanduser().resolve(strict=True)
        ),
        "motion_commands_sent": False,
        "tx_commands_sent": commands_sent > 0,
        "tx_commands_sent_count": commands_sent,
        "tx_joint_frames_sent_count": commands_sent * len(selected),
        "tx_mode": "mit-zero-disabled",
        "motors_enabled_by_probe": False,
        "active_report_commands_sent": enable_attempted,
        "stop_reason": stop_reason,
        "schedule_overruns": schedule_overruns,
        "initial_qpos_rad": initial_positions,
        "initial_sequence": initial_sequence,
        "freshness_verified": False,
        "freshness_semantics": (
            "initial_vendor_snapshot_only; type-24 socket-receive timing; "
            "device-source timestamp unavailable"
        ),
        "motorbridge_state_observation_metadata_available": (
            backend.state_observation_metadata_verified
        ),
        "errors": tuple(primary_errors),
        "cleanup_errors": tuple(cleanup_errors),
        "frequency": frequency,
        "active_report_acceptance": report_acceptance,
        "notes": (
            "Foreground single-process load probe: Type-24 active-report RX and zero-effort MIT TX share one bus interval.",
            "Zero kp, kd, tau, and torque limits are sent; no motor enable is issued and no physical motion is requested.",
            "A PASS measures host dispatch and complete type-24 receive cadence only; it does not admit feedback or prove haptic/control stability.",
            "The passive receiver is a joined non-daemon thread and is stopped before active-report disable and final safe_stop.",
        ),
    }
    if enable_receipt is not None:
        result["active_report_enable_receipt"] = _receipt_payload(enable_receipt)
    if disable_active_report_receipt is not None:
        result["active_report_disable_receipt"] = _receipt_payload(
            disable_active_report_receipt
        )
    if preflight_disable_receipt is not None:
        result["preflight_disable_receipt"] = _receipt_payload(
            preflight_disable_receipt
        )
    if prepare_mit_receipt is not None:
        result["prepare_mit_mode_receipt"] = _receipt_payload(prepare_mit_receipt)
    if post_mode_disable_receipt is not None:
        result["post_mode_disable_receipt"] = _receipt_payload(
            post_mode_disable_receipt
        )
    if cleanup_receipt is not None:
        result["cleanup_receipt"] = _receipt_payload(cleanup_receipt)
    if capture_payload is not None:
        result["capture"] = capture_payload
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Foreground B601-RS Type-24 active-report RX plus zero-effort MIT TX load probe."
        )
    )
    parser.add_argument("--config", required=True, help="explicit rebotarm_rs.yaml")
    parser.add_argument(
        "--rebot-source-root",
        type=Path,
        required=True,
        help="reBotArm source root containing the pinned URDF",
    )
    parser.add_argument("--channel", default=None)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--target-tx-hz", type=float, default=125.0)
    parser.add_argument("--target-report-hz", type=float, default=40.0)
    parser.add_argument("--minimum-tx-rate-ratio", type=float, default=0.90)
    parser.add_argument("--minimum-report-rate-ratio", type=float, default=0.90)
    parser.add_argument("--maximum-tx-deadline-miss-ratio", type=float, default=0.01)
    parser.add_argument("--maximum-report-p99-factor", type=float, default=2.0)
    parser.add_argument("--minimum-tx-sample-count", type=int, default=10)
    parser.add_argument("--minimum-report-sample-count", type=int, default=100)
    parser.add_argument(
        "--joint",
        action="append",
        help="joint name or comma-separated names; default is all six arm joints",
    )
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--allow-active-report", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--auto-disable", action="store_true")
    parser.add_argument("--auto-disable-active-report", action="store_true")
    parser.add_argument("--output", type=Path, help="write JSON outside the worktree")
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="write raw SocketCAN JSONL outside the worktree; never overwrites",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = _run_active_report_load_probe(args)
    except (ActiveReportLoadProbeError, B601HardwareError, OSError, ValueError) as exc:
        payload = {
            "mode": "active-report-tx-load",
            "execution_status": "ERROR",
            "acceptance_status": "FAIL",
            "stream_transport_acceptance": "FAIL",
            "tx_acceptance": "FAIL",
            "control_feedback_admission": "NOT_ADMITTED",
            "hardware_acceptance": "NOT_CLAIMED",
            "motion_commands_sent": False,
            "motors_enabled_by_probe": False,
            "error": str(exc),
        }
    status = "OK"
    exit_code = 0
    if payload.get("execution_status") == "INTERRUPTED":
        status = "ERROR"
        exit_code = 130
    elif payload.get("execution_status") == "ERROR":
        status = "ERROR"
        exit_code = 2
    elif payload.get("acceptance_status") != "PASS":
        status = "FAIL"
        exit_code = 1
    encoded = json.dumps({"status": status, **payload}, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
