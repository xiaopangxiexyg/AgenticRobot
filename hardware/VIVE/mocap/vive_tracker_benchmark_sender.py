"""Progressively benchmark the VIVE tracker -> UDP sender pipeline.

Run this on the computer that has SteamVR/OpenVR.  It sends a separate
benchmark protocol and never sends robot commands.  Start the matching
``vive_tracker_benchmark_receiver.py`` first.
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

try:  # Supports both ``python path/to/script.py`` and ``python -m ...``.
    from .vive_data_sender import (
        TrackerSpec,
        auto_tracker_specs,
        load_triad_openvr,
        missing_tracker_names,
        openvr_pose_to_matrix4x4,
        tracker_mapping_text,
        wait_for_tracker_pair,
    )
except ImportError:  # pragma: no cover - exercised when launched as a file.
    from vive_data_sender import (  # type: ignore[no-redef]
        TrackerSpec,
        auto_tracker_specs,
        load_triad_openvr,
        missing_tracker_names,
        openvr_pose_to_matrix4x4,
        tracker_mapping_text,
        wait_for_tracker_pair,
    )


PROTOCOL = "dexteleop.vive_tracker_benchmark/v1"
DEFAULT_RATES_HZ = (30.0, 60.0, 90.0, 100.0, 120.0, 144.0, 180.0, 200.0, 240.0)


@dataclass
class StageMetrics:
    stage_id: int
    target_hz: float
    started_at: float
    measurement_duration_sec: float = 0.0
    sent_frames: int = 0
    valid_frames: int = 0
    sample_intervals_ms: list[float] = field(default_factory=list)
    get_pose_ms: list[float] = field(default_factory=list)
    tracker_read_ms: list[float] = field(default_factory=list)
    sample_to_send_ms: list[float] = field(default_factory=list)
    rtt_ms: list[float] = field(default_factory=list)
    acknowledged_sequences: set[int] = field(default_factory=set)
    pending_sent_at: dict[int, float] = field(default_factory=dict)
    receiver_summary: dict[str, Any] | None = None
    last_sample_started_at: float | None = None
    tracker_error_counts: dict[str, int] = field(default_factory=dict)
    last_tracker_errors: dict[str, str] = field(default_factory=dict)
    clock_offset_sec: float | None = None
    clock_sync_rtt_ms: float | None = None
    clock_sync_sample_count: int = 0


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return parsed


def parse_rate_list(value: str) -> tuple[float, ...]:
    try:
        rates = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rates must be comma-separated numbers") from exc
    if not rates or any(rate <= 0.0 for rate in rates):
        raise argparse.ArgumentTypeError("provide one or more rates greater than zero")
    return rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step through VIVE read/send rates and measure sender rate, UDP ACK RTT, "
            "and the receiver's delivery result. This program does not control a robot."
        )
    )
    parser.add_argument("--host", required=True, help="benchmark receiver IP or hostname")
    parser.add_argument("--port", type=int, default=10002, help="benchmark receiver UDP port")
    parser.add_argument(
        "--rates",
        type=parse_rate_list,
        default=DEFAULT_RATES_HZ,
        help="comma-separated target rates, low to high (default: 30,60,90,100,120,144,180,200,240)",
    )
    parser.add_argument(
        "--stage-sec",
        type=positive_float,
        default=12.0,
        help="measurement duration for every rate (default: 12)",
    )
    parser.add_argument(
        "--ack-wait-sec",
        type=float,
        default=1.0,
        help="extra time to collect delayed ACKs and receiver summary (default: 1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="optional triad_openvr config.json path",
    )
    parser.add_argument(
        "--wait-for-trackers",
        type=float,
        default=10.0,
        help="seconds to wait for requested trackers at startup (default: 10; 0 disables)",
    )
    parser.add_argument(
        "--min-rate-ratio",
        type=ratio,
        default=0.95,
        help="sender/receiver rate must reach this fraction of target (default: 0.95)",
    )
    parser.add_argument(
        "--min-ack-ratio",
        type=ratio,
        default=0.99,
        help="fraction of sent packets that must receive an ACK (default: 0.99)",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=ratio,
        default=1.0,
        help="fraction of frames with every tracker valid (default: 1.0)",
    )
    parser.add_argument(
        "--max-rtt-p95-ms",
        type=float,
        default=0.0,
        help="optional RTT p95 stability limit; <=0 disables the limit (default: disabled)",
    )
    parser.add_argument(
        "--clock-sync-samples",
        type=non_negative_int,
        default=12,
        help=(
            "NTP-style clock-sync probes before each stage for one-way latency "
            "estimation; use 0 to disable (default: 12)"
        ),
    )
    parser.add_argument(
        "--clock-sync-timeout-sec",
        type=positive_float,
        default=0.25,
        help="maximum wait for each clock-sync response (default: 0.25)",
    )
    return parser.parse_args()


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * (percent / 100.0))
    return ordered[index]


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}ms"


def fmt_hz(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}Hz"


def drain_receiver_messages(
    sock: socket.socket,
    *,
    run_id: str,
    stages: dict[int, StageMetrics],
    timeout_sec: float = 0.0,
) -> None:
    """Consume ACKs without ever blocking the OpenVR sampling loop."""

    first = True
    while True:
        wait_sec = max(0.0, timeout_sec) if first else 0.0
        first = False
        readable, _, _ = select.select([sock], [], [], wait_sec)
        if not readable:
            return
        try:
            raw = sock.recv(65536)
        except BlockingIOError:
            return
        try:
            message = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(message, dict) or message.get("protocol") != PROTOCOL:
            continue
        if message.get("run_id") != run_id:
            continue
        try:
            stage = stages[int(message["stage_id"])]
        except (KeyError, TypeError, ValueError):
            continue

        message_type = message.get("type")
        if message_type == "ack":
            try:
                sequence = int(message["seq"])
            except (KeyError, TypeError, ValueError):
                continue
            sent_at = stage.pending_sent_at.pop(sequence, None)
            if sent_at is not None and sequence not in stage.acknowledged_sequences:
                stage.acknowledged_sequences.add(sequence)
                stage.rtt_ms.append((time.perf_counter() - sent_at) * 1000.0)
        elif message_type == "stage_summary":
            summary = message.get("summary")
            if isinstance(summary, dict):
                stage.receiver_summary = summary


def calculate_clock_offset(
    t1_sender_sent: float,
    t2_receiver_received: float,
    t3_receiver_sent: float,
    t4_sender_received: float,
) -> tuple[float, float]:
    """Return receiver-minus-sender clock offset and network RTT in seconds."""

    offset_sec = (
        (t2_receiver_received - t1_sender_sent)
        + (t3_receiver_sent - t4_sender_received)
    ) / 2.0
    rtt_sec = (t4_sender_received - t1_sender_sent) - (
        t3_receiver_sent - t2_receiver_received
    )
    return offset_sec, max(0.0, rtt_sec)


def estimate_clock_offset(
    sock: socket.socket,
    *,
    run_id: str,
    samples: int,
    timeout_sec: float,
) -> tuple[float | None, float | None, int]:
    """Estimate direct-link clock offset with the lowest-RTT NTP-style probe."""

    probes: list[tuple[float, float]] = []  # (rtt_sec, receiver_minus_sender_sec)
    for _ in range(max(0, samples)):
        sync_id = uuid.uuid4().hex
        t1 = time.time()
        try:
            sock.send(
                json.dumps(
                    {
                        "protocol": PROTOCOL,
                        "type": "clock_sync_request",
                        "run_id": run_id,
                        "sync_id": sync_id,
                        "t1_sender_sent": t1,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        except OSError:
            continue
        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            readable, _, _ = select.select([sock], [], [], deadline - time.perf_counter())
            if not readable:
                break
            try:
                raw = sock.recv(65536)
            except BlockingIOError:
                break
            t4 = time.time()
            try:
                response = json.loads(raw.decode("utf-8"))
                if (
                    not isinstance(response, dict)
                    or response.get("protocol") != PROTOCOL
                    or response.get("type") != "clock_sync_response"
                    or response.get("run_id") != run_id
                    or response.get("sync_id") != sync_id
                ):
                    continue
                t2 = float(response["t2_receiver_received"])
                t3 = float(response["t3_receiver_sent"])
            except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            offset_sec, rtt_sec = calculate_clock_offset(t1, t2, t3, t4)
            probes.append((rtt_sec, offset_sec))
            break
    if not probes:
        return None, None, 0
    best_rtt_sec, best_offset_sec = min(probes, key=lambda probe: probe[0])
    return best_offset_sec, best_rtt_sec, len(probes)


def read_trackers(
    vive: Any,
    specs: list[TrackerSpec],
) -> tuple[list[dict[str, Any]], float, list[float]]:
    get_pose_start = time.perf_counter()
    try:
        snapshot = vive.get_pose()
        snapshot_error: str | None = None
    except Exception as exc:  # OpenVR errors differ across driver versions.
        snapshot = None
        snapshot_error = f"{type(exc).__name__}: {exc}"
    get_pose_elapsed_ms = (time.perf_counter() - get_pose_start) * 1000.0

    trackers: list[dict[str, Any]] = []
    tracker_read_ms: list[float] = []
    for spec in specs:
        tracker_started_at = time.perf_counter()
        try:
            if snapshot_error is not None:
                raise RuntimeError(snapshot_error)
            device = vive.devices[spec.device_name]
            pose = openvr_pose_to_matrix4x4(device.get_pose_matrix(snapshot))
            valid = True
            error: str | None = None
        except Exception as exc:
            pose = None
            valid = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - tracker_started_at) * 1000.0
        tracker_read_ms.append(elapsed_ms)
        trackers.append(
            {
                "tracker_name": spec.device_name,
                "tracker_role": spec.tracker_role,
                "valid": valid,
                "pose_matrix_4x4": pose,
                "error": error,
                "read_elapsed_ms": elapsed_ms,
            }
        )
    return trackers, get_pose_elapsed_ms, tracker_read_ms


def receiver_is_stable(summary: dict[str, Any] | None) -> tuple[bool, str]:
    if summary is None:
        return False, "receiver summary not received"
    if bool(summary.get("stable")):
        return True, "ok"
    reason = summary.get("reason")
    return False, str(reason) if reason else "receiver rejected stage"


def stage_sender_stability(
    stage: StageMetrics,
    elapsed_sec: float,
    args: argparse.Namespace,
) -> tuple[bool, str, float, float, float]:
    actual_hz = stage.sent_frames / max(elapsed_sec, 1e-9)
    ack_ratio = len(stage.acknowledged_sequences) / max(stage.sent_frames, 1)
    valid_ratio = stage.valid_frames / max(stage.sent_frames, 1)
    checks = [
        (
            actual_hz >= stage.target_hz * args.min_rate_ratio,
            f"sender rate {actual_hz:.1f}Hz < {args.min_rate_ratio * stage.target_hz:.1f}Hz",
        ),
        (
            ack_ratio >= args.min_ack_ratio,
            f"ACK ratio {ack_ratio:.3f} < {args.min_ack_ratio:.3f}",
        ),
        (
            valid_ratio >= args.min_valid_ratio,
            f"valid-frame ratio {valid_ratio:.3f} < {args.min_valid_ratio:.3f}",
        ),
    ]
    rtt_p95 = percentile(stage.rtt_ms, 95.0)
    if args.max_rtt_p95_ms > 0.0:
        checks.append(
            (
                rtt_p95 is not None and rtt_p95 <= args.max_rtt_p95_ms,
                f"RTT p95 {fmt_ms(rtt_p95)} > {args.max_rtt_p95_ms:.2f}ms",
            )
        )
    failed_reasons = [reason for passed, reason in checks if not passed]
    return not failed_reasons, "; ".join(failed_reasons) or "ok", actual_hz, ack_ratio, valid_ratio


def run_stage(
    sock: socket.socket,
    vive: Any,
    specs: list[TrackerSpec],
    *,
    run_id: str,
    stage_id: int,
    target_hz: float,
    sequence: int,
    stages: dict[int, StageMetrics],
    args: argparse.Namespace,
) -> tuple[StageMetrics, int]:
    stage = StageMetrics(stage_id=stage_id, target_hz=target_hz, started_at=time.perf_counter())
    (
        stage.clock_offset_sec,
        clock_sync_rtt_sec,
        stage.clock_sync_sample_count,
    ) = estimate_clock_offset(
        sock,
        run_id=run_id,
        samples=args.clock_sync_samples,
        timeout_sec=args.clock_sync_timeout_sec,
    )
    if clock_sync_rtt_sec is not None:
        stage.clock_sync_rtt_ms = clock_sync_rtt_sec * 1000.0
    stage.started_at = time.perf_counter()
    stages[stage_id] = stage
    sock.send(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "type": "stage_start",
                "run_id": run_id,
                "stage_id": stage_id,
                "target_hz": target_hz,
                "started_timestamp": time.time(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    print(f"\n[Stage {stage_id}] target={target_hz:g}Hz duration={args.stage_sec:g}s")

    period_sec = 1.0 / target_hz
    deadline = stage.started_at + args.stage_sec
    next_sample_at = stage.started_at
    while True:
        drain_receiver_messages(sock, run_id=run_id, stages=stages)
        now = time.perf_counter()
        if now >= deadline:
            break
        if now < next_sample_at:
            time.sleep(min(next_sample_at - now, 0.002))
            continue

        sample_started_at = time.perf_counter()
        # This is an application-level sampling timestamp: it intentionally
        # includes the following get_pose() call and tracker extraction.
        sample_timestamp = time.time()
        if stage.last_sample_started_at is not None:
            stage.sample_intervals_ms.append(
                (sample_started_at - stage.last_sample_started_at) * 1000.0
            )
        stage.last_sample_started_at = sample_started_at
        trackers, get_pose_elapsed_ms, tracker_read_ms = read_trackers(vive, specs)
        for tracker in trackers:
            if tracker["valid"]:
                continue
            tracker_label = f"{tracker['tracker_name']}:{tracker['tracker_role']}"
            stage.tracker_error_counts[tracker_label] = (
                stage.tracker_error_counts.get(tracker_label, 0) + 1
            )
            stage.last_tracker_errors[tracker_label] = str(tracker.get("error") or "unknown error")
        send_timestamp = time.time()
        message = {
            "protocol": PROTOCOL,
            "type": "data",
            "run_id": run_id,
            "stage_id": stage_id,
            "target_hz": target_hz,
            "seq": sequence,
            "sample_timestamp": sample_timestamp,
            "send_timestamp": send_timestamp,
            "sample_to_send_elapsed_ms": (send_timestamp - sample_timestamp) * 1000.0,
            "clock_offset_sec": stage.clock_offset_sec,
            "clock_sync_rtt_ms": stage.clock_sync_rtt_ms,
            "get_pose_elapsed_ms": get_pose_elapsed_ms,
            "tracker_read_elapsed_ms": tracker_read_ms,
            "trackers": trackers,
        }
        try:
            payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
            sent_at = time.perf_counter()
            sock.send(payload)
        except (OSError, TypeError, ValueError) as exc:
            print(f"[Warn] stage={stage_id} seq={sequence} send skipped: {type(exc).__name__}: {exc}")
        else:
            stage.sent_frames += 1
            stage.valid_frames += int(all(bool(item["valid"]) for item in trackers))
            stage.get_pose_ms.append(get_pose_elapsed_ms)
            stage.tracker_read_ms.extend(tracker_read_ms)
            stage.sample_to_send_ms.append((sent_at - sample_started_at) * 1000.0)
            stage.pending_sent_at[sequence] = sent_at
            sequence += 1
        # Keep an absolute schedule.  A short get_pose()/serialization overrun
        # is caught up by the next iteration instead of being added to every
        # period.  If the target is above the pipeline capacity, this naturally
        # becomes a back-to-back max-rate loop and the measured rate reveals it.
        next_sample_at += period_sec

    elapsed_sec = time.perf_counter() - stage.started_at
    stage.measurement_duration_sec = elapsed_sec
    sock.send(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "type": "stage_end",
                "run_id": run_id,
                "stage_id": stage_id,
                "target_hz": target_hz,
                "sent_frames": stage.sent_frames,
                "valid_frames": stage.valid_frames,
                "stage_duration_sec": elapsed_sec,
                "ended_timestamp": time.time(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    wait_deadline = time.perf_counter() + max(0.0, args.ack_wait_sec)
    while time.perf_counter() < wait_deadline:
        drain_receiver_messages(
            sock,
            run_id=run_id,
            stages=stages,
            timeout_sec=min(0.05, wait_deadline - time.perf_counter()),
        )
        if stage.receiver_summary is not None and not stage.pending_sent_at:
            break

    sender_ok, sender_reason, actual_hz, ack_ratio, valid_ratio = stage_sender_stability(
        stage, elapsed_sec, args
    )
    receiver_ok, receiver_reason = receiver_is_stable(stage.receiver_summary)
    stable = sender_ok and receiver_ok
    rtt_p50 = percentile(stage.rtt_ms, 50.0)
    rtt_p95 = percentile(stage.rtt_ms, 95.0)
    print(
        f"[Sender] target={target_hz:g}Hz actual={actual_hz:.1f}Hz "
        f"ack={ack_ratio:.3f} valid={valid_ratio:.3f} "
        f"rtt(p50/p95)={fmt_ms(rtt_p50)}/{fmt_ms(rtt_p95)} "
        f"get_pose_p95={fmt_ms(percentile(stage.get_pose_ms, 95.0))} "
        f"clock_sync={'n/a' if stage.clock_sync_rtt_ms is None else f'{stage.clock_sync_rtt_ms:.2f}ms/{stage.clock_sync_sample_count}'} "
        f"stable={'YES' if stable else 'NO'}"
    )
    if not stable:
        reasons = [reason for ok, reason in ((sender_ok, sender_reason), (receiver_ok, receiver_reason)) if not ok]
        print(f"[Sender] unstable: {'; '.join(reasons)}")
    if stage.tracker_error_counts:
        error_text = "; ".join(
            f"{tracker} invalid={count} last_error={stage.last_tracker_errors[tracker]}"
            for tracker, count in sorted(stage.tracker_error_counts.items())
        )
        print(f"[Sender] tracker diagnostics: {error_text}")
    return stage, sequence


def main() -> None:
    args = parse_args()
    triad_openvr_module = load_triad_openvr()
    vive = triad_openvr_module.triad_openvr(args.config) if args.config else triad_openvr_module.triad_openvr()
    vive.print_discovered_objects()

    try:
        if args.wait_for_trackers > 0.0:
            tracker_specs = wait_for_tracker_pair(vive, args.wait_for_trackers)
            if tracker_specs is None:
                raise ValueError(
                    f"did not find exactly 2 GenericTrackers within {args.wait_for_trackers:.1f}s"
                )
        else:
            tracker_specs = auto_tracker_specs(vive)
    except ValueError as exc:
        raise SystemExit(f"[Error] {exc}") from exc

    missing = missing_tracker_names(vive, tracker_specs)
    if missing:
        message = f"missing trackers: {', '.join(missing)}"
        print(f"[Warn] {message}; invalid frames make a stage unstable.")

    rates = tuple(args.rates)
    if tuple(sorted(rates)) != rates:
        print("[Warn] --rates is not ascending; the final result uses the highest stable target.")
    run_id = uuid.uuid4().hex[:12]
    print(
        f"[Start] run_id={run_id} receiver={args.host}:{args.port} "
        f"trackers={tracker_mapping_text(tracker_specs)} rates={','.join(f'{rate:g}' for rate in rates)}Hz"
    )
    print(
        "[Info] RTT is clock-independent. Before every stage the sender also performs "
        "NTP-style probes over the direct link, so the receiver can estimate one-way latency."
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    try:
        sock.connect((args.host, args.port))
    except OSError as exc:
        sock.close()
        raise SystemExit(f"[Error] unable to reach receiver {args.host}:{args.port}: {exc}") from exc

    stages: dict[int, StageMetrics] = {}
    stable_stages: list[StageMetrics] = []
    sequence = 0
    try:
        for stage_id, target_hz in enumerate(rates, start=1):
            stage, sequence = run_stage(
                sock,
                vive,
                tracker_specs,
                run_id=run_id,
                stage_id=stage_id,
                target_hz=target_hz,
                sequence=sequence,
                stages=stages,
                args=args,
            )
            sender_ok, _, _, _, _ = stage_sender_stability(
                stage, stage.measurement_duration_sec, args
            )
            receiver_ok, _ = receiver_is_stable(stage.receiver_summary)
            if sender_ok and receiver_ok:
                stable_stages.append(stage)
    except KeyboardInterrupt:
        print("\n[Exit] benchmark stopped by user")
    finally:
        sock.close()

    if stable_stages:
        best = max(stable_stages, key=lambda stage: stage.target_hz)
        print(
            f"\n[Result] highest jointly stable target={best.target_hz:g}Hz "
            f"(sender actual={best.sent_frames / max(best.measurement_duration_sec, 1e-9):.1f}Hz, "
            f"RTT p95={fmt_ms(percentile(best.rtt_ms, 95.0))})."
        )
    else:
        print("\n[Result] no jointly stable stage. Check the sender and receiver rejection reasons above.")


if __name__ == "__main__":
    main()
