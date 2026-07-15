"""Receive the VIVE tracker benchmark stream and report delivery/latency.

Start this program before ``hardware/VIVE/mocap/vive_tracker_benchmark_sender.py``.
It only receives UDP benchmark packets and sends UDP acknowledgements; it does
not initialise teleoperation hardware or issue motion commands.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass, field
from typing import Any


PROTOCOL = "dexteleop.vive_tracker_benchmark/v1"


@dataclass
class StageState:
    run_id: str
    stage_id: int
    target_hz: float
    received_sequences: set[int] = field(default_factory=set)
    received_at_monotonic: list[float] = field(default_factory=list)
    valid_frames: int = 0
    duplicate_packets: int = 0
    sample_to_receiver_ms: list[float] = field(default_factory=list)
    send_to_receiver_ms: list[float] = field(default_factory=list)
    sender_get_pose_ms: list[float] = field(default_factory=list)
    sender_tracker_read_ms: list[float] = field(default_factory=list)
    sender_sample_to_send_ms: list[float] = field(default_factory=list)
    clock_sync_packets: int = 0
    clock_sync_rtt_ms: float | None = None
    sent_frames: int | None = None
    valid_sent_frames: int | None = None
    stage_duration_sec: float | None = None
    finished: bool = False
    summary: dict[str, Any] | None = None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Receive the standalone VIVE tracker benchmark stream and report the "
            "highest stable delivered rate. No robot hardware is accessed."
        )
    )
    parser.add_argument("--host", default="0.0.0.0", help="local bind host (default: all interfaces)")
    parser.add_argument("--port", type=int, default=10002, help="local UDP port (default: 10002)")
    parser.add_argument(
        "--min-rate-ratio",
        type=ratio,
        default=0.95,
        help="received rate must reach this fraction of target (default: 0.95)",
    )
    parser.add_argument(
        "--min-delivery-ratio",
        type=ratio,
        default=0.99,
        help="received/sent packet fraction required for stability (default: 0.99)",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=ratio,
        default=1.0,
        help="fraction of received frames with every tracker valid (default: 1.0)",
    )
    parser.add_argument(
        "--assume-clock-synced",
        action="store_true",
        help=(
            "fall back to system-clock one-way latency if sender clock sync is disabled. "
            "Use only when sender and receiver clocks are synchronized with NTP/PTP."
        ),
    )
    parser.add_argument(
        "--max-datagram-size",
        type=positive_float,
        default=65536,
        help="maximum UDP datagram size in bytes (default: 65536)",
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


def get_stage(
    stages: dict[tuple[str, int], StageState],
    *,
    run_id: str,
    stage_id: int,
    target_hz: float,
) -> StageState:
    key = (run_id, stage_id)
    stage = stages.get(key)
    if stage is None:
        stage = StageState(run_id=run_id, stage_id=stage_id, target_hz=target_hz)
        stages[key] = stage
    return stage


def interarrival_stats(arrivals: list[float]) -> tuple[float | None, float | None, float | None]:
    if len(arrivals) < 2:
        return None, None, None
    gaps_ms = [
        (current - previous) * 1000.0
        for previous, current in zip(arrivals, arrivals[1:])
    ]
    return percentile(gaps_ms, 50.0), percentile(gaps_ms, 95.0), percentile(gaps_ms, 99.0)


def finalise_stage(stage: StageState, args: argparse.Namespace) -> dict[str, Any]:
    received = len(stage.received_sequences)
    expected = stage.sent_frames
    duration_sec = stage.stage_duration_sec
    if duration_sec is not None and duration_sec > 0.0:
        receiver_hz = received / duration_sec
    elif len(stage.received_at_monotonic) >= 2:
        receiver_hz = (received - 1) / max(
            stage.received_at_monotonic[-1] - stage.received_at_monotonic[0], 1e-9
        )
    else:
        receiver_hz = 0.0
    delivery_ratio = received / expected if expected else 0.0
    valid_ratio = stage.valid_frames / received if received else 0.0
    loss = max(0, expected - received) if expected is not None else None
    gap_p50, gap_p95, gap_p99 = interarrival_stats(stage.received_at_monotonic)
    checks = [
        (
            receiver_hz >= stage.target_hz * args.min_rate_ratio,
            f"receiver rate {receiver_hz:.1f}Hz < {args.min_rate_ratio * stage.target_hz:.1f}Hz",
        ),
        (
            delivery_ratio >= args.min_delivery_ratio,
            f"delivery ratio {delivery_ratio:.3f} < {args.min_delivery_ratio:.3f}",
        ),
        (
            valid_ratio >= args.min_valid_ratio,
            f"valid-frame ratio {valid_ratio:.3f} < {args.min_valid_ratio:.3f}",
        ),
    ]
    reasons = [reason for passed, reason in checks if not passed]
    summary: dict[str, Any] = {
        "target_hz": stage.target_hz,
        "received_frames": received,
        "sent_frames": expected,
        "receiver_hz": receiver_hz,
        "delivery_ratio": delivery_ratio,
        "valid_ratio": valid_ratio,
        "loss": loss,
        "duplicate_packets": stage.duplicate_packets,
        "interarrival_p50_ms": gap_p50,
        "interarrival_p95_ms": gap_p95,
        "interarrival_p99_ms": gap_p99,
        "sender_get_pose_p95_ms": percentile(stage.sender_get_pose_ms, 95.0),
        "sender_tracker_read_p95_ms": percentile(stage.sender_tracker_read_ms, 95.0),
        "sender_sample_to_send_p95_ms": percentile(stage.sender_sample_to_send_ms, 95.0),
        "stable": not reasons,
        "reason": "; ".join(reasons) if reasons else "ok",
    }
    latency_method: str | None = None
    if stage.clock_sync_packets:
        latency_method = "direct-link clock-sync estimate"
    elif args.assume_clock_synced:
        latency_method = "externally synchronized clocks"
    if latency_method is not None:
        summary.update(
            {
                "latency_method": latency_method,
                "send_to_receiver_p50_ms": percentile(stage.send_to_receiver_ms, 50.0),
                "send_to_receiver_p95_ms": percentile(stage.send_to_receiver_ms, 95.0),
                "sample_to_receiver_p50_ms": percentile(stage.sample_to_receiver_ms, 50.0),
                "sample_to_receiver_p95_ms": percentile(stage.sample_to_receiver_ms, 95.0),
                "clock_sync_rtt_ms": stage.clock_sync_rtt_ms,
            }
        )
    return summary


def print_stage_summary(stage: StageState, summary: dict[str, Any]) -> None:
    print(
        f"[Receiver] stage={stage.stage_id} target={stage.target_hz:g}Hz "
        f"received={summary['received_frames']}/{summary['sent_frames']} "
        f"rate={summary['receiver_hz']:.1f}Hz delivery={summary['delivery_ratio']:.3f} "
        f"valid={summary['valid_ratio']:.3f} loss={summary['loss']} "
        f"gap(p50/p95)={fmt_ms(summary['interarrival_p50_ms'])}/{fmt_ms(summary['interarrival_p95_ms'])} "
        f"stable={'YES' if summary['stable'] else 'NO'}"
    )
    latency_method = summary.get("latency_method")
    if latency_method:
        print(
            f"[Receiver] {latency_method}: send->receive(p50/p95)="
            f"{fmt_ms(summary['send_to_receiver_p50_ms'])}/{fmt_ms(summary['send_to_receiver_p95_ms'])}, "
            f"sample->receive(p50/p95)={fmt_ms(summary['sample_to_receiver_p50_ms'])}/"
            f"{fmt_ms(summary['sample_to_receiver_p95_ms'])} "
            f"clock-sync RTT={fmt_ms(summary.get('clock_sync_rtt_ms'))}"
        )
    else:
        print("[Receiver] one-way latency not shown: sender clock sync failed; check the direct UDP path.")
    if not summary["stable"]:
        print(f"[Receiver] unstable: {summary['reason']}")


def send_message(sock: socket.socket, message: dict[str, Any], address: tuple[str, int]) -> None:
    try:
        sock.sendto(json.dumps(message, separators=(",", ":")).encode("utf-8"), address)
    except OSError as exc:
        print(f"[Warn] failed to send benchmark response to {address[0]}:{address[1]}: {exc}")


def main() -> None:
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as exc:
        sock.close()
        raise SystemExit(f"[Error] unable to bind benchmark receiver {args.host}:{args.port}: {exc}") from exc

    stages: dict[tuple[str, int], StageState] = {}
    print(
        f"[Start] VIVE benchmark receiver listening on {args.host}:{args.port}. "
        "Start vive_tracker_benchmark_sender.py on the VIVE/OpenVR host."
    )
    print(
        "[Info] sender automatically performs direct-link NTP-style clock sync before each "
        "stage. --assume-clock-synced is only a fallback for an external NTP/PTP setup."
    )

    try:
        while True:
            raw, address = sock.recvfrom(int(args.max_datagram_size))
            received_monotonic = time.perf_counter()
            received_timestamp = time.time()
            try:
                message = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(message, dict) or message.get("protocol") != PROTOCOL:
                continue
            run_id = message.get("run_id")
            message_type = message.get("type")
            if not isinstance(run_id, str) or not run_id:
                continue
            if message_type == "clock_sync_request":
                try:
                    sync_id = str(message["sync_id"])
                    t1 = float(message["t1_sender_sent"])
                except (KeyError, TypeError, ValueError):
                    continue
                t2 = received_timestamp
                send_message(
                    sock,
                    {
                        "protocol": PROTOCOL,
                        "type": "clock_sync_response",
                        "run_id": run_id,
                        "sync_id": sync_id,
                        "t1_sender_sent": t1,
                        "t2_receiver_received": t2,
                        "t3_receiver_sent": time.time(),
                    },
                    address,
                )
                continue
            try:
                stage_id = int(message["stage_id"])
                target_hz = float(message.get("target_hz", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if target_hz <= 0.0:
                continue
            stage = get_stage(
                stages,
                run_id=run_id,
                stage_id=stage_id,
                target_hz=target_hz,
            )
            if message_type == "stage_start":
                print(f"\n[Stage {stage_id}] run_id={run_id} target={target_hz:g}Hz")
                continue
            if message_type == "data":
                try:
                    sequence = int(message["seq"])
                except (KeyError, TypeError, ValueError):
                    continue
                if sequence in stage.received_sequences:
                    stage.duplicate_packets += 1
                else:
                    stage.received_sequences.add(sequence)
                    stage.received_at_monotonic.append(received_monotonic)
                    trackers = message.get("trackers")
                    if isinstance(trackers, list) and trackers and all(
                        isinstance(tracker, dict) and bool(tracker.get("valid"))
                        for tracker in trackers
                    ):
                        stage.valid_frames += 1
                    get_pose_ms = message.get("get_pose_elapsed_ms")
                    if isinstance(get_pose_ms, (float, int)):
                        stage.sender_get_pose_ms.append(float(get_pose_ms))
                    sample_to_send_ms = message.get("sample_to_send_elapsed_ms")
                    if isinstance(sample_to_send_ms, (float, int)):
                        stage.sender_sample_to_send_ms.append(float(sample_to_send_ms))
                    tracker_read_ms = message.get("tracker_read_elapsed_ms")
                    if isinstance(tracker_read_ms, list):
                        stage.sender_tracker_read_ms.extend(
                            float(value)
                            for value in tracker_read_ms
                            if isinstance(value, (float, int))
                        )
                    send_timestamp = message.get("send_timestamp")
                    sample_timestamp = message.get("sample_timestamp")
                    clock_offset_sec = message.get("clock_offset_sec")
                    if isinstance(clock_offset_sec, (float, int)):
                        stage.clock_sync_packets += 1
                        clock_sync_rtt_ms = message.get("clock_sync_rtt_ms")
                        if isinstance(clock_sync_rtt_ms, (float, int)):
                            stage.clock_sync_rtt_ms = float(clock_sync_rtt_ms)
                        if isinstance(send_timestamp, (float, int)):
                            stage.send_to_receiver_ms.append(
                                (
                                    received_timestamp
                                    - (float(send_timestamp) + float(clock_offset_sec))
                                )
                                * 1000.0
                            )
                        if isinstance(sample_timestamp, (float, int)):
                            stage.sample_to_receiver_ms.append(
                                (
                                    received_timestamp
                                    - (float(sample_timestamp) + float(clock_offset_sec))
                                )
                                * 1000.0
                            )
                    elif args.assume_clock_synced:
                        if isinstance(send_timestamp, (float, int)):
                            stage.send_to_receiver_ms.append(
                                (received_timestamp - float(send_timestamp)) * 1000.0
                            )
                        if isinstance(sample_timestamp, (float, int)):
                            stage.sample_to_receiver_ms.append(
                                (received_timestamp - float(sample_timestamp)) * 1000.0
                            )
                send_message(
                    sock,
                    {
                        "protocol": PROTOCOL,
                        "type": "ack",
                        "run_id": run_id,
                        "stage_id": stage_id,
                        "seq": sequence,
                        "receiver_timestamp": received_timestamp,
                    },
                    address,
                )
                continue
            if message_type != "stage_end":
                continue
            try:
                stage.sent_frames = int(message["sent_frames"])
                stage.valid_sent_frames = int(message.get("valid_frames", 0))
                stage.stage_duration_sec = float(message["stage_duration_sec"])
            except (KeyError, TypeError, ValueError):
                print(f"[Warn] stage={stage_id} received malformed stage_end")
                continue
            stage.finished = True
            stage.summary = finalise_stage(stage, args)
            print_stage_summary(stage, stage.summary)
            send_message(
                sock,
                {
                    "protocol": PROTOCOL,
                    "type": "stage_summary",
                    "run_id": run_id,
                    "stage_id": stage_id,
                    "summary": stage.summary,
                },
                address,
            )
    except KeyboardInterrupt:
        print("\n[Exit] receiver stopped by user")
    finally:
        sock.close()

    finished = [stage for stage in stages.values() if stage.finished and stage.summary]
    stable = [stage for stage in finished if bool(stage.summary and stage.summary.get("stable"))]
    if stable:
        best = max(stable, key=lambda stage: stage.target_hz)
        assert best.summary is not None
        print(
            f"[Result] highest receiver-stable target={best.target_hz:g}Hz "
            f"(received={best.summary['receiver_hz']:.1f}Hz, "
            f"delivery={best.summary['delivery_ratio']:.3f})."
        )
    elif finished:
        print("[Result] no receiver-stable stage; see the rejection reasons above.")
    else:
        print("[Result] no complete benchmark stage was received.")


if __name__ == "__main__":
    main()
