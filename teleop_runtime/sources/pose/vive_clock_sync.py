"""Optional VIVE clock-control server for the collector host.

The data-plane VIVE receiver intentionally remains a plain v1 UDP pose
receiver.  This module listens on a separate, explicitly configured port for
the sender-initiated four-timestamp exchange.  Its estimates are diagnostic
capture-time mappings only; callers must use local receive availability time
for causal ``<= T0`` selection.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from hardware.VIVE.mocap.clock_sync import (
    ClockSyncEstimate,
    ClockSyncMeasurement,
    ClockSyncSampleStore,
    ClockSyncResult,
    make_reply,
    measurement_from_result,
    parse_probe,
    parse_result,
)


@dataclass(frozen=True)
class ViveClockSyncConfig:
    bind_host: str = "0.0.0.0"
    port: int = 0
    max_round_trip_sec: float = 0.025
    stale_after_sec: float = 15.0
    pending_ttl_sec: float = 2.0
    max_pending: int = 256
    min_samples: int = 4
    expected_sender_host: str | None = None


@dataclass(frozen=True)
class ViveClockSyncPeerState:
    sender_host: str
    status: str
    estimate: ClockSyncEstimate | None
    last_success_collector_monotonic_ns: int | None


@dataclass(frozen=True)
class _PendingProbe:
    sender_instance_id: str
    probe_id: int
    t1_sender_monotonic_ns: int
    t2_collector_monotonic_ns: int
    t3_collector_monotonic_ns: int
    created_collector_monotonic_ns: int


@dataclass
class _PeerSamples:
    sample_store: ClockSyncSampleStore
    last_success_collector_monotonic_ns: int | None = None


class ViveClockSyncServer:
    """Bounded UDP control server keyed by sender IP address and instance ID."""

    def __init__(
        self,
        config: ViveClockSyncConfig,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.config = config
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._socket_lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._ready = threading.Event()
        self._pending: dict[tuple[str, int, str, int], _PendingProbe] = {}
        self._peers: dict[str, _PeerSamples] = {}
        self._probe_count = 0
        self._result_count = 0
        self._accepted_count = 0
        self._rejected_count = 0
        self._error_count = 0
        self._last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return int(self.config.port) > 0

    @property
    def bound_port(self) -> int | None:
        with self._socket_lock:
            sock = self._sock
            if sock is None:
                return None
            return int(sock.getsockname()[1])

    def start(self) -> None:
        if not self.enabled or self._running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.config.bind_host, int(self.config.port)))
            sock.settimeout(0.1)
        except OSError:
            try:
                sock.close()
            except OSError:
                pass
            raise
        with self._socket_lock:
            self._sock = sock
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop,
            daemon=True,
            name="vive_clock_sync_collector",
        )
        self._thread.start()

    def close(self) -> None:
        self._running = False
        with self._socket_lock:
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)

    def wait_until_ready(self, timeout_sec: float, *, min_samples: int | None = None) -> bool:
        required = max(
            1,
            int(self.config.min_samples if min_samples is None else min_samples),
        )
        deadline = time.perf_counter() + max(0.0, float(timeout_sec))
        while True:
            if self._has_ready_peer(required):
                return True
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return False
            self._ready.wait(timeout=min(0.05, remaining))
            self._ready.clear()

    def peer_state(
        self,
        sender_host: str | None,
        *,
        now_collector_monotonic_ns: int,
    ) -> ViveClockSyncPeerState:
        normalized_host = str(sender_host or "")
        with self._lock:
            peer = self._peers.get(normalized_host)
            if peer is None:
                return ViveClockSyncPeerState(
                    sender_host=normalized_host,
                    status="unavailable",
                    estimate=None,
                    last_success_collector_monotonic_ns=None,
                )
            estimate = peer.sample_store.estimate()
            last_success = peer.last_success_collector_monotonic_ns
        if (
            estimate is None
            or last_success is None
            or estimate.sample_count < max(1, int(self.config.min_samples))
        ):
            status = "awaiting_samples"
        elif (
            int(now_collector_monotonic_ns) - last_success
            > int(max(0.0, self.config.stale_after_sec) * 1_000_000_000)
        ):
            status = "stale"
        else:
            status = "synchronized"
        return ViveClockSyncPeerState(
            sender_host=normalized_host,
            status=status,
            estimate=estimate,
            last_success_collector_monotonic_ns=last_success,
        )

    def invalidate_peer(self, sender_host: str | None) -> None:
        """Drop a mapping when the data-plane sender sequence restarts."""

        normalized_host = str(sender_host or "")
        if not normalized_host:
            return
        with self._lock:
            peer = self._peers.get(normalized_host)
            if peer is not None:
                peer.sample_store.clear()
                peer.last_success_collector_monotonic_ns = None

    def debug_stats(self) -> dict[str, Any]:
        with self._lock:
            peer_stats = {}
            for host, peer in self._peers.items():
                estimate = peer.sample_store.estimate()
                peer_stats[host] = {
                    "sender_instance_id": None
                    if estimate is None
                    else estimate.sender_instance_id,
                    "sample_count": 0 if estimate is None else estimate.sample_count,
                    "collector_minus_sender_offset_ns": None
                    if estimate is None
                    else estimate.collector_minus_sender_offset_ns,
                    "round_trip_ns": None if estimate is None else estimate.round_trip_ns,
                    "uncertainty_ns": None if estimate is None else estimate.uncertainty_ns,
                    "last_success_collector_monotonic_ns": (
                        peer.last_success_collector_monotonic_ns
                    ),
                }
            return {
                "enabled": self.enabled,
                "bound_port": self.bound_port,
                "probe_count": self._probe_count,
                "result_count": self._result_count,
                "accepted_count": self._accepted_count,
                "rejected_count": self._rejected_count,
                "error_count": self._error_count,
                "last_error": self._last_error,
                "peers": peer_stats,
            }

    def _recv_loop(self) -> None:
        while self._running:
            with self._socket_lock:
                sock = self._sock
            if sock is None:
                return
            try:
                packet, address = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._running:
                    self._record_error(exc)
                continue
            try:
                self._handle_packet(packet, (str(address[0]), int(address[1])))
            except Exception as exc:  # Keep an untrusted UDP datagram isolated.
                self._record_error(exc)

    def _handle_packet(self, packet: bytes, address: tuple[str, int]) -> None:
        sender_host, _sender_port = address
        expected = self.config.expected_sender_host
        if expected and sender_host != expected:
            with self._lock:
                self._rejected_count += 1
            return

        probe = parse_probe(packet)
        if probe is not None:
            self._handle_probe(probe, address)
            return
        result = parse_result(packet)
        if result is not None:
            self._handle_result(result, address)
            return
        with self._lock:
            self._rejected_count += 1

    def _handle_probe(self, probe: Any, address: tuple[str, int]) -> None:
        t2 = int(self._clock_ns())
        send_error: OSError | None = None
        with self._socket_lock:
            sock = self._sock
            if sock is None:
                return
            # t3 must be immediately before send; work such as dictionary
            # bookkeeping belongs outside the measured processing interval.
            t3 = int(self._clock_ns())
            payload = make_reply(
                sender_instance_id=probe.sender_instance_id,
                probe_id=probe.probe_id,
                t1_sender_monotonic_ns=probe.t1_sender_monotonic_ns,
                t2_collector_monotonic_ns=t2,
                t3_collector_monotonic_ns=t3,
            )
            try:
                sock.sendto(payload, address)
            except OSError as exc:
                send_error = exc
        if send_error is not None:
            self._record_error(send_error)
            return
        pending_key = (
            address[0],
            address[1],
            probe.sender_instance_id,
            probe.probe_id,
        )
        with self._lock:
            self._purge_pending_locked(t3)
            max_pending = max(1, int(self.config.max_pending))
            while len(self._pending) >= max_pending:
                oldest_key = next(iter(self._pending))
                self._pending.pop(oldest_key, None)
            peer = self._peers.get(address[0])
            if peer is None:
                peer = _PeerSamples(sample_store=ClockSyncSampleStore())
                self._peers[address[0]] = peer
            previous_instance_id = peer.sample_store.sender_instance_id
            peer.sample_store.begin_instance(probe.sender_instance_id)
            if previous_instance_id != probe.sender_instance_id:
                peer.last_success_collector_monotonic_ns = None
            self._pending[pending_key] = _PendingProbe(
                sender_instance_id=probe.sender_instance_id,
                probe_id=probe.probe_id,
                t1_sender_monotonic_ns=probe.t1_sender_monotonic_ns,
                t2_collector_monotonic_ns=t2,
                t3_collector_monotonic_ns=t3,
                created_collector_monotonic_ns=t3,
            )
            self._probe_count += 1

    def _handle_result(self, result: ClockSyncResult, address: tuple[str, int]) -> None:
        key = (address[0], address[1], result.sender_instance_id, result.probe_id)
        with self._lock:
            self._result_count += 1
            pending = self._pending.pop(key, None)
        if pending is None or (
            pending.t1_sender_monotonic_ns != result.t1_sender_monotonic_ns
            or pending.t2_collector_monotonic_ns != result.t2_collector_monotonic_ns
            or pending.t3_collector_monotonic_ns != result.t3_collector_monotonic_ns
        ):
            with self._lock:
                self._rejected_count += 1
            return
        measurement = measurement_from_result(result)
        max_rtt_ns = int(max(0.0, self.config.max_round_trip_sec) * 1_000_000_000)
        if measurement is None or (
            max_rtt_ns > 0 and measurement.round_trip_ns > max_rtt_ns
        ):
            with self._lock:
                self._rejected_count += 1
            return
        self._record_measurement(address[0], measurement)

    def _record_measurement(
        self,
        sender_host: str,
        measurement: ClockSyncMeasurement,
    ) -> None:
        now_ns = int(self._clock_ns())
        with self._lock:
            peer = self._peers.get(sender_host)
            if peer is None:
                peer = _PeerSamples(
                    sample_store=ClockSyncSampleStore(
                        max_round_trip_ns=(
                            None
                            if self.config.max_round_trip_sec <= 0.0
                            else int(
                                self.config.max_round_trip_sec * 1_000_000_000
                            )
                        ),
                    )
                )
                self._peers[sender_host] = peer
            estimate = peer.sample_store.add(measurement)
            if estimate is None:
                self._rejected_count += 1
                return
            peer.last_success_collector_monotonic_ns = now_ns
            self._accepted_count += 1
            self._ready.set()

    def _has_ready_peer(self, min_samples: int) -> bool:
        with self._lock:
            expected = self.config.expected_sender_host
            peers = (
                [self._peers.get(expected)] if expected else list(self._peers.values())
            )
            return any(
                peer is not None
                and (estimate := peer.sample_store.estimate()) is not None
                and estimate.sample_count >= min_samples
                for peer in peers
            )

    def _purge_pending_locked(self, now_ns: int) -> None:
        ttl_ns = int(max(0.1, self.config.pending_ttl_sec) * 1_000_000_000)
        stale_keys = [
            key
            for key, pending in self._pending.items()
            if now_ns - pending.created_collector_monotonic_ns > ttl_ns
        ]
        for key in stale_keys:
            self._pending.pop(key, None)

    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            self._error_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
