from __future__ import annotations

import json
import select
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...interfaces import ObservationSource
from ...schemas import ObservationChannel
from ...transforms import identity_pose
from .vive_clock_sync import ViveClockSyncConfig, ViveClockSyncServer
from .vive_motion import summarize_tracker_motion_window


# ``None`` resolves to the local new-protocol default.  Pass ``0`` explicitly
# to use a v1-only legacy receiver profile.
DEFAULT_VIVE_CLOCK_SYNC_PORT = 10001


@dataclass(frozen=True)
class ViveTrackerConfig:
    name: str
    side: str | None
    role: str
    channel_name: str
    legacy_index: int | None = None
    critical: bool = True


class VivePoseSource(ObservationSource):
    def __init__(
        self,
        tracker_configs: list[ViveTrackerConfig],
        *,
        host: str = "0.0.0.0",
        port: int = 9999,
        legacy_tracker_count: int = 3,
        socket_timeout_sec: float = 0.2,
        drain_stale_packets: bool = True,
        max_drain_packets: int = 64,
        max_datagram_size: int = 65536,
        history_window_sec: float = 1.0,
        history_reset_gap_sec: float = 0.1,
        motion_window_sec: float = 0.1,
        clock_sync_bind_host: str | None = None,
        clock_sync_port: int | None = None,
        clock_sync_max_round_trip_sec: float = 0.025,
        clock_sync_stale_after_sec: float = 15.0,
        clock_sync_min_samples: int = 4,
        clock_sync_expected_sender_host: str | None = None,
        name: str = "vive_pose",
        allow_socket_reuse: bool = False,
    ):
        self._name = name
        self.host = host
        self.port = int(port)
        self.legacy_tracker_count = int(legacy_tracker_count)
        self.socket_timeout_sec = float(socket_timeout_sec)
        self.drain_stale_packets = bool(drain_stale_packets)
        self.max_drain_packets = max(0, int(max_drain_packets))
        self.max_datagram_size = int(max_datagram_size)
        self.history_window_sec = max(0.0, float(history_window_sec))
        self.history_reset_gap_sec = max(0.0, float(history_reset_gap_sec))
        self.motion_window_sec = max(0.0, float(motion_window_sec))
        self.clock_sync_bind_host = str(clock_sync_bind_host or host)
        self.clock_sync_port = int(
            DEFAULT_VIVE_CLOCK_SYNC_PORT
            if clock_sync_port is None
            else clock_sync_port
        )
        self.clock_sync_max_round_trip_sec = max(
            0.0,
            float(clock_sync_max_round_trip_sec),
        )
        self.clock_sync_stale_after_sec = max(0.0, float(clock_sync_stale_after_sec))
        self.clock_sync_min_samples = max(1, int(clock_sync_min_samples))
        self.clock_sync_expected_sender_host = (
            None
            if not clock_sync_expected_sender_host
            else str(clock_sync_expected_sender_host)
        )
        # A live control receiver is a singleton for its UDP port.  Reusing
        # the address by default lets a crashed/duplicate GUI silently split
        # datagrams between two consumers, which looks exactly like a sender
        # stale condition.  Legacy diagnostics may opt in explicitly.
        self.allow_socket_reuse = bool(allow_socket_reuse)
        if self.clock_sync_port < 0 or self.clock_sync_port > 65535:
            raise ValueError("clock_sync_port must be in [0, 65535]")
        if self.clock_sync_port and self.clock_sync_port == self.port:
            raise ValueError(
                "clock_sync_port must be separate from the VIVE pose data port."
            )
        self._tracker_configs = list(tracker_configs)
        self.tracker_configs = {config.name: config for config in tracker_configs}
        self._configs_by_role = {config.role: config for config in tracker_configs}
        # Incremented whenever the logical source-role mapping changes.  This
        # is carried on observation metadata so a downstream session can reset
        # its pose baseline without coupling itself to VIVE.
        self._source_mapping_generation = 0
        self.sock: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._worker_started_once = False
        self._lock = threading.Lock()
        self._socket_lock = threading.Lock()
        self._latest_by_channel: dict[str, dict[str, Any]] = {}
        # Passive diagnostics retain the sender's upstream tracker roles even
        # when a role is not selected by the current logical-channel mapping.
        # This cache never participates in retargeting or control admission.
        self._latest_upstream_by_role: dict[str, dict[str, Any]] = {}
        self._upstream_role_update_counts: dict[str, int] = {}
        self._last_upstream_role_update_at: dict[str, float] = {}
        self._packet_count = 0
        self._last_packet_received_at: float | None = None
        # Wall time is retained for human-readable provenance, while this
        # monotonic receive cut is the only clock used to classify transport
        # freshness.  A wall-clock adjustment must not turn a live sender into
        # ``packet_stale`` (or make an old packet appear current).
        self._last_packet_received_monotonic_time: float | None = None
        # Transport heartbeat is captured at the socket boundary, before any
        # JSON/legacy parsing or history publication.  It is intentionally
        # separate from ``last_packet_received_*`` which is retained for the
        # processed-batch compatibility counters.
        self._datagram_count = 0
        self._last_datagram_received_at: float | None = None
        self._last_datagram_received_monotonic_time: float | None = None
        self._dropped_stale_packet_count = 0
        self._last_drain_count = 0
        self._channel_update_counts: dict[str, int] = {}
        self._last_channel_update_at: dict[str, float] = {}
        self._history_by_channel: dict[str, deque[dict[str, Any]]] = {}
        self._history_clock_source: str | None = None
        self._history_last_time: float | None = None
        self._history_last_seq: int | None = None
        self._history_last_receiver_monotonic_time: float | None = None
        self._history_packet_count = 0
        self._history_reset_count = 0
        self._history_last_reset_reason: str | None = None
        self._history_duplicate_packet_count = 0
        self._history_out_of_order_packet_count = 0
        self._history_missing_seq_count = 0
        self._recv_error_count = 0
        self._last_recv_error: str | None = None
        self._socket_restart_count = 0
        self._last_socket_restart_at: float | None = None
        self._process_error_count = 0
        self._last_process_error: str | None = None
        self._last_process_error_at: float | None = None
        self._last_process_started_at: float | None = None
        self._last_process_finished_at: float | None = None
        self._last_process_finished_monotonic_time: float | None = None
        self._last_process_duration_sec: float | None = None
        self._last_process_batch_size = 0
        self._last_process_parse_error_count = 0
        self._legacy_format = f"<{self.legacy_tracker_count * 12}f"
        self._legacy_packet_size = struct.calcsize(self._legacy_format)
        self._clock_sync_server = ViveClockSyncServer(
            ViveClockSyncConfig(
                bind_host=self.clock_sync_bind_host,
                port=self.clock_sync_port,
                max_round_trip_sec=self.clock_sync_max_round_trip_sec,
                stale_after_sec=self.clock_sync_stale_after_sec,
                min_samples=self.clock_sync_min_samples,
                expected_sender_host=self.clock_sync_expected_sender_host,
            )
        )

    def set_tracker_configs(self, tracker_configs: list[ViveTrackerConfig]) -> None:
        configs = list(tracker_configs)
        names = [config.name for config in configs]
        roles = [config.role for config in configs]
        channels = [config.channel_name for config in configs]
        if len(names) != len(set(names)):
            raise ValueError("VIVE tracker config names must be unique.")
        if len(roles) != len(set(roles)):
            raise ValueError("VIVE tracker source roles must be unique.")
        if len(channels) != len(set(channels)):
            raise ValueError("VIVE tracker output channels must be unique.")
        with self._lock:
            self._tracker_configs = configs
            self.tracker_configs = {config.name: config for config in configs}
            self._configs_by_role = {config.role: config for config in configs}
            self._source_mapping_generation += 1
            self._latest_by_channel = {}
            self._channel_update_counts = {}
            self._last_channel_update_at = {}
            self._clear_history_locked(reason="tracker_configs_changed")

    @property
    def source_role_by_output_channel(self) -> dict[str, str]:
        """Return the current physical/source-role to logical-channel binding."""

        with self._lock:
            return {
                config.channel_name: config.role
                for config in self._tracker_configs
            }

    def swap_left_right_source_roles(self) -> dict[str, object]:
        """Swap left/right upstream roles without renaming logical channels.

        This works for both cardinalities: a single config toggles which
        upstream wrist role feeds its logical channel, while a bilateral
        config exchanges both upstream sources. Auxiliary roles are unchanged.
        Buffered history is cleared; the latest sender snapshot is exposed
        under the new mapping so setup status updates immediately.
        """

        role_swap = {
            "left_wrist": "right_wrist",
            "right_wrist": "left_wrist",
        }
        with self._lock:
            before = {
                config.channel_name: config.role
                for config in self._tracker_configs
            }
            if not any(role in role_swap for role in before.values()):
                raise RuntimeError(
                    "VIVE source has no left_wrist/right_wrist role to swap."
                )
            configs = [
                replace(
                    config,
                    role=role_swap.get(config.role, config.role),
                )
                for config in self._tracker_configs
            ]
            roles = [config.role for config in configs]
            if len(roles) != len(set(roles)):
                raise RuntimeError(
                    "VIVE source-role swap would create duplicate roles."
                )
            self._tracker_configs = configs
            self.tracker_configs = {
                config.name: config for config in configs
            }
            self._configs_by_role = {
                config.role: config for config in configs
            }
            self._source_mapping_generation += 1
            self._latest_by_channel = {}
            self._channel_update_counts = {}
            self._last_channel_update_at = {}
            self._clear_history_locked(
                reason="left_right_source_roles_swapped"
            )
            # Keep the latest sender snapshot visible under the new mapping.
            # Also seed the causal history with that same snapshot.  The
            # session queries history (not the display cache) when it applies a
            # cutoff, so exposing only ``_latest_by_channel`` created a
            # one-cycle missing_channel after every swap.
            mapped_latest: dict[str, dict[str, Any]] = {}
            for config in configs:
                upstream = self._latest_upstream_by_role.get(config.role)
                if upstream is None:
                    continue
                payload = self._copy_tracker_payload(upstream)
                payload["role"] = config.role
                payload["tracker_name"] = payload.get("tracker_name") or config.name
                payload["config"] = config
                payload["source_mapping_generation"] = (
                    self._source_mapping_generation
                )
                mapped_latest[config.channel_name] = payload

            if mapped_latest:
                first_payload = next(iter(mapped_latest.values()))
                received_at = self._finite_timestamp(
                    first_payload.get("receiver_timestamp"),
                    fallback=time.time(),
                )
                receiver_monotonic_time = self._finite_timestamp(
                    first_payload.get("receiver_monotonic_time"),
                    fallback=time.perf_counter(),
                )
                self._append_history_packet_locked(
                    mapped_latest,
                    received_at=received_at,
                    receiver_monotonic_time=receiver_monotonic_time,
                )
            self._latest_by_channel = {}
            for channel_name, payload in mapped_latest.items():
                live_payload = self._copy_tracker_payload(payload)
                history = self._history_by_channel.get(channel_name)
                if history:
                    live_payload["history_time"] = history[-1]["history_time"]
                    live_payload["history_clock_source"] = history[-1][
                        "history_clock_source"
                    ]
                    for key in (
                        "source_timestamp",
                        "source_clock",
                        "available_timestamp",
                        "available_clock",
                    ):
                        live_payload[key] = history[-1].get(key)
                self._latest_by_channel[channel_name] = live_payload
            after = {
                config.channel_name: config.role
                for config in configs
            }
        return {
            "status": "swapped",
            "before_source_role_by_output_channel": before,
            "after_source_role_by_output_channel": after,
        }

    @property
    def name(self) -> str:
        return self._name

    def expected_channel_names(self) -> tuple[str, ...]:
        return tuple(
            config.channel_name for config in self._tracker_configs if config.critical
        )

    @property
    def clock_sync_enabled(self) -> bool:
        return self._clock_sync_server.enabled

    def wait_for_clock_sync(
        self,
        timeout_sec: float,
        *,
        min_samples: int | None = None,
    ) -> bool:
        """Wait for optional control-plane calibration without changing pose I/O.

        Call this after :meth:`start` and before beginning a recording when a
        collection profile requires remote capture-time diagnostics.  The
        normal local availability history is valid even when this returns
        ``False``.
        """

        if not self.clock_sync_enabled:
            return False
        return self._clock_sync_server.wait_until_ready(
            timeout_sec,
            min_samples=min_samples,
        )

    def clock_sync_debug_stats(self) -> dict[str, Any]:
        return self._clock_sync_server.debug_stats()

    def start(self) -> None:
        if self._running:
            return
        try:
            with self._socket_lock:
                self._open_socket_locked()
            self._clock_sync_server.start()
        except Exception:
            with self._socket_lock:
                self._close_socket_locked()
            self._clock_sync_server.close()
            raise
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop,
            daemon=True,
            name=f"{self.name}_recv",
        )
        self._worker_started_once = True
        self._thread.start()

    def close(self) -> None:
        self._running = False
        with self._socket_lock:
            self._close_socket_locked()
        self._clock_sync_server.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _open_socket_locked(self) -> None:
        if self.sock is not None:
            self._close_socket_locked()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.allow_socket_reuse:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(self.socket_timeout_sec)
        self.sock = sock

    def _close_socket_locked(self) -> None:
        sock = self.sock
        self.sock = None
        if sock is None:
            return
        try:
            sock.close()
        except OSError:
            pass

    def _record_recv_error(self, exc: BaseException) -> None:
        with self._lock:
            self._recv_error_count += 1
            self._last_recv_error = f"{type(exc).__name__}: {exc}"

    def _record_process_error(
        self,
        exc: BaseException,
        *,
        packet_index: int | None = None,
    ) -> None:
        detail = f"{type(exc).__name__}: {exc}"
        if packet_index is not None:
            detail = f"packet_index={int(packet_index)};{detail}"
        with self._lock:
            self._process_error_count += 1
            self._last_process_error = detail
            self._last_process_error_at = time.time()

    def _restart_socket_after_error(self, exc: BaseException) -> bool:
        self._record_recv_error(exc)
        if not self._running:
            return False
        try:
            with self._socket_lock:
                if not self._running:
                    return False
                self._open_socket_locked()
        except OSError as restart_exc:
            self._record_recv_error(restart_exc)
            return False
        with self._lock:
            self._socket_restart_count += 1
            self._last_socket_restart_at = time.time()
        return True

    def _ensure_socket_open(self) -> bool:
        if not self._running:
            return False
        with self._socket_lock:
            if self.sock is not None:
                return True
            try:
                self._open_socket_locked()
            except OSError as exc:
                self._record_recv_error(exc)
                return False
        with self._lock:
            self._socket_restart_count += 1
            self._last_socket_restart_at = time.time()
        return True

    def _recv_loop(self) -> None:
        try:
            while self._running:
                try:
                    received_batch = self._recv_packet_batch()
                except socket.timeout:
                    continue
                except (OSError, ValueError) as exc:
                    if not self._restart_socket_after_error(exc) and not self._running:
                        break
                    time.sleep(min(0.25, max(0.01, self.socket_timeout_sec)))
                    continue
                if not received_batch:
                    self._ensure_socket_open()
                    time.sleep(min(0.05, max(0.0, self.socket_timeout_sec)))
                    continue

                # Parsing/retaining a malformed or unexpected datagram must
                # never terminate the receive worker.  The batch method also
                # isolates individual packets, while this outer guard protects
                # the worker from a future regression in history publication.
                try:
                    self._process_received_batch(received_batch)
                except Exception as exc:  # noqa: BLE001 - worker must survive bad UDP input.
                    self._record_process_error(exc)
        except Exception as exc:  # noqa: BLE001 - expose an unhealthy worker explicitly.
            self._record_process_error(exc)
        finally:
            # An unexpected worker exit must not leave a live socket behind or
            # report ``running=True`` indefinitely.  ``close()`` remains the
            # normal lifecycle path; this is only the crash containment path.
            if self._running:
                self._running = False
                with self._socket_lock:
                    self._close_socket_locked()

    def _recv_packet_batch(
        self,
    ) -> list[tuple[bytes, float, float, tuple[str, int]]]:
        """Receive one UDP datagram and drain the currently queued batch.

        Every datagram is returned in receive order.  The control path still
        publishes only the last datagram, while the full batch is retained in
        the time-based history used by later intent estimation.
        """

        with self._socket_lock:
            sock = self.sock
        if sock is None:
            return []

        packet, sender_address = sock.recvfrom(self.max_datagram_size)
        received_at = time.time()
        receiver_monotonic_time = time.perf_counter()
        self._record_datagram_received(
            received_at=received_at,
            receiver_monotonic_time=receiver_monotonic_time,
        )
        batch = [
            (
                packet,
                received_at,
                receiver_monotonic_time,
                (str(sender_address[0]), int(sender_address[1])),
            )
        ]

        if not self.drain_stale_packets:
            return batch

        stale_count = 0
        while stale_count < self.max_drain_packets:
            with self._socket_lock:
                if sock is not self.sock:
                    break
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break
            try:
                packet, sender_address = sock.recvfrom(self.max_datagram_size)
            except (BlockingIOError, socket.timeout):
                break
            received_at = time.time()
            receiver_monotonic_time = time.perf_counter()
            self._record_datagram_received(
                received_at=received_at,
                receiver_monotonic_time=receiver_monotonic_time,
            )
            batch.append(
                (
                    packet,
                    received_at,
                    receiver_monotonic_time,
                    (str(sender_address[0]), int(sender_address[1])),
                )
            )
            stale_count += 1

        return batch

    def _record_datagram_received(
        self,
        *,
        received_at: float,
        receiver_monotonic_time: float,
    ) -> None:
        """Record transport arrival before parsing can delay or fail."""

        with self._lock:
            self._datagram_count += 1
            self._last_datagram_received_at = float(received_at)
            self._last_datagram_received_monotonic_time = float(
                receiver_monotonic_time
            )

    def _recv_latest_packet(self) -> tuple[bytes, float, int] | None:
        """Compatibility helper returning only the last packet in a batch."""

        batch = self._recv_packet_batch()
        if not batch:
            return None
        packet, received_at, _, _ = batch[-1]
        return packet, received_at, len(batch) - 1

    def _process_received_batch(
        self,
        batch: list[tuple[Any, ...]],
    ) -> None:
        if not batch:
            return
        process_started_at = time.perf_counter()
        process_started_wall = time.time()
        parse_error_count = 0
        with self._lock:
            self._last_process_started_at = process_started_wall
            self._last_process_batch_size = len(batch)

        parsed_batch: list[
            tuple[dict[str, dict[str, Any]], float, float, str | None]
        ] = []
        upstream_batch: list[
            tuple[dict[str, dict[str, Any]] | None, float, float]
        ] = []
        try:
            for packet_index, received in enumerate(batch):
                # UDP receive entries are created internally, but retaining the
                # fallback values here makes direct diagnostics and malformed
                # test inputs unable to kill the worker.
                received_at = time.time()
                receiver_monotonic_time = time.perf_counter()
                sender_host: str | None = None
                try:
                    if len(received) == 3:
                        packet, received_at, receiver_monotonic_time = received
                    elif len(received) == 4:
                        packet, received_at, receiver_monotonic_time, sender_address = received
                        sender_host = str(sender_address[0]) if sender_address else None
                    else:
                        raise ValueError(
                            "VIVE receive batch entries must contain 3 or 4 values"
                        )
                    received_at = float(received_at)
                    receiver_monotonic_time = float(receiver_monotonic_time)
                    decoded = self._decode_json_packet(packet)
                    if decoded is None:
                        parsed = (
                            self._parse_legacy_packet(
                                packet,
                                received_at,
                                sender_host=sender_host,
                            )
                            or {}
                        )
                        upstream = None
                    else:
                        parsed = self._parse_decoded_json_packet(
                            decoded,
                            received_at,
                            sender_host=sender_host,
                        )
                        upstream = self._parse_decoded_upstream_role_diagnostics(
                            decoded,
                            received_at,
                            sender_host=sender_host,
                        )
                except Exception as exc:  # noqa: BLE001 - one bad datagram is not a worker failure.
                    parse_error_count += 1
                    self._record_process_error(exc, packet_index=packet_index)
                    parsed = {}
                    upstream = None
                parsed_batch.append(
                    (
                        parsed,
                        float(received_at),
                        float(receiver_monotonic_time),
                        sender_host,
                    )
                )
                upstream_batch.append(
                    (
                        upstream,
                        float(received_at),
                        float(receiver_monotonic_time),
                    )
                )

            stale_count = max(0, len(batch) - 1)
            # A malformed trailing datagram must not erase the last valid live
            # sample.  Select the newest successfully parsed datagram instead.
            latest_valid = next(
                (entry for entry in reversed(parsed_batch) if entry[0]),
                None,
            )
            if latest_valid is None:
                latest_parsed = {}
                latest_received_at = float(batch[-1][1])
                latest_receiver_monotonic_time = float(batch[-1][2])
                latest_sender_host = None
            else:
                (
                    latest_parsed,
                    latest_received_at,
                    latest_receiver_monotonic_time,
                    latest_sender_host,
                ) = latest_valid
            with self._lock:
                self._packet_count += len(batch)
                self._last_packet_received_at = float(batch[-1][1])
                self._last_packet_received_monotonic_time = float(batch[-1][2])
                self._dropped_stale_packet_count += stale_count
                self._last_drain_count = stale_count
                self._last_process_parse_error_count = parse_error_count

                for upstream, received_at, _receiver_monotonic_time in upstream_batch:
                    if upstream is None:
                        continue
                    for tracker_role in upstream:
                        self._upstream_role_update_counts[tracker_role] = (
                            self._upstream_role_update_counts.get(tracker_role, 0) + 1
                        )
                        self._last_upstream_role_update_at[tracker_role] = received_at

                latest_upstream = next(
                    (entry[0] for entry in reversed(upstream_batch) if entry[0] is not None),
                    None,
                )
                if latest_upstream is not None:
                    self._latest_upstream_by_role = {
                        role: self._copy_tracker_payload(payload)
                        for role, payload in latest_upstream.items()
                    }
                    for payload in self._latest_upstream_by_role.values():
                        payload["receiver_monotonic_time"] = (
                            latest_receiver_monotonic_time
                        )
                        payload["availability_monotonic_time"] = (
                            latest_receiver_monotonic_time
                        )

                for parsed, received_at, receiver_monotonic_time, _sender_host in parsed_batch:
                    if parsed:
                        self._append_history_packet_locked(
                            parsed,
                            received_at=received_at,
                            receiver_monotonic_time=receiver_monotonic_time,
                        )

                # Preserve the existing low-latency contract: only the newest
                # successfully parsed UDP datagram becomes the live observation.
                for channel_name, tracker_payload in latest_parsed.items():
                    live_payload = self._copy_tracker_payload(tracker_payload)
                    live_payload["receiver_monotonic_time"] = latest_receiver_monotonic_time
                    live_payload["availability_monotonic_time"] = latest_receiver_monotonic_time
                    live_payload["source_mapping_generation"] = (
                        self._source_mapping_generation
                    )
                    live_payload.update(
                        self._clock_sync_metadata_for_payload(
                            live_payload,
                            sender_host=latest_sender_host,
                            availability_monotonic_time=latest_receiver_monotonic_time,
                        )
                    )
                    history = self._history_by_channel.get(channel_name)
                    if history:
                        live_payload["history_time"] = history[-1]["history_time"]
                        live_payload["history_clock_source"] = history[-1][
                            "history_clock_source"
                        ]
                        for key in (
                            "source_timestamp",
                            "source_clock",
                            "available_timestamp",
                            "available_clock",
                        ):
                            live_payload[key] = history[-1].get(key)
                    self._latest_by_channel[channel_name] = live_payload
                    self._channel_update_counts[channel_name] = (
                        self._channel_update_counts.get(channel_name, 0) + 1
                    )
                    self._last_channel_update_at[channel_name] = latest_received_at
        finally:
            finished_wall = time.time()
            with self._lock:
                self._last_process_finished_at = finished_wall
                self._last_process_finished_monotonic_time = time.perf_counter()
                self._last_process_duration_sec = time.perf_counter() - process_started_at
                self._last_process_parse_error_count = parse_error_count

    @staticmethod
    def _packet_seq(parsed: dict[str, dict[str, Any]]) -> int | None:
        for payload in parsed.values():
            seq = payload.get("seq")
            if isinstance(seq, int) and not isinstance(seq, bool):
                return int(seq)
        return None

    @staticmethod
    def _copy_tracker_payload(payload: dict[str, Any]) -> dict[str, Any]:
        copied = dict(payload)
        pose = copied.get("pose")
        if pose is not None:
            copied["pose"] = np.asarray(pose, dtype=float).copy()
        for field_name in ("linear_velocity_m_s", "angular_velocity_rad_s"):
            value = copied.get(field_name)
            if value is not None:
                copied[field_name] = list(value)
        return copied

    @staticmethod
    def _finite_timestamp(value: Any, *, fallback: float) -> float:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        return candidate if np.isfinite(candidate) else float(fallback)

    def _clock_sync_metadata_for_payload(
        self,
        payload: dict[str, Any],
        *,
        sender_host: str | None,
        availability_monotonic_time: float,
    ) -> dict[str, Any]:
        """Return diagnostic capture-time metadata without affecting causality."""

        if not self.clock_sync_enabled:
            return {
                "clock_sync_status": "disabled",
                "clock_sync_sender_host": sender_host,
                "clock_sync_sender_instance_id": None,
                "capture_receiver_monotonic_estimate": None,
                "capture_estimate_uncertainty_sec": None,
                "clock_sync_offset_sec": None,
                "clock_sync_round_trip_sec": None,
                "clock_sync_sample_count": 0,
            }
        state = self._clock_sync_server.peer_state(
            sender_host,
            now_collector_monotonic_ns=int(
                float(availability_monotonic_time) * 1_000_000_000
            ),
        )
        estimate = state.estimate
        capture_time = self._optional_finite_float(
            payload.get("pose_sample_monotonic_time")
        )
        capture_estimate: float | None = None
        uncertainty_sec: float | None = None
        offset_sec: float | None = None
        round_trip_sec: float | None = None
        if estimate is not None:
            uncertainty_sec = estimate.uncertainty_ns / 1_000_000_000.0
            offset_sec = estimate.collector_minus_sender_offset_ns / 1_000_000_000.0
            round_trip_sec = estimate.round_trip_ns / 1_000_000_000.0
            if state.status == "synchronized" and capture_time is not None:
                capture_ns = int(round(capture_time * 1_000_000_000.0))
                capture_estimate = (
                    estimate.sender_to_collector_monotonic_ns(capture_ns)
                    / 1_000_000_000.0
                )
        return {
            "clock_sync_status": state.status,
            "clock_sync_sender_host": state.sender_host,
            "clock_sync_sender_instance_id": (
                None if estimate is None else estimate.sender_instance_id
            ),
            "capture_receiver_monotonic_estimate": capture_estimate,
            "capture_estimate_uncertainty_sec": uncertainty_sec,
            "clock_sync_offset_sec": offset_sec,
            "clock_sync_round_trip_sec": round_trip_sec,
            "clock_sync_sample_count": 0 if estimate is None else estimate.sample_count,
        }

    def _clear_history_locked(self, *, reason: str) -> None:
        had_history = any(self._history_by_channel.values())
        self._history_by_channel = {}
        self._history_clock_source = None
        self._history_last_time = None
        self._history_last_seq = None
        self._history_last_receiver_monotonic_time = None
        if had_history:
            self._history_reset_count += 1
            self._history_last_reset_reason = reason

    def _append_history_packet_locked(
        self,
        parsed: dict[str, dict[str, Any]],
        *,
        received_at: float,
        receiver_monotonic_time: float,
    ) -> None:
        if self.history_window_sec <= 0.0:
            return

        seq = self._packet_seq(parsed)
        reset_reason: str | None = None
        if (
            seq is not None
            and self._history_last_seq is not None
            and seq == self._history_last_seq
        ):
            self._history_duplicate_packet_count += 1
            return
        if (
            seq is not None
            and self._history_last_seq is not None
            and seq < self._history_last_seq
        ):
            if seq == 0:
                reset_reason = "sender_session_restart"
            else:
                self._history_out_of_order_packet_count += 1
                return
        elif (
            self.history_reset_gap_sec > 0.0
            and self._history_last_receiver_monotonic_time is not None
            and receiver_monotonic_time - self._history_last_receiver_monotonic_time
            > self.history_reset_gap_sec
        ):
            reset_reason = "receiver_gap"

        if (
            reset_reason is None
            and seq is not None
            and self._history_last_seq is not None
            and seq > self._history_last_seq + 1
        ):
            self._history_missing_seq_count += seq - self._history_last_seq - 1

        if reset_reason is not None:
            self._clear_history_locked(reason=reset_reason)
            if reset_reason == "sender_session_restart":
                for payload in parsed.values():
                    self._clock_sync_server.invalidate_peer(payload.get("sender_host"))

        # ``receiver_monotonic_time`` is the availability time in the same
        # local clock domain as T0.  Sender perf-counter values are retained as
        # capture diagnostics only; they must never order this history.
        history_time = receiver_monotonic_time
        clock_source = "receiver_monotonic_time"
        self._history_clock_source = clock_source

        cutoff = history_time - self.history_window_sec
        for channel_name, tracker_payload in parsed.items():
            history_sample = self._copy_tracker_payload(tracker_payload)
            history_sample.update(
                {
                    "receiver_timestamp": received_at,
                    "receiver_monotonic_time": receiver_monotonic_time,
                    "availability_monotonic_time": receiver_monotonic_time,
                    "history_time": history_time,
                    "history_clock_source": clock_source,
                    "source_mapping_generation": self._source_mapping_generation,
                }
            )
            history_sample.update(
                self._clock_sync_metadata_for_payload(
                    history_sample,
                    sender_host=history_sample.get("sender_host"),
                    availability_monotonic_time=receiver_monotonic_time,
                )
            )
            calibrated_capture = self._optional_finite_float(
                history_sample.get("capture_receiver_monotonic_estimate")
            )
            if calibrated_capture is not None:
                # Clock-sync maps sender perf-counter time into the receiver
                # perf-counter domain.  Convert it into the wall-clock domain
                # used by T0 using this packet's receive pair.
                source_timestamp = float(
                    calibrated_capture
                    + float(received_at)
                    - float(receiver_monotonic_time)
                )
                source_clock = "vive_sender_monotonic_calibrated_to_local_wall"
            else:
                # Legacy/unsynchronized senders remain compatible, but the
                # fallback is explicit: source is local receive time.
                source_timestamp = float(received_at)
                source_clock = "vive_receiver_wall_fallback"
            history_sample.update(
                {
                    "source_timestamp": source_timestamp,
                    "source_clock": source_clock,
                    # ``received_at`` is captured at the socket boundary and
                    # is the causal availability instant.  Calling time.time()
                    # again here can move the sample a few microseconds into
                    # the future relative to a same-cycle cutoff.
                    "available_timestamp": float(received_at),
                    "available_clock": "local_wall_time",
                }
            )
            history = self._history_by_channel.setdefault(channel_name, deque())
            history.append(history_sample)

        for history in self._history_by_channel.values():
            while history and float(history[0]["history_time"]) < cutoff:
                history.popleft()

        self._history_last_time = history_time
        self._history_last_seq = seq
        self._history_last_receiver_monotonic_time = receiver_monotonic_time
        self._history_packet_count += 1

    def _parse_packet(
        self,
        packet: bytes,
        received_at: float,
        *,
        sender_host: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        json_payload = self._parse_json_packet(
            packet,
            received_at,
            sender_host=sender_host,
        )
        if json_payload is not None:
            return json_payload
        legacy_payload = self._parse_legacy_packet(
            packet,
            received_at,
            sender_host=sender_host,
        )
        if legacy_payload is not None:
            return legacy_payload
        return {}

    def _parse_json_packet(
        self,
        packet: bytes,
        received_at: float,
        *,
        sender_host: str | None = None,
    ) -> dict[str, dict[str, Any]] | None:
        decoded = self._decode_json_packet(packet)
        if decoded is None:
            return None
        return self._parse_decoded_json_packet(
            decoded,
            received_at,
            sender_host=sender_host,
        )

    @staticmethod
    def _decode_json_packet(packet: bytes) -> dict[str, Any] | None:
        try:
            decoded = json.loads(packet.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return decoded if isinstance(decoded, dict) else None

    def _parse_decoded_json_packet(
        self,
        decoded: dict[str, Any],
        received_at: float,
        *,
        sender_host: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        if "trackers" in decoded:
            seq = decoded.get("seq")
            sender_timestamp = decoded.get("sender_timestamp", received_at)
            pose_sample_timestamp = decoded.get(
                "pose_sample_timestamp",
                sender_timestamp,
            )
            pose_sample_monotonic_time = decoded.get(
                "pose_sample_monotonic_time"
            )
            trackers = decoded.get("trackers", [])
        else:
            seq = decoded.get("seq")
            sender_timestamp = decoded.get("sender_timestamp", received_at)
            pose_sample_timestamp = decoded.get(
                "pose_sample_timestamp",
                sender_timestamp,
            )
            pose_sample_monotonic_time = decoded.get(
                "pose_sample_monotonic_time"
            )
            trackers = [decoded]

        parsed: dict[str, dict[str, Any]] = {}
        for tracker in trackers:
            if not isinstance(tracker, dict):
                continue
            tracker_name = str(tracker.get("tracker_name") or tracker.get("name") or "")
            tracker_role = str(tracker.get("tracker_role") or tracker.get("role") or "")
            config = self._resolve_tracker_config(
                tracker_name=tracker_name,
                tracker_role=tracker_role,
            )
            if config is None:
                continue
            pose = tracker.get("pose_matrix_4x4") or tracker.get("pose")
            try:
                pose_matrix = self._coerce_pose_matrix(pose)
            except ValueError:
                pose_matrix = identity_pose()
                valid = False
            else:
                valid = bool(tracker.get("valid", True))

            velocity_error = tracker.get("velocity_error")
            linear_velocity: list[float] | None = None
            angular_velocity: list[float] | None = None
            try:
                linear_velocity = self._coerce_optional_vector3(
                    tracker.get("linear_velocity_m_s"),
                    name="linear_velocity_m_s",
                )
                angular_velocity = self._coerce_optional_vector3(
                    tracker.get("angular_velocity_rad_s"),
                    name="angular_velocity_rad_s",
                )
            except ValueError as exc:
                velocity_error = f"receiver_validation: {exc}"
                linear_velocity = None
                angular_velocity = None
            velocity_valid = bool(
                tracker.get(
                    "velocity_valid",
                    linear_velocity is not None and angular_velocity is not None,
                )
                and linear_velocity is not None
                and angular_velocity is not None
            )
            if bool(tracker.get("velocity_valid", False)) and not velocity_valid:
                velocity_error = (
                    velocity_error
                    or "receiver_validation: velocity vectors missing"
                )

            parsed[config.channel_name] = {
                "pose": pose_matrix,
                "timestamp": float(sender_timestamp),
                "pose_sample_timestamp": self._optional_finite_float(
                    pose_sample_timestamp
                ),
                "pose_sample_monotonic_time": self._optional_finite_float(
                    pose_sample_monotonic_time
                ),
                "receiver_timestamp": received_at,
                "sender_host": sender_host,
                "seq": seq,
                "tracker_name": tracker_name or config.name,
                "role": tracker_role or config.role,
                "valid": valid,
                "error": tracker.get("error"),
                "sender_read_elapsed_ms": tracker.get("read_elapsed_ms"),
                "linear_velocity_m_s": linear_velocity,
                "angular_velocity_rad_s": angular_velocity,
                "velocity_frame_name": tracker.get(
                    "velocity_frame_name",
                    "vive_world",
                ),
                "velocity_source": tracker.get("velocity_source"),
                "velocity_valid": velocity_valid,
                "velocity_error": velocity_error,
                "config": config,
            }
        return parsed

    def _parse_decoded_upstream_role_diagnostics(
        self,
        decoded: dict[str, Any],
        received_at: float,
        *,
        sender_host: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        seq = decoded.get("seq")
        sender_timestamp = decoded.get("sender_timestamp", received_at)
        trackers = (
            decoded.get("trackers", [])
            if "trackers" in decoded
            else [decoded]
        )
        if not isinstance(trackers, list):
            return {}

        parsed: dict[str, dict[str, Any]] = {}
        for tracker in trackers:
            if not isinstance(tracker, dict):
                continue
            tracker_role = str(
                tracker.get("tracker_role") or tracker.get("role") or ""
            ).strip()
            if not tracker_role:
                continue
            tracker_name = str(
                tracker.get("tracker_name") or tracker.get("name") or ""
            )
            pose = tracker.get("pose_matrix_4x4") or tracker.get("pose")
            sender_error = tracker.get("error")
            try:
                pose_matrix = self._coerce_pose_matrix(pose)
            except ValueError as exc:
                pose_matrix = identity_pose()
                valid = False
                if not sender_error:
                    sender_error = f"receiver_validation: {exc}"
            else:
                valid = bool(tracker.get("valid", True))
            parsed[tracker_role] = {
                "pose": pose_matrix,
                "timestamp": float(sender_timestamp),
                "receiver_timestamp": float(received_at),
                "sender_host": sender_host,
                "seq": seq,
                "tracker_name": tracker_name,
                "role": tracker_role,
                "valid": valid,
                "error": sender_error,
                "sender_read_elapsed_ms": tracker.get("read_elapsed_ms"),
            }
        return parsed

    def _parse_legacy_packet(
        self,
        packet: bytes,
        received_at: float,
        *,
        sender_host: str | None = None,
    ) -> dict[str, dict[str, Any]] | None:
        if len(packet) != self._legacy_packet_size:
            return None
        values = struct.unpack(self._legacy_format, packet)
        parsed: dict[str, dict[str, Any]] = {}
        for config in self._tracker_configs:
            if config.legacy_index is None:
                continue
            start = config.legacy_index * 12
            chunk = np.asarray(values[start : start + 12], dtype=float)
            pose_matrix = identity_pose()
            pose_matrix[:3, 3] = chunk[:3]
            pose_matrix[:3, :3] = chunk[3:].reshape(3, 3)
            valid = not np.allclose(chunk, 0.0)
            parsed[config.channel_name] = {
                "pose": pose_matrix,
                "timestamp": received_at,
                "pose_sample_timestamp": None,
                "pose_sample_monotonic_time": None,
                "receiver_timestamp": received_at,
                "sender_host": sender_host,
                "seq": None,
                "tracker_name": config.name,
                "role": config.role,
                "valid": valid,
                "linear_velocity_m_s": None,
                "angular_velocity_rad_s": None,
                "velocity_frame_name": "vive_world",
                "velocity_source": None,
                "velocity_valid": False,
                "velocity_error": "legacy_packet_has_no_velocity",
                "config": config,
            }
        return parsed

    def _resolve_tracker_config(
        self, *, tracker_name: str, tracker_role: str
    ) -> ViveTrackerConfig | None:
        # A declared v1 role is authoritative.  Falling back to tracker name
        # after a role mismatch would silently defeat an explicit role swap
        # (or any other role-based routing profile).  Name matching remains a
        # compatibility path only for older JSON records that omit the role.
        if tracker_role:
            return self._configs_by_role.get(tracker_role)
        if tracker_name and tracker_name in self.tracker_configs:
            return self.tracker_configs[tracker_name]
        return None

    def _coerce_pose_matrix(self, pose: Any) -> np.ndarray:
        array = np.asarray(pose, dtype=float)
        if array.shape == (4, 4):
            return array.copy()
        if array.shape == (16,):
            return array.reshape(4, 4)
        raise ValueError(f"Unsupported pose shape for VIVE packet: {array.shape}")

    def _coerce_optional_vector3(
        self,
        value: Any,
        *,
        name: str,
    ) -> list[float] | None:
        if value is None:
            return None
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain finite values")
        return vector.astype(float).tolist()

    def _optional_finite_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result) else None

    def get_latest_channels(
        self,
        *,
        cutoff_timestamp: float | None = None,
    ) -> dict[str, ObservationChannel]:
        channels: dict[str, ObservationChannel] = {}
        with self._lock:
            cutoff_wall = None if cutoff_timestamp is None else float(cutoff_timestamp)
            if cutoff_wall is not None and not np.isfinite(cutoff_wall):
                raise ValueError("cutoff_timestamp must be finite")
            if cutoff_wall is None:
                latest = dict(self._latest_by_channel)
            else:
                latest = {}
                for channel_name, history in self._history_by_channel.items():
                    for candidate in reversed(history):
                        source = self._optional_finite_float(candidate.get("source_timestamp"))
                        available = self._optional_finite_float(candidate.get("available_timestamp"))
                        if source is None or available is None:
                            continue
                        if source <= cutoff_wall and available <= cutoff_wall:
                            latest[channel_name] = self._copy_tracker_payload(candidate)
                            break
            tracker_configs = list(self._tracker_configs)
            motion_histories: dict[str, list[dict[str, Any]]] = {}
            for channel_name, history in self._history_by_channel.items():
                if not history:
                    motion_histories[channel_name] = []
                    continue
                motion_cutoff = (
                    float(history[-1]["history_time"])
                    - self.motion_window_sec
                )
                if cutoff_wall is not None and channel_name in latest:
                    motion_cutoff = (
                        float(latest[channel_name].get("history_time", history[-1]["history_time"]))
                        - self.motion_window_sec
                    )
                selected_reversed: list[dict[str, Any]] = []
                for sample in reversed(history):
                    if float(sample["history_time"]) < motion_cutoff:
                        break
                    if cutoff_wall is not None:
                        source = self._optional_finite_float(sample.get("source_timestamp"))
                        available = self._optional_finite_float(sample.get("available_timestamp"))
                        if source is None or available is None or source > cutoff_wall or available > cutoff_wall:
                            continue
                    selected_reversed.append(self._copy_tracker_payload(sample))
                motion_histories[channel_name] = list(reversed(selected_reversed))
            history_reset_count = self._history_reset_count
            history_last_reset_reason = self._history_last_reset_reason

        for config in tracker_configs:
            tracker_payload = latest.get(config.channel_name)
            if tracker_payload is None:
                continue
            receiver_timestamp = tracker_payload.get("receiver_timestamp")
            sender_timestamp = tracker_payload.get("timestamp")
            motion_window = summarize_tracker_motion_window(
                motion_histories.get(config.channel_name, ()),
                window_sec=self.motion_window_sec,
            )
            motion_window.update(
                {
                    "history_reset_count": int(history_reset_count),
                    "history_last_reset_reason": history_last_reset_reason,
                }
            )
            channel_timestamp = (
                float(receiver_timestamp)
                if receiver_timestamp is not None
                else float(sender_timestamp)
            )
            channels[config.channel_name] = ObservationChannel(
                channel_name=config.channel_name,
                capability="pose6d",
                side=config.side,
                value=np.asarray(tracker_payload["pose"], dtype=float).copy(),
                frame_name="vive_world",
                timestamp=channel_timestamp,
                source_name=self.name,
                valid=bool(tracker_payload["valid"]),
                critical=config.critical,
                metadata={
                    "tracker_name": tracker_payload.get("tracker_name", config.name),
                    "tracker_role": tracker_payload.get("role", config.role),
                    "seq": tracker_payload.get("seq"),
                    "receiver_timestamp": receiver_timestamp,
                    "receiver_monotonic_time": tracker_payload.get(
                        "receiver_monotonic_time"
                    ),
                    "availability_monotonic_time": tracker_payload.get(
                        "availability_monotonic_time"
                    ),
                    "source_mapping_generation": tracker_payload.get(
                        "source_mapping_generation",
                        self._source_mapping_generation,
                    ),
                    "source_timestamp": tracker_payload.get("source_timestamp"),
                    "source_clock": tracker_payload.get("source_clock"),
                    "available_timestamp": tracker_payload.get("available_timestamp"),
                    "available_clock": tracker_payload.get("available_clock"),
                    "history_time": tracker_payload.get("history_time"),
                    "history_clock_source": tracker_payload.get(
                        "history_clock_source"
                    ),
                    "sender_timestamp": sender_timestamp,
                    "pose_sample_timestamp": tracker_payload.get(
                        "pose_sample_timestamp"
                    ),
                    "pose_sample_monotonic_time": tracker_payload.get(
                        "pose_sample_monotonic_time"
                    ),
                    "clock_sync_status": tracker_payload.get("clock_sync_status"),
                    "clock_sync_sender_host": tracker_payload.get(
                        "clock_sync_sender_host"
                    ),
                    "clock_sync_sender_instance_id": tracker_payload.get(
                        "clock_sync_sender_instance_id"
                    ),
                    "capture_receiver_monotonic_estimate": tracker_payload.get(
                        "capture_receiver_monotonic_estimate"
                    ),
                    "capture_estimate_uncertainty_sec": tracker_payload.get(
                        "capture_estimate_uncertainty_sec"
                    ),
                    "clock_sync_offset_sec": tracker_payload.get(
                        "clock_sync_offset_sec"
                    ),
                    "clock_sync_round_trip_sec": tracker_payload.get(
                        "clock_sync_round_trip_sec"
                    ),
                    "clock_sync_sample_count": tracker_payload.get(
                        "clock_sync_sample_count"
                    ),
                    "sender_read_elapsed_ms": tracker_payload.get(
                        "sender_read_elapsed_ms"
                    ),
                    "sender_error": tracker_payload.get("error"),
                    "linear_velocity_m_s": tracker_payload.get(
                        "linear_velocity_m_s"
                    ),
                    "angular_velocity_rad_s": tracker_payload.get(
                        "angular_velocity_rad_s"
                    ),
                    "velocity_frame_name": tracker_payload.get(
                        "velocity_frame_name"
                    ),
                    "velocity_source": tracker_payload.get("velocity_source"),
                    "velocity_valid": bool(
                        tracker_payload.get("velocity_valid", False)
                    ),
                    "velocity_error": tracker_payload.get("velocity_error"),
                    "tracker_motion_window": motion_window,
                },
            )
        return channels

    def get_latest_channels_at_or_before(
        self,
        cutoff_timestamp: float,
    ) -> dict[str, ObservationChannel]:
        """Return VIVE channels satisfying both causal timestamp bounds."""

        return self.get_latest_channels(cutoff_timestamp=float(cutoff_timestamp))

    def get_latest_upstream_role_channels(self) -> dict[str, ObservationChannel]:
        """Return diagnostic-only channels keyed by sender tracker role.

        Unlike :meth:`get_latest_channels`, this snapshot is taken before the
        source-role to logical-channel mapping.  Callers may use it for status
        display and fault diagnosis only; it is not an observation-source
        output and must not feed retargeting, composition, or execution.
        """

        with self._lock:
            latest = {
                role: self._copy_tracker_payload(payload)
                for role, payload in self._latest_upstream_by_role.items()
            }

        channels: dict[str, ObservationChannel] = {}
        for tracker_role, payload in latest.items():
            receiver_timestamp = payload.get("receiver_timestamp")
            sender_timestamp = payload.get("timestamp")
            timestamp = (
                float(receiver_timestamp)
                if receiver_timestamp is not None
                else float(sender_timestamp)
            )
            side = (
                "left"
                if tracker_role == "left_wrist"
                else "right"
                if tracker_role == "right_wrist"
                else None
            )
            channels[tracker_role] = ObservationChannel(
                channel_name=f"vive_upstream_role:{tracker_role}",
                capability="pose6d",
                side=side,
                value=np.asarray(payload["pose"], dtype=float).copy(),
                frame_name="vive_world",
                timestamp=timestamp,
                source_name=self.name,
                valid=bool(payload.get("valid", False)),
                critical=False,
                metadata={
                    "diagnostic_only": True,
                    "tracker_name": payload.get("tracker_name", ""),
                    "tracker_role": tracker_role,
                    "seq": payload.get("seq"),
                    "receiver_timestamp": receiver_timestamp,
                    "receiver_monotonic_time": payload.get(
                        "receiver_monotonic_time"
                    ),
                    "sender_timestamp": sender_timestamp,
                    "sender_host": payload.get("sender_host"),
                    "sender_error": payload.get("error"),
                    "sender_read_elapsed_ms": payload.get(
                        "sender_read_elapsed_ms"
                    ),
                },
            )
        return channels

    def get_pose_history(
        self,
        channel_name: str,
        *,
        window_sec: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return a snapshot selected by elapsed time, never by frame count.

        ``window_sec`` is measured backwards from the newest receiver-local
        availability timestamp.  Irregular 300 Hz packet spacing therefore
        changes the number of returned samples, not the represented duration.
        """

        if window_sec is not None and float(window_sec) < 0.0:
            raise ValueError("window_sec must be >= 0")
        with self._lock:
            history = list(self._history_by_channel.get(channel_name, ()))
        if not history:
            return []
        if window_sec is not None:
            cutoff = float(history[-1]["history_time"]) - float(window_sec)
            history = [
                sample
                for sample in history
                if float(sample["history_time"]) >= cutoff
            ]
        return [self._copy_tracker_payload(sample) for sample in history]

    def get_pose_snapshot_at_or_before(
        self,
        channel_name: str,
        cutoff_monotonic_time: float,
        *,
        valid_only: bool = False,
    ) -> dict[str, Any] | None:
        """Freeze the newest locally available tracker sample at or before T0.

        ``cutoff_monotonic_time`` must be in the collector's
        ``time.perf_counter`` domain.  Sender capture timestamps, including a
        calibrated estimate, are intentionally not used for this causal query.
        """

        cutoff = float(cutoff_monotonic_time)
        if not np.isfinite(cutoff):
            raise ValueError("cutoff_monotonic_time must be finite")
        with self._lock:
            history = self._history_by_channel.get(channel_name, ())
            for sample in reversed(history):
                availability = sample.get("availability_monotonic_time")
                if availability is None or float(availability) > cutoff:
                    continue
                if valid_only and not bool(sample.get("valid", False)):
                    continue
                return self._copy_tracker_payload(sample)
        return None

    def debug_stats(self) -> dict[str, Any]:
        clock_sync_stats = self._clock_sync_server.debug_stats()
        thread = self._thread
        with self._lock:
            return {
                "running": self._running,
                "worker_alive": bool(thread is not None and thread.is_alive()),
                "worker_started_once": self._worker_started_once,
                "bind_host": self.host,
                "port": self.port,
                "allow_socket_reuse": self.allow_socket_reuse,
                "source_mapping_generation": self._source_mapping_generation,
                "packet_count": self._packet_count,
                "datagram_count": self._datagram_count,
                "last_packet_received_at": self._last_packet_received_at,
                "last_packet_received_monotonic_time": (
                    self._last_packet_received_monotonic_time
                ),
                "last_datagram_received_at": self._last_datagram_received_at,
                "last_datagram_received_monotonic_time": (
                    self._last_datagram_received_monotonic_time
                ),
                "dropped_stale_packet_count": self._dropped_stale_packet_count,
                "last_drain_count": self._last_drain_count,
                "channel_update_counts": dict(self._channel_update_counts),
                "last_channel_update_at": dict(self._last_channel_update_at),
                "upstream_role_update_counts": dict(
                    self._upstream_role_update_counts
                ),
                "last_upstream_role_update_at": dict(
                    self._last_upstream_role_update_at
                ),
                "latest_upstream_roles": sorted(
                    self._latest_upstream_by_role
                ),
                "history_window_sec": self.history_window_sec,
                "motion_window_sec": self.motion_window_sec,
                "history_clock_source": self._history_clock_source,
                "history_packet_count": self._history_packet_count,
                "history_sample_counts": {
                    channel_name: len(history)
                    for channel_name, history in self._history_by_channel.items()
                },
                "history_reset_count": self._history_reset_count,
                "history_last_reset_reason": self._history_last_reset_reason,
                "history_duplicate_packet_count": self._history_duplicate_packet_count,
                "history_out_of_order_packet_count": (
                    self._history_out_of_order_packet_count
                ),
                "history_missing_seq_count": self._history_missing_seq_count,
                "recv_error_count": self._recv_error_count,
                "last_recv_error": self._last_recv_error,
                "socket_restart_count": self._socket_restart_count,
                "last_socket_restart_at": self._last_socket_restart_at,
                "process_error_count": self._process_error_count,
                "last_process_error": self._last_process_error,
                "last_process_error_at": self._last_process_error_at,
                "last_process_started_at": self._last_process_started_at,
                "last_process_finished_at": self._last_process_finished_at,
                "last_process_finished_monotonic_time": (
                    self._last_process_finished_monotonic_time
                ),
                "last_process_duration_sec": self._last_process_duration_sec,
                "last_process_batch_size": self._last_process_batch_size,
                "last_process_parse_error_count": (
                    self._last_process_parse_error_count
                ),
                "clock_sync": clock_sync_stats,
            }
