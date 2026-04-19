from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...interfaces import ObservationSource
from ...schemas import ObservationChannel
from ...transforms import identity_pose


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
        name: str = "vive_pose",
    ):
        self._name = name
        self.host = host
        self.port = int(port)
        self.legacy_tracker_count = int(legacy_tracker_count)
        self.socket_timeout_sec = float(socket_timeout_sec)
        self._tracker_configs = list(tracker_configs)
        self.tracker_configs = {config.name: config for config in tracker_configs}
        self._configs_by_role = {config.role: config for config in tracker_configs}
        self.sock: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_by_channel: dict[str, dict[str, Any]] = {}
        self._legacy_format = f"<{self.legacy_tracker_count * 12}f"
        self._legacy_packet_size = struct.calcsize(self._legacy_format)

    @property
    def name(self) -> str:
        return self._name

    def expected_channel_names(self) -> tuple[str, ...]:
        return tuple(
            config.channel_name for config in self._tracker_configs if config.critical
        )

    def start(self) -> None:
        if self._running:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(self.socket_timeout_sec)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _recv_loop(self) -> None:
        while self._running and self.sock is not None:
            try:
                packet, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break

            received_at = time.time()
            parsed = self._parse_packet(packet, received_at)
            if not parsed:
                continue

            with self._lock:
                for channel_name, tracker_payload in parsed.items():
                    self._latest_by_channel[channel_name] = tracker_payload

    def _parse_packet(self, packet: bytes, received_at: float) -> dict[str, dict[str, Any]]:
        json_payload = self._parse_json_packet(packet, received_at)
        if json_payload is not None:
            return json_payload
        legacy_payload = self._parse_legacy_packet(packet, received_at)
        if legacy_payload is not None:
            return legacy_payload
        return {}

    def _parse_json_packet(
        self, packet: bytes, received_at: float
    ) -> dict[str, dict[str, Any]] | None:
        try:
            decoded = json.loads(packet.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        if isinstance(decoded, dict) and "trackers" in decoded:
            seq = decoded.get("seq")
            sender_timestamp = decoded.get("sender_timestamp", received_at)
            trackers = decoded.get("trackers", [])
        elif isinstance(decoded, dict):
            seq = decoded.get("seq")
            sender_timestamp = decoded.get("sender_timestamp", received_at)
            trackers = [decoded]
        else:
            return None

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

            parsed[config.channel_name] = {
                "pose": pose_matrix,
                "timestamp": float(sender_timestamp),
                "receiver_timestamp": received_at,
                "seq": seq,
                "tracker_name": tracker_name or config.name,
                "role": tracker_role or config.role,
                "valid": valid,
                "error": tracker.get("error"),
                "config": config,
            }
        return parsed

    def _parse_legacy_packet(
        self, packet: bytes, received_at: float
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
                "receiver_timestamp": received_at,
                "seq": None,
                "tracker_name": config.name,
                "role": config.role,
                "valid": valid,
                "config": config,
            }
        return parsed

    def _resolve_tracker_config(
        self, *, tracker_name: str, tracker_role: str
    ) -> ViveTrackerConfig | None:
        if tracker_role and tracker_role in self._configs_by_role:
            return self._configs_by_role[tracker_role]
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

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        channels: dict[str, ObservationChannel] = {}
        with self._lock:
            latest = dict(self._latest_by_channel)

        for config in self._tracker_configs:
            tracker_payload = latest.get(config.channel_name)
            if tracker_payload is None:
                continue
            receiver_timestamp = tracker_payload.get("receiver_timestamp")
            sender_timestamp = tracker_payload.get("timestamp")
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
                    "sender_timestamp": sender_timestamp,
                    "sender_error": tracker_payload.get("error"),
                },
            )
        return channels
