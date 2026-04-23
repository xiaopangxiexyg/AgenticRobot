from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from ..common import ObservationChannel


@dataclass(frozen=True)
class HandReaderSpec:
    side: str
    port: str
    baudrate: int = 115200
    timeout: float = 0.05
    show_raw_lines: bool = False


class HandQ6Source:
    def __init__(
        self,
        *,
        reader_specs: dict[str, HandReaderSpec],
        name: str = "hand_q6_serial",
        critical: bool = False,
    ):
        if not reader_specs:
            raise ValueError("HandQ6Source requires at least one reader spec.")
        self.reader_specs = dict(reader_specs)
        self.name = str(name)
        self.critical = bool(critical)
        self._lock = threading.Lock()
        self._latest_channels: dict[str, ObservationChannel] = {}
        self._readers: dict[str, object] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._running = False

    @classmethod
    def from_ports(
        cls,
        *,
        left_port: str | None = None,
        right_port: str | None = None,
        baudrate: int = 115200,
        timeout: float = 0.05,
        show_raw_lines: bool = False,
        critical: bool = False,
    ) -> "HandQ6Source":
        specs: dict[str, HandReaderSpec] = {}
        if left_port:
            specs["left"] = HandReaderSpec(
                side="left",
                port=str(left_port),
                baudrate=int(baudrate),
                timeout=float(timeout),
                show_raw_lines=bool(show_raw_lines),
            )
        if right_port:
            specs["right"] = HandReaderSpec(
                side="right",
                port=str(right_port),
                baudrate=int(baudrate),
                timeout=float(timeout),
                show_raw_lines=bool(show_raw_lines),
            )
        return cls(reader_specs=specs, critical=critical)

    def expected_channel_names(self) -> tuple[str, ...]:
        return tuple(f"{side}_hand_q6" for side in ("left", "right") if side in self.reader_specs)

    def _channel_name_for_side(self, side: str) -> str:
        return f"{side}_hand_q6"

    def start(self) -> None:
        if self._running:
            return
        from .serial_hand_q6_reader import SingleHandQ6Reader

        readers: dict[str, object] = {}
        startup_errors: dict[str, Exception] = {}
        for side, spec in self.reader_specs.items():
            try:
                readers[side] = SingleHandQ6Reader(
                    port=spec.port,
                    side=side,
                    baudrate=spec.baudrate,
                    timeout=spec.timeout,
                    raw_only=False,
                    show_raw_lines=spec.show_raw_lines,
                )
            except Exception as exc:
                startup_errors[side] = exc
                if self.critical:
                    for reader in readers.values():
                        try:
                            reader.close()
                        except Exception:
                            pass
                    raise

        self._readers = readers
        self._running = True
        with self._lock:
            for side, exc in startup_errors.items():
                self._latest_channels[self._channel_name_for_side(side)] = self._invalid_channel(
                    side,
                    error=f"startup_failed: {exc}",
                )
        for side, reader in self._readers.items():
            thread = threading.Thread(
                target=self._reader_loop,
                args=(side, reader),
                name=f"{self.name}_{side}",
                daemon=True,
            )
            thread.start()
            self._threads[side] = thread

    def _channel_from_packet(self, side: str, packet: dict[str, object]) -> ObservationChannel:
        channel_name = self._channel_name_for_side(side)
        timestamp = float(np.asarray(packet.get("timestamp", [0.0]), dtype=float).reshape(-1)[0])
        return ObservationChannel(
            channel_name=channel_name,
            side=side,
            value=np.asarray(packet["norm"], dtype=float).copy(),
            timestamp=timestamp,
            valid=True,
            source_name=self.name,
            metadata={
                "port": packet.get("port"),
                "baudrate": packet.get("baudrate"),
            },
        )

    def _invalid_channel(self, side: str, *, error: str) -> ObservationChannel:
        channel_name = self._channel_name_for_side(side)
        previous = self._latest_channels.get(channel_name)
        last_value = (
            np.asarray(previous.value, dtype=float).reshape(6).copy()
            if previous is not None
            else np.zeros(6, dtype=float)
        )
        spec = self.reader_specs[side]
        return ObservationChannel(
            channel_name=channel_name,
            side=side,
            value=last_value,
            timestamp=time.time(),
            valid=False,
            source_name=self.name,
            metadata={
                "port": spec.port,
                "baudrate": spec.baudrate,
                "read_error": str(error),
            },
        )

    def _reader_loop(self, side: str, reader: object) -> None:
        while self._running:
            try:
                packet = reader.read_frame()
                channel = self._channel_from_packet(side, packet)
            except Exception as exc:
                with self._lock:
                    self._latest_channels[self._channel_name_for_side(side)] = self._invalid_channel(
                        side,
                        error=str(exc),
                    )
                return
            with self._lock:
                self._latest_channels[channel.channel_name] = channel

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        with self._lock:
            return {
                channel_name: channel.copy()
                for channel_name, channel in self._latest_channels.items()
            }

    def close(self) -> None:
        self._running = False
        for reader in self._readers.values():
            try:
                reader.close()
            except Exception:
                pass
        for thread in self._threads.values():
            thread.join(timeout=1.0)
        self._threads = {}
        self._readers = {}
