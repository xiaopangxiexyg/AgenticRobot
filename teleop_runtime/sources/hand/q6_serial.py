from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from ...interfaces import ObservationSource
from ...schemas import ObservationChannel


@dataclass(frozen=True)
class _HandReaderSpec:
    side: str
    port: str
    baudrate: int = 115200
    timeout: float = 0.05
    show_raw_lines: bool = False


class HandQ6Source(ObservationSource):
    def __init__(
        self,
        *,
        reader_specs: dict[str, _HandReaderSpec],
        name: str = "hand_q6_serial",
        critical: bool = False,
        strict_expected: bool = False,
    ):
        if not reader_specs:
            raise ValueError("HandQ6Source requires at least one reader spec.")
        self._reader_specs = dict(reader_specs)
        self._name = name
        self._critical = bool(critical)
        self._strict_expected = bool(strict_expected)
        self._expected = tuple(
            f"{side}_hand_q6" for side in ("left", "right") if side in self._reader_specs
        )
        self._lock = threading.Lock()
        self._latest_channels: dict[str, ObservationChannel] = {}
        self._readers: dict[str, object] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._running = False

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_single_hand(
        cls,
        *,
        port: str,
        side: str,
        baudrate: int = 115200,
        timeout: float = 0.05,
        show_raw_lines: bool = False,
        critical: bool = False,
        strict_expected: bool = False,
    ) -> "HandQ6Source":
        spec = _HandReaderSpec(
            side=str(side),
            port=str(port),
            baudrate=int(baudrate),
            timeout=float(timeout),
            show_raw_lines=bool(show_raw_lines),
        )
        return cls(
            reader_specs={spec.side: spec},
            name=f"hand_q6_{spec.side}",
            critical=critical,
            strict_expected=strict_expected,
        )

    @classmethod
    def from_bimanual(
        cls,
        *,
        left_port: str | None = None,
        right_port: str | None = None,
        baudrate: int = 115200,
        timeout: float = 0.05,
        show_raw_lines: bool = False,
        critical: bool = False,
        strict_expected: bool = False,
    ) -> "HandQ6Source":
        reader_specs: dict[str, _HandReaderSpec] = {}
        if left_port is not None:
            reader_specs["left"] = _HandReaderSpec(
                side="left",
                port=str(left_port),
                baudrate=int(baudrate),
                timeout=float(timeout),
                show_raw_lines=bool(show_raw_lines),
            )
        if right_port is not None:
            reader_specs["right"] = _HandReaderSpec(
                side="right",
                port=str(right_port),
                baudrate=int(baudrate),
                timeout=float(timeout),
                show_raw_lines=bool(show_raw_lines),
            )
        return cls(
            reader_specs=reader_specs,
            name="hand_q6_bimanual",
            critical=critical,
            strict_expected=strict_expected,
        )

    def expected_channel_names(self) -> tuple[str, ...]:
        return self._expected if self._strict_expected else ()

    def start(self) -> None:
        if self._running:
            return

        from hardware.teleop_hand.hand_q6_reader import SingleHandQ6Reader

        readers: dict[str, object] = {}
        startup_errors: dict[str, Exception] = {}
        for side, spec in self._reader_specs.items():
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
                if self._critical or self._strict_expected:
                    for reader in readers.values():
                        try:
                            reader.close()
                        except Exception:
                            pass
                    raise

        self._readers = readers
        self._running = True
        self._threads = {}
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

    def _channel_name_for_side(self, side: str) -> str:
        return f"{side}_hand_q6"

    def _channel_from_packet(self, side: str, packet: dict[str, object]) -> ObservationChannel:
        channel_name = self._channel_name_for_side(side)
        timestamp = float(np.asarray(packet.get("timestamp", [0.0]), dtype=float).ravel()[0])
        return ObservationChannel(
            channel_name=channel_name,
            capability="joint_vector",
            side=side,
            value=np.asarray(packet["norm"], dtype=float).copy(),
            frame_name=None,
            timestamp=timestamp,
            source_name=self.name,
            valid=True,
            critical=self._critical,
            metadata={
                "port": packet.get("port"),
                "baudrate": packet.get("baudrate"),
            },
        )

    def _invalid_channel(self, side: str, *, error: str) -> ObservationChannel:
        channel_name = self._channel_name_for_side(side)
        last_value = np.zeros(6, dtype=float)
        previous = self._latest_channels.get(channel_name)
        if previous is not None:
            last_value = np.asarray(previous.value, dtype=float).reshape(6).copy()
        spec = self._reader_specs[side]
        return ObservationChannel(
            channel_name=channel_name,
            capability="joint_vector",
            side=side,
            value=last_value,
            frame_name=None,
            timestamp=time.time(),
            source_name=self.name,
            valid=False,
            critical=self._critical,
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
