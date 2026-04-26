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
    exo_hand_calibration: bool = False
    calibration_path: str | None = None


class HandQ6Source:
    _READ_ERROR_RETRY_SEC = 0.25
    _READ_ERROR_LOG_INTERVAL_SEC = 2.0

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
        self._reader_error_counts: dict[str, int] = {}
        self._reader_recovery_counts: dict[str, int] = {}
        self._last_error_log_at: dict[str, float] = {}
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
        exo_hand_calibration: bool = False,
        calibration_path: str | None = None,
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
                exo_hand_calibration=bool(exo_hand_calibration),
                calibration_path=calibration_path,
            )
        if right_port:
            specs["right"] = HandReaderSpec(
                side="right",
                port=str(right_port),
                baudrate=int(baudrate),
                timeout=float(timeout),
                show_raw_lines=bool(show_raw_lines),
                exo_hand_calibration=bool(exo_hand_calibration),
                calibration_path=calibration_path,
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
                    exo_hand_calibration=spec.exo_hand_calibration,
                    calibration_path=spec.calibration_path,
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
                    metadata={"error_count": 1, "reader_running": False},
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

    def _channel_from_packet(
        self,
        side: str,
        packet: dict[str, object],
        *,
        recovered_after_errors: int = 0,
    ) -> ObservationChannel:
        channel_name = self._channel_name_for_side(side)
        timestamp = float(np.asarray(packet.get("timestamp", [0.0]), dtype=float).reshape(-1)[0])
        metadata = {
            "port": packet.get("port"),
            "baudrate": packet.get("baudrate"),
            "reader_error_count": int(self._reader_error_counts.get(side, 0)),
            "reader_recovery_count": int(self._reader_recovery_counts.get(side, 0)),
        }
        if recovered_after_errors > 0:
            metadata["recovered_after_errors"] = int(recovered_after_errors)
        return ObservationChannel(
            channel_name=channel_name,
            side=side,
            value=np.asarray(packet["norm"], dtype=float).copy(),
            timestamp=timestamp,
            valid=True,
            source_name=self.name,
            metadata=metadata,
        )

    def _invalid_channel(
        self,
        side: str,
        *,
        error: str,
        metadata: dict[str, object] | None = None,
    ) -> ObservationChannel:
        channel_name = self._channel_name_for_side(side)
        previous = self._latest_channels.get(channel_name)
        last_value = (
            np.asarray(previous.value, dtype=float).reshape(6).copy()
            if previous is not None
            else np.zeros(6, dtype=float)
        )
        spec = self.reader_specs[side]
        channel_metadata = {
            "port": spec.port,
            "baudrate": spec.baudrate,
            "read_error": str(error),
        }
        if metadata is not None:
            channel_metadata.update(metadata)
        return ObservationChannel(
            channel_name=channel_name,
            side=side,
            value=last_value,
            timestamp=time.time(),
            valid=False,
            source_name=self.name,
            metadata=channel_metadata,
        )

    def _log_reader_error(self, side: str, message: str) -> None:
        now = time.time()
        last = float(self._last_error_log_at.get(side, 0.0))
        if now - last < self._READ_ERROR_LOG_INTERVAL_SEC:
            return
        self._last_error_log_at[side] = now
        print(f"[HandQ6Source:{side}] {message}", flush=True)

    def _reopen_reader_serial(self, side: str, reader: object) -> tuple[bool, str | None]:
        serial_obj = getattr(reader, "ser", None)
        if serial_obj is None:
            return False, "reader has no serial handle"
        try:
            if bool(getattr(serial_obj, "is_open", False)):
                serial_obj.close()
        except Exception as exc:
            return False, f"serial close failed: {exc}"
        if not self._running:
            return False, "source is stopping"
        time.sleep(self._READ_ERROR_RETRY_SEC)
        if not self._running:
            return False, "source is stopping"
        try:
            serial_obj.open()
            reset_input = getattr(serial_obj, "reset_input_buffer", None)
            if callable(reset_input):
                reset_input()
            reset_output = getattr(serial_obj, "reset_output_buffer", None)
            if callable(reset_output):
                reset_output()
        except Exception as exc:
            return False, f"serial reopen failed: {exc}"
        return True, None

    def _reader_loop(self, side: str, reader: object) -> None:
        consecutive_errors = 0
        while self._running:
            try:
                packet = reader.read_frame()
                channel = self._channel_from_packet(
                    side,
                    packet,
                    recovered_after_errors=consecutive_errors,
                )
            except Exception as exc:
                if not self._running:
                    break
                consecutive_errors += 1
                self._reader_error_counts[side] = consecutive_errors
                error_text = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._latest_channels[self._channel_name_for_side(side)] = self._invalid_channel(
                        side,
                        error=error_text,
                        metadata={
                            "error_count": int(consecutive_errors),
                            "reader_running": True,
                            "reader_recovery_count": int(
                                self._reader_recovery_counts.get(side, 0)
                            ),
                        },
                    )
                self._log_reader_error(
                    side,
                    f"read failed ({error_text}); attempting serial reopen",
                )
                reopened, reopen_error = self._reopen_reader_serial(side, reader)
                if reopened:
                    self._reader_recovery_counts[side] = (
                        int(self._reader_recovery_counts.get(side, 0)) + 1
                    )
                    continue
                with self._lock:
                    self._latest_channels[self._channel_name_for_side(side)] = self._invalid_channel(
                        side,
                        error=f"{error_text}; {reopen_error}",
                        metadata={
                            "error_count": int(consecutive_errors),
                            "reader_running": True,
                            "reader_recovery_count": int(
                                self._reader_recovery_counts.get(side, 0)
                            ),
                        },
                    )
                self._log_reader_error(
                    side,
                    f"serial reopen failed ({reopen_error}); retrying",
                )
                time.sleep(self._READ_ERROR_RETRY_SEC)
                continue
            with self._lock:
                self._latest_channels[channel.channel_name] = channel
            if consecutive_errors:
                self._log_reader_error(
                    side,
                    f"recovered after {consecutive_errors} read error(s)",
                )
            consecutive_errors = 0
            self._reader_error_counts[side] = 0

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
