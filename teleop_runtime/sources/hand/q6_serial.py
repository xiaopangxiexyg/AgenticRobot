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
    exo_hand_calibration: bool = False
    calibration_path: str | None = None


class HandQ6Source(ObservationSource):
    _READ_ERROR_RETRY_SEC = 0.25
    _READ_ERROR_LOG_INTERVAL_SEC = 2.0

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
        self._reader_error_counts: dict[str, int] = {}
        self._reader_recovery_counts: dict[str, int] = {}
        self._last_error_log_at: dict[str, float] = {}
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
        exo_hand_calibration: bool = False,
        calibration_path: str | None = None,
        critical: bool = False,
        strict_expected: bool = False,
    ) -> "HandQ6Source":
        spec = _HandReaderSpec(
            side=str(side),
            port=str(port),
            baudrate=int(baudrate),
            timeout=float(timeout),
            show_raw_lines=bool(show_raw_lines),
            exo_hand_calibration=bool(exo_hand_calibration),
            calibration_path=calibration_path,
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
        exo_hand_calibration: bool = False,
        calibration_path: str | None = None,
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
                exo_hand_calibration=bool(exo_hand_calibration),
                calibration_path=calibration_path,
            )
        if right_port is not None:
            reader_specs["right"] = _HandReaderSpec(
                side="right",
                port=str(right_port),
                baudrate=int(baudrate),
                timeout=float(timeout),
                show_raw_lines=bool(show_raw_lines),
                exo_hand_calibration=bool(exo_hand_calibration),
                calibration_path=calibration_path,
            )
        return cls(
            reader_specs=reader_specs,
            name="hand_q6_bimanual",
            critical=critical,
            strict_expected=strict_expected,
        )

    @classmethod
    def from_default_config(
        cls,
        *,
        active_sides: tuple[str, ...] = ("left", "right"),
        config: str = "latest",
        baudrate: int = 115200,
        timeout: float = 0.05,
        show_raw_lines: bool = False,
        exo_hand_calibration: bool = False,
        calibration_path: str | None = None,
        critical: bool = False,
        strict_expected: bool = False,
    ) -> "HandQ6Source | None":
        from ...configs.q6_hardware import load_latest_q6_ports

        ports = load_latest_q6_ports(active_sides=active_sides, config=config, allow_missing=True)
        if not ports:
            return None
        return cls.from_bimanual(
            left_port=ports.get("left"),
            right_port=ports.get("right"),
            baudrate=baudrate,
            timeout=timeout,
            show_raw_lines=show_raw_lines,
            exo_hand_calibration=exo_hand_calibration,
            calibration_path=calibration_path,
            critical=critical,
            strict_expected=strict_expected,
        )

    def expected_channel_names(self) -> tuple[str, ...]:
        return self._expected if self._strict_expected else ()

    def start(self) -> None:
        if self._running:
            return

        readers: dict[str, object] = {}
        startup_errors: dict[str, Exception] = {}
        for side in self._reader_specs:
            try:
                readers[side] = self._open_reader(side)
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
                self._reader_error_counts[side] = 1
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
        for side in startup_errors:
            thread = threading.Thread(
                target=self._reader_startup_retry_loop,
                args=(side,),
                name=f"{self.name}_{side}_startup_retry",
                daemon=True,
            )
            thread.start()
            self._threads[side] = thread

    def _open_reader(self, side: str) -> object:
        from hardware.teleop_hand.hand_q6_reader import SingleHandQ6Reader

        spec = self._reader_specs[side]
        return SingleHandQ6Reader(
            port=spec.port,
            side=side,
            baudrate=spec.baudrate,
            timeout=spec.timeout,
            raw_only=False,
            show_raw_lines=spec.show_raw_lines,
            exo_hand_calibration=spec.exo_hand_calibration,
            calibration_path=spec.calibration_path,
        )

    def _reader_startup_retry_loop(self, side: str) -> None:
        """Recover an optional channel whose reader constructor initially failed."""

        consecutive_errors = int(self._reader_error_counts.get(side, 1))
        while self._running:
            time.sleep(self._READ_ERROR_RETRY_SEC)
            if not self._running:
                return
            try:
                reader = self._open_reader(side)
            except Exception as exc:  # noqa: BLE001 - optional source stays retryable.
                consecutive_errors += 1
                self._reader_error_counts[side] = consecutive_errors
                error_text = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._latest_channels[
                        self._channel_name_for_side(side)
                    ] = self._invalid_channel(
                        side,
                        error=f"startup_retry_failed: {error_text}",
                        metadata={
                            "error_count": int(consecutive_errors),
                            "reader_running": False,
                            "startup_retry_active": True,
                        },
                    )
                self._log_reader_error(
                    side,
                    f"reader construction retry failed ({error_text}); retrying",
                )
                continue

            close_after_open = False
            with self._lock:
                if not self._running:
                    close_after_open = True
                else:
                    self._readers[side] = reader
                    self._reader_recovery_counts[side] = (
                        int(self._reader_recovery_counts.get(side, 0)) + 1
                    )
            if close_after_open:
                try:
                    reader.close()
                except Exception:
                    pass
                return
            self._log_reader_error(
                side,
                f"reader construction recovered after {consecutive_errors} error(s)",
            )
            self._reader_loop(side, reader)
            return

    def _channel_name_for_side(self, side: str) -> str:
        return f"{side}_hand_q6"

    def _channel_from_packet(
        self,
        side: str,
        packet: dict[str, object],
        *,
        recovered_after_errors: int = 0,
    ) -> ObservationChannel:
        channel_name = self._channel_name_for_side(side)
        timestamp = float(np.asarray(packet.get("timestamp", [0.0]), dtype=float).ravel()[0])
        metadata = {
            "port": packet.get("port"),
            "baudrate": packet.get("baudrate"),
            "reader_error_count": int(self._reader_error_counts.get(side, 0)),
            "reader_recovery_count": int(self._reader_recovery_counts.get(side, 0)),
        }
        serial_read_stats = packet.get("serial_read_stats")
        if isinstance(serial_read_stats, dict):
            metadata["serial_read_stats"] = dict(serial_read_stats)
            metadata["stale_q6_frames_dropped"] = int(
                serial_read_stats.get("stale_frames_dropped", 0) or 0
            )
        if recovered_after_errors > 0:
            metadata["recovered_after_errors"] = int(recovered_after_errors)
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
        last_value = np.zeros(6, dtype=float)
        previous = self._latest_channels.get(channel_name)
        if previous is not None:
            last_value = np.asarray(previous.value, dtype=float).reshape(6).copy()
        spec = self._reader_specs[side]
        channel_metadata = {
            "port": spec.port,
            "baudrate": spec.baudrate,
            "read_error": str(error),
        }
        if metadata is not None:
            channel_metadata.update(metadata)
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
        with self._lock:
            readers = tuple(self._readers.values())
        for reader in readers:
            try:
                reader.close()
            except Exception:
                pass
        for thread in self._threads.values():
            thread.join(timeout=1.0)
        self._threads = {}
        with self._lock:
            self._readers = {}
