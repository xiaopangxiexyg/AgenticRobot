from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from robot_motion.runtime.lifecycle import (
    LifecycleStartupError,
    collect_lifecycle_failures,
    raise_lifecycle_failures,
)


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    return value


def _vector_or_none(value: Any, *, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if array.shape != (int(length),) or not np.all(np.isfinite(array)):
        return None
    return array.copy()


def _pose_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=float).reshape(4, 4)
    except Exception:
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose.copy()


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out == value else None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


@dataclass(frozen=True)
class ArmHardwareStatePollerConfig:
    poll_hz: float = 30.0
    stale_after_sec: float = 0.2
    hardware_read_retries: int = 0
    hardware_read_retry_delay_sec: float = 0.0
    source_label: str = "rm65_hardware_state_poller"
    wrist_camera_pose_source: str = "rm65_feedback_qpos_fk"
    history_size: int = 256

    def __post_init__(self) -> None:
        if float(self.poll_hz) <= 0.0:
            raise ValueError("poll_hz must be positive.")
        if float(self.stale_after_sec) <= 0.0:
            raise ValueError("stale_after_sec must be positive.")
        if int(self.history_size) <= 0:
            raise ValueError("history_size must be positive.")
        if int(self.hardware_read_retries) < 0:
            raise ValueError("hardware_read_retries must be non-negative.")
        if float(self.hardware_read_retry_delay_sec) < 0.0:
            raise ValueError("hardware_read_retry_delay_sec must be non-negative.")


class ArmHardwareStatePoller:
    """Background RM65 state reader used to annotate samples without blocking control."""

    def __init__(
        self,
        *,
        backends_by_side: Mapping[str, Any],
        config: ArmHardwareStatePollerConfig | None = None,
    ):
        self.config = config if config is not None else ArmHardwareStatePollerConfig()
        self.backends_by_side = {
            str(side): backend
            for side, backend in dict(backends_by_side).items()
            if backend is not None
            and (
                hasattr(backend, "refresh_state_from_hardware")
                or callable(getattr(backend, "get_state", None))
            )
        }
        self._lock = threading.RLock()
        self._history_by_side: dict[str, deque[dict[str, Any]]] = {
            side: deque(maxlen=int(self.config.history_size))
            for side in self.backends_by_side
        }
        self._sequence_by_side: dict[str, int] = {
            side: 0 for side in self.backends_by_side
        }
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def enabled(self) -> bool:
        return bool(self.backends_by_side)

    def start(self) -> None:
        if not self.enabled or self._threads:
            return
        self._stop_event.clear()
        try:
            for side, backend in self.backends_by_side.items():
                thread = threading.Thread(
                    target=self._poll_loop,
                    args=(side, backend),
                    daemon=True,
                    name=f"arm_actual_state_{side}",
                )
                thread.start()
                self._threads.append(thread)
        except BaseException as start_error:  # noqa: BLE001 - stop already-started pollers.
            self._stop_event.set()
            rollback_failures = collect_lifecycle_failures(
                (
                    (
                        f"worker[{thread.name}]",
                        lambda thread=thread: self._join_thread_or_raise(thread),
                    )
                    for thread in reversed(self._threads)
                )
            )
            self._threads = [thread for thread in self._threads if thread.is_alive()]
            if rollback_failures:
                raise LifecycleStartupError(
                    "arm_state_poller.start",
                    start_error,
                    rollback_failures,
                ) from start_error
            raise

    def close(self) -> None:
        self._stop_event.set()
        failures = collect_lifecycle_failures(
            (
                (
                    f"worker[{thread.name}]",
                    lambda thread=thread: self._join_thread_or_raise(thread),
                )
                for thread in reversed(self._threads)
            )
        )
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        raise_lifecycle_failures("arm_state_poller.close", failures)

    @staticmethod
    def _join_thread_or_raise(thread: threading.Thread) -> None:
        thread.join(timeout=1.0)
        if thread.is_alive():
            raise RuntimeError(
                f"arm state poller worker {thread.name!r} did not stop within 1.000s"
            )

    def _poll_loop(self, side: str, backend: Any) -> None:
        period_sec = 1.0 / float(self.config.poll_hz)
        next_deadline = time.perf_counter()
        while not self._stop_event.is_set():
            self._poll_once(side, backend)
            next_deadline += period_sec
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec < -period_sec:
                next_deadline = time.perf_counter()
                sleep_sec = period_sec
            if sleep_sec > 0.0:
                self._stop_event.wait(sleep_sec)

    def _poll_once(self, side: str, backend: Any) -> None:
        read_start_time = time.time()
        read_start_perf = time.perf_counter()
        error_text: str | None = None
        try:
            state = self._refresh_state_from_hardware(backend)
        except Exception as exc:  # noqa: BLE001 - this is a non-critical annotation path.
            error_text = f"{type(exc).__name__}: {exc}"
            get_state = getattr(backend, "get_state", None)
            try:
                state = dict(get_state()) if callable(get_state) else {}
            except Exception:
                state = {}
        read_end_time = time.time()
        read_elapsed_s = time.perf_counter() - read_start_perf

        cached_hardware_qpos = _vector_or_none(state.get("hardware_qpos"), length=6)
        hardware_refresh_time = _float_or_none(
            state.get("hardware_measurement_timestamp", state.get("hardware_refresh_time"))
        )
        measurement_sequence = _int_or_none(
            state.get("hardware_measurement_sequence", state.get("measured_sequence"))
        )
        read_success = _bool_or_none(state.get("hardware_read_success"))
        measurement_valid = _bool_or_none(state.get("hardware_measurement_valid"))
        result_code = _int_or_none(state.get("hardware_result_code"))
        if read_success is None:
            # Compatibility for older/fake backends that predate explicit read
            # status. RM65 code 9 is usable only when it carries finite qpos.
            read_success = bool(
                error_text is None
                and result_code in (0, 9)
                and cached_hardware_qpos is not None
                and hardware_refresh_time is not None
            )
        if measurement_valid is None:
            measurement_valid = bool(
                cached_hardware_qpos is not None and hardware_refresh_time is not None
            )
        sequence_valid = measurement_sequence is None or measurement_sequence > 0
        actual_valid = bool(
            error_text is None
            and read_success
            and measurement_valid
            and sequence_valid
            and cached_hardware_qpos is not None
            and hardware_refresh_time is not None
        )
        hardware_qpos = cached_hardware_qpos if actual_valid else None
        qpos = hardware_qpos
        tcp_pose = _pose_or_none(state.get("current_tcp_pose")) if actual_valid else None
        if tcp_pose is None and qpos is not None:
            compute_tcp_pose = getattr(backend, "compute_tcp_pose", None)
            if callable(compute_tcp_pose):
                try:
                    tcp_pose = _pose_or_none(compute_tcp_pose(qpos=qpos))
                except Exception:
                    tcp_pose = None

        pose_timestamp = hardware_refresh_time if actual_valid else None
        hardware_pose_6d = (
            _vector_or_none(state.get("hardware_pose_6d"), length=6)
            if actual_valid
            else None
        )
        result_text = state.get("hardware_result_text")
        if error_text is not None:
            result_text = f"poll_error:{error_text}"

        snapshot: dict[str, Any] = {
            "actual_state_source": self.config.source_label,
            "actual_state_read_start_time": float(read_start_time),
            "actual_state_read_end_time": float(read_end_time),
            "actual_state_read_elapsed_s": float(read_elapsed_s),
            "actual_state_event_timestamp": float(read_end_time),
            "actual_state_timestamp": pose_timestamp,
            # Standard causal-history names.  The source value is the device
            # measurement time; availability is recorded after this payload is
            # fully assembled and frozen below.
            "source_timestamp": pose_timestamp,
            "source_clock": "local_wall_time",
            "actual_state_read_success": bool(read_success),
            "actual_state_valid": actual_valid,
            "actual_state_sequence": measurement_sequence,
            "actual_hardware_refresh_time": hardware_refresh_time,
            "actual_hardware_result_code": result_code,
            "actual_hardware_result_text": result_text,
            "actual_hardware_arm_err": state.get("hardware_arm_err"),
            "actual_hardware_sys_err": state.get("hardware_sys_err"),
            "actual_hardware_qpos": hardware_qpos,
            "actual_qpos": qpos,
            "actual_tcp_pose": tcp_pose,
            "actual_hardware_pose_6d": hardware_pose_6d,
            "wrist_camera_pose_base": tcp_pose,
            "wrist_camera_pose_timestamp": pose_timestamp,
            "wrist_camera_pose_source": (
                self.config.wrist_camera_pose_source if actual_valid else None
            ),
            "wrist_camera_pose_read_elapsed_s": float(read_elapsed_s),
        }
        if error_text is not None:
            snapshot["actual_state_error"] = error_text
        with self._lock:
            history = self._history_by_side.setdefault(
                str(side),
                deque(maxlen=int(self.config.history_size)),
            )
            self._sequence_by_side[str(side)] = self._sequence_by_side.get(str(side), 0) + 1
            freeze_started_perf = time.perf_counter()
            frozen_snapshot = _copy_value(snapshot)
            # Convert the short local freeze duration back into the wall-clock
            # domain without another wall-clock syscall.  This marks the point
            # after the immutable copy completed while keeping injected-clock
            # tests deterministic and the hot poll loop syscall-light.
            freeze_elapsed = max(0.0, time.perf_counter() - freeze_started_perf)
            frozen_snapshot["available_timestamp"] = float(read_end_time + freeze_elapsed)
            frozen_snapshot["available_clock"] = "local_wall_time"
            frozen_snapshot["snapshot_sequence"] = self._sequence_by_side[str(side)]
            history.append(frozen_snapshot)

    def _refresh_state_from_hardware(self, backend: Any) -> dict[str, Any]:
        refresh = getattr(backend, "refresh_state_from_hardware", None)
        if not callable(refresh):
            get_state = getattr(backend, "get_state", None)
            if not callable(get_state):
                return {}
            return dict(get_state())
        code = getattr(refresh, "__code__", None)
        arg_names = set(getattr(code, "co_varnames", ()) or ())
        kwargs: dict[str, Any] = {}
        if "hardware_read_retries" in arg_names:
            kwargs["hardware_read_retries"] = int(self.config.hardware_read_retries)
        if "hardware_read_retry_delay_sec" in arg_names:
            kwargs["hardware_read_retry_delay_sec"] = float(
                self.config.hardware_read_retry_delay_sec
            )
        return dict(refresh(**kwargs))

    @staticmethod
    def _snapshot_timestamp(snapshot: Mapping[str, Any]) -> float | None:
        if snapshot.get("actual_state_valid") is not False:
            timestamp = _float_or_none(snapshot.get("actual_state_timestamp"))
            if timestamp is not None:
                return timestamp
            timestamp = _float_or_none(snapshot.get("wrist_camera_pose_timestamp"))
            if timestamp is not None:
                return timestamp
        return _float_or_none(snapshot.get("actual_state_event_timestamp"))

    def latest_by_side(
        self,
        *,
        sample_timestamp: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return one causal snapshot per side.

        When ``sample_timestamp`` is supplied, a snapshot is eligible only when
        its state timestamp is less than or equal to the sample cut.  A future
        hardware read is never substituted for a missing past state.
        """

        reference_time = time.time() if sample_timestamp is None else float(sample_timestamp)
        if not np.isfinite(reference_time):
            raise ValueError("sample_timestamp must be finite when provided.")
        # A few offline/unit callers use an injected synthetic timestamp domain
        # (e.g. ``99.0``) while the poller availability clock is wall time.  In
        # that impossible-to-compare domain, preserve the historical source-
        # timestamp query and mark the result below; real collection/deploy T0
        # values are current wall time and always take the strict path.
        availability_comparable = abs(reference_time - time.time()) < 86400.0
        with self._lock:
            copied: dict[str, dict[str, Any]] = {}
            for side, history in self._history_by_side.items():
                eligible: list[tuple[float, float, int, Mapping[str, Any]]] = []
                for insertion_index, snapshot in enumerate(history):
                    timestamp = self._snapshot_timestamp(snapshot)
                    if timestamp is None:
                        continue
                    source_timestamp = _float_or_none(snapshot.get("source_timestamp"))
                    available_timestamp = _float_or_none(
                        snapshot.get("available_timestamp", snapshot.get("actual_state_read_end_time"))
                    )
                    if (
                        (
                            source_timestamp is None
                            and snapshot.get("actual_state_valid") is not False
                        )
                        or (
                            source_timestamp is not None
                            and source_timestamp > reference_time
                        )
                        or (
                            availability_comparable
                            and (
                                available_timestamp is None
                                or available_timestamp > reference_time
                            )
                        )
                    ):
                        continue
                    eligible.append(
                        (
                            timestamp,
                            -float("inf") if available_timestamp is None else available_timestamp,
                            insertion_index,
                            snapshot,
                        )
                    )
                if not eligible:
                    continue
                _timestamp, _available_timestamp, _insertion_index, selected = max(
                    eligible,
                    key=lambda item: (item[0], item[1], item[2]),
                )
                copied[side] = _copy_value(selected)
                if not availability_comparable:
                    copied[side]["causal_availability_unverified"] = True
        for snapshot in copied.values():
            measurement_timestamp = _float_or_none(snapshot.get("actual_state_timestamp"))
            if snapshot.get("actual_state_valid") is False or measurement_timestamp is None:
                snapshot["actual_sample_age_sec"] = None
                snapshot["actual_state_stale"] = True
                snapshot["wrist_camera_pose_age_sec"] = None
                continue
            age_sec = max(0.0, reference_time - measurement_timestamp)
            snapshot["actual_sample_age_sec"] = float(age_sec)
            available_timestamp = _float_or_none(snapshot.get("available_timestamp"))
            snapshot["actual_snapshot_available_age_sec"] = (
                None
                if available_timestamp is None
                else max(0.0, reference_time - available_timestamp)
            )
            snapshot["causal_snapshot_contract"] = (
                "source_timestamp<=T0_and_available_timestamp<=T0"
            )
            snapshot["actual_state_stale"] = bool(age_sec > float(self.config.stale_after_sec))
            snapshot["wrist_camera_pose_age_sec"] = float(age_sec)
            if sample_timestamp is not None:
                snapshot["wrist_camera_pose_sample_time_offset_sec"] = (
                    float(sample_timestamp) - float(measurement_timestamp)
                )
        return copied

    def latest_at_or_before(self, sample_timestamp: float) -> dict[str, dict[str, Any]]:
        """Explicit causal lookup alias used by collection sample cuts."""

        return self.latest_by_side(sample_timestamp=float(sample_timestamp))

    def readiness_by_side(
        self,
        *,
        required_sides: tuple[str, ...] | list[str] | None = None,
        sample_timestamp: float | None = None,
    ) -> dict[str, str | None]:
        """Report whether each required side has a fresh causal TCP feedback sample.

        This is intentionally a read-only gate for beginning an episode.  It
        never triggers a hardware read or changes the selected snapshot; the
        background poller remains the sole feedback producer.
        """

        reference_time = time.time() if sample_timestamp is None else float(sample_timestamp)
        if not np.isfinite(reference_time):
            raise ValueError("sample_timestamp must be finite when provided.")
        sides = (
            tuple(self.backends_by_side)
            if required_sides is None
            else tuple(str(side) for side in required_sides)
        )
        snapshots = self.latest_by_side(sample_timestamp=reference_time)
        readiness: dict[str, str | None] = {}
        for side in sides:
            snapshot = snapshots.get(side)
            if snapshot is None:
                readiness[side] = "no_causal_feedback"
            elif snapshot.get("actual_state_read_success") is not True:
                readiness[side] = "hardware_read_failed"
            elif snapshot.get("actual_state_valid") is not True:
                readiness[side] = "measurement_invalid"
            elif snapshot.get("actual_state_stale") is not False:
                readiness[side] = "feedback_stale"
            elif _pose_or_none(snapshot.get("actual_tcp_pose")) is None:
                readiness[side] = "tcp_pose_unavailable"
            else:
                readiness[side] = None
        return readiness

    def wait_until_ready(
        self,
        *,
        required_sides: tuple[str, ...] | list[str] | None = None,
        timeout_sec: float,
        poll_interval_sec: float = 0.005,
    ) -> dict[str, str | None]:
        """Wait passively for valid causal feedback, then return per-side reasons.

        No polling is initiated here.  The bounded wait only gives already
        running background pollers time to publish their first valid snapshot.
        """

        if float(timeout_sec) < 0.0:
            raise ValueError("timeout_sec must be non-negative.")
        if float(poll_interval_sec) <= 0.0:
            raise ValueError("poll_interval_sec must be positive.")
        deadline = time.monotonic() + float(timeout_sec)
        while True:
            readiness = self.readiness_by_side(required_sides=required_sides)
            if all(reason is None for reason in readiness.values()):
                return readiness
            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0.0:
                return readiness
            self._stop_event.wait(min(float(poll_interval_sec), remaining_sec))
