from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from robot_motion.runtime.causal_history import CausalSnapshotHistory
from robot_motion.runtime.lifecycle import (
    LifecycleStartupError,
    collect_lifecycle_failures,
    raise_lifecycle_failures,
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _state_arg_names(get_state: Any) -> set[str]:
    code = getattr(get_state, "__code__", None)
    if code is None:
        return set()
    count = int(getattr(code, "co_argcount", 0) or 0) + int(
        getattr(code, "co_kwonlyargcount", 0) or 0
    )
    return {str(value) for value in code.co_varnames[:count]}


def _copy_state_without_tactile_arrays(state: Mapping[str, Any]) -> dict[str, Any]:
    # The backend's filtered get_state() is responsible for avoiding the
    # expensive array copy.  Remove any legacy backend arrays defensively so a
    # custom backend cannot make the light telemetry history retain mutable
    # buffers.
    result = dict(state)
    result["tactile"] = None
    result["tactile_dict"] = None
    return result


def _tactile_key(state: Mapping[str, Any]) -> tuple[str, Any] | None:
    version = state.get("tactile_version")
    if version is not None:
        return "version", version
    frame_ids = state.get("tactile_frame_ids")
    if isinstance(frame_ids, Mapping) and frame_ids:
        return "frame_ids", tuple(sorted((str(k), v) for k, v in frame_ids.items()))
    timestamp = _finite(state.get("tactile_timestamp"))
    if timestamp is not None:
        return "timestamp", timestamp
    if state.get("tactile_dict") or state.get("tactile") is not None:
        return "presence", True
    return None


def _has_tactile(state: Mapping[str, Any]) -> bool:
    if state.get("tactile_timestamp") is not None:
        return bool(
            state.get("tactile_dict")
            or state.get("tactile") is not None
            or state.get("tactile_version") is not None
            or state.get("tactile_frame_ids")
            or state.get("tactile_devices")
        )
    return bool(state.get("tactile_dict") or state.get("tactile") is not None)


def hand_state_tactile_arrays(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Normalize the tactile arrays accepted by the shared external writer."""

    arrays: dict[str, np.ndarray] = {}
    tactile_dict = state.get("tactile_dict")
    if isinstance(tactile_dict, Mapping):
        for key, value in tactile_dict.items():
            if value is None:
                continue
            array = np.asarray(value)
            if array.size > 0:
                arrays[str(key)] = array
    if arrays:
        return arrays
    tactile = state.get("tactile")
    if tactile is None:
        return {}
    array = np.asarray(tactile, dtype=np.float32).reshape(-1)
    return {} if array.size <= 0 else {"vector": array}


@dataclass(frozen=True)
class HandStateHistoryPollerConfig:
    poll_hz: float = 60.0
    history_size: int = 256
    tactile_history_size: int = 64
    source_label: str = "hand_state_history_poller"

    def __post_init__(self) -> None:
        if float(self.poll_hz) <= 0.0:
            raise ValueError("poll_hz must be positive")
        if int(self.history_size) <= 0 or int(self.tactile_history_size) <= 0:
            raise ValueError("history sizes must be positive")


class HandStateHistoryPoller:
    """Capture cached hand telemetry/tactile state into causal histories.

    This poller never performs a hardware read.  Hand backends already own the
    hardware telemetry workers; this layer only freezes their cached state and
    provides a time-indexed view to collection and deployment consumers.
    Telemetry and tactile streams have separate histories so an unchanged
    tactile frame is not copied at every control step.
    """

    def __init__(
        self,
        *,
        backends_by_side: Mapping[str, Any],
        config: HandStateHistoryPollerConfig | None = None,
    ) -> None:
        self.config = config if config is not None else HandStateHistoryPollerConfig()
        self.backends_by_side = {
            str(side): backend
            for side, backend in dict(backends_by_side).items()
            if backend is not None and callable(getattr(backend, "get_state", None))
        }
        self._telemetry_by_side = {
            side: CausalSnapshotHistory(
                maxlen=int(self.config.history_size),
                name=f"hand:{side}:telemetry",
            )
            for side in self.backends_by_side
        }
        self._tactile_by_side = {
            side: CausalSnapshotHistory(
                maxlen=int(self.config.tactile_history_size),
                name=f"hand:{side}:tactile",
                copy_arrays_on_read=False,
            )
            for side in self.backends_by_side
        }
        self._last_tactile_key: dict[str, tuple[str, Any] | None] = {
            side: None for side in self.backends_by_side
        }
        self._lock = threading.RLock()
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
                    name=f"hand_state_history_{side}",
                )
                thread.start()
                self._threads.append(thread)
        except BaseException as start_error:  # noqa: BLE001
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
                    "hand_state_history_poller.start", start_error, rollback_failures
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
        raise_lifecycle_failures("hand_state_history_poller.close", failures)

    @staticmethod
    def _join_thread_or_raise(thread: threading.Thread) -> None:
        thread.join(timeout=1.0)
        if thread.is_alive():
            raise RuntimeError(
                f"hand state history worker {thread.name!r} did not stop within 1.000s"
            )

    def _poll_loop(self, side: str, backend: Any) -> None:
        period = 1.0 / float(self.config.poll_hz)
        next_deadline = time.perf_counter()
        while not self._stop_event.is_set():
            self._poll_once(side, backend)
            next_deadline += period
            sleep = next_deadline - time.perf_counter()
            if sleep < -period:
                next_deadline = time.perf_counter()
                sleep = period
            if sleep > 0.0:
                self._stop_event.wait(sleep)

    @staticmethod
    def _get_state(backend: Any, *, include_tactile_arrays: bool) -> dict[str, Any]:
        get_state = getattr(backend, "get_state")
        names = _state_arg_names(get_state)
        kwargs: dict[str, Any] = {}
        if "include_tactile_arrays" in names:
            kwargs["include_tactile_arrays"] = bool(include_tactile_arrays)
        if "include_tactile_vector" in names:
            kwargs["include_tactile_vector"] = bool(include_tactile_arrays)
        if "copy_tactile_arrays" in names:
            kwargs["copy_tactile_arrays"] = bool(include_tactile_arrays)
        return dict(get_state(**kwargs))

    @staticmethod
    def _telemetry_source_timestamp(state: Mapping[str, Any]) -> float | None:
        # Position feedback is only new when the hardware joint registers were
        # read successfully.  A telemetry worker loop may still complete while
        # that read is deliberately skipped, so its loop timestamp must not
        # make cached joint feedback appear fresh.
        joint_timestamp = _finite(state.get("joint_timestamp"))
        if joint_timestamp is not None:
            return joint_timestamp
        return _finite(state.get("telemetry_timestamp"))

    def _poll_once(self, side: str, backend: Any) -> None:
        try:
            light_state = self._get_state(backend, include_tactile_arrays=False)
            telemetry_state = _copy_state_without_tactile_arrays(light_state)
            telemetry_state["snapshot_stream"] = "telemetry"
            telemetry_state["snapshot_source"] = self.config.source_label
            self._telemetry_by_side[side].publish(
                telemetry_state,
                source_timestamp=self._telemetry_source_timestamp(light_state),
                available_timestamp=None,
            )

            key = _tactile_key(light_state)
            with self._lock:
                changed = key is not None and key != self._last_tactile_key.get(side)
                if changed:
                    self._last_tactile_key[side] = key
            if not changed or not _has_tactile(light_state):
                return

            tactile_state = self._get_state(backend, include_tactile_arrays=True)
            tactile_state["snapshot_stream"] = "tactile"
            tactile_state["snapshot_source"] = self.config.source_label
            self._tactile_by_side[side].publish(
                tactile_state,
                source_timestamp=_finite(tactile_state.get("tactile_timestamp")),
                available_timestamp=None,
            )
        except Exception as exc:  # Keep a backend failure diagnostic and continue.
            error_state = {
                "snapshot_stream": "telemetry",
                "snapshot_source": self.config.source_label,
                "telemetry_error": f"history_poll_failed:{type(exc).__name__}:{exc}",
            }
            self._telemetry_by_side[side].publish(
                error_state,
                source_timestamp=None,
                available_timestamp=None,
            )

    def latest_by_side(self, sample_timestamp: float) -> dict[str, dict[str, Any]]:
        cutoff = float(sample_timestamp)
        result: dict[str, dict[str, Any]] = {}
        for side in self.backends_by_side:
            telemetry = self._telemetry_by_side[side].latest_at_or_before(cutoff)
            tactile = self._tactile_by_side[side].latest_at_or_before(cutoff)
            if telemetry is None and tactile is None:
                continue
            state: dict[str, Any] = {}
            telemetry_selection = None
            tactile_selection = None
            if telemetry is not None:
                state, telemetry_selection = telemetry
                state.pop("snapshot_stream", None)
                state.pop("snapshot_source", None)
                telemetry_source_clock = state.pop("source_clock", None)
                telemetry_available_clock = state.pop("available_clock", None)
                state.pop("source_timestamp", None)
                state.pop("available_timestamp", None)
                state.pop("snapshot_sequence", None)
            else:
                telemetry_source_clock = None
                telemetry_available_clock = None
            if tactile is not None:
                tactile_state, tactile_selection = tactile
                tactile_source_clock = tactile_state.get("source_clock")
                tactile_available_clock = tactile_state.get("available_clock")
                for key, value in tactile_state.items():
                    if key.startswith("tactile") or key in {
                        "tactile",
                        "tactile_dict",
                        "tactile_timestamp",
                    }:
                        state[key] = value
            if telemetry_selection is not None:
                telemetry_available_age = telemetry_selection.age_sec
                telemetry_source_age = telemetry_selection.source_age_sec
                telemetry_effective_age = max(
                    value
                    for value in (telemetry_available_age, telemetry_source_age)
                    if value is not None
                )
                state.update(
                    {
                        "telemetry_snapshot_sequence": telemetry_selection.sequence,
                        "telemetry_snapshot_source_timestamp": telemetry_selection.source_timestamp,
                        "telemetry_snapshot_available_timestamp": telemetry_selection.available_timestamp,
                        "telemetry_snapshot_age_sec": telemetry_effective_age,
                        "telemetry_snapshot_source_age_sec": telemetry_source_age,
                        "telemetry_snapshot_available_age_sec": telemetry_available_age,
                        "telemetry_snapshot_source_clock": telemetry_source_clock,
                        "telemetry_snapshot_available_clock": telemetry_available_clock,
                    }
                )
            else:
                state.update(
                    {
                        "telemetry_snapshot_sequence": None,
                        "telemetry_snapshot_source_timestamp": None,
                        "telemetry_snapshot_available_timestamp": None,
                        "telemetry_snapshot_age_sec": None,
                        "telemetry_snapshot_source_age_sec": None,
                        "telemetry_snapshot_available_age_sec": None,
                        "telemetry_snapshot_source_clock": None,
                        "telemetry_snapshot_available_clock": None,
                    }
                )
            if tactile_selection is not None:
                tactile_available_age = tactile_selection.age_sec
                tactile_source_age = tactile_selection.source_age_sec
                tactile_effective_age = max(
                    value
                    for value in (tactile_available_age, tactile_source_age)
                    if value is not None
                )
                state.update(
                    {
                        "tactile_snapshot_sequence": tactile_selection.sequence,
                        "tactile_snapshot_source_timestamp": tactile_selection.source_timestamp,
                        "tactile_snapshot_available_timestamp": tactile_selection.available_timestamp,
                        "tactile_snapshot_age_sec": tactile_effective_age,
                        "tactile_snapshot_source_age_sec": tactile_source_age,
                        "tactile_snapshot_available_age_sec": tactile_available_age,
                        "tactile_snapshot_source_clock": tactile_source_clock,
                        "tactile_snapshot_available_clock": tactile_available_clock,
                    }
                )
            else:
                state.update(
                    {
                        "tactile_snapshot_sequence": None,
                        "tactile_snapshot_source_timestamp": None,
                        "tactile_snapshot_available_timestamp": None,
                        "tactile_snapshot_age_sec": None,
                        "tactile_snapshot_source_age_sec": None,
                        "tactile_snapshot_available_age_sec": None,
                        "tactile_snapshot_source_clock": None,
                        "tactile_snapshot_available_clock": None,
                    }
                )
            state["causal_snapshot_cutoff_timestamp"] = cutoff
            joint_timestamp = _finite(state.get("joint_timestamp"))
            state["joint_age_sec"] = (
                None if joint_timestamp is None else cutoff - joint_timestamp
            )
            state["causal_snapshot_contract"] = (
                "source_timestamp<=T0_and_available_timestamp<=T0"
            )
            result[side] = state
        return result

    def latest_at_or_before(self, sample_timestamp: float) -> dict[str, dict[str, Any]]:
        return self.latest_by_side(float(sample_timestamp))

    def history_sizes(self) -> dict[str, dict[str, int]]:
        return {
            side: {
                "telemetry": len(self._telemetry_by_side[side]),
                "tactile": len(self._tactile_by_side[side]),
            }
            for side in self.backends_by_side
        }
