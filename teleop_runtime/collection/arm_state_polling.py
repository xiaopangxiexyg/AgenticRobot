from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


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


@dataclass(frozen=True)
class ArmHardwareStatePollerConfig:
    poll_hz: float = 30.0
    stale_after_sec: float = 0.2
    hardware_read_retries: int = 0
    hardware_read_retry_delay_sec: float = 0.0
    source_label: str = "rm65_hardware_state_poller"
    wrist_camera_pose_source: str = "rm65_feedback_qpos_fk"

    def __post_init__(self) -> None:
        if float(self.poll_hz) <= 0.0:
            raise ValueError("poll_hz must be positive.")
        if float(self.stale_after_sec) <= 0.0:
            raise ValueError("stale_after_sec must be positive.")
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
            if backend is not None and hasattr(backend, "refresh_state_from_hardware")
        }
        self._lock = threading.RLock()
        self._latest_by_side: dict[str, dict[str, Any]] = {}
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def enabled(self) -> bool:
        return bool(self.backends_by_side)

    def start(self) -> None:
        if not self.enabled or self._threads:
            return
        self._stop_event.clear()
        for side, backend in self.backends_by_side.items():
            thread = threading.Thread(
                target=self._poll_loop,
                args=(side, backend),
                daemon=True,
                name=f"arm_actual_state_{side}",
            )
            self._threads.append(thread)
            thread.start()

    def close(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()

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

        hardware_qpos = _vector_or_none(state.get("hardware_qpos"), length=6)
        current_qpos = _vector_or_none(state.get("current_qpos"), length=6)
        qpos = hardware_qpos if hardware_qpos is not None else current_qpos
        tcp_pose = _pose_or_none(state.get("current_tcp_pose"))
        if tcp_pose is None and qpos is not None:
            compute_tcp_pose = getattr(backend, "compute_tcp_pose", None)
            if callable(compute_tcp_pose):
                try:
                    tcp_pose = _pose_or_none(compute_tcp_pose(qpos=qpos))
                except Exception:
                    tcp_pose = None

        hardware_refresh_time = _float_or_none(state.get("hardware_refresh_time"))
        pose_timestamp = hardware_refresh_time if hardware_refresh_time is not None else read_end_time
        hardware_pose_6d = _vector_or_none(state.get("hardware_pose_6d"), length=6)
        result_text = state.get("hardware_result_text")
        if error_text is not None:
            result_text = f"poll_error:{error_text}"

        snapshot: dict[str, Any] = {
            "actual_state_source": self.config.source_label,
            "actual_state_read_start_time": float(read_start_time),
            "actual_state_read_end_time": float(read_end_time),
            "actual_state_read_elapsed_s": float(read_elapsed_s),
            "actual_state_timestamp": float(pose_timestamp),
            "actual_hardware_refresh_time": hardware_refresh_time,
            "actual_hardware_result_code": state.get("hardware_result_code"),
            "actual_hardware_result_text": result_text,
            "actual_hardware_arm_err": state.get("hardware_arm_err"),
            "actual_hardware_sys_err": state.get("hardware_sys_err"),
            "actual_hardware_qpos": hardware_qpos,
            "actual_qpos": qpos,
            "actual_tcp_pose": tcp_pose,
            "actual_hardware_pose_6d": hardware_pose_6d,
            "wrist_camera_pose_base": tcp_pose,
            "wrist_camera_pose_timestamp": float(pose_timestamp),
            "wrist_camera_pose_source": self.config.wrist_camera_pose_source,
            "wrist_camera_pose_read_elapsed_s": float(read_elapsed_s),
        }
        if error_text is not None:
            snapshot["actual_state_error"] = error_text
        with self._lock:
            self._latest_by_side[str(side)] = snapshot

    def _refresh_state_from_hardware(self, backend: Any) -> dict[str, Any]:
        refresh = getattr(backend, "refresh_state_from_hardware")
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

    def latest_by_side(self, *, sample_timestamp: float | None = None) -> dict[str, dict[str, Any]]:
        now = time.time()
        with self._lock:
            copied = {
                side: _copy_value(snapshot)
                for side, snapshot in self._latest_by_side.items()
            }
        for snapshot in copied.values():
            pose_timestamp = _float_or_none(snapshot.get("wrist_camera_pose_timestamp"))
            if pose_timestamp is None:
                continue
            age_sec = max(0.0, now - pose_timestamp)
            snapshot["actual_sample_age_sec"] = float(age_sec)
            snapshot["actual_state_stale"] = bool(age_sec > float(self.config.stale_after_sec))
            snapshot["wrist_camera_pose_age_sec"] = float(age_sec)
            if sample_timestamp is not None:
                snapshot["wrist_camera_pose_sample_time_offset_sec"] = (
                    float(sample_timestamp) - float(pose_timestamp)
                )
        return copied
