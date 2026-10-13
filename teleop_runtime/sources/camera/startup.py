"""Bound a native RealSense startup without starting another SDK operation on timeout."""

from __future__ import annotations

import math
import threading

from robot_motion.runtime.contracts import ObservationChannel
from robot_motion.runtime.interfaces import ObservationSource

from .realsense_rgb import RealSenseRGBSource


def start_real_sense_source_with_containment(
    source: RealSenseRGBSource,
    *,
    timeout_sec: float,
    software_retry_attempts: int | None = 0,
) -> None:
    """Fail closed if librealsense does not return within the caller's budget.

    The daemon worker may still own the SDK lifecycle lock on timeout. The
    caller must not immediately retry, close, or start a sibling camera then.
    """

    normalized_timeout_sec = float(timeout_sec)
    if not math.isfinite(normalized_timeout_sec) or normalized_timeout_sec < 0.0:
        raise ValueError("timeout_sec must be finite and >= 0.")
    if source._startup_worker is not None and source._startup_worker.is_alive():
        raise RuntimeError("prior RealSense startup worker is still active; no new SDK call was issued")
    errors: list[BaseException] = []
    start_method_is_base = (
        getattr(source.start, "__func__", source.start) is RealSenseRGBSource.start
    )
    pipeline_method_is_base = (
        getattr(source._start_pipeline, "__func__", source._start_pipeline)
        is RealSenseRGBSource._start_pipeline
    )

    def start_native() -> None:
        try:
            if (
                start_method_is_base
                and pipeline_method_is_base
                and software_retry_attempts is not None
            ):
                source.start(
                    startup_software_retry_attempts=software_retry_attempts
                )
            else:
                source.start()
        except BaseException as exc:  # noqa: BLE001 - preserve the native failure type.
            errors.append(exc)

    worker = threading.Thread(
        target=start_native,
        name=f"{source.name}_startup",
        daemon=True,
    )
    source._startup_worker = worker
    worker.start()
    worker.join(timeout=normalized_timeout_sec)
    if worker.is_alive():
        source.mark_startup_timeout(normalized_timeout_sec)
        raise TimeoutError(
            f"{source.name} startup exceeded {normalized_timeout_sec:.3f}s; "
            "native SDK worker remains blocked; no immediate retry/close was "
            "attempted"
        )
    if errors:
        pending_stop = source.device_info().get("pipeline_stop")
        if pending_stop and pending_stop["worker_alive"]:
            raise TimeoutError(
                f"{source.name} startup failed and native cleanup remains active; "
                f"do not start sibling cameras or retry this session: {errors[0]}"
            ) from errors[0]
        raise errors[0]


class BoundedRealSenseSessionSource(ObservationSource):
    """Give legacy session-owned cameras the same native-start containment.

    A timed-out worker must not be closed by the generic session rollback while
    it may still hold librealsense's lifecycle lock. The source is faulted by
    the containment helper and the process must be reconstructed before retry.
    """

    def __init__(self, source: RealSenseRGBSource, *, timeout_sec: float = 30.0):
        self.source = source
        self.timeout_sec = float(timeout_sec)
        self._native_start_timed_out = False

    @property
    def name(self) -> str:
        return self.source.name

    def start(self) -> None:
        if self._native_start_timed_out:
            raise RuntimeError(
                "RealSense startup timed out; reconstruct the source before retrying"
            )
        try:
            start_real_sense_source_with_containment(
                self.source,
                timeout_sec=self.timeout_sec,
                software_retry_attempts=None,
            )
        except TimeoutError:
            self._native_start_timed_out = True
            raise

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        return self.source.get_latest_channels()

    def expected_channel_names(self) -> tuple[str, ...]:
        return self.source.expected_channel_names()

    def close(self) -> None:
        if self._native_start_timed_out:
            worker = self.source._startup_worker
            if worker is None or worker.is_alive():
                raise RuntimeError(
                    "RealSense startup timed out; native cleanup is not proved complete"
                )
        self.source.close()
