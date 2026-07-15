from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .contracts import (
    POSITION_CONTROL_PIPELINE_VERSION,
    PositionCommandTick,
    PositionReferenceFrame,
)
from .timeline import OutOfOrderReferenceError


DEFAULT_POSITION_REFERENCE_HZ = 75.0
DEFAULT_POSITION_COMMAND_HZ = 150.0
DEFAULT_POSITION_SEGMENT_DURATION_SEC = 1.0 / DEFAULT_POSITION_REFERENCE_HZ
DEFAULT_REFERENCE_STALE_AFTER_SEC = 0.1


@dataclass(frozen=True)
class _AcceptedJointTarget:
    sequence: int
    published_monotonic_time: float
    source: str
    pipeline_version: str
    hold: bool


def reference_age_and_stale(
    *,
    published_monotonic_time: float,
    sample_monotonic_time: float,
    stale_after_sec: float | None,
) -> tuple[float, bool]:
    """Return target age and whether it has crossed its stale deadline.

    The helper is intentionally sim/real neutral so a transport cannot subtly
    redefine the reference-expiry rule used by the shared reference -> command
    planner.
    """

    published = float(published_monotonic_time)
    sample = float(sample_monotonic_time)
    if not math.isfinite(published) or not math.isfinite(sample):
        raise ValueError("reference timestamps must be finite")
    if stale_after_sec is None:
        stale_after = None
    else:
        stale_after = float(stale_after_sec)
        if not math.isfinite(stale_after) or stale_after < 0.0:
            raise ValueError("stale_after_sec must be finite and >= 0, or None")
    age = max(0.0, sample - published)
    return age, bool(stale_after is not None and age > stale_after)


def interpolate_joint_command_segment(
    start_qpos: np.ndarray | Sequence[float],
    target_qpos: np.ndarray | Sequence[float],
    *,
    segment_start_time: float | None,
    segment_duration_sec: float,
    sample_time: float,
) -> tuple[np.ndarray, float]:
    """Sample one timestamp-driven joint segment.

    This is the sim/real-neutral reference -> high-rate qcmd primitive.
    ``JointCommandInterpolator`` owns the shared timestamp, stale and target
    preemption semantics around this primitive. Environment-specific code owns
    only wakeup/transport and final sink acknowledgement.  A
    ``segment_start_time`` of ``None`` means the target is already active and
    therefore returns it with ``alpha=1``.
    """

    start = np.asarray(start_qpos, dtype=float).reshape(-1)
    target = np.asarray(target_qpos, dtype=float).reshape(-1)
    if start.size == 0 or target.size == 0 or start.shape != target.shape:
        raise ValueError("start_qpos and target_qpos must be matching non-empty vectors")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
        raise ValueError("start_qpos and target_qpos must contain only finite values")

    duration = float(segment_duration_sec)
    sample = float(sample_time)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("segment_duration_sec must be finite and >= 0")
    if not math.isfinite(sample):
        raise ValueError("sample_time must be finite")
    if segment_start_time is None:
        alpha = 1.0
    else:
        start_time = float(segment_start_time)
        if not math.isfinite(start_time):
            raise ValueError("segment_start_time must be finite or None")
        alpha = (
            1.0
            if duration <= 0.0
            else float(np.clip((sample - start_time) / duration, 0.0, 1.0))
        )
    return start + (target - start) * alpha, alpha


class JointCommandInterpolator:
    """Sim/real-neutral monotonic qpos planning for one atomic arm group.

    A newly accepted reference starts at the last command emitted by the
    planner. A transport with acknowledgements can explicitly substitute its
    last accepted command. Target preemption therefore preserves continuity
    without ever depending on an unobserved theoretical trajectory point.
    """

    def __init__(
        self,
        *,
        segment_duration_sec: float = DEFAULT_POSITION_SEGMENT_DURATION_SEC,
        stale_after_sec: float | None = DEFAULT_REFERENCE_STALE_AFTER_SEC,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ):
        duration = float(segment_duration_sec)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("segment_duration_sec must be finite and >= 0")
        if stale_after_sec is None:
            normalized_stale_after = None
        else:
            normalized_stale_after = float(stale_after_sec)
            if not math.isfinite(normalized_stale_after) or normalized_stale_after < 0.0:
                raise ValueError("stale_after_sec must be finite and >= 0, or None")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")

        self.segment_duration_sec = duration
        self.stale_after_sec = normalized_stale_after
        self._clock = monotonic_clock
        self._lock = threading.RLock()
        self._epoch: int | None = None
        self._last_emitted_qpos: dict[str, np.ndarray] | None = None
        self._last_emitted_monotonic: float | None = None
        self._active_target: _AcceptedJointTarget | None = None
        self._segment_start_qpos: dict[str, np.ndarray] | None = None
        self._target_qpos: dict[str, np.ndarray] | None = None
        self._segment_start_monotonic: float | None = None
        self._last_reference_sequence: int | None = None
        self._last_reference_published_monotonic: float | None = None
        self._last_alpha = 1.0
        self._next_command_sequence = 0

    @property
    def epoch(self) -> int | None:
        with self._lock:
            return self._epoch

    @property
    def active_reference_sequence(self) -> int | None:
        with self._lock:
            return self._last_reference_sequence

    @property
    def last_emitted_qpos(self) -> Mapping[str, np.ndarray] | None:
        with self._lock:
            if self._last_emitted_qpos is None:
                return None
            return {
                side: qpos.copy()
                for side, qpos in self._last_emitted_qpos.items()
            }

    def reset(
        self,
        q_cmd_by_side: Mapping[str, np.ndarray | Sequence[float]],
        *,
        epoch: int,
        monotonic_time: float | None = None,
    ) -> None:
        now = self._resolve_time(monotonic_time)
        normalized_epoch = _normalize_epoch(epoch)
        qpos = _normalize_qpos_mapping(q_cmd_by_side, name="q_cmd_by_side")
        with self._lock:
            self._epoch = normalized_epoch
            self._last_emitted_qpos = qpos
            self._last_emitted_monotonic = now
            self._active_target = None
            self._segment_start_qpos = None
            self._target_qpos = None
            self._segment_start_monotonic = None
            self._last_reference_sequence = None
            self._last_reference_published_monotonic = None
            self._last_alpha = 1.0
            self._next_command_sequence = 0

    def accept_target(
        self,
        frame: PositionReferenceFrame,
        *,
        monotonic_time: float | None = None,
    ) -> None:
        if not isinstance(frame, PositionReferenceFrame):
            raise TypeError("frame must be a PositionReferenceFrame")
        if not frame.valid:
            raise ValueError("cannot interpolate an invalid position reference")
        self.accept_qpos_target(
            frame.q_ref_by_side,
            epoch=frame.epoch,
            sequence=frame.sequence,
            published_monotonic_time=frame.published_monotonic_time,
            monotonic_time=monotonic_time,
            source=frame.source,
            pipeline_version=frame.pipeline_version,
            hold=frame.hold,
        )

    def accept_qpos_target(
        self,
        q_target_by_side: Mapping[str, np.ndarray | Sequence[float]],
        *,
        epoch: int,
        sequence: int,
        published_monotonic_time: float,
        monotonic_time: float | None = None,
        source: str = "unknown",
        pipeline_version: str = POSITION_CONTROL_PIPELINE_VERSION,
        hold: bool = False,
        anchor_qpos_by_side: Mapping[str, np.ndarray | Sequence[float]] | None = None,
    ) -> None:
        """Accept a high-rate qpos reference without a runtime-specific wrapper.

        ``anchor_qpos_by_side`` defaults to the last command emitted by this
        planner.  A real transport should pass its last *accepted* qpos so a
        target arriving while an older SDK call is in flight cannot anchor to
        an unaccepted theoretical command.  Deterministic simulation normally
        leaves it unset because every planned command is applied.
        """

        now = self._resolve_time(monotonic_time)
        normalized_epoch = _normalize_epoch(epoch)
        normalized_sequence = _normalize_non_negative_int(sequence, name="sequence")
        published = float(published_monotonic_time)
        if not math.isfinite(published):
            raise ValueError("published_monotonic_time must be finite")
        normalized_source = str(source).strip()
        normalized_pipeline_version = str(pipeline_version).strip()
        if not normalized_source:
            raise ValueError("source must not be empty")
        if not normalized_pipeline_version:
            raise ValueError("pipeline_version must not be empty")
        target_qpos = _normalize_qpos_mapping(
            q_target_by_side,
            name="q_target_by_side",
        )
        anchor_qpos = (
            None
            if anchor_qpos_by_side is None
            else _normalize_qpos_mapping(
                anchor_qpos_by_side,
                name="anchor_qpos_by_side",
            )
        )
        with self._lock:
            self._require_initialized()
            assert self._epoch is not None
            assert self._last_emitted_qpos is not None
            assert self._last_emitted_monotonic is not None
            if now < self._last_emitted_monotonic:
                raise ValueError("monotonic_time cannot precede the last emitted command")
            if (
                self._segment_start_monotonic is not None
                and now < self._segment_start_monotonic
            ):
                raise ValueError("monotonic_time cannot precede the active segment start")
            if normalized_epoch != self._epoch:
                raise ValueError(
                    f"reference epoch {normalized_epoch} does not match active epoch {self._epoch}; "
                    "reset is required"
                )
            if published > now:
                raise ValueError("cannot accept a reference published in the monotonic future")
            if set(target_qpos) != set(self._last_emitted_qpos):
                raise ValueError(
                    "reference sides must exactly match reset sides for atomic command output"
                )
            if self._last_reference_sequence is not None:
                if normalized_sequence <= self._last_reference_sequence:
                    raise OutOfOrderReferenceError(
                        f"reference sequence {normalized_sequence} is not newer than "
                        f"{self._last_reference_sequence}"
                    )
                assert self._last_reference_published_monotonic is not None
                if published < self._last_reference_published_monotonic:
                    raise OutOfOrderReferenceError(
                        "reference published_monotonic_time moved backwards"
                    )

            for side, target in target_qpos.items():
                if target.shape != self._last_emitted_qpos[side].shape:
                    raise ValueError(
                        f"{side} target shape {target.shape} does not match reset shape "
                        f"{self._last_emitted_qpos[side].shape}"
                    )

            segment_anchor = self._last_emitted_qpos if anchor_qpos is None else anchor_qpos
            if set(segment_anchor) != set(self._last_emitted_qpos):
                raise ValueError(
                    "anchor sides must exactly match reset sides for atomic command output"
                )
            for side, anchor in segment_anchor.items():
                if anchor.shape != self._last_emitted_qpos[side].shape:
                    raise ValueError(
                        f"{side} anchor shape {anchor.shape} does not match reset shape "
                        f"{self._last_emitted_qpos[side].shape}"
                    )
            self._segment_start_qpos = {
                side: qpos.copy() for side, qpos in segment_anchor.items()
            }
            self._target_qpos = target_qpos
            self._segment_start_monotonic = now
            self._active_target = _AcceptedJointTarget(
                sequence=normalized_sequence,
                published_monotonic_time=published,
                source=normalized_source,
                pipeline_version=normalized_pipeline_version,
                hold=bool(hold),
            )
            self._last_reference_sequence = normalized_sequence
            self._last_reference_published_monotonic = published
            self._last_alpha = 0.0

    def set_segment_duration_sec(self, value: float) -> None:
        duration = float(value)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("segment_duration_sec must be finite and >= 0")
        with self._lock:
            self.segment_duration_sec = duration

    def step(self, *, monotonic_time: float | None = None) -> PositionCommandTick:
        now = self._resolve_time(monotonic_time)
        with self._lock:
            self._require_initialized()
            assert self._epoch is not None
            assert self._last_emitted_qpos is not None
            assert self._last_emitted_monotonic is not None
            if now < self._last_emitted_monotonic:
                raise ValueError("monotonic_time cannot move backwards")
            if (
                self._segment_start_monotonic is not None
                and now < self._segment_start_monotonic
            ):
                raise ValueError("monotonic_time cannot precede the active segment start")

            target = self._active_target
            if target is None:
                q_cmd = {
                    side: qpos.copy()
                    for side, qpos in self._last_emitted_qpos.items()
                }
                q_target = {
                    side: qpos.copy()
                    for side, qpos in self._last_emitted_qpos.items()
                }
                alpha = 1.0
                stale = False
                hold = True
                segment_start = None
                reference_sequence = None
                reference_published_monotonic = None
                metadata = {"reason": "waiting_for_reference"}
            else:
                assert self._segment_start_qpos is not None
                assert self._target_qpos is not None
                assert self._segment_start_monotonic is not None
                age, stale = reference_age_and_stale(
                    published_monotonic_time=target.published_monotonic_time,
                    sample_monotonic_time=now,
                    stale_after_sec=self.stale_after_sec,
                )
                if stale:
                    alpha = self._last_alpha
                    q_cmd = {
                        side: qpos.copy()
                        for side, qpos in self._last_emitted_qpos.items()
                    }
                else:
                    q_cmd = {}
                    alpha = 1.0
                    for side, start in self._segment_start_qpos.items():
                        qpos, alpha = interpolate_joint_command_segment(
                            start,
                            self._target_qpos[side],
                            segment_start_time=self._segment_start_monotonic,
                            segment_duration_sec=self.segment_duration_sec,
                            sample_time=now,
                        )
                        q_cmd[side] = qpos
                    self._last_alpha = alpha
                q_target = {
                    side: qpos.copy() for side, qpos in self._target_qpos.items()
                }
                hold = bool(target.hold or stale)
                segment_start = self._segment_start_monotonic
                reference_sequence = target.sequence
                reference_published_monotonic = target.published_monotonic_time
                metadata = {
                    "reference_source": target.source,
                    "pipeline_version": target.pipeline_version,
                    "reference_age_sec": age,
                }

            command_sequence = self._next_command_sequence
            tick = PositionCommandTick(
                epoch=self._epoch,
                command_sequence=command_sequence,
                reference_sequence=reference_sequence,
                emitted_monotonic_time=now,
                reference_published_monotonic_time=reference_published_monotonic,
                segment_start_monotonic_time=segment_start,
                segment_duration_sec=self.segment_duration_sec,
                interpolation_alpha=alpha,
                q_cmd_by_side=q_cmd,
                q_target_by_side=q_target,
                stale=stale,
                hold=hold,
                metadata=metadata,
            )
            self._last_emitted_qpos = {
                side: np.array(qpos, dtype=float, copy=True)
                for side, qpos in q_cmd.items()
            }
            self._last_emitted_monotonic = now
            self._next_command_sequence += 1
            return tick

    def _resolve_time(self, monotonic_time: float | None) -> float:
        value = self._clock() if monotonic_time is None else monotonic_time
        now = float(value)
        if not math.isfinite(now):
            raise ValueError("monotonic_time must be finite")
        return now

    def _require_initialized(self) -> None:
        if self._epoch is None or self._last_emitted_qpos is None:
            raise RuntimeError("interpolator must be reset before accepting or emitting commands")


def _normalize_epoch(value: int) -> int:
    return _normalize_non_negative_int(value, name="epoch")


def _normalize_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _normalize_qpos_mapping(
    values: Mapping[str, np.ndarray | Sequence[float]],
    *,
    name: str,
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must contain at least one side")
    normalized: dict[str, np.ndarray] = {}
    for raw_side, raw_qpos in values.items():
        side = str(raw_side).strip()
        if not side:
            raise ValueError(f"{name} contains an empty side")
        qpos = np.asarray(raw_qpos, dtype=float).reshape(-1)
        if qpos.size == 0 or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{name}.{side} must be a non-empty finite vector")
        if side in normalized:
            raise ValueError(f"{name} contains duplicate side {side!r}")
        normalized[side] = qpos.astype(float, copy=True)
    return normalized


__all__ = [
    "DEFAULT_POSITION_COMMAND_HZ",
    "DEFAULT_POSITION_REFERENCE_HZ",
    "DEFAULT_POSITION_SEGMENT_DURATION_SEC",
    "DEFAULT_REFERENCE_STALE_AFTER_SEC",
    "JointCommandInterpolator",
    "interpolate_joint_command_segment",
    "reference_age_and_stale",
]
