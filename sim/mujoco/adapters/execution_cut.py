"""Atomic read-only cuts exported by the MuJoCo execution owner.

The cut is private backend state used to derive typed Reference/Measurement
observations.  It does not grant authority, advance a cursor, or expose a
second execution timeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _qpos_mapping(
    values: Mapping[str, Sequence[float]],
    *,
    name: str,
) -> Mapping[str, tuple[float, ...]]:
    normalized: dict[str, tuple[float, ...]] = {}
    for raw_source_id, raw_qpos in values.items():
        source_id = str(raw_source_id).strip()
        qpos = np.asarray(raw_qpos, dtype=float).reshape(-1)
        if (
            not source_id
            or source_id in normalized
            or qpos.size == 0
            or not np.all(np.isfinite(qpos))
        ):
            raise ValueError(f"{name} contains an invalid qpos for {source_id!r}.")
        normalized[source_id] = tuple(float(value) for value in qpos)
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class MujocoExecutionStateCut:
    """One lock-consistent qref/measurement/model-state snapshot."""

    sampled_at: float
    measured_qpos_by_source: Mapping[str, Sequence[float]]
    model_qpos: Sequence[float]
    physics_tick: int | None
    active_reference_sequence: int | None
    active_qref_by_source: Mapping[str, Sequence[float]] | None
    active_reference_segment_id: str | None = None
    active_execution_fence_sequence: int | None = None
    active_group_segment_generation: int | None = None

    def __post_init__(self) -> None:
        sampled_at = _finite(self.sampled_at, name="sampled_at")
        measured = _qpos_mapping(
            self.measured_qpos_by_source,
            name="measured_qpos_by_source",
        )
        model_qpos = np.asarray(self.model_qpos, dtype=float).reshape(-1)
        if model_qpos.size == 0 or not np.all(np.isfinite(model_qpos)):
            raise ValueError("model_qpos must contain finite values.")
        frozen_model_qpos = tuple(float(value) for value in model_qpos)
        tick = self.physics_tick
        if tick is not None:
            if isinstance(tick, bool) or int(tick) != tick or int(tick) < 0:
                raise ValueError("physics_tick must be a non-negative integer.")
            tick = int(tick)

        sequence = self.active_reference_sequence
        qref = self.active_qref_by_source
        if (sequence is None) != (qref is None):
            raise ValueError(
                "active reference sequence and qref must appear together."
            )
        if sequence is not None:
            if (
                isinstance(sequence, bool)
                or int(sequence) != sequence
                or int(sequence) < 0
            ):
                raise ValueError(
                    "active_reference_sequence must be a non-negative integer."
                )
            sequence = int(sequence)
            qref = _qpos_mapping(qref or {}, name="active_qref_by_source")
            if set(qref) != set(measured):
                raise ValueError(
                    "Active qref and measured qpos must cover the same sources."
                )

        segment_id = (
            None
            if self.active_reference_segment_id is None
            else str(self.active_reference_segment_id).strip()
        )
        fence = self.active_execution_fence_sequence
        generation = self.active_group_segment_generation
        scheduled_identity = (segment_id, fence, generation)
        if any(value is not None for value in scheduled_identity) and not all(
            value is not None for value in scheduled_identity
        ):
            raise ValueError(
                "Scheduled active reference identity must be complete."
            )
        if segment_id == "":
            raise ValueError("active_reference_segment_id must be non-empty.")
        for name, value in (
            ("active_execution_fence_sequence", fence),
            ("active_group_segment_generation", generation),
        ):
            if value is not None and (
                isinstance(value, bool) or int(value) != value or int(value) < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer.")

        object.__setattr__(self, "sampled_at", sampled_at)
        object.__setattr__(self, "measured_qpos_by_source", measured)
        object.__setattr__(self, "model_qpos", frozen_model_qpos)
        object.__setattr__(self, "physics_tick", tick)
        object.__setattr__(self, "active_reference_sequence", sequence)
        object.__setattr__(self, "active_qref_by_source", qref)
        object.__setattr__(self, "active_reference_segment_id", segment_id)
        object.__setattr__(
            self,
            "active_execution_fence_sequence",
            None if fence is None else int(fence),
        )
        object.__setattr__(
            self,
            "active_group_segment_generation",
            None if generation is None else int(generation),
        )


__all__ = ["MujocoExecutionStateCut"]
