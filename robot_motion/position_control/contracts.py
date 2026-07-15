from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


POSITION_CONTROL_PIPELINE_VERSION = "position_control.v1"


def _finite_time(value: float, *, name: str) -> float:
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise ValueError(f"{name} must be finite")
    return timestamp


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _readonly_array(
    value: np.ndarray | Sequence[float],
    *,
    name: str,
    vector: bool = False,
) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if vector:
        array = array.reshape(-1)
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
    elif array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _freeze_value(value: Any) -> Any:
    """Detach common metadata containers and make their contents read-only."""

    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return MappingProxyType(
            {copy.deepcopy(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return copy.deepcopy(value)


def _freeze_mapping(value: Mapping[str, Any] | None, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _normalize_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    if isinstance(reasons, str):
        reasons = (reasons,)
    normalized: list[str] = []
    for reason in reasons:
        text = str(reason).strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


@dataclass(frozen=True)
class PositionSideReference:
    """One side of an atomic position reference frame.

    ``q_ref`` and ``pose_ref`` are always finite executable values. ``q_ik_raw``
    may be ``None`` when IK failed and the side publishes an invalid hold.
    Arrays and metadata are detached from caller-owned objects and exposed as
    read-only values.
    """

    side: str
    q_ik_raw: np.ndarray | Sequence[float] | None
    q_ref: np.ndarray | Sequence[float]
    pose_ref: np.ndarray | Sequence[float]
    valid: bool = True
    hold: bool = False
    modification_reasons: tuple[str, ...] | Sequence[str] = ()
    frame_metadata: Mapping[str, Any] = field(default_factory=dict)
    profile_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        side = str(self.side).strip()
        if not side:
            raise ValueError("side must not be empty")
        q_ref = _readonly_array(self.q_ref, name=f"{side}.q_ref", vector=True)
        q_ik_raw = (
            None
            if self.q_ik_raw is None
            else _readonly_array(self.q_ik_raw, name=f"{side}.q_ik_raw", vector=True)
        )
        if q_ik_raw is not None and q_ik_raw.shape != q_ref.shape:
            raise ValueError(
                f"{side}.q_ik_raw shape {q_ik_raw.shape} does not match "
                f"q_ref shape {q_ref.shape}"
            )

        object.__setattr__(self, "side", side)
        object.__setattr__(self, "q_ik_raw", q_ik_raw)
        object.__setattr__(self, "q_ref", q_ref)
        object.__setattr__(
            self,
            "pose_ref",
            _readonly_array(self.pose_ref, name=f"{side}.pose_ref"),
        )
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "hold", bool(self.hold))
        object.__setattr__(
            self,
            "modification_reasons",
            _normalize_reasons(self.modification_reasons),
        )
        object.__setattr__(
            self,
            "frame_metadata",
            _freeze_mapping(self.frame_metadata, name=f"{side}.frame_metadata"),
        )
        object.__setattr__(
            self,
            "profile_metadata",
            _freeze_mapping(self.profile_metadata, name=f"{side}.profile_metadata"),
        )


@dataclass(frozen=True)
class PositionReferenceFrame:
    """Immutable, atomically published single- or multi-arm qpos reference."""

    epoch: int
    sequence: int
    source_wall_time: float
    generated_wall_time: float
    published_wall_time: float
    generated_monotonic_time: float
    published_monotonic_time: float
    references: Mapping[str, PositionSideReference]
    source: str = "unknown"
    pipeline_version: str = POSITION_CONTROL_PIPELINE_VERSION
    frame_metadata: Mapping[str, Any] = field(default_factory=dict)
    profile_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        epoch = _non_negative_int(self.epoch, name="epoch")
        sequence = _non_negative_int(self.sequence, name="sequence")
        source = str(self.source).strip()
        pipeline_version = str(self.pipeline_version).strip()
        if not source:
            raise ValueError("source must not be empty")
        if not pipeline_version:
            raise ValueError("pipeline_version must not be empty")
        if not isinstance(self.references, Mapping) or not self.references:
            raise ValueError("references must contain at least one side")

        detached_references: dict[str, PositionSideReference] = {}
        for raw_side, reference in self.references.items():
            side = str(raw_side).strip()
            if not isinstance(reference, PositionSideReference):
                raise TypeError(f"references[{side!r}] must be a PositionSideReference")
            if side != reference.side:
                raise ValueError(
                    f"reference mapping key {side!r} does not match side {reference.side!r}"
                )
            if side in detached_references:
                raise ValueError(f"duplicate side {side!r}")
            detached_references[side] = reference

        generated_monotonic = _finite_time(
            self.generated_monotonic_time,
            name="generated_monotonic_time",
        )
        published_monotonic = _finite_time(
            self.published_monotonic_time,
            name="published_monotonic_time",
        )
        if published_monotonic < generated_monotonic:
            raise ValueError(
                "published_monotonic_time must be >= generated_monotonic_time"
            )

        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(
            self,
            "source_wall_time",
            _finite_time(self.source_wall_time, name="source_wall_time"),
        )
        object.__setattr__(
            self,
            "generated_wall_time",
            _finite_time(self.generated_wall_time, name="generated_wall_time"),
        )
        object.__setattr__(
            self,
            "published_wall_time",
            _finite_time(self.published_wall_time, name="published_wall_time"),
        )
        object.__setattr__(self, "generated_monotonic_time", generated_monotonic)
        object.__setattr__(self, "published_monotonic_time", published_monotonic)
        object.__setattr__(self, "references", MappingProxyType(detached_references))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "pipeline_version", pipeline_version)
        object.__setattr__(
            self,
            "frame_metadata",
            _freeze_mapping(self.frame_metadata, name="frame_metadata"),
        )
        object.__setattr__(
            self,
            "profile_metadata",
            _freeze_mapping(self.profile_metadata, name="profile_metadata"),
        )

    @property
    def sides(self) -> tuple[str, ...]:
        return tuple(self.references)

    @property
    def valid(self) -> bool:
        return all(reference.valid for reference in self.references.values())

    @property
    def hold(self) -> bool:
        return any(reference.hold for reference in self.references.values())

    @property
    def all_hold(self) -> bool:
        return all(reference.hold for reference in self.references.values())

    @property
    def modification_reasons_by_side(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                side: reference.modification_reasons
                for side, reference in self.references.items()
            }
        )

    @property
    def q_ik_raw_by_side(self) -> Mapping[str, np.ndarray | None]:
        return MappingProxyType(
            {side: reference.q_ik_raw for side, reference in self.references.items()}
        )

    @property
    def q_ref_by_side(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {side: reference.q_ref for side, reference in self.references.items()}
        )

    @property
    def pose_ref_by_side(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {side: reference.pose_ref for side, reference in self.references.items()}
        )


@dataclass(frozen=True)
class PositionCommandTick:
    """One atomic command tick emitted for every side in a control epoch."""

    epoch: int
    command_sequence: int
    reference_sequence: int | None
    emitted_monotonic_time: float
    reference_published_monotonic_time: float | None
    segment_start_monotonic_time: float | None
    segment_duration_sec: float
    interpolation_alpha: float
    q_cmd_by_side: Mapping[str, np.ndarray | Sequence[float]]
    q_target_by_side: Mapping[str, np.ndarray | Sequence[float]]
    stale: bool = False
    hold: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        epoch = _non_negative_int(self.epoch, name="epoch")
        command_sequence = _non_negative_int(
            self.command_sequence,
            name="command_sequence",
        )
        reference_sequence = (
            None
            if self.reference_sequence is None
            else _non_negative_int(self.reference_sequence, name="reference_sequence")
        )
        emitted_monotonic = _finite_time(
            self.emitted_monotonic_time,
            name="emitted_monotonic_time",
        )
        reference_published_monotonic = (
            None
            if self.reference_published_monotonic_time is None
            else _finite_time(
                self.reference_published_monotonic_time,
                name="reference_published_monotonic_time",
            )
        )
        segment_start_monotonic = (
            None
            if self.segment_start_monotonic_time is None
            else _finite_time(
                self.segment_start_monotonic_time,
                name="segment_start_monotonic_time",
            )
        )
        duration = float(self.segment_duration_sec)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("segment_duration_sec must be finite and >= 0")
        alpha = float(self.interpolation_alpha)
        if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError("interpolation_alpha must be finite and in [0, 1]")

        q_cmd = _freeze_qpos_mapping(self.q_cmd_by_side, name="q_cmd_by_side")
        q_target = _freeze_qpos_mapping(self.q_target_by_side, name="q_target_by_side")
        if set(q_cmd) != set(q_target):
            raise ValueError("q_cmd_by_side and q_target_by_side must contain the same sides")
        for side in q_cmd:
            if q_cmd[side].shape != q_target[side].shape:
                raise ValueError(f"{side} command and target qpos shapes do not match")

        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "command_sequence", command_sequence)
        object.__setattr__(self, "reference_sequence", reference_sequence)
        object.__setattr__(self, "emitted_monotonic_time", emitted_monotonic)
        object.__setattr__(
            self,
            "reference_published_monotonic_time",
            reference_published_monotonic,
        )
        object.__setattr__(
            self,
            "segment_start_monotonic_time",
            segment_start_monotonic,
        )
        object.__setattr__(self, "segment_duration_sec", duration)
        object.__setattr__(self, "interpolation_alpha", alpha)
        object.__setattr__(self, "q_cmd_by_side", q_cmd)
        object.__setattr__(self, "q_target_by_side", q_target)
        object.__setattr__(self, "stale", bool(self.stale))
        object.__setattr__(self, "hold", bool(self.hold))
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, name="metadata"),
        )

    @property
    def sides(self) -> tuple[str, ...]:
        return tuple(self.q_cmd_by_side)


def _freeze_qpos_mapping(
    values: Mapping[str, np.ndarray | Sequence[float]],
    *,
    name: str,
) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must contain at least one side")
    normalized: dict[str, np.ndarray] = {}
    for raw_side, qpos in values.items():
        side = str(raw_side).strip()
        if not side:
            raise ValueError(f"{name} contains an empty side")
        if side in normalized:
            raise ValueError(f"{name} contains duplicate side {side!r}")
        normalized[side] = _readonly_array(qpos, name=f"{name}.{side}", vector=True)
    return MappingProxyType(normalized)


__all__ = [
    "POSITION_CONTROL_PIPELINE_VERSION",
    "PositionCommandTick",
    "PositionReferenceFrame",
    "PositionSideReference",
]
