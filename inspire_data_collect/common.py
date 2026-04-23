from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np

Side = Literal["left", "right"]
ALL_SIDES: tuple[Side, Side] = ("left", "right")


@dataclass
class ObservationChannel:
    channel_name: str
    side: str
    value: np.ndarray
    timestamp: float
    valid: bool = True
    source_name: str = ""
    metadata: dict[str, Any] | None = None

    def copy(self) -> "ObservationChannel":
        return ObservationChannel(
            channel_name=self.channel_name,
            side=self.side,
            value=np.asarray(self.value, dtype=float).copy(),
            timestamp=float(self.timestamp),
            valid=bool(self.valid),
            source_name=self.source_name,
            metadata={} if self.metadata is None else dict(self.metadata),
        )


@dataclass
class EffectorCommand:
    effector_name: str
    side: str
    value: np.ndarray
    timestamp: float
    metadata: dict[str, Any] | None = None

    def copy(self) -> "EffectorCommand":
        return EffectorCommand(
            effector_name=self.effector_name,
            side=self.side,
            value=np.asarray(self.value, dtype=np.float32).copy(),
            timestamp=float(self.timestamp),
            metadata={} if self.metadata is None else dict(self.metadata),
        )


def parse_active_sides(text: str) -> tuple[Side, ...]:
    value = str(text).strip().lower()
    if value == "both":
        return ALL_SIDES
    if value in ALL_SIDES:
        return (cast(Side, value),)
    raise argparse.ArgumentTypeError("active-sides must be one of: left, right, both")


def normalize_active_sides(value: object) -> tuple[Side, ...]:
    if value is None:
        return ALL_SIDES
    if isinstance(value, str):
        return parse_active_sides(value)
    sides = tuple(str(side).strip().lower() for side in cast(Any, value) if str(side).strip())
    invalid = set(sides) - set(ALL_SIDES)
    if invalid:
        raise ValueError(f"Unsupported side(s): {sorted(invalid)}")
    ordered = tuple(side for side in ALL_SIDES if side in set(sides))
    if not ordered:
        raise ValueError("At least one side is required.")
    return cast(tuple[Side, ...], ordered)


def assert_inactive_side_config_empty(
    args: Any,
    active_sides: Sequence[str],
    *,
    suffixes: Sequence[str],
) -> None:
    active = set(str(side) for side in active_sides)
    for side in ALL_SIDES:
        if side in active:
            continue
        present = [
            f"--{side.replace('_', '-')}-{suffix.replace('_', '-')}"
            for suffix in suffixes
            if str(getattr(args, f"{side}_{suffix}", "") or "")
        ]
        if present:
            raise ValueError(
                f"Inactive side '{side}' received configuration {present}; "
                "remove those flags or include the side in --active-sides."
            )


def vector_or_none(value: Any, *, length: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if length is not None and array.shape != (int(length),):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array.astype(np.float32, copy=True)


def vector_list(value: Any, *, length: int | None = None) -> list[float] | None:
    array = vector_or_none(value, length=length)
    if array is None:
        return None
    return array.tolist()


def nan_vector(length: int) -> np.ndarray:
    return np.full(int(length), np.nan, dtype=np.float32)


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value
