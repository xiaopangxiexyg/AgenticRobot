"""Embodiment-independent operator annotations for collection rows."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Mapping

from .arm_mode_labels import build_operator_annotation_arm_mode_metadata


OPERATOR_LABEL_SCHEMA = "dexteleop.operator_labels.v1"
SIDES = ("left", "right")
LABEL_TELEOP = "teleop"
LABEL_FIX_ARM = "fixed_hold"
LABEL_INIT_ARM = "init_pose"
LABELS = (LABEL_TELEOP, LABEL_FIX_ARM, LABEL_INIT_ARM)


def _side(value: str) -> str:
    side = str(value).strip().lower()
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {value!r}")
    return side


def _label(value: str) -> str:
    aliases = {
        "fix_arm": LABEL_FIX_ARM,
        "fix": LABEL_FIX_ARM,
        "init_arm": LABEL_INIT_ARM,
        "init": LABEL_INIT_ARM,
        "normal": LABEL_TELEOP,
    }
    label = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    if label not in LABELS:
        raise ValueError(f"unsupported operator label {value!r}")
    return label


@dataclass(frozen=True)
class OperatorLabelEvent:
    side: str
    label: str
    source: str
    timestamp: float
    sequence: int
    active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "label": self.label,
            "source": self.source,
            "timestamp": float(self.timestamp),
            "sequence": int(self.sequence),
            "active": bool(self.active),
        }


@dataclass(frozen=True)
class OperatorLabelSnapshot:
    labels_by_side: Mapping[str, str]
    source_by_side: Mapping[str, str]
    changed_at_by_side: Mapping[str, float]
    sequence: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OPERATOR_LABEL_SCHEMA,
            "annotation_only": True,
            "control_effect": False,
            "embodiment_has_arm": False,
            "labels_by_side": dict(self.labels_by_side),
            "source_by_side": dict(self.source_by_side),
            "changed_at_by_side": {
                side: float(value) for side, value in self.changed_at_by_side.items()
            },
            "sequence": int(self.sequence),
        }

    def arm_mode_compatibility_payload(self) -> dict[str, Any]:
        return build_operator_annotation_arm_mode_metadata(
            labels_by_side=self.labels_by_side,
            source_by_side=self.source_by_side,
            changed_at_by_side=self.changed_at_by_side,
            sequence=self.sequence,
        )


class OperatorLabelStateMachine:
    """Thread-safe, per-side latched labels shared by GUI/key/pedal inputs."""

    def __init__(self) -> None:
        now = time.time()
        self._labels = {side: LABEL_TELEOP for side in SIDES}
        self._sources = {side: "default" for side in SIDES}
        self._changed_at = {side: now for side in SIDES}
        self._sequence = 0
        self._events: list[OperatorLabelEvent] = []
        self._lock = threading.RLock()

    def reset(self, *, source: str = "episode_reset", timestamp: float | None = None) -> None:
        instant = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            for side in SIDES:
                self._set_locked(side, LABEL_TELEOP, source=source, timestamp=instant)

    def toggle(
        self,
        side: str,
        label: str,
        *,
        source: str,
        timestamp: float | None = None,
    ) -> OperatorLabelEvent:
        normalized_side = _side(side)
        normalized_label = _label(label)
        if normalized_label == LABEL_TELEOP:
            raise ValueError("toggle requires fixed_hold or init_pose")
        instant = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            target = (
                LABEL_TELEOP
                if self._labels[normalized_side] == normalized_label
                else normalized_label
            )
            return self._set_locked(
                normalized_side,
                target,
                source=str(source),
                timestamp=instant,
            )

    def set(
        self,
        side: str,
        label: str,
        *,
        source: str,
        timestamp: float | None = None,
    ) -> OperatorLabelEvent:
        instant = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            return self._set_locked(
                _side(side), _label(label), source=str(source), timestamp=instant
            )

    def _set_locked(
        self,
        side: str,
        label: str,
        *,
        source: str,
        timestamp: float,
    ) -> OperatorLabelEvent:
        self._sequence += 1
        self._labels[side] = label
        self._sources[side] = source
        self._changed_at[side] = float(timestamp)
        event = OperatorLabelEvent(
            side=side,
            label=label,
            source=source,
            timestamp=float(timestamp),
            sequence=self._sequence,
            active=label != LABEL_TELEOP,
        )
        self._events.append(event)
        return event

    def snapshot(self) -> OperatorLabelSnapshot:
        with self._lock:
            return OperatorLabelSnapshot(
                labels_by_side=dict(self._labels),
                source_by_side=dict(self._sources),
                changed_at_by_side=dict(self._changed_at),
                sequence=self._sequence,
            )

    def events_since(self, sequence: int = 0) -> tuple[OperatorLabelEvent, ...]:
        with self._lock:
            return tuple(event for event in self._events if event.sequence > int(sequence))


__all__ = [
    "LABEL_FIX_ARM",
    "LABEL_INIT_ARM",
    "LABEL_TELEOP",
    "OPERATOR_LABEL_SCHEMA",
    "OperatorLabelEvent",
    "OperatorLabelSnapshot",
    "OperatorLabelStateMachine",
]
