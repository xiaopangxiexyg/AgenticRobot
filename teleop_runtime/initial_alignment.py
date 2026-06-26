from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .arm_frames import TOOL_TCP_FRAME
from .tcp import canonical_tcp_frame
from .transforms import as_pose_matrix


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _pose_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return as_pose_matrix(value)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


@dataclass
class AlignmentSideState:
    side: str
    arm_qpos: np.ndarray | None = None
    tcp_pose_base: np.ndarray | None = None
    tcp_pose_desk: np.ndarray | None = None
    hand_qpos_q6: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.side = str(self.side)
        self.arm_qpos = _array_or_none(self.arm_qpos)
        self.tcp_pose_base = _pose_or_none(self.tcp_pose_base)
        self.tcp_pose_desk = _pose_or_none(self.tcp_pose_desk)
        self.hand_qpos_q6 = _array_or_none(self.hand_qpos_q6)


@dataclass
class TeleopInitialAlignment:
    tcp_frame: str = TOOL_TCP_FRAME
    sides: dict[str, AlignmentSideState] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tcp_frame = canonical_tcp_frame(self.tcp_frame)
        normalized: dict[str, AlignmentSideState] = {}
        for side, state in self.sides.items():
            normalized[str(side)] = state if isinstance(state, AlignmentSideState) else AlignmentSideState(**state)
        self.sides = normalized

    def get_side(self, side: str) -> AlignmentSideState | None:
        return self.sides.get(str(side))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "teleop_initial_alignment",
            "format_version": 1,
            "tcp_frame": self.tcp_frame,
            "sides": {
                side: {
                    "side": state.side,
                    "arm_qpos": _to_serializable(state.arm_qpos),
                    "tcp_pose_base": _to_serializable(state.tcp_pose_base),
                    "tcp_pose_desk": _to_serializable(state.tcp_pose_desk),
                    "hand_qpos_q6": _to_serializable(state.hand_qpos_q6),
                    "metadata": _to_serializable(state.metadata),
                }
                for side, state in self.sides.items()
            },
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeleopInitialAlignment":
        return cls(
            tcp_frame=payload.get("tcp_frame", TOOL_TCP_FRAME),
            sides={
                side: AlignmentSideState(**state_payload)
                for side, state_payload in payload.get("sides", {}).items()
            },
            metadata=payload.get("metadata", {}),
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "TeleopInitialAlignment":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
