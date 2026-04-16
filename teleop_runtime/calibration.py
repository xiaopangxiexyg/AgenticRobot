from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import as_pose_matrix, clip_pose_translation, identity_pose, invert_pose


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _array_dict(value: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(item, dtype=float) for key, item in value.items()}


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


@dataclass
class DeskFrameConfig:
    name: str = "desk"
    axes: dict[str, str] = field(
        default_factory=lambda: {"x": "forward", "y": "left", "z": "up"}
    )


@dataclass
class ArmCalibration:
    name: str
    side: str
    T_D_B: np.ndarray | None = None
    workspace_min: np.ndarray | None = None
    workspace_max: np.ndarray | None = None
    home_qpos: np.ndarray | None = None
    home_pose_debug: np.ndarray | None = None
    home_pose_fine: np.ndarray | None = None
    tcp_debug_frame: str = "rm_eef"
    tcp_fine_frame: str = "hand_base"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_D_B = _array_or_none(self.T_D_B)
        self.workspace_min = _array_or_none(self.workspace_min)
        self.workspace_max = _array_or_none(self.workspace_max)
        self.home_qpos = _array_or_none(self.home_qpos)
        self.home_pose_debug = _array_or_none(self.home_pose_debug)
        self.home_pose_fine = _array_or_none(self.home_pose_fine)

    def tcp_frame_name(self, tcp_mode: str) -> str:
        if tcp_mode == "rm_eef":
            return self.tcp_debug_frame
        if tcp_mode == "hand_base":
            return self.tcp_fine_frame
        raise ValueError(f"Unsupported tcp_mode: {tcp_mode}")

    def home_pose_for_tcp_mode(self, tcp_mode: str) -> np.ndarray | None:
        if tcp_mode == "rm_eef":
            return None if self.home_pose_debug is None else self.home_pose_debug.copy()
        if tcp_mode == "hand_base":
            return None if self.home_pose_fine is None else self.home_pose_fine.copy()
        raise ValueError(f"Unsupported tcp_mode: {tcp_mode}")


@dataclass
class HandCalibration:
    name: str
    side: str
    backend: str = "inspire_tac"
    open_command: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.open_command = _array_or_none(self.open_command)


@dataclass
class TeleopCalibration:
    desk_frame: DeskFrameConfig = field(default_factory=DeskFrameConfig)
    arms: dict[str, ArmCalibration] = field(default_factory=dict)
    hands: dict[str, HandCalibration] = field(default_factory=dict)
    T_W_O: np.ndarray | None = None
    T_O_D: np.ndarray | None = None
    tracker_offsets: dict[str, np.ndarray] = field(default_factory=dict)
    captured_home_qpos: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_W_O = _array_or_none(self.T_W_O)
        self.T_O_D = _array_or_none(self.T_O_D)
        self.tracker_offsets = _array_dict(self.tracker_offsets)
        self.captured_home_qpos = _array_dict(self.captured_home_qpos)

    def set_world_to_operator(self, pose: Any) -> None:
        self.T_W_O = as_pose_matrix(pose)

    def set_operator_to_desk(self, pose: Any) -> None:
        self.T_O_D = as_pose_matrix(pose)

    def set_world_to_desk(self, pose: Any) -> None:
        self.T_W_O = identity_pose()
        self.T_O_D = as_pose_matrix(pose)
        self.metadata["world_to_desk_mode"] = "direct"

    def world_to_desk_pose(self) -> np.ndarray | None:
        if self.T_W_O is None or self.T_O_D is None:
            return None
        return as_pose_matrix(self.T_W_O) @ as_pose_matrix(self.T_O_D)

    def set_tracker_offset(self, tracker_role: str, pose: Any) -> None:
        self.tracker_offsets[tracker_role] = as_pose_matrix(pose)

    def capture_home_qpos(self, effector_name: str, qpos: Any) -> None:
        qpos_array = np.asarray(qpos, dtype=float)
        self.captured_home_qpos[effector_name] = qpos_array
        if effector_name in self.arms:
            self.arms[effector_name].home_qpos = qpos_array.copy()

    def capture_home_pose(self, effector_name: str, pose: Any, *, tcp_mode: str) -> None:
        pose_matrix = as_pose_matrix(pose)
        if effector_name not in self.arms:
            raise KeyError(f"No arm calibration registered for effector '{effector_name}'.")
        if tcp_mode == "rm_eef":
            self.arms[effector_name].home_pose_debug = pose_matrix
            return
        if tcp_mode == "hand_base":
            self.arms[effector_name].home_pose_fine = pose_matrix
            return
        raise ValueError(f"Unsupported tcp_mode: {tcp_mode}")

    def get_tracker_offset(self, tracker_role: str) -> np.ndarray:
        return self.tracker_offsets.get(tracker_role, identity_pose()).copy()

    def get_arm_for_side(self, side: str) -> ArmCalibration:
        for arm in self.arms.values():
            if arm.side == side:
                return arm
        raise KeyError(f"No arm calibration registered for side '{side}'.")

    def world_to_operator(self, pose_world: Any, tracker_role: str | None = None) -> np.ndarray:
        if self.T_W_O is None:
            raise RuntimeError("Runtime calibration missing T_W_O.")
        pose_operator = invert_pose(self.T_W_O) @ as_pose_matrix(pose_world)
        if tracker_role is not None:
            pose_operator = pose_operator @ self.get_tracker_offset(tracker_role)
        return pose_operator

    def operator_to_desk(self, pose_operator: Any) -> np.ndarray:
        if self.T_O_D is None:
            raise RuntimeError("Runtime calibration missing T_O_D.")
        return invert_pose(self.T_O_D) @ as_pose_matrix(pose_operator)

    def world_to_desk(self, pose_world: Any, tracker_role: str | None = None) -> np.ndarray:
        return self.operator_to_desk(self.world_to_operator(pose_world, tracker_role=tracker_role))

    def desk_to_base(self, pose_desk: Any, side: str) -> np.ndarray:
        arm = self.get_arm_for_side(side)
        if arm.T_D_B is None:
            raise RuntimeError(f"Arm calibration for side '{side}' is missing T_D_B.")
        return invert_pose(arm.T_D_B) @ as_pose_matrix(pose_desk)

    def world_to_base(self, pose_world: Any, side: str, tracker_role: str | None = None) -> np.ndarray:
        return self.desk_to_base(
            self.world_to_desk(pose_world, tracker_role=tracker_role),
            side=side,
        )

    def target_frame_name(self, side: str, tcp_mode: str) -> str:
        return self.get_arm_for_side(side).tcp_frame_name(tcp_mode)

    def clip_pose_to_workspace(self, pose_desk: Any, side: str) -> np.ndarray:
        arm = self.get_arm_for_side(side)
        return clip_pose_translation(pose_desk, arm.workspace_min, arm.workspace_max)

    def to_dict(self) -> dict[str, Any]:
        return {
            "desk_frame": {
                "name": self.desk_frame.name,
                "axes": dict(self.desk_frame.axes),
            },
            "arms": {
                name: {
                    "name": arm.name,
                    "side": arm.side,
                    "T_D_B": _to_serializable(arm.T_D_B),
                    "workspace_min": _to_serializable(arm.workspace_min),
                    "workspace_max": _to_serializable(arm.workspace_max),
                    "home_qpos": _to_serializable(arm.home_qpos),
                    "home_pose_debug": _to_serializable(arm.home_pose_debug),
                    "home_pose_fine": _to_serializable(arm.home_pose_fine),
                    "tcp_debug_frame": arm.tcp_debug_frame,
                    "tcp_fine_frame": arm.tcp_fine_frame,
                    "metadata": _to_serializable(arm.metadata),
                }
                for name, arm in self.arms.items()
            },
            "hands": {
                name: {
                    "name": hand.name,
                    "side": hand.side,
                    "backend": hand.backend,
                    "open_command": _to_serializable(hand.open_command),
                    "metadata": _to_serializable(hand.metadata),
                }
                for name, hand in self.hands.items()
            },
            "T_W_O": _to_serializable(self.T_W_O),
            "T_O_D": _to_serializable(self.T_O_D),
            "tracker_offsets": _to_serializable(self.tracker_offsets),
            "captured_home_qpos": _to_serializable(self.captured_home_qpos),
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeleopCalibration":
        desk_payload = payload.get("desk_frame", {})
        arms = {
            name: ArmCalibration(**arm_payload)
            for name, arm_payload in payload.get("arms", {}).items()
        }
        hands = {
            name: HandCalibration(**hand_payload)
            for name, hand_payload in payload.get("hands", {}).items()
        }
        return cls(
            desk_frame=DeskFrameConfig(
                name=desk_payload.get("name", "desk"),
                axes=desk_payload.get("axes", {"x": "forward", "y": "left", "z": "up"}),
            ),
            arms=arms,
            hands=hands,
            T_W_O=payload.get("T_W_O"),
            T_O_D=payload.get("T_O_D"),
            tracker_offsets=payload.get("tracker_offsets", {}),
            captured_home_qpos=payload.get("captured_home_qpos", {}),
            metadata=payload.get("metadata", {}),
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "TeleopCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
