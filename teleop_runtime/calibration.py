from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .arm_frames import TOOL_TCP_FRAME
from .tcp import canonical_tcp_frame
from .transforms import as_pose_matrix, clip_pose_translation, identity_pose, invert_pose

VIRTUAL_DESK_SWEEP_CONVENTION_VERSION = 2
DEFAULT_VIRTUAL_DESK_POST_YAW_DEG = 180.0


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _array_dict(value: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(item, dtype=float) for key, item in value.items()}


def _validate_transform_alias(
    payload: dict[str, Any], *, canonical_key: str, legacy_key: str, context: str
) -> None:
    if canonical_key not in payload or legacy_key not in payload:
        return
    canonical = _array_or_none(payload[canonical_key])
    legacy = _array_or_none(payload[legacy_key])
    if canonical is None or legacy is None:
        equivalent = canonical is None and legacy is None
    else:
        equivalent = canonical.shape == legacy.shape and bool(
            np.allclose(canonical, legacy, atol=1e-9, rtol=0.0)
        )
    if not equivalent:
        raise ValueError(
            f"Conflicting calibration aliases {context}.{canonical_key} and "
            f"{context}.{legacy_key}."
        )


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def _yaw_pose_deg(yaw_deg: float) -> np.ndarray:
    yaw_rad = np.deg2rad(float(yaw_deg))
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    pose = identity_pose()
    pose[:3, :3] = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return pose


@dataclass
class ViveRetargetFrameConfig:
    """Historical Vive retarget frame; unrelated to dataset canonical desk."""

    name: str = "vive_retarget_frame"
    axes: dict[str, str] = field(
        default_factory=lambda: {"x": "forward", "y": "left", "z": "up"}
    )


# Compatibility import for existing presets and downstream integrations.
DeskFrameConfig = ViveRetargetFrameConfig


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
    tcp_debug_frame: str = TOOL_TCP_FRAME
    tcp_fine_frame: str = TOOL_TCP_FRAME
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_D_B = _array_or_none(self.T_D_B)
        self.workspace_min = _array_or_none(self.workspace_min)
        self.workspace_max = _array_or_none(self.workspace_max)
        self.home_qpos = _array_or_none(self.home_qpos)
        self.home_pose_debug = _array_or_none(self.home_pose_debug)
        self.home_pose_fine = _array_or_none(self.home_pose_fine)
        active_home_pose = self.home_pose_fine
        if active_home_pose is None:
            active_home_pose = self.home_pose_debug
        if active_home_pose is not None:
            pose = as_pose_matrix(active_home_pose)
            self.home_pose_debug = pose.copy()
            self.home_pose_fine = pose.copy()
        self.tcp_debug_frame = canonical_tcp_frame(self.tcp_debug_frame or TOOL_TCP_FRAME)
        self.tcp_fine_frame = canonical_tcp_frame(self.tcp_fine_frame or TOOL_TCP_FRAME)

    def tcp_frame_name(self, tcp_mode: str | None = None) -> str:
        return canonical_tcp_frame(TOOL_TCP_FRAME if tcp_mode is None else tcp_mode)

    @property
    def T_VR_B(self) -> np.ndarray | None:
        """``T_vive_retarget_frame_from_arm_base`` compatibility-safe name."""

        return None if self.T_D_B is None else self.T_D_B.copy()

    @T_VR_B.setter
    def T_VR_B(self, value: Any) -> None:
        self.T_D_B = _array_or_none(value)

    def home_pose_for_tcp_mode(self, tcp_mode: str | None = None) -> np.ndarray | None:
        _ = canonical_tcp_frame(TOOL_TCP_FRAME if tcp_mode is None else tcp_mode)
        if self.home_pose_fine is not None:
            return self.home_pose_fine.copy()
        if self.home_pose_debug is not None:
            return self.home_pose_debug.copy()
        return None


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

    @property
    def vive_retarget_frame(self) -> ViveRetargetFrameConfig:
        return self.desk_frame

    @vive_retarget_frame.setter
    def vive_retarget_frame(self, value: ViveRetargetFrameConfig) -> None:
        self.desk_frame = value

    @property
    def T_O_VR(self) -> np.ndarray | None:
        """Historical operator-from-Vive-retarget transform."""

        return None if self.T_O_D is None else self.T_O_D.copy()

    @T_O_VR.setter
    def T_O_VR(self, value: Any) -> None:
        self.T_O_D = _array_or_none(value)

    def set_world_to_operator(self, pose: Any) -> None:
        self.T_W_O = as_pose_matrix(pose)

    def set_operator_to_desk(self, pose: Any) -> None:
        self.set_operator_to_vive_retarget(pose)

    def set_operator_to_vive_retarget(self, pose: Any) -> None:
        self.T_O_D = as_pose_matrix(pose)

    def set_world_to_desk(self, pose: Any) -> None:
        self.set_world_to_vive_retarget(pose)

    def set_world_to_vive_retarget(self, pose: Any) -> None:
        self.T_W_O = identity_pose()
        self.T_O_D = as_pose_matrix(pose)
        self.metadata["world_to_vive_retarget_mode"] = "direct"
        # Preserve the historical audit key for old calibration consumers.
        self.metadata["world_to_desk_mode"] = "direct"

    def world_to_desk_pose(self) -> np.ndarray | None:
        return self.world_to_vive_retarget_pose()

    def world_to_vive_retarget_pose(self) -> np.ndarray | None:
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

    def capture_home_pose(
        self,
        effector_name: str,
        pose: Any,
        *,
        tcp_mode: str | None = None,
    ) -> None:
        pose_matrix = as_pose_matrix(pose)
        if effector_name not in self.arms:
            raise KeyError(f"No arm calibration registered for effector '{effector_name}'.")
        _ = canonical_tcp_frame(TOOL_TCP_FRAME if tcp_mode is None else tcp_mode)
        self.arms[effector_name].home_pose_debug = pose_matrix.copy()
        self.arms[effector_name].home_pose_fine = pose_matrix.copy()

    def clear_home_state(self) -> None:
        self.captured_home_qpos = {}
        for arm in self.arms.values():
            arm.home_qpos = None
            arm.home_pose_debug = None
            arm.home_pose_fine = None
        self.metadata.pop("home_capture", None)
        self.metadata.pop("home_ready", None)

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
        return self.operator_to_vive_retarget(pose_operator)

    def operator_to_vive_retarget(self, pose_operator: Any) -> np.ndarray:
        if self.T_O_D is None:
            raise RuntimeError("Runtime calibration missing T_O_VR (legacy T_O_D).")
        return invert_pose(self.T_O_D) @ as_pose_matrix(pose_operator)

    def world_to_desk(self, pose_world: Any, tracker_role: str | None = None) -> np.ndarray:
        return self.world_to_vive_retarget(pose_world, tracker_role=tracker_role)

    def world_to_vive_retarget(
        self, pose_world: Any, tracker_role: str | None = None
    ) -> np.ndarray:
        return self.operator_to_vive_retarget(
            self.world_to_operator(pose_world, tracker_role=tracker_role)
        )

    def desk_to_base(self, pose_desk: Any, side: str) -> np.ndarray:
        return self.vive_retarget_to_base(pose_desk, side=side)

    def vive_retarget_to_base(self, pose_vive_retarget: Any, side: str) -> np.ndarray:
        arm = self.get_arm_for_side(side)
        if arm.T_D_B is None:
            raise RuntimeError(
                f"Arm calibration for side '{side}' is missing T_VR_B (legacy T_D_B)."
            )
        return invert_pose(arm.T_D_B) @ as_pose_matrix(pose_vive_retarget)

    def world_to_base(self, pose_world: Any, side: str, tracker_role: str | None = None) -> np.ndarray:
        return self.vive_retarget_to_base(
            self.world_to_vive_retarget(pose_world, tracker_role=tracker_role),
            side=side,
        )

    def target_frame_name(self, side: str, tcp_mode: str | None = None) -> str:
        return self.get_arm_for_side(side).tcp_frame_name(tcp_mode)

    def clip_pose_to_workspace(self, pose_desk: Any, side: str) -> np.ndarray:
        arm = self.get_arm_for_side(side)
        return clip_pose_translation(pose_desk, arm.workspace_min, arm.workspace_max)

    def _apply_virtual_desk_post_rotation(self, yaw_deg: float, *, update_offsets: bool) -> None:
        current_world_to_desk = self.world_to_desk_pose()
        if current_world_to_desk is None:
            return

        post_rotation = _yaw_pose_deg(yaw_deg)
        self.set_world_to_desk(current_world_to_desk @ post_rotation)
        self.metadata["virtual_desk_post_yaw_deg"] = float(yaw_deg)

        virtual_desk_metadata = self.metadata.get("virtual_desk")
        if isinstance(virtual_desk_metadata, dict):
            updated_world_to_desk = self.world_to_desk_pose()
            if updated_world_to_desk is not None:
                virtual_desk_metadata["T_W_V"] = updated_world_to_desk.tolist()
                rotation = updated_world_to_desk[:3, :3]
                virtual_desk_metadata["x_axis_world"] = rotation[:, 0].tolist()
                virtual_desk_metadata["y_axis_world"] = rotation[:, 1].tolist()
                virtual_desk_metadata["z_axis_world"] = rotation[:, 2].tolist()

        if not update_offsets:
            return

        tracker_metadata = self.metadata.get("tracker_offset_calibration")
        for tracker_role, current_offset in tuple(self.tracker_offsets.items()):
            updated_offset = as_pose_matrix(current_offset) @ post_rotation
            self.tracker_offsets[str(tracker_role)] = updated_offset
            if not isinstance(tracker_metadata, dict):
                continue
            tracker_info = tracker_metadata.get(str(tracker_role))
            if not isinstance(tracker_info, dict):
                continue
            if tracker_info.get("coarse_offset") is not None:
                tracker_info["coarse_offset"] = updated_offset.tolist()
            if tracker_info.get("final_offset") is not None:
                tracker_info["final_offset"] = updated_offset.tolist()
            if tracker_info.get("fine_tune_rpy_deg") is not None:
                tracker_info["fine_tune_rpy_deg"] = [0.0, 0.0, 0.0]

    def ensure_virtual_desk_scene_alignment(self) -> bool:
        calibration_mode = str(self.metadata.get("calibration_mode", ""))
        if calibration_mode != "vive_rm65_virtual_desk":
            return False

        explicit_post_yaw = self.metadata.get("virtual_desk_post_yaw_deg")
        if explicit_post_yaw is None:
            self._apply_virtual_desk_post_rotation(
                DEFAULT_VIRTUAL_DESK_POST_YAW_DEG,
                update_offsets=True,
            )
            self.metadata["virtual_desk_scene_alignment_defaulted"] = True

        self.metadata["virtual_desk_sweep_convention_version"] = (
            VIRTUAL_DESK_SWEEP_CONVENTION_VERSION
        )
        return explicit_post_yaw is None

    def to_dict(self) -> dict[str, Any]:
        vive_retarget_payload = {
            "name": self.vive_retarget_frame.name,
            "axes": dict(self.vive_retarget_frame.axes),
        }
        return {
            "vive_retarget_frame": vive_retarget_payload,
            "desk_frame": {
                "name": self.desk_frame.name,
                "axes": dict(self.desk_frame.axes),
            },
            "arms": {
                name: {
                    "name": arm.name,
                    "side": arm.side,
                    "T_VR_B": _to_serializable(arm.T_VR_B),
                    "T_D_B": _to_serializable(arm.T_D_B),
                    "workspace_min": _to_serializable(arm.workspace_min),
                    "workspace_max": _to_serializable(arm.workspace_max),
                    "home_qpos": _to_serializable(arm.home_qpos),
                    "home_pose": _to_serializable(arm.home_pose_for_tcp_mode()),
                    "tcp_frame": arm.tcp_frame_name(),
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
            "T_O_VR": _to_serializable(self.T_O_VR),
            "T_O_D": _to_serializable(self.T_O_D),
            "tracker_offsets": _to_serializable(self.tracker_offsets),
            "captured_home_qpos": _to_serializable(self.captured_home_qpos),
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeleopCalibration":
        _validate_transform_alias(
            payload, canonical_key="T_O_VR", legacy_key="T_O_D", context="calibration"
        )
        frame_payload = payload.get("vive_retarget_frame", payload.get("desk_frame", {}))
        arms: dict[str, ArmCalibration] = {}
        for name, arm_payload in payload.get("arms", {}).items():
            _validate_transform_alias(
                arm_payload,
                canonical_key="T_VR_B",
                legacy_key="T_D_B",
                context=f"arms.{name}",
            )
            home_pose = arm_payload.get("home_pose")
            if home_pose is None:
                home_pose = arm_payload.get("home_pose_fine")
            if home_pose is None:
                home_pose = arm_payload.get("home_pose_debug")
            tcp_frame = arm_payload.get("tcp_frame")
            if tcp_frame is None:
                tcp_frame = arm_payload.get("tcp_fine_frame")
            if tcp_frame is None:
                tcp_frame = arm_payload.get("tcp_debug_frame")
            if tcp_frame is None:
                tcp_frame = TOOL_TCP_FRAME
            arms[name] = ArmCalibration(
                name=arm_payload["name"],
                side=arm_payload["side"],
                T_D_B=arm_payload.get("T_VR_B", arm_payload.get("T_D_B")),
                workspace_min=arm_payload.get("workspace_min"),
                workspace_max=arm_payload.get("workspace_max"),
                home_qpos=arm_payload.get("home_qpos"),
                home_pose_debug=home_pose,
                home_pose_fine=home_pose,
                tcp_debug_frame=canonical_tcp_frame(tcp_frame),
                tcp_fine_frame=canonical_tcp_frame(tcp_frame),
                metadata=arm_payload.get("metadata", {}),
            )
        hands = {
            name: HandCalibration(**hand_payload)
            for name, hand_payload in payload.get("hands", {}).items()
        }
        calibration = cls(
            desk_frame=ViveRetargetFrameConfig(
                name=(
                    "vive_retarget_frame"
                    if frame_payload.get("name", "vive_retarget_frame") == "desk"
                    else frame_payload.get("name", "vive_retarget_frame")
                ),
                axes=frame_payload.get(
                    "axes", {"x": "forward", "y": "left", "z": "up"}
                ),
            ),
            arms=arms,
            hands=hands,
            T_W_O=payload.get("T_W_O"),
            T_O_D=payload.get("T_O_VR", payload.get("T_O_D")),
            tracker_offsets=payload.get("tracker_offsets", {}),
            captured_home_qpos=payload.get("captured_home_qpos", {}),
            metadata=payload.get("metadata", {}),
        )
        calibration.ensure_virtual_desk_scene_alignment()
        return calibration

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "TeleopCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
