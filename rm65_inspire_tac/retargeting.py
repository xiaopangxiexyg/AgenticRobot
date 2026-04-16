from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
from scipy.spatial.transform import Rotation as R

from .schemas import (
    BimanualRetargetingResult,
    HandMotionInput,
    HandRetargetingResult,
    TeleopFrame,
)

HAND_LANDMARK_SELECTION = np.array(
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
)

OPERATOR2AVP_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=float,
)
OPERATOR2AVP_LEFT = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ],
    dtype=float,
)


class MotionSource(Protocol):
    def get_latest(self) -> Any:
        ...


class OfflineVisionProSource:
    def __init__(self, path: str | Path):
        from teleop.TeleVision import OfflineAVPStreamerDebug

        self._streamer = OfflineAVPStreamerDebug(path=str(path))

    def get_latest(self) -> Any:
        return self._streamer.get_latest()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._streamer, name)


class VisionProSource:
    def __init__(self, ip: str, record: bool = False):
        from avp_stream import VisionProStreamer

        self._streamer = VisionProStreamer(ip, record=record)

    def get_latest(self) -> Any:
        return self._streamer.get_latest()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._streamer, name)


class OpenTeleVisionSource:
    def __init__(self, resolution: tuple[int, int] = (720, 1280)):
        from teleop.TeleVision import OpenTeleVisionStreamer

        self._streamer = OpenTeleVisionStreamer(resolution=resolution)

    def get_latest(self) -> Any:
        return self._streamer.get_latest()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._streamer, name)


@dataclass
class _SideRuntime:
    side: str
    hand_type: Any
    operator_to_avp: np.ndarray
    world_to_base: np.ndarray
    base_to_world: np.ndarray
    retargeting: Any
    hand_joint_names: list[str]
    hand_active_joint_names: list[str]


def _two_mat_batch_mul(batch_mat: np.ndarray, left_rot: np.ndarray) -> np.ndarray:
    result = np.tile(np.eye(4), [batch_mat.shape[0], 1, 1])
    result[:, :3, :3] = np.matmul(left_rot[None, ...], batch_mat[:, :3, :3])
    result[:, :3, 3] = batch_mat[:, :3, 3] @ left_rot.T
    return result


def _joint_avp2hand(data: np.ndarray) -> np.ndarray:
    if data.shape[0] == 21:
        return data
    if data.shape[0] != 25:
        raise ValueError(f"Expected 21 or 25 landmarks, got shape {data.shape}.")
    return data[HAND_LANDMARK_SELECTION]


def _build_base_transforms(
    side: str, left_right_base_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    if side == "right":
        start_pos = np.array([left_right_base_distance / 2.0, 0.0, 0.6], dtype=float)
        start_rot = R.from_euler("xyz", [0.0, 0.0, 0.0]).as_matrix()
    else:
        start_pos = np.array([-left_right_base_distance / 2.0, 0.0, 0.6], dtype=float)
        start_rot = R.from_euler("xyz", [0.0, 0.0, -np.pi]).as_matrix()

    base_to_world = np.eye(4, dtype=float)
    base_to_world[:3, :3] = start_rot
    base_to_world[:3, 3] = start_pos

    world_to_base = np.eye(4, dtype=float)
    world_to_base[:3, :3] = start_rot.T
    world_to_base[:3, 3] = -start_rot.T @ start_pos
    return world_to_base, base_to_world


class RM65InspireTacRetargeter:
    def __init__(
        self,
        mode: str = "right",
        source: MotionSource | None = None,
        avp_ip: str | None = None,
        offline_path: str | Path | None = None,
        use_television: bool = False,
        television_resolution: tuple[int, int] = (720, 1280),
        hand_name: Any = None,
        retargeting_type: Any = None,
        left_right_base_distance: float = 0.525,
        safe_workspace: list[list[float]] | np.ndarray | None = None,
    ):
        if mode not in {"left", "right", "both"}:
            raise ValueError(f"Unsupported mode: {mode}")

        try:
            from dex_retargeting.constants import (
                HandType,
                RobotName,
                RetargetingType,
                get_default_config_path,
            )
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RM65InspireTacRetargeter requires dex_retargeting to be importable."
            ) from exc

        self._HandType = HandType
        self._get_default_config_path = get_default_config_path
        self._RetargetingConfig = RetargetingConfig

        if hand_name is None:
            hand_name = RobotName.inspiretacofficial
        if retargeting_type is None:
            retargeting_type = RetargetingType.dexpilot

        self.mode = mode
        self.use_left_hand = mode in {"left", "both"}
        self.use_right_hand = mode in {"right", "both"}
        self.use_television = use_television
        self.safe_workspace = (
            np.asarray(safe_workspace, dtype=float) if safe_workspace is not None else None
        )

        self.source = self._build_source(
            source=source,
            avp_ip=avp_ip,
            offline_path=offline_path,
            use_television=use_television,
            television_resolution=television_resolution,
        )

        robot_dir = (
            Path(importlib.util.find_spec("dex_retargeting").origin).resolve().parent.parent
            / "assets"
            / "robots"
            / "hands"
        )
        self._RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        override = dict(add_dummy_free_joint=True)

        self._sides: dict[str, _SideRuntime] = {}
        if self.use_left_hand:
            self._sides["left"] = self._build_side_runtime(
                side="left",
                hand_type=self._HandType.left,
                hand_name=hand_name,
                retargeting_type=retargeting_type,
                override=override,
                left_right_base_distance=left_right_base_distance,
                operator_to_avp=OPERATOR2AVP_LEFT,
            )
        if self.use_right_hand:
            self._sides["right"] = self._build_side_runtime(
                side="right",
                hand_type=self._HandType.right,
                hand_name=hand_name,
                retargeting_type=retargeting_type,
                override=override,
                left_right_base_distance=left_right_base_distance,
                operator_to_avp=OPERATOR2AVP_RIGHT,
            )

        self.left_hand_joint_names = (
            self._sides["left"].hand_joint_names if "left" in self._sides else None
        )
        self.right_hand_joint_names = (
            self._sides["right"].hand_joint_names if "right" in self._sides else None
        )
        self.left_hand_active_joint_names = (
            self._sides["left"].hand_active_joint_names if "left" in self._sides else None
        )
        self.right_hand_active_joint_names = (
            self._sides["right"].hand_active_joint_names if "right" in self._sides else None
        )

    def _build_source(
        self,
        source: MotionSource | None,
        avp_ip: str | None,
        offline_path: str | Path | None,
        use_television: bool,
        television_resolution: tuple[int, int],
    ) -> MotionSource | None:
        if source is not None:
            return source
        if offline_path is not None:
            return OfflineVisionProSource(offline_path)
        if use_television:
            return OpenTeleVisionSource(television_resolution)
        if avp_ip is not None:
            return VisionProSource(avp_ip)
        return None

    def _build_side_runtime(
        self,
        side: str,
        hand_type: Any,
        hand_name: Any,
        retargeting_type: Any,
        override: dict[str, Any],
        left_right_base_distance: float,
        operator_to_avp: np.ndarray,
    ) -> _SideRuntime:
        config_path = self._get_default_config_path(hand_name, retargeting_type, hand_type)
        config = self._RetargetingConfig.load_from_file(config_path, override=override)
        retargeting = config.build()
        world_to_base, base_to_world = _build_base_transforms(
            side=side, left_right_base_distance=left_right_base_distance
        )
        return _SideRuntime(
            side=side,
            hand_type=hand_type,
            operator_to_avp=operator_to_avp,
            world_to_base=world_to_base,
            base_to_world=base_to_world,
            retargeting=retargeting,
            hand_joint_names=list(retargeting.optimizer.robot.dof_joint_names),
            hand_active_joint_names=list(retargeting.optimizer.target_joint_names),
        )

    def fetch(self) -> TeleopFrame:
        if self.source is None:
            raise RuntimeError("No motion source configured. Pass a source or a frame explicitly.")
        return self.normalize_frame(self.source.get_latest())

    def normalize_frame(self, frame: TeleopFrame | Mapping[str, Any]) -> TeleopFrame:
        if isinstance(frame, TeleopFrame):
            return frame
        if not isinstance(frame, Mapping):
            raise TypeError(f"Unsupported frame type: {type(frame)!r}")

        left = self._parse_side_from_mapping("left", frame)
        right = self._parse_side_from_mapping("right", frame)
        return TeleopFrame(
            left=left,
            right=right,
            timestamp=frame.get("timestamp"),
            raw_data=frame,
        )

    def _parse_side_from_mapping(
        self, side: str, frame: Mapping[str, Any]
    ) -> HandMotionInput | None:
        direct_value = frame.get(side)
        if isinstance(direct_value, HandMotionInput):
            return direct_value
        if isinstance(direct_value, Mapping):
            return HandMotionInput(**direct_value)

        wrist_pose = frame.get(f"{side}_wrist_pose", frame.get(f"{side}_wrist"))
        hand_keypoints = frame.get(f"{side}_hand_keypoints")
        finger_matrices = frame.get(f"{side}_finger_matrices")
        raw_fingers = frame.get(f"{side}_fingers")

        if finger_matrices is None and raw_fingers is not None:
            raw_fingers_array = np.asarray(raw_fingers)
            if raw_fingers_array.ndim == 3 and raw_fingers_array.shape[-2:] == (4, 4):
                finger_matrices = raw_fingers_array
            elif raw_fingers_array.ndim == 2 and raw_fingers_array.shape[-1] == 3:
                hand_keypoints = raw_fingers_array

        if wrist_pose is None and hand_keypoints is None and finger_matrices is None:
            return None

        return HandMotionInput(
            wrist_pose=wrist_pose,
            hand_keypoints=hand_keypoints,
            finger_matrices=finger_matrices,
            hand_keypoints_in_operator_frame=False,
        )

    def retarget(
        self,
        frame: TeleopFrame | Mapping[str, Any] | None = None,
        include_raw: bool = False,
        include_keypoints: bool = False,
    ) -> BimanualRetargetingResult:
        normalized_frame = self.fetch() if frame is None else self.normalize_frame(frame)
        left = self.retarget_hand("left", normalized_frame.left, include_keypoints=include_keypoints)
        right = self.retarget_hand(
            "right", normalized_frame.right, include_keypoints=include_keypoints
        )
        return BimanualRetargetingResult(
            left=left,
            right=right,
            timestamp=normalized_frame.timestamp,
            raw_frame=normalized_frame.raw_data if include_raw else None,
        )

    def retarget_frame(
        self,
        frame: TeleopFrame | Mapping[str, Any],
        include_raw: bool = False,
        include_keypoints: bool = False,
    ) -> BimanualRetargetingResult:
        return self.retarget(
            frame=frame,
            include_raw=include_raw,
            include_keypoints=include_keypoints,
        )

    def retarget_hand(
        self,
        side: str,
        hand_input: HandMotionInput | None,
        include_keypoints: bool = False,
    ) -> HandRetargetingResult | None:
        if hand_input is None or side not in self._sides:
            return None

        runtime = self._sides[side]
        hand_keypoints = self._resolve_hand_keypoints(runtime, hand_input)
        wrist_pose_operator = self._resolve_wrist_pose(runtime, hand_input)

        indices = runtime.retargeting.optimizer.target_link_human_indices
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = hand_keypoints[task_indices] - hand_keypoints[origin_indices]
        qpos = runtime.retargeting.retarget(ref_value)

        wrist_pose_robot = runtime.world_to_base @ wrist_pose_operator
        if self.safe_workspace is not None:
            wrist_pose_robot[:3, 3] = np.clip(
                wrist_pose_robot[:3, 3], self.safe_workspace[0], self.safe_workspace[1]
            )

        hardware_hand_qpos = self._map_hand_qpos_to_hardware(runtime, qpos)
        return HandRetargetingResult(
            side=side,
            wrist_pose_robot=wrist_pose_robot,
            retargeted_hand_qpos=qpos,
            hardware_hand_qpos=hardware_hand_qpos,
            hand_keypoints=hand_keypoints if include_keypoints else None,
            wrist_pose_operator=wrist_pose_operator if include_keypoints else None,
        )

    def _resolve_hand_keypoints(
        self, runtime: _SideRuntime, hand_input: HandMotionInput
    ) -> np.ndarray:
        if hand_input.hand_keypoints is not None:
            hand_keypoints = np.asarray(hand_input.hand_keypoints, dtype=float)
            if hand_keypoints.shape[-1] != 3:
                raise ValueError(
                    f"{runtime.side} hand_keypoints should end with dimension 3, got {hand_keypoints.shape}."
                )
            hand_keypoints = _joint_avp2hand(hand_keypoints)
            if hand_input.hand_keypoints_in_operator_frame:
                hand_keypoints = hand_keypoints @ runtime.operator_to_avp
            return hand_keypoints

        if hand_input.finger_matrices is None:
            raise ValueError(
                f"{runtime.side} hand input requires either hand_keypoints or finger_matrices."
            )

        finger_matrices = np.asarray(hand_input.finger_matrices, dtype=float)
        if finger_matrices.ndim != 3 or finger_matrices.shape[-2:] != (4, 4):
            raise ValueError(
                f"{runtime.side} finger_matrices should have shape (N, 4, 4), got {finger_matrices.shape}."
            )

        transformed = _two_mat_batch_mul(finger_matrices, runtime.operator_to_avp.T)
        return _joint_avp2hand(transformed)[:, :3, 3]

    def _resolve_wrist_pose(
        self, runtime: _SideRuntime, hand_input: HandMotionInput
    ) -> np.ndarray:
        if hand_input.wrist_pose is None:
            raise ValueError(f"{runtime.side} wrist pose is missing.")

        wrist_pose = np.asarray(hand_input.wrist_pose, dtype=float)
        if wrist_pose.shape == (1, 4, 4):
            wrist_pose = wrist_pose[0]
        if wrist_pose.shape != (4, 4):
            raise ValueError(
                f"{runtime.side} wrist pose should have shape (4, 4), got {wrist_pose.shape}."
            )
        wrist_pose = wrist_pose.copy()

        if self.use_television and self.source is not None and hasattr(
            self.source, "open_tv_pose_2_avp_pose"
        ):
            wrist_pose = self.source.open_tv_pose_2_avp_pose(wrist_pose)
        elif hand_input.wrist_in_operator_frame:
            wrist_pose[:3, :3] = wrist_pose[:3, :3] @ runtime.operator_to_avp
        return wrist_pose

    def _map_hand_qpos_to_hardware(
        self, runtime: _SideRuntime, qpos: np.ndarray
    ) -> np.ndarray:
        idx_pin2target = runtime.retargeting.optimizer.idx_pin2target
        joint_limits = runtime.retargeting.joint_limits
        hardware_hand_qpos = (qpos[idx_pin2target] - joint_limits[:, 0]) / (
            joint_limits[:, 1] - joint_limits[:, 0]
        )
        hardware_hand_qpos = np.clip(1.0 - hardware_hand_qpos, 0.0, 1.0)[6:12]
        return np.flip(hardware_hand_qpos)
