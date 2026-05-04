from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..schemas import EffectorCommand
from .schema import (
    ACTION_DIM,
    ACTION_ARM_POSE_SLICES,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    ARM_JOINT_DOF,
    ARM_POSE_DOF,
    HAND_DOF,
    HAND_VECTOR_ORDER,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
    STATE_HAND_SLICES,
    nan_vector,
    normalize_active_sides,
    pose_matrix_to_xyzquat,
    to_serializable,
    vector_or_none,
)


@dataclass(frozen=True)
class DeploymentObservation:
    """Policy-time observation for teleop_runtime deployment.

    Vector layouts match teleop_runtime collection:
    action = [left_arm_pose7, right_arm_pose7, left_hand6, right_hand6]
    state = [left_qpos6, left_pose7, left_hand6, right_qpos6, right_pose7, right_hand6]
    """

    timestamp: float
    step_index: int
    active_sides: tuple[str, ...]
    arms_enabled: bool
    hands_enabled: bool
    images: dict[str, np.ndarray] = field(default_factory=dict)
    image_timestamps: dict[str, float | None] = field(default_factory=dict)
    state: np.ndarray = field(default_factory=lambda: nan_vector(OBSERVATION_STATE_DIM))
    state_dict: dict[str, Any] = field(default_factory=dict)
    state_valid: np.ndarray = field(default_factory=lambda: np.zeros(OBSERVATION_STATE_DIM, dtype=bool))
    tactile: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    tactile_by_side: dict[str, np.ndarray] = field(default_factory=dict)
    tactile_valid_by_side: dict[str, bool] = field(default_factory=dict)
    last_action: np.ndarray = field(default_factory=lambda: nan_vector(ACTION_DIM))
    arm_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    hand_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image(self) -> np.ndarray | dict[str, np.ndarray] | None:
        if not self.images:
            return None
        if len(self.images) == 1:
            return next(iter(self.images.values()))
        return self.images

    def model_input(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "images": self.images,
            "image_timestamps": self.image_timestamps,
            "state": self.state,
            "state_dict": self.state_dict,
            "state_valid": self.state_valid,
            "tactile": self.tactile,
            "tactile_by_side": self.tactile_by_side,
            "tactile_valid_by_side": self.tactile_valid_by_side,
            "last_action": self.last_action,
            "timestamp": float(self.timestamp),
            "step_index": int(self.step_index),
            "active_sides": tuple(self.active_sides),
            "arms_enabled": bool(self.arms_enabled),
            "hands_enabled": bool(self.hands_enabled),
            "faults": list(self.faults),
            "metadata": dict(self.metadata),
        }

    def hold_action(self) -> np.ndarray:
        action = nan_vector(ACTION_DIM)
        last = np.asarray(self.last_action, dtype=np.float32).reshape(-1)
        if last.shape != (ACTION_DIM,):
            last = nan_vector(ACTION_DIM)
        for side in ALL_SIDES:
            pose = vector_or_none(
                self.state[STATE_ARM_POSE_SLICES[side]],
                length=ARM_POSE_DOF,
            )
            if pose is not None:
                action[ACTION_ARM_POSE_SLICES[side]] = pose
            hand = vector_or_none(last[ACTION_HAND_SLICES[side]], length=HAND_DOF)
            if hand is None:
                hand = vector_or_none(self.state[STATE_HAND_SLICES[side]], length=HAND_DOF)
            if hand is not None:
                action[ACTION_HAND_SLICES[side]] = hand
        missing = ~np.isfinite(action)
        action[missing] = last[missing]
        return action.astype(np.float32, copy=False)


def _command_pose(command: Any) -> np.ndarray | None:
    if not isinstance(command, EffectorCommand):
        return None
    if command.command_type == "cartesian_pose":
        return pose_matrix_to_xyzquat(command.value)
    pose = command.metadata.get("target_pose_base") or command.metadata.get("source_target_pose_base")
    return pose_matrix_to_xyzquat(pose)


def _arm_pose_from_state(state: Mapping[str, Any]) -> np.ndarray | None:
    pose = pose_matrix_to_xyzquat(state.get("current_tcp_pose"))
    if pose is not None:
        return pose
    pose = pose_matrix_to_xyzquat(state.get("target_pose_base"))
    if pose is not None:
        return pose
    return _command_pose(state.get("last_command"))


def build_state_vector(
    *,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
    arm_states: Mapping[str, Mapping[str, Any] | None],
    hand_states: Mapping[str, Mapping[str, Any] | None],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    state = nan_vector(OBSERVATION_STATE_DIM)
    valid = np.zeros(OBSERVATION_STATE_DIM, dtype=bool)
    state_dict: dict[str, Any] = {}

    for side in ALL_SIDES:
        side_active = side in active_sides
        arm_state = dict(arm_states.get(side) or {})
        hand_state = dict(hand_states.get(side) or {})

        qpos = (
            vector_or_none(arm_state.get("current_qpos"), length=ARM_JOINT_DOF)
            if arms_enabled and side_active
            else None
        )
        pose = _arm_pose_from_state(arm_state) if arms_enabled and side_active else None
        hand = (
            vector_or_none(hand_state.get("joint_pos"), length=HAND_DOF)
            if hands_enabled and side_active
            else None
        )

        if qpos is not None:
            state[STATE_ARM_QPOS_SLICES[side]] = qpos
            valid[STATE_ARM_QPOS_SLICES[side]] = True
        if pose is not None:
            state[STATE_ARM_POSE_SLICES[side]] = pose
            valid[STATE_ARM_POSE_SLICES[side]] = True
        if hand is not None:
            state[STATE_HAND_SLICES[side]] = hand
            valid[STATE_HAND_SLICES[side]] = True

        state_dict[f"{side}_arm"] = {
            "qpos": state[STATE_ARM_QPOS_SLICES[side]].copy(),
            "pose": state[STATE_ARM_POSE_SLICES[side]].copy(),
        }
        state_dict[f"{side}_arm.qpos"] = state[STATE_ARM_QPOS_SLICES[side]].copy()
        state_dict[f"{side}_arm.pose"] = state[STATE_ARM_POSE_SLICES[side]].copy()
        state_dict[f"{side}_hand"] = state[STATE_HAND_SLICES[side]].copy()

    return state.astype(np.float32, copy=False), state_dict, valid


def build_tactile_vectors(
    *,
    active_sides: tuple[str, ...],
    hands_enabled: bool,
    hand_states: Mapping[str, Mapping[str, Any] | None],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, bool]]:
    tactile_by_side: dict[str, np.ndarray] = {}
    tactile_valid_by_side: dict[str, bool] = {}
    concatenated: list[np.ndarray] = []
    for side in HAND_VECTOR_ORDER:
        if not hands_enabled or side not in active_sides:
            tactile_valid_by_side[side] = False
            tactile_by_side[side] = np.zeros((0,), dtype=np.float32)
            continue
        raw = (hand_states.get(side) or {}).get("tactile")
        if raw is None:
            tactile_valid_by_side[side] = False
            tactile_by_side[side] = np.zeros((0,), dtype=np.float32)
            continue
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        valid = bool(vector.size > 0 and np.all(np.isfinite(vector)))
        tactile_by_side[side] = vector.copy()
        tactile_valid_by_side[side] = valid
        concatenated.append(vector)
    tactile = (
        np.concatenate(concatenated).astype(np.float32, copy=False)
        if concatenated
        else np.zeros((0,), dtype=np.float32)
    )
    return tactile, tactile_by_side, tactile_valid_by_side


def latest_camera_frames(camera_sources: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, float | None]]:
    images: dict[str, np.ndarray] = {}
    timestamps: dict[str, float | None] = {}
    for camera_key, camera_source in sorted(camera_sources.items()):
        key = str(camera_key)
        frame = camera_source.get_latest_frame()
        if frame is None:
            timestamps[key] = None
            continue
        images[key] = np.asarray(frame.image).copy()
        timestamps[key] = float(frame.source_timestamp)
    return images, timestamps


def observation_faults(
    *,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
    state_valid: np.ndarray,
    image_timestamps: Mapping[str, float | None],
    require_images: bool,
    arm_states: Mapping[str, Mapping[str, Any] | None],
    hand_states: Mapping[str, Mapping[str, Any] | None],
) -> list[str]:
    faults: list[str] = []
    for side in active_sides:
        if arms_enabled:
            if not np.all(state_valid[STATE_ARM_QPOS_SLICES[side]]):
                faults.append(f"{side}_arm_qpos_unavailable")
            if not np.all(state_valid[STATE_ARM_POSE_SLICES[side]]):
                faults.append(f"{side}_arm_pose_unavailable")
            state = arm_states.get(side) or {}
            result_code = state.get("hardware_result_code")
            if result_code not in (None, 0, 9):
                faults.append(f"{side}_arm_state:{result_code}")
        if hands_enabled:
            if not np.all(state_valid[STATE_HAND_SLICES[side]]):
                faults.append(f"{side}_hand_state_unavailable")
            error = (hand_states.get(side) or {}).get("telemetry_error")
            if error:
                faults.append(f"{side}_hand_telemetry:{error}")
    if require_images:
        missing = [key for key, ts in image_timestamps.items() if ts is None]
        if missing:
            faults.append("camera_frame_unavailable:" + ",".join(sorted(missing)))
    return faults


def build_deployment_observation(
    *,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
    step_index: int,
    arm_states: Mapping[str, Mapping[str, Any] | None],
    hand_states: Mapping[str, Mapping[str, Any] | None],
    camera_sources: Mapping[str, Any] | None = None,
    last_action: np.ndarray | None = None,
    timestamp: float | None = None,
    require_images: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> DeploymentObservation:
    active = normalize_active_sides(active_sides)
    now = time.time() if timestamp is None else float(timestamp)
    state, state_dict, state_valid = build_state_vector(
        active_sides=active,
        arms_enabled=bool(arms_enabled),
        hands_enabled=bool(hands_enabled),
        arm_states=arm_states,
        hand_states=hand_states,
    )
    tactile, tactile_by_side, tactile_valid_by_side = build_tactile_vectors(
        active_sides=active,
        hands_enabled=bool(hands_enabled),
        hand_states=hand_states,
    )
    images, image_timestamps = latest_camera_frames(camera_sources or {})
    for camera_key in (camera_sources or {}):
        image_timestamps.setdefault(str(camera_key), None)
    faults = observation_faults(
        active_sides=active,
        arms_enabled=bool(arms_enabled),
        hands_enabled=bool(hands_enabled),
        state_valid=state_valid,
        image_timestamps=image_timestamps,
        require_images=bool(require_images),
        arm_states=arm_states,
        hand_states=hand_states,
    )
    last = nan_vector(ACTION_DIM) if last_action is None else np.asarray(last_action, dtype=np.float32).reshape(-1)
    if last.shape != (ACTION_DIM,):
        last = nan_vector(ACTION_DIM)
    return DeploymentObservation(
        timestamp=now,
        step_index=int(step_index),
        active_sides=active,
        arms_enabled=bool(arms_enabled),
        hands_enabled=bool(hands_enabled),
        images=images,
        image_timestamps=image_timestamps,
        state=state,
        state_dict=state_dict,
        state_valid=state_valid,
        tactile=tactile,
        tactile_by_side=tactile_by_side,
        tactile_valid_by_side=tactile_valid_by_side,
        last_action=last.astype(np.float32, copy=True),
        arm_states={str(side): dict(to_serializable(dict(state or {}))) for side, state in arm_states.items()},
        hand_states={str(side): dict(to_serializable(dict(state or {}))) for side, state in hand_states.items()},
        faults=faults,
        metadata={} if metadata is None else dict(to_serializable(dict(metadata))),
    )
