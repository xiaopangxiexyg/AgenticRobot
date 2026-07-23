from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.runtime import EffectorCommand

from .schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
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
    deploy_arm_specs_metadata,
    deploy_hand_specs_metadata,
    nan_vector,
    normalize_active_sides,
    pose_matrix_to_xyzquat,
    to_serializable,
    vector_or_none,
)


DEFAULT_CAMERA_FRAME_MAX_AGE_SEC = 0.1
DEFAULT_HAND_STATE_MAX_AGE_SEC = 0.2


def normalize_end_effector_sides(
    *,
    active_sides: Sequence[str],
    hands_enabled: bool,
    end_effector_sides: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve configured end-effector sides without changing legacy defaults."""

    active = normalize_active_sides(active_sides)
    if not hands_enabled:
        if end_effector_sides:
            raise ValueError("end_effector_sides requires hands_enabled=True.")
        return ()
    requested = active if end_effector_sides is None else tuple(str(side) for side in end_effector_sides)
    invalid = set(requested) - set(active)
    if invalid:
        raise ValueError(
            "end_effector_sides must be a subset of active_sides, got "
            f"{sorted(invalid)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("end_effector_sides must not contain duplicates.")
    return tuple(side for side in ALL_SIDES if side in set(requested))


@dataclass(frozen=True)
class DeploymentObservation:
    """Policy-time observation shared by sim and real deployment adapters.

    Vector layouts match the collection contract:
    action = [left_arm_pose7, right_arm_pose7, left_hand6, right_hand6]
    state = [left_qpos6, left_pose7, left_hand6, right_qpos6, right_pose7, right_hand6]
    """

    timestamp: float
    step_index: int
    active_sides: tuple[str, ...]
    arms_enabled: bool
    hands_enabled: bool
    end_effector_sides: tuple[str, ...] | None = None
    images: dict[str, np.ndarray] = field(default_factory=dict)
    image_timestamps: dict[str, float | None] = field(default_factory=dict)
    image_ages_sec: dict[str, float | None] = field(default_factory=dict)
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_sides", normalize_active_sides(self.active_sides))
        object.__setattr__(
            self,
            "end_effector_sides",
            normalize_end_effector_sides(
                active_sides=self.active_sides,
                hands_enabled=bool(self.hands_enabled),
                end_effector_sides=self.end_effector_sides,
            ),
        )

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
            "image_ages_sec": self.image_ages_sec,
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
            "end_effector_sides": tuple(self.end_effector_sides or ()),
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
            if side in set(self.end_effector_sides or ()):
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
    end_effector_sides: Sequence[str] | None = None,
    arm_states: Mapping[str, Mapping[str, Any] | None],
    hand_states: Mapping[str, Mapping[str, Any] | None],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    state = nan_vector(OBSERVATION_STATE_DIM)
    valid = np.zeros(OBSERVATION_STATE_DIM, dtype=bool)
    state_dict: dict[str, Any] = {}

    effective_end_effector_sides = set(
        normalize_end_effector_sides(
            active_sides=active_sides,
            hands_enabled=bool(hands_enabled),
            end_effector_sides=end_effector_sides,
        )
    )
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
            if side in effective_end_effector_sides
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
    end_effector_sides: Sequence[str] | None = None,
    hand_states: Mapping[str, Mapping[str, Any] | None],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, bool]]:
    tactile_by_side: dict[str, np.ndarray] = {}
    tactile_valid_by_side: dict[str, bool] = {}
    concatenated: list[np.ndarray] = []
    effective_end_effector_sides = set(
        normalize_end_effector_sides(
            active_sides=active_sides,
            hands_enabled=bool(hands_enabled),
            end_effector_sides=end_effector_sides,
        )
    )
    for side in HAND_VECTOR_ORDER:
        if side not in effective_end_effector_sides:
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


def _latest_camera_frames_with_availability(
    camera_sources: Mapping[str, Any],
    *,
    cutoff_timestamp: float | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float | None],
    dict[str, float | None],
]:
    images: dict[str, np.ndarray] = {}
    timestamps: dict[str, float | None] = {}
    available_timestamps: dict[str, float | None] = {}
    for camera_key, camera_source in sorted(camera_sources.items()):
        key = str(camera_key)
        frame = None
        causal_lookup_attempted = False
        if cutoff_timestamp is not None:
            frame_buffer = getattr(camera_source, "frame_buffer", None)
            if callable(frame_buffer):
                buffer = frame_buffer()
                lookup = getattr(buffer, "get_latest_before_or_at", None)
                if callable(lookup):
                    causal_lookup_attempted = True
                    frame = lookup(float(cutoff_timestamp))
        if cutoff_timestamp is None and frame is None and not causal_lookup_attempted:
            frame = camera_source.get_latest_frame()
        if frame is None:
            timestamps[key] = None
            available_timestamps[key] = None
            continue
        try:
            source_timestamp = float(frame.source_timestamp)
            available_timestamp = float(frame.available_timestamp)
        except (TypeError, ValueError):
            source_timestamp = float("nan")
            available_timestamp = float("nan")
        if not np.isfinite(source_timestamp) or not np.isfinite(available_timestamp):
            timestamps[key] = None
            available_timestamps[key] = None
            continue
        images[key] = np.asarray(frame.image).copy()
        timestamps[key] = source_timestamp
        available_timestamps[key] = available_timestamp
    return images, timestamps, available_timestamps


def latest_camera_frames(
    camera_sources: Mapping[str, Any],
    *,
    cutoff_timestamp: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | None]]:
    """Compatibility view; causal availability is retained by the observation builder."""

    images, timestamps, _available_timestamps = _latest_camera_frames_with_availability(
        camera_sources,
        cutoff_timestamp=cutoff_timestamp,
    )
    return images, timestamps


def observation_faults(
    *,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
    end_effector_sides: Sequence[str] | None = None,
    state_valid: np.ndarray,
    image_timestamps: Mapping[str, float | None],
    require_images: bool,
    arm_states: Mapping[str, Mapping[str, Any] | None],
    hand_states: Mapping[str, Mapping[str, Any] | None],
    observation_timestamp: float | None = None,
    max_image_age_sec: float | None = DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
    max_hand_state_age_sec: float | None = None,
) -> list[str]:
    faults: list[str] = []
    effective_end_effector_sides = set(
        normalize_end_effector_sides(
            active_sides=active_sides,
            hands_enabled=bool(hands_enabled),
            end_effector_sides=end_effector_sides,
        )
    )
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
            send_code = state.get("send_result_code")
            if send_code not in (None, 0):
                faults.append(f"{side}_arm_send:{send_code}")
        if side in effective_end_effector_sides:
            if not np.all(state_valid[STATE_HAND_SLICES[side]]):
                faults.append(f"{side}_hand_state_unavailable")
            hand_state = hand_states.get(side) or {}
            error = hand_state.get("telemetry_error")
            if error:
                faults.append(f"{side}_hand_telemetry:{error}")
            if observation_timestamp is not None and max_hand_state_age_sec is not None:
                joint_timestamp = hand_state.get("joint_timestamp")
                try:
                    joint_timestamp = float(joint_timestamp)
                except (TypeError, ValueError):
                    joint_timestamp = float("nan")
                if not np.isfinite(joint_timestamp):
                    faults.append(f"{side}_hand_state_timestamp_unavailable")
                else:
                    age_sec = float(observation_timestamp) - joint_timestamp
                    max_age = float(max_hand_state_age_sec)
                    if age_sec > max_age:
                        faults.append(f"{side}_hand_state_stale:{age_sec:.6f}")
                    elif age_sec < -max_age:
                        faults.append(f"{side}_hand_state_timestamp_in_future:{age_sec:.6f}")
    if require_images:
        if not image_timestamps:
            faults.append("camera_frame_unavailable:no_configured_camera")
        missing = [key for key, ts in image_timestamps.items() if ts is None]
        if missing:
            faults.append("camera_frame_unavailable:" + ",".join(sorted(missing)))
        if observation_timestamp is not None and max_image_age_sec is not None:
            now = float(observation_timestamp)
            max_age = float(max_image_age_sec)
            stale: list[str] = []
            future: list[str] = []
            for key, value in image_timestamps.items():
                if value is None:
                    continue
                age_sec = float(now - float(value))
                if age_sec > max_age:
                    stale.append(f"{key}:{age_sec:.6f}")
                elif age_sec < -max_age:
                    future.append(f"{key}:{age_sec:.6f}")
            if stale:
                faults.append("camera_frame_stale:" + ",".join(sorted(stale)))
            if future:
                faults.append("camera_frame_timestamp_in_future:" + ",".join(sorted(future)))
    return faults


def build_deployment_observation(
    *,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
    end_effector_sides: Sequence[str] | None = None,
    step_index: int,
    arm_states: Mapping[str, Mapping[str, Any] | None],
    hand_states: Mapping[str, Mapping[str, Any] | None],
    camera_sources: Mapping[str, Any] | None = None,
    last_action: np.ndarray | None = None,
    timestamp: float | None = None,
    require_images: bool = False,
    camera_frame_max_age_sec: float | None = DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
    hand_state_max_age_sec: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DeploymentObservation:
    active = normalize_active_sides(active_sides)
    effective_end_effector_sides = normalize_end_effector_sides(
        active_sides=active,
        hands_enabled=bool(hands_enabled),
        end_effector_sides=end_effector_sides,
    )
    now = time.time() if timestamp is None else float(timestamp)
    state, state_dict, state_valid = build_state_vector(
        active_sides=active,
        arms_enabled=bool(arms_enabled),
        hands_enabled=bool(hands_enabled),
        end_effector_sides=effective_end_effector_sides,
        arm_states=arm_states,
        hand_states=hand_states,
    )
    tactile, tactile_by_side, tactile_valid_by_side = build_tactile_vectors(
        active_sides=active,
        hands_enabled=bool(hands_enabled),
        end_effector_sides=effective_end_effector_sides,
        hand_states=hand_states,
    )
    images, image_timestamps, image_available_timestamps = (
        _latest_camera_frames_with_availability(
            camera_sources or {}, cutoff_timestamp=now
        )
    )
    for camera_key in (camera_sources or {}):
        image_timestamps.setdefault(str(camera_key), None)
        image_available_timestamps.setdefault(str(camera_key), None)
    if camera_frame_max_age_sec is not None:
        camera_frame_max_age_sec = float(camera_frame_max_age_sec)
        if camera_frame_max_age_sec <= 0.0:
            raise ValueError("camera_frame_max_age_sec must be positive or None.")
    if hand_state_max_age_sec is not None:
        hand_state_max_age_sec = float(hand_state_max_age_sec)
        if hand_state_max_age_sec <= 0.0:
            raise ValueError("hand_state_max_age_sec must be positive or None.")
    image_ages_sec = {
        key: None if value is None else float(now - float(value))
        for key, value in image_timestamps.items()
    }
    hand_state_ages_sec: dict[str, float | None] = {}
    for side in effective_end_effector_sides:
        try:
            joint_timestamp = float((hand_states.get(side) or {}).get("joint_timestamp"))
        except (TypeError, ValueError):
            joint_timestamp = float("nan")
        hand_state_ages_sec[side] = (
            float(now - joint_timestamp) if np.isfinite(joint_timestamp) else None
        )
    faults = observation_faults(
        active_sides=active,
        arms_enabled=bool(arms_enabled),
        hands_enabled=bool(hands_enabled),
        end_effector_sides=effective_end_effector_sides,
        state_valid=state_valid,
        image_timestamps=image_timestamps,
        require_images=bool(require_images),
        arm_states=arm_states,
        hand_states=hand_states,
        observation_timestamp=now,
        max_image_age_sec=camera_frame_max_age_sec,
        max_hand_state_age_sec=hand_state_max_age_sec,
    )
    last = nan_vector(ACTION_DIM) if last_action is None else np.asarray(last_action, dtype=np.float32).reshape(-1)
    if last.shape != (ACTION_DIM,):
        last = nan_vector(ACTION_DIM)
    observation_metadata = {} if metadata is None else dict(to_serializable(dict(metadata)))
    observation_metadata.setdefault("arm_action_specs_by_side", deploy_arm_specs_metadata())
    observation_metadata.setdefault(
        "arm_controlled_frame_by_side",
        {
            side: str(spec.get("controlled_frame", ""))
            for side, spec in observation_metadata["arm_action_specs_by_side"].items()
        },
    )
    observation_metadata.setdefault(
        "arm_reference_frame_by_side",
        {
            side: str(spec.get("reference_frame", ""))
            for side, spec in observation_metadata["arm_action_specs_by_side"].items()
        },
    )
    observation_metadata.setdefault("arm_spec_required", True)
    observation_metadata.setdefault("hand_action_specs_by_side", deploy_hand_specs_metadata())
    observation_metadata.setdefault(
        "hand_active_mask_by_side",
        {
            side: list(spec.get("active_mask", ()))
            for side, spec in observation_metadata["hand_action_specs_by_side"].items()
        },
    )
    observation_metadata.setdefault("hand_spec_required", True)
    observation_metadata.setdefault("end_effector_enabled_sides", list(effective_end_effector_sides))
    observation_metadata.setdefault("camera_frame_max_age_sec", camera_frame_max_age_sec)
    observation_metadata.setdefault("hand_state_max_age_sec", hand_state_max_age_sec)
    observation_metadata.setdefault(
        "hand_state_age_sec_by_side",
        dict(hand_state_ages_sec),
    )
    observation_metadata.setdefault("camera_frame_age_sec_by_key", dict(image_ages_sec))
    observation_metadata.setdefault(
        "camera_frame_source_timestamp_by_key", dict(image_timestamps)
    )
    observation_metadata.setdefault(
        "camera_frame_available_timestamp_by_key", dict(image_available_timestamps)
    )
    observation_metadata.setdefault(
        "camera_frame_available_age_sec_by_key",
        {
            key: None if value is None else float(now - float(value))
            for key, value in image_available_timestamps.items()
        },
    )
    observation_metadata.setdefault(
        "camera_causal_snapshot_contract",
        "source_timestamp<=T0_and_available_timestamp<=T0",
    )
    return DeploymentObservation(
        timestamp=now,
        step_index=int(step_index),
        active_sides=active,
        arms_enabled=bool(arms_enabled),
        hands_enabled=bool(hands_enabled),
        end_effector_sides=effective_end_effector_sides,
        images=images,
        image_timestamps=image_timestamps,
        image_ages_sec=image_ages_sec,
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
        metadata=observation_metadata,
    )


__all__ = [
    "DEFAULT_CAMERA_FRAME_MAX_AGE_SEC",
    "DEFAULT_HAND_STATE_MAX_AGE_SEC",
    "DeploymentObservation",
    "build_deployment_observation",
    "build_state_vector",
    "build_tactile_vectors",
    "latest_camera_frames",
    "normalize_end_effector_sides",
    "observation_faults",
]
