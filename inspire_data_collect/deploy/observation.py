from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..common import ALL_SIDES, nan_vector, to_serializable, vector_or_none
from ..utils.writer import ACTION_DIM, ARM_DOF, HAND_DOF, HAND_VECTOR_ORDER


@dataclass(frozen=True)
class DeploymentObservation:
    """Single policy-time observation assembled from live robot/camera state."""

    timestamp: float
    step_index: int
    active_sides: tuple[str, ...]
    arm_enabled: bool
    images: dict[str, np.ndarray] = field(default_factory=dict)
    image_timestamps: dict[str, float | None] = field(default_factory=dict)
    state: np.ndarray = field(default_factory=lambda: nan_vector(ACTION_DIM))
    state_dict: dict[str, np.ndarray] = field(default_factory=dict)
    state_valid: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_DIM, dtype=bool))
    tactile: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    tactile_by_side: dict[str, np.ndarray] = field(default_factory=dict)
    tactile_valid_by_side: dict[str, bool] = field(default_factory=dict)
    last_action: np.ndarray = field(default_factory=lambda: nan_vector(ACTION_DIM))
    hand_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    arm_state: dict[str, Any] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image(self) -> np.ndarray | dict[str, np.ndarray] | None:
        """Compatibility accessor for policies that expect `image`."""

        if not self.images:
            return None
        if len(self.images) == 1:
            return next(iter(self.images.values()))
        return self.images

    def model_input(self) -> dict[str, Any]:
        """Return the default dict passed to generic policy callables."""

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
            "arm_enabled": bool(self.arm_enabled),
            "faults": list(self.faults),
            "metadata": dict(self.metadata),
        }

    def hold_action(self) -> np.ndarray:
        """Best-effort absolute action that holds the latest observed state."""

        action = self.state.astype(np.float32, copy=True)
        last = np.asarray(self.last_action, dtype=np.float32).reshape(-1)
        if last.shape == (ACTION_DIM,):
            slices = _state_slices()
            for side in HAND_VECTOR_ORDER:
                target_slice = slices[f"{side}_hand"]
                if np.all(np.isfinite(last[target_slice])):
                    action[target_slice] = last[target_slice]
            missing = ~np.isfinite(action)
            action[missing] = last[missing]
        return action


def _vector_or_nan(value: Any, *, length: int) -> np.ndarray:
    vector = vector_or_none(value, length=length)
    return vector if vector is not None else nan_vector(length)


def _state_slices() -> dict[str, slice]:
    return {
        "arm": slice(0, ARM_DOF),
        "right_hand": slice(ARM_DOF, ARM_DOF + HAND_DOF),
        "left_hand": slice(ARM_DOF + HAND_DOF, ACTION_DIM),
    }


def build_state_vector(
    *,
    active_sides: tuple[str, ...],
    arm_enabled: bool,
    arm_state: Mapping[str, Any] | None,
    hand_states: Mapping[str, Mapping[str, Any] | None],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    state = nan_vector(ACTION_DIM)
    state_dict: dict[str, np.ndarray] = {}
    valid = np.zeros(ACTION_DIM, dtype=bool)
    slices = _state_slices()

    if arm_enabled:
        arm_qpos = _vector_or_nan((arm_state or {}).get("qpos"), length=ARM_DOF)
        state[slices["arm"]] = arm_qpos
        state_dict["arm"] = arm_qpos.copy()
        valid[slices["arm"]] = np.isfinite(arm_qpos)

    for side in HAND_VECTOR_ORDER:
        key = f"{side}_hand"
        if side not in active_sides:
            state_dict[key] = nan_vector(HAND_DOF)
            continue
        hand_state = hand_states.get(side) or {}
        qpos = _vector_or_nan(hand_state.get("joint_pos"), length=HAND_DOF)
        state[slices[key]] = qpos
        state_dict[key] = qpos.copy()
        valid[slices[key]] = np.isfinite(qpos)

    return state.astype(np.float32, copy=False), state_dict, valid


def build_tactile_vectors(
    *,
    active_sides: tuple[str, ...],
    hand_states: Mapping[str, Mapping[str, Any] | None],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, bool]]:
    tactile_by_side: dict[str, np.ndarray] = {}
    tactile_valid_by_side: dict[str, bool] = {}
    concatenated: list[np.ndarray] = []
    for side in HAND_VECTOR_ORDER:
        if side not in active_sides:
            tactile_valid_by_side[side] = False
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
        frame = camera_source.get_latest_frame()
        key = str(camera_key)
        if frame is None:
            timestamps[key] = None
            continue
        images[key] = np.asarray(frame.image).copy()
        timestamps[key] = float(frame.source_timestamp)
    return images, timestamps


def observation_faults(
    *,
    active_sides: tuple[str, ...],
    arm_enabled: bool,
    state_valid: np.ndarray,
    image_timestamps: Mapping[str, float | None],
    require_images: bool,
    hand_states: Mapping[str, Mapping[str, Any] | None],
    arm_state: Mapping[str, Any] | None,
) -> list[str]:
    faults: list[str] = []
    slices = _state_slices()
    if arm_enabled and not np.all(state_valid[slices["arm"]]):
        faults.append("arm_state_unavailable")
    for side in active_sides:
        if not np.all(state_valid[slices[f"{side}_hand"]]):
            faults.append(f"{side}_hand_state_unavailable")
        error = (hand_states.get(side) or {}).get("telemetry_error")
        if error:
            faults.append(f"{side}_hand_telemetry:{error}")
    if arm_enabled:
        error = (arm_state or {}).get("state_error")
        if error:
            faults.append(f"arm_state:{error}")
    if require_images:
        missing = [key for key, ts in image_timestamps.items() if ts is None]
        if missing:
            faults.append("camera_frame_unavailable:" + ",".join(sorted(missing)))
    return faults


def build_deployment_observation(
    *,
    active_sides: tuple[str, ...],
    arm_enabled: bool,
    step_index: int,
    hand_states: Mapping[str, Mapping[str, Any] | None],
    arm_state: Mapping[str, Any] | None = None,
    camera_sources: Mapping[str, Any] | None = None,
    last_action: np.ndarray | None = None,
    timestamp: float | None = None,
    require_images: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> DeploymentObservation:
    active = tuple(side for side in ALL_SIDES if side in set(active_sides))
    now = time.time() if timestamp is None else float(timestamp)
    state, state_dict, state_valid = build_state_vector(
        active_sides=active,
        arm_enabled=bool(arm_enabled),
        arm_state=arm_state,
        hand_states=hand_states,
    )
    tactile, tactile_by_side, tactile_valid_by_side = build_tactile_vectors(
        active_sides=active,
        hand_states=hand_states,
    )
    images, image_timestamps = latest_camera_frames(camera_sources or {})
    # Preserve configured camera keys even before the first frame arrives.
    for camera_key in (camera_sources or {}):
        image_timestamps.setdefault(str(camera_key), None)
    faults = observation_faults(
        active_sides=active,
        arm_enabled=bool(arm_enabled),
        state_valid=state_valid,
        image_timestamps=image_timestamps,
        require_images=bool(require_images),
        hand_states=hand_states,
        arm_state=arm_state,
    )
    last = nan_vector(ACTION_DIM) if last_action is None else np.asarray(last_action, dtype=np.float32).reshape(-1)
    if last.shape != (ACTION_DIM,):
        last = nan_vector(ACTION_DIM)
    return DeploymentObservation(
        timestamp=now,
        step_index=int(step_index),
        active_sides=active,
        arm_enabled=bool(arm_enabled),
        images=images,
        image_timestamps=image_timestamps,
        state=state,
        state_dict=state_dict,
        state_valid=state_valid,
        tactile=tactile,
        tactile_by_side=tactile_by_side,
        tactile_valid_by_side=tactile_valid_by_side,
        last_action=last.astype(np.float32, copy=True),
        hand_states={
            str(side): dict(to_serializable(dict(state or {})))
            for side, state in hand_states.items()
        },
        arm_state={} if arm_state is None else dict(to_serializable(dict(arm_state))),
        faults=faults,
        metadata={} if metadata is None else dict(to_serializable(dict(metadata))),
    )
