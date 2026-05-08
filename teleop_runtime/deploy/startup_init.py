from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .actions import NormalizedAction
from .observation import DeploymentObservation
from .replay import load_replay_actions_from_parquet, resolve_replay_parquet_path
from .schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    HAND_VECTOR_ORDER,
    STATE_ARM_POSE_SLICES,
)

_STARTUP_ARM_POSITION_TOLERANCE_M = 0.01
_STARTUP_ARM_ROTATION_TOLERANCE_DEG = 5.0


def _action_vector(value: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"{label} must have shape ({ACTION_DIM},), got {vector.shape}.")
    return vector.astype(np.float32, copy=True)


def _policy_replay_first_action(policy: Any) -> np.ndarray | None:
    wrapped = getattr(policy, "policy", policy)
    actions = getattr(wrapped, "actions", None)
    if actions is None:
        return None
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM or array.shape[0] <= 0:
        raise ValueError(
            f"Replay startup init actions must have shape (N, {ACTION_DIM}), got {array.shape}."
        )
    return array[0].astype(np.float32, copy=True)


def _load_dataset_init_action(args: Any) -> tuple[np.ndarray, str]:
    dataset_root = getattr(args, "init_dataset", None)
    parquet_path = getattr(args, "init_parquet", None)
    if not dataset_root and not parquet_path:
        raise ValueError("--init-source dataset requires --init-dataset or --init-parquet.")
    frame_index = int(getattr(args, "init_frame", 0))
    path, metadata = resolve_replay_parquet_path(
        dataset_root=dataset_root,
        parquet_path=parquet_path,
        episode_index=int(getattr(args, "init_episode", 0)),
    )
    actions = load_replay_actions_from_parquet(
        path,
        action_column=str(getattr(args, "init_action_column", "action") or "action"),
        start_index=frame_index,
        end_index=frame_index + 1,
    )
    source = (
        f"dataset:{Path(path)}"
        f":episode={metadata.get('episode_index', getattr(args, 'init_episode', 0))}"
        f":frame={frame_index}"
    )
    return actions[0].astype(np.float32, copy=True), source


def resolve_startup_init_action(args: Any, policy: Any) -> tuple[np.ndarray | None, str]:
    source = str(getattr(args, "init_source", "auto") or "auto").strip().lower()
    if source in {"none", "off", "disabled"} or float(getattr(args, "init_time_s", 0.0)) <= 0.0:
        return None, "disabled"
    if source not in {"auto", "current", "dataset", "replay"}:
        raise ValueError("--init-source must be one of: auto, current, dataset, replay, none.")

    replay_first = _policy_replay_first_action(policy)
    has_dataset_init = bool(getattr(args, "init_dataset", None) or getattr(args, "init_parquet", None))
    if source == "auto":
        if replay_first is not None:
            return replay_first, "replay:first_frame"
        if has_dataset_init:
            return _load_dataset_init_action(args)
        return None, "current"
    if source == "current":
        return None, "current"
    if source == "replay":
        if replay_first is None:
            raise ValueError("--init-source replay requires --debug-replay-dataset or --debug-replay-parquet.")
        return replay_first, "replay:first_frame"
    return _load_dataset_init_action(args)


def _required_slices(
    *,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
) -> list[tuple[str, slice]]:
    required: list[tuple[str, slice]] = []
    active = set(active_sides)
    if arms_enabled:
        for side in ALL_SIDES:
            if side in active:
                required.append((f"{side}_arm", ACTION_ARM_POSE_SLICES[side]))
    if hands_enabled:
        for side in HAND_VECTOR_ORDER:
            if side in active:
                required.append((f"{side}_hand", ACTION_HAND_SLICES[side]))
    return required


def _missing_required(vector: np.ndarray, required: list[tuple[str, slice]]) -> list[str]:
    return [name for name, target_slice in required if not np.all(np.isfinite(vector[target_slice]))]


def _smoothstep(alpha: float) -> float:
    x = float(np.clip(alpha, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _quat_distance_deg(current: np.ndarray, target: np.ndarray) -> float:
    first = np.asarray(current, dtype=float).reshape(4)
    second = np.asarray(target, dtype=float).reshape(4)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return float("inf")
    current_rot = R.from_quat(first / first_norm)
    target_rot = R.from_quat(second / second_norm)
    return float(np.rad2deg((target_rot * current_rot.inv()).magnitude()))


def _hardware_pose_xyzquat(observation: DeploymentObservation, side: str) -> np.ndarray | None:
    state = dict(observation.arm_states.get(side) or {})
    pose_6d = state.get("hardware_pose_6d")
    if pose_6d is None:
        return None
    try:
        pose = np.asarray(pose_6d, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        return None
    quat = R.from_euler("xyz", np.asarray(pose[3:6], dtype=float)).as_quat().astype(np.float32)
    return np.concatenate((pose[:3], quat), dtype=np.float32)


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    first = np.asarray(q0, dtype=np.float64).reshape(4)
    second = np.asarray(q1, dtype=np.float64).reshape(4)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return ((1.0 - alpha) * first + alpha * second).astype(np.float32)
    first = first / first_norm
    second = second / second_norm
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        quat = first + float(alpha) * (second - first)
    else:
        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        quat = (
            np.sin((1.0 - float(alpha)) * theta) / sin_theta * first
            + np.sin(float(alpha) * theta) / sin_theta * second
        )
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm <= 1e-12:
        return first.astype(np.float32)
    return (quat / quat_norm).astype(np.float32)


def _interpolate_action(
    start: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
) -> np.ndarray:
    action = (start + (target - start) * float(alpha)).astype(np.float32, copy=True)
    active = set(active_sides)
    if hands_enabled:
        for side in HAND_VECTOR_ORDER:
            if side in active:
                action[ACTION_HAND_SLICES[side]] = target[ACTION_HAND_SLICES[side]]
    else:
        for side in HAND_VECTOR_ORDER:
            if side in active:
                action[ACTION_HAND_SLICES[side]] = start[ACTION_HAND_SLICES[side]]
    if not arms_enabled:
        return action
    for side in ALL_SIDES:
        if side not in active:
            continue
        target_slice = ACTION_ARM_POSE_SLICES[side]
        start_pose = start[target_slice]
        target_pose = target[target_slice]
        if not (np.all(np.isfinite(start_pose)) and np.all(np.isfinite(target_pose))):
            continue
        action[target_slice.start : target_slice.start + 3] = (
            start_pose[:3] + (target_pose[:3] - start_pose[:3]) * float(alpha)
        )
        action[target_slice.start + 3 : target_slice.stop] = _slerp_quat(
            start_pose[3:7],
            target_pose[3:7],
            alpha,
        )
    return action


def _startup_action_metadata() -> dict[str, Any]:
    return {
        "safety_overrides": {
            "max_arm_xyz_step_m": 0.0,
            "max_arm_rot_step_deg": 0.0,
            "max_arm_joint_step_rad": 0.0,
            "max_hand_step": 0.0,
        }
    }


@dataclass
class StartupInitializer:
    duration_s: float
    source: str
    target_action: np.ndarray | None
    active_sides: tuple[str, ...]
    arms_enabled: bool
    hands_enabled: bool
    started_at: float | None = None
    start_action: np.ndarray | None = None
    resolved_target_action: np.ndarray | None = None
    completed: bool = False
    progress: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.source != "disabled" and float(self.duration_s) > 0.0

    def reset(self) -> None:
        self.started_at = None
        self.start_action = None
        self.resolved_target_action = None
        self.completed = not self.enabled
        self.progress = 1.0 if self.completed else 0.0

    def _start(self, observation: DeploymentObservation, now: float) -> list[str]:
        start = _action_vector(observation.hold_action(), label="startup init start action")
        if self.target_action is None:
            target = start.copy()
        else:
            target = _action_vector(self.target_action, label="startup init target action")
            missing_target = ~np.isfinite(target)
            target[missing_target] = start[missing_target]

        required = _required_slices(
            active_sides=self.active_sides,
            arms_enabled=bool(self.arms_enabled),
            hands_enabled=bool(self.hands_enabled),
        )
        missing_start = _missing_required(start, required)
        missing_target = _missing_required(target, required)
        if missing_start or missing_target:
            missing = sorted(set(missing_start + missing_target))
            return ["startup_init_reference_unavailable:" + ",".join(missing)]

        self.started_at = float(now)
        self.start_action = start
        self.resolved_target_action = target
        self.progress = 0.0
        print(
            "[StartupInit] moving to initial target "
            f"source={self.source} duration={float(self.duration_s):.2f}s"
        )
        return []

    def next_action(
        self,
        observation: DeploymentObservation,
        *,
        now: float | None = None,
    ) -> tuple[NormalizedAction | None, list[str], bool]:
        if not self.enabled or self.completed:
            return None, [], True
        current_time = time.time() if now is None else float(now)
        if self.started_at is None:
            faults = self._start(observation, current_time)
            if faults:
                return None, faults, False

        assert self.started_at is not None
        assert self.start_action is not None
        assert self.resolved_target_action is not None
        elapsed = max(0.0, current_time - float(self.started_at))
        raw_alpha = 1.0 if float(self.duration_s) <= 0.0 else min(1.0, elapsed / float(self.duration_s))
        alpha = _smoothstep(raw_alpha)
        self.progress = float(raw_alpha)
        if raw_alpha >= 1.0:
            return (
                NormalizedAction(
                    vector=self.resolved_target_action.copy(),
                    metadata=_startup_action_metadata(),
                ),
                [],
                self._target_reached(observation),
            )
        action = _interpolate_action(
            self.start_action,
            self.resolved_target_action,
            alpha=alpha,
            active_sides=self.active_sides,
            arms_enabled=bool(self.arms_enabled),
            hands_enabled=bool(self.hands_enabled),
        )
        if not bool(self.arms_enabled):
            return NormalizedAction(vector=action, metadata=_startup_action_metadata()), [], True
        return NormalizedAction(vector=action, metadata=_startup_action_metadata()), [], False

    def _target_reached(self, observation: DeploymentObservation) -> bool:
        assert self.resolved_target_action is not None
        target = self.resolved_target_action
        state = np.asarray(observation.state, dtype=np.float32).reshape(-1)
        active = set(self.active_sides)
        if bool(self.arms_enabled):
            for side in ALL_SIDES:
                if side not in active:
                    continue
                target_pose = target[ACTION_ARM_POSE_SLICES[side]]
                current_pose = _hardware_pose_xyzquat(observation, side)
                pose_source = "hardware"
                if current_pose is None:
                    current_pose = state[STATE_ARM_POSE_SLICES[side]]
                    pose_source = "state"
                if not (np.all(np.isfinite(target_pose)) and np.all(np.isfinite(current_pose))):
                    return False
                position_error = float(np.linalg.norm(current_pose[:3] - target_pose[:3]))
                rotation_error_deg = _quat_distance_deg(current_pose[3:7], target_pose[3:7])
                if position_error > _STARTUP_ARM_POSITION_TOLERANCE_M:
                    return False
                if rotation_error_deg > _STARTUP_ARM_ROTATION_TOLERANCE_DEG:
                    return False
                print(
                    "[StartupInit] arm reached target "
                    f"side={side} source={pose_source} "
                    f"pos_err_mm={position_error * 1000.0:.2f} rot_err_deg={rotation_error_deg:.2f}"
                )
        return True

    def mark_complete(self) -> None:
        if not self.completed:
            print("[StartupInit] complete; policy/replay execution may start.")
        self.completed = True
        self.progress = 1.0

    @property
    def status_text(self) -> str:
        if not self.enabled:
            return ""
        state = "done" if self.completed else ("wait" if self.started_at is None else f"{100.0 * self.progress:.0f}%")
        return f"init={state}:{self.source}"
