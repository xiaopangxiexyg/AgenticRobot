from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_HAND_SLICES,
    ARM_JOINT_DOF,
    ARM_POSE_DOF,
    HAND_DOF,
    xyzquat_to_pose_matrix,
)
from robot_motion.runtime import EffectorCommand

from .replay import (
    infer_replay_action_hz,
    load_replay_actions_from_parquet,
    load_replay_arm_joint_targets_from_parquet,
    resolve_replay_parquet_path,
)


_MAX_QPOS_INIT_DELTA_RAD = float(np.pi)


@dataclass(frozen=True)
class FixedInitEpisodeSpec:
    episode_index: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class DeployFixedInitPlan:
    source_path: str
    source_episode_index: int
    start_index: int
    end_index: int
    frame_count: int
    action_hz: float
    actions: np.ndarray
    arm_joint_targets: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FixedInitStepResult:
    last_action: np.ndarray | None = None
    completed: bool = False
    message: str = ""


def _parse_replay_init_episode_spec(text: str, *, total_frames: int) -> FixedInitEpisodeSpec:
    parts = [part.strip() for part in str(text or "0:0:end").split(":")]
    if not parts or not parts[0]:
        raise ValueError("fixed init episode spec must start with an episode index.")
    episode_index = int(parts[0])
    if len(parts) == 1:
        return FixedInitEpisodeSpec(
            episode_index=episode_index,
            start_index=0,
            end_index=int(total_frames),
        )
    if len(parts) != 3:
        raise ValueError("fixed init episode spec must be N or N:start:end.")
    start_index = int(parts[1] or 0)
    end_text = parts[2].lower()
    end_index = int(total_frames) if end_text in {"", "end"} else int(parts[2])
    if start_index < 0 or start_index >= int(total_frames):
        raise ValueError(
            f"fixed init start_index={start_index} is outside frame range [0, {total_frames})."
        )
    if end_index <= start_index or end_index > int(total_frames):
        raise ValueError(
            f"fixed init end_index={end_index} must satisfy start < end <= {total_frames}."
        )
    return FixedInitEpisodeSpec(
        episode_index=episode_index,
        start_index=start_index,
        end_index=end_index,
    )


def _episode_index_from_spec(text: str) -> int:
    token = str(text or "0:0:end").split(":", 1)[0].strip()
    if not token:
        raise ValueError("fixed init episode spec must start with an episode index.")
    return int(token)


def load_deploy_fixed_init_plan(
    *,
    dataset_root: str | Path | None,
    parquet_path: str | Path | None,
    episode_spec: str,
    action_column: str,
    active_sides: Sequence[str],
) -> DeployFixedInitPlan | None:
    if not dataset_root and not parquet_path:
        return None
    active = tuple(str(side).strip().lower() for side in active_sides)
    if active != ("right",):
        raise ValueError("Deploy fixed init is currently right-arm/right-hand only; use --active-sides right.")

    requested_episode_index = _episode_index_from_spec(episode_spec)
    source_path, resolved_metadata = resolve_replay_parquet_path(
        dataset_root=dataset_root,
        parquet_path=parquet_path,
        episode_index=requested_episode_index,
    )
    all_actions = load_replay_actions_from_parquet(
        source_path,
        action_column=str(action_column or "action"),
        start_index=0,
        end_index=None,
    )
    total_frames = int(all_actions.shape[0])
    parsed = _parse_replay_init_episode_spec(episode_spec, total_frames=total_frames)
    actions = all_actions[parsed.start_index : parsed.end_index].astype(np.float32, copy=True)
    arm_joint_targets = load_replay_arm_joint_targets_from_parquet(
        source_path,
        active_sides=("right",),
        start_index=int(parsed.start_index),
        end_index=int(parsed.end_index),
        required=True,
    )
    action_hz_info = infer_replay_action_hz(
        parquet_path=source_path,
        dataset_root=dataset_root,
        episode_index=int(parsed.episode_index),
        start_index=int(parsed.start_index),
        end_index=int(parsed.end_index),
        resolved_metadata=resolved_metadata,
    )
    hz = float(action_hz_info.get("hz", 0.0) or 0.0)
    if hz <= 0.0:
        raise ValueError("Fixed init trajectory action Hz must be positive.")
    source_episode_index = int(resolved_metadata.get("episode_index", int(parsed.episode_index)))
    return DeployFixedInitPlan(
        source_path=str(source_path),
        source_episode_index=source_episode_index,
        start_index=int(parsed.start_index),
        end_index=int(parsed.end_index),
        frame_count=int(actions.shape[0]),
        action_hz=hz,
        actions=actions,
        arm_joint_targets=arm_joint_targets,
        metadata={
            "action_hz_estimate": dict(action_hz_info),
            "source_metadata": dict(resolved_metadata),
            "action_column": str(action_column or "action"),
        },
    )


def _set_backend_max_angvel(backend: Any, max_angvel: Sequence[float]) -> None:
    values = tuple(float(value) for value in max_angvel)
    if len(values) != ARM_JOINT_DOF or any(value <= 0.0 for value in values):
        raise ValueError("fixed init arm max angvel must contain 6 positive values.")
    backend.config.max_angvel = values


def _send_commands(
    commands: Mapping[str, EffectorCommand],
    *,
    arm_backends: Mapping[str, Any],
    hand_backends: Mapping[str, Any],
) -> None:
    arm_command = commands.get("right_arm")
    if arm_command is not None and "right" in arm_backends:
        arm_backends["right"].send_command(arm_command)
    hand_command = commands.get("right_hand")
    if hand_command is not None and "right" in hand_backends:
        hand_backends["right"].send_command(hand_command)


def _right_hand_command_from_action(action: np.ndarray, *, source: str) -> EffectorCommand | None:
    target = np.asarray(action[ACTION_HAND_SLICES["right"]], dtype=np.float32).reshape(-1)
    if target.shape != (HAND_DOF,) or not np.all(np.isfinite(target)):
        return None
    return EffectorCommand(
        effector_name="right_hand",
        side="right",
        command_type="joint_vector",
        value=np.clip(target, 0.0, 1.0),
        frame_name=None,
        metadata={"source": source},
    )


def _right_arm_pose_command_from_action(action: np.ndarray, *, source: str) -> EffectorCommand:
    target = np.asarray(action[ACTION_ARM_POSE_SLICES["right"]], dtype=np.float32).reshape(-1)
    if target.shape != (ARM_POSE_DOF,) or not np.all(np.isfinite(target)):
        raise RuntimeError("Fixed init action is missing a finite right arm pose.")
    pose_matrix = xyzquat_to_pose_matrix(target)
    if pose_matrix is None:
        raise RuntimeError("Fixed init action contains an invalid right arm pose.")
    return EffectorCommand(
        effector_name="right_arm",
        side="right",
        command_type="cartesian_pose",
        value=pose_matrix,
        frame_name="right_arm_base",
        metadata={
            "source": source,
            "target_pose_base": pose_matrix.copy(),
        },
    )


def _right_arm_qpos_command(qpos: np.ndarray, *, source: str) -> EffectorCommand:
    target = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if target.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(target)):
        raise RuntimeError("Fixed init qpos target must be a finite 6D vector.")
    return EffectorCommand(
        effector_name="right_arm",
        side="right",
        command_type="joint_vector",
        value=target.astype(float, copy=True),
        frame_name=None,
        metadata={"source": source},
    )


def _action_commands(action: np.ndarray, *, source: str, include_arm_pose: bool) -> dict[str, EffectorCommand]:
    commands: dict[str, EffectorCommand] = {}
    if include_arm_pose:
        commands["right_arm"] = _right_arm_pose_command_from_action(action, source=source)
    hand_command = _right_hand_command_from_action(action, source=source)
    if hand_command is not None:
        commands["right_hand"] = hand_command
    return commands


def _qpos_reached(
    backend: Any,
    target_qpos: np.ndarray,
    *,
    tolerance: float,
    send_actions: bool,
) -> bool:
    if not send_actions:
        return True
    try:
        state = backend.refresh_state_from_hardware()
    except Exception:
        state = backend.get_state()
    for key in ("hardware_qpos", "current_qpos", "last_target_qpos"):
        current = state.get(key)
        if current is None:
            continue
        vector = np.asarray(current, dtype=float).reshape(-1)
        if vector.shape == (ARM_JOINT_DOF,) and np.all(np.isfinite(vector)):
            error = float(np.max(np.abs(np.asarray(target_qpos, dtype=float).reshape(ARM_JOINT_DOF) - vector)))
            return error <= float(tolerance)
    return False


def _current_qpos_for_init_safety(backend: Any) -> np.ndarray:
    try:
        state = backend.refresh_state_from_hardware()
    except Exception:
        state = backend.get_state()
    for key in ("hardware_qpos", "current_qpos"):
        current = state.get(key)
        if current is None:
            continue
        vector = np.asarray(current, dtype=float).reshape(-1)
        if vector.shape == (ARM_JOINT_DOF,) and np.all(np.isfinite(vector)):
            return vector.astype(float, copy=True)
    raise RuntimeError(
        "Cannot verify fixed-init qpos safety: right_arm current qpos is unavailable."
    )


def _assert_qpos_init_within_180_deg(backend: Any, target_qpos: np.ndarray) -> None:
    current = _current_qpos_for_init_safety(backend)
    target = np.asarray(target_qpos, dtype=float).reshape(ARM_JOINT_DOF)
    diff = np.abs(target - current)
    unsafe = np.flatnonzero(diff > _MAX_QPOS_INIT_DELTA_RAD)
    if unsafe.size <= 0:
        return
    details = ", ".join(
        (
            f"j{int(index) + 1}:"
            f"current={np.rad2deg(current[index]):.1f}deg,"
            f"target={np.rad2deg(target[index]):.1f}deg,"
            f"diff={np.rad2deg(diff[index]):.1f}deg"
        )
        for index in unsafe
    )
    raise RuntimeError(
        "Unsafe fixed-init qpos target: right_arm joint delta exceeds 180deg; "
        f"{details}. Command was not sent."
    )


class DeployFixedInitController:
    def __init__(
        self,
        plan: DeployFixedInitPlan,
        *,
        arm_max_angvel: Sequence[float],
        qpos_tolerance: float = 0.1,
    ) -> None:
        self.plan = plan
        self.arm_max_angvel = tuple(float(value) for value in arm_max_angvel)
        self.qpos_tolerance = float(qpos_tolerance)
        self.state = "idle"
        self._original_max_angvel: dict[str, tuple[float, ...]] = {}
        self._execute_started_at: float | None = None
        self._replay_index = 0

    @property
    def blocks_policy(self) -> bool:
        return self.state != "released"

    @property
    def running(self) -> bool:
        return self.state in {"moving_to_start", "executing"}

    @property
    def status_text(self) -> str:
        if self.state == "idle":
            return "fixed_init=wait:H"
        if self.state == "moving_to_start":
            return "fixed_init=qpos_start"
        if self.state == "start_ready":
            return "fixed_init=ready:H"
        if self.state == "executing":
            return f"fixed_init=T:{self._replay_index}/{self.plan.frame_count}"
        return "fixed_init=released"

    def reset_to_idle(self, arm_backends: Mapping[str, Any]) -> None:
        self.restore_speed(arm_backends)
        self.state = "idle"
        self._execute_started_at = None
        self._replay_index = 0

    def restore_speed(self, arm_backends: Mapping[str, Any]) -> None:
        for side, max_angvel in list(self._original_max_angvel.items()):
            backend = arm_backends.get(side)
            if backend is not None:
                _set_backend_max_angvel(backend, max_angvel)
        self._original_max_angvel.clear()

    def _begin_speed_override(self, arm_backends: Mapping[str, Any]) -> None:
        if self._original_max_angvel:
            return
        backend = arm_backends.get("right")
        if backend is None:
            return
        self._original_max_angvel["right"] = tuple(float(value) for value in backend.config.max_angvel)
        _set_backend_max_angvel(backend, self.arm_max_angvel)

    def handle_h(
        self,
        *,
        arm_backends: Mapping[str, Any],
        now: float,
    ) -> str:
        if self.state in {"moving_to_start", "executing"}:
            return f"[FixedInit] ignore H while {self.state}."
        if self.state in {"idle", "released"}:
            if "right" not in arm_backends:
                raise RuntimeError("Fixed init requires a right arm backend.")
            self._begin_speed_override(arm_backends)
            self._execute_started_at = None
            self._replay_index = 0
            self.state = "moving_to_start"
            return (
                "[FixedInit] moving to T first-frame qpos; "
                "press H again after start is reached."
            )
        if self.state == "start_ready":
            self._begin_speed_override(arm_backends)
            self._execute_started_at = float(now)
            self._replay_index = 0
            self.state = "executing"
            return (
                "[FixedInit] executing T without policy "
                f"frames={self.plan.frame_count} hz={self.plan.action_hz:.2f}."
            )
        return ""

    def step(
        self,
        *,
        now: float,
        send_actions: bool,
        arm_backends: Mapping[str, Any],
        hand_backends: Mapping[str, Any],
    ) -> FixedInitStepResult:
        if self.state == "moving_to_start":
            backend = arm_backends.get("right")
            if backend is None:
                raise RuntimeError("Fixed init requires a right arm backend.")
            first_action = self.plan.actions[0].astype(np.float32, copy=True)
            target_qpos = self.plan.arm_joint_targets["right"][0].astype(np.float32, copy=True)
            commands = _action_commands(first_action, source="deploy_fixed_init_qpos", include_arm_pose=False)
            commands["right_arm"] = _right_arm_qpos_command(target_qpos, source="deploy_fixed_init_qpos")
            if send_actions:
                _assert_qpos_init_within_180_deg(backend, target_qpos)
                _send_commands(commands, arm_backends=arm_backends, hand_backends=hand_backends)
            if _qpos_reached(
                backend,
                target_qpos,
                tolerance=self.qpos_tolerance,
                send_actions=send_actions,
            ):
                self.restore_speed(arm_backends)
                self.state = "start_ready"
                return FixedInitStepResult(
                    last_action=first_action,
                    message="[FixedInit] T first-frame qpos reached; press H again to execute T.",
                )
            return FixedInitStepResult(last_action=first_action)

        if self.state != "executing":
            return FixedInitStepResult()

        assert self._execute_started_at is not None
        due_time = float(self._execute_started_at) + float(self._replay_index) / float(self.plan.action_hz)
        if now + 1e-9 < due_time:
            return FixedInitStepResult()
        if self._replay_index >= int(self.plan.frame_count):
            self.restore_speed(arm_backends)
            self.state = "released"
            return FixedInitStepResult(
                completed=True,
                message="[FixedInit] T finished; deployment startup init is released.",
            )

        action = self.plan.actions[self._replay_index].astype(np.float32, copy=True)
        commands = _action_commands(action, source="deploy_fixed_init_trajectory", include_arm_pose=True)
        if send_actions:
            _send_commands(commands, arm_backends=arm_backends, hand_backends=hand_backends)
        self._replay_index += 1
        if self._replay_index >= int(self.plan.frame_count):
            self.restore_speed(arm_backends)
            self.state = "released"
            return FixedInitStepResult(
                last_action=action,
                completed=True,
                message="[FixedInit] T finished; deployment startup init is released.",
            )
        return FixedInitStepResult(last_action=action)
