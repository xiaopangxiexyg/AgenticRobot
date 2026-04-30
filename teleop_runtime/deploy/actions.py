from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..schemas import EffectorCommand
from .observation import DeploymentObservation
from .schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    ARM_JOINT_DOF,
    ARM_POSE_DOF,
    HAND_DOF,
    HAND_VECTOR_ORDER,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
    STATE_HAND_SLICES,
    mapping_get,
    nan_vector,
    vector_or_none,
    xyzquat_to_pose_matrix,
)


@dataclass(frozen=True)
class NormalizedAction:
    vector: np.ndarray
    arm_joint_targets: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeploymentAction:
    vector: np.ndarray
    commands: dict[str, EffectorCommand] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _base_action(observation: DeploymentObservation) -> np.ndarray:
    base = observation.hold_action()
    if base.shape != (ACTION_DIM,):
        base = nan_vector(ACTION_DIM)
    return base.astype(np.float32, copy=True)


def _replace_vector_slice(
    vector: np.ndarray,
    target_slice: slice,
    value: Any,
    *,
    expected_dim: int,
) -> None:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (expected_dim,):
        raise ValueError(f"Expected action component dim {expected_dim}, got {arr.shape}.")
    vector[target_slice] = arr


def _ordered_active_sides(active_sides: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(side for side in ALL_SIDES if side in set(active_sides))


def _compact_vector_to_action(
    value: Any,
    observation: DeploymentObservation,
    *,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
) -> NormalizedAction:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    base = _base_action(observation)
    active = _ordered_active_sides(active_sides)

    if arr.shape == (ACTION_DIM,):
        return NormalizedAction(vector=arr.astype(np.float32, copy=True))

    if arms_enabled:
        if arr.shape == (ARM_POSE_DOF * len(active),) and active:
            offset = 0
            for side in active:
                base[ACTION_ARM_POSE_SLICES[side]] = arr[offset : offset + ARM_POSE_DOF]
                offset += ARM_POSE_DOF
            return NormalizedAction(vector=base)
        if arr.shape == (ARM_POSE_DOF,) and len(active) == 1:
            base[ACTION_ARM_POSE_SLICES[active[0]]] = arr
            return NormalizedAction(vector=base)

    if hands_enabled:
        if arr.shape == (HAND_DOF * len(active),) and active:
            offset = 0
            for side in active:
                base[ACTION_HAND_SLICES[side]] = arr[offset : offset + HAND_DOF]
                offset += HAND_DOF
            return NormalizedAction(vector=base)
        if arr.shape == (HAND_DOF,) and len(active) == 1:
            base[ACTION_HAND_SLICES[active[0]]] = arr
            return NormalizedAction(vector=base)

    raise ValueError(
        "Unsupported action vector shape "
        f"{arr.shape}; expected {ACTION_DIM}, active arm pose dims, or active hand dims."
    )


def _array_to_action_chunk(
    value: Any,
    observation: DeploymentObservation,
    *,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
) -> list[NormalizedAction]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return [
            _compact_vector_to_action(
                arr,
                observation,
                active_sides=active_sides,
                arms_enabled=arms_enabled,
                hands_enabled=hands_enabled,
            )
        ]
    if arr.ndim == 2:
        return [
            _compact_vector_to_action(
                arr[index],
                observation,
                active_sides=active_sides,
                arms_enabled=arms_enabled,
                hands_enabled=hands_enabled,
            )
            for index in range(arr.shape[0])
        ]
    raise ValueError(f"Unsupported action array ndim {arr.ndim}; expected 1 or 2.")


def _component_horizon(value: Any) -> int:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return 1
    if arr.ndim == 2:
        return int(arr.shape[0])
    raise ValueError(f"Unsupported component ndim {arr.ndim}; expected 1 or 2.")


def _component_at(value: Any, index: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1)
    return arr[int(index)].reshape(-1)


def _side_component_aliases(side: str, kind: str) -> tuple[str, ...]:
    if kind == "arm_pose":
        return (
            f"{side}_arm_pose",
            f"{side}_arm_target_pose",
            f"action.{side}_arm.pose",
            f"action.{side}_arm.target_pose",
        )
    if kind == "arm_qpos":
        return (
            f"{side}_arm_qpos",
            f"{side}_arm_joint_target",
            f"action.{side}_arm.joint_target",
            f"action.replay.{side}_arm_joint_target",
        )
    if kind == "hand":
        return (
            f"{side}_hand",
            f"{side}_hand_q6",
            f"{side}_hand_joint_target",
            f"action.{side}_hand.joint_target",
            f"action.replay.{side}_hand_joint_target",
        )
    raise ValueError(kind)


def _mapping_to_action_chunk(
    mapping: Mapping[str, Any],
    observation: DeploymentObservation,
    *,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
) -> list[NormalizedAction]:
    for key in ("action_chunk", "actions", "action"):
        if key in mapping:
            return normalize_policy_output(
                mapping[key],
                observation,
                active_sides=active_sides,
                arms_enabled=arms_enabled,
                hands_enabled=hands_enabled,
            )

    active = _ordered_active_sides(active_sides)
    components: dict[tuple[str, str], Any] = {}
    for side in active:
        arm_value = mapping_get(mapping, (f"{side}_arm",))
        if arm_value is not None:
            arr = np.asarray(arm_value, dtype=np.float32)
            if arr.shape[-1] == ARM_JOINT_DOF:
                components[(side, "arm_qpos")] = arm_value
            else:
                components[(side, "arm_pose")] = arm_value
        pose_value = mapping_get(mapping, _side_component_aliases(side, "arm_pose"))
        qpos_value = mapping_get(mapping, _side_component_aliases(side, "arm_qpos"))
        hand_value = mapping_get(mapping, _side_component_aliases(side, "hand"))
        if pose_value is not None:
            components[(side, "arm_pose")] = pose_value
        if qpos_value is not None:
            components[(side, "arm_qpos")] = qpos_value
        if hand_value is not None:
            components[(side, "hand")] = hand_value

    if len(active) == 1:
        side = active[0]
        arm_value = mapping_get(mapping, ("arm", "arm_pose", "arm_target_pose"))
        if arm_value is not None:
            arr = np.asarray(arm_value, dtype=np.float32)
            components[(side, "arm_qpos" if arr.shape[-1] == ARM_JOINT_DOF else "arm_pose")] = arm_value
        qpos_value = mapping_get(mapping, ("arm_qpos", "arm_joint_target", "action.arm.joint_target"))
        hand_value = mapping_get(mapping, ("hand", "hand_q6", "hand_joint_target"))
        if qpos_value is not None:
            components[(side, "arm_qpos")] = qpos_value
        if hand_value is not None:
            components[(side, "hand")] = hand_value

    if not components:
        raise ValueError("Policy mapping did not contain action/action_chunk or supported effector keys.")

    horizon = max(_component_horizon(value) for value in components.values())
    chunk: list[NormalizedAction] = []
    for index in range(horizon):
        vector = _base_action(observation)
        arm_joint_targets: dict[str, np.ndarray] = {}
        for side in active:
            pose_key = (side, "arm_pose")
            if arms_enabled and pose_key in components:
                value = _component_at(
                    components[pose_key],
                    min(index, _component_horizon(components[pose_key]) - 1),
                )
                _replace_vector_slice(
                    vector,
                    ACTION_ARM_POSE_SLICES[side],
                    value,
                    expected_dim=ARM_POSE_DOF,
                )
            qpos_key = (side, "arm_qpos")
            if arms_enabled and qpos_key in components:
                value = _component_at(
                    components[qpos_key],
                    min(index, _component_horizon(components[qpos_key]) - 1),
                )
                arm_joint_targets[side] = np.asarray(value, dtype=np.float32).reshape(ARM_JOINT_DOF)
            hand_key = (side, "hand")
            if hands_enabled and hand_key in components:
                value = _component_at(
                    components[hand_key],
                    min(index, _component_horizon(components[hand_key]) - 1),
                )
                _replace_vector_slice(
                    vector,
                    ACTION_HAND_SLICES[side],
                    value,
                    expected_dim=HAND_DOF,
                )
        chunk.append(NormalizedAction(vector=vector, arm_joint_targets=arm_joint_targets))
    return chunk


def normalize_policy_output(
    output: Any,
    observation: DeploymentObservation,
    *,
    active_sides: tuple[str, ...] | None = None,
    arms_enabled: bool | None = None,
    hands_enabled: bool | None = None,
) -> list[NormalizedAction]:
    """Normalize common policy outputs to teleop_runtime deployment actions."""

    active = tuple(observation.active_sides if active_sides is None else active_sides)
    use_arms = bool(observation.arms_enabled if arms_enabled is None else arms_enabled)
    use_hands = bool(observation.hands_enabled if hands_enabled is None else hands_enabled)
    if output is None:
        return []
    if isinstance(output, NormalizedAction):
        return [output]
    for attr in ("action_chunk", "actions", "action"):
        if hasattr(output, attr):
            return normalize_policy_output(
                getattr(output, attr),
                observation,
                active_sides=active,
                arms_enabled=use_arms,
                hands_enabled=use_hands,
            )
    if isinstance(output, Mapping):
        return _mapping_to_action_chunk(
            output,
            observation,
            active_sides=active,
            arms_enabled=use_arms,
            hands_enabled=use_hands,
        )
    if isinstance(output, (list, tuple)) and not output:
        return []
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], NormalizedAction):
        return list(output)
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], Mapping):
        chunk: list[NormalizedAction] = []
        for item in output:
            chunk.extend(
                normalize_policy_output(
                    item,
                    observation,
                    active_sides=active,
                    arms_enabled=use_arms,
                    hands_enabled=use_hands,
                )
            )
        return chunk
    return _array_to_action_chunk(
        output,
        observation,
        active_sides=active,
        arms_enabled=use_arms,
        hands_enabled=use_hands,
    )


class ActionChunkBuffer:
    def __init__(self):
        self._queue: deque[NormalizedAction] = deque()
        self.replaced_count = 0
        self.popped_count = 0

    def __len__(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()

    def replace(self, actions: Sequence[NormalizedAction]) -> None:
        self._queue.clear()
        self._queue.extend(actions)
        self.replaced_count += 1

    def append(self, actions: Sequence[NormalizedAction]) -> None:
        self._queue.extend(actions)

    def pop(self, default: NormalizedAction | None = None) -> NormalizedAction | None:
        if self._queue:
            self.popped_count += 1
            return self._queue.popleft()
        return default


@dataclass(frozen=True)
class ActionSafetyFilterConfig:
    active_sides: tuple[str, ...]
    arms_enabled: bool = True
    hands_enabled: bool = False
    workspace_min: tuple[float, float, float] = (-0.60, -0.80, 0.00)
    workspace_max: tuple[float, float, float] = (0.60, 0.80, 0.90)
    max_arm_xyz_step_m: float = 0.03
    max_arm_rot_step_deg: float = 8.0
    max_arm_joint_step_rad: float = 0.08
    max_hand_step: float = 0.08


class ActionSafetyFilter:
    def __init__(self, config: ActionSafetyFilterConfig):
        self.config = config

    @staticmethod
    def _rate_limit_vector(
        target: np.ndarray,
        reference: np.ndarray,
        *,
        max_step: float,
    ) -> tuple[np.ndarray, bool]:
        if float(max_step) <= 0.0:
            return target.astype(np.float32, copy=True), False
        if reference.shape != target.shape or not np.all(np.isfinite(reference)):
            return target.astype(np.float32, copy=True), False
        delta = target - reference
        clipped_delta = np.clip(delta, -float(max_step), float(max_step))
        limited = reference + clipped_delta
        limited_any = bool(np.any(np.abs(clipped_delta - delta) > 1e-12))
        return limited.astype(np.float32, copy=True), limited_any

    @staticmethod
    def _reference_action_slice(
        observation: DeploymentObservation,
        target_slice: slice,
        fallback: np.ndarray,
    ) -> np.ndarray:
        last = vector_or_none(observation.last_action, length=ACTION_DIM)
        if last is not None and np.all(np.isfinite(last[target_slice])):
            return last[target_slice].copy()
        return fallback.astype(np.float32, copy=True)

    def _filter_arm_pose(
        self,
        *,
        side: str,
        target: np.ndarray,
        observation: DeploymentObservation,
        raw: np.ndarray,
    ) -> tuple[np.ndarray, EffectorCommand | None, list[str], dict[str, Any]]:
        faults: list[str] = []
        metadata: dict[str, Any] = {
            "workspace_clipped": False,
            "xyz_rate_limited": False,
            "rot_rate_limited": False,
        }
        pose = vector_or_none(target, length=ARM_POSE_DOF)
        if pose is None:
            return target, None, [f"{side}_arm_pose_invalid"], metadata

        workspace_min = np.asarray(self.config.workspace_min, dtype=np.float32).reshape(3)
        workspace_max = np.asarray(self.config.workspace_max, dtype=np.float32).reshape(3)
        clipped_xyz = np.clip(pose[:3], workspace_min, workspace_max)
        metadata["workspace_clipped"] = bool(np.any(np.abs(clipped_xyz - pose[:3]) > 1e-12))
        pose = pose.copy()
        pose[:3] = clipped_xyz

        fallback = np.asarray(observation.state[STATE_ARM_POSE_SLICES[side]], dtype=np.float32)
        reference = self._reference_action_slice(observation, ACTION_ARM_POSE_SLICES[side], fallback)
        if float(self.config.max_arm_xyz_step_m) > 0.0 and not np.all(np.isfinite(reference[:3])):
            faults.append(f"{side}_arm_pose_reference_unavailable")
        else:
            pose[:3], limited = self._rate_limit_vector(
                pose[:3],
                reference[:3],
                max_step=float(self.config.max_arm_xyz_step_m),
            )
            metadata["xyz_rate_limited"] = bool(limited)

        quat = np.asarray(pose[3:7], dtype=float)
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1e-12:
            return target, None, [f"{side}_arm_quat_invalid"], metadata
        pose[3:7] = (quat / quat_norm).astype(np.float32)
        if (
            float(self.config.max_arm_rot_step_deg) > 0.0
            and np.all(np.isfinite(reference[3:7]))
            and np.linalg.norm(reference[3:7]) > 1e-12
        ):
            ref_rot = R.from_quat(np.asarray(reference[3:7], dtype=float) / np.linalg.norm(reference[3:7]))
            target_rot = R.from_quat(np.asarray(pose[3:7], dtype=float))
            delta = target_rot * ref_rot.inv()
            angle = float(delta.magnitude())
            max_angle = np.deg2rad(float(self.config.max_arm_rot_step_deg))
            if angle > max_angle > 0.0:
                limited_delta = R.from_rotvec(delta.as_rotvec() / angle * max_angle)
                pose[3:7] = (limited_delta * ref_rot).as_quat().astype(np.float32)
                metadata["rot_rate_limited"] = True

        pose_matrix = xyzquat_to_pose_matrix(pose)
        if pose_matrix is None:
            return target, None, [f"{side}_arm_pose_invalid"], metadata
        command = EffectorCommand(
            effector_name=f"{side}_arm",
            side=side,
            command_type="cartesian_pose",
            value=pose_matrix,
            frame_name=f"{side}_arm_base",
            metadata={
                "source": "policy",
                "raw_action": raw[ACTION_ARM_POSE_SLICES[side]].copy(),
                "target_pose_base": pose_matrix.copy(),
                "safety": dict(metadata),
            },
        )
        return pose.astype(np.float32, copy=True), command, faults, metadata

    def _filter_arm_qpos(
        self,
        *,
        side: str,
        target: np.ndarray,
        observation: DeploymentObservation,
    ) -> tuple[np.ndarray, EffectorCommand | None, list[str], dict[str, Any]]:
        faults: list[str] = []
        metadata = {"joint_rate_limited": False}
        qpos = vector_or_none(target, length=ARM_JOINT_DOF)
        if qpos is None:
            return target, None, [f"{side}_arm_qpos_invalid"], metadata
        reference = np.asarray(observation.state[STATE_ARM_QPOS_SLICES[side]], dtype=np.float32)
        if float(self.config.max_arm_joint_step_rad) > 0.0 and not np.all(np.isfinite(reference)):
            faults.append(f"{side}_arm_qpos_reference_unavailable")
        else:
            qpos, limited = self._rate_limit_vector(
                qpos,
                reference,
                max_step=float(self.config.max_arm_joint_step_rad),
            )
            metadata["joint_rate_limited"] = bool(limited)
        command = EffectorCommand(
            effector_name=f"{side}_arm",
            side=side,
            command_type="joint_vector",
            value=qpos,
            frame_name=None,
            metadata={
                "source": "policy",
                "raw_joint_target": target.copy(),
                "safety": dict(metadata),
            },
        )
        return qpos.astype(np.float32, copy=True), command, faults, metadata

    def filter(
        self,
        action: NormalizedAction | Any,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> DeploymentAction:
        normalized = (
            action
            if isinstance(action, NormalizedAction)
            else NormalizedAction(vector=np.asarray(action, dtype=np.float32).reshape(ACTION_DIM))
        )
        raw = np.asarray(normalized.vector, dtype=np.float32).reshape(-1)
        if raw.shape != (ACTION_DIM,):
            raise ValueError(f"Expected full action shape {(ACTION_DIM,)}, got {raw.shape}.")

        filtered = raw.astype(np.float32, copy=True)
        faults: list[str] = []
        commands: dict[str, EffectorCommand] = {}
        safety_metadata: dict[str, Any] = {
            "policy_name": policy_name,
            "arm": {},
            "hand_rate_limited": {},
        }

        active = set(self.config.active_sides)
        if self.config.arms_enabled:
            for side in ALL_SIDES:
                if side not in active:
                    continue
                if side in normalized.arm_joint_targets:
                    _, command, arm_faults, metadata = self._filter_arm_qpos(
                        side=side,
                        target=np.asarray(normalized.arm_joint_targets[side], dtype=np.float32),
                        observation=observation,
                    )
                else:
                    pose, command, arm_faults, metadata = self._filter_arm_pose(
                        side=side,
                        target=filtered[ACTION_ARM_POSE_SLICES[side]],
                        observation=observation,
                        raw=raw,
                    )
                    filtered[ACTION_ARM_POSE_SLICES[side]] = pose
                faults.extend(arm_faults)
                safety_metadata["arm"][side] = metadata
                if command is not None:
                    command.metadata["policy_name"] = policy_name
                    commands[f"{side}_arm"] = command

        if self.config.hands_enabled:
            for side in HAND_VECTOR_ORDER:
                if side not in active:
                    continue
                target_slice = ACTION_HAND_SLICES[side]
                hand_target = vector_or_none(filtered[target_slice], length=HAND_DOF)
                if hand_target is None:
                    faults.append(f"{side}_hand_action_invalid")
                    continue
                clipped = np.clip(hand_target, 0.0, 1.0)
                reference = self._reference_action_slice(
                    observation,
                    target_slice,
                    np.asarray(observation.state[STATE_HAND_SLICES[side]], dtype=np.float32),
                )
                if float(self.config.max_hand_step) > 0.0 and not np.all(np.isfinite(reference)):
                    faults.append(f"{side}_hand_reference_unavailable")
                    continue
                clipped, limited = self._rate_limit_vector(
                    clipped,
                    reference,
                    max_step=float(self.config.max_hand_step),
                )
                filtered[target_slice] = clipped
                safety_metadata["hand_rate_limited"][side] = bool(limited)
                commands[f"{side}_hand"] = EffectorCommand(
                    effector_name=f"{side}_hand",
                    side=side,
                    command_type="joint_vector",
                    value=clipped,
                    frame_name=None,
                    metadata={
                        "source": "policy",
                        "policy_name": policy_name,
                        "raw_action": raw[target_slice].copy(),
                        "safety": {
                            "clip_range": [0.0, 1.0],
                            "rate_limited": bool(limited),
                            "max_step": float(self.config.max_hand_step),
                        },
                    },
                )

        return DeploymentAction(
            vector=filtered,
            commands=commands,
            faults=faults,
            metadata=safety_metadata,
        )
