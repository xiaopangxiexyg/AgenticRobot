from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..action import ArmActionSpec, arm_action_specs_from_metadata, hand_action_specs_from_metadata
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
    deploy_arm_specs_metadata,
    deploy_hand_specs_metadata,
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


@dataclass(frozen=True)
class FilteredNormalizedAction:
    action: NormalizedAction
    faults: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _base_action(observation: DeploymentObservation) -> np.ndarray:
    base = observation.hold_action()
    if base.shape != (ACTION_DIM,):
        base = nan_vector(ACTION_DIM)
    return base.astype(np.float32, copy=True)


def _normalized_action_metadata(
    observation: DeploymentObservation,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {} if extra is None else dict(extra)
    metadata.setdefault(
        "arm_action_specs_by_side",
        dict(observation.metadata.get("arm_action_specs_by_side", deploy_arm_specs_metadata())),
    )
    metadata.setdefault(
        "arm_controlled_frame_by_side",
        dict(observation.metadata.get("arm_controlled_frame_by_side", {})),
    )
    metadata.setdefault(
        "arm_reference_frame_by_side",
        dict(observation.metadata.get("arm_reference_frame_by_side", {})),
    )
    metadata.setdefault("arm_spec_required", True)
    metadata.setdefault(
        "hand_action_specs_by_side",
        dict(observation.metadata.get("hand_action_specs_by_side", deploy_hand_specs_metadata())),
    )
    metadata.setdefault(
        "hand_active_mask_by_side",
        dict(observation.metadata.get("hand_active_mask_by_side", {})),
    )
    metadata.setdefault("hand_spec_required", True)
    return metadata


def _with_extra_metadata(
    actions: list[NormalizedAction],
    extra: Mapping[str, Any] | None,
) -> list[NormalizedAction]:
    if not isinstance(extra, Mapping):
        return actions
    output: list[NormalizedAction] = []
    for action in actions:
        metadata = dict(action.metadata)
        metadata.update(dict(extra))
        output.append(
            NormalizedAction(
                vector=action.vector,
                arm_joint_targets=dict(action.arm_joint_targets),
                metadata=metadata,
            )
        )
    return output


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
        return NormalizedAction(
            vector=arr.astype(np.float32, copy=True),
            metadata=_normalized_action_metadata(observation),
        )

    if arms_enabled:
        if arr.shape == (ARM_POSE_DOF * len(active),) and active:
            offset = 0
            for side in active:
                base[ACTION_ARM_POSE_SLICES[side]] = arr[offset : offset + ARM_POSE_DOF]
                offset += ARM_POSE_DOF
            return NormalizedAction(vector=base, metadata=_normalized_action_metadata(observation))
        if arr.shape == (ARM_POSE_DOF,) and len(active) == 1:
            base[ACTION_ARM_POSE_SLICES[active[0]]] = arr
            return NormalizedAction(vector=base, metadata=_normalized_action_metadata(observation))

    if hands_enabled:
        if arr.shape == (HAND_DOF * len(active),) and active:
            offset = 0
            for side in active:
                base[ACTION_HAND_SLICES[side]] = arr[offset : offset + HAND_DOF]
                offset += HAND_DOF
            return NormalizedAction(vector=base, metadata=_normalized_action_metadata(observation))
        if arr.shape == (HAND_DOF,) and len(active) == 1:
            base[ACTION_HAND_SLICES[active[0]]] = arr
            return NormalizedAction(vector=base, metadata=_normalized_action_metadata(observation))

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
            return _with_extra_metadata(
                normalize_policy_output(
                    mapping[key],
                    observation,
                    active_sides=active_sides,
                    arms_enabled=arms_enabled,
                    hands_enabled=hands_enabled,
                ),
                mapping.get("metadata") if isinstance(mapping.get("metadata"), Mapping) else None,
            )

    active = _ordered_active_sides(active_sides)
    action_metadata = _normalized_action_metadata(
        observation,
        mapping.get("metadata") if isinstance(mapping.get("metadata"), Mapping) else None,
    )
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
        chunk.append(
            NormalizedAction(
                vector=vector,
                arm_joint_targets=arm_joint_targets,
                metadata=dict(action_metadata),
            )
        )
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
    if hasattr(output, "actions") and hasattr(output, "metadata"):
        return _with_extra_metadata(
            normalize_policy_output(
                getattr(output, "actions"),
                observation,
                active_sides=active,
                arms_enabled=use_arms,
                hands_enabled=use_hands,
            ),
            getattr(output, "metadata"),
        )
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
    max_arm_xyz_step_m: float = 0.08
    max_arm_rot_step_deg: float = 20.0
    max_arm_joint_step_rad: float = 0.20
    # Reserved for future hand q6 rate limiting; current deploy only clips hand q6 to [0, 1].
    max_hand_step: float = 0.0


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

    def _filter_arm_pose_value(
        self,
        *,
        side: str,
        target: np.ndarray,
        observation: DeploymentObservation,
        max_arm_xyz_step_m: float,
        max_arm_rot_step_deg: float,
    ) -> tuple[np.ndarray, bool, list[str], dict[str, Any]]:
        faults: list[str] = []
        metadata: dict[str, Any] = {
            "workspace_clipped": False,
            "xyz_rate_limited": False,
            "rot_rate_limited": False,
        }
        pose = vector_or_none(target, length=ARM_POSE_DOF)
        if pose is None:
            return target, False, [f"{side}_arm_pose_invalid"], metadata

        workspace_min = np.asarray(self.config.workspace_min, dtype=np.float32).reshape(3)
        workspace_max = np.asarray(self.config.workspace_max, dtype=np.float32).reshape(3)
        clipped_xyz = np.clip(pose[:3], workspace_min, workspace_max)
        metadata["workspace_clipped"] = bool(np.any(np.abs(clipped_xyz - pose[:3]) > 1e-12))
        pose = pose.copy()
        pose[:3] = clipped_xyz

        fallback = np.asarray(observation.state[STATE_ARM_POSE_SLICES[side]], dtype=np.float32)
        reference = self._reference_action_slice(observation, ACTION_ARM_POSE_SLICES[side], fallback)
        if float(max_arm_xyz_step_m) > 0.0 and not np.all(np.isfinite(reference[:3])):
            faults.append(f"{side}_arm_pose_reference_unavailable")
        else:
            pose[:3], limited = self._rate_limit_vector(
                pose[:3],
                reference[:3],
                max_step=float(max_arm_xyz_step_m),
            )
            metadata["xyz_rate_limited"] = bool(limited)

        quat = np.asarray(pose[3:7], dtype=float)
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1e-12:
            return target, False, [f"{side}_arm_quat_invalid"], metadata
        pose[3:7] = (quat / quat_norm).astype(np.float32)
        if (
            float(max_arm_rot_step_deg) > 0.0
            and np.all(np.isfinite(reference[3:7]))
            and np.linalg.norm(reference[3:7]) > 1e-12
        ):
            ref_rot = R.from_quat(np.asarray(reference[3:7], dtype=float) / np.linalg.norm(reference[3:7]))
            target_rot = R.from_quat(np.asarray(pose[3:7], dtype=float))
            delta = target_rot * ref_rot.inv()
            angle = float(delta.magnitude())
            max_angle = np.deg2rad(float(max_arm_rot_step_deg))
            if angle > max_angle > 0.0:
                limited_delta = R.from_rotvec(delta.as_rotvec() / angle * max_angle)
                pose[3:7] = (limited_delta * ref_rot).as_quat().astype(np.float32)
                metadata["rot_rate_limited"] = True

        pose_matrix = xyzquat_to_pose_matrix(pose)
        if pose_matrix is None:
            return target, False, [f"{side}_arm_pose_invalid"], metadata
        return pose.astype(np.float32, copy=True), True, faults, metadata

    def _filter_arm_pose(
        self,
        *,
        side: str,
        target: np.ndarray,
        observation: DeploymentObservation,
        raw: np.ndarray,
        max_arm_xyz_step_m: float,
        max_arm_rot_step_deg: float,
    ) -> tuple[np.ndarray, EffectorCommand | None, list[str], dict[str, Any]]:
        pose, valid, faults, metadata = self._filter_arm_pose_value(
            side=side,
            target=target,
            observation=observation,
            max_arm_xyz_step_m=max_arm_xyz_step_m,
            max_arm_rot_step_deg=max_arm_rot_step_deg,
        )
        if not valid:
            return pose, None, faults, metadata
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

    def _filter_arm_qpos_value(
        self,
        *,
        side: str,
        target: np.ndarray,
        observation: DeploymentObservation,
        max_arm_joint_step_rad: float,
    ) -> tuple[np.ndarray, bool, list[str], dict[str, Any]]:
        faults: list[str] = []
        metadata = {"joint_rate_limited": False}
        qpos = vector_or_none(target, length=ARM_JOINT_DOF)
        if qpos is None:
            return target, False, [f"{side}_arm_qpos_invalid"], metadata
        reference = np.asarray(observation.state[STATE_ARM_QPOS_SLICES[side]], dtype=np.float32)
        if float(max_arm_joint_step_rad) > 0.0 and not np.all(np.isfinite(reference)):
            faults.append(f"{side}_arm_qpos_reference_unavailable")
        else:
            qpos, limited = self._rate_limit_vector(
                qpos,
                reference,
                max_step=float(max_arm_joint_step_rad),
            )
            metadata["joint_rate_limited"] = bool(limited)
        return qpos.astype(np.float32, copy=True), True, faults, metadata

    def _filter_arm_qpos(
        self,
        *,
        side: str,
        target: np.ndarray,
        observation: DeploymentObservation,
        max_arm_joint_step_rad: float,
    ) -> tuple[np.ndarray, EffectorCommand | None, list[str], dict[str, Any]]:
        qpos, valid, faults, metadata = self._filter_arm_qpos_value(
            side=side,
            target=target,
            observation=observation,
            max_arm_joint_step_rad=max_arm_joint_step_rad,
        )
        if not valid:
            return qpos, None, faults, metadata
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

    def filter_normalized(
        self,
        action: NormalizedAction | Any,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> FilteredNormalizedAction:
        """Filter an action without creating backend commands.

        This is the canonical deploy path: policy outputs are safety-filtered
        as NormalizedAction values and are then lifted to RobotAction. The older
        filter() method below is retained only for legacy command-path callers.

        Migration target: the value-level safety logic in this method should
        move to robot_motion/deploy with the canonical action pipeline. The
        direct EffectorCommand-producing filter() method should remain archived
        in teleop_runtime or be deleted after all old callers are gone.
        """

        normalized = (
            action
            if isinstance(action, NormalizedAction)
            else NormalizedAction(vector=np.asarray(action, dtype=np.float32).reshape(ACTION_DIM))
        )
        raw = np.asarray(normalized.vector, dtype=np.float32).reshape(-1)
        if raw.shape != (ACTION_DIM,):
            raise ValueError(f"Expected full action shape {(ACTION_DIM,)}, got {raw.shape}.")

        filtered = raw.astype(np.float32, copy=True)
        filtered_arm_joint_targets: dict[str, np.ndarray] = {}
        faults: list[str] = []
        override_config = {}
        if isinstance(normalized.metadata, Mapping):
            raw_overrides = normalized.metadata.get("safety_overrides", {})
            if isinstance(raw_overrides, Mapping):
                override_config = dict(raw_overrides)

        def _override_float(name: str, default: float) -> float:
            if name not in override_config:
                return float(default)
            try:
                return float(override_config[name])
            except (TypeError, ValueError):
                return float(default)

        max_arm_xyz_step_m = _override_float("max_arm_xyz_step_m", self.config.max_arm_xyz_step_m)
        max_arm_rot_step_deg = _override_float("max_arm_rot_step_deg", self.config.max_arm_rot_step_deg)
        max_arm_joint_step_rad = _override_float("max_arm_joint_step_rad", self.config.max_arm_joint_step_rad)
        safety_metadata: dict[str, Any] = {
            "policy_name": policy_name,
            "arm": {},
            "arm_action_specs_by_side": {},
            "hand_rate_limited": {},
            "hand_rate_limit_enabled": False,
            "max_hand_step": float(self.config.max_hand_step),
            "hand_action_specs_by_side": {},
        }

        active = set(self.config.active_sides)
        arm_specs = arm_action_specs_from_metadata(normalized.metadata)
        if not arm_specs:
            arm_specs = arm_action_specs_from_metadata(observation.metadata)
        if not arm_specs:
            arm_specs = arm_action_specs_from_metadata(
                {"arm_action_specs_by_side": deploy_arm_specs_metadata()}
            )
        if self.config.arms_enabled:
            for side in ALL_SIDES:
                if side not in active:
                    continue
                arm_spec = arm_specs.get(side) or ArmActionSpec.tool_tcp(side)
                if side in normalized.arm_joint_targets:
                    qpos, valid, arm_faults, metadata = self._filter_arm_qpos_value(
                        side=side,
                        target=np.asarray(normalized.arm_joint_targets[side], dtype=np.float32),
                        observation=observation,
                        max_arm_joint_step_rad=max_arm_joint_step_rad,
                    )
                    if valid:
                        filtered_arm_joint_targets[side] = qpos.astype(np.float32, copy=True)
                else:
                    pose, valid, arm_faults, metadata = self._filter_arm_pose_value(
                        side=side,
                        target=filtered[ACTION_ARM_POSE_SLICES[side]],
                        observation=observation,
                        max_arm_xyz_step_m=max_arm_xyz_step_m,
                        max_arm_rot_step_deg=max_arm_rot_step_deg,
                    )
                    if valid:
                        filtered[ACTION_ARM_POSE_SLICES[side]] = pose
                faults.extend(arm_faults)
                safety_metadata["arm"][side] = metadata
                safety_metadata["arm_action_specs_by_side"][side] = arm_spec.to_metadata()

        if self.config.hands_enabled:
            hand_specs = hand_action_specs_from_metadata(normalized.metadata)
            if not hand_specs:
                hand_specs = hand_action_specs_from_metadata(observation.metadata)
            if not hand_specs:
                hand_specs = hand_action_specs_from_metadata(
                    {"hand_action_specs_by_side": deploy_hand_specs_metadata()}
                )
            for side in HAND_VECTOR_ORDER:
                if side not in active:
                    continue
                spec = hand_specs.get(side)
                if spec is None:
                    raise ValueError(f"{side}_hand action requires hand_action_specs_by_side metadata.")
                safety_metadata["hand_action_specs_by_side"][side] = spec.to_metadata()
                target_slice = ACTION_HAND_SLICES[side]
                hand_target = vector_or_none(filtered[target_slice], length=HAND_DOF)
                if hand_target is None:
                    faults.append(f"{side}_hand_action_invalid")
                    continue
                if int(spec.slot_count) != HAND_DOF:
                    raise ValueError(
                        f"{side}_hand deploy q6 schema requires HandActionSpec.slot_count={HAND_DOF}, "
                        f"got {spec.slot_count}. Use a hand adapter for non-q6 embodiments."
                    )
                clipped = np.clip(hand_target, 0.0, 1.0).astype(np.float32, copy=True)
                filtered[target_slice] = clipped
                safety_metadata["hand_rate_limited"][side] = False

        filtered_metadata = dict(normalized.metadata)
        filtered_metadata["deploy_safety"] = dict(safety_metadata)
        if safety_metadata["arm_action_specs_by_side"]:
            filtered_metadata["arm_action_specs_by_side"] = dict(
                safety_metadata["arm_action_specs_by_side"]
            )
        if safety_metadata["hand_action_specs_by_side"]:
            filtered_metadata["hand_action_specs_by_side"] = dict(
                safety_metadata["hand_action_specs_by_side"]
            )
            filtered_metadata["hand_active_mask_by_side"] = {
                side: list(spec.get("active_mask", ()))
                for side, spec in safety_metadata["hand_action_specs_by_side"].items()
            }
        filtered_metadata.setdefault("arm_spec_required", True)
        filtered_metadata.setdefault("hand_spec_required", True)

        return FilteredNormalizedAction(
            action=NormalizedAction(
                vector=filtered,
                arm_joint_targets=filtered_arm_joint_targets,
                metadata=filtered_metadata,
            ),
            faults=faults,
            metadata=safety_metadata,
        )

    def filter(
        self,
        action: NormalizedAction | Any,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> DeploymentAction:
        """Legacy direct-command filter.

        New deploy code should use filter_normalized() and resolve the result
        through RobotActionResolver. This method is kept for older tests and
        private callers that still need the archived DeploymentAction command
        path.
        """

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
        override_config = {}
        if isinstance(normalized.metadata, Mapping):
            raw_overrides = normalized.metadata.get("safety_overrides", {})
            if isinstance(raw_overrides, Mapping):
                override_config = dict(raw_overrides)

        def _override_float(name: str, default: float) -> float:
            if name not in override_config:
                return float(default)
            try:
                return float(override_config[name])
            except (TypeError, ValueError):
                return float(default)

        max_arm_xyz_step_m = _override_float("max_arm_xyz_step_m", self.config.max_arm_xyz_step_m)
        max_arm_rot_step_deg = _override_float("max_arm_rot_step_deg", self.config.max_arm_rot_step_deg)
        max_arm_joint_step_rad = _override_float("max_arm_joint_step_rad", self.config.max_arm_joint_step_rad)
        safety_metadata: dict[str, Any] = {
            "policy_name": policy_name,
            "arm": {},
            "arm_action_specs_by_side": {},
            "hand_rate_limited": {},
            "hand_rate_limit_enabled": False,
            "max_hand_step": float(self.config.max_hand_step),
            "hand_action_specs_by_side": {},
        }

        active = set(self.config.active_sides)
        arm_specs = arm_action_specs_from_metadata(normalized.metadata)
        if not arm_specs:
            arm_specs = arm_action_specs_from_metadata(observation.metadata)
        if not arm_specs:
            arm_specs = arm_action_specs_from_metadata(
                {"arm_action_specs_by_side": deploy_arm_specs_metadata()}
            )
        if self.config.arms_enabled:
            for side in ALL_SIDES:
                if side not in active:
                    continue
                arm_spec = arm_specs.get(side) or ArmActionSpec.tool_tcp(side)
                if side in normalized.arm_joint_targets:
                    _, command, arm_faults, metadata = self._filter_arm_qpos(
                        side=side,
                        target=np.asarray(normalized.arm_joint_targets[side], dtype=np.float32),
                        observation=observation,
                        max_arm_joint_step_rad=max_arm_joint_step_rad,
                    )
                else:
                    pose, command, arm_faults, metadata = self._filter_arm_pose(
                        side=side,
                        target=filtered[ACTION_ARM_POSE_SLICES[side]],
                        observation=observation,
                        raw=raw,
                        max_arm_xyz_step_m=max_arm_xyz_step_m,
                        max_arm_rot_step_deg=max_arm_rot_step_deg,
                    )
                    filtered[ACTION_ARM_POSE_SLICES[side]] = pose
                faults.extend(arm_faults)
                safety_metadata["arm"][side] = metadata
                safety_metadata["arm_action_specs_by_side"][side] = arm_spec.to_metadata()
                if command is not None:
                    command.metadata["policy_name"] = policy_name
                    command.metadata["arm_action_spec"] = arm_spec.to_metadata()
                    command.metadata["arm_controlled_frame"] = arm_spec.controlled_frame
                    command.metadata["target_pose_controlled_frame"] = arm_spec.controlled_frame
                    command.metadata["arm_reference_frame"] = arm_spec.reference_frame
                    commands[f"{side}_arm"] = command

        if self.config.hands_enabled:
            hand_specs = hand_action_specs_from_metadata(normalized.metadata)
            if not hand_specs:
                hand_specs = hand_action_specs_from_metadata(observation.metadata)
            if not hand_specs:
                hand_specs = hand_action_specs_from_metadata(
                    {"hand_action_specs_by_side": deploy_hand_specs_metadata()}
                )
            for side in HAND_VECTOR_ORDER:
                if side not in active:
                    continue
                spec = hand_specs.get(side)
                if spec is None:
                    raise ValueError(f"{side}_hand action requires hand_action_specs_by_side metadata.")
                safety_metadata["hand_action_specs_by_side"][side] = spec.to_metadata()
                target_slice = ACTION_HAND_SLICES[side]
                hand_target = vector_or_none(filtered[target_slice], length=HAND_DOF)
                if hand_target is None:
                    faults.append(f"{side}_hand_action_invalid")
                    continue
                if int(spec.slot_count) != HAND_DOF:
                    raise ValueError(
                        f"{side}_hand deploy q6 schema requires HandActionSpec.slot_count={HAND_DOF}, "
                        f"got {spec.slot_count}. Use a hand adapter for non-q6 embodiments."
                    )
                clipped = np.clip(hand_target, 0.0, 1.0)
                filtered[target_slice] = clipped
                safety_metadata["hand_rate_limited"][side] = False
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
                        "hand_action_spec": spec.to_metadata(),
                        "hand_active_mask": list(spec.active_mask or ()),
                        "safety": {
                            "clip_range": [0.0, 1.0],
                            "rate_limited": False,
                            "max_step": 0.0,
                        },
                    },
                )

        return DeploymentAction(
            vector=filtered,
            commands=commands,
            faults=faults,
            metadata=safety_metadata,
        )
