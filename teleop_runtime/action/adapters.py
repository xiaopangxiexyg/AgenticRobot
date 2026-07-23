from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from robot_motion.action import (
    ArmActionSpec,
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_MIXED_HAND16_V1,
    BIMANUAL_MIXED_V1,
    BIMANUAL_QPOS_HAND16_V1,
    BIMANUAL_QPOS_V1,
    HAND16_SLOT_DOF,
    HandActionSpec,
    INSPIRE_HAND_DOF,
    EndEffectorTarget,
    RobotAction,
    arm_action_specs_metadata,
    default_arm_frame,
    effector_name,
    hand_action_specs_metadata,
    split_effector_name,
)
from robot_motion.action.schema import hand_dof_for_action_space
from robot_motion.runtime import EffectorCommand, TeleopCommand


def matrix_to_xyzquat(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Expected a finite 4x4 pose matrix.")
    rot = matrix[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rot[2, 1] - rot[1, 2]) / scale
        qy = (rot[0, 2] - rot[2, 0]) / scale
        qz = (rot[1, 0] - rot[0, 1]) / scale
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rot[0, 1] + rot[1, 0]) / scale
        qz = (rot[0, 2] + rot[2, 0]) / scale
    elif rot[1, 1] > rot[2, 2]:
        scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / scale
        qx = (rot[0, 1] + rot[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rot[1, 2] + rot[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / scale
        qx = (rot[0, 2] + rot[2, 0]) / scale
        qy = (rot[1, 2] + rot[2, 1]) / scale
        qz = 0.25 * scale
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("Pose matrix produced a zero quaternion.")
    xyz = matrix[:3, 3].astype(np.float32, copy=True)
    return np.concatenate((xyz, (quat / norm).astype(np.float32)), dtype=np.float32)


@dataclass(frozen=True)
class TeleopCommandActionAdapterConfig:
    arm_joint_policy: str = "source_pose_when_available"
    pose_metadata_keys: tuple[str, ...] = (
        "target_pose_base",
        "target_pose_tcp_base",
        "solver_output_tcp_pose_base",
        "solver_output_endpoint_pose_base",
        "source_target_pose_base",
    )
    hand_slot_dof: int = INSPIRE_HAND_DOF
    hand_pad_value: float = float("nan")
    arm_specs: Mapping[str, ArmActionSpec | Mapping[str, Any]] | None = None
    hand_specs: Mapping[str, HandActionSpec | Mapping[str, Any]] | None = None
    native_end_effector_action_space: str = "dexteleop_policy_io_v2.2"
    output_action_space: str | None = None
    source: str = "teleop"
    include_command_metadata: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arm_joint_policy not in {"qpos", "source_pose_when_available"}:
            raise ValueError("arm_joint_policy must be 'qpos' or 'source_pose_when_available'.")
        if int(self.hand_slot_dof) not in {INSPIRE_HAND_DOF, HAND16_SLOT_DOF}:
            raise ValueError(f"hand_slot_dof must be {INSPIRE_HAND_DOF} or {HAND16_SLOT_DOF}.")
        object.__setattr__(self, "pose_metadata_keys", tuple(str(k) for k in self.pose_metadata_keys))
        object.__setattr__(self, "hand_slot_dof", int(self.hand_slot_dof))
        object.__setattr__(self, "hand_pad_value", float(self.hand_pad_value))
        arm_specs = {}
        for side, spec in dict(self.arm_specs or {}).items():
            if isinstance(spec, ArmActionSpec):
                resolved_arm = spec
            elif isinstance(spec, Mapping):
                resolved_arm = ArmActionSpec.from_metadata(spec)
            else:
                raise ValueError(f"Unsupported arm spec for side {side!r}: {type(spec).__name__}.")
            arm_specs[resolved_arm.side] = resolved_arm
        object.__setattr__(self, "arm_specs", arm_specs)
        hand_specs = {}
        for side, spec in dict(self.hand_specs or {}).items():
            if isinstance(spec, HandActionSpec):
                resolved = spec
            elif isinstance(spec, Mapping):
                resolved = HandActionSpec.from_metadata(spec)
            else:
                raise ValueError(f"Unsupported hand spec for side {side!r}: {type(spec).__name__}.")
            if int(resolved.slot_count) != int(self.hand_slot_dof):
                raise ValueError(
                    f"HandActionSpec for {side!r} has slot_count={resolved.slot_count}, "
                    f"expected hand_slot_dof={self.hand_slot_dof}."
                )
            hand_specs[resolved.side] = resolved
        object.__setattr__(self, "hand_specs", hand_specs)
        object.__setattr__(self, "source", str(self.source))
        native_action_space = str(self.native_end_effector_action_space).strip()
        if not native_action_space:
            raise ValueError("native_end_effector_action_space must be non-empty.")
        object.__setattr__(self, "native_end_effector_action_space", native_action_space)
        output_action_space = (
            None
            if self.output_action_space is None
            else str(self.output_action_space).strip()
        )
        if self.output_action_space is not None and not output_action_space:
            raise ValueError("output_action_space must be non-empty when provided.")
        object.__setattr__(self, "output_action_space", output_action_space)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _pose_from_metadata(
    command: EffectorCommand,
    *,
    keys: tuple[str, ...],
) -> np.ndarray | None:
    for key in keys:
        value = command.metadata.get(key)
        if value is None:
            continue
        try:
            return matrix_to_xyzquat(value)
        except Exception:
            continue
    return None


def _action_space_for(
    *,
    arm_pose_by_side: Mapping[str, np.ndarray],
    arm_qpos_by_side: Mapping[str, np.ndarray],
    hand_slot_dof: int,
    native_end_effector_targets_by_side: Mapping[str, EndEffectorTarget],
    native_end_effector_action_space: str,
    output_action_space: str | None,
) -> str:
    if output_action_space is not None:
        return str(output_action_space)
    if native_end_effector_targets_by_side:
        return str(native_end_effector_action_space)
    has_pose = bool(arm_pose_by_side)
    has_qpos = bool(arm_qpos_by_side)
    if int(hand_slot_dof) == HAND16_SLOT_DOF:
        if has_pose and has_qpos:
            return BIMANUAL_MIXED_HAND16_V1
        if has_pose:
            return BIMANUAL_EE_POSE_HAND16_V1
        return BIMANUAL_QPOS_HAND16_V1
    if has_pose and has_qpos:
        return BIMANUAL_MIXED_V1
    if has_pose:
        return BIMANUAL_EE_POSE_V1
    return BIMANUAL_QPOS_V1


def _hand_spec_for(
    *,
    side: str,
    cfg: TeleopCommandActionAdapterConfig,
    command_metadata: Mapping[str, Any],
    value_size: int,
    finite_count: int,
) -> HandActionSpec:
    configured = dict(cfg.hand_specs or {}).get(side)
    if configured is not None:
        return configured
    spec_payload = command_metadata.get("hand_action_spec")
    if isinstance(spec_payload, HandActionSpec):
        return spec_payload
    if isinstance(spec_payload, Mapping):
        return HandActionSpec.from_metadata(spec_payload)
    if cfg.hand_slot_dof == INSPIRE_HAND_DOF:
        return HandActionSpec.inspire_tac(side)
    active_dof = int(finite_count if int(value_size) == cfg.hand_slot_dof else value_size)
    return HandActionSpec.q16(side, active_dof=active_dof)


def _arm_spec_for_command(
    command: EffectorCommand,
    *,
    side: str,
    cfg: TeleopCommandActionAdapterConfig,
) -> ArmActionSpec:
    configured = dict(cfg.arm_specs or {}).get(side)
    if configured is not None:
        return configured
    spec_payload = command.metadata.get("arm_action_spec")
    if isinstance(spec_payload, ArmActionSpec):
        return spec_payload
    if isinstance(spec_payload, Mapping):
        return ArmActionSpec.from_metadata(spec_payload)
    controlled_frame = (
        command.metadata.get("arm_controlled_frame")
        or command.metadata.get("target_pose_controlled_frame")
        or command.metadata.get("controlled_frame")
    )
    if controlled_frame is not None:
        return ArmActionSpec(
            side=side,
            controlled_frame=str(controlled_frame),
            reference_frame=str(command.frame_name or default_arm_frame(side)),
        )
    return ArmActionSpec.tool_tcp(
        side,
        reference_frame=str(command.frame_name or default_arm_frame(side)),
    )


def _hand_vector(
    value: Any,
    *,
    side: str,
    cfg: TeleopCommandActionAdapterConfig,
    command_metadata: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, HandActionSpec]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if np.any(np.isinf(arr)):
        raise ValueError(f"{side}_hand joint_vector contains non-finite values.")
    finite_count = int(np.sum(np.isfinite(arr)))
    spec = _hand_spec_for(
        side=side,
        cfg=cfg,
        command_metadata={} if command_metadata is None else command_metadata,
        value_size=int(arr.size),
        finite_count=finite_count,
    )
    if arr.shape == (spec.slot_count,):
        if spec.slot_count == INSPIRE_HAND_DOF and not np.all(np.isfinite(arr)):
            raise ValueError(f"{side}_hand q6 joint_vector must be fully finite.")
        if spec.slot_count == HAND16_SLOT_DOF:
            spec.active_values(arr)
        return arr.astype(np.float32, copy=True), spec
    if arr.shape != (spec.active_dof,):
        raise ValueError(
            f"{side}_hand joint_vector must have shape ({spec.active_dof},) or ({spec.slot_count},), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{side}_hand active joint_vector contains non-finite values.")
    if spec.slot_count == INSPIRE_HAND_DOF:
        return arr.astype(np.float32, copy=True), spec
    if arr.size > cfg.hand_slot_dof:
        raise ValueError(
            f"{side}_hand joint_vector has {arr.size} values, exceeds hand_slot_dof={cfg.hand_slot_dof}."
        )
    padded = np.full((cfg.hand_slot_dof,), float(cfg.hand_pad_value), dtype=np.float32)
    active_indices = [index for index, active in enumerate(spec.active_mask or ()) if active]
    padded[np.asarray(active_indices, dtype=int)] = arr.astype(np.float32, copy=False)
    return padded, spec


def _hand_active_mask(active_dof: int, *, slot_dof: int) -> list[bool]:
    return [index < int(active_dof) for index in range(int(slot_dof))]


def robot_action_from_teleop_command(
    command: TeleopCommand,
    *,
    config: TeleopCommandActionAdapterConfig | None = None,
) -> RobotAction | None:
    cfg = config if config is not None else TeleopCommandActionAdapterConfig()
    arm_pose_by_side: dict[str, np.ndarray] = {}
    arm_qpos_by_side: dict[str, np.ndarray] = {}
    hand_qpos_by_side: dict[str, np.ndarray] = {}
    end_effector_targets_by_side: dict[str, EndEffectorTarget] = {}
    arm_specs_by_side: dict[str, ArmActionSpec] = {}
    hand_specs_by_side: dict[str, HandActionSpec] = {}
    controlled_frame_by_effector: dict[str, str] = {}
    active_effectors: list[str] = []

    for command_name, effector_command in command.commands.items():
        side, kind = split_effector_name(effector_command.effector_name or command_name)
        effector = effector_name(side, kind)
        if kind == "arm":
            arm_spec = _arm_spec_for_command(effector_command, side=side, cfg=cfg)
            arm_specs_by_side[side] = arm_spec
            controlled_frame_by_effector[effector] = arm_spec.controlled_frame
            if effector_command.command_type == "cartesian_pose":
                arm_pose_by_side[side] = matrix_to_xyzquat(effector_command.value)
            elif effector_command.command_type == "joint_vector":
                pose = None
                if cfg.arm_joint_policy == "source_pose_when_available":
                    pose = _pose_from_metadata(effector_command, keys=cfg.pose_metadata_keys)
                if pose is not None:
                    arm_pose_by_side[side] = pose
                else:
                    arm_qpos_by_side[side] = np.asarray(
                        effector_command.value,
                        dtype=np.float32,
                    ).reshape(6)
            else:
                raise ValueError(
                    f"Unsupported arm command_type '{effector_command.command_type}' for {effector}."
                )
        elif kind == "hand":
            if effector_command.command_type in {
                "end_effector_continuous",
                "end_effector_discrete",
            }:
                if side in hand_qpos_by_side:
                    raise ValueError(
                        f"Cannot mix native and joint-vector hand targets for {side}."
                    )
                adapter = str(
                    effector_command.metadata.get("end_effector_adapter") or ""
                ).strip()
                if not adapter:
                    raise ValueError(
                        f"{effector} native target requires end_effector_adapter metadata."
                    )
                names = tuple(
                    str(value)
                    for value in effector_command.metadata.get(
                        "end_effector_target_names", ()
                    )
                )
                end_effector_targets_by_side[side] = EndEffectorTarget(
                    value_type=(
                        "continuous"
                        if effector_command.command_type == "end_effector_continuous"
                        else "discrete"
                    ),
                    values=effector_command.value,
                    adapter=adapter,
                    names=names,
                )
            elif effector_command.command_type != "joint_vector":
                raise ValueError(
                    f"Unsupported hand command_type '{effector_command.command_type}' for {effector}."
                )
            else:
                if side in end_effector_targets_by_side:
                    raise ValueError(
                        f"Cannot mix native and joint-vector hand targets for {side}."
                    )
                hand_qpos_by_side[side], hand_specs_by_side[side] = _hand_vector(
                    effector_command.value,
                    side=side,
                    cfg=cfg,
                    command_metadata=effector_command.metadata,
                )
        active_effectors.append(effector)

    if not active_effectors:
        return None

    if cfg.output_action_space is not None:
        expected_hand_dof = hand_dof_for_action_space(cfg.output_action_space)
        if hand_qpos_by_side and expected_hand_dof != cfg.hand_slot_dof:
            raise ValueError(
                "output_action_space hand-slot width does not match "
                f"hand_slot_dof: {expected_hand_dof}!={cfg.hand_slot_dof}."
            )
        if end_effector_targets_by_side and (
            cfg.output_action_space != cfg.native_end_effector_action_space
        ):
            raise ValueError(
                "Native end-effector targets require output_action_space to match "
                "native_end_effector_action_space."
            )
        pose_spaces = {
            BIMANUAL_EE_POSE_V1,
            BIMANUAL_EE_POSE_HAND16_V1,
            BIMANUAL_MIXED_V1,
            BIMANUAL_MIXED_HAND16_V1,
        }
        qpos_spaces = {
            BIMANUAL_QPOS_V1,
            BIMANUAL_QPOS_HAND16_V1,
            BIMANUAL_MIXED_V1,
            BIMANUAL_MIXED_HAND16_V1,
        }
        known_arm_spaces = pose_spaces | qpos_spaces
        if (
            arm_pose_by_side
            and cfg.output_action_space in known_arm_spaces
            and cfg.output_action_space not in pose_spaces
        ):
            raise ValueError(
                "Arm pose commands are incompatible with output_action_space "
                f"{cfg.output_action_space!r}."
            )
        if (
            arm_qpos_by_side
            and cfg.output_action_space in known_arm_spaces
            and cfg.output_action_space not in qpos_spaces
        ):
            raise ValueError(
                "Arm qpos commands are incompatible with output_action_space "
                f"{cfg.output_action_space!r}."
            )

    metadata: dict[str, Any] = dict(cfg.metadata)
    if arm_specs_by_side:
        metadata["arm_action_specs_by_side"] = arm_action_specs_metadata(arm_specs_by_side)
        metadata["arm_controlled_frame_by_side"] = {
            side: spec.controlled_frame for side, spec in arm_specs_by_side.items()
        }
        metadata["arm_reference_frame_by_side"] = {
            side: spec.reference_frame for side, spec in arm_specs_by_side.items()
        }
        metadata["arm_spec_required"] = True
    if hand_specs_by_side:
        metadata["hand_action_specs_by_side"] = hand_action_specs_metadata(hand_specs_by_side)
        metadata["hand_slot_dof_by_side"] = {
            side: int(spec.slot_count) for side, spec in hand_specs_by_side.items()
        }
        metadata["hand_active_dof_by_side"] = {
            side: int(spec.active_dof) for side, spec in hand_specs_by_side.items()
        }
        metadata["hand_active_mask_by_side"] = {
            side: list(spec.active_mask or ()) for side, spec in hand_specs_by_side.items()
        }
        metadata["hand_invalid_slot_value_by_side"] = {
            side: spec.invalid_slot_value for side, spec in hand_specs_by_side.items()
        }
        metadata["hand_slot_semantics"] = {
            side: spec.semantics for side, spec in hand_specs_by_side.items()
        }
    if cfg.include_command_metadata:
        metadata["teleop_command_metadata"] = dict(command.metadata)
    return RobotAction(
        action_space=_action_space_for(
            arm_pose_by_side=arm_pose_by_side,
            arm_qpos_by_side=arm_qpos_by_side,
            hand_slot_dof=cfg.hand_slot_dof,
            native_end_effector_targets_by_side=end_effector_targets_by_side,
            native_end_effector_action_space=cfg.native_end_effector_action_space,
            output_action_space=cfg.output_action_space,
        ),
        timestamp=command.timestamp,
        arm_pose_by_side=arm_pose_by_side,
        arm_qpos_by_side=arm_qpos_by_side,
        hand_qpos_by_side=hand_qpos_by_side,
        end_effector_targets_by_side=end_effector_targets_by_side,
        active_effectors=tuple(dict.fromkeys(active_effectors)),
        controlled_frame_by_effector=controlled_frame_by_effector,
        source=cfg.source,
        metadata=metadata,
    )
