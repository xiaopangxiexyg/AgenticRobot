from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import (
    ARM_POSE_DOF,
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_MIXED_HAND16_V1,
    BIMANUAL_MIXED_V1,
    BIMANUAL_QPOS_HAND16_V1,
    BIMANUAL_QPOS_V1,
    HAND16_SLOT_DOF,
    INSPIRE_HAND_DOF,
    RM65_ARM_DOF,
    RobotAction,
    effector_name,
)


# Migration plan:
# 1. Move RobotAction and its hardware-free helpers from teleop_runtime/action
#    to robot_motion/action once sim deploy imports the same action contract.
# 2. Move this vector adapter with RobotAction, because it only depends on
#    numeric action layout/specs and has no teleop_runtime backend dependency.
# 3. Keep CLI policy adapters, cameras, real hardware dispatch, and legacy
#    teleop-specific compatibility in teleop_runtime. Those are launcher
#    concerns, not robot_motion common motion-control concerns.
# 4. After the move, teleop_runtime and sim should import the same
#    robot_motion.action vector -> RobotAction adapter before dispatching
#    TeleopCommand through their own runtime backends.
@dataclass(frozen=True)
class VectorActionLayout:
    """Layout used to lift dense action vectors into RobotAction.

    This module is intentionally independent from deploy/replay backend code.
    It is the current staging adapter for policy/replay vectors before they
    become canonical RobotAction values. Its target home is robot_motion/action,
    together with RobotAction, because this conversion must be shared by sim
    deploy, real deploy, and dataset replay.
    """

    action_dim: int
    arm_pose_slices: Mapping[str, slice]
    hand_slices: Mapping[str, slice]
    arm_pose_dof: int = ARM_POSE_DOF
    arm_joint_dof: int = RM65_ARM_DOF
    hand_dof: int = INSPIRE_HAND_DOF
    side_order: tuple[str, ...] = ("left", "right")

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_dim", int(self.action_dim))
        object.__setattr__(self, "arm_pose_dof", int(self.arm_pose_dof))
        object.__setattr__(self, "arm_joint_dof", int(self.arm_joint_dof))
        object.__setattr__(self, "hand_dof", int(self.hand_dof))
        object.__setattr__(self, "side_order", tuple(str(side) for side in self.side_order))


@dataclass(frozen=True)
class VectorRobotActionBuildResult:
    robot_action: RobotAction | None
    faults: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "faults", tuple(str(fault) for fault in self.faults))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _finite_vector(value: Any, *, length: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if vector.shape != (int(length),) or not np.all(np.isfinite(vector)):
        return None
    return vector.astype(np.float32, copy=True)


def _action_space_for(
    *,
    arm_pose_by_side: Mapping[str, np.ndarray],
    arm_qpos_by_side: Mapping[str, np.ndarray],
    hand_dof: int,
) -> str:
    has_pose = bool(arm_pose_by_side)
    has_qpos = bool(arm_qpos_by_side)
    if int(hand_dof) == HAND16_SLOT_DOF:
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


def robot_action_from_vector_action(
    *,
    vector: Any,
    arm_joint_targets: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    active_sides: Sequence[str] = ("left", "right"),
    arms_enabled: bool = True,
    hands_enabled: bool = False,
    layout: VectorActionLayout,
    timestamp: float | None = None,
    source: str = "",
) -> VectorRobotActionBuildResult:
    action_vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if action_vector.shape != (int(layout.action_dim),):
        raise ValueError(
            f"Expected action vector shape {(int(layout.action_dim),)}, got {action_vector.shape}."
        )

    active = {str(side) for side in active_sides}
    metadata_dict = {} if metadata is None else dict(metadata)
    arm_joint_targets = {} if arm_joint_targets is None else dict(arm_joint_targets)
    arm_pose_by_side: dict[str, np.ndarray] = {}
    arm_qpos_by_side: dict[str, np.ndarray] = {}
    hand_qpos_by_side: dict[str, np.ndarray] = {}
    faults: list[str] = []

    for side in layout.side_order:
        if side not in active:
            continue
        if arms_enabled:
            if side in arm_joint_targets:
                qpos = _finite_vector(
                    arm_joint_targets[side],
                    length=int(layout.arm_joint_dof),
                )
                if qpos is None:
                    faults.append(f"{side}_arm_qpos_invalid")
                else:
                    arm_qpos_by_side[side] = qpos
            else:
                pose_slice = layout.arm_pose_slices.get(side)
                pose = (
                    _finite_vector(action_vector[pose_slice], length=int(layout.arm_pose_dof))
                    if pose_slice is not None
                    else None
                )
                if pose is None:
                    faults.append(f"{side}_arm_pose_invalid")
                else:
                    arm_pose_by_side[side] = pose

        if hands_enabled:
            hand_slice = layout.hand_slices.get(side)
            hand = (
                _finite_vector(action_vector[hand_slice], length=int(layout.hand_dof))
                if hand_slice is not None
                else None
            )
            if hand is None:
                faults.append(f"{side}_hand_action_invalid")
            else:
                hand_qpos_by_side[side] = hand

    active_effectors: list[str] = []
    for side in layout.side_order:
        if side in arm_pose_by_side or side in arm_qpos_by_side:
            active_effectors.append(effector_name(side, "arm"))
        if side in hand_qpos_by_side:
            active_effectors.append(effector_name(side, "hand"))

    if not active_effectors:
        return VectorRobotActionBuildResult(
            robot_action=None,
            faults=tuple(faults or ["action_no_valid_effectors"]),
            metadata={"active_effectors": ()},
        )

    robot_action = RobotAction(
        action_space=_action_space_for(
            arm_pose_by_side=arm_pose_by_side,
            arm_qpos_by_side=arm_qpos_by_side,
            hand_dof=int(layout.hand_dof),
        ),
        timestamp=timestamp,
        arm_pose_by_side=arm_pose_by_side,
        arm_qpos_by_side=arm_qpos_by_side,
        hand_qpos_by_side=hand_qpos_by_side,
        active_effectors=tuple(active_effectors),
        source=str(source),
        metadata=metadata_dict,
    )
    return VectorRobotActionBuildResult(
        robot_action=robot_action,
        faults=tuple(faults),
        metadata={"active_effectors": tuple(active_effectors)},
    )
