"""Composition of real-teleop command filters, guards, and fault filters."""

from __future__ import annotations

from collections.abc import Sequence

from robot_motion.runtime import TeleopCommand

from ..backends.grippers import DMTacGripperBackend
from ..backends.hands import InspireTacHandBackend
from ..retargeters import DualArmCollisionAwareRetargeter, ScalarToGripperRetargeter
from ..special_tasks.fix_arm import FixArmCommandController
from ..special_tasks.init_arm import InitArmCommandController
from .fix_arm_collision_guard import fixed_arm_final_collision_fault
from .temporary_dual_arm_safety import TemporaryDualArmSafetyLimiter

def _compose_real_command_filter(
    *,
    temporary_safety_limiter: TemporaryDualArmSafetyLimiter | None,
    fix_arm_controller: FixArmCommandController | None,
    init_arm_controller: InitArmCommandController | None,
):
    if (
        temporary_safety_limiter is None
        and fix_arm_controller is None
        and init_arm_controller is None
    ):
        return None

    def filter_command(command: TeleopCommand, observation, now: float) -> TeleopCommand:
        filtered = command
        if init_arm_controller is not None:
            filtered = init_arm_controller.filter_command(filtered, observation, now)
        if temporary_safety_limiter is not None:
            filtered = temporary_safety_limiter.filter_command(filtered, observation, now)
        if fix_arm_controller is not None:
            filtered = fix_arm_controller.filter_command(filtered, observation, now)
        if temporary_safety_limiter is not None and (
            fix_arm_controller is not None or init_arm_controller is not None
        ):
            temporary_safety_limiter.sync_from_command(filtered)
        return filtered

    return filter_command


def _compose_real_command_guard(
    *,
    fix_arm_controller: FixArmCommandController | None,
    init_arm_controller: InitArmCommandController | None,
    dual_arm_retargeter: DualArmCollisionAwareRetargeter | None,
):
    if dual_arm_retargeter is None:
        return None

    def guard(command: TeleopCommand, _observation, now: float):
        if init_arm_controller is not None:
            fault = init_arm_controller.qpos_final_collision_fault(command, now=now)
            if fault is not None:
                return fault
        if fix_arm_controller is not None:
            return fixed_arm_final_collision_fault(
                command,
                fix_arm_controller=fix_arm_controller,
                dual_arm_retargeter=dual_arm_retargeter,
                now=now,
            )
        return None

    return guard


def _compose_real_fault_filter(
    *,
    fix_arm_controller: FixArmCommandController | None,
    init_arm_controller: InitArmCommandController | None,
):
    if fix_arm_controller is None and init_arm_controller is None:
        return None

    def filter_faults(faults, channels, now: float):
        filtered = list(faults)
        if fix_arm_controller is not None:
            filtered = fix_arm_controller.filter_faults(filtered, channels, now)
        if init_arm_controller is not None:
            filtered = init_arm_controller.filter_faults(filtered, channels, now)
        return filtered

    return filter_faults




def _fix_arm_required_effectors_by_side(
    *,
    active_sides: Sequence[str],
    left_hand_control_via_rig: bool,
    right_hand_control_via_rig: bool,
    left_gripper_retargeter: ScalarToGripperRetargeter | None,
    right_gripper_retargeter: ScalarToGripperRetargeter | None,
    left_hand_backend: InspireTacHandBackend | DMTacGripperBackend | None,
    right_hand_backend: InspireTacHandBackend | DMTacGripperBackend | None,
) -> dict[str, tuple[str, ...]]:
    required: dict[str, tuple[str, ...]] = {}
    for side in active_sides:
        names = [f"{side}_arm"]
        if side == "left":
            hand_controlled = bool(
                left_hand_control_via_rig
                or (left_gripper_retargeter is not None and left_hand_backend is not None)
            )
        else:
            hand_controlled = bool(
                right_hand_control_via_rig
                or (right_gripper_retargeter is not None and right_hand_backend is not None)
            )
        if hand_controlled:
            names.append(f"{side}_hand")
        required[str(side)] = tuple(names)
    return required













def _filter_manual_real_command(
    components: "RealTeleopComponents",
    command: TeleopCommand | None,
    observation,
    now: float,
) -> TeleopCommand | None:
    if command is None or observation is None:
        return command
    filter_command = _compose_real_command_filter(
        temporary_safety_limiter=components.temporary_safety_limiter,
        fix_arm_controller=components.fix_arm_controller,
        init_arm_controller=getattr(components, "init_arm_controller", None),
    )
    if filter_command is None:
        return command
    return filter_command(command, observation, now)


__all__ = [
    "_compose_real_command_filter",
    "_compose_real_command_guard",
    "_compose_real_fault_filter",
    "_fix_arm_required_effectors_by_side",
    "_filter_manual_real_command",
]
