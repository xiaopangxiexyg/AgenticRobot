"""Runtime component aggregate used by the real-teleop entrypoint.

The dataclass only describes already-created components.  Construction,
device startup, and command dispatch remain in the legacy coordinator.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

# ``from __future__ import annotations`` keeps this aggregate import-light;
# the legacy entrypoint still has the concrete runtime classes in scope when
# resolving annotations or constructing the object.
@dataclass(frozen=True)
class RealTeleopComponents:
    session: TeleopSession
    active_sides: tuple[str, ...]
    left_command_backend: RM65ArmBackend | RM65ImpedanceArmBackend | None
    right_command_backend: RM65ArmBackend | RM65ImpedanceArmBackend | None
    left_preview_backend: RM65ArmBackend
    right_preview_backend: RM65ArmBackend
    left_state_backend: RM65ArmBackend | None
    right_state_backend: RM65ArmBackend | None
    left_hand_backend: InspireTacHandBackend | DMTacGripperBackend | None
    right_hand_backend: InspireTacHandBackend | DMTacGripperBackend | None
    initial_alignment: TeleopInitialAlignment | None
    loaded_initial_alignment: TeleopInitialAlignment | None
    logical_arm_ip_mapping: dict[str, str]
    collision_scene_profile: DualArmSceneProfile
    dual_arm_retargeter: DualArmCollisionAwareRetargeter | None
    collision_guard_requested: bool
    collision_guard_setup_error: str | None
    renderer: SapienArmHandRenderer | None
    keyboard_source: KeyboardButtonSource
    left_retargeter: PoseToArmRetargeter
    right_retargeter: PoseToArmRetargeter
    left_hand_retargeter: JointsToDexHandRetargeter | None
    right_hand_retargeter: JointsToDexHandRetargeter | None
    left_gripper_retargeter: ScalarToGripperRetargeter | None
    right_gripper_retargeter: ScalarToGripperRetargeter | None
    left_hand_control_via_rig: bool
    right_hand_control_via_rig: bool
    hand_backend_mode: str
    gripper_input_source: str
    gripper_policy: dict[str, object]
    joint_distance_weights: np.ndarray
    arm_state_read_executor: ThreadPoolExecutor | None
    arm_loop_step_dt_tracker: dict[str, float | None]
    human_shadow_frontend: HumanShadowFrontend | None
    temporary_safety_limiter: TemporaryDualArmSafetyLimiter | None
    fix_arm_controller: FixArmCommandController | None
    init_arm_controller: InitArmCommandController | None
    position_reference_publisher: PositionReferencePublisher | None

__all__ = ["RealTeleopComponents"]
