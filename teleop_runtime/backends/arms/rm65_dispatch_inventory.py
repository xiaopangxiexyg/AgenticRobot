"""Pure M5.8-a inventory for the first arms-only RM65 execution profile.

The builder records concrete call paths in :mod:`rm65` and the policy qpos
startup controller.  It intentionally does not import the vendor SDK, create a
backend, attach a gate, or claim that any listed seam is mediated.
"""

from __future__ import annotations

from robot_motion.action.execution_manifest import (
    M6_CANDIDATE_POLICY_ARM_QPOS_V1,
    ControlExecutionManifest,
)
from robot_motion.runtime.dispatch_inventory import (
    DispatchSeamInventoryAssessment,
    DispatchSeamInventoryEntry,
    DispatchSeamInventoryManifest,
    assess_dispatch_seam_inventory,
)


RM65_ARMS_ONLY_DISPATCH_INVENTORY_V1 = (
    "m6_candidate_policy_arm_qpos_v1.rm65_dispatch_inventory_v1"
)

RM65_ARMS_ONLY_REQUIRED_SEAM_ROLES_V1 = (
    "close_disconnect",
    "close_stop_pre_sdk",
    "emergency_stop_pre_sdk",
    "fault_stop_pre_sdk",
    "force_position_start_pre_sdk",
    "hold_stop_pre_sdk",
    "lifecycle_home_pre_sdk",
    "lifecycle_home_target_admission",
    "lifecycle_rearm_state_transition",
    "motion_pre_sdk",
    "motion_target_admission",
    "motion_worker_dequeue",
    "position_stream_disengage_stop_pre_sdk",
    "quick_stop_operation_admission",
    "quick_stop_tick_pre_sdk",
    "startup_rollback_stop_pre_sdk",
)

_RM65_BACKEND = "teleop_runtime.backends.arms.rm65.RM65ArmBackend"
_QPOS_INIT = "teleop_runtime.deploy.policy_qpos_init"
_POSITION_EXECUTOR = "teleop_runtime.position_control.deploy_executor.PositionDeployExecutor"


def _entry(
    manifest: ControlExecutionManifest,
    *,
    seam_id: str,
    seam_role: str,
    implementation_path: str,
    call_stage: str,
    path_kind: str,
    permit_kind: str,
    typed_operation: str | None,
    authorization_requirement: str,
    required_checks: tuple[str, ...],
    retry_behavior: str,
    activation_condition: str,
    required_for_legacy_mediation: bool = True,
    integration_status: str = "inventoried_unwrapped",
) -> DispatchSeamInventoryEntry:
    return DispatchSeamInventoryEntry.create(
        seam_id=seam_id,
        seam_role=seam_role,
        implementation_path=implementation_path,
        call_stage=call_stage,
        path_kind=path_kind,
        permit_kind=permit_kind,
        typed_operation=typed_operation,
        authorization_requirement=authorization_requirement,
        required_checks=required_checks,
        retry_behavior=retry_behavior,
        activation_condition=activation_condition,
        enabled_profile_ids=(manifest.manifest_id,),
        required_for_legacy_mediation=required_for_legacy_mediation,
        integration_status=integration_status,
        evidence_ids=(f"code:{implementation_path}",),
    )


def build_rm65_arms_only_dispatch_inventory(
    manifest: ControlExecutionManifest,
) -> DispatchSeamInventoryManifest:
    """Inventory exact RM65 seams without installing production mediation."""

    if not isinstance(manifest, ControlExecutionManifest):
        raise TypeError("manifest must be a ControlExecutionManifest.")
    if manifest.manifest_id != M6_CANDIDATE_POLICY_ARM_QPOS_V1:
        raise ValueError("RM65 arms-only inventory requires the first M6 candidate manifest.")
    if manifest.motion_authority_sources != ("policy",):
        raise ValueError("RM65 arms-only inventory requires policy-only motion authority.")
    if manifest.controlled_components != ("left_arm", "right_arm"):
        raise ValueError("RM65 arms-only inventory requires ordered left/right arm components.")
    if manifest.selection_groups != ("arms",):
        raise ValueError("RM65 arms-only inventory requires the coupled arms group.")

    common_motion_checks = (
        "typed_control_context",
        "owner_token",
        "owner_generation",
        "authority_epoch",
        "execution_epoch",
        "group_generation",
    )
    final_motion_checks = common_motion_checks + ("one_shot_per_call",)
    final_safety_checks = (
        "typed_control_context",
        "execution_epoch",
        "active_safety_operation",
        "one_shot_per_call",
    )
    lifecycle_checks = (
        "typed_control_context",
        "execution_epoch",
        "lifecycle_authorization",
        "one_shot_per_call",
    )

    entries = (
        _entry(
            manifest,
            seam_id="rm65.motion.target_admission.send_command",
            seam_role="motion_target_admission",
            implementation_path=f"{_RM65_BACKEND}.send_command",
            call_stage="logical_target_admission",
            path_kind="direct",
            permit_kind="motion",
            typed_operation=None,
            authorization_requirement="typed_context",
            required_checks=("typed_control_context", "execution_epoch", "group_generation"),
            retry_behavior="single logical admission; no permit may be reused downstream",
            activation_condition="normal policy qpos command",
        ),
        _entry(
            manifest,
            seam_id="rm65.motion.background.worker_dequeue",
            seam_role="motion_worker_dequeue",
            implementation_path=f"{_RM65_BACKEND}._background_stream_loop_impl",
            call_stage="worker_dequeue",
            path_kind="background_worker",
            permit_kind="motion",
            typed_operation=None,
            authorization_requirement="epoch_guard",
            required_checks=(
                "typed_control_context",
                "authority_epoch",
                "execution_epoch",
                "group_generation",
                "worker_drain",
            ),
            retry_behavior="every planner tick rechecks live work identity",
            activation_condition="background_streaming_enabled",
        ),
        _entry(
            manifest,
            seam_id="rm65.motion.pre_sdk.send_joint_target",
            seam_role="motion_pre_sdk",
            implementation_path=f"{_RM65_BACKEND}._send_joint_target",
            call_stage="pre_sdk_call",
            path_kind="direct",
            permit_kind="motion",
            typed_operation=None,
            authorization_requirement="one_shot_dispatch_permit",
            required_checks=final_motion_checks,
            retry_behavior="one permit for each Force_Position_Move_Joint or Movej_CANFD call",
            activation_condition="normal direct or background qpos dispatch",
            integration_status="pure_wrapper_validated",
        ),
        _entry(
            manifest,
            seam_id="rm65.lifecycle.home.target_admission",
            seam_role="lifecycle_home_target_admission",
            implementation_path=f"{_QPOS_INIT}._send_qpos_targets",
            call_stage="logical_target_admission",
            path_kind="lifecycle_control",
            permit_kind="lifecycle_control",
            typed_operation="home",
            authorization_requirement="lifecycle_state_machine_then_one_shot",
            required_checks=(
                "typed_control_context",
                "execution_epoch",
                "lifecycle_authorization",
            ),
            retry_behavior="each qpos-init step is a new lifecycle-authorized motion lineage",
            activation_condition="operator-authorized policy qpos initialization/home",
        ),
        _entry(
            manifest,
            seam_id="rm65.lifecycle.home.pre_sdk.send_joint_target",
            seam_role="lifecycle_home_pre_sdk",
            implementation_path=f"{_RM65_BACKEND}._send_joint_target",
            call_stage="pre_sdk_call",
            path_kind="lifecycle_control",
            permit_kind="lifecycle_control",
            typed_operation="home",
            authorization_requirement="lifecycle_state_machine_then_one_shot",
            required_checks=lifecycle_checks,
            retry_behavior="one lifecycle permit per actual home SDK target",
            activation_condition="qpos initialization/home command reaches RM65 transport",
            integration_status="pure_wrapper_validated",
        ),
        _entry(
            manifest,
            seam_id="rm65.lifecycle.rearm.position_executor",
            seam_role="lifecycle_rearm_state_transition",
            implementation_path=f"{_POSITION_EXECUTOR}.manual_rearm",
            call_stage="operation_admission",
            path_kind="lifecycle_control",
            permit_kind="lifecycle_control",
            typed_operation="rearm",
            authorization_requirement="lifecycle_state_guard",
            required_checks=(
                "typed_control_context",
                "execution_epoch",
                "lifecycle_authorization",
            ),
            retry_behavior="single state transition after fresh measured-state evidence",
            activation_condition="operator-authorized recovery/rearm",
        ),
        _entry(
            manifest,
            seam_id="rm65.lifecycle.force_position.start",
            seam_role="force_position_start_pre_sdk",
            implementation_path=f"{_RM65_BACKEND}._start_force_position_control",
            call_stage="pre_sdk_call",
            path_kind="lifecycle_control",
            permit_kind="lifecycle_control",
            typed_operation="startup",
            authorization_requirement="lifecycle_state_machine_then_one_shot",
            required_checks=lifecycle_checks,
            retry_behavior="Clear_Force_Data and Start_Force_Position_Move require separate calls",
            activation_condition="force_position_control enabled for the bound backend",
            integration_status="pure_wrapper_validated",
        ),
        _entry(
            manifest,
            seam_id="rm65.safety.quick_stop.operation_admission",
            seam_role="quick_stop_operation_admission",
            implementation_path=f"{_RM65_BACKEND}._activate_stream_quick_stop_locked",
            call_stage="operation_admission",
            path_kind="safety_control",
            permit_kind="safety_control",
            typed_operation="rm65_quick_stop",
            authorization_requirement="bounded_operation_then_one_shot",
            required_checks=(
                "typed_control_context",
                "execution_epoch",
                "active_safety_operation",
            ),
            retry_behavior="one bounded operation; it does not authorize SDK ticks",
            activation_condition="background trajectory planner can queue quick-stop",
        ),
        _entry(
            manifest,
            seam_id="rm65.safety.quick_stop.tick.pre_sdk",
            seam_role="quick_stop_tick_pre_sdk",
            implementation_path=f"{_RM65_BACKEND}._send_joint_target",
            call_stage="pre_sdk_call",
            path_kind="background_worker",
            permit_kind="safety_control",
            typed_operation="rm65_quick_stop_tick",
            authorization_requirement="bounded_operation_then_one_shot",
            required_checks=final_safety_checks,
            retry_behavior="each quick-stop planner/SDK tick requires new evidence and permit",
            activation_condition="active admitted RM65 quick-stop operation",
            integration_status="pure_wrapper_validated",
        ),
        _entry(
            manifest,
            seam_id="rm65.safety.hold.direct_stop",
            seam_role="hold_stop_pre_sdk",
            implementation_path=f"{_RM65_BACKEND}.hold_motion",
            call_stage="pre_sdk_call",
            path_kind="safety_control",
            permit_kind="safety_control",
            typed_operation="hold",
            authorization_requirement="one_shot_dispatch_permit",
            required_checks=(
                "typed_control_context",
                "execution_epoch",
                "one_shot_per_call",
            ),
            retry_behavior="chosen Arm_Slow_Stop/Arm_Stop call requires one permit",
            activation_condition="quick-stop cannot be queued and a hardware arm exists",
            integration_status="pure_wrapper_validated",
        ),
        _entry(
            manifest,
            seam_id="rm65.safety.position_stream_disengage.stop",
            seam_role="position_stream_disengage_stop_pre_sdk",
            implementation_path=f"{_RM65_BACKEND}.disengage_position_command_stream",
            call_stage="pre_sdk_call",
            path_kind="safety_control",
            permit_kind="safety_control",
            typed_operation="disable_motion",
            authorization_requirement="one_shot_dispatch_permit",
            required_checks=(
                "typed_control_context",
                "execution_epoch",
                "one_shot_per_call",
                "worker_drain",
            ),
            retry_behavior="chosen Arm_Slow_Stop/Arm_Stop call requires one permit",
            activation_condition="operator leaves persistent position interaction",
            integration_status="pure_wrapper_validated",
        ),
        _entry(
            manifest,
            seam_id="rm65.safety.background_fault.stop_fallback_loop",
            seam_role="fault_stop_pre_sdk",
            implementation_path=f"{_RM65_BACKEND}._attempt_background_fault_stop",
            call_stage="pre_sdk_call",
            path_kind="safety_control",
            permit_kind="safety_control",
            typed_operation="stop",
            authorization_requirement="one_shot_dispatch_permit",
            required_checks=(
                "typed_control_context",
                "execution_epoch",
                "one_shot_per_call",
            ),
            retry_behavior="each Arm_Slow_Stop/Arm_Stop fallback attempt requires a new permit",
            activation_condition="background sender fault latched",
            integration_status="pure_wrapper_validated",
        ),
        _entry(
            manifest,
            seam_id="rm65.safety.emergency_stop.force_and_arm",
            seam_role="emergency_stop_pre_sdk",
            implementation_path=f"{_RM65_BACKEND}.emergency_stop",
            call_stage="pre_sdk_call",
            path_kind="safety_control",
            permit_kind="safety_control",
            typed_operation="stop",
            authorization_requirement="one_shot_dispatch_permit",
            required_checks=(
                "typed_control_context",
                "execution_epoch",
                "one_shot_per_call",
                "worker_drain",
            ),
            retry_behavior="Stop_Force_Position_Move and Arm_Stop fallback are distinct calls",
            activation_condition="emergency stop requested",
            integration_status="pure_wrapper_validated",
        ),
        _entry(
            manifest,
            seam_id="rm65.cleanup.startup_rollback.force_stop",
            seam_role="startup_rollback_stop_pre_sdk",
            implementation_path=f"{_RM65_BACKEND}._rollback_failed_start",
            call_stage="pre_sdk_call",
            path_kind="cleanup",
            permit_kind="safety_control",
            typed_operation="stop",
            authorization_requirement="one_shot_dispatch_permit",
            required_checks=(
                "typed_control_context",
                "execution_epoch",
                "one_shot_per_call",
            ),
            retry_behavior="best-effort force-position stop before uninit",
            activation_condition="partially completed backend startup fails",
            integration_status="pure_wrapper_validated",
        ),
        _entry(
            manifest,
            seam_id="rm65.cleanup.close.stop_before_disconnect",
            seam_role="close_stop_pre_sdk",
            implementation_path=f"{_RM65_BACKEND}.close",
            call_stage="pre_sdk_call",
            path_kind="cleanup",
            permit_kind="safety_control",
            typed_operation="stop",
            authorization_requirement="one_shot_dispatch_permit",
            required_checks=(
                "typed_control_context",
                "execution_epoch",
                "one_shot_per_call",
                "worker_drain",
            ),
            retry_behavior="force-position stop and arm stop are distinct calls",
            activation_condition="backend closes after motion or active force-position mode",
            integration_status="pure_wrapper_validated",
        ),
        _entry(
            manifest,
            seam_id="rm65.cleanup.close.disconnect",
            seam_role="close_disconnect",
            implementation_path=f"{_RM65_BACKEND}.close",
            call_stage="disconnect_call",
            path_kind="cleanup",
            permit_kind="not_applicable",
            typed_operation=None,
            authorization_requirement="lifecycle_state_guard",
            required_checks=("typed_control_context", "worker_drain"),
            retry_behavior="RM_API_UnInit then Arm_Socket_Close; no motion permit",
            activation_condition="worker drained and stop-before-disconnect completed or not required",
        ),
    )
    return DispatchSeamInventoryManifest.create(
        inventory_id=RM65_ARMS_ONLY_DISPATCH_INVENTORY_V1,
        inventory_version="1.1.0",
        execution_manifest_id=manifest.manifest_id,
        execution_manifest_hash=manifest.semantic_content_hash,
        entries=entries,
    )


def assess_rm65_arms_only_dispatch_inventory(
    inventory: DispatchSeamInventoryManifest,
) -> DispatchSeamInventoryAssessment:
    return assess_dispatch_seam_inventory(
        inventory,
        required_roles=RM65_ARMS_ONLY_REQUIRED_SEAM_ROLES_V1,
    )
