from __future__ import annotations

import pytest

from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    DecisionScheduleSpec,
    build_m6_candidate_policy_arm_qpos_manifest,
    build_scheduled_reference_segment_activation_spec,
)
from robot_motion.position_control import ExecutionCapabilityProfile
from robot_motion.runtime import DispatchSeamInventoryManifest
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend
from teleop_runtime.backends.arms.rm65_dispatch_inventory import (
    RM65_ARMS_ONLY_REQUIRED_SEAM_ROLES_V1,
    assess_rm65_arms_only_dispatch_inventory,
    build_rm65_arms_only_dispatch_inventory,
)
from teleop_runtime.deploy.policy_qpos_init import DeployPolicyQposInitController
from teleop_runtime.position_control.deploy_executor import PositionDeployExecutor


def _execution_manifest():
    catalog = ActionComponentCatalog(
        catalog_id="rm65-arms-only-inventory-v1",
        action_space="bimanual_qpos_v1",
        components=(
            ActionComponentSpec("left_arm", "arms", "left_arm", coupled=True),
            ActionComponentSpec("right_arm", "arms", "right_arm", coupled=True),
        ),
    )
    capability = ExecutionCapabilityProfile(
        profile_id="rm65-arms-only-inventory-capability-v1",
        validated_max_selected_keyframe_hz=60.0,
        configured_selected_keyframe_hz=30.0,
        validated_max_coupled_ik_hz=None,
        configured_coupled_ik_hz=None,
        reference_resample_hz=60.0,
        command_hz=120.0,
        backend_servo_hz=1000.0,
        validated_action_spaces=("bimanual_qpos_v1",),
        validated_selection_groups=("arms",),
        supports_local_group_stop=False,
        acceptance_evidence_id="scoped-scheduled-qpos-evidence-v1",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=1.0 / 60.0,
        end_to_end_control_budget_sec=1.0 / 30.0,
        required_cancellation_ack_levels=("software_queue",),
    )
    schedule = DecisionScheduleSpec(
        spec_id="rm65-arms-only-inventory-schedule-v1",
        decision_hz=30.0,
        target_lead_time_sec=0.05,
    )
    activation = build_scheduled_reference_segment_activation_spec(
        schedule,
        capability,
        minimum_activation_margin_sec=1.0 / 60.0,
        activation_margin_evidence_id="one-reference-tick-margin-v1",
    )
    return build_m6_candidate_policy_arm_qpos_manifest(
        catalog=catalog,
        capability=capability,
        activation=activation,
    )


def test_candidate_inventory_is_complete_but_explicitly_unmediated() -> None:
    inventory = build_rm65_arms_only_dispatch_inventory(_execution_manifest())
    assessment = assess_rm65_arms_only_dispatch_inventory(inventory)

    assert assessment.inventory_structurally_complete
    assert assessment.missing_roles == ()
    assert not assessment.legacy_mediation_ready
    assert not assessment.confers_execution_authorization
    assert assessment.unmediated_seam_ids
    statuses = {entry.seam_id: entry.integration_status for entry in inventory.entries}
    assert {
        seam_id
        for seam_id, status in statuses.items()
        if status == "pure_wrapper_validated"
    } == {
        "rm65.motion.pre_sdk.send_joint_target",
        "rm65.lifecycle.home.pre_sdk.send_joint_target",
        "rm65.safety.quick_stop.tick.pre_sdk",
        "rm65.lifecycle.force_position.start",
        "rm65.safety.hold.direct_stop",
        "rm65.safety.position_stream_disengage.stop",
        "rm65.safety.background_fault.stop_fallback_loop",
        "rm65.safety.emergency_stop.force_and_arm",
        "rm65.cleanup.startup_rollback.force_stop",
        "rm65.cleanup.close.stop_before_disconnect",
    }
    assert all(
        status in {"inventoried_unwrapped", "pure_wrapper_validated"}
        for status in statuses.values()
    )
    assert tuple(entry.seam_id for entry in inventory.entries) == tuple(
        sorted(entry.seam_id for entry in inventory.entries)
    )


def test_shared_send_joint_target_path_has_distinct_motion_safety_and_home_semantics() -> None:
    inventory = build_rm65_arms_only_dispatch_inventory(_execution_manifest())
    shared = tuple(
        entry
        for entry in inventory.entries
        if entry.implementation_path.endswith("RM65ArmBackend._send_joint_target")
    )

    assert {(entry.permit_kind, entry.typed_operation) for entry in shared} == {
        ("motion", None),
        ("safety_control", "rm65_quick_stop_tick"),
        ("lifecycle_control", "home"),
    }
    assert all("one_shot_per_call" in entry.required_checks for entry in shared)


def test_inventory_records_existing_methods_without_initializing_hardware() -> None:
    assert hasattr(RM65ArmBackend, "send_command")
    assert hasattr(RM65ArmBackend, "_background_stream_loop_impl")
    assert hasattr(RM65ArmBackend, "_send_joint_target")
    assert hasattr(RM65ArmBackend, "_attempt_background_fault_stop")
    assert hasattr(RM65ArmBackend, "hold_motion")
    assert hasattr(RM65ArmBackend, "disengage_position_command_stream")
    assert hasattr(RM65ArmBackend, "emergency_stop")
    assert hasattr(RM65ArmBackend, "close")
    assert hasattr(DeployPolicyQposInitController, "step")
    assert hasattr(PositionDeployExecutor, "manual_rearm")


def test_missing_required_role_fails_structural_assessment() -> None:
    inventory = build_rm65_arms_only_dispatch_inventory(_execution_manifest())
    omitted_role = "motion_pre_sdk"
    incomplete = DispatchSeamInventoryManifest.create(
        inventory_id="incomplete-rm65-inventory-v1",
        inventory_version="1.0.0",
        execution_manifest_id=inventory.execution_manifest_id,
        execution_manifest_hash=inventory.execution_manifest_hash,
        entries=tuple(
            entry for entry in inventory.entries if entry.seam_role != omitted_role
        ),
    )

    assessment = assess_rm65_arms_only_dispatch_inventory(incomplete)
    assert not assessment.inventory_structurally_complete
    assert assessment.missing_roles == (omitted_role,)
    assert not assessment.legacy_mediation_ready


def test_candidate_inventory_rejects_a_different_execution_manifest() -> None:
    manifest = _execution_manifest()
    object.__setattr__(manifest, "manifest_id", "different-profile")
    with pytest.raises(ValueError, match="first M6 candidate"):
        build_rm65_arms_only_dispatch_inventory(manifest)


def test_arms_only_inventory_does_not_silently_include_excluded_hand_seams() -> None:
    inventory = build_rm65_arms_only_dispatch_inventory(_execution_manifest())
    joined = " ".join(
        f"{entry.seam_id} {entry.implementation_path} {entry.activation_condition}"
        for entry in inventory.entries
    ).lower()
    assert "dmtac" not in joined
    assert "inspire" not in joined
    assert set(RM65_ARMS_ONLY_REQUIRED_SEAM_ROLES_V1) == {
        entry.seam_role for entry in inventory.entries
    }
