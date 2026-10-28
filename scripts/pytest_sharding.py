#!/usr/bin/env python3
"""Deterministic pytest target sharding and aggregate JUnit evidence."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import socket
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

# These weights are deliberately sparse.  They capture the modules that
# dominate recent protected runs; unknown/new modules use a stable source-size
# estimate and are still assigned automatically.  Keeping the timing profile
# here makes the default plan useful in a clean checkout without depending on
# mutable Harness history.
FULL_WEIGHTS = {
    # Exact green development e37c6a46d, Phase56 root acceptance (12/2),
    # run locally on 2026-10-28. These estimates include the current full
    # matrices and newly integrated checkpoint, anchor, scene and B601 tests.
    # Refresh existing entries; add new modules only at five seconds or more.
    # This is scheduling data, never selection, timeouts or acceptance limits.
    "tests/test_agent_gateway_mujoco_safe_tcp_pose.py": 148.473,
    "tests/test_b601_calibration_cycle.py": 139.714,
    "tests/test_mujoco_temp_pipeline.py": 118.275,
    "tests/test_agent_gateway_mujoco_fast_grasp.py": 106.963,
    "tests/test_rebot_b601_staged_assist.py": 105.838,
    "tests/test_observed_point_motion.py": 103.522,
    "tests/test_replay_position_admittance_active_pipeline.py": 101.928,
    "tests/test_b601_cruise_velocity.py": 97.752,
    "tests/test_b601_coordinated_holding.py": 94.215,
    "tests/test_mujoco_target_checkpoint.py": 88.543,
    "tests/test_rm65_grasp_trajectory.py": 85.626,
    "tests/test_b601_automatic_calibration.py": 83.114,
    "tests/test_mujoco_skill_scene_suite.py": 77.423,
    "tests/test_agent_gateway_surface_anchor.py": 65.045,
    "tests/test_rm65_ik_checkpoint.py": 64.406,
    "tests/test_mujoco_observation_checkpoint.py": 61.907,
    "tests/test_agent_gateway_real_rm65_composition.py": 50.408,
    "tests/test_agent_gateway_surface_anchor_v3.py": 49.186,
    "tests/test_mujoco_tcp_motion_chain.py": 47.444,
    "tests/test_dmtac_cube_contact_probe.py": 45.668,
    "tests/test_mujoco_execution_checkpoint.py": 43.339,
    "tests/test_mujoco_observed_safe_motion.py": 41.413,
    "tests/test_agent_gateway_mujoco_grasp_approach.py": 37.5,
    "tests/test_b601_workspace_calibration.py": 35.947,
    "tests/test_agent_gateway_mujoco_gripper_contact.py": 30.257,
    "tests/test_mujoco_fast_runtime.py": 27.069,
    "tests/test_rm65_dmtac_sweep_probe.py": 23.603,
    "tests/test_collection_process_writer.py": 23.278,
    "tests/test_rebot_b601_mujoco_haptics.py": 21.909,
    "tests/test_rebot_b601_real2sim.py": 21.849,
    "tests/test_mujoco_runtime_checkpoint.py": 20.161,
    "tests/test_agent_gateway_mujoco_object_motion.py": 18.508,
    "tests/test_b601_workspace_identification.py": 18.024,
    "tests/test_b601_workspace_collision.py": 17.989,
    "tests/test_replay_position_admittance_ik_trace.py": 17.575,
    "tests/test_agent_gateway_mujoco_tcp_pose_comparison.py": 17.253,
    "tests/test_control_station_sim_hil_allocation_profile.py": 16.954,
    "tests/test_openpi_cpu_resources.py": 16.268,
    "tests/test_b601_command_field_runtime.py": 14.469,
    "tests/test_sim_point_grasp_experiment.py": 14.281,
    "tests/test_readonly_grasp_snapshot.py": 14.044,
    "tests/test_agent_gateway_mujoco_fast_grasp_boundaries.py": 13.929,
    "tests/test_runtime_domain_boundaries.py": 13.358,
    "tests/test_rm65_self_observation_urdf.py": 12.922,
    "tests/test_sim_real2sim_artifact.py": 12.581,
    "tests/test_agent_gateway_mujoco_back_projection.py": 12.49,
    "tests/test_refinement_annotation_gui.py": 11.819,
    "tests/test_camera_semantic_materialize.py": 11.035,
    "tests/test_run_full_acceptance.py": 10.899,
    "tests/test_sam3_service.py": 10.576,
    "tests/test_mujoco_scene_qualification.py": 10.54,
    "tests/test_position_admittance_active_shared_ik.py": 10.285,
    "tests/test_mujoco_physics_scene.py": 10.257,
    "tests/test_agent_gateway_mujoco_object_state.py": 10.123,
    "tests/test_b601_managed_cycle.py": 9.185,
    "tests/test_real_teleop_progress_adapter.py": 9.183,
    "tests/test_mujoco_cube_lift.py": 8.851,
    "tests/test_mujoco_actuation_checkpoint.py": 8.525,
    "tests/test_policy_learning_boundaries.py": 8.334,
    "tests/test_rm65_dual_arm_scene_profile.py": 8.053,
    "tests/test_mujoco_state_bc_evaluation.py": 8.022,
    "tests/test_agent_gateway_segmentation_rgb_input.py": 8.017,
    "tests/test_control_station_boundaries.py": 7.989,
    "tests/test_real_shadow_acceptance.py": 7.779,
    "tests/test_runtime_artifact_migration.py": 7.655,
    "tests/test_v25p_dual_write_shadow.py": 7.437,
    "tests/test_rm65_object_transport.py": 6.972,
    "tests/test_agent_gateway_region_continuation_registry.py": 6.945,
    "tests/test_dmtac_viewer.py": 6.852,
    "tests/test_mujoco_agent_control_checkpoint.py": 6.642,
    "tests/test_mujoco_actuated_position.py": 6.55,
    "tests/test_mujoco_native_checkpoint.py": 6.495,
    "tests/test_mujoco.py": 6.467,
    "tests/test_tactile_bundle_recompression.py": 6.448,
    "tests/test_rm65_dual_safe_sequence.py": 6.336,
    "tests/test_b601_torque_console.py": 6.297,
    "tests/test_lazy_package_root_exports.py": 6.293,
    "tests/test_agent_gateway_mujoco_left_visual.py": 6.148,
    "tests/test_run_pytest_asset_root.py": 6.126,
    "tests/test_sim_point_grasp_terminal.py": 6.064,
    "tests/test_rm65_dmtac_stepped_proxy.py": 5.939,
    "tests/test_rebot_b601_mujoco_mit.py": 5.83,
    "tests/test_agent_gateway_rgbd_capture.py": 5.765,
    "tests/test_dataset_dependency_boundaries.py": 5.722,
    "tests/test_v25p_producer_dual_write.py": 5.671,
    "tests/test_control_station_exclusive_hitl.py": 5.608,
    "tests/test_modality_storage.py": 5.531,
    "tests/test_b601_holding_calibration.py": 5.511,
    "tests/test_agent_gateway_pose_candidate_set.py": 5.456,
    "tests/test_mujoco_pose_execution_observation.py": 5.278,
    "tests/test_mujoco_trajectory_composition_publication.py": 5.274,
    "tests/test_mujoco_state_bc_live_hitl.py": 5.272,
    "tests/test_agent_gateway_real_station_point_grasp.py": 5.176,
    "tests/test_b601_calibration_session.py": 5.068,
    "tests/test_mujoco_collision_safety.py": 4.935,
    "tests/test_mujoco_component_action_profile.py": 4.888,
    "tests/test_mujoco_planned_tcp_constraints.py": 4.44,
    "tests/test_openpi_policy_emission_source.py": 4.369,
    "tests/test_mujoco_formal_compound_authority.py": 4.29,
    "tests/test_mujoco_safe_tcp_goal_envelope.py": 3.707,
    "tests/test_mujoco_dmtac_gripper.py": 3.597,
    "tests/test_mujoco_cube_drop.py": 3.274,
    "tests/test_owned_worker_cleanup.py": 2.184,
    "tests/test_runtime_artifact_resolution.py": 1.579,
    "tests/test_runtime_artifact_migration_boundaries.py": 1.352,
    "tests/test_rebot_b601_submodule_boundary.py": 0.688,
    "tests/test_control_station_contracts.py": 0.112,
    # Keep the prior complete estimate: this Run skipped two optional
    # checkpoint cases, so its partial-module time cannot replace it.
    "tests/test_control_station_mujoco_sim_hil.py": 36.534,
}

# Modules normally remain the isolation unit. Reviewed exceptions have
# independent top-level tests and no shared live runtime fixture. Native
# Gateway families split to bound closed scene cycles within each process;
# replay uses function-local logs and fresh runtimes. Parameter matrices stay
# intact. Unlisted functions share the module residual and stay collected.
FULL_SPLIT_NODE_WEIGHTS = {
    # Same fixed green Run as FULL_WEIGHTS. Timings sum every parameter
    # in each measured family; each family remains one indivisible target.
    # Sub-0.25-second and new functions keep the module residual fallback.
    "tests/test_agent_gateway_mujoco_fast_grasp.py": {
        "test_native_refinement_shared_deadlines_preserve_pending_owner": 5.444,
        "test_actual_proposal_filter_and_search_report_rejection_without_motion": 4.617,
        "test_native_fast_selector_prefers_deeper_support_at_equal_motion_cost": 2.428,
        "test_native_fast_publishes_all_sixty_four_rejections_with_bounded_summary": 1.973,
        "test_native_axial_child_and_parent_use_exact_pose_and_single_terminal_offset": 2.535,
        "test_invalid_paired_reference_finishes_with_receipt_before_dispatch": 22.244,
        "test_paired_ttl_expiring_after_source_validation_rejects_before_consumption": 3.228,
        "test_grazing_retained_target_can_close_then_reopen_without_linkage_failure": 27.76,
        "test_native_selector_recovers_raw_pose_after_cheap_derived_endpoint_rejection": 2.426,
        "test_frozen_scene_digest_preserves_velocity_and_time_invalidation": 3.582,
        "test_cancel_after_query_does_not_publish_or_hold_from_readonly_route": 3.365,
        "test_native_proposal_preparation_stops_at_the_shared_deadline": 1.215,
        "test_fast_planning_rejects_changed_frozen_scene_before_model_inference": 1.848,
        "test_native_public_frozen_observation_is_atomic_and_model_lifecycle_is_session_owned": 1.737,
        "test_native_factory_preserves_the_explicit_alignment_choice": 2.605,
        "test_published_fast_grasp_pairs_retained_proposal_native_chain_without_replanning": 12.35,
        "test_native_fast_selector_balances_motion_and_capture_support": 7.602
    },
    "tests/test_agent_gateway_mujoco_safe_tcp_pose.py": {
        "test_object_relation_profile_changes_only_explicit_safe_route": 4.029,
        "test_safe_tcp_rejects_gripper_object_collision_without_dispatch": 2.285,
        "test_cancelled_safe_move_replans_from_measured_hold_without_old_references": 4.162,
        "test_observed_object_goal_region_selects_shorter_path_then_executes_existing_skill": 17.32,
        "test_safe_tcp_searches_around_blocked_direct_path": 4.377,
        "test_native_gateway_rejects_archived_skill_requests_without_dispatch": 1.217,
        "test_safe_tcp_consumes_planned_joints_without_re_ik": 1.823,
        "test_object_path_query_is_passive_and_does_not_own_execution": 3.882,
        "test_safe_tcp_policy_rejects_invalid_values_before_planning_pose_goal_policy": 1.093,
        "test_native_continuation_query_rejects_retained_dead_end_before_motion": 6.062,
        "test_safe_plan_ownership_reference_order_and_cancellation": 1.074,
        "test_query_and_execution_share_precise_hover_contract": 21.481,
        "test_safe_tcp_v2_narrow_cone_accounts_for_published_float32_qref": 1.128,
        "test_preserve_target_reports_existing_arrival_region_without_claiming_exact_convergence": 2.69,
        "test_safe_tcp_v2_endpoint_outside_axis_cone_is_rejected_without_dispatch": 1.126,
        "test_left_safe_gateway_close_fences_an_inflight_request": 1.171,
        "test_retained_agent_hover_precision_uses_deployed_contract": 29.594,
        "test_safe_tcp_v2_preserves_sampled_axis_and_retains_null_behavior": 3.6,
        "test_safe_gateway_close_fences_an_inflight_request": 1.166,
        "test_safe_tcp_v2_rechecks_actual_to_reference_axis_before_dispatch": 1.263,
        "test_left_cancelled_safe_move_replans_from_measured_hold_without_old_references": 2.385,
        "test_safe_move_already_satisfied_confirms_hold_without_commands": 2.502,
        "test_spatial_goal_calculation_then_existing_safe_move": 3.941,
        "test_native_safe_session_does_not_republish_archived_debug_skills": 5.302,
        "test_safe_tcp_consumes_planned_joints_without_re_ik_pose_goal_policy": 23.736
    },
    "tests/test_mujoco_temp_pipeline.py": {
        "test_randomized_pose_task_repeated_seed_has_identical_closed_loop_trace": 3.693,
        "test_contact_curriculum_reset_starts_from_real_bilateral_contact": 2.991,
        "test_privileged_near_lift_v2_switches_to_lift_only_after_real_contact": 2.888,
        "test_separate_actor_observation_critic_cannot_update_actor_backbone": 3.414,
        "test_lift_short_v2_zero_policy_holds_and_positive_policy_lifts": 6.614,
        "test_contact_curriculum_v3_scripted_gate_lifts_from_reset": 3.377,
        "test_contact_curriculum_v2_applies_lift_first_action_constraints": 2.858,
        "test_contact_curriculum_v2_scripted_gate_lifts_from_reset": 3.384,
        "test_contact_curriculum_v3_applies_only_nonnegative_lift_action": 2.754,
        "test_contact_curriculum_reset_retries_recoverable_failure": 2.913,
        "test_privileged_critic_checkpoint_keeps_actor_inference_self_contained": 3.196,
        "test_descend_close_task_is_feasible_without_hidden_xy_actions": 2.829,
        "test_reach_goal_delta_matches_pose_target_in_public_action_frame": 2.033,
        "test_compute_gae_bootstraps_time_limit_but_not_true_terminal": 1.542,
        "test_contact_curriculum_v3_projection_preserves_locked_gripper_target": 3.082,
        "test_reach_task_kernel_preserves_pose_action_lineage_and_seeded_reset": 3.975,
        "test_settled_snapshot_preserves_first_actuated_policy_step": 2.781,
        "test_temp_policy_expands_compact_action_into_public_layout": 1.505,
        "test_contact_curriculum_v2_repeated_reset_preserves_liftable_grasp": 5.675,
        "test_goal_delta_pick_physical_oracle_grasps_from_home": 3.091,
        "test_temp_policy_checkpoint_round_trip": 1.796,
        "test_pose_reset_rebuilds_solver_and_syncs_settled_actuator_state": 5.459,
        "test_contact_curriculum_v2_tracks_contact_loss_without_early_termination": 2.923,
        "test_pose_task_suite_has_a_scripted_feasibility_gate": 35.135,
        "test_normal_v2_scripted_physical_grasp_gate_reaches_bilateral_lift": 3.357,
        "test_post_contact_phase_exits_after_configured_contact_loss_grace": 2.511,
        "test_goal_delta_actor_bootstrap_is_optional_and_nonprivileged": 1.753
    },
    "tests/test_replay_position_admittance_active_pipeline.py": {
        "test_named_larger_active_envelopes_close_through_shared_ik_and_release": 12.953,
        "test_raw_wrench_active_pipeline_exercises_shared_ik": 4.202,
        "test_normal_shadow_envelope_is_replayed_under_cautious_active_caps": 4.207,
        "test_right_combined_formal_active_runs_through_physical_mujoco": 27.617,
        "test_half_normal_mujoco_endpoint_guard_holds_before_larger_qref_dispatch": 14.434,
        "test_sub_hold_wrench_is_not_promoted_as_active_ik_coverage": 3.841,
        "test_larger_envelope_mujoco_abrupt_release_retains_frozen_catchup_gate": 25.749,
        "test_raw_wrench_active_pipeline_runs_through_physical_mujoco": 8.916
    }
}

FAST_WEIGHTS = {
    "tests/test_real_shadow_acceptance.py": 18.562,
    "tests/test_runtime_domain_boundaries.py": 9.370,
    "tests/test_refinement_annotation_gui.py": 7.908,
    "tests/test_modality_storage.py": 6.843,
    "tests/test_real_teleop_progress_adapter.py": 6.224,
    "tests/test_lazy_package_root_exports.py": 5.753,
    "tests/test_control_station_contracts.py": 5.290,
    "tests/test_run_pytest_asset_root.py": 4.251,
    "tests/test_official_all_ik_governor.py": 4.179,
    "tests/test_annotation_projection.py": 3.834,
    "tests/test_agent_gateway_compatibility.py": 2.835,
    "tests/test_idle_tactile_compression.py": 2.795,
    "tests/test_control_station_boundaries.py": 2.740,
    "tests/test_agent_gateway_artifact_store.py": 2.604,
    "tests/test_position_admittance_active_shared_ik.py": 2.598,
}

FAST_INTEGRATION_WEIGHTS = {
    # Exact module sums from the unsharded integration tier at 47d7be6e.
    # Unit-heavy modules deliberately do not influence this plan.
    "tests/test_rm65_grasp_trajectory.py": 74.288,
    "tests/test_runtime_artifact_migration.py": 25.525,
    "tests/test_collection_process_writer.py": 14.611,
    "tests/test_runtime_artifact_resolution.py": 8.269,
    "tests/test_v25p_producer_dual_write.py": 6.729,
    "tests/test_agent_gateway_mujoco_long_move_trace.py": 5.382,
    "tests/test_v25p_dual_write_shadow.py": 3.672,
    "tests/test_mujoco.py": 3.477,
    "tests/test_writer_v21_import_path.py": 3.145,
    "tests/test_handeye_utils.py": 2.990,
    "tests/test_position_deploy_executor.py": 2.812,
    "tests/test_dmtac_spawn_import_path.py": 2.554,
    "tests/test_worktree_coordination.py": 1.956,
    "tests/test_run_full_acceptance.py": 1.948,
    "tests/test_collection_progress_runtime.py": 1.715,
    "tests/test_mujoco_formal_capture.py": 1.206,
    "tests/test_v25p_native_store.py": 0.923,
    "tests/test_run_fast_feedback.py": 0.886,
    "tests/test_position_force_guard.py": 0.840,
    "tests/test_progress_timeline.py": 0.820,
    "tests/test_v25p_sidecar_store.py": 0.670,
    "tests/test_control_trace_episode.py": 0.638,
    "tests/test_tactile_external_storage.py": 0.569,
    "tests/test_control_station_vive_deflection.py": 0.511,
}

WEIGHT_PROFILES = {
    "fast": FAST_WEIGHTS,
    "fast-integration": FAST_INTEGRATION_WEIGHTS,
    "full": FULL_WEIGHTS,
}


def check_required_local_sockets() -> None:
    """Verify the local transports exercised by complete acceptance."""

    with tempfile.TemporaryDirectory(prefix="dt-full-socket-") as directory:
        probes = (
            (socket.AF_UNIX, str(Path(directory) / "probe.sock"), "AF_UNIX"),
            (socket.AF_INET, ("127.0.0.1", 0), "loopback AF_INET"),
        )
        for family, address, label in probes:
            try:
                with socket.socket(family, socket.SOCK_STREAM) as server:
                    server.settimeout(1.0)
                    server.bind(address)
                    server.listen(1)
                    with socket.socket(family, socket.SOCK_STREAM) as client:
                        client.settimeout(1.0)
                        client.connect(server.getsockname())
                        connection, _ = server.accept()
                        connection.close()
            except OSError as error:
                raise OSError(
                    f"required local socket capability unavailable for {label}: {error}"
                ) from error


def _estimated_weight(path: Path, relative_path: str, profile: str) -> float:
    measured = WEIGHT_PROFILES[profile].get(relative_path)
    if measured is not None:
        return measured
    # Source size is a better fallback than file-count round robin for large
    # assertion/parameter tables, while the floor prevents tiny modules from
    # all accumulating in one shard.
    return max(0.05, path.stat().st_size / 50_000.0)


def _split_test_functions(path: Path, relative_path: str) -> list[str]:
    """Return every statically discoverable top-level test in a split module."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise ValueError(
            f"could not parse split test module {relative_path}: {error}"
        ) from error
    unsupported = []
    functions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and (
            node.name.startswith("Test")
            or any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test_")
                for child in node.body
            )
        ):
            unsupported.append(f"test class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in {
                "pytest_collect_file",
                "pytest_collection_modifyitems",
                "pytest_generate_tests",
            }:
                unsupported.append(f"collection hook {node.name}")
            elif node.name.startswith("test_"):
                functions.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if any(
                isinstance(target, ast.Name) and target.id.startswith("test_")
                for target in targets
            ):
                unsupported.append("dynamically assigned test")
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (alias.asname or alias.name).startswith("test_") for alias in node.names
        ):
            unsupported.append("imported test")
    nested_tests = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    } - set(functions)
    unsupported.extend(f"nested test {name}" for name in sorted(nested_tests))
    if unsupported:
        raise ValueError(
            f"split test module {relative_path} has unsupported collection: "
            + ", ".join(unsupported)
        )
    if not functions:
        raise ValueError(f"split test module has no top-level tests: {relative_path}")
    if len(functions) != len(set(functions)):
        raise ValueError(
            f"split test module has duplicate test functions: {relative_path}"
        )
    return functions


def _shard_items(
    path: Path,
    relative_path: str,
    profile: str,
) -> list[tuple[str, Path, float]]:
    node_weights = (
        FULL_SPLIT_NODE_WEIGHTS.get(relative_path) if profile == "full" else None
    )
    if node_weights is None:
        return [(relative_path, path, _estimated_weight(path, relative_path, profile))]

    functions = _split_test_functions(path, relative_path)
    unknown_weights = set(node_weights) - set(functions)
    if unknown_weights:
        raise ValueError(
            f"split timing profile names missing tests in {relative_path}: "
            f"{sorted(unknown_weights)}"
        )
    module_weight = _estimated_weight(path, relative_path, profile)
    measured_weight = sum(node_weights.values())
    unmeasured = len(functions) - len(node_weights)
    residual = module_weight - measured_weight
    if residual < 0.0 or (unmeasured == 0 and residual > 1e-6):
        raise ValueError(
            f"split timing profile does not match module weight for {relative_path}: "
            f"module={module_weight} measured={measured_weight}"
        )
    fallback = residual / unmeasured if unmeasured else 0.0
    return [
        (
            f"{relative_path}::{name}",
            Path(f"{path}::{name}"),
            node_weights.get(name, fallback),
        )
        for name in functions
    ]


def plan_shards(
    repository: Path,
    tests_root: Path,
    shard_count: int,
    profile: str,
    *,
    test_modules: Iterable[str] | None = None,
) -> list[list[tuple[str, Path, float]]]:
    repository = repository.resolve()
    tests_root = tests_root.resolve()
    files = sorted(path for path in tests_root.rglob("test_*.py") if path.is_file())
    selected = None
    if test_modules is not None:
        requested = list(test_modules)
        available = {path.relative_to(repository).as_posix(): path for path in files}
        if (
            not requested
            or len(requested) != len(set(requested))
            or not set(requested) <= available.keys()
        ):
            raise ValueError(
                "selected test modules contain duplicates or unknown paths"
            )
        selected = set(requested)
    if not files:
        raise ValueError(f"no pytest modules found under {tests_root}")
    if not 1 <= shard_count <= len(files):
        raise ValueError(
            f"shard count must be between 1 and {len(files)}: {shard_count}"
        )

    weighted: list[tuple[str, Path, float]] = []
    expected_modules: list[str] = []
    for path in files:
        try:
            relative = path.relative_to(repository).as_posix()
        except ValueError as error:
            raise ValueError(
                f"tests root is outside repository: {tests_root}"
            ) from error
        expected_modules.append(relative)
        weighted.extend(_shard_items(path, relative, profile))

    shards: list[list[tuple[str, Path, float]]] = [[] for _ in range(shard_count)]
    totals = [0.0] * shard_count
    for item in sorted(weighted, key=lambda value: (-value[2], value[0])):
        index = min(
            range(shard_count),
            key=lambda candidate: (
                totals[candidate],
                len(shards[candidate]),
                candidate,
            ),
        )
        shards[index].append(item)
        totals[index] += item[2]

    planned = [target for shard in shards for target, _, _ in shard]
    if len(planned) != len(set(planned)):
        raise ValueError("internal shard plan contains duplicate test targets")
    planned_modules = {target.split("::", 1)[0] for target in planned}
    if planned_modules != set(expected_modules):
        missing = sorted(set(expected_modules) - planned_modules)
        extra = sorted(planned_modules - set(expected_modules))
        raise ValueError(
            f"internal shard coverage mismatch: missing={missing} extra={extra}"
        )
    if selected is not None:
        # Preserve the complete plan's native-runtime co-location and order.
        # Repacking a smaller set by CPU time can put two large native heaps
        # together even when the original full plan passed its memory gates.
        shards = [
            [item for item in shard if item[0].split("::", 1)[0] in selected]
            for shard in shards
        ]
        if {
            item[0].split("::", 1)[0] for shard in shards for item in shard
        } != selected:
            raise ValueError("filtered complete plan lost selected test modules")
    return shards


def write_plan(
    repository: Path,
    tests_root: Path,
    output_dir: Path,
    shard_count: int,
    profile: str,
    prefix: str,
    manifest_name: str = "shard-plan.json",
    *,
    test_modules: Iterable[str] | None = None,
) -> None:
    if not output_dir.is_dir():
        raise ValueError(f"output directory does not exist: {output_dir}")
    if Path(manifest_name).name != manifest_name or not manifest_name.endswith(".json"):
        raise ValueError(f"invalid shard manifest name: {manifest_name}")
    shards = plan_shards(
        repository, tests_root, shard_count, profile, test_modules=test_modules
    )
    manifest_shards = []
    for index, shard in enumerate(shards):
        paths_file = output_dir / f"{prefix}-{index}-paths.txt"
        paths_file.write_text(
            "".join(f"{path}\n" for _, path, _ in shard), encoding="utf-8"
        )
        test_targets = [item[0] for item in shard]
        test_modules = sorted({target.split("::", 1)[0] for target in test_targets})
        manifest_shards.append(
            {
                "index": index,
                "estimated_seconds": round(sum(item[2] for item in shard), 6),
                "file_count": len(test_modules),
                "target_count": len(shard),
                "paths_file": paths_file.name,
                "test_modules": test_modules,
                "test_targets": test_targets,
            }
        )
    manifest = {
        "schema": "dexteleop.pytest-shard-plan.v2",
        "profile": profile,
        "shard_count": shard_count,
        "file_count": len(
            {target.split("::", 1)[0] for shard in shards for target, _, _ in shard}
        ),
        "target_count": sum(len(shard) for shard in shards),
        "estimated_seconds": round(
            sum(item[2] for shard in shards for item in shard), 6
        ),
        "shards": manifest_shards,
    }
    (output_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _test_suites(root: ET.Element, source: Path) -> Iterable[ET.Element]:
    if root.tag == "testsuite":
        return (root,)
    if root.tag == "testsuites":
        suites = tuple(child for child in root if child.tag == "testsuite")
        if suites:
            return suites
        raise ValueError(f"JUnit shard has no <testsuite>: {source}")
    raise ValueError(f"JUnit shard root is not <testsuite> or <testsuites>: {source}")


def merge_junit(destination: Path, sources: list[Path]) -> None:
    if not sources:
        raise ValueError(f"no JUnit reports were supplied for {destination}")
    roots: list[ET.Element] = []
    for source in sources:
        if not source.is_file():
            raise ValueError(f"missing JUnit shard report: {source}")
        roots.extend(_test_suites(ET.parse(source).getroot(), source))

    merged = ET.Element("testsuite", dict(roots[0].attrib))
    for name in ("tests", "errors", "failures", "skipped"):
        merged.set(name, str(sum(int(root.attrib.get(name, "0")) for root in roots)))
    merged.set(
        "time",
        format(sum(float(root.attrib.get("time", "0")) for root in roots), ".6f"),
    )

    seen_cases: set[tuple[str, str]] = set()
    for root in roots:
        for child in root:
            if child.tag == "testcase":
                identity = (
                    child.attrib.get("classname", ""),
                    child.attrib.get("name", ""),
                )
                if identity in seen_cases:
                    raise ValueError(
                        "duplicate testcase in JUnit shards: "
                        f"{identity[0]}::{identity[1]}"
                    )
                seen_cases.add(identity)
            merged.append(copy.deepcopy(child))

    ET.indent(merged, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged).write(destination, encoding="utf-8", xml_declaration=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="write deterministic shard path files")
    plan.add_argument("--repository", type=Path, required=True)
    plan.add_argument("--tests-root", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--shards", type=int, required=True)
    plan.add_argument("--profile", choices=sorted(WEIGHT_PROFILES), required=True)
    plan.add_argument("--prefix", default="shard")
    plan.add_argument("--manifest-name", default="shard-plan.json")
    merge = commands.add_parser("merge-junit", help="merge and validate shard JUnit")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("sources", type=Path, nargs="+")
    commands.add_parser(
        "check-sockets",
        help="verify loopback TCP and Unix sockets before complete acceptance",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "plan":
            write_plan(
                arguments.repository,
                arguments.tests_root,
                arguments.output_dir,
                arguments.shards,
                arguments.profile,
                arguments.prefix,
                arguments.manifest_name,
            )
        elif arguments.command == "merge-junit":
            merge_junit(arguments.output, arguments.sources)
        else:
            check_required_local_sockets()
    except (OSError, ValueError, ET.ParseError) as error:
        print(f"pytest_sharding: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
