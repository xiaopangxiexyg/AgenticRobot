"""Supported real-policy deployment composition runtime.

The canonical shell entrypoint is ``python -m teleop_runtime.deploy``.  Its
thin CLI wrapper delegates to :func:`main`; this module also remains a
supported direct/programmatic entrypoint for active callers such as the OpenPI
real launcher.  It has no announced end-of-life.  Any future retirement must
first migrate direct callers, publish a replacement and migration guide, then
run through an explicit deprecation period.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.layout import episode_id_from_index
from dataset_contracts.policy_artifact import policy_artifact_from_metadata
from dataset_contracts.pose_frames import load_pose_frame_contract
from dataset_contracts.pose_training import POSE_TRAINING_ORIGIN_NATIVE_V23
from dataset_contracts.v22 import (
    DEXTELEOP_DATASET_SCHEMA_V22,
    policy_io_layout_from_metadata,
)
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from deployments.client import load_policy_artifact
from deployments.client.policy import load_policy
from deployments.client.rtc_executor import (
    rtc_client_inference_bundle_debug_payload,
)
from deployments.client.rtc_frontend import (
    build_rtc_conditioned_model_shadow_frontend_v3,
    build_rtc_model_shadow_frontend,
)
from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    ArmActionSpec,
    CanonicalActionSafetyAnchor,
    CanonicalKeyframeSafetyProfile,
    EndEffectorSafetyCapability,
    EndEffectorTarget,
    HandActionSpec,
    RobotAction,
    RobotActionResolver,
    build_robot_action_v1_runtime_schema,
)
from robot_motion.deploy.actions import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    NormalizedAction,
)
from robot_motion.deploy.async_producer import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1,
    POLICY_ASYNC_PRODUCER_ELIGIBLE,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_V1,
)
from robot_motion.deploy.canonical_action import (
    CanonicalDeployActionPipeline,
    CanonicalDeployActionResult,
    CanonicalPolicyActionPipeline,
    CanonicalPolicyActionResult,
)
from robot_motion.deploy.canonical_policy_action import (
    CanonicalPolicyAction,
    build_canonical_policy_action,
)
from robot_motion.deploy.emission_port import (
    PassivePolicyEmissionObserverPort,
    require_passive_policy_emission_observer,
)
from robot_motion.deploy.next_step import sync_queue_preserves_same_chunk_successor
from robot_motion.deploy.observation import (
    DEFAULT_HAND_STATE_MAX_AGE_SEC,
    DeploymentObservation,
    build_deployment_observation,
)
from robot_motion.deploy.pose_adapter import (
    convert_policy_absolute_vector_to_runtime_raw,
)
from robot_motion.deploy.runtime import (
    PolicyChunkPredictionResult,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    resolve_action_queue_policy,
)
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    DEPLOY_ACTION_VECTOR_ACTION_SPACE,
    STATE_ARM_POSE_SLICES,
    to_serializable,
)
from robot_motion.deploy.session_contract import (
    POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
    POLICY_CONTRACT_V22_NATIVE,
    ResolvedDeploySessionContract,
    RuntimeEndEffectorCapability,
    resolve_deploy_session_contract,
)
from robot_motion.deploy.v22 import V22DeploymentAdapter
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract,
)
from robot_motion.position_control import (
    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_CHOICES,
    POSITION_PLAN_SPLICE_DYNAMICS_OFF,
    POSITION_PLAN_SPLICE_DYNAMICS_PROFILE_CHOICES,
    PositionKeyframeSafetyConfig,
    PositionStreamStartAcquisitionConfig,
    resolve_position_plan_splice_dynamics_config,
)
from robot_motion.replay.trajectory import QPOS_INIT_SOURCES
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
    DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M,
    LEGACY_POSITION_CONTROL_PROFILE_NAME,
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    RM65_ARM_CONTROL_MODE_IMPEDANCE,
    RM65_ARM_CONTROL_MODE_POSITION,
    RM65_ARM_CONTROL_MODES,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSE_LIFT_PROFILE_CHOICES,
    RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
    RM65_POSITION_TRACKING_FOLLOW_ALPHA,
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
    RM65_QREF_ACCELERATION_COMFORT_75_V1,
    RM65_QREF_ACCELERATION_HARD_V1,
    RM65_QREF_ACCELERATION_PROFILE_CHOICES,
    normalize_rm65_arm_control_mode,
    resolve_rm65_pose_lift_profile,
    resolve_rm65_qref_acceleration_profile,
)
from robot_motion.runtime import (
    TeleopCommand,
    available_cpu_ids,
    cpu_thread_sibling_ids,
)
from robot_motion.runtime_gate import IKActionChunkGate
from runtime_artifact_paths import (
    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    configured_runtime_artifact_layout,
)

from ..action.legacy_execution_context import (
    LegacyControlOwnerStateTransitionOutcome,
    LegacyControlOwnerStateTransitionRequest,
)
from ..action.replay_proposal import (
    DatasetReplayProposalShadowAdapter,
    DatasetReplayProposalShadowConfig,
    replay_adapter_debug_payload,
)
from ..backends.arms import (
    DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN,
    RM65_BACKGROUND_TARGET_ADMISSION_CHOICES,
    RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    RM65ArmBackend,
    RM65ImpedanceArmBackend,
)
from ..backends.grippers import (
    DMTAC_TACTILE_FIELDS,
    DMTacGripperBackendConfig,
    GRIPPER_ACTION_MAX,
)
from ..backends.hands import (
    DEFAULT_INSPIRE_HAND_MAX_FORCE,
)
from ..collection.camera_preview import CameraPreviewConfig, CameraPreviewer
from ..collection.hand_state_polling import (
    HandStateHistoryPoller,
    HandStateHistoryPollerConfig,
)
from ..configs.camera_paths import DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
from ..configs.dmtac_hardware import (
    add_dmtac_assignment_arguments,
    dmtac_backend_configs_from_assignment,
    load_dmtac_hardware_assignment,
    manual_dmtac_backend_config_for_side,
    require_dmtac_sides_resolved,
)
from ..defaults import (
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from ..interfaces import EffectorBackend
from ..log_utils import ChangeGate
from ..log_utils import log_event as console_log_event
from ..position_control import (
    ARM_EXECUTION_PROFILE_CHOICES,
    CONTINUOUS_WAYPOINT_WORKER_ISOLATION_CHOICES,
    CONTINUOUS_WAYPOINT_WORKER_PROCESS,
    CONTINUOUS_WAYPOINT_WORKER_THREAD,
    OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND,
    PINK_POSE_LIFT_BACKEND,
    POSITION_15_60_120HZ_PROFILE_NAME,
    POSITION_PATH_LIFT_BACKEND_CHOICES,
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
    apply_rm65_position_profile_defaults,
    build_official_all_ik_governor_from_started_backends,
    build_position_collision_planner_from_started_backends,
    resolve_real_arm_execution_config,
)
from ..position_force_guard import (
    POSITION_FORCE_SOURCES,
    PositionForceGuardFault,
    PositionForceGuardMonitor,
    PositionForceGuardTripped,
    position_force_guard_config,
)
from ..sources.camera import (
    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
    REALSENSE_STARTUP_WARMUP_POLICIES,
)
from ..tactile_visualizer import (
    TactileUVVisualizer,
    TactileVisualizerConfig,
    tactile_by_side_from_hand_states,
)
from ..target_safety.deploy_tcp_pose_envelope import (
    DeployTCPPoseEnvelopeProjector,
)
from ..target_safety.tcp_pose_envelope_runtime import (
    load_tcp_pose_envelope_file_spec,
)
from .action_dispatch import (
    command_effector_kind_subset,
    dispatch_position_discrete_command_checked as _dispatch_position_discrete_command_checked,
)
from .action_execution import (
    PreparedActionExecution as _PreparedActionExecution,
    PreparedActionExecutor,
)
from .arbiter_shadow import (
    ARBITER_SHADOW_MODE_SHADOW,
    ARBITER_SHADOW_MODES_M3,
    PassivePolicyArbiterShadow,
    PolicyArbiterShadowConfig,
)
from .arm_backend_runtime import (
    DEPLOY_ARM_STEP_DT_ALPHA,
    DEPLOY_ARM_STEP_DT_MAX_SCALE,
    DEPLOY_ARM_STEP_DT_STALL_RESET_SEC,
    DEPLOY_MAX_ARM_JOINT_STEP_RAD,
    DEPLOY_RM65_ARM_MAX_ANGVEL,
    annotate_deploy_arm_command_limits,
    apply_arm_step_dt_defaults,
    build_arm_backends,
    deploy_arm_backend_control_dt_sec,
    deploy_arm_command_max_joint_step,
    deploy_arm_command_step_dt_sec,
    effective_arm_canfd_follow,
    effective_arm_control_hz,
    effective_position_reference_hz,
    _position_deploy_ik_tuning,
    resolve_arm_step_dt_max_scale,
    resolve_position_reference_and_command_hz as _resolve_position_reference_and_command_hz,
)
from .arm_safety_runtime import (
    emergency_stop_deploy_arm_backends,
    emergency_stop_position_deploy_on_keyboard_interrupt as _emergency_stop_position_deploy_on_keyboard_interrupt,
    hold_deploy_arm_backends,
    request_policy_inference_soft_hold,
)
from .arm_state_reader import ArmStateReader
from .camera_runtime import (
    DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC,
    DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST,
    REAL_TELEOP_CAMERA_START_STAGGER_SEC,
    build_camera_sources,
    resolve_policy_camera_input_keys,
    run_camera_startup_self_check,
    select_preview_camera_sources,
    start_camera_sources,
)
from .camera_cli_validation import validate_camera_cli_args
from .cli_args import argument_was_supplied as _argument_was_supplied
from .cli_value_parsers import (
    _non_negative_float,
    _non_negative_int,
    _parse_active_sides,
    _parse_arm_max_angvel,
    _parse_disconnected_gripper_sides,
    _parse_dmtac_tactile_fields,
    _parse_end_effector_sides,
    _parse_float_tuple,
    _parse_qpos6,
    _parse_vector3,
    _positive_float,
    _positive_int,
    _unit_interval_float,
)
from .console_runtime import LoopRate, NonBlockingKeyboard
from .deploy_cli_parser import add_args as _register_deployment_cli_args
from .debug_payload import (
    DEPLOY_DEBUG_PAYLOAD_COMPACT,
    DEPLOY_DEBUG_PAYLOAD_FULL,
    DEPLOY_DEBUG_PAYLOAD_MODES,
    _DEBUG_EFFECTOR_METADATA_KEYS,
    _DEBUG_LAST_COMMAND_METADATA_KEYS,
    _debug_action_summary,
    _debug_action_vector,
    _debug_arm_joint_targets,
    _debug_args_payload,
    _debug_arm_state_summary,
    _debug_backend_arm_states,
    _debug_backend_dispatch_by_effector_summary,
    _debug_backend_group_health_summary,
    _debug_backend_limits,
    _debug_background_streaming_summary,
    _debug_compact_value,
    _debug_command_arm_pose_by_side,
    _debug_command_summary,
    _debug_hand_by_side_from_vector,
    _debug_metadata_summary,
    _debug_pose_by_side_from_vector,
    _debug_pose_delta,
    _debug_pose_delta_by_side,
    _debug_position_executor_event_summary,
    _debug_quat_delta_deg,
    _debug_redact,
    _debug_remote_policy_summary,
    _debug_state_pose_by_side,
    _debug_vector,
    _debug_xyzquat,
)
from .debug_logger import (
    DeploymentDebugLogger,
    RuntimeArtifactLayout,
    canonical_deployment_debug_root,
    resolve_deployment_debug_root,
)
from .dispatch import (
    POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES,
    POSITION_REFERENCE_DISPATCH_AUDIT_POST,
    dispatch_position_reference_command,
    require_arm_backends_ready_for_position_dispatch,
)
from .fixed_init import DeployFixedInitController, load_deploy_fixed_init_plan
from .hand_backend_config import (
    GRIPPER_POLICY_DEFAULTS,
    _apply_gripper_policy_to_dmtac_config,
    _dmtac_tactile_enabled,
    _hand_backend_mode,
    _hand_backend_mode_for_side,
    _hands_enabled,
    _load_gripper_task_profile,
    _profile_value,
    _resolved_gripper_policy,
)
from .hand_backend_runtime import (
    HAND_BACKEND_CHOICES,
    HAND_BACKEND_DMTAC_GRIPPER,
    HAND_BACKEND_INSPIRE,
    build_hand_backends as _build_resolved_hand_backends,
    disconnected_gripper_selection_label as _disconnected_gripper_selection_label,
    filter_policy_qpos_init_hand_action_specs,
    hand_action_specs_for_modes,
    hand_backend_startup_diagnostic as _hand_backend_startup_diagnostic,
    start_hand_backends_checked as _start_hand_backends_checked,
)
from .impedance_force_guard import (
    ImpedanceForceGuardConfig,
    ImpedanceForceGuardFault,
    ImpedanceForceGuardMonitor,
)
from .impedance_runtime import (
    enable_impedance_force_control_backends,
    warm_start_impedance_force_control_backends,
)
from .policy_qpos_init import (
    DeployPolicyQposInitController,
    DeployPolicyQposInitSettings,
    load_deploy_policy_qpos_init_plan,
)
from .policy_binding import (
    bind_policy_pose_mode_from_artifact as _bind_policy_pose_mode_from_artifact,
    build_policy_observation_contract as _policy_observation_contract,
    parse_policy_kwargs,
)
from .position_plan_cut import (
    DeploymentPositionPlanCutDetector,
    prepare_position_plan_cut_for_action_step as _prepare_position_plan_cut_for_action_step,
)
from .position_stream_handoff import (
    defer_position_action_after_synchronous_prediction as _defer_position_action_after_synchronous_prediction,
    hold_active_position_stream_for_synchronous_chunk_gap as _hold_active_position_stream_for_synchronous_chunk_gap,
    position_source_tail_pending_before_synchronous_chunk_gap as _position_source_tail_pending_before_synchronous_chunk_gap,
)
from .position_profile_validation import (
    validate_position_worker_args as _validate_position_worker_args,
)
from .replay import (
    bind_debug_replay_arm_control_mode,
    debug_replay_execution_exhausted as _debug_replay_execution_exhausted,
    load_dataset_replay_policy,
    skip_startup_replay_frames as _skip_startup_replay_frames,
)
from .replay_cli_validation import (
    validate_debug_replay_chunk_size,
    validate_debug_replay_window_and_sources,
)
from .rtc_cli_validation import validate_rtc_cli_args
from .tactile_cli_validation import validate_tactile_cli_args
from .quest_paired_pose_capture import (
    QuestPairedPoseCaptureApplicationPort,
    QuestPairedPoseCaptureFactory,
    bind_passive_robot_pose_journal_source,
    require_quest_paired_pose_capture_application,
)
from .resource_lifecycle import PolicyDeployResourceLifecycle
from .rm65_legacy_worker_binding import RM65LegacyWorkerExecutionBinding
from .rm65_pose_observation import (
    RM65PoseObservationAssembly,
    assemble_rm65_pose_observation,
    build_policy_deploy_receiver_clock,
)
from .rtc_debug import (
    RTC_DEBUG_LEVEL_OFF,
    RTC_DEBUG_LEVEL_SUMMARY,
    RTC_DEBUG_LEVEL_TRACE,
    RTC_DEBUG_LEVELS,
    RtcDebugEventObserver,
)
from .status_view import (
    camera_status_text,
    compact_faults as _compact_faults,
    compact_log_value,
    hand_state_snapshot,
    policy_status_text as _policy_status_text,
    print_controls,
)
from .rtc_frontend import RtcPolicyShadowConfig, RtcPolicyShadowFrontend
from .rtc_step_fanout import (
    PolicyActionEmissionFanout,
    RtcFrontendStepFanout,
    RtcPolicyEmissionPublisher,
)
from .runtime_timing import (
    _action_hz_reference,
    _runtime_action_keyframe_period_sec,
    resolve_effective_action_hz,
    resolve_effective_policy_hz,
)
from .safety_shadow import evaluate_policy_legacy_aligned_safety
from .startup_init import (
    StartupInitializer,
    finish_policy_qpos_init_reference_cut as _finish_policy_qpos_init_reference_cut,
    reset_startup_initializer as _reset_startup_initializer,
    resolve_startup_init_action,
)

DEFAULT_CAMERA_CONFIG_PATH = DEFAULT_REALSENSE_DEVICE_CONFIG_PATH


def _dmtac_backend_configs(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, DMTacGripperBackendConfig]:
    cached = getattr(args, "_dmtac_backend_configs", None)
    if isinstance(cached, dict):
        return dict(cached)
    disconnected_sides = set(
        _parse_disconnected_gripper_sides(
            getattr(args, "disconnected_gripper_sides", ())
        )
    )
    dmtac_sides = tuple(
        side
        for side in active_sides
        if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
        and side not in disconnected_sides
    )
    if not dmtac_sides:
        args._dmtac_backend_configs = {}
        return {}

    policy = _resolved_gripper_policy(args)
    configs: dict[str, DMTacGripperBackendConfig] = {}
    if bool(getattr(args, "dmtac_default_config", True)):
        try:
            validate_current_usb = bool(
                getattr(args, "dmtac_validate_current_usb", True)
            ) and not bool(disconnected_sides)
            if disconnected_sides and bool(
                getattr(args, "dmtac_validate_current_usb", True)
            ):
                print(
                    "[DMTAC] saved assignment will be read without a USB/gripper "
                    "probe because intentionally disconnected sides must receive "
                    "no transport I/O; connected sides remain fail-closed at backend startup.",
                    flush=True,
                )
            assignment = load_dmtac_hardware_assignment(
                str(getattr(args, "dmtac_config", "latest") or "latest"),
                validate_current_usb=validate_current_usb,
                allow_missing=bool(getattr(args, "dmtac_allow_missing", False)),
                assume_yes_for_extra=bool(
                    getattr(args, "dmtac_assume_yes_for_extra", True)
                ),
            )
            configs.update(
                dmtac_backend_configs_from_assignment(
                    assignment,
                    active_sides=dmtac_sides,
                    prefer_existing_by_path_port=bool(disconnected_sides),
                    tactile_backend=str(
                        getattr(args, "dmtac_tactile_backend", "cpu") or "cpu"
                    ),
                    speed=float(policy["speed"]),
                    torque=float(policy["torque"]),
                    max_action=float(policy["max_action"]),
                )
            )
        except Exception as exc:
            if bool(getattr(args, "dmtac_config_required", False)):
                raise RuntimeError(
                    f"Failed to load DMTAC hardware assignment {getattr(args, 'dmtac_config', 'latest')!r}: {exc}"
                ) from exc
            print(
                f"[DMTAC] WARNING: could not load DMTAC hardware assignment: {exc}",
                flush=True,
            )

    for side in dmtac_sides:
        manual = manual_dmtac_backend_config_for_side(args, side, policy)
        if manual is not None:
            configs[side] = manual

    resolved = {
        side: _apply_gripper_policy_to_dmtac_config(config, args=args, policy=policy)
        for side, config in configs.items()
        if side in active_sides and (config.connect_hardware or config.connect_tactile)
    }
    if bool(getattr(args, "dmtac_config_required", False)):
        require_dmtac_sides_resolved(
            dmtac_sides,
            resolved,
            config=str(getattr(args, "dmtac_config", "latest") or "latest"),
        )
    args._dmtac_backend_configs = dict(resolved)
    args.dmtac_config_loaded_sides = sorted(resolved)
    return dict(resolved)


def _runtime_end_effector_safety_capability(
    *,
    args: argparse.Namespace,
    deploy_layout: Any,
    side: str,
) -> EndEffectorSafetyCapability | None:
    """Bind one native policy channel to reviewed backend command limits.

    Native targets must not inherit an unconstrained range from model data.
    The only currently supported native real-deploy adapter is the canonical
    scalar DMTAC command space. Other backends remain fail-closed for Active
    v22-native sessions until they publish an equivalent typed capability.
    """

    backend = _hand_backend_mode_for_side(args, side)
    if backend != HAND_BACKEND_DMTAC_GRIPPER:
        return None
    effector = deploy_layout.end_effector(side)
    action_channel = deploy_layout.channel("action", side)
    state_channel = deploy_layout.channel("state", side)
    if effector.adapter != "dmtac_gripper_v1":
        raise ValueError(
            f"{side} DMTAC runtime requires adapter 'dmtac_gripper_v1', "
            f"got {effector.adapter!r}."
        )
    if (
        action_channel.value_type != "continuous"
        or state_channel.value_type != "continuous"
        or action_channel.dim != 1
        or state_channel.dim != 1
    ):
        raise ValueError(
            f"{side} DMTAC native deployment requires matching scalar "
            "continuous action/state channels."
        )
    if action_channel.units is None or str(action_channel.units) != "normalized_0_1":
        raise ValueError(
            f"{side} DMTAC native deployment requires normalized_0_1 units."
        )
    return EndEffectorSafetyCapability(
        adapter=effector.adapter,
        value_type="continuous",
        names=action_channel.names,
        units=(str(action_channel.units),),
        lower=(0.0,),
        upper=(float(GRIPPER_ACTION_MAX),),
    )


def add_args(parser: argparse.ArgumentParser) -> None:
    """Compatibility wrapper for deployment CLI argument registration."""
    _register_deployment_cli_args(parser)


def _validate_model_rtc_executor_compatibility(
    *,
    rtc_profile_id: str,
    arm_control_mode: str,
    send_actions: bool,
) -> None:
    if (
        str(rtc_profile_id) == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1
        and str(arm_control_mode) == RM65_ARM_CONTROL_MODE_POSITION
        and bool(send_actions)
    ):
        raise ValueError(
            "conditioned Model-RTC typed canonical actions currently require "
            "--arm-execution-profile impedance_40hz; the position executor "
            "accepts only legacy filtered-vector actions."
        )


def validate_args(
    args: argparse.Namespace,
    *,
    raw_argv: tuple[str, ...] | list[str] = (),
    policy_artifact_has_arm_hold_output: bool = False,
) -> tuple[str, ...]:
    if not isinstance(policy_artifact_has_arm_hold_output, bool):
        raise TypeError("policy_artifact_has_arm_hold_output must be a bool.")
    active_sides = _parse_active_sides(args.active_sides)
    args.active_sides = active_sides
    no_init_qpos = bool(getattr(args, "no_init_qpos", False))
    if no_init_qpos and (
        args.deploy_fixed_init_dataset or args.deploy_fixed_init_parquet
    ):
        raise ValueError(
            "--no-init-qpos cannot be combined with the explicit legacy "
            "--deploy-fixed-init-* trajectory."
        )
    if no_init_qpos:
        # Deployment profiles are expanded before the delegated runtime parses
        # its arguments.  Clear their dataset-frame qpos-init inputs before any
        # mode-specific validation so one explicit trailing override works for
        # every current and future profile.  Keep the remaining startup/current-
        # pose and ready-confirm gates intact.
        args.policy_qpos_init_suppressed = bool(
            args.policy_qpos_init_dataset or args.policy_qpos_init_parquet
        )
        args.policy_qpos_init_dataset = None
        args.policy_qpos_init_parquet = None
    else:
        args.policy_qpos_init_suppressed = False
    if bool(getattr(args, "position_force_guard", False)):
        # Resolve the complete config before any backend or device is created.
        # In particular, RM65 force polling above 20 Hz must fail at CLI
        # validation rather than after a real-arm connection is opened.
        position_force_guard_config(args)
    rtc_profile_id = str(getattr(args, "rtc_profile", "off")).strip()
    rtc_profile_enabled = rtc_profile_id != "off"
    minimum_rtc_emissions = int(
        getattr(args, "min_rtc_emissions_before_exit", 0)
    )
    if minimum_rtc_emissions > 0:
        if not rtc_profile_enabled:
            raise ValueError(
                "--min-rtc-emissions-before-exit requires an explicit "
                "--rtc-profile."
            )
        if int(args.max_steps) <= 0:
            raise ValueError(
                "--min-rtc-emissions-before-exit requires bounded --max-steps."
            )
    if bool(args.position_stream_start_acquisition) and not bool(
        args.position_plan_splice_governor
    ):
        raise ValueError(
            "--position-stream-start-acquisition requires "
            "--position-plan-splice-governor."
        )
    arm_hold_output_enabled = bool(policy_artifact_has_arm_hold_output)
    if rtc_profile_enabled or arm_hold_output_enabled:
        # Neither timed RTC emissions nor arm-hold-resolved actions expose an
        # already-resolved same-chunk successor.  Resolve this compatibility
        # constraint before choosing the path-lift backend/profile so the
        # effective configuration cannot claim official lookahead or async C2.
        args.position_chunk_lookahead = False
    pose_lift_profile = resolve_rm65_pose_lift_profile(args.pose_lift_profile)
    path_lift_backend = str(args.position_path_lift_backend)
    plan_splice_dynamics_config = resolve_position_plan_splice_dynamics_config(
        args.position_plan_splice_dynamics_profile
    )
    (
        args.position_qref_acceleration_scale_resolved,
        args.position_qref_max_acceleration_rad_s2_resolved,
    ) = resolve_rm65_qref_acceleration_profile(
        args.position_qref_acceleration_profile,
        hard_max_acceleration_rad_s2=args.position_max_acceleration_rad_s2,
    )
    qref_acceleration_profile_is_comfort = not math.isclose(
        float(args.position_qref_acceleration_scale_resolved),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    qref_acceleration_profile_explicit = bool(
        _argument_was_supplied(
            list(raw_argv),
            "--position-qref-acceleration-profile",
        )
        or str(args.position_qref_acceleration_profile)
        != RM65_QREF_ACCELERATION_COMFORT_75_V1
    )
    plan_splice_dynamics_enabled = any(
        weight > 0.0
        for weight in (
            plan_splice_dynamics_config.hold_restart_initial_jerk_delta_regularization,
            plan_splice_dynamics_config.hold_restart_jerk_delta_regularization,
        )
    )
    if plan_splice_dynamics_enabled and not bool(args.position_plan_splice_governor):
        raise ValueError(
            "non-off --position-plan-splice-dynamics-profile requires "
            "--position-plan-splice-governor."
        )
    if (
        plan_splice_dynamics_enabled
        and path_lift_backend != OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND
    ):
        raise ValueError(
            "non-off --position-plan-splice-dynamics-profile requires the "
            "official source all-IK governor backend."
        )
    if bool(args.send_actions) and not pose_lift_profile.hardware_dispatch_allowed:
        raise ValueError(
            f"pose-lift profile {pose_lift_profile.name!r} forbids hardware "
            "dispatch; select a profile with hardware_dispatch_allowed=true."
        )
    if int(args.position_all_ik_lookahead_source_steps) < 2:
        raise ValueError("--position-all-ik-lookahead-source-steps must be >= 2.")
    if float(args.position_all_ik_batched_solver_tolerance) > 1e-4:
        raise ValueError("--position-all-ik-batched-solver-tolerance must be <= 1e-4.")
    continuous_waypoint_research = bool(
        str(args.position_continuous_waypoint_profile).strip() == "async_c2"
        or args.position_continuous_waypoint_shadow
        or args.position_continuous_waypoint_experimental_admission
        or args.position_continuous_waypoint_async_precompute
    )
    no_debug_deadline_soak = bool(
        args.position_continuous_waypoint_no_debug_deadline_soak
    )
    if no_debug_deadline_soak:
        soak_violations: list[str] = []
        if bool(args.debug):
            soak_violations.append("--debug must be omitted")
        if str(args.position_continuous_waypoint_profile).strip() != "async_c2":
            soak_violations.append(
                "--position-continuous-waypoint-profile must be async_c2"
            )
        if bool(args.send_actions):
            soak_violations.append("--send-actions is forbidden")
        if not (0 < int(args.max_steps) <= 900):
            soak_violations.append("--max-steps must be in [1, 900]")
        if args.position_reference_worker_cpu is None:
            soak_violations.append("--position-reference-worker-cpu is required")
        if args.position_continuous_waypoint_worker_cpu is None:
            soak_violations.append(
                "--position-continuous-waypoint-worker-cpu is required"
            )
        if not no_init_qpos:
            soak_violations.append("--no-init-qpos is required")
        if not math.isclose(
            float(args.init_time_s),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            soak_violations.append("--init-time-s 0 is required")
        if not bool(args.auto_start):
            soak_violations.append("--auto-start is required")
        if bool(args.wait_for_ready_confirm):
            soak_violations.append("--no-wait-for-ready-confirm is required")
        if soak_violations:
            raise ValueError(
                "--position-continuous-waypoint-no-debug-deadline-soak is "
                "restricted to the isolated bounded dry-run contract: "
                + "; ".join(soak_violations)
            )
    if bool(args.position_continuous_waypoint_async_precompute) and not bool(
        args.position_continuous_waypoint_experimental_admission
    ):
        raise ValueError(
            "--position-continuous-waypoint-async-precompute requires "
            "--position-continuous-waypoint-experimental-admission."
        )
    if (
        continuous_waypoint_research
        and not bool(args.debug)
        and not no_debug_deadline_soak
    ):
        raise ValueError(
            "continuous waypoint research switch requires --debug so its "
            "bounded observational results are persisted; the only "
            "exception is the explicit bounded no-debug deadline-soak "
            "contract."
        )
    if path_lift_backend == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND:
        if active_sides != ("left", "right"):
            if _argument_was_supplied(
                list(raw_argv),
                "--position-path-lift-backend",
            ):
                raise ValueError(
                    "official source all-IK governor requires active dual arms; "
                    "use --position-path-lift-backend pink_pose_lift for single-arm deploy."
                )
            # Preserve the existing single-arm compatibility surface.  The
            # promoted default is authoritative only for formal dual-arm
            # position deployment.
            args.position_path_lift_backend = PINK_POSE_LIFT_BACKEND
            path_lift_backend = PINK_POSE_LIFT_BACKEND
        if not bool(args.position_chunk_lookahead):
            if _argument_was_supplied(
                list(raw_argv),
                "--position-path-lift-backend",
            ):
                raise ValueError(
                    "official source all-IK governor requires "
                    "--position-chunk-lookahead."
                )
            args.position_path_lift_backend = PINK_POSE_LIFT_BACKEND
            path_lift_backend = PINK_POSE_LIFT_BACKEND
        if path_lift_backend == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND and not bool(
            args.position_collision_check
        ):
            if _argument_was_supplied(
                list(raw_argv),
                "--position-path-lift-backend",
            ):
                raise ValueError(
                    "official source all-IK governor requires "
                    "--position-collision-check."
                )
            args.position_path_lift_backend = PINK_POSE_LIFT_BACKEND
            path_lift_backend = PINK_POSE_LIFT_BACKEND
    if qref_acceleration_profile_is_comfort and (
        path_lift_backend != OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND
        or str(args.arm_control_mode) != RM65_ARM_CONTROL_MODE_POSITION
    ):
        if qref_acceleration_profile_explicit:
            raise ValueError(
                "non-hard --position-qref-acceleration-profile requires the "
                "effective official source all-IK governor backend."
            )
        args.position_qref_acceleration_profile = RM65_QREF_ACCELERATION_HARD_V1
        (
            args.position_qref_acceleration_scale_resolved,
            args.position_qref_max_acceleration_rad_s2_resolved,
        ) = resolve_rm65_qref_acceleration_profile(
            args.position_qref_acceleration_profile,
            hard_max_acceleration_rad_s2=(args.position_max_acceleration_rad_s2),
        )
    official_governor_effective = bool(
        path_lift_backend == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND
        and str(args.arm_control_mode) == RM65_ARM_CONTROL_MODE_POSITION
    )
    continuous_waypoint_profile = str(args.position_continuous_waypoint_profile).strip()
    continuous_waypoint_profile_explicit = _argument_was_supplied(
        list(raw_argv),
        "--position-continuous-waypoint-profile",
    )
    if (
        continuous_waypoint_profile == "async_c2"
        and continuous_waypoint_profile_explicit
        and not official_governor_effective
    ):
        raise ValueError(
            "--position-continuous-waypoint-profile async_c2 requires the "
            "effective official source all-IK governor, dual arms, chunk "
            "lookahead, and position collision checking."
        )
    if continuous_waypoint_research and not official_governor_effective:
        raise ValueError(
            "continuous waypoint research switches cannot follow an automatic "
            "fallback away from the official source all-IK governor."
        )
    production_async_c2 = bool(
        continuous_waypoint_profile == "async_c2" and official_governor_effective
    )
    args.position_continuous_waypoint_profile_effective = (
        "async_c2" if production_async_c2 else "formal_third_order"
    )
    # Keep the original research booleans as compatibility inputs, then expose
    # their effective values to the existing narrow factory/wiring surface.
    args.position_continuous_waypoint_shadow = bool(
        production_async_c2 or args.position_continuous_waypoint_shadow
    )
    args.position_continuous_waypoint_experimental_admission = bool(
        production_async_c2 or args.position_continuous_waypoint_experimental_admission
    )
    args.position_continuous_waypoint_async_precompute = bool(
        production_async_c2 or args.position_continuous_waypoint_async_precompute
    )
    _validate_position_worker_args(
        args,
        available_cpu_ids=available_cpu_ids,
        cpu_thread_sibling_ids=cpu_thread_sibling_ids,
        continuous_worker_thread=CONTINUOUS_WAYPOINT_WORKER_THREAD,
        continuous_worker_process=CONTINUOUS_WAYPOINT_WORKER_PROCESS,
    )
    observation_firewall_mode = (
        str(getattr(args, "policy_observation_firewall_mode", "strict")).strip().lower()
    )
    rtc_real_shadow = bool(getattr(args, "rtc_real_shadow", False))
    validate_rtc_cli_args(
        args,
        rtc_profile_id=rtc_profile_id,
        rtc_profile_enabled=rtc_profile_enabled,
        rtc_real_shadow=rtc_real_shadow,
        hand_backend_mode_for_side=_hand_backend_mode_for_side,
    )
    if bool(args.send_actions) and observation_firewall_mode != "strict":
        raise ValueError(
            "--send-actions requires --policy-observation-firewall-mode strict."
        )
    if (
        bool(args.send_actions)
        and str(getattr(args, "action_arbiter_mode", "off"))
        == ARBITER_SHADOW_MODE_SHADOW
    ):
        raise ValueError(
            "M3 --action-arbiter-mode shadow is dry-run/replay only; "
            "formal real shadow requires the M4B durable trace gate."
        )
    raw_tokens_list = [str(value) for value in raw_argv]
    if (
        not any(
            token in ("--arm-canfd-follow", "--no-arm-canfd-follow")
            for token in raw_tokens_list
        )
        and getattr(args, "arm_canfd_follow", None) is False
    ):
        raw_tokens_list.append("--no-arm-canfd-follow")
    if (
        not any(
            token == "--position-ik-solve-min-iterations"
            or token.startswith("--position-ik-solve-min-iterations=")
            for token in raw_tokens_list
        )
        and int(args.position_ik_solve_min_iterations) != 2
    ):
        raw_tokens_list.append("--position-ik-solve-min-iterations")
    if (
        not any(
            token == "--position-ik-solve-max-iterations"
            or token.startswith("--position-ik-solve-max-iterations=")
            for token in raw_tokens_list
        )
        and int(args.position_ik_solve_max_iterations) != 4
    ):
        raw_tokens_list.append("--position-ik-solve-max-iterations")
    raw_tokens = tuple(raw_tokens_list)
    requested_profile = getattr(args, "arm_execution_profile", None)
    position_requested = bool(
        requested_profile != "impedance_40hz"
        and str(getattr(args, "arm_control_mode", "")).strip()
        == RM65_ARM_CONTROL_MODE_POSITION
    )
    if requested_profile in (None, "") and position_requested:
        # Deployment defaults to the clock contract already validated by the
        # current position teleop. Explicit rate flags remain profile
        # overrides through resolve_real_arm_execution_config().
        args.arm_execution_profile = POSITION_15_60_120HZ_PROFILE_NAME
    explicit_compatibility_arm_profile = any(
        token == "--arm-execution-profile"
        or token.startswith("--arm-execution-profile=")
        for token in raw_tokens
    ) and requested_profile not in (None, "", POSITION_15_60_120HZ_PROFILE_NAME)
    if explicit_compatibility_arm_profile:
        args.position_control_profile_resolved = LEGACY_POSITION_CONTROL_PROFILE_NAME
        args.position_control_profile_metadata = {
            "name": LEGACY_POSITION_CONTROL_PROFILE_NAME,
            "compatibility_mode": True,
            "reason": f"explicit_arm_execution_profile:{requested_profile}",
        }
    else:
        if str(args.position_control_profile) == LEGACY_POSITION_CONTROL_PROFILE_NAME:
            if not any(
                token == "--position-ik-solve-max-iterations"
                or token.startswith("--position-ik-solve-max-iterations=")
                for token in raw_tokens
            ):
                args.position_ik_solve_max_iterations = 2
            if not any(
                token
                in (
                    "--position-ik-adaptive-iterations",
                    "--no-position-ik-adaptive-iterations",
                )
                for token in raw_tokens
            ):
                args.position_ik_adaptive_iterations = False
        apply_rm65_position_profile_defaults(
            args,
            raw_argv=raw_tokens,
            mode=(
                "impedance"
                if requested_profile == "impedance_40hz"
                else str(args.arm_control_mode)
            ),
            include_source_shaping=False,
        )
    current_position_profile = bool(
        args.arm_execution_profile == POSITION_15_60_120HZ_PROFILE_NAME
    )
    if current_position_profile and not any(
        token == "--arm-max-angvel" or token.startswith("--arm-max-angvel=")
        for token in raw_tokens
    ):
        args.arm_max_angvel = RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S
    if current_position_profile and not any(
        token == "--arm-follow-qpos-alpha"
        or token.startswith("--arm-follow-qpos-alpha=")
        for token in raw_tokens
    ):
        args.arm_follow_qpos_alpha = RM65_POSITION_TRACKING_FOLLOW_ALPHA
    execution_config = resolve_real_arm_execution_config(
        args,
        raw_argv=raw_tokens,
    )
    args._real_arm_execution_config = execution_config
    args.arm_control_mode = execution_config.mode
    args.arm_canfd_follow = execution_config.arm_canfd_follow
    args.arm_control_hz = execution_config.arm_control_hz
    args.position_reference_hz = execution_config.position_reference_hz
    args.arm_follow_qpos_alpha = execution_config.arm_follow_qpos_alpha
    if bool(args.rm65_pose_observation):
        if not bool(args.arms_enabled):
            raise ValueError("--rm65-pose-observation requires enabled arms.")
        if args.arm_control_mode != RM65_ARM_CONTROL_MODE_POSITION:
            raise ValueError(
                "--rm65-pose-observation currently supports position mode only."
            )
        if bool(args.send_actions):
            raise ValueError(
                "--rm65-pose-observation is forbidden with --send-actions until "
                "the full attestation cost has real deadline evidence."
            )
        if bool(args.allow_arm_connect_failure):
            raise ValueError(
                "--rm65-pose-observation requires every selected RM65 backend "
                "to start and attest; omit --allow-arm-connect-failure."
            )
        arm_state_refresh_hz = float(args.arm_state_refresh_hz)
        if not math.isfinite(arm_state_refresh_hz) or arm_state_refresh_hz <= 0.0:
            raise ValueError(
                "--rm65-pose-observation requires a finite "
                "--arm-state-refresh-hz > 0."
            )
        for side in active_sides:
            if not str(getattr(args, f"{side}_arm_ip", "") or "").strip():
                raise ValueError(
                    "--rm65-pose-observation requires "
                    f"--{side}-arm-ip for active side {side!r}."
                )
    tcp_pose_envelope_requested = args.tcp_pose_envelope_profile is not None
    if bool(args.tcp_pose_envelope_active) and not tcp_pose_envelope_requested:
        raise ValueError(
            "--tcp-pose-envelope-active requires --tcp-pose-envelope-profile."
        )
    if tcp_pose_envelope_requested and (
        not bool(args.arms_enabled)
        or args.arm_control_mode
        not in {
            RM65_ARM_CONTROL_MODE_POSITION,
            RM65_ARM_CONTROL_MODE_IMPEDANCE,
        }
    ):
        raise ValueError(
            "deploy TCP pose envelopes require enabled arms in position or "
            "pose-impedance mode; hand-only paths are unsupported."
        )
    if rtc_profile_enabled and args.arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION:
        rates = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.position_rates
        if not math.isclose(
            float(args.position_reference_hz),
            rates.reference_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(args.arm_control_hz),
            rates.command_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "RTC-3c position cadence is fixed at 60 Hz reference / "
                "120 Hz command."
            )
    _validate_model_rtc_executor_compatibility(
        rtc_profile_id=rtc_profile_id,
        arm_control_mode=args.arm_control_mode,
        send_actions=bool(args.send_actions),
    )
    if args.arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION and float(
        args.arm_control_hz
    ) < float(args.position_reference_hz):
        raise ValueError(
            "--arm-control-hz must be >= --position-reference-hz in position mode "
            "so every reference can be represented by the command stream."
        )
    position_collision_enabled = bool(getattr(args, "position_collision_check", True))
    if float(args.position_collision_distance_epsilon_m) > float(
        args.position_collision_min_distance_m
    ):
        raise ValueError(
            "--position-collision-distance-epsilon-m must not exceed "
            "--position-collision-min-distance-m."
        )
    if int(args.position_ik_solve_min_iterations) > int(
        args.position_ik_solve_max_iterations
    ):
        raise ValueError(
            "--position-ik-solve-min-iterations must be <= "
            "--position-ik-solve-max-iterations."
        )
    args.arm_workspace_min = _parse_vector3(args.arm_workspace_min)
    args.arm_workspace_max = _parse_vector3(args.arm_workspace_max)
    if np.any(
        np.asarray(args.arm_workspace_min, dtype=float)
        > np.asarray(args.arm_workspace_max, dtype=float)
    ):
        raise ValueError("--arm-workspace-min must not exceed --arm-workspace-max.")
    configured_hands_enabled = _hands_enabled(args)
    end_effector_sides = _parse_end_effector_sides(
        args.end_effector_sides,
        active_sides=active_sides,
        hands_enabled=configured_hands_enabled,
    )
    args.end_effector_sides = end_effector_sides
    hands_enabled = bool(end_effector_sides)
    disconnected_gripper_sides = _parse_disconnected_gripper_sides(
        getattr(args, "disconnected_gripper_sides", "none")
    )
    unavailable_disconnected = set(disconnected_gripper_sides) - set(
        end_effector_sides
    )
    if unavailable_disconnected:
        raise ValueError(
            "--disconnected-gripper-sides must be a subset of configured "
            "--end-effector-sides, got "
            + ",".join(sorted(unavailable_disconnected))
        )
    for side in disconnected_gripper_sides:
        if _hand_backend_mode_for_side(args, side) != HAND_BACKEND_DMTAC_GRIPPER:
            raise ValueError(
                f"--disconnected-gripper-sides {side} requires that side to use "
                "--hand-backend dmtac-gripper."
            )
        if str(getattr(args, f"{side}_gripper_port", "") or "").strip():
            raise ValueError(
                f"--{side}-gripper-port cannot be combined with intentionally "
                f"disconnected {side} gripper."
            )
    disconnected_state = float(getattr(args, "disconnected_gripper_state", 0.0))
    if (
        not np.isfinite(disconnected_state)
        or not 0.0 <= disconnected_state <= 1.0
    ):
        raise ValueError("--disconnected-gripper-state must be finite and in [0, 1].")
    args.disconnected_gripper_sides = disconnected_gripper_sides
    args.disconnected_gripper_state = disconnected_state
    if bool(args.allow_missing_v22_end_effectors) and hands_enabled:
        raise ValueError(
            "--allow-missing-v22-end-effectors is only valid with "
            "--end-effector-sides none for an explicit arm-only run."
        )
    if not bool(args.arms_enabled) and not hands_enabled:
        raise ValueError("Enable at least one actuator family: arms or hands.")
    validate_debug_replay_chunk_size(args)
    if float(args.init_time_s) < 0.0:
        raise ValueError("--init-time-s must be non-negative.")
    if args.init_dataset and args.init_parquet:
        raise ValueError("Use only one of --init-dataset or --init-parquet.")
    if args.deploy_fixed_init_dataset and args.deploy_fixed_init_parquet:
        raise ValueError(
            "Use only one of --deploy-fixed-init-dataset or --deploy-fixed-init-parquet."
        )
    if args.policy_qpos_init_dataset and args.policy_qpos_init_parquet:
        raise ValueError(
            "Use only one of --policy-qpos-init-dataset or --policy-qpos-init-parquet."
        )
    if args.deploy_fixed_init_dataset or args.deploy_fixed_init_parquet:
        if args.policy_qpos_init_dataset or args.policy_qpos_init_parquet:
            raise ValueError(
                "Use only one of --deploy-fixed-init-* or --policy-qpos-init-*."
            )
        if active_sides != ("right",):
            raise ValueError(
                "--deploy-fixed-init-* is currently right-arm/right-hand only; use --active-sides right."
            )
        if not bool(args.arms_enabled):
            raise ValueError("--deploy-fixed-init-* requires arms to be enabled.")
    if (
        args.policy_qpos_init_dataset or args.policy_qpos_init_parquet
    ) and not bool(args.arms_enabled):
        raise ValueError("--policy-qpos-init-* requires arms to be enabled.")
    has_dataset_qpos_start = bool(
        args.policy_qpos_init_dataset
        or args.policy_qpos_init_parquet
        or args.deploy_fixed_init_dataset
        or args.deploy_fixed_init_parquet
    )
    if (
        bool(args.send_actions)
        and bool(args.arms_enabled)
        and not has_dataset_qpos_start
        and not bool(args.start_init_qpos)
        and not no_init_qpos
    ):
        raise ValueError(
            "Real arm policy execution requires dataset-frame qpos initialization "
            "(--policy-qpos-init-dataset or --policy-qpos-init-parquet). "
            "To intentionally start from the current qpos, pass --no-init-qpos "
            "(--start-init_qpos remains supported when no profile injects qpos init)."
        )
    if (
        raw_argv
        and bool(args.send_actions)
        and bool(args.arms_enabled)
        and not any(
            token in {"--arm-execution-profile", "--arm-control-mode"}
            or token.startswith(
                ("--arm-execution-profile=", "--arm-control-mode=")
            )
            for token in raw_tokens
        )
    ):
        raise ValueError(
            "Real arm policy execution requires an explicit execution branch: "
            "pass --arm-execution-profile "
            "impedance_40hz|position_15_60_120hz|position "
            "or --arm-control-mode impedance|position."
        )
    if (
        bool(args.send_actions)
        and bool(args.arms_enabled)
        and args.arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION
        and not position_collision_enabled
    ):
        raise ValueError(
            "--send-actions in position mode requires --position-collision-check; "
            "it may only be disabled for a non-motion dry run"
        )
    if (
        bool(args.send_actions)
        and bool(args.arms_enabled)
        and args.arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION
        and active_sides != ("left", "right")
    ):
        raise ValueError(
            "High-rate coupled collision IK requires --active-sides left right (or both). "
            "A single active arm has no authoritative passive-arm qpos/TCP, so the "
            "dual-arm model must fail closed rather than guess it."
        )
    validate_debug_replay_window_and_sources(args)
    validate_tactile_cli_args(
        args,
        active_sides=active_sides,
        end_effector_sides=end_effector_sides,
        hands_enabled=hands_enabled,
        hand_backend_mode_for_side=_hand_backend_mode_for_side,
    )
    validate_camera_cli_args(args)
    return active_sides


def emergency_stop_position_deploy_on_keyboard_interrupt(
    args: argparse.Namespace,
    *,
    send_actions: bool,
    arm_backends: Mapping[str, EffectorBackend],
) -> bool:
    """Compatibility wrapper preserving the run_policy monkeypatch seam."""

    return _emergency_stop_position_deploy_on_keyboard_interrupt(
        args,
        send_actions=send_actions,
        arm_backends=arm_backends,
        emergency_stop=emergency_stop_deploy_arm_backends,
    )


def build_hand_backends(
    args: argparse.Namespace, active_sides: tuple[str, ...]
) -> dict[str, EffectorBackend]:
    if not active_sides or not _hands_enabled(args):
        return {}
    disconnected_sides = _parse_disconnected_gripper_sides(
        getattr(args, "disconnected_gripper_sides", ())
    )
    return _build_resolved_hand_backends(
        args,
        active_sides,
        backend_mode_by_side={
            side: _hand_backend_mode_for_side(args, side) for side in active_sides
        },
        dmtac_configs=_dmtac_backend_configs(args, active_sides),
        disconnected_gripper_sides=disconnected_sides,
        disconnected_gripper_policy=(
            _resolved_gripper_policy(args) if disconnected_sides else {}
        ),
    )


def hand_action_specs_for_backends(
    args: argparse.Namespace,
    backends: Mapping[str, EffectorBackend],
    active_sides: tuple[str, ...],
) -> dict[str, Any]:
    return hand_action_specs_for_modes(
        {
            side: _hand_backend_mode_for_side(args, side) for side in active_sides
        },
        active_sides,
    )


def policy_qpos_init_hand_action_specs(
    args: argparse.Namespace,
    hand_action_specs: Mapping[str, Any],
) -> dict[str, Any]:
    return filter_policy_qpos_init_hand_action_specs(
        hand_action_specs,
        disconnected_gripper_sides=_parse_disconnected_gripper_sides(
            getattr(args, "disconnected_gripper_sides", ())
        ),
    )


def _resolve_action_queue_policy(
    args: argparse.Namespace, *, debug_replay: bool
) -> str:
    """Compatibility wrapper for the shared queue-policy resolver."""

    return resolve_action_queue_policy(
        action_queue_policy=getattr(args, "action_queue_policy", "auto"),
        append_action_chunks=bool(getattr(args, "append_action_chunks", False)),
        debug_replay=bool(debug_replay),
    )


def _resolve_runtime_pose_frame_contract(
    configured_contract: Any | None,
    *,
    policy_artifact: Any | None,
    policy_metadata: Mapping[str, Any],
) -> tuple[Any | None, str]:
    """Resolve the RM65 workspace-near-boundary default without changing v2.2 semantics."""

    if configured_contract is not None:
        return configured_contract, "explicit"

    pose_training_data = (
        None if policy_artifact is None else policy_artifact.pose_training_data
    )
    if (
        pose_training_data is not None
        and pose_training_data.origin == POSE_TRAINING_ORIGIN_NATIVE_V23
    ):
        return (
            default_rm65_workspace_near_boundary_runtime_pose_frame_contract(),
            "default_rm65_dual_arm_workspace_near_boundary_v1",
        )
    if policy_artifact is not None:
        # In particular, v2.2 raw-identity compatibility must stay identity.
        return None, "v22_identity_compatibility"

    dataset_info = policy_metadata.get("dataset_info")
    if (
        isinstance(dataset_info, Mapping)
        and str(dataset_info.get("dexteleop_dataset_schema") or "")
        == DEXTELEOP_DATASET_SCHEMA_V23
    ):
        return (
            default_rm65_workspace_near_boundary_runtime_pose_frame_contract(),
            "default_rm65_dual_arm_workspace_near_boundary_v1",
        )
    return None, "not_applicable"


def _validate_debug_replay_arm_control_mode(
    policy: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload, warning = bind_debug_replay_arm_control_mode(
        policy,
        selected_mode=args.arm_control_mode,
        hardware_output=bool(args.send_actions),
    )
    if warning:
        print(f"[ReplayMode] WARNING: {warning}", flush=True)
    return payload


def _announce_policy_qpos_init_complete(
    *,
    wait_for_ready_confirm: bool,
    plan_only: bool,
    now: float,
) -> None:
    """Print the audited qpos boundary before policy inference or execution."""

    if plan_only:
        console_log_event(
            "Init",
            "READY: G PLAN-ONLY VALIDATION COMPLETE - dataset-frame qpos-init "
            "planning and collision checks passed; no arm or end-effector "
            "command was sent.",
            now=now,
        )
    else:
        console_log_event(
            "Init",
            "READY: G INITIALIZATION COMPLETE - arms and end effectors reached "
            "the dataset start state.",
            now=now,
        )
    if wait_for_ready_confirm:
        console_log_event(
            "Policy",
            "WAIT: PRESS A to start policy inference/execution.",
            now=now,
        )
    else:
        console_log_event(
            "Policy",
            "READY: policy inference/execution released automatically.",
            now=now,
        )


def _begin_policy_qpos_init_with_force_guard(
    controller: DeployPolicyQposInitController,
    *,
    monitor: PositionForceGuardMonitor | None,
    arm_backends: Mapping[str, EffectorBackend],
    now: float,
) -> str:
    """Plan qpos init without admitting motion while force reads are blocked."""

    if monitor is None:
        return controller.begin(arm_backends=arm_backends, now=now)
    token = monitor.suspend_stale_watchdog_for_non_motion()
    print(
        "[PositionForceGuard] stale watchdog paused for non-motion qpos-init "
        f"planning token={token}; force limits/read errors remain active.",
        flush=True,
    )
    try:
        message = controller.begin(arm_backends=arm_backends, now=now)
    finally:
        snapshot = monitor.resume_stale_watchdog_after_non_motion(token)
        print(
            "[PositionForceGuard] qpos-init planning window closed; fresh "
            "post-plan samples received for sides="
            f"{sorted(snapshot.get('last_reading_by_side', {}))}; stale "
            "watchdog re-armed before motion.",
            flush=True,
        )
    return message


def _build_arm_state_reader(
    *,
    args: argparse.Namespace,
    arm_backends: Mapping[str, EffectorBackend],
    pose_observation: RM65PoseObservationAssembly | None,
) -> ArmStateReader:
    """Construct the one deployment reader from an optional immutable binding."""

    pose_kwargs: dict[str, Any] = {}
    if pose_observation is not None:
        pose_kwargs = {
            "pose_projection_by_side": pose_observation.projection_by_side,
            "pose_state_read_by_side": pose_observation.state_read_by_side,
            "receipt_clock": pose_observation.receipt_clock,
            "receipt_monotonic_ns_clock": (
                pose_observation.receipt_monotonic_ns_clock
            ),
        }
    return ArmStateReader(
        arm_backends,
        refresh_hz=float(args.arm_state_refresh_hz),
        compute_tcp_pose=bool(args.arms_enabled),
        **pose_kwargs,
    )


class _RuntimeCanonicalPolicyActionResolver(RobotActionResolver):
    """Resolve typed policy actions to the selected real backend protocol."""

    def __init__(self, session: ResolvedDeploySessionContract) -> None:
        self._session = session

    def resolve(
        self,
        action: RobotAction,
        *,
        observation: Any = None,
        context: Any = None,
    ) -> TeleopCommand:
        command = super().resolve(
            action,
            observation=observation,
            context=context,
        )
        for side, target in action.end_effector_targets_by_side.items():
            capability = self._session.runtime_end_effectors.get(side)
            if capability is None or not capability.enabled:
                raise RuntimeError(
                    f"Typed {side} end-effector target has no enabled "
                    "runtime capability."
                )
            effector_command = command.commands.get(f"{side}_hand")
            if effector_command is None:
                raise RuntimeError(
                    f"Typed {side} end-effector target produced no command."
                )
            if (
                capability.backend != HAND_BACKEND_DMTAC_GRIPPER
                or capability.adapter != "dmtac_gripper_v1"
                or target.value_type != "continuous"
                or target.values.shape != (1,)
            ):
                raise RuntimeError(
                    f"Typed {side} end-effector target is unsupported by runtime "
                    f"backend {capability.backend!r}."
                )
            effector_command.metadata["canonical_command_type"] = (
                effector_command.command_type
            )
            effector_command.metadata["runtime_end_effector_backend"] = (
                capability.backend
            )
            effector_command.command_type = "gripper_scalar"
            effector_command.value = np.asarray(
                target.values,
                dtype=np.float32,
            ).reshape(1)
        return command


class _ModelRtcCanonicalActionPreparer:
    """Prepare conditioned model-RTC actions for the real runtime boundary.

    The formal RTC emission retains the model's immutable canonical action.
    This application adapter converts canonical-desk arm poses to runtime raw,
    anchors safety only to the current measured observation, and then delegates
    the actual bounds and command resolution to ``CanonicalPolicyActionPipeline``.
    """

    def __init__(
        self,
        *,
        deploy_session_contract: ResolvedDeploySessionContract,
        safety_filter: ActionSafetyFilter,
    ) -> None:
        layout = deploy_session_contract.policy_io_layout
        if layout is None:
            raise ValueError(
                "Model-RTC typed preparation requires a policy I/O layout."
            )
        end_effector_capabilities: dict[str, EndEffectorSafetyCapability] = {}
        for side in deploy_session_contract.end_effector_sides:
            runtime = deploy_session_contract.runtime_end_effectors.get(side)
            if runtime is None or not runtime.enabled or runtime.safety is None:
                raise ValueError(
                    f"Model-RTC typed preparation requires {side} runtime "
                    "end-effector safety capability."
                )
            end_effector_capabilities[side] = runtime.safety
        config = safety_filter.config
        self._session = deploy_session_contract
        self._layout = layout
        self._state_adapter = V22DeploymentAdapter(layout)
        self.pipeline = CanonicalPolicyActionPipeline(
            safety_profile=CanonicalKeyframeSafetyProfile(
                profile_id="model-rtc-real-canonical-keyframe-v1",
                limits=PositionKeyframeSafetyConfig(
                    workspace_min=config.workspace_min,
                    workspace_max=config.workspace_max,
                    workspace_soft_margin_m=config.workspace_soft_margin_m,
                    max_translation_step_m=config.max_arm_xyz_step_m,
                    max_rotation_step_deg=config.max_arm_rot_step_deg,
                    max_joint_step_rad=config.max_arm_joint_step_rad,
                    joint_lower_rad=config.arm_joint_lower_rad,
                    joint_upper_rad=config.arm_joint_upper_rad,
                ),
                end_effector_capabilities_by_side=end_effector_capabilities,
            ),
            anchor_provider=self._measured_anchor,
            resolver=_RuntimeCanonicalPolicyActionResolver(
                deploy_session_contract
            ),
        )

    def _runtime_action(
        self,
        action: CanonicalPolicyAction,
    ) -> CanonicalPolicyAction:
        if self._session.policy_absolute_frame != POLICY_ABSOLUTE_FRAME_CANONICAL_DESK:
            return action
        pose_contract = self._session.runtime_pose_frame_contract
        if pose_contract is None:
            raise RuntimeError(
                "Canonical-desk model-RTC action has no runtime pose-frame contract."
            )
        runtime_poses = {
            side: np.asarray(
                pose_contract.canonical_desk_pose_vector_to_storage(
                    pose,
                    side=side,
                ),
                dtype=np.float32,
            )
            for side, pose in action.action.arm_pose_by_side.items()
        }
        runtime_frames = dict(action.action.frame_by_effector)
        for side in runtime_poses:
            runtime_frames[f"{side}_arm"] = str(
                self._layout.arm_reference_frame_by_side[side]
            )
        source_receipt = action.canonicalization_receipt
        runtime_robot_action = replace(
            action.action,
            arm_pose_by_side=runtime_poses,
            frame_by_effector=runtime_frames,
        )
        return build_canonical_policy_action(
            runtime_robot_action,
            source_pose_mode="absolute",
            observation_cut_id=source_receipt.observation_cut_id,
            source_action_hash=action.semantic_content_hash,
            action_codec_hash=source_receipt.action_codec_hash,
            frame_contract_hash=source_receipt.frame_contract_hash,
            layout_hash=source_receipt.layout_hash,
            normalization_hash=source_receipt.normalization_hash,
            conversion_contract_hash=canonical_content_hash(
                {
                    "adapter": "model_rtc_canonical_desk_to_runtime_raw_v1",
                    "source_conversion_contract_hash": (
                        source_receipt.conversion_contract_hash
                    ),
                    "source_canonical_policy_action_hash": (
                        action.semantic_content_hash
                    ),
                    "runtime_pose_frame_contract_hash": pose_contract.contract_hash,
                },
                domain="dexteleop.model_rtc.runtime_action_conversion.v1",
            ),
        )

    def _measured_end_effector_targets(
        self,
        action: RobotAction,
        observation: DeploymentObservation,
    ) -> dict[str, EndEffectorTarget]:
        if not action.end_effector_targets_by_side:
            return {}
        typed_observation = self._state_adapter.observation_from_deployment(
            observation
        )
        unpacked = self._layout.unpack(
            "state",
            continuous=typed_observation.state,
            discrete=typed_observation.state_discrete,
        )
        measured_by_side = unpacked["end_effector_by_side"]
        targets: dict[str, EndEffectorTarget] = {}
        for side in action.end_effector_targets_by_side:
            capability = (
                self.pipeline.safety_profile.end_effector_capabilities_by_side.get(
                    side
                )
            )
            if capability is None:
                raise RuntimeError(
                    f"Measured {side} end-effector anchor has no safety capability."
                )
            state_channel = self._layout.channel("state", side)
            if state_channel.value_type != capability.value_type:
                raise RuntimeError(
                    f"Measured {side} end-effector state/action value types differ."
                )
            values = measured_by_side.get(side)
            if values is None:
                raise RuntimeError(
                    f"Measured {side} end-effector state component is unavailable."
                )
            targets[side] = EndEffectorTarget(
                value_type=capability.value_type,
                values=np.asarray(values),
                adapter=capability.adapter,
                names=capability.names,
                metadata={"units": capability.units},
            )
        return targets

    def _measured_anchor(
        self,
        action: CanonicalPolicyAction,
        observation: DeploymentObservation,
        _target_effective_at: float,
    ) -> CanonicalActionSafetyAnchor:
        robot_action = action.action
        if robot_action.arm_qpos_by_side or robot_action.hand_qpos_by_side:
            raise RuntimeError(
                "Model-RTC production preparation only accepts arm-pose and "
                "typed end-effector actions."
            )
        arm_poses: dict[str, np.ndarray] = {}
        frames: dict[str, str] = {}
        for side in robot_action.arm_pose_by_side:
            measured = np.asarray(
                observation.state[STATE_ARM_POSE_SLICES[side]],
                dtype=np.float32,
            ).reshape(7)
            if not np.all(np.isfinite(measured)):
                raise RuntimeError(
                    f"Measured {side} runtime arm pose is unavailable."
                )
            frame = robot_action.arm_frame(side)
            if frame is None or not str(frame).strip():
                raise RuntimeError(
                    f"Runtime {side} arm action frame is unavailable."
                )
            arm_poses[side] = measured
            frames[side] = str(frame)
        return CanonicalActionSafetyAnchor(
            anchor_id=(
                "model-rtc-measured-runtime:"
                f"observation:{observation.step_index}:"
                f"action:{action.canonical_action_hash}"
            ),
            anchor_kind="measured_state",
            action_space=robot_action.action_space,
            arm_pose_by_side=arm_poses,
            end_effector_targets_by_side=(
                self._measured_end_effector_targets(
                    robot_action,
                    observation,
                )
            ),
            frame_by_side=frames,
        )

    def prepare_canonical(
        self,
        action: CanonicalPolicyAction,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
        input_action_id: str,
        target_effective_at: float,
    ) -> CanonicalPolicyActionResult:
        return self.pipeline.prepare_canonical(
            self._runtime_action(action),
            observation,
            policy_name=policy_name,
            input_action_id=input_action_id,
            target_effective_at=target_effective_at,
        )


def _rtc_action_preparer_for_profile(
    *,
    rtc_profile_id: str,
    canonical_action_pipeline: CanonicalDeployActionPipeline,
    deploy_session_contract: ResolvedDeploySessionContract,
    safety_filter: ActionSafetyFilter,
) -> CanonicalDeployActionPipeline | _ModelRtcCanonicalActionPreparer:
    """Select typed preparation only for the conditioned Model-RTC profile."""

    if str(rtc_profile_id) != MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1:
        return canonical_action_pipeline
    return _ModelRtcCanonicalActionPreparer(
        deploy_session_contract=deploy_session_contract,
        safety_filter=safety_filter,
    )


def _position_executor_for_prepared_action(
    *,
    rtc_profile_id: str,
    send_actions: bool,
    position_deploy_executor: Any | None,
) -> Any | None:
    """Keep passive typed Model-RTC out of the legacy vector planner."""

    if (
        str(rtc_profile_id) == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1
        and not bool(send_actions)
    ):
        return None
    return position_deploy_executor


def _prepared_action_last_action(
    prepared_action: CanonicalDeployActionResult | CanonicalPolicyActionResult,
    observation: DeploymentObservation,
    *,
    deploy_layout: Any | None,
) -> np.ndarray:
    """Project an accepted action into the legacy observation-only last-action view."""

    if isinstance(prepared_action, CanonicalDeployActionResult):
        return prepared_action.action.vector.astype(np.float32, copy=True)
    safe_action = prepared_action.robot_action
    if safe_action is None:
        return observation.hold_action()
    projected = observation.hold_action()
    for side, pose in safe_action.arm_pose_by_side.items():
        projected[ACTION_ARM_POSE_SLICES[side]] = np.asarray(
            pose,
            dtype=np.float32,
        )
    if safe_action.end_effector_targets_by_side:
        if deploy_layout is None:
            raise RuntimeError(
                "Typed end-effector last-action projection requires a policy "
                "I/O layout."
            )
        for side, target in safe_action.end_effector_targets_by_side.items():
            channel = deploy_layout.channel("action", side)
            slots = channel.legacy_q6_slot_indices
            if target.value_type != "continuous" or slots is None:
                raise RuntimeError(
                    f"Typed {side} end-effector target has no lossless legacy "
                    "last-action observation projection."
                )
            hand = np.asarray(
                projected[ACTION_HAND_SLICES[side]],
                dtype=np.float32,
            ).copy()
            if not np.all(np.isfinite(hand)):
                hand = np.zeros(6, dtype=np.float32)
            hand[np.asarray(slots, dtype=int)] = np.asarray(
                target.values,
                dtype=np.float32,
            )
            projected[ACTION_HAND_SLICES[side]] = hand
    return projected.astype(np.float32, copy=False)


def _hold_active_rtc_gap(
    *,
    reason: str | None,
    send_actions: bool,
    arm_backends: Mapping[str, EffectorBackend],
    position_deploy_executor: Any | None,
) -> dict[str, dict[str, object]]:
    """Hold measured qpos when an Active RTC producer has no safe plan.

    Normal between-keyframe ticks pass ``reason=None`` and remain untouched.
    A real empty-plan underrun or non-eligible producer first clears any
    position plan, then leases the latest measured qpos in each impedance
    backend.  A backend without that seam is not silently admitted: all arms
    are asked to enter their stronger Hold before the error escapes.
    """

    normalized_reason = None if reason is None else str(reason).strip()
    if not send_actions or not normalized_reason:
        return {}
    if position_deploy_executor is not None:
        position_deploy_executor.clear()
    try:
        receipts = request_policy_inference_soft_hold(
            arm_backends,
            reason=normalized_reason,
        )
        unsupported = tuple(
            sorted(
                side
                for side, receipt in receipts.items()
                if str(receipt.get("status") or "") == "unsupported"
            )
        )
        if unsupported:
            raise RuntimeError(
                "Active RTC measured-qpos Hold is unsupported for sides: "
                + ", ".join(unsupported)
            )
    except Exception:
        hold_deploy_arm_backends(
            arm_backends,
            reason=f"{normalized_reason}:measured_qpos_hold_failed",
        )
        raise
    return receipts


def _rtc_active_completion_latency_sec(
    *,
    inference_started_at: float,
    released_at: float,
) -> float:
    """Return one trustworthy end-to-end Active RTC latency sample."""

    started = float(inference_started_at)
    released = float(released_at)
    if not math.isfinite(started) or not math.isfinite(released):
        raise ValueError("RTC Active completion timestamps must be finite.")
    if released < started:
        raise ValueError("RTC Active completion release predates inference start.")
    return released - started


def _rtc_active_latency_gate_result(
    samples_sec: list[float] | tuple[float, ...],
    *,
    discard_completions: int,
    required_completions: int,
    maximum_latency_sec: float,
) -> tuple[bool, bool, float | None]:
    """Return ``(complete, accepted, observed_max)`` for Active RTC warmup.

    ``maximum_latency_sec`` is an exclusive upper bound: an observation equal
    to the configured threshold is not below the latency gate.
    """

    discard = int(discard_completions)
    required = int(required_completions)
    maximum = float(maximum_latency_sec)
    if discard < 0 or required <= 0 or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("RTC Active latency gate configuration is invalid.")
    normalized = tuple(float(value) for value in samples_sec)
    if any(not math.isfinite(value) or value < 0 for value in normalized):
        raise ValueError("RTC Active latency samples must be finite and non-negative.")
    admitted = normalized[discard : discard + required]
    if len(admitted) < required:
        return False, False, None
    observed_max = max(admitted)
    return True, observed_max < maximum, observed_max


def _bounded_policy_step_limit_reached(
    *,
    max_steps: int,
    control_loop_steps: int,
    rtc_active_dispatch_counting: bool,
    accepted_rtc_dispatch_steps: int,
) -> bool:
    """Apply ``--max-steps`` to dispatches only for hardware Active RTC."""

    limit = int(max_steps)
    loop_steps = int(control_loop_steps)
    dispatch_steps = int(accepted_rtc_dispatch_steps)
    if loop_steps < 0 or dispatch_steps < 0:
        raise ValueError("Bounded policy step counters must be non-negative.")
    if limit <= 0:
        return False
    observed = dispatch_steps if rtc_active_dispatch_counting else loop_steps
    return observed >= limit


def _prepare_position_lookahead_chain(
    raw_actions: tuple[NormalizedAction, ...],
    observation: DeploymentObservation,
    current_action: CanonicalDeployActionResult,
    *,
    pipeline: CanonicalDeployActionPipeline,
    policy_name: str,
    phase_timing: dict[str, Any],
) -> tuple[tuple[CanonicalDeployActionResult, DeploymentObservation], ...]:
    """Prepare causal same-chunk successors until the first safety fault.

    Each successor is filtered against the immediately preceding safe action.
    This is chained canonical validation, not an extra temporal smoothing
    stage. The returned sequence contains only fault-free candidates; timing
    still includes the first rejected candidate so diagnostics preserve the
    existing per-stage contract.
    """

    prepared: list[tuple[CanonicalDeployActionResult, DeploymentObservation]] = []
    previous_action = current_action
    for offset, raw_action in enumerate(raw_actions, start=1):
        successor_observation = replace(
            observation,
            last_action=previous_action.action.vector.astype(
                np.float32,
                copy=True,
            ),
        )
        prepare_started = time.monotonic()
        successor = pipeline.prepare(
            raw_action,
            successor_observation,
            policy_name=policy_name,
        )
        phase_timing[f"lookahead_prepare_{offset}_ms"] = max(
            0.0,
            time.monotonic() - prepare_started,
        ) * 1000.0
        if successor.faults:
            break
        prepared.append((successor, successor_observation))
        previous_action = successor
    return tuple(prepared)


def main(
    argv: list[str] | None = None,
    *,
    rtc_step_fanout: RtcFrontendStepFanout | None = None,
    rtc_emission_fanout: PolicyActionEmissionFanout | None = None,
    rtc_emission_publisher: RtcPolicyEmissionPublisher | None = None,
    rtc_model_adapter_factory: Callable[..., Any] | None = None,
    policy_emission_observer: PassivePolicyEmissionObserverPort | None = None,
    legacy_worker_binding: RM65LegacyWorkerExecutionBinding | None = None,
    quest_paired_pose_capture_factory: QuestPairedPoseCaptureFactory | None = None,
) -> int:
    if rtc_step_fanout is not None and not isinstance(
        rtc_step_fanout,
        RtcFrontendStepFanout,
    ):
        raise TypeError("rtc_step_fanout must be an RtcFrontendStepFanout or None.")
    if rtc_emission_fanout is not None and not isinstance(
        rtc_emission_fanout,
        PolicyActionEmissionFanout,
    ):
        raise TypeError(
            "rtc_emission_fanout must be a PolicyActionEmissionFanout or None."
        )
    if rtc_emission_publisher is not None and not isinstance(
        rtc_emission_publisher,
        RtcPolicyEmissionPublisher,
    ):
        raise TypeError(
            "rtc_emission_publisher must be an " "RtcPolicyEmissionPublisher or None."
        )
    if rtc_emission_fanout is not None and rtc_emission_publisher is not None:
        raise ValueError(
            "Use rtc_emission_publisher or the compatibility "
            "rtc_emission_fanout, not both."
        )
    if policy_emission_observer is not None:
        require_passive_policy_emission_observer(policy_emission_observer)
    if legacy_worker_binding is not None and not isinstance(
        legacy_worker_binding,
        RM65LegacyWorkerExecutionBinding,
    ):
        raise TypeError(
            "legacy_worker_binding must be an RM65LegacyWorkerExecutionBinding or None."
        )
    if quest_paired_pose_capture_factory is not None and not isinstance(
        quest_paired_pose_capture_factory,
        QuestPairedPoseCaptureFactory,
    ):
        raise TypeError(
            "quest_paired_pose_capture_factory must be a "
            "QuestPairedPoseCaptureFactory or None."
        )
    parser = argparse.ArgumentParser(
        description="Deploy a policy on teleop_runtime RM65 arms, optional InspireTac hands, and RealSense."
    )
    add_args(parser)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    if legacy_worker_binding is not None:
        args._rm65_legacy_worker_binding = legacy_worker_binding
    debug_replay = bool(args.debug_replay_dataset or args.debug_replay_parquet)
    policy_kwargs = {} if debug_replay else parse_policy_kwargs(args.policy_kwargs_json)
    policy_artifact = (
        None
        if args.policy_artifact_json is None
        else load_policy_artifact(args.policy_artifact_json)
    )
    kwargs_artifact = policy_kwargs.get("policy_artifact")
    if kwargs_artifact is not None:
        kwargs_contract = policy_artifact_from_metadata(kwargs_artifact)
        if (
            policy_artifact is not None
            and kwargs_contract.to_dict() != policy_artifact.to_dict()
        ):
            raise ValueError(
                "--policy-kwargs-json policy_artifact disagrees with --policy-artifact-json."
            )
        policy_artifact = kwargs_contract
    _bind_policy_pose_mode_from_artifact(
        args,
        policy_artifact,
        raw_argv=raw_argv,
    )
    if debug_replay and str(args.debug_replay_output_pose_mode) != str(
        args.policy_action_pose_mode
    ):
        raise ValueError(
            "debug replay output pose mode must match policy action pose mode: "
            f"{args.debug_replay_output_pose_mode!r} != {args.policy_action_pose_mode!r}"
        )
    if policy_artifact is not None:
        layout_sides = tuple(
            side
            for side in ALL_SIDES
            if policy_artifact.policy_io_layout.end_effector(side).present
        )
        if not any(
            token == "--end-effector-sides" or token.startswith("--end-effector-sides=")
            for token in raw_argv
        ):
            args.end_effector_sides = (
                "none" if not layout_sides else " ".join(layout_sides)
            )
        if args.hands_enabled is None:
            args.hands_enabled = bool(layout_sides)
        if policy_artifact.arm_hold_output is not None and str(
            args.rtc_profile
        ) not in {"off", RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1}:
            raise ValueError(
                "arm-hold action semantics support sync deployment and "
                "raw-RTC final-chunk shadow only; model continuation and "
                "virtual RTC profiles remain fail closed."
            )
    active_sides = validate_args(
        args,
        raw_argv=raw_argv,
        policy_artifact_has_arm_hold_output=bool(
            policy_artifact is not None and policy_artifact.arm_hold_output is not None
        ),
    )
    if quest_paired_pose_capture_factory is not None and not bool(
        args.rm65_pose_observation
    ):
        raise ValueError(
            "quest_paired_pose_capture_factory requires --rm65-pose-observation."
        )
    if rtc_step_fanout is not None and str(args.rtc_profile) != (
        RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1
    ):
        raise ValueError("rtc_step_fanout requires the explicit RTC-3c profile.")
    if rtc_emission_fanout is not None and str(args.rtc_profile) == "off":
        raise ValueError("rtc_emission_fanout requires an explicit RTC profile.")
    if rtc_emission_publisher is not None and str(args.rtc_profile) == "off":
        raise ValueError("rtc_emission_publisher requires an explicit RTC profile.")
    if rtc_model_adapter_factory is not None and not callable(
        rtc_model_adapter_factory
    ):
        raise TypeError("rtc_model_adapter_factory must be callable or None.")
    selected_camera_keys = resolve_policy_camera_input_keys(
        args,
        policy_artifact,
        raw_argv=raw_argv,
    )
    end_effector_sides = tuple(args.end_effector_sides)
    hands_enabled = bool(end_effector_sides)
    if debug_replay:
        policy = load_dataset_replay_policy(
            dataset_root=args.debug_replay_dataset,
            parquet_path=args.debug_replay_parquet,
            episode_index=int(args.debug_replay_episode),
            action_column=str(args.debug_replay_action_column),
            start_index=int(args.debug_replay_start_index),
            end_index=args.debug_replay_end_index,
            chunk_size=int(args.debug_replay_chunk_size),
            loop=bool(args.debug_replay_loop),
            active_sides=active_sides,
            end_effector_sides=end_effector_sides,
            output_pose_mode=str(args.debug_replay_output_pose_mode),
            policy_io_layout=(
                None if policy_artifact is None else policy_artifact.policy_io_layout
            ),
        )
        _validate_debug_replay_arm_control_mode(policy, args)
    else:
        if policy_artifact is not None:
            policy_kwargs["policy_artifact"] = policy_artifact.to_dict()
        policy = load_policy(args.policy_spec, kwargs=policy_kwargs)
    deploy_layout = (
        None if policy_artifact is None else policy_artifact.policy_io_layout
    )
    if deploy_layout is None and debug_replay:
        dataset_info = policy.metadata.get("dataset_info")
        if isinstance(dataset_info, Mapping) and str(
            dataset_info.get("dexteleop_dataset_schema") or ""
        ) in {DEXTELEOP_DATASET_SCHEMA_V22, DEXTELEOP_DATASET_SCHEMA_V23}:
            deploy_layout = policy_io_layout_from_metadata(
                dataset_info,
                allow_legacy_v21=False,
            )
    runtime_capabilities = None
    if deploy_layout is not None:
        runtime_capabilities = {
            side: RuntimeEndEffectorCapability(
                side=side,
                adapter=deploy_layout.end_effector(side).adapter,
                backend=_hand_backend_mode_for_side(args, side),
                safety=_runtime_end_effector_safety_capability(
                    args=args,
                    deploy_layout=deploy_layout,
                    side=side,
                ),
            )
            for side in end_effector_sides
        }
    (
        runtime_pose_frame_contract,
        runtime_pose_frame_contract_source,
    ) = _resolve_runtime_pose_frame_contract(
        args.runtime_pose_frame_contract,
        policy_artifact=policy_artifact,
        policy_metadata=getattr(policy, "metadata", {}),
    )
    deploy_session_contract = resolve_deploy_session_contract(
        active_sides=active_sides,
        model_metadata=getattr(policy, "metadata", {}),
        policy_artifact=policy_artifact,
        policy_io_layout=deploy_layout,
        runtime_pose_frame_contract=runtime_pose_frame_contract,
        runtime_end_effectors=runtime_capabilities,
        hardware_output=bool(args.send_actions),
        allow_missing_end_effectors=bool(args.allow_missing_v22_end_effectors),
    )
    deploy_tcp_pose_envelope_projector: DeployTCPPoseEnvelopeProjector | None = None
    tcp_pose_envelope_requested = args.tcp_pose_envelope_profile is not None
    if tcp_pose_envelope_requested:
        pose_frame_contract = deploy_session_contract.runtime_pose_frame_contract
        if pose_frame_contract is None:
            raise ValueError(
                "deploy TCP pose envelopes require an explicit runtime pose-frame "
                "contract so arm-base targets can be projected in canonical desk."
            )
        envelope_file_spec = load_tcp_pose_envelope_file_spec(
            args.tcp_pose_envelope_profile,
            active_sides=active_sides,
        )
        deploy_tcp_pose_envelope_projector = DeployTCPPoseEnvelopeProjector(
            file_spec=envelope_file_spec,
            pose_frame_contract=pose_frame_contract,
            active_sides=active_sides,
            apply_to_target=bool(args.tcp_pose_envelope_active),
        )
        print(
            "[TCPPoseEnvelope] deploy "
            + ("ACTIVE" if args.tcp_pose_envelope_active else "SHADOW")
            + " configured; frame=canonical_desk controlled=tool_tcp "
            + f"profile={envelope_file_spec.profile_id!r}; binding is deferred "
            "to a fresh runtime-start Pose.",
            flush=True,
        )
    policy_observation_contract = _policy_observation_contract(
        policy=policy,
        policy_artifact=policy_artifact,
        deploy_layout=deploy_layout,
        pose_mode=str(args.policy_action_pose_mode),
    )
    policy_deploy_runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=active_sides,
            arms_enabled=bool(args.arms_enabled),
            hands_enabled=hands_enabled,
            policy_action_pose_mode=str(args.policy_action_pose_mode),
            policy_absolute_frame=str(deploy_session_contract.policy_absolute_frame),
            runtime_pose_frame_contract=deploy_session_contract.runtime_pose_frame_contract,
            policy_observation_contract=policy_observation_contract,
            observation_firewall_mode=str(
                args.policy_observation_firewall_mode
            ).replace("-", "_"),
            action_chunk_execution_steps=(
                None
                if getattr(args, "action_chunk_execution_steps", None) is None
                else int(args.action_chunk_execution_steps)
            ),
            action_keyframe_period_sec=_runtime_action_keyframe_period_sec(args),
            arm_hold_output=(
                None if policy_artifact is None else policy_artifact.arm_hold_output
            ),
        )
    )
    policy.metadata["resolved_deploy_session_contract"] = (
        deploy_session_contract.to_dict()
    )
    policy.metadata["policy_observation_firewall"] = (
        policy_deploy_runtime.observation_firewall_manifest()
    )
    policy.metadata["runtime_pose_frame_contract_source"] = (
        runtime_pose_frame_contract_source
    )
    policy.metadata["action_chunk_execution_steps"] = (
        policy_deploy_runtime.config.action_chunk_execution_steps
    )
    if (
        runtime_pose_frame_contract_source
        == "default_rm65_dual_arm_workspace_near_boundary_v1"
    ):
        print(
            "[Deploy] runtime pose frame default: dual RM65 calibrated URDF "
            "workspace_near_boundary_frame.",
            flush=True,
        )
    if (
        deploy_session_contract.policy_contract_mode == POLICY_CONTRACT_V22_NATIVE
        and str(args.rtc_profile)
        != MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1
    ):
        raise RuntimeError(
            "v22_native typed RobotAction transport is available only through "
            "the conditioned Model-RTC safety/dispatch adapter in this "
            "run_policy loop. Use that profile or an explicit "
            "legacy_q6_slot_indices bridge."
        )
    action_queue_policy = _resolve_action_queue_policy(args, debug_replay=debug_replay)
    startup_init_action, startup_init_source = resolve_startup_init_action(args, policy)
    if startup_init_action is not None:
        startup_absolute_frame = deploy_session_contract.policy_absolute_frame
        if "policy_absolute_frame=runtime_raw" in startup_init_source:
            startup_absolute_frame = "runtime_raw"
        elif "policy_absolute_frame=canonical_desk" in startup_init_source:
            startup_absolute_frame = "canonical_desk"
        startup_init_action = convert_policy_absolute_vector_to_runtime_raw(
            startup_init_action,
            active_sides=active_sides,
            policy_absolute_frame=startup_absolute_frame,
            runtime_pose_frame_contract=(
                deploy_session_contract.runtime_pose_frame_contract
            ),
        )
    fixed_init_plan = load_deploy_fixed_init_plan(
        dataset_root=args.deploy_fixed_init_dataset,
        parquet_path=args.deploy_fixed_init_parquet,
        episode_spec=str(args.deploy_fixed_init_episode),
        action_column=str(args.deploy_fixed_init_action_column),
        active_sides=active_sides,
    )
    policy_qpos_init_plan = load_deploy_policy_qpos_init_plan(
        dataset_root=args.policy_qpos_init_dataset,
        parquet_path=args.policy_qpos_init_parquet,
        episode_index=int(args.policy_qpos_init_episode),
        frame_index=int(args.policy_qpos_init_frame),
        action_column=str(args.policy_qpos_init_action_column),
        active_sides=active_sides,
        qpos_init_source=str(args.policy_qpos_init_source),
    )

    effective_action_hz, action_hz_info = resolve_effective_action_hz(args, policy)
    effective_policy_hz, policy_hz_source = resolve_effective_policy_hz(
        args,
        action_hz=float(effective_action_hz),
        action_hz_source=str(action_hz_info.get("source", "")),
    )

    policy_arbiter_shadow: PassivePolicyArbiterShadow | None = None
    replay_proposal_shadow_adapter: DatasetReplayProposalShadowAdapter | None = None
    if str(args.action_arbiter_mode) == ARBITER_SHADOW_MODE_SHADOW:
        shadow_effectors = tuple(
            [f"{side}_arm" for side in active_sides if bool(args.arms_enabled)]
            + [f"{side}_hand" for side in end_effector_sides if hands_enabled]
        )
        runtime_session_id = f"deploy-shadow-{uuid.uuid4()}"
        shadow_catalog = ActionComponentCatalog(
            catalog_id=("deploy-shadow:" f"{'-'.join(shadow_effectors)}:v1"),
            action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
            components=tuple(
                ActionComponentSpec(
                    component_id=effector,
                    selection_group_id=effector,
                    effector_name=effector,
                )
                for effector in shadow_effectors
            ),
        )
        shadow_action_schema = build_robot_action_v1_runtime_schema(
            component_catalog=shadow_catalog,
            arm_action_specs_by_side={
                side: ArmActionSpec.tool_tcp(side)
                for side in active_sides
                if bool(args.arms_enabled)
            },
            hand_action_specs_by_side={
                side: HandActionSpec.inspire_tac(side)
                for side in end_effector_sides
                if hands_enabled
            },
        )
        policy_arbiter_shadow = PassivePolicyArbiterShadow(
            PolicyArbiterShadowConfig(
                runtime_session_id=runtime_session_id,
                control_domain_id="deploy-control-0",
                embodiment_instance_id="rm65-bimanual-0",
                producer_id=(
                    f"replay:{policy.name}" if debug_replay else f"policy:{policy.name}"
                ),
                active_effectors=shadow_effectors,
                action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
                # Event-limited: ordinary 15/30/40 Hz emissions are evaluated
                # immediately, while >60 Hz producers are coalesced by the
                # shared production decision-rate ceiling.
                decision_hz=60.0,
                policy_node_id="single_policy",
                clock_domain_id="local_monotonic",
                boot_id=runtime_session_id,
                component_catalog=shadow_catalog,
                action_schema=shadow_action_schema,
            )
        )
        print(
            "[ArbiterShadow] passive non-authoritative evaluation enabled "
            "with a 60 Hz event-limited gate; legacy execution remains owner.",
            flush=True,
        )
        if debug_replay:
            dataset_info = policy.metadata.get("dataset_info")
            dataset_info = dataset_info if isinstance(dataset_info, Mapping) else {}
            try:
                replay_proposal_shadow_adapter = DatasetReplayProposalShadowAdapter(
                    DatasetReplayProposalShadowConfig(
                        runtime_session_id=runtime_session_id,
                        control_domain_id="deploy-control-0",
                        embodiment_instance_id="rm65-bimanual-0",
                        producer_id=policy_arbiter_shadow.config.producer_id,
                        producer_epoch=policy_arbiter_shadow.config.producer_epoch,
                        action_schema=shadow_action_schema,
                        component_schema_hash=(
                            policy_arbiter_shadow.component_catalog.semantic_content_hash
                        ),
                        component_coverage=shadow_effectors,
                        source_path=str(policy.metadata.get("source_path") or ""),
                        episode_id=str(
                            policy.metadata.get("episode_id")
                            or episode_id_from_index(int(args.debug_replay_episode))
                        ),
                        action_field=str(args.debug_replay_action_column),
                        source_action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
                        source_hz=float(policy.metadata["action_hz"]),
                        source_start_index=int(args.debug_replay_start_index),
                        dataset_metadata=dataset_info,
                        target_frame_contract_hash=(
                            None
                            if deploy_session_contract.runtime_pose_frame_contract
                            is None
                            else deploy_session_contract.runtime_pose_frame_contract.contract_hash
                        ),
                    )
                )
                print(
                    "[ReplayProposalShadow] typed passive replay adapter enabled; "
                    "DatasetReplayPolicy remains the only cursor/queue owner.",
                    flush=True,
                )
            except (KeyError, TypeError, ValueError) as exc:
                # Legacy execution stays unchanged, but an untyped replay must
                # never be mislabeled as a policy proposal in the new facts.
                print(
                    "[ReplayProposalShadow][WARN] typed replay shadow disabled: "
                    f"{exc}",
                    file=sys.stderr,
                    flush=True,
                )

    backend_control_hz = float(args.arm_control_hz)
    arm_backends = build_arm_backends(
        args,
        active_sides,
        control_hz=backend_control_hz,
        action_hz=float(effective_action_hz),
    )
    if (
        deploy_tcp_pose_envelope_projector is not None
        and args.arm_control_mode == RM65_ARM_CONTROL_MODE_IMPEDANCE
    ):
        for side, backend in arm_backends.items():
            if not isinstance(backend, RM65ImpedanceArmBackend):
                raise TypeError(
                    f"{side} deploy impedance envelope requires "
                    "RM65ImpedanceArmBackend"
                )
            backend.set_final_tcp_pose_projector(
                deploy_tcp_pose_envelope_projector.project_arm_base_tool_tcp
            )
    hand_backends = build_hand_backends(args, end_effector_sides)
    hand_state_poller = (
        HandStateHistoryPoller(
            backends_by_side=hand_backends,
            config=HandStateHistoryPollerConfig(
                poll_hz=max(60.0, 2.0 * float(effective_action_hz)),
            ),
        )
        if hand_backends
        else None
    )
    hand_action_specs = (
        hand_action_specs_for_backends(args, hand_backends, end_effector_sides)
        if hands_enabled
        else {}
    )
    disconnected_gripper_sides = set(
        _parse_disconnected_gripper_sides(
            getattr(args, "disconnected_gripper_sides", ())
        )
    )
    qpos_init_hand_action_specs = policy_qpos_init_hand_action_specs(
        args,
        hand_action_specs,
    )
    pose_observation: RM65PoseObservationAssembly | None = None
    arm_reader: ArmStateReader | None = (
        None
        if bool(args.rm65_pose_observation)
        else _build_arm_state_reader(
            args=args,
            arm_backends=arm_backends,
            pose_observation=None,
        )
    )
    quest_paired_pose_capture: QuestPairedPoseCaptureApplicationPort | None = None
    quest_paired_pose_poll_enabled = False
    quest_paired_pose_capture_error: str | None = None
    quest_paired_pose_factory_attempted = False
    camera_sources = build_camera_sources(
        args,
        selected_camera_keys=selected_camera_keys,
    )
    preview_camera_sources = (
        select_preview_camera_sources(camera_sources, str(args.vis_cam_key))
        if bool(args.vis_cam) and camera_sources
        else {}
    )
    camera_previewer = (
        CameraPreviewer(CameraPreviewConfig())
        if bool(args.vis_cam) and preview_camera_sources
        else None
    )
    visualizer = (
        TactileUVVisualizer(
            TactileVisualizerConfig(
                enabled_sides=end_effector_sides,
                image_size=(int(args.tactile_vis_size), int(args.tactile_vis_size)),
                value_scale=float(args.tactile_vis_value_scale),
            )
        )
        if args.visualize_tactile
        else None
    )
    safety_filter = ActionSafetyFilter(
        ActionSafetyFilterConfig(
            active_sides=active_sides,
            arms_enabled=bool(args.arms_enabled),
            hands_enabled=bool(hands_enabled),
            end_effector_sides=end_effector_sides,
            workspace_min=tuple(float(value) for value in args.arm_workspace_min),
            workspace_max=tuple(float(value) for value in args.arm_workspace_max),
            max_arm_xyz_step_m=float(args.max_arm_xyz_step_m),
            max_arm_rot_step_deg=float(args.max_arm_rot_step_deg),
            max_arm_joint_step_rad=float(args.max_arm_joint_step_rad),
            max_hand_step=float(args.max_hand_step),
            arm_joint_lower_rad=tuple(
                float(value) for value in RM65_ABSOLUTE_Q_MIN_RAD
            ),
            arm_joint_upper_rad=tuple(
                float(value) for value in RM65_ABSOLUTE_Q_MAX_RAD
            ),
        )
    )
    canonical_action_pipeline = CanonicalDeployActionPipeline(
        safety_filter=safety_filter,
        active_sides=active_sides,
        arms_enabled=bool(args.arms_enabled),
        hands_enabled=bool(hands_enabled),
    )
    rtc_action_preparer = _rtc_action_preparer_for_profile(
        rtc_profile_id=str(args.rtc_profile),
        canonical_action_pipeline=canonical_action_pipeline,
        deploy_session_contract=deploy_session_contract,
        safety_filter=safety_filter,
    )
    rtc_frontend: RtcPolicyShadowFrontend | None = None
    rtc_debug_observer: RtcDebugEventObserver | None = None

    def observe_rtc_transport_exchange(bundle: Any) -> None:
        observer = rtc_debug_observer
        if observer is None or not observer.accepts(RTC_DEBUG_LEVEL_SUMMARY):
            return
        observer.offer(
            "rtc_transport_exchange",
            rtc_client_inference_bundle_debug_payload(bundle),
        )

    if str(args.rtc_profile) == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1:
        rtc_session_id = f"deploy-rtc-shadow-{uuid.uuid4()}"
        runtime_frame_hash = getattr(
            deploy_session_contract.runtime_pose_frame_contract,
            "contract_hash",
            None,
        )
        if runtime_frame_hash is None:
            runtime_frame_hash = canonical_content_hash(
                {
                    "policy_absolute_frame": (
                        deploy_session_contract.policy_absolute_frame
                    ),
                    "runtime_pose_frame_contract": None,
                },
                domain="dexteleop.rtc_3c.runtime_frame_binding.v1",
            )
        rtc_frontend = RtcPolicyShadowFrontend(
            config=RtcPolicyShadowConfig(
                runtime_session_id=rtc_session_id,
                control_domain_id="deploy-control-0",
                producer_id=f"policy:{policy.name}:rtc-shadow",
                policy_node_id=policy_deploy_runtime.config.policy_node_id,
                artifact_hash=canonical_content_hash(
                    {
                        "policy_name": policy.name,
                        "policy_metadata": policy.metadata,
                    },
                    domain="dexteleop.rtc_3c.virtual_policy_artifact.v1",
                ),
                action_codec_hash=canonical_content_hash(
                    {
                        "action_space": DEPLOY_ACTION_VECTOR_ACTION_SPACE,
                        "pose_mode": str(args.policy_action_pose_mode),
                    },
                    domain="dexteleop.rtc_3c.action_codec.v1",
                ),
                frame_contract_hash=str(runtime_frame_hash),
                layout_hash=(
                    deploy_layout.layout_hash
                    if deploy_layout is not None
                    else canonical_content_hash(
                        {
                            "active_sides": active_sides,
                            "end_effector_sides": end_effector_sides,
                            "arms_enabled": bool(args.arms_enabled),
                            "hands_enabled": hands_enabled,
                        },
                        domain="dexteleop.rtc_3c.deploy_layout.v1",
                    )
                ),
                normalization_hash=canonical_content_hash(
                    {"normalization": "runtime_absolute_none_v1"},
                    domain="dexteleop.rtc_3c.normalization.v1",
                ),
                request_validity_steps=int(args.rtc_request_validity_steps),
                worker_stop_timeout_sec=float(args.rtc_worker_stop_timeout_sec),
            ),
            deploy_runtime=policy_deploy_runtime,
            policy=policy,
            created_at=time.monotonic(),
            emission_adapter=canonical_action_pipeline.action_adapter,
        )
        if rtc_step_fanout is not None:
            rtc_step_fanout.bind_frontend(rtc_frontend)
    elif str(args.rtc_profile) == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
        if rtc_model_adapter_factory is None:
            raise ValueError(
                "raw-RTC final-chunk requires a model-package adapter factory; "
                "the generic deploy runtime will not infer one from policy names."
            )
        model_adapter = rtc_model_adapter_factory(
            policy=policy,
            policy_artifact=policy_artifact,
            deploy_session_contract=deploy_session_contract,
            deploy_runtime=policy_deploy_runtime,
            args=args,
            active_sides=active_sides,
        )
        rtc_frontend = build_rtc_model_shadow_frontend(
            model_adapter,
            runtime_session_id=f"deploy-raw-rtc-{uuid.uuid4()}",
            control_domain_id="deploy-control-0",
            producer_id=f"policy:{policy.name}:raw-rtc",
            policy_node_id=policy_deploy_runtime.config.policy_node_id,
            deploy_runtime=policy_deploy_runtime,
            policy_name=policy.name,
            preparer=canonical_action_pipeline,
            created_at=time.monotonic(),
            profile=RAW_RTC_FINAL_CHUNK_PROFILE_V1,
            request_validity_steps=int(args.rtc_request_validity_steps),
            worker_stop_timeout_sec=float(args.rtc_worker_stop_timeout_sec),
            observe_bundle=observe_rtc_transport_exchange,
        )
    elif str(args.rtc_profile) == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1:
        if rtc_model_adapter_factory is None:
            raise ValueError(
                "model-RTC flow-inpaint requires a model-package adapter "
                "factory; the generic deploy runtime will not infer one from "
                "policy names."
            )
        model_adapter = rtc_model_adapter_factory(
            policy=policy,
            policy_artifact=policy_artifact,
            deploy_session_contract=deploy_session_contract,
            deploy_runtime=policy_deploy_runtime,
            args=args,
            active_sides=active_sides,
        )
        rtc_frontend = build_rtc_conditioned_model_shadow_frontend_v3(
            model_adapter,
            runtime_session_id=f"deploy-model-rtc-{uuid.uuid4()}",
            control_domain_id="deploy-control-0",
            producer_id=f"policy:{policy.name}:model-rtc",
            policy_node_id=policy_deploy_runtime.config.policy_node_id,
            deploy_runtime=policy_deploy_runtime,
            policy_name=policy.name,
            preparer=rtc_action_preparer,
            created_at=time.monotonic(),
            profile=MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1,
            request_validity_steps=int(args.rtc_request_validity_steps),
            worker_stop_timeout_sec=float(args.rtc_worker_stop_timeout_sec),
            observe_bundle=observe_rtc_transport_exchange,
        )
    if rtc_emission_fanout is not None:
        if rtc_frontend is None:
            raise RuntimeError("RTC emission fanout has no RTC frontend.")
        rtc_emission_fanout.bind_frontend(rtc_frontend)
    if rtc_emission_publisher is not None:
        if rtc_frontend is None:
            raise RuntimeError("RTC emission publisher has no RTC frontend.")
        rtc_emission_publisher.bind_frontend(rtc_frontend)
    ik_action_gate = IKActionChunkGate(
        stable_frames_required=int(args.ik_recovery_stable_frames)
    )
    startup_initializer = StartupInitializer(
        duration_s=float(args.init_time_s),
        source=startup_init_source,
        target_action=startup_init_action,
        active_sides=active_sides,
        arms_enabled=bool(args.arms_enabled),
        hands_enabled=bool(hands_enabled),
        arm_position_tolerance_m=float(args.init_arm_position_tolerance_m),
        arm_rotation_tolerance_deg=float(args.init_arm_rotation_tolerance_deg),
    )
    current_pose_start_override = bool(args.no_init_qpos or args.start_init_qpos)
    current_startup_satisfied_by_override = _reset_startup_initializer(
        startup_initializer,
        current_pose_start_allowed=current_pose_start_override,
    )
    if current_startup_satisfied_by_override:
        current_pose_override_name = (
            "--no-init-qpos" if bool(args.no_init_qpos) else "--start-init_qpos"
        )
        print(
            f"[StartupInit] source=current already satisfied by {current_pose_override_name}; "
            "press A to confirm ready and start policy execution."
        )
    fixed_init_controller = (
        None
        if fixed_init_plan is None
        else DeployFixedInitController(
            fixed_init_plan,
            arm_max_angvel=tuple(
                float(value) for value in args.deploy_fixed_init_arm_max_angvel
            ),
            qpos_tolerance=float(args.deploy_fixed_init_qpos_tolerance),
        )
    )
    policy_qpos_init_controller = (
        None
        if policy_qpos_init_plan is None
        else DeployPolicyQposInitController(
            policy_qpos_init_plan,
            active_sides=active_sides,
            hand_action_specs_by_side=qpos_init_hand_action_specs,
            settings=DeployPolicyQposInitSettings(
                qpos_reached_tolerance_rad=float(args.policy_qpos_init_qpos_tolerance),
                move_to_start_timeout_s=float(args.policy_qpos_init_timeout_s),
                qpos_init_dt_sec=float(args.policy_qpos_init_dt_sec),
                qpos_init_joint_velocity_profile=str(
                    args.policy_qpos_init_joint_velocity_profile
                ),
                qpos_init_max_angvel_rad_s=args.policy_qpos_init_arm_max_angvel,
                qpos_init_max_steps_per_side=int(
                    args.policy_qpos_init_max_steps_per_side
                ),
                qpos_init_collision_check=bool(args.policy_qpos_init_collision_check),
                qpos_init_collision_scene_profile=str(
                    args.policy_qpos_init_collision_scene_profile
                ),
                qpos_init_branch_search=bool(args.policy_qpos_init_branch_search),
                qpos_init_branch_search_max_branches=int(
                    args.policy_qpos_init_branch_search_max_branches
                ),
                qpos_init_branch_neutral_qpos_path=args.policy_qpos_init_branch_neutral_qpos_path,
                initialize_hands=bool(args.policy_qpos_init_hands),
                hand_reached_tolerance=float(args.policy_qpos_init_hand_tolerance),
                dmtac_hand_reached_tolerance=float(
                    args.policy_qpos_init_dmtac_tolerance
                ),
                hand_progress_timeout_sec=float(
                    args.policy_qpos_init_hand_progress_timeout_sec
                ),
                hand_progress_epsilon=float(
                    args.policy_qpos_init_hand_progress_epsilon
                ),
                hand_reconnect_attempts=int(
                    args.policy_qpos_init_hand_reconnect_attempts
                ),
                validate_plan_only=bool(
                    getattr(args, "rtc_real_shadow", False)
                    and not bool(args.send_actions)
                ),
            ),
        )
    )

    policy_active = (
        bool(args.auto_start)
        and fixed_init_controller is None
        and policy_qpos_init_controller is None
    )
    send_actions = bool(args.send_actions)
    wait_for_ready_confirm = (
        bool(send_actions) and fixed_init_controller is None
        if args.wait_for_ready_confirm is None
        else bool(args.wait_for_ready_confirm)
    )
    ready_confirmed = not wait_for_ready_confirm
    last_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
    rtc_active_latency_gate_required = bool(
        rtc_frontend is not None
        and send_actions
        and str(args.rtc_profile)
        in {
            RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
            MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
        }
    )
    rtc_active_latency_gate_passed = not rtc_active_latency_gate_required
    rtc_active_latency_samples: list[float] = []
    accepted_rtc_dispatch_steps = 0
    step_index = 0
    loop_rate = LoopRate()
    last_status_at = 0.0
    last_fault_text: str | None = None
    collision_reset_console_gate = ChangeGate()
    started_at = time.time()
    debug_storage_resolution = (
        resolve_deployment_debug_root(args, now=started_at)
        if bool(args.debug)
        else None
    )
    debug_logger = DeploymentDebugLogger(
        enabled=bool(args.debug),
        root_dir=(
            None if debug_storage_resolution is None else debug_storage_resolution.path
        ),
        active_sides=active_sides,
        payload_mode=str(args.debug_payload_mode),
        flush_every_events=16,
        flush_interval_sec=0.25,
    )
    if debug_storage_resolution is not None:
        debug_logger.log_event(
            "artifact_storage",
            {
                "requested_layout": getattr(args, "artifact_layout", None),
                "selection_source": debug_storage_resolution.source,
                "resolved_layout": debug_storage_resolution.layout_name,
                "resolved_output_root": str(
                    debug_storage_resolution.path.expanduser().resolve()
                ),
            },
        )
    if rtc_frontend is not None:
        rtc_debug_observer = RtcDebugEventObserver(
            level=str(getattr(args, "_rtc_debug_level", RTC_DEBUG_LEVEL_OFF)),
            sink=debug_logger.log_event,
            queue_capacity=int(args.rtc_debug_queue_capacity),
        )
    if rtc_debug_observer is not None and rtc_debug_observer.enabled:
        source_binding = rtc_frontend.source_binding_candidate
        rtc_debug_observer.offer(
            "rtc_source_binding_candidate_prepared",
            {
                "schema": "dexteleop_rtc_source_binding_candidate_debug_v1",
                "source_binding_hash": source_binding.semantic_content_hash,
                "source_descriptor_hash": source_binding.descriptor_hash,
                "source_slot_id": source_binding.source_slot_id,
                "producer_id": source_binding.producer_id,
                "producer_epoch": source_binding.producer_epoch,
                "binding_generation": source_binding.binding_generation,
                "runtime_session_id": source_binding.runtime_session_id,
                "control_domain_id": source_binding.control_domain_id,
            },
        )
        if (
            rtc_emission_publisher is not None
            and rtc_emission_publisher.active_binding is not None
        ):
            rtc_debug_observer.offer(
                "rtc_source_binding_activated",
                {
                    "schema": ("dexteleop_rtc_source_binding_activation_debug_v1"),
                    "source_binding_hash": (
                        rtc_emission_publisher.active_binding.semantic_content_hash
                    ),
                    "source_descriptor_hash": (
                        rtc_emission_publisher.active_binding.descriptor_hash
                    ),
                    "producer_id": (rtc_emission_publisher.active_binding.producer_id),
                    "producer_epoch": (
                        rtc_emission_publisher.active_binding.producer_epoch
                    ),
                    "binding_generation": (
                        rtc_emission_publisher.active_binding.binding_generation
                    ),
                    "activation_owner": "application_publisher",
                },
            )
    position_deploy_executor: PositionDeployExecutor | None = None
    position_worker_binding: RM65LegacyWorkerExecutionBinding | None = None
    position_plan_cut_detector = (
        DeploymentPositionPlanCutDetector()
        if bool(args.position_plan_splice_governor)
        else None
    )
    if (
        bool(args.arms_enabled)
        and normalize_rm65_arm_control_mode(args.arm_control_mode)
        == RM65_ARM_CONTROL_MODE_POSITION
    ):
        position_collision_required = bool(
            args.position_collision_check and active_sides == ("left", "right")
        )
        position_path_lift_backend = str(args.position_path_lift_backend)
        (
            position_reference_hz,
            position_command_hz,
        ) = _resolve_position_reference_and_command_hz(
            args,
            backend_control_hz=float(backend_control_hz),
        )
        if position_path_lift_backend == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND:
            if not math.isclose(
                float(effective_action_hz),
                15.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "the promoted official all-IK governor contract requires "
                    "a 15Hz policy action stream."
                )
            if not math.isclose(
                float(position_command_hz),
                120.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "the promoted official all-IK governor contract requires "
                    "the 120Hz RM65 command stream."
                )
            # The official governor owns the certified qref cut, while the
            # already-existing CANFD backend owns the downstream 120 Hz qcmd
            # stream. Keep those clocks separate so the planner runs at the
            # declared 60 Hz reference rate rather than overrunning at 120 Hz.
        arm_only_pipeline = CanonicalDeployActionPipeline(
            safety_filter=safety_filter,
            active_sides=active_sides,
            arms_enabled=True,
            hands_enabled=False,
        )

        def dispatch_position_reference(command: TeleopCommand) -> Mapping[str, Any]:
            annotate_deploy_arm_command_limits(
                command,
                args=args,
                active_sides=active_sides,
                control_hz=float(backend_control_hz),
                action_hz=float(position_reference_hz),
            )
            return dispatch_position_reference_command(
                command,
                arm_backends=arm_backends,
                required_sides=active_sides,
                audit_mode=str(args.position_reference_dispatch_audit_mode),
            )

        def dispatch_position_discrete(command: TeleopCommand) -> Mapping[str, Any]:
            # Recheck after the arm target was queued and immediately before
            # any scheduled hand packet.  This catches a background stream fault
            # that appeared in the arm->hand worker gap and makes the executor
            # latch/hold instead of tail-sending the hand keyframe.
            require_arm_backends_ready_for_position_dispatch(
                arm_backends,
                require_background_target=True,
                stage="pre_discrete_hand_dispatch",
                required_sides=active_sides,
            )
            return _dispatch_position_discrete_command_checked(
                command,
                hand_backends=hand_backends,
            )

        position_worker_binding = getattr(args, "_rm65_legacy_worker_binding", None)
        if position_worker_binding is not None and not isinstance(
            position_worker_binding,
            RM65LegacyWorkerExecutionBinding,
        ):
            raise TypeError("_rm65_legacy_worker_binding has an unsupported type")
        position_deploy_executor = PositionDeployExecutor(
            config=PositionDeployExecutorConfig(
                sample_hz=float(effective_action_hz),
                reference_hz=float(position_reference_hz),
                command_hz=float(position_command_hz),
                stale_after_sec=max(0.25, 3.0 / float(effective_action_hz)),
                queue_size=max(32, int(args.min_action_queue) * 4),
                debug_events_enabled=bool(debug_logger.enabled),
                collision_required=position_collision_required,
                allow_non_authoritative_collision_reset_in_dry_run=not bool(
                    send_actions
                ),
                chunk_lookahead_enabled=bool(args.position_chunk_lookahead),
                plan_splice_governor_enabled=bool(args.position_plan_splice_governor),
                plan_splice_clock_rebase_enabled=bool(
                    args.position_plan_splice_governor
                ),
                stream_start_acquisition=(
                    PositionStreamStartAcquisitionConfig(
                        enabled=bool(args.position_stream_start_acquisition),
                        max_translation_step_m=float(
                            args.position_stream_start_max_translation_step_m
                        ),
                        max_rotation_step_rad=math.radians(
                            float(args.position_stream_start_max_rotation_step_deg)
                        ),
                        max_source_intervals=int(
                            args.position_stream_start_max_source_intervals
                        ),
                    )
                ),
                max_reference_tick_skip_burst=(
                    1
                    if position_path_lift_backend
                    == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND
                    else None
                ),
                reference_worker_cpu=args.position_reference_worker_cpu,
                cyclic_gc_guard_enabled=bool(args.position_cyclic_gc_guard),
            ),
            canonical_pipeline=arm_only_pipeline,
            dispatch_callback=(dispatch_position_reference if send_actions else None),
            discrete_dispatch_callback=(
                dispatch_position_discrete if send_actions and hands_enabled else None
            ),
            hold_callback=(
                (
                    lambda: hold_deploy_arm_backends(
                        arm_backends,
                        reason="position deploy executor pause/fault",
                    )
                )
                if send_actions
                else None
            ),
            pose_target_projector=deploy_tcp_pose_envelope_projector,
            collision_planner_factory=(
                (
                    (
                        lambda: build_official_all_ik_governor_from_started_backends(
                            arm_backends,
                            source_hz=float(effective_action_hz),
                            command_hz=float(position_reference_hz),
                            lookahead_source_steps=int(
                                args.position_all_ik_lookahead_source_steps
                            ),
                            jerk_ramp_sec=float(args.position_all_ik_jerk_ramp_sec),
                            batched_solver_tolerance=float(
                                args.position_all_ik_batched_solver_tolerance
                            ),
                            continuous_waypoint_shadow_enabled=bool(
                                args.position_continuous_waypoint_shadow
                                or args.position_continuous_waypoint_experimental_admission
                            ),
                            continuous_waypoint_admission_enabled=bool(
                                args.position_continuous_waypoint_experimental_admission
                            ),
                            continuous_waypoint_async_precompute_enabled=bool(
                                args.position_continuous_waypoint_async_precompute
                            ),
                            continuous_waypoint_worker_cpu=(
                                args.position_continuous_waypoint_worker_cpu
                            ),
                            continuous_waypoint_worker_isolation=str(
                                args.position_continuous_waypoint_worker_isolation
                            ),
                            plan_splice_dynamics_config=(
                                resolve_position_plan_splice_dynamics_config(
                                    args.position_plan_splice_dynamics_profile
                                )
                            ),
                            scene_profile=str(args.position_collision_scene_profile),
                            collision_min_distance_m=float(
                                args.position_collision_min_distance_m
                            ),
                            collision_distance_epsilon_m=float(
                                args.position_collision_distance_epsilon_m
                            ),
                            max_joint_velocity_rad_s=tuple(
                                float(value)
                                for value in (
                                    args.arm_max_angvel
                                    if args.position_qref_max_angvel is None
                                    else args.position_qref_max_angvel
                                )
                            ),
                            max_joint_acceleration_rad_s2=tuple(
                                float(value)
                                for value in args.position_qref_max_acceleration_rad_s2_resolved
                            ),
                            debug_profile=bool(args.position_collision_debug_profile),
                        )
                    )
                    if position_path_lift_backend
                    == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND
                    else (
                        lambda: build_position_collision_planner_from_started_backends(
                            arm_backends,
                            reference_hz=float(position_reference_hz),
                            scene_profile=str(args.position_collision_scene_profile),
                            collision_min_distance_m=float(
                                args.position_collision_min_distance_m
                            ),
                            collision_distance_epsilon_m=float(
                                args.position_collision_distance_epsilon_m
                            ),
                            max_joint_velocity_rad_s=tuple(
                                float(value)
                                for value in (
                                    args.arm_max_angvel
                                    if args.position_qref_max_angvel is None
                                    else args.position_qref_max_angvel
                                )
                            ),
                            max_joint_acceleration_rad_s2=tuple(
                                float(value)
                                for value in args.position_max_acceleration_rad_s2
                            ),
                            max_joint_step_rad=(
                                None
                                if float(args.max_arm_joint_step_rad) <= 0.0
                                else float(args.max_arm_joint_step_rad)
                            ),
                            damping=float(args.position_ik_damping),
                            solve_max_iterations=int(
                                args.position_ik_solve_max_iterations
                            ),
                            solve_min_iterations=int(
                                args.position_ik_solve_min_iterations
                            ),
                            solve_adaptive_iterations_enabled=bool(
                                args.position_ik_adaptive_iterations
                            ),
                            solve_iteration_reserve_ms=float(
                                args.position_ik_iteration_reserve_ms
                            ),
                            pose_lift_profile=str(args.pose_lift_profile),
                            debug_profile=bool(args.position_collision_debug_profile),
                            ik_tuning=_position_deploy_ik_tuning(args),
                        )
                    )
                )
                if position_collision_required
                else None
            ),
            **(
                {}
                if position_worker_binding is None
                else position_worker_binding.position_executor_kwargs()
            ),
        )

    def flush_position_executor_debug_events() -> None:
        if position_deploy_executor is None:
            return
        for raw_event in position_deploy_executor.drain_debug_events():
            event = dict(raw_event)
            event_name = str(event.pop("event", "unknown"))
            debug_logger.log_event(f"position_executor_{event_name}", event)

    def latch_canonical_safety_fault(
        canonical_action: Any,
        observation: Any,
        *,
        policy_name: str,
    ) -> None:
        """Turn a blocked 15 Hz canonical action into a real hold boundary."""

        canonical_faults = tuple(
            str(item) for item in getattr(canonical_action, "faults", ())
        )
        if not canonical_faults:
            return
        if position_deploy_executor is not None:
            position_deploy_executor.latch_safety_fault(
                canonical_faults,
                observation=observation,
                stage="15hz_keyframe_safety",
            )
            return
        if send_actions:
            hold_deploy_arm_backends(
                arm_backends,
                reason=(
                    f"15 Hz deploy safety fault ({policy_name}): "
                    + ", ".join(canonical_faults)
                ),
            )

    impedance_force_control_is_deferred = bool(
        send_actions
        and bool(args.arms_enabled)
        and normalize_rm65_arm_control_mode(args.arm_control_mode)
        == RM65_ARM_CONTROL_MODE_IMPEDANCE
    )
    impedance_force_control_enabled = False
    impedance_force_guard_monitor: ImpedanceForceGuardMonitor | None = None
    position_force_guard_monitor: PositionForceGuardMonitor | None = None
    position_force_guard_required = bool(
        send_actions
        and bool(args.arms_enabled)
        and bool(args.position_force_guard)
        and normalize_rm65_arm_control_mode(args.arm_control_mode)
        == RM65_ARM_CONTROL_MODE_POSITION
    )

    if impedance_force_control_is_deferred:
        guard_readers: dict[str, Callable[[], Mapping[str, Any]]] = {}
        for side, backend in arm_backends.items():
            reader = getattr(backend, "read_impedance_force_guard_data", None)
            set_callback = getattr(backend, "set_force_hard_stop_callback", None)
            if not callable(reader) or not callable(set_callback):
                # Missing backend capability is a runtime integration failure.
                raise RuntimeError(  # noqa: TRY004
                    f"{side} impedance backend does not expose the force-guard contract"
                )
            guard_readers[str(side)] = reader

        def impedance_force_guard_stop(
            fault: ImpedanceForceGuardFault,
        ) -> None:
            reason = f"impedance_force_guard:{fault.reason}:" f"{fault.side or 'all'}"
            print(
                "[ImpedanceForceGuard] TRIPPED; requesting concurrent bimanual "
                f"Arm_Stop: {fault.message}",
                flush=True,
            )
            try:
                emergency_stop_deploy_arm_backends(
                    arm_backends,
                    reason=reason,
                )
            except Exception as exc:
                print(
                    "[ImpedanceForceGuard] CRITICAL: bimanual Arm_Stop "
                    f"acknowledgement failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

        impedance_force_guard_monitor = ImpedanceForceGuardMonitor(
            readers_by_side=guard_readers,
            emergency_stop=impedance_force_guard_stop,
            config=ImpedanceForceGuardConfig(),
        )
        for side, backend in arm_backends.items():
            set_callback = backend.set_force_hard_stop_callback
            set_callback(impedance_force_guard_monitor.notify_backend_hard_stop)

    if position_force_guard_required:
        position_guard_readers: dict[str, Callable[[], Mapping[str, Any]]] = {}
        for side, backend in arm_backends.items():
            reader = getattr(backend, "read_force_data", None)
            if not callable(reader):
                # Missing backend capability is a runtime integration failure.
                raise RuntimeError(  # noqa: TRY004
                    f"{side} position backend does not expose read_force_data"
                )
            position_guard_readers[str(side)] = reader

        def position_force_guard_stop(
            fault: PositionForceGuardFault,
        ) -> None:
            reason = f"position_force_guard:{fault.reason}:{fault.side or 'all'}"
            print(
                "[PositionForceGuard] TRIPPED; requesting concurrent bimanual "
                f"Arm_Stop: {fault.message}",
                flush=True,
            )
            try:
                emergency_stop_deploy_arm_backends(
                    arm_backends,
                    reason=reason,
                )
            except Exception as exc:
                print(
                    "[PositionForceGuard] CRITICAL: bimanual Arm_Stop "
                    f"acknowledgement failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

        position_force_guard_monitor = PositionForceGuardMonitor(
            readers_by_side=position_guard_readers,
            emergency_stop=position_force_guard_stop,
            config=position_force_guard_config(args),
        )

    def enable_deferred_impedance_force_control(*, trigger: str) -> None:
        """Start impedance output only at an explicit policy/init start boundary."""

        nonlocal impedance_force_control_enabled
        if not impedance_force_control_is_deferred or impedance_force_control_enabled:
            return
        enable_impedance_force_control_backends(arm_backends, ready_timeout_sec=1.0)
        if impedance_force_guard_monitor is not None:
            guard_snapshot = impedance_force_guard_monitor.start()
            guard_config = guard_snapshot.get("config", {})
            print(
                "[Deploy] impedance force guard armed "
                f"config={guard_config} "
                f"sides={sorted(guard_snapshot.get('last_reading_by_side', {}))}",
                flush=True,
            )
        impedance_force_control_enabled = True
        print(
            "[Deploy] impedance periodic output enabled "
            f"after explicit start trigger={trigger}.",
            flush=True,
        )

    def hold_rtc_if_eligible(*, now: float, reason: str) -> None:
        if (
            rtc_frontend is not None
            and rtc_frontend.state.lifecycle_state == POLICY_ASYNC_PRODUCER_ELIGIBLE
        ):
            transition = rtc_frontend.hold(now=now, reason=reason)
            debug_logger.log_event(
                "rtc_lifecycle_transition",
                {
                    "target": "HOLD",
                    "reason": reason,
                    "producer_transition_hash": (
                        transition.producer_transition.receipt.semantic_content_hash
                    ),
                    "worker_stop_hash": (
                        transition.worker_stop_receipt.semantic_content_hash
                    ),
                },
            )

    def rearm_rtc_if_needed(*, now: float, reason: str) -> bool:
        if rtc_frontend is None:
            return True
        if rtc_frontend.state.lifecycle_state == POLICY_ASYNC_PRODUCER_ELIGIBLE:
            return True
        try:
            transition = rtc_frontend.rearm(now=now, reason=reason)
        except RuntimeError as exc:
            print(f"[RTC-3c] rearm deferred: {exc}", flush=True)
            return False
        debug_logger.log_event(
            "rtc_lifecycle_transition",
            {
                "target": "ELIGIBLE",
                "reason": reason,
                "producer_transition_hash": (transition.receipt.semantic_content_hash),
            },
        )
        return True

    def submit_legacy_owner_state(
        *,
        lifecycle_state: str,
        execution_gate: str,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
        recovery_authorization_id: str | None = None,
    ) -> LegacyControlOwnerStateTransitionOutcome | None:
        """Submit state to the unique lower owner; never mutate a mirror."""

        if legacy_worker_binding is None:
            return None
        outcome = legacy_worker_binding.context.transition_owner_state(
            LegacyControlOwnerStateTransitionRequest(
                target_lifecycle_state=lifecycle_state,
                target_execution_gate=execution_gate,
                reason=reason,
                source_event_ids=source_event_ids,
                recovery_authorization_id=recovery_authorization_id,
            )
        )
        if outcome.changed and outcome.receipt is not None:
            debug_logger.log_event(
                "legacy_control_owner_transition",
                {
                    "transition_id": outcome.receipt.transition_id,
                    "transition_hash": outcome.receipt.semantic_content_hash,
                    "lifecycle_state": outcome.snapshot.lifecycle_state,
                    "execution_gate": outcome.snapshot.execution_gate,
                    "owner_state_revision": outcome.snapshot.owner_state_revision,
                    "reason": reason,
                },
            )
        return outcome

    prepared_action_position_executor = _position_executor_for_prepared_action(
        rtc_profile_id=str(args.rtc_profile),
        send_actions=send_actions,
        position_deploy_executor=position_deploy_executor,
    )
    prepared_action_executor = PreparedActionExecutor(
        position_deploy_executor=prepared_action_position_executor,
        send_actions=send_actions,
        arm_backends=arm_backends,
        hand_backends=hand_backends,
        active_sides=active_sides,
        position_worker_binding=position_worker_binding,
        submit_legacy_owner_state=submit_legacy_owner_state,
    )

    def reset_policy_for_activation() -> None:
        nonlocal rtc_active_latency_gate_passed
        policy.reset()
        rtc_active_latency_samples.clear()
        rtc_active_latency_gate_passed = not rtc_active_latency_gate_required

    def record_policy_prediction_result(
        prediction: PolicyChunkPredictionResult,
        *,
        queue_policy: str,
        faults: list[str],
        timestamp: float,
        step_index: int,
        observation_step_index: int,
    ) -> tuple[bool, bool]:
        """Record caller-admitted prediction state and return defer/pause."""

        queue_len_before = prediction.queue_len_before
        defer_until_fresh_cut = False
        pause_policy = False
        if prediction.error is None:
            if prediction.accepted:
                ik_action_gate.mark_policy_output_accepted()
                defer_until_fresh_cut = (
                    _defer_position_action_after_synchronous_prediction(
                        prediction_accepted=True,
                        position_executor_enabled=(
                            position_deploy_executor is not None
                        ),
                    )
                )
            elif prediction.discarded_by_gate:
                print("[Deploy] IK recovery active; discarded action chunk.")
            else:
                faults.extend(prediction.faults)
            debug_logger.log_policy_chunk(
                timestamp=timestamp,
                step_index=step_index,
                policy_name=policy.name,
                policy_output=prediction.policy_output,
                actions=(
                    list(prediction.actions)
                    if prediction.accepted or prediction.discarded_by_gate
                    else []
                ),
                accepted=bool(prediction.accepted),
                discarded_by_ik=bool(prediction.discarded_by_gate),
                queue_policy=queue_policy,
                queue_len_before=queue_len_before,
                queue_len_after=prediction.queue_len_after,
                faults=faults,
                ik_recovery_statuses=ik_action_gate.last_statuses,
                ik_recovery_diagnostics=ik_action_gate.last_diagnostics,
            )
            if prediction.accepted:
                if defer_until_fresh_cut:
                    debug_logger.log_event(
                        "policy_chunk_execution_deferred",
                        {
                            "schema": (
                                "dexteleop_policy_chunk_execution_deferred_v1"
                            ),
                            "reason": (
                                "fresh_post_inference_observation_cut_required"
                            ),
                            "queue_len": int(prediction.queue_len_after),
                            "observation_step_index": int(observation_step_index),
                        },
                    )
        else:
            faults.extend(prediction.faults)
            if prediction.rollback_error is not None:
                rollback_error = prediction.rollback_error
                print(
                    "[Deploy] policy prediction rollback error: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
            exc = prediction.error
            print(
                "[Deploy] policy error: "
                f"{type(exc).__name__}: {exc}"
            )
            faults.append(f"policy_error:{type(exc).__name__}")
            debug_logger.log_policy_error(
                timestamp=timestamp,
                step_index=step_index,
                policy_name=policy.name,
                exc=exc,
                queue_len_before=queue_len_before,
            )
            pause_policy = bool(args.pause_on_policy_error)
        return defer_until_fresh_cut, pause_policy

    resource_lifecycle = PolicyDeployResourceLifecycle(
        debug_logger=debug_logger,
        rtc_frontend=rtc_frontend,
        rtc_debug_observer=rtc_debug_observer,
        position_deploy_executor=position_deploy_executor,
        position_force_guard_monitor=position_force_guard_monitor,
        impedance_force_guard_monitor=impedance_force_guard_monitor,
        fixed_init_controller=fixed_init_controller,
        hand_state_poller=hand_state_poller,
        arm_backends=arm_backends,
        hand_backends=hand_backends,
        camera_sources=camera_sources,
        camera_previewer=camera_previewer,
        tactile_visualizer=visualizer,
        submit_legacy_owner_state=submit_legacy_owner_state,
        flush_position_executor_debug_events=flush_position_executor_debug_events,
    )
    cleanup_errors: tuple[str, ...] = ()
    keyboard_interrupt_stop_error: str | None = None
    pending_legacy_owner_recovery_authorization_id: str | None = None
    try:
        print(
            "[Deploy] policy="
            f"{policy.name} active_sides={active_sides} "
            f"arms={bool(args.arms_enabled)} hands={hands_enabled} "
            "hand_backends="
            f"{ {side: _hand_backend_mode_for_side(args, side) for side in end_effector_sides} } "
            f"arm_profile={args._real_arm_execution_config.resolved_profile} "
            f"arm_control_mode={args.arm_control_mode} "
            f"arm_canfd_follow={bool(args.arm_canfd_follow)} "
            f"hz={float(effective_action_hz):.2f} "
            f"policy_hz={float(effective_policy_hz):.2f} "
            f"backend_hz={backend_control_hz:.2f} "
            f"queue_policy={action_queue_policy} "
            f"rtc_profile={args.rtc_profile}"
        )
        if disconnected_gripper_sides:
            print(
                "[Deploy] intentionally disconnected DMTAC gripper(s): "
                f"sides={sorted(disconnected_gripper_sides)} "
                f"assumed_state={float(args.disconnected_gripper_state):.3f}; "
                "serial/tactile connection and command transmission are disabled.",
                flush=True,
            )
        if action_hz_info.get("source") != "cli.hz":
            print(
                "[Deploy] action_hz="
                f"{float(effective_action_hz):.3f} source={action_hz_info.get('source')} "
                f"policy_hz_source={policy_hz_source}"
            )
        if position_deploy_executor is not None:
            executor_config = position_deploy_executor.config
            official_governor_active = (
                str(args.position_path_lift_backend)
                == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND
            )
            collision_reset_label = (
                ("fresh_dual_hardware" if send_actions else "dry_run_state_allowed")
                if executor_config.collision_required
                else "disabled"
            )
            tcp_pose_envelope_label = (
                "off"
                if deploy_tcp_pose_envelope_projector is None
                else (
                    "active(pre_ik_canonical_desk)"
                    if args.tcp_pose_envelope_active
                    else "shadow(pre_ik_canonical_desk)"
                )
            )
            print(
                "[PositionDeploy] chain="
                f"safe_{executor_config.sample_hz:g}hz"
                f"->causal_{executor_config.reference_hz:g}hz"
                f"->{executor_config.command_hz:g}hz "
                f"path_lift_backend={args.position_path_lift_backend} "
                f"delay={'0_action_period' if executor_config.chunk_lookahead_enabled else '1_action_period'} "
                f"interpolation={'linear_source_qref' if official_governor_active else 'xyz_linear+shortest_slerp'} "
                f"temporal_smoothing={'third_order_jerk_horizon' if official_governor_active else 'off'} "
                "plan_splice_dynamics="
                f"{args.position_plan_splice_dynamics_profile} "
                "qref_acceleration_profile="
                f"{args.position_qref_acceleration_profile} "
                "qref_qdd_scale="
                f"{float(args.position_qref_acceleration_scale_resolved):g} "
                "qcmd_qdd_envelope=hard_execution_profile "
                "batched_solver_tolerance="
                f"{float(args.position_all_ik_batched_solver_tolerance):g} "
                "continuous_waypoint_shadow="
                f"{'on(candidate_path)' if bool(args.position_continuous_waypoint_shadow) else 'off'} "
                "continuous_waypoint_profile="
                f"{args.position_continuous_waypoint_profile_effective!s} "
                "continuous_waypoint_experimental_admission="
                f"{'on(source_cut_atomic)' if bool(args.position_continuous_waypoint_experimental_admission) else 'off'} "
                "continuous_waypoint_async_precompute="
                f"{'on(bounded_next_cut)' if bool(args.position_continuous_waypoint_async_precompute) else 'off'} "
                "worker_cpu_affinity="
                f"reference:{args.position_reference_worker_cpu},"
                f"c2:{args.position_continuous_waypoint_worker_cpu} "
                "cyclic_gc_guard="
                f"{'on(first_submit_to_close)' if executor_config.cyclic_gc_guard_enabled else 'off'} "
                "c2_worker_isolation="
                f"{args.position_continuous_waypoint_worker_isolation} "
                "client_cpu_reservation="
                f"{'on(process_local)' if bool(args.position_worker_cpu_reservation) else 'off'} "
                "reference_dispatch_audit="
                f"{args.position_reference_dispatch_audit_mode} "
                f"tcp_pose_envelope={tcp_pose_envelope_label} "
                f"collision_reference={'on' if executor_config.collision_required else 'off(dry-run)'} "
                f"collision_scope={'selected_scene_endpoint+sampled_interval' if official_governor_active else 'selected_scene_endpoint_only'} "
                "segment_reference_to_command_certified=false "
                "collision_reset="
                f"{collision_reset_label} "
                f"collision_reset_max_age_s={executor_config.collision_reset_measurement_max_age_sec:g} "
                f"collision_reset_pair_skew_s={executor_config.collision_reset_pair_max_skew_sec:g}"
            )
        if policy.metadata.get("debug_replay"):
            print(
                "[Replay] source="
                f"{policy.metadata.get('source_path')} frames={policy.metadata.get('frame_count')} "
                f"chunk_size={int(args.debug_replay_chunk_size)} loop={bool(args.debug_replay_loop)} "
                f"action_hz={policy.metadata.get('action_hz')}"
            )
        if fixed_init_plan is not None:
            print(
                "[FixedInit] source="
                f"{fixed_init_plan.source_path} "
                f"episode={fixed_init_plan.source_episode_index} "
                f"frames={fixed_init_plan.frame_count} "
                f"range=[{fixed_init_plan.start_index},{fixed_init_plan.end_index}) "
                f"hz={fixed_init_plan.action_hz:.3f} "
                f"after=T_then_startup_init:{startup_init_source}"
            )
        if policy_qpos_init_plan is not None:
            print(
                "[PolicyQposInit] source="
                f"{policy_qpos_init_plan.source_path} "
                f"episode={policy_qpos_init_plan.source_episode_index} "
                f"frame={policy_qpos_init_plan.frame_index} "
                f"qpos_source={policy_qpos_init_plan.metadata.get('qpos_init_source')} "
                f"targets={policy_qpos_init_plan.target_sources_by_side} "
                f"after=startup_init:{startup_init_source}"
            )
        elif (
            (bool(args.start_init_qpos) or bool(args.no_init_qpos))
            and bool(args.send_actions)
            and bool(args.arms_enabled)
        ):
            override_name = (
                "--no-init-qpos" if bool(args.no_init_qpos) else "--start-init_qpos"
            )
            suppressed_suffix = (
                " Profile/CLI dataset-frame qpos-init inputs were suppressed."
                if bool(getattr(args, "policy_qpos_init_suppressed", False))
                else ""
            )
            print(
                f"[Deploy] {override_name} override enabled: "
                "real arm policy execution may start from current qpos without "
                "dataset-frame qpos init." + suppressed_suffix
            )
        if not send_actions:
            print(
                "[Deploy] dry-run: actions are filtered and logged but not sent to hardware."
            )
        if send_actions and hands_enabled:
            missing_hand_sides = [
                side
                for side in end_effector_sides
                if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
                and side not in hand_backends
            ]
            if missing_hand_sides:
                raise RuntimeError(
                    "DMTAC gripper backend config is missing for active side(s): "
                    + ",".join(missing_hand_sides)
                )
        debug_logger.log_startup(
            args=args,
            policy=policy,
            active_sides=active_sides,
            hands_enabled=hands_enabled,
            effective_action_hz=float(effective_action_hz),
            action_hz_info=action_hz_info,
            effective_policy_hz=float(effective_policy_hz),
            policy_hz_source=policy_hz_source,
            backend_control_hz=backend_control_hz,
            action_queue_policy=action_queue_policy,
            send_actions=send_actions,
            arm_backends=arm_backends,
        )

        resource_lifecycle.start(
            args=args,
            disconnected_gripper_sides=disconnected_gripper_sides,
            impedance_force_control_is_deferred=(
                impedance_force_control_is_deferred
            ),
            warm_start_impedance_force_control_backends=(
                warm_start_impedance_force_control_backends
            ),
        )
        if bool(args.rm65_pose_observation):
            receipt_monotonic_ns_clock = time.monotonic_ns
            pose_observation = assemble_rm65_pose_observation(
                arm_backends,
                receipt_clock=build_policy_deploy_receiver_clock(),
                control_domain_id="deploy-control-0",
                bound_at=time.time(),
                receipt_monotonic_ns_clock=receipt_monotonic_ns_clock,
            )
            arm_reader = _build_arm_state_reader(
                args=args,
                arm_backends=arm_backends,
                pose_observation=pose_observation,
            )
            print(
                "[RM65PoseObservation] READY: passive robot pose journal "
                f"bound for sides={sorted(pose_observation.projection_by_side)} "
                f"receipt_clock={pose_observation.receipt_clock.clock_domain_id}; "
                "Quest subscription, calibration, and trajectory rendering remain "
                "disconnected.",
                flush=True,
            )
        if arm_reader is None:
            raise RuntimeError("ArmStateReader was not constructed after startup.")
        if quest_paired_pose_capture_factory is not None:
            if pose_observation is None:
                raise RuntimeError("RM65 pose observation assembly is unavailable.")
            robot_pose_journal = bind_passive_robot_pose_journal_source(
                arm_reader=arm_reader,
                pose_observation=pose_observation,
            )
            quest_paired_pose_factory_attempted = True
            created_quest_capture = quest_paired_pose_capture_factory(
                robot_pose_journal=robot_pose_journal,
            )
            if (
                quest_paired_pose_capture_factory.application
                is not created_quest_capture
            ):
                raise RuntimeError(
                    "Quest paired-pose factory did not retain its returned application"
                )
            quest_paired_pose_capture = require_quest_paired_pose_capture_application(
                created_quest_capture,
                robot_pose_journal=robot_pose_journal,
            )
            quest_paired_pose_poll_enabled = True
            print(
                "[QuestPairedPoseCapture] READY: unique Quest host is bound "
                "to the existing passive robot journal; controller/side/frame "
                "selection remains explicit.",
                flush=True,
            )
        print_controls(
            send_actions=send_actions,
            auto_start=bool(args.auto_start),
            wait_for_ready_confirm=wait_for_ready_confirm,
            fixed_init_enabled=fixed_init_controller is not None,
            policy_qpos_init_enabled=policy_qpos_init_controller is not None,
            policy_qpos_init_plan_only=bool(
                policy_qpos_init_controller is not None
                and policy_qpos_init_controller.settings.validate_plan_only
            ),
        )
        submit_legacy_owner_state(
            lifecycle_state="shadow_ready",
            execution_gate="hold",
            reason="policy_deploy_fake_binding_started_default_hold",
        )
        if policy_qpos_init_controller is not None and bool(args.auto_start):
            enable_deferred_impedance_force_control(
                trigger="auto_start_policy_qpos_init"
            )
            try:
                message = _begin_policy_qpos_init_with_force_guard(
                    policy_qpos_init_controller,
                    monitor=position_force_guard_monitor,
                    arm_backends=arm_backends,
                    now=time.time(),
                )
            except PositionForceGuardTripped as exc:
                policy_active = False
                message = (
                    "[PolicyQposInit] automatic planning completed without "
                    "motion, but fresh post-plan force samples were not "
                    "admitted; motion remains blocked: " + exc.fault.message
                )
            if message:
                print(message)
        elif bool(args.auto_start) and fixed_init_controller is None:
            enable_deferred_impedance_force_control(trigger="auto_start")

        position_force_guard_fault_handled = False
        tcp_pose_envelope_bind_pending = (
            deploy_tcp_pose_envelope_projector is not None
        )
        tcp_pose_envelope_execution_epoch = 0
        tcp_pose_envelope_last_bind_error: str | None = None
        tcp_pose_envelope_fault_latched = False
        with NonBlockingKeyboard() as keyboard:
            while True:
                loop_started_at = time.time()
                control_loop_started_at = time.monotonic()
                pre_action_executor_started_at = control_loop_started_at
                position_safety_fault_latched = False
                position_force_guard_fault = (
                    None
                    if position_force_guard_monitor is None
                    else position_force_guard_monitor.fault
                )
                position_force_guard_fault_latched = (
                    position_force_guard_fault is not None
                )
                impedance_force_guard_fault = (
                    None
                    if impedance_force_guard_monitor is None
                    else impedance_force_guard_monitor.fault
                )
                impedance_force_guard_fault_latched = (
                    impedance_force_guard_fault is not None
                )
                if position_deploy_executor is not None:
                    flush_position_executor_debug_events()
                    executor_fault = position_deploy_executor.snapshot()
                    if bool(executor_fault.get("fault_latched")):
                        if str(executor_fault.get("fault_stage") or "") in {
                            "safety_gate",
                            "15hz_keyframe_safety",
                            "reference_anchor_safety",
                        }:
                            position_safety_fault_latched = True
                        else:
                            position_deploy_executor.raise_if_fault()
                pre_action_executor_completed_at = time.monotonic()
                legacy_owner_desired_state = (
                    ("fault", "fault", "legacy_safety_fault_latched")
                    if (
                        position_safety_fault_latched
                        or position_force_guard_fault_latched
                        or impedance_force_guard_fault_latched
                    )
                    else (
                        "hold",
                        "hold",
                        (
                            "legacy_output_eligibility_pending"
                            if policy_active
                            else "legacy_policy_inactive"
                        ),
                    )
                )
                if position_force_guard_fault_latched:
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                    hold_rtc_if_eligible(
                        now=control_loop_started_at,
                        reason="position_force_guard_latched",
                    )
                    if not position_force_guard_fault_handled:
                        if position_deploy_executor is not None:
                            # Arm_Stop is already latched by the monitor. Clear
                            # queued reference work without re-enabling motion;
                            # RM65 hold_motion is a no-op under that latch.
                            position_deploy_executor.clear()
                        position_force_guard_fault_handled = True
                if impedance_force_guard_fault_latched:
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                    hold_rtc_if_eligible(
                        now=control_loop_started_at,
                        reason="impedance_force_guard_latched",
                    )
                key = keyboard.poll()
                preview_keys: list[str] = []
                if camera_previewer is not None:
                    preview_keys = camera_previewer.update(preview_camera_sources)
                if preview_keys and key is None:
                    key = preview_keys[0]
                key = None if key is None else key.lower()
                rearm_requested = bool(key == "r")
                if rearm_requested and position_force_guard_fault_latched:
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                    print(
                        "[Deploy] r ignored: position force emergency stop is "
                        "latched; restart and perform a fresh qpos init."
                    )
                elif (
                    rearm_requested
                    and not position_safety_fault_latched
                    and not impedance_force_guard_fault_latched
                ):
                    print(
                        "[Deploy] r ignored: no manually re-armable safety fault is latched."
                    )
                elif rearm_requested:
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                elif key == "g":
                    if (
                        position_safety_fault_latched
                        or position_force_guard_fault_latched
                        or impedance_force_guard_fault_latched
                    ):
                        policy_active = False
                        print(
                            "[Deploy] g ignored: a safety fault is latched; position-force "
                            "trips require restart, other safety faults require manual re-arm."
                        )
                        continue
                    if position_deploy_executor is not None:
                        position_deploy_executor.clear()
                        last_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
                    if not rearm_rtc_if_needed(
                        now=control_loop_started_at,
                        reason="operator_g",
                    ):
                        policy_active = False
                        continue
                    reset_policy_for_activation()
                    policy_deploy_runtime.clear_action_queue()
                    _reset_startup_initializer(
                        startup_initializer,
                        current_pose_start_allowed=current_pose_start_override,
                    )
                    policy_deploy_runtime.reset_prediction_schedule()
                    ready_confirmed = not wait_for_ready_confirm
                    if (
                        fixed_init_controller is not None
                        and fixed_init_controller.blocks_policy
                    ):
                        print(
                            "[Deploy] ignore g: complete fixed-init H -> H before starting policy."
                        )
                    elif (
                        policy_qpos_init_controller is not None
                        and policy_qpos_init_controller.blocks_policy
                    ):
                        policy_active = False
                        enable_deferred_impedance_force_control(
                            trigger="g_policy_qpos_init"
                        )
                        try:
                            message = _begin_policy_qpos_init_with_force_guard(
                                policy_qpos_init_controller,
                                monitor=position_force_guard_monitor,
                                arm_backends=arm_backends,
                                now=loop_started_at,
                            )
                        except PositionForceGuardTripped as exc:
                            policy_active = False
                            message = (
                                "[PolicyQposInit] planning completed without "
                                "motion, but fresh post-plan force samples were "
                                "not admitted; motion remains blocked: "
                                + exc.fault.message
                            )
                        if message:
                            print(message)
                    else:
                        enable_deferred_impedance_force_control(trigger="g_policy")
                        policy_active = True
                        print("[Deploy] policy active")
                elif key == "a":
                    if position_force_guard_fault_latched:
                        policy_active = False
                        print(
                            "[Deploy] ready confirm ignored: position force emergency "
                            "stop is latched; restart and perform a fresh qpos init."
                        )
                    elif impedance_force_guard_fault_latched:
                        policy_active = False
                        print(
                            "[Deploy] ready confirm ignored: impedance force guard is "
                            "latched; complete manual re-arm with r."
                        )
                    elif (
                        fixed_init_controller is not None
                        and fixed_init_controller.blocks_policy
                    ):
                        print(
                            "[Deploy] ready confirm ignored: complete fixed-init H -> H first."
                        )
                    elif (
                        policy_qpos_init_controller is not None
                        and policy_qpos_init_controller.blocks_policy
                    ):
                        print(
                            "[Deploy] ready confirm ignored: complete policy qpos init first (press g)."
                        )
                    else:
                        if not rearm_rtc_if_needed(
                            now=control_loop_started_at,
                            reason="operator_ready_confirm",
                        ):
                            policy_active = False
                            continue
                        if not policy_active:
                            policy_active = True
                        enable_deferred_impedance_force_control(
                            trigger="A_policy_or_ready_confirm"
                        )
                        if (
                            startup_initializer.enabled
                            and not startup_initializer.completed
                        ):
                            print(
                                "[Deploy] ready confirm ignored: startup init is not complete yet."
                            )
                        else:
                            ready_confirmed = True
                            policy_deploy_runtime.clear_action_queue()
                            policy_deploy_runtime.reset_prediction_schedule()
                            console_log_event(
                                "Policy",
                                "READY: policy inference/execution may start.",
                                now=loop_started_at,
                            )
                elif key == "h":
                    hold_rtc_if_eligible(
                        now=control_loop_started_at,
                        reason="operator_hold",
                    )
                    if position_deploy_executor is not None:
                        position_deploy_executor.clear()
                        last_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
                    policy_deploy_runtime.clear_action_queue()
                    ready_confirmed = not wait_for_ready_confirm
                    if policy_active:
                        policy_active = False
                        print("[Deploy] policy paused; queue cleared")
                    elif fixed_init_controller is not None:
                        enable_deferred_impedance_force_control(trigger="H_fixed_init")
                        message = fixed_init_controller.handle_h(
                            arm_backends=arm_backends,
                            now=loop_started_at,
                        )
                        if message:
                            print(message)
                    elif (
                        policy_qpos_init_controller is not None
                        and policy_qpos_init_controller.running
                    ):
                        policy_active = False
                        print("[PolicyQposInit] ignore h while qpos init is running.")
                    else:
                        policy_active = False
                        print("[Deploy] policy paused")
                elif (
                    key == "x"
                    and fixed_init_controller is not None
                    and fixed_init_controller.running
                ):
                    print("[FixedInit] ignore x while fixed trajectory is running.")
                elif (
                    key == "x"
                    and policy_qpos_init_controller is not None
                    and policy_qpos_init_controller.running
                ):
                    print("[PolicyQposInit] ignore x while qpos init is running.")
                elif key == "q":
                    if position_deploy_executor is not None:
                        position_deploy_executor.clear()
                    print("\n[Deploy] quitting policy deployment loop\n")
                    break

                pre_action_operator_completed_at = time.monotonic()
                arm_snapshot_cpu_started = time.thread_time()
                arm_states = arm_reader.snapshot(loop_started_at)
                arm_snapshot_cpu_completed = time.thread_time()
                pre_action_arm_snapshot_completed_at = time.monotonic()
                if (
                    quest_paired_pose_capture is not None
                    and quest_paired_pose_poll_enabled
                ):
                    try:
                        quest_paired_pose_capture.poll()
                    except Exception as exc:  # noqa: BLE001 - passive observer only.
                        quest_paired_pose_capture_error = (
                            "quest_paired_pose_capture_poll_failed:"
                            f"{type(exc).__name__}:{exc}"
                        )
                        print(
                            "[QuestPairedPoseCapture] ERROR: passive capture "
                            "disabled after poll failure; policy control is unchanged: "
                            f"{quest_paired_pose_capture_error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        quest_paired_pose_poll_enabled = False
                ik_gate_event = ik_action_gate.update(arm_states)
                if ik_action_gate.blocks_action_output:
                    policy_deploy_runtime.clear_action_queue()
                    hold_rtc_if_eligible(
                        now=control_loop_started_at,
                        reason="ik_action_gate_blocked",
                    )
                if ik_gate_event:
                    policy_deploy_runtime.reset_prediction_schedule()
                    print(f"[Deploy] {ik_gate_event}; action queue cleared")
                hand_states = hand_state_snapshot(
                    hand_backends,
                    history_poller=hand_state_poller,
                    timestamp=loop_started_at,
                )
                pre_action_hand_snapshot_completed_at = time.monotonic()
                control_hz = loop_rate.tick(loop_started_at)
                observation = build_deployment_observation(
                    active_sides=active_sides,
                    arms_enabled=bool(args.arms_enabled),
                    hands_enabled=hands_enabled,
                    end_effector_sides=end_effector_sides,
                    step_index=step_index,
                    arm_states=arm_states,
                    hand_states=hand_states,
                    camera_sources=camera_sources,
                    last_action=last_action,
                    timestamp=loop_started_at,
                    require_images=bool(args.require_camera_frame),
                    camera_frame_max_age_sec=float(args.camera_frame_max_age_sec),
                    hand_state_max_age_sec=DEFAULT_HAND_STATE_MAX_AGE_SEC,
                    metadata={
                        "control_hz": control_hz,
                        "causal_snapshot_contract": (
                            "source_timestamp<=T0_and_available_timestamp<=T0"
                        ),
                        "causal_snapshot_cutoff_timestamp": float(loop_started_at),
                        "policy_active": policy_active,
                        "send_actions": send_actions,
                        "policy_absolute_frame": deploy_session_contract.policy_absolute_frame,
                        "runtime_pose_frame_contract_hash": (
                            None
                            if deploy_session_contract.runtime_pose_frame_contract
                            is None
                            else deploy_session_contract.runtime_pose_frame_contract.contract_hash
                        ),
                        "hand_action_specs_by_side": hand_action_specs,
                        "hand_active_mask_by_side": {
                            side: list(spec.get("active_mask", ()))
                            for side, spec in hand_action_specs.items()
                        },
                    },
                )
                pre_action_observation_completed_at = time.monotonic()
                faults = list(observation.faults)
                if tcp_pose_envelope_fault_latched:
                    faults.append("tcp_pose_envelope_restart_required")
                envelope_init_blocks = bool(
                    (
                        fixed_init_controller is not None
                        and fixed_init_controller.blocks_policy
                    )
                    or (
                        policy_qpos_init_controller is not None
                        and policy_qpos_init_controller.blocks_policy
                    )
                )
                if (
                    tcp_pose_envelope_bind_pending
                    and deploy_tcp_pose_envelope_projector is not None
                    and not envelope_init_blocks
                    and not observation.faults
                ):
                    next_envelope_epoch = tcp_pose_envelope_execution_epoch + 1
                    try:
                        anchor = deploy_tcp_pose_envelope_projector.bind_from_observation(
                            observation,
                            execution_epoch=next_envelope_epoch,
                            captured_at=control_loop_started_at,
                        )
                    except Exception as exc:  # noqa: BLE001 - Active must fail closed.
                        bind_error = f"{type(exc).__name__}: {exc}"
                        if bind_error != tcp_pose_envelope_last_bind_error:
                            console_log_event(
                                "TCPPoseEnvelope",
                                (
                                    "ERROR" if args.tcp_pose_envelope_active else "WAIT"
                                )
                                + ": fresh tool_tcp start-Pose binding failed: "
                                + bind_error,
                                now=loop_started_at,
                            )
                            tcp_pose_envelope_last_bind_error = bind_error
                        if bool(args.tcp_pose_envelope_active):
                            policy_active = False
                            policy_deploy_runtime.clear_action_queue()
                            tcp_pose_envelope_bind_pending = False
                            tcp_pose_envelope_fault_latched = True
                            faults.append("tcp_pose_envelope_restart_required")
                            if position_deploy_executor is not None:
                                position_deploy_executor.latch_safety_fault(
                                    ["tcp_pose_envelope_binding_failed", bind_error],
                                    observation=observation,
                                    stage="target_pose_envelope",
                                    manual_rearm_allowed=False,
                                )
                                position_safety_fault_latched = True
                            else:
                                hold_deploy_arm_backends(
                                    arm_backends,
                                    reason=(
                                        "active TCP pose envelope binding failed: "
                                        + bind_error
                                    ),
                                )
                    else:
                        tcp_pose_envelope_execution_epoch = next_envelope_epoch
                        tcp_pose_envelope_bind_pending = False
                        tcp_pose_envelope_last_bind_error = None
                        tcp_pose_envelope_fault_latched = False
                        console_log_event(
                            "TCPPoseEnvelope",
                            "READY: fresh tool_tcp start Pose bound; mode="
                            + (
                                "active"
                                if args.tcp_pose_envelope_active
                                else "shadow"
                            )
                            + f" epoch={next_envelope_epoch} anchor="
                            + (
                                "none:absolute_only"
                                if anchor is None
                                else anchor.semantic_content_hash
                            ),
                            now=loop_started_at,
                        )
                if position_force_guard_fault_latched:
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                    guard_reason = (
                        "unknown"
                        if position_force_guard_fault is None
                        else position_force_guard_fault.reason
                    )
                    faults.append(
                        "position_force_guard_restart_required:" + guard_reason
                    )
                if impedance_force_guard_fault_latched:
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                    hold_rtc_if_eligible(
                        now=control_loop_started_at,
                        reason="impedance_force_guard_latched",
                    )
                    guard_reason = (
                        "unknown"
                        if impedance_force_guard_fault is None
                        else impedance_force_guard_fault.reason
                    )
                    faults.append(
                        "impedance_force_guard_manual_rearm_required:" + guard_reason
                    )
                    if rearm_requested and impedance_force_guard_monitor is not None:
                        try:
                            for side, backend in arm_backends.items():
                                state = arm_states.get(side)
                                if not isinstance(state, Mapping):
                                    # This is missing live state, not caller misuse.
                                    raise RuntimeError(  # noqa: TRY004
                                        f"{side} has no current hardware state for manual re-arm"
                                    )
                                if not bool(state.get("hardware_read_success", False)):
                                    raise RuntimeError(
                                        f"{side} hardware feedback is not fresh/valid"
                                    )
                                qpos = state.get("hardware_qpos")
                                if qpos is None:
                                    qpos = state.get("current_qpos")
                                if qpos is None:
                                    raise RuntimeError(
                                        f"{side} has no qpos for manual impedance re-arm"
                                    )
                                measured_at = state.get(
                                    "hardware_measurement_timestamp"
                                )
                                if measured_at is not None:
                                    measured_age = loop_started_at - float(measured_at)
                                    if measured_age > 0.5:
                                        raise RuntimeError(
                                            f"{side} hardware qpos is stale ({measured_age:.3f}s)"
                                        )
                                rearm = getattr(
                                    backend,
                                    "manual_rearm_after_emergency_stop",
                                    None,
                                )
                                if not callable(rearm):
                                    # Missing backend capability is a runtime failure.
                                    raise RuntimeError(  # noqa: TRY004
                                        f"{side} backend does not expose manual impedance re-arm"
                                    )
                                rearm(qpos)
                            impedance_force_guard_monitor.rearm_after_manual_home(
                                start=False
                            )
                            impedance_force_control_enabled = False
                        except Exception as exc:  # noqa: BLE001
                            # Re-arm failures must preserve the safety latch.
                            try:
                                emergency_stop_deploy_arm_backends(
                                    arm_backends,
                                    reason="impedance_force_guard_rearm_failed",
                                )
                            except Exception as stop_exc:  # noqa: BLE001
                                print(
                                    "[Deploy] impedance re-arm fallback stop failed: "
                                    f"{type(stop_exc).__name__}: {stop_exc}",
                                    file=sys.stderr,
                                )
                            print(
                                "[Deploy] impedance force guard re-arm rejected; "
                                f"fresh manual Home is required: {type(exc).__name__}: {exc}"
                            )
                        else:
                            impedance_force_guard_fault = None
                            impedance_force_guard_fault_latched = False
                            pending_legacy_owner_recovery_authorization_id = (
                                "legacy-owner-recovery:"
                                + canonical_content_hash(
                                    {
                                        "recovery_kind": (
                                            "impedance_force_guard_manual_rearm"
                                        ),
                                        "runtime_session_id": (
                                            None
                                            if legacy_worker_binding is None
                                            else legacy_worker_binding.context.runtime_session_id
                                        ),
                                        "control_domain_id": (
                                            None
                                            if legacy_worker_binding is None
                                            else legacy_worker_binding.context.control_domain_id
                                        ),
                                        "rearmed_sides": tuple(sorted(arm_backends)),
                                        "observed_at": control_loop_started_at,
                                    },
                                    domain=(
                                        "dexteleop.legacy_owner_recovery_authorization.v1"
                                    ),
                                )
                            )
                            reset_policy_for_activation()
                            _reset_startup_initializer(
                                startup_initializer,
                                current_pose_start_allowed=current_pose_start_override,
                            )
                            policy_deploy_runtime.clear_action_queue()
                            policy_deploy_runtime.reset_prediction_schedule()
                            ready_confirmed = not wait_for_ready_confirm
                            faults.append(
                                "impedance_force_guard_rearmed_press_g_to_resume"
                            )
                            print(
                                "[Deploy] impedance force guard re-armed after fresh "
                                "hardware qpos; press g to resume."
                            )
                if position_safety_fault_latched:
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                    hold_rtc_if_eligible(
                        now=control_loop_started_at,
                        reason="position_safety_fault_latched",
                    )
                    faults.append("position_safety_fault_manual_rearm_required")
                    if rearm_requested and position_deploy_executor is not None:
                        try:
                            rearm_details = position_deploy_executor.manual_rearm(
                                observation
                            )
                        except Exception as exc:  # noqa: BLE001
                            # Convert any re-arm failure into an operator-facing hold.
                            print(
                                "[Deploy] manual re-arm rejected; re-home is required if feedback is "
                                f"unavailable/out of domain: {type(exc).__name__}: {exc}"
                            )
                        else:
                            pending_legacy_owner_recovery_authorization_id = (
                                "legacy-owner-recovery:"
                                + canonical_content_hash(
                                    {
                                        "recovery_kind": (
                                            "position_safety_manual_rearm"
                                        ),
                                        "runtime_session_id": (
                                            None
                                            if legacy_worker_binding is None
                                            else legacy_worker_binding.context.runtime_session_id
                                        ),
                                        "control_domain_id": (
                                            None
                                            if legacy_worker_binding is None
                                            else legacy_worker_binding.context.control_domain_id
                                        ),
                                        "rearm_details": to_serializable(
                                            rearm_details
                                        ),
                                        "observed_at": control_loop_started_at,
                                    },
                                    domain=(
                                        "dexteleop.legacy_owner_recovery_authorization.v1"
                                    ),
                                )
                            )
                            last_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
                            reset_policy_for_activation()
                            _reset_startup_initializer(
                                startup_initializer,
                                current_pose_start_allowed=current_pose_start_override,
                            )
                            policy_deploy_runtime.reset_prediction_schedule()
                            ready_confirmed = not wait_for_ready_confirm
                            faults.append("position_safety_rearmed_press_g_to_resume")
                            print(
                                "[Deploy] safety fault re-armed; press g to begin a fresh policy stream."
                            )
                if loop_started_at - started_at < float(args.startup_warmup_sec):
                    faults.append("startup_warmup")

                if visualizer is not None:
                    visualizer.update(
                        tactile_by_side_from_hand_states(active_sides, hand_states)
                    )

                live_position_force_guard_fault = (
                    None
                    if position_force_guard_monitor is None
                    else position_force_guard_monitor.fault
                )
                if live_position_force_guard_fault is not None:
                    position_force_guard_fault = live_position_force_guard_fault
                    position_force_guard_fault_latched = True
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                    force_fault_label = (
                        "position_force_guard_restart_required:"
                        + live_position_force_guard_fault.reason
                    )
                    if force_fault_label not in faults:
                        faults.append(force_fault_label)

                fixed_completed_this_step = False
                if (
                    fixed_init_controller is not None
                    and not position_force_guard_fault_latched
                ):
                    fixed_result = fixed_init_controller.step(
                        now=loop_started_at,
                        send_actions=send_actions,
                        arm_backends=arm_backends,
                        hand_backends=hand_backends,
                    )
                    if fixed_result.last_action is not None:
                        last_action = fixed_result.last_action.astype(
                            np.float32, copy=True
                        )
                    if fixed_result.message:
                        print(fixed_result.message)
                    if fixed_result.completed:
                        reset_policy_for_activation()
                        policy_deploy_runtime.clear_action_queue()
                        _reset_startup_initializer(
                            startup_initializer,
                            current_pose_start_allowed=current_pose_start_override,
                        )
                        policy_deploy_runtime.reset_prediction_schedule()
                        ready_confirmed = not wait_for_ready_confirm
                        if bool(args.auto_start):
                            policy_active = True
                            print(
                                "[Deploy] fixed-init complete; policy auto-start released."
                            )
                        else:
                            policy_active = False
                            print(
                                "[Deploy] fixed-init complete; press g to run startup init/policy."
                            )
                        if deploy_tcp_pose_envelope_projector is not None:
                            tcp_pose_envelope_bind_pending = True
                        fixed_completed_this_step = True

                policy_qpos_completed_this_step = False
                if (
                    policy_qpos_init_controller is not None
                    and not position_force_guard_fault_latched
                ):
                    policy_qpos_result = policy_qpos_init_controller.step(
                        now=loop_started_at,
                        send_actions=send_actions,
                        arm_backends=arm_backends,
                        hand_backends=hand_backends,
                    )
                    if policy_qpos_result.last_action is not None:
                        last_action = policy_qpos_result.last_action.astype(
                            np.float32,
                            copy=True,
                        )
                    if policy_qpos_result.message and not policy_qpos_result.completed:
                        print(policy_qpos_result.message)
                    if policy_qpos_result.completed:
                        reset_policy_for_activation()
                        policy_deploy_runtime.clear_action_queue()
                        if not policy_qpos_result.requires_fresh_observation:
                            raise RuntimeError(
                                "policy qpos init completion is missing its "
                                "fresh-observation reference cut"
                            )
                        (
                            last_action,
                            current_startup_satisfied,
                        ) = _finish_policy_qpos_init_reference_cut(startup_initializer)
                        policy_deploy_runtime.reset_prediction_schedule()
                        ready_confirmed = not wait_for_ready_confirm
                        policy_active = True
                        if deploy_tcp_pose_envelope_projector is not None:
                            # This loop's observation precedes the final qpos-init
                            # command/hold. Bind only from the next fresh causal
                            # observation, matching the existing reference cut.
                            tcp_pose_envelope_bind_pending = True
                        policy_qpos_completed_this_step = True
                        _announce_policy_qpos_init_complete(
                            wait_for_ready_confirm=wait_for_ready_confirm,
                            plan_only=bool(policy_qpos_result.plan_only),
                            now=loop_started_at,
                        )
                        last_status_at = loop_started_at
                        if current_startup_satisfied:
                            console_log_event(
                                "Init",
                                "READY: init-source=current already satisfied; "
                                "no redundant Cartesian startup command will run.",
                                now=loop_started_at,
                            )

                live_impedance_force_guard_fault = (
                    None
                    if impedance_force_guard_monitor is None
                    else impedance_force_guard_monitor.fault
                )
                if live_impedance_force_guard_fault is not None:
                    impedance_force_guard_fault = live_impedance_force_guard_fault
                    impedance_force_guard_fault_latched = True
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                if impedance_force_guard_fault_latched:
                    policy_active = False
                if position_force_guard_fault_latched:
                    policy_active = False
                if policy_active:
                    position_action_submitted_this_loop = False
                    defer_position_action_until_fresh_cut = False
                    can_run_policy = not (
                        bool(args.require_ready_observation) and faults
                    )
                    if (
                        bool(args.tcp_pose_envelope_active)
                        and tcp_pose_envelope_bind_pending
                    ):
                        can_run_policy = False
                    if _debug_replay_execution_exhausted(
                        policy,
                        queued_actions=policy_deploy_runtime.action_queue_length(),
                    ):
                        can_run_policy = False
                    if fixed_completed_this_step:
                        can_run_policy = False
                    if policy_qpos_completed_this_step:
                        can_run_policy = False
                    if (
                        fixed_init_controller is not None
                        and fixed_init_controller.blocks_policy
                    ):
                        can_run_policy = False
                    if (
                        policy_qpos_init_controller is not None
                        and policy_qpos_init_controller.blocks_policy
                    ):
                        can_run_policy = False
                    if (
                        startup_initializer.enabled
                        and not startup_initializer.completed
                    ):
                        if can_run_policy:
                            init_action, init_faults, init_ready = (
                                startup_initializer.next_action(
                                    observation,
                                    now=loop_started_at,
                                )
                            )
                            faults.extend(init_faults)
                            if init_action is not None:
                                canonical_action = canonical_action_pipeline.prepare(
                                    init_action,
                                    observation,
                                    policy_name="startup_init",
                                )
                                faults.extend(canonical_action.faults)
                                latch_canonical_safety_fault(
                                    canonical_action,
                                    observation,
                                    policy_name="startup_init",
                                )
                                if (
                                    startup_initializer.progress >= 1.0
                                    and not canonical_action.faults
                                ):
                                    startup_initializer.update_resolved_target_action(
                                        canonical_action.action.vector
                                    )
                                if (
                                    canonical_action.command is not None
                                    and canonical_action.command.commands
                                ):
                                    execution = prepared_action_executor.execute(
                                        canonical_action,
                                        observation,
                                        policy_name="startup_init",
                                        source_decision_id=(
                                            f"startup-init-decision:{step_index}"
                                        ),
                                        source_reference_segment_id=(
                                            f"startup-init-reference:{step_index}"
                                        ),
                                    )
                                    if execution.accepted:
                                        position_action_submitted_this_loop = (
                                            prepared_action_position_executor
                                            is not None
                                        )
                                        last_action = (
                                            canonical_action.action.vector.astype(
                                                np.float32, copy=True
                                            )
                                        )
                                if init_ready and not canonical_action.faults:
                                    startup_initializer.mark_complete()
                                    if startup_initializer.source in {
                                        "replay:first_frame",
                                        "policy:first_frame",
                                    }:
                                        skipped = _skip_startup_replay_frames(
                                            policy, count=1
                                        )
                                        if skipped:
                                            print(
                                                "[StartupInit] replay startup target consumed; "
                                                f"skipped {skipped} replay frame(s)."
                                            )
                                    policy_deploy_runtime.clear_action_queue()
                                    policy_deploy_runtime.reset_prediction_schedule()
                                    if wait_for_ready_confirm and not ready_confirmed:
                                        print(
                                            "[StartupInit] ready; press A to start policy inference/execution."
                                        )
                        can_run_policy = False
                    if (
                        can_run_policy
                        and wait_for_ready_confirm
                        and not ready_confirmed
                    ):
                        can_run_policy = False
                    if (
                        rtc_frontend is None
                        and position_deploy_executor is not None
                        and can_run_policy
                        and policy_deploy_runtime.action_queue_length() == 0
                        and _position_source_tail_pending_before_synchronous_chunk_gap(
                            position_deploy_executor
                        )
                    ):
                        # The motion layer shifted this non-RTC chunk to begin
                        # after planner preparation. Keep the deployment-owned
                        # chunk boundary open until its final staged source
                        # timestamp has actually been emitted; otherwise the
                        # next clear/Hold would truncate the shifted tail.
                        can_run_policy = False
                    if (
                        rtc_frontend is None
                        and position_deploy_executor is not None
                        and can_run_policy
                        and policy_deploy_runtime.action_queue_length() == 0
                        and _hold_active_position_stream_for_synchronous_chunk_gap(
                            position_deploy_executor
                        )
                    ):
                        if position_plan_cut_detector is not None:
                            position_plan_cut_detector.mark_hold_acknowledged()
                        console_log_event(
                            "Policy",
                            "WAIT: action chunk exhausted; position stream entered "
                            "Hold before synchronous inference.",
                            now=loop_started_at,
                        )
                        debug_logger.log_event(
                            "position_chunk_gap_hold",
                            {
                                "schema": "dexteleop_position_chunk_gap_hold_v1",
                                "reason": (
                                    "synchronous_inference_with_empty_action_queue"
                                ),
                                "queue_len": 0,
                                "observation_step_index": int(observation.step_index),
                            },
                        )
                    if (
                        rtc_frontend is None
                        and can_run_policy
                        and bool(send_actions)
                        and bool(getattr(args, "policy_inference_soft_hold", True))
                        and policy_deploy_runtime.action_queue_length() == 0
                    ):
                        request_policy_inference_soft_hold(
                            arm_backends,
                            reason="policy_inference_wait_for_chunk",
                        )
                    if (
                        rtc_frontend is None
                        and can_run_policy
                        and policy_deploy_runtime.should_predict(
                            now=control_loop_started_at,
                            policy_interval_s=1.0 / float(effective_policy_hz),
                            min_action_queue=int(args.min_action_queue),
                        )
                    ):
                        prediction = policy_deploy_runtime.predict_and_queue(
                            policy,
                            observation,
                            now=control_loop_started_at,
                            queue_policy=action_queue_policy,
                            discard_policy_output=ik_action_gate.discards_policy_output,
                            action_keyframe_period_sec=(
                                1.0 / float(effective_action_hz)
                            ),
                        )
                        (
                            synchronous_defer,
                            synchronous_pause,
                        ) = record_policy_prediction_result(
                            prediction,
                            queue_policy=action_queue_policy,
                            faults=faults,
                            timestamp=loop_started_at,
                            step_index=step_index,
                            observation_step_index=observation.step_index,
                        )
                        defer_position_action_until_fresh_cut = (
                            defer_position_action_until_fresh_cut
                            or synchronous_defer
                        )
                        if synchronous_pause:
                            policy_active = False
                            can_run_policy = False
                            if position_deploy_executor is not None:
                                position_deploy_executor.clear()
                                last_action = observation.hold_action()

                    # An empty policy queue is a measured-pose soft hold for
                    # impedance arms.  A fresh policy command releases that
                    # lease in the backend before the next 40 Hz tick; the
                    # explicit option above keeps a rollback to legacy
                    # target continuation available for controlled debugging.
                    collision_reset_waiting = False
                    if (
                        position_deploy_executor is not None
                        and can_run_policy
                        and not defer_position_action_until_fresh_cut
                        and policy_deploy_runtime.action_queue_length() > 0
                    ):
                        collision_reset_preflight = (
                            position_deploy_executor.collision_reset_preflight(
                                observation,
                                timestamp_monotonic=control_loop_started_at,
                            )
                        )
                        if bool(collision_reset_preflight.get("required")):
                            collision_reset_waiting = not bool(
                                collision_reset_preflight.get("ready")
                            )
                            preflight_state = (
                                "waiting:"
                                + str(
                                    collision_reset_preflight.get("reason") or "unknown"
                                )
                                if collision_reset_waiting
                                else "ready"
                            )
                            if collision_reset_console_gate.should_emit(
                                "first_collision_reset",
                                preflight_state,
                                now=loop_started_at,
                                heartbeat_sec=5.0,
                                min_interval_sec=1.0,
                            ):
                                if collision_reset_waiting:
                                    console_log_event(
                                        "Collision",
                                        "WAIT: queued policy action retained until fresh "
                                        "dual-arm collision-reset state is available; "
                                        + str(collision_reset_preflight.get("reason")),
                                        now=loop_started_at,
                                    )
                                else:
                                    console_log_event(
                                        "Collision",
                                        "READY: fresh dual-arm collision-reset state admitted.",
                                        now=loop_started_at,
                                    )
                                debug_logger.log_event(
                                    "collision_reset_preflight",
                                    dict(collision_reset_preflight),
                                )
                    rtc_active_latency_warmup = bool(
                        rtc_active_latency_gate_required
                        and not rtc_active_latency_gate_passed
                        and can_run_policy
                        and not ik_action_gate.blocks_action_output
                        and not defer_position_action_until_fresh_cut
                        and not collision_reset_waiting
                    )
                    allow_policy_output = (
                        can_run_policy
                        and not ik_action_gate.blocks_action_output
                        and not defer_position_action_until_fresh_cut
                        and not collision_reset_waiting
                        and not rtc_active_latency_warmup
                    )
                    legacy_owner_desired_state = (
                        (
                            "running",
                            "ready",
                            "legacy_output_eligible",
                        )
                        if allow_policy_output
                        else (
                            "hold",
                            "hold",
                            "legacy_output_not_eligible",
                        )
                    )
                    if policy_arbiter_shadow is not None:
                        shadow_gate_status = (
                            "fault"
                            if position_safety_fault_latched
                            else "ready" if allow_policy_output else "hold"
                        )
                        policy_arbiter_shadow.update_execution_gate(
                            shadow_gate_status,
                            reason=(
                                "legacy_output_eligible"
                                if shadow_gate_status == "ready"
                                else (
                                    "legacy_safety_fault_latched"
                                    if shadow_gate_status == "fault"
                                    else "legacy_output_not_eligible"
                                )
                            ),
                            produced_at=control_loop_started_at,
                        )
                    action_phase_timing: dict[str, Any] = {
                        "executor_debug_and_fault_ms": max(
                            0.0,
                            pre_action_executor_completed_at
                            - pre_action_executor_started_at,
                        )
                        * 1000.0,
                        "operator_and_lifecycle_ms": max(
                            0.0,
                            pre_action_operator_completed_at
                            - pre_action_executor_completed_at,
                        )
                        * 1000.0,
                        "arm_snapshot_ms": max(
                            0.0,
                            pre_action_arm_snapshot_completed_at
                            - pre_action_operator_completed_at,
                        )
                        * 1000.0,
                        "arm_snapshot_thread_cpu_ms": max(
                            0.0,
                            arm_snapshot_cpu_completed - arm_snapshot_cpu_started,
                        )
                        * 1000.0,
                        "ik_gate_and_hand_snapshot_ms": max(
                            0.0,
                            pre_action_hand_snapshot_completed_at
                            - pre_action_arm_snapshot_completed_at,
                        )
                        * 1000.0,
                        "observation_build_ms": max(
                            0.0,
                            pre_action_observation_completed_at
                            - pre_action_hand_snapshot_completed_at,
                        )
                        * 1000.0,
                    }
                    if rtc_frontend is None:
                        primary_prepare_started = time.monotonic()
                        primary_prepare_cpu_started = time.thread_time()
                        action_phase_timing["post_observation_policy_and_gate_ms"] = max(
                            0.0,
                            primary_prepare_started
                            - pre_action_observation_completed_at,
                        ) * 1000.0
                        action_phase_timing["before_primary_prepare_ms"] = max(
                            0.0,
                            primary_prepare_started - control_loop_started_at,
                        ) * 1000.0
                        action_step = policy_deploy_runtime.prepare_next_action(
                            observation,
                            preparer=canonical_action_pipeline,
                            policy_name=policy.name,
                            allow_output=allow_policy_output,
                            same_chunk_next_step_will_execute=(
                                sync_queue_preserves_same_chunk_successor(
                                    queue_policy=action_queue_policy,
                                    min_action_queue=args.min_action_queue,
                                )
                            ),
                        )
                        primary_prepare_completed = time.monotonic()
                        primary_prepare_cpu_completed = time.thread_time()
                        action_phase_timing["prepare_next_action_ms"] = max(
                            0.0,
                            primary_prepare_completed - primary_prepare_started,
                        ) * 1000.0
                        action_phase_timing[
                            "prepare_next_action_thread_cpu_ms"
                        ] = max(
                            0.0,
                            primary_prepare_cpu_completed
                            - primary_prepare_cpu_started,
                        ) * 1000.0
                    else:
                        rtc_step = rtc_frontend.step(
                            observation,
                            now=control_loop_started_at,
                            allow_output=allow_policy_output,
                            allow_request=(
                                allow_policy_output
                                or rtc_active_latency_warmup
                            ),
                        )
                        if (
                            rtc_active_latency_warmup
                            and rtc_step.completion is not None
                            and rtc_step.result_transition is not None
                            and rtc_step.result_transition.receipt.accepted
                        ):
                            latency_sec = _rtc_active_completion_latency_sec(
                                inference_started_at=(
                                    rtc_step.completion.result.inference_started_at
                                ),
                                released_at=rtc_step.completion.released_at,
                            )
                            rtc_active_latency_samples.append(latency_sec)
                            gate_complete, gate_accepted, observed_max = (
                                _rtc_active_latency_gate_result(
                                    rtc_active_latency_samples,
                                    discard_completions=int(
                                        args.rtc_active_warmup_discard_completions
                                    ),
                                    required_completions=int(
                                        args.rtc_active_warmup_completions
                                    ),
                                    maximum_latency_sec=float(
                                        args.rtc_active_max_inference_latency_sec
                                    ),
                                )
                            )
                            hold_rtc_if_eligible(
                                now=control_loop_started_at,
                                reason="rtc_active_latency_warmup_sample",
                            )
                            if gate_complete:
                                assert observed_max is not None
                                if not gate_accepted:
                                    _hold_active_rtc_gap(
                                        reason="rtc_active_latency_gate_rejected",
                                        send_actions=send_actions,
                                        arm_backends=arm_backends,
                                        position_deploy_executor=(
                                            position_deploy_executor
                                        ),
                                    )
                                    raise RuntimeError(
                                        "RTC Active latency gate rejected dispatch: "
                                        f"warm maximum {observed_max:.6f}s is not below "
                                        "the exclusive configured bound "
                                        f"{float(args.rtc_active_max_inference_latency_sec):.6f}s."
                                    )
                                rtc_active_latency_gate_passed = True
                                console_log_event(
                                    "RTC",
                                    "READY: Active inference latency gate passed; "
                                    f"warm_max={observed_max:.6f}s samples="
                                    f"{int(args.rtc_active_warmup_completions)}.",
                                    now=loop_started_at,
                                )
                            if not rearm_rtc_if_needed(
                                now=control_loop_started_at,
                                reason="rtc_active_latency_warmup_next_sample",
                            ):
                                raise RuntimeError(
                                    "RTC Active latency warmup could not rearm "
                                    "a fresh producer epoch."
                                )
                        rtc_gap_reason = (
                            "rtc_active_latency_warmup"
                            if rtc_active_latency_warmup
                            else (
                                "rtc_producer_not_eligible"
                            if rtc_step.state.lifecycle_state
                            != POLICY_ASYNC_PRODUCER_ELIGIBLE
                            else (
                                "rtc_plan_underrun"
                                if rtc_step.underrun
                                else None
                            )
                            )
                        )
                        rtc_hold_receipts = _hold_active_rtc_gap(
                            reason=rtc_gap_reason,
                            send_actions=send_actions,
                            arm_backends=arm_backends,
                            position_deploy_executor=position_deploy_executor,
                        )
                        if rtc_hold_receipts:
                            last_action = observation.hold_action()
                            debug_logger.log_event(
                                "rtc_active_gap_hold",
                                {
                                    "schema": "dexteleop_rtc_active_gap_hold_v1",
                                    "reason": rtc_gap_reason,
                                    "measured_qpos_hold_receipts": (
                                        rtc_hold_receipts
                                    ),
                                },
                            )
                        if rtc_step_fanout is not None:
                            rtc_step_fanout.put_nowait(rtc_step)
                        emission = rtc_step.emission_result.emission
                        if rtc_emission_publisher is not None and emission is not None:
                            publication_receipt = (
                                rtc_emission_publisher.publish_committed_emission(
                                    emission,
                                    rtc_step.emission_commit_receipt,
                                    now=control_loop_started_at,
                                )
                            )
                            if (
                                rtc_debug_observer is not None
                                and rtc_debug_observer.accepts(RTC_DEBUG_LEVEL_SUMMARY)
                            ):
                                rtc_debug_observer.offer(
                                    "rtc_emission_publication",
                                    {
                                        "schema": (
                                            "dexteleop_rtc_emission_publication_debug_v1"
                                        ),
                                        **publication_receipt.to_payload(),
                                        "publication_receipt_hash": (
                                            publication_receipt.semantic_content_hash
                                        ),
                                    },
                                )
                            if not publication_receipt.accepted:
                                raise RuntimeError(
                                    "RTC committed emission publication was "
                                    "rejected: "
                                    f"{publication_receipt.reasons}"
                                )
                        if rtc_emission_fanout is not None and emission is not None:
                            commit = rtc_step.emission_commit_transition
                            if (
                                commit is None
                                or not commit.receipt.accepted
                                or commit.receipt.emission_id != emission.emission_id
                            ):
                                raise RuntimeError(
                                    "RTC emission publication requires exact "
                                    "accepted producer commit evidence."
                                )
                            rtc_emission_fanout.put_nowait(emission)
                        primary_prepare_started = time.monotonic()
                        primary_prepare_cpu_started = time.thread_time()
                        action_phase_timing["before_primary_prepare_ms"] = max(
                            0.0,
                            primary_prepare_started - control_loop_started_at,
                        ) * 1000.0
                        action_step = policy_deploy_runtime.prepare_emission_result(
                            rtc_step.emission_result,
                            observation,
                            preparer=rtc_action_preparer,
                            policy_name=policy.name,
                        )
                        primary_prepare_completed = time.monotonic()
                        primary_prepare_cpu_completed = time.thread_time()
                        action_phase_timing["prepare_emission_result_ms"] = max(
                            0.0,
                            primary_prepare_completed - primary_prepare_started,
                        ) * 1000.0
                        action_phase_timing[
                            "prepare_emission_result_thread_cpu_ms"
                        ] = max(
                            0.0,
                            primary_prepare_cpu_completed
                            - primary_prepare_cpu_started,
                        ) * 1000.0
                        if rtc_debug_observer is not None:
                            summary_worthy = bool(
                                rtc_step.completion is not None
                                or rtc_step.result_transition is not None
                                or rtc_step.emission_result.emission is not None
                                or rtc_step.request_transition is not None
                                or rtc_step.lifecycle_transition is not None
                                or rtc_step.budget_exceeded
                            )
                            minimum_level = (
                                RTC_DEBUG_LEVEL_SUMMARY
                                if summary_worthy
                                else RTC_DEBUG_LEVEL_TRACE
                            )
                            if rtc_debug_observer.accepts(minimum_level):
                                rtc_debug_observer.offer(
                                    "rtc_frontend_step",
                                    rtc_step.to_debug_payload(),
                                    minimum_level=minimum_level,
                                )
                        if (
                            rtc_step.result_transition is not None
                            and not rtc_step.result_transition.receipt.accepted
                        ):
                            faults.extend(
                                "rtc_result_rejected:" f"{reason}"
                                for reason in (
                                    rtc_step.result_transition.receipt.reasons
                                )
                            )
                        elif (
                            rtc_step.result_transition is not None
                            and rtc_step.result_transition.receipt.accepted
                        ):
                            ik_action_gate.mark_policy_output_accepted()
                        if rtc_step.budget_exceeded:
                            faults.append("rtc_control_budget_exceeded")
                        if (
                            rtc_step.state.lifecycle_state
                            != POLICY_ASYNC_PRODUCER_ELIGIBLE
                        ):
                            policy_active = False
                    emission_observer_error: str | None = None
                    if (
                        policy_emission_observer is not None
                        and action_step.emission is not None
                    ):
                        try:
                            policy_emission_observer.put_nowait(
                                action_step.emission
                            )
                        except Exception as error:  # noqa: BLE001 - passive tap.
                            emission_observer_error = (
                                f"{type(error).__name__}:{error}"
                            )
                            debug_logger.log_event(
                                "passive_policy_emission_observer_error",
                                {
                                    "emission_id": (
                                        action_step.emission.emission_id
                                    ),
                                    "error": emission_observer_error,
                                    "authoritative": False,
                                },
                            )
                    arbiter_shadow_step = None
                    if policy_arbiter_shadow is not None:
                        if action_step.emission is not None:
                            if debug_replay:
                                if (
                                    replay_proposal_shadow_adapter is not None
                                    and action_step.raw_action is not None
                                ):
                                    try:
                                        replay_adapter_result = replay_proposal_shadow_adapter.adapt_emission(
                                            action_step.emission,
                                            raw_action_metadata=(
                                                action_step.raw_action.metadata
                                            ),
                                        )
                                        debug_logger.log_event(
                                            "replay_proposal_adapter",
                                            replay_adapter_debug_payload(
                                                replay_adapter_result
                                            ),
                                        )
                                        if (
                                            replay_adapter_result.proposal_draft
                                            is not None
                                        ):
                                            arbiter_shadow_step = policy_arbiter_shadow.evaluate_proposal_draft(
                                                replay_adapter_result.proposal_draft,
                                                source_event_id=(
                                                    replay_adapter_result.source_adapter_receipt.representation_trace_id
                                                ),
                                                emission=action_step.emission,
                                            )
                                        else:
                                            arbiter_shadow_step = policy_arbiter_shadow.status(
                                                "replay_adapter_rejected",
                                                reason="replay_source_or_binding_invalid",
                                                emission_id=action_step.emission.emission_id,
                                                emission=action_step.emission,
                                                occurred_at=control_loop_started_at,
                                            )
                                    except (KeyError, TypeError, ValueError) as exc:
                                        arbiter_shadow_step = policy_arbiter_shadow.status(
                                            "replay_adapter_error",
                                            reason=type(exc).__name__,
                                            emission_id=action_step.emission.emission_id,
                                            emission=action_step.emission,
                                            occurred_at=control_loop_started_at,
                                        )
                                else:
                                    arbiter_shadow_step = policy_arbiter_shadow.status(
                                        "replay_adapter_unavailable",
                                        reason="typed_replay_contract_unavailable",
                                        emission_id=action_step.emission.emission_id,
                                        emission=action_step.emission,
                                        occurred_at=control_loop_started_at,
                                    )
                            else:
                                arbiter_shadow_step = (
                                    policy_arbiter_shadow.evaluate_emission(
                                        action_step.emission
                                    )
                                )
                        elif action_step.emission_error is not None:
                            arbiter_shadow_step = policy_arbiter_shadow.status(
                                "emission_error",
                                reason=type(action_step.emission_error).__name__,
                                occurred_at=control_loop_started_at,
                            )
                        if arbiter_shadow_step is not None:
                            debug_logger.log_event(
                                "arbitration_shadow_step",
                                arbiter_shadow_step.to_debug_payload(),
                            )
                    if (
                        action_step.emission_error is not None
                        and action_step.raw_action is None
                    ):
                        faults.append(
                            "policy_action_semantic_resolution_error:"
                            + type(action_step.emission_error).__name__
                        )
                        policy_deploy_runtime.clear_action_queue()
                        if bool(args.pause_on_policy_error):
                            policy_active = False
                    queue_len_before_pop = action_step.queue_len_before
                    raw_action = action_step.raw_action
                    raw_lookahead_action = action_step.raw_lookahead_action
                    raw_second_lookahead_action = (
                        action_step.raw_second_lookahead_action
                    )
                    raw_third_lookahead_action = action_step.raw_third_lookahead_action
                    queue_len_after_pop = action_step.queue_len_after
                    if action_step.prepared_action is not None:
                        if action_step.emission is None:
                            emission_error = action_step.emission_error
                            debug_logger.log_event(
                                "policy_action_emission_unavailable",
                                {
                                    "schema": (
                                        "dexteleop_policy_action_emission_unavailable_v1"
                                    ),
                                    "non_authoritative": True,
                                    "motion_lineage_source": (
                                        "normalized_action_chunk_metadata"
                                        if rtc_frontend is None
                                        else "blocked_missing_rtc_emission"
                                    ),
                                    "error_type": (
                                        None
                                        if emission_error is None
                                        else type(emission_error).__name__
                                    ),
                                    "error": (
                                        None
                                        if emission_error is None
                                        else str(emission_error)
                                    ),
                                },
                            )
                        canonical_action = action_step.prepared_action
                        if (
                            raw_action is not None
                            and not debug_replay
                            and action_step.emission is not None
                            and arbiter_shadow_step is not None
                            and arbiter_shadow_step.evaluation is not None
                        ):
                            try:
                                policy_safety_comparison = (
                                    evaluate_policy_legacy_aligned_safety(
                                        emission=action_step.emission,
                                        arbitration_evaluation_id=(
                                            arbiter_shadow_step.evaluation.evaluation_id
                                        ),
                                        observation=observation,
                                        legacy_result=canonical_action,
                                        safety_filter=(
                                            canonical_action_pipeline.safety_filter
                                        ),
                                    )
                                )
                                debug_logger.log_event(
                                    "policy_safety_shadow_comparison",
                                    policy_safety_comparison.to_debug_payload(),
                                )
                            except Exception as exc:  # noqa: BLE001
                                # This is a passive observer.  Its failure must
                                # not alter the legacy fault list, last_action,
                                # queue state, or execution result.
                                debug_logger.log_event(
                                    "policy_safety_shadow_error",
                                    {
                                        "non_authoritative": True,
                                        "emission_id": action_step.emission.emission_id,
                                        "arbitration_evaluation_id": (
                                            arbiter_shadow_step.evaluation.evaluation_id
                                        ),
                                        "error_type": type(exc).__name__,
                                    },
                                )
                        prepared_lookahead_chain = ()
                        if (
                            position_deploy_executor is not None
                            and bool(args.position_chunk_lookahead)
                            and raw_lookahead_action is not None
                            and not canonical_action.faults
                        ):
                            raw_lookahead_chain = [raw_lookahead_action]
                            if (
                                raw_second_lookahead_action is not None
                                and action_step.next_step_execution_signal is not None
                            ):
                                raw_lookahead_chain.append(
                                    raw_second_lookahead_action
                                )
                                if (
                                    bool(
                                        args.position_continuous_waypoint_async_precompute
                                    )
                                    and raw_third_lookahead_action is not None
                                ):
                                    raw_lookahead_chain.append(
                                        raw_third_lookahead_action
                                    )
                            prepared_lookahead_chain = _prepare_position_lookahead_chain(
                                tuple(raw_lookahead_chain),
                                observation,
                                canonical_action,
                                pipeline=canonical_action_pipeline,
                                policy_name=policy.name,
                                phase_timing=action_phase_timing,
                            )
                        lookahead_slots = prepared_lookahead_chain + (
                            (None, None),
                        ) * (3 - len(prepared_lookahead_chain))
                        faults.extend(canonical_action.faults)
                        latch_canonical_safety_fault(
                            canonical_action,
                            observation,
                            policy_name=policy.name,
                        )
                        if (
                            rtc_frontend is not None
                            and canonical_action.faults
                            and rtc_frontend.state.lifecycle_state
                            == POLICY_ASYNC_PRODUCER_ELIGIBLE
                        ):
                            rtc_fault = rtc_frontend.fault(
                                now=control_loop_started_at,
                                reason="canonical_safety_fault",
                            )
                            debug_logger.log_event(
                                "rtc_lifecycle_transition",
                                {
                                    "target": "FAULT",
                                    "reason": "canonical_safety_fault",
                                    "producer_transition_hash": (
                                        rtc_fault.producer_transition.receipt.semantic_content_hash
                                    ),
                                    "worker_stop_hash": (
                                        rtc_fault.worker_stop_receipt.semantic_content_hash
                                    ),
                                },
                            )
                            policy_active = False
                        dispatch_error: BaseException | None = None
                        execution: _PreparedActionExecution | None = None
                        plan_cut_proposal = None
                        post_arm_states: dict[str, dict[str, Any]] | None = None
                        if position_deploy_executor is None:
                            annotate_deploy_arm_command_limits(
                                canonical_action.command,
                                args=args,
                                active_sides=active_sides,
                                control_hz=backend_control_hz,
                                action_hz=float(effective_action_hz),
                            )
                        try:
                            if (
                                canonical_action.command is not None
                                and canonical_action.command.commands
                            ):
                                if (
                                    position_plan_cut_detector is not None
                                    and position_deploy_executor is not None
                                ):
                                    plan_cut_prepare_started = time.monotonic()
                                    plan_cut_proposal = _prepare_position_plan_cut_for_action_step(
                                        position_plan_cut_detector,
                                        action_step,
                                        rtc_enabled=rtc_frontend is not None,
                                        policy_node_id=(
                                            policy_deploy_runtime.config.policy_node_id
                                        ),
                                        hold_generation=int(
                                            position_deploy_executor.snapshot().get(
                                                "clear_count",
                                                0,
                                            )
                                        ),
                                        detected_at=control_loop_started_at,
                                    )
                                    action_phase_timing["plan_cut_prepare_ms"] = max(
                                        0.0,
                                        time.monotonic() - plan_cut_prepare_started,
                                    ) * 1000.0
                                execute_prepared_started = time.monotonic()
                                execution = prepared_action_executor.execute(
                                    canonical_action,
                                    observation,
                                    policy_name=policy.name,
                                    source_decision_id=(
                                        action_step.emission.emission_id
                                        if action_step.emission is not None
                                        else (
                                            "legacy-policy-decision:"
                                            f"{policy.name}:{step_index}"
                                        )
                                    ),
                                    source_reference_segment_id=(
                                        plan_cut_proposal.cut.cut_id
                                        if plan_cut_proposal is not None
                                        else (
                                            "legacy-policy-reference:"
                                            f"{policy.name}:{step_index}"
                                        )
                                    ),
                                    lookahead_canonical_action=lookahead_slots[0][0],
                                    lookahead_observation=lookahead_slots[0][1],
                                    second_lookahead_canonical_action=(
                                        lookahead_slots[1][0]
                                    ),
                                    second_lookahead_observation=(
                                        lookahead_slots[1][1]
                                    ),
                                    third_lookahead_canonical_action=(
                                        lookahead_slots[2][0]
                                    ),
                                    third_lookahead_observation=(
                                        lookahead_slots[2][1]
                                    ),
                                    source_plan_cut=(
                                        None
                                        if plan_cut_proposal is None
                                        else plan_cut_proposal.cut
                                    ),
                                    phase_timing=action_phase_timing,
                                )
                                action_phase_timing["execute_prepared_action_ms"] = max(
                                    0.0,
                                    time.monotonic() - execute_prepared_started,
                                ) * 1000.0
                                if plan_cut_proposal is not None:
                                    if execution.accepted:
                                        position_plan_cut_detector.commit(
                                            plan_cut_proposal
                                        )
                                    else:
                                        position_plan_cut_detector.reject(
                                            plan_cut_proposal
                                        )
                                    plan_cut_proposal = None
                                if execution.accepted:
                                    if rtc_active_latency_gate_required:
                                        accepted_rtc_dispatch_steps += 1
                                    position_action_submitted_this_loop = (
                                        prepared_action_position_executor is not None
                                    )
                                    last_action = _prepared_action_last_action(
                                        canonical_action,
                                        observation,
                                        deploy_layout=deploy_layout,
                                    )
                            if debug_logger.enabled:
                                post_state_started = time.monotonic()
                                post_arm_states = _debug_backend_arm_states(
                                    arm_backends
                                )
                                action_phase_timing["post_arm_state_snapshot_ms"] = max(
                                    0.0,
                                    time.monotonic() - post_state_started,
                                ) * 1000.0
                        except Exception as exc:
                            if plan_cut_proposal is not None:
                                position_plan_cut_detector.reject(plan_cut_proposal)
                                plan_cut_proposal = None
                            dispatch_error = exc
                            if (
                                policy_arbiter_shadow is not None
                                and action_step.emission is not None
                            ):
                                legacy_shadow_event = (
                                    policy_arbiter_shadow.observe_legacy_execution(
                                        emission=action_step.emission,
                                        shadow_step=arbiter_shadow_step,
                                        canonical_action=canonical_action,
                                        execution=execution,
                                        dispatch_error=dispatch_error,
                                    )
                                )
                                debug_logger.log_event(
                                    "legacy_execution_observation",
                                    legacy_shadow_event.to_debug_payload(),
                                )
                            if debug_logger.enabled:
                                post_state_started = time.monotonic()
                                post_arm_states = _debug_backend_arm_states(
                                    arm_backends
                                )
                                action_phase_timing["post_arm_state_snapshot_ms"] = max(
                                    0.0,
                                    time.monotonic() - post_state_started,
                                ) * 1000.0
                                action_phase_timing["loop_before_debug_ms"] = max(
                                    0.0,
                                    time.monotonic() - control_loop_started_at,
                                ) * 1000.0
                                debug_logger.log_action_step(
                                    timestamp=loop_started_at,
                                    step_index=step_index,
                                    policy_name=policy.name,
                                    raw_action=raw_action,
                                    canonical_action=canonical_action,
                                    pre_arm_states=arm_states,
                                    post_arm_states=post_arm_states,
                                    arm_backends=arm_backends,
                                    send_actions=send_actions,
                                    queue_len_before_pop=queue_len_before_pop,
                                    queue_len_after_pop=queue_len_after_pop,
                                    faults=faults,
                                    dispatch_error=dispatch_error,
                                    execution=execution,
                                    phase_timing=action_phase_timing,
                                )
                            raise
                        if (
                            policy_arbiter_shadow is not None
                            and action_step.emission is not None
                        ):
                            legacy_shadow_event = (
                                policy_arbiter_shadow.observe_legacy_execution(
                                    emission=action_step.emission,
                                    shadow_step=arbiter_shadow_step,
                                    canonical_action=canonical_action,
                                    execution=execution,
                                    dispatch_error=dispatch_error,
                                )
                            )
                            debug_logger.log_event(
                                "legacy_execution_observation",
                                legacy_shadow_event.to_debug_payload(),
                            )
                        action_phase_timing["loop_before_debug_ms"] = max(
                            0.0,
                            time.monotonic() - control_loop_started_at,
                        ) * 1000.0
                        debug_logger.log_action_step(
                            timestamp=loop_started_at,
                            step_index=step_index,
                            policy_name=policy.name,
                            raw_action=raw_action,
                            canonical_action=canonical_action,
                            pre_arm_states=arm_states,
                            post_arm_states=post_arm_states,
                            arm_backends=arm_backends,
                            send_actions=send_actions,
                            queue_len_before_pop=queue_len_before_pop,
                            queue_len_after_pop=queue_len_after_pop,
                            faults=faults,
                            dispatch_error=dispatch_error,
                            execution=execution,
                            phase_timing=action_phase_timing,
                        )

                    if (
                        position_deploy_executor is not None
                        and not position_action_submitted_this_loop
                        and (
                            ik_action_gate.blocks_action_output
                            or bool(faults)
                            or not can_run_policy
                            or (wait_for_ready_confirm and not ready_confirmed)
                        )
                        and bool(position_deploy_executor.snapshot().get("active"))
                    ):
                        position_deploy_executor.clear()
                        last_action = observation.hold_action()

                if (
                    position_safety_fault_latched
                    or position_force_guard_fault_latched
                    or impedance_force_guard_fault_latched
                ):
                    legacy_owner_desired_state = (
                        "fault",
                        "fault",
                        "legacy_safety_fault_latched",
                    )
                elif not policy_active:
                    legacy_owner_desired_state = (
                        "hold",
                        "hold",
                        "legacy_policy_inactive",
                    )
                owner_transition = submit_legacy_owner_state(
                    lifecycle_state=legacy_owner_desired_state[0],
                    execution_gate=legacy_owner_desired_state[1],
                    reason=legacy_owner_desired_state[2],
                    source_event_ids=(
                        ()
                        if pending_legacy_owner_recovery_authorization_id is None
                        else (
                            pending_legacy_owner_recovery_authorization_id,
                        )
                    ),
                    recovery_authorization_id=(
                        pending_legacy_owner_recovery_authorization_id
                        if legacy_owner_desired_state[0] == "hold"
                        else None
                    ),
                )
                if (
                    owner_transition is not None
                    and owner_transition.changed
                    and owner_transition.receipt is not None
                    and owner_transition.receipt.recovery_authorization_id
                    == pending_legacy_owner_recovery_authorization_id
                ):
                    pending_legacy_owner_recovery_authorization_id = None

                fault_text = _compact_faults(faults)
                if fault_text != last_fault_text:
                    console_log_event(
                        "Safety",
                        (
                            "READY: faults=none"
                            if fault_text == "none"
                            else f"ERROR: faults={fault_text}"
                        ),
                        now=loop_started_at,
                    )
                    last_fault_text = fault_text

                if loop_started_at - last_status_at >= float(args.status_sec):
                    policy_status = _policy_status_text(policy)
                    pending_action_count = (
                        policy_deploy_runtime.action_queue_length()
                        if rtc_frontend is None
                        else len(rtc_frontend.state.plan.keyframes)
                    )
                    position_executor_status = ""
                    if position_deploy_executor is not None:
                        position_snapshot = position_deploy_executor.snapshot()
                        last_collision = dict(
                            position_snapshot.get("last_collision") or {}
                        )
                        last_collision_reset = dict(
                            position_snapshot.get("last_collision_reset") or {}
                        )
                        collision_reset_status = (
                            "pending"
                            if not last_collision_reset
                            else (
                                "authoritative"
                                if bool(last_collision_reset.get("authoritative"))
                                else "dry_run_fallback"
                            )
                        )
                        collision_distance = last_collision.get(
                            "position_collision_distance_m"
                        )
                        try:
                            collision_distance_text = (
                                "n/a"
                                if collision_distance is None
                                else f"{float(collision_distance):.4f}"
                            )
                        except (TypeError, ValueError):
                            collision_distance_text = "invalid"
                        reference_count = int(
                            position_snapshot.get("reference_count", 0)
                        )
                        queue_depth = int(position_snapshot.get("queue_depth", 0))
                        missed_ticks = int(
                            position_snapshot.get("missed_reference_ticks", 0)
                        )
                        collision_status = str(
                            last_collision.get(
                                "position_collision_status",
                                "pending",
                            )
                        )
                        position_status_parts = [
                            (
                                "position="
                                f"refs:{reference_count}/queue:{queue_depth}"
                                f"/missed:{missed_ticks}"
                            ),
                            (
                                "collision="
                                f"{collision_status}/d:{collision_distance_text}"
                                f"/reset:{collision_reset_status}"
                            ),
                        ]
                        hand_attempts = int(
                            position_snapshot.get(
                                "discrete_dispatch_attempt_count",
                                0,
                            )
                        )
                        hand_dispatches = int(
                            position_snapshot.get("discrete_dispatch_count", 0)
                        )
                        hand_skips = int(
                            position_snapshot.get("discrete_skipped_count", 0)
                        ) + int(
                            position_snapshot.get(
                                "discrete_backend_skipped_count",
                                0,
                            )
                        )
                        if hand_attempts or hand_dispatches or hand_skips:
                            position_status_parts.append(
                                "hands="
                                f"{hand_dispatches}/{hand_attempts}"
                                f"/skip:{hand_skips}"
                            )
                        debug_drops = int(
                            position_snapshot.get("debug_event_drop_count", 0)
                        )
                        if debug_drops:
                            position_status_parts.append(f"debug_drop={debug_drops}")
                        position_executor_status = " ".join(position_status_parts)
                    status_parts = [
                        text
                        for text in (
                            (
                                fixed_init_controller.status_text
                                if fixed_init_controller is not None
                                else ""
                            ),
                            (
                                policy_qpos_init_controller.status_text
                                if policy_qpos_init_controller is not None
                                else ""
                            ),
                            startup_initializer.status_text,
                            (
                                "ready=wait:A"
                                if (
                                    wait_for_ready_confirm
                                    and policy_active
                                    and startup_initializer.completed
                                    and not ready_confirmed
                                )
                                else ""
                            ),
                            policy_status,
                            ik_action_gate.status_text(),
                            (
                                "position_force_guard=latched:"
                                + str(
                                    position_force_guard_fault.reason
                                    if position_force_guard_fault is not None
                                    else "unknown"
                                )
                                if position_force_guard_fault_latched
                                else (
                                    "position_force_guard=armed"
                                    if (
                                        position_force_guard_monitor is not None
                                        and position_force_guard_monitor.motion_ready
                                    )
                                    else ""
                                )
                            ),
                            (
                                "impedance_guard=latched:"
                                + str(
                                    impedance_force_guard_fault.reason
                                    if impedance_force_guard_fault is not None
                                    else "unknown"
                                )
                                if impedance_force_guard_fault_latched
                                else (
                                    "impedance_guard=armed"
                                    if (
                                        impedance_force_guard_monitor is not None
                                        and impedance_force_guard_monitor.motion_ready
                                    )
                                    else ""
                                )
                            ),
                            position_executor_status,
                            (
                                "rtc="
                                f"{rtc_frontend.state.lifecycle_state.lower()} "
                                "rtc_worker="
                                f"{rtc_frontend.state.worker_session_state.lower()} "
                                "rtc_emissions="
                                f"{rtc_frontend.selected_emission_count} "
                                "rtc_active_dispatch_steps="
                                f"{accepted_rtc_dispatch_steps}"
                                if rtc_frontend is not None
                                else ""
                            ),
                        )
                        if text
                    ]
                    suffix = f" {' '.join(status_parts)}" if status_parts else ""
                    console_log_event(
                        "Status",
                        f"step={step_index} hz={control_hz:.2f} "
                        f"policy={'active' if policy_active else 'paused'} "
                        f"send={'on' if send_actions else 'dry-run'} "
                        f"queue={pending_action_count} "
                        f"state_valid={int(np.sum(observation.state_valid))}/{observation.state_valid.size} "
                        f"{camera_status_text(camera_sources, camera_previewer)}"
                        f"{suffix}",
                        now=loop_started_at,
                    )
                    last_status_at = loop_started_at

                step_index += 1
                if _bounded_policy_step_limit_reached(
                    max_steps=int(args.max_steps),
                    control_loop_steps=step_index,
                    rtc_active_dispatch_counting=rtc_active_latency_gate_required,
                    accepted_rtc_dispatch_steps=accepted_rtc_dispatch_steps,
                ):
                    required_emissions = int(
                        getattr(args, "min_rtc_emissions_before_exit", 0)
                    )
                    actual_emissions = (
                        0
                        if rtc_frontend is None
                        else int(rtc_frontend.selected_emission_count)
                    )
                    if actual_emissions < required_emissions:
                        raise RuntimeError(
                            "Bounded RTC acceptance reached --max-steps before "
                            "the required committed emissions: "
                            f"{actual_emissions} < {required_emissions}."
                        )
                    break
                loop_target_hz = float(effective_action_hz)
                if fixed_init_controller is not None and fixed_init_controller.running:
                    loop_target_hz = max(
                        loop_target_hz, float(fixed_init_controller.plan.action_hz)
                    )
                if (
                    policy_qpos_init_controller is not None
                    and policy_qpos_init_controller.running
                ):
                    loop_target_hz = max(
                        loop_target_hz,
                        1.0 / max(0.001, float(args.policy_qpos_init_dt_sec)),
                    )
                sleep_time = (1.0 / loop_target_hz) - (time.time() - loop_started_at)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[Deploy] keyboard interrupt; requesting stop\n", flush=True)
        try:
            position_stop_requested = (
                emergency_stop_position_deploy_on_keyboard_interrupt(
                    args,
                    send_actions=send_actions,
                    arm_backends=arm_backends,
                )
            )
        except Exception as exc:  # noqa: BLE001 - cleanup still must run.
            keyboard_interrupt_stop_error = f"{type(exc).__name__}: {exc}"
            print(
                "[Deploy] CRITICAL: Ctrl+C bimanual Arm_Stop was not fully "
                f"acknowledged: {keyboard_interrupt_stop_error}",
                file=sys.stderr,
                flush=True,
            )
        else:
            if position_stop_requested:
                print(
                    "[Deploy] Ctrl+C bimanual Arm_Stop acknowledged.",
                    flush=True,
                )
    finally:
        try:
            if (
                quest_paired_pose_factory_attempted
                and quest_paired_pose_capture_factory is not None
            ):
                try:
                    quest_paired_pose_capture_factory.close()
                except Exception as exc:  # noqa: BLE001 - continue robot cleanup.
                    factory_error = (
                        "quest_paired_pose_factory_close_failed:"
                        f"{type(exc).__name__}:{exc}"
                    )
                    quest_paired_pose_capture_error = (
                        factory_error
                        if quest_paired_pose_capture_error is None
                        else f"{quest_paired_pose_capture_error};{factory_error}"
                    )
        finally:
            try:
                if arm_reader is not None:
                    arm_reader.invalidate_pose_measurements(
                        reason="policy_deploy_shutdown",
                        terminal=True,
                    )
            finally:
                cleanup_errors = resource_lifecycle.close(
                    keyboard_interrupt_stop_error=keyboard_interrupt_stop_error,
                )
                if quest_paired_pose_capture_error is not None:
                    cleanup_errors = (
                        quest_paired_pose_capture_error,
                        *cleanup_errors,
                    )
    return 1 if cleanup_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
