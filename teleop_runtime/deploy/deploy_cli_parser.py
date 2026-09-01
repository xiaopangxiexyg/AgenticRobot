"""Argument registration for the supported real-policy deployment runtime.

This module owns argparse registration only. It must remain free of deployment
startup, device I/O, policy execution, and control authority. The historical
``teleop_runtime.deploy.run_policy.add_args`` entrypoint remains a compatibility
wrapper around :func:`add_args`.
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
    HandActionSpec,
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

from robot_motion.deploy.canonical_action import CanonicalDeployActionPipeline

from robot_motion.deploy.emission_port import (
    PassivePolicyEmissionObserverPort,
    require_passive_policy_emission_observer,
)

from robot_motion.deploy.next_step import sync_queue_preserves_same_chunk_successor

from robot_motion.deploy.observation import (
    DEFAULT_HAND_STATE_MAX_AGE_SEC,
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
    ACTION_DIM,
    ALL_SIDES,
    DEPLOY_ACTION_VECTOR_ACTION_SPACE,
    to_serializable,
)

from robot_motion.deploy.session_contract import (
    POLICY_CONTRACT_V22_NATIVE,
    RuntimeEndEffectorCapability,
    resolve_deploy_session_contract,
)

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

from ..hand_mapping import parse_tactile_regions

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

from .replay import (
    bind_debug_replay_arm_control_mode,
    debug_replay_execution_exhausted as _debug_replay_execution_exhausted,
    load_dataset_replay_policy,
    skip_startup_replay_frames as _skip_startup_replay_frames,
)

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
    normalize_rtc_debug_level,
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

def add_args(parser: argparse.ArgumentParser) -> None:
    policy_qpos_defaults = DeployPolicyQposInitSettings()
    parser.add_argument(
        "--policy",
        "--policy-spec",
        dest="policy_spec",
        type=str,
        default="hold",
        help="Policy loader spec: hold, package.module:Policy, or /path/policy.py:Policy.",
    )
    parser.add_argument("--policy-kwargs-json", type=str, default="{}")
    parser.add_argument(
        "--rtc-profile",
        choices=(
            "off",
            RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
            MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
        ),
        default="off",
        help=(
            "Experimental default-off RTC frontend profile. RTC-3c remains the builtin-hold "
            "virtual shadow; raw_rtc_final_chunk enables an explicit model adapter with "
            "the existing final-chunk model/server contract; model_rtc_flow_inpaint "
            "uses the artifact-bound conditioned v3 provider. All force 30 Hz "
            "action/decision, 15 Hz observation requests, disable position chunk "
            "lookahead, and forbid motion output."
        ),
    )
    parser.add_argument(
        "--rtc-real-shadow",
        action="store_true",
        help=(
            "Opt into the explicit live observation/transport no-action shadow. "
            "Requires the raw or conditioned model RTC profile; hardware backends are "
            "configured for observation-only startup (no command, force, tool-frame, "
            "singularity, or background-stream writes). Default is off."
        ),
    )
    parser.add_argument(
        "--rtc-request-validity-steps",
        type=_positive_int,
        default=10,
        help=(
            "RTC request/plan validity in action-keyframe steps. The default 10 "
            "preserves the existing fail-closed profile; explicit model deployments "
            "may bind a larger tested window without changing model/server payloads."
        ),
    )
    parser.add_argument(
        "--rtc-worker-stop-timeout-sec",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Bounded wait for an in-flight RTC worker during frontend close. "
            "0 preserves the existing non-waiting close behavior."
        ),
    )
    parser.add_argument(
        "--policy-artifact-json",
        type=Path,
        default=None,
        help=(
            "Validated dexteleop_policy_artifact_v1 sidecar. Enables strict v2.2 "
            "layout/dimension binding before backend construction."
        ),
    )
    parser.add_argument(
        "--policy-action-pose-mode",
        choices=("absolute", "relative"),
        default="absolute",
        help=(
            "Pose representation emitted by the policy. Relative actions are converted using "
            "the current measured TCP before canonical safety/IK; default is absolute."
        ),
    )
    parser.add_argument(
        "--policy-observation-firewall-mode",
        choices=("strict", "dual-validate"),
        default="strict",
        help=(
            "Model-input firewall mode. strict (default) makes the typed actor view the "
            "only prediction source; dual-validate is a dry-run compatibility check and "
            "is forbidden with --send-actions."
        ),
    )
    parser.add_argument(
        "--action-arbiter-mode",
        choices=ARBITER_SHADOW_MODES_M3,
        default="off",
        help=(
            "M3 passive ActionArbiter mode. off preserves the legacy path; shadow "
            "evaluates the same popped PolicyActionEmission without dispatch authority "
            "and is currently restricted to dry-run/replay."
        ),
    )
    parser.add_argument(
        "--runtime-pose-frame-contract",
        type=load_pose_frame_contract,
        default=None,
        metavar="JSON",
        help=(
            "Current robot runtime raw_absolute->canonical-desk pose contract. "
            "Native-v2.3-trained pose policies default to the calibrated dual-RM65 "
            "URDF workspace-near-boundary profile; use this option for another station. Policies "
            "declared as v2.2 raw identity compatibility may only use the matching "
            "same-source identity binding; omitting it selects that audited identity "
            "profile."
        ),
    )
    parser.add_argument(
        "--debug-replay-output-pose-mode",
        choices=("absolute", "relative"),
        default="absolute",
        help=(
            "Representation emitted by the v2.2 debug-replay bridge. Relative mode "
            "uses the measured TCP at each replay chunk start and must be paired with "
            "--policy-action-pose-mode relative."
        ),
    )
    parser.add_argument("--active-sides", nargs="+", default=("left", "right"))
    parser.add_argument(
        "--end-effector-sides",
        type=str,
        default="active",
        help=(
            "Configured hand/gripper sides: active (default), none, left, right, or both. "
            "This is independent from the two fixed arm-pose slots."
        ),
    )
    parser.add_argument(
        "--hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_SAMPLE_HZ,
        help="Fallback action rate when --action-hz is 0 and no dataset/policy rate is inferred.",
    )
    parser.add_argument(
        "--arm-execution-profile",
        choices=ARM_EXECUTION_PROFILE_CHOICES,
        default=None,
        help=(
            "Canonical real-arm profile shared with the existing teleop runtime: "
            "position deploy defaults to position_15_60_120hz. position and "
            "position_45_150hz remain explicit compatibility profiles; per-stage "
            "rate flags remain supported as overrides."
        ),
    )
    parser.add_argument(
        "--arm-control-mode",
        choices=RM65_ARM_CONTROL_MODES,
        default=RM65_ARM_CONTROL_MODE_POSITION,
        help=(
            "RM65 deployment execution mode. position uses Movej_CANFD passthrough by default; "
            "impedance uses the force-admittance backend at 40Hz without CANFD follow by default."
        ),
    )
    parser.add_argument(
        "--position-control-profile",
        choices=RM65_POSITION_CONTROL_PROFILE_CHOICES,
        default=RM65_BIMANUAL_POSITION_V1_NAME,
        help=(
            "Versioned position action-execution profile. Policy/dataset main actions "
            "are already source-shaped, so the formal deploy path does not repeat "
            "collection-side inactive, tracker, or post-IK qref smoothing."
        ),
    )
    parser.add_argument(
        "--tcp-pose-envelope-profile",
        type=Path,
        default=None,
        help=(
            "Optional strict JSON tool-TCP workspace/rotation envelope. The "
            "deploy adapter projects runtime-raw Pose targets in canonical desk "
            "before IK; providing a profile alone is shadow-only and leaves "
            "commands unchanged."
        ),
    )
    parser.add_argument(
        "--tcp-pose-envelope-active",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly activate the configured deploy tool-TCP envelope. "
            "Default is off; requires --tcp-pose-envelope-profile and position "
            "Pose execution."
        ),
    )
    parser.add_argument(
        "--position-path-lift-backend",
        choices=POSITION_PATH_LIFT_BACKEND_CHOICES,
        default=OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND,
        help=(
            "Pose-to-QPOS execution backend. The default promotes the "
            "offline-validated official source all-IK + two-source-step "
            "120Hz jerk-limited governor. pink_pose_lift preserves the former "
            "V4 deployment path and honors --pose-lift-profile."
        ),
    )
    parser.add_argument(
        "--position-all-ik-lookahead-source-steps",
        type=int,
        default=2,
        help="Source-keyframe horizon for the official all-IK governor (default: 2).",
    )
    parser.add_argument(
        "--position-all-ik-jerk-ramp-sec",
        type=_positive_float,
        default=0.05,
        help="Acceleration-to-jerk ramp used by the official all-IK governor.",
    )
    parser.add_argument(
        "--position-qref-acceleration-profile",
        choices=RM65_QREF_ACCELERATION_PROFILE_CHOICES,
        default=RM65_QREF_ACCELERATION_COMFORT_75_V1,
        help=(
            "Deploy-only accepted-qref acceleration envelope (default: "
            "comfort_75_v1 for the official all-IK governor). hard_v1 is the "
            "exact rollback; comfort profiles scale the 60 Hz all-IK governor "
            "qdd and derived jerk limits while leaving the 120 Hz backend hard "
            "limits and all live-human teleoperation defaults unchanged. "
            "An implicit non-official backend fallback retains hard_v1."
        ),
    )
    parser.add_argument(
        "--position-all-ik-batched-solver-tolerance",
        type=_positive_float,
        default=1e-6,
        help=(
            "OSQP tolerance for the reusable official all-IK horizon workspace. "
            "The default matches its hard-certificate scale; pass 1e-4 for "
            "the former faster implementation as an explicit rollback."
        ),
    )
    parser.add_argument(
        "--position-continuous-waypoint-profile",
        choices=("async_c2", "formal_third_order"),
        default="formal_third_order",
        help=(
            "Source-cut execution profile for formal dual-arm position deploy. "
            "formal_third_order is the production default after real-runtime "
            "deadline evidence showed that the bounded async C2 worker can "
            "contend with the authoritative 60 Hz reference loop. async_c2 "
            "remains an explicit, atomically reversible research profile and "
            "falls back by complete source cut when its candidate is unavailable; "
            "it requires --debug."
        ),
    )
    parser.add_argument(
        "--position-continuous-waypoint-shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute and debug-log a non-authoritative continuous C2 waypoint "
            "candidate beside the official all-IK governor. This default-off "
            "shadow never changes qcmd, collision admission, safety, or dispatch "
            "and requires --debug. Use with --position-continuous-waypoint-profile "
            "formal_third_order for a shadow-only comparison."
        ),
    )
    parser.add_argument(
        "--position-continuous-waypoint-experimental-admission",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compatibility/research switch selecting a synchronously built C2 path "
            "for a complete 15 Hz source cut after whole-cut sampled collision "
            "admission. Default off; requires --debug."
        ),
    )
    parser.add_argument(
        "--position-continuous-waypoint-async-precompute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compatibility/research switch that adds bounded precompute to the "
            "experimental admission switch. The 120 Hz owner never waits and consumes "
            "only an exact epoch/source/anchor match; otherwise the whole cut "
            "falls back to the formal governor. The isolated async C2 experiment "
            "is selected by --position-continuous-waypoint-profile async_c2 and "
            "requires --debug."
        ),
    )
    parser.add_argument(
        "--position-reference-worker-cpu",
        type=_non_negative_int,
        default=None,
        help=(
            "Default-off native CPU pin for the authoritative position-reference "
            "worker. It may be used alone for formal_third_order A/B soaks. "
            "When async_c2 is selected it must be paired with a distinct "
            "--position-continuous-waypoint-worker-cpu."
        ),
    )
    parser.add_argument(
        "--position-cyclic-gc-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run one generation-2 cyclic collection immediately before the first "
            "position stream is queued, then suppress automatic cyclic GC until "
            "the position executor closes. This prevents a process-wide GIL pause "
            "from starving the authoritative reference worker; ordinary reference "
            "counting remains active. Use --no-position-cyclic-gc-guard for the "
            "exact rollback."
        ),
    )
    parser.add_argument(
        "--position-continuous-waypoint-worker-cpu",
        type=_non_negative_int,
        default=None,
        help=(
            "Default-off native CPU pin for the async C2 proposal worker. It "
            "must be paired with a distinct --position-reference-worker-cpu. "
            "Removing both options is the atomic affinity rollback."
        ),
    )
    parser.add_argument(
        "--position-continuous-waypoint-worker-isolation",
        choices=CONTINUOUS_WAYPOINT_WORKER_ISOLATION_CHOICES,
        default=CONTINUOUS_WAYPOINT_WORKER_THREAD,
        help=(
            "Default-off C2 proposal-worker isolation. 'thread' preserves the "
            "existing rollback path; 'process' spawn-isolates C2 construction "
            "and proposal collision checks from the authoritative reference "
            "thread. Process isolation remains dry-run research only."
        ),
    )
    parser.add_argument(
        "--position-worker-cpu-reservation",
        action="store_true",
        help=(
            "Default-off process-local reservation for the authoritative reference "
            "worker, and for the C2 worker when process-isolated async_c2 is selected. "
            "After workers start, remove their CPUs and in-mask SMT siblings from "
            "every other native thread in this deployment client while leaving the "
            "reference worker pinned. This does not isolate other processes; launch "
            "the OpenPI server with matching --exclude-cpu options. Omit this option for the "
            "atomic rollback."
        ),
    )
    parser.add_argument(
        "--position-continuous-waypoint-no-debug-deadline-soak",
        action="store_true",
        help=(
            "Narrow dry-run-only exception to the async_c2 --debug requirement. "
            "It is accepted only for a bounded, auto-started, no-init-qpos "
            "deadline soak with distinct reference/C2 worker CPUs and never "
            "permits --send-actions. Omit this option for every ordinary or "
            "hardware deployment."
        ),
    )
    parser.add_argument(
        "--pose-lift-profile",
        choices=RM65_POSE_LIFT_PROFILE_CHOICES,
        default=RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
        help=(
            "Pink Pose-to-QPOS profile used only with "
            "--position-path-lift-backend pink_pose_lift. V4 is the former "
            "deployment default; V1 remains its explicit atomic rollback."
        ),
    )
    parser.add_argument(
        "--arm-control-hz",
        type=_positive_float,
        default=None,
        help=(
            "RM65 backend control/streaming rate after each action update. Position deploy "
            "defaults to 120Hz; explicit compatibility profile 'position' uses 150Hz."
        ),
    )
    parser.add_argument(
        "--position-reference-hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        help=(
            "Position-mode pose-reference/IK rate between the policy action rate and "
            "the RM65 command stream. Position deploy defaults to 60Hz; impedance mode ignores it."
        ),
    )
    parser.add_argument(
        "--position-background-target-admission",
        choices=RM65_BACKGROUND_TARGET_ADMISSION_CHOICES,
        default=RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
        help=(
            "How a new position reference is admitted while the background RM65 "
            "worker may be inside a CANFD SDK call. latest_nonblocking publishes "
            "the newest validated target under the backend state lock without "
            "waiting for that SDK call; synchronous_lock restores the former "
            "serialized behavior as an explicit rollback."
        ),
    )
    parser.add_argument(
        "--position-reference-dispatch-audit-mode",
        choices=POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES,
        default=POSITION_REFERENCE_DISPATCH_AUDIT_POST,
        help=(
            "Synchronous backend evidence collected on each authoritative 60 Hz "
            "qref. post (default) relies on fail-closed RM65 send_command admission "
            "and one dual-arm post-dispatch health snapshot. full restores the "
            "former pre/per-effector/post diagnostic audit and is the explicit "
            "rollback when diagnosing backend state transitions."
        ),
    )
    parser.add_argument(
        "--position-chunk-lookahead",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the already-predicted next action inside one policy chunk for "
            "zero-period causal 15->reference interpolation (default: enabled). "
            "Disable to restore the compatibility one-action-period delayed path."
        ),
    )
    parser.add_argument(
        "--position-plan-splice-governor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Classify policy/RTC source-plan cuts and apply robot_motion "
            "Hold-restart versus live-splice dynamic-state semantics "
            "(default: enabled). Use --no-position-plan-splice-governor "
            "together with --no-position-stream-start-acquisition for the "
            "pre-splice rollback path."
        ),
    )
    parser.add_argument(
        "--position-plan-splice-dynamics-profile",
        choices=POSITION_PLAN_SPLICE_DYNAMICS_PROFILE_CHOICES,
        default=POSITION_PLAN_SPLICE_DYNAMICS_OFF,
        help=(
            "Default-off post-Hold chunk-boundary dynamics tuning. "
            "conservative_v1 adds a bounded jerk-delta continuity cost only "
            "to the first acknowledged-Hold restart horizon; off is the exact "
            "current-behavior rollback."
        ),
    )
    parser.add_argument(
        "--position-stream-start-acquisition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Insert bounded measured-anchor SE(3) source samples before a "
            "discontinuous first policy Pose while preserving later source "
            "spacing (default: enabled for synchronous and RTC emissions). "
            "Requires --position-plan-splice-governor."
        ),
    )
    parser.add_argument(
        "--position-stream-start-max-translation-step-m",
        type=_positive_float,
        default=0.005,
        help="Maximum translation per synthetic 15 Hz stream-start source interval.",
    )
    parser.add_argument(
        "--position-stream-start-max-rotation-step-deg",
        type=_positive_float,
        default=0.5,
        help="Maximum rotation per synthetic 15 Hz stream-start source interval.",
    )
    parser.add_argument(
        "--position-stream-start-max-source-intervals",
        type=_positive_int,
        default=16,
        help=(
            "Fail-closed bound on synthetic stream-start source intervals "
            "(default: 16)."
        ),
    )
    parser.add_argument(
        "--position-collision-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run coupled dual-arm collision-aware IK on every position reference. "
            "This is mandatory with --send-actions; --no-position-collision-check "
            "is accepted only for non-motion legacy dry-run inspection."
        ),
    )
    parser.add_argument(
        "--position-collision-scene-profile",
        type=str,
        default=DEFAULT_COLLISION_SCENE,
        help="Shared RM65 scene profile used by high-rate coupled collision IK.",
    )
    parser.add_argument(
        "--position-collision-min-distance-m",
        type=_non_negative_float,
        default=DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M,
        help="Minimum configured collision distance for each accepted high-rate qpos endpoint.",
    )
    parser.add_argument(
        "--position-collision-distance-epsilon-m",
        type=_non_negative_float,
        default=DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
        help="Numeric tolerance used by the high-rate endpoint collision check.",
    )
    parser.add_argument(
        "--position-collision-debug-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable collision-solver timing profiling. Full q/pose/distance diagnostics "
            "are written by the existing --debug logger independently."
        ),
    )
    parser.add_argument(
        "--position-ik-solve-max-iterations",
        type=_positive_int,
        default=4,
        help=(
            "Position-only coupled IK iteration upper bound (formal default 4). "
            "The impedance path does not consume this option."
        ),
    )
    parser.add_argument(
        "--position-ik-solve-min-iterations",
        type=_positive_int,
        default=2,
        help=("Position-only coupled IK hard minimum iterations (default 2)."),
    )
    parser.add_argument(
        "--position-ik-damping",
        type=_positive_float,
        default=0.03,
        help="Position-only Pink QP damping (default 0.03).",
    )
    parser.add_argument(
        "--position-ik-orientation-weight",
        type=_non_negative_float,
        default=0.35,
        help="Position-only IK orientation objective weight; position weight is 1.0.",
    )
    parser.add_argument(
        "--position-ik-adaptive-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After the minimum iterations, continue only while the shared "
            "planner's frame budget has room."
        ),
    )
    parser.add_argument(
        "--position-ik-iteration-reserve-ms",
        type=_non_negative_float,
        default=5.0,
        help="Time reserved after optional IK iterations for limits/collision selection.",
    )
    parser.add_argument(
        "--arm-canfd-follow",
        dest="arm_canfd_follow",
        action="store_true",
        help="Send RM65 CANFD commands with SDK follow=True. Defaults on only for position mode.",
    )
    parser.add_argument(
        "--no-arm-canfd-follow",
        dest="arm_canfd_follow",
        action="store_false",
        help="Send RM65 CANFD commands with SDK follow=False.",
    )
    parser.set_defaults(arm_canfd_follow=None)
    parser.add_argument(
        "--arm-canfd-trajectory-mode",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="RM65 Movej_CANFD trajectory mode for position deployment.",
    )
    parser.add_argument(
        "--arm-canfd-radio",
        type=int,
        choices=tuple(range(101)),
        default=0,
        help="RM65 Movej_CANFD radio/smoothing value for position deployment.",
    )
    parser.add_argument(
        "--force-tare-sec",
        type=_positive_float,
        default=3.0,
        help="Tare duration before enabling impedance force control in deployment.",
    )
    parser.add_argument(
        "--position-force-guard",
        dest="position_force_guard",
        action="store_true",
        help=(
            "Require the latched RM65 force/torque emergency-stop monitor for "
            "real position deployment. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-position-force-guard",
        dest="position_force_guard",
        action="store_false",
        help=(
            "Disable the real-position RM65 force emergency-stop monitor. "
            "Unsafe; intended only for explicit diagnostics."
        ),
    )
    parser.set_defaults(position_force_guard=True)
    parser.add_argument(
        "--position-force-guard-source",
        choices=POSITION_FORCE_SOURCES,
        default="tool",
        help="RM65 force source used by the position emergency-stop monitor.",
    )
    parser.add_argument(
        "--position-force-guard-max-force-n",
        type=_positive_float,
        default=30.0,
        help="Position-deploy bimanual emergency-stop threshold for force-vector norm.",
    )
    parser.add_argument(
        "--position-force-guard-max-torque-nm",
        type=_positive_float,
        default=5.0,
        help="Position-deploy bimanual emergency-stop threshold for torque-vector norm.",
    )
    parser.add_argument(
        "--position-force-guard-hz",
        type=_positive_float,
        default=20.0,
        help=(
            "Per-arm RM65 force polling rate. Must be <=20 Hz because the SDK "
            "requires at least 50 ms between repeated force queries."
        ),
    )
    parser.add_argument(
        "--position-force-guard-consecutive-samples",
        type=_positive_int,
        default=1,
        help="Consecutive over-limit samples required before the latched bimanual stop.",
    )
    parser.add_argument(
        "--position-force-guard-max-read-errors",
        type=_positive_int,
        default=3,
        help="Consecutive RM65 force-read errors allowed before fail-closed bimanual stop.",
    )
    parser.add_argument(
        "--position-force-guard-stale-sec",
        type=_positive_float,
        default=0.4,
        help="Maximum valid force-sample age before fail-closed bimanual stop.",
    )
    parser.add_argument(
        "--position-force-guard-startup-timeout-sec",
        type=_positive_float,
        default=2.0,
        help=(
            "Startup deadline for one valid force sample from every active arm; "
            "failure stops both arms and aborts arming."
        ),
    )
    parser.add_argument(
        "--strict-tool-profile",
        dest="strict_tool_profile",
        action="store_true",
        default=True,
        help="Require confirmed TCP, payload, and impedance profile for impedance deployment.",
    )
    parser.add_argument(
        "--no-strict-tool-profile",
        dest="strict_tool_profile",
        action="store_false",
        help="Allow incomplete RM65 tool profiles for lab debugging.",
    )
    parser.add_argument(
        "--action-hz",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Action send/control rate. 0 keeps backward-compatible --hz unless a replay/reference "
            "dataset is provided, in which case the real dataset rate is inferred."
        ),
    )
    parser.add_argument("--action-hz-dataset", type=Path, default=None)
    parser.add_argument("--action-hz-parquet", type=Path, default=None)
    parser.add_argument("--action-hz-episode", type=_non_negative_int, default=None)
    parser.add_argument("--action-hz-start-index", type=_non_negative_int, default=None)
    parser.add_argument("--action-hz-end-index", type=int, default=None)
    parser.add_argument(
        "--policy-hz",
        type=_positive_float,
        default=None,
        help=(
            "Policy inference rate. Omit to match the effective action/control rate."
        ),
    )
    parser.add_argument(
        "--policy-inference-soft-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During a synchronous policy inference gap, freeze each RM65 impedance arm "
            "at its latest measured qpos while retaining force monitoring (default: enabled)."
        ),
    )
    parser.add_argument(
        "--deployment-full-ik-preempts-local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the historical dual-rate behavior that holds the previous safe command "
            "while the full 15 Hz IK worker is busy; default keeps bounded local IK running."
        ),
    )
    parser.add_argument("--min-action-queue", type=_non_negative_int, default=0)
    parser.add_argument(
        "--action-chunk-execution-steps",
        type=_positive_int,
        default=None,
        metavar="X",
        help=(
            "Execute only the first X actions from every received policy chunk, "
            "then request/consume the next chunk. Omit to keep the complete chunk."
        ),
    )
    parser.add_argument(
        "--action-queue-policy",
        choices=["auto", "replace", "append"],
        default="auto",
        help=(
            "How newly predicted chunks update the pending action queue. "
            "auto preserves replace for normal policies and uses append for debug replay."
        ),
    )
    parser.add_argument(
        "--append-action-chunks",
        action="store_true",
        help="Compatibility alias for --action-queue-policy append.",
    )
    parser.add_argument("--max-steps", type=_non_negative_int, default=0)
    parser.add_argument(
        "--status-sec",
        type=_positive_float,
        default=3.0,
        help=(
            "Periodic deployment status heartbeat in seconds (default: 3). "
            "Safety and lifecycle transitions are still printed immediately."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Write deployment action/pose debug JSONL to the resolved "
            "deployment debug root."
        ),
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory for --debug JSONL logs. Relative paths resolve under the DexTeleop repo.",
    )
    parser.add_argument(
        "--debug-payload-mode",
        choices=DEPLOY_DEBUG_PAYLOAD_MODES,
        default=DEPLOY_DEBUG_PAYLOAD_COMPACT,
        help=(
            "Payload detail for --debug. compact is bounded for live deployment; "
            "full restores the historical unbounded metadata payload for offline diagnosis."
        ),
    )
    parser.add_argument(
        "--artifact-layout",
        choices=(
            LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
            RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        ),
        default=configured_runtime_artifact_layout(),
        help=(
            "Select the managed --debug output layout when --debug-dir is omitted. "
            "The repository default is legacy; local_paths.json can coordinate "
            "a project-wide canonical cutover."
        ),
    )
    parser.add_argument(
        "--rtc-debug-level",
        choices=RTC_DEBUG_LEVELS,
        default=None,
        help=(
            "RTC JSONL detail when --debug and an RTC profile are enabled. "
            "Default: summary. trace records every RTC control tick."
        ),
    )
    parser.add_argument(
        "--rtc-debug-queue-capacity",
        type=_positive_int,
        default=2048,
        help=(
            "Bounded non-blocking RTC debug event queue capacity. "
            "Full queues drop diagnostic events, never control work."
        ),
    )
    parser.add_argument(
        "--ik-recovery-stable-frames",
        type=_positive_int,
        default=3,
        help=(
            "After IK smoothing/recovery clears, wait this many stable control frames, "
            "then require a fresh policy inference before accepting action chunks."
        ),
    )
    parser.add_argument("--startup-warmup-sec", type=float, default=1.0)
    parser.add_argument("--debug-replay-dataset", type=Path, default=None)
    parser.add_argument("--debug-replay-parquet", type=Path, default=None)
    parser.add_argument("--debug-replay-episode", type=_non_negative_int, default=0)
    parser.add_argument("--debug-replay-action-column", type=str, default="action")
    parser.add_argument("--debug-replay-start-index", type=_non_negative_int, default=0)
    parser.add_argument("--debug-replay-end-index", type=int, default=None)
    parser.add_argument("--debug-replay-chunk-size", type=_non_negative_int, default=32)
    parser.add_argument("--debug-replay-loop", action="store_true")
    parser.add_argument(
        "--init-time-s",
        type=float,
        default=5.0,
        help="Seconds used to smoothly move to the selected initial target before execution.",
    )
    parser.add_argument(
        "--init-arm-position-tolerance-m",
        type=_positive_float,
        default=0.03,
        help="Startup init arm position tolerance in meters before policy execution is released.",
    )
    parser.add_argument(
        "--init-arm-rotation-tolerance-deg",
        type=_positive_float,
        default=20.0,
        help="Startup init arm orientation tolerance in degrees before policy execution is released.",
    )
    parser.add_argument(
        "--init-source",
        choices=["auto", "current", "dataset", "replay", "none"],
        default="auto",
        help=(
            "Initial target source. auto uses replay first frame in replay mode, "
            "an explicit init dataset when provided, otherwise current robot state."
        ),
    )
    parser.add_argument("--init-dataset", type=Path, default=None)
    parser.add_argument("--init-parquet", type=Path, default=None)
    parser.add_argument("--init-episode", type=_non_negative_int, default=0)
    parser.add_argument("--init-frame", type=_non_negative_int, default=0)
    parser.add_argument("--init-action-column", type=str, default="action")
    parser.add_argument("--deploy-fixed-init-dataset", type=Path, default=None)
    parser.add_argument("--deploy-fixed-init-parquet", type=Path, default=None)
    parser.add_argument("--deploy-fixed-init-episode", type=str, default="0:0:end")
    parser.add_argument("--deploy-fixed-init-action-column", type=str, default="action")
    parser.add_argument(
        "--deploy-fixed-init-arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=(4.5, 4.5, 4.5, 6.0, 6.0, 8.0),
    )
    parser.add_argument(
        "--deploy-fixed-init-qpos-tolerance", type=_positive_float, default=0.1
    )
    parser.add_argument(
        "--policy-qpos-init-dataset",
        type=Path,
        default=None,
        help=(
            "Dataset root used to move to a recorded qpos frame before policy inference. "
            "This uses the same qpos target selection/planning path as replay init."
        ),
    )
    parser.add_argument("--policy-qpos-init-parquet", type=Path, default=None)
    parser.add_argument("--policy-qpos-init-episode", type=_non_negative_int, default=0)
    parser.add_argument("--policy-qpos-init-frame", type=_non_negative_int, default=0)
    parser.add_argument("--policy-qpos-init-action-column", type=str, default="action")
    parser.add_argument(
        "--policy-qpos-init-source",
        choices=QPOS_INIT_SOURCES,
        default="auto",
        help="Recorded qpos source for dataset-frame initialization.",
    )
    parser.add_argument(
        "--policy-qpos-init-timeout-s",
        type=_positive_float,
        default=policy_qpos_defaults.move_to_start_timeout_s,
    )
    parser.add_argument(
        "--policy-qpos-init-dt-sec",
        type=_positive_float,
        default=policy_qpos_defaults.qpos_init_dt_sec,
    )
    parser.add_argument(
        "--policy-qpos-init-joint-velocity-profile",
        type=str,
        default=policy_qpos_defaults.qpos_init_joint_velocity_profile,
    )
    parser.add_argument(
        "--policy-qpos-init-arm-max-angvel",
        "--qpos-init-arm-max-angvel",
        dest="policy_qpos_init_arm_max_angvel",
        type=_parse_arm_max_angvel,
        default=None,
        help=(
            "Optional first-frame QPOS init joint velocity limit in rad/s. "
            "Accepts one value or six comma-separated values; overrides "
            "--policy-qpos-init-joint-velocity-profile for qpos init planning."
        ),
    )
    parser.add_argument(
        "--policy-qpos-init-max-steps-per-side",
        type=_positive_int,
        default=policy_qpos_defaults.qpos_init_max_steps_per_side,
    )
    parser.add_argument(
        "--policy-qpos-init-collision-check",
        dest="policy_qpos_init_collision_check",
        action="store_true",
        default=policy_qpos_defaults.qpos_init_collision_check,
    )
    parser.add_argument(
        "--policy-qpos-init-no-collision-check",
        dest="policy_qpos_init_collision_check",
        action="store_false",
    )
    parser.add_argument(
        "--policy-qpos-init-collision-scene-profile",
        type=str,
        default=policy_qpos_defaults.qpos_init_collision_scene_profile,
    )
    parser.add_argument(
        "--policy-qpos-init-branch-search",
        dest="policy_qpos_init_branch_search",
        action="store_true",
        default=policy_qpos_defaults.qpos_init_branch_search,
    )
    parser.add_argument(
        "--policy-qpos-init-no-branch-search",
        dest="policy_qpos_init_branch_search",
        action="store_false",
    )
    parser.add_argument(
        "--policy-qpos-init-branch-search-max-branches",
        type=_non_negative_int,
        default=policy_qpos_defaults.qpos_init_branch_search_max_branches,
    )
    parser.add_argument(
        "--policy-qpos-init-branch-neutral-qpos-path",
        type=Path,
        default=policy_qpos_defaults.qpos_init_branch_neutral_qpos_path,
    )
    parser.add_argument(
        "--policy-qpos-init-qpos-tolerance",
        type=_positive_float,
        default=policy_qpos_defaults.qpos_reached_tolerance_rad,
    )
    parser.add_argument(
        "--policy-qpos-init-hands",
        action=argparse.BooleanOptionalAction,
        default=policy_qpos_defaults.initialize_hands,
        help=(
            "Initialize enabled end effectors from the same recorded frame after "
            "the arm qpos target is reached. Use --no-policy-qpos-init-hands as "
            "an explicit rollback to the former arm-only behavior."
        ),
    )
    parser.add_argument(
        "--policy-qpos-init-hand-tolerance",
        type=_positive_float,
        default=policy_qpos_defaults.hand_reached_tolerance,
        help=(
            "Maximum normalized measured hand-target error before startup release "
            "for end effectors whose command and measured state share one domain. "
            "DM-Tac uses its separate tolerance and measured-progress recovery."
        ),
    )
    parser.add_argument(
        "--policy-qpos-init-dmtac-tolerance",
        type=_positive_float,
        default=policy_qpos_defaults.dmtac_hand_reached_tolerance,
        help=(
            "Maximum normalized DM-Tac measured-target error before qpos-init "
            "release. Default matches the existing real-teleop warmup probe."
        ),
    )
    parser.add_argument(
        "--policy-qpos-init-hand-progress-timeout-sec",
        type=_positive_float,
        default=policy_qpos_defaults.hand_progress_timeout_sec,
        help=(
            "Maximum time without significant DM-Tac measured progress before "
            "one bounded reconnect and force-resend."
        ),
    )
    parser.add_argument(
        "--policy-qpos-init-hand-progress-epsilon",
        type=_positive_float,
        default=policy_qpos_defaults.hand_progress_epsilon,
        help="Minimum normalized DM-Tac error reduction counted as progress.",
    )
    parser.add_argument(
        "--policy-qpos-init-hand-reconnect-attempts",
        type=_non_negative_int,
        default=policy_qpos_defaults.hand_reconnect_attempts,
        help=(
            "Bounded DM-Tac reconnect/force-resend attempts during dataset-frame "
            "initialization. Zero disables automatic recovery and fails closed."
        ),
    )
    parser.add_argument(
        "--start-init_qpos",
        "--start-init-qpos",
        dest="start_init_qpos",
        action="store_true",
        help=(
            "Explicit safety override: allow real arm policy execution to start from the "
            "current qpos without a dataset-frame policy qpos init."
        ),
    )
    parser.add_argument(
        "--no-init-qpos",
        "--no--init-qpos",
        dest="no_init_qpos",
        action="store_true",
        help=(
            "Explicit safety override: suppress profile- or CLI-provided dataset-frame "
            "policy qpos initialization and allow real arm policy execution to start "
            "from the current qpos. With --init-source current, the redundant Cartesian "
            "startup motion is treated as already satisfied; manual deployments still "
            "require A ready confirmation. The legacy --no--init-qpos spelling is "
            "accepted for command-handoff compatibility."
        ),
    )
    parser.add_argument("--require-ready-observation", action="store_true")
    parser.add_argument(
        "--pause-on-policy-error",
        dest="pause_on_policy_error",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-pause-on-policy-error", dest="pause_on_policy_error", action="store_false"
    )
    parser.add_argument("--send-actions", action="store_true")
    parser.add_argument(
        "--allow-missing-v22-end-effectors",
        action="store_true",
        help=(
            "Explicit arm-only v2.2 hardware test mode: allow dataset layout end-effector "
            "slots to be omitted from runtime when --end-effector-sides none is set. "
            "No hand command is dispatched. Default remains fail-closed."
        ),
    )
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument(
        "--wait-for-ready-confirm",
        dest="wait_for_ready_confirm",
        action="store_true",
        default=None,
        help="After startup init reaches ready, wait for A/a before policy inference and action execution.",
    )
    parser.add_argument(
        "--no-wait-for-ready-confirm",
        dest="wait_for_ready_confirm",
        action="store_false",
        help="Start policy immediately after startup init completes.",
    )
    parser.add_argument("--allow-arm-connect-failure", action="store_true")
    parser.add_argument("--allow-hand-connect-failure", action="store_true")

    parser.add_argument(
        "--enable-arms", dest="arms_enabled", action="store_true", default=True
    )
    parser.add_argument("--no-arms", dest="arms_enabled", action="store_false")
    parser.add_argument("--left-arm-ip", type=str, default="")
    parser.add_argument("--right-arm-ip", type=str, default="")
    parser.add_argument(
        "--left-tool-profile",
        type=str,
        default="",
        help=(
            "Optional explicit left RM65 tool profile name or JSON path. "
            "Use position_endpoint for endpoint-TCP position control without payload/impedance metadata."
        ),
    )
    parser.add_argument(
        "--right-tool-profile",
        type=str,
        default="",
        help=(
            "Optional explicit right RM65 tool profile name or JSON path. "
            "Use position_endpoint for endpoint-TCP position control without payload/impedance metadata."
        ),
    )
    parser.add_argument("--arm-urdf-path", type=str, default="")
    parser.add_argument(
        "--rm65-ik-framework",
        choices=["robot_motion"],
        default="robot_motion",
        help=(
            "RM65 IK implementation for position arm backends. "
            "robot_motion routes shared RM65 IK construction and solving through robot_motion.rm65."
        ),
    )
    parser.add_argument(
        "--arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=DEPLOY_RM65_ARM_MAX_ANGVEL,
        help=(
            "Scalar or 6 comma-separated RM65 joint velocity limits in rad/s. "
            "Default matches run_vive_rm65_real_teleop."
        ),
    )
    parser.add_argument(
        "--position-qref-max-angvel",
        type=_parse_arm_max_angvel,
        default=None,
        help=(
            "Position-only 60 Hz accepted-qref velocity cap in rad/s. "
            "Omit to reuse --arm-max-angvel. This is independent from the "
            "120 Hz qcmd planner envelope."
        ),
    )
    parser.add_argument(
        "--position-command-planner",
        choices=POSITION_COMMAND_PLANNER_CHOICES,
        default=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        help="Position-only high-rate qcmd planner.",
    )
    parser.add_argument(
        "--position-velocity-ramp-sec",
        type=_positive_float,
        default=DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    )
    parser.add_argument(
        "--position-target-velocity-filter-tau-sec",
        type=_non_negative_float,
        default=DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
        help="Position-only target qd low-pass time constant; default 0.03 s.",
    )
    parser.add_argument(
        "--position-max-acceleration-rad-s2",
        type=_parse_arm_max_angvel,
        default=RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    )
    parser.add_argument(
        "--position-max-command-step-deg",
        type=_positive_float,
        default=RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG,
    )
    parser.add_argument(
        "--position-degraded-speed-scale",
        type=_unit_interval_float,
        default=DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    )
    parser.add_argument(
        "--arm-follow-qpos-alpha",
        type=_unit_interval_float,
        default=None,
        help="Position-only final qref-follow alpha applied by the 120 Hz backend.",
    )
    parser.add_argument(
        "--arm-step-dt-scaling",
        dest="arm_step_dt_scaling",
        action="store_true",
        default=True,
        help="Enable real-arm step scaling based on deploy action dt. Default matches real teleop.",
    )
    parser.add_argument(
        "--disable-arm-step-dt-scaling",
        dest="arm_step_dt_scaling",
        action="store_false",
        help="Disable deploy arm step scaling and use one backend control tick per policy action.",
    )
    parser.add_argument(
        "--arm-step-dt-alpha",
        type=_unit_interval_float,
        default=DEPLOY_ARM_STEP_DT_ALPHA,
        help="EMA factor for RM65 backend measured-send-dt scaling. Default matches real teleop.",
    )
    parser.add_argument(
        "--arm-step-dt-max-scale",
        type=_positive_float,
        default=None,
        help=(
            "Maximum multiplier for arm step limits when action dt is slower than backend control dt. "
            "Omit to use max(3.0, 1.1 * --arm-control-hz / effective_action_hz), matching real teleop."
        ),
    )
    parser.add_argument(
        "--arm-step-dt-stall-reset-sec",
        type=_positive_float,
        default=DEPLOY_ARM_STEP_DT_STALL_RESET_SEC,
        help="Stall threshold for RM65 backend measured-send-dt scaling. Default matches real teleop.",
    )
    parser.add_argument(
        "--singularity-qpos-init-slowdown",
        type=_unit_interval_float,
        default=DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN,
        help=(
            "Scale RM65 joint steps when first-frame QPOS init crosses singularity. "
            "Only applies to replay/deploy QPOS initialization sources."
        ),
    )
    parser.add_argument("--arm-state-refresh-hz", type=float, default=10.0)
    parser.add_argument(
        "--rm65-pose-observation",
        action="store_true",
        help=(
            "Default-off passive RM65 hardware-qpos/FK pose journal for a "
            "started position backend. This does not start Quest, perform "
            "calibration, render a trajectory, or grant control authority; "
            "--send-actions is forbidden until proof-path timing is validated."
        ),
    )
    parser.add_argument(
        "--rm65-hardware-read-retries", type=_non_negative_int, default=1
    )
    parser.add_argument(
        "--rm65-hardware-read-retry-delay-sec", type=float, default=0.02
    )
    parser.add_argument("--left-initial-qpos", type=_parse_qpos6, default=None)
    parser.add_argument("--right-initial-qpos", type=_parse_qpos6, default=None)
    parser.add_argument(
        "--arm-workspace-min",
        nargs="+",
        default=(-0.60, -0.80, 0.00),
        help="Arm workspace minimum xyz. Accepts either three values or --arm-workspace-min=x,y,z.",
    )
    parser.add_argument(
        "--arm-workspace-max",
        nargs="+",
        default=(0.60, 0.80, 0.90),
        help="Arm workspace maximum xyz. Accepts either three values or --arm-workspace-max=x,y,z.",
    )
    parser.add_argument("--max-arm-xyz-step-m", type=_non_negative_float, default=0.1)
    parser.add_argument(
        "--max-arm-rot-step-deg", type=_non_negative_float, default=15.0
    )
    parser.add_argument(
        "--max-arm-joint-step-rad",
        type=_non_negative_float,
        default=DEPLOY_MAX_ARM_JOINT_STEP_RAD,
        help=(
            "Per-action joint-vector safety limit in rad. "
            "Default is 6deg, matching real teleop temporary joint-step limit."
        ),
    )

    parser.add_argument(
        "--enable-hands", dest="hands_enabled", action="store_true", default=None
    )
    parser.add_argument("--no-hands", dest="hands_enabled", action="store_false")
    parser.add_argument(
        "--hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=HAND_BACKEND_INSPIRE,
        help="Hand backend type. Use dmtac-gripper to deploy with saved DMTAC gripper/tactile assignment.",
    )
    parser.add_argument(
        "--left-hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=None,
        help="Optional left-side override for --hand-backend.",
    )
    parser.add_argument(
        "--right-hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=None,
        help="Optional right-side override for --hand-backend.",
    )
    parser.add_argument(
        "--disconnected-gripper-sides",
        choices=("none", "left", "right", "both"),
        default="none",
        help=(
            "DMTAC side(s) intentionally kept offline: none, left, right, or both. "
            "Selected sides open no gripper/tactile transport and accept policy "
            "gripper commands only as audited no-ops."
        ),
    )
    parser.add_argument(
        "--disconnected-gripper-state",
        type=float,
        default=0.0,
        metavar="SCALAR",
        help=(
            "Fixed assumed [0,1] state for intentionally disconnected DMTAC "
            "grippers (0=open, 1=closed). This is not hardware feedback."
        ),
    )
    parser.add_argument("--left-hand-ip", type=str, default="")
    parser.add_argument("--right-hand-ip", type=str, default="")
    parser.add_argument("--hand-modbus-port", type=int, default=6000)
    parser.add_argument("--hand-send-threshold", type=float, default=0.002)
    parser.add_argument("--hand-telemetry-hz", type=_positive_float, default=20.0)
    parser.add_argument(
        "--hand-tactile-regions",
        "--tactile-regions",
        "--tactile-region",
        dest="hand_tactile_regions",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--inspire-hand-max-force",
        "--inspire-tac-max-fingertip-force",
        "--max-force",
        "--max-fingertip-force",
        "--hand-max-force",
        dest="max_fingertip_force",
        type=_non_negative_int,
        default=DEFAULT_INSPIRE_HAND_MAX_FORCE,
        help=(
            "InspireTac startup max_fingertip_force written to all six fingers. "
            f"Default: {DEFAULT_INSPIRE_HAND_MAX_FORCE}."
        ),
    )
    parser.add_argument(
        "--max-hand-step",
        type=float,
        default=0.0,
        help=(
            "Reserved/no-op for future hand q6 rate limiting. "
            "Current deploy clips hand q6 to [0, 1] but does not rate-limit it."
        ),
    )
    add_dmtac_assignment_arguments(parser, assignment_includes_tactile=True)
    parser.add_argument("--dmtac-tactile-backend", type=str, default="cpu")
    parser.add_argument("--dmtac-tactile-timeout-ms", type=_positive_int, default=None)
    parser.add_argument(
        "--dmtac-tactile-wait-for-new", action="store_true", default=None
    )
    parser.add_argument("--dmtac-tactile-sequential-read", action="store_true")
    parser.add_argument(
        "--dmtac-tactile-fields",
        type=_parse_dmtac_tactile_fields,
        default=None,
        metavar="FIELDS",
        help=(
            "DM-Tac fields to read/store. Default is raw-image-only collection; pass "
            "'distributed_force' for real-time force output. Valid fields: "
            + ", ".join(DMTAC_TACTILE_FIELDS)
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-compute-profile",
        choices=("dist-only", "legacy", "requested", "shear-depth"),
        default=None,
    )
    parser.add_argument(
        "--dmtac-with-tactile", dest="dmtac_tactile", action="store_true", default=None
    )
    parser.add_argument(
        "--dmtac-no-tactile", dest="dmtac_tactile", action="store_false"
    )
    parser.add_argument(
        "--gripper-task-profile",
        type=str,
        default="",
        help="JSON profile with top-level gripper defaults and/or tasks.<name>.gripper overrides.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="",
        help="Task id used to select gripper task profile entries.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="",
        help="Task name used to select gripper task profile entries.",
    )
    parser.add_argument(
        "--gripper-task",
        type=str,
        default="",
        help="Explicit gripper task key; overrides --task-id/--task-name for --gripper-task-profile.",
    )
    parser.add_argument("--gripper-output-dim", type=int, default=None)
    parser.add_argument("--gripper-speed", type=float, default=None)
    parser.add_argument(
        "--gripper-torque",
        "--gripper-force",
        dest="gripper_torque",
        type=float,
        default=None,
    )
    parser.add_argument("--gripper-max-action", type=float, default=None)
    parser.add_argument("--gripper-send-threshold", type=float, default=None)
    parser.add_argument(
        "--gripper-continuous",
        dest="gripper_continuous_grasp",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--gripper-no-continuous", dest="gripper_continuous_grasp", action="store_false"
    )
    parser.add_argument("--gripper-command-hz", type=_positive_float, default=None)
    parser.add_argument("--gripper-telemetry-hz", type=_positive_float, default=None)
    parser.add_argument("--gripper-telemetry-command-grace-s", type=float, default=None)

    parser.add_argument(
        "--camera-mode", choices=["none", "single", "multi"], default="none"
    )
    parser.add_argument(
        "--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG_PATH
    )
    parser.add_argument("--camera-key", type=str, default="front")
    parser.add_argument(
        "--camera-serial", "--camera-id", dest="camera_serial", type=str, default=""
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-keep-last",
        type=int,
        default=DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST,
        help=(
            "Minimum causal frame history retained by deployment camera sources. "
            "The value also applies when cameras are loaded from --camera-config."
        ),
    )
    parser.add_argument("--camera-color-space", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--camera-wait-timeout-ms", type=int, default=1000)
    parser.add_argument("--camera-stale-after-sec", type=float, default=2.0)
    parser.add_argument(
        "--camera-frame-max-age-sec",
        type=_positive_float,
        default=DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC,
        help=(
            "Maximum causal camera source age accepted by the deployment observation "
            f"firewall (default: {DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC:.3f}s)."
        ),
    )
    parser.add_argument("--camera-auto-exposure-priority", type=int, default=None)
    parser.add_argument(
        "--camera-color-format",
        choices=("bgr8", "rgb8", "yuyv", "uyvy", "y8", "rgba8", "bgra8"),
        default="bgr8",
        help=(
            "Native RealSense color transport. The default matches formal "
            "real-teleop collection startup."
        ),
    )
    parser.add_argument(
        "--camera-startup-warmup-policy",
        choices=REALSENSE_STARTUP_WARMUP_POLICIES,
        default=REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
        help=(
            "RealSense warmup/recovery profile shared with formal collection. "
            "bounded_first_frame requires complete configured streams and bounded "
            "recovery; legacy preserves the historical rollback behavior."
        ),
    )
    parser.add_argument(
        "--camera-startup-hardware-reset",
        action="store_true",
        help=(
            "Reset each selected RealSense by serial before opening its stream. "
            "Default-off so existing deployment profiles and camera configs are unchanged."
        ),
    )
    parser.add_argument(
        "--camera-start-stagger-sec",
        type=float,
        default=REAL_TELEOP_CAMERA_START_STAGGER_SEC,
        help=(
            "Delay between sequential RealSense starts. The default matches formal "
            "real-teleop collection; metadata.startup_priority owns hardware order."
        ),
    )
    parser.add_argument("--require-camera-frame", action="store_true")
    parser.add_argument("--vis_cam", "--vis-cam", dest="vis_cam", action="store_true")
    parser.add_argument("--vis-cam-key", "--vis-camera-key", type=str, default="")
    parser.add_argument("--visualize-tactile", action="store_true")
    parser.add_argument("--tactile-vis-value-scale", type=float, default=0.05)
    parser.add_argument("--tactile-vis-size", type=int, default=640)


__all__ = ["add_args"]
