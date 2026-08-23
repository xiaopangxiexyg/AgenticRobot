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
from robot_motion.deploy.prediction_worker import ThreadedPolicyPredictionWorker
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
from .resource_lifecycle import PolicyDeployResourceLifecycle
from .rm65_legacy_worker_binding import RM65LegacyWorkerExecutionBinding
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
        "--policy-prediction-mode",
        choices=("synchronous", "threaded-prefetch"),
        default="synchronous",
        help=(
            "Prediction scheduling for the legacy action queue. synchronous "
            "preserves the compatibility path; threaded-prefetch runs only "
            "model inference on one bounded worker while the control thread "
            "retains queue admission, safety and dispatch authority."
        ),
    )
    parser.add_argument(
        "--policy-prediction-prefetch-queue-threshold",
        type=_non_negative_int,
        default=4,
        metavar="ACTIONS",
        help=(
            "Submit the next threaded prediction when at most this many "
            "admitted actions remain (default: 4)."
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
    threaded_prediction = str(args.policy_prediction_mode) == "threaded-prefetch"
    if threaded_prediction and rtc_profile_enabled:
        raise ValueError(
            "--policy-prediction-mode threaded-prefetch is only valid for the "
            "legacy non-RTC action queue."
        )
    if threaded_prediction and (
        args.debug_replay_dataset is not None or args.debug_replay_parquet is not None
    ):
        raise ValueError(
            "dataset replay owns prediction transactions and must use "
            "--policy-prediction-mode synchronous."
        )
    if (
        threaded_prediction
        and args.action_chunk_execution_steps is not None
        and int(args.policy_prediction_prefetch_queue_threshold)
        >= int(args.action_chunk_execution_steps)
    ):
        raise ValueError(
            "--policy-prediction-prefetch-queue-threshold must be smaller "
            "than --action-chunk-execution-steps."
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
    continuous_worker_isolation = (
        str(args.position_continuous_waypoint_worker_isolation).strip().lower()
    )
    if continuous_worker_isolation != CONTINUOUS_WAYPOINT_WORKER_THREAD and not bool(
        args.position_continuous_waypoint_async_precompute
    ):
        raise ValueError(
            "--position-continuous-waypoint-worker-isolation process requires "
            "async_c2 precompute"
        )
    reference_worker_cpu = args.position_reference_worker_cpu
    continuous_worker_cpu = args.position_continuous_waypoint_worker_cpu
    if continuous_worker_cpu is not None and reference_worker_cpu is None:
        raise ValueError(
            "--position-continuous-waypoint-worker-cpu requires "
            "--position-reference-worker-cpu"
        )
    if bool(args.position_continuous_waypoint_async_precompute) and (
        (reference_worker_cpu is None) != (continuous_worker_cpu is None)
    ):
        raise ValueError(
            "async_c2 CPU isolation requires both "
            "--position-reference-worker-cpu and "
            "--position-continuous-waypoint-worker-cpu"
        )
    args.position_worker_cpu_affinity_allowed = None
    if reference_worker_cpu is not None:
        if continuous_worker_cpu is not None and not bool(
            args.position_continuous_waypoint_async_precompute
        ):
            raise ValueError(
                "--position-continuous-waypoint-worker-cpu requires async_c2 "
                "precompute"
            )
        if continuous_worker_cpu is not None and int(reference_worker_cpu) == int(
            continuous_worker_cpu
        ):
            raise ValueError(
                "position reference and continuous C2 workers require distinct CPUs"
            )
        allowed_cpus = available_cpu_ids()
        missing_cpus = sorted(
            (
                {int(reference_worker_cpu)}
                if continuous_worker_cpu is None
                else {
                    int(reference_worker_cpu),
                    int(continuous_worker_cpu),
                }
            ).difference(allowed_cpus)
        )
        if missing_cpus:
            raise ValueError(
                "requested position worker CPUs are outside this process affinity "
                f"mask: missing={missing_cpus} allowed={list(allowed_cpus)}"
            )
        args.position_worker_cpu_affinity_allowed = allowed_cpus
    args.position_worker_cpu_reservation_cpu_ids = ()
    if bool(args.position_worker_cpu_reservation):
        reservation_violations: list[str] = []
        if reference_worker_cpu is None:
            reservation_violations.append("--position-reference-worker-cpu is required")
        if bool(args.position_continuous_waypoint_async_precompute):
            if continuous_worker_cpu is None:
                reservation_violations.append(
                    "--position-continuous-waypoint-worker-cpu is required"
                )
            if continuous_worker_isolation != CONTINUOUS_WAYPOINT_WORKER_PROCESS:
                reservation_violations.append(
                    "--position-continuous-waypoint-worker-isolation process is required"
                )
        if reservation_violations:
            raise ValueError(
                "--position-worker-cpu-reservation requirements are not met: "
                + "; ".join(reservation_violations)
            )
        assert args.position_worker_cpu_affinity_allowed is not None
        reservation_cpu_ids: set[int] = set()
        for worker_cpu in (reference_worker_cpu, continuous_worker_cpu):
            if worker_cpu is None:
                continue
            reservation_cpu_ids.update(
                int(cpu)
                for cpu in cpu_thread_sibling_ids(int(worker_cpu))
                if int(cpu) in args.position_worker_cpu_affinity_allowed
            )
        args.position_worker_cpu_reservation_cpu_ids = tuple(
            sorted(reservation_cpu_ids)
        )
    if bool(args.send_actions) and bool(
        args.position_continuous_waypoint_experimental_admission
    ):
        raise ValueError(
            "continuous C2 waypoint admission is not authorized for real "
            "hardware dispatch: the 2026-08-08 full-runtime trace exceeded "
            "the authoritative 60 Hz deadline. Use formal_third_order for "
            "--send-actions; re-authorize C2 only after isolated real-executor "
            "deadline soak passes."
        )
    observation_firewall_mode = (
        str(getattr(args, "policy_observation_firewall_mode", "strict")).strip().lower()
    )
    rtc_real_shadow = bool(getattr(args, "rtc_real_shadow", False))
    requested_rtc_debug_level = getattr(args, "rtc_debug_level", None)
    if requested_rtc_debug_level is None:
        rtc_debug_level = (
            RTC_DEBUG_LEVEL_SUMMARY
            if bool(getattr(args, "debug", False)) and rtc_profile_enabled
            else RTC_DEBUG_LEVEL_OFF
        )
    else:
        rtc_debug_level = normalize_rtc_debug_level(requested_rtc_debug_level)
    if rtc_debug_level != RTC_DEBUG_LEVEL_OFF and not bool(
        getattr(args, "debug", False)
    ):
        raise ValueError("--rtc-debug-level summary/trace requires --debug.")
    if rtc_debug_level != RTC_DEBUG_LEVEL_OFF and not rtc_profile_enabled:
        raise ValueError("--rtc-debug-level summary/trace requires an RTC profile.")
    args._rtc_debug_level = rtc_debug_level
    model_transport_profiles = {
        RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
        MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    }
    if rtc_real_shadow and rtc_profile_id not in model_transport_profiles:
        raise ValueError(
            "--rtc-real-shadow requires an explicit model-backed RTC profile; "
            "it is unavailable for RTC-3c virtual policy or sync deployment."
        )
    if rtc_profile_enabled:
        if rtc_profile_id not in {
            RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
            MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
        }:
            raise ValueError(f"Unsupported RTC profile: {rtc_profile_id!r}.")
        if bool(args.send_actions):
            raise ValueError("RTC shadow profile forbids --send-actions.")
        if args.debug_replay_dataset or args.debug_replay_parquet:
            raise ValueError("RTC profiles do not modify or consume debug replay.")
        if args.action_hz_dataset or args.action_hz_parquet:
            raise ValueError(
                "RTC profiles do not consume replay/reference rate inputs."
            )
        if bool(args.append_action_chunks) or str(args.action_queue_policy) == "append":
            raise ValueError(
                "RTC profiles use timed mutable-future merge, not queue append."
            )
        if rtc_profile_id == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1:
            if str(args.policy_spec or "hold").strip().lower() not in {
                "",
                "hold",
                "builtin:hold",
            }:
                raise ValueError(
                    "RTC-3c is restricted to the builtin hold virtual policy."
                )
            if args.policy_artifact_json is not None:
                raise ValueError(
                    "RTC-3c virtual shadow does not bind a model artifact yet."
                )
            if parse_policy_kwargs(args.policy_kwargs_json):
                raise ValueError("RTC-3c builtin hold policy accepts no policy kwargs.")
        elif args.policy_artifact_json is None:
            raise ValueError(
                "Model-backed RTC requires an explicit --policy-artifact-json "
                "sidecar; model identity must not be inferred from a model name."
            )
        if (
            rtc_profile_id in model_transport_profiles
            and not rtc_real_shadow
            and (
                bool(args.left_arm_ip)
                or bool(args.right_arm_ip)
                or bool(args.left_hand_ip)
                or bool(args.right_hand_ip)
                or (
                    bool(getattr(args, "hands_enabled", False))
                    and any(
                        _hand_backend_mode_for_side(args, side)
                        == HAND_BACKEND_DMTAC_GRIPPER
                        for side in ALL_SIDES
                    )
                )
            )
        ):
            raise ValueError(
                "Model-backed RTC with live hardware endpoints requires explicit "
                "--rtc-real-shadow; otherwise use synthetic/local inputs."
            )
        if float(args.action_hz) not in {
            0.0,
            RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.action_hz,
        }:
            raise ValueError("RTC shadow action rate is fixed at 30 Hz.")
        if args.policy_hz is not None and not math.isclose(
            float(args.policy_hz),
            RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.decision_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("RTC shadow decision rate is fixed at 30 Hz.")
        args.position_chunk_lookahead = False
        args.action_hz = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.action_hz
        args.policy_hz = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.decision_hz
    elif (
        int(getattr(args, "rtc_request_validity_steps", 10)) != 10
        or float(getattr(args, "rtc_worker_stop_timeout_sec", 0.0)) != 0.0
    ):
        raise ValueError(
            "RTC validity/worker-stop overrides require an explicit --rtc-profile."
        )
    if rtc_real_shadow:
        # A live observation shadow has no init/qpos authority.  Keeping a
        # dataset-frame init controller alive would either block the model on a
        # target it cannot execute or accidentally reintroduce a dispatch path
        # if a future caller forgets to pass send_actions=False.
        if args.deploy_fixed_init_dataset or args.deploy_fixed_init_parquet:
            raise ValueError(
                "--rtc-real-shadow does not accept deploy-fixed-init dataset inputs."
            )
        if args.policy_qpos_init_dataset or args.policy_qpos_init_parquet:
            raise ValueError(
                "--rtc-real-shadow does not accept policy-qpos-init dataset inputs."
            )
        if bool(args.send_actions):
            raise ValueError("--rtc-real-shadow forbids --send-actions.")
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
    if int(args.debug_replay_chunk_size) <= 0:
        raise ValueError("--debug-replay-chunk-size must be positive.")
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
    if args.debug_replay_end_index is not None and int(
        args.debug_replay_end_index
    ) <= int(args.debug_replay_start_index):
        raise ValueError(
            "--debug-replay-end-index must be greater than --debug-replay-start-index."
        )
    action_hz_start = (
        args.debug_replay_start_index
        if args.action_hz_start_index is None
        else args.action_hz_start_index
    )
    action_hz_end = (
        args.debug_replay_end_index
        if args.action_hz_end_index is None
        else args.action_hz_end_index
    )
    if action_hz_end is not None and int(action_hz_end) <= int(action_hz_start):
        raise ValueError(
            "--action-hz-end-index must be greater than --action-hz-start-index."
        )
    if args.debug_replay_dataset and args.debug_replay_parquet:
        raise ValueError(
            "Use only one of --debug-replay-dataset or --debug-replay-parquet."
        )
    if args.action_hz_dataset and args.action_hz_parquet:
        raise ValueError("Use only one of --action-hz-dataset or --action-hz-parquet.")
    if (args.debug_replay_dataset or args.debug_replay_parquet) and str(
        args.policy_spec or "hold"
    ).strip().lower() not in {"", "hold", "builtin:hold"}:
        raise ValueError(
            "--debug-replay-* replaces --policy; omit --policy in replay mode."
        )
    parse_tactile_regions(args.hand_tactile_regions)
    if bool(args.send_actions) and bool(args.arms_enabled):
        for side in active_sides:
            if not getattr(args, f"{side}_arm_ip"):
                raise ValueError(
                    f"--send-actions requires --{side}-arm-ip for active side '{side}'."
                )
    if bool(args.send_actions) and hands_enabled:
        for side in end_effector_sides:
            if (
                _hand_backend_mode_for_side(args, side) == HAND_BACKEND_INSPIRE
                and not getattr(args, f"{side}_hand_ip")
            ):
                raise ValueError(
                    f"--send-actions with hands requires --{side}-hand-ip."
                )
    if bool(args.visualize_tactile):
        if any(
            _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
            for side in end_effector_sides
        ):
            raise ValueError(
                "--visualize-tactile currently supports InspireTac UV tactile, not DMTAC gripper tactile."
            )
        if not hands_enabled:
            raise ValueError("--visualize-tactile requires hands to be enabled.")
        for side in end_effector_sides:
            if not getattr(args, f"{side}_hand_ip"):
                raise ValueError(f"--visualize-tactile requires --{side}-hand-ip.")
    if args.camera_mode in {"single", "multi"}:
        if args.camera_mode == "multi" and args.camera_serial:
            raise ValueError(
                "--camera-mode multi uses --camera-config; omit --camera-serial."
            )
        if not args.camera_serial and not Path(args.camera_config).exists():
            raise ValueError(
                f"--camera-mode {args.camera_mode} requires an existing --camera-config."
            )
        if int(args.camera_width) <= 0 or int(args.camera_height) <= 0:
            raise ValueError("--camera-width/--camera-height must be positive.")
        if int(args.camera_fps) <= 0:
            raise ValueError("--camera-fps must be positive.")
        if int(args.camera_keep_last) <= 0:
            raise ValueError("--camera-keep-last must be positive.")
        if float(args.camera_frame_max_age_sec) <= 0.0:
            raise ValueError("--camera-frame-max-age-sec must be positive.")
        if int(args.camera_wait_timeout_ms) <= 0:
            raise ValueError("--camera-wait-timeout-ms must be positive.")
        if float(args.camera_stale_after_sec) <= 0.0:
            raise ValueError("--camera-stale-after-sec must be positive.")
        if float(args.camera_start_stagger_sec) < 0.0:
            raise ValueError("--camera-start-stagger-sec must be >= 0.")
    elif bool(args.vis_cam):
        raise ValueError(
            "--vis_cam requires --camera-mode single or --camera-mode multi."
        )
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


def _send_legacy_deployment_action_commands(
    commands: Mapping[str, Any],
    *,
    arm_backends: Mapping[str, EffectorBackend],
    hand_backends: Mapping[str, EffectorBackend],
) -> None:
    """Archived direct DeploymentAction command dispatcher.

    The active deploy policy path now resolves NormalizedAction through
    RobotAction/TeleopCommand in teleop_runtime.deploy.canonical_action.
    """

    for side in ALL_SIDES:
        arm_command = commands.get(f"{side}_arm")
        arm_backend = arm_backends.get(side)
        if arm_command is not None and arm_backend is not None:
            arm_backend.send_command(arm_command)
        hand_command = commands.get(f"{side}_hand")
        hand_backend = hand_backends.get(side)
        if hand_command is not None and hand_backend is not None:
            hand_backend.send_command(hand_command)


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
    now: float,
) -> None:
    """Print a high-salience boundary between initialization and policy motion."""

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


def main(
    argv: list[str] | None = None,
    *,
    rtc_step_fanout: RtcFrontendStepFanout | None = None,
    rtc_emission_fanout: PolicyActionEmissionFanout | None = None,
    rtc_emission_publisher: RtcPolicyEmissionPublisher | None = None,
    rtc_model_adapter_factory: Callable[..., Any] | None = None,
    policy_emission_observer: PassivePolicyEmissionObserverPort | None = None,
    legacy_worker_binding: RM65LegacyWorkerExecutionBinding | None = None,
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
    policy_prediction_worker = (
        ThreadedPolicyPredictionWorker(
            source_id="legacy-openpi-prefetch",
            runtime=policy_deploy_runtime,
            policy=policy,
            lifecycle_policy=getattr(policy, "policy", policy),
            transaction_policy=getattr(policy, "policy", policy),
        )
        if str(args.policy_prediction_mode) == "threaded-prefetch"
        else None
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
    if deploy_session_contract.policy_contract_mode == POLICY_CONTRACT_V22_NATIVE:
        raise RuntimeError(
            "v22_native typed RobotAction transport is available, but this run_policy loop "
            "does not yet have a configured native end-effector safety/dispatch adapter. "
            "Use an explicit legacy_q6_slot_indices bridge for this runtime."
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
    arm_reader = ArmStateReader(
        arm_backends,
        refresh_hz=float(args.arm_state_refresh_hz),
        compute_tcp_pose=bool(args.arms_enabled),
    )
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
            preparer=canonical_action_pipeline,
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

    prepared_action_executor = PreparedActionExecutor(
        position_deploy_executor=position_deploy_executor,
        send_actions=send_actions,
        arm_backends=arm_backends,
        hand_backends=hand_backends,
        active_sides=active_sides,
        position_worker_binding=position_worker_binding,
        submit_legacy_owner_state=submit_legacy_owner_state,
    )

    def reset_policy_for_activation() -> None:
        """Reset without racing an in-flight threaded model call."""

        if policy_prediction_worker is None:
            policy.reset()
        else:
            policy_prediction_worker.reset_for_activation()

    def invalidate_in_flight_policy_prediction() -> None:
        if (
            policy_prediction_worker is not None
            and policy_prediction_worker.in_flight
            and not policy_prediction_worker.reset_pending
        ):
            policy_prediction_worker.reset_for_activation()

    def record_policy_prediction_result(
        prediction: PolicyChunkPredictionResult,
        *,
        queue_policy: str,
        synchronous: bool,
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
                    synchronous
                    and _defer_position_action_after_synchronous_prediction(
                        prediction_accepted=True,
                        position_executor_enabled=(
                            position_deploy_executor is not None
                        ),
                    )
                )
            elif prediction.discarded_by_gate:
                print(
                    "[Deploy] IK recovery active; discarded "
                    + ("predicted" if synchronous else "prefetched")
                    + " action chunk."
                )
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
                if synchronous and defer_until_fresh_cut:
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
                elif not synchronous:
                    debug_logger.log_event(
                        "policy_prediction_prefetch_admitted",
                        {
                            "schema": "dexteleop_policy_prediction_prefetch_v1",
                            "queue_policy": queue_policy,
                            "queue_len_before": int(queue_len_before),
                            "queue_len_after": int(prediction.queue_len_after),
                            "inference_duration_sec": (
                                None
                                if prediction.chunk_lineage is None
                                else max(
                                    0.0,
                                    prediction.chunk_lineage.inference_completed_at
                                    - prediction.chunk_lineage.inference_started_at,
                                )
                            ),
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
                "[Deploy] policy "
                + ("error: " if synchronous else "prefetch error: ")
                + f"{type(exc).__name__}: {exc}"
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
        policy_prediction_worker=policy_prediction_worker,
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
            f"prediction_mode={args.policy_prediction_mode} "
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
        print_controls(
            send_actions=send_actions,
            auto_start=bool(args.auto_start),
            wait_for_ready_confirm=wait_for_ready_confirm,
            fixed_init_enabled=fixed_init_controller is not None,
            policy_qpos_init_enabled=policy_qpos_init_controller is not None,
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
                if not policy_active:
                    invalidate_in_flight_policy_prediction()
                    if (
                        policy_prediction_worker is not None
                        and policy_prediction_worker.reset_pending
                    ):
                        policy_prediction_worker.poll_completion()
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
                                            position_deploy_executor is not None
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
                        and can_run_policy
                        and policy_prediction_worker is not None
                    ):
                        prediction_candidate = (
                            policy_prediction_worker.poll_completion()
                        )
                        if prediction_candidate is not None:
                            prediction_queue_policy = (
                                "append"
                                if policy_deploy_runtime.action_queue_length() > 0
                                else "replace"
                            )
                            prediction = (
                                policy_deploy_runtime.admit_prediction_result(
                                    prediction_candidate,
                                    now=control_loop_started_at,
                                    queue_policy=prediction_queue_policy,
                                    discard_policy_output=(
                                        ik_action_gate.discards_policy_output
                                    ),
                                    action_keyframe_period_sec=(
                                        1.0 / float(effective_action_hz)
                                    ),
                                )
                            )
                            (
                                prefetch_defer,
                                prefetch_pause,
                            ) = record_policy_prediction_result(
                                prediction,
                                queue_policy=prediction_queue_policy,
                                synchronous=False,
                                faults=faults,
                                timestamp=loop_started_at,
                                step_index=step_index,
                                observation_step_index=observation.step_index,
                            )
                            defer_position_action_until_fresh_cut = (
                                defer_position_action_until_fresh_cut
                                or prefetch_defer
                            )
                            if prefetch_pause:
                                policy_active = False
                                can_run_policy = False
                                if position_deploy_executor is not None:
                                    position_deploy_executor.clear()
                                    last_action = observation.hold_action()
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
                            + (
                                "Hold because threaded prefetch missed the chunk boundary."
                                if policy_prediction_worker is not None
                                else "Hold before synchronous inference."
                            ),
                            now=loop_started_at,
                        )
                        debug_logger.log_event(
                            "position_chunk_gap_hold",
                            {
                                "schema": "dexteleop_position_chunk_gap_hold_v1",
                                "reason": (
                                    "threaded_prefetch_missed_empty_action_queue"
                                    if policy_prediction_worker is not None
                                    else "synchronous_inference_with_empty_action_queue"
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
                        and policy_prediction_worker is not None
                        and not policy_prediction_worker.in_flight
                        and not policy_prediction_worker.reset_pending
                        and policy_deploy_runtime.should_predict(
                            now=control_loop_started_at,
                            policy_interval_s=1.0 / float(effective_policy_hz),
                            min_action_queue=int(
                                args.policy_prediction_prefetch_queue_threshold
                            ),
                        )
                    ):
                        inference_id = (
                            policy_deploy_runtime.reserve_prediction_inference_id()
                        )
                        if policy_prediction_worker.submit(
                            observation,
                            inference_id=inference_id,
                        ):
                            debug_logger.log_event(
                                "policy_prediction_prefetch_submitted",
                                {
                                    "schema": (
                                        "dexteleop_policy_prediction_prefetch_v1"
                                    ),
                                    "inference_id": inference_id,
                                    "queue_len": int(
                                        policy_deploy_runtime.action_queue_length()
                                    ),
                                    "threshold": int(
                                        args.policy_prediction_prefetch_queue_threshold
                                    ),
                                    "observation_step_index": int(
                                        observation.step_index
                                    ),
                                },
                            )
                    if (
                        rtc_frontend is None
                        and can_run_policy
                        and policy_prediction_worker is None
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
                            synchronous=True,
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
                    allow_policy_output = (
                        can_run_policy
                        and not ik_action_gate.blocks_action_output
                        and not defer_position_action_until_fresh_cut
                        and not collision_reset_waiting
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
                            preparer=canonical_action_pipeline,
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
                        invalidate_in_flight_policy_prediction()
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
                    if raw_action is not None:
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
                            not debug_replay
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
                        lookahead_canonical_action = None
                        lookahead_observation = None
                        second_lookahead_canonical_action = None
                        second_lookahead_observation = None
                        third_lookahead_canonical_action = None
                        third_lookahead_observation = None
                        if (
                            position_deploy_executor is not None
                            and bool(args.position_chunk_lookahead)
                            and raw_lookahead_action is not None
                            and not canonical_action.faults
                        ):
                            # The current safe action is the correct 15 Hz
                            # reference for filtering the next action in the
                            # same already-decoded chunk.  This is validation,
                            # not an extra temporal smoothing stage.
                            lookahead_observation = replace(
                                observation,
                                last_action=canonical_action.action.vector.astype(
                                    np.float32,
                                    copy=True,
                                ),
                            )
                            lookahead_prepare_started = time.monotonic()
                            prepared_lookahead = canonical_action_pipeline.prepare(
                                raw_lookahead_action,
                                lookahead_observation,
                                policy_name=policy.name,
                            )
                            action_phase_timing["lookahead_prepare_1_ms"] = max(
                                0.0,
                                time.monotonic() - lookahead_prepare_started,
                            ) * 1000.0
                            if not prepared_lookahead.faults:
                                lookahead_canonical_action = prepared_lookahead
                                if (
                                    raw_second_lookahead_action is not None
                                    and action_step.next_step_execution_signal
                                    is not None
                                ):
                                    second_lookahead_observation = replace(
                                        observation,
                                        last_action=(
                                            prepared_lookahead.action.vector.astype(
                                                np.float32,
                                                copy=True,
                                            )
                                        ),
                                    )
                                    lookahead_prepare_started = time.monotonic()
                                    prepared_second_lookahead = (
                                        canonical_action_pipeline.prepare(
                                            raw_second_lookahead_action,
                                            second_lookahead_observation,
                                            policy_name=policy.name,
                                        )
                                    )
                                    action_phase_timing["lookahead_prepare_2_ms"] = max(
                                        0.0,
                                        time.monotonic() - lookahead_prepare_started,
                                    ) * 1000.0
                                    if not prepared_second_lookahead.faults:
                                        second_lookahead_canonical_action = (
                                            prepared_second_lookahead
                                        )
                                        if (
                                            bool(
                                                args.position_continuous_waypoint_async_precompute
                                            )
                                            and raw_third_lookahead_action is not None
                                        ):
                                            third_lookahead_observation = replace(
                                                observation,
                                                last_action=(
                                                    prepared_second_lookahead.action.vector.astype(
                                                        np.float32,
                                                        copy=True,
                                                    )
                                                ),
                                            )
                                            lookahead_prepare_started = time.monotonic()
                                            prepared_third_lookahead = (
                                                canonical_action_pipeline.prepare(
                                                    raw_third_lookahead_action,
                                                    third_lookahead_observation,
                                                    policy_name=policy.name,
                                                )
                                            )
                                            action_phase_timing[
                                                "lookahead_prepare_3_ms"
                                            ] = max(
                                                0.0,
                                                time.monotonic()
                                                - lookahead_prepare_started,
                                            ) * 1000.0
                                            if not prepared_third_lookahead.faults:
                                                third_lookahead_canonical_action = (
                                                    prepared_third_lookahead
                                                )
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
                                    lookahead_canonical_action=(
                                        lookahead_canonical_action
                                    ),
                                    lookahead_observation=lookahead_observation,
                                    second_lookahead_canonical_action=(
                                        second_lookahead_canonical_action
                                    ),
                                    second_lookahead_observation=(
                                        second_lookahead_observation
                                    ),
                                    third_lookahead_canonical_action=(
                                        third_lookahead_canonical_action
                                    ),
                                    third_lookahead_observation=(
                                        third_lookahead_observation
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
                                    position_action_submitted_this_loop = (
                                        position_deploy_executor is not None
                                    )
                                    last_action = canonical_action.action.vector.astype(
                                        np.float32, copy=True
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
                                f"{rtc_frontend.selected_emission_count}"
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
                if int(args.max_steps) > 0 and step_index >= int(args.max_steps):
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
        cleanup_errors = resource_lifecycle.close(
            keyboard_interrupt_stop_error=keyboard_interrupt_stop_error,
        )
    return 1 if cleanup_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
