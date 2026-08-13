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
import select
import sys
import threading
import termios
import time
import tty
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..configs.camera_paths import DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
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
from project_paths import repo_path
from runtime_artifact_paths import (
    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    ArtifactWriteResolution,
    RuntimeArtifactLayout,
    configured_runtime_artifact_layout,
    resolve_artifact_write_path,
)
from robot_motion.rm65.assets import require_asset_path
from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    ArmActionSpec,
    HandActionSpec,
    build_robot_action_v1_runtime_schema,
    hand_action_specs_metadata,
)
from robot_motion.deploy.actions import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    NormalizedAction,
)
from robot_motion.deploy.canonical_action import CanonicalDeployActionPipeline
from robot_motion.deploy.async_producer import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1,
    POLICY_ASYNC_PRODUCER_ELIGIBLE,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_V1,
)
from robot_motion.deploy.observation import (
    DEFAULT_HAND_STATE_MAX_AGE_SEC,
    build_deployment_observation,
)
from robot_motion.deploy.observation_view import (
    default_deploy_policy_observation_contract,
)
from robot_motion.deploy.next_step import sync_queue_preserves_same_chunk_successor
from robot_motion.deploy.pose_adapter import (
    convert_policy_absolute_vector_to_runtime_raw,
)
from robot_motion.deploy.runtime import (
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    resolve_action_queue_policy,
)
from robot_motion.deploy.emission_port import (
    PassivePolicyEmissionObserverPort,
    require_passive_policy_emission_observer,
)
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    DEPLOY_ACTION_VECTOR_ACTION_SPACE,
    pose_matrix_to_xyzquat,
    to_serializable,
)
from robot_motion.deploy.session_contract import (
    POLICY_CONTRACT_V22_NATIVE,
    RuntimeEndEffectorCapability,
    resolve_deploy_session_contract,
    resolve_policy_action_pose_mode_from_artifact,
)
from robot_motion.replay.trajectory import QPOS_INIT_SOURCES
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
    DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M,
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    LEGACY_POSITION_CONTROL_PROFILE_NAME,
    RM65_BIMANUAL_POSITION_V1,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
    RM65_POSE_LIFT_PROFILE_CHOICES,
    RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
    RM65_POSITION_TRACKING_FOLLOW_ALPHA,
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
    RM65_QREF_ACCELERATION_COMFORT_75_V1,
    RM65_QREF_ACCELERATION_HARD_V1,
    RM65_QREF_ACCELERATION_PROFILE_CHOICES,
    build_rm65_pose_lift_ik_tuning,
    resolve_rm65_pose_lift_profile,
    resolve_rm65_qref_acceleration_profile,
)
from robot_motion.position_control import (
    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    POSITION_PLAN_SPLICE_DYNAMICS_OFF,
    POSITION_PLAN_SPLICE_DYNAMICS_PROFILE_CHOICES,
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_CHOICES,
    PositionSourcePlanCut,
    PositionStreamStartAcquisitionConfig,
    resolve_position_plan_splice_dynamics_config,
)
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract,
)
from robot_motion.runtime import (
    TeleopCommand,
    available_cpu_ids,
    cpu_thread_sibling_ids,
    exclude_cpu_ids_from_current_process,
)
from robot_motion.runtime.causal_history import CausalSnapshotHistory
from robot_motion.runtime_gate import IKActionChunkGate

from ..backends.arms import (
    DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN,
    RM65ArmBackend,
    RM65ArmBackendConfig,
    RM65_BACKGROUND_TARGET_ADMISSION_CHOICES,
    RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    RM65ImpedanceArmBackend,
    RM65ImpedanceArmBackendConfig,
)
from ..action.replay_proposal import (
    DatasetReplayProposalShadowAdapter,
    DatasetReplayProposalShadowConfig,
    replay_adapter_debug_payload,
)
from ..action.legacy_execution_context import (
    LegacyControlOwnerStateTransitionOutcome,
    LegacyControlOwnerStateTransitionRequest,
)
from .rm65_legacy_worker_binding import RM65LegacyWorkerExecutionBinding
from .position_plan_cut import DeploymentPositionPlanCutDetector
from .rtc_frontend import RtcPolicyShadowConfig, RtcPolicyShadowFrontend
from .rtc_debug import (
    RTC_DEBUG_LEVEL_OFF,
    RTC_DEBUG_LEVEL_SUMMARY,
    RTC_DEBUG_LEVEL_TRACE,
    RTC_DEBUG_LEVELS,
    RtcDebugEventObserver,
    normalize_rtc_debug_level,
)
from .rtc_step_fanout import (
    PolicyActionEmissionFanout,
    RtcFrontendStepFanout,
    RtcPolicyEmissionPublisher,
)
from deployments.client.rtc_frontend import (
    build_rtc_conditioned_model_shadow_frontend_v3,
    build_rtc_model_shadow_frontend,
)
from ..backends.grippers import (
    DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
    DMTAC_TACTILE_FIELDS,
    DMTacGripperBackend,
    DMTacGripperBackendConfig,
    GRIPPER_ACTION_MAX,
    normalize_dmtac_tactile_fields,
)
from ..backends.hands import (
    DEFAULT_INSPIRE_HAND_MAX_FORCE,
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from ..collection.camera_preview import CameraPreviewConfig, CameraPreviewer
from ..collection.hand_state_polling import (
    HandStateHistoryPoller,
    HandStateHistoryPollerConfig,
)
from ..configs.dmtac_hardware import (
    add_dmtac_assignment_arguments,
    dmtac_backend_configs_from_assignment,
    load_dmtac_hardware_assignment,
    manual_dmtac_backend_config_for_side,
    require_dmtac_sides_resolved,
)
from ..defaults import (
    DEFAULT_RM65_ARM_CONTROL_HZ,
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from ..execution_mode_contract import validate_dataset_arm_control_mode
from ..hand_mapping import parse_tactile_regions
from ..interfaces import EffectorBackend
from ..log_utils import ChangeGate, log_event as console_log_event
from ..position_control import (
    ARM_EXECUTION_PROFILE_CHOICES,
    CONTINUOUS_WAYPOINT_WORKER_ISOLATION_CHOICES,
    CONTINUOUS_WAYPOINT_WORKER_PROCESS,
    CONTINUOUS_WAYPOINT_WORKER_THREAD,
    POSITION_15_60_120HZ_PROFILE_NAME,
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
    PositionDeployExecutorFault,
    OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND,
    PINK_POSE_LIFT_BACKEND,
    POSITION_PATH_LIFT_BACKEND_CHOICES,
    apply_rm65_position_profile_defaults,
    build_official_all_ik_governor_from_started_backends,
    build_position_collision_planner_from_started_backends,
    resolve_real_arm_execution_config,
)
from ..position_control.deploy_tcp_pose_envelope import (
    DeployTCPPoseEnvelopeProjector,
)
from ..position_control.tcp_pose_envelope_runtime import (
    TCPPoseEnvelopeRuntimeError,
    load_tcp_pose_envelope_file_spec,
)
from ..sources.camera import (
    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
    REALSENSE_STARTUP_WARMUP_POLICIES,
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
    load_realsense_camera_config_map,
)
from ..real_teleop.camera_runtime import (
    CAMERA_START_STAGGER_SEC as REAL_TELEOP_CAMERA_START_STAGGER_SEC,
    CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES,
    CAMERA_STARTUP_SELF_CHECK_DURATION_SEC,
    CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC,
    CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC,
    CameraStartupFreshnessGate,
    _start_camera_sources_best_effort as _start_real_teleop_camera_sources_best_effort,
    apply_real_teleop_camera_startup_defaults,
)
from ..real_teleop.status_payloads import _build_camera_status
from ..tactile_visualizer import (
    TactileUVVisualizer,
    TactileVisualizerConfig,
    tactile_by_side_from_hand_states,
)
from .dispatch import (
    POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES,
    POSITION_REFERENCE_DISPATCH_AUDIT_POST,
    dispatch_position_reference_command,
    dispatch_teleop_command,
    require_arm_backends_ready_for_position_dispatch,
)
from .arbiter_shadow import (
    ARBITER_SHADOW_MODE_SHADOW,
    ARBITER_SHADOW_MODES_M3,
    PassivePolicyArbiterShadow,
    PolicyArbiterShadowConfig,
)
from .fixed_init import DeployFixedInitController, load_deploy_fixed_init_plan
from .impedance_force_guard import (
    ImpedanceForceGuardConfig,
    ImpedanceForceGuardFault,
    ImpedanceForceGuardMonitor,
    bimanual_emergency_stop,
)
from ..real_teleop.position_force_guard import (
    POSITION_FORCE_SOURCES,
    PositionForceGuardFault,
    PositionForceGuardMonitor,
    PositionForceGuardTripped,
    position_force_guard_config,
)
from deployments.client.policy import load_policy
from deployments.client.rtc_executor import (
    rtc_client_inference_bundle_debug_payload,
)
from .policy_qpos_init import (
    DeployPolicyQposInitController,
    DeployPolicyQposInitSettings,
    load_deploy_policy_qpos_init_plan,
)
from .replay import (
    DatasetReplayPolicy,
    infer_replay_action_hz,
    load_dataset_replay_policy,
    resolve_replay_parquet_path,
)
from .safety_shadow import evaluate_policy_legacy_aligned_safety
from .startup_init import StartupInitializer, resolve_startup_init_action


DEFAULT_CAMERA_CONFIG_PATH = DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
# Deployment runs the vision policy at roughly 15 Hz on the RM65 profile.  A
# 100 ms observation gate leaves only one half-frame of jitter budget at that
# rate, so keep a deployment-specific margin without changing collection's
# stricter sampling contract.
DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC = 0.2
DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST = 4
HAND_BACKEND_INSPIRE = "inspire"
HAND_BACKEND_DMTAC_GRIPPER = "dmtac-gripper"
HAND_BACKEND_CHOICES = (HAND_BACKEND_INSPIRE, HAND_BACKEND_DMTAC_GRIPPER)
GRIPPER_POLICY_DEFAULTS = {
    "output_dim": 0,
    "speed": 0.2,
    "torque": 0.2,
    "max_action": GRIPPER_ACTION_MAX,
    "send_threshold": 0.002,
    "command_hz": DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    "continuous_grasp": True,
    "telemetry_hz": DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    "telemetry_command_grace_s": DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
}
DEPLOY_RM65_ARM_MAX_ANGVEL = (0.7, 0.7, 0.9, 1.5, 1.5, 2.0)
DEPLOY_ARM_STEP_DT_ALPHA = 0.25
DEPLOY_ARM_STEP_DT_MAX_SCALE = 3.0
DEPLOY_ARM_STEP_DT_STALL_RESET_SEC = 0.25
DEPLOY_MAX_ARM_JOINT_STEP_RAD = float(np.deg2rad(6.0))
RM65_ARM_CONTROL_MODE_POSITION = "position"
RM65_ARM_CONTROL_MODE_IMPEDANCE = "impedance"
RM65_ARM_CONTROL_MODES = (
    RM65_ARM_CONTROL_MODE_POSITION,
    RM65_ARM_CONTROL_MODE_IMPEDANCE,
)


@dataclass(frozen=True)
class _PreparedActionExecution:
    accepted: bool
    mode: str
    dispatch_attempted: bool
    dispatch_completed: bool
    asynchronous: bool = False
    position_source_index: int | None = None


def normalize_rm65_arm_control_mode(value: object) -> str:
    text = (
        str(value or RM65_ARM_CONTROL_MODE_POSITION).strip().lower().replace("-", "_")
    )
    aliases = {
        "pos": RM65_ARM_CONTROL_MODE_POSITION,
        "movej": RM65_ARM_CONTROL_MODE_POSITION,
        "canfd": RM65_ARM_CONTROL_MODE_POSITION,
        "force": RM65_ARM_CONTROL_MODE_IMPEDANCE,
        "force_control": RM65_ARM_CONTROL_MODE_IMPEDANCE,
        "admittance": RM65_ARM_CONTROL_MODE_IMPEDANCE,
        "force_admittance": RM65_ARM_CONTROL_MODE_IMPEDANCE,
    }
    mode = aliases.get(text, text)
    if mode not in RM65_ARM_CONTROL_MODES:
        raise ValueError(
            f"Unsupported RM65 arm control mode {value!r}; expected one of {RM65_ARM_CONTROL_MODES}."
        )
    return mode


def effective_arm_control_hz(args: argparse.Namespace) -> float:
    value = getattr(args, "arm_control_hz", None)
    if value is not None:
        return float(value)
    if (
        normalize_rm65_arm_control_mode(getattr(args, "arm_control_mode", None))
        == RM65_ARM_CONTROL_MODE_IMPEDANCE
    ):
        return float(DEFAULT_RM65_ARM_CONTROL_HZ)
    return float(DEFAULT_RM65_POSITION_ARM_CONTROL_HZ)


def effective_position_reference_hz(args: argparse.Namespace) -> float:
    return float(
        getattr(
            args,
            "position_reference_hz",
            DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        )
    )


def _resolve_position_reference_and_command_hz(
    args: argparse.Namespace,
    *,
    backend_control_hz: float,
) -> tuple[float, float]:
    """Keep planner-reference and downstream CANFD clocks distinct."""

    reference_hz = effective_position_reference_hz(args)
    command_hz = (
        float(backend_control_hz)
        if effective_arm_canfd_follow(args)
        else float(reference_hz)
    )
    return float(reference_hz), float(command_hz)


def effective_arm_canfd_follow(args: argparse.Namespace) -> bool:
    value = getattr(args, "arm_canfd_follow", None)
    if value is not None:
        return bool(value)
    return (
        normalize_rm65_arm_control_mode(getattr(args, "arm_control_mode", None))
        == RM65_ARM_CONTROL_MODE_POSITION
    )


class NonBlockingKeyboard:
    def __init__(self):
        self._fd: int | None = None
        self._old_settings: list[Any] | None = None

    def __enter__(self) -> "NonBlockingKeyboard":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def poll(self) -> str | None:
        if self._fd is None:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


class LoopRate:
    def __init__(self):
        self._last: float | None = None
        self.hz = 0.0

    def tick(self, now: float) -> float:
        if self._last is not None:
            dt = max(float(now - self._last), 1e-9)
            instant = 1.0 / dt
            self.hz = instant if self.hz <= 0.0 else 0.2 * instant + 0.8 * self.hz
        self._last = float(now)
        return float(self.hz)


_DEBUG_REDACT_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "password",
    "secret",
}

DEPLOY_DEBUG_PAYLOAD_COMPACT = "compact"
DEPLOY_DEBUG_PAYLOAD_FULL = "full"
DEPLOY_DEBUG_PAYLOAD_MODES = (
    DEPLOY_DEBUG_PAYLOAD_COMPACT,
    DEPLOY_DEBUG_PAYLOAD_FULL,
)

_DEBUG_EFFECTOR_METADATA_KEYS = (
    "source",
    "action_source",
    "deploy_source",
    "policy_name",
    "action_space",
    "action_timestamp",
    "action_pose_source",
    "position_reference_sequence",
    "position_reference_scheduled_monotonic_time",
    "position_reference_query_monotonic_time",
    "position_reference_emitted_monotonic_time",
    "position_reference_emitted_timestamp",
    "position_reference_wall_time",
    "position_reference_hz",
    "position_command_hz",
    "position_transport_target_sequence",
    "position_transport_target_source",
    "position_transport_target_monotonic_time",
    "position_transport_target_wall_time",
    "position_transport_planner",
    "position_collision_status",
    "position_collision_distance_m",
    "position_collision_hold",
    "position_collision_certification_status",
    "position_collision_time_budget_exceeded",
    "position_collision_modification_reasons",
    "position_collision_pair_count",
    "position_collision_planner",
    "position_collision_planner_sequence",
    "position_collision_scene_profile",
    "position_collision_geometry_model",
    "position_endpoint_collision_checked",
    "position_endpoint_collision_passed",
    "position_collision_frame_total_ms",
    "position_collision_horizon_plan_ms",
    "position_collision_precompute_submit_ms",
    "position_collision_selection_ms",
    "position_collision_solve_ms",
    "position_collision_validation_ms",
    "position_collision_endpoint_fk_ms",
    "position_collision_shadow_metadata_ms",
    "position_trajectory_policy_mode",
    "position_trajectory_policy_reasons",
    "position_trajectory_policy_speed_scale",
    "position_trajectory_policy_quick_stop",
    "position_lookahead_available_at_emit",
    "position_lookahead_used",
    "position_causal",
    "position_causal_delay_periods",
    "position_causal_output_lag_sec",
    "position_interpolation_alpha",
    "position_interpolation_alpha_by_modality",
    "position_parent_source_indices",
    "position_parent_source_indices_by_modality",
    "position_parent_source_timestamps_by_modality",
    "backend_final_action_source",
    "backend_final_target_qpos",
    "solver_output_qpos",
    "rm65_background_target_admission_timing",
    "rm65_background_command_phase_timing",
    "target_pose_available",
    "target_pose_base",
    "target_pose_controlled_frame",
    "target_pose_tcp_base",
    "singularity_metadata",
)

_DEBUG_LAST_COMMAND_METADATA_KEYS = (
    "source",
    "action_source",
    "deploy_source",
    "position_reference_sequence",
    "position_reference_scheduled_monotonic_time",
    "position_reference_emitted_monotonic_time",
    "position_reference_hz",
    "position_command_hz",
    "position_transport_target_sequence",
    "position_transport_target_source",
    "position_transport_planner",
    "position_collision_status",
    "position_collision_distance_m",
    "position_collision_hold",
    "position_collision_certification_status",
    "position_collision_time_budget_exceeded",
    "position_collision_frame_total_ms",
    "position_collision_horizon_plan_ms",
    "position_collision_solve_ms",
    "position_collision_validation_ms",
    "position_trajectory_policy_mode",
    "position_trajectory_policy_reasons",
    "position_trajectory_policy_speed_scale",
    "position_trajectory_policy_quick_stop",
    "backend_final_action_source",
    "backend_final_target_qpos",
    "rm65_background_target_admission_timing",
    "rm65_background_command_phase_timing",
)


def _debug_redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _DEBUG_REDACT_KEYS:
                redacted[key_text] = "<redacted>"
            else:
                redacted[key_text] = _debug_redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_debug_redact(item) for item in value]
    return value


def _debug_compact_value(
    value: Any,
    *,
    depth: int = 0,
    max_items: int = 32,
) -> Any:
    """Bound passive debug values before they reach JSON serialization.

    Deployment debug runs beside a deadline-sensitive Python reference worker.
    Keeping bounds here avoids repeatedly walking artifact-sized policy metadata
    while retaining the scalar timing and identity fields used for diagnosis.
    """

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return value.copy()
        return {
            "omitted": True,
            "reason": "compact_deploy_debug_array",
            "shape": tuple(int(item) for item in value.shape),
            "dtype": str(value.dtype),
            "size": int(value.size),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if len(value) <= 512:
            return value
        return value[:512] + f"<... {len(value) - 512} chars omitted>"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        if depth >= 3:
            return {
                "omitted": True,
                "reason": "compact_deploy_debug_depth",
                "type": "mapping",
                "size": int(len(value)),
            }
        items = list(value.items())
        payload = {
            str(key): _debug_compact_value(
                item,
                depth=depth + 1,
                max_items=max_items,
            )
            for key, item in items[:max_items]
        }
        if len(items) > max_items:
            payload["_debug_omitted_item_count"] = int(len(items) - max_items)
        return payload
    if isinstance(value, (list, tuple)):
        if depth >= 3:
            return {
                "omitted": True,
                "reason": "compact_deploy_debug_depth",
                "type": "sequence",
                "size": int(len(value)),
            }
        payload = [
            _debug_compact_value(
                item,
                depth=depth + 1,
                max_items=max_items,
            )
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            payload.append(
                {"_debug_omitted_item_count": int(len(value) - max_items)}
            )
        return payload
    return str(value)


def _debug_remote_policy_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    server = value.get("server")
    server_identity: dict[str, Any] = {}
    if isinstance(server, Mapping):
        for key in (
            "protocol_version",
            "deploy_schema_version",
            "model_id",
            "normalization_hash",
            "policy_io_layout_hash",
            "pose_action_chunk_contract_hash",
            "pose_frame_contract_hashes",
        ):
            if key in server:
                server_identity[key] = _debug_compact_value(server.get(key))
    payload = {
        key: _debug_compact_value(value.get(key), max_items=32)
        for key in (
            "action_key",
            "action_discrete_key",
            "server_summary",
            "validation_warnings",
            "response_keys",
            "server_timing",
            "policy_timing",
        )
        if key in value
    }
    if server_identity:
        payload["server_identity"] = server_identity
    payload["_debug_omitted_server_detail"] = bool(isinstance(server, Mapping))
    return payload


def _debug_metadata_summary(
    metadata: Any,
    *,
    full_payload: bool,
    selected_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    if full_payload:
        return dict(metadata)

    keys = (
        tuple(str(key) for key in metadata)
        if selected_keys is None
        else selected_keys
    )
    payload: dict[str, Any] = {}
    for key in keys:
        if key not in metadata:
            continue
        if key == "remote_policy":
            payload[key] = _debug_remote_policy_summary(metadata.get(key))
        else:
            payload[key] = _debug_compact_value(metadata.get(key), max_items=32)

    omitted = [str(key) for key in metadata if str(key) not in payload]
    if omitted:
        payload["_debug_omitted_metadata"] = {
            "count": int(len(omitted)),
            "keys": omitted[:32],
        }
    return payload


def _debug_background_streaming_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    keys = (
        "enabled",
        "healthy",
        "health_reasons",
        "thread_alive",
        "target_active",
        "target_sequence",
        "target_age_sec",
        "target_stale",
        "fault",
        "fault_count",
        "actual_hz",
        "last_call_ms",
        "last_deadline_lag_ms",
        "last_send_dt_sec",
        "deadline_miss_count",
        "low_rate_count",
        "target_admission_mode",
        "last_target_admission_timing",
        "inflight_target_preemption_count",
        "last_inflight_target_preemption",
    )
    return {
        key: _debug_compact_value(value.get(key), max_items=16)
        for key in keys
        if key in value
    }


def _debug_backend_dispatch_by_effector_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    payload: dict[str, Any] = {}
    for effector_name, raw_item in value.items():
        if not isinstance(raw_item, Mapping):
            continue
        keys = (
            "accepted",
            "rewritten",
            "backend_ready",
            "operation",
            "status",
            "result_code",
            "singularity_gate_status",
            "dispatch_timing",
            "target_admission_timing",
            "command_phase_timing",
            "expected_qpos",
            "requested_qpos",
            "final_qpos",
            "backend_final_target_qpos",
        )
        item = {
            key: _debug_compact_value(raw_item.get(key), max_items=16)
            for key in keys
            if key in raw_item
        }
        if "background_streaming" in raw_item:
            item["background_streaming"] = _debug_background_streaming_summary(
                raw_item.get("background_streaming")
            )
        payload[str(effector_name)] = item
    return payload


def _debug_backend_group_health_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    payload = {
        key: _debug_compact_value(value.get(key), max_items=16)
        for key in ("ready", "require_background_target", "required_sides", "timing")
        if key in value
    }
    raw_sides = value.get("sides")
    if isinstance(raw_sides, Mapping):
        payload["sides"] = {}
        for side, raw_item in raw_sides.items():
            if not isinstance(raw_item, Mapping):
                continue
            payload["sides"][str(side)] = {
                key: _debug_compact_value(raw_item.get(key), max_items=16)
                for key in (
                    "ready",
                    "reasons",
                    "backend_ready",
                    "state_error",
                    "hardware_ready",
                    "emergency_stopped",
                    "send_result_code",
                    "warnings",
                    "timing",
                )
                if key in raw_item
            }
    return payload


def _debug_position_executor_event_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    collision_debug = compact.get("position_collision_debug")
    if isinstance(collision_debug, Mapping):
        compact["position_collision_debug"] = {
            "omitted": True,
            "reason": "compact_deploy_debug_redundant_detail",
            "available_keys": tuple(str(key) for key in collision_debug),
            "q_ik_raw_by_side": _debug_compact_value(
                collision_debug.get("q_ik_raw_by_side"),
                max_items=16,
            ),
            "q_ref_by_side": _debug_compact_value(
                collision_debug.get("q_ref_by_side"),
                max_items=16,
            ),
        }
    if "position_arm_backend_dispatch_by_effector" in compact:
        compact["position_arm_backend_dispatch_by_effector"] = (
            _debug_backend_dispatch_by_effector_summary(
                compact.get("position_arm_backend_dispatch_by_effector")
            )
        )
    if "position_arm_backend_group_health" in compact:
        compact["position_arm_backend_group_health"] = (
            _debug_backend_group_health_summary(
                compact.get("position_arm_backend_group_health")
            )
        )
    return compact


def _debug_args_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(vars(args))
    policy_kwargs_text = payload.get("policy_kwargs_json")
    if isinstance(policy_kwargs_text, str):
        try:
            policy_kwargs = json.loads(policy_kwargs_text or "{}")
        except json.JSONDecodeError:
            policy_kwargs = policy_kwargs_text
        payload["policy_kwargs_json"] = _debug_redact(policy_kwargs)
    return _debug_redact(payload)


def _debug_vector(
    value: Any,
    *,
    length: int | None = None,
    require_finite: bool = True,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if length is not None and vector.shape != (int(length),):
        return None
    if require_finite and not np.all(np.isfinite(vector)):
        return None
    return vector.astype(np.float32, copy=True)


def _debug_xyzquat(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    if array.shape == (4, 4):
        pose = pose_matrix_to_xyzquat(array)
        return None if pose is None else pose.astype(np.float32, copy=True)
    vector = _debug_vector(array, length=7)
    if vector is None:
        return None
    quat = np.asarray(vector[3:7], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return None
    vector = vector.astype(np.float32, copy=True)
    vector[3:7] = (quat / norm).astype(np.float32)
    return vector


def _debug_action_vector(action: Any) -> np.ndarray | None:
    vector = getattr(action, "vector", action)
    return _debug_vector(vector, length=ACTION_DIM, require_finite=False)


def _debug_pose_by_side_from_vector(
    vector: np.ndarray | None,
    active_sides: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if vector is None:
        return {}
    poses: dict[str, np.ndarray] = {}
    for side in active_sides:
        target_slice = ACTION_ARM_POSE_SLICES[side]
        pose = _debug_xyzquat(vector[target_slice])
        if pose is not None:
            poses[side] = pose
    return poses


def _debug_hand_by_side_from_vector(
    vector: np.ndarray | None,
    active_sides: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if vector is None:
        return {}
    hands: dict[str, np.ndarray] = {}
    for side in active_sides:
        target_slice = ACTION_HAND_SLICES[side]
        target = _debug_vector(vector[target_slice])
        if target is not None:
            hands[side] = target
    return hands


def _debug_arm_joint_targets(
    action: Any, active_sides: tuple[str, ...]
) -> dict[str, np.ndarray]:
    targets = getattr(action, "arm_joint_targets", {}) or {}
    if not isinstance(targets, Mapping):
        return {}
    resolved: dict[str, np.ndarray] = {}
    for side in active_sides:
        target = _debug_vector(targets.get(side), length=6)
        if target is not None:
            resolved[side] = target
    return resolved


def _debug_action_summary(
    action: Any,
    active_sides: tuple[str, ...],
    *,
    full_payload: bool = False,
) -> dict[str, Any] | None:
    vector = _debug_action_vector(action)
    if vector is None:
        return None
    metadata = getattr(action, "metadata", {})
    return {
        "source": (
            str(metadata.get("source", "")) if isinstance(metadata, Mapping) else ""
        ),
        "metadata": _debug_metadata_summary(
            metadata,
            full_payload=full_payload,
        ),
        "vector": vector,
        "arm_pose_by_side": _debug_pose_by_side_from_vector(vector, active_sides),
        "arm_joint_targets_by_side": _debug_arm_joint_targets(action, active_sides),
        "hand_by_side": _debug_hand_by_side_from_vector(vector, active_sides),
    }


def _debug_quat_delta_deg(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    ref_quat = np.asarray(reference[3:7], dtype=float)
    candidate_quat = np.asarray(candidate[3:7], dtype=float)
    ref_norm = float(np.linalg.norm(ref_quat))
    candidate_norm = float(np.linalg.norm(candidate_quat))
    if ref_norm <= 1e-12 or candidate_norm <= 1e-12:
        return None
    ref_quat = ref_quat / ref_norm
    candidate_quat = candidate_quat / candidate_norm
    dot = float(np.clip(abs(np.dot(ref_quat, candidate_quat)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _debug_pose_delta(reference: Any, candidate: Any) -> dict[str, Any] | None:
    ref_pose = _debug_xyzquat(reference)
    candidate_pose = _debug_xyzquat(candidate)
    if ref_pose is None or candidate_pose is None:
        return None
    xyz_delta = np.asarray(candidate_pose[:3] - ref_pose[:3], dtype=np.float32)
    return {
        "xyz_delta_m": xyz_delta,
        "xyz_norm_m": float(np.linalg.norm(xyz_delta)),
        "rot_delta_deg": _debug_quat_delta_deg(ref_pose, candidate_pose),
    }


def _debug_pose_delta_by_side(
    reference_by_side: Mapping[str, Any],
    candidate_by_side: Mapping[str, Any],
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for side in ALL_SIDES:
        if side not in reference_by_side or side not in candidate_by_side:
            continue
        delta = _debug_pose_delta(reference_by_side[side], candidate_by_side[side])
        if delta is not None:
            deltas[side] = delta
    return deltas


def _debug_command_arm_pose_by_side(
    command: Any, active_sides: tuple[str, ...]
) -> dict[str, np.ndarray]:
    if command is None:
        return {}
    commands = getattr(command, "commands", {}) or {}
    poses: dict[str, np.ndarray] = {}
    for side in active_sides:
        effector_command = commands.get(f"{side}_arm")
        if (
            effector_command is None
            or getattr(effector_command, "command_type", "") != "cartesian_pose"
        ):
            continue
        metadata = getattr(effector_command, "metadata", {}) or {}
        pose = None
        if isinstance(metadata, Mapping):
            pose = _debug_xyzquat(metadata.get("target_xyzquat"))
            if pose is None:
                pose = _debug_xyzquat(metadata.get("target_pose_base"))
        if pose is None:
            pose = _debug_xyzquat(getattr(effector_command, "value", None))
        if pose is not None:
            poses[side] = pose
    return poses


def _debug_command_summary(
    command: Any,
    active_sides: tuple[str, ...],
    *,
    full_payload: bool = False,
) -> dict[str, Any] | None:
    if command is None:
        return None
    commands = getattr(command, "commands", {}) or {}
    command_payload: dict[str, Any] = {}
    for effector_name, effector_command in commands.items():
        metadata = getattr(effector_command, "metadata", {}) or {}
        item: dict[str, Any] = {
            "command_type": getattr(effector_command, "command_type", ""),
            "side": getattr(effector_command, "side", None),
            "frame_name": getattr(effector_command, "frame_name", None),
            "value": getattr(effector_command, "value", None),
            "metadata": _debug_metadata_summary(
                metadata,
                full_payload=full_payload,
                selected_keys=(
                    None if full_payload else _DEBUG_EFFECTOR_METADATA_KEYS
                ),
            ),
        }
        if item["command_type"] == "cartesian_pose":
            item["target_xyzquat"] = _debug_xyzquat(
                item["metadata"].get("target_xyzquat")
                if isinstance(item["metadata"], Mapping)
                else None
            )
            if item["target_xyzquat"] is None:
                item["target_xyzquat"] = _debug_xyzquat(
                    getattr(effector_command, "value", None)
                )
        command_payload[str(effector_name)] = item
    return {
        "timestamp": getattr(command, "timestamp", None),
        "metadata": _debug_metadata_summary(
            getattr(command, "metadata", {}) or {},
            full_payload=full_payload,
        ),
        "commands": command_payload,
        "arm_pose_by_side": _debug_command_arm_pose_by_side(command, active_sides),
    }


def _debug_arm_state_summary(
    states: Mapping[str, Mapping[str, Any]] | None,
    *,
    full_payload: bool = False,
) -> dict[str, Any]:
    if not states:
        return {}
    keys = (
        "current_qpos",
        "current_tcp_pose",
        "hardware_qpos",
        "last_target_qpos",
        "last_command",
        "last_command_timestamp",
        "hardware_pose_6d",
        "hardware_refresh_time",
        "hardware_read_phase_timing",
        "hardware_result_code",
        "hardware_result_text",
        "hardware_arm_err",
        "hardware_sys_err",
        "send_result_code",
        "send_result_text",
        "emergency_stopped",
        "effective_control_dt_sec",
        "effective_step_scale",
        "ik_metadata",
        "singularity_metadata",
        "tcp_pose_source",
        "state_error",
        "pose_error",
        "active_tool_name",
        "configured_tcp_enabled",
        "tool_tcp_source",
        "tool_tcp_endpoint_delta_m",
        "tool_tcp_endpoint_delta_deg",
    )
    payload: dict[str, Any] = {}
    for side, state in states.items():
        side_payload = {key: state.get(key) for key in keys if key in state}
        if not full_payload:
            last_command = side_payload.get("last_command")
            if isinstance(last_command, Mapping):
                last_command_payload = dict(last_command)
                last_command_payload["metadata"] = _debug_metadata_summary(
                    last_command.get("metadata", {}),
                    full_payload=False,
                    selected_keys=_DEBUG_LAST_COMMAND_METADATA_KEYS,
                )
                last_command_payload.pop("execution_work_item", None)
                side_payload["last_command"] = last_command_payload
            for key in ("ik_metadata", "singularity_metadata"):
                if key in side_payload:
                    side_payload[key] = _debug_compact_value(
                        side_payload[key],
                        max_items=32,
                    )
        pose = _debug_xyzquat(side_payload.get("current_tcp_pose"))
        if pose is not None:
            side_payload["current_tcp_xyzquat"] = pose
        payload[str(side)] = side_payload
    return payload


def _debug_state_pose_by_side(
    states: Mapping[str, Mapping[str, Any]] | None
) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    if not states:
        return poses
    for side, state in states.items():
        pose = _debug_xyzquat(state.get("current_tcp_pose"))
        if pose is not None:
            poses[str(side)] = pose
    return poses


def _debug_backend_arm_states(
    backends: Mapping[str, EffectorBackend]
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for side, backend in backends.items():
        try:
            states[side] = dict(backend.get_state())
        except Exception as exc:  # noqa: BLE001
            states[side] = {"state_error": f"{type(exc).__name__}: {exc}"}
    return states


def _debug_backend_limits(
    backends: Mapping[str, EffectorBackend],
    *,
    now: float,
) -> dict[str, Any]:
    limits: dict[str, Any] = {}
    for side, backend in backends.items():
        get_effective_step = getattr(backend, "get_effective_max_joint_step", None)
        try:
            effective_step = (
                get_effective_step(now=now) if callable(get_effective_step) else None
            )
        except Exception as exc:  # noqa: BLE001
            effective_step = None
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = ""
        config = backend.config
        limits[side] = {
            "control_dt_sec": float(config.control_dt),
            "max_angvel_rad_s": tuple(float(value) for value in config.max_angvel),
            "effective_max_joint_step_rad": effective_step,
            "background_streaming_enabled": bool(
                getattr(config, "background_streaming_enabled", False)
            ),
            "background_streaming_hz": (
                None
                if getattr(config, "background_streaming_hz", None) is None
                else float(getattr(config, "background_streaming_hz"))
            ),
            "background_streaming_target_admission_mode": getattr(
                config,
                "background_streaming_target_admission_mode",
                None,
            ),
            "canfd_follow": getattr(config, "canfd_follow", None),
            "observation_only": bool(getattr(config, "observation_only", False)),
            "send_commands": getattr(config, "send_commands", None),
            "error": error,
        }
    return limits


def canonical_deployment_debug_root(*, now: float | None = None) -> Path:
    timestamp_time = time.time() if now is None else float(now)
    local_time = time.localtime(timestamp_time)
    date_partition = time.strftime("%Y-%m-%d", local_time)
    stamp = time.strftime("%Y%m%d_%H%M%S", local_time)
    millis = int((timestamp_time % 1.0) * 1000.0)
    return RuntimeArtifactLayout.configured().run_dir(
        kind="policy_deploy",
        date_partition=date_partition,
        run_id=f"deploy_{stamp}_{millis:03d}",
    )


def resolve_deployment_debug_root(
    args: argparse.Namespace,
    *,
    now: float | None = None,
) -> ArtifactWriteResolution:
    return resolve_artifact_write_path(
        explicit_path=getattr(args, "debug_dir", None),
        artifact_layout=getattr(args, "artifact_layout", None),
        legacy_path=repo_path("teleop_records", "deployment_debug"),
        canonical_path=canonical_deployment_debug_root(now=now),
    )


class DeploymentDebugLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        root_dir: Path | None,
        active_sides: tuple[str, ...],
        payload_mode: str = DEPLOY_DEBUG_PAYLOAD_COMPACT,
        flush_every_events: int = 1,
        flush_interval_sec: float = 0.0,
    ):
        self.enabled = bool(enabled)
        self.active_sides = tuple(active_sides)
        self.payload_mode = str(payload_mode).strip().lower()
        if self.payload_mode not in DEPLOY_DEBUG_PAYLOAD_MODES:
            raise ValueError(
                "payload_mode must be one of "
                f"{DEPLOY_DEBUG_PAYLOAD_MODES}, got {payload_mode!r}."
            )
        self.full_payload = self.payload_mode == DEPLOY_DEBUG_PAYLOAD_FULL
        self.path: Path | None = None
        self._handle: Any = None
        self._lock = threading.RLock()
        self._write_failed = False
        self._flush_every_events = int(flush_every_events)
        self._flush_interval_sec = float(flush_interval_sec)
        if self._flush_every_events <= 0:
            raise ValueError("flush_every_events must be positive.")
        if (
            not math.isfinite(self._flush_interval_sec)
            or self._flush_interval_sec < 0.0
        ):
            raise ValueError("flush_interval_sec must be finite and non-negative.")
        self._pending_flush_events = 0
        self._last_flush_at = time.monotonic()
        if not self.enabled:
            return
        debug_root = (
            root_dir
            if root_dir is not None
            else repo_path("teleop_records", "deployment_debug")
        )
        debug_root = Path(debug_root).expanduser()
        if not debug_root.is_absolute():
            debug_root = repo_path(debug_root)
        debug_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1.0) * 1000.0)
        self.path = debug_root / f"deploy_debug_{stamp}_{millis:03d}.jsonl"
        self._handle = self.path.open("a", encoding="utf-8")
        print(
            f"[DeployDebug] writing action/pose debug JSONL to {self.path}", flush=True
        )

    def close(self) -> None:
        with self._lock:
            if self._handle is None:
                return
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            if not self.enabled or self._handle is None:
                return
            event_name = str(event)
            event_payload = dict(payload)
            if (
                not self.full_payload
                and event_name.startswith("position_executor_")
            ):
                event_payload = _debug_position_executor_event_summary(event_payload)
            record = {
                "event": event_name,
                "wall_time": time.time(),
                "debug_payload_mode": self.payload_mode,
                **event_payload,
            }
            try:
                try:
                    encoded = json.dumps(
                        to_serializable(record),
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                except Exception as serialization_exc:  # noqa: BLE001
                    # One newly added passive field must not disable every
                    # subsequent record.  Preserve a compact failure event and
                    # continue; actual file I/O failures below remain terminal
                    # for this observational sink.
                    encoded = json.dumps(
                        {
                            "event": "debug_serialization_error",
                            "wall_time": time.time(),
                            "source_event": str(event),
                            "error_type": type(serialization_exc).__name__,
                            "error": str(serialization_exc),
                        },
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                self._handle.write(encoded + "\n")
                self._pending_flush_events += 1
                now = time.monotonic()
                if self._pending_flush_events >= self._flush_every_events or (
                    self._flush_interval_sec > 0.0
                    and now - self._last_flush_at >= self._flush_interval_sec
                ):
                    self._handle.flush()
                    self._pending_flush_events = 0
                    self._last_flush_at = now
            except Exception as exc:  # noqa: BLE001
                if not self._write_failed:
                    print(
                        f"[DeployDebug] disabled after write failure: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                self._write_failed = True
                self.enabled = False

    def log_startup(
        self,
        *,
        args: argparse.Namespace,
        policy: Any,
        active_sides: tuple[str, ...],
        hands_enabled: bool,
        effective_action_hz: float,
        action_hz_info: Mapping[str, Any],
        effective_policy_hz: float,
        policy_hz_source: str,
        backend_control_hz: float,
        action_queue_policy: str,
        send_actions: bool,
        arm_backends: Mapping[str, EffectorBackend],
    ) -> None:
        if not self.enabled:
            return
        self.log_event(
            "startup",
            {
                "args": _debug_args_payload(args),
                "policy_name": getattr(policy, "name", ""),
                "debug_payload_mode": self.payload_mode,
                "policy_metadata": _debug_redact(getattr(policy, "metadata", {})),
                "active_sides": tuple(active_sides),
                "arms_enabled": bool(args.arms_enabled),
                "hands_enabled": bool(hands_enabled),
                "send_actions": bool(send_actions),
                "effective_action_hz": float(effective_action_hz),
                "effective_policy_hz": float(effective_policy_hz),
                "action_hz_info": dict(action_hz_info),
                "policy_hz_source": str(policy_hz_source),
                "backend_control_hz": float(backend_control_hz),
                "action_queue_policy": str(action_queue_policy),
                "arm_execution": (
                    args._real_arm_execution_config.to_metadata()
                    if getattr(args, "_real_arm_execution_config", None) is not None
                    else None
                ),
                "safety_limits": {
                    "arm_workspace_min": tuple(
                        float(value) for value in args.arm_workspace_min
                    ),
                    "arm_workspace_max": tuple(
                        float(value) for value in args.arm_workspace_max
                    ),
                    "max_arm_xyz_step_m": float(args.max_arm_xyz_step_m),
                    "max_arm_rot_step_deg": float(args.max_arm_rot_step_deg),
                    "max_arm_joint_step_rad": float(args.max_arm_joint_step_rad),
                    "max_hand_step": float(args.max_hand_step),
                },
                "backend_limits_by_side": _debug_backend_limits(
                    arm_backends, now=time.time()
                ),
            },
        )

    def log_policy_chunk(
        self,
        *,
        timestamp: float,
        step_index: int,
        policy_name: str,
        policy_output: Any,
        actions: list[NormalizedAction],
        accepted: bool,
        discarded_by_ik: bool,
        queue_policy: str,
        queue_len_before: int,
        queue_len_after: int,
        faults: list[str],
        ik_recovery_statuses: Mapping[str, str] | None = None,
        ik_recovery_diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.log_event(
            "policy_chunk",
            {
                "timestamp": float(timestamp),
                "step_index": int(step_index),
                "policy_name": str(policy_name),
                "policy_output_type": type(policy_output).__name__,
                "chunk_size": int(len(actions)),
                "accepted": bool(accepted),
                "discarded_by_ik": bool(discarded_by_ik),
                "queue_policy": str(queue_policy),
                "queue_len_before": int(queue_len_before),
                "queue_len_after": int(queue_len_after),
                "faults": list(faults),
                "ik_recovery_statuses": (
                    {} if ik_recovery_statuses is None else dict(ik_recovery_statuses)
                ),
                "ik_recovery_diagnostics": (
                    {}
                    if ik_recovery_diagnostics is None
                    else {
                        str(side): dict(value)
                        for side, value in ik_recovery_diagnostics.items()
                    }
                ),
                "actions": [
                    _debug_action_summary(
                        action,
                        self.active_sides,
                        full_payload=self.full_payload,
                    )
                    for action in actions
                ],
            },
        )

    def log_policy_error(
        self,
        *,
        timestamp: float,
        step_index: int,
        policy_name: str,
        exc: BaseException,
        queue_len_before: int,
    ) -> None:
        if not self.enabled:
            return
        self.log_event(
            "policy_error",
            {
                "timestamp": float(timestamp),
                "step_index": int(step_index),
                "policy_name": str(policy_name),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "queue_len_before": int(queue_len_before),
            },
        )

    def log_action_step(
        self,
        *,
        timestamp: float,
        step_index: int,
        policy_name: str,
        raw_action: NormalizedAction,
        canonical_action: Any,
        pre_arm_states: Mapping[str, Mapping[str, Any]],
        post_arm_states: Mapping[str, Mapping[str, Any]] | None,
        arm_backends: Mapping[str, EffectorBackend],
        send_actions: bool,
        queue_len_before_pop: int,
        queue_len_after_pop: int,
        faults: list[str],
        dispatch_error: BaseException | None,
        execution: _PreparedActionExecution | None = None,
        phase_timing: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        raw_summary = _debug_action_summary(
            raw_action,
            self.active_sides,
            full_payload=self.full_payload,
        )
        filtered_summary = _debug_action_summary(
            getattr(canonical_action, "action", None),
            self.active_sides,
            full_payload=self.full_payload,
        )
        command_summary = _debug_command_summary(
            getattr(canonical_action, "command", None),
            self.active_sides,
            full_payload=self.full_payload,
        )
        raw_pose = (
            {} if raw_summary is None else raw_summary.get("arm_pose_by_side", {})
        )
        filtered_pose = (
            {}
            if filtered_summary is None
            else filtered_summary.get("arm_pose_by_side", {})
        )
        command_pose = (
            {}
            if command_summary is None
            else command_summary.get("arm_pose_by_side", {})
        )
        pre_tcp_pose = _debug_state_pose_by_side(pre_arm_states)
        post_tcp_pose = _debug_state_pose_by_side(post_arm_states)
        command = getattr(canonical_action, "command", None)
        command_count = (
            len(getattr(command, "commands", {}) or {}) if command is not None else 0
        )
        dispatch_attempted = (
            bool(execution.dispatch_attempted)
            if execution is not None
            else bool(command_count and send_actions)
        )
        self.log_event(
            "action_step",
            {
                "timestamp": float(timestamp),
                "step_index": int(step_index),
                "policy_name": str(policy_name),
                "send_actions": bool(send_actions),
                "dispatch_attempted": bool(dispatch_attempted),
                "dispatch_completed": bool(
                    execution.dispatch_completed
                    if execution is not None
                    else dispatch_attempted and dispatch_error is None
                ),
                "dispatch_asynchronous": bool(
                    execution.asynchronous if execution is not None else False
                ),
                "dispatch_mode": (
                    execution.mode
                    if execution is not None
                    else ("send_actions" if send_actions else "dry_run")
                ),
                "position_source_index": (
                    None if execution is None else execution.position_source_index
                ),
                "dispatch_error": (
                    None
                    if dispatch_error is None
                    else {
                        "type": type(dispatch_error).__name__,
                        "message": str(dispatch_error),
                    }
                ),
                "queue_len_before_pop": int(queue_len_before_pop),
                "queue_len_after_pop": int(queue_len_after_pop),
                "faults": list(faults),
                "phase_timing": _debug_compact_value(
                    {} if phase_timing is None else phase_timing,
                    max_items=32,
                ),
                "raw_action": raw_summary,
                "filtered_action": filtered_summary,
                "command": command_summary,
                "canonical_metadata": _debug_metadata_summary(
                    getattr(canonical_action, "metadata", {}),
                    full_payload=self.full_payload,
                ),
                "raw_to_filtered_pose_delta": _debug_pose_delta_by_side(
                    raw_pose, filtered_pose
                ),
                "filtered_to_command_pose_delta": _debug_pose_delta_by_side(
                    filtered_pose, command_pose
                ),
                "command_to_pre_tcp_delta": _debug_pose_delta_by_side(
                    command_pose, pre_tcp_pose
                ),
                "command_to_post_tcp_delta": _debug_pose_delta_by_side(
                    command_pose, post_tcp_pose
                ),
                "pre_arm_state_by_side": _debug_arm_state_summary(
                    pre_arm_states,
                    full_payload=self.full_payload,
                ),
                "post_arm_state_by_side": _debug_arm_state_summary(
                    post_arm_states,
                    full_payload=self.full_payload,
                ),
                "backend_limits_by_side": _debug_backend_limits(
                    arm_backends, now=time.time()
                ),
            },
        )


class ArmStateReader:
    def __init__(
        self,
        backends: Mapping[str, EffectorBackend],
        *,
        refresh_hz: float,
        compute_tcp_pose: bool = True,
    ):
        self.backends = dict(backends)
        self.refresh_hz = float(refresh_hz)
        self.compute_tcp_pose = bool(compute_tcp_pose)
        self._last_refresh_at = {side: 0.0 for side in self.backends}
        self._history_by_side = {
            side: CausalSnapshotHistory(maxlen=256, name=f"deploy:arm:{side}")
            for side in self.backends
        }

    def snapshot(self, now: float) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        refresh_period = 0.0 if self.refresh_hz <= 0.0 else 1.0 / self.refresh_hz
        for side, backend in self.backends.items():
            refresh_error: str | None = None
            if (
                refresh_period > 0.0
                and backend.is_ready()
                and now - self._last_refresh_at.get(side, 0.0) >= refresh_period
            ):
                try:
                    refresh = getattr(backend, "refresh_state_from_hardware", None)
                    if callable(refresh):
                        refresh()
                except Exception as exc:  # noqa: BLE001
                    refresh_error = f"refresh_failed:{type(exc).__name__}:{exc}"
                self._last_refresh_at[side] = now
            state = dict(backend.get_state())
            if refresh_error is not None:
                state["state_error"] = refresh_error
            state["actual_tcp_pose"] = None
            state["actual_tcp_pose_source"] = None
            state["actual_tcp_pose_timestamp"] = None
            if self.compute_tcp_pose:
                try:
                    hardware_qpos = state.get("hardware_qpos")
                    compute_pose = getattr(backend, "compute_tcp_pose", None)
                    if not callable(compute_pose):
                        raise AttributeError("backend has no compute_tcp_pose")
                    if hardware_qpos is not None:
                        measured_pose = compute_pose(qpos=hardware_qpos)
                        state["current_tcp_pose"] = measured_pose
                        state["tcp_pose_source"] = "hardware_qpos_fk"
                        try:
                            hardware_qpos_vector = np.asarray(
                                hardware_qpos, dtype=float
                            ).reshape(-1)
                            measurement_timestamp = float(
                                state.get("hardware_measurement_timestamp")
                            )
                            measurement_sequence = state.get(
                                "hardware_measurement_sequence"
                            )
                            measurement_valid = state.get("hardware_measurement_valid")
                            read_success = state.get("hardware_read_success")
                            sequence_valid = (
                                measurement_sequence is None
                                or int(measurement_sequence) > 0
                            )
                            actual_valid = bool(
                                hardware_qpos_vector.shape == (6,)
                                and np.all(np.isfinite(hardware_qpos_vector))
                                and np.isfinite(measurement_timestamp)
                                and isinstance(measurement_valid, (bool, np.bool_))
                                and bool(measurement_valid)
                                and not (
                                    isinstance(read_success, (bool, np.bool_))
                                    and not bool(read_success)
                                )
                                and sequence_valid
                            )
                        except (TypeError, ValueError, OverflowError):
                            actual_valid = False
                            measurement_timestamp = None
                        if actual_valid:
                            state["actual_tcp_pose"] = np.asarray(
                                measured_pose, dtype=float
                            ).copy()
                            state["actual_tcp_pose_source"] = "hardware_qpos_fk"
                            state["actual_tcp_pose_timestamp"] = float(
                                measurement_timestamp
                            )
                    else:
                        state["current_tcp_pose"] = compute_pose()
                        state["tcp_pose_source"] = "current_qpos_fk"
                except Exception as exc:  # noqa: BLE001
                    state.setdefault(
                        "pose_error",
                        f"compute_tcp_pose_failed:{type(exc).__name__}:{exc}",
                    )
            source_timestamp = state.get(
                "hardware_measurement_timestamp",
                state.get("hardware_refresh_time"),
            )
            self._history_by_side[side].publish(
                state,
                source_timestamp=source_timestamp,
                available_timestamp=None,
            )
            selected = self._history_by_side[side].latest_at_or_before(float(now))
            if selected is None:
                states[side] = {
                    "causal_snapshot_missing": True,
                    "causal_snapshot_cutoff_timestamp": float(now),
                    "causal_snapshot_contract": (
                        "source_timestamp<=T0_and_available_timestamp<=T0"
                    ),
                }
            else:
                selected_state, selection = selected
                selected_state.update(
                    {
                        "causal_snapshot_missing": False,
                        "causal_snapshot_sequence": selection.sequence,
                        "causal_snapshot_source_timestamp": selection.source_timestamp,
                        "causal_snapshot_available_timestamp": selection.available_timestamp,
                        "causal_snapshot_age_sec": selection.age_sec,
                        "causal_snapshot_cutoff_timestamp": float(now),
                        "causal_snapshot_contract": (
                            "source_timestamp<=T0_and_available_timestamp<=T0"
                        ),
                    }
                )
                states[side] = selected_state
        return states


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _unit_interval_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parse_dmtac_tactile_fields(text: str) -> tuple[str, ...]:
    try:
        return normalize_dmtac_tactile_fields(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_active_sides(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip().lower()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
    else:
        parts = [
            part.strip().lower()
            for part in str(value).replace(",", " ").split()
            if part.strip()
        ]
    if not parts:
        raise argparse.ArgumentTypeError(
            "active-sides must be one of: left, right, both"
        )
    if parts == ["both"]:
        return ("left", "right")
    invalid = sorted(set(parts) - set(ALL_SIDES))
    if invalid:
        raise argparse.ArgumentTypeError(
            "active-sides must be one of: left, right, both"
        )
    return tuple(side for side in ALL_SIDES if side in set(parts))


def _parse_disconnected_gripper_sides(
    value: str | list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip().lower()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
    else:
        parts = [
            part.strip().lower()
            for part in str(value).replace(",", " ").split()
            if part.strip()
        ]
    if not parts or parts == ["none"]:
        return ()
    if parts == ["both"]:
        return tuple(ALL_SIDES)
    invalid = sorted(set(parts) - set(ALL_SIDES))
    if invalid:
        raise argparse.ArgumentTypeError(
            "disconnected-gripper-sides must be none, left, right, or both"
        )
    return tuple(side for side in ALL_SIDES if side in set(parts))


def _parse_end_effector_sides(
    value: str | list[str] | tuple[str, ...],
    *,
    active_sides: tuple[str, ...],
    hands_enabled: bool,
) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip().lower()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
    else:
        parts = [
            part.strip().lower()
            for part in str(value).replace(",", " ").split()
            if part.strip()
        ]
    if not parts or parts == ["active"]:
        return tuple(active_sides) if hands_enabled else ()
    if parts == ["none"]:
        return ()
    if parts == ["both"]:
        parts = ["left", "right"]
    invalid = set(parts) - set(ALL_SIDES)
    if invalid:
        raise argparse.ArgumentTypeError(
            "end-effector-sides must be active, none, left, right, or both"
        )
    if not hands_enabled:
        raise ValueError("--end-effector-sides requires hands to be enabled.")
    requested = set(parts)
    unavailable = requested - set(active_sides)
    if unavailable:
        raise ValueError(
            "--end-effector-sides must be a subset of --active-sides, got "
            + ",".join(sorted(unavailable))
        )
    return tuple(side for side in ALL_SIDES if side in requested)


def _parse_float_tuple(text: str, *, length: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != int(length):
        raise argparse.ArgumentTypeError(
            f"{name} must contain {length} comma-separated floats"
        )
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {name}: {text}") from exc
    return values


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _parse_vector3(
    value: str | list[str] | tuple[object, ...]
) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
        text = " ".join(parts)
    else:
        text = str(value)
        parts = [
            part.strip() for part in text.replace(",", " ").split() if part.strip()
        ]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("vector3 must contain 3 floats")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid vector3: {text}") from exc
    return (values[0], values[1], values[2])


def _parse_qpos6(text: str) -> tuple[float, ...]:
    return _parse_float_tuple(text, length=6, name="qpos6")


def _parse_arm_max_angvel(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("arm max angvel must contain 1 or 6 floats")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid arm max angvel: {text}") from exc
    if len(values) == 1:
        if values[0] <= 0.0:
            raise argparse.ArgumentTypeError("arm max angvel must be positive")
        return (float(values[0]),) * 6
    if len(values) != 6 or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError(
            "arm max angvel must contain 1 or 6 positive floats"
        )
    return tuple(float(value) for value in values)


def _hand_backend_mode(args: argparse.Namespace) -> str:
    mode = str(
        getattr(args, "hand_backend", HAND_BACKEND_INSPIRE) or HAND_BACKEND_INSPIRE
    )
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported hand backend {mode!r}; expected one of {HAND_BACKEND_CHOICES}."
        )
    return mode


def _hand_backend_mode_for_side(args: argparse.Namespace, side: str) -> str:
    side = str(side).strip().lower()
    if side not in ALL_SIDES:
        raise ValueError(f"Unsupported hand backend side {side!r}.")
    override = getattr(args, f"{side}_hand_backend", None)
    mode = _hand_backend_mode(args) if override in (None, "") else str(override)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported {side} hand backend {mode!r}; expected one of {HAND_BACKEND_CHOICES}."
        )
    return mode


def _load_gripper_task_profile(args: argparse.Namespace) -> dict[str, Any]:
    path_text = str(getattr(args, "gripper_task_profile", "") or "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Gripper task profile {path} must contain a JSON object.")
    base_gripper = payload.get("gripper")
    if base_gripper is not None and not isinstance(base_gripper, Mapping):
        raise ValueError(f"Profile {path} field 'gripper' must be an object.")
    selected: Mapping[str, Any] | None = None
    tasks = payload.get("tasks")
    if isinstance(tasks, Mapping):
        task_keys = (
            str(getattr(args, "gripper_task", "") or ""),
            str(getattr(args, "task_id", "") or ""),
            str(getattr(args, "task_name", "") or ""),
            str(getattr(args, "task", "") or ""),
        )
        for key in task_keys:
            item = tasks.get(key)
            if isinstance(item, Mapping):
                selected = item
                setattr(args, "_resolved_gripper_task", key)
                break
    if selected is not None:
        gripper = selected.get("gripper")
        if gripper is not None and not isinstance(gripper, Mapping):
            raise ValueError(
                f"Profile {path} selected task field 'gripper' must be an object."
            )
        merged = dict(base_gripper) if isinstance(base_gripper, Mapping) else {}
        merged.update(dict(gripper) if isinstance(gripper, Mapping) else dict(selected))
        return merged
    if isinstance(base_gripper, Mapping):
        return dict(base_gripper)
    if "gripper" in payload:
        raise ValueError(f"Profile {path} field 'gripper' must be an object.")
    return dict(payload)


def _profile_value(profile: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in profile:
            return profile[key]
    return None


def _resolved_gripper_policy(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_resolved_gripper_policy", None)
    if isinstance(cached, dict):
        return dict(cached)

    profile = _load_gripper_task_profile(args)
    policy: dict[str, Any] = dict(GRIPPER_POLICY_DEFAULTS)
    profile_keys = {
        "output_dim": ("output_dim", "gripper_slot"),
        "speed": ("speed",),
        "torque": ("torque", "force"),
        "max_action": ("max_action", "max_close"),
        "send_threshold": ("send_threshold",),
        "command_hz": ("command_hz", "max_command_hz"),
        "continuous_grasp": ("continuous_grasp", "continuous"),
        "telemetry_hz": ("telemetry_hz",),
        "telemetry_command_grace_s": ("telemetry_command_grace_s",),
        "tactile_fields": ("tactile_fields", "dmtac_tactile_fields"),
        "tactile_compute_profile": (
            "tactile_compute_profile",
            "dmtac_tactile_compute_profile",
        ),
    }
    for key, aliases in profile_keys.items():
        value = _profile_value(profile, *aliases)
        if value is not None:
            policy[key] = value

    cli_overrides = {
        "output_dim": getattr(args, "gripper_output_dim", None),
        "speed": getattr(args, "gripper_speed", None),
        "torque": getattr(args, "gripper_torque", None),
        "max_action": getattr(args, "gripper_max_action", None),
        "send_threshold": getattr(args, "gripper_send_threshold", None),
        "command_hz": getattr(args, "gripper_command_hz", None),
        "continuous_grasp": getattr(args, "gripper_continuous_grasp", None),
        "telemetry_hz": getattr(args, "gripper_telemetry_hz", None),
        "telemetry_command_grace_s": getattr(
            args, "gripper_telemetry_command_grace_s", None
        ),
    }
    for key, value in cli_overrides.items():
        if value is not None:
            policy[key] = value

    policy["output_dim"] = int(policy["output_dim"])
    policy["speed"] = float(policy["speed"])
    policy["torque"] = float(policy["torque"])
    policy["max_action"] = float(policy["max_action"])
    policy["send_threshold"] = float(policy["send_threshold"])
    policy["command_hz"] = float(policy["command_hz"])
    policy["continuous_grasp"] = bool(policy["continuous_grasp"])
    policy["telemetry_hz"] = float(policy["telemetry_hz"])
    policy["telemetry_command_grace_s"] = float(policy["telemetry_command_grace_s"])
    if policy.get("tactile_fields") is not None:
        policy["tactile_fields"] = normalize_dmtac_tactile_fields(
            policy["tactile_fields"]
        )
    if policy.get("tactile_compute_profile") is not None:
        policy["tactile_compute_profile"] = str(policy["tactile_compute_profile"])
    if int(policy["output_dim"]) != 0:
        raise ValueError("dmtac-gripper output_dim/gripper_slot must be 0.")
    if float(policy["send_threshold"]) < 0.0:
        raise ValueError("gripper send_threshold must be >= 0.")
    if float(policy["command_hz"]) <= 0.0:
        raise ValueError("gripper command_hz must be > 0.")
    if float(policy["telemetry_hz"]) <= 0.0:
        raise ValueError("gripper telemetry_hz must be > 0.")
    if float(policy["telemetry_command_grace_s"]) < 0.0:
        raise ValueError("gripper telemetry_command_grace_s must be >= 0.")
    setattr(args, "_resolved_gripper_policy", dict(policy))
    return dict(policy)


def _dmtac_tactile_enabled(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "dmtac_tactile", None)
    if explicit is not None:
        return bool(explicit)
    return True


def _apply_gripper_policy_to_dmtac_config(
    config: DMTacGripperBackendConfig,
    *,
    args: argparse.Namespace,
    policy: Mapping[str, Any],
) -> DMTacGripperBackendConfig:
    config.speed = float(policy["speed"])
    config.torque = float(policy["torque"])
    config.max_action = float(policy["max_action"])
    config.command_dim = int(policy["output_dim"])
    config.send_threshold = float(policy["send_threshold"])
    config.command_hz = float(policy["command_hz"])
    config.continuous_grasp = bool(policy["continuous_grasp"])
    config.telemetry_hz = float(policy["telemetry_hz"])
    config.gripper_telemetry_command_grace_s = float(
        policy.get(
            "telemetry_command_grace_s",
            DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
        )
    )
    tactile_timeout_ms = getattr(args, "dmtac_tactile_timeout_ms", None)
    if tactile_timeout_ms is not None:
        config.tactile_timeout_ms = int(tactile_timeout_ms)
    tactile_wait_for_new = getattr(args, "dmtac_tactile_wait_for_new", None)
    if tactile_wait_for_new is not None:
        config.tactile_wait_for_new = bool(tactile_wait_for_new)
    tactile_fields = getattr(args, "dmtac_tactile_fields", None)
    if tactile_fields is None:
        tactile_fields = policy.get("tactile_fields")
    if tactile_fields is not None:
        config.tactile_fields = normalize_dmtac_tactile_fields(tactile_fields)
    tactile_compute_profile = getattr(args, "dmtac_tactile_compute_profile", None)
    if tactile_compute_profile:
        config.tactile_compute_profile = str(tactile_compute_profile)
    elif policy.get("tactile_compute_profile"):
        config.tactile_compute_profile = str(policy["tactile_compute_profile"])
    if bool(getattr(args, "dmtac_tactile_sequential_read", False)):
        config.tactile_parallel_read = False
    if not _dmtac_tactile_enabled(args):
        config.connect_tactile = False
        config.tactile_ids = ()
    if bool(getattr(args, "rtc_real_shadow", False)):
        # Live raw-RTC shadow may read gripper/tactile observations, but its
        # startup must not clear faults, write speed/torque, open the gripper,
        # reset tactile calibration, or accept a command through a bypass API.
        config.observation_only = True
        config.send_commands = False
        config.clear_fault_on_start = False
        config.open_on_start = False
        config.tactile_reset_on_start = False
        config.tactile_warmup_frames = 0
    config.enable_telemetry = bool(config.connect_hardware or config.connect_tactile)
    return config


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
        setattr(args, "_dmtac_backend_configs", {})
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
                    tactile_backend=str(
                        getattr(args, "dmtac_tactile_backend", "cpu") or "cpu"
                    ),
                    speed=float(policy["speed"]),
                    torque=float(policy["torque"]),
                    max_action=float(policy["max_action"]),
                )
            )
        except Exception as exc:  # noqa: BLE001
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
    setattr(args, "_dmtac_backend_configs", dict(resolved))
    setattr(args, "dmtac_config_loaded_sides", sorted(resolved))
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
        choices=tuple(range(0, 101)),
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
            "Write deployment action/pose debug JSONL under "
            "teleop_records/deployment_debug by default."
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
        help=("RealSense warmup policy shared with formal real-teleop collection."),
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


def parse_policy_kwargs(text: str) -> dict[str, Any]:
    payload = json.loads(str(text or "{}"))
    if not isinstance(payload, dict):
        raise ValueError("--policy-kwargs-json must decode to a JSON object.")
    return payload


def _hands_enabled(args: argparse.Namespace) -> bool:
    if args.hands_enabled is not None:
        return bool(args.hands_enabled)
    if any(
        _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
        for side in ALL_SIDES
    ):
        return True
    return bool(args.left_hand_ip or args.right_hand_ip or args.visualize_tactile)


def _position_deploy_ik_tuning(args: argparse.Namespace):
    """Bind the formal action-to-qref objective without source-side shaping."""

    return build_rm65_pose_lift_ik_tuning(
        position_control_profile=(
            str(getattr(args, "position_control_profile_resolved", ""))
            or RM65_BIMANUAL_POSITION_V1_NAME
        ),
        pose_lift_profile=str(args.pose_lift_profile),
        orientation_weight=float(args.position_ik_orientation_weight),
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
    if continuous_waypoint_research:
        if not bool(args.debug) and not no_debug_deadline_soak:
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
    setattr(args, "_rtc_debug_level", rtc_debug_level)
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
    setattr(args, "_real_arm_execution_config", execution_config)
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
    if args.policy_qpos_init_dataset or args.policy_qpos_init_parquet:
        if not bool(args.arms_enabled):
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
            token == "--arm-execution-profile"
            or token.startswith("--arm-execution-profile=")
            or token == "--arm-control-mode"
            or token.startswith("--arm-control-mode=")
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
            if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_INSPIRE:
                if not getattr(args, f"{side}_hand_ip"):
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


def _runtime_action_keyframe_period_sec(args: argparse.Namespace) -> float:
    """Bind RTC runtime timing without changing the legacy sync default."""

    profile_id = str(getattr(args, "rtc_profile", "off")).strip()
    if profile_id == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1:
        return 1.0 / RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.action_hz
    if profile_id == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
        return 1.0 / RAW_RTC_FINAL_CHUNK_PROFILE_V1.action_hz
    if profile_id == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1:
        return 1.0 / MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1.action_hz
    # Preserve the historical PolicyDeployRuntimeConfig default for sync_queue_v1.
    return 1.0 / 15.0


def _action_hz_reference(
    args: argparse.Namespace,
) -> tuple[Path | None, Path | None, int, int, int | None]:
    dataset = args.action_hz_dataset or args.debug_replay_dataset
    parquet = args.action_hz_parquet or args.debug_replay_parquet
    episode = (
        int(args.action_hz_episode)
        if args.action_hz_episode is not None
        else int(args.debug_replay_episode)
    )
    start_index = (
        int(args.action_hz_start_index)
        if args.action_hz_start_index is not None
        else int(args.debug_replay_start_index)
    )
    end_index = (
        int(args.action_hz_end_index)
        if args.action_hz_end_index is not None
        else args.debug_replay_end_index
    )
    return dataset, parquet, episode, start_index, end_index


def resolve_effective_action_hz(
    args: argparse.Namespace, policy: Any | None = None
) -> tuple[float, dict[str, Any]]:
    explicit = float(args.action_hz)
    if explicit > 0.0:
        return explicit, {"source": "cli.action_hz", "hz": explicit}

    if policy is not None:
        estimate = getattr(policy, "metadata", {}).get("action_hz_estimate")
        if isinstance(estimate, Mapping):
            hz = float(estimate.get("hz", 0.0) or 0.0)
            if hz > 0.0:
                return hz, dict(estimate)

    dataset, parquet, episode, start_index, end_index = _action_hz_reference(args)
    if dataset or parquet:
        path, metadata = resolve_replay_parquet_path(
            dataset_root=dataset,
            parquet_path=parquet,
            episode_index=int(episode),
        )
        estimate = infer_replay_action_hz(
            parquet_path=path,
            dataset_root=dataset,
            episode_index=int(episode),
            start_index=int(start_index),
            end_index=end_index,
            resolved_metadata=metadata,
        )
        hz = float(estimate.get("hz", 0.0) or 0.0)
        if hz > 0.0:
            estimate = dict(estimate)
            estimate["source_path"] = str(path)
            return hz, estimate

    default_hz = float(args.hz)
    return default_hz, {"source": "cli.hz", "hz": default_hz}


def resolve_effective_policy_hz(
    args: argparse.Namespace, *, action_hz: float, action_hz_source: str
) -> tuple[float, str]:
    if args.policy_hz is not None:
        return float(args.policy_hz), "cli.policy_hz"
    return float(action_hz), "action_hz"


def resolve_arm_step_dt_max_scale(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float | None = None,
) -> tuple[float, bool]:
    explicit = getattr(args, "arm_step_dt_max_scale", None)
    if explicit is not None:
        return float(explicit), False
    target_hz = float(action_hz or 0.0)
    if target_hz <= 0.0:
        target_hz = float(
            getattr(args, "hz", DEFAULT_TELEOP_SAMPLE_HZ) or DEFAULT_TELEOP_SAMPLE_HZ
        )
    target_hz = max(1e-6, target_hz)
    required_scale = 1.1 * max(1e-6, float(control_hz)) / target_hz
    return max(DEPLOY_ARM_STEP_DT_MAX_SCALE, required_scale), True


def apply_arm_step_dt_defaults(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float | None = None,
) -> None:
    max_scale, auto = resolve_arm_step_dt_max_scale(
        args,
        control_hz=float(control_hz),
        action_hz=action_hz,
    )
    setattr(args, "arm_step_dt_max_scale", float(max_scale))
    setattr(args, "arm_step_dt_max_scale_auto", bool(auto))


def deploy_arm_command_step_dt_sec(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float,
) -> float:
    base_dt = 1.0 / max(1e-6, float(control_hz))
    if not bool(getattr(args, "arm_step_dt_scaling", True)):
        return base_dt
    action_dt = 1.0 / max(1e-6, float(action_hz))
    max_scale = float(
        getattr(args, "arm_step_dt_max_scale", DEPLOY_ARM_STEP_DT_MAX_SCALE)
    )
    return float(np.clip(action_dt, base_dt, base_dt * max_scale))


def deploy_arm_backend_control_dt_sec(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float | None,
) -> float:
    arm_control_mode = normalize_rm65_arm_control_mode(
        getattr(args, "arm_control_mode", None)
    )
    if arm_control_mode != RM65_ARM_CONTROL_MODE_POSITION:
        return 1.0 / max(1e-6, float(control_hz))
    del action_hz
    return 1.0 / max(1e-6, effective_position_reference_hz(args))


def deploy_arm_command_max_joint_step(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float,
) -> np.ndarray:
    step_dt = deploy_arm_command_step_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    return np.asarray(args.arm_max_angvel, dtype=float).reshape(6) * step_dt


def annotate_deploy_arm_command_limits(
    command: Any,
    *,
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    control_hz: float,
    action_hz: float,
) -> None:
    if command is None:
        return
    commands = getattr(command, "commands", {}) or {}
    if not commands:
        return
    step_dt = deploy_arm_command_step_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    max_joint_step = deploy_arm_command_max_joint_step(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    backend_control_dt = deploy_arm_backend_control_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    for side in active_sides:
        effector_command = commands.get(f"{side}_arm")
        if effector_command is None:
            continue
        effector_command.metadata["max_joint_step_rad"] = max_joint_step.copy()
        effector_command.metadata["deploy_max_joint_step_rad"] = max_joint_step.copy()
        effector_command.metadata["deploy_action_step_dt_sec"] = float(step_dt)
        effector_command.metadata["deploy_backend_control_dt_sec"] = float(
            backend_control_dt
        )
        effector_command.metadata["deploy_arm_step_dt_scaling"] = bool(
            getattr(args, "arm_step_dt_scaling", True)
        )
        effector_command.metadata["deploy_arm_step_dt_max_scale"] = float(
            getattr(args, "arm_step_dt_max_scale", DEPLOY_ARM_STEP_DT_MAX_SCALE)
        )


def command_effector_kind_subset(
    command: TeleopCommand | None,
    *,
    kind: str,
) -> TeleopCommand | None:
    """Copy only arm or hand commands while preserving command metadata."""

    if command is None:
        return None
    suffix = f"_{str(kind).strip()}"
    subset = TeleopCommand(
        timestamp=command.timestamp,
        metadata=dict(command.metadata),
    )
    for effector_name, effector_command in command.commands.items():
        if str(effector_name).endswith(suffix):
            subset.add_command(effector_command.copy())
    return subset if subset.commands else None


def _dispatch_position_discrete_command_checked(
    command: TeleopCommand,
    *,
    hand_backends: Mapping[str, EffectorBackend],
) -> Mapping[str, Any]:
    """Dispatch one reference-clock-aligned hand keyframe and preserve receipts.

    Hand backends historically used both exceptions and return values to
    report I/O failures.  Position deploy needs an explicit result so the
    asynchronous executor can distinguish a safe no-op/rate skip from a real
    rejection and hold the arm stream on the latter.
    """

    receipts: dict[str, dict[str, Any]] = {}
    accepted_all = True
    skipped_any = False
    for effector_name, effector_command in command.commands.items():
        side = str(effector_command.side or "")
        backend = hand_backends.get(side)
        if backend is None:
            raise KeyError(
                f"Deploy backend missing for discrete effector {effector_name!r}."
            )
        started = time.monotonic()
        try:
            result = backend.send_command(effector_command)
            state = (
                backend.get_state(
                    include_tactile_arrays=False,
                    include_tactile_vector=False,
                )
                if isinstance(backend, DMTacGripperBackend)
                else backend.get_state()
            )
        except Exception as exc:
            receipt = {
                "accepted": False,
                "skipped": False,
                "fatal": True,
                "status": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "duration_ms": (time.monotonic() - started) * 1000.0,
            }
            receipts[str(effector_name)] = receipt
            command.metadata["position_discrete_backend_receipts"] = dict(receipts)
            effector_command.metadata["position_discrete_backend_receipt"] = receipt
            raise RuntimeError(
                f"{effector_name} discrete dispatch failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        status = "accepted"
        accepted = result is not False
        skipped = False
        if isinstance(backend, DMTacGripperBackend):
            status = str(state.get("gripper_last_send_status") or "unknown")
            if status == "rate_limited":
                accepted = False
                skipped = True
            elif status == "intentionally_disconnected_noop":
                accepted = True
                skipped = True
            elif status == "skipped_threshold":
                # The requested value is already within the configured no-op
                # threshold; no additional physical packet is required.
                accepted = True
            elif status != "sent" and result is False:
                accepted = False
        elif isinstance(backend, InspireTacHandBackend):
            status = "sent_or_threshold_noop" if result is not False else "rejected"

        fatal = bool(not accepted and not skipped)
        receipt = {
            "accepted": bool(accepted),
            "skipped": bool(skipped),
            "fatal": bool(fatal),
            "status": status,
            "backend_type": type(backend).__name__,
            "backend_ready": bool(backend.is_ready()),
            "telemetry_error": state.get("telemetry_error"),
            "duration_ms": (time.monotonic() - started) * 1000.0,
        }
        receipts[str(effector_name)] = receipt
        effector_command.metadata["position_discrete_backend_receipt"] = receipt
        command.metadata["position_discrete_backend_receipts"] = dict(receipts)
        if fatal:
            raise RuntimeError(
                f"{effector_name} discrete dispatch rejected: "
                f"status={status} telemetry_error={state.get('telemetry_error')}"
            )
        accepted_all = accepted_all and bool(accepted)
        skipped_any = skipped_any or bool(skipped)
    return {
        "accepted": bool(accepted_all),
        "skipped": bool(skipped_any),
        "fatal": False,
        "effectors": receipts,
    }


def hold_deploy_arm_backends(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    reason: str,
) -> None:
    """Request all configured real arm backends to hold before reporting errors."""

    errors: list[str] = []
    for side, backend in arm_backends.items():
        try:
            backend.hold_motion(str(reason))
        except Exception as exc:  # noqa: BLE001 - attempt every active side.
            errors.append(f"{side}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("deploy arm hold failed: " + "; ".join(errors))


def request_policy_inference_soft_hold(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    reason: str,
) -> dict[str, dict[str, object]]:
    """Freeze measured arm state while synchronous policy inference is pending.

    The hook is optional so position-only and non-RM65 backends retain their
    existing semantics.  An implemented backend must fail closed when it
    cannot produce a valid measured snapshot.
    """

    receipts: dict[str, dict[str, object]] = {}
    for side, backend in arm_backends.items():
        request = getattr(backend, "request_policy_inference_soft_hold", None)
        if not callable(request):
            receipts[str(side)] = {
                "active": False,
                "status": "unsupported",
                "backend_type": type(backend).__name__,
            }
            continue
        receipt = request(str(reason))
        receipts[str(side)] = dict(receipt or {})
    return receipts


def _defer_position_action_after_synchronous_prediction(
    *,
    prediction_accepted: bool,
    position_executor_enabled: bool,
) -> bool:
    """Keep a newly predicted position chunk for the next observation cut.

    Legacy policy inference is synchronous.  A slow inference therefore ages
    the observation that was used to request it, even though the model output
    itself remains valid.  The first position command must not reuse that
    frozen observation as the authoritative collision-planner reset state.
    Deferring consumption by one control-loop iteration lets the frontend
    acquire a fresh dual-arm measurement without weakening collision-reset
    freshness or changing the queued action chunk.
    """

    return bool(prediction_accepted and position_executor_enabled)


def _prepare_position_plan_cut_for_action_step(
    detector: DeploymentPositionPlanCutDetector,
    action_step: Any,
    *,
    rtc_enabled: bool,
    policy_node_id: str,
    hold_generation: int,
    detected_at: float,
):
    """Build the mandatory typed cut from the owning source cursor.

    RTC owns a committed timed emission.  The synchronous compatibility path
    instead owns a shared queue cursor and its attached chunk metadata.  Its
    optional passive emission must not become motion authority.
    """

    detector.observe_hold_generation(int(hold_generation))
    if rtc_enabled:
        emission = getattr(action_step, "emission", None)
        if emission is None:
            raise RuntimeError(
                "RTC position action is missing its committed emission lineage; "
                "motion submission is blocked before the executor."
            )
        return detector.prepare(
            emission,
            detected_at=detected_at,
        )
    raw_action = getattr(action_step, "raw_action", None)
    if raw_action is None:
        raise RuntimeError(
            "Synchronous position action is missing its queue-owned source "
            "lineage; motion submission is blocked before the executor."
        )
    return detector.prepare_synchronous_action(
        raw_action,
        policy_node_id=policy_node_id,
        detected_at=detected_at,
    )


def _hold_active_position_stream_for_synchronous_chunk_gap(
    position_executor: PositionDeployExecutor | None,
) -> bool:
    """End an exhausted synchronous chunk without faulting its executor.

    The legacy OpenPI client performs inference synchronously.  With a full
    chunk execution policy, the position stream can therefore be active while
    the policy queue has just become empty.  Waiting for inference inside that
    active stream would trip the executor's missing-keyframe watchdog.  Clear
    the stream first so the arms enter the existing bounded Hold path; the next
    chunk must then establish a fresh collision reset before it can move.
    """

    if position_executor is None:
        return False
    tail_status = position_executor.source_tail_drain_status()
    if bool(tail_status.get("required")) and not bool(tail_status.get("drained")):
        return False
    if not bool(position_executor.snapshot().get("active")):
        return False
    position_executor.clear()
    return True


def _position_source_tail_pending_before_synchronous_chunk_gap(
    position_executor: PositionDeployExecutor | None,
) -> bool:
    """Keep the upper chunk boundary open until motion emitted its tail."""

    if position_executor is None:
        return False
    status = position_executor.source_tail_drain_status()
    return bool(status.get("required")) and not bool(status.get("drained"))


def emergency_stop_deploy_arm_backends(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    reason: str,
) -> None:
    """Request a concurrent, latched emergency stop on every active arm."""

    errors = bimanual_emergency_stop(arm_backends, reason=str(reason))
    if errors:
        raise RuntimeError(
            "deploy bimanual emergency stop was not acknowledged by all active arms: "
            + "; ".join(f"{side}={detail}" for side, detail in sorted(errors.items()))
        )


def emergency_stop_position_deploy_on_keyboard_interrupt(
    args: argparse.Namespace,
    *,
    send_actions: bool,
    arm_backends: Mapping[str, EffectorBackend],
) -> bool:
    """Request Arm_Stop for an active real-position deployment.

    Returns False for dry-run, arm-disabled, non-position, or backend-free
    processes so Ctrl+C retains its ordinary resource-cleanup behavior there.
    Any incomplete real-arm acknowledgement is intentionally propagated.
    """

    if not (
        bool(send_actions)
        and bool(getattr(args, "arms_enabled", False))
        and normalize_rm65_arm_control_mode(getattr(args, "arm_control_mode", None))
        == RM65_ARM_CONTROL_MODE_POSITION
        and arm_backends
    ):
        return False
    emergency_stop_deploy_arm_backends(
        arm_backends,
        reason="keyboard_interrupt",
    )
    return True


def build_arm_backends(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    *,
    control_hz: float | None,
    action_hz: float | None = None,
) -> dict[str, EffectorBackend]:
    if not bool(args.arms_enabled):
        return {}
    arm_urdf_path = str(args.arm_urdf_path or require_asset_path("rm65_arm.urdf"))
    control_hz = (
        effective_arm_control_hz(args) if control_hz is None else float(control_hz)
    )
    control_dt = 1.0 / float(control_hz)
    backend_control_dt = deploy_arm_backend_control_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=action_hz,
    )
    arm_control_mode = normalize_rm65_arm_control_mode(
        getattr(args, "arm_control_mode", None)
    )
    worker_binding = getattr(args, "_rm65_legacy_worker_binding", None)
    if worker_binding is not None:
        if not isinstance(worker_binding, RM65LegacyWorkerExecutionBinding):
            raise TypeError("_rm65_legacy_worker_binding has an unsupported type")
        if tuple(active_sides) != ("left", "right"):
            raise ValueError(
                "RM65 arms-only worker binding requires coupled left/right sides"
            )
        if arm_control_mode != RM65_ARM_CONTROL_MODE_POSITION:
            raise ValueError("RM65 arms-only worker binding excludes impedance control")
    command_input_hz = (
        effective_position_reference_hz(args)
        if arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION
        else action_hz
    )
    apply_arm_step_dt_defaults(
        args,
        control_hz=float(control_hz),
        action_hz=command_input_hz,
    )
    arm_canfd_follow = effective_arm_canfd_follow(args)
    observation_only = bool(getattr(args, "rtc_real_shadow", False))
    if observation_only and bool(getattr(args, "send_actions", False)):
        raise ValueError(
            "rtc real shadow backend construction requires send_actions=False"
        )
    if observation_only:
        # A read-only shadow must not accidentally inherit a force/CANFD
        # execution setting from the checked-in lab profile.
        arm_canfd_follow = False
    backends: dict[str, EffectorBackend] = {}
    for side in active_sides:
        arm_ip = str(getattr(args, f"{side}_arm_ip") or "")
        if arm_control_mode == RM65_ARM_CONTROL_MODE_IMPEDANCE:
            backends[side] = RM65ImpedanceArmBackend(
                RM65ImpedanceArmBackendConfig(
                    arm_ip=arm_ip or "0.0.0.0",
                    side=side,
                    connect_hardware=bool(arm_ip),
                    observation_only=observation_only,
                    tool_profile=getattr(args, f"{side}_tool_profile", "") or None,
                    strict_tool_profile=bool(
                        getattr(args, "strict_tool_profile", True)
                    ),
                    impedance_mode="pose",
                    force_tare_sec=float(getattr(args, "force_tare_sec", 3.0)),
                    control_dt=control_dt,
                    action_hz=(None if action_hz is None else float(action_hz)),
                    max_angvel=tuple(float(value) for value in args.arm_max_angvel),
                    actual_send_dt_alpha=float(args.arm_step_dt_alpha),
                    actual_send_dt_min_scale=1.0,
                    actual_send_dt_max_scale=float(args.arm_step_dt_max_scale),
                    actual_send_dt_stall_reset_sec=float(
                        args.arm_step_dt_stall_reset_sec
                    ),
                    debug_timing=bool(getattr(args, "debug", False)),
                    # Force-policy deployment follows the impedance-only
                    # safety envelope.  The position branch below keeps its
                    # established planner and timing behavior unchanged.
                    impedance_overrides={
                        "canfd_follow": bool(arm_canfd_follow),
                        "joint_rate_limit_use_nominal_dt": True,
                        "admittance_use_nominal_dt": True,
                        "joint_operating_margin_deg": 3.0,
                        "ik_hard_deadline_enabled": True,
                        # Deployment has a 15 Hz policy cadence but must keep
                        # force/admittance and safe output at 40 Hz.  The
                        # controller therefore owns full Pinocchio IK on a
                        # separate 15 Hz worker; its bounded 40 Hz local path
                        # continues while that advisory solve is in flight.
                        # The historical fail-closed preemption behavior stays
                        # available as an explicit rollback/debug switch.
                        "deployment_dual_rate_ik_enabled": True,
                        "deployment_full_ik_hz": 15.0,
                        "deployment_full_ik_deadline_ms": 66.66666666666667,
                        "deployment_local_ik_deadline_ms": 6.0,
                        "deployment_full_ik_preempts_local": bool(
                            getattr(args, "deployment_full_ik_preempts_local", False)
                        ),
                        "ik_resolver_candidate_provider_names": ["rm65_pinocchio"],
                        "ik_resolver_local_step_provider_names": ["rm65_pinocchio"],
                        "ik_resolver_local_step_max_position_worsen_m": 0.003,
                        "ik_resolver_local_step_max_orientation_worsen_deg": 0.5,
                        "ik_resolver_local_step_dynamic_worsen_enabled": False,
                        "direct_joint_hint_max_linear_offset_m": 0.0,
                        "direct_joint_hint_max_angular_offset_rad": 0.0,
                    },
                )
            )
            continue
        binding_kwargs = (
            {}
            if worker_binding is None
            else worker_binding.arm_backend_kwargs(
                side,
                connect_hardware=bool(arm_ip),
            )
        )
        backends[side] = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip=arm_ip or "0.0.0.0",
                side=side,
                arm_urdf_path=arm_urdf_path,
                connect_hardware=bool(arm_ip),
                observation_only=observation_only,
                control_dt=backend_control_dt,
                max_angvel=tuple(float(value) for value in args.arm_max_angvel),
                singularity_qpos_init_slowdown=float(
                    args.singularity_qpos_init_slowdown
                ),
                dry_run_fast=not bool(arm_ip),
                hardware_read_retries=int(args.rm65_hardware_read_retries),
                hardware_read_retry_delay_sec=float(
                    args.rm65_hardware_read_retry_delay_sec
                ),
                initial_qpos=getattr(args, f"{side}_initial_qpos"),
                tool_profile=getattr(args, f"{side}_tool_profile", "") or None,
                ik_framework=str(getattr(args, "rm65_ik_framework", "robot_motion")),
                use_actual_send_dt_scaling=bool(
                    getattr(args, "arm_step_dt_scaling", True)
                ),
                actual_send_dt_alpha=float(args.arm_step_dt_alpha),
                actual_send_dt_min_scale=1.0,
                actual_send_dt_max_scale=float(args.arm_step_dt_max_scale),
                actual_send_dt_stall_reset_sec=float(args.arm_step_dt_stall_reset_sec),
                background_streaming_enabled=bool(arm_ip) and bool(arm_canfd_follow),
                background_streaming_hz=float(control_hz),
                background_streaming_target_admission_mode=str(
                    getattr(
                        args,
                        "position_background_target_admission",
                        RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
                    )
                ),
                position_command_planner=str(args.position_command_planner),
                position_velocity_ramp_sec=float(args.position_velocity_ramp_sec),
                position_target_velocity_filter_tau_sec=float(
                    args.position_target_velocity_filter_tau_sec
                ),
                position_max_acceleration_rad_s2=tuple(
                    float(value) for value in args.position_max_acceleration_rad_s2
                ),
                position_max_command_step_deg=float(args.position_max_command_step_deg),
                position_degraded_speed_scale=float(args.position_degraded_speed_scale),
                arm_follow_qpos_alpha=float(args.arm_follow_qpos_alpha),
                canfd_follow=bool(arm_canfd_follow),
                canfd_trajectory_mode=int(
                    getattr(args, "arm_canfd_trajectory_mode", 0)
                ),
                canfd_radio=int(getattr(args, "arm_canfd_radio", 0)),
            ),
            **binding_kwargs,
        )
    return backends


def build_hand_backends(
    args: argparse.Namespace, active_sides: tuple[str, ...]
) -> dict[str, EffectorBackend]:
    if not active_sides or not _hands_enabled(args):
        return {}
    configs = _dmtac_backend_configs(args, active_sides)
    backends: dict[str, EffectorBackend] = {
        side: DMTacGripperBackend(config)
        for side, config in configs.items()
        if side in active_sides
        and _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
    }
    disconnected_sides = _parse_disconnected_gripper_sides(
        getattr(args, "disconnected_gripper_sides", ())
    )
    if disconnected_sides:
        policy = _resolved_gripper_policy(args)
        for side in disconnected_sides:
            if side not in active_sides:
                continue
            backends[side] = DMTacGripperBackend(
                DMTacGripperBackendConfig(
                    side=side,
                    effector_name=f"{side}_hand",
                    connect_hardware=False,
                    connect_tactile=False,
                    send_commands=False,
                    enable_telemetry=False,
                    command_dim=int(policy["output_dim"]),
                    max_action=float(policy["max_action"]),
                    disconnected_state_scalar=float(
                        args.disconnected_gripper_state
                    ),
                )
            )

    tactile_regions = parse_tactile_regions(args.hand_tactile_regions)
    observation_only = bool(getattr(args, "rtc_real_shadow", False))
    for side in active_sides:
        if _hand_backend_mode_for_side(args, side) != HAND_BACKEND_INSPIRE:
            continue
        hand_ip = str(getattr(args, f"{side}_hand_ip") or "")
        backends[side] = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=hand_ip or "0.0.0.0",
                port=int(args.hand_modbus_port),
                side=side,
                effector_name=f"{side}_hand",
                connect_hardware=bool(hand_ip),
                observation_only=observation_only,
                max_fingertip_force=int(args.max_fingertip_force),
                send_threshold=float(args.hand_send_threshold),
                enable_telemetry=bool(hand_ip),
                telemetry_hz=float(args.hand_telemetry_hz),
                tactile_regions=tuple(tactile_regions),
            )
        )
    return backends


def _argument_was_supplied(raw_argv: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in raw_argv)


def resolve_policy_camera_input_keys(
    args: argparse.Namespace,
    policy_artifact: Any | None,
    *,
    raw_argv: list[str],
) -> tuple[str, ...] | None:
    """Resolve artifact-bound logical cameras before any device is opened.

    ``None`` deliberately means a legacy artifact with no formal visual
    contract, preserving its existing runtime camera behavior.  A tuple means
    the exact complete set that may be captured and forwarded to the policy.
    """

    camera_input = None if policy_artifact is None else policy_artifact.camera_input
    if camera_input is None:
        return None
    selected_keys = camera_input.source_keys
    if not selected_keys:
        if args.camera_mode != "none":
            raise ValueError(
                "policy artifact declares no camera inputs; use --camera-mode none."
            )
        return selected_keys
    if args.camera_mode == "none":
        if camera_input.required:
            raise ValueError(
                "policy artifact requires camera inputs "
                f"{list(selected_keys)}; choose --camera-mode single or multi."
            )
        return selected_keys
    if args.camera_serial:
        if len(selected_keys) != 1:
            raise ValueError(
                "--camera-serial can satisfy only one artifact camera input; "
                f"artifact requires {list(selected_keys)}."
            )
        selected_key = selected_keys[0]
        if (
            _argument_was_supplied(raw_argv, "--camera-key")
            and args.camera_key != selected_key
        ):
            raise ValueError(
                "--camera-key disagrees with policy artifact camera input: "
                f"expected {selected_key!r}, got {args.camera_key!r}."
            )
        # The default CLI key is ``front``.  For a wrist-only artifact it is
        # safe to substitute the artifact's logical key before source creation.
        args.camera_key = selected_key
    elif args.camera_mode == "single" and len(selected_keys) != 1:
        raise ValueError(
            "--camera-mode single requires exactly one artifact camera input; "
            f"artifact requires {list(selected_keys)}."
        )
    return selected_keys


def build_camera_sources(
    args: argparse.Namespace,
    *,
    selected_camera_keys: tuple[str, ...] | None = None,
) -> dict[str, RealSenseRGBSource]:
    if args.camera_mode == "none":
        return {}
    if selected_camera_keys == ():
        raise ValueError(
            "A policy artifact without camera inputs requires --camera-mode none."
        )
    if args.camera_serial:
        if args.camera_mode != "single":
            raise ValueError(
                "--camera-serial can only be used with --camera-mode single."
            )
        config = RealSenseRGBCameraConfig(
            serial=str(args.camera_serial),
            width=int(args.camera_width),
            height=int(args.camera_height),
            fps=int(args.camera_fps),
            color_format=str(getattr(args, "camera_color_format", "bgr8")),
            camera_key=str(args.camera_key),
            keep_last=int(args.camera_keep_last),
            color_space=str(args.camera_color_space),
            wait_timeout_ms=int(args.camera_wait_timeout_ms),
            stale_after_sec=float(args.camera_stale_after_sec),
            auto_exposure_priority=args.camera_auto_exposure_priority,
            startup_hardware_reset=bool(
                getattr(args, "camera_startup_hardware_reset", False)
            ),
            startup_warmup_policy=str(
                getattr(
                    args,
                    "camera_startup_warmup_policy",
                    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
                )
            ),
            metadata={"source": "teleop_runtime_deploy_cli"},
        )
        if (
            selected_camera_keys is not None
            and config.camera_key not in selected_camera_keys
        ):
            raise ValueError(
                f"--camera-key {config.camera_key!r} is not selected by policy artifact "
                f"{list(selected_camera_keys)}."
            )
        return {config.camera_key: RealSenseRGBSource(config)}
    camera_configs = load_realsense_camera_config_map(Path(args.camera_config))
    if selected_camera_keys is not None:
        missing_keys = tuple(
            key for key in selected_camera_keys if key not in camera_configs
        )
        if missing_keys:
            raise ValueError(
                "--camera-config does not provide policy artifact camera inputs: "
                f"{list(missing_keys)}."
            )
        selected = set(selected_camera_keys)
        camera_configs = {
            key: config for key, config in camera_configs.items() if key in selected
        }
    real_camera_configs = {
        key: config
        for key, config in camera_configs.items()
        if isinstance(config, RealSenseRGBCameraConfig)
    }
    apply_real_teleop_camera_startup_defaults(
        real_camera_configs,
        color_format=str(getattr(args, "camera_color_format", "bgr8")),
        warmup_policy=str(
            getattr(
                args,
                "camera_startup_warmup_policy",
                REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
            )
        ),
    )
    # Config files historically omitted keep_last, which left deployment with
    # the source default of two frames even though --camera-keep-last already
    # defaulted to four for the single-camera CLI path.  Apply the deployment
    # setting to config-backed sources as a minimum while preserving a larger
    # value explicitly requested by a config file.
    requested_keep_last = int(
        getattr(args, "camera_keep_last", DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST)
    )
    for config in camera_configs.values():
        if isinstance(config, RealSenseRGBCameraConfig):
            config.keep_last = max(int(config.keep_last), requested_keep_last)
    if bool(getattr(args, "camera_startup_hardware_reset", False)):
        for config in camera_configs.values():
            config.startup_hardware_reset = True
    if args.camera_mode == "single" and len(camera_configs) != 1:
        raise ValueError(
            f"--camera-mode single expects exactly one camera config, got {len(camera_configs)}."
        )
    if args.camera_mode == "multi" and not camera_configs:
        raise ValueError("--camera-mode multi requires at least one camera config.")
    return {key: RealSenseRGBSource(config) for key, config in camera_configs.items()}


def start_camera_sources(
    camera_sources: Mapping[str, RealSenseRGBSource],
    *,
    stagger_sec: float = REAL_TELEOP_CAMERA_START_STAGGER_SEC,
) -> None:
    """Use the formal real-teleop collection camera startup lifecycle."""

    errors = _start_real_teleop_camera_sources_best_effort(
        camera_sources,
        stagger_sec=float(stagger_sec),
    )
    if errors:
        detail = "; ".join(
            f"{camera_key}={errors[camera_key]}" for camera_key in sorted(errors)
        )
        raise RuntimeError(
            "RealSense startup failed after the formal real-teleop deferred "
            f"retry lifecycle: {detail}"
        )


def run_camera_startup_self_check(
    camera_sources: Mapping[str, RealSenseRGBSource],
    *,
    duration_sec: float = CAMERA_STARTUP_SELF_CHECK_DURATION_SEC,
    interval_sec: float = CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC,
) -> None:
    """Run the formal collection consecutive-fresh-frame admission headlessly."""

    if not camera_sources:
        return
    started_at = time.time()
    duration_sec = max(0.0, float(duration_sec))
    interval_sec = max(0.01, float(interval_sec))
    freshness_gate = CameraStartupFreshnessGate(
        tuple(str(camera_key) for camera_key in camera_sources)
    )
    final_status: Mapping[str, object] = {}
    while True:
        now = time.time()
        final_status = _build_camera_status(camera_sources, now=now)
        if freshness_gate.update(final_status):
            print(
                "[CameraSelfCheck] formal real-teleop admission passed: "
                f"cameras={sorted(camera_sources)} "
                f"consecutive={CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES} "
                f"max_age_sec={CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC:.3f}",
                flush=True,
            )
            return
        elapsed_sec = now - started_at
        if elapsed_sec >= duration_sec:
            status_detail = "; ".join(
                f"{camera_key}={final_status.get(camera_key)}"
                for camera_key in sorted(camera_sources)
            )
            raise RuntimeError(
                "RealSense formal real-teleop startup self-check failed: "
                f"elapsed={elapsed_sec:.3f}s "
                f"streaks={freshness_gate.fresh_frame_streaks}; {status_detail}"
            )
        sleep_sec = min(interval_sec, max(0.0, duration_sec - elapsed_sec))
        if sleep_sec > 0.0:
            time.sleep(sleep_sec)


def select_preview_camera_sources(
    camera_sources: dict[str, RealSenseRGBSource], requested_key: str
) -> dict[str, RealSenseRGBSource]:
    if not camera_sources:
        return {}
    key = str(requested_key or "").strip()
    if key:
        if key not in camera_sources:
            available = ", ".join(sorted(camera_sources))
            raise ValueError(
                f"--vis-cam-key {key!r} not found. Available camera keys: {available}"
            )
        return {key: camera_sources[key]}
    selected_key = "front" if "front" in camera_sources else sorted(camera_sources)[0]
    return {selected_key: camera_sources[selected_key]}


def compact_log_value(value: object, *, max_chars: int = 90) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def camera_status_text(
    camera_sources: dict[str, RealSenseRGBSource],
    camera_previewer: CameraPreviewer | None,
) -> str:
    if not camera_sources:
        return "camera=off"
    bits: list[str] = []
    if camera_previewer is not None and camera_previewer.disabled_reason:
        bits.append(
            "preview=disabled:"
            + compact_log_value(camera_previewer.disabled_reason, max_chars=80)
        )
    for camera_key, camera_source in sorted(camera_sources.items()):
        try:
            info = camera_source.device_info()
        except Exception as exc:  # noqa: BLE001
            bits.append(f"{camera_key}=status_error:{type(exc).__name__}")
            continue
        frame_count = int(info.get("frame_count", 0) or 0)
        last_error = info.get("last_error")
        if last_error:
            bits.append(
                f"{camera_key}=frames:{frame_count},err:{compact_log_value(last_error, max_chars=70)}"
            )
        else:
            bits.append(f"{camera_key}=frames:{frame_count}")
    return "camera=" + ";".join(bits)


def hand_state_snapshot(
    backends: Mapping[str, EffectorBackend],
    *,
    history_poller: HandStateHistoryPoller | None = None,
    timestamp: float | None = None,
) -> dict[str, dict[str, Any]]:
    if history_poller is not None and timestamp is not None:
        return history_poller.latest_by_side(float(timestamp))
    return {side: backend.get_state() for side, backend in backends.items()}


def hand_action_specs_for_backends(
    args: argparse.Namespace,
    backends: Mapping[str, EffectorBackend],
    active_sides: tuple[str, ...],
) -> dict[str, Any]:
    specs = {}
    for side in active_sides:
        if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER:
            specs[side] = HandActionSpec.dmtac_gripper(side)
        else:
            specs[side] = HandActionSpec.inspire_tac(side)
    return hand_action_specs_metadata(specs)


def policy_qpos_init_hand_action_specs(
    args: argparse.Namespace,
    hand_action_specs: Mapping[str, Any],
) -> dict[str, Any]:
    """Exclude no-I/O grippers from physical dataset-start initialization."""

    disconnected_sides = set(
        _parse_disconnected_gripper_sides(
            getattr(args, "disconnected_gripper_sides", ())
        )
    )
    return {
        side: spec
        for side, spec in hand_action_specs.items()
        if side not in disconnected_sides
    }


def print_controls(
    *,
    send_actions: bool,
    auto_start: bool,
    wait_for_ready_confirm: bool,
    fixed_init_enabled: bool = False,
    policy_qpos_init_enabled: bool = False,
) -> None:
    print("\n" + "=" * 60)
    print("teleop_runtime policy deployment loop started")
    print(f"  send_actions={'on' if send_actions else 'off(dry-run)'}")
    if (fixed_init_enabled or policy_qpos_init_enabled) and auto_start:
        initial_policy_state = "paused(auto-start after init)"
    else:
        initial_policy_state = "active" if auto_start else "paused"
    print(f"  initial_policy_state={initial_policy_state}")
    if wait_for_ready_confirm:
        print("  ready_confirm=on(A/a required after startup init)")
    if fixed_init_enabled:
        print("  fixed_init=on(H first-frame qpos, H execute T, then startup init)")
        print(
            "Controls: H=pause policy or fixed-init step, g=start after fixed-init, r=manual safety re-arm, A=confirm, q=quit"
        )
    elif policy_qpos_init_enabled:
        print("  policy_qpos_init=on(g moves to dataset frame qpos, then startup init)")
        print(
            "Controls: g=start qpos init / resume, r=manual safety re-arm, A=confirm ready, h=pause+clear queue, q=quit"
        )
    else:
        print(
            "Controls: g=start/resume, r=manual safety re-arm, A=confirm ready/start policy, h=pause+clear queue, q=quit"
        )
    print("=" * 60 + "\n")


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


def enable_impedance_force_control_backends(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    ready_timeout_sec: float = 1.0,
) -> None:
    for side, backend in arm_backends.items():
        enable = getattr(backend, "enable_force_control", None)
        if not callable(enable):
            continue
        enable()
        wait_ready = getattr(backend, "wait_until_force_control_ready", None)
        if callable(wait_ready):
            wait_ready(timeout_sec=float(ready_timeout_sec))
        print(f"[Deploy] {side} impedance force control enabled.", flush=True)


def warm_start_impedance_force_control_backends(
    arm_backends: Mapping[str, EffectorBackend],
) -> tuple[str, ...]:
    """Prepare impedance controllers without starting their periodic output loops.

    The pose-impedance backend performs force-sensor setup and captures its
    initial safe reference in ``warm_start_force_control``.  Starting the
    40 Hz loop is deliberately a separate, explicit action: otherwise a
    deployment which is still waiting for ``g`` would already issue hold
    commands and continuously run IK.
    """

    warmed_sides: list[str] = []
    for side, backend in arm_backends.items():
        warm_start = getattr(backend, "warm_start_force_control", None)
        if not callable(warm_start):
            continue
        warm_start()
        warmed_sides.append(str(side))
    return tuple(warmed_sides)


def _compact_faults(faults: list[str], *, limit: int = 4) -> str:
    if not faults:
        return "none"
    shown = faults[:limit]
    suffix = "" if len(faults) <= limit else f",+{len(faults) - limit}"
    return ";".join(shown) + suffix


def _policy_status_text(policy: Any) -> str:
    wrapped_policy = getattr(policy, "policy", None)
    status = getattr(wrapped_policy, "status_text", None)
    if isinstance(status, str) and status:
        return status
    if callable(status):
        text = status()
        if text:
            return str(text)
    return ""


def _debug_replay_execution_exhausted(policy: Any, *, queued_actions: int) -> bool:
    """Stop non-loop replay after its already-decoded final chunk is consumed."""

    wrapped_policy = getattr(policy, "policy", policy)
    if not isinstance(wrapped_policy, DatasetReplayPolicy):
        return False
    return (
        not bool(wrapped_policy.loop)
        and bool(wrapped_policy.finished)
        and int(queued_actions) <= 0
    )


def _skip_startup_replay_frames(policy: Any, count: int = 1) -> int:
    wrapped_policy = getattr(policy, "policy", policy)
    skip = getattr(wrapped_policy, "skip_startup_frames", None)
    if not callable(skip):
        return 0
    return int(skip(count))


def _validate_debug_replay_arm_control_mode(
    policy: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    contract = validate_dataset_arm_control_mode(
        getattr(policy, "metadata", {}),
        selected_mode=args.arm_control_mode,
        hardware_output=bool(args.send_actions),
    )
    payload = contract.to_dict()
    metadata = getattr(policy, "metadata", None)
    if isinstance(metadata, dict):
        metadata["dataset_arm_control_mode_contract"] = dict(payload)
    if contract.warning:
        print(f"[ReplayMode] WARNING: {contract.warning}", flush=True)
    return payload


def _bind_policy_pose_mode_from_artifact(
    args: argparse.Namespace,
    policy_artifact: Any,
    *,
    raw_argv: list[str],
) -> None:
    """Compatibility wrapper around the shared artifact binding contract."""

    explicit = any(
        token == "--policy-action-pose-mode"
        or token.startswith("--policy-action-pose-mode=")
        for token in raw_argv
    )
    args.policy_action_pose_mode = resolve_policy_action_pose_mode_from_artifact(
        policy_artifact,
        configured_mode=str(args.policy_action_pose_mode),
        explicit=explicit,
    )


def _policy_observation_contract(
    *,
    policy: Any,
    policy_artifact: Any,
    deploy_layout: Any,
    pose_mode: str,
) -> Any:
    """Bind the strict actor view to this artifact/layout and its adapter needs."""

    effective_layout = deploy_layout
    if effective_layout is None:
        wrapped_policy = getattr(policy, "policy", None)
        policy_config = getattr(wrapped_policy, "config", None)
        effective_layout = getattr(policy_config, "policy_io_layout", None)
    hand_state_keys: set[str] = set()
    if effective_layout is not None:
        for side in ALL_SIDES:
            spec = effective_layout.end_effector(side)
            if spec.present:
                hand_state_keys.add(str(spec.state_source_field))
            channel = effective_layout.channel("state", side)
            for token in channel.tokens:
                if token.source_metadata_key:
                    hand_state_keys.add(str(token.source_metadata_key))

    if policy_artifact is not None:
        artifact_hash = canonical_content_hash(
            policy_artifact.to_dict(),
            domain="dexteleop.policy_observation_artifact_binding.v1",
        )
        binding_id = f"artifact={artifact_hash}"
    elif effective_layout is not None:
        binding_id = f"layout={effective_layout.layout_hash};model=unversioned"
    else:
        binding_id = f"policy={getattr(policy, 'name', 'unknown')}"
    return default_deploy_policy_observation_contract(
        binding_id=binding_id,
        last_action_pose_mode=str(pose_mode),
        compatibility_hand_state_keys=tuple(sorted(hand_state_keys)),
    )


def _reset_startup_initializer(
    startup_initializer: StartupInitializer,
    *,
    current_pose_start_allowed: bool,
) -> bool:
    """Reset startup init without inventing motion for an allowed current start.

    ``init-source=current`` resolves its target from the same live observation
    that supplies its start action.  When the caller has explicitly admitted a
    current-qpos start, replaying that no-op through the Cartesian position
    stream is both redundant and semantically wrong: it creates a new motion
    stream without a source-plan cut.  Keep non-current startup targets intact.
    """

    startup_initializer.reset()
    current_startup_satisfied = (
        bool(current_pose_start_allowed)
        and startup_initializer.enabled
        and str(startup_initializer.source).strip().lower() == "current"
    )
    if current_startup_satisfied:
        startup_initializer.mark_complete()
    return current_startup_satisfied


def _finish_policy_qpos_init_reference_cut(
    startup_initializer: StartupInitializer,
) -> tuple[np.ndarray, bool]:
    """Cut dataset-frame pose history before live Cartesian execution.

    The caller suppresses action generation for the qpos-init completion loop.
    An all-NaN ``last_action`` then makes the freshly sampled observation in
    the next loop the safety/reference anchor. A ``current`` startup init is a
    no-op after qpos init, so it is completed without submitting that same pose
    through IK for another initialization interval.
    """

    last_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
    current_startup_satisfied = _reset_startup_initializer(
        startup_initializer,
        current_pose_start_allowed=True,
    )
    return last_action, current_startup_satisfied


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

    def execute_prepared_action(
        canonical_action: Any,
        observation: Any,
        *,
        policy_name: str,
        source_decision_id: str | None = None,
        source_reference_segment_id: str | None = None,
        lookahead_canonical_action: Any | None = None,
        lookahead_observation: Any | None = None,
        second_lookahead_canonical_action: Any | None = None,
        second_lookahead_observation: Any | None = None,
        third_lookahead_canonical_action: Any | None = None,
        third_lookahead_observation: Any | None = None,
        source_plan_cut: PositionSourcePlanCut | None = None,
        phase_timing: dict[str, Any] | None = None,
    ) -> _PreparedActionExecution:
        command = getattr(canonical_action, "command", None)
        if command is None or not command.commands:
            return _PreparedActionExecution(
                accepted=False,
                mode="no_command",
                dispatch_attempted=False,
                dispatch_completed=False,
            )
        if position_deploy_executor is None:
            direct_dispatch_started = time.monotonic()
            if send_actions:
                dispatch_teleop_command(
                    command,
                    arm_backends=arm_backends,
                    hand_backends=hand_backends,
                )
            if phase_timing is not None:
                phase_timing["direct_dispatch_ms"] = max(
                    0.0,
                    time.monotonic() - direct_dispatch_started,
                ) * 1000.0
            return _PreparedActionExecution(
                accepted=True,
                mode="direct" if send_actions else "dry_run",
                dispatch_attempted=bool(send_actions),
                dispatch_completed=bool(send_actions),
            )

        arm_command = command_effector_kind_subset(command, kind="arm")
        hand_command = command_effector_kind_subset(command, kind="hand")
        lookahead_hand_command = None
        if lookahead_canonical_action is not None:
            lookahead_hand_command = command_effector_kind_subset(
                getattr(lookahead_canonical_action, "command", None),
                kind="hand",
            )
        if arm_command is not None:
            execution_work_item = None
            lookahead_execution_work_item = None
            if position_worker_binding is not None:
                if not source_decision_id or not source_reference_segment_id:
                    raise RuntimeError(
                        "epoch-mediated position submission requires exact source lineage"
                    )
                submit_legacy_owner_state(
                    lifecycle_state="running",
                    execution_gate="ready",
                    reason="legacy_reference_work_admission",
                    source_event_ids=(
                        str(source_decision_id),
                        str(source_reference_segment_id),
                    ),
                )
                execution_work_item = (
                    position_worker_binding.context.create_reference_segment_work_item(
                        source_decision_id=str(source_decision_id),
                        source_reference_segment_id=str(
                            source_reference_segment_id
                        ),
                    )
                )
                if lookahead_canonical_action is not None:
                    lookahead_execution_work_item = (
                        position_worker_binding.context.create_reference_segment_work_item(
                            source_decision_id=str(source_decision_id),
                            source_reference_segment_id=(
                                str(source_reference_segment_id) + ":lookahead"
                            ),
                        )
                    )
            position_submit_started = time.monotonic()
            try:
                source_index = position_deploy_executor.submit(
                    canonical_action.filtered,
                    observation,
                    policy_name=str(policy_name),
                    discrete_command=hand_command,
                    lookahead_filtered=(
                        None
                        if lookahead_canonical_action is None
                        else lookahead_canonical_action.filtered
                    ),
                    lookahead_observation=lookahead_observation,
                    lookahead_discrete_command=lookahead_hand_command,
                    second_lookahead_filtered=(
                        None
                        if second_lookahead_canonical_action is None
                        else second_lookahead_canonical_action.filtered
                    ),
                    second_lookahead_observation=second_lookahead_observation,
                    third_lookahead_filtered=(
                        None
                        if third_lookahead_canonical_action is None
                        else third_lookahead_canonical_action.filtered
                    ),
                    third_lookahead_observation=third_lookahead_observation,
                    execution_work_item=execution_work_item,
                    lookahead_execution_work_item=lookahead_execution_work_item,
                    source_plan_cut=source_plan_cut,
                )
            except PositionDeployExecutorFault:
                fault_stage = str(
                    position_deploy_executor.snapshot().get("fault_stage") or ""
                )
                if fault_stage in {
                    "safety_gate",
                    "15hz_keyframe_safety",
                    "reference_anchor_safety",
                    "target_pose_envelope",
                }:
                    return _PreparedActionExecution(
                        accepted=False,
                        mode="position_safety_fault_latched",
                        dispatch_attempted=False,
                        dispatch_completed=False,
                    )
                raise
            finally:
                if phase_timing is not None:
                    phase_timing["position_submit_ms"] = max(
                        0.0,
                        time.monotonic() - position_submit_started,
                    ) * 1000.0
            return _PreparedActionExecution(
                accepted=True,
                mode=(
                    "position_async_queue" if send_actions else "position_dry_run_queue"
                ),
                dispatch_attempted=False,
                dispatch_completed=False,
                asynchronous=True,
                position_source_index=int(source_index),
            )
        elif send_actions and hand_command is not None:
            # A hand-only command has no arm keyframe whose causal reference
            # clock could carry it. Mixed arm+hand actions are queued together
            # above so both become effective on the same logical reference tick.
            try:
                require_arm_backends_ready_for_position_dispatch(
                    arm_backends,
                    require_background_target=False,
                    stage="pre_direct_hand_only_dispatch",
                    required_sides=active_sides,
                )
                dispatch_teleop_command(
                    hand_command,
                    arm_backends={},
                    hand_backends=hand_backends,
                )
            except Exception:
                position_deploy_executor.clear()
                raise
            return _PreparedActionExecution(
                accepted=True,
                mode="position_hand_only_direct",
                dispatch_attempted=True,
                dispatch_completed=True,
            )
        return _PreparedActionExecution(
            accepted=bool(hand_command is not None),
            mode="position_hand_only_dry_run",
            dispatch_attempted=False,
            dispatch_completed=False,
        )

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
                raise RuntimeError(
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
            set_callback = getattr(backend, "set_force_hard_stop_callback")
            set_callback(impedance_force_guard_monitor.notify_backend_hard_stop)

    if position_force_guard_required:
        position_guard_readers: dict[str, Callable[[], Mapping[str, Any]]] = {}
        for side, backend in arm_backends.items():
            reader = getattr(backend, "read_force_data", None)
            if not callable(reader):
                raise RuntimeError(
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
                f"{str(args.position_continuous_waypoint_profile_effective)} "
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

        for side, backend in arm_backends.items():
            backend.start()
            if not backend.is_ready() and not bool(args.allow_arm_connect_failure):
                raise RuntimeError(
                    f"{side} arm backend is not ready: {backend.get_state()}"
                )
        if position_deploy_executor is not None:
            position_deploy_executor.prepare_collision_planner()
            position_deploy_executor.start()
            if args.position_reference_worker_cpu is not None:
                affinity = position_deploy_executor.snapshot().get(
                    "reference_worker_affinity_receipt"
                )
                if args.position_continuous_waypoint_worker_cpu is None:
                    print(
                        "[PositionCPU] READY: reference worker affinity "
                        f"verified cpu={args.position_reference_worker_cpu}; "
                        "C2 worker is disabled for this formal A/B run.",
                        flush=True,
                    )
                else:
                    print(
                        "[PositionCPU] READY: reference worker affinity verified "
                        f"cpu={args.position_reference_worker_cpu}; "
                        "C2 worker affinity is verified by its planner startup "
                        f"receipt cpu={args.position_continuous_waypoint_worker_cpu} "
                        "isolation="
                        f"{args.position_continuous_waypoint_worker_isolation}.",
                        flush=True,
                    )
                if not isinstance(affinity, Mapping) or not bool(
                    affinity.get("applied", False)
                ):
                    raise RuntimeError(
                        "position reference worker affinity receipt is missing "
                        "after successful startup"
                    )
                if bool(args.position_worker_cpu_reservation):
                    reference_native_thread_id = affinity.get("native_thread_id")
                    if isinstance(reference_native_thread_id, bool) or not isinstance(
                        reference_native_thread_id, int
                    ):
                        raise RuntimeError(
                            "position reference worker affinity receipt lacks "
                            "a native thread id for CPU reservation"
                        )
                    reservation = exclude_cpu_ids_from_current_process(
                        args.position_worker_cpu_reservation_cpu_ids,
                        role="position_deploy_client_ordinary_threads",
                        exempt_native_thread_ids=(int(reference_native_thread_id),),
                    )
                    print(
                        "[PositionCPU] READY: deployment-client ordinary "
                        "threads exclude reserved CPUs "
                        f"{reservation['excluded_cpu_ids']}; reference_tid="
                        f"{reference_native_thread_id} remains pinned; mode="
                        + (
                            "async_c2_process"
                            if args.position_continuous_waypoint_worker_cpu is not None
                            else "formal_reference_only"
                        )
                        + "; migrated_native_threads="
                        f"{reservation['native_threads_migrated']}.",
                        flush=True,
                    )
                if bool(args.position_continuous_waypoint_no_debug_deadline_soak):
                    print(
                        "[PositionCPU] SOAK: bounded no-debug dry-run exception "
                        f"active max_steps={int(args.max_steps)}; hardware "
                        "dispatch remains forbidden.",
                        flush=True,
                    )
        if position_force_guard_monitor is not None:
            guard_snapshot = position_force_guard_monitor.start(
                arm_stale_watchdog=False,
            )
            print(
                "[PositionForceGuard] force limits armed; stale watchdog "
                "deferred until camera startup completes "
                f"config={guard_snapshot.get('config', {})} "
                f"sides={sorted(guard_snapshot.get('last_reading_by_side', {}))}",
                flush=True,
            )
        if impedance_force_control_is_deferred:
            warmed_sides = warm_start_impedance_force_control_backends(arm_backends)
            if warmed_sides:
                print(
                    "[Deploy] impedance prepared without periodic output; "
                    "press g/H/A to enable it, or use --auto-start.",
                    flush=True,
                )
        for side, backend in hand_backends.items():
            backend.start()
            if not backend.is_ready() and not bool(args.allow_hand_connect_failure):
                raise RuntimeError(
                    f"{side} hand backend is not ready: {backend.get_state()}"
                )
        if hand_state_poller is not None:
            hand_state_poller.start()
        if camera_sources:
            print(
                "[Camera] configured "
                f"cameras={sorted(camera_sources)} mode={args.camera_mode} "
                f"vis_cam={bool(args.vis_cam)}"
            )
            if camera_previewer is not None:
                if camera_previewer.ensure_available():
                    print("[Camera] --vis_cam display preflight ok.")
                else:
                    print(
                        f"[Camera] preview disabled: {camera_previewer.disabled_reason}"
                    )
        start_camera_sources(
            camera_sources,
            stagger_sec=float(args.camera_start_stagger_sec),
        )
        run_camera_startup_self_check(camera_sources)
        if position_force_guard_monitor is not None:
            guard_snapshot = position_force_guard_monitor.arm_stale_watchdog()
            print(
                "[PositionForceGuard] armed fail_closed=true latched=true "
                "stop_scope=bimanual "
                f"config={guard_snapshot.get('config', {})}",
                flush=True,
            )
        if rtc_frontend is not None:
            rtc_frontend.start()
            if str(args.rtc_profile) == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
                print(
                    "[raw-RTC] OpenPI/model final-chunk shadow active: "
                    "send_actions=false, position_lookahead=false, "
                    "timed-plan admission enabled."
                    + (
                        " live observation-only hardware guard enabled."
                        if bool(getattr(args, "rtc_real_shadow", False))
                        else ""
                    ),
                    flush=True,
                )
            elif str(args.rtc_profile) == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1:
                print(
                    "[model-RTC] conditioned OpenPI shadow active: "
                    "send_actions=false, position_lookahead=false, "
                    "timed-plan admission and prefix conditioning enabled."
                    + (
                        " live observation-only hardware guard enabled."
                        if bool(getattr(args, "rtc_real_shadow", False))
                        else ""
                    ),
                    flush=True,
                )
            else:
                print(
                    "[RTC-3c] virtual-policy shadow active: send_actions=false, "
                    "position_lookahead=false, timed-plan admission enabled.",
                    flush=True,
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
                    policy.reset()
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
                                    raise RuntimeError(
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
                                    raise RuntimeError(
                                        f"{side} backend does not expose manual impedance re-arm"
                                    )
                                rearm(qpos)
                            impedance_force_guard_monitor.rearm_after_manual_home(
                                start=False
                            )
                            impedance_force_control_enabled = False
                        except (
                            Exception
                        ) as exc:  # noqa: BLE001 - keep the latch on failure.
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
                            policy.reset()
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
                        except (
                            Exception
                        ) as exc:  # noqa: BLE001 - operator-facing re-arm gate.
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
                            policy.reset()
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
                        policy.reset()
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
                        policy.reset()
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
                                    execution = execute_prepared_action(
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
                                "reason": "synchronous_inference_with_empty_action_queue",
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
                        queue_len_before_update = prediction.queue_len_before
                        if prediction.accepted:
                            ik_action_gate.mark_policy_output_accepted()
                            defer_position_action_until_fresh_cut = (
                                _defer_position_action_after_synchronous_prediction(
                                    prediction_accepted=True,
                                    position_executor_enabled=(
                                        position_deploy_executor is not None
                                    ),
                                )
                            )
                            debug_logger.log_policy_chunk(
                                timestamp=loop_started_at,
                                step_index=step_index,
                                policy_name=policy.name,
                                policy_output=prediction.policy_output,
                                actions=list(prediction.actions),
                                accepted=True,
                                discarded_by_ik=False,
                                queue_policy=action_queue_policy,
                                queue_len_before=queue_len_before_update,
                                queue_len_after=prediction.queue_len_after,
                                faults=faults,
                                ik_recovery_statuses=ik_action_gate.last_statuses,
                                ik_recovery_diagnostics=ik_action_gate.last_diagnostics,
                            )
                            if defer_position_action_until_fresh_cut:
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
                                        "observation_step_index": int(
                                            observation.step_index
                                        ),
                                    },
                                )
                        elif prediction.discarded_by_gate:
                            print(
                                "[Deploy] IK recovery active; discarded predicted action chunk."
                            )
                            debug_logger.log_policy_chunk(
                                timestamp=loop_started_at,
                                step_index=step_index,
                                policy_name=policy.name,
                                policy_output=prediction.policy_output,
                                actions=list(prediction.actions),
                                accepted=False,
                                discarded_by_ik=True,
                                queue_policy=action_queue_policy,
                                queue_len_before=queue_len_before_update,
                                queue_len_after=prediction.queue_len_after,
                                faults=faults,
                                ik_recovery_statuses=ik_action_gate.last_statuses,
                                ik_recovery_diagnostics=ik_action_gate.last_diagnostics,
                            )
                        elif prediction.error is None:
                            faults.extend(prediction.faults)
                            debug_logger.log_policy_chunk(
                                timestamp=loop_started_at,
                                step_index=step_index,
                                policy_name=policy.name,
                                policy_output=prediction.policy_output,
                                actions=[],
                                accepted=False,
                                discarded_by_ik=False,
                                queue_policy=action_queue_policy,
                                queue_len_before=queue_len_before_update,
                                queue_len_after=prediction.queue_len_after,
                                faults=faults,
                                ik_recovery_statuses=ik_action_gate.last_statuses,
                                ik_recovery_diagnostics=ik_action_gate.last_diagnostics,
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
                            print(f"[Deploy] policy error: {type(exc).__name__}: {exc}")
                            faults.append(f"policy_error:{type(exc).__name__}")
                            debug_logger.log_policy_error(
                                timestamp=loop_started_at,
                                step_index=step_index,
                                policy_name=policy.name,
                                exc=exc,
                                queue_len_before=queue_len_before_update,
                            )
                            if bool(args.pause_on_policy_error):
                                policy_active = False
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
                                execution = execute_prepared_action(
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
                        except Exception as exc:  # noqa: BLE001
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
                            "position="
                            f"refs:{reference_count}/queue:{queue_depth}"
                            f"/missed:{missed_ticks}",
                            "collision="
                            f"{collision_status}/d:{collision_distance_text}"
                            f"/reset:{collision_reset_status}",
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
        position_workers_stopped = True
        cleanup_errors: list[str] = []
        try:
            submit_legacy_owner_state(
                lifecycle_state="stopped",
                execution_gate="hold",
                reason="policy_deploy_cleanup_started",
            )
        except Exception as exc:  # noqa: BLE001 - cleanup must continue.
            cleanup_errors.append(
                "legacy_control_owner_stop: " + f"{type(exc).__name__}: {exc}"
            )
        if keyboard_interrupt_stop_error is not None:
            cleanup_errors.append(
                "keyboard_interrupt_emergency_stop: " + keyboard_interrupt_stop_error
            )

        def close_resource(label: str, resource: Any) -> None:
            try:
                resource.close()
            except Exception as exc:  # noqa: BLE001 - close every independent resource.
                error = f"{label}: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                debug_logger.log_event(
                    "cleanup_error",
                    {
                        "resource": str(label),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(f"[DeployCleanup] {error}", file=sys.stderr)

        if rtc_frontend is not None:
            try:
                rtc_stop = rtc_frontend.close(now=time.monotonic())
                if rtc_debug_observer is not None:
                    rtc_debug_observer.offer(
                        "rtc_frontend_close",
                        {
                            "worker_stop_hash": rtc_stop.semantic_content_hash,
                            "worker_session_state": rtc_stop.session_state,
                            "thread_exited": rtc_stop.thread_exited,
                            "execution_drain_evidence": (
                                rtc_stop.execution_drain_evidence
                            ),
                            "stop_status": rtc_stop.status,
                        },
                    )
            except Exception as exc:  # noqa: BLE001 - continue all cleanup.
                error = f"rtc_frontend: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                print(f"[DeployCleanup] {error}", file=sys.stderr)
        if rtc_debug_observer is not None:
            try:
                rtc_debug_was_enabled = rtc_debug_observer.enabled
                rtc_debug_status = rtc_debug_observer.close()
                if rtc_debug_was_enabled:
                    debug_logger.log_event(
                        "rtc_debug_observer_status",
                        rtc_debug_status.to_payload(),
                    )
            except Exception as exc:  # noqa: BLE001 - diagnostic shutdown only.
                error = f"rtc_debug_observer: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                print(f"[DeployCleanup] {error}", file=sys.stderr)
        if position_deploy_executor is not None:
            try:
                position_deploy_executor.close()
            except (
                Exception
            ) as exc:  # noqa: BLE001 - inspect worker liveness before backend close.
                close_snapshot = position_deploy_executor.snapshot()
                position_workers_stopped = not bool(
                    close_snapshot.get("reference_worker_alive")
                    or close_snapshot.get("discrete_worker_alive")
                )
                print(
                    f"[PositionDeploy] close/hold error: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                cleanup_errors.append(f"position_executor: {type(exc).__name__}: {exc}")
                debug_logger.log_event(
                    "position_executor_close_error",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "snapshot": close_snapshot,
                        "backend_close_skipped": bool(not position_workers_stopped),
                    },
                )
            flush_position_executor_debug_events()
            debug_logger.log_event(
                "position_executor_final_snapshot",
                position_deploy_executor.snapshot(),
            )
        if position_force_guard_monitor is not None:
            debug_logger.log_event(
                "position_force_guard_final_snapshot",
                position_force_guard_monitor.snapshot(),
            )
            close_resource("position_force_guard", position_force_guard_monitor)
        if impedance_force_guard_monitor is not None:
            close_resource("impedance_force_guard", impedance_force_guard_monitor)
        if fixed_init_controller is not None:
            try:
                fixed_init_controller.restore_speed(arm_backends)
            except Exception as exc:  # noqa: BLE001
                error = f"fixed_init_restore_speed: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                debug_logger.log_event(
                    "cleanup_error",
                    {
                        "resource": "fixed_init_restore_speed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(f"[DeployCleanup] {error}", file=sys.stderr)
        if hand_state_poller is not None:
            close_resource("hand_state_poller", hand_state_poller)
        if position_workers_stopped:
            for side, backend in arm_backends.items():
                close_resource(f"{side}_arm_backend", backend)
            for side, backend in hand_backends.items():
                close_resource(f"{side}_hand_backend", backend)
        else:
            cleanup_errors.append(
                "position_backends: close skipped because a dispatch worker is still alive"
            )
            print(
                "[PositionDeploy] ERROR: a dispatch worker is still alive; "
                "arm/hand backend close was intentionally skipped to avoid concurrent I/O.",
                file=sys.stderr,
            )
        for camera_key, camera_source in camera_sources.items():
            close_resource(f"camera:{camera_key}", camera_source)
        if camera_previewer is not None:
            close_resource("camera_previewer", camera_previewer)
        if visualizer is not None:
            close_resource("tactile_visualizer", visualizer)
        try:
            debug_logger.close()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"debug_logger: {type(exc).__name__}: {exc}")
            print(
                f"[DeployCleanup] debug_logger: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        if cleanup_errors:
            print(
                f"[DeployCleanup] completed with {len(cleanup_errors)} error(s).",
                file=sys.stderr,
            )
    return 1 if cleanup_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
