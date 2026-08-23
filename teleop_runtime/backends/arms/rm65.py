from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
import math
import threading
from typing import Callable, Mapping, Sequence

import numpy as np
import time

from hardware.rm65.calibration_profiles import (
    ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD,
    is_position_endpoint_profile,
    tcp_endpoint_delta,
    tcp_safety_warnings,
)
from ...arm_frames import (
    ARM_ENDPOINT_FRAME,
    EMPTY_TOOL_FRAME,
    TOOL_TCP_FRAME,
    canonical_arm_controlled_frame,
)
from robot_motion.rm65.ik_tuning import (
    RM65IKTuningConfig,
    hard_branch_jump_exceeds,
    joint_jump_exceeds,
    load_ik_tuning_config,
    pose_delta_metrics,
)
from robot_motion.rm65.assets import require_asset_path
from robot_motion.rm65 import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    RM65SingleArmIKConfig,
    build_rm65_single_arm_ik_solver,
    rm65_single_arm_candidate_frames,
    solve_rm65_single_arm_ik,
)
from robot_motion.position_control.audit import (
    POSITION_TRANSPORT_TARGET_SEQUENCE_KEY,
    PercentileStats,
    PositionAuditEvent,
    PositionAuditSink,
    PositionCommandAuditEvent,
    PositionTimingAuditEvent,
    resolve_periodic_tick,
)
from robot_motion.position_control.interpolation import (
    JointCommandInterpolator,
    interpolate_joint_command_segment,
    reference_age_and_stale,
)
from robot_motion.position_control.execution_policy import (
    PositionTrajectoryPolicy as _PositionTrajectoryPolicy,
    classify_position_trajectory_policy as _position_trajectory_policy,
    final_follow_qpos_target as _final_follow_qpos_target,
)
from robot_motion.position_control.contracts import (
    POSITION_QREF_FINGERPRINT_SCHEMA,
    position_qref_fingerprint,
)
from robot_motion.position_control.trajectory import (
    AccelerationLimitedJointTrajectoryPlanner,
    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    JointTrajectoryLimits,
    JointTrajectoryProposal,
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_LINEAR,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
    normalize_position_command_planner,
)
from robot_motion.position_control.horizon_governor import JointHorizonLimits
from robot_motion.position_control.sample_aligned_trajectory import (
    SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET,
    SampleAlignedJerkLimitedJointTrajectoryPlanner,
    SampleAlignedTargetAdmission,
    normalize_sample_aligned_source_latch_mode,
)
from robot_motion.runtime.dispatch import BackendDispatchError, CommandDispatchReceipt
from robot_motion.runtime import (
    ExecutionCancellation,
    ExecutionCancellationParticipantSpec,
    ExecutionEpochGuard,
    ExecutionWorkItem,
    WorkerEpochDrainReceipt,
)

from ...interfaces import EffectorBackend
from ...log_utils import log_event as _console_log_event
from ...registry import register_backend


_BACKGROUND_STREAM_WARNING_PRINT_INTERVAL_SEC = 5.0
from ...schemas import EffectorCommand
from ...defaults import DEFAULT_RM65_POSITION_ARM_CONTROL_HZ
from .rm65_dispatch_call import (
    RM65JointTargetDispatchContext,
    RM65PreSdkDispatchAdapter,
)
from .rm65_control_dispatch_call import (
    RM65ControlDispatchContext,
    RM65PreSdkControlAdapter,
)


RM65_IK_FRAMEWORK_ROBOT_MOTION = "robot_motion"
RM65_IK_FRAMEWORK_CHOICES = (RM65_IK_FRAMEWORK_ROBOT_MOTION,)
RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK = "synchronous_lock"
RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING = "latest_nonblocking"
RM65_BACKGROUND_TARGET_ADMISSION_CHOICES = (
    RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK,
    RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
)

# Shared lower execution defaults for position reference sources.  Deployment
# frontends may keep compatibility aliases, but a non-policy Skill process must
# not import a policy-deployment composition module merely to bind the same
# reviewed RM65 backend timing behavior.
RM65_POSITION_SEND_DT_ALPHA = 0.25
RM65_POSITION_SEND_DT_MAX_SCALE = 3.0
RM65_POSITION_SEND_DT_STALL_RESET_SEC = 0.25
HAND_BASE_POSE_FLANGE_M_RAD = (0.0, 0.0, 0.009, 0.0, 0.0, 0.0)
_SINGULARITY_WARNING_INTERVAL_SEC = 1.0
DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN = 0.2
DEFAULT_SINGULARITY_REPLAY_SLOWDOWN = 0.2
_SINGULARITY_QPOS_INIT_SLOW_SOURCES = frozenset(
    {
        "replay_qpos_init",
        "replay_qpos_init_final_hold",
        "deploy_fixed_init_qpos",
        "gui_init_qpos_plan",
    }
)
_SERIALIZED_QPOS_INIT_TARGET_SOURCES = frozenset(
    {
        "replay_qpos_init",
        "replay_qpos_init_final_hold",
    }
)
_SINGULARITY_LABELS = {
    0: "normal",
    -1: "shoulder_singularity",
    -2: "elbow_singularity",
    -3: "wrist_singularity",
}


@dataclass(frozen=True)
class _DeferredAccelerationTarget:
    qpos: np.ndarray
    source: str
    timestamp_wall: float
    timestamp_monotonic: float
    sequence: int
    reference_epoch: int | None
    reference_sequence: int | None
    execution_work_item: ExecutionWorkItem | None
    policy_mode: str
    policy_speed_scale: float
    policy_reason: str


# These are the absolute position limits in the canonical RM65 URDF.  They are
# intentionally enforced again at the transport boundary: upstream IK/safety
# may be bypassed by a direct joint command, but an unsafe value must never
# reach Movej_CANFD.
_RM65_ABSOLUTE_Q_MIN_RAD = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
_RM65_ABSOLUTE_Q_MAX_RAD = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
# The final transport boundary remains a hard qpos envelope.  Upstream solver
# and canonicalization checks admit 1e-8 rad of floating-point endpoint noise.
# Keep a 2x transport comparison guard band, then always project every accepted
# value to a slightly *smaller* envelope before it can reach a planner or SDK.
# The comparison tolerance therefore never widens the physical joint range.
_RM65_ABSOLUTE_LIMIT_TOLERANCE_RAD = 2.0e-8
_RM65_ABSOLUTE_LIMIT_DISPATCH_MARGIN_RAD = 1.0e-8
# RM65 SDK joint feedback is reported in degrees and converted back to radians
# here. A local preview/home anchor may therefore exceed an endpoint by a few
# encoder/float quanta even though no command was sent beyond the envelope.
# This tolerance is intentionally *not* used by the transport validator above.
_RM65_LOCAL_QPOS_OBSERVATION_TOLERANCE_RAD = float(np.deg2rad(0.01))
_BACKGROUND_AUDIT_STATS_MAX_SAMPLES = 20_000
_BACKGROUND_AUDIT_SNAPSHOT_CACHE_SEC = 1.0
_BACKGROUND_WORKER_TERMINAL_STAGE = "rm65_background_stream_worker_terminal"
# A vendor ``-2`` result means that the SDK did not receive the controller
# acknowledgement.  The command may still have reached the controller, so a
# bounded idempotent retry is safer than declaring the stop unacknowledged
# after one response timeout.  Keep this short: the stream is already latched
# and no new motion can be admitted while these calls run.
_EMERGENCY_STOP_RETRY_DELAY_SEC = 0.05
_EMERGENCY_STOP_ARM_STOP_ATTEMPTS = 2


def _derive_backend_execution_work_item(
    upstream: ExecutionWorkItem | None,
    *,
    side: str,
    target_sequence: int | None,
) -> ExecutionWorkItem | None:
    if upstream is None:
        return None
    sequence_label = "direct" if target_sequence is None else str(int(target_sequence))
    return ExecutionWorkItem(
        work_item_id=(
            f"{upstream.work_item_id}:rm65:{str(side)}:target:{sequence_label}"
        ),
        work_kind="backend_command",
        control_domain_id=upstream.control_domain_id,
        authority_epoch=upstream.authority_epoch,
        execution_epoch=upstream.execution_epoch,
        segment_generation_by_group=dict(upstream.segment_generation_by_group),
        source_decision_id=upstream.source_decision_id,
        source_reference_segment_id=upstream.source_reference_segment_id,
    )


def _metadata_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def normalize_rm65_ik_framework(value: object) -> str:
    text = str(value or RM65_IK_FRAMEWORK_ROBOT_MOTION).strip().lower().replace("-", "_")
    aliases = {
        "default": RM65_IK_FRAMEWORK_ROBOT_MOTION,
        "shared": RM65_IK_FRAMEWORK_ROBOT_MOTION,
        "robotmotion": RM65_IK_FRAMEWORK_ROBOT_MOTION,
        "robot_motion": RM65_IK_FRAMEWORK_ROBOT_MOTION,
    }
    resolved = aliases.get(text, text)
    if resolved not in RM65_IK_FRAMEWORK_CHOICES:
        raise ValueError(
            f"Unsupported RM65 IK framework {value!r}. Expected one of {RM65_IK_FRAMEWORK_CHOICES}."
        )
    return resolved


def _is_empty_tool_frame_name(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    stem = Path(text).stem
    normalized = stem if stem else text
    return normalized == EMPTY_TOOL_FRAME or normalized.endswith(f"_{EMPTY_TOOL_FRAME}")


def _load_rm65_sdk():
    try:
        from hardware.rm65.rm_api2_adapter import Arm, RM65
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RM65ArmBackend requires the RM65 SDK to be importable."
        ) from exc
    return Arm, RM65


def _parse_rm65_result(result: object) -> tuple[int | None, str]:
    if isinstance(result, (int, np.integer)):
        code = int(result)
        return code, str(code)
    text = "" if result is None else str(result)
    prefix = text.split(":", 1)[0].strip()
    try:
        code = int(prefix)
    except (TypeError, ValueError):
        code = None
    return code, text


def _singularity_status_label(status_code: int | None) -> str:
    if status_code is None:
        return "unavailable"
    return _SINGULARITY_LABELS.get(int(status_code), f"unknown_status_{int(status_code)}")


def _command_source(command: EffectorCommand) -> str:
    metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
    source_text = str(metadata.get("source") or "").strip()
    action_source_text = str(metadata.get("action_source") or "").strip()
    if source_text == "robot_action" and action_source_text:
        return action_source_text
    return source_text or action_source_text or str(command.command_type)


def _command_max_joint_step(command: EffectorCommand) -> np.ndarray | None:
    metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
    for key in ("max_joint_step_rad", "deploy_max_joint_step_rad"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            step = np.asarray(value, dtype=float).reshape(6)
        except Exception:
            continue
        if np.all(np.isfinite(step)) and np.all(step >= 0.0):
            return step
    return None


def _command_is_prelimited_solver_qpos(command: EffectorCommand) -> bool:
    metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
    if str(metadata.get("action_pose_source") or "") == "limited_solver_output_qpos":
        return True
    execution_step = metadata.get("execution_step_metadata")
    return isinstance(execution_step, Mapping) and bool(execution_step.get("enabled"))


def _metadata_non_negative_int(metadata: Mapping[str, object], key: str) -> int | None:
    value = metadata.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    if normalized < 0 or normalized != value:
        return None
    return normalized


def _is_qpos_init_slow_singularity_source(source: object) -> bool:
    return str(source or "").strip() in _SINGULARITY_QPOS_INIT_SLOW_SOURCES


def _is_replay_slow_singularity_source(source: object) -> bool:
    """Return whether a source is a replay frame, never a deploy/init command."""

    text = str(source or "").strip()
    return text in {"replay_qpos_trajectory", "replay_action_pose"} or text.startswith(
        (
            "replay_collision_qpos_reference_",
            "replay_qpos_reference_",
            "replay_action_pose_reference_",
        )
    )


def _joint_now_to_qpos(joint_now: object) -> np.ndarray | None:
    if joint_now is None:
        return None
    try:
        qpos = np.asarray(joint_now, dtype=float).reshape(-1)
    except Exception:
        return None
    if qpos.shape[0] != 6 or not np.all(np.isfinite(qpos)):
        return None
    return qpos / 180.0 * np.pi


def _state_result_is_usable(
    parsed_code: int | None,
    *,
    joint_now: object,
    pose_now: object,
) -> bool:
    if parsed_code == 0:
        return True
    if parsed_code == 9:
        return _joint_now_to_qpos(joint_now) is not None or pose_now is not None
    return False


def _tool_name_from_frame_result(frame_result: object) -> str:
    if isinstance(frame_result, tuple) and len(frame_result) >= 2:
        frame = frame_result[1]
    else:
        frame = frame_result
    if isinstance(frame, dict):
        value = frame.get("name") or frame.get("frame_name") or frame.get("tool_name")
        if value is not None:
            return str(value).strip("\x00")
    return ""


def _rotation_matrix_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = np.sin(float(rx)), np.cos(float(rx))
    sy, cy = np.sin(float(ry)), np.cos(float(ry))
    sz, cz = np.sin(float(rz)), np.cos(float(rz))
    rx_mat = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry_mat = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz_mat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz_mat @ ry_mat @ rx_mat


def _pose6_to_matrix(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=float).reshape(6)
    if not np.all(np.isfinite(values)):
        raise ValueError("TCP pose contains non-finite values.")
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _rotation_matrix_from_euler_xyz(*values[3:6])
    matrix[:3, 3] = values[:3]
    return matrix


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    out = np.eye(4, dtype=float)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ matrix[:3, 3]
    return out


def _as_pose_matrix(value: object) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape == (4, 4):
        return pose.copy()
    if pose.shape == (7,):
        x, y, z, qx, qy, qz, qw = pose
        norm = np.linalg.norm(pose[3:7])
        if norm <= 0.0:
            raise ValueError("target quaternion norm must be positive")
        qx, qy, qz, qw = np.asarray([qx, qy, qz, qw], dtype=float) / norm
        matrix = np.eye(4, dtype=float)
        matrix[:3, 3] = [x, y, z]
        matrix[:3, :3] = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )
        return matrix
    raise ValueError(f"Unsupported pose shape for configured TCP transform: {pose.shape}")


def _warn_if_tcp_outside_safety_limits(
    *,
    owner: str,
    pose_flange_m_rad: Sequence[float],
    source: str,
) -> None:
    warnings = tcp_safety_warnings(pose_flange_m_rad, label=f"{owner} TCP")
    for warning in warnings:
        print(f"WARNING: {warning}; source={source}. Verify this TCP before robot motion.")


@dataclass
class RM65ArmBackendConfig:
    arm_ip: str
    side: str
    effector_name: str | None = None
    default_tcp_frame: str = TOOL_TCP_FRAME
    arm_urdf_path: str | Path | None = None
    connect_hardware: bool = True
    # Deployment-only live observation shadow.  This is intentionally opt-in
    # and leaves teleop/replay/default dry-run semantics unchanged.  When set,
    # startup may read feedback but must not configure, stream, stop, or send
    # any RM65 motion command.
    control_dt: float = 0.02
    max_angvel: tuple[float, ...] = (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)
    dry_run_fast: bool = True
    dry_run_max_iterations: int = 48
    dry_run_max_joint_step: tuple[float, ...] = (0.3, 0.3, 0.3, 0.4, 0.4, 0.4)
    dry_run_damping: float = 0.03
    hardware_read_retries: int = 2
    hardware_read_retry_delay_sec: float = 0.02
    initial_qpos: tuple[float, ...] | None = None
    verified_hardware_initial_qpos: tuple[float, ...] | None = None
    use_actual_send_dt_scaling: bool = False
    actual_send_dt_alpha: float = 0.25
    actual_send_dt_min_scale: float = 1.0
    actual_send_dt_max_scale: float = 2.0
    actual_send_dt_stall_reset_sec: float = 0.25
    background_streaming_enabled: bool = False
    background_streaming_hz: float = DEFAULT_RM65_POSITION_ARM_CONTROL_HZ
    background_streaming_warning_ratio: float = 0.9
    background_streaming_target_max_age_sec: float = 0.25
    background_streaming_shutdown_timeout_sec: float = 1.0
    # Keep the backend-wide default compatible for teleop/replay callers.
    # Policy deployment explicitly selects latest_nonblocking so a slow SDK
    # call cannot make 60 Hz reference admission wait on the 120 Hz worker.
    background_streaming_target_admission_mode: str = (
        RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK
    )
    position_command_planner: str = POSITION_COMMAND_PLANNER_LINEAR
    position_velocity_ramp_sec: float = DEFAULT_POSITION_VELOCITY_RAMP_SEC
    position_target_velocity_filter_tau_sec: float = (
        DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC
    )
    position_max_acceleration_rad_s2: tuple[float, ...] | None = None
    # Optional position-only envelope applied to every high-rate planner
    # proposal.  It is intentionally independent of max_angvel so a peak-step
    # guard does not reduce sustained tracking velocity.
    position_max_command_step_deg: float | None = None
    position_degraded_speed_scale: float = DEFAULT_POSITION_DEGRADED_SPEED_SCALE
    # Experimental, default-off execution profile. Incoming qrefs keep running
    # for IK/safety, while the selected latch admits one reachable endpoint per
    # source cut and caches its high-rate jerk-limited control cut.
    position_sample_aligned_source_hz: float = 15.0
    position_sample_aligned_source_latch_mode: str = (
        SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET
    )
    position_sample_aligned_horizon_source_steps: int = 2
    position_sample_aligned_safety_horizon_source_steps: int | None = None
    position_sample_aligned_scale_degraded_target: bool = True
    position_sample_aligned_jerk_regularization: float = 1e-9
    position_sample_aligned_jerk_delta_regularization: float = 0.0
    position_sample_aligned_initial_jerk_delta_regularization: float = 0.0
    position_sample_aligned_terminal_velocity_regularization: float = 0.0
    position_sample_aligned_terminal_acceleration_regularization: float = 0.0
    position_sample_aligned_reference_mode: str = "linear_hold"
    position_sample_aligned_observer_window: int = 4
    position_sample_aligned_observer_efficiency_power: float = 2.0
    position_sample_aligned_observer_velocity_ema_alpha: float = 0.5
    position_sample_aligned_observer_velocity_cap_scale: float = 0.5
    position_sample_aligned_observer_coherence_min: float = 0.5
    position_sample_aligned_observer_speed_guard_scale: float = 0.4
    position_sample_aligned_terminal_velocity_displacement_scale: float = 0.75
    position_sample_aligned_observer_stop_threshold_deg: float = 0.01
    position_sample_aligned_observer_stop_velocity_decay: float = 0.0
    arm_follow_qpos_alpha: float = 1.0
    canfd_follow: bool = True
    canfd_trajectory_mode: int = 0
    canfd_radio: int = 0
    force_position_control: bool = False
    force_sensor: int = 1
    force_mode: int = 0
    force_dir: int = 0
    force: float = 0.0
    force_follow: bool = False
    force_clear_on_start: bool = True
    auto_load_tool_profile: bool = True
    tool_profile: str | None = None
    tcp_pose_flange_m_rad: tuple[float, ...] | None = None
    singularity_protection_enabled: bool = True
    # Position-teleop-only opt-in. A singular target is allowed only when the
    # upstream candidate selector explicitly reports that no fully
    # non-wrist-singular feasible branch existed.
    allow_position_wrist_singularity_fallback: bool = False
    # Position-teleop-only opt-in.  The backend still requires an upstream
    # candidate-selection certificate that no fully non-singular feasible
    # candidate was found.
    allow_position_elbow_singularity_fallback: bool = False
    # Position-only shoulder counterpart.  This must be enabled explicitly and
    # is ignored unless the upstream candidate selector certifies that it
    # evaluated the shoulder state and found no clean feasible alternative.
    allow_position_shoulder_singularity_fallback: bool = False
    # Position-only certificate hardening.  When enabled, a singular fallback
    # must be bound to the exact final qref emitted by the coupled selector.
    # False preserves direct/replay compatibility and the previous gate.
    require_position_singularity_candidate_certificate: bool = False
    # When a certificate explicitly claims that no clean branch was
    # available, enter the singular neighborhood with a smaller position
    # step.  The limit is released only after several normal frames so the
    # SDK boundary cannot chatter between hold and pass.
    position_singularity_fallback_step_scale: float = 0.25
    position_singularity_fallback_release_frames: int = 3
    # Formal position teleop may treat the SDK singularity result as advisory
    # when the local J3/J5 proxy and a valid coupled-candidate certificate say
    # the selected qref is outside the local singularity neighborhood.  This
    # never applies to force/replay/direct commands and never bypasses the
    # hard joint, collision, or force guards.
    position_sdk_singularity_advisory: bool = False
    configure_singularity_thresholds: bool = True
    singularity_elbow_threshold_deg: float = 3.0
    singularity_wrist_threshold_deg: float = 3.0
    singularity_shoulder_threshold_m: float = 0.05
    singularity_qpos_init_slowdown: float = DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN
    singularity_replay_slowdown: float = DEFAULT_SINGULARITY_REPLAY_SLOWDOWN
    ik_tuning_json_path: str | None = None
    ik_tuning: RM65IKTuningConfig | None = field(default=None, repr=False)
    ik_framework: str = RM65_IK_FRAMEWORK_ROBOT_MOTION
    builtin_empty_tool_frame: bool = field(default=False, init=False, repr=False)
    position_endpoint_profile: bool = field(default=False, init=False, repr=False)
    # Deployment-only live observation shadow.  Kept at the end of the public
    # init fields so legacy positional construction keeps its old meaning.
    observation_only: bool = False

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        if bool(self.observation_only):
            # These are safety-side effects, not observation prerequisites.
            # Normalize them at the config boundary so an accidental caller
            # cannot re-enable them after the deploy profile has been chosen.
            self.background_streaming_enabled = False
            self.force_position_control = False
            self.force_clear_on_start = False
            self.configure_singularity_thresholds = False
            self.canfd_follow = False
        self.default_tcp_frame = canonical_arm_controlled_frame(self.default_tcp_frame)
        if len(self.max_angvel) != 6:
            raise ValueError("max_angvel must contain 6 joint velocity limits.")
        max_angvel = np.asarray(self.max_angvel, dtype=float).reshape(6)
        if not np.all(np.isfinite(max_angvel)) or np.any(max_angvel <= 0.0):
            raise ValueError("max_angvel must contain 6 positive finite joint velocity limits.")
        self.max_angvel = tuple(float(value) for value in max_angvel)
        if len(self.dry_run_max_joint_step) != 6:
            raise ValueError("dry_run_max_joint_step must contain 6 joint step limits.")
        if self.initial_qpos is not None and len(self.initial_qpos) != 6:
            raise ValueError("initial_qpos must contain exactly 6 joint values.")
        if self.initial_qpos is not None:
            initial_qpos = np.asarray(self.initial_qpos, dtype=float).reshape(6)
            if not np.all(np.isfinite(initial_qpos)):
                raise ValueError("initial_qpos must contain 6 finite joint values.")
            if np.any(initial_qpos < _RM65_ABSOLUTE_Q_MIN_RAD) or np.any(
                initial_qpos > _RM65_ABSOLUTE_Q_MAX_RAD
            ):
                raise ValueError("initial_qpos exceeds the canonical RM65 absolute joint limits.")
        if self.verified_hardware_initial_qpos is not None:
            verified_qpos = np.asarray(
                self.verified_hardware_initial_qpos,
                dtype=float,
            ).reshape(-1)
            if verified_qpos.shape != (6,) or not np.all(np.isfinite(verified_qpos)):
                raise ValueError(
                    "verified_hardware_initial_qpos must contain exactly 6 finite joint values."
                )
            if np.any(verified_qpos < _RM65_ABSOLUTE_Q_MIN_RAD) or np.any(
                verified_qpos > _RM65_ABSOLUTE_Q_MAX_RAD
            ):
                raise ValueError(
                    "verified_hardware_initial_qpos exceeds the canonical RM65 absolute joint limits."
                )
            self.verified_hardware_initial_qpos = tuple(float(value) for value in verified_qpos)
        if not 0.0 < float(self.actual_send_dt_alpha) <= 1.0:
            raise ValueError("actual_send_dt_alpha must be in (0, 1].")
        if float(self.actual_send_dt_min_scale) < 1.0:
            raise ValueError("actual_send_dt_min_scale must be >= 1.0.")
        if float(self.actual_send_dt_max_scale) < float(self.actual_send_dt_min_scale):
            raise ValueError("actual_send_dt_max_scale must be >= actual_send_dt_min_scale.")
        if float(self.actual_send_dt_stall_reset_sec) <= 0.0:
            raise ValueError("actual_send_dt_stall_reset_sec must be positive.")
        if float(self.background_streaming_hz) <= 0.0:
            raise ValueError("background_streaming_hz must be positive.")
        if not 0.0 < float(self.background_streaming_warning_ratio) <= 1.0:
            raise ValueError("background_streaming_warning_ratio must be in (0, 1].")
        if float(self.background_streaming_target_max_age_sec) <= 0.0:
            raise ValueError("background_streaming_target_max_age_sec must be positive.")
        if float(self.background_streaming_shutdown_timeout_sec) <= 0.0:
            raise ValueError("background_streaming_shutdown_timeout_sec must be positive.")
        target_admission_mode = str(
            self.background_streaming_target_admission_mode
        ).strip().lower()
        if target_admission_mode not in RM65_BACKGROUND_TARGET_ADMISSION_CHOICES:
            raise ValueError(
                "background_streaming_target_admission_mode must be one of "
                f"{RM65_BACKGROUND_TARGET_ADMISSION_CHOICES}; got "
                f"{self.background_streaming_target_admission_mode!r}."
            )
        self.background_streaming_target_admission_mode = target_admission_mode
        self.position_command_planner = normalize_position_command_planner(
            self.position_command_planner
        )
        if not math.isfinite(float(self.position_velocity_ramp_sec)) or float(
            self.position_velocity_ramp_sec
        ) <= 0.0:
            raise ValueError("position_velocity_ramp_sec must be finite and positive.")
        target_velocity_filter_tau = float(
            self.position_target_velocity_filter_tau_sec
        )
        if not math.isfinite(target_velocity_filter_tau) or target_velocity_filter_tau < 0.0:
            raise ValueError(
                "position_target_velocity_filter_tau_sec must be finite and non-negative."
            )
        self.position_target_velocity_filter_tau_sec = target_velocity_filter_tau
        if self.position_max_acceleration_rad_s2 is not None:
            max_acceleration = np.asarray(
                self.position_max_acceleration_rad_s2,
                dtype=float,
            ).reshape(-1)
            if (
                max_acceleration.shape != (6,)
                or not np.all(np.isfinite(max_acceleration))
                or np.any(max_acceleration <= 0.0)
            ):
                raise ValueError(
                    "position_max_acceleration_rad_s2 must contain 6 positive "
                    "finite joint acceleration limits."
                )
            self.position_max_acceleration_rad_s2 = tuple(
                float(value) for value in max_acceleration
            )
        if self.position_max_command_step_deg is not None:
            max_command_step_deg = float(self.position_max_command_step_deg)
            if not math.isfinite(max_command_step_deg) or max_command_step_deg <= 0.0:
                raise ValueError(
                    "position_max_command_step_deg must be finite and positive when set."
                )
            if self.position_command_planner not in {
                POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
                POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
            }:
                raise ValueError(
                    "position_max_command_step_deg requires an acceleration/jerk-limited "
                    "position command planner."
                )
            self.position_max_command_step_deg = max_command_step_deg
        if not math.isfinite(float(self.position_degraded_speed_scale)) or not 0.0 < float(
            self.position_degraded_speed_scale
        ) <= 1.0:
            raise ValueError("position_degraded_speed_scale must be finite and in (0, 1].")
        if not math.isfinite(float(self.arm_follow_qpos_alpha)) or not 0.0 < float(
            self.arm_follow_qpos_alpha
        ) <= 1.0:
            raise ValueError("arm_follow_qpos_alpha must be finite and in (0, 1].")
        if (
            self.position_command_planner
            == POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
        ):
            source_hz = float(self.position_sample_aligned_source_hz)
            ratio = float(self.background_streaming_hz) / source_hz
            if (
                not math.isfinite(source_hz)
                or source_hz <= 0.0
                or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9)
            ):
                raise ValueError(
                    "sample-aligned jerk execution requires background_streaming_hz / "
                    "position_sample_aligned_source_hz to be a positive integer."
                )
            self.position_sample_aligned_scale_degraded_target = bool(
                self.position_sample_aligned_scale_degraded_target
            )
            self.position_sample_aligned_source_latch_mode = (
                normalize_sample_aligned_source_latch_mode(
                    self.position_sample_aligned_source_latch_mode
                )
            )
            for name in (
                "position_sample_aligned_jerk_regularization",
                "position_sample_aligned_jerk_delta_regularization",
                "position_sample_aligned_initial_jerk_delta_regularization",
                "position_sample_aligned_terminal_velocity_regularization",
                "position_sample_aligned_terminal_acceleration_regularization",
            ):
                value = float(getattr(self, name))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative.")
                setattr(self, name, value)
            reference_mode = str(
                self.position_sample_aligned_reference_mode
            ).strip().lower()
            if reference_mode not in {
                "linear_hold",
                "linear_hold_nullspace_smooth",
                "causal_hermite",
                "causal_hermite_selective",
            }:
                raise ValueError(
                    "position_sample_aligned_reference_mode must be "
                    "'linear_hold', 'linear_hold_nullspace_smooth', "
                    "'causal_hermite' or 'causal_hermite_selective'."
                )
            self.position_sample_aligned_reference_mode = reference_mode
            observer_window = self.position_sample_aligned_observer_window
            if (
                isinstance(observer_window, bool)
                or int(observer_window) != observer_window
                or int(observer_window) < 2
            ):
                raise ValueError(
                    "position_sample_aligned_observer_window must be an integer >= 2."
                )
            self.position_sample_aligned_observer_window = int(observer_window)
            for name in (
                "position_sample_aligned_observer_efficiency_power",
                "position_sample_aligned_observer_velocity_ema_alpha",
                "position_sample_aligned_observer_velocity_cap_scale",
                "position_sample_aligned_observer_speed_guard_scale",
                "position_sample_aligned_terminal_velocity_displacement_scale",
                "position_sample_aligned_observer_stop_threshold_deg",
                "position_sample_aligned_observer_stop_velocity_decay",
            ):
                value = float(getattr(self, name))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative.")
                setattr(self, name, value)
            for name in (
                "position_sample_aligned_observer_velocity_ema_alpha",
                "position_sample_aligned_observer_velocity_cap_scale",
                "position_sample_aligned_observer_speed_guard_scale",
                "position_sample_aligned_observer_stop_velocity_decay",
            ):
                if float(getattr(self, name)) > 1.0:
                    raise ValueError(f"{name} must be <= 1.")
            coherence = float(
                self.position_sample_aligned_observer_coherence_min
            )
            if not math.isfinite(coherence) or not -1.0 <= coherence <= 1.0:
                raise ValueError(
                    "position_sample_aligned_observer_coherence_min must be in [-1, 1]."
                )
            self.position_sample_aligned_observer_coherence_min = coherence
            if self.position_sample_aligned_terminal_velocity_displacement_scale <= 0.0:
                raise ValueError(
                    "position_sample_aligned_terminal_velocity_displacement_scale "
                    "must be positive."
                )
            horizon_steps = self.position_sample_aligned_horizon_source_steps
            if (
                isinstance(horizon_steps, bool)
                or int(horizon_steps) != horizon_steps
                or int(horizon_steps) < 2
            ):
                raise ValueError(
                    "position_sample_aligned_horizon_source_steps must be an integer >= 2."
                )
            safety_horizon_steps = (
                None
                if self.position_sample_aligned_safety_horizon_source_steps
                is None
                else int(
                    self.position_sample_aligned_safety_horizon_source_steps
                )
            )
            if safety_horizon_steps is not None and (
                isinstance(
                    self.position_sample_aligned_safety_horizon_source_steps,
                    bool,
                )
                or safety_horizon_steps
                != self.position_sample_aligned_safety_horizon_source_steps
                or safety_horizon_steps <= int(horizon_steps)
            ):
                raise ValueError(
                    "position_sample_aligned_safety_horizon_source_steps must "
                    "be an integer greater than "
                    "position_sample_aligned_horizon_source_steps."
                )
            if not math.isclose(
                float(self.arm_follow_qpos_alpha),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "sample-aligned jerk execution requires arm_follow_qpos_alpha=1; "
                    "post-planner EMA would invalidate the jerk certificate."
                )
            self.position_sample_aligned_source_hz = source_hz
            self.position_sample_aligned_horizon_source_steps = int(horizon_steps)
            self.position_sample_aligned_safety_horizon_source_steps = (
                safety_horizon_steps
            )
        if int(self.canfd_trajectory_mode) not in (0, 1, 2):
            raise ValueError("canfd_trajectory_mode must be one of 0, 1, or 2.")
        if not 0 <= int(self.canfd_radio) <= 100:
            raise ValueError("canfd_radio must be in [0, 100].")
        if self.force_sensor not in (0, 1):
            raise ValueError("force_sensor must be 0 (one-axis) or 1 (six-axis).")
        if self.force_mode not in (0, 1):
            raise ValueError("force_mode must be 0 (base frame) or 1 (tool frame).")
        if not 0 <= int(self.force_dir) <= 5:
            raise ValueError("force_dir must be in [0, 5].")
        if float(self.singularity_elbow_threshold_deg) < 0.0:
            raise ValueError("singularity_elbow_threshold_deg must be >= 0.")
        if float(self.singularity_wrist_threshold_deg) < 0.0:
            raise ValueError("singularity_wrist_threshold_deg must be >= 0.")
        if float(self.singularity_shoulder_threshold_m) < 0.0:
            raise ValueError("singularity_shoulder_threshold_m must be >= 0.")
        if not 0.0 < float(self.singularity_qpos_init_slowdown) <= 1.0:
            raise ValueError("singularity_qpos_init_slowdown must be in (0, 1].")
        if not 0.0 < float(self.singularity_replay_slowdown) <= 1.0:
            raise ValueError("singularity_replay_slowdown must be in (0, 1].")
        if not 0.0 < float(self.position_singularity_fallback_step_scale) <= 1.0:
            raise ValueError(
                "position_singularity_fallback_step_scale must be in (0, 1]."
            )
        if int(self.position_singularity_fallback_release_frames) <= 0:
            raise ValueError(
                "position_singularity_fallback_release_frames must be positive."
            )
        self.ik_framework = normalize_rm65_ik_framework(self.ik_framework)
        if self.ik_tuning is None:
            self.ik_tuning = load_ik_tuning_config(self.ik_tuning_json_path)
        if _is_empty_tool_frame_name(self.tool_profile):
            self.builtin_empty_tool_frame = True
            self.tool_profile = None
            self.auto_load_tool_profile = False
            if self.tcp_pose_flange_m_rad is None:
                self.tcp_pose_flange_m_rad = tuple(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD)
        if self.tool_profile:
            from hardware.rm65.calibration_profiles import load_profile

            profile = load_profile(self.tool_profile, arm=self.side if self.side in ("left", "right") else None)
            if str(profile.get("arm")) != self.side:
                raise ValueError(f"tool_profile arm={profile.get('arm')} does not match backend side={self.side}")
            tcp = profile.get("tcp") if isinstance(profile.get("tcp"), dict) else {}
            pose = tcp.get("pose_flange_m_rad")
            if pose is None:
                raise ValueError(f"tool_profile {self.tool_profile!r} does not contain tcp.pose_flange_m_rad")
            self.tcp_pose_flange_m_rad = tuple(float(value) for value in pose)
            if is_position_endpoint_profile(profile):
                self.position_endpoint_profile = True
                self.auto_load_tool_profile = False
        if self.tcp_pose_flange_m_rad is not None:
            pose = np.asarray(self.tcp_pose_flange_m_rad, dtype=float).reshape(-1)
            if pose.shape != (6,) or not np.all(np.isfinite(pose)):
                raise ValueError("tcp_pose_flange_m_rad must contain 6 finite values.")
            self.tcp_pose_flange_m_rad = tuple(float(value) for value in pose)
            _warn_if_tcp_outside_safety_limits(
                owner=f"rm65_{self.side}",
                pose_flange_m_rad=self.tcp_pose_flange_m_rad,
                source="RM65ArmBackendConfig.tcp_pose_flange_m_rad",
            )


@register_backend("rm65_arm")
class RM65ArmBackend(EffectorBackend):
    def __init__(
        self,
        config: RM65ArmBackendConfig,
        *,
        pre_sdk_dispatch_adapter: RM65PreSdkDispatchAdapter | None = None,
        pre_sdk_control_adapter: RM65PreSdkControlAdapter | None = None,
        execution_epoch_guard: ExecutionEpochGuard | None = None,
        worker_drain_participant: ExecutionCancellationParticipantSpec | None = None,
        hardware_idle_worker_preflight_binding: object | None = None,
    ):
        self.config = config
        self._hardware_idle_worker_preflight_binding_id: str | None = None
        if hardware_idle_worker_preflight_binding is not None:
            from .rm65_idle_worker_preflight_binding import (
                RM65IdleWorkerBackendBinding,
            )

            if not isinstance(
                hardware_idle_worker_preflight_binding,
                RM65IdleWorkerBackendBinding,
            ):
                raise TypeError(
                    "hardware_idle_worker_preflight_binding has an unsupported type"
                )
            if any(
                value is not None
                for value in (
                    pre_sdk_dispatch_adapter,
                    pre_sdk_control_adapter,
                    execution_epoch_guard,
                    worker_drain_participant,
                )
            ):
                raise ValueError(
                    "The isolated hardware idle-worker binding owns all adapters "
                    "and epoch/drain inputs; explicit duplicates are forbidden."
                )
            hardware_idle_worker_preflight_binding.validate_config(
                side=config.side,
                connect_hardware=config.connect_hardware,
                background_streaming_enabled=config.background_streaming_enabled,
                force_position_control=config.force_position_control,
                configure_singularity_thresholds=(
                    config.configure_singularity_thresholds
                ),
                initial_qpos=config.initial_qpos,
                now=time.monotonic(),
            )
            pre_sdk_dispatch_adapter = (
                hardware_idle_worker_preflight_binding.pre_sdk_dispatch_adapter
            )
            pre_sdk_control_adapter = (
                hardware_idle_worker_preflight_binding.pre_sdk_control_adapter
            )
            execution_epoch_guard = (
                hardware_idle_worker_preflight_binding.execution_epoch_guard
            )
            worker_drain_participant = (
                hardware_idle_worker_preflight_binding.worker_drain_participant
            )
            self._hardware_idle_worker_preflight_binding_id = (
                hardware_idle_worker_preflight_binding.binding_id
            )
        if pre_sdk_dispatch_adapter is not None and not callable(
            getattr(pre_sdk_dispatch_adapter, "invoke", None)
        ):
            raise TypeError("pre_sdk_dispatch_adapter must provide invoke().")
        # Default-off M5.8-a seam.  No production composition currently binds
        # this adapter, so the legacy path remains the sole real motion owner.
        self._pre_sdk_dispatch_adapter = pre_sdk_dispatch_adapter
        if pre_sdk_control_adapter is not None and not callable(
            getattr(pre_sdk_control_adapter, "invoke", None)
        ):
            raise TypeError("pre_sdk_control_adapter must provide invoke().")
        self._pre_sdk_control_adapter = pre_sdk_control_adapter
        if (execution_epoch_guard is None) != (worker_drain_participant is None):
            raise ValueError(
                "execution_epoch_guard and worker_drain_participant must be configured together"
            )
        if execution_epoch_guard is not None and not isinstance(
            execution_epoch_guard,
            ExecutionEpochGuard,
        ):
            raise TypeError("execution_epoch_guard must be an ExecutionEpochGuard")
        if worker_drain_participant is not None:
            if not isinstance(
                worker_drain_participant,
                ExecutionCancellationParticipantSpec,
            ):
                raise TypeError(
                    "worker_drain_participant must be an ExecutionCancellationParticipantSpec"
                )
            if worker_drain_participant.worker_kind != "backend_stream_worker":
                raise ValueError("RM65 drain participant must be a backend_stream_worker")
        if (
            config.connect_hardware
            and execution_epoch_guard is not None
            and hardware_idle_worker_preflight_binding is None
        ):
            raise RuntimeError(
                "default-off RM65 epoch/drain binding is fake-only until the "
                "production owner/quick-stop/trace preflight is complete"
            )
        self._execution_epoch_guard = execution_epoch_guard
        self._worker_drain_participant = worker_drain_participant
        self.arm = None
        self._solvers: dict[str, object] = {}
        self._current_qpos = (
            None
            if config.initial_qpos is None
            else np.asarray(config.initial_qpos, dtype=float).copy()
        )
        # Commanded and measured state are deliberately separate.  In
        # particular, hardware polling must not move the interpolation anchor
        # after a q_cmd has been accepted by the controller.
        self._last_attempted_qpos: np.ndarray | None = None
        self._last_attempted_timestamp: float | None = None
        self._last_attempted_monotonic: float | None = None
        self._last_attempted_sequence = 0
        self._last_accepted_qpos: np.ndarray | None = (
            None if self._current_qpos is None else self._current_qpos.copy()
        )
        self._last_accepted_timestamp: float | None = None
        self._last_accepted_monotonic: float | None = None
        self._last_accepted_sequence = 0
        self._measured_qpos: np.ndarray | None = None
        self._measured_timestamp: float | None = None
        self._measured_monotonic: float | None = None
        self._measured_sequence = 0
        self._last_local_qpos_normalization: dict[str, object] = {
            "status": "not_set",
            "tolerance_deg": float(np.rad2deg(_RM65_LOCAL_QPOS_OBSERVATION_TOLERANCE_RAD)),
        }
        self._last_command: EffectorCommand | None = None
        self._last_target_qpos: np.ndarray | None = None
        self._last_command_timestamp: float | None = None
        self._last_hardware_qpos: np.ndarray | None = None
        self._last_hardware_pose_6d: np.ndarray | None = None
        self._last_hardware_refresh_time: float | None = None
        self._last_hardware_read_time: float | None = None
        self._last_hardware_read_monotonic: float | None = None
        self._last_hardware_read_success: bool | None = None
        self._last_hardware_read_phase_timing: dict[str, float] = {}
        self._hardware_read_sequence = 0
        self._last_hardware_result_code: int | None = None
        self._last_hardware_result_text: str | None = None
        self._last_hardware_arm_err: int | None = None
        self._last_hardware_sys_err: int | None = None
        self._last_send_result_code: int | None = None
        self._last_send_result_text: str | None = None
        self._hardware_ready = not bool(config.connect_hardware)
        self._hardware_ready_source = (
            "local_backend" if not config.connect_hardware else "not_started"
        )
        self._singularity_config_attempted = False
        self._singularity_protection_ready = False
        self._last_singularity_metadata: dict[str, object] = {
            "singularity_gate_status": "not_checked",
        }
        self._last_singularity_warning_time = 0.0
        self._last_singularity_warning_key: tuple[object, ...] | None = None
        self._position_singularity_fallback_active = False
        self._position_singularity_fallback_normal_frames = 0
        self._position_singularity_fallback_entry_count = 0
        self._base_control_dt = float(config.control_dt)
        self._base_max_joint_step = self._position_max_command_step_rad()
        self._last_send_time: float | None = None
        self._smoothed_send_dt = self._base_control_dt
        self._last_effective_control_dt = self._base_control_dt
        self._last_effective_step_scale = 1.0
        self._prepared_effective_max_joint_step: np.ndarray | None = None
        self._prepared_effective_step_time: float | None = None
        self._state_lock = threading.RLock()
        self._send_lock = threading.RLock()
        # Dedicated state connections may serve both dataset qpos polling and
        # the position-force safety guard.  The RM65 SDK handle is not safe for
        # concurrent request/response calls, so serialize reads without
        # contending with the independent command connection.
        self._hardware_read_lock = threading.RLock()
        self._last_force_query_monotonic: float | None = None
        self._singularity_analysis_lock = threading.RLock()
        self._emergency_stopped = False
        self._stream_target_qpos: np.ndarray | None = None
        self._stream_target_source = ""
        self._stream_target_timestamp: float | None = None
        self._stream_target_monotonic: float | None = None
        self._stream_target_sequence = 0
        self._last_background_target_admission_timing: dict[str, object] = {}
        self._stream_inflight_target_preemption_count = 0
        self._stream_last_inflight_target_preemption: dict[str, object] = {}
        self._stream_sdk_inflight_target_sequence: int | None = None
        self._stream_deferred_target: _DeferredAccelerationTarget | None = None
        self._stream_deferred_target_activation_count = 0
        self._stream_deferred_target_replacement_count = 0
        self._stream_last_deferred_target_activation: dict[str, object] = {}
        self._stream_segment_start_qpos: np.ndarray | None = None
        self._stream_segment_start_time: float | None = None
        self._stream_segment_duration_sec = self._base_control_dt
        self._stream_reference_epoch: int | None = None
        self._stream_reference_sequence: int | None = None
        self._stream_execution_work_item: ExecutionWorkItem | None = None
        self._execution_inflight_work_item: ExecutionWorkItem | None = None
        self._last_completed_execution_work_item: ExecutionWorkItem | None = None
        self._last_old_epoch_invocation_started_at: float | None = None
        self._last_old_epoch_invocation_completed_at: float | None = None
        self._execution_epoch_rejected_count = 0
        self._stream_planner_mode = normalize_position_command_planner(
            config.position_command_planner
        )
        planner_initial_qpos = (
            np.zeros(6, dtype=float)
            if self._current_qpos is None
            else self._current_qpos.copy()
        )
        if self._stream_planner_mode in {
            POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
        }:
            max_velocity = np.asarray(config.max_angvel, dtype=float).reshape(6)
            max_acceleration = (
                max_velocity / float(config.position_velocity_ramp_sec)
                if config.position_max_acceleration_rad_s2 is None
                else np.asarray(
                    config.position_max_acceleration_rad_s2,
                    dtype=float,
                ).reshape(6)
            )
            if (
                self._stream_planner_mode
                == POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
            ):
                # The position step remains a hard bound even when its implied
                # velocity is below the nominal joint velocity profile.
                step_velocity = (
                    np.asarray(self._base_max_joint_step, dtype=float)
                    * float(config.background_streaming_hz)
                )
                effective_velocity = np.minimum(max_velocity, step_velocity)
                self._stream_planner = (
                    SampleAlignedJerkLimitedJointTrajectoryPlanner(
                        JointHorizonLimits(
                            min_position_rad=_RM65_ABSOLUTE_Q_MIN_RAD,
                            max_position_rad=_RM65_ABSOLUTE_Q_MAX_RAD,
                            max_velocity_rad_s=effective_velocity,
                            max_acceleration_rad_s2=max_acceleration,
                            max_jerk_rad_s3=(
                                max_acceleration
                                / float(config.position_velocity_ramp_sec)
                            ),
                        ),
                        source_hz=float(config.position_sample_aligned_source_hz),
                        command_hz=float(config.background_streaming_hz),
                        source_latch_mode=str(
                            config.position_sample_aligned_source_latch_mode
                        ),
                        horizon_source_steps=int(
                            config.position_sample_aligned_horizon_source_steps
                        ),
                        safety_horizon_source_steps=(
                            config.position_sample_aligned_safety_horizon_source_steps
                        ),
                        scale_degraded_target=bool(
                            config.position_sample_aligned_scale_degraded_target
                        ),
                        jerk_regularization=float(
                            config.position_sample_aligned_jerk_regularization
                        ),
                        jerk_delta_regularization=float(
                            config.position_sample_aligned_jerk_delta_regularization
                        ),
                        initial_jerk_delta_regularization=float(
                            config.position_sample_aligned_initial_jerk_delta_regularization
                        ),
                        terminal_velocity_regularization=float(
                            config.position_sample_aligned_terminal_velocity_regularization
                        ),
                        terminal_acceleration_regularization=float(
                            config.position_sample_aligned_terminal_acceleration_regularization
                        ),
                        reference_mode=str(
                            config.position_sample_aligned_reference_mode
                        ),
                        observer_window=int(
                            config.position_sample_aligned_observer_window
                        ),
                        observer_efficiency_power=float(
                            config.position_sample_aligned_observer_efficiency_power
                        ),
                        observer_velocity_ema_alpha=float(
                            config.position_sample_aligned_observer_velocity_ema_alpha
                        ),
                        observer_velocity_cap_scale=float(
                            config.position_sample_aligned_observer_velocity_cap_scale
                        ),
                        observer_coherence_min=float(
                            config.position_sample_aligned_observer_coherence_min
                        ),
                        observer_speed_guard_scale=float(
                            config.position_sample_aligned_observer_speed_guard_scale
                        ),
                        terminal_velocity_displacement_scale=float(
                            config.position_sample_aligned_terminal_velocity_displacement_scale
                        ),
                        observer_stop_threshold_rad=math.radians(
                            float(
                                config.position_sample_aligned_observer_stop_threshold_deg
                            )
                        ),
                        observer_stop_velocity_decay=float(
                            config.position_sample_aligned_observer_stop_velocity_decay
                        ),
                    )
                )
            else:
                self._stream_planner = AccelerationLimitedJointTrajectoryPlanner(
                    JointTrajectoryLimits(
                        max_velocity_rad_s=max_velocity,
                        max_acceleration_rad_s2=max_acceleration,
                        # One 150 Hz SDK tick may never jump farther than one
                        # outer safe-reference step, even after scheduler delay.
                        max_command_step_rad=self._base_max_joint_step,
                    ),
                    velocity_ramp_sec=float(config.position_velocity_ramp_sec),
                    target_velocity_filter_tau_sec=float(
                        config.position_target_velocity_filter_tau_sec
                    ),
                    reference_period_sec=self._base_control_dt,
                )
            self._stream_planner.reset(
                planner_initial_qpos,
                monotonic_time=time.monotonic(),
            )
        else:
            self._stream_planner = JointCommandInterpolator(
                segment_duration_sec=self._base_control_dt,
                stale_after_sec=float(config.background_streaming_target_max_age_sec),
            )
            self._stream_planner.reset(
                {str(config.side): planner_initial_qpos},
                epoch=0,
                monotonic_time=time.monotonic(),
            )
        self._stream_quick_stop_active = False
        self._stream_quick_stop_reason = ""
        self._stream_planner_policy_mode = "linear"
        self._stream_planner_policy_speed_scale = 1.0
        self._stream_planner_policy_reason = "normal"
        self._stream_last_planner_debug: dict[str, object] = {
            "planner_mode": self._stream_planner_mode,
        }
        self._stream_stop_event = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._stream_tick_sequence = 0
        self._stream_scheduled_tick_count = 0
        self._stream_executed_tick_count = 0
        self._stream_skipped_tick_count = 0
        self._stream_send_count = 0
        self._stream_accepted_send_count = 0
        self._stream_failed_send_count = 0
        self._stream_deadline_miss_count = 0
        self._stream_executed_deadline_miss_count = 0
        self._stream_low_rate_count = 0
        self._stream_last_send_perf_time: float | None = None
        self._stream_last_send_dt_sec: float | None = None
        self._stream_actual_hz: float | None = None
        self._stream_max_inter_send_gap_sec: float | None = None
        self._stream_last_call_ms: float | None = None
        self._stream_last_deadline_lag_ms: float | None = None
        self._stream_last_sent_qpos: np.ndarray | None = None
        self._stream_last_sent_step_deg: float | None = None
        self._stream_warning: dict[str, object] | None = None
        self._stream_last_warning_print_time = 0.0
        self._stream_stale_count = 0
        self._stream_last_stale_age_sec: float | None = None
        self._stream_fault: str | None = None
        self._stream_fault_timestamp: float | None = None
        self._stream_fault_monotonic: float | None = None
        self._stream_fault_count = 0
        self._stream_fault_stop_attempted = False
        self._stream_fault_stop_success = False
        self._stream_fault_stop_attempts: list[dict[str, object]] = []
        self._stream_fault_stop_timestamp: float | None = None
        self._stream_shutdown_requested = False
        self._stream_shutdown_complete = False
        self._stream_join_timed_out = False
        self._stream_wakeup_lag_ms_stats = PercentileStats(
            max_samples=_BACKGROUND_AUDIT_STATS_MAX_SAMPLES
        )
        self._stream_raw_wakeup_lag_ms_stats = PercentileStats(
            max_samples=_BACKGROUND_AUDIT_STATS_MAX_SAMPLES
        )
        self._stream_wakeup_gap_ms_stats = PercentileStats(
            max_samples=_BACKGROUND_AUDIT_STATS_MAX_SAMPLES
        )
        self._stream_send_gap_ms_stats = PercentileStats(
            max_samples=_BACKGROUND_AUDIT_STATS_MAX_SAMPLES
        )
        self._stream_sdk_call_ms_stats = PercentileStats(
            max_samples=_BACKGROUND_AUDIT_STATS_MAX_SAMPLES
        )
        self._stream_timing_snapshot_lock = threading.Lock()
        self._stream_timing_snapshot_cache: dict[str, dict[str, int | float | None]] = {}
        self._stream_timing_snapshot_cache_monotonic = 0.0
        self._position_audit_sink: PositionAuditSink | None = None
        self._actuator_command_trace_sink: Callable[..., object] | None = None
        self._actuator_command_trace_error_count = 0
        self._last_actuator_command_trace_error: str | None = None
        self._singularity_transition_trace_sink: Callable[..., object] | None = None
        self._singularity_transition_trace_error_count = 0
        self._last_singularity_transition_trace_error: str | None = None
        self._last_singularity_transition_trace_key: tuple[object, ...] | None = None
        self._position_audit_urdf_root_from_base: np.ndarray | None = None
        self._position_audit_error_count = 0
        self._last_position_audit_error: str | None = None
        self._closed = False
        self._force_position_active = False
        self._ik_tuning = config.ik_tuning
        assert self._ik_tuning is not None
        self._joint_distance_weights = np.asarray(
            self._ik_tuning.branch_normalization.joint_distance_weights,
            dtype=float,
        )
        self._jump_gate_active = False
        self._jump_hold_qpos: np.ndarray | None = None
        self._jump_hold_anchor_pose: np.ndarray | None = None
        self._jump_hold_frames = 0
        self._last_target_pose: np.ndarray | None = None
        self._last_ik_metadata: dict[str, object] = {}
        self._hand_base_to_configured_tcp = self._build_hand_base_to_configured_tcp()
        self._configured_tcp_enabled = not np.allclose(
            self._hand_base_to_configured_tcp,
            np.eye(4, dtype=float),
            atol=1e-9,
        )
        self._active_tool_profile_path: str | None = None
        self._active_tool_name: str | None = None
        self._builtin_empty_tool_frame = bool(config.builtin_empty_tool_frame)
        self._position_endpoint_profile = bool(config.position_endpoint_profile)

    def set_command_control_dt(self, control_dt_sec: float) -> None:
        value = float(control_dt_sec)
        if value <= 0.0:
            raise ValueError(f"{self.name} command control_dt must be positive.")
        with self._send_lock:
            with self._state_lock:
                self.config.control_dt = value
                self._base_control_dt = value
                self._base_max_joint_step = self._position_max_command_step_rad()
                self._smoothed_send_dt = value
                self._last_effective_control_dt = value
                self._last_effective_step_scale = 1.0
                self._prepared_effective_max_joint_step = None
                self._prepared_effective_step_time = None
                self._stream_segment_duration_sec = value
                if isinstance(self._stream_planner, JointCommandInterpolator):
                    self._stream_planner.set_segment_duration_sec(value)
                elif isinstance(
                    self._stream_planner,
                    SampleAlignedJerkLimitedJointTrajectoryPlanner,
                ):
                    # The sample-aligned planner has an explicit immutable
                    # source period and a fixed 120 Hz command grid.
                    pass
                else:
                    self._stream_planner.set_max_command_step(
                        self._base_max_joint_step
                    )
                    self._stream_planner.set_reference_period_sec(value)

    def _position_max_command_step_rad(self) -> np.ndarray:
        explicit_deg = self.config.position_max_command_step_deg
        if explicit_deg is not None:
            return np.full(6, np.deg2rad(float(explicit_deg)), dtype=float)
        return (
            np.asarray(self.config.max_angvel, dtype=float).reshape(6)
            * float(self._base_control_dt)
        )

    def set_position_audit_sink(self, sink: PositionAuditSink | None) -> None:
        """Attach a best-effort sink for command ticks and worker lifecycle events."""
        if sink is not None and not callable(sink):
            raise TypeError("Position audit sink must be callable or None.")
        with self._state_lock:
            self._position_audit_sink = sink

    def set_actuator_command_trace_sink(
        self,
        sink: Callable[..., object] | None,
    ) -> None:
        """Attach the fixed-field, post-SDK actuator-identification trace sink."""
        if sink is not None and not callable(sink):
            raise TypeError("Actuator command trace sink must be callable or None.")
        # Attachment happens before the worker starts and detachment happens
        # before shutdown.  The worker-side attribute read stays lock-free so
        # enabling this diagnostic does not add another hot-path lock.
        with self._state_lock:
            self._actuator_command_trace_sink = sink

    def set_singularity_transition_trace_sink(
        self,
        sink: Callable[..., object] | None,
    ) -> None:
        """Attach a sparse, memory-only position singularity trace sink."""

        if sink is not None and not callable(sink):
            raise TypeError("Singularity transition trace sink must be callable or None.")
        with self._state_lock:
            self._singularity_transition_trace_sink = sink
            self._last_singularity_transition_trace_key = None

    def _emit_singularity_transition_trace(
        self,
        *,
        target_sequence: int,
        monotonic_time: float,
        qref: np.ndarray,
        reference_qpos: np.ndarray,
        command_metadata: Mapping[str, object],
    ) -> None:
        sink = self._singularity_transition_trace_sink
        if sink is None:
            return
        singularity = _metadata_mapping(command_metadata.get("singularity_metadata"))
        certificate_validation = _metadata_mapping(
            singularity.get("position_candidate_certificate_validation")
        )
        gate = str(singularity.get("singularity_gate_status") or "")
        target_code = singularity.get("target_singularity_code")
        current_code = singularity.get("current_singularity_code")
        fallback_requested = bool(
            singularity.get("position_wrist_singularity_fallback_requested", False)
            or singularity.get("position_elbow_singularity_fallback_requested", False)
            or singularity.get("position_shoulder_singularity_fallback_requested", False)
        )
        fallback_allowed = bool(
            singularity.get("position_singularity_fallback_allowed", False)
        )
        key = (
            gate,
            target_code,
            current_code,
            fallback_requested,
            fallback_allowed,
            certificate_validation.get("status"),
        )
        normal = bool(
            gate in {"pass", "not_checked", "disabled"}
            and target_code in {0, None}
            and current_code in {0, None}
            and not fallback_requested
            and not fallback_allowed
        )
        if normal and key == self._last_singularity_transition_trace_key:
            return
        self._last_singularity_transition_trace_key = key
        try:
            sink(
                self.config.side,
                int(target_sequence),
                float(monotonic_time),
                singularity,
                np.asarray(qref, dtype=float).reshape(6),
                np.asarray(reference_qpos, dtype=float).reshape(6),
            )
        except Exception as exc:  # noqa: BLE001 - tracing must never stop control.
            self._singularity_transition_trace_error_count += 1
            self._last_singularity_transition_trace_error = (
                f"{type(exc).__name__}: {exc}"
            )

    def _emit_actuator_command_trace(
        self,
        side: str,
        tick_sequence: int,
        target_sequence: int | None,
        reference_epoch: int | None,
        reference_sequence: int | None,
        scheduled_monotonic_time: float,
        wakeup_monotonic_time: float,
        sdk_attempt_sequence: int | None,
        sdk_attempt_wall_time: float | None,
        sdk_attempt_monotonic_time: float | None,
        sdk_result_monotonic_time: float | None,
        q_cmd: np.ndarray,
        sdk_result_code: int | None,
        sdk_accepted: bool | None,
        sdk_accepted_sequence: int | None,
    ) -> None:
        sink = self._actuator_command_trace_sink
        if sink is None:
            return
        try:
            sink(
                side,
                tick_sequence,
                target_sequence,
                reference_epoch,
                reference_sequence,
                scheduled_monotonic_time,
                wakeup_monotonic_time,
                sdk_attempt_sequence,
                sdk_attempt_wall_time,
                sdk_attempt_monotonic_time,
                sdk_result_monotonic_time,
                q_cmd,
                sdk_result_code,
                sdk_accepted,
                sdk_accepted_sequence,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must never stop control.
            with self._state_lock:
                self._actuator_command_trace_error_count += 1
                self._last_actuator_command_trace_error = (
                    f"{type(exc).__name__}: {exc}"
                )

    def set_position_audit_urdf_root_from_base(
        self,
        transform: Sequence[float] | np.ndarray | None,
    ) -> None:
        """Configure an observational FK transform for accepted-qcmd audit rows.

        This transform is never consulted by command generation or transport.  It
        lets a full position-debug capture compare the accepted SDK qcmd against
        the retargeter's target in the dual-arm URDF root frame.
        """
        pose = None if transform is None else _as_pose_matrix(transform)
        with self._state_lock:
            self._position_audit_urdf_root_from_base = (
                None if pose is None else pose.copy()
            )

    def _emit_position_audit_event(self, event: PositionAuditEvent) -> None:
        with self._state_lock:
            sink = self._position_audit_sink
        if sink is None:
            return
        try:
            sink(event)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never stop control.
            with self._state_lock:
                self._position_audit_error_count += 1
                self._last_position_audit_error = f"{type(exc).__name__}: {exc}"

    def _accepted_qcmd_fk_audit_metadata(self, q_cmd: np.ndarray) -> dict[str, object]:
        """Return optional root-frame FK for an SDK-accepted qcmd.

        This is deliberately diagnostic-only and is called after the SDK result
        has been accepted.  A failure to compute FK is represented in the audit
        row and must not affect the command worker.
        """
        with self._state_lock:
            urdf_root_from_base = (
                None
                if self._position_audit_urdf_root_from_base is None
                else self._position_audit_urdf_root_from_base.copy()
            )
        if urdf_root_from_base is None:
            return {}
        try:
            qcmd_pose_base = self.compute_tcp_pose(qpos=np.asarray(q_cmd, dtype=float))
            qcmd_pose_urdf_root = urdf_root_from_base @ qcmd_pose_base
        except Exception as exc:  # noqa: BLE001 - audit must not change control behavior.
            return {
                "sdk_accepted_qcmd_fk_available": False,
                "sdk_accepted_qcmd_fk_error_type": type(exc).__name__,
                "sdk_accepted_qcmd_fk_error": str(exc),
            }
        return {
            "sdk_accepted_qcmd_fk_available": True,
            "sdk_accepted_qcmd_pose_base": qcmd_pose_base,
            "sdk_accepted_qcmd_pose_urdf_root": qcmd_pose_urdf_root,
            "sdk_accepted_qcmd_fk_frame": "desk",
        }

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def _command_anchor_qpos_locked(self) -> np.ndarray | None:
        if self._last_accepted_qpos is not None:
            return self._last_accepted_qpos.copy()
        if self._current_qpos is not None:
            return self._current_qpos.copy()
        return None

    def _record_command_attempt_locked(
        self,
        qpos: np.ndarray,
        *,
        timestamp: float,
        monotonic_time: float,
    ) -> int:
        self._last_attempted_sequence += 1
        self._last_attempted_qpos = np.asarray(qpos, dtype=float).reshape(6).copy()
        self._last_attempted_timestamp = float(timestamp)
        self._last_attempted_monotonic = float(monotonic_time)
        return int(self._last_attempted_sequence)

    def _record_command_accepted_locked(
        self,
        qpos: np.ndarray,
        *,
        timestamp: float,
        monotonic_time: float,
    ) -> int:
        self._last_accepted_sequence += 1
        accepted = np.asarray(qpos, dtype=float).reshape(6).copy()
        self._last_accepted_qpos = accepted
        # ``current_qpos`` remains a compatibility alias for the last accepted
        # command/local anchor.  Hardware feedback is exposed separately as
        # measured_qpos/hardware_qpos.
        self._current_qpos = accepted.copy()
        self._last_accepted_timestamp = float(timestamp)
        self._last_accepted_monotonic = float(monotonic_time)
        return int(self._last_accepted_sequence)

    def _record_measured_qpos_locked(
        self,
        qpos: np.ndarray,
        *,
        timestamp: float,
        monotonic_time: float,
    ) -> int:
        measured = np.asarray(qpos, dtype=float).reshape(6).copy()
        self._measured_sequence += 1
        self._measured_qpos = measured
        self._last_hardware_qpos = measured.copy()
        self._measured_timestamp = float(timestamp)
        self._measured_monotonic = float(monotonic_time)
        # Before the first attempted command, live feedback is the only valid
        # interpolation anchor.  Once dispatch starts, polling must never
        # overwrite the last accepted q_cmd.
        if self._last_attempted_sequence == 0 and self._last_accepted_sequence == 0:
            self._last_accepted_qpos = measured.copy()
            self._current_qpos = measured.copy()
        return int(self._measured_sequence)

    def _clear_stream_target_locked(self) -> None:
        deferred_sequence = (
            None
            if self._stream_deferred_target is None
            else int(self._stream_deferred_target.sequence)
        )
        self._stream_deferred_target = None
        self._stream_target_qpos = None
        self._stream_target_source = ""
        self._stream_target_timestamp = None
        self._stream_target_monotonic = None
        self._stream_segment_start_qpos = None
        self._stream_segment_start_time = None
        self._stream_reference_epoch = None
        self._stream_reference_sequence = None
        self._stream_execution_work_item = None
        # Invalidate a target already copied by the worker before it can enter
        # the SDK call.  This is what makes hold/close a hard send boundary.
        self._stream_target_sequence = (
            max(
                int(self._stream_target_sequence),
                -1 if deferred_sequence is None else deferred_sequence,
            )
            + 1
        )
        self._stream_quick_stop_active = False
        self._stream_quick_stop_reason = ""
        self._stream_planner_policy_mode = "linear"
        self._stream_planner_policy_speed_scale = 1.0
        self._stream_planner_policy_reason = "normal"

    def _reset_stream_planner_locked(
        self,
        anchor_qpos: np.ndarray,
        *,
        monotonic_time: float,
    ) -> None:
        anchor = np.asarray(anchor_qpos, dtype=float).reshape(6)
        if isinstance(self._stream_planner, JointCommandInterpolator):
            self._stream_planner.reset(
                {str(self.config.side): anchor},
                epoch=0,
                monotonic_time=monotonic_time,
            )
        else:
            self._stream_planner.reset(anchor, monotonic_time=monotonic_time)

    def _rebase_acceleration_planner_after_inflight_target_preemption_locked(
        self,
        *,
        accepted_qpos: np.ndarray,
        completed_monotonic_time: float,
        completed_target_sequence: int,
    ) -> None:
        """Reconcile an unavoidable old SDK completion with the newest target.

        A CANFD call already inside the vendor SDK cannot be cancelled. In
        latest-target admission mode a newer validated target may therefore be
        published while that call is in flight. The physical command must be
        committed as the new planner anchor, then the newest target is safely
        re-applied; committing the stale transactional proposal would either
        fault or leave the planner one accepted qcmd behind the robot.

        This operation intentionally restarts velocity history from zero. That
        is a bounded conservative response to preemption, not an attempt to
        infer motion across an SDK call whose exact application time is unknown.
        """

        if not isinstance(
            self._stream_planner,
            AccelerationLimitedJointTrajectoryPlanner,
        ):
            raise RuntimeError(
                "in-flight target preemption rebase requires the acceleration-limited planner"
            )
        active_target = (
            None
            if self._stream_target_qpos is None
            else self._stream_target_qpos.copy()
        )
        active_sequence = int(self._stream_target_sequence)
        active_published = self._stream_target_monotonic
        active_quick_stop = bool(self._stream_quick_stop_active)
        active_reason = str(self._stream_planner_policy_reason or "normal")
        accepted = np.asarray(accepted_qpos, dtype=float).reshape(6).copy()
        completed = float(completed_monotonic_time)

        self._reset_stream_planner_locked(accepted, monotonic_time=completed)
        if active_target is not None:
            if active_quick_stop:
                self._stream_planner.quick_stop(
                    sequence=active_sequence,
                    monotonic_time=completed,
                    reason=active_reason,
                )
                self._stream_target_qpos = accepted.copy()
            else:
                published = (
                    completed
                    if active_published is None
                    else min(float(active_published), completed)
                )
                self._stream_planner.accept_target(
                    active_target,
                    sequence=active_sequence,
                    published_monotonic_time=published,
                    monotonic_time=completed,
                    speed_scale=float(self._stream_planner_policy_speed_scale),
                    mode=str(self._stream_planner_policy_mode),
                    reason=active_reason,
                    anchor_q_command=accepted,
                )
            self._stream_segment_start_qpos = accepted.copy()
            self._stream_segment_start_time = completed
            self._stream_segment_duration_sec = float(
                self.config.position_velocity_ramp_sec
            )

        self._stream_inflight_target_preemption_count += 1
        self._stream_last_inflight_target_preemption = {
            "completed_target_sequence": int(completed_target_sequence),
            "active_target_sequence": active_sequence,
            "completed_monotonic_time": completed,
            "active_target_present": active_target is not None,
            "quick_stop_active": active_quick_stop,
            "planner_rebased_from_zero_velocity": True,
        }

    def _next_stream_target_sequence_locked(self) -> int:
        deferred_sequence = (
            -1
            if self._stream_deferred_target is None
            else int(self._stream_deferred_target.sequence)
        )
        return max(int(self._stream_target_sequence), deferred_sequence) + 1

    def _activate_deferred_acceleration_target_locked(
        self,
        *,
        accepted_qpos: np.ndarray,
        activated_monotonic_time: float,
        completed_target_sequence: int,
    ) -> bool:
        deferred = self._stream_deferred_target
        if deferred is None:
            return False
        if not isinstance(
            self._stream_planner,
            AccelerationLimitedJointTrajectoryPlanner,
        ):
            raise TypeError(
                "deferred target activation requires the acceleration-limited planner"
            )
        anchor = np.asarray(accepted_qpos, dtype=float).reshape(6).copy()
        activated = float(activated_monotonic_time)
        planner_before = self._stream_planner.snapshot()
        qd_before = np.asarray(
            planner_before["qd_command_rad_s"],
            dtype=float,
        ).reshape(6)
        self._stream_planner.accept_target(
            deferred.qpos,
            sequence=int(deferred.sequence),
            published_monotonic_time=float(deferred.timestamp_monotonic),
            monotonic_time=activated,
            speed_scale=float(deferred.policy_speed_scale),
            mode=str(deferred.policy_mode),
            reason=str(deferred.policy_reason),
            anchor_q_command=anchor,
        )
        planner_after = self._stream_planner.snapshot()
        qd_after = np.asarray(
            planner_after["qd_command_rad_s"],
            dtype=float,
        ).reshape(6)
        self._stream_deferred_target = None
        self._stream_target_sequence = int(deferred.sequence)
        self._stream_target_qpos = deferred.qpos.copy()
        self._stream_target_source = str(deferred.source)
        self._stream_target_timestamp = float(deferred.timestamp_wall)
        self._stream_target_monotonic = float(deferred.timestamp_monotonic)
        self._stream_segment_start_qpos = anchor
        self._stream_segment_start_time = activated
        self._stream_segment_duration_sec = float(
            self.config.position_velocity_ramp_sec
        )
        self._stream_reference_epoch = deferred.reference_epoch
        self._stream_reference_sequence = deferred.reference_sequence
        self._stream_execution_work_item = deferred.execution_work_item
        self._stream_quick_stop_active = False
        self._stream_quick_stop_reason = ""
        self._stream_planner_policy_mode = str(deferred.policy_mode)
        self._stream_planner_policy_speed_scale = float(
            deferred.policy_speed_scale
        )
        self._stream_planner_policy_reason = str(deferred.policy_reason)
        velocity_preserved = bool(
            np.allclose(qd_after, qd_before, rtol=0.0, atol=1e-12)
        )
        self._stream_deferred_target_activation_count += 1
        self._stream_last_deferred_target_activation = {
            "completed_target_sequence": int(completed_target_sequence),
            "activated_target_sequence": int(deferred.sequence),
            "activated_monotonic_time": activated,
            "planner_velocity_preserved": velocity_preserved,
            "planner_rebased_from_zero_velocity": False,
        }
        if not velocity_preserved:
            raise RuntimeError(
                "deferred target activation changed committed planner velocity"
            )
        return True

    def _activate_stream_quick_stop_locked(
        self,
        *,
        reason: str,
        monotonic_time: float,
        source: str,
    ) -> int:
        if not isinstance(
            self._stream_planner,
            (
                AccelerationLimitedJointTrajectoryPlanner,
                SampleAlignedJerkLimitedJointTrajectoryPlanner,
            ),
        ):
            raise RuntimeError(
                "software quick stop requires an acceleration/jerk-limited planner mode"
            )
        anchor = self._command_anchor_qpos_locked()
        if anchor is None:
            anchor = np.zeros(6, dtype=float)
        normalized_reason = str(reason).strip() or "quick_stop"
        planner_snapshot = self._stream_planner.snapshot()
        planner_q = planner_snapshot.get("q_command")
        if planner_q is None or not np.allclose(
            np.asarray(planner_q, dtype=float).reshape(6),
            anchor,
            rtol=0.0,
            atol=1e-10,
        ):
            self._stream_planner.reset(anchor, monotonic_time=float(monotonic_time))
        deferred_sequence = (
            None
            if self._stream_deferred_target is None
            else int(self._stream_deferred_target.sequence)
        )
        self._stream_deferred_target = None
        target_sequence = (
            max(
                int(self._stream_target_sequence),
                -1 if deferred_sequence is None else deferred_sequence,
            )
            + 1
        )
        self._stream_planner.quick_stop(
            sequence=target_sequence,
            monotonic_time=float(monotonic_time),
            reason=normalized_reason,
        )
        self._stream_target_sequence = target_sequence
        self._stream_target_qpos = anchor.copy()
        self._stream_target_source = str(source)
        self._stream_target_timestamp = time.time()
        self._stream_target_monotonic = float(monotonic_time)
        self._stream_segment_start_qpos = anchor.copy()
        self._stream_segment_start_time = float(monotonic_time)
        self._stream_segment_duration_sec = 0.0
        self._stream_quick_stop_active = True
        self._stream_quick_stop_reason = normalized_reason
        self._stream_planner_policy_mode = "quick_stop"
        self._stream_planner_policy_speed_scale = 0.0
        self._stream_planner_policy_reason = normalized_reason
        self._stream_last_planner_debug = {
            "planner_mode": self._stream_planner_mode,
            "trajectory_mode": "quick_stop",
            "trajectory_reason": normalized_reason,
            "trajectory_quick_stop": True,
            "trajectory_speed_scale": 0.0,
        }
        return target_sequence

    def _latch_background_stream_fault(self, error: object) -> None:
        message = str(error)
        if isinstance(error, BaseException):
            message = f"{type(error).__name__}: {error}"
        timestamp = time.time()
        monotonic_time = time.monotonic()
        with self._state_lock:
            self._stream_fault = message
            self._stream_fault_timestamp = timestamp
            self._stream_fault_monotonic = monotonic_time
            self._stream_fault_count += 1
            self._clear_stream_target_locked()
            self._stream_sdk_inflight_target_sequence = None
        self._stream_stop_event.set()

    def _attempt_background_fault_stop(self) -> None:
        """Best-effort hardware stop after the 150 Hz sender becomes unusable.

        Ordinarily the next outer dispatch observes the latched stream fault
        and the bimanual rig stops every backend.  During a manual/continuous
        hold there may be no next dispatch, so the failed worker must not leave
        the arm relying indefinitely on its last CANFD target.
        """

        with self._send_lock:
            with self._state_lock:
                if self._stream_fault_stop_attempted:
                    return
                self._stream_fault_stop_attempted = True
                self._stream_fault_stop_timestamp = time.time()
                arm = self.arm

            attempts: list[dict[str, object]] = []
            success = False
            if arm is None:
                attempts.append(
                    {
                        "method": None,
                        "result_code": None,
                        "result_text": "arm_not_connected",
                        "exception": None,
                    }
                )
            else:
                candidates = (
                    ("Arm_Slow_Stop", getattr(arm, "Arm_Slow_Stop", None)),
                    ("Arm_Stop", getattr(arm, "Arm_Stop", None)),
                )
                for method_name, method in candidates:
                    if not callable(method):
                        continue
                    try:
                        raw_result = self._invoke_pre_sdk_control(
                            seam_id="rm65.safety.background_fault.stop_fallback_loop",
                            permit_kind="safety_control",
                            operation="stop",
                            source="background_stream_fault",
                            sdk_method=method_name,
                            attempt_sequence=len(attempts) + 1,
                            backend_call=method,
                        )
                        result_code, result_text = _parse_rm65_result(raw_result)
                        attempts.append(
                            {
                                "method": method_name,
                                "result_code": result_code,
                                "result_text": result_text,
                                "exception": None,
                            }
                        )
                        if result_code == 0:
                            success = True
                            break
                    except Exception as exc:  # noqa: BLE001 - terminal best effort.
                        attempts.append(
                            {
                                "method": method_name,
                                "result_code": None,
                                "result_text": None,
                                "exception": f"{type(exc).__name__}: {exc}",
                            }
                        )
                if not attempts:
                    attempts.append(
                        {
                            "method": None,
                            "result_code": None,
                            "result_text": "no_stop_method",
                            "exception": None,
                        }
                    )

            with self._state_lock:
                self._stream_fault_stop_attempts = [dict(item) for item in attempts]
                self._stream_fault_stop_success = bool(success)

    def _joint_position_limits_rad(self) -> tuple[np.ndarray, np.ndarray]:
        lower = _RM65_ABSOLUTE_Q_MIN_RAD.copy()
        upper = _RM65_ABSOLUTE_Q_MAX_RAD.copy()
        # A configured solver may intentionally use tighter URDF limits.  Use
        # their intersection, while never allowing a solver to widen the
        # canonical RM65 hardware envelope.
        for solver in tuple(self._solvers.values()):
            try:
                solver_lower = np.asarray(solver.q_min, dtype=float).reshape(6)
                solver_upper = np.asarray(solver.q_max, dtype=float).reshape(6)
            except Exception:
                continue
            if not np.all(np.isfinite(solver_lower)) or not np.all(np.isfinite(solver_upper)):
                continue
            if np.any(solver_upper <= solver_lower):
                continue
            lower = np.maximum(lower, solver_lower)
            upper = np.minimum(upper, solver_upper)
        return lower, upper

    def _validate_final_joint_target(self, qpos: object, *, label: str) -> np.ndarray:
        try:
            target = np.asarray(qpos, dtype=float).reshape(6)
        except Exception as exc:
            raise ValueError(f"{self.name} {label} must be a 6D joint vector.") from exc
        if not np.all(np.isfinite(target)):
            raise ValueError(f"{self.name} {label} contains non-finite joint values.")
        lower, upper = self._joint_position_limits_rad()
        tolerance = _RM65_ABSOLUTE_LIMIT_TOLERANCE_RAD
        invalid = np.flatnonzero((target < lower - tolerance) | (target > upper + tolerance))
        if invalid.size:
            index = int(invalid[0])
            raise ValueError(
                f"{self.name} {label} joint {index + 1}={target[index]:.6f}rad exceeds "
                f"absolute limits [{lower[index]:.6f}, {upper[index]:.6f}]rad."
            )
        dispatch_margin = _RM65_ABSOLUTE_LIMIT_DISPATCH_MARGIN_RAD
        safe_lower = lower + dispatch_margin
        safe_upper = upper - dispatch_margin
        if np.any(safe_upper <= safe_lower):
            raise RuntimeError(
                f"{self.name} absolute joint envelope is narrower than the "
                "configured dispatch margin."
            )
        return np.clip(target, safe_lower, safe_upper)

    def _normalize_local_qpos_observation(
        self,
        qpos: object,
        *,
        label: str,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Bound a hardware-observed local anchor without relaxing transport.

        ``set_local_qpos`` seeds preview/IK/home state from live feedback; it
        is not a motion command. Permit only a tiny SDK degree-to-radian
        quantization overshoot there, then clamp the anchor to the same
        solver/hardware intersection used by final dispatch. Larger feedback
        violations remain explicit failures.
        """

        try:
            observed = np.asarray(qpos, dtype=float).reshape(6)
        except Exception as exc:
            raise ValueError(f"{self.name} {label} must be a 6D joint vector.") from exc
        if not np.all(np.isfinite(observed)):
            raise ValueError(f"{self.name} {label} contains non-finite joint values.")
        lower, upper = self._joint_position_limits_rad()
        tolerance = _RM65_LOCAL_QPOS_OBSERVATION_TOLERANCE_RAD
        invalid = np.flatnonzero((observed < lower - tolerance) | (observed > upper + tolerance))
        if invalid.size:
            index = int(invalid[0])
            raise ValueError(
                f"{self.name} {label} joint {index + 1}={observed[index]:.6f}rad exceeds "
                f"absolute limits [{lower[index]:.6f}, {upper[index]:.6f}]rad by more than "
                f"local-observation tolerance {np.rad2deg(tolerance):.4f}deg."
            )
        normalized = np.clip(observed, lower, upper)
        normalized_indices = np.flatnonzero(np.abs(normalized - observed) > 0.0)
        metadata: dict[str, object] = {
            "status": "clipped" if normalized_indices.size else "exact",
            "label": str(label),
            "tolerance_deg": float(np.rad2deg(tolerance)),
            "clipped_joint_indices": [int(index + 1) for index in normalized_indices.tolist()],
            "observed_qpos_rad": observed.astype(float).tolist(),
            "normalized_qpos_rad": normalized.astype(float).tolist(),
        }
        return normalized.copy(), metadata

    def normalize_local_qpos_observation(
        self,
        qpos: object,
        *,
        label: str = "local qpos",
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Return a command-safe anchor for a hardware-observed qpos.

        This is deliberately an observation-only normalization API.  It
        permits only the tiny SDK degree/radian quantization tolerance used by
        ``set_local_qpos`` and never changes the strict final transport gate.
        """

        return self._normalize_local_qpos_observation(qpos, label=label)

    def start(self) -> None:
        if not self.config.connect_hardware:
            return
        with self._state_lock:
            if self._closed:
                raise RuntimeError(f"{self.name} is closed and cannot be started again.")
            if self.arm is not None:
                if not self._hardware_ready:
                    raise RuntimeError(
                        f"{self.name} has an RM65 connection but no trusted initial joint state."
                    )
                return
        Arm, RM65 = _load_rm65_sdk()
        self.arm = Arm(RM65, self.config.arm_ip)
        try:
            self._auto_load_active_tool_profile()
            if not self.config.observation_only:
                self._configure_singularity_protection()
            if self.config.force_position_control and not self.config.observation_only:
                self._start_force_position_control()
            state = self.refresh_state_from_hardware()
            trusted_hardware_qpos = (
                bool(state.get("hardware_read_success"))
                and state.get("hardware_qpos") is not None
                and int(state.get("measured_sequence") or 0) > 0
            )
            if not trusted_hardware_qpos:
                verified_qpos = self.config.verified_hardware_initial_qpos
                if verified_qpos is None:
                    result_text = str(state.get("hardware_result_text") or "unavailable")
                    raise RuntimeError(
                        f"{self.name} initial RM65 state read did not produce a trusted 6D qpos "
                        f"(result={result_text}); refusing to become ready."
                    )
                fallback_qpos = self._validate_final_joint_target(
                    verified_qpos,
                    label="verified hardware initial qpos",
                )
                with self._state_lock:
                    self._last_accepted_qpos = fallback_qpos.copy()
                    self._current_qpos = fallback_qpos.copy()
                    self._hardware_ready = True
                    self._hardware_ready_source = "verified_hardware_initial_qpos"
            else:
                with self._state_lock:
                    self._hardware_ready = True
                    self._hardware_ready_source = "hardware_feedback"
            with self._state_lock:
                planner_anchor = self._command_anchor_qpos_locked()
                if planner_anchor is None:
                    raise RuntimeError(
                        f"{self.name} has no trusted planner anchor after hardware startup."
                    )
                self._reset_stream_planner_locked(
                    planner_anchor,
                    monotonic_time=time.monotonic(),
                )
            if self.config.background_streaming_enabled and self._stream_thread is None:
                self._stream_stop_event.clear()
                with self._state_lock:
                    self._stream_shutdown_requested = False
                    self._stream_shutdown_complete = False
                    self._stream_join_timed_out = False
                self._stream_thread = threading.Thread(
                    target=self._background_stream_loop,
                    daemon=True,
                    name=f"{self.name}_stream",
                )
                self._stream_thread.start()
        except Exception:
            self._rollback_failed_start()
            raise

    def _rollback_failed_start(self) -> None:
        """Best-effort cleanup after any partially completed hardware start."""

        self._stream_stop_event.set()
        with self._state_lock:
            self._hardware_ready = False
            self._hardware_ready_source = "startup_failed"
            # ``initial_qpos`` is a local/preview seed, not trusted hardware
            # feedback. Do not leave it exposed as a live hardware anchor after
            # startup failure. Even an explicitly verified fallback is cleared
            # if a later startup phase fails; readiness is all-or-nothing.
            self._current_qpos = None
            self._last_accepted_qpos = None
            self._clear_stream_target_locked()
            arm = self.arm
            self._stream_thread = None
        if arm is None:
            return
        if self._force_position_active:
            stop_force = getattr(arm, "Stop_Force_Position_Move", None)
            if callable(stop_force):
                try:
                    self._invoke_pre_sdk_control(
                        seam_id="rm65.cleanup.startup_rollback.force_stop",
                        permit_kind="safety_control",
                        operation="stop",
                        source="startup_rollback",
                        sdk_method="Stop_Force_Position_Move",
                        backend_call=lambda: stop_force(True),
                    )
                except Exception:
                    pass
            self._force_position_active = False
        uninit = getattr(arm, "RM_API_UnInit", None)
        if callable(uninit):
            try:
                uninit()
            except Exception:
                pass
        close_socket = getattr(arm, "Arm_Socket_Close", None)
        if callable(close_socket):
            try:
                close_socket()
            except Exception:
                pass
        with self._state_lock:
            if self.arm is arm:
                self.arm = None

    def is_ready(self) -> bool:
        if not self.config.connect_hardware:
            return True
        with self._state_lock:
            return bool(self.arm is not None and self._hardware_ready)

    def hold_motion(self, reason: str = "manual hold") -> None:
        if self.config.observation_only:
            with self._state_lock:
                self._last_send_result_code = 0
                self._last_send_result_text = f"observation_only_hold:{reason}"
            return
        quick_stop_queued = False
        with self._send_lock:
            with self._state_lock:
                # A force-guard emergency stop is already the authoritative
                # hardware stop. Repeating Arm_Stop from ordinary Hold can
                # return the SDK transient -2 and turn a recoverable GUI Hold
                # into a process exit.
                if self._emergency_stopped:
                    self._clear_stream_target_locked()
                    self._last_target_qpos = None
                    self._last_target_pose = None
                    self._last_send_result_code = 0
                    self._last_send_result_text = (
                        f"hold_skipped_emergency_latched:{reason}"
                    )
                    return
                stream_thread_alive = bool(
                    self._stream_thread is not None and self._stream_thread.is_alive()
                )
                if (
                    isinstance(
                        self._stream_planner,
                        AccelerationLimitedJointTrajectoryPlanner,
                    )
                    and self.config.background_streaming_enabled
                    and self.arm is not None
                    and stream_thread_alive
                    and self._stream_fault is None
                    and not self._closed
                    and not self._emergency_stopped
                ):
                    now = time.monotonic()
                    self._activate_stream_quick_stop_locked(
                        reason=f"manual_hold:{reason}",
                        monotonic_time=now,
                        source="manual_hold",
                    )
                    self._last_target_qpos = self._command_anchor_qpos_locked()
                    self._last_send_result_code = 0
                    self._last_send_result_text = "planner_quick_stop_queued"
                    quick_stop_queued = True
                else:
                    self._clear_stream_target_locked()
                    self._last_target_qpos = None
                self._last_target_pose = None
                self._jump_gate_active = False
                self._jump_hold_qpos = None
                self._jump_hold_anchor_pose = None
                self._jump_hold_frames = 0
            if quick_stop_queued:
                return
            if self.arm is None:
                with self._state_lock:
                    self._last_send_result_code = 0
                    self._last_send_result_text = "hold_no_arm"
                return
            stop_method = getattr(self.arm, "Arm_Slow_Stop", None)
            stop_label = "Arm_Slow_Stop"
            if stop_method is None:
                stop_method = getattr(self.arm, "Arm_Stop", None)
                stop_label = "Arm_Stop"
            if stop_method is None:
                with self._state_lock:
                    self._last_send_result_code = 0
                    self._last_send_result_text = "hold_no_stop_method"
                return
            stop_result = self._invoke_pre_sdk_control(
                seam_id="rm65.safety.hold.direct_stop",
                permit_kind="safety_control",
                operation="hold",
                source=f"manual_hold:{reason}",
                sdk_method=stop_label,
                backend_call=stop_method,
            )
            stop_code, stop_text = _parse_rm65_result(stop_result)
            with self._state_lock:
                self._last_send_result_code = stop_code
                self._last_send_result_text = f"hold:{stop_label}:{stop_text}"
            if stop_code != 0:
                raise RuntimeError(
                    f"{self.name} {stop_label} failed with return code {stop_text}"
                )

    def disengage_position_command_stream(
        self,
        reason: str = "manual interaction",
    ) -> None:
        """Stop once, then withdraw this backend from position interaction.

        ``hold_motion`` intentionally keeps sending the frozen qpos while a
        short safety hold may auto-recover.  The operator Hold screen has a
        different contract: the arm may be moved manually or by another mode.
        In that state a persistent 120 Hz qpos target would fight the other
        interaction, so invalidate it at a hard send boundary after issuing a
        one-shot SDK stop.
        """

        normalized_reason = str(reason).strip() or "manual interaction"
        if self.config.observation_only:
            with self._state_lock:
                self._clear_stream_target_locked()
                self._last_target_qpos = None
                self._last_target_pose = None
                self._last_send_result_code = 0
                self._last_send_result_text = (
                    f"observation_only_disengaged:{normalized_reason}"
                )
            return
        with self._send_lock:
            with self._state_lock:
                self._clear_stream_target_locked()
                self._last_target_qpos = None
                self._last_target_pose = None
                arm = self.arm
                if self._emergency_stopped:
                    self._last_send_result_code = 0
                    self._last_send_result_text = (
                        f"position_stream_disengaged_emergency_latched:{normalized_reason}"
                    )
                    return
            if arm is None:
                with self._state_lock:
                    self._last_send_result_code = 0
                    self._last_send_result_text = "position_stream_disengaged_no_arm"
                return
            stop_method = getattr(arm, "Arm_Slow_Stop", None)
            stop_label = "Arm_Slow_Stop"
            if not callable(stop_method):
                stop_method = getattr(arm, "Arm_Stop", None)
                stop_label = "Arm_Stop"
            if not callable(stop_method):
                raise RuntimeError(
                    f"{self.name} RM65 SDK exposes no stop method while disengaging "
                    "the position command stream"
                )
            stop_result = self._invoke_pre_sdk_control(
                seam_id="rm65.safety.position_stream_disengage.stop",
                permit_kind="safety_control",
                operation="disable_motion",
                source=f"position_stream_disengage:{normalized_reason}",
                sdk_method=stop_label,
                backend_call=stop_method,
            )
            stop_code, stop_text = _parse_rm65_result(stop_result)
            with self._state_lock:
                self._last_send_result_code = stop_code
                self._last_send_result_text = (
                    f"position_stream_disengaged:{normalized_reason}:"
                    f"{stop_label}:{stop_text}"
                )
            if stop_code != 0:
                raise RuntimeError(
                    f"{self.name} {stop_label} failed while disengaging the position "
                    f"command stream with return code {stop_text}"
                )

    def is_emergency_stopped(self) -> bool:
        """Return whether an emergency stop latch still blocks dispatch."""

        with self._state_lock:
            return bool(self._emergency_stopped)

    def manual_rearm_after_emergency_stop(
        self,
        qpos: Sequence[float] | np.ndarray,
    ) -> dict[str, object]:
        """Re-arm this backend after a fresh operator-led runtime Home.

        The method is intentionally explicit and never called by start or a
        safety callback. It requires a finite, hard-limit-valid qpos captured
        from the live arm, refuses a background-worker fault, and restarts the
        high-rate stream only after the old worker has exited.
        """

        reset_qpos = self._validate_final_joint_target(
            qpos,
            label="manual rearm qpos",
        )
        if self.config.observation_only:
            with self._state_lock:
                self._emergency_stopped = False
                self._last_send_result_code = 0
                self._last_send_result_text = "observation_only_manual_rearm"
            return {"rearmed": True, "observation_only": True}

        # Do not hold _send_lock while waiting for the old sender: a worker
        # may be completing its final SDK call under that lock.
        with self._state_lock:
            if self._closed:
                raise RuntimeError(f"{self.name} is closed; cannot manually re-arm.")
            if self.arm is None or not self._hardware_ready:
                raise RuntimeError(
                    f"{self.name} has no ready RM65 connection for manual re-arm."
                )
            if not self._emergency_stopped:
                return {"rearmed": False, "reason": "not_emergency_stopped"}
            if self._stream_fault is not None:
                raise RuntimeError(
                    f"{self.name} background stream fault requires restart: "
                    f"{self._stream_fault}"
                )
            stream_thread = self._stream_thread
            self._stream_stop_event.set()
        if stream_thread is not None and stream_thread is not threading.current_thread():
            stream_thread.join(
                timeout=float(self.config.background_streaming_shutdown_timeout_sec)
            )
            if stream_thread.is_alive():
                raise RuntimeError(
                    f"{self.name} background stream did not stop before manual re-arm."
                )

        timestamp = time.time()
        monotonic_time = time.monotonic()
        with self._send_lock:
            with self._state_lock:
                # Recheck after the worker join; no safety callback may clear
                # this latch concurrently with an operator transition.
                if self._closed:
                    raise RuntimeError(f"{self.name} was closed during manual re-arm.")
                if self.arm is None or not self._hardware_ready:
                    raise RuntimeError(
                        f"{self.name} lost its RM65 connection during manual re-arm."
                    )
                if not self._emergency_stopped:
                    return {"rearmed": False, "reason": "latch_cleared_concurrently"}
                self._stream_thread = None
                self._stream_stop_event.clear()
                self._stream_shutdown_requested = False
                self._stream_shutdown_complete = False
                self._stream_join_timed_out = False
                self._emergency_stopped = False
                self._record_command_accepted_locked(
                    reset_qpos,
                    timestamp=timestamp,
                    monotonic_time=monotonic_time,
                )
                self._record_measured_qpos_locked(
                    reset_qpos,
                    timestamp=timestamp,
                    monotonic_time=monotonic_time,
                )
                self._clear_stream_target_locked()
                self._reset_stream_planner_locked(
                    reset_qpos,
                    monotonic_time=monotonic_time,
                )
                self._reset_background_stream_metrics_locked()
                self._last_target_qpos = None
                self._last_target_pose = None
                self._last_command = None
                self._last_command_timestamp = None
                self._last_send_time = None
                self._smoothed_send_dt = self._base_control_dt
                self._last_effective_control_dt = self._base_control_dt
                self._last_effective_step_scale = 1.0
                self._prepared_effective_max_joint_step = None
                self._prepared_effective_step_time = None
                self._last_send_result_code = 0
                self._last_send_result_text = "manual_rearm_ready"
                if bool(self.config.background_streaming_enabled):
                    self._stream_thread = threading.Thread(
                        target=self._background_stream_loop,
                        daemon=True,
                        name=f"{self.name}_stream",
                    )
                    self._stream_thread.start()
        return {
            "rearmed": True,
            "background_streaming": bool(self.config.background_streaming_enabled),
            "qpos_rad": reset_qpos.astype(float).tolist(),
        }

    def reset_motion_state(self, qpos: Sequence[float] | np.ndarray | None = None) -> None:
        reset_qpos = (
            None
            if qpos is None
            else self._validate_final_joint_target(qpos, label="reset_motion_state qpos")
        )
        with self._send_lock:
            with self._state_lock:
                if reset_qpos is not None:
                    timestamp = time.time()
                    monotonic_time = time.monotonic()
                    self._record_command_accepted_locked(
                        reset_qpos,
                        timestamp=timestamp,
                        monotonic_time=monotonic_time,
                    )
                    self._record_measured_qpos_locked(
                        reset_qpos,
                        timestamp=timestamp,
                        monotonic_time=monotonic_time,
                    )
                self._clear_stream_target_locked()
                planner_anchor = self._command_anchor_qpos_locked()
                if planner_anchor is None:
                    planner_anchor = np.zeros(6, dtype=float)
                self._reset_stream_planner_locked(
                    planner_anchor,
                    monotonic_time=time.monotonic(),
                )
                self._reset_background_stream_metrics_locked()
                self._last_target_qpos = None
                self._last_target_pose = None
                self._last_command = None
                self._last_command_timestamp = None
                self._last_send_time = None
                self._smoothed_send_dt = self._base_control_dt
                self._last_effective_control_dt = self._base_control_dt
                self._last_effective_step_scale = 1.0
                self._prepared_effective_max_joint_step = None
                self._prepared_effective_step_time = None
                self._jump_gate_active = False
                self._jump_hold_qpos = None
                self._jump_hold_anchor_pose = None
                self._jump_hold_frames = 0

    def _configure_singularity_protection(self) -> None:
        if self.config.observation_only:
            return
        with self._singularity_analysis_lock:
            self._configure_singularity_protection_unlocked()

    def _configure_singularity_protection_unlocked(self) -> None:
        if self._singularity_config_attempted:
            return
        self._singularity_protection_ready = False
        if not self.config.singularity_protection_enabled:
            self._singularity_config_attempted = True
            return
        # ``arm is None`` is a transient pre-start state.  Candidate analysis
        # may be wired before the backend connects, so do not permanently burn
        # the one configuration attempt here.
        if self.arm is None:
            return
        self._singularity_config_attempted = True
        analyzer = getattr(self.arm, "Algo_Kin_Singularity_Analyse", None)
        if analyzer is None:
            print(f"WARNING: {self.name} RM65 SDK does not expose singularity analysis; backend gate disabled.")
            return
        if not self.config.configure_singularity_thresholds:
            self._singularity_protection_ready = True
            return
        setter = getattr(self.arm, "Algo_Kin_Set_Singularity_Thresholds", None)
        if setter is None:
            print(
                f"WARNING: {self.name} RM65 SDK does not expose singularity threshold configuration; "
                "backend gate disabled to avoid using unknown SDK defaults."
            )
            return
        try:
            setter(
                self.config.singularity_elbow_threshold_deg,
                self.config.singularity_wrist_threshold_deg,
                self.config.singularity_shoulder_threshold_m,
            )
            self._singularity_protection_ready = True
        except Exception as exc:  # noqa: BLE001 - protection should fail open for teleop.
            print(
                f"WARNING: {self.name} failed to configure singularity thresholds: {exc}; "
                "backend singularity gate disabled."
            )
            return
        getter = getattr(self.arm, "Algo_Kin_Get_Singularity_Thresholds", None)
        if getter is None:
            print(
                f"{self.name}: singularity gate enabled "
                f"elbow={self.config.singularity_elbow_threshold_deg:.3f}deg "
                f"wrist={self.config.singularity_wrist_threshold_deg:.3f}deg "
                f"shoulder={self.config.singularity_shoulder_threshold_m:.3f}m"
            )
            return
        try:
            qe, qw, d = getter()
            print(f"{self.name}: singularity gate enabled elbow={qe:.3f}deg wrist={qw:.3f}deg shoulder={d:.3f}m")
        except Exception as exc:  # noqa: BLE001 - diagnostic only.
            print(f"WARNING: {self.name} failed to read singularity thresholds after setting: {exc}")

    def _analyze_singularity(self, qpos_rad: Sequence[float] | np.ndarray) -> dict[str, object]:
        analysis_started = time.perf_counter()
        lock_wait_started = analysis_started
        with self._singularity_analysis_lock:
            lock_acquired = time.perf_counter()
            if self.arm is None:
                return {
                    "status_code": None,
                    "label": "unavailable",
                    "reason": "no_arm",
                    "analysis_timing": {
                        "lock_wait_ms": (lock_acquired - lock_wait_started) * 1000.0,
                        "sdk_call_ms": 0.0,
                        "total_ms": (time.perf_counter() - analysis_started) * 1000.0,
                    },
                }
            analyzer = getattr(self.arm, "Algo_Kin_Singularity_Analyse", None)
            if analyzer is None:
                return {
                    "status_code": None,
                    "label": "unavailable",
                    "reason": "missing_sdk_method",
                    "analysis_timing": {
                        "lock_wait_ms": (lock_acquired - lock_wait_started) * 1000.0,
                        "sdk_call_ms": 0.0,
                        "total_ms": (time.perf_counter() - analysis_started) * 1000.0,
                    },
                }
            joint_deg = (
                np.asarray(qpos_rad, dtype=float).reshape(6) / np.pi * 180.0
            ).tolist()
            sdk_call_started = time.perf_counter()
            try:
                result = analyzer(joint_deg)
            except Exception as exc:  # noqa: BLE001 - protection should fail open for teleop.
                sdk_call_completed = time.perf_counter()
                return {
                    "status_code": None,
                    "label": "unavailable",
                    "reason": str(exc),
                    "analysis_timing": {
                        "lock_wait_ms": (lock_acquired - lock_wait_started) * 1000.0,
                        "sdk_call_ms": (sdk_call_completed - sdk_call_started) * 1000.0,
                        "total_ms": (sdk_call_completed - analysis_started) * 1000.0,
                    },
                }
            sdk_call_completed = time.perf_counter()
        analysis_timing = {
            "lock_wait_ms": (lock_acquired - lock_wait_started) * 1000.0,
            "sdk_call_ms": (sdk_call_completed - sdk_call_started) * 1000.0,
            "total_ms": (sdk_call_completed - analysis_started) * 1000.0,
        }
        if not isinstance(result, tuple) or len(result) < 2:
            return {
                "status_code": None,
                "label": "unavailable",
                "reason": f"unexpected_result:{result}",
                "analysis_timing": analysis_timing,
            }
        status, distance = result[:2]
        try:
            status_code = int(status)
        except (TypeError, ValueError):
            return {
                "status_code": None,
                "label": "unavailable",
                "reason": f"unexpected_status:{status}",
                "analysis_timing": analysis_timing,
            }
        metadata: dict[str, object] = {
            "status_code": status_code,
            "label": _singularity_status_label(status_code),
            "analysis_timing": analysis_timing,
        }
        try:
            metadata["wrist_center_distance_m"] = float(distance)
        except (TypeError, ValueError):
            metadata["wrist_center_distance_m"] = str(distance)
        return metadata

    def analyze_position_singularity_candidate(
        self,
        qpos_rad: Sequence[float] | np.ndarray,
    ) -> dict[str, object]:
        """Read-only SDK classification for the shared position candidate selector."""

        self._configure_singularity_protection()
        if not self._singularity_protection_ready:
            return {
                "status_code": None,
                "label": "unavailable",
                "reason": "singularity_protection_unavailable",
            }
        return self._analyze_singularity(qpos_rad)

    def prepare_replay_singularity_target(
        self,
        qpos_target: Sequence[float] | np.ndarray,
        *,
        source: str,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Precompute a replay-only slow target before coupled collision validation.

        This method intentionally does not change command state or enqueue a target.
        Deploy and qpos-init sources are excluded by `_is_replay_slow_singularity_source`.
        """

        target = np.asarray(qpos_target, dtype=float).reshape(6).copy()
        metadata: dict[str, object] = {
            "applied": False,
            "source": str(source),
            "singularity_replay_slowdown": float(self.config.singularity_replay_slowdown),
        }
        if not _is_replay_slow_singularity_source(source):
            metadata["reason"] = "non_replay_source"
            return target, metadata
        with self._state_lock:
            reference = self._command_anchor_qpos_locked()
        if reference is None:
            metadata["reason"] = "missing_reference_qpos"
            return target, metadata
        if not self.config.singularity_protection_enabled or self.arm is None:
            metadata["reason"] = "protection_disabled_or_arm_unavailable"
            return target, metadata
        self._configure_singularity_protection()
        if not self._singularity_protection_ready:
            metadata["reason"] = "singularity_gate_unavailable"
            return target, metadata
        target_status = self._analyze_singularity(target)
        current_status = self._analyze_singularity(reference)
        target_code = target_status.get("status_code")
        current_code = current_status.get("status_code")
        target_singular = target_code not in (0, None)
        current_singular = current_code not in (0, None)
        metadata.update(
            {
                "target_singularity_code": target_code,
                "target_singularity_label": target_status.get("label"),
                "current_singularity_code": current_code,
                "current_singularity_label": current_status.get("label"),
            }
        )
        if target_code is None or current_code is None:
            metadata["reason"] = "singularity_analysis_unavailable"
            return target, metadata
        if not target_singular and not current_singular:
            metadata["reason"] = "normal"
            return target, metadata
        max_joint_step = self.get_effective_max_joint_step()
        slow_step = np.maximum(0.0, max_joint_step) * float(
            self.config.singularity_replay_slowdown
        )
        slow_target = reference + np.clip(target - reference, -slow_step, slow_step)
        metadata.update(
            {
                "applied": True,
                "reason": "target_or_reference_singular",
                "max_joint_step_rad": max_joint_step.tolist(),
                "reference_qpos": reference.tolist(),
                "requested_target_qpos": target.tolist(),
                "slow_target_qpos": slow_target.tolist(),
            }
        )
        return slow_target, metadata

    def _record_singularity_metadata(self, metadata: dict[str, object]) -> None:
        with self._state_lock:
            self._last_singularity_metadata = dict(metadata)

    def _position_candidate_certificate_validation(
        self,
        qpos_target: np.ndarray,
        candidate_selection: Mapping[str, object],
    ) -> dict[str, object]:
        certificate = _metadata_mapping(
            candidate_selection.get("candidate_certificate")
        )
        if not certificate:
            return {
                "status": "missing",
                "valid": False,
                "required": bool(
                    self.config.require_position_singularity_candidate_certificate
                ),
            }
        schema = str(certificate.get("schema") or "")
        fingerprint_schema = str(
            certificate.get("qref_fingerprint_schema") or ""
        )
        raw_sequence = certificate.get("sequence")
        fingerprints = _metadata_mapping(
            certificate.get("qref_fingerprint_by_side")
        )
        expected = str(fingerprints.get(self.config.side) or "")
        try:
            sequence = int(raw_sequence)
            sequence_valid = not isinstance(raw_sequence, bool) and sequence >= 0
        except (TypeError, ValueError):
            sequence = -1
            sequence_valid = False
        actual = position_qref_fingerprint(qpos_target)
        valid = bool(
            schema == "robot_motion.rm65.position_candidate_certificate/v1"
            and fingerprint_schema == POSITION_QREF_FINGERPRINT_SCHEMA
            and sequence_valid
            and expected
            and expected == actual
        )
        if valid:
            status = "matched"
        elif schema != "robot_motion.rm65.position_candidate_certificate/v1":
            status = "schema_mismatch"
        elif fingerprint_schema != POSITION_QREF_FINGERPRINT_SCHEMA:
            status = "fingerprint_schema_mismatch"
        elif not sequence_valid:
            status = "invalid_sequence"
        elif not expected:
            status = "side_fingerprint_missing"
        else:
            status = "qref_fingerprint_mismatch"
        result: dict[str, object] = {
            "status": status,
            "valid": valid,
            "required": bool(
                self.config.require_position_singularity_candidate_certificate
            ),
            "sequence": sequence if sequence_valid else None,
            "schema": schema,
            "qref_fingerprint_schema": fingerprint_schema,
            "expected_qref_fingerprint": expected or None,
            "actual_qref_fingerprint": actual,
            "singular_fallback_claimed": bool(
                certificate.get("singular_fallback_claimed", False)
            ),
            "singularity_aware_wrist_traversal": bool(
                _metadata_mapping(
                    certificate.get(
                        "singularity_aware_wrist_traversal_by_side"
                    )
                ).get(self.config.side, False)
            ),
            "singularity_aware_elbow_traversal": bool(
                _metadata_mapping(
                    certificate.get(
                        "singularity_aware_elbow_traversal_by_side"
                    )
                ).get(self.config.side, False)
            ),
        }
        if not valid or bool(certificate.get("singular_fallback_claimed", False)) or bool(
            _metadata_mapping(
                certificate.get("singularity_aware_elbow_traversal_by_side")
            ).get(self.config.side, False)
        ):
            result["certificate"] = certificate
        return result

    def _emit_singularity_warning(self, metadata: dict[str, object]) -> None:
        now = time.time()
        key = (
            metadata.get("singularity_gate_status"),
            metadata.get("target_singularity_label"),
            metadata.get("source"),
        )
        if (
            key == self._last_singularity_warning_key
            and now - self._last_singularity_warning_time < _SINGULARITY_WARNING_INTERVAL_SEC
        ):
            return
        self._last_singularity_warning_key = key
        self._last_singularity_warning_time = now
        print(
            f"WARNING: {self.name} singularity gate hold "
            f"source={metadata.get('source')} "
            f"target={metadata.get('target_singularity_label')} "
            f"current={metadata.get('current_singularity_label')}"
        )

    def _apply_singularity_protection(
        self,
        qpos_target: np.ndarray,
        *,
        reference_qpos: np.ndarray,
        source: str,
        max_joint_step: Sequence[float] | np.ndarray | None = None,
        replay_slowdown_preapplied: bool = False,
        allow_position_wrist_fallback: bool = False,
        position_wrist_fallback_metadata: Mapping[str, object] | None = None,
        allow_position_elbow_fallback: bool = False,
        allow_position_shoulder_fallback: bool = False,
        position_singularity_fallback_metadata: Mapping[str, object] | None = None,
        position_sdk_singularity_advisory_used: bool = False,
        position_candidate_certificate_validation: Mapping[str, object] | None = None,
    ) -> np.ndarray:
        target = np.asarray(qpos_target, dtype=float).reshape(6)
        reference = np.asarray(reference_qpos, dtype=float).reshape(6)
        if not self.config.singularity_protection_enabled:
            self._record_singularity_metadata(
                {
                    "singularity_gate_status": "disabled",
                    "source": source,
                }
            )
            return target
        if self.arm is None:
            self._record_singularity_metadata(
                {
                    "singularity_gate_status": "not_checked",
                    "source": source,
                    "reason": "no_arm",
                }
            )
            return target
        self._configure_singularity_protection()
        if not self._singularity_protection_ready:
            self._record_singularity_metadata(
                {
                    "singularity_gate_status": "unavailable",
                    "source": source,
                }
            )
            return target

        target_status = self._analyze_singularity(target)
        target_code = target_status.get("status_code")
        target_singular = target_code not in (0, None)
        qpos_init_slow_source = _is_qpos_init_slow_singularity_source(source)
        replay_slow_source = _is_replay_slow_singularity_source(source)
        current_status = (
            self._analyze_singularity(reference)
            if target_singular
            or qpos_init_slow_source
            or self._position_singularity_fallback_active
            or (
                replay_slow_source and replay_slowdown_preapplied
            )
            else {"status_code": 0, "label": "normal"}
        )
        current_code = current_status.get("status_code")
        current_singular = current_code not in (0, None)
        entering_singularity = bool(target_singular and current_code == 0)
        wrist_escape_continuation = bool(
            target_singular
            and current_singular
            and target_status.get("label") == "wrist_singularity"
            and current_status.get("label") == "wrist_singularity"
            and allow_position_wrist_fallback
            and abs(float(target[4])) > abs(float(reference[4])) + 1e-9
            and float(target[4]) * float(reference[4]) >= -1e-9
        )
        elbow_escape_continuation = bool(
            target_singular
            and current_singular
            and target_status.get("label") == "elbow_singularity"
            and current_status.get("label") == "elbow_singularity"
            and allow_position_elbow_fallback
            and abs(float(target[2])) > abs(float(reference[2])) + 1e-9
            and float(target[2]) * float(reference[2]) >= -1e-9
        )
        gate_status = "hold_entry" if entering_singularity else "pass"
        allow_wrist_fallback = bool(
            (entering_singularity or wrist_escape_continuation)
            and self.config.allow_position_wrist_singularity_fallback
            and allow_position_wrist_fallback
            and target_status.get("label") == "wrist_singularity"
        )
        allow_elbow_fallback = bool(
            (entering_singularity or elbow_escape_continuation)
            and self.config.allow_position_elbow_singularity_fallback
            and allow_position_elbow_fallback
            and target_status.get("label") == "elbow_singularity"
        )
        allow_shoulder_fallback = bool(
            entering_singularity
            and self.config.allow_position_shoulder_singularity_fallback
            and allow_position_shoulder_fallback
            and target_status.get("label") == "shoulder_singularity"
        )
        allow_position_fallback = bool(
            allow_wrist_fallback or allow_elbow_fallback or allow_shoulder_fallback
        )
        if allow_wrist_fallback:
            gate_status = "allow_position_wrist_singular"
        elif allow_elbow_fallback:
            gate_status = "allow_position_elbow_singular"
        elif allow_shoulder_fallback:
            gate_status = "allow_position_shoulder_singular"
        position_sdk_singularity_advisory_used = bool(
            position_sdk_singularity_advisory_used and allow_position_fallback
        )
        certificate_bound_fallback = bool(
            position_candidate_certificate_validation is not None
            and position_candidate_certificate_validation.get("valid", False)
            and (
                position_candidate_certificate_validation.get(
                    "singular_fallback_claimed", False
                )
                or position_candidate_certificate_validation.get(
                    "singularity_aware_wrist_traversal", False
                )
                or position_candidate_certificate_validation.get(
                    "singularity_aware_elbow_traversal", False
                )
            )
        )
        certificate_context_valid = bool(
            position_candidate_certificate_validation is not None
            and position_candidate_certificate_validation.get("valid", False)
        )
        explicit_position_fallback = bool(
            allow_position_fallback
            and not position_sdk_singularity_advisory_used
            and certificate_bound_fallback
        )
        with self._state_lock:
            fallback_active_before = bool(self._position_singularity_fallback_active)
            if fallback_active_before and not certificate_context_valid:
                # A direct/replay command does not carry the coupled position
                # certificate.  Never let a previous formal-teleop latch
                # change that command's historical singularity semantics.
                self._position_singularity_fallback_active = False
                self._position_singularity_fallback_normal_frames = 0
                fallback_active_before = False
            if (
                (
                    entering_singularity
                    or wrist_escape_continuation
                    or elbow_escape_continuation
                )
                and explicit_position_fallback
            ):
                self._position_singularity_fallback_active = True
                self._position_singularity_fallback_normal_frames = 0
                self._position_singularity_fallback_entry_count += 1
            elif target_singular or current_singular:
                if fallback_active_before:
                    self._position_singularity_fallback_normal_frames = 0
            elif fallback_active_before:
                self._position_singularity_fallback_normal_frames += 1
                if self._position_singularity_fallback_normal_frames >= int(
                    self.config.position_singularity_fallback_release_frames
                ):
                    self._position_singularity_fallback_active = False
                    self._position_singularity_fallback_normal_frames = 0
            fallback_active = bool(self._position_singularity_fallback_active)
            fallback_normal_frames = int(self._position_singularity_fallback_normal_frames)
            fallback_entry_count = int(self._position_singularity_fallback_entry_count)
        rate_limited_position_fallback = bool(
            fallback_active
            and certificate_context_valid
            and not position_sdk_singularity_advisory_used
            and (target_singular or current_singular)
        )
        slow_singularity_mode: str | None = None
        if (
            qpos_init_slow_source
            and target_code is not None
            and (target_singular or current_singular)
        ):
            slow_singularity_mode = "qpos_init"
        elif (
            replay_slow_source
            and replay_slowdown_preapplied
            and target_code is not None
            and (target_singular or current_singular)
        ):
            slow_singularity_mode = "replay"
        slow_singularity = slow_singularity_mode is not None
        if slow_singularity_mode == "qpos_init":
            if entering_singularity:
                gate_status = "slow_qpos_init_entering_singularity"
            elif target_singular:
                gate_status = "slow_qpos_init_current_singular"
            else:
                gate_status = "slow_qpos_init_exiting_singularity"
        elif slow_singularity_mode == "replay":
            if entering_singularity:
                gate_status = "slow_replay_entering_singularity"
            elif target_singular:
                gate_status = "slow_replay_current_singular"
            else:
                gate_status = "slow_replay_exiting_singularity"
        elif target_singular and current_code not in (0, None):
            # Remaining inside a singular neighborhood is not a safe normal
            # tracking state. Only an explicit certified fallback (which is
            # rate-limited below) or a dedicated qpos-init/replay slow path
            # may continue. A normal position target must hold until it points
            # outside the neighborhood; that exit path remains unchanged.
            if not allow_position_fallback and not rate_limited_position_fallback:
                gate_status = "hold_current_singular"
            elif not (
                allow_wrist_fallback
                or allow_elbow_fallback
                or allow_shoulder_fallback
            ):
                gate_status = "allow_current_singular"
        elif target_code is None:
            gate_status = "analysis_unavailable"
        metadata: dict[str, object] = {
            "singularity_gate_status": gate_status,
            "source": source,
            "target_singularity_code": target_code,
            "target_singularity_label": target_status.get("label"),
            "current_singularity_code": current_code,
            "current_singularity_label": current_status.get("label"),
            "position_wrist_escape_continuation": bool(
                wrist_escape_continuation
            ),
            "position_elbow_escape_continuation": bool(
                elbow_escape_continuation
            ),
            "configured_elbow_threshold_deg": float(self.config.singularity_elbow_threshold_deg),
            "configured_wrist_threshold_deg": float(self.config.singularity_wrist_threshold_deg),
            "configured_shoulder_threshold_m": float(self.config.singularity_shoulder_threshold_m),
            "position_wrist_singularity_fallback_requested": bool(
                allow_position_wrist_fallback
            ),
            "position_wrist_singularity_fallback_allowed": bool(allow_wrist_fallback),
            "position_elbow_singularity_fallback_requested": bool(
                allow_position_elbow_fallback
            ),
            "position_elbow_singularity_fallback_allowed": bool(allow_elbow_fallback),
            "position_shoulder_singularity_fallback_requested": bool(
                allow_position_shoulder_fallback
            ),
            "position_shoulder_singularity_fallback_allowed": bool(
                allow_shoulder_fallback
            ),
            "position_singularity_fallback_allowed": bool(allow_position_fallback),
            "position_singularity_fallback_explicit": bool(explicit_position_fallback),
            "position_singularity_fallback_certificate_context_valid": bool(
                certificate_context_valid
            ),
            "position_singularity_fallback_rate_limited": bool(
                rate_limited_position_fallback
            ),
            "position_singularity_fallback_hysteresis_active": bool(fallback_active),
            "position_singularity_fallback_normal_frames": int(fallback_normal_frames),
            "position_singularity_fallback_release_frames": int(
                self.config.position_singularity_fallback_release_frames
            ),
            "position_singularity_fallback_entry_count": int(fallback_entry_count),
            "position_sdk_singularity_advisory": bool(
                self.config.position_sdk_singularity_advisory
            ),
            "position_sdk_singularity_advisory_used": bool(
                position_sdk_singularity_advisory_used
            ),
            "target_singularity_analysis_timing": dict(
                target_status.get("analysis_timing") or {}
            ),
            "current_singularity_analysis_timing": dict(
                current_status.get("analysis_timing") or {}
            ),
        }
        if position_candidate_certificate_validation is not None:
            metadata["position_candidate_certificate_validation"] = dict(
                position_candidate_certificate_validation
            )
        if position_wrist_fallback_metadata is not None:
            metadata["position_wrist_singularity_fallback"] = dict(
                position_wrist_fallback_metadata
            )
        if position_singularity_fallback_metadata is not None:
            metadata["position_singularity_fallback"] = dict(
                position_singularity_fallback_metadata
            )
        if slow_singularity:
            slowdown = (
                float(self.config.singularity_qpos_init_slowdown)
                if slow_singularity_mode == "qpos_init"
                else float(self.config.singularity_replay_slowdown)
            )
            if slow_singularity_mode == "replay" and replay_slowdown_preapplied:
                slow_target = target.copy()
            elif max_joint_step is None:
                slow_target = target.copy()
            else:
                step_limit = np.asarray(max_joint_step, dtype=float).reshape(6)
                step_limit = np.maximum(0.0, step_limit) * slowdown
                slow_target = reference + np.clip(target - reference, -step_limit, step_limit)
            if slow_singularity_mode == "qpos_init":
                metadata["singularity_qpos_init_slowdown"] = slowdown
                metadata["singularity_qpos_init_slow_source"] = True
                metadata["singularity_qpos_init_requested_target"] = target.tolist()
                metadata["singularity_qpos_init_slow_target"] = slow_target.tolist()
            else:
                metadata["singularity_replay_slowdown"] = slowdown
                metadata["singularity_replay_slow_source"] = True
                metadata["singularity_replay_slowdown_preapplied"] = bool(
                    replay_slowdown_preapplied
                )
                metadata["singularity_replay_requested_target"] = target.tolist()
                metadata["singularity_replay_slow_target"] = slow_target.tolist()
            target = slow_target
        if rate_limited_position_fallback and not slow_singularity:
            fallback_scale = float(self.config.position_singularity_fallback_step_scale)
            if max_joint_step is None:
                limited_target = target.copy()
            else:
                step_limit = np.maximum(0.0, np.asarray(max_joint_step, dtype=float).reshape(6))
                step_limit *= fallback_scale
                limited_target = reference + np.clip(target - reference, -step_limit, step_limit)
            metadata.update(
                {
                    "position_singularity_fallback_step_scale": fallback_scale,
                    "position_singularity_fallback_requested_target": target.tolist(),
                    "position_singularity_fallback_limited_target": limited_target.tolist(),
                }
            )
            target = limited_target
        if "wrist_center_distance_m" in target_status:
            metadata["target_wrist_center_distance_m"] = target_status["wrist_center_distance_m"]
        if "wrist_center_distance_m" in current_status:
            metadata["current_wrist_center_distance_m"] = current_status["wrist_center_distance_m"]
        self._record_singularity_metadata(metadata)
        if (
            entering_singularity
            and not slow_singularity
            and not allow_position_fallback
            and not rate_limited_position_fallback
        ):
            self._emit_singularity_warning(metadata)
            return reference.copy()
        if (
            target_singular
            and current_singular
            and not slow_singularity
            and not allow_position_fallback
            and not rate_limited_position_fallback
        ):
            self._emit_singularity_warning(metadata)
            return reference.copy()
        return target

    def _stamp_final_action_target(
        self,
        command: EffectorCommand,
        qpos_target: np.ndarray,
        *,
        source: str,
    ) -> None:
        qpos = np.asarray(qpos_target, dtype=float).reshape(6)
        metadata = command.metadata if isinstance(command.metadata, dict) else dict(command.metadata or {})
        command.metadata = metadata
        metadata.update(
            {
                "action_pose_source": "non_impedance_final_qpos",
                "backend_final_action_source": str(source),
                "backend_final_target_qpos": qpos.copy(),
                "solver_output_qpos": qpos.copy(),
                "singularity_metadata": dict(self._last_singularity_metadata),
            }
        )
        try:
            target_pose = self.compute_tcp_pose(qpos=qpos)
        except Exception as exc:
            target_pose = None
            metadata.update(
                {
                    "target_pose_available": False,
                    "target_pose_error_type": type(exc).__name__,
                    "target_pose_error": str(exc),
                }
            )
        if target_pose is not None:
            metadata.update(
                {
                    "target_pose_available": True,
                    "target_pose_base": target_pose.copy(),
                    "target_pose_tcp_base": target_pose.copy(),
                    "target_pose_controlled_frame": TOOL_TCP_FRAME,
                }
            )
            if command.command_type == "cartesian_pose":
                command.value = target_pose.copy()
        if command.command_type == "joint_vector":
            command.value = qpos.copy()

    def drain_execution_cancellation(
        self,
        cancellation: ExecutionCancellation,
    ) -> WorkerEpochDrainReceipt:
        """Drain one bound RM65 background sender without issuing a stop.

        The caller must first advance the shared execution epoch and must
        drain the upstream reference worker before draining this backend.  The
        send lock waits for an SDK call already in flight; clearing the target
        under the state lock then prevents another old-epoch tick from being
        formed.
        """

        if not isinstance(cancellation, ExecutionCancellation):
            raise TypeError("cancellation must be an ExecutionCancellation")
        guard = self._execution_epoch_guard
        participant = self._worker_drain_participant
        if guard is None or participant is None:
            raise RuntimeError("RM65 execution drain is not configured")
        snapshot = guard.registry.snapshot()
        if (
            snapshot.control_domain_id != cancellation.control_domain_id
            or snapshot.execution_epoch != cancellation.execution_epoch
        ):
            raise RuntimeError(
                "RM65 drain requires the cancellation execution epoch to be current"
            )

        discarded: tuple[str, ...] = ()
        started_at: float | None = None
        completed_at: float | None = None
        with self._send_lock:
            with self._state_lock:
                if self._execution_inflight_work_item is not None:
                    raise RuntimeError(
                        "RM65 SDK invocation remained in flight while holding send lock"
                    )
                item = self._stream_execution_work_item
                if (
                    item is not None
                    and item.execution_epoch
                    == cancellation.previous_execution_epoch
                ):
                    discarded = (item.work_item_id,)
                    self._clear_stream_target_locked()
                completed_item = self._last_completed_execution_work_item
                if (
                    completed_item is not None
                    and completed_item.execution_epoch
                    == cancellation.previous_execution_epoch
                ):
                    started_at = self._last_old_epoch_invocation_started_at
                    completed_at = self._last_old_epoch_invocation_completed_at

        acknowledged_at = max(time.monotonic(), cancellation.effective_at)
        return WorkerEpochDrainReceipt.evaluate(
            cancellation=cancellation,
            participant=participant,
            observed_execution_epoch=snapshot.execution_epoch,
            discarded_work_ids=discarded,
            last_old_epoch_invocation_started_at=started_at,
            last_old_epoch_invocation_completed_at=completed_at,
            in_flight_old_epoch_count=0,
            no_future_old_epoch_calls=True,
            acknowledged_at=acknowledged_at,
        )

    def send_command(self, command: EffectorCommand) -> CommandDispatchReceipt:
        command_phase_started = time.perf_counter()
        if command.effector_name != self.name:
            raise ValueError(
                f"{self.name} received command for effector '{command.effector_name}'."
            )
        if self.config.observation_only:
            raise RuntimeError(
                f"{self.name} is observation-only; refusing RM65 command dispatch."
            )
        entry_state_lock_wait_started = time.perf_counter()
        with self._state_lock:
            entry_state_lock_acquired = time.perf_counter()
            if self._closed:
                raise RuntimeError(f"{self.name} is closed; refusing command dispatch.")
            if self._emergency_stopped:
                raise RuntimeError(f"{self.name} emergency_stop is latched; refusing command dispatch.")
            if self.config.connect_hardware and not self._hardware_ready:
                raise RuntimeError(
                    f"{self.name} has no trusted initial hardware state; refusing command dispatch."
                )
            if self.config.background_streaming_enabled and self._stream_fault is not None:
                raise RuntimeError(
                    f"{self.name} background stream fault is latched; refusing command dispatch: "
                    f"{self._stream_fault}"
                )
            reference_qpos = self._command_anchor_qpos_locked()
            if reference_qpos is None:
                reference_qpos = np.zeros(6, dtype=float)
                self._last_accepted_qpos = reference_qpos.copy()
                self._current_qpos = reference_qpos.copy()
        entry_state_snapshot_completed = time.perf_counter()
        command_start_time = time.time()
        command_source = _command_source(command)
        upstream_execution_work = command.execution_work_item
        guard = self._execution_epoch_guard
        if guard is not None:
            if not isinstance(upstream_execution_work, ExecutionWorkItem):
                raise RuntimeError(
                    f"{self.name} epoch-mediated dispatch requires typed execution work"
                )
            if upstream_execution_work.work_kind not in {
                "position_command",
                "backend_command",
            }:
                raise RuntimeError(
                    f"{self.name} received unsupported execution work kind "
                    f"{upstream_execution_work.work_kind!r}"
                )
            dequeue_check = guard.check_at_dequeue(upstream_execution_work)
            if not dequeue_check.accepted:
                with self._state_lock:
                    self._execution_epoch_rejected_count += 1
                raise RuntimeError(
                    f"{self.name} rejected stale execution work at target admission: "
                    f"{dequeue_check.status}"
                )
        command_metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
        replay_slowdown_preapplied = bool(
            command_metadata.get("singularity_replay_slowdown_preapplied", False)
        ) and bool(
            command_metadata.get(
                "singularity_replay_slowdown_collision_rechecked",
                False,
            )
        )
        max_joint_step = self._consume_prepared_effective_max_joint_step()
        if max_joint_step is None:
            max_joint_step = self._resolve_effective_max_joint_step(
                now=command_start_time,
                update_state=True,
            )
        command_max_joint_step = _command_max_joint_step(command)
        if command_max_joint_step is not None:
            max_joint_step = command_max_joint_step
        max_joint_step_completed = time.perf_counter()
        ik_solve_elapsed_ms: float | None = None

        if command.command_type == "joint_vector":
            qpos_target = np.asarray(command.value, dtype=float)
            if qpos_target.shape != (6,):
                raise ValueError(
                    f"{self.name} expects joint_vector with shape (6,), got {qpos_target.shape}."
                )
            prelimited_solver_qpos = _command_is_prelimited_solver_qpos(command)
            if (
                not prelimited_solver_qpos
                and (self.config.connect_hardware or not self.config.dry_run_fast)
            ):
                delta = np.clip(
                    qpos_target - reference_qpos,
                    -max_joint_step,
                    max_joint_step,
                )
                qpos_target = reference_qpos + delta
        elif command.command_type == "cartesian_pose":
            ik_solve_started = time.perf_counter()
            qpos_target = self._solve_cartesian_command(command, max_joint_step=max_joint_step)
            ik_solve_elapsed_ms = (time.perf_counter() - ik_solve_started) * 1000.0
        else:
            raise ValueError(
                f"{self.name} does not support command_type '{command.command_type}'."
            )
        if ik_solve_elapsed_ms is not None:
            self._last_ik_metadata = {
                **dict(self._last_ik_metadata),
                "ik_status": dict(self._last_ik_metadata).get("ik_jump_gate_status", "pass"),
                "ik_backend_source": "rm65_arm_backend",
                "ik_framework": self.config.ik_framework,
                "ik_backend_solve_ms": float(ik_solve_elapsed_ms),
                "ik_frame_total_ms": float(ik_solve_elapsed_ms),
                "ik_solve_total_ms": float(ik_solve_elapsed_ms),
            }

        qpos_target = self._validate_final_joint_target(
            qpos_target,
            label="pre-singularity target",
        )
        command_metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
        candidate_selection = _metadata_mapping(command_metadata.get("ik_candidate_selection"))
        candidate_certificate_validation = (
            self._position_candidate_certificate_validation(
                qpos_target,
                candidate_selection,
            )
        )
        candidate_certificate = _metadata_mapping(
            candidate_selection.get("candidate_certificate")
        )
        certificate_valid = bool(candidate_certificate_validation.get("valid", False))
        certificate_claims_singular_fallback = bool(
            candidate_certificate.get("singular_fallback_claimed", False)
        )
        certificate_claims_wrist_traversal = bool(
            _metadata_mapping(
                candidate_certificate.get(
                    "singularity_aware_wrist_traversal_by_side"
                )
            ).get(self.config.side, False)
        )
        certificate_claims_elbow_traversal = bool(
            _metadata_mapping(
                candidate_certificate.get(
                    "singularity_aware_elbow_traversal_by_side"
                )
            ).get(self.config.side, False)
        )
        certificate_allows_fallback = bool(
            (certificate_valid and certificate_claims_singular_fallback)
            or not self.config.require_position_singularity_candidate_certificate
        )
        certificate_allows_wrist_traversal = bool(
            certificate_valid and certificate_claims_wrist_traversal
        )
        certificate_allows_elbow_traversal = bool(
            certificate_valid and certificate_claims_elbow_traversal
        )
        certificate_allows_sdk_advisory = bool(
            self.config.position_sdk_singularity_advisory
            and certificate_valid
        )
        wrist_preference = _metadata_mapping(
            candidate_selection.get("wrist_singularity_preference")
        )
        selected_wrist = _metadata_mapping(wrist_preference.get("selected"))
        selected_wrist_sides = _metadata_mapping(selected_wrist.get("sides"))
        selected_side_wrist = _metadata_mapping(selected_wrist_sides.get(self.config.side))
        certified_wrist = _metadata_mapping(
            candidate_certificate.get("final_qref_wrist_singularity")
        )
        certified_wrist_sides = _metadata_mapping(certified_wrist.get("sides"))
        certified_side_wrist = _metadata_mapping(
            certified_wrist_sides.get(self.config.side)
        )
        fallback_side_wrist = (
            certified_side_wrist if certificate_valid else selected_side_wrist
        )
        allow_position_wrist_fallback = bool(
            (
                wrist_preference.get("enabled", False)
                and wrist_preference.get("fallback_to_singular", False)
                and fallback_side_wrist.get("is_wrist_singular", False)
                and certificate_allows_fallback
            )
            or (
                self.config.allow_position_wrist_singularity_fallback
                and certificate_allows_wrist_traversal
            )
        )
        # The SDK is the final hardware safety signal, but its boundary is not
        # identical to the local J5 proxy used by the coupled position
        # selector.  For the formal position path, a valid candidate
        # certificate allows that local classification to take priority when
        # it is explicitly outside the local wrist neighborhood.  The SDK
        # result remains recorded; the backend still requires the explicit
        # position fallback capability and an exact final-qref certificate.
        position_sdk_wrist_advisory_candidate = bool(
            certificate_allows_sdk_advisory
            and wrist_preference.get("enabled", False)
            and selected_side_wrist.get("is_wrist_singular") is False
            # Candidate selection is followed by post-IK qref smoothing.  A
            # valid certificate for the pre-smoothed candidate must not allow
            # the actually emitted qref to enter the local J5 neighborhood at
            # full speed merely because the SDK boundary is advisory.
            and abs(float(np.rad2deg(qpos_target[4])))
            >= float(self.config.singularity_wrist_threshold_deg)
        )
        position_singularity_preference = _metadata_mapping(
            candidate_selection.get("position_singularity_preference")
        )
        selected_position_singularity = _metadata_mapping(
            position_singularity_preference.get("selected")
        )
        selected_position_sides = _metadata_mapping(
            selected_position_singularity.get("sides")
        )
        selected_position_side = _metadata_mapping(
            selected_position_sides.get(self.config.side)
        )
        certified_position = _metadata_mapping(
            candidate_certificate.get("final_qref_position_singularity")
        )
        certified_position_sides = _metadata_mapping(
            certified_position.get("sides")
        )
        certified_position_side = _metadata_mapping(
            certified_position_sides.get(self.config.side)
        )
        fallback_position_side = (
            certified_position_side if certificate_valid else selected_position_side
        )
        position_sdk_singularity_advisory_used = False
        if (
            position_sdk_wrist_advisory_candidate
        ):
            allow_position_wrist_fallback = True
            position_sdk_singularity_advisory_used = True
        allow_position_elbow_fallback = bool(
            position_singularity_preference.get("enabled", False)
            and position_singularity_preference.get("fallback_to_singular", False)
            and int(
                position_singularity_preference.get(
                    "fully_non_singular_count", 0
                )
                or 0
            )
            == 0
            and fallback_position_side.get("is_elbow_singular", False)
            and certificate_allows_fallback
        )
        # A local directional escape can remain inside the SDK elbow band for
        # several bounded frames.  Permit only the exact, fingerprint-bound
        # outward q3 traversal certificate; ordinary singular elbow targets
        # still use the existing explicit fallback branch above.
        if (
            self.config.allow_position_elbow_singularity_fallback
            and certificate_allows_elbow_traversal
            and position_singularity_preference.get("enabled", False)
        ):
            allow_position_elbow_fallback = True
        if (
            certificate_allows_sdk_advisory
            and position_singularity_preference.get("enabled", False)
            and selected_position_side.get("elbow_margin_deg") is not None
        ):
            try:
                local_elbow_non_singular = float(
                    selected_position_side["elbow_margin_deg"]
                ) >= 0.0
            except (TypeError, ValueError):
                local_elbow_non_singular = False
            # The shared solver uses a one-degree selection margin above the
            # SDK elbow threshold. Re-check the final emitted qref too, since
            # output smoothing can move it inside that local neighborhood.
            final_qref_elbow_non_singular = (
                abs(float(np.rad2deg(qpos_target[2])))
                >= float(self.config.singularity_elbow_threshold_deg) + 1.0
            )
            if local_elbow_non_singular and final_qref_elbow_non_singular:
                allow_position_elbow_fallback = True
                position_sdk_singularity_advisory_used = True
        allow_position_shoulder_fallback = bool(
            position_singularity_preference.get("enabled", False)
            and position_singularity_preference.get("fallback_to_singular", False)
            and int(
                position_singularity_preference.get(
                    "fully_non_singular_count", 0
                )
                or 0
            )
            == 0
            and fallback_position_side.get("analyzer_available", False)
            and fallback_position_side.get("is_shoulder_singular", False)
            and certificate_allows_fallback
        )
        pre_singularity_completed = time.perf_counter()
        qpos_target = self._apply_singularity_protection(
            qpos_target,
            reference_qpos=reference_qpos,
            source=command_source,
            max_joint_step=max_joint_step,
            replay_slowdown_preapplied=replay_slowdown_preapplied,
            allow_position_wrist_fallback=allow_position_wrist_fallback,
            position_wrist_fallback_metadata=wrist_preference,
            allow_position_elbow_fallback=allow_position_elbow_fallback,
            allow_position_shoulder_fallback=allow_position_shoulder_fallback,
            position_singularity_fallback_metadata=position_singularity_preference,
            position_sdk_singularity_advisory_used=position_sdk_singularity_advisory_used,
            position_candidate_certificate_validation=(
                candidate_certificate_validation
            ),
        )
        singularity_completed = time.perf_counter()
        qpos_target = self._validate_final_joint_target(qpos_target, label="final target")

        trajectory_policy: _PositionTrajectoryPolicy | None = None
        if (
            self.config.background_streaming_enabled
            and isinstance(
                self._stream_planner,
                (
                    AccelerationLimitedJointTrajectoryPlanner,
                    SampleAlignedJerkLimitedJointTrajectoryPlanner,
                ),
            )
        ):
            command_metadata = (
                command.metadata if isinstance(command.metadata, Mapping) else {}
            )
            trajectory_policy = _position_trajectory_policy(
                command_metadata,
                _metadata_mapping(command_metadata.get("singularity_metadata")),
                degraded_speed_scale=float(self.config.position_degraded_speed_scale),
            )
            if not isinstance(command.metadata, dict):
                command.metadata = dict(command_metadata)
            command.metadata.update(
                {
                    "position_command_planner": self._stream_planner_mode,
                    "position_trajectory_policy_mode": trajectory_policy.mode,
                    "position_trajectory_policy_speed_scale": trajectory_policy.speed_scale,
                    "position_trajectory_policy_quick_stop": trajectory_policy.quick_stop,
                    "position_trajectory_policy_reasons": list(trajectory_policy.reasons),
                }
            )
            if guard is not None and trajectory_policy.quick_stop:
                raise RuntimeError(
                    "epoch-mediated RM65 quick-stop cannot inherit ordinary motion "
                    "work; admit a new-epoch SafetyControlOperation tick instead"
                )

        sample_aligned_admission: SampleAlignedTargetAdmission | None = None
        if (
            isinstance(
                self._stream_planner,
                SampleAlignedJerkLimitedJointTrajectoryPlanner,
            )
            and trajectory_policy is not None
            and not trajectory_policy.quick_stop
            and self.arm is None
        ):
            # EffectorCommand deliberately has no independent clock.  Use the
            # backend admission wall time unless a deterministic replay/test
            # binds an explicit source-cut timestamp in metadata.
            source_timestamp = float(
                command_metadata.get(
                    "sample_aligned_source_timestamp",
                    command_metadata.get(
                        "_parent_command_timestamp",
                        command_start_time,
                    ),
                )
            )
            sample_aligned_admission = self._stream_planner.prepare_target(
                qpos_target,
                source_timestamp=source_timestamp,
                speed_scale=trajectory_policy.speed_scale,
                mode=trajectory_policy.mode,
                reason=trajectory_policy.reason,
            )
            requested_qpos_target = qpos_target.copy()
            qpos_target = np.asarray(
                sample_aligned_admission.admitted_qpos,
                dtype=float,
            ).copy()
            qpos_target = self._validate_final_joint_target(
                qpos_target,
                label="sample-aligned admitted target",
            )
            if not isinstance(command.metadata, dict):
                command.metadata = dict(command.metadata or {})
            command.metadata.update(
                {
                    "sample_aligned_execution": True,
                    "sample_aligned_requested_qpos": requested_qpos_target.tolist(),
                    "sample_aligned_admitted_qpos": qpos_target.tolist(),
                    "sample_aligned_new_source_bucket": bool(
                        sample_aligned_admission.new_source_bucket
                    ),
                    "sample_aligned_admission": dict(
                        sample_aligned_admission.metadata
                    ),
                }
            )

        # The dataset action is the exact endpoint admitted by the selected
        # transport profile, never an unreachable pre-governor request.
        self._stamp_final_action_target(command, qpos_target, source=command_source)

        if self.arm is not None and self.config.background_streaming_enabled:
            target_admission_started = time.perf_counter()
            target_admission_mode = str(
                self.config.background_streaming_target_admission_mode
            )
            effective_target_admission_mode = target_admission_mode
            target_admission_fallback_reason: str | None = None
            if (
                target_admission_mode
                == RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING
                and command_source in _SERIALIZED_QPOS_INIT_TARGET_SOURCES
            ):
                # G-triggered qpos initialization publishes a slow, ordered
                # reference trajectory. Do not let a new reference supersede
                # an SDK call already in flight: the conservative preemption
                # recovery must reset planner velocity to zero because the
                # physical application time is unknown, which turns repeated
                # preemptions into a perceptible stop-start vibration.
                effective_target_admission_mode = (
                    RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK
                )
                target_admission_fallback_reason = (
                    "qpos_init_requires_serialized_velocity_continuity"
                )
            elif (
                target_admission_mode
                == RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING
                and not isinstance(
                    self._stream_planner,
                    AccelerationLimitedJointTrajectoryPlanner,
                )
            ):
                # Linear and sample-aligned admission both mutate cached
                # multi-tick plans at publication time. Keep them on the old
                # serialized contract until they have the acceleration
                # planner's explicit accepted-qcmd rebase operation.
                effective_target_admission_mode = (
                    RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK
                )
                target_admission_fallback_reason = str(
                    f"{self._stream_planner_mode}_planner_requires_"
                    "synchronous_activation"
                )
            send_lock_context = (
                self._send_lock
                if effective_target_admission_mode
                == RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK
                else nullcontext()
            )
            send_lock_wait_started = time.perf_counter()
            with send_lock_context:
                send_lock_acquired = time.perf_counter()
                with self._state_lock:
                    if self._closed:
                        raise RuntimeError(f"{self.name} is closed; refusing command dispatch.")
                    if self._emergency_stopped:
                        raise RuntimeError(f"{self.name} emergency_stop is latched; refusing command dispatch.")
                    if self._stream_fault is not None:
                        raise RuntimeError(
                            f"{self.name} background stream fault is latched; refusing command dispatch: "
                            f"{self._stream_fault}"
                        )
                    segment_start_qpos = self._command_anchor_qpos_locked()
                    if segment_start_qpos is None:
                        segment_start_qpos = reference_qpos.copy()
                    # The high-rate qpos only becomes publishable after IK/safety
                    # completes.  Do not backdate this segment to callback
                    # entry: the 150 Hz worker may have emitted newer commands
                    # while the solver was running.
                    transport_target_wall_time = time.time()
                    transport_target_monotonic = time.monotonic()
                    transport_target_sequence = (
                        self._next_stream_target_sequence_locked()
                    )
                    sdk_inflight_target_sequence_at_admission = (
                        self._stream_sdk_inflight_target_sequence
                    )
                    backend_execution_work = _derive_backend_execution_work_item(
                        upstream_execution_work,
                        side=str(self.config.side),
                        target_sequence=transport_target_sequence,
                    )
                    reference_epoch = _metadata_non_negative_int(
                        command.metadata,
                        "position_reference_epoch",
                    )
                    reference_sequence = _metadata_non_negative_int(
                        command.metadata,
                        "position_reference_sequence",
                    )
                    if (
                        isinstance(
                            self._stream_planner,
                            SampleAlignedJerkLimitedJointTrajectoryPlanner,
                        )
                        and trajectory_policy is not None
                        and not trajectory_policy.quick_stop
                    ):
                        # Planning, action admission and plan activation share
                        # the same send/state lock domain as the 120 Hz commit.
                        # Otherwise a worker can commit one old tick after the
                        # QP snapshots state but before this target is active.
                        source_timestamp = float(
                            command_metadata.get(
                                "sample_aligned_source_timestamp",
                                command_metadata.get(
                                    "_parent_command_timestamp",
                                    command_start_time,
                                ),
                            )
                        )
                        sample_aligned_admission = (
                            self._stream_planner.prepare_target(
                                qpos_target,
                                source_timestamp=source_timestamp,
                                speed_scale=trajectory_policy.speed_scale,
                                mode=trajectory_policy.mode,
                                reason=trajectory_policy.reason,
                            )
                        )
                        requested_qpos_target = qpos_target.copy()
                        qpos_target = np.asarray(
                            sample_aligned_admission.admitted_qpos,
                            dtype=float,
                        ).copy()
                        qpos_target = self._validate_final_joint_target(
                            qpos_target,
                            label="sample-aligned admitted target",
                        )
                        if not isinstance(command.metadata, dict):
                            command.metadata = dict(command.metadata or {})
                        command.metadata.update(
                            {
                                "sample_aligned_execution": True,
                                "sample_aligned_requested_qpos": (
                                    requested_qpos_target.tolist()
                                ),
                                "sample_aligned_admitted_qpos": qpos_target.tolist(),
                                "sample_aligned_new_source_bucket": bool(
                                    sample_aligned_admission.new_source_bucket
                                ),
                                "sample_aligned_admission": dict(
                                    sample_aligned_admission.metadata
                                ),
                            }
                        )
                        self._stamp_final_action_target(
                            command,
                            qpos_target,
                            source=command_source,
                        )
                    sample_aligned_bucket_held = False
                    target_activation_deferred = False
                    if trajectory_policy is not None and trajectory_policy.quick_stop:
                        transport_target_sequence = self._activate_stream_quick_stop_locked(
                            reason=trajectory_policy.reason,
                            monotonic_time=transport_target_monotonic,
                            source=f"{command_source}:quick_stop",
                        )
                        self._stream_target_timestamp = transport_target_wall_time
                        qpos_target = self._command_anchor_qpos_locked()
                        if qpos_target is None:
                            qpos_target = segment_start_qpos.copy()
                    elif isinstance(
                        self._stream_planner,
                        SampleAlignedJerkLimitedJointTrajectoryPlanner,
                    ):
                        if sample_aligned_admission is None:
                            raise RuntimeError(
                                "sample-aligned target admission is missing."
                            )
                        if sample_aligned_admission.new_source_bucket:
                            self._stream_planner.activate_target(
                                sample_aligned_admission,
                                sequence=transport_target_sequence,
                            )
                            self._stream_target_sequence = transport_target_sequence
                            self._stream_target_qpos = qpos_target.copy()
                            self._stream_target_source = command_source
                            self._stream_target_timestamp = transport_target_wall_time
                            self._stream_target_monotonic = transport_target_monotonic
                            self._stream_segment_start_qpos = segment_start_qpos
                            self._stream_segment_start_time = transport_target_monotonic
                            self._stream_segment_duration_sec = (
                                1.0
                                / float(
                                    self.config.position_sample_aligned_source_hz
                                )
                            )
                            self._stream_quick_stop_active = False
                            self._stream_quick_stop_reason = ""
                            self._stream_planner_policy_mode = trajectory_policy.mode
                            self._stream_planner_policy_speed_scale = (
                                trajectory_policy.speed_scale
                            )
                            self._stream_planner_policy_reason = trajectory_policy.reason
                        else:
                            # Intermediate 60 Hz IK frames are still evaluated
                            # upstream for input freshness and safety, but must
                            # not supersede the active 15 Hz execution cut or
                            # create a 120 Hz skipped tick.
                            sample_aligned_bucket_held = True
                            transport_target_sequence = int(
                                self._stream_target_sequence
                            )
                            backend_execution_work = self._stream_execution_work_item
                    elif isinstance(self._stream_planner, JointCommandInterpolator):
                        self._stream_planner.accept_qpos_target(
                            {str(self.config.side): qpos_target},
                            epoch=0,
                            sequence=transport_target_sequence,
                            published_monotonic_time=transport_target_monotonic,
                            monotonic_time=transport_target_monotonic,
                            source=command_source,
                            anchor_qpos_by_side={
                                str(self.config.side): segment_start_qpos,
                            },
                        )
                        self._stream_target_sequence = transport_target_sequence
                        self._stream_target_qpos = qpos_target.copy()
                        self._stream_target_source = command_source
                        self._stream_target_timestamp = transport_target_wall_time
                        self._stream_target_monotonic = transport_target_monotonic
                        self._stream_segment_start_qpos = segment_start_qpos
                        self._stream_segment_start_time = transport_target_monotonic
                        self._stream_segment_duration_sec = self._base_control_dt
                        self._stream_quick_stop_active = False
                        self._stream_quick_stop_reason = ""
                        self._stream_planner_policy_mode = "linear"
                        self._stream_planner_policy_speed_scale = 1.0
                        self._stream_planner_policy_reason = "normal"
                    else:
                        assert trajectory_policy is not None
                        target_activation_deferred = bool(
                            effective_target_admission_mode
                            == RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING
                            and sdk_inflight_target_sequence_at_admission
                            is not None
                        )
                        if target_activation_deferred:
                            if self._stream_deferred_target is not None:
                                self._stream_deferred_target_replacement_count += 1
                            self._stream_deferred_target = (
                                _DeferredAccelerationTarget(
                                    qpos=qpos_target.copy(),
                                    source=str(command_source),
                                    timestamp_wall=float(
                                        transport_target_wall_time
                                    ),
                                    timestamp_monotonic=float(
                                        transport_target_monotonic
                                    ),
                                    sequence=int(transport_target_sequence),
                                    reference_epoch=reference_epoch,
                                    reference_sequence=reference_sequence,
                                    execution_work_item=backend_execution_work,
                                    policy_mode=str(trajectory_policy.mode),
                                    policy_speed_scale=float(
                                        trajectory_policy.speed_scale
                                    ),
                                    policy_reason=str(trajectory_policy.reason),
                                )
                            )
                        else:
                            self._stream_planner.accept_target(
                                qpos_target,
                                sequence=transport_target_sequence,
                                published_monotonic_time=(
                                    transport_target_monotonic
                                ),
                                monotonic_time=transport_target_monotonic,
                                speed_scale=trajectory_policy.speed_scale,
                                mode=trajectory_policy.mode,
                                reason=trajectory_policy.reason,
                                anchor_q_command=segment_start_qpos,
                            )
                            self._stream_target_sequence = (
                                transport_target_sequence
                            )
                            self._stream_target_qpos = qpos_target.copy()
                            self._stream_target_source = command_source
                            self._stream_target_timestamp = (
                                transport_target_wall_time
                            )
                            self._stream_target_monotonic = (
                                transport_target_monotonic
                            )
                            self._stream_segment_start_qpos = (
                                segment_start_qpos
                            )
                            self._stream_segment_start_time = (
                                transport_target_monotonic
                            )
                            self._stream_segment_duration_sec = float(
                                self.config.position_velocity_ramp_sec
                            )
                            self._stream_quick_stop_active = False
                            self._stream_quick_stop_reason = ""
                            self._stream_planner_policy_mode = (
                                trajectory_policy.mode
                            )
                            self._stream_planner_policy_speed_scale = (
                                trajectory_policy.speed_scale
                            )
                            self._stream_planner_policy_reason = (
                                trajectory_policy.reason
                            )
                    if (
                        not sample_aligned_bucket_held
                        and not target_activation_deferred
                    ):
                        self._stream_reference_epoch = reference_epoch
                        self._stream_reference_sequence = reference_sequence
                        self._stream_execution_work_item = backend_execution_work
                    command.metadata.update(
                        {
                            POSITION_TRANSPORT_TARGET_SEQUENCE_KEY: transport_target_sequence,
                            "position_transport_target_wall_time": transport_target_wall_time,
                            "position_transport_target_monotonic_time": transport_target_monotonic,
                            "position_transport_target_source": command_source,
                            "position_transport_planner": self._stream_planner_mode,
                            "position_transport_sample_aligned_bucket_held": bool(
                                sample_aligned_bucket_held
                            ),
                            "position_transport_target_activation_deferred": bool(
                                target_activation_deferred
                            ),
                        }
                    )
                    self._last_target_qpos = qpos_target.copy()
                    self._last_command = command.copy()
                    self._last_command_timestamp = command_start_time
            target_admission_completed = time.perf_counter()
            target_admission_timing = {
                "requested_mode": target_admission_mode,
                "mode": effective_target_admission_mode,
                "fallback_reason": target_admission_fallback_reason,
                "total_ms": max(
                    0.0,
                    (target_admission_completed - target_admission_started) * 1000.0,
                ),
                "send_lock_wait_ms": (
                    max(
                        0.0,
                        (send_lock_acquired - send_lock_wait_started) * 1000.0,
                    )
                    if effective_target_admission_mode
                    == RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK
                    else 0.0
                ),
                "transport_target_sequence": int(transport_target_sequence),
                "sample_aligned_bucket_held": bool(sample_aligned_bucket_held),
                "target_activation_deferred": bool(target_activation_deferred),
                "sdk_inflight_target_sequence": (
                    None
                    if sdk_inflight_target_sequence_at_admission is None
                    else int(sdk_inflight_target_sequence_at_admission)
                ),
            }
            post_admission_metadata_started = time.perf_counter()
            singularity_timing = _metadata_mapping(
                _metadata_mapping(command.metadata.get("singularity_metadata")).get(
                    "target_singularity_analysis_timing"
                )
            )
            command_phase_timing = {
                "entry_state_lock_wait_ms": max(
                    0.0,
                    (entry_state_lock_acquired - entry_state_lock_wait_started)
                    * 1000.0,
                ),
                "entry_state_snapshot_ms": max(
                    0.0,
                    (entry_state_snapshot_completed - entry_state_lock_acquired)
                    * 1000.0,
                ),
                "max_joint_step_setup_ms": max(
                    0.0,
                    (max_joint_step_completed - entry_state_snapshot_completed)
                    * 1000.0,
                ),
                "pre_singularity_ms": max(
                    0.0,
                    (pre_singularity_completed - max_joint_step_completed) * 1000.0,
                ),
                "singularity_protection_ms": max(
                    0.0,
                    (singularity_completed - pre_singularity_completed) * 1000.0,
                ),
                "singularity_analysis_lock_wait_ms": float(
                    singularity_timing.get("lock_wait_ms", 0.0) or 0.0
                ),
                "singularity_analysis_sdk_call_ms": float(
                    singularity_timing.get("sdk_call_ms", 0.0) or 0.0
                ),
                "post_singularity_and_target_admission_ms": max(
                    0.0,
                    (target_admission_completed - singularity_completed) * 1000.0,
                ),
                "target_admission_ms": float(target_admission_timing["total_ms"]),
                "total_through_target_admission_ms": max(
                    0.0,
                    (target_admission_completed - command_phase_started) * 1000.0,
                ),
            }
            command.metadata["rm65_background_target_admission_timing"] = dict(
                target_admission_timing
            )
            command.metadata["rm65_background_command_phase_timing"] = dict(
                command_phase_timing
            )
            post_admission_metadata_completed = time.perf_counter()
            final_state_lock_wait_started = time.perf_counter()
            with self._state_lock:
                final_state_lock_acquired = time.perf_counter()
                self._last_background_target_admission_timing = dict(
                    target_admission_timing
                )
                self._last_command = command.copy()
            final_state_snapshot_completed = time.perf_counter()
            quick_stop_metadata_started = time.perf_counter()
            if trajectory_policy is not None and trajectory_policy.quick_stop:
                # The executable action/reference must describe the exact
                # frozen qpos, not the rejected collision/singularity target.
                self._stamp_final_action_target(
                    command,
                    qpos_target,
                    source=f"{command_source}:planner_quick_stop",
                )
                with self._state_lock:
                    self._last_command = command.copy()
            quick_stop_metadata_completed = time.perf_counter()
            singularity_trace_started = time.perf_counter()
            self._emit_singularity_transition_trace(
                target_sequence=transport_target_sequence,
                monotonic_time=transport_target_monotonic,
                qref=qpos_target,
                reference_qpos=reference_qpos,
                command_metadata=(
                    command.metadata
                    if isinstance(command.metadata, Mapping)
                    else {}
                ),
            )
            singularity_trace_completed = time.perf_counter()
            command_phase_timing.update(
                {
                    "post_admission_metadata_ms": max(
                        0.0,
                        (
                            post_admission_metadata_completed
                            - post_admission_metadata_started
                        )
                        * 1000.0,
                    ),
                    "final_state_lock_wait_ms": max(
                        0.0,
                        (final_state_lock_acquired - final_state_lock_wait_started)
                        * 1000.0,
                    ),
                    "final_state_snapshot_ms": max(
                        0.0,
                        (final_state_snapshot_completed - final_state_lock_acquired)
                        * 1000.0,
                    ),
                    "quick_stop_metadata_ms": max(
                        0.0,
                        (
                            quick_stop_metadata_completed
                            - quick_stop_metadata_started
                        )
                        * 1000.0,
                    ),
                    "singularity_transition_trace_ms": max(
                        0.0,
                        (singularity_trace_completed - singularity_trace_started)
                        * 1000.0,
                    ),
                    "total_send_command_ms": max(
                        0.0,
                        (singularity_trace_completed - command_phase_started)
                        * 1000.0,
                    ),
                }
            )
            command.metadata["rm65_background_command_phase_timing"] = dict(
                command_phase_timing
            )
            return CommandDispatchReceipt(
                effector_name=self.name,
                accepted=True,
                operation="queue_background_joint_target",
                status="queued",
                result_code=None,
                result_text="accepted_by_background_stream",
                metadata={
                    "transport_target_sequence": transport_target_sequence,
                    "transport": "rm65_background_stream",
                    "position_command_planner": self._stream_planner_mode,
                    "trajectory_policy_mode": self._stream_planner_policy_mode,
                    "trajectory_policy_speed_scale": self._stream_planner_policy_speed_scale,
                    "trajectory_policy_reason": self._stream_planner_policy_reason,
                    "target_admission_timing": dict(target_admission_timing),
                    "command_phase_timing": dict(command_phase_timing),
                },
            )

        with self._send_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError(f"{self.name} is closed; refusing command dispatch.")
                if self._emergency_stopped:
                    raise RuntimeError(f"{self.name} emergency_stop is latched; refusing command dispatch.")
                attempt_timestamp = time.time()
                attempt_monotonic = time.monotonic()
                self._record_command_attempt_locked(
                    qpos_target,
                    timestamp=attempt_timestamp,
                    monotonic_time=attempt_monotonic,
                )
            if self.arm is not None:
                try:
                    backend_execution_work = _derive_backend_execution_work_item(
                        upstream_execution_work,
                        side=str(self.config.side),
                        target_sequence=None,
                    )
                    raw_send_result = self._send_joint_target(
                        qpos_target,
                        command_source=command_source,
                        call_path="direct",
                        execution_work_item=backend_execution_work,
                    )
                except Exception as exc:
                    with self._state_lock:
                        self._last_send_result_code = None
                        self._last_send_result_text = f"EXCEPTION: {type(exc).__name__}: {exc}"
                    raise
                send_code, send_text = _parse_rm65_result(raw_send_result)
            else:
                send_code, send_text = 0, "local_no_arm"
            accepted_timestamp = time.time()
            accepted_monotonic = time.monotonic()
            with self._state_lock:
                self._last_send_result_code = send_code
                self._last_send_result_text = send_text
                if send_code == 0:
                    self._record_command_accepted_locked(
                        qpos_target,
                        timestamp=accepted_timestamp,
                        monotonic_time=accepted_monotonic,
                    )
                    self._last_send_time = command_start_time
                    self._last_target_qpos = qpos_target.copy()
                    self._last_command = command.copy()
                    self._last_command_timestamp = command_start_time

        transport_operation = (
            "Force_Position_Move_Joint"
            if self.config.force_position_control
            else "Movej_CANFD"
        )
        receipt = CommandDispatchReceipt(
            effector_name=self.name,
            accepted=bool(send_code == 0),
            operation=transport_operation,
            status="accepted" if send_code == 0 else "rejected",
            result_code=send_code,
            result_text=send_text,
            metadata={"transport": "rm65_direct"},
        )
        if not receipt.accepted:
            raise BackendDispatchError(receipt)
        return receipt

    def _solver_for_tcp_frame(self, tcp_frame: str):
        tcp_frame = canonical_arm_controlled_frame(tcp_frame)
        if tcp_frame not in self._solvers:
            try:
                arm_urdf_path = self.config.arm_urdf_path or require_asset_path("rm65_arm.urdf")
                self._solvers[tcp_frame] = build_rm65_single_arm_ik_solver(
                    RM65SingleArmIKConfig(
                        arm_urdf_path=arm_urdf_path,
                        controlled_frame=tcp_frame,
                        ik_tuning=self._ik_tuning,
                        dry_run_max_iterations=int(self.config.dry_run_max_iterations),
                        dry_run_max_joint_step_rad=tuple(self.config.dry_run_max_joint_step),
                        dry_run_damping=float(self.config.dry_run_damping),
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"{self.name} could not resolve tcp_frame '{tcp_frame}' against the arm URDF."
                ) from exc
        return self._solvers[tcp_frame]

    def _candidate_tcp_frames(self, tcp_frame: str) -> tuple[str, ...]:
        return rm65_single_arm_candidate_frames(canonical_arm_controlled_frame(tcp_frame))

    def _build_hand_base_to_configured_tcp(self) -> np.ndarray:
        pose_flange_m_rad = self._active_tcp_pose_flange_m_rad()
        t_flange_hand_base = _pose6_to_matrix(HAND_BASE_POSE_FLANGE_M_RAD)
        t_flange_tcp = _pose6_to_matrix(pose_flange_m_rad)
        return _invert_transform(t_flange_hand_base) @ t_flange_tcp

    def _active_tcp_pose_flange_m_rad(self) -> tuple[float, ...]:
        pose_flange_m_rad = (
            ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD
            if self.config.tcp_pose_flange_m_rad is None
            else self.config.tcp_pose_flange_m_rad
        )
        return tuple(float(value) for value in pose_flange_m_rad)

    def _active_tcp_source(self) -> str:
        if self._position_endpoint_profile:
            return "position_endpoint_profile"
        if self._builtin_empty_tool_frame:
            return "builtin_empty_arm_endpoint"
        if self.config.tcp_pose_flange_m_rad is None:
            return "arm_endpoint_default"
        if self._active_tool_profile_path:
            return "active_tool_profile"
        if self.config.tool_profile:
            return "configured_tool_profile"
        return "configured_tcp_pose"

    def _set_configured_tcp_pose(self, pose_flange_m_rad: Sequence[float]) -> None:
        pose = np.asarray(pose_flange_m_rad, dtype=float).reshape(6)
        if not np.all(np.isfinite(pose)):
            raise ValueError("tcp_pose_flange_m_rad must contain finite values.")
        self.config.tcp_pose_flange_m_rad = tuple(float(value) for value in pose)
        _warn_if_tcp_outside_safety_limits(
            owner=self.name,
            pose_flange_m_rad=self.config.tcp_pose_flange_m_rad,
            source="active tool profile",
        )
        self._hand_base_to_configured_tcp = self._build_hand_base_to_configured_tcp()
        self._configured_tcp_enabled = not np.allclose(
            self._hand_base_to_configured_tcp,
            np.eye(4, dtype=float),
            atol=1e-9,
        )

    def _auto_load_active_tool_profile(self) -> None:
        if self.arm is None:
            return
        if self.config.tcp_pose_flange_m_rad is not None:
            return
        if self.config.tool_profile or not self.config.auto_load_tool_profile:
            return
        if self.config.side not in ("left", "right"):
            return
        try:
            code, frame = self.arm.Get_Current_Tool_Frame()
        except Exception as exc:
            print(f"WARNING: {self.name} could not read active RM65 tool frame for TCP profile auto-load: {exc}")
            return
        if int(code) != 0:
            print(f"WARNING: {self.name} Get_Current_Tool_Frame failed during TCP profile auto-load: {code}")
            return
        tool_name = _tool_name_from_frame_result((code, frame))
        self._active_tool_name = tool_name or None
        if not tool_name:
            return
        if _is_empty_tool_frame_name(tool_name):
            self._builtin_empty_tool_frame = True
            print(
                f"{self.name}: active tool frame is '{tool_name}'; using built-in "
                "arm_endpoint TCP and skipping calibration profile auto-load for position control."
            )
            return
        try:
            from hardware.rm65.calibration_profiles import load_profile

            profile = load_profile(tool_name, arm=self.config.side)
            profile_tool = str(profile.get("tool_name") or profile.get("name") or "")
            if profile_tool != tool_name:
                raise ValueError(f"profile tool_name={profile_tool!r} does not match active tool {tool_name!r}")
            tcp = profile.get("tcp") if isinstance(profile.get("tcp"), dict) else {}
            pose = tcp.get("pose_flange_m_rad")
            if pose is None:
                print(
                    f"WARNING: {self.name} active tool profile arm={self.config.side} "
                    f"tool={tool_name} has no tcp.pose_flange_m_rad; using arm_endpoint"
                )
                return
            self._set_configured_tcp_pose(pose)
            metrics = tcp_endpoint_delta(self.config.tcp_pose_flange_m_rad)
            self._active_tool_profile_path = str(profile.get("_profile_path") or "")
            print(
                f"{self.name}: auto-loaded TCP profile "
                f"tool={tool_name} path={self._active_tool_profile_path or '(unknown)'} "
                f"configured_tcp_enabled={self._configured_tcp_enabled} "
                f"tcp_endpoint_delta={metrics['translation_m']:.4f}m/{metrics['rotation_deg']:.2f}deg"
            )
        except Exception as exc:
            print(
                f"WARNING: {self.name} no usable TCP profile for active tool "
                f"arm={self.config.side} tool={tool_name}: {exc}. Using arm_endpoint."
            )

    def _target_pose_for_solver(self, tcp_frame: str, target_pose: object) -> np.ndarray:
        matrix = _as_pose_matrix(target_pose)
        controlled_frame = canonical_arm_controlled_frame(tcp_frame)
        if controlled_frame == ARM_ENDPOINT_FRAME:
            return matrix
        if controlled_frame == TOOL_TCP_FRAME and self._configured_tcp_enabled:
            return matrix @ _invert_transform(self._hand_base_to_configured_tcp)
        return matrix

    def _pose_from_solver_frame(self, tcp_frame: str, solver_pose: np.ndarray) -> np.ndarray:
        matrix = np.asarray(solver_pose, dtype=float).reshape(4, 4)
        controlled_frame = canonical_arm_controlled_frame(tcp_frame)
        if controlled_frame == ARM_ENDPOINT_FRAME:
            return matrix
        if controlled_frame == TOOL_TCP_FRAME and self._configured_tcp_enabled:
            return matrix @ self._hand_base_to_configured_tcp
        return matrix

    def _controlled_frame_for_cartesian_command(self, command: EffectorCommand) -> str:
        for key in ("arm_controlled_frame", "target_pose_controlled_frame", "controlled_frame"):
            value = command.metadata.get(key)
            if value:
                return canonical_arm_controlled_frame(str(value))
        return canonical_arm_controlled_frame(command.metadata.get("tcp_frame", self.config.default_tcp_frame))

    def _solve_cartesian_command(
        self,
        command: EffectorCommand,
        *,
        max_joint_step: np.ndarray | None = None,
    ) -> np.ndarray:
        tcp_frame = self._controlled_frame_for_cartesian_command(command)
        solver = self._solver_for_tcp_frame(tcp_frame)
        target_pose = self._target_pose_for_solver(tcp_frame, command.value)
        current_qpos = np.asarray(self._current_qpos, dtype=float).copy()
        command_max_joint_step = (
            np.asarray(max_joint_step, dtype=float).reshape(6)
            if max_joint_step is not None
            else self.get_effective_max_joint_step()
        )
        solve_kwargs = {
            "target_pose": target_pose,
            "current_qpos": current_qpos,
            "fast_iterative": bool(not self.config.connect_hardware and self.config.dry_run_fast),
            "max_joint_step_rad": command_max_joint_step,
            "dry_run_max_iterations": int(self.config.dry_run_max_iterations),
            "dry_run_max_joint_step_rad": tuple(self.config.dry_run_max_joint_step),
            "dry_run_damping": float(self.config.dry_run_damping),
        }
        qpos_target = solve_rm65_single_arm_ik(solver, **solve_kwargs)
        return self._apply_jump_gate(
            solver=solver,
            target_pose=target_pose,
            current_qpos=current_qpos,
            candidate_qpos=np.asarray(qpos_target, dtype=float),
        )

    def _estimate_effective_control_dt(
        self,
        *,
        now: float | None,
        update_state: bool,
    ) -> float:
        if not self.config.use_actual_send_dt_scaling:
            if update_state:
                self._last_effective_control_dt = self._base_control_dt
                self._last_effective_step_scale = 1.0
            return self._base_control_dt

        current_time = time.time() if now is None else float(now)
        smoothed_dt = float(self._smoothed_send_dt)
        if self._last_send_time is None:
            smoothed_dt = self._base_control_dt
        else:
            raw_dt = max(0.0, current_time - float(self._last_send_time))
            if raw_dt > float(self.config.actual_send_dt_stall_reset_sec):
                smoothed_dt = self._base_control_dt
            else:
                alpha = float(self.config.actual_send_dt_alpha)
                smoothed_dt = alpha * raw_dt + (1.0 - alpha) * float(self._smoothed_send_dt)
        effective_dt = float(
            np.clip(
                smoothed_dt,
                self._base_control_dt * float(self.config.actual_send_dt_min_scale),
                self._base_control_dt * float(self.config.actual_send_dt_max_scale),
            )
        )
        if update_state:
            self._smoothed_send_dt = smoothed_dt
            self._last_effective_control_dt = effective_dt
            self._last_effective_step_scale = (
                effective_dt / self._base_control_dt if self._base_control_dt > 0.0 else 1.0
            )
        return effective_dt

    def _resolve_effective_max_joint_step(
        self,
        *,
        now: float | None = None,
        update_state: bool,
    ) -> np.ndarray:
        effective_dt = self._estimate_effective_control_dt(now=now, update_state=update_state)
        return np.asarray(self.config.max_angvel, dtype=float) * effective_dt

    def get_effective_max_joint_step(self, *, now: float | None = None) -> np.ndarray:
        return self._resolve_effective_max_joint_step(now=now, update_state=False).copy()

    def prepare_effective_max_joint_step(self, *, now: float | None = None) -> np.ndarray:
        max_joint_step = self._resolve_effective_max_joint_step(now=now, update_state=True)
        with self._state_lock:
            self._prepared_effective_max_joint_step = max_joint_step.copy()
            self._prepared_effective_step_time = time.time() if now is None else float(now)
        return max_joint_step.copy()

    def _consume_prepared_effective_max_joint_step(self) -> np.ndarray | None:
        with self._state_lock:
            if self._prepared_effective_max_joint_step is None:
                return None
            max_joint_step = self._prepared_effective_max_joint_step.copy()
            self._prepared_effective_max_joint_step = None
            self._prepared_effective_step_time = None
            return max_joint_step

    @staticmethod
    def _interpolate_stream_state(
        *,
        start_qpos: np.ndarray,
        target_qpos: np.ndarray,
        segment_start_time: float | None,
        segment_duration_sec: float,
        now: float,
    ) -> tuple[np.ndarray, float]:
        # Keep this compatibility wrapper for existing diagnostics/tests, but
        # make the actual planner sim/real-neutral in robot_motion.
        return interpolate_joint_command_segment(
            start_qpos,
            target_qpos,
            segment_start_time=segment_start_time,
            segment_duration_sec=segment_duration_sec,
            sample_time=now,
        )

    @staticmethod
    def _interpolate_stream_qpos(
        *,
        start_qpos: np.ndarray,
        target_qpos: np.ndarray,
        segment_start_time: float | None,
        segment_duration_sec: float,
        now: float,
    ) -> np.ndarray:
        qpos, _alpha = RM65ArmBackend._interpolate_stream_state(
            start_qpos=start_qpos,
            target_qpos=target_qpos,
            segment_start_time=segment_start_time,
            segment_duration_sec=segment_duration_sec,
            now=now,
        )
        return qpos

    def _reset_background_stream_metrics_locked(self) -> None:
        self._stream_tick_sequence = 0
        self._stream_scheduled_tick_count = 0
        self._stream_executed_tick_count = 0
        self._stream_skipped_tick_count = 0
        self._stream_send_count = 0
        self._stream_accepted_send_count = 0
        self._stream_failed_send_count = 0
        self._stream_deadline_miss_count = 0
        self._stream_executed_deadline_miss_count = 0
        self._stream_low_rate_count = 0
        self._stream_last_send_perf_time = None
        self._stream_last_send_dt_sec = None
        self._stream_actual_hz = None
        self._stream_max_inter_send_gap_sec = None
        self._stream_last_call_ms = None
        self._stream_last_deadline_lag_ms = None
        self._stream_last_sent_qpos = None
        self._stream_last_sent_step_deg = None
        self._stream_warning = None
        self._stream_last_warning_print_time = 0.0
        self._stream_stale_count = 0
        self._stream_last_stale_age_sec = None
        self._stream_wakeup_lag_ms_stats.reset()
        self._stream_raw_wakeup_lag_ms_stats.reset()
        self._stream_wakeup_gap_ms_stats.reset()
        self._stream_send_gap_ms_stats.reset()
        self._stream_sdk_call_ms_stats.reset()
        with self._stream_timing_snapshot_lock:
            self._stream_timing_snapshot_cache = {}
            self._stream_timing_snapshot_cache_monotonic = 0.0

    def _background_stream_snapshot_locked(self) -> dict[str, object]:
        thread = self._stream_thread
        target_hz = float(self.config.background_streaming_hz)
        warning_ratio = float(self.config.background_streaming_warning_ratio)
        now_monotonic = time.monotonic()
        target_max_age_sec = float(self.config.background_streaming_target_max_age_sec)
        quick_stop_active = bool(self._stream_quick_stop_active)
        if self._stream_target_monotonic is None:
            target_age_sec = None
            target_stale = False
        else:
            target_age_sec, target_stale = reference_age_and_stale(
                published_monotonic_time=float(self._stream_target_monotonic),
                sample_monotonic_time=now_monotonic,
                stale_after_sec=target_max_age_sec,
            )
            if quick_stop_active:
                target_stale = False
        current_inter_send_gap_sec = (
            None
            if self._stream_last_send_perf_time is None
            else max(0.0, now_monotonic - float(self._stream_last_send_perf_time))
        )
        max_inter_send_gap_sec = self._stream_max_inter_send_gap_sec
        if current_inter_send_gap_sec is not None and self._stream_target_qpos is not None:
            max_inter_send_gap_sec = max(
                float(max_inter_send_gap_sec or 0.0),
                float(current_inter_send_gap_sec),
            )
        thread_alive = bool(thread is not None and thread.is_alive())
        health_reasons: list[str] = []
        if self._stream_fault is not None:
            health_reasons.append("fault_latched")
        if target_stale:
            health_reasons.append("target_stale")
        if self._stream_warning is not None and bool(self._stream_warning.get("active")):
            health_reasons.append("low_send_rate")
        if (
            self.config.background_streaming_enabled
            and self.arm is not None
            and not self._stream_shutdown_requested
            and not thread_alive
        ):
            health_reasons.append("thread_not_alive")
        return {
            "enabled": bool(self.config.background_streaming_enabled),
            "hz": target_hz,
            "target_admission_mode": str(
                self.config.background_streaming_target_admission_mode
            ),
            "last_target_admission_timing": dict(
                self._last_background_target_admission_timing
            ),
            "inflight_target_preemption_count": int(
                self._stream_inflight_target_preemption_count
            ),
            "last_inflight_target_preemption": dict(
                self._stream_last_inflight_target_preemption
            ),
            "sdk_inflight_target_sequence": (
                None
                if self._stream_sdk_inflight_target_sequence is None
                else int(self._stream_sdk_inflight_target_sequence)
            ),
            "deferred_target_pending": self._stream_deferred_target is not None,
            "deferred_target_sequence": (
                None
                if self._stream_deferred_target is None
                else int(self._stream_deferred_target.sequence)
            ),
            "deferred_target_activation_count": int(
                self._stream_deferred_target_activation_count
            ),
            "deferred_target_replacement_count": int(
                self._stream_deferred_target_replacement_count
            ),
            "last_deferred_target_activation": dict(
                self._stream_last_deferred_target_activation
            ),
            "warning_threshold_hz": target_hz * warning_ratio,
            "warning_ratio": warning_ratio,
            "thread_alive": thread_alive,
            "target_active": self._stream_target_qpos is not None,
            "target_source": self._stream_target_source,
            "target_timestamp": self._stream_target_timestamp,
            "target_timestamp_monotonic": self._stream_target_monotonic,
            "target_sequence": int(self._stream_target_sequence),
            "execution_epoch_binding_enabled": self._execution_epoch_guard is not None,
            "hardware_idle_worker_preflight_binding_id": (
                self._hardware_idle_worker_preflight_binding_id
            ),
            "execution_work_item_id": (
                None
                if self._stream_execution_work_item is None
                else self._stream_execution_work_item.work_item_id
            ),
            "execution_epoch_rejected_count": int(
                self._execution_epoch_rejected_count
            ),
            "target_age_sec": target_age_sec,
            "target_max_age_sec": target_max_age_sec,
            "target_stale": target_stale,
            "continuous_hold": quick_stop_active,
            "quick_stop_active": quick_stop_active,
            "quick_stop_reason": self._stream_quick_stop_reason,
            "position_command_planner": self._stream_planner_mode,
            "trajectory_policy_mode": self._stream_planner_policy_mode,
            "trajectory_policy_speed_scale": float(
                self._stream_planner_policy_speed_scale
            ),
            "trajectory_policy_reason": self._stream_planner_policy_reason,
            "trajectory_planner": (
                self._stream_planner.snapshot()
                if isinstance(
                    self._stream_planner,
                    AccelerationLimitedJointTrajectoryPlanner,
                )
                else None
            ),
            "last_trajectory_debug": dict(self._stream_last_planner_debug),
            "command_control_dt_sec": float(self._base_control_dt),
            "segment_duration_sec": float(self._stream_segment_duration_sec),
            "tick_sequence": int(self._stream_tick_sequence),
            "scheduled_tick_count": int(self._stream_scheduled_tick_count),
            "executed_tick_count": int(self._stream_executed_tick_count),
            "skipped_tick_count": int(self._stream_skipped_tick_count),
            "send_count": int(self._stream_send_count),
            "attempted_send_count": int(self._stream_send_count),
            "accepted_send_count": int(self._stream_accepted_send_count),
            "failed_send_count": int(self._stream_failed_send_count),
            "deadline_miss_count": int(self._stream_deadline_miss_count),
            "executed_deadline_miss_count": int(
                self._stream_executed_deadline_miss_count
            ),
            "deadline_miss_definition": (
                "skipped scheduled ticks plus executed ticks finishing after their period"
            ),
            "low_rate_count": int(self._stream_low_rate_count),
            "last_send_dt_sec": self._stream_last_send_dt_sec,
            "current_inter_send_gap_sec": current_inter_send_gap_sec,
            "max_inter_send_gap_sec": max_inter_send_gap_sec,
            "actual_hz": self._stream_actual_hz,
            "last_call_ms": self._stream_last_call_ms,
            "last_deadline_lag_ms": self._stream_last_deadline_lag_ms,
            "timing_percentile_window_max_samples": (
                _BACKGROUND_AUDIT_STATS_MAX_SAMPLES
            ),
            "last_sent_step_deg": self._stream_last_sent_step_deg,
            "stale_count": int(self._stream_stale_count),
            "last_stale_age_sec": self._stream_last_stale_age_sec,
            "fault": self._stream_fault,
            "fault_timestamp": self._stream_fault_timestamp,
            "fault_timestamp_monotonic": self._stream_fault_monotonic,
            "fault_count": int(self._stream_fault_count),
            "fault_stop_attempted": bool(self._stream_fault_stop_attempted),
            "fault_stop_success": bool(self._stream_fault_stop_success),
            "fault_stop_attempts": [
                dict(attempt) for attempt in self._stream_fault_stop_attempts
            ],
            "fault_stop_timestamp": self._stream_fault_stop_timestamp,
            "shutdown_requested": bool(self._stream_shutdown_requested),
            "shutdown_complete": bool(self._stream_shutdown_complete),
            "join_timed_out": bool(self._stream_join_timed_out),
            "audit_sink_attached": self._position_audit_sink is not None,
            "audit_sink_error_count": int(self._position_audit_error_count),
            "last_audit_sink_error": self._last_position_audit_error,
            "actuator_command_trace_sink_attached": (
                self._actuator_command_trace_sink is not None
            ),
            "actuator_command_trace_error_count": int(
                self._actuator_command_trace_error_count
            ),
            "last_actuator_command_trace_error": (
                self._last_actuator_command_trace_error
            ),
            "singularity_transition_trace_sink_attached": (
                self._singularity_transition_trace_sink is not None
            ),
            "singularity_transition_trace_error_count": int(
                self._singularity_transition_trace_error_count
            ),
            "last_singularity_transition_trace_error": (
                self._last_singularity_transition_trace_error
            ),
            "healthy": not health_reasons,
            "health_reasons": health_reasons,
            "warning": None if self._stream_warning is None else dict(self._stream_warning),
        }

    def _background_stream_timing_percentiles_snapshot(
        self,
    ) -> dict[str, dict[str, int | float | None]]:
        now = time.monotonic()
        with self._stream_timing_snapshot_lock:
            if (
                self._stream_timing_snapshot_cache
                and now - self._stream_timing_snapshot_cache_monotonic
                < _BACKGROUND_AUDIT_SNAPSHOT_CACHE_SEC
            ):
                return {
                    name: dict(snapshot)
                    for name, snapshot in self._stream_timing_snapshot_cache.items()
                }
        # Snapshot outside ``_state_lock``.  PercentileStats itself also sorts
        # outside its producer lock, keeping health polling observational.
        snapshot = {
            "wakeup_lag_ms": self._stream_wakeup_lag_ms_stats.snapshot().as_dict(),
            "raw_wakeup_lag_ms": (
                self._stream_raw_wakeup_lag_ms_stats.snapshot().as_dict()
            ),
            "wakeup_gap_ms": self._stream_wakeup_gap_ms_stats.snapshot().as_dict(),
            "send_gap_ms": self._stream_send_gap_ms_stats.snapshot().as_dict(),
            "sdk_call_ms": self._stream_sdk_call_ms_stats.snapshot().as_dict(),
        }
        with self._stream_timing_snapshot_lock:
            self._stream_timing_snapshot_cache = {
                name: dict(values) for name, values in snapshot.items()
            }
            self._stream_timing_snapshot_cache_monotonic = now
        return snapshot

    def _expire_stale_stream_target(self, *, now_monotonic: float | None = None) -> bool:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        with self._state_lock:
            if self._stream_target_qpos is None or self._stream_target_monotonic is None:
                return False
            if self._stream_quick_stop_active:
                return False
            age_sec, stale = reference_age_and_stale(
                published_monotonic_time=float(self._stream_target_monotonic),
                sample_monotonic_time=now,
                stale_after_sec=float(self.config.background_streaming_target_max_age_sec),
            )
            if not stale:
                return False
            self._stream_stale_count += 1
            self._stream_last_stale_age_sec = float(age_sec)
            if isinstance(
                self._stream_planner,
                AccelerationLimitedJointTrajectoryPlanner,
            ):
                self._activate_stream_quick_stop_locked(
                    reason=f"target_stale:{age_sec:.6f}s",
                    monotonic_time=now,
                    source="background_target_stale",
                )
            else:
                anchor = self._command_anchor_qpos_locked()
                self._clear_stream_target_locked()
                if isinstance(
                    self._stream_planner,
                    SampleAlignedJerkLimitedJointTrajectoryPlanner,
                ):
                    if anchor is None:
                        anchor = np.zeros(6, dtype=float)
                    # No qcmd is emitted while a target is stale.  Reconcile
                    # the planner with that command-space hold so the next
                    # source cannot resume from nonzero qd/qdd that were never
                    # physically dispatched.
                    self._reset_stream_planner_locked(
                        anchor,
                        monotonic_time=now,
                    )
            return True

    def _record_background_stream_send(
        self,
        *,
        qpos_rad: np.ndarray,
        sent_at_perf: float,
        call_ms: float,
        deadline_lag_ms: float,
        accepted: bool = True,
        account_deadline_miss_from_lag: bool = True,
    ) -> float | None:
        qpos = np.asarray(qpos_rad, dtype=float).reshape(6)
        with self._state_lock:
            previous_send_time = self._stream_last_send_perf_time
            previous_qpos = None if self._stream_last_sent_qpos is None else self._stream_last_sent_qpos.copy()
            send_dt_sec = (
                None
                if previous_send_time is None
                else max(0.0, float(sent_at_perf) - float(previous_send_time))
            )
            actual_hz = None if not send_dt_sec else 1.0 / send_dt_sec
            if send_dt_sec is not None:
                self._stream_max_inter_send_gap_sec = max(
                    float(self._stream_max_inter_send_gap_sec or 0.0),
                    float(send_dt_sec),
                )
                self._stream_send_gap_ms_stats.record(float(send_dt_sec) * 1000.0)
            self._stream_sdk_call_ms_stats.record(float(call_ms))
            target_hz = float(self.config.background_streaming_hz)
            warning_ratio = float(self.config.background_streaming_warning_ratio)
            warning_threshold_hz = target_hz * warning_ratio
            warning = None
            if actual_hz is not None and actual_hz < warning_threshold_hz:
                self._stream_low_rate_count += 1
                warning = {
                    "active": True,
                    "message": (
                        f"WARNING: {self.name} RM65 background CANFD stream actual send rate "
                        f"{actual_hz:.1f} Hz below target {target_hz:.1f} Hz "
                        f"(threshold {warning_threshold_hz:.1f} Hz)."
                    ),
                    "target_hz": target_hz,
                    "actual_hz": float(actual_hz),
                    "threshold_hz": float(warning_threshold_hz),
                    "last_send_dt_sec": float(send_dt_sec),
                    "low_rate_count": int(self._stream_low_rate_count),
                }
                now_for_print = time.monotonic()
                if (
                    now_for_print - self._stream_last_warning_print_time
                    >= _BACKGROUND_STREAM_WARNING_PRINT_INTERVAL_SEC
                ):
                    _console_log_event(
                        "CANFD",
                        f"WARNING: {self.name} stream low rate "
                        f"actual={actual_hz:.1f}Hz target={target_hz:.1f}Hz "
                        f"threshold={warning_threshold_hz:.1f}Hz; "
                        f"suppressed repeats for "
                        f"{_BACKGROUND_STREAM_WARNING_PRINT_INTERVAL_SEC:.0f}s",
                    )
                    self._stream_last_warning_print_time = now_for_print
            else:
                warning = {
                    "active": False,
                    "target_hz": target_hz,
                    "actual_hz": None if actual_hz is None else float(actual_hz),
                    "threshold_hz": float(warning_threshold_hz),
                    "last_send_dt_sec": None if send_dt_sec is None else float(send_dt_sec),
                    "low_rate_count": int(self._stream_low_rate_count),
                }

            step_deg = None
            if accepted and previous_qpos is not None:
                step_deg = float(np.max(np.abs(qpos - previous_qpos)) * 180.0 / np.pi)
            if accepted:
                self._stream_last_sent_qpos = qpos.copy()
            self._stream_last_send_perf_time = float(sent_at_perf)
            self._stream_last_send_dt_sec = send_dt_sec
            self._stream_actual_hz = None if actual_hz is None else float(actual_hz)
            self._stream_last_call_ms = float(call_ms)
            self._stream_last_deadline_lag_ms = float(deadline_lag_ms)
            if accepted:
                self._stream_last_sent_step_deg = step_deg
            self._stream_warning = warning
            self._stream_send_count += 1
            if accepted:
                self._stream_accepted_send_count += 1
            else:
                self._stream_failed_send_count += 1
            if (
                account_deadline_miss_from_lag
                and deadline_lag_ms > (1.0 / max(target_hz, 1e-9)) * 1000.0
            ):
                self._stream_deadline_miss_count += 1
            return send_dt_sec

    def _record_background_stream_tick(
        self,
        *,
        wakeup_lag_sec: float,
        wakeup_gap_sec: float | None,
        skipped_tick_count: int,
        deadline_missed: bool,
        raw_wakeup_lag_sec: float | None = None,
    ) -> int:
        skipped = max(0, int(skipped_tick_count))
        with self._state_lock:
            self._stream_tick_sequence += 1
            self._stream_scheduled_tick_count += 1 + skipped
            self._stream_executed_tick_count += 1
            self._stream_skipped_tick_count += skipped
            # Every skipped scheduled tick missed its deadline.  The executed
            # tick contributes one more only when its SDK/no-op work finished
            # after the end of its own period.
            self._stream_deadline_miss_count += skipped + int(bool(deadline_missed))
            self._stream_executed_deadline_miss_count += int(bool(deadline_missed))
            self._stream_wakeup_lag_ms_stats.record(
                max(0.0, float(wakeup_lag_sec)) * 1000.0
            )
            self._stream_raw_wakeup_lag_ms_stats.record(
                max(
                    0.0,
                    float(
                        wakeup_lag_sec
                        if raw_wakeup_lag_sec is None
                        else raw_wakeup_lag_sec
                    ),
                )
                * 1000.0
            )
            if wakeup_gap_sec is not None:
                self._stream_wakeup_gap_ms_stats.record(
                    max(0.0, float(wakeup_gap_sec)) * 1000.0
                )
            return int(self._stream_tick_sequence)

    def _background_stream_loop(self) -> None:
        started_monotonic = time.monotonic()
        worker_exception: Exception | None = None
        unexpected_exit = False
        try:
            self._background_stream_loop_impl()
        except Exception as exc:
            worker_exception = exc
            self._latch_background_stream_fault(exc)
        finally:
            if worker_exception is None:
                with self._state_lock:
                    fault_latched = self._stream_fault is not None
                    shutdown_requested = bool(self._stream_shutdown_requested)
                stop_requested = self._stream_stop_event.is_set()
                if not fault_latched and not shutdown_requested and not stop_requested:
                    unexpected_exit = True
                    self._latch_background_stream_fault(
                        "background stream worker exited without a stop request"
                    )
            with self._state_lock:
                fault_requires_stop = bool(
                    self._stream_fault is not None
                    and not self._stream_shutdown_requested
                )
            if fault_requires_stop:
                self._attempt_background_fault_stop()
            finished_monotonic = time.monotonic()
            with self._state_lock:
                self._stream_shutdown_complete = True
                fault_message = self._stream_fault
                fault_count = int(self._stream_fault_count)
                shutdown_requested = bool(self._stream_shutdown_requested)
                tick_sequence = int(self._stream_tick_sequence)
                scheduled_tick_count = int(self._stream_scheduled_tick_count)
                executed_tick_count = int(self._stream_executed_tick_count)
                skipped_tick_count = int(self._stream_skipped_tick_count)
                send_count = int(self._stream_send_count)
                accepted_send_count = int(self._stream_accepted_send_count)
                failed_send_count = int(self._stream_failed_send_count)
                deadline_miss_count = int(self._stream_deadline_miss_count)
                fault_stop_attempted = bool(self._stream_fault_stop_attempted)
                fault_stop_success = bool(self._stream_fault_stop_success)
                fault_stop_attempts = [
                    dict(attempt) for attempt in self._stream_fault_stop_attempts
                ]
            stop_requested = self._stream_stop_event.is_set()
            if fault_message is not None:
                terminal_status = "fault"
            elif shutdown_requested:
                terminal_status = "shutdown_requested"
            else:
                terminal_status = "stop_requested"
            self._emit_position_audit_event(
                PositionTimingAuditEvent(
                    stage=_BACKGROUND_WORKER_TERMINAL_STAGE,
                    elapsed_ms=max(0.0, finished_monotonic - started_monotonic)
                    * 1000.0,
                    timestamp_wall=time.time(),
                    timestamp_monotonic=finished_monotonic,
                    side=str(self.config.side),
                    tick_sequence=tick_sequence,
                    metadata={
                        "terminal": True,
                        "terminal_status": terminal_status,
                        "fault_latched": fault_message is not None,
                        "fault_message": fault_message,
                        "fault_count": fault_count,
                        "worker_exception_type": (
                            None
                            if worker_exception is None
                            else type(worker_exception).__name__
                        ),
                        "worker_exception_message": (
                            None if worker_exception is None else str(worker_exception)
                        ),
                        "unexpected_exit": unexpected_exit,
                        "shutdown_requested": shutdown_requested,
                        "stop_requested": stop_requested,
                        "shutdown_complete": True,
                        "worker_thread_name": threading.current_thread().name,
                        "configured_hz": float(self.config.background_streaming_hz),
                        "scheduled_tick_count": scheduled_tick_count,
                        "executed_tick_count": executed_tick_count,
                        "skipped_tick_count": skipped_tick_count,
                        "send_count": send_count,
                        "accepted_send_count": accepted_send_count,
                        "failed_send_count": failed_send_count,
                        "deadline_miss_count": deadline_miss_count,
                        "fault_stop_attempted": fault_stop_attempted,
                        "fault_stop_success": fault_stop_success,
                        "fault_stop_attempts": fault_stop_attempts,
                    },
                )
            )

    def _background_stream_loop_impl(self) -> None:
        period_sec = 1.0 / float(self.config.background_streaming_hz)
        next_scheduled = time.monotonic()
        previous_wakeup: float | None = None
        while not self._stream_stop_event.is_set():
            now_monotonic = time.monotonic()
            wait_sec = next_scheduled - now_monotonic
            if wait_sec > 0.0:
                self._stream_stop_event.wait(wait_sec)
                continue
            wakeup_monotonic = time.monotonic()
            wakeup_wall_time = time.time()
            tick_timing = resolve_periodic_tick(
                scheduled_monotonic_time=next_scheduled,
                wakeup_monotonic_time=wakeup_monotonic,
                period_sec=period_sec,
            )
            next_scheduled = tick_timing.deadline_monotonic_time
            wakeup_gap_sec = (
                None
                if previous_wakeup is None
                else max(0.0, wakeup_monotonic - previous_wakeup)
            )
            previous_wakeup = wakeup_monotonic
            stale_target_expired = self._expire_stale_stream_target(
                now_monotonic=wakeup_monotonic
            )
            trajectory_proposal: JointTrajectoryProposal | None = None
            with self._state_lock:
                # ``wakeup_monotonic`` is captured before acquiring the state
                # lock and is intentionally retained for scheduler auditing.
                # A high-rate publisher can create a newer interpolation segment
                # while this worker is waiting for the lock, so that old
                # wakeup timestamp is not a valid planner sample time.  Sample
                # again under the same lock that protects target publication.
                planner_sample_monotonic = time.monotonic()
                planner_lock_delay_sec = max(
                    0.0,
                    planner_sample_monotonic - wakeup_monotonic,
                )
                target_qpos = (
                    None if self._stream_target_qpos is None else self._stream_target_qpos.copy()
                )
                target_sequence = (
                    None if target_qpos is None else int(self._stream_target_sequence)
                )
                target_source = self._stream_target_source or None
                target_timestamp_wall = self._stream_target_timestamp
                target_timestamp_monotonic = self._stream_target_monotonic
                segment_start_qpos = (
                    None
                    if self._stream_segment_start_qpos is None
                    else self._stream_segment_start_qpos.copy()
                )
                segment_start_time = self._stream_segment_start_time
                segment_duration_sec = float(self._stream_segment_duration_sec)
                reference_epoch = self._stream_reference_epoch
                reference_sequence = self._stream_reference_sequence
                execution_work_item = self._stream_execution_work_item
                anchor_qpos = self._command_anchor_qpos_locked()
                if target_qpos is None:
                    planned_tick = None
                elif isinstance(self._stream_planner, JointCommandInterpolator):
                    planned_tick = self._stream_planner.step(
                        monotonic_time=planner_sample_monotonic,
                    )
                else:
                    trajectory_proposal = self._stream_planner.propose(
                        monotonic_time=planner_sample_monotonic,
                    )
                    planned_tick = trajectory_proposal
            target_age_sec = (
                None
                if target_timestamp_monotonic is None
                else max(
                    0.0,
                    planner_sample_monotonic - float(target_timestamp_monotonic),
                )
            )

            next_qpos: np.ndarray | None = None
            planned_qpos_before_follow_smoothing: np.ndarray | None = None
            follow_smoothing_applied = False
            interpolation_alpha: float | None = None
            sdk_attempted = False
            sdk_attempt_sequence: int | None = None
            sdk_attempt_wall_time: float | None = None
            sdk_attempt_monotonic_time: float | None = None
            sdk_result_monotonic_time: float | None = None
            sdk_send_gap_sec: float | None = None
            sdk_call_ms: float | None = None
            send_code: int | None = None
            send_text: str | None = None
            accepted: bool | None = None
            accepted_sequence: int | None = None
            transport_block_reason: str | None = None
            sdk_exception: Exception | None = None
            break_after_event = False

            if target_qpos is not None:
                if segment_start_qpos is None:
                    segment_start_qpos = (
                        np.zeros(6, dtype=float) if anchor_qpos is None else anchor_qpos.copy()
                    )
                if isinstance(planned_tick, JointTrajectoryProposal):
                    if planned_tick.target_sequence != target_sequence:
                        raise RuntimeError(
                            f"{self.name} online trajectory target does not match active transport target"
                        )
                    next_qpos = np.asarray(planned_tick.q_command, dtype=float).copy()
                    interpolation_alpha = None
                    with self._state_lock:
                        self._stream_last_planner_debug = planned_tick.debug_metadata()
                else:
                    if planned_tick is None or planned_tick.reference_sequence != target_sequence:
                        raise RuntimeError(
                            f"{self.name} shared qcmd planner target does not match active transport target"
                        )
                    next_qpos = np.asarray(
                        planned_tick.q_cmd_by_side[str(self.config.side)],
                        dtype=float,
                    ).copy()
                    interpolation_alpha = float(planned_tick.interpolation_alpha)
                planned_qpos_before_follow_smoothing = next_qpos.copy()
                follow_alpha = float(self.config.arm_follow_qpos_alpha)
                with self._state_lock:
                    quick_stop_active = bool(self._stream_quick_stop_active)
                next_qpos, follow_smoothing_applied = _final_follow_qpos_target(
                    next_qpos,
                    anchor_qpos,
                    alpha=follow_alpha,
                    bypass=quick_stop_active,
                )
                next_qpos = self._validate_final_joint_target(next_qpos, label="q_cmd_150")
                if self.arm is not None:
                    with self._send_lock:
                        with self._state_lock:
                            target_is_current = bool(
                                self._stream_target_qpos is not None
                                and self._stream_target_sequence == target_sequence
                            )
                            if self._closed:
                                transport_block_reason = "backend_closed"
                                break_after_event = True
                            elif self._emergency_stopped:
                                transport_block_reason = "emergency_stopped"
                                break_after_event = True
                            elif self._stream_stop_event.is_set():
                                transport_block_reason = "stream_stop_requested"
                                break_after_event = True
                            elif not target_is_current:
                                transport_block_reason = "target_superseded_before_send"
                            else:
                                sdk_attempt_wall_time = time.time()
                                sdk_attempt_monotonic_time = time.monotonic()
                                guard = self._execution_epoch_guard
                                if guard is not None:
                                    if execution_work_item is None:
                                        transport_block_reason = (
                                            "execution_work_item_missing"
                                        )
                                        self._execution_epoch_rejected_count += 1
                                    else:
                                        epoch_check = guard.check_before_backend_call(
                                            execution_work_item
                                        )
                                        if not epoch_check.accepted:
                                            transport_block_reason = epoch_check.status
                                            self._execution_epoch_rejected_count += 1
                                            if target_is_current:
                                                self._clear_stream_target_locked()
                                if transport_block_reason is None:
                                    sdk_attempt_sequence = self._record_command_attempt_locked(
                                        next_qpos,
                                        timestamp=sdk_attempt_wall_time,
                                        monotonic_time=sdk_attempt_monotonic_time,
                                    )
                                    self._execution_inflight_work_item = (
                                        execution_work_item
                                    )
                                    self._last_old_epoch_invocation_started_at = (
                                        sdk_attempt_monotonic_time
                                    )
                                    self._stream_sdk_inflight_target_sequence = (
                                        None
                                        if target_sequence is None
                                        else int(target_sequence)
                                    )
                                    sdk_attempted = True
                        if sdk_attempted:
                            call_started = time.monotonic()
                            raw_send_result: object = None
                            try:
                                raw_send_result = self._send_joint_target(
                                    next_qpos,
                                    command_source=target_source or "background_joint_target",
                                    call_path="background_worker",
                                    quick_stop_active=quick_stop_active,
                                    target_sequence=target_sequence,
                                    execution_work_item=execution_work_item,
                                )
                            except Exception as exc:  # audited below, then fault-latched by caller.
                                sdk_exception = exc
                            finally:
                                call_finished = time.monotonic()
                                with self._state_lock:
                                    self._last_old_epoch_invocation_completed_at = (
                                        call_finished
                                    )
                                    self._last_completed_execution_work_item = (
                                        execution_work_item
                                    )
                                    self._execution_inflight_work_item = None
                            sdk_result_monotonic_time = call_finished
                            sdk_call_ms = (call_finished - call_started) * 1000.0
                            if sdk_exception is None:
                                send_code, send_text = _parse_rm65_result(raw_send_result)
                            else:
                                send_code = None
                                send_text = (
                                    f"EXCEPTION: {type(sdk_exception).__name__}: {sdk_exception}"
                                )
                            accepted = bool(sdk_exception is None and send_code == 0)
                            sdk_send_gap_sec = self._record_background_stream_send(
                                qpos_rad=next_qpos,
                                sent_at_perf=call_finished,
                                call_ms=sdk_call_ms,
                                deadline_lag_ms=tick_timing.wakeup_lag_sec * 1000.0,
                                accepted=accepted,
                                account_deadline_miss_from_lag=False,
                            )
                            with self._state_lock:
                                self._last_send_result_code = send_code
                                self._last_send_result_text = send_text
                                if accepted:
                                    if trajectory_proposal is not None:
                                        target_superseded_during_sdk = bool(
                                            target_sequence is not None
                                            and self._stream_target_sequence
                                            != target_sequence
                                        )
                                        if target_superseded_during_sdk:
                                            self._rebase_acceleration_planner_after_inflight_target_preemption_locked(
                                                accepted_qpos=next_qpos,
                                                completed_monotonic_time=call_finished,
                                                completed_target_sequence=int(
                                                    target_sequence
                                                ),
                                            )
                                            self._stream_deferred_target = None
                                        else:
                                            self._stream_planner.commit(
                                                trajectory_proposal,
                                                accepted_monotonic_time=call_finished,
                                                accepted_q_command=next_qpos,
                                                allow_bounded_accepted_adjustment=bool(
                                                    follow_smoothing_applied
                                                ),
                                            )
                                            self._activate_deferred_acceleration_target_locked(
                                                accepted_qpos=next_qpos,
                                                activated_monotonic_time=(
                                                    call_finished
                                                ),
                                                completed_target_sequence=int(
                                                    target_sequence
                                                ),
                                            )
                                    accepted_sequence = self._record_command_accepted_locked(
                                        next_qpos,
                                        timestamp=time.time(),
                                        monotonic_time=call_finished,
                                    )
                                    self._last_send_time = wakeup_wall_time
                                elif self._stream_deferred_target is not None:
                                    # A failed SDK call provides no accepted
                                    # qcmd anchor on which a deferred reference
                                    # can be safely activated. The worker fault
                                    # path will hold/stop before accepting more
                                    # motion.
                                    self._stream_deferred_target = None
                                self._stream_sdk_inflight_target_sequence = None
                else:
                    transport_block_reason = "arm_not_connected"

            if trajectory_proposal is not None and not bool(accepted):
                self._stream_planner.reject(trajectory_proposal)

            finished_monotonic = time.monotonic()
            deadline_missed = tick_timing.deadline_missed(finished_monotonic)
            tick_sequence = self._record_background_stream_tick(
                wakeup_lag_sec=tick_timing.wakeup_lag_sec,
                raw_wakeup_lag_sec=tick_timing.raw_wakeup_lag_sec,
                wakeup_gap_sec=wakeup_gap_sec,
                skipped_tick_count=tick_timing.skipped_tick_count,
                deadline_missed=deadline_missed,
            )
            if (
                sdk_attempted
                and next_qpos is not None
                and self._actuator_command_trace_sink is not None
            ):
                # Post-SDK fixed-field capture only: no FK, JSON, queueing, or
                # disk I/O is allowed in this callback.
                self._emit_actuator_command_trace(
                    str(self.config.side),
                    tick_sequence,
                    target_sequence,
                    reference_epoch,
                    reference_sequence,
                    tick_timing.scheduled_monotonic_time,
                    wakeup_monotonic,
                    sdk_attempt_sequence,
                    sdk_attempt_wall_time,
                    sdk_attempt_monotonic_time,
                    sdk_result_monotonic_time,
                    next_qpos,
                    send_code,
                    accepted,
                    accepted_sequence,
                )
            # Constructing a full command-audit event is deliberately lazy.  In
            # particular, accepted-qcmd FK and metadata freezing are sizeable
            # Python/GIL work at two 120 Hz command workers.  The production
            # collection profile has no position-audit sink, so doing that work
            # before _emit_position_audit_event() discovers ``sink is None``
            # only steals scheduler budget without producing an observation.
            # Sink attachment happens before a diagnostic session and detach is
            # best-effort during shutdown; a stale positive read is harmless
            # because the emitter rechecks under _state_lock.
            if self._position_audit_sink is not None:
                self._emit_position_audit_event(
                    PositionCommandAuditEvent(
                        side=str(self.config.side),
                        tick_sequence=tick_sequence,
                        timestamp_wall=time.time(),
                        timestamp_monotonic=finished_monotonic,
                        scheduled_monotonic_time=tick_timing.scheduled_monotonic_time,
                        wakeup_monotonic_time=wakeup_monotonic,
                        deadline_monotonic_time=tick_timing.deadline_monotonic_time,
                        wakeup_lag_sec=tick_timing.wakeup_lag_sec,
                        raw_wakeup_lag_sec=tick_timing.raw_wakeup_lag_sec,
                        skipped_tick_count=tick_timing.skipped_tick_count,
                        deadline_missed=deadline_missed,
                        planner_sample_monotonic_time=planner_sample_monotonic,
                        planner_lock_delay_sec=planner_lock_delay_sec,
                        wakeup_gap_sec=wakeup_gap_sec,
                        target_active=target_qpos is not None,
                        target_sequence=target_sequence,
                        target_source=target_source,
                        target_timestamp_wall=target_timestamp_wall,
                        target_timestamp_monotonic=target_timestamp_monotonic,
                        target_age_sec=target_age_sec,
                        reference_epoch=reference_epoch,
                        reference_sequence=reference_sequence,
                        segment_start_qpos=segment_start_qpos,
                        segment_target_qpos=target_qpos,
                        segment_start_monotonic_time=segment_start_time,
                        segment_duration_sec=(
                            segment_duration_sec
                            if target_qpos is not None
                            and trajectory_proposal is None
                            else None
                        ),
                        interpolation_alpha=interpolation_alpha,
                        q_cmd=next_qpos,
                        sdk_attempted=sdk_attempted,
                        sdk_attempt_sequence=sdk_attempt_sequence,
                        sdk_attempt_wall_time=sdk_attempt_wall_time,
                        sdk_attempt_monotonic_time=sdk_attempt_monotonic_time,
                        sdk_result_monotonic_time=sdk_result_monotonic_time,
                        sdk_send_gap_sec=sdk_send_gap_sec,
                        sdk_call_ms=sdk_call_ms,
                        sdk_result_code=send_code,
                        sdk_result_text=send_text,
                        sdk_accepted=accepted,
                        sdk_accepted_sequence=accepted_sequence,
                        stale_target_expired=stale_target_expired,
                        metadata={
                            "effector_name": self.name,
                            "segment_planner": (
                                "robot_motion.position_control."
                                + (
                                    "SampleAlignedJerkLimitedJointTrajectoryPlanner"
                                    if isinstance(
                                        self._stream_planner,
                                        SampleAlignedJerkLimitedJointTrajectoryPlanner,
                                    )
                                    else (
                                        "AccelerationLimitedJointTrajectoryPlanner"
                                        if trajectory_proposal is not None
                                        else "JointCommandInterpolator"
                                    )
                                )
                            ),
                            "position_command_planner": self._stream_planner_mode,
                            "scheduler_original_scheduled_monotonic_time": (
                                tick_timing.original_scheduled_monotonic_time
                            ),
                            "scheduled_period_sec": period_sec,
                            "transport_block_reason": transport_block_reason,
                            "arm_follow_qpos_alpha": float(
                                self.config.arm_follow_qpos_alpha
                            ),
                            "arm_follow_qpos_smoothing_applied": bool(
                                follow_smoothing_applied
                            ),
                            "arm_follow_qpos_pre_smoothing": (
                                None
                                if planned_qpos_before_follow_smoothing is None
                                else planned_qpos_before_follow_smoothing.tolist()
                            ),
                            **(
                                {}
                                if not accepted
                                else self._accepted_qcmd_fk_audit_metadata(next_qpos)
                            ),
                            **(
                                {}
                                if trajectory_proposal is None
                                else trajectory_proposal.debug_metadata()
                            ),
                        },
                    )
                )

            if sdk_exception is not None:
                raise sdk_exception
            if sdk_attempted and accepted is False:
                self._latch_background_stream_fault(
                    f"Movej_CANFD returned non-zero/unknown result: {send_text}"
                )
                break
            if break_after_event:
                break

    def _apply_jump_gate(
        self,
        *,
        solver,
        target_pose: np.ndarray,
        current_qpos: np.ndarray,
        candidate_qpos: np.ndarray,
    ) -> np.ndarray:
        jump_gate = self._ik_tuning.jump_gate
        target_pose = np.asarray(target_pose, dtype=float)
        candidate_qpos = np.asarray(candidate_qpos, dtype=float).reshape(-1)
        if not jump_gate.enabled:
            self._last_target_pose = target_pose.copy()
            self._last_ik_metadata = {
                "ik_objective_mode": self._ik_tuning.ik_objective_mode,
                "ik_branch_selection": getattr(solver, "last_branch_selection", "unknown"),
                "ik_jump_gate_status": "disabled",
                "configured_tcp_enabled": self._configured_tcp_enabled,
                "ik_framework": self.config.ik_framework,
            }
            return candidate_qpos

        relaxed = self._jump_gate_active
        per_joint_limit = (
            jump_gate.relaxed_per_joint_jump_deg
            if relaxed
            else jump_gate.per_joint_jump_deg
        )
        weighted_limit = (
            jump_gate.relaxed_weighted_jump_threshold_deg
            if relaxed
            else jump_gate.weighted_jump_threshold_deg
        )
        pose_translation_m = float("inf")
        pose_rotation_deg = float("inf")
        if self._last_target_pose is not None:
            pose_translation_m, pose_rotation_deg = pose_delta_metrics(
                self._last_target_pose,
                target_pose,
            )
        small_pose = bool(
            self._last_target_pose is not None
            and pose_translation_m <= float(jump_gate.small_pose_translation_m)
            and pose_rotation_deg <= float(jump_gate.small_pose_rotation_deg)
        )
        moderate_pose_translation_m = max(
            float(jump_gate.small_pose_translation_m),
            float(getattr(jump_gate, "moderate_pose_translation_m", jump_gate.release_translation_m)),
        )
        moderate_pose_rotation_deg = max(
            float(jump_gate.small_pose_rotation_deg),
            float(getattr(jump_gate, "moderate_pose_rotation_deg", jump_gate.release_rotation_deg)),
        )
        moderate_pose = bool(
            self._last_target_pose is not None
            and pose_translation_m <= moderate_pose_translation_m
            and pose_rotation_deg <= moderate_pose_rotation_deg
        )
        jump_exceeded, per_joint_delta_deg, weighted_jump_deg = joint_jump_exceeds(
            reference_q=current_qpos,
            candidate_q=candidate_qpos,
            joint_distance_weights=self._joint_distance_weights,
            per_joint_jump_deg=per_joint_limit,
            weighted_jump_threshold_deg=weighted_limit,
        )
        moderate_jump_scale = max(
            1.0,
            float(getattr(jump_gate, "moderate_pose_joint_jump_scale", 1.25)),
        )
        moderate_per_joint_limit = np.asarray(per_joint_limit, dtype=float).reshape(-1) * moderate_jump_scale
        moderate_weighted_limit = float(weighted_limit) * moderate_jump_scale
        moderate_jump_exceeded = bool(
            np.any(per_joint_delta_deg > moderate_per_joint_limit)
            or weighted_jump_deg > moderate_weighted_limit
        )
        moderate_pose_large_jump = bool(moderate_pose and moderate_jump_exceeded)
        hard_branch_exceeded, hard_branch_delta_deg = hard_branch_jump_exceeds(
            reference_q=current_qpos,
            candidate_q=candidate_qpos,
            joint_indices=jump_gate.hard_branch_joint_indices,
            max_delta_deg=jump_gate.hard_branch_max_delta_deg,
        )
        release_ready = False
        if self._jump_gate_active and self._jump_hold_anchor_pose is not None:
            release_translation_m, release_rotation_deg = pose_delta_metrics(
                self._jump_hold_anchor_pose,
                target_pose,
            )
            release_ready = bool(
                release_translation_m >= float(jump_gate.release_translation_m)
                or release_rotation_deg >= float(jump_gate.release_rotation_deg)
            )

        metadata = {
            "ik_objective_mode": self._ik_tuning.ik_objective_mode,
            "ik_branch_selection": getattr(solver, "last_branch_selection", "unknown"),
            "ik_branch_distance_score_deg": getattr(
                solver,
                "last_branch_distance_score_deg",
                None,
            ),
            "ik_jump_gate_status": "pass",
            "ik_jump_gate_hold_frames": self._jump_hold_frames,
            "ik_pose_delta_m": None if not np.isfinite(pose_translation_m) else pose_translation_m,
            "ik_pose_delta_deg": None if not np.isfinite(pose_rotation_deg) else pose_rotation_deg,
            "ik_pose_is_small": small_pose,
            "ik_pose_is_moderate": moderate_pose,
            "ik_joint_jump_max_deg": float(np.max(per_joint_delta_deg)),
            "ik_joint_jump_weighted_deg": float(weighted_jump_deg),
            "ik_joint_jump_threshold_deg": float(weighted_limit),
            "ik_moderate_pose_translation_m": float(moderate_pose_translation_m),
            "ik_moderate_pose_rotation_deg": float(moderate_pose_rotation_deg),
            "ik_moderate_pose_joint_jump_scale": float(moderate_jump_scale),
            "ik_moderate_pose_joint_jump_exceeded": bool(moderate_jump_exceeded),
            "ik_moderate_pose_large_jump": bool(moderate_pose_large_jump),
            "ik_moderate_pose_per_joint_threshold_deg": moderate_per_joint_limit.tolist(),
            "ik_moderate_pose_weighted_threshold_deg": float(moderate_weighted_limit),
            "ik_hard_branch_joint_indices": list(jump_gate.hard_branch_joint_indices),
            "ik_hard_branch_delta_deg": hard_branch_delta_deg.tolist(),
            "ik_hard_branch_max_delta_deg": list(jump_gate.hard_branch_max_delta_deg),
            "ik_hard_branch_exceeded": bool(hard_branch_exceeded),
            "ik_release_ready": release_ready,
            "configured_tcp_enabled": self._configured_tcp_enabled,
            "ik_framework": self.config.ik_framework,
        }

        should_hold = bool(
            (
                hard_branch_exceeded
                or (small_pose and jump_exceeded)
                or moderate_pose_large_jump
            )
            and not release_ready
        )
        if self._jump_gate_active and not should_hold and release_ready:
            metadata["ik_jump_gate_status"] = "released"
        elif self._jump_gate_active and should_hold:
            metadata["ik_jump_gate_status"] = "holding"
            if hard_branch_exceeded:
                metadata["ik_jump_gate_reason"] = "hard_branch_jump"
            elif small_pose and jump_exceeded:
                metadata["ik_jump_gate_reason"] = "small_pose_large_q_jump"
            else:
                metadata["ik_jump_gate_reason"] = "moderate_pose_large_q_jump"
        elif should_hold:
            metadata["ik_jump_gate_status"] = "hold_entered"
            if hard_branch_exceeded:
                metadata["ik_jump_gate_reason"] = "hard_branch_jump"
            elif small_pose and jump_exceeded:
                metadata["ik_jump_gate_reason"] = "small_pose_large_q_jump"
            else:
                metadata["ik_jump_gate_reason"] = "moderate_pose_large_q_jump"

        if should_hold:
            if not self._jump_gate_active:
                self._jump_gate_active = True
                self._jump_hold_qpos = current_qpos.copy()
                self._jump_hold_anchor_pose = target_pose.copy()
                self._jump_hold_frames = 0
            self._jump_hold_frames += 1
            metadata["ik_jump_gate_hold_frames"] = self._jump_hold_frames
            self._last_target_pose = target_pose.copy()
            self._last_ik_metadata = metadata
            return current_qpos.copy()

        self._jump_gate_active = False
        self._jump_hold_qpos = None
        self._jump_hold_anchor_pose = None
        self._jump_hold_frames = 0
        self._last_target_pose = target_pose.copy()
        self._last_ik_metadata = metadata
        return candidate_qpos.copy()

    def _start_force_position_control(self) -> None:
        if self.config.observation_only:
            return
        if self.arm is None:
            return
        if self.config.force_clear_on_start:
            clear_result = self._invoke_pre_sdk_control(
                seam_id="rm65.lifecycle.force_position.start",
                permit_kind="lifecycle_control",
                operation="startup",
                source="force_position_startup",
                sdk_method="Clear_Force_Data",
                attempt_sequence=1,
                backend_call=lambda: self.arm.Clear_Force_Data(True),
            )
            clear_code, clear_text = _parse_rm65_result(clear_result)
            if clear_code != 0:
                raise RuntimeError(
                    f"{self.name} failed to clear force data before force-position control: {clear_text}"
                )
        start_result = self._invoke_pre_sdk_control(
            seam_id="rm65.lifecycle.force_position.start",
            permit_kind="lifecycle_control",
            operation="startup",
            source="force_position_startup",
            sdk_method="Start_Force_Position_Move",
            attempt_sequence=2 if self.config.force_clear_on_start else 1,
            backend_call=lambda: self.arm.Start_Force_Position_Move(True),
        )
        start_code, start_text = _parse_rm65_result(start_result)
        if start_code != 0:
            raise RuntimeError(
                f"{self.name} failed to start force-position control: {start_text}"
            )
        self._force_position_active = True

    def _invoke_pre_sdk_control(
        self,
        *,
        seam_id: str,
        permit_kind: str,
        operation: str,
        source: str,
        sdk_method: str,
        backend_call: Callable[[], object],
        attempt_sequence: int = 1,
    ) -> object:
        """Invoke one safety/lifecycle SDK call through the optional gate."""

        adapter = self._pre_sdk_control_adapter
        if adapter is None:
            return backend_call()
        context = RM65ControlDispatchContext.create(
            side=str(self.config.side),
            seam_id=seam_id,
            permit_kind=permit_kind,
            operation=operation,
            source=source,
            sdk_method=sdk_method,
            attempt_sequence=attempt_sequence,
        )
        return adapter.invoke(context=context, backend_call=backend_call)

    def _send_joint_target(
        self,
        qpos_rad: np.ndarray,
        *,
        command_source: str | None = None,
        call_path: str = "direct",
        quick_stop_active: bool = False,
        target_sequence: int | None = None,
        execution_work_item: ExecutionWorkItem | None = None,
    ) -> object:
        if self.config.observation_only:
            raise RuntimeError(
                f"{self.name} is observation-only; refusing RM65 transport call."
            )
        qpos = self._validate_final_joint_target(qpos_rad, label="transport joint target")
        if self.arm is None:
            return None
        guard = self._execution_epoch_guard
        if guard is not None:
            if not isinstance(execution_work_item, ExecutionWorkItem):
                raise RuntimeError(
                    "RM65 epoch-mediated SDK call requires typed execution work"
                )
            epoch_check = guard.check_before_backend_call(execution_work_item)
            if not epoch_check.accepted:
                with self._state_lock:
                    self._execution_epoch_rejected_count += 1
                raise RuntimeError(
                    "RM65 final execution epoch check rejected SDK call: "
                    f"{epoch_check.status}"
                )
        joint_deg = qpos / np.pi * 180.0

        def backend_call() -> object:
            if self.config.force_position_control:
                return self.arm.Force_Position_Move_Joint(
                    joint_deg,
                    sensor=self.config.force_sensor,
                    mode=self.config.force_mode,
                    dir=self.config.force_dir,
                    force=self.config.force,
                    follow=self.config.force_follow,
                )
            return self.arm.Movej_CANFD(
                joint_deg,
                follow=bool(self.config.canfd_follow),
                trajectory_mode=int(self.config.canfd_trajectory_mode),
                radio=int(self.config.canfd_radio),
            )

        adapter = self._pre_sdk_dispatch_adapter
        if adapter is None:
            return backend_call()
        source = str(command_source or "").strip()
        if not source:
            raise RuntimeError(
                "RM65 pre-SDK dispatch mediation requires an explicit command source."
            )
        context = RM65JointTargetDispatchContext.create(
            side=str(self.config.side),
            qpos_rad=qpos,
            command_source=source,
            call_path=call_path,
            quick_stop_active=bool(quick_stop_active),
            lifecycle_home=_is_qpos_init_slow_singularity_source(source),
            target_sequence=target_sequence,
            execution_work_item=execution_work_item,
        )
        return adapter.invoke(context=context, backend_call=backend_call)

    def refresh_state_from_hardware(
        self,
        *,
        hardware_read_retries: int | None = None,
        hardware_read_retry_delay_sec: float | None = None,
        include_derived_pose: bool = True,
        include_timing_percentiles: bool = True,
    ) -> dict[str, object]:
        if self.arm is None:
            return self.get_state(
                include_derived_pose=bool(include_derived_pose),
                include_timing_percentiles=bool(include_timing_percentiles),
            )
        refresh_wall_started = time.perf_counter()
        refresh_cpu_started = time.thread_time()
        lock_wait_wall_sec = 0.0
        lock_wait_cpu_sec = 0.0
        sdk_call_wall_sec = 0.0
        sdk_call_cpu_sec = 0.0
        parse_validation_wall_sec = 0.0
        parse_validation_cpu_sec = 0.0
        retry_sleep_wall_sec = 0.0
        retry_count = max(
            1,
            int(
                self.config.hardware_read_retries
                if hardware_read_retries is None
                else hardware_read_retries
            )
            + 1,
        )
        retry_delay_sec = float(
            self.config.hardware_read_retry_delay_sec
            if hardware_read_retry_delay_sec is None
            else hardware_read_retry_delay_sec
        )
        raw_result = -1
        joint_now = None
        pose_now = None
        arm_err = None
        sys_err = None
        parsed_code: int | None = None
        parsed_text = ""
        qpos_from_joint_now = None
        for attempt in range(retry_count):
            lock_wait_started = time.perf_counter()
            lock_wait_cpu_started = time.thread_time()
            try:
                with self._hardware_read_lock:
                    lock_acquired = time.perf_counter()
                    lock_wait_wall_sec += max(0.0, lock_acquired - lock_wait_started)
                    lock_wait_cpu_sec += max(
                        0.0,
                        time.thread_time() - lock_wait_cpu_started,
                    )
                    sdk_call_started = time.perf_counter()
                    sdk_call_cpu_started = time.thread_time()
                    try:
                        raw_result, joint_now, pose_now, arm_err, sys_err = (
                            self.arm.Get_Current_Arm_State()
                        )
                    finally:
                        sdk_call_wall_sec += max(
                            0.0,
                            time.perf_counter() - sdk_call_started,
                        )
                        sdk_call_cpu_sec += max(
                            0.0,
                            time.thread_time() - sdk_call_cpu_started,
                        )
            except Exception as exc:
                raw_result = f"EXCEPTION: {exc}"
                joint_now = None
                pose_now = None
                arm_err = None
                sys_err = None
            parse_validation_started = time.perf_counter()
            parse_validation_cpu_started = time.thread_time()
            parsed_code, parsed_text = _parse_rm65_result(raw_result)
            qpos_from_joint_now = _joint_now_to_qpos(joint_now)
            usable_attempt = _state_result_is_usable(
                parsed_code,
                joint_now=joint_now,
                pose_now=pose_now,
            )
            parse_validation_wall_sec += max(
                0.0,
                time.perf_counter() - parse_validation_started,
            )
            parse_validation_cpu_sec += max(
                0.0,
                time.thread_time() - parse_validation_cpu_started,
            )
            if usable_attempt:
                break
            if attempt + 1 < retry_count:
                retry_sleep_started = time.perf_counter()
                time.sleep(retry_delay_sec)
                retry_sleep_wall_sec += max(
                    0.0,
                    time.perf_counter() - retry_sleep_started,
                )
        refresh_timestamp = time.time()
        refresh_monotonic = time.monotonic()
        usable_result = _state_result_is_usable(
            parsed_code,
            joint_now=joint_now,
            pose_now=pose_now,
        )
        measurement_success = bool(
            usable_result and qpos_from_joint_now is not None
        )
        refresh_phase_timing = {
            "total_wall_ms": max(
                0.0,
                (time.perf_counter() - refresh_wall_started) * 1000.0,
            ),
            "total_thread_cpu_ms": max(
                0.0,
                (time.thread_time() - refresh_cpu_started) * 1000.0,
            ),
            "hardware_read_lock_wait_wall_ms": lock_wait_wall_sec * 1000.0,
            "hardware_read_lock_wait_thread_cpu_ms": lock_wait_cpu_sec * 1000.0,
            "sdk_call_wall_ms": sdk_call_wall_sec * 1000.0,
            "sdk_call_thread_cpu_ms": sdk_call_cpu_sec * 1000.0,
            "parse_validation_wall_ms": parse_validation_wall_sec * 1000.0,
            "parse_validation_thread_cpu_ms": parse_validation_cpu_sec * 1000.0,
            "retry_sleep_wall_ms": retry_sleep_wall_sec * 1000.0,
            "attempt_count": float(attempt + 1),
        }
        with self._state_lock:
            self._hardware_read_sequence += 1
            self._last_hardware_read_time = refresh_timestamp
            self._last_hardware_read_monotonic = refresh_monotonic
            self._last_hardware_read_success = measurement_success
            self._last_hardware_read_phase_timing = dict(refresh_phase_timing)
            self._last_hardware_result_code = parsed_code
            self._last_hardware_result_text = parsed_text
            self._last_hardware_arm_err = None if arm_err is None else int(arm_err)
            self._last_hardware_sys_err = None if sys_err is None else int(sys_err)
            if usable_result and pose_now is not None:
                self._last_hardware_pose_6d = np.asarray(pose_now, dtype=float)
            if measurement_success:
                self._last_hardware_refresh_time = refresh_timestamp
                self._record_measured_qpos_locked(
                    qpos_from_joint_now,
                    timestamp=refresh_timestamp,
                    monotonic_time=refresh_monotonic,
                )
                if self.config.connect_hardware and self.arm is not None:
                    self._hardware_ready = True
                    self._hardware_ready_source = "hardware_feedback"
        return self.get_state(
            include_derived_pose=bool(include_derived_pose),
            include_timing_percentiles=bool(include_timing_percentiles),
        )

    def read_force_data(self) -> dict[str, object]:
        """Read one validated RM65 force sample from this backend connection."""

        if self.arm is None:
            raise RuntimeError(f"{self.name} RM65 force read requires a hardware connection")
        started_monotonic = time.monotonic()
        read_wall_started = time.perf_counter()
        read_cpu_started = time.thread_time()
        pacing_wait_requested_sec = 0.0
        pacing_wait_wall_sec = 0.0
        pacing_wait_cpu_sec = 0.0
        lock_wait_wall_sec = 0.0
        lock_wait_cpu_sec = 0.0
        sdk_call_wall_sec = 0.0
        sdk_call_cpu_sec = 0.0
        # The vendor SDK requires at least 50 ms between *actual* repeated
        # Get_Force_Data calls.  Pacing only the monitor loop is insufficient:
        # an intervening qpos read can shift one force call later and make the
        # next call occur less than 50 ms afterwards.  Enforce the interval at
        # the serialized SDK boundary while leaving the lock free during the
        # wait so qpos feedback can continue.
        while True:
            lock_wait_started = time.perf_counter()
            lock_wait_cpu_started = time.thread_time()
            with self._hardware_read_lock:
                lock_acquired = time.perf_counter()
                lock_wait_wall_sec += max(0.0, lock_acquired - lock_wait_started)
                lock_wait_cpu_sec += max(
                    0.0,
                    time.thread_time() - lock_wait_cpu_started,
                )
                query_now = time.monotonic()
                previous_query = self._last_force_query_monotonic
                wait_sec = (
                    0.0
                    if previous_query is None
                    else 0.05 - (query_now - previous_query)
                )
                if wait_sec <= 0.0:
                    self._last_force_query_monotonic = query_now
                    sdk_call_started = time.perf_counter()
                    sdk_call_cpu_started = time.thread_time()
                    try:
                        result = self.arm.Get_Force_Data()
                    finally:
                        sdk_call_wall_sec += max(
                            0.0,
                            time.perf_counter() - sdk_call_started,
                        )
                        sdk_call_cpu_sec += max(
                            0.0,
                            time.thread_time() - sdk_call_cpu_started,
                        )
                    break
            pacing_wait_requested_sec += max(0.0, wait_sec)
            pacing_wait_started = time.perf_counter()
            pacing_wait_cpu_started = time.thread_time()
            time.sleep(wait_sec)
            pacing_wait_wall_sec += max(
                0.0,
                time.perf_counter() - pacing_wait_started,
            )
            pacing_wait_cpu_sec += max(
                0.0,
                time.thread_time() - pacing_wait_cpu_started,
            )
        validation_wall_started = time.perf_counter()
        validation_cpu_started = time.thread_time()
        if not isinstance(result, (tuple, list)) or len(result) != 5:
            raise RuntimeError(
                f"{self.name} Get_Force_Data returned an invalid result shape"
            )
        raw_code, raw, zero, work, tool = result
        code, result_text = _parse_rm65_result(raw_code)
        if code != 0:
            raise RuntimeError(
                f"{self.name} Get_Force_Data failed: {result_text}"
            )
        sources: dict[str, np.ndarray] = {}
        for source, values in (
            ("raw", raw),
            ("zero", zero),
            ("work", work),
            ("tool", tool),
        ):
            try:
                vector = np.asarray(values, dtype=float).reshape(-1)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.name} {source} force is not numeric"
                ) from exc
            if vector.shape != (6,) or not np.all(np.isfinite(vector)):
                raise RuntimeError(
                    f"{self.name} {source} force must contain 6 finite values"
                )
            sources[source] = vector.copy()
        validation_wall_sec = max(
            0.0,
            time.perf_counter() - validation_wall_started,
        )
        validation_cpu_sec = max(
            0.0,
            time.thread_time() - validation_cpu_started,
        )
        read_phase_timing = {
            "total_wall_ms": max(
                0.0,
                (time.perf_counter() - read_wall_started) * 1000.0,
            ),
            "total_thread_cpu_ms": max(
                0.0,
                (time.thread_time() - read_cpu_started) * 1000.0,
            ),
            "pacing_wait_requested_ms": pacing_wait_requested_sec * 1000.0,
            "pacing_wait_wall_ms": pacing_wait_wall_sec * 1000.0,
            "pacing_wait_thread_cpu_ms": pacing_wait_cpu_sec * 1000.0,
            "hardware_read_lock_wait_wall_ms": lock_wait_wall_sec * 1000.0,
            "hardware_read_lock_wait_thread_cpu_ms": lock_wait_cpu_sec * 1000.0,
            "sdk_call_wall_ms": sdk_call_wall_sec * 1000.0,
            "sdk_call_thread_cpu_ms": sdk_call_cpu_sec * 1000.0,
            "validation_wall_ms": validation_wall_sec * 1000.0,
            "validation_thread_cpu_ms": validation_cpu_sec * 1000.0,
        }
        return {
            "sources": sources,
            "monotonic_time_sec": time.monotonic(),
            "wall_time_sec": time.time(),
            "read_elapsed_sec": time.monotonic() - started_monotonic,
            "read_phase_timing": read_phase_timing,
            "result_code": code,
            "result_text": result_text,
        }

    def get_state(
        self,
        *,
        include_derived_pose: bool = True,
        include_timing_percentiles: bool = True,
    ) -> dict[str, object]:
        """Return a backend snapshot.

        The compatibility default remains the complete diagnostic snapshot.
        High-rate callers may omit FK-derived poses and percentile sorting;
        neither field participates in command admission or transport safety.
        """

        with self._state_lock:
            current_qpos = None if self._current_qpos is None else self._current_qpos.copy()
            attempted_qpos = (
                None if self._last_attempted_qpos is None else self._last_attempted_qpos.copy()
            )
            accepted_qpos = (
                None if self._last_accepted_qpos is None else self._last_accepted_qpos.copy()
            )
            measured_qpos = None if self._measured_qpos is None else self._measured_qpos.copy()
            hardware_qpos = (
                None if self._last_hardware_qpos is None else self._last_hardware_qpos.copy()
            )
            last_target_qpos = (
                None if self._last_target_qpos is None else self._last_target_qpos.copy()
            )
            last_command = None if self._last_command is None else self._last_command.copy()
            hardware_pose_6d = (
                None if self._last_hardware_pose_6d is None else self._last_hardware_pose_6d.copy()
            )
            hardware_read_time = self._last_hardware_read_time
            hardware_read_monotonic = self._last_hardware_read_monotonic
            hardware_read_success = self._last_hardware_read_success
            hardware_read_phase_timing = dict(
                self._last_hardware_read_phase_timing
            )
            hardware_read_sequence = int(self._hardware_read_sequence)
            hardware_refresh_time = self._last_hardware_refresh_time
            measured_timestamp = self._measured_timestamp
            measured_monotonic = self._measured_monotonic
            measured_sequence = int(self._measured_sequence)
            local_qpos_normalization = dict(self._last_local_qpos_normalization)
            hardware_result_code = self._last_hardware_result_code
            hardware_result_text = self._last_hardware_result_text
            hardware_arm_err = self._last_hardware_arm_err
            hardware_sys_err = self._last_hardware_sys_err
            hardware_ready = bool(self._hardware_ready)
            hardware_ready_source = str(self._hardware_ready_source)
            singularity_metadata = dict(self._last_singularity_metadata)
            emergency_stopped = bool(self._emergency_stopped)
            background_streaming = self._background_stream_snapshot_locked()
        if include_timing_percentiles:
            background_streaming["timing_percentiles"] = (
                self._background_stream_timing_percentiles_snapshot()
            )
        current_tcp_pose = None
        if include_derived_pose and current_qpos is not None:
            try:
                current_tcp_pose = self.compute_tcp_pose(qpos=current_qpos)
            except Exception:
                current_tcp_pose = None
        last_target_tcp_pose = None
        if include_derived_pose and last_target_qpos is not None:
            try:
                last_target_tcp_pose = self.compute_tcp_pose(qpos=last_target_qpos)
            except Exception:
                last_target_tcp_pose = None
        tcp_pose_flange = self._active_tcp_pose_flange_m_rad()
        tcp_delta = tcp_endpoint_delta(tcp_pose_flange)
        ik_metadata = dict(self._last_ik_metadata)
        ik_metadata.update(singularity_metadata)
        return {
            "current_qpos": current_qpos,
            "current_tcp_pose": current_tcp_pose,
            "current_qpos_source": (
                "hardware_feedback"
                if self.config.observation_only
                else "last_accepted_command_or_local_anchor"
            ),
            "last_attempted_qpos": attempted_qpos,
            "last_attempted_timestamp": self._last_attempted_timestamp,
            "last_attempted_timestamp_monotonic": self._last_attempted_monotonic,
            "last_attempted_sequence": int(self._last_attempted_sequence),
            "last_accepted_qpos": accepted_qpos,
            "last_accepted_timestamp": self._last_accepted_timestamp,
            "last_accepted_timestamp_monotonic": self._last_accepted_monotonic,
            "last_accepted_sequence": int(self._last_accepted_sequence),
            "measured_qpos": measured_qpos,
            "measured_timestamp": measured_timestamp,
            "measured_timestamp_monotonic": measured_monotonic,
            "measured_sequence": measured_sequence,
            "local_qpos_normalization": local_qpos_normalization,
            "hardware_qpos": hardware_qpos,
            "last_target_qpos": last_target_qpos,
            "last_target_tcp_pose": last_target_tcp_pose,
            "last_command": last_command,
            "last_command_timestamp": self._last_command_timestamp,
            "arm_state_source": (
                "hardware_observation_only"
                if self.config.observation_only
                else "last_commanded_action"
            ),
            "observation_only": bool(self.config.observation_only),
            "hardware_pose_6d": hardware_pose_6d,
            # Compatibility name: this is the timestamp of the last successful
            # 6D joint measurement, never the timestamp of a failed read.
            "hardware_refresh_time": hardware_refresh_time,
            "hardware_measurement_timestamp": measured_timestamp,
            "hardware_measurement_sequence": measured_sequence,
            "hardware_measurement_valid": bool(
                hardware_qpos is not None and hardware_refresh_time is not None
            ),
            "hardware_read_time": hardware_read_time,
            "hardware_read_timestamp_monotonic": hardware_read_monotonic,
            "hardware_read_sequence": hardware_read_sequence,
            "hardware_read_success": hardware_read_success,
            "hardware_read_phase_timing": hardware_read_phase_timing,
            "hardware_result_code": hardware_result_code,
            "hardware_result_text": hardware_result_text,
            "hardware_arm_err": hardware_arm_err,
            "hardware_sys_err": hardware_sys_err,
            "send_result_code": self._last_send_result_code,
            "send_result_text": self._last_send_result_text,
            "hardware_ready": hardware_ready,
            "hardware_ready_source": hardware_ready_source,
            "emergency_stopped": emergency_stopped,
            "background_streaming": background_streaming,
            "canfd_follow": bool(self.config.canfd_follow),
            "canfd_trajectory_mode": int(self.config.canfd_trajectory_mode),
            "canfd_radio": int(self.config.canfd_radio),
            "effective_control_dt_sec": float(self._last_effective_control_dt),
            "effective_step_scale": float(self._last_effective_step_scale),
            "ik_metadata": ik_metadata,
            "singularity_metadata": singularity_metadata,
            "singularity_protection_ready": bool(self._singularity_protection_ready),
            "active_tool_name": self._active_tool_name,
            "active_tool_profile_path": self._active_tool_profile_path,
            "configured_tcp_enabled": self._configured_tcp_enabled,
            "tool_tcp_source": self._active_tcp_source(),
            "tool_tcp_pose_flange_m_rad": np.asarray(tcp_pose_flange, dtype=float),
            "tool_tcp_endpoint_delta_m": float(tcp_delta["translation_m"]),
            "tool_tcp_endpoint_delta_deg": float(tcp_delta["rotation_deg"]),
        }

    def set_local_qpos(self, qpos: Sequence[float] | np.ndarray | None) -> None:
        if qpos is None:
            qpos_array = None
            normalization = {
                "status": "cleared",
                "tolerance_deg": float(np.rad2deg(_RM65_LOCAL_QPOS_OBSERVATION_TOLERANCE_RAD)),
            }
        else:
            qpos_array, normalization = self.normalize_local_qpos_observation(
                qpos,
                label="local qpos",
            )
        with self._state_lock:
            self._last_local_qpos_normalization = dict(normalization)
            if qpos_array is None:
                self._current_qpos = None
                self._last_accepted_qpos = None
                return
            self._record_command_accepted_locked(
                qpos_array,
                timestamp=time.time(),
                monotonic_time=time.monotonic(),
            )
            self._jump_gate_active = False
            self._jump_hold_qpos = None
            self._jump_hold_anchor_pose = None
            self._jump_hold_frames = 0

    def compute_tcp_pose(
        self,
        tcp_frame: str | None = None,
        *,
        qpos: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        if qpos is None:
            with self._state_lock:
                if self._current_qpos is None:
                    raise RuntimeError(f"{self.name} does not have a current joint state yet.")
                qpos_array = self._current_qpos.copy()
        else:
            qpos_array = np.asarray(qpos, dtype=float).reshape(-1)
            if qpos_array.shape != (6,) or not np.all(np.isfinite(qpos_array)):
                raise RuntimeError(f"{self.name} does not have a current joint state yet.")
        tcp_frame = canonical_arm_controlled_frame(
            self.config.default_tcp_frame if tcp_frame is None else tcp_frame
        )
        solver = self._solver_for_tcp_frame(tcp_frame)
        solver_pose = np.asarray(solver.compute_pose(qpos_array), dtype=float)
        return self._pose_from_solver_frame(tcp_frame, solver_pose)

    def emergency_stop(self) -> None:
        if self.config.observation_only:
            with self._state_lock:
                self._emergency_stopped = True
                self._last_send_result_code = 0
                self._last_send_result_text = "observation_only_emergency_stop"
            return
        with self._send_lock:
            self._stream_stop_event.set()
            with self._state_lock:
                self._emergency_stopped = True
                self._stream_shutdown_requested = True
                self._clear_stream_target_locked()
            if self.arm is None:
                with self._state_lock:
                    self._last_send_result_code = 0
                    self._last_send_result_text = "emergency_stop_no_arm"
                return
            force_stop_called = bool(self._force_position_active)
            force_stop_error: str | None = None
            if force_stop_called:
                # A failed force-position stop must not prevent the ordinary
                # arm stop from being attempted.  The two SDK calls are
                # independent safety actions, and the latter is the final
                # stop path for both position and force-position modes.
                try:
                    force_stop_result = self._invoke_pre_sdk_control(
                        seam_id="rm65.safety.emergency_stop.force_and_arm",
                        permit_kind="safety_control",
                        operation="stop",
                        source="emergency_stop",
                        sdk_method="Stop_Force_Position_Move",
                        attempt_sequence=1,
                        backend_call=lambda: self.arm.Stop_Force_Position_Move(True),
                    )
                    force_stop_code, force_stop_text = _parse_rm65_result(
                        force_stop_result
                    )
                    if force_stop_code not in (0, None):
                        force_stop_error = (
                            f"Stop_Force_Position_Move returned {force_stop_text}"
                        )
                except Exception as exc:  # noqa: BLE001 - continue to Arm_Stop.
                    force_stop_error = f"{type(exc).__name__}: {exc}"
                self._force_position_active = False
            arm_stop_method = getattr(self.arm, "Arm_Stop", None)
            slow_stop_method = getattr(self.arm, "Arm_Slow_Stop", None)
            if not callable(arm_stop_method) and not callable(slow_stop_method):
                raise RuntimeError(f"{self.name} RM65 SDK exposes no Arm_Stop/Arm_Slow_Stop method")

            # Arm_Stop is the fastest stop and is safe to repeat after a
            # missing acknowledgement.  If both bounded attempts fail, use
            # Arm_Slow_Stop as a distinct final fallback when available.
            stop_attempts: list[tuple[str, Callable[[], object]]] = []
            if callable(arm_stop_method):
                stop_attempts.extend(
                    [("Arm_Stop", arm_stop_method)]
                    * int(_EMERGENCY_STOP_ARM_STOP_ATTEMPTS)
                )
            if callable(slow_stop_method):
                stop_attempts.append(("Arm_Slow_Stop", slow_stop_method))
            first_stop_sequence = 2 if force_stop_called else 1
            stop_failures: list[str] = []
            successful_stop: tuple[str, int | None, str] | None = None
            for index, (stop_label, stop_method) in enumerate(stop_attempts):
                if index:
                    time.sleep(float(_EMERGENCY_STOP_RETRY_DELAY_SEC))
                try:
                    stop_result = self._invoke_pre_sdk_control(
                        seam_id="rm65.safety.emergency_stop.force_and_arm",
                        permit_kind="safety_control",
                        operation="stop",
                        source="emergency_stop",
                        sdk_method=stop_label,
                        attempt_sequence=first_stop_sequence + index,
                        backend_call=stop_method,
                    )
                    stop_code, stop_text = _parse_rm65_result(stop_result)
                    if stop_code in (0, None):
                        successful_stop = (stop_label, stop_code, stop_text)
                        break
                    stop_failures.append(
                        f"{stop_label} returned {stop_text}"
                    )
                except Exception as exc:  # noqa: BLE001 - try the next stop path.
                    stop_failures.append(
                        f"{stop_label} {type(exc).__name__}: {exc}"
                    )

            if successful_stop is not None:
                stop_label, stop_code, stop_text = successful_stop
                with self._state_lock:
                    self._last_send_result_code = stop_code
                    self._last_send_result_text = (
                        f"emergency_stop:{stop_label}:{stop_text}"
                    )
                return

            details = stop_failures or ["no stop attempt completed"]
            if force_stop_error is not None:
                details.insert(0, force_stop_error)
            with self._state_lock:
                self._last_send_result_code = None
                self._last_send_result_text = (
                    "emergency_stop_failed:" + " | ".join(details)
                )
            raise RuntimeError(
                f"{self.name} emergency stop was not acknowledged after "
                f"{len(stop_attempts)} stop attempt(s): "
                + "; ".join(details)
            )

    def close(self) -> None:
        with self._send_lock:
            self._stream_stop_event.set()
            with self._state_lock:
                stop_before_disconnect = bool(
                    self.arm is not None
                    and not self._emergency_stopped
                    and self._last_attempted_sequence > 0
                    and not self._stream_fault_stop_success
                )
                self._closed = True
                self._hardware_ready = False
                self._hardware_ready_source = "closed"
                self._stream_shutdown_requested = True
                self._clear_stream_target_locked()
                stream_thread = self._stream_thread
        if stream_thread is not None and stream_thread is not threading.current_thread():
            stream_thread.join(timeout=float(self.config.background_streaming_shutdown_timeout_sec))
            join_timed_out = bool(stream_thread.is_alive())
            with self._state_lock:
                self._stream_join_timed_out = join_timed_out
                self._stream_shutdown_complete = not join_timed_out
                if not join_timed_out:
                    self._stream_thread = None
            if join_timed_out:
                self._latch_background_stream_fault(
                    "background stream did not stop before shutdown timeout"
                )
                raise RuntimeError(
                    f"{self.name} background stream did not stop within "
                    f"{float(self.config.background_streaming_shutdown_timeout_sec):.3f}s; "
                    "hardware connection was left open to avoid racing the worker."
                )
        else:
            with self._state_lock:
                self._stream_join_timed_out = False
                self._stream_shutdown_complete = True
                if stream_thread is not threading.current_thread():
                    self._stream_thread = None
        if self.arm is None:
            return
        try:
            if self.config.observation_only:
                self.arm.RM_API_UnInit()
                return
            force_stop_called = bool(self._force_position_active)
            if force_stop_called:
                self._invoke_pre_sdk_control(
                    seam_id="rm65.cleanup.close.stop_before_disconnect",
                    permit_kind="safety_control",
                    operation="stop",
                    source="close_pre_disconnect",
                    sdk_method="Stop_Force_Position_Move",
                    attempt_sequence=1,
                    backend_call=lambda: self.arm.Stop_Force_Position_Move(True),
                )
                self._force_position_active = False
            if stop_before_disconnect:
                stop_method = getattr(self.arm, "Arm_Slow_Stop", None)
                stop_label = "Arm_Slow_Stop"
                if not callable(stop_method):
                    stop_method = getattr(self.arm, "Arm_Stop", None)
                    stop_label = "Arm_Stop"
                if not callable(stop_method):
                    raise RuntimeError(
                        f"{self.name} RM65 SDK exposes no stop method for safe disconnect"
                    )
                stop_result = self._invoke_pre_sdk_control(
                    seam_id="rm65.cleanup.close.stop_before_disconnect",
                    permit_kind="safety_control",
                    operation="stop",
                    source="close_pre_disconnect",
                    sdk_method=stop_label,
                    attempt_sequence=2 if force_stop_called else 1,
                    backend_call=stop_method,
                )
                stop_code, stop_text = _parse_rm65_result(stop_result)
                with self._state_lock:
                    self._last_send_result_code = stop_code
                    self._last_send_result_text = (
                        f"close_pre_disconnect:{stop_label}:{stop_text}"
                    )
                if stop_code != 0:
                    raise RuntimeError(
                        f"{self.name} {stop_label} failed before disconnect with "
                        f"return code {stop_text}"
                    )
            self.arm.RM_API_UnInit()
        finally:
            try:
                self.arm.Arm_Socket_Close()
            finally:
                self.arm = None
