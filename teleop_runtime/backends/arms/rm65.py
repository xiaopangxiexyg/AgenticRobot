from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import threading
from typing import Mapping, Sequence

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
from rm65_inspire_tac.ik_tuning import (
    RM65IKTuningConfig,
    hard_branch_jump_exceeds,
    joint_jump_exceeds,
    load_ik_tuning_config,
    pose_delta_metrics,
)
from rm65_inspire_tac.paths import require_asset_path
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
from robot_motion.position_control.trajectory import (
    AccelerationLimitedJointTrajectoryPlanner,
    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    JointTrajectoryLimits,
    JointTrajectoryProposal,
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_LINEAR,
    normalize_position_command_planner,
)
from robot_motion.runtime.dispatch import BackendDispatchError, CommandDispatchReceipt

from ...interfaces import EffectorBackend
from ...registry import register_backend
from ...schemas import EffectorCommand
from ...defaults import DEFAULT_RM65_POSITION_ARM_CONTROL_HZ


RM65_IK_FRAMEWORK_ROBOT_MOTION = "robot_motion"
RM65_IK_FRAMEWORK_CHOICES = (RM65_IK_FRAMEWORK_ROBOT_MOTION,)
HAND_BASE_POSE_FLANGE_M_RAD = (0.0, 0.0, 0.009, 0.0, 0.0, 0.0)
_SINGULARITY_WARNING_INTERVAL_SEC = 1.0
DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN = 0.2
DEFAULT_SINGULARITY_REPLAY_SLOWDOWN = 0.2
_SINGULARITY_QPOS_INIT_SLOW_SOURCES = frozenset(
    {
        "replay_qpos_init",
        "replay_qpos_init_final_hold",
        "deploy_fixed_init_qpos",
    }
)
_SINGULARITY_LABELS = {
    0: "normal",
    -1: "shoulder_singularity",
    -2: "elbow_singularity",
    -3: "wrist_singularity",
}

# These are the absolute position limits in the canonical RM65 URDF.  They are
# intentionally enforced again at the transport boundary: upstream IK/safety
# may be bypassed by a direct joint command, but an unsafe value must never
# reach Movej_CANFD.
_RM65_ABSOLUTE_Q_MIN_RAD = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
_RM65_ABSOLUTE_Q_MAX_RAD = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
# The final 150 Hz transport boundary is a hard qpos envelope.  Upstream
# interpolation may approach an endpoint, but it may never use a tolerance to
# emit a value beyond it.
_RM65_ABSOLUTE_LIMIT_TOLERANCE_RAD = 0.0
# RM65 SDK joint feedback is reported in degrees and converted back to radians
# here. A local preview/home anchor may therefore exceed an endpoint by a few
# encoder/float quanta even though no command was sent beyond the envelope.
# This tolerance is intentionally *not* used by the transport validator above.
_RM65_LOCAL_QPOS_OBSERVATION_TOLERANCE_RAD = float(np.deg2rad(0.01))
_BACKGROUND_AUDIT_STATS_MAX_SAMPLES = 20_000
_BACKGROUND_AUDIT_SNAPSHOT_CACHE_SEC = 1.0
_BACKGROUND_WORKER_TERMINAL_STAGE = "rm65_background_stream_worker_terminal"
_POSITION_TRAJECTORY_HARD_STOP_COLLISION_STATUSES = frozenset(
    {"hold_last_safe", "jump_gate_hold", "unsafe_candidate", "failed"}
)
_POSITION_TRAJECTORY_HARD_STOP_JUMP_STATUSES = frozenset(
    {"hold_entered", "holding"}
)
_POSITION_TRAJECTORY_DEGRADED_JUMP_STATUSES = frozenset(
    {"warning", "recovering", "released"}
)
_POSITION_TRAJECTORY_NORMAL_SINGULARITY_STATUSES = frozenset(
    {"pass", "disabled", "not_checked"}
)
@dataclass(frozen=True)
class _PositionTrajectoryPolicy:
    mode: str
    speed_scale: float
    quick_stop: bool
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        return ",".join(self.reasons) if self.reasons else "normal"


def _metadata_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _final_follow_qpos_target(
    planned_qpos: np.ndarray,
    accepted_qpos: np.ndarray | None,
    *,
    alpha: float,
    bypass: bool,
) -> tuple[np.ndarray, bool]:
    """Apply the final transport EMA around the last SDK-accepted qpos."""

    planned = np.asarray(planned_qpos, dtype=float).reshape(6)
    if bypass or float(alpha) >= 1.0 or accepted_qpos is None:
        return planned.copy(), False
    accepted = np.asarray(accepted_qpos, dtype=float).reshape(6)
    output = accepted + float(alpha) * (planned - accepted)
    return output, not np.allclose(output, planned, rtol=0.0, atol=1e-12)


def _position_trajectory_policy(
    command_metadata: Mapping[str, object],
    singularity_metadata: Mapping[str, object],
    *,
    degraded_speed_scale: float,
) -> _PositionTrajectoryPolicy:
    """Classify an already validated teleop qref for the optional planner.

    Collision/no-valid-IK holds are fail-closed software quick stops.  A
    singularity-entry target is already replaced by the current accepted qpos
    in ``_apply_singularity_protection``; it therefore decelerates at the
    degraded speed instead of introducing another instantaneous stop. Other
    warnings whose qpos is still finite and safety-approved retain the target
    but lower desired speed. A finite IK pose residual is deliberately not a
    speed-degradation reason: it means the accepted safe qref has not matched
    the Cartesian request, and slowing its position convergence makes that
    residual worse. Transport faults never reach this classifier; they remain
    fault-latched by the background worker.
    """

    hard_stop_reasons: list[str] = []
    degraded_reasons: list[str] = []

    collision_status = str(command_metadata.get("collision_avoidance_status") or "").strip()
    if collision_status in _POSITION_TRAJECTORY_HARD_STOP_COLLISION_STATUSES:
        hard_stop_reasons.append(f"collision:{collision_status}")
    elif collision_status == "boundary_active":
        degraded_reasons.append("collision:boundary_active")
    elif collision_status and collision_status != "tracking":
        degraded_reasons.append(f"collision:{collision_status}")
    fallback_reason = str(
        command_metadata.get("collision_avoidance_fallback_reason") or ""
    ).strip()
    if fallback_reason and not hard_stop_reasons:
        degraded_reasons.append(f"collision_fallback:{fallback_reason}")

    jump_status = str(command_metadata.get("ik_jump_gate_status") or "").strip()
    if jump_status in _POSITION_TRAJECTORY_HARD_STOP_JUMP_STATUSES:
        hard_stop_reasons.append(f"ik_jump:{jump_status}")
    elif jump_status in _POSITION_TRAJECTORY_DEGRADED_JUMP_STATUSES:
        degraded_reasons.append(f"ik_jump:{jump_status}")

    singularity_status = str(
        singularity_metadata.get("singularity_gate_status") or ""
    ).strip()
    if (
        singularity_status
        and singularity_status not in _POSITION_TRAJECTORY_NORMAL_SINGULARITY_STATUSES
    ):
        degraded_reasons.append(f"singularity:{singularity_status}")

    execution_step = _metadata_mapping(command_metadata.get("execution_step_metadata"))
    if bool(execution_step.get("step_limited", False)):
        degraded_reasons.append("execution_step_limited")
    if bool(command_metadata.get("ik_time_budget_exceeded", False)):
        degraded_reasons.append("ik_time_budget_exceeded")

    pose_error = _metadata_mapping(command_metadata.get("ik_solve_pose_error"))
    if pose_error:
        try:
            translation_m = float(pose_error.get("translation_m", 0.0) or 0.0)
            rotation_deg = float(pose_error.get("rotation_deg", 0.0) or 0.0)
        except (TypeError, ValueError):
            hard_stop_reasons.append("ik_pose_error_invalid")
        else:
            if not math.isfinite(translation_m) or not math.isfinite(rotation_deg):
                hard_stop_reasons.append("ik_pose_error_nonfinite")
            # Finite pose error remains available in command/debug metadata.
            # The qref has already passed collision, joint-limit and IK safety
            # gates, so it must retain normal speed to converge position-first.

    def unique(values: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(value) for value in values if str(value)))

    if hard_stop_reasons:
        return _PositionTrajectoryPolicy(
            mode="quick_stop",
            speed_scale=0.0,
            quick_stop=True,
            reasons=unique(hard_stop_reasons),
        )
    if degraded_reasons:
        return _PositionTrajectoryPolicy(
            mode="degraded",
            speed_scale=float(degraded_speed_scale),
            quick_stop=False,
            reasons=unique(degraded_reasons),
        )
    return _PositionTrajectoryPolicy(
        mode="tracking",
        speed_scale=1.0,
        quick_stop=False,
        reasons=(),
    )


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
    position_command_planner: str = POSITION_COMMAND_PLANNER_LINEAR
    position_velocity_ramp_sec: float = DEFAULT_POSITION_VELOCITY_RAMP_SEC
    position_max_acceleration_rad_s2: tuple[float, ...] | None = None
    # Optional position-only envelope applied to every high-rate planner
    # proposal.  It is intentionally independent of max_angvel so a peak-step
    # guard does not reduce sustained tracking velocity.
    position_max_command_step_deg: float | None = None
    position_degraded_speed_scale: float = DEFAULT_POSITION_DEGRADED_SPEED_SCALE
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

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
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
        self.position_command_planner = normalize_position_command_planner(
            self.position_command_planner
        )
        if not math.isfinite(float(self.position_velocity_ramp_sec)) or float(
            self.position_velocity_ramp_sec
        ) <= 0.0:
            raise ValueError("position_velocity_ramp_sec must be finite and positive.")
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
            if self.position_command_planner != POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED:
                raise ValueError(
                    "position_max_command_step_deg requires the acceleration_limited "
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
    def __init__(self, config: RM65ArmBackendConfig):
        self.config = config
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
        self._emergency_stopped = False
        self._stream_target_qpos: np.ndarray | None = None
        self._stream_target_source = ""
        self._stream_target_timestamp: float | None = None
        self._stream_target_monotonic: float | None = None
        self._stream_target_sequence = 0
        self._stream_segment_start_qpos: np.ndarray | None = None
        self._stream_segment_start_time: float | None = None
        self._stream_segment_duration_sec = self._base_control_dt
        self._stream_reference_epoch: int | None = None
        self._stream_reference_sequence: int | None = None
        self._stream_planner_mode = normalize_position_command_planner(
            config.position_command_planner
        )
        planner_initial_qpos = (
            np.zeros(6, dtype=float)
            if self._current_qpos is None
            else self._current_qpos.copy()
        )
        if self._stream_planner_mode == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED:
            max_velocity = np.asarray(config.max_angvel, dtype=float).reshape(6)
            max_acceleration = (
                max_velocity / float(config.position_velocity_ramp_sec)
                if config.position_max_acceleration_rad_s2 is None
                else np.asarray(
                    config.position_max_acceleration_rad_s2,
                    dtype=float,
                ).reshape(6)
            )
            self._stream_planner: (
                JointCommandInterpolator | AccelerationLimitedJointTrajectoryPlanner
            ) = AccelerationLimitedJointTrajectoryPlanner(
                JointTrajectoryLimits(
                    max_velocity_rad_s=max_velocity,
                    max_acceleration_rad_s2=max_acceleration,
                    # One 150 Hz SDK tick may never jump farther than one
                    # outer safe-reference step, even after scheduler delay.
                    max_command_step_rad=self._base_max_joint_step,
                ),
                velocity_ramp_sec=float(config.position_velocity_ramp_sec),
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
        self._stream_target_qpos = None
        self._stream_target_source = ""
        self._stream_target_timestamp = None
        self._stream_target_monotonic = None
        self._stream_segment_start_qpos = None
        self._stream_segment_start_time = None
        self._stream_reference_epoch = None
        self._stream_reference_sequence = None
        # Invalidate a target already copied by the worker before it can enter
        # the SDK call.  This is what makes hold/close a hard send boundary.
        self._stream_target_sequence += 1
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

    def _activate_stream_quick_stop_locked(
        self,
        *,
        reason: str,
        monotonic_time: float,
        source: str,
    ) -> int:
        if not isinstance(
            self._stream_planner,
            AccelerationLimitedJointTrajectoryPlanner,
        ):
            raise RuntimeError("software quick stop requires acceleration_limited planner mode")
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
        target_sequence = int(self._stream_target_sequence + 1)
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
                        raw_result = method()
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
        return target.copy()

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
            self._configure_singularity_protection()
            if self.config.force_position_control:
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
                    stop_force(True)
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
        quick_stop_queued = False
        with self._send_lock:
            with self._state_lock:
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
            stop_result = stop_method()
            stop_code, stop_text = _parse_rm65_result(stop_result)
            with self._state_lock:
                self._last_send_result_code = stop_code
                self._last_send_result_text = f"hold:{stop_label}:{stop_text}"
            if stop_code != 0:
                raise RuntimeError(
                    f"{self.name} {stop_label} failed with return code {stop_text}"
                )

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
        if self._singularity_config_attempted:
            return
        self._singularity_config_attempted = True
        self._singularity_protection_ready = False
        if not self.config.singularity_protection_enabled or self.arm is None:
            return
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
        if self.arm is None:
            return {"status_code": None, "label": "unavailable", "reason": "no_arm"}
        analyzer = getattr(self.arm, "Algo_Kin_Singularity_Analyse", None)
        if analyzer is None:
            return {"status_code": None, "label": "unavailable", "reason": "missing_sdk_method"}
        joint_deg = (np.asarray(qpos_rad, dtype=float).reshape(6) / np.pi * 180.0).tolist()
        try:
            result = analyzer(joint_deg)
        except Exception as exc:  # noqa: BLE001 - protection should fail open for teleop.
            return {"status_code": None, "label": "unavailable", "reason": str(exc)}
        if not isinstance(result, tuple) or len(result) < 2:
            return {
                "status_code": None,
                "label": "unavailable",
                "reason": f"unexpected_result:{result}",
            }
        status, distance = result[:2]
        try:
            status_code = int(status)
        except (TypeError, ValueError):
            return {
                "status_code": None,
                "label": "unavailable",
                "reason": f"unexpected_status:{status}",
            }
        metadata: dict[str, object] = {
            "status_code": status_code,
            "label": _singularity_status_label(status_code),
        }
        try:
            metadata["wrist_center_distance_m"] = float(distance)
        except (TypeError, ValueError):
            metadata["wrist_center_distance_m"] = str(distance)
        return metadata

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
            if target_singular or qpos_init_slow_source or (
                replay_slow_source and replay_slowdown_preapplied
            )
            else {"status_code": 0, "label": "normal"}
        )
        current_code = current_status.get("status_code")
        current_singular = current_code not in (0, None)
        entering_singularity = bool(target_singular and current_code == 0)
        gate_status = "hold_entry" if entering_singularity else "pass"
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
            "configured_elbow_threshold_deg": float(self.config.singularity_elbow_threshold_deg),
            "configured_wrist_threshold_deg": float(self.config.singularity_wrist_threshold_deg),
            "configured_shoulder_threshold_m": float(self.config.singularity_shoulder_threshold_m),
        }
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
        if "wrist_center_distance_m" in target_status:
            metadata["target_wrist_center_distance_m"] = target_status["wrist_center_distance_m"]
        if "wrist_center_distance_m" in current_status:
            metadata["current_wrist_center_distance_m"] = current_status["wrist_center_distance_m"]
        self._record_singularity_metadata(metadata)
        if entering_singularity and not slow_singularity:
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

    def send_command(self, command: EffectorCommand) -> CommandDispatchReceipt:
        if command.effector_name != self.name:
            raise ValueError(
                f"{self.name} received command for effector '{command.effector_name}'."
            )
        with self._state_lock:
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
        command_start_time = time.time()
        command_source = _command_source(command)
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
        qpos_target = self._apply_singularity_protection(
            qpos_target,
            reference_qpos=reference_qpos,
            source=command_source,
            max_joint_step=max_joint_step,
            replay_slowdown_preapplied=replay_slowdown_preapplied,
        )
        qpos_target = self._validate_final_joint_target(qpos_target, label="final target")
        self._stamp_final_action_target(command, qpos_target, source=command_source)

        trajectory_policy: _PositionTrajectoryPolicy | None = None
        if (
            self.config.background_streaming_enabled
            and isinstance(
                self._stream_planner,
                AccelerationLimitedJointTrajectoryPlanner,
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

        if self.arm is not None and self.config.background_streaming_enabled:
            with self._send_lock:
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
                    transport_target_sequence = int(self._stream_target_sequence + 1)
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
                        self._stream_planner.accept_target(
                            qpos_target,
                            sequence=transport_target_sequence,
                            published_monotonic_time=transport_target_monotonic,
                            monotonic_time=transport_target_monotonic,
                            speed_scale=trajectory_policy.speed_scale,
                            mode=trajectory_policy.mode,
                            reason=trajectory_policy.reason,
                            anchor_q_command=segment_start_qpos,
                        )
                        self._stream_target_sequence = transport_target_sequence
                        self._stream_target_qpos = qpos_target.copy()
                        self._stream_target_source = command_source
                        self._stream_target_timestamp = transport_target_wall_time
                        self._stream_target_monotonic = transport_target_monotonic
                        self._stream_segment_start_qpos = segment_start_qpos
                        self._stream_segment_start_time = transport_target_monotonic
                        self._stream_segment_duration_sec = float(
                            self.config.position_velocity_ramp_sec
                        )
                        self._stream_quick_stop_active = False
                        self._stream_quick_stop_reason = ""
                        self._stream_planner_policy_mode = trajectory_policy.mode
                        self._stream_planner_policy_speed_scale = trajectory_policy.speed_scale
                        self._stream_planner_policy_reason = trajectory_policy.reason
                    self._stream_reference_epoch = _metadata_non_negative_int(
                        command.metadata,
                        "position_reference_epoch",
                    )
                    self._stream_reference_sequence = _metadata_non_negative_int(
                        command.metadata,
                        "position_reference_sequence",
                    )
                    command.metadata.update(
                        {
                            POSITION_TRANSPORT_TARGET_SEQUENCE_KEY: transport_target_sequence,
                            "position_transport_target_wall_time": transport_target_wall_time,
                            "position_transport_target_monotonic_time": transport_target_monotonic,
                            "position_transport_target_source": command_source,
                            "position_transport_planner": self._stream_planner_mode,
                        }
                    )
                    self._last_target_qpos = qpos_target.copy()
                    self._last_command = command.copy()
                    self._last_command_timestamp = command_start_time
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
                    raw_send_result = self._send_joint_target(qpos_target)
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
            "warning_threshold_hz": target_hz * warning_ratio,
            "warning_ratio": warning_ratio,
            "thread_alive": thread_alive,
            "target_active": self._stream_target_qpos is not None,
            "target_source": self._stream_target_source,
            "target_timestamp": self._stream_target_timestamp,
            "target_timestamp_monotonic": self._stream_target_monotonic,
            "target_sequence": int(self._stream_target_sequence),
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
                self._clear_stream_target_locked()
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
                if now_for_print - self._stream_last_warning_print_time >= 1.0:
                    print(
                        f"WARNING: {self.name} RM65 background CANFD stream low rate "
                        f"actual={actual_hz:.1f}Hz target={target_hz:.1f}Hz "
                        f"threshold={warning_threshold_hz:.1f}Hz"
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
                                sdk_attempt_sequence = self._record_command_attempt_locked(
                                    next_qpos,
                                    timestamp=sdk_attempt_wall_time,
                                    monotonic_time=sdk_attempt_monotonic_time,
                                )
                                sdk_attempted = True
                        if sdk_attempted:
                            call_started = time.monotonic()
                            raw_send_result: object = None
                            try:
                                raw_send_result = self._send_joint_target(next_qpos)
                            except Exception as exc:  # audited below, then fault-latched by caller.
                                sdk_exception = exc
                            call_finished = time.monotonic()
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
                                        self._stream_planner.commit(
                                            trajectory_proposal,
                                            accepted_monotonic_time=call_finished,
                                            accepted_q_command=next_qpos,
                                            allow_bounded_accepted_adjustment=bool(
                                                follow_smoothing_applied
                                            ),
                                        )
                                    accepted_sequence = self._record_command_accepted_locked(
                                        next_qpos,
                                        timestamp=time.time(),
                                        monotonic_time=call_finished,
                                    )
                                    self._last_send_time = wakeup_wall_time
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
                                "AccelerationLimitedJointTrajectoryPlanner"
                                if trajectory_proposal is not None
                                else "JointCommandInterpolator"
                            )
                        ),
                        "position_command_planner": self._stream_planner_mode,
                        "scheduler_original_scheduled_monotonic_time": (
                            tick_timing.original_scheduled_monotonic_time
                        ),
                        "scheduled_period_sec": period_sec,
                        "transport_block_reason": transport_block_reason,
                        "arm_follow_qpos_alpha": float(self.config.arm_follow_qpos_alpha),
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
        if self.arm is None:
            return
        if self.config.force_clear_on_start:
            clear_result = self.arm.Clear_Force_Data(True)
            clear_code, clear_text = _parse_rm65_result(clear_result)
            if clear_code != 0:
                raise RuntimeError(
                    f"{self.name} failed to clear force data before force-position control: {clear_text}"
                )
        start_result = self.arm.Start_Force_Position_Move(True)
        start_code, start_text = _parse_rm65_result(start_result)
        if start_code != 0:
            raise RuntimeError(
                f"{self.name} failed to start force-position control: {start_text}"
            )
        self._force_position_active = True

    def _send_joint_target(self, qpos_rad: np.ndarray) -> object:
        qpos = self._validate_final_joint_target(qpos_rad, label="transport joint target")
        if self.arm is None:
            return None
        joint_deg = qpos / np.pi * 180.0
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

    def refresh_state_from_hardware(
        self,
        *,
        hardware_read_retries: int | None = None,
        hardware_read_retry_delay_sec: float | None = None,
    ) -> dict[str, object]:
        if self.arm is None:
            return self.get_state()
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
            try:
                raw_result, joint_now, pose_now, arm_err, sys_err = self.arm.Get_Current_Arm_State()
            except Exception as exc:
                raw_result = f"EXCEPTION: {exc}"
                joint_now = None
                pose_now = None
                arm_err = None
                sys_err = None
            parsed_code, parsed_text = _parse_rm65_result(raw_result)
            qpos_from_joint_now = _joint_now_to_qpos(joint_now)
            if _state_result_is_usable(
                parsed_code,
                joint_now=joint_now,
                pose_now=pose_now,
            ):
                break
            if attempt + 1 < retry_count:
                time.sleep(retry_delay_sec)
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
        with self._state_lock:
            self._hardware_read_sequence += 1
            self._last_hardware_read_time = refresh_timestamp
            self._last_hardware_read_monotonic = refresh_monotonic
            self._last_hardware_read_success = measurement_success
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
        return self.get_state()

    def get_state(self) -> dict[str, object]:
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
        background_streaming["timing_percentiles"] = (
            self._background_stream_timing_percentiles_snapshot()
        )
        current_tcp_pose = None
        if current_qpos is not None:
            try:
                current_tcp_pose = self.compute_tcp_pose(qpos=current_qpos)
            except Exception:
                current_tcp_pose = None
        last_target_tcp_pose = None
        if last_target_qpos is not None:
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
            "current_qpos_source": "last_accepted_command_or_local_anchor",
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
            "arm_state_source": "last_commanded_action",
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
            qpos_array, normalization = self._normalize_local_qpos_observation(
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
            if self._force_position_active:
                self.arm.Stop_Force_Position_Move(True)
                self._force_position_active = False
            stop_method = getattr(self.arm, "Arm_Stop", None)
            stop_label = "Arm_Stop"
            if stop_method is None:
                stop_method = getattr(self.arm, "Arm_Slow_Stop", None)
                stop_label = "Arm_Slow_Stop"
            if stop_method is None:
                raise RuntimeError(f"{self.name} RM65 SDK exposes no Arm_Stop/Arm_Slow_Stop method")
            stop_result = stop_method()
            stop_code, stop_text = _parse_rm65_result(stop_result)
            with self._state_lock:
                self._last_send_result_code = stop_code
                self._last_send_result_text = f"emergency_stop:{stop_label}:{stop_text}"
            if stop_code not in (0, None):
                raise RuntimeError(f"{self.name} {stop_label} failed with return code {stop_text}")

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
            if self._force_position_active:
                self.arm.Stop_Force_Position_Move(True)
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
                stop_result = stop_method()
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
