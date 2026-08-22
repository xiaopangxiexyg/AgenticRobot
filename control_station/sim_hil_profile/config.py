"""Pure configuration contract for the default-off MuJoCo Sim-HIL profile.

This module validates profile selection and Station lifecycle settings only.  It
does not construct devices, load a policy, initialize MuJoCo, or own execution.
The canonical CLI continues to live in :mod:`control_station.sim_hil`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from robot_motion.position_control import POSITION_15_60_120_PROFILE_NAME

from .profile_selection import SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY
from .validation import _finite

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_HIL_CHECKPOINT_ENV = "DEXTELEOP_SIM_HIL_CHECKPOINT"
LOCAL_DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "sim"
    / "mujoco_temp_pipeline"
    / "records"
    / "20260728_215045_cube_lift_pregrasp_bridge_v5_center_guard"
    / "checkpoints"
    / "step_000050176_final.pt"
)


def resolve_default_checkpoint_path(
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the optional host-supplied Sim-HIL checkpoint fail closed."""

    environment = os.environ if environ is None else environ
    configured = str(environment.get(SIM_HIL_CHECKPOINT_ENV, "")).strip()
    if not configured:
        return LOCAL_DEFAULT_CHECKPOINT
    checkpoint = Path(configured).expanduser()
    if not checkpoint.is_absolute():
        raise ValueError(f"{SIM_HIL_CHECKPOINT_ENV} must be an absolute path.")
    return checkpoint.resolve()


DEFAULT_CHECKPOINT = resolve_default_checkpoint_path()
DEFAULT_TASK_CONFIG = (
    REPO_ROOT
    / "sim"
    / "configs"
    / "tasks"
    / "rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json"
)
DEFAULT_CALIBRATION_PROFILE = (
    REPO_ROOT
    / "teleop_records"
    / "calibration_profiles"
    / "20260627_112114_vive_rm65_virtual_desk.json"
)
DEFAULT_VIVE_COORDINATE_PROFILE = (
    REPO_ROOT
    / "teleop_records"
    / "control_station_vive_coordinate_profiles"
    / "vive_rm65_virtual_desk_coordinate_v1.json"
)
DEFAULT_VIVE_PORT = 9999
DEFAULT_STATUS_HOST = "127.0.0.1"
DEFAULT_STATUS_PORT = 8787
DEFAULT_SIM_HIL_COMPOSE_TARGET_HZ = 30.0
SIM_HIL_BLEND_FAILURE_FALLBACKS = frozenset({"hold", "human", "policy"})
SIM_HIL_HUMAN_TIMEOUT_FALLBACKS = frozenset({"hold", "policy"})
SIM_HIL_POLICY_BACKENDS = frozenset({"temp_ppo", "openpi_state"})
_LOOPBACK_STATUS_HOSTS = frozenset({"127.0.0.1", "localhost"})


@dataclass(frozen=True)
class MujocoSimHILApplicationConfig:
    """Station-only profile selection and lifecycle settings."""

    checkpoint_path: Path = DEFAULT_CHECKPOINT
    policy_backend: str = "temp_ppo"
    openpi_policy_artifact_path: Path | None = None
    openpi_host: str = "127.0.0.1"
    openpi_port: int = 8000
    openpi_prompt: str | None = None
    openpi_action_hz: float = 15.0
    openpi_policy_hz: float = 15.0
    openpi_action_chunk_execution_steps: int | None = None
    openpi_prediction_queue_threshold: int = 4
    task_config_path: Path = DEFAULT_TASK_CONFIG
    calibration_profile_path: Path = DEFAULT_CALIBRATION_PROFILE
    vive_coordinate_profile_path: Path | None = DEFAULT_VIVE_COORDINATE_PROFILE
    vive_port: int = DEFAULT_VIVE_PORT
    vive_clock_sync_port: int | None = None
    vive_source_tracker_role: str = "left_wrist"
    left_q6_port: str | None = None
    right_q6_port: str | None = None
    gripper_input_source: str = "auto"
    q6_config: str = "latest"
    angle_trigger_config_root: Path | None = None
    angle_trigger_rate_hz: float = 30.0
    device: str = "auto"
    enabled: bool = False
    runtime_session_id: str = "control-station-sim-hil-ppo-session"
    control_domain_id: str = "mujoco-sim-hil"
    # Application-only selection. The compatibility profile remains default;
    # other profiles must be explicitly selected and completely assembled.
    control_profile_id: str = (
        SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY
    )
    realtime_worker: bool = True
    # Explicit final compose/authority target cadence.  ``control_rate_hz``
    # remains a compatibility alias for the historical --rate option.
    position_control_profile_name: str = POSITION_15_60_120_PROFILE_NAME
    # 30 Hz is the measured interactive compromise for the current
    # synchronous composition path.  Policy inference retains its own lower
    # cadence; the selected 15->60->120 position profile permits composed
    # targets anywhere from 15 through 60 Hz.
    compose_target_hz: float | None = None
    control_rate_hz: float | None = None
    warmup_timeout_sec: float = 10.0
    # Sim-HIL-only human input freshness window.  This is deliberately
    # separate from teleop_runtime's tracker-loss HOLD thresholds: it allows
    # a short network gap to be observed without changing real-robot safety
    # semantics or retaining an old pose as a fresh command.
    human_emission_timeout_sec: float = 5.0
    target_admission_margin_sec: float = 0.05
    initial_human_alpha: float = 0.5
    blend_failure_fallback: str = "hold"
    # The lower execution path still HOLDs every infeasible target
    # immediately. This grace only delays a Station-owned source switch from
    # proportional blend to human or policy.
    blend_failure_grace_sec: float = 0.0
    human_timeout_fallback: str = "hold"
    defer_human_input: bool = False
    execution_difference_enabled: bool = True
    # Pre-construction Sim behavior. Enabled preserves the existing collision
    # rejection path; disabled allows MuJoCo contacts without stopping the
    # Sim-HIL episode. This never applies to a real-robot target.
    sim_collision_stop_enabled: bool = True
    # Reuse the real RM65 pre-IK target-shaping contract by default.  This
    # remains independently switchable for A/B diagnosis and rollback.
    target_pose_smoothing_enabled: bool = True
    mujoco_viewer_enabled: bool = False
    status_host: str = DEFAULT_STATUS_HOST
    status_port: int = DEFAULT_STATUS_PORT

    def __post_init__(self) -> None:
        if not isinstance(self.sim_collision_stop_enabled, bool):
            raise TypeError("sim_collision_stop_enabled must be a bool.")
        if not isinstance(self.target_pose_smoothing_enabled, bool):
            raise TypeError("target_pose_smoothing_enabled must be a bool.")
        for name in (
            "checkpoint_path",
            "task_config_path",
            "calibration_profile_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        backend = str(self.policy_backend).strip().lower()
        if backend not in SIM_HIL_POLICY_BACKENDS:
            raise ValueError(
                "policy_backend must be temp_ppo or openpi_state."
            )
        object.__setattr__(self, "policy_backend", backend)
        artifact_path = self.openpi_policy_artifact_path
        if artifact_path is not None:
            artifact_path = Path(artifact_path).expanduser()
            object.__setattr__(self, "openpi_policy_artifact_path", artifact_path)
        if backend == "openpi_state" and artifact_path is None:
            raise ValueError(
                "openpi_state policy backend requires openpi_policy_artifact_path."
            )
        openpi_host = str(self.openpi_host).strip()
        openpi_port = int(self.openpi_port)
        if not openpi_host or not 0 < openpi_port <= 65535:
            raise ValueError("openpi_host/openpi_port are invalid.")
        object.__setattr__(self, "openpi_host", openpi_host)
        object.__setattr__(self, "openpi_port", openpi_port)
        if self.openpi_prompt is not None:
            prompt = str(self.openpi_prompt).strip()
            object.__setattr__(self, "openpi_prompt", prompt or None)
        for name in ("openpi_action_hz", "openpi_policy_hz"):
            value = _finite(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        execution_steps = self.openpi_action_chunk_execution_steps
        if execution_steps is not None:
            execution_steps = int(execution_steps)
            if execution_steps <= 0:
                raise ValueError(
                    "openpi_action_chunk_execution_steps must be positive."
                )
            object.__setattr__(
                self,
                "openpi_action_chunk_execution_steps",
                execution_steps,
            )
        queue_threshold = int(self.openpi_prediction_queue_threshold)
        if queue_threshold < 0:
            raise ValueError(
                "openpi_prediction_queue_threshold must be non-negative."
            )
        object.__setattr__(
            self,
            "openpi_prediction_queue_threshold",
            queue_threshold,
        )
        for name in ("runtime_session_id", "control_domain_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        if self.vive_coordinate_profile_path is not None:
            object.__setattr__(
                self,
                "vive_coordinate_profile_path",
                Path(self.vive_coordinate_profile_path).expanduser(),
            )
        if self.angle_trigger_config_root is not None:
            object.__setattr__(
                self,
                "angle_trigger_config_root",
                Path(self.angle_trigger_config_root).expanduser(),
            )
        for name in ("left_q6_port", "right_q6_port"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} must be non-empty when supplied.")
        vive_role = str(self.vive_source_tracker_role).strip()
        if vive_role not in {"left_wrist", "right_wrist"}:
            raise ValueError(
                "vive_source_tracker_role must be left_wrist or right_wrist."
            )
        object.__setattr__(self, "vive_source_tracker_role", vive_role)
        source = str(self.gripper_input_source).strip().lower()
        if source not in {"auto", "q6", "angle_trigger"}:
            raise ValueError(
                "gripper_input_source must be auto, q6, or angle_trigger."
            )
        object.__setattr__(self, "gripper_input_source", source)
        profile_id = str(self.control_profile_id).strip()
        if not profile_id:
            raise ValueError("control_profile_id must be non-empty.")
        object.__setattr__(self, "control_profile_id", profile_id)
        q6_config = str(self.q6_config).strip()
        if not q6_config:
            raise ValueError("q6_config must be non-empty.")
        object.__setattr__(self, "q6_config", q6_config)
        angle_rate = _finite(
            self.angle_trigger_rate_hz,
            name="angle_trigger_rate_hz",
        )
        if angle_rate <= 0.0:
            raise ValueError("angle_trigger_rate_hz must be positive.")
        object.__setattr__(self, "angle_trigger_rate_hz", angle_rate)
        vive_port = int(self.vive_port)
        if not 0 <= vive_port <= 65535:
            raise ValueError("vive_port must be in [0, 65535].")
        object.__setattr__(self, "vive_port", vive_port)
        if self.vive_clock_sync_port is not None:
            sync_port = int(self.vive_clock_sync_port)
            if not 0 <= sync_port <= 65535:
                raise ValueError("vive_clock_sync_port must be in [0, 65535].")
            object.__setattr__(self, "vive_clock_sync_port", sync_port)
        timeout = _finite(self.warmup_timeout_sec, name="warmup_timeout_sec")
        if timeout <= 0.0:
            raise ValueError("warmup_timeout_sec must be positive.")
        object.__setattr__(self, "warmup_timeout_sec", timeout)
        human_timeout = _finite(
            self.human_emission_timeout_sec,
            name="human_emission_timeout_sec",
        )
        if human_timeout <= 0.0:
            raise ValueError("human_emission_timeout_sec must be positive.")
        object.__setattr__(self, "human_emission_timeout_sec", human_timeout)
        target_margin = _finite(
            self.target_admission_margin_sec,
            name="target_admission_margin_sec",
        )
        if target_margin <= 0.0:
            raise ValueError("target_admission_margin_sec must be positive.")
        object.__setattr__(self, "target_admission_margin_sec", target_margin)
        human_alpha = _finite(self.initial_human_alpha, name="initial_human_alpha")
        if human_alpha < 0.0 or human_alpha > 1.0:
            raise ValueError("initial_human_alpha must be in [0, 1].")
        object.__setattr__(self, "initial_human_alpha", human_alpha)
        blend_failure_fallback = str(self.blend_failure_fallback).strip().lower()
        if blend_failure_fallback not in SIM_HIL_BLEND_FAILURE_FALLBACKS:
            raise ValueError(
                "blend_failure_fallback must be hold, human, or policy."
            )
        object.__setattr__(
            self,
            "blend_failure_fallback",
            blend_failure_fallback,
        )
        blend_failure_grace = _finite(
            self.blend_failure_grace_sec,
            name="blend_failure_grace_sec",
        )
        if blend_failure_grace < 0.0:
            raise ValueError("blend_failure_grace_sec must be non-negative.")
        object.__setattr__(
            self,
            "blend_failure_grace_sec",
            blend_failure_grace,
        )
        human_timeout_fallback = str(self.human_timeout_fallback).strip().lower()
        if human_timeout_fallback not in SIM_HIL_HUMAN_TIMEOUT_FALLBACKS:
            raise ValueError("human_timeout_fallback must be hold or policy.")
        object.__setattr__(
            self,
            "human_timeout_fallback",
            human_timeout_fallback,
        )
        object.__setattr__(self, "defer_human_input", bool(self.defer_human_input))
        profile_name = str(self.position_control_profile_name).strip()
        if not profile_name:
            raise ValueError("position_control_profile_name must be non-empty.")
        object.__setattr__(self, "position_control_profile_name", profile_name)
        compose_rate = None
        if self.compose_target_hz is not None:
            compose_rate = _finite(self.compose_target_hz, name="compose_target_hz")
            if compose_rate <= 0.0:
                raise ValueError("compose_target_hz must be positive.")
            object.__setattr__(self, "compose_target_hz", compose_rate)
        if self.control_rate_hz is not None:
            legacy_rate = _finite(self.control_rate_hz, name="control_rate_hz")
            if legacy_rate <= 0.0:
                raise ValueError("control_rate_hz must be positive.")
            if compose_rate is not None and not math.isclose(
                legacy_rate,
                compose_rate,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "control_rate_hz is a compatibility alias and must match "
                    "compose_target_hz when both are supplied."
                )
            object.__setattr__(self, "control_rate_hz", legacy_rate)
            if compose_rate is None:
                compose_rate = legacy_rate
                object.__setattr__(self, "compose_target_hz", legacy_rate)
        elif compose_rate is None:
            compose_rate = DEFAULT_SIM_HIL_COMPOSE_TARGET_HZ
            object.__setattr__(self, "compose_target_hz", compose_rate)
        status_port = int(self.status_port)
        if not 0 <= status_port <= 65535:
            raise ValueError("status_port must be in [0, 65535].")
        object.__setattr__(self, "status_port", status_port)
        if not isinstance(self.execution_difference_enabled, bool):
            raise TypeError("execution_difference_enabled must be a bool.")
        if not isinstance(self.mujoco_viewer_enabled, bool):
            raise TypeError("mujoco_viewer_enabled must be a bool.")
        status_host = str(self.status_host).strip().lower()
        if status_host not in _LOOPBACK_STATUS_HOSTS:
            raise ValueError(
                "status_host must be loopback-only (127.0.0.1 or localhost)."
            )
        object.__setattr__(self, "status_host", status_host)


def acceptance_profile_config(
    config: MujocoSimHILApplicationConfig,
) -> MujocoSimHILApplicationConfig:
    """Return the base configuration for interactive control."""

    return replace(config, initial_human_alpha=1.0)


__all__ = [
    "DEFAULT_CALIBRATION_PROFILE",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_SIM_HIL_COMPOSE_TARGET_HZ",
    "DEFAULT_STATUS_HOST",
    "DEFAULT_STATUS_PORT",
    "DEFAULT_TASK_CONFIG",
    "DEFAULT_VIVE_COORDINATE_PROFILE",
    "DEFAULT_VIVE_PORT",
    "LOCAL_DEFAULT_CHECKPOINT",
    "SIM_HIL_BLEND_FAILURE_FALLBACKS",
    "SIM_HIL_CHECKPOINT_ENV",
    "SIM_HIL_HUMAN_TIMEOUT_FALLBACKS",
    "SIM_HIL_POLICY_BACKENDS",
    "MujocoSimHILApplicationConfig",
    "acceptance_profile_config",
    "resolve_default_checkpoint_path",
]
