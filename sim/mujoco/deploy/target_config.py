"""Shared configuration contract for Station MuJoCo target profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_motion.action import ExecutionObservationPlan
from robot_motion.position_control import POSITION_15_60_120_PROFILE


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATION_POSE_CONTRACT_PATH = (
    _REPO_ROOT
    / "robot_motion/configs/pose_frames/"
    "rm65_dual_arm_workspace_near_boundary_v1_collection_desk_absolute.json"
)


@dataclass(frozen=True)
class MujocoSimHILTargetConfig:
    """Target-side selection shared by Station MuJoCo target profiles."""

    task_config_path: Path
    # Optional canonical station binding. Existing policy/human Sim-HIL
    # profiles preserve their historical scene-world frame when omitted;
    # semantic motion Skills select the reviewed station contract explicitly.
    station_pose_contract_path: Path | None = None
    realtime_worker: bool = True
    runtime_session_id: str = "control-station-sim-hil-ppo-session"
    control_domain_id: str = "mujoco-sim-hil"
    embodiment_instance_id: str = "mujoco-rm65-sim-hil"
    producer_id: str = "sim-hil-temp-ppo"
    composer_producer_id: str = "control-station-sim-hil-composer"
    policy_node_id: str = "sim-hil-temp-ppo"
    clock_domain_id: str = "mujoco.sim_hil.monotonic"
    # Synchronous pose resolution is admitted at 15 Hz; the lower planner and
    # command clocks remain 60/120 Hz.  A 60 Hz compose profile remains an
    # explicit option for integrations with a sufficiently fast resolver.
    position_control_profile_name: str = POSITION_15_60_120_PROFILE.name
    # Optional explicit final compose/authority cadence.  When omitted, the
    # selected profile's accepted input cadence is used.
    compose_target_hz: float | None = None
    target_admission_margin_sec: float = 0.05
    # Sim-only startup selection.  When false the coupled pose planner keeps
    # joint limits and IK but installs no collision pairs; MuJoCo contact
    # physics itself is unchanged.
    collision_stop_enabled: bool = True
    target_pose_smoothing_enabled: bool = True
    execution_observation_plan: ExecutionObservationPlan | None = None
    exposed_pose_sides: tuple[str, ...] = ("left", "right")
    # Generic Agent motion must not inherit task-specific scripted reset
    # trajectories. Existing CubeLift Sim-HIL profiles retain their behavior.
    allow_scripted_task_reset: bool = True
    agent_skill_producer_id: str = "control-station-agent-skill"
    # Optional application-owned scene asset overrides. Keep these at the end
    # of the dataclass so existing positional construction remains stable.
    scene_urdf_path_override: Path | None = None
    scene_camera_registry_path_override: Path | None = None
    scene_gripper_actuation_profile_name_override: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_config_path",
            Path(self.task_config_path).expanduser(),
        )
        for name in (
            "scene_urdf_path_override",
            "scene_camera_registry_path_override",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser())
        if self.scene_gripper_actuation_profile_name_override is not None:
            profile_name = str(
                self.scene_gripper_actuation_profile_name_override
            ).strip()
            if not profile_name:
                raise ValueError(
                    "scene_gripper_actuation_profile_name_override must be non-empty."
                )
            object.__setattr__(
                self,
                "scene_gripper_actuation_profile_name_override",
                profile_name,
            )
        if self.station_pose_contract_path is not None:
            object.__setattr__(
                self,
                "station_pose_contract_path",
                Path(self.station_pose_contract_path).expanduser(),
            )
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "producer_id",
            "composer_producer_id",
            "policy_node_id",
            "clock_domain_id",
            "position_control_profile_name",
            "agent_skill_producer_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if not isinstance(self.allow_scripted_task_reset, bool):
            raise TypeError("allow_scripted_task_reset must be a bool.")
        if not isinstance(self.collision_stop_enabled, bool):
            raise TypeError("collision_stop_enabled must be a bool.")
        if not isinstance(self.target_pose_smoothing_enabled, bool):
            raise TypeError("target_pose_smoothing_enabled must be a bool.")
        if self.compose_target_hz is not None:
            compose_target_hz = float(self.compose_target_hz)
            if not np.isfinite(compose_target_hz) or compose_target_hz <= 0.0:
                raise ValueError("compose_target_hz must be finite and positive.")
            object.__setattr__(self, "compose_target_hz", compose_target_hz)
        target_margin = float(self.target_admission_margin_sec)
        if not np.isfinite(target_margin) or target_margin <= 0.0:
            raise ValueError(
                "target_admission_margin_sec must be finite and positive."
            )
        object.__setattr__(
            self,
            "target_admission_margin_sec",
            target_margin,
        )
        if self.execution_observation_plan is not None and not isinstance(
            self.execution_observation_plan,
            ExecutionObservationPlan,
        ):
            raise TypeError(
                "execution_observation_plan must be an "
                "ExecutionObservationPlan or None."
            )
        sides = tuple(str(side).strip().lower() for side in self.exposed_pose_sides)
        if (
            not sides
            or any(side not in {"left", "right"} for side in sides)
            or len(sides) != len(set(sides))
        ):
            raise ValueError("exposed_pose_sides must contain unique known sides.")
        object.__setattr__(self, "exposed_pose_sides", sides)


__all__ = [
    "DEFAULT_STATION_POSE_CONTRACT_PATH",
    "MujocoSimHILTargetConfig",
]
