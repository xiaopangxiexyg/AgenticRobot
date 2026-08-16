"""Shared MuJoCo scene preparation for deploy target factories.

This module owns loading and initializing simulator resources before an
application chooses composition, a direct Agent source, authority, safety, IK,
or dispatch wiring.  The historical Sim-HIL names remain stable because both
existing target profiles consume the same preparation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from robot_motion.action import ExecutionObservationPlan
from robot_motion.coordinate_profiles import TargetFrameProfile, ToolFrameProfile
from robot_motion.position_control import (
    POSITION_15_60_120_PROFILE,
    POSITION_45_150_PROFILE,
    POSITION_60_120_PROFILE,
    POSITION_PROFILE,
    PositionControlProfile,
)

from .target_measurement import build_mujoco_sim_hil_coordinate_profiles


_POSITION_CONTROL_PROFILES_BY_NAME = {
    profile.name: profile
    for profile in (
        POSITION_PROFILE,
        POSITION_15_60_120_PROFILE,
        POSITION_60_120_PROFILE,
        POSITION_45_150_PROFILE,
    )
}


@dataclass(frozen=True)
class MujocoSimHILTargetConfig:
    """Target-side selection shared by Station MuJoCo target profiles."""

    task_config_path: Path
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_config_path",
            Path(self.task_config_path).expanduser(),
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


@dataclass(frozen=True)
class MujocoSimHILPreparedTarget:
    """Prepared lower-owned resources before application contract binding.

    This phase exists so an application leaf can inspect the exact target
    component schema derived from the loaded scene, then bind either composed
    or direct-source contracts without constructing MuJoCo twice.
    """

    config: MujocoSimHILTargetConfig
    task_config: Any
    scene: Any
    position_profile: PositionControlProfile
    compose_target_hz: float
    timing: Any
    arm_profile: Any
    gripper_profile: Any
    target_frame_profile: TargetFrameProfile
    tool_frame_profile: ToolFrameProfile
    task_initialization_info: Mapping[str, object]


def prepare_mujoco_sim_hil_target(
    config: MujocoSimHILTargetConfig,
) -> MujocoSimHILPreparedTarget:
    """Load and initialize one MuJoCo target without authority or execution."""

    if not isinstance(config, MujocoSimHILTargetConfig):
        raise TypeError("config must be a MujocoSimHILTargetConfig.")
    from sim.contracts import TimingSpec
    from sim.mujoco.actuation import (
        identified_bimanual_dmtac_gripper_profile,
        rm65_bimanual_udp_feedback_profile,
    )
    from sim.mujoco.scene import MujocoScene
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    task_config = CubeLiftTaskConfig.load(config.task_config_path)
    contract = task_config.base_contract()
    if (
        not config.allow_scripted_task_reset
        and (
            bool(task_config.scripted_pregrasp_reset)
            or bool(task_config.scripted_contact_reset)
        )
    ):
        raise ValueError(
            "Generic Agent motion forbids task-specific scripted reset motion."
        )
    try:
        position_profile: PositionControlProfile = _POSITION_CONTROL_PROFILES_BY_NAME[
            str(config.position_control_profile_name).strip()
        ]
    except KeyError as exc:
        raise ValueError(
            "Unsupported Sim-HIL position control profile: "
            f"{config.position_control_profile_name!r}."
        ) from exc
    if not np.isclose(
        float(task_config.reference_hz),
        float(position_profile.rates.reference_hz),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Sim-HIL task reference_hz does not match the selected position "
            f"profile {position_profile.name!r}."
        )
    if not np.isclose(
        float(task_config.command_hz),
        float(position_profile.rates.command_hz),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Sim-HIL task command_hz does not match the selected position "
            f"profile {position_profile.name!r}."
        )
    compose_target_hz = position_profile.validate_compose_target_hz(
        position_profile.rates.sample_hz
        if config.compose_target_hz is None
        else config.compose_target_hz
    )
    timing = TimingSpec(
        sample_hz=compose_target_hz,
        policy_hz=task_config.policy_hz,
        reference_hz=position_profile.rates.reference_hz,
        ik_hz=position_profile.rates.ik_hz,
        command_hz=position_profile.rates.command_hz,
        physics_hz=task_config.physics_hz,
        parameter_source=(
            f"sim_mujoco_sim_hil:{position_profile.name}:"
            f"compose_{compose_target_hz:g}hz"
        ),
    )
    arm_profile = (
        task_config.position_actuation_profile()
        or rm65_bimanual_udp_feedback_profile()
    )
    gripper_profile = (
        task_config.gripper_actuation_profile()
        or identified_bimanual_dmtac_gripper_profile()
    )
    scene = MujocoScene.from_contract(
        contract,
        position_actuation_profile=arm_profile,
        gripper_actuation_profile=gripper_profile,
    )
    if config.collision_stop_enabled and not scene.collision_safety_available:
        raise ValueError(
            "Sim-HIL collision protection requires an enabled collision "
            "contract in the selected MuJoCo scene."
        )
    if task_config.scripted_pregrasp_reset or task_config.scripted_contact_reset:
        # The task kernel is the single owner of its simulator-only episode
        # preparation. Reuse it instead of copying the scripted trajectory.
        from sim.mujoco.tasks.cube_lift.kernel import CubeLiftKernel

        reset_kernel = CubeLiftKernel(
            task_config,
            scene=scene,
            info_mode="terminal",
            use_settled_snapshot=False,
        )
        try:
            _observation, reset_info = reset_kernel.reset(seed=0)
        finally:
            reset_kernel.close()
        if (
            bool(reset_info.get("scripted_pregrasp_reset", False))
            != bool(task_config.scripted_pregrasp_reset)
            or bool(reset_info.get("scripted_contact_reset", False))
            != bool(task_config.scripted_contact_reset)
        ):
            raise RuntimeError(
                "CubeLift task initialization did not apply its declared "
                "scripted reset semantics."
            )
        task_initialization_info = {
            "owner": "sim.mujoco.tasks.cube_lift.CubeLiftKernel.reset",
            "seed": 0,
            "resting_center_z_m": float(reset_info["resting_center_z_m"]),
            "scripted_pregrasp_reset": bool(
                reset_info.get("scripted_pregrasp_reset", False)
            ),
            "scripted_contact_reset": bool(
                reset_info.get("scripted_contact_reset", False)
            ),
        }
    else:
        scene.reset_data()
        scene.set_qpos_by_side(
            {
                "left": task_config.left_home_qpos,
                "right": task_config.right_home_qpos,
            }
        )
        for name, close in task_config.initial_gripper_close_by_name.items():
            scene.set_gripper_close(name, float(close))
        scene.set_freejoint_state(
            task_config.object_freejoint_name,
            qpos=task_config.cube_spawn_qpos,
            qvel=(0.0,) * 6,
        )
        scene.forward()
        task_initialization_info = {
            "owner": "sim.mujoco.deploy.prepare_mujoco_sim_hil_target",
            "seed": None,
            "scripted_pregrasp_reset": False,
            "scripted_contact_reset": False,
        }
    target_frame_profile, tool_frame_profile = (
        build_mujoco_sim_hil_coordinate_profiles(
            scene,
            task_config,
            active_sides=config.exposed_pose_sides,
        )
    )
    return MujocoSimHILPreparedTarget(
        config=config,
        task_config=task_config,
        scene=scene,
        position_profile=position_profile,
        compose_target_hz=compose_target_hz,
        timing=timing,
        arm_profile=arm_profile,
        gripper_profile=gripper_profile,
        target_frame_profile=target_frame_profile,
        tool_frame_profile=tool_frame_profile,
        task_initialization_info=MappingProxyType(task_initialization_info),
    )


__all__ = [
    "MujocoSimHILPreparedTarget",
    "MujocoSimHILTargetConfig",
    "prepare_mujoco_sim_hil_target",
]
