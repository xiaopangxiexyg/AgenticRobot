"""MuJoCo target adapters used by the Control Station Sim-HIL profile.

This module contains only target-side measurement binding.  It does not own
composition, authority, canonical safety application, IK, or dispatch.  The
Station application supplies the profile selection and calls this adapter when
the shared safety callback needs a current measured anchor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from robot_motion.action import (
    ComponentActionSchema,
    ExecutionObservationPlan,
    ExecutionObservationSink,
    HandActionSpec,
)
from robot_motion.coordinate_profiles import (
    TargetFrameProfile,
    ToolFrameProfile,
)
from robot_motion.position_control import (
    POSITION_15_60_120_PROFILE,
    POSITION_45_150_PROFILE,
    POSITION_60_120_PROFILE,
    POSITION_PROFILE,
    PositionControlProfile,
)

from .target_measurement import (
    build_mujoco_sim_hil_coordinate_profiles,
    build_mujoco_sim_hil_pose_observation_projector,
    capture_mujoco_sim_hil_safety_anchor,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)

if TYPE_CHECKING:
    from .composition_binding import MujocoCompositionRuntimeConfig


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
    """Target-side selection for the Station Sim-HIL profile."""

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
        object.__setattr__(self, "task_config_path", Path(self.task_config_path).expanduser())
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
class MujocoSimHILTarget:
    """Already-owned MuJoCo target resources selected by the Station."""

    task_config: Any
    scene: Any
    runtime: Any
    hand_specs_by_side: Mapping[str, HandActionSpec]
    target_frame_profile: TargetFrameProfile
    tool_frame_profile: ToolFrameProfile
    task_initialization_info: Mapping[str, object]


@dataclass(frozen=True)
class MujocoSimHILPreparedTarget:
    """Prepared lower-owned resources before application contract binding.

    This phase exists so the application leaf can inspect the exact target
    component schema derived from the loaded scene, then build composition
    contracts, without constructing MuJoCo twice or moving composition
    semantics into the simulator package.
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
        # preparation.  Reuse it instead of copying the scripted trajectory
        # into Station, composition, or the policy source.
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
            "resting_center_z_m": float(
                reset_info["resting_center_z_m"]
            ),
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
            {"left": task_config.left_home_qpos, "right": task_config.right_home_qpos}
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


def build_mujoco_sim_hil_target(
    *,
    config: MujocoSimHILTargetConfig,
    contracts: Any,
    hand_specs_by_side: Mapping[str, HandActionSpec],
    composition_input_action_schema_by_role: Mapping[
        str, ComponentActionSchema
    ]
    | None = None,
    composition_runtime_config: MujocoCompositionRuntimeConfig | None = None,
    composition_strategy_control_factory: Callable[..., Any] | None = None,
    prepared_target: MujocoSimHILPreparedTarget | None = None,
    component_execution_binding: Any | None = None,
    execution_observation_sink: ExecutionObservationSink | None = None,
    trace_sink: Callable[[Any], None] | None = None,
) -> MujocoSimHILTarget:
    """Build scene, shared safety, coupled IK and one deploy runtime.

    The composition contract is supplied by the application leaf as an
    opaque binding.  This target factory only uses its already-built strategy
    and compound authority profile; it does not define composition semantics.
    """

    from .composition_binding import MujocoCompositionRuntimeConfig

    if not isinstance(config, MujocoSimHILTargetConfig):
        raise TypeError("config must be a MujocoSimHILTargetConfig.")
    if not hasattr(contracts, "authority_profile") or not hasattr(
        contracts,
        "strategy",
    ):
        raise TypeError("contracts must expose authority_profile and strategy.")
    if composition_runtime_config is None:
        raise ValueError(
            "MuJoCo Sim-HIL requires an application-supplied "
            "composition_runtime_config."
        )
    if not isinstance(
        composition_runtime_config,
        MujocoCompositionRuntimeConfig,
    ):
        raise TypeError(
            "composition_runtime_config must be a "
            "MujocoCompositionRuntimeConfig."
        )
    if not composition_runtime_config.enable_source_eligibility:
        raise ValueError(
            "MuJoCo Sim-HIL requires source-eligibility admission."
        )
    expected_information_plan = getattr(contracts, "information_plan", None)
    if (
        composition_runtime_config.information_plan
        is not expected_information_plan
    ):
        raise ValueError(
            "composition_runtime_config must preserve the contracts' "
            "information_plan identity."
        )
    if composition_strategy_control_factory is not None and not callable(
        composition_strategy_control_factory
    ):
        raise TypeError(
            "composition_strategy_control_factory must be callable or None."
        )
    if (
        getattr(contracts, "dynamic_parameter_contract", None) is not None
        and composition_strategy_control_factory is None
    ):
        raise ValueError(
            "Dynamic composition contracts require an application-supplied "
            "composition_strategy_control_factory."
        )
    hand_specs = {
        str(side): (
            spec
            if isinstance(spec, HandActionSpec)
            else HandActionSpec.from_metadata(spec)
        )
        for side, spec in dict(hand_specs_by_side).items()
    }
    if not hand_specs or not set(hand_specs).issubset({"left", "right"}):
        raise ValueError(
            "MuJoCo Sim-HIL target requires at least one known-side hand spec."
        )

    from robot_motion.action import (
        CanonicalKeyframeSafetyProfile,
        HandQposSafetyLimits,
        IMMEDIATE_QREF_SIM_V1,
        apply_canonical_keyframe_safety,
    )
    from robot_motion.position_control import PositionKeyframeSafetyConfig
    from robot_motion.pose_smoothing import (
        RM65_POSITION_TARGET_POSE_SMOOTHING_V1,
        build_bimanual_pose_smoother,
    )
    from sim.mujoco.adapters import (
        MujocoPositionExecutionConfig,
        MujocoPositionExecutionRig,
    )
    from robot_motion.rm65 import (
        DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
        RM65PositionCollisionPlannerConfig,
    )
    from sim.mujoco.ik import RM65CanonicalIK
    from .pose_resolution import MujocoCoupledPoseToQposResolver
    from .runtime import MujocoDeployConfig, MujocoDeployRuntime

    prepared = (
        prepare_mujoco_sim_hil_target(config)
        if prepared_target is None
        else prepared_target
    )
    if (
        not isinstance(prepared, MujocoSimHILPreparedTarget)
        or prepared.config != config
    ):
        raise ValueError("prepared_target belongs to another target config.")
    task_config = prepared.task_config
    scene = prepared.scene
    position_profile = prepared.position_profile
    compose_target_hz = prepared.compose_target_hz
    timing = prepared.timing
    arm_profile = prepared.arm_profile
    gripper_profile = prepared.gripper_profile
    target_frame_profile = prepared.target_frame_profile
    tool_frame_profile = prepared.tool_frame_profile
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing,
            active_sides=("left", "right"),
            actuation_profile=arm_profile,
            gripper_actuation_profile=gripper_profile,
            realtime_worker=config.realtime_worker,
            follow_qpos_alpha=1.0,
        ),
    )
    safety_profile = CanonicalKeyframeSafetyProfile(
        profile_id="control-station-mujoco-sim-hil-safety-v1",
        limits=PositionKeyframeSafetyConfig(workspace_enabled=False),
        hand_limits_by_side={
            side: HandQposSafetyLimits.uniform(spec, lower=0.0, upper=1.0)
            for side, spec in hand_specs.items()
        },
    )

    def shared_safety(action: Any, input_id: str) -> Any:
        anchor = rig.capture_scene(
            lambda captured_scene: capture_mujoco_sim_hil_safety_anchor(
                captured_scene,
                task_config,
                hand_specs,
                now=time.monotonic(),
            )
        )
        return apply_canonical_keyframe_safety(
            action,
            input_action_id=input_id,
            anchor=anchor,
            profile=safety_profile,
        )

    sim_collision_clearance_m = float(
        scene.scene_spec.collision.safety_minimum_clearance_m
    )
    ik = RM65CanonicalIK.from_tool_profiles(
        scene,
        {
            "left": task_config.left_tool_profile_path,
            "right": task_config.right_tool_profile_path,
        },
        planner_config=RM65PositionCollisionPlannerConfig(
            # This synchronous resolver advances once per composed target,
            # not at the downstream reference interpolation rate.  Computing
            # the per-call joint step at 60 Hz while invoking it at 15/30 Hz
            # silently reduced Cartesian tracking amplitude and speed.
            reference_hz=compose_target_hz,
            # Preserve the selected lower IK stage's computation budget.
            solve_time_budget_ms=1000.0 / float(position_profile.rates.ik_hz),
            # Sim-HIL uses the selected scene's preventive-clearance contract.
            # Inheriting the real-side 30 mm planner default here made a
            # proxy-distance rejection look like a physical MuJoCo collision,
            # even though the task contract declares a 2 mm Sim clearance.
            collision_min_distance_m=float(
                sim_collision_clearance_m
            ),
            collision_distance_epsilon_m=min(
                DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
                sim_collision_clearance_m,
            ),
            collision_constraints_enabled=config.collision_stop_enabled,
        ),
    )
    source_support_horizon_sec = max(
        0.25,
        1.0 / compose_target_hz,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id=config.runtime_session_id,
            control_domain_id=config.control_domain_id,
            embodiment_instance_id=config.embodiment_instance_id,
            producer_id=config.producer_id,
            composer_producer_id=config.composer_producer_id,
            enable_primary_policy_authority=False,
            enable_composer_authority=True,
            enable_replay_authority=False,
            producer_epoch=1,
            policy_node_id=config.policy_node_id,
            # The authoritative compose schedule is independent from the
            # lower policy source/inference schedule.
            decision_hz=compose_target_hz,
            # Interactive Sim-HIL publishes the latest qref immediately.
            # The scheduled-segment profile remains available to deterministic
            # tests, but its single future reservation is not a live stream.
            reference_activation_profile_id=IMMEDIATE_QREF_SIM_V1,
            selected_segment_max_lifetime_sec=(
                source_support_horizon_sec
            ),
            proposal_horizon_sec=source_support_horizon_sec,
            clock_domain_id=config.clock_domain_id,
            pose_frame_contract_hash="mujoco-sim-hil-rm65-arm-base-v1",
            compound_hand_action_specs_by_side=hand_specs,
            compound_composer_authority_profile=contracts.authority_profile,
            initial_control_mode="policy",
            enable_component_execution_handoff=True,
            target_admission_margin_sec=(
                config.target_admission_margin_sec
            ),
            execution_observation_plan=config.execution_observation_plan,
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(
            ik,
            target_pose_smoother=(
                build_bimanual_pose_smoother(
                    RM65_POSITION_TARGET_POSE_SMOOTHING_V1
                )
                if config.target_pose_smoothing_enabled
                else None
            ),
        ),
        shared_safety=shared_safety,
        composition_runtime_config=composition_runtime_config,
        composition_strategy=contracts.strategy,
        composition_strategy_control_factory=(
            composition_strategy_control_factory
        ),
        composition_input_action_schema_by_role=(
            composition_input_action_schema_by_role
        ),
        component_execution_binding=component_execution_binding,
        execution_observation_sink=execution_observation_sink,
        execution_pose_observation_projector=(
            build_mujoco_sim_hil_pose_observation_projector(
                prepared,
                component_execution_binding,
            )
            if (
                config.execution_observation_plan is not None
                and set(
                    config.execution_observation_plan
                    .required_representation_by_component.values()
                )
                == {"cartesian_pose"}
            )
            else None
        ),
        trace_sink=trace_sink,
    )
    return MujocoSimHILTarget(
        task_config=task_config,
        scene=scene,
        runtime=runtime,
        hand_specs_by_side=hand_specs,
        target_frame_profile=target_frame_profile,
        tool_frame_profile=tool_frame_profile,
        task_initialization_info=prepared.task_initialization_info,
    )


_AGENT_SKILL_COMPAT_EXPORTS = frozenset(
    {
        "MujocoAgentSkillMotionTarget",
        "build_mujoco_agent_skill_motion_target",
    }
)


def __getattr__(name: str):
    """Preserve the historical Agent target path as a lazy compatibility view."""

    if name not in _AGENT_SKILL_COMPAT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    owner = import_module(".agent_skill_target", __package__)
    value = getattr(owner, name)
    globals()[name] = value
    return value


__all__ = [
    "MujocoAgentSkillMotionTarget",
    "MujocoSimHILPreparedTarget",
    "MujocoSimHILTarget",
    "MujocoSimHILTargetConfig",
    "build_mujoco_agent_skill_motion_target",
    "build_mujoco_sim_hil_coordinate_profiles",
    "build_mujoco_sim_hil_pose_observation_projector",
    "build_mujoco_sim_hil_target",
    "capture_mujoco_sim_hil_safety_anchor",
    "capture_mujoco_sim_hil_tool_tcp_pose_by_side",
    "prepare_mujoco_sim_hil_target",
]
