"""Composition-capable MuJoCo target factory for the Station Sim-HIL profile.

Shared scene preparation and measurement adapters live in their target-owned
modules.  This factory binds application-supplied composition contracts to the
lower MuJoCo runtime without defining composition strategy or application
lifecycle semantics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from robot_motion.action import (
    ComponentActionSchema,
    ExecutionObservationSink,
    HandActionSpec,
)
from robot_motion.coordinate_profiles import (
    TargetFrameProfile,
    ToolFrameProfile,
)
from .target_measurement import (
    build_mujoco_sim_hil_coordinate_profiles,
    build_mujoco_sim_hil_pose_observation_projector,
    capture_mujoco_sim_hil_safety_anchor,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)
from .target_preparation import (
    MujocoSimHILPreparedTarget,
    MujocoSimHILTargetConfig,
    prepare_mujoco_sim_hil_target,
)

if TYPE_CHECKING:
    from .composition_binding import MujocoCompositionRuntimeConfig


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
