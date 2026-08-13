"""MuJoCo target adapters used by the Control Station Sim-HIL profile.

This module contains only target-side measurement binding.  It does not own
composition, authority, canonical safety application, IK, or dispatch.  The
Station application supplies the profile selection and calls this adapter when
the shared safety callback needs a current measured anchor.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from project_paths import resolve_path
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    CanonicalActionSafetyAnchor,
    ComponentActionSchema,
    ExecutionObservationPlan,
    ExecutionObservationSink,
    HandActionSpec,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import (
    STATION_RETARGET_FRAME_CONTRACT_V1,
    TargetFrameProfile,
    ToolFrameBinding,
    ToolFrameProfile,
)
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.position_control import (
    POSITION_15_60_120_PROFILE,
    POSITION_45_150_PROFILE,
    POSITION_60_120_PROFILE,
    POSITION_PROFILE,
    PositionControlProfile,
)
from robot_motion.rm65 import endpoint_to_tool_transform, load_rm65_tool_profile


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


@dataclass(frozen=True)
class MujocoAgentSkillMotionTarget:
    """No-model, arms-only target bound to the shared MuJoCo executor."""

    task_config: Any
    scene: Any
    runtime: Any
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

    if not isinstance(config, MujocoSimHILTargetConfig):
        raise TypeError("config must be a MujocoSimHILTargetConfig.")
    if not hasattr(contracts, "authority_profile") or not hasattr(contracts, "strategy"):
        raise TypeError("contracts must expose authority_profile and strategy.")
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
    from control_composition.composition import (
        FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1,
        CompositionStrategyControl,
        build_initial_strategy_revision,
    )
    from control_composition.retention import TimeWindowRetentionSpec
    from sim.mujoco.adapters import (
        MujocoPositionExecutionConfig,
        MujocoPositionExecutionRig,
    )
    from robot_motion.rm65 import RM65PositionCollisionPlannerConfig
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
            collision_constraints_enabled=config.collision_stop_enabled,
        ),
    )
    strategy_control_factory = None
    dynamic_contract = getattr(
        contracts,
        "dynamic_parameter_contract",
        None,
    )
    if dynamic_contract is not None:
        initial_alpha = dict(contracts.initial_alpha_by_component)

        def strategy_control_factory(
            coordinator,
            producer_epochs,
            available_at,
        ):
            initial_revision = build_initial_strategy_revision(
                runtime_session_id=config.runtime_session_id,
                control_domain_id=config.control_domain_id,
                strategy_instance_id=(
                    f"{contracts.strategy_spec.strategy_id}:control"
                ),
                request_id=(
                    f"{contracts.strategy_spec.strategy_id}:initial"
                ),
                strategy_mode=(
                    FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1
                ),
                composition_spec=contracts.composition_spec,
                strategy_spec=contracts.strategy_spec,
                component_rule_set=contracts.component_rule_set,
                dynamic_parameter_contract=dynamic_contract,
                initial_alpha_by_component=initial_alpha,
                available_at=float(available_at),
                available_clock_domain_id=config.clock_domain_id,
            )
            return CompositionStrategyControl(
                runtime_session_id=config.runtime_session_id,
                control_domain_id=config.control_domain_id,
                strategy_instance_id=initial_revision.strategy_instance_id,
                coordinator=coordinator,
                producer_epochs=producer_epochs,
                retention_spec=TimeWindowRetentionSpec(
                    spec_id=(
                        "mujoco-sim-hil-strategy-control-window-v1"
                    ),
                    history_window_sec=2.0,
                    max_future_lead_sec=1.0,
                    max_effective_duration_sec=1.0,
                    max_late_arrival_sec=0.0,
                ),
                supported_strategy_modes=(
                    FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1,
                ),
                initial_revision=initial_revision,
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
            enable_composition_source_eligibility=True,
            composition_information_plan=getattr(
                contracts,
                "information_plan",
                None,
            ),
            execution_observation_plan=config.execution_observation_plan,
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(ik),
        shared_safety=shared_safety,
        composition_strategy=contracts.strategy,
        composition_strategy_control_factory=strategy_control_factory,
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


def build_mujoco_agent_skill_motion_target(
    *,
    config: MujocoSimHILTargetConfig,
    component_execution_binding: Any,
    prepared_target: MujocoSimHILPreparedTarget | None = None,
    execution_observation_sink: ExecutionObservationSink | None = None,
    trace_sink: Callable[[Any], None] | None = None,
) -> MujocoAgentSkillMotionTarget:
    """Build the first no-model, right-arm Agent Skill execution target.

    The direct source only emits the runtime's exact ``ComponentAction``.
    Canonical safety, coupled IK, reference generation and dispatch remain in
    :class:`MujocoDeployRuntime`; this builder installs no second controller.
    """

    if not isinstance(config, MujocoSimHILTargetConfig):
        raise TypeError("config must be a MujocoSimHILTargetConfig.")
    if config.allow_scripted_task_reset:
        raise ValueError(
            "Agent Skill motion target requires allow_scripted_task_reset=False."
        )
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
    action_schema = getattr(component_execution_binding, "action_schema", None)
    materializer = getattr(component_execution_binding, "materializer", None)
    profile = getattr(materializer, "profile", None)
    if (
        tuple(getattr(action_schema, "component_ids", ())) != ("right_arm",)
        or dict(getattr(profile, "arm_side_by_component", {}))
        != {"right_arm": "right"}
        or dict(getattr(profile, "gripper_name_by_component", {}))
    ):
        raise ValueError(
            "The first Agent motion target requires one right_arm pose "
            "component and no hand/gripper component."
        )

    from robot_motion.action import (
        CanonicalKeyframeSafetyProfile,
        IMMEDIATE_QREF_SIM_V1,
        apply_canonical_keyframe_safety,
    )
    from robot_motion.position_control import PositionKeyframeSafetyConfig
    from robot_motion.rm65 import RM65PositionCollisionPlannerConfig
    from sim.mujoco.adapters import (
        MujocoPositionExecutionConfig,
        MujocoPositionExecutionRig,
    )
    from sim.mujoco.ik import RM65CanonicalIK
    from .pose_resolution import MujocoCoupledPoseToQposResolver
    from .runtime import (
        MujocoDeployConfig,
        MujocoDeployRuntime,
        MujocoDirectComponentProposalSource,
    )

    rig = MujocoPositionExecutionRig(
        prepared.scene,
        MujocoPositionExecutionConfig(
            timing=prepared.timing,
            active_sides=("left", "right"),
            actuation_profile=prepared.arm_profile,
            # The generic profile has no hand action surface.
            gripper_actuation_profile=None,
            realtime_worker=config.realtime_worker,
            follow_qpos_alpha=1.0,
        ),
    )
    safety_profile = CanonicalKeyframeSafetyProfile(
        profile_id="control-station-mujoco-agent-skill-arms-safety-v1",
        limits=PositionKeyframeSafetyConfig(workspace_enabled=True),
        hand_limits_by_side={},
    )

    def shared_safety(action: Any, input_id: str) -> Any:
        anchor = rig.capture_scene(
            lambda captured_scene: capture_mujoco_sim_hil_safety_anchor(
                captured_scene,
                prepared.task_config,
                {},
                now=time.monotonic(),
            )
        )
        return apply_canonical_keyframe_safety(
            action,
            input_action_id=input_id,
            anchor=anchor,
            profile=safety_profile,
        )

    ik = RM65CanonicalIK.from_tool_profiles(
        prepared.scene,
        {
            "left": prepared.task_config.left_tool_profile_path,
            "right": prepared.task_config.right_tool_profile_path,
        },
        planner_config=RM65PositionCollisionPlannerConfig(
            reference_hz=prepared.compose_target_hz,
            solve_time_budget_ms=(
                1000.0 / float(prepared.position_profile.rates.ik_hz)
            ),
        ),
    )
    support_horizon_sec = max(0.25, 1.0 / prepared.compose_target_hz)
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id=config.runtime_session_id,
            control_domain_id=config.control_domain_id,
            embodiment_instance_id=config.embodiment_instance_id,
            producer_id=config.producer_id,
            composer_producer_id=config.composer_producer_id,
            enable_primary_policy_authority=False,
            enable_composer_authority=False,
            enable_replay_authority=False,
            producer_epoch=1,
            policy_node_id=config.policy_node_id,
            decision_hz=prepared.compose_target_hz,
            reference_activation_profile_id=IMMEDIATE_QREF_SIM_V1,
            selected_segment_max_lifetime_sec=support_horizon_sec,
            proposal_horizon_sec=support_horizon_sec,
            clock_domain_id=config.clock_domain_id,
            pose_frame_contract_hash="mujoco-agent-skill-rm65-arm-base-v1",
            initial_control_mode="agent_skill",
            enable_component_execution_handoff=True,
            target_admission_margin_sec=(
                config.target_admission_margin_sec
            ),
            execution_observation_plan=config.execution_observation_plan,
            direct_component_proposal_source=(
                MujocoDirectComponentProposalSource(
                    producer_id=config.agent_skill_producer_id,
                    producer_epoch=1,
                    authority_priority=10,
                )
            ),
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(ik),
        shared_safety=shared_safety,
        component_execution_binding=component_execution_binding,
        execution_observation_sink=execution_observation_sink,
        execution_pose_observation_projector=(
            build_mujoco_sim_hil_pose_observation_projector(
                prepared,
                component_execution_binding,
            )
            if config.execution_observation_plan is not None
            else None
        ),
        trace_sink=trace_sink,
    )
    return MujocoAgentSkillMotionTarget(
        task_config=prepared.task_config,
        scene=prepared.scene,
        runtime=runtime,
        target_frame_profile=prepared.target_frame_profile,
        tool_frame_profile=prepared.tool_frame_profile,
        task_initialization_info=prepared.task_initialization_info,
    )


def build_mujoco_sim_hil_pose_observation_projector(
    prepared_target: MujocoSimHILPreparedTarget,
    component_execution_binding: Any,
) -> Any:
    """Bind the active component profile to private-MjData FK observation."""

    if not isinstance(prepared_target, MujocoSimHILPreparedTarget):
        raise TypeError(
            "prepared_target must be a MujocoSimHILPreparedTarget."
        )
    materializer = getattr(component_execution_binding, "materializer", None)
    profile = getattr(materializer, "profile", None)
    arm_side_by_component = getattr(profile, "arm_side_by_component", None)
    action_schema = getattr(component_execution_binding, "action_schema", None)
    if not isinstance(arm_side_by_component, Mapping) or action_schema is None:
        raise TypeError(
            "Component execution binding must expose its MuJoCo arm profile."
        )
    from .pose_kinematics import (
        build_mujoco_pose_observation_projector,
    )

    scene = prepared_target.scene
    task_config = prepared_target.task_config
    tool_profile_path_by_source = {
        "left": task_config.left_tool_profile_path,
        "right": task_config.right_tool_profile_path,
    }
    endpoint_to_controlled_by_component = {}
    reference_body_by_component = {}
    endpoint_body_by_component = {}
    joint_names_by_source = {}
    for component_id, raw_source in arm_side_by_component.items():
        source = str(raw_source)
        payload_schema = action_schema.payload_schema_by_component[component_id]
        if payload_schema.controlled_frame != "tool_tcp":
            raise ValueError(
                "The current Sim-HIL FK profile requires tool_tcp arm outputs; "
                "another controlled frame must provide its own explicit offset."
            )
        tool_profile = load_rm65_tool_profile(
            tool_profile_path_by_source[source],
            side=source,
        )
        endpoint_to_controlled_by_component[component_id] = (
            endpoint_to_tool_transform(tool_profile)
        )
        reference_body_by_component[component_id] = _arm_base_body_name(
            scene,
            source,
        )
        endpoint_body_by_component[component_id] = (
            scene.arm_specs[source].controlled_body
        )
        joint_names_by_source[source] = tuple(
            scene.arm_specs[source].joint_names
        )

    model_contract_hash = canonical_content_hash(
        {
            "target_frame_profile_hash": (
                prepared_target.target_frame_profile.semantic_content_hash
            ),
            "tool_frame_profile_hash": (
                prepared_target.tool_frame_profile.semantic_content_hash
            ),
            "action_schema_hash": action_schema.semantic_content_hash,
            "scene_profile_name": scene.conversion.scene_profile_name,
            "model_nq": int(scene.model.nq),
            "model_nv": int(scene.model.nv),
        },
        domain="dexteleop.sim.mujoco.sim_hil_fk_model_contract.development",
    )
    return build_mujoco_pose_observation_projector(
        scene,
        action_schema=action_schema,
        qpos_source_by_component={
            str(component_id): str(source)
            for component_id, source in arm_side_by_component.items()
        },
        joint_names_by_source=joint_names_by_source,
        reference_body_by_component=reference_body_by_component,
        endpoint_body_by_component=endpoint_body_by_component,
        endpoint_to_controlled_by_component=(
            endpoint_to_controlled_by_component
        ),
        model_contract_id=(
            f"mujoco-sim-hil-fk:{scene.conversion.scene_profile_name}"
        ),
        model_contract_hash=model_contract_hash,
    )


def _file_sha256(path: str | Path) -> str:
    candidate = resolve_path(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_base_body_name(scene: Any, side: str) -> str:
    candidates = (f"{side}_arm_base_link", "arm_base_link", "base_link")
    base_name = next((name for name in candidates if scene.has_body(name)), None)
    if base_name is None:
        raise KeyError(f"Could not resolve {side} arm base body.")
    return base_name


def build_mujoco_sim_hil_coordinate_profiles(
    scene: Any,
    task_config: Any,
    *,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> tuple[TargetFrameProfile, ToolFrameProfile]:
    """Derive immutable target/tool identities from the loaded MuJoCo scene."""

    canonical_frame_id = "mujoco_sim_hil/station"
    frames = [
        CoordinateFrameSpec(
            canonical_frame_id,
            semantic_parameters={
                "frame_family": "target_station",
                "source": "loaded_mujoco_scene_world",
            },
        )
    ]
    transforms = []
    reference_by_component: dict[str, str] = {}
    sides = tuple(str(side).strip().lower() for side in active_sides)
    if (
        not sides
        or any(side not in {"left", "right"} for side in sides)
        or len(sides) != len(set(sides))
    ):
        raise ValueError("active_sides must contain unique known sides.")
    for side in sides:
        component_id = f"{side}_arm"
        reference_frame_id = f"{side}_arm_base"
        body_name = _arm_base_body_name(scene, side)
        frames.append(
            CoordinateFrameSpec(
                reference_frame_id,
                semantic_parameters={
                    "frame_family": "arm_base",
                    "side": side,
                    "scene_body_name": body_name,
                },
            )
        )
        transforms.append(
            RigidFrameTransformSpec(
                transform_id=f"mujoco-scene:{reference_frame_id}-to-station",
                source_frame_id=reference_frame_id,
                target_frame_id=canonical_frame_id,
                matrix_target_source=scene.body_pose(body_name),
                semantic_parameters={
                    "source": "loaded_scene_body_pose",
                    "scene_body_name": body_name,
                },
            )
        )
        reference_by_component[component_id] = reference_frame_id

    urdf_path = Path(scene.conversion.urdf_path).expanduser().resolve()
    target_profile = TargetFrameProfile(
        profile_id=f"mujoco-target-frame:{scene.conversion.scene_profile_name}",
        target_kind="simulation",
        artifact_id=str(urdf_path),
        artifact_hash=_file_sha256(urdf_path),
        frame_graph=CoordinateFrameGraph(
            graph_id=f"mujoco-target-frame:{scene.conversion.scene_profile_name}",
            frames=tuple(frames),
            transforms=tuple(transforms),
        ),
        canonical_reference_frame_id=canonical_frame_id,
        reference_frame_by_component=reference_by_component,
        semantic_parameters={
            "backend": "mujoco",
            "derivation": "loaded_scene_body_pose",
            "canonical_target_frame_contract_id": (
                STATION_RETARGET_FRAME_CONTRACT_V1
            ),
        },
    )

    tool_bindings = []
    profile_path_by_side = {
        "left": task_config.left_tool_profile_path,
        "right": task_config.right_tool_profile_path,
    }
    for side in sides:
        profile_path = profile_path_by_side[side]
        resolved = resolve_path(profile_path)
        tool_bindings.append(
            ToolFrameBinding(
                component_id=f"{side}_arm",
                controlled_frame_id="tool_tcp",
                artifact_id=str(resolved),
                artifact_hash=_file_sha256(resolved),
            )
        )
    tool_profile = ToolFrameProfile(
        profile_id=f"mujoco-tool-frame:{task_config.name}",
        target_frame_profile_hash=target_profile.semantic_content_hash,
        bindings=tuple(tool_bindings),
    )
    return target_profile, tool_profile


def capture_mujoco_sim_hil_tool_tcp_pose_by_side(
    scene: Any,
    task_config: Any,
) -> dict[str, np.ndarray]:
    """Capture current ``T_arm_base_tool_tcp`` for each active arm."""

    result: dict[str, np.ndarray] = {}
    tool_profiles = {
        "left": load_rm65_tool_profile(
            task_config.left_tool_profile_path,
            side="left",
        ),
        "right": load_rm65_tool_profile(
            task_config.right_tool_profile_path,
            side="right",
        ),
    }
    for side in ("left", "right"):
        base_name = _arm_base_body_name(scene, side)
        endpoint = scene.body_pose(scene.arm_specs[side].controlled_body)
        pose = np.linalg.inv(scene.body_pose(base_name)) @ endpoint
        result[side] = pose @ endpoint_to_tool_transform(tool_profiles[side])
    return result


def capture_mujoco_sim_hil_safety_anchor(
    scene: Any,
    task_config: Any,
    hand_specs: Mapping[str, HandActionSpec],
    *,
    now: float,
) -> CanonicalActionSafetyAnchor:
    """Capture one measured bimanual pose/hand anchor for shared safety."""

    arm_pose_by_side: dict[str, np.ndarray] = {}
    tool_tcp_pose_by_side = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
        scene,
        task_config,
    )
    for side, pose in tool_tcp_pose_by_side.items():
        vector = pose_matrix_to_xyzquat(pose)
        if vector is None:
            raise ValueError(f"Current {side} TCP pose cannot be represented as xyzquat.")
        arm_pose_by_side[side] = np.asarray(vector, dtype=np.float32)

    hand_qpos_by_side: dict[str, np.ndarray] = {}
    gripper_name_by_side = {
        "left": task_config.left_gripper_name,
        "right": task_config.right_gripper_name,
    }
    for side in hand_specs:
        name = gripper_name_by_side[side]
        values = np.zeros(6, dtype=np.float32)
        values[0] = float(scene.get_gripper_close(name))
        hand_qpos_by_side[side] = values
    return CanonicalActionSafetyAnchor(
        anchor_id=f"mujoco-sim-hil-measured-anchor:{float(now):.9f}",
        anchor_kind="measured_state",
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side=arm_pose_by_side,
        hand_qpos_by_side=hand_qpos_by_side,
        hand_action_specs_by_side=hand_specs,
        frame_by_side={side: f"{side}_arm_base" for side in ("left", "right")},
        observed_at=float(now),
    )


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
