"""Single-policy MuJoCo target for a semantic Skill execution route.

The target consumes the backend-neutral ``PolicyActionEmission`` surface of
the shared deploy runtime.  It deliberately installs no composition runtime:
the selected policy is the sole proposal source, while canonical safety,
coupled IK, reference generation, execution-gate transitions and dispatch
remain owned by :class:`MujocoDeployRuntime`.

This module does not define a public Skill, choose a checkpoint, or decide a
task terminal predicate.  Those are application/deployment bindings above the
target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from robot_motion.action import ExecutionObservationSink, HandActionSpec
from robot_motion.coordinate_profiles import TargetFrameProfile, ToolFrameProfile

from .sim_hil_target import (
    MujocoSimHILPreparedTarget,
    MujocoSimHILTargetConfig,
    build_mujoco_sim_hil_pose_observation_projector,
    capture_mujoco_sim_hil_safety_anchor,
    prepare_mujoco_sim_hil_target,
)


@dataclass(frozen=True)
class MujocoPolicySkillMotionTarget:
    """One default-off, single-policy target over the common lower runtime."""

    task_config: Any
    scene: Any
    runtime: Any
    hand_specs_by_side: Mapping[str, HandActionSpec]
    target_frame_profile: TargetFrameProfile
    tool_frame_profile: ToolFrameProfile
    task_initialization_info: Mapping[str, object]
    tcp_pose_envelope_runtime: Any | None = None


def _bound_hand_specs(component_execution_binding: Any) -> dict[str, HandActionSpec]:
    materializer = getattr(component_execution_binding, "materializer", None)
    profile = getattr(materializer, "profile", None)
    configured = dict(
        getattr(materializer, "hand_action_spec_by_component", {}) or {}
    )
    grippers = dict(
        getattr(profile, "gripper_name_by_component", {}) or {}
    )
    if set(configured) != set(grippers):
        raise ValueError(
            "Policy target materializer must bind every gripper component."
        )
    by_side: dict[str, HandActionSpec] = {}
    for component_id, spec in configured.items():
        if not isinstance(spec, HandActionSpec):
            raise TypeError("Policy target hand action specs must be typed.")
        if spec.side in by_side:
            raise ValueError("Policy target supports at most one hand per side.")
        by_side[spec.side] = spec
    return by_side


def build_mujoco_policy_skill_motion_target(
    *,
    config: MujocoSimHILTargetConfig,
    component_execution_binding: Any,
    prepared_target: MujocoSimHILPreparedTarget | None = None,
    execution_observation_sink: ExecutionObservationSink | None = None,
    trace_sink: Callable[[Any], None] | None = None,
    tcp_pose_envelope_runtime: Any | None = None,
) -> MujocoPolicySkillMotionTarget:
    """Build a no-composition target for one already-selected policy source.

    The component binding owns the exact legacy-policy-to-native-component
    representation edge.  This builder does not infer or widen its effector
    scope from model output.
    """

    if not isinstance(config, MujocoSimHILTargetConfig):
        raise TypeError("config must be a MujocoSimHILTargetConfig.")
    if config.allow_scripted_task_reset:
        raise ValueError(
            "Policy Skill target requires allow_scripted_task_reset=False."
        )
    ingress = getattr(component_execution_binding, "policy_action_ingress", None)
    ingress_hash = getattr(
        component_execution_binding,
        "policy_action_ingress_contract_hash",
        None,
    )
    if not callable(ingress) or not isinstance(ingress_hash, str) or not ingress_hash:
        raise ValueError(
            "Policy Skill target requires an exact target-owned policy ingress."
        )
    materializer = getattr(component_execution_binding, "materializer", None)
    profile = getattr(materializer, "profile", None)
    action_schema = getattr(component_execution_binding, "action_schema", None)
    if (
        action_schema is None
        or profile is None
        or getattr(profile, "action_schema", None) is not action_schema
        or not dict(getattr(profile, "arm_side_by_component", {}) or {})
    ):
        raise ValueError(
            "Policy Skill target requires one exact MuJoCo pose-component profile."
        )
    hand_specs = _bound_hand_specs(component_execution_binding)
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
    from robot_motion.rm65 import (
        DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
        RM65PositionCollisionPlannerConfig,
    )
    from robot_motion.target_safety import (
        RobotActionTCPPoseEnvelopeFilter,
        TCPPoseEnvelopeSafetyRuntime,
    )
    from sim.mujoco.adapters import (
        MujocoPositionExecutionConfig,
        MujocoPositionExecutionRig,
    )
    from sim.mujoco.ik import RM65CanonicalIK

    from .pose_resolution import MujocoCoupledPoseToQposResolver
    from .runtime import MujocoDeployConfig, MujocoDeployRuntime

    rig = MujocoPositionExecutionRig(
        prepared.scene,
        MujocoPositionExecutionConfig(
            timing=prepared.timing,
            active_sides=("left", "right"),
            actuation_profile=prepared.arm_profile,
            gripper_actuation_profile=(
                prepared.gripper_profile if hand_specs else None
            ),
            realtime_worker=config.realtime_worker,
            follow_qpos_alpha=1.0,
        ),
    )
    safety_profile = CanonicalKeyframeSafetyProfile(
        profile_id="control-station-mujoco-policy-skill-safety-v1",
        limits=PositionKeyframeSafetyConfig(workspace_enabled=True),
        hand_limits_by_side={
            side: HandQposSafetyLimits.uniform(spec, lower=0.0, upper=1.0)
            for side, spec in hand_specs.items()
        },
    )

    def shared_safety(action: Any, input_id: str) -> Any:
        anchor = rig.capture_scene(
            lambda scene: capture_mujoco_sim_hil_safety_anchor(
                scene,
                prepared.task_config,
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

    collision_clearance_m = float(
        prepared.scene.scene_spec.collision.safety_minimum_clearance_m
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
            collision_min_distance_m=collision_clearance_m,
            collision_distance_epsilon_m=min(
                DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
                collision_clearance_m,
            ),
            collision_constraints_enabled=config.collision_stop_enabled,
        ),
    )
    pose_target_safety_filter = None
    if tcp_pose_envelope_runtime is not None:
        if not isinstance(
            tcp_pose_envelope_runtime,
            TCPPoseEnvelopeSafetyRuntime,
        ):
            raise TypeError("tcp_pose_envelope_runtime must be typed or None.")
        pose_target_safety_filter = RobotActionTCPPoseEnvelopeFilter(
            tcp_pose_envelope_runtime
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
            enable_primary_policy_authority=True,
            enable_composer_authority=False,
            enable_replay_authority=False,
            producer_epoch=1,
            policy_node_id=config.policy_node_id,
            decision_hz=prepared.compose_target_hz,
            reference_activation_profile_id=IMMEDIATE_QREF_SIM_V1,
            selected_segment_max_lifetime_sec=support_horizon_sec,
            proposal_horizon_sec=support_horizon_sec,
            clock_domain_id=config.clock_domain_id,
            pose_frame_contract_hash="mujoco-policy-skill-rm65-arm-base-v1",
            compound_hand_action_specs_by_side=hand_specs,
            initial_control_mode="policy",
            enable_component_execution_handoff=True,
            target_admission_margin_sec=config.target_admission_margin_sec,
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
            pose_target_safety_filter=pose_target_safety_filter,
        ),
        shared_safety=shared_safety,
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
    return MujocoPolicySkillMotionTarget(
        task_config=prepared.task_config,
        scene=prepared.scene,
        runtime=runtime,
        hand_specs_by_side=hand_specs,
        target_frame_profile=prepared.target_frame_profile,
        tool_frame_profile=prepared.tool_frame_profile,
        task_initialization_info=prepared.task_initialization_info,
        tcp_pose_envelope_runtime=tcp_pose_envelope_runtime,
    )


__all__ = [
    "MujocoPolicySkillMotionTarget",
    "build_mujoco_policy_skill_motion_target",
]
