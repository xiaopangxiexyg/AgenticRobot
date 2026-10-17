"""No-model Agent-skill target over the general MuJoCo deploy runtime."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from robot_motion.action import ExecutionObservationSink
from robot_motion.coordinate_profiles import TargetFrameProfile, ToolFrameProfile

from .target_config import MujocoSimHILTargetConfig
from .target_measurement import (
    build_mujoco_sim_hil_pose_observation_projector,
    capture_mujoco_sim_hil_safety_anchor,
)
from .target_preparation import (
    MujocoSimHILPreparedTarget,
    prepare_mujoco_sim_hil_target,
)


@dataclass(frozen=True)
class MujocoAgentSkillMotionTarget:
    """No-model right-arm action target on a rig with a separate gripper actuator."""

    task_config: Any
    scene: Any
    runtime: Any
    target_frame_profile: TargetFrameProfile
    tool_frame_profile: ToolFrameProfile
    task_initialization_info: Mapping[str, object]
    tcp_pose_envelope_runtime: Any | None = None
    planned_tcp_motion: Any | None = None


def build_mujoco_agent_skill_motion_target(
    *,
    config: MujocoSimHILTargetConfig,
    component_execution_binding: Any,
    prepared_target: MujocoSimHILPreparedTarget | None = None,
    execution_observation_sink: ExecutionObservationSink | None = None,
    trace_sink: Callable[[Any], None] | None = None,
    tcp_pose_envelope_runtime: Any | None = None,
    enable_safe_tcp_pose: bool = False,
    safe_tcp_planning_clearance_padding_m: float = 0.002,
) -> MujocoAgentSkillMotionTarget:
    """Build the first no-model, right-arm Agent Skill execution target.

    The direct source only emits the runtime's exact ``ComponentAction``.
    Canonical safety, coupled IK, reference generation and dispatch remain in
    :class:`MujocoDeployRuntime`; this builder installs no second controller.
    """

    if not isinstance(config, MujocoSimHILTargetConfig):
        raise TypeError("config must be a MujocoSimHILTargetConfig.")
    if not isinstance(enable_safe_tcp_pose, bool):
        raise TypeError("enable_safe_tcp_pose must be boolean.")
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
        IMMEDIATE_QREF_SIM_V1,
        CanonicalKeyframeSafetyProfile,
        apply_canonical_keyframe_safety,
    )
    from robot_motion.position_control import PositionKeyframeSafetyConfig
    from robot_motion.rm65 import (
        DEFAULT_COLLISION_SCENE,
        DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
        RM65_BIMANUAL_POSITION_V1,
        DualArmCollisionAwareRetargeter,
        DualArmCollisionAwareRetargeterConfig,
        RM65PositionCollisionPlannerConfig,
        load_ik_tuning_config,
        load_rm65_tool_profile,
        load_world_from_arm_base_transforms,
        tool_tcp_pose_flange_m_rad,
        with_rm65_position_ik_objective,
    )
    from robot_motion.target_safety import (
        RobotActionTCPPoseEnvelopeFilter,
        TCPPoseEnvelopeSafetyRuntime,
    )
    from sim.mujoco.adapters import (
        MujocoPositionExecutionConfig,
        MujocoPositionExecutionRig,
    )
    from sim.mujoco.ik import RM65CanonicalIK, RM65RetargeterIK

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
            # The ComponentAction schema remains right_arm-only. The existing
            # gripper Skill binds this actuator through its own lower port.
            gripper_actuation_profile=prepared.gripper_profile,
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

    ik_profile = RM65_BIMANUAL_POSITION_V1.ik_execution
    sim_collision_clearance_m = float(
        prepared.scene.scene_spec.collision.safety_minimum_clearance_m
    )
    ik_tuning = with_rm65_position_ik_objective(
        load_ik_tuning_config(),
        position_weight=ik_profile.position_weight,
        orientation_weight=ik_profile.orientation_weight,
        mode_name="mujoco_agent_skill_position_v1",
    )
    if config.collision_stop_enabled:
        initial_qpos = {
            side: prepared.scene.get_arm_qpos(side).copy()
            for side in ("left", "right")
        }
        retargeter = DualArmCollisionAwareRetargeter(
            DualArmCollisionAwareRetargeterConfig(
                scene_profile=DEFAULT_COLLISION_SCENE,
                solver_frequency=prepared.position_profile.rates.ik_hz,
                damping=ik_profile.damping,
                solve_time_budget_ms=(
                    1000.0 / float(prepared.position_profile.rates.ik_hz)
                ),
                solve_min_iterations=ik_profile.solve_min_iterations,
                solve_max_iterations=ik_profile.solve_max_iterations,
                solve_adaptive_iterations_enabled=ik_profile.adaptive_iterations,
                solve_iteration_reserve_ms=ik_profile.iteration_reserve_ms,
                candidate_certification_reserve_ms=(
                    ik_profile.candidate_certification_reserve_ms
                ),
                ik_tuning=ik_tuning,
                bimanual_result_contract_enabled=True,
                enforce_realistic_joint_step=True,
                left_initial_qpos=initial_qpos["left"],
                right_initial_qpos=initial_qpos["right"],
                left_tool_tcp_pose_flange_m_rad=tool_tcp_pose_flange_m_rad(
                    load_rm65_tool_profile(
                        prepared.task_config.left_tool_profile_path,
                        side="left",
                    )
                ),
                right_tool_tcp_pose_flange_m_rad=tool_tcp_pose_flange_m_rad(
                    load_rm65_tool_profile(
                        prepared.task_config.right_tool_profile_path,
                        side="right",
                    )
                ),
                joint_velocity_limit_profile=(
                    ik_profile.joint_velocity_limit_profile
                ),
                output_qpos_smoothing_alpha=1.0,
                output_qpos_smoothing_adaptive_enabled=False,
                output_qpos_smoothing_fast_alpha=1.0,
                inactive_side_hold_enabled=False,
                inactive_side_hold_execution_cap_enabled=False,
                inactive_side_hold_execution_release_enabled=False,
                prefer_non_wrist_singular_branch=(
                    ik_profile.prefer_non_wrist_singular_branch
                ),
                prefer_non_elbow_singular_branch=(
                    ik_profile.prefer_non_elbow_singular_branch
                ),
                prefer_non_shoulder_singular_branch=(
                    ik_profile.prefer_non_shoulder_singular_branch
                ),
                local_step_candidate_clip_scale=(
                    ik_profile.local_step_candidate_clip_scale
                ),
                collision_min_distance_m=sim_collision_clearance_m,
                collision_distance_epsilon_m=min(
                    DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
                    sim_collision_clearance_m,
                ),
            )
        )
        ik = RM65RetargeterIK(
            prepared.scene,
            retargeter,
            world_from_arm_base_by_side=(
                load_world_from_arm_base_transforms(retargeter.profile)
            ),
            # The coupled retargeter is a 60 Hz qref producer.  Its continuity
            # state is the command stream admitted by the 120 Hz planner, not
            # the lagging plant measurement.  The rig exposes measured qpos
            # separately for endpoint/recovery observation; feeding that lag
            # back into branch selection makes a physically normal servo delay
            # look like a new IK branch and causes target-edge chatter.
            admitted_qpos_source=rig.accepted_qcmd_by_side,
        )
    else:
        # The retargeter deliberately refuses to claim a no-collision
        # contract. Preserve the existing explicit diagnostic mode through
        # the canonical planner instead of weakening the production adapter.
        ik = RM65CanonicalIK.from_tool_profiles(
            prepared.scene,
            {
                "left": prepared.task_config.left_tool_profile_path,
                "right": prepared.task_config.right_tool_profile_path,
            },
            planner_config=RM65PositionCollisionPlannerConfig(
                # Component proposals are admitted at the 15 Hz compose
                # cadence, but the coupled planner owns the profile's 60 Hz
                # IK reference stage. Its joint-step envelope is defined at
                # that IK cadence; binding it to compose_target_hz would
                # silently permit a four-times larger per-reference step.
                reference_hz=prepared.position_profile.rates.ik_hz,
                solve_time_budget_ms=(
                    1000.0 / float(prepared.position_profile.rates.ik_hz)
                ),
                collision_min_distance_m=sim_collision_clearance_m,
                collision_distance_epsilon_m=min(
                    DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
                    sim_collision_clearance_m,
                ),
                collision_constraints_enabled=False,
                damping=ik_profile.damping,
                solve_min_iterations=ik_profile.solve_min_iterations,
                solve_max_iterations=ik_profile.solve_max_iterations,
                solve_adaptive_iterations_enabled=(
                    ik_profile.adaptive_iterations
                ),
                solve_iteration_reserve_ms=ik_profile.iteration_reserve_ms,
                ik_tuning=ik_tuning,
            ),
        )
    support_horizon_sec = max(0.25, 1.0 / prepared.compose_target_hz)
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
    pose_projector = (
        build_mujoco_sim_hil_pose_observation_projector(
            prepared,
            component_execution_binding,
        )
        if config.execution_observation_plan is not None
        else None
    )
    planned_tcp_motion = None
    if enable_safe_tcp_pose:
        if not config.collision_stop_enabled or pose_projector is None:
            raise ValueError(
                "Safe TCP motion requires collision checking and FK observation."
            )
        from .planned_tcp_motion import MujocoPlannedTcpMotion

        planned_tcp_motion = MujocoPlannedTcpMotion(
            rig=rig,
            task_config=prepared.task_config,
            pose_projector=pose_projector,
            reference_hz=prepared.compose_target_hz,
            planning_clearance_padding_m=safe_tcp_planning_clearance_padding_m,
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
            target_admission_margin_sec=config.target_admission_margin_sec,
            execution_observation_plan=config.execution_observation_plan,
            direct_component_proposal_source=(
                MujocoDirectComponentProposalSource(
                    producer_id=config.agent_skill_producer_id,
                    producer_epoch=1,
                    authority_priority=10,
                )
            ),
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(
            ik,
            planned_tcp_motion=planned_tcp_motion,
            pose_target_safety_filter=pose_target_safety_filter,
            support_only_arm_sides=tuple(
                effector.removesuffix("_arm")
                for effector in component_execution_binding.support_only_output_effectors
            ),
        ),
        shared_safety=shared_safety,
        component_execution_binding=component_execution_binding,
        execution_observation_sink=execution_observation_sink,
        execution_pose_observation_projector=pose_projector,
        trace_sink=trace_sink,
    )
    return MujocoAgentSkillMotionTarget(
        task_config=prepared.task_config,
        scene=prepared.scene,
        runtime=runtime,
        target_frame_profile=prepared.target_frame_profile,
        tool_frame_profile=prepared.tool_frame_profile,
        task_initialization_info=prepared.task_initialization_info,
        tcp_pose_envelope_runtime=tcp_pose_envelope_runtime,
        planned_tcp_motion=planned_tcp_motion,
    )


__all__ = [
    "MujocoAgentSkillMotionTarget",
    "build_mujoco_agent_skill_motion_target",
]
