"""No-model Agent-skill target over the general MuJoCo deploy runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from robot_motion.action import ExecutionObservationSink
from robot_motion.coordinate_profiles import TargetFrameProfile, ToolFrameProfile

from .sim_hil_target import (
    MujocoSimHILPreparedTarget,
    MujocoSimHILTargetConfig,
    build_mujoco_sim_hil_pose_observation_projector,
    capture_mujoco_sim_hil_safety_anchor,
    prepare_mujoco_sim_hil_target,
)


@dataclass(frozen=True)
class MujocoAgentSkillMotionTarget:
    """No-model, arms-only target bound to the shared MuJoCo executor."""

    task_config: Any
    scene: Any
    runtime: Any
    target_frame_profile: TargetFrameProfile
    tool_frame_profile: ToolFrameProfile
    task_initialization_info: Mapping[str, object]
    tcp_pose_envelope_runtime: Any | None = None


def build_mujoco_agent_skill_motion_target(
    *,
    config: MujocoSimHILTargetConfig,
    component_execution_binding: Any,
    prepared_target: MujocoSimHILPreparedTarget | None = None,
    execution_observation_sink: ExecutionObservationSink | None = None,
    trace_sink: Callable[[Any], None] | None = None,
    tcp_pose_envelope_runtime: Any | None = None,
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
        tcp_pose_envelope_runtime=tcp_pose_envelope_runtime,
    )


__all__ = [
    "MujocoAgentSkillMotionTarget",
    "build_mujoco_agent_skill_motion_target",
]
