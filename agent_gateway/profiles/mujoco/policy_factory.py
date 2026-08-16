"""Default-off factory for a single-policy MuJoCo Skill application.

The caller selects the policy source through the common
``PolicyEmissionSourcePort`` builder.  OpenPI is one possible builder, not a
Skill identity and not a dependency of this assembly module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ExecutionObservationSink, HandActionSpec
from robot_motion.deploy import require_policy_emission_source
from sim.mujoco.deploy import (
    MujocoComponentActionRobotActionV1Materializer,
    MujocoRobotActionV1ComponentIngressAdapter,
    MujocoSimHILTargetConfig,
    build_mujoco_component_execution_binding,
    build_mujoco_policy_skill_motion_target,
    build_mujoco_pose_qpos_component_target_profile,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
    prepare_mujoco_sim_hil_target,
)

from control_station.sim_hil_profile.policy_source import (
    SimHILPolicySourceBuildContext,
)
from .policy_application import MujocoPolicySkillMotionApplication


@dataclass(frozen=True)
class MujocoPolicySkillApplicationProfile:
    """Trusted assembly choice for one policy's authoritative output scope."""

    output_active_sides: tuple[str, ...]
    component_profile_id: str = "mujoco.policy_skill.pose_gripper.v1"

    def __post_init__(self) -> None:
        sides = tuple(str(side).strip().lower() for side in self.output_active_sides)
        if (
            not sides
            or any(side not in {"left", "right"} for side in sides)
            or len(sides) != len(set(sides))
        ):
            raise ValueError(
                "output_active_sides must contain unique left and/or right."
            )
        profile_id = str(self.component_profile_id).strip()
        if not profile_id:
            raise ValueError("component_profile_id must be non-empty.")
        object.__setattr__(self, "output_active_sides", sides)
        object.__setattr__(self, "component_profile_id", profile_id)


def build_mujoco_policy_skill_application(
    *,
    target_config: MujocoSimHILTargetConfig,
    application_profile: MujocoPolicySkillApplicationProfile,
    policy_source_builder: Callable[[SimHILPolicySourceBuildContext], Any],
    execution_observation_sink: ExecutionObservationSink | None = None,
    trace_sink: Callable[[Any], None] | None = None,
    tcp_pose_envelope_runtime: Any | None = None,
) -> MujocoPolicySkillMotionApplication:
    """Assemble one no-composition target and one injected policy source.

    Construction may prepare MuJoCo resources and validate policy artifacts,
    but neither the runtime nor the policy source is started here.
    """

    if not isinstance(target_config, MujocoSimHILTargetConfig):
        raise TypeError("target_config must be a MujocoSimHILTargetConfig.")
    if not isinstance(application_profile, MujocoPolicySkillApplicationProfile):
        raise TypeError("application_profile must be typed.")
    if not callable(policy_source_builder):
        raise TypeError("policy_source_builder must be callable.")
    if target_config.allow_scripted_task_reset:
        raise ValueError(
            "Policy Skill application requires allow_scripted_task_reset=False."
        )
    output_sides = application_profile.output_active_sides
    if not set(output_sides).issubset(set(target_config.exposed_pose_sides)):
        raise ValueError(
            "Target exposed_pose_sides must cover every policy output side."
        )

    prepared = prepare_mujoco_sim_hil_target(target_config)
    arm_components = {f"{side}_arm": side for side in output_sides}
    gripper_components = {
        f"{side}_gripper": getattr(
            prepared.task_config,
            f"{side}_gripper_name",
        )
        for side in output_sides
    }
    component_profile = build_mujoco_pose_qpos_component_target_profile(
        prepared.scene,
        profile_id=application_profile.component_profile_id,
        arm_side_by_component=arm_components,
        gripper_name_by_component=gripper_components,
        arm_reference_frame_by_component={
            component_id: f"{side}_arm_base"
            for component_id, side in arm_components.items()
        },
        arm_controlled_frame_by_component=dict.fromkeys(
            arm_components,
            "tool_tcp",
        ),
    )
    runtime_rig_holder: dict[str, Any] = {}
    support_sides = tuple(
        side for side in ("left", "right") if side not in output_sides
    )

    def capture_support_arm_pose() -> Mapping[str, Any]:
        rig = runtime_rig_holder.get("rig")
        if rig is None:
            raise RuntimeError(
                "MuJoCo support-pose capture requested before target binding."
            )
        return rig.capture_scene(
            lambda scene: {
                side: capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                    scene,
                    prepared.task_config,
                )[side]
                for side in support_sides
            }
        )

    hand_specs_by_component = {
        f"{side}_gripper": HandActionSpec.dmtac_gripper(side)
        for side in output_sides
    }
    materializer = MujocoComponentActionRobotActionV1Materializer(
        prepared.scene,
        component_profile,
        hand_action_spec_by_component=hand_specs_by_component,
        support_arm_pose_provider=capture_support_arm_pose,
        support_arm_sides=support_sides,
        support_pose_contract_id=(
            "mujoco.measured_support_tool_tcp_for_policy_skill_coupled_ik.v1"
        ),
    )
    component_id_by_effector = {
        **{f"{side}_arm": f"{side}_arm" for side in output_sides},
        **{f"{side}_hand": f"{side}_gripper" for side in output_sides},
    }
    ingress = MujocoRobotActionV1ComponentIngressAdapter(
        prepared.scene,
        component_profile,
        component_id_by_effector=component_id_by_effector,
    )
    execution_binding = build_mujoco_component_execution_binding(
        component_profile,
        materializer,
        policy_action_ingress=ingress,
    )
    target = build_mujoco_policy_skill_motion_target(
        config=target_config,
        component_execution_binding=execution_binding,
        prepared_target=prepared,
        execution_observation_sink=execution_observation_sink,
        trace_sink=trace_sink,
        tcp_pose_envelope_runtime=tcp_pose_envelope_runtime,
    )
    runtime_rig_holder["rig"] = target.runtime.rig
    if tcp_pose_envelope_runtime is not None:
        captured_at = float(target.runtime.clock())
        initial_pose_by_side = target.runtime.rig.capture_scene(
            lambda scene: {
                side: capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                    scene,
                    prepared.task_config,
                )[side]
                for side in output_sides
            }
        )
        initialization_hash = canonical_content_hash(
            {
                "task_initialization_info": dict(
                    prepared.task_initialization_info
                ),
                "measured_tool_tcp_by_side": initial_pose_by_side,
                "captured_at": captured_at,
                "clock_domain_id": target_config.clock_domain_id,
            },
            domain="dexteleop.mujoco.policy_skill_qpos_initialization.v1",
        )
        tcp_pose_envelope_runtime.commit_reference(
            reference_id=(
                "mujoco-policy-skill-qpos-init-reference:"
                + initialization_hash
            ),
            tcp_pose_by_side=initial_pose_by_side,
            captured_at_by_side=dict.fromkeys(output_sides, captured_at),
            source_event_ids=(
                "mujoco-qpos-initialization-completed:" + initialization_hash,
            ),
            lower_state="configured",
        )
    source = require_policy_emission_source(
        policy_source_builder(
            SimHILPolicySourceBuildContext(
                scene=target.scene,
                task_config=target.task_config,
                task_config_path=target_config.task_config_path,
                target_runtime=target.runtime,
                task_initialization_info=dict(target.task_initialization_info),
                downstream_target_lead_sec=(
                    target.runtime.reference_activation_spec.target_lead_time_sec
                ),
                scene_capture=target.runtime.rig.capture_scene,
            )
        )
    )
    return MujocoPolicySkillMotionApplication(
        target=target,
        policy_source=source,
    )


__all__ = [
    "MujocoPolicySkillApplicationProfile",
    "build_mujoco_policy_skill_application",
]
