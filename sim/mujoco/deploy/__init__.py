"""Deployable MuJoCo frontend using the shared authoritative control core."""

from importlib import import_module

from .contact_trace import (
    MUJOCO_CONTACT_GUARD_ID_V1,
    project_mujoco_forbidden_contact_guard_result,
)
from .observation import MujocoPolicyObservationAdapter
from .deployment_observation import (
    MujocoDeploymentObservationAdapter,
    MujocoDeploymentObservationConfig,
)
from .measurement import (
    MUJOCO_QPOS_MEASUREMENT_HASH_DOMAIN_V1,
    MujocoQposMeasurement,
    build_mujoco_qpos_measurement_from_execution_cut,
    capture_mujoco_qpos_measurement,
)
from .execution_pose_observation import (
    MUJOCO_FK_POSE_MEASUREMENT_STAGE_ID,
    MUJOCO_FK_POSE_REFERENCE_STAGE_ID,
    project_mujoco_fk_pose_measurement_observation,
    project_mujoco_fk_pose_reference_observation,
)
from .pose_kinematics import (
    MujocoPoseKinematicsBinding,
    MujocoPoseObservationProjector,
    build_mujoco_pose_observation_projector,
)
from .component_action_profile import (
    MujocoComponentActionRobotActionV1Materializer,
    MujocoComponentTargetProfile,
    MujocoRobotActionV1ComponentIngressAdapter,
    build_mujoco_component_execution_binding,
    build_mujoco_pose_component_target_profile,
    build_mujoco_pose_qpos_component_target_profile,
    build_rm65_right_cube_lift_component_target_profile,
)
from .pose_resolution import (
    MujocoCoupledPoseToQposResolver,
    MujocoPoseToQposResolution,
    POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1,
    POSE_TO_QPOS_RESOLVER_CONTRACT_ID_V1,
    build_mujoco_pose_qpos_execution_target,
)
from .sim_hil_target import (
    MujocoSimHILTarget,
    build_mujoco_sim_hil_target,
)
from .target_config import MujocoSimHILTargetConfig
from .target_preparation import (
    MujocoSimHILPreparedTarget,
    prepare_mujoco_sim_hil_target,
)
from .target_measurement import (
    build_mujoco_sim_hil_coordinate_profiles,
    build_mujoco_sim_hil_pose_observation_projector,
    capture_mujoco_sim_hil_safety_anchor,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)
from .agent_skill_target import (
    MujocoAgentSkillMotionTarget,
    build_mujoco_agent_skill_motion_target,
)
from .resolved_reference_receiver import (
    MujocoScheduledResolvedQposReferenceReceiver,
    ResolvedQposReferenceAdmissionResult,
)
from .rtc_policy import (
    DeterministicVirtualPolicyRtcRunner,
    VirtualBimanualQposPolicy,
    VirtualBimanualQposPolicySpec,
    VirtualPolicyCompletionRelease,
    VirtualPolicyRequestSubmission,
    VirtualRtcDecisionResult,
    VirtualRtcPlanInvalidationReceipt,
)
from .runtime import (
    build_mujoco_pose_action_schema,
    MujocoComponentExecutionBinding,
    MujocoDeployConfig,
    MujocoDeployRuntime,
    MujocoDirectComponentProposalSource,
    MujocoPolicyActionChunkSubmission,
    MujocoPolicyEmissionSubmission,
    MujocoReplaySubmission,
)
from .policy_rollout import (
    DecodedPolicy,
    DeployRuntime,
    DeploymentObservationSource,
    MujocoPolicyCycleResult,
    MujocoPolicyRollout,
    MujocoPolicyRolloutConfig,
    MujocoPolicyRolloutError,
)


_LAZY_EXPORTS = {
    "MujocoCompositionRuntimeConfig": (
        ".composition_binding",
        "MujocoCompositionRuntimeConfig",
    ),
    **{
        name: (".execution_difference", name)
        for name in (
            "MUJOCO_QPOS_EXECUTION_DIFFERENCE_CAPABILITY_ID",
            "MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_HASH",
            "MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_ID",
            "MujocoExecutionDifferenceObserver",
            "MujocoExecutionDifferenceSnapshot",
            "build_mujoco_qpos_execution_difference_observation",
            "project_mujoco_execution_measurement_observation",
            "project_mujoco_immediate_execution_reference_sample",
            "project_mujoco_execution_reference_frame",
            "project_mujoco_execution_reference_sample",
        )
    },
}


def __getattr__(name: str):
    """Load opt-in composition integrations only when requested."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "build_mujoco_pose_action_schema",
    "MujocoDeployConfig",
    "MujocoCompositionRuntimeConfig",
    "MujocoComponentExecutionBinding",
    "MujocoDeployRuntime",
    "MujocoDirectComponentProposalSource",
    "MujocoPolicyActionChunkSubmission",
    "MujocoPolicyEmissionSubmission",
    "MujocoReplaySubmission",
    "MujocoPolicyObservationAdapter",
    "MujocoDeploymentObservationAdapter",
    "MujocoDeploymentObservationConfig",
    "MUJOCO_QPOS_MEASUREMENT_HASH_DOMAIN_V1",
    "MujocoQposMeasurement",
    "build_mujoco_qpos_measurement_from_execution_cut",
    "capture_mujoco_qpos_measurement",
    "MUJOCO_FK_POSE_MEASUREMENT_STAGE_ID",
    "MUJOCO_FK_POSE_REFERENCE_STAGE_ID",
    "MujocoPoseKinematicsBinding",
    "MujocoPoseObservationProjector",
    "build_mujoco_pose_observation_projector",
    "project_mujoco_fk_pose_measurement_observation",
    "project_mujoco_fk_pose_reference_observation",
    "MUJOCO_QPOS_EXECUTION_DIFFERENCE_CAPABILITY_ID",
    "MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_HASH",
    "MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_ID",
    "MujocoExecutionDifferenceObserver",
    "MujocoExecutionDifferenceSnapshot",
    "build_mujoco_qpos_execution_difference_observation",
    "project_mujoco_execution_measurement_observation",
    "project_mujoco_immediate_execution_reference_sample",
    "project_mujoco_execution_reference_frame",
    "project_mujoco_execution_reference_sample",
    "MujocoCoupledPoseToQposResolver",
    "MujocoPoseToQposResolution",
    "MujocoScheduledResolvedQposReferenceReceiver",
    "ResolvedQposReferenceAdmissionResult",
    "POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1",
    "POSE_TO_QPOS_RESOLVER_CONTRACT_ID_V1",
    "build_mujoco_pose_qpos_execution_target",
    "MujocoSimHILTarget",
    "MujocoAgentSkillMotionTarget",
    "MujocoSimHILTargetConfig",
    "MujocoSimHILPreparedTarget",
    "build_mujoco_sim_hil_coordinate_profiles",
    "build_mujoco_sim_hil_pose_observation_projector",
    "build_mujoco_sim_hil_target",
    "capture_mujoco_sim_hil_safety_anchor",
    "capture_mujoco_sim_hil_tool_tcp_pose_by_side",
    "prepare_mujoco_sim_hil_target",
    "DeterministicVirtualPolicyRtcRunner",
    "VirtualBimanualQposPolicy",
    "VirtualBimanualQposPolicySpec",
    "VirtualPolicyCompletionRelease",
    "VirtualPolicyRequestSubmission",
    "VirtualRtcDecisionResult",
    "VirtualRtcPlanInvalidationReceipt",
    "DecodedPolicy",
    "DeployRuntime",
    "DeploymentObservationSource",
    "MujocoPolicyCycleResult",
    "MujocoPolicyRollout",
    "MujocoPolicyRolloutConfig",
    "MujocoPolicyRolloutError",
    "MujocoComponentTargetProfile",
    "MujocoComponentActionRobotActionV1Materializer",
    "MujocoRobotActionV1ComponentIngressAdapter",
    "build_mujoco_component_execution_binding",
    "build_mujoco_agent_skill_motion_target",
    "build_mujoco_pose_component_target_profile",
    "build_mujoco_pose_qpos_component_target_profile",
    "build_rm65_right_cube_lift_component_target_profile",
    "MUJOCO_CONTACT_GUARD_ID_V1",
    "project_mujoco_forbidden_contact_guard_result",
]
