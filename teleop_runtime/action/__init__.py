"""Legacy teleop action namespace.

Common action contracts live in robot_motion.action. This package keeps
teleop-specific adapters/replay helpers plus compatibility exports.
"""

from .adapters import (
    TeleopCommandActionAdapterConfig,
    matrix_to_xyzquat,
    robot_action_from_teleop_command,
)
from .arm_spec import (
    ArmActionSpec,
    arm_action_specs_from_metadata,
    arm_action_specs_metadata,
    normalize_arm_action_specs,
)
from .codec import (
    EE_POSE_HAND16_VECTOR_DIM,
    EE_POSE_VECTOR_DIM,
    QPOS_HAND16_VECTOR_DIM,
    QPOS_VECTOR_DIM,
    ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION,
    ROBOT_ACTION_VECTOR_LAYOUT_VERSION,
    ActionVectorEncoding,
    decode_robot_action,
    encode_robot_action,
)
from .executor import (
    ActionExecutor,
    ActionExecutorConfig,
    NullCommandDispatcher,
    RigCommandDispatcher,
    RobotActionResolver,
    xyzquat_to_matrix,
)
from .hand_spec import (
    DEFAULT_HAND_SLOT_SEMANTICS,
    HandActionSpec,
    hand_action_specs_from_metadata,
    hand_action_specs_metadata,
    normalize_hand_action_specs,
)
from .human_joint_residual import (
    HUMAN_JOINT_POSITION_RESIDUAL_ALGORITHM_DEVELOPMENT,
    HumanJointPositionResidualAdapter,
    HumanJointPositionResidualAnchor,
    HumanJointPositionResidualAnchorEntry,
    HumanJointPositionResidualReceipt,
    HumanJointPositionResidualResult,
    HumanJointPositionSample,
)
from .human_observer import (
    HumanPoseSourceContract,
    PassiveHumanCommandEmissionIngress,
    PassiveHumanCommandEmissionIngressStatus,
    PassiveHumanCommandObserverFanout,
)
from .human_pose_source import (
    HumanPoseChannelAdaptationReceipt,
    HumanPoseChannelAdaptationResult,
    HumanPoseChannelAdapter,
    HumanPoseChannelAdapterConfig,
    HumanPoseChannelBinding,
)
from .human_position_shadow import (
    HUMAN_POSITION_BACKEND_BINDING_HASH_DOMAIN_V1,
    HumanPositionBackendBindingEvidence,
    bind_human_position_backend_evidence,
)
from .human_proposal import (
    HUMAN_SOURCE_ADAPTER_CONTRACT_ID_V1,
    HumanCommandContribution,
    HumanCommandEmission,
    HumanCommandFilterObservation,
    HumanDispatchReceiptSnapshot,
    HumanEffectorCommandSnapshot,
    HumanLegacyActionAdapterResult,
    HumanLegacyCommandObservation,
    HumanProposalAdapter,
    HumanProposalAdapterConfig,
    HumanProposalAdapterResult,
    SourceAdapterReceipt,
    SourceComponentMappingReceipt,
)
from .human_residual import (
    HUMAN_RESIDUAL_POSE_ALGORITHM_DEVELOPMENT,
    HumanPoseSample,
    HumanResidualPoseAdaptationReceipt,
    HumanResidualPoseAdaptationResult,
    HumanResidualPoseAdapter,
    HumanResidualPoseAnchor,
    HumanResidualPoseAnchorEntry,
)
from .human_safety_shadow import (
    HUMAN_LEGACY_SAFETY_BINDING_HASH_DOMAIN_V1,
    HUMAN_LEGACY_SAFETY_COMPARISON_HASH_DOMAIN_V1,
    HumanLegacySafetyBindingEvidence,
    HumanLegacySafetyComparison,
    bind_human_legacy_safety_evidence,
    build_human_legacy_aligned_safety_profile,
    evaluate_human_legacy_aligned_safety,
)
from .legacy_execution_context import (
    LEGACY_CONTROL_LIFECYCLE_GATE_COMPATIBILITY_V1,
    LEGACY_CONTROL_LIFECYCLE_STATES_V1,
    LEGACY_EXECUTION_CONTEXT_MODES_V1,
    LegacyControlDomainExecutionContext,
    LegacyControlDomainExecutionContextSnapshot,
    LegacyControlOwnerStateTransitionOutcome,
    LegacyControlOwnerStateTransitionReceipt,
    LegacyControlOwnerStateTransitionRequest,
    LegacyMotionDispatchPreparation,
    LegacySafetyDispatchPreparation,
)
from .schema import (
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_MIXED_HAND16_V1,
    BIMANUAL_MIXED_V1,
    BIMANUAL_QPOS_HAND16_V1,
    BIMANUAL_QPOS_V1,
    HAND16_SLOT_DOF,
    INSPIRE_HAND_DOF,
    ActionExecutionResult,
    ActionRecord,
    EndEffectorTarget,
    RobotAction,
    default_arm_controlled_frame,
    default_arm_frame,
    effector_name,
    normalize_side,
    split_effector_name,
)
from .sources import CallableActionSource, RobotActionSource, StaticActionSource
from .synthetic_human import (
    SYNTHETIC_HUMAN_POSE_GENERIC_SOURCE_CONTRACT_ID_V1,
    SYNTHETIC_HUMAN_POSE_SOURCE_CONTRACT_ID_V1,
    SYNTHETIC_HUMAN_QPOS_SOURCE_CONTRACT_ID_V1,
    SyntheticHumanPoseSource,
    SyntheticHumanPoseSourceSpec,
    SyntheticHumanQposSource,
    SyntheticHumanQposSourceSpec,
)
from .teleop_source import (
    TeleopRetargeterActionSource,
    TeleopRetargeterActionSourceConfig,
)
from .vector_pipeline import (
    VectorActionLayout,
    VectorRobotActionBuildResult,
    robot_action_from_vector_action,
)
from .vive_residual import (
    ViveHumanResidualCoordinateBinding,
    build_vive_human_residual_coordinate_binding,
)

_LAZY_REPLAY_PROPOSAL_EXPORTS = {
    "DatasetReplayProposalShadowAdapter",
    "DatasetReplayProposalShadowConfig",
    "replay_adapter_debug_payload",
}
_LAZY_REPLAY_SHADOW_FRONTEND_EXPORTS = {
    "ReplayGuiShadowBinding",
    "ReplayGuiShadowCloseReport",
    "ReplayGuiShadowConsumer",
    "ReplayGuiShadowProfile",
}
_LAZY_HUMAN_ADMISSION_EXPORTS = {
    "HumanObservationCutAdmissionCore",
    "HumanProposalAdmissionCore",
    "admit_human_command_emission",
    "admit_human_observation_cut",
}
_LAZY_HUMAN_COMPOSITION_EXPORTS = {
    "HumanComponentActionAdaptation",
    "HumanCompositionInputAdaptation",
    "adapt_human_emission_to_component_action",
    "adapt_human_emission_to_composition_input",
}
_LAZY_HUMAN_COMPONENT_RESIDUAL_EXPORTS = {
    "HUMAN_COMPONENT_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT",
    "HumanComponentResidualAdaptationReceipt",
    "HumanComponentResidualAdapter",
    "HumanComponentResidualAnchor",
    "HumanComponentResidualResult",
}
_LAZY_PASSIVE_TELEOP_EXPORTS = {
    "AttachablePassiveTeleopComponentSource",
    "PASSIVE_TELEOP_COMPONENT_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_GENERIC_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1",
    "PassiveTeleopComponentBindingReceipt",
    "PassiveTeleopComponentClockBridge",
    "PassiveTeleopComponentSessionSource",
    "PassiveTeleopComponentSourceSpec",
    "PassiveTeleopComponentSpec",
    "PassiveTeleopComponentStep",
    "PassiveTeleopPoseBindingReceipt",
    "PassiveTeleopPoseSessionSource",
    "PassiveTeleopPoseSourceSpec",
    "PassiveTeleopPoseStep",
    "build_passive_teleop_component_session_source",
    "build_passive_teleop_pose_session_source",
}
_LAZY_RM65_CANDIDATE_SAFETY_EXPORTS = {
    "M6_CANDIDATE_POLICY_ARM_QPOS_SAFETY_MATRIX_V1",
    "build_m6_candidate_policy_arm_qpos_scoped_safety_matrix",
}


def __getattr__(name: str):
    """Load dependency-heavy adapters without polluting collection imports."""

    if name in _LAZY_HUMAN_ADMISSION_EXPORTS:
        from . import human_admission

        return getattr(human_admission, name)
    if name in _LAZY_HUMAN_COMPOSITION_EXPORTS:
        from . import human_composition_input

        return getattr(human_composition_input, name)
    if name in _LAZY_HUMAN_COMPONENT_RESIDUAL_EXPORTS:
        from . import human_component_residual

        return getattr(human_component_residual, name)
    if name in _LAZY_PASSIVE_TELEOP_EXPORTS:
        from . import passive_teleop

        return getattr(passive_teleop, name)
    if name in _LAZY_RM65_CANDIDATE_SAFETY_EXPORTS:
        from . import rm65_candidate_safety_matrix

        return getattr(rm65_candidate_safety_matrix, name)

    if name in _LAZY_REPLAY_PROPOSAL_EXPORTS:
        from . import replay_proposal

        return getattr(replay_proposal, name)
    if name in _LAZY_REPLAY_SHADOW_FRONTEND_EXPORTS:
        from . import replay_shadow_frontend

        return getattr(replay_shadow_frontend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BIMANUAL_EE_POSE_HAND16_V1",
    "BIMANUAL_EE_POSE_V1",
    "BIMANUAL_MIXED_HAND16_V1",
    "BIMANUAL_MIXED_V1",
    "BIMANUAL_QPOS_HAND16_V1",
    "BIMANUAL_QPOS_V1",
    "DEFAULT_HAND_SLOT_SEMANTICS",
    "EE_POSE_HAND16_VECTOR_DIM",
    "EE_POSE_VECTOR_DIM",
    "HAND16_SLOT_DOF",
    "HUMAN_COMPONENT_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_JOINT_POSITION_RESIDUAL_ALGORITHM_DEVELOPMENT",
    "HUMAN_LEGACY_SAFETY_BINDING_HASH_DOMAIN_V1",
    "HUMAN_LEGACY_SAFETY_COMPARISON_HASH_DOMAIN_V1",
    "HUMAN_POSITION_BACKEND_BINDING_HASH_DOMAIN_V1",
    "HUMAN_RESIDUAL_POSE_ALGORITHM_DEVELOPMENT",
    "HUMAN_SOURCE_ADAPTER_CONTRACT_ID_V1",
    "INSPIRE_HAND_DOF",
    "LEGACY_CONTROL_LIFECYCLE_GATE_COMPATIBILITY_V1",
    "LEGACY_CONTROL_LIFECYCLE_STATES_V1",
    "LEGACY_EXECUTION_CONTEXT_MODES_V1",
    "M6_CANDIDATE_POLICY_ARM_QPOS_SAFETY_MATRIX_V1",
    "PASSIVE_TELEOP_COMPONENT_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_GENERIC_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1",
    "QPOS_HAND16_VECTOR_DIM",
    "QPOS_VECTOR_DIM",
    "ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION",
    "ROBOT_ACTION_VECTOR_LAYOUT_VERSION",
    "SYNTHETIC_HUMAN_POSE_GENERIC_SOURCE_CONTRACT_ID_V1",
    "SYNTHETIC_HUMAN_POSE_SOURCE_CONTRACT_ID_V1",
    "SYNTHETIC_HUMAN_QPOS_SOURCE_CONTRACT_ID_V1",
    "ActionExecutionResult",
    "ActionExecutor",
    "ActionExecutorConfig",
    "ActionRecord",
    "ActionVectorEncoding",
    "ArmActionSpec",
    "AttachablePassiveTeleopComponentSource",
    "CallableActionSource",
    "DatasetReplayProposalShadowAdapter",
    "DatasetReplayProposalShadowConfig",
    "EndEffectorTarget",
    "HandActionSpec",
    "HumanCommandContribution",
    "HumanCommandEmission",
    "HumanCommandFilterObservation",
    "HumanComponentActionAdaptation",
    "HumanComponentResidualAdaptationReceipt",
    "HumanComponentResidualAdapter",
    "HumanComponentResidualAnchor",
    "HumanComponentResidualResult",
    "HumanCompositionInputAdaptation",
    "HumanDispatchReceiptSnapshot",
    "HumanEffectorCommandSnapshot",
    "HumanJointPositionResidualAdapter",
    "HumanJointPositionResidualAnchor",
    "HumanJointPositionResidualAnchorEntry",
    "HumanJointPositionResidualReceipt",
    "HumanJointPositionResidualResult",
    "HumanJointPositionSample",
    "HumanLegacyActionAdapterResult",
    "HumanLegacyCommandObservation",
    "HumanLegacySafetyBindingEvidence",
    "HumanLegacySafetyComparison",
    "HumanObservationCutAdmissionCore",
    "HumanPoseChannelAdaptationReceipt",
    "HumanPoseChannelAdaptationResult",
    "HumanPoseChannelAdapter",
    "HumanPoseChannelAdapterConfig",
    "HumanPoseChannelBinding",
    "HumanPoseSample",
    "HumanPoseSourceContract",
    "HumanPositionBackendBindingEvidence",
    "HumanProposalAdapter",
    "HumanProposalAdapterConfig",
    "HumanProposalAdapterResult",
    "HumanProposalAdmissionCore",
    "HumanResidualPoseAdaptationReceipt",
    "HumanResidualPoseAdaptationResult",
    "HumanResidualPoseAdapter",
    "HumanResidualPoseAnchor",
    "HumanResidualPoseAnchorEntry",
    "LegacyControlDomainExecutionContext",
    "LegacyControlDomainExecutionContextSnapshot",
    "LegacyControlOwnerStateTransitionOutcome",
    "LegacyControlOwnerStateTransitionReceipt",
    "LegacyControlOwnerStateTransitionRequest",
    "LegacyMotionDispatchPreparation",
    "LegacySafetyDispatchPreparation",
    "NullCommandDispatcher",
    "PassiveHumanCommandEmissionIngress",
    "PassiveHumanCommandEmissionIngressStatus",
    "PassiveHumanCommandObserverFanout",
    "PassiveTeleopComponentBindingReceipt",
    "PassiveTeleopComponentClockBridge",
    "PassiveTeleopComponentSessionSource",
    "PassiveTeleopComponentSourceSpec",
    "PassiveTeleopComponentSpec",
    "PassiveTeleopComponentStep",
    "PassiveTeleopPoseBindingReceipt",
    "PassiveTeleopPoseSessionSource",
    "PassiveTeleopPoseSourceSpec",
    "PassiveTeleopPoseStep",
    "ReplayGuiShadowBinding",
    "ReplayGuiShadowCloseReport",
    "ReplayGuiShadowConsumer",
    "ReplayGuiShadowProfile",
    "RigCommandDispatcher",
    "RobotAction",
    "RobotActionResolver",
    "RobotActionSource",
    "SourceAdapterReceipt",
    "SourceComponentMappingReceipt",
    "StaticActionSource",
    "SyntheticHumanPoseSource",
    "SyntheticHumanPoseSourceSpec",
    "SyntheticHumanQposSource",
    "SyntheticHumanQposSourceSpec",
    "TeleopCommandActionAdapterConfig",
    "TeleopRetargeterActionSource",
    "TeleopRetargeterActionSourceConfig",
    "VectorActionLayout",
    "VectorRobotActionBuildResult",
    "ViveHumanResidualCoordinateBinding",
    "adapt_human_emission_to_component_action",
    "adapt_human_emission_to_composition_input",
    "admit_human_command_emission",
    "admit_human_observation_cut",
    "arm_action_specs_from_metadata",
    "arm_action_specs_metadata",
    "bind_human_legacy_safety_evidence",
    "bind_human_position_backend_evidence",
    "build_human_legacy_aligned_safety_profile",
    "build_m6_candidate_policy_arm_qpos_scoped_safety_matrix",
    "build_passive_teleop_component_session_source",
    "build_passive_teleop_pose_session_source",
    "build_vive_human_residual_coordinate_binding",
    "decode_robot_action",
    "default_arm_controlled_frame",
    "default_arm_frame",
    "effector_name",
    "encode_robot_action",
    "evaluate_human_legacy_aligned_safety",
    "hand_action_specs_from_metadata",
    "hand_action_specs_metadata",
    "matrix_to_xyzquat",
    "normalize_arm_action_specs",
    "normalize_hand_action_specs",
    "normalize_side",
    "replay_adapter_debug_payload",
    "robot_action_from_teleop_command",
    "robot_action_from_vector_action",
    "split_effector_name",
    "xyzquat_to_matrix",
]
