"""Legacy teleop action namespace.

Common action contracts live in robot_motion.action. This package keeps
teleop-specific adapters/replay helpers plus compatibility exports.
"""

from .adapters import (
    TeleopCommandActionAdapterConfig,
    matrix_to_xyzquat,
    robot_action_from_teleop_command,
)
from .codec import (
    EE_POSE_VECTOR_DIM,
    EE_POSE_HAND16_VECTOR_DIM,
    QPOS_VECTOR_DIM,
    QPOS_HAND16_VECTOR_DIM,
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
from .arm_spec import (
    ArmActionSpec,
    arm_action_specs_from_metadata,
    arm_action_specs_metadata,
    normalize_arm_action_specs,
)
from .hand_spec import (
    DEFAULT_HAND_SLOT_SEMANTICS,
    HandActionSpec,
    hand_action_specs_from_metadata,
    hand_action_specs_metadata,
    normalize_hand_action_specs,
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
from .teleop_source import TeleopRetargeterActionSource, TeleopRetargeterActionSourceConfig
from .human_proposal import (
    HUMAN_SOURCE_ADAPTER_CONTRACT_ID_V1,
    HumanCommandContribution,
    HumanCommandEmission,
    HumanCommandFilterObservation,
    HumanDispatchReceiptSnapshot,
    HumanEffectorCommandSnapshot,
    HumanLegacyCommandObservation,
    HumanLegacyActionAdapterResult,
    HumanProposalAdapter,
    HumanProposalAdapterConfig,
    HumanProposalAdapterResult,
    SourceAdapterReceipt,
    SourceComponentMappingReceipt,
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
from .human_position_shadow import (
    HUMAN_POSITION_BACKEND_BINDING_HASH_DOMAIN_V1,
    HumanPositionBackendBindingEvidence,
    bind_human_position_backend_evidence,
)
from .legacy_execution_context import (
    LEGACY_EXECUTION_CONTEXT_MODES_V1,
    LegacyControlDomainExecutionContext,
    LegacyControlDomainExecutionContextSnapshot,
    LegacyMotionDispatchPreparation,
    LegacySafetyDispatchPreparation,
)
from .rm65_candidate_safety_matrix import (
    M6_CANDIDATE_POLICY_ARM_QPOS_SAFETY_MATRIX_V1,
    build_m6_candidate_policy_arm_qpos_scoped_safety_matrix,
)
from .vector_pipeline import (
    VectorActionLayout,
    VectorRobotActionBuildResult,
    robot_action_from_vector_action,
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


def __getattr__(name: str):
    """Load replay-only adapters without polluting normal collection imports."""

    if name in _LAZY_REPLAY_PROPOSAL_EXPORTS:
        from . import replay_proposal

        return getattr(replay_proposal, name)
    if name in _LAZY_REPLAY_SHADOW_FRONTEND_EXPORTS:
        from . import replay_shadow_frontend

        return getattr(replay_shadow_frontend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionExecutionResult",
    "ActionExecutor",
    "ActionExecutorConfig",
    "ActionRecord",
    "ActionVectorEncoding",
    "ArmActionSpec",
    "BIMANUAL_EE_POSE_HAND16_V1",
    "BIMANUAL_EE_POSE_V1",
    "BIMANUAL_MIXED_HAND16_V1",
    "BIMANUAL_MIXED_V1",
    "BIMANUAL_QPOS_HAND16_V1",
    "BIMANUAL_QPOS_V1",
    "CallableActionSource",
    "EE_POSE_VECTOR_DIM",
    "EndEffectorTarget",
    "EE_POSE_HAND16_VECTOR_DIM",
    "HAND16_SLOT_DOF",
    "DEFAULT_HAND_SLOT_SEMANTICS",
    "DatasetReplayProposalShadowAdapter",
    "DatasetReplayProposalShadowConfig",
    "ReplayGuiShadowBinding",
    "ReplayGuiShadowCloseReport",
    "ReplayGuiShadowConsumer",
    "ReplayGuiShadowProfile",
    "HandActionSpec",
    "HUMAN_LEGACY_SAFETY_BINDING_HASH_DOMAIN_V1",
    "HUMAN_LEGACY_SAFETY_COMPARISON_HASH_DOMAIN_V1",
    "HUMAN_POSITION_BACKEND_BINDING_HASH_DOMAIN_V1",
    "HUMAN_SOURCE_ADAPTER_CONTRACT_ID_V1",
    "HumanCommandContribution",
    "HumanCommandEmission",
    "HumanCommandFilterObservation",
    "HumanDispatchReceiptSnapshot",
    "HumanEffectorCommandSnapshot",
    "HumanLegacyCommandObservation",
    "HumanLegacyActionAdapterResult",
    "HumanLegacySafetyBindingEvidence",
    "HumanLegacySafetyComparison",
    "HumanProposalAdapter",
    "HumanProposalAdapterConfig",
    "HumanProposalAdapterResult",
    "HumanPositionBackendBindingEvidence",
    "INSPIRE_HAND_DOF",
    "LEGACY_EXECUTION_CONTEXT_MODES_V1",
    "LegacyControlDomainExecutionContext",
    "LegacyControlDomainExecutionContextSnapshot",
    "LegacyMotionDispatchPreparation",
    "LegacySafetyDispatchPreparation",
    "M6_CANDIDATE_POLICY_ARM_QPOS_SAFETY_MATRIX_V1",
    "NullCommandDispatcher",
    "QPOS_VECTOR_DIM",
    "QPOS_HAND16_VECTOR_DIM",
    "ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION",
    "ROBOT_ACTION_VECTOR_LAYOUT_VERSION",
    "RigCommandDispatcher",
    "RobotAction",
    "RobotActionResolver",
    "RobotActionSource",
    "StaticActionSource",
    "SourceAdapterReceipt",
    "SourceComponentMappingReceipt",
    "TeleopCommandActionAdapterConfig",
    "TeleopRetargeterActionSource",
    "TeleopRetargeterActionSourceConfig",
    "VectorActionLayout",
    "VectorRobotActionBuildResult",
    "default_arm_frame",
    "default_arm_controlled_frame",
    "decode_robot_action",
    "encode_robot_action",
    "effector_name",
    "evaluate_human_legacy_aligned_safety",
    "hand_action_specs_metadata",
    "hand_action_specs_from_metadata",
    "arm_action_specs_metadata",
    "arm_action_specs_from_metadata",
    "bind_human_legacy_safety_evidence",
    "bind_human_position_backend_evidence",
    "build_human_legacy_aligned_safety_profile",
    "build_m6_candidate_policy_arm_qpos_scoped_safety_matrix",
    "matrix_to_xyzquat",
    "normalize_side",
    "normalize_arm_action_specs",
    "normalize_hand_action_specs",
    "robot_action_from_teleop_command",
    "robot_action_from_vector_action",
    "replay_adapter_debug_payload",
    "split_effector_name",
    "xyzquat_to_matrix",
]
