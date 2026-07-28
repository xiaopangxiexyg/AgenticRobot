"""Control-station source adapters.

These adapters terminate at an existing typed source-emission seam.  They do
not own an Arbiter, action queue, dispatcher, device, or episode lifecycle.
"""

from .synthetic_human import (
    SYNTHETIC_HUMAN_POSE_GENERIC_SOURCE_CONTRACT_ID_V1,
    SYNTHETIC_HUMAN_POSE_SOURCE_CONTRACT_ID_V1,
    SYNTHETIC_HUMAN_QPOS_SOURCE_CONTRACT_ID_V1,
    SyntheticHumanPoseSource,
    SyntheticHumanPoseSourceSpec,
    SyntheticHumanQposSource,
    SyntheticHumanQposSourceSpec,
)
from .human import (
    HUMAN_TAKEOVER_READINESS_CANDIDATE_HASH_DOMAIN_V1,
    HumanObservationCutAdmissionCore,
    HumanProposalAdmissionCore,
    HumanPoseSourceContract,
    HumanTakeoverReadinessCandidate,
    HumanTakeoverReadinessProvider,
    PassiveHumanCommandEmissionIngress,
    PassiveHumanCommandEmissionIngressStatus,
    PassiveHumanCommandObserverFanout,
    admit_human_command_emission,
    admit_human_observation_cut,
)
from .teleop_pose import (
    PASSIVE_TELEOP_COMPONENT_SOURCE_CONTRACT_ID_V1,
    PASSIVE_TELEOP_POSE_GENERIC_SOURCE_CONTRACT_ID_V1,
    PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1,
    PassiveTeleopComponentBindingReceipt,
    PassiveTeleopComponentClockBridge,
    PassiveTeleopComponentSessionSource,
    PassiveTeleopComponentSourceSpec,
    PassiveTeleopComponentSpec,
    PassiveTeleopComponentStep,
    PassiveTeleopPoseBindingReceipt,
    PassiveTeleopPoseSessionSource,
    PassiveTeleopPoseSourceSpec,
    PassiveTeleopPoseStep,
    build_passive_teleop_component_session_source,
    build_passive_teleop_pose_session_source,
)
from .replay_gui import (
    ReplayGuiStationConsumer,
    ReplayGuiStationConsumerStatus,
)
from .replay import ReplaySourceObserverBinding
from .policy import (
    POLICY_COMPOSITION_INPUT_BINDING_HASH_DOMAIN_V1,
    PolicyCompositionInputAdapter,
    PolicyCompositionInputBinding,
    SyncPolicyEmissionBinding,
)
from .rtc_policy import (
    RtcPolicyEmissionBinding,
    RtcPolicyStationConsumer,
    RtcPolicyStationConsumerStatus,
)

__all__ = [
    "SYNTHETIC_HUMAN_QPOS_SOURCE_CONTRACT_ID_V1",
    "SYNTHETIC_HUMAN_POSE_GENERIC_SOURCE_CONTRACT_ID_V1",
    "SYNTHETIC_HUMAN_POSE_SOURCE_CONTRACT_ID_V1",
    "HUMAN_TAKEOVER_READINESS_CANDIDATE_HASH_DOMAIN_V1",
    "PASSIVE_TELEOP_COMPONENT_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_GENERIC_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1",
    "HumanPoseSourceContract",
    "HumanObservationCutAdmissionCore",
    "HumanProposalAdmissionCore",
    "HumanTakeoverReadinessCandidate",
    "HumanTakeoverReadinessProvider",
    "PassiveHumanCommandEmissionIngress",
    "PassiveHumanCommandEmissionIngressStatus",
    "PassiveHumanCommandObserverFanout",
    "admit_human_command_emission",
    "admit_human_observation_cut",
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
    "ReplayGuiStationConsumer",
    "ReplayGuiStationConsumerStatus",
    "ReplaySourceObserverBinding",
    "POLICY_COMPOSITION_INPUT_BINDING_HASH_DOMAIN_V1",
    "PolicyCompositionInputAdapter",
    "PolicyCompositionInputBinding",
    "SyncPolicyEmissionBinding",
    "RtcPolicyStationConsumer",
    "RtcPolicyStationConsumerStatus",
    "RtcPolicyEmissionBinding",
    "SyntheticHumanQposSource",
    "SyntheticHumanQposSourceSpec",
    "SyntheticHumanPoseSource",
    "SyntheticHumanPoseSourceSpec",
]
