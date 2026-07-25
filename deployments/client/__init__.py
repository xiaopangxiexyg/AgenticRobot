"""Model-agnostic deployment client primitives."""

from .action import PolicyActionChunk, UnifiedActionChunk
from .artifact import deploy_contract_from_policy_artifact, load_policy_artifact
from .contracts import (
    DEFAULT_ACTION_KEY,
    DEXTELEOP_DEPLOY_SCHEMA_V2,
    DEXTELEOP_DEPLOY_SCHEMA_VERSION,
    DeployPolicyContract,
    MetadataValidationResult,
    ServerMetadataError,
    validate_server_metadata,
)
from .remote_policy import (
    ObservationAdapter,
    RemoteModelPolicy,
    RemotePolicyConfig,
    RemoteTransport,
)
from .policy import HoldPolicy, LoadedPolicy, load_policy
from .runtime import (
    ActionPreparer,
    ClientRuntimeStepResult,
    CommandDispatcher,
    DeploymentClientRuntime,
    DeploymentObservationSource,
)
from .rtc import (
    ModelAgnosticRtcClient,
    RtcClientInferenceBundle,
    RtcClientProtocolError,
    RtcModelBackend,
    RtcModelSpaceDecoder,
)
from .rtc_executor import (
    RtcModelClientInferenceExecutor,
    RtcModelInputPreparer,
    RtcPreparedModelInput,
    rtc_client_inference_bundle_debug_payload,
)
from .rtc_final_chunk import (
    FinalChunkModelOutput,
    FinalChunkRtcBackend,
    FinalChunkRtcBinding,
)
from .rtc_final_chunk_adapter import (
    MappingFinalChunkOutputAdapter,
    create_final_chunk_rtc_runtime_adapter,
)
from .rtc_frontend import (
    RtcModelPolicyIdentity,
    RtcModelRuntimeAdapter,
    SimpleRtcModelRuntimeAdapter,
    build_rtc_model_producer_runtime,
    build_rtc_model_shadow_frontend,
    rtc_producer_config_from_model_adapter,
    rtc_shadow_config_from_model_adapter,
)
from .rtc_transport import (
    RtcTransportDeadlineExceeded,
    RtcTransportDeadlineSpec,
    RtcTransportPhaseReceipt,
)

__all__ = [
    "ActionPreparer",
    "ClientRuntimeStepResult",
    "CommandDispatcher",
    "DEFAULT_ACTION_KEY",
    "DEXTELEOP_DEPLOY_SCHEMA_VERSION",
    "DEXTELEOP_DEPLOY_SCHEMA_V2",
    "DeployPolicyContract",
    "DeploymentClientRuntime",
    "DeploymentObservationSource",
    "FinalChunkModelOutput",
    "FinalChunkRtcBackend",
    "FinalChunkRtcBinding",
    "MappingFinalChunkOutputAdapter",
    "MetadataValidationResult",
    "ModelAgnosticRtcClient",
    "ObservationAdapter",
    "PolicyActionChunk",
    "RemoteModelPolicy",
    "RemotePolicyConfig",
    "RemoteTransport",
    "RtcClientInferenceBundle",
    "RtcClientProtocolError",
    "RtcModelClientInferenceExecutor",
    "RtcModelBackend",
    "RtcModelInputPreparer",
    "RtcModelPolicyIdentity",
    "RtcModelRuntimeAdapter",
    "RtcModelSpaceDecoder",
    "RtcPreparedModelInput",
    "rtc_client_inference_bundle_debug_payload",
    "SimpleRtcModelRuntimeAdapter",
    "RtcTransportDeadlineExceeded",
    "RtcTransportDeadlineSpec",
    "RtcTransportPhaseReceipt",
    "HoldPolicy",
    "LoadedPolicy",
    "ServerMetadataError",
    "UnifiedActionChunk",
    "validate_server_metadata",
    "build_rtc_model_producer_runtime",
    "build_rtc_model_shadow_frontend",
    "create_final_chunk_rtc_runtime_adapter",
    "deploy_contract_from_policy_artifact",
    "load_policy_artifact",
    "load_policy",
    "rtc_producer_config_from_model_adapter",
    "rtc_shadow_config_from_model_adapter",
]
