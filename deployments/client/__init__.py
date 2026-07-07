"""Model-agnostic deployment client primitives."""

from .action import PolicyActionChunk, UnifiedActionChunk
from .contracts import (
    DEFAULT_ACTION_KEY,
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
from .runtime import (
    ActionPreparer,
    ClientRuntimeStepResult,
    CommandDispatcher,
    DeploymentClientRuntime,
    DeploymentObservationSource,
)

__all__ = [
    "ActionPreparer",
    "ClientRuntimeStepResult",
    "CommandDispatcher",
    "DEFAULT_ACTION_KEY",
    "DEXTELEOP_DEPLOY_SCHEMA_VERSION",
    "DeployPolicyContract",
    "DeploymentClientRuntime",
    "DeploymentObservationSource",
    "MetadataValidationResult",
    "ObservationAdapter",
    "PolicyActionChunk",
    "RemoteModelPolicy",
    "RemotePolicyConfig",
    "RemoteTransport",
    "ServerMetadataError",
    "UnifiedActionChunk",
    "validate_server_metadata",
]

