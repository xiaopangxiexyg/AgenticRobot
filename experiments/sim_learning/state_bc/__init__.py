"""State-only behavior-cloning and exclusive HITL experiment helpers.

The package is intentionally offline and non-authoritative.  It owns the
reproducible selector, artifact, and source-lineage probes used before a
policy is connected to the existing simulator deployment runtime.
"""

from .contracts import (
    STATE_BC_ACTION_DIM,
    STATE_BC_ACTION_MODE,
    STATE_BC_ACCEPTED_LAYOUT_HASHES,
    STATE_BC_GRIPPER_SEMANTICS,
    STATE_BC_LAYOUT_HASH_V1,
    STATE_BC_MODEL_SCHEMA_V1,
    STATE_BC_POSE_FRAME,
    STATE_BC_POSE_REPRESENTATION,
    STATE_BC_RUNTIME_LAYOUT_HASH_V1,
    STATE_BC_SCHEMA_V1,
    STATE_BC_STATE_DIM,
    StateBCArtifactMetadata,
    StateBCNormalization,
    StateBCSelector,
    validate_action_vector,
    validate_state_vector,
)
from .hitl import (
    HITL_EVENT_ACTION,
    HITL_EVENT_ACTIVATION,
    HITL_EVENT_HOLD,
    HITL_EVENT_RTC_NO_ACTION,
    HITL_SOURCE_HUMAN,
    HITL_SOURCE_POLICY,
    HITL_SOURCE_RTC,
    StateBCHITLEvent,
    StateBCHITLRunner,
    StateBCHITLTrace,
)
from .model import StateBCModel, fit_linear_state_bc
from .mujoco_stream import (
    StateBCMujocoActionReceipt,
    StateBCMujocoStreamBridge,
    StateBCMujocoTransitionReceipt,
)
from .stream import (
    STREAM_RUNTIME_ACTION_DIM,
    STREAM_RUNTIME_ACTION_LAYOUT,
    STREAM_RUNTIME_ACTION_SPACE,
    StateBCStreamEvent,
    StateBCStreamTrace,
    StateBCStreamingRunner,
)

__all__ = [
    "HITL_EVENT_ACTION",
    "HITL_EVENT_ACTIVATION",
    "HITL_EVENT_HOLD",
    "HITL_EVENT_RTC_NO_ACTION",
    "HITL_SOURCE_HUMAN",
    "HITL_SOURCE_POLICY",
    "HITL_SOURCE_RTC",
    "STATE_BC_ACTION_DIM",
    "STATE_BC_ACTION_MODE",
    "STATE_BC_ACCEPTED_LAYOUT_HASHES",
    "STATE_BC_GRIPPER_SEMANTICS",
    "STATE_BC_LAYOUT_HASH_V1",
    "STATE_BC_MODEL_SCHEMA_V1",
    "STATE_BC_POSE_FRAME",
    "STATE_BC_POSE_REPRESENTATION",
    "STATE_BC_RUNTIME_LAYOUT_HASH_V1",
    "STATE_BC_SCHEMA_V1",
    "STATE_BC_STATE_DIM",
    "StateBCArtifactMetadata",
    "StateBCModel",
    "StateBCMujocoActionReceipt",
    "StateBCMujocoStreamBridge",
    "StateBCMujocoTransitionReceipt",
    "StateBCNormalization",
    "StateBCSelector",
    "StateBCHITLEvent",
    "StateBCHITLRunner",
    "StateBCHITLTrace",
    "StateBCStreamEvent",
    "StateBCStreamTrace",
    "StateBCStreamingRunner",
    "STREAM_RUNTIME_ACTION_DIM",
    "STREAM_RUNTIME_ACTION_LAYOUT",
    "STREAM_RUNTIME_ACTION_SPACE",
    "fit_linear_state_bc",
    "validate_action_vector",
    "validate_state_vector",
]
