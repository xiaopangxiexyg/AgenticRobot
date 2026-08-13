"""Adapters connecting shared robot-motion contracts to MuJoCo execution."""

from .execution_cut import MujocoExecutionStateCut
from .demonstration_trace import (
    SAFE_ACTION_DEMONSTRATION_STAGE_V1,
    CommittedTraceDemonstrationSelection,
    ControlTraceDemonstrationError,
    DemonstrationSuccessEvidence,
    demonstration_from_committed_control_trace,
)
from .robot_motion import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from .reference_lineage import (
    MujocoPlannerCommandFrameObservation,
    MujocoReferenceTargetActivationObservation,
)

__all__ = [
    "SAFE_ACTION_DEMONSTRATION_STAGE_V1",
    "CommittedTraceDemonstrationSelection",
    "ControlTraceDemonstrationError",
    "DemonstrationSuccessEvidence",
    "MujocoPlannerCommandFrameObservation",
    "MujocoExecutionStateCut",
    "MujocoPositionExecutionConfig",
    "MujocoPositionExecutionRig",
    "MujocoReferenceTargetActivationObservation",
    "demonstration_from_committed_control_trace",
]
