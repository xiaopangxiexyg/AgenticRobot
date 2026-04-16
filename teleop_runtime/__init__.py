"""Generic teleoperation runtime package."""

from .backends import (
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
    RM65ArmBackend,
    RM65ArmBackendConfig,
)
from .calibration import (
    ArmCalibration,
    DeskFrameConfig,
    HandCalibration,
    TeleopCalibration,
)
from .clutch import ClutchAnchor, ClutchState, DeltaClutchController
from .diagnostics import (
    FaultEvent,
    FaultSeverity,
    RateTracker,
    TeleopRunState,
    TeleopStateMachine,
)
from .recording import RecordingConfig, SessionRecorder
from .retargeters import (
    JointsToDexHandRetargeter,
    JointsToDexHandRetargeterConfig,
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
)
from .registry import (
    backend_registry,
    register_backend,
    register_retargeter,
    register_rig,
    register_source,
    retargeter_registry,
    rig_registry,
    source_registry,
)
from .safety import SafetyMonitor, SafetyViolation, TeleopSafetyConfig
from .schemas import EffectorCommand, ObservationChannel, TeleopCommand, TeleopObservation
from .session import SessionStepResult, TeleopSession, TeleopSessionConfig
from .sources.hand import HandQ6Source
from .sources.misc import KeyboardButtonSource
from .sources.pose import VivePoseSource, ViveTrackerConfig
from .configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    build_annotated_dual_rm65_calibration,
    build_bimanual_session,
    default_vive_tracker_configs,
)

__all__ = [
    "ANNOTATED_DUAL_RM65_BASE_DISTANCE_M",
    "ArmCalibration",
    "ClutchAnchor",
    "ClutchState",
    "DeltaClutchController",
    "DeskFrameConfig",
    "EffectorCommand",
    "FaultEvent",
    "FaultSeverity",
    "HandCalibration",
    "HandQ6Source",
    "InspireTacHandBackend",
    "InspireTacHandBackendConfig",
    "JointsToDexHandRetargeter",
    "JointsToDexHandRetargeterConfig",
    "KeyboardButtonSource",
    "ObservationChannel",
    "PoseToArmRetargeter",
    "PoseToArmRetargeterConfig",
    "RateTracker",
    "RecordingConfig",
    "RM65ArmBackend",
    "RM65ArmBackendConfig",
    "SafetyMonitor",
    "SafetyViolation",
    "SessionRecorder",
    "SessionStepResult",
    "TeleopCalibration",
    "TeleopCommand",
    "TeleopObservation",
    "TeleopRunState",
    "TeleopSafetyConfig",
    "TeleopSession",
    "TeleopSessionConfig",
    "TeleopStateMachine",
    "VivePoseSource",
    "ViveTrackerConfig",
    "backend_registry",
    "build_annotated_dual_rm65_calibration",
    "build_bimanual_session",
    "default_vive_tracker_configs",
    "register_backend",
    "register_retargeter",
    "register_rig",
    "register_source",
    "retargeter_registry",
    "rig_registry",
    "source_registry",
]
