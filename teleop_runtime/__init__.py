"""Generic teleoperation runtime package with lazy compatibility exports.

Importing a package namespace must not initialize vendor integrations.  The
historical top-level symbols remain available, but their owning modules load
only when a caller explicitly accesses the symbol.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "ANNOTATED_DUAL_RM65_BASE_DISTANCE_M",
    "ArmCalibration",
    "DeskFrameConfig",
    "ViveRetargetFrameConfig",
    "DMTacGripperBackend",
    "DMTacGripperBackendConfig",
    "DualArmCollisionAwareRetargeter",
    "DualArmCollisionAwareRetargeterConfig",
    "DualArmQposCollisionCheck",
    "EpisodeRecorder",
    "EpisodeRecorderConfig",
    "EpisodeStartSideCheck",
    "EpisodeValidationReport",
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
    "RealSenseRGBCameraConfig",
    "RealSenseRGBSource",
    "RecordingConfig",
    "RM65ArmBackend",
    "RM65ArmBackendConfig",
    "RM65ImpedanceArmBackend",
    "RM65ImpedanceArmBackendConfig",
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
    "build_bimanual_passive_session",
    "build_bimanual_session",
    "compare_start_qpos_pair",
    "default_vive_tracker_configs",
    "dmtac_backend_configs_from_assignment",
    "episode_start_qpos_pair",
    "load_dmtac_hardware_assignment",
    "load_episode",
    "load_latest_dmtac_backend_configs",
    "load_latest_q6_ports",
    "load_q6_hardware_assignment",
    "load_realsense_camera_config_map",
    "q6_ports_from_assignment",
    "require_valid_episode",
    "register_backend",
    "register_retargeter",
    "register_rig",
    "register_source",
    "retargeter_registry",
    "rig_registry",
    "source_registry",
    "validate_episode",
]


_LAZY_EXPORT_GROUPS = (
    (
        "teleop_runtime.backends",
        (
            "DMTacGripperBackend",
            "DMTacGripperBackendConfig",
            "InspireTacHandBackend",
            "InspireTacHandBackendConfig",
            "RM65ArmBackend",
            "RM65ArmBackendConfig",
            "RM65ImpedanceArmBackend",
            "RM65ImpedanceArmBackendConfig",
        ),
    ),
    (
        "teleop_runtime.calibration",
        (
            "ArmCalibration",
            "DeskFrameConfig",
            "ViveRetargetFrameConfig",
            "HandCalibration",
            "TeleopCalibration",
        ),
    ),
    (
        "teleop_runtime.diagnostics",
        (
            "FaultEvent",
            "FaultSeverity",
            "RateTracker",
            "TeleopRunState",
            "TeleopStateMachine",
        ),
    ),
    (
        "teleop_runtime.episodes",
        (
            "EpisodeRecorder",
            "EpisodeRecorderConfig",
            "EpisodeStartSideCheck",
            "EpisodeValidationReport",
            "compare_start_qpos_pair",
            "episode_start_qpos_pair",
            "load_episode",
            "require_valid_episode",
            "validate_episode",
        ),
    ),
    (
        "teleop_runtime.recording",
        ("RecordingConfig", "SessionRecorder"),
    ),
    (
        "teleop_runtime.retargeters",
        (
            "DualArmCollisionAwareRetargeter",
            "DualArmCollisionAwareRetargeterConfig",
            "DualArmQposCollisionCheck",
            "JointsToDexHandRetargeter",
            "JointsToDexHandRetargeterConfig",
            "PoseToArmRetargeter",
            "PoseToArmRetargeterConfig",
        ),
    ),
    (
        "teleop_runtime.registry",
        (
            "backend_registry",
            "register_backend",
            "register_retargeter",
            "register_rig",
            "register_source",
            "retargeter_registry",
            "rig_registry",
            "source_registry",
        ),
    ),
    (
        "teleop_runtime.safety",
        ("SafetyMonitor", "SafetyViolation", "TeleopSafetyConfig"),
    ),
    (
        "teleop_runtime.schemas",
        (
            "EffectorCommand",
            "ObservationChannel",
            "TeleopCommand",
            "TeleopObservation",
        ),
    ),
    (
        "teleop_runtime.session",
        ("SessionStepResult", "TeleopSession", "TeleopSessionConfig"),
    ),
    (
        "teleop_runtime.sources",
        (
            "RealSenseRGBCameraConfig",
            "RealSenseRGBSource",
            "HandQ6Source",
            "KeyboardButtonSource",
            "VivePoseSource",
            "ViveTrackerConfig",
        ),
    ),
    (
        "teleop_runtime.configs",
        (
            "ANNOTATED_DUAL_RM65_BASE_DISTANCE_M",
            "build_annotated_dual_rm65_calibration",
            "build_bimanual_passive_session",
            "build_bimanual_session",
            "default_vive_tracker_configs",
            "dmtac_backend_configs_from_assignment",
            "load_dmtac_hardware_assignment",
            "load_latest_dmtac_backend_configs",
            "load_latest_q6_ports",
            "load_q6_hardware_assignment",
            "load_realsense_camera_config_map",
            "q6_ports_from_assignment",
        ),
    ),
)
_LAZY_EXPORTS = {
    name: module_name
    for module_name, names in _LAZY_EXPORT_GROUPS
    for name in names
}

if set(_LAZY_EXPORTS) != set(__all__):
    raise RuntimeError("teleop_runtime lazy exports do not match __all__.")


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
