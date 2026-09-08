"""Unified Inspire gripper + DM-Tac tactile sensor package."""

from ._local_paths import ensure_local_paths

ensure_local_paths()

from portable_dm_tac import (
    DMTacDevice,
    DMTacMultiDevice,
    DeviceConfig,
    TacFrame,
    discover_device_candidates,
    discover_device_ids,
    discover_devices,
    format_frame_summary,
)

from .device import (
    DEFAULT_TACTILE_FIELDS,
    TACTILE_COMPUTE_PROFILES,
    TACTILE_FIELDS,
    DMTacGripperDevice,
    DeviceSnapshot,
    GripperConfig,
    TactileConfig,
    format_snapshot_summary,
    normalize_tactile_compute_profile,
    normalize_tactile_fields,
)
from .force_visualizer import DistributedForceXYVisualizer, ForceVizConfig
from .gripper_driver import (
    GRIPPER_ERROR_BITS,
    GripperError,
    GripperState,
    InspireGripper,
    InspireGripperModbus,
    decode_gripper_error_code,
    describe_gripper_error_code,
    list_serial_devices,
)
from .tactile_filter import DistributedForceDenoiser, TactileDenoiseConfig
from .tactile_calibration import (
    CALIBRATION_POLICIES,
    TactileCalibration,
    build_stable_base_frame,
    default_calibration_dir,
    load_tactile_calibration,
    save_tactile_calibration,
)

__all__ = [
    "DEFAULT_TACTILE_FIELDS",
    "TACTILE_COMPUTE_PROFILES",
    "TACTILE_FIELDS",
    "CALIBRATION_POLICIES",
    "DMTacDevice",
    "DMTacGripperDevice",
    "DMTacMultiDevice",
    "DistributedForceDenoiser",
    "DistributedForceXYVisualizer",
    "DeviceConfig",
    "DeviceSnapshot",
    "ForceVizConfig",
    "GRIPPER_ERROR_BITS",
    "GripperConfig",
    "GripperError",
    "GripperState",
    "InspireGripper",
    "InspireGripperModbus",
    "decode_gripper_error_code",
    "describe_gripper_error_code",
    "TacFrame",
    "TactileConfig",
    "TactileCalibration",
    "build_stable_base_frame",
    "TactileDenoiseConfig",
    "discover_device_candidates",
    "discover_device_ids",
    "discover_devices",
    "default_calibration_dir",
    "format_frame_summary",
    "format_snapshot_summary",
    "list_serial_devices",
    "normalize_tactile_compute_profile",
    "normalize_tactile_fields",
    "load_tactile_calibration",
    "save_tactile_calibration",
]
