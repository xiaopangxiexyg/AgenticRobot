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
    GripperError,
    GripperState,
    InspireGripper,
    InspireGripperModbus,
    list_serial_devices,
)
from .tactile_filter import DistributedForceDenoiser, TactileDenoiseConfig

__all__ = [
    "DEFAULT_TACTILE_FIELDS",
    "TACTILE_COMPUTE_PROFILES",
    "TACTILE_FIELDS",
    "DMTacDevice",
    "DMTacGripperDevice",
    "DMTacMultiDevice",
    "DistributedForceDenoiser",
    "DistributedForceXYVisualizer",
    "DeviceConfig",
    "DeviceSnapshot",
    "ForceVizConfig",
    "GripperConfig",
    "GripperError",
    "GripperState",
    "InspireGripper",
    "InspireGripperModbus",
    "TacFrame",
    "TactileConfig",
    "TactileDenoiseConfig",
    "discover_device_candidates",
    "discover_device_ids",
    "discover_devices",
    "format_frame_summary",
    "format_snapshot_summary",
    "list_serial_devices",
    "normalize_tactile_compute_profile",
    "normalize_tactile_fields",
]
