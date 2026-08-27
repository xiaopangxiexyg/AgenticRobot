"""Wired B601-RS adapter and commissioning measurements.

The package is safe to import without ``motorbridge`` and does not open CAN or
send a device command until an explicitly motion-authorized caller invokes one
of the backend command methods.
"""

from .backend import (
    B601CommandReceipt,
    B601HardwareDependencyError,
    B601HardwareError,
    B601HardwareState,
    B601JointConfig,
    B601JointState,
    B601ModeReceipt,
    B601MotionAuthorizationError,
    B601RSBackend,
    B601RSBackendConfig,
    B601_RS_DEFAULT_CHANNEL,
    B601_RS_DEFAULT_FEEDBACK_TIMEOUT_SEC,
    B601_RS_DEFAULT_RATE_HZ,
    B601_RS_DEFAULT_WATCHDOG_TIMEOUT_SEC,
    B601_RS_FEEDBACK_FRAME,
    B601_RS_POSITION_FRAME,
)
from robot_motion.rebot_b601_rs import B601_RS_ARM_JOINT_NAMES
from .frequency import FrequencyLedger, FrequencyStats, summarize_timestamps

__all__ = [
    "B601CommandReceipt",
    "B601HardwareDependencyError",
    "B601HardwareError",
    "B601HardwareState",
    "B601JointConfig",
    "B601JointState",
    "B601ModeReceipt",
    "B601MotionAuthorizationError",
    "B601RSBackend",
    "B601RSBackendConfig",
    "B601_RS_DEFAULT_CHANNEL",
    "B601_RS_DEFAULT_FEEDBACK_TIMEOUT_SEC",
    "B601_RS_DEFAULT_RATE_HZ",
    "B601_RS_DEFAULT_WATCHDOG_TIMEOUT_SEC",
    "B601_RS_ARM_JOINT_NAMES",
    "B601_RS_FEEDBACK_FRAME",
    "B601_RS_POSITION_FRAME",
    "FrequencyLedger",
    "FrequencyStats",
    "summarize_timestamps",
]
