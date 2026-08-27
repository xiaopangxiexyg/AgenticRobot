"""Wired B601-RS adapter and commissioning measurements.

The package is safe to import without ``motorbridge`` and does not open CAN
until an explicit ``connect()`` call.  The default feedback observation is
receive-only; an active caller may explicitly request vendor feedback through
``poll_feedback(request_feedback=True)``.  Motion commands still require an
explicitly motion-authorized backend.
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
    B601_RS_MECH_POS_PARAM_ID,
    B601_RS_POS_VEL_LIMIT_SPD_PARAM_ID,
    B601_RS_POS_VEL_POS_KP_PARAM_ID,
    B601_RS_POS_VEL_VEL_KI_PARAM_ID,
    B601_RS_POS_VEL_VEL_KP_PARAM_ID,
    B601_RS_POSITION_FRAME,
)
from robot_motion.rebot_b601_rs import B601_RS_ARM_JOINT_NAMES
from .frequency import (
    FrequencyAcceptance,
    FrequencyLedger,
    FrequencyStats,
    evaluate_frequency_target,
    summarize_timestamps,
)

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
    "B601_RS_MECH_POS_PARAM_ID",
    "B601_RS_POS_VEL_LIMIT_SPD_PARAM_ID",
    "B601_RS_POS_VEL_POS_KP_PARAM_ID",
    "B601_RS_POS_VEL_VEL_KI_PARAM_ID",
    "B601_RS_POS_VEL_VEL_KP_PARAM_ID",
    "B601_RS_POSITION_FRAME",
    "FrequencyAcceptance",
    "FrequencyLedger",
    "FrequencyStats",
    "evaluate_frequency_target",
    "summarize_timestamps",
]
