"""Wired B601-RS adapter and commissioning measurements.

The package is safe to import without ``motorbridge`` and does not open CAN
until an explicit ``connect()`` call.  The default feedback observation is
receive-only; an active caller may explicitly request vendor feedback through
``poll_feedback(request_feedback=True)``.  Motion commands still require an
explicitly motion-authorized backend; active-report configuration uses its own
authorization and does not imply motion authorization.
"""

from robot_motion.rebot_b601_rs import B601_RS_ARM_JOINT_NAMES

from .active_report import (
    B601ActiveReportAcceptance,
    B601ActiveReportDeviceAcceptance,
    evaluate_active_report_capture,
)
from .backend import (
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
    B601_RS_TORQUE_LIMIT_MAX_NM,
    B601_RS_TORQUE_LIMIT_PARAM_ID,
    B601_RS_TORQUE_LIMIT_VERIFY_ABS_TOL_NM,
    B601ActiveReportAuthorizationError,
    B601ActiveReportReceipt,
    B601CommandReceipt,
    B601DriverTorqueLimitReceipt,
    B601HardwareDependencyError,
    B601HardwareError,
    B601HardwareState,
    B601JointConfig,
    B601JointState,
    B601ModeReceipt,
    B601MotionAuthorizationError,
    B601RSBackend,
    B601RSBackendConfig,
)
from .frequency import (
    FrequencyAcceptance,
    FrequencyLedger,
    FrequencyStats,
    evaluate_frequency_target,
    summarize_timestamps,
)
from .gripper import (
    B601_RS_GRIPPER_CLOCK_DOMAIN,
    B601_RS_GRIPPER_JOINT_NAME,
    B601_RS_GRIPPER_SOURCE_ID,
    B601RSGripperPositionObservation,
    B601RSGripperReader,
    B601RSGripperReaderConfig,
)
from .gripper_control import (
    B601RSGripperControlConfig,
    B601RSGripperController,
)
from .shared_control import (
    B601_SHARED_POLICY_FRAME,
    B601PolicyAssistCommand,
    B601PolicyCommandBuffer,
    B601PolicyCommandReceipt,
    B601PolicyCommandSample,
    B601SharedControlConfig,
    B601SharedControlMode,
    B601SharedControlOutput,
    B601SharedControlState,
    compute_b601_shared_control_step,
)

__all__ = [
    "B601_RS_ARM_JOINT_NAMES",
    "B601_RS_DEFAULT_CHANNEL",
    "B601_RS_DEFAULT_FEEDBACK_TIMEOUT_SEC",
    "B601_RS_DEFAULT_RATE_HZ",
    "B601_RS_DEFAULT_WATCHDOG_TIMEOUT_SEC",
    "B601_RS_FEEDBACK_FRAME",
    "B601_RS_GRIPPER_CLOCK_DOMAIN",
    "B601_RS_GRIPPER_JOINT_NAME",
    "B601_RS_GRIPPER_SOURCE_ID",
    "B601_RS_MECH_POS_PARAM_ID",
    "B601_RS_POSITION_FRAME",
    "B601_RS_POS_VEL_LIMIT_SPD_PARAM_ID",
    "B601_RS_POS_VEL_POS_KP_PARAM_ID",
    "B601_RS_POS_VEL_VEL_KI_PARAM_ID",
    "B601_RS_POS_VEL_VEL_KP_PARAM_ID",
    "B601_RS_TORQUE_LIMIT_MAX_NM",
    "B601_RS_TORQUE_LIMIT_PARAM_ID",
    "B601_RS_TORQUE_LIMIT_VERIFY_ABS_TOL_NM",
    "B601_SHARED_POLICY_FRAME",
    "B601ActiveReportAcceptance",
    "B601ActiveReportAuthorizationError",
    "B601ActiveReportDeviceAcceptance",
    "B601ActiveReportReceipt",
    "B601CommandReceipt",
    "B601DriverTorqueLimitReceipt",
    "B601HardwareDependencyError",
    "B601HardwareError",
    "B601HardwareState",
    "B601JointConfig",
    "B601JointState",
    "B601ModeReceipt",
    "B601MotionAuthorizationError",
    "B601PolicyAssistCommand",
    "B601PolicyCommandBuffer",
    "B601PolicyCommandReceipt",
    "B601PolicyCommandSample",
    "B601RSBackend",
    "B601RSBackendConfig",
    "B601RSGripperControlConfig",
    "B601RSGripperController",
    "B601RSGripperPositionObservation",
    "B601RSGripperReader",
    "B601RSGripperReaderConfig",
    "B601SharedControlConfig",
    "B601SharedControlMode",
    "B601SharedControlOutput",
    "B601SharedControlState",
    "FrequencyAcceptance",
    "FrequencyLedger",
    "FrequencyStats",
    "compute_b601_shared_control_step",
    "evaluate_active_report_capture",
    "evaluate_frequency_target",
    "summarize_timestamps",
]
