"""Compatibility classes for the shared calibration RM65 pose reader."""

from dataclasses import dataclass

from utils.calibration_common import robot as _shared

_CommonRM65PoseReader = _shared.RM65PoseReader
_CommonRobotPoseSample = _shared.RobotPoseSample
_finite_float_list = _shared._finite_float_list


@dataclass
class RobotPoseSample(_CommonRobotPoseSample):
    """Preserve the historical eye-to-hand sample type path."""


class RM65PoseReader(_CommonRM65PoseReader):
    """Use the shared reader while returning the historical sample type."""

    sample_type = RobotPoseSample
