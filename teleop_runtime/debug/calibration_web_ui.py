"""Compatibility wrapper for the Control Station browser GUI.

The implementation lives in :mod:`control_station.web.calibration_web_ui`.
This module keeps older debug entrypoints and tests import-compatible.
"""

from control_station.web.calibration_web_ui import (
    CalibrationWebUi,
    CalibrationWebUiState,
    WebUiKeyInput,
)

__all__ = [
    "CalibrationWebUi",
    "CalibrationWebUiState",
    "WebUiKeyInput",
]
