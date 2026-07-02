"""Compatibility wrapper for the runtime browser GUI.

The implementation lives in :mod:`teleop_runtime.gui.calibration_web_ui`.
This module keeps older debug entrypoints and tests import-compatible.
"""

from ..gui.calibration_web_ui import CalibrationWebUi, CalibrationWebUiState, WebUiKeyInput

__all__ = [
    "CalibrationWebUi",
    "CalibrationWebUiState",
    "WebUiKeyInput",
]
