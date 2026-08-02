"""Compatibility proxy for the Control Station browser UI.

New application wiring imports :mod:`control_station.web.calibration_web_ui`.
This module preserves every historic public and test/debug attribute while
remaining free of a second implementation or state owner.
"""

from __future__ import annotations

from control_station.web import calibration_web_ui as _implementation

CalibrationWebUi = _implementation.CalibrationWebUi
CalibrationWebUiState = _implementation.CalibrationWebUiState
WebUiKeyInput = _implementation.WebUiKeyInput

__all__ = ["CalibrationWebUi", "CalibrationWebUiState", "WebUiKeyInput"]


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
