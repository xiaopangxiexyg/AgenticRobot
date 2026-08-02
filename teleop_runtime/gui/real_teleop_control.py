"""Compatibility module alias for the RealTeleop entity-side adapter."""

from __future__ import annotations

import sys

from ..real_teleop import operator_control_runtime as _implementation

# Preserve historical imports and monkeypatch targets as the exact same module
# object.  A proxy object would silently break patches of module-level support
# functions used by offline diagnostics and compatibility tests.
sys.modules[__name__] = _implementation
