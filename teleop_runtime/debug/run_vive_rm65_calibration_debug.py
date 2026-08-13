"""Historic command shim for the Station-owned RM65 calibration application."""

from __future__ import annotations

import sys

from control_station import rm65_calibration_debug_application as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
