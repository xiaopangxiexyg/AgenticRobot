"""Historic command shim for the Station-owned RealTeleop application."""

from __future__ import annotations

import sys

from control_station import real_teleop_entrypoint as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
