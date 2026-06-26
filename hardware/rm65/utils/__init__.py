"""RM65 utility package.

The user-facing calibration scripts and data live in this package. Legacy
interpolation helpers are re-exported here for compatibility with older code
that imported ``hardware.rm65.utils`` as a module.
"""

from hardware.rm65.motion_utils import (
    interpolate_6dpose,
    interpolate_vectors_constant_speed,
    interpolate_vectors_linear,
)

__all__ = [
    "interpolate_6dpose",
    "interpolate_vectors_constant_speed",
    "interpolate_vectors_linear",
]
