"""Backend-neutral motion primitives.

Primitives in this package only construct motion intent.  They do not read
state, own authority, apply safety, solve IK, interpolate, or dispatch.
"""

from .tcp_displacement import (
    CARDINAL_TCP_SHORT_STEP_BY_DIRECTION,
    CARDINAL_TCP_SHORT_STEP_PRIMITIVES,
    SHORT_STEP_DISTANCE_M,
    CardinalTCPPoseDirection,
    TCPPoseDisplacementPrimitive,
    TCPPoseDisplacementTarget,
    cardinal_tcp_short_step,
)
from .tcp_orientation import (
    LEVEL_TCP_ORIENTATION_BY_HEADING,
    LEVEL_TCP_ORIENTATION_PRIMITIVES,
    TCP_FORWARD_AXIS,
    TCP_LEFT_AXIS,
    TCP_LEVEL_DOWN_AXIS,
    LevelTCPHeading,
    LevelTCPOrientationPrimitive,
    LevelTCPOrientationTarget,
    level_tcp_orientation,
)

__all__ = [
    "CARDINAL_TCP_SHORT_STEP_BY_DIRECTION",
    "CARDINAL_TCP_SHORT_STEP_PRIMITIVES",
    "LEVEL_TCP_ORIENTATION_BY_HEADING",
    "LEVEL_TCP_ORIENTATION_PRIMITIVES",
    "SHORT_STEP_DISTANCE_M",
    "TCP_FORWARD_AXIS",
    "TCP_LEFT_AXIS",
    "TCP_LEVEL_DOWN_AXIS",
    "CardinalTCPPoseDirection",
    "LevelTCPHeading",
    "LevelTCPOrientationPrimitive",
    "LevelTCPOrientationTarget",
    "TCPPoseDisplacementPrimitive",
    "TCPPoseDisplacementTarget",
    "cardinal_tcp_short_step",
    "level_tcp_orientation",
]
