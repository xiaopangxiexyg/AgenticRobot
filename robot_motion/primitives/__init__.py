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

__all__ = [
    "CARDINAL_TCP_SHORT_STEP_BY_DIRECTION",
    "CARDINAL_TCP_SHORT_STEP_PRIMITIVES",
    "SHORT_STEP_DISTANCE_M",
    "CardinalTCPPoseDirection",
    "TCPPoseDisplacementPrimitive",
    "TCPPoseDisplacementTarget",
    "cardinal_tcp_short_step",
]
