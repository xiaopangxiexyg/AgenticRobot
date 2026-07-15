from __future__ import annotations

ARM_DOF = 6
LEFT = "left"
RIGHT = "right"
SIDES = (LEFT, RIGHT)

QPOS_UNIT = "rad"
POSE6_UNITS = ("m", "m", "m", "rad", "rad", "rad")

BARE_ENDPOINT_FRAME = "bare_endpoint"
ARM_ENDPOINT_FRAME = "arm_endpoint"
TOOL_TCP_FRAME = "tool_tcp"
POSITION_ENDPOINT_PROFILE_NAME = "position_endpoint"

DEFAULT_COLLISION_SCENE = "dual_arm_hand_proxy_open"

# Canonical RM65 URDF/hardware joint envelope in radians.  Keep this in the
# sim/real-neutral robot package so 15 Hz keyframe guards and the final real
# transport boundary enforce the same absolute limits.
RM65_ABSOLUTE_Q_MIN_RAD = (-3.1, -1.57079632679, -2.355, -3.1, -2.233, -6.28)
RM65_ABSOLUTE_Q_MAX_RAD = tuple(-value for value in RM65_ABSOLUTE_Q_MIN_RAD)
