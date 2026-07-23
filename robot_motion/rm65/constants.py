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

# Static mirror of the active canonical RM65 URDF arm-joint limits, in radians.
# The URDF is authoritative; the guard test in ``test_robot_motion_rm65_shared``
# intentionally fails if this transport-friendly tuple ever drifts from it.
# Keep the mirror in this sim/real-neutral package so keyframe safety and the
# final SDK boundary enforce the same envelope without importing a URDF parser
# in a control-loop module.
RM65_ABSOLUTE_Q_MIN_RAD = (
    -3.1,
    -1.57079632679,
    -2.355,
    -2.443460952792061,
    -2.233,
    -2.0943951023931953,
)
RM65_ABSOLUTE_Q_MAX_RAD = tuple(-value for value in RM65_ABSOLUTE_Q_MIN_RAD)
