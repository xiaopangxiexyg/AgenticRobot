"""Compatibility import for the control-mode-neutral TCP pose envelope.

The safety semantics are owned by :mod:`robot_motion.target_safety`.  Keep
this module temporarily so existing callers can migrate without changing
runtime behavior or persisted profile semantics.
"""

from robot_motion.target_safety.tcp_pose_envelope import *  # noqa: F403
from robot_motion.target_safety.tcp_pose_envelope import __all__
