"""Compatibility import for the shared RTC inference worker.

The implementation lives in :mod:`robot_motion.deploy.policy_worker`.  Keep
this module so existing deploy callers do not change while RTC ownership moves
out of the teleop/replay application package.
"""

from robot_motion.deploy.policy_worker import *  # noqa: F403
from robot_motion.deploy.policy_worker import __all__ as __all__
