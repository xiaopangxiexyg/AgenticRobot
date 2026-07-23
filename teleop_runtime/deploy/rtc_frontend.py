"""Compatibility import for the shared RTC chunk deployment coordinator.

The implementation lives in :mod:`robot_motion.deploy.rtc_runtime`.  This
module remains only as an import-stable application shim; it owns no RTC state,
worker, plan, model adapter, target backend, or Control Station integration.
"""

from robot_motion.deploy.rtc_runtime import *  # noqa: F403
from robot_motion.deploy.rtc_runtime import __all__ as __all__
