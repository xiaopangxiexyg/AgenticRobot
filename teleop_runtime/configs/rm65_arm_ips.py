"""Module-identical compatibility alias for the neutral RM65 config owner."""

import sys

from teleop_runtime.hardware_config import rm65_arm_ips as _canonical


sys.modules[__name__] = _canonical
