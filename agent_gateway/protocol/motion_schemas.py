"""Compatibility re-export for motion Skill-specific schemas.

New code imports :mod:`agent_gateway.skills.motion.schemas`.  This module keeps
the former canonical path stable for external consumers and reconciliation
tooling during migration.
"""

from agent_gateway.skills.motion.schemas import *  # noqa: F401,F403
from agent_gateway.skills.motion.schemas import __all__
