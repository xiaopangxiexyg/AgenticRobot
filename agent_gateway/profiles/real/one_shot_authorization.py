"""Compatibility imports for the persistent real primitive session authority.

The historic class name remains available so existing composition callers do
not need an atomic import migration. It no longer imposes a process-lifetime
one-shot limit; every matching request receives a distinct request-bound,
one-use Grant from :mod:`session_authorization`.
"""

from .session_authorization import (
    REAL_AGENT_PRIMITIVE_ACTIONS,
    REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
    REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
    RealAgentPrimitiveSessionAuthorization,
)

RealAgentPrimitiveOneShotAuthorization = RealAgentPrimitiveSessionAuthorization

__all__ = [
    "REAL_AGENT_PRIMITIVE_ACTIONS",
    "REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN",
    "REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN",
    "RealAgentPrimitiveOneShotAuthorization",
    "RealAgentPrimitiveSessionAuthorization",
]
