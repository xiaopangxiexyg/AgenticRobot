"""Agent Gateway package discovery surface.

Public contracts, runtime mechanics, and transports have distinct owners.
Import them from :mod:`agent_gateway.protocol`,
:mod:`agent_gateway.runtime`, or
:mod:`agent_gateway.transport`; the package root deliberately
does not recreate a cross-layer facade.
"""

from . import protocol, runtime, transport

__all__ = ["protocol", "runtime", "transport"]
