"""Agent Gateway package discovery surface.

Public contracts, runtime mechanics, and transports have distinct owners.
Import them from :mod:`control_station.agent_gateway.protocol`,
:mod:`control_station.agent_gateway.runtime`, or
:mod:`control_station.agent_gateway.transport`; the package root deliberately
does not recreate a cross-layer facade.
"""

from . import protocol, runtime, transport

__all__ = ["protocol", "runtime", "transport"]
