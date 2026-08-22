"""Versioned core messages shared by the Quest development host and client."""

from __future__ import annotations

import time
from typing import Any

PROTOCOL_VERSION = 2


def server_hello(*, server_instance_id: str) -> dict[str, Any]:
    return {
        "type": "server_hello",
        "protocol_version": PROTOCOL_VERSION,
        "server_timestamp": time.time(),
        "mode": "quest_dev",
        "server_instance_id": str(server_instance_id),
        "authentication_required": True,
    }


def client_auth_result(
    *,
    authenticated: bool,
    server_instance_id: str,
    client_id: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build the only pre-authentication response accepted by the browser."""
    payload: dict[str, Any] = {
        "type": "client_auth_result",
        "protocol_version": PROTOCOL_VERSION,
        "server_timestamp": time.time(),
        "server_instance_id": str(server_instance_id),
        "authenticated": bool(authenticated),
    }
    if client_id is not None:
        payload["client_id"] = int(client_id)
    if error is not None:
        payload["error"] = str(error)
    return payload
