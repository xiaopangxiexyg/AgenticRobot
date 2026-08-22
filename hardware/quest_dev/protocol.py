"""Versioned core messages shared by the Quest development host and client."""

from __future__ import annotations

import time
from typing import Any

PROTOCOL_VERSION = 1


def server_hello() -> dict[str, Any]:
    return {
        "type": "server_hello",
        "protocol_version": PROTOCOL_VERSION,
        "server_timestamp": time.time(),
        "mode": "quest_dev",
    }
