"""Host-to-Quest messages owned by the visualization feature."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

try:
    from ...protocol import PROTOCOL_VERSION
except ImportError:  # pragma: no cover - direct module compatibility.
    from protocol import PROTOCOL_VERSION


FEATURE_NAME = "visualization"
FEATURE_PROTOCOL_VERSION = 2


def robot_tcp_message(robots: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "robot_tcp",
        "feature": FEATURE_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
        "server_timestamp": time.time(),
        "robots": dict(robots),
    }


def feedback_error_message(error: str) -> dict[str, Any]:
    return {
        "type": "robot_tcp",
        "feature": FEATURE_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
        "server_timestamp": time.time(),
        "robots": {},
        "source_error": str(error),
    }
