"""Compatibility exports for the historical debug logging import path."""

from ..log_utils import (
    DEFAULT_LOG_HEARTBEAT_SEC,
    ChangeGate,
    log_event,
    log_stage,
)

__all__ = [
    "DEFAULT_LOG_HEARTBEAT_SEC",
    "ChangeGate",
    "log_event",
    "log_stage",
]
