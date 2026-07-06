from __future__ import annotations

from importlib import import_module
from typing import Any

from .safety import (
    RM65ForceGuardConfig,
    RM65ForceLimitExceeded,
    RM65ForceReading,
    check_rm65_force_guard,
    emergency_stop_rm65,
)

_LAZY_EXPORTS = {
    "CONFIRM_TEXT": (".qpos_execute", "CONFIRM_TEXT"),
    "RM65LiveQposInitConfig": (".qpos_execute", "RM65LiveQposInitConfig"),
    "execute_rm65_live_qpos_init": (".qpos_execute", "execute_rm65_live_qpos_init"),
    "read_current_qpos_by_side": (".qpos_execute", "read_current_qpos_by_side"),
    "send_rm65_qpos_plan": (".qpos_execute", "send_rm65_qpos_plan"),
    "RM65QposSequenceExecuteConfig": (
        ".qpos_sequence_execute",
        "RM65QposSequenceExecuteConfig",
    ),
    "execute_rm65_qpos_test_sequence": (
        ".qpos_sequence_execute",
        "execute_rm65_qpos_test_sequence",
    ),
}

__all__ = [
    "CONFIRM_TEXT",
    "RM65ForceGuardConfig",
    "RM65ForceLimitExceeded",
    "RM65ForceReading",
    "RM65LiveQposInitConfig",
    "RM65QposSequenceExecuteConfig",
    "check_rm65_force_guard",
    "emergency_stop_rm65",
    "execute_rm65_live_qpos_init",
    "execute_rm65_qpos_test_sequence",
    "read_current_qpos_by_side",
    "send_rm65_qpos_plan",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
