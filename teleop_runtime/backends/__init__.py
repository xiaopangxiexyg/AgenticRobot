"""Backend compatibility exports loaded only on explicit access."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DMTacGripperBackend",
    "DMTacGripperBackendConfig",
    "InspireTacHandBackend",
    "InspireTacHandBackendConfig",
    "RM65ArmBackend",
    "RM65ArmBackendConfig",
    "RM65ImpedanceArmBackend",
    "RM65ImpedanceArmBackendConfig",
]

_LAZY_EXPORTS = {
    "DMTacGripperBackend": "teleop_runtime.backends.grippers.dmtac",
    "DMTacGripperBackendConfig": "teleop_runtime.backends.grippers.dmtac",
    "InspireTacHandBackend": "teleop_runtime.backends.hands.inspire_tac",
    "InspireTacHandBackendConfig": "teleop_runtime.backends.hands.inspire_tac",
    "RM65ArmBackend": "teleop_runtime.backends.arms.rm65",
    "RM65ArmBackendConfig": "teleop_runtime.backends.arms.rm65",
    "RM65ImpedanceArmBackend": "teleop_runtime.backends.arms.rm65_impedance",
    "RM65ImpedanceArmBackendConfig": (
        "teleop_runtime.backends.arms.rm65_impedance"
    ),
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
