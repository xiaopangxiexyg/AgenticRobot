"""RM65 embodiment package for simulation adapters, actuation, and evidence."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sapien_renderer import (
        DEFAULT_ARM_JOINT_NAMES,
        DebugBoxOverlay,
        DebugBoxTemplate,
        SapienArmHandRenderer,
    )

__all__ = [
    "DEFAULT_ARM_JOINT_NAMES",
    "DebugBoxOverlay",
    "DebugBoxTemplate",
    "SapienArmHandRenderer",
]


def __getattr__(name: str):
    """Keep actuator/evidence imports independent from SAPIEN preview code."""

    if name in set(__all__):
        from .sapien_renderer import (
            DEFAULT_ARM_JOINT_NAMES,
            DebugBoxOverlay,
            DebugBoxTemplate,
            SapienArmHandRenderer,
        )

        return {
            "DEFAULT_ARM_JOINT_NAMES": DEFAULT_ARM_JOINT_NAMES,
            "DebugBoxOverlay": DebugBoxOverlay,
            "DebugBoxTemplate": DebugBoxTemplate,
            "SapienArmHandRenderer": SapienArmHandRenderer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
