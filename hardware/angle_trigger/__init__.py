"""Side-keyed USB angle-trigger calibration and reading."""

from .angle_trigger import (
    AngleTriggerReader,
    DeferredAngleTriggerReader,
    main,
)

__all__ = ("AngleTriggerReader", "DeferredAngleTriggerReader", "main")
