"""Compatibility imports for the former version-named action timing module."""

from .component_capture.action_timing import (
    ComponentActionTiming,
    attach_projected_component_action_timing,
    attach_projected_component_action_timings,
)


V25pActionComponentTiming = ComponentActionTiming

__all__ = [
    "V25pActionComponentTiming",
    "attach_projected_component_action_timing",
    "attach_projected_component_action_timings",
]
