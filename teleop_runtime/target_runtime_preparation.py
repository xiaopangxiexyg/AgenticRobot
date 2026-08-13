"""Compatibility exports for target-runtime handle preparation.

The canonical implementation lives in :mod:`teleop_runtime.target_runtime`.
This module remains a stateless import shim for existing callers.
"""

from .target_runtime.preparation import (
    TargetRuntimeHandlePreparation,
    prepare_target_runtime_handle,
)

__all__ = [
    "TargetRuntimeHandlePreparation",
    "prepare_target_runtime_handle",
]
