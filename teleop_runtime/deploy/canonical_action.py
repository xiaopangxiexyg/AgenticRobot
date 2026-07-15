"""Compatibility re-export for legacy teleop_runtime.deploy.canonical_action imports.

New common deploy code should import from robot_motion.deploy.canonical_action directly.
"""

from __future__ import annotations

from robot_motion.deploy.canonical_action import (
    DEPLOY_VECTOR_ACTION_LAYOUT,
    CanonicalDeployActionPipeline,
    CanonicalDeployActionResult,
)

__all__ = [
    "CanonicalDeployActionPipeline",
    "CanonicalDeployActionResult",
    "DEPLOY_VECTOR_ACTION_LAYOUT",
]
