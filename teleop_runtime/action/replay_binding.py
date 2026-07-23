"""Frontend-neutral builders for replay target binding evidence.

The builders consume explicit runtime contracts only.  They never inspect a
device, authorize motion, acquire a dispatch token, or call a backend.
"""

from __future__ import annotations

from robot_motion.replay.compatibility import (
    build_replay_binding_assessment,
    build_target_runtime_action_contract,
)


__all__ = [
    "build_replay_binding_assessment",
    "build_target_runtime_action_contract",
]
