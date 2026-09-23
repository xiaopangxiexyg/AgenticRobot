"""Collection gripper profile loading and policy normalization.

These helpers resolve task-profile and CLI values into the collection gripper
policy shape.  They may read an explicitly configured JSON profile, but they
do not construct a backend, inspect hardware, start a worker, or dispatch a
command.  Backend assignment loading and tactile config mutation remain in
the collection composition facade.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..backends.grippers import (
    DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
    GRIPPER_ACTION_MAX,
)
from ..gripper_policy_core import (
    load_gripper_task_profile as _load_gripper_task_profile,
)
from ..gripper_policy_core import (
    normalize_gripper_policy,
    overlay_gripper_policy,
    profile_value,
)
from ..retargeters import DEFAULT_GRIPPER_INPUT_MAX, DEFAULT_GRIPPER_INPUT_MIN

GRIPPER_POLICY_DEFAULTS = {
    "source_dim": 2,
    "output_dim": 0,
    "invert": False,
    "alpha": 0.95,
    "input_min": DEFAULT_GRIPPER_INPUT_MIN,
    "input_max": DEFAULT_GRIPPER_INPUT_MAX,
    "speed": 0.2,
    "torque": 0.2,
    "max_action": GRIPPER_ACTION_MAX,
    "send_threshold": 0.002,
    "command_hz": DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    "continuous_grasp": True,
    "telemetry_hz": DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    "telemetry_command_grace_s": DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
}


def load_gripper_task_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Load the selected task profile without constructing a gripper backend."""

    return _load_gripper_task_profile(args)


def resolved_gripper_policy(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve profile and CLI overrides into the canonical policy mapping."""

    cached = getattr(args, "_resolved_gripper_policy", None)
    if isinstance(cached, dict):
        return dict(cached)

    profile = load_gripper_task_profile(args)
    policy = overlay_gripper_policy(args, GRIPPER_POLICY_DEFAULTS, profile)
    normalize_gripper_policy(policy, canonical_max_action=True)
    args._resolved_gripper_policy = dict(policy)
    return dict(policy)


def dmtac_tactile_enabled(args: argparse.Namespace) -> bool:
    """Resolve explicit DMTAC tactile enablement before collection defaults."""

    explicit = getattr(args, "dmtac_tactile", None)
    if explicit is not None:
        return bool(explicit)
    return bool(getattr(args, "with_tactile", True))


__all__ = [
    "GRIPPER_POLICY_DEFAULTS",
    "dmtac_tactile_enabled",
    "load_gripper_task_profile",
    "profile_value",
    "resolved_gripper_policy",
]
