"""Collection gripper profile loading and policy normalization.

These helpers resolve task-profile and CLI values into the collection gripper
policy shape.  They may read an explicitly configured JSON profile, but they
do not construct a backend, inspect hardware, start a worker, or dispatch a
command.  Backend assignment loading and tactile config mutation remain in
the collection composition facade.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..backends.grippers import (
    DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
    GRIPPER_ACTION_MAX,
    normalize_dmtac_tactile_fields,
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

    path_text = str(getattr(args, "gripper_task_profile", "") or "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(  # noqa: TRY004 - preserve the legacy ValueError contract.
            f"Gripper task profile {path} must contain a JSON object."
        )
    base_gripper = payload.get("gripper")
    if base_gripper is not None and not isinstance(base_gripper, Mapping):
        raise ValueError(f"Profile {path} field 'gripper' must be an object.")
    selected: Mapping[str, Any] | None = None
    tasks = payload.get("tasks")
    if isinstance(tasks, Mapping):
        task_keys = (
            str(getattr(args, "gripper_task", "") or ""),
            str(getattr(args, "task_id", "") or ""),
            str(getattr(args, "task_name", "") or ""),
            str(getattr(args, "task", "") or ""),
        )
        for key in task_keys:
            item = tasks.get(key)
            if isinstance(item, Mapping):
                selected = item
                args._resolved_gripper_task = key
                break
    if selected is not None:
        gripper = selected.get("gripper")
        if gripper is not None and not isinstance(gripper, Mapping):
            raise ValueError(
                f"Profile {path} selected task field 'gripper' must be an object."
            )
        merged = dict(base_gripper) if isinstance(base_gripper, Mapping) else {}
        merged.update(dict(gripper) if isinstance(gripper, Mapping) else dict(selected))
        return merged
    if isinstance(base_gripper, Mapping):
        return dict(base_gripper)
    if "gripper" in payload:
        raise ValueError(f"Profile {path} field 'gripper' must be an object.")
    return dict(payload)


def profile_value(profile: Mapping[str, Any], *keys: str) -> Any:
    """Return the first configured profile key, preserving legacy precedence."""

    for key in keys:
        if key in profile:
            return profile[key]
    return None


def resolved_gripper_policy(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve profile and CLI overrides into the canonical policy mapping."""

    cached = getattr(args, "_resolved_gripper_policy", None)
    if isinstance(cached, dict):
        return dict(cached)

    profile = load_gripper_task_profile(args)
    policy: dict[str, Any] = dict(GRIPPER_POLICY_DEFAULTS)
    profile_keys = {
        "source_dim": ("source_dim", "q6_dim"),
        "output_dim": ("output_dim", "gripper_slot"),
        "invert": ("invert",),
        "alpha": ("alpha", "command_alpha"),
        "input_min": ("input_min", "q6_min", "source_min"),
        "input_max": ("input_max", "q6_max", "source_max"),
        "speed": ("speed",),
        "torque": ("torque", "force"),
        "max_action": ("max_action", "max_close"),
        "send_threshold": ("send_threshold",),
        "command_hz": ("command_hz", "max_command_hz"),
        "continuous_grasp": ("continuous_grasp", "continuous"),
        "telemetry_hz": ("telemetry_hz",),
        "telemetry_command_grace_s": ("telemetry_command_grace_s",),
        "tactile_fields": ("tactile_fields", "dmtac_tactile_fields"),
        "tactile_compute_profile": (
            "tactile_compute_profile",
            "dmtac_tactile_compute_profile",
        ),
    }
    for key, aliases in profile_keys.items():
        value = profile_value(profile, *aliases)
        if value is not None:
            policy[key] = value

    cli_overrides = {
        "source_dim": getattr(args, "gripper_source_dim", None),
        "output_dim": getattr(args, "gripper_output_dim", None),
        "invert": getattr(args, "gripper_invert", None),
        "alpha": getattr(args, "gripper_alpha", None),
        "input_min": getattr(args, "gripper_input_min", None),
        "input_max": getattr(args, "gripper_input_max", None),
        "speed": getattr(args, "gripper_speed", None),
        "torque": getattr(args, "gripper_torque", None),
        "max_action": getattr(args, "gripper_max_action", None),
        "send_threshold": getattr(args, "gripper_send_threshold", None),
        "command_hz": getattr(args, "gripper_command_hz", None),
        "continuous_grasp": getattr(args, "gripper_continuous_grasp", None),
        "telemetry_hz": getattr(args, "gripper_telemetry_hz", None),
        "telemetry_command_grace_s": getattr(
            args, "gripper_telemetry_command_grace_s", None
        ),
    }
    for key, value in cli_overrides.items():
        if value is not None:
            policy[key] = value

    policy["source_dim"] = int(policy["source_dim"])
    policy["output_dim"] = int(policy["output_dim"])
    policy["invert"] = bool(policy["invert"])
    policy["alpha"] = float(policy["alpha"])
    policy["input_min"] = float(policy["input_min"])
    policy["input_max"] = float(policy["input_max"])
    policy["speed"] = float(policy["speed"])
    policy["torque"] = float(policy["torque"])
    # Do not let old 0.9/0.95 task profiles create another policy action space.
    policy["max_action"] = float(GRIPPER_ACTION_MAX)
    policy["send_threshold"] = float(policy["send_threshold"])
    policy["command_hz"] = float(policy["command_hz"])
    policy["continuous_grasp"] = bool(policy["continuous_grasp"])
    policy["telemetry_hz"] = float(policy["telemetry_hz"])
    policy["telemetry_command_grace_s"] = float(
        policy["telemetry_command_grace_s"]
    )
    if policy.get("tactile_fields") is not None:
        policy["tactile_fields"] = normalize_dmtac_tactile_fields(
            policy["tactile_fields"]
        )
    if policy.get("tactile_compute_profile") is not None:
        policy["tactile_compute_profile"] = str(policy["tactile_compute_profile"])
    if not 0 <= int(policy["source_dim"]) < 6:
        raise ValueError("gripper source_dim must be in [0, 5].")
    if int(policy["output_dim"]) != 0:
        raise ValueError("dmtac-gripper output_dim/gripper_slot must be 0.")
    if not 0.0 < float(policy["alpha"]) <= 1.0:
        raise ValueError("gripper alpha must be in (0, 1].")
    if not 0.0 <= float(policy["input_min"]) < float(policy["input_max"]) <= 1.0:
        raise ValueError(
            "gripper input_min/input_max must satisfy 0 <= input_min < input_max <= 1."
        )
    if float(policy["send_threshold"]) < 0.0:
        raise ValueError("gripper send_threshold must be >= 0.")
    if float(policy["command_hz"]) <= 0.0:
        raise ValueError("gripper command_hz must be > 0.")
    if float(policy["telemetry_hz"]) <= 0.0:
        raise ValueError("gripper telemetry_hz must be > 0.")
    if float(policy["telemetry_command_grace_s"]) < 0.0:
        raise ValueError("gripper telemetry_command_grace_s must be >= 0.")
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
