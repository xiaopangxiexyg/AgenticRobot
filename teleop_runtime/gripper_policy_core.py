"""Task-profile loading and base DM-Tac gripper policy resolution.

Application entrypoints own their defaults, side-specific policy, and backend
construction. This module only resolves profile/CLI values into a mapping.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .backends.grippers import GRIPPER_ACTION_MAX, normalize_dmtac_tactile_fields

PROFILE_KEYS: dict[str, tuple[str, ...]] = {
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


def profile_value(profile: Mapping[str, Any], *keys: str) -> Any:
    """Return the first present key, including an explicitly configured None."""

    for key in keys:
        if key in profile:
            return profile[key]
    return None


def load_gripper_task_profile(
    args: argparse.Namespace,
    *,
    invalid_root_error: type[Exception] = ValueError,
    merge_profile: Callable[
        [Mapping[str, Any] | None, Mapping[str, Any] | None], dict[str, Any]
    ]
    | None = None,
) -> dict[str, Any]:
    """Select the task profile; optionally retain real-teleop side overlays."""

    path_text = str(getattr(args, "gripper_task_profile", "") or "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise invalid_root_error(
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
        override = gripper if isinstance(gripper, Mapping) else selected
        if merge_profile is not None:
            return merge_profile(base_gripper, override)
        merged = dict(base_gripper) if isinstance(base_gripper, Mapping) else {}
        merged.update(dict(override))
        return merged
    if isinstance(base_gripper, Mapping):
        return (
            merge_profile(base_gripper, None)
            if merge_profile is not None
            else dict(base_gripper)
        )
    if "gripper" in payload:
        raise ValueError(f"Profile {path} field 'gripper' must be an object.")
    return merge_profile(payload, None) if merge_profile is not None else dict(payload)


def overlay_gripper_policy(
    args: argparse.Namespace,
    defaults: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply profile aliases and then explicit CLI fields in legacy order."""

    policy = dict(defaults)
    overlay_profile_values(
        policy, profile, (*defaults, "tactile_fields", "tactile_compute_profile")
    )
    for key in defaults:
        value = getattr(args, f"gripper_{key}", None)
        if value is not None:
            policy[key] = value
    return policy


def overlay_profile_values(
    policy: dict[str, Any], profile: Mapping[str, Any], keys: Iterable[str]
) -> None:
    """Apply only the requested profile fields, preserving alias precedence."""

    for key in keys:
        value = profile_value(profile, *PROFILE_KEYS[key])
        if value is not None:
            policy[key] = value


def normalize_gripper_policy(
    policy: dict[str, Any], *, canonical_max_action: bool
) -> None:
    """Normalize shared scalar fields without changing application defaults."""

    if "source_dim" in policy:
        policy["source_dim"] = int(policy["source_dim"])
    policy["output_dim"] = int(policy["output_dim"])
    if "invert" in policy:
        policy["invert"] = bool(policy["invert"])
        policy["alpha"] = float(policy["alpha"])
        policy["input_min"] = float(policy["input_min"])
        policy["input_max"] = float(policy["input_max"])
    policy["speed"] = float(policy["speed"])
    policy["torque"] = float(policy["torque"])
    policy["max_action"] = (
        float(GRIPPER_ACTION_MAX)
        if canonical_max_action
        else float(policy["max_action"])
    )
    policy["send_threshold"] = float(policy["send_threshold"])
    policy["command_hz"] = float(policy["command_hz"])
    policy["continuous_grasp"] = bool(policy["continuous_grasp"])
    policy["telemetry_hz"] = float(policy["telemetry_hz"])
    policy["telemetry_command_grace_s"] = float(policy["telemetry_command_grace_s"])
    if policy.get("tactile_fields") is not None:
        policy["tactile_fields"] = normalize_dmtac_tactile_fields(
            policy["tactile_fields"]
        )
    if policy.get("tactile_compute_profile") is not None:
        policy["tactile_compute_profile"] = str(policy["tactile_compute_profile"])
    if "source_dim" in policy and not 0 <= policy["source_dim"] < 6:
        raise ValueError("gripper source_dim must be in [0, 5].")
    if policy["output_dim"] != 0:
        raise ValueError("dmtac-gripper output_dim/gripper_slot must be 0.")
    if "alpha" in policy:
        if not 0.0 < policy["alpha"] <= 1.0:
            raise ValueError("gripper alpha must be in (0, 1].")
        if not 0.0 <= policy["input_min"] < policy["input_max"] <= 1.0:
            raise ValueError(
                "gripper input_min/input_max must satisfy 0 <= input_min < input_max <= 1."
            )
    if policy["send_threshold"] < 0.0:
        raise ValueError("gripper send_threshold must be >= 0.")
    if policy["command_hz"] <= 0.0:
        raise ValueError("gripper command_hz must be > 0.")
    if policy["telemetry_hz"] <= 0.0:
        raise ValueError("gripper telemetry_hz must be > 0.")
    if policy["telemetry_command_grace_s"] < 0.0:
        raise ValueError("gripper telemetry_command_grace_s must be >= 0.")
