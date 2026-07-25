"""Pure gripper-policy and dataset-contract resolution for real teleoperation.

The helpers in this module normalize CLI/task-profile values and cache the
resolved policy on the supplied argument namespace.  They do not initialize
hardware, start workers, or issue gripper commands.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dataset_contracts.gripper_force import GripperForceDatasetContract

from ..backends.grippers import (
    DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
    GRIPPER_ACTION_MAX,
    normalize_dmtac_tactile_fields,
)
from ..retargeters import DEFAULT_GRIPPER_INPUT_MAX, DEFAULT_GRIPPER_INPUT_MIN
from .hand_config import (
    HAND_BACKEND_DMTAC_GRIPPER,
    hand_backend_mode_for_side,
)
from .components import RealTeleopComponents


GRIPPER_POLICY_DEFAULTS = {
    "source_dim": 2,
    "output_dim": 0,
    "invert": False,
    "alpha": 0.95,
    "input_min": DEFAULT_GRIPPER_INPUT_MIN,
    "input_max": DEFAULT_GRIPPER_INPUT_MAX,
    "speed": 1.0,
    "torque": 0.2,
    "max_action": GRIPPER_ACTION_MAX,
    "send_threshold": 0.002,
    "command_hz": DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    "continuous_grasp": True,
    "telemetry_hz": DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    "telemetry_command_grace_s": DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
}


def profile_value(profile: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in profile:
            return profile[key]
    return None


def strip_side_gripper_overrides(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in profile.items()
        if str(key) not in {"left", "right", "sides"}
    }


def side_gripper_overrides_from_profile(
    profile: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(profile, Mapping):
        return {}
    by_side: dict[str, dict[str, Any]] = {}
    sides = profile.get("sides")
    if isinstance(sides, Mapping):
        for side in ("left", "right"):
            item = sides.get(side)
            if isinstance(item, Mapping):
                by_side[side] = dict(item)
    for side in ("left", "right"):
        item = profile.get(side)
        if isinstance(item, Mapping):
            merged = dict(by_side.get(side, {}))
            merged.update(dict(item))
            by_side[side] = merged
    return by_side


def merge_gripper_profile(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = strip_side_gripper_overrides(base or {})
    side_overrides = side_gripper_overrides_from_profile(base)
    if isinstance(override, Mapping):
        merged.update(strip_side_gripper_overrides(override))
        for side, item in side_gripper_overrides_from_profile(override).items():
            side_merged = dict(side_overrides.get(side, {}))
            side_merged.update(item)
            side_overrides[side] = side_merged
    for side, item in side_overrides.items():
        merged[side] = dict(item)
    return merged


def load_gripper_task_profile(args: argparse.Namespace) -> dict[str, Any]:
    path_text = str(getattr(args, "gripper_task_profile", "") or "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Gripper task profile {path} must contain a JSON object.")
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
                setattr(args, "_resolved_gripper_task", key)
                break
    if selected is not None:
        gripper = selected.get("gripper")
        if gripper is not None and not isinstance(gripper, Mapping):
            raise ValueError(f"Profile {path} selected task field 'gripper' must be an object.")
        return merge_gripper_profile(
            base_gripper if isinstance(base_gripper, Mapping) else None,
            gripper if isinstance(gripper, Mapping) else selected,
        )
    if isinstance(base_gripper, Mapping):
        return merge_gripper_profile(base_gripper, None)
    if "gripper" in payload:
        raise ValueError(f"Profile {path} field 'gripper' must be an object.")
    return merge_gripper_profile(payload, None)


def gripper_force_dataset_contract_from_args(
    args: argparse.Namespace,
) -> GripperForceDatasetContract | None:
    raw = getattr(args, "_gripper_force_dataset_contract", None)
    if raw is None:
        return None
    if isinstance(raw, GripperForceDatasetContract):
        return raw
    if isinstance(raw, Mapping):
        return GripperForceDatasetContract.from_dict(raw)
    raise ValueError("_gripper_force_dataset_contract must be a mapping")


def resolved_gripper_policy(args: argparse.Namespace) -> dict[str, Any]:
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
        "tactile_compute_profile": ("tactile_compute_profile", "dmtac_tactile_compute_profile"),
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

    dataset_force_contract = gripper_force_dataset_contract_from_args(args)
    if dataset_force_contract is not None and len(dataset_force_contract.force_by_side) == 1:
        # The base policy is only a compatibility summary; side-specific
        # policies below remain authoritative for bimanual operation.
        policy["torque"] = next(iter(dataset_force_contract.force_by_side_dict.values()))

    policy["source_dim"] = int(policy["source_dim"])
    policy["output_dim"] = int(policy["output_dim"])
    policy["invert"] = bool(policy["invert"])
    policy["alpha"] = float(policy["alpha"])
    policy["input_min"] = float(policy["input_min"])
    policy["input_max"] = float(policy["input_max"])
    policy["speed"] = float(policy["speed"])
    policy["torque"] = float(policy["torque"])
    # Old task profiles used 0.9/0.95 as hardware workarounds.  v2.2 stores
    # the canonical normalized DM-Tac range, so force the runtime to 1.0.
    policy["max_action"] = float(GRIPPER_ACTION_MAX)
    policy["send_threshold"] = float(policy["send_threshold"])
    policy["command_hz"] = float(policy["command_hz"])
    policy["continuous_grasp"] = bool(policy["continuous_grasp"])
    policy["telemetry_hz"] = float(policy["telemetry_hz"])
    policy["telemetry_command_grace_s"] = float(
        policy["telemetry_command_grace_s"]
    )
    if policy.get("tactile_fields") is not None:
        policy["tactile_fields"] = normalize_dmtac_tactile_fields(policy["tactile_fields"])
    if policy.get("tactile_compute_profile") is not None:
        policy["tactile_compute_profile"] = str(policy["tactile_compute_profile"])
    if not 0 <= int(policy["source_dim"]) < 6:
        raise ValueError("gripper source_dim must be in [0, 5].")
    if int(policy["output_dim"]) != 0:
        raise ValueError("dmtac-gripper output_dim/gripper_slot must be 0.")
    if not 0.0 < float(policy["alpha"]) <= 1.0:
        raise ValueError("gripper alpha must be in (0, 1].")
    if not 0.0 <= float(policy["input_min"]) < float(policy["input_max"]) <= 1.0:
        raise ValueError("gripper input_min/input_max must satisfy 0 <= input_min < input_max <= 1.")
    if float(policy["send_threshold"]) < 0.0:
        raise ValueError("gripper send_threshold must be >= 0.")
    if float(policy["command_hz"]) <= 0.0:
        raise ValueError("gripper command_hz must be > 0.")
    if float(policy["telemetry_hz"]) <= 0.0:
        raise ValueError("gripper telemetry_hz must be > 0.")
    if float(policy["telemetry_command_grace_s"]) < 0.0:
        raise ValueError("gripper telemetry_command_grace_s must be >= 0.")
    setattr(args, "_resolved_gripper_policy", dict(policy))
    return dict(policy)


def gripper_side_override(profile: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    item = profile.get(side)
    if isinstance(item, Mapping):
        return item
    sides = profile.get("sides")
    if isinstance(sides, Mapping):
        item = sides.get(side)
        if isinstance(item, Mapping):
            return item
    return {}


def resolved_gripper_policy_for_side(
    args: argparse.Namespace,
    side: str,
) -> dict[str, Any]:
    cached = getattr(args, "_resolved_gripper_policy_by_side", None)
    if isinstance(cached, dict) and side in cached and isinstance(cached[side], dict):
        return dict(cached[side])

    base_policy = resolved_gripper_policy(args)
    profile = load_gripper_task_profile(args)
    side_override = gripper_side_override(profile, side)
    policy = dict(base_policy)
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
        "tactile_compute_profile": ("tactile_compute_profile", "dmtac_tactile_compute_profile"),
    }
    for key, aliases in profile_keys.items():
        value = profile_value(side_override, *aliases)
        if value is not None:
            policy[key] = value
    dataset_force_contract = gripper_force_dataset_contract_from_args(args)
    if dataset_force_contract is not None:
        force_by_side = dataset_force_contract.force_by_side_dict
        if side in force_by_side:
            policy["torque"] = float(force_by_side[side])
    policy["source_dim"] = int(policy["source_dim"])
    policy["output_dim"] = int(policy["output_dim"])
    policy["invert"] = bool(policy["invert"])
    policy["alpha"] = float(policy["alpha"])
    policy["input_min"] = float(policy["input_min"])
    policy["input_max"] = float(policy["input_max"])
    policy["speed"] = float(policy["speed"])
    policy["torque"] = float(policy["torque"])
    policy["max_action"] = float(GRIPPER_ACTION_MAX)
    policy["send_threshold"] = float(policy["send_threshold"])
    policy["command_hz"] = float(policy["command_hz"])
    policy["continuous_grasp"] = bool(policy["continuous_grasp"])
    policy["telemetry_hz"] = float(policy["telemetry_hz"])
    policy["telemetry_command_grace_s"] = float(
        policy["telemetry_command_grace_s"]
    )
    if policy.get("tactile_fields") is not None:
        policy["tactile_fields"] = normalize_dmtac_tactile_fields(policy["tactile_fields"])
    if policy.get("tactile_compute_profile") is not None:
        policy["tactile_compute_profile"] = str(policy["tactile_compute_profile"])
    if not 0 <= int(policy["source_dim"]) < 6:
        raise ValueError("gripper source_dim must be in [0, 5].")
    if int(policy["output_dim"]) != 0:
        raise ValueError("dmtac-gripper output_dim/gripper_slot must be 0.")
    if not 0.0 < float(policy["alpha"]) <= 1.0:
        raise ValueError("gripper alpha must be in (0, 1].")
    if not 0.0 <= float(policy["input_min"]) < float(policy["input_max"]) <= 1.0:
        raise ValueError("gripper input_min/input_max must satisfy 0 <= input_min < input_max <= 1.")
    if not 0.0 <= float(policy["speed"]) <= 1.0:
        raise ValueError("gripper speed must be in [0, 1].")
    if not 0.0 <= float(policy["torque"]) <= 1.0:
        raise ValueError("gripper force/torque must be in [0, 1].")
    if float(policy["telemetry_hz"]) <= 0.0:
        raise ValueError("gripper telemetry_hz must be > 0.")
    if float(policy["telemetry_command_grace_s"]) < 0.0:
        raise ValueError("gripper telemetry_command_grace_s must be >= 0.")
    by_side = dict(cached) if isinstance(cached, dict) else {}
    by_side[side] = dict(policy)
    setattr(args, "_resolved_gripper_policy_by_side", by_side)
    return dict(policy)


def resolved_gripper_policies_by_side(
    args: argparse.Namespace,
    active_sides: Sequence[str],
) -> dict[str, dict[str, Any]]:
    return {
        str(side): resolved_gripper_policy_for_side(args, str(side))
        for side in active_sides
        if str(side) in {"left", "right"}
    }


def current_gripper_force_dataset_contract(
    args: argparse.Namespace,
    active_sides: Sequence[str],
) -> GripperForceDatasetContract | None:
    """Build the immutable force/profile identity used by dataset metadata."""

    dmtac_sides = tuple(
        side
        for side in active_sides
        if hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
    )
    if not dmtac_sides:
        return None
    policies = resolved_gripper_policies_by_side(args, dmtac_sides)
    force_by_side = {
        side: float(policy["torque"])
        for side, policy in policies.items()
        if isinstance(policy, Mapping) and "torque" in policy
    }
    if not force_by_side:
        return None
    return GripperForceDatasetContract(
        force_by_side=tuple(force_by_side.items()),
        profile_name=str(getattr(args, "gripper_task_profile", "") or ""),
        task=str(getattr(args, "gripper_task", "") or ""),
    )


def gripper_force_dataset_metadata(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    contract: GripperForceDatasetContract | None = None,
) -> dict[str, object]:
    effective = contract or gripper_force_dataset_contract_from_args(args)
    if effective is None:
        return {}
    policies = resolved_gripper_policies_by_side(args, components.active_sides)
    return {
        "gripper_force_profile": effective.to_dict(),
        "gripper_task_profile": str(getattr(args, "gripper_task_profile", "") or "") or None,
        "gripper_task": str(getattr(args, "gripper_task", "") or "") or None,
        "gripper_policy_by_side": policies,
    }


def clear_gripper_policy_cache(args: argparse.Namespace) -> None:
    for name in (
        "_resolved_gripper_policy",
        "_resolved_gripper_policy_by_side",
        "_dmtac_backend_configs",
    ):
        if hasattr(args, name):
            delattr(args, name)


__all__ = [
    "GRIPPER_POLICY_DEFAULTS",
    "clear_gripper_policy_cache",
    "current_gripper_force_dataset_contract",
    "gripper_force_dataset_contract_from_args",
    "gripper_force_dataset_metadata",
    "gripper_side_override",
    "load_gripper_task_profile",
    "merge_gripper_profile",
    "profile_value",
    "resolved_gripper_policies_by_side",
    "resolved_gripper_policy",
    "resolved_gripper_policy_for_side",
    "side_gripper_overrides_from_profile",
    "strip_side_gripper_overrides",
]
