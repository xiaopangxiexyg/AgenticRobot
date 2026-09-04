"""Deployment hand-backend and gripper-policy configuration resolution.

These helpers normalize CLI and task-profile inputs and apply the resolved
values to backend configuration objects.  They do not probe devices, construct
backends, start workers, or issue commands.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_motion.deploy.schema import ALL_SIDES

from ..backends.grippers import (
    DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
    GRIPPER_ACTION_MAX,
    DMTacGripperBackendConfig,
    normalize_dmtac_tactile_fields,
)


HAND_BACKEND_INSPIRE = "inspire"
HAND_BACKEND_DMTAC_GRIPPER = "dmtac-gripper"
HAND_BACKEND_CHOICES = (HAND_BACKEND_INSPIRE, HAND_BACKEND_DMTAC_GRIPPER)

# Deployment runs the vision policy at roughly 15 Hz on the RM65 profile.  A
# 100 ms observation gate leaves only one half-frame of jitter budget at that
# rate, so keep a deployment-specific margin without changing collection's
# stricter sampling contract.
GRIPPER_POLICY_DEFAULTS = {
    "output_dim": 0,
    "speed": 0.2,
    "torque": 0.2,
    "max_action": GRIPPER_ACTION_MAX,
    "send_threshold": 0.002,
    "command_hz": DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    "continuous_grasp": True,
    "telemetry_hz": DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    "telemetry_command_grace_s": DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
}


def _hand_backend_mode(args: argparse.Namespace) -> str:
    mode = str(
        getattr(args, "hand_backend", HAND_BACKEND_INSPIRE) or HAND_BACKEND_INSPIRE
    )
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported hand backend {mode!r}; expected one of {HAND_BACKEND_CHOICES}."
        )
    return mode


def _hand_backend_mode_for_side(args: argparse.Namespace, side: str) -> str:
    side = str(side).strip().lower()
    if side not in ALL_SIDES:
        raise ValueError(f"Unsupported hand backend side {side!r}.")
    override = getattr(args, f"{side}_hand_backend", None)
    mode = _hand_backend_mode(args) if override in (None, "") else str(override)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported {side} hand backend {mode!r}; expected one of {HAND_BACKEND_CHOICES}."
        )
    return mode


def _hands_enabled(args: argparse.Namespace) -> bool:
    if args.hands_enabled is not None:
        return bool(args.hands_enabled)
    if any(
        _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
        for side in ALL_SIDES
    ):
        return True
    return bool(args.left_hand_ip or args.right_hand_ip or args.visualize_tactile)


def _load_gripper_task_profile(args: argparse.Namespace) -> dict[str, Any]:
    path_text = str(getattr(args, "gripper_task_profile", "") or "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Gripper task profile {path} must contain a JSON object.")
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


def _profile_value(profile: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in profile:
            return profile[key]
    return None


def _resolved_gripper_policy(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_resolved_gripper_policy", None)
    if isinstance(cached, dict):
        return dict(cached)

    profile = _load_gripper_task_profile(args)
    policy: dict[str, Any] = dict(GRIPPER_POLICY_DEFAULTS)
    profile_keys = {
        "output_dim": ("output_dim", "gripper_slot"),
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
        value = _profile_value(profile, *aliases)
        if value is not None:
            policy[key] = value

    cli_overrides = {
        "output_dim": getattr(args, "gripper_output_dim", None),
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

    policy["output_dim"] = int(policy["output_dim"])
    policy["speed"] = float(policy["speed"])
    policy["torque"] = float(policy["torque"])
    policy["max_action"] = float(policy["max_action"])
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
    if int(policy["output_dim"]) != 0:
        raise ValueError("dmtac-gripper output_dim/gripper_slot must be 0.")
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


def _dmtac_tactile_enabled(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "dmtac_tactile", None)
    if explicit is not None:
        return bool(explicit)
    return True


def _apply_gripper_policy_to_dmtac_config(
    config: DMTacGripperBackendConfig,
    *,
    args: argparse.Namespace,
    policy: Mapping[str, Any],
) -> DMTacGripperBackendConfig:
    config.speed = float(policy["speed"])
    config.torque = float(policy["torque"])
    config.max_action = float(policy["max_action"])
    config.command_dim = int(policy["output_dim"])
    config.send_threshold = float(policy["send_threshold"])
    config.command_hz = float(policy["command_hz"])
    config.continuous_grasp = bool(policy["continuous_grasp"])
    config.telemetry_hz = float(policy["telemetry_hz"])
    config.gripper_telemetry_command_grace_s = float(
        policy.get(
            "telemetry_command_grace_s",
            DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
        )
    )
    tactile_timeout_ms = getattr(args, "dmtac_tactile_timeout_ms", None)
    if tactile_timeout_ms is not None:
        config.tactile_timeout_ms = int(tactile_timeout_ms)
    tactile_wait_for_new = getattr(args, "dmtac_tactile_wait_for_new", None)
    if tactile_wait_for_new is not None:
        config.tactile_wait_for_new = bool(tactile_wait_for_new)
    tactile_fields = getattr(args, "dmtac_tactile_fields", None)
    if tactile_fields is None:
        tactile_fields = policy.get("tactile_fields")
    if tactile_fields is not None:
        config.tactile_fields = normalize_dmtac_tactile_fields(tactile_fields)
    tactile_compute_profile = getattr(args, "dmtac_tactile_compute_profile", None)
    if tactile_compute_profile:
        config.tactile_compute_profile = str(tactile_compute_profile)
    elif policy.get("tactile_compute_profile"):
        config.tactile_compute_profile = str(policy["tactile_compute_profile"])
    if bool(getattr(args, "dmtac_tactile_sequential_read", False)):
        config.tactile_parallel_read = False
    if not _dmtac_tactile_enabled(args):
        config.connect_tactile = False
        config.tactile_ids = ()
    if bool(getattr(args, "rtc_real_shadow", False)) and not bool(
        getattr(args, "send_actions", False)
    ):
        # Live raw-RTC shadow may read gripper/tactile observations, but its
        # startup must not clear faults, write speed/torque, open the gripper,
        # reset tactile calibration, or accept a command through a bypass API.
        config.observation_only = True
        config.send_commands = False
        config.clear_fault_on_start = False
        config.open_on_start = False
        config.tactile_reset_on_start = False
        config.tactile_warmup_frames = 0
    config.enable_telemetry = bool(config.connect_hardware or config.connect_tactile)
    return config


__all__ = [
    "GRIPPER_POLICY_DEFAULTS",
    "HAND_BACKEND_CHOICES",
    "HAND_BACKEND_DMTAC_GRIPPER",
    "HAND_BACKEND_INSPIRE",
    "_apply_gripper_policy_to_dmtac_config",
    "_dmtac_tactile_enabled",
    "_hand_backend_mode",
    "_hand_backend_mode_for_side",
    "_hands_enabled",
    "_load_gripper_task_profile",
    "_profile_value",
    "_resolved_gripper_policy",
]
