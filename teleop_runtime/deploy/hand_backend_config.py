"""Deployment hand-backend and gripper-policy configuration resolution.

These helpers normalize CLI and task-profile inputs and apply the resolved
values to backend configuration objects.  They do not probe devices, construct
backends, start workers, or issue commands.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
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
from ..gripper_policy_core import (
    load_gripper_task_profile,
    normalize_gripper_policy,
    overlay_gripper_policy,
    profile_value,
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
    return load_gripper_task_profile(args, invalid_root_error=TypeError)


_profile_value = profile_value


def _resolved_gripper_policy(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_resolved_gripper_policy", None)
    if isinstance(cached, dict):
        return dict(cached)

    profile = _load_gripper_task_profile(args)
    policy = overlay_gripper_policy(args, GRIPPER_POLICY_DEFAULTS, profile)
    normalize_gripper_policy(policy, canonical_max_action=False)
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
