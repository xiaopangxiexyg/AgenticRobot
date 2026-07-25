"""DMTac backend-config normalization for the real-teleop runtime.

These helpers only update a not-yet-started ``DMTacGripperBackendConfig``.
Hardware assignment loading, backend construction, startup, and command
dispatch remain in the real-teleop coordinator.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from ..backends.grippers import (
    DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
    DMTacGripperBackendConfig,
    normalize_dmtac_tactile_fields,
)
from .dataset_layout import expand_cli_path


def dmtac_tactile_enabled(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "dmtac_tactile", None)
    if explicit is not None:
        return bool(explicit)
    return False


def apply_gripper_policy_to_dmtac_config(
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
    tactile_read_hz = getattr(args, "dmtac_tactile_read_hz", None)
    if tactile_read_hz is not None:
        config.tactile_read_hz = float(tactile_read_hz)
    elif not bool(getattr(args, "no_collection", False)):
        # Formal collection stores tactile at the sample cadence.  The legacy
        # 1 Hz backend default remains unchanged for non-collection tools.
        config.tactile_read_hz = min(
            float(config.tactile_max_fps),
            max(1.0, float(getattr(args, "hz", 15.0))),
        )
    tactile_process_read = getattr(args, "dmtac_tactile_process_read", None)
    if tactile_process_read is not None:
        config.tactile_process_read = bool(tactile_process_read)
    tactile_worker_cpus = getattr(args, "dmtac_tactile_worker_cpus", None)
    if tactile_worker_cpus is not None:
        config.tactile_process_cpu_affinity = tuple(int(cpu) for cpu in tactile_worker_cpus)
    tactile_worker_nice = getattr(args, "dmtac_tactile_worker_nice", None)
    if tactile_worker_nice is not None:
        config.tactile_process_nice = int(tactile_worker_nice)
    tactile_worker_queue_size = getattr(args, "dmtac_tactile_worker_queue_size", None)
    if tactile_worker_queue_size is not None:
        config.tactile_process_queue_size = int(tactile_worker_queue_size)
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
    tactile_calibrate = getattr(args, "dmtac_tactile_calibrate", None)
    if tactile_calibrate is not None:
        config.tactile_reset_on_start = bool(tactile_calibrate)
    tactile_calibration_policy = getattr(args, "dmtac_tactile_calibration_policy", None)
    if tactile_calibration_policy:
        config.tactile_calibration_policy = str(tactile_calibration_policy)
    tactile_calibration_dir = str(
        getattr(args, "dmtac_tactile_calibration_dir", "") or ""
    ).strip()
    if tactile_calibration_dir:
        config.tactile_calibration_dir = expand_cli_path(tactile_calibration_dir)
    tactile_max_fps = getattr(args, "dmtac_tactile_max_fps", None)
    if tactile_max_fps is not None:
        config.tactile_max_fps = int(tactile_max_fps)
    tactile_warmup_frames = getattr(args, "dmtac_tactile_warmup_frames", None)
    if tactile_warmup_frames is not None:
        config.tactile_warmup_frames = int(tactile_warmup_frames)
    tactile_denoise = getattr(args, "dmtac_tactile_denoise", None)
    if tactile_denoise is not None:
        config.tactile_denoise = bool(tactile_denoise)
    tactile_flat_vector = getattr(args, "dmtac_tactile_flat_vector", None)
    if tactile_flat_vector is not None:
        config.tactile_store_flat_vector = bool(tactile_flat_vector)
    tactile_parallel_read = getattr(args, "dmtac_tactile_parallel_read", None)
    if tactile_parallel_read is not None:
        config.tactile_parallel_read = bool(tactile_parallel_read)
    elif bool(getattr(args, "dmtac_tactile_sequential_read", False)):
        config.tactile_parallel_read = False
    tactile_global_read_lock = getattr(args, "dmtac_tactile_global_read_lock", None)
    if tactile_global_read_lock is not None:
        config.tactile_global_read_lock = bool(tactile_global_read_lock)
    if not dmtac_tactile_enabled(args):
        config.connect_tactile = False
        config.tactile_ids = ()
    config.enable_telemetry = bool(config.connect_hardware or config.connect_tactile)
    return config


__all__ = [
    "apply_gripper_policy_to_dmtac_config",
    "dmtac_tactile_enabled",
]
