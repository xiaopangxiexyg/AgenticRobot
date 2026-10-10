"""DMTac backend-config normalization for the real-teleop runtime.

These helpers only update a not-yet-started ``DMTacGripperBackendConfig``.
Hardware assignment loading, backend construction, startup, and command
dispatch remain in the real-teleop coordinator.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from collection_runtime.dataset_layout import expand_cli_path

from ..backends.grippers import DMTacGripperBackendConfig
from ..dmtac_config import apply_dmtac_capture_options, apply_dmtac_policy_options


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
    apply_dmtac_policy_options(config, args=args, policy=policy)
    config.startup_recovery_mode = str(
        getattr(args, "dmtac_startup_recovery_mode", "stabilized")
        or "stabilized"
    )
    if getattr(args, "dmtac_tactile_read_hz", None) is None and not bool(
        getattr(args, "no_collection", False)
    ):
        # Formal collection stores tactile at the sample cadence.  The legacy
        # 1 Hz backend default remains unchanged for non-collection tools.
        config.tactile_read_hz = min(
            float(config.tactile_max_fps),
            max(1.0, float(getattr(args, "hz", 15.0))),
        )
    apply_dmtac_capture_options(config, args=args)
    tactile_calibration_policy = getattr(args, "dmtac_tactile_calibration_policy", None)
    if tactile_calibration_policy:
        config.tactile_calibration_policy = str(tactile_calibration_policy)
    tactile_calibration_dir = str(
        getattr(args, "dmtac_tactile_calibration_dir", "") or ""
    ).strip()
    if tactile_calibration_dir:
        config.tactile_calibration_dir = expand_cli_path(tactile_calibration_dir)
    if not dmtac_tactile_enabled(args):
        config.connect_tactile = False
        config.tactile_ids = ()
    config.enable_telemetry = bool(config.connect_hardware or config.connect_tactile)
    return config


__all__ = [
    "apply_gripper_policy_to_dmtac_config",
    "dmtac_tactile_enabled",
]
