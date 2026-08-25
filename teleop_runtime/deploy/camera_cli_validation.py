"""Pure camera CLI validation for policy deployment."""

from __future__ import annotations

import argparse
from pathlib import Path


def validate_camera_cli_args(args: argparse.Namespace) -> None:
    """Validate camera CLI values before deployment sources are constructed.

    This function intentionally owns no device or policy-artifact state.  It
    preserves the deployment CLI's existing validation order and error text
    while keeping pure argument checks out of the orchestration module.
    """

    if args.camera_mode in {"single", "multi"}:
        if args.camera_mode == "multi" and args.camera_serial:
            raise ValueError(
                "--camera-mode multi uses --camera-config; omit --camera-serial."
            )
        if not args.camera_serial and not Path(args.camera_config).exists():
            raise ValueError(
                f"--camera-mode {args.camera_mode} requires an existing --camera-config."
            )
        if int(args.camera_width) <= 0 or int(args.camera_height) <= 0:
            raise ValueError("--camera-width/--camera-height must be positive.")
        if int(args.camera_fps) <= 0:
            raise ValueError("--camera-fps must be positive.")
        if int(args.camera_keep_last) <= 0:
            raise ValueError("--camera-keep-last must be positive.")
        if float(args.camera_frame_max_age_sec) <= 0.0:
            raise ValueError("--camera-frame-max-age-sec must be positive.")
        if int(args.camera_wait_timeout_ms) <= 0:
            raise ValueError("--camera-wait-timeout-ms must be positive.")
        if float(args.camera_stale_after_sec) <= 0.0:
            raise ValueError("--camera-stale-after-sec must be positive.")
        if float(args.camera_start_stagger_sec) < 0.0:
            raise ValueError("--camera-start-stagger-sec must be >= 0.")
    elif bool(args.vis_cam):
        raise ValueError(
            "--vis_cam requires --camera-mode single or --camera-mode multi."
        )


__all__ = ["validate_camera_cli_args"]
