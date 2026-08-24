"""Policy-artifact camera input binding for deployment.

This module owns the pure contract decision that maps a policy artifact's
declared logical camera inputs to deployment CLI arguments.  Device/source
construction and camera lifecycle remain in :mod:`camera_runtime`.
"""

from __future__ import annotations

import argparse
from typing import Any

from .cli_args import argument_was_supplied


def resolve_policy_camera_input_keys(
    args: argparse.Namespace,
    policy_artifact: Any | None,
    *,
    raw_argv: list[str],
) -> tuple[str, ...] | None:
    """Resolve artifact-bound logical cameras before any device is opened.

    ``None`` deliberately means a legacy artifact with no formal visual
    contract, preserving its existing runtime camera behavior.  A tuple means
    the exact complete set that may be captured and forwarded to the policy.
    """

    camera_input = None if policy_artifact is None else policy_artifact.camera_input
    if camera_input is None:
        return None
    selected_keys = camera_input.source_keys
    if not selected_keys:
        if args.camera_mode != "none":
            raise ValueError(
                "policy artifact declares no camera inputs; use --camera-mode none."
            )
        return selected_keys
    if args.camera_mode == "none":
        if camera_input.required:
            raise ValueError(
                "policy artifact requires camera inputs "
                f"{list(selected_keys)}; choose --camera-mode single or multi."
            )
        return selected_keys
    if args.camera_serial:
        if len(selected_keys) != 1:
            raise ValueError(
                "--camera-serial can satisfy only one artifact camera input; "
                f"artifact requires {list(selected_keys)}."
            )
        selected_key = selected_keys[0]
        if (
            argument_was_supplied(raw_argv, "--camera-key")
            and args.camera_key != selected_key
        ):
            raise ValueError(
                "--camera-key disagrees with policy artifact camera input: "
                f"expected {selected_key!r}, got {args.camera_key!r}."
            )
        # The default CLI key is ``front``.  For a wrist-only artifact it is
        # safe to substitute the artifact's logical key before source creation.
        args.camera_key = selected_key
    elif args.camera_mode == "single" and len(selected_keys) != 1:
        raise ValueError(
            "--camera-mode single requires exactly one artifact camera input; "
            f"artifact requires {list(selected_keys)}."
        )
    return selected_keys


__all__ = ["resolve_policy_camera_input_keys"]
