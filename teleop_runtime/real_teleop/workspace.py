"""Real-teleop workspace configuration and command-line override policy.

Static workspace-file parsing remains in ``teleop_runtime.configs.arm_workspace``.
This module owns only the entrypoint-specific step that applies those resolved
bounds to a live teleop calibration and records the runtime provenance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..calibration import TeleopCalibration
from ..configs import DEFAULT_ARM_WORKSPACE_CONFIG_PATH, load_arm_workspace_config


def parse_vector3_arg(value: object, *, name: str) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
        text = " ".join(parts)
    else:
        text = str(value)
        parts = [part.strip() for part in text.replace(",", " ").split() if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must contain 3 floats")
    try:
        parsed = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {name}: {text}") from exc
    return (parsed[0], parsed[1], parsed[2])


def resolve_workspace_override_args(args: argparse.Namespace) -> None:
    for attr in (
        "arm_workspace_min",
        "arm_workspace_max",
        "left_arm_workspace_min",
        "left_arm_workspace_max",
        "right_arm_workspace_min",
        "right_arm_workspace_max",
    ):
        value = getattr(args, attr, None)
        if value is None:
            continue
        setattr(args, attr, parse_vector3_arg(value, name=f"--{attr.replace('_', '-')}"))


def workspace_override_for_side(
    args: argparse.Namespace,
    side: str,
    bound: str,
) -> tuple[float, float, float] | None:
    side_value = getattr(args, f"{side}_arm_workspace_{bound}", None)
    if side_value is not None:
        return side_value
    return getattr(args, f"arm_workspace_{bound}", None)


def arm_workspace_config_path(args: argparse.Namespace) -> Path | None:
    value = str(
        getattr(args, "arm_workspace_config", str(DEFAULT_ARM_WORKSPACE_CONFIG_PATH))
        or ""
    ).strip()
    if value.lower() in {"", "none", "off", "false", "disabled"}:
        return None
    return Path(value).expanduser()


def apply_workspace_config(
    args: argparse.Namespace,
    calibration: TeleopCalibration,
) -> dict[str, dict[str, object]]:
    config_path = arm_workspace_config_path(args)
    if config_path is None:
        calibration.metadata.pop("workspace_config_runtime", None)
        calibration.metadata["workspace_config_source"] = None
        return {}
    bounds_by_side = load_arm_workspace_config(config_path)
    applied: dict[str, dict[str, object]] = {}
    for arm_name, arm in calibration.arms.items():
        side = str(arm.side or "")
        bounds = bounds_by_side.get(side)
        if bounds is None:
            continue
        arm.workspace_min = np.asarray(bounds.workspace_min, dtype=float)
        arm.workspace_max = np.asarray(bounds.workspace_max, dtype=float)
        arm.metadata["workspace_config_source"] = bounds.source_path
        applied[arm_name] = bounds.to_metadata()
    calibration.metadata["workspace_config_source"] = str(config_path)
    if applied:
        calibration.metadata["workspace_config_runtime"] = applied
    else:
        calibration.metadata.pop("workspace_config_runtime", None)
    return applied


def apply_workspace_overrides(
    args: argparse.Namespace,
    calibration: TeleopCalibration,
) -> dict[str, dict[str, list[float]]]:
    applied: dict[str, dict[str, list[float]]] = {}
    for arm_name, arm in calibration.arms.items():
        side = str(arm.side or "")
        if side not in {"left", "right"}:
            continue
        min_override = workspace_override_for_side(args, side, "min")
        max_override = workspace_override_for_side(args, side, "max")
        if min_override is None and max_override is None:
            continue
        current_min = (
            None
            if arm.workspace_min is None
            else np.asarray(arm.workspace_min, dtype=float)
        )
        current_max = (
            None
            if arm.workspace_max is None
            else np.asarray(arm.workspace_max, dtype=float)
        )
        workspace_min = (
            np.asarray(min_override, dtype=float)
            if min_override is not None
            else current_min
        )
        workspace_max = (
            np.asarray(max_override, dtype=float)
            if max_override is not None
            else current_max
        )
        if workspace_min is None or workspace_max is None:
            raise ValueError(
                f"Cannot partially override workspace for {arm_name}: existing profile "
                "must define both workspace_min and workspace_max, or pass both bounds."
            )
        if np.any(workspace_min > workspace_max):
            raise ValueError(
                f"Invalid workspace override for {arm_name}: workspace_min "
                f"{workspace_min.tolist()} exceeds workspace_max {workspace_max.tolist()}."
            )
        arm.workspace_min = workspace_min.copy()
        arm.workspace_max = workspace_max.copy()
        arm.metadata["workspace_override_runtime"] = True
        applied[arm_name] = {
            "workspace_min": arm.workspace_min.tolist(),
            "workspace_max": arm.workspace_max.tolist(),
        }
    if applied:
        calibration.metadata["workspace_override_runtime"] = applied
    else:
        calibration.metadata.pop("workspace_override_runtime", None)
    return applied


__all__ = [
    "apply_workspace_config",
    "apply_workspace_overrides",
    "arm_workspace_config_path",
    "parse_vector3_arg",
    "resolve_workspace_override_args",
    "workspace_override_for_side",
]
