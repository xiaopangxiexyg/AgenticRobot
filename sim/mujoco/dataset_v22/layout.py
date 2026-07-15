"""Shared RM65 bimanual DM-Tac v2.2 layout loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.v22 import PolicyIOLayout


DEFAULT_RM65_BIMANUAL_DMTAC_LAYOUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "teleop_runtime"
    / "configs"
    / "policy_io_layouts"
    / "dual_arm_bimanual_dmtac_gripper_v22.json"
)


def _layout_payload(path: Path) -> Mapping[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"policy I/O layout must be a JSON object: {path}")
    nested = value.get("policy_io_layout")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError(f"policy_io_layout must be a JSON object: {path}")
        value = nested
    return value


def validate_rm65_bimanual_dmtac_layout(layout: PolicyIOLayout) -> PolicyIOLayout:
    """Reject layouts that would silently change the real/sim gripper meaning."""

    for side in ("left", "right"):
        spec = layout.end_effector(side)
        channel = layout.channel("action", side)
        state_channel = layout.channel("state", side)
        if spec.adapter != "dmtac_gripper_v1":
            raise ValueError(
                f"RM65 sim v2.2 requires {side} adapter='dmtac_gripper_v1', got {spec.adapter!r}."
            )
        if channel.value_type != "continuous" or channel.dim != 1:
            raise ValueError(f"RM65 sim v2.2 requires one continuous {side} gripper action.")
        if state_channel.value_type != "continuous" or state_channel.dim != 1:
            raise ValueError(f"RM65 sim v2.2 requires one continuous {side} gripper state.")
        if channel.units != "normalized_0_1":
            raise ValueError(
                f"RM65 sim v2.2 requires {side} gripper units='normalized_0_1', got {channel.units!r}."
            )
        normalization = dict(channel.normalization)
        if normalization.get("minimum") != 0.0 or normalization.get("maximum") != 1.0:
            raise ValueError(
                f"RM65 sim v2.2 requires {side} gripper normalization [0.0, 1.0]."
            )
    return layout


def load_rm65_bimanual_dmtac_layout(path: str | Path | None = None) -> PolicyIOLayout:
    """Load the exact real-robot layout used for bimanual DM-Tac collection."""

    layout_path = (
        DEFAULT_RM65_BIMANUAL_DMTAC_LAYOUT_PATH if path is None else Path(path).expanduser()
    ).resolve()
    if not layout_path.is_file():
        raise FileNotFoundError(f"RM65 bimanual DM-Tac policy layout does not exist: {layout_path}")
    layout = PolicyIOLayout.from_dict(_layout_payload(layout_path))
    return validate_rm65_bimanual_dmtac_layout(layout)


__all__ = [
    "DEFAULT_RM65_BIMANUAL_DMTAC_LAYOUT_PATH",
    "load_rm65_bimanual_dmtac_layout",
    "validate_rm65_bimanual_dmtac_layout",
]
