"""Tactile visualizer construction for the real-teleop frontend."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..tactile_visualizer import (
    AsyncDMTacDistributedForceVisualizer,
    AsyncTactileUVVisualizer,
    DMTacDistributedForceVisualizer,
    DMTacForceVisualizerConfig,
    TactileUVVisualizer,
    TactileVisualizerConfig,
)
from .hand_config import dmtac_sides as _dmtac_sides


DEFAULT_DMTAC_ROOT = Path(__file__).resolve().parents[2] / "hardware" / "DMTAC_gripper"


def _build_tactile_visualizer(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    *,
    force: bool = False,
) -> TactileUVVisualizer | AsyncTactileUVVisualizer | DMTacDistributedForceVisualizer | AsyncDMTacDistributedForceVisualizer | None:
    if not (bool(args.visualize_tactile) or bool(force)):
        return None
    if _dmtac_sides(args, active_sides):
        config = DMTacForceVisualizerConfig(
            enabled_sides=active_sides,
            dmtac_root=DEFAULT_DMTAC_ROOT,
            fixed_ref=float(getattr(args, "dmtac_tactile_vis_fixed_ref", 1.0)),
        )
        if not bool(getattr(args, "skip_renderer", False)):
            return AsyncDMTacDistributedForceVisualizer(config)
        return DMTacDistributedForceVisualizer(config)
    config = TactileVisualizerConfig(
        enabled_sides=active_sides,
        image_size=(int(args.tactile_vis_size), int(args.tactile_vis_size)),
        value_scale=float(args.tactile_vis_value_scale),
    )
    if not bool(args.skip_renderer):
        return AsyncTactileUVVisualizer(config)
    return TactileUVVisualizer(config)


__all__ = ["DEFAULT_DMTAC_ROOT", "_build_tactile_visualizer"]

