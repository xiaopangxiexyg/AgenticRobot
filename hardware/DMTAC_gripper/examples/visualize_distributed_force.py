#!/usr/bin/env python3
import argparse
import os
import sys

import _bootstrap  # noqa: F401

from portable_dmtac_gripper import TactileConfig
from portable_dmtac_gripper.cli import _parse_dev_id
from portable_dmtac_gripper.force_visualizer import ForceVizConfig, run_live_distributed_force_xy


def main() -> None:
    parser = argparse.ArgumentParser(description="Live DM-Tac distributed_force XY visualization")
    parser.add_argument("--tactile_ids", nargs="*", default=None)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "Flux", "flux"])
    parser.add_argument("--mode", default="standard", choices=["standard", "high"])
    parser.add_argument("--max_fps", type=int, default=120)
    parser.add_argument("--timeout_ms", type=int, default=100)
    parser.add_argument("--count", type=int, default=0, help="0 means run until q/ESC")
    parser.add_argument("--max_hz", type=float, default=0.0, help="0 means no extra sleep")
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--arrow_scale", type=float, default=18.0)
    parser.add_argument("--min_arrow_norm", type=float, default=0.015)
    parser.add_argument("--deadband_abs", type=float, default=5e-5)
    parser.add_argument("--temporal_alpha", type=float, default=0.35)
    parser.add_argument("--spatial_blur_ksize", type=int, default=5)
    parser.add_argument("--arrow_pool", type=int, default=16)
    parser.add_argument("--viz_mode", default="absolute", choices=["delta", "absolute", "frame_delta"])
    parser.add_argument("--baseline_frames", type=int, default=8)
    parser.add_argument("--fixed_ref", type=float, default=None)
    parser.add_argument("--norm_percentile", type=float, default=99.0)
    parser.add_argument("--reset_tactile", action="store_true")
    parser.add_argument("--warmup_frames", type=int, default=0)
    parser.add_argument("--window", default="DM-Tac distributed_force XY")
    args = parser.parse_args()

    dev_ids = [_parse_dev_id(item) for item in args.tactile_ids] if args.tactile_ids else None
    tactile = TactileConfig(
        dev_ids=dev_ids,
        backend=args.backend,
        mode=args.mode,
        max_fps=args.max_fps,
        fields=("distributed_force",),
    )
    viz = ForceVizConfig(
        stride=args.stride,
        scale=args.scale,
        arrow_scale=args.arrow_scale,
        min_arrow_norm=args.min_arrow_norm,
        deadband_abs=args.deadband_abs,
        temporal_alpha=args.temporal_alpha,
        spatial_blur_ksize=args.spatial_blur_ksize,
        arrow_pool=args.arrow_pool,
        mode=args.viz_mode,
        baseline_frames=args.baseline_frames,
        fixed_ref=args.fixed_ref,
        norm_percentile=args.norm_percentile,
    )
    run_live_distributed_force_xy(
        tactile_config=tactile,
        timeout_ms=args.timeout_ms,
        count=args.count,
        max_hz=args.max_hz,
        window=args.window,
        viz_config=viz,
        reset_tactile=args.reset_tactile,
        warmup_frames=args.warmup_frames,
        close_devices=False,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
