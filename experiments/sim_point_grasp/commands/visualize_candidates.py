"""Render point-grasp candidates in three MuJoCo views without motion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.sim_point_grasp.visualization import (
    SimPointGraspVisualizationConfig,
    visualize_sim_point_grasps,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run front-camera MuJoCo RGB-D through SAM2 + GraspNet and project "
            "each ranked DMTAC M wireframe into separate front, left_wrist, "
            "and right_wrist images. Wrist views use visualization-only "
            "look-at cameras aimed at the cube. No IK, robot motion, or "
            "gripper action is run."
        )
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("runtime_artifacts/runs/experiments/sim_point_grasp_visualization")
            / stamp
        ),
    )
    parser.add_argument(
        "--sam2-asset-bundle",
        type=Path,
        default=Path("assets/perception/sam2_grounding_dino"),
    )
    parser.add_argument(
        "--graspnet-asset-bundle",
        type=Path,
        default=Path("assets/perception/graspnet-point-grasp"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--arm-side",
        choices=("left", "right"),
        default="right",
        help="Arm-base frame used for grasp generation and overlay transforms.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-candidates", type=int, default=5)
    return parser


def run(args: argparse.Namespace) -> dict:
    return visualize_sim_point_grasps(
        SimPointGraspVisualizationConfig(
            output_dir=args.output_dir,
            sam2_asset_bundle=args.sam2_asset_bundle,
            graspnet_asset_bundle=args.graspnet_asset_bundle,
            arm_side=args.arm_side,
            seed=args.seed,
            device=args.device,
            max_candidates=args.max_candidates,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        report = run(parser.parse_args(argv))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"simulation point-grasp visualization failed: {exc}\n")
    print(
        json.dumps(
            {
                "candidate_count": report["candidate_count"],
                "arm_side": report["controlled_side"],
                "selected_ordinal": report["selected_ordinal"],
                "camera_names": report["camera_names"],
                "simulation_state_unchanged": report["simulation_state_unchanged"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
