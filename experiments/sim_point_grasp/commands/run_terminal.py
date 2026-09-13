"""Run one real-like point-grasp generation and execution cycle in MuJoCo."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.sim_point_grasp.evaluation import (
    EVALUATION_OBJECT_CENTER_XY_M,
    SimPointGraspEvaluationConfig,
    evaluate_sim_point_grasps,
)

_DEFAULT_POINT_GRASP_BUNDLE_BY_BACKEND = {
    "vgn": Path("assets/perception/sam2_vgn_point_grasp"),
    "graspnet": Path("assets/perception/graspnet-point-grasp"),
    "graspgenx": Path("assets/perception/graspgenx-point-grasp"),
}


def build_parser() -> argparse.ArgumentParser:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Render aligned MuJoCo RGB-D, generate and render point-grasp "
            "candidates, then execute the selected grasp through the physical "
            "CubeLift motion path. This command never connects to real hardware."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime_artifacts/runs/experiments/sim_point_grasp_terminal")
        / stamp,
    )
    parser.add_argument("--arm-side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--backend",
        choices=("vgn", "graspnet", "graspgenx"),
        default="graspgenx",
        help="Point-grasp generator; matches the real Gateway backend choices.",
    )
    parser.add_argument(
        "--view",
        choices=("front", "selected_wrist"),
        default="selected_wrist",
        help=(
            "RGB-D source used for grasp generation. selected_wrist uses the "
            "actual controlled-arm wrist camera from its cube-facing reset pose."
        ),
    )
    parser.add_argument(
        "--sam2-asset-bundle",
        type=Path,
        default=Path("assets/perception/sam2_grounding_dino"),
    )
    parser.add_argument(
        "--point-grasp-asset-bundle",
        type=Path,
        default=None,
        help="Backend bundle; defaults to the reviewed bundle for --backend.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument(
        "--cube-center-x-m",
        type=float,
        default=EVALUATION_OBJECT_CENTER_XY_M[0],
        help="Deterministic simulated cube center x coordinate in meters.",
    )
    parser.add_argument(
        "--cube-center-y-m",
        type=float,
        default=EVALUATION_OBJECT_CENTER_XY_M[1],
        help="Deterministic simulated cube center y coordinate in meters.",
    )
    parser.add_argument(
        "--minimum-downward-approach-cosine",
        type=float,
        default=0.5,
        help=(
            "Tabletop admission: require tool +Z approach to point downward "
            "by at least this cosine; 0 allows horizontal approaches."
        ),
    )
    parser.add_argument(
        "--candidate-selection-policy",
        choices=("reachability_aware", "provider_selected_ordinal"),
        default="reachability_aware",
    )
    parser.add_argument(
        "--graspgenx-mode",
        choices=("service", "in-process"),
        default="service",
    )
    parser.add_argument("--graspgenx-host", default="127.0.0.1")
    parser.add_argument("--graspgenx-port", type=int, default=5556)
    parser.add_argument("--graspgenx-timeout-ms", type=int, default=120_000)
    parser.add_argument("--graspgenx-source-root", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    point_grasp_bundle = (
        args.point_grasp_asset_bundle
        if args.point_grasp_asset_bundle is not None
        else _DEFAULT_POINT_GRASP_BUNDLE_BY_BACKEND[args.backend]
    )
    config = SimPointGraspEvaluationConfig(
        output_dir=args.output_dir,
        sam2_asset_bundle=args.sam2_asset_bundle,
        # This legacy field now carries the selected backend's reviewed bundle;
        # keeping it avoids breaking the existing GraspNet evaluation API.
        graspnet_asset_bundle=point_grasp_bundle,
        arm_side=args.arm_side,
        point_grasp_backend=args.backend,
        generation_view=args.view,
        object_profile="cube",
        use_terminal_observation_home=True,
        graspgenx_mode=args.graspgenx_mode,
        graspgenx_host=args.graspgenx_host,
        graspgenx_port=args.graspgenx_port,
        graspgenx_timeout_ms=args.graspgenx_timeout_ms,
        graspgenx_source_root=args.graspgenx_source_root,
        trials=1,
        seed_start=args.seed,
        max_candidates=args.max_candidates,
        minimum_downward_approach_cosine=args.minimum_downward_approach_cosine,
        target_point_count=1,
        cube_center_xy_m=(args.cube_center_x_m, args.cube_center_y_m),
        cube_spawn_xy_jitter_m=(0.0, 0.0),
        cube_spawn_yaw_jitter_rad=0.0,
        candidate_selection_policy=args.candidate_selection_policy,
        save_video=not args.no_video,
    )
    return evaluate_sim_point_grasps(config)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"MuJoCo grasp terminal failed: {exc}\n")
    trial = report["trials"][0]
    print(
        json.dumps(
            {
                "backend": report["skill"]["backend"],
                "generation_camera": report["camera"]["name"],
                "candidate_count": trial["candidate_count"],
                "selected_ordinal": trial["selected_ordinal"],
                "execution_success": trial["success"],
                "failure_stage": trial["failure_stage"],
                "reason": trial["reason"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if trial["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
