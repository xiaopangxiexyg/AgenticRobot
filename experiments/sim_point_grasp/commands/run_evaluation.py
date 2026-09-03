"""Run physical MuJoCo evaluation of SAM2 + GraspNet point grasps."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.sim_point_grasp.evaluation import (
    SimPointGraspEvaluationConfig,
    evaluate_sim_point_grasps,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render fixed-camera MuJoCo RGB-D, call the formal point-grasp provider, "
            "and execute selected DMTAC M grasps through physical Cube-Lift motion."
        )
    )
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime_artifacts/runs/experiments/sim_point_grasp") / stamp,
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
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--target-point-count", type=int, default=1)
    parser.add_argument(
        "--object-mass-kg", type=float, choices=(0.08, 0.20), default=0.08
    )
    parser.add_argument(
        "--gripper-force-profile",
        choices=("standard", "strong"),
        default="standard",
    )
    parser.add_argument("--object-center-x-m", type=float, default=-0.272)
    parser.add_argument("--object-center-y-m", type=float, default=0.0)
    parser.add_argument("--spawn-x-jitter-m", type=float, default=0.025)
    parser.add_argument("--spawn-y-jitter-m", type=float, default=0.080)
    parser.add_argument(
        "--spawn-yaw-jitter-rad", type=float, default=0.7853981633974483
    )
    parser.add_argument("--pregrasp-offset-m", type=float, default=0.070)
    parser.add_argument(
        "--adaptive-pregrasp-offsets-m",
        type=float,
        nargs="*",
        default=(0.050, 0.035),
        help="Fallback offsets tested only when no candidate is executable at the primary offset.",
    )
    parser.add_argument(
        "--candidate-selection-policy",
        choices=("reachability_aware", "provider_selected_ordinal"),
        default="reachability_aware",
    )
    parser.add_argument("--lift-offset-m", type=float, default=0.100)
    parser.add_argument("--phase-step-limit", type=int, default=90)
    parser.add_argument("--lift-step-limit", type=int, default=120)
    parser.add_argument("--close-step-limit", type=int, default=15)
    parser.add_argument("--close-action-scale", type=float, default=1.0)
    parser.add_argument("--contact-stability-steps", type=int, default=1)
    parser.add_argument("--lift-translation-action-scale", type=float, default=0.25)
    parser.add_argument("--no-video", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    config = SimPointGraspEvaluationConfig(
        output_dir=args.output_dir,
        sam2_asset_bundle=args.sam2_asset_bundle,
        graspnet_asset_bundle=args.graspnet_asset_bundle,
        trials=args.trials,
        seed_start=args.seed_start,
        device=args.device,
        max_candidates=args.max_candidates,
        target_point_count=args.target_point_count,
        object_mass_kg=args.object_mass_kg,
        gripper_force_profile=args.gripper_force_profile,
        cube_center_xy_m=(args.object_center_x_m, args.object_center_y_m),
        cube_spawn_xy_jitter_m=(args.spawn_x_jitter_m, args.spawn_y_jitter_m),
        cube_spawn_yaw_jitter_rad=args.spawn_yaw_jitter_rad,
        pregrasp_offset_m=args.pregrasp_offset_m,
        adaptive_pregrasp_offsets_m=tuple(args.adaptive_pregrasp_offsets_m),
        candidate_selection_policy=args.candidate_selection_policy,
        lift_offset_m=args.lift_offset_m,
        phase_step_limit=args.phase_step_limit,
        lift_step_limit=args.lift_step_limit,
        close_step_limit=args.close_step_limit,
        close_action_scale=args.close_action_scale,
        contact_stability_steps=args.contact_stability_steps,
        lift_translation_action_scale=args.lift_translation_action_scale,
        save_video=not args.no_video,
    )
    return evaluate_sim_point_grasps(config)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        report = run(parser.parse_args(argv))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"simulation point-grasp evaluation failed: {exc}\n")
    print(
        json.dumps(
            {
                "attempted_trials": report["attempted_trials"],
                "success_count": report["success_count"],
                "success_rate": report["success_rate"],
                "failure_count_by_stage": report["failure_count_by_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
