from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tracker_replay import (
    OfflineTrackerReplayConfig,
    report_exceeds_thresholds,
    run_offline_tracker_replay,
    save_offline_tracker_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay recorded VIVE-derived target poses through the shared RM65 IK stack "
            "and compare the result with the original dataset action/qpos."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="LeRobot-style dataset root.")
    parser.add_argument("--episode", default="0", help="Episode index/id, e.g. 0 or episode_000000.")
    parser.add_argument("--start-frame", type=int, default=0, help="Sorted row offset to start from.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to replay; 0 means all.")
    parser.add_argument(
        "--active-sides",
        default="left,right",
        help="Comma-separated sides to compare. Default: left,right.",
    )
    parser.add_argument(
        "--initial-qpos-source",
        choices=("first_observation", "calibration_home", "default"),
        default="first_observation",
        help="Initial solver qpos for offline replay.",
    )
    parser.add_argument(
        "--disable-target-smoothing",
        action="store_true",
        help="Disable the bimanual target pose smoother during offline replay.",
    )
    parser.add_argument(
        "--disable-realistic-joint-step",
        action="store_true",
        help="Disable realistic per-frame joint-step limits during offline replay.",
    )
    parser.add_argument(
        "--output-command-type",
        choices=("cartesian_pose", "joint_vector"),
        default="cartesian_pose",
        help="Dual-arm retargeter output type to compare.",
    )
    parser.add_argument(
        "--solve-max-iterations",
        type=int,
        default=1,
        help="Dual-arm IK solve iterations per frame. Default matches runtime.",
    )
    parser.add_argument(
        "--seed-each-frame-from-observation",
        action="store_true",
        help="Reset solver qpos from the dataset observation qpos before each frame.",
    )
    parser.add_argument("--scene-profile", default="", help="Override collision scene profile.")
    parser.add_argument("--contact-mode", default="", help="Override collision avoidance mode.")
    parser.add_argument("--solver-frequency", type=float, default=0.0, help="Override dual-arm IK Hz.")
    parser.add_argument("--ik-tuning-json", default="", help="Optional IK tuning JSON override.")
    parser.add_argument("--json-out", type=Path, default=None, help="Write full replay report JSON.")
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="Print compact JSON summary instead of human-readable text.",
    )
    parser.add_argument(
        "--max-translation-error-m",
        type=float,
        default=0.0,
        help="Fail if any side's max translation error exceeds this value. 0 disables.",
    )
    parser.add_argument(
        "--max-rotation-error-deg",
        type=float,
        default=0.0,
        help="Fail if any side's max rotation error exceeds this value. 0 disables.",
    )
    parser.add_argument(
        "--max-qpos-error-deg",
        type=float,
        default=0.0,
        help="Fail if any side's max qpos error exceeds this value. 0 disables.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    active_sides = tuple(
        side.strip().lower()
        for side in str(args.active_sides).replace("both", "left,right").split(",")
        if side.strip()
    )
    report = run_offline_tracker_replay(
        OfflineTrackerReplayConfig(
            dataset_root=args.dataset_root,
            episode=args.episode,
            start_frame=int(args.start_frame),
            max_frames=int(args.max_frames),
            active_sides=active_sides,
            use_target_smoothing=not bool(args.disable_target_smoothing),
            enforce_realistic_joint_step=not bool(args.disable_realistic_joint_step),
            initial_qpos_source=str(args.initial_qpos_source),
            output_command_type=str(args.output_command_type),
            solve_max_iterations=int(args.solve_max_iterations),
            seed_each_frame_from_observation=bool(args.seed_each_frame_from_observation),
            scene_profile=str(args.scene_profile or "") or None,
            contact_mode=str(args.contact_mode or "") or None,
            solver_frequency=float(args.solver_frequency) if float(args.solver_frequency) > 0.0 else None,
            ik_tuning_json_path=str(args.ik_tuning_json or "") or None,
        )
    )
    if args.json_out is not None:
        save_offline_tracker_report(report, args.json_out)

    exceeded, failures = report_exceeds_thresholds(
        report,
        max_translation_error_m=(
            float(args.max_translation_error_m)
            if float(args.max_translation_error_m) > 0.0
            else None
        ),
        max_rotation_error_deg=(
            float(args.max_rotation_error_deg)
            if float(args.max_rotation_error_deg) > 0.0
            else None
        ),
        max_qpos_error_deg=(
            float(args.max_qpos_error_deg)
            if float(args.max_qpos_error_deg) > 0.0
            else None
        ),
    )
    if args.summary_json:
        print(
            json.dumps(
                {
                    "dataset_root": str(report.dataset_root),
                    "episode_id": report.episode_id,
                    "frame_count": report.frame_count,
                    "summary": report.summary,
                    "threshold_failures": failures,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        _print_text_summary(report, threshold_failures=failures)
    return 2 if exceeded else 0


def _print_text_summary(report, *, threshold_failures: list[str]) -> None:
    print(f"Offline tracker replay compare: {report.dataset_root}")
    print(
        f"episode={report.episode_id} frames={report.frame_count}/{report.source_frame_count} "
        f"hz={report.nominal_hz:.3f} timing={report.timing_source}"
    )
    print(f"status_counts={report.summary.get('status_counts', {})}")
    by_side = report.summary.get("by_side", {})
    for side in report.active_sides:
        metrics = by_side.get(side, {}) if isinstance(by_side, dict) else {}
        source_translation = (
            metrics.get("source_vs_action_translation_error_m", {})
            if isinstance(metrics, dict)
            else {}
        )
        source_rotation = (
            metrics.get("source_vs_action_rotation_error_deg", {})
            if isinstance(metrics, dict)
            else {}
        )
        translation = (
            metrics.get("ik_vs_action_translation_error_m", {})
            if isinstance(metrics, dict)
            else {}
        )
        rotation = (
            metrics.get("ik_vs_action_rotation_error_deg", {})
            if isinstance(metrics, dict)
            else {}
        )
        qpos = metrics.get("qpos_max_abs_error_deg", {}) if isinstance(metrics, dict) else {}
        print(
            f"{side}: "
            f"source_pos_max={_fmt(source_translation.get('max'))}m "
            f"source_rot_max={_fmt(source_rotation.get('max'))}deg "
            f"ik_pos_mean={_fmt(translation.get('mean'))}m ik_pos_max={_fmt(translation.get('max'))}m "
            f"ik_rot_mean={_fmt(rotation.get('mean'))}deg ik_rot_max={_fmt(rotation.get('max'))}deg "
            f"qpos_max={_fmt(qpos.get('max'))}deg"
        )
    if threshold_failures:
        print("threshold_failures:")
        for failure in threshold_failures:
            print(f"  - {failure}")


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
