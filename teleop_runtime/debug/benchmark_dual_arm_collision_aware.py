from __future__ import annotations

import argparse
import math
import statistics
import time

import numpy as np

from ..retargeters.dual_arm_collision_aware import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    DualArmTargetContext,
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _rotation_about_z(angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(float(angle_deg))
    cos_angle = float(math.cos(angle_rad))
    sin_angle = float(math.sin(angle_rad))
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.array(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return pose


def _mean_ms(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline benchmark for the dual-arm collision-aware retargeter. "
            "It drives both wrists inward along a deterministic trajectory and "
            "reports where the frame time is spent."
        )
    )
    parser.add_argument(
        "--steps",
        type=_positive_int,
        default=20,
        help="Number of synthetic control frames to run.",
    )
    parser.add_argument(
        "--scene-profile",
        choices=[
            "dual_arm",
            "dual_arm_hand_proxy_open",
            "dual_arm_hand",
            "dual_arm_cylinder_table",
        ],
        default="dual_arm_hand_proxy_open",
        help="Dual-arm collision scene profile.",
    )
    parser.add_argument(
        "--translation-amplitude-m",
        type=float,
        default=0.14,
        help="Max inward translation applied to each wrist along desk Y.",
    )
    parser.add_argument(
        "--rotation-amplitude-deg",
        type=float,
        default=25.0,
        help="Max yaw rotation applied to each wrist.",
    )
    parser.add_argument(
        "--solve-max-iterations",
        type=_positive_int,
        default=6,
        help="Retargeter solve_max_iterations override.",
    )
    parser.add_argument(
        "--solve-translation-tolerance-m",
        type=_positive_float,
        default=5e-4,
        help="Retargeter solve_translation_tolerance_m override.",
    )
    parser.add_argument(
        "--solve-orientation-tolerance-deg",
        type=_positive_float,
        default=0.5,
        help="Retargeter solve_orientation_tolerance_deg override.",
    )
    parser.add_argument(
        "--print-frames",
        action="store_true",
        help="Print per-frame timing and error breakdown.",
    )
    parser.add_argument(
        "--profile-print-every",
        type=int,
        default=0,
        help="Forwarded to debug_profile_print_every inside the retargeter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            control_mode="absolute",
            scene_profile=args.scene_profile,
            solve_max_iterations=args.solve_max_iterations,
            solve_translation_tolerance_m=args.solve_translation_tolerance_m,
            solve_orientation_tolerance_deg=args.solve_orientation_tolerance_deg,
            debug_profile=True,
            debug_profile_print_every=max(0, int(args.profile_print_every)),
        )
    )

    base_targets = retargeter.current_world_targets()
    left_base = base_targets["left"].copy()
    right_base = base_targets["right"].copy()

    frame_times_ms: list[float] = []
    solve_calls: list[int] = []
    solve_times_ms: list[float] = []
    qp_times_ms: list[float] = []
    integrate_times_ms: list[float] = []
    min_distance_times_ms: list[float] = []
    left_translation_errors_mm: list[float] = []
    right_translation_errors_mm: list[float] = []
    left_rotation_errors_deg: list[float] = []
    right_rotation_errors_deg: list[float] = []
    statuses: dict[str, int] = {}

    for step in range(args.steps):
        alpha = 0.0
        if args.steps > 1:
            alpha = math.sin((step / (args.steps - 1)) * (math.pi * 0.5))

        left_target = left_base.copy()
        right_target = right_base.copy()
        left_target[1, 3] += float(args.translation_amplitude_m) * alpha
        right_target[1, 3] -= float(args.translation_amplitude_m) * alpha
        left_target[:3, :3] = (
            left_target[:3, :3] @ _rotation_about_z(args.rotation_amplitude_deg * alpha)[:3, :3]
        )
        right_target[:3, :3] = (
            right_target[:3, :3]
            @ _rotation_about_z(-args.rotation_amplitude_deg * alpha)[:3, :3]
        )

        context = DualArmTargetContext(
            timestamp=float(step),
            left_target=left_target,
            right_target=right_target,
            left_cartesian=None,
            right_cartesian=None,
            left_input_skipped=None,
            right_input_skipped=None,
        )
        frame_start = time.perf_counter()
        command = retargeter.build_command_for_context(context)
        frame_ms = (time.perf_counter() - frame_start) * 1000.0
        frame_times_ms.append(frame_ms)

        status = str(command.metadata.get("collision_avoidance_status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1

        profile = command.metadata.get("ik_debug_profile", {})
        if isinstance(profile, dict):
            solve_calls.append(int(profile.get("solve_calls", 0)))
            solve_times_ms.append(float(profile.get("solve_total_ms", 0.0)))
            qp_times_ms.append(float(profile.get("solve_qp_ms", 0.0)))
            integrate_times_ms.append(float(profile.get("solve_integrate_ms", 0.0)))
            min_distance_times_ms.append(float(profile.get("min_distance_total_ms", 0.0)))

        pose_error = command.metadata.get("ik_solve_pose_error", {})
        left_error = pose_error.get("left", {}) if isinstance(pose_error, dict) else {}
        right_error = pose_error.get("right", {}) if isinstance(pose_error, dict) else {}
        left_translation_errors_mm.append(
            float(left_error.get("translation_m", 0.0)) * 1000.0
        )
        right_translation_errors_mm.append(
            float(right_error.get("translation_m", 0.0)) * 1000.0
        )
        left_rotation_errors_deg.append(float(left_error.get("rotation_deg", 0.0)))
        right_rotation_errors_deg.append(float(right_error.get("rotation_deg", 0.0)))

        if args.print_frames:
            print(
                f"frame={step:03d} "
                f"total_ms={frame_ms:.3f} "
                f"solve_calls={solve_calls[-1] if solve_calls else 0} "
                f"solve_ms={solve_times_ms[-1] if solve_times_ms else 0.0:.3f} "
                f"qp_ms={qp_times_ms[-1] if qp_times_ms else 0.0:.3f} "
                f"integrate_ms={integrate_times_ms[-1] if integrate_times_ms else 0.0:.3f} "
                f"min_dist_ms={min_distance_times_ms[-1] if min_distance_times_ms else 0.0:.3f} "
                f"status={status}"
            )

    print("Benchmark Summary")
    print(
        f"  scene_profile={args.scene_profile} steps={args.steps} "
        f"solve_max_iterations={args.solve_max_iterations} "
        f"translation_tolerance_m={args.solve_translation_tolerance_m}"
    )
    print(
        f"  frame_ms mean={_mean_ms(frame_times_ms):.3f} "
        f"p95={float(np.percentile(frame_times_ms, 95)):.3f} "
        f"max={max(frame_times_ms):.3f}"
    )
    print(
        f"  solve_calls mean={_mean_ms([float(value) for value in solve_calls]):.3f} "
        f"solve_ms mean={_mean_ms(solve_times_ms):.3f} "
        f"qp_ms mean={_mean_ms(qp_times_ms):.3f} "
        f"integrate_ms mean={_mean_ms(integrate_times_ms):.3f}"
    )
    print(
        f"  min_distance_ms mean={_mean_ms(min_distance_times_ms):.3f} "
        f"left_err_mm mean={_mean_ms(left_translation_errors_mm):.3f} "
        f"right_err_mm mean={_mean_ms(right_translation_errors_mm):.3f}"
    )
    print(
        f"  left_rot_err_deg mean={_mean_ms(left_rotation_errors_deg):.3f} "
        f"right_rot_err_deg mean={_mean_ms(right_rotation_errors_deg):.3f}"
    )
    print(f"  statuses={statuses}")


if __name__ == "__main__":
    main()
