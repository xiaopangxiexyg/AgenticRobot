from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_motion.rm65 import RM65_POSE_LIFT_PROFILE_CHOICES
from teleop_runtime.debug.view_rm65_pose_lift_sim_replay import (
    OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME,
    REPRESENTATIVE_SUITE_0727,
    REPRESENTATIVE_SUITE_0727_NAME,
    build_parser,
    generate_trajectory,
    planner_config_from_args,
    resolved_episode_specs,
    selected_profiles,
)


def _args(*extra: str):
    return build_parser().parse_args(
        ["--dataset-root", "/tmp/dataset", "--episode", "7", *extra]
    )


def test_profile_selection_defaults_and_all_profiles() -> None:
    default = _args()
    assert selected_profiles(default) == ("rm65_pose_lift_v1_baseline",)

    explicit = _args(
        "--pose-lift-profile",
        "rm65_pose_lift_v3_bounded_continuous",
        "--pose-lift-profile",
        "rm65_pose_lift_v1_baseline",
    )
    assert selected_profiles(explicit) == (
        "rm65_pose_lift_v3_bounded_continuous",
        "rm65_pose_lift_v1_baseline",
    )

    all_profiles = _args("--all-profiles")
    assert selected_profiles(all_profiles) == RM65_POSE_LIFT_PROFILE_CHOICES


def test_representative_suite_selects_v4_then_current_better_candidate() -> None:
    args = build_parser().parse_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--representative-suite",
            REPRESENTATIVE_SUITE_0727_NAME,
        ]
    )
    assert selected_profiles(args) == (
        "rm65_pose_lift_v4_qp_constrained",
        OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME,
    )
    assert resolved_episode_specs(args) == REPRESENTATIVE_SUITE_0727
    assert [
        (spec.episode_index, spec.start_frame, spec.max_frames)
        for spec in resolved_episode_specs(args)
    ] == [
        (96, 40, 75),
        (37, 157, 62),
        (68, 0, 163),
    ]


def test_fixed_iterations_override_adaptive_solver() -> None:
    args = _args(
        "--fixed-solve-iterations",
        "5",
        "--posture-cost",
        "0.025",
        "--damping",
        "0.04",
        "--orientation-weight",
        "0.2",
        "--max-joint-velocity-rad-s",
        "1,2,3,4,5,6",
    )
    config = planner_config_from_args(
        args,
        profile_name="rm65_pose_lift_v1_baseline",
    )
    assert config.solve_min_iterations == 5
    assert config.solve_max_iterations == 5
    assert config.solve_adaptive_iterations_enabled is False
    assert config.posture_cost == 0.025
    assert config.damping == 0.04
    assert config.max_joint_velocity_rad_s == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


class _FakePlanner:
    def __init__(self) -> None:
        self.config = SimpleNamespace(reference_hz=60.0)
        self.reset_qpos = None
        self.plan_count = 0

    def reset(self, qpos_by_side):
        self.reset_qpos = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in qpos_by_side.items()
        }

    def plan_pose_reference(self, pose_by_side, **kwargs):
        del pose_by_side, kwargs
        self.plan_count += 1
        qpos = {
            "left": np.full(6, 0.1 * self.plan_count),
            "right": np.full(6, -0.1 * self.plan_count),
        }
        hold = self.plan_count == 2
        return SimpleNamespace(
            sequence=self.plan_count - 1,
            status="hold_last_safe" if hold else "tracking",
            hold=hold,
            valid=not hold,
            q_ref_by_side=qpos,
            modification_reasons=("test_hold",) if hold else (),
            metadata={
                "pose_lift": {
                    "applied": self.plan_count == 1,
                    "reason": "test",
                },
                "collision_distance_m": 0.05,
            },
        )


def test_generate_trajectory_uses_saved_qref_then_pose_references() -> None:
    initial_left = np.arange(6, dtype=float)
    initial_right = -np.arange(6, dtype=float)
    group = SimpleNamespace(
        arm_joint_targets={
            "left": np.stack((initial_left, initial_left + 0.1)),
            "right": np.stack((initial_right, initial_right - 0.1)),
        },
        source_episode_index=7,
        source_path="/tmp/episode_000007.parquet",
        frame_count=2,
    )
    references = (
        SimpleNamespace(
            sequence=0,
            timestamp=0.0,
            pose_by_side={"left": np.eye(4), "right": np.eye(4)},
            left_source_index=0,
            right_source_index=0,
            interpolation_alpha=0.0,
        ),
        SimpleNamespace(
            sequence=1,
            timestamp=1.0 / 60.0,
            pose_by_side={"left": np.eye(4), "right": np.eye(4)},
            left_source_index=0,
            right_source_index=1,
            interpolation_alpha=0.25,
        ),
        SimpleNamespace(
            sequence=2,
            timestamp=2.0 / 60.0,
            pose_by_side={"left": np.eye(4), "right": np.eye(4)},
            left_source_index=0,
            right_source_index=1,
            interpolation_alpha=0.5,
        ),
    )
    planner = _FakePlanner()

    trajectory = generate_trajectory(
        group,
        profile_name="rm65_pose_lift_v2_continuous",
        planner=planner,
        references=references,
    )

    assert np.array_equal(planner.reset_qpos["left"], initial_left)
    assert np.array_equal(planner.reset_qpos["right"], initial_right)
    assert planner.plan_count == 2
    assert len(trajectory.frames) == 2
    assert trajectory.hold_count == 1
    assert trajectory.pose_lift_applied_count == 1
    assert trajectory.status_counts == {"tracking": 1, "hold_last_safe": 1}
    assert trajectory.frames[0].source_frame_index == 1
    assert trajectory.frames[0].interpolation_alpha == 0.25
