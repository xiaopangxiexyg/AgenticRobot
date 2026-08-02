from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from teleop_runtime.debug.evaluate_rm65_singularity_wind_tunnel import (
    WindCase,
    WindSeed,
    build_wind_cases,
    run_wind_case,
    se3_exp_local,
    select_wind_seeds,
    summarize_profile_cases,
)


def _group() -> SimpleNamespace:
    left = np.zeros((4, 6), dtype=float)
    right = np.zeros((4, 6), dtype=float)
    left[:, 2] = np.radians([5.0, 0.4, 2.0, 1.0])
    left[:, 4] = np.radians([12.0, 11.0, 0.3, 10.0])
    right[:, 2] = np.radians([3.0, 2.0, 1.0, 0.2])
    right[:, 4] = np.radians([10.0, 9.0, 8.0, 7.0])
    return SimpleNamespace(
        source_episode_index=96,
        arm_joint_targets={"left": left, "right": right},
    )


def test_se3_exp_local_matches_translation_and_rotation_limits() -> None:
    translation = se3_exp_local([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(translation[:3, 3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(translation[:3, :3], np.eye(3))

    rotation = se3_exp_local([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(
        rotation[:3, :3]
        @ np.array([1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_seed_selection_respects_other_structural_margin_and_is_deterministic() -> None:
    seeds = select_wind_seeds(
        [_group()],
        min_other_structural_abs_deg=8.0,
        holdout_modulus=4,
        split="all",
        max_seeds=0,
    )

    assert len(seeds) == 4
    left_q3 = next(
        seed
        for seed in seeds
        if seed.side == "left" and seed.structural_joint == "q3"
    )
    assert left_q3.frame_index == 1
    assert left_q3.source_abs_joint_deg == 0.4
    assert left_q3.source_other_structural_abs_deg == 11.0
    assert left_q3.split in {"tune", "heldout"}


def test_case_product_is_explicit_and_bounded() -> None:
    seed = WindSeed(96, 1, "left", "q3", 0.4, 11.0, "tune")

    cases = build_wind_cases(
        [seed],
        offsets_deg=(-0.75, 0.75),
        initial_motions=("stationary", "inward"),
        directions=("weakest",),
        direction_signs=(-1, 1),
        scenarios=("persistent_push",),
        max_cases=3,
    )

    assert len(cases) == 3
    assert len({case.key for case in cases}) == 3
    assert all("ep96:f1:left:q3" in case.key for case in cases)


class _FakePlanner:
    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }

    def reset(self, qpos_by_side):
        self.qpos = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in qpos_by_side.items()
        }
        return {"initialized": True}

    def observe_endpoint_poses_world(self, qpos_by_side):
        poses = {}
        for side, value in qpos_by_side.items():
            pose = np.eye(4, dtype=float)
            pose[0, 3] = float(np.asarray(value)[0])
            poses[side] = pose
        return poses

    def analyze_task_conditioning(self, qpos_by_side, targets, *, dt_sec):
        del qpos_by_side, targets, dt_sec
        side = {
            "task_direction_rows": np.eye(6).tolist(),
            "task_cost": np.ones(6).tolist(),
            "sigma_task_velocity_min": 0.2,
            "max_required_velocity_utilization": 0.5,
        }
        return {"sides": {"left": dict(side), "right": dict(side)}}

    def plan_pose_reference(self, targets, **kwargs):
        del kwargs
        realized = self.observe_endpoint_poses_world(self.qpos)
        return SimpleNamespace(
            valid=True,
            hold=False,
            status="tracking",
            q_ref_by_side={
                side: value.copy() for side, value in self.qpos.items()
            },
            target_endpoint_pose_world_by_side={
                side: np.asarray(value, dtype=float).copy()
                for side, value in targets.items()
            },
            realized_endpoint_pose_world_by_side=realized,
            metadata={"pose_lift": {"applied": False}},
        )


def test_continuous_wind_case_uses_fixed_pink_direction_without_dispatch() -> None:
    seed = WindSeed(96, 1, "left", "q3", 0.4, 11.0, "tune")
    case = WindCase(
        seed=seed,
        offset_deg=0.75,
        initial_motion="stationary",
        direction="weakest",
        direction_sign=1,
        scenario="persistent_push",
    )

    report = run_wind_case(
        _FakePlanner(),
        _group(),
        case,
        reference_hz=10.0,
        duration_sec=0.3,
        weighted_task_speed_per_sec=0.04,
        warmstart_joint_speed_deg_s=5.0,
        include_frames=True,
        include_constrained_feasibility=False,
    )

    assert report["status"] == "completed"
    assert report["frame_count"] == 3
    assert report["hold_count"] == 0
    assert report["branch_crossing_count"] == 0
    assert report["direction_index"] == 5
    assert len(report["frames"]) == 3

    summary = summarize_profile_cases([report])
    assert summary["completed_count"] == 1
    assert summary["case_with_hold_count"] == 0
