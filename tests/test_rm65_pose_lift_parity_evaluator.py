from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from robot_motion.deploy import ACTION_ARM_POSE_SLICES, DEPLOY_ACTION_DIM
from robot_motion.replay import ReplayTrajectoryGroup
from robot_motion.transforms import parse_pose_matrix
from teleop_runtime.debug.evaluate_rm65_pose_lift_parity import (
    _AdaptivePostureCandidatePlanner,
    _wrist_null_redistribution,
    compare_profile_reports,
    evaluate_group,
    rank_singular_groups,
)


@dataclass(frozen=True)
class _CandidateResult:
    valid: bool
    hold: bool
    active_sides: tuple[str, ...]
    q_ref_by_side: dict[str, np.ndarray]
    target_endpoint_pose_world_by_side: dict[str, np.ndarray]
    realized_endpoint_pose_world_by_side: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)


class _CandidatePlanner:
    collision_pair_count = 0

    def __init__(self, *, q0: float, realized_x: float) -> None:
        self.q0 = float(q0)
        self.realized_x = float(realized_x)

    def reset(self, qpos_by_side):
        self.anchor = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in qpos_by_side.items()
        }
        return {"initialized": True}

    def analyze_configuration(self, qpos_by_side, **_kwargs):
        return {
            "sides": {
                side: {
                    "sigma_min": float(value[0]),
                    "condition_number": 1.0 / max(float(value[0]), 1e-6),
                    "min_abs_q3_or_q5_deg": 10.0,
                }
                for side, value in qpos_by_side.items()
            }
        }

    def plan_pose_reference(self, *_args, **_kwargs):
        qpos = {
            side: np.asarray([self.q0, 0, 0, 0, 0, 0], dtype=float)
            for side in ("left", "right")
        }
        target = np.eye(4, dtype=float)
        target[0, 3] = 0.1
        realized = np.eye(4, dtype=float)
        realized[0, 3] = self.realized_x
        return _CandidateResult(
            valid=True,
            hold=False,
            active_sides=("left", "right"),
            q_ref_by_side=qpos,
            target_endpoint_pose_world_by_side={
                side: target.copy() for side in ("left", "right")
            },
            realized_endpoint_pose_world_by_side={
                side: realized.copy() for side in ("left", "right")
            },
        )


class _WindowCandidatePlanner(_CandidatePlanner):
    def __init__(
        self,
        samples: dict[float, tuple[float, float]],
    ) -> None:
        super().__init__(q0=0.0, realized_x=0.0)
        self.samples = dict(samples)
        self.invalid_x: set[float] = set()

    def plan_pose_reference(self, pose_by_side, **_kwargs):
        target = parse_pose_matrix(pose_by_side["left"])
        target_x = round(float(target[0, 3]), 6)
        q0, realized_x = self.samples[target_x]
        qpos = {
            side: np.asarray([q0, 0, 0, 0, 0, 0], dtype=float)
            for side in ("left", "right")
        }
        realized = np.eye(4, dtype=float)
        realized[0, 3] = realized_x
        valid = target_x not in self.invalid_x
        return _CandidateResult(
            valid=valid,
            hold=not valid,
            active_sides=("left", "right"),
            q_ref_by_side=qpos,
            target_endpoint_pose_world_by_side={
                side: target.copy() for side in ("left", "right")
            },
            realized_endpoint_pose_world_by_side={
                side: realized.copy() for side in ("left", "right")
            },
        )


def _pose_reference(x: float, timestamp: float) -> dict:
    pose = np.eye(4, dtype=float)
    pose[0, 3] = float(x)
    return {
        "pose_by_side": {
            side: pose.copy() for side in ("left", "right")
        },
        "source_timestamp": float(timestamp),
        "metadata": {},
    }


def _horizon_candidate(
    *,
    baseline_samples: dict[float, tuple[float, float]],
    alternate_samples: dict[float, tuple[float, float]],
    lock_steps: int = 1,
) -> tuple[_AdaptivePostureCandidatePlanner, _WindowCandidatePlanner]:
    baseline = _WindowCandidatePlanner(baseline_samples)
    alternate = _WindowCandidatePlanner(alternate_samples)
    shadow = _WindowCandidatePlanner(baseline_samples)
    planner = _AdaptivePostureCandidatePlanner(
        baseline,
        alternate,
        shadow,
        guard_sigma=0.04,
        sigma_tolerance=0.0,
        sigma_improvement=1e-4,
        pose_score_slack=0.05,
        max_step_slack_deg=1.0,
        characteristic_length_m=0.5,
        rollout_baseline=_WindowCandidatePlanner(baseline_samples),
        rollout_alternate=_WindowCandidatePlanner(alternate_samples),
        rollout_shadow_baseline=_WindowCandidatePlanner(baseline_samples),
        horizon_steps=2,
        lock_steps=lock_steps,
    )
    planner.reset(
        {
            side: np.asarray([0.02, 0, 0, 0, 0, 0], dtype=float)
            for side in ("left", "right")
        }
    )
    return planner, alternate


class _ExactPoseLiftPlanner:
    def reset(self, qpos_by_side):
        self.qpos = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in qpos_by_side.items()
        }
        return {"initialized": True}

    def plan_pose_reference(self, pose_by_side, **_kwargs):
        targets = {
            side: parse_pose_matrix(value)
            for side, value in pose_by_side.items()
        }
        qpos = {}
        for side, pose in targets.items():
            value = np.zeros(6, dtype=float)
            value[0] = float(pose[0, 3])
            qpos[side] = value
        self.qpos = {side: value.copy() for side, value in qpos.items()}
        return SimpleNamespace(
            status="tracking",
            valid=True,
            hold=False,
            q_ref_by_side=qpos,
            target_endpoint_pose_world_by_side=targets,
            realized_endpoint_pose_world_by_side=targets,
        )

    def analyze_configuration(
        self,
        qpos_by_side,
        *,
        characteristic_length_m,
    ):
        return {
            "sides": {
                side: {
                    "sigma_min": 0.1 + abs(float(value[0])),
                    "condition_number": 10.0,
                    "min_abs_q3_or_q5_deg": float(
                        min(abs(np.rad2deg(value[2])), abs(np.rad2deg(value[4])))
                    ),
                    "characteristic_length_m": float(characteristic_length_m),
                }
                for side, value in qpos_by_side.items()
            }
        }


def _group() -> ReplayTrajectoryGroup:
    actions = np.full((3, DEPLOY_ACTION_DIM), np.nan, dtype=np.float32)
    qpos = {
        "left": np.zeros((3, 6), dtype=float),
        "right": np.zeros((3, 6), dtype=float),
    }
    for index, x in enumerate((0.0, 0.1, 0.2)):
        for side in ("left", "right"):
            actions[index, ACTION_ARM_POSE_SLICES[side]] = np.asarray(
                [x, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
            )
            qpos[side][index, 0] = x
    return ReplayTrajectoryGroup(
        key="parity",
        label="fixture",
        source_path="fixture.parquet",
        source_episode_index=7,
        start_index=0,
        end_index=3,
        frame_count=3,
        action_hz=15.0,
        actions=actions,
        frame_times_sec=(0.0, 1.0 / 15.0, 2.0 / 15.0),
        arm_joint_targets=qpos,
        metadata={
            "action_metadata": {"pose_action_absolute_frame": "arm_base"},
        },
    )


def test_parity_evaluator_uses_qpos_init_and_saved_qref_truth() -> None:
    report = evaluate_group(
        _group(),
        planner_factory=_ExactPoseLiftPlanner,
        reference_hz=60.0,
    )

    assert report["first_frame_contract"] == "qpos_init_from_saved_accepted_qref"
    assert report["truth_contract"] == "saved_accepted_qref_not_q_ik_raw"
    assert report["reference_clock_contract"] == (
        "all_modes_resampled_to_deploy_reference_hz"
    )
    for mode in ("teacher_forced", "closed_loop", "runtime_causal"):
        values = report["modes"][mode]
        assert values["evaluated_references"] == 8
        assert values["hold_count"] == 0
        assert values["max_joint_error_deg"]["max"] < 1e-4
        assert values["requested_to_admitted_translation_m"]["max"] == 0.0
        assert values["singularity_diagnostics"]["available"] is True
        assert values["admitted_max_velocity_deg_s"]["count"] > 0
        assert (
            values["singularity_diagnostics"]["streams"]["admitted"]["left"]
            ["sigma_min"]["count"]
            > 0
        )


def test_include_frames_exposes_causal_qpos_pose_and_jacobian_evidence() -> None:
    report = evaluate_group(
        _group(),
        planner_factory=_ExactPoseLiftPlanner,
        modes=("runtime_causal",),
        reference_hz=60.0,
        include_frames=True,
    )

    frame = report["modes"]["runtime_causal"]["frames"][0]
    assert frame["timestamp"] > 0.0
    assert frame["sides"]["left"]["truth_q_deg"] is not None
    assert frame["sides"]["left"]["admitted_q_deg"] is not None
    assert (
        frame["sides"]["left"]["requested_to_admitted_translation_m"]
        == 0.0
    )
    assert frame["singularity"]["admitted"]["left"]["sigma_min"] > 0.0


def test_adaptive_posture_selector_is_guarded_and_baseline_biased() -> None:
    guarded = _AdaptivePostureCandidatePlanner(
        _CandidatePlanner(q0=0.02, realized_x=0.08),
        _CandidatePlanner(q0=0.025, realized_x=0.09),
        _CandidatePlanner(q0=0.02, realized_x=0.08),
        guard_sigma=0.04,
        sigma_tolerance=1e-6,
        sigma_improvement=1e-4,
        pose_score_slack=0.05,
        max_step_slack_deg=0.5,
        characteristic_length_m=0.5,
    )
    guarded.reset(
        {
            side: np.asarray([0.02, 0, 0, 0, 0, 0], dtype=float)
            for side in ("left", "right")
        }
    )

    selected = guarded.plan_pose_reference({})

    assert selected.metadata["adaptive_posture_candidate_selection"]["selected"] == (
        "alternate"
    )

    normal = _AdaptivePostureCandidatePlanner(
        _CandidatePlanner(q0=0.11, realized_x=0.08),
        _CandidatePlanner(q0=0.12, realized_x=0.09),
        _CandidatePlanner(q0=0.11, realized_x=0.08),
        guard_sigma=0.04,
        sigma_tolerance=1e-6,
        sigma_improvement=1e-4,
        pose_score_slack=0.05,
        max_step_slack_deg=1.0,
        characteristic_length_m=0.5,
    )
    normal.reset(
        {
            side: np.asarray([0.1, 0, 0, 0, 0, 0], dtype=float)
            for side in ("left", "right")
        }
    )

    selected = normal.plan_pose_reference({})

    assert selected.metadata["adaptive_posture_candidate_selection"]["selected"] == (
        "baseline"
    )


def test_adaptive_posture_horizon_rejects_single_step_false_improvement() -> None:
    planner, _alternate = _horizon_candidate(
        baseline_samples={0.1: (0.02, 0.08), 0.2: (0.03, 0.18)},
        alternate_samples={0.1: (0.025, 0.09), 0.2: (0.01, 0.19)},
    )
    references = (_pose_reference(0.1, 0.1), _pose_reference(0.2, 0.2))
    planner.prepare_reference_window(references)

    result = planner.plan_pose_reference(
        references[0]["pose_by_side"], source_timestamp=0.1
    )
    selection = result.metadata["adaptive_posture_candidate_selection"]

    assert selection["selected"] == "baseline"
    assert selection["reason"] == "horizon_baseline_default"
    assert selection["horizon"]["alternate"]["sigma_min"] == 0.01
    assert selection["horizon"]["baseline"]["sigma_min"] == 0.02


def test_adaptive_posture_horizon_rejects_numerical_pose_gain() -> None:
    planner, _alternate = _horizon_candidate(
        baseline_samples={0.1: (0.02, 0.08), 0.2: (0.021, 0.18)},
        alternate_samples={
            0.1: (0.02005, 0.08005),
            0.2: (0.02105, 0.18005),
        },
    )
    references = (_pose_reference(0.1, 0.1), _pose_reference(0.2, 0.2))
    planner.prepare_reference_window(references)

    result = planner.plan_pose_reference(
        references[0]["pose_by_side"], source_timestamp=0.1
    )

    assert (
        result.metadata["adaptive_posture_candidate_selection"]["selected"]
        == "baseline"
    )


def test_adaptive_posture_plan_lock_prevents_valid_candidate_flip() -> None:
    baseline_samples = {0.1: (0.02, 0.08), 0.2: (0.021, 0.18)}
    alternate_samples = {0.1: (0.025, 0.09), 0.2: (0.026, 0.19)}
    planner, alternate = _horizon_candidate(
        baseline_samples=baseline_samples,
        alternate_samples=alternate_samples,
        lock_steps=2,
    )
    references = (_pose_reference(0.1, 0.1), _pose_reference(0.2, 0.2))
    planner.prepare_reference_window(references)
    first = planner.plan_pose_reference(
        references[0]["pose_by_side"], source_timestamp=0.1
    )
    first_selection = first.metadata["adaptive_posture_candidate_selection"]
    assert first_selection["selected"] == "alternate"

    alternate.samples[0.2] = (0.019, 0.10)
    planner.prepare_reference_window((references[1],))
    second = planner.plan_pose_reference(
        references[1]["pose_by_side"], source_timestamp=0.2
    )
    second_selection = second.metadata["adaptive_posture_candidate_selection"]

    assert second_selection["selected"] == "alternate"
    assert second_selection["reason"] == "plan_lock"
    assert second_selection["plan_id"] == first_selection["plan_id"]
    assert second.q_ref_by_side["left"][0] == 0.026


def test_adaptive_posture_invalid_locked_candidate_falls_back() -> None:
    baseline_samples = {0.1: (0.02, 0.08), 0.2: (0.021, 0.18)}
    alternate_samples = {0.1: (0.025, 0.09), 0.2: (0.026, 0.19)}
    planner, _alternate = _horizon_candidate(
        baseline_samples=baseline_samples,
        alternate_samples=alternate_samples,
        lock_steps=2,
    )
    references = (_pose_reference(0.1, 0.1), _pose_reference(0.2, 0.2))
    planner.prepare_reference_window(references)
    planner.plan_pose_reference(
        references[0]["pose_by_side"], source_timestamp=0.1
    )

    queued_result = planner._committed_path[0]["result"]
    planner._committed_path[0]["result"] = _CandidateResult(
        valid=False,
        hold=True,
        active_sides=queued_result.active_sides,
        q_ref_by_side=queued_result.q_ref_by_side,
        target_endpoint_pose_world_by_side=(
            queued_result.target_endpoint_pose_world_by_side
        ),
        realized_endpoint_pose_world_by_side=(
            queued_result.realized_endpoint_pose_world_by_side
        ),
    )
    planner.prepare_reference_window((references[1],))
    result = planner.plan_pose_reference(
        references[1]["pose_by_side"], source_timestamp=0.2
    )
    selection = result.metadata["adaptive_posture_candidate_selection"]

    assert selection["selected"] == "baseline"
    assert selection["plan_lock_cancelled"] is True
    assert selection["reason"].startswith("plan_lock_cancelled:")


def test_singularity_ranking_uses_saved_qref() -> None:
    group = _group()
    group.arm_joint_targets["left"][:, 4] = np.deg2rad([8.0, 2.0, 7.0])

    ranking = rank_singular_groups([group], limit=1)

    assert ranking[0]["episode_index"] == 7
    assert ranking[0]["min_abs_q3_or_q5_deg"] == 0.0
    assert ranking[0]["sides"]["left"]["side_frames_inside_3_25deg"] == 3


def test_wrist_null_redistribution_detects_large_equivalent_branch_step() -> None:
    before = np.deg2rad([0.0, 0.0, 20.0, 10.0, 1.0, 20.0])
    after = np.deg2rad([0.0, 0.0, 20.0, 22.0, 1.5, 8.0])

    result = _wrist_null_redistribution(before, after)

    assert result["redistribution"] is True
    assert result["branch_jump"] is True
    assert result["equivalent_motion_deg"] < 1e-9


def test_profile_comparison_rejects_jacobian_margin_regression() -> None:
    baseline = evaluate_group(
        _group(),
        planner_factory=_ExactPoseLiftPlanner,
        modes=("closed_loop",),
    )
    identical = compare_profile_reports(baseline, copy.deepcopy(baseline))
    assert identical["accepted"] is True

    candidate = copy.deepcopy(baseline)
    sigma = candidate["modes"]["closed_loop"]["singularity_diagnostics"][
        "streams"
    ]["admitted"]["right"]["sigma_min"]
    sigma["min"] = 0.02

    comparison = compare_profile_reports(baseline, candidate)

    assert comparison["accepted"] is False
    assert any(
        check["metric"] == "admitted.right.sigma_min.min"
        for check in comparison["failed_checks"]
    )
