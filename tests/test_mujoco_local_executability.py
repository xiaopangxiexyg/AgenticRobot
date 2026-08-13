from __future__ import annotations

import numpy as np
import pytest


def test_local_candidates_preserve_non_translation_components() -> None:
    from sim.mujoco.diagnostics.local_executability import (
        generate_local_action_candidates,
    )

    candidates = generate_local_action_candidates(
        (0.2, -0.4, 0.6, 0.7),
        translation_indices=(0, 1, 2),
        radial_scales=(0.0, 0.5),
        repair_radii=(0.25,),
        sphere_directions=0,
    )

    assert len(candidates) == 9
    assert candidates[0].label == "raw"
    assert all(candidate.policy_action[3] == pytest.approx(0.7) for candidate in candidates)
    assert candidates[1].policy_action[:3] == pytest.approx((0.0, 0.0, 0.0))
    assert candidates[2].policy_action[:3] == pytest.approx((0.1, -0.2, 0.3))


def test_local_summary_separates_ik_and_projector_failures() -> None:
    from sim.mujoco.diagnostics.local_executability import (
        LocalActionCandidate,
        LocalActionOutcome,
        summarize_local_outcomes,
    )

    raw = LocalActionCandidate(
        label="raw",
        family="raw",
        policy_action=(1.0, 0.0, 0.0, 0.5),
        radial_scale=1.0,
    )
    shorter = LocalActionCandidate(
        label="radial_scale:0.5",
        family="radial_scale",
        policy_action=(0.5, 0.0, 0.0, 0.5),
        radial_scale=0.5,
    )
    turn = LocalActionCandidate(
        label="local:r=1:d=0",
        family="local_repair",
        policy_action=(0.0, 1.0, 0.0, 0.5),
    )
    outcomes = (
        LocalActionOutcome(
            candidate=raw,
            task_action=raw.policy_action,
            ik_feasible=True,
            projector_identity=False,
            executable=False,
            projection_alpha=0.2,
            blocking_category="robot_environment",
            blocking_body_pair="link/table",
        ),
        LocalActionOutcome(
            candidate=shorter,
            task_action=shorter.policy_action,
            ik_feasible=False,
            projector_identity=True,
            executable=False,
            projection_alpha=1.0,
        ),
        LocalActionOutcome(
            candidate=turn,
            task_action=turn.policy_action,
            ik_feasible=True,
            projector_identity=True,
            executable=True,
            projection_alpha=1.0,
            objective_progress_delta=0.1,
            goal_distance_progress_m=0.002,
        ),
    )

    summary = summarize_local_outcomes(outcomes, translation_indices=(0, 1, 2))

    assert summary["raw_executable"] is False
    assert summary["executable_ratio"] == pytest.approx(1.0 / 3.0)
    assert summary["ik_infeasible_count"] == 1
    assert summary["projector_modified_count"] == 1
    assert summary["nearest_executable_distance"] == pytest.approx(np.sqrt(2.0))
    assert summary["nearest_executable_correction_angle_deg"] == pytest.approx(90.0)
    assert summary["maximum_executable_radial_scale"] is None
    assert summary["progressing_feasible_count"] == 1
    assert summary["goal_progressing_feasible_count"] == 1
    assert summary["blocking_category_counts"] == {"robot_environment": 1}


def test_local_candidate_validation_rejects_ambiguous_translation_layout() -> None:
    from sim.mujoco.diagnostics.local_executability import (
        generate_local_action_candidates,
    )

    with pytest.raises(ValueError, match="three unique"):
        generate_local_action_candidates(
            (0.0, 0.0, 0.0, 0.0),
            translation_indices=(0, 0, 2),
        )


def test_executability_probe_parser_defaults_to_exact_compatibility() -> None:
    from sim.mujoco_temp_pipeline.probe_executability import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--task-config",
            "task.json",
        ]
    )

    assert args.checkpoint.name == "checkpoint.pt"
    assert args.task_config.name == "task.json"
    assert not any(action.dest == "allow_task_config_mismatch" for action in parser._actions)
    filtered = parser.parse_args(
        [
            "--onset-category",
            "robot_environment",
            "--onset-phase",
            "post_contact",
        ]
    )
    assert filtered.onset_category == ["robot_environment"]
    assert filtered.onset_phase == ["post_contact"]


def test_pose_target_diagnostics_separate_clear_projected_and_onset_lag() -> None:
    from sim.mujoco_temp_pipeline.pose_target_diagnostics import (
        PoseTargetDiagnostics,
    )

    diagnostics = PoseTargetDiagnostics()
    diagnostics.update(
        np.asarray((0.01, 0.02, 0.03, np.nan)),
        projection_alpha=np.asarray((1.0, 0.5, 0.0, 1.0)),
        preventive_intervention=np.asarray((False, True, True, False)),
        preventive_onset=np.asarray((False, True, False, False)),
    )

    summary = diagnostics.summary()
    assert summary["pose_target_translation_error_m_mean"] == pytest.approx(0.02)
    assert summary["pose_target_translation_error_m_max"] == pytest.approx(0.03)
    assert summary["pose_target_translation_error_m_clear_mean"] == pytest.approx(0.01)
    assert summary["pose_target_translation_error_m_projected_mean"] == pytest.approx(
        0.025
    )
    assert summary["pose_target_translation_error_m_onset_mean"] == pytest.approx(0.02)
    assert summary["pose_target_lag_blocked_fraction_correlation"] > 0.9
