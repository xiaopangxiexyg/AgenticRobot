from __future__ import annotations

import numpy as np
import pytest


def test_collision_event_tracker_counts_onset_once_per_blocked_streak() -> None:
    from sim.mujoco.collision_safety import CollisionEventTracker

    tracker = CollisionEventTracker(2)
    first = tracker.observe(
        preventive_intervention=np.asarray((True, False)),
        projection_alpha=np.asarray((0.75, 0.25)),
        preventive_category=np.asarray(("robot_self", "robot_environment")),
    )
    assert first.preventive_onset.tolist() == [True, False]
    assert first.preventive_streak_steps.tolist() == [1, 0]
    assert first.preventive_blocked_fraction.tolist() == pytest.approx((0.25, 0.0))
    assert first.preventive_category.tolist() == ["robot_self", None]

    sustained = tracker.observe(
        preventive_intervention=np.asarray((True, False)),
        projection_alpha=np.asarray((0.5, 1.0)),
        preventive_category=np.asarray(("robot_self", None), dtype=object),
    )
    assert sustained.preventive_onset.tolist() == [False, False]
    assert sustained.preventive_streak_steps.tolist() == [2, 0]

    released = tracker.observe(
        preventive_intervention=np.asarray((False, False)),
        projection_alpha=np.ones(2),
    )
    assert released.preventive_streak_steps.tolist() == [0, 0]

    restarted = tracker.observe(
        preventive_intervention=np.asarray((True, False)),
        projection_alpha=np.asarray((0.9, 1.0)),
    )
    assert restarted.preventive_onset.tolist() == [True, False]


def test_collision_event_tracker_resets_done_stream_after_final_event() -> None:
    from sim.mujoco.collision_safety import CollisionEventTracker

    tracker = CollisionEventTracker(2)
    terminal = tracker.observe(
        preventive_intervention=np.asarray((True, True)),
        projection_alpha=np.asarray((0.0, 0.5)),
        actual_forbidden_contact=np.asarray((True, False)),
        actual_category=np.asarray(("robot_environment", None), dtype=object),
        hard_violation=np.asarray((True, False)),
        episode_done=np.asarray((True, False)),
    )
    assert terminal.preventive_onset.tolist() == [True, True]
    assert terminal.actual_forbidden_contact.tolist() == [True, False]
    assert terminal.actual_category.tolist() == ["robot_environment", None]
    assert terminal.hard_violation.tolist() == [True, False]

    following = tracker.observe(
        preventive_intervention=np.asarray((True, True)),
        projection_alpha=np.asarray((0.8, 0.8)),
    )
    assert following.preventive_onset.tolist() == [True, False]
    assert following.preventive_streak_steps.tolist() == [1, 2]


def test_collision_event_tracker_validates_vector_contract() -> None:
    from sim.mujoco.collision_safety import CollisionEventTracker

    tracker = CollisionEventTracker(2)
    with pytest.raises(ValueError, match="shape"):
        tracker.observe(
            preventive_intervention=np.asarray((True,)),
            projection_alpha=np.asarray((1.0, 1.0)),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tracker.observe(
            preventive_intervention=np.asarray((True, False)),
            projection_alpha=np.asarray((1.1, 1.0)),
        )


def test_forbidden_contact_guard_projection_preserves_terminal_pair() -> None:
    from sim.mujoco.deploy.contact_trace import (
        project_mujoco_forbidden_contact_guard_result,
    )

    fact = project_mujoco_forbidden_contact_guard_result(
        {
            "available": True,
            "safe": False,
            "observed_at": 12.0,
            "physics_tick": 123,
            "minimum_distance_m": -0.0015,
            "penetration_depth_m": 0.0015,
            "hard_penetration_limit_m": 0.001,
            "geom1": "right_gripper_collision",
            "geom2": "table_collision",
            "body1": "right_gripper",
            "body2": "table",
            "pair_class": "1:4",
            "collision_category": "gripper_environment",
        },
        runtime_session_id="runtime-session",
        control_domain_id="sim-domain",
        embodiment_instance_id="sim-embodiment",
        clock_domain_id="sim-clock",
    )

    assert fact.event_type == "backend_guard_result"
    assert fact.payload["status"] == "unsafe_terminal"
    assert fact.payload["diagnostics"]["geom1"] == (
        "right_gripper_collision"
    )
    assert fact.payload["diagnostics"]["penetration_depth_m"] == pytest.approx(
        0.0015
    )


@pytest.mark.optional
def test_cube_lift_info_separates_projected_and_actual_collision_facts() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        "sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json"
    )
    try:
        env.reset(seed=17)
        expected = {
            "safety_actual_forbidden_contact",
            "safety_actual_collision_category",
            "safety_actual_minimum_distance_m",
            "safety_actual_penetration_depth_m",
            "safety_actual_hard_violation",
        }
        _observation, _reward, _terminated, _truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert expected.issubset(info)
        assert isinstance(info["safety_actual_forbidden_contact"], bool)
        assert isinstance(info["safety_actual_hard_violation"], bool)
        assert info["safety_actual_penetration_depth_m"] >= 0.0
    finally:
        env.close()
