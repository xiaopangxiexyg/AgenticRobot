from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


TASK_CONFIG = Path("sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json")


@pytest.mark.parametrize(
    ("separation_m", "clearance_m", "expected_safe"),
    (
        (0.026565366, 0.002, True),
        (0.0015, 0.002, False),
        (0.003, 0.002, True),
        (0.003, 0.005, False),
        (-0.005, 0.002, False),
        (-0.005, -0.001, False),
        (-0.0005, -0.001, True),
    ),
)
def test_distance_admission_uses_required_margin_and_retains_real_penetration(
    separation_m, clearance_m, expected_safe
):
    mujoco = pytest.importorskip("mujoco")
    from sim.mujoco.collision_safety import CollisionSafetyOracle

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody>'
        '<body name="first"><geom type="box" size=".02 .02 .02" contype="2" conaffinity="2"/></body>'
        f'<body name="second" pos="{.04 + separation_m} 0 0">'
        '<joint type="slide" axis="1 0 0"/>'
        '<geom type="box" size=".02 .02 .02" contype="2" conaffinity="2"/>'
        '</body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    oracle = CollisionSafetyOracle(mujoco, model)
    check = oracle.check(data, minimum_clearance_m=clearance_m)

    assert oracle.forbidden_pair_count == 1
    assert check.safe is expected_safe
    assert check.minimum_distance_m == pytest.approx(
        min(separation_m, max(0.0, clearance_m)), abs=1e-8
    )
    measured = oracle.check(
        data, minimum_clearance_m=clearance_m, distance_query_limit_m=0.05
    )
    assert measured.safe is expected_safe
    assert measured.minimum_distance_m == pytest.approx(separation_m, abs=1e-8)
    with pytest.raises(ValueError, match="distance_query_limit_m"):
        oracle.check(data, minimum_clearance_m=0.002, distance_query_limit_m=0.001)


@pytest.mark.parametrize(
    ("pair_class", "expected"),
    (
        ("1:2", "robot_environment"),
        ("1:4", "gripper_environment"),
        ("2:8", "robot_object"),
        ("2:2", "robot_self"),
        ("2:4", "robot_gripper_self"),
        ("4:4", "gripper_self"),
        (None, None),
        ("8:8", None),
    ),
)
def test_forbidden_collision_pair_classes_have_semantic_categories(
    pair_class: str | None,
    expected: str | None,
) -> None:
    from sim.mujoco.collision_safety import (
        CollisionCheckResult,
        forbidden_collision_category,
    )

    assert forbidden_collision_category(pair_class) == expected
    check = CollisionCheckResult(
        safe=False,
        minimum_distance_m=0.0,
        pair_class=pair_class,
    )
    assert check.collision_category == expected
    assert check.to_dict()["collision_category"] == expected


def test_clearance_recovery_only_accepts_monotonic_retreat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sim.mujoco.collision_safety import (
        CollisionCheckResult,
        CollisionSafetyPolicy,
        KinematicCollisionProjector,
    )

    projector = object.__new__(KinematicCollisionProjector)
    projector.controlled_qpos_indices = np.asarray((0,), dtype=np.int32)
    projector.policy = CollisionSafetyPolicy(sweep_max_joint_step_rad=0.1)

    def check(_source_data, controlled_qpos):
        distance = float(np.asarray(controlled_qpos)[0]) * 0.01
        return CollisionCheckResult(
            safe=distance >= projector.policy.minimum_clearance_m,
            minimum_distance_m=distance,
            body1_name="table",
            body2_name="right_gripper_left_finger",
        )

    monkeypatch.setattr(projector, "_check_candidate", check)
    recovered = projector.recover_from_clearance_violation(
        object(),
        start_qpos=np.asarray((0.0,)),
        requested_qpos=np.asarray((1.0,)),
    )
    assert recovered.status == "recovered"
    assert recovered.projection_alpha == pytest.approx(1.0)

    worsening = projector.recover_from_clearance_violation(
        object(),
        start_qpos=np.asarray((0.0,)),
        requested_qpos=np.asarray((-1.0,)),
    )
    assert worsening.status == "invalid_safe_state"
    assert worsening.projection_alpha == pytest.approx(0.0)


def test_soft_penetration_projection_can_use_hard_bound_for_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sim.mujoco.collision_safety import (
        CollisionCheckResult,
        CollisionSafetyPolicy,
        KinematicCollisionProjector,
    )

    projector = object.__new__(KinematicCollisionProjector)
    projector.controlled_qpos_indices = np.asarray((0,), dtype=np.int32)
    projector.policy = CollisionSafetyPolicy(
        minimum_clearance_m=0.002,
        hard_penetration_m=0.001,
        sweep_max_joint_step_rad=0.1,
    )

    def check(_source_data, controlled_qpos, *, minimum_clearance_m=None):
        distance = float(np.asarray(controlled_qpos)[0]) * 0.01
        threshold = (
            projector.policy.minimum_clearance_m
            if minimum_clearance_m is None
            else minimum_clearance_m
        )
        return CollisionCheckResult(
            safe=distance >= threshold,
            minimum_distance_m=distance,
            body1_name="table",
            body2_name="right_realman_to_inspire_gripper_flange",
        )

    monkeypatch.setattr(projector, "_check_candidate", check)
    result = projector.project(
        object(),
        start_qpos=np.asarray((-0.05,)),
        requested_qpos=np.asarray((1.0,)),
        minimum_clearance_m=-projector.policy.hard_penetration_m,
    )
    assert result.status == "accepted"
    assert result.check.safe
    assert result.check.minimum_distance_m == pytest.approx(0.01)


@pytest.mark.optional
def test_collision_safety_oracle_keeps_active_data_immutable() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import CubeLiftEnv

    env = CubeLiftEnv(TASK_CONFIG)
    try:
        env.reset(seed=1)
        projector = env.scene.collision_projector()
        assert projector is not None
        before_qpos = env.scene.data.qpos.copy()
        before_qvel = env.scene.data.qvel.copy()
        before_time = float(env.scene.data.time)
        start = env.scene.kinematic_controlled_qpos()
        requested = start.copy()
        right = env.scene.arm_qpos_from_kinematic_vector(requested, "right")
        right[1] += 1.0
        requested = env.scene.kinematic_target_vector(
            requested,
            {"right": right},
        )
        projector.project(
            env.scene.data,
            start_qpos=start,
            requested_qpos=requested,
        )
        np.testing.assert_array_equal(env.scene.data.qpos, before_qpos)
        np.testing.assert_array_equal(env.scene.data.qvel, before_qvel)
        assert env.scene.data.time == before_time
    finally:
        env.close()


@pytest.mark.optional
def test_scripted_downward_motion_is_projected_before_table_penetration() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import CubeLiftEnv

    env = CubeLiftEnv(TASK_CONFIG)
    action = np.asarray((0.45, 0.99, -0.99, -0.99, 1.0, -0.99, -0.99), dtype=np.float32)
    modified = False
    minimum_contact_distance = float("inf")
    try:
        env.reset(seed=2)
        for _ in range(30):
            _observation, _reward, terminated, truncated, info = env.step(action)
            safety = info["safety"]
            modified = modified or bool(safety and safety["modified"])
            for contact in env.scene.contacts():
                bodies = {contact.body1_name, contact.body2_name}
                if "table" in bodies and any(name.startswith("right_") for name in bodies):
                    minimum_contact_distance = min(
                        minimum_contact_distance, contact.distance_m
                    )
            assert not terminated
            assert not truncated
        assert modified
        assert minimum_contact_distance == float("inf")
        active = env.controller.last_active_collision_check
        assert active is not None and active.safe
    finally:
        env.close()


@pytest.mark.optional
def test_fast_and_reference_safety_projection_remain_identical() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import CubeLiftEnv, FastCubeLiftEnv

    reference = CubeLiftEnv(TASK_CONFIG)
    fast = FastCubeLiftEnv(TASK_CONFIG, use_settled_snapshot=False)
    action = np.asarray((0.45, 0.99, -0.99, -0.99, 1.0, -0.99, -0.99), dtype=np.float32)
    try:
        reference.reset(seed=3)
        fast.reset(seed=3)
        for _ in range(20):
            ref = reference.step(action)
            candidate = fast.step(action)
            np.testing.assert_allclose(candidate[0], ref[0], atol=1e-6, rtol=0.0)
            assert candidate[2:4] == ref[2:4]
            assert candidate[4]["safety_modified"] == bool(
                ref[4]["safety"] and ref[4]["safety"]["modified"]
            )
            assert candidate[4]["safety_projection_alpha"] == pytest.approx(
                1.0
                if ref[4]["safety"] is None
                else ref[4]["safety"]["projection_alpha"],
                abs=1e-8,
            )
    finally:
        fast.close()
        reference.close()


@pytest.mark.optional
def test_general_qpos_controller_holds_last_state_for_unsafe_target() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.controllers import QposController
    from sim.mujoco.tasks.cube_lift import CubeLiftEnv

    env = CubeLiftEnv(TASK_CONFIG)
    try:
        env.reset(seed=4)
        before = env.scene.get_arm_qpos("right")
        controller = QposController(env.scene)
        controller.apply(
                {
                    "right": np.asarray(
                        (
                            -2.91216875,
                            -0.08460413,
                            -1.86886070,
                            -1.83527002,
                            -1.36042304,
                            1.66282724,
                        ),
                        dtype=float,
                    )
                }
        )
        assert controller.last_safety_result is not None
        assert controller.last_safety_result.modified
        np.testing.assert_allclose(env.scene.get_arm_qpos("right"), before, atol=1e-12)
    finally:
        env.close()


@pytest.mark.optional
def test_fast_env_clearance_violation_terminates_worker_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.collision_safety import (
        CollisionCheckResult,
        CollisionProjectionResult,
    )
    from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv

    env = FastCubeLiftEnv(TASK_CONFIG)
    try:
        env.reset(seed=5)
        projector = env.kernel._collision_projector
        assert projector is not None

        def invalid_project(_self, _source_data, *, start_qpos, requested_qpos):
            check = CollisionCheckResult(
                safe=False,
                minimum_distance_m=0.0,
                body1_name="table",
                body2_name="right_gripper_left_finger",
            )
            return CollisionProjectionResult(
                status="invalid_safe_state",
                requested_qpos=np.asarray(requested_qpos).copy(),
                accepted_qpos=np.asarray(start_qpos).copy(),
                projection_alpha=0.0,
                modified=True,
                check=check,
                sweep_samples=0,
                blocking_check=check,
            )

        monkeypatch.setattr(type(projector), "project", invalid_project)
        _observation, _reward, terminated, truncated, info = env.step(
            np.asarray((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
        )
        assert terminated
        assert not truncated
        assert info["termination_reason"] == "robot_environment_clearance_violation"
        assert info["safety_projection_alpha"] == 0.0
    finally:
        env.close()


@pytest.mark.optional
def test_fast_direct_qpos_observation_keeps_collision_approved_robot_state() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv

    env = FastCubeLiftEnv(TASK_CONFIG, use_settled_snapshot=False)
    try:
        env.reset(seed=6)
        observation, _reward, terminated, truncated, _info = env.step(
            np.asarray((0.1, -0.1, 0.1, 0.0, 0.0, 0.0, 0.1), dtype=np.float32)
        )
        assert not terminated
        assert not truncated
        expected_right = env.kernel.scene.arm_qpos_from_kinematic_vector(
            env.kernel._accepted_controlled_qpos,
            env.kernel.config.controlled_side,
        )
        expected_close = env.kernel.scene.gripper_close_from_kinematic_vector(
            env.kernel._accepted_controlled_qpos,
            env.kernel.config.right_gripper_name,
        )
        np.testing.assert_allclose(observation[:6], expected_right, atol=1e-7)
        assert observation[6] == pytest.approx(expected_close, abs=1e-7)
        assert observation[6] > 0.0
    finally:
        env.close()
