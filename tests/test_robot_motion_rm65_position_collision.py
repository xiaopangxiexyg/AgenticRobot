from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from robot_motion.rm65 import (
    RM65_POSE_LIFT_V2_CONTINUOUS_NAME,
    RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME,
    RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
    RM65_POSE_LIFT_V5_SIGMA_ESCAPE_NAME,
    RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH_NAME,
    RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE_NAME,
    RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER_NAME,
    RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD_NAME,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
    build_rm65_dual_arm_scene_profile,
    load_world_from_arm_base_transforms,
    load_world_from_fixed_link_transform,
)
from robot_motion.rm65.position_collision import _rm65_jacobian_metrics


def test_jacobian_metrics_normalize_linear_rows_by_characteristic_length() -> None:
    jacobian = np.diag([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

    metrics = _rm65_jacobian_metrics(
        jacobian,
        characteristic_length_m=0.5,
    )

    np.testing.assert_allclose(metrics["singular_values"], np.ones(6))
    assert metrics["sigma_min"] == pytest.approx(1.0)
    assert metrics["condition_number"] == pytest.approx(1.0)
    assert metrics["jacobian_row_order"] == "linear_xyz_then_angular_xyz"


class _FakeModel:
    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self.lowerPositionLimit = np.asarray(lower, dtype=float).copy()
        self.upperPositionLimit = np.asarray(upper, dtype=float).copy()


class _FakeRobot:
    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self.model = _FakeModel(lower, upper)


class _FakeDualSolver:
    def __init__(
        self,
        *,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        distance: float | None = 0.10,
        wrist_null_orientation_gain: float = 0.0,
    ) -> None:
        self.q = np.zeros(12, dtype=float)
        self.solve_output = np.zeros(12, dtype=float)
        self.solve_outputs: list[np.ndarray] = []
        self.solve_error: BaseException | None = None
        self.distance: float | None | object = distance
        self.solve_targets: list[list[np.ndarray]] = []
        self.distance_q: list[np.ndarray] = []
        self.wrist_null_orientation_gain = float(wrist_null_orientation_gain)
        self.configured_velocity_limit: np.ndarray | None = None
        self.configured_acceleration_limit: np.ndarray | None = None
        self.previous_velocities: list[np.ndarray] = []
        self.integration_dt: float | None = None
        self.admitted_motion_references: list[
            tuple[np.ndarray, np.ndarray, float]
        ] = []
        self.task_conditioning_calls: list[dict[str, object]] = []
        self.sigma_escape_configuration: dict[str, float] | None = None
        self.structural_branch_configuration: dict[str, object] | None = None
        self.motion_state_reset_count = 0
        self.robot = _FakeRobot(
            np.full(12, -1.0) if lower is None else lower,
            np.full(12, 1.0) if upper is None else upper,
        )

    @property
    def current_q(self) -> np.ndarray:
        return self.q.copy()

    def set_configuration_q(self, qpos: np.ndarray) -> None:
        self.q = np.asarray(qpos, dtype=float).reshape(12).copy()

    def solve(self, *, targets: list[np.ndarray]) -> np.ndarray:
        self.solve_targets.append([np.asarray(target, dtype=float).copy() for target in targets])
        if self.solve_error is not None:
            raise self.solve_error
        output = (
            self.solve_outputs[len(self.solve_targets) - 1]
            if len(self.solve_outputs) >= len(self.solve_targets)
            else self.solve_output
        )
        self.q = np.asarray(output, dtype=float).reshape(12).copy()
        return self.q.copy()

    def configure_velocity_limit(self, values: np.ndarray) -> None:
        self.configured_velocity_limit = np.asarray(values, dtype=float).copy()

    def configure_acceleration_limit(self, values: np.ndarray) -> None:
        self.configured_acceleration_limit = np.asarray(values, dtype=float).copy()

    def set_last_integration_velocity(self, values: np.ndarray) -> None:
        self.previous_velocities.append(np.asarray(values, dtype=float).copy())

    def set_integration_dt(self, dt_sec: float) -> None:
        self.integration_dt = float(dt_sec)

    def configure_admitted_reference_motion_limit(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray | None,
    ) -> None:
        self.configured_velocity_limit = np.asarray(velocity, dtype=float).copy()
        self.configured_acceleration_limit = (
            None
            if acceleration is None
            else np.asarray(acceleration, dtype=float).copy()
        )

    def set_admitted_reference_motion(
        self,
        anchor: np.ndarray,
        velocity: np.ndarray,
        dt_sec: float,
    ) -> None:
        self.admitted_motion_references.append(
            (
                np.asarray(anchor, dtype=float).copy(),
                np.asarray(velocity, dtype=float).copy(),
                float(dt_sec),
            )
        )

    def configure_normalized_sigma_escape_task(
        self,
        *,
        cost: float,
        guard_sigma: float,
        rate_per_sec: float,
    ) -> None:
        self.sigma_escape_configuration = {
            "cost": float(cost),
            "guard_sigma": float(guard_sigma),
            "rate_per_sec": float(rate_per_sec),
        }

    def configure_structural_branch_guard(
        self,
        *,
        guard_sigma: float,
        margin_deg: float,
        release_sigma: float,
        constraint_mode: str = "hard_margin",
        barrier_gain: float = 0.0,
        barrier_activation_sigma: float = 0.0,
        prediction_horizon_sec: float = 0.0,
    ) -> None:
        self.structural_branch_configuration = {
            "guard_sigma": float(guard_sigma),
            "margin_deg": float(margin_deg),
            "release_sigma": float(release_sigma),
            "constraint_mode": str(constraint_mode),
            "barrier_gain": float(barrier_gain),
            "barrier_activation_sigma": float(barrier_activation_sigma),
            "prediction_horizon_sec": float(prediction_horizon_sec),
        }

    def reset_admitted_reference_motion_state(self) -> None:
        self.motion_state_reset_count += 1

    def compute_min_collision_distance(self, q: np.ndarray) -> float | None:
        qpos = np.asarray(q, dtype=float).reshape(12).copy()
        self.distance_q.append(qpos)
        value = self.distance(qpos) if callable(self.distance) else self.distance
        return None if value is None else float(value)

    def compute_frame_pose(self, frame_name: str, q: np.ndarray) -> np.ndarray:
        qpos = np.asarray(q, dtype=float).reshape(12)
        pose = np.eye(4, dtype=float)
        if frame_name == "left_endpoint":
            pose[0, 3] = qpos[0]
            wrist_start = 0
        elif frame_name == "right_endpoint":
            pose[0, 3] = qpos[6]
            wrist_start = 6
        else:
            raise ValueError(frame_name)
        wrist_angle = (
            qpos[wrist_start + 3]
            + qpos[wrist_start + 5]
            + self.wrist_null_orientation_gain
            * (qpos[wrist_start + 3] - qpos[wrist_start + 5])
        )
        pose[:3, :3] = R.from_rotvec([0.0, 0.0, wrist_angle]).as_matrix()
        return pose

    def compute_frame_jacobian(self, frame_name: str, q: np.ndarray) -> np.ndarray:
        start = 0 if frame_name == "left_endpoint" else 6
        jacobian = np.zeros((6, 12), dtype=float)
        jacobian[:, start : start + 6] = np.diag(
            [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
        )
        return jacobian

    def compute_frame_task_conditioning(
        self,
        targets: list[np.ndarray],
        *,
        q: np.ndarray,
        joint_indices_by_target: list[np.ndarray],
        joint_velocity_limit_rad_s: np.ndarray,
        dt_sec: float,
    ) -> list[dict[str, object]]:
        self.task_conditioning_calls.append(
            {
                "targets": [
                    np.asarray(target, dtype=float).copy() for target in targets
                ],
                "q": np.asarray(q, dtype=float).copy(),
                "joint_indices_by_target": [
                    np.asarray(indices, dtype=int).copy()
                    for indices in joint_indices_by_target
                ],
                "joint_velocity_limit_rad_s": np.asarray(
                    joint_velocity_limit_rad_s,
                    dtype=float,
                ).copy(),
                "dt_sec": float(dt_sec),
            }
        )
        return [
            {
                "sigma_task_velocity_min": 0.1 + side_index,
                "condition_number_task_velocity": 10.0 + side_index,
                "max_required_velocity_utilization": 0.5 + side_index,
            }
            for side_index in range(2)
        ]


def _identity_pair() -> dict[str, np.ndarray]:
    return {"left": np.eye(4, dtype=float), "right": np.eye(4, dtype=float)}


def _bundle(solver: _FakeDualSolver, *, collision_pairs: tuple = (("a", "b"),)):
    profile = SimpleNamespace(
        name="fake_dual_scene",
        urdf_path="unused.urdf",
        left_frame_name="left_endpoint",
        right_frame_name="right_endpoint",
        collision_pairs=collision_pairs,
        solution_layout="auto",
    )
    return SimpleNamespace(solver=solver, scene_profile=profile)


def _planner(
    solver: _FakeDualSolver,
    *,
    config: RM65PositionCollisionPlannerConfig | None = None,
) -> RM65PositionCollisionPlanner:
    return RM65PositionCollisionPlanner(
        config
        or RM65PositionCollisionPlannerConfig(
            reference_hz=45.0,
            max_joint_velocity_rad_s=(0.45,) * 6,
        ),
        endpoint_to_tool_by_side=_identity_pair(),
        world_from_arm_base_by_side=_identity_pair(),
        bundle=_bundle(solver),
    )


def _target(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = (x, y, z)
    return pose


def test_task_conditioning_is_observational_and_uses_formal_velocity_limits() -> None:
    solver = _FakeDualSolver()
    solver.q = np.linspace(0.0, 0.11, 12)
    planner = _planner(solver)
    original_solver_q = solver.q.copy()
    qpos = {
        "left": np.linspace(0.1, 0.6, 6),
        "right": np.linspace(-0.1, -0.6, 6),
    }

    diagnostics = planner.analyze_task_conditioning(
        qpos,
        {"left": _target(0.1), "right": _target(-0.1)},
        dt_sec=0.02,
    )

    assert diagnostics["schema"] == "robot_motion.rm65.task_conditioning/v1"
    assert diagnostics["observational_only"] is True
    assert diagnostics["control_authority"] == "none"
    assert diagnostics["sides"]["left"]["sigma_task_velocity_min"] == pytest.approx(
        0.1
    )
    assert diagnostics["sides"]["right"]["sigma_task_velocity_min"] == pytest.approx(
        1.1
    )
    np.testing.assert_allclose(solver.q, original_solver_q)
    call = solver.task_conditioning_calls[-1]
    np.testing.assert_allclose(call["q"][:6], qpos["left"])
    np.testing.assert_allclose(call["q"][6:], qpos["right"])
    np.testing.assert_allclose(
        call["joint_velocity_limit_rad_s"],
        np.full(12, 0.45),
    )
    np.testing.assert_array_equal(call["joint_indices_by_target"][0], np.arange(6))
    np.testing.assert_array_equal(
        call["joint_indices_by_target"][1], np.arange(6, 12)
    )


def test_endpoint_fk_observer_uses_requested_qpos_without_advancing_solver() -> None:
    solver = _FakeDualSolver()
    solver.q = np.linspace(-0.2, 0.2, 12)
    planner = _planner(solver)
    original_solver_q = solver.q.copy()
    qpos = {
        "left": np.array([0.12, 0.0, 0.0, 0.2, 0.0, -0.1]),
        "right": np.array([-0.34, 0.0, 0.0, -0.3, 0.0, 0.15]),
    }

    poses = planner.observe_endpoint_poses_world(qpos)

    assert poses["left"][0, 3] == pytest.approx(0.12)
    assert poses["right"][0, 3] == pytest.approx(-0.34)
    np.testing.assert_allclose(solver.q, original_solver_q)
    with pytest.raises(ValueError, match="both arms"):
        planner.observe_endpoint_poses_world({"left": qpos["left"]})


def test_resolve_target_endpoint_pose_world_matches_plan_input_without_state_change() -> None:
    solver = _FakeDualSolver()
    solver.q = np.linspace(-0.2, 0.2, 12)
    planner = _planner(solver)
    original_solver_q = solver.q.copy()
    targets = {"left": _target(0.12), "right": _target(-0.34)}

    resolved = planner.resolve_target_endpoint_pose_world(
        targets,
        target_reference_frame="arm_base",
        target_controlled_frame="tool_tcp",
    )

    np.testing.assert_allclose(resolved["left"], targets["left"])
    np.testing.assert_allclose(resolved["right"], targets["right"])
    np.testing.assert_allclose(solver.q, original_solver_q)


def test_scene_urdf_base_transforms_include_mount_translation_and_rotation() -> None:
    profile = build_rm65_dual_arm_scene_profile("dual_arm_hand_proxy_open")

    transforms = load_world_from_arm_base_transforms(profile)

    np.testing.assert_allclose(transforms["left"][:3, 3], [0.0, -0.2625, 0.0])
    np.testing.assert_allclose(transforms["right"][:3, 3], [0.0, 0.2625, 0.0])
    np.testing.assert_allclose(
        transforms["left"][:3, :3],
        R.from_euler("z", -np.pi / 2.0).as_matrix(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transforms["right"][:3, :3],
        R.from_euler("z", np.pi / 2.0).as_matrix(),
        atol=1e-12,
    )
    assert not np.allclose(transforms["left"][:3, 3], profile.left_pose_transform[:3, 3])


def test_scene_urdf_fixed_link_transform_resolves_exact_chain(tmp_path) -> None:
    urdf = tmp_path / "fixed-chain.urdf"
    urdf.write_text(
        """<robot name="fixed-chain">
  <link name="world"/><link name="table"/><link name="station"/>
  <joint name="world_to_table" type="fixed">
    <parent link="world"/><child link="table"/>
    <origin xyz="1 2 3" rpy="0 0 0"/>
  </joint>
  <joint name="table_to_station" type="fixed">
    <parent link="table"/><child link="station"/>
    <origin xyz="0.5 0 0" rpy="0 0 3.141592653589793"/>
  </joint>
</robot>""",
        encoding="utf-8",
    )

    world_from_station = load_world_from_fixed_link_transform(
        urdf,
        link_name="station",
    )

    np.testing.assert_allclose(world_from_station[:3, 3], [1.5, 2.0, 3.0])
    np.testing.assert_allclose(
        world_from_station[:3, :3],
        R.from_euler("z", np.pi).as_matrix(),
        atol=1e-12,
    )


def test_scene_urdf_fixed_link_transform_rejects_invalid_chains(tmp_path) -> None:
    moving = tmp_path / "moving.urdf"
    moving.write_text(
        """<robot name="moving">
  <link name="world"/><link name="station"/>
  <joint name="moving_station" type="revolute">
    <parent link="world"/><child link="station"/>
  </joint>
</robot>""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="is not fixed"):
        load_world_from_fixed_link_transform(moving, link_name="station")
    with pytest.raises(ValueError, match="missing link"):
        load_world_from_fixed_link_transform(moving, link_name="absent")

    cyclic = tmp_path / "cyclic.urdf"
    cyclic.write_text(
        """<robot name="cyclic">
  <link name="world"/><link name="first"/><link name="second"/>
  <joint name="second_to_first" type="fixed">
    <parent link="second"/><child link="first"/>
  </joint>
  <joint name="first_to_second" type="fixed">
    <parent link="first"/><child link="second"/>
  </joint>
</robot>""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Cycle detected"):
        load_world_from_fixed_link_transform(cyclic, link_name="first")


def test_collision_epsilon_cannot_remove_the_configured_minimum_margin() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        RM65PositionCollisionPlannerConfig(
            collision_min_distance_m=0.01,
            collision_distance_epsilon_m=0.02,
        )


def test_iterative_solve_policy_runs_shared_minimum_and_reports_pose_error() -> None:
    solver = _FakeDualSolver()
    planner = _planner(
        solver,
        config=RM65PositionCollisionPlannerConfig(
            reference_hz=60.0,
            max_joint_velocity_rad_s=(6.0,) * 6,
            damping=0.03,
            solve_min_iterations=2,
            solve_max_iterations=2,
        ),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})

    result = planner.plan_pose_reference(
        {"left": _target(), "right": _target()}
    )

    assert len(solver.solve_targets) == 2
    assert result.metadata["ik_solve_iterations"] == 2
    assert result.metadata["ik_solve_iteration_min"] == 2
    assert result.metadata["ik_solve_iteration_max"] == 2
    assert result.metadata["ik_solve_iteration_stop_reason"] == "pose_tolerance"
    assert result.metadata["ik_damping"] == pytest.approx(0.03)
    assert result.metadata["ik_solve_pose_error"]["left"]["translation_m"] == 0.0


def test_iterative_solve_policy_rejects_invalid_iteration_range() -> None:
    with pytest.raises(ValueError, match="solve_min_iterations"):
        RM65PositionCollisionPlannerConfig(
            solve_min_iterations=3,
            solve_max_iterations=2,
        )


def test_v4_configures_qp_velocity_limit_and_commits_only_admitted_velocity() -> None:
    solver = _FakeDualSolver()
    solver.solve_output = np.array(
        [0.02, 0.0, 0.0, 0.0, 0.0, 0.0] * 2,
        dtype=float,
    )
    velocity = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
    acceleration = (6.0, 7.0, 8.0, 9.0, 10.0, 11.0)
    planner = _planner(
        solver,
        config=RM65PositionCollisionPlannerConfig(
            reference_hz=50.0,
            max_joint_velocity_rad_s=velocity,
            max_joint_acceleration_rad_s2=acceleration,
            solve_min_iterations=2,
            solve_max_iterations=4,
            pose_lift_profile=RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
        ),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})

    result = planner.plan_pose_reference(
        {"left": _target(0.02), "right": _target(0.02)}
    )

    assert len(solver.solve_targets) == 2
    assert solver.integration_dt is None
    np.testing.assert_allclose(solver.configured_velocity_limit[:6], velocity)
    np.testing.assert_allclose(solver.configured_velocity_limit[6:12], velocity)
    np.testing.assert_allclose(solver.configured_acceleration_limit[:6], acceleration)
    np.testing.assert_allclose(solver.configured_acceleration_limit[6:12], acceleration)
    np.testing.assert_allclose(
        solver.admitted_motion_references[-1][1],
        np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0] * 2),
    )
    assert solver.admitted_motion_references[-1][2] == pytest.approx(0.02)
    assert result.metadata["ik_solve_iteration_mode"] == "fixed_max"
    assert result.metadata["ik_solve_iteration_requested_max"] == 4
    assert result.metadata["ik_solve_iteration_max"] == 4
    assert result.metadata["qp_motion_constraints"]["velocity_limit_enabled"] is True
    assert result.metadata["qp_motion_constraints"]["acceleration_limit_enabled"] is True


def test_v5_sigma_escape_is_explicit_soft_metadata_and_v4_stays_disabled() -> None:
    v4_solver = _FakeDualSolver()
    v4_planner = _planner(
        v4_solver,
        config=RM65PositionCollisionPlannerConfig(
            pose_lift_profile=RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
        ),
    )
    v4_planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    v4_result = v4_planner.plan_pose_reference(
        {"left": _target(0.0), "right": _target(0.0)}
    )

    assert v4_solver.sigma_escape_configuration is None
    assert v4_result.metadata["qp_motion_constraints"]["sigma_escape_enabled"] is False

    v5_solver = _FakeDualSolver()
    v5_planner = _planner(
        v5_solver,
        config=RM65PositionCollisionPlannerConfig(
            pose_lift_profile=RM65_POSE_LIFT_V5_SIGMA_ESCAPE_NAME,
        ),
    )
    v5_planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    v5_result = v5_planner.plan_pose_reference(
        {"left": _target(0.0), "right": _target(0.0)}
    )

    assert v5_solver.sigma_escape_configuration == pytest.approx(
        {"cost": 0.15, "guard_sigma": 0.04, "rate_per_sec": 0.03}
    )
    constraints = v5_result.metadata["qp_motion_constraints"]
    assert constraints["sigma_escape_enabled"] is True
    assert constraints["sigma_escape_authority"].startswith("soft_objective_only")


def test_v6_structural_branch_guard_is_explicit_offline_metadata() -> None:
    solver = _FakeDualSolver()
    planner = _planner(
        solver,
        config=RM65PositionCollisionPlannerConfig(
            pose_lift_profile=RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH_NAME,
        ),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    result = planner.plan_pose_reference(
        {"left": _target(0.0), "right": _target(0.0)}
    )

    assert solver.structural_branch_configuration == {
        "guard_sigma": pytest.approx(0.04),
        "margin_deg": pytest.approx(0.75),
        "release_sigma": pytest.approx(0.05),
        "constraint_mode": "hard_margin",
        "barrier_gain": pytest.approx(0.0),
        "barrier_activation_sigma": pytest.approx(0.0),
        "prediction_horizon_sec": pytest.approx(0.0),
    }
    constraints = result.metadata["qp_motion_constraints"]
    assert constraints["structural_branch_guard_enabled"] is True
    assert constraints["structural_branch_guard_margin_deg"] == pytest.approx(0.75)
    assert solver.motion_state_reset_count == 1


def test_v9_configures_predictive_smooth_branch_barrier_without_enabling_hardware() -> None:
    solver = _FakeDualSolver()
    planner = _planner(
        solver,
        config=RM65PositionCollisionPlannerConfig(
            pose_lift_profile=RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER_NAME,
        ),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    result = planner.plan_pose_reference(
        {"left": _target(0.0), "right": _target(0.0)}
    )

    assert solver.structural_branch_configuration == {
        "guard_sigma": pytest.approx(0.04),
        "margin_deg": pytest.approx(0.75),
        "release_sigma": pytest.approx(0.05),
        "constraint_mode": "hybrid_deep_barrier",
        "barrier_gain": pytest.approx(6.0),
        "barrier_activation_sigma": pytest.approx(0.01),
        "prediction_horizon_sec": pytest.approx(0.15),
    }
    constraints = result.metadata["qp_motion_constraints"]
    assert (
        constraints["structural_branch_constraint_mode"]
        == "hybrid_deep_barrier"
    )
    assert constraints["structural_branch_barrier_gain"] == pytest.approx(6.0)
    assert constraints[
        "structural_branch_barrier_activation_sigma"
    ] == pytest.approx(0.01)
    assert constraints["structural_branch_prediction_horizon_sec"] == pytest.approx(
        0.15
    )


def test_v10_shapes_only_overdemanded_pink_direction_before_offline_qp() -> None:
    solver = _FakeDualSolver()

    def directional_metrics(
        targets: list[np.ndarray],
        *,
        q: np.ndarray,
        joint_indices_by_target: list[np.ndarray],
        joint_velocity_limit_rad_s: np.ndarray,
        dt_sec: float,
    ) -> list[dict[str, object]]:
        del targets, q, joint_indices_by_target, joint_velocity_limit_rad_s, dt_sec
        metrics = {
            "task_error_local": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            "task_cost": [1.0] * 6,
            "task_direction_rows": np.eye(6).tolist(),
            "joint_utilization_direction_rows": np.eye(6).tolist(),
            "singular_values_task_velocity": [1.0] * 6,
            "directional_demand": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "directional_required_velocity_utilization": [
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "joint_velocity_limit_rad_s": [1.0] * 6,
            "max_required_velocity_utilization": 2.0,
            "sigma_task_velocity_min": 0.02,
        }
        return [dict(metrics), dict(metrics)]

    solver.compute_frame_task_conditioning = directional_metrics
    planner = _planner(
        solver,
        config=RM65PositionCollisionPlannerConfig(
            solve_min_iterations=1,
            solve_max_iterations=1,
            pose_lift_profile=(
                RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD_NAME
            ),
        ),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})

    result = planner.plan_pose_reference(
        {"left": _target(0.1), "right": _target(0.1)}
    )

    assert solver.solve_targets[0][0][0] == pytest.approx(0.0425)
    assert solver.solve_targets[0][1][0] == pytest.approx(0.0425)
    guard = result.metadata["qp_motion_constraints"]["directional_task_guard"]
    assert guard["enabled"] is True
    assert guard["reason"] == "directionally_shaped"
    assert guard["sides"]["left"]["transition"] == "entered"
    assert guard["sides"]["left"]["minimum_direction_scale"] == pytest.approx(
        0.425
    )
    assert (
        result.metadata["qp_motion_constraints"]["structural_branch_constraint_mode"]
        == "barrier"
    )


def test_v8_selects_exact_nonlinear_best_candidate_instead_of_last_iteration() -> None:
    solver = _FakeDualSolver()
    solver.solve_outputs = [
        np.array([0.01, 0, 0, 0, 0, 0] * 2, dtype=float),
        np.array([0.03, 0, 0, 0, 0, 0] * 2, dtype=float),
        np.array([0.02, 0, 0, 0, 0, 0] * 2, dtype=float),
        np.array([0.015, 0, 0, 0, 0, 0] * 2, dtype=float),
    ]
    planner = _planner(
        solver,
        config=RM65PositionCollisionPlannerConfig(
            reference_hz=60.0,
            max_joint_velocity_rad_s=(6.0,) * 6,
            solve_min_iterations=4,
            solve_max_iterations=4,
            pose_lift_profile=(
                RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE_NAME
            ),
        ),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})

    result = planner.plan_pose_reference(
        {"left": _target(0.04), "right": _target(0.04)}
    )

    np.testing.assert_allclose(
        result.q_ref_by_side["left"],
        np.array([0.03, 0, 0, 0, 0, 0], dtype=float),
    )
    selection = result.metadata["qp_motion_constraints"][
        "nonlinear_best_so_far"
    ]
    assert selection["selected_iteration"] == 2
    assert len(selection["candidates"]) == 3
    assert (
        result.metadata["ik_solve_iteration_stop_reason"]
        == "max_iterations:best_so_far_iteration_2"
    )


def test_reset_requires_complete_dual_qpos_and_fail_closes_unavailable_distance() -> None:
    solver = _FakeDualSolver()
    planner = _planner(solver)

    with pytest.raises(ValueError, match="both arms"):
        planner.reset({"left": np.zeros(6)})

    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    assert planner.initialized is True
    solver.distance = None
    with pytest.raises(ValueError, match="not certifiably safe"):
        planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    assert planner.initialized is False


def test_pose_reference_uses_explicit_base_and_tool_transforms_without_smoothing() -> None:
    solver = _FakeDualSolver()
    world_from_base = _identity_pair()
    world_from_base["left"][:3, 3] = (1.0, 2.0, 3.0)
    endpoint_to_tool = _identity_pair()
    endpoint_to_tool["left"][2, 3] = 0.10
    planner = RM65PositionCollisionPlanner(
        RM65PositionCollisionPlannerConfig(
            reference_hz=45.0,
            max_joint_velocity_rad_s=(45.0,) * 6,
        ),
        endpoint_to_tool_by_side=endpoint_to_tool,
        world_from_arm_base_by_side=world_from_base,
        bundle=_bundle(solver),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})

    first = planner.plan_pose_reference({"left": _target(0.2, 0.3, 0.4)})
    planner.plan_pose_reference({"left": _target(0.9, 0.0, 0.0)})

    np.testing.assert_allclose(solver.solve_targets[0][0][:3], [1.2, 2.3, 3.3])
    # The second target is passed directly, not EMA/deadband filtered with the first.
    np.testing.assert_allclose(solver.solve_targets[1][0][:3], [1.9, 2.0, 2.9])
    assert first.metadata["temporal_smoothing_applied"] is False
    assert first.metadata["target_pose_smoothing"] == "none"
    assert first.metadata["output_qpos_smoothing"] == "none"
    assert first.metadata["segment_collision_certified"] is False
    assert first.metadata["collision_check_scope"] == "reference_limited_qpos_endpoint_only"


def test_absolute_and_per_reference_step_limits_are_applied_before_collision_recheck() -> None:
    lower = np.full(12, -0.02, dtype=float)
    upper = np.full(12, 0.02, dtype=float)
    solver = _FakeDualSolver(lower=lower, upper=upper)
    planner = _planner(solver)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    solver.solve_output[0] = 0.50

    result = planner.plan_pose_reference({"left": _target(0.2)})

    assert result.status == "limited"
    assert result.q_ik_raw_by_side["left"][0] == pytest.approx(0.50)
    # 0.45 rad/s / 45 Hz = 0.01 rad, stricter than the effective 0.02-rad bound.
    assert result.q_ref_by_side["left"][0] == pytest.approx(0.01)
    assert "joint_position_limited:left" in result.modification_reasons
    assert "joint_step_limited:left" in result.modification_reasons
    np.testing.assert_allclose(solver.distance_q[-1][0], 0.01)
    assert result.metadata["joint_limit_order"] == (
        "hard_joint_bounds_intersect_per_reference_step_then_collision_recheck"
    )


def test_collision_unsafe_limited_candidate_backtracks_joint_step() -> None:
    solver = _FakeDualSolver()
    solver.distance = lambda q: 0.02 if q[0] > 0.006 else 0.05
    planner = _planner(solver)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    solver.solve_output[0] = 0.01

    result = planner.plan_pose_reference({"left": _target(0.1)})

    assert result.status == "collision_backoff"
    assert result.q_ref_by_side["left"][0] == pytest.approx(0.005)
    assert result.metadata["collision_backoff_scale"] == pytest.approx(0.5)
    attempts = result.metadata["collision_check_attempts"]
    assert [attempt["scale"] for attempt in attempts] == [1.0, 0.5]
    assert attempts[0]["distance_m"] == pytest.approx(0.02)
    assert attempts[1]["distance_m"] == pytest.approx(0.05)


def test_collision_distance_unavailable_holds_last_safe_pair_atomically() -> None:
    solver = _FakeDualSolver()
    planner = _planner(solver)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    solver.solve_output[0] = 0.005
    solver.distance = None

    result = planner.plan_pose_reference({"left": _target(0.1)})

    assert result.status == "hold_last_safe"
    assert result.hold is True
    np.testing.assert_allclose(result.q_ref_by_side["left"], np.zeros(6))
    np.testing.assert_allclose(result.q_ref_by_side["right"], np.zeros(6))
    assert result.metadata["failure"]["code"] == "collision_distance_unavailable"
    assert result.metadata["candidate_endpoint_collision_checked"] is False
    assert result.metadata["candidate_endpoint_collision_passed"] is False
    assert result.metadata["endpoint_collision_passed"] is True
    assert result.metadata["segment_collision_certified"] is False


def test_collision_failure_records_closest_proxy_pair_for_diagnostics() -> None:
    solver = _FakeDualSolver()

    def distance_details(qpos):
        solver.distance_q.append(np.asarray(qpos, dtype=float).reshape(12).copy())
        return {
            "minimum_distance_m": (
                0.10 if len(solver.distance_q) == 1 else 0.001
            ),
            "collision_pair_index": 4,
            "collision_geometry_1": "right_hand_proxy_0",
            "collision_geometry_2": "table_0",
        }

    solver.compute_min_collision_distance_details = distance_details
    planner = _planner(solver)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    solver.solve_output[0] = 0.01

    result = planner.plan_pose_reference({"left": _target(0.1)})

    assert result.hold is True
    assert result.metadata["failure"]["code"] == "collision_margin_low"
    attempts = result.metadata["collision_check_attempts"]
    assert attempts
    assert all(
        attempt["collision_geometry_1"] == "right_hand_proxy_0"
        and attempt["collision_geometry_2"] == "table_0"
        for attempt in attempts
    )


def test_solver_exception_holds_last_safe_pair() -> None:
    solver = _FakeDualSolver()
    planner = _planner(solver)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    solver.solve_error = RuntimeError("qp failed")

    result = planner.plan_pose_reference({"left": _target(0.1), "right": _target(-0.1)})

    assert result.hold is True
    assert result.metadata["failure"]["code"] == "planner_exception:RuntimeError"
    np.testing.assert_allclose(solver.current_q, np.zeros(12))


def test_single_side_target_freezes_passive_arm_inside_full_collision_model() -> None:
    solver = _FakeDualSolver()
    planner = _planner(solver)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    solver.solve_output[0] = 0.005
    solver.solve_output[6] = 0.009

    result = planner.plan_pose_reference({"left": _target(0.1)})

    assert result.q_ref_by_side["left"][0] == pytest.approx(0.005)
    assert result.q_ref_by_side["right"][0] == pytest.approx(0.0)
    assert "passive_side_held:right" in result.modification_reasons
    assert solver.distance_q[-1][6] == pytest.approx(0.0)
    assert result.metadata["passive_side_policy"] == (
        "freeze_at_last_safe_qpos_in_full_collision_model"
    )


def test_qpos_endpoint_validation_is_fail_closed_and_does_not_commit_state() -> None:
    solver = _FakeDualSolver()
    planner = _planner(solver)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})

    accepted = planner.validate_qpos_reference({"left": np.full(6, 0.005)})

    assert accepted.safe is True
    np.testing.assert_allclose(accepted.qpos_by_side["right"], np.zeros(6))
    np.testing.assert_allclose(planner.last_safe_qpos_by_side["left"], np.zeros(6))
    assert accepted.metadata["segment_collision_certified"] is False
    assert accepted.metadata["mutates_last_safe_state"] is False

    solver.distance = None
    rejected = planner.validate_qpos_reference({"left": np.full(6, 0.005)})
    assert rejected.safe is False
    assert rejected.reason == "collision_distance_unavailable"


def test_qpos_endpoint_validation_rejects_absolute_joint_limit() -> None:
    solver = _FakeDualSolver()
    planner = _planner(solver)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})

    result = planner.validate_qpos_reference({"left": np.full(6, 2.0)})

    assert result.safe is False
    assert result.reason == "joint_position_limit"
    assert result.collision_distance_m is None


def test_qpos_endpoint_admission_commits_exact_reference_without_modification() -> None:
    solver = _FakeDualSolver()
    planner = _planner(solver)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    requested = {
        "left": np.full(6, 0.005),
        "right": np.full(6, -0.004),
    }

    result = planner.admit_qpos_reference(requested, source="strict_qpos_replay")

    assert result.safe is True
    assert result.metadata["qpos_target_modified"] is False
    assert result.metadata["mutates_last_safe_state"] is True
    np.testing.assert_allclose(result.qpos_by_side["left"], requested["left"])
    np.testing.assert_allclose(result.qpos_by_side["right"], requested["right"])
    np.testing.assert_allclose(planner.last_safe_qpos_by_side["left"], requested["left"])
    np.testing.assert_allclose(planner.last_safe_qpos_by_side["right"], requested["right"])


def test_pose_lift_v2_unwraps_equivalent_joint_near_last_admitted_anchor() -> None:
    solver = _FakeDualSolver(lower=np.full(12, -10.0), upper=np.full(12, 10.0))
    planner = _planner(
        solver,
        config=RM65PositionCollisionPlannerConfig(
            reference_hz=60.0,
            max_joint_velocity_rad_s=(100.0,) * 6,
            pose_lift_profile=RM65_POSE_LIFT_V2_CONTINUOUS_NAME,
        ),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    solver.solve_output[0] = 2.0 * np.pi + 0.1

    result = planner.plan_pose_reference({"left": _target(0.1)})

    assert result.valid is True
    assert result.hold is False
    assert result.q_ik_raw_by_side["left"][0] == pytest.approx(2.0 * np.pi + 0.1)
    assert result.q_ref_by_side["left"][0] == pytest.approx(0.1)
    assert result.metadata["pose_lift"]["applied"] is True
    assert result.metadata["pose_lift"]["reason"] == "accepted_wrist_canonicalized"


def test_pose_lift_v3_accepts_equivalent_candidate_despite_inherited_raw_residual() -> None:
    solver = _FakeDualSolver()
    planner = _planner(
        solver,
        config=RM65PositionCollisionPlannerConfig(
            reference_hz=60.0,
            max_joint_velocity_rad_s=(100.0,) * 6,
            pose_lift_profile=RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME,
        ),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    solver.solve_output[3] = 0.2
    solver.solve_output[5] = -0.2
    target = _target()
    target[:3, :3] = R.from_euler("z", 2.0, degrees=True).as_matrix()

    result = planner.plan_pose_reference({"left": target})

    assert result.hold is False
    assert result.q_ik_raw_by_side["left"][3] == pytest.approx(0.2)
    assert result.q_ref_by_side["left"][3] == pytest.approx(0.0)
    assert result.q_ref_by_side["left"][5] == pytest.approx(0.0)
    assert result.metadata["pose_lift"]["reason"] == (
        "accepted_bounded_wrist_continuity"
    )
    assert result.metadata["pose_lift"]["selected_scales"]["left"] == 1.0


def test_pose_lift_v3_uses_strongest_per_side_correction_inside_pose_slack() -> None:
    solver = _FakeDualSolver(wrist_null_orientation_gain=0.01)
    planner = _planner(
        solver,
        config=RM65PositionCollisionPlannerConfig(
            reference_hz=60.0,
            max_joint_velocity_rad_s=(100.0,) * 6,
            pose_lift_profile=RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME,
        ),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    solver.solve_output[3] = 0.2
    solver.solve_output[5] = -0.2
    target = _target()
    target[:3, :3] = R.from_rotvec([0.0, 0.0, 0.004]).as_matrix()

    result = planner.plan_pose_reference({"left": target})

    assert result.hold is False
    assert result.metadata["pose_lift"]["selected_scales"]["left"] == pytest.approx(
        0.125
    )
    assert result.q_ref_by_side["left"][3] == pytest.approx(0.175)
    assert result.q_ref_by_side["left"][5] == pytest.approx(-0.175)
    stage = result.metadata["pose_lift"]["stages"][-1]
    assert stage["final_certification"]["accepted"] is True
