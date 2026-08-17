from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.joint_limits import JointVelocityLimitProfile
from robot_motion.profiles import IKSafetyProfile
from robot_motion.rm65 import (
    BimanualIKResult,
    BimanualPoseTarget,
    BimanualPoseTargetSource,
)
import robot_motion.rm65.ik as rm65_ik_module
import robot_motion.rm65.dual_arm_ik as rm65_dual_arm_ik_module
from robot_motion.rm65.ik import (
    RM65DualArmCollisionIKConfig,
    build_rm65_dual_arm_collision_ik_solver,
)
from robot_motion.rm65.scene_profiles import DualArmSceneProfile
from robot_motion.rm65.dual_arm_ik import (
    MultiTaskIKSolver,
    _AdmittedReferenceMotionLimit,
    _ExplicitVelocityLimit,
    _NormalizedSigmaEscapeTask,
    _StructuralBranchGuardLimit,
    _task_conditioned_jacobian_metrics,
)
import robot_motion.rm65.dual_arm_target_solver as target_solver_module
from robot_motion.types import PoseTolerance
import teleop_runtime.retargeters.dual_arm_collision_aware as dual_module
from teleop_runtime.retargeters.dual_arm_collision_aware import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    DualArmTargetContext,
)
from teleop_runtime.schemas import EffectorCommand


class _FakeModel:
    def __init__(self, *, lower: np.ndarray, upper: np.ndarray):
        self.lowerPositionLimit = lower
        self.upperPositionLimit = upper


class _FakeRobot:
    def __init__(self, *, lower: np.ndarray, upper: np.ndarray):
        self.model = _FakeModel(lower=lower, upper=upper)


class _FakeDualArmSolver:
    def __init__(self, *, lower: np.ndarray | None = None, upper: np.ndarray | None = None):
        self.q = np.zeros(12, dtype=float)
        lower_array = np.full(12, -np.pi, dtype=float) if lower is None else lower
        upper_array = np.full(12, np.pi, dtype=float) if upper is None else upper
        self.robot = _FakeRobot(lower=lower_array, upper=upper_array)
        self.last_solve_profile = {}
        self.last_collision_distance_profile = {}
        self.min_collision_distance = 0.25
        self.configured_motion_limits: list[tuple[np.ndarray, np.ndarray | None]] = []
        self.admitted_motion_references: list[tuple[np.ndarray, np.ndarray, float]] = []

    @property
    def current_q(self) -> np.ndarray:
        return self.q.copy()

    def add_frame_task(self, *args, **kwargs) -> None:
        return None

    def add_posture_task(self, *args, **kwargs) -> None:
        return None

    def set_configuration_q(self, qpos: np.ndarray) -> None:
        self.q = np.asarray(qpos, dtype=float).reshape(12).copy()

    def compute_frame_pose(self, frame_name: str, q: np.ndarray | None = None) -> np.ndarray:
        qpos = self.q if q is None else np.asarray(q, dtype=float).reshape(12)
        pose = np.eye(4, dtype=float)
        if frame_name == "arm_link6":
            pose[0, 3] = -float(qpos[0])
        elif frame_name == "right_arm_link6":
            pose[0, 3] = float(qpos[6])
        else:
            raise ValueError(frame_name)
        return pose

    def compute_min_collision_distance(self, q: np.ndarray | None = None) -> float:
        del q
        return float(self.min_collision_distance)

    def configure_admitted_reference_motion_limit(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray | None,
    ) -> None:
        self.configured_motion_limits.append(
            (
                np.asarray(velocity, dtype=float).copy(),
                None
                if acceleration is None
                else np.asarray(acceleration, dtype=float).copy(),
            )
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


def test_task_conditioned_metrics_use_pink_cost_and_velocity_capability() -> None:
    metrics = _task_conditioned_jacobian_metrics(
        np.eye(6),
        np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        task_cost=np.ones(6),
        task_gain=1.0,
        joint_velocity_limit_rad_s=np.arange(1.0, 7.0),
        dt_sec=0.1,
    )

    np.testing.assert_allclose(
        metrics["singular_values_task_velocity"],
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    )
    assert metrics["sigma_task_velocity_min"] == pytest.approx(1.0)
    assert metrics["sigma_task_velocity_max"] == pytest.approx(6.0)
    assert metrics["condition_number_task_velocity"] == pytest.approx(6.0)
    assert metrics["weakest_direction_index"] == 5
    assert metrics["max_required_velocity_utilization"] == pytest.approx(1.0)
    assert metrics["weakest_direction_required_velocity_utilization"] == pytest.approx(
        1.0
    )
    right_rows = np.asarray(
        metrics["joint_utilization_direction_rows"],
        dtype=float,
    )
    np.testing.assert_allclose(right_rows @ right_rows.T, np.eye(6), atol=1e-12)
    assert (
        metrics["joint_utilization_direction_contract"]
        == "right_singular_vectors_in_joint_velocity_limit_normalized_coordinates"
    )
    np.testing.assert_allclose(
        metrics["least_squares_joint_velocity_rad_s"],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        atol=1e-12,
    )
    assert (
        metrics["least_squares_joint_velocity_contract"]
        == "unconstrained_weighted_pseudoinverse_diagnostic_not_command"
    )


def test_task_conditioned_metrics_report_unavailable_demand_at_zero_authority() -> None:
    jacobian = np.eye(6)
    jacobian[5, 5] = 0.0
    metrics = _task_conditioned_jacobian_metrics(
        jacobian,
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.1, 0.1]),
        task_cost=np.ones(6),
        task_gain=1.0,
        joint_velocity_limit_rad_s=np.ones(6),
        dt_sec=0.1,
    )

    assert metrics["sigma_task_velocity_min"] == pytest.approx(0.0)
    assert np.isinf(metrics["max_required_velocity_utilization"])
    assert np.isinf(
        metrics["weakest_direction_required_velocity_utilization"]
    )


def test_solver_generated_configuration_is_strictly_clamped_before_pink_update() -> None:
    solver = object.__new__(MultiTaskIKSolver)
    solver.robot = _FakeRobot(
        lower=np.array([-2.355, -np.inf, -1.0]),
        upper=np.array([2.355, np.inf, 1.0]),
    )

    bounded = solver._clamp_integrated_configuration_to_limits(
        np.array([2.3585, 42.0, -1.2])
    )

    np.testing.assert_allclose(bounded, [2.355, 42.0, -1.0])


def test_analytic_branch_anchor_survives_per_solve_posture_reset() -> None:
    class _Task:
        def __init__(self) -> None:
            self.target = None
            self.configuration_reset_count = 0

        def set_target(self, target: np.ndarray) -> None:
            self.target = np.asarray(target, dtype=float).copy()

        def set_target_from_configuration(self, configuration) -> None:
            self.configuration_reset_count += 1
            self.set_target(configuration.q)

    class _Configuration:
        def __init__(self) -> None:
            self.q = np.zeros(2, dtype=float)

        def update(self, q: np.ndarray) -> None:
            self.q = np.asarray(q, dtype=float).copy()

    regular_task = _Task()
    branch_task = _Task()
    branch_target = np.array([0.4, -0.3], dtype=float)
    branch_task.set_target(branch_target)

    solver = object.__new__(MultiTaskIKSolver)
    solver.debug_profile = False
    solver.configuration = _Configuration()
    solver.tasks = [regular_task, branch_task]
    solver.FrameTask = type("_NoFrameTask", (), {})
    solver.analytic_branch_anchor_task = branch_task
    solver._analytic_branch_anchor_q = branch_target.copy()
    solver.solve_ik_fn = lambda *_args, **_kwargs: np.zeros(2, dtype=float)
    solver.dt = 0.1
    solver.solver = "fake"
    solver.limits = None
    solver.damping = 0.01
    solver.pin = SimpleNamespace(
        integrate=lambda _model, q, displacement: (
            np.asarray(q, dtype=float) + np.asarray(displacement, dtype=float)
        )
    )
    solver.robot = SimpleNamespace(
        model=SimpleNamespace(
            nq=2,
            nv=2,
            lowerPositionLimit=np.full(2, -1.0),
            upperPositionLimit=np.full(2, 1.0),
        ),
        collision_model=SimpleNamespace(collisionPairs=[]),
    )

    result = solver.solve(targets=[])

    np.testing.assert_allclose(result, np.zeros(2))
    assert regular_task.configuration_reset_count == 1
    assert branch_task.configuration_reset_count == 0
    np.testing.assert_allclose(branch_task.target, branch_target)


def test_branch_continuity_anchor_survives_per_solve_posture_reset() -> None:
    class _Task:
        def __init__(self) -> None:
            self.target = None
            self.configuration_reset_count = 0

        def set_target(self, target: np.ndarray) -> None:
            self.target = np.asarray(target, dtype=float).copy()

        def set_target_from_configuration(self, configuration) -> None:
            self.configuration_reset_count += 1
            self.set_target(configuration.q)

    class _Configuration:
        def __init__(self) -> None:
            self.q = np.zeros(2, dtype=float)

        def update(self, q: np.ndarray) -> None:
            self.q = np.asarray(q, dtype=float).copy()

    regular_task = _Task()
    branch_task = _Task()
    branch_target = np.array([0.4, -0.3], dtype=float)
    branch_task.set_target(branch_target)

    solver = object.__new__(MultiTaskIKSolver)
    solver.debug_profile = False
    solver.configuration = _Configuration()
    solver.tasks = [regular_task, branch_task]
    solver.FrameTask = type("_NoFrameTask", (), {})
    solver.analytic_branch_anchor_task = None
    solver._analytic_branch_anchor_q = None
    solver.branch_continuity_anchor_task = branch_task
    solver._branch_continuity_anchor_q = branch_target.copy()
    solver.solve_ik_fn = lambda *_args, **_kwargs: np.zeros(2, dtype=float)
    solver.dt = 0.1
    solver.solver = "fake"
    solver.limits = None
    solver.damping = 0.01
    solver.pin = SimpleNamespace(
        integrate=lambda _model, q, displacement: (
            np.asarray(q, dtype=float) + np.asarray(displacement, dtype=float)
        )
    )
    solver.robot = SimpleNamespace(
        model=SimpleNamespace(
            nq=2,
            nv=2,
            lowerPositionLimit=np.full(2, -1.0),
            upperPositionLimit=np.full(2, 1.0),
        ),
        collision_model=SimpleNamespace(collisionPairs=[]),
    )

    result = solver.solve(targets=[])

    np.testing.assert_allclose(result, np.zeros(2))
    assert regular_task.configuration_reset_count == 1
    assert branch_task.configuration_reset_count == 0
    np.testing.assert_allclose(branch_task.target, branch_target)


def test_branch_continuity_anchor_mask_preserves_unmasked_cost_rows() -> None:
    class _Task:
        def __init__(self) -> None:
            self.cost = np.array([1.0, 2.0], dtype=float)
            self.target = None

        def set_target(self, target: np.ndarray) -> None:
            self.target = np.asarray(target, dtype=float).copy()

        def set_target_from_configuration(self, configuration) -> None:
            self.set_target(configuration.q)

    task = _Task()
    solver = object.__new__(MultiTaskIKSolver)
    solver.branch_continuity_anchor_task = task
    solver._branch_continuity_anchor_cost_spec = np.array([1.0, 2.0])
    solver._branch_continuity_anchor_cost_vector = np.array([1.0, 2.0])
    solver.robot = SimpleNamespace(
        model=SimpleNamespace(
            nq=2,
            nv=2,
            lowerPositionLimit=np.full(2, -1.0),
            upperPositionLimit=np.full(2, 1.0),
        )
    )

    solver.set_branch_continuity_anchor_q(
        np.zeros(2),
        active_joint_mask=np.array([True, False]),
    )

    np.testing.assert_allclose(task.cost, [1.0, 0.0])
    np.testing.assert_allclose(task.target, [0.0, 0.0])


def test_explicit_velocity_limit_builds_per_step_qp_bounds() -> None:
    limit = _ExplicitVelocityLimit(np.array([1.0, np.inf, 2.0]))

    inequalities = limit.compute_qp_inequalities(None, 0.02)

    assert inequalities is not None
    matrix, bound = inequalities
    np.testing.assert_allclose(
        matrix,
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
    )
    np.testing.assert_allclose(bound, [0.02, 0.04, 0.02, 0.04])


def test_admitted_reference_limit_does_not_multiply_budget_per_ik_iteration() -> None:
    model = SimpleNamespace(
        nq=2,
        nv=2,
        lowerPositionLimit=np.array([-2.0, -2.0]),
        upperPositionLimit=np.array([2.0, 2.0]),
    )
    pin = SimpleNamespace(
        difference=lambda _model, start, end: np.asarray(end) - np.asarray(start)
    )
    limit = _AdmittedReferenceMotionLimit(
        model,
        pin,
        np.array([1.0, 2.0]),
        None,
    )
    limit.set_reference(
        np.zeros(2),
        np.zeros(2),
        0.1,
    )

    first = limit.compute_qp_inequalities(
        SimpleNamespace(q=np.array([0.01, -0.02])),
        123.0,
    )
    second = limit.compute_qp_inequalities(
        SimpleNamespace(q=np.array([0.09, -0.18])),
        456.0,
    )

    assert first is not None and second is not None
    np.testing.assert_allclose(first[1], [0.09, 0.22, 0.11, 0.18])
    np.testing.assert_allclose(second[1], [0.01, 0.38, 0.19, 0.02])


def test_structural_branch_barrier_tapers_only_inward_motion() -> None:
    limit = object.__new__(_StructuralBranchGuardLimit)
    limit.entries = [
        {"joint_index": 2, "sign": 1, "barrier_active": True}
    ]
    limit.model = SimpleNamespace(nq=6, nv=6)
    limit.constraint_mode = "barrier"
    limit.barrier_gain = 6.0
    limit.margin_rad = np.deg2rad(0.75)
    configuration = SimpleNamespace(
        q=np.array([0.0, 0.0, np.deg2rad(1.0), 0.0, 0.0, 0.0])
    )

    inequalities = limit.compute_qp_inequalities(configuration, 0.02)

    assert inequalities is not None
    matrix, bound = inequalities
    np.testing.assert_allclose(matrix[0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    # h=0.25deg and alpha*dt=0.12: at most 0.03deg inward this cycle.
    assert np.rad2deg(bound[0]) == pytest.approx(0.03)
    # Outward delta_q remains feasible because -delta_q <= positive bound.
    assert -np.deg2rad(0.5) <= bound[0]


def test_structural_branch_prediction_arms_before_current_sigma_guard() -> None:
    class _PredictionModel:
        nq = 6
        nv = 6

        @staticmethod
        def createData() -> object:
            return object()

        @staticmethod
        def getFrameId(_name: str) -> int:
            return 0

    def _jacobian(_model, _data, q, _frame_id, _reference):
        jacobian = np.eye(6, dtype=float)
        jacobian[2, 2] = abs(float(q[2])) + 0.01
        return jacobian

    pin = SimpleNamespace(
        ReferenceFrame=SimpleNamespace(LOCAL_WORLD_ALIGNED=0),
        computeFrameJacobian=_jacobian,
        integrate=lambda _model, q, tangent: np.asarray(q) + np.asarray(tangent),
    )
    task = SimpleNamespace(frame="arm_link6")
    limit = _StructuralBranchGuardLimit(
        frame_tasks=[task],
        model=_PredictionModel(),
        pin=pin,
        guard_sigma=0.04,
        margin_deg=0.75,
        release_sigma=0.05,
        constraint_mode="barrier",
        barrier_gain=6.0,
        prediction_horizon_sec=0.15,
    )
    anchor = np.array([0.0, 0.0, 0.03, 0.0, 0.5, 0.0])
    previous_velocity = np.array([0.0, 0.0, -0.2, 0.0, 0.0, 0.0])

    limit.set_reference(anchor, 1.0 / 60.0, previous_velocity)

    frame = limit.metadata["frames"]["arm_link6"]
    assert frame["active"] is True
    assert frame["prediction_armed"] is True
    assert frame["sigma_min"] > 0.05
    assert frame["predicted_sigma_min"] < 0.04
    assert frame["joint_index"] == 2
    assert frame["sign"] == 1


def test_structural_branch_reset_relatches_from_new_authoritative_side() -> None:
    class _ResetModel:
        nq = 6
        nv = 6

        @staticmethod
        def createData() -> object:
            return object()

        @staticmethod
        def getFrameId(_name: str) -> int:
            return 0

    def _jacobian(_model, _data, q, _frame_id, _reference):
        jacobian = np.eye(6, dtype=float)
        jacobian[2, 2] = abs(float(q[2])) + 0.001
        return jacobian

    pin = SimpleNamespace(
        ReferenceFrame=SimpleNamespace(LOCAL_WORLD_ALIGNED=0),
        computeFrameJacobian=_jacobian,
        integrate=lambda _model, q, tangent: np.asarray(q) + np.asarray(tangent),
    )
    limit = _StructuralBranchGuardLimit(
        frame_tasks=[SimpleNamespace(frame="arm_link6")],
        model=_ResetModel(),
        pin=pin,
        guard_sigma=0.04,
        margin_deg=0.25,
        release_sigma=0.05,
        constraint_mode="barrier",
        barrier_gain=6.0,
    )

    positive = np.array([0.0, 0.0, 0.01, 0.0, 0.5, 0.0])
    negative = positive.copy()
    negative[2] = -0.01
    limit.set_reference(positive, 1.0 / 60.0)
    assert limit.metadata["frames"]["arm_link6"]["sign"] == 1

    # Without an epoch reset the latch intentionally remains continuous.
    limit.set_reference(negative, 1.0 / 60.0)
    assert limit.metadata["frames"]["arm_link6"]["sign"] == 1

    limit.reset_reference_state()
    limit.set_reference(negative, 1.0 / 60.0)

    assert limit.metadata["frames"]["arm_link6"]["sign"] == -1
    inequalities = limit.compute_qp_inequalities(
        SimpleNamespace(q=negative),
        1.0 / 60.0,
    )
    assert inequalities is not None
    matrix, bound = inequalities
    assert matrix[0, 2] == pytest.approx(1.0)
    assert bound[0] > 0.0


def test_admitted_reference_limit_anchors_acceleration_to_prior_admission() -> None:
    model = SimpleNamespace(
        nq=2,
        nv=2,
        lowerPositionLimit=np.array([-100.0, -100.0]),
        upperPositionLimit=np.array([100.0, 100.0]),
    )
    pin = SimpleNamespace(
        difference=lambda _model, start, end: np.asarray(end) - np.asarray(start)
    )
    limit = _AdmittedReferenceMotionLimit(
        model,
        pin,
        np.array([1.0, 2.0]),
        np.array([0.5, 1.0]),
    )
    limit.set_reference(
        np.zeros(2),
        np.array([0.2, -0.4]),
        0.1,
    )

    inequalities = limit.compute_qp_inequalities(
        SimpleNamespace(q=np.array([0.01, -0.02])),
        999.0,
    )

    assert inequalities is not None
    np.testing.assert_allclose(inequalities[1], [0.015, -0.01, -0.005, 0.03])


def test_normalized_sigma_escape_task_points_along_guarded_sigma_gradient() -> None:
    class Model:
        nq = 6
        nv = 6

        @staticmethod
        def createData():
            return object()

        @staticmethod
        def getFrameId(_name: str) -> int:
            return 0

    pin = SimpleNamespace(
        integrate=lambda _model, q, tangent: np.asarray(q) + np.asarray(tangent),
        ReferenceFrame=SimpleNamespace(LOCAL_WORLD_ALIGNED=0),
        computeFrameJacobian=lambda _model, _data, q, _frame, _reference: np.diag(
            [1.0, 1.0, 1.0, 1.0, 1.0, float(np.asarray(q)[5])]
        ),
    )
    task = _NormalizedSigmaEscapeTask(
        frame_tasks=[SimpleNamespace(frame="tool")],
        model=Model(),
        pin=pin,
        cost=0.2,
        guard_sigma=0.04,
        rate_per_sec=0.03,
    )

    task.set_reference(np.array([0, 0, 0, 0, 0, 0.02], dtype=float), 0.1)
    hessian, linear = task.compute_qp_objective(
        SimpleNamespace(q=np.array([0, 0, 0, 0, 0, 0.02], dtype=float))
    )

    assert task.metadata["active_frame_count"] == 1
    assert hessian[5, 5] == pytest.approx(0.04, rel=1e-4)
    assert linear[5] < 0.0
    np.testing.assert_allclose(hessian[:5], 0.0, atol=1e-8)

    task.set_reference(np.array([0, 0, 0, 0, 0, 0.10], dtype=float), 0.1)
    assert task.metadata["active_frame_count"] == 0


def _rm65_posture_metadata(*, local_escape_enabled: bool = True) -> dict[str, object]:
    return {
        "joint_posture_preference": {
            "enabled": True,
            "name": "rm65_elbow_good_branch",
            "j2_index": 1,
            "j3_index": 2,
            "j2_high_deg": 70.0,
            "j2_preferred_min_deg": -20.0,
            "j2_preferred_max_deg": 60.0,
            "j3_zero_margin_deg": 10.0,
            "j3_low_deg": 30.0,
            "j3_bad_max_deg": 60.0,
            "j3_preferred_min_deg": 80.0,
            "j3_preferred_max_deg": 130.0,
            "bad_branch_penalty_deg": 180.0,
            "j3_negative_penalty_deg": 160.0,
            "high_j2_low_j3_penalty_deg": 120.0,
            "hard_branch_escape_enabled": True,
            "hard_branch_escape_allowed_joint_indices": [1, 2],
            "hard_branch_escape_min_improvement_deg": 40.0,
            "local_escape_enabled": local_escape_enabled,
            "local_escape_max_candidates": 6,
            "local_escape_max_position_worsen_m": 0.04,
            "local_escape_max_orientation_worsen_deg": 15.0,
            "local_escape_min_improvement_deg": 1.0,
            "wrist_unload_enabled": True,
            "wrist_unload_j1_index": 0,
            "wrist_unload_j4_index": 3,
            "wrist_unload_j5_index": 4,
            "wrist_unload_j6_index": 5,
            "wrist_unload_j4_abs_activation_deg": 160.0,
            "wrist_unload_j5_abs_activation_deg": 75.0,
            "wrist_unload_j6_wrap_activation_deg": 165.0,
            "wrist_unload_step_fraction": 1.0,
            "wrist_unload_max_candidates": 8,
            "wrist_unload_max_position_worsen_m": 0.05,
            "wrist_unload_max_orientation_worsen_deg": 20.0,
            "wrist_unload_min_cost_improvement_deg": 0.5,
        }
    }


def _profile(
    *,
    rm65_posture: bool = False,
    rm65_local_escape: bool = True,
    component_progress: bool = False,
    loose_dynamic_worsening: bool = False,
) -> IKSafetyProfile:
    return IKSafetyProfile(
        name="dual_test",
        target_tolerance=PoseTolerance(position_m=0.03, orientation_deg=10.0),
        hard_branch_joint_indices=(0, 1, 2),
        hard_branch_max_delta_deg=(90.0, 90.0, 90.0),
        local_step_enabled=True,
        local_step_scale_factors=(1.0, 0.5, 0.25),
        local_step_limit_escape_enabled=True,
        local_step_limit_escape_activation_margin_deg=2.0,
        local_step_limit_escape_direction_hysteresis_deg=2.0,
        local_step_limit_escape_max_candidates=8,
        local_step_accept_worse_within_tolerance=False,
        local_step_component_progress_enabled=component_progress,
        local_step_component_progress_min_position_m=5e-4,
        local_step_component_progress_min_orientation_deg=0.25,
        local_step_component_progress_max_score_regression=1.0,
        local_step_component_progress_max_position_worsen_m=0.015,
        local_step_component_progress_max_orientation_worsen_deg=6.0,
        local_step_component_progress_max_after_position_error_m=0.25,
        local_step_dynamic_worsen_enabled=loose_dynamic_worsening,
        local_step_dynamic_worsen_error_start_m=0.02,
        local_step_dynamic_worsen_position_fraction=1.0,
        local_step_dynamic_worsen_max_position_m=0.10,
        metadata=_rm65_posture_metadata(local_escape_enabled=rm65_local_escape) if rm65_posture else {},
    )


def _velocity() -> JointVelocityLimitProfile:
    return JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        max_dt_sec=1.0,
    )


def _scene_profile() -> SimpleNamespace:
    return SimpleNamespace(
        name="dual_arm_hand_proxy_open",
        urdf_path="dual.urdf",
        package_dirs=("assets",),
        collision_pairs=(("left", "right"),),
        left_frame_name="arm_link6",
        right_frame_name="right_arm_link6",
        solution_layout="dual_arm",
    )


def _target_pose(*, x_m: float = 0.0, rotation_deg: float = 0.0) -> np.ndarray:
    angle_rad = np.deg2rad(rotation_deg)
    pose = np.eye(4, dtype=float)
    pose[:2, :2] = (
        (np.cos(angle_rad), -np.sin(angle_rad)),
        (np.sin(angle_rad), np.cos(angle_rad)),
    )
    pose[0, 3] = float(x_m)
    return pose


def _retargeter(
    solver: _FakeDualArmSolver,
    *,
    rm65_posture: bool = False,
    rm65_local_escape: bool = True,
    component_progress: bool = False,
    loose_dynamic_worsening: bool = False,
    joint_operating_margin_deg: float = 0.0,
    joint_limit_guard_mode: str = "legacy",
    joint_limit_boundary_guard_margin_deg: float = 1.5,
) -> DualArmCollisionAwareRetargeter:
    return DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(
                rm65_posture=rm65_posture,
                rm65_local_escape=rm65_local_escape,
                component_progress=component_progress,
                loose_dynamic_worsening=loose_dynamic_worsening,
            ),
            joint_velocity_profile=_velocity(),
            joint_operating_margin_deg=joint_operating_margin_deg,
            joint_limit_guard_mode=joint_limit_guard_mode,
            joint_limit_boundary_guard_margin_deg=(
                joint_limit_boundary_guard_margin_deg
            ),
            inactive_side_hold_execution_release_enabled=True,
        ),
        ik_solver=solver,
    )


def test_dual_arm_reset_target_pose_smoothing_clears_target_history() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    targets = retargeter.current_world_targets()
    retargeter._last_target_poses = {
        "left": targets["left"].copy(),
        "right": targets["right"].copy(),
    }

    retargeter.reset_target_pose_smoothing()

    assert retargeter._last_target_poses == {"left": None, "right": None}


def test_dual_arm_solve_min_iterations_precedes_tolerance_and_budget_exit() -> None:
    solver = _FakeDualArmSolver()
    solve_calls = 0

    def solve(*, targets):
        nonlocal solve_calls
        del targets
        solve_calls += 1
        return solver.current_q

    solver.solve = solve  # type: ignore[attr-defined]
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_max_iterations=3,
            solve_min_iterations=2,
            solve_adaptive_iterations_enabled=True,
            solve_time_budget_ms=0.01,
            solve_iteration_reserve_ms=0.0,
            solve_translation_tolerance_m=100.0,
            solve_orientation_tolerance_deg=360.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=solver,
    )

    _, iterations, _ = retargeter._solve_toward_targets(
        left_target=np.eye(4, dtype=float),
        right_target=np.eye(4, dtype=float),
    )

    assert solve_calls == 2
    assert iterations == 2
    assert retargeter._last_solve_iteration_stop_reason == "pose_tolerance"


def test_single_step_singularity_mode_advances_solver_once_per_outer_tick() -> None:
    solver = _FakeDualArmSolver()
    solve_calls = 0

    def solve(*, targets):
        nonlocal solve_calls
        del targets
        solve_calls += 1
        return solver.current_q

    solver.solve = solve  # type: ignore[attr-defined]
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_max_iterations=4,
            solve_min_iterations=2,
            solve_adaptive_iterations_enabled=True,
            solve_step_mode="single_step_singularity_aware",
            solve_translation_tolerance_m=100.0,
            solve_orientation_tolerance_deg=360.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=solver,
    )

    _, iterations, _ = retargeter._solve_toward_targets(
        left_target=np.eye(4, dtype=float),
        right_target=np.eye(4, dtype=float),
    )

    assert solve_calls == 1
    assert iterations == 1
    assert retargeter._last_solve_iteration_stop_reason == "pose_tolerance"


def test_single_step_position_admittance_preserves_one_solve_per_tick() -> None:
    solver = _FakeDualArmSolver()
    solve_calls = 0

    def solve(*, targets):
        nonlocal solve_calls
        del targets
        solve_calls += 1
        return solver.current_q

    solver.solve = solve  # type: ignore[attr-defined]
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_max_iterations=4,
            solve_min_iterations=2,
            solve_adaptive_iterations_enabled=True,
            solve_time_budget_ms=1000.0 / 60.0,
            solve_iteration_reserve_ms=5.0,
            solve_step_mode="single_step_singularity_aware",
            solve_translation_tolerance_m=100.0,
            solve_orientation_tolerance_deg=360.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=solver,
    )

    profile: dict[str, object] = {}
    _, iterations, _ = retargeter._solve_toward_targets(
        left_target=np.eye(4, dtype=float),
        right_target=np.eye(4, dtype=float),
        target_activity={
            "left": {
                "source": "position_admittance_active_modifier",
                "state": "ACTIVE",
                "inactive": False,
            },
            "right": {
                "source": "tracker_motion_intent",
                "state": "INACTIVE",
                "inactive": True,
            },
        },
        profile=profile,
    )

    assert solve_calls == 1
    assert iterations == 1
    assert retargeter._last_solve_iteration_stop_reason == "pose_tolerance"
    assert profile["solve_iteration_mode"] == "single_physical_step"
    assert profile["solve_iteration_effective_min"] == 1
    assert profile["solve_iteration_effective_max"] == 1


def test_single_step_admitted_motion_limit_uses_formal_velocity_and_admitted_anchor() -> None:
    solver = _FakeDualArmSolver()
    acceleration = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            joint_acceleration_limit_rad_s2=acceleration,
            admitted_reference_motion_limit_enabled=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=solver,
    )

    assert len(solver.configured_motion_limits) == 1
    velocity_limit, acceleration_limit = solver.configured_motion_limits[-1]
    np.testing.assert_allclose(velocity_limit, np.deg2rad([10.0] * 12))
    np.testing.assert_allclose(acceleration_limit, np.tile(acceleration, 2))
    anchor, previous_velocity, dt_sec = solver.admitted_motion_references[-1]
    np.testing.assert_allclose(anchor, solver.current_q)
    np.testing.assert_allclose(previous_velocity, np.zeros(12))
    assert dt_sec == pytest.approx(1.0 / 60.0)

    admitted = np.deg2rad(np.array([0.1] * 12))
    retargeter._commit_admitted_velocity(
        reference_full_q=np.zeros(12),
        admitted_full_q=admitted,
    )
    anchor, previous_velocity, dt_sec = solver.admitted_motion_references[-1]
    np.testing.assert_allclose(anchor, admitted)
    np.testing.assert_allclose(previous_velocity, admitted * 60.0)
    assert dt_sec == pytest.approx(1.0 / 60.0)


def test_wrist_coaxial_projection_preserves_q4_plus_q6_and_removes_null_motion() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[3] = np.deg2rad(10.0)
    candidate[5] = np.deg2rad(-8.0)
    candidate[9] = np.deg2rad(-7.0)
    candidate[11] = np.deg2rad(11.0)

    canonical, metadata = retargeter._canonicalize_wrist_coaxial_solution(
        reference_full_q=reference,
        candidate_full_q=candidate,
    )

    assert metadata["applied"] is True
    np.testing.assert_allclose(
        canonical[[3, 5]],
        np.deg2rad([1.0, 1.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        canonical[[9, 11]],
        np.deg2rad([2.0, 2.0]),
        atol=1e-12,
    )
    assert canonical[3] + canonical[5] == pytest.approx(
        candidate[3] + candidate[5]
    )
    assert canonical[9] + canonical[11] == pytest.approx(
        candidate[9] + candidate[11]
    )


def test_wrist_coaxial_projection_is_noop_outside_soft_band() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    reference = np.zeros(12, dtype=float)
    reference[[4, 10]] = np.deg2rad(12.0)
    candidate = reference.copy()
    candidate[3] = np.deg2rad(10.0)
    candidate[5] = np.deg2rad(-8.0)

    canonical, metadata = retargeter._canonicalize_wrist_coaxial_solution(
        reference_full_q=reference,
        candidate_full_q=candidate,
    )

    np.testing.assert_allclose(canonical, candidate)
    assert metadata["applied"] is False


def test_wrist_branch_guard_keeps_continuous_solver_branch_near_coaxial_band() -> None:
    solver = _FakeDualArmSolver()
    guarded = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_branch_switch_guard_deg=12.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    # The left raw/reference q5 pair is inside the explicit branch guard, while
    # the right pair remains ordinary and may still expose its flip candidate.
    reference[4] = np.deg2rad(13.0)
    reference[10] = np.deg2rad(20.0)
    raw[3] = np.deg2rad(10.0)
    raw[4] = np.deg2rad(11.0)
    raw[5] = np.deg2rad(-20.0)
    raw[6 + 3] = np.deg2rad(10.0)
    raw[6 + 4] = np.deg2rad(20.0)
    raw[6 + 5] = np.deg2rad(-20.0)

    names = [
        name
        for name, _candidate, _score in guarded._branch_candidates(
            raw_full_q=raw,
            reference_full_q=reference,
        )
    ]
    assert "original" in names
    assert "left_flip" not in names
    assert "both_flip" not in names
    assert "right_flip" in names

    rollback = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_branch_switch_guard_deg=0.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    rollback_names = [
        name
        for name, _candidate, _score in rollback._branch_candidates(
            raw_full_q=raw,
            reference_full_q=reference,
        )
    ]
    assert "left_flip" in rollback_names


def test_wrist_branch_guard_blocks_flip_that_candidate_clip_would_pull_into_gate() -> None:
    # This is the residual pattern from the 2026-07-26 remote trace: both the
    # raw and reference q5 values are safely outside 12 deg, but the formal
    # 8x candidate clip turns the +/-pi wrist alternative into roughly +4 deg.
    # Offering that candidate makes canonical q4/q5/q6 jump, then the next
    # frame returns to the continuous raw branch.
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_branch_switch_guard_deg=12.0,
            local_step_candidate_clip_scale=8.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=JointVelocityLimitProfile(
                name="trace",
                max_velocity_deg_s=(240.0,) * 6,
                max_dt_sec=1.0,
            ),
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    reference[3:6] = np.deg2rad([121.52, -28.02, -120.0])
    raw[3:6] = np.deg2rad([117.98, -24.35, -120.0])

    names = [
        name
        for name, _candidate, _score in retargeter._branch_candidates(
            raw_full_q=raw,
            reference_full_q=reference,
        )
    ]
    assert "original" in names
    assert "left_flip" not in names


def test_wrist_branch_guard_rejects_auxiliary_candidate_after_post_clip_crossing() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_branch_switch_guard_deg=12.0,
            local_step_candidate_clip_scale=8.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=JointVelocityLimitProfile(
                name="trace",
                max_velocity_deg_s=(240.0,) * 6,
                max_dt_sec=1.0,
            ),
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    reference[3:6] = np.deg2rad([121.52, -28.02, -120.0])
    # The candidate itself is an equivalent wrist branch.  It is not the
    # preferred-flip source, so this exercises the common post-clip admission
    # guard used by posture/wrist-unload/other auxiliary candidates too.
    candidate = reference.copy()
    candidate[3:6] = np.deg2rad([-62.02, 24.35, 60.0])
    accepted, output, metadata = retargeter._evaluate_dual_arm_step_candidate(
        candidate=dual_module._DualArmCandidate(
            name="auxiliary_flip",
            qpos=candidate,
            source="wrist_unload",
            method="wrist_unload_left",
        ),
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        step_scale=1.0,
    )

    assert accepted is False
    assert metadata["reject_reason"] == "wrist_branch_guard_output_crossing:left"
    assert metadata["wrist_branch_guard_output_crossing_sides"] == ["left"]
    assert np.rad2deg(output[4]) == pytest.approx(3.98, abs=0.05)


def test_wrist_branch_guard_blocks_same_side_large_wrist_flip_after_clip() -> None:
    """A clipped +/-pi branch must not outrank a small continuous raw step."""

    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_branch_switch_guard_deg=12.0,
            local_step_candidate_clip_scale=8.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=JointVelocityLimitProfile(
                name="trace",
                max_velocity_deg_s=(240.0,) * 6,
                max_dt_sec=1.0,
            ),
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    reference = np.zeros(6, dtype=float)
    raw = np.zeros(6, dtype=float)
    reference[3:6] = np.deg2rad([-72.111, -46.072, 82.463])
    raw[3:6] = np.deg2rad([-71.476, -45.455, 81.810])
    # This is the same-side clipped equivalent branch observed in the latest
    # remote trace: q5 stays negative, but the wrist step grows to roughly
    # 50 degrees while the raw step is only a few degrees.
    candidate = np.zeros(6, dtype=float)
    candidate[3:6] = np.deg2rad([-46.946, -16.928, 53.558])

    assert retargeter._wrist_branch_switch_guard_active(
        raw_q=raw,
        reference_q=reference,
        candidate_q=candidate,
        side="right",
    ) is True


def test_far_target_relaxes_q5_hysteresis_but_keeps_escape_boundary() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_q5_hysteresis_release_deg=6.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=JointVelocityLimitProfile(
                name="trace",
                max_velocity_deg_s=(240.0,) * 6,
                max_dt_sec=1.0,
            ),
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    reference[4] = np.deg2rad(5.0)
    candidate[4] = np.deg2rad(2.0)
    targets = {
        "left": _target_pose(x_m=0.20),
        "right": _target_pose(),
    }

    output, metadata = retargeter._stabilize_wrist_q5_candidate(
        reference_full_q=reference,
        candidate_full_q=candidate,
        current_targets=targets,
    )

    # The old fixed 6 deg release band would have held +5 deg.  Far-target
    # relief instead moves to the protected 3.25 deg boundary, never through
    # the SDK gate and never by more than one q5 execution step.
    assert np.rad2deg(output[4]) == pytest.approx(3.25, abs=1e-9)
    assert metadata["far_target_hysteresis_relaxation"]["sides"]["left"][
        "far_target"
    ] is True
    assert metadata["sides"]["left"]["far_target_hysteresis_relaxed"] is True


def test_singularity_escape_wide_search_activates_only_for_far_target() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.solve_step_mode = "single_step_singularity_aware"
    # The fake solver does not infer the elbow preference from the step mode;
    # enable the same local singularity analyzer flag that the position
    # runtime supplies before asking for escape candidates.
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.config.local_step_candidate_clip_scale = 4.0
    retargeter.ik_safety_profile = replace(
        retargeter.ik_safety_profile,
        local_step_singularity_escape_enabled=True,
        local_step_singularity_escape_directional_enabled=True,
        local_step_singularity_escape_wide_search_enabled=True,
        local_step_singularity_escape_wide_search_position_m=0.10,
        local_step_singularity_escape_wide_search_radius_factors=(1.0, 2.0, 4.0),
        local_step_singularity_escape_wide_search_max_position_worsen_m=0.05,
        local_step_singularity_escape_wide_search_max_orientation_worsen_deg=18.0,
    )
    reference = np.zeros(12, dtype=float)
    reference[2] = np.deg2rad(3.5)
    near_targets = {"left": _target_pose(), "right": _target_pose()}
    far_targets = {"left": _target_pose(x_m=0.20), "right": _target_pose()}

    near_candidates, near_metadata = retargeter._dual_arm_singularity_escape_candidates(
        reference_full_q=reference,
        current_targets=near_targets,
    )
    far_candidates, far_metadata = retargeter._dual_arm_singularity_escape_candidates(
        reference_full_q=reference,
        current_targets=far_targets,
    )

    assert near_metadata["wide_search_enabled"] is False
    assert {candidate.metadata["local_step_singularity_escape_radius_factor"] for candidate in near_candidates} == {1.0}
    assert far_metadata["wide_search_enabled"] is True
    assert max(
        candidate.metadata["local_step_singularity_escape_radius_factor"]
        for candidate in far_candidates
    ) > 1.0
    assert any(
        candidate.metadata["local_step_singularity_escape_wide_search"] is True
        for candidate in far_candidates
    )
    # Wide search only chooses a direction.  The emitted qref must remain one
    # ordinary execution step; otherwise a formal clip scale >1 turns the
    # probe itself into a visible jump.
    wide_candidates = [
        candidate
        for candidate in far_candidates
        if candidate.metadata["local_step_singularity_escape_radius_factor"] > 1.0
    ]
    assert wide_candidates
    assert all(
        candidate.metadata["local_step_singularity_escape_command_radius_factor"]
        == pytest.approx(1.0)
        for candidate in wide_candidates
    )
    assert all(
        candidate.metadata["local_step_singularity_escape_wide_probe"] is True
        for candidate in wide_candidates
    )
    assert all(
        candidate.metadata["local_step_singularity_escape_wide_search"]
        is (candidate.metadata["local_step_singularity_escape_side"] == "left")
        for candidate in far_candidates
    )


def test_wide_singularity_escape_allows_only_bounded_far_target_regression() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.ik_safety_profile = replace(
        retargeter.ik_safety_profile,
        local_step_singularity_escape_enabled=True,
        local_step_singularity_escape_wide_search_max_position_worsen_m=0.05,
    )
    reference = np.zeros(12, dtype=float)
    reference[2] = np.deg2rad(3.5)
    candidate_q = reference.copy()
    candidate_q[2] = np.deg2rad(4.5)
    local_reference = retargeter._local_proxy_singularity_metadata(reference)

    def pose_metrics(qpos: np.ndarray, _targets: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        worsening = 0.04 if abs(float(qpos[2] - reference[2])) > 1e-8 else 0.0
        return {
            "left": {"translation_m": 0.10 + worsening, "rotation_deg": 0.0},
            "right": {"translation_m": 0.0, "rotation_deg": 0.0},
        }

    retargeter._dual_pose_metrics = pose_metrics  # type: ignore[method-assign]
    targets = {"left": _target_pose(), "right": _target_pose()}
    for wide_search, expected in ((False, False), (True, True)):
        candidate = dual_module._DualArmCandidate(
            name="singularity_escape_test",
            qpos=candidate_q,
            source="singularity_escape",
            method="singularity_escape_test",
            metadata={
                "local_step_singularity_escape_candidate": True,
                "local_step_singularity_escape_wide_search": wide_search,
                "local_step_singularity_escape_reference": local_reference,
            },
        )
        accepted, _output, metadata = retargeter._evaluate_dual_arm_step_candidate(
            candidate=candidate,
            reference_full_q=reference,
            current_targets=targets,
            step_scale=1.0,
        )
        assert accepted is expected
        assert metadata["dual_singularity_escape_wide_search"] is wide_search


def test_ik_shadow_sink_is_observational_and_receives_fixed_arm_pair_layout() -> None:
    retargeter = _retargeter(_FakeDualArmSolver())
    observed: list[tuple[object, ...]] = []
    retargeter.set_ik_shadow_trace_sink(lambda *sample: observed.append(sample))
    targets = retargeter.current_world_targets()
    pre_q = np.arange(12, dtype=float) * 0.01

    retargeter._record_ik_shadow_trace(
        monotonic_time=1.25,
        pre_solve_q=pre_q,
        raw_solver_q=pre_q + 0.01,
        canonical_solver_q=pre_q + 0.005,
        admitted_q=pre_q + 0.002,
        current_targets=targets,
        status="tracking",
    )

    assert len(observed) == 1
    assert observed[0][0] == pytest.approx(1.25)
    np.testing.assert_allclose(observed[0][1], pre_q)
    np.testing.assert_allclose(observed[0][2], pre_q + 0.01)
    np.testing.assert_allclose(observed[0][3], pre_q + 0.005)
    np.testing.assert_allclose(observed[0][4], pre_q + 0.002)
    assert observed[0][-1] == "tracking"


def test_dual_arm_target_activity_prefers_tracker_motion_state_machine() -> None:
    retargeter = _retargeter(_FakeDualArmSolver())
    targets = retargeter.current_world_targets()
    left_command = EffectorCommand(
        effector_name="left_arm",
        side="left",
        command_type="cartesian_pose",
        value=np.eye(4, dtype=float),
        frame_name="arm_base",
        metadata={
            "tracker_motion_intent": {
                "state": "INACTIVE",
                "inactive": True,
                "hold_weight": 1.0,
                "transition_reason": "enter_dwell_complete",
                "motion_window": {"quality_valid": True},
            }
        },
    )

    activity = retargeter._target_activity_metadata(
        targets,
        cartesian_commands={"left": left_command, "right": None},
    )

    assert activity["left"]["source"] == "tracker_motion_state_machine"
    assert activity["left"]["state"] == "INACTIVE"
    assert activity["left"]["hold_weight"] == 1.0
    assert activity["right"]["source"] == "target_delta_legacy"


def test_dual_arm_target_activity_preserves_explicit_fix_arm_authority() -> None:
    retargeter = _retargeter(_FakeDualArmSolver())
    targets = retargeter.current_world_targets()

    activity = retargeter._target_activity_metadata(
        targets,
        source_metadata_by_side={
            "left": {
                "fix_arm_frozen": True,
                "fix_arm_fixed_side": "left",
                "tracker_motion_intent": {
                    "state": "INACTIVE",
                    "inactive": True,
                    "hold_weight": 1.0,
                },
            },
            "right": None,
        },
    )

    assert activity["left"]["source"] == "fix_arm_controller"
    assert activity["left"]["state"] == "FIXED"
    assert activity["left"]["fixed_by_authority"] is True
    assert retargeter._authority_fixed_sides(activity) == frozenset({"left"})


def test_dual_arm_target_activity_accepts_pose_action_interval_intent() -> None:
    retargeter = _retargeter(_FakeDualArmSolver())
    targets = retargeter.current_world_targets()
    activity = retargeter._target_activity_metadata(
        targets,
        source_metadata_by_side={
            "left": {
                "pose_action_motion_intent": {
                    "source": "pose_action_interval_velocity_state_machine",
                    "state": "ACTIVE",
                    "inactive": False,
                    "hold_weight": 0.0,
                    "evidence": {
                        "quality_valid": True,
                        "linear_speed_p90_m_s": 0.12,
                        "angular_speed_p90_rad_s": 0.3,
                    },
                }
            },
            "right": None,
        },
    )

    assert activity["left"]["source"] == (
        "pose_action_interval_velocity_state_machine"
    )
    assert activity["left"]["state"] == "ACTIVE"
    assert activity["left"]["inactive"] is False
    assert activity["left"]["evidence"]["quality_valid"] is True


def test_dual_arm_public_kinematics_interface_is_read_only() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    solver.set_configuration_q(np.linspace(-0.3, 0.3, 12))
    initial_full_q = solver.current_q.copy()

    current = retargeter.current_qpos_by_side()
    np.testing.assert_allclose(current["left"], initial_full_q[:6])
    np.testing.assert_allclose(current["right"], initial_full_q[6:12])
    current["left"][:] = 99.0
    np.testing.assert_allclose(solver.current_q, initial_full_q)

    left = np.linspace(0.1, 0.6, 6)
    right = np.linspace(-0.1, -0.6, 6)
    embedded = retargeter.solver_q_from_qpos_by_side(
        {"left": left, "right": right}
    )
    np.testing.assert_allclose(embedded[:6], left)
    np.testing.assert_allclose(embedded[6:12], right)

    solver_targets = retargeter.solver_world_targets_for_qpos_pair(
        left_qpos=left,
        right_qpos=right,
    )
    tcp_targets = retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=left,
        right_qpos=right,
    )
    np.testing.assert_allclose(solver_targets["left"][0, 3], -left[0])
    np.testing.assert_allclose(solver_targets["right"][0, 3], right[0])
    np.testing.assert_allclose(tcp_targets["left"], solver_targets["left"])
    np.testing.assert_allclose(tcp_targets["right"], solver_targets["right"])
    np.testing.assert_allclose(solver.current_q, initial_full_q)


def test_bimanual_pose_target_owns_detached_pure_data() -> None:
    left_target = _target_pose(x_m=0.10)
    source_value = _target_pose(x_m=0.05)
    tracker_intent = {"state": "ACTIVE", "inactive": False}
    target = BimanualPoseTarget(
        timestamp=1.25,
        target_pose_by_side={"left": left_target, "right": _target_pose()},
        active_sides=("left",),
        source_by_side={
            "left": BimanualPoseTargetSource(
                command_type="cartesian_pose",
                value=source_value,
                frame_name="arm_base",
                metadata={"tracker_motion_intent": tracker_intent},
            ),
            "right": None,
        },
    )

    left_target[0, 3] = 99.0
    source_value[0, 3] = 99.0
    tracker_intent["state"] = "INACTIVE"

    assert target.target_pose_by_side["left"][0, 3] == pytest.approx(0.10)
    assert target.source_by_side["left"] is not None
    assert target.source_by_side["left"].value[0, 3] == pytest.approx(0.05)
    assert (
        target.source_by_side["left"].metadata["tracker_motion_intent"]["state"]
        == "ACTIVE"
    )


def test_pure_bimanual_solve_returns_atomic_result_and_facade_round_trips() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    candidate = np.zeros(12, dtype=float)
    candidate[0] = 0.04
    candidate[6] = -0.06
    solver.solve = lambda *, targets: candidate.copy()  # type: ignore[attr-defined]
    target = BimanualPoseTarget(
        timestamp=2.0,
        target_pose_by_side={
            "left": _target_pose(x_m=0.10),
            "right": _target_pose(x_m=-0.10),
        },
        active_sides=("left", "right"),
        source_by_side={
            side: BimanualPoseTargetSource(
                command_type="cartesian_pose",
                value=_target_pose(),
                frame_name="arm_base",
                metadata={
                    "tracker_motion_intent": {
                        "state": "ACTIVE",
                        "inactive": False,
                        "hold_weight": 0.0,
                    }
                },
            )
            for side in ("left", "right")
        },
    )

    result = retargeter.solve_target_pair(target)

    assert isinstance(result, BimanualIKResult)
    assert tuple(result.side_results) == ("left", "right")
    assert result.status in {"tracking", "boundary_active"}
    assert result.metadata["target_activity"]["left"]["source"] == (
        "tracker_motion_state_machine"
    )
    np.testing.assert_allclose(
        result.qpos_by_side()["left"],
        result.side_results["left"].metadata["solver_output_qpos"],
    )
    np.testing.assert_allclose(
        result.qpos_by_side()["right"],
        result.side_results["right"].metadata["solver_output_qpos"],
    )

    command = retargeter._command_from_bimanual_result(result)
    assert command.timestamp == 2.0
    assert command.metadata == result.metadata
    for side in ("left", "right"):
        effector = command.commands[f"{side}_arm"]
        side_result = result.side_results[side]
        assert effector.command_type == side_result.command_type
        assert effector.frame_name == side_result.command_frame_name
        np.testing.assert_allclose(effector.value, side_result.command_value)


def test_direct_bimanual_target_hard_holds_inactive_side_at_final_q_boundary() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    initial_q = solver.current_q.copy()
    candidate = initial_q.copy()
    candidate[0] -= 0.04
    candidate[6] -= 0.06
    solver.solve = lambda *, targets: candidate.copy()  # type: ignore[attr-defined]
    current_targets = retargeter.current_world_targets()
    left_target = current_targets["left"].copy()
    left_target[0, 3] += 0.10
    target = BimanualPoseTarget(
        timestamp=2.0,
        target_pose_by_side={
            "left": left_target,
            "right": current_targets["right"],
        },
        active_sides=("left",),
        hard_hold_inactive_sides=True,
        source_by_side={
            "left": BimanualPoseTargetSource(
                command_type="tcp_pose",
                value=left_target,
                frame_name="dual_arm_solver_world",
                metadata={},
            ),
            "right": None,
        },
    )

    result = retargeter.solve_target_pair(target)

    assert result.qpos_by_side()["left"][0] < initial_q[0]
    np.testing.assert_allclose(result.qpos_by_side()["right"], initial_q[6:12])
    assert result.metadata["target_activity"]["right"]["fixed_by_authority"] is True
    hold = result.metadata["direct_target_inactive_side_hold"]
    assert hold["active_sides"] == ["left"]
    assert hold["held_sides"] == ["right"]
    assert hold["applied"] is True
    assert hold["collision_rechecked"] is True
    assert hold["collision_fallback"] is False


def test_direct_bimanual_target_holds_both_when_exact_inactive_hold_collides() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    initial_q = solver.current_q.copy()
    candidate = initial_q.copy()
    candidate[0] -= 0.04
    candidate[6] -= 0.06
    solver.solve = lambda *, targets: candidate.copy()  # type: ignore[attr-defined]
    retargeter._profiled_min_collision_distance = (  # type: ignore[method-assign]
        lambda qpos, profile, *, phase: (
            0.0 if phase == "direct_target_inactive_side_hold" else 1.0
        )
    )
    current_targets = retargeter.current_world_targets()
    left_target = current_targets["left"].copy()
    left_target[0, 3] += 0.10
    target = BimanualPoseTarget(
        timestamp=2.0,
        target_pose_by_side={
            "left": left_target,
            "right": current_targets["right"],
        },
        active_sides=("left",),
        hard_hold_inactive_sides=True,
        source_by_side={
            "left": BimanualPoseTargetSource(
                command_type="tcp_pose",
                value=left_target,
                frame_name="dual_arm_solver_world",
                metadata={},
            ),
            "right": None,
        },
    )

    result = retargeter.solve_target_pair(target)

    np.testing.assert_allclose(result.qpos_by_side()["left"], initial_q[:6])
    np.testing.assert_allclose(result.qpos_by_side()["right"], initial_q[6:12])
    assert result.status == "hold_last_safe"
    assert result.fallback_reason is not None
    assert result.fallback_reason.startswith("direct_target_inactive_side_hold:")
    hold = result.metadata["direct_target_inactive_side_hold"]
    assert hold["collision_rechecked"] is True
    assert hold["collision_fallback"] is True


def test_bimanual_contract_facade_has_explicit_compatibility_rollback() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            enforce_realistic_joint_step=False,
            bimanual_result_contract_enabled=False,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=solver,
    )
    context = DualArmTargetContext(
        timestamp=3.0,
        left_target=_target_pose(),
        right_target=_target_pose(),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=None,
        right_cartesian=None,
        left_input_skipped=None,
        right_input_skipped=None,
    )

    command = retargeter.build_command_for_context(context)

    assert command.timestamp == 3.0
    assert tuple(command.commands) == ("left_arm", "right_arm")
    assert command.metadata["collision_avoidance_status"] in {
        "tracking",
        "boundary_active",
        "hold_last_safe",
    }


def test_pre_ik_target_safety_filter_covers_direct_context_solve() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    safety_calls = 0
    activity_targets: list[dict[str, np.ndarray]] = []
    original_activity = retargeter._target_activity_metadata

    def record_activity(targets, *, source_metadata_by_side):
        activity_targets.append(
            {
                side: np.asarray(pose, dtype=float).copy()
                for side, pose in targets.items()
            }
        )
        return original_activity(
            targets,
            source_metadata_by_side=source_metadata_by_side,
        )

    retargeter._target_activity_metadata = record_activity  # type: ignore[method-assign]

    def project(targets):
        nonlocal safety_calls
        safety_calls += 1
        projected = {
            side: np.asarray(pose, dtype=float).copy()
            for side, pose in targets.items()
        }
        projected["left"][0, 3] = 0.05
        return projected

    project.command_metadata = lambda: {  # type: ignore[attr-defined]
        "mode": "active",
        "modified_sides": ["left"],
    }

    retargeter.pre_ik_target_safety_filter = project
    active_source = EffectorCommand(
        effector_name="left_arm",
        side="left",
        command_type="cartesian_pose",
        value=_target_pose(),
        frame_name="arm_base",
        metadata={
            "tracker_motion_intent": {
                "state": "ACTIVE",
                "inactive": False,
                "hold_weight": 0.0,
            }
        },
    )
    context = DualArmTargetContext(
        timestamp=3.0,
        left_target=_target_pose(x_m=0.20),
        right_target=_target_pose(x_m=-0.10),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=active_source,
        right_cartesian=active_source.copy(),
        left_input_skipped=None,
        right_input_skipped=None,
    )

    command = retargeter.build_command_for_context(context)

    assert safety_calls == 1
    assert len(activity_targets) == 1
    assert activity_targets[0]["left"][0, 3] == pytest.approx(0.05)
    assert command.metadata["tcp_pose_envelope"] == {
        "mode": "active",
        "modified_sides": ["left"],
    }


def test_pre_ik_target_safety_filter_rejects_incomplete_pair_before_ik() -> None:
    solver = _FakeDualArmSolver()
    solve_calls = 0

    def solve(*, targets):
        nonlocal solve_calls
        del targets
        solve_calls += 1
        return solver.current_q

    solver.solve = solve  # type: ignore[attr-defined]
    retargeter = _retargeter(solver)
    retargeter.pre_ik_target_safety_filter = lambda targets: {
        "left": targets["left"]
    }
    context = DualArmTargetContext(
        timestamp=3.0,
        left_target=_target_pose(x_m=0.20),
        right_target=_target_pose(x_m=-0.10),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=None,
        right_cartesian=None,
        left_input_skipped=None,
        right_input_skipped=None,
    )

    with pytest.raises(ValueError, match="exactly left and right"):
        retargeter.build_command_for_context(context)

    assert solve_calls == 0


def test_bimanual_contract_facade_can_be_enabled_explicitly() -> None:
    retargeter = _retargeter(_FakeDualArmSolver())
    retargeter.config.bimanual_result_contract_enabled = True
    results: list[BimanualIKResult] = []
    original_convert = retargeter._command_from_bimanual_result

    def recording_convert(result: BimanualIKResult):
        results.append(result)
        return original_convert(result)

    retargeter._command_from_bimanual_result = recording_convert  # type: ignore[method-assign]
    context = DualArmTargetContext(
        timestamp=4.0,
        left_target=_target_pose(),
        right_target=_target_pose(),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=None,
        right_cartesian=None,
        left_input_skipped=None,
        right_input_skipped=None,
    )

    command = retargeter.build_command_for_context(context)

    assert len(results) == 1
    assert isinstance(results[0], BimanualIKResult)
    assert command.timestamp == 4.0
    assert tuple(command.commands) == ("left_arm", "right_arm")


@pytest.mark.parametrize("output_command_type", ["joint_vector", "cartesian_pose"])
def test_native_and_compat_bimanual_outputs_are_field_equivalent(
    output_command_type: str,
) -> None:
    compat = _retargeter(_FakeDualArmSolver())
    native = _retargeter(_FakeDualArmSolver())
    compat.config.output_command_type = output_command_type
    native.config.output_command_type = output_command_type
    compat.config.bimanual_result_contract_enabled = False
    native.config.bimanual_result_contract_enabled = True
    source = EffectorCommand(
        effector_name="left_arm",
        side="left",
        command_type="cartesian_pose",
        value=_target_pose(),
        frame_name="arm_base",
        metadata={
            "tracker_motion_intent": {
                "state": "ACTIVE",
                "inactive": False,
                "hold_weight": 0.0,
            }
        },
    )
    context = DualArmTargetContext(
        timestamp=5.0,
        left_target=_target_pose(),
        right_target=_target_pose(),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=source,
        right_cartesian=None,
        left_input_skipped=None,
        right_input_skipped="inactive",
    )

    compat_command = compat.build_command_for_context(context)
    native_command = native.build_command_for_context(context)

    def assert_equivalent(left, right, *, key: str | None = None) -> None:
        if key == "ik_frame_total_ms":
            assert isinstance(left, float)
            assert isinstance(right, float)
            return
        if isinstance(left, dict):
            assert isinstance(right, dict)
            assert set(left) == set(right)
            for child_key in left:
                assert_equivalent(left[child_key], right[child_key], key=child_key)
            return
        if isinstance(left, np.ndarray):
            np.testing.assert_allclose(left, right)
            return
        if isinstance(left, float):
            assert float(right) == pytest.approx(left)
            return
        assert right == left

    assert tuple(compat_command.commands) == tuple(native_command.commands)
    assert_equivalent(compat_command.metadata, native_command.metadata)
    for effector_name in compat_command.commands:
        left = compat_command.commands[effector_name]
        right = native_command.commands[effector_name]
        assert left.effector_name == right.effector_name
        assert left.side == right.side
        assert left.command_type == right.command_type
        assert left.frame_name == right.frame_name
        np.testing.assert_allclose(left.value, right.value)
        assert_equivalent(left.metadata, right.metadata)


def test_dual_arm_collision_retargeter_defaults_to_robot_motion_builder(monkeypatch) -> None:
    scene = _scene_profile()
    solver = _FakeDualArmSolver()
    calls: list[str] = []

    def fake_robot_motion_builder(config):
        calls.append(config.scene_profile)
        return SimpleNamespace(solver=solver, scene_profile=scene)

    monkeypatch.setattr(dual_module, "build_rm65_dual_arm_scene_profile", lambda _name: scene)
    monkeypatch.setattr(dual_module, "build_rm65_dual_arm_collision_ik_solver", fake_robot_motion_builder)

    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            scene_profile="dual_arm_hand_proxy_open",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        )
    )

    assert retargeter.config.ik_framework == "robot_motion"
    assert retargeter.ik_solver is solver
    assert retargeter.profile is scene
    assert calls == ["dual_arm_hand_proxy_open"]


def test_rm65_ik_can_omit_collision_pairs_without_changing_kinematic_profile(
    monkeypatch,
    tmp_path,
) -> None:
    profile = DualArmSceneProfile(
        name="test_scene",
        urdf_path=tmp_path / "robot.urdf",
        package_dirs=(tmp_path,),
        left_frame_name="left_tcp",
        right_frame_name="right_tcp",
        collision_pairs=(("left_link", "right_link"),),
        hand_proxy_boxes=(),
        fixed_scene_boxes=(),
        left_pose_transform=np.eye(4),
        right_pose_transform=np.eye(4),
        solution_layout="auto",
    )
    constructed: list[dict[str, object]] = []

    class Solver:
        def __init__(self, **kwargs) -> None:
            constructed.append(dict(kwargs))

        def add_frame_task(self, *_args, **_kwargs) -> None:
            return None

        def add_posture_task(self, *_args, **_kwargs) -> None:
            return None

    tuning = SimpleNamespace(
        objective_weights=lambda: SimpleNamespace(
            dual_arm_position_cost=1.0,
            dual_arm_orientation_cost=1.0,
        )
    )
    monkeypatch.setattr(
        rm65_ik_module,
        "build_rm65_dual_arm_scene_profile",
        lambda _name: profile,
    )
    monkeypatch.setattr(
        rm65_dual_arm_ik_module,
        "MultiTaskIKSolver",
        Solver,
    )

    bundle = build_rm65_dual_arm_collision_ik_solver(
        RM65DualArmCollisionIKConfig(
            scene_profile="test_scene",
            collision_constraints_enabled=False,
            ik_tuning=tuning,
        )
    )

    assert bundle.scene_profile.collision_pairs == ()
    assert bundle.scene_profile.urdf_path == profile.urdf_path
    assert bundle.scene_profile.left_frame_name == profile.left_frame_name
    assert constructed[0]["robot_collision_pair_names"] == []


def test_dual_arm_collision_retargeter_rejects_legacy_framework() -> None:
    try:
        DualArmCollisionAwareRetargeterConfig(
            ik_framework="legacy",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        )
    except ValueError as exc:
        assert "Unsupported dual-arm IK framework" in str(exc)
    else:
        raise AssertionError("legacy IK framework should not be accepted")


def test_dual_arm_candidate_layer_generates_joint_limit_escape_for_left_arm() -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    reference[0] = 0.99
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0 / 15.0)
    targets = retargeter.current_world_targets()
    targets["left"] = targets["left"].copy()
    targets["left"][0, 3] = -0.50

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=reference.copy(),
        reference_full_q=reference.copy(),
        current_targets=targets,
        profile=None,
    )

    assert name == "joint_limit_escape_left_j1_neg"
    assert qpos[0] < reference[0]
    assert metadata["limit_escape"]["status"] == "active"
    assert metadata["selected_source"] == "joint_limit_escape"


def test_limit_escape_latch_uses_candidate_joint_margin_when_another_joint_is_closer() -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(
        solver,
        joint_limit_guard_mode="forced_escape_latched",
    )
    retargeter.set_execution_dt_sec(1.0 / 60.0)
    reference = np.zeros(12, dtype=float)
    reference[2] = upper[2] - np.deg2rad(0.50)
    reference[8] = upper[8] - np.deg2rad(1.00)
    targets = retargeter.current_world_targets()

    candidates, _metadata = retargeter._dual_arm_limit_escape_candidates(
        reference_full_q=reference,
        current_targets=targets,
    )
    right_candidate = next(
        candidate
        for candidate in candidates
        if candidate.metadata["local_step_limit_escape_side"] == "right"
    )
    candidate_metadata = right_candidate.metadata

    # The left J3 remains the global minimum, so the old global metric is zero
    # even though the selected right J3 candidate has moved outward.
    assert candidate_metadata["local_step_limit_escape_margin_progress_deg"] == pytest.approx(0.0)
    assert candidate_metadata["local_step_limit_escape_joint_margin_progress_deg"] > 0.0

    receipt = retargeter._update_joint_limit_escape_direction_latch(
        reference_full_q=reference,
        admitted_full_q=right_candidate.qpos,
        raw_full_q=reference,
        status="tracking",
        selection_metadata={
            "selected_source": "joint_limit_escape",
            "selected_metadata": {
                "candidate_metadata": candidate_metadata,
            },
        },
    )
    assert receipt["active"] == {"right:j3": -1}


def test_limit_escape_excludes_only_authority_fixed_side() -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    reference[2] = upper[2] - np.deg2rad(0.50)
    reference[8] = upper[8] - np.deg2rad(0.75)
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0 / 60.0)
    targets = retargeter.current_world_targets()

    tracker_inactive_candidates, _ = retargeter._dual_arm_limit_escape_candidates(
        reference_full_q=reference,
        current_targets=targets,
        target_activity={
            "enabled": True,
            "left": {
                "source": "tracker_motion_state_machine",
                "state": "INACTIVE",
                "inactive": True,
                "hold_weight": 1.0,
            },
            "right": {"inactive": False, "hold_weight": 0.0},
        },
    )
    assert any(
        candidate.metadata["local_step_limit_escape_side"] == "left"
        for candidate in tracker_inactive_candidates
    )

    fixed_candidates, metadata = retargeter._dual_arm_limit_escape_candidates(
        reference_full_q=reference,
        current_targets=targets,
        target_activity={
            "enabled": True,
            "left": {
                "source": "fix_arm_controller",
                "state": "FIXED",
                "inactive": True,
                "hold_weight": 1.0,
                "fixed_by_authority": True,
            },
            "right": {"inactive": False, "hold_weight": 0.0},
        },
    )

    assert metadata["authority_fixed_sides"] == ["left"]
    assert fixed_candidates
    assert all(
        candidate.metadata["local_step_limit_escape_side"] == "right"
        for candidate in fixed_candidates
    )
    assert any(candidate.name == "joint_limit_escape_right_j3_neg" for candidate in fixed_candidates)

    bounded_candidates, bounded_metadata = retargeter._dual_arm_limit_escape_candidates(
        reference_full_q=reference,
        current_targets=targets,
        target_activity={
            "enabled": True,
            "left": {
                "source": "fix_arm_controller",
                "state": "FIXED",
                "inactive": True,
                "hold_weight": 1.0,
                "fixed_by_authority": True,
            },
            "right": {"inactive": False, "hold_weight": 0.0},
        },
        max_candidates_override=1,
    )
    assert len(bounded_candidates) == 1
    assert bounded_metadata["candidate_count"] == 1


def test_bounded_limit_escape_scores_only_emitted_candidate(monkeypatch) -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    reference[2] = upper[2] - np.deg2rad(0.50)
    reference[8] = upper[8] - np.deg2rad(0.75)
    targets = retargeter.current_world_targets()
    calls = 0
    original_metrics = retargeter._dual_pose_metrics

    def counted_metrics(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(retargeter, "_dual_pose_metrics", counted_metrics)
    candidates, metadata = retargeter._dual_arm_limit_escape_candidates(
        reference_full_q=reference,
        current_targets=targets,
        max_candidates_override=1,
    )

    assert len(candidates) == 1
    assert metadata["candidate_count"] == 1
    # The bounded hysteresis path orders by latch/margin and leaves the
    # authoritative pose-progress decision to candidate admission; no
    # non-authoritative pose scoring is needed for candidate generation.
    assert calls == 0
    assert "local_step_limit_escape_pose_progress" in candidates[0].metadata
    assert candidates[0].metadata["local_step_limit_escape_pose_progress_deferred"] is True


def test_dual_arm_operating_margin_rejects_outward_tracking_and_selects_escape() -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(solver, joint_operating_margin_deg=3.0)
    reference = np.zeros(12, dtype=float)
    reference[0] = 0.99
    raw = reference.copy()
    raw[0] = 1.0
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0 / 15.0)
    targets = retargeter.current_world_targets()
    targets["left"] = targets["left"].copy()
    # The raw tracking solution improves Cartesian error by moving J1 outward,
    # but would consume the last operating margin and must be rejected.
    targets["left"][0, 3] = -1.5

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "joint_limit_escape_left_j1_neg"
    assert qpos[0] < reference[0]
    assert metadata["selected_source"] == "joint_limit_escape"
    assert any(
        item["reject_reason"] == "joint_operating_margin"
        for item in metadata["last_rejections"]
    )
    operating_margin = metadata["selected_metadata"]["joint_operating_margin"]
    assert operating_margin["status"] == "recovery_only"
    assert operating_margin["violations"][0]["moving_inward"] is True


def test_dual_arm_candidate_layer_returns_base_candidate_without_auxiliary_expansion() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[6] = np.deg2rad(2.0)
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    targets["right"] = targets["right"].copy()
    targets["right"][0, 3] = np.deg2rad(2.0)

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert qpos[6] > reference[6]
    assert metadata["selected_source"] == "multitask"
    assert metadata["candidate_eval_mode"] == "base_only"
    assert metadata["limit_escape"]["status"] == "deferred"
    assert [stage["stage"] for stage in metadata["candidate_eval_stages"]] == ["base_primary"]


def test_position_candidate_layer_prefers_feasible_non_wrist_singular_branch(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_wrist_singular_branch = True
    retargeter.config.wrist_singularity_threshold_deg = 3.0
    reference = np.zeros(12, dtype=float)
    singular = reference.copy()
    singular[4] = np.deg2rad(1.0)
    singular[10] = np.deg2rad(5.0)
    non_singular = reference.copy()
    non_singular[4] = np.deg2rad(5.0)
    non_singular[10] = np.deg2rad(5.0)
    candidates = [
        dual_module._DualArmCandidate(
            name="singular_low_score",
            qpos=singular,
            source="multitask",
            method="singular_low_score",
        ),
        dual_module._DualArmCandidate(
            name="non_singular_high_score",
            qpos=non_singular,
            source="multitask",
            method="non_singular_high_score",
        ),
    ]

    monkeypatch.setattr(
        retargeter,
        "_dual_arm_tracking_candidates",
        lambda **_kwargs: (candidates, {"enabled": False, "status": "deferred"}),
    )

    def evaluate(*, candidate, **_kwargs):
        return (
            True,
            candidate.qpos.copy(),
            {"score": 1.0 if candidate.name.startswith("singular") else 100.0},
        )

    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", evaluate)
    monkeypatch.setattr(retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=singular,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "non_singular_high_score"
    assert np.allclose(qpos, non_singular)
    preference = metadata["wrist_singularity_preference"]
    assert preference["fully_non_singular_count"] == 1
    assert preference["fallback_to_singular"] is False
    assert preference["selected"]["singular_side_count"] == 0

    retargeter.config.prefer_non_wrist_singular_branch = False
    rollback_qpos, rollback_name, *_ = retargeter._select_safe_tracking_candidate(
        raw_full_q=singular,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )
    assert rollback_name == "singular_low_score"
    assert np.allclose(rollback_qpos, singular)


def test_position_candidate_layer_accepts_singular_when_no_clean_branch(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_wrist_singular_branch = True
    reference = np.zeros(12, dtype=float)
    lower_score = reference.copy()
    lower_score[4] = np.deg2rad(1.0)
    lower_score[10] = np.deg2rad(5.0)
    higher_score = lower_score.copy()
    higher_score[4] = np.deg2rad(2.0)
    candidates = [
        dual_module._DualArmCandidate("singular_best", lower_score, "multitask", "best"),
        dual_module._DualArmCandidate("singular_other", higher_score, "multitask", "other"),
    ]
    monkeypatch.setattr(
        retargeter,
        "_dual_arm_tracking_candidates",
        lambda **_kwargs: (candidates, {"enabled": False, "status": "deferred"}),
    )
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {"score": 1.0 if candidate.name == "singular_best" else 2.0},
        ),
    )
    monkeypatch.setattr(retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=lower_score,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "singular_best"
    assert np.allclose(qpos, lower_score)
    preference = metadata["wrist_singularity_preference"]
    assert preference["fully_non_singular_count"] == 0
    assert preference["fallback_to_singular"] is True
    assert preference["selected"]["sides"]["left"]["is_wrist_singular"] is True


def test_local_singularity_escape_emits_one_step_j3_candidate() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.ik_safety_profile = replace(
        retargeter.ik_safety_profile,
        local_step_singularity_escape_enabled=True,
    )
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.config.prefer_non_wrist_singular_branch = False
    reference = np.zeros(12, dtype=float)
    reference[2] = np.deg2rad(3.5)
    targets = retargeter.current_world_targets()

    candidates, metadata = retargeter._dual_arm_singularity_escape_candidates(
        reference_full_q=reference,
        current_targets=targets,
    )

    assert metadata["status"] == "active"
    assert candidates
    assert all(
        candidate.metadata["local_step_singularity_escape_candidate"]
        for candidate in candidates
    )
    assert any(
        candidate.metadata["local_step_singularity_escape_depth_progress"] > 0
        if "local_step_singularity_escape_depth_progress" in candidate.metadata
        else candidate.metadata["local_step_singularity_escape_side_candidate"][
            "elbow_margin_deg"
        ]
        > 0
        for candidate in candidates
    )


def test_directional_singularity_escape_latches_outward_sign() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.config.prefer_non_wrist_singular_branch = False
    retargeter.ik_safety_profile = replace(
        retargeter.ik_safety_profile,
        local_step_singularity_escape_enabled=True,
        local_step_singularity_escape_directional_enabled=True,
    )
    positive = np.zeros(12, dtype=float)
    positive[2] = np.deg2rad(3.5)
    targets = retargeter.current_world_targets()
    first, first_metadata = retargeter._dual_arm_singularity_escape_candidates(
        reference_full_q=positive,
        current_targets=targets,
    )
    assert first_metadata["directional_enabled"] is True
    assert first
    assert all(
        candidate.metadata["local_step_singularity_escape_direction"] == 1
        for candidate in first
        if candidate.metadata["local_step_singularity_escape_joint_index"] == 2
    )

    # Simulate the raw IK crossing zero by a small amount.  The admitted
    # escape direction must remain positive instead of emitting a negative
    # candidate on the next frame.
    crossed = positive.copy()
    crossed[2] = np.deg2rad(-0.2)
    second, _second_metadata = retargeter._dual_arm_singularity_escape_candidates(
        reference_full_q=crossed,
        current_targets=targets,
    )
    assert second
    assert all(
        candidate.metadata["local_step_singularity_escape_direction"] == 1
        for candidate in second
        if candidate.metadata["local_step_singularity_escape_joint_index"] == 2
    )


def test_directional_singularity_escape_can_search_coupled_j3_j5_move() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.config.prefer_non_wrist_singular_branch = True
    retargeter.ik_safety_profile = replace(
        retargeter.ik_safety_profile,
        local_step_singularity_escape_enabled=True,
        local_step_singularity_escape_directional_enabled=True,
        local_step_singularity_escape_combined_enabled=True,
    )
    reference = np.zeros(12, dtype=float)
    reference[2] = np.deg2rad(3.5)
    reference[4] = np.deg2rad(2.5)
    candidates, _metadata = retargeter._dual_arm_singularity_escape_candidates(
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
    )
    combined = [
        candidate
        for candidate in candidates
        if candidate.metadata.get("local_step_singularity_escape_combined") is True
    ]
    assert combined
    directions = combined[0].metadata[
        "local_step_singularity_escape_direction_by_joint"
    ]
    side = next(iter(directions)).split(":", 1)[0]
    assert directions == {f"{side}:j3": 1, f"{side}:j5": 1}


def test_singularity_escape_uses_raw_crossing_hint_without_rebasing_progress() -> None:
    """A raw IK branch entering the pocket must not erase the admitted escape."""

    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.config.prefer_non_wrist_singular_branch = False
    retargeter.config.elbow_singularity_selection_margin_deg = 0.25
    retargeter.ik_safety_profile = replace(
        retargeter.ik_safety_profile,
        local_step_singularity_escape_enabled=True,
        local_step_singularity_escape_directional_enabled=True,
    )
    reference = np.zeros(12, dtype=float)
    # Just outside the local 3.25-degree proxy; the raw solver branch crosses
    # inward.  This is exactly the frame that previously dropped all escape
    # candidates because it classified the raw seed instead of qref.
    reference[2] = np.deg2rad(3.5)
    raw = reference.copy()
    raw[2] = np.deg2rad(3.0)
    candidates, metadata = retargeter._dual_arm_singularity_escape_candidates(
        reference_full_q=reference,
        seed_full_q=raw,
        current_targets=retargeter.current_world_targets(),
    )
    assert metadata["status"] == "active"
    left_j3 = [
        candidate
        for candidate in candidates
        if candidate.metadata.get("local_step_singularity_escape_side") == "left"
        and candidate.metadata.get("local_step_singularity_escape_joint_index") == 2
    ]
    assert left_j3
    assert all(
        candidate.metadata["local_step_singularity_escape_direction"] == 1
        for candidate in left_j3
    )


def test_directional_escape_locks_selected_axis_until_release() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.config.prefer_non_wrist_singular_branch = True
    retargeter.ik_safety_profile = replace(
        retargeter.ik_safety_profile,
        local_step_singularity_escape_enabled=True,
        local_step_singularity_escape_directional_enabled=True,
        local_step_singularity_escape_combined_enabled=True,
    )
    reference = np.zeros(12, dtype=float)
    reference[2] = np.deg2rad(1.0)
    reference[4] = np.deg2rad(1.0)
    targets = retargeter.current_world_targets()
    first, _ = retargeter._dual_arm_singularity_escape_candidates(
        reference_full_q=reference,
        current_targets=targets,
    )
    j3 = next(
        candidate
        for candidate in first
        if candidate.metadata.get("local_step_singularity_escape_side") == "left"
        and candidate.metadata.get("local_step_singularity_escape_joint_index") == 2
        and not candidate.metadata.get("local_step_singularity_escape_combined", False)
    )
    retargeter._update_singularity_escape_direction_state(
        selected_q=j3.qpos,
        selection_metadata={
            "selected_source": "singularity_escape",
            "selected_metadata": {"candidate_metadata": j3.metadata},
        },
    )
    second, metadata = retargeter._dual_arm_singularity_escape_candidates(
        reference_full_q=j3.qpos,
        current_targets=targets,
    )
    assert metadata["active_axis_plan_by_side"] == {"left": [2]}
    assert second
    assert all(
        set(candidate.metadata["local_step_singularity_escape_direction_by_joint"])
        <= {"left:j3"}
        for candidate in second
        if candidate.metadata.get("local_step_singularity_escape_side") == "left"
    )


def test_directional_escape_certificate_requires_emitted_outward_q3() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.solve_step_mode = "single_step_singularity_aware"
    retargeter.config.prefer_non_elbow_singular_branch = True
    reference = np.zeros(12, dtype=float)
    reference[2] = np.deg2rad(2.0)
    admitted = reference.copy()
    admitted[2] = np.deg2rad(2.5)
    selection = {
        "selected_source": "singularity_escape",
        "selected_metadata": {
            "candidate_metadata": {
                "local_step_singularity_escape_direction_by_joint": {
                    "left:j3": 1,
                }
            }
        },
    }
    metadata = retargeter._singularity_escape_traversal_metadata(
        reference_full_q=reference,
        admitted_full_q=admitted,
        selection_metadata=selection,
    )
    assert metadata["by_side"]["left"] is True

    inward = admitted.copy()
    inward[2] = np.deg2rad(1.5)
    inward_metadata = retargeter._singularity_escape_traversal_metadata(
        reference_full_q=reference,
        admitted_full_q=inward,
        selection_metadata=selection,
    )
    assert inward_metadata["by_side"]["left"] is False


def test_position_candidate_layer_expands_auxiliary_for_elbow_singularity(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.config.prefer_non_wrist_singular_branch = False
    reference = np.zeros(12, dtype=float)
    reference[2] = np.deg2rad(6.0)
    reference[8] = np.deg2rad(6.0)
    elbow_singular = reference.copy()
    elbow_singular[2] = np.deg2rad(1.0)
    elbow_clean = reference.copy()
    elbow_clean[2] = np.deg2rad(7.0)
    base = dual_module._DualArmCandidate(
        "original", elbow_singular, "multitask", "original"
    )
    auxiliary = dual_module._DualArmCandidate(
        "posture_escape_left_j3",
        elbow_clean,
        "posture_escape",
        "posture_escape_left_j3",
    )

    def candidates(**kwargs):
        if kwargs.get("include_posture_escape", False):
            return [auxiliary], {
                "posture_escape": {"enabled": True, "status": "active"}
            }
        return [base], {"enabled": False, "status": "deferred"}

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", candidates)
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {"score": 1.0 if candidate.name == "original" else 100.0},
        ),
    )
    monkeypatch.setattr(retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=elbow_singular,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "posture_escape_left_j3"
    assert np.allclose(qpos, elbow_clean)
    assert metadata["candidate_eval_mode"] == "base_then_auxiliary_singularity_search"
    assert metadata["expanded_for_elbow_singularity"] is True
    preference = metadata["position_singularity_preference"]
    assert preference["fully_non_singular_count"] == 1
    assert preference["fallback_to_singular"] is False
    assert preference["selected"]["any_elbow_singular"] is False


def test_position_candidate_layer_marks_elbow_fallback_when_no_clean_candidate(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.config.prefer_non_wrist_singular_branch = False
    reference = np.zeros(12, dtype=float)
    reference[2] = np.deg2rad(6.0)
    reference[8] = np.deg2rad(6.0)
    singular = reference.copy()
    singular[2] = np.deg2rad(1.0)
    candidate = dual_module._DualArmCandidate(
        "original", singular, "multitask", "original"
    )

    monkeypatch.setattr(
        retargeter,
        "_dual_arm_tracking_candidates",
        lambda **_kwargs: ([candidate], {"enabled": False, "status": "deferred"}),
    )
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {"score": 1.0},
        ),
    )
    monkeypatch.setattr(retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=singular,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "original"
    assert np.allclose(qpos, singular)
    preference = metadata["position_singularity_preference"]
    assert preference["fully_non_singular_count"] == 0
    assert preference["fallback_to_singular"] is True
    assert preference["selected"]["sides"]["left"]["is_elbow_singular"] is True

    retargeter.config.prefer_non_elbow_singular_branch = False
    rollback = retargeter._position_singularity_metadata(singular)
    assert rollback["any_elbow_singular"] is False


def test_position_candidate_layer_prefers_clean_shoulder_branch_and_is_rollbackable(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    retargeter.config.prefer_non_elbow_singular_branch = False
    retargeter.config.prefer_non_wrist_singular_branch = False
    retargeter.config.prefer_non_shoulder_singular_branch = True
    reference = np.zeros(12, dtype=float)
    reference[0] = 0.2
    reference[6] = 0.2
    shoulder_singular = reference.copy()
    shoulder_singular[0] = 0.0
    shoulder_clean = reference.copy()
    shoulder_clean[0] = 0.3
    base = dual_module._DualArmCandidate(
        "original", shoulder_singular, "multitask", "original"
    )
    auxiliary = dual_module._DualArmCandidate(
        "shoulder_escape", shoulder_clean, "posture_escape", "shoulder_escape"
    )

    def analyze(qpos: np.ndarray) -> dict[str, object]:
        singular = abs(float(np.asarray(qpos)[0])) < 0.05
        return {
            "status_code": -1 if singular else 0,
            "label": "shoulder_singularity" if singular else "normal",
            "wrist_center_distance_m": 0.01 if singular else 0.20,
        }

    retargeter.set_position_singularity_analyzers(
        {"left": analyze, "right": analyze}
    )

    def candidates(**kwargs):
        if kwargs.get("include_posture_escape", False):
            return [auxiliary], {
                "posture_escape": {"enabled": True, "status": "active"}
            }
        return [base], {"enabled": False, "status": "deferred"}

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", candidates)
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {"score": 1.0 if candidate.name == "original" else 100.0},
        ),
    )
    monkeypatch.setattr(
        retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0
    )

    qpos, name, _score, _distance, metadata = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=shoulder_singular,
            reference_full_q=reference,
            current_targets=retargeter.current_world_targets(),
            profile=None,
        )
    )

    assert name == "shoulder_escape"
    assert np.allclose(qpos, shoulder_clean)
    assert metadata["expanded_for_shoulder_singularity"] is True
    preference = metadata["position_singularity_preference"]
    assert preference["fully_non_singular_count"] == 1
    assert preference["fallback_to_singular"] is False
    assert preference["selected"]["any_shoulder_singular"] is False

    retargeter.config.prefer_non_shoulder_singular_branch = False
    rollback = retargeter._position_singularity_metadata(shoulder_singular)
    assert rollback["any_shoulder_singular"] is False


def test_position_candidate_layer_certifies_shoulder_fallback_when_no_clean_branch(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_elbow_singular_branch = False
    retargeter.config.prefer_non_wrist_singular_branch = False
    retargeter.config.prefer_non_shoulder_singular_branch = True
    reference = np.full(12, 0.2, dtype=float)
    singular = reference.copy()
    singular[0] = 0.0
    candidate = dual_module._DualArmCandidate(
        "original", singular, "multitask", "original"
    )

    def analyze(qpos: np.ndarray) -> dict[str, object]:
        singular_side = abs(float(np.asarray(qpos)[0])) < 0.05
        return {
            "status_code": -1 if singular_side else 0,
            "label": "shoulder_singularity" if singular_side else "normal",
            "wrist_center_distance_m": 0.01 if singular_side else 0.20,
        }

    retargeter.set_position_singularity_analyzers(
        {"left": analyze, "right": analyze}
    )
    monkeypatch.setattr(
        retargeter,
        "_dual_arm_tracking_candidates",
        lambda **_kwargs: ([candidate], {"enabled": False, "status": "deferred"}),
    )
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {"score": 1.0},
        ),
    )
    monkeypatch.setattr(
        retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0
    )

    qpos, name, _score, _distance, metadata = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=singular,
            reference_full_q=reference,
            current_targets=retargeter.current_world_targets(),
            profile=None,
        )
    )

    assert name == "original"
    assert np.allclose(qpos, singular)
    preference = metadata["position_singularity_preference"]
    assert preference["fully_non_singular_count"] == 0
    assert preference["fallback_to_singular"] is True
    selected_left = preference["selected"]["sides"]["left"]
    assert selected_left["analyzer_available"] is True
    assert selected_left["is_shoulder_singular"] is True


def test_dual_arm_candidate_deadline_holds_without_fallback_or_auxiliary_expansion(monkeypatch) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.solve_time_budget_ms = 10.0
    reference = np.zeros(12, dtype=float)
    raw = np.deg2rad(np.arange(1.0, 13.0, dtype=float))
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    evaluation_calls: list[str] = []
    candidate_generation_calls: list[dict[str, object]] = []
    clock = {"value": 0.0}
    original_candidates = retargeter._dual_arm_tracking_candidates

    def fake_candidates(**kwargs):
        candidate_generation_calls.append(dict(kwargs))
        return original_candidates(**kwargs)

    def reject_candidate(*, candidate, **_kwargs):
        evaluation_calls.append(candidate.name)
        clock["value"] += 0.002
        return False, candidate.qpos.copy(), {"score": 1e18, "reject_reason": "test_reject"}

    monkeypatch.setattr(dual_module.time, "perf_counter", lambda: clock["value"])
    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", fake_candidates)
    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", reject_candidate)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile={"solve_total_ms": 0.0},
    )

    assert name == "local_step_hold"
    assert np.allclose(qpos, reference)
    assert len(evaluation_calls) == 3
    assert len(candidate_generation_calls) == 1
    assert metadata["reason"] == "branch_budget_exhausted_no_safe_candidate"
    assert metadata["candidate_eval_mode"] == "deadline_hold"
    assert metadata["candidate_eval_stop_reason"] == "branch_budget_exhausted_before_candidate"
    assert [stage["stage"] for stage in metadata["candidate_eval_stages"]] == ["base_primary"]


def test_dual_arm_candidate_deadline_accounts_for_pre_selection_frame_time(monkeypatch) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.solve_time_budget_ms = 10.0
    reference = np.zeros(12, dtype=float)
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    evaluation_calls: list[str] = []

    def fail_if_evaluated(*, candidate, **_kwargs):
        evaluation_calls.append(candidate.name)
        raise AssertionError("candidate evaluation must not start after the frame budget is exhausted")

    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", fail_if_evaluated)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=np.deg2rad(np.arange(1.0, 13.0, dtype=float)),
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
        elapsed_before_selection_ms=10.0,
    )

    assert name == "local_step_hold"
    assert np.allclose(qpos, reference)
    assert evaluation_calls == []
    assert metadata["candidate_eval_stop_reason"] == "branch_budget_exhausted_before_candidates"
    assert metadata["candidate_eval_pre_selection_elapsed_ms"] == 10.0


def test_candidate_certification_reserve_reassigns_existing_total_reserve() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.solve_time_budget_ms = 10.0
    retargeter.config.solve_iteration_reserve_ms = 5.0
    reference = np.zeros(12, dtype=float)
    raw = np.deg2rad(np.arange(1.0, 13.0, dtype=float))
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()

    retargeter.config.candidate_certification_reserve_ms = 3.5
    _qpos, _name, _score, _distance, reserved = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=raw,
            reference_full_q=reference,
            current_targets=targets,
            profile=None,
            elapsed_before_selection_ms=4.0,
        )
    )
    assert reserved["candidate_certification_reserve_ms"] == pytest.approx(3.5)
    assert reserved["candidate_post_selection_reserve_ms"] == pytest.approx(1.5)
    assert reserved["candidate_eval_branch_budget_ms"] == pytest.approx(4.5)

    retargeter.config.candidate_certification_reserve_ms = 0.0
    _qpos, _name, _score, _distance, rollback = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=raw,
            reference_full_q=reference,
            current_targets=targets,
            profile=None,
            elapsed_before_selection_ms=4.0,
        )
    )
    assert rollback["candidate_post_selection_reserve_ms"] == pytest.approx(5.0)
    assert rollback["candidate_eval_branch_budget_ms"] == pytest.approx(1.0)


def test_dual_arm_candidate_deadline_selects_best_already_safe_candidate(monkeypatch) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.solve_time_budget_ms = 10.0
    reference = np.zeros(12, dtype=float)
    raw = np.deg2rad(np.arange(1.0, 13.0, dtype=float))
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    evaluation_calls: list[str] = []
    clock = {"value": 0.0}

    def accept_candidate(*, candidate, **_kwargs):
        evaluation_calls.append(candidate.name)
        clock["value"] += 0.002
        return True, candidate.qpos.copy(), {"score": float(10 - len(evaluation_calls))}

    monkeypatch.setattr(dual_module.time, "perf_counter", lambda: clock["value"])
    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", accept_candidate)

    _qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile={"solve_total_ms": 0.0},
    )

    assert len(evaluation_calls) == 3
    assert name == evaluation_calls[1]
    assert metadata["candidate_eval_mode"] == "deadline_partial"
    assert metadata["candidate_eval_stopped_early"] is True
    assert metadata["collision_safe_count"] == 2


def test_dual_arm_candidate_layer_holds_on_hard_branch_jump() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[0] = np.deg2rad(100.0)
    targets = retargeter.current_world_targets()

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "local_step_hold"
    assert np.allclose(qpos, reference)
    assert metadata["reason"] == "no_progress_candidate"
    assert metadata["last_rejections"][-1]["reject_reason"] == "hard_branch_jump:left"


def test_dual_arm_position_passthrough_skips_local_step_hold_and_keeps_final_step_limit() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            local_step_candidate_selection_enabled=False,
            inactive_side_hold_enabled=False,
            inactive_side_hold_execution_cap_enabled=False,
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[0] = np.deg2rad(100.0)
    targets = retargeter.current_world_targets()

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )
    limited_qpos, step_metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=qpos,
    )

    assert name == "original"
    assert metadata["enabled"] is False
    assert metadata["reason"] == "local_step_candidate_selection_disabled"
    assert np.isclose(limited_qpos[0], np.deg2rad(10.0 / 15.0))
    assert step_metadata["step_limited"] is True


def test_dual_arm_component_progress_accepts_bounded_fine_tracking_step() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, component_progress=True)
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[0] = 0.002
    raw[6] = -0.002
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    targets["left"] = targets["left"].copy()
    targets["right"] = targets["right"].copy()
    targets["left"][0, 3] = -0.010
    targets["right"][0, 3] = 0.180

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert qpos[0] > reference[0]
    assert qpos[6] < reference[6]
    selected = metadata["selected_metadata"]
    assert selected["dual_progress_accept_reason"] == "dual_component_progress_bounded_score_regression"
    assert selected["dual_component_progress_ok"] is True
    assert selected["dual_max_position_progress_m"] >= 5e-4


def test_dual_arm_component_progress_still_holds_on_large_score_regression() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, component_progress=True)
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[0] = 0.002
    raw[6] = -0.020
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    targets["left"] = targets["left"].copy()
    targets["right"] = targets["right"].copy()
    targets["left"][0, 3] = -0.010
    targets["right"][0, 3] = 0.180

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "local_step_hold"
    assert np.allclose(qpos, reference)
    assert metadata["reason"] == "no_progress_candidate"
    assert metadata["last_rejections"][-1]["dual_component_progress_ok"] is False


def test_boundary_hold_accepts_other_joint_motion_with_ordinary_bounded_worsening() -> None:
    retargeter = _retargeter(_FakeDualArmSolver())
    reference = np.zeros(12, dtype=float)
    candidate_q = reference.copy()
    candidate_q[0] = 0.002

    def pose_metrics(
        qpos: np.ndarray,
        _targets: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        worsening = 0.001 if abs(float(qpos[0] - reference[0])) > 1e-9 else 0.0
        return {
            "left": {"translation_m": 0.10 + worsening, "rotation_deg": 0.0},
            "right": {"translation_m": 0.0, "rotation_deg": 0.0},
        }

    retargeter._dual_pose_metrics = pose_metrics  # type: ignore[method-assign]
    accepted, output, metadata = retargeter._evaluate_dual_arm_step_candidate(
        candidate=dual_module._DualArmCandidate(
            name="joint_limit_boundary_hold",
            qpos=candidate_q,
            source="joint_limit_boundary_hold",
            method="project_inward_limit_components",
            metadata={"joint_limit_boundary_hold_candidate": True},
        ),
        reference_full_q=reference,
        current_targets={"left": _target_pose(), "right": _target_pose()},
        step_scale=1.0,
    )

    assert accepted is True
    assert output[0] > reference[0]
    assert (
        metadata["dual_progress_accept_reason"]
        == "joint_limit_boundary_hold_bounded_worsening"
    )
    assert metadata["dual_boundary_hold_candidate"] is True
    assert metadata["dual_boundary_hold_bounded_worsening"] is True
    assert metadata["dual_recovery_state"] == "joint_limit_boundary_hold"


def test_boundary_hold_still_rejects_unbounded_pose_worsening() -> None:
    retargeter = _retargeter(_FakeDualArmSolver())
    reference = np.zeros(12, dtype=float)
    candidate_q = reference.copy()
    candidate_q[0] = 0.002

    def pose_metrics(
        qpos: np.ndarray,
        _targets: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        worsening = 0.20 if abs(float(qpos[0] - reference[0])) > 1e-9 else 0.0
        return {
            "left": {"translation_m": 0.10 + worsening, "rotation_deg": 0.0},
            "right": {"translation_m": 0.0, "rotation_deg": 0.0},
        }

    retargeter._dual_pose_metrics = pose_metrics  # type: ignore[method-assign]
    accepted, _output, metadata = retargeter._evaluate_dual_arm_step_candidate(
        candidate=dual_module._DualArmCandidate(
            name="joint_limit_boundary_hold",
            qpos=candidate_q,
            source="joint_limit_boundary_hold",
            method="project_inward_limit_components",
            metadata={"joint_limit_boundary_hold_candidate": True},
        ),
        reference_full_q=reference,
        current_targets={"left": _target_pose(), "right": _target_pose()},
        step_scale=1.0,
    )

    assert accepted is False
    assert metadata["reject_reason"] == "dual_local_step_no_pose_progress"
    assert metadata["dual_boundary_hold_candidate"] is True
    assert metadata["dual_boundary_hold_bounded_worsening"] is False


def test_dual_arm_activity_hold_reduces_unmoved_side_motion() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, loose_dynamic_worsening=True)
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[0] = 0.050
    raw[6] = 0.050
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)
    targets = retargeter.current_world_targets()
    retargeter._last_target_poses = {
        "left": targets["left"].copy(),
        "right": targets["right"].copy(),
    }
    targets["left"] = targets["left"].copy()
    targets["right"] = targets["right"].copy()
    targets["right"][0, 3] = 0.080

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert np.isclose(qpos[0], np.deg2rad(0.5))
    assert np.isclose(qpos[6], 0.050)
    assert metadata["inactive_side_hold_active"] is True
    selected = metadata["selected_metadata"]
    assert selected["step_scale"] == 1.0
    assert selected["inactive_side_hold_execution_cap"]["applied"] is True
    assert selected["inactive_side_hold_execution_cap"]["sides"]["left"]["applied"] is True
    assert selected["inactive_side_hold"]["sides"]["left"]["hold_weight"] == 1.0
    assert selected["inactive_side_hold"]["sides"]["left"]["unexplained_position_m"] <= 0.025


def test_dual_arm_inactive_execution_step_cap_limits_stationary_side() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[6] = np.deg2rad(3.0)
    target_activity = {
        "enabled": True,
        "right": {"inactive": True, "hold_weight": 1.0},
        "left": {"inactive": False, "hold_weight": 0.0},
    }
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
    )

    assert np.isclose(qpos[6], np.deg2rad(0.5))
    assert metadata["step_limited"] is True
    cap = metadata["inactive_side_hold_execution_cap"]
    assert cap["applied"] is True
    assert np.isclose(cap["sides"]["right"]["requested_max_abs_delta_deg"], 3.0)
    assert np.isclose(cap["sides"]["right"]["capped_max_abs_delta_deg"], 0.5)


def test_tracker_confirmed_inactive_side_freezes_below_static_error_thresholds() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[:6] = np.deg2rad([0.4, -0.3, 0.2, -0.1, 0.4, -0.2])
    candidate[6:] = np.deg2rad([-0.4, 0.3, -0.2, 0.1, -0.4, 0.2])
    target_activity = {
        "enabled": True,
        "left": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
        "right": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
    }

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
        current_targets={
            "left": _target_pose(x_m=0.029, rotation_deg=4.9),
            "right": _target_pose(x_m=0.029, rotation_deg=4.9),
        },
    )

    np.testing.assert_allclose(qpos, reference, atol=1e-12)
    cap = metadata["inactive_side_hold_execution_cap"]
    assert cap["applied"] is True
    for side in ("left", "right"):
        assert cap["sides"][side]["tracker_inactive_freeze"] is True
        assert cap["sides"][side]["effective_cap_deg"] == 0.0
        assert cap["sides"][side]["capped_max_abs_delta_deg"] == 0.0
        release = cap["sides"][side]["tracker_inactive_execution_release"]
        assert release["available"] is True
        assert release["velocity_release_scale"] == 0.0


def test_tracker_inactive_execution_release_linearly_adds_pose_error_weights() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[6] = np.deg2rad(0.4)
    target_activity = {
        "enabled": True,
        "left": {"inactive": False, "hold_weight": 0.0},
        "right": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
    }
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
        current_targets={
            "left": _target_pose(),
            "right": _target_pose(x_m=0.0475, rotation_deg=11.25),
        },
    )

    # (4.75 - 3.0) / (10.0 - 3.0) == 0.25 and
    # (11.25 - 5.0) / (30.0 - 5.0) == 0.25, hence 50% release.
    assert np.isclose(qpos[6], np.deg2rad(0.2))
    release = metadata["inactive_side_hold_execution_cap"]["sides"]["right"][
        "tracker_inactive_execution_release"
    ]
    assert release["reference_qpos_source"] == "execution_reference_qcmd_or_qref"
    assert np.isclose(release["translation_weight"], 0.25)
    assert np.isclose(release["rotation_weight"], 0.25)
    assert np.isclose(release["velocity_release_scale"], 0.5)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
        current_targets={
            "left": _target_pose(),
            "right": _target_pose(x_m=0.065, rotation_deg=17.5),
        },
    )

    # Both normalized weights are 0.5, so their equal-weight sum releases the
    # inactive cap completely and preserves the normal execution-step limit.
    assert np.isclose(qpos[6], np.deg2rad(0.4))
    release = metadata["inactive_side_hold_execution_cap"]["sides"]["right"][
        "tracker_inactive_execution_release"
    ]
    assert np.isclose(release["translation_weight"], 0.5)
    assert np.isclose(release["rotation_weight"], 0.5)
    assert np.isclose(release["velocity_release_scale"], 1.0)
    assert metadata["inactive_side_hold_execution_cap"]["sides"]["right"]["applied"] is False


def test_tracker_inactive_task_scale_release_converges_before_jitter_hold() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.inactive_side_hold_execution_release_translation_static_m = 0.0015
    retargeter.config.inactive_side_hold_execution_release_translation_full_m = 0.012
    retargeter.config.inactive_side_hold_execution_release_rotation_static_deg = 0.25
    retargeter.config.inactive_side_hold_execution_release_rotation_full_deg = 2.0
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[6] = np.deg2rad(0.4)
    target_activity = {
        "enabled": True,
        "left": {"inactive": False, "hold_weight": 0.0},
        "right": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
    }
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)

    converging, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
        current_targets={
            "left": _target_pose(),
            "right": _target_pose(x_m=0.012, rotation_deg=2.0),
        },
    )
    np.testing.assert_allclose(converging[6], candidate[6])
    release = metadata["inactive_side_hold_execution_cap"]["sides"]["right"][
        "tracker_inactive_execution_release"
    ]
    assert release["velocity_release_scale"] == pytest.approx(1.0)

    settled, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
        current_targets={
            "left": _target_pose(),
            "right": _target_pose(x_m=0.001, rotation_deg=0.2),
        },
    )
    np.testing.assert_allclose(settled[6], 0.0)
    side = metadata["inactive_side_hold_execution_cap"]["sides"]["right"]
    assert side["tracker_inactive_freeze"] is True


def test_candidate_local_step_uses_tracker_inactive_execution_release() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[6] = np.deg2rad(0.4)
    target_activity = {
        "enabled": True,
        "left": {"inactive": False, "hold_weight": 0.0},
        "right": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
    }
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)

    qpos, metadata = retargeter._limit_full_joint_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        step_scale=1.0,
        target_activity=target_activity,
        current_targets={
            "left": _target_pose(),
            "right": _target_pose(x_m=0.0475, rotation_deg=11.25),
        },
    )

    assert np.isclose(qpos[6], np.deg2rad(0.2))
    release = metadata["inactive_side_hold_execution_cap"]["sides"]["right"][
        "tracker_inactive_execution_release"
    ]
    assert release["available"] is True
    assert np.isclose(release["velocity_release_scale"], 0.5)


def test_tracker_inactive_execution_release_is_opt_in_for_position_only() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.inactive_side_hold_execution_release_enabled = False
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[6] = np.deg2rad(0.4)
    target_activity = {
        "enabled": True,
        "left": {"inactive": False, "hold_weight": 0.0},
        "right": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
    }
    retargeter.set_execution_dt_sec(1.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
        current_targets={"left": _target_pose(), "right": _target_pose(x_m=0.10)},
    )

    # Generic callers (including force-control callers) retain the historical
    # confirmed-inactive freeze unless a position entrypoint opts in.
    assert np.isclose(qpos[6], 0.0)
    side = metadata["inactive_side_hold_execution_cap"]["sides"]["right"]
    assert side["tracker_inactive_execution_release_enabled"] is False
    assert side["tracker_inactive_execution_release"] is None
    assert side["tracker_inactive_freeze"] is True


def test_build_command_passes_targets_to_inactive_execution_release() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    candidate = np.zeros(12, dtype=float)
    candidate[6] = np.deg2rad(5.0)
    solver.solve = lambda *, targets: candidate.copy()  # type: ignore[attr-defined]
    inactive_right = EffectorCommand(
        effector_name="right_arm",
        side="right",
        command_type="cartesian_pose",
        value=_target_pose(x_m=0.10),
        frame_name="arm_base",
        metadata={
            "tracker_motion_intent": {
                "state": "INACTIVE",
                "inactive": True,
                "hold_weight": 1.0,
            }
        },
    )
    context = DualArmTargetContext(
        timestamp=0.0,
        left_target=_target_pose(),
        right_target=_target_pose(x_m=0.10),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=None,
        right_cartesian=inactive_right,
        left_input_skipped=None,
        right_input_skipped=None,
    )

    command = retargeter.build_command_for_context(context)

    right = command.commands["right_arm"].metadata
    assert np.max(np.abs(command.commands["right_arm"].value)) > 0.0
    release = right["execution_step_metadata"]["inactive_side_hold_execution_cap"]["sides"][
        "right"
    ]["tracker_inactive_execution_release"]
    assert release["available"] is True
    assert "unavailable_reason" not in release
    assert release["velocity_release_scale"] == 1.0


def test_dual_arm_velocity_profile_and_explicit_execution_step_use_strictest_limit() -> None:
    solver = _FakeDualArmSolver()
    explicit_step = np.deg2rad(np.full(6, 0.25, dtype=float))
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=JointVelocityLimitProfile(
                name="fast_profile",
                max_velocity_deg_s=(1000.0,) * 6,
                max_dt_sec=1.0,
            ),
            left_execution_max_joint_step_rad=explicit_step,
            right_execution_max_joint_step_rad=explicit_step,
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[0] = np.deg2rad(3.0)
    candidate[6] = -np.deg2rad(4.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
    )

    assert np.isclose(qpos[0], np.deg2rad(0.25))
    assert np.isclose(qpos[6], -np.deg2rad(0.25))
    assert metadata["step_limited"] is True
    assert metadata["execution_step_source"] == "joint_velocity_profile+explicit_execution_max_joint_step"
    assert np.allclose(metadata["left_max_step_deg"], np.full(6, 0.25))
    assert np.allclose(metadata["right_max_step_deg"], np.full(6, 0.25))


def test_candidate_local_clip_scale_does_not_raise_final_execution_step() -> None:
    solver = _FakeDualArmSolver()
    explicit_step = np.deg2rad(np.full(6, 0.25, dtype=float))
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=JointVelocityLimitProfile(
                name="fast_profile",
                max_velocity_deg_s=(1000.0,) * 6,
                max_dt_sec=1.0,
            ),
            left_execution_max_joint_step_rad=explicit_step,
            right_execution_max_joint_step_rad=explicit_step,
            local_step_candidate_clip_scale=2.0,
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[0] = np.deg2rad(3.0)

    local_qpos, local_metadata = retargeter._limit_full_joint_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        step_scale=1.0,
    )
    final_qpos, final_metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=local_qpos,
    )

    assert np.isclose(local_qpos[0], np.deg2rad(0.5))
    assert local_metadata["candidate_local_step_clip_scale"] == pytest.approx(2.0)
    assert np.isclose(final_qpos[0], np.deg2rad(0.25))
    assert final_metadata["step_limited"] is True


def test_dual_arm_runtime_execution_step_update_limits_velocity_profile() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=JointVelocityLimitProfile(
                name="fast_profile",
                max_velocity_deg_s=(1000.0,) * 6,
                max_dt_sec=1.0,
            ),
        ),
        ik_solver=solver,
    )
    runtime_step = np.deg2rad(np.full(6, 0.2, dtype=float))
    retargeter.set_execution_max_joint_step(
        left_step_rad=runtime_step,
        right_step_rad=runtime_step,
    )
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[0] = np.deg2rad(3.0)
    candidate[6] = -np.deg2rad(4.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
    )

    assert np.isclose(qpos[0], np.deg2rad(0.2))
    assert np.isclose(qpos[6], -np.deg2rad(0.2))
    assert metadata["step_limited"] is True
    assert metadata["execution_step_source"] == "joint_velocity_profile+explicit_execution_max_joint_step"
    assert np.allclose(metadata["left_max_step_deg"], np.full(6, 0.2))
    assert np.allclose(metadata["right_max_step_deg"], np.full(6, 0.2))


def test_dual_arm_output_qpos_smoothing_runs_before_action_fk() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.5,
        ),
        ik_solver=solver,
    )
    raw_left = np.deg2rad(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    raw_right = np.deg2rad(np.array([-8.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    raw_full = retargeter._embed_arm_qpos(
        current_full_q=solver.current_q,
        left_qpos=raw_left,
        right_qpos=raw_right,
    )

    first_full, first_left, first_right, first_metadata = retargeter._smooth_output_qpos_before_action(
        active_q=raw_full,
        left_qpos=raw_left,
        right_qpos=raw_right,
    )
    second_full, second_left, second_right, second_metadata = retargeter._smooth_output_qpos_before_action(
        active_q=np.zeros_like(raw_full),
        left_qpos=np.zeros(6, dtype=float),
        right_qpos=np.zeros(6, dtype=float),
    )

    np.testing.assert_allclose(first_left, raw_left)
    np.testing.assert_allclose(first_right, raw_right)
    assert first_metadata["initialized"] is True
    assert first_metadata["applied"] is False
    np.testing.assert_allclose(first_full, raw_full)
    np.testing.assert_allclose(second_left[0], np.deg2rad(5.0))
    np.testing.assert_allclose(second_right[0], -np.deg2rad(4.0))
    assert second_metadata["applied"] is True
    assert second_metadata["stage"] == "post_ik_pre_action_fk"
    assert second_metadata["downstream_command_stage"] == "interpolation_only"
    assert np.isclose(second_metadata["left"]["output_target_joint_deg"][0], 5.0)
    assert np.isclose(second_metadata["right"]["output_target_joint_deg"][0], -4.0)


def test_single_step_output_smoothing_cannot_reenter_wrist_singularity() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.14,
        ),
        ik_solver=solver,
    )
    reference = solver.current_q.copy()
    reference[4] = np.deg2rad(-2.2)
    reference[10] = np.deg2rad(-30.0)
    candidate = reference.copy()
    candidate[4] = np.deg2rad(-2.9)
    candidate[10] = np.deg2rad(-29.0)

    output, output_left, _output_right, metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=candidate,
            left_qpos=candidate[:6],
            right_qpos=candidate[6:12],
            reference_full_q=reference,
        )
    )

    assert metadata["wrist_q5_escape"]["applied"] is True
    # The fixture's deliberately small q5 velocity limit makes the escape
    # progress over multiple outer ticks; it must still move outward rather
    # than re-entering through zero.
    assert -4.0 < np.rad2deg(output_left[4]) < -2.2
    assert abs(np.rad2deg(output[10])) > 3.0


def test_single_step_candidate_q5_hysteresis_stabilizes_inward_escape_toggle() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_q5_hysteresis_release_deg=6.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=solver,
    )
    reference = solver.current_q.copy()
    reference[4] = np.deg2rad(-5.0)
    candidate = reference.copy()
    candidate[4] = np.deg2rad(-3.7)

    output, metadata = retargeter._stabilize_wrist_q5_candidate(
        reference_full_q=reference,
        candidate_full_q=candidate,
    )

    assert metadata["applied"] is True
    assert metadata["sides"]["left"]["hysteresis_hold"] is True
    assert np.rad2deg(output[4]) == pytest.approx(-5.0)


def test_dual_arm_output_qpos_smoothing_adapts_to_valid_tracker_speed() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.18,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.7,
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    activity = {
        "left": {
            "evidence": {
                "quality_valid": True,
                "linear_speed_p90_m_s": 0.30,
                "angular_speed_p90_rad_s": 0.2,
            }
        }
    }

    alpha, metadata = retargeter._effective_output_qpos_smoothing_alpha(activity)

    assert 0.18 < alpha["left"] <= 0.7
    assert alpha["right"] == pytest.approx(0.18)
    assert metadata["selection"] == "per_side_tracker_speed"
    assert metadata["left"]["reason"] == "tracker_speed"
    assert metadata["left"]["requested_alpha"] == 0.7
    assert metadata["left"]["transition"] == "attack"
    assert metadata["right"]["reason"] == "invalid_or_missing_speed"


def test_normal_zone_fast_path_bypasses_only_ordinary_tracking_side() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.14,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.45,
            output_qpos_normal_fast_path_enabled=True,
            output_qpos_normal_fast_path_alpha=1.0,
            output_qpos_normal_fast_path_guard_deg=4.0,
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    reference[[2, 4, 8, 10]] = np.deg2rad([30.0, 20.0, 30.0, 20.0])
    candidate = reference.copy()
    candidate[0] += np.deg2rad(2.0)
    candidate[6] -= np.deg2rad(2.0)
    retargeter._last_output_qpos_smoothing_full_q = reference.copy()
    retargeter._last_wrist_coaxial_metadata = {
        "sides": {
            "left": {"effective_null_scale": 1.0},
            "right": {"effective_null_scale": 1.0},
        }
    }
    activity = {
        side: {
            "evidence": {
                "quality_valid": True,
                "linear_speed_p90_m_s": 0.05,
                "angular_speed_p90_rad_s": 0.2,
            }
        }
        for side in ("left", "right")
    }

    output, left, right, metadata = retargeter._smooth_output_qpos_before_action(
        active_q=candidate,
        left_qpos=candidate[:6],
        right_qpos=candidate[6:12],
        reference_full_q=reference,
        target_activity=activity,
        admission_status="tracking",
    )

    np.testing.assert_allclose(output, candidate)
    np.testing.assert_allclose(left, candidate[:6])
    np.testing.assert_allclose(right, candidate[6:12])
    assert metadata["alpha_by_side"] == {"left": 1.0, "right": 1.0}
    assert metadata["normal_fast_path"]["left"]["eligible"] is True
    assert metadata["normal_fast_path"]["right"]["eligible"] is True
    assert metadata["applied"] is False


def test_normal_zone_fast_path_filters_micro_reversals_and_syncs_alpha_state() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.14,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.45,
            output_qpos_normal_fast_path_enabled=True,
            output_qpos_normal_fast_path_alpha=0.70,
            output_qpos_normal_fast_path_guard_deg=4.0,
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    reference[[2, 4, 8, 10]] = np.deg2rad([30.0, 20.0, 30.0, 20.0])
    retargeter._last_output_qpos_smoothing_full_q = reference.copy()
    retargeter._last_wrist_coaxial_metadata = {
        "sides": {
            "left": {"effective_null_scale": 1.0},
            "right": {"effective_null_scale": 1.0},
        }
    }
    activity = {
        side: {
            "evidence": {
                "quality_valid": True,
                "linear_speed_p90_m_s": 0.05,
                "angular_speed_p90_rad_s": 0.2,
            }
        }
        for side in ("left", "right")
    }
    positive = reference.copy()
    positive[0] += np.deg2rad(1.0)

    first, _left, _right, first_metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=positive,
            left_qpos=positive[:6],
            right_qpos=positive[6:12],
            reference_full_q=reference,
            target_activity=activity,
            admission_status="tracking",
        )
    )

    assert np.rad2deg(first[0] - reference[0]) == pytest.approx(0.70)
    assert first_metadata["alpha_by_side"]["left"] == pytest.approx(0.70)
    assert first_metadata["normal_fast_path"]["left"]["eligible"] is True
    assert retargeter._output_qpos_smoothing_effective_alpha["left"] == pytest.approx(
        0.70
    )

    negative = reference.copy()
    negative[0] -= np.deg2rad(1.0)
    second, _left, _right, _metadata = retargeter._smooth_output_qpos_before_action(
        active_q=negative,
        left_qpos=negative[:6],
        right_qpos=negative[6:12],
        reference_full_q=first,
        target_activity=activity,
        admission_status="tracking",
    )

    # The raw target reverses by 2 degrees.  The responsive EMA admits only
    # 1.19 degrees on this frame instead of passing the full reversal through.
    assert np.rad2deg(second[0] - first[0]) == pytest.approx(-1.19)


def test_active_admittance_fast_path_is_modifier_only_and_inherits_guards() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.14,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.45,
            output_qpos_normal_fast_path_enabled=True,
            output_qpos_normal_fast_path_alpha=0.55,
            output_qpos_normal_fast_path_guard_deg=4.0,
            output_qpos_admittance_fast_path_enabled=True,
            output_qpos_admittance_fast_path_alpha=0.85,
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    reference = np.zeros(12, dtype=float)
    reference[[2, 4, 8, 10]] = np.deg2rad([30.0, 20.0, 30.0, 20.0])
    candidate = reference.copy()
    candidate[0] += np.deg2rad(1.0)
    candidate[6] -= np.deg2rad(1.0)
    retargeter._last_output_qpos_smoothing_full_q = reference.copy()
    retargeter._last_wrist_coaxial_metadata = {
        "sides": {
            "left": {"effective_null_scale": 1.0},
            "right": {"effective_null_scale": 1.0},
        }
    }
    quality = {"quality_valid": False}
    activity = {
        "left": {
            "source": "position_admittance_active_modifier",
            "modifier_active": True,
            "modifier_qpos_fast_path_alpha": 0.80,
            "inactive": False,
            "evidence": dict(quality),
        },
        "right": {"evidence": dict(quality)},
    }

    output, _left, _right, metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=candidate,
            left_qpos=candidate[:6],
            right_qpos=candidate[6:12],
            reference_full_q=reference,
            target_activity=activity,
            admission_status="tracking",
        )
    )

    assert metadata["alpha_by_side"]["left"] == pytest.approx(0.80)
    assert metadata["alpha_by_side"]["right"] == pytest.approx(0.14)
    assert metadata["admittance_fast_path"]["left"]["eligible"] is True
    assert metadata["admittance_fast_path"]["right"]["eligible"] is False
    assert (
        metadata["admittance_fast_path"]["left"]["tracker_quality_required"]
        is False
    )
    assert np.rad2deg(output[0] - reference[0]) == pytest.approx(0.80)
    assert np.rad2deg(output[6] - reference[6]) == pytest.approx(-0.14)

    retargeter._last_wrist_coaxial_metadata["sides"]["left"] = {
        "effective_null_scale": 0.5
    }
    _output, _left, _right, guarded = (
        retargeter._smooth_output_qpos_before_action(
            active_q=candidate,
            left_qpos=candidate[:6],
            right_qpos=candidate[6:12],
            reference_full_q=output,
            target_activity=activity,
            admission_status="tracking",
        )
    )
    assert guarded["normal_fast_path"]["left"]["eligible"] is False
    assert guarded["admittance_fast_path"]["left"]["eligible"] is False
    assert guarded["alpha_by_side"]["left"] < 0.80


def test_rotational_admittance_owns_contact_and_return_smoothing_endpoints() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.14,
            output_qpos_smoothing_adaptive_enabled=False,
            output_qpos_admittance_fast_path_enabled=True,
            output_qpos_admittance_fast_path_alpha=0.85,
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    reference = np.zeros(12, dtype=float)
    reference[[2, 4, 8, 10]] = np.deg2rad([30.0, 20.0, 30.0, 20.0])
    candidate = reference.copy()
    candidate[0] += np.deg2rad(1.0)
    retargeter._last_output_qpos_smoothing_full_q = reference.copy()
    retargeter._last_wrist_coaxial_metadata = {
        "sides": {
            "left": {"effective_null_scale": 1.0},
            "right": {"effective_null_scale": 1.0},
        }
    }

    def smooth(alpha: float, phase: str) -> tuple[np.ndarray, dict[str, object]]:
        output, _left, _right, metadata = (
            retargeter._smooth_output_qpos_before_action(
                active_q=candidate,
                left_qpos=candidate[:6],
                right_qpos=candidate[6:12],
                reference_full_q=reference,
                target_activity={
                    "left": {
                        "source": "position_admittance_active_modifier",
                        "modifier_active": True,
                        "modifier_rotation_offset_norm_rad": 0.01,
                        "modifier_qpos_fast_path_alpha": alpha,
                        "modifier_phase": phase,
                        "inactive": False,
                    },
                    "right": {},
                },
                admission_status="tracking",
            )
        )
        return output, metadata

    contact, contact_metadata = smooth(0.55, "contact")
    assert contact_metadata["alpha_by_side"]["left"] == pytest.approx(0.55)
    assert np.rad2deg(contact[0] - reference[0]) == pytest.approx(0.55)
    assert contact_metadata["admittance_fast_path"]["left"][
        "rotational_modifier"
    ] is True

    retargeter._last_output_qpos_smoothing_full_q = reference.copy()
    returning, return_metadata = smooth(0.85, "return")
    assert return_metadata["alpha_by_side"]["left"] == pytest.approx(0.85)
    assert np.rad2deg(returning[0] - reference[0]) == pytest.approx(0.85)
    assert return_metadata["admittance_fast_path"]["left"][
        "modifier_phase"
    ] == "return"


@pytest.mark.parametrize(
    ("creep_deg", "expected_confirmation_step_deg"),
    ((0.0, -0.055), (0.005, -0.051)),
)
def test_active_admittance_softens_one_tiny_qpos_reversal_then_confirms_it(
    creep_deg: float,
    expected_confirmation_step_deg: float,
) -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.14,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.45,
            output_qpos_normal_fast_path_enabled=True,
            output_qpos_normal_fast_path_alpha=0.55,
            output_qpos_normal_fast_path_guard_deg=4.0,
            output_qpos_admittance_fast_path_enabled=True,
            output_qpos_admittance_fast_path_alpha=0.85,
            output_qpos_admittance_micro_reversal_hold_deg=0.05,
            output_qpos_admittance_micro_reversal_creep_deg=creep_deg,
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    reference = np.zeros(12, dtype=float)
    reference[[2, 4, 8, 10]] = np.deg2rad([30.0, 20.0, 30.0, 20.0])
    retargeter._last_output_qpos_smoothing_full_q = reference.copy()
    retargeter._last_wrist_coaxial_metadata = {
        "sides": {
            "left": {"effective_null_scale": 1.0},
            "right": {"effective_null_scale": 1.0},
        }
    }
    activity = {
        "left": {
            "source": "position_admittance_active_modifier",
            "modifier_active": True,
            "modifier_qpos_fast_path_alpha": 0.80,
            "inactive": False,
            "evidence": {"quality_valid": False},
        },
        "right": {"evidence": {"quality_valid": False}},
    }

    positive = reference.copy()
    positive[0] += np.deg2rad(0.05)
    first, _left, _right, _metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=positive,
            left_qpos=positive[:6],
            right_qpos=positive[6:12],
            reference_full_q=reference,
            target_activity=activity,
            admission_status="tracking",
        )
    )
    assert np.rad2deg(first[0] - reference[0]) == pytest.approx(0.04)
    retargeter._commit_admitted_velocity(
        reference_full_q=reference,
        admitted_full_q=first,
    )

    reversed_candidate = first.copy()
    reversed_candidate[0] -= np.deg2rad(0.06875)
    held, _left, _right, held_metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=reversed_candidate,
            left_qpos=reversed_candidate[:6],
            right_qpos=reversed_candidate[6:12],
            reference_full_q=first,
            target_activity=activity,
            admission_status="tracking",
        )
    )
    assert np.rad2deg(held[0] - first[0]) == pytest.approx(-creep_deg)
    expected_held = [0] if creep_deg == 0.0 else []
    expected_softened = [] if creep_deg == 0.0 else [0]
    assert held_metadata["admittance_fast_path"]["left"][
        "micro_reversal_held_joint_indices"
    ] == expected_held
    assert held_metadata["admittance_fast_path"]["left"][
        "micro_reversal_softened_joint_indices"
    ] == expected_softened
    assert held_metadata["admittance_fast_path"]["left"][
        "micro_reversal_pending_joint_indices"
    ] == [0]
    assert held_metadata["admittance_fast_path"]["left"][
        "micro_reversal_creep_deg"
    ] == pytest.approx(creep_deg)
    assert held_metadata["left"]["admittance_fast_path"][
        "micro_reversal_held_joint_indices"
    ] == expected_held
    assert held_metadata["left"]["admittance_fast_path"][
        "micro_reversal_softened_joint_indices"
    ] == expected_softened
    retargeter._commit_admitted_velocity(
        reference_full_q=first,
        admitted_full_q=held,
    )

    confirmed, _left, _right, confirmed_metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=reversed_candidate,
            left_qpos=reversed_candidate[:6],
            right_qpos=reversed_candidate[6:12],
            reference_full_q=held,
            target_activity=activity,
            admission_status="tracking",
        )
    )
    assert np.rad2deg(confirmed[0] - held[0]) == pytest.approx(
        expected_confirmation_step_deg
    )
    assert confirmed_metadata["admittance_fast_path"]["left"][
        "micro_reversal_held_joint_indices"
    ] == []
    assert confirmed_metadata["admittance_fast_path"]["left"][
        "micro_reversal_pending_joint_indices"
    ] == []


def test_active_admittance_reversal_creep_cannot_exceed_hold_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="micro_reversal_creep_deg",
    ):
        DualArmCollisionAwareRetargeterConfig(
            output_qpos_admittance_micro_reversal_hold_deg=0.05,
            output_qpos_admittance_micro_reversal_creep_deg=0.051,
        )


def test_normal_zone_fast_path_graces_soft_quality_dropout_but_not_hard_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [10.0]
    monkeypatch.setattr(target_solver_module.time, "monotonic", lambda: clock[0])
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.14,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.45,
            output_qpos_normal_fast_path_enabled=True,
            output_qpos_normal_fast_path_alpha=0.70,
            output_qpos_normal_fast_path_guard_deg=4.0,
            output_qpos_normal_fast_path_quality_grace_sec=0.05,
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    reference[[2, 4, 8, 10]] = np.deg2rad([30.0, 20.0, 30.0, 20.0])
    retargeter._last_output_qpos_smoothing_full_q = reference.copy()
    retargeter._last_wrist_coaxial_metadata = {
        "sides": {
            "left": {"effective_null_scale": 1.0},
            "right": {"effective_null_scale": 1.0},
        }
    }
    valid_activity = {
        side: {
            "evidence": {
                "quality_valid": True,
                "linear_speed_p90_m_s": 0.05,
                "angular_speed_p90_rad_s": 0.2,
            }
        }
        for side in ("left", "right")
    }
    invalid_activity = {
        side: {"evidence": {"quality_valid": False}}
        for side in ("left", "right")
    }
    candidate = reference.copy()
    candidate[0] += np.deg2rad(1.0)
    first, _left, _right, _metadata = retargeter._smooth_output_qpos_before_action(
        active_q=candidate,
        left_qpos=candidate[:6],
        right_qpos=candidate[6:12],
        reference_full_q=reference,
        target_activity=valid_activity,
        admission_status="tracking",
    )

    clock[0] = 10.02
    candidate[0] += np.deg2rad(1.0)
    second, _left, _right, grace_metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=candidate,
            left_qpos=candidate[:6],
            right_qpos=candidate[6:12],
            reference_full_q=first,
            target_activity=invalid_activity,
            admission_status="tracking",
        )
    )

    left_grace = grace_metadata["normal_fast_path"]["left"]
    assert left_grace["tracker_quality_valid"] is False
    assert left_grace["tracker_quality_grace_active"] is True
    assert left_grace["eligible"] is True
    assert grace_metadata["alpha_by_side"]["left"] == pytest.approx(0.70)
    assert retargeter._output_qpos_smoothing_effective_alpha["left"] == pytest.approx(
        0.70
    )

    # A null projection is a hard singularity guard.  It must disable the
    # normal fast path immediately even while the tracker-quality grace is live.
    clock[0] = 10.03
    retargeter._last_wrist_coaxial_metadata["sides"]["left"] = {
        "effective_null_scale": 0.5
    }
    candidate[0] += np.deg2rad(1.0)
    _third, _left, _right, hard_metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=candidate,
            left_qpos=candidate[:6],
            right_qpos=candidate[6:12],
            reference_full_q=second,
            target_activity=invalid_activity,
            admission_status="tracking",
        )
    )
    left_hard = hard_metadata["normal_fast_path"]["left"]
    assert left_hard["tracker_quality_grace_active"] is True
    assert left_hard["null_projection_inactive"] is False
    assert left_hard["eligible"] is False
    assert hard_metadata["alpha_by_side"]["left"] < 0.70

    # With all hard guards healthy again, an expired quality window falls back
    # continuously from the synchronized 0.70 state rather than stale 0.14.
    clock[0] = 10.10
    retargeter._last_wrist_coaxial_metadata["sides"]["left"] = {
        "effective_null_scale": 1.0
    }
    candidate[0] += np.deg2rad(1.0)
    _fourth, _left, _right, expired_metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=candidate,
            left_qpos=candidate[:6],
            right_qpos=candidate[6:12],
            reference_full_q=second,
            target_activity=invalid_activity,
            admission_status="tracking",
        )
    )
    left_expired = expired_metadata["normal_fast_path"]["left"]
    assert left_expired["tracker_quality_grace_active"] is False
    assert left_expired["eligible"] is False
    assert 0.14 < expired_metadata["alpha_by_side"]["left"] < 0.70


def test_normal_zone_fast_path_stays_off_inside_wrist_guard() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.14,
            output_qpos_smoothing_fast_alpha=0.45,
            output_qpos_normal_fast_path_enabled=True,
            output_qpos_normal_fast_path_alpha=1.0,
            output_qpos_normal_fast_path_guard_deg=4.0,
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    reference[[2, 8, 10]] = np.deg2rad([30.0, 30.0, 20.0])
    reference[4] = np.deg2rad(3.5)
    candidate = reference.copy()
    candidate[0] += np.deg2rad(2.0)
    retargeter._last_output_qpos_smoothing_full_q = reference.copy()
    retargeter._last_wrist_coaxial_metadata = {
        "sides": {
            "left": {"effective_null_scale": 0.5},
            "right": {"effective_null_scale": 1.0},
        }
    }

    output, _left, _right, metadata = retargeter._smooth_output_qpos_before_action(
        active_q=candidate,
        left_qpos=candidate[:6],
        right_qpos=candidate[6:12],
        reference_full_q=reference,
        target_activity={
            "left": {
                "evidence": {
                    "quality_valid": True,
                    "linear_speed_p90_m_s": 0.05,
                    "angular_speed_p90_rad_s": 0.2,
                }
            }
        },
        admission_status="tracking",
    )

    assert metadata["normal_fast_path"]["left"]["eligible"] is False
    assert metadata["normal_fast_path"]["left"]["outside_joint_guard"] is False
    assert metadata["normal_fast_path"]["left"]["null_projection_inactive"] is False
    assert not np.allclose(output, candidate)


def test_q5_hysteresis_latch_arms_only_after_admitted_escape() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_q5_hysteresis_release_deg=6.0,
            wrist_q5_hysteresis_requires_escape_latch=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=solver,
    )
    ordinary_reference = np.zeros(12, dtype=float)
    ordinary_reference[4] = np.deg2rad(-5.0)
    ordinary_candidate = ordinary_reference.copy()
    ordinary_candidate[4] = np.deg2rad(-3.7)

    ordinary_output, ordinary_metadata = retargeter._stabilize_wrist_q5_candidate(
        reference_full_q=ordinary_reference,
        candidate_full_q=ordinary_candidate,
    )

    assert np.rad2deg(ordinary_output[4]) == pytest.approx(-3.7)
    assert ordinary_metadata["sides"]["left"]["hysteresis_allowed"] is False
    assert ordinary_metadata["sides"]["left"]["hysteresis_hold"] is False

    escape_reference = np.zeros(12, dtype=float)
    escape_reference[4] = np.deg2rad(-3.4)
    escape_candidate = escape_reference.copy()
    escape_candidate[4] = np.deg2rad(-2.0)
    escaped, escape_metadata = retargeter._stabilize_wrist_q5_candidate(
        reference_full_q=escape_reference,
        candidate_full_q=escape_candidate,
    )
    latch_metadata = retargeter._update_wrist_q5_hysteresis_escape_latch(
        reference_full_q=escape_reference,
        admitted_full_q=escaped,
        status="tracking",
        candidate_metadata=escape_metadata,
        output_metadata=None,
    )

    assert latch_metadata["sides"]["left"]["transition"] == (
        "armed_from_admitted_escape"
    )
    assert retargeter._wrist_q5_hysteresis_escape_latch["left"] is True

    inward_candidate = escaped.copy()
    inward_candidate[4] = np.deg2rad(-3.0)
    held, held_metadata = retargeter._stabilize_wrist_q5_candidate(
        reference_full_q=escaped,
        candidate_full_q=inward_candidate,
    )

    assert held_metadata["sides"]["left"]["hysteresis_allowed"] is True
    assert held_metadata["sides"]["left"]["hysteresis_hold"] is True
    assert held[4] == pytest.approx(escaped[4])


def test_q5_latch_mode_holds_only_first_inward_boundary_entry() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_q5_hysteresis_requires_escape_latch=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    reference = np.zeros(12, dtype=float)
    reference[4] = np.deg2rad(-3.7)
    reference[10] = np.deg2rad(20.0)
    inward = reference.copy()
    inward[1] += 0.01
    inward[4] = np.deg2rad(-2.8)

    held = retargeter._wrist_q5_entry_boundary_hold_candidate(
        ordinary_q=inward,
        reference_full_q=reference,
    )

    assert held is not None
    assert held.source == "wrist_q5_boundary_hold"
    assert held.qpos[4] == pytest.approx(reference[4])
    assert held.qpos[1] == pytest.approx(inward[1])
    assert held.metadata["held_sides"]["left"]["reason"] == (
        "inward_entry_or_near_crossing"
    )

    outward = reference.copy()
    outward[4] = np.deg2rad(-4.2)
    assert (
        retargeter._wrist_q5_entry_boundary_hold_candidate(
            ordinary_q=outward,
            reference_full_q=reference,
        )
        is None
    )

    inside = reference.copy()
    inside[4] = np.deg2rad(-2.5)
    inside_candidate = inside.copy()
    inside_candidate[4] = np.deg2rad(-1.5)
    assert (
        retargeter._wrist_q5_entry_boundary_hold_candidate(
            ordinary_q=inside_candidate,
            reference_full_q=inside,
        )
        is None
    )

    rollback = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_q5_hysteresis_requires_escape_latch=False,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    assert (
        rollback._wrist_q5_entry_boundary_hold_candidate(
            ordinary_q=inward,
            reference_full_q=reference,
        )
        is None
    )


def test_q5_entry_boundary_hold_bypasses_auxiliary_escape_search(
    monkeypatch,
) -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            wrist_q5_hysteresis_requires_escape_latch=True,
            ik_safety_profile=replace(
                _profile(),
                local_step_singularity_escape_enabled=True,
                local_step_singularity_escape_directional_enabled=True,
            ),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    reference = np.zeros(12, dtype=float)
    reference[4] = np.deg2rad(-3.7)
    reference[10] = np.deg2rad(20.0)
    ordinary = reference.copy()
    ordinary[1] += 0.002
    ordinary[4] = np.deg2rad(-2.8)
    candidate = target_solver_module._DualArmCandidate(
        name="ordinary_inward_q5",
        qpos=ordinary,
        source="multitask",
        method="ordinary_inward_q5",
    )
    calls: list[dict[str, object]] = []

    def fake_candidates(**kwargs):
        calls.append(dict(kwargs))
        return [candidate], {"enabled": True, "status": "active"}

    def fake_evaluate(*, candidate, step_scale, **_kwargs):
        return True, candidate.qpos.copy(), {
            "candidate_metadata": dict(candidate.metadata),
            "score": 0.0,
            "step_scale": float(step_scale),
        }

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", fake_candidates)
    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", fake_evaluate)

    qpos, name, _score, _distance, metadata = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=ordinary,
            reference_full_q=reference,
            current_targets=retargeter.current_world_targets(),
            profile=None,
        )
    )

    assert name == "wrist_q5_entry_boundary_hold"
    assert metadata["selected_source"] == "wrist_q5_boundary_hold"
    assert metadata["candidate_eval_mode"] == "wrist_q5_entry_boundary_hold"
    assert qpos[4] == pytest.approx(reference[4])
    assert qpos[1] == pytest.approx(ordinary[1])
    assert len(calls) == 1


def test_dual_arm_output_qpos_smoothing_uses_motion_window_validity() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.18,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.7,
        ),
        ik_solver=_FakeDualArmSolver(),
    )

    alpha, metadata = retargeter._effective_output_qpos_smoothing_alpha(
        {
            "right": {
                "evidence": {
                    "linear_speed_p90_m_s": 0.30,
                    "angular_speed_p90_rad_s": 0.2,
                },
                "motion_window": {"quality_valid": True},
            }
        }
    )

    assert alpha["left"] == pytest.approx(0.18)
    assert 0.18 < alpha["right"] <= 0.7
    assert metadata["right"]["reason"] == "tracker_speed"
    assert metadata["right"]["requested_alpha"] == 0.7


def test_dual_arm_output_qpos_smoothing_ignores_invalid_tracker_speed() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.18,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.7,
        ),
        ik_solver=_FakeDualArmSolver(),
    )

    alpha, metadata = retargeter._effective_output_qpos_smoothing_alpha(
        {"left": {"evidence": {"quality_valid": False, "linear_speed_p90_m_s": 1.0}}}
    )

    assert alpha == {"left": pytest.approx(0.18), "right": pytest.approx(0.18)}
    assert metadata["left"]["reason"] == "invalid_or_missing_speed"


def test_dual_arm_output_qpos_smoothing_applies_distinct_side_alphas() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.14,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.45,
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    retargeter._last_output_qpos_smoothing_full_q = np.zeros(12, dtype=float)
    activity = {
        "left": {
            "evidence": {
                "quality_valid": True,
                "linear_speed_p90_m_s": 0.30,
                "angular_speed_p90_rad_s": 0.2,
            }
        },
        "right": {"evidence": {"quality_valid": False}},
    }
    _, left, right, metadata = retargeter._smooth_output_qpos_before_action(
        active_q=np.ones(12, dtype=float),
        left_qpos=np.ones(6, dtype=float),
        right_qpos=np.ones(6, dtype=float),
        target_activity=activity,
    )

    assert metadata["left"]["alpha"] > metadata["right"]["alpha"]
    assert np.all(left > right)


def test_dual_arm_current_qpos_sync_can_preserve_output_qpos_smoothing_state() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.5,
        ),
        ik_solver=solver,
    )
    raw_left = np.deg2rad(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    raw_right = np.deg2rad(np.array([-8.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    raw_full = retargeter._embed_arm_qpos(
        current_full_q=solver.current_q,
        left_qpos=raw_left,
        right_qpos=raw_right,
    )

    retargeter._smooth_output_qpos_before_action(
        active_q=raw_full,
        left_qpos=raw_left,
        right_qpos=raw_right,
    )
    retargeter.set_current_qpos_pair(
        left_qpos=raw_left,
        right_qpos=raw_right,
        reset_tracking=False,
    )
    _, preserved_left, preserved_right, preserved_metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=np.zeros_like(raw_full),
            left_qpos=np.zeros(6, dtype=float),
            right_qpos=np.zeros(6, dtype=float),
        )
    )

    assert preserved_metadata["initialized"] is False
    assert preserved_metadata["applied"] is True
    np.testing.assert_allclose(preserved_left[0], np.deg2rad(5.0))
    np.testing.assert_allclose(preserved_right[0], -np.deg2rad(4.0))

    retargeter.set_current_qpos_pair(
        left_qpos=raw_left,
        right_qpos=raw_right,
        reset_tracking=True,
    )
    _, reset_left, reset_right, reset_metadata = retargeter._smooth_output_qpos_before_action(
        active_q=np.zeros_like(raw_full),
        left_qpos=np.zeros(6, dtype=float),
        right_qpos=np.zeros(6, dtype=float),
    )

    assert reset_metadata["initialized"] is True
    assert reset_metadata["applied"] is False
    np.testing.assert_allclose(reset_left, np.zeros(6, dtype=float))
    np.testing.assert_allclose(reset_right, np.zeros(6, dtype=float))


def test_single_step_runtime_sync_preserves_velocity_and_rebases_smoothing() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_step_mode="single_step_singularity_aware",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.5,
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    stale_qref = np.deg2rad(
        np.array(
            [
                0.0,
                0.0,
                0.0,
                15.0,
                1.0,
                -15.0,
                0.0,
                0.0,
                0.0,
                -12.0,
                -1.0,
                12.0,
            ]
        )
    )
    accepted_qcmd = np.deg2rad(
        np.array(
            [
                0.0,
                0.0,
                0.0,
                2.0,
                3.0,
                -2.0,
                0.0,
                0.0,
                0.0,
                -3.0,
                -3.0,
                3.0,
            ]
        )
    )
    admitted_velocity = np.linspace(-0.6, 0.5, 12)
    retargeter._last_output_qpos_smoothing_full_q = stale_qref.copy()
    retargeter._last_admitted_velocity_full = admitted_velocity.copy()

    retargeter.set_current_qpos_pair(
        left_qpos=accepted_qcmd[:6],
        right_qpos=accepted_qcmd[6:],
        reset_tracking=False,
    )

    np.testing.assert_allclose(
        retargeter._last_admitted_velocity_full,
        admitted_velocity,
    )
    np.testing.assert_allclose(
        retargeter._last_output_qpos_smoothing_full_q,
        accepted_qcmd,
    )

    retargeter.set_current_qpos_pair(
        left_qpos=accepted_qcmd[:6],
        right_qpos=accepted_qcmd[6:],
        reset_tracking=True,
    )

    np.testing.assert_allclose(
        retargeter._last_admitted_velocity_full,
        np.zeros(12),
    )
    assert retargeter._last_output_qpos_smoothing_full_q is None


def test_dual_arm_explicit_execution_step_is_fallback_without_velocity_profile() -> None:
    solver = _FakeDualArmSolver()
    explicit_step = np.deg2rad(np.full(6, 0.25, dtype=float))
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            left_execution_max_joint_step_rad=explicit_step,
            right_execution_max_joint_step_rad=explicit_step,
        ),
        ik_solver=solver,
    )
    retargeter.joint_velocity_profile = None
    retargeter.config.joint_velocity_profile = None
    retargeter._full_execution_joint_step = retargeter._make_full_execution_joint_step(
        solver.current_q
    )
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[0] = np.deg2rad(3.0)
    candidate[6] = -np.deg2rad(4.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
    )

    assert np.isclose(qpos[0], np.deg2rad(0.25))
    assert np.isclose(qpos[6], -np.deg2rad(0.25))
    assert metadata["step_limited"] is True
    assert metadata["execution_step_source"] == "explicit_execution_max_joint_step"


def test_dual_arm_activity_hold_allows_unmoved_side_to_reduce_existing_error() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, loose_dynamic_worsening=True)
    reference = np.zeros(12, dtype=float)
    reference[0] = -0.050
    raw = reference.copy()
    raw[0] = -0.020
    raw[6] = 0.030
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)
    targets = retargeter.current_world_targets()
    targets["left"] = np.eye(4, dtype=float)
    retargeter._last_target_poses = {
        "left": targets["left"].copy(),
        "right": targets["right"].copy(),
    }
    targets["right"] = targets["right"].copy()
    targets["right"][0, 3] = 0.060

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert qpos[0] > reference[0]
    selected = metadata["selected_metadata"]
    left_hold = selected["inactive_side_hold"]["sides"]["left"]
    assert left_hold["hold_weight"] == 1.0
    assert left_hold["unexplained_position_m"] == 0.0
    assert selected["reject_reason"] is None


def test_dual_arm_rm65_posture_escape_allows_right_j23_bad_to_good_branch() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True, rm65_local_escape=False)
    reference = np.zeros(12, dtype=float)
    reference[1] = np.deg2rad(20.0)
    reference[2] = np.deg2rad(120.0)
    reference[7] = np.deg2rad(100.0)
    reference[8] = np.deg2rad(-90.0)
    raw = reference.copy()
    raw[7] = np.deg2rad(20.0)
    raw[8] = np.deg2rad(130.0)
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert qpos[8] > reference[8]
    selected = metadata["selected_metadata"]
    assert selected["right_hard_branch_escape_allowed"] is True
    assert selected["posture_preference"]["right"]["candidate"]["score_deg"] < selected["posture_preference"]["right"]["reference"]["score_deg"]


def test_dual_arm_rm65_wrist_unload_candidate_is_collision_checked() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    reference = np.zeros(12, dtype=float)
    reference[3] = np.deg2rad(177.0)
    reference[4] = np.deg2rad(88.0)
    reference[5] = np.deg2rad(-170.0)
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    targets["left"] = targets["left"].copy()
    targets["left"][0, 3] = -0.5

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=reference.copy(),
        reference_full_q=reference.copy(),
        current_targets=targets,
        profile=None,
    )

    assert name.startswith("wrist_unload_left_")
    assert not np.allclose(qpos, reference)
    assert metadata["wrist_unload"]["status"] == "active"
    assert metadata["selected_source"] == "wrist_unload"


def test_dual_arm_prefers_tracking_candidate_over_wrist_unload_auxiliary_recovery() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    reference = np.zeros(12, dtype=float)
    reference[3] = np.deg2rad(177.0)
    reference[4] = np.deg2rad(88.0)
    reference[5] = np.deg2rad(-170.0)
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=reference.copy(),
        reference_full_q=reference.copy(),
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert np.allclose(qpos, reference)
    assert metadata["selected_source"] == "multitask"
    assert metadata["candidate_eval_mode"] == "base_only"
    assert metadata["wrist_unload"]["status"] == "deferred"


def test_dual_arm_jump_gate_warns_but_does_not_hold_for_wrist_jump() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[3] = np.deg2rad(55.0)
    targets = retargeter.current_world_targets()
    retargeter._last_target_poses = {"left": targets["left"].copy(), "right": targets["right"].copy()}

    gate = retargeter._evaluate_jump_gate(
        reference_full_q=reference,
        candidate_full_q=candidate,
        current_targets=targets,
    )

    assert gate["hold"] is False
    assert gate["status"] == "warning"
    assert gate["reason"] == "small_pose_large_q_jump_warning_only:left"


def test_final_qpos_pair_collision_check_is_read_only_and_fail_closed() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    initial_q = solver.current_q.copy()

    safe = retargeter.check_qpos_pair_collision(
        left_qpos=np.zeros(6),
        right_qpos=np.zeros(6),
    )
    assert safe.safe
    assert safe.collision_pair_count == len(retargeter.profile.collision_pairs)
    assert safe.min_distance_m == 0.25
    np.testing.assert_allclose(solver.current_q, initial_q)

    solver.min_collision_distance = -0.01
    unsafe = retargeter.check_qpos_pair_collision(
        left_qpos=np.zeros(6),
        right_qpos=np.zeros(6),
    )
    assert not unsafe.safe
    assert unsafe.reason == "collision_margin_low"
    np.testing.assert_allclose(solver.current_q, initial_q)


def test_admitted_qref_fk_is_read_only_and_uses_both_final_arm_vectors() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    initial_q = solver.current_q.copy()
    left_qpos = np.zeros(6, dtype=float)
    right_qpos = np.zeros(6, dtype=float)
    left_qpos[0] = 0.2
    right_qpos[0] = 0.3

    poses = retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=left_qpos,
        right_qpos=right_qpos,
    )

    assert poses["left"][0, 3] == pytest.approx(-0.2)
    assert poses["right"][0, 3] == pytest.approx(0.3)
    np.testing.assert_allclose(solver.current_q, initial_q)


def test_active_limit_band_blocks_inward_candidate_before_latch_arms(
    monkeypatch,
) -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(
        solver,
        joint_limit_guard_mode="forced_escape_latched",
    )
    reference = np.zeros(12, dtype=float)
    reference[8] = 1.0 - np.deg2rad(1.0)
    ordinary = reference.copy()
    ordinary[8] += 0.005
    escape = reference.copy()
    escape[8] -= 0.005

    ordinary_candidate = target_solver_module._DualArmCandidate(
        name="ordinary_inward",
        qpos=ordinary,
        source="multitask",
        method="ordinary_inward",
    )
    escape_metadata = {
        "local_step_limit_escape_candidate": True,
        "local_step_limit_escape_side": "right",
        "local_step_limit_escape_joint_index": 2,
        "local_step_limit_escape_full_joint_index": 8,
        "local_step_limit_escape_direction": -1,
        "local_step_limit_escape_margin_progress_deg": 1.0,
    }
    escape_candidate = target_solver_module._DualArmCandidate(
        name="joint_limit_escape_right_j3_neg",
        qpos=escape,
        source="joint_limit_escape",
        method="joint_limit_escape_right_j3_neg",
        metadata=escape_metadata,
    )
    calls = []

    def fake_candidates(**kwargs):
        calls.append(dict(kwargs))
        if bool(kwargs.get("include_base", True)):
            return [ordinary_candidate], {"enabled": True, "status": "active"}
        return [escape_candidate], {"enabled": True, "status": "active"}

    def fake_evaluate(*, candidate, step_scale, **_kwargs):
        return True, candidate.qpos.copy(), {
            "candidate_metadata": dict(candidate.metadata),
            "score": 0.0,
            "step_scale": float(step_scale),
        }

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", fake_candidates)
    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", fake_evaluate)

    assert retargeter._joint_limit_escape_direction_latch == {}
    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        # Keep the raw result neutral so this test exercises the second,
        # canonical-branch trigger after the ordinary candidate is evaluated.
        raw_full_q=reference,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "joint_limit_escape_right_j3_neg"
    assert metadata["selected_source"] == "joint_limit_escape"
    assert metadata["joint_limit_escape_hysteresis"]["status"] == (
        "pending_single_escape"
    )
    assert qpos[8] < reference[8]
    assert len(calls) == 2
    assert calls[1]["include_base"] is False
    assert calls[1]["include_posture_escape"] is False
    assert calls[1]["include_wrist_unload"] is False
    assert calls[1]["include_singularity_escape"] is False
    assert calls[1]["limit_escape_max_candidates"] == 1


def test_near_limit_outward_tracking_does_not_launch_escape_search(monkeypatch) -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(
        solver,
        joint_limit_guard_mode="forced_escape_latched",
    )
    reference = np.zeros(12, dtype=float)
    reference[8] = 1.0 - np.deg2rad(1.0)
    ordinary = reference.copy()
    ordinary[8] -= 0.005
    escape = reference.copy()
    escape[8] -= 0.01

    ordinary_candidate = target_solver_module._DualArmCandidate(
        name="ordinary_outward",
        qpos=ordinary,
        source="multitask",
        method="ordinary_outward",
    )
    escape_candidate = target_solver_module._DualArmCandidate(
        name="joint_limit_escape_right_j3_neg",
        qpos=escape,
        source="joint_limit_escape",
        method="joint_limit_escape_right_j3_neg",
        metadata={
            "local_step_limit_escape_candidate": True,
            "local_step_limit_escape_side": "right",
            "local_step_limit_escape_joint_index": 2,
            "local_step_limit_escape_full_joint_index": 8,
            "local_step_limit_escape_direction": -1,
            "local_step_limit_escape_margin_progress_deg": 1.0,
        },
    )
    calls = []

    def fake_candidates(**kwargs):
        calls.append(dict(kwargs))
        if bool(kwargs.get("include_base", True)):
            return [ordinary_candidate], {"enabled": True, "status": "active"}
        return [escape_candidate], {"enabled": True, "status": "active"}

    def fake_evaluate(*, candidate, step_scale, **_kwargs):
        return True, candidate.qpos.copy(), {
            "candidate_metadata": dict(candidate.metadata),
            "score": 0.0,
            "step_scale": float(step_scale),
        }

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", fake_candidates)
    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", fake_evaluate)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=ordinary,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "ordinary_outward"
    np.testing.assert_allclose(qpos, ordinary)
    assert metadata["joint_limit_escape_hysteresis"]["status"] == "normal_tracking"
    assert len(calls) == 1


def test_default_limit_guard_preserves_stable_legacy_candidate_path(
    monkeypatch,
) -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    reference[8] = 1.0 - np.deg2rad(1.0)
    ordinary = reference.copy()
    ordinary[8] += 0.005
    calls = []
    candidate = target_solver_module._DualArmCandidate(
        name="ordinary_inward",
        qpos=ordinary,
        source="multitask",
        method="ordinary_inward",
    )

    def fake_candidates(**kwargs):
        calls.append(dict(kwargs))
        return [candidate], {"enabled": True, "status": "active"}

    def fake_evaluate(*, candidate, step_scale, **_kwargs):
        return True, candidate.qpos.copy(), {
            "candidate_metadata": dict(candidate.metadata),
            "score": 0.0,
            "step_scale": float(step_scale),
        }

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", fake_candidates)
    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", fake_evaluate)

    qpos, name, _score, _distance, metadata = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=ordinary,
            reference_full_q=reference,
            current_targets=retargeter.current_world_targets(),
            profile=None,
        )
    )

    assert name == "ordinary_inward"
    np.testing.assert_allclose(qpos, ordinary)
    assert metadata["joint_limit_guard"]["mode"] == "legacy"
    assert len(calls) == 1


def test_boundary_hold_projects_only_inward_near_limit_joint(
    monkeypatch,
) -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(
        solver,
        joint_limit_guard_mode="boundary_hold",
        joint_limit_boundary_guard_margin_deg=1.5,
    )
    reference = np.zeros(12, dtype=float)
    reference[8] = 1.0 - np.deg2rad(1.0)
    ordinary = reference.copy()
    ordinary[7] += 0.002
    ordinary[8] += 0.005
    candidate = target_solver_module._DualArmCandidate(
        name="ordinary_inward",
        qpos=ordinary,
        source="multitask",
        method="ordinary_inward",
    )
    calls = []

    def fake_candidates(**kwargs):
        calls.append(dict(kwargs))
        return [candidate], {"enabled": True, "status": "active"}

    def fake_evaluate(*, candidate, step_scale, **_kwargs):
        return True, candidate.qpos.copy(), {
            "candidate_metadata": dict(candidate.metadata),
            "score": 0.0,
            "step_scale": float(step_scale),
        }

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", fake_candidates)
    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", fake_evaluate)

    qpos, name, _score, _distance, metadata = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=ordinary,
            reference_full_q=reference,
            current_targets=retargeter.current_world_targets(),
            profile=None,
        )
    )

    assert name == "joint_limit_boundary_hold"
    assert metadata["selected_source"] == "joint_limit_boundary_hold"
    assert metadata["candidate_eval_mode"] == "joint_limit_boundary_hold"
    assert qpos[8] == reference[8]
    assert qpos[7] == ordinary[7]
    assert len(calls) == 1

    repeated_qpos, repeated_name, *_rest = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=ordinary,
            reference_full_q=reference,
            current_targets=retargeter.current_world_targets(),
            profile=None,
        )
    )
    assert repeated_name == "joint_limit_boundary_hold"
    np.testing.assert_allclose(repeated_qpos, qpos)
    assert retargeter._joint_limit_escape_direction_latch == {}


def test_boundary_hold_releases_immediately_for_outward_candidate(
    monkeypatch,
) -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(
        solver,
        joint_limit_guard_mode="boundary_hold",
    )
    reference = np.zeros(12, dtype=float)
    reference[8] = 1.0 - np.deg2rad(1.0)
    ordinary = reference.copy()
    ordinary[8] -= 0.005
    candidate = target_solver_module._DualArmCandidate(
        name="ordinary_outward",
        qpos=ordinary,
        source="multitask",
        method="ordinary_outward",
    )

    monkeypatch.setattr(
        retargeter,
        "_dual_arm_tracking_candidates",
        lambda **_kwargs: (
            [candidate],
            {"enabled": True, "status": "active"},
        ),
    )
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, step_scale, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {
                "candidate_metadata": dict(candidate.metadata),
                "score": 0.0,
                "step_scale": float(step_scale),
            },
        ),
    )

    qpos, name, _score, _distance, metadata = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=ordinary,
            reference_full_q=reference,
            current_targets=retargeter.current_world_targets(),
            profile=None,
        )
    )

    assert name == "ordinary_outward"
    np.testing.assert_allclose(qpos, ordinary)
    assert metadata["joint_limit_guard"]["mode"] == "boundary_hold"


def test_boundary_hold_removes_legacy_escape_from_auxiliary_candidate_pool(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(
        solver,
        joint_limit_guard_mode="boundary_hold",
    )
    retargeter.ik_safety_profile = replace(
        retargeter.ik_safety_profile,
        local_step_singularity_escape_enabled=True,
    )
    reference = np.zeros(12, dtype=float)
    ordinary = reference.copy()
    ordinary[0] += 0.002
    candidate = target_solver_module._DualArmCandidate(
        name="ordinary",
        qpos=ordinary,
        source="multitask",
        method="ordinary",
    )
    calls: list[dict[str, object]] = []

    def fake_candidates(**kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("include_base", True):
            return [candidate], {"enabled": True, "status": "base"}
        return [], {"enabled": True, "status": "auxiliary"}

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", fake_candidates)
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, step_scale, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {
                "candidate_metadata": dict(candidate.metadata),
                "score": 0.0,
                "step_scale": float(step_scale),
            },
        ),
    )
    monkeypatch.setattr(
        retargeter,
        "_local_proxy_singularity_metadata",
        lambda _qpos: {"any_singular": True},
    )

    qpos, name, *_rest = retargeter._select_safe_tracking_candidate(
        raw_full_q=ordinary,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "ordinary"
    np.testing.assert_allclose(qpos, ordinary)
    assert len(calls) == 2
    assert calls[0]["include_limit_escape"] is False
    assert calls[1]["include_limit_escape"] is False


def test_limit_escape_direction_latch_blocks_immediate_inward_ik_reselection(
    monkeypatch,
) -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(
        solver,
        joint_limit_guard_mode="forced_escape_latched",
    )
    reference = np.zeros(12, dtype=float)
    reference[8] = 1.0 - np.deg2rad(3.0)
    ordinary = reference.copy()
    ordinary[8] += 0.01
    escape = reference.copy()
    escape[8] -= 0.01
    retargeter._joint_limit_escape_direction_latch[("right", 2)] = -1

    ordinary_candidate = target_solver_module._DualArmCandidate(
        name="ordinary_inward",
        qpos=ordinary,
        source="multitask",
        method="ordinary_inward",
    )
    escape_metadata = {
        "local_step_limit_escape_candidate": True,
        "local_step_limit_escape_side": "right",
        "local_step_limit_escape_joint_index": 2,
        "local_step_limit_escape_full_joint_index": 8,
        "local_step_limit_escape_direction": -1,
        "local_step_limit_escape_margin_progress_deg": 1.0,
    }
    escape_candidate = target_solver_module._DualArmCandidate(
        name="joint_limit_escape_right_j3_neg",
        qpos=escape,
        source="joint_limit_escape",
        method="joint_limit_escape_right_j3_neg",
        metadata=escape_metadata,
    )
    calls = []

    def fake_candidates(**kwargs):
        calls.append(dict(kwargs))
        if bool(kwargs.get("include_base", True)):
            return [ordinary_candidate], {"enabled": True, "status": "active"}
        return [escape_candidate], {"enabled": True, "status": "active"}

    def fake_evaluate(*, candidate, step_scale, **_kwargs):
        return True, candidate.qpos.copy(), {
            "candidate_metadata": dict(candidate.metadata),
            "score": 0.0,
            "step_scale": float(step_scale),
        }

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", fake_candidates)
    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", fake_evaluate)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=ordinary,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "joint_limit_escape_right_j3_neg"
    assert metadata["selected_source"] == "joint_limit_escape"
    assert metadata["joint_limit_escape_hysteresis"]["status"] == "latched_escape_only"
    assert qpos[8] < reference[8]
    assert len(calls) == 1
    assert calls[0]["include_base"] is False
    assert calls[0]["include_posture_escape"] is False
    assert calls[0]["include_wrist_unload"] is False
    assert calls[0]["include_singularity_escape"] is False
    assert calls[0]["limit_escape_max_candidates"] == 1


def test_limit_escape_direction_latch_releases_after_extra_margin() -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(
        solver,
        joint_limit_guard_mode="forced_escape_latched",
    )
    reference = np.zeros(12, dtype=float)
    reference[8] = 1.0 - np.deg2rad(3.0)
    admitted = reference.copy()
    admitted[8] -= 0.01
    selection_metadata = {
        "selected_source": "joint_limit_escape",
        "selected_metadata": {
            "candidate_metadata": {
                "local_step_limit_escape_side": "right",
                "local_step_limit_escape_joint_index": 2,
                "local_step_limit_escape_full_joint_index": 8,
                "local_step_limit_escape_direction": -1,
                "local_step_limit_escape_margin_progress_deg": 1.0,
            }
        },
    }

    armed = retargeter._update_joint_limit_escape_direction_latch(
        reference_full_q=reference,
        admitted_full_q=admitted,
        status="tracking",
        selection_metadata=selection_metadata,
    )
    assert armed["armed"]["joint"] == "right:j3"
    assert armed["active"] == {"right:j3": -1}

    released_q = admitted.copy()
    released_q[8] = 1.0 - np.deg2rad(5.0)
    raw_inward = admitted.copy()
    raw_inward[8] += 0.02
    guarded = retargeter._update_joint_limit_escape_direction_latch(
        reference_full_q=admitted,
        admitted_full_q=released_q,
        raw_full_q=raw_inward,
        status="tracking",
        selection_metadata=selection_metadata,
    )
    assert guarded["released"] == []
    assert guarded["active"] == {"right:j3": -1}
    assert retargeter._joint_limit_escape_release_guard == {("right", 2): -1}

    raw_outward = released_q.copy()
    raw_outward[8] -= 0.02
    released = retargeter._update_joint_limit_escape_direction_latch(
        reference_full_q=released_q,
        admitted_full_q=released_q,
        raw_full_q=raw_outward,
        status="tracking",
        selection_metadata=None,
    )
    assert released["released"] == ["right:j3"]
    assert released["active"] == {}
