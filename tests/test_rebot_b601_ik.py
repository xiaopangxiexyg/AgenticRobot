from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import (
    B601_RS_URDF_RELATIVE_PATH,
    B601DampedLeastSquaresIK,
    B601IKConfig,
    B601PinocchioModel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.fixture(scope="module")
def b601_model() -> B601PinocchioModel:
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")
    return B601PinocchioModel.from_rebot_source_root(REBOT_SOURCE_ROOT)


def test_reachable_pose_converges_from_an_explicit_nearby_seed(
    b601_model: B601PinocchioModel,
) -> None:
    target_qpos = np.array([0.2, 1.1, 1.25, 0.2, -0.3, 0.4])
    seed_qpos = target_qpos + np.array([-0.03, 0.02, -0.02, 0.01, 0.02, -0.03])

    result = B601DampedLeastSquaresIK(b601_model).solve(
        b601_model.forward_kinematics(target_qpos),
        seed_qpos=seed_qpos,
    )

    assert result.success is True
    assert result.status == "converged"
    assert result.position_error_m <= 1e-5
    assert result.orientation_error_rad <= 1e-4
    np.testing.assert_allclose(result.qpos, target_qpos, atol=5e-5)
    assert result.seed_distance_rad < 0.1
    assert result.sigma_min >= 0.0
    assert result.condition_number > 0.0


def test_backtracking_converges_for_a_previously_oscillating_nearby_seed(
    b601_model: B601PinocchioModel,
) -> None:
    target_qpos = np.array(
        [-1.8900744, 0.38894494, 1.18435438, -1.20818622, -0.84029475, 1.131321]
    )
    seed_qpos = np.array(
        [-1.75552907, 0.52726457, 1.29309203, -1.18776915, -0.73224474, 1.13228526]
    )

    result = B601DampedLeastSquaresIK(b601_model).solve(
        b601_model.forward_kinematics(target_qpos),
        seed_qpos=seed_qpos,
    )

    assert result.success is True
    assert result.status == "converged"
    assert result.position_error_m <= 1e-5
    assert result.orientation_error_rad <= 1e-4


def test_previous_solution_seed_preserves_continuity_over_pose_stream(
    b601_model: B601PinocchioModel,
) -> None:
    start = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    finish = np.array([0.2, 1.2, 1.3, 0.2, -0.25, 0.35])
    solver = B601DampedLeastSquaresIK(b601_model)
    previous = start.copy()

    for alpha in np.linspace(0.05, 1.0, 20):
        target_qpos = start + float(alpha) * (finish - start)
        result = solver.solve(
            b601_model.forward_kinematics(target_qpos),
            seed_qpos=previous,
        )

        assert result.success is True
        assert result.seed_distance_rad < 0.04
        np.testing.assert_allclose(result.qpos, target_qpos, atol=2e-5)
        previous = np.asarray(result.qpos, dtype=float)


def test_failed_unreachable_solve_returns_bounded_diagnostic_not_success(
    b601_model: B601PinocchioModel,
) -> None:
    seed = np.array([0.1, 1.0, 1.1, 0.1, -0.1, 0.2])
    target_pose = b601_model.forward_kinematics(seed)
    target_pose[:3, 3] += np.array([3.0, 3.0, 3.0])
    solver = B601DampedLeastSquaresIK(
        b601_model,
        B601IKConfig(max_iterations=30),
    )

    result = solver.solve(target_pose, seed_qpos=seed)
    lower, upper = b601_model.arm_position_limits()

    assert result.success is False
    assert result.status == "line_search_stalled"
    assert result.position_error_m > 1.0
    assert np.all(np.asarray(result.qpos) >= lower)
    assert np.all(np.asarray(result.qpos) <= upper)


@pytest.mark.parametrize(
    "target_mutation, expected",
    [
        (lambda pose: pose[:3, :3].__setitem__((0, 0), 2.0), "invalid target_pose"),
        (lambda pose: pose.__setitem__((3, 3), float("nan")), "invalid target_pose"),
    ],
)
def test_invalid_pose_is_rejected_before_iteration(
    b601_model: B601PinocchioModel,
    target_mutation,
    expected: str,
) -> None:
    pose = b601_model.forward_kinematics(np.zeros(6))
    target_mutation(pose)

    with pytest.raises(ValueError, match=expected):
        B601DampedLeastSquaresIK(b601_model).solve(
            pose,
            seed_qpos=np.zeros(6),
        )


def test_seed_shape_finiteness_and_physical_limits_are_fail_closed(
    b601_model: B601PinocchioModel,
) -> None:
    pose = b601_model.forward_kinematics(np.zeros(6))
    solver = B601DampedLeastSquaresIK(b601_model)

    with pytest.raises(ValueError, match="shape"):
        solver.solve(pose, seed_qpos=np.zeros(5))
    with pytest.raises(ValueError, match="non-finite"):
        solver.solve(pose, seed_qpos=[0.0, 1.0, 1.0, 0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="exceeds"):
        solver.solve(pose, seed_qpos=[3.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="positive integer"):
        B601IKConfig(max_iterations=1.5)
    with pytest.raises(ValueError, match="positive integer"):
        B601IKConfig(backtracking_steps=0)
