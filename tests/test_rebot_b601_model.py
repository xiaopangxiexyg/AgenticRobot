from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.rebot_b601_rs import (
    B601ModelError,
    B601PinocchioModel,
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_END_EFFECTOR_FRAME,
    B601_RS_JACOBIAN_REFERENCE,
    B601_RS_TWIST_ORDER,
    B601_RS_URDF_RELATIVE_PATH,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
pytestmark = [pytest.mark.unit, pytest.mark.contract]

SYNTHETIC_B601_URDF = """\
<?xml version="1.0"?>
<robot name="synthetic-b601-contract">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="link3"/>
  <link name="link4"/>
  <link name="link5"/>
  <link name="link6"/>
  <link name="gripper_end"/>
  <link name="gripper_left"/>
  <link name="gripper_right"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" effort="1" velocity="1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0.1 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/><child link="link3"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="joint4" type="revolute">
    <parent link="link3"/><child link="link4"/>
    <origin xyz="0.15 0.02 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
  </joint>
  <joint name="joint5" type="revolute">
    <parent link="link4"/><child link="link5"/>
    <origin xyz="0.08 0 0.04" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
  </joint>
  <joint name="joint6" type="revolute">
    <parent link="link5"/><child link="link6"/>
    <origin xyz="0.05 0 0.05" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="j_gripper_end" type="fixed">
    <parent link="link6"/><child link="gripper_end"/>
    <origin xyz="0.08 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="joint_left" type="prismatic">
    <parent link="gripper_end"/><child link="gripper_left"/>
    <origin xyz="0 0.02 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="0.05" effort="1" velocity="1"/>
  </joint>
  <joint name="joint_right" type="prismatic">
    <parent link="gripper_end"/><child link="gripper_right"/>
    <origin xyz="0 -0.02 0" rpy="0 0 0"/><axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.0715" effort="1" velocity="1"/>
  </joint>
</robot>
"""


@pytest.fixture(scope="module")
def b601_model() -> B601PinocchioModel:
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip(
            "initialize hardware/reBotArm_control_py at its recorded gitlink "
            "to exercise the B601-RS model"
        )
    return B601PinocchioModel.from_rebot_source_root(REBOT_SOURCE_ROOT)


@pytest.fixture(scope="module")
def synthetic_b601_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> B601PinocchioModel:
    urdf_path = tmp_path_factory.mktemp("b601-model") / "synthetic-b601.urdf"
    urdf_path.write_text(SYNTHETIC_B601_URDF, encoding="utf-8")
    return B601PinocchioModel(urdf_path)


def _finite_difference_world_aligned_twist(
    model: B601PinocchioModel,
    q: np.ndarray,
    velocity: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    center = model.forward_kinematics(q)
    plus = model.forward_kinematics(q + epsilon * velocity)
    minus = model.forward_kinematics(q - epsilon * velocity)
    linear_velocity = (plus[:3, 3] - minus[:3, 3]) / (2.0 * epsilon)
    angular_plus = Rotation.from_matrix(
        plus[:3, :3] @ center[:3, :3].T
    ).as_rotvec()
    angular_minus = Rotation.from_matrix(
        minus[:3, :3] @ center[:3, :3].T
    ).as_rotvec()
    angular_velocity = (angular_plus - angular_minus) / (2.0 * epsilon)
    return np.concatenate((linear_velocity, angular_velocity))


def test_package_import_does_not_import_vendor_runtime() -> None:
    forbidden_prefixes = (
        "agent_gateway",
        "control_composition",
        "control_station",
        "hardware",
        "pinocchio",
        "reBotArm_control_py",
        "sim",
        "teleop_runtime",
    )
    script = (
        "import sys; import robot_motion.rebot_b601_rs; "
        f"prefixes = {forbidden_prefixes!r}; "
        "assert not any(name == prefix or name.startswith(prefix + '.') "
        "for name in sys.modules for prefix in prefixes)"
    )

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_robot_motion_root_does_not_load_b601_model() -> None:
    script = (
        "import sys; import robot_motion; "
        "assert 'robot_motion.rebot_b601_rs' not in sys.modules; "
        "assert 'pinocchio' not in sys.modules"
    )

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_root_resolver_rejects_urdf_symlink_escape(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    urdf_parent = source_root / B601_RS_URDF_RELATIVE_PATH.parent
    urdf_parent.mkdir(parents=True)
    outside = tmp_path / "outside.urdf"
    outside.write_text("<robot name='outside'/>", encoding="utf-8")
    (source_root / B601_RS_URDF_RELATIVE_PATH).symlink_to(outside)

    with pytest.raises(B601ModelError, match="outside"):
        B601PinocchioModel.from_rebot_source_root(source_root)


def test_model_selects_six_arm_joints_from_eight_dof_urdf(
    b601_model: B601PinocchioModel,
) -> None:
    assert b601_model.arm_joint_names == B601_RS_ARM_JOINT_NAMES
    assert b601_model.arm_q_indices == (0, 1, 2, 3, 4, 5)
    assert b601_model.arm_v_indices == (0, 1, 2, 3, 4, 5)
    assert b601_model.arm_dof == 6
    assert b601_model.full_model_nq == 8
    assert b601_model.full_model_nv == 8
    assert b601_model.full_model_joint_names[-2:] == (
        "joint_left",
        "joint_right",
    )
    assert b601_model.end_effector_frame == B601_RS_END_EFFECTOR_FRAME


def test_arm_position_limits_follow_joint_coordinate_indices(
    b601_model: B601PinocchioModel,
) -> None:
    lower, upper = b601_model.arm_position_limits()

    np.testing.assert_allclose(lower, [-2.8, 0.0, 0.0, -1.57, -1.57, -3.14])
    np.testing.assert_allclose(upper, [2.8, 3.14, 3.14, 1.57, 1.57, 3.14])

    lower[0] = 999.0
    lower_again, _ = b601_model.arm_position_limits()
    assert lower_again[0] == pytest.approx(-2.8)


def test_zero_configuration_fk_is_a_valid_base_to_handle_pose(
    b601_model: B601PinocchioModel,
) -> None:
    pose = b601_model.forward_kinematics(np.zeros(6))

    assert pose.shape == (4, 4)
    assert np.all(np.isfinite(pose))
    np.testing.assert_allclose(
        pose[:3, 3],
        [0.30171017, 2.609017957e-07, 0.2177467069],
        atol=1e-9,
    )
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(pose[3], [0.0, 0.0, 0.0, 1.0])


def test_world_aligned_jacobian_matches_se3_finite_difference(
    b601_model: B601PinocchioModel,
) -> None:
    q = np.array([0.2, 1.0, 1.2, 0.25, -0.35, 0.4], dtype=float)
    velocity = np.array([0.4, -0.2, 0.3, -0.5, 0.25, 0.1], dtype=float)
    velocity /= np.linalg.norm(velocity)

    jacobian = b601_model.arm_jacobian(q)
    numerical_twist = _finite_difference_world_aligned_twist(
        b601_model,
        q,
        velocity,
    )

    assert B601_RS_JACOBIAN_REFERENCE == "LOCAL_WORLD_ALIGNED"
    assert B601_RS_TWIST_ORDER == ("vx", "vy", "vz", "wx", "wy", "wz")
    assert jacobian.shape == (6, 6)
    assert np.all(np.isfinite(jacobian))
    np.testing.assert_allclose(
        jacobian @ velocity,
        numerical_twist,
        atol=5e-7,
        rtol=5e-6,
    )


def test_jacobian_invariant_over_deterministic_interior_sweep(
    b601_model: B601PinocchioModel,
) -> None:
    lower, upper = b601_model.arm_position_limits()
    rng = np.random.default_rng(601)

    for _ in range(32):
        fraction = 0.15 + 0.70 * rng.random(6)
        q = lower + fraction * (upper - lower)
        velocity = rng.normal(size=6)
        velocity /= np.linalg.norm(velocity)
        expected = _finite_difference_world_aligned_twist(
            b601_model,
            q,
            velocity,
        )

        np.testing.assert_allclose(
            b601_model.arm_jacobian(q) @ velocity,
            expected,
            atol=5e-7,
            rtol=5e-6,
        )


def test_synthetic_contract_exercises_fk_and_jacobian_without_vendor_checkout(
    synthetic_b601_model: B601PinocchioModel,
) -> None:
    q = np.array([0.1, 0.9, 1.1, -0.2, 0.3, -0.4], dtype=float)
    velocity = np.array([0.3, -0.4, 0.1, 0.2, -0.5, 0.6], dtype=float)
    velocity /= np.linalg.norm(velocity)

    pose = synthetic_b601_model.forward_kinematics(q)
    jacobian = synthetic_b601_model.arm_jacobian(q)

    assert pose.shape == (4, 4)
    assert np.all(np.isfinite(pose))
    np.testing.assert_allclose(
        jacobian @ velocity,
        _finite_difference_world_aligned_twist(
            synthetic_b601_model,
            q,
            velocity,
        ),
        atol=5e-7,
        rtol=5e-6,
    )


def test_singularity_metrics_are_the_jacobian_svd(
    synthetic_b601_model: B601PinocchioModel,
) -> None:
    q = np.array([0.2, 1.0, 1.2, 0.25, -0.35, 0.4], dtype=float)
    singular_values = np.linalg.svd(
        synthetic_b601_model.arm_jacobian(q), compute_uv=False
    )
    metrics = synthetic_b601_model.singularity_metrics(q)

    np.testing.assert_allclose(metrics.singular_values, singular_values)
    assert metrics.sigma_max == pytest.approx(singular_values[0])
    assert metrics.sigma_min == pytest.approx(singular_values[-1])
    assert metrics.rank == np.count_nonzero(
        singular_values > metrics.rank_tolerance
    )
    expected_condition = singular_values[0] / singular_values[-1]
    assert metrics.condition_number == pytest.approx(expected_condition)
    assert metrics.characteristic_length_m is None


def test_singularity_metrics_require_explicit_valid_length_scaling(
    synthetic_b601_model: B601PinocchioModel,
) -> None:
    q = np.array([0.2, 1.0, 1.2, 0.25, -0.35, 0.4], dtype=float)
    characteristic_length_m = 0.3
    scaled_jacobian = synthetic_b601_model.arm_jacobian(q)
    scaled_jacobian[:3, :] /= characteristic_length_m
    expected = np.linalg.svd(scaled_jacobian, compute_uv=False)

    metrics = synthetic_b601_model.singularity_metrics(
        q,
        characteristic_length_m=characteristic_length_m,
    )

    np.testing.assert_allclose(metrics.singular_values, expected)
    assert metrics.characteristic_length_m == pytest.approx(
        characteristic_length_m
    )

    for invalid in (0.0, -0.1, np.inf, np.nan, "invalid"):
        with pytest.raises(ValueError, match="positive finite"):
            synthetic_b601_model.singularity_metrics(
                q,
                characteristic_length_m=invalid,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    "q_arm, expected_message",
    [
        ([0.0] * 5, r"shape \(6,\)"),
        (np.zeros((2, 3)), r"shape \(6,\)"),
        ([0.0, 0.0, 0.0, 0.0, 0.0, np.nan], "non-finite"),
    ],
)
def test_model_rejects_malformed_arm_configurations(
    synthetic_b601_model: B601PinocchioModel,
    q_arm: object,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        synthetic_b601_model.forward_kinematics(q_arm)  # type: ignore[arg-type]


def test_model_rejects_unknown_joint_or_frame(
    synthetic_b601_model: B601PinocchioModel,
) -> None:
    with pytest.raises(B601ModelError, match="arm joint is absent"):
        B601PinocchioModel(
            synthetic_b601_model.urdf_path,
            arm_joint_names=(*B601_RS_ARM_JOINT_NAMES[:-1], "unknown_joint"),
        )
    with pytest.raises(B601ModelError, match="end-effector frame is absent"):
        B601PinocchioModel(
            synthetic_b601_model.urdf_path,
            end_effector_frame="unknown_frame",
        )
