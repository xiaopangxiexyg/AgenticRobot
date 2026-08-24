from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import (
    B601_RS_URDF_RELATIVE_PATH,
    B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
    B601PinocchioModel,
    B601SpatialWrench,
    B601TranslationalVirtualSpring,
    B601VirtualWall,
    map_world_aligned_ee_wrench_to_joint_torque,
    transform_spatial_twist,
    transform_spatial_wrench,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.fixture(scope="module")
def b601_model() -> B601PinocchioModel:
    pytest.importorskip("pinocchio")
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")
    return B601PinocchioModel.from_rebot_source_root(REBOT_SOURCE_ROOT)


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def test_wrench_transform_rotates_force_and_shifts_torque_origin() -> None:
    target_T_source = np.eye(4)
    target_T_source[:3, :3] = _rotation_z(np.pi / 2.0)
    target_T_source[:3, 3] = (1.0, 0.0, 0.0)
    source = B601SpatialWrench(
        force_n=(1.0, 0.0, 0.0),
        torque_nm=(0.0, 0.0, 2.0),
        frame_id="source",
    )

    target = transform_spatial_wrench(
        source,
        target_frame_id="target",
        target_T_source=target_T_source,
    )

    np.testing.assert_allclose(target.force_n, (0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(target.torque_nm, (0.0, 0.0, 3.0), atol=1e-12)
    assert target.frame_id == "target"


def test_wrench_and_twist_transforms_preserve_power_and_round_trip() -> None:
    target_T_source = np.eye(4)
    target_T_source[:3, :3] = _rotation_z(0.63)
    target_T_source[:3, 3] = (0.2, -0.1, 0.35)
    source_wrench = B601SpatialWrench(
        force_n=(2.0, -3.0, 1.5),
        torque_nm=(0.4, 0.2, -0.8),
        frame_id="source",
    )
    source_twist = np.asarray((0.4, -0.2, 0.1, 0.3, -0.5, 0.2))

    target_wrench = transform_spatial_wrench(
        source_wrench,
        target_frame_id="target",
        target_T_source=target_T_source,
    )
    target_twist = transform_spatial_twist(
        source_twist,
        target_T_source=target_T_source,
    )
    source_power = float(source_wrench.as_vector() @ source_twist)
    target_power = float(target_wrench.as_vector() @ target_twist)

    recovered = transform_spatial_wrench(
        target_wrench,
        target_frame_id="source",
        target_T_source=np.linalg.inv(target_T_source),
    )
    assert target_power == pytest.approx(source_power, abs=1e-12)
    np.testing.assert_allclose(recovered.as_vector(), source_wrench.as_vector())


@pytest.mark.parametrize(
    "transform, message",
    [
        (np.eye(3), "4-by-4"),
        (np.diag([1.0, 1.0, 1.0, 2.0]), "homogeneous"),
        (np.diag([-1.0, 1.0, 1.0, 1.0]), "right handed"),
        (
            np.asarray(
                [
                    [1.0, 0.1, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "orthonormal",
        ),
    ],
)
def test_spatial_transform_rejects_non_se3_inputs(
    transform: np.ndarray,
    message: str,
) -> None:
    wrench = B601SpatialWrench((0.0,) * 3, (0.0,) * 3, "source")
    with pytest.raises(ValueError, match=message):
        transform_spatial_wrench(
            wrench,
            target_frame_id="target",
            target_T_source=transform,
        )


def test_jacobian_transpose_mapping_preserves_instantaneous_power(
    b601_model: B601PinocchioModel,
) -> None:
    qpos = (0.2, 1.0, 1.1, -0.2, 0.15, 0.3)
    qvel = (0.1, -0.2, 0.05, 0.3, -0.1, 0.2)
    wrench = B601SpatialWrench(
        force_n=(1.2, -2.5, 0.4),
        torque_nm=(0.2, 0.1, -0.3),
        frame_id=B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
    )

    result = map_world_aligned_ee_wrench_to_joint_torque(
        b601_model,
        qpos,
        wrench,
        qvel_rad_s=qvel,
    )
    expected = b601_model.arm_jacobian(qpos).T @ wrench.as_vector()

    np.testing.assert_allclose(result.joint_torque_nm, expected, atol=1e-12)
    assert result.cartesian_power_w == pytest.approx(result.joint_power_w, abs=1e-12)


def test_jacobian_transpose_mapping_rejects_wrong_frame(
    b601_model: B601PinocchioModel,
) -> None:
    with pytest.raises(ValueError, match="wrench frame"):
        map_world_aligned_ee_wrench_to_joint_torque(
            b601_model,
            (0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            B601SpatialWrench((1.0, 0.0, 0.0), (0.0,) * 3, "tool_local"),
        )


def test_virtual_spring_has_restoring_sign_damping_and_component_limits() -> None:
    spring = B601TranslationalVirtualSpring(
        anchor_position_m=(0.3, 0.0, 0.4),
        stiffness_n_per_m=(100.0, 50.0, 20.0),
        damping_n_s_per_m=(2.0, 4.0, 6.0),
        max_abs_force_n=(3.0, 2.0, 1.0),
    )

    result = spring.evaluate(
        position_m=(0.35, -0.02, 0.41),
        linear_velocity_m_s=(0.2, -0.1, 0.5),
    )

    np.testing.assert_allclose(result.displacement_m, (0.05, -0.02, 0.01))
    np.testing.assert_allclose(result.raw_force_n, (-5.4, 1.4, -3.2))
    np.testing.assert_allclose(result.wrench.force_n, (-3.0, 1.4, -1.0))
    np.testing.assert_allclose(result.wrench.torque_nm, 0.0)
    assert result.force_limited is True


def test_virtual_wall_is_unilateral_bounded_and_never_attractive() -> None:
    wall = B601VirtualWall(
        normal=(2.0, 0.0, 0.0),
        offset_m=0.35,
        stiffness_n_per_m=200.0,
        damping_n_s_per_m=10.0,
        max_force_n=4.0,
    )

    outside = wall.evaluate((0.34, 0.0, 0.0), (1.0, 0.0, 0.0))
    entering = wall.evaluate((0.36, 0.0, 0.0), (0.2, 0.0, 0.0))
    exiting = wall.evaluate((0.36, 0.0, 0.0), (-1.0, 0.0, 0.0))
    limited = wall.evaluate((0.40, 0.0, 0.0), (0.0, 0.0, 0.0))

    np.testing.assert_allclose(outside.wrench.force_n, 0.0)
    np.testing.assert_allclose(entering.wrench.force_n, (-4.0, 0.0, 0.0))
    np.testing.assert_allclose(exiting.wrench.force_n, 0.0)
    np.testing.assert_allclose(limited.wrench.force_n, (-4.0, 0.0, 0.0))
    assert entering.penetration_m == pytest.approx(0.01)
    assert entering.force_limited is True
    assert exiting.raw_restoring_force_n == pytest.approx(0.0)
    assert limited.force_limited is True


def test_virtual_fixture_configuration_and_samples_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive"):
        B601TranslationalVirtualSpring(
            anchor_position_m=(0.0,) * 3,
            stiffness_n_per_m=(1.0,) * 3,
            damping_n_s_per_m=(1.0,) * 3,
            max_abs_force_n=(1.0, 0.0, 1.0),
        )
    with pytest.raises(ValueError, match="non-zero"):
        B601VirtualWall(
            normal=(0.0,) * 3,
            offset_m=0.0,
            stiffness_n_per_m=1.0,
            damping_n_s_per_m=0.0,
            max_force_n=1.0,
        )
    spring = B601TranslationalVirtualSpring(
        anchor_position_m=(0.0,) * 3,
        stiffness_n_per_m=(1.0,) * 3,
        damping_n_s_per_m=(1.0,) * 3,
        max_abs_force_n=(1.0,) * 3,
    )
    with pytest.raises(ValueError, match="finite"):
        spring.evaluate((np.nan, 0.0, 0.0), (0.0,) * 3)
