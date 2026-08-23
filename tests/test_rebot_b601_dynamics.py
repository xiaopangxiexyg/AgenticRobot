from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_AUXILIARY_JOINT_NAMES,
    B601_RS_URDF_RELATIVE_PATH,
    B601MitControlProfile,
    B601MitReference,
    B601PinocchioModel,
    compute_b601_mit_torque,
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


def _profile(**overrides) -> B601MitControlProfile:
    values = {
        "name": "test-simulation-only",
        "position_kp_nm_per_rad": (2.0,) * 6,
        "damping_kd_nm_s_per_rad": (1.0,) * 6,
        "max_abs_torque_nm": (10.0,) * 6,
        "max_torque_rate_nm_s": (100.0,) * 6,
        "max_abs_overlay_torque_nm": (0.5,) * 6,
        "max_abs_reference_velocity_rad_s": (2.0,) * 6,
        "max_abs_state_velocity_rad_s": (2.0,) * 6,
        "joint_names": B601_RS_ARM_JOINT_NAMES,
        "metadata": {"hardware_safety_claim": False},
    }
    values.update(overrides)
    return B601MitControlProfile(**values)


def _reference(**overrides) -> B601MitReference:
    values = {
        "sequence": 1,
        "source_timestamp_sec": 1.0,
        "receive_timestamp_sec": 1.01,
        "qpos_rad": (0.1, 1.0, 1.1, 0.2, -0.1, 0.3),
    }
    values.update(overrides)
    return B601MitReference(**values)


def test_gravity_api_exposes_auxiliary_configuration_and_returns_copy(
    b601_model: B601PinocchioModel,
) -> None:
    qpos = np.asarray([0.1, 1.0, 1.1, 0.2, -0.1, 0.3], dtype=float)
    open_gripper = {"joint_left": 0.05, "joint_right": 0.0715}

    neutral = b601_model.arm_generalized_gravity(qpos)
    opened = b601_model.arm_generalized_gravity(
        qpos,
        auxiliary_joint_positions=open_gripper,
    )

    assert b601_model.auxiliary_joint_names == B601_RS_AUXILIARY_JOINT_NAMES
    assert b601_model.gravity_mps2 == pytest.approx((0.0, 0.0, -9.81))
    assert neutral.shape == (6,)
    assert np.all(np.isfinite(neutral))
    assert np.max(np.abs(neutral - opened)) > 1e-3
    neutral[0] = 123.0
    assert b601_model.arm_generalized_gravity(qpos)[0] != pytest.approx(123.0)


@pytest.mark.parametrize(
    "auxiliary, message",
    [
        ({"joint_left": 0.05}, "exactly"),
        (
            {"joint_left": 0.05, "joint_right": 0.0715, "extra": 0.0},
            "unexpected",
        ),
        ({"joint_left": np.nan, "joint_right": 0.0715}, "finite"),
        ({"joint_left": 0.051, "joint_right": 0.0715}, "exceeds"),
    ],
)
def test_gravity_api_rejects_ambiguous_or_invalid_auxiliary_positions(
    b601_model: B601PinocchioModel,
    auxiliary: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        b601_model.arm_generalized_gravity(
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            auxiliary_joint_positions=auxiliary,
        )


def test_mit_profile_is_local_bounded_and_immutable() -> None:
    profile = _profile()

    assert profile.joint_names == B601_RS_ARM_JOINT_NAMES
    assert profile.max_abs_reference_velocity_rad_s == (2.0,) * 6
    assert profile.max_abs_state_velocity_rad_s == (2.0,) * 6
    assert profile.metadata["hardware_safety_claim"] is False
    with pytest.raises(TypeError):
        profile.metadata["changed"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="non-negative"):
        _profile(damping_kd_nm_s_per_rad=(1.0, 1.0, -1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        _profile(max_abs_torque_nm=(10.0, 10.0, 0.0, 10.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="joint_names"):
        _profile(joint_names=tuple(reversed(B601_RS_ARM_JOINT_NAMES)))
    with pytest.raises(ValueError, match="positive"):
        _profile(max_abs_reference_velocity_rad_s=(0.0,) + (2.0,) * 5)
    with pytest.raises(ValueError, match="positive"):
        _profile(max_abs_state_velocity_rad_s=(0.0,) + (2.0,) * 5)


def test_mit_torque_decomposition_has_expected_sign_and_dissipative_damping() -> None:
    qpos = np.asarray([0.0, 1.1, 1.0, 0.0, 0.0, 0.2], dtype=float)
    qvel = np.asarray([0.2, -0.1, 0.3, -0.4, 0.5, -0.6], dtype=float)
    gravity = np.asarray([0.1, -2.0, 6.0, 1.0, 0.0, -0.02], dtype=float)
    reference = _reference(
        qvel_rad_s=(0.0,) * 6,
        joint_torque_overlay_nm=(1.0, -1.0, 0.25, 0.0, 0.6, -0.6),
    )
    profile = _profile()

    result = compute_b601_mit_torque(
        reference,
        profile,
        measured_qpos_rad=qpos,
        measured_qvel_rad_s=qvel,
        gravity_torque_nm=gravity,
    )

    expected_position = 2.0 * (np.asarray(reference.qpos_rad) - qpos)
    expected_damping = -qvel
    expected_overlay = np.clip(reference.joint_torque_overlay_nm, -0.5, 0.5)
    np.testing.assert_allclose(result.position_torque_nm, expected_position)
    np.testing.assert_allclose(result.damping_torque_nm, expected_damping)
    np.testing.assert_allclose(result.gravity_torque_nm, gravity)
    np.testing.assert_allclose(result.overlay_effective_nm, expected_overlay)
    np.testing.assert_allclose(
        result.requested_torque_nm,
        expected_position + expected_damping + gravity + expected_overlay,
    )
    assert result.overlay_limited is True
    assert float(np.dot(result.damping_torque_nm, qvel)) <= 1e-12


def test_mit_reference_rejects_malformed_temporal_or_vector_data() -> None:
    with pytest.raises(TypeError, match="non-negative integer"):
        _reference(sequence=True)
    with pytest.raises(TypeError, match="finite and non-negative"):
        _reference(source_timestamp_sec=True)
    with pytest.raises(ValueError, match="six finite"):
        _reference(qpos_rad=(0.0,) * 5)
    with pytest.raises(ValueError, match="six finite"):
        _reference(joint_torque_overlay_nm=(0.0,) * 5 + (np.nan,))
    with pytest.raises(TypeError, match="valid must be a bool"):
        _reference(valid="false")


def test_mit_torque_rejects_finite_inputs_that_overflow() -> None:
    profile = _profile(
        position_kp_nm_per_rad=(np.finfo(float).max,) * 6,
    )
    reference = _reference(qpos_rad=(2.0,) * 6)

    with pytest.raises(ValueError, match="non-finite"):
        compute_b601_mit_torque(
            reference,
            profile,
            measured_qpos_rad=(0.0,) * 6,
            measured_qvel_rad_s=(0.0,) * 6,
            gravity_torque_nm=(0.0,) * 6,
        )
