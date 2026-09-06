from __future__ import annotations

import numpy as np
import pytest

from robot_motion.position_control.cartesian_admittance import (
    CartesianAdmittanceWrenchSample,
    discrete_lowpass_alpha_to_tau,
)
from robot_motion.position_control.rotational_admittance import (
    BoundedRotationalAdmittance,
    RotationalAdmittanceConfig,
    compose_world_rotation_offset,
    rotational_admittance_config_for_profile,
)


def _config(**overrides: object) -> RotationalAdmittanceConfig:
    values: dict[str, object] = {
        "enabled_axes": (True, True, True),
        "control_frame": "solver_world",
        "stiffness_nm_rad": (10.0, 10.0, 10.0),
        "torque_deadband_nm": (0.04, 0.04, 0.04),
        "torque_filter_tau_sec": discrete_lowpass_alpha_to_tau(
            0.9,
            sample_hz=40.0,
        ),
        "torque_sign": (1.0, 1.0, 1.0),
        "max_offset_rad": (0.4, 0.4, 0.4),
        "max_velocity_rad_s": (0.35, 0.35, 0.35),
        "max_return_velocity_rad_s": (0.1, 0.1, 0.1),
        "max_output_slew_rad_s": (0.35, 0.35, 0.35),
        "return_torque_deadband_nm": (0.04, 0.04, 0.04),
        "return_rate_s": 1.0,
        "response_reference_hz": 40.0,
        "stale_after_sec": 0.2,
        "max_torque_dt_sec": 0.1,
    }
    values.update(overrides)
    return RotationalAdmittanceConfig(**values)


def _sample(
    *,
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0),
    source_time: float,
    frame: str = "tool",
    generation: int = 1,
) -> CartesianAdmittanceWrenchSample:
    return CartesianAdmittanceWrenchSample(
        values_n_nm=(0.0, 0.0, 0.0, *torque),
        frame=frame,
        source_time_sec=source_time,
        receive_time_sec=source_time,
        tare_generation=generation,
    )


def _controller(
    config: RotationalAdmittanceConfig | None = None,
) -> BoundedRotationalAdmittance:
    controller = BoundedRotationalAdmittance(config or _config())
    controller.reset(tare_sample=_sample(source_time=1.0), output_time_sec=1.0)
    return controller


def test_default_is_disabled() -> None:
    controller = BoundedRotationalAdmittance()
    controller.reset(tare_sample=_sample(source_time=1.0), output_time_sec=1.0)

    result = controller.step_new_sample(
        _sample(torque=(1.0, 0.0, 0.0), source_time=1.05),
        now_sec=1.05,
        rotation_control_from_sample=np.eye(3),
    )

    assert result.reason == "disabled"
    assert result.output_rotvec_rad == (0.0, 0.0, 0.0)


def test_new_20hz_torque_sample_is_integrated_only_once_across_60hz_reads() -> None:
    controller = _controller(_config(torque_filter_tau_sec=0.0))
    sample = _sample(torque=(1.0, 0.0, 0.0), source_time=1.05)

    first = controller.step_new_sample(
        sample,
        now_sec=1.05,
        rotation_control_from_sample=np.eye(3),
    )
    duplicate = controller.step_new_sample(
        sample,
        now_sec=1.06,
        rotation_control_from_sample=np.eye(3),
    )
    read_one = controller.sample_output(now_sec=1.05 + 1.0 / 60.0)
    read_two = controller.sample_output(now_sec=1.05 + 2.0 / 60.0)

    assert first.new_sample_integrated is True
    assert first.integration_dt_sec == pytest.approx(0.05)
    assert duplicate.reason == "duplicate_sample"
    assert duplicate.requested_rotvec_rad == first.requested_rotvec_rad
    assert read_one.requested_rotvec_rad == first.requested_rotvec_rad
    assert read_two.requested_rotvec_rad == first.requested_rotvec_rad


def test_rotation_output_follow_is_continuous_and_never_catches_up_after_release() -> None:
    controller = _controller(
        _config(
            torque_filter_tau_sec=0.0,
            output_follow_tau_sec=0.025,
        )
    )
    controller.sample_output(now_sec=1.0 + 1.0 / 60.0)
    controller.sample_output(now_sec=1.0 + 2.0 / 60.0)

    integrated = controller.step_new_sample(
        _sample(torque=(0.05, 0.0, 0.0), source_time=1.05),
        now_sec=1.05,
        rotation_control_from_sample=np.eye(3),
    )
    first = controller.sample_output(now_sec=1.05)
    second = controller.sample_output(now_sec=1.05 + 1.0 / 60.0)

    assert integrated.requested_rotvec_rad[0] == pytest.approx(0.001)
    assert 0.0 < first.output_rotvec_rad[0] < integrated.requested_rotvec_rad[0]
    assert first.output_rotvec_rad[0] < second.output_rotvec_rad[0]
    assert second.output_rotvec_rad[0] < integrated.requested_rotvec_rad[0]

    controller.step_new_sample(
        _sample(source_time=1.10),
        now_sec=1.10,
        rotation_control_from_sample=np.eye(3),
    )
    after_release = controller.sample_output(now_sec=1.10)
    assert after_release.output_rotvec_rad[0] <= second.output_rotvec_rad[0] + 1e-12


def test_rotation_output_acceleration_limit_bounds_nonpassive_velocity_changes() -> None:
    acceleration_limit = 3.0
    controller = _controller(
        _config(
            torque_filter_tau_sec=0.0,
            output_follow_tau_sec=0.025,
            max_output_angular_acceleration_rad_s2=(acceleration_limit,) * 3,
        )
    )
    outputs = [np.asarray(controller.snapshot(now_sec=1.0).output_rotvec_rad)]
    times = [1.0]
    for tick in range(1, 13):
        now = 1.0 + tick / 60.0
        if tick % 3 == 0:
            torque = 1.0 if tick <= 6 else -1.0
            controller.step_new_sample(
                _sample(torque=(torque, 0.0, 0.0), source_time=now),
                now_sec=now,
                rotation_control_from_sample=np.eye(3),
            )
        state = controller.sample_output(now_sec=now)
        outputs.append(np.asarray(state.output_rotvec_rad))
        times.append(now)

    output = np.asarray(outputs)
    dt = np.diff(np.asarray(times))
    velocity = np.diff(output, axis=0) / dt[:, None]
    acceleration = np.diff(velocity, axis=0) / dt[1:, None]

    assert np.max(np.linalg.norm(acceleration, axis=1)) <= (
        acceleration_limit + 1e-9
    )


def test_tool_torque_is_rotated_into_control_frame_before_deadband() -> None:
    controller = _controller(
        _config(
            enabled_axes=(False, True, False),
            torque_deadband_nm=(0.0, 0.0, 0.0),
            torque_filter_tau_sec=0.0,
        )
    )
    control_from_tool = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    result = controller.step_new_sample(
        _sample(torque=(1.0, 0.0, 0.0), source_time=1.05),
        now_sec=1.05,
        rotation_control_from_sample=control_from_tool,
    )

    assert result.filtered_torque_control_nm == pytest.approx((0.0, 1.0, 0.0))
    assert result.requested_rotvec_rad[1] > 0.0
    assert result.requested_rotvec_rad[0] == 0.0


def test_profile_compliance_matches_established_two_40hz_updates() -> None:
    from hardware.rm65.force_compensation import PoseAdmittance, WrenchProcessor

    processor = WrenchProcessor(
        tare=[0.0] * 6,
        lowpass_alpha=0.9,
        force_deadband_n=0.45,
        torque_deadband_nm=0.04,
    )
    legacy = PoseAdmittance(
        enabled_axes=[False, False, False, False, False, True],
        angular_force_sign=1.0,
        angular_axis_signs=[1.0, 1.0, 1.0],
        angular_compliance_rad_per_nm=0.1,
        max_angular_velocity_rad_s=0.35,
        max_angular_return_velocity_rad_s=0.1,
        max_angular_offset_rad=0.4,
        return_rate_s=1.0,
    )
    controller = _controller(_config(enabled_axes=(False, False, True)))

    for _ in range(2):
        processed = processor.process([0.0] * 6)
        legacy.update(
            processed["deadbanded"],
            0.025,
            return_activity_wrench=processed["filtered"],
            return_torque_deadband_nm=0.04,
        )
    controller.step_new_sample(
        _sample(source_time=1.05),
        now_sec=1.05,
        rotation_control_from_sample=np.eye(3),
    )

    for index, torque_z in enumerate((1.0, 1.0, 0.0, 0.0), start=2):
        for _ in range(2):
            processed = processor.process([0.0, 0.0, 0.0, 0.0, 0.0, torque_z])
            legacy_result = legacy.update(
                processed["deadbanded"],
                0.025,
                return_activity_wrench=processed["filtered"],
                return_torque_deadband_nm=0.04,
            )
        state = controller.step_new_sample(
            _sample(torque=(0.0, 0.0, torque_z), source_time=1.0 + index * 0.05),
            now_sec=1.0 + index * 0.05,
            rotation_control_from_sample=np.eye(3),
        )
        assert state.requested_rotvec_rad[2] == pytest.approx(
            legacy_result["offset"][5], abs=1e-12
        )


def test_total_offset_and_output_slew_are_norm_bounded() -> None:
    controller = _controller(
        _config(
            torque_deadband_nm=(0.0, 0.0, 0.0),
            torque_filter_tau_sec=0.0,
            max_offset_rad=(0.1, 0.1, 0.1),
            max_velocity_rad_s=(10.0, 10.0, 10.0),
            max_output_slew_rad_s=(0.2, 0.2, 0.2),
        )
    )
    controller.step_new_sample(
        _sample(torque=(10.0, 10.0, 10.0), source_time=1.05),
        now_sec=1.05,
        rotation_control_from_sample=np.eye(3),
    )
    output = controller.sample_output(now_sec=1.05)

    assert np.linalg.norm(output.requested_rotvec_rad) == pytest.approx(0.1)
    assert np.linalg.norm(output.output_rotvec_rad) == pytest.approx(0.01)


def test_sudden_release_returns_without_jump_or_hidden_catchup() -> None:
    controller = _controller(
        _config(
            torque_filter_tau_sec=0.0,
            max_velocity_rad_s=(0.35, 0.35, 0.35),
            max_return_velocity_rad_s=(0.1, 0.1, 0.1),
            max_output_slew_rad_s=(0.35, 0.35, 0.35),
        )
    )
    controller.step_new_sample(
        _sample(torque=(1.0, 0.0, 0.0), source_time=1.05),
        now_sec=1.05,
        rotation_control_from_sample=np.eye(3),
    )
    before = controller.sample_output(now_sec=1.05)
    controller.step_new_sample(
        _sample(source_time=1.10),
        now_sec=1.10,
        rotation_control_from_sample=np.eye(3),
    )
    after = controller.sample_output(now_sec=1.10)

    assert np.linalg.norm(after.output_rotvec_rad) <= np.linalg.norm(
        before.output_rotvec_rad
    ) + 1e-12
    assert np.linalg.norm(after.angular_velocity_rad_s) <= 0.1 + 1e-12

    now = 1.10
    for index in range(1, 201):
        if index % 3 == 0:
            controller.step_new_sample(
                _sample(source_time=now + index / 60.0),
                now_sec=now + index / 60.0,
                rotation_control_from_sample=np.eye(3),
            )
        state = controller.sample_output(now_sec=now + index / 60.0)
    assert np.linalg.norm(state.output_rotvec_rad) < 1e-3


def test_balanced_profile_abrupt_multiaxis_reversal_and_release_is_slew_bounded() -> None:
    config = rotational_admittance_config_for_profile("balanced-5deg")
    controller = _controller(config)
    previous = np.zeros(3, dtype=float)
    previous_time = 1.0

    for tick in range(1, 361):
        now = 1.0 + tick / 60.0
        elapsed = now - 1.0
        if tick % 3 == 0:
            torque = (
                (4.0, -4.0, 4.0)
                if elapsed <= 1.0 + 1e-12
                else (-4.0, 4.0, -4.0)
                if elapsed <= 1.25 + 1e-12
                else (0.0, 0.0, 0.0)
            )
            controller.step_new_sample(
                _sample(torque=torque, source_time=now),
                now_sec=now,
                rotation_control_from_sample=np.eye(3),
            )
        state = controller.sample_output(now_sec=now)
        output = np.asarray(state.output_rotvec_rad, dtype=float)
        dt = now - previous_time
        assert np.linalg.norm(output - previous) <= (
            max(config.max_output_slew_rad_s) * dt + 1e-12
        )
        assert np.linalg.norm(output) <= max(config.max_offset_rad) + 1e-12
        previous = output
        previous_time = now

    assert np.linalg.norm(previous) < 1e-3


def test_rotation_composition_is_world_frame_so3_not_euler_addition() -> None:
    raw = compose_world_rotation_offset(
        np.eye(3),
        (0.0, np.deg2rad(90.0), 0.0),
    )
    offset = compose_world_rotation_offset(
        raw,
        (np.deg2rad(10.0), 0.0, 0.0),
    )

    np.testing.assert_allclose(offset.T @ offset, np.eye(3), atol=1e-12)
    assert np.linalg.det(offset) == pytest.approx(1.0)
    expected = compose_world_rotation_offset(np.eye(3), (np.deg2rad(10.0), 0.0, 0.0)) @ raw
    np.testing.assert_allclose(offset, expected, atol=1e-12)


@pytest.mark.parametrize(
    ("sample", "now", "reason", "rotation"),
    [
        (
            _sample(source_time=1.05, generation=2),
            1.05,
            "tare_generation_mismatch",
            np.eye(3),
        ),
        (
            _sample(source_time=1.05, frame="arm_base"),
            1.05,
            "tare_frame_mismatch",
            np.eye(3),
        ),
        (
            _sample(source_time=1.05),
            1.05,
            "frame_transform_missing",
            None,
        ),
        (
            _sample(source_time=0.95),
            1.05,
            "sample_time_not_increasing",
            np.eye(3),
        ),
        (
            _sample(source_time=1.15),
            1.15,
            "torque_dt_exceeded",
            np.eye(3),
        ),
        (
            _sample(source_time=1.05),
            1.30,
            "sample_stale",
            np.eye(3),
        ),
    ],
)
def test_invalid_rotation_sample_contract_freezes_without_snapback(
    sample: CartesianAdmittanceWrenchSample,
    now: float,
    reason: str,
    rotation: np.ndarray | None,
) -> None:
    controller = _controller()
    controller.rebase_realized_rotation((0.03, 0.0, 0.0), output_time_sec=1.0)

    result = controller.step_new_sample(
        sample,
        now_sec=now,
        rotation_control_from_sample=rotation,
    )

    assert result.frozen is True
    assert result.reason == reason
    assert result.requested_rotvec_rad == pytest.approx((0.03, 0.0, 0.0))
    assert result.output_rotvec_rad == pytest.approx((0.03, 0.0, 0.0))
    assert result.angular_velocity_rad_s == (0.0, 0.0, 0.0)


def test_rotation_freeze_resume_rearms_before_integration_and_reset_clears() -> None:
    controller = _controller(_config(torque_filter_tau_sec=0.0))
    controller.rebase_realized_rotation((0.03, 0.0, 0.0), output_time_sec=1.0)
    controller.freeze("hold")

    frozen = controller.step_new_sample(
        _sample(torque=(1.0, 0.0, 0.0), source_time=1.05),
        now_sec=1.05,
        rotation_control_from_sample=np.eye(3),
    )
    assert frozen.reason == "hold"
    assert frozen.output_rotvec_rad == pytest.approx((0.03, 0.0, 0.0))

    controller.resume()
    rearmed = controller.step_new_sample(
        _sample(torque=(1.0, 0.0, 0.0), source_time=1.50),
        now_sec=1.50,
        rotation_control_from_sample=np.eye(3),
    )
    assert rearmed.reason == "sample_clock_rearmed"
    assert rearmed.new_sample_integrated is False
    assert rearmed.output_rotvec_rad == pytest.approx((0.03, 0.0, 0.0))

    resumed = controller.step_new_sample(
        _sample(torque=(1.0, 0.0, 0.0), source_time=1.55),
        now_sec=1.55,
        rotation_control_from_sample=np.eye(3),
    )
    assert resumed.new_sample_integrated is True
    assert resumed.integration_dt_sec == pytest.approx(0.05)

    controller.reset(
        tare_sample=_sample(source_time=2.0, generation=2),
        output_time_sec=2.0,
    )
    reset = controller.snapshot(now_sec=2.0)
    assert reset.frozen is False
    assert reset.tare_generation == 2
    assert reset.requested_rotvec_rad == (0.0, 0.0, 0.0)
    assert reset.output_rotvec_rad == (0.0, 0.0, 0.0)


def test_rotation_rebase_and_output_clocks_preserve_exact_fail_closed_state() -> None:
    controller = _controller(_config(enabled_axes=(True, False, True)))
    rebased = controller.rebase_realized_rotation(
        (1.0, 1.0, -1.0),
        output_time_sec=1.10,
    )

    assert np.linalg.norm(rebased.output_rotvec_rad) == pytest.approx(0.4)
    assert rebased.output_rotvec_rad[1] == 0.0
    with pytest.raises(ValueError, match="must not move backwards"):
        controller.rebase_realized_rotation(
            (0.02, 0.0, 0.0),
            output_time_sec=1.05,
        )

    duplicate = controller.sample_output(now_sec=1.10)
    assert duplicate.frozen is False
    assert duplicate.reason == "duplicate_output_time"
    rollback = controller.sample_output(now_sec=1.09)
    assert rollback.frozen is True
    assert rollback.reason == "output_time_not_increasing"
    assert rollback.output_rotvec_rad == pytest.approx(rebased.output_rotvec_rad)


def test_rotation_future_and_inactive_events_freeze_without_zeroing_output() -> None:
    future_controller = _controller()
    future_controller.rebase_realized_rotation(
        (0.03, 0.0, 0.0),
        output_time_sec=1.0,
    )
    future = future_controller.step_new_sample(
        _sample(source_time=1.06),
        now_sec=1.05,
        rotation_control_from_sample=np.eye(3),
    )
    assert future.frozen is True
    assert future.reason == "sample_from_future"
    assert future.output_rotvec_rad == pytest.approx((0.03, 0.0, 0.0))

    inactive_controller = _controller()
    inactive_controller.rebase_realized_rotation(
        (0.03, 0.0, 0.0),
        output_time_sec=1.0,
    )
    inactive = inactive_controller.sample_output(now_sec=1.05, active=False)
    assert inactive.frozen is True
    assert inactive.reason == "inactive"
    assert inactive.output_rotvec_rad == pytest.approx((0.03, 0.0, 0.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled_axes": (True, False)},
        {"torque_sign": (1.0, 0.0, 1.0)},
        {"max_offset_rad": (0.0, 0.1, 0.1)},
        {"torque_deadband_nm": (-0.1, 0.0, 0.0)},
        {"output_follow_tau_sec": -0.1},
        {"max_output_angular_acceleration_rad_s2": (-0.1, 0.0, 0.0)},
        {
            "enabled_axes": (True, True, True),
            "max_output_angular_acceleration_rad_s2": (3.0, 0.0, 3.0),
        },
    ],
)
def test_invalid_config_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RotationalAdmittanceConfig(**kwargs)  # type: ignore[arg-type]
