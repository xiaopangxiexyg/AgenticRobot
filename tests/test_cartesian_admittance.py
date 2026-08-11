from __future__ import annotations

import numpy as np
import pytest

from robot_motion.position_control.cartesian_admittance import (
    BoundedCartesianAdmittance,
    CartesianAdmittanceConfig,
    CartesianAdmittanceWrenchSample,
    discrete_lowpass_alpha_to_tau,
)


def _config(**overrides: object) -> CartesianAdmittanceConfig:
    values: dict[str, object] = {
        "enabled_axes": (True, False, False),
        "control_frame": "arm_base",
        "virtual_mass_kg": (1.0, 1.0, 1.0),
        "damping_n_s_m": (0.0, 0.0, 0.0),
        "stiffness_n_m": (0.0, 0.0, 0.0),
        "force_deadband_n": (0.0, 0.0, 0.0),
        "force_filter_tau_sec": 0.0,
        "max_offset_m": (0.01, 0.01, 0.01),
        "max_velocity_m_s": (0.1, 0.1, 0.1),
        "max_output_slew_m_s": (0.02, 0.02, 0.02),
        "stale_after_sec": 0.2,
        "max_force_dt_sec": 0.1,
    }
    values.update(overrides)
    return CartesianAdmittanceConfig(**values)


def _sample(
    *,
    force: tuple[float, float, float] = (0.0, 0.0, 0.0),
    source_time: float,
    generation: int = 1,
    frame: str = "arm_base",
) -> CartesianAdmittanceWrenchSample:
    return CartesianAdmittanceWrenchSample(
        values_n_nm=(*force, 0.0, 0.0, 0.0),
        frame=frame,
        source_time_sec=source_time,
        receive_time_sec=source_time + 0.001,
        tare_generation=generation,
    )


def _controller(config: CartesianAdmittanceConfig | None = None) -> BoundedCartesianAdmittance:
    controller = BoundedCartesianAdmittance(config or _config())
    controller.reset(tare_sample=_sample(source_time=1.0), output_time_sec=1.0)
    return controller


def _profile_compliance_config(**overrides: object) -> CartesianAdmittanceConfig:
    values: dict[str, object] = {
        "enabled_axes": (True, True, True),
        "control_frame": "arm_base",
        "response_model": "profile_compliance",
        "stiffness_n_m": (250.0, 250.0, 250.0),
        "force_deadband_n": (0.45, 0.45, 0.45),
        "return_force_deadband_n": (0.45, 0.45, 0.45),
        "force_filter_tau_sec": discrete_lowpass_alpha_to_tau(
            0.9,
            sample_hz=40.0,
        ),
        "max_offset_m": (0.08, 0.08, 0.08),
        "max_velocity_m_s": (0.06, 0.06, 0.06),
        "max_return_velocity_m_s": (0.03, 0.03, 0.03),
        "max_output_slew_m_s": (0.06, 0.06, 0.06),
        "return_rate_s": 1.0,
        "response_reference_hz": 40.0,
        "stale_after_sec": 0.2,
        "max_force_dt_sec": 0.1,
    }
    values.update(overrides)
    return CartesianAdmittanceConfig(**values)


def test_default_config_is_strictly_disabled() -> None:
    controller = BoundedCartesianAdmittance()
    controller.reset(tare_sample=_sample(source_time=1.0), output_time_sec=1.0)
    result = controller.step_new_sample(
        _sample(force=(10.0, 0.0, 0.0), source_time=1.05),
        now_sec=1.05,
    )

    assert result.reason == "disabled"
    assert result.requested_offset_m == (0.0, 0.0, 0.0)
    assert result.output_offset_m == (0.0, 0.0, 0.0)


def test_new_20hz_sample_is_integrated_only_once_across_60hz_reads() -> None:
    controller = _controller()
    sample = _sample(force=(1.0, 0.0, 0.0), source_time=1.05)

    first = controller.step_new_sample(sample, now_sec=1.05)
    duplicate = controller.step_new_sample(sample, now_sec=1.06)
    read_one = controller.sample_output(now_sec=1.05 + 1.0 / 60.0)
    read_two = controller.sample_output(now_sec=1.05 + 2.0 / 60.0)

    assert first.new_sample_integrated is True
    assert first.integration_dt_sec == pytest.approx(0.05)
    assert first.requested_offset_m[0] == pytest.approx(0.0025)
    assert duplicate.reason == "duplicate_sample"
    assert duplicate.requested_offset_m == first.requested_offset_m
    assert read_one.requested_offset_m == first.requested_offset_m
    assert read_two.requested_offset_m == first.requested_offset_m
    assert read_two.output_offset_m[0] == pytest.approx(0.02 * (1.05 + 2.0 / 60.0 - 1.0))


def test_deadband_and_disabled_axes_are_identity() -> None:
    controller = _controller(
        _config(
            enabled_axes=(True, False, True),
            force_deadband_n=(1.0, 0.0, 1.0),
        )
    )
    result = controller.step_new_sample(
        _sample(force=(0.9, 100.0, -0.5), source_time=1.05),
        now_sec=1.05,
    )

    assert result.deadbanded_force_control_n == (0.0, 0.0, 0.0)
    assert result.requested_offset_m == (0.0, 0.0, 0.0)


def test_force_low_pass_is_causal_and_reported_separately() -> None:
    controller = _controller(_config(force_filter_tau_sec=0.05))
    result = controller.step_new_sample(
        _sample(force=(1.0, 0.0, 0.0), source_time=1.05),
        now_sec=1.05,
    )

    expected = 1.0 - np.exp(-1.0)
    assert result.deadbanded_force_control_n[0] == pytest.approx(1.0)
    assert result.filtered_force_control_n[0] == pytest.approx(expected)
    assert result.requested_offset_m[0] == pytest.approx(expected * 0.05**2)


def test_discrete_filter_conversion_preserves_established_40hz_pole() -> None:
    tau = discrete_lowpass_alpha_to_tau(0.9, sample_hz=40.0)
    alpha_20hz = 1.0 - np.exp(-0.05 / tau)

    assert tau == pytest.approx(0.010857362, rel=1e-7)
    assert alpha_20hz == pytest.approx(0.99)


def test_profile_compliance_matches_two_40hz_updates_at_20hz_boundaries() -> None:
    """Match the legacy filtered-compliance law without duplicate integration."""

    from hardware.rm65.force_compensation import PoseAdmittance, WrenchProcessor

    legacy_processor = WrenchProcessor(
        tare=[0.0] * 6,
        lowpass_alpha=0.9,
        force_deadband_n=0.45,
        torque_deadband_nm=0.04,
    )
    legacy_admittance = PoseAdmittance(
        enabled_axes=[True, True, True, False, False, False],
        force_sign=1.0,
        linear_gain_m_s_per_n=0.0,
        linear_compliance_m_per_n=0.004,
        max_linear_velocity_m_s=0.06,
        max_linear_return_velocity_m_s=0.03,
        max_linear_offset_m=0.08,
        return_rate_s=1.0,
    )
    controller = _controller(_profile_compliance_config())

    # Initialize both filters from a zero tare-corrected reading.
    for _ in range(2):
        processed = legacy_processor.process([0.0] * 6)
        legacy_admittance.update(
            processed["deadbanded"],
            0.025,
            return_activity_wrench=processed["filtered"],
            return_force_deadband_n=0.45,
        )
    initialized = controller.step_new_sample(
        _sample(source_time=1.05),
        now_sec=1.05,
    )
    assert initialized.filtered_force_control_n == (0.0, 0.0, 0.0)

    # The 20 Hz sample represents the same held force observed by two legacy
    # 40 Hz updates.  Compare only common physical boundaries.
    force_steps = (4.0, 4.0, 0.0, 0.0)
    for index, force_x in enumerate(force_steps, start=2):
        for _ in range(2):
            processed = legacy_processor.process([force_x, 0.0, 0.0, 0.0, 0.0, 0.0])
            legacy_admittance.update(
                processed["deadbanded"],
                0.025,
                return_activity_wrench=processed["filtered"],
                return_force_deadband_n=0.45,
            )
        now = 1.0 + index * 0.05
        state = controller.step_new_sample(
            _sample(force=(force_x, 0.0, 0.0), source_time=now),
            now_sec=now,
        )
        assert state.filtered_force_control_n[0] == pytest.approx(
            processed["filtered"][0],
            abs=1e-12,
        )
        assert state.deadbanded_force_control_n[0] == pytest.approx(
            processed["deadbanded"][0],
            abs=1e-12,
        )
        assert state.requested_offset_m[0] == pytest.approx(
            legacy_admittance.offset[0],
            abs=1e-12,
        )


def test_profile_compliance_60hz_output_is_a_bounded_ramp_not_three_integrations() -> None:
    controller = _controller(
        _profile_compliance_config(
            max_velocity_m_s=(0.06, 0.06, 0.06),
            max_output_slew_m_s=(0.06, 0.06, 0.06),
        )
    )
    controller.sample_output(now_sec=1.0 + 1.0 / 60.0)
    controller.sample_output(now_sec=1.0 + 2.0 / 60.0)
    controller.step_new_sample(_sample(source_time=1.05), now_sec=1.05)
    for tick in range(3, 7):
        controller.sample_output(now_sec=1.0 + tick / 60.0)
    requested = controller.step_new_sample(
        _sample(force=(4.0, 0.0, 0.0), source_time=1.10),
        now_sec=1.10,
    )

    outputs = [
        controller.sample_output(now_sec=1.10 + tick / 60.0).output_offset_m[0]
        for tick in range(1, 4)
    ]
    assert requested.requested_offset_m[0] == pytest.approx(0.003)
    assert outputs == pytest.approx((0.001, 0.002, 0.003))
    assert controller.snapshot(now_sec=1.15).requested_offset_m[0] == pytest.approx(0.003)


def test_profile_compliance_abrupt_release_is_monotonic_and_slew_bounded() -> None:
    controller = _controller(
        _profile_compliance_config(
            max_velocity_m_s=(0.06, 0.06, 0.06),
            max_return_velocity_m_s=(0.03, 0.03, 0.03),
            max_output_slew_m_s=(0.06, 0.06, 0.06),
        )
    )
    controller.sample_output(now_sec=1.0 + 1.0 / 60.0)
    controller.sample_output(now_sec=1.0 + 2.0 / 60.0)
    controller.step_new_sample(
        _sample(source_time=1.05),
        now_sec=1.05,
    )
    for tick in range(3, 7):
        controller.sample_output(now_sec=1.0 + tick / 60.0)
    controller.step_new_sample(
        _sample(force=(4.0, 0.0, 0.0), source_time=1.10),
        now_sec=1.10,
    )
    loaded = [
        controller.sample_output(now_sec=1.10 + tick / 60.0).output_offset_m[0]
        for tick in range(1, 4)
    ]
    assert loaded == pytest.approx((0.001, 0.002, 0.003))

    released = controller.step_new_sample(
        _sample(force=(0.0, 0.0, 0.0), source_time=1.15),
        now_sec=1.15,
    )
    outputs = [loaded[-1]]
    for tick in range(1, 4):
        outputs.append(
            controller.sample_output(
                now_sec=1.15 + tick / 60.0
            ).output_offset_m[0]
        )

    assert 0.0 <= released.requested_offset_m[0] < loaded[-1]
    assert all(
        current <= previous + 1e-12
        for previous, current in zip(outputs, outputs[1:])
    )
    assert all(current >= -1e-12 for current in outputs)
    assert max(
        abs(current - previous)
        for previous, current in zip(outputs, outputs[1:])
    ) <= 0.001 + 1e-12


def test_frame_transform_is_required_and_applied_explicitly() -> None:
    missing = _controller()
    missing_result = missing.step_new_sample(
        _sample(force=(0.0, 2.0, 0.0), source_time=1.05, frame="tool"),
        now_sec=1.05,
    )
    assert missing_result.frozen is True
    assert missing_result.reason == "tare_frame_mismatch"

    controller = BoundedCartesianAdmittance(_config())
    controller.reset(
        tare_sample=_sample(source_time=1.0, frame="tool"),
        output_time_sec=1.0,
    )
    rotation = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = controller.step_new_sample(
        _sample(force=(0.0, 2.0, 0.0), source_time=1.05, frame="tool"),
        now_sec=1.05,
        rotation_control_from_sample=rotation,
    )
    assert result.deadbanded_force_control_n == pytest.approx((2.0, 0.0, 0.0))
    assert result.requested_offset_m[0] > 0.0


@pytest.mark.parametrize(
    ("sample", "now", "reason"),
    [
        (_sample(source_time=1.05, generation=2), 1.05, "tare_generation_mismatch"),
        (_sample(source_time=0.95), 1.05, "sample_time_not_increasing"),
        (_sample(source_time=1.15), 1.15, "force_dt_exceeded"),
        (_sample(source_time=1.05), 1.30, "sample_stale"),
    ],
)
def test_invalid_sample_contract_freezes_without_snapback(
    sample: CartesianAdmittanceWrenchSample,
    now: float,
    reason: str,
) -> None:
    controller = _controller()
    controller.rebase_realized_offset((0.003, 0.0, 0.0), output_time_sec=1.0)
    result = controller.step_new_sample(sample, now_sec=now)

    assert result.frozen is True
    assert result.reason == reason
    assert result.requested_offset_m == pytest.approx((0.003, 0.0, 0.0))
    assert result.output_offset_m == pytest.approx((0.003, 0.0, 0.0))
    assert result.velocity_m_s == (0.0, 0.0, 0.0)


def test_realized_offset_rebase_rejects_a_backdated_output_clock() -> None:
    controller = _controller()
    controller.rebase_realized_offset(
        (0.003, 0.0, 0.0),
        output_time_sec=1.10,
    )

    with pytest.raises(ValueError, match="must not move backwards"):
        controller.rebase_realized_offset(
            (0.004, 0.0, 0.0),
            output_time_sec=1.05,
        )

    snapshot = controller.snapshot(now_sec=1.10)
    assert snapshot.output_offset_m == pytest.approx((0.003, 0.0, 0.0))


def test_stale_output_freezes_at_last_value_instead_of_returning_to_zero() -> None:
    controller = _controller()
    controller.rebase_realized_offset((0.004, 0.0, 0.0), output_time_sec=1.0)
    result = controller.sample_output(now_sec=1.25)

    assert result.frozen is True
    assert result.reason == "sample_stale"
    assert result.output_offset_m == pytest.approx((0.004, 0.0, 0.0))


def test_freeze_resume_requires_fresh_sample_and_reset_clears_old_offset() -> None:
    controller = _controller()
    controller.rebase_realized_offset((0.004, 0.0, 0.0), output_time_sec=1.0)
    controller.freeze("hold")
    frozen = controller.step_new_sample(
        _sample(force=(2.0, 0.0, 0.0), source_time=1.05),
        now_sec=1.05,
    )
    assert frozen.reason == "hold"
    assert frozen.output_offset_m == pytest.approx((0.004, 0.0, 0.0))

    controller.resume()
    rearmed = controller.step_new_sample(
        _sample(force=(2.0, 0.0, 0.0), source_time=1.5),
        now_sec=1.5,
    )
    assert rearmed.reason == "sample_clock_rearmed"
    assert rearmed.new_sample_integrated is False
    assert rearmed.output_offset_m == pytest.approx((0.004, 0.0, 0.0))

    resumed = controller.step_new_sample(
        _sample(force=(2.0, 0.0, 0.0), source_time=1.55),
        now_sec=1.55,
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
    assert reset.requested_offset_m == (0.0, 0.0, 0.0)
    assert reset.output_offset_m == (0.0, 0.0, 0.0)


def test_offset_velocity_and_output_slew_limits_are_independent() -> None:
    controller = _controller(
        _config(
            max_offset_m=(0.003, 0.003, 0.003),
            max_velocity_m_s=(0.02, 0.02, 0.02),
            max_output_slew_m_s=(0.005, 0.005, 0.005),
        )
    )
    integrated = controller.step_new_sample(
        _sample(force=(100.0, 0.0, 0.0), source_time=1.05),
        now_sec=1.05,
    )
    output = controller.sample_output(now_sec=1.05)

    assert integrated.velocity_m_s[0] == pytest.approx(0.02)
    assert integrated.requested_offset_m[0] == pytest.approx(0.001)
    assert output.output_offset_m[0] == pytest.approx(0.00025)


def test_multiaxis_limits_bound_total_translation_norm() -> None:
    controller = _controller(
        _config(
            enabled_axes=(True, True, True),
            max_offset_m=(0.003, 0.003, 0.003),
            max_velocity_m_s=(0.02, 0.02, 0.02),
            max_output_slew_m_s=(0.005, 0.005, 0.005),
        )
    )

    first = controller.step_new_sample(
        _sample(force=(100.0, 100.0, 100.0), source_time=1.05),
        now_sec=1.05,
    )
    assert np.linalg.norm(first.velocity_m_s) == pytest.approx(0.02)

    for index in range(2, 8):
        integrated = controller.step_new_sample(
            _sample(
                force=(100.0, 100.0, 100.0),
                source_time=1.0 + index * 0.05,
            ),
            now_sec=1.0 + index * 0.05,
        )
    assert np.linalg.norm(integrated.requested_offset_m) == pytest.approx(0.003)

    output = controller.sample_output(now_sec=1.05)
    assert np.linalg.norm(output.output_offset_m) == pytest.approx(0.00025)


def test_multiaxis_rebase_uses_total_translation_norm_limit() -> None:
    controller = _controller(
        _config(
            enabled_axes=(True, True, True),
            max_offset_m=(0.01, 0.01, 0.01),
        )
    )
    result = controller.rebase_realized_offset(
        (0.5, 0.5, 0.5),
        output_time_sec=1.1,
    )

    assert np.linalg.norm(result.requested_offset_m) == pytest.approx(0.01)
    assert result.output_offset_m == pytest.approx(result.requested_offset_m)


def test_output_sampling_reasserts_absolute_total_translation_envelope() -> None:
    controller = _controller(
        _config(
            enabled_axes=(True, True, True),
            max_offset_m=(0.005, 0.005, 0.005),
            max_output_slew_m_s=(0.01, 0.01, 0.01),
        )
    )
    controller._output_offset = np.asarray((0.004, 0.004, 0.0), dtype=float)

    result = controller.sample_output(now_sec=1.05)

    assert np.linalg.norm(result.output_offset_m) == pytest.approx(0.005)


def test_rebase_clamps_realized_offset_and_never_enables_disabled_axis() -> None:
    controller = _controller()
    result = controller.rebase_realized_offset(
        (0.5, 0.5, -0.5),
        output_time_sec=1.1,
    )

    assert result.requested_offset_m == pytest.approx((0.01, 0.0, 0.0))
    assert result.output_offset_m == pytest.approx((0.01, 0.0, 0.0))
    assert result.velocity_m_s == (0.0, 0.0, 0.0)


def test_configuration_rejects_rotation_like_or_unsafe_values() -> None:
    with pytest.raises(ValueError, match="enabled_axes"):
        CartesianAdmittanceConfig(enabled_axes=(True, False))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="force_sign"):
        CartesianAdmittanceConfig(force_sign=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        CartesianAdmittanceConfig(max_offset_m=(0.0, 0.01, 0.01))


def test_future_sample_and_output_clock_rollback_freeze() -> None:
    future = _controller()
    future_result = future.step_new_sample(
        _sample(source_time=1.06),
        now_sec=1.05,
    )
    assert future_result.frozen is True
    assert future_result.reason == "sample_from_future"

    rollback = _controller()
    duplicate = rollback.sample_output(now_sec=1.0)
    assert duplicate.frozen is False
    assert duplicate.reason == "duplicate_output_time"
    rollback_result = rollback.sample_output(now_sec=0.99)
    assert rollback_result.frozen is True
    assert rollback_result.reason == "output_time_not_increasing"
