from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.position_control.cartesian_admittance import (
    BoundedCartesianAdmittance,
    CartesianAdmittanceConfig,
    CartesianAdmittanceWrenchSample,
)
from robot_motion.position_control.rotational_admittance import (
    ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
    BoundedRotationalAdmittance,
    rotational_admittance_config_for_profile,
)
from robot_motion.runtime import EffectorCommand, TeleopCommand
from teleop_runtime.real_teleop.cli import build_real_teleop_argument_parser
from teleop_runtime.real_teleop.position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP,
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_RUNTIME_HOME_HOLD,
    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_HALF_NORMAL,
    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
    POSITION_ADMITTANCE_SCHEMA_VERSION,
    PositionAdmittanceActiveFault,
    PositionAdmittanceEndpointExcursionGuard,
    PositionAdmittanceShadowTargetFilter,
    admitted_solver_world_poses_from_joint_command,
    install_position_admittance_active,
    install_position_admittance_shadow,
    position_admittance_active_qpos_fast_path_alpha,
    position_admittance_active_uses_home_endpoint_guard,
    position_admittance_shadow_config,
    position_rotational_admittance_config,
    validate_position_admittance_active_config,
    validate_position_rotational_admittance_active_config,
)
from teleop_runtime.real_teleop.runtime_metadata import (
    position_admittance_response_metadata,
)
from teleop_runtime.real_teleop.runtime import (
    _position_admittance_runtime_event_payload,
    run_real_teleop,
)
from teleop_runtime.retargeters.dual_arm_collision_aware import (
    DualArmCollisionAwareRetargeter,
    DualArmTargetContext,
)


def test_active_endpoint_policy_selects_only_the_temporary_home_guard() -> None:
    assert position_admittance_active_uses_home_endpoint_guard(
        POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_RUNTIME_HOME_HOLD
    )
    assert not position_admittance_active_uses_home_endpoint_guard(
        POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP
    )
    with pytest.raises(ValueError, match="active endpoint policy"):
        position_admittance_active_uses_home_endpoint_guard("unknown")


class _Monitor:
    def __init__(self, *, source_time: float = 1.0) -> None:
        self.source_time = source_time
        self.values_by_side = {
            "left": [0.0] * 6,
            "right": [0.0] * 6,
        }
        self.fault = None
        self.snapshot_count = 0

    def snapshot(self) -> dict[str, object]:
        self.snapshot_count += 1
        return {
            "fault": self.fault,
            "last_reading_by_side": {
                side: {
                    "side": side,
                    "source": "tool",
                    "values": list(values),
                    "monotonic_time_sec": self.source_time,
                }
                for side, values in self.values_by_side.items()
            },
        }


def _config() -> CartesianAdmittanceConfig:
    return CartesianAdmittanceConfig(
        enabled_axes=(True, True, True),
        control_frame="solver_world",
        virtual_mass_kg=(1.0, 1.0, 1.0),
        damping_n_s_m=(0.0, 0.0, 0.0),
        stiffness_n_m=(0.0, 0.0, 0.0),
        force_deadband_n=(0.0, 0.0, 0.0),
        force_filter_tau_sec=0.0,
        max_offset_m=(0.01, 0.01, 0.01),
        max_velocity_m_s=(0.1, 0.1, 0.1),
        max_output_slew_m_s=(0.1, 0.1, 0.1),
        stale_after_sec=0.2,
        max_force_dt_sec=0.1,
    )


def _active_cautious_config() -> CartesianAdmittanceConfig:
    return replace(
        _config(),
        response_model="profile_compliance",
        stiffness_n_m=(250.0, 250.0, 250.0),
        force_deadband_n=(0.45, 0.45, 0.45),
        return_force_deadband_n=(0.45, 0.45, 0.45),
        max_offset_m=(0.002, 0.002, 0.002),
        max_velocity_m_s=(0.005, 0.005, 0.005),
        max_return_velocity_m_s=(0.005, 0.005, 0.005),
        max_output_slew_m_s=(0.005, 0.005, 0.005),
        stale_after_sec=0.15,
        max_force_dt_sec=0.15,
    )


def _targets() -> dict[str, np.ndarray]:
    left = np.eye(4)
    right = np.eye(4)
    left[:3, 3] = (-0.4, -0.2, 0.2)
    right[:3, 3] = (-0.4, 0.2, 0.2)
    return {"left": left, "right": right}


def _observer(
    monitor: _Monitor,
    *,
    previous_filter=None,
    side_activity=None,
    apply_to_target: bool = False,
    config: CartesianAdmittanceConfig | None = None,
) -> PositionAdmittanceShadowTargetFilter:
    observer = PositionAdmittanceShadowTargetFilter(
        monitor=monitor,
        active_sides=("left", "right"),
        config=config or _config(),
        previous_filter=previous_filter,
        side_activity=side_activity or (lambda _side: (True, "running")),
        tool_rotation_from_solver_by_side={
            "left": np.eye(3),
            "right": np.eye(3),
        },
        apply_to_target=apply_to_target,
        apply_sides=("left",) if apply_to_target else (),
    )
    observer.reset_from_monitor(now_sec=1.0)
    return observer


def _record_aligned_admitted_feedback(
    observer: PositionAdmittanceShadowTargetFilter,
    result: dict[str, np.ndarray] | object,
    *,
    now_sec: float,
) -> None:
    assert isinstance(result, dict)
    observer.observe_admitted_poses(
        {side: pose.copy() for side, pose in result.items()},
        now_sec=now_sec,
    )


def test_shadow_default_is_off_and_cli_parameters_are_explicit() -> None:
    parser = build_real_teleop_argument_parser()
    args = parser.parse_args([])

    assert args.position_admittance_shadow is False
    assert args.position_admittance_active is False
    config = position_admittance_shadow_config(args)
    assert config.enabled_axes == (True, True, True)
    assert config.control_frame == "solver_world"
    assert config.response_model == "second_order"
    assert config.force_filter_tau_sec == pytest.approx(0.022)
    assert config.stiffness_n_m == pytest.approx((250.0, 250.0, 250.0))


def test_profile_compliance_cli_resolves_without_changing_default_mode() -> None:
    parser = build_real_teleop_argument_parser()
    args = parser.parse_args(
        [
            "--position-admittance-response-model",
            "profile_compliance",
            "--position-admittance-shadow-filter-tau-sec",
            "0.010857362",
            "--position-admittance-shadow-max-return-velocity-m-s",
            "0.03",
        ]
    )

    config = position_admittance_shadow_config(args)
    assert config.response_model == "profile_compliance"
    assert config.force_filter_tau_sec == pytest.approx(0.010857362)
    assert config.max_return_velocity_m_s == pytest.approx((0.03, 0.03, 0.03))
    assert config.return_force_deadband_n == pytest.approx((0.45, 0.45, 0.45))
    assert config.return_rate_s == pytest.approx(1.0)
    assert config.response_reference_hz == pytest.approx(40.0)


def test_public_admittance_weights_and_effective_damping_are_traceable() -> None:
    parser = build_real_teleop_argument_parser()
    args = parser.parse_args(
        [
            "--position-admittance-active",
            "--position-admittance-rotation-active",
            "--position-admittance-combined-active",
            "--position-admittance-rotation-active-profile",
            ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
            "--position-admittance-response-model",
            "profile_compliance",
            "--position-admittance-translation-response-weight",
            "2",
            "--position-admittance-rotation-response-weight",
            "1.5",
            "--position-admittance-translation-filter-tau-sec",
            "0.03",
            "--position-admittance-rotation-filter-tau-sec",
            "0.04",
            "--position-admittance-translation-output-follow-tau-sec",
            "0.025",
            "--position-admittance-rotation-output-follow-tau-sec",
            "0.025",
            "--position-admittance-translation-max-output-acceleration-m-s2",
            "1.2",
            "--position-admittance-rotation-max-output-angular-acceleration-rad-s2",
            "3.0",
        ]
    )

    translation = position_admittance_shadow_config(args)
    rotation = position_rotational_admittance_config(
        args,
        profile=ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
    )
    assert translation.stiffness_n_m == pytest.approx((125.0,) * 3)
    assert translation.force_filter_tau_sec == pytest.approx(0.03)
    assert translation.output_follow_tau_sec == pytest.approx(0.025)
    assert translation.max_output_acceleration_m_s2 == pytest.approx((1.2,) * 3)
    assert rotation.stiffness_nm_rad == pytest.approx((25.0 / 1.5,) * 3)
    assert rotation.torque_filter_tau_sec == pytest.approx(0.04)
    assert rotation.output_follow_tau_sec == pytest.approx(0.025)
    assert rotation.max_output_angular_acceleration_rad_s2 == pytest.approx(
        (3.0,) * 3
    )

    metadata = position_admittance_response_metadata(args)
    assert metadata["modifier_stage"] == "pre_ik_target"
    assert metadata["ik_logic_modified"] is False
    assert metadata["translation"]["response_weight"] == pytest.approx(2.0)
    assert metadata["translation"]["effective_response_weight"] == pytest.approx(
        2.0
    )
    assert metadata["translation"]["declared_weight_matches_effective"] is True
    assert metadata["translation"]["effective_compliance_m_n"] == pytest.approx(
        [0.008] * 3
    )
    assert metadata["translation"]["physical_damping_n_s_m_applied"] is False
    assert metadata["translation"]["output_follow_tau_sec"] == pytest.approx(
        0.025
    )
    assert metadata["translation"]["max_output_acceleration_m_s2"] == pytest.approx(
        [1.2] * 3
    )
    assert metadata["rotation"]["response_weight"] == pytest.approx(1.5)
    assert metadata["rotation"]["effective_compliance_rad_nm"] == pytest.approx(
        [0.06] * 3
    )
    assert metadata["rotation"]["physical_damping_term_applied"] is False
    assert metadata["rotation"]["output_follow_tau_sec"] == pytest.approx(0.025)
    assert metadata["rotation"][
        "max_output_angular_acceleration_rad_s2"
    ] == pytest.approx([3.0] * 3)


def test_second_order_zero_stiffness_metadata_preserves_legacy_damping_model() -> None:
    args = build_real_teleop_argument_parser().parse_args(
        [
            "--position-admittance-shadow",
            "--position-admittance-response-model",
            "second_order",
            "--position-admittance-shadow-stiffness-n-m",
            "0",
            "--position-admittance-shadow-damping-n-s-m",
            "35",
        ]
    )

    metadata = position_admittance_response_metadata(args)
    assert metadata["translation"]["effective_response_weight"] is None
    assert metadata["translation"]["effective_compliance_m_n"] is None
    assert metadata["translation"]["physical_damping_n_s_m_applied"] is True
    assert metadata["translation"]["declared_physical_damping_n_s_m"] == [
        35.0,
        35.0,
        35.0,
    ]

    weighted_second_order = build_real_teleop_argument_parser().parse_args(
        ["--position-admittance-translation-response-weight", "2"]
    )
    with pytest.raises(ValueError, match="requires profile_compliance"):
        position_admittance_shadow_config(weighted_second_order)


def test_accepted_admittance_weight_defaults_are_exactly_behavior_preserving() -> None:
    parser = build_real_teleop_argument_parser()
    args = parser.parse_args(
        [
            "--position-admittance-response-model",
            "profile_compliance",
            "--position-admittance-translation-filter-tau-sec",
            "0.010857362",
        ]
    )

    translation = position_admittance_shadow_config(args)
    rotation = position_rotational_admittance_config(
        args,
        profile=ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
    )
    assert translation.stiffness_n_m == (250.0, 250.0, 250.0)
    assert translation.force_filter_tau_sec == pytest.approx(0.010857362)
    assert translation.output_follow_tau_sec == 0.0
    assert rotation == rotational_admittance_config_for_profile(
        ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG
    )


def test_response_weights_scale_synthetic_wrench_targets_before_hard_bounds() -> None:
    parser = build_real_teleop_argument_parser()
    baseline_args = parser.parse_args(
        [
            "--position-admittance-response-model",
            "profile_compliance",
            "--position-admittance-translation-filter-tau-sec",
            "0",
            "--position-admittance-rotation-filter-tau-sec",
            "0",
        ]
    )
    weighted_args = parser.parse_args(
        [
            "--position-admittance-response-model",
            "profile_compliance",
            "--position-admittance-translation-response-weight",
            "2",
            "--position-admittance-rotation-response-weight",
            "2",
            "--position-admittance-translation-filter-tau-sec",
            "0",
            "--position-admittance-rotation-filter-tau-sec",
            "0",
        ]
    )
    sample_zero = CartesianAdmittanceWrenchSample(
        values_n_nm=(0.0,) * 6,
        frame="solver_world",
        source_time_sec=1.0,
        receive_time_sec=1.001,
        tare_generation=1,
    )
    sample_load = CartesianAdmittanceWrenchSample(
        values_n_nm=(1.45, 0.0, 0.0, 0.0, 0.0, 0.1),
        frame="solver_world",
        source_time_sec=1.05,
        receive_time_sec=1.051,
        tare_generation=1,
    )

    translation_requested = []
    rotation_requested = []
    for args in (baseline_args, weighted_args):
        translation = BoundedCartesianAdmittance(
            replace(
                position_admittance_shadow_config(args),
                max_offset_m=(0.08,) * 3,
                max_velocity_m_s=(1.0,) * 3,
            )
        )
        rotation = BoundedRotationalAdmittance(
            position_rotational_admittance_config(
                args,
                profile=ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
            )
        )
        translation.reset(tare_sample=sample_zero, output_time_sec=1.0)
        rotation.reset(tare_sample=sample_zero, output_time_sec=1.0)
        translation_requested.append(
            translation.step_new_sample(sample_load, now_sec=1.05).requested_offset_m[0]
        )
        rotation_requested.append(
            rotation.step_new_sample(
                sample_load,
                now_sec=1.05,
                rotation_control_from_sample=np.eye(3),
            ).requested_rotvec_rad[2]
        )

    assert translation_requested[1] == pytest.approx(2 * translation_requested[0])
    assert rotation_requested[1] == pytest.approx(2 * rotation_requested[0])


def test_active_response_tuning_is_bounded_without_relaxing_hard_envelopes() -> None:
    translation = _active_cautious_config()
    with pytest.raises(ValueError, match="translation response weight"):
        validate_position_admittance_active_config(
            replace(translation, stiffness_n_m=(250.0 / 4.01,) * 3)
        )
    with pytest.raises(ValueError, match="translation filter tau"):
        validate_position_admittance_active_config(
            replace(translation, force_filter_tau_sec=0.101)
        )

    rotation = rotational_admittance_config_for_profile(
        ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG
    )
    with pytest.raises(ValueError, match="response weight"):
        validate_position_rotational_admittance_active_config(
            replace(rotation, stiffness_nm_rad=(25.0 / 4.01,) * 3),
            active_profile=ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
        )
    with pytest.raises(ValueError, match="filter tau"):
        validate_position_rotational_admittance_active_config(
            replace(rotation, torque_filter_tau_sec=0.101),
            active_profile=ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
        )


def test_shadow_returns_historical_none_and_does_not_mutate_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    observer = _observer(monitor)
    targets = _targets()
    originals = {side: pose.copy() for side, pose in targets.items()}

    monitor.source_time = 1.05
    monitor.values_by_side["left"][0] = 2.0
    clock["now"] = 1.05
    first_result = observer(targets, object(), object())
    monitor.source_time = 1.10
    clock["now"] = 1.10
    result = observer(targets, object(), object())

    assert first_result is None
    assert result is None
    for side in targets:
        np.testing.assert_array_equal(targets[side], originals[side])
    snapshot = observer.snapshot(now_sec=1.10)
    assert snapshot["schema_version"] == POSITION_ADMITTANCE_SCHEMA_VERSION
    assert snapshot["shadow_only"] is True
    assert snapshot["applied_to_target"] is False
    assert snapshot["target_identity_preserved"] is True
    assert snapshot["composition"] == {
        "frame": "solver_world",
        "equation": "p_applied = p_raw + output_offset",
        "transient_terms": ["output_offset_m"],
        "persistent_terms": [],
        "final_modifier_field": "applied_total_offset_m",
        "final_envelope": "max_offset_m_and_max_output_slew_m_s",
    }
    left = snapshot["sides"]["left"]
    assert left["new_sample_integrated"] is True
    assert left["requested_offset_m"][0] > 0.0
    np.testing.assert_allclose(
        left["raw_target_pose_solver_world"],
        originals["left"],
    )
    np.testing.assert_allclose(
        np.asarray(left["would_apply_target_pose_solver_world"])[:3, 3],
        left["would_apply_target_translation"],
    )
    np.testing.assert_allclose(
        left["applied_target_pose_solver_world"],
        originals["left"],
    )
    assert left["would_apply_target_translation"] != originals["left"][:3, 3].tolist()


def test_shadow_preserves_exact_init_override_result_and_bypasses_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: 1.05,
    )
    monitor = _Monitor()
    override = _targets()

    def init_filter(_targets, _observation, _calibration):
        return override

    observer = _observer(monitor, previous_filter=init_filter)
    monitor.source_time = 1.05
    monitor.values_by_side["left"][0] = 5.0
    result = observer(_targets(), object(), object())

    assert result is override
    snapshot = observer.snapshot(now_sec=1.05)
    assert snapshot["sides"]["left"]["activity_reason"] == "init_arm_override"
    assert snapshot["sides"]["left"]["requested_offset_m"] == [0.0, 0.0, 0.0]


def test_active_returns_copied_targets_with_only_the_bounded_translation_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    observer = _observer(monitor, apply_to_target=True)
    targets = _targets()
    originals = {side: target.copy() for side, target in targets.items()}

    monitor.values_by_side["left"][0] = 2.0
    monitor.source_time = 1.05
    clock["now"] = 1.05
    first = observer(targets, object(), object())
    _record_aligned_admitted_feedback(observer, first, now_sec=1.05)
    monitor.source_time = 1.10
    clock["now"] = 1.10
    result = observer(targets, object(), object())

    assert first is not None
    assert result is not None
    for side in targets:
        np.testing.assert_array_equal(targets[side], originals[side])
        assert result[side] is not targets[side]
        np.testing.assert_array_equal(result[side][:3, :3], originals[side][:3, :3])
    assert result["left"][0, 3] == pytest.approx(originals["left"][0, 3] + 0.005)
    np.testing.assert_array_equal(result["right"], originals["right"])
    snapshot = observer.snapshot(now_sec=1.10)
    assert snapshot["mode"] == "active"
    assert snapshot["shadow_only"] is False
    assert snapshot["applied_to_target"] is True
    assert snapshot["active_qpos_fast_path_alpha"] == pytest.approx(0.80)
    assert snapshot["sides"]["left"]["applied_offset_m"][0] == pytest.approx(0.005)
    assert observer.target_activity_override_by_side() == {
        "left": {
            "source": "position_admittance_active_modifier",
            "state": "ACTIVE",
            "inactive": False,
            "hold_weight": 0.0,
            "modifier_active": True,
            "modifier_offset_norm_m": pytest.approx(0.005),
            "modifier_qpos_fast_path_alpha": pytest.approx(0.80),
        }
    }


def test_shadow_never_publishes_target_activity_override() -> None:
    observer = _observer(_Monitor(), apply_to_target=False)

    assert observer.target_activity_override_by_side() == {}


def test_retargeter_projects_modifier_activity_into_owned_source_command() -> None:
    class _Override:
        @staticmethod
        def target_activity_override_by_side():
            return {
                "left": {
                    "source": "position_admittance_active_modifier",
                    "state": "ACTIVE",
                    "inactive": False,
                    "hold_weight": 0.0,
                }
            }

    left = EffectorCommand(
        effector_name="left_arm",
        side="left",
        command_type="cartesian_pose",
        value=np.eye(4),
        metadata={
            "tracker_motion_intent": {
                "state": "INACTIVE",
                "inactive": True,
                "hold_weight": 1.0,
                "evidence": {
                    "linear_speed_m_s": 0.04,
                    "angular_speed_deg_s": 3.0,
                },
                "motion_window": {
                    "quality_valid": True,
                    "linear_speed_m_s": 0.04,
                    "angular_speed_deg_s": 3.0,
                },
            }
        },
    )
    right = EffectorCommand(
        effector_name="right_arm",
        side="right",
        command_type="cartesian_pose",
        value=np.eye(4),
    )
    context = DualArmTargetContext(
        timestamp=1.0,
        left_target=np.eye(4),
        right_target=np.eye(4),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=left,
        right_cartesian=right,
        left_input_skipped=None,
        right_input_skipped=None,
    )
    fake = SimpleNamespace(
        target_override_filter=_Override(),
        _validated_side=lambda side: side,
    )

    DualArmCollisionAwareRetargeter._apply_target_override_activity_metadata(
        fake,
        context,
    )

    assert left.metadata["tracker_motion_intent"]["state"] == "INACTIVE"
    modifier_intent = left.metadata["pose_action_motion_intent"]
    assert modifier_intent == {
        "source": "position_admittance_active_modifier",
        "state": "ACTIVE",
        "inactive": False,
        "hold_weight": 0.0,
        "evidence": {
            "linear_speed_m_s": 0.04,
            "angular_speed_deg_s": 3.0,
        },
        "motion_window": {
            "quality_valid": True,
            "linear_speed_m_s": 0.04,
            "angular_speed_deg_s": 3.0,
        },
    }
    assert modifier_intent["evidence"] is not left.metadata[
        "tracker_motion_intent"
    ]["evidence"]
    assert modifier_intent["motion_window"] is not left.metadata[
        "tracker_motion_intent"
    ]["motion_window"]


def test_admitted_joint_command_feedback_uses_final_solver_world_qref_pair() -> None:
    calls: list[dict[str, np.ndarray]] = []

    def resolve(**kwargs):
        calls.append({side: np.asarray(value).copy() for side, value in kwargs.items()})
        left = np.eye(4)
        right = np.eye(4)
        left[0, 3] = float(kwargs["left_qpos"][0])
        right[0, 3] = float(kwargs["right_qpos"][0])
        return {"left": left, "right": right}

    command = TeleopCommand(timestamp=1.0)
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.arange(6, dtype=float),
        )
    )
    command.add_command(
        EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="joint_vector",
            value=np.arange(6, dtype=float) + 10.0,
        )
    )

    poses = admitted_solver_world_poses_from_joint_command(
        retargeter=SimpleNamespace(solver_world_targets_for_qpos_pair=resolve),
        command=command,
    )

    np.testing.assert_array_equal(calls[0]["left_qpos"], np.arange(6, dtype=float))
    np.testing.assert_array_equal(
        calls[0]["right_qpos"],
        np.arange(6, dtype=float) + 10.0,
    )
    assert poses["left"][0, 3] == pytest.approx(0.0)
    assert poses["right"][0, 3] == pytest.approx(10.0)


def test_active_endpoint_excursion_guard_holds_last_safe_tool_qref() -> None:
    class _Retargeter:
        @staticmethod
        def current_qpos_by_side():
            return {"left": np.zeros(6), "right": np.zeros(6)}

        @staticmethod
        def tool_tcp_world_targets_for_qpos_pair(*, left_qpos, right_qpos):
            left = np.eye(4)
            right = np.eye(4)
            left[0, 3] = float(np.asarray(left_qpos)[0])
            right[0, 3] = float(np.asarray(right_qpos)[0])
            return {"left": left, "right": right}

    def command(left_x: float) -> TeleopCommand:
        result = TeleopCommand(timestamp=1.0)
        for side, first_joint in (("left", left_x), ("right", 0.0)):
            qpos = np.zeros(6)
            qpos[0] = first_joint
            result.add_command(
                EffectorCommand(
                    effector_name=f"{side}_arm",
                    side=side,
                    command_type="joint_vector",
                    value=qpos,
                )
            )
        return result

    guard = PositionAdmittanceEndpointExcursionGuard(
        retargeter=_Retargeter(),
        active_side="left",
        max_tool_excursion_m=0.08,
    )
    with pytest.raises(ValueError, match="must be <= 0.08"):
        PositionAdmittanceEndpointExcursionGuard(
            retargeter=_Retargeter(),
            active_side="left",
            max_tool_excursion_m=0.081,
        )
    unarmed = command(0.20)
    assert guard.filter_command(unarmed, now=1.0) is unarmed
    receipt = guard.reset_anchor_from_qpos()
    assert receipt["armed"] is True
    assert receipt["anchor_tool_translation_m"] == [0.0, 0.0, 0.0]
    safe = command(0.08)
    assert guard.filter_command(safe, now=1.1) is safe

    held = guard.filter_command(command(0.080001), now=1.2)
    assert np.asarray(held.commands["left_arm"].value)[0] == pytest.approx(0.08)
    hold_payload = held.metadata["position_admittance_endpoint_hold"]
    assert hold_payload["code"] == "position_admittance_endpoint_excursion_hold"
    assert hold_payload["candidate_excursion_m"] == pytest.approx(0.080001)
    assert hold_payload["requires_runtime_home"] is True
    snapshot = guard.snapshot()
    assert snapshot["hold_latched"] is True
    assert snapshot["hold_count"] == 1

    still_held = guard.filter_command(command(0.01), now=1.3)
    assert np.asarray(still_held.commands["left_arm"].value)[0] == pytest.approx(
        0.08
    )

    reset = guard.reset_anchor_from_qpos(
        {"left": np.array([0.10, 0, 0, 0, 0, 0]), "right": np.zeros(6)}
    )
    assert reset["hold_latched"] is False
    post_home = command(0.18)
    assert guard.filter_command(post_home, now=1.4) is post_home


def test_active_endpoint_excursion_guard_holds_last_safe_tool_rotation() -> None:
    class _Retargeter:
        @staticmethod
        def current_qpos_by_side():
            return {"left": np.zeros(6), "right": np.zeros(6)}

        @staticmethod
        def tool_tcp_world_targets_for_qpos_pair(*, left_qpos, right_qpos):
            poses = {"left": np.eye(4), "right": np.eye(4)}
            for side, qpos in (("left", left_qpos), ("right", right_qpos)):
                angle = float(np.asarray(qpos)[0])
                poses[side][:3, :3] = np.asarray(
                    [
                        [np.cos(angle), -np.sin(angle), 0.0],
                        [np.sin(angle), np.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
            return poses

    def command(left_rotation_deg: float) -> TeleopCommand:
        result = TeleopCommand(timestamp=1.0)
        for side, rotation_deg in (
            ("left", left_rotation_deg),
            ("right", 0.0),
        ):
            qpos = np.zeros(6)
            qpos[0] = np.deg2rad(rotation_deg)
            result.add_command(
                EffectorCommand(
                    effector_name=f"{side}_arm",
                    side=side,
                    command_type="joint_vector",
                    value=qpos,
                )
            )
        return result

    with pytest.raises(ValueError, match="rotation excursion must be <= 3.5"):
        PositionAdmittanceEndpointExcursionGuard(
            retargeter=_Retargeter(),
            active_side="left",
            max_tool_excursion_m=0.08,
            max_tool_rotation_excursion_deg=3.51,
        )
    guard = PositionAdmittanceEndpointExcursionGuard(
        retargeter=_Retargeter(),
        active_side="left",
        max_tool_excursion_m=0.08,
        max_tool_rotation_excursion_deg=3.5,
    )
    guard.reset_anchor_from_qpos()
    safe = command(3.5)
    assert guard.filter_command(safe, now=1.1) is safe

    held = guard.filter_command(command(3.5001), now=1.2)
    assert np.rad2deg(np.asarray(held.commands["left_arm"].value)[0]) == pytest.approx(
        3.5
    )
    hold_payload = held.metadata["position_admittance_endpoint_hold"]
    assert hold_payload["code"] == (
        "position_admittance_endpoint_rotation_excursion_hold"
    )
    assert hold_payload["candidate_rotation_excursion_deg"] == pytest.approx(
        3.5001
    )

    balanced_guard = PositionAdmittanceEndpointExcursionGuard(
        retargeter=_Retargeter(),
        active_side="left",
        max_tool_excursion_m=0.08,
        max_tool_rotation_excursion_deg=5.5,
        maximum_allowed_tool_rotation_excursion_deg=5.5,
    )
    balanced_guard.reset_anchor_from_qpos()
    balanced_safe = command(5.5)
    assert balanced_guard.filter_command(balanced_safe, now=2.0) is balanced_safe
    balanced_held = balanced_guard.filter_command(command(5.5001), now=2.1)
    assert balanced_held.metadata["position_admittance_endpoint_hold"]["code"] == (
        "position_admittance_endpoint_rotation_excursion_hold"
    )
    with pytest.raises(ValueError, match="maximum allowed.*<= 5.5"):
        PositionAdmittanceEndpointExcursionGuard(
            retargeter=_Retargeter(),
            active_side="left",
            max_tool_excursion_m=0.08,
            max_tool_rotation_excursion_deg=5.5,
            maximum_allowed_tool_rotation_excursion_deg=5.5001,
        )


def test_active_admitted_qref_feedback_is_diagnostic_with_large_nominal_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    observer = _observer(monitor, apply_to_target=True)
    targets = _targets()
    monitor.values_by_side["left"][0] = 2.0
    for now in (1.05, 1.10):
        monitor.source_time = now
        clock["now"] = now
        result = observer(targets, object(), object())
        _record_aligned_admitted_feedback(observer, result, now_sec=now)

    before = observer.snapshot(now_sec=1.10)["sides"]["left"]
    admitted = {side: pose.copy() for side, pose in targets.items()}
    admitted["left"][:3, 3] += (-0.140, 0.011, -0.002)
    observer.observe_admitted_poses(admitted, now_sec=1.10)
    left = observer.snapshot(now_sec=1.10)["sides"]["left"]

    assert left["output_offset_m"] == pytest.approx(before["output_offset_m"])
    assert left["anchor_offset_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert left["applied_total_offset_m"] == pytest.approx(
        before["applied_total_offset_m"]
    )
    feedback = left["admission_feedback"]
    assert feedback["active_axes"] == ["x"]
    assert feedback["antiwindup_applied"] is False
    assert feedback["state_update_applied"] is False
    assert feedback["reason"] == "diagnostic_only_no_state_update"
    assert feedback["raw_realized_offset_m"] == pytest.approx(
        [-0.140, 0.011, -0.002]
    )
    assert feedback["anchor_correction_m"] == [0.0, 0.0, 0.0]


def test_active_profile_compliance_feedback_preserves_the_60hz_slew_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    config = replace(
        _config(),
        response_model="profile_compliance",
        stiffness_n_m=(250.0, 250.0, 250.0),
        force_deadband_n=(0.45, 0.45, 0.45),
        return_force_deadband_n=(0.45, 0.45, 0.45),
        max_offset_m=(0.005, 0.005, 0.005),
        max_velocity_m_s=(0.01, 0.01, 0.01),
        max_return_velocity_m_s=(0.01, 0.01, 0.01),
        max_output_slew_m_s=(0.01, 0.01, 0.01),
    )
    observer = _observer(
        monitor,
        apply_to_target=True,
        config=config,
    )
    targets = _targets()

    for now in (1.05, 1.0667, 1.0834):
        clock["now"] = now
        monitor.source_time = 1.05
        result = observer(targets, object(), object())
        _record_aligned_admitted_feedback(observer, result, now_sec=now)

    monitor.values_by_side["left"][0] = 5.0
    monitor.source_time = 1.10
    clock["now"] = 1.10
    first = observer(targets, object(), object())
    assert first is not None
    first_offset = first["left"][0, 3] - targets["left"][0, 3]
    observer.observe_admitted_poses(first, now_sec=1.101)
    feedback = observer.snapshot(now_sec=1.101)["sides"]["left"]

    clock["now"] = 1.1167
    second = observer(targets, object(), object())
    assert second is not None
    second_offset = second["left"][0, 3] - targets["left"][0, 3]

    assert first_offset == pytest.approx(0.000166, abs=2e-6)
    assert feedback["output_offset_m"][0] == pytest.approx(first_offset)
    assert feedback["requested_offset_m"][0] == pytest.approx(0.0005)
    assert second_offset > first_offset
    assert second_offset <= 0.0005


def test_active_contact_release_returns_transient_without_persistent_rebase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    config = replace(
        _config(),
        response_model="profile_compliance",
        stiffness_n_m=(250.0, 250.0, 250.0),
        force_deadband_n=(0.45, 0.45, 0.45),
        return_force_deadband_n=(0.45, 0.45, 0.45),
        max_offset_m=(0.005, 0.005, 0.005),
        max_velocity_m_s=(0.01, 0.01, 0.01),
        max_return_velocity_m_s=(0.01, 0.01, 0.01),
        max_output_slew_m_s=(0.01, 0.01, 0.01),
    )
    observer = _observer(monitor, apply_to_target=True, config=config)
    targets = _targets()
    monitor.values_by_side["left"][0] = 5.0
    for now in (1.05, 1.10):
        monitor.source_time = now
        clock["now"] = now
        result = observer(targets, object(), object())
        _record_aligned_admitted_feedback(observer, result, now_sec=now)

    before = observer.snapshot(now_sec=1.10)["sides"]["left"]
    assert before["output_offset_m"][0] > 0.0

    monitor.values_by_side["left"][0] = 0.0
    monitor.source_time = 1.15
    clock["now"] = 1.15
    released = observer(targets, object(), object())

    assert released is not None
    left = observer.snapshot(now_sec=1.15)["sides"]["left"]
    assert 0.0 <= left["output_offset_m"][0] < before["output_offset_m"][0]
    assert released["left"][0, 3] == pytest.approx(
        targets["left"][0, 3] + left["output_offset_m"][0]
    )
    assert left["anchor_offset_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert left["contact_active"] is False
    assert left["contact_rebase_count"] == 0
    assert left["last_contact_rebase"] is None


def test_active_final_envelope_admits_only_bounded_motion_toward_zero_at_return_rate(
) -> None:
    monitor = _Monitor()
    config = replace(
        _active_cautious_config(),
        max_offset_m=(0.02,) * 3,
        max_velocity_m_s=(0.06,) * 3,
        max_return_velocity_m_s=(0.12,) * 3,
        max_output_slew_m_s=(0.06,) * 3,
        return_rate_s=100.0,
    )
    observer = _observer(monitor, apply_to_target=True, config=config)
    controller = observer.controllers["left"]
    controller.rebase_realized_offset(
        (0.01, 0.0, 0.0),
        output_time_sec=1.0,
    )
    observer._last_composed_offset_by_side["left"] = np.asarray(
        (0.01, 0.0, 0.0),
        dtype=float,
    )
    observer._last_composed_time_by_side["left"] = 1.0
    returned = controller.step_new_sample(
        CartesianAdmittanceWrenchSample(
            values_n_nm=(0.0,) * 6,
            frame="tool",
            source_time_sec=1.05,
            receive_time_sec=1.05,
            tare_generation=1,
        ),
        now_sec=1.05,
        rotation_control_from_sample=np.eye(3),
    )
    returned = controller.sample_output(now_sec=1.05)

    admitted = observer._checked_final_modifier(
        side="left",
        state=returned,
        now_sec=1.05,
    )
    assert admitted == pytest.approx((0.004, 0.0, 0.0))

    # The faster limit is never available for an outward step, even if a
    # malformed caller marks the snapshot as passive return.
    malformed = replace(
        returned,
        output_offset_m=(0.011, 0.0, 0.0),
        passive_return_active=True,
    )
    observer._last_composed_offset_by_side["left"] = np.asarray(
        (0.004, 0.0, 0.0),
        dtype=float,
    )
    observer._last_composed_time_by_side["left"] = 1.05
    with pytest.raises(
        PositionAdmittanceActiveFault,
        match="component slew limit",
    ):
        observer._checked_final_modifier(
            side="left",
            state=malformed,
            now_sec=1.10,
        )


def test_active_incident_residual_is_rejected_before_nonzero_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    config = replace(
        _config(),
        response_model="profile_compliance",
        stiffness_n_m=(250.0, 250.0, 250.0),
        force_deadband_n=(0.45, 0.45, 0.45),
        return_force_deadband_n=(0.45, 0.45, 0.45),
        max_offset_m=(0.002, 0.002, 0.002),
        max_velocity_m_s=(0.005, 0.005, 0.005),
        max_return_velocity_m_s=(0.005, 0.005, 0.005),
        max_output_slew_m_s=(0.005, 0.005, 0.005),
    )
    observer = _observer(monitor, apply_to_target=True, config=config)
    targets = _targets()
    clock["now"] = 1.05
    monitor.source_time = 1.05
    first = observer(targets, object(), object())
    assert first is not None
    admitted = {side: pose.copy() for side, pose in first.items()}
    admitted["left"][:3, 3] += (-0.140, 0.011, -0.002)
    observer.observe_admitted_poses(admitted, now_sec=1.05)

    monitor.values_by_side["left"][0] = 1.0
    monitor.source_time = 1.10
    clock["now"] = 1.10
    with pytest.raises(
        PositionAdmittanceActiveFault,
        match="nominal target-to-admitted translation residual",
    ):
        observer(targets, object(), object())

    left = observer.snapshot(now_sec=1.10)["sides"]["left"]
    assert left["anchor_offset_m"] == [0.0, 0.0, 0.0]
    assert left["admission_feedback_fault"] is not None


def test_active_nominal_rotation_residual_is_rejected_before_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.05}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor(source_time=1.05)
    observer = _observer(monitor, apply_to_target=True)
    targets = _targets()
    first = observer(targets, object(), object())
    assert first is not None
    angle = np.deg2rad(10.0)
    rotation_z = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    admitted = {side: pose.copy() for side, pose in first.items()}
    admitted["left"][:3, :3] = rotation_z
    observer.observe_admitted_poses(admitted, now_sec=1.05)

    monitor.values_by_side["left"][0] = 2.0
    monitor.source_time = 1.10
    clock["now"] = 1.10
    with pytest.raises(
        PositionAdmittanceActiveFault,
        match="nominal target-to-admitted rotation residual",
    ):
        observer(targets, object(), object())


def test_active_stale_admitted_baseline_is_rejected_before_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.05}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor(source_time=1.05)
    observer = _observer(monitor, apply_to_target=True)
    targets = _targets()
    first = observer(targets, object(), object())
    _record_aligned_admitted_feedback(observer, first, now_sec=1.05)

    for now in (1.10, 1.15, 1.20):
        clock["now"] = now
        monitor.source_time = now
        result = observer(targets, object(), object())
        assert result is not None
        np.testing.assert_array_equal(result["left"], targets["left"])

    clock["now"] = 1.26
    monitor.source_time = 1.26
    monitor.values_by_side["left"][0] = 2.0
    with pytest.raises(
        PositionAdmittanceActiveFault,
        match="admitted-qref baseline is stale",
    ):
        observer(targets, object(), object())


def test_active_composed_target_allows_sum_of_nominal_and_modifier_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.05}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor(source_time=1.05)
    config = replace(
        _config(),
        max_offset_m=(0.002, 0.002, 0.002),
        max_output_slew_m_s=(0.1, 0.1, 0.1),
    )
    observer = _observer(monitor, apply_to_target=True, config=config)
    targets = _targets()
    first = observer(targets, object(), object())
    assert first is not None
    admitted = {side: pose.copy() for side, pose in first.items()}
    admitted["left"][0, 3] -= 0.0015
    observer.observe_admitted_poses(admitted, now_sec=clock["now"])

    controller = observer.controllers["left"]
    original_sample_output = controller.sample_output

    def offset_sample_output(*, now_sec: float, active: bool = True):
        state = original_sample_output(now_sec=now_sec, active=active)
        return replace(
            state,
            output_offset_m=(0.001, 0.0, 0.0),
            frozen=False,
        )

    monkeypatch.setattr(controller, "sample_output", offset_sample_output)
    clock["now"] = 1.10
    monitor.source_time = 1.10
    result = observer(targets, object(), object())

    assert result is not None
    assert result["left"][0, 3] == pytest.approx(targets["left"][0, 3] + 0.001)


def test_active_nominal_alignment_removes_previous_translation_modifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.05}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor(source_time=1.05)
    config = replace(
        _config(),
        max_offset_m=(0.002, 0.002, 0.002),
        max_output_slew_m_s=(0.1, 0.1, 0.1),
    )
    observer = _observer(monitor, apply_to_target=True, config=config)
    targets = _targets()
    initial = observer(targets, object(), object())
    assert initial is not None
    observer.observe_admitted_poses(initial, now_sec=clock["now"])

    controller = observer.controllers["left"]
    original_sample_output = controller.sample_output

    def saturated_sample_output(*, now_sec: float, active: bool = True):
        state = original_sample_output(now_sec=now_sec, active=active)
        return replace(
            state,
            output_offset_m=(0.002, 0.0, 0.0),
            frozen=False,
        )

    monkeypatch.setattr(controller, "sample_output", saturated_sample_output)
    clock["now"] = 1.10
    monitor.source_time = 1.10
    saturated = observer(targets, object(), object())
    assert saturated is not None
    admitted = {side: pose.copy() for side, pose in saturated.items()}
    admitted["left"][0, 3] += 0.0007
    observer.observe_admitted_poses(admitted, now_sec=clock["now"])

    clock["now"] = 1.15
    monitor.source_time = 1.15
    result = observer(targets, object(), object())

    assert result is not None
    assert result["left"][0, 3] == pytest.approx(
        targets["left"][0, 3] + 0.002
    )


def test_active_long_horizon_aligned_closed_loop_stays_inside_total_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    config = replace(
        _config(),
        response_model="profile_compliance",
        stiffness_n_m=(250.0, 250.0, 250.0),
        force_deadband_n=(0.45, 0.45, 0.45),
        return_force_deadband_n=(0.45, 0.45, 0.45),
        max_offset_m=(0.002, 0.002, 0.002),
        max_velocity_m_s=(0.005, 0.005, 0.005),
        max_return_velocity_m_s=(0.005, 0.005, 0.005),
        max_output_slew_m_s=(0.005, 0.005, 0.005),
    )
    observer = _observer(monitor, apply_to_target=True, config=config)
    targets = _targets()
    previous_offset = np.zeros(3, dtype=float)

    for tick in range(240):
        clock["now"] = 1.0 + (tick + 1) / 60.0
        if tick % 3 == 0:
            monitor.source_time = clock["now"]
            monitor.values_by_side["left"][0] = (
                1.0 if (tick // 12) % 2 == 0 else 0.0
            )
        result = observer(targets, object(), object())
        assert result is not None
        offset = result["left"][:3, 3] - targets["left"][:3, 3]
        assert np.linalg.norm(offset) <= 0.002 + 1e-9
        assert np.linalg.norm(offset - previous_offset) <= (
            0.005 / 60.0 + 1e-9
        )
        previous_offset = offset
        _record_aligned_admitted_feedback(
            observer,
            result,
            now_sec=clock["now"],
        )
        left = observer.snapshot(now_sec=clock["now"])["sides"]["left"]
        assert left["anchor_offset_m"] == [0.0, 0.0, 0.0]
        assert left["applied_total_offset_m"] == pytest.approx(offset)
        assert left["admission_feedback"]["state_update_applied"] is False


def test_active_20_60hz_closed_loop_with_admitted_pose_lag_stays_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    config = replace(
        _config(),
        response_model="profile_compliance",
        stiffness_n_m=(250.0, 250.0, 250.0),
        force_deadband_n=(0.45, 0.45, 0.45),
        return_force_deadband_n=(0.45, 0.45, 0.45),
        max_offset_m=(0.002, 0.002, 0.002),
        max_velocity_m_s=(0.005, 0.005, 0.005),
        max_return_velocity_m_s=(0.005, 0.005, 0.005),
        max_output_slew_m_s=(0.005, 0.005, 0.005),
    )
    observer = _observer(monitor, apply_to_target=True, config=config)
    targets = _targets()
    clock["now"] = 1.05
    monitor.source_time = 1.05
    first = observer(targets, object(), object())
    assert first is not None
    admitted = {side: pose.copy() for side, pose in first.items()}
    observer.observe_admitted_poses(admitted, now_sec=clock["now"])
    previous_modifier = np.zeros(3, dtype=float)
    previous_admitted = admitted["left"][:3, 3].copy()

    for tick in range(630):
        clock["now"] = 1.05 + (tick + 1) / 60.0
        if tick % 3 == 0:
            monitor.source_time = clock["now"]
            if 60 <= tick < 150:
                force_x = 1.2
            elif 210 <= tick < 270:
                force_x = -1.2
            else:
                force_x = 0.0
            monitor.values_by_side["left"][0] = force_x

        result = observer(targets, object(), object())
        assert result is not None
        modifier = result["left"][:3, 3] - targets["left"][:3, 3]
        assert np.linalg.norm(modifier) <= 0.002 + 1e-9
        assert np.linalg.norm(modifier - previous_modifier) <= (
            0.005 / 60.0 + 1e-9
        )
        previous_modifier = modifier

        command_error = result["left"][:3, 3] - admitted["left"][:3, 3]
        command_error_norm = float(np.linalg.norm(command_error))
        plant_step_limit = 0.003 / 60.0
        if command_error_norm > plant_step_limit:
            command_error *= plant_step_limit / command_error_norm
        admitted["left"][:3, 3] += command_error
        admitted_step = admitted["left"][:3, 3] - previous_admitted
        assert np.linalg.norm(admitted_step) <= plant_step_limit + 1e-12
        assert np.linalg.norm(
            admitted["left"][:3, 3] - targets["left"][:3, 3]
        ) <= 0.002 + 1e-9
        previous_admitted = admitted["left"][:3, 3].copy()
        observer.observe_admitted_poses(admitted, now_sec=clock["now"])

        left = observer.snapshot(now_sec=clock["now"])["sides"]["left"]
        assert left["anchor_offset_m"] == [0.0, 0.0, 0.0]
        assert left["admission_feedback"]["state_update_applied"] is False

    # profile_compliance returns exponentially at return_rate_s=1.0; it does
    # not promise a finite-time exact zero.  Six seconds of release must leave
    # only a sub-10 um tail, with the lagged admitted pose caught up to it.
    assert np.linalg.norm(previous_modifier) <= 1e-5
    assert np.linalg.norm(
        admitted["left"][:3, 3] - targets["left"][:3, 3]
    ) <= 1e-5


def test_active_final_composed_envelope_violation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.05}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor(source_time=1.05)
    observer = _observer(monitor, apply_to_target=True)
    first = observer(_targets(), object(), object())
    _record_aligned_admitted_feedback(observer, first, now_sec=clock["now"])
    controller = observer.controllers["left"]
    original_sample_output = controller.sample_output

    def unsafe_sample_output(*, now_sec: float, active: bool = True):
        state = original_sample_output(now_sec=now_sec, active=active)
        return replace(
            state,
            output_offset_m=(0.009, 0.009, 0.0),
            frozen=False,
        )

    monkeypatch.setattr(controller, "sample_output", unsafe_sample_output)
    clock["now"] = 1.10
    monitor.source_time = 1.10

    with pytest.raises(
        PositionAdmittanceActiveFault,
        match="total translation limit",
    ):
        observer(_targets(), object(), object())

    snapshot = observer.snapshot(now_sec=clock["now"])
    assert snapshot["last_error"] is not None
    assert snapshot["sides"]["left"]["admission_feedback_fault"] is not None


def test_active_final_composed_slew_violation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.05}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor(source_time=1.05)
    observer = _observer(monitor, apply_to_target=True)
    first = observer(_targets(), object(), object())
    _record_aligned_admitted_feedback(observer, first, now_sec=clock["now"])

    controller = observer.controllers["left"]
    original_sample_output = controller.sample_output

    def unsafe_sample_output(*, now_sec: float, active: bool = True):
        state = original_sample_output(now_sec=now_sec, active=active)
        return replace(
            state,
            output_offset_m=(0.005, 0.0, 0.0),
            frozen=False,
        )

    monkeypatch.setattr(controller, "sample_output", unsafe_sample_output)
    clock["now"] = 1.06
    monitor.source_time = 1.06

    with pytest.raises(
        PositionAdmittanceActiveFault,
        match="component slew limit",
    ):
        observer(_targets(), object(), object())


def test_active_admission_feedback_fault_stays_latched_until_home_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.05}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    observer = _observer(monitor, apply_to_target=True)
    observer.latch_admission_feedback_fault("missing final qref")

    monitor.source_time = 1.05
    result = observer(_targets(), object(), object())
    assert result is not None
    left = observer.snapshot(now_sec=1.05)["sides"]["left"]
    assert left["active"] is False
    assert left["activity_reason"] == "admission_feedback_fault"
    assert left["admission_feedback_fault"] == "missing final qref"

    monitor.source_time = 1.10
    observer.reset_from_monitor(now_sec=1.10)
    assert observer.snapshot(now_sec=1.10)["sides"]["left"][
        "admission_feedback_fault"
    ] is None


def test_active_never_overlays_an_authoritative_init_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: 1.05,
    )
    monitor = _Monitor()
    override = _targets()
    observer = _observer(
        monitor,
        previous_filter=lambda *_args: override,
        apply_to_target=True,
    )
    monitor.source_time = 1.05
    monitor.values_by_side["left"][0] = 5.0

    result = observer(_targets(), object(), object())

    assert result is override
    assert observer.snapshot(now_sec=1.05)["sides"]["left"][
        "applied_offset_m"
    ] == [0.0, 0.0, 0.0]


def test_active_removes_overlay_while_inactive_and_does_not_reapply_old_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    active = {"left": True, "right": True}
    observer = _observer(
        monitor,
        side_activity=lambda side: (active[side], "running" if active[side] else "fixed_side"),
        apply_to_target=True,
    )
    targets = _targets()
    monitor.values_by_side["left"][0] = 2.0
    for now in (1.05, 1.10):
        monitor.source_time = now
        clock["now"] = now
        result = observer(targets, object(), object())
        _record_aligned_admitted_feedback(observer, result, now_sec=now)
    assert result is not None
    assert result["left"][0, 3] > targets["left"][0, 3]

    active["left"] = False
    monitor.source_time = 1.15
    clock["now"] = 1.15
    fixed_result = observer(targets, object(), object())
    assert fixed_result is not None
    np.testing.assert_array_equal(fixed_result["left"], targets["left"])

    active["left"] = True
    monitor.source_time = 1.20
    clock["now"] = 1.20
    resumed_result = observer(targets, object(), object())
    assert resumed_result is not None
    np.testing.assert_array_equal(resumed_result["left"], targets["left"])
    left = observer.snapshot(now_sec=1.20)["sides"]["left"]
    assert left["state_reason"] in {"duplicate_output_time", "sample_clock_rearmed"}
    assert left["applied_offset_m"] == [0.0, 0.0, 0.0]


def test_known_control_plane_pause_rearms_force_clock_without_resetting_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: clock["now"],
    )
    monitor = _Monitor()
    observer = _observer(
        monitor,
        apply_to_target=True,
        config=_active_cautious_config(),
    )
    targets = _targets()
    monitor.values_by_side["left"][0] = 5.0
    monitor.source_time = 1.05
    clock["now"] = 1.05
    first = observer(targets, object(), object())
    _record_aligned_admitted_feedback(observer, first, now_sec=1.05)
    monitor.source_time = 1.10
    clock["now"] = 1.10
    second = observer(targets, object(), object())
    _record_aligned_admitted_feedback(observer, second, now_sec=1.10)
    before = observer.snapshot(now_sec=1.10)["sides"]["left"]
    assert np.linalg.norm(before["applied_offset_m"]) > 0.0

    # A writer/quiesce transition can block the control thread for much longer
    # than max_force_dt_sec.  The explicit transition marker clears only the
    # sample clocks and must retain the already admitted modifier.
    monitor.source_time = 1.70
    clock["now"] = 1.70
    rearm = observer.rearm_after_control_plane_pause(
        now_sec=1.70,
        pause_duration_sec=0.60,
        reason="formal_episode_start",
    )
    assert rearm["sides"] == ["left", "right"]

    resumed = observer(targets, object(), object())
    left = observer.snapshot(now_sec=1.70)["sides"]["left"]
    assert left["sample_state_reason"] == "sample_clock_rearmed"
    assert left["applied_offset_m"] == pytest.approx(before["applied_offset_m"])
    assert left["admission_feedback_fault"] is None
    assert isinstance(resumed, dict)


def test_active_observer_failure_fails_closed_instead_of_dispatching_base_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: 1.05,
    )
    monitor = _Monitor()
    observer = _observer(monitor, apply_to_target=True)
    targets = _targets()

    def broken_snapshot():
        raise RuntimeError("read snapshot failed")

    monitor.snapshot = broken_snapshot  # type: ignore[method-assign]
    with pytest.raises(PositionAdmittanceActiveFault, match="read snapshot failed"):
        observer(targets, object(), object())

    assert observer.snapshot(now_sec=1.05)["last_error"] == (
        "RuntimeError: read snapshot failed"
    )
    assert observer.snapshot(now_sec=1.05)["sides"]["left"][
        "admission_feedback_fault"
    ] is not None


def test_fixed_side_is_frozen_while_live_side_remains_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: 1.05,
    )
    monitor = _Monitor()
    observer = _observer(
        monitor,
        side_activity=lambda side: (
            (False, "fixed_side") if side == "right" else (True, "running")
        ),
    )
    monitor.source_time = 1.05
    monitor.values_by_side["left"][0] = 2.0
    monitor.values_by_side["right"][0] = 2.0
    observer(_targets(), object(), object())
    monitor.source_time = 1.10
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: 1.10,
    )
    observer(_targets(), object(), object())
    snapshot = observer.snapshot(now_sec=1.10)

    assert snapshot["sides"]["left"]["active"] is True
    assert snapshot["sides"]["left"]["new_sample_integrated"] is True
    assert snapshot["sides"]["right"]["active"] is False
    assert snapshot["sides"]["right"]["state_reason"] == "fixed_side"


def test_monitor_error_is_isolated_and_previous_filter_result_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "teleop_runtime.real_teleop.position_admittance_shadow.time.monotonic",
        lambda: 1.05,
    )
    monitor = _Monitor()
    override = _targets()
    observer = _observer(
        monitor,
        previous_filter=lambda *_args: override,
    )

    def broken_snapshot():
        raise RuntimeError("read snapshot failed")

    monitor.snapshot = broken_snapshot  # type: ignore[method-assign]
    result = observer(_targets(), object(), object())

    assert result is override
    assert observer.snapshot(now_sec=1.05)["last_error"] == (
        "RuntimeError: read snapshot failed"
    )


def test_selected_side_home_reset_only_advances_that_tare_generation() -> None:
    monitor = _Monitor()
    observer = _observer(monitor)
    before = observer.snapshot(now_sec=1.0)

    monitor.source_time = 2.0
    observer.reset_from_monitor(sides=("left",), now_sec=2.0)
    after = observer.snapshot(now_sec=2.0)

    assert before["sides"]["left"]["tare_generation"] == 1
    assert before["sides"]["right"]["tare_generation"] == 1
    assert after["sides"]["left"]["tare_generation"] == 2
    assert after["sides"]["right"]["tare_generation"] == 1


def test_install_is_atomic_and_composes_existing_filter() -> None:
    monitor = _Monitor()
    previous = lambda *_args: None
    retargeter = SimpleNamespace(
        target_override_filter=previous,
        _tool_tcp_from_solver={
            "left": np.eye(4),
            "right": np.eye(4),
        },
    )
    observer = install_position_admittance_shadow(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=("left", "right"),
        config=_config(),
        side_activity=lambda _side: (True, "running"),
        now_sec=1.0,
    )

    assert retargeter.target_override_filter is observer
    assert observer.previous_filter is previous


def test_active_install_is_explicit_and_keeps_the_previous_filter() -> None:
    monitor = _Monitor()
    previous = lambda *_args: None
    retargeter = SimpleNamespace(
        target_override_filter=previous,
        _tool_tcp_from_solver={
            "left": np.eye(4),
            "right": np.eye(4),
        },
    )

    observer = install_position_admittance_active(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=("left", "right"),
        config=_active_cautious_config(),
        side_activity=lambda _side: (True, "running"),
        apply_sides=("left",),
        now_sec=1.0,
    )

    assert retargeter.target_override_filter is observer
    assert observer.previous_filter is previous
    assert observer.snapshot(now_sec=1.0)["mode"] == "active"
    assert observer.snapshot(now_sec=1.0)["apply_sides"] == ["left"]
    assert observer.snapshot(now_sec=1.0)["active_envelope"] == "cautious"


def test_active_half_normal_requires_its_explicit_named_envelope() -> None:
    half_normal_config = replace(
        _active_cautious_config(),
        max_offset_m=(0.04,) * 3,
        max_velocity_m_s=(0.03,) * 3,
        max_return_velocity_m_s=(0.015,) * 3,
        max_output_slew_m_s=(0.03,) * 3,
    )

    with pytest.raises(ValueError, match="max_offset_m must be <= 0.002"):
        validate_position_admittance_active_config(half_normal_config)

    with pytest.raises(
        ValueError,
        match="requires exact max_return_velocity_m_s=0.06",
    ):
        validate_position_admittance_active_config(
            half_normal_config,
            active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_HALF_NORMAL,
        )
    release_braked_half = replace(
        half_normal_config,
        max_return_velocity_m_s=(0.06,) * 3,
        return_rate_s=2.0,
    )
    assert validate_position_admittance_active_config(
        release_braked_half,
        active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_HALF_NORMAL,
    ) is release_braked_half

    full_config = replace(
        half_normal_config,
        max_offset_m=(0.08,) * 3,
        max_velocity_m_s=(0.06,) * 3,
        max_return_velocity_m_s=(0.03,) * 3,
        max_output_slew_m_s=(0.06,) * 3,
    )
    with pytest.raises(ValueError, match="max_offset_m must be <= 0.04"):
        validate_position_admittance_active_config(
            full_config,
            active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_HALF_NORMAL,
        )
    with pytest.raises(
        ValueError,
        match="requires exact max_return_velocity_m_s=0.42",
    ):
        validate_position_admittance_active_config(
            full_config,
            active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
        )


def test_normal_parity_allows_only_bounded_return_braking_parameters() -> None:
    release_braked = replace(
        _active_cautious_config(),
        max_offset_m=(0.08,) * 3,
        max_velocity_m_s=(0.06,) * 3,
        max_return_velocity_m_s=(0.42,) * 3,
        max_output_slew_m_s=(0.06,) * 3,
        return_rate_s=4.0,
    )

    assert validate_position_admittance_active_config(
        release_braked,
        active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
    ) is release_braked
    with pytest.raises(ValueError, match="max_return_velocity_m_s must be <= 0.42"):
        validate_position_admittance_active_config(
            replace(release_braked, max_return_velocity_m_s=(0.4201,) * 3),
            active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
        )
    with pytest.raises(ValueError, match="requires exact return_rate_s=4"):
        validate_position_admittance_active_config(
            replace(release_braked, return_rate_s=4.01),
            active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
        )


def test_active_install_rejects_the_normal_shadow_parity_envelope() -> None:
    monitor = _Monitor()
    previous = lambda *_args: None
    retargeter = SimpleNamespace(
        target_override_filter=previous,
        _tool_tcp_from_solver={
            "left": np.eye(4),
            "right": np.eye(4),
        },
    )
    normal_shadow_config = replace(
        _active_cautious_config(),
        max_offset_m=(0.08, 0.08, 0.08),
        max_velocity_m_s=(0.06, 0.06, 0.06),
        max_return_velocity_m_s=(0.03, 0.03, 0.03),
        max_output_slew_m_s=(0.06, 0.06, 0.06),
    )

    with pytest.raises(ValueError, match="max_offset_m must be <= 0.002"):
        install_position_admittance_active(
            retargeter=retargeter,
            monitor=monitor,
            active_sides=("left", "right"),
            config=normal_shadow_config,
            side_activity=lambda _side: (True, "running"),
            apply_sides=("left",),
            now_sec=1.0,
        )

    assert retargeter.target_override_filter is previous


def test_active_install_rejects_the_default_second_order_model() -> None:
    monitor = _Monitor()
    previous = lambda *_args: None
    retargeter = SimpleNamespace(
        target_override_filter=previous,
        _tool_tcp_from_solver={
            "left": np.eye(4),
            "right": np.eye(4),
        },
    )

    with pytest.raises(ValueError, match="requires profile_compliance"):
        install_position_admittance_active(
            retargeter=retargeter,
            monitor=monitor,
            active_sides=("left", "right"),
            config=replace(
                _active_cautious_config(),
                response_model="second_order",
            ),
            side_activity=lambda _side: (True, "running"),
            apply_sides=("left",),
            now_sec=1.0,
        )

    assert retargeter.target_override_filter is previous


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_offset_m", (0.0021,) * 3),
        ("max_velocity_m_s", (0.0051,) * 3),
        ("max_return_velocity_m_s", (0.0051,) * 3),
        ("max_output_slew_m_s", (0.0051,) * 3),
        ("stale_after_sec", 0.151),
        ("max_force_dt_sec", 0.151),
    ),
)
def test_active_config_rejects_each_cautious_envelope_overrun(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        validate_position_admittance_active_config(
            replace(_active_cautious_config(), **{field: value})
        )


def test_active_output_follow_tau_is_explicitly_bounded_for_both_modifiers() -> None:
    translation = replace(
        _active_cautious_config(),
        output_follow_tau_sec=0.025,
        max_output_acceleration_m_s2=(1.2,) * 3,
    )
    rotation = replace(
        rotational_admittance_config_for_profile(
            ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG
        ),
        output_follow_tau_sec=0.025,
        max_output_angular_acceleration_rad_s2=(3.0,) * 3,
    )

    assert validate_position_admittance_active_config(translation) is translation
    assert (
        validate_position_rotational_admittance_active_config(
            rotation,
            active_profile=(
                ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG
            ),
        )
        is rotation
    )
    with pytest.raises(ValueError, match="output-follow tau"):
        validate_position_admittance_active_config(
            replace(translation, output_follow_tau_sec=0.1001)
        )
    with pytest.raises(ValueError, match="output-follow tau"):
        validate_position_rotational_admittance_active_config(
            replace(rotation, output_follow_tau_sec=0.1001),
            active_profile=(
                ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG
            ),
        )


@pytest.mark.parametrize(
    ("max_slew_m_s", "expected_alpha"),
    ((0.005, 0.85), (0.03, 0.85), (0.06, 0.80)),
)
def test_active_qpos_fast_path_schedule_is_explicit_by_modifier_slew(
    max_slew_m_s: float,
    expected_alpha: float,
) -> None:
    config = replace(
        _active_cautious_config(),
        max_output_slew_m_s=(max_slew_m_s,) * 3,
    )

    assert position_admittance_active_qpos_fast_path_alpha(config) == pytest.approx(
        expected_alpha
    )


def test_direct_runtime_active_overrun_is_rejected_before_startup() -> None:
    args = build_real_teleop_argument_parser().parse_args(
        [
            "--position-admittance-active",
            "--position-admittance-active-side",
            "left",
            "--position-admittance-response-model",
            "profile_compliance",
            "--position-admittance-shadow-max-offset-m",
            "0.0021",
            "--position-admittance-shadow-max-velocity-m-s",
            "0.005",
            "--position-admittance-shadow-max-return-velocity-m-s",
            "0.005",
            "--position-admittance-shadow-max-slew-m-s",
            "0.005",
            "--position-admittance-shadow-stale-sec",
            "0.15",
            "--position-admittance-shadow-max-force-dt-sec",
            "0.15",
        ]
    )
    with pytest.raises(
        RuntimeError,
        match="max_offset_m must be <= 0.002",
    ):
        run_real_teleop(args)


def test_runtime_startup_receipts_distinguish_enabled_and_disabled() -> None:
    disabled = _position_admittance_runtime_event_payload(
        requested_mode="shadow",
        enabled=False,
        active_sides=["left"],
        coupled_pre_ik_target_filter_available=False,
        reason="the coupled pre-IK target filter is unavailable",
    )
    assert disabled == {
        "runtime_schema": "teleop_runtime.position_admittance_startup/v1",
        "enabled": False,
        "requested_mode": "shadow",
        "active_sides": ["left"],
        "coupled_pre_ik_target_filter_available": False,
        "reason": "the coupled pre-IK target filter is unavailable",
    }

    enabled = _position_admittance_runtime_event_payload(
        requested_mode="active",
        enabled=True,
        active_sides=("left", "right"),
        coupled_pre_ik_target_filter_available=True,
        snapshot={"schema_version": "position-admittance-v5", "mode": "active"},
    )
    assert enabled["enabled"] is True
    assert enabled["active_sides"] == ["left", "right"]
    assert enabled["schema_version"] == "position-admittance-v5"
    assert enabled["mode"] == "active"


@pytest.mark.parametrize("axes", ("", "xx", "xq"))
def test_shadow_axes_reject_ambiguous_values(axes: str) -> None:
    args = build_real_teleop_argument_parser().parse_args(
        ["--position-admittance-shadow-axes", axes]
    )
    with pytest.raises(ValueError, match="axes"):
        position_admittance_shadow_config(args)
