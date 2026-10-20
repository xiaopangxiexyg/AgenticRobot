from __future__ import annotations

import ast
import inspect
import io
import json
import math
import os
import pty
import termios
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hardware.rebot_b601_rs import B601HardwareState, B601JointState
from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601PolicyAssistCommand,
    B601SharedControlConfig,
    B601SharedControlRuntime,
)
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_joint_angle_display_uses_signed_measured_degrees_and_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices = []
    monkeypatch.setattr(probe, "_operator_notice", notices.append)
    display = probe._JointAngleDisplay()
    q = (0.0, -0.005, math.pi / 6, -math.pi / 2, 0.2, -0.3)
    display.observe(q, observed_at_sec=10.0, phase="ACTIVE")
    display.observe(q, observed_at_sec=10.49, phase="ACTIVE")
    moved = (*q[:2], math.pi / 3, *q[3:])
    display.observe(moved, observed_at_sec=10.5, phase="PROTECTIVE_HOLD")

    assert len(notices) == 2
    assert "phase=ACTIVE unit=deg" in notices[0]
    assert "J2=  -0.286" in notices[0]
    assert "J3= +30.000" in notices[0]
    assert "J4= -90.000" in notices[0]
    assert "J6=" in notices[0]
    assert "J7=" not in notices[0]
    assert "phase=PROTECTIVE_HOLD" in notices[1]
    assert "J3= +60.000" in notices[1]


def _shared_runtime() -> B601SharedControlRuntime:
    return B601SharedControlRuntime(
        B601SharedControlConfig(
            position_kp_nm_per_rad=(2.0,) * 6,
            velocity_kd_nm_s_per_rad=(0.0,) * 6,
            max_target_error_rad=(0.04,) * 6,
            max_abs_assist_torque_nm=(0.10,) * 6,
            max_assist_torque_rate_nm_s=(0.50,) * 6,
            authority_rise_per_sec=5.0,
            policy_stale_timeout_sec=0.08,
        )
    )


def test_injected_shared_control_uses_current_state_and_bounded_overlay() -> None:
    runtime = _shared_runtime()
    qpos = (0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    target = (*qpos[:5], 0.04)
    receipt = runtime.publish_policy(
        B601PolicyAssistCommand(
            sequence=1,
            source_timestamp_sec=1.0,
            receive_timestamp_sec=1.0,
            qpos_target_rad=target,
        )
    )
    assert receipt.accepted

    reference, audit = probe._step_shared_control_runtime(
        runtime,
        control_sequence=7,
        control_timestamp_sec=1.0,
        measured_qpos_rad=qpos,
        measured_qvel_rad_s=(0.0,) * 6,
        dt_sec=0.01,
    )

    assert reference.qpos_rad == qpos
    assert reference.qvel_rad_s == (0.0,) * 6
    assert reference.joint_torque_overlay_nm == pytest.approx(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.004)
    )
    assert audit["mode"] == "policy_assist"
    assert audit["authority"] == pytest.approx(0.05)
    assert audit["last_policy_sequence"] == 1


def test_injected_shared_control_rejects_stacked_position_controller() -> None:
    args = probe.build_parser().parse_args(
        [
            "--mode",
            "hold",
            "--config",
            "unused.yaml",
            "--rebot-source-root",
            "unused",
            "--kp",
            "1.0",
        ]
    )

    with pytest.raises(probe.LowGainProbeError, match="zero software"):
        probe._validate_shared_control_integration(args, _shared_runtime())


def test_injected_shared_control_must_fit_inside_final_torque_envelope() -> None:
    runtime = _shared_runtime()

    assert probe._shared_control_overlay_limits(
        runtime,
        total_torque_limits_nm=(1.0,) * 6,
    ) == (0.1,) * 6
    with pytest.raises(probe.LowGainProbeError, match="final software torque"):
        probe._shared_control_overlay_limits(
            runtime,
            total_torque_limits_nm=(0.05,) * 6,
        )


def test_injected_shared_control_leaves_existing_hold_contract_unchanged() -> None:
    args = probe.build_parser().parse_args(
        [
            "--mode",
            "hold",
            "--config",
            "unused.yaml",
            "--rebot-source-root",
            "unused",
        ]
    )

    probe._validate_shared_control_integration(args, None)


def test_gravity_debug_profile_copies_tuning_parameters_as_immutable_snapshot() -> None:
    scales = [0.91, 0.92, 0.93, 0.94, 0.95, 0.96]
    offsets = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    profile = probe.B601GravityCompensationDebugProfile.capture(
        gravity_scale=0.87,
        gravity_joint_scales=scales,
        gravity_joint_offsets_rad=offsets,
        gravity_joint_bias_nm=(0.0, -0.02, -0.03, -0.04, 0.0, 0.01),
        gravity_coulomb_friction_nm=(0.01,) * 6,
        gravity_coulomb_velocity_threshold_rad_s=0.02,
        driver_kd_nm_s_per_rad=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    )
    scales[0] = 99.0
    offsets[0] = 99.0

    assert profile.gravity_scale == pytest.approx(0.87)
    assert profile.gravity_joint_scales == pytest.approx(
        (0.91, 0.92, 0.93, 0.94, 0.95, 0.96)
    )
    assert profile.gravity_joint_offsets_rad == pytest.approx(
        (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
    )
    with pytest.raises(AttributeError):
        profile.gravity_scale = 1.0  # type: ignore[misc]


def test_condition_torque_applies_absolute_then_slew_limits() -> None:
    conditioned, torque_limited, slew_limited = probe.condition_torque(
        (10.0, -10.0, 0.1, 0.0, 0.0, 0.0),
        (0.0,) * 6,
        absolute_limits_nm=(2.0,) * 6,
        max_rate_nm_s=(1.0,) * 6,
        dt_sec=0.5,
    )

    assert conditioned == (0.5, -0.5, 0.1, 0.0, 0.0, 0.0)
    assert torque_limited is True
    assert slew_limited is True


def test_cartesian_assist_release_slew_reaches_braking_torque_faster() -> None:
    decision = probe.select_cartesian_assist_torque_slew(
        requested_nm=(0.5,) + (0.0,) * 5,
        previous_nm=(-0.2,) + (0.0,) * 5,
        measured_qvel_rad_s=(-0.7,) + (0.0,) * 5,
        rise_rate_nm_s=0.15,
        release_rate_nm_s=2.0,
        brake_velocity_deadband_rad_s=0.02,
    )
    conditioned, torque_limited, slew_limited = probe.condition_torque(
        decision.conditioning_target_nm,
        (-0.2,) + (0.0,) * 5,
        absolute_limits_nm=(0.5,) * 6,
        max_rate_nm_s=decision.selected_rate_nm_s,
        dt_sec=0.05,
    )

    assert decision.mode[0] == "brake"
    assert conditioned[0] == pytest.approx(-0.1)
    assert torque_limited is False
    assert slew_limited is True


def test_cartesian_assist_brake_reverses_obsolete_j2_torque_before_target_crossing() -> None:
    # Reduced samples from a pre-fix negative-X commissioning trace: J2 was
    # moving negative while its requested overlay had turned positive.
    requests_nm = (0.00084, 0.03767, 0.09491, 0.19049, 0.25739)
    previous_nm = (-0.09676,) + (0.0,) * 5
    sent_j2_nm = []

    for requested_j2_nm in requests_nm:
        decision = probe.select_cartesian_assist_torque_slew(
            requested_nm=(requested_j2_nm,) + (0.0,) * 5,
            previous_nm=previous_nm,
            measured_qvel_rad_s=(-0.3,) + (0.0,) * 5,
            rise_rate_nm_s=0.15,
            release_rate_nm_s=2.0,
            brake_velocity_deadband_rad_s=0.02,
        )
        previous_nm, _, _ = probe.condition_torque(
            decision.conditioning_target_nm,
            previous_nm,
            absolute_limits_nm=(0.4,) * 6,
            max_rate_nm_s=decision.selected_rate_nm_s,
            dt_sec=0.05,
        )
        assert decision.mode[0] == "brake"
        sent_j2_nm.append(previous_nm[0])

    assert sent_j2_nm[0] == pytest.approx(requests_nm[0])
    assert sent_j2_nm[-1] == pytest.approx(requests_nm[-1])
    assert sent_j2_nm == sorted(sent_j2_nm)


def test_cartesian_assist_response_notice_distinguishes_target_from_minimum() -> None:
    notice = probe.format_cartesian_assist_response_notice(
        {
            "status": "PASS",
            "settled_signed_displacement_m": 0.0015,
            "commanded_offset_m": -0.015,
            "minimum_response_m": 0.001,
            "peak_signed_force_n": 0.89,
            "max_lateral_drift_m": 0.0035,
            "max_rotation_drift_rad": 0.01,
        }
    )

    assert "status=PASS" in notice
    assert "displacement=1.500mm target=15.000mm minimum=1.000mm" in notice


def test_torque_conditioning_sample_diagnostics_identifies_absolute_and_slew_limits() -> None:
    audit = probe.torque_conditioning_sample_diagnostics(
        (3.0, -0.5, 0.0, 0.0, 0.0, 0.0),
        (0.0,) * 6,
        (0.5, -0.5, 0.0, 0.0, 0.0, 0.0),
        absolute_limits_nm=(2.0,) * 6,
        max_rate_nm_s=(1.0,) * 6,
        dt_sec=0.5,
    )

    assert audit["absolute_torque_saturated_joint_names"] == ("joint1",)
    assert audit["slew_torque_saturated_joint_names"] == ("joint1",)
    assert audit["absolute_torque_margin_nm"][0] == pytest.approx(-1.0)
    assert audit["absolute_torque_request_ratio"][0] == pytest.approx(1.5)
    assert audit["absolute_clipped_tau_nm"][0] == pytest.approx(2.0)


def test_summarize_torque_conditioning_reports_per_joint_authority() -> None:
    samples = [
        {
            "elapsed_sec": 0.0,
            "requested_tau_nm": (1.0, 3.0, 0.0, 0.0, 0.0, 0.0),
            "conditioned_tau_nm": (1.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            "torque_limited": True,
            "slew_limited": False,
        },
        {
            "elapsed_sec": 0.05,
            "requested_tau_nm": (1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            "conditioned_tau_nm": (0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
            "torque_limited": False,
            "slew_limited": True,
        },
    ]

    summary = probe.summarize_torque_conditioning(
        samples,
        absolute_limits_nm=(2.0,) * 6,
    )

    assert summary["sample_count"] == 2
    assert summary["any_absolute_saturation"] is True
    assert summary["any_slew_saturation"] is True
    assert summary["per_joint"]["joint2"]["absolute_saturation_count"] == 1
    assert summary["per_joint"]["joint1"]["slew_saturation_count"] == 1
    assert summary["per_joint"]["joint2"]["minimum_absolute_margin_nm"] == pytest.approx(-1.0)


def test_condition_torque_rejects_nonpositive_dt() -> None:
    with pytest.raises(ValueError, match="dt_sec must be positive"):
        probe.condition_torque(
            (0.0,) * 6,
            (0.0,) * 6,
            absolute_limits_nm=(1.0,) * 6,
            max_rate_nm_s=(1.0,) * 6,
            dt_sec=0.0,
        )


def test_gravity_calibration_capture_repeats_free_drag_and_fixed_windows() -> None:
    capture = probe.GravityCalibrationCapture(20.0)
    first_pose = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    moved_pose = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

    assert capture.reference(first_pose) == first_pose
    first = capture.trigger(
        first_pose,
        elapsed_sec=5.0,
        sample_start_index=12,
    )
    assert first is not None
    assert first["index"] == 1
    assert capture.state == "RECORDING"
    assert capture.reference(moved_pose) == first_pose
    assert (
        capture.trigger(moved_pose, elapsed_sec=6.0, sample_start_index=13)
        is None
    )
    assert capture.complete_if_due(elapsed_sec=24.99, sample_end_index=410) is None

    completed = capture.complete_if_due(elapsed_sec=25.0, sample_end_index=412)
    assert completed is first
    assert completed["completed"] is True
    assert completed["completion_reason"] == "duration_complete"
    assert completed["sample_end_index"] == 412
    assert completed["observed_duration_sec"] == pytest.approx(20.0)
    assert capture.state == "FREE_DRAG"
    assert capture.reference(moved_pose) == moved_pose

    second = capture.trigger(
        moved_pose,
        elapsed_sec=30.0,
        sample_start_index=500,
    )
    assert second is not None
    assert second["index"] == 2
    interrupted = capture.interrupt(
        elapsed_sec=34.0,
        sample_end_index=580,
        reason="operator_interrupt",
    )
    assert interrupted is second
    assert interrupted["completed"] is False
    assert interrupted["completion_reason"] == "operator_interrupt"
    assert capture.state == "FREE_DRAG"

    acceptance = probe.evaluate_gravity_calibration_capture(capture)
    assert acceptance["status"] == "FAIL"
    assert acceptance["completed_record_count"] == 1
    assert acceptance["incomplete_record_count"] == 1


def test_gravity_calibration_acceptance_requires_a_complete_window() -> None:
    capture = probe.GravityCalibrationCapture(20.0)

    acceptance = probe.evaluate_gravity_calibration_capture(capture)

    assert acceptance == {
        "status": "FAIL",
        "completed_record_count": 0,
        "incomplete_record_count": 0,
        "reasons": ("no complete calibration recording window",),
    }


def test_gravity_calibration_acceptance_passes_only_complete_windows() -> None:
    capture = probe.GravityCalibrationCapture(20.0)
    capture.trigger((0.0,) * 6, elapsed_sec=1.0, sample_start_index=0)
    capture.complete_if_due(elapsed_sec=21.0, sample_end_index=400)

    acceptance = probe.evaluate_gravity_calibration_capture(capture)

    assert acceptance == {
        "status": "PASS",
        "completed_record_count": 1,
        "incomplete_record_count": 0,
        "reasons": (),
    }


def test_gravity_calibration_capture_has_settling_before_recording() -> None:
    capture = probe.GravityCalibrationCapture(
        2.0,
        settling_duration_sec=1.0,
    )
    record = capture.trigger(
        (0.1,) * 6,
        elapsed_sec=5.0,
        sample_start_index=10,
    )
    assert record is not None
    assert capture.state == "SETTLING"
    assert record["sample_start_index"] is None
    assert capture.start_recording_if_due(
        elapsed_sec=5.99,
        sample_start_index=20,
    ) is None
    started = capture.start_recording_if_due(
        elapsed_sec=6.0,
        sample_start_index=30,
    )
    assert started is record
    assert capture.state == "RECORDING"
    assert record["sample_start_index"] == 30
    completed = capture.complete_if_due(elapsed_sec=8.0, sample_end_index=70)
    assert completed is record
    assert record["recording_duration_sec"] == pytest.approx(2.0)


def _synthetic_calibration_payload(*, software_kp: float = 5.0) -> dict[str, object]:
    gravity_vectors = (
        (0.0, 2.0, 4.0, 1.0, 0.1, -0.1),
        (0.0, 3.0, 2.0, 1.5, -0.2, 0.2),
    )
    references = (
        (0.0, 0.8, 0.9, 0.1, 0.0, 0.0),
        (0.2, 1.2, 0.6, 0.4, 0.1, -0.2),
    )
    # residual = 0.1 * nominal gravity + a fixed joint bias
    bias = np.asarray((0.0, 0.05, -0.03, 0.02, 0.01, -0.01))
    samples: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for pose_index, (gravity, reference) in enumerate(
        zip(gravity_vectors, references, strict=True)
    ):
        residual = 0.1 * (np.asarray(gravity) / 0.5) + bias
        start = len(samples)
        for _offset in range(3):
            samples.append(
                {
                    "elapsed_sec": float(len(samples)) * 0.05,
                    "qpos_rad": reference,
                    "qref_rad": reference,
                    "qvel_fd_rad_s": (0.0,) * 6,
                    "gravity_tau_nm": gravity,
                    "position_tau_nm": tuple(residual),
                    "requested_tau_nm": gravity,
                    "conditioned_tau_nm": gravity,
                    "torque_limited": False,
                    "slew_limited": False,
                }
            )
        records.append(
            {
                "index": len(records) + 1,
                "completed": True,
                "sample_start_index": start,
                "sample_end_index": len(samples),
                "reference_qpos_rad": reference,
            }
        )
    return {
        "mode": "gravity-calibration",
        "profile": {
            "gravity_scale": 0.5,
            "software_kp_nm_per_rad": (software_kp,) * 6,
        },
        "gravity_calibration": {"records": records},
        "samples": samples,
    }


def test_calibration_window_reduction_and_r_only_fit() -> None:
    payload = _synthetic_calibration_payload()
    reduced = probe.reduce_gravity_calibration_windows(payload)
    assert reduced["completed_record_count"] == 2
    assert all(window["valid"] for window in reduced["windows"])
    fit = probe.fit_gravity_calibration(
        payload,
        pose_tolerance_rad=0.01,
        model="global-scale-joint-bias",
    )
    assert fit["status"] == "PASS"
    assert fit["gravity_scale_correction"] == pytest.approx(0.1, abs=1e-10)
    assert fit["nominal_gravity_joint_scales"] == pytest.approx((1.0,) * 6)
    assert fit["effective_gravity_multipliers"] == pytest.approx((0.6,) * 6)
    assert fit["joint_bias_nm"] == pytest.approx(
        (0.0, 0.05, -0.03, 0.02, 0.01, -0.01),
        abs=1e-10,
    )
    assert fit["pairing"]["pair_count"] == 0
    assert fit["single_window_observation_count"] == 2


def test_calibration_validation_uses_independent_r_windows() -> None:
    payload = _synthetic_calibration_payload()
    fit = probe.fit_gravity_calibration(
        payload,
        pose_tolerance_rad=0.01,
        model="global-scale-joint-bias",
    )

    validation = probe.evaluate_gravity_calibration_model(
        fit,
        payload,
        pose_tolerance_rad=0.01,
    )

    assert validation["status"] == "PASS"
    assert validation["observation_count"] == 2
    assert validation["pairing"]["pair_count"] == 0
    assert validation["residual_reduced"] is True


def test_calibration_fit_rejects_zero_kp_residual_observation() -> None:
    payload = _synthetic_calibration_payload(software_kp=0.0)
    fit = probe.fit_gravity_calibration(
        payload,
        pose_tolerance_rad=0.01,
        model="global-scale",
    )
    assert fit["status"] == "FAIL"
    assert "software Kp is zero" in fit["reasons"][0]


def test_formal_fit_does_not_require_operator_direction_labels() -> None:
    payload = _synthetic_calibration_payload()

    fit = probe.fit_gravity_calibration(
        payload,
        pose_tolerance_rad=0.01,
        model="global-scale",
    )

    assert fit["status"] == "PASS"
    assert fit["pairing"]["pair_count"] == 0
    assert fit["observation_semantics"] == (
        "each R capture is reduced independently; nearby repeats may be averaged"
    )


def test_pose_selection_deduplicates_repeated_reference_windows() -> None:
    observations = [
        {
            "reference_qpos_rad": (0.0,) * 6,
            "gravity_tau_nm": (3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
        {
            "reference_qpos_rad": (0.01,) * 6,
            "gravity_tau_nm": (1.0, 0.1, 0.0, 0.0, 0.0, 0.0),
        },
        {
            "reference_qpos_rad": (0.4,) * 6,
            "gravity_tau_nm": (2.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        },
    ]

    selection = probe.select_calibration_poses(
        observations,
        model="global-scale",
        pose_tolerance_rad=0.08,
    )

    assert len(selection["selected_indices"]) == 2
    assert selection["duplicate_rejected_indices"] == (1,)
    assert selection["rejected_indices"] == (1,)


def test_global_scale_payload_fit_recovers_mass_and_first_moment(tmp_path) -> None:
    class FakePayloadModel:
        urdf_path = tmp_path / "payload.urdf"
        gravity_mps2 = (0.0, 0.0, -9.81)

        def forward_kinematics(self, qpos_rad: object) -> np.ndarray:
            q = np.asarray(qpos_rad, dtype=float)
            a, b, c = 0.2 * q[0], -0.1 * q[1], 0.15 * q[2]
            ca, sa = np.cos(a), np.sin(a)
            cb, sb = np.cos(b), np.sin(b)
            cc, sc = np.cos(c), np.sin(c)
            rx = np.asarray(((1.0, 0.0, 0.0), (0.0, ca, -sa), (0.0, sa, ca)))
            ry = np.asarray(((cb, 0.0, sb), (0.0, 1.0, 0.0), (-sb, 0.0, cb)))
            rz = np.asarray(((cc, -sc, 0.0), (sc, cc, 0.0), (0.0, 0.0, 1.0)))
            transform = np.eye(4)
            transform[:3, :3] = rx @ ry @ rz
            return transform

        def arm_jacobian(self, qpos_rad: object) -> np.ndarray:
            q = np.asarray(qpos_rad, dtype=float)
            jacobian = np.asarray(
                (
                    (1.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                    (0.2, 1.0, 0.1, 0.2, 0.3, 0.4),
                    (0.3, 0.2, 1.0, 0.1, 0.2, 0.3),
                    (0.4, 0.3, 0.2, 1.0, 0.1, 0.2),
                    (0.5, 0.4, 0.3, 0.2, 1.0, 0.1),
                    (0.6, 0.5, 0.4, 0.3, 0.2, 1.0),
                )
            )
            for index in range(6):
                jacobian += (0.03 * q[index]) * np.roll(
                    np.eye(6),
                    index,
                    axis=1,
                )
            return jacobian

    model = FakePayloadModel()
    model.urdf_path.write_text("<robot name='payload'/>", encoding="utf-8")
    payload_params = np.asarray((0.4, 0.02, -0.01, 0.03))
    samples: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for pose_index in range(8):
        reference = (
            0.1 + 0.2 * pose_index,
            0.5 + 0.1 * pose_index,
            1.0 + 0.07 * pose_index,
            0.2 + 0.05 * pose_index,
            -0.3 + 0.08 * pose_index,
            0.1 - 0.03 * pose_index,
        )
        gravity = np.asarray(
            (
                1.0 + 0.3 * pose_index,
                2.0 - 0.2 * pose_index,
                3.0 + 0.4 * pose_index,
                0.5 + 0.2 * pose_index,
                -0.1 + 0.1 * pose_index,
                0.3 - 0.15 * pose_index,
            )
        )
        residual = 0.2 * gravity + probe.payload_gravity_torque_regressor(
            model,
            reference,
        ) @ payload_params
        start = len(samples)
        for _offset in range(3):
            samples.append(
                {
                    "elapsed_sec": len(samples) * 0.05,
                    "qpos_rad": reference,
                    "qref_rad": reference,
                    "qvel_fd_rad_s": (0.0,) * 6,
                    "gravity_tau_nm": tuple(gravity),
                    "position_tau_nm": tuple(residual),
                    "requested_tau_nm": tuple(gravity),
                    "conditioned_tau_nm": tuple(gravity),
                    "torque_limited": False,
                    "slew_limited": False,
                }
            )
        records.append(
            {
                "index": len(records) + 1,
                "completed": True,
                "sample_start_index": start,
                "sample_end_index": len(samples),
                "reference_qpos_rad": reference,
            }
        )
    payload = {
        "mode": "gravity-calibration",
        "profile": {
            "gravity_scale": 1.0,
            "software_kp_nm_per_rad": (5.0,) * 6,
        },
        "gravity_calibration": {"records": records},
        "samples": samples,
    }

    fit = probe.fit_gravity_calibration(
        payload,
        pose_tolerance_rad=0.01,
        model="global-scale-payload",
        kinematic_model=model,
    )

    assert fit["status"] == "PASS"
    assert fit["fit"]["rank"] == 5
    assert fit["gravity_scale_correction"] == pytest.approx(0.2)
    assert fit["payload_mass_kg"] == pytest.approx(0.4)
    assert fit["payload_first_moment_kg_m"] == pytest.approx(
        (0.02, -0.01, 0.03)
    )
    assert fit["single_window_observation_count"] == 8


def test_enabled_shutdown_always_attempts_hold_before_disable() -> None:
    assert (
        probe.resolve_shutdown_action(
            enabled=True,
            full_disable_requested=False,
        )
        == "protective_hold"
    )
    assert (
        probe.resolve_shutdown_action(
            enabled=True,
            full_disable_requested=True,
        )
        == "safe_stop"
    )
    assert (
        probe.resolve_shutdown_action(
            enabled=False,
            full_disable_requested=False,
        )
        == "safe_stop"
    )


@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_terminal_host_release_is_separate_from_device_cleanup(confirmed, fails):
    events, errors = [], ["safe_stop: unconfirmed"] if not confirmed else []
    class Backend:
        def close(self):
            events.append("close")
            if fails:
                raise OSError("synthetic release failure")
        def release_transport(self):
            events.append("release_transport")
            if fails:
                raise OSError("synthetic release failure")
    outcome = probe._release_owner_transport(Backend(), device_cleanup_confirmed=confirmed,
                                              cleanup_errors=errors)
    assert events == ["close" if confirmed else "release_transport"]
    assert outcome["completed"] is (not fails)
    assert bool(outcome["error"]) is fails
    assert ("safe_stop: unconfirmed" in errors) is (not confirmed)
    assert len(errors) == int(not confirmed)+int(fails)


def test_incomplete_legacy_transport_release_is_not_success():
    errors = []
    outcome = probe._release_owner_transport(SimpleNamespace(close=lambda: False),
        device_cleanup_confirmed=False, cleanup_errors=errors)
    assert outcome["method"] == "close"
    assert outcome["completed"] is False
    assert errors == ["close: host transport release is incomplete"]


def test_failed_active_report_cleanup_releases_real_backend_handle_without_confirming_safety(monkeypatch, tmp_path):
    from test_rebot_b601_hardware_backend import _config, _FakeController

    from hardware.rebot_b601_rs.backend import B601HardwareError, B601RSBackend

    class Controller(_FakeController):
        def add_robstride_motor(self, *args):
            motor = super().add_robstride_motor(*args)
            motor.fail_active_report_calls = {1, 2, 3}
            motor.fail_method = "disable"
            return motor
    controller = Controller()
    config = _config(allow_motion=True, allow_active_reporting=True)
    backend = B601RSBackend(config, controller_factory=lambda _channel: controller)
    path = tmp_path/"config.yaml"
    path.write_text("offline fixture\n")
    monkeypatch.setattr(probe.B601RSBackendConfig, "from_rs_yaml", lambda *a, **kw: config)
    monkeypatch.setattr(probe, "B601RSBackend", lambda _config: backend)
    monkeypatch.setattr(probe, "_resolve_model_and_limits", lambda root: (
        SimpleNamespace(urdf_path=path), _fault_hold_limits()))
    monkeypatch.setattr(probe, "_resolve_effort_limits", lambda *a, **kw: (1.,)*6)
    args = probe.build_parser().parse_args([
        "--mode", "hold", "--config", str(path), "--rebot-source-root", str(tmp_path),
        "--allow-motion", "--auto-disable", "--confirmation", probe.CONFIRMATION,
        "--apply-driver-torque-limits",
        "--feedback-source", "active-report", "--allow-active-report", "--auto-disable-active-report",
    ])
    with pytest.raises(probe.LowGainProbeError, match="active-report enable failed") as caught:
        probe.run_probe(args)
    cleanup = caught.value.cleanup
    assert cleanup["device_cleanup_confirmed"] is False
    assert cleanup["motor_disable_confirmed"] is False
    assert cleanup["active_report_disable_confirmed"] is False
    assert cleanup["transport_release"] == {"method": "release_transport", "completed": True, "error": None}
    assert cleanup["errors"] and any("safe_stop" in e for e in cleanup["errors"])
    assert controller.calls[-1] == "close"
    assert not backend.connected
    assert backend.active_report_joint_names == ("joint1",)
    assert all(any(method == "disable" for method, args in motor.calls) for motor in controller.motors.values())
    assert not any(method in {"enable", "send_mit"} for motor in controller.motors.values() for method, args in motor.calls)
    with pytest.raises(B601HardwareError, match="unconfirmed device state"):
        backend.connect()


@pytest.fixture
def startup_recovery(monkeypatch):
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    clock = [10.0]
    stop = {"requested": True, "full_disable_requested": False}

    class Backend:
        calls = 0
        dt = 0.05
        states = (q, q)
        stale = False
        valid = True
        full_stop_during_read = False

        def poll_feedback(self, *, request_feedback):
            assert request_feedback
            value = self.states[self.calls]
            self.calls += 1
            clock[0] += self.dt
            if self.full_stop_during_read:
                stop["full_disable_requested"] = True
            if isinstance(value, Exception):
                raise value
            return SimpleNamespace(
                valid=self.valid, stale=self.stale, positions_rad=value
            )

        def send_mit(self, *args, **kwargs):
            pytest.fail("feedback recovery must not issue torque or position commands")

        def safe_stop(self, *args, **kwargs):
            pytest.fail("feedback recovery must not disable the arm")

    backend = Backend()
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])

    def recover():
        return probe._recover_startup_hold_snapshot(
            backend,
            pre_enable_qpos_rad=q,
            position_limits_rad=_fault_hold_limits(),
            max_age_sec=0.15,
            max_velocity_rad_s=0.02,
            max_rebase_rad=0.05,
            stop=stop,
        )

    return backend, stop, recover


def test_startup_recovery_reads_current_pose_despite_first_interrupt(startup_recovery):
    backend, _stop, recover = startup_recovery
    current = (0.1, 0.701, 1.1, 0.0, -0.1, 0.2)
    backend.states = [current, current]
    q, velocity, observed = recover()
    assert q == current
    assert velocity == (0.0,) * 6
    assert observed == pytest.approx(10.1)
    assert backend.calls == 2


@pytest.mark.parametrize("fault", [
    "stale", "invalid", "nan", "incomplete", "out_of_range", "moving",
    "displaced", "slow", "zero_dt", "negative_dt", "rx_error",
])
def test_startup_recovery_rejects_untrusted_samples(startup_recovery, fault):
    backend, _stop, recover = startup_recovery
    q = backend.states[0]
    if fault == "stale":
        backend.stale = True
    elif fault == "invalid":
        backend.valid = False
    elif fault == "nan":
        backend.states = [q, (float("nan"), *q[1:])]
    elif fault == "incomplete":
        backend.states = [q, q[:5]]
    elif fault == "out_of_range":
        backend.states = [q, (20.0, *q[1:])]
    elif fault == "moving":
        backend.states = [q, (q[0] + 0.01, *q[1:])]
    elif fault == "displaced":
        moved = (q[0] + 0.1, *q[1:])
        backend.states = [moved, moved]
    elif fault == "slow":
        backend.dt = 0.1
    elif fault == "zero_dt":
        backend.dt = 0.0
    elif fault == "negative_dt":
        backend.dt = -0.01
    else:
        backend.states = [q, OSError("CAN unavailable")]
    with pytest.raises((probe.LowGainProbeError, ValueError, OSError)):
        recover()


@pytest.mark.parametrize("during_read", [False, True])
def test_startup_recovery_respects_explicit_full_stop(startup_recovery, during_read):
    backend, stop, recover = startup_recovery
    backend.full_stop_during_read = during_read
    stop["full_disable_requested"] = not during_read
    with pytest.raises(probe.LowGainProbeError, match="cancelled by full disable"):
        recover()
    assert backend.calls == int(during_read)


def test_hold_entry_does_not_send_after_explicit_full_stop(monkeypatch):
    stopped = []

    class Backend:
        def safe_stop(self, *, joint_names):
            stopped.append(joint_names)
            return probe.B601ModeReceipt(
                operation="safe_stop", requested_joint_names=joint_names,
                completed_joint_names=joint_names, failed_joint_names=(),
                errors=(), completed=True,
            )

    def forbidden(*args, **kwargs):
        pytest.fail("full stop must precede any hold frame")

    monkeypatch.setattr(probe, "_send_fault_hold_command", forbidden)
    monkeypatch.setattr(probe, "_send_cached_fault_hold_command", forbidden)
    result = probe._run_latched_fault_hold(
        Backend(), object(), qpos_rad=None, qvel_rad_s=None,
        observed_age_sec=0.0, fault_qpos_rad=None, fault_qvel_rad_s=None,
        position_limits_rad=_fault_hold_limits(), max_age_sec=0.15,
        max_velocity_rad_s=0.02, max_rebase_rad=0.05,
        profile=object(), gravity_scale=1.0, previous_tau=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6, rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": True},
    )
    assert result[0] == 0
    assert result[4].completed
    assert stopped == [probe._ALL_JOINTS]


@pytest.mark.parametrize("failure,cleanup_fault", [
    ("interrupt", None), ("first_rx_error", None), ("persistent_rx_error", None),
    ("first_rx_error", "owner_stopped"), ("first_rx_error", "record_flush"),
    ("first_rx_error", "hold_trace"), ("first_rx_error", "combined"),
    ("first_rx_error", "operator_keys"),
])
def test_enabled_startup_exit_attempts_recovery_before_stop(monkeypatch, tmp_path, failure, cleanup_fault):
    import signal

    events = []
    handlers = {}
    clock = [10.0]
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test fixture\n")
    config = SimpleNamespace(
        channel="fake-no-hardware",
        joints=[SimpleNamespace(mit_kp=1.0, mit_kd=0.1) for _ in range(6)],
        feedback_timeout_sec=0.05,
    )

    def receipt():
        return probe.B601ModeReceipt(
            operation="safe_stop",
            requested_joint_names=probe._ALL_JOINTS,
            completed_joint_names=probe._ALL_JOINTS, failed_joint_names=(),
            errors=(), completed=True,
        )

    class Backend:
        enabled = False
        reads = 0
        enabled_reads = 0

        def connect(self):
            pass

        def poll_feedback(self, **kwargs):
            self.reads += 1
            self.enabled_reads += int(self.enabled)
            clock[0] += 0.05
            events.append("read")
            if self.enabled and (
                failure == "persistent_rx_error"
                or (failure == "first_rx_error" and self.enabled_reads == 1)
            ):
                raise OSError("injected RX failure")
            if cleanup_fault:
                joints = tuple(B601JointState(
                    name=name, motor_id=i + 1, feedback_id=0xFD, arbitration_id=24 << 24,
                    status_code=0, position_rad=q[i], velocity_reported=0., torque_reported_nm=0.,
                    observed_at_sec=clock[0], sample_sequence=self.reads, valid=True, stale=False,
                    position_source="robstride_operation_status_cache", status_sequence=self.reads,
                    receive_age_sec=0., host_receive_freshness_verified=True,
                ) for i, name in enumerate(probe._ALL_JOINTS))
                return B601HardwareState(
                    sequence=self.reads, receive_timestamp_sec=clock[0], joints=joints,
                    valid=True, stale=False, host_receive_freshness_verified=True,
                )
            return SimpleNamespace(
                valid=True, stale=False, positions_rad=q, sequence=self.reads,
                receive_timestamp_sec=clock[0], joints=(),
            )

        def safe_stop(self, **kwargs):
            events.append("disable")
            return receipt()

        def set_active_reporting(self, enabled, **kwargs):
            events.append("report_on" if enabled else "report_off")
            return receipt()

        def enable(self, **kwargs):
            self.enabled = True
            events.append("enable")
            if cleanup_fault and cleanup_fault != "operator_keys":
                # Enable queued-record cleanup only after CLI admission. The
                # external backend and its failures remain entirely synthetic.
                args.torque_console_auto_calibration_profile = tmp_path / "unused-profile.json"
            if failure == "interrupt":
                handlers[signal.SIGINT](signal.SIGINT, None)
            return receipt()

        def apply_temporary_driver_torque_limits(self, *args, **kwargs):
            return SimpleNamespace(completed=True)

        def restore_driver_torque_limits(self):
            events.append("restore")
            return SimpleNamespace(completed=True)

        def prepare_mit_mode(self, **kwargs):
            return receipt()

        def close(self):
            events.append("close")

    def hold(backend, model, **kwargs):
        assert kwargs["stop"]["requested"] is True
        assert kwargs["qpos_rad"] == q
        assert kwargs["qvel_rad_s"] == (0.0,) * 6
        assert kwargs["cached_gravity_tau_nm"] is None
        assert kwargs["hold_profile"].position_kp_nm_per_rad == (0.0,) * 6
        assert kwargs["driver_kd_nm_s_per_rad"] == probe._DEFAULT_DRIVER_KD
        assert kwargs["gravity_joint_bias_nm"] == probe._DEFAULT_GRAVITY_JOINT_BIAS_NM
        events.append("hold")
        # The real unbounded hold is separately tested. Simulate its explicit
        # final operator stop, never a timer-based or implicit release.
        kwargs["stop"]["full_disable_requested"] = True
        return 1, (0.0,) * 6, None, None, backend.safe_stop(), []

    monkeypatch.setattr(probe.signal, "signal", lambda signum, handler: handlers.update({signum: handler}))
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(probe, "B601RSBackend", lambda config: Backend())
    monkeypatch.setattr(probe.B601RSBackendConfig, "from_rs_yaml", lambda *a, **kw: config)
    monkeypatch.setattr(probe, "_resolve_model_and_limits", lambda root: (SimpleNamespace(urdf_path=config_path), _fault_hold_limits()))
    monkeypatch.setattr(probe, "_resolve_effort_limits", lambda *a, **kw: (1.0,) * 6)
    monkeypatch.setattr(probe, "_drive_enable_countdown", lambda stop: True)
    monkeypatch.setattr(probe, "_operator_notice", lambda message: None)
    monkeypatch.setattr(probe, "_run_latched_fault_hold", hold)
    args = probe.build_parser().parse_args([
        "--mode", "hold", "--config", str(config_path),
        "--rebot-source-root", str(tmp_path), "--allow-motion", "--auto-disable",
        "--apply-driver-torque-limits", "--confirmation", probe.CONFIRMATION,
    ])
    console = None
    trace = None
    if cleanup_fault:
        args.feedback_source = "active-report"
        args.allow_active_report = args.auto_disable_active_report = True
        def owner_stopped(**kwargs):
            events.append("owner_stopped")
            if cleanup_fault in {"owner_stopped", "combined"}:
                raise RuntimeError("injected console finalization failure")
            real_owner_stopped(**kwargs)

        def drain_records():
            events.append("record_flush")
            if cleanup_fault in {"record_flush", "combined"}:
                return [{"kind": "automatic_calibration_cycle_transition", "time_sec": clock[0]}]
            return []

        if cleanup_fault == "operator_keys":
            args.mode = "gravity-calibration"

            class Keys:
                def close(self):
                    events.append("keys_close")
                    raise OSError("injected terminal cleanup failure")

            monkeypatch.setattr(probe, "OperatorKeyReader", Keys)
        else:
            args.torque_console = True
            args.gravity_debug_mode = True
            args.torque_console_confirmation = "B601_RS_TORQUE_CONSOLE"
            console = probe._direct_torque_console_from_args(args)
            real_owner_stopped = console.owner_stopped
            monkeypatch.setattr(console, "owner_stopped", owner_stopped)
            monkeypatch.setattr(console, "drain_records", drain_records, raising=False)

        class Trace(probe._DynamicsTrace):
            def record(self, kind, **fields):
                if kind == "protective_hold_attempt" and cleanup_fault in {"hold_trace", "combined"}:
                    raise RuntimeError("injected hold trace failure")
                super().record(kind, **fields)

        trace = Trace(io.StringIO())
    old_handlers = {s: probe.signal.getsignal(s) for s in (signal.SIGINT, signal.SIGUSR1)}
    bindings = []
    checkpoints = []
    if console is not None:
        def result_record():
            if cleanup_fault == "owner_stopped":
                raise ValueError("unconsumed terminal receipt")
            return {"state": "failed", "outcome": {"severity": "error", "reason": "startup_failure"}}
        monkeypatch.setattr(console, "result_record", result_record, raising=False)

    def checkpoint(record):
        # Real run_probe with synthetic I/O, not a mocked lifecycle: save
        # only after all device cleanup and host release were attempted.
        assert events[-1] == "close"
        assert trace.gc_observation_end is not None
        assert record["status"] == record["execution_status"] == "ERROR"
        assert record["cleanup"]["transport_release"]["completed"]
        assert record["postprocessing"]["status"] == "PENDING"
        if cleanup_fault == "owner_stopped":
            assert record["automatic_calibration_export_error"] == "ValueError: unconsumed terminal receipt"
        checkpoints.append(record)

    with pytest.raises(probe.LowGainProbeError, match="operator interrupt|injected RX failure") as raised:
        probe.run_probe(args, torque_console=console, debug_trace=trace,
                        console_stop_binding=bindings.append, post_cleanup_checkpoint=checkpoint)
    assert len(checkpoints) == int(console is not None)
    if cleanup_fault:
        if cleanup_fault == "operator_keys":
            assert events.index("keys_close") < events.index("hold")
            assert "injected terminal cleanup failure" in str(raised.value)
        else:
            assert events.index("owner_stopped") < events.index("record_flush") < events.index("hold")
        assert trace.gc_observation_end["cleanup_errors"]
        if cleanup_fault in {"record_flush", "combined"}:
            assert "requires recorded_monotonic_sec" in str(raised.value)
        if cleanup_fault in {"owner_stopped", "combined"}:
            assert "injected console finalization failure" in str(raised.value)
        if cleanup_fault in {"hold_trace", "combined"}:
            assert "injected hold trace failure" in str(raised.value)
    after_enable = events[events.index("enable") + 1:]
    if failure == "persistent_rx_error":
        assert "hold" not in after_enable
        assert after_enable.index("read") < after_enable.index("disable")
    else:
        assert after_enable.index("hold") < after_enable.index("disable")
        assert after_enable.count("read") >= 2
    if cleanup_fault:
        assert after_enable[-3:] == ["restore", "report_off", "close"]
    else:
        assert after_enable[-2:] == ["restore", "close"]
    assert bindings[-1] is None
    assert handlers == old_handlers


@pytest.mark.parametrize("trace_enabled", [False, True])
def test_active_report_warmup_requires_advancing_type24_and_disables_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    trace_enabled: bool,
) -> None:
    events: list[str] = []
    trace_stream = io.StringIO()
    debug_trace = probe._DynamicsTrace(trace_stream) if trace_enabled else None
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test fixture\n", encoding="utf-8")
    config = SimpleNamespace(
        channel="fake-no-hardware",
        joints=[SimpleNamespace(mit_kp=1.0, mit_kd=0.1) for _ in range(6)],
        feedback_timeout_sec=0.05,
    )

    def state(sequence: int) -> B601HardwareState:
        joints = tuple(
            B601JointState(
                name=f"joint{index}",
                motor_id=index,
                feedback_id=0xFD,
                arbitration_id=0x180000FD | (index << 8),
                status_code=0,
                position_rad=q[index - 1],
                velocity_reported=0.0,
                torque_reported_nm=0.0,
                observed_at_sec=float(sequence),
                sample_sequence=sequence,
                valid=True,
                stale=False,
                position_source="robstride_operation_status_cache",
                status_sequence=sequence,
                receive_age_sec=0.001,
                host_receive_freshness_verified=True,
            )
            for index in range(1, 7)
        )
        return B601HardwareState(
            sequence=sequence,
            receive_timestamp_sec=float(sequence),
            joints=joints,
            valid=True,
            stale=False,
            host_receive_freshness_verified=True,
        )

    class Backend:
        sequence = 0

        def connect(self) -> None:
            if trace_enabled:
                context = json.loads(trace_stream.getvalue())
                assert context["kind"] == "calibration_context"
                assert context["schema"] == "b601.command-calibration-context.v1"
                assert context["profile"]["gravity_joint_scales"] == list(probe._DEFAULT_GRAVITY_JOINT_SCALES)
                assert context["profile"]["gravity_joint_bias_nm"] == list(probe._DEFAULT_GRAVITY_JOINT_BIAS_NM)
                assert context["profile"]["driver_kd_nm_s_per_rad"] == list(probe._DEFAULT_DRIVER_KD)
                assert context["profile"]["config_sha256"] == probe.hashlib.sha256(config_path.read_bytes()).hexdigest()
                assert context["physical_load_source"] == "not_independently_measured_run_local_only"
            events.append("connect")

        def set_active_reporting(self, enabled: bool, **_kwargs: object) -> object:
            events.append("report_on" if enabled else "report_off")
            return SimpleNamespace(
                completed=True,
                errors=(),
                operation="active_report",
                requested_joint_names=probe._ALL_JOINTS,
                completed_joint_names=probe._ALL_JOINTS,
                failed_joint_names=(),
            )

        def poll_feedback(self, **kwargs: object) -> B601HardwareState:
            assert kwargs == {"require_host_freshness": True, "prefer_active_report": True}
            self.sequence += 1
            events.append(f"type24_{self.sequence}")
            return state(self.sequence)

        def safe_stop(self, **_kwargs: object) -> object:
            events.append("safe_stop")
            return SimpleNamespace(completed=True, errors=())

        def apply_temporary_driver_torque_limits(self, *_args: object, **_kwargs: object) -> object:
            events.append("limits_on")
            return SimpleNamespace(completed=True, errors=())

        def restore_driver_torque_limits(self) -> object:
            events.append("limits_off")
            return SimpleNamespace(completed=True, errors=())

        def prepare_mit_mode(self, **_kwargs: object) -> object:
            events.append("mit_mode")
            return SimpleNamespace(completed=True, errors=())

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(probe, "B601RSBackend", lambda _config: Backend())
    monkeypatch.setattr(
        probe.B601RSBackendConfig,
        "from_rs_yaml",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        probe,
        "_resolve_model_and_limits",
        lambda _root: (SimpleNamespace(urdf_path=config_path), _fault_hold_limits()),
    )
    monkeypatch.setattr(probe, "_resolve_effort_limits", lambda *_args, **_kwargs: (1.0,) * 6)
    monkeypatch.setattr(probe, "_drive_enable_countdown", lambda _stop: False)
    monkeypatch.setattr(probe, "_operator_notice", lambda _message: None)
    args = probe.build_parser().parse_args(
        [
            "--mode",
            "hold",
            "--config",
            str(config_path),
            "--rebot-source-root",
            str(tmp_path),
            "--feedback-source",
            "active-report",
            "--allow-active-report",
            "--auto-disable-active-report",
            "--allow-motion",
            "--auto-disable",
            "--apply-driver-torque-limits",
            "--confirmation",
            probe.CONFIRMATION,
        ]
    )

    with pytest.raises(probe.LowGainProbeError, match="drive countdown"):
        probe.run_probe(args, debug_trace=debug_trace)

    assert events[:4] == ["connect", "report_on", "type24_1", "type24_2"]
    assert events[-3:] == ["limits_off", "report_off", "close"]


def test_host_observation_payload_preserves_exact_type24_metadata() -> None:
    joints = tuple(
        B601JointState(
            name=f"joint{index}",
            motor_id=index,
            feedback_id=0xFD,
            arbitration_id=0x180000FD | (index << 8),
            status_code=0,
            position_rad=0.01 * index,
            velocity_reported=0.0,
            torque_reported_nm=0.0,
            observed_at_sec=10.0 + index * 0.001,
            sample_sequence=20,
            valid=True,
            stale=False,
            status_sequence=100 + index,
            receive_age_sec=index * 0.001,
            host_receive_freshness_verified=True,
        )
        for index in range(1, 7)
    )
    state = B601HardwareState(
        sequence=20,
        receive_timestamp_sec=10.01,
        joints=joints,
        valid=True,
        stale=False,
        freshness_semantics=(
            "host_receive_sequence_and_age_verified; "
            "device_source_timestamp_unavailable"
        ),
        host_receive_freshness_verified=True,
    )

    result = probe._host_observation_payload(state)

    assert result["schema"] == "b601.type24-host-observation.v1"
    assert result["host_receive_freshness_verified"] is True
    assert result["device_source_timestamp_verified"] is False
    assert tuple(row["name"] for row in result["joints"]) == probe._ALL_JOINTS
    assert tuple(row["status_sequence"] for row in result["joints"]) == tuple(
        range(101, 107)
    )
    assert tuple(row["receive_age_sec"] for row in result["joints"]) == pytest.approx(
        tuple(index * 0.001 for index in range(1, 7))
    )


def test_protective_hold_notice_is_high_visibility_and_log_safe() -> None:
    notice = probe._protective_hold_notice(
        "protective hold degraded",
        "ARM STATE: joint1..joint6 remain ENABLED.",
        "DAMPING: UNAVAILABLE while feedback is untrusted.",
    )

    lines = notice.splitlines()
    assert lines[0] == "=" * 72
    assert lines[-1] == "=" * 72
    assert "[B601] !!! PROTECTIVE HOLD DEGRADED !!!" in lines
    assert "[B601] ARM STATE: joint1..joint6 remain ENABLED." in lines
    assert "[B601] DAMPING: UNAVAILABLE while feedback is untrusted." in lines
    assert "\x1b" not in notice


def test_duration_completion_latches_hold_and_interrupts_active_record() -> None:
    stop = {"requested": False, "full_disable_requested": False}
    capture = probe.GravityCalibrationCapture(20.0)
    record = capture.trigger((0.0,) * 6, elapsed_sec=285.0, sample_start_index=10)

    reason = probe.complete_active_duration(
        stop=stop,
        calibration_capture=capture,
        elapsed_sec=300.0,
        sample_end_index=310,
    )

    assert reason == "duration_complete"
    assert stop == {"requested": True, "full_disable_requested": False}
    assert record is not None
    assert record["completed"] is False
    assert record["completion_reason"] == "probe_duration_complete"
    assert capture.state == "FREE_DRAG"


def test_operator_key_reader_reads_r_without_enter_and_restores_terminal() -> None:
    master_fd, slave_fd = pty.openpty()
    tty_path = Path(os.ttyname(slave_fd))
    original = termios.tcgetattr(slave_fd)
    reader = probe.OperatorKeyReader(tty_path)
    try:
        os.write(master_fd, b"R")
        assert reader.poll_record_request() is True
        assert reader.poll_record_request() is False
    finally:
        reader.close()
        restored = termios.tcgetattr(slave_fd)
        os.close(master_fd)
        os.close(slave_fd)

    assert restored == original


def test_operator_key_reader_ignores_non_r_calibration_keys() -> None:
    master_fd, slave_fd = pty.openpty()
    tty_path = Path(os.ttyname(slave_fd))
    reader = probe.OperatorKeyReader(tty_path)
    try:
        os.write(master_fd, b"pN")
        assert reader.poll_record_request() is False
        os.write(master_fd, b"R")
        assert reader.poll_record_request() is True
    finally:
        reader.close()
        os.close(master_fd)
        os.close(slave_fd)


def test_operator_key_reader_returns_one_normalized_key_batch() -> None:
    master_fd, slave_fd = pty.openpty()
    tty_path = Path(os.ttyname(slave_fd))
    reader = probe.OperatorKeyReader(tty_path)
    try:
        os.write(master_fd, b"AnD")
        assert reader.poll_keys() == frozenset({"a", "n", "d"})
        assert reader.poll_keys() == frozenset()
    finally:
        reader.close()
        os.close(master_fd)
        os.close(slave_fd)


def test_startup_replay_ramps_lock_in_holds_and_hands_back_out() -> None:
    capture = probe.StartupReplayCapture(
        transition_duration_sec=0.5,
        locked_duration_sec=1.0,
    )
    reference = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    zero_velocity = (0.0,) * 6

    event = capture.trigger(
        reference,
        zero_velocity,
        elapsed_sec=3.0,
        sample_index=60,
    )

    assert event is not None
    assert event["state"] == "RAMP_IN"
    assert capture.lock_fraction(3.0) == 0.0
    assert capture.lock_fraction(3.25) == pytest.approx(0.5)
    assert capture.advance(
        reference,
        zero_velocity,
        elapsed_sec=3.5,
        sample_index=70,
    )[0]["state"] == "RELOCKED"
    assert capture.lock_fraction(3.5) == 1.0
    assert capture.advance(
        reference,
        zero_velocity,
        elapsed_sec=4.5,
        sample_index=90,
    )[0]["state"] == "HANDOFF"
    assert capture.lock_fraction(4.75) == pytest.approx(0.5)
    assert capture.advance(
        (0.1, 0.21, 0.3, 0.4, 0.5, 0.6),
        zero_velocity,
        elapsed_sec=5.0,
        sample_index=100,
    )[0]["state"] == "SECOND_ACTIVE"
    assert capture.lock_fraction(5.0) == 0.0

    result = probe.evaluate_startup_replay_capture(
        capture,
        final_qpos_rad=(0.1, 0.22, 0.3, 0.4, 0.5, 0.6),
        sample_count=102,
    )

    assert result["status"] == "PASS"
    assert result["reference_to_second_active_delta_rad"][1] == pytest.approx(0.01)
    assert result["reference_to_final_delta_rad"][1] == pytest.approx(0.02)
    assert result["second_active_observation_count"] == 1
    assert result["displacement_gate_rad"] is None


def test_startup_replay_without_r_is_incomplete_without_a_numeric_gate() -> None:
    capture = probe.StartupReplayCapture(
        transition_duration_sec=0.5,
        locked_duration_sec=1.0,
    )

    result = probe.evaluate_startup_replay_capture(
        capture,
        final_qpos_rad=(0.0,) * 6,
        sample_count=0,
    )

    assert result["status"] == "FAIL"
    assert result["state"] == "WAITING_FOR_R"
    assert result["displacement_gate_rad"] is None
    assert result["reasons"] == ("operator did not trigger startup replay with R",)


def test_shared_gravity_ramp_preserves_vector_and_keeps_damping_live() -> None:
    target = probe.B601MitTorqueBreakdown(
        position_torque_nm=(0.0,) * 6,
        damping_torque_nm=(-0.2,) * 6,
        gravity_torque_nm=(1.0, -2.0, 4.0, -8.0, 0.5, -0.25),
        overlay_requested_nm=(0.0,) * 6,
        overlay_effective_nm=(0.0,) * 6,
        requested_torque_nm=(0.8, -2.2, 3.8, -8.2, 0.3, -0.45),
        overlay_limited=False,
    )

    fraction = probe.gravity_ramp_fraction(elapsed_sec=0.25, rise_time_sec=0.5)
    ramped = probe.apply_shared_gravity_ramp(target, fraction=fraction)

    assert fraction == pytest.approx(0.5)
    assert ramped.gravity_torque_nm == pytest.approx(
        tuple(0.5 * value for value in target.gravity_torque_nm)
    )
    assert ramped.damping_torque_nm == target.damping_torque_nm
    assert ramped.requested_torque_nm == pytest.approx(
        tuple(
            damping + gravity
            for damping, gravity in zip(
                target.damping_torque_nm,
                ramped.gravity_torque_nm,
                strict=True,
            )
        )
    )


def test_startup_handoff_uses_the_same_rise_time_after_gravity() -> None:
    assert probe.startup_handoff_fraction(elapsed_sec=0.0, rise_time_sec=0.5) == 0.0
    assert probe.startup_handoff_fraction(elapsed_sec=0.5, rise_time_sec=0.5) == 0.0
    assert probe.startup_handoff_fraction(elapsed_sec=0.75, rise_time_sec=0.5) == pytest.approx(
        0.5
    )
    assert probe.startup_handoff_fraction(elapsed_sec=2.0, rise_time_sec=0.5) == 1.0


def test_vendor_status_cache_diagnostics_is_passive_and_explicitly_unverified() -> None:
    state = SimpleNamespace(
        velocities_reported=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        torques_reported_nm=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        freshness_verified=False,
    )

    result = probe.vendor_status_cache_diagnostics(state)

    assert result == {
        "velocity_rad_s": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        "torque_nm": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        "complete": True,
        "errors": (),
        "freshness_verified": False,
        "velocity_source": "motorbridge_operation_status_cache_unverified",
        "velocity_role": "diagnostic_only_not_used_for_control",
        "velocity_scale_sign_calibrated": False,
        "observation_timing": None,
        "velocity_semantics": None,
        "control_qvel_source": "normalized_qpos_finite_difference_over_feedback_dt",
        "semantics": (
            "passive_motorbridge_operation_status_cache_unverified; "
            "host_receive_metadata_if_available; device_source_time_unavailable; "
            "no_extra_can_request; not command-response correlated; "
            "diagnostic_only_not_used_for_control"
        ),
    }


@pytest.mark.parametrize("verified", [True, False])
def test_vendor_status_timing_copies_the_admitted_snapshot_without_trusting_velocity(verified):
    joints = tuple(B601JointState(
        name=f"joint{i}", motor_id=i, feedback_id=0xFD, arbitration_id=None,
        status_code=0, position_rad=0.1, velocity_reported=.02,
        torque_reported_nm=.1, observed_at_sec=1 - .001 * i,
        sample_sequence=10, valid=True, stale=False,
        position_source="robstride_operation_status_cache",
        status_sequence=100 + i if verified else None,
        receive_age_sec=.001 * i if verified else None,
        host_receive_freshness_verified=verified,
    ) for i in range(1, 7))
    state = B601HardwareState(
        sequence=10, receive_timestamp_sec=1, joints=joints, valid=True, stale=False,
        host_receive_freshness_verified=verified, velocity_semantics="test_cached_state",
    )
    result = probe.vendor_status_cache_diagnostics(state)
    timing = result["observation_timing"]
    assert timing["snapshot_observed_at_sec"] == 1
    assert timing["host_receive_freshness_verified"] is verified
    assert timing["device_source_timestamp_available"] is False
    assert len(timing["joint_observations"]) == 6
    for original, copied in zip(joints, timing["joint_observations"], strict=True):
        assert copied["status_sequence"] == original.status_sequence
        assert copied["receive_age_sec"] == original.receive_age_sec
        assert copied["observed_at_sec"] == original.observed_at_sec
    assert result["velocity_scale_sign_calibrated"] is False
    assert result["velocity_role"] == "diagnostic_only_not_used_for_control"
    assert result["control_qvel_source"] == "normalized_qpos_finite_difference_over_feedback_dt"
    assert result["velocity_semantics"] == "test_cached_state"
    assert state.joints is joints


def _velocity_observation_state(snapshot_time, joint_time, sequence):
    return B601HardwareState(
        sequence=sequence, receive_timestamp_sec=snapshot_time, valid=True, stale=False,
        host_receive_freshness_verified=True,
        joints=tuple(B601JointState(
            name=f"joint{i}", motor_id=i, feedback_id=0xFD, arbitration_id=(24<<24)|i,
            status_code=0, position_rad=0., velocity_reported=99., torque_reported_nm=0.,
            observed_at_sec=joint_time, sample_sequence=sequence, valid=True, stale=False,
            position_source="robstride_operation_status_cache", status_sequence=sequence,
            receive_age_sec=snapshot_time-joint_time, host_receive_freshness_verified=True,
        ) for i in range(1,7)),
    )


def _joint_receive_velocity_console():
    return SimpleNamespace(config=SimpleNamespace(feedback_velocity_timebase="joint_receive"),
        settings=SimpleNamespace(staged_assist=SimpleNamespace(
            motion_control=SimpleNamespace(max_control_gap_sec=.04))))


@pytest.mark.parametrize('execution,initial,severity,state,expected_code,expected_status',[
    ('STOPPED_BY_OPERATOR','FAIL','warning','failed',0,'STOPPED_WITH_WARNINGS'),
    ('STOPPED_BY_OPERATOR','FAIL','info','complete',0,'STOPPED'),
    ('STOPPED_BY_OPERATOR','FAIL','error','failed',1,'FAIL'),
    ('STOPPED_BY_OPERATOR','FAIL','info','onset_scan',1,'FAIL'),
    ('FAILED','FAIL','warning','failed',1,'FAIL'),
    ('ERROR','ERROR','warning','failed',2,'ERROR'),
])
def test_automatic_result_separates_clean_stop_from_acceptance_without_hiding_errors(execution,initial,severity,state,expected_code,expected_status):
    payload={'execution_status':execution,'status':initial,'acceptance_status':'FAIL'}
    record={'state':state,'outcome':{'severity':severity,'scope':'acquisition'}}
    assert probe._finalize_automatic_result(payload,record)==expected_code
    assert payload['status']==expected_status
    assert payload['acceptance_status']=='FAIL'
    assert payload['acquisition_outcome']==record['outcome']


def test_automatic_postrun_timing_quality_is_a_visible_warning_not_a_live_guard_override():
    payload={'status':'FAIL','execution_status':'STOPPED_BY_OPERATOR','acceptance_status':'FAIL',
        'frequency_acceptance':{'tx':{'passed':False},'feedback':{'passed':True}}}
    record={'state':'complete','outcome':{'severity':'info','scope':'acquisition'}}
    assert probe._finalize_automatic_result(payload,record)==0
    assert payload['status']=='STOPPED_WITH_WARNINGS'
    assert payload['execution_warnings']==[{'scope':'timing_quality','reason':'tx_frequency_acceptance_not_met'}]
    assert payload['frequency_acceptance']['tx']['passed'] is False


def test_automatic_clean_stop_classification_cannot_hide_a_degraded_protective_hold():
    payload={'status':'FAIL','execution_status':'STOPPED_BY_OPERATOR','acceptance_status':'FAIL',
        'protective_hold':{'degraded':{'entry_count':1,'recovery_count':1}}}
    record={'state':'failed','outcome':{'severity':'warning','scope':'acquisition'}}
    assert probe._finalize_automatic_result(payload,record)==1
    assert payload['status']=='FAIL' and payload['execution_status']=='FAILED'


@pytest.mark.parametrize('reason', ['page_unload', 'operator_withdraw'])
@pytest.mark.parametrize('timing_warning', [False, True])
def test_automatic_normal_second_stop_is_not_a_degraded_fault(reason, timing_warning):
    payload = {'status': 'FAIL', 'execution_status': 'STOPPED_BY_OPERATOR',
        'acceptance_status': 'FAIL', 'stop_reason': 'operator_interrupt',
        'protective_hold': {'degraded': {'entry_count': 0,
            'escalation_reason': 'explicit second stop request'}},
        'frequency_acceptance': {'tx': {'passed': not timing_warning}}}
    record = {'state': 'cancelled', 'acquisition_complete': False,
        'outcome': {'severity': 'info', 'scope': 'acquisition', 'reason': reason}}
    assert probe._finalize_automatic_result(payload, record) == 0
    assert payload['status'] == ('STOPPED_WITH_WARNINGS' if timing_warning else 'STOPPED')
    assert payload['execution_status'] == 'STOPPED_BY_OPERATOR'
    assert payload['acceptance_status'] == 'FAIL'
    assert not record['acquisition_complete']
    assert payload['acquisition_outcome'] == record['outcome']
    assert payload['protective_hold']['degraded']['escalation_reason'] == 'explicit second stop request'


@pytest.mark.parametrize('entries,escalation', [
    (1, 'explicit second stop request'),
    (0, 'no verified degraded hold cache'),
    (0, 'live and cached all-six-joint hold dispatch failed after retry'),
    (0, 'unknown future failure'),
])
def test_automatic_second_stop_fix_keeps_real_and_unknown_faults_failed(entries, escalation):
    payload = {'status': 'FAIL', 'execution_status': 'STOPPED_BY_OPERATOR',
        'protective_hold': {'degraded': {'entry_count': entries, 'escalation_reason': escalation}}}
    record = {'state': 'cancelled', 'outcome': {'severity': 'info', 'scope': 'acquisition'}}
    assert probe._finalize_automatic_result(payload, record) == 1
    assert payload['status'] == 'FAIL'
    assert payload['execution_status'] == 'FAILED'


@pytest.mark.parametrize('status,execution,code', [('FAIL', 'FAILED', 1), ('ERROR', 'ERROR', 2)])
def test_automatic_normal_second_stop_cannot_promote_upstream_cleanup_or_control_failure(status, execution, code):
    payload = {'status': status, 'execution_status': execution,
        'cleanup': {'device_cleanup_confirmed': False, 'errors': ['disable not confirmed']},
        'protective_hold': {'degraded': {'entry_count': 0,
            'escalation_reason': 'explicit second stop request'}}}
    record = {'state': 'cancelled', 'outcome': {'severity': 'info', 'scope': 'acquisition'}}
    assert probe._finalize_automatic_result(payload, record) == code
    assert payload['status'] == status and payload['execution_status'] == execution
    assert not payload['cleanup']['device_cleanup_confirmed']


def test_receive_interval_above_watchdog_is_valid_when_paired_snapshot_and_ages_are_fresh():
    # Preserved J1 timing from the field failure: advancing, fresh metadata,
    # not a backwards clock. No snapshot/vendor speed fallback is allowed.
    snapshot_dt=.03896361100487411
    previous_age=.004892083
    current_age=.003755465
    previous=_velocity_observation_state(100., 100.-previous_age, 10)
    current=_velocity_observation_state(100.+snapshot_dt,100.+snapshot_dt-current_age,11)
    delta=.000383496
    diagnostic=probe.feedback_velocity_timebase_diagnostics(current,previous,
        qpos_rad=(delta,)*6,previous_qpos_rad=(0.,)*6,snapshot_dt_sec=snapshot_dt,
        raw_qvel_rad_s=(99.,)*6)
    assert diagnostic['joint_observations'][0]['joint_receive_dt_sec'] > .04
    velocity=probe._torque_console_feedback_velocity(_joint_receive_velocity_console(),(99.,)*6,diagnostic)
    assert velocity == pytest.approx((delta/(snapshot_dt+previous_age-current_age),)*6)
    assert max(velocity) < .01


@pytest.mark.parametrize('case',['snapshot_gap','old_previous','old_current','identity','nonadvancing'])
def test_receive_clock_correction_keeps_watchdog_freshness_and_identity_independent(case):
    previous=_velocity_observation_state(1.,.99,10)
    current=_velocity_observation_state(1.01,1.,11)
    diagnostic=probe.feedback_velocity_timebase_diagnostics(current,previous,
        qpos_rad=(0.,)*6,previous_qpos_rad=(0.,)*6,snapshot_dt_sec=.01,raw_qvel_rad_s=(0.,)*6)
    if case == 'snapshot_gap':diagnostic['snapshot_dt_sec']=.040001
    elif case == 'old_previous':diagnostic['joint_observations'][0]['previous_receive_age_sec']=.040001
    elif case == 'old_current':diagnostic['joint_observations'][0]['receive_age_sec']=.040001
    elif case == 'identity':diagnostic['joint_observations'][0]['joint_receive_dt_sec']=.02
    else:diagnostic['joint_observations'][0]['joint_receive_dt_sec']=0.
    with pytest.raises(probe.LowGainProbeError,match='joint receive velocity'):
        probe._torque_console_feedback_velocity(_joint_receive_velocity_console(),(0.,)*6,diagnostic)


@pytest.mark.parametrize('joint_dt,delta,expected_block',[
    (.018227736,-.0007669925689697266,False),
    (.009,-.0007669925689697266,True),
    (.007536903955042362,-.0003834962844848633,True),
])
def test_joint_receive_velocity_aligns_recorded_two_count_guard_without_filtering(joint_dt,delta,expected_block):
    from robot_motion.rebot_b601_rs.joint_motion import (
        B601JointMotionConfig,
        B601JointMotionTracker,
        B601WindowStopEvidence,
    )
    previous=_velocity_observation_state(1.,.99,10)
    current=_velocity_observation_state(1.011199411,.99+joint_dt,11)
    snapshot=(0.,0.,delta/.011199411,0.,0.,0.)
    q=(0.,0.,.1+delta,0.,0.,0.)
    diagnostic=probe.feedback_velocity_timebase_diagnostics(current,previous,
        qpos_rad=q,previous_qpos_rad=(0.,0.,.1,0.,0.,0.),snapshot_dt_sec=.011199411,
        raw_qvel_rad_s=snapshot)
    corrected=probe._torque_console_feedback_velocity(_joint_receive_velocity_console(),snapshot,diagnostic)
    assert corrected[2]==pytest.approx(delta/joint_dt)
    assert (abs(snapshot[2])>.05) is (abs(delta)>.0004)
    window=(0.,0.,-.0157743905,0.,0.,0.)
    tracker=B601JointMotionTracker(B601JointMotionConfig(windowed_soft_velocity=True,
        position_resolution_rad=.0004,responsive_tracking_brake=True))
    tracker.start(now=1.,q=.1,target=.08)
    frame=tracker.update(now=1.011199411,q=q[2],raw_v=corrected[2],control_v=window[2],
        control_sequence=1,authority=1.,kp=12.,kd=1.,stop_error_rad=.003,
        stop_velocity_rad_s=.005,stop_dwell_sec=.15,
        stop_evidence=B601WindowStopEvidence(1.011199411,q,corrected,window,
            (0.,0.,.001534,0.,0.,0.),.08,.05))
    assert (frame.motion_evidence_block_reason=='raw_velocity_hard_guard') is expected_block
    assert frame.phase==('authority_exhausted' if expected_block else 'track')


@pytest.mark.parametrize('case',['missing','incomplete','unverified','wrong_joint','gap','age','nonfinite'])
def test_joint_receive_velocity_fails_closed_without_snapshot_or_vendor_fallback(case):
    previous=_velocity_observation_state(1.,.99,10)
    current=_velocity_observation_state(1.01,1.,11)
    raw=(.168,)*6
    diagnostic=probe.feedback_velocity_timebase_diagnostics(current,previous,
        qpos_rad=(0.,)*6,previous_qpos_rad=(0.,)*6,snapshot_dt_sec=.01,raw_qvel_rad_s=raw)
    if case=='missing': diagnostic=None
    elif case=='incomplete': diagnostic['joint_observations'].pop()
    else:
        edits={'unverified':{'comparison_valid':False},'wrong_joint':{'joint_name':'joint7'},
            'gap':{'joint_receive_dt_sec':.041},'age':{'receive_age_sec':.041},
            'nonfinite':{'delta_qpos_rad':float('nan')}}[case]
        diagnostic['joint_observations'][2].update(edits)
    with pytest.raises(probe.LowGainProbeError,match='joint receive velocity'):
        probe._torque_console_feedback_velocity(_joint_receive_velocity_console(),raw,diagnostic)
    # Absent explicit selection keeps the original snapshot path unchanged.
    assert probe._torque_console_feedback_velocity(SimpleNamespace(),raw,diagnostic)==raw


def test_velocity_timebase_comparison_preserves_snapshot_speed_and_uses_only_host_receive_metadata():
    previous = _velocity_observation_state(1., .99, 10)
    current = _velocity_observation_state(1.00228362, 1., 11)
    count = .0003834962844848633
    q = (0.,0.,-count,0.,0.,0.)
    raw = (0.,0.,-count/.00228362,0.,0.,0.)
    result = probe.feedback_velocity_timebase_diagnostics(current,previous,
        qpos_rad=q,previous_qpos_rad=(0.,)*6,snapshot_dt_sec=.00228362,raw_qvel_rad_s=raw)
    j3 = result['joint_observations'][2]
    assert j3['raw_snapshot_qvel_rad_s'] == raw[2]
    assert j3['joint_receive_dt_sec'] == pytest.approx(.01)
    assert j3['diagnostic_joint_qvel_rad_s'] == pytest.approx(-count/.01)
    assert j3['comparison_valid'] and j3['status_sequence'] == 11 and j3['previous_status_sequence'] == 10
    assert result['role'] == 'diagnostic_only_not_used_for_control'
    assert result['device_source_timestamp_available'] is False
    assert current.joints[2].velocity_reported == 99.  # Never used in the estimate.


@pytest.mark.parametrize('case,reason',[
    ('missing','paired_metadata_unavailable'),
    ('unaligned','snapshot_history_not_aligned'),
    ('unverified','host_metadata_unverified'),
    ('stale','host_metadata_unverified'),
    ('wrong_type','host_metadata_unverified'),
    ('time_missing','joint_receive_time_unavailable'),
    ('time_reversed','joint_receive_time_not_advanced'),
    ('sequence','joint_status_sequence_not_advanced'),
    ('inconsistent_age','inconsistent_host_receive_metadata'),
])
def test_velocity_timebase_invalid_metadata_never_falls_back_to_raw_or_vendor_speed(case,reason):
    previous = _velocity_observation_state(1., .99, 10)
    current = _velocity_observation_state(1.01, 1., 11)
    dt = .01
    if case == 'missing': previous = None
    elif case == 'unaligned': dt = .02
    elif case == 'unverified': current = replace(current,host_receive_freshness_verified=False)
    elif case == 'stale': current = replace(current,stale=True)
    else:
        edits = {
            'wrong_type':{'arbitration_id':(2<<24)|3},
            'time_missing':{'observed_at_sec':None},
            'time_reversed':{'observed_at_sec':.98,'receive_age_sec':.03},
            'sequence':{'status_sequence':10},
            'inconsistent_age':{'receive_age_sec':.02},
        }[case]
        current = replace(current,joints=(*current.joints[:2],replace(current.joints[2],**edits),*current.joints[3:]))
    result = probe.feedback_velocity_timebase_diagnostics(current,previous,
        qpos_rad=(0.,)*6,previous_qpos_rad=(0.,)*6,snapshot_dt_sec=dt,raw_qvel_rad_s=(.168,)*6)
    j3 = result['joint_observations'][2]
    assert j3['reason'] == reason and not j3['comparison_valid']
    assert j3['diagnostic_joint_qvel_rad_s'] is None
    assert j3['raw_snapshot_qvel_rad_s'] == .168
    with pytest.raises(probe.LowGainProbeError,match='joint receive velocity'):
        probe._torque_console_feedback_velocity(_joint_receive_velocity_console(),(.168,)*6,result)


def test_velocity_timebase_wiring_records_fault_and_quarantine_without_rebasing_the_observer():
    source = inspect.getsource(probe.run_probe)
    tree = ast.parse(source)
    assignments = [node for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='previous_velocity_diagnostic_state' for t in node.targets)]
    assert len(assignments)==1 and ast.unparse(assignments[0].value)=='state'
    assert source.index('continue',source.index('if motion_rejection_action == "quarantine"')) < source.index('previous_velocity_diagnostic_state = state')
    guard = next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='record' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='torque_console_guard')
    fields = {n.arg:ast.unparse(n.value) for n in guard.keywords if n.arg}
    assert fields['velocity_timebase']=='velocity_timebase'
    assert fields['vendor_status_diagnostic_only']=='velocity_vendor_status'
    assert fields['previous_qpos_rad']=='previous_q'
    step = next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='torque_console' and n.func.attr=='step')
    assert next(n.value.id for n in step.keywords if n.arg=='qvel_rad_s')=='console_qvel'
    selection=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='console_qvel' for t in n.targets))
    assert ast.unparse(selection.value)=='_torque_console_feedback_velocity(torque_console, qvel, velocity_timebase)'
    rejection=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='record' and n.args and isinstance(n.args[0],ast.Constant)
        and n.args[0].value=='torque_console_velocity_rejected')
    assert next(n.value.value for n in rejection.keywords if n.arg=='command_dispatched') is False
    fields = {keyword.arg for keyword in rejection.keywords}
    assert {"cycle_timing", "feedback_read_diagnostics", "requested_schedule_wait_sec",
            "previous_trace_record_timing"} <= fields
    assert selection.lineno < step.lineno


def test_vendor_status_cache_diagnostics_does_not_fail_when_cache_is_absent() -> None:
    result = probe.vendor_status_cache_diagnostics(SimpleNamespace())

    assert result["complete"] is False
    assert result["velocity_rad_s"] is None
    assert result["torque_nm"] is None
    assert result["velocity_role"] == "diagnostic_only_not_used_for_control"
    assert result["control_qvel_source"] == (
        "normalized_qpos_finite_difference_over_feedback_dt"
    )
    assert result["errors"] == (
        "vendor_status_velocity_rad_s: unavailable",
        "vendor_status_torque_nm: unavailable",
    )


def test_startup_diagnostic_labels_finite_difference_velocity_contract() -> None:
    result = probe.build_startup_diagnostic_observation(
        phase="ACTIVE",
        state=SimpleNamespace(
            sequence=4,
            receive_timestamp_sec=12.5,
            velocities_reported=(99.0,) * 6,
            torques_reported_nm=(0.0,) * 6,
            freshness_verified=False,
        ),
        qpos_rad=(0.1,) * 6,
        qvel_fd_rad_s=(0.02,) * 6,
    )

    assert result["qvel_source"] == (
        "normalized_qpos_finite_difference_over_feedback_dt"
    )
    assert result["vendor_status"]["velocity_role"] == (
        "diagnostic_only_not_used_for_control"
    )
    assert result["vendor_status"]["control_qvel_source"] == result["qvel_source"]


def test_startup_diagnostics_reports_observed_deltas_without_a_numeric_gate() -> None:
    samples = [
        {
            "startup_state": "GRAVITY_RAMP",
            "elapsed_sec": 0.25,
            "qpos_rad": (0.0,) * 6,
        },
        {
            "startup_state": "ACTIVE",
            "elapsed_sec": 1.0,
            "qpos_rad": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        },
    ]

    result = probe.summarize_startup_diagnostics(
        pre_enable_qpos_rad=(-0.1,) * 6,
        lock_reference_qpos_rad=(0.0,) * 6,
        pre_control_observations=({"phase": "PRE_ENABLE_FEEDBACK"},),
        samples=samples,
    )

    assert result["acceptance_status"] == "CAPTURED"
    assert result["extra_can_requests"] == 0
    assert result["pre_enable_to_post_enable_delta_rad"] == pytest.approx((0.1,) * 6)
    assert result["post_enable_to_first_active_delta_rad"] == pytest.approx(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    )
    assert result["pre_enable_to_first_active_delta_rad"] == pytest.approx(
        (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    )
    assert result["displacement_gate_rad"] is None


def test_startup_diagnostics_is_incomplete_without_active_sample() -> None:
    result = probe.summarize_startup_diagnostics(
        pre_enable_qpos_rad=None,
        lock_reference_qpos_rad=None,
        pre_control_observations=(),
        samples=(),
    )

    assert result["acceptance_status"] == "INCOMPLETE"
    assert result["first_active_qpos_rad"] is None
    assert result["displacement_gate_rad"] is None


def test_startup_lock_profile_keeps_yaml_gains_driver_local() -> None:
    profile = probe.build_startup_lock_profile(
        mit_kp_nm_per_rad=(50.0, 150.0, 150.0, 50.0, 50.0, 50.0),
        mit_kd_nm_s_per_rad=(3.0, 10.0, 10.0, 5.0, 4.0, 4.0),
        torque_limits_nm=(9.0, 9.0, 9.0, 3.5, 3.5, 3.5),
        torque_rise_time_sec=0.5,
        velocity_limit_rad_s=5.0,
    )

    assert profile.position_kp_nm_per_rad == (0.0,) * 6
    assert profile.damping_kd_nm_s_per_rad == (0.0,) * 6
    assert profile.max_torque_rate_nm_s == (18.0, 18.0, 18.0, 7.0, 7.0, 7.0)
    assert profile.max_abs_overlay_torque_nm == (0.0,) * 6
    assert profile.metadata["gain_source"] == "rebotarm_rs_yaml.MIT"
    assert profile.metadata["vendor_kp_nm_per_rad"] == (
        50.0,
        150.0,
        150.0,
        50.0,
        50.0,
        50.0,
    )
    assert profile.metadata["vendor_kd_nm_s_per_rad"] == (
        3.0,
        10.0,
        10.0,
        5.0,
        4.0,
        4.0,
    )


def test_startup_lock_host_breakdown_contains_gravity_only() -> None:
    class FakeModel:
        def arm_generalized_gravity(self, _qpos_rad):
            return (1.0, -2.0, 3.0, -4.0, 5.0, -6.0)

    profile = probe.build_startup_lock_profile(
        mit_kp_nm_per_rad=(50.0, 150.0, 150.0, 50.0, 50.0, 50.0),
        mit_kd_nm_s_per_rad=(3.0, 10.0, 10.0, 5.0, 4.0, 4.0),
        torque_limits_nm=(9.0, 9.0, 9.0, 3.5, 3.5, 3.5),
        torque_rise_time_sec=0.5,
        velocity_limit_rad_s=5.0,
    )
    reference = probe.build_reference(
        sequence=1,
        timestamp_sec=1.0,
        qpos_rad=(0.5,) * 6,
        qvel_rad_s=(0.0,) * 6,
    )

    breakdown = probe.compute_software_torque_breakdown(
        FakeModel(),
        reference=reference,
        measured_qpos_rad=(0.1,) * 6,
        measured_qvel_rad_s=(2.0,) * 6,
        profile=profile,
        gravity_scale=0.8,
        gravity_joint_scales=(1.0, 0.95, 1.0, 0.5, 1.2, 1.0),
    )

    assert breakdown.position_torque_nm == (0.0,) * 6
    assert breakdown.damping_torque_nm == (0.0,) * 6
    assert breakdown.gravity_torque_nm == pytest.approx(
        (0.8, -1.52, 2.4, -1.6, 4.8, -4.8)
    )
    assert breakdown.requested_torque_nm == breakdown.gravity_torque_nm


def test_gravity_compensation_applies_offset_bias_and_directional_friction() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.qpos: tuple[float, ...] | None = None

        def arm_generalized_gravity(self, qpos_rad):
            self.qpos = tuple(qpos_rad)
            return (1.0, -2.0, 3.0, -4.0, 5.0, -6.0)

    model = FakeModel()
    profile = probe.build_software_profile(
        kp=0.0,
        kd=0.0,
        torque_limits_nm=(20.0,) * 6,
        torque_rise_time_sec=0.5,
        velocity_limit_rad_s=5.0,
    )
    reference = probe.build_reference(
        sequence=1,
        timestamp_sec=1.0,
        qpos_rad=(0.0,) * 6,
    )

    breakdown = probe.compute_software_torque_breakdown(
        model,
        reference=reference,
        measured_qpos_rad=(0.1,) * 6,
        measured_qvel_rad_s=(0.03, -0.03, 0.01, -0.01, 0.0, 0.02),
        profile=profile,
        gravity_scale=0.5,
        gravity_joint_scales=(1.0, 0.5, 1.0, 0.5, 1.0, 0.5),
        gravity_joint_offsets_rad=(0.01, -0.02, 0.03, -0.04, 0.05, -0.06),
        gravity_joint_bias_nm=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        gravity_coulomb_friction_nm=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
        gravity_coulomb_velocity_threshold_rad_s=0.02,
    )

    assert model.qpos == pytest.approx((0.11, 0.08, 0.13, 0.06, 0.15, 0.04))
    assert breakdown.gravity_torque_nm == pytest.approx(
        (0.61, -0.32, 1.8, -0.6, 3.0, -0.9)
    )
    assert breakdown.requested_torque_nm == breakdown.gravity_torque_nm


def test_same_tick_lock_composition_reuses_calibrated_gravity_without_cross_tick_cache() -> None:
    class Model:
        def __init__(self):
            self.calls = []

        def arm_generalized_gravity(self, qpos):
            self.calls.append(tuple(qpos))
            return tuple((i + 1) * math.sin(q) for i, q in enumerate(qpos))

    kwargs = {
        "gravity_scale": .8,
        "gravity_joint_scales": (1., .96, 1., .9, 1., 1.),
        "gravity_joint_offsets_rad": (.01, -.02, .03, -.04, .05, -.06),
        "gravity_joint_bias_nm": (0., -.22, -.11, -.08, 0., 0.),
        "gravity_coulomb_friction_nm": (.01, .02, .03, .04, .05, .06),
        "gravity_coulomb_velocity_threshold_rad_s": .02,
    }
    profile = probe.build_software_profile(
        kp=2., kd=.3, torque_limits_nm=(20.,) * 6,
        torque_rise_time_sec=.5, velocity_limit_rad_s=5.,
        overlay_limits_nm=(.2,) * 6,
    )
    lock_profile = probe.build_startup_lock_profile(
        mit_kp_nm_per_rad=(50.,) * 6, mit_kd_nm_s_per_rad=(4.,) * 6,
        torque_limits_nm=(20.,) * 6, torque_rise_time_sec=.5,
        velocity_limit_rad_s=5.,
    )
    optimized_model, baseline_model = Model(), Model()
    for i in range(3):
        q = tuple(.1 * (j + i + 1) for j in range(6))
        v = (.03, -.03, .01, -.01, 0., .02)
        reference = probe.build_reference(sequence=i, timestamp_sec=1.+i, qpos_rad=(.2,) * 6,
                                          joint_torque_overlay_nm=(.1,) * 6)
        lock_ref = probe.build_reference(sequence=i, timestamp_sec=1.+i, qpos_rad=(.1,) * 6)
        active = probe.compute_software_torque_breakdown(
            optimized_model, reference=reference, measured_qpos_rad=q,
            measured_qvel_rad_s=v, profile=profile, **kwargs,
        )
        reused_lock = probe.compute_b601_mit_torque(
            lock_ref, lock_profile, measured_qpos_rad=q,
            measured_qvel_rad_s=v, gravity_torque_nm=active.gravity_torque_nm,
        )
        baseline_lock = probe.compute_software_torque_breakdown(
            baseline_model, reference=lock_ref, measured_qpos_rad=q,
            measured_qvel_rad_s=v, profile=lock_profile, **kwargs,
        )
        assert reused_lock == baseline_lock
        assert len(optimized_model.calls) == i + 1
        assert optimized_model.calls[-1] == pytest.approx(tuple(a+b for a,b in zip(q,kwargs["gravity_joint_offsets_rad"])))
        for fraction in (0., .25, .5, 1.):
            assert probe.blend_startup_lock_torque_breakdown(
                reused_lock, active, gravity_fraction=1., handoff_fraction=fraction,
            ) == probe.blend_startup_lock_torque_breakdown(
                baseline_lock, active, gravity_fraction=1., handoff_fraction=fraction,
            )


def test_driver_local_lock_gain_blend_has_clean_active_endpoints() -> None:
    lock_kp = (50.0, 150.0, 150.0, 50.0, 50.0, 50.0)
    lock_kd = (3.0, 10.0, 10.0, 5.0, 4.0, 4.0)
    active_kd = (1.0,) * 6

    active_kp, active_driver_kd = probe.blend_driver_local_lock_gains(
        lock_kp_nm_per_rad=lock_kp,
        lock_kd_nm_s_per_rad=lock_kd,
        active_kd_nm_s_per_rad=active_kd,
        lock_fraction=0.0,
    )
    halfway_kp, halfway_kd = probe.blend_driver_local_lock_gains(
        lock_kp_nm_per_rad=lock_kp,
        lock_kd_nm_s_per_rad=lock_kd,
        active_kd_nm_s_per_rad=active_kd,
        lock_fraction=0.5,
    )
    locked_kp, locked_kd = probe.blend_driver_local_lock_gains(
        lock_kp_nm_per_rad=lock_kp,
        lock_kd_nm_s_per_rad=lock_kd,
        active_kd_nm_s_per_rad=active_kd,
        lock_fraction=1.0,
    )

    assert active_kp == (0.0,) * 6
    assert active_driver_kd == active_kd
    assert halfway_kp == pytest.approx(tuple(value * 0.5 for value in lock_kp))
    assert halfway_kd == pytest.approx(
        tuple((start + end) * 0.5 for start, end in zip(active_kd, lock_kd))
    )
    assert locked_kp == lock_kp
    assert locked_kd == lock_kd


def test_driver_local_lock_command_has_continuous_gain_and_velocity_endpoints() -> None:
    limits = ((-1.0, 1.0),) * 6
    lock_reference = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    active_reference = (-0.1, -0.2, -0.3, -0.4, -0.5, -0.6)
    active_velocity = (0.2,) * 6
    lock_kp = (50.0, 150.0, 150.0, 50.0, 50.0, 50.0)
    lock_kd = (3.0, 10.0, 10.0, 5.0, 4.0, 4.0)
    active_kd = (1.0,) * 6

    active = probe.resolve_driver_local_lock_command(
        lock_reference_qpos_rad=lock_reference,
        active_reference_qpos_rad=active_reference,
        active_reference_qvel_rad_s=active_velocity,
        position_limits_rad=limits,
        lock_kp_nm_per_rad=lock_kp,
        lock_kd_nm_s_per_rad=lock_kd,
        active_kd_nm_s_per_rad=active_kd,
        lock_fraction=0.0,
    )
    halfway = probe.resolve_driver_local_lock_command(
        lock_reference_qpos_rad=lock_reference,
        active_reference_qpos_rad=active_reference,
        active_reference_qvel_rad_s=active_velocity,
        position_limits_rad=limits,
        lock_kp_nm_per_rad=lock_kp,
        lock_kd_nm_s_per_rad=lock_kd,
        active_kd_nm_s_per_rad=active_kd,
        lock_fraction=0.5,
    )
    locked = probe.resolve_driver_local_lock_command(
        lock_reference_qpos_rad=lock_reference,
        active_reference_qpos_rad=active_reference,
        active_reference_qvel_rad_s=active_velocity,
        position_limits_rad=limits,
        lock_kp_nm_per_rad=lock_kp,
        lock_kd_nm_s_per_rad=lock_kd,
        active_kd_nm_s_per_rad=active_kd,
        lock_fraction=1.0,
    )

    assert active.reference_qpos_rad == active_reference
    assert active.reference_qvel_rad_s == active_velocity
    assert active.kp_nm_per_rad == (0.0,) * 6
    assert active.kd_nm_s_per_rad == active_kd
    assert active.position_lock_joint_names == ()
    assert halfway.reference_qpos_rad == lock_reference
    assert halfway.reference_qvel_rad_s == pytest.approx((0.1,) * 6)
    assert halfway.kp_nm_per_rad == pytest.approx(
        tuple(value * 0.5 for value in lock_kp)
    )
    assert halfway.position_lock_joint_names == tuple(probe._ALL_JOINTS)
    assert locked.reference_qpos_rad == lock_reference
    assert locked.reference_qvel_rad_s == (0.0,) * 6
    assert locked.kp_nm_per_rad == lock_kp
    assert locked.kd_nm_s_per_rad == lock_kd


def test_driver_local_lock_suppresses_kp_when_reference_needs_urdf_clamp() -> None:
    limits = (
        (-2.8, 2.8),
        (0.0, 3.14),
        (0.0, 3.14),
        (-math.pi / 2.0, math.pi / 2.0),
        (-1.57, 1.57),
        (-3.14, 3.14),
    )
    command = probe.resolve_driver_local_lock_command(
        lock_reference_qpos_rad=(0.0, -0.004, 0.5, -1.62, 0.0, 0.0),
        active_reference_qpos_rad=(0.0,) * 6,
        active_reference_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=limits,
        lock_kp_nm_per_rad=(50.0, 150.0, 150.0, 50.0, 50.0, 50.0),
        lock_kd_nm_s_per_rad=(3.0, 10.0, 10.0, 5.0, 4.0, 4.0),
        active_kd_nm_s_per_rad=(1.0,) * 6,
        lock_fraction=1.0,
    )

    assert command.reference_qpos_rad == pytest.approx(
        (0.0, 0.0, 0.5, -math.pi / 2.0, 0.0, 0.0)
    )
    assert command.kp_nm_per_rad == (50.0, 0.0, 150.0, 0.0, 50.0, 50.0)
    assert command.kd_nm_s_per_rad == (3.0, 10.0, 10.0, 5.0, 4.0, 4.0)
    assert command.position_lock_joint_names == (
        "joint1",
        "joint3",
        "joint5",
        "joint6",
    )
    assert command.position_lock_suppressed_joint_names == ("joint2", "joint4")


def test_gravity_calibration_active_endpoint_keeps_software_residual_semantics() -> None:
    profile = probe.build_software_profile(
        kp=5.0,
        kd=0.0,
        torque_limits_nm=(14.4, 14.4, 14.4, 5.6, 5.6, 5.6),
        torque_rise_time_sec=0.5,
        velocity_limit_rad_s=5.0,
    )
    command = probe.resolve_driver_local_lock_command(
        lock_reference_qpos_rad=(0.0,) * 6,
        active_reference_qpos_rad=(0.2,) * 6,
        active_reference_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=((-1.0, 1.0),) * 6,
        lock_kp_nm_per_rad=(50.0, 150.0, 150.0, 50.0, 50.0, 50.0),
        lock_kd_nm_s_per_rad=(3.0, 10.0, 10.0, 5.0, 4.0, 4.0),
        active_kd_nm_s_per_rad=(1.0,) * 6,
        lock_fraction=0.0,
    )

    assert profile.position_kp_nm_per_rad == (5.0,) * 6
    assert profile.damping_kd_nm_s_per_rad == (0.0,) * 6
    assert command.reference_qpos_rad == (0.2,) * 6
    assert command.kp_nm_per_rad == (0.0,) * 6
    assert command.kd_nm_s_per_rad == (1.0,) * 6
    assert command.position_lock_joint_names == ()


def test_protective_hold_uses_zero_position_stiffness_and_one_damping_owner() -> None:
    startup = probe.build_startup_lock_profile(
        mit_kp_nm_per_rad=(50.0, 150.0, 150.0, 50.0, 50.0, 50.0),
        mit_kd_nm_s_per_rad=(3.0, 10.0, 10.0, 5.0, 4.0, 4.0),
        torque_limits_nm=(9.0, 9.0, 9.0, 3.5, 3.5, 3.5),
        torque_rise_time_sec=0.5,
        velocity_limit_rad_s=5.0,
    )

    hold = probe.build_protective_hold_profile(
        startup,
        driver_kd_nm_s_per_rad=(0.0, 0.5, 0.5, 0.0, 0.0, 0.5),
    )

    assert hold.position_kp_nm_per_rad == (0.0,) * 6
    assert hold.damping_kd_nm_s_per_rad == (1.0, 0.0, 0.0, 1.0, 1.0, 0.0)
    assert hold.max_abs_torque_nm == startup.max_abs_torque_nm
    assert hold.max_torque_rate_nm_s == startup.max_torque_rate_nm_s
    assert hold.metadata["software_damping_disabled_where_driver_kd_nonzero"] is True
    assert hold.metadata["position_stiffness_disabled"] is True


def test_protective_hold_uses_default_damping_without_driver_kd() -> None:
    startup = probe.build_startup_lock_profile(
        mit_kp_nm_per_rad=(50.0,) * 6,
        mit_kd_nm_s_per_rad=(3.0,) * 6,
        torque_limits_nm=(2.0,) * 6,
        torque_rise_time_sec=1.0,
        velocity_limit_rad_s=1.0,
    )

    hold = probe.build_protective_hold_profile(
        startup,
        driver_kd_nm_s_per_rad=(0.0,) * 6,
    )

    assert hold.damping_kd_nm_s_per_rad == (1.0,) * 6


def test_protective_hold_default_damping_can_be_selected_for_drag_test() -> None:
    startup = probe.build_startup_lock_profile(
        mit_kp_nm_per_rad=(50.0,) * 6,
        mit_kd_nm_s_per_rad=(3.0,) * 6,
        torque_limits_nm=(2.0,) * 6,
        torque_rise_time_sec=1.0,
        velocity_limit_rad_s=1.0,
    )

    hold = probe.build_protective_hold_profile(
        startup,
        driver_kd_nm_s_per_rad=(0.0,) * 6,
        default_damping_kd_nm_s_per_rad=1.4,
    )

    assert hold.position_kp_nm_per_rad == (0.0,) * 6
    assert hold.damping_kd_nm_s_per_rad == (1.4,) * 6
    assert hold.metadata["default_software_damping_kd_nm_s_per_rad"] == 1.4


def test_gravity_debug_protective_hold_preserves_zero_driver_damping() -> None:
    startup = probe.build_startup_lock_profile(
        mit_kp_nm_per_rad=(50.0,) * 6,
        mit_kd_nm_s_per_rad=(3.0,) * 6,
        torque_limits_nm=(2.0,) * 6,
        torque_rise_time_sec=1.0,
        velocity_limit_rad_s=1.0,
    )

    hold = probe.build_protective_hold_profile(
        startup,
        driver_kd_nm_s_per_rad=(0.0, 0.0, 0.2, 0.15, 0.1, 0.05),
        # The debug run must not acquire a new host damping term on J1/J2.
        default_damping_kd_nm_s_per_rad=1.0,
        preserve_zero_driver_damping=True,
    )

    assert hold.damping_kd_nm_s_per_rad == (0.0,) * 6
    assert hold.metadata["gain_source"] == "gravity_debug_frozen_driver_kd"
    assert hold.metadata["software_damping_fallback_enabled"] is False
    assert hold.metadata["preserve_zero_driver_damping"] is True
    assert hold.metadata["default_software_damping_kd_nm_s_per_rad"] == 0.0


def test_startup_lock_blend_preserves_full_gravity_and_handoffs_final_terms() -> None:
    startup = probe.B601MitTorqueBreakdown(
        position_torque_nm=(0.0,) * 6,
        damping_torque_nm=(0.0,) * 6,
        gravity_torque_nm=(4.0, -3.0, 2.0, -1.0, 0.5, -0.25),
        overlay_requested_nm=(0.0,) * 6,
        overlay_effective_nm=(0.0,) * 6,
        requested_torque_nm=(4.0, -3.0, 2.0, -1.0, 0.5, -0.25),
        overlay_limited=False,
    )
    final = probe.B601MitTorqueBreakdown(
        position_torque_nm=(0.1,) * 6,
        damping_torque_nm=(0.2,) * 6,
        gravity_torque_nm=startup.gravity_torque_nm,
        overlay_requested_nm=(0.3,) * 6,
        overlay_effective_nm=(0.25,) * 6,
        requested_torque_nm=(0.55,) * 6,
        overlay_limited=True,
    )

    halfway = probe.blend_startup_lock_torque_breakdown(
        startup,
        final,
        gravity_fraction=1.0,
        handoff_fraction=0.5,
    )
    assert halfway.position_torque_nm == pytest.approx((0.05,) * 6)
    assert halfway.damping_torque_nm == pytest.approx((0.1,) * 6)
    assert halfway.gravity_torque_nm == startup.gravity_torque_nm
    assert halfway.overlay_requested_nm == pytest.approx((0.15,) * 6)
    assert halfway.overlay_effective_nm == pytest.approx((0.125,) * 6)
    assert halfway.requested_torque_nm == pytest.approx(
        tuple(
            position + damping + gravity + overlay
            for position, damping, gravity, overlay in zip(
                halfway.position_torque_nm,
                halfway.damping_torque_nm,
                halfway.gravity_torque_nm,
                halfway.overlay_effective_nm,
                strict=True,
            )
        )
    )

    completed = probe.blend_startup_lock_torque_breakdown(
        startup,
        final,
        gravity_fraction=1.0,
        handoff_fraction=1.0,
    )
    assert completed.position_torque_nm == final.position_torque_nm
    assert completed.damping_torque_nm == final.damping_torque_nm
    assert completed.gravity_torque_nm == final.gravity_torque_nm
    assert completed.overlay_effective_nm == final.overlay_effective_nm


def test_feedback_stability_summary_reports_raw_finite_difference_excursion() -> None:
    positions = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0012, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0012, 0.0, 0.0, 0.0),
    )
    summary = probe.summarize_feedback_stability(
        positions,
        (0.0, 0.05, 0.10),
        stationarity_velocity_limit_rad_s=0.02,
    )

    joint3 = summary["per_joint"]["joint3"]
    assert joint3["max_abs_step_rad"] == pytest.approx(0.0012)
    assert joint3["max_abs_velocity_rad_s"] == pytest.approx(0.024)
    assert joint3["stationarity_exceed_count"] == 1
    assert summary["stationarity_passed"] is False


def test_feedback_stability_summary_rejects_non_monotonic_time() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        probe.summarize_feedback_stability(
            ((0.0,) * 6, (0.0,) * 6),
            (1.0, 1.0),
        )


def test_feedback_boundary_normalization_is_per_joint_and_preserves_raw_values() -> None:
    raw = (0.1, -4.943e-3, -6.82e-4, -5.0e-6, 0.0, 0.0)

    normalized, normalized_joint_names = probe.normalize_feedback_boundary_qpos(raw)

    assert normalized == (0.1, 0.0, 0.0, -5.0e-6, 0.0, 0.0)
    assert normalized_joint_names == ("joint2", "joint3")
    assert raw[1:3] == (-4.943e-3, -6.82e-4)


def test_feedback_boundary_normalization_does_not_admit_more_negative_values() -> None:
    for index, value in ((1, -10.0001e-3), (2, -3.0001e-3)):
        raw_values = [0.0] * 6
        raw_values[index] = value
        raw = tuple(raw_values)

        normalized, normalized_joint_names = probe.normalize_feedback_boundary_qpos(raw)

        assert normalized == raw
        assert normalized_joint_names == ()
        with pytest.raises(
            probe.LowGainProbeError,
            match="outside position acceptance limits",
        ):
            probe._validate_fault_hold_snapshot(
                raw,
                qvel_rad_s=(0.0,) * 6,
                observed_age_sec=0.0,
                position_limits_rad=_fault_hold_limits(),
            )


def test_feedback_boundary_normalization_admits_j3_and_keeps_other_joints_unchanged() -> None:
    raw = (0.1, 0.0, -2.216e-3, -0.002, -0.003, -0.004)

    normalized, normalized_joint_names = probe.normalize_feedback_boundary_qpos(raw)

    assert normalized == (0.1, 0.0, 0.0, -0.002, -0.003, -0.004)
    assert normalized_joint_names == ("joint3",)


def test_feedback_admission_limits_separate_boundary_windows_and_wrist_margin() -> None:
    strict = (
        (-2.8, 2.8),
        (0.0, 3.14),
        (0.0, 3.14),
        (-math.pi / 2.0, math.pi / 2.0),
        (-1.57, 1.57),
        (-3.14, 3.14),
    )

    admission = probe.build_feedback_admission_position_limits(strict)

    assert admission[:1] == strict[:1]
    assert admission[1] == pytest.approx((-0.01, 3.14))
    assert admission[2] == pytest.approx((-0.003, 3.14))
    assert admission[3] == pytest.approx(
        (-math.pi / 2.0 - 0.1, math.pi / 2.0 + 0.1)
    )
    assert admission[4] == pytest.approx((-1.67, 1.67))
    assert admission[5] == pytest.approx((-3.24, 3.24))


def test_feedback_boundary_normalization_admits_observed_j2_startup_transient_only() -> None:
    observed = -0.0076273283921182156
    raw = (0.1, observed, 0.2, 0.3, 0.4, 0.5)

    normalized, normalized_joint_names = probe.normalize_feedback_boundary_qpos(raw)

    assert normalized == (0.1, 0.0, 0.2, 0.3, 0.4, 0.5)
    assert normalized_joint_names == ("joint2",)

    beyond = (0.1, -0.0100001, 0.2, 0.3, 0.4, 0.5)
    unchanged, unchanged_joint_names = probe.normalize_feedback_boundary_qpos(beyond)
    assert unchanged == beyond
    assert unchanged_joint_names == ()


def test_backend_feedback_diagnostics_preserve_bad_finite_position() -> None:
    bad = 2.686622237168953e36
    state = SimpleNamespace(
        joints=(
            SimpleNamespace(
                name="joint4",
                motor_id=4,
                feedback_id=0xFD,
                position_rad=bad,
                position_source="robstride_mech_pos_parameter",
                valid=False,
                stale=True,
                error="joint4: mechPos(0x7019) raw=2.686622237168953e+36",
            ),
        )
    )

    diagnostics = probe._backend_feedback_diagnostics(state)

    assert diagnostics[0]["position_rad"] == bad
    assert diagnostics[0]["motor_id"] == 4
    assert "mechPos(0x7019)" in diagnostics[0]["error"]


def test_feedback_boundary_normalization_admits_j4_precision_window_only() -> None:
    j4_lower = -math.pi / 2.0
    raw = (0.1, 0.2, 0.3, j4_lower - 0.001, -0.3, 0.4)

    normalized, normalized_joint_names = probe.normalize_feedback_boundary_qpos(raw)

    assert normalized[3] == pytest.approx(j4_lower)
    assert normalized_joint_names == ("joint4",)
    assert normalized[:3] == raw[:3]
    assert normalized[4:] == raw[4:]


def test_j4_operating_margin_does_not_project_real_position() -> None:
    j4_lower = -math.pi / 2.0
    raw = (0.1, 0.2, 0.3, j4_lower - 0.001 - 1e-9, -0.3, 0.4)

    normalized, normalized_joint_names = probe.normalize_feedback_boundary_qpos(raw)

    assert normalized == raw
    assert normalized_joint_names == ()
    strict = (
        (-2.8, 2.8),
        (0.0, 3.14),
        (0.0, 3.14),
        (j4_lower, math.pi / 2.0),
        (-1.57, 1.57),
        (-3.14, 3.14),
    )
    admitted = probe._validate_in_limit_qpos(
        raw,
        position_limits_rad=probe.build_feedback_admission_position_limits(strict),
        name="j4_operating_qpos_rad",
    )
    assert admitted[3] == pytest.approx(raw[3])


def test_wrist_feedback_margin_admits_observed_j6_but_rejects_beyond_margin() -> None:
    strict = (
        (-2.8, 2.8),
        (0.0, 3.14),
        (0.0, 3.14),
        (-math.pi / 2.0, math.pi / 2.0),
        (-1.57, 1.57),
        (-3.14, 3.14),
    )
    admission = probe.build_feedback_admission_position_limits(strict)
    observed = (0.1, 0.2, 0.3, 0.4, 0.5, 3.2327115535736084)

    assert probe._validate_in_limit_qpos(
        observed,
        position_limits_rad=admission,
        name="observed_j6_qpos_rad",
    ) == observed
    with pytest.raises(probe.LowGainProbeError, match="position acceptance limits"):
        probe._validate_in_limit_qpos(
            (*observed[:5], 3.240001),
            position_limits_rad=admission,
            name="beyond_j6_qpos_rad",
        )


def test_feedback_reference_clamp_keeps_outgoing_command_limits_nominal() -> None:
    strict = (
        (-2.8, 2.8),
        (0.0, 3.14),
        (0.0, 3.14),
        (-math.pi / 2.0, math.pi / 2.0),
        (-1.57, 1.57),
        (-3.14, 3.14),
    )

    assert probe.clamp_reference_qpos_to_command_limits(
        (0.0, 0.5, 0.5, -1.62, 1.63, 3.2327115535736084),
        position_limits_rad=strict,
    ) == pytest.approx((0.0, 0.5, 0.5, -math.pi / 2.0, 1.57, 3.14))


def test_resolve_model_and_limits_uses_precise_j4_nominal_boundary(monkeypatch, tmp_path) -> None:
    class FakeModel:
        urdf_path = tmp_path / "b601.urdf"

        @classmethod
        def from_rebot_source_root(cls, source_root):
            return cls()

        def arm_position_limits(self):
            return (
                np.asarray((-2.8, 0.0, 0.0, -1.57, -1.57, -3.14)),
                np.asarray((2.8, 3.14, 3.14, 1.57, 1.57, 3.14)),
            )

    monkeypatch.setattr(probe, "B601PinocchioModel", FakeModel)
    _model, limits = probe._resolve_model_and_limits(tmp_path)

    assert limits[3] == pytest.approx((-math.pi / 2.0, math.pi / 2.0))
    assert limits[:3] == ((-2.8, 2.8), (0.0, 3.14), (0.0, 3.14))
    assert limits[4:] == ((-1.57, 1.57), (-3.14, 3.14))


def test_feedback_position_guard_records_out_of_limit_sample_without_dispatch() -> None:
    raw = (0.1, 0.2, 0.3, 2.448736388725043e24, 0.5, 0.6)
    rejection = probe.evaluate_feedback_position_guard(
        sequence=742,
        feedback_timestamp_sec=484802.391661432,
        raw_qpos_rad=raw,
        normalized_qpos_rad=raw,
        position_limits_rad=(
            (-2.8, 2.8),
            (0.0, 3.14),
            (0.0, 3.14),
            (-1.57, 1.57),
            (-1.57, 1.57),
            (-3.14, 3.14),
        ),
        previous_qvel_rad_s=(0.0,) * 6,
    )

    assert rejection is not None
    assert rejection["guard"] == "position_limits"
    assert rejection["offending_joint"] == "joint4"
    assert rejection["offending_joint_index"] == 3
    assert rejection["observed_value"] == raw[3]
    assert rejection["raw_qpos_rad"] == raw
    assert rejection["normalized_qpos_rad"] == raw
    assert rejection["sequence"] == 742
    assert rejection["command_dispatched"] is False
    assert rejection["eligible_for_last_safe"] is False
    assert rejection["feedback_dt_sec"] is None


def test_feedback_position_guard_accepts_only_the_pinned_position_limits() -> None:
    assert (
        probe.evaluate_feedback_position_guard(
            sequence=1,
            feedback_timestamp_sec=1.0,
            raw_qpos_rad=(0.1, -4.0e-3, -2.0e-3, 0.0, 0.0, 0.0),
            normalized_qpos_rad=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            position_limits_rad=(
                (-2.8, 2.8),
                (0.0, 3.14),
                (0.0, 3.14),
                (-1.57, 1.57),
                (-1.57, 1.57),
                (-3.14, 3.14),
            ),
        )
        is None
    )


def test_motion_rejection_confirmation_quarantines_first_rejection() -> None:
    rejection = {"reason": "velocity abort", "sequence": 10}

    pending, action = probe.advance_motion_rejection_confirmation(None, rejection)

    assert pending is rejection
    assert action == "quarantine"


def test_motion_rejection_confirmation_recovers_on_next_valid_frame() -> None:
    pending, action = probe.advance_motion_rejection_confirmation(
        {"reason": "velocity abort", "sequence": 10},
        None,
    )

    assert pending is None
    assert action == "recovered"


def test_motion_rejection_confirmation_aborts_on_consecutive_rejection() -> None:
    first = {"reason": "velocity abort", "sequence": 10}
    second = {"reason": "acceleration abort", "sequence": 11}

    pending, action = probe.advance_motion_rejection_confirmation(first, second)

    assert pending is second
    assert action == "abort"


def test_feedback_stability_probe_never_enters_motion_path(tmp_path) -> None:
    class FakeBackend:
        config = SimpleNamespace(channel="can0")

        def __init__(self) -> None:
            self.events: list[object] = []
            self.poll_count = 0

        def connect(self) -> None:
            self.events.append("connect")

        def poll_feedback(self, *, request_feedback: bool) -> object:
            assert request_feedback is True
            self.events.append("poll_feedback")
            self.poll_count += 1
            return SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=(0.0, 0.0, 0.0012 * self.poll_count, 0.0, 0.0, 0.0),
                receive_timestamp_sec=0.05 * self.poll_count,
                fault_reason=None,
            )

        def close(self) -> None:
            self.events.append("close")

    now = [0.0]

    def clock() -> float:
        now[0] += 0.01
        return now[0]

    def sleeper(duration: float) -> None:
        now[0] += duration

    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("joints: []\n", encoding="utf-8")
    urdf_path = tmp_path / "b601.urdf"
    urdf_path.write_text("<robot name='b601'/>", encoding="utf-8")
    backend = FakeBackend()

    result = probe._run_feedback_stability_probe(
        backend,
        config_path=config_path,
        source_root=tmp_path,
        model=SimpleNamespace(urdf_path=urdf_path),
        position_limits=((-1.0, 1.0),) * 6,
        duration_sec=1.0,
        rate_hz=20.0,
        stationarity_velocity_limit_rad_s=0.02,
        clock=clock,
        sleeper=sleeper,
        max_samples=3,
    )

    assert result["status"] == "OK"
    assert result["motion_commands_sent"] is False
    assert result["mode_transitions_sent"] is False
    assert result["disable_commands_sent"] is False
    assert result["acceptance_status"] == "CAPTURED"
    assert result["stability"]["per_joint"]["joint3"]["stationarity_exceed_count"] == 2
    assert backend.events == ["connect", "poll_feedback", "poll_feedback", "poll_feedback", "close"]


def test_feedback_stability_probe_normalizes_only_admitted_j2_j3_boundary(tmp_path) -> None:
    class FakeBackend:
        config = SimpleNamespace(channel="can0")

        def __init__(self) -> None:
            self.poll_count = 0

        def connect(self) -> None:
            return None

        def poll_feedback(self, *, request_feedback: bool) -> object:
            assert request_feedback is True
            self.poll_count += 1
            return SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=(
                    0.0,
                    -4.517e-3,
                    -2.216e-3,
                    0.0,
                    0.0,
                    0.0,
                ),
                receive_timestamp_sec=0.05 * self.poll_count,
                fault_reason=None,
            )

        def close(self) -> None:
            return None

    now = [0.0]

    def clock() -> float:
        now[0] += 0.01
        return now[0]

    def sleeper(duration: float) -> None:
        now[0] += duration

    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("joints: []\n", encoding="utf-8")
    urdf_path = tmp_path / "b601.urdf"
    urdf_path.write_text("<robot name='b601'/>", encoding="utf-8")
    backend = FakeBackend()
    limits = _fault_hold_limits()

    result = probe._run_feedback_stability_probe(
        backend,
        config_path=config_path,
        source_root=tmp_path,
        model=SimpleNamespace(urdf_path=urdf_path),
        position_limits=limits,
        duration_sec=1.0,
        rate_hz=20.0,
        stationarity_velocity_limit_rad_s=0.02,
        clock=clock,
        sleeper=sleeper,
        max_samples=3,
    )

    assert result["status"] == "OK"
    boundary = result["feedback_boundary_normalization"]
    assert boundary["applied_sample_count"] == 3
    assert boundary["applied_joint_counts"] == {
        "joint2": 3,
        "joint3": 3,
        "joint4": 0,
    }
    assert boundary["intervals_rad"] == {
        "joint2": (-1.0e-2, 0.0),
        "joint3": (-3.0e-3, 0.0),
        "joint4": pytest.approx(
            (-math.pi / 2.0 - 1.0e-3, -math.pi / 2.0)
        ),
    }
    assert result["samples"][0]["raw_qpos_rad"][1:3] == (-4.517e-3, -2.216e-3)
    assert result["samples"][0]["qpos_rad"][1:3] == (0.0, 0.0)
    assert result["samples"][0]["boundary_normalized_joint_names"] == (
        "joint2",
        "joint3",
    )


def test_driver_torque_limit_roundtrip_never_enables_or_sends_motion(tmp_path) -> None:
    class FakeBackend:
        config = SimpleNamespace(channel="can0")

        def __init__(self) -> None:
            self.events: list[object] = []

        def connect(self) -> None:
            self.events.append("connect")

        def poll_feedback(self, *, request_feedback: bool) -> object:
            assert request_feedback is True
            self.events.append("poll_feedback")
            return SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=(0.0,) * 6,
                fault_reason=None,
            )

        def safe_stop(self, *, joint_names: tuple[str, ...]) -> object:
            assert joint_names == tuple(probe._ALL_JOINTS)
            self.events.append("safe_stop")
            return probe.B601ModeReceipt(
                operation="safe_stop",
                requested_joint_names=joint_names,
                completed_joint_names=joint_names,
                failed_joint_names=(),
                errors=(),
                completed=True,
            )

        def apply_temporary_driver_torque_limits(
            self,
            limits_nm: tuple[float, ...],
            *,
            joint_names: tuple[str, ...],
        ) -> object:
            assert limits_nm == (3.6, 3.6, 3.6, 1.4, 1.4, 1.4)
            assert joint_names == tuple(probe._ALL_JOINTS)
            self.events.append("apply")
            return probe.B601DriverTorqueLimitReceipt(
                operation="apply_temporary_driver_torque_limits",
                requested_joint_names=joint_names,
                target_limits_nm=limits_nm,
                prior_limits_nm=(36.0, 36.0, 36.0, 14.0, 14.0, 14.0),
                verified_limits_nm=limits_nm,
                applied_joint_names=joint_names,
                restored_joint_names=(),
                failed_joint_names=(),
                errors=(),
                completed=True,
            )

        def restore_driver_torque_limits(self) -> object:
            self.events.append("restore")
            return probe.B601DriverTorqueLimitReceipt(
                operation="restore_driver_torque_limits",
                requested_joint_names=tuple(probe._ALL_JOINTS),
                target_limits_nm=(36.0, 36.0, 36.0, 14.0, 14.0, 14.0),
                prior_limits_nm=(3.6, 3.6, 3.6, 1.4, 1.4, 1.4),
                verified_limits_nm=(36.0, 36.0, 36.0, 14.0, 14.0, 14.0),
                applied_joint_names=(),
                restored_joint_names=tuple(probe._ALL_JOINTS),
                failed_joint_names=(),
                errors=(),
                completed=True,
            )

        def close(self) -> None:
            self.events.append("close")

    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("joints: []\n", encoding="utf-8")
    urdf_path = tmp_path / "b601.urdf"
    urdf_path.write_text("<robot name='b601'/>", encoding="utf-8")
    backend = FakeBackend()

    result = probe._run_driver_torque_limit_roundtrip_probe(
        backend,
        config_path=config_path,
        source_root=tmp_path,
        model=SimpleNamespace(urdf_path=urdf_path),
        position_limits=((-1.0, 1.0),) * 6,
        torque_limits_nm=(3.6, 3.6, 3.6, 1.4, 1.4, 1.4),
    )

    assert result["status"] == "OK"
    assert result["motors_enabled_by_probe"] is False
    assert result["motion_commands_sent"] is False
    assert result["mode_transitions_sent"] is False
    assert result["joint_names"] == tuple(probe._ALL_JOINTS)
    assert result["target_limits_nm"] == (3.6, 3.6, 3.6, 1.4, 1.4, 1.4)
    assert result["apply_receipt"]["prior_limits_nm"] == (
        36.0,
        36.0,
        36.0,
        14.0,
        14.0,
        14.0,
    )
    assert result["restore_receipt"]["verified_limits_nm"] == (
        36.0,
        36.0,
        36.0,
        14.0,
        14.0,
        14.0,
    )
    assert backend.events == [
        "connect",
        "poll_feedback",
        "safe_stop",
        "apply",
        "safe_stop",
        "restore",
        "close",
    ]


def test_software_profile_has_single_torque_owner() -> None:
    profile = probe.build_software_profile(
        kp=0.5,
        kd=0.5,
        torque_limits_nm=(2.0,) * 6,
        torque_rise_time_sec=1.0,
        velocity_limit_rad_s=0.1,
    )

    assert profile.position_kp_nm_per_rad == (0.5,) * 6
    assert profile.damping_kd_nm_s_per_rad == (0.5,) * 6
    assert profile.max_torque_rate_nm_s == (2.0,) * 6
    assert profile.metadata["torque_owner"] == "software_composed_tau_excludes_vendor_pd"


def test_software_profile_overrides_only_the_target_joint_gains() -> None:
    profile = probe.build_software_profile(
        kp=0.25,
        kd=0.5,
        torque_limits_nm=(2.0,) * 6,
        torque_rise_time_sec=1.0,
        velocity_limit_rad_s=0.1,
        target_joint="joint6",
        target_kp=2.0,
        target_kd=1.0,
    )

    assert profile.position_kp_nm_per_rad == (0.25,) * 5 + (2.0,)
    assert profile.damping_kd_nm_s_per_rad == (0.5,) * 5 + (1.0,)
    assert profile.metadata["target_joint"] == "joint6"


def test_software_profile_can_bound_cartesian_overlay_independently() -> None:
    profile = probe.build_software_profile(
        kp=0.0,
        kd=0.5,
        torque_limits_nm=(2.0,) * 6,
        torque_rise_time_sec=1.0,
        velocity_limit_rad_s=0.1,
        overlay_limits_nm=(0.25,) * 6,
    )

    assert profile.max_abs_overlay_torque_nm == (0.25,) * 6


def test_reference_carries_joint_torque_overlay() -> None:
    reference = probe.build_reference(
        sequence=1,
        timestamp_sec=1.0,
        qpos_rad=(0.0,) * 6,
        joint_torque_overlay_nm=(0.1, -0.1, 0.2, -0.2, 0.3, -0.3),
    )

    assert reference.joint_torque_overlay_nm == (
        0.1,
        -0.1,
        0.2,
        -0.2,
        0.3,
        -0.3,
    )


def test_compute_software_torque_composes_gravity_pd_and_damping() -> None:
    class FakeModel:
        def arm_generalized_gravity(self, qpos_rad: object) -> tuple[float, ...]:
            assert tuple(qpos_rad) == (0.0,) * 6
            return (1.0, -2.0, 3.0, -4.0, 5.0, -6.0)

    reference = probe.build_reference(
        sequence=1,
        timestamp_sec=1.0,
        qpos_rad=(0.1, -0.1, 0.2, -0.2, 0.3, -0.3),
        qvel_rad_s=(0.0,) * 6,
    )
    profile = probe.build_software_profile(
        kp=1.0,
        kd=0.5,
        torque_limits_nm=(20.0,) * 6,
        torque_rise_time_sec=1.0,
        velocity_limit_rad_s=1.0,
    )

    torque = probe.compute_software_torque(
        FakeModel(),
        reference=reference,
        measured_qpos_rad=(0.0,) * 6,
        measured_qvel_rad_s=(0.1,) * 6,
        profile=profile,
        gravity_scale=0.5,
    )

    expected = tuple(
        0.5 * gravity + position_error - 0.05
        for gravity, position_error in zip(
            (1.0, -2.0, 3.0, -4.0, 5.0, -6.0),
            (0.1, -0.1, 0.2, -0.2, 0.3, -0.3),
            strict=True,
        )
    )
    assert np.allclose(torque, expected)


def test_hold_reference_never_moves_from_initial_pose() -> None:
    q0 = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    qref, qvel, done = probe.plan_reference(
        q0,
        mode="hold",
        joint="joint3",
        elapsed_sec=10.0,
        step_rad=0.05,
        step_speed_rad_s=0.02,
        step_direction=1,
        position_limits_rad=((-2.8, 2.8), (0.0, 3.14), (0.0, 3.14), (-1.57, 1.57), (-1.57, 1.57), (-3.14, 3.14)),
    )

    assert qref == q0
    assert qvel == (0.0,) * 6
    assert done is True


def test_cartesian_hold_reference_never_adds_joint_motion() -> None:
    q0 = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    qref, qvel, done = probe.plan_reference(
        q0,
        mode="cartesian-hold",
        joint="joint1",
        elapsed_sec=10.0,
        step_rad=0.05,
        step_speed_rad_s=0.02,
        step_direction=1,
        position_limits_rad=_fault_hold_limits(),
    )

    assert qref == q0
    assert qvel == (0.0,) * 6
    assert done is True


def test_pose_hold_reference_never_adds_joint_motion() -> None:
    q0 = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    qref, qvel, done = probe.plan_reference(
        q0,
        mode="pose-hold",
        joint="joint1",
        elapsed_sec=10.0,
        step_rad=0.05,
        step_speed_rad_s=0.02,
        step_direction=1,
        position_limits_rad=_fault_hold_limits(),
    )

    assert qref == q0
    assert qvel == (0.0,) * 6
    assert done is True


def test_pose_axis_guide_has_settle_signed_axes_and_recovery_order() -> None:
    phases = probe.build_pose_axis_guide_phases(
        translation_m=0.03,
        rotation_rad=math.radians(8.0),
        settle_sec=5.0,
        action_sec=6.0,
        rest_sec=4.0,
    )

    assert len(phases) == 25
    assert phases[0].name == "SETTLE"
    assert [phase.name for phase in phases[1::2]] == [
        "TRANSLATION_X_POSITIVE",
        "TRANSLATION_X_NEGATIVE",
        "TRANSLATION_Y_POSITIVE",
        "TRANSLATION_Y_NEGATIVE",
        "TRANSLATION_Z_POSITIVE",
        "TRANSLATION_Z_NEGATIVE",
        "ROTATION_RX_POSITIVE",
        "ROTATION_RX_NEGATIVE",
        "ROTATION_RY_POSITIVE",
        "ROTATION_RY_NEGATIVE",
        "ROTATION_RZ_POSITIVE",
        "ROTATION_RZ_NEGATIVE",
    ]
    assert all(phase.kind == "rest" for phase in phases[2::2])
    assert sum(phase.duration_sec for phase in phases) == pytest.approx(125.0)


def test_pose_axis_guide_labels_samples_and_emits_each_transition_once() -> None:
    phases = (
        probe.PoseAxisGuidePhase("SETTLE", "settle", 2.0),
        probe.PoseAxisGuidePhase(
            "TRANSLATION_X_POSITIVE",
            "translation",
            3.0,
            axis="X",
            direction=1,
            target_amplitude=0.03,
            target_unit="m",
        ),
        probe.PoseAxisGuidePhase(
            "REST_AFTER_TRANSLATION_X_POSITIVE",
            "rest",
            1.0,
            axis="X",
        ),
    )
    guide = probe.PoseAxisOperatorGuide(phases)

    first = guide.start(elapsed_sec=10.0, sample_index=20)
    assert guide.sample_metadata(elapsed_sec=11.5) == {
        "phase": "SETTLE",
        "phase_index": 0,
        "phase_elapsed_sec": pytest.approx(1.5),
    }
    transitions = guide.advance(elapsed_sec=12.0, sample_index=25)
    assert len(transitions) == 1
    assert transitions[0]["phase"]["name"] == "TRANSLATION_X_POSITIVE"
    assert guide.advance(elapsed_sec=12.1, sample_index=26) == ()
    transitions = guide.advance(elapsed_sec=16.0, sample_index=40)
    assert [event["state"] for event in transitions] == ["PHASE", "COMPLETE"]
    assert guide.completed is True
    assert guide.sample_metadata(elapsed_sec=16.0)["phase"] == "COMPLETE"
    assert len(guide.events) == 4
    assert first is guide.events[0]


def test_pose_axis_guide_terminal_notices_are_operator_only() -> None:
    phase = probe.PoseAxisGuidePhase(
        "ROTATION_RZ_NEGATIVE",
        "rotation",
        6.0,
        axis="RZ",
        direction=-1,
        target_amplitude=math.radians(8.0),
        target_unit="rad",
    )
    event = {
        "state": "PHASE",
        "phase_index": 7,
        "phase": phase.as_dict(),
    }

    notice = probe.format_pose_axis_guide_notice(event, phase_count=25)

    assert "[CAPTURE 8/25]" in notice
    assert "rotate -RZ" in notice
    assert "8 deg" in notice
    assert "move slowly" in notice
    assert set(event) == {"state", "phase_index", "phase"}


def test_interactive_pose_axis_actions_use_fixed_signed_order() -> None:
    actions = probe.build_interactive_pose_axis_actions(
        translation_m=0.03,
        rotation_rad=math.radians(8.0),
    )

    assert len(actions) == 12
    assert [action.name for action in actions] == [
        "TRANSLATION_X_POSITIVE",
        "TRANSLATION_X_NEGATIVE",
        "TRANSLATION_Y_POSITIVE",
        "TRANSLATION_Y_NEGATIVE",
        "TRANSLATION_Z_POSITIVE",
        "TRANSLATION_Z_NEGATIVE",
        "ROTATION_RX_POSITIVE",
        "ROTATION_RX_NEGATIVE",
        "ROTATION_RY_POSITIVE",
        "ROTATION_RY_NEGATIVE",
        "ROTATION_RZ_POSITIVE",
        "ROTATION_RZ_NEGATIVE",
    ]
    assert all(action.duration_sec == 0.0 for action in actions)


def test_interactive_pose_axis_guide_gates_anchor_and_manual_states() -> None:
    actions = probe.build_interactive_pose_axis_actions(
        translation_m=0.03,
        rotation_rad=math.radians(8.0),
    )
    guide = probe.InteractivePoseAxisOperatorGuide(actions)
    qpos = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    stationary = (0.0,) * 6
    moving = (0.0, 0.0, 0.03, 0.0, 0.0, 0.0)
    pose = np.eye(4)
    pose[0, 3] = 0.4

    positioning = guide.activate(elapsed_sec=1.0, sample_index=10)
    assert positioning["state"] == "POSITIONING"
    rejected = guide.handle_keys(
        frozenset({"a"}),
        qpos_rad=qpos,
        qvel_rad_s=moving,
        pose=pose,
        elapsed_sec=2.0,
        sample_index=20,
        stationarity_limit_rad_s=0.02,
    )
    assert rejected[0]["event"] == "REJECTED"
    assert guide.state == "POSITIONING"

    # A and N in one input batch capture only the anchor; they cannot skip READY.
    anchored = guide.handle_keys(
        frozenset({"a", "n"}),
        qpos_rad=qpos,
        qvel_rad_s=stationary,
        pose=pose,
        elapsed_sec=3.0,
        sample_index=30,
        stationarity_limit_rad_s=0.02,
    )
    assert anchored[0]["event"] == "ANCHOR_CAPTURED"
    assert guide.state == "READY"
    assert guide.anchor_qpos_rad == qpos
    assert guide.anchor_pose == tuple(tuple(value for value in row) for row in pose)

    assert guide.handle_keys(
        frozenset({"n"}),
        qpos_rad=qpos,
        qvel_rad_s=stationary,
        pose=pose,
        elapsed_sec=4.0,
        sample_index=40,
        stationarity_limit_rad_s=0.02,
    )[0]["state"] == "ACTION"
    assert guide.handle_keys(
        frozenset({"d"}),
        qpos_rad=qpos,
        qvel_rad_s=moving,
        pose=pose,
        elapsed_sec=5.0,
        sample_index=50,
        stationarity_limit_rad_s=0.02,
    )[0]["state"] == "REST"
    rejected_rest = guide.handle_keys(
        frozenset({"r"}),
        qpos_rad=qpos,
        qvel_rad_s=moving,
        pose=pose,
        elapsed_sec=6.0,
        sample_index=60,
        stationarity_limit_rad_s=0.02,
    )
    assert rejected_rest[0]["event"] == "REJECTED"
    assert guide.state == "REST"
    advanced = guide.handle_keys(
        frozenset({"r"}),
        qpos_rad=qpos,
        qvel_rad_s=stationary,
        pose=pose,
        elapsed_sec=7.0,
        sample_index=70,
        stationarity_limit_rad_s=0.02,
    )
    assert advanced[0]["state"] == "READY"
    assert guide.action_index == 1


def test_interactive_pose_axis_guide_completes_only_after_stationary_rest() -> None:
    action = probe.PoseAxisGuidePhase(
        "TRANSLATION_X_POSITIVE",
        "translation",
        0.0,
        axis="X",
        direction=1,
        target_amplitude=0.03,
        target_unit="m",
    )
    guide = probe.InteractivePoseAxisOperatorGuide((action,))
    qpos = (0.0,) * 6
    qvel = (0.0,) * 6
    pose = np.eye(4)
    guide.activate(elapsed_sec=0.0, sample_index=0)
    for key, elapsed in (("a", 1.0), ("n", 2.0), ("d", 3.0), ("r", 4.0)):
        guide.handle_keys(
            frozenset({key}),
            qpos_rad=qpos,
            qvel_rad_s=qvel,
            pose=pose,
            elapsed_sec=elapsed,
            sample_index=int(elapsed * 10),
            stationarity_limit_rad_s=0.02,
        )

    assert guide.completed is True
    assert guide.completed_elapsed_sec == pytest.approx(4.0)
    assert guide.sample_metadata(elapsed_sec=4.0) == {
        "phase": "COMPLETE",
        "phase_index": 1,
        "phase_elapsed_sec": pytest.approx(0.0),
        "state": "COMPLETE",
    }
    assert "protective hold" in probe.format_interactive_pose_axis_notice(
        guide.events[-1],
        action_count=1,
    )


def test_cartesian_impedance_overlay_restores_toward_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def forward_kinematics(self, qpos_rad: object) -> np.ndarray:
            transform = np.eye(4)
            transform[0, 3] = next(iter(qpos_rad))
            return transform

        def arm_jacobian(self, qpos_rad: object) -> np.ndarray:
            assert len(tuple(qpos_rad)) == 6
            return np.eye(6)

    def fake_map(
        model: object,
        qpos_rad: object,
        wrench: object,
        *,
        qvel_rad_s: object,
    ) -> object:
        assert isinstance(model, FakeModel)
        assert len(tuple(qpos_rad)) == 6
        force = tuple(wrench.force_n)
        torque = (*force, 0.0, 0.0, 0.0)
        qvel = tuple(qvel_rad_s)
        power = sum(value * velocity for value, velocity in zip(torque, qvel, strict=True))
        return type(
            "Mapping",
            (),
            {
                "joint_torque_nm": torque,
                "cartesian_power_w": power,
                "joint_power_w": power,
            },
        )()

    monkeypatch.setattr(
        probe,
        "map_world_aligned_ee_wrench_to_joint_torque",
        fake_map,
    )
    spring = probe.B601TranslationalVirtualSpring(
        anchor_position_m=(0.0, 0.0, 0.0),
        stiffness_n_per_m=(20.0, 20.0, 20.0),
        damping_n_s_per_m=(1.0, 1.0, 1.0),
        max_abs_force_n=(2.0, 2.0, 2.0),
    )

    overlay, audit = probe.compute_cartesian_impedance_overlay(
        FakeModel(),
        spring,
        qpos_rad=(0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
        qvel_rad_s=(0.0,) * 6,
        max_displacement_m=0.02,
    )

    assert np.allclose(overlay, (-0.2, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert audit["displacement_norm_m"] == pytest.approx(0.01)
    assert audit["force_n"] == pytest.approx((-0.2, 0.0, 0.0))
    assert audit["power_error_w"] == pytest.approx(0.0)


def test_cartesian_impedance_overlay_aborts_large_displacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def forward_kinematics(self, _qpos_rad: object) -> np.ndarray:
            transform = np.eye(4)
            transform[0, 3] = 0.021
            return transform

        def arm_jacobian(self, _qpos_rad: object) -> np.ndarray:
            return np.eye(6)

    monkeypatch.setattr(
        probe,
        "map_world_aligned_ee_wrench_to_joint_torque",
        lambda *_args, **_kwargs: pytest.fail("mapping must not run after abort"),
    )
    spring = probe.B601TranslationalVirtualSpring(
        anchor_position_m=(0.0, 0.0, 0.0),
        stiffness_n_per_m=(20.0,) * 3,
        damping_n_s_per_m=(1.0,) * 3,
        max_abs_force_n=(2.0,) * 3,
    )

    with pytest.raises(probe.LowGainProbeError, match="displacement abort"):
        probe.compute_cartesian_impedance_overlay(
            FakeModel(),
            spring,
            qpos_rad=(0.0,) * 6,
            qvel_rad_s=(0.0,) * 6,
            max_displacement_m=0.02,
        )


def test_pose_impedance_overlay_restores_translation_and_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def forward_kinematics(self, qpos_rad: object) -> np.ndarray:
            values = tuple(qpos_rad)
            transform = np.eye(4)
            transform[0, 3] = values[0]
            cosine = np.cos(values[5])
            sine = np.sin(values[5])
            transform[:3, :3] = (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            )
            return transform

        def arm_jacobian(self, _qpos_rad: object) -> np.ndarray:
            return np.eye(6)

    def fake_map(
        _model: object,
        _qpos_rad: object,
        wrench: object,
        *,
        qvel_rad_s: object,
    ) -> object:
        torque = tuple(wrench.as_vector())
        qvel = tuple(qvel_rad_s)
        power = sum(
            value * velocity for value, velocity in zip(torque, qvel, strict=True)
        )
        return type(
            "Mapping",
            (),
            {
                "joint_torque_nm": torque,
                "cartesian_power_w": power,
                "joint_power_w": power,
            },
        )()

    monkeypatch.setattr(
        probe,
        "map_world_aligned_ee_wrench_to_joint_torque",
        fake_map,
    )
    spring = probe.B601PoseVirtualSpring(
        anchor_pose=np.eye(4),
        translational_stiffness_n_per_m=(20.0,) * 3,
        translational_damping_n_s_per_m=(1.0,) * 3,
        rotational_stiffness_nm_per_rad=(1.0,) * 3,
        rotational_damping_nm_s_per_rad=(0.1,) * 3,
        max_abs_force_n=(2.0,) * 3,
        max_abs_torque_nm=(0.25,) * 3,
    )

    overlay, audit = probe.compute_pose_impedance_overlay(
        FakeModel(),
        spring,
        qpos_rad=(0.01, 0.0, 0.0, 0.0, 0.0, 0.02),
        qvel_rad_s=(0.0,) * 6,
        max_displacement_m=0.02,
        max_rotation_rad=0.05,
    )

    assert np.allclose(overlay, (-0.2, 0.0, 0.0, 0.0, 0.0, -0.02))
    assert audit["displacement_norm_m"] == pytest.approx(0.01)
    assert audit["rotation_error_norm_rad"] == pytest.approx(0.02)
    assert audit["power_error_w"] == pytest.approx(0.0)


def test_pose_impedance_overlay_aborts_excessive_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def forward_kinematics(self, _qpos_rad: object) -> np.ndarray:
            transform = np.eye(4)
            cosine = np.cos(0.06)
            sine = np.sin(0.06)
            transform[:3, :3] = (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            )
            return transform

        def arm_jacobian(self, _qpos_rad: object) -> np.ndarray:
            return np.eye(6)

    monkeypatch.setattr(
        probe,
        "map_world_aligned_ee_wrench_to_joint_torque",
        lambda *_args, **_kwargs: pytest.fail("mapping must not run after abort"),
    )
    spring = probe.B601PoseVirtualSpring(
        anchor_pose=np.eye(4),
        translational_stiffness_n_per_m=(20.0,) * 3,
        translational_damping_n_s_per_m=(1.0,) * 3,
        rotational_stiffness_nm_per_rad=(1.0,) * 3,
        rotational_damping_nm_s_per_rad=(0.1,) * 3,
        max_abs_force_n=(2.0,) * 3,
        max_abs_torque_nm=(0.25,) * 3,
    )

    with pytest.raises(probe.LowGainProbeError, match="rotation abort"):
        probe.compute_pose_impedance_overlay(
            FakeModel(),
            spring,
            qpos_rad=(0.0,) * 6,
            qvel_rad_s=(0.0,) * 6,
            max_displacement_m=0.02,
            max_rotation_rad=0.05,
        )


def test_active_joint_resolution_always_uses_all_six_arm_joints() -> None:
    assert probe.resolve_active_joint_names() == tuple(probe._ALL_JOINTS)


def test_protective_hold_uses_all_six_arm_joints_and_excludes_no_arm_joint() -> None:
    assert probe._FAULT_HOLD_JOINTS == tuple(probe._ALL_JOINTS)
    assert probe._NON_FAULT_HOLD_JOINTS == ()


def test_hold_reference_uses_the_complete_settled_six_joint_snapshot() -> None:
    settled = (0.12, 0.34, 0.56, -0.21, 0.09, -0.45)

    qref, qvel_ref, done = probe.plan_reference(
        settled,
        mode="hold",
        joint="joint1",
        elapsed_sec=1.0,
        step_rad=0.02,
        step_speed_rad_s=0.005,
        step_direction=1,
        position_limits_rad=((-3.14, 3.14),) * 6,
    )

    assert qref == settled
    assert qvel_ref == (0.0,) * 6
    assert done is True


def _fault_hold_limits() -> tuple[tuple[float, float], ...]:
    return (
        (-2.8, 2.8),
        (0.0, 3.14),
        (0.0, 3.14),
        (-math.pi / 2.0, math.pi / 2.0),
        (-1.57, 1.57),
        (-3.14, 3.14),
    )


def _fault_hold_command_receipt(
    *,
    accepted: bool,
    sequence: int = 1,
) -> probe.B601CommandReceipt:
    sent = tuple(probe._ALL_JOINTS) if accepted else tuple(probe._ALL_JOINTS[:-1])
    failed = () if accepted else (probe._ALL_JOINTS[-1],)
    return probe.B601CommandReceipt(
        sequence=sequence,
        mode="mit",
        frame_id="b601_rs/joint_mit_reference",
        requested_at_sec=1.0,
        completed_at_sec=1.0,
        requested_joint_names=tuple(probe._ALL_JOINTS),
        sent_joint_names=sent,
        failed_joint_names=failed,
        errors=() if accepted else ("simulated incomplete dispatch",),
        accepted=accepted,
    )


def test_fault_hold_uses_real_feedback_for_gravity_and_nominal_dispatch_reference() -> None:
    observed = (0.1, 0.7, 1.1, 0.0, -0.1, 3.2327115535736084)

    class FakeModel:
        def __init__(self) -> None:
            self.gravity_qpos: list[tuple[float, ...]] = []

        def arm_generalized_gravity(
            self,
            qpos_rad: object,
        ) -> tuple[float, ...]:
            self.gravity_qpos.append(tuple(qpos_rad))
            return (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    class FakeBackend:
        def __init__(self) -> None:
            self.reference: object | None = None

        def send_mit(
            self,
            reference: object,
            **_kwargs: object,
        ) -> probe.B601CommandReceipt:
            self.reference = reference
            return _fault_hold_command_receipt(accepted=True)

    model = FakeModel()
    backend = FakeBackend()
    profile = probe.build_software_profile(
        kp=0.0,
        kd=0.0,
        torque_limits_nm=(5.0,) * 6,
        torque_rise_time_sec=0.5,
        velocity_limit_rad_s=5.0,
    )

    _receipt, _conditioned, gravity_tau = probe._send_fault_hold_command(
        backend,
        model,
        qpos_rad=observed,
        measured_qpos_rad=observed,
        measured_qvel_rad_s=(0.03, -0.03, 0.0, 0.0, 0.0, 0.0),
        profile=profile,
        gravity_scale=1.0,
        previous_tau=(0.0,) * 6,
        torque_limits_nm=(5.0,) * 6,
        command_position_limits_rad=_fault_hold_limits(),
        rate_hz=20.0,
        gravity_joint_scales=(1.0, 0.95, 1.0, 1.0, 1.0, 1.0),
        gravity_joint_offsets_rad=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
        gravity_joint_bias_nm=(0.01,) * 6,
        gravity_coulomb_friction_nm=(0.02,) * 6,
        gravity_coulomb_velocity_threshold_rad_s=0.02,
    )

    assert model.gravity_qpos == [
        pytest.approx(tuple(value + 0.01 * index for index, value in enumerate(observed, 1)))
    ]
    assert gravity_tau == pytest.approx((0.13, 0.18, 0.31, 0.41, 0.51, 0.61))
    assert backend.reference is not None
    assert backend.reference.qpos_rad == pytest.approx((*observed[:5], 3.14))


class _FaultHoldBackend:
    def __init__(
        self,
        *,
        feedback: tuple[object, ...] = (),
        cached_receipts: tuple[probe.B601CommandReceipt, ...] = (),
        allow_safe_stop: bool = False,
    ) -> None:
        self.feedback = list(feedback)
        self.cached_receipts = list(cached_receipts)
        self.allow_safe_stop = allow_safe_stop
        self.cached_dispatches: list[dict[str, object]] = []
        self.safe_stop_names: list[tuple[str, ...]] = []

    def poll_feedback(self, *, request_feedback: bool) -> object:
        assert request_feedback is True
        if not self.feedback:
            raise AssertionError("unexpected feedback poll")
        return self.feedback.pop(0)

    def send_mit(
        self,
        reference: object,
        **kwargs: object,
    ) -> probe.B601CommandReceipt:
        self.cached_dispatches.append({"reference": reference, **kwargs})
        if not self.cached_receipts:
            raise AssertionError("unexpected cached MIT dispatch")
        return self.cached_receipts.pop(0)

    def safe_stop(self, *, joint_names: tuple[str, ...]) -> probe.B601ModeReceipt:
        self.safe_stop_names.append(tuple(joint_names))
        if not self.allow_safe_stop:
            raise AssertionError(f"unexpected safe_stop for {joint_names}")
        return probe.B601ModeReceipt(
            operation="full_safe_stop",
            requested_joint_names=tuple(joint_names),
            completed_joint_names=tuple(joint_names),
            failed_joint_names=(),
            errors=(),
            completed=True,
        )


def test_fault_hold_snapshot_accepts_only_fresh_stationary_in_limit_state() -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)

    accepted = probe._validate_fault_hold_snapshot(
        q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.05,
        position_limits_rad=_fault_hold_limits(),
        fault_qpos_rad=(0.1, 0.72, 1.08, 0.0, -0.1, 0.2),
        fault_qvel_rad_s=(0.0,) * 6,
    )

    assert accepted == q


def test_fault_hold_snapshot_rejects_stale_state() -> None:
    with pytest.raises(probe.LowGainProbeError, match="stale"):
        probe._validate_fault_hold_snapshot(
            (0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            qvel_rad_s=(0.0,) * 6,
            observed_age_sec=0.16,
            position_limits_rad=_fault_hold_limits(),
        )


def test_fault_hold_snapshot_rejects_motion_and_large_rebase() -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    with pytest.raises(probe.LowGainProbeError, match="moving"):
        probe._validate_fault_hold_snapshot(
            q,
            qvel_rad_s=(0.0, 0.021, 0.0, 0.0, 0.0, 0.0),
            observed_age_sec=0.01,
            position_limits_rad=_fault_hold_limits(),
        )
    with pytest.raises(probe.LowGainProbeError, match="rebase"):
        probe._validate_fault_hold_snapshot(
            q,
            qvel_rad_s=(0.0,) * 6,
            observed_age_sec=0.01,
            position_limits_rad=_fault_hold_limits(),
            fault_qpos_rad=(0.1, 0.76, 1.1, 0.0, -0.1, 0.2),
        )


def test_moving_control_sample_is_not_promoted_as_fault_hold_candidate() -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)

    accepted, rejection = probe._evaluate_fault_hold_candidate(
        q,
        qvel_rad_s=(0.0, 0.0, 0.0, 0.0, 0.022988, 0.0),
        velocity_abort_rad_s=0.05,
        observed_age_sec=0.01,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
    )

    assert accepted is None
    assert rejection is not None
    assert "moving" in rejection
    assert "0.022988" in rejection


def test_stationary_control_sample_is_promoted_as_fault_hold_candidate() -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)

    accepted, rejection = probe._evaluate_fault_hold_candidate(
        q,
        qvel_rad_s=(0.0, 0.0, 0.0, 0.0, 0.013779, 0.0),
        velocity_abort_rad_s=0.05,
        observed_age_sec=0.01,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
    )

    assert accepted == q
    assert rejection is None


def test_control_sample_above_normal_velocity_limit_still_aborts() -> None:
    with pytest.raises(probe.LowGainProbeError, match="velocity abort"):
        probe._evaluate_fault_hold_candidate(
            (0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            qvel_rad_s=(0.0, 0.0, 0.0, 0.0, 0.050001, 0.0),
            velocity_abort_rad_s=0.05,
            observed_age_sec=0.01,
            position_limits_rad=_fault_hold_limits(),
            max_age_sec=0.15,
            max_velocity_rad_s=0.02,
            max_rebase_rad=0.05,
        )


def test_arming_settle_transition_accepts_stationary_feedback() -> None:
    previous = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    current = (0.1, 0.7001, 1.1001, 0.0, -0.1, 0.2)

    accepted, qvel = probe._validate_arming_settle_transition(
        current,
        previous_qpos_rad=previous,
        current_timestamp_sec=1.05,
        previous_timestamp_sec=1.0,
        position_limits_rad=_fault_hold_limits(),
        max_velocity_rad_s=0.02,
    )

    assert accepted == current
    assert qvel[1] == pytest.approx(0.002)
    assert qvel[2] == pytest.approx(0.002)


def test_arming_settle_transition_accepts_bounded_enable_transient() -> None:
    previous = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    current = (0.1, 0.7032942, 1.1, 0.0, -0.1, 0.2)

    accepted, qvel = probe._validate_arming_settle_transition(
        current,
        previous_qpos_rad=previous,
        current_timestamp_sec=1.05,
        previous_timestamp_sec=1.0,
        position_limits_rad=_fault_hold_limits(),
        # Startup uses the normal motion watchdog (the command under test uses
        # 5 rad/s), not the 0.02 rad/s protective-hold admission gate.
        max_velocity_rad_s=5.0,
    )

    assert accepted == current
    assert qvel[1] == pytest.approx(0.065884)


def test_arming_settle_transition_rejects_motion() -> None:
    with pytest.raises(probe.LowGainProbeError, match="moving"):
        probe._validate_arming_settle_transition(
            (0.1, 0.7012, 1.1, 0.0, -0.1, 0.2),
            previous_qpos_rad=(0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            current_timestamp_sec=1.05,
            previous_timestamp_sec=1.0,
            position_limits_rad=_fault_hold_limits(),
            max_velocity_rad_s=0.02,
        )


def test_arming_settle_transition_rejects_non_advancing_timestamp() -> None:
    with pytest.raises(probe.LowGainProbeError, match="timestamp did not advance"):
        probe._validate_arming_settle_transition(
            (0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            previous_qpos_rad=(0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            current_timestamp_sec=1.0,
            previous_timestamp_sec=1.0,
            position_limits_rad=_fault_hold_limits(),
            max_velocity_rad_s=0.02,
        )


def test_fault_hold_snapshot_rejects_fault_position_without_velocity() -> None:
    with pytest.raises(probe.LowGainProbeError, match="velocity is unavailable"):
        probe._validate_fault_hold_snapshot(
            (0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            qvel_rad_s=(0.0,) * 6,
            observed_age_sec=0.01,
            position_limits_rad=_fault_hold_limits(),
            fault_qpos_rad=(0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
        )


def test_protective_hold_feedback_allows_in_limit_manual_drag() -> None:
    current = probe._validate_protective_hold_feedback(
        (0.1, 0.76, 1.1, 0.0, -0.1, 0.2),
        qvel_rad_s=(0.0, 0.4, 0.0, 0.0, 0.0, 0.0),
        hold_qpos_rad=(0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
        observed_age_sec=0.0,
        position_limits_rad=_fault_hold_limits(),
        max_velocity_rad_s=1.0,
    )

    assert current == (0.1, 0.76, 1.1, 0.0, -0.1, 0.2)


def test_protective_hold_feedback_rejects_motion_above_hold_monitor_limit() -> None:
    with pytest.raises(probe.LowGainProbeError, match="velocity abort"):
        probe._validate_protective_hold_feedback(
            (0.1, 0.76, 1.1, 0.0, -0.1, 0.2),
            qvel_rad_s=(0.0, 1.01, 0.0, 0.0, 0.0, 0.0),
            hold_qpos_rad=(0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            observed_age_sec=0.0,
            position_limits_rad=_fault_hold_limits(),
            max_velocity_rad_s=1.0,
        )


@pytest.mark.parametrize("display_enabled", [False, True])
def test_latched_protective_hold_references_current_pose_after_manual_drag(
    monkeypatch: pytest.MonkeyPatch,
    display_enabled: bool,
) -> None:
    q0 = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    q1 = (0.1, 0.71, 1.1, 0.0, -0.1, 0.2)
    command_qrefs: list[tuple[float, ...]] = []
    measured_qvels: list[tuple[float, ...]] = []
    displayed = []
    display = SimpleNamespace(observe=lambda q, **kwargs: displayed.append((q, kwargs)))

    def fake_send_hold(
        _backend: object,
        _model: object,
        *,
        qpos_rad: tuple[float, ...],
        measured_qvel_rad_s: tuple[float, ...],
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        command_qrefs.append(qpos_rad)
        measured_qvels.append(measured_qvel_rad_s)
        receipt = probe.B601CommandReceipt(
            sequence=len(command_qrefs),
            mode="mit",
            frame_id="b601_rs/joint_mit_reference",
            requested_at_sec=1.0,
            completed_at_sec=1.0,
            requested_joint_names=tuple(probe._ALL_JOINTS),
            sent_joint_names=tuple(probe._ALL_JOINTS),
            failed_joint_names=(),
            errors=(),
            accepted=True,
        )
        return receipt, (0.0,) * 6, (0.0,) * 6

    class FakeBackend:
        def __init__(self) -> None:
            self.poll_count = 0

        def poll_feedback(self, *, request_feedback: bool) -> object:
            assert request_feedback is True
            self.poll_count += 1
            q = q0 if self.poll_count < 2 else q1
            return SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=q,
                fault_reason=None,
            )

        def safe_stop(self, *, joint_names: tuple[str, ...]) -> probe.B601ModeReceipt:
            raise AssertionError(f"unexpected safe_stop for {joint_names}")

    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        FakeBackend(),
        object(),
        qpos_rad=q0,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=q0,
        fault_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=object(),
        gravity_scale=1.0,
        previous_tau=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=3,
        operator_interrupt=True,
        monitor_velocity_rad_s=1.0,
        joint_angle_display=display if display_enabled else None,
    )

    hold_count, _latest_tau, _first, non_hold, full, errors = result
    assert hold_count == 3
    assert command_qrefs[:2] == [q0, q0]
    assert command_qrefs[2] == q1
    assert measured_qvels[2][1] > 0.0
    assert non_hold is None
    assert full is None
    assert errors == []
    assert [q for q, _ in displayed] == (command_qrefs[1:] if display_enabled else [])
    assert all(metadata["phase"] == "PROTECTIVE_HOLD" for _, metadata in displayed)


def test_latched_protective_hold_uses_injected_control_feedback_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    reader_calls = 0

    def fake_send_hold(
        _backend: object,
        _model: object,
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        return _fault_hold_command_receipt(accepted=True), (0.0,) * 6, (0.0,) * 6

    def feedback_reader() -> object:
        nonlocal reader_calls
        reader_calls += 1
        return SimpleNamespace(
            valid=True,
            stale=False,
            positions_rad=q,
            fault_reason=None,
        )

    class FakeBackend:
        def poll_feedback(self, **_kwargs: object) -> object:
            raise AssertionError("legacy parameter feedback path must not be used")

        def safe_stop(self, *, joint_names: tuple[str, ...]) -> probe.B601ModeReceipt:
            raise AssertionError(f"unexpected safe_stop for {joint_names}")

    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        FakeBackend(),
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=q,
        fault_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=object(),
        gravity_scale=1.0,
        previous_tau=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=2,
        operator_interrupt=True,
        feedback_reader=feedback_reader,
    )

    assert result[0] == 2
    assert result[-1] == []
    assert reader_calls == 1


def test_operator_interrupt_ignores_transient_fault_velocity_for_first_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    command_qrefs: list[tuple[float, ...]] = []
    measured_qvels: list[tuple[float, ...]] = []

    def fake_send_hold(
        _backend: object,
        _model: object,
        *,
        qpos_rad: tuple[float, ...],
        measured_qvel_rad_s: tuple[float, ...],
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        command_qrefs.append(qpos_rad)
        measured_qvels.append(measured_qvel_rad_s)
        receipt = probe.B601CommandReceipt(
            sequence=len(command_qrefs),
            mode="mit",
            frame_id="b601_rs/joint_mit_reference",
            requested_at_sec=1.0,
            completed_at_sec=1.0,
            requested_joint_names=("joint2", "joint3"),
            sent_joint_names=("joint2", "joint3"),
            failed_joint_names=(),
            errors=(),
            accepted=True,
        )
        return receipt, (0.0,) * 6, (0.0,) * 6

    class FakeBackend:
        def poll_feedback(self, *, request_feedback: bool) -> object:
            assert request_feedback is True
            return SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=q,
                fault_reason=None,
            )

        def safe_stop(self, *, joint_names: tuple[str, ...]) -> probe.B601ModeReceipt:
            operation = (
                "safe_stop"
                if joint_names == probe._NON_FAULT_HOLD_JOINTS
                else "full_safe_stop"
            )
            return probe.B601ModeReceipt(
                operation=operation,
                requested_joint_names=tuple(joint_names),
                completed_joint_names=tuple(joint_names),
                failed_joint_names=(),
                errors=(),
                completed=True,
            )

    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        FakeBackend(),
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        # This is the interrupted loop sample. It is intentionally too fast
        # for the strict fault-snapshot gate, but is not used for the first
        # operator-requested hold frame.
        fault_qpos_rad=q,
        fault_qvel_rad_s=(0.37,) * 6,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=object(),
        gravity_scale=1.0,
        previous_tau=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=2,
        operator_interrupt=True,
    )

    hold_count, _latest_tau, first_receipt, non_hold_receipt, full_receipt, errors = result
    assert hold_count == 2
    assert command_qrefs == [q, q]
    assert measured_qvels == [(0.0,) * 6, (0.0,) * 6]
    assert first_receipt is not None and first_receipt.accepted
    assert non_hold_receipt is None
    assert full_receipt is None
    assert errors == []


def test_latched_fault_hold_uses_explicit_protective_hold_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    profiles: list[object] = []

    def fake_send_hold(
        _backend: object,
        _model: object,
        *,
        profile: object,
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        profiles.append(profile)
        receipt = probe.B601CommandReceipt(
            sequence=1,
            mode="mit",
            frame_id="b601_rs/joint_mit_reference",
            requested_at_sec=1.0,
            completed_at_sec=1.0,
            requested_joint_names=("joint2", "joint3"),
            sent_joint_names=("joint2", "joint3"),
            failed_joint_names=(),
            errors=(),
            accepted=True,
        )
        return receipt, (0.0,) * 6, (0.0,) * 6

    class FakeBackend:
        def poll_feedback(self, *, request_feedback: bool) -> object:
            raise AssertionError("max_iterations should stop before polling")

        def safe_stop(self, *, joint_names: tuple[str, ...]) -> probe.B601ModeReceipt:
            return probe.B601ModeReceipt(
                operation="safe_stop",
                requested_joint_names=tuple(joint_names),
                completed_joint_names=tuple(joint_names),
                failed_joint_names=(),
                errors=(),
                completed=True,
            )

    final_profile = object()
    startup_lock_profile = object()
    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        FakeBackend(),
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=q,
        fault_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=final_profile,
        gravity_scale=1.0,
        previous_tau=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=1,
        hold_profile=startup_lock_profile,
        operator_interrupt=True,
    )

    hold_count, _latest_tau, _first, non_hold, full, errors = result
    assert hold_count == 1
    assert profiles == [startup_lock_profile]
    assert non_hold is None
    assert full is None
    assert errors == []


@pytest.mark.parametrize("safety_damping", [None, (0.4, 0.6, 0.4, 0.6, 0.2, 0.1)])
@pytest.mark.parametrize("frozen_kd", [(0.0,) * 6, (0.11, 0.12, 0.13, 0.14, 0.15, 0.16)])
def test_gravity_debug_hold_uses_frozen_profile_over_mutable_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    safety_damping: tuple[float, ...] | None,
    frozen_kd: tuple[float, ...],
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    frozen = probe.B601GravityCompensationDebugProfile.capture(
        gravity_scale=0.83,
        gravity_joint_scales=(0.8, 0.81, 0.82, 0.83, 0.84, 0.85),
        gravity_joint_offsets_rad=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
        gravity_joint_bias_nm=(0.01, -0.02, -0.03, -0.04, 0.05, 0.06),
        gravity_coulomb_friction_nm=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
        gravity_coulomb_velocity_threshold_rad_s=0.07,
        driver_kd_nm_s_per_rad=frozen_kd,
    )
    observed: dict[str, object] = {}

    def fake_send_hold(
        _backend: object,
        _model: object,
        **kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        observed.update(kwargs)
        return _fault_hold_command_receipt(accepted=True), (0.0,) * 6, (0.0,) * 6

    class FakeBackend:
        def safe_stop(self, *, joint_names: tuple[str, ...]) -> probe.B601ModeReceipt:
            raise AssertionError(f"unexpected safe_stop for {joint_names}")

    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        FakeBackend(),
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=q,
        fault_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=object(),
        # Deliberately conflicting mutable/legacy arguments.  The explicit
        # debug snapshot is the only source allowed to win in this mode.
        gravity_scale=1.0,
        gravity_joint_scales=(1.0,) * 6,
        gravity_joint_offsets_rad=(0.0,) * 6,
        gravity_joint_bias_nm=(0.0,) * 6,
        gravity_coulomb_friction_nm=(0.0,) * 6,
        gravity_coulomb_velocity_threshold_rad_s=0.0,
        driver_kd_nm_s_per_rad=(0.0,) * 6,
        gravity_debug_profile=frozen,
        protective_driver_kd_nm_s_per_rad=safety_damping,
        previous_tau=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=1,
        operator_interrupt=True,
    )

    assert result[0] == 1
    assert result[-1] == []
    assert observed["gravity_scale"] == pytest.approx(0.83)
    assert observed["gravity_joint_scales"] == frozen.gravity_joint_scales
    assert observed["gravity_joint_offsets_rad"] == frozen.gravity_joint_offsets_rad
    assert observed["gravity_joint_bias_nm"] == frozen.gravity_joint_bias_nm
    assert observed["gravity_coulomb_friction_nm"] == (
        frozen.gravity_coulomb_friction_nm
    )
    assert observed["gravity_coulomb_velocity_threshold_rad_s"] == pytest.approx(
        0.07
    )
    expected_kd = frozen.driver_kd_nm_s_per_rad if safety_damping is None else safety_damping
    assert observed["driver_kd_nm_s_per_rad"] == expected_kd
    terminal = capsys.readouterr().out
    assert ("damping are ACTIVE" in terminal) == any(expected_kd)
    if not any(expected_kd):
        assert "damping is NOT configured" in terminal


def test_motion_fault_handoff_sends_damping_with_frozen_zero_debug_baseline() -> None:
    # Recorded J2 fault state; replay only the command boundary, not the plant.
    q = (0.050239369, 0.030297620, 0.313324571, 0.019174760, -0.065579928, -0.005368933)
    v = (0.0, 0.149171745, 0.0, -0.038226257, 0.0, 0.0)
    raw_gravity = (0.0, 2.425241056, 5.877818163, 1.843219289, 0.016251496, -0.000782174)
    frozen = probe.B601GravityCompensationDebugProfile.capture(
        gravity_scale=1.0, gravity_joint_scales=(1.0, .96, 1.0, 1.0, 1.0, 1.0),
        gravity_joint_offsets_rad=(0.0,) * 6,
        gravity_joint_bias_nm=(0.0, -.22, -.11, -.08, 0.0, 0.0),
        gravity_coulomb_friction_nm=(0.0,) * 6,
        gravity_coulomb_velocity_threshold_rad_s=.02,
        driver_kd_nm_s_per_rad=(0.0,) * 6,
    )
    safety_kd = (.4, .6, .4, .6, .2, .1)
    limits = (14.4, 14.4, 14.4, 2.4, 2.4, 2.4)
    gravity = tuple(g*s+b for g,s,b in zip(
        raw_gravity, frozen.gravity_joint_scales, frozen.gravity_joint_bias_nm, strict=True,
    ))
    profile = probe.build_software_profile(
        kp=0.0, kd=0.0, torque_limits_nm=limits,
        torque_rise_time_sec=.5, velocity_limit_rad_s=5.0,
    )
    hold = probe.build_protective_hold_profile(
        profile, driver_kd_nm_s_per_rad=safety_kd, preserve_zero_driver_damping=True,
    )
    backend = _FaultHoldBackend(cached_receipts=(_fault_hold_command_receipt(accepted=True),))
    result = probe._run_latched_fault_hold(
        backend, SimpleNamespace(arm_generalized_gravity=lambda _q: raw_gravity),
        qpos_rad=q, qvel_rad_s=v, observed_age_sec=.01,
        fault_qpos_rad=q, fault_qvel_rad_s=v, position_limits_rad=_fault_hold_limits(),
        max_age_sec=.15, max_velocity_rad_s=.02, max_rebase_rad=.05,
        profile=profile, hold_profile=hold, gravity_scale=1.0,
        previous_tau=gravity, torque_limits_nm=limits, rate_hz=100.,
        stop={"requested": False, "full_disable_requested": False},
        fault_entry=True, fault_entry_max_velocity_rad_s=5.0, max_iterations=1,
        gravity_debug_profile=frozen, protective_driver_kd_nm_s_per_rad=safety_kd,
    )
    assert result[0] == 1 and result[-1] == []
    sent = backend.cached_dispatches[0]
    assert sent["kp"] == (0.0,) * 6
    assert sent["kd"] == safety_kd
    assert sent["tau_nm"] == pytest.approx(gravity)
    assert sent["reference"].joint_torque_overlay_nm == (0.0,) * 6
    assert sent["reference"].qvel_rad_s == (0.0,) * 6
    assert not backend.safe_stop_names


def test_fault_entry_uses_current_snapshot_when_last_safe_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    current_qvel = (0.08, -0.12, 0.05, 0.03, -0.04, 0.06)
    stale_q = (0.1, 0.65, 1.05, 0.0, -0.1, 0.2)
    command_qrefs: list[tuple[float, ...]] = []
    measured_qvels: list[tuple[float, ...]] = []

    def fake_send_hold(
        _backend: object,
        _model: object,
        *,
        qpos_rad: tuple[float, ...],
        measured_qvel_rad_s: tuple[float, ...],
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        command_qrefs.append(qpos_rad)
        measured_qvels.append(measured_qvel_rad_s)
        receipt = probe.B601CommandReceipt(
            sequence=len(command_qrefs),
            mode="mit",
            frame_id="b601_rs/joint_mit_reference",
            requested_at_sec=1.0,
            completed_at_sec=1.0,
            requested_joint_names=("joint2", "joint3"),
            sent_joint_names=("joint2", "joint3"),
            failed_joint_names=(),
            errors=(),
            accepted=True,
        )
        return receipt, (0.0,) * 6, (0.0,) * 6

    class FakeBackend:
        def poll_feedback(self, *, request_feedback: bool) -> object:
            assert request_feedback is True
            return SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=current_q,
                fault_reason=None,
            )

        def safe_stop(self, *, joint_names: tuple[str, ...]) -> probe.B601ModeReceipt:
            operation = (
                "safe_stop"
                if joint_names == probe._NON_FAULT_HOLD_JOINTS
                else "full_safe_stop"
            )
            return probe.B601ModeReceipt(
                operation=operation,
                requested_joint_names=tuple(joint_names),
                completed_joint_names=tuple(joint_names),
                failed_joint_names=(),
                errors=(),
                completed=True,
            )

    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        FakeBackend(),
        object(),
        # The stationary candidate is stale and differs from the live fault
        # frame.  A normal fault entry must use the live frame instead.
        qpos_rad=stale_q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=current_q,
        fault_qvel_rad_s=current_qvel,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=object(),
        gravity_scale=1.0,
        previous_tau=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": False, "full_disable_requested": False},
        max_iterations=2,
        fault_entry=True,
        fault_entry_max_velocity_rad_s=0.5,
    )

    hold_count, _latest_tau, first_receipt, non_hold_receipt, full_receipt, errors = result
    assert hold_count == 2
    assert command_qrefs == [current_q, current_q]
    assert measured_qvels == [current_qvel, (0.0,) * 6]
    assert first_receipt is not None and first_receipt.accepted
    assert non_hold_receipt is None
    assert full_receipt is None
    assert errors == []


@pytest.mark.parametrize("debug_motion", [False, True])
def test_latched_fault_hold_uses_cached_gravity_on_invalid_feedback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    debug_motion: bool,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    gravity = (0.0, 0.8, 0.4, 0.1, 0.0, 0.0)
    command_qrefs: list[tuple[float, ...]] = []
    telemetry: dict[str, object] = {}
    trace_stream = io.StringIO()
    debug_trace = probe._DynamicsTrace(trace_stream)
    safety_kd = (0.4, 0.6, 0.4, 0.6, 0.2, 0.1) if debug_motion else (1.0,) * 6
    frozen = probe.B601GravityCompensationDebugProfile.capture(
        gravity_scale=1.0, gravity_joint_scales=(1.0,) * 6,
        gravity_joint_offsets_rad=(0.0,) * 6, gravity_joint_bias_nm=(0.0,) * 6,
        gravity_coulomb_friction_nm=(0.0,) * 6,
        gravity_coulomb_velocity_threshold_rad_s=0.02,
        driver_kd_nm_s_per_rad=(0.0,) * 6,
    ) if debug_motion else None

    def fake_send_hold(
        _backend: object,
        _model: object,
        *,
        qpos_rad: tuple[float, ...],
        measured_qpos_rad: tuple[float, ...],
        measured_qvel_rad_s: tuple[float, ...],
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        assert measured_qpos_rad == q
        assert measured_qvel_rad_s == (0.0,) * 6
        assert _kwargs["driver_kd_nm_s_per_rad"] == safety_kd
        command_qrefs.append(qpos_rad)
        return _fault_hold_command_receipt(accepted=True), gravity, gravity

    backend = _FaultHoldBackend(
        feedback=(
            SimpleNamespace(
                valid=False,
                stale=True,
                positions_rad=None,
                fault_reason="simulated feedback loss",
            ),
        ),
        cached_receipts=(_fault_hold_command_receipt(accepted=True, sequence=2),),
    )

    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        backend,
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=q,
        fault_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=SimpleNamespace(max_torque_rate_nm_s=(10.0,) * 6),
        gravity_scale=1.0,
        previous_tau=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=2,
        driver_kd_nm_s_per_rad=(1.0,) * 6,
        gravity_debug_profile=frozen,
        protective_driver_kd_nm_s_per_rad=safety_kd if debug_motion else None,
        cached_gravity_tau_nm=gravity,
        degraded_telemetry=telemetry,
        debug_trace=debug_trace,
    )

    hold_count, _latest_tau, first_receipt, non_hold_receipt, full_receipt, errors = result
    assert hold_count == 2
    assert command_qrefs == [q]
    assert len(backend.cached_dispatches) == 1
    assert backend.cached_dispatches[0]["kp"] == (0.0,) * 6
    assert backend.cached_dispatches[0]["kd"] == safety_kd
    assert backend.cached_dispatches[0]["tau_nm"] == gravity
    assert backend.cached_dispatches[0]["joint_names"] == probe._ALL_JOINTS
    assert first_receipt is not None and first_receipt.accepted
    assert non_hold_receipt is None
    assert full_receipt is None
    assert errors == []
    assert telemetry["entry_count"] == 1
    assert telemetry["command_count"] == 2
    assert telemetry["escalation_reason"] is None
    terminal = capsys.readouterr().out
    assert "!!! PROTECTIVE HOLD LATCHED !!!" in terminal
    assert "CONTROL: live gravity compensation and damping are ACTIVE." in terminal
    assert "!!! PROTECTIVE HOLD DEGRADED !!!" in terminal
    assert "DAMPING: motor-local damping remains ACTIVE." in terminal
    assert "REASON: fault hold: protective hold feedback is not complete" in terminal
    trace = [json.loads(line) for line in trace_stream.getvalue().splitlines()]
    assert trace[0]["kind"] == "protective_hold_entry"
    assert trace[-1]["kind"] == "protective_hold_cycle"
    assert trace[-1]["command_source"] == "cached_gravity"
    assert trace[-1]["cycle_error"] is not None


def test_dynamics_trace_writer_failure_is_diagnostic_only() -> None:
    class BrokenStream:
        def write(self, _text: str) -> None:
            raise OSError("disk full")

    trace = probe._DynamicsTrace(BrokenStream())
    trace.record("active_dispatch", qpos_rad=(0.0,) * 6)
    trace.record("active_dispatch", qpos_rad=(0.0,) * 6)

    assert trace.sample_count == 0
    assert trace.error == "OSError: disk full"
    assert trace.last_record_timing["completed"] is False


def test_sample_archive_preserves_completed_fields_and_builtin_types() -> None:
    archive = probe._ProbeSampleArchive()
    first = {"sequence": 1, "qpos_rad": (0.0,) * 6, "nested": {"values": [1, None, True]}}
    archive.append(first)
    first["control_cycle_timing"] = {"total_wall_sec": 0.01}
    second = {"sequence": 2, "qpos_rad": (0.1,) * 6}
    archive.append(second)

    assert archive[-1] is second
    assert archive[0] == first
    assert isinstance(archive[0]["qpos_rad"], tuple)
    assert isinstance(archive[0]["nested"]["values"], list)
    assert isinstance(archive._completed[0], bytes)
    assert archive.materialize() == [first, second]
    assert archive.diagnostics()["packed_sample_count"] == 1
    assert archive.diagnostics()["packing_fallback_count"] == 0


def test_workspace_summary_decodes_once_and_keeps_original_metrics(monkeypatch):
    archive = probe._ProbeSampleArchive()
    originals = [{"sequence": n, "elapsed_sec": n*.025,
        "qpos_rad": (n*.001,)*6, "startup_state": "ACTIVE" if n > 1 else "HANDOFF",
        "requested_tau_nm": (n*.1,)*6, "conditioned_tau_nm": (n*.09,)*6,
        "boundary_normalized_joint_names": ("joint2",) if n%2 else (),
        "torque_console": {"large_nested_history": list(range(100))}}
        for n in range(10)]
    for sample in originals:
        archive.append(sample)
    loads = probe.pickle.loads
    decoded = []

    def count(blob):
        decoded.append(len(blob))
        return loads(blob)

    monkeypatch.setattr(probe.pickle, "loads", count)
    compact = probe._workspace_analysis_samples(archive)
    assert len(decoded) == len(originals)-1
    assert all("torque_console" not in sample for sample in compact)
    assert probe.summarize_torque_conditioning(compact, absolute_limits_nm=(.5,)*6) == probe.summarize_torque_conditioning(originals, absolute_limits_nm=(.5,)*6)
    startup = {"pre_enable_qpos_rad": (0.,)*6, "lock_reference_qpos_rad": (0.,)*6,
        "pre_control_observations": []}
    assert probe.summarize_startup_diagnostics(**startup, samples=compact) == probe.summarize_startup_diagnostics(**startup, samples=originals)
    assert sum(bool(s.get("boundary_normalized_joint_names")) for s in compact) == 5
    assert len(decoded) == len(originals)-1
    assert archive.materialize() == originals


def test_automatic_rejection_record_is_flushed_once_with_current_cycle_timing() -> None:
    from collections import deque

    queued = deque([{
        "kind": "automatic_calibration_dispatch_rejected",
        "recorded_monotonic_sec": 12.041,
        "reason": "dispatch_completion_deadline_exceeded",
        "native_receipt": {"accepted": True, "sent_joint_names": list(probe._ALL_JOINTS)},
    }])

    def drain():
        result = list(queued)
        queued.clear()
        return result

    stream = io.StringIO()
    trace = probe._DynamicsTrace(stream)
    timing = probe._ProbeStageTiming()
    timing.mark("six_joint_dispatch")
    controller = SimpleNamespace(drain_records=drain)
    probe._flush_automatic_calibration_records(controller, trace, timing)
    probe._flush_automatic_calibration_records(controller, trace, timing)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(records) == 1
    assert records[0]["rejection_observed_at_sec"] == 12.041
    assert records[0]["native_receipt"]["accepted"] is True
    assert "six_joint_dispatch" in records[0]["cycle_timing"]["stages"]
    assert "dispatch_rejection_capture" in records[0]["cycle_timing"]["stages"]


@pytest.mark.parametrize("timestamp", [None, True, "12.0", float("nan"), float("inf")])
def test_automatic_records_reject_missing_or_invalid_event_time_without_fabricating_one(timestamp):
    controller = SimpleNamespace(drain_records=lambda: [{
        "kind": "automatic_calibration_cycle_transition",
        **({} if timestamp is None else {"recorded_monotonic_sec": timestamp}),
    }])
    stream = io.StringIO()
    with pytest.raises(probe.LowGainProbeError, match="(requires|has invalid) recorded_monotonic_sec"):
        probe._flush_automatic_calibration_records(controller, probe._DynamicsTrace(stream))
    assert stream.getvalue() == ""


def test_automatic_pre_dispatch_snapshot_preserves_explicit_null_receipt_time():
    controller = SimpleNamespace(drain_records=lambda: [{
        "kind": "automatic_calibration_started", "recorded_monotonic_sec": None,
    }])
    stream = io.StringIO()
    probe._flush_automatic_calibration_records(controller, probe._DynamicsTrace(stream))
    assert json.loads(stream.getvalue())["acquisition_completed_at_sec"] is None


def test_workspace_terminal_notice_is_immediate_and_emitted_once(monkeypatch):
    notices = []
    monkeypatch.setattr(probe, "_operator_notice", notices.append)
    queued = [{"kind": "workspace_calibration_transition", "recorded_monotonic_sec": 40.,
        "phase": "failed", "reason": "leg_timeout", "task_index": 3}]

    def drain():
        records = queued[:]
        queued.clear()
        return records

    controller = SimpleNamespace(drain_records=drain)
    stream = io.StringIO()
    probe._flush_automatic_calibration_records(controller, probe._DynamicsTrace(stream))
    probe._flush_automatic_calibration_records(controller, probe._DynamicsTrace(stream))
    assert len(notices) == 1
    assert "leg_timeout" in notices[0]
    assert "no automatic retry or return" in notices[0]
    assert len(stream.getvalue().splitlines()) == 1


def test_automatic_dispatch_admission_and_fault_flush_are_wired_before_send_and_hold():
    tree = ast.parse(inspect.getsource(probe.run_probe)).body[0]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    admission = next(n for n in calls if isinstance(n.func, ast.Attribute)
                     and n.func.attr == "validate_dispatch_start")
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and n.lineno < admission.lineno <= n.end_lineno]
    assert any("torque_console_audit is not None" in ast.unparse(n.test) for n in guards)
    sends = [n for n in calls if isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "backend"
             and n.func.attr == "send_mit" and n.lineno > admission.lineno]
    assert sends and min(n.lineno for n in sends) > admission.lineno
    cleanup = next(n for n in ast.walk(tree) if isinstance(n, ast.Try)
                   and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                           and c.func.attr == "owner_stopped"
                           for statement in n.finalbody for c in ast.walk(statement)))
    cleanup_calls = [c for statement in cleanup.finalbody for c in ast.walk(statement)
                     if isinstance(c, ast.Call)]
    flush = next(c for c in cleanup_calls if isinstance(c.func, ast.Name)
                 and c.func.id == "_flush_automatic_calibration_records")
    hold = next(c for c in cleanup_calls if isinstance(c.func, ast.Name)
                and c.func.id == "_run_latched_fault_hold")
    assert flush.lineno < hold.lineno


def test_probe_bulk_sample_evaluation_and_materialization_follow_cleanup() -> None:
    tree = ast.parse(inspect.getsource(probe.run_probe)).body[0]
    cleanup_blocks = [node for node in tree.body if isinstance(node, ast.Try) and any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id == "_release_owner_transport"
        for statement in node.finalbody for call in ast.walk(statement)
    )]
    assert len(cleanup_blocks) == 1
    cold_evaluators = {"evaluate_slow_step_response", "evaluate_assist_commissioning_response",
                       "evaluate_cartesian_assist_commissioning_response", "_workspace_analysis_samples"}
    bulk_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Name) and node.func.id in cold_evaluators
        or isinstance(node.func, ast.Attribute) and node.func.attr == "materialize"
    )]
    assert len(bulk_calls) == 5
    assert all(node.lineno > cleanup_blocks[0].end_lineno for node in bulk_calls)
    finish_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "finish_control_observation"]
    assert len(finish_calls) == 1
    assert cleanup_blocks[0].end_lineno < finish_calls[0].lineno < min(node.lineno for node in bulk_calls)
    restores = [node for node in bulk_calls if isinstance(node.func, ast.Attribute)]
    assert len(restores) == 1
    checkpoints = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "post_cleanup_checkpoint"]
    assert len(checkpoints) == 1
    assert finish_calls[0].lineno < checkpoints[0].lineno < min(node.lineno for node in bulk_calls)
    # Cold evaluators share full or compact analysis, never re-decode the
    # original archive. Lossless export still owns result_samples separately.
    assert all(keyword.value.id == "analysis_samples"
               for node in bulk_calls if isinstance(node.func, ast.Name)
               for keyword in node.keywords if keyword.arg == "samples")


def test_sample_archive_preserves_existing_response_evaluation() -> None:
    samples = _slow_step_samples((0.0, 0.005, 0.01, 0.015, 0.019, 0.02))
    archive = probe._ProbeSampleArchive()
    for sample in samples:
        archive.append(sample)
    args = {"mode": "slow-step", "target_joint": "joint6", "step_rad": 0.02,
            "step_direction": 1, "initial_qpos_rad": (0.0,) * 6,
            "active_joint_names": probe.resolve_active_joint_names()}

    assert probe.evaluate_slow_step_response(samples=archive, **args) == probe.evaluate_slow_step_response(samples=samples, **args)


def test_sample_archive_does_not_alias_frozen_frames_to_decoded_consumers() -> None:
    archive = probe._ProbeSampleArchive()
    archive.append({"sequence": 1, "nested": {"values": [1, 2]}})
    archive.append({"sequence": 2})
    restored = archive[0]
    restored["nested"]["values"].append(3)

    assert archive[0]["nested"]["values"] == [1, 2]
    assert [sample["sequence"] for sample in archive] == [1, 2]


@pytest.mark.parametrize("index", [0, -1, 3, -4])
def test_empty_sample_archive_rejects_indexes(index: int) -> None:
    archive = probe._ProbeSampleArchive()
    assert archive.materialize() == []
    assert archive.diagnostics()["sample_count"] == 0
    with pytest.raises(IndexError):
        archive[index]


def test_sample_archive_supports_slices_for_existing_evaluators() -> None:
    archive = probe._ProbeSampleArchive()
    for sequence in range(4):
        archive.append({"sequence": sequence})

    assert archive[1:3] == [{"sequence": 1}, {"sequence": 2}]
    assert archive[::-1] == [{"sequence": sequence} for sequence in reversed(range(4))]
    with pytest.raises(IndexError):
        archive[4]
    with pytest.raises(IndexError):
        archive[-5]


def test_sample_archive_preserves_numpy_buffer_types() -> None:
    archive = probe._ProbeSampleArchive()
    first = {"sequence": 1, "diagnostic_array": np.asarray([0.1, 0.2])}
    archive.append(first)
    archive.append({"sequence": 2})
    restored = archive[0]

    assert isinstance(restored["diagnostic_array"], np.ndarray)
    np.testing.assert_array_equal(restored["diagnostic_array"], first["diagnostic_array"])
    assert archive.diagnostics()["packing_fallback_count"] == 0


def test_sample_archive_retains_a_diagnostic_that_rejects_pickling() -> None:
    class Diagnostic:
        def __reduce__(self):
            raise ValueError("diagnostic cannot be packed")

        def __iter__(self):
            return iter([1, 2])

    archive = probe._ProbeSampleArchive()
    first = {"sequence": 1, "diagnostic": Diagnostic()}
    archive.append(first)
    archive.append({"sequence": 2})

    assert archive[0] is first
    assert archive.diagnostics()["first_packing_error"] == "ValueError: diagnostic cannot be packed"
    assert json.loads(json.dumps(archive.materialize(), default=list))[0]["diagnostic"] == [1, 2]


def test_sample_archive_retains_unsupported_diagnostic_values_without_losing_frames() -> None:
    archive = probe._ProbeSampleArchive()
    first = {"sequence": 1, "diagnostic_values": (value for value in [0.1, 0.2])}
    archive.append(first)
    archive.append({"sequence": 2})
    archive.append({"sequence": 3})

    assert archive[0] is first
    assert [sample["sequence"] for sample in archive] == [1, 2, 3]
    audit = archive.diagnostics()
    assert audit["sample_count"] == 3
    assert audit["packed_sample_count"] == 1
    assert audit["packing_fallback_count"] == 1
    assert audit["first_packing_error"]
    assert json.loads(json.dumps(archive.materialize(), default=list))[0]["diagnostic_values"] == [0.1, 0.2]


def test_probe_stage_timing_separates_wall_and_owner_cpu_and_snapshots_do_not_alias(monkeypatch):
    wall, cpu = iter((10., 10.03, 10.08)), iter((2., 2.002, 2.004))
    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(probe.time, "thread_time", lambda: next(cpu))
    timer = probe._ProbeStageTiming()
    timer.mark("feedback_read_and_admission")
    first = timer.snapshot()
    timer.mark("gravity_and_active_torque")
    second = timer.snapshot()
    assert first["stages"]["feedback_read_and_admission"]["wall_sec"] == pytest.approx(.03)
    assert first["stages"]["feedback_read_and_admission"]["thread_cpu_sec"] == pytest.approx(.002)
    assert "gravity_and_active_torque" not in first["stages"]
    assert second["total_wall_sec"] == pytest.approx(.08)
    assert second["total_thread_cpu_sec"] == pytest.approx(.004)
    assert first["stages"]["feedback_read_and_admission"]["started_perf_counter_sec"] == 10.
    assert first["stages"]["feedback_read_and_admission"]["completed_perf_counter_sec"] == 10.03


def test_feedback_substage_timing_sums_disjoint_intervals_without_hiding_retry_counts(monkeypatch):
    wall, cpu = iter((10., 10.03, 10.04, 10.09)), iter((2., 2.002, 2.003, 2.007))
    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(probe.time, "thread_time", lambda: next(cpu))
    diagnostics = probe._ProbeFeedbackTiming()
    diagnostics.mark("backend_poll")
    before = diagnostics.snapshot()
    diagnostics.mark("retry_wait")
    diagnostics.mark("backend_poll")
    after = diagnostics.snapshot()
    poll = after["stages"]["backend_poll"]
    assert poll["wall_sec"] == pytest.approx(.08)
    assert poll["thread_cpu_sec"] == pytest.approx(.006)
    assert poll["max_interval_wall_sec"] == pytest.approx(.05)
    assert poll["interval_count"] == 2
    assert after["total_wall_sec"] == pytest.approx(.09)
    assert before["stages"]["backend_poll"]["interval_count"] == 1
    assert "not_contiguous" in after["interval_semantics"]


def test_feedback_rejection_evidence_is_bounded_and_snapshot_does_not_change():
    diagnostics = probe._ProbeFeedbackTiming()
    diagnostics.reject("host_freshness_admission", "joint1 latest status frame is communication type 2, not Type-24")
    before = diagnostics.snapshot()
    for _ in range(1000):
        diagnostics.reject("backend_poll", "x"*2000)
    after = diagnostics.snapshot()
    assert before["last_rejection"]["ordinal"] == 1
    assert after["first_rejection"] == before["first_rejection"]
    assert after["rejection_counts"] == {"host_freshness_admission": 1, "backend_poll": 1000}
    assert after["last_rejection"]["ordinal"] == 1001
    assert len(after["last_rejection"]["reason"]) == 400
    assert not any(isinstance(v, BaseException) for v in after["last_rejection"].values())


def test_feedback_rejection_distinguishes_backend_retry_from_gate_retry(monkeypatch):
    now, calls = [10.], []
    frames = iter((None, _timed_feedback_state(1), _timed_feedback_state(2)))

    def poll(**kwargs):
        calls.append(kwargs)
        state = next(frames)
        if state is None:
            raise probe.LowGainProbeError("declared transient backend rejection")
        return state

    monkeypatch.setattr(probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(probe.time, "sleep", lambda delay: now.__setitem__(0, now[0]+delay))
    diagnostics = probe._ProbeFeedbackTiming()
    gate = probe.B601ActiveReportControlGate(probe._ALL_JOINTS, maximum_receive_age_sec=.10)
    state = probe._read_admitted_control_feedback(SimpleNamespace(poll_feedback=poll), gate, .01, diagnostics)
    assert state.sequence == 2 and len(calls) == 3
    evidence = diagnostics.snapshot()
    assert evidence["rejection_counts"] == {"backend_poll": 1, "baseline_not_advanced": 1}
    assert evidence["first_rejection"]["reason"] == "declared transient backend rejection"
    assert evidence["last_rejection"]["stage"] == "baseline_not_advanced"


def test_gc_observer_bounds_events_distinguishes_threads_and_does_not_change_gc(monkeypatch):
    settings = probe.gc.isenabled(), probe.gc.get_threshold()
    observer = probe._ProbeGCObserver(max_pending_events=2)
    owner_thread = probe.threading.get_ident()
    thread = [owner_thread]
    monkeypatch.setattr(probe.threading, "get_ident", lambda: thread[0])
    wall, cpu = iter((1., 1.02, 2., 2.03, 3., 3.04)), iter((0., .01, 1., 1.005, 2., 2.02))
    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(probe.time, "thread_time", lambda: next(cpu))
    for generation in range(3):
        thread[0] = owner_thread + 1 if generation == 1 else owner_thread
        observer._observe("start", {"generation": generation})
        observer._observe("stop", {"generation": generation, "collected": 0, "uncollectable": 0})
    captured = observer.drain()
    assert captured["completed_count"] == 3
    assert captured["dropped_event_count"] == 1
    assert len(captured["events"]) == captured["max_pending_events"] == 2
    worker, owner = captured["events"]
    assert worker["sequence"] == 2 and worker["generation"] == 1
    assert worker["collector_is_owner_thread"] is False
    assert worker["collector_thread_cpu_sec"] == pytest.approx(.005)
    assert owner["collector_is_owner_thread"] is True
    assert owner["wall_sec"] == pytest.approx(.04)
    assert owner["started_perf_counter_sec"] == 3.
    assert owner["completed_perf_counter_sec"] == 3.04
    assert observer.drain()["events"] == ()
    assert (probe.gc.isenabled(), probe.gc.get_threshold()) == settings


def test_gc_observer_registration_preserves_foreign_callbacks_on_exception(monkeypatch):
    foreign = lambda _phase, _info: None
    callbacks = [foreign]
    monkeypatch.setattr(probe.gc, "callbacks", callbacks)
    settings = probe.gc.isenabled(), probe.gc.get_threshold()
    observer = probe._ProbeGCObserver()
    with pytest.raises(RuntimeError, match="trial error"), observer:
        assert callbacks == [foreign, observer._callback]
        raise RuntimeError("trial error")
    assert callbacks == [foreign]
    observer.close()
    assert callbacks == [foreign]
    assert (probe.gc.isenabled(), probe.gc.get_threshold()) == settings


def test_gc_observer_does_not_report_loss_when_full_queue_is_drained_before_append():
    from collections import deque

    append_pending, resume_append = probe.threading.Event(), probe.threading.Event()
    errors = []

    class PausingDeque(deque):
        def append(self, event):
            if event[0] == 3:
                append_pending.set()
                if not resume_append.wait(2):
                    raise RuntimeError("offline interleave did not resume")
            super().append(event)

    observer = probe._ProbeGCObserver(max_pending_events=2)
    observer._events = PausingDeque(maxlen=2)

    def complete_collection():
        observer._observe("start", {"generation": 0})
        observer._observe("stop", {"generation": 0, "collected": 0, "uncollectable": 0})

    complete_collection()
    complete_collection()

    def third_collection():
        try:
            complete_collection()
        except RuntimeError as exc:
            errors.append(exc)

    worker = probe.threading.Thread(target=third_collection)
    worker.start()
    try:
        assert append_pending.wait(2)
        first = observer.drain()
    finally:
        resume_append.set()
        worker.join(2)
    assert not worker.is_alive()
    assert not errors
    second = observer.drain()
    delivered = [event["sequence"] for batch in (first, second) for event in batch["events"]]
    assert delivered == [1, 2, 3]
    assert first["dropped_event_count"] == second["dropped_event_count"] == 0


def test_gc_event_is_deferred_to_next_trace_with_original_bounds_and_no_control_changes():
    stream = io.StringIO()
    settings = probe.gc.isenabled(), probe.gc.get_threshold()
    with probe._ProbeGCObserver() as observer:
        trace = probe._DynamicsTrace(stream, gc_observer=observer)
        trace.record("active_dispatch", sequence=7, qpos_rad=(0.,) * 6)
        probe.gc.collect(0)  # Offline test only; production never requests collection.
        trace.record("active_dispatch", sequence=8, qpos_rad=(0.,) * 6)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [row["sequence"] for row in rows] == [7, 8]
    assert all(row["qpos_rad"] == [0.] * 6 for row in rows)
    events = rows[1]["gc_observation"]["events"]
    assert any(event["generation"] == 0 for event in events)
    assert all(event["completed_perf_counter_sec"] >= event["started_perf_counter_sec"] for event in events)
    assert (probe.gc.isenabled(), probe.gc.get_threshold()) == settings
    assert observer._callback not in probe.gc.callbacks


def test_gc_observation_ends_after_cleanup_before_bulk_result_collections():
    settings = probe.gc.isenabled(), probe.gc.get_threshold()
    stream = io.StringIO()
    with probe._ProbeGCObserver(max_pending_events=2) as observer:
        trace = probe._DynamicsTrace(stream, gc_observer=observer)
        trace.record("active_dispatch", sequence=1)
        probe.gc.collect(0)
        trace.record("protective_hold_cycle", command_count=1)
        probe.gc.collect(0)  # Cleanup work must still be observed.
        trace.finish_control_observation(cleanup_errors=())
        count_at_end = observer._completed
        assert observer._callback not in probe.gc.callbacks
        for _ in range(80):
            probe.gc.collect(0)  # Offline result allocations must not overflow the online queue.
        trace.finish_control_observation(cleanup_errors=("must not overwrite first boundary",))
        final = observer.drain()
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [row["kind"] for row in rows] == [
        "active_dispatch", "protective_hold_cycle", "control_observation_complete",
    ]
    assert rows[-1]["boundary"] == "after_control_cleanup_and_backend_close_attempt"
    assert rows[-1]["cleanup_errors"] == []
    assert "sample_materialization" in rows[-1]["excluded_phases"]
    events = [event for row in rows for event in row["gc_observation"]["events"]]
    assert len(events) == count_at_end == final["completed_count"]
    assert [event["sequence"] for event in events] == list(range(1, count_at_end + 1))
    assert final["dropped_event_count"] == 0
    assert final["events"] == ()
    assert (probe.gc.isenabled(), probe.gc.get_threshold()) == settings


def test_broken_trace_still_ends_gc_observation_and_preserves_cleanup_failure():
    class BrokenStream:
        def write(self, _text):
            raise OSError("disk full")

    with probe._ProbeGCObserver() as observer:
        trace = probe._DynamicsTrace(BrokenStream(), gc_observer=observer)
        trace.record("active_dispatch", sequence=1)
        probe.gc.collect(0)
        trace.finish_control_observation(cleanup_errors=("close: synthetic failure",))
        assert observer._callback not in probe.gc.callbacks
        final = observer.drain()
    assert trace.error == "OSError: disk full"
    assert trace.gc_observation_end["cleanup_errors"] == ("close: synthetic failure",)
    assert final["completed_count"] >= 1 and final["dropped_event_count"] == 0


def _timed_feedback_state(sequence: int, *, age: float = .005) -> B601HardwareState:
    joints = tuple(B601JointState(
        name=f"joint{i}", motor_id=i, feedback_id=0xFD, arbitration_id=24 << 24,
        status_code=0, position_rad=.1, velocity_reported=99., torque_reported_nm=99.,
        observed_at_sec=10. - age, sample_sequence=sequence, valid=True, stale=False,
        position_source="robstride_operation_status_cache", status_sequence=sequence,
        receive_age_sec=age, host_receive_freshness_verified=True,
    ) for i in range(1, 7))
    return B601HardwareState(
        sequence=sequence, receive_timestamp_sec=10., joints=joints,
        valid=True, stale=False, host_receive_freshness_verified=True,
    )


@pytest.mark.parametrize("planned,previous,period,expected", [
    (1.00228362, 1., .01, 1.01),
    (1.010, 1., .01, 1.010),
    (1.020, 1., .01, 1.020),
    (1.050, 1., .05, 1.050),
    (1.002, None, .01, 1.002),
])
def test_feedback_cadence_floor_preserves_later_deadlines_and_actual_period(
    planned, previous, period, expected,
):
    assert probe._active_report_feedback_deadline(
        planned, previous_feedback_timestamp_sec=previous, period_sec=period,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("planned,previous,period", [
    (math.nan, 1., .01),
    (1., math.inf, .01),
    (1., 1., 0.),
    (1., 1., -.01),
    (1., 1., math.nan),
    (1., 1e308, 1e308),
])
def test_feedback_cadence_floor_rejects_invalid_clocks_and_periods(planned, previous, period):
    with pytest.raises(ValueError):
        probe._active_report_feedback_deadline(
            planned, previous_feedback_timestamp_sec=previous, period_sec=period,
        )


@pytest.mark.parametrize("counts", [1, 2])
def test_feedback_cadence_reads_after_real_wait_and_preserves_raw_overspeed_guard(monkeypatch, counts):
    now = [1.00228362]
    waits = []
    def sleep(delay):
        waits.append(delay)
        now[0] += delay
    monkeypatch.setattr(probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(probe.time, "sleep", sleep)
    next_read = probe._active_report_feedback_deadline(
        now[0], previous_feedback_timestamp_sec=1., period_sec=.01,
    )
    probe.time.sleep(next_read - probe.time.monotonic())
    q = (0., 0., counts * .0003834962844848633, 0., 0., 0.)
    actual_snapshot_time = probe.time.monotonic()
    actual_dt = actual_snapshot_time - 1.
    raw = tuple(value / actual_dt for value in q)
    _, rejection = probe.evaluate_feedback_motion_guard(
        sequence=2, feedback_timestamp_sec=actual_snapshot_time, dt_sec=actual_dt,
        raw_qpos_rad=q, normalized_qpos_rad=q, previous_qvel_rad_s=None,
        current_qvel_rad_s=raw, velocity_abort_rad_s=.05, acceleration_abort_rad_s2=80.,
    )
    assert waits == pytest.approx([.00771638])
    assert actual_dt == pytest.approx(.01)
    assert raw[2] == pytest.approx(counts * .03834962844848633)
    if counts == 1:
        assert rejection is None
    else:
        assert rejection["guard"] == "velocity"
        assert rejection["limit"] == .05
        assert rejection["command_dispatched"] is False


def test_feedback_cadence_wiring_only_precedes_active_report_read_without_changing_velocity():
    source = inspect.getsource(probe.run_probe)
    tree = ast.parse(source)
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "active_report_gate is not None"]
    calls = [node for branch in branches for node in ast.walk(branch)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "_active_report_feedback_deadline"]
    assert len(calls) == 1
    assert {arg.arg: ast.unparse(arg.value) for arg in calls[0].keywords} == {
        "previous_feedback_timestamp_sec": "previous_feedback_time", "period_sec": "period",
    }
    assert source.index("next_deadline = _active_report_feedback_deadline(") < source.index(
        "requested_schedule_wait_sec = max(0.0, next_deadline - now)"
    ) < source.index("state = _read_admitted_control_feedback(", source.index("next_deadline = _active_report_feedback_deadline("))
    assert "dt = state.receive_timestamp_sec - previous_feedback_time" in source
    assert "(current - previous) / dt" in source


@pytest.mark.parametrize("advances", [False, True])
def test_feedback_retries_release_rejected_tracebacks_without_waiting_for_gc(monkeypatch, advances):
    errors = []
    original_error = probe.LowGainProbeError

    class ObservedError(original_error):
        def __init__(self, message):
            super().__init__(message)
            errors.append(weakref.ref(self))

    monkeypatch.setattr(probe, "LowGainProbeError", ObservedError)
    now, polls = [10.], [0]
    monkeypatch.setattr(probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(probe.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    gate = probe.B601ActiveReportControlGate(probe._ALL_JOINTS, maximum_receive_age_sec=.10)
    gate.observe(_timed_feedback_state(1))

    def poll(**kwargs):
        assert kwargs == {"require_host_freshness": True, "prefer_active_report": True}
        polls[0] += 1
        return _timed_feedback_state(2 if advances and polls[0] > 3 else 1)

    diagnostics = probe._ProbeFeedbackTiming()
    enabled = probe.gc.isenabled()
    # Offline lifetime assertion: automatic GC must not hide retained cycles.
    probe.gc.disable()
    try:
        if advances:
            state = probe._read_admitted_control_feedback(SimpleNamespace(poll_feedback=poll), gate, .01, diagnostics)
            assert state.sequence == 2 and polls[0] == 4
            assert all(reference() is None for reference in errors)
        else:
            with pytest.raises(ObservedError, match="joint1 status_sequence did not advance") as caught:
                probe._read_admitted_control_feedback(SimpleNamespace(poll_feedback=poll), gate, .002, diagnostics)
            assert all(reference() is None for reference in errors[:-1])
            assert errors[-1]() is caught.value  # The delivered timeout remains inspectable.
        assert diagnostics.retryable_error_count == polls[0] - int(advances)
        assert gate.last_status_sequences == {f"joint{i}": 2 if advances else 1 for i in range(1, 7)}
    finally:
        if enabled:
            probe.gc.enable()


@pytest.mark.parametrize("observe", [False, True])
def test_feedback_diagnostics_preserve_real_admission_retries_and_source_requests(monkeypatch, observe):
    baseline, advancing = _timed_feedback_state(1), _timed_feedback_state(2)
    frames = iter((baseline, baseline, _timed_feedback_state(2, age=.20), advancing))
    calls, waits, now = [], [], [10.]

    def poll(**kwargs):
        calls.append(kwargs)
        return next(frames)

    def sleep(delay):
        waits.append(delay)
        now[0] += delay

    monkeypatch.setattr(probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(probe.time, "sleep", sleep)
    gate = probe.B601ActiveReportControlGate(probe._ALL_JOINTS, maximum_receive_age_sec=.10)
    diagnostics = probe._ProbeFeedbackTiming() if observe else None
    admitted = probe._read_admitted_control_feedback(SimpleNamespace(poll_feedback=poll), gate, .05, diagnostics)
    assert admitted is advancing
    assert calls == [{"require_host_freshness": True, "prefer_active_report": True}] * 4
    assert waits == [.001] * 3
    assert gate.last_status_sequences == {f"joint{i}": 2 for i in range(1, 7)}
    if diagnostics is not None:
        report = diagnostics.snapshot()
        assert report["not_advanced_count"] == 1
        assert report["retryable_error_count"] == 2
        assert report["stages"]["backend_poll"]["interval_count"] == 4
        assert report["stages"]["host_freshness_admission"]["interval_count"] == 4
        assert report["stages"]["retry_wait"]["interval_count"] == 3
        assert report["rejection_counts"] == {"baseline_not_advanced": 1, "host_freshness_admission": 2}
        assert report["first_rejection"]["stage"] == "baseline_not_advanced"
        assert "did not advance" in report["last_rejection"]["reason"] or "receive_age_sec" in report["last_rejection"]["reason"]
        assert report["last_rejection"]["ordinal"] == 3


@pytest.mark.parametrize("observe", [False, True])
def test_feedback_diagnostics_keep_parameter_source_and_fatal_backend_errors(monkeypatch, observe):
    state, calls = _timed_feedback_state(1), []
    diagnostics = probe._ProbeFeedbackTiming() if observe else None

    def poll(**kwargs):
        calls.append(kwargs)
        return state

    backend = SimpleNamespace(poll_feedback=poll)
    assert probe._read_admitted_control_feedback(backend, None, .10, diagnostics) is state
    assert calls == [{"request_feedback": True}]
    if diagnostics is not None:
        assert set(diagnostics.snapshot()["stages"]) == {"backend_poll"}

    def fatal(**_kwargs):
        raise RuntimeError("transport failed")

    failed_diagnostics = probe._ProbeFeedbackTiming() if observe else None
    with pytest.raises(RuntimeError, match="transport failed"):
        probe._read_admitted_control_feedback(SimpleNamespace(poll_feedback=fatal), None, .10, failed_diagnostics)
    if failed_diagnostics is not None:
        assert failed_diagnostics.snapshot()["stages"]["backend_poll"]["interval_count"] == 1


@pytest.mark.parametrize("observe", [False, True])
def test_feedback_diagnostics_preserve_timeout_and_last_rejection(monkeypatch, observe):
    state, now, calls = _timed_feedback_state(1), [10.], []

    def poll(**kwargs):
        calls.append(kwargs)
        return state

    monkeypatch.setattr(probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(probe.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    gate = probe.B601ActiveReportControlGate(probe._ALL_JOINTS, maximum_receive_age_sec=.10)
    diagnostics = probe._ProbeFeedbackTiming() if observe else None
    with pytest.raises(probe.LowGainProbeError, match=r"within 0.002000s: active-report feedback rejected: joint1 status_sequence did not advance"):
        probe._read_admitted_control_feedback(SimpleNamespace(poll_feedback=poll), gate, .002, diagnostics)
    assert gate.last_status_sequences == {f"joint{i}": 1 for i in range(1, 7)}
    if diagnostics is not None:
        assert diagnostics.snapshot()["retryable_error_count"] == len(calls) - 1


def test_trace_timing_retains_origin_sequence_without_changing_record_payload():
    stream = io.StringIO()
    trace = probe._DynamicsTrace(stream)
    trace.record("active_dispatch", sequence=7, qpos_rad=(0.,) * 6)
    previous = trace.last_record_timing
    trace.record("active_dispatch", sequence=8, previous_trace_record_timing=previous)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert rows[0]["sequence"] == 7
    assert rows[0]["qpos_rad"] == [0.] * 6
    assert rows[1]["previous_trace_record_timing"]["sequence"] == 7
    assert trace.last_record_timing["sequence"] == 8
    assert previous["sequence"] == 7
    assert trace.sample_count == 2
    assert set(previous["stages"]) == {"encode", "write", "flush"}
    assert previous["completed"] is True


@pytest.mark.parametrize("handled_error", [False, True])
def test_debug_trace_main_writes_separate_jsonl_without_hardware(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    handled_error: bool,
) -> None:
    callbacks_before = tuple(probe.gc.callbacks)

    def fake_run_probe(_args: object, **kwargs: object) -> dict[str, object]:
        trace = kwargs["debug_trace"]
        assert isinstance(trace, probe._DynamicsTrace)
        assert trace.gc_observer._callback in probe.gc.callbacks
        trace.record("synthetic", commanded_power_w=(0.1,) * 6)
        diagnostics = probe._ProbeFeedbackTiming()
        diagnostics.mark("backend_poll")
        trace.last_feedback_read_diagnostics = diagnostics
        if handled_error:
            raise probe.LowGainProbeError("synthetic feedback timeout", cleanup={
                "device_cleanup_confirmed": False,
                "transport_release": {"completed": True, "method": "release_transport", "error": None},
                "errors": ["safe_stop: synthetic failure"],
            })
        return {"status": "OK", "mode": "hold"}

    monkeypatch.setattr(probe, "run_probe", fake_run_probe)
    output = tmp_path / "result.json"
    status = probe.main(
        [
            "--mode", "hold",
            "--config", str(tmp_path / "unused-config.yaml"),
            "--rebot-source-root", str(tmp_path / "unused-vendor"),
            "--output", str(output),
            "--log", str(tmp_path / "runtime.log"),
            "--debug-dynamics-trace",
        ]
    )

    assert status == (2 if handled_error else 0)
    result = json.loads(output.read_text(encoding="utf-8"))
    if handled_error:
        assert result["status"] == "ERROR"
        assert result["cleanup"]["device_cleanup_confirmed"] is False
        assert result["cleanup"]["transport_release"]["completed"] is True
        assert result["cleanup"]["errors"] == ["safe_stop: synthetic failure"]
    trace_path = tmp_path / "result.debug.jsonl"
    assert result["debug_dynamics_trace"]["path"] == str(trace_path)
    assert result["debug_dynamics_trace"]["sample_count"] == 1
    assert result["debug_dynamics_trace"]["gc_observation"]["registered_after_cleanup"] is False
    assert result["debug_dynamics_trace"]["gc_observation"]["settings_at_start"]["enabled"] == probe.gc.isenabled()
    assert result["debug_dynamics_trace"]["last_feedback_read_diagnostics"]["stages"]["backend_poll"]["interval_count"] == 1
    assert tuple(probe.gc.callbacks) == callbacks_before
    assert json.loads(trace_path.read_text(encoding="utf-8"))["kind"] == "synthetic"


def test_main_without_debug_does_not_register_gc_observer(monkeypatch, tmp_path):
    def forbidden_observer():
        pytest.fail("plain owner must not construct the diagnostic GC observer")

    def fake_run_probe(_args, **kwargs):
        assert kwargs["debug_trace"] is None
        return {"status": "OK", "mode": "hold"}

    monkeypatch.setattr(probe, "_ProbeGCObserver", forbidden_observer)
    monkeypatch.setattr(probe, "run_probe", fake_run_probe)
    output = tmp_path / "result.json"
    assert probe.main([
        "--mode", "hold", "--config", str(tmp_path / "unused.yaml"),
        "--rebot-source-root", str(tmp_path / "unused-vendor"),
        "--output", str(output), "--log", str(tmp_path / "runtime.log"),
    ]) == 0
    assert "debug_dynamics_trace" not in json.loads(output.read_text())


@pytest.mark.parametrize("export_error", [None, ValueError("unconsumed receipt"), TypeError("invalid snapshot"), KeyError("outcome")])
@pytest.mark.parametrize("interrupt_report", [False, True])
def test_main_result_export_error_still_writes_primary_failure_json(monkeypatch, tmp_path, export_error, interrupt_report):
    import control_station.b601_torque_console as console_module
    from control_station.b601_torque_console import automatic_report

    class Console:
        set_human_intervention = recording_failed = lambda *args: None

        def result_record(self):
            if export_error is None:
                return {"kind": "automatic_calibration_result", "recorded_monotonic_sec": 1.,
                    "operation_mode": "workspace", "state": "failed", "reason": "leg_timeout",
                    "outcome": {"severity": "warning", "reason": "leg_timeout"}}
            raise export_error

    class Server:
        url = "http://127.0.0.1:1"

        def __init__(self, *args, **kwargs):
            pass

        start = close = set_stop_handler = lambda *args: None
        assessment_status = status_response_diagnostics = lambda *args: {}

    def no_report(_path):
        # The primary control/cleanup result must already survive a slow or
        # interrupted report, independent of whether export itself succeeded.
        saved = json.loads((tmp_path / "b601_owner.json").read_text())
        assert saved["stop_reason"] == "original_acquisition_failure"
        assert saved["automatic_calibration_report_status"] == "PENDING"
        if export_error is None:
            assert saved["sample_storage"]["sample_archive_status"] == "COMPLETE"
            assert Path(saved["sample_storage"]["sample_archive_path"]).is_file()
        if interrupt_report:
            raise KeyboardInterrupt
        raise ValueError("terminal record missing")

    monkeypatch.setenv("DISPLAY", ":synthetic-offline")
    monkeypatch.setattr(probe, "_validate_cli", lambda args: None)
    monkeypatch.setattr(probe, "_automatic_calibration_controller_from_args", lambda args: Console())
    monkeypatch.setattr(console_module, "B601TorqueConsoleServer", Server)
    monkeypatch.setattr(probe.webbrowser, "open_new_tab", lambda url: True)
    monkeypatch.setattr(probe, "run_probe", lambda *args, **kwargs: {
        "status": "FAIL", "mode": "hold", "stop_reason": "original_acquisition_failure", "samples": []})
    monkeypatch.setattr(automatic_report, "save_automatic_report", no_report)
    output = tmp_path / "b601_owner.json"
    assert probe.main([
        "--mode", "hold", "--config", str(tmp_path / "unused.yaml"),
        "--rebot-source-root", str(tmp_path / "unused-vendor"),
        "--output", str(output), "--log", str(tmp_path / "runtime.log"),
        "--torque-console", "--torque-console-auto-calibration-profile", str(tmp_path / "unused-profile.json"),
    ]) == (130 if interrupt_report else 2)
    result = json.loads(output.read_text())
    if export_error is not None or not interrupt_report:
        assert result["status"] == result["execution_status"] == "ERROR"
    else:
        assert result["status"] == "FAIL"
        assert result["acquisition_outcome"]["reason"] == "leg_timeout"
    assert result["stop_reason"] == "original_acquisition_failure"
    if export_error is not None:
        assert type(export_error).__name__ in result["automatic_calibration_export_error"]
    assert result["automatic_calibration_report_error"] == (
        "operator_interrupted_report_after_control_cleanup" if interrupt_report else "terminal record missing")
    assert result["automatic_calibration_report_status"] == ("INTERRUPTED" if interrupt_report else "ERROR")
    assert result["debug_dynamics_trace"]["gc_observation"]["registered_after_cleanup"] is False


@pytest.mark.parametrize("checkpoint_saved", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_main_interrupt_after_cleanup_keeps_primary_result(monkeypatch, tmp_path, checkpoint_saved, failed):
    import control_station.b601_torque_console as console_module

    closed = []

    class Console:
        set_human_intervention = recording_failed = lambda *args: None

    class Server:
        url = "http://127.0.0.1:1"
        def __init__(self, *args, **kwargs):
            pass
        start = set_stop_handler = lambda *args: None
        def close(self):
            closed.append(True)

    output = tmp_path / "b601_owner.json"
    def interrupted(_args, **kwargs):
        if checkpoint_saved:
            kwargs["post_cleanup_checkpoint"]({
                "mode": "hold", "status": "FAIL" if failed else "STOPPED_WITH_WARNINGS",
                "execution_status": "FAILED" if failed else "STOPPED_BY_OPERATOR",
                "stop_reason": "coordinated_reference_lag_stop" if failed else "explicit_full_disable",
                "cleanup": {"device_cleanup_confirmed": True, "transport_release": {"completed": True}},
                "postprocessing": {"status": "PENDING", "stage": "summary"},
            })
            assert json.loads(output.read_text())["postprocessing"]["status"] == "PENDING"
        raise KeyboardInterrupt

    monkeypatch.setenv("DISPLAY", ":synthetic-offline")
    monkeypatch.setattr(probe, "_validate_cli", lambda args: None)
    monkeypatch.setattr(probe, "_automatic_calibration_controller_from_args", lambda args: Console())
    monkeypatch.setattr(console_module, "B601TorqueConsoleServer", Server)
    monkeypatch.setattr(probe.webbrowser, "open_new_tab", lambda url: True)
    monkeypatch.setattr(probe, "run_probe", interrupted)
    args = ["--mode", "hold", "--config", str(tmp_path/"unused.yaml"),
        "--rebot-source-root", str(tmp_path/"unused-vendor"), "--output", str(output),
        "--log", str(tmp_path/"runtime.log"), "--torque-console",
        "--torque-console-auto-calibration-profile", str(tmp_path/"unused-profile.json")]
    if checkpoint_saved:
        assert probe.main(args) == 130
        result = json.loads(output.read_text())
        assert result["cleanup"]["device_cleanup_confirmed"]
        assert result["cleanup"]["transport_release"]["completed"]
        assert result["postprocessing"]["status"] == "INTERRUPTED"
        assert result["execution_status"] == ("FAILED" if failed else "STOPPED_BY_OPERATOR")
        assert result["stop_reason"] == ("coordinated_reference_lag_stop" if failed else "explicit_full_disable")
    else:
        with pytest.raises(KeyboardInterrupt):
            probe.main(args)
        assert not output.exists()
    assert closed == [True]


def test_degraded_fault_hold_recovers_live_gravity_without_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    gravity = (0.0, 0.8, 0.4, 0.1, 0.0, 0.0)
    telemetry: dict[str, object] = {}
    live_dispatch_count = 0

    def fake_send_hold(
        _backend: object,
        _model: object,
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        nonlocal live_dispatch_count
        live_dispatch_count += 1
        return _fault_hold_command_receipt(
            accepted=True,
            sequence=live_dispatch_count,
        ), gravity, gravity

    backend = _FaultHoldBackend(
        feedback=(
            SimpleNamespace(
                valid=False,
                stale=True,
                positions_rad=None,
                fault_reason="simulated one-frame feedback loss",
            ),
            SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=q,
                fault_reason=None,
            ),
            SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=q,
                fault_reason=None,
            ),
            SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=q,
                fault_reason=None,
            ),
            SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=q,
                fault_reason=None,
            ),
            SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=q,
                fault_reason=None,
            ),
        ),
        cached_receipts=(_fault_hold_command_receipt(accepted=True, sequence=2),),
    )
    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        backend,
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=q,
        fault_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=SimpleNamespace(max_torque_rate_nm_s=(10.0,) * 6),
        gravity_scale=1.0,
        previous_tau=gravity,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=6,
        driver_kd_nm_s_per_rad=(1.0,) * 6,
        cached_gravity_tau_nm=gravity,
        degraded_telemetry=telemetry,
    )

    hold_count, _latest_tau, _first, _non_hold, full_receipt, errors = result
    assert hold_count == 6
    assert live_dispatch_count == 5
    assert len(backend.cached_dispatches) == 1
    assert full_receipt is None
    assert errors == []
    assert telemetry["entry_count"] == 1
    assert telemetry["recovery_count"] == 1
    assert telemetry["recovery_samples_required"] == 3
    assert telemetry["recovery_candidate_count"] == 0
    assert telemetry["state"] == "LIVE"


def test_degraded_entry_uses_first_live_frame_as_baseline_without_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    gravity = (0.0, 0.8, 0.4, 0.1, 0.0, 0.0)
    telemetry: dict[str, object] = {}
    live_dispatch_count = 0

    def fake_send_hold(
        _backend: object,
        _model: object,
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        nonlocal live_dispatch_count
        live_dispatch_count += 1
        return _fault_hold_command_receipt(
            accepted=True,
            sequence=live_dispatch_count,
        ), gravity, gravity

    backend = _FaultHoldBackend(
        feedback=(
            SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=q,
                fault_reason=None,
            ),
        ),
        cached_receipts=(_fault_hold_command_receipt(accepted=True),),
    )
    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        backend,
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=float("inf"),
        fault_qpos_rad=None,
        fault_qvel_rad_s=None,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=SimpleNamespace(max_torque_rate_nm_s=(10.0,) * 6),
        gravity_scale=1.0,
        previous_tau=gravity,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=2,
        driver_kd_nm_s_per_rad=(1.0,) * 6,
        cached_qpos_rad=q,
        cached_gravity_tau_nm=gravity,
        degraded_entry_reason="simulated missing active feedback",
        degraded_telemetry=telemetry,
    )

    hold_count, _latest_tau, _first, _non_hold, full_receipt, errors = result
    assert hold_count == 2
    assert live_dispatch_count == 1
    assert full_receipt is None
    assert errors == []
    assert telemetry["entry_count"] == 1
    assert telemetry["recovery_count"] == 0
    assert telemetry["recovery_candidate_count"] == 0
    assert telemetry["state"] == "DEGRADED"


def test_entry_dispatch_failure_uses_first_live_frame_as_recovery_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    recovered_q = (0.3, 0.9, 1.4, 0.1, -0.2, 0.3)
    gravity = (0.0, 0.8, 0.4, 0.1, 0.0, 0.0)
    telemetry: dict[str, object] = {}
    live_dispatch_count = 0

    def fake_send_hold(
        _backend: object,
        _model: object,
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        nonlocal live_dispatch_count
        live_dispatch_count += 1
        if live_dispatch_count <= 2:
            raise probe.LowGainProbeError("simulated entry dispatch failure")
        return _fault_hold_command_receipt(
            accepted=True,
            sequence=live_dispatch_count,
        ), gravity, gravity

    backend = _FaultHoldBackend(
        feedback=(
            SimpleNamespace(
                valid=True,
                stale=False,
                positions_rad=recovered_q,
                fault_reason=None,
            ),
        ),
        cached_receipts=(_fault_hold_command_receipt(accepted=True),),
    )
    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        backend,
        object(),
        qpos_rad=entry_q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=None,
        fault_qvel_rad_s=None,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=SimpleNamespace(max_torque_rate_nm_s=(10.0,) * 6),
        gravity_scale=1.0,
        previous_tau=gravity,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": False, "full_disable_requested": False},
        max_iterations=2,
        driver_kd_nm_s_per_rad=(1.0,) * 6,
        cached_qpos_rad=entry_q,
        cached_gravity_tau_nm=gravity,
        degraded_telemetry=telemetry,
    )

    hold_count, _latest_tau, _first, _non_hold, full_receipt, errors = result
    assert hold_count == 2
    assert live_dispatch_count == 3
    assert len(backend.cached_dispatches) == 1
    assert full_receipt is None
    assert errors == []
    assert telemetry["entry_count"] == 1
    assert telemetry["recovery_count"] == 0
    assert telemetry["state"] == "DEGRADED"
    assert telemetry["last_transition"] == "recovery_baseline"


def test_fault_hold_retries_one_incomplete_all_joint_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    attempts = 0
    telemetry: dict[str, object] = {}

    def fake_send_hold(
        _backend: object,
        _model: object,
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        nonlocal attempts
        attempts += 1
        return (
            _fault_hold_command_receipt(accepted=attempts > 1, sequence=attempts),
            (0.0,) * 6,
            (0.0,) * 6,
        )

    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        _FaultHoldBackend(),
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=q,
        fault_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=object(),
        gravity_scale=1.0,
        previous_tau=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=1,
        degraded_telemetry=telemetry,
    )

    hold_count, _latest_tau, _first, _non_hold, full_receipt, errors = result
    assert hold_count == 1
    assert attempts == 2
    assert full_receipt is None
    assert errors == []
    assert telemetry["tx_retry_count"] == 1
    assert telemetry["tx_retry_recovered_count"] == 1


def test_degraded_hold_disables_only_after_cached_frame_retry_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    gravity = (0.0, 0.8, 0.4, 0.1, 0.0, 0.0)
    telemetry: dict[str, object] = {}

    def fake_send_hold(
        _backend: object,
        _model: object,
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
        return _fault_hold_command_receipt(accepted=True), gravity, gravity

    backend = _FaultHoldBackend(
        feedback=(
            SimpleNamespace(
                valid=False,
                stale=True,
                positions_rad=None,
                fault_reason="simulated sustained feedback loss",
            ),
        ),
        cached_receipts=(
            _fault_hold_command_receipt(accepted=False, sequence=2),
            _fault_hold_command_receipt(accepted=False, sequence=3),
        ),
        allow_safe_stop=True,
    )

    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        backend,
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=q,
        fault_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=SimpleNamespace(max_torque_rate_nm_s=(10.0,) * 6),
        gravity_scale=1.0,
        previous_tau=gravity,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=2,
        cached_gravity_tau_nm=gravity,
        degraded_telemetry=telemetry,
    )

    hold_count, _latest_tau, _first, _non_hold, full_receipt, errors = result
    assert hold_count == 1
    assert len(backend.cached_dispatches) == 2
    assert backend.safe_stop_names == [tuple(probe._ALL_JOINTS)]
    assert full_receipt is not None and full_receipt.completed
    assert errors
    assert telemetry["escalation_reason"] == (
        "live and cached all-six-joint hold dispatch failed after retry"
    )


@pytest.mark.parametrize(
    ("degraded_entry_reason", "fault_entry", "fault_qpos", "fault_qvel"),
    (
        ("simulated missing active feedback", False, None, None),
        (
            None,
            True,
            (0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            (1.0,) * 6,
        ),
    ),
)
def test_degraded_entry_uses_cached_gravity_when_live_entry_is_unavailable(
    degraded_entry_reason: str | None,
    fault_entry: bool,
    fault_qpos: tuple[float, ...] | None,
    fault_qvel: tuple[float, ...] | None,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    gravity = (0.0, 0.8, 0.4, 0.1, 0.0, 0.0)
    telemetry: dict[str, object] = {}
    backend = _FaultHoldBackend(
        cached_receipts=(_fault_hold_command_receipt(accepted=True),),
    )

    result = probe._run_latched_fault_hold(
        backend,
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=fault_qpos,
        fault_qvel_rad_s=fault_qvel,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=SimpleNamespace(max_torque_rate_nm_s=(10.0,) * 6),
        gravity_scale=1.0,
        previous_tau=gravity,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=1,
        driver_kd_nm_s_per_rad=(1.0,) * 6,
        cached_qpos_rad=q,
        cached_gravity_tau_nm=gravity,
        degraded_entry_reason=degraded_entry_reason,
        fault_entry=fault_entry,
        degraded_telemetry=telemetry,
    )

    hold_count, _latest_tau, first, _non_hold, full_receipt, errors = result
    assert hold_count == 1
    assert [item["tau_nm"] for item in backend.cached_dispatches] == [gravity]
    assert first is not None and first.accepted
    assert full_receipt is None
    assert errors == []
    assert telemetry["entry_count"] == 1


def test_degraded_entry_without_driver_damping_reports_damping_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    gravity = (0.0, 0.8, 0.4, 0.1, 0.0, 0.0)
    backend = _FaultHoldBackend(
        cached_receipts=(_fault_hold_command_receipt(accepted=True),),
    )

    result = probe._run_latched_fault_hold(
        backend,
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=None,
        fault_qvel_rad_s=None,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=SimpleNamespace(max_torque_rate_nm_s=(10.0,) * 6),
        gravity_scale=1.0,
        previous_tau=gravity,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=1,
        driver_kd_nm_s_per_rad=(0.0,) * 6,
        cached_qpos_rad=q,
        cached_gravity_tau_nm=gravity,
        degraded_entry_reason="simulated unavailable feedback",
    )

    hold_count, _latest_tau, first, _non_hold, full_receipt, errors = result
    assert hold_count == 1
    assert first is not None and first.accepted
    assert full_receipt is None
    assert errors == []
    terminal = capsys.readouterr().out
    assert "!!! PROTECTIVE HOLD DEGRADED !!!" in terminal
    assert "DAMPING: UNAVAILABLE while feedback is untrusted" in terminal
    assert "!!! PROTECTIVE HOLD LATCHED !!!" in terminal
    assert "CONTROL: DEGRADED cached gravity" in terminal


def test_fault_hold_snapshot_rejects_out_of_limit_q() -> None:
    with pytest.raises(
        probe.LowGainProbeError,
        match="outside position acceptance limits",
    ):
        probe._validate_fault_hold_snapshot(
            (0.1, 3.2, 1.1, 0.0, -0.1, 0.2),
            qvel_rad_s=(0.0,) * 6,
            observed_age_sec=0.01,
            position_limits_rad=_fault_hold_limits(),
        )


def test_sigusr1_requests_full_disable() -> None:
    stop = {"requested": False, "full_disable_requested": False}

    probe._signal_handler(stop, probe.signal.SIGUSR1, None)

    assert stop == {"requested": True, "full_disable_requested": True}


def test_second_sigint_requests_full_disable() -> None:
    stop = {"requested": False, "full_disable_requested": False}

    probe._signal_handler(stop, probe.signal.SIGINT, None)
    assert stop == {"requested": True, "full_disable_requested": False}

    probe._signal_handler(stop, probe.signal.SIGINT, None)
    assert stop == {"requested": True, "full_disable_requested": True}


def test_drive_enable_countdown_is_logged_and_mirrored(tmp_path) -> None:
    stop = {"requested": False, "full_disable_requested": False}
    log_stream = io.StringIO()
    operator_output = tmp_path / "operator-tty.log"
    sleeps: list[float] = []

    completed = probe._drive_enable_countdown(
        stop,
        sleeper=sleeps.append,
        log_stream=log_stream,
        operator_tty_path=operator_output,
    )

    expected = "[DRIVE] 3\n[DRIVE] 2\n[DRIVE] 1\n[DRIVE] ENABLE\n"
    assert completed is True
    assert sleeps == [1.0, 1.0, 1.0]
    assert log_stream.getvalue() == expected
    assert operator_output.read_text(encoding="utf-8") == expected


def test_drive_enable_countdown_interrupts_before_enable(tmp_path) -> None:
    stop = {"requested": False, "full_disable_requested": False}
    log_stream = io.StringIO()

    def interrupt_after_first_tick(_seconds: float) -> None:
        stop["requested"] = True

    completed = probe._drive_enable_countdown(
        stop,
        sleeper=interrupt_after_first_tick,
        log_stream=log_stream,
        operator_tty_path=tmp_path / "operator-tty.log",
    )

    assert completed is False
    assert log_stream.getvalue() == "[DRIVE] 3\n[DRIVE] CANCELLED\n"
    assert "ENABLE" not in log_stream.getvalue()


def test_driver_torque_limit_receipt_payload_preserves_transaction_evidence() -> None:
    receipt = probe.B601DriverTorqueLimitReceipt(
        operation="apply_temporary_driver_torque_limits",
        requested_joint_names=("joint6",),
        target_limits_nm=(1.4,),
        prior_limits_nm=(14.0,),
        verified_limits_nm=(1.4,),
        applied_joint_names=("joint6",),
        restored_joint_names=(),
        failed_joint_names=(),
        errors=(),
        completed=True,
    )

    assert probe._driver_torque_limit_receipt_payload(receipt) == {
        "operation": "apply_temporary_driver_torque_limits",
        "requested_joint_names": ("joint6",),
        "target_limits_nm": (1.4,),
        "prior_limits_nm": (14.0,),
        "verified_limits_nm": (1.4,),
        "applied_joint_names": ("joint6",),
        "restored_joint_names": (),
        "failed_joint_names": (),
        "errors": (),
        "completed": True,
        "rollback_attempted": False,
        "rollback_completed": None,
    }


def test_slow_step_is_relative_and_bounded() -> None:
    q0 = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    limits = ((-2.8, 2.8), (0.0, 3.14), (0.0, 3.14), (-1.57, 1.57), (-1.57, 1.57), (-3.14, 3.14))
    qref, qvel, done = probe.plan_reference(
        q0,
        mode="slow-step",
        joint="joint3",
        elapsed_sec=1.0,
        step_rad=0.05,
        step_speed_rad_s=0.02,
        step_direction=1,
        position_limits_rad=limits,
    )

    assert np.allclose(qref, (0.1, 0.7, 1.12, 0.0, -0.1, 0.2))
    assert qvel == (0.0, 0.0, 0.02, 0.0, 0.0, 0.0)
    assert done is False


def test_slow_step_rejects_target_outside_urdf_limit() -> None:
    with pytest.raises(ValueError, match="outside URDF limits"):
        probe.plan_reference(
            (0.1, 0.7, 3.13, 0.0, -0.1, 0.2),
            mode="slow-step",
            joint="joint3",
            elapsed_sec=0.0,
            step_rad=0.05,
            step_speed_rad_s=0.02,
            step_direction=1,
            position_limits_rad=((-2.8, 2.8), (0.0, 3.14), (0.0, 3.14), (-1.57, 1.57), (-1.57, 1.57), (-3.14, 3.14)),
        )


def test_slow_step_accepts_exactly_twenty_degrees_and_rejects_more() -> None:
    limits = ((-3.14, 3.14),) * 6
    qref, _qvel, done = probe.plan_reference(
        (0.0,) * 6,
        mode="slow-step",
        joint="joint6",
        elapsed_sec=100.0,
        step_rad=probe._MAX_STEP_RAD,
        step_speed_rad_s=0.02,
        step_direction=1,
        position_limits_rad=limits,
    )

    assert qref[5] == pytest.approx(probe._MAX_STEP_RAD)
    assert done is True
    with pytest.raises(ValueError, match="step_rad must be"):
        probe.plan_reference(
            (0.0,) * 6,
            mode="slow-step",
            joint="joint6",
            elapsed_sec=0.0,
            step_rad=probe._MAX_STEP_RAD + 1e-6,
            step_speed_rad_s=0.02,
            step_direction=1,
            position_limits_rad=limits,
        )


def test_observed_target_excursion_has_twenty_degree_hard_limit() -> None:
    anchor = (0.0,) * 6
    at_limit = (0.0, 0.0, 0.0, 0.0, 0.0, probe._MAX_STEP_RAD)
    over_limit = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        probe._MAX_STEP_RAD + 1e-6,
    )

    assert probe.validate_target_excursion(
        at_limit,
        anchor_qpos_rad=anchor,
        target_joint="joint6",
    ) == pytest.approx(probe._MAX_STEP_RAD)
    with pytest.raises(probe.LowGainProbeError, match="target excursion abort"):
        probe.validate_target_excursion(
            over_limit,
            anchor_qpos_rad=anchor,
            target_joint="joint6",
        )


def _slow_step_samples(
    target_positions: tuple[float, ...],
    *,
    support_joint1_position: float = 0.0,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "qpos_rad": (
                support_joint1_position,
                0.0,
                0.0,
                0.0,
                0.0,
                target,
            ),
            "step_done": True,
        }
        for target in target_positions
    )


def test_slow_step_response_requires_real_correct_direction_motion() -> None:
    active = probe.resolve_active_joint_names()
    passed = probe.evaluate_slow_step_response(
        mode="slow-step",
        target_joint="joint6",
        step_rad=0.02,
        step_direction=1,
        initial_qpos_rad=(0.0,) * 6,
        samples=_slow_step_samples((0.0, 0.005, 0.01, 0.015, 0.019, 0.02)),
        active_joint_names=active,
    )
    insufficient = probe.evaluate_slow_step_response(
        mode="slow-step",
        target_joint="joint6",
        step_rad=0.02,
        step_direction=1,
        initial_qpos_rad=(0.0,) * 6,
        samples=_slow_step_samples((0.0004,) * 6),
        active_joint_names=active,
    )
    wrong_direction = probe.evaluate_slow_step_response(
        mode="slow-step",
        target_joint="joint6",
        step_rad=0.02,
        step_direction=1,
        initial_qpos_rad=(0.0,) * 6,
        samples=_slow_step_samples((-0.003,) * 6),
        active_joint_names=active,
    )

    assert passed["status"] == "PASS"
    assert passed["passed"] is True
    assert insufficient["status"] == "INSUFFICIENT_RESPONSE"
    assert insufficient["passed"] is False
    assert wrong_direction["status"] == "WRONG_DIRECTION"
    assert wrong_direction["passed"] is False


def test_slow_step_response_rejects_support_joint_drift() -> None:
    active = probe.resolve_active_joint_names()
    result = probe.evaluate_slow_step_response(
        mode="slow-step",
        target_joint="joint6",
        step_rad=0.02,
        step_direction=1,
        initial_qpos_rad=(0.0,) * 6,
        samples=_slow_step_samples((0.02,) * 6, support_joint1_position=0.01),
        active_joint_names=active,
    )

    assert result["status"] == "SUPPORT_DRIFT_EXCEEDED"
    assert result["support_peak_drift_rad"]["joint1"] == pytest.approx(0.01)


def test_assist_commissioning_response_does_not_accept_state_machine_only() -> None:
    config = probe.B601SingleJointAssistConfig(
        joint_name="joint6",
        offset_rad=0.1,
        max_abs_assist_torque_nm=0.05,
        countdown_sec=1.0,
        assist_duration_sec=2.0,
    )
    runtime = probe.B601SharedControlRuntime(config.shared_control_config())
    session = probe.B601SingleJointAssistSession(config, runtime)
    session.anchor_qpos_rad = (0.0,) * 6
    session.state = probe.B601AssistCommissioningState.COMPLETE
    samples = tuple(
        {
            "qpos_rad": (0.0, 0.0, 0.0, 0.0, 0.0, 0.00038),
            "shared_control_commissioning_state": "active",
            "shared_control": {
                "assist_torque_nm": (0.0, 0.0, 0.0, 0.0, 0.0, 0.05)
            },
        }
        for _ in range(6)
    )

    result = probe.evaluate_assist_commissioning_response(
        session=session,
        samples=samples,
        active_joint_names=probe.resolve_active_joint_names(),
        minimum_response_rad=0.001,
    )

    assert result["status"] == "INSUFFICIENT_RESPONSE"
    assert result["passed"] is False
    assert "0.022deg" in probe.format_assist_response_notice(result)


def test_effort_limits_are_read_from_urdf_and_scaled(tmp_path) -> None:
    effort_values = (36.0, 36.0, 36.0, 14.0, 14.0, 14.0)
    joints = "".join(
        f'<joint name="{name}" type="revolute">'
        f'<limit effort="{effort}" lower="-1" upper="1" velocity="1"/>'
        "</joint>"
        for name, effort in zip(probe._ALL_JOINTS, effort_values, strict=True)
    )
    urdf = tmp_path / "b601_test.urdf"
    urdf.write_text(f'<robot name="b601">{joints}</robot>', encoding="utf-8")

    limits = probe._resolve_effort_limits(urdf, scale=0.25)

    assert limits == (9.0, 9.0, 9.0, 3.5, 3.5, 3.5)


def test_parser_help_is_read_only() -> None:
    parser = probe.build_parser()
    help_text = parser.format_help()
    assert "--allow-motion" in help_text
    assert "duration completion itself enters" in help_text
    assert "protective hold" in help_text
    assert tuple(B601_RS_ARM_JOINT_NAMES) == tuple(probe._ALL_JOINTS)
    args = parser.parse_args(["--config", "config.yaml", "--rebot-source-root", "source"])
    assert args.joint == "joint1"
    assert "--support-joint" not in parser.format_help()
    assert args.target_kp is None
    assert args.target_kd is None
    assert args.kp == pytest.approx(0.0)
    assert args.kd == pytest.approx(0.0)
    assert args.driver_kd is None
    assert args.feedback_source == "parameter"
    assert args.allow_active_report is False
    assert args.auto_disable_active_report is False
    assert args.velocity_abort_rad_s == pytest.approx(5.0)
    assert args.acceleration_abort_rad_s2 == pytest.approx(80.0)
    assert args.gravity_scale == pytest.approx(1.0)
    assert args.gravity_joint_scales == pytest.approx(
        (1.0, 0.96, 1.0, 1.0, 1.0, 1.0)
    )
    assert args.gravity_joint_offsets_rad == pytest.approx((0.0,) * 6)
    assert args.gravity_joint_bias_nm == pytest.approx(
        (0.0, -0.22, -0.11, -0.08, 0.0, 0.0)
    )
    assert args.gravity_coulomb_friction_nm == pytest.approx((0.0,) * 6)
    assert args.gravity_coulomb_velocity_threshold_rad_s == pytest.approx(0.02)
    assert args.cartesian_assist_breakaway_positive_nm == pytest.approx(
        (0.0,) * 6
    )
    assert args.cartesian_assist_breakaway_negative_nm == pytest.approx(
        (0.0,) * 6
    )
    assert args.cartesian_assist_breakaway_velocity_fade_rad_s == pytest.approx(
        0.03
    )
    assert args.cartesian_assist_breakaway_intent_deadband_nm == pytest.approx(
        0.01
    )
    assert args.cartesian_assist_breakaway_intent_ramp_nm == pytest.approx(0.04)
    assert args.cartesian_assist_joint_torque_release_slew_nm_s is None
    assert args.apply_driver_torque_limits is False
    calibration_args = parser.parse_args(
        [
            "--mode",
            "gravity-calibration",
            "--config",
            "config.yaml",
            "--rebot-source-root",
            "source",
            "--allow-motion",
            "--auto-disable",
            "--apply-driver-torque-limits",
            "--confirmation",
            probe.CONFIRMATION,
        ]
    )
    assert calibration_args.calibration_duration_sec == pytest.approx(20.0)
    assert calibration_args.calibration_settling_sec == pytest.approx(2.0)
    probe._validate_cli(calibration_args)

    calibrated_args = parser.parse_args(
        [
            "--mode",
            "hold",
            "--config",
            "config.yaml",
            "--rebot-source-root",
            "source",
            "--allow-motion",
            "--auto-disable",
            "--apply-driver-torque-limits",
            "--confirmation",
            probe.CONFIRMATION,
            "--gravity-joint-scales",
            "1",
            "0.95",
            "1",
            "1",
            "1",
            "1",
            "--gravity-joint-offsets-rad",
            "0",
            "-0.06",
            "0.02",
            "0",
            "0",
            "0",
            "--gravity-joint-bias-nm",
            "0",
            "-0.70",
            "0.12",
            "0",
            "0",
            "0",
            "--gravity-coulomb-friction-nm",
            "0",
            "0.53",
            "0.49",
            "0",
            "0",
            "0",
            "--gravity-coulomb-velocity-threshold-rad-s",
            "0.02",
        ]
    )
    probe._validate_cli(calibrated_args)
    assert calibrated_args.gravity_joint_scales == pytest.approx(
        (1.0, 0.95, 1.0, 1.0, 1.0, 1.0)
    )
    assert calibrated_args.gravity_joint_offsets_rad == pytest.approx(
        (0.0, -0.06, 0.02, 0.0, 0.0, 0.0)
    )
    assert calibrated_args.gravity_joint_bias_nm == pytest.approx(
        (0.0, -0.70, 0.12, 0.0, 0.0, 0.0)
    )
    assert calibrated_args.gravity_coulomb_friction_nm == pytest.approx(
        (0.0, 0.53, 0.49, 0.0, 0.0, 0.0)
    )
    assert calibrated_args.gravity_coulomb_velocity_threshold_rad_s == pytest.approx(
        0.02
    )

    calibrated_args.gravity_joint_scales[1] = -0.1
    with pytest.raises(ValueError, match="six non-negative"):
        probe._validate_cli(calibrated_args)
    calibrated_args.gravity_joint_scales[1] = 0.95
    calibrated_args.gravity_coulomb_friction_nm[1] = -0.1
    with pytest.raises(ValueError, match="six non-negative"):
        probe._validate_cli(calibrated_args)


def test_active_report_control_feedback_requires_both_state_acknowledgements() -> None:
    parser = probe.build_parser()
    base = [
        "--mode",
        "hold",
        "--config",
        "config.yaml",
        "--rebot-source-root",
        "source",
        "--feedback-source",
        "active-report",
        "--allow-motion",
        "--auto-disable",
        "--apply-driver-torque-limits",
        "--confirmation",
        probe.CONFIRMATION,
    ]

    with pytest.raises(probe.LowGainProbeError, match="--allow-active-report"):
        probe._validate_cli(parser.parse_args(base))
    with pytest.raises(
        probe.LowGainProbeError,
        match="--auto-disable-active-report",
    ):
        probe._validate_cli(parser.parse_args([*base, "--allow-active-report"]))

    args = parser.parse_args(
        [
            *base,
            "--allow-active-report",
            "--auto-disable-active-report",
        ]
    )
    probe._validate_cli(args)

def test_feedback_stability_cli_is_explicitly_observation_only() -> None:
    parser = probe.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "feedback-stability",
            "--config",
            "config.yaml",
            "--rebot-source-root",
            "source",
            "--confirmation",
            probe.FEEDBACK_STABILITY_CONFIRMATION,
        ]
    )
    probe._validate_cli(args)

    motion_args = parser.parse_args(
        [
            "--mode",
            "feedback-stability",
            "--config",
            "config.yaml",
            "--rebot-source-root",
            "source",
            "--allow-motion",
            "--confirmation",
            probe.FEEDBACK_STABILITY_CONFIRMATION,
        ]
    )
    with pytest.raises(probe.LowGainProbeError, match="observation-only"):
        probe._validate_cli(motion_args)

    driver_write_args = parser.parse_args(
        [
            "--mode",
            "feedback-stability",
            "--config",
            "config.yaml",
            "--rebot-source-root",
            "source",
            "--apply-driver-torque-limits",
            "--confirmation",
            probe.FEEDBACK_STABILITY_CONFIRMATION,
        ]
    )
    with pytest.raises(probe.LowGainProbeError, match="observation-only"):
        probe._validate_cli(driver_write_args)


def test_startup_replay_cli_requires_gravity_only_observation_profile() -> None:
    parser = probe.build_parser()
    base = [
        "--mode",
        "startup-replay",
        "--config",
        "config.yaml",
        "--rebot-source-root",
        "source",
        "--allow-motion",
        "--auto-disable",
        "--apply-driver-torque-limits",
        "--confirmation",
        probe.CONFIRMATION,
    ]

    with pytest.raises(probe.LowGainProbeError, match="requires --kp 0"):
        probe._validate_cli(parser.parse_args([*base, "--kp", "0.5"]))

    args = parser.parse_args(base)
    probe._validate_cli(args)
    qref, qvel_ref, done = probe.plan_reference(
        (0.1,) * 6,
        mode="startup-replay",
        joint="joint1",
        elapsed_sec=0.0,
        step_rad=0.0,
        step_speed_rad_s=0.1,
        step_direction=1,
        position_limits_rad=((-1.0, 1.0),) * 6,
    )
    assert qref == (0.1,) * 6
    assert qvel_ref == (0.0,) * 6
    assert done is True


def test_enabled_mode_requires_verified_driver_torque_limits_for_startup_lock() -> None:
    args = probe.build_parser().parse_args(
        [
            "--mode",
            "hold",
            "--config",
            "config.yaml",
            "--rebot-source-root",
            "source",
            "--allow-motion",
            "--auto-disable",
            "--confirmation",
            probe.CONFIRMATION,
        ]
    )

    with pytest.raises(probe.LowGainProbeError, match="0x700B"):
        probe._validate_cli(args)


def test_driver_torque_limit_roundtrip_cli_requires_explicit_authority() -> None:
    parser = probe.build_parser()
    base = [
        "--mode",
        "driver-torque-limit-roundtrip",
        "--config",
        "config.yaml",
        "--rebot-source-root",
        "source",
    ]

    with pytest.raises(probe.LowGainProbeError, match="requires --allow-motion"):
        probe._validate_cli(parser.parse_args(base))

    args = parser.parse_args(
        [
            *base,
            "--allow-motion",
            "--auto-disable",
            "--confirmation",
            probe.DRIVER_TORQUE_LIMIT_CONFIRMATION,
        ]
    )
    probe._validate_cli(args)


def _parse_cartesian_args(*extra: str) -> object:
    return probe.build_parser().parse_args(
        [
            "--mode",
            "cartesian-hold",
            "--config",
            "config.yaml",
            "--rebot-source-root",
            "source",
            "--allow-motion",
            "--auto-disable",
            "--apply-driver-torque-limits",
            "--confirmation",
            probe.CONFIRMATION,
            *extra,
        ]
    )


def test_cartesian_cli_uses_all_six_joints() -> None:
    args = _parse_cartesian_args()

    probe._validate_cli(args)
    assert probe.resolve_active_joint_names() == tuple(probe._ALL_JOINTS)


def test_cartesian_breakaway_torque_requires_explicit_assist_mode() -> None:
    args = probe.build_parser().parse_args(
        [
            "--mode",
            "hold",
            "--config",
            "config.yaml",
            "--rebot-source-root",
            "source",
            "--cartesian-assist-breakaway-positive-nm",
            "0",
            "0.1",
            "0",
            "0",
            "0",
            "0",
        ]
    )

    with pytest.raises(
        probe.LowGainProbeError,
        match="breakaway torque requires --cartesian-assist",
    ):
        probe._validate_cartesian_assist_cli(args)


def _parse_pose_args(*extra: str) -> object:
    return probe.build_parser().parse_args(
        [
            "--mode",
            "pose-hold",
            "--config",
            "config.yaml",
            "--rebot-source-root",
            "source",
            "--allow-motion",
            "--auto-disable",
            "--confirmation",
            probe.CONFIRMATION,
            *extra,
        ]
    )


def test_pose_cli_uses_all_six_joints() -> None:
    args = _parse_pose_args()
    args.apply_driver_torque_limits = True
    probe._validate_cli(args)
    assert probe.resolve_active_joint_names() == tuple(probe._ALL_JOINTS)


def test_guided_pose_axis_cli_requires_pose_mode_and_complete_safe_envelope() -> None:
    valid = _parse_pose_args(
        "--apply-driver-torque-limits",
        "--guided-pose-axis-protocol",
        "--duration-sec",
        "180",
        "--torque-rise-time-sec",
        "0.5",
        "--cartesian-max-displacement-m",
        "0.06",
        "--rotational-max-displacement-rad",
        "0.35",
    )
    probe._validate_cli(valid)

    assert valid.guided_pose_translation_m == pytest.approx(0.03)
    assert valid.guided_pose_rotation_rad == pytest.approx(math.radians(8.0))

    wrong_mode = _parse_cartesian_args("--guided-pose-axis-protocol")
    with pytest.raises(probe.LowGainProbeError, match="requires --mode pose-hold"):
        probe._validate_cli(wrong_mode)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            ("--cartesian-max-displacement-m", "0.03"),
            "translation target must remain strictly below",
        ),
        (
            (
                "--cartesian-max-displacement-m",
                "0.06",
                "--rotational-max-displacement-rad",
                str(math.radians(8.0)),
            ),
            "rotation target must remain strictly below",
        ),
        (
            (
                "--cartesian-max-displacement-m",
                "0.06",
                "--rotational-max-displacement-rad",
                "0.35",
                "--duration-sec",
                "125",
            ),
            "must cover startup handoff",
        ),
    ],
)
def test_guided_pose_axis_cli_fails_closed(
    extra: tuple[str, ...],
    message: str,
) -> None:
    args = _parse_pose_args(
        "--apply-driver-torque-limits",
        "--guided-pose-axis-protocol",
        *extra,
    )

    with pytest.raises(probe.LowGainProbeError, match=message):
        probe._validate_cli(args)


def test_interactive_pose_axis_cli_uses_timeout_not_timed_schedule() -> None:
    args = _parse_pose_args(
        "--apply-driver-torque-limits",
        "--interactive-pose-axis-protocol",
        "--duration-sec",
        "1",
        "--cartesian-max-displacement-m",
        "0.06",
        "--rotational-max-displacement-rad",
        "0.35",
    )

    probe._validate_cli(args)


def test_pose_axis_protocols_are_mutually_exclusive() -> None:
    args = _parse_pose_args(
        "--apply-driver-torque-limits",
        "--guided-pose-axis-protocol",
        "--interactive-pose-axis-protocol",
        "--duration-sec",
        "180",
        "--cartesian-max-displacement-m",
        "0.06",
        "--rotational-max-displacement-rad",
        "0.35",
    )

    with pytest.raises(probe.LowGainProbeError, match="mutually exclusive"):
        probe._validate_cli(args)


def test_interactive_pose_axis_cli_requires_pose_mode() -> None:
    args = _parse_cartesian_args("--interactive-pose-axis-protocol")

    with pytest.raises(
        probe.LowGainProbeError,
        match="--interactive-pose-axis-protocol requires --mode pose-hold",
    ):
        probe._validate_cli(args)


def test_driver_damping_requires_verified_driver_torque_limits() -> None:
    args = _parse_pose_args("--kd", "0", "--driver-kd", "1")

    with pytest.raises(probe.LowGainProbeError, match="driver-torque-limits"):
        probe._validate_cli(args)

    args.apply_driver_torque_limits = True
    probe._validate_cli(args)

    args.kd = 1.0
    with pytest.raises(probe.LowGainProbeError, match="software --kd 0"):
        probe._validate_cli(args)


def test_driver_damping_accepts_explicit_six_joint_gains() -> None:
    args = _parse_pose_args(
        "--kd",
        "0",
        "--driver-kd",
        "0.4",
        "0.6",
        "0.6",
        "0.3",
        "0.25",
        "0.2",
    )
    args.apply_driver_torque_limits = True

    probe._validate_cli(args)

    assert probe._one_or_six_non_negative(args.driver_kd, name="driver_kd") == (
        0.4,
        0.6,
        0.6,
        0.3,
        0.25,
        0.2,
    )


def test_driver_damping_rejects_ambiguous_joint_count() -> None:
    args = _parse_pose_args("--kd", "0", "--driver-kd", "0.4", "0.6")
    args.apply_driver_torque_limits = True

    with pytest.raises(ValueError, match="one or six"):
        probe._validate_cli(args)
