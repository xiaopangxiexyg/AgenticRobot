from __future__ import annotations

import io
import os
import pty
import termios
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import B601_RS_ARM_JOINT_NAMES
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe

pytestmark = [pytest.mark.unit, pytest.mark.contract]


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


def test_startup_lock_profile_uses_per_joint_yaml_gains_and_final_bounds() -> None:
    profile = probe.build_startup_lock_profile(
        mit_kp_nm_per_rad=(50.0, 150.0, 150.0, 50.0, 50.0, 50.0),
        mit_kd_nm_s_per_rad=(3.0, 10.0, 10.0, 5.0, 4.0, 4.0),
        torque_limits_nm=(9.0, 9.0, 9.0, 3.5, 3.5, 3.5),
        torque_rise_time_sec=0.5,
        velocity_limit_rad_s=5.0,
    )

    assert profile.position_kp_nm_per_rad == (50.0, 150.0, 150.0, 50.0, 50.0, 50.0)
    assert profile.damping_kd_nm_s_per_rad == (3.0, 10.0, 10.0, 5.0, 4.0, 4.0)
    assert profile.max_torque_rate_nm_s == (18.0, 18.0, 18.0, 7.0, 7.0, 7.0)
    assert profile.max_abs_overlay_torque_nm == (0.0,) * 6
    assert profile.metadata["gain_source"] == "rebotarm_rs_yaml.MIT"


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


def test_startup_lock_blend_preserves_gravity_vector_and_handoffs_overlay() -> None:
    startup = probe.B601MitTorqueBreakdown(
        position_torque_nm=(1.0,) * 6,
        damping_torque_nm=(2.0,) * 6,
        gravity_torque_nm=(4.0, -3.0, 2.0, -1.0, 0.5, -0.25),
        overlay_requested_nm=(0.0,) * 6,
        overlay_effective_nm=(0.0,) * 6,
        requested_torque_nm=(7.0, 7.0, 7.0, 7.0, 7.5, 6.75),
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
        gravity_fraction=0.5,
        handoff_fraction=0.5,
    )
    assert halfway.position_torque_nm == pytest.approx((0.55,) * 6)
    assert halfway.damping_torque_nm == pytest.approx((1.1,) * 6)
    assert halfway.gravity_torque_nm == pytest.approx(
        tuple(0.5 * value for value in startup.gravity_torque_nm)
    )
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
    for index, value in ((1, -6.0001e-3), (2, -3.0001e-3)):
        raw_values = [0.0] * 6
        raw_values[index] = value
        raw = tuple(raw_values)

        normalized, normalized_joint_names = probe.normalize_feedback_boundary_qpos(raw)

        assert normalized == raw
        assert normalized_joint_names == ()
        with pytest.raises(probe.LowGainProbeError, match="outside URDF limits"):
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
    assert boundary["applied_joint_counts"] == {"joint2": 3, "joint3": 3}
    assert boundary["intervals_rad"] == {
        "joint2": (-6.0e-3, 0.0),
        "joint3": (-3.0e-3, 0.0),
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
        (-1.57, 1.57),
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


def test_latched_protective_hold_references_current_pose_after_manual_drag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q0 = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    q1 = (0.1, 0.71, 1.1, 0.0, -0.1, 0.2)
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
    )

    hold_count, _latest_tau, _first, non_hold, full, errors = result
    assert hold_count == 3
    assert command_qrefs[:2] == [q0, q0]
    assert command_qrefs[2] == q1
    assert measured_qvels[2][1] > 0.0
    assert non_hold is None
    assert full is None
    assert errors == []


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


def test_latched_fault_hold_uses_cached_gravity_on_invalid_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    gravity = (0.0, 0.8, 0.4, 0.1, 0.0, 0.0)
    command_qrefs: list[tuple[float, ...]] = []
    telemetry: dict[str, object] = {}

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
        cached_gravity_tau_nm=gravity,
        degraded_telemetry=telemetry,
    )

    hold_count, _latest_tau, first_receipt, non_hold_receipt, full_receipt, errors = result
    assert hold_count == 2
    assert command_qrefs == [q]
    assert len(backend.cached_dispatches) == 1
    assert backend.cached_dispatches[0]["kp"] == (0.0,) * 6
    assert backend.cached_dispatches[0]["kd"] == (1.0,) * 6
    assert backend.cached_dispatches[0]["tau_nm"] == gravity
    assert backend.cached_dispatches[0]["joint_names"] == probe._ALL_JOINTS
    assert first_receipt is not None and first_receipt.accepted
    assert non_hold_receipt is None
    assert full_receipt is None
    assert errors == []
    assert telemetry["entry_count"] == 1
    assert telemetry["command_count"] == 2
    assert telemetry["escalation_reason"] is None


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
        max_iterations=3,
        driver_kd_nm_s_per_rad=(1.0,) * 6,
        cached_gravity_tau_nm=gravity,
        degraded_telemetry=telemetry,
    )

    hold_count, _latest_tau, _first, _non_hold, full_receipt, errors = result
    assert hold_count == 3
    assert live_dispatch_count == 2
    assert len(backend.cached_dispatches) == 1
    assert full_receipt is None
    assert errors == []
    assert telemetry["entry_count"] == 1
    assert telemetry["recovery_count"] == 1


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


def test_fault_hold_snapshot_rejects_out_of_limit_q() -> None:
    with pytest.raises(probe.LowGainProbeError, match="outside URDF limits"):
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
    assert "--allow-motion" in parser.format_help()
    assert tuple(B601_RS_ARM_JOINT_NAMES) == tuple(probe._ALL_JOINTS)
    args = parser.parse_args(["--config", "config.yaml", "--rebot-source-root", "source"])
    assert args.joint == "joint1"
    assert "--support-joint" not in parser.format_help()
    assert args.target_kp is None
    assert args.target_kd is None
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
            "--confirmation",
            probe.CONFIRMATION,
        ]
    )
    assert calibration_args.calibration_duration_sec == pytest.approx(20.0)
    probe._validate_cli(calibration_args)


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
            "--confirmation",
            probe.CONFIRMATION,
            *extra,
        ]
    )


def test_cartesian_cli_uses_all_six_joints() -> None:
    args = _parse_cartesian_args()

    probe._validate_cli(args)
    assert probe.resolve_active_joint_names() == tuple(probe._ALL_JOINTS)


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
    probe._validate_cli(_parse_pose_args())
    assert probe.resolve_active_joint_names() == tuple(probe._ALL_JOINTS)


def test_driver_damping_requires_verified_driver_torque_limits() -> None:
    args = _parse_pose_args("--kd", "0", "--driver-kd", "1")

    with pytest.raises(probe.LowGainProbeError, match="driver-torque-limits"):
        probe._validate_cli(args)

    args.apply_driver_torque_limits = True
    probe._validate_cli(args)

    args.kd = 1.0
    with pytest.raises(probe.LowGainProbeError, match="software --kd 0"):
        probe._validate_cli(args)
