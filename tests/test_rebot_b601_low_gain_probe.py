from __future__ import annotations

import io
import math
import os
import pty
import termios
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hardware.rebot_b601_rs import B601HardwareState, B601JointState
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


@pytest.mark.parametrize("failure", ["interrupt", "first_rx_error", "persistent_rx_error"])
def test_enabled_startup_exit_attempts_recovery_before_stop(monkeypatch, tmp_path, failure):
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

        def connect(self):
            pass

        def poll_feedback(self, **kwargs):
            self.reads += 1
            clock[0] += 0.05
            events.append("read")
            if self.enabled and (
                failure == "persistent_rx_error"
                or (failure == "first_rx_error" and self.reads == 2)
            ):
                raise OSError("injected RX failure")
            return SimpleNamespace(
                valid=True, stale=False, positions_rad=q, sequence=self.reads,
                receive_timestamp_sec=clock[0], joints=(),
            )

        def safe_stop(self, **kwargs):
            events.append("disable")
            return receipt()

        def enable(self, **kwargs):
            self.enabled = True
            events.append("enable")
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
    with pytest.raises(probe.LowGainProbeError, match="operator interrupt|injected RX failure"):
        probe.run_probe(args)
    after_enable = events[events.index("enable") + 1:]
    if failure == "persistent_rx_error":
        assert "hold" not in after_enable
        assert after_enable.index("read") < after_enable.index("disable")
    else:
        assert after_enable.index("hold") < after_enable.index("disable")
        assert after_enable.count("read") >= 2
    assert after_enable[-2:] == ["restore", "close"]


def test_active_report_warmup_requires_advancing_type24_and_disables_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
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
            assert kwargs == {"require_host_freshness": True}
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
        probe.run_probe(args)

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
        "semantics": (
            "passive_motorbridge_operation_status_cache_unverified; "
            "no_extra_can_request; not command-response correlated"
        ),
    }


def test_vendor_status_cache_diagnostics_does_not_fail_when_cache_is_absent() -> None:
    result = probe.vendor_status_cache_diagnostics(SimpleNamespace())

    assert result["complete"] is False
    assert result["velocity_rad_s"] is None
    assert result["torque_nm"] is None
    assert result["errors"] == (
        "vendor_status_velocity_rad_s: unavailable",
        "vendor_status_torque_nm: unavailable",
    )


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
    capsys: pytest.CaptureFixture[str],
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
    terminal = capsys.readouterr().out
    assert "!!! PROTECTIVE HOLD LATCHED !!!" in terminal
    assert "CONTROL: live gravity compensation and damping are ACTIVE." in terminal
    assert "!!! PROTECTIVE HOLD DEGRADED !!!" in terminal
    assert "DAMPING: motor-local damping remains ACTIVE." in terminal
    assert "REASON: fault hold: protective hold feedback is not complete" in terminal


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
        (1.0, 0.95, 1.0, 1.0, 1.0, 1.0)
    )
    assert args.gravity_joint_offsets_rad == pytest.approx((0.0,) * 6)
    assert args.gravity_joint_bias_nm == pytest.approx(
        (0.0, -0.105, -0.1, -0.05, 0.0, 0.0)
    )
    assert args.gravity_coulomb_friction_nm == pytest.approx((0.0,) * 6)
    assert args.gravity_coulomb_velocity_threshold_rad_s == pytest.approx(0.0)
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
