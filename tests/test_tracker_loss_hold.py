from __future__ import annotations

import numpy as np
import pytest

from robot_motion.runtime import EffectorCommand, ObservationChannel, TeleopCommand
from robot_motion.safety import MotionSafetyConfig, SafetyMonitor
from teleop_runtime.diagnostics import FaultEvent, FaultSeverity, TeleopRunState
from teleop_runtime.session import (
    TRACKER_HOLD_AUTOMATIC,
    TRACKER_HOLD_GRACE,
    TRACKER_HOLD_MANUAL,
    TRACKER_HOLD_RECOVERING,
    TeleopSession,
    TeleopSessionConfig,
)


def _pose(x: float = 0.0) -> np.ndarray:
    value = np.eye(4, dtype=float)
    value[0, 3] = float(x)
    return value


def _channel(*, value: np.ndarray, timestamp: float, valid: bool = True) -> ObservationChannel:
    return ObservationChannel(
        channel_name="right_wrist_pose",
        capability="pose6d",
        side="right",
        value=value,
        frame_name="desk",
        timestamp=float(timestamp),
        source_name="test_tracker",
        valid=bool(valid),
        critical=True,
    )


def test_invalid_pose_counter_resets_only_after_payload_validation() -> None:
    monitor = SafetyMonitor(
        MotionSafetyConfig(
            invalid_frame_limit=3,
            stale_timeout_sec=10.0,
            required_channels=("right_wrist_pose",),
        )
    )
    malformed = np.zeros((4, 4), dtype=float)

    assert monitor.evaluate_channels(
        {"right_wrist_pose": _channel(value=malformed, timestamp=0.0)},
        0.0,
    ) == []
    assert monitor.evaluate_channels(
        {"right_wrist_pose": _channel(value=malformed, timestamp=0.1)},
        0.1,
    ) == []
    violations = monitor.evaluate_channels(
        {"right_wrist_pose": _channel(value=malformed, timestamp=0.2)},
        0.2,
    )
    assert [violation.code for violation in violations] == ["invalid_pose"]
    assert violations[0].metadata["invalid_count"] == 3

    assert monitor.evaluate_channels(
        {"right_wrist_pose": _channel(value=_pose(), timestamp=0.3)},
        0.3,
    ) == []
    assert monitor.evaluate_channels(
        {"right_wrist_pose": _channel(value=malformed, timestamp=0.4)},
        0.4,
    ) == []


class _TrackerSource:
    name = "test_tracker"

    def __init__(self) -> None:
        self.timestamp = 0.0
        self.valid = True
        self.pose = _pose()

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def expected_channel_names(self) -> tuple[str, ...]:
        return ("right_wrist_pose",)

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        return {
            "right_wrist_pose": _channel(
                value=self.pose,
                timestamp=self.timestamp,
                valid=self.valid,
            )
        }


class _Rig:
    name = "test_rig"

    def __init__(self) -> None:
        self.sent: list[TeleopCommand] = []
        self.holds: list[str] = []

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    def send_commands(self, command: TeleopCommand, phase_recorder=None) -> None:
        del phase_recorder
        self.sent.append(command)

    def hold_motion(self, reason: str = "manual hold") -> None:
        self.holds.append(str(reason))


class _FailingHoldRig(_Rig):
    def hold_motion(self, reason: str = "manual hold") -> None:
        super().hold_motion(reason)
        raise RuntimeError("simulated hold transport failure")


class _Retargeter:
    name = "test_retargeter"

    def retarget(self, observation, calibration) -> TeleopCommand:
        del calibration
        command = TeleopCommand(timestamp=observation.timestamp)
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.full(6, float(observation.timestamp or 0.0), dtype=float),
            )
        )
        return command


def _session(
    *,
    healthy_frames: int = 2,
    rig: _Rig | None = None,
    command_guard=None,
) -> tuple[TeleopSession, _TrackerSource, _Rig]:
    source = _TrackerSource()
    rig = _Rig() if rig is None else rig
    session = TeleopSession(
        config=TeleopSessionConfig(
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
            recovery_healthy_frame_count=healthy_frames,
            tracker_loss_hold_enabled=True,
            tracker_channel_names=("right_wrist_pose",),
            tracker_auto_hold_sec=0.1,
            tracker_manual_hold_sec=1.0,
        ),
        sources=[source],
        retargeters=[_Retargeter()],
        rig=rig,
        command_guard=command_guard,
        safety_config=MotionSafetyConfig(
            freeze_on_fault=True,
            invalid_frame_limit=3,
            stale_timeout_sec=10.0,
            required_channels=("right_wrist_pose",),
            minimum_control_hz=0.0,
        ),
    )
    session.start()
    session.set_calibration_ready()
    session.mark_debug_passed()
    session.arm_hardware()
    return session, source, rig


def test_default_tracker_manual_hold_retains_one_second_auto_recovery_window() -> None:
    config = TeleopSessionConfig()

    assert config.tracker_auto_hold_sec == pytest.approx(0.1)
    assert config.tracker_manual_hold_sec == pytest.approx(1.0)


def test_pre_dispatch_command_guard_holds_and_faults_without_sending_command() -> None:
    def guard(_command, _observation, now: float):
        return FaultEvent(
            code="test_final_collision_rejected",
            message="test collision rejection",
            severity=FaultSeverity.ERROR,
            timestamp=now,
            component="test",
        )

    session, source, rig = _session(command_guard=guard)
    try:
        result = _step(session, source, 0.0, valid=True)

        assert result.state == TeleopRunState.FAULT
        assert result.command is None
        assert not result.command_dispatched
        assert [fault.code for fault in result.faults] == ["test_final_collision_rejected"]
        assert len(rig.sent) == 0
        assert len(rig.holds) == 1
    finally:
        session.close()


def _step(
    session: TeleopSession,
    source: _TrackerSource,
    now: float,
    *,
    valid: bool,
):
    source.timestamp = float(now)
    source.valid = bool(valid)
    return session.run_step(now=float(now))


def test_tracker_loss_grace_auto_hold_and_recovery_keep_last_action_observable() -> None:
    session, source, rig = _session()
    try:
        running = _step(session, source, 0.0, valid=True)
        assert running.state == TeleopRunState.RUNNING
        assert running.command_dispatched
        assert len(rig.sent) == 1

        grace = _step(session, source, 0.05, valid=False)
        assert grace.state == TeleopRunState.RUNNING
        assert grace.motion_hold_mode == TRACKER_HOLD_GRACE
        assert grace.tracker_loss_duration_sec == pytest.approx(0.05)
        assert grace.command is not None
        assert not grace.command_dispatched
        assert len(rig.sent) == 1
        assert rig.holds == []

        at_threshold = _step(session, source, 0.1, valid=False)
        assert at_threshold.motion_hold_mode == TRACKER_HOLD_GRACE
        assert at_threshold.state == TeleopRunState.RUNNING
        assert rig.holds == []

        automatic = _step(session, source, 0.11, valid=False)
        assert automatic.state == TeleopRunState.FROZEN
        assert automatic.motion_hold_mode == TRACKER_HOLD_AUTOMATIC
        assert automatic.command is not None
        assert automatic.command.metadata["motion_hold_mode"] == TRACKER_HOLD_AUTOMATIC
        assert not automatic.command_dispatched
        assert len(rig.holds) == 1
        assert all(fault.code != "FAULT" for fault in automatic.faults)

        first_healthy = _step(session, source, 0.12, valid=True)
        assert first_healthy.state == TeleopRunState.FROZEN
        assert first_healthy.command is not None
        recovering = _step(session, source, 0.13, valid=True)
        assert recovering.state == TeleopRunState.RECOVERING
        assert recovering.motion_hold_mode == TRACKER_HOLD_RECOVERING

        session.finish_recovery()
        resumed = _step(session, source, 0.14, valid=True)
        assert resumed.state == TeleopRunState.RUNNING
        assert resumed.motion_hold_mode == ""
        assert resumed.command_dispatched
        assert len(rig.sent) == 2
    finally:
        session.close()


def test_tracker_reacquisition_inside_grace_resumes_without_rig_hold() -> None:
    session, source, rig = _session()
    try:
        _step(session, source, 0.0, valid=True)
        grace = _step(session, source, 0.05, valid=False)
        assert grace.motion_hold_mode == TRACKER_HOLD_GRACE
        assert not grace.command_dispatched

        resumed = _step(session, source, 0.09, valid=True)
        assert resumed.state == TeleopRunState.RUNNING
        assert resumed.motion_hold_mode == ""
        assert resumed.command_dispatched
        assert rig.holds == []
    finally:
        session.close()


def test_collection_copy_keeps_held_action_out_of_motion_dispatch_path() -> None:
    from teleop_runtime.collection.run_collect import (
        CollectControlState,
        _collection_step_is_recordable,
        _reset_recording_state,
    )
    from teleop_runtime.debug.run_vive_rm65_real_teleop import _collection_step_result

    session, source, _rig = _session()
    try:
        _step(session, source, 0.0, valid=True)
        automatic = _step(session, source, 0.11, valid=False)
        assert automatic.command is not None

        collection_result = _collection_step_result(
            automatic,
            active_command=automatic.command,
            gripper_command=None,
        )
        assert collection_result.command is not None
        assert not collection_result.command_dispatched
        assert collection_result.motion_hold_mode == TRACKER_HOLD_AUTOMATIC
        assert collection_result.tracker_loss_channels == ("right_wrist_pose",)

        control = CollectControlState(recording_active=True)
        assert _collection_step_is_recordable(control, collection_result)
        control.recording_paused = True
        control.recording_pause_reason = "tracker_loss/manual_hold"
        assert not _collection_step_is_recordable(control, collection_result)
        _reset_recording_state(control)
        assert not control.recording_active
        assert not control.recording_paused
        assert control.recording_pause_reason == ""
    finally:
        session.close()


def test_tracker_loss_manual_hold_is_latched_until_explicit_home_recalibration() -> None:
    session, source, rig = _session(healthy_frames=1)
    try:
        _step(session, source, 0.0, valid=True)
        _step(session, source, 0.11, valid=False)
        manual = _step(session, source, 1.01, valid=False)

        assert manual.state == TeleopRunState.DEBUG_ONLY
        assert manual.motion_hold_mode == TRACKER_HOLD_MANUAL
        assert manual.home_recalibration_required
        assert len(rig.holds) == 2
        assert any(fault.code == "tracker_target_manual_hold" for fault in manual.faults)

        tracker_returned = _step(session, source, 1.02, valid=True)
        assert tracker_returned.state == TeleopRunState.DEBUG_ONLY
        assert tracker_returned.home_recalibration_required
        assert not tracker_returned.command_dispatched
        with pytest.raises(RuntimeError, match="runtime home"):
            session.arm_hardware()

        session.mark_home_recalibrated("test home captured")
        session.arm_hardware()
        resumed = _step(session, source, 1.03, valid=True)
        assert resumed.state == TeleopRunState.RUNNING
        assert not resumed.home_recalibration_required
    finally:
        session.close()


def test_reacquisition_frame_cannot_bypass_manual_threshold_after_observed_loss() -> None:
    session, source, _rig = _session(healthy_frames=1)
    try:
        _step(session, source, 0.0, valid=True)
        grace = _step(session, source, 0.05, valid=False)
        assert grace.motion_hold_mode == TRACKER_HOLD_GRACE

        # Even if the next control observation is already fresh, the previously
        # observed loss interval exceeded one second and must latch manual hold.
        reacquired = _step(session, source, 1.01, valid=True)
        assert reacquired.state == TeleopRunState.DEBUG_ONLY
        assert reacquired.home_recalibration_required
        assert reacquired.motion_hold_mode == TRACKER_HOLD_MANUAL
        assert any(fault.code == "tracker_target_manual_hold" for fault in reacquired.faults)
    finally:
        session.close()


def test_stale_tracker_reacquisition_resets_pose_jump_baseline() -> None:
    session, source, _rig = _session(healthy_frames=1)
    try:
        _step(session, source, 0.0, valid=True)

        # The source still exposes its cached pose as valid, but its timestamp no
        # longer advances.  Tracker freshness, rather than the source's valid bit,
        # must trigger the automatic hold at 0.1 s.
        source.timestamp = 0.0
        source.valid = True
        stale = session.run_step(now=0.11)
        assert stale.state == TeleopRunState.FROZEN
        assert stale.motion_hold_mode == TRACKER_HOLD_AUTOMATIC

        source.pose = _pose(x=0.5)
        recovered = _step(session, source, 0.12, valid=True)
        assert recovered.state == TeleopRunState.RECOVERING
        assert all(fault.code != "translation_jump" for fault in recovered.faults)
    finally:
        session.close()


def test_tracker_return_never_auto_recovers_an_existing_fault_state() -> None:
    session, source, _rig = _session(healthy_frames=1)
    try:
        _step(session, source, 0.0, valid=True)
        automatic = _step(session, source, 0.11, valid=False)
        assert automatic.state == TeleopRunState.FROZEN

        session.fault("independent hardware fault")
        for now in (0.12, 0.13, 0.14):
            result = _step(session, source, now, valid=True)
            assert result.state == TeleopRunState.FAULT
            assert not result.command_dispatched
    finally:
        session.close()


def test_automatic_tracker_hold_failure_becomes_fault_and_cannot_auto_recover() -> None:
    session, source, rig = _session(rig=_FailingHoldRig(), healthy_frames=1)
    try:
        _step(session, source, 0.0, valid=True)
        failed_hold = _step(session, source, 0.11, valid=False)
        assert failed_hold.state == TeleopRunState.FAULT
        assert len(rig.holds) == 1

        tracker_returned = _step(session, source, 0.12, valid=True)
        assert tracker_returned.state == TeleopRunState.FAULT
        assert not tracker_returned.command_dispatched
    finally:
        session.close()


@pytest.mark.parametrize(
    ("auto_sec", "manual_sec"),
    [(-0.1, 1.0), (float("nan"), 1.0), (0.1, 0.05)],
)
def test_tracker_hold_thresholds_are_validated(auto_sec: float, manual_sec: float) -> None:
    with pytest.raises(ValueError):
        TeleopSessionConfig(
            tracker_auto_hold_sec=auto_sec,
            tracker_manual_hold_sec=manual_sec,
        )


def test_equal_tracker_threshold_enters_manual_hold_directly_after_grace() -> None:
    session, source, rig = _session()
    session.config.tracker_manual_hold_sec = session.config.tracker_auto_hold_sec
    try:
        _step(session, source, 0.0, valid=True)

        at_threshold = _step(session, source, 0.1, valid=False)
        assert at_threshold.motion_hold_mode == TRACKER_HOLD_GRACE
        assert not at_threshold.home_recalibration_required

        manual = _step(session, source, 0.101, valid=False)
        assert manual.state == TeleopRunState.DEBUG_ONLY
        assert manual.motion_hold_mode == TRACKER_HOLD_MANUAL
        assert manual.home_recalibration_required
        assert any(fault.code == "tracker_target_manual_hold" for fault in manual.faults)
        assert len(rig.holds) == 1
    finally:
        session.close()
