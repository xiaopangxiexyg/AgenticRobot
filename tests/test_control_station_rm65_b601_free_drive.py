from __future__ import annotations

import numpy as np
import pytest

from control_station.sim_hil_profile.rm65_b601_free_drive import (
    RM65FreeDriveAdmissionError,
    RM65ToB601FreeDriveBridge,
    RM65ToB601FreeDriveSimConfig,
)
from control_station.sources.rm65_free_drive import RM65FreeDriveStep
from robot_motion.rebot_b601_rs import (
    B601IKResult,
    RM65FreeDriveSample,
    RM65ToB601RetargetResult,
)
from sim.mujoco.rebot_b601_rs import B601PositionTick


def _sample(
    sequence: int,
    *,
    receive_timestamp_monotonic_sec: float = 20.0,
) -> RM65FreeDriveSample:
    return RM65FreeDriveSample(
        sequence=sequence,
        source_timestamp_sec=10.0,
        receive_timestamp_sec=10.0,
        receive_timestamp_monotonic_sec=receive_timestamp_monotonic_sec,
        qpos_rad=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        pose_source_reference_source_controlled=tuple(
            tuple(float(item) for item in row)
            for row in np.eye(4, dtype=float)
        ),
        source_id="rm65-test",
        source_clock_domain_id="local_wall_time",
        source_reference_frame_id="right_arm_base",
        source_controlled_frame_id="tool_tcp",
    )


class _Source:
    def __init__(self, steps: list[RM65FreeDriveStep]) -> None:
        self.steps = iter(steps)

    def step(self) -> RM65FreeDriveStep:
        return next(self.steps)


class _Scene:
    def __init__(self) -> None:
        self.qpos = np.asarray((0.0, 1.0, 1.0, 0.0, 0.0, 0.0), dtype=float)

    def get_arm_qpos(self, _side: str) -> np.ndarray:
        return self.qpos.copy()


class _Driver:
    command_due = True

    def __init__(self) -> None:
        self.physics_time_sec = 0.0
        self.scene = _Scene()
        self.commands = []

    def step_physics(self, command=None) -> B601PositionTick:
        self.commands.append(command)
        self.physics_time_sec += 0.001
        return B601PositionTick(
            physics_tick=len(self.commands) - 1,
            physics_time_sec=self.physics_time_sec - 0.001,
            command_due=True,
            scheduled_command_time_sec=self.physics_time_sec - 0.001,
            command_received=command is not None,
            command_applied=command is not None,
            accepted_sequence=None if command is None else command.sequence,
            accepted_source_timestamp_sec=(
                None if command is None else command.source_timestamp_sec
            ),
            accepted_receive_timestamp_sec=(
                None if command is None else command.receive_timestamp_sec
            ),
            accepted_frame_id=None if command is None else command.frame_id,
            rejection_reason=None,
            target_qpos=(0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            measured_qpos=(0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            measured_qvel=(0.0,) * 6,
            state_valid=True,
        )


class _Retargeter:
    def __init__(self) -> None:
        self.anchor_captures = 0
        self.clears = 0
        self.seeds: list[tuple[float, ...]] = []

    def capture_anchor(self, sample, *, target_qpos_rad) -> None:
        self.anchor_captures += 1

    def clear_anchor(self) -> None:
        self.clears += 1

    def retarget(self, sample, *, seed_qpos_rad):
        self.seeds.append(tuple(float(value) for value in seed_qpos_rad))
        result = B601IKResult(
            success=True,
            status="converged",
            iterations=1,
            qpos=(0.01, 1.01, 1.01, 0.01, 0.01, 0.01),
            position_error_m=0.0,
            orientation_error_rad=0.0,
            seed_distance_rad=0.01,
            limit_projection_applied=False,
            sigma_min=1.0,
            condition_number=1.0,
        )
        return RM65ToB601RetargetResult(
            sample=sample,
            accepted=True,
            qpos_rad=result.qpos,
            target_pose=tuple(
                tuple(float(item) for item in row) for row in np.eye(4)
            ),
            ik_result=result,
            rejection_reason=None,
        )


class _RejectingRetargeter(_Retargeter):
    def retarget(self, sample, *, seed_qpos_rad):
        return RM65ToB601RetargetResult(
            sample=sample,
            accepted=False,
            qpos_rad=None,
            target_pose=None,
            ik_result=None,
            rejection_reason="ik_rejected:line_search_stalled",
        )


def _admitted(
    sequence: int,
    *,
    receive_timestamp_monotonic_sec: float = 20.0,
) -> RM65FreeDriveStep:
    return RM65FreeDriveStep(
        state={},
        sample=_sample(
            sequence,
            receive_timestamp_monotonic_sec=receive_timestamp_monotonic_sec,
        ),
        rejection_reason=None,
    )


def _rejected(reason: str) -> RM65FreeDriveStep:
    return RM65FreeDriveStep(
        state={},
        sample=None,
        rejection_reason=reason,
    )


def _bridge(
    source: _Source,
    driver: _Driver,
    retargeter: _Retargeter,
    *,
    source_poll_clock=None,
    source_poll_hz: float = 1000.0,
):
    return RM65ToB601FreeDriveBridge(
        source,
        retargeter,
        driver,
        source_poll_hz=source_poll_hz,
        source_poll_clock=source_poll_clock,
    )


def test_bridge_reuses_driver_and_queues_only_fresh_retargeted_qpos() -> None:
    source = _Source([_admitted(1), _admitted(2)])
    driver = _Driver()
    retargeter = _Retargeter()
    bridge = _bridge(source, driver, retargeter)

    bridge.activate()
    assert bridge.state == "active"
    first = bridge.step()
    second = bridge.step()

    assert first.source_step is None
    assert second.source_step is not None
    assert second.retarget_result is not None
    assert second.retarget_result.accepted
    assert second.position_tick.command_applied
    assert driver.commands[-1] is not None
    assert driver.commands[-1].sequence == 2
    assert bridge.pending_command is None


def test_source_rejection_enters_held_and_does_not_reissue_old_target() -> None:
    source = _Source([_admitted(1), _rejected("receive_stale")])
    driver = _Driver()
    retargeter = _Retargeter()
    bridge = _bridge(source, driver, retargeter)
    bridge.activate()

    bridge.step()
    rejected = bridge.step()

    assert rejected.state == "held"
    assert rejected.source_step is not None
    assert rejected.source_step.rejection_reason == "receive_stale"
    assert rejected.rejection_reason == "receive_stale"
    assert bridge.last_rejection_reason == "receive_stale"
    assert rejected.retarget_result is None
    assert bridge.pending_command is None


def test_activation_failure_is_explicit_and_recenter_requires_clutch() -> None:
    driver = _Driver()
    retargeter = _Retargeter()
    bridge = _bridge(_Source([_rejected("backend_not_observation_only")]), driver, retargeter)

    try:
        bridge.activate()
    except RM65FreeDriveAdmissionError as exc:
        assert "backend_not_observation_only" in str(exc)
    else:
        raise AssertionError("activation should fail closed")
    assert bridge.state == "held"

    bridge = _bridge(_Source([_admitted(1), _admitted(2)]), driver, retargeter)
    bridge.activate()
    bridge.engage_clutch()
    assert bridge.state == "clutched"
    bridge.release_clutch()
    assert bridge.state == "active"
    assert retargeter.anchor_captures >= 2


def test_live_source_poll_clock_is_not_slowed_by_simulation_clock() -> None:
    source = _Source([_admitted(1), _admitted(2)])
    driver = _Driver()
    retargeter = _Retargeter()
    now = [10.0]
    bridge = _bridge(
        source,
        driver,
        retargeter,
        source_poll_clock=lambda: now[0],
    )

    bridge.activate()
    first = bridge.step()
    assert first.source_step is None
    now[0] += 0.001
    second = bridge.step()
    assert second.source_step is not None
    assert source.steps is not None


def test_bridge_reports_source_to_target_dispatch_latency() -> None:
    source = _Source(
        [
            _admitted(1),
            _admitted(2, receive_timestamp_monotonic_sec=10.0005),
        ]
    )
    driver = _Driver()
    retargeter = _Retargeter()
    now = [10.0]
    bridge = _bridge(
        source,
        driver,
        retargeter,
        source_poll_clock=lambda: now[0],
    )

    bridge.activate()
    now[0] = 10.001
    tick = bridge.step()

    assert tick.position_tick.command_applied
    assert tick.source_receive_monotonic_sec == pytest.approx(10.0005)
    assert tick.source_step_started_monotonic_sec == pytest.approx(10.001)
    assert tick.source_step_completed_monotonic_sec == pytest.approx(10.001)
    assert tick.source_admission_monotonic_sec == pytest.approx(10.001)
    assert tick.target_dispatch_monotonic_sec == pytest.approx(10.001)
    assert tick.source_to_target_dispatch_latency_sec == pytest.approx(0.0005)
    assert tick.source_admission_to_target_dispatch_latency_sec == pytest.approx(0.0)


def test_bridge_retains_admission_timestamp_until_late_target_dispatch() -> None:
    class _OneTickLateDriver(_Driver):
        def __init__(self) -> None:
            super().__init__()
            self._command_due_queries = 0

        @property
        def command_due(self) -> bool:
            self._command_due_queries += 1
            return self._command_due_queries >= 2

    source = _Source([_admitted(1), _admitted(2)])
    driver = _OneTickLateDriver()
    retargeter = _Retargeter()
    now = [10.0]
    bridge = _bridge(
        source,
        driver,
        retargeter,
        source_poll_clock=lambda: now[0],
        source_poll_hz=20.0,
    )

    bridge.activate()
    now[0] = 10.05
    not_dispatched = bridge.step()
    assert not_dispatched.source_step is not None
    assert not_dispatched.position_tick.command_applied is False
    now[0] = 10.051
    dispatched = bridge.step()

    assert dispatched.source_step is None
    assert dispatched.position_tick.command_applied is True
    assert dispatched.source_admission_monotonic_sec == pytest.approx(10.05)
    assert dispatched.source_admission_to_target_dispatch_latency_sec == pytest.approx(
        0.001
    )


def test_live_source_poll_uses_fixed_start_to_start_deadline() -> None:
    now = [0.0]

    class _SlowSource(_Source):
        def __init__(self, steps):
            super().__init__(steps)
            self.calls = 0

        def step(self) -> RM65FreeDriveStep:
            self.calls += 1
            step = super().step()
            if self.calls > 1:
                now[0] += 0.006
            return step

    source = _SlowSource([_admitted(1), _admitted(2), _admitted(3)])
    driver = _Driver()
    retargeter = _Retargeter()
    bridge = RM65ToB601FreeDriveBridge(
        source,
        retargeter,
        driver,
        source_poll_hz=20.0,
        source_poll_clock=lambda: now[0],
    )

    bridge.activate()
    now[0] = 0.05
    first = bridge.step()
    assert first.source_step is not None
    assert first.source_poll_deadline_sec == pytest.approx(0.05)
    assert first.source_poll_lateness_sec == pytest.approx(0.0)
    assert first.source_poll_schedule_skipped is False

    # The source read advanced wall time to 56 ms.  A completion-relative
    # scheduler would wait until 106 ms; fixed deadline scheduling polls at
    # 100 ms instead.
    now[0] = 0.10
    second = bridge.step()
    assert second.source_step is not None
    assert second.source_poll_deadline_sec == pytest.approx(0.10)


def test_live_source_poll_skips_missed_deadline_without_catch_up_burst() -> None:
    now = [0.0]

    class _OverrunningSource(_Source):
        def __init__(self, steps):
            super().__init__(steps)
            self.calls = 0

        def step(self) -> RM65FreeDriveStep:
            self.calls += 1
            step = super().step()
            if self.calls > 1:
                now[0] += 0.060
            return step

    source = _OverrunningSource([_admitted(1), _admitted(2), _admitted(3)])
    driver = _Driver()
    retargeter = _Retargeter()
    bridge = RM65ToB601FreeDriveBridge(
        source,
        retargeter,
        driver,
        source_poll_hz=20.0,
        source_poll_clock=lambda: now[0],
    )

    bridge.activate()
    now[0] = 0.05
    first = bridge.step()
    assert first.source_step is not None
    assert first.source_poll_schedule_skipped is True

    # The overrun was 60 ms against a 50 ms period, so the next deadline is
    # 160 ms from the current clock cut, not a burst at 100 ms.
    now[0] = 0.159
    not_due = bridge.step()
    assert not_due.source_step is None
    now[0] = 0.160
    second = bridge.step()
    assert second.source_step is not None


def test_bridge_seeds_ik_from_last_accepted_target_not_measured_qpos() -> None:
    source = _Source([_admitted(1), _admitted(2), _admitted(3)])
    driver = _Driver()
    retargeter = _Retargeter()
    bridge = _bridge(source, driver, retargeter)

    bridge.activate()
    bridge.step()
    first = bridge.step()
    assert first.retarget_result is not None
    assert first.retarget_result.accepted

    # The simulated plant can lag far behind the accepted position target.
    # The next IK solve must remain continuous instead of reseeding from this
    # lagging measured state.
    driver.scene.qpos = np.asarray((1.0, 1.0, 1.0, 0.0, 0.0, 0.0), dtype=float)
    second = bridge.step()
    assert second.retarget_result is not None
    assert second.retarget_result.accepted
    assert len(retargeter.seeds) == 2
    np.testing.assert_allclose(
        retargeter.seeds[1],
        (0.01, 1.01, 1.01, 0.01, 0.01, 0.01),
    )


def test_retarget_rejection_is_exposed_and_latches_held_reason() -> None:
    source = _Source([_admitted(1), _admitted(2)])
    driver = _Driver()
    bridge = _bridge(source, driver, _RejectingRetargeter())

    bridge.activate()
    bridge.step()
    rejected = bridge.step()
    assert rejected.state == "held"
    assert rejected.rejection_reason == "ik_rejected:line_search_stalled"
    assert bridge.last_rejection_reason == "ik_rejected:line_search_stalled"


def test_sim_config_keeps_reverse_gain_explicit() -> None:
    config = RM65ToB601FreeDriveSimConfig(
        translation_gain_target_reference_xyz=(1.5, 1.0, 0.75),
        rotation_gain=0.8,
        source_poll_hz=20.0,
    )
    assert config.translation_gain_target_reference_xyz == (1.5, 1.0, 0.75)
    assert config.rotation_gain == 0.8
    assert config.target_velocity_limit_rad_s == 5.0
