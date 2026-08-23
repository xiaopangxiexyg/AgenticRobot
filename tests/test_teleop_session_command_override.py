from __future__ import annotations

import numpy as np

from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.schemas import EffectorCommand, TeleopCommand
from teleop_runtime.session import (
    TeleopCommandOverride,
    TeleopSession,
    TeleopSessionConfig,
)


def _command(*, source: str, timestamp: float | None = None) -> TeleopCommand:
    command = TeleopCommand(timestamp=timestamp, metadata={"source": source})
    command.add_command(
        EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="joint_vector",
            value=np.zeros(6, dtype=float),
        )
    )
    return command


class _Rig:
    name = "override_test_rig"

    def __init__(self) -> None:
        self.sent: list[TeleopCommand] = []
        self.holds: list[str] = []

    def start(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    def send_commands(self, command, phase_recorder=None):
        del phase_recorder
        self.sent.append(command)
        return {"right_arm": "accepted"}

    def hold_motion(self, reason: str = "manual hold") -> None:
        self.holds.append(reason)

    def close(self) -> None:
        return None


class _HumanRetargeter:
    name = "human"

    def __init__(self) -> None:
        self.calls = 0

    def retarget(self, observation, calibration):
        del calibration
        self.calls += 1
        return _command(source="human", timestamp=observation.timestamp)


def _arm(session: TeleopSession) -> None:
    session.start()
    session.set_calibration_ready()
    session.mark_debug_passed()
    session.arm_hardware()


def test_inactive_override_preserves_human_command_path() -> None:
    rig = _Rig()
    retargeter = _HumanRetargeter()
    provider_calls: list[float] = []

    def provider(observation, now):
        del observation
        provider_calls.append(now)

    session = TeleopSession(
        config=TeleopSessionConfig(
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        retargeters=[retargeter],
        rig=rig,
        command_override_provider=provider,
    )
    try:
        _arm(session)
        step = session.run_step(now=1.0)
    finally:
        session.close()

    assert provider_calls == [1.0]
    assert retargeter.calls == 1
    assert step.command is not None
    assert step.command.metadata["source"] == "human"
    assert step.command_dispatched is True
    assert len(rig.sent) == 1
    assert rig.holds == []


def test_override_uses_existing_filter_guard_admission_and_dispatch_path() -> None:
    rig = _Rig()
    retargeter = _HumanRetargeter()
    phases: list[str] = []

    def command_filter(command, observation, now):
        del observation, now
        phases.append("filter")
        command.metadata["filtered"] = True
        return command

    def command_guard(command, observation, now):
        del observation, now
        phases.append("guard")
        assert command.metadata["filtered"] is True

    def admission(command, observation, now):
        del observation, now
        phases.append("admission")
        assert command.metadata["source"] == "agent"
        return True

    session = TeleopSession(
        config=TeleopSessionConfig(
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        retargeters=[retargeter],
        rig=rig,
        command_override_provider=(
            lambda observation, now: _command(source="agent", timestamp=None)
        ),
        command_filter=command_filter,
        command_guard=command_guard,
        command_dispatch_admission=admission,
    )
    try:
        _arm(session)
        step = session.run_step(now=2.0)
    finally:
        session.close()

    assert retargeter.calls == 0
    assert phases == ["filter", "guard", "admission"]
    assert step.command is not None
    assert step.command.timestamp == 2.0
    assert step.command.metadata["tcp_mode"] == "tool_tcp"
    assert step.command_dispatched is True
    assert len(rig.sent) == 1
    assert rig.holds == []


def test_claimed_commandless_override_inhibits_human_fallback() -> None:
    rig = _Rig()
    retargeter = _HumanRetargeter()
    session = TeleopSession(
        config=TeleopSessionConfig(
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        retargeters=[retargeter],
        rig=rig,
        command_override_provider=(
            lambda observation, now: TeleopCommandOverride(command=None)
        ),
    )
    try:
        _arm(session)
        step = session.run_step(now=2.5)
    finally:
        session.close()

    assert retargeter.calls == 0
    assert step.state == TeleopRunState.ARMED
    assert step.command is None
    assert step.command_dispatched is False
    assert rig.sent == []
    assert rig.holds == []


def test_override_provider_failure_holds_and_faults_without_human_fallback() -> None:
    rig = _Rig()
    retargeter = _HumanRetargeter()

    def provider(observation, now):
        del observation, now
        raise RuntimeError("operation target unavailable")

    session = TeleopSession(
        config=TeleopSessionConfig(
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        retargeters=[retargeter],
        rig=rig,
        command_override_provider=provider,
    )
    try:
        _arm(session)
        step = session.run_step(now=3.0)
    finally:
        session.close()

    assert retargeter.calls == 0
    assert step.state == TeleopRunState.FAULT
    assert step.command is None
    assert step.command_dispatched is False
    assert [fault.code for fault in step.faults] == [
        "command_override_provider_error"
    ]
    assert rig.sent == []
    assert len(rig.holds) == 1
    assert "operation target unavailable" in rig.holds[0]


def test_post_human_selection_uses_common_filter_guard_and_dispatch_path() -> None:
    rig = _Rig()
    retargeter = _HumanRetargeter()
    phases: list[str] = []

    def selector(observation, human_command, now):
        del observation, now
        assert human_command.metadata["source"] == "human"
        return TeleopCommandOverride(_command(source="policy", timestamp=None))

    def command_filter(command, observation, now):
        del observation, now
        phases.append("filter")
        command.metadata["filtered"] = True
        return command

    def command_guard(command, observation, now):
        del observation, now
        phases.append("guard")
        assert command.metadata["source"] == "policy"
        assert command.metadata["filtered"] is True

    session = TeleopSession(
        config=TeleopSessionConfig(
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        retargeters=[retargeter],
        rig=rig,
        human_command_selection_provider=selector,
        command_filter=command_filter,
        command_guard=command_guard,
    )
    try:
        _arm(session)
        step = session.run_step(now=4.0)
    finally:
        session.close()

    assert retargeter.calls == 1
    assert phases == ["filter", "guard"]
    assert step.command is not None
    assert step.command.metadata["source"] == "policy"
    assert step.command.timestamp == 4.0
    assert step.command_dispatched is True
    assert len(rig.sent) == 1


def test_post_human_selection_commandless_claim_never_falls_back() -> None:
    rig = _Rig()
    retargeter = _HumanRetargeter()
    session = TeleopSession(
        config=TeleopSessionConfig(
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        retargeters=[retargeter],
        rig=rig,
        human_command_selection_provider=(
            lambda observation, human_command, now: TeleopCommandOverride(None)
        ),
    )
    try:
        _arm(session)
        step = session.run_step(now=4.5)
    finally:
        session.close()

    assert retargeter.calls == 1
    assert step.command is None
    assert step.command_dispatched is False
    assert rig.sent == []


def test_post_human_selection_failure_holds_without_dispatching_human() -> None:
    rig = _Rig()
    retargeter = _HumanRetargeter()

    def selector(observation, human_command, now):
        del observation, human_command, now
        raise RuntimeError("policy selector unavailable")

    session = TeleopSession(
        config=TeleopSessionConfig(
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        retargeters=[retargeter],
        rig=rig,
        human_command_selection_provider=selector,
    )
    try:
        _arm(session)
        step = session.run_step(now=5.0)
    finally:
        session.close()

    assert retargeter.calls == 1
    assert step.state == TeleopRunState.FAULT
    assert step.command is None
    assert step.command_dispatched is False
    assert [fault.code for fault in step.faults] == [
        "human_command_selection_provider_error"
    ]
    assert rig.sent == []
    assert len(rig.holds) == 1
    assert "policy selector unavailable" in rig.holds[0]
