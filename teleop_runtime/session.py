from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .calibration import TeleopCalibration
from .diagnostics import FaultEvent, FaultSeverity, RateTracker, TeleopRunState, TeleopStateMachine
from .interfaces import ObservationSource, Retargeter, Rig
from .recording import RecordingConfig, SessionRecorder
from .safety import SafetyMonitor, SafetyViolation, TeleopSafetyConfig
from .schemas import (
    ObservationChannel,
    TeleopCommand,
    TeleopObservation,
    latest_channel_timestamp,
    merge_teleop_commands,
)


@dataclass
class TeleopSessionConfig:
    mode: str = "bimanual"
    source_profile: str = ""
    embodiment_profile: str = ""
    teleop_profile: str = ""
    tcp_mode: str = "hand_base"
    scale: float = 1.0
    stale_timeout_sec: float = 0.5
    record_enabled: bool = True
    allow_hardware_after_debug_pass: bool = True


@dataclass
class SessionStepResult:
    timestamp: float
    state: TeleopRunState
    control_hz: float
    channels: dict[str, ObservationChannel] = field(default_factory=dict)
    observation: TeleopObservation | None = None
    command: TeleopCommand | None = None
    faults: list[FaultEvent] = field(default_factory=list)


class TeleopSession:
    def __init__(
        self,
        config: TeleopSessionConfig,
        *,
        sources: Sequence[ObservationSource] | None = None,
        retargeters: Sequence[Retargeter] | None = None,
        rig: Rig | None = None,
        calibration: TeleopCalibration | None = None,
        safety_config: TeleopSafetyConfig | None = None,
        recorder: SessionRecorder | None = None,
        state_machine: TeleopStateMachine | None = None,
    ):
        self.config = config
        self.sources = list(sources or [])
        self.retargeters = list(retargeters or [])
        self.rig = rig
        self.calibration = calibration
        self.safety_monitor = SafetyMonitor(safety_config)
        self.recorder = recorder
        if self.recorder is None and config.record_enabled:
            self.recorder = SessionRecorder(config=RecordingConfig(enabled=True))
        self.state_machine = state_machine if state_machine is not None else TeleopStateMachine()
        self.rate_tracker = RateTracker()
        self.debug_passed = False
        self._started = False

    def _transition(self, new_state: TeleopRunState, reason: str = "") -> None:
        old_state = self.state_machine.state
        self.state_machine.transition(new_state, reason)
        if self.recorder is not None:
            self.recorder.record_state_transition(old_state.value, new_state.value, reason)

    def start(self) -> None:
        if self._started:
            return
        for source in self.sources:
            source.start()
        if self.rig is not None:
            self.rig.start()
        self._transition(TeleopRunState.CALIBRATION, "session started")
        if self.calibration is not None and self.recorder is not None:
            self.recorder.record_calibration(self.calibration.to_dict())
        self._started = True

    def set_calibration_ready(self) -> None:
        self._transition(TeleopRunState.DEBUG_ONLY, "calibration ready")

    def mark_debug_passed(self) -> None:
        self.debug_passed = True
        if self.state_machine.state == TeleopRunState.CALIBRATION:
            self._transition(TeleopRunState.DEBUG_ONLY, "debug passed")

    def arm_hardware(self) -> None:
        if self.config.allow_hardware_after_debug_pass and not self.debug_passed:
            raise RuntimeError("Cannot arm hardware before debug pass is marked complete.")
        if self.rig is not None and not self.rig.is_ready():
            raise RuntimeError("Rig is not ready.")
        self._transition(TeleopRunState.ARMED, "hardware armed")

    def freeze(self, reason: str) -> None:
        self._transition(TeleopRunState.FROZEN, reason)

    def fault(self, reason: str) -> None:
        self._transition(TeleopRunState.FAULT, reason)

    def _fault_from_violation(self, violation: SafetyViolation, timestamp: float) -> FaultEvent:
        return FaultEvent(
            code=violation.code,
            message=violation.message,
            severity=FaultSeverity.ERROR,
            timestamp=timestamp,
            channel_name=violation.channel_name,
            metadata=dict(violation.metadata),
        )

    def poll_sources(self, now: float) -> tuple[dict[str, ObservationChannel], list[FaultEvent]]:
        merged: dict[str, ObservationChannel] = {}
        faults: list[FaultEvent] = []

        for source in self.sources:
            source_channels = source.get_latest_channels()
            expected = set(source.expected_channel_names())
            returned = set(source_channels)
            missing = expected - returned
            for channel_name in missing:
                faults.append(
                    FaultEvent(
                        code="missing_channel",
                        message=f"Source '{source.name}' did not provide expected channel '{channel_name}'.",
                        severity=FaultSeverity.ERROR,
                        timestamp=now,
                        component=source.name,
                        channel_name=channel_name,
                    )
                )
            merged.update(source_channels)
        return merged, faults

    def _build_observation(
        self, channels: dict[str, ObservationChannel], now: float
    ) -> TeleopObservation:
        timestamp = latest_channel_timestamp(channels)
        if timestamp is None:
            timestamp = now
        return TeleopObservation(
            channels={name: channel.copy() for name, channel in channels.items()},
            timestamp=timestamp,
            metadata={"tcp_mode": self.config.tcp_mode},
        )

    def _build_command(self, observation: TeleopObservation) -> TeleopCommand:
        commands: list[TeleopCommand] = []
        for retargeter in self.retargeters:
            commands.append(retargeter.retarget(observation, self.calibration))
        command = merge_teleop_commands(commands)
        if command.timestamp is None:
            command.timestamp = observation.timestamp
        command.metadata.setdefault("tcp_mode", self.config.tcp_mode)
        return command

    def run_step(self, now: float | None = None) -> SessionStepResult:
        if not self._started:
            self.start()
        current_time = time.time() if now is None else float(now)
        control_hz = self.rate_tracker.tick(current_time)

        channels, faults = self.poll_sources(current_time)
        if self.recorder is not None:
            self.recorder.record_channels(channels, timestamp=current_time)

        for violation in self.safety_monitor.evaluate_channels(channels, current_time):
            faults.append(self._fault_from_violation(violation, current_time))

        if (
            self.state_machine.state == TeleopRunState.RUNNING
            and self.rate_tracker.num_samples >= 5
        ):
            for violation in self.safety_monitor.evaluate_control_rate(control_hz):
                faults.append(self._fault_from_violation(violation, current_time))

        for fault in faults:
            if self.recorder is not None:
                self.recorder.record_fault(fault)

        if faults and self.safety_monitor.config.freeze_on_fault:
            self.freeze(faults[0].code)
            return SessionStepResult(
                timestamp=current_time,
                state=self.state_machine.state,
                control_hz=control_hz,
                channels=channels,
                faults=faults,
            )

        observation = self._build_observation(channels, current_time)
        command = self._build_command(observation)

        if self.recorder is not None:
            self.recorder.record_command(command)

        if self.rig is not None and self.state_machine.state in {
            TeleopRunState.ARMED,
            TeleopRunState.RUNNING,
        }:
            self.rig.send_commands(command)
            if self.state_machine.state != TeleopRunState.RUNNING:
                self._transition(TeleopRunState.RUNNING, "first command dispatched")

        return SessionStepResult(
            timestamp=current_time,
            state=self.state_machine.state,
            control_hz=control_hz,
            channels=channels,
            observation=observation,
            command=command,
            faults=faults,
        )

    def close(self) -> None:
        for source in self.sources:
            source.close()
        if self.rig is not None:
            self.rig.close()
        if self.recorder is not None:
            self.recorder.close()
        self._started = False

    def __enter__(self) -> "TeleopSession":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
