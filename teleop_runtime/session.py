from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .action.schema import ActionRecord
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
from .tcp import canonical_tcp_frame


@dataclass
class TeleopSessionConfig:
    mode: str = "bimanual"
    source_profile: str = ""
    embodiment_profile: str = ""
    teleop_profile: str = ""
    tcp_mode: str = "tool_tcp"
    scale: float = 1.0
    stale_timeout_sec: float = 0.5
    record_enabled: bool = True
    allow_hardware_after_debug_pass: bool = True
    auto_recovery_enabled: bool = True
    recovery_healthy_frame_count: int = 10
    hold_motion_on_freeze: bool = False
    non_freezing_fault_codes: tuple[str, ...] = ("low_control_rate",)
    auto_recoverable_fault_codes: tuple[str, ...] = (
        "missing_channel",
        "invalid_channel",
        "invalid_pose",
        "stale_channel",
        "translation_jump",
        "rotation_jump",
        "low_control_rate",
    )

    def __post_init__(self) -> None:
        self.tcp_mode = canonical_tcp_frame(self.tcp_mode)


@dataclass
class SessionStepResult:
    timestamp: float
    state: TeleopRunState
    control_hz: float
    channels: dict[str, ObservationChannel] = field(default_factory=dict)
    raw_channels: dict[str, ObservationChannel] = field(default_factory=dict)
    observation: TeleopObservation | None = None
    command: TeleopCommand | None = None
    command_dispatched: bool = False
    action_record: ActionRecord | None = None
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
        channel_filter: Callable[
            [dict[str, ObservationChannel], float], dict[str, ObservationChannel]
        ]
        | None = None,
        command_filter: Callable[[TeleopCommand, TeleopObservation, float], TeleopCommand] | None = None,
        fault_filter: Callable[
            [Sequence[FaultEvent], dict[str, ObservationChannel], float], Sequence[FaultEvent]
        ]
        | None = None,
    ):
        self.config = config
        self.sources = list(sources or [])
        self.retargeters = list(retargeters or [])
        self.rig = rig
        self.calibration = calibration
        self.safety_monitor = SafetyMonitor(safety_config)
        self.recorder = recorder
        self.channel_filter = channel_filter
        self.command_filter = command_filter
        self.fault_filter = fault_filter
        if self.recorder is None and config.record_enabled:
            self.recorder = SessionRecorder(config=RecordingConfig(enabled=True))
        self.state_machine = state_machine if state_machine is not None else TeleopStateMachine()
        self.rate_tracker = RateTracker()
        self.debug_passed = False
        self._started = False
        self._freeze_fault_codes: tuple[str, ...] = ()
        self._freeze_channel_names: tuple[str, ...] = ()
        self._healthy_frames_since_fault_clear = 0

    def _transition(self, new_state: TeleopRunState, reason: str = "") -> None:
        old_state = self.state_machine.state
        if old_state == new_state:
            return
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

    def _hold_rig_motion(self, reason: str, *, context: str) -> None:
        if self.rig is not None:
            try:
                self.rig.hold_motion(reason)
            except Exception as exc:  # noqa: BLE001
                print(
                    "\033[1;31;40m"
                    + f"[Debug Info] {context} hold_motion failed: {type(exc).__name__}: {exc}"
                    + "\033[0m"
                )

    def hold_hardware(self, reason: str = "manual hold") -> None:
        self._clear_recovery_context()
        self.safety_monitor.request_pose_baseline_reset_many(self.safety_monitor.config.required_channels)
        self._reset_retargeter_tracking()
        self._hold_rig_motion(reason, context="manual hold")
        self._transition(TeleopRunState.DEBUG_ONLY, reason)

    def freeze(self, reason: str) -> None:
        already_frozen = self.state_machine.state == TeleopRunState.FROZEN
        print("\033[1;31;40m" + f"[Debug Info] FROZEN triggered due to: {reason}" + "\033[0m")
        self._transition(TeleopRunState.FROZEN, reason)
        if self.config.hold_motion_on_freeze and self.rig is not None and not already_frozen:
            self._hold_rig_motion(f"safety freeze: {reason}", context="safety freeze")

    def fault(self, reason: str) -> None:
        print("\033[1;31;40m" + f"[Debug Info] FAULT triggered due to: {reason}" + "\033[0m")
        self._transition(TeleopRunState.FAULT, reason)

    def build_command(self, observation: TeleopObservation) -> TeleopCommand:
        return self._build_command(observation)

    def finish_recovery(self, reason: str = "recovery complete") -> None:
        if self.state_machine.state not in {
            TeleopRunState.FROZEN,
            TeleopRunState.RECOVERING,
        }:
            return
        self._clear_recovery_context()
        self._transition(TeleopRunState.ARMED, reason)

    def _clear_recovery_context(self) -> None:
        self._freeze_fault_codes = ()
        self._freeze_channel_names = ()
        self._healthy_frames_since_fault_clear = 0

    def _record_freeze_context(self, faults: Sequence[FaultEvent]) -> None:
        self._freeze_fault_codes = tuple(sorted({fault.code for fault in faults}))
        self._freeze_channel_names = tuple(
            sorted({fault.channel_name for fault in faults if fault.channel_name})
        )
        self._healthy_frames_since_fault_clear = 0
        self.safety_monitor.request_pose_baseline_reset_many(self._freeze_channel_names)
        self._reset_retargeter_tracking()

    def _reset_retargeter_tracking(self) -> None:
        for retargeter in self.retargeters:
            reset = getattr(retargeter, "reset_target_pose_smoothing", None)
            if callable(reset):
                reset()

    def _freeze_is_auto_recoverable(self) -> bool:
        if not self.config.auto_recovery_enabled:
            return False
        if not self._freeze_fault_codes:
            return False
        recoverable = set(self.config.auto_recoverable_fault_codes)
        return all(code in recoverable for code in self._freeze_fault_codes)

    def _fault_from_violation(self, violation: SafetyViolation, timestamp: float) -> FaultEvent:
        return FaultEvent(
            code=violation.code,
            message=violation.message,
            severity=FaultSeverity.ERROR,
            timestamp=timestamp,
            channel_name=violation.channel_name,
            metadata=dict(violation.metadata),
        )

    def _fault_should_freeze(self, fault: FaultEvent) -> bool:
        return fault.code not in set(self.config.non_freezing_fault_codes)

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

    def _build_command(
        self,
        observation: TeleopObservation,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> TeleopCommand:
        commands: list[TeleopCommand] = []
        for index, retargeter in enumerate(self.retargeters):
            started_at = time.perf_counter()
            commands.append(retargeter.retarget(observation, self.calibration))
            if phase_recorder is not None:
                retargeter_name = str(getattr(retargeter, "name", "") or type(retargeter).__name__)
                phase_recorder(
                    f"session.retargeter.{index}.{retargeter_name}",
                    time.perf_counter() - started_at,
                )
        command = merge_teleop_commands(commands)
        if command.timestamp is None:
            command.timestamp = observation.timestamp
        command.metadata.setdefault("tcp_mode", self.config.tcp_mode)
        return command

    def run_step(
        self,
        now: float | None = None,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> SessionStepResult:
        phase_started = time.perf_counter()

        def mark_phase(name: str) -> None:
            nonlocal phase_started
            if phase_recorder is None:
                return
            current = time.perf_counter()
            phase_recorder(name, current - phase_started)
            phase_started = current

        if not self._started:
            self.start()
            mark_phase("session.start")
        current_time = time.time() if now is None else float(now)
        control_hz = self.rate_tracker.tick(current_time)
        mark_phase("session.rate_tracker")

        channels, faults = self.poll_sources(current_time)
        raw_channels = {name: channel.copy() for name, channel in channels.items()}
        mark_phase("session.poll_sources")
        if self.channel_filter is not None:
            channels = dict(self.channel_filter(channels, current_time))
            mark_phase("session.filter_channels")
        if self.recorder is not None:
            self.recorder.record_channels(channels, timestamp=current_time)
            mark_phase("session.record_channels")

        for violation in self.safety_monitor.evaluate_channels(channels, current_time):
            faults.append(self._fault_from_violation(violation, current_time))
        mark_phase("session.evaluate_channels")

        if self.fault_filter is not None and faults:
            faults = list(self.fault_filter(faults, channels, current_time))
            mark_phase("session.filter_faults")

        if (
            self.state_machine.state == TeleopRunState.RUNNING
            and self.rate_tracker.num_samples >= 5
        ):
            for violation in self.safety_monitor.evaluate_control_rate(control_hz):
                faults.append(self._fault_from_violation(violation, current_time))
            mark_phase("session.evaluate_control_rate")

        for fault in faults:
            if self.recorder is not None:
                self.recorder.record_fault(fault)
        if faults and self.recorder is not None:
            mark_phase("session.record_faults")

        freeze_faults = [fault for fault in faults if self._fault_should_freeze(fault)]
        mark_phase("session.freeze_fault_check")

        if freeze_faults and self.safety_monitor.config.freeze_on_fault:
            self._record_freeze_context(freeze_faults)
            self.freeze(freeze_faults[0].code)
            mark_phase("session.freeze_on_fault")
            return SessionStepResult(
                timestamp=current_time,
                state=self.state_machine.state,
                control_hz=control_hz,
                channels=channels,
                raw_channels=raw_channels,
                faults=faults,
            )

        if self.state_machine.state in {
            TeleopRunState.FAULT,
        }:
            return SessionStepResult(
                timestamp=current_time,
                state=self.state_machine.state,
                control_hz=control_hz,
                channels=channels,
                raw_channels=raw_channels,
                faults=faults,
            )

        if self.state_machine.state == TeleopRunState.FROZEN:
            if not self._freeze_is_auto_recoverable():
                return SessionStepResult(
                    timestamp=current_time,
                    state=self.state_machine.state,
                    control_hz=control_hz,
                    channels=channels,
                    raw_channels=raw_channels,
                    faults=faults,
                )
            self._healthy_frames_since_fault_clear += 1
            if self._healthy_frames_since_fault_clear < max(
                1, int(self.config.recovery_healthy_frame_count)
            ):
                return SessionStepResult(
                    timestamp=current_time,
                    state=self.state_machine.state,
                    control_hz=control_hz,
                    channels=channels,
                    raw_channels=raw_channels,
                    faults=faults,
                )
            observation = self._build_observation(channels, current_time)
            mark_phase("session.build_observation")
            self._transition(TeleopRunState.RECOVERING, "auto recovery ready")
            mark_phase("session.transition_recovering")
            print("\033[1;32;40m" + f"[Debug Info] Auto-recovery triggered" + "\033[0m")
            return SessionStepResult(
                timestamp=current_time,
                state=self.state_machine.state,
                control_hz=control_hz,
                channels=channels,
                raw_channels=raw_channels,
                observation=observation,
                faults=faults,
            )

        if self.state_machine.state == TeleopRunState.RECOVERING:
            observation = self._build_observation(channels, current_time)
            mark_phase("session.build_observation")
            return SessionStepResult(
                timestamp=current_time,
                state=self.state_machine.state,
                control_hz=control_hz,
                channels=channels,
                raw_channels=raw_channels,
                observation=observation,
                faults=faults,
            )

        observation = self._build_observation(channels, current_time)
        mark_phase("session.build_observation")
        command = self._build_command(observation, phase_recorder=phase_recorder)
        mark_phase("session.build_command")
        if self.command_filter is not None:
            command = self.command_filter(command, observation, current_time)
            mark_phase("session.filter_command")

        if self.recorder is not None:
            self.recorder.record_command(command)
            mark_phase("session.record_command")

        command_dispatched = False
        if self.rig is not None and self.state_machine.state in {
            TeleopRunState.ARMED,
            TeleopRunState.RUNNING,
        }:
            self.rig.send_commands(command, phase_recorder=phase_recorder)
            command_dispatched = True
            mark_phase("session.rig.send_commands")
            if self.state_machine.state != TeleopRunState.RUNNING:
                self._transition(TeleopRunState.RUNNING, "first command dispatched")
                mark_phase("session.transition_running")

        return SessionStepResult(
            timestamp=current_time,
            state=self.state_machine.state,
            control_hz=control_hz,
            channels=channels,
            raw_channels=raw_channels,
            observation=observation,
            command=command,
            command_dispatched=command_dispatched,
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
