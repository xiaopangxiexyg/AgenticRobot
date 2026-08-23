from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .action.schema import ActionRecord
from .action.human_proposal import HumanCommandEmission, HumanLegacyCommandObservation
from .calibration import TeleopCalibration
from .command_override import TeleopCommandOverride
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
from .transforms import validate_pose_matrix
from robot_motion.runtime.lifecycle import (
    LifecycleStartupError,
    collect_lifecycle_failures,
    run_best_effort_cleanup,
)
from robot_motion.runtime.observation_cut import (
    ObservationSampleRef,
    ObservationSampleTracker,
)


TRACKER_HOLD_NONE = ""
TRACKER_HOLD_GRACE = "tracker_loss_grace"
TRACKER_HOLD_AUTOMATIC = "automatic_tracker_hold"
TRACKER_HOLD_RECOVERING = "automatic_tracker_recovery"
TRACKER_HOLD_MANUAL = "manual_home_required"

_TRACKER_AVAILABILITY_FAULT_CODES = frozenset(
    {"missing_channel", "invalid_channel", "invalid_pose", "stale_channel"}
)


@dataclass
class TeleopSessionConfig:
    mode: str = "bimanual"
    source_profile: str = ""
    embodiment_profile: str = ""
    teleop_profile: str = ""
    tcp_mode: str = "tool_tcp"
    scale: float = 1.0
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
        "tracker_target_lost",
    )
    tracker_loss_hold_enabled: bool = False
    tracker_channel_names: tuple[str, ...] = ()
    tracker_auto_hold_sec: float = 0.10
    tracker_manual_hold_sec: float = 1.0

    def __post_init__(self) -> None:
        self.tcp_mode = canonical_tcp_frame(self.tcp_mode)
        self.tracker_channel_names = tuple(
            dict.fromkeys(
                str(name).strip() for name in self.tracker_channel_names if str(name).strip()
            )
        )
        auto_hold_sec = float(self.tracker_auto_hold_sec)
        manual_hold_sec = float(self.tracker_manual_hold_sec)
        if not math.isfinite(auto_hold_sec) or auto_hold_sec < 0.0:
            raise ValueError("tracker_auto_hold_sec must be finite and >= 0.")
        if not math.isfinite(manual_hold_sec) or manual_hold_sec < auto_hold_sec:
            raise ValueError(
                "tracker_manual_hold_sec must be finite and >= tracker_auto_hold_sec."
            )
        self.tracker_auto_hold_sec = auto_hold_sec
        self.tracker_manual_hold_sec = manual_hold_sec


@dataclass
class SessionStepResult:
    timestamp: float
    state: TeleopRunState
    control_hz: float
    channels: dict[str, ObservationChannel] = field(default_factory=dict)
    raw_channels: dict[str, ObservationChannel] = field(default_factory=dict)
    input_sample_refs: dict[str, ObservationSampleRef] = field(
        default_factory=dict
    )
    observation: TeleopObservation | None = None
    command: TeleopCommand | None = None
    command_dispatched: bool = False
    action_record: ActionRecord | None = None
    faults: list[FaultEvent] = field(default_factory=list)
    motion_hold_mode: str = TRACKER_HOLD_NONE
    motion_hold_reason: str = ""
    tracker_loss_duration_sec: float = 0.0
    tracker_loss_channels: tuple[str, ...] = ()
    home_recalibration_required: bool = False


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
        command_guard: Callable[[TeleopCommand, TeleopObservation, float], FaultEvent | None]
        | None = None,
        command_dispatch_admission: Callable[
            [TeleopCommand, TeleopObservation, float], bool
        ]
        | None = None,
        command_override_provider: Callable[
            [TeleopObservation, float],
            TeleopCommandOverride | TeleopCommand | None,
        ]
        | None = None,
        human_command_selection_provider: Callable[
            [TeleopObservation, TeleopCommand, float],
            TeleopCommandOverride | None,
        ]
        | None = None,
        fault_filter: Callable[
            [Sequence[FaultEvent], dict[str, ObservationChannel], float], Sequence[FaultEvent]
        ]
        | None = None,
        human_command_observer: Callable[[HumanCommandEmission], None] | None = None,
        human_command_filter_observer: Callable[
            [HumanCommandEmission, TeleopCommand, float], None
        ]
        | None = None,
        human_execution_observer: Callable[[HumanLegacyCommandObservation], None]
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
        self.command_guard = command_guard
        self.command_dispatch_admission = command_dispatch_admission
        self.command_override_provider = command_override_provider
        self.human_command_selection_provider = human_command_selection_provider
        self.fault_filter = fault_filter
        self.human_command_observer = human_command_observer
        self.human_command_filter_observer = human_command_filter_observer
        self.human_execution_observer = human_execution_observer
        if self.recorder is None and config.record_enabled:
            self.recorder = SessionRecorder(config=RecordingConfig(enabled=True))
        self.state_machine = state_machine if state_machine is not None else TeleopStateMachine()
        self.rate_tracker = RateTracker()
        self.debug_passed = False
        self._started = False
        self._freeze_fault_codes: tuple[str, ...] = ()
        self._freeze_channel_names: tuple[str, ...] = ()
        self._healthy_frames_since_fault_clear = 0
        self.last_fault_reason = ""
        self._motion_hold_mode = TRACKER_HOLD_NONE
        self._motion_hold_reason = ""
        self._tracker_loss_duration_sec = 0.0
        self._tracker_loss_channels: tuple[str, ...] = ()
        self._tracker_unavailable_since: dict[str, float] = {}
        self._tracker_last_healthy_at: dict[str, float] = {}
        self._home_recalibration_required = False
        self._last_home_recalibration_reason = ""
        self._last_safe_command: TeleopCommand | None = None
        self._human_command_emission_sequence = 0
        self.human_command_observer_error_count = 0
        self.last_human_command_observer_error: Exception | None = None
        self.human_command_filter_observer_error_count = 0
        self.last_human_command_filter_observer_error: Exception | None = None
        self.human_execution_observer_error_count = 0
        self.last_human_execution_observer_error: Exception | None = None
        self._current_human_command_emission: HumanCommandEmission | None = None
        self._input_sample_tracker = ObservationSampleTracker()
        self._current_input_sample_refs: dict[str, ObservationSampleRef] = {}
        self._source_mapping_generation_by_channel: dict[str, int] = {}

    @property
    def motion_hold_mode(self) -> str:
        return self._motion_hold_mode

    @property
    def motion_hold_reason(self) -> str:
        return self._motion_hold_reason

    @property
    def tracker_loss_duration_sec(self) -> float:
        return float(self._tracker_loss_duration_sec)

    @property
    def tracker_loss_channels(self) -> tuple[str, ...]:
        return self._tracker_loss_channels

    @property
    def home_recalibration_required(self) -> bool:
        return bool(self._home_recalibration_required)

    def evaluate_command_guard(
        self,
        command: TeleopCommand,
        observation: TeleopObservation,
        now: float,
    ) -> FaultEvent | None:
        """Evaluate the optional pre-dispatch guard without dispatching a command.

        Recovery paths that intentionally dispatch outside :meth:`run_step`
        must call this method too.  Guard failures are represented as faults so
        callers can use the common fail-closed handler below.
        """

        if self.command_guard is None:
            return None
        try:
            return self.command_guard(command, observation, float(now))
        except Exception as exc:  # noqa: BLE001 - a pre-dispatch guard must fail closed.
            return FaultEvent(
                code="command_guard_exception",
                message=f"pre-dispatch command guard raised {type(exc).__name__}: {exc}",
                severity=FaultSeverity.ERROR,
                timestamp=float(now),
                component="teleop_session",
            )

    def reject_command_for_guard_fault(self, fault: FaultEvent) -> None:
        """Hold and fault after a failed pre-dispatch command guard."""

        if self.recorder is not None:
            self.recorder.record_fault(fault)
        reason = (
            "pre-dispatch command safety guard failed: "
            f"{fault.code}: {fault.message}"
        )
        held = self._hold_rig_motion(reason, context="command safety guard")
        self.fault(reason if held else f"{reason}; hold_motion failed")

    def _reject_command_override_provider_error(
        self,
        error: BaseException,
        *,
        timestamp: float,
    ) -> FaultEvent:
        """Fail closed when an optional external command source cannot answer.

        The provider is evaluated by the control-loop owner, never by an
        external worker.  A provider failure therefore uses the same owned rig
        HOLD seam as other pre-dispatch failures and cannot fall back to a
        human command in the failing frame.
        """

        fault = FaultEvent(
            code="command_override_provider_error",
            message=(
                "command override provider raised "
                f"{type(error).__name__}: {error}"
            ),
            severity=FaultSeverity.ERROR,
            timestamp=float(timestamp),
            component="teleop_session",
        )
        if self.recorder is not None:
            self.recorder.record_fault(fault)
        reason = f"pre-dispatch command override failed: {fault.message}"
        held = self._hold_rig_motion(reason, context="command override provider")
        self.fault(reason if held else f"{reason}; hold_motion failed")
        return fault

    def _reject_human_command_selection_provider_error(
        self,
        error: BaseException,
        *,
        timestamp: float,
    ) -> FaultEvent:
        """Fail closed when exclusive source selection cannot answer.

        Selection runs after the Human candidate has been built, but before
        command filtering, canonical safety, recording, or dispatch.  A
        selector failure therefore cannot silently fall back to Human in the
        failing frame.
        """

        fault = FaultEvent(
            code="human_command_selection_provider_error",
            message=(
                "human command selection provider raised "
                f"{type(error).__name__}: {error}"
            ),
            severity=FaultSeverity.ERROR,
            timestamp=float(timestamp),
            component="teleop_session",
        )
        if self.recorder is not None:
            self.recorder.record_fault(fault)
        reason = f"pre-dispatch exclusive source selection failed: {fault.message}"
        held = self._hold_rig_motion(reason, context="human command selection provider")
        self.fault(reason if held else f"{reason}; hold_motion failed")
        return fault

    def _step_result(self, **kwargs: Any) -> SessionStepResult:
        kwargs.setdefault(
            "input_sample_refs",
            dict(self._current_input_sample_refs),
        )
        return SessionStepResult(
            **kwargs,
            motion_hold_mode=self._motion_hold_mode,
            motion_hold_reason=self._motion_hold_reason,
            tracker_loss_duration_sec=float(self._tracker_loss_duration_sec),
            tracker_loss_channels=self._tracker_loss_channels,
            home_recalibration_required=bool(self._home_recalibration_required),
        )

    def _tracker_channel_names(self) -> tuple[str, ...]:
        if self.config.tracker_channel_names:
            return self.config.tracker_channel_names
        return tuple(str(name) for name in self.safety_monitor.config.required_channels)

    def _tracker_hold_is_enabled(self) -> bool:
        return bool(self.config.tracker_loss_hold_enabled and self._tracker_channel_names())

    @staticmethod
    def _copy_command(command: TeleopCommand) -> TeleopCommand:
        return merge_teleop_commands((command,))

    def _held_command(self, now: float) -> TeleopCommand | None:
        if self._last_safe_command is None:
            return None
        held = self._copy_command(self._last_safe_command)
        held.metadata["motion_hold_mode"] = self._motion_hold_mode
        held.metadata["motion_hold_reason"] = self._motion_hold_reason
        held.metadata["held_from_command_timestamp"] = self._last_safe_command.timestamp
        held.timestamp = float(now)
        return held

    def _reset_tracker_loss_timing(self) -> None:
        self._tracker_unavailable_since.clear()
        self._tracker_last_healthy_at.clear()
        self._tracker_loss_duration_sec = 0.0
        self._tracker_loss_channels = ()

    def _observe_source_mapping_transitions(
        self,
        channels: Mapping[str, ObservationChannel],
    ) -> None:
        """Reset pose baselines when a source changes its logical binding.

        A source may keep a live display snapshot while rebuilding its causal
        history (for example, after swapping VIVE left/right roles).  The
        generation is deliberately a generic channel metadata seam rather than
        a VIVE-specific dependency.  The first sample in a new generation is
        accepted as the new baseline; it is not compared to the old physical
        source pose.
        """

        transitioned: list[str] = []
        for channel_name, channel in channels.items():
            raw_generation = (channel.metadata or {}).get(
                "source_mapping_generation"
            )
            if raw_generation is None:
                continue
            try:
                generation = int(raw_generation)
            except (TypeError, ValueError):
                continue
            previous = self._source_mapping_generation_by_channel.get(channel_name)
            if previous is not None and previous != generation:
                self.safety_monitor.request_pose_baseline_reset(channel_name)
                transitioned.append(channel_name)
            self._source_mapping_generation_by_channel[channel_name] = generation
        if transitioned:
            # Retargeter smoothing has the same physical-source continuity
            # assumption as the pose safety baseline.  Reset it once per
            # mapping transition, without changing authority or hold state.
            self._reset_retargeter_tracking()

    def mark_home_recalibrated(self, reason: str = "runtime home recalibrated") -> None:
        """Explicitly release the manual tracker-loss hold after a new home capture."""

        self._home_recalibration_required = False
        self._last_home_recalibration_reason = str(reason)
        self._motion_hold_mode = TRACKER_HOLD_NONE
        self._motion_hold_reason = ""
        self._last_safe_command = None
        self._reset_tracker_loss_timing()
        self._clear_recovery_context()
        self.safety_monitor.request_pose_baseline_reset_many(self._tracker_channel_names())
        self._reset_retargeter_tracking()

    def _update_tracker_loss(
        self,
        channels: dict[str, ObservationChannel],
        now: float,
    ) -> tuple[tuple[str, ...], float]:
        if not self._tracker_hold_is_enabled():
            self._tracker_loss_duration_sec = 0.0
            self._tracker_loss_channels = ()
            return (), 0.0

        unavailable: dict[str, str] = {}
        recovered_gap_by_channel: dict[str, float] = {}
        auto_hold_sec = float(self.config.tracker_auto_hold_sec)
        manual_hold_sec = float(self.config.tracker_manual_hold_sec)
        for channel_name in self._tracker_channel_names():
            channel = channels.get(channel_name)
            reason = ""
            stale_since: float | None = None
            if channel is None:
                reason = "missing"
            elif not bool(channel.valid):
                reason = "marked_invalid"
            elif channel.capability != "pose6d":
                reason = f"unexpected_capability:{channel.capability}"
            else:
                valid_pose, pose_error = validate_pose_matrix(channel.value)
                if not valid_pose:
                    reason = f"invalid_pose:{pose_error}"
                else:
                    try:
                        channel_timestamp = float(channel.timestamp)
                    except (TypeError, ValueError):
                        channel_timestamp = float("nan")
                    if not math.isfinite(channel_timestamp):
                        reason = "invalid_timestamp"
                    else:
                        age_sec = max(0.0, float(now) - channel_timestamp)
                        if age_sec > auto_hold_sec:
                            reason = f"stale:{age_sec:.6f}s"
                            stale_since = min(float(now), channel_timestamp)

            if not reason:
                previous_started_at = self._tracker_unavailable_since.get(channel_name)
                if previous_started_at is not None:
                    recovered_gap = max(0.0, float(now) - float(previous_started_at))
                    if recovered_gap > manual_hold_sec or (
                        recovered_gap > auto_hold_sec
                        and self._motion_hold_mode
                        not in {TRACKER_HOLD_AUTOMATIC, TRACKER_HOLD_RECOVERING}
                    ):
                        recovered_gap_by_channel[channel_name] = recovered_gap
                self._tracker_unavailable_since.pop(channel_name, None)
                self._tracker_last_healthy_at[channel_name] = float(now)
                continue

            unavailable[channel_name] = reason
            if channel_name not in self._tracker_unavailable_since:
                started_at = self._tracker_last_healthy_at.get(channel_name, float(now))
                if stale_since is not None:
                    started_at = min(float(started_at), float(stale_since))
                if not math.isfinite(started_at) or started_at > float(now):
                    started_at = float(now)
                self._tracker_unavailable_since[channel_name] = float(started_at)

        for channel_name in tuple(self._tracker_unavailable_since):
            if channel_name not in unavailable:
                self._tracker_unavailable_since.pop(channel_name, None)

        for channel_name, recovered_gap in recovered_gap_by_channel.items():
            unavailable[channel_name] = f"recovered_after_gap:{recovered_gap:.6f}s"

        loss_channels = tuple(sorted(unavailable))
        loss_duration = max(
            [
                max(0.0, float(now) - self._tracker_unavailable_since[channel_name])
                for channel_name in loss_channels
                if channel_name in self._tracker_unavailable_since
            ]
            + list(recovered_gap_by_channel.values()),
            default=0.0,
        )
        self._tracker_loss_channels = loss_channels
        self._tracker_loss_duration_sec = float(loss_duration)
        motion_active = self.state_machine.state in {
            TeleopRunState.ARMED,
            TeleopRunState.RUNNING,
        }
        if loss_channels and motion_active and self._motion_hold_mode == TRACKER_HOLD_NONE:
            self._motion_hold_mode = TRACKER_HOLD_GRACE
            self._motion_hold_reason = (
                "tracker target unavailable: "
                + ", ".join(f"{name}={unavailable[name]}" for name in loss_channels)
            )
        elif not loss_channels and self._motion_hold_mode == TRACKER_HOLD_GRACE:
            self._motion_hold_mode = TRACKER_HOLD_NONE
            self._motion_hold_reason = ""
        return loss_channels, float(loss_duration)

    def _is_tracker_availability_fault(self, fault: FaultEvent) -> bool:
        if fault.code in {"tracker_target_lost", "tracker_target_manual_hold"}:
            return True
        return bool(
            self._tracker_hold_is_enabled()
            and fault.code in _TRACKER_AVAILABILITY_FAULT_CODES
            and fault.channel_name in set(self._tracker_channel_names())
        )

    def _tracker_hold_fault(self, now: float, *, manual: bool) -> FaultEvent:
        threshold = (
            float(self.config.tracker_manual_hold_sec)
            if manual
            else float(self.config.tracker_auto_hold_sec)
        )
        code = "tracker_target_manual_hold" if manual else "tracker_target_lost"
        mode = TRACKER_HOLD_MANUAL if manual else TRACKER_HOLD_AUTOMATIC
        return FaultEvent(
            code=code,
            message=(
                f"Tracker target unavailable for {self._tracker_loss_duration_sec:.3f}s "
                f"(threshold {threshold:.3f}s); entering {mode}."
            ),
            severity=FaultSeverity.ERROR,
            timestamp=float(now),
            component="teleop_session",
            metadata={
                "motion_hold_mode": mode,
                "tracker_loss_duration_sec": float(self._tracker_loss_duration_sec),
                "tracker_loss_channels": list(self._tracker_loss_channels),
                "home_recalibration_required": bool(manual),
            },
        )

    def _enter_tracker_automatic_hold(self, fault: FaultEvent) -> None:
        if self.state_machine.state in {TeleopRunState.FAULT, TeleopRunState.ESTOP}:
            return
        entering = self._motion_hold_mode != TRACKER_HOLD_AUTOMATIC
        self._motion_hold_mode = TRACKER_HOLD_AUTOMATIC
        self._motion_hold_reason = fault.message
        if not entering:
            return
        self._record_freeze_context((fault,))
        self._freeze_channel_names = self._tracker_loss_channels
        self.safety_monitor.request_pose_baseline_reset_many(self._freeze_channel_names)
        if not self._hold_rig_motion(fault.message, context="automatic tracker hold"):
            self.fault("automatic tracker hold failed; manual intervention required")
            return
        self._transition(TeleopRunState.FROZEN, fault.message)

    def _enter_tracker_manual_hold(self, fault: FaultEvent) -> None:
        if self.state_machine.state in {TeleopRunState.FAULT, TeleopRunState.ESTOP}:
            return
        if self._home_recalibration_required:
            return
        self.hold_hardware(fault.message, require_home_recalibration=True)

    def _held_step_result(
        self,
        *,
        timestamp: float,
        control_hz: float,
        channels: dict[str, ObservationChannel],
        raw_channels: dict[str, ObservationChannel],
        faults: list[FaultEvent],
    ) -> SessionStepResult:
        return self._step_result(
            timestamp=float(timestamp),
            state=self.state_machine.state,
            control_hz=float(control_hz),
            channels=channels,
            raw_channels=raw_channels,
            observation=self._build_observation(channels, float(timestamp)),
            command=self._held_command(float(timestamp)),
            command_dispatched=False,
            faults=faults,
        )

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
        rollback_steps: list[tuple[str, Callable[[], None]]] = []
        try:
            for source in self.sources:
                rollback_steps.append((f"source[{source.name}]", source.close))
                source.start()
            if self.rig is not None:
                rollback_steps.append((f"rig[{self.rig.name}]", self.rig.close))
                self.rig.start()
            if self.recorder is not None:
                rollback_steps.append(("recorder", self.recorder.close))
            self._transition(TeleopRunState.CALIBRATION, "session started")
            if self.calibration is not None and self.recorder is not None:
                self.recorder.record_calibration(self.calibration.to_dict())
        except BaseException as start_error:  # noqa: BLE001 - rollback before propagating.
            self._started = False
            rollback_failures = collect_lifecycle_failures(reversed(rollback_steps))
            if rollback_failures:
                raise LifecycleStartupError(
                    "teleop_session.start",
                    start_error,
                    rollback_failures,
                ) from start_error
            raise
        self._started = True

    def set_calibration_ready(self) -> None:
        self._transition(TeleopRunState.DEBUG_ONLY, "calibration ready")

    def mark_debug_passed(self) -> None:
        self.debug_passed = True
        if self.state_machine.state == TeleopRunState.CALIBRATION:
            self._transition(TeleopRunState.DEBUG_ONLY, "debug passed")

    def arm_hardware(self) -> None:
        if self._home_recalibration_required:
            raise RuntimeError(
                "Cannot arm hardware after a manual tracker-loss hold until runtime home "
                "has been recalibrated and mark_home_recalibrated() has been called."
            )
        if self.config.allow_hardware_after_debug_pass and not self.debug_passed:
            raise RuntimeError("Cannot arm hardware before debug pass is marked complete.")
        if self.rig is not None and not self.rig.is_ready():
            raise RuntimeError("Rig is not ready.")
        self._transition(TeleopRunState.ARMED, "hardware armed")

    def _hold_rig_motion(self, reason: str, *, context: str) -> bool:
        if self.rig is not None:
            try:
                self.rig.hold_motion(reason)
            except Exception as exc:  # noqa: BLE001
                print(
                    "\033[1;31;40m"
                    + f"[Debug Info] {context} hold_motion failed: {type(exc).__name__}: {exc}"
                    + "\033[0m"
                )
                return False
        return True

    def hold_hardware(
        self,
        reason: str = "manual hold",
        *,
        require_home_recalibration: bool = False,
    ) -> bool:
        self._clear_recovery_context()
        self.safety_monitor.request_pose_baseline_reset_many(self.safety_monitor.config.required_channels)
        self._reset_retargeter_tracking()
        hold_succeeded = self._hold_rig_motion(reason, context="manual hold")
        if require_home_recalibration:
            self._home_recalibration_required = True
            self._last_home_recalibration_reason = ""
            self._motion_hold_mode = TRACKER_HOLD_MANUAL
            self._motion_hold_reason = str(reason)
        self._transition(TeleopRunState.DEBUG_ONLY, reason)
        if require_home_recalibration and not hold_succeeded:
            self.fault("manual tracker-loss hold failed; manual intervention required")
        return bool(hold_succeeded)

    def freeze(self, reason: str) -> None:
        already_frozen = self.state_machine.state == TeleopRunState.FROZEN
        print("\033[1;31;40m" + f"[Debug Info] FROZEN triggered due to: {reason}" + "\033[0m")
        self._transition(TeleopRunState.FROZEN, reason)
        if self.config.hold_motion_on_freeze and self.rig is not None and not already_frozen:
            self._hold_rig_motion(f"safety freeze: {reason}", context="safety freeze")

    def fault(self, reason: str) -> None:
        self.last_fault_reason = str(reason)
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
        self._motion_hold_mode = TRACKER_HOLD_NONE
        self._motion_hold_reason = ""
        self._last_safe_command = None
        self._reset_tracker_loss_timing()
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
            causal_reader = getattr(source, "get_latest_channels_at_or_before", None)
            source_channels = (
                causal_reader(float(now))
                if callable(causal_reader)
                else source.get_latest_channels()
            )
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
        self._current_human_command_emission = None
        commands: list[TeleopCommand] = []
        contribution_names: list[str] = []
        for index, retargeter in enumerate(self.retargeters):
            started_at = time.perf_counter()
            commands.append(retargeter.retarget(observation, self.calibration))
            contribution_names.append(
                str(getattr(retargeter, "name", "") or type(retargeter).__name__)
            )
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
        if (
            self.human_command_observer is not None
            or self.human_execution_observer is not None
        ):
            observer_started_at = time.perf_counter()
            try:
                emission = HumanCommandEmission.capture(
                    emission_sequence=self._human_command_emission_sequence,
                    # Eligibility uses the local monotonic observer clock. The
                    # source/command timestamp remains separately preserved in
                    # ``HumanCommandEmission.command_timestamp``.
                    emitted_at=time.monotonic(),
                    contributions=tuple(zip(contribution_names, commands, strict=True)),
                    merged_command=command,
                    observation_sample_refs=self._current_input_sample_refs,
                )
                self._human_command_emission_sequence += 1
                self._current_human_command_emission = emission
                if self.human_command_observer is not None:
                    self.human_command_observer(emission)
                    self.last_human_command_observer_error = None
            except Exception as exc:  # noqa: BLE001 - observe-only hook must not affect control.
                self.human_command_observer_error_count += 1
                self.last_human_command_observer_error = exc
            if phase_recorder is not None:
                phase_recorder(
                    "session.observe_human_command",
                    time.perf_counter() - observer_started_at,
                )
        return command

    def _observe_human_legacy_execution(
        self,
        *,
        command: TeleopCommand,
        observed_at: float,
        outcome: str,
        command_dispatched: bool,
        guard_fault_code: str | None = None,
        dispatch_receipts: Mapping[str, object | None] | None = None,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> None:
        emission = self._current_human_command_emission
        observer = self.human_execution_observer
        if emission is None or observer is None:
            return
        started_at = time.perf_counter()
        try:
            observer(
                HumanLegacyCommandObservation.capture(
                    emission=emission,
                    observed_at=float(observed_at),
                    outcome=outcome,
                    command_dispatched=command_dispatched,
                    run_state=self.state_machine.state.value,
                    command=command,
                    guard_fault_code=guard_fault_code,
                    dispatch_receipts=dispatch_receipts,
                )
            )
            self.last_human_execution_observer_error = None
        except Exception as exc:  # noqa: BLE001 - observe-only hook must not affect control.
            self.human_execution_observer_error_count += 1
            self.last_human_execution_observer_error = exc
        if phase_recorder is not None:
            phase_recorder(
                "session.observe_human_legacy_execution",
                time.perf_counter() - started_at,
            )

    def dispatch_command(
        self,
        command: TeleopCommand,
        *,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> Any:
        """Dispatch an already-validated command through the session's rig.

        Normal teleop steps, recovery helpers, and headless compatibility
        actions use this seam.  It intentionally does not perform another
        safety pass or state transition: callers that construct an external
        command remain responsible for their existing guard/state semantics.
        Keeping the transport call here prevents frontends from growing
        independent direct-rig dispatch paths.
        """

        if self.rig is None:
            raise RuntimeError("TeleopSession has no rig for command dispatch.")
        return self.rig.send_commands(command, phase_recorder=phase_recorder)

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
        self._observe_source_mapping_transitions(channels)
        self._current_input_sample_refs = dict(
            self._input_sample_tracker.observe(
                raw_channels,
                available_at=time.monotonic(),
                available_clock_domain_id="teleop_session.monotonic",
            )
        )
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

        tracker_loss_channels, tracker_loss_duration = self._update_tracker_loss(
            channels,
            current_time,
        )
        mark_phase("session.evaluate_tracker_loss")
        motion_active = self.state_machine.state in {
            TeleopRunState.ARMED,
            TeleopRunState.RUNNING,
            TeleopRunState.FROZEN,
            TeleopRunState.RECOVERING,
        }
        if (
            tracker_loss_channels
            and self.state_machine.state == TeleopRunState.FROZEN
            and self._motion_hold_mode == TRACKER_HOLD_AUTOMATIC
        ):
            self._healthy_frames_since_fault_clear = 0
        if tracker_loss_channels and motion_active:
            if tracker_loss_duration > float(self.config.tracker_manual_hold_sec):
                if not self._home_recalibration_required:
                    tracker_fault = self._tracker_hold_fault(current_time, manual=True)
                    faults.append(tracker_fault)
                    self._enter_tracker_manual_hold(tracker_fault)
                    mark_phase("session.tracker_manual_hold")
            elif (
                tracker_loss_duration > float(self.config.tracker_auto_hold_sec)
                and self.state_machine.state != TeleopRunState.FROZEN
            ):
                tracker_fault = self._tracker_hold_fault(current_time, manual=False)
                faults.append(tracker_fault)
                self._enter_tracker_automatic_hold(tracker_fault)
                mark_phase("session.tracker_automatic_hold")

        for fault in faults:
            if self.recorder is not None:
                self.recorder.record_fault(fault)
        if faults and self.recorder is not None:
            mark_phase("session.record_faults")

        freeze_faults = [
            fault
            for fault in faults
            if self._fault_should_freeze(fault)
            and not self._is_tracker_availability_fault(fault)
            and not self._home_recalibration_required
        ]
        mark_phase("session.freeze_fault_check")

        if freeze_faults and self.safety_monitor.config.freeze_on_fault:
            self._record_freeze_context(freeze_faults)
            self.freeze(freeze_faults[0].code)
            mark_phase("session.freeze_on_fault")
            return self._step_result(
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
            return self._step_result(
                timestamp=current_time,
                state=self.state_machine.state,
                control_hz=control_hz,
                channels=channels,
                raw_channels=raw_channels,
                faults=faults,
            )

        if self._home_recalibration_required or tracker_loss_channels:
            return self._held_step_result(
                timestamp=current_time,
                control_hz=control_hz,
                channels=channels,
                raw_channels=raw_channels,
                faults=faults,
            )

        if self.state_machine.state == TeleopRunState.FROZEN:
            if not self._freeze_is_auto_recoverable():
                return self._step_result(
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
                if self._motion_hold_mode == TRACKER_HOLD_AUTOMATIC:
                    return self._held_step_result(
                        timestamp=current_time,
                        control_hz=control_hz,
                        channels=channels,
                        raw_channels=raw_channels,
                        faults=faults,
                    )
                return self._step_result(
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
            if self._motion_hold_mode == TRACKER_HOLD_AUTOMATIC:
                self._motion_hold_mode = TRACKER_HOLD_RECOVERING
            mark_phase("session.transition_recovering")
            print("\033[1;32;40m" + f"[Debug Info] Auto-recovery triggered" + "\033[0m")
            return self._step_result(
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
            return self._step_result(
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
        command: TeleopCommand | None = None
        command_override_claimed = False
        if self.command_override_provider is not None:
            self._current_human_command_emission = None
            try:
                override = self.command_override_provider(
                    observation,
                    current_time,
                )
                if isinstance(override, TeleopCommandOverride):
                    command_override_claimed = True
                    command = override.command
                else:
                    command = override
                if command is not None and not isinstance(command, TeleopCommand):
                    raise TypeError(
                        "command_override_provider must return TeleopCommandOverride, "
                        "TeleopCommand, or None."
                    )
            except Exception as exc:  # noqa: BLE001 - external source must fail closed.
                fault = self._reject_command_override_provider_error(
                    exc,
                    timestamp=current_time,
                )
                faults.append(fault)
                mark_phase("session.command_override_provider_fault")
                return self._step_result(
                    timestamp=current_time,
                    state=self.state_machine.state,
                    control_hz=control_hz,
                    channels=channels,
                    raw_channels=raw_channels,
                    observation=observation,
                    faults=faults,
                )
            mark_phase("session.command_override_provider")
        if command_override_claimed and command is None:
            mark_phase("session.command_override_inhibited")
            return self._step_result(
                timestamp=current_time,
                state=self.state_machine.state,
                control_hz=control_hz,
                channels=channels,
                raw_channels=raw_channels,
                observation=observation,
                faults=faults,
            )
        if command is None:
            command = self._build_command(
                observation,
                phase_recorder=phase_recorder,
            )
            mark_phase("session.build_command")
            if self.human_command_selection_provider is not None:
                try:
                    selection = self.human_command_selection_provider(
                        observation,
                        command,
                        current_time,
                    )
                    if selection is not None and not isinstance(
                        selection,
                        TeleopCommandOverride,
                    ):
                        raise TypeError(
                            "human_command_selection_provider must return "
                            "TeleopCommandOverride or None."
                        )
                    if selection is not None:
                        # An explicit selection claim means that the Human
                        # candidate was not selected.  Clear the observe-only
                        # Human execution carrier before all downstream hooks.
                        self._current_human_command_emission = None
                        command = selection.command
                except Exception as exc:  # noqa: BLE001 - selection must fail closed.
                    fault = self._reject_human_command_selection_provider_error(
                        exc,
                        timestamp=current_time,
                    )
                    faults.append(fault)
                    mark_phase("session.human_command_selection_provider_fault")
                    return self._step_result(
                        timestamp=current_time,
                        state=self.state_machine.state,
                        control_hz=control_hz,
                        channels=channels,
                        raw_channels=raw_channels,
                        observation=observation,
                        faults=faults,
                    )
                mark_phase("session.human_command_selection_provider")
                if command is None:
                    mark_phase("session.human_command_selection_inhibited")
                    return self._step_result(
                        timestamp=current_time,
                        state=self.state_machine.state,
                        control_hz=control_hz,
                        channels=channels,
                        raw_channels=raw_channels,
                        observation=observation,
                        faults=faults,
                    )
                command = self._copy_command(command)
                if command.timestamp is None:
                    command.timestamp = observation.timestamp
                command.metadata.setdefault("tcp_mode", self.config.tcp_mode)
                mark_phase("session.copy_selected_command")
        else:
            command = self._copy_command(command)
            if command.timestamp is None:
                command.timestamp = observation.timestamp
            command.metadata.setdefault("tcp_mode", self.config.tcp_mode)
            mark_phase("session.copy_override_command")
        if self.command_filter is not None:
            command = self.command_filter(command, observation, current_time)
            mark_phase("session.filter_command")
        emission = self._current_human_command_emission
        if emission is not None and self.human_command_filter_observer is not None:
            observer_started_at = time.perf_counter()
            try:
                self.human_command_filter_observer(
                    emission,
                    self._copy_command(command),
                    time.monotonic(),
                )
                self.last_human_command_filter_observer_error = None
            except Exception as exc:  # noqa: BLE001 - observe-only hook.
                self.human_command_filter_observer_error_count += 1
                self.last_human_command_filter_observer_error = exc
            if phase_recorder is not None:
                phase_recorder(
                    "session.observe_human_filtered_command",
                    time.perf_counter() - observer_started_at,
                )

        command_guard_fault = self.evaluate_command_guard(command, observation, current_time)
        if command_guard_fault is not None:
            faults.append(command_guard_fault)
            self.reject_command_for_guard_fault(command_guard_fault)
            self._observe_human_legacy_execution(
                command=command,
                observed_at=time.monotonic(),
                outcome="guard_rejected",
                command_dispatched=False,
                guard_fault_code=command_guard_fault.code,
                phase_recorder=phase_recorder,
            )
            mark_phase("session.command_guard_fault")
            return self._step_result(
                timestamp=current_time,
                state=self.state_machine.state,
                control_hz=control_hz,
                channels=channels,
                raw_channels=raw_channels,
                observation=observation,
                faults=faults,
            )
        mark_phase("session.command_guard")

        dispatch_admitted = True
        if self.command_dispatch_admission is not None:
            dispatch_admitted = bool(
                self.command_dispatch_admission(
                    command,
                    observation,
                    current_time,
                )
            )
            mark_phase("session.command_dispatch_admission")

        if self.recorder is not None:
            self.recorder.record_command(command)
            mark_phase("session.record_command")

        command_dispatched = False
        dispatch_receipts: Mapping[str, object | None] | None = None
        if dispatch_admitted and self.rig is not None and self.state_machine.state in {
            TeleopRunState.ARMED,
            TeleopRunState.RUNNING,
        }:
            dispatch_receipts = self.dispatch_command(
                command,
                phase_recorder=phase_recorder,
            )
            command_dispatched = True
            self._last_safe_command = self._copy_command(command)
            mark_phase("session.rig.send_commands")
            if self.state_machine.state != TeleopRunState.RUNNING:
                self._transition(TeleopRunState.RUNNING, "first command dispatched")
                mark_phase("session.transition_running")

        self._observe_human_legacy_execution(
            command=command,
            observed_at=time.monotonic(),
            outcome=(
                "dispatched"
                if command_dispatched
                else "dispatch_inhibited"
                if not dispatch_admitted
                else "not_dispatched"
            ),
            command_dispatched=command_dispatched,
            dispatch_receipts=dispatch_receipts,
            phase_recorder=phase_recorder,
        )

        return self._step_result(
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
        steps: list[tuple[str, Callable[[], None]]] = []
        if self.recorder is not None:
            steps.append(("recorder", self.recorder.close))
        if self.rig is not None:
            steps.append((f"rig[{self.rig.name}]", self.rig.close))
        steps.extend(
            (f"source[{source.name}]", source.close)
            for source in reversed(self.sources)
        )
        try:
            run_best_effort_cleanup("teleop_session.close", steps)
        finally:
            self._started = False

    def __enter__(self) -> "TeleopSession":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
