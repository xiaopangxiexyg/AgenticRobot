"""UI-neutral contracts for RealTeleop progress application bindings.

Control Station owns browser events, operator intents, and rebuildable UI
state. The real-teleop core owns the command and runtime surfaces it needs
after those intents have been admitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Literal, Protocol


ProgressMutationKind = Literal[
    "ADVANCE_PROGRESS",
    "UNDO_LAST_ADVANCE",
    "REPORT_FAILURE",
]

PROGRESS_ARM_CONTROL_MODE_STRICT = "strict_profile"
PROGRESS_ARM_CONTROL_MODE_PROFILE = "profile"
PROGRESS_ARM_CONTROL_MODE_IGNORE = "ignore_profile"
PROGRESS_ARM_CONTROL_MODES = (
    PROGRESS_ARM_CONTROL_MODE_STRICT,
    PROGRESS_ARM_CONTROL_MODE_PROFILE,
    PROGRESS_ARM_CONTROL_MODE_IGNORE,
)


def normalize_progress_arm_control_mode(value: object) -> str:
    mode = str(value or PROGRESS_ARM_CONTROL_MODE_PROFILE).strip().lower()
    if mode not in PROGRESS_ARM_CONTROL_MODES:
        raise ValueError(f"Unsupported progress arm-control mode {mode!r}.")
    return mode


@dataclass(frozen=True)
class ProgressMutationCommand:
    """One Station-admitted command consumed by the real-teleop core."""

    command_id: str
    kind: ProgressMutationKind
    target_progress_step_id: str | None = None

    def __post_init__(self) -> None:
        command_id = str(self.command_id).strip()
        kind = str(self.kind).strip().upper()
        target = (
            None
            if self.target_progress_step_id is None
            else str(self.target_progress_step_id).strip()
        )
        if not command_id:
            raise ValueError("command_id must be non-empty.")
        if kind not in {
            "ADVANCE_PROGRESS",
            "UNDO_LAST_ADVANCE",
            "REPORT_FAILURE",
        }:
            raise ValueError(f"Unsupported progress mutation command {kind!r}.")
        if kind != "ADVANCE_PROGRESS" and target is not None:
            raise ValueError(
                "Only ADVANCE_PROGRESS may select a target progress step."
            )
        if target == "":
            raise ValueError("target_progress_step_id must be non-empty.")
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target_progress_step_id", target)

    @property
    def intent_id(self) -> str:
        """Compatibility spelling used by the progress fact adapter."""

        return self.command_id


@dataclass(frozen=True)
class ProgressProfileSelectionCommand:
    """Station-admitted managed-profile selection consumed by RealTeleop."""

    command_id: str
    profile_id: str

    def __post_init__(self) -> None:
        for name in ("command_id", "profile_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)

    @property
    def intent_id(self) -> str:
        return self.command_id


@dataclass(frozen=True)
class ProgressArmControlModeSelectionCommand:
    """Station-admitted idle arm-control-mode selection."""

    command_id: str
    mode: str

    def __post_init__(self) -> None:
        command_id = str(self.command_id).strip()
        if not command_id:
            raise ValueError("command_id must be non-empty.")
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(
            self,
            "mode",
            normalize_progress_arm_control_mode(self.mode),
        )

    @property
    def intent_id(self) -> str:
        return self.command_id


@dataclass(frozen=True)
class ProgressEventBindings:
    """Opaque Station event identities needed by physical input routing."""

    advance: str
    undo: str
    failure: str

    def __post_init__(self) -> None:
        for name in ("advance", "undo", "failure"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ProgressApplicationProjection:
    """One rebuildable Station projection returned to the lower adapter."""

    configured: bool
    active_controls: tuple[str, ...]
    recording_active: bool
    phase: str
    application_held: bool
    allowed_next_step_ids: tuple[str, ...]
    blocked_reason: str
    ui_payload: Mapping[str, object]


@dataclass(frozen=True)
class PromotableHoldPause:
    """Exact lower-owned effects for one uninterrupted ordinary Hold."""

    episode_capture_identity_hash: str
    hold_requested_at: float
    application_hold_effective_at: float
    formal_capture_pause_effective_at: float
    hold_reason: str

    def __post_init__(self) -> None:
        identity_hash = str(self.episode_capture_identity_hash).strip()
        reason = str(self.hold_reason).strip()
        if not identity_hash:
            raise ValueError(
                "episode_capture_identity_hash must be non-empty."
            )
        if not reason:
            raise ValueError("hold_reason must be non-empty.")
        times = {
            "hold_requested_at": self.hold_requested_at,
            "application_hold_effective_at": (
                self.application_hold_effective_at
            ),
            "formal_capture_pause_effective_at": (
                self.formal_capture_pause_effective_at
            ),
        }
        normalized: dict[str, float] = {}
        for name, value in times.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite.")
            normalized[name] = float(value)
        if (
            normalized["application_hold_effective_at"]
            < normalized["hold_requested_at"]
            or normalized["formal_capture_pause_effective_at"]
            < normalized["hold_requested_at"]
        ):
            raise ValueError(
                "Applied Hold/Pause effects cannot precede the Hold request."
            )
        object.__setattr__(
            self,
            "episode_capture_identity_hash",
            identity_hash,
        )
        object.__setattr__(self, "hold_reason", reason)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


class ProgressWorkflowResultView(Protocol):
    handled: bool
    phase: str
    message: str


class RealTeleopProgressWorkflowPort(Protocol):
    """Lower-facing surface implemented by the Station application adapter."""

    def availability(self, **kwargs: Any) -> Any: ...

    def handle_command(
        self,
        command: ProgressMutationCommand,
        **kwargs: Any,
    ) -> ProgressWorkflowResultView: ...

    def request_pending_failure_effect(
        self,
        **kwargs: Any,
    ) -> ProgressWorkflowResultView: ...

    def record_pending_failure_effect_receipt(
        self,
        **kwargs: Any,
    ) -> ProgressWorkflowResultView: ...

    def observe_applied_resume(
        self,
        **kwargs: Any,
    ) -> ProgressWorkflowResultView: ...

    def consume_sample_acceptance(
        self,
        **kwargs: Any,
    ) -> ProgressWorkflowResultView: ...


class ProgressApplicationPort(Protocol):
    """Application-leaf operations consumed by lifecycle/control core."""

    @property
    def event_bindings(self) -> ProgressEventBindings: ...

    def build_progress_workflow(
        self,
        recorder: object,
    ) -> RealTeleopProgressWorkflowPort: ...

    def command_from_event(
        self,
        *,
        event: str,
        command_id: str,
    ) -> ProgressMutationCommand | None: ...

    def profile_selection_from_event(
        self,
        *,
        event: str,
        command_id: str,
    ) -> ProgressProfileSelectionCommand | None: ...

    def arm_control_mode_selection_from_event(
        self,
        *,
        event: str,
        command_id: str,
    ) -> ProgressArmControlModeSelectionCommand | None: ...

    def build_startup_progress_ui_payload(
        self,
        *,
        dataset_recorder: object | None,
        profile_selection_enabled: bool,
    ) -> dict[str, object]: ...

    def select_managed_progress_profile(
        self,
        *,
        dataset_recorder: object | None,
        profile_id: str,
    ) -> Any: ...

    def project_progress_application_state(
        self,
        *,
        dataset_recorder: object | None,
        progress_workflow: RealTeleopProgressWorkflowPort | None,
        promotable_hold_pause: PromotableHoldPause | None,
        application_held: bool,
        arm_transition_pending: bool,
        profile_selection_enabled: bool,
    ) -> ProgressApplicationProjection: ...


__all__ = [
    "PROGRESS_ARM_CONTROL_MODES",
    "PROGRESS_ARM_CONTROL_MODE_IGNORE",
    "PROGRESS_ARM_CONTROL_MODE_PROFILE",
    "PROGRESS_ARM_CONTROL_MODE_STRICT",
    "ProgressApplicationProjection",
    "ProgressApplicationPort",
    "ProgressArmControlModeSelectionCommand",
    "ProgressEventBindings",
    "ProgressMutationCommand",
    "ProgressMutationKind",
    "ProgressProfileSelectionCommand",
    "ProgressWorkflowResultView",
    "PromotableHoldPause",
    "RealTeleopProgressWorkflowPort",
    "normalize_progress_arm_control_mode",
]
