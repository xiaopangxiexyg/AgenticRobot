"""Control Station coordinator for optional RealTeleop collection progress.

The generic collection reducer owns progress facts.  This application-leaf
object coordinates lower-owned Hold and formal-capture lifecycle ports and
aligns Retry to the first writer-accepted sample.  It owns neither robot
safety, writer admission, nor dataset Episode commitment, and it imports no
concrete RealTeleop implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any, Protocol, cast
import uuid

from .progress import (
    ProgressControlAvailability,
    ProgressOperatorIntent,
)


class ProgressEffectRequestPort(Protocol):
    """Stable effect identity exposed by the lower progress runtime."""

    idempotency_key: str
    source_failure_fact_id: str
    requested_operations: tuple[str, ...]


class RealTeleopProgressEpisodePort(Protocol):
    """Lower-owned progress Episode operations used by the Station workflow."""

    identity: Any
    controller: Any
    store: Any

    @property
    def state(self) -> Any: ...

    @property
    def allowed_next_step_ids(self) -> tuple[str, ...]: ...

    @property
    def can_undo_last_advance(self) -> bool: ...

    def advance(self, **kwargs: Any) -> Any: ...

    def undo_last_advance(self, **kwargs: Any) -> Any: ...

    def report_failure(self, **kwargs: Any) -> ProgressEffectRequestPort: ...

    def record_applied_hold_pause(self, **kwargs: Any) -> Any: ...

    def record_applied_resume(self, **kwargs: Any) -> Any: ...

    def retry_on_accepted_sample(self, receipt: Any, **kwargs: Any) -> Any: ...


class RealTeleopProgressRecorderPort(Protocol):
    """Recorder-facing surface; persistence and Episode ownership stay below."""

    @property
    def active_progress_episode(
        self,
    ) -> RealTeleopProgressEpisodePort | None: ...

    @property
    def is_recording(self) -> bool: ...

    def drain_accepted_sample_receipts(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class PromotableHoldPause:
    """Exact lower-owned effects for one uninterrupted ordinary Hold.

    This is application coordination state, not a persisted dataset contract.
    The eventual Failure fact and Hold/Pause receipt remain the canonical
    progress records.
    """

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


@dataclass(frozen=True)
class ProgressWorkflowResult:
    handled: bool
    phase: str
    message: str = ""


class RealTeleopProgressWorkflow:
    """Thin, default-off bridge around one active formal collection Episode."""

    def __init__(self, *, recorder: RealTeleopProgressRecorderPort) -> None:
        self.recorder = recorder

    def _episode(self) -> RealTeleopProgressEpisodePort | None:
        return cast(
            RealTeleopProgressEpisodePort | None,
            getattr(self.recorder, "active_progress_episode", None),
        )

    def _hold_episode_identity_matches(
        self,
        candidate: PromotableHoldPause | None,
    ) -> bool:
        episode = self._episode()
        if episode is None or candidate is None:
            return False
        identity_hash = str(
            getattr(episode.identity, "semantic_content_hash", "")
        )
        return bool(
            identity_hash == candidate.episode_capture_identity_hash
        )

    def _matches_promotable_hold(
        self,
        candidate: PromotableHoldPause | None,
    ) -> bool:
        episode = self._episode()
        return bool(
            episode is not None
            and episode.state.phase == "ACTIVE_ATTEMPT"
            and self._hold_episode_identity_matches(candidate)
        )

    def availability(
        self,
        *,
        application_held: bool = False,
        promotable_hold_pause: PromotableHoldPause | None = None,
    ) -> ProgressControlAvailability:
        episode = self._episode()
        recording = bool(getattr(self.recorder, "is_recording", False))
        if episode is None:
            return ProgressControlAvailability()
        promotion_available = bool(
            application_held
            and self._matches_promotable_hold(promotable_hold_pause)
        )
        return ProgressControlAvailability(
            enabled=True,
            recording_active=recording,
            phase=episode.state.phase,
            allowed_next_step_ids=episode.allowed_next_step_ids,
            can_undo_last_advance=episode.can_undo_last_advance,
            application_held=bool(application_held),
            failure_promotes_existing_hold=promotion_available,
        )

    def handle_operator_intent(
        self,
        intent: ProgressOperatorIntent,
        *,
        now: float,
        request_application_hold: Callable[[str], None],
        pause_formal_capture: Callable[[float, str], bool],
        application_held: bool = False,
        promotable_hold_pause: PromotableHoldPause | None = None,
    ) -> ProgressWorkflowResult:
        if not isinstance(intent, ProgressOperatorIntent):
            raise TypeError("intent must be ProgressOperatorIntent.")
        episode = self._episode()
        if episode is None:
            return ProgressWorkflowResult(
                handled=False,
                phase="ABSENT",
                message="collection progress is not enabled for this Episode",
            )
        promotion_available = bool(
            application_held
            and self._matches_promotable_hold(promotable_hold_pause)
        )
        if application_held and (
            intent.kind != "REPORT_FAILURE" or not promotion_available
        ):
            raise RuntimeError(
                "While application Hold is active, only a Failure promotion "
                "bound to that same uninterrupted Hold/Pause is allowed."
            )
        if intent.kind == "ADVANCE_PROGRESS":
            result = episode.advance(
                intent_id=intent.intent_id,
                effective_at=now,
                target_progress_step_id=intent.target_progress_step_id,
            )
            return ProgressWorkflowResult(True, result.state.phase)
        if intent.kind == "UNDO_LAST_ADVANCE":
            result = episode.undo_last_advance(
                intent_id=intent.intent_id,
                effective_at=now,
            )
            return ProgressWorkflowResult(True, result.state.phase)
        if intent.kind != "REPORT_FAILURE":
            raise ValueError(f"Unsupported progress intent {intent.kind!r}.")

        # Direct Failure fsyncs before requesting effects. A promoted Hold
        # instead binds already-applied exact effects and records its later
        # classification through available_at. In both paths, clear old
        # admission receipts so a pre-Failure in-flight sample can never be
        # mistaken for the first Retry sample.
        if promotion_available:
            assert promotable_hold_pause is not None
            effect_request = episode.report_failure(
                intent_id=intent.intent_id,
                effective_at=(
                    promotable_hold_pause.formal_capture_pause_effective_at
                ),
                available_at=now,
                timing_semantics="PROMOTED_HOLD_TIME",
            )
        else:
            effect_request = episode.report_failure(
                intent_id=intent.intent_id,
                effective_at=now,
            )
        self.recorder.drain_accepted_sample_receipts()
        if promotion_available:
            # The lower capture owner keeps the original pause boundary and
            # appends this classification as a second reason.  It must not
            # reopen, shift, or manufacture a new pause.
            pause_formal_capture(
                now,
                "progress_failure_promoted_from_hold",
            )
            result = episode.record_applied_hold_pause(
                effect_request=effect_request,
                requested_at=promotable_hold_pause.hold_requested_at,
                application_hold_effective_at=(
                    promotable_hold_pause.application_hold_effective_at
                ),
                formal_capture_pause_effective_at=(
                    promotable_hold_pause.formal_capture_pause_effective_at
                ),
                available_at=now,
            )
            return ProgressWorkflowResult(
                True,
                result.state.phase,
                message=(
                    "The existing uninterrupted Hold/Pause was promoted to "
                    "the current Attempt Failure without moving its cut."
                ),
            )
        return self._request_failure_effect(
            phase=episode.state.phase,
            now=now,
            request_application_hold=request_application_hold,
            pause_formal_capture=pause_formal_capture,
        )

    def request_pending_failure_effect(
        self,
        *,
        now: float,
        request_application_hold: Callable[[str], None],
        pause_formal_capture: Callable[[float, str], bool],
        promotable_hold_pause: PromotableHoldPause | None = None,
    ) -> ProgressWorkflowResult:
        """Replay a direct Failure effect or bind a verified promoted receipt."""

        episode = self._episode()
        if episode is None or episode.state.phase != "AWAITING_HOLD_RECEIPT":
            return ProgressWorkflowResult(
                False,
                "ABSENT" if episode is None else episode.state.phase,
            )
        effects = episode.controller.pending_effect_requests()
        if len(effects) != 1:
            raise RuntimeError(
                "Pending progress Failure must expose exactly one effect request."
            )
        effect = cast(ProgressEffectRequestPort, effects[0])
        failure_fact = episode.store.fact_by_id(effect.source_failure_fact_id)
        if failure_fact is None:
            raise RuntimeError("Pending progress Failure fact is unavailable.")
        if getattr(
            failure_fact.payload,
            "timing_semantics",
            "",
        ) == "PROMOTED_HOLD_TIME":
            if not self._hold_episode_identity_matches(
                promotable_hold_pause
            ):
                return ProgressWorkflowResult(
                    False,
                    episode.state.phase,
                    message=(
                        "Promoted Failure is durable but its original "
                        "Hold/Pause receipt is unavailable; remaining fail-closed."
                    ),
                )
            assert promotable_hold_pause is not None
            pause_formal_capture(
                now,
                "progress_failure_promoted_from_hold",
            )
            return self.record_pending_failure_effect_receipt(
                application_hold_effective_at=(
                    promotable_hold_pause.application_hold_effective_at
                ),
                formal_capture_pause_effective_at=(
                    promotable_hold_pause.formal_capture_pause_effective_at
                ),
                available_at=now,
                requested_at=promotable_hold_pause.hold_requested_at,
            )
        return self._request_failure_effect(
            phase=episode.state.phase,
            now=now,
            request_application_hold=request_application_hold,
            pause_formal_capture=pause_formal_capture,
        )

    @staticmethod
    def _request_failure_effect(
        *,
        phase: str,
        now: float,
        request_application_hold: Callable[[str], None],
        pause_formal_capture: Callable[[float, str], bool],
    ) -> ProgressWorkflowResult:
        request_application_hold("progress_failure")
        pause_formal_capture(now, "progress_failure")
        return ProgressWorkflowResult(
            handled=True,
            phase=phase,
            message=(
                "Failure is durable; waiting for lower-owned application Hold "
                "and formal capture Pause effects"
            ),
        )

    def record_pending_failure_effect_receipt(
        self,
        *,
        application_hold_effective_at: float,
        formal_capture_pause_effective_at: float,
        available_at: float,
        requested_at: float | None = None,
    ) -> ProgressWorkflowResult:
        """Persist APPLIED only after both lower owners report exact effects."""

        episode = self._episode()
        if episode is None or episode.state.phase != "AWAITING_HOLD_RECEIPT":
            return ProgressWorkflowResult(
                False,
                "ABSENT" if episode is None else episode.state.phase,
            )
        effects = episode.controller.pending_effect_requests()
        if len(effects) != 1:
            raise RuntimeError(
                "Pending progress Failure must expose exactly one effect request."
            )
        effect = cast(ProgressEffectRequestPort, effects[0])
        failure_fact = episode.store.fact_by_id(effect.source_failure_fact_id)
        if failure_fact is None:
            raise RuntimeError("Pending progress Failure fact is unavailable.")
        operation_requested_at = (
            float(failure_fact.available_at)
            if requested_at is None
            else float(requested_at)
        )
        # A direct Failure may arrive while a lower-owned modality-health
        # pause is already active.  Re-applying FORMAL_CAPTURE_PAUSE is then
        # an idempotent operation: this Failure request is satisfied at its
        # own request time, while the recorder keeps the earlier raw pause cut
        # in the Episode pause segments.  Do not claim that the new request
        # caused an effect in the past, which ProgressOperationReceiptV1
        # correctly rejects.
        application_hold_satisfied_at = max(
            float(application_hold_effective_at),
            operation_requested_at,
        )
        formal_capture_pause_satisfied_at = max(
            float(formal_capture_pause_effective_at),
            operation_requested_at,
        )
        result = episode.record_applied_hold_pause(
            effect_request=effect,
            requested_at=operation_requested_at,
            application_hold_effective_at=application_hold_satisfied_at,
            formal_capture_pause_effective_at=(
                formal_capture_pause_satisfied_at
            ),
            available_at=available_at,
        )
        return ProgressWorkflowResult(True, result.state.phase)

    def observe_applied_resume(
        self,
        *,
        now: float,
        application_running: bool,
        formal_capture_paused: bool,
        formal_capture_resume_effective_at: float | None,
    ) -> ProgressWorkflowResult:
        episode = self._episode()
        if episode is None:
            return ProgressWorkflowResult(False, "ABSENT")
        if episode.state.phase != "PAUSED_NO_ACTIVE_ATTEMPT":
            return ProgressWorkflowResult(False, episode.state.phase)
        if (
            not application_running
            or formal_capture_paused
            or formal_capture_resume_effective_at is None
        ):
            return ProgressWorkflowResult(False, episode.state.phase)
        operation_id = f"resume:{episode.identity.episode_uuid}:{uuid.uuid4().hex}"
        result = episode.record_applied_resume(
            resume_operation_id=operation_id,
            requested_at=formal_capture_resume_effective_at,
            application_resume_effective_at=now,
            formal_capture_resume_effective_at=(
                formal_capture_resume_effective_at
            ),
            available_at=now,
        )
        return ProgressWorkflowResult(True, result.state.phase)

    def consume_sample_acceptance(
        self,
        *,
        available_at: float,
    ) -> ProgressWorkflowResult:
        episode = self._episode()
        receipts = tuple(self.recorder.drain_accepted_sample_receipts())
        if episode is None:
            return ProgressWorkflowResult(False, "ABSENT")
        if episode.state.phase != "RESUME_RECEIVED":
            return ProgressWorkflowResult(False, episode.state.phase)
        resume_fact_id = episode.state.resume_receipt_fact_id
        resume_fact = (
            None
            if resume_fact_id is None
            else episode.store.fact_by_id(resume_fact_id)
        )
        # Retry may begin only after both lower-owned Resume effects. The
        # envelope effective time is their validated maximum.
        resume_at = None if resume_fact is None else resume_fact.effective_at
        for receipt in receipts:
            if resume_at is not None and receipt.sample_timestamp < resume_at:
                continue
            result = episode.retry_on_accepted_sample(
                receipt,
                available_at=available_at,
            )
            if result is not None:
                return ProgressWorkflowResult(True, result.state.phase)
        return ProgressWorkflowResult(False, episode.state.phase)


__all__ = [
    "PromotableHoldPause",
    "ProgressWorkflowResult",
    "ProgressEffectRequestPort",
    "RealTeleopProgressEpisodePort",
    "RealTeleopProgressRecorderPort",
    "RealTeleopProgressWorkflow",
]
