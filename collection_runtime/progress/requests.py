"""Application-neutral requests accepted by the progress controller."""

from __future__ import annotations

from dataclasses import dataclass
import math

from dataset_contracts.progress_facts import FAILURE_TIMING_SEMANTICS_V1


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean.")
    return value


@dataclass(frozen=True)
class ProgressRequestContext:
    """Caller-owned identity and causal time for one accepted request."""

    request_id: str
    effective_at: float
    effective_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    actor_id: str
    origin: str = "online_operator"

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "effective_clock_domain_id",
            "available_clock_domain_id",
            "actor_id",
            "origin",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        effective = _finite(self.effective_at, name="effective_at")
        available = _finite(self.available_at, name="available_at")
        if (
            self.effective_clock_domain_id == self.available_clock_domain_id
            and available < effective
        ):
            raise ValueError("Request availability cannot precede effective time.")
        object.__setattr__(self, "effective_at", effective)
        object.__setattr__(self, "available_at", available)


@dataclass(frozen=True)
class StartEpisodeProgressRequest:
    attempt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt_id",
            _required(self.attempt_id, name="attempt_id"),
        )


@dataclass(frozen=True)
class AdvanceProgressStepRequest:
    to_attempt_id: str
    target_progress_step_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "to_attempt_id",
            _required(self.to_attempt_id, name="to_attempt_id"),
        )
        target = (
            None
            if self.target_progress_step_id is None
            else _required(
                self.target_progress_step_id,
                name="target_progress_step_id",
            )
        )
        object.__setattr__(self, "target_progress_step_id", target)


@dataclass(frozen=True)
class UndoLastProgressAdvanceRequest:
    advance_fact_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "advance_fact_id",
            _required(self.advance_fact_id, name="advance_fact_id"),
        )


@dataclass(frozen=True)
class ReportAttemptFailureRequest:
    """Report failure at the accepted request's Episode effective time.

    ``capture_failure_trigger`` only controls the optional capture-time
    provenance receipt.  It does not alter the Failure fact, its timing, or
    any reducer/effect behavior; the default keeps the new receipt enabled.
    """

    timing_semantics: str = "OPERATOR_REPORT_TIME"
    capture_failure_trigger: bool = True

    def __post_init__(self) -> None:
        semantics = _required(
            self.timing_semantics,
            name="timing_semantics",
        ).upper()
        if semantics not in FAILURE_TIMING_SEMANTICS_V1:
            raise ValueError(
                f"Unsupported failure timing semantics {semantics!r}."
            )
        object.__setattr__(self, "timing_semantics", semantics)
        object.__setattr__(
            self,
            "capture_failure_trigger",
            _boolean(
                self.capture_failure_trigger,
                name="capture_failure_trigger",
            ),
        )


@dataclass(frozen=True)
class ResumeFailedAttemptRequest:
    new_attempt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "new_attempt_id",
            _required(self.new_attempt_id, name="new_attempt_id"),
        )


@dataclass(frozen=True)
class CompleteEpisodeProgressRequest:
    """Normal owner termination; windows are truncated unless explicitly complete."""

    completed_progress_step: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "completed_progress_step",
            _boolean(
                self.completed_progress_step,
                name="completed_progress_step",
            ),
        )


@dataclass(frozen=True)
class AbortEpisodeProgressRequest:
    termination_cause: str = "ABORTED"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "termination_cause",
            _required(self.termination_cause, name="termination_cause").upper(),
        )


__all__ = [
    "AbortEpisodeProgressRequest",
    "AdvanceProgressStepRequest",
    "CompleteEpisodeProgressRequest",
    "ProgressRequestContext",
    "ReportAttemptFailureRequest",
    "ResumeFailedAttemptRequest",
    "StartEpisodeProgressRequest",
    "UndoLastProgressAdvanceRequest",
]
