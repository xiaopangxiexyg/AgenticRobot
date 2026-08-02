"""Typed source-of-truth facts for online collection progress."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, ClassVar, Mapping, Protocol, TypeAlias

from .canonical_encoding import canonical_content_hash


PROGRESS_FACT_ENVELOPE_SCHEMA_V1 = "dexteleop.progress_fact_envelope.v1"
EPISODE_PROGRESS_STARTED_SCHEMA_V1 = "dexteleop.episode_progress_started.v1"
PROGRESS_ADVANCE_EVENT_SCHEMA_V1 = "dexteleop.progress_advance_event.v1"
PROGRESS_UNDO_EVENT_SCHEMA_V1 = "dexteleop.progress_undo_event.v1"
ATTEMPT_FAILURE_EVENT_SCHEMA_V1 = "dexteleop.attempt_failure_event.v1"
HOLD_PAUSE_EFFECT_RECEIPT_SCHEMA_V1 = (
    "dexteleop.hold_pause_effect_receipt.v1"
)
RESUME_AFTER_FAILURE_RECEIPT_SCHEMA_V1 = (
    "dexteleop.resume_after_failure_receipt.v1"
)
ATTEMPT_RETRY_EVENT_SCHEMA_V1 = "dexteleop.attempt_retry_event.v1"
EPISODE_PROGRESS_TERMINATION_SCHEMA_V1 = (
    "dexteleop.episode_progress_termination.v1"
)
PROGRESS_EPISODE_MANIFEST_SCHEMA_V1 = "dexteleop.progress_episode_manifest.v1"

PROGRESS_FACT_FILE_DOMAIN_V1 = "dexteleop.progress_fact.file.v1"
PROGRESS_PROFILE_FILE_DOMAIN_V1 = "dexteleop.progress_profile.file.v1"
PROGRESS_BINDING_FILE_DOMAIN_V1 = "dexteleop.progress_binding.file.v1"
PROGRESS_MANIFEST_FILE_DOMAIN_V1 = "dexteleop.progress_manifest.file.v1"

PROGRESS_EFFECT_STATUSES_V1 = frozenset({"APPLIED", "REJECTED", "TIMEOUT"})
PROGRESS_EFFECT_OPERATIONS_V1 = frozenset(
    {
        "APPLICATION_HOLD",
        "FORMAL_CAPTURE_PAUSE",
        "APPLICATION_RESUME",
        "FORMAL_CAPTURE_RESUME",
    }
)
FAILURE_TIMING_SEMANTICS_V1 = frozenset(
    {
        "OPERATOR_REPORT_TIME",
        # The operator first requested an ordinary application Hold and only
        # then classified that same uninterrupted pause as an Attempt
        # Failure.  The fact effective time is the authoritative original
        # formal-capture pause boundary; envelope available_at remains the
        # later classification time.
        "PROMOTED_HOLD_TIME",
    }
)
PROGRESS_TERMINATION_CAUSES_V1 = frozenset(
    {
        "FULL_TASK_COMPLETED",
        "WINDOW_COMPLETED",
        "MANUAL_STOP",
        "SAFETY_STOP",
        "ABORTED",
        "UNKNOWN",
    }
)
PROGRESS_ATTEMPT_DISPOSITIONS_V1 = frozenset(
    {
        "SUCCESS",
        "FAILURE",
        "ABORTED",
        "TRUNCATED_AT_CAPTURE_END",
        "UNKNOWN",
        "NO_ACTIVE_ATTEMPT",
    }
)
PROGRESS_FULL_TASK_OUTCOMES_V1 = frozenset(
    {"SUCCESS", "FAILURE", "UNKNOWN", "NOT_APPLICABLE"}
)

_PAYLOAD_HASH_DOMAIN = "dexteleop.progress_fact_payload.semantic.v1"
_ENVELOPE_HASH_DOMAIN = "dexteleop.progress_fact_envelope.semantic.v1"
_PAYLOAD_SCHEMA_HASH_DOMAIN = "dexteleop.progress_payload_schema.definition.v1"
_FACT_BUNDLE_HASH_DOMAIN = "dexteleop.progress_fact_bundle.semantic.v1"
_MANIFEST_HASH_DOMAIN = "dexteleop.progress_episode_manifest.semantic.v1"


class ProgressFactContractError(ValueError):
    """Raised when a persisted progress fact is incomplete or contradictory."""


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProgressFactContractError(
            f"{name} must be a non-empty string."
        )
    return value.strip()


def _sha256(value: object, *, name: str) -> str:
    result = _required(value, name=name).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ProgressFactContractError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return result


def _finite(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProgressFactContractError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ProgressFactContractError(f"{name} must be finite.")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ProgressFactContractError(f"{name} must be a positive integer.")
    return value


def _optional_finite(value: object | None, *, name: str) -> float | None:
    return None if value is None else _finite(value, name=name)


def _unique_strings(values: Any, *, name: str) -> tuple[str, ...]:
    result = tuple(_required(value, name=name) for value in tuple(values))
    if len(result) != len(set(result)):
        raise ProgressFactContractError(f"{name} entries must be unique.")
    return result


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return canonical_content_hash(payload, domain=_PAYLOAD_HASH_DOMAIN)


def _verify_declared_hash(
    payload: Mapping[str, Any],
    *,
    actual: str,
    name: str,
) -> None:
    declared = payload.get("semantic_content_hash")
    if declared is not None and str(declared) != actual:
        raise ProgressFactContractError(f"{name} semantic hash mismatch.")


class ProgressFactPayload(Protocol):
    schema: str
    semantic_content_hash: str

    def to_payload(self, *, include_hash: bool = True) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class EpisodeProgressStartedV1:
    progress_step_id: str
    attempt_id: str
    attempt_ordinal_in_capture: int = 1
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=EPISODE_PROGRESS_STARTED_SCHEMA_V1, init=False)

    FIELDS: ClassVar[tuple[str, ...]] = (
        "progress_step_id",
        "attempt_id",
        "attempt_ordinal_in_capture",
    )

    def __post_init__(self) -> None:
        for name in ("progress_step_id", "attempt_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "attempt_ordinal_in_capture",
            _positive_int(
                self.attempt_ordinal_in_capture,
                name="attempt_ordinal_in_capture",
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            _payload_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "progress_step_id": self.progress_step_id,
            "attempt_id": self.attempt_id,
            "attempt_ordinal_in_capture": self.attempt_ordinal_in_capture,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeProgressStartedV1":
        result = cls(
            progress_step_id=payload.get("progress_step_id", ""),
            attempt_id=payload.get("attempt_id", ""),
            attempt_ordinal_in_capture=payload.get("attempt_ordinal_in_capture", 0),
        )
        _verify_declared_hash(payload, actual=result.semantic_content_hash, name="start")
        return result


@dataclass(frozen=True)
class ProgressAdvanceEventV1:
    from_progress_step_id: str
    from_attempt_id: str
    to_progress_step_id: str
    to_attempt_id: str
    from_attempt_outcome: str = "SUCCESS"
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=PROGRESS_ADVANCE_EVENT_SCHEMA_V1, init=False)

    FIELDS: ClassVar[tuple[str, ...]] = (
        "from_progress_step_id",
        "from_attempt_id",
        "to_progress_step_id",
        "to_attempt_id",
        "from_attempt_outcome",
    )

    def __post_init__(self) -> None:
        for name in (
            "from_progress_step_id",
            "from_attempt_id",
            "to_progress_step_id",
            "to_attempt_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        outcome = _required(
            self.from_attempt_outcome,
            name="from_attempt_outcome",
        ).upper()
        if outcome != "SUCCESS":
            raise ProgressFactContractError(
                "Progress advance must close the prior Attempt as SUCCESS."
            )
        object.__setattr__(self, "from_attempt_outcome", outcome)
        object.__setattr__(
            self,
            "semantic_content_hash",
            _payload_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "from_progress_step_id": self.from_progress_step_id,
            "from_attempt_id": self.from_attempt_id,
            "to_progress_step_id": self.to_progress_step_id,
            "to_attempt_id": self.to_attempt_id,
            "from_attempt_outcome": self.from_attempt_outcome,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressAdvanceEventV1":
        result = cls(
            from_progress_step_id=payload.get("from_progress_step_id", ""),
            from_attempt_id=payload.get("from_attempt_id", ""),
            to_progress_step_id=payload.get("to_progress_step_id", ""),
            to_attempt_id=payload.get("to_attempt_id", ""),
            from_attempt_outcome=payload.get("from_attempt_outcome", ""),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="advance",
        )
        return result


@dataclass(frozen=True)
class ProgressUndoEventV1:
    advance_fact_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=PROGRESS_UNDO_EVENT_SCHEMA_V1, init=False)

    FIELDS: ClassVar[tuple[str, ...]] = ("advance_fact_id",)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "advance_fact_id",
            _required(self.advance_fact_id, name="advance_fact_id"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            _payload_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "advance_fact_id": self.advance_fact_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressUndoEventV1":
        result = cls(advance_fact_id=payload.get("advance_fact_id", ""))
        _verify_declared_hash(payload, actual=result.semantic_content_hash, name="undo")
        return result


@dataclass(frozen=True)
class AttemptFailureEventV1:
    progress_step_id: str
    attempt_id: str
    reported_at: float
    timing_semantics: str = "OPERATOR_REPORT_TIME"
    requested_operations: tuple[str, ...] = (
        "APPLICATION_HOLD",
        "FORMAL_CAPTURE_PAUSE",
    )
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=ATTEMPT_FAILURE_EVENT_SCHEMA_V1, init=False)

    FIELDS: ClassVar[tuple[str, ...]] = (
        "progress_step_id",
        "attempt_id",
        "reported_at",
        "timing_semantics",
        "requested_operations",
    )

    def __post_init__(self) -> None:
        for name in ("progress_step_id", "attempt_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        object.__setattr__(self, "reported_at", _finite(self.reported_at, name="reported_at"))
        semantics = _required(
            self.timing_semantics,
            name="timing_semantics",
        ).upper()
        if semantics not in FAILURE_TIMING_SEMANTICS_V1:
            raise ProgressFactContractError(
                f"Unsupported failure timing semantics {semantics!r}."
            )
        operations = _unique_strings(
            self.requested_operations,
            name="requested_operations",
        )
        required = ("APPLICATION_HOLD", "FORMAL_CAPTURE_PAUSE")
        if set(operations) != set(required):
            raise ProgressFactContractError(
                "Failure must request application Hold and formal capture pause."
            )
        object.__setattr__(self, "timing_semantics", semantics)
        object.__setattr__(self, "requested_operations", required)
        object.__setattr__(
            self,
            "semantic_content_hash",
            _payload_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "progress_step_id": self.progress_step_id,
            "attempt_id": self.attempt_id,
            "reported_at": self.reported_at,
            "timing_semantics": self.timing_semantics,
            "requested_operations": self.requested_operations,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AttemptFailureEventV1":
        result = cls(
            progress_step_id=payload.get("progress_step_id", ""),
            attempt_id=payload.get("attempt_id", ""),
            reported_at=payload.get("reported_at", float("nan")),
            timing_semantics=payload.get("timing_semantics", ""),
            requested_operations=tuple(payload.get("requested_operations", ())),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="failure",
        )
        return result


@dataclass(frozen=True)
class ProgressOperationReceiptV1:
    """Nested result; it is not an independently persisted progress fact."""

    operation: str
    status: str
    requested_at: float
    clock_domain_id: str
    effective_at: float | None = None

    def __post_init__(self) -> None:
        operation = _required(self.operation, name="operation").upper()
        status = _required(self.status, name="status").upper()
        if operation not in PROGRESS_EFFECT_OPERATIONS_V1:
            raise ProgressFactContractError(f"Unsupported operation {operation!r}.")
        if status not in PROGRESS_EFFECT_STATUSES_V1:
            raise ProgressFactContractError(f"Unsupported effect status {status!r}.")
        requested = _finite(self.requested_at, name="requested_at")
        effective = _optional_finite(self.effective_at, name="effective_at")
        clock = _required(self.clock_domain_id, name="clock_domain_id")
        if status == "APPLIED" and effective is None:
            raise ProgressFactContractError(
                "An APPLIED operation requires effective_at."
            )
        if status != "APPLIED" and effective is not None:
            raise ProgressFactContractError(
                "A non-APPLIED operation cannot claim an effective_at."
            )
        if effective is not None and effective < requested:
            raise ProgressFactContractError(
                "Operation effective_at cannot precede requested_at."
            )
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "requested_at", requested)
        object.__setattr__(self, "effective_at", effective)
        object.__setattr__(self, "clock_domain_id", clock)

    def to_payload(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "status": self.status,
            "requested_at": self.requested_at,
            "effective_at": self.effective_at,
            "clock_domain_id": self.clock_domain_id,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ProgressOperationReceiptV1":
        return cls(
            operation=payload.get("operation", ""),
            status=payload.get("status", ""),
            requested_at=payload.get("requested_at", float("nan")),
            effective_at=payload.get("effective_at"),
            clock_domain_id=payload.get("clock_domain_id", ""),
        )


def _validate_operation(
    receipt: ProgressOperationReceiptV1,
    *,
    expected: str,
) -> None:
    if not isinstance(receipt, ProgressOperationReceiptV1):
        raise TypeError("Operation receipts must be ProgressOperationReceiptV1.")
    if receipt.operation != expected:
        raise ProgressFactContractError(
            f"Expected operation {expected!r}; got {receipt.operation!r}."
        )


@dataclass(frozen=True)
class HoldPauseEffectReceiptV1:
    source_failure_fact_id: str
    application_hold: ProgressOperationReceiptV1
    formal_capture_pause: ProgressOperationReceiptV1
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=HOLD_PAUSE_EFFECT_RECEIPT_SCHEMA_V1, init=False)

    FIELDS: ClassVar[tuple[str, ...]] = (
        "source_failure_fact_id",
        "application_hold",
        "formal_capture_pause",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_failure_fact_id",
            _required(
                self.source_failure_fact_id,
                name="source_failure_fact_id",
            ),
        )
        _validate_operation(self.application_hold, expected="APPLICATION_HOLD")
        _validate_operation(
            self.formal_capture_pause,
            expected="FORMAL_CAPTURE_PAUSE",
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            _payload_hash(self.to_payload(include_hash=False)),
        )

    @property
    def applied(self) -> bool:
        return (
            self.application_hold.status == "APPLIED"
            and self.formal_capture_pause.status == "APPLIED"
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "source_failure_fact_id": self.source_failure_fact_id,
            "application_hold": self.application_hold.to_payload(),
            "formal_capture_pause": self.formal_capture_pause.to_payload(),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "HoldPauseEffectReceiptV1":
        result = cls(
            source_failure_fact_id=payload.get("source_failure_fact_id", ""),
            application_hold=ProgressOperationReceiptV1.from_payload(
                payload.get("application_hold", {})
            ),
            formal_capture_pause=ProgressOperationReceiptV1.from_payload(
                payload.get("formal_capture_pause", {})
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="hold/pause receipt",
        )
        return result


@dataclass(frozen=True)
class ResumeAfterFailureReceiptV1:
    source_failure_fact_id: str
    hold_pause_receipt_fact_id: str
    resume_operation_id: str
    application_resume: ProgressOperationReceiptV1
    formal_capture_resume: ProgressOperationReceiptV1
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=RESUME_AFTER_FAILURE_RECEIPT_SCHEMA_V1, init=False)

    FIELDS: ClassVar[tuple[str, ...]] = (
        "source_failure_fact_id",
        "hold_pause_receipt_fact_id",
        "resume_operation_id",
        "application_resume",
        "formal_capture_resume",
    )

    def __post_init__(self) -> None:
        for name in (
            "source_failure_fact_id",
            "hold_pause_receipt_fact_id",
            "resume_operation_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        _validate_operation(
            self.application_resume,
            expected="APPLICATION_RESUME",
        )
        _validate_operation(
            self.formal_capture_resume,
            expected="FORMAL_CAPTURE_RESUME",
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            _payload_hash(self.to_payload(include_hash=False)),
        )

    @property
    def applied(self) -> bool:
        return (
            self.application_resume.status == "APPLIED"
            and self.formal_capture_resume.status == "APPLIED"
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "source_failure_fact_id": self.source_failure_fact_id,
            "hold_pause_receipt_fact_id": self.hold_pause_receipt_fact_id,
            "resume_operation_id": self.resume_operation_id,
            "application_resume": self.application_resume.to_payload(),
            "formal_capture_resume": self.formal_capture_resume.to_payload(),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ResumeAfterFailureReceiptV1":
        result = cls(
            source_failure_fact_id=payload.get("source_failure_fact_id", ""),
            hold_pause_receipt_fact_id=payload.get(
                "hold_pause_receipt_fact_id",
                "",
            ),
            resume_operation_id=payload.get("resume_operation_id", ""),
            application_resume=ProgressOperationReceiptV1.from_payload(
                payload.get("application_resume", {})
            ),
            formal_capture_resume=ProgressOperationReceiptV1.from_payload(
                payload.get("formal_capture_resume", {})
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="resume receipt",
        )
        return result


@dataclass(frozen=True)
class AttemptRetryEventV1:
    source_failure_fact_id: str
    hold_pause_receipt_fact_id: str
    resume_receipt_fact_id: str
    progress_step_id: str
    new_attempt_id: str
    attempt_ordinal_in_capture: int
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=ATTEMPT_RETRY_EVENT_SCHEMA_V1, init=False)

    FIELDS: ClassVar[tuple[str, ...]] = (
        "source_failure_fact_id",
        "hold_pause_receipt_fact_id",
        "resume_receipt_fact_id",
        "progress_step_id",
        "new_attempt_id",
        "attempt_ordinal_in_capture",
    )

    def __post_init__(self) -> None:
        for name in (
            "source_failure_fact_id",
            "hold_pause_receipt_fact_id",
            "resume_receipt_fact_id",
            "progress_step_id",
            "new_attempt_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "attempt_ordinal_in_capture",
            _positive_int(
                self.attempt_ordinal_in_capture,
                name="attempt_ordinal_in_capture",
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            _payload_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "source_failure_fact_id": self.source_failure_fact_id,
            "hold_pause_receipt_fact_id": self.hold_pause_receipt_fact_id,
            "resume_receipt_fact_id": self.resume_receipt_fact_id,
            "progress_step_id": self.progress_step_id,
            "new_attempt_id": self.new_attempt_id,
            "attempt_ordinal_in_capture": self.attempt_ordinal_in_capture,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AttemptRetryEventV1":
        result = cls(
            source_failure_fact_id=payload.get("source_failure_fact_id", ""),
            hold_pause_receipt_fact_id=payload.get(
                "hold_pause_receipt_fact_id",
                "",
            ),
            resume_receipt_fact_id=payload.get("resume_receipt_fact_id", ""),
            progress_step_id=payload.get("progress_step_id", ""),
            new_attempt_id=payload.get("new_attempt_id", ""),
            attempt_ordinal_in_capture=payload.get("attempt_ordinal_in_capture", 0),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="retry",
        )
        return result


@dataclass(frozen=True)
class EpisodeProgressTerminationV1:
    termination_cause: str
    active_attempt_disposition: str
    full_task_outcome: str
    unresolved_source_failure_fact_ids: tuple[str, ...] = ()
    unresolved_resume_operation_ids: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=EPISODE_PROGRESS_TERMINATION_SCHEMA_V1, init=False)

    FIELDS: ClassVar[tuple[str, ...]] = (
        "termination_cause",
        "active_attempt_disposition",
        "full_task_outcome",
        "unresolved_source_failure_fact_ids",
        "unresolved_resume_operation_ids",
    )

    def __post_init__(self) -> None:
        cause = _required(self.termination_cause, name="termination_cause").upper()
        disposition = _required(
            self.active_attempt_disposition,
            name="active_attempt_disposition",
        ).upper()
        outcome = _required(
            self.full_task_outcome,
            name="full_task_outcome",
        ).upper()
        if cause not in PROGRESS_TERMINATION_CAUSES_V1:
            raise ProgressFactContractError(
                f"Unsupported termination cause {cause!r}."
            )
        if disposition not in PROGRESS_ATTEMPT_DISPOSITIONS_V1:
            raise ProgressFactContractError(
                f"Unsupported attempt disposition {disposition!r}."
            )
        if outcome not in PROGRESS_FULL_TASK_OUTCOMES_V1:
            raise ProgressFactContractError(
                f"Unsupported full-task outcome {outcome!r}."
            )
        object.__setattr__(self, "termination_cause", cause)
        object.__setattr__(self, "active_attempt_disposition", disposition)
        object.__setattr__(self, "full_task_outcome", outcome)
        object.__setattr__(
            self,
            "unresolved_source_failure_fact_ids",
            _unique_strings(
                self.unresolved_source_failure_fact_ids,
                name="unresolved_source_failure_fact_ids",
            ),
        )
        object.__setattr__(
            self,
            "unresolved_resume_operation_ids",
            _unique_strings(
                self.unresolved_resume_operation_ids,
                name="unresolved_resume_operation_ids",
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            _payload_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "termination_cause": self.termination_cause,
            "active_attempt_disposition": self.active_attempt_disposition,
            "full_task_outcome": self.full_task_outcome,
            "unresolved_source_failure_fact_ids": (
                self.unresolved_source_failure_fact_ids
            ),
            "unresolved_resume_operation_ids": (
                self.unresolved_resume_operation_ids
            ),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "EpisodeProgressTerminationV1":
        result = cls(
            termination_cause=payload.get("termination_cause", ""),
            active_attempt_disposition=payload.get(
                "active_attempt_disposition",
                "",
            ),
            full_task_outcome=payload.get("full_task_outcome", ""),
            unresolved_source_failure_fact_ids=tuple(
                payload.get("unresolved_source_failure_fact_ids", ())
            ),
            unresolved_resume_operation_ids=tuple(
                payload.get("unresolved_resume_operation_ids", ())
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="termination",
        )
        return result


ProgressFactPayloadV1: TypeAlias = (
    EpisodeProgressStartedV1
    | ProgressAdvanceEventV1
    | ProgressUndoEventV1
    | AttemptFailureEventV1
    | HoldPauseEffectReceiptV1
    | ResumeAfterFailureReceiptV1
    | AttemptRetryEventV1
    | EpisodeProgressTerminationV1
)

_PAYLOAD_TYPE_BY_SCHEMA: Mapping[str, type[ProgressFactPayloadV1]] = {
    EPISODE_PROGRESS_STARTED_SCHEMA_V1: EpisodeProgressStartedV1,
    PROGRESS_ADVANCE_EVENT_SCHEMA_V1: ProgressAdvanceEventV1,
    PROGRESS_UNDO_EVENT_SCHEMA_V1: ProgressUndoEventV1,
    ATTEMPT_FAILURE_EVENT_SCHEMA_V1: AttemptFailureEventV1,
    HOLD_PAUSE_EFFECT_RECEIPT_SCHEMA_V1: HoldPauseEffectReceiptV1,
    RESUME_AFTER_FAILURE_RECEIPT_SCHEMA_V1: ResumeAfterFailureReceiptV1,
    ATTEMPT_RETRY_EVENT_SCHEMA_V1: AttemptRetryEventV1,
    EPISODE_PROGRESS_TERMINATION_SCHEMA_V1: EpisodeProgressTerminationV1,
}

_PAYLOAD_FIELDS_BY_SCHEMA = {
    schema: payload_type.FIELDS
    for schema, payload_type in _PAYLOAD_TYPE_BY_SCHEMA.items()
}


def progress_payload_schema_hash(payload_schema_id: str) -> str:
    schema = _required(payload_schema_id, name="payload_schema_id")
    fields = _PAYLOAD_FIELDS_BY_SCHEMA.get(schema)
    if fields is None:
        raise ProgressFactContractError(
            f"Unsupported progress payload schema {schema!r}."
        )
    return canonical_content_hash(
        {"schema": schema, "fields": fields},
        domain=_PAYLOAD_SCHEMA_HASH_DOMAIN,
    )


def progress_payload_from_payload(
    payload: Mapping[str, Any],
) -> ProgressFactPayloadV1:
    schema = payload.get("schema")
    payload_type = _PAYLOAD_TYPE_BY_SCHEMA.get(str(schema))
    if payload_type is None:
        raise ProgressFactContractError(
            f"Unsupported progress payload schema {schema!r}."
        )
    return payload_type.from_payload(payload)


@dataclass(frozen=True)
class ProgressFactEnvelopeV1:
    fact_id: str
    episode_capture_identity_hash: str
    progress_sequence_index: int
    effective_at: float
    effective_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    origin: str
    actor_id: str
    payload: ProgressFactPayloadV1
    source_fact_ids: tuple[str, ...] = ()
    payload_schema_id: str = field(init=False)
    payload_schema_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=PROGRESS_FACT_ENVELOPE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _required(self.fact_id, name="fact_id"))
        object.__setattr__(
            self,
            "episode_capture_identity_hash",
            _sha256(
                self.episode_capture_identity_hash,
                name="episode_capture_identity_hash",
            ),
        )
        object.__setattr__(
            self,
            "progress_sequence_index",
            _positive_int(
                self.progress_sequence_index,
                name="progress_sequence_index",
            ),
        )
        effective = _finite(self.effective_at, name="effective_at")
        available = _finite(self.available_at, name="available_at")
        effective_clock = _required(
            self.effective_clock_domain_id,
            name="effective_clock_domain_id",
        )
        available_clock = _required(
            self.available_clock_domain_id,
            name="available_clock_domain_id",
        )
        if effective_clock == available_clock and available < effective:
            raise ProgressFactContractError(
                "available_at cannot precede effective_at in the same clock."
            )
        if not isinstance(self.payload, tuple(_PAYLOAD_TYPE_BY_SCHEMA.values())):
            raise TypeError("payload is not a registered progress fact payload.")
        payload_schema_id = self.payload.schema
        object.__setattr__(self, "effective_at", effective)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "effective_clock_domain_id", effective_clock)
        object.__setattr__(self, "available_clock_domain_id", available_clock)
        object.__setattr__(self, "origin", _required(self.origin, name="origin"))
        object.__setattr__(self, "actor_id", _required(self.actor_id, name="actor_id"))
        object.__setattr__(
            self,
            "source_fact_ids",
            _unique_strings(self.source_fact_ids, name="source_fact_ids"),
        )
        object.__setattr__(self, "payload_schema_id", payload_schema_id)
        object.__setattr__(
            self,
            "payload_schema_hash",
            progress_payload_schema_hash(payload_schema_id),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_ENVELOPE_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "fact_id": self.fact_id,
            "episode_capture_identity_hash": self.episode_capture_identity_hash,
            "progress_sequence_index": self.progress_sequence_index,
            "effective_at": self.effective_at,
            "effective_clock_domain_id": self.effective_clock_domain_id,
            "available_at": self.available_at,
            "available_clock_domain_id": self.available_clock_domain_id,
            "origin": self.origin,
            "actor_id": self.actor_id,
            "payload_schema_id": self.payload_schema_id,
            "payload_schema_hash": self.payload_schema_hash,
            "payload": self.payload.to_payload(),
            "source_fact_ids": self.source_fact_ids,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressFactEnvelopeV1":
        if payload.get("schema") != PROGRESS_FACT_ENVELOPE_SCHEMA_V1:
            raise ProgressFactContractError("Unsupported progress fact envelope.")
        fact_payload_raw = payload.get("payload")
        if not isinstance(fact_payload_raw, Mapping):
            raise ProgressFactContractError("Progress fact payload must be a mapping.")
        fact_payload = progress_payload_from_payload(fact_payload_raw)
        result = cls(
            fact_id=payload.get("fact_id", ""),
            episode_capture_identity_hash=payload.get(
                "episode_capture_identity_hash",
                "",
            ),
            progress_sequence_index=payload.get("progress_sequence_index", 0),
            effective_at=payload.get("effective_at", float("nan")),
            effective_clock_domain_id=payload.get(
                "effective_clock_domain_id",
                "",
            ),
            available_at=payload.get("available_at", float("nan")),
            available_clock_domain_id=payload.get(
                "available_clock_domain_id",
                "",
            ),
            origin=payload.get("origin", ""),
            actor_id=payload.get("actor_id", ""),
            payload=fact_payload,
            source_fact_ids=tuple(payload.get("source_fact_ids", ())),
        )
        if payload.get("payload_schema_id") != result.payload_schema_id:
            raise ProgressFactContractError("Progress payload schema ID mismatch.")
        if payload.get("payload_schema_hash") != result.payload_schema_hash:
            raise ProgressFactContractError("Progress payload schema hash mismatch.")
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="progress fact envelope",
        )
        return result


@dataclass(frozen=True)
class ProgressEpisodeManifestV1:
    episode_capture_identity_hash: str
    task_progress_profile_hash: str
    progress_episode_binding_hash: str
    fact_count: int
    fact_bundle_hash: str
    terminal_fact_id: str
    sealed_at: float
    sealed_clock_domain_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=PROGRESS_EPISODE_MANIFEST_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in (
            "episode_capture_identity_hash",
            "task_progress_profile_hash",
            "progress_episode_binding_hash",
            "fact_bundle_hash",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "fact_count",
            _positive_int(self.fact_count, name="fact_count"),
        )
        object.__setattr__(
            self,
            "terminal_fact_id",
            _required(self.terminal_fact_id, name="terminal_fact_id"),
        )
        object.__setattr__(self, "sealed_at", _finite(self.sealed_at, name="sealed_at"))
        object.__setattr__(
            self,
            "sealed_clock_domain_id",
            _required(
                self.sealed_clock_domain_id,
                name="sealed_clock_domain_id",
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_MANIFEST_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "episode_capture_identity_hash": self.episode_capture_identity_hash,
            "task_progress_profile_hash": self.task_progress_profile_hash,
            "progress_episode_binding_hash": self.progress_episode_binding_hash,
            "fact_count": self.fact_count,
            "fact_bundle_hash": self.fact_bundle_hash,
            "terminal_fact_id": self.terminal_fact_id,
            "sealed_at": self.sealed_at,
            "sealed_clock_domain_id": self.sealed_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressEpisodeManifestV1":
        if payload.get("schema") != PROGRESS_EPISODE_MANIFEST_SCHEMA_V1:
            raise ProgressFactContractError(
                "Unsupported progress episode manifest schema."
            )
        result = cls(
            episode_capture_identity_hash=payload.get(
                "episode_capture_identity_hash",
                "",
            ),
            task_progress_profile_hash=payload.get(
                "task_progress_profile_hash",
                "",
            ),
            progress_episode_binding_hash=payload.get(
                "progress_episode_binding_hash",
                "",
            ),
            fact_count=payload.get("fact_count", 0),
            fact_bundle_hash=payload.get("fact_bundle_hash", ""),
            terminal_fact_id=payload.get("terminal_fact_id", ""),
            sealed_at=payload.get("sealed_at", float("nan")),
            sealed_clock_domain_id=payload.get("sealed_clock_domain_id", ""),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="progress manifest",
        )
        return result

    @staticmethod
    def fact_bundle_hash_for(
        facts: tuple[ProgressFactEnvelopeV1, ...],
    ) -> str:
        return canonical_content_hash(
            tuple(fact.semantic_content_hash for fact in facts),
            domain=_FACT_BUNDLE_HASH_DOMAIN,
        )


__all__ = [
    "ATTEMPT_FAILURE_EVENT_SCHEMA_V1",
    "ATTEMPT_RETRY_EVENT_SCHEMA_V1",
    "EPISODE_PROGRESS_STARTED_SCHEMA_V1",
    "EPISODE_PROGRESS_TERMINATION_SCHEMA_V1",
    "FAILURE_TIMING_SEMANTICS_V1",
    "HOLD_PAUSE_EFFECT_RECEIPT_SCHEMA_V1",
    "PROGRESS_ADVANCE_EVENT_SCHEMA_V1",
    "PROGRESS_ATTEMPT_DISPOSITIONS_V1",
    "PROGRESS_BINDING_FILE_DOMAIN_V1",
    "PROGRESS_EFFECT_OPERATIONS_V1",
    "PROGRESS_EFFECT_STATUSES_V1",
    "PROGRESS_EPISODE_MANIFEST_SCHEMA_V1",
    "PROGRESS_FACT_ENVELOPE_SCHEMA_V1",
    "PROGRESS_FACT_FILE_DOMAIN_V1",
    "PROGRESS_FULL_TASK_OUTCOMES_V1",
    "PROGRESS_MANIFEST_FILE_DOMAIN_V1",
    "PROGRESS_PROFILE_FILE_DOMAIN_V1",
    "PROGRESS_TERMINATION_CAUSES_V1",
    "PROGRESS_UNDO_EVENT_SCHEMA_V1",
    "RESUME_AFTER_FAILURE_RECEIPT_SCHEMA_V1",
    "AttemptFailureEventV1",
    "AttemptRetryEventV1",
    "EpisodeProgressStartedV1",
    "EpisodeProgressTerminationV1",
    "HoldPauseEffectReceiptV1",
    "ProgressAdvanceEventV1",
    "ProgressEpisodeManifestV1",
    "ProgressFactContractError",
    "ProgressFactEnvelopeV1",
    "ProgressFactPayloadV1",
    "ProgressOperationReceiptV1",
    "ProgressUndoEventV1",
    "ResumeAfterFailureReceiptV1",
    "progress_payload_from_payload",
    "progress_payload_schema_hash",
]
