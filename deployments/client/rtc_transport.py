"""Model- and protocol-family-neutral RTC transport deadline facts.

The types in this module describe client-side operation budgets only.  They do
not describe a model, checkpoint, action representation, prediction horizon,
transport protocol, worker drain, or motion authority.  A concrete transport
adapter may map these budgets to websocket, HTTP, IPC, or local-runtime calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash


RTC_TRANSPORT_DEADLINE_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.rtc_transport_deadline_spec.v1"
)
RTC_TRANSPORT_PHASE_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.rtc_transport_phase_receipt.v1"
)

RTC_TRANSPORT_PHASE_CONNECT = "connect"
RTC_TRANSPORT_PHASE_METADATA = "metadata"
RTC_TRANSPORT_PHASE_INFERENCE = "inference"
RTC_TRANSPORT_PHASE_CLOSE = "close"
RTC_TRANSPORT_PHASES = (
    RTC_TRANSPORT_PHASE_CONNECT,
    RTC_TRANSPORT_PHASE_METADATA,
    RTC_TRANSPORT_PHASE_INFERENCE,
    RTC_TRANSPORT_PHASE_CLOSE,
)

RTC_TRANSPORT_PHASE_COMPLETED = "completed"
RTC_TRANSPORT_PHASE_TIMED_OUT = "timed_out"
RTC_TRANSPORT_PHASE_CANCELLED = "cancelled"
RTC_TRANSPORT_PHASE_FAILED = "failed"
RTC_TRANSPORT_PHASE_STATUSES = (
    RTC_TRANSPORT_PHASE_COMPLETED,
    RTC_TRANSPORT_PHASE_TIMED_OUT,
    RTC_TRANSPORT_PHASE_CANCELLED,
    RTC_TRANSPORT_PHASE_FAILED,
)


def _optional_positive_finite(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive when provided.")
    return normalized


@dataclass(frozen=True)
class RtcTransportDeadlineSpec:
    """Optional per-phase wall-clock budgets for one client transport.

    ``None`` preserves the concrete transport's legacy behavior for that
    phase.  Values are relative durations measured on a local monotonic clock;
    they are deliberately absent from model capability negotiation.
    """

    connect_timeout_sec: float | None = None
    metadata_timeout_sec: float | None = None
    inference_timeout_sec: float | None = None
    close_timeout_sec: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "connect_timeout_sec",
            "metadata_timeout_sec",
            "inference_timeout_sec",
            "close_timeout_sec",
        ):
            object.__setattr__(
                self,
                name,
                _optional_positive_finite(getattr(self, name), name=name),
            )

    @property
    def enabled(self) -> bool:
        return any(
            getattr(self, name) is not None
            for name in (
                "connect_timeout_sec",
                "metadata_timeout_sec",
                "inference_timeout_sec",
                "close_timeout_sec",
            )
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "connect_timeout_sec": self.connect_timeout_sec,
                "metadata_timeout_sec": self.metadata_timeout_sec,
                "inference_timeout_sec": self.inference_timeout_sec,
                "close_timeout_sec": self.close_timeout_sec,
            },
            domain=RTC_TRANSPORT_DEADLINE_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class RtcTransportPhaseReceipt:
    """Completion fact for one concrete transport operation phase.

    Session and connection generations are transport-local correlation facts.
    They do not prove that an inference worker or execution participant drained.
    """

    sequence: int
    phase: str
    status: str
    started_at: float
    completed_at: float
    timeout_sec: float | None
    session_generation: int
    connection_generation: int
    failure_type: str | None = None
    failure_detail: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in RTC_TRANSPORT_PHASES:
            raise ValueError(f"Unsupported RTC transport phase: {self.phase!r}.")
        if self.status not in RTC_TRANSPORT_PHASE_STATUSES:
            raise ValueError(
                f"Unsupported RTC transport phase status: {self.status!r}."
            )
        for name in ("sequence", "session_generation", "connection_generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in ("started_at", "completed_at"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.completed_at < self.started_at:
            raise ValueError("Transport phase completion cannot precede its start.")
        object.__setattr__(
            self,
            "timeout_sec",
            _optional_positive_finite(self.timeout_sec, name="timeout_sec"),
        )
        if self.status == RTC_TRANSPORT_PHASE_TIMED_OUT and self.timeout_sec is None:
            raise ValueError("Timed-out transport phase requires timeout_sec.")
        failed = self.status in (
            RTC_TRANSPORT_PHASE_TIMED_OUT,
            RTC_TRANSPORT_PHASE_CANCELLED,
            RTC_TRANSPORT_PHASE_FAILED,
        )
        if failed:
            if not self.failure_type or not self.failure_detail:
                raise ValueError("Non-completed transport phase requires failure details.")
        elif self.failure_type is not None or self.failure_detail is not None:
            raise ValueError("Completed transport phase cannot carry a failure.")

    @property
    def duration_sec(self) -> float:
        return self.completed_at - self.started_at

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "sequence": self.sequence,
                "phase": self.phase,
                "status": self.status,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "timeout_sec": self.timeout_sec,
                "session_generation": self.session_generation,
                "connection_generation": self.connection_generation,
                "failure_type": self.failure_type,
                "failure_detail": self.failure_detail,
            },
            domain=RTC_TRANSPORT_PHASE_RECEIPT_HASH_DOMAIN_V1,
        )


class RtcTransportDeadlineExceeded(TimeoutError):
    """A concrete transport phase exceeded an explicitly enabled budget."""

    def __init__(self, receipt: RtcTransportPhaseReceipt) -> None:
        if receipt.status != RTC_TRANSPORT_PHASE_TIMED_OUT:
            raise ValueError("Deadline exception requires a timed-out receipt.")
        assert receipt.timeout_sec is not None
        self.receipt = receipt
        super().__init__(
            f"RTC transport {receipt.phase} exceeded its "
            f"{receipt.timeout_sec:.6f}s deadline."
        )


__all__ = [
    "RTC_TRANSPORT_DEADLINE_SPEC_HASH_DOMAIN_V1",
    "RTC_TRANSPORT_PHASE_CANCELLED",
    "RTC_TRANSPORT_PHASE_CLOSE",
    "RTC_TRANSPORT_PHASE_COMPLETED",
    "RTC_TRANSPORT_PHASE_CONNECT",
    "RTC_TRANSPORT_PHASE_FAILED",
    "RTC_TRANSPORT_PHASE_INFERENCE",
    "RTC_TRANSPORT_PHASE_METADATA",
    "RTC_TRANSPORT_PHASE_RECEIPT_HASH_DOMAIN_V1",
    "RTC_TRANSPORT_PHASE_STATUSES",
    "RTC_TRANSPORT_PHASE_TIMED_OUT",
    "RTC_TRANSPORT_PHASES",
    "RtcTransportDeadlineExceeded",
    "RtcTransportDeadlineSpec",
    "RtcTransportPhaseReceipt",
]
