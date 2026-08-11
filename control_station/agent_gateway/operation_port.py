"""Typed operation-port receipts plus strict B2a fake/null implementations.

Authoritative receipts only reference evidence already produced by the lower
owner.  Fake/null receipts remain application-test evidence and can never
claim a lower HOLD, motion, dispatch, or device/simulator contact.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash

from .contracts import AgentSkillOperation, AgentSkillRequest

AGENT_GATEWAY_PORT_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_gateway.no_io_port_receipt.v1"
)
AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_gateway.authoritative_port_receipt.v1"
)
AGENT_GATEWAY_PORT_OPERATION_KINDS_V1 = ("activate", "cancel", "hold")


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _finite_time(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


@dataclass(frozen=True)
class AgentGatewayNoIoPortReceipt:
    """Non-authoritative receipt from one deterministic fake/null call."""

    port_id: str
    operation_kind: str
    operation_id: str | None
    succeeded: bool
    status: str
    completed_at: float
    call_sequence: int
    lower_operation_id: str | None = None
    evidence_refs: tuple[str, ...] = ()
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "port_id",
            _required_text(self.port_id, field_name="port_id"),
        )
        kind = _required_text(self.operation_kind, field_name="operation_kind")
        if kind not in AGENT_GATEWAY_PORT_OPERATION_KINDS_V1:
            raise ValueError(f"Unsupported operation_kind: {kind!r}.")
        object.__setattr__(self, "operation_kind", kind)
        if self.operation_id is not None:
            object.__setattr__(
                self,
                "operation_id",
                _required_text(self.operation_id, field_name="operation_id"),
            )
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        object.__setattr__(
            self,
            "status",
            _required_text(self.status, field_name="status"),
        )
        object.__setattr__(
            self,
            "completed_at",
            _finite_time(self.completed_at, field_name="completed_at"),
        )
        if (
            isinstance(self.call_sequence, bool)
            or not isinstance(self.call_sequence, int)
            or self.call_sequence < 0
        ):
            raise ValueError("call_sequence must be a non-negative integer.")
        if self.lower_operation_id is not None:
            object.__setattr__(
                self,
                "lower_operation_id",
                _required_text(
                    self.lower_operation_id,
                    field_name="lower_operation_id",
                ),
            )
        normalized_refs = tuple(
            sorted(
                {
                    _required_text(reference, field_name="evidence_refs item")
                    for reference in self.evidence_refs
                }
            )
        )
        object.__setattr__(self, "evidence_refs", normalized_refs)
        object.__setattr__(
            self,
            "receipt_id",
            "agent-gateway-no-io-receipt:" + self.semantic_content_hash,
        )

    @property
    def authoritative_lower_receipt(self) -> bool:
        return False

    @property
    def proves_motion_or_hold(self) -> bool:
        return False

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self._semantic_payload(),
            domain=AGENT_GATEWAY_PORT_RECEIPT_HASH_DOMAIN_V1,
        )

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "port_id": self.port_id,
            "operation_kind": self.operation_kind,
            "operation_id": self.operation_id,
            "succeeded": self.succeeded,
            "status": self.status,
            "completed_at": self.completed_at,
            "call_sequence": self.call_sequence,
            "lower_operation_id": self.lower_operation_id,
            "evidence_refs": self.evidence_refs,
            "evidence_scope": "fake_or_null_application_only",
            "authoritative_lower_receipt": False,
            "proves_motion_or_hold": False,
        }

    def to_wire(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            **self._semantic_payload(),
            "receipt_hash": self.semantic_content_hash,
        }


@runtime_checkable
class AgentSkillOperationPortReceipt(Protocol):
    """Common receipt projection accepted by the application service."""

    port_id: str
    operation_kind: str
    operation_id: str | None
    succeeded: bool
    status: str
    completed_at: float
    call_sequence: int
    lower_operation_id: str | None
    evidence_refs: tuple[str, ...]
    receipt_id: str

    @property
    def authoritative_lower_receipt(self) -> bool: ...

    @property
    def proves_motion_or_hold(self) -> bool: ...


@dataclass(frozen=True)
class AgentGatewayAuthoritativePortReceipt:
    """Application receipt that references already-produced lower evidence.

    Activation is a start acknowledgement only and cannot prove execution
    completion.  A successful hold receipt must prove the lower HOLD boundary.
    """

    port_id: str
    operation_kind: str
    operation_id: str | None
    succeeded: bool
    status: str
    completed_at: float
    call_sequence: int
    lower_operation_id: str | None = None
    evidence_refs: tuple[str, ...] = ()
    proves_motion_or_hold: bool = False
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "port_id",
            _required_text(self.port_id, field_name="port_id"),
        )
        kind = _required_text(self.operation_kind, field_name="operation_kind")
        if kind not in AGENT_GATEWAY_PORT_OPERATION_KINDS_V1:
            raise ValueError(f"Unsupported operation_kind: {kind!r}.")
        object.__setattr__(self, "operation_kind", kind)
        if self.operation_id is not None:
            object.__setattr__(
                self,
                "operation_id",
                _required_text(self.operation_id, field_name="operation_id"),
            )
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        object.__setattr__(
            self,
            "status",
            _required_text(self.status, field_name="status"),
        )
        object.__setattr__(
            self,
            "completed_at",
            _finite_time(self.completed_at, field_name="completed_at"),
        )
        if (
            isinstance(self.call_sequence, bool)
            or not isinstance(self.call_sequence, int)
            or self.call_sequence < 0
        ):
            raise ValueError("call_sequence must be a non-negative integer.")
        if self.lower_operation_id is not None:
            object.__setattr__(
                self,
                "lower_operation_id",
                _required_text(
                    self.lower_operation_id,
                    field_name="lower_operation_id",
                ),
            )
        refs = tuple(
            sorted(
                {
                    _required_text(item, field_name="evidence_refs item")
                    for item in self.evidence_refs
                }
            )
        )
        object.__setattr__(self, "evidence_refs", refs)
        if not isinstance(self.proves_motion_or_hold, bool):
            raise TypeError("proves_motion_or_hold must be a boolean.")
        if self.succeeded and not refs:
            raise ValueError(
                "A successful authoritative receipt requires lower evidence refs."
            )
        if self.succeeded and kind == "activate" and self.lower_operation_id is None:
            raise ValueError(
                "A successful activation requires a lower_operation_id."
            )
        if self.succeeded and kind == "hold" and not self.proves_motion_or_hold:
            raise ValueError(
                "A successful authoritative HOLD receipt must prove lower HOLD."
            )
        object.__setattr__(
            self,
            "receipt_id",
            "agent-gateway-authoritative-receipt:"
            + self.semantic_content_hash,
        )

    @property
    def authoritative_lower_receipt(self) -> bool:
        return True

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self._semantic_payload(),
            domain=AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_HASH_DOMAIN_V1,
        )

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "port_id": self.port_id,
            "operation_kind": self.operation_kind,
            "operation_id": self.operation_id,
            "succeeded": self.succeeded,
            "status": self.status,
            "completed_at": self.completed_at,
            "call_sequence": self.call_sequence,
            "lower_operation_id": self.lower_operation_id,
            "evidence_refs": self.evidence_refs,
            "evidence_scope": "authoritative_lower_reference",
            "authoritative_lower_receipt": True,
            "proves_motion_or_hold": self.proves_motion_or_hold,
        }

    def to_wire(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            **self._semantic_payload(),
            "receipt_hash": self.semantic_content_hash,
        }


@runtime_checkable
class AgentSkillExecutionPort(Protocol):
    """Route-selected start/cancel seam; operation bodies remain executor-owned."""

    port_id: str
    no_io: bool

    def activate(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> AgentSkillOperationPortReceipt: ...

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> AgentSkillOperationPortReceipt: ...


@runtime_checkable
class AgentDomainControlPort(Protocol):
    """Sole control-domain HOLD seam, independent of Route selection."""

    port_id: str
    no_io: bool

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> AgentSkillOperationPortReceipt: ...


@runtime_checkable
class AgentSkillOperationPort(
    AgentSkillExecutionPort,
    AgentDomainControlPort,
    Protocol,
):
    """Legacy combined execution and control-domain port contract."""


@runtime_checkable
class NoIoAgentSkillOperationPort(AgentSkillOperationPort, Protocol):
    """B2a-only subtype whose receipts cannot claim lower authority."""


class _NoIoPortBase:
    no_io = True

    def __init__(
        self,
        *,
        port_id: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.port_id = _required_text(port_id, field_name="port_id")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._clock = clock
        self._lock = threading.Lock()
        self._next_call_sequence = 0
        self._receipts: list[AgentGatewayNoIoPortReceipt] = []

    @property
    def receipts(self) -> tuple[AgentGatewayNoIoPortReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    def call_count(self, operation_kind: str) -> int:
        normalized_kind = _required_text(
            operation_kind,
            field_name="operation_kind",
        )
        with self._lock:
            return sum(
                receipt.operation_kind == normalized_kind
                for receipt in self._receipts
            )

    def _record(
        self,
        *,
        operation_kind: str,
        operation_id: str | None,
        succeeded: bool,
        status: str,
        lower_operation_id: str | None = None,
        evidence_refs: tuple[str, ...] = (),
    ) -> AgentGatewayNoIoPortReceipt:
        with self._lock:
            sequence = self._next_call_sequence
            self._next_call_sequence += 1
            completed_at = _finite_time(self._clock(), field_name="clock result")
            receipt = AgentGatewayNoIoPortReceipt(
                port_id=self.port_id,
                operation_kind=operation_kind,
                operation_id=operation_id,
                succeeded=succeeded,
                status=status,
                completed_at=completed_at,
                call_sequence=sequence,
                lower_operation_id=lower_operation_id,
                evidence_refs=evidence_refs,
            )
            self._receipts.append(receipt)
            return receipt

    @staticmethod
    def _validate_binding(
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> None:
        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        if not isinstance(operation, AgentSkillOperation):
            raise TypeError("operation must be an AgentSkillOperation.")
        if request.semantic_content_hash != operation.request_hash:
            raise ValueError("request_hash does not match operation projection.")


class NullAgentSkillOperationPort(_NoIoPortBase):
    """Fail-closed placeholder that performs no operation or safety action."""

    def __init__(
        self,
        *,
        port_id: str = "agent-gateway:null-port",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(port_id=port_id, clock=clock)

    def activate(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> AgentGatewayNoIoPortReceipt:
        self._validate_binding(request=request, operation=operation)
        return self._record(
            operation_kind="activate",
            operation_id=operation.operation_id,
            succeeded=False,
            status="null_port_has_no_operation_binding",
        )

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> AgentGatewayNoIoPortReceipt:
        self._validate_binding(request=request, operation=operation)
        _required_text(reason, field_name="reason")
        return self._record(
            operation_kind="cancel",
            operation_id=operation.operation_id,
            succeeded=False,
            status="null_port_cannot_confirm_cancel",
        )

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> AgentGatewayNoIoPortReceipt:
        if operation is not None and not isinstance(operation, AgentSkillOperation):
            raise TypeError("operation must be an AgentSkillOperation or None.")
        _required_text(reason, field_name="reason")
        return self._record(
            operation_kind="hold",
            operation_id=None if operation is None else operation.operation_id,
            succeeded=False,
            status="null_port_cannot_confirm_hold",
        )


class DeterministicNoIoAgentSkillOperationPort(_NoIoPortBase):
    """Configurable fake port for deterministic concurrency and failure tests."""

    def __init__(
        self,
        *,
        port_id: str = "agent-gateway:fake-no-io-port",
        succeeds_by_kind: Mapping[str, bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(port_id=port_id, clock=clock)
        configured = dict(succeeds_by_kind or {})
        unknown = sorted(set(configured) - set(AGENT_GATEWAY_PORT_OPERATION_KINDS_V1))
        if unknown:
            raise ValueError(f"Unknown fake port operation kinds: {unknown}.")
        if any(not isinstance(value, bool) for value in configured.values()):
            raise TypeError("succeeds_by_kind values must be booleans.")
        self._succeeds_by_kind = {
            kind: configured.get(kind, True)
            for kind in AGENT_GATEWAY_PORT_OPERATION_KINDS_V1
        }

    def activate(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> AgentGatewayNoIoPortReceipt:
        self._validate_binding(request=request, operation=operation)
        succeeded = self._succeeds_by_kind["activate"]
        lower_operation_id = None
        if succeeded:
            lower_operation_id = "fake-lower-operation:" + canonical_content_hash(
                {
                    "port_id": self.port_id,
                    "operation_id": operation.operation_id,
                },
                domain="dexteleop.agent_gateway.fake_lower_operation_id.v1",
            )
        return self._record(
            operation_kind="activate",
            operation_id=operation.operation_id,
            succeeded=succeeded,
            status=(
                "fake_activation_accepted_not_completed"
                if succeeded
                else "fake_activation_injected_failure"
            ),
            lower_operation_id=lower_operation_id,
            evidence_refs=(f"fake-no-io-port:{self.port_id}:activate",),
        )

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> AgentGatewayNoIoPortReceipt:
        self._validate_binding(request=request, operation=operation)
        _required_text(reason, field_name="reason")
        succeeded = self._succeeds_by_kind["cancel"]
        return self._record(
            operation_kind="cancel",
            operation_id=operation.operation_id,
            succeeded=succeeded,
            status=(
                "fake_cancel_confirmed"
                if succeeded
                else "fake_cancel_injected_failure"
            ),
            lower_operation_id=operation.lower_operation_id,
            evidence_refs=(f"fake-no-io-port:{self.port_id}:cancel",),
        )

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> AgentGatewayNoIoPortReceipt:
        if operation is not None and not isinstance(operation, AgentSkillOperation):
            raise TypeError("operation must be an AgentSkillOperation or None.")
        _required_text(reason, field_name="reason")
        succeeded = self._succeeds_by_kind["hold"]
        return self._record(
            operation_kind="hold",
            operation_id=None if operation is None else operation.operation_id,
            succeeded=succeeded,
            status=(
                "fake_hold_confirmed"
                if succeeded
                else "fake_hold_injected_failure"
            ),
            lower_operation_id=(
                None if operation is None else operation.lower_operation_id
            ),
            evidence_refs=(f"fake-no-io-port:{self.port_id}:hold",),
        )


__all__ = [
    "AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_HASH_DOMAIN_V1",
    "AGENT_GATEWAY_PORT_OPERATION_KINDS_V1",
    "AGENT_GATEWAY_PORT_RECEIPT_HASH_DOMAIN_V1",
    "AgentDomainControlPort",
    "AgentGatewayAuthoritativePortReceipt",
    "AgentGatewayNoIoPortReceipt",
    "AgentSkillExecutionPort",
    "AgentSkillOperationPort",
    "AgentSkillOperationPortReceipt",
    "DeterministicNoIoAgentSkillOperationPort",
    "NoIoAgentSkillOperationPort",
    "NullAgentSkillOperationPort",
]
