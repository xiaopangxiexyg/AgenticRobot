"""Typed final-call wrapper for dispatch-permit protected backend seams.

This module is the M5.8-a *pure* integration seam.  It does not install a
registry on any real backend and it does not mint permits.  A caller must
provide an already-issued permit and the live authority/execution epochs.

The wrapped callback is invoked immediately after the owner registry accepts
the one-shot permit.  Permit-kind/operation mismatches and owner/epoch
rejections are represented by a typed receipt and never reach the callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from dataset_contracts.canonical_encoding import canonical_content_hash

from .dispatch_owner import (
    DISPATCH_PERMIT_KINDS,
    DispatchOwnerRegistry,
    DispatchPermit,
    DispatchPermitValidationReceipt,
)


DISPATCH_SEAM_SPEC_HASH_DOMAIN_V1 = "dexteleop.dispatch_seam_spec.v1"
DISPATCH_SEAM_VALIDATION_HASH_DOMAIN_V1 = "dexteleop.dispatch_seam_validation.v1"
DISPATCH_SEAM_STAGES = (
    "logical_backend_call",
    "worker_dequeue",
    "pre_sdk_call",
)

_T = TypeVar("_T")
_P = TypeVar("_P")


def _identifier(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class DispatchSeamSpec:
    """Expected permit semantics at one concrete backend-call seam."""

    seam_id: str
    permit_kind: str
    operation: str | None
    call_stage: str
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        seam_id: str,
        permit_kind: str,
        operation: str | None = None,
        call_stage: str = "logical_backend_call",
    ) -> "DispatchSeamSpec":
        normalized_kind = str(permit_kind).strip()
        normalized_operation = (
            None if operation is None else _identifier(operation, name="operation")
        )
        normalized_stage = str(call_stage).strip()
        if normalized_kind not in DISPATCH_PERMIT_KINDS:
            raise ValueError("Unsupported dispatch seam permit kind.")
        if normalized_stage not in DISPATCH_SEAM_STAGES:
            raise ValueError("Unsupported dispatch seam call stage.")
        if normalized_kind == "motion" and normalized_operation is not None:
            raise ValueError("Motion permits do not carry an operation field.")
        if normalized_kind != "motion" and normalized_operation is None:
            raise ValueError("Safety/lifecycle seams require an exact operation.")
        payload = {
            "seam_id": _identifier(seam_id, name="seam_id"),
            "permit_kind": normalized_kind,
            "operation": normalized_operation,
            "call_stage": normalized_stage,
        }
        return cls(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=DISPATCH_SEAM_SPEC_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class DispatchSeamValidationReceipt:
    seam_id: str
    seam_spec_hash: str
    permit_id: str
    expected_permit_kind: str
    actual_permit_kind: str
    expected_operation: str | None
    actual_operation: str | None
    accepted: bool
    status: str
    owner_validation_receipt_id: str | None
    owner_validation_hash: str | None
    semantic_content_hash: str


@dataclass(frozen=True)
class PermittedDispatchResult(Generic[_T]):
    result: _T
    validation: DispatchSeamValidationReceipt
    owner_validation: DispatchPermitValidationReceipt


@dataclass(frozen=True)
class DispatchPermitInvocation:
    """One already-issued permit plus the live epochs for one final call."""

    permit: DispatchPermit
    current_authority_epoch: int
    current_execution_epoch: int


class PermittedDispatchAdapter(Generic[_P, _T]):
    """Preserve a one-argument legacy callback while adding a final gate.

    The provider runs before authorization and must mint/select a *new* permit
    for this concrete call.  This makes the same adapter usable at a direct
    backend seam or at each tick of an asynchronous worker; it deliberately
    does not cache or reuse an upstream keyframe permit.
    """

    def __init__(
        self,
        *,
        gate: "DispatchSeamGate",
        invocation_provider: Callable[[_P], DispatchPermitInvocation],
        backend_call: Callable[[_P], _T],
    ) -> None:
        if not isinstance(gate, DispatchSeamGate):
            raise TypeError("gate must be a DispatchSeamGate.")
        if not callable(invocation_provider):
            raise TypeError("invocation_provider must be callable.")
        if not callable(backend_call):
            raise TypeError("backend_call must be callable.")
        self.gate = gate
        self.invocation_provider = invocation_provider
        self.backend_call = backend_call

    def invoke(self, payload: _P) -> PermittedDispatchResult[_T]:
        invocation = self.invocation_provider(payload)
        if not isinstance(invocation, DispatchPermitInvocation):
            raise TypeError(
                "invocation_provider must return DispatchPermitInvocation."
            )
        return self.gate.invoke(
            invocation.permit,
            current_authority_epoch=invocation.current_authority_epoch,
            current_execution_epoch=invocation.current_execution_epoch,
            backend_call=lambda: self.backend_call(payload),
        )

    def __call__(self, payload: _P) -> _T:
        """Compatibility form: return exactly the wrapped callback result."""

        return self.invoke(payload).result


class DispatchSeamRejectedError(RuntimeError):
    """Raised before a backend callback when its typed permit is rejected."""

    def __init__(self, receipt: DispatchSeamValidationReceipt) -> None:
        self.receipt = receipt
        super().__init__(
            f"Dispatch seam {receipt.seam_id!r} rejected permit "
            f"{receipt.permit_id!r}: {receipt.status}"
        )


class DispatchSeamGate:
    """Validate one permit and invoke one already-bound backend callback."""

    def __init__(
        self,
        *,
        registry: DispatchOwnerRegistry,
        spec: DispatchSeamSpec,
    ) -> None:
        if not isinstance(registry, DispatchOwnerRegistry):
            raise TypeError("registry must be a DispatchOwnerRegistry.")
        if not isinstance(spec, DispatchSeamSpec):
            raise TypeError("spec must be a DispatchSeamSpec.")
        self.registry = registry
        self.spec = spec

    def invoke(
        self,
        permit: DispatchPermit,
        *,
        current_authority_epoch: int,
        current_execution_epoch: int,
        backend_call: Callable[[], _T],
    ) -> PermittedDispatchResult[_T]:
        if not callable(backend_call):
            raise TypeError("backend_call must be callable.")

        permit_id = str(getattr(permit, "permit_id", "invalid-permit"))
        actual_kind = str(getattr(permit, "permit_kind", "unknown"))
        actual_operation_value = getattr(permit, "operation", None)
        actual_operation = (
            None
            if actual_operation_value is None
            else str(actual_operation_value).strip()
        )
        if actual_kind != self.spec.permit_kind:
            receipt = self._receipt(
                permit_id=permit_id,
                actual_permit_kind=actual_kind,
                actual_operation=actual_operation,
                status="permit_kind_mismatch",
                owner_validation=None,
            )
            raise DispatchSeamRejectedError(receipt)
        if (
            self.spec.operation is not None
            and actual_operation != self.spec.operation
        ):
            receipt = self._receipt(
                permit_id=permit_id,
                actual_permit_kind=actual_kind,
                actual_operation=actual_operation,
                status="operation_mismatch",
                owner_validation=None,
            )
            raise DispatchSeamRejectedError(receipt)

        owner_validation = self.registry.authorize(
            permit,
            current_authority_epoch=current_authority_epoch,
            current_execution_epoch=current_execution_epoch,
        )
        if not owner_validation.accepted:
            receipt = self._receipt(
                permit_id=permit_id,
                actual_permit_kind=actual_kind,
                actual_operation=actual_operation,
                status=owner_validation.status,
                owner_validation=owner_validation,
            )
            raise DispatchSeamRejectedError(receipt)

        # Keep the accepted permit's linearization point adjacent to the real
        # call.  Trace projection and other mutable work belong after return.
        result = backend_call()
        receipt = self._receipt(
            permit_id=permit_id,
            actual_permit_kind=actual_kind,
            actual_operation=actual_operation,
            status="accepted",
            owner_validation=owner_validation,
        )
        return PermittedDispatchResult(
            result=result,
            validation=receipt,
            owner_validation=owner_validation,
        )

    def _receipt(
        self,
        *,
        permit_id: str,
        actual_permit_kind: str,
        actual_operation: str | None,
        status: str,
        owner_validation: DispatchPermitValidationReceipt | None,
    ) -> DispatchSeamValidationReceipt:
        payload = {
            "seam_id": self.spec.seam_id,
            "seam_spec_hash": self.spec.semantic_content_hash,
            "permit_id": permit_id,
            "expected_permit_kind": self.spec.permit_kind,
            "actual_permit_kind": actual_permit_kind,
            "expected_operation": self.spec.operation,
            "actual_operation": actual_operation,
            "accepted": status == "accepted",
            "status": str(status),
            "owner_validation_receipt_id": (
                None if owner_validation is None else owner_validation.receipt_id
            ),
            "owner_validation_hash": (
                None
                if owner_validation is None
                else owner_validation.semantic_content_hash
            ),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=DISPATCH_SEAM_VALIDATION_HASH_DOMAIN_V1,
        )
        return DispatchSeamValidationReceipt(
            **payload,
            semantic_content_hash=semantic_hash,
        )


__all__ = [
    "DISPATCH_SEAM_STAGES",
    "DispatchSeamGate",
    "DispatchSeamRejectedError",
    "DispatchSeamSpec",
    "DispatchSeamValidationReceipt",
    "DispatchPermitInvocation",
    "PermittedDispatchAdapter",
    "PermittedDispatchResult",
]
