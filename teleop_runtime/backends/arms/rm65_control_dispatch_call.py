"""Default-off typed mediation for RM65 safety/lifecycle SDK calls.

Unlike joint targets, these calls do not carry a qpos payload.  They still
need exact per-call semantics because one method may issue multiple SDK writes
and every write requires its own one-shot permit.  This module supplies the
typed context and a permit-gated adapter, but does not bind either to a
production frontend or authorize hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.dispatch_owner import DispatchOwnerRegistry
from robot_motion.runtime.dispatch_seam import (
    DispatchPermitInvocation,
    DispatchSeamGate,
    DispatchSeamSpec,
    PermittedDispatchResult,
)


RM65_CONTROL_CALL_HASH_DOMAIN_V1 = "dexteleop.rm65_control_call.v1"


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class RM65ControlDispatchContext:
    side: str
    seam_id: str
    permit_kind: str
    operation: str
    source: str
    sdk_method: str
    attempt_sequence: int
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        side: str,
        seam_id: str,
        permit_kind: str,
        operation: str,
        source: str,
        sdk_method: str,
        attempt_sequence: int = 1,
    ) -> "RM65ControlDispatchContext":
        normalized_kind = _identifier(permit_kind, name="permit_kind")
        if normalized_kind not in {"safety_control", "lifecycle_control"}:
            raise ValueError("RM65 control calls require safety/lifecycle permits.")
        if isinstance(attempt_sequence, bool) or int(attempt_sequence) != attempt_sequence:
            raise ValueError("attempt_sequence must be an integer.")
        if int(attempt_sequence) <= 0:
            raise ValueError("attempt_sequence must be positive.")
        payload = {
            "side": _identifier(side, name="side"),
            "seam_id": _identifier(seam_id, name="seam_id"),
            "permit_kind": normalized_kind,
            "operation": _identifier(operation, name="operation"),
            "source": _identifier(source, name="source"),
            "sdk_method": _identifier(sdk_method, name="sdk_method"),
            "attempt_sequence": int(attempt_sequence),
        }
        return cls(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=RM65_CONTROL_CALL_HASH_DOMAIN_V1,
            ),
        )


class RM65PreSdkControlAdapter(Protocol):
    def invoke(
        self,
        *,
        context: RM65ControlDispatchContext,
        backend_call: Callable[[], object],
    ) -> object: ...


class RM65PermittedPreSdkControlAdapter:
    """Consume one provider-supplied permit for each exact SDK call."""

    def __init__(
        self,
        *,
        registry: DispatchOwnerRegistry,
        invocation_provider: Callable[
            [RM65ControlDispatchContext], DispatchPermitInvocation
        ],
        receipt_sink: Callable[
            [RM65ControlDispatchContext, PermittedDispatchResult[object]], None
        ]
        | None = None,
    ) -> None:
        if not isinstance(registry, DispatchOwnerRegistry):
            raise TypeError("registry must be a DispatchOwnerRegistry.")
        if not callable(invocation_provider):
            raise TypeError("invocation_provider must be callable.")
        if receipt_sink is not None and not callable(receipt_sink):
            raise TypeError("receipt_sink must be callable when provided.")
        self.registry = registry
        self.invocation_provider = invocation_provider
        self.receipt_sink = receipt_sink

    def invoke(
        self,
        *,
        context: RM65ControlDispatchContext,
        backend_call: Callable[[], object],
    ) -> object:
        if not isinstance(context, RM65ControlDispatchContext):
            raise TypeError("context must be an RM65ControlDispatchContext.")
        invocation = self.invocation_provider(context)
        if not isinstance(invocation, DispatchPermitInvocation):
            raise TypeError(
                "invocation_provider must return DispatchPermitInvocation."
            )
        result = DispatchSeamGate(
            registry=self.registry,
            spec=DispatchSeamSpec.create(
                seam_id=context.seam_id,
                permit_kind=context.permit_kind,
                operation=context.operation,
                call_stage="pre_sdk_call",
            ),
        ).invoke(
            invocation.permit,
            current_authority_epoch=invocation.current_authority_epoch,
            current_execution_epoch=invocation.current_execution_epoch,
            backend_call=backend_call,
        )
        if self.receipt_sink is not None:
            self.receipt_sink(context, result)
        return result.result


__all__ = [
    "RM65ControlDispatchContext",
    "RM65PermittedPreSdkControlAdapter",
    "RM65PreSdkControlAdapter",
]
