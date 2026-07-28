"""Generic activation-local residual anchor contracts.

These contracts bind an absolute-to-relative source adaptation to one exact
composition activation, source role, source binding, action schemas, and
atomic component set.  They do not assume that either role is human, policy,
replay, or planner, and they own no device, authority, safety, or execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash

from .activation import CompositionActivationScope


RESIDUAL_ANCHOR_REQUEST_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.residual_anchor_request.development"
)
RESIDUAL_ANCHOR_RECEIPT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.residual_anchor_receipt.development"
)
RESIDUAL_ANCHOR_COMPONENT_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.residual_anchor_component_binding.development"
)
RESIDUAL_ANCHOR_STATE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.residual_anchor_state.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _components(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_required(value, name="component_id") for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError("component_ids must be non-empty and unique.")
    return result


@dataclass(frozen=True)
class ResidualAnchorRequest:
    """Request one atomic anchor for a profile-selected residual source."""

    request_generation: int
    source_role: str
    component_ids: tuple[str, ...]
    composition_activation_id: str
    composition_activation_hash: str
    source_binding_id: str
    source_binding_hash: str
    absolute_action_schema_hash: str
    relative_action_schema_hash: str
    requested_at: float
    clock_domain_id: str
    input_barrier_at: float
    previous_request_id: str | None = None
    previous_request_hash: str | None = None
    semantic_content_hash: str = field(init=False)
    request_id: str = field(init=False)

    def __post_init__(self) -> None:
        generation = int(self.request_generation)
        if (
            isinstance(self.request_generation, bool)
            or generation != self.request_generation
            or generation < 0
        ):
            raise ValueError(
                "request_generation must be a non-negative integer."
            )
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "source_role",
                "composition_activation_id",
                "composition_activation_hash",
                "source_binding_id",
                "source_binding_hash",
                "absolute_action_schema_hash",
                "relative_action_schema_hash",
                "clock_domain_id",
            )
        }
        components = _components(self.component_ids)
        requested_at = _finite(self.requested_at, name="requested_at")
        barrier = _finite(self.input_barrier_at, name="input_barrier_at")
        if barrier > requested_at:
            raise ValueError("input_barrier_at cannot follow requested_at.")
        previous_id = self.previous_request_id
        previous_hash = self.previous_request_hash
        if (previous_id is None) != (previous_hash is None):
            raise ValueError(
                "previous_request_id/hash must be both present or both absent."
            )
        if generation == 0 and previous_id is not None:
            raise ValueError(
                "Generation zero cannot declare a predecessor request."
            )
        if generation > 0 and previous_id is None:
            raise ValueError(
                "A later anchor generation requires its predecessor request."
            )
        if previous_id is not None:
            previous_id = _required(previous_id, name="previous_request_id")
            previous_hash = _required(
                previous_hash,
                name="previous_request_hash",
            )
        payload = {
            "request_generation": generation,
            **text,
            "component_ids": components,
            "requested_at": requested_at,
            "input_barrier_at": barrier,
            "previous_request_id": previous_id,
            "previous_request_hash": previous_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=RESIDUAL_ANCHOR_REQUEST_HASH_DOMAIN_DEVELOPMENT,
        )
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "request_id",
            f"residual-anchor-request:{generation}:{semantic_hash}",
        )

    @classmethod
    def for_activation(
        cls,
        activation_scope: CompositionActivationScope,
        *,
        request_generation: int,
        source_role: str,
        component_ids: Sequence[str],
        source_binding_id: str,
        source_binding_hash: str,
        absolute_action_schema_hash: str,
        relative_action_schema_hash: str,
        requested_at: float,
        previous_request: "ResidualAnchorRequest | None" = None,
    ) -> "ResidualAnchorRequest":
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError(
                "activation_scope must be CompositionActivationScope."
            )
        generation = int(request_generation)
        if previous_request is not None:
            if not isinstance(previous_request, ResidualAnchorRequest):
                raise TypeError(
                    "previous_request must be ResidualAnchorRequest or None."
                )
            if generation != previous_request.request_generation + 1:
                raise ValueError(
                    "Residual anchor generation must advance by exactly one."
                )
            if previous_request.source_role != str(source_role).strip():
                raise ValueError(
                    "Residual anchor source_role cannot change within a chain."
                )
            if previous_request.component_ids != tuple(component_ids):
                raise ValueError(
                    "Residual anchor component coverage cannot change within "
                    "a chain."
                )
        request_time = _finite(requested_at, name="requested_at")
        if request_time < activation_scope.input_barrier_at:
            raise ValueError(
                "Residual anchor request cannot predate the composition "
                "activation barrier."
            )
        return cls(
            request_generation=generation,
            source_role=source_role,
            component_ids=tuple(component_ids),
            composition_activation_id=activation_scope.activation_id,
            composition_activation_hash=(
                activation_scope.semantic_content_hash
            ),
            source_binding_id=source_binding_id,
            source_binding_hash=source_binding_hash,
            absolute_action_schema_hash=absolute_action_schema_hash,
            relative_action_schema_hash=relative_action_schema_hash,
            requested_at=request_time,
            clock_domain_id=activation_scope.clock_domain_id,
            input_barrier_at=request_time,
            previous_request_id=(
                None
                if previous_request is None
                else previous_request.request_id
            ),
            previous_request_hash=(
                None
                if previous_request is None
                else previous_request.semantic_content_hash
            ),
        )


@dataclass(frozen=True)
class ResidualAnchorComponentBinding:
    """Bind one component sample to its representation-specific anchor."""

    component_id: str
    source_sample_id: str
    source_sample_hash: str
    absolute_payload_schema_hash: str
    anchor_id: str
    anchor_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "component_id",
                "source_sample_id",
                "source_sample_hash",
                "absolute_payload_schema_hash",
                "anchor_id",
                "anchor_hash",
            )
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    RESIDUAL_ANCHOR_COMPONENT_BINDING_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


@dataclass(frozen=True)
class ResidualAnchorReceipt:
    """Atomic evidence that one request captured its complete component set."""

    request_id: str
    request_hash: str
    request_generation: int
    source_role: str
    composition_activation_id: str
    composition_activation_hash: str
    source_binding_id: str
    source_binding_hash: str
    absolute_action_schema_hash: str
    relative_action_schema_hash: str
    source_action_id: str
    source_action_hash: str
    source_available_at: float
    captured_at: float
    clock_domain_id: str
    component_bindings: tuple[ResidualAnchorComponentBinding, ...]
    semantic_content_hash: str = field(init=False)
    receipt_id: str = field(init=False)
    component_binding_by_id: Mapping[
        str,
        ResidualAnchorComponentBinding,
    ] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        generation = int(self.request_generation)
        if (
            isinstance(self.request_generation, bool)
            or generation != self.request_generation
            or generation < 0
        ):
            raise ValueError(
                "request_generation must be a non-negative integer."
            )
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "request_id",
                "request_hash",
                "source_role",
                "composition_activation_id",
                "composition_activation_hash",
                "source_binding_id",
                "source_binding_hash",
                "absolute_action_schema_hash",
                "relative_action_schema_hash",
                "source_action_id",
                "source_action_hash",
                "clock_domain_id",
            )
        }
        available_at = _finite(
            self.source_available_at,
            name="source_available_at",
        )
        captured_at = _finite(self.captured_at, name="captured_at")
        if captured_at < available_at:
            raise ValueError(
                "Residual anchor cannot be captured before source availability."
            )
        bindings = tuple(self.component_bindings)
        if not bindings or any(
            not isinstance(item, ResidualAnchorComponentBinding)
            for item in bindings
        ):
            raise TypeError(
                "component_bindings must contain at least one "
                "ResidualAnchorComponentBinding."
            )
        by_id = {item.component_id: item for item in bindings}
        if len(by_id) != len(bindings):
            raise ValueError(
                "Residual anchor component bindings must be unique."
            )
        payload = {
            "request_generation": generation,
            **text,
            "source_available_at": available_at,
            "captured_at": captured_at,
            "component_binding_hash_by_id": {
                component_id: by_id[component_id].semantic_content_hash
                for component_id in sorted(by_id)
            },
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=RESIDUAL_ANCHOR_RECEIPT_HASH_DOMAIN_DEVELOPMENT,
        )
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "request_generation", generation)
        object.__setattr__(self, "source_available_at", available_at)
        object.__setattr__(self, "captured_at", captured_at)
        object.__setattr__(
            self,
            "component_bindings",
            tuple(by_id[component_id] for component_id in sorted(by_id)),
        )
        object.__setattr__(
            self,
            "component_binding_by_id",
            MappingProxyType(by_id),
        )
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "receipt_id",
            f"residual-anchor-receipt:{generation}:{semantic_hash}",
        )

    @classmethod
    def capture(
        cls,
        request: ResidualAnchorRequest,
        *,
        source_action_id: str,
        source_action_hash: str,
        source_available_at: float,
        captured_at: float,
        component_bindings: Sequence[ResidualAnchorComponentBinding],
    ) -> "ResidualAnchorReceipt":
        if not isinstance(request, ResidualAnchorRequest):
            raise TypeError("request must be ResidualAnchorRequest.")
        available_at = _finite(
            source_available_at,
            name="source_available_at",
        )
        if available_at < request.input_barrier_at:
            raise ValueError(
                "Residual anchor source became available before its request "
                "barrier."
            )
        bindings = tuple(component_bindings)
        if {item.component_id for item in bindings} != set(
            request.component_ids
        ):
            raise ValueError(
                "Residual anchor receipt must atomically cover exactly the "
                "requested components."
            )
        return cls(
            request_id=request.request_id,
            request_hash=request.semantic_content_hash,
            request_generation=request.request_generation,
            source_role=request.source_role,
            composition_activation_id=request.composition_activation_id,
            composition_activation_hash=request.composition_activation_hash,
            source_binding_id=request.source_binding_id,
            source_binding_hash=request.source_binding_hash,
            absolute_action_schema_hash=(
                request.absolute_action_schema_hash
            ),
            relative_action_schema_hash=(
                request.relative_action_schema_hash
            ),
            source_action_id=source_action_id,
            source_action_hash=source_action_hash,
            source_available_at=available_at,
            captured_at=captured_at,
            clock_domain_id=request.clock_domain_id,
            component_bindings=bindings,
        )

    def validate_request(self, request: ResidualAnchorRequest) -> None:
        if not isinstance(request, ResidualAnchorRequest):
            raise TypeError("request must be ResidualAnchorRequest.")
        if (
            self.request_id != request.request_id
            or self.request_hash != request.semantic_content_hash
        ):
            raise ValueError(
                "Residual anchor receipt belongs to another request."
            )
        if set(self.component_binding_by_id) != set(request.component_ids):
            raise ValueError(
                "Residual anchor receipt component coverage changed."
            )


@dataclass(frozen=True)
class ResidualAnchorState:
    """Small immutable reducer state for one active anchor request."""

    request: ResidualAnchorRequest
    receipt: ResidualAnchorReceipt | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ResidualAnchorRequest):
            raise TypeError("request must be ResidualAnchorRequest.")
        if self.receipt is not None:
            if not isinstance(self.receipt, ResidualAnchorReceipt):
                raise TypeError(
                    "receipt must be ResidualAnchorReceipt or None."
                )
            self.receipt.validate_request(self.request)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "request_hash": self.request.semantic_content_hash,
                    "receipt_hash": (
                        None
                        if self.receipt is None
                        else self.receipt.semantic_content_hash
                    ),
                },
                domain=RESIDUAL_ANCHOR_STATE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def ready(self) -> bool:
        return self.receipt is not None

    def accept(self, receipt: ResidualAnchorReceipt) -> "ResidualAnchorState":
        if self.receipt is not None:
            if (
                self.receipt.semantic_content_hash
                != receipt.semantic_content_hash
            ):
                raise ValueError(
                    "Residual anchor request already has another receipt."
                )
            return self
        return ResidualAnchorState(request=self.request, receipt=receipt)


__all__ = [
    "RESIDUAL_ANCHOR_COMPONENT_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "RESIDUAL_ANCHOR_RECEIPT_HASH_DOMAIN_DEVELOPMENT",
    "RESIDUAL_ANCHOR_REQUEST_HASH_DOMAIN_DEVELOPMENT",
    "RESIDUAL_ANCHOR_STATE_HASH_DOMAIN_DEVELOPMENT",
    "ResidualAnchorComponentBinding",
    "ResidualAnchorReceipt",
    "ResidualAnchorRequest",
    "ResidualAnchorState",
]
