"""Persistent, activation-local overrides for composition allocation.

The override is an operator/profile selection, not motion authority.  It only
selects which already-declared composition mode the allocation reducer should
emit.  Source admission, required-role handling, canonical safety, target
resolution, interpolation, IK, and dispatch remain with their existing owners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from .contracts import (
    COMPOSITION_ALLOCATION_MODES_DEVELOPMENT,
    _finite,
    _nonnegative_int,
    _required,
)


COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC = "automatic"
COMPOSITION_ALLOCATION_OVERRIDE_OPERATION_KIND = (
    "set_composition_allocation_override"
)
COMPOSITION_ALLOCATION_OVERRIDE_PAYLOAD_CONTRACT_DEVELOPMENT = (
    "dexteleop.composition_allocation_override_payload.development"
)
COMPOSITION_ALLOCATION_OVERRIDE_REQUEST_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_allocation_override_request.development"
)
COMPOSITION_ALLOCATION_OVERRIDE_STATE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_allocation_override_state.development"
)
COMPOSITION_ALLOCATION_OVERRIDE_RECEIPT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_allocation_override_receipt.development"
)


def _lineage(
    fact_ids: tuple[str, ...],
    fact_hashes: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ids = tuple(_required(value, name="source_fact_id") for value in fact_ids)
    hashes = tuple(
        _required(value, name="source_fact_hash") for value in fact_hashes
    )
    if not ids or len(ids) != len(hashes):
        raise ValueError("Override source lineage must be non-empty and paired.")
    if len(ids) != len(set(ids)):
        raise ValueError("Override source fact ids must be unique.")
    return ids, hashes


def _optional_mode_map(
    values: Mapping[str, str | None],
    *,
    component_ids: tuple[str, ...],
    allowed_modes: tuple[str, ...],
) -> Mapping[str, str | None]:
    raw = dict(values)
    if set(raw) != set(component_ids):
        raise ValueError("Override modes must cover the component set exactly.")
    normalized: dict[str, str | None] = {}
    for component_id in component_ids:
        value = raw[component_id]
        if value is None:
            normalized[component_id] = None
            continue
        mode = _required(value, name=f"override_mode[{component_id!r}]")
        if mode not in allowed_modes:
            raise ValueError("Override selected a mode outside its allowlist.")
        normalized[component_id] = mode
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class CompositionAllocationOverrideRequest:
    """One typed request created from an admitted button/trigger proposal."""

    runtime_session_id: str
    control_domain_id: str
    composition_activation_id: str
    composition_activation_hash: str
    allocation_spec_hash: str
    component_ids: tuple[str, ...]
    requested_mode: str
    requested_at: float
    available_at: float
    clock_domain_id: str
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "runtime_session_id",
                "control_domain_id",
                "composition_activation_id",
                "composition_activation_hash",
                "allocation_spec_hash",
                "clock_domain_id",
            )
        }
        component_ids = tuple(
            sorted(_required(value, name="component_id") for value in self.component_ids)
        )
        if not component_ids or len(component_ids) != len(set(component_ids)):
            raise ValueError("Override component_ids must be non-empty and unique.")
        mode = _required(self.requested_mode, name="requested_mode")
        if mode != COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC and (
            mode not in COMPOSITION_ALLOCATION_MODES_DEVELOPMENT
        ):
            raise ValueError("Override request selected an unsupported mode.")
        requested_at = _finite(self.requested_at, name="requested_at")
        available_at = _finite(self.available_at, name="available_at")
        if requested_at > available_at:
            raise ValueError("Override request cannot be available before requested.")
        fact_ids, fact_hashes = _lineage(
            tuple(self.source_fact_ids),
            tuple(self.source_fact_hashes),
        )
        payload = {
            **text,
            "component_ids": component_ids,
            "requested_mode": mode,
            "requested_at": requested_at,
            "available_at": available_at,
            "source_fact_ids": fact_ids,
            "source_fact_hashes": fact_hashes,
            "authoritative": False,
        }
        for name, value in payload.items():
            if name != "authoritative":
                object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    COMPOSITION_ALLOCATION_OVERRIDE_REQUEST_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def request_id(self) -> str:
        return f"composition-allocation-override-request:{self.semantic_content_hash}"

    @property
    def authoritative(self) -> bool:
        return False


@dataclass(frozen=True)
class CompositionAllocationOverrideState:
    """Current persistent override for one exact composition activation."""

    runtime_session_id: str
    control_domain_id: str
    composition_activation_id: str
    composition_activation_hash: str
    allocation_spec_hash: str
    clock_domain_id: str
    component_ids: tuple[str, ...]
    allowed_modes: tuple[str, ...]
    revision: int
    override_mode_by_component: Mapping[str, str | None]
    last_request_id: str | None = None
    last_request_hash: str | None = None
    last_changed_at: float | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "runtime_session_id",
                "control_domain_id",
                "composition_activation_id",
                "composition_activation_hash",
                "allocation_spec_hash",
                "clock_domain_id",
            )
        }
        component_ids = tuple(
            _required(value, name="component_id") for value in self.component_ids
        )
        if not component_ids or len(component_ids) != len(set(component_ids)):
            raise ValueError("Override state components must be non-empty and unique.")
        allowed_modes = tuple(
            _required(value, name="allowed_mode") for value in self.allowed_modes
        )
        if (
            not allowed_modes
            or len(allowed_modes) != len(set(allowed_modes))
            or any(
                mode not in COMPOSITION_ALLOCATION_MODES_DEVELOPMENT
                for mode in allowed_modes
            )
        ):
            raise ValueError("Override state requires a unique supported allowlist.")
        revision = _nonnegative_int(self.revision, name="revision")
        modes = _optional_mode_map(
            self.override_mode_by_component,
            component_ids=component_ids,
            allowed_modes=allowed_modes,
        )
        request_pair = (self.last_request_id, self.last_request_hash)
        if (request_pair[0] is None) != (request_pair[1] is None):
            raise ValueError("Override state request identity must be complete.")
        request_id = (
            None
            if request_pair[0] is None
            else _required(request_pair[0], name="last_request_id")
        )
        request_hash = (
            None
            if request_pair[1] is None
            else _required(request_pair[1], name="last_request_hash")
        )
        changed_at = (
            None
            if self.last_changed_at is None
            else _finite(self.last_changed_at, name="last_changed_at")
        )
        if revision == 0:
            if request_id is not None or changed_at is not None or any(
                value is not None for value in modes.values()
            ):
                raise ValueError("Initial override state must be automatic and empty.")
        elif request_id is None or changed_at is None:
            raise ValueError("Advanced override state requires request lineage.")
        payload = {
            **text,
            "component_ids": component_ids,
            "allowed_modes": allowed_modes,
            "revision": revision,
            "override_mode_by_component": modes,
            "last_request_id": request_id,
            "last_request_hash": request_hash,
            "last_changed_at": changed_at,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    COMPOSITION_ALLOCATION_OVERRIDE_STATE_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def state_id(self) -> str:
        return f"composition-allocation-override-state:{self.semantic_content_hash}"

    @property
    def requested_mode(self) -> str:
        selected = {value for value in self.override_mode_by_component.values()}
        if selected == {None}:
            return COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC
        if len(selected) == 1:
            value = next(iter(selected))
            assert value is not None
            return value
        return "component_specific"


@dataclass(frozen=True)
class CompositionAllocationOverrideReceipt:
    request_id: str
    request_hash: str
    state_before_hash: str
    state_after: CompositionAllocationOverrideState
    admitted_at: float
    clock_domain_id: str
    status: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state_after, CompositionAllocationOverrideState):
            raise TypeError("state_after must be an allocation override state.")
        status = _required(self.status, name="status")
        if status not in {"applied", "duplicate"}:
            raise ValueError("Unsupported allocation override receipt status.")
        admitted_at = _finite(self.admitted_at, name="admitted_at")
        payload = {
            "request_id": _required(self.request_id, name="request_id"),
            "request_hash": _required(self.request_hash, name="request_hash"),
            "state_before_hash": _required(
                self.state_before_hash,
                name="state_before_hash",
            ),
            "state_after_id": self.state_after.state_id,
            "state_after_hash": self.state_after.semantic_content_hash,
            "admitted_at": admitted_at,
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "status": status,
            "authoritative": False,
        }
        object.__setattr__(self, "admitted_at", admitted_at)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    COMPOSITION_ALLOCATION_OVERRIDE_RECEIPT_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def receipt_id(self) -> str:
        return f"composition-allocation-override-receipt:{self.semantic_content_hash}"

    @property
    def authoritative(self) -> bool:
        return False


def empty_composition_allocation_override_state(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    composition_activation_id: str,
    composition_activation_hash: str,
    allocation_spec_hash: str,
    clock_domain_id: str,
    component_ids: tuple[str, ...],
    allowed_modes: tuple[str, ...],
) -> CompositionAllocationOverrideState:
    return CompositionAllocationOverrideState(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        composition_activation_id=composition_activation_id,
        composition_activation_hash=composition_activation_hash,
        allocation_spec_hash=allocation_spec_hash,
        clock_domain_id=clock_domain_id,
        component_ids=component_ids,
        allowed_modes=allowed_modes,
        revision=0,
        override_mode_by_component={component_id: None for component_id in component_ids},
    )


def apply_composition_allocation_override(
    state: CompositionAllocationOverrideState,
    request: CompositionAllocationOverrideRequest,
    *,
    admitted_at: float,
) -> CompositionAllocationOverrideReceipt:
    """Apply one persistent override without changing motion authority."""

    if not isinstance(state, CompositionAllocationOverrideState):
        raise TypeError("state must be CompositionAllocationOverrideState.")
    if not isinstance(request, CompositionAllocationOverrideRequest):
        raise TypeError("request must be CompositionAllocationOverrideRequest.")
    now = _finite(admitted_at, name="admitted_at")
    if now < request.available_at:
        raise ValueError("Override admission predates request availability.")
    exact = (
        request.runtime_session_id == state.runtime_session_id
        and request.control_domain_id == state.control_domain_id
        and request.composition_activation_id == state.composition_activation_id
        and request.composition_activation_hash == state.composition_activation_hash
        and request.allocation_spec_hash == state.allocation_spec_hash
        and request.clock_domain_id == state.clock_domain_id
    )
    if not exact:
        raise ValueError("Override request belongs to another runtime activation.")
    if not set(request.component_ids).issubset(state.component_ids):
        raise ValueError("Override request contains undeclared components.")
    if state.last_request_id == request.request_id:
        if state.last_request_hash != request.semantic_content_hash:
            raise ValueError("Override request id was reused with another hash.")
        return CompositionAllocationOverrideReceipt(
            request_id=request.request_id,
            request_hash=request.semantic_content_hash,
            state_before_hash=state.semantic_content_hash,
            state_after=state,
            admitted_at=now,
            clock_domain_id=request.clock_domain_id,
            status="duplicate",
        )
    if state.last_changed_at is not None and request.available_at < state.last_changed_at:
        raise ValueError("Override request ordering moved backwards.")
    if (
        request.requested_mode != COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC
        and request.requested_mode not in state.allowed_modes
    ):
        raise ValueError("Override request selected a non-promotable mode.")
    next_modes = dict(state.override_mode_by_component)
    selected = (
        None
        if request.requested_mode == COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC
        else request.requested_mode
    )
    for component_id in request.component_ids:
        next_modes[component_id] = selected
    next_state = CompositionAllocationOverrideState(
        runtime_session_id=state.runtime_session_id,
        control_domain_id=state.control_domain_id,
        composition_activation_id=state.composition_activation_id,
        composition_activation_hash=state.composition_activation_hash,
        allocation_spec_hash=state.allocation_spec_hash,
        clock_domain_id=state.clock_domain_id,
        component_ids=state.component_ids,
        allowed_modes=state.allowed_modes,
        revision=state.revision + 1,
        override_mode_by_component=next_modes,
        last_request_id=request.request_id,
        last_request_hash=request.semantic_content_hash,
        last_changed_at=now,
    )
    return CompositionAllocationOverrideReceipt(
        request_id=request.request_id,
        request_hash=request.semantic_content_hash,
        state_before_hash=state.semantic_content_hash,
        state_after=next_state,
        admitted_at=now,
        clock_domain_id=request.clock_domain_id,
        status="applied",
    )


__all__ = [
    "COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC",
    "COMPOSITION_ALLOCATION_OVERRIDE_OPERATION_KIND",
    "COMPOSITION_ALLOCATION_OVERRIDE_PAYLOAD_CONTRACT_DEVELOPMENT",
    "CompositionAllocationOverrideReceipt",
    "CompositionAllocationOverrideRequest",
    "CompositionAllocationOverrideState",
    "apply_composition_allocation_override",
    "empty_composition_allocation_override_state",
]
