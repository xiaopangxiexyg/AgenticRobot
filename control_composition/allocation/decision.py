"""Immutable decisions and reducer state for composition allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from .contracts import (
    COMPOSITION_ALLOCATION_DECISION_HASH_DOMAIN_DEVELOPMENT,
    COMPOSITION_ALLOCATION_MODES_DEVELOPMENT,
    COMPONENT_ALLOCATION_SELECTION_HASH_DOMAIN_DEVELOPMENT,
    COMPOSITION_ALLOCATION_STATE_HASH_DOMAIN_DEVELOPMENT,
    COMPOSITION_ALLOCATION_STATUSES_DEVELOPMENT,
    COMPOSITION_ALLOCATION_STATUS_READY,
    _finite,
    _mode_map,
    _nonnegative,
    _nonnegative_int,
    _required,
    _transition_time_map,
)


@dataclass(frozen=True)
class ComponentAllocationSelection:
    component_id: str
    previous_mode: str
    selected_mode: str
    transitioned: bool
    reason: str
    motion_observation_id: str
    motion_observation_hash: str
    motion_activity_level: float
    residual_magnitude_observation_id: str
    residual_magnitude_observation_hash: str
    residual_magnitude_level: float
    anchor_receipt_id: str
    anchor_receipt_hash: str
    anchor_request_generation: int
    valid_until: float
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, name="component_id")
        previous = _required(self.previous_mode, name="previous_mode")
        selected = _required(self.selected_mode, name="selected_mode")
        if (
            previous not in COMPOSITION_ALLOCATION_MODES_DEVELOPMENT
            or selected not in (COMPOSITION_ALLOCATION_MODES_DEVELOPMENT)
        ):
            raise ValueError("Allocation selection contains an unsupported mode.")
        transitioned = bool(self.transitioned)
        if transitioned != (previous != selected):
            raise ValueError("transitioned must match the mode change.")
        reason = _required(self.reason, name="reason")
        motion_id = _required(
            self.motion_observation_id,
            name="motion_observation_id",
        )
        motion_hash = _required(
            self.motion_observation_hash,
            name="motion_observation_hash",
        )
        motion_level = _nonnegative(
            self.motion_activity_level,
            name="motion_activity_level",
        )
        magnitude_id = _required(
            self.residual_magnitude_observation_id,
            name="residual_magnitude_observation_id",
        )
        magnitude_hash = _required(
            self.residual_magnitude_observation_hash,
            name="residual_magnitude_observation_hash",
        )
        magnitude_level = _nonnegative(
            self.residual_magnitude_level,
            name="residual_magnitude_level",
        )
        anchor_receipt_id = _required(
            self.anchor_receipt_id,
            name="anchor_receipt_id",
        )
        anchor_receipt_hash = _required(
            self.anchor_receipt_hash,
            name="anchor_receipt_hash",
        )
        anchor_generation = _nonnegative_int(
            self.anchor_request_generation,
            name="anchor_request_generation",
        )
        valid_until = _finite(self.valid_until, name="valid_until")
        payload = {
            "component_id": component_id,
            "previous_mode": previous,
            "selected_mode": selected,
            "transitioned": transitioned,
            "reason": reason,
            "motion_observation_id": motion_id,
            "motion_observation_hash": motion_hash,
            "motion_activity_level": motion_level,
            "residual_magnitude_observation_id": magnitude_id,
            "residual_magnitude_observation_hash": magnitude_hash,
            "residual_magnitude_level": magnitude_level,
            "anchor_receipt_id": anchor_receipt_id,
            "anchor_receipt_hash": anchor_receipt_hash,
            "anchor_request_generation": anchor_generation,
            "valid_until": valid_until,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(COMPONENT_ALLOCATION_SELECTION_HASH_DOMAIN_DEVELOPMENT),
            ),
        )


@dataclass(frozen=True)
class CompositionAllocationDecision:
    allocation_spec_hash: str
    previous_state_hash: str
    evaluated_at: float
    clock_domain_id: str
    output_status: str
    reason: str
    selections: tuple[ComponentAllocationSelection, ...] = ()
    unavailable_reason_by_component: Mapping[str, str] = field(default_factory=dict)
    allocation_override_state_id: str | None = None
    allocation_override_state_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        spec_hash = _required(
            self.allocation_spec_hash,
            name="allocation_spec_hash",
        )
        previous_hash = _required(
            self.previous_state_hash,
            name="previous_state_hash",
        )
        evaluated_at = _finite(self.evaluated_at, name="evaluated_at")
        clock = _required(self.clock_domain_id, name="clock_domain_id")
        status = _required(self.output_status, name="output_status")
        if status not in COMPOSITION_ALLOCATION_STATUSES_DEVELOPMENT:
            raise ValueError("Unsupported allocation decision status.")
        reason = _required(self.reason, name="reason")
        selections = tuple(self.selections)
        unavailable = MappingProxyType(
            {
                _required(component, name="unavailable_component"): _required(
                    unavailable_reason,
                    name=f"unavailable_reason[{component!r}]",
                )
                for component, unavailable_reason in sorted(
                    dict(self.unavailable_reason_by_component).items()
                )
            }
        )
        if status == COMPOSITION_ALLOCATION_STATUS_READY:
            if not selections or unavailable:
                raise ValueError(
                    "A ready allocation decision requires selections only."
                )
        elif selections or not unavailable:
            raise ValueError(
                "A no-decision allocation requires unavailable components only."
            )
        if any(
            not isinstance(selection, ComponentAllocationSelection)
            for selection in selections
        ):
            raise TypeError("selections must contain ComponentAllocationSelection.")
        component_ids = tuple(selection.component_id for selection in selections)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("Allocation selections must have unique components.")
        override_pair = (
            self.allocation_override_state_id,
            self.allocation_override_state_hash,
        )
        if (override_pair[0] is None) != (override_pair[1] is None):
            raise ValueError("Allocation override state identity must be complete.")
        override_id = (
            None
            if override_pair[0] is None
            else _required(override_pair[0], name="allocation_override_state_id")
        )
        override_hash = (
            None
            if override_pair[1] is None
            else _required(
                override_pair[1],
                name="allocation_override_state_hash",
            )
        )
        payload = {
            "allocation_spec_hash": spec_hash,
            "previous_state_hash": previous_hash,
            "evaluated_at": evaluated_at,
            "clock_domain_id": clock,
            "output_status": status,
            "reason": reason,
            "selection_hashes": tuple(
                selection.semantic_content_hash for selection in selections
            ),
            "unavailable_reason_by_component": unavailable,
            "authoritative": False,
        }
        if override_id is not None:
            payload.update(
                {
                    "allocation_override_state_id": override_id,
                    "allocation_override_state_hash": override_hash,
                }
            )
        object.__setattr__(self, "allocation_spec_hash", spec_hash)
        object.__setattr__(self, "previous_state_hash", previous_hash)
        object.__setattr__(self, "evaluated_at", evaluated_at)
        object.__setattr__(self, "clock_domain_id", clock)
        object.__setattr__(self, "output_status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "selections", selections)
        object.__setattr__(
            self,
            "unavailable_reason_by_component",
            unavailable,
        )
        object.__setattr__(self, "allocation_override_state_id", override_id)
        object.__setattr__(self, "allocation_override_state_hash", override_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPOSITION_ALLOCATION_DECISION_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def decision_id(self) -> str:
        return f"composition-allocation-decision:{self.semantic_content_hash}"

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def selected_mode_by_component(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                selection.component_id: selection.selected_mode
                for selection in self.selections
            }
        )

    @property
    def unavailable_components(self) -> tuple[str, ...]:
        return tuple(self.unavailable_reason_by_component)


@dataclass(frozen=True)
class CompositionAllocationState:
    reducer_id: str
    revision: int
    allocation_spec_hash: str
    component_ids: tuple[str, ...]
    mode_by_component: Mapping[str, str]
    mode_entered_at_by_component: Mapping[str, float | None]
    last_decision_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        reducer_id = _required(self.reducer_id, name="reducer_id")
        revision = _nonnegative_int(self.revision, name="revision")
        spec_hash = _required(
            self.allocation_spec_hash,
            name="allocation_spec_hash",
        )
        component_ids = tuple(
            _required(value, name="component_id") for value in self.component_ids
        )
        if not component_ids or len(component_ids) != len(set(component_ids)):
            raise ValueError("component_ids must be non-empty and unique.")
        modes = _mode_map(
            self.mode_by_component,
            component_ids=component_ids,
        )
        entered = _transition_time_map(
            self.mode_entered_at_by_component,
            component_ids=component_ids,
        )
        last_hash = self.last_decision_hash
        if last_hash is not None:
            last_hash = _required(last_hash, name="last_decision_hash")
        if revision == 0 and last_hash is not None:
            raise ValueError("Initial allocation state cannot bind a decision.")
        if revision > 0 and last_hash is None:
            raise ValueError("Advanced allocation state requires a decision hash.")
        payload = {
            "reducer_id": reducer_id,
            "revision": revision,
            "allocation_spec_hash": spec_hash,
            "component_ids": component_ids,
            "mode_by_component": modes,
            "mode_entered_at_by_component": entered,
            "last_decision_hash": last_hash,
        }
        object.__setattr__(self, "reducer_id", reducer_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "allocation_spec_hash", spec_hash)
        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "mode_by_component", modes)
        object.__setattr__(self, "mode_entered_at_by_component", entered)
        object.__setattr__(self, "last_decision_hash", last_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPOSITION_ALLOCATION_STATE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


@dataclass(frozen=True)
class CompositionAllocationEvaluation:
    state: CompositionAllocationState
    decision: CompositionAllocationDecision

    def __post_init__(self) -> None:
        if not isinstance(self.state, CompositionAllocationState):
            raise TypeError("state must be a CompositionAllocationState.")
        if not isinstance(self.decision, CompositionAllocationDecision):
            raise TypeError("decision must be a CompositionAllocationDecision.")
        if self.state.allocation_spec_hash != self.decision.allocation_spec_hash:
            raise ValueError("Allocation evaluation spec hashes disagree.")
        if self.decision.output_status == COMPOSITION_ALLOCATION_STATUS_READY:
            if self.state.last_decision_hash != self.decision.semantic_content_hash:
                raise ValueError(
                    "Ready allocation state must bind the emitted decision."
                )
        elif self.state.semantic_content_hash != self.decision.previous_state_hash:
            raise ValueError("No-decision evaluation must preserve state.")


__all__ = [
    "ComponentAllocationSelection",
    "CompositionAllocationDecision",
    "CompositionAllocationEvaluation",
    "CompositionAllocationState",
]
