"""Immutable contracts for allocation-mode target continuity.

This module corrects only the seam introduced by changing a composition mode.
It does not interpolate execution References, run safety/IK, limit motion, or
dispatch commands.  The reducer is component-keyed and backend-neutral; pose
and arbitrary joint-position payloads share the same lifecycle contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction

from .contracts import (
    COMPOSITION_ALLOCATION_MODES_DEVELOPMENT,
)


ALLOCATION_TRANSITION_CONTINUITY_POLICY_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.allocation_transition_continuity_policy.development"
)
ALLOCATION_TRANSITION_CONTINUITY_STATE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.allocation_transition_continuity_state.development"
)
ALLOCATION_TRANSITION_CONTINUITY_RECEIPT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.allocation_transition_continuity_receipt.development"
)

ALLOCATION_CONTINUITY_CORRECTION_NONE = "none"
ALLOCATION_CONTINUITY_CORRECTION_PERSISTENT_ABSOLUTE_ALIGNMENT = (
    "persistent_absolute_alignment"
)
ALLOCATION_CONTINUITY_CORRECTION_DECAYING_HANDOFF = "decaying_handoff"
ALLOCATION_CONTINUITY_CORRECTION_KINDS_DEVELOPMENT = frozenset(
    {
        ALLOCATION_CONTINUITY_CORRECTION_NONE,
        ALLOCATION_CONTINUITY_CORRECTION_PERSISTENT_ABSOLUTE_ALIGNMENT,
        ALLOCATION_CONTINUITY_CORRECTION_DECAYING_HANDOFF,
    }
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


def _components(values: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(_required(value, name="component_id") for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError("component_ids must be non-empty and unique.")
    return result


def _mode_map(
    values: Mapping[str, str | None],
    *,
    component_ids: tuple[str, ...],
) -> Mapping[str, str | None]:
    raw = dict(values)
    if set(raw) != set(component_ids):
        raise ValueError("mode_by_component must cover the topology exactly.")
    normalized: dict[str, str | None] = {}
    for component_id in component_ids:
        value = raw[component_id]
        if value is None:
            normalized[component_id] = None
            continue
        mode = _required(value, name=f"mode_by_component[{component_id!r}]")
        if mode not in COMPOSITION_ALLOCATION_MODES_DEVELOPMENT:
            raise ValueError("mode_by_component contains an unsupported mode.")
        normalized[component_id] = mode
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class AllocationTransitionContinuityPolicy:
    """Profile-owned target seam behavior for one component topology."""

    policy_id: str
    component_ids: tuple[str, ...]
    handoff_duration_sec: float
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        policy_id = _required(self.policy_id, name="policy_id")
        component_ids = _components(tuple(self.component_ids))
        duration = _finite(
            self.handoff_duration_sec,
            name="handoff_duration_sec",
        )
        if duration <= 0.0:
            raise ValueError("handoff_duration_sec must be positive.")
        payload = {
            "policy_id": policy_id,
            "component_ids": component_ids,
            "absolute_entry": "align_current_absolute_to_previous_composed_target",
            "absolute_alignment_lifecycle": "persistent_until_mode_exit",
            "non_absolute_transition": "decay_previous_to_raw_target_seam",
            "handoff_duration_sec": duration,
            "state_advance": "exact_proposal_admission_accepted",
            "execution_reference_semantics": "not_owned",
        }
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "handoff_duration_sec", duration)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    ALLOCATION_TRANSITION_CONTINUITY_POLICY_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


@dataclass(frozen=True)
class AllocationComponentContinuityState:
    component_id: str
    selected_mode: str | None
    correction_kind: str
    correction_values: tuple[float, ...] = ()
    correction_started_at: float | None = None
    correction_ends_at: float | None = None

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, name="component_id")
        mode = self.selected_mode
        if mode is not None:
            mode = _required(mode, name="selected_mode")
            if mode not in COMPOSITION_ALLOCATION_MODES_DEVELOPMENT:
                raise ValueError("selected_mode is unsupported.")
        kind = _required(self.correction_kind, name="correction_kind")
        if kind not in ALLOCATION_CONTINUITY_CORRECTION_KINDS_DEVELOPMENT:
            raise ValueError("correction_kind is unsupported.")
        values = tuple(
            _finite(value, name="correction_value")
            for value in self.correction_values
        )
        started = self.correction_started_at
        ends = self.correction_ends_at
        if kind == ALLOCATION_CONTINUITY_CORRECTION_NONE:
            if values or started is not None or ends is not None:
                raise ValueError("An inactive correction cannot retain payload.")
        else:
            if not values or started is None:
                raise ValueError("An active correction requires values and start.")
            started = _finite(started, name="correction_started_at")
            if kind == (
                ALLOCATION_CONTINUITY_CORRECTION_PERSISTENT_ABSOLUTE_ALIGNMENT
            ):
                if ends is not None:
                    raise ValueError("Persistent alignment has no end time.")
            else:
                if ends is None:
                    raise ValueError("A decaying correction requires an end time.")
                ends = _finite(ends, name="correction_ends_at")
                if ends <= started:
                    raise ValueError("Correction end must follow its start.")
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "selected_mode", mode)
        object.__setattr__(self, "correction_kind", kind)
        object.__setattr__(self, "correction_values", values)
        object.__setattr__(self, "correction_started_at", started)
        object.__setattr__(self, "correction_ends_at", ends)

    def semantic_payload(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "selected_mode": self.selected_mode,
            "correction_kind": self.correction_kind,
            "correction_values": self.correction_values,
            "correction_started_at": self.correction_started_at,
            "correction_ends_at": self.correction_ends_at,
        }


@dataclass(frozen=True)
class AllocationTransitionContinuityState:
    policy_hash: str
    composition_activation_id: str
    composition_activation_hash: str
    revision: int
    component_states: tuple[AllocationComponentContinuityState, ...]
    last_target_effective_at: float | None = None
    last_output: ComponentAction | None = None
    last_invocation_id: str | None = None
    last_invocation_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        policy_hash = _required(self.policy_hash, name="policy_hash")
        activation_id = _required(
            self.composition_activation_id,
            name="composition_activation_id",
        )
        activation_hash = _required(
            self.composition_activation_hash,
            name="composition_activation_hash",
        )
        revision = self.revision
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("revision must be a non-negative integer.")
        component_states = tuple(self.component_states)
        if not component_states or any(
            not isinstance(item, AllocationComponentContinuityState)
            for item in component_states
        ):
            raise TypeError("component_states must be typed and non-empty.")
        component_ids = tuple(item.component_id for item in component_states)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component continuity states must be unique.")
        target = self.last_target_effective_at
        invocation_pair = (self.last_invocation_id, self.last_invocation_hash)
        if (invocation_pair[0] is None) != (invocation_pair[1] is None):
            raise ValueError("Invocation identity must be complete.")
        if revision == 0:
            if target is not None or self.last_output is not None or any(
                value is not None for value in invocation_pair
            ):
                raise ValueError("Revision zero cannot contain an output.")
        else:
            if target is None or not isinstance(self.last_output, ComponentAction):
                raise ValueError("An advanced state requires target and output.")
            target = _finite(target, name="last_target_effective_at")
            if any(value is None for value in invocation_pair):
                raise ValueError("An advanced state requires invocation lineage.")
        payload = {
            "policy_hash": policy_hash,
            "composition_activation_id": activation_id,
            "composition_activation_hash": activation_hash,
            "revision": revision,
            "component_states": tuple(
                item.semantic_payload() for item in component_states
            ),
            "last_target_effective_at": target,
            "last_output_hash": (
                None
                if self.last_output is None
                else self.last_output.semantic_content_hash
            ),
            "last_invocation_id": invocation_pair[0],
            "last_invocation_hash": invocation_pair[1],
        }
        object.__setattr__(self, "policy_hash", policy_hash)
        object.__setattr__(self, "composition_activation_id", activation_id)
        object.__setattr__(self, "composition_activation_hash", activation_hash)
        object.__setattr__(self, "component_states", component_states)
        object.__setattr__(self, "last_target_effective_at", target)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    ALLOCATION_TRANSITION_CONTINUITY_STATE_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def component_state_by_id(
        self,
    ) -> Mapping[str, AllocationComponentContinuityState]:
        return MappingProxyType(
            {item.component_id: item for item in self.component_states}
        )


@dataclass(frozen=True)
class AllocationTransitionContinuityReceipt:
    policy_hash: str
    composition_activation_id: str
    composition_activation_hash: str
    invocation_id: str
    invocation_hash: str
    target_effective_at: float
    state_before_hash: str
    state_after_hash: str
    raw_candidate_hash: str
    output_candidate_hash: str
    previous_mode_by_component: Mapping[str, str | None]
    selected_mode_by_component: Mapping[str, str]
    transition_by_component: Mapping[str, str]
    correction_kind_by_component: Mapping[str, str]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "policy_hash",
                "composition_activation_id",
                "composition_activation_hash",
                "invocation_id",
                "invocation_hash",
                "state_before_hash",
                "state_after_hash",
                "raw_candidate_hash",
                "output_candidate_hash",
            )
        }
        target = _finite(self.target_effective_at, name="target_effective_at")
        components = tuple(self.selected_mode_by_component)
        selected = _mode_map(
            self.selected_mode_by_component,
            component_ids=components,
        )
        if any(value is None for value in selected.values()):
            raise ValueError("Receipt selected modes cannot be None.")
        previous = _mode_map(
            self.previous_mode_by_component,
            component_ids=components,
        )
        transitions = MappingProxyType(
            {
                component_id: _required(
                    self.transition_by_component[component_id],
                    name=f"transition[{component_id!r}]",
                )
                for component_id in components
            }
        )
        corrections = MappingProxyType(
            {
                component_id: _required(
                    self.correction_kind_by_component[component_id],
                    name=f"correction_kind[{component_id!r}]",
                )
                for component_id in components
            }
        )
        if set(transitions) != set(components) or set(corrections) != set(components):
            raise ValueError("Receipt component maps must cover the topology.")
        if any(
            value not in ALLOCATION_CONTINUITY_CORRECTION_KINDS_DEVELOPMENT
            for value in corrections.values()
        ):
            raise ValueError("Receipt contains an unsupported correction kind.")
        payload = {
            **text,
            "target_effective_at": target,
            "previous_mode_by_component": previous,
            "selected_mode_by_component": selected,
            "transition_by_component": transitions,
            "correction_kind_by_component": corrections,
            "state_after_status": (
                "prepared_until_exact_proposal_admission_accepted"
            ),
        }
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(self, "previous_mode_by_component", previous)
        object.__setattr__(self, "selected_mode_by_component", selected)
        object.__setattr__(self, "transition_by_component", transitions)
        object.__setattr__(self, "correction_kind_by_component", corrections)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    ALLOCATION_TRANSITION_CONTINUITY_RECEIPT_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def receipt_id(self) -> str:
        return f"allocation-transition-continuity:{self.semantic_content_hash}"

    def semantic_payload(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "receipt_hash": self.semantic_content_hash,
            "policy_hash": self.policy_hash,
            "composition_activation_id": self.composition_activation_id,
            "composition_activation_hash": self.composition_activation_hash,
            "invocation_id": self.invocation_id,
            "invocation_hash": self.invocation_hash,
            "target_effective_at": self.target_effective_at,
            "state_before_hash": self.state_before_hash,
            "state_after_hash": self.state_after_hash,
            "raw_candidate_hash": self.raw_candidate_hash,
            "output_candidate_hash": self.output_candidate_hash,
            "previous_mode_by_component": dict(self.previous_mode_by_component),
            "selected_mode_by_component": dict(self.selected_mode_by_component),
            "transition_by_component": dict(self.transition_by_component),
            "correction_kind_by_component": dict(
                self.correction_kind_by_component
            ),
            "state_after_status": (
                "prepared_until_exact_proposal_admission_accepted"
            ),
        }


@dataclass(frozen=True)
class AllocationTransitionContinuityEvaluation:
    state: AllocationTransitionContinuityState
    output: ComponentAction
    receipt: AllocationTransitionContinuityReceipt


def empty_allocation_transition_continuity_state(
    policy: AllocationTransitionContinuityPolicy,
    *,
    composition_activation_id: str,
    composition_activation_hash: str,
) -> AllocationTransitionContinuityState:
    if not isinstance(policy, AllocationTransitionContinuityPolicy):
        raise TypeError("policy must be typed.")
    return AllocationTransitionContinuityState(
        policy_hash=policy.semantic_content_hash,
        composition_activation_id=composition_activation_id,
        composition_activation_hash=composition_activation_hash,
        revision=0,
        component_states=tuple(
            AllocationComponentContinuityState(
                component_id=component_id,
                selected_mode=None,
                correction_kind=ALLOCATION_CONTINUITY_CORRECTION_NONE,
            )
            for component_id in policy.component_ids
        ),
    )


__all__ = [
    "ALLOCATION_CONTINUITY_CORRECTION_DECAYING_HANDOFF",
    "ALLOCATION_CONTINUITY_CORRECTION_KINDS_DEVELOPMENT",
    "ALLOCATION_CONTINUITY_CORRECTION_NONE",
    "ALLOCATION_CONTINUITY_CORRECTION_PERSISTENT_ABSOLUTE_ALIGNMENT",
    "ALLOCATION_TRANSITION_CONTINUITY_POLICY_HASH_DOMAIN_DEVELOPMENT",
    "ALLOCATION_TRANSITION_CONTINUITY_RECEIPT_HASH_DOMAIN_DEVELOPMENT",
    "ALLOCATION_TRANSITION_CONTINUITY_STATE_HASH_DOMAIN_DEVELOPMENT",
    "AllocationComponentContinuityState",
    "AllocationTransitionContinuityEvaluation",
    "AllocationTransitionContinuityPolicy",
    "AllocationTransitionContinuityReceipt",
    "AllocationTransitionContinuityState",
    "empty_allocation_transition_continuity_state",
]
