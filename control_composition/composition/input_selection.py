"""Pure target-time selection shared by admitted and projected inputs.

This module deliberately knows nothing about receiver admission receipts or
store retention.  Callers first establish which immutable facts are eligible
for one causal cut, then pass small selection candidates here.  Production
records and non-authoritative preview projections therefore share temporal and
role-dependency semantics without sharing identity domains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Mapping, TypeVar


T = TypeVar("T")


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


@dataclass(frozen=True)
class CompositionInputSelectionCandidate(Generic[T]):
    """One already-eligible carrier presented to the pure selector.

    ``selection_sequence`` is local to the carrier's identity domain.  For an
    admitted record it is the receiver admission sequence; projected inputs
    use a projection-local ordinal and never claim receiver admission.
    """

    carrier: T
    input_role: str
    effective_from: float
    effective_until: float
    selection_sequence: int
    candidate_id: str

    def __post_init__(self) -> None:
        role = _required(self.input_role, name="input_role")
        effective_from = _finite(
            self.effective_from,
            name="effective_from",
        )
        effective_until = _finite(
            self.effective_until,
            name="effective_until",
        )
        if effective_until <= effective_from:
            raise ValueError("Selection candidate interval must be non-empty.")
        sequence = self.selection_sequence
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("selection_sequence must be a non-negative integer.")
        candidate_id = _required(self.candidate_id, name="candidate_id")
        object.__setattr__(self, "input_role", role)
        object.__setattr__(self, "effective_from", effective_from)
        object.__setattr__(self, "effective_until", effective_until)
        object.__setattr__(self, "candidate_id", candidate_id)

    @property
    def temporal_rank(self) -> tuple[float, int, str]:
        return (
            self.effective_from,
            self.selection_sequence,
            self.candidate_id,
        )


@dataclass(frozen=True)
class CompositionInputSelection(Generic[T]):
    """Result of one target-time and role-dependency selection."""

    target_effective_at: float
    selected_by_role: Mapping[str, T]
    selected_before_dependency_by_role: Mapping[str, T]
    dependency_ineligibility_reason_by_role: Mapping[str, str]
    missing_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        target = _finite(
            self.target_effective_at,
            name="target_effective_at",
        )
        selected = dict(self.selected_by_role)
        before_dependency = dict(self.selected_before_dependency_by_role)
        reasons = {
            _required(role, name="dependency role"): _required(
                reason,
                name="dependency reason",
            )
            for role, reason in dict(
                self.dependency_ineligibility_reason_by_role
            ).items()
        }
        missing = tuple(
            _required(role, name="missing_role") for role in self.missing_roles
        )
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(
            self,
            "selected_by_role",
            MappingProxyType(selected),
        )
        object.__setattr__(
            self,
            "selected_before_dependency_by_role",
            MappingProxyType(before_dependency),
        )
        object.__setattr__(
            self,
            "dependency_ineligibility_reason_by_role",
            MappingProxyType(reasons),
        )
        object.__setattr__(self, "missing_roles", missing)


def select_composition_input_candidates(
    candidates: tuple[CompositionInputSelectionCandidate[T], ...],
    *,
    input_roles: tuple[str, ...],
    depends_on_roles_by_role: Mapping[str, tuple[str, ...]],
    target_effective_at: float,
    overlap_behavior: str,
) -> CompositionInputSelection[T]:
    """Select target-covering candidates without touching any runtime owner."""

    target = _finite(
        target_effective_at,
        name="target_effective_at",
    )
    roles = tuple(_required(role, name="input_role") for role in input_roles)
    if not roles or len(roles) != len(set(roles)):
        raise ValueError("input_roles must be non-empty and unique.")
    dependencies = {
        _required(role, name="dependency role"): tuple(
            _required(value, name="dependency") for value in values
        )
        for role, values in dict(depends_on_roles_by_role).items()
    }
    unknown_dependency_roles = (
        set(dependencies) | {
            dependency
            for values in dependencies.values()
            for dependency in values
        }
    ) - set(roles)
    if unknown_dependency_roles:
        raise ValueError("Role dependencies must reference declared input roles.")
    behavior = _required(overlap_behavior, name="overlap_behavior")
    if behavior not in {
        "latest_effective_start_then_receiver_admission",
        "reject_multiple_covering",
    }:
        raise ValueError("Unsupported overlap behavior.")

    eligible_by_role: dict[
        str,
        list[CompositionInputSelectionCandidate[T]],
    ] = {role: [] for role in roles}
    for candidate in tuple(candidates):
        if not isinstance(candidate, CompositionInputSelectionCandidate):
            raise TypeError(
                "candidates must contain CompositionInputSelectionCandidate."
            )
        if candidate.input_role not in eligible_by_role:
            continue
        if candidate.effective_from <= target < candidate.effective_until:
            eligible_by_role[candidate.input_role].append(candidate)

    selected: dict[str, T] = {}
    for role in roles:
        eligible = eligible_by_role[role]
        if not eligible:
            continue
        if behavior == "reject_multiple_covering" and len(eligible) > 1:
            raise ValueError(
                "Multiple composition inputs cover one role/target under "
                f"reject_multiple_covering: {role!r}."
            )
        selected[role] = max(
            eligible,
            key=lambda candidate: candidate.temporal_rank,
        ).carrier

    selected_before_dependency = dict(selected)
    dependency_ineligibility: dict[str, str] = {}
    dependency_changed = True
    while dependency_changed:
        dependency_changed = False
        for role in roles:
            if role not in selected:
                continue
            missing_dependencies = tuple(
                dependency
                for dependency in dependencies.get(role, ())
                if dependency not in selected
            )
            if not missing_dependencies:
                continue
            del selected[role]
            dependency_ineligibility[role] = (
                "prerequisite_unavailable:" + ",".join(missing_dependencies)
            )
            dependency_changed = True

    return CompositionInputSelection(
        target_effective_at=target,
        selected_by_role=selected,
        selected_before_dependency_by_role=selected_before_dependency,
        dependency_ineligibility_reason_by_role=dependency_ineligibility,
        missing_roles=tuple(role for role in roles if role not in selected),
    )


__all__ = [
    "CompositionInputSelection",
    "CompositionInputSelectionCandidate",
    "select_composition_input_candidates",
]
