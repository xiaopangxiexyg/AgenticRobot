"""Typed fail-closed authority scope for one compound proposal producer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from .components import ActionComponentCatalogV2
from .proposal import ActionProposalDraft

if TYPE_CHECKING:
    from .authority import AuthorityPolicySpec


CONSERVATIVE_COMPOUND_AUTHORITY_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.conservative_compound_authority_profile.v1"
)
CONSERVATIVE_COMPOUND_AUTHORITY_ATOMICITY_V1 = (
    "full_control_domain_all_required_v1"
)


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _identifiers(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, name=name) for value in values)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be non-empty and unique.")
    return normalized


@dataclass(frozen=True)
class ConservativeCompoundAuthorityProfile:
    """Exact all-required proposal and activation contract.

    The profile is backend-neutral.  It does not grant authority by itself;
    ``AuthorityPolicySpec`` still decides which producer may be selected.
    When bound to an authoritative runtime, every proposal from ``producer_id``
    must cover the complete declared component/group scope.  Missing, stale,
    invalid, unsafe, rejected, or partially receipted work therefore falls
    back to one control-domain HOLD rather than a partial commit.
    """

    profile_id: str
    producer_id: str
    component_coverage: tuple[str, ...]
    selection_groups: tuple[str, ...]
    composition_spec_hash: str
    component_rule_set_hash: str
    formal_strategy_spec_hash: str
    atomicity_mode: str = CONSERVATIVE_COMPOUND_AUTHORITY_ATOMICITY_V1
    missing_or_stale_behavior: str = "hold_control_domain"
    safety_failure_behavior: str = "hold_control_domain"
    downstream_failure_behavior: str = "hold_control_domain"
    activation_commit_mode: str = "all_required_groups_atomic"
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload: dict[str, Any] = {
            "profile_id": _identifier(self.profile_id, name="profile_id"),
            "producer_id": _identifier(self.producer_id, name="producer_id"),
            "component_coverage": _identifiers(
                self.component_coverage,
                name="component_coverage",
            ),
            "selection_groups": _identifiers(
                self.selection_groups,
                name="selection_groups",
            ),
            "composition_spec_hash": _identifier(
                self.composition_spec_hash,
                name="composition_spec_hash",
            ),
            "component_rule_set_hash": _identifier(
                self.component_rule_set_hash,
                name="component_rule_set_hash",
            ),
            "formal_strategy_spec_hash": _identifier(
                self.formal_strategy_spec_hash,
                name="formal_strategy_spec_hash",
            ),
            "atomicity_mode": _identifier(
                self.atomicity_mode,
                name="atomicity_mode",
            ),
            "missing_or_stale_behavior": _identifier(
                self.missing_or_stale_behavior,
                name="missing_or_stale_behavior",
            ),
            "safety_failure_behavior": _identifier(
                self.safety_failure_behavior,
                name="safety_failure_behavior",
            ),
            "downstream_failure_behavior": _identifier(
                self.downstream_failure_behavior,
                name="downstream_failure_behavior",
            ),
            "activation_commit_mode": _identifier(
                self.activation_commit_mode,
                name="activation_commit_mode",
            ),
        }
        if payload["atomicity_mode"] != (
            CONSERVATIVE_COMPOUND_AUTHORITY_ATOMICITY_V1
        ):
            raise ValueError("Unsupported compound authority atomicity mode.")
        for field_name in (
            "missing_or_stale_behavior",
            "safety_failure_behavior",
            "downstream_failure_behavior",
        ):
            if payload[field_name] != "hold_control_domain":
                raise ValueError(
                    "Conservative compound authority requires control-domain "
                    f"HOLD for {field_name}."
                )
        if payload["activation_commit_mode"] != "all_required_groups_atomic":
            raise ValueError(
                "Conservative compound authority requires atomic group activation."
            )
        for field_name, value in payload.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    CONSERVATIVE_COMPOUND_AUTHORITY_PROFILE_HASH_DOMAIN_V1
                ),
            ),
        )

    @property
    def profile_fact_id(self) -> str:
        return (
            "conservative-compound-authority-profile:"
            f"{self.semantic_content_hash}"
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "producer_id": self.producer_id,
            "component_coverage": self.component_coverage,
            "selection_groups": self.selection_groups,
            "composition_spec_hash": self.composition_spec_hash,
            "component_rule_set_hash": self.component_rule_set_hash,
            "formal_strategy_spec_hash": self.formal_strategy_spec_hash,
            "atomicity_mode": self.atomicity_mode,
            "missing_or_stale_behavior": self.missing_or_stale_behavior,
            "safety_failure_behavior": self.safety_failure_behavior,
            "downstream_failure_behavior": self.downstream_failure_behavior,
            "activation_commit_mode": self.activation_commit_mode,
        }

    def validate_runtime_binding(
        self,
        *,
        component_catalog: ActionComponentCatalogV2,
        active_selection_groups: tuple[str, ...],
        authority_policy: "AuthorityPolicySpec",
    ) -> None:
        """Require exact full-domain catalog and authority-policy binding."""

        catalog_components = tuple(
            component.component_id for component in component_catalog.components
        )
        catalog_groups = tuple(component_catalog.selection_group_ids)
        active_groups = tuple(active_selection_groups)
        if self.component_coverage != catalog_components:
            raise ValueError(
                "Conservative compound authority must cover the complete "
                "component catalog in catalog order."
            )
        if (
            self.selection_groups != catalog_groups
            or self.selection_groups != active_groups
        ):
            raise ValueError(
                "Conservative compound authority must cover every active "
                "selection group in catalog order."
            )
        if len(self.selection_groups) < 2:
            raise ValueError(
                "Conservative compound authority requires multiple groups."
            )
        for group in self.selection_groups:
            rules = authority_policy.priority_by_group.get(group, {})
            if set(rules) != {self.producer_id}:
                raise ValueError(
                    "Conservative compound authority requires its producer to "
                    "be the only authorized selector in every required group."
                )
            if authority_policy.fallback_by_group.get(group) != "hold":
                raise ValueError(
                    "Conservative compound authority requires HOLD fallback "
                    "for every required group."
                )
        disallowed_takeover = set(
            authority_policy.takeover_allowed_producers
        ) - {self.producer_id}
        if disallowed_takeover:
            raise ValueError(
                "Conservative compound authority cannot allow another "
                "producer to acquire a required-group lease."
            )

    def validate_proposal_draft(
        self,
        draft: ActionProposalDraft,
        *,
        component_catalog: ActionComponentCatalogV2,
    ) -> str | None:
        """Return a stable rejection reason for a non-exact proposal."""

        if draft.producer_id != self.producer_id:
            return "producer_id_mismatch"
        if draft.producer_kind != "composer":
            return "producer_kind_mismatch"
        if tuple(draft.action.component_coverage) != self.component_coverage:
            return "component_coverage_mismatch"
        selection_groups = tuple(
            dict.fromkeys(
                component_catalog.component_by_id[
                    component_id
                ].selection_group_id
                for component_id in draft.action.component_coverage
            )
        )
        if selection_groups != self.selection_groups:
            return "selection_groups_mismatch"
        if draft.composition_spec_hash != self.composition_spec_hash:
            return "composition_spec_hash_mismatch"
        if (
            not draft.composition_input_ids
            or not draft.composition_evaluation_id
            or not draft.composition_evaluation_hash
        ):
            return "formal_composition_lineage_missing"
        return None


__all__ = [
    "CONSERVATIVE_COMPOUND_AUTHORITY_ATOMICITY_V1",
    "CONSERVATIVE_COMPOUND_AUTHORITY_PROFILE_HASH_DOMAIN_V1",
    "ConservativeCompoundAuthorityProfile",
]
