"""MuJoCo-owned policy for promoting composed actions to target proposals.

This module validates target topology, action-schema, and lifetime facts.  It
does not admit proposals, advance producer sequences, grant authority, apply
safety, or dispatch commands; those stateful responsibilities remain in
``MujocoDeployRuntime``.
"""

from __future__ import annotations

from dataclasses import dataclass

from robot_motion.action import (
    ActionComponentCatalog,
    ComponentAction,
    ComponentActionSchema,
    ConservativeCompoundAuthorityProfile,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
)


@dataclass(frozen=True)
class MujocoCompositionPromotionPolicy:
    """Resolve one composed candidate against the active MuJoCo target contract."""

    catalog: ActionComponentCatalog
    action_schema: ComponentActionSchema
    compound_authority_profile: ConservativeCompoundAuthorityProfile | None
    decision_hz: float
    proposal_horizon_sec: float
    reference_activation_profile_id: str

    def validate_scope(
        self,
        *,
        component_coverage: tuple[str, ...] | None,
        selection_groups: tuple[str, ...] | None,
        action: object,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the exact target-owned component and group promotion scope."""

        if not isinstance(action, ComponentAction):
            raise TypeError("Composed candidate must be a ComponentAction.")
        catalog_by_component = self.catalog.component_by_id
        if component_coverage is None and selection_groups is None:
            if self.compound_authority_profile is None:
                # The ordinary composer profile follows the exact typed
                # candidate instead of assuming a bimanual-arm topology.
                # Catalog and coupled-group validation below still reject an
                # unknown or partial physical ownership scope.
                coverage = action.component_coverage
                groups = tuple(
                    dict.fromkeys(
                        catalog_by_component[component].selection_group_id
                        for component in coverage
                        if component in catalog_by_component
                    )
                )
            else:
                coverage = self.compound_authority_profile.component_coverage
                groups = self.compound_authority_profile.selection_groups
        elif component_coverage is None or selection_groups is None:
            raise ValueError(
                "component_coverage and selection_groups must be provided together."
            )
        else:
            coverage = tuple(str(value).strip() for value in component_coverage)
            groups = tuple(str(value).strip() for value in selection_groups)
        if (
            not coverage
            or not groups
            or any(not value for value in (*coverage, *groups))
            or len(coverage) != len(set(coverage))
            or len(groups) != len(set(groups))
        ):
            raise ValueError(
                "Composition promotion scope must be non-empty and unique."
            )
        unknown_components = tuple(
            component for component in coverage if component not in catalog_by_component
        )
        if unknown_components:
            raise ValueError(
                "Composition promotion scope contains components outside the "
                f"active MuJoCo catalog: {unknown_components!r}."
            )
        catalog_order = tuple(
            component.component_id
            for component in self.catalog.components
            if component.component_id in coverage
        )
        if coverage != catalog_order:
            raise ValueError(
                "Composition component_coverage must follow catalog order."
            )
        expected_groups = tuple(
            dict.fromkeys(
                catalog_by_component[component].selection_group_id
                for component in coverage
            )
        )
        if groups != expected_groups:
            raise ValueError(
                "Composition selection_groups must exactly match component "
                "catalog ownership."
            )
        for group in groups:
            members = self.catalog.components_for_group(group)
            if members and members[0].coupled:
                covered_members = {
                    component
                    for component in coverage
                    if catalog_by_component[component].selection_group_id == group
                }
                required_members = {component.component_id for component in members}
                if covered_members != required_members:
                    raise ValueError(
                        "Composition promotion cannot partially cover coupled "
                        f"selection group {group!r}."
                    )
        if (
            action.action_schema.semantic_content_hash
            != self.action_schema.semantic_content_hash
        ):
            raise ValueError("Composed action uses another runtime action schema.")
        action_coverage = action.component_coverage
        if coverage != action_coverage:
            raise ValueError(
                "Composition promotion scope must exactly match the composed "
                "action components."
            )
        if self.compound_authority_profile is not None and (
            coverage != self.compound_authority_profile.component_coverage
            or groups != self.compound_authority_profile.selection_groups
        ):
            raise ValueError(
                "Conservative compound composer authority requires the exact "
                "all-required component/group scope."
            )
        return coverage, groups

    def requested_effective_until(self, *, target_effective_at: float) -> float:
        """Return the target-side proposal deadline before source support capping."""

        horizon = (
            1.0 / self.decision_hz
            if (
                self.compound_authority_profile is not None
                and self.reference_activation_profile_id
                == SCHEDULED_REFERENCE_SEGMENT_SIM_V1
            )
            else self.proposal_horizon_sec
        )
        return target_effective_at + horizon


__all__ = ["MujocoCompositionPromotionPolicy"]
