"""Versioned action-component and selection-group contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash


ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V1 = "dexteleop.action_component_catalog.v1"


@dataclass(frozen=True)
class ActionComponentSpec:
    component_id: str
    selection_group_id: str
    effector_name: str
    coupled: bool = False

    def __post_init__(self) -> None:
        for name in ("component_id", "selection_group_id", "effector_name"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ActionComponentCatalog:
    catalog_id: str
    action_space: str
    components: tuple[ActionComponentSpec, ...]

    def __post_init__(self) -> None:
        if not str(self.catalog_id).strip() or not str(self.action_space).strip():
            raise ValueError("catalog_id and action_space must be non-empty.")
        object.__setattr__(self, "components", tuple(self.components))
        ids = tuple(item.component_id for item in self.components)
        effectors = tuple(item.effector_name for item in self.components)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Component IDs must be non-empty and unique.")
        if len(effectors) != len(set(effectors)):
            raise ValueError("Each effector may belong to only one component.")
        for group in self.selection_group_ids:
            members = self.components_for_group(group)
            coupled_values = {member.coupled for member in members}
            if len(coupled_values) != 1:
                raise ValueError("All components in one selection group must agree on coupled.")

    @property
    def selection_group_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.selection_group_id for item in self.components))

    @property
    def component_by_id(self) -> Mapping[str, ActionComponentSpec]:
        return MappingProxyType({item.component_id: item for item in self.components})

    def components_for_group(self, group_id: str) -> tuple[ActionComponentSpec, ...]:
        group = str(group_id)
        return tuple(item for item in self.components if item.selection_group_id == group)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "catalog_id": self.catalog_id,
                "action_space": self.action_space,
                "components": tuple(
                    {
                        "component_id": item.component_id,
                        "selection_group_id": item.selection_group_id,
                        "effector_name": item.effector_name,
                        "coupled": item.coupled,
                    }
                    for item in self.components
                ),
            },
            domain=ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V1,
        )


__all__ = [
    "ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V1",
    "ActionComponentCatalog",
    "ActionComponentSpec",
]
