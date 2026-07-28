"""Versioned action-component and selection-group contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.action_schema import ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V2
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


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _identifiers(
    values: tuple[str, ...] | list[str],
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    normalized = tuple(_required(value, name=name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique.")
    return normalized


@dataclass(frozen=True)
class ControlComponentSpecV2:
    """Backend-neutral identity and resource scope for one control component.

    A component is a logical command channel, not a robot side or a fixed
    joint layout.  Several components may refer to the same controlled entity
    (for example an arm TCP and its gripper), while ``resource_claims`` makes
    possible command conflicts explicit for downstream capability checks.
    """

    component_id: str
    controlled_entity_id: str
    command_channel_id: str
    selection_group_id: str
    resource_claims: tuple[str, ...]
    accepted_payload_schema_hashes: tuple[str, ...]
    atomic_group_id: str | None = None
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, name="component_id")
        controlled_entity_id = _required(
            self.controlled_entity_id,
            name="controlled_entity_id",
        )
        command_channel_id = _required(
            self.command_channel_id,
            name="command_channel_id",
        )
        selection_group_id = _required(
            self.selection_group_id,
            name="selection_group_id",
        )
        resource_claims = _identifiers(
            tuple(self.resource_claims),
            name="resource_claim",
            allow_empty=False,
        )
        accepted = _identifiers(
            tuple(self.accepted_payload_schema_hashes),
            name="accepted_payload_schema_hash",
            allow_empty=False,
        )
        atomic_group_id = (
            None
            if self.atomic_group_id is None
            else _required(self.atomic_group_id, name="atomic_group_id")
        )
        dependencies = _identifiers(
            tuple(self.dependencies),
            name="component dependency",
        )
        if component_id in dependencies:
            raise ValueError("A control component cannot depend on itself.")
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "controlled_entity_id", controlled_entity_id)
        object.__setattr__(self, "command_channel_id", command_channel_id)
        object.__setattr__(self, "selection_group_id", selection_group_id)
        object.__setattr__(self, "resource_claims", resource_claims)
        object.__setattr__(
            self,
            "accepted_payload_schema_hashes",
            accepted,
        )
        object.__setattr__(self, "atomic_group_id", atomic_group_id)
        object.__setattr__(self, "dependencies", dependencies)

    def semantic_payload(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "controlled_entity_id": self.controlled_entity_id,
            "command_channel_id": self.command_channel_id,
            "selection_group_id": self.selection_group_id,
            "resource_claims": self.resource_claims,
            "accepted_payload_schema_hashes": (
                self.accepted_payload_schema_hashes
            ),
            "atomic_group_id": self.atomic_group_id,
            "dependencies": self.dependencies,
        }


@dataclass(frozen=True)
class ActionComponentCatalogV2:
    """Arbitrary-topology component catalog for canonical component actions."""

    catalog_id: str
    catalog_version: str
    components: tuple[ControlComponentSpecV2, ...]

    def __post_init__(self) -> None:
        catalog_id = _required(self.catalog_id, name="catalog_id")
        catalog_version = _required(
            self.catalog_version,
            name="catalog_version",
        )
        components = tuple(self.components)
        if not components or any(
            not isinstance(item, ControlComponentSpecV2)
            for item in components
        ):
            raise ValueError(
                "components must contain at least one ControlComponentSpecV2."
            )
        component_ids = tuple(item.component_id for item in components)
        channel_ids = tuple(item.command_channel_id for item in components)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("Component IDs must be unique.")
        if len(channel_ids) != len(set(channel_ids)):
            raise ValueError("Command channel IDs must be unique.")
        known = set(component_ids)
        for item in components:
            unknown = sorted(set(item.dependencies) - known)
            if unknown:
                raise ValueError(
                    f"Component {item.component_id!r} has unknown dependencies: "
                    f"{unknown}."
                )
        for atomic_group_id in {
            item.atomic_group_id
            for item in components
            if item.atomic_group_id is not None
        }:
            selection_groups = {
                item.selection_group_id
                for item in components
                if item.atomic_group_id == atomic_group_id
            }
            if len(selection_groups) != 1:
                raise ValueError(
                    "All components in one atomic group must share one "
                    "selection group."
                )
        self._validate_dependency_graph(components)
        object.__setattr__(self, "catalog_id", catalog_id)
        object.__setattr__(self, "catalog_version", catalog_version)
        object.__setattr__(self, "components", components)

    @staticmethod
    def _validate_dependency_graph(
        components: tuple[ControlComponentSpecV2, ...],
    ) -> None:
        dependencies = {
            item.component_id: item.dependencies for item in components
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(component_id: str) -> None:
            if component_id in visited:
                return
            if component_id in visiting:
                raise ValueError("Control component dependencies must be acyclic.")
            visiting.add(component_id)
            for dependency in dependencies[component_id]:
                visit(dependency)
            visiting.remove(component_id)
            visited.add(component_id)

        for component_id in dependencies:
            visit(component_id)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(item.component_id for item in self.components)

    @property
    def component_by_id(self) -> Mapping[str, ControlComponentSpecV2]:
        return MappingProxyType(
            {item.component_id: item for item in self.components}
        )

    @property
    def selection_group_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(item.selection_group_id for item in self.components)
        )

    @property
    def atomic_group_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                item.atomic_group_id
                for item in self.components
                if item.atomic_group_id is not None
            )
        )

    def components_for_selection_group(
        self,
        group_id: str,
    ) -> tuple[ControlComponentSpecV2, ...]:
        requested = _required(group_id, name="selection_group_id")
        return tuple(
            item
            for item in self.components
            if item.selection_group_id == requested
        )

    def components_for_atomic_group(
        self,
        group_id: str,
    ) -> tuple[ControlComponentSpecV2, ...]:
        requested = _required(group_id, name="atomic_group_id")
        return tuple(
            item for item in self.components if item.atomic_group_id == requested
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "catalog_id": self.catalog_id,
                "catalog_version": self.catalog_version,
                "components": tuple(
                    item.semantic_payload() for item in self.components
                ),
            },
            domain=ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V2,
        )


__all__ = [
    "ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V1",
    "ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V2",
    "ActionComponentCatalog",
    "ActionComponentCatalogV2",
    "ActionComponentSpec",
    "ControlComponentSpecV2",
]
