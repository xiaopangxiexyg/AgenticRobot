"""Small, target-neutral construction template for one complete Skill surface."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .skill_catalog import (
    AgentSkillSchemaCatalog,
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillDefinitionCatalog,
    SkillExecutionRoute,
    validate_skill_definition_route_effect,
)
from .skill_io_contracts import SkillIOCatalog, SkillIOContract


@dataclass(frozen=True)
class SkillRegistrationTemplate:
    """Definition, schemas, I/O, and Routes that must evolve together."""

    definition: SkillDefinition
    schema_entry: AgentSkillSchemaEntry
    io_contract: SkillIOContract
    routes: tuple[SkillExecutionRoute, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.definition, SkillDefinition):
            raise TypeError("definition must be a SkillDefinition.")
        if not isinstance(self.schema_entry, AgentSkillSchemaEntry):
            raise TypeError("schema_entry must be an AgentSkillSchemaEntry.")
        if not isinstance(self.io_contract, SkillIOContract):
            raise TypeError("io_contract must be a SkillIOContract.")
        routes = tuple(self.routes)
        if not routes or any(
            not isinstance(item, SkillExecutionRoute) for item in routes
        ):
            raise ValueError("routes must contain at least one SkillExecutionRoute.")
        identities = tuple((item.route_id, item.route_version) for item in routes)
        if len(set(identities)) != len(identities):
            raise ValueError("Skill registration Route identities must be unique.")

        skill_identity = (
            self.definition.skill_id,
            self.definition.skill_version,
        )
        if (
            self.schema_entry.skill_id,
            self.schema_entry.skill_version,
        ) != skill_identity or (
            self.io_contract.skill_id,
            self.io_contract.skill_version,
        ) != skill_identity:
            raise ValueError("Registration components use different Skill identities.")
        expected_hashes = (
            self.definition.semantic_content_hash,
            self.schema_entry.parameter_schema_hash,
            self.schema_entry.execution_result_schema_hash,
            self.schema_entry.task_evaluation_schema_hash,
            self.io_contract.semantic_content_hash,
        )
        if self.schema_entry.skill_definition_hash != expected_hashes[0]:
            raise ValueError("Schema entry does not bind the Skill definition.")
        io_hashes = (
            self.io_contract.skill_definition_hash,
            self.io_contract.parameter_schema_hash,
            self.io_contract.execution_result_schema_hash,
            self.io_contract.task_evaluation_schema_hash,
        )
        if io_hashes != expected_hashes[:4]:
            raise ValueError("Skill I/O contract does not bind definition/schemas.")
        for route in routes:
            validate_skill_definition_route_effect(
                definition=self.definition,
                route=route,
            )
            route_hashes = (
                route.skill_definition_hash,
                route.parameter_schema_hash,
                route.execution_result_schema_hash,
                route.task_evaluation_schema_hash,
                route.skill_io_contract_hash,
            )
            if (route.skill_id, route.skill_version) != skill_identity:
                raise ValueError("Registration Route uses another Skill identity.")
            if route_hashes != expected_hashes:
                raise ValueError(
                    "Registration Route does not close over Skill contracts."
                )
            allowed = set(route.allowed_parameter_profiles)
            if not allowed <= set(self.schema_entry.parameter_profiles):
                raise ValueError(
                    "Registration Route uses a profile absent from its schema entry."
                )
        object.__setattr__(
            self,
            "routes",
            tuple(sorted(routes, key=lambda item: (item.route_id, item.route_version))),
        )


@dataclass(frozen=True)
class SkillRegistrationCatalogBundle:
    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    io_catalog: SkillIOCatalog


def build_skill_registration_catalogs(
    registrations: Iterable[SkillRegistrationTemplate],
    *,
    catalog_version: str = "1",
    definition_catalog_id: str = "dexteleop.agent_skill_definitions",
    schema_catalog_id: str = "dexteleop.agent_skill_schemas",
    io_catalog_id: str = "dexteleop.agent_skill_io",
) -> SkillRegistrationCatalogBundle:
    """Build the three discovery catalogs from one reviewed registration list."""

    items = tuple(registrations)
    if not items or any(
        not isinstance(item, SkillRegistrationTemplate) for item in items
    ):
        raise ValueError(
            "registrations must contain at least one SkillRegistrationTemplate."
        )
    identities = tuple(
        (item.definition.skill_id, item.definition.skill_version) for item in items
    )
    if len(set(identities)) != len(identities):
        raise ValueError("Skill registration identities must be unique.")
    items = tuple(
        sorted(
            items,
            key=lambda item: (
                item.definition.skill_id,
                item.definition.skill_version,
            ),
        )
    )
    return SkillRegistrationCatalogBundle(
        definition_catalog=SkillDefinitionCatalog(
            catalog_id=definition_catalog_id,
            catalog_version=catalog_version,
            skill_definitions=tuple(item.definition for item in items),
        ),
        schema_catalog=AgentSkillSchemaCatalog(
            catalog_id=schema_catalog_id,
            catalog_version=catalog_version,
            schema_entries=tuple(item.schema_entry for item in items),
        ),
        io_catalog=SkillIOCatalog(
            catalog_id=io_catalog_id,
            catalog_version=catalog_version,
            io_contracts=tuple(item.io_contract for item in items),
        ),
    )


__all__ = [
    "SkillRegistrationCatalogBundle",
    "SkillRegistrationTemplate",
    "build_skill_registration_catalogs",
]
