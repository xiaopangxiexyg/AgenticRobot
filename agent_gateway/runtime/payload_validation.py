"""Server-side payload validators bound to advertised Skill Route schemas.

Schema hashes make payload semantics discoverable, but a matching hash does not
validate a request or result body.  This module binds executable validators to
the exact Route identity and schema hashes used by one Gateway service.  It has
no transport, worker, lower-runtime, simulator, or device dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..protocol.skill_catalog import SkillExecutionRoute, SkillRouteManifest

PayloadValidator = Callable[[Mapping[str, Any]], None]
ContextualResultPayloadValidator = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    None,
]


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


@dataclass(frozen=True)
class AgentSkillPayloadValidatorSet:
    """Executable validators for one exact Skill Route schema triple."""

    validator_id: str
    skill_id: str
    skill_version: str
    route_id: str
    route_version: str
    parameter_schema_hash: str
    execution_result_schema_hash: str
    task_evaluation_schema_hash: str
    validate_parameters: PayloadValidator
    validate_execution_result: PayloadValidator
    validate_task_evaluation: PayloadValidator
    validate_execution_result_for_parameters: (
        ContextualResultPayloadValidator | None
    ) = None

    def __post_init__(self) -> None:
        for field_name in (
            "validator_id",
            "skill_id",
            "skill_version",
            "route_id",
            "route_version",
            "parameter_schema_hash",
            "execution_result_schema_hash",
            "task_evaluation_schema_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "validate_parameters",
            "validate_execution_result",
            "validate_task_evaluation",
        ):
            if not callable(getattr(self, field_name)):
                raise TypeError(f"{field_name} must be callable.")
        contextual = self.validate_execution_result_for_parameters
        if contextual is not None and not callable(contextual):
            raise TypeError(
                "validate_execution_result_for_parameters must be callable or None."
            )

    @property
    def route_key(self) -> tuple[str, str, str, str]:
        return (
            self.skill_id,
            self.skill_version,
            self.route_id,
            self.route_version,
        )

    def validate_route(self, route: SkillExecutionRoute) -> None:
        if not isinstance(route, SkillExecutionRoute):
            raise TypeError("route must be a SkillExecutionRoute.")
        if self.route_key != (
            route.skill_id,
            route.skill_version,
            route.route_id,
            route.route_version,
        ):
            raise ValueError("Payload validator Route identity mismatch.")
        for field_name in (
            "parameter_schema_hash",
            "execution_result_schema_hash",
            "task_evaluation_schema_hash",
        ):
            if getattr(self, field_name) != getattr(route, field_name):
                raise ValueError(
                    f"Payload validator {field_name} does not match the Route."
                )


class AgentSkillPayloadValidatorRegistry:
    """Immutable lookup that must cover every authoritative advertised Route."""

    def __init__(
        self,
        *,
        registry_id: str,
        validators: tuple[AgentSkillPayloadValidatorSet, ...],
    ) -> None:
        self.registry_id = _required(registry_id, field_name="registry_id")
        normalized = tuple(validators)
        if not normalized:
            raise ValueError("validators must not be empty.")
        if any(
            not isinstance(item, AgentSkillPayloadValidatorSet)
            for item in normalized
        ):
            raise TypeError(
                "validators must contain AgentSkillPayloadValidatorSet values."
            )
        by_key = {item.route_key: item for item in normalized}
        if len(by_key) != len(normalized):
            raise ValueError("Payload validator Route identities must be unique.")
        self._validators_by_key = by_key

    @staticmethod
    def _route_key(route: SkillExecutionRoute) -> tuple[str, str, str, str]:
        return (
            route.skill_id,
            route.skill_version,
            route.route_id,
            route.route_version,
        )

    def validator_for(
        self,
        route: SkillExecutionRoute,
    ) -> AgentSkillPayloadValidatorSet:
        if not isinstance(route, SkillExecutionRoute):
            raise TypeError("route must be a SkillExecutionRoute.")
        try:
            validator = self._validators_by_key[self._route_key(route)]
        except KeyError as error:
            raise KeyError(
                "No payload validator is registered for the exact Route."
            ) from error
        validator.validate_route(route)
        return validator

    def validate_manifest(self, manifest: SkillRouteManifest) -> None:
        if not isinstance(manifest, SkillRouteManifest):
            raise TypeError("manifest must be a SkillRouteManifest.")
        for entry in manifest.skill_routes:
            self.validator_for(entry.route)

    def validate_parameters(
        self,
        *,
        route: SkillExecutionRoute,
        payload: Mapping[str, Any],
    ) -> None:
        self.validator_for(route).validate_parameters(payload)

    def validate_execution_result(
        self,
        *,
        route: SkillExecutionRoute,
        payload: Mapping[str, Any],
        request_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        validator = self.validator_for(route)
        validator.validate_execution_result(payload)
        contextual = validator.validate_execution_result_for_parameters
        if contextual is not None:
            if not isinstance(request_parameters, Mapping):
                raise TypeError(
                    "Contextual result validation requires request_parameters."
                )
            contextual(request_parameters, payload)

    def validate_task_evaluation(
        self,
        *,
        route: SkillExecutionRoute,
        payload: Mapping[str, Any],
    ) -> None:
        self.validator_for(route).validate_task_evaluation(payload)


__all__ = [
    "AgentSkillPayloadValidatorRegistry",
    "AgentSkillPayloadValidatorSet",
    "ContextualResultPayloadValidator",
    "PayloadValidator",
]
