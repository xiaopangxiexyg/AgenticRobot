"""Transport-neutral wire facade over one bound Agent Gateway session.

Authentication and byte transport remain outside this module.  One facade is
constructed for one already-authenticated client identity; payloads cannot
select or impersonate another client.  Submission still performs only bounded
admission in the service, then asks the exact Route launcher to schedule its
registered executor.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .access_policy import SkillExecutionGrant
from .contracts import (
    AgentSkillRequest,
    agent_own_domain_hold_request_from_json,
    agent_own_operation_cancel_request_from_json,
    agent_own_operation_query_from_json,
    agent_skill_request_from_json,
)
from .service import (
    AgentGatewayObservationOutcome,
    AgentGatewayOperationOutcome,
    AgentGatewayService,
)
from .skill_catalog import AgentSkillSchemaCatalog, SkillDefinitionCatalog


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "current_station_revision provider must return a non-negative integer."
        )
    return value


class BoundAgentGatewayWireAdapter:
    """One authenticated-client JSON-object facade with no network server."""

    def __init__(
        self,
        *,
        authenticated_client_id: str,
        definition_catalog: SkillDefinitionCatalog,
        schema_catalog: AgentSkillSchemaCatalog,
        service: AgentGatewayService,
        publish_observation: Callable[[], AgentGatewayObservationOutcome],
        current_station_revision: Callable[[], int],
        operation_launcher: Callable[[str], Any] | None = None,
        grant_resolver: (
            Callable[[AgentSkillRequest], SkillExecutionGrant | None] | None
        ) = None,
    ) -> None:
        client_id = _required(
            authenticated_client_id,
            field_name="authenticated_client_id",
        )
        if not isinstance(definition_catalog, SkillDefinitionCatalog):
            raise TypeError("definition_catalog must be typed.")
        if not isinstance(schema_catalog, AgentSkillSchemaCatalog):
            raise TypeError("schema_catalog must be typed.")
        if not isinstance(service, AgentGatewayService):
            raise TypeError("service must be an AgentGatewayService.")
        if not callable(publish_observation):
            raise TypeError("publish_observation must be callable.")
        if not callable(current_station_revision):
            raise TypeError("current_station_revision must be callable.")
        if operation_launcher is not None and not callable(operation_launcher):
            raise TypeError("operation_launcher must be callable or None.")
        if grant_resolver is not None and not callable(grant_resolver):
            raise TypeError("grant_resolver must be callable or None.")
        if service.executor_registry.has_authoritative_execution and (
            operation_launcher is None
        ):
            raise TypeError(
                "An authoritative Gateway requires an explicit operation launcher."
            )
        if service.manifest.skill_definition_catalog_hash != (
            definition_catalog.semantic_content_hash
        ):
            raise ValueError("Definition catalog does not match service manifest.")
        if service.manifest.skill_schema_catalog_hash != (
            schema_catalog.semantic_content_hash
        ):
            raise ValueError("Schema catalog does not match service manifest.")
        if client_id not in service.access_policy.allowed_client_ids:
            raise ValueError(
                "Authenticated client is absent from the Gateway access policy."
            )

        self.authenticated_client_id = client_id
        self.definition_catalog = definition_catalog
        self.schema_catalog = schema_catalog
        self.service = service
        self._publish_observation = publish_observation
        self._current_station_revision = current_station_revision
        self._operation_launcher = operation_launcher
        self._grant_resolver = grant_resolver

    def _revision(self) -> int:
        return _revision(self._current_station_revision())

    def _assert_client(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("Gateway payload must be a JSON object.")
        if payload.get("client_id") != self.authenticated_client_id:
            raise PermissionError(
                "Payload client_id does not match authenticated connection."
            )

    def get_skill_definition_catalog(self) -> Mapping[str, Any]:
        return self.definition_catalog.to_wire()

    def get_agent_skill_schema_catalog(self) -> Mapping[str, Any]:
        return self.schema_catalog.to_wire()

    def get_skill_route_manifest(self) -> Mapping[str, Any]:
        return self.service.manifest.to_wire()

    def get_observation_snapshot(self) -> Mapping[str, Any]:
        outcome = self._publish_observation()
        if not isinstance(outcome, AgentGatewayObservationOutcome):
            raise TypeError("Observation publisher returned an invalid outcome.")
        if not outcome.accepted or outcome.snapshot is None:
            raise RuntimeError(
                "Gateway observation publication failed: " + outcome.reason_code
            )
        return outcome.snapshot.to_wire()

    def submit_skill_request(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_client(payload)
        request = agent_skill_request_from_json(payload)
        grant = (
            None
            if self._grant_resolver is None
            else self._grant_resolver(request)
        )
        if grant is not None and not isinstance(grant, SkillExecutionGrant):
            raise TypeError("grant_resolver returned an invalid grant.")
        outcome = self.service.submit_request(
            request=request,
            current_station_revision=self._revision(),
            grant=grant,
        )
        operation = outcome.operation
        if (
            outcome.accepted
            and operation is not None
            and operation.status == "admitted"
            and self._operation_launcher is not None
        ):
            launch = self._operation_launcher(operation.operation_id)
            if not bool(getattr(launch, "accepted", False)):
                raise RuntimeError(
                    "Station operation launch failed for admitted operation "
                    f"{operation.operation_id}: "
                    f"{getattr(launch, 'status', 'unknown')}"
                )
        return {
            "accepted": outcome.accepted,
            "reason_code": outcome.reason_code,
            "idempotent_replay": outcome.idempotent_replay,
            "operation": None if operation is None else operation.to_wire(),
        }

    def get_own_operation(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_client(payload)
        query = agent_own_operation_query_from_json(payload)
        return self.service.get_own_operation(query=query).to_wire()

    def cancel_own_operation(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_client(payload)
        request = agent_own_operation_cancel_request_from_json(payload)
        return self.service.cancel_own_operation(
            request=request,
            current_station_revision=self._revision(),
        ).to_wire()

    def request_hold_own_domain(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_client(payload)
        request = agent_own_domain_hold_request_from_json(payload)
        return self.service.request_hold_own_domain(
            request=request,
            current_station_revision=self._revision(),
        ).to_wire()

    def handle_authenticated_disconnect(
        self,
        *,
        reason: str,
    ) -> AgentGatewayOperationOutcome:
        return self.service.handle_client_disconnect(
            client_id=self.authenticated_client_id,
            reason=_required(reason, field_name="reason"),
            current_station_revision=self._revision(),
        )

    def close(self) -> AgentGatewayOperationOutcome:
        return self.service.close(
            current_station_revision=self._revision(),
        )


__all__ = ["BoundAgentGatewayWireAdapter"]
