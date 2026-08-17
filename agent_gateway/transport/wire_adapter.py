"""Transport-neutral wire facade over one bound Agent Gateway session.

Authentication and byte transport remain outside this module.  One facade is
constructed for one already-authenticated client identity; payloads cannot
select or impersonate another client.  Submission still performs only bounded
admission in the service, then asks the exact Route launcher to schedule its
registered executor.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from ..protocol._contract_validation import (
    finite_number as _finite_number,
    nonnegative_integer as _nonnegative_integer,
    positive_integer as _positive_integer,
    required_text as _required,
)
from ..protocol.artifact_contracts import (
    AgentArtifactMaterializationReceipt,
    AgentArtifactRef,
)
from ..protocol.contracts import (
    AgentSkillRequest,
    agent_own_domain_hold_request_from_json,
    agent_own_operation_cancel_request_from_json,
    agent_own_operation_query_from_json,
    agent_skill_request_from_json,
)
from ..protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    validate_skill_route_catalog_closure,
)
from ..runtime.access_policy import SkillExecutionGrant
from ..runtime.service import (
    AgentGatewayObservationOutcome,
    AgentGatewayOperationOutcome,
    AgentGatewayService,
)


def _revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "current_station_revision provider must return a non-negative integer."
        )
    return value


def _exact_fields(payload: Mapping[str, Any], fields: set[str]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError(
            "Gateway payload fields must match the method contract exactly."
        )


def _decode_base64(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("data_base64 must be a non-empty base64 string.")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("data_base64 is not canonical base64.") from error


def _upload_lease_to_wire(lease: Any) -> Mapping[str, Any]:
    return {
        "upload_id": lease.upload_id,
        "artifact": lease.artifact.to_wire(),
        "state": lease.state,
        "received_bytes": lease.received_bytes,
        "created_at": lease.created_at,
        "expires_at": lease.expires_at,
    }


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

        protocol_runtime = service.protocol_runtime
        if protocol_runtime.io_catalog.semantic_content_hash != (
            service.manifest.skill_io_catalog_hash
        ):
            raise ValueError("Skill I/O catalog does not match service manifest.")
        validate_skill_route_catalog_closure(
            definition_catalog=definition_catalog,
            schema_catalog=schema_catalog,
            route_manifest=service.manifest,
            io_catalog=protocol_runtime.io_catalog,
        )

    def _revision(self) -> int:
        return _revision(self._current_station_revision())

    def _assert_client(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("Gateway payload must be a JSON object.")
        if payload.get("client_id") != self.authenticated_client_id:
            raise PermissionError(
                "Payload client_id does not match authenticated connection."
            )

    def _assert_session(
        self,
        payload: Mapping[str, Any],
        *,
        fields: set[str],
    ) -> None:
        _exact_fields(payload, fields)
        self._assert_client(payload)
        if payload.get("runtime_session_id") != (
            self.service.protocol_runtime.runtime_session_id
        ):
            raise PermissionError("Payload runtime_session_id does not match.")

    def get_skill_definition_catalog(self) -> Mapping[str, Any]:
        return self.definition_catalog.to_wire()

    def get_agent_skill_schema_catalog(self) -> Mapping[str, Any]:
        return self.schema_catalog.to_wire()

    def get_skill_route_manifest(self) -> Mapping[str, Any]:
        return self.service.manifest.to_wire()

    def get_skill_io_catalog(self) -> Mapping[str, Any]:
        return self.service.protocol_runtime.io_catalog.to_wire()

    def get_gateway_session_capabilities(self) -> Mapping[str, Any]:
        return self.service.protocol_runtime.session_capabilities.to_wire()

    def get_gateway_events(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_session(
            payload,
            fields={
                "client_id",
                "runtime_session_id",
                "after_sequence",
                "max_events",
            },
        )
        batch = self.service.protocol_runtime.replay_events(
            after_sequence=_nonnegative_integer(
                payload["after_sequence"],
                field_name="after_sequence",
            ),
            max_events=_positive_integer(
                payload["max_events"],
                field_name="max_events",
            ),
            operations=self.service.operation_store.operations(),
        )
        return batch.to_wire()

    def begin_artifact_upload(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_session(
            payload,
            fields={
                "client_id",
                "runtime_session_id",
                "idempotency_key",
                "artifact",
                "mechanism",
                "ttl_sec",
            },
        )
        store, broker = self.service.protocol_runtime.require_artifact_plane()
        artifact = AgentArtifactRef.from_wire(payload["artifact"])
        lease = store.begin_upload(
            client_id=self.authenticated_client_id,
            runtime_session_id=payload["runtime_session_id"],
            idempotency_key=_required(
                payload["idempotency_key"],
                field_name="idempotency_key",
            ),
            artifact=artifact,
            now=self.service.protocol_runtime.now(),
        )
        grant = None
        if not lease.committed:
            grant = broker.issue_write_grant(
                lease=lease,
                mechanism=_required(payload["mechanism"], field_name="mechanism"),
                now=self.service.protocol_runtime.now(),
                ttl_sec=_finite_number(payload["ttl_sec"], field_name="ttl_sec"),
            )
        return {
            "upload": _upload_lease_to_wire(lease),
            "access_grant": None if grant is None else grant.to_wire(),
        }

    def write_artifact_chunk(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_session(
            payload,
            fields={
                "client_id",
                "runtime_session_id",
                "grant_id",
                "offset",
                "data_base64",
            },
        )
        _store, broker = self.service.protocol_runtime.require_artifact_plane()
        lease = broker.write_chunk(
            grant_id=_required(payload["grant_id"], field_name="grant_id"),
            client_id=self.authenticated_client_id,
            runtime_session_id=payload["runtime_session_id"],
            offset=_nonnegative_integer(payload["offset"], field_name="offset"),
            data=_decode_base64(payload["data_base64"]),
            now=self.service.protocol_runtime.now(),
        )
        return _upload_lease_to_wire(lease)

    def commit_artifact_upload(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_session(
            payload,
            fields={"client_id", "runtime_session_id", "grant_id"},
        )
        _store, broker = self.service.protocol_runtime.require_artifact_plane()
        artifact, receipt = broker.commit_write(
            grant_id=_required(payload["grant_id"], field_name="grant_id"),
            client_id=self.authenticated_client_id,
            runtime_session_id=payload["runtime_session_id"],
            now=self.service.protocol_runtime.now(),
        )
        return {"artifact": artifact.to_wire(), "receipt": receipt.to_wire()}

    def begin_artifact_download(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_session(
            payload,
            fields={
                "client_id",
                "runtime_session_id",
                "artifact_id",
                "mechanism",
                "ttl_sec",
            },
        )
        _store, broker = self.service.protocol_runtime.require_artifact_plane()
        grant = broker.issue_read_grant(
            client_id=self.authenticated_client_id,
            runtime_session_id=payload["runtime_session_id"],
            artifact_id=_required(payload["artifact_id"], field_name="artifact_id"),
            mechanism=_required(payload["mechanism"], field_name="mechanism"),
            now=self.service.protocol_runtime.now(),
            ttl_sec=_finite_number(payload["ttl_sec"], field_name="ttl_sec"),
        )
        return grant.to_wire()

    def read_artifact_chunk(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_session(
            payload,
            fields={
                "client_id",
                "runtime_session_id",
                "grant_id",
                "offset",
                "max_bytes",
            },
        )
        _store, broker = self.service.protocol_runtime.require_artifact_plane()
        offset = _nonnegative_integer(payload["offset"], field_name="offset")
        data = broker.read_chunk(
            grant_id=_required(payload["grant_id"], field_name="grant_id"),
            client_id=self.authenticated_client_id,
            runtime_session_id=payload["runtime_session_id"],
            offset=offset,
            max_bytes=_positive_integer(payload["max_bytes"], field_name="max_bytes"),
            now=self.service.protocol_runtime.now(),
        )
        return {
            "offset": offset,
            "byte_length": len(data),
            "data_base64": base64.b64encode(data).decode("ascii"),
        }

    def acknowledge_artifact_download(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_session(
            payload,
            fields={
                "client_id",
                "runtime_session_id",
                "grant_id",
                "transferred_bytes",
                "verified_sha256",
            },
        )
        _store, broker = self.service.protocol_runtime.require_artifact_plane()
        receipt = broker.acknowledge_read(
            grant_id=_required(payload["grant_id"], field_name="grant_id"),
            client_id=self.authenticated_client_id,
            runtime_session_id=payload["runtime_session_id"],
            transferred_bytes=_nonnegative_integer(
                payload["transferred_bytes"],
                field_name="transferred_bytes",
            ),
            verified_sha256=_required(
                payload["verified_sha256"],
                field_name="verified_sha256",
            ),
            now=self.service.protocol_runtime.now(),
        )
        return receipt.to_wire()

    def acknowledge_artifact_materialization(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._assert_session(
            payload,
            fields={
                "client_id",
                "runtime_session_id",
                "receipt_id",
                "artifact_id",
                "durable_store_id",
                "durable_object_ref",
                "materialized_at",
                "verified_sha256",
                "materialization_receipt_hash",
            },
        )
        store, _broker = self.service.protocol_runtime.require_artifact_plane()
        receipt = AgentArtifactMaterializationReceipt(
            receipt_id=payload["receipt_id"],
            artifact_id=payload["artifact_id"],
            client_id=self.authenticated_client_id,
            runtime_session_id=payload["runtime_session_id"],
            durable_store_id=payload["durable_store_id"],
            durable_object_ref=payload["durable_object_ref"],
            materialized_at=payload["materialized_at"],
            verified_sha256=payload["verified_sha256"],
        )
        if payload["materialization_receipt_hash"] != (receipt.semantic_content_hash):
            raise ValueError("Materialization receipt hash does not match.")
        store.record_materialization(receipt)
        return receipt.to_wire()

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
        if grant is not None and request.skill_execution_grant_id is None:
            # An application-local operator resolver may atomically mint the
            # exact request-bound one-use grant.  The external client never
            # receives reusable grant material and cannot select another ID.
            request = replace(
                request,
                skill_execution_grant_id=grant.grant_id,
            )
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
