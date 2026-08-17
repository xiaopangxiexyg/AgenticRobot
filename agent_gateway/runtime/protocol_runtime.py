"""Session-level owner for the single Agent Skill protocol v2 surface."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable

from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.protocol.event_contracts import AgentGatewayEventBatch
from agent_gateway.protocol.session_contracts import AgentGatewaySessionCapabilities
from agent_gateway.protocol.skill_catalog import SkillRouteManifest
from agent_gateway.protocol.skill_io_contracts import (
    SkillIOCatalog,
    inline_only_skill_io_catalog_for_routes,
)

from .artifact_store import RecoverableAgentArtifactStore
from .artifact_transfer import AgentArtifactTransferBroker
from .artifact_validation import validate_skill_artifact_bindings
from .event_journal import BoundedAgentGatewayEventJournal


class AgentGatewayProtocolRuntime:
    """Bind discovery, events, and the optional Artifact plane to one session.

    The runtime is target-neutral.  It owns no robot, camera, simulator,
    dataset, or durable upper-layer storage lifecycle.  Events are bounded
    notification hints; the operation query remains authoritative.
    """

    def __init__(
        self,
        *,
        manifest: SkillRouteManifest,
        io_catalog: SkillIOCatalog,
        session_capabilities: AgentGatewaySessionCapabilities,
        event_journal: BoundedAgentGatewayEventJournal,
        artifact_store: RecoverableAgentArtifactStore | None = None,
        artifact_transfer_broker: AgentArtifactTransferBroker | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(manifest, SkillRouteManifest):
            raise TypeError("manifest must be a SkillRouteManifest.")
        if not isinstance(io_catalog, SkillIOCatalog):
            raise TypeError("io_catalog must be a SkillIOCatalog.")
        if not isinstance(
            session_capabilities,
            AgentGatewaySessionCapabilities,
        ):
            raise TypeError(
                "session_capabilities must be AgentGatewaySessionCapabilities."
            )
        if not isinstance(event_journal, BoundedAgentGatewayEventJournal):
            raise TypeError("event_journal must be BoundedAgentGatewayEventJournal.")
        if not callable(clock):
            raise TypeError("clock must be callable.")

        session_id = manifest.runtime_session_id
        if session_capabilities.runtime_session_id != session_id:
            raise ValueError("Session capabilities do not match the manifest session.")
        if event_journal.runtime_session_id != session_id:
            raise ValueError("Event journal does not match the manifest session.")
        if event_journal.max_events != session_capabilities.event_retention_count:
            raise ValueError("Event retention does not match session capabilities.")
        if (
            event_journal.event_payload_max_bytes
            != session_capabilities.event_payload_max_bytes
        ):
            raise ValueError(
                "Event payload bound does not match session capabilities."
            )
        if manifest.skill_io_catalog_hash != io_catalog.semantic_content_hash:
            raise ValueError("Skill I/O catalog does not match the manifest.")

        has_artifact_slots = False
        for entry in manifest.skill_routes:
            route = entry.route
            contract = io_catalog.contract(route.skill_id, route.skill_version)
            if contract.semantic_content_hash != route.skill_io_contract_hash:
                raise ValueError("Route does not match its Skill I/O contract.")
            has_artifact_slots = has_artifact_slots or bool(
                contract.input_artifact_slots or contract.output_artifact_slots
            )
            for slot in (
                *contract.input_artifact_slots,
                *contract.output_artifact_slots,
            ):
                if slot.max_artifact_bytes > session_capabilities.max_artifact_bytes:
                    raise ValueError(
                        "Skill Artifact slot exceeds the session Artifact bound."
                    )
                if slot.max_count > (
                    session_capabilities.max_artifact_refs_per_operation
                ):
                    raise ValueError(
                        "Skill Artifact slot count exceeds the session reference bound."
                    )

        enabled = session_capabilities.artifact_transfer_enabled
        if has_artifact_slots and not enabled:
            raise ValueError(
                "Artifact-bearing Skills require an Artifact-enabled session."
            )
        if enabled:
            if not isinstance(artifact_store, RecoverableAgentArtifactStore):
                raise TypeError("Artifact-enabled sessions require an Artifact store.")
            if not isinstance(
                artifact_transfer_broker,
                AgentArtifactTransferBroker,
            ):
                raise TypeError(
                    "Artifact-enabled sessions require an Artifact transfer broker."
                )
            if artifact_transfer_broker.store is not artifact_store:
                raise ValueError(
                    "Artifact broker and protocol runtime use different stores."
                )
            if artifact_transfer_broker.session_capabilities != session_capabilities:
                raise ValueError(
                    "Artifact broker capabilities do not match the protocol runtime."
                )
        elif artifact_store is not None or artifact_transfer_broker is not None:
            raise ValueError(
                "Artifact-disabled sessions cannot expose a store or transfer broker."
            )

        self.manifest = manifest
        self.io_catalog = io_catalog
        self.session_capabilities = session_capabilities
        self.event_journal = event_journal
        self.artifact_store = artifact_store
        self.artifact_transfer_broker = artifact_transfer_broker
        self._clock = clock
        self._operation_lock = threading.RLock()
        self._published_operation_hashes: dict[str, str] = {}
        self._published_artifact_bindings: set[tuple[str, str]] = set()

    @classmethod
    def control_plane_only(
        cls,
        *,
        manifest: SkillRouteManifest,
        max_control_request_bytes: int = 2 * 1024 * 1024,
        max_control_response_bytes: int = 2 * 1024 * 1024,
        event_payload_max_bytes: int = 65_536,
        event_retention_count: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> AgentGatewayProtocolRuntime:
        """Create the default v2 profile for Skills with inline JSON only."""

        if not isinstance(manifest, SkillRouteManifest):
            raise TypeError("manifest must be a SkillRouteManifest.")
        io_catalog = inline_only_skill_io_catalog_for_routes(
            entry.route for entry in manifest.skill_routes
        )
        capabilities = AgentGatewaySessionCapabilities.control_plane_only(
            runtime_session_id=manifest.runtime_session_id,
            max_control_request_bytes=max_control_request_bytes,
            max_control_response_bytes=max_control_response_bytes,
            event_payload_max_bytes=event_payload_max_bytes,
            event_retention_count=event_retention_count,
        )
        return cls(
            manifest=manifest,
            io_catalog=io_catalog,
            session_capabilities=capabilities,
            event_journal=BoundedAgentGatewayEventJournal(
                runtime_session_id=manifest.runtime_session_id,
                max_events=event_retention_count,
                event_payload_max_bytes=event_payload_max_bytes,
            ),
            clock=clock,
        )

    @property
    def runtime_session_id(self) -> str:
        return self.manifest.runtime_session_id

    def now(self) -> float:
        return float(self._clock())

    def require_artifact_plane(
        self,
    ) -> tuple[RecoverableAgentArtifactStore, AgentArtifactTransferBroker]:
        if self.artifact_store is None or self.artifact_transfer_broker is None:
            raise RuntimeError("artifact_transfer_not_enabled")
        return self.artifact_store, self.artifact_transfer_broker

    def validate_and_pin_request_inputs(
        self,
        *,
        request: AgentSkillRequest,
        now: float,
    ) -> None:
        """Validate exact input slots and pin available bytes for the request."""

        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        contract = self.io_catalog.contract(
            request.skill_id,
            request.skill_version,
        )
        validate_skill_artifact_bindings(
            io_contract=contract,
            direction="input",
            bindings=request.input_artifact_bindings,
            session_capabilities=self.session_capabilities,
        )
        if not request.input_artifact_bindings:
            return
        store, _broker = self.require_artifact_plane()
        expires_at = float(now) + request.requested_timeout_ms / 1000.0
        for binding in request.input_artifact_bindings:
            record = store.artifact(binding.artifact.artifact_id)
            if not record.available or record.artifact != binding.artifact:
                raise ValueError("Input Artifact is unavailable or does not match.")
        pinned: list[str] = []
        try:
            for binding in request.input_artifact_bindings:
                pin_id = self._input_pin_id(
                    request=request,
                    binding_hash=binding.semantic_content_hash,
                )
                store.pin(
                    pin_id=pin_id,
                    artifact_id=binding.artifact.artifact_id,
                    pin_kind="operation_input",
                    owner_id=request.semantic_content_hash,
                    now=now,
                    expires_at=expires_at,
                )
                pinned.append(pin_id)
        except BaseException:
            for pin_id in pinned:
                store.release_pin(
                    pin_id=pin_id,
                    owner_id=request.semantic_content_hash,
                )
            raise

    @staticmethod
    def _input_pin_id(*, request: AgentSkillRequest, binding_hash: str) -> str:
        return f"pin:input:{request.semantic_content_hash}:{binding_hash}"

    def release_request_input_pins(self, request: AgentSkillRequest) -> None:
        if not request.input_artifact_bindings or self.artifact_store is None:
            return
        for binding in request.input_artifact_bindings:
            self.artifact_store.release_pin(
                pin_id=self._input_pin_id(
                    request=request,
                    binding_hash=binding.semantic_content_hash,
                ),
                owner_id=request.semantic_content_hash,
            )

    def synchronize_operations(
        self,
        operations: Iterable[AgentSkillOperation],
    ) -> None:
        """Project newly observed operation revisions into bounded events."""

        with self._operation_lock:
            for operation in operations:
                if not isinstance(operation, AgentSkillOperation):
                    raise TypeError(
                        "operations must contain AgentSkillOperation values."
                    )
                if operation.runtime_session_id != self.runtime_session_id:
                    raise ValueError("Operation belongs to another runtime session.")
                operation_hash = operation.semantic_content_hash
                if (
                    self._published_operation_hashes.get(operation.operation_id)
                    == operation_hash
                ):
                    continue
                self.event_journal.append(
                    event_kind=(
                        "operation_terminal"
                        if operation.terminal
                        else (
                            "operation_progress"
                            if operation.status == "running"
                            else "operation_updated"
                        )
                    ),
                    emitted_at=float(self._clock()),
                    operation_id=operation.operation_id,
                    operation_hash=operation_hash,
                    payload={
                        "status": operation.status,
                        "reason_code": operation.reason_code,
                        "terminal": operation.terminal,
                        "execution_outcome": operation.execution_outcome,
                        "evaluation_status": operation.evaluation_status,
                        "task_outcome": operation.task_outcome,
                        "updated_at": operation.updated_at,
                    },
                )
                self._published_operation_hashes[operation.operation_id] = (
                    operation_hash
                )
                for binding in operation.output_artifact_bindings:
                    binding_hash = binding.semantic_content_hash
                    published_key = (operation.operation_id, binding_hash)
                    if published_key in self._published_artifact_bindings:
                        continue
                    self.event_journal.append(
                        event_kind="artifact_ready",
                        emitted_at=float(self._clock()),
                        operation_id=operation.operation_id,
                        operation_hash=operation_hash,
                        artifact_binding=binding,
                        payload={
                            "slot_id": binding.slot_id,
                            "ordinal": binding.ordinal,
                            "artifact_id": binding.artifact.artifact_id,
                        },
                    )
                    self._published_artifact_bindings.add(published_key)

    def replay_events(
        self,
        *,
        after_sequence: int,
        max_events: int,
        operations: Iterable[AgentSkillOperation] = (),
    ) -> AgentGatewayEventBatch:
        self.synchronize_operations(operations)
        return self.event_journal.replay(
            after_sequence=after_sequence,
            max_events=max_events,
        )


__all__ = ["AgentGatewayProtocolRuntime"]
