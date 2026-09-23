"""Attach-only real gripper-state Gateway composition.

The composition publishes ``state.read_gripper_state`` over the normal Agent
Gateway discovery and transport surface.  It consumes only an immutable state
cut already published by the lower gripper owner; assembly and reads never
open or refresh a hardware device.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_gateway.protocol.contracts import AgentObservationSnapshot
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    validate_skill_route_catalog_closure,
)
from agent_gateway.protocol.skill_io_contracts import SkillIOCatalog
from agent_gateway.protocol.skill_registration import build_skill_registration_catalogs
from agent_gateway.runtime.access_policy import AgentSkillAccessPolicy
from agent_gateway.runtime.executor_registry import (
    SkillExecutorBinding,
    SkillExecutorRegistry,
)
from agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from agent_gateway.runtime.passive_read_executor import PassiveReadSkillExecutor
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorRegistry
from agent_gateway.runtime.service import AgentGatewayService
from agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter
from dataset_contracts.canonical_encoding import canonical_content_hash

from .gripper_state_deployment import RealGripperStateRegistration
from .gripper_state_gateway import (
    PublishedGripperStateSource,
    build_real_gripper_state_executor,
    read_published_gripper_state,
)
from .observation_gateway import ObservationOnlyDomainControlPort


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True)
class RealGripperStateContractBundle:
    """Closed discovery and validation surface for one registration."""

    registration: RealGripperStateRegistration
    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    io_catalog: SkillIOCatalog
    route_manifest: SkillRouteManifest
    payload_validators: AgentSkillPayloadValidatorRegistry


def build_real_gripper_state_contract_bundle(
    registration: RealGripperStateRegistration,
) -> RealGripperStateContractBundle:
    """Close one passive registration into Gateway discovery catalogs."""

    if not isinstance(registration, RealGripperStateRegistration):
        raise TypeError("registration must be typed.")
    route = registration.skill.routes[0]
    if route.supported_modes != ("real_observe",):
        raise ValueError(
            "Standalone gripper-state composition requires a real_observe route."
        )
    catalogs = build_skill_registration_catalogs(
        (registration.skill,),
        definition_catalog_id="dexteleop.agent_skill_definitions.real_gripper_state",
        schema_catalog_id="dexteleop.agent_skill_schemas.real_gripper_state",
        io_catalog_id="dexteleop.agent_skill_io",
    )
    manifest = SkillRouteManifest(
        runtime_session_id=registration.binding.runtime_session_id,
        control_domain_id=registration.binding.control_domain_id,
        station_manifest_hash=registration.binding.station_manifest_hash,
        execution_manifest_hash=registration.binding.execution_manifest_hash,
        skill_definition_catalog_hash=catalogs.definition_catalog.semantic_content_hash,
        skill_schema_catalog_hash=catalogs.schema_catalog.semantic_content_hash,
        skill_io_catalog_hash=catalogs.io_catalog.semantic_content_hash,
        target_binding_id=registration.binding.target_binding_id,
        target_kind="real",
        current_mode="real_observe",
        supported_transport_features=(
            "idempotency_v1",
            "own_operation_cancel_v1",
            "own_operation_status_v1",
        ),
        skill_routes=(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=registration.binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=registration.availability_hash,
            ),
        ),
        explicit_exclusions=(
            "agent_estop",
            "agent_rearm",
            "agent_resume",
            "control_authority",
            "device_connection",
            "device_start",
            "direct_hardware_refresh",
            "domain_hold",
            "motion",
            "raw_action_submission",
            "second_dispatch_owner",
        ),
    )
    validate_skill_route_catalog_closure(
        definition_catalog=catalogs.definition_catalog,
        schema_catalog=catalogs.schema_catalog,
        route_manifest=manifest,
        io_catalog=catalogs.io_catalog,
    )
    validators = AgentSkillPayloadValidatorRegistry(
        registry_id=(
            "dexteleop.agent_payload_validators.real_gripper_state:"
            + registration.binding.runtime_session_id
        ),
        validators=(registration.payload_validator,),
    )
    validators.validate_manifest(manifest)
    return RealGripperStateContractBundle(
        registration=registration,
        definition_catalog=catalogs.definition_catalog,
        schema_catalog=catalogs.schema_catalog,
        io_catalog=catalogs.io_catalog,
        route_manifest=manifest,
        payload_validators=validators,
    )


class RealGripperStateObservationProducer:
    """Build planning snapshots from the same lower-owned no-I/O source."""

    def __init__(
        self,
        *,
        contract_bundle: RealGripperStateContractBundle,
        source: PublishedGripperStateSource,
        lower_station_revision: Callable[[], int],
        clock: Callable[[], float],
    ) -> None:
        if not isinstance(contract_bundle, RealGripperStateContractBundle):
            raise TypeError("contract_bundle must be typed.")
        if not callable(getattr(source, "latest_gripper_state", None)) or not bool(
            getattr(source, "no_io", False)
        ):
            raise TypeError("source must be an explicitly no-I/O gripper-state source.")
        if not callable(lower_station_revision) or not callable(clock):
            raise TypeError("lower_station_revision and clock must be callable.")
        self.contract_bundle = contract_bundle
        self.source = source
        self._lower_station_revision = lower_station_revision
        self._clock = clock
        self._lock = threading.RLock()
        self._latest_station_revision = 0

    @property
    def current_station_revision(self) -> int:
        with self._lock:
            return self._latest_station_revision

    def capture_snapshot(
        self,
        *,
        active_operation_id: str | None,
    ) -> AgentObservationSnapshot:
        registration = self.contract_bundle.registration
        profile = registration.profile
        station_revision = self._lower_station_revision()
        if (
            isinstance(station_revision, bool)
            or not isinstance(station_revision, int)
            or station_revision < 0
        ):
            raise ValueError("lower_station_revision must return a non-negative int.")
        now = float(self._clock())
        if not math.isfinite(now):
            raise ValueError("clock must return a finite value.")
        read = read_published_gripper_state(
            registration=registration,
            source=self.source,
            parameters={
                "component_ids": list(profile.component_ids),
                "max_measurement_age_ms": profile.maximum_measurement_age_ms,
            },
            station_revision=station_revision,
            now=now,
        )
        result = read.execution_result
        measured_at = tuple(result["measured_at_by_component"].values())
        freshness_ms = max(0, math.ceil((now - min(measured_at)) * 1_000.0))
        identity = {
            "profile_hash": profile.semantic_content_hash,
            "deployment_binding_hash": registration.binding.semantic_content_hash,
        }

        def identity_hash(kind: str) -> str:
            return canonical_content_hash(
                identity | {"identity_kind": kind},
                domain="dexteleop.agent_skill.real_gripper_state_identity.v1",
            )

        snapshot_hash = canonical_content_hash(
            {
                "measurement_hash": result["measurement_hash"],
                "cutoff_at": now,
                "station_revision": station_revision,
                "active_operation_id": active_operation_id,
                "manifest_hash": (
                    self.contract_bundle.route_manifest.semantic_content_hash
                ),
            },
            domain="dexteleop.agent_skill.real_gripper_state_snapshot_id.v1",
        )
        snapshot = AgentObservationSnapshot(
            snapshot_id="real-gripper-state:" + snapshot_hash,
            runtime_session_id=registration.binding.runtime_session_id,
            skill_route_manifest_hash=(
                self.contract_bundle.route_manifest.semantic_content_hash
            ),
            station_revision=station_revision,
            captured_at=max(measured_at),
            available_at=result["available_at"],
            clock_domain_id=profile.clock_domain_id,
            authority_epoch=0,
            execution_epoch=0,
            dispatch_owner_generation=0,
            lifecycle_state="lower_owner_attached",
            execution_gate="passive_read_only",
            active_operation_id=active_operation_id,
            source_binding_id=profile.observation_binding_id,
            producer_epoch=1,
            target_binding_id=registration.binding.target_binding_id,
            execution_manifest_hash=registration.binding.execution_manifest_hash,
            frame_graph_hash=identity_hash("not_applicable_frame_graph"),
            tool_config_hash=identity_hash("gripper_profile"),
            calibration_hash=identity_hash("close_fraction_calibration"),
            measurement_summary={
                "measurement_id": result["measurement_id"],
                "measurement_hash": result["measurement_hash"],
                "component_ids": result["component_ids"],
                "measured_at_by_component": result["measured_at_by_component"],
                "close_fraction_by_component": (result["close_fraction_by_component"]),
                "state_source_by_component": result["state_source_by_component"],
                "observation_binding_hash": profile.observation_binding_hash,
                "evidence_refs": list(read.evidence_refs),
            },
            measurement_freshness_ms=freshness_ms,
            measurement_valid=True,
        )
        with self._lock:
            if station_revision < self._latest_station_revision:
                raise ValueError("Lower station revision regressed.")
            self._latest_station_revision = station_revision
        return snapshot

    def capture_and_publish(self, service: AgentGatewayService):
        if not isinstance(service, AgentGatewayService):
            raise TypeError("service must be an AgentGatewayService.")
        snapshot = self.capture_snapshot(
            active_operation_id=service.operation_store.active_operation_id,
        )
        return service.publish_observation(snapshot=snapshot, now=float(self._clock()))


@dataclass
class RealGripperStateGatewaySession:
    contract_bundle: RealGripperStateContractBundle
    observation_producer: RealGripperStateObservationProducer
    state_executor: PassiveReadSkillExecutor
    domain_control_port: ObservationOnlyDomainControlPort
    service: AgentGatewayService
    _close_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _close_outcome: Any = field(default=None, init=False, repr=False)

    def wire_adapter(
        self, *, authenticated_client_id: str
    ) -> BoundAgentGatewayWireAdapter:
        return BoundAgentGatewayWireAdapter(
            authenticated_client_id=authenticated_client_id,
            definition_catalog=self.contract_bundle.definition_catalog,
            schema_catalog=self.contract_bundle.schema_catalog,
            service=self.service,
            publish_observation=lambda: self.observation_producer.capture_and_publish(
                self.service
            ),
            current_station_revision=lambda: (
                self.observation_producer.current_station_revision
            ),
            operation_launcher=self.service.launch_operation,
        )

    def build_unix_json_server(
        self,
        *,
        socket_path: str,
        client_id_by_uid: Mapping[int, str],
        socket_mode: int = 0o600,
        max_frame_bytes: int = 2 * 1024 * 1024,
        peer_idle_timeout_sec: float = 10.0,
    ) -> AgentGatewayUnixJsonServer:
        peer_map = AgentGatewayUnixPeerClientMap(client_id_by_uid)
        if set(peer_map.client_id_by_uid.values()) != set(
            self.service.access_policy.allowed_client_ids
        ):
            raise ValueError("Unix peer map must exactly cover the access policy.")
        return AgentGatewayUnixJsonServer(
            socket_path=socket_path,
            peer_client_map=peer_map,
            facade_factory=lambda client_id: self.wire_adapter(
                authenticated_client_id=client_id
            ),
            close_gateway=self.close,
            socket_mode=socket_mode,
            max_frame_bytes=max_frame_bytes,
            peer_idle_timeout_sec=peer_idle_timeout_sec,
        )

    def close(self):
        with self._close_lock:
            if self._close_outcome is None:
                self._close_outcome = self.service.close(
                    current_station_revision=(
                        self.observation_producer.current_station_revision
                    )
                )
            return self._close_outcome


def assemble_real_gripper_state_gateway_session(
    *,
    contract_bundle: RealGripperStateContractBundle,
    source: PublishedGripperStateSource,
    lower_station_revision: Callable[[], int],
    allowed_client_ids: tuple[str, ...],
    clock: Callable[[], float] = time.monotonic,
    max_retained_operations: int = 32,
) -> RealGripperStateGatewaySession:
    """Assemble one discoverable session without opening gripper hardware."""

    if not isinstance(contract_bundle, RealGripperStateContractBundle):
        raise TypeError("contract_bundle must be typed.")
    clients = tuple(
        _required(item, name="allowed_client_id") for item in allowed_client_ids
    )
    if not clients or len(set(clients)) != len(clients):
        raise ValueError("allowed_client_ids must be non-empty and unique.")
    retention = _positive_integer(
        max_retained_operations,
        name="max_retained_operations",
    )
    producer = RealGripperStateObservationProducer(
        contract_bundle=contract_bundle,
        source=source,
        lower_station_revision=lower_station_revision,
        clock=clock,
    )
    registration = contract_bundle.registration
    executor = build_real_gripper_state_executor(
        registration=registration,
        source=source,
        current_station_revision=lambda: producer.current_station_revision,
        max_retained_operations=retention,
        clock=clock,
    )
    entry = contract_bundle.route_manifest.route("state.read_gripper_state", "1")
    registry = SkillExecutorRegistry(
        manifest=contract_bundle.route_manifest,
        bindings=(
            SkillExecutorBinding.from_manifest_entry(
                manifest=contract_bundle.route_manifest,
                entry=entry,
                execution_port=executor,
            ),
        ),
    )
    domain_port = ObservationOnlyDomainControlPort(
        runtime_session_id=registration.binding.runtime_session_id,
        clock=clock,
    )
    service = AgentGatewayService(
        manifest=contract_bundle.route_manifest,
        access_policy=AgentSkillAccessPolicy(
            access_policy_id=(
                "real-gripper-state-observe:" + registration.binding.runtime_session_id
            ),
            allowed_client_ids=clients,
            allowed_modes=("real_observe",),
            max_request_timeout_ms=2_000,
            max_grant_lifetime_sec=2.0,
        ),
        executor_registry=registry,
        domain_control_port=domain_port,
        payload_validators=contract_bundle.payload_validators,
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=retention
        ),
        clock=clock,
        clock_domain_id=registration.profile.clock_domain_id,
    )
    return RealGripperStateGatewaySession(
        contract_bundle=contract_bundle,
        observation_producer=producer,
        state_executor=executor,
        domain_control_port=domain_port,
        service=service,
    )


__all__ = [
    "RealGripperStateContractBundle",
    "RealGripperStateGatewaySession",
    "RealGripperStateObservationProducer",
    "assemble_real_gripper_state_gateway_session",
    "build_real_gripper_state_contract_bundle",
]
