"""Gateway composition for paired real open/close gripper Skills.

Assembly is default-off.  It owns no device construction or backend lifecycle;
the application must inject one exclusive lower gripper port whose backends are
already part of the same resource graph.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

from agent_gateway.protocol.contracts import AgentObservationSnapshot, AgentSkillRequest
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    validate_skill_route_catalog_closure,
)
from agent_gateway.protocol.skill_io_contracts import SkillIOCatalog
from agent_gateway.protocol.skill_registration import build_skill_registration_catalogs
from agent_gateway.runtime.access_policy import (
    AgentSkillAccessPolicy,
    SkillExecutionGrant,
)
from agent_gateway.runtime.executor_registry import (
    SkillExecutorBinding,
    SkillExecutorRegistry,
)
from agent_gateway.runtime.operation_port import AgentGatewayAuthoritativePortReceipt
from agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorRegistry
from agent_gateway.runtime.reusable_station_executor import (
    ReusableStationAgentSkillOperationExecutor,
)
from agent_gateway.runtime.service import (
    AgentGatewayOperationOutcome,
    AgentGatewayService,
)
from agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter
from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.skill_deploy.gripper_port import DMTacGripperActionPort

from .gripper_action_deployment import RealGripperActionRegistration
from .gripper_action_gateway import build_real_gripper_action_executor


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class RealGripperActionContractBundle:
    registrations: tuple[RealGripperActionRegistration, ...]
    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    io_catalog: SkillIOCatalog
    route_manifest: SkillRouteManifest
    payload_validators: AgentSkillPayloadValidatorRegistry

    @property
    def profile(self):
        return self.registrations[0].profile


def build_real_gripper_action_contract_bundle(
    registrations: tuple[RealGripperActionRegistration, ...],
) -> RealGripperActionContractBundle:
    items = tuple(registrations)
    if len(items) != 2 or any(
        not isinstance(item, RealGripperActionRegistration) for item in items
    ):
        raise ValueError("registrations must contain the paired open/close actions.")
    skill_ids = {item.skill.definition.skill_id for item in items}
    if skill_ids != {"motion.open_gripper", "motion.close_gripper"}:
        raise ValueError("registrations must contain exactly open and close.")
    first = items[0]
    identity_fields = (
        "runtime_session_id",
        "control_domain_id",
        "target_binding_id",
        "station_manifest_hash",
        "execution_manifest_hash",
        "profile_hash",
    )
    for item in items[1:]:
        if any(
            getattr(item.binding, name) != getattr(first.binding, name)
            for name in identity_fields
        ):
            raise ValueError("Paired gripper registrations use different deployments.")
    catalogs = build_skill_registration_catalogs(
        tuple(item.skill for item in items),
        definition_catalog_id="dexteleop.agent_skill_definitions.real_gripper_actions",
        schema_catalog_id="dexteleop.agent_skill_schemas.real_gripper_actions",
        io_catalog_id="dexteleop.agent_skill_io",
    )
    manifest = SkillRouteManifest(
        runtime_session_id=first.binding.runtime_session_id,
        control_domain_id=first.binding.control_domain_id,
        station_manifest_hash=first.binding.station_manifest_hash,
        execution_manifest_hash=first.binding.execution_manifest_hash,
        skill_definition_catalog_hash=catalogs.definition_catalog.semantic_content_hash,
        skill_schema_catalog_hash=catalogs.schema_catalog.semantic_content_hash,
        skill_io_catalog_hash=catalogs.io_catalog.semantic_content_hash,
        target_binding_id=first.binding.target_binding_id,
        target_kind="real",
        current_mode="real_active",
        supported_transport_features=(
            "idempotency_v1",
            "own_domain_hold_v1",
            "own_operation_cancel_v1",
            "own_operation_status_v1",
        ),
        skill_routes=tuple(
            SkillRouteManifestEntry(
                route=item.skill.routes[0],
                deployment_binding=item.binding.to_ref(),
                evaluation_plan_hash=canonical_content_hash(
                    {
                        "skill_id": item.skill.definition.skill_id,
                        "profile_hash": item.profile.semantic_content_hash,
                        "evaluator": "lower_measured_gripper_endpoint.v1",
                    },
                    domain="dexteleop.real_gripper_action_evaluation_plan.v1",
                ),
                availability_hash=item.availability_hash,
            )
            for item in items
        ),
        explicit_exclusions=(
            "agent_estop",
            "agent_rearm",
            "agent_resume",
            "arbitrary_gripper_opening",
            "caller_selected_force",
            "caller_selected_speed",
            "device_connection",
            "device_start",
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
        registry_id="dexteleop.agent_payload_validators.real_gripper_actions:"
        + manifest.runtime_session_id,
        validators=tuple(item.payload_validator for item in items),
    )
    validators.validate_manifest(manifest)
    return RealGripperActionContractBundle(
        registrations=tuple(
            sorted(items, key=lambda item: item.skill.definition.skill_id)
        ),
        definition_catalog=catalogs.definition_catalog,
        schema_catalog=catalogs.schema_catalog,
        io_catalog=catalogs.io_catalog,
        route_manifest=manifest,
        payload_validators=validators,
    )


class RealGripperActionObservationProducer:
    def __init__(
        self,
        *,
        contract_bundle: RealGripperActionContractBundle,
        port: DMTacGripperActionPort,
        lower_station_revision: Callable[[], int],
        clock: Callable[[], float],
    ) -> None:
        self.contract_bundle = contract_bundle
        self.port = port
        self._lower_station_revision = lower_station_revision
        self._clock = clock
        self._lock = threading.RLock()
        self._latest_station_revision = 0

    @property
    def current_station_revision(self) -> int:
        with self._lock:
            return self._latest_station_revision

    def capture_snapshot(
        self, *, active_operation_id: str | None
    ) -> AgentObservationSnapshot:
        revision = self._lower_station_revision()
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("lower_station_revision must return a non-negative int.")
        now = float(self._clock())
        if not math.isfinite(now):
            raise ValueError("clock must return a finite value.")
        feedback = tuple(
            self.port.feedback(component_id=component_id)
            for component_id in self.contract_bundle.profile.component_ids
        )
        if any(item.error_code != 0 for item in feedback):
            raise RuntimeError("Cannot publish a ready snapshot with a gripper fault.")
        captured_at = max(item.measured_at for item in feedback)
        available_at = max(item.available_at for item in feedback)
        maximum_age = self.contract_bundle.profile.maximum_measurement_age_ms
        if any((now - item.measured_at) * 1_000.0 > maximum_age for item in feedback):
            raise RuntimeError(
                "Cannot publish a ready snapshot from stale gripper state."
            )
        manifest = self.contract_bundle.route_manifest
        measurement_summary = {
            "component_ids": list(self.contract_bundle.profile.component_ids),
            "close_fraction_by_component": {
                item.component_id: item.close_fraction for item in feedback
            },
            "moving_by_component": {
                item.component_id: item.is_moving for item in feedback
            },
            "measurement_hash_by_component": {
                item.component_id: item.measurement_hash for item in feedback
            },
        }
        snapshot_hash = canonical_content_hash(
            {
                "manifest_hash": manifest.semantic_content_hash,
                "station_revision": revision,
                "active_operation_id": active_operation_id,
                "measurement_summary": measurement_summary,
            },
            domain="dexteleop.real_gripper_action_snapshot.v1",
        )
        identity = {
            "profile_hash": self.contract_bundle.profile.semantic_content_hash,
            "port_id": self.port.port_id,
        }

        def identity_hash(kind: str) -> str:
            return canonical_content_hash(
                identity | {"identity_kind": kind},
                domain="dexteleop.real_gripper_action_identity.v1",
            )

        snapshot = AgentObservationSnapshot(
            snapshot_id="real-gripper-actions:" + snapshot_hash,
            runtime_session_id=manifest.runtime_session_id,
            skill_route_manifest_hash=manifest.semantic_content_hash,
            station_revision=revision,
            captured_at=captured_at,
            available_at=available_at,
            clock_domain_id=self.contract_bundle.profile.clock_domain_id,
            authority_epoch=0,
            execution_epoch=0,
            dispatch_owner_generation=1,
            lifecycle_state="lower_gripper_owner_attached",
            execution_gate="fixed_endpoint_actions_ready",
            active_operation_id=active_operation_id,
            source_binding_id=self.port.port_id,
            producer_epoch=1,
            target_binding_id=manifest.target_binding_id,
            execution_manifest_hash=manifest.execution_manifest_hash,
            frame_graph_hash=identity_hash("not_applicable_frame_graph"),
            tool_config_hash=identity_hash("gripper_endpoint_profile"),
            calibration_hash=identity_hash("close_fraction_calibration"),
            measurement_summary=measurement_summary,
            measurement_freshness_ms=max(
                0,
                math.ceil((now - min(item.measured_at for item in feedback)) * 1_000.0),
            ),
            measurement_valid=True,
        )
        with self._lock:
            if revision < self._latest_station_revision:
                raise ValueError("Lower station revision regressed.")
            self._latest_station_revision = revision
        return snapshot

    def capture_and_publish(self, service: AgentGatewayService):
        snapshot = self.capture_snapshot(
            active_operation_id=service.operation_store.active_operation_id
        )
        return service.publish_observation(snapshot=snapshot, now=float(self._clock()))


@dataclass
class RealGripperActionGatewaySession:
    contract_bundle: RealGripperActionContractBundle
    port: DMTacGripperActionPort
    observation_producer: RealGripperActionObservationProducer
    executors_by_skill_id: Mapping[str, ReusableStationAgentSkillOperationExecutor]
    service: AgentGatewayService
    _close_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _close_outcome: AgentGatewayOperationOutcome | None = field(
        default=None, init=False
    )

    def wire_adapter(
        self,
        *,
        authenticated_client_id: str,
        grant_resolver: Callable[[AgentSkillRequest], SkillExecutionGrant | None]
        | None = None,
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
            grant_resolver=grant_resolver,
        )

    def build_unix_json_server(
        self,
        *,
        socket_path: str,
        client_id_by_uid: Mapping[int, str],
        socket_mode: int = 0o600,
        max_frame_bytes: int = 2 * 1024 * 1024,
        peer_idle_timeout_sec: float = 10.0,
        grant_resolver: Callable[[AgentSkillRequest], SkillExecutionGrant | None]
        | None = None,
        read_only_client_ids: Collection[str] = (),
    ) -> AgentGatewayUnixJsonServer:
        peer_map = AgentGatewayUnixPeerClientMap(client_id_by_uid)
        if set(peer_map.client_id_by_uid.values()) != set(
            self.service.access_policy.allowed_client_ids
        ):
            raise ValueError("Unix peer map must exactly cover allowed clients.")
        if read_only_client_ids:
            raise ValueError("The action-only gripper Gateway has no read-only Route.")
        return AgentGatewayUnixJsonServer(
            socket_path=socket_path,
            peer_client_map=peer_map,
            facade_factory=lambda client_id: self.wire_adapter(
                authenticated_client_id=client_id,
                grant_resolver=grant_resolver,
            ),
            close_gateway=self.close,
            socket_mode=socket_mode,
            max_frame_bytes=max_frame_bytes,
            peer_idle_timeout_sec=peer_idle_timeout_sec,
        )

    def wait(self, *, timeout_sec: float | None = None) -> bool:
        active = tuple(
            executor
            for executor in self.executors_by_skill_id.values()
            if executor.active_operation_id is not None
        )
        return bool(active) and all(
            executor.wait(timeout_sec=timeout_sec) for executor in active
        )

    def close(self) -> AgentGatewayOperationOutcome:
        with self._close_lock:
            if self._close_outcome is None:
                outcome = self.service.close(
                    current_station_revision=self.observation_producer.current_station_revision
                )
                if outcome.accepted:
                    self._close_outcome = outcome
            else:
                outcome = self._close_outcome
            if not outcome.accepted:
                return outcome
            receipt = outcome.port_receipt
            if not (
                isinstance(receipt, AgentGatewayAuthoritativePortReceipt)
                and receipt.succeeded
                and receipt.proves_motion_or_hold
            ):
                raise RuntimeError("Gripper Gateway close requires lower HOLD proof.")
            for executor in self.executors_by_skill_id.values():
                hold = executor.request_hold(
                    operation=None,
                    reason="real_gripper_gateway_close_route_fence",
                )
                if not hold.succeeded or not hold.proves_motion_or_hold:
                    raise RuntimeError("Gripper route executor failed to confirm HOLD.")
                executor.close_after_hold(hold_receipt=hold)
            return outcome


def assemble_real_gripper_action_gateway_session(
    *,
    contract_bundle: RealGripperActionContractBundle,
    port: DMTacGripperActionPort,
    lower_station_revision: Callable[[], int],
    allowed_client_ids: tuple[str, ...],
    max_retained_operations: int = 32,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> RealGripperActionGatewaySession:
    if tuple(sorted(port.component_ids)) != contract_bundle.profile.component_ids:
        raise ValueError("Lower port components differ from the registered profile.")
    if port.clock_domain_id != contract_bundle.profile.clock_domain_id:
        raise ValueError("Lower port clock domain differs from the registered profile.")
    clients = tuple(
        sorted({_required(item, name="client_id") for item in allowed_client_ids})
    )
    if not clients:
        raise ValueError("allowed_client_ids must not be empty.")
    producer = RealGripperActionObservationProducer(
        contract_bundle=contract_bundle,
        port=port,
        lower_station_revision=lower_station_revision,
        clock=clock,
    )
    executors = {
        registration.skill.definition.skill_id: build_real_gripper_action_executor(
            registration=registration,
            port=port,
            current_station_revision=lambda: producer.current_station_revision,
            clock=clock,
            sleep=sleep,
        )
        for registration in contract_bundle.registrations
    }
    registry = SkillExecutorRegistry(
        manifest=contract_bundle.route_manifest,
        bindings=tuple(
            SkillExecutorBinding.from_manifest_entry(
                manifest=contract_bundle.route_manifest,
                entry=entry,
                execution_port=executors[entry.route.skill_id],
            )
            for entry in contract_bundle.route_manifest.skill_routes
        ),
    )
    domain_executor = executors["motion.open_gripper"]
    service = AgentGatewayService(
        manifest=contract_bundle.route_manifest,
        access_policy=AgentSkillAccessPolicy(
            access_policy_id="agent-access:real-gripper-actions:"
            + contract_bundle.route_manifest.runtime_session_id,
            allowed_client_ids=clients,
            allowed_modes=("real_active",),
            max_request_timeout_ms=5_000,
            max_grant_lifetime_sec=30.0,
            max_grant_invocations=1,
        ),
        executor_registry=registry,
        domain_control_port=domain_executor,
        payload_validators=contract_bundle.payload_validators,
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=max_retained_operations
        ),
        clock=clock,
        clock_domain_id=contract_bundle.profile.clock_domain_id,
    )
    return RealGripperActionGatewaySession(
        contract_bundle=contract_bundle,
        port=port,
        observation_producer=producer,
        executors_by_skill_id=executors,
        service=service,
    )


__all__ = [
    "RealGripperActionContractBundle",
    "RealGripperActionGatewaySession",
    "RealGripperActionObservationProducer",
    "assemble_real_gripper_action_gateway_session",
    "build_real_gripper_action_contract_bundle",
]
