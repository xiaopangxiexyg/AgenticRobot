"""Default-off Agent Gateway session for real 6+3 TCP primitives."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from agent_gateway.protocol.contracts import AgentObservationSnapshot, AgentSkillRequest
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
from agent_gateway.runtime.reusable_station_executor import (
    ReusableStationAgentSkillOperationExecutor,
)
from agent_gateway.runtime.service import (
    AgentGatewayOperationOutcome,
    AgentGatewayService,
)
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleControlEvidence,
    StationOperationWorkerBinding,
)
from agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import pose_matrix_to_xyzquat
from teleop_runtime.skill_deploy.motion_port import (
    RealAgentMotionPhase,
    RealAgentMotionPort,
)

from .motion_primitive_profiles import (
    RealAgentPrimitiveContractBundle,
)
from .motion_primitive_skill import RealAgentPrimitiveSkillLifecycle
from .observation_gateway import RealControlOwnerState


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


class RealAgentPrimitiveObservationProducer:
    """Project lower owner state and measured port cuts without device I/O."""

    def __init__(
        self,
        *,
        contract_bundle: RealAgentPrimitiveContractBundle,
        port: RealAgentMotionPort,
        owner_state_provider: Callable[[], RealControlOwnerState],
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(contract_bundle, RealAgentPrimitiveContractBundle):
            raise TypeError("contract_bundle must be typed.")
        if not isinstance(port, RealAgentMotionPort):
            raise TypeError("port must be typed.")
        if not callable(owner_state_provider) or not callable(clock):
            raise TypeError("owner_state_provider and clock must be callable.")
        self.contract_bundle = contract_bundle
        self.port = port
        self.owner_state_provider = owner_state_provider
        self.clock = clock
        self._validate_port_binding()

    @property
    def current_station_revision(self) -> int:
        return self.port.snapshot().revision

    def _owner_state(self) -> RealControlOwnerState:
        owner = self.owner_state_provider()
        if not isinstance(owner, RealControlOwnerState):
            raise TypeError("owner_state_provider must return RealControlOwnerState.")
        attachment = self.contract_bundle.attachment
        checks = {
            "runtime_session_id": attachment.runtime_session_id,
            "control_domain_id": attachment.control_domain_id,
            "embodiment_instance_id": attachment.embodiment_instance_id,
            "target_binding_id": attachment.target_binding_id,
            "execution_manifest_hash": attachment.execution_manifest_hash,
            "backend_inventory_hash": attachment.backend_inventory_hash,
            "clock_domain_id": attachment.clock_domain_id,
        }
        for field_name, expected in checks.items():
            if getattr(owner, field_name) != expected:
                raise ValueError(f"Real motion owner {field_name} mismatch.")
        if owner.lifecycle_state != "running" or owner.execution_gate != "ready":
            raise RuntimeError("Real motion owner is not RUNNING/READY.")
        return owner

    def _validate_port_binding(self) -> None:
        attachment = self.contract_bundle.attachment
        profile = self.contract_bundle.cardinal_profile
        config = self.port.config
        checks = {
            "execution_frame_id": profile.execution_reference_frame_id,
            "execution_frame_spec_hash": (
                profile.execution_reference_frame_spec_hash
            ),
        }
        for field_name, expected in checks.items():
            if getattr(config, field_name) != expected:
                raise ValueError(f"Real motion port {field_name} mismatch.")
        if set(config.component_id_by_side.values()) != set(
            attachment.component_ids
        ):
            raise ValueError("Real motion port component binding mismatch.")
        if config.component_id_by_side.get(profile.side) != profile.component_id:
            raise ValueError("Real motion port component/side binding mismatch.")

    def capture_snapshot(
        self,
        *,
        active_operation_id: str | None,
    ) -> AgentObservationSnapshot:
        now = float(self.clock())
        owner = self._owner_state()
        cut = self.port.latest_measured_pose()
        if cut is None:
            raise RuntimeError("No lower measured TCP-pose cut is available.")
        age_ms = max(0, math.ceil((now - cut.measured_at) * 1_000.0))
        maximum_age = self.contract_bundle.convergence_profile.max_measurement_age_ms
        if age_ms > maximum_age:
            raise RuntimeError("Lower measured TCP-pose cut is stale.")
        if owner.captured_at > cut.available_at:
            raise ValueError(
                "Lower owner state was captured after the measured cut became "
                "available."
            )
        available_at = max(cut.available_at, owner.captured_at)
        if available_at > now:
            raise RuntimeError("Lower real observation inputs are not yet available.")
        attachment = self.contract_bundle.attachment
        poses = {
            component_id: [
                float(value)
                for value in pose_matrix_to_xyzquat(
                    pose,
                    name=f"real_agent_observation.{component_id}",
                )
            ]
            for component_id, pose in cut.pose_world_by_component.items()
        }
        station_revision = self.current_station_revision
        snapshot_hash = canonical_content_hash(
            {
                "manifest_hash": self.contract_bundle.route_manifest.semantic_content_hash,
                "station_revision": station_revision,
                "owner_state_hash": owner.semantic_content_hash,
                "measured_cut_hash": cut.semantic_content_hash,
                "active_operation_id": active_operation_id,
            },
            domain="dexteleop.real_agent_motion.observation_snapshot_id.v1",
        )
        return AgentObservationSnapshot(
            snapshot_id="real-agent-motion-observation:" + snapshot_hash,
            runtime_session_id=attachment.runtime_session_id,
            skill_route_manifest_hash=(
                self.contract_bundle.route_manifest.semantic_content_hash
            ),
            station_revision=station_revision,
            captured_at=cut.measured_at,
            available_at=available_at,
            clock_domain_id=attachment.clock_domain_id,
            authority_epoch=owner.authority_epoch,
            execution_epoch=owner.execution_epoch,
            dispatch_owner_generation=owner.owner_generation,
            lifecycle_state=owner.lifecycle_state,
            execution_gate=owner.execution_gate,
            active_operation_id=active_operation_id,
            source_binding_id=attachment.source_binding_id,
            producer_epoch=attachment.producer_epoch,
            target_binding_id=attachment.target_binding_id,
            execution_manifest_hash=attachment.execution_manifest_hash,
            frame_graph_hash=attachment.frame_graph_hash,
            tool_config_hash=attachment.tool_config_hash,
            calibration_hash=attachment.calibration_hash,
            measurement_summary={
                "attachment_hash": attachment.semantic_content_hash,
                "owner_state_hash": owner.semantic_content_hash,
                "measured_state_hash": cut.semantic_content_hash,
                "pose_xyz_quat_xyzw_by_component": poses,
                "reference_frame_id": attachment.reference_frame_id,
                "reference_frame_spec_hash": attachment.reference_frame_spec_hash,
            },
            measurement_freshness_ms=age_ms,
            measurement_valid=True,
        )

    def capture_and_publish(self, service: AgentGatewayService):
        snapshot = self.capture_snapshot(
            active_operation_id=service.operation_store.active_operation_id,
        )
        return service.publish_observation(snapshot=snapshot, now=snapshot.available_at)


def _shared_port_hold_evidence(
    *,
    port: RealAgentMotionPort,
    reason: str,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    timeout_sec: float = 2.0,
) -> StationLifecycleControlEvidence:
    now = float(clock())
    snapshot = port.snapshot()
    lower_operation_id = (
        None
        if snapshot.operation_id is None
        else "real-agent-motion:" + snapshot.operation_id
    )
    if snapshot.phase == RealAgentMotionPhase.HELD and snapshot.hold_succeeded:
        return StationLifecycleControlEvidence(
            succeeded=True,
            status="real_agent_domain_hold_confirmed",
            lower_operation_id=lower_operation_id,
            evidence_refs=(snapshot.evidence_ref,),
            lower_hold_confirmed=True,
        )
    if snapshot.operation_id is not None and snapshot.phase in {
        RealAgentMotionPhase.PENDING_ACTIVATION,
        RealAgentMotionPhase.ACTIVE,
        RealAgentMotionPhase.FAULTED,
    }:
        port.request_hold(
            operation_id=snapshot.operation_id,
            reason=reason,
            requested_at=now,
        )
    elif snapshot.phase not in {
        RealAgentMotionPhase.HOLD_REQUESTED,
        RealAgentMotionPhase.HOLD_APPLYING,
    }:
        port.request_domain_hold(reason=reason, requested_at=now)
    deadline = now + float(timeout_sec)
    while float(clock()) < deadline:
        snapshot = port.snapshot()
        if snapshot.phase == RealAgentMotionPhase.HELD:
            confirmed = bool(snapshot.hold_succeeded)
            return StationLifecycleControlEvidence(
                succeeded=confirmed,
                status=(
                    "real_agent_domain_hold_confirmed"
                    if confirmed
                    else "real_agent_domain_hold_unconfirmed"
                ),
                lower_operation_id=lower_operation_id,
                evidence_refs=(snapshot.evidence_ref,),
                lower_hold_confirmed=confirmed,
            )
        if snapshot.phase == RealAgentMotionPhase.FAULTED and (
            snapshot.hold_succeeded is False
        ):
            break
        sleep(0.01)
    snapshot = port.snapshot()
    return StationLifecycleControlEvidence(
        succeeded=False,
        status="real_agent_domain_hold_timeout",
        lower_operation_id=lower_operation_id,
        evidence_refs=(snapshot.evidence_ref,),
        lower_hold_confirmed=False,
    )


@dataclass
class RealAgentPrimitiveGatewaySession:
    contract_bundle: RealAgentPrimitiveContractBundle
    port: RealAgentMotionPort
    observation_producer: RealAgentPrimitiveObservationProducer
    executors_by_skill_id: Mapping[str, ReusableStationAgentSkillOperationExecutor]
    service: AgentGatewayService
    _close_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _close_outcome: AgentGatewayOperationOutcome | None = field(
        default=None,
        init=False,
    )

    def wire_adapter(
        self,
        *,
        authenticated_client_id: str,
        grant_resolver: (
            Callable[[AgentSkillRequest], SkillExecutionGrant | None] | None
        ) = None,
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
        grant_resolver: (
            Callable[[AgentSkillRequest], SkillExecutionGrant | None] | None
        ) = None,
    ) -> AgentGatewayUnixJsonServer:
        peer_map = AgentGatewayUnixPeerClientMap(client_id_by_uid)
        if set(peer_map.client_id_by_uid.values()) != set(
            self.service.access_policy.allowed_client_ids
        ):
            raise ValueError("Unix peer map must exactly cover allowed clients.")
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
                self._close_outcome = self.service.close(
                    current_station_revision=(
                        self.observation_producer.current_station_revision
                    )
                )
            outcome = self._close_outcome
            receipt = outcome.port_receipt
            if (
                outcome.accepted
                and isinstance(receipt, AgentGatewayAuthoritativePortReceipt)
                and receipt.succeeded
                and receipt.proves_motion_or_hold
            ):
                for executor in self.executors_by_skill_id.values():
                    executor_receipt = executor.request_hold(
                        operation=None,
                        reason="real_agent_gateway_close_route_fence",
                    )
                    if executor_receipt.succeeded and executor_receipt.proves_motion_or_hold:
                        executor.close_after_hold(hold_receipt=executor_receipt)
            return outcome


def assemble_real_agent_primitive_gateway_session(
    *,
    contract_bundle: RealAgentPrimitiveContractBundle,
    port: RealAgentMotionPort,
    owner_state_provider: Callable[[], RealControlOwnerState],
    allowed_client_ids: tuple[str, ...],
    max_retained_operations: int = 16,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> RealAgentPrimitiveGatewaySession:
    """Assemble four exact Route executors without starting a server or robot."""

    if not isinstance(contract_bundle, RealAgentPrimitiveContractBundle):
        raise TypeError("contract_bundle must be typed.")
    if not isinstance(port, RealAgentMotionPort):
        raise TypeError("port must be typed.")
    clients = tuple(
        sorted(
            {
                _required(item, field_name="client_id")
                for item in allowed_client_ids
            }
        )
    )
    if not clients:
        raise ValueError("allowed_client_ids must not be empty.")
    if isinstance(max_retained_operations, bool) or not isinstance(
        max_retained_operations, int
    ) or max_retained_operations <= 0:
        raise ValueError("max_retained_operations must be positive.")
    producer = RealAgentPrimitiveObservationProducer(
        contract_bundle=contract_bundle,
        port=port,
        owner_state_provider=owner_state_provider,
        clock=clock,
    )

    def domain_hold(*, operation, reason):
        del operation
        return _shared_port_hold_evidence(
            port=port,
            reason=reason,
            clock=clock,
            sleep=sleep,
        )

    executors: dict[str, ReusableStationAgentSkillOperationExecutor] = {}
    execution_port_by_route_hash = {}
    for entry in contract_bundle.route_manifest.skill_routes:
        route = entry.route
        binding = contract_bundle.binding_for_skill(route.skill_id)
        if route.skill_id == "motion.step_tcp_cardinal":
            profile = contract_bundle.cardinal_profile
        else:
            profile = contract_bundle.orientation_profile(route.skill_id)

        def lifecycle_factory(
            *,
            request,
            operation,
            _route=route,
            _binding=binding,
            _profile=profile,
        ):
            return RealAgentPrimitiveSkillLifecycle(
                port=port,
                contract_bundle=contract_bundle,
                route=_route,
                binding=_binding,
                profile=_profile,
                request=request,
                operation=operation,
                clock=clock,
                sleep=sleep,
            )

        executor = ReusableStationAgentSkillOperationExecutor(
            port_id=(
                "agent-gateway:real-primitive:"
                + route.skill_id
                + ":"
                + contract_bundle.route_manifest.runtime_session_id
            ),
            binding=StationOperationWorkerBinding(
                worker_id=(
                    "real-agent-primitive-worker:"
                    + route.skill_id
                    + ":"
                    + contract_bundle.route_manifest.runtime_session_id
                ),
                executor_id=binding.executor_id,
                runtime_session_id=binding.runtime_session_id,
                skill_id=route.skill_id,
                skill_version=route.skill_version,
                route_id=route.route_id,
                route_version=route.route_version,
                deployment_binding_hash=binding.semantic_content_hash,
            ),
            lifecycle_factory=lifecycle_factory,
            domain_hold=domain_hold,
            current_station_revision=lambda: producer.current_station_revision,
            clock=clock,
        )
        executors[route.skill_id] = executor
        execution_port_by_route_hash[route.semantic_content_hash] = executor
    registry = SkillExecutorRegistry(
        manifest=contract_bundle.route_manifest,
        bindings=tuple(
            SkillExecutorBinding.from_manifest_entry(
                manifest=contract_bundle.route_manifest,
                entry=entry,
                execution_port=execution_port_by_route_hash[
                    entry.route.semantic_content_hash
                ],
            )
            for entry in contract_bundle.route_manifest.skill_routes
        ),
    )
    access_policy = AgentSkillAccessPolicy(
        access_policy_id=(
            "agent-access:real-motion-primitives:"
            + contract_bundle.route_manifest.runtime_session_id
        ),
        allowed_client_ids=clients,
        allowed_modes=("real_active",),
        max_request_timeout_ms=max(
            entry.route.max_operation_duration_ms
            for entry in contract_bundle.route_manifest.skill_routes
        ),
        max_grant_lifetime_sec=30.0,
        max_grant_invocations=1,
    )
    domain_executor = executors["motion.step_tcp_cardinal"]
    service = AgentGatewayService(
        manifest=contract_bundle.route_manifest,
        access_policy=access_policy,
        executor_registry=registry,
        domain_control_port=domain_executor,
        payload_validators=contract_bundle.payload_validators,
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=max_retained_operations,
            max_admissions_by_route_hash={
                entry.route.semantic_content_hash: max_retained_operations
                for entry in contract_bundle.route_manifest.skill_routes
            },
        ),
        clock=clock,
        clock_domain_id=contract_bundle.attachment.clock_domain_id,
    )
    return RealAgentPrimitiveGatewaySession(
        contract_bundle=contract_bundle,
        port=port,
        observation_producer=producer,
        executors_by_skill_id=executors,
        service=service,
    )


__all__ = [
    "RealAgentPrimitiveGatewaySession",
    "RealAgentPrimitiveObservationProducer",
    "assemble_real_agent_primitive_gateway_session",
]
