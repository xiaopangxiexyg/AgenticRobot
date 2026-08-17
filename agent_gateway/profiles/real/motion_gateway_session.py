"""Default-off Agent Gateway session for real 6+3 TCP primitives."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from agent_gateway.protocol.contracts import AgentObservationSnapshot, AgentSkillRequest
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.session_contracts import AgentGatewaySessionCapabilities
from agent_gateway.runtime.access_policy import (
    AgentSkillAccessPolicy,
    SkillExecutionGrant,
)
from agent_gateway.runtime.executor_registry import (
    SkillExecutorBinding,
    SkillExecutorRegistry,
)
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.artifact_transfer import AgentArtifactTransferBroker
from agent_gateway.runtime.event_journal import BoundedAgentGatewayEventJournal
from agent_gateway.runtime.operation_port import AgentGatewayAuthoritativePortReceipt
from agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.protocol_runtime import AgentGatewayProtocolRuntime
from agent_gateway.runtime.reusable_station_executor import (
    ReusableStationAgentSkillOperationExecutor,
)
from agent_gateway.runtime.service import (
    AgentGatewayOperationOutcome,
    AgentGatewayService,
)
from agent_gateway.runtime.station_revision import AgentStationRevisionTracker
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleControlEvidence,
    StationOperationWorkerBinding,
)
from agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter
from agent_gateway.skills.perception.schemas import (
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)
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
from .perception_deployment import (
    REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
    REAL_CAPTURE_RGB_OUTPUT_SLOT_ID_V1,
)
from .realsense_perception import RealSenseAgentPerceptionProvider


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
        revision_tracker: AgentStationRevisionTracker | None = None,
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
        self.revision_tracker = (
            AgentStationRevisionTracker()
            if revision_tracker is None
            else revision_tracker
        )
        if not isinstance(self.revision_tracker, AgentStationRevisionTracker):
            raise TypeError("revision_tracker must be an AgentStationRevisionTracker.")
        self.clock = clock
        self._validate_port_binding()

    @property
    def current_station_revision(self) -> int:
        return self._observe_control_state()

    def _control_fingerprint(self) -> dict[str, object]:
        """Return only state that can change admission semantics.

        The lower port revision also advances for every measured/control cycle
        and for dispatch bookkeeping. Those are fresh measurement facts, not
        optimistic-concurrency changes to the station control state.
        """

        attachment = self.contract_bundle.attachment
        port_snapshot = self.port.snapshot()
        owner_cut = self.port.latest_owner_state()
        fingerprint: dict[str, object] = {
            "runtime_session_id": attachment.runtime_session_id,
            "control_domain_id": attachment.control_domain_id,
            "target_binding_id": attachment.target_binding_id,
            "execution_manifest_hash": attachment.execution_manifest_hash,
            "phase": port_snapshot.phase.value,
            "operation_id": port_snapshot.operation_id,
            "request_id": port_snapshot.request_id,
            "activation_grant_id": port_snapshot.activation_grant_id,
            "hold_reason": port_snapshot.hold_reason,
            "hold_succeeded": port_snapshot.hold_succeeded,
            "fault": port_snapshot.fault,
        }
        if owner_cut is not None:
            fingerprint.update(
                {
                    "owner_token_id": owner_cut.owner_token_id,
                    "owner_kind": owner_cut.owner_kind,
                    "owner_generation": owner_cut.owner_generation,
                    "authority_epoch": owner_cut.authority_epoch,
                    "execution_epoch": owner_cut.execution_epoch,
                    "lifecycle_state": owner_cut.lifecycle_state,
                    "execution_gate": owner_cut.execution_gate,
                }
            )
        return fingerprint

    def _observe_control_state(self) -> int:
        return self.revision_tracker.observe(
            self._control_fingerprint()
        ).station_revision

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

    def _capture_snapshot_with_read_evidence(
        self,
        *,
        active_operation_id: str | None,
    ) -> tuple[AgentObservationSnapshot, str, tuple[str, ...], float]:
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
        station_revision = self._observe_control_state()
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
        snapshot = AgentObservationSnapshot(
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
        return (
            snapshot,
            cut.semantic_content_hash,
            tuple(sorted((*owner.evidence_refs, *cut.evidence_refs))),
            now,
        )

    def capture_snapshot(
        self,
        *,
        active_operation_id: str | None,
    ) -> AgentObservationSnapshot:
        snapshot, _, _, _ = self._capture_snapshot_with_read_evidence(
            active_operation_id=active_operation_id,
        )
        return snapshot

    def read_tcp_pose(
        self,
        *,
        active_operation_id: str,
        parameters: Mapping[str, object],
    ) -> PassiveReadResult:
        route = self.contract_bundle.route_manifest.route(
            "state.read_tcp_pose",
            "1",
        ).route
        self.contract_bundle.payload_validators.validate_parameters(
            route=route,
            payload=parameters,
        )
        snapshot, source_measurement_hash, evidence_refs, cutoff_at = (
            self._capture_snapshot_with_read_evidence(
                active_operation_id=active_operation_id,
            )
        )
        attachment = self.contract_bundle.attachment
        requested_components = tuple(sorted(parameters["component_ids"]))
        all_poses = snapshot.measurement_summary[
            "pose_xyz_quat_xyzw_by_component"
        ]
        poses = {
            component_id: list(all_poses[component_id])
            for component_id in requested_components
        }
        measurement_observation_hash = canonical_content_hash(
            {
                "attachment_hash": attachment.semantic_content_hash,
                "source_measurement_hash": source_measurement_hash,
                "poses": poses,
            },
            domain="dexteleop.real_agent_motion.tcp_pose_measurement_observation.v1",
        )
        observation_cut_hash = canonical_content_hash(
            {
                "snapshot_id": snapshot.snapshot_id,
                "source_measurement_hash": source_measurement_hash,
                "cutoff_at": cutoff_at,
                "active_operation_id": active_operation_id,
            },
            domain="dexteleop.real_agent_motion.tcp_pose_read_cut.v1",
        )
        result = {
            "observation_cut_id": "real-agent-motion-cut:" + observation_cut_hash,
            "observation_cut_hash": observation_cut_hash,
            "cutoff_at": cutoff_at,
            "cutoff_clock_domain_id": attachment.clock_domain_id,
            "component_ids": list(requested_components),
            "reference_frame_id": attachment.reference_frame_id,
            "reference_frame_spec_hash": attachment.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": attachment.frame_graph_hash,
            "tool_config_hash": attachment.tool_config_hash,
            "calibration_hash": attachment.calibration_hash,
            "measurement_id": "real-agent-measured-pose:" + source_measurement_hash,
            "measurement_observation_hash": measurement_observation_hash,
            "source_measurement_hash": source_measurement_hash,
            "measured_at": snapshot.captured_at,
            "available_at": snapshot.available_at,
            "clock_domain_id": attachment.clock_domain_id,
            "pose_by_component": poses,
            "pose_schema_hash_by_component": {
                component_id: attachment.pose_schema_hash
                for component_id in requested_components
            },
            "representation_by_component": {
                component_id: "cartesian_pose"
                for component_id in requested_components
            },
            "observation_binding_id": attachment.observation_binding_id,
            "observation_plan_hash": attachment.observation_plan_hash,
        }
        self.contract_bundle.payload_validators.validate_execution_result(
            route=route,
            payload=result,
            request_parameters=parameters,
        )
        return PassiveReadResult(
            execution_result=result,
            evidence_refs=evidence_refs,
            reason_code="passive_real_tcp_pose_read_completed",
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
    timeout_sec: float,
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
    state_executor: PassiveReadSkillExecutor
    perception_provider: RealSenseAgentPerceptionProvider | None
    rgb_executor: PassiveReadSkillExecutor | None
    depth_executor: PassiveReadSkillExecutor | None
    artifact_store: RecoverableAgentArtifactStore | None
    owns_artifact_store: bool
    service: AgentGatewayService
    _close_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _close_outcome: AgentGatewayOperationOutcome | None = field(
        default=None,
        init=False,
    )
    _artifact_store_closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        enabled = self.contract_bundle.perception_capture is not None
        if enabled != (self.perception_provider is not None):
            raise ValueError("Real perception registration and provider disagree.")
        if enabled != (self.artifact_store is not None):
            raise ValueError("Real perception requires an exact Artifact store.")
        if enabled != (self.rgb_executor is not None and self.depth_executor is not None):
            raise ValueError("Real perception requires both passive executors.")
        if self.owns_artifact_store and self.artifact_store is None:
            raise ValueError("Owned Artifact store must be present.")
        if not isinstance(self.state_executor, PassiveReadSkillExecutor):
            raise TypeError("Real TCP-pose state executor must be passive.")

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
            passive_reads_settled = all(
                executor is None or executor.wait_all(timeout_sec=5.0)
                for executor in (
                    self.state_executor,
                    self.rgb_executor,
                    self.depth_executor,
                )
            )
            if not passive_reads_settled:
                raise RuntimeError(
                    "Passive real reads did not quiesce before Gateway close."
                )
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
            if (
                self.owns_artifact_store
                and self.artifact_store is not None
                and not self._artifact_store_closed
            ):
                self.artifact_store.close()
                self._artifact_store_closed = True
            return outcome


def assemble_real_agent_primitive_gateway_session(
    *,
    contract_bundle: RealAgentPrimitiveContractBundle,
    port: RealAgentMotionPort,
    owner_state_provider: Callable[[], RealControlOwnerState],
    allowed_client_ids: tuple[str, ...],
    max_retained_operations: int = 16,
    perception_provider: RealSenseAgentPerceptionProvider | None = None,
    artifact_store: RecoverableAgentArtifactStore | None = None,
    owns_artifact_store: bool = False,
    revision_tracker: AgentStationRevisionTracker | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> RealAgentPrimitiveGatewaySession:
    """Assemble motion and optional passive Routes without starting resources."""

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
    if not isinstance(owns_artifact_store, bool):
        raise TypeError("owns_artifact_store must be boolean.")
    perception_enabled = contract_bundle.perception_capture is not None
    if perception_enabled != isinstance(
        perception_provider,
        RealSenseAgentPerceptionProvider,
    ):
        raise ValueError("Real perception registration requires its exact provider.")
    if perception_enabled != isinstance(
        artifact_store,
        RecoverableAgentArtifactStore,
    ):
        raise ValueError("Real perception registration requires its Artifact store.")
    if owns_artifact_store and artifact_store is None:
        raise ValueError("Owned Artifact store must be injected.")
    producer = RealAgentPrimitiveObservationProducer(
        contract_bundle=contract_bundle,
        port=port,
        owner_state_provider=owner_state_provider,
        revision_tracker=revision_tracker,
        clock=clock,
    )
    if perception_provider is not None:
        assert contract_bundle.perception_capture is not None
        perception_provider.bind_profile(contract_bundle.perception_capture.profile)

    def domain_hold(*, operation, reason):
        del operation
        return _shared_port_hold_evidence(
            port=port,
            reason=reason,
            clock=clock,
            sleep=sleep,
            timeout_sec=(
                contract_bundle.convergence_profile.hold_confirmation_timeout_sec
            ),
        )

    executors: dict[str, ReusableStationAgentSkillOperationExecutor] = {}
    execution_port_by_route_hash = {}
    state_entry = contract_bundle.route_manifest.route(
        "state.read_tcp_pose",
        "1",
    )
    state_executor = PassiveReadSkillExecutor(
        port_id=(
            "agent-gateway:real-passive-tcp-pose:"
            + contract_bundle.route_manifest.runtime_session_id
        ),
        executor_id=contract_bundle.tcp_pose_observe.deployment_binding.executor_id,
        read=lambda request, operation: producer.read_tcp_pose(
            active_operation_id=operation.operation_id,
            parameters=request.parameters,
        ),
        current_station_revision=lambda: producer.current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )
    execution_port_by_route_hash[
        state_entry.route.semantic_content_hash
    ] = state_executor
    motion_entries = tuple(
        entry
        for entry in contract_bundle.route_manifest.skill_routes
        if entry.route.control_effect_class == "motion"
    )
    for entry in motion_entries:
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
    rgb_executor = None
    depth_executor = None
    if perception_provider is not None and artifact_store is not None:
        registration = contract_bundle.perception_capture
        assert registration is not None
        rgb_route = registration.rgb.routes[0]
        depth_route = registration.depth.routes[0]

        def read_rgb(request, operation) -> PassiveReadResult:
            capture = perception_provider.capture_rgb(
                request.parameters,
                expected_station_revision=request.expected_station_revision,
            )
            bindings = []
            frame_metadata = []
            total_bytes = 0
            for ordinal, frame in enumerate(capture.frames):
                artifact = artifact_ref_for_bytes(
                    frame.data,
                    media_type=frame.media_type,
                    schema_id=RGB_FRAME_SCHEMA_ID_V1,
                    encoding=RGB_FRAME_ENCODING_V1,
                    compression=RGB_FRAME_COMPRESSION_V1,
                )
                committed = commit_artifact_bytes(
                    store=artifact_store,
                    data=frame.data,
                    artifact=artifact,
                    producer_id="agent_gateway.provider.real_realsense_rgb",
                    runtime_session_id=contract_bundle.attachment.runtime_session_id,
                    idempotency_key=f"rgb-output:{operation.operation_id}:{ordinal}",
                    now=capture.captured_at,
                )
                total_bytes += committed.byte_length
                metadata = {
                    "ordinal": ordinal,
                    "source_id": frame.source_id,
                    "role": frame.role,
                    "media_type": committed.media_type,
                    "width": frame.width,
                    "height": frame.height,
                    "media_content_hash": committed.sha256,
                }
                frame_metadata.append(metadata)
                bindings.append(
                    AgentArtifactBinding(
                        slot_id=REAL_CAPTURE_RGB_OUTPUT_SLOT_ID_V1,
                        ordinal=ordinal,
                        artifact=committed,
                        production_receipt_ref=(
                            f"real-rgb-frame:{operation.operation_id}:{ordinal}:"
                            + committed.sha256
                        ),
                    )
                )
            identity = canonical_content_hash(
                {
                    "operation_id": operation.operation_id,
                    "request_hash": request.semantic_content_hash,
                    "station_revision": capture.station_revision,
                    "captured_at": capture.captured_at,
                    "frames": frame_metadata,
                },
                domain="dexteleop.agent_skill.real_rgb_capture_receipt.v1",
            )
            return PassiveReadResult(
                execution_result={
                    "snapshot_id": "rgb-snapshot:" + identity,
                    "station_revision": capture.station_revision,
                    "captured_at": capture.captured_at,
                    "frames": frame_metadata,
                    "total_bytes": total_bytes,
                },
                evidence_refs=("real-rgb-capture:" + identity,),
                output_artifact_bindings=tuple(bindings),
                reason_code="passive_rgb_capture_completed",
            )

        def read_depth(request, operation) -> PassiveReadResult:
            frame = perception_provider.capture_depth(
                role=str(request.parameters["role"]),
                expected_station_revision=request.expected_station_revision,
            )
            artifact = artifact_ref_for_bytes(
                frame.data,
                media_type=frame.media_type,
                schema_id=frame.schema_id,
                encoding=frame.encoding,
                compression=frame.compression,
            )
            committed = commit_artifact_bytes(
                store=artifact_store,
                data=frame.data,
                artifact=artifact,
                producer_id="agent_gateway.provider.real_realsense_depth",
                runtime_session_id=contract_bundle.attachment.runtime_session_id,
                idempotency_key="depth-output:" + operation.operation_id,
                now=frame.captured_at,
            )
            result = {
                "source_id": frame.source_id,
                "role": frame.role,
                "captured_at": frame.captured_at,
                "width": frame.width,
                "height": frame.height,
                "depth_unit": "meter",
                "invalid_depth_value": 0.0,
                "camera_pose_world_xyz_quat_xyzw": list(
                    frame.camera_pose_world_xyz_quat_xyzw
                ),
                "intrinsics": dict(frame.intrinsics),
            }
            identity = canonical_content_hash(
                {
                    "operation_id": operation.operation_id,
                    "request_hash": request.semantic_content_hash,
                    "result": result,
                    "artifact_ref_hash": committed.semantic_content_hash,
                },
                domain="dexteleop.agent_skill.real_depth_capture_receipt.v1",
            )
            evidence_ref = "real-depth-capture:" + identity
            return PassiveReadResult(
                execution_result=result,
                evidence_refs=(evidence_ref,),
                output_artifact_bindings=(
                    AgentArtifactBinding(
                        slot_id=REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
                        ordinal=0,
                        artifact=committed,
                        production_receipt_ref=evidence_ref,
                    ),
                ),
                reason_code="passive_depth_capture_completed",
            )

        rgb_executor = PassiveReadSkillExecutor(
            port_id=(
                "agent-gateway:real-passive-rgb:"
                + contract_bundle.attachment.runtime_session_id
            ),
            executor_id=registration.rgb_binding.executor_id,
            read=read_rgb,
            current_station_revision=lambda: producer.current_station_revision,
            clock=clock,
            max_retained_operations=max_retained_operations,
        )
        depth_executor = PassiveReadSkillExecutor(
            port_id=(
                "agent-gateway:real-passive-depth:"
                + contract_bundle.attachment.runtime_session_id
            ),
            executor_id=registration.depth_binding.executor_id,
            read=read_depth,
            current_station_revision=lambda: producer.current_station_revision,
            clock=clock,
            max_retained_operations=max_retained_operations,
        )
        execution_port_by_route_hash.update(
            {
                rgb_route.semantic_content_hash: rgb_executor,
                depth_route.semantic_content_hash: depth_executor,
            }
        )
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
    if artifact_store is None:
        protocol_runtime = AgentGatewayProtocolRuntime.control_plane_only(
            manifest=contract_bundle.route_manifest,
            clock=clock,
        )
    else:
        capabilities = AgentGatewaySessionCapabilities(
            runtime_session_id=contract_bundle.attachment.runtime_session_id,
            artifact_transfer_enabled=True,
            artifact_access_mechanisms=("chunked_rpc", "local_path"),
            max_control_request_bytes=2 * 1024 * 1024,
            max_control_response_bytes=2 * 1024 * 1024,
            max_artifact_bytes=artifact_store.max_artifact_bytes,
            max_artifact_chunk_bytes=artifact_store.max_chunk_bytes,
            max_artifact_refs_per_operation=8,
            max_concurrent_uploads=8,
            max_concurrent_downloads=8,
            artifact_store_quota_bytes=artifact_store.quota_bytes,
            event_payload_max_bytes=65_536,
            event_retention_count=4096,
        )
        broker = AgentArtifactTransferBroker(
            store=artifact_store,
            session_capabilities=capabilities,
            max_chunk_bytes=artifact_store.max_chunk_bytes,
        )
        protocol_runtime = AgentGatewayProtocolRuntime(
            manifest=contract_bundle.route_manifest,
            io_catalog=contract_bundle.io_catalog,
            session_capabilities=capabilities,
            event_journal=BoundedAgentGatewayEventJournal(
                runtime_session_id=contract_bundle.attachment.runtime_session_id,
                max_events=capabilities.event_retention_count,
                event_payload_max_bytes=capabilities.event_payload_max_bytes,
            ),
            artifact_store=artifact_store,
            artifact_transfer_broker=broker,
            clock=clock,
        )
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
        protocol_runtime=protocol_runtime,
        clock=clock,
        clock_domain_id=contract_bundle.attachment.clock_domain_id,
    )
    return RealAgentPrimitiveGatewaySession(
        contract_bundle=contract_bundle,
        port=port,
        observation_producer=producer,
        executors_by_skill_id=executors,
        state_executor=state_executor,
        perception_provider=perception_provider,
        rgb_executor=rgb_executor,
        depth_executor=depth_executor,
        artifact_store=artifact_store,
        owns_artifact_store=owns_artifact_store,
        service=service,
    )


__all__ = [
    "RealAgentPrimitiveGatewaySession",
    "RealAgentPrimitiveObservationProducer",
    "assemble_real_agent_primitive_gateway_session",
]
