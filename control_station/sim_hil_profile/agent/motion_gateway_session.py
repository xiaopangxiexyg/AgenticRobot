"""Assemble the reusable no-model TCP-pose Agent Gateway session.

Construction is default-off: it creates contracts, executors and a service but
does not start MuJoCo, advance physics, bind a socket or admit motion.  Each
sequential motion operation gets a fresh lifecycle while all operations share
the same lower runtime, dispatch owner and domain-HOLD boundary.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from collection_runtime.planning_history import (
    PlanningHistoryRecorder,
    SQLitePlanningHistoryStore,
)
from control_station.agent_gateway.runtime.access_policy import AgentSkillAccessPolicy
from control_station.agent_gateway.runtime.executor_registry import (
    SkillExecutorBinding,
    SkillExecutorRegistry,
)
from control_station.agent_gateway.runtime.operation_port import (
    AgentGatewayAuthoritativePortReceipt,
)
from control_station.agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from control_station.agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from control_station.agent_gateway.runtime.reusable_station_executor import (
    ReusableStationAgentSkillOperationExecutor,
)
from control_station.agent_gateway.runtime.service import (
    AgentGatewayOperationOutcome,
    AgentGatewayService,
)
from control_station.agent_gateway.runtime.station_operation_worker import (
    StationOperationWorkerBinding,
)
from control_station.agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from control_station.agent_gateway.transport.wire_adapter import (
    BoundAgentGatewayWireAdapter,
)

from .motion_application import MujocoAgentSkillMotionApplication
from .motion_deployment import MujocoAgentTcpPoseContractBundle
from .motion_skill import (
    MujocoAgentTcpPoseSkillLifecycle,
    mujoco_agent_skill_hold_evidence_from_transition,
)
from .observation import MujocoAgentObservationProducer
from .observation_binding import (
    build_mujoco_agent_motion_observation_source_binding,
)
from .tcp_pose_history import MujocoAgentTcpPoseHistoryProvider


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


@dataclass
class MujocoAgentTcpPoseGatewaySession:
    """One reusable exact-Route session over one lower motion owner."""

    application: MujocoAgentSkillMotionApplication
    contract_bundle: MujocoAgentTcpPoseContractBundle
    observation_producer: MujocoAgentObservationProducer
    motion_executor: ReusableStationAgentSkillOperationExecutor
    state_executor: PassiveReadSkillExecutor
    history_provider: MujocoAgentTcpPoseHistoryProvider | None
    history_executor: PassiveReadSkillExecutor | None
    history_recorder: PlanningHistoryRecorder | None
    planning_history_store: SQLitePlanningHistoryStore | None
    owns_planning_history_store: bool
    service: AgentGatewayService
    _close_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _close_outcome: AgentGatewayOperationOutcome | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _planning_history_store_closed: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.owns_planning_history_store, bool):
            raise TypeError("owns_planning_history_store must be boolean.")
        if self.owns_planning_history_store and self.planning_history_store is None:
            raise ValueError("Owned planning history store must be present.")

    def wire_adapter(
        self,
        *,
        authenticated_client_id: str,
    ) -> BoundAgentGatewayWireAdapter:
        return BoundAgentGatewayWireAdapter(
            authenticated_client_id=authenticated_client_id,
            definition_catalog=self.contract_bundle.definition_catalog,
            schema_catalog=self.contract_bundle.schema_catalog,
            service=self.service,
            publish_observation=lambda: (
                self.observation_producer.capture_and_publish(self.service)
            ),
            current_station_revision=lambda: (
                self.observation_producer.current_station_revision
            ),
            operation_launcher=self.service.launch_operation,
        )

    def wait(self, *, timeout_sec: float | None = None) -> bool:
        return self.motion_executor.wait(timeout_sec=timeout_sec)

    def build_unix_json_server(
        self,
        *,
        socket_path: str,
        client_id_by_uid: Mapping[int, str],
        socket_mode: int = 0o600,
        max_frame_bytes: int = 2 * 1024 * 1024,
        peer_idle_timeout_sec: float = 10.0,
    ) -> AgentGatewayUnixJsonServer:
        """Bind the local endpoint without starting its serve loop."""

        peer_map = AgentGatewayUnixPeerClientMap(client_id_by_uid)
        if set(peer_map.client_id_by_uid.values()) != set(
            self.service.access_policy.allowed_client_ids
        ):
            raise ValueError(
                "Unix peer client map must exactly cover the session access policy."
            )
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

    def _finish_lower_close(
        self,
        outcome: AgentGatewayOperationOutcome,
    ) -> None:
        receipt = outcome.port_receipt
        if not (
            outcome.accepted
            and isinstance(receipt, AgentGatewayAuthoritativePortReceipt)
            and receipt.succeeded
            and receipt.proves_motion_or_hold
        ):
            if not self.application.started:
                self.application.close_after_hold()
            return
        self.motion_executor.close_after_hold(hold_receipt=receipt)
        workers = self.motion_executor.workers
        if not workers or self.motion_executor.wait(timeout_sec=5.0):
            self.application.close_after_hold()

    def close(self) -> AgentGatewayOperationOutcome:
        """Fence admissions, establish lower HOLD, then close when quiescent."""

        with self._close_lock:
            if self.history_recorder is not None:
                self.history_recorder.close(timeout_sec=1.0)
            if self._close_outcome is None:
                self._close_outcome = self.service.close(
                    current_station_revision=(
                        self.observation_producer.current_station_revision
                    )
                )
            self._finish_lower_close(self._close_outcome)
            if (
                self.owns_planning_history_store
                and self.planning_history_store is not None
                and not self._planning_history_store_closed
            ):
                self.planning_history_store.close()
                self._planning_history_store_closed = True
            return self._close_outcome


def assemble_mujoco_agent_tcp_pose_gateway_session(
    *,
    application: MujocoAgentSkillMotionApplication,
    contract_bundle: MujocoAgentTcpPoseContractBundle,
    allowed_client_ids: tuple[str, ...],
    max_retained_operations: int = 16,
    clock: Callable[[], float] | None = None,
    planning_history_store: SQLitePlanningHistoryStore | None = None,
    owns_planning_history_store: bool = False,
) -> MujocoAgentTcpPoseGatewaySession:
    """Wire exact Routes without starting the application or history recorder."""

    if not isinstance(application, MujocoAgentSkillMotionApplication):
        raise TypeError("application must be a motion application.")
    if not isinstance(contract_bundle, MujocoAgentTcpPoseContractBundle):
        raise TypeError("contract_bundle must be typed.")
    if not isinstance(owns_planning_history_store, bool):
        raise TypeError("owns_planning_history_store must be boolean.")
    history_enabled = contract_bundle.history_profile is not None
    if history_enabled and not isinstance(
        planning_history_store,
        SQLitePlanningHistoryStore,
    ):
        raise ValueError("Enabled history requires an exact injected disk store.")
    if not history_enabled and planning_history_store is not None:
        raise ValueError("A history store cannot be bound without a history Route.")
    if owns_planning_history_store and planning_history_store is None:
        raise ValueError("Owned planning history store must be injected.")
    clients = tuple(
        _required(value, field_name="allowed_client_ids item")
        for value in allowed_client_ids
    )
    if not clients or len(clients) != len(set(clients)):
        raise ValueError("allowed_client_ids must be non-empty and unique.")
    retention = _positive_integer(
        max_retained_operations,
        field_name="max_retained_operations",
    )
    runtime = application.runtime
    runtime_clock = runtime.clock if clock is None else clock
    if not callable(runtime_clock):
        raise TypeError("clock must be callable.")
    binding = contract_bundle.deployment_binding
    checks = (
        (
            runtime.config.runtime_session_id == binding.runtime_session_id,
            "runtime session",
        ),
        (
            runtime.config.control_domain_id == binding.control_domain_id,
            "control domain",
        ),
        (
            contract_bundle.station_manifest.semantic_content_hash
            == binding.station_manifest_hash,
            "Station manifest",
        ),
        (
            contract_bundle.execution_manifest.semantic_content_hash
            == binding.execution_manifest_hash,
            "execution manifest",
        ),
        (
            contract_bundle.motion_profile.semantic_content_hash
            == binding.motion_profile_hash,
            "motion profile",
        ),
    )
    for matches, label in checks:
        if not matches:
            raise ValueError(f"Agent TCP pose session {label} mismatch.")

    motion_entry = contract_bundle.route_manifest.route(
        "motion.move_tcp_to_pose",
        "1",
    )
    motion_route = motion_entry.route
    state_entry = contract_bundle.route_manifest.route(
        "state.read_tcp_pose",
        "1",
    )
    history_entry = (
        contract_bundle.route_manifest.route(
            "state.read_tcp_pose_history",
            "1",
        )
        if history_enabled
        else None
    )
    observation_producer = MujocoAgentObservationProducer(
        application=application,
        contract_bundle=contract_bundle,
        source_binding=build_mujoco_agent_motion_observation_source_binding(
            application=application,
            contract_bundle=contract_bundle,
        ),
    )
    history_provider = (
        MujocoAgentTcpPoseHistoryProvider(
            observation_producer=observation_producer,
            contract_bundle=contract_bundle,
            store=planning_history_store,
        )
        if isinstance(planning_history_store, SQLitePlanningHistoryStore)
        else None
    )

    def lifecycle_factory(*, request, operation):
        return MujocoAgentTcpPoseSkillLifecycle(
            application=application,
            route=motion_route,
            deployment_binding=binding,
            evaluation_plan=contract_bundle.evaluation_plan,
            motion_profile=contract_bundle.motion_profile,
            request=request,
            operation=operation,
            clock=runtime_clock,
        )

    def domain_hold(*, operation, reason):
        transition = application.request_hold(
            reason=reason,
            operation_id=(
                None if operation is None else operation.operation_id
            ),
        )
        return mujoco_agent_skill_hold_evidence_from_transition(
            transition,
            lower_operation_id=(
                None if operation is None else operation.lower_operation_id
            ),
        )

    motion_executor = ReusableStationAgentSkillOperationExecutor(
        port_id="agent-gateway:station-worker:" + binding.runtime_session_id,
        binding=StationOperationWorkerBinding(
            worker_id="agent-tcp-pose-worker:" + binding.runtime_session_id,
            executor_id=binding.executor_id,
            runtime_session_id=binding.runtime_session_id,
            skill_id=motion_route.skill_id,
            skill_version=motion_route.skill_version,
            route_id=motion_route.route_id,
            route_version=motion_route.route_version,
            deployment_binding_hash=binding.semantic_content_hash,
        ),
        lifecycle_factory=lifecycle_factory,
        domain_hold=domain_hold,
        current_station_revision=lambda: (
            observation_producer.current_station_revision
        ),
        clock=runtime_clock,
    )

    def read_tcp_pose(request, operation) -> PassiveReadResult:
        state_read = observation_producer.capture_tcp_pose_state(
            active_operation_id=operation.operation_id,
            parameters=request.parameters,
        )
        return PassiveReadResult(
            execution_result=state_read.execution_result,
            evidence_refs=state_read.evidence_refs,
            reason_code="passive_tcp_pose_read_completed",
        )

    state_executor = PassiveReadSkillExecutor(
        port_id=(
            "agent-gateway:passive-tcp-pose:" + binding.runtime_session_id
        ),
        executor_id=contract_bundle.state_deployment_binding.executor_id,
        read=read_tcp_pose,
        current_station_revision=lambda: (
            observation_producer.current_station_revision
        ),
        clock=runtime_clock,
        max_retained_operations=retention,
    )
    history_executor = None
    if history_provider is not None:
        history_binding = contract_bundle.history_deployment_binding
        if history_binding is None:
            raise ValueError("History deployment binding is unavailable.")

        def read_tcp_pose_history(request, operation) -> PassiveReadResult:
            state_read = history_provider.read(
                active_operation_id=operation.operation_id,
                parameters=request.parameters,
            )
            return PassiveReadResult(
                execution_result=state_read.execution_result,
                evidence_refs=state_read.evidence_refs,
                reason_code="passive_tcp_pose_history_read_completed",
            )

        history_executor = PassiveReadSkillExecutor(
            port_id=(
                "agent-gateway:passive-tcp-pose-history:" + binding.runtime_session_id
            ),
            executor_id=history_binding.executor_id,
            read=read_tcp_pose_history,
            current_station_revision=lambda: (
                observation_producer.current_station_revision
            ),
            clock=runtime_clock,
            max_retained_operations=retention,
        )
    execution_port_by_route_hash = {
        motion_route.semantic_content_hash: motion_executor,
        state_entry.route.semantic_content_hash: state_executor,
    }
    if history_entry is not None and history_executor is not None:
        execution_port_by_route_hash[history_entry.route.semantic_content_hash] = (
            history_executor
        )
    executor_registry = SkillExecutorRegistry(
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
        access_policy_id="agent-access:tcp-pose:" + binding.runtime_session_id,
        allowed_client_ids=clients,
        allowed_modes=(contract_bundle.route_manifest.current_mode,),
        max_request_timeout_ms=motion_route.max_operation_duration_ms,
        max_grant_lifetime_sec=30.0,
    )
    admissions_by_route_hash = {
        motion_route.semantic_content_hash: retention,
        state_entry.route.semantic_content_hash: retention,
    }
    if history_entry is not None:
        admissions_by_route_hash[history_entry.route.semantic_content_hash] = retention
    service = AgentGatewayService(
        manifest=contract_bundle.route_manifest,
        access_policy=access_policy,
        executor_registry=executor_registry,
        domain_control_port=motion_executor,
        payload_validators=contract_bundle.payload_validators,
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=retention,
            max_admissions_by_route_hash=admissions_by_route_hash,
        ),
        clock=runtime_clock,
        clock_domain_id=_required(
            runtime.config.clock_domain_id,
            field_name="runtime clock_domain_id",
        ),
    )
    history_recorder = (
        PlanningHistoryRecorder(
            store=planning_history_store,
            capture=lambda: history_provider.capture_record(
                active_operation_id=service.operation_store.active_operation_id,
            ),
            interval_sec=contract_bundle.history_profile.capture_interval_sec,
        )
        if history_provider is not None
        and planning_history_store is not None
        and contract_bundle.history_profile is not None
        else None
    )
    return MujocoAgentTcpPoseGatewaySession(
        application=application,
        contract_bundle=contract_bundle,
        observation_producer=observation_producer,
        motion_executor=motion_executor,
        state_executor=state_executor,
        history_provider=history_provider,
        history_executor=history_executor,
        history_recorder=history_recorder,
        planning_history_store=planning_history_store,
        owns_planning_history_store=owns_planning_history_store,
        service=service,
    )


__all__ = [
    "MujocoAgentTcpPoseGatewaySession",
    "assemble_mujoco_agent_tcp_pose_gateway_session",
]
