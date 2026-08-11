"""Assemble one exact CubeLift Gateway session without starting it.

The session binds catalogs, snapshot/revision ownership, the bounded Gateway
service, one single-use motion lifecycle worker, and one reusable passive state
reader.  A sole domain-control port remains independent of exact Route
selection.  Construction owns no running socket server and performs no
application start; the explicit Unix-server builder binds an endpoint but
never starts the serve loop or lower application.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from control_station.agent_gateway import (
    AgentGatewayOperationOutcome,
    AgentGatewayService,
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
    AgentSkillAccessPolicy,
    BoundAgentGatewayWireAdapter,
    BoundedInMemoryAgentSkillOperationStore,
    BoundedStationAgentSkillOperationWorker,
    PassiveReadResult,
    PassiveReadSkillExecutor,
    SkillExecutorBinding,
    SkillExecutorRegistry,
    StationOperationWorkerBinding,
)

from .agent_observation import MujocoCubeLiftAgentObservationProducer
from .agent_skill_deployment import MujocoCubeLiftAgentSkillContractBundle


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
class MujocoCubeLiftAgentGatewaySession:
    """One constructed B3 session with single-use motion and reusable reads."""

    application: Any
    contract_bundle: MujocoCubeLiftAgentSkillContractBundle
    lifecycle: Any
    observation_producer: MujocoCubeLiftAgentObservationProducer
    worker: BoundedStationAgentSkillOperationWorker
    state_executor: PassiveReadSkillExecutor
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

    def wire_adapter(
        self,
        *,
        authenticated_client_id: str,
    ) -> BoundAgentGatewayWireAdapter:
        """Bind one already-authenticated connection to this exact session."""

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
        return self.worker.wait(timeout_sec=timeout_sec)

    def build_unix_json_server(
        self,
        *,
        socket_path: str,
        client_id_by_uid: Mapping[int, str],
        socket_mode: int = 0o600,
        max_frame_bytes: int = 2 * 1024 * 1024,
        peer_idle_timeout_sec: float = 10.0,
    ) -> AgentGatewayUnixJsonServer:
        """Bind the explicit local endpoint without starting its serve loop.

        The Control Station caller remains responsible for serving and
        shutdown. RPent never constructs this session or starts the lower
        application.
        """

        peer_map = AgentGatewayUnixPeerClientMap(client_id_by_uid)
        configured_clients = set(peer_map.client_id_by_uid.values())
        allowed_clients = set(self.service.access_policy.allowed_client_ids)
        if configured_clients != allowed_clients:
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

    def close(self) -> AgentGatewayOperationOutcome:
        """Close the Gateway through its HOLD-first service boundary once."""

        with self._close_lock:
            if self._close_outcome is None:
                self._close_outcome = self.service.close(
                    current_station_revision=(
                        self.observation_producer.current_station_revision
                    )
                )
            return self._close_outcome


def assemble_mujoco_cube_lift_agent_gateway_session(
    *,
    application: Any,
    contract_bundle: MujocoCubeLiftAgentSkillContractBundle,
    lifecycle: Any,
    allowed_client_ids: tuple[str, ...],
    max_retained_operations: int = 16,
    clock: Callable[[], float] | None = None,
) -> MujocoCubeLiftAgentGatewaySession:
    """Wire existing exact B3 artifacts; do not start the application."""

    if not isinstance(
        contract_bundle,
        MujocoCubeLiftAgentSkillContractBundle,
    ):
        raise TypeError("contract_bundle must be typed.")
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
    runtime = getattr(application, "runtime", None)
    runtime_clock = getattr(runtime, "clock", None) if clock is None else clock
    if not callable(runtime_clock):
        raise TypeError("clock must be callable.")
    if getattr(lifecycle, "application", None) is not application:
        raise ValueError("Lifecycle must own the exact application instance.")

    route_entry = contract_bundle.route_manifest.route("task.cube_lift", "1")
    route = route_entry.route
    lifecycle_route = getattr(lifecycle, "route", None)
    lifecycle_binding = getattr(lifecycle, "deployment_binding", None)
    exact_checks = (
        (
            getattr(lifecycle, "executor_id", None)
            == contract_bundle.deployment_binding.executor_id,
            "executor",
        ),
        (
            getattr(lifecycle_route, "semantic_content_hash", None)
            == route.semantic_content_hash,
            "Route",
        ),
        (
            getattr(lifecycle_binding, "semantic_content_hash", None)
            == contract_bundle.deployment_binding.semantic_content_hash,
            "deployment binding",
        ),
        (
            getattr(
                getattr(lifecycle, "evaluation_plan", None),
                "semantic_content_hash",
                None,
            )
            == contract_bundle.evaluation_plan.semantic_content_hash,
            "evaluation plan",
        ),
        (
            getattr(
                getattr(lifecycle, "station_manifest", None),
                "semantic_content_hash",
                None,
            )
            == contract_bundle.station_manifest.semantic_content_hash,
            "Station manifest",
        ),
        (
            getattr(
                getattr(lifecycle, "execution_manifest", None),
                "semantic_content_hash",
                None,
            )
            == contract_bundle.execution_manifest.semantic_content_hash,
            "execution manifest",
        ),
    )
    for matches, label in exact_checks:
        if not matches:
            raise ValueError(f"Lifecycle {label} does not match contract bundle.")

    observation_producer = MujocoCubeLiftAgentObservationProducer(
        application=application,
        contract_bundle=contract_bundle,
    )
    worker = BoundedStationAgentSkillOperationWorker(
        port_id=(
            "agent-gateway:station-worker:"
            + contract_bundle.deployment_binding.runtime_session_id
        ),
        binding=StationOperationWorkerBinding(
            worker_id=(
                "cube-lift-worker:"
                + contract_bundle.deployment_binding.runtime_session_id
            ),
            executor_id=contract_bundle.deployment_binding.executor_id,
            runtime_session_id=(
                contract_bundle.deployment_binding.runtime_session_id
            ),
            skill_id=route.skill_id,
            skill_version=route.skill_version,
            route_id=route.route_id,
            route_version=route.route_version,
            deployment_binding_hash=(
                contract_bundle.deployment_binding.semantic_content_hash
            ),
        ),
        lifecycle=lifecycle,
        current_station_revision=lambda: (
            observation_producer.current_station_revision
        ),
        clock=runtime_clock,
    )
    state_entry = contract_bundle.route_manifest.route(
        "state.read_tcp_pose",
        "1",
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
            "agent-gateway:passive-tcp-pose:"
            + contract_bundle.deployment_binding.runtime_session_id
        ),
        executor_id=contract_bundle.state_deployment_binding.executor_id,
        read=read_tcp_pose,
        current_station_revision=lambda: (
            observation_producer.current_station_revision
        ),
        clock=runtime_clock,
        max_retained_operations=retention,
    )
    executor_registry = SkillExecutorRegistry(
        manifest=contract_bundle.route_manifest,
        bindings=tuple(
            SkillExecutorBinding.from_manifest_entry(
                manifest=contract_bundle.route_manifest,
                entry=entry,
                execution_port=(
                    state_executor
                    if entry.route.semantic_content_hash
                    == state_entry.route.semantic_content_hash
                    else worker
                ),
            )
            for entry in contract_bundle.route_manifest.skill_routes
        ),
    )
    access_policy = AgentSkillAccessPolicy(
        access_policy_id=(
            "agent-access:cube-lift:"
            + contract_bundle.deployment_binding.runtime_session_id
        ),
        allowed_client_ids=clients,
        allowed_modes=(contract_bundle.route_manifest.current_mode,),
        max_request_timeout_ms=route.max_operation_duration_ms,
        max_grant_lifetime_sec=30.0,
    )
    service = AgentGatewayService(
        manifest=contract_bundle.route_manifest,
        access_policy=access_policy,
        executor_registry=executor_registry,
        domain_control_port=worker,
        payload_validators=contract_bundle.payload_validators,
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=retention,
            max_admissions_by_route_hash={
                route.semantic_content_hash: 1,
                state_entry.route.semantic_content_hash: retention,
            },
        ),
        clock=runtime_clock,
        clock_domain_id=_required(
            getattr(getattr(runtime, "config", None), "clock_domain_id", None),
            field_name="runtime clock_domain_id",
        ),
    )
    return MujocoCubeLiftAgentGatewaySession(
        application=application,
        contract_bundle=contract_bundle,
        lifecycle=lifecycle,
        observation_producer=observation_producer,
        worker=worker,
        state_executor=state_executor,
        service=service,
    )


__all__ = [
    "MujocoCubeLiftAgentGatewaySession",
    "assemble_mujoco_cube_lift_agent_gateway_session",
]
