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

from collection_runtime.source_observation_history import (
    SourceObservationHistoryRecorder,
    SQLiteSourceObservationHistoryStore,
)
from agent_gateway.profiles.motion_primitive_resolution import (
    MotionPrimitiveTargetResolution,
    resolve_level_tcp_orientation_primitive_target,
    resolve_tcp_displacement_primitive_target,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.session_contracts import AgentGatewaySessionCapabilities
from agent_gateway.runtime.access_policy import AgentSkillAccessPolicy
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.artifact_transfer import AgentArtifactTransferBroker
from agent_gateway.runtime.event_journal import BoundedAgentGatewayEventJournal
from agent_gateway.runtime.executor_registry import (
    SkillExecutorBinding,
    SkillExecutorRegistry,
)
from agent_gateway.runtime.operation_port import (
    AgentGatewayAuthoritativePortReceipt,
)
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
from agent_gateway.runtime.station_operation_worker import (
    StationOperationWorkerBinding,
)
from agent_gateway.skills.motion.definition import (
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
)
from agent_gateway.skills.perception.definition import CAPTURE_DEPTH_SKILL_ID_V1
from agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from agent_gateway.transport.wire_adapter import (
    BoundAgentGatewayWireAdapter,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .depth_deployment import MUJOCO_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1
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
from .visual_observation import MujocoAgentVisualSnapshotProvider


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
    visual_snapshot_provider: MujocoAgentVisualSnapshotProvider
    motion_executor: ReusableStationAgentSkillOperationExecutor
    cardinal_step_executor: ReusableStationAgentSkillOperationExecutor
    fixed_orientation_executors: Mapping[
        str, ReusableStationAgentSkillOperationExecutor
    ]
    state_executor: PassiveReadSkillExecutor
    depth_executor: PassiveReadSkillExecutor | None
    history_provider: MujocoAgentTcpPoseHistoryProvider | None
    history_executor: PassiveReadSkillExecutor | None
    history_recorder: SourceObservationHistoryRecorder | None
    source_observation_history_store: SQLiteSourceObservationHistoryStore | None
    owns_source_observation_history_store: bool
    artifact_store: RecoverableAgentArtifactStore | None
    owns_artifact_store: bool
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
    _source_observation_history_store_closed: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _artifact_store_closed: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if set(self.fixed_orientation_executors) != set(
            FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
        ):
            raise ValueError(
                "fixed_orientation_executors must exactly cover three Skills."
            )
        if not isinstance(self.owns_source_observation_history_store, bool):
            raise TypeError("owns_source_observation_history_store must be boolean.")
        if not isinstance(self.owns_artifact_store, bool):
            raise TypeError("owns_artifact_store must be boolean.")
        if (
            self.owns_source_observation_history_store
            and self.source_observation_history_store is None
        ):
            raise ValueError("Owned planning history store must be present.")
        if self.owns_artifact_store and self.artifact_store is None:
            raise ValueError("Owned Artifact store must be present.")
        if (self.depth_executor is None) != (self.artifact_store is None):
            raise ValueError(
                "Depth executor and Artifact store must be enabled together."
            )

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
            visual_source_catalog=self.visual_snapshot_provider.source_catalog,
            capture_visual_snapshot=self.visual_snapshot_provider.capture,
            operation_launcher=self.service.launch_operation,
        )

    def wait(self, *, timeout_sec: float | None = None) -> bool:
        active = tuple(
            executor
            for executor in self._motion_executors()
            if executor.active_operation_id is not None
        )
        return bool(active) and all(
            executor.wait(timeout_sec=timeout_sec) for executor in active
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
        for executor in self._motion_executors()[1:]:
            executor_receipt = executor.request_hold(
                operation=None,
                reason="agent_gateway_close_motion_route_fence",
            )
            if (
                executor_receipt.succeeded
                and executor_receipt.proves_motion_or_hold
            ):
                executor.close_after_hold(hold_receipt=executor_receipt)
        workers = tuple(
            worker
            for executor in self._motion_executors()
            for worker in executor.workers
        )
        settled = all(
            executor.active_operation_id is None
            or executor.wait(timeout_sec=5.0)
            for executor in self._motion_executors()
        )
        if not workers or settled:
            self.application.close_after_hold()

    def _motion_executors(
        self,
    ) -> tuple[ReusableStationAgentSkillOperationExecutor, ...]:
        return (
            self.motion_executor,
            self.cardinal_step_executor,
            *tuple(
                self.fixed_orientation_executors[skill_id]
                for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
            ),
        )

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
            depth_settled = self.depth_executor is None or self.depth_executor.wait_all(
                timeout_sec=5.0
            )
            if not depth_settled:
                raise RuntimeError(
                    "Passive depth capture did not quiesce before Gateway close."
                )
            self._finish_lower_close(self._close_outcome)
            if (
                self.owns_source_observation_history_store
                and self.source_observation_history_store is not None
                and not self._source_observation_history_store_closed
            ):
                self.source_observation_history_store.close()
                self._source_observation_history_store_closed = True
            if (
                self.owns_artifact_store
                and self.artifact_store is not None
                and not self._artifact_store_closed
            ):
                self.artifact_store.close()
                self._artifact_store_closed = True
            return self._close_outcome


def assemble_mujoco_agent_tcp_pose_gateway_session(
    *,
    application: MujocoAgentSkillMotionApplication,
    contract_bundle: MujocoAgentTcpPoseContractBundle,
    allowed_client_ids: tuple[str, ...],
    max_retained_operations: int = 16,
    clock: Callable[[], float] | None = None,
    source_observation_history_store: SQLiteSourceObservationHistoryStore | None = None,
    owns_source_observation_history_store: bool = False,
    artifact_store: RecoverableAgentArtifactStore | None = None,
    owns_artifact_store: bool = False,
) -> MujocoAgentTcpPoseGatewaySession:
    """Wire exact Routes without starting the application or history recorder."""

    if not isinstance(application, MujocoAgentSkillMotionApplication):
        raise TypeError("application must be a motion application.")
    if not isinstance(contract_bundle, MujocoAgentTcpPoseContractBundle):
        raise TypeError("contract_bundle must be typed.")
    if not isinstance(owns_source_observation_history_store, bool):
        raise TypeError("owns_source_observation_history_store must be boolean.")
    if not isinstance(owns_artifact_store, bool):
        raise TypeError("owns_artifact_store must be boolean.")
    history_enabled = contract_bundle.history_profile is not None
    if history_enabled and not isinstance(
        source_observation_history_store,
        SQLiteSourceObservationHistoryStore,
    ):
        raise ValueError("Enabled history requires an exact injected disk store.")
    if not history_enabled and source_observation_history_store is not None:
        raise ValueError("A history store cannot be bound without a history Route.")
    if (
        owns_source_observation_history_store
        and source_observation_history_store is None
    ):
        raise ValueError("Owned planning history store must be injected.")
    depth_enabled = contract_bundle.depth_capture is not None
    if depth_enabled and not isinstance(
        artifact_store,
        RecoverableAgentArtifactStore,
    ):
        raise ValueError("Enabled depth capture requires an exact Artifact store.")
    if not depth_enabled and artifact_store is not None:
        raise ValueError("Artifact store cannot be bound without an Artifact Skill.")
    if owns_artifact_store and artifact_store is None:
        raise ValueError("Owned Artifact store must be injected.")
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
    cardinal_step_entry = contract_bundle.route_manifest.route(
        "motion.step_tcp_cardinal",
        "1",
    )
    cardinal_step_route = cardinal_step_entry.route
    fixed_orientation_entry_by_skill_id = {
        skill_id: contract_bundle.route_manifest.route(skill_id, "1")
        for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
    }
    fixed_orientation_contract_by_skill_id = {
        item.profile.skill_id: item
        for item in contract_bundle.fixed_orientation_contracts
    }
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
    depth_entry = (
        contract_bundle.route_manifest.route(CAPTURE_DEPTH_SKILL_ID_V1, "1")
        if depth_enabled
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
    visual_snapshot_provider = MujocoAgentVisualSnapshotProvider(
        application=application,
        current_station_revision=lambda: (
            observation_producer.current_station_revision
        ),
    )
    history_provider = (
        MujocoAgentTcpPoseHistoryProvider(
            observation_producer=observation_producer,
            contract_bundle=contract_bundle,
            store=source_observation_history_store,
        )
        if isinstance(
            source_observation_history_store,
            SQLiteSourceObservationHistoryStore,
        )
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

    frame_graph = application.target_frame_profile.frame_graph
    reference_frame_spec = frame_graph.frame_by_id[
        contract_bundle.cardinal_step_profile.reference_frame_id
    ]

    def cardinal_step_lifecycle_factory(*, request, operation):
        parameters = request.parameters
        if not isinstance(parameters, Mapping):
            raise TypeError("Cardinal-step parameters must be an object.")
        primitive = contract_bundle.cardinal_step_profile.primitive(
            str(parameters.get("direction"))
        )

        def resolve_target(
            measured_pose: tuple[float, ...],
        ) -> MotionPrimitiveTargetResolution:
            return resolve_tcp_displacement_primitive_target(
                measured_pose=measured_pose,
                primitive=primitive,
                reference_frame_spec=reference_frame_spec,
                frame_graph=frame_graph,
                measured_reference_frame_id=(
                    contract_bundle.cardinal_step_profile.execution_reference_frame_id
                ),
            )

        return MujocoAgentTcpPoseSkillLifecycle(
            application=application,
            route=cardinal_step_route,
            deployment_binding=(
                contract_bundle.cardinal_step_deployment_binding
            ),
            evaluation_plan=(
                contract_bundle.cardinal_step_evaluation_plan
            ),
            motion_profile=contract_bundle.motion_profile,
            request=request,
            operation=operation,
            target_resolver=resolve_target,
            parameter_lineage={
                "reference_frame_id": (
                    contract_bundle.cardinal_step_profile.reference_frame_id
                ),
                "reference_frame_spec_hash": (
                    contract_bundle.cardinal_step_profile.reference_frame_spec_hash
                ),
                "controlled_frame_id": "tool_tcp",
                "frame_graph_hash": (
                    contract_bundle.cardinal_step_profile.frame_graph_hash
                ),
                "tool_config_hash": (
                    contract_bundle.cardinal_step_profile.tool_config_hash
                ),
                "calibration_hash": (
                    contract_bundle.cardinal_step_profile.calibration_hash
                ),
            },
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
    cardinal_step_binding = contract_bundle.cardinal_step_deployment_binding
    cardinal_step_executor = ReusableStationAgentSkillOperationExecutor(
        port_id=(
            "agent-gateway:cardinal-step-worker:"
            + cardinal_step_binding.runtime_session_id
        ),
        binding=StationOperationWorkerBinding(
            worker_id=(
                "agent-tcp-cardinal-step-worker:"
                + cardinal_step_binding.runtime_session_id
            ),
            executor_id=cardinal_step_binding.executor_id,
            runtime_session_id=cardinal_step_binding.runtime_session_id,
            skill_id=cardinal_step_route.skill_id,
            skill_version=cardinal_step_route.skill_version,
            route_id=cardinal_step_route.route_id,
            route_version=cardinal_step_route.route_version,
            deployment_binding_hash=(
                cardinal_step_binding.semantic_content_hash
            ),
        ),
        lifecycle_factory=cardinal_step_lifecycle_factory,
        domain_hold=domain_hold,
        current_station_revision=lambda: (
            observation_producer.current_station_revision
        ),
        clock=runtime_clock,
    )
    fixed_orientation_executors: dict[
        str, ReusableStationAgentSkillOperationExecutor
    ] = {}
    for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1:
        orientation_entry = fixed_orientation_entry_by_skill_id[skill_id]
        orientation_route = orientation_entry.route
        orientation_contract = fixed_orientation_contract_by_skill_id[skill_id]
        orientation_profile = orientation_contract.profile
        orientation_reference_frame_spec = frame_graph.frame_by_id[
            orientation_profile.reference_frame_id
        ]

        def orientation_lifecycle_factory(
            *,
            request,
            operation,
            _route=orientation_route,
            _contract=orientation_contract,
            _profile=orientation_profile,
            _reference_frame_spec=orientation_reference_frame_spec,
        ):
            def resolve_target(
                measured_pose: tuple[float, ...],
            ) -> MotionPrimitiveTargetResolution:
                return resolve_level_tcp_orientation_primitive_target(
                    measured_pose=measured_pose,
                    primitive=_profile.primitive,
                    reference_frame_spec=_reference_frame_spec,
                    frame_graph=frame_graph,
                    measured_reference_frame_id=(
                        _profile.execution_reference_frame_id
                    ),
                )

            return MujocoAgentTcpPoseSkillLifecycle(
                application=application,
                route=_route,
                deployment_binding=_contract.deployment_binding,
                evaluation_plan=_contract.evaluation_plan,
                motion_profile=_contract.motion_profile,
                request=request,
                operation=operation,
                target_resolver=resolve_target,
                parameter_lineage={
                    "reference_frame_id": _profile.reference_frame_id,
                    "reference_frame_spec_hash": (
                        _profile.reference_frame_spec_hash
                    ),
                    "controlled_frame_id": "tool_tcp",
                    "frame_graph_hash": _profile.frame_graph_hash,
                    "tool_config_hash": _profile.tool_config_hash,
                    "calibration_hash": _profile.calibration_hash,
                },
                clock=runtime_clock,
            )

        orientation_binding = orientation_contract.deployment_binding
        fixed_orientation_executors[skill_id] = (
            ReusableStationAgentSkillOperationExecutor(
                port_id=(
                    "agent-gateway:fixed-orientation-worker:"
                    + skill_id
                    + ":"
                    + orientation_binding.runtime_session_id
                ),
                binding=StationOperationWorkerBinding(
                    worker_id=(
                        "agent-tcp-fixed-orientation-worker:"
                        + skill_id
                        + ":"
                        + orientation_binding.runtime_session_id
                    ),
                    executor_id=orientation_binding.executor_id,
                    runtime_session_id=orientation_binding.runtime_session_id,
                    skill_id=orientation_route.skill_id,
                    skill_version=orientation_route.skill_version,
                    route_id=orientation_route.route_id,
                    route_version=orientation_route.route_version,
                    deployment_binding_hash=(
                        orientation_binding.semantic_content_hash
                    ),
                ),
                lifecycle_factory=orientation_lifecycle_factory,
                domain_hold=domain_hold,
                current_station_revision=lambda: (
                    observation_producer.current_station_revision
                ),
                clock=runtime_clock,
            )
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
    depth_executor = None
    if depth_entry is not None and artifact_store is not None:
        depth_registration = contract_bundle.depth_capture
        if depth_registration is None:
            raise RuntimeError("Depth Route lost its registration.")

        def read_depth(request, operation) -> PassiveReadResult:
            frame = application.capture_depth_npy_frame(
                view_id=str(request.parameters["view_id"]),
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
                producer_id="agent_gateway.provider.mujoco_depth",
                runtime_session_id=binding.runtime_session_id,
                idempotency_key="depth-output:" + operation.operation_id,
                now=frame.captured_at,
            )
            result = {
                "view_id": frame.view_id,
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
            capture_ref = "mujoco-depth-capture:" + canonical_content_hash(
                {
                    "operation_id": operation.operation_id,
                    "request_hash": request.semantic_content_hash,
                    "result": result,
                    "artifact_ref_hash": committed.semantic_content_hash,
                },
                domain="dexteleop.agent_skill.mujoco_depth_capture_receipt.v1",
            )
            return PassiveReadResult(
                execution_result=result,
                evidence_refs=(capture_ref,),
                output_artifact_bindings=(
                    AgentArtifactBinding(
                        slot_id=MUJOCO_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
                        ordinal=0,
                        artifact=committed,
                        production_receipt_ref=capture_ref,
                    ),
                ),
                reason_code="passive_depth_capture_completed",
            )

        depth_executor = PassiveReadSkillExecutor(
            port_id=(
                "agent-gateway:passive-depth-capture:" + binding.runtime_session_id
            ),
            executor_id=depth_registration.deployment_binding.executor_id,
            read=read_depth,
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
        cardinal_step_route.semantic_content_hash: cardinal_step_executor,
        state_entry.route.semantic_content_hash: state_executor,
    }
    execution_port_by_route_hash.update(
        {
            fixed_orientation_entry_by_skill_id[
                skill_id
            ].route.semantic_content_hash: fixed_orientation_executors[skill_id]
            for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
        }
    )
    if history_entry is not None and history_executor is not None:
        execution_port_by_route_hash[history_entry.route.semantic_content_hash] = (
            history_executor
        )
    if depth_entry is not None and depth_executor is not None:
        execution_port_by_route_hash[depth_entry.route.semantic_content_hash] = (
            depth_executor
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
        max_request_timeout_ms=max(
            entry.route.max_operation_duration_ms
            for entry in contract_bundle.route_manifest.skill_routes
        ),
        max_grant_lifetime_sec=30.0,
    )
    admissions_by_route_hash = {
        entry.route.semantic_content_hash: retention
        for entry in contract_bundle.route_manifest.skill_routes
    }
    if artifact_store is None:
        protocol_runtime = AgentGatewayProtocolRuntime.control_plane_only(
            manifest=contract_bundle.route_manifest,
            clock=runtime_clock,
        )
    else:
        session_capabilities = AgentGatewaySessionCapabilities(
            runtime_session_id=binding.runtime_session_id,
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
        artifact_broker = AgentArtifactTransferBroker(
            store=artifact_store,
            session_capabilities=session_capabilities,
            max_chunk_bytes=artifact_store.max_chunk_bytes,
        )
        protocol_runtime = AgentGatewayProtocolRuntime(
            manifest=contract_bundle.route_manifest,
            io_catalog=contract_bundle.io_catalog,
            session_capabilities=session_capabilities,
            event_journal=BoundedAgentGatewayEventJournal(
                runtime_session_id=binding.runtime_session_id,
                max_events=session_capabilities.event_retention_count,
                event_payload_max_bytes=(session_capabilities.event_payload_max_bytes),
            ),
            artifact_store=artifact_store,
            artifact_transfer_broker=artifact_broker,
            clock=runtime_clock,
        )
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
        protocol_runtime=protocol_runtime,
        clock=runtime_clock,
        clock_domain_id=_required(
            runtime.config.clock_domain_id,
            field_name="runtime clock_domain_id",
        ),
    )
    history_recorder = (
        SourceObservationHistoryRecorder(
            store=source_observation_history_store,
            capture=lambda: history_provider.capture_record(
                active_operation_id=service.operation_store.active_operation_id,
            ),
            interval_sec=contract_bundle.history_profile.capture_interval_sec,
        )
        if history_provider is not None
        and source_observation_history_store is not None
        and contract_bundle.history_profile is not None
        else None
    )
    return MujocoAgentTcpPoseGatewaySession(
        application=application,
        contract_bundle=contract_bundle,
        observation_producer=observation_producer,
        visual_snapshot_provider=visual_snapshot_provider,
        motion_executor=motion_executor,
        cardinal_step_executor=cardinal_step_executor,
        fixed_orientation_executors=fixed_orientation_executors,
        state_executor=state_executor,
        depth_executor=depth_executor,
        history_provider=history_provider,
        history_executor=history_executor,
        history_recorder=history_recorder,
        source_observation_history_store=source_observation_history_store,
        owns_source_observation_history_store=owns_source_observation_history_store,
        artifact_store=artifact_store,
        owns_artifact_store=owns_artifact_store,
        service=service,
    )


__all__ = [
    "MujocoAgentTcpPoseGatewaySession",
    "assemble_mujoco_agent_tcp_pose_gateway_session",
]
