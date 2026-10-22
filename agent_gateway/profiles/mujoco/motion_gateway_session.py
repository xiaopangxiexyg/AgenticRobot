"""Assemble the reusable TCP-pose Agent Gateway session.

Construction is default-off: it creates contracts, executors and a service but
does not start MuJoCo, advance physics, bind a socket or admit motion.  Each
sequential motion operation gets a fresh lifecycle while all operations share
the same lower runtime, dispatch owner and domain-HOLD boundary.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace

from agent_gateway.profiles.back_projection import (
    LiveBackProjectionProvider,
    build_back_projection_executor,
)
from agent_gateway.profiles.motion_primitive_resolution import (
    MotionPrimitiveTargetResolution,
    resolve_level_tcp_orientation_primitive_target,
    resolve_tcp_displacement_primitive_target,
)
from agent_gateway.profiles.placement import (
    PLACEMENT_MAX_OPERATION_DURATION_MS_V1,
    PlacementProvider,
    build_placement_executor,
)
from agent_gateway.profiles.segmentation import (
    LiveSegmentationProvider,
    build_segmentation_executor,
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
from agent_gateway.runtime.hardware_view_provider import (
    HardwareResourceBinding,
    HardwareViewProvider,
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
from agent_gateway.skills.gripper.runtime import (
    GripperConvergenceProfile,
    build_gripper_skill_executor,
    hold_gripper_ports,
)
from agent_gateway.skills.motion.definition import (
    FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1,
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
)
from agent_gateway.skills.perception.definition import (
    BACK_PROJECT_SKILL_ID_V1,
    CAPTURE_DEPTH_SKILL_ID_V1,
    CAPTURE_RGB_SKILL_ID_V1,
    PLAN_PLACEMENT_SKILL_ID_V1,
    SEGMENT_SKILL_ID_V1,
)
from agent_gateway.skills.perception.schemas import (
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)
from agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from agent_gateway.transport.wire_adapter import (
    BoundAgentGatewayWireAdapter,
)
from collection_runtime.source_observation_history import (
    SourceObservationHistoryRecorder,
    SQLiteSourceObservationHistoryStore,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .back_projection import MujocoBackProjectionCapturePort
from .depth_deployment import MUJOCO_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1
from .gripper_skill import MujocoGripperControlPort
from .motion_application import MujocoAgentSkillMotionApplication
from .motion_deployment import MujocoAgentTcpPoseContractBundle
from .motion_skill import (
    MujocoAgentSafeTcpPoseSkillLifecycle,
    MujocoAgentTcpPoseSkillLifecycle,
    mujoco_agent_skill_hold_evidence_from_transition,
)
from .observation import MujocoAgentObservationProducer
from .observation_binding import (
    build_mujoco_agent_motion_observation_source_binding,
)
from .placement import MujocoPlacementRobotStatePort
from .rgb_deployment import MUJOCO_CAPTURE_RGB_OUTPUT_SLOT_ID_V1
from .segmentation import MujocoSegmentationCapturePort
from .tcp_pose_history import MujocoAgentTcpPoseHistoryProvider
from .visual_observation import MujocoAgentRgbCaptureProvider


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
    rgb_capture_provider: MujocoAgentRgbCaptureProvider
    motion_executor: ReusableStationAgentSkillOperationExecutor
    cardinal_step_executor: ReusableStationAgentSkillOperationExecutor
    fixed_cardinal_step_executors: Mapping[
        str, ReusableStationAgentSkillOperationExecutor
    ]
    fixed_orientation_executors: Mapping[
        str, ReusableStationAgentSkillOperationExecutor
    ]
    state_executor: PassiveReadSkillExecutor
    depth_executor: PassiveReadSkillExecutor | None
    rgb_executor: PassiveReadSkillExecutor | None
    back_projection_provider: LiveBackProjectionProvider | None
    back_projection_executor: PassiveReadSkillExecutor | None
    segmentation_localizer: object | None
    segmentation_provider: LiveSegmentationProvider | None
    segmentation_executor: PassiveReadSkillExecutor | None
    placement_provider: PlacementProvider | None
    placement_executor: PassiveReadSkillExecutor | None
    history_provider: MujocoAgentTcpPoseHistoryProvider | None
    history_executor: PassiveReadSkillExecutor | None
    history_recorder: SourceObservationHistoryRecorder | None
    source_observation_history_store: SQLiteSourceObservationHistoryStore | None
    owns_source_observation_history_store: bool
    artifact_store: RecoverableAgentArtifactStore | None
    owns_artifact_store: bool
    hardware_view_provider: HardwareViewProvider
    service: AgentGatewayService
    safe_tcp_pose_executor: ReusableStationAgentSkillOperationExecutor | None = None
    spatial_goal_executor: PassiveReadSkillExecutor | None = None
    tcp_pose_comparison_executor: PassiveReadSkillExecutor | None = None
    grasp_approach_executor: ReusableStationAgentSkillOperationExecutor | None = None
    grasp_contact_executor: ReusableStationAgentSkillOperationExecutor | None = None
    gripper_executor: ReusableStationAgentSkillOperationExecutor | None = None
    gripper_ports: Mapping[str, MujocoGripperControlPort] = field(default_factory=dict)
    _close_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _close_started: bool = field(default=False, init=False, repr=False)
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
    _segmentation_model_started: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (self.contract_bundle.spatial_goal_registration is not None) != (self.spatial_goal_executor is not None):
            raise ValueError("Spatial goal registration and executor must coexist.")
        if (self.contract_bundle.tcp_pose_comparison_registration is not None) != (self.tcp_pose_comparison_executor is not None):
            raise ValueError("TCP comparison registration and executor must coexist.")
        hand_parts = (self.contract_bundle.gripper_contract is not None,
                      self.gripper_executor is not None, bool(self.gripper_ports))
        if any(hand_parts) and not all(hand_parts):
            raise ValueError("Composed gripper contract and executor must coexist.")
        if set(self.fixed_cardinal_step_executors) != set(
            FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
        ):
            raise ValueError(
                "fixed_cardinal_step_executors must exactly cover six Skills."
            )
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
        if (self.rgb_executor is None) != (self.artifact_store is None):
            raise ValueError(
                "RGB executor and Artifact store must be enabled together."
            )
        back_projection_enabled = bool(
            self.contract_bundle.depth_capture is not None
            and self.contract_bundle.depth_capture.back_projection is not None
        )
        if back_projection_enabled != all(
            item is not None
            for item in (
                self.back_projection_provider,
                self.back_projection_executor,
            )
        ):
            raise ValueError(
                "MuJoCo back-projection registration and runtime resources disagree."
            )
        if (self.segmentation_provider is None) != (
            self.segmentation_executor is None
        ) or (self.segmentation_provider is None) != (
            self.segmentation_localizer is None
        ):
            raise ValueError("MuJoCo segmentation resources must be all-or-none.")
        if self.segmentation_provider is not None and self.artifact_store is None:
            raise ValueError("MuJoCo segmentation requires the Artifact store.")
        placement_enabled = self.contract_bundle.placement is not None
        if placement_enabled != all(
            item is not None
            for item in (self.placement_provider, self.placement_executor)
        ):
            raise ValueError(
                "MuJoCo placement registration and runtime resources disagree."
            )
        if not isinstance(self.hardware_view_provider, HardwareViewProvider):
            raise TypeError("MuJoCo hardware view provider must be typed.")

    def build_grasp_candidate_selector(self, *, world_from_candidate_reference, config=None):
        """Bind the existing model selector seam; never activate a motion plan."""
        from .grasp_candidate_selection import (
            MujocoGraspCandidateSelector,
            MujocoGraspSelectionConfig,
        )

        return MujocoGraspCandidateSelector(
            application=self.application, contract_bundle=self.contract_bundle,
            world_from_candidate_reference=world_from_candidate_reference,
            config=MujocoGraspSelectionConfig() if config is None else config,
        )

    def wire_adapter(
        self,
        *,
        authenticated_client_id: str,
        read_only_access: bool = False,
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
            hardware_view_provider=self.hardware_view_provider,
            read_only_access=read_only_access,
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

    def start_perception_resources(self) -> None:
        """Start optional model resources before transport publication."""

        with self._close_lock:
            if self._close_started or self.service.is_closed:
                raise RuntimeError("Closed Gateway session cannot start resources.")
            if self.segmentation_localizer is None or self._segmentation_model_started:
                return
            start = getattr(self.segmentation_localizer, "start", None)
            if not callable(start):
                raise TypeError("Segmentation localizer must expose start().")
            try:
                start()
            except Exception as start_error:
                close_model = getattr(self.segmentation_localizer, "close", None)
                if not callable(close_model):
                    raise TypeError(
                        "Segmentation model startup failed without a cleanup port."
                    ) from start_error
                try:
                    close_model()
                except Exception as cleanup_error:  # noqa: BLE001 - preserve both failures.
                    raise RuntimeError(
                        "Segmentation model startup cleanup failed: "
                        f"{type(cleanup_error).__name__}:{cleanup_error}"
                    ) from start_error
                raise
            self._segmentation_model_started = True

    def build_unix_json_server(
        self,
        *,
        socket_path: str,
        client_id_by_uid: Mapping[int, str],
        socket_mode: int = 0o600,
        max_frame_bytes: int = 2 * 1024 * 1024,
        peer_idle_timeout_sec: float = 10.0,
        read_only_client_ids: Collection[str] = (),
    ) -> AgentGatewayUnixJsonServer:
        """Bind the local endpoint without starting its serve loop."""

        peer_map = AgentGatewayUnixPeerClientMap(client_id_by_uid)
        if set(peer_map.client_id_by_uid.values()) != set(
            self.service.access_policy.allowed_client_ids
        ):
            raise ValueError(
                "Unix peer client map must exactly cover the session access policy."
            )
        read_only_clients = frozenset(read_only_client_ids)
        if not read_only_clients <= frozenset(peer_map.client_id_by_uid.values()):
            raise ValueError("Read-only clients must exist in the Unix peer map.")
        return AgentGatewayUnixJsonServer(
            socket_path=socket_path,
            peer_client_map=peer_map,
            facade_factory=lambda client_id: self.wire_adapter(
                authenticated_client_id=client_id,
                read_only_access=client_id in read_only_clients,
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
        if self.application.closed:
            # Preserve the first close receipt. Repeated teardown must not
            # send another HOLD into a runtime that has already stopped.
            return
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
            if executor_receipt.succeeded and executor_receipt.proves_motion_or_hold:
                executor.close_after_hold(hold_receipt=executor_receipt)
        workers = tuple(
            worker
            for executor in self._motion_executors()
            for worker in executor.workers
        )
        settled = all(
            executor.active_operation_id is None or executor.wait(timeout_sec=5.0)
            for executor in self._motion_executors()
        )
        if not workers or settled:
            self.application.close_after_hold()

    def _motion_executors(
        self,
    ) -> tuple[ReusableStationAgentSkillOperationExecutor, ...]:
        return (
            self.motion_executor,
            *((self.gripper_executor,) if self.gripper_executor is not None else ()),
            *(
                (self.safe_tcp_pose_executor,)
                if self.safe_tcp_pose_executor is not None
                else ()
            ),
            *(
                (self.grasp_approach_executor,)
                if self.grasp_approach_executor is not None
                else ()
            ),
            *((self.grasp_contact_executor,) if self.grasp_contact_executor is not None else ()),
            self.cardinal_step_executor,
            *tuple(
                self.fixed_cardinal_step_executors[skill_id]
                for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
            ),
            *tuple(
                self.fixed_orientation_executors[skill_id]
                for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
            ),
        )

    def close(self) -> AgentGatewayOperationOutcome:
        """Fence admissions, establish lower HOLD, then close when quiescent."""

        with self._close_lock:
            # A failed HOLD remains retryable, but resource startup never reopens.
            self._close_started = True
            outcome = self._close_outcome
            cleanup_errors: list[tuple[str, Exception]] = []
            recorder_settled = True
            if self.history_recorder is not None:
                try:
                    recorder_settled = self.history_recorder.close(timeout_sec=1.0)
                    if not recorder_settled:
                        raise RuntimeError(
                            "Planning history recorder did not quiesce before close."
                        )
                except Exception as error:  # noqa: BLE001 - continue owned cleanup.
                    recorder_settled = False
                    cleanup_errors.append(("history_recorder", error))
            if outcome is None:
                try:
                    outcome = self.service.close(
                        current_station_revision=(
                            self.observation_producer.current_station_revision
                        )
                    )
                    if outcome.accepted:
                        self._close_outcome = outcome
                except Exception as error:  # noqa: BLE001 - continue owned cleanup.
                    cleanup_errors.append(("gateway_service", error))
            gateway_closed = outcome is not None
            passive_settled = gateway_closed
            if gateway_closed:
                for name, executor, timeout_sec in (
                    ("state", self.state_executor, 5.0),
                    ("depth", self.depth_executor, 5.0),
                    ("rgb", self.rgb_executor, 5.0),
                    (
                        "back_projection",
                        getattr(self, "back_projection_executor", None),
                        5.0,
                    ),
                    ("history", self.history_executor, 5.0),
                    ("spatial_goal", self.spatial_goal_executor, 5.0),
                    ("tcp_pose_comparison", self.tcp_pose_comparison_executor, 5.0),
                    ("segmentation", self.segmentation_executor, 65.0),
                    (
                        "placement",
                        getattr(self, "placement_executor", None),
                        PLACEMENT_MAX_OPERATION_DURATION_MS_V1 / 1_000.0 + 5.0,
                    ),
                ):
                    try:
                        if executor is not None and not executor.wait_all(
                            timeout_sec=timeout_sec
                        ):
                            passive_settled = False
                            raise RuntimeError(
                                f"Passive {name} reads did not quiesce before close."
                            )
                    except Exception as error:  # noqa: BLE001 - continue safe waits.
                        passive_settled = False
                        cleanup_errors.append((f"{name}_executor", error))
            if passive_settled and self._segmentation_model_started:
                close_model = getattr(self.segmentation_localizer, "close", None)
                if not callable(close_model):
                    cleanup_errors.append(
                        (
                            "segmentation_model",
                            TypeError("Segmentation localizer must expose close()."),
                        )
                    )
                else:
                    try:
                        close_model()
                    except Exception as error:  # noqa: BLE001 - continue cleanup.
                        cleanup_errors.append(("segmentation_model", error))
                    else:
                        self._segmentation_model_started = False
            if passive_settled and recorder_settled:
                try:
                    assert outcome is not None
                    self._finish_lower_close(outcome)
                except Exception as error:  # noqa: BLE001 - continue owned cleanup.
                    cleanup_errors.append(("lower_application", error))
            if (
                gateway_closed
                and passive_settled
                and recorder_settled
                and self.owns_source_observation_history_store
                and self.source_observation_history_store is not None
                and not self._source_observation_history_store_closed
            ):
                try:
                    self.source_observation_history_store.close()
                except Exception as error:  # noqa: BLE001 - continue owned cleanup.
                    cleanup_errors.append(("history_store", error))
                else:
                    self._source_observation_history_store_closed = True
            if (
                gateway_closed
                and passive_settled
                and self.owns_artifact_store
                and self.artifact_store is not None
                and not self._artifact_store_closed
            ):
                try:
                    self.artifact_store.close()
                except Exception as error:  # noqa: BLE001 - preserve every failure.
                    cleanup_errors.append(("artifact_store", error))
                else:
                    self._artifact_store_closed = True
            if cleanup_errors:
                details = "; ".join(
                    f"{stage}:{type(error).__name__}:{error}"
                    for stage, error in cleanup_errors
                )
                raise RuntimeError(
                    "MuJoCo Gateway close cleanup failed: " + details
                ) from cleanup_errors[0][1]
            assert outcome is not None
            return outcome


def _mujoco_runtime_health(
    application: MujocoAgentSkillMotionApplication,
) -> str:
    """Project cached lower lifecycle/gate state without stepping or rendering."""

    if application.closed or not application.started:
        return "unavailable"
    gate = application.runtime.capture_execution_gate_snapshot(
        now=float(application.runtime.clock())
    )
    status = str(getattr(gate, "status", ""))
    if status == "fault":
        return "fault"
    if status in {"hold", "ready"}:
        return "ready"
    return "unavailable"


def _mujoco_runtime_control_claimed(
    application: MujocoAgentSkillMotionApplication,
) -> bool:
    """The application claim spans READY/HOLD and ends only after close."""

    return application.started and not application.closed


def _mujoco_hardware_view_provider(
    *,
    application: MujocoAgentSkillMotionApplication,
    contract_bundle: MujocoAgentTcpPoseContractBundle,
    operation_store: BoundedInMemoryAgentSkillOperationStore,
    segmentation_provider: LiveSegmentationProvider | None,
) -> HardwareViewProvider:
    """Map the single MuJoCo scene owner to its semantic Skill surface."""

    route_hashes = tuple(
        entry.route.semantic_content_hash
        for entry in contract_bundle.route_manifest.skill_routes
        if entry.route.control_effect_class in {"read_only", "motion"}
    )
    segment_route_hash = (
        None
        if contract_bundle.segmentation is None
        else contract_bundle.segmentation.skill.routes[0].semantic_content_hash
    )
    bindings = [
        HardwareResourceBinding(
            resource_id="simulation.runtime",
            resource_kind="simulation_runtime",
            route_hashes=route_hashes,
            health_provider=lambda: _mujoco_runtime_health(application),
            control_claimed_provider=lambda: _mujoco_runtime_control_claimed(
                application
            ),
        )
    ]
    if segment_route_hash is not None:
        bindings.append(
            HardwareResourceBinding(
                resource_id="perception.segmentation_model",
                resource_kind="local_model",
                route_hashes=(segment_route_hash,),
                health_provider=lambda: (
                    "ready"
                    if segmentation_provider is not None
                    and segmentation_provider.started
                    else "unavailable"
                ),
                control_claimed_provider=lambda: False,
            )
        )
    return HardwareViewProvider(
        manifest=contract_bundle.route_manifest,
        operation_store=operation_store,
        bindings=tuple(bindings),
    )


def assemble_mujoco_agent_tcp_pose_gateway_session(
    *,
    application: MujocoAgentSkillMotionApplication,
    contract_bundle: MujocoAgentTcpPoseContractBundle,
    allowed_client_ids: tuple[str, ...],
    max_retained_operations: int = 16,
    max_session_admissions_per_route: int | None = None,
    clock: Callable[[], float] | None = None,
    source_observation_history_store: SQLiteSourceObservationHistoryStore | None = None,
    owns_source_observation_history_store: bool = False,
    artifact_store: RecoverableAgentArtifactStore | None = None,
    owns_artifact_store: bool = False,
    segmentation_localizer: object | None = None,
    gripper_ports: Mapping[str, MujocoGripperControlPort] | None = None,
) -> MujocoAgentTcpPoseGatewaySession:
    """Wire exact Routes without starting the application or history recorder."""

    if not isinstance(application, MujocoAgentSkillMotionApplication):
        raise TypeError("application must be a motion application.")
    if not isinstance(contract_bundle, MujocoAgentTcpPoseContractBundle):
        raise TypeError("contract_bundle must be typed.")
    gripper_ports = dict(gripper_ports or {})
    if (
        contract_bundle.grasp_contact_contract is not None
        and contract_bundle.grasp_approach_contract is None
    ):
        raise ValueError("Terminal contact requires the separate grasp approach Route.")
    if contract_bundle.grasp_approach_contract is not None:
        if not gripper_ports or contract_bundle.gripper_contract is None:
            raise ValueError("Grasp approach requires composed grippers and their contract.")
        if application.runtime.rig.config.realtime_worker is not False:
            raise ValueError("Grasp approach requires deterministic physics ownership.")
    if bool(gripper_ports) != (contract_bundle.gripper_contract is not None):
        raise ValueError("Composed gripper contract and ports must coexist.")
    for hand in gripper_ports.values():
        if not isinstance(hand, MujocoGripperControlPort):
            raise TypeError("Composed grippers require the reviewed MuJoCo adapter.")
        if hand.rig is not application.runtime.rig:
            raise ValueError("Composed grippers must share the exact application physics owner.")
        if not hand.requires_arm_hold:
            raise ValueError("Composed grippers require an arm-HOLD-bound port.")
    for hand in gripper_ports.values():
        hand.bind_motion_guard(lambda: application.held)
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
    perception_enabled = contract_bundle.depth_capture is not None
    if perception_enabled != (contract_bundle.rgb_capture is not None):
        raise ValueError("RGB and depth Artifact Skills must be enabled together.")
    if perception_enabled and not isinstance(
        artifact_store,
        RecoverableAgentArtifactStore,
    ):
        raise ValueError("Enabled perception capture requires an exact Artifact store.")
    if not perception_enabled and artifact_store is not None:
        raise ValueError("Artifact store cannot be bound without an Artifact Skill.")
    if owns_artifact_store and artifact_store is None:
        raise ValueError("Owned Artifact store must be injected.")
    segmentation_enabled = contract_bundle.segmentation is not None
    if segmentation_enabled != (segmentation_localizer is not None):
        raise ValueError("Segmentation registration and localizer must coexist.")
    if segmentation_enabled and artifact_store is None:
        raise ValueError("Enabled segmentation requires an Artifact store.")
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
    session_admission_limit = (
        None
        if max_session_admissions_per_route is None
        else _positive_integer(
            max_session_admissions_per_route,
            field_name="max_session_admissions_per_route",
        )
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
    fixed_cardinal_entry_by_skill_id = {
        skill_id: contract_bundle.route_manifest.route(skill_id, "1")
        for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
    }
    fixed_cardinal_contract_by_skill_id = {
        item.skill_id: item for item in contract_bundle.fixed_cardinal_step_contracts
    }
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
    depth_registration = contract_bundle.depth_capture
    depth_entry = (
        contract_bundle.route_manifest.route(
            CAPTURE_DEPTH_SKILL_ID_V1,
            depth_registration.registration.definition.skill_version,
        )
        if depth_registration is not None
        else None
    )
    back_projection_registration = (
        None
        if depth_registration is None
        else depth_registration.back_projection
    )
    back_projection_entry = (
        contract_bundle.route_manifest.route(BACK_PROJECT_SKILL_ID_V1, "1")
        if back_projection_registration is not None
        else None
    )
    rgb_entry = (
        contract_bundle.route_manifest.route(
            CAPTURE_RGB_SKILL_ID_V1,
            contract_bundle.rgb_capture.registration.definition.skill_version,
        )
        if contract_bundle.rgb_capture is not None
        else None
    )
    segmentation_entry = (
        contract_bundle.route_manifest.route(SEGMENT_SKILL_ID_V1, "1")
        if contract_bundle.segmentation is not None
        else None
    )
    placement_entry = (
        contract_bundle.route_manifest.route(PLAN_PLACEMENT_SKILL_ID_V1, "1")
        if contract_bundle.placement is not None
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
    rgb_capture_provider = MujocoAgentRgbCaptureProvider(
        application=application,
        current_station_revision=lambda: observation_producer.current_station_revision,
    )
    segmentation_provider = None
    segmentation_executor = None
    if contract_bundle.segmentation is not None:
        assert segmentation_localizer is not None
        assert artifact_store is not None
        segmentation_provider = LiveSegmentationProvider(
            localizer=segmentation_localizer,
            capture_port=MujocoSegmentationCapturePort(
                rgb_provider=rgb_capture_provider,
            ),
            registration=contract_bundle.segmentation,
        )
        segmentation_executor = build_segmentation_executor(
            registration=contract_bundle.segmentation,
            provider=segmentation_provider,
            artifact_store=artifact_store,
            runtime_session_id=binding.runtime_session_id,
            current_station_revision=lambda: (
                observation_producer.current_station_revision
            ),
            max_retained_operations=retention,
            clock=runtime_clock,
        )
    back_projection_capture_port = None
    back_projection_provider = None
    back_projection_executor = None
    if back_projection_registration is not None:
        depth_binding = depth_registration.deployment_binding
        back_projection_capture_port = MujocoBackProjectionCapturePort(
            application=application,
            source_id_by_role=dict(
                back_projection_registration.profile.source_id_by_role
            ),
            source_frame_id=depth_binding.source_frame_id,
            reference_frame_id=depth_binding.reference_frame_id,
            current_station_revision=lambda: (
                observation_producer.current_station_revision
            ),
        )
        back_projection_provider = LiveBackProjectionProvider(
            capture_port=back_projection_capture_port,
            registration=back_projection_registration,
        )
        back_projection_executor = build_back_projection_executor(
            registration=back_projection_registration,
            provider=back_projection_provider,
            current_station_revision=lambda: (
                observation_producer.current_station_revision
            ),
            max_retained_operations=retention,
            clock=runtime_clock,
        )
    placement_provider = None
    placement_executor = None
    if contract_bundle.placement is not None:
        assert contract_bundle.segmentation is not None
        assert segmentation_executor is not None
        assert segmentation_provider is not None
        assert back_projection_registration is not None
        assert back_projection_capture_port is not None
        placement_provider = PlacementProvider(
            registration=contract_bundle.placement,
            segmentation_registration=contract_bundle.segmentation,
            segmentation_executor=segmentation_executor,
            segmentation_provider=segmentation_provider,
            back_projection_registration=back_projection_registration,
            capture_port=back_projection_capture_port,
            robot_state_port=MujocoPlacementRobotStatePort(
                application=application,
                component_id=contract_bundle.placement.profile.component_id,
                source_frame_id=(
                    contract_bundle.motion_profile.execution_reference_frame_id
                    or contract_bundle.motion_profile.reference_frame_id
                ),
                reference_frame_id=(
                    contract_bundle.placement.profile.reference_frame_id
                ),
                current_station_revision=lambda: (
                    observation_producer.current_station_revision
                ),
            ),
        )
        placement_executor = build_placement_executor(
            registration=contract_bundle.placement,
            provider=placement_provider,
            current_station_revision=lambda: (
                observation_producer.current_station_revision
            ),
            max_retained_operations=retention,
            clock=runtime_clock,
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
            domain_hold=domain_hold if gripper_ports else None,
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

    def build_cardinal_step_lifecycle(
        *,
        request,
        operation,
        route,
        deployment_binding,
        evaluation_plan,
        fixed_direction: str | None = None,
    ):
        parameters = request.parameters
        if not isinstance(parameters, Mapping):
            raise TypeError("Cardinal-step parameters must be an object.")
        requested_direction = str(parameters.get("direction"))
        if fixed_direction is not None and requested_direction != fixed_direction:
            raise ValueError("Fixed cardinal-step direction changed in request.")
        primitive = contract_bundle.cardinal_step_profile.primitive(requested_direction)

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
            domain_hold=domain_hold if gripper_ports else None,
            route=route,
            deployment_binding=deployment_binding,
            evaluation_plan=evaluation_plan,
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

    def cardinal_step_lifecycle_factory(*, request, operation):
        return build_cardinal_step_lifecycle(
            request=request,
            operation=operation,
            route=cardinal_step_route,
            deployment_binding=(contract_bundle.cardinal_step_deployment_binding),
            evaluation_plan=(contract_bundle.cardinal_step_evaluation_plan),
        )

    def domain_hold(*, operation, reason):
        failure = None
        try:
            transition = application.request_hold(
                reason=reason,
                operation_id=(None if operation is None else operation.operation_id),
            )
        except (Exception, KeyboardInterrupt, SystemExit) as error:
            failure = error
        hand_hold = None
        if gripper_ports:
            try:
                hand_hold = hold_gripper_ports(ports=gripper_ports, reason=reason)
            except (Exception, KeyboardInterrupt, SystemExit) as error:
                if failure is None or (
                    isinstance(error, (KeyboardInterrupt, SystemExit))
                    and not isinstance(failure, (KeyboardInterrupt, SystemExit))
                ):
                    raise
                raise failure from error
        if failure is not None:
            raise failure
        evidence = mujoco_agent_skill_hold_evidence_from_transition(
            transition,
            lower_operation_id=(
                None if operation is None else operation.lower_operation_id
            ),
        )
        if hand_hold is not None:
            return replace(evidence, succeeded=evidence.succeeded and hand_hold.succeeded,
                lower_hold_confirmed=evidence.lower_hold_confirmed and hand_hold.succeeded,
                evidence_refs=(*evidence.evidence_refs, *hand_hold.evidence_refs))
        return evidence

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
        current_station_revision=lambda: observation_producer.current_station_revision,
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
            deployment_binding_hash=(cardinal_step_binding.semantic_content_hash),
        ),
        lifecycle_factory=cardinal_step_lifecycle_factory,
        domain_hold=domain_hold,
        current_station_revision=lambda: observation_producer.current_station_revision,
        clock=runtime_clock,
    )
    fixed_cardinal_step_executors: dict[
        str, ReusableStationAgentSkillOperationExecutor
    ] = {}
    for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1:
        fixed_entry = fixed_cardinal_entry_by_skill_id[skill_id]
        fixed_route = fixed_entry.route
        fixed_contract = fixed_cardinal_contract_by_skill_id[skill_id]

        def fixed_cardinal_lifecycle_factory(
            *,
            request,
            operation,
            _route=fixed_route,
            _contract=fixed_contract,
        ):
            return build_cardinal_step_lifecycle(
                request=request,
                operation=operation,
                route=_route,
                deployment_binding=_contract.deployment_binding,
                evaluation_plan=_contract.evaluation_plan,
                fixed_direction=_contract.direction,
            )

        fixed_binding = fixed_contract.deployment_binding
        fixed_cardinal_step_executors[skill_id] = (
            ReusableStationAgentSkillOperationExecutor(
                port_id=(
                    "agent-gateway:fixed-cardinal-step-worker:"
                    + skill_id
                    + ":"
                    + fixed_binding.runtime_session_id
                ),
                binding=StationOperationWorkerBinding(
                    worker_id=(
                        "agent-tcp-fixed-cardinal-step-worker:"
                        + skill_id
                        + ":"
                        + fixed_binding.runtime_session_id
                    ),
                    executor_id=fixed_binding.executor_id,
                    runtime_session_id=fixed_binding.runtime_session_id,
                    skill_id=fixed_route.skill_id,
                    skill_version=fixed_route.skill_version,
                    route_id=fixed_route.route_id,
                    route_version=fixed_route.route_version,
                    deployment_binding_hash=(fixed_binding.semantic_content_hash),
                ),
                lifecycle_factory=fixed_cardinal_lifecycle_factory,
                domain_hold=domain_hold,
                current_station_revision=lambda: (
                    observation_producer.current_station_revision
                ),
                clock=runtime_clock,
            )
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
                    measured_reference_frame_id=(_profile.execution_reference_frame_id),
                )

            return MujocoAgentTcpPoseSkillLifecycle(
                application=application,
                domain_hold=domain_hold if gripper_ports else None,
                route=_route,
                deployment_binding=_contract.deployment_binding,
                evaluation_plan=_contract.evaluation_plan,
                motion_profile=_contract.motion_profile,
                request=request,
                operation=operation,
                target_resolver=resolve_target,
                max_reference_rotation_step_rad=(
                    _profile.max_reference_rotation_step_rad
                ),
                max_measured_translation_excursion_m=(
                    _profile.max_measured_translation_excursion_m
                ),
                parameter_lineage={
                    "reference_frame_id": _profile.reference_frame_id,
                    "reference_frame_spec_hash": (_profile.reference_frame_spec_hash),
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
                    deployment_binding_hash=(orientation_binding.semantic_content_hash),
                ),
                lifecycle_factory=orientation_lifecycle_factory,
                domain_hold=domain_hold,
                current_station_revision=lambda: (
                    observation_producer.current_station_revision
                ),
                clock=runtime_clock,
            )
        )

    def capture_tcp_pose(parameters, active_operation_id) -> PassiveReadResult:
        state_read = observation_producer.capture_tcp_pose_state(
            active_operation_id=active_operation_id,
            parameters=parameters,
        )
        return PassiveReadResult(
            execution_result=state_read.execution_result,
            evidence_refs=state_read.evidence_refs,
            reason_code="passive_tcp_pose_read_completed",
        )

    def read_tcp_pose(request, operation) -> PassiveReadResult:
        return capture_tcp_pose(request.parameters, operation.operation_id)

    state_executor = PassiveReadSkillExecutor(
        port_id=("agent-gateway:passive-tcp-pose:" + binding.runtime_session_id),
        executor_id=contract_bundle.state_deployment_binding.executor_id,
        read=read_tcp_pose,
        current_station_revision=lambda: observation_producer.current_station_revision,
        clock=runtime_clock,
        max_retained_operations=retention,
    )
    depth_executor = None
    if depth_entry is not None and artifact_store is not None:
        depth_registration = contract_bundle.depth_capture
        if depth_registration is None:
            raise RuntimeError("Depth Route lost its registration.")

        def read_depth(request, operation) -> PassiveReadResult:
            role = str(request.parameters["role"])
            try:
                source_id = depth_registration.profile.source_id_by_role[role]
            except KeyError as exc:
                raise ValueError("Depth role is not bound to a source.") from exc
            station_revision = observation_producer.current_station_revision
            frame = application.capture_depth_npy_frame(
                view_id=source_id,
            )
            if observation_producer.current_station_revision != station_revision:
                raise RuntimeError(
                    "Station revision changed during passive depth capture."
                )
            depth_binding = depth_registration.deployment_binding
            camera_pose_reference = application.transform_tcp_pose_xyzquat(
                frame.camera_pose_world_xyz_quat_xyzw,
                source_frame_id=depth_binding.source_frame_id,
                target_frame_id=depth_binding.reference_frame_id,
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
                "source_id": frame.view_id,
                "role": role,
                "captured_at": frame.captured_at,
                "station_revision": station_revision,
                "width": frame.width,
                "height": frame.height,
                "depth_unit": "meter",
                "invalid_depth_value": 0.0,
                "reference_frame_id": depth_binding.reference_frame_id,
                "reference_frame_spec_hash": (depth_binding.reference_frame_spec_hash),
                "frame_graph_hash": depth_binding.frame_graph_hash,
                "camera_pose_reference_xyz_quat_xyzw": list(camera_pose_reference),
                "intrinsics": dict(frame.intrinsics),
            }
            capture_ref = "mujoco-depth-capture:" + canonical_content_hash(
                {
                    "operation_id": operation.operation_id,
                    "request_hash": request.semantic_content_hash,
                    "result": result,
                    "artifact_ref_hash": committed.semantic_content_hash,
                },
                domain="dexteleop.agent_skill.mujoco_depth_capture_receipt.v2",
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
    rgb_executor = None
    if rgb_entry is not None and artifact_store is not None:
        rgb_registration = contract_bundle.rgb_capture
        if rgb_registration is None:
            raise RuntimeError("RGB Route lost its registration.")

        def read_rgb(request, operation) -> PassiveReadResult:
            is_v2 = rgb_registration.registration.definition.skill_version == "2"
            capture = rgb_capture_provider.capture(
                request.parameters,
                expected_station_revision=request.expected_station_revision,
            )
            bindings = []
            frame_metadata = []
            total_bytes = 0
            for ordinal, (source_id, role, frame) in enumerate(capture.frames):
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
                    producer_id="agent_gateway.provider.mujoco_rgb",
                    runtime_session_id=binding.runtime_session_id,
                    idempotency_key=(f"rgb-output:{operation.operation_id}:{ordinal}"),
                    now=capture.captured_at,
                )
                total_bytes += committed.byte_length
                frame_metadata.append(
                    {
                        "ordinal": ordinal,
                        "source_id": source_id,
                        "role": role,
                        **({"acquired_at": capture.captured_at} if is_v2 else {}),
                        "media_type": committed.media_type,
                        "width": frame.width,
                        "height": frame.height,
                        "media_content_hash": committed.sha256,
                    }
                )
                bindings.append(
                    AgentArtifactBinding(
                        slot_id=MUJOCO_CAPTURE_RGB_OUTPUT_SLOT_ID_V1,
                        ordinal=ordinal,
                        artifact=committed,
                        production_receipt_ref=(
                            f"mujoco-rgb-frame:{operation.operation_id}:{ordinal}:"
                            + committed.sha256
                        ),
                    )
                )
            result_identity = canonical_content_hash(
                {
                    "operation_id": operation.operation_id,
                    "request_hash": request.semantic_content_hash,
                    "station_revision": capture.station_revision,
                    **(
                        {
                            "capture_window_started_at": (
                                capture.capture_window_started_at
                            ),
                            "capture_window_ended_at": (
                                capture.capture_window_ended_at
                            ),
                        }
                        if is_v2
                        else {"captured_at": capture.captured_at}
                    ),
                    "frames": frame_metadata,
                },
                domain=(
                    "dexteleop.agent_skill.mujoco_rgb_capture_receipt.v2"
                    if is_v2
                    else "dexteleop.agent_skill.mujoco_rgb_capture_receipt.v1"
                ),
            )
            result = {
                "snapshot_id": "rgb-snapshot:" + result_identity,
                "station_revision": capture.station_revision,
                **(
                    {
                        "capture_window_started_at": (
                            capture.capture_window_started_at
                        ),
                        "capture_window_ended_at": capture.capture_window_ended_at,
                    }
                    if is_v2
                    else {"captured_at": capture.captured_at}
                ),
                "frames": frame_metadata,
                "total_bytes": total_bytes,
            }
            return PassiveReadResult(
                execution_result=result,
                evidence_refs=("mujoco-rgb-capture:" + result_identity,),
                output_artifact_bindings=tuple(bindings),
                reason_code="passive_rgb_capture_completed",
            )

        rgb_executor = PassiveReadSkillExecutor(
            port_id=("agent-gateway:passive-rgb-capture:" + binding.runtime_session_id),
            executor_id=rgb_registration.deployment_binding.executor_id,
            read=read_rgb,
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
    tcp_pose_comparison_executor = None
    if contract_bundle.tcp_pose_comparison_registration is not None:
        from agent_gateway.profiles.tcp_pose_comparison import (
            build_tcp_pose_comparison_executor,
        )

        def lookup_comparison_target(operation_id):
            # The service owns this bounded store; never scan or copy history.
            source_request = service.operation_store.request_for_operation(operation_id)
            source_operation = service.operation_store.get(operation_id)
            if source_request is None or source_operation is None:
                return None
            return source_request, source_operation

        registration = contract_bundle.tcp_pose_comparison_registration
        tcp_pose_comparison_executor = build_tcp_pose_comparison_executor(
            registration=registration, capture_state=capture_tcp_pose,
            lookup_target=lookup_comparison_target,
            route_manifest=contract_bundle.route_manifest,
            payload_validators=contract_bundle.payload_validators,
            clock_domain_id=runtime.config.clock_domain_id,
            current_station_revision=lambda: observation_producer.current_station_revision,
            clock=runtime_clock, max_retained_operations=retention,
        )
        execution_port_by_route_hash[registration.entry.route.semantic_content_hash] = tcp_pose_comparison_executor
    spatial_goal_executor = None
    if contract_bundle.spatial_goal_registration is not None:
        from agent_gateway.profiles.spatial_goal import build_spatial_goal_executor

        registration = contract_bundle.spatial_goal_registration
        spatial_goal_executor = build_spatial_goal_executor(
            registration=registration, current_station_revision=lambda: observation_producer.current_station_revision,
            clock=runtime_clock, max_retained_operations=retention,
        )
        execution_port_by_route_hash[registration.entry.route.semantic_content_hash] = spatial_goal_executor
    planned_executors = {}
    for skill_id, label, safe_contract in (
        (
            "motion.move_tcp_to_pose_safe",
            "safe-tcp-pose",
            contract_bundle.safe_tcp_pose_contract,
        ),
        (
            "motion.approach_grasp_pose",
            "grasp-approach",
            contract_bundle.grasp_approach_contract,
        ),
        (
            "motion.approach_grasp_contact",
            "grasp-contact",
            contract_bundle.grasp_contact_contract,
        ),
    ):
        if safe_contract is None:
            continue
        safe_route = contract_bundle.route_manifest.route(skill_id, "1").route
        safe_binding = safe_contract.deployment_binding

        def safe_lifecycle_factory(
            *, request, operation, _route=safe_route, _contract=safe_contract
        ):
            return MujocoAgentSafeTcpPoseSkillLifecycle(
                path_search_time_budget_sec=_contract.path_search_time_budget_sec,
                contact_binding=_contract.contact_binding,
                application=application,
                domain_hold=domain_hold if gripper_ports else None,
                route=_route,
                deployment_binding=_contract.deployment_binding,
                evaluation_plan=_contract.evaluation_plan,
                motion_profile=_contract.motion_profile,
                request=request,
                operation=operation,
                clock=runtime_clock,
            )

        planned_executor = ReusableStationAgentSkillOperationExecutor(
            port_id="agent-gateway:"
            + label
            + "-worker:"
            + safe_binding.runtime_session_id,
            binding=StationOperationWorkerBinding(
                worker_id="agent-"
                + label
                + "-worker:"
                + safe_binding.runtime_session_id,
                executor_id=safe_binding.executor_id,
                runtime_session_id=safe_binding.runtime_session_id,
                skill_id=safe_route.skill_id,
                skill_version=safe_route.skill_version,
                route_id=safe_route.route_id,
                route_version=safe_route.route_version,
                deployment_binding_hash=safe_binding.semantic_content_hash,
            ),
            lifecycle_factory=safe_lifecycle_factory,
            domain_hold=domain_hold,
            current_station_revision=lambda: (
                observation_producer.current_station_revision
            ),
            clock=runtime_clock,
        )
        execution_port_by_route_hash[safe_route.semantic_content_hash] = (
            planned_executor
        )
        planned_executors[skill_id] = planned_executor
    execution_port_by_route_hash.update(
        {
            fixed_cardinal_entry_by_skill_id[
                skill_id
            ].route.semantic_content_hash: fixed_cardinal_step_executors[skill_id]
            for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
        }
    )
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
    if rgb_entry is not None and rgb_executor is not None:
        execution_port_by_route_hash[rgb_entry.route.semantic_content_hash] = (
            rgb_executor
        )
    if segmentation_entry is not None and segmentation_executor is not None:
        execution_port_by_route_hash[segmentation_entry.route.semantic_content_hash] = (
            segmentation_executor
        )
    if back_projection_entry is not None and back_projection_executor is not None:
        execution_port_by_route_hash[
            back_projection_entry.route.semantic_content_hash
        ] = back_projection_executor
    if placement_entry is not None and placement_executor is not None:
        execution_port_by_route_hash[
            placement_entry.route.semantic_content_hash
        ] = placement_executor
    gripper_executor = None
    if contract_bundle.gripper_contract is not None:
        gripper_executor = build_gripper_skill_executor(
            contract_bundle=contract_bundle.gripper_contract, ports=gripper_ports,
            current_station_revision=lambda: observation_producer.current_station_revision,
            convergence=GripperConvergenceProfile(poll_interval_sec=0.),
            domain_hold=domain_hold, clock=runtime_clock)
        hand_route = contract_bundle.gripper_contract.route_manifest.skill_routes[0].route
        execution_port_by_route_hash[hand_route.semantic_content_hash] = gripper_executor
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
    admissions_by_route_hash = (
        {}
        if session_admission_limit is None
        else {
            entry.route.semantic_content_hash: session_admission_limit
            for entry in contract_bundle.route_manifest.skill_routes
        }
    )
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
    hardware_view_provider = _mujoco_hardware_view_provider(
        application=application,
        contract_bundle=contract_bundle,
        operation_store=service.operation_store,
        segmentation_provider=segmentation_provider,
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
        rgb_capture_provider=rgb_capture_provider,
        motion_executor=motion_executor,
        cardinal_step_executor=cardinal_step_executor,
        fixed_cardinal_step_executors=fixed_cardinal_step_executors,
        fixed_orientation_executors=fixed_orientation_executors,
        state_executor=state_executor,
        tcp_pose_comparison_executor=tcp_pose_comparison_executor,
        spatial_goal_executor=spatial_goal_executor,
        depth_executor=depth_executor,
        rgb_executor=rgb_executor,
        back_projection_provider=back_projection_provider,
        back_projection_executor=back_projection_executor,
        segmentation_localizer=segmentation_localizer,
        segmentation_provider=segmentation_provider,
        segmentation_executor=segmentation_executor,
        placement_provider=placement_provider,
        placement_executor=placement_executor,
        history_provider=history_provider,
        history_executor=history_executor,
        history_recorder=history_recorder,
        source_observation_history_store=source_observation_history_store,
        owns_source_observation_history_store=owns_source_observation_history_store,
        artifact_store=artifact_store,
        owns_artifact_store=owns_artifact_store,
        hardware_view_provider=hardware_view_provider,
        service=service,
        safe_tcp_pose_executor=planned_executors.get("motion.move_tcp_to_pose_safe"),
        grasp_approach_executor=planned_executors.get("motion.approach_grasp_pose"),
        grasp_contact_executor=planned_executors.get("motion.approach_grasp_contact"),
        gripper_executor=gripper_executor,
        gripper_ports=gripper_ports,
    )


__all__ = [
    "MujocoAgentTcpPoseGatewaySession",
    "assemble_mujoco_agent_tcp_pose_gateway_session",
]
