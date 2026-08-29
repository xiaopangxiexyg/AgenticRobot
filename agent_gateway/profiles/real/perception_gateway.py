"""Camera-only real perception Gateway with no robot-control dependency."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.contracts import AgentObservationSnapshot
from agent_gateway.protocol.session_contracts import AgentGatewaySessionCapabilities
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    validate_skill_route_catalog_closure,
)
from agent_gateway.protocol.skill_io_contracts import SkillIOCatalog
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
from agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
)
from agent_gateway.runtime.protocol_runtime import AgentGatewayProtocolRuntime
from agent_gateway.runtime.service import AgentGatewayService
from agent_gateway.runtime.station_revision import AgentStationRevisionTracker
from agent_gateway.skills.perception.schemas import (
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)
from agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter
from dataset_contracts.canonical_encoding import canonical_content_hash

from .observation_gateway import ObservationOnlyDomainControlPort
from .localization_deployment import RealObjectLocalizationRegistration
from .localization_gateway import (
    ObjectLocalizationProvider,
    build_real_object_localization_executor,
)
from .point_grasp_deployment import RealPointGraspRegistration
from .point_grasp_gateway import PointGraspProvider, build_real_point_grasp_executor
from .perception_deployment import (
    REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
    REAL_CAPTURE_RGB_OUTPUT_SLOT_ID_V1,
    RealPerceptionCaptureRegistration,
)
from .realsense_perception import (
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
)

REAL_PERCEPTION_CLOCK_DOMAIN_ID_V1 = "agent-gateway-real-perception-wall-clock"


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class RealPerceptionContractBundle:
    """Exact passive perception catalogs and real-observe Route manifest."""

    registration: RealPerceptionCaptureRegistration
    localization_registration: RealObjectLocalizationRegistration | None
    point_grasp_registration: RealPointGraspRegistration | None
    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    io_catalog: SkillIOCatalog
    route_manifest: SkillRouteManifest
    payload_validators: AgentSkillPayloadValidatorRegistry
    source_binding_id: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.registration, RealPerceptionCaptureRegistration):
            raise TypeError("registration must be typed.")
        validate_skill_route_catalog_closure(
            definition_catalog=self.definition_catalog,
            schema_catalog=self.schema_catalog,
            route_manifest=self.route_manifest,
            io_catalog=self.io_catalog,
        )
        self.payload_validators.validate_manifest(self.route_manifest)
        if self.route_manifest.current_mode != "real_observe":
            raise ValueError("Camera-only perception requires real_observe mode.")
        expected_routes = (
            2
            + int(self.localization_registration is not None)
            + int(self.point_grasp_registration is not None)
        )
        if len(self.route_manifest.skill_routes) != expected_routes:
            raise ValueError("Real perception Route count does not match its resources.")
        for field_name in (
            "source_binding_id",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
        ):
            _required(getattr(self, field_name), field_name=field_name)
        frame_binding = self.registration.frame_binding
        if (
            frame_binding is not None
            and frame_binding.frame_graph.semantic_content_hash
            != self.frame_graph_hash
        ):
            raise ValueError(
                "Real Depth v2 frame graph differs from the Gateway bundle."
            )


def build_real_perception_contract_bundle(
    *,
    registration: RealPerceptionCaptureRegistration,
    source_binding_id: str,
    frame_graph_hash: str,
    tool_config_hash: str,
    calibration_hash: str,
    localization_registration: RealObjectLocalizationRegistration | None = None,
    point_grasp_registration: RealPointGraspRegistration | None = None,
) -> RealPerceptionContractBundle:
    """Build the passive-only manifest without starting a camera or socket."""

    if not isinstance(registration, RealPerceptionCaptureRegistration):
        raise TypeError("registration must be typed.")
    if localization_registration is not None and not isinstance(
        localization_registration,
        RealObjectLocalizationRegistration,
    ):
        raise TypeError("localization_registration must be typed or None.")
    if point_grasp_registration is not None and not isinstance(
        point_grasp_registration, RealPointGraspRegistration
    ):
        raise TypeError("point_grasp_registration must be typed or None.")
    raw_pairs = [
        (
            registration.rgb,
            registration.rgb_binding,
            registration.availability_hash_by_route_hash[
                registration.rgb.routes[0].semantic_content_hash
            ],
        ),
        (
            registration.depth,
            registration.depth_binding,
            registration.availability_hash_by_route_hash[
                registration.depth.routes[0].semantic_content_hash
            ],
        ),
    ]
    if localization_registration is not None:
        raw_pairs.append(
            (
                localization_registration.skill,
                localization_registration.binding,
                localization_registration.availability_hash,
            )
        )
    if point_grasp_registration is not None:
        raw_pairs.append(
            (
                point_grasp_registration.skill,
                point_grasp_registration.binding,
                point_grasp_registration.availability_hash,
            )
        )
    registration_pairs = tuple(
        sorted(
            raw_pairs,
            key=lambda item: (
                item[0].definition.skill_id,
                item[0].definition.skill_version,
            ),
        )
    )
    registrations = tuple(item[0] for item in registration_pairs)
    bindings = tuple(item[1] for item in registration_pairs)
    first = bindings[0]
    for binding in bindings[1:]:
        for field_name in (
            "runtime_session_id",
            "control_domain_id",
            "target_binding_id",
            "station_manifest_hash",
            "execution_manifest_hash",
        ):
            if getattr(binding, field_name) != getattr(first, field_name):
                raise ValueError("Perception deployment bindings disagree.")
    definitions = SkillDefinitionCatalog(
        catalog_id="dexteleop.agent_skill_definitions.real_perception_only",
        catalog_version="1",
        skill_definitions=tuple(item.definition for item in registrations),
    )
    schemas = AgentSkillSchemaCatalog(
        catalog_id="dexteleop.agent_skill_schemas.real_perception_only",
        catalog_version="1",
        schema_entries=tuple(item.schema_entry for item in registrations),
    )
    io_catalog = SkillIOCatalog(
        catalog_id="dexteleop.agent_skill_io.real_perception_only",
        catalog_version="1",
        io_contracts=tuple(item.io_contract for item in registrations),
    )
    route_items = tuple(
        SkillRouteManifestEntry(
            route=item.routes[0],
            deployment_binding=binding.to_ref(),
            evaluation_plan_hash=None,
            availability_hash=availability_hash,
        )
        for item, binding, availability_hash in registration_pairs
    )
    manifest = SkillRouteManifest(
        runtime_session_id=first.runtime_session_id,
        control_domain_id=first.control_domain_id,
        station_manifest_hash=first.station_manifest_hash,
        execution_manifest_hash=first.execution_manifest_hash,
        skill_definition_catalog_hash=definitions.semantic_content_hash,
        skill_schema_catalog_hash=schemas.semantic_content_hash,
        skill_io_catalog_hash=io_catalog.semantic_content_hash,
        target_binding_id=first.target_binding_id,
        target_kind="real",
        current_mode="real_observe",
        supported_transport_features=(
            "idempotency_v1",
            "own_operation_cancel_v1",
            "own_operation_status_v1",
        ),
        skill_routes=route_items,
        explicit_exclusions=(
            "agent_estop",
            "agent_rearm",
            "agent_resume",
            "control_authority",
            "device_reset",
            "direct_hardware_refresh",
            "domain_hold",
            "motion",
            "robot_backend",
        ),
    )
    validators = AgentSkillPayloadValidatorRegistry(
        registry_id="dexteleop.agent_payload_validators.real_perception_only:"
        + first.runtime_session_id,
        validators=registration.payload_validators
        + (
            ()
            if localization_registration is None
            else (localization_registration.payload_validator,)
        )
        + (
            ()
            if point_grasp_registration is None
            else (point_grasp_registration.payload_validator,)
        ),
    )
    return RealPerceptionContractBundle(
        registration=registration,
        localization_registration=localization_registration,
        point_grasp_registration=point_grasp_registration,
        definition_catalog=definitions,
        schema_catalog=schemas,
        io_catalog=io_catalog,
        route_manifest=manifest,
        payload_validators=validators,
        source_binding_id=_required(source_binding_id, field_name="source_binding_id"),
        frame_graph_hash=_required(frame_graph_hash, field_name="frame_graph_hash"),
        tool_config_hash=_required(tool_config_hash, field_name="tool_config_hash"),
        calibration_hash=_required(calibration_hash, field_name="calibration_hash"),
    )


def build_real_perception_passive_executors(
    *,
    registration: RealPerceptionCaptureRegistration,
    provider: RealSenseAgentPerceptionProvider,
    artifact_store: RecoverableAgentArtifactStore,
    runtime_session_id: str,
    current_station_revision: Callable[[], int],
    max_retained_operations: int,
    clock: Callable[[], float] = time.time,
) -> tuple[PassiveReadSkillExecutor, PassiveReadSkillExecutor]:
    """Bind shared passive RGB/depth execution to one provider and CAS."""

    if not isinstance(registration, RealPerceptionCaptureRegistration):
        raise TypeError("registration must be typed.")
    if not isinstance(provider, RealSenseAgentPerceptionProvider):
        raise TypeError("provider must be typed.")
    if not isinstance(artifact_store, RecoverableAgentArtifactStore):
        raise TypeError("artifact_store must be typed.")
    session_id = _required(runtime_session_id, field_name="runtime_session_id")

    def read_rgb(request, operation) -> PassiveReadResult:
        is_v2 = registration.rgb.definition.skill_version == "2"
        capture = provider.capture_rgb(
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
                runtime_session_id=session_id,
                idempotency_key=f"rgb-output:{operation.operation_id}:{ordinal}",
                now=frame.acquired_at if is_v2 else capture.captured_at,
            )
            total_bytes += committed.byte_length
            metadata = {
                "ordinal": ordinal,
                "source_id": frame.source_id,
                "role": frame.role,
                **({"acquired_at": frame.acquired_at} if is_v2 else {}),
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
            },
            domain=(
                "dexteleop.agent_skill.real_rgb_capture_receipt.v2"
                if is_v2
                else "dexteleop.agent_skill.real_rgb_capture_receipt.v1"
            ),
        )
        return PassiveReadResult(
            execution_result={
                "snapshot_id": "rgb-snapshot:" + identity,
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
            },
            evidence_refs=("real-rgb-capture:" + identity,),
            output_artifact_bindings=tuple(bindings),
            reason_code="passive_rgb_capture_completed",
        )

    def read_depth(request, operation) -> PassiveReadResult:
        frame = provider.capture_depth(
            role=str(request.parameters["role"]),
            expected_station_revision=request.expected_station_revision,
        )
        expected_revision = request.expected_station_revision
        if current_station_revision() != expected_revision:
            raise RuntimeError(
                "Station revision changed after passive depth capture."
            )
        frame_binding = registration.frame_binding
        if frame_binding is None:
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
            receipt_domain = "dexteleop.agent_skill.real_depth_capture_receipt.v1"
        else:
            camera_pose_reference = (
                frame_binding.camera_pose_reference_xyz_quat_xyzw(
                    frame.camera_pose_world_xyz_quat_xyzw
                )
            )
            result = {
                "source_id": frame.source_id,
                "role": frame.role,
                "captured_at": frame.captured_at,
                "station_revision": expected_revision,
                "width": frame.width,
                "height": frame.height,
                "depth_unit": "meter",
                "invalid_depth_value": 0.0,
                "reference_frame_id": frame_binding.reference_frame_id,
                "reference_frame_spec_hash": (
                    frame_binding.reference_frame_spec_hash
                ),
                "frame_graph_hash": (
                    frame_binding.frame_graph.semantic_content_hash
                ),
                "camera_pose_reference_xyz_quat_xyzw": list(
                    camera_pose_reference
                ),
                "intrinsics": dict(frame.intrinsics),
            }
            receipt_domain = "dexteleop.agent_skill.real_depth_capture_receipt.v2"
        if current_station_revision() != expected_revision:
            raise RuntimeError(
                "Station revision changed before passive depth Artifact commit."
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
            runtime_session_id=session_id,
            idempotency_key="depth-output:" + operation.operation_id,
            now=frame.captured_at,
        )
        if current_station_revision() != expected_revision:
            raise RuntimeError(
                "Station revision changed during passive depth Artifact commit."
            )
        identity = canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "request_hash": request.semantic_content_hash,
                "result": result,
                "artifact_ref_hash": committed.semantic_content_hash,
            },
            domain=receipt_domain,
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

    return (
        PassiveReadSkillExecutor(
            port_id="agent-gateway:real-passive-rgb:" + session_id,
            executor_id=registration.rgb_binding.executor_id,
            read=read_rgb,
            current_station_revision=current_station_revision,
            clock=clock,
            max_retained_operations=max_retained_operations,
        ),
        PassiveReadSkillExecutor(
            port_id="agent-gateway:real-passive-depth:" + session_id,
            executor_id=registration.depth_binding.executor_id,
            read=read_depth,
            current_station_revision=current_station_revision,
            clock=clock,
            max_retained_operations=max_retained_operations,
        ),
    )


def _protocol_runtime(
    *,
    bundle: RealPerceptionContractBundle,
    artifact_store: RecoverableAgentArtifactStore,
    clock: Callable[[], float],
) -> AgentGatewayProtocolRuntime:
    manifest = bundle.route_manifest
    capabilities = AgentGatewaySessionCapabilities(
        runtime_session_id=manifest.runtime_session_id,
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
    return AgentGatewayProtocolRuntime(
        manifest=manifest,
        io_catalog=bundle.io_catalog,
        session_capabilities=capabilities,
        event_journal=BoundedAgentGatewayEventJournal(
            runtime_session_id=manifest.runtime_session_id,
            max_events=capabilities.event_retention_count,
            event_payload_max_bytes=capabilities.event_payload_max_bytes,
        ),
        artifact_store=artifact_store,
        artifact_transfer_broker=AgentArtifactTransferBroker(
            store=artifact_store,
            session_capabilities=capabilities,
            max_chunk_bytes=artifact_store.max_chunk_bytes,
        ),
        clock=clock,
    )


class RealPerceptionObservationProducer:
    """Publish bounded camera lifecycle status, never image bytes or control."""

    def __init__(
        self,
        *,
        bundle: RealPerceptionContractBundle,
        provider: RealSenseAgentPerceptionProvider,
        revision_tracker: AgentStationRevisionTracker,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.bundle = bundle
        self.provider = provider
        self.revision_tracker = revision_tracker
        self.clock = clock

    def _status(self) -> tuple[dict[str, object], bool]:
        sources = []
        healthy = True
        for source_id, source in sorted(
            self.provider.source_binding.camera_sources.items()
        ):
            info_provider = getattr(source, "device_info", None)
            info = info_provider() if callable(info_provider) else {}
            running = bool(isinstance(info, Mapping) and info.get("running") is True)
            healthy = healthy and running
            sources.append(
                {
                    "source_id": source_id,
                    "running": running,
                    "generation": (
                        info.get("generation") if isinstance(info, Mapping) else None
                    ),
                    "serial_number": (
                        info.get("serial_number") if isinstance(info, Mapping) else None
                    ),
                }
            )
        return {"camera_sources": sources}, healthy

    @property
    def current_station_revision(self) -> int:
        status, _healthy = self._status()
        return self.revision_tracker.observe(
            {
                "source_binding_hash": (
                    self.provider.source_binding.semantic_content_hash
                ),
                "camera_status": status,
            }
        ).station_revision

    def capture_and_publish(self, service: AgentGatewayService):
        status, healthy = self._status()
        revision = self.current_station_revision
        captured_at = float(self.clock())
        manifest = self.bundle.route_manifest
        identity = canonical_content_hash(
            {
                "runtime_session_id": manifest.runtime_session_id,
                "station_revision": revision,
                "captured_at": captured_at,
                "camera_status": status,
            },
            domain="dexteleop.agent_gateway.real_perception_observation.v1",
        )
        snapshot = AgentObservationSnapshot(
            snapshot_id="real-perception-observation:" + identity,
            runtime_session_id=manifest.runtime_session_id,
            skill_route_manifest_hash=manifest.semantic_content_hash,
            station_revision=revision,
            captured_at=captured_at,
            available_at=captured_at,
            clock_domain_id=REAL_PERCEPTION_CLOCK_DOMAIN_ID_V1,
            authority_epoch=0,
            execution_epoch=0,
            dispatch_owner_generation=0,
            lifecycle_state="running" if healthy else "unavailable",
            execution_gate="read_only",
            active_operation_id=None,
            source_binding_id=self.bundle.source_binding_id,
            producer_epoch=1,
            target_binding_id=manifest.target_binding_id,
            execution_manifest_hash=manifest.execution_manifest_hash,
            frame_graph_hash=self.bundle.frame_graph_hash,
            tool_config_hash=self.bundle.tool_config_hash,
            calibration_hash=self.bundle.calibration_hash,
            measurement_summary=status,
            measurement_freshness_ms=0,
            measurement_valid=healthy,
            fault_class=None if healthy else "camera_source_unavailable",
            fault_rearmable=False,
        )
        return service.publish_observation(snapshot=snapshot, now=captured_at)


@dataclass
class RealPerceptionGatewaySession:
    bundle: RealPerceptionContractBundle
    provider: RealSenseAgentPerceptionProvider
    observation_producer: RealPerceptionObservationProducer
    rgb_executor: PassiveReadSkillExecutor
    depth_executor: PassiveReadSkillExecutor
    localization_executor: PassiveReadSkillExecutor | None
    point_grasp_executor: PassiveReadSkillExecutor | None
    hardware_view_provider: HardwareViewProvider
    service: AgentGatewayService
    _close_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _close_outcome: object | None = field(default=None, init=False)

    def wire_adapter(
        self, *, authenticated_client_id: str
    ) -> BoundAgentGatewayWireAdapter:
        return BoundAgentGatewayWireAdapter(
            authenticated_client_id=authenticated_client_id,
            definition_catalog=self.bundle.definition_catalog,
            schema_catalog=self.bundle.schema_catalog,
            service=self.service,
            publish_observation=lambda: self.observation_producer.capture_and_publish(
                self.service
            ),
            current_station_revision=lambda: (
                self.observation_producer.current_station_revision
            ),
            operation_launcher=self.service.launch_operation,
            hardware_view_provider=self.hardware_view_provider,
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
            raise ValueError("Unix peer map must exactly cover allowed clients.")
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
            outcome = self._close_outcome
        if not self.rgb_executor.wait_all(timeout_sec=5.0):
            raise RuntimeError("Passive RGB reads did not quiesce during close.")
        if not self.depth_executor.wait_all(timeout_sec=5.0):
            raise RuntimeError("Passive depth reads did not quiesce during close.")
        if self.localization_executor is not None and not self.localization_executor.wait_all(
            timeout_sec=65.0
        ):
            raise RuntimeError("Passive object localization did not quiesce during close.")
        if self.point_grasp_executor is not None and not self.point_grasp_executor.wait_all(
            timeout_sec=125.0
        ):
            raise RuntimeError("Passive point-grasp planning did not quiesce during close.")
        return outcome


def assemble_real_perception_gateway_session(
    *,
    bundle: RealPerceptionContractBundle,
    source_binding: RealPerceptionSourceBinding,
    artifact_store: RecoverableAgentArtifactStore,
    allowed_client_ids: tuple[str, ...],
    localization_provider: ObjectLocalizationProvider | None = None,
    point_grasp_provider: PointGraspProvider | None = None,
    max_retained_operations: int = 16,
    clock: Callable[[], float] = time.time,
) -> RealPerceptionGatewaySession:
    """Assemble a passive-only session without starting cameras or transport."""

    clients = tuple(
        sorted({_required(item, field_name="client_id") for item in allowed_client_ids})
    )
    if not clients:
        raise ValueError("allowed_client_ids must not be empty.")
    revision_tracker = AgentStationRevisionTracker()
    provider = RealSenseAgentPerceptionProvider(
        source_binding=source_binding,
        current_station_revision=lambda: revision_tracker.current_revision,
        max_frame_age_ms=bundle.registration.profile.max_frame_age_ms,
        max_inter_camera_skew_ms=(bundle.registration.profile.max_inter_camera_skew_ms),
        clock=clock,
    )
    provider.bind_profile(bundle.registration.profile)
    producer = RealPerceptionObservationProducer(
        bundle=bundle,
        provider=provider,
        revision_tracker=revision_tracker,
        clock=clock,
    )
    rgb_executor, depth_executor = build_real_perception_passive_executors(
        registration=bundle.registration,
        provider=provider,
        artifact_store=artifact_store,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        current_station_revision=lambda: producer.current_station_revision,
        max_retained_operations=max_retained_operations,
        clock=clock,
    )
    ports = {
        bundle.registration.rgb.routes[0].semantic_content_hash: rgb_executor,
        bundle.registration.depth.routes[0].semantic_content_hash: depth_executor,
    }
    localization_executor = None
    if bundle.localization_registration is not None:
        if localization_provider is None:
            raise ValueError("Localization registration requires its model provider.")
        localization_executor = build_real_object_localization_executor(
            registration=bundle.localization_registration,
            provider=localization_provider,
            artifact_store=artifact_store,
            runtime_session_id=bundle.route_manifest.runtime_session_id,
            current_station_revision=lambda: producer.current_station_revision,
            max_retained_operations=max_retained_operations,
            clock=clock,
        )
        ports[
            bundle.localization_registration.skill.routes[0].semantic_content_hash
        ] = localization_executor
    elif localization_provider is not None:
        raise ValueError("Localization provider cannot exist without its registration.")
    point_grasp_executor = None
    if bundle.point_grasp_registration is not None:
        if point_grasp_provider is None:
            raise ValueError("Point-grasp registration requires its model provider.")
        point_grasp_executor = build_real_point_grasp_executor(
            registration=bundle.point_grasp_registration,
            provider=point_grasp_provider,
            artifact_store=artifact_store,
            runtime_session_id=bundle.route_manifest.runtime_session_id,
            current_station_revision=lambda: producer.current_station_revision,
            max_retained_operations=max_retained_operations,
            clock=clock,
        )
        ports[bundle.point_grasp_registration.skill.routes[0].semantic_content_hash] = point_grasp_executor
    elif point_grasp_provider is not None:
        raise ValueError("Point-grasp provider cannot exist without its registration.")
    registry = SkillExecutorRegistry(
        manifest=bundle.route_manifest,
        bindings=tuple(
            SkillExecutorBinding.from_manifest_entry(
                manifest=bundle.route_manifest,
                entry=entry,
                execution_port=ports[entry.route.semantic_content_hash],
            )
            for entry in bundle.route_manifest.skill_routes
        ),
    )
    operation_store = BoundedInMemoryAgentSkillOperationStore(
        max_retained_operations=max_retained_operations
    )
    domain_port = ObservationOnlyDomainControlPort(
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        clock=clock,
    )
    service = AgentGatewayService(
        manifest=bundle.route_manifest,
        access_policy=AgentSkillAccessPolicy(
            access_policy_id="real-perception-only:"
            + bundle.route_manifest.runtime_session_id,
            allowed_client_ids=clients,
            allowed_modes=("real_observe",),
            max_request_timeout_ms=max(
                entry.route.max_operation_duration_ms
                for entry in bundle.route_manifest.skill_routes
            ),
            max_grant_lifetime_sec=1.0,
        ),
        executor_registry=registry,
        domain_control_port=domain_port,
        payload_validators=bundle.payload_validators,
        operation_store=operation_store,
        protocol_runtime=_protocol_runtime(
            bundle=bundle,
            artifact_store=artifact_store,
            clock=clock,
        ),
        clock=clock,
        clock_domain_id=REAL_PERCEPTION_CLOCK_DOMAIN_ID_V1,
    )
    camera_route_hashes = (
        bundle.registration.rgb.routes[0].semantic_content_hash,
        bundle.registration.depth.routes[0].semantic_content_hash,
    )
    resource_bindings = [
        HardwareResourceBinding(
            resource_id="perception.cameras",
            resource_kind="camera_bundle",
            route_hashes=camera_route_hashes,
            health_provider=lambda: (
                "ready" if producer._status()[1] else "unavailable"
            ),
            control_claimed_provider=lambda: False,
        )
    ]
    if bundle.localization_registration is not None:
        resource_bindings.append(
            HardwareResourceBinding(
                resource_id="perception.object_localization_model",
                resource_kind="local_model",
                route_hashes=(
                    bundle.localization_registration.skill.routes[
                        0
                    ].semantic_content_hash,
                ),
                health_provider=lambda: (
                    "ready"
                    if localization_provider is not None
                    and localization_provider.started
                    else "unavailable"
                ),
                control_claimed_provider=lambda: False,
            )
        )
    if bundle.point_grasp_registration is not None:
        resource_bindings.append(
            HardwareResourceBinding(
                resource_id="perception.point_grasp_model",
                resource_kind="local_model",
                route_hashes=(bundle.point_grasp_registration.skill.routes[0].semantic_content_hash,),
                health_provider=lambda: (
                    "ready" if point_grasp_provider is not None and point_grasp_provider.started else "unavailable"
                ),
                control_claimed_provider=lambda: False,
            )
        )
    hardware_view = HardwareViewProvider(
        manifest=bundle.route_manifest,
        operation_store=operation_store,
        bindings=tuple(resource_bindings),
    )
    return RealPerceptionGatewaySession(
        bundle=bundle,
        provider=provider,
        observation_producer=producer,
        rgb_executor=rgb_executor,
        depth_executor=depth_executor,
        localization_executor=localization_executor,
        point_grasp_executor=point_grasp_executor,
        hardware_view_provider=hardware_view,
        service=service,
    )


__all__ = [
    "REAL_PERCEPTION_CLOCK_DOMAIN_ID_V1",
    "RealPerceptionContractBundle",
    "RealPerceptionGatewaySession",
    "RealPerceptionObservationProducer",
    "assemble_real_perception_gateway_session",
    "build_real_perception_contract_bundle",
    "build_real_perception_passive_executors",
]
