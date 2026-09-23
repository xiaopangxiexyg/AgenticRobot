"""Process owner for the independent camera-only real perception Gateway."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from agent_gateway.profiles.segmentation import (
    LiveSegmentationProvider,
    SegmentationRegistration,
    build_segmentation_registration,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.transport.client_bindings import AgentGatewayUnixClientBindings
from dataset_contracts.canonical_encoding import canonical_content_hash

from .localization_deployment import (
    RealObjectLocalizationRegistration,
    build_real_object_localization_registration,
)
from .localization_gateway import ObjectLocalizationProvider
from .perception_deployment import (
    REAL_CAPTURE_MAX_ARTIFACT_BYTES_V1,
    RealPerceptionCaptureRegistration,
    build_real_perception_capture_registration_v2,
)
from .perception_frames import RealPerceptionFrameBinding
from .perception_gateway import (
    RealPerceptionContractBundle,
    RealPerceptionGatewaySession,
    assemble_real_perception_gateway_session,
    build_real_perception_contract_bundle,
)
from .realsense_perception import (
    CameraOnlyRealPerceptionResources,
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
)
from .station_point_grasp_deployment import (
    StationBoundPointGraspRegistration,
    build_station_bound_point_grasp_registration,
)
from .station_point_grasp_render_deployment import (
    StationBoundPointGraspRenderRegistration,
    build_station_bound_point_grasp_render_registration,
)


class CameraOnlyRealPerceptionApplicationState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    FAULT = "fault"
    CLOSED = "closed"


class CameraOnlyRealPerceptionApplication:
    """Start camera resources before publishing the Unix endpoint."""

    def __init__(
        self,
        *,
        session: RealPerceptionGatewaySession,
        resources: CameraOnlyRealPerceptionResources,
        server_factory: Callable[[], object],
    ) -> None:
        if not isinstance(session, RealPerceptionGatewaySession):
            raise TypeError("session must be typed.")
        if not isinstance(resources, CameraOnlyRealPerceptionResources):
            raise TypeError("resources must be typed.")
        if not callable(server_factory):
            raise TypeError("server_factory must be callable.")
        self.session = session
        self.resources = resources
        self.server_factory = server_factory
        self._lock = threading.RLock()
        self._state = CameraOnlyRealPerceptionApplicationState.CREATED
        self._server: object | None = None

    @property
    def state(self) -> CameraOnlyRealPerceptionApplicationState:
        with self._lock:
            return self._state

    def start(self) -> None:
        with self._lock:
            if self._state != CameraOnlyRealPerceptionApplicationState.CREATED:
                raise RuntimeError("camera-only perception application starts once.")
        resources_started = False
        server = None
        try:
            self.resources.start()
            resources_started = True
            # Publish one admitted planning snapshot before accepting requests.
            outcome = self.session.observation_producer.capture_and_publish(
                self.session.service
            )
            if not outcome.accepted or outcome.snapshot is None:
                raise RuntimeError("camera-only startup observation was rejected.")
            server = self.server_factory()
            for name in ("serve_forever", "shutdown", "server_close"):
                if not callable(getattr(server, name, None)):
                    raise TypeError(f"server must expose {name}.")
            with self._lock:
                self._server = server
                self._state = CameraOnlyRealPerceptionApplicationState.RUNNING
        except Exception as start_error:
            cleanup_errors = []
            if server is not None:
                try:
                    server.server_close()
                except Exception as exc:  # noqa: BLE001 - aggregate startup cleanup.
                    cleanup_errors.append(f"gateway:{type(exc).__name__}:{exc}")
            else:
                try:
                    self.session.close()
                except Exception as exc:  # noqa: BLE001 - aggregate startup cleanup.
                    cleanup_errors.append(f"gateway:{type(exc).__name__}:{exc}")
            if resources_started:
                try:
                    self.resources.close()
                except Exception as exc:  # noqa: BLE001 - aggregate startup cleanup.
                    cleanup_errors.append(f"resources:{type(exc).__name__}:{exc}")
            with self._lock:
                self._state = CameraOnlyRealPerceptionApplicationState.FAULT
            if cleanup_errors:
                raise RuntimeError(
                    "camera-only perception startup cleanup failed: "
                    + "; ".join(cleanup_errors)
                ) from start_error
            raise

    def serve_forever(
        self,
        *,
        ready_callback: Callable[[], None] | None = None,
        poll_interval_sec: float = 0.05,
    ) -> None:
        if ready_callback is not None and not callable(ready_callback):
            raise TypeError("ready_callback must be callable or None.")
        self.start()
        with self._lock:
            server = self._server
        assert server is not None
        try:
            if ready_callback is not None:
                ready_callback()
            server.serve_forever(poll_interval=float(poll_interval_sec))
        finally:
            self.close()

    def request_shutdown(self) -> bool:
        with self._lock:
            server = self._server
        if server is None:
            return False
        server.shutdown()
        return True

    def close(self) -> None:
        with self._lock:
            if self._state == CameraOnlyRealPerceptionApplicationState.CLOSED:
                return
            server = self._server
        errors = []
        if server is not None:
            try:
                server.server_close()
            except Exception as exc:  # noqa: BLE001 - resources still must close.
                errors.append(f"gateway:{type(exc).__name__}:{exc}")
        else:
            try:
                self.session.close()
            except Exception as exc:  # noqa: BLE001 - resources still must close.
                errors.append(f"gateway:{type(exc).__name__}:{exc}")
        try:
            self.resources.close()
        except Exception as exc:  # noqa: BLE001 - aggregate cleanup evidence.
            errors.append(f"resources:{type(exc).__name__}:{exc}")
        with self._lock:
            self._state = (
                CameraOnlyRealPerceptionApplicationState.CLOSED
                if not errors
                else CameraOnlyRealPerceptionApplicationState.FAULT
            )
        if errors:
            raise RuntimeError(
                "camera-only perception shutdown failed: " + "; ".join(errors)
            )


@dataclass(frozen=True)
class CameraOnlyRealPerceptionComposition:
    source_binding: RealPerceptionSourceBinding
    frame_binding: RealPerceptionFrameBinding
    registration: RealPerceptionCaptureRegistration
    localization_registration: RealObjectLocalizationRegistration | None
    localization_provider: ObjectLocalizationProvider | None
    segmentation_registration: SegmentationRegistration | None
    segmentation_provider: LiveSegmentationProvider | None
    station_point_grasp_registration: StationBoundPointGraspRegistration | None
    station_point_grasp_render_registration: (
        StationBoundPointGraspRenderRegistration | None
    )
    point_grasp_planner: object | None
    contract_bundle: RealPerceptionContractBundle
    artifact_store: RecoverableAgentArtifactStore
    resources: CameraOnlyRealPerceptionResources
    session: RealPerceptionGatewaySession
    application: CameraOnlyRealPerceptionApplication
    socket_path: Path
    client_bindings: AgentGatewayUnixClientBindings
    deployment_profile_hash: str


def assemble_camera_only_real_perception_composition(
    *,
    source_binding: RealPerceptionSourceBinding,
    socket_path: str | Path,
    artifact_root: str | Path,
    client_bindings: AgentGatewayUnixClientBindings,
    runtime_session_id: str,
    control_domain_id: str,
    frame_binding: RealPerceptionFrameBinding,
    object_localizer: object | None = None,
    point_grasp_planner: object | None = None,
    max_retained_operations: int = 16,
    peer_idle_timeout_sec: float = 10.0,
    clock: Callable[[], float] = time.time,
) -> CameraOnlyRealPerceptionComposition:
    """Assemble camera-only resources and Routes without starting hardware."""

    if not isinstance(source_binding, RealPerceptionSourceBinding):
        raise TypeError("source_binding must be typed.")
    if not isinstance(frame_binding, RealPerceptionFrameBinding):
        raise TypeError("frame_binding must be typed.")
    if not isinstance(client_bindings, AgentGatewayUnixClientBindings):
        raise TypeError("client_bindings must be typed.")
    endpoint = Path(socket_path)
    root = Path(artifact_root)
    if not endpoint.is_absolute() or not root.is_absolute():
        raise ValueError("socket_path and artifact_root must be absolute.")
    runtime_id = str(runtime_session_id).strip()
    domain_id = str(control_domain_id).strip()
    if not runtime_id or not domain_id:
        raise ValueError("session and control-domain IDs are required.")
    frame_graph_hash = frame_binding.frame_graph.semantic_content_hash
    localizer_hash = (
        None
        if object_localizer is None
        else str(getattr(object_localizer, "semantic_content_hash", "")).strip()
    )
    if object_localizer is not None and not localizer_hash:
        raise ValueError("object_localizer must expose semantic_content_hash.")
    planner_hash = (
        None
        if point_grasp_planner is None
        else str(getattr(point_grasp_planner, "semantic_content_hash", "")).strip()
    )
    if point_grasp_planner is not None:
        if not planner_hash:
            raise ValueError("point_grasp_planner must expose semantic_content_hash.")
        if object_localizer is None:
            raise ValueError("Live point-grasp requires object localization.")
        if (
            getattr(point_grasp_planner, "point_segmenter", None)
            is not object_localizer
        ):
            raise ValueError(
                "Live point-grasp must share the exact object-localizer instance."
            )
    target_binding_payload = {
        "source_binding_hash": source_binding.semantic_content_hash,
        "frame_binding_hash": frame_binding.semantic_content_hash,
        "resource_kind": "camera_bundle",
        "object_localizer_hash": localizer_hash,
    }
    if planner_hash is not None:
        target_binding_payload["point_grasp_planner_hash"] = planner_hash
    target_binding_hash = canonical_content_hash(
        target_binding_payload,
        domain="dexteleop.real_perception.target_binding.v1",
    )
    target_binding_id = "real-realsense-only:" + target_binding_hash
    station_manifest_hash = canonical_content_hash(
        {
            "frame_graph_hash": frame_graph_hash,
            "reference_frame_id": frame_binding.reference_frame_id,
            "reference_frame_spec_hash": (frame_binding.reference_frame_spec_hash),
            "source_binding_hash": source_binding.semantic_content_hash,
        },
        domain="dexteleop.real_perception.station_manifest.v1",
    )
    execution_manifest_payload = {
        "resource_owner": "camera_only_real_perception_application",
        "source_binding_hash": source_binding.semantic_content_hash,
        "robot_backend": "absent",
        "command_owner_count": 0,
        "downstream_control_effects": "forbidden",
        "object_localizer_hash": localizer_hash,
    }
    if planner_hash is not None:
        execution_manifest_payload["point_grasp_planner_hash"] = planner_hash
    execution_manifest_hash = canonical_content_hash(
        execution_manifest_payload,
        domain="dexteleop.real_perception.execution_manifest.v1",
    )
    catalog_provider = RealSenseAgentPerceptionProvider(
        source_binding=source_binding,
        current_station_revision=lambda: 0,
        clock=clock,
    )
    registration = build_real_perception_capture_registration_v2(
        rgb_source_catalog=catalog_provider.rgb_source_catalog(),
        depth_source_catalog=catalog_provider.depth_source_catalog(),
        runtime_session_id=runtime_id,
        control_domain_id=domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        source_binding_hash=source_binding.semantic_content_hash,
        frame_binding=frame_binding,
        current_mode="real_observe",
    )
    localization_registration = None
    localization_provider = None
    segmentation_registration = None
    if object_localizer is not None:
        model_spec = getattr(object_localizer, "model_spec", None)
        model_profile_id = str(getattr(model_spec, "model_profile_id", "")).strip()
        model_profile_hash = str(
            getattr(model_spec, "semantic_content_hash", "")
        ).strip()
        if not model_profile_id or not model_profile_hash:
            raise ValueError("object_localizer must expose one verified model_spec.")
        localization_registration = build_real_object_localization_registration(
            model_profile_id=model_profile_id,
            model_profile_hash=model_profile_hash,
            inference_implementation_hash=localizer_hash,
            local_path_enabled=False,
            runtime_session_id=runtime_id,
            control_domain_id=domain_id,
            target_binding_id=target_binding_id,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
        )
        localization_provider = ObjectLocalizationProvider(
            localizer=object_localizer,
            registration=localization_registration,
            allowed_local_image_roots=(),
        )
        segmentation_registration = build_segmentation_registration(
            target_kind="real",
            current_mode="real_observe",
            source_id_by_role={
                str(item["role"]): str(item["source_id"])
                for item in catalog_provider.rgb_source_catalog()
            },
            model_profile_id=model_profile_id,
            model_profile_hash=model_profile_hash,
            inference_implementation_hash=localizer_hash,
            runtime_session_id=runtime_id,
            control_domain_id=domain_id,
            target_binding_id=target_binding_id,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
        )
    station_point_grasp_registration = None
    station_point_grasp_render_registration = None
    if point_grasp_planner is not None:
        model_spec = getattr(point_grasp_planner, "model_spec", None)
        model_profile_id = str(getattr(model_spec, "model_profile_id", "")).strip()
        model_profile_hash = str(
            getattr(model_spec, "semantic_content_hash", "")
        ).strip()
        backend_id = str(getattr(point_grasp_planner, "backend_id", "")).strip()
        if not model_profile_id or not model_profile_hash or not backend_id:
            raise ValueError(
                "point_grasp_planner must expose backend_id and verified model_spec."
            )
        station_point_grasp_registration = build_station_bound_point_grasp_registration(
            aligned_rgbd_source_catalog=tuple(
                dict(item) for item in catalog_provider.aligned_rgbd_source_catalog()
            ),
            source_binding_hash=source_binding.semantic_content_hash,
            frame_binding_hash=frame_binding.semantic_content_hash,
            frame_graph_hash=frame_graph_hash,
            source_frame_id=frame_binding.source_frame_id,
            source_frame_spec_hash=frame_binding.source_frame_spec_hash,
            reference_frame_id=frame_binding.reference_frame_id,
            reference_frame_spec_hash=frame_binding.reference_frame_spec_hash,
            backend_id=backend_id,
            model_profile_id=model_profile_id,
            model_profile_hash=model_profile_hash,
            inference_implementation_hash=planner_hash,
            runtime_session_id=runtime_id,
            control_domain_id=domain_id,
            target_binding_id=target_binding_id,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
        )
        station_point_grasp_render_registration = (
            build_station_bound_point_grasp_render_registration(
                point_grasp_registration=station_point_grasp_registration,
            )
        )
    tool_config_hash = canonical_content_hash(
        {"tool_configuration": "not_applicable_camera_only"},
        domain="dexteleop.real_perception.no_tool_config.v1",
    )
    source_binding_id = "real-realsense:" + source_binding.semantic_content_hash
    bundle = build_real_perception_contract_bundle(
        registration=registration,
        source_binding_id=source_binding_id,
        frame_graph_hash=frame_graph_hash,
        tool_config_hash=tool_config_hash,
        calibration_hash=source_binding.semantic_content_hash,
        localization_registration=localization_registration,
        segmentation_registration=segmentation_registration,
        station_point_grasp_registration=station_point_grasp_registration,
        station_point_grasp_render_registration=(
            station_point_grasp_render_registration
        ),
    )
    partition = canonical_content_hash(
        {"runtime_session_id": runtime_id},
        domain="dexteleop.agent_gateway.artifact_store_partition.v1",
    )
    artifact_store = RecoverableAgentArtifactStore(
        root=root / partition,
        quota_bytes=512 * 1024 * 1024,
        max_artifact_bytes=REAL_CAPTURE_MAX_ARTIFACT_BYTES_V1,
        max_chunk_bytes=1024 * 1024,
    )
    try:
        session = assemble_real_perception_gateway_session(
            bundle=bundle,
            source_binding=source_binding,
            artifact_store=artifact_store,
            allowed_client_ids=client_bindings.allowed_client_ids,
            localization_provider=localization_provider,
            segmentation_localizer=object_localizer,
            station_point_grasp_planner=point_grasp_planner,
            max_retained_operations=max_retained_operations,
            clock=clock,
        )
    except Exception:
        artifact_store.close()
        raise
    resources = CameraOnlyRealPerceptionResources(
        camera_sources=source_binding.camera_sources,
        artifact_store=artifact_store,
        additional_resources=(
            tuple(
                resource
                for resource in (
                    localization_provider,
                    session.station_point_grasp_provider,
                )
                if resource is not None
            )
        ),
    )

    def server_factory():
        return session.build_unix_json_server(
            socket_path=str(endpoint),
            client_id_by_uid=client_bindings.client_id_by_uid,
            socket_mode=client_bindings.socket_mode,
            peer_idle_timeout_sec=peer_idle_timeout_sec,
        )

    application = CameraOnlyRealPerceptionApplication(
        session=session,
        resources=resources,
        server_factory=server_factory,
    )
    deployment_profile_payload = {
        "source_binding_hash": source_binding.semantic_content_hash,
        "frame_binding_hash": frame_binding.semantic_content_hash,
        "frame_graph_hash": frame_graph_hash,
        "route_manifest_hash": bundle.route_manifest.semantic_content_hash,
        "artifact_max_bytes": REAL_CAPTURE_MAX_ARTIFACT_BYTES_V1,
        "object_localizer_hash": localizer_hash,
        "local_path_test_input": False,
        "robot_backend": "absent",
    }
    if planner_hash is not None:
        deployment_profile_payload["point_grasp_planner_hash"] = planner_hash
        deployment_profile_payload["point_grasp_enabled"] = True
        deployment_profile_payload["point_grasp_candidate_render_enabled"] = True
    deployment_profile_hash = canonical_content_hash(
        deployment_profile_payload,
        domain="dexteleop.real_perception.camera_only_profile.v1",
    )
    return CameraOnlyRealPerceptionComposition(
        source_binding=source_binding,
        frame_binding=frame_binding,
        registration=registration,
        localization_registration=localization_registration,
        localization_provider=localization_provider,
        segmentation_registration=segmentation_registration,
        segmentation_provider=session.segmentation_provider,
        station_point_grasp_registration=station_point_grasp_registration,
        station_point_grasp_render_registration=(
            station_point_grasp_render_registration
        ),
        point_grasp_planner=point_grasp_planner,
        contract_bundle=bundle,
        artifact_store=artifact_store,
        resources=resources,
        session=session,
        application=application,
        socket_path=endpoint,
        client_bindings=client_bindings,
        deployment_profile_hash=deployment_profile_hash,
    )


__all__ = [
    "CameraOnlyRealPerceptionApplication",
    "CameraOnlyRealPerceptionApplicationState",
    "CameraOnlyRealPerceptionComposition",
    "assemble_camera_only_real_perception_composition",
]
