"""Finite no-motion probe for the checked-in real perception Skill profile."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import tempfile
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agent_gateway.protocol.contracts import (
    AgentOwnOperationQuery,
    AgentSkillRequest,
)
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from agent_gateway.transport.unix_endpoint import (
    allocate_agent_gateway_unix_endpoint,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_MEDIA_TYPE_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CAMERA_CONFIG = (
    _REPO_ROOT / "teleop_runtime/configs/devices/agent_gateway_realsense.json"
)
_CAMERA_REGISTRY = _REPO_ROOT / "robot_motion/configs/rm65_camera_registry.json"
_COLLISION_SCENE_PROFILE = "dual_arm_hand_proxy_open"
_RGB_SKILL_ID = "perception.capture_rgb"
_DEPTH_SKILL_ID = "perception.capture_depth"
_SKILL_VERSION_BY_ID = {
    _RGB_SKILL_ID: "1",
    _DEPTH_SKILL_ID: "2",
}
_REQUESTED_TIMEOUT_MS = 3_000
_ARTIFACT_CHUNK_BYTES = 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_gateway.profiles.real.perception_probe",
        description=(
            "Run the checked-in RealSense camera-only Gateway, invoke its RGB and "
            "depth read-only Skills, and validate the returned Artifacts. The probe "
            "has no robot backend, motion Route, dispatch, HOLD, or hardware reset."
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="absolute isolated directory for transient Artifact and JSON evidence",
    )
    parser.add_argument("--client-id", default="rpent.real-perception-probe")
    parser.add_argument("--operation-timeout-sec", type=float, default=10.0)
    return parser


def _prepare_run_root(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError("--run-root must be absolute.")
    root = Path(os.path.abspath(expanded))
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("--run-root must be a real directory, not a symlink.")
    return root


def _allocate_default_socket_path() -> tuple[Path, Path, Path]:
    """Allocate one short private endpoint independent of ambient ``TMPDIR``."""

    # Harness and other isolated runners intentionally use descriptive, deeply
    # nested temporary roots.  AF_UNIX has a separate 107-byte pathname limit,
    # so reserve a short private root explicitly instead of deriving the socket
    # pathname from that ambient artifact namespace.
    endpoint_root = Path(
        tempfile.mkdtemp(prefix=f"ar-gw-{os.getuid()}-", dir="/tmp")
    )
    try:
        allocation = allocate_agent_gateway_unix_endpoint(
            endpoint_root=endpoint_root,
        )
    except BaseException:
        endpoint_root.rmdir()
        raise
    return allocation.socket_path, allocation.endpoint_dir, endpoint_root


def _scene_geometry() -> tuple[Mapping[str, np.ndarray], Any]:
    # Keep ``--help`` and parser inspection free of the optional control-stack
    # environment. Actual probing resolves only the checked-in scene geometry.
    from robot_motion.rm65 import (
        build_rm65_dual_arm_scene_profile,
        load_world_from_arm_base_transforms,
        load_world_from_fixed_link_transform,
    )

    from .perception_frames import build_real_perception_station_frame_binding

    scene_profile = build_rm65_dual_arm_scene_profile(_COLLISION_SCENE_PROFILE)
    world_from_base = load_world_from_arm_base_transforms(scene_profile)
    scene_mount_hash = canonical_content_hash(
        {side: world_from_base[side].tolist() for side in ("left", "right")},
        domain="dexteleop.real_skill.rm65_scene_mounts.v1",
    )
    frame_binding = build_real_perception_station_frame_binding(
        collision_scene_profile=_COLLISION_SCENE_PROFILE,
        scene_mount_hash=scene_mount_hash,
        scene_world_from_canonical_station=load_world_from_fixed_link_transform(
            scene_profile,
            link_name="workspace_near_boundary_frame",
        ),
    )
    return world_from_base, frame_binding


def _build_source_binding(**kwargs: Any) -> Any:
    from .realsense_perception import build_real_perception_source_binding

    return build_real_perception_source_binding(**kwargs)


def _assemble_composition(**kwargs: Any) -> Any:
    from .perception_application import (
        assemble_camera_only_real_perception_composition,
    )

    return assemble_camera_only_real_perception_composition(**kwargs)


def _assert_no_motion_surface(composition: Any) -> None:
    entries = composition.contract_bundle.route_manifest.skill_routes
    if {entry.route.skill_id for entry in entries} != {
        _RGB_SKILL_ID,
        _DEPTH_SKILL_ID,
    }:
        raise RuntimeError("probe route surface is not perception-only.")
    for entry in entries:
        route = entry.route
        if (
            route.control_effect_class != "read_only"
            or route.privilege_tier != "t0_read_only"
            or route.may_move_robot
            or route.may_open_publication
            or route.requires_operator_confirmation
            or route.requires_deadman
        ):
            raise RuntimeError(f"probe route is not strictly passive: {route.route_id}")
    if hasattr(composition, "lower") or hasattr(composition.resources, "lower"):
        raise RuntimeError("probe composition unexpectedly owns a lower robot backend.")


def _capture_executor_errors(composition: Any) -> dict[str, str]:
    """Retain the bounded provider error that the generic worker seals by type."""

    errors: dict[str, str] = {}
    for executor in (
        composition.session.rgb_executor,
        composition.session.depth_executor,
    ):
        reader = executor._read

        def wrapped(request: Any, operation: Any, *, _reader: Any = reader) -> Any:
            try:
                return _reader(request, operation)
            except Exception as error:
                errors[operation.operation_id] = f"{type(error).__name__}:{error}"
                raise

        executor._read = wrapped
    return errors


def _request_payload(
    *,
    composition: Any,
    client_id: str,
    client_sequence: int,
    observation: Mapping[str, Any],
    skill_id: str,
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest = composition.contract_bundle.route_manifest
    try:
        skill_version = _SKILL_VERSION_BY_ID[skill_id]
    except KeyError as exc:
        raise ValueError("Probe requested an unknown perception Skill.") from exc
    entry = manifest.route(skill_id, skill_version)
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    request_nonce = uuid.uuid4().hex
    return AgentSkillRequest(
        client_id=client_id,
        request_id=f"request:real-perception-probe:{request_nonce}",
        idempotency_key=f"idempotency:real-perception-probe:{request_nonce}",
        client_sequence=client_sequence,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        expected_station_revision=int(observation["station_revision"]),
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters=parameters,
        observation_snapshot_id=str(observation["snapshot_id"]),
        requested_mode="real_observe",
        requested_timeout_ms=_REQUESTED_TIMEOUT_MS,
    ).to_wire()


def _wait_for_terminal_operation(
    *,
    adapter: Any,
    client_id: str,
    runtime_session_id: str,
    manifest_hash: str,
    operation_id: str,
    timeout_sec: float,
    executor_errors: Mapping[str, str],
) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout_sec
    query_index = 0
    while True:
        outcome = adapter.get_own_operation(
            AgentOwnOperationQuery(
                client_id=client_id,
                query_id=f"query:real-perception-probe:{operation_id}:{query_index}",
                runtime_session_id=runtime_session_id,
                skill_route_manifest_hash=manifest_hash,
                operation_id=operation_id,
            ).to_wire()
        )
        operation = outcome.get("operation")
        if not outcome.get("accepted") or not isinstance(operation, Mapping):
            raise RuntimeError(
                f"perception operation query failed: {outcome.get('reason_code')}"
            )
        if bool(operation["terminal"]):
            if operation["status"] != "completed":
                raise RuntimeError(
                    "perception Skill did not complete: "
                    f"{operation['status']}:{operation['reason_code']}"
                )
            return operation
        execution_port = adapter.service.execution_port_for_operation(operation_id)
        if execution_port.wait(operation_id, timeout_sec=0.0):
            completion = execution_port.completion_for(operation_id)
            raise RuntimeError(
                "perception executor exited without a terminal operation: "
                f"status={operation.get('status')}:"
                f"completion={getattr(completion, 'reason_code', None)}:"
                f"executor_error={executor_errors.get(operation_id)}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "perception Skill timed out: "
                f"{operation_id}:status={operation.get('status')}:"
                f"reason={operation.get('reason_code')}:"
                f"started_at={operation.get('started_at')}:"
                f"deadline_at={operation.get('deadline_at')}:now={time.time()}"
            )
        query_index += 1
        time.sleep(0.02)


def _run_skill(
    *,
    composition: Any,
    adapter: Any,
    client_id: str,
    client_sequence: int,
    skill_id: str,
    parameters: Mapping[str, Any],
    timeout_sec: float,
    executor_errors: Mapping[str, str],
) -> Mapping[str, Any]:
    observation = adapter.get_observation_snapshot()
    submitted = adapter.submit_read_only_skill_request(
        _request_payload(
            composition=composition,
            client_id=client_id,
            client_sequence=client_sequence,
            observation=observation,
            skill_id=skill_id,
            parameters=parameters,
        )
    )
    operation = submitted.get("operation")
    if not submitted.get("accepted") or not isinstance(operation, Mapping):
        raise RuntimeError(
            f"perception Skill admission failed: {submitted.get('reason_code')}"
        )
    manifest = composition.contract_bundle.route_manifest
    return _wait_for_terminal_operation(
        adapter=adapter,
        client_id=client_id,
        runtime_session_id=manifest.runtime_session_id,
        manifest_hash=manifest.semantic_content_hash,
        operation_id=str(operation["operation_id"]),
        timeout_sec=timeout_sec,
        executor_errors=executor_errors,
    )


def _download_artifact(
    *,
    adapter: Any,
    client_id: str,
    runtime_session_id: str,
    artifact: Mapping[str, Any],
) -> bytes:
    grant = adapter.begin_artifact_download(
        {
            "client_id": client_id,
            "runtime_session_id": runtime_session_id,
            "artifact_id": artifact["artifact_id"],
            "mechanism": "chunked_rpc",
            "ttl_sec": 10.0,
        }
    )
    chunks = []
    offset = 0
    expected_length = int(artifact["byte_length"])
    while offset < expected_length:
        chunk = adapter.read_artifact_chunk(
            {
                "client_id": client_id,
                "runtime_session_id": runtime_session_id,
                "grant_id": grant["grant_id"],
                "offset": offset,
                "max_bytes": min(_ARTIFACT_CHUNK_BYTES, expected_length - offset),
            }
        )
        data = base64.b64decode(chunk["data_base64"], validate=True)
        if int(chunk["offset"]) != offset or int(chunk["byte_length"]) != len(data):
            raise RuntimeError("Artifact chunk boundary mismatch.")
        if not data:
            raise RuntimeError("Artifact transfer ended before its declared length.")
        chunks.append(data)
        offset += len(data)
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_length or digest != artifact["sha256"]:
        raise RuntimeError("Artifact bytes differ from the immutable descriptor.")
    adapter.acknowledge_artifact_download(
        {
            "client_id": client_id,
            "runtime_session_id": runtime_session_id,
            "grant_id": grant["grant_id"],
            "transferred_bytes": len(payload),
            "verified_sha256": digest,
        }
    )
    return payload


def _validate_rgb_operation(
    *,
    adapter: Any,
    client_id: str,
    runtime_session_id: str,
    operation: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    result = operation["execution_result"]
    metadata = {int(item["ordinal"]): item for item in result["frames"]}
    validated = []
    for binding in operation["output_artifact_bindings"]:
        artifact = binding["artifact"]
        ordinal = int(binding["ordinal"])
        if binding["slot_id"] != "rgb_frames" or ordinal not in metadata:
            raise RuntimeError("RGB Artifact binding does not match its result frame.")
        expected_contract = (
            RGB_FRAME_MEDIA_TYPE_V1,
            RGB_FRAME_SCHEMA_ID_V1,
            RGB_FRAME_ENCODING_V1,
            RGB_FRAME_COMPRESSION_V1,
        )
        actual_contract = tuple(
            artifact[field]
            for field in ("media_type", "schema_id", "encoding", "compression")
        )
        if actual_contract != expected_contract:
            raise RuntimeError("RGB Artifact descriptor changed.")
        payload = _download_artifact(
            adapter=adapter,
            client_id=client_id,
            runtime_session_id=runtime_session_id,
            artifact=artifact,
        )
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.format != "JPEG" or image.mode != "RGB":
                raise RuntimeError("RGB Artifact is not a decoded RGB JPEG.")
            pixels = np.asarray(image)
        frame = metadata[ordinal]
        if (pixels.shape[1], pixels.shape[0]) != (
            int(frame["width"]),
            int(frame["height"]),
        ):
            raise RuntimeError("RGB Artifact dimensions differ from the Skill result.")
        validated.append(
            {
                "ordinal": ordinal,
                "role": frame["role"],
                "source_id": frame["source_id"],
                "width": int(frame["width"]),
                "height": int(frame["height"]),
                "byte_length": len(payload),
                "sha256": artifact["sha256"],
                "pixel_min": int(pixels.min()),
                "pixel_max": int(pixels.max()),
            }
        )
    if len(validated) != len(metadata):
        raise RuntimeError("RGB Artifact count differs from the requested role count.")
    return sorted(validated, key=lambda item: int(item["ordinal"]))


def _validate_depth_operation(
    *,
    adapter: Any,
    client_id: str,
    runtime_session_id: str,
    operation: Mapping[str, Any],
) -> Mapping[str, Any]:
    bindings = operation["output_artifact_bindings"]
    if len(bindings) != 1 or bindings[0]["slot_id"] != "depth_frame":
        raise RuntimeError("Depth Skill did not return exactly one depth Artifact.")
    artifact = bindings[0]["artifact"]
    expected_contract = (
        DEPTH_FRAME_MEDIA_TYPE_V1,
        DEPTH_FRAME_SCHEMA_ID_V1,
        DEPTH_FRAME_ENCODING_V1,
        DEPTH_FRAME_COMPRESSION_V1,
    )
    actual_contract = tuple(
        artifact[field]
        for field in ("media_type", "schema_id", "encoding", "compression")
    )
    if actual_contract != expected_contract:
        raise RuntimeError("Depth Artifact descriptor changed.")
    payload = _download_artifact(
        adapter=adapter,
        client_id=client_id,
        runtime_session_id=runtime_session_id,
        artifact=artifact,
    )
    depth = np.load(io.BytesIO(payload), allow_pickle=False)
    result = operation["execution_result"]
    if depth.dtype != np.float32 or depth.ndim != 2:
        raise RuntimeError("Depth Artifact is not a 2-D float32 NPY array.")
    if depth.shape != (int(result["height"]), int(result["width"])):
        raise RuntimeError("Depth Artifact dimensions differ from the Skill result.")
    if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
        raise RuntimeError("Depth Artifact contains non-finite or negative meters.")
    return {
        "role": result["role"],
        "source_id": result["source_id"],
        "width": int(result["width"]),
        "height": int(result["height"]),
        "byte_length": len(payload),
        "sha256": artifact["sha256"],
        "depth_min_m": float(depth.min()),
        "depth_max_m": float(depth.max()),
        "station_revision": int(result["station_revision"]),
        "reference_frame_id": result["reference_frame_id"],
        "reference_frame_spec_hash": result["reference_frame_spec_hash"],
        "frame_graph_hash": result["frame_graph_hash"],
        "camera_pose_reference_xyz_quat_xyzw": list(
            result["camera_pose_reference_xyz_quat_xyzw"]
        ),
    }


def run_probe(
    *,
    run_root: Path,
    client_id: str,
    operation_timeout_sec: float,
    socket_path: Path | None = None,
) -> Mapping[str, Any]:
    root = _prepare_run_root(run_root)
    timeout_sec = float(operation_timeout_sec)
    if not np.isfinite(timeout_sec) or timeout_sec <= 0.0:
        raise ValueError("--operation-timeout-sec must be positive and finite.")
    endpoint = None if socket_path is None else Path(socket_path)
    owned_endpoint_dirs: tuple[Path, Path] | None = None
    world_from_base, frame_binding = _scene_geometry()
    source_binding = _build_source_binding(
        camera_config_path=_CAMERA_CONFIG,
        camera_registry_path=_CAMERA_REGISTRY,
        scene_world_from_arm_base_by_side=world_from_base,
    )
    composition = None
    result = None
    started_at = time.time()
    try:
        if endpoint is None:
            endpoint, endpoint_dir, endpoint_root = _allocate_default_socket_path()
            owned_endpoint_dirs = (endpoint_dir, endpoint_root)
        client_bindings = compile_agent_gateway_unix_client_bindings(
            primary_client_id=client_id,
            primary_client_uid=os.getuid(),
        )
        artifact_root = root / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        composition = _assemble_composition(
            source_binding=source_binding,
            socket_path=endpoint,
            artifact_root=artifact_root,
            client_bindings=client_bindings,
            runtime_session_id="real-perception-probe:" + uuid.uuid4().hex,
            control_domain_id="real-perception-probe",
            frame_binding=frame_binding,
        )
        _assert_no_motion_surface(composition)
        executor_errors = _capture_executor_errors(composition)
        composition.application.start()
        if not endpoint.is_socket():
            raise RuntimeError("camera-only Gateway socket was not published.")
        adapter = composition.session.wire_adapter(authenticated_client_id=client_id)
        manifest = composition.contract_bundle.route_manifest
        roles = list(composition.registration.profile.rgb_roles)
        rgb_operation = _run_skill(
            composition=composition,
            adapter=adapter,
            client_id=client_id,
            client_sequence=1,
            skill_id=_RGB_SKILL_ID,
            parameters={
                "selected_roles": roles,
                "width": 320,
                "height": 240,
                "max_total_bytes": 16 * 1024 * 1024,
            },
            timeout_sec=timeout_sec,
            executor_errors=executor_errors,
        )
        depth_operation = _run_skill(
            composition=composition,
            adapter=adapter,
            client_id=client_id,
            client_sequence=2,
            skill_id=_DEPTH_SKILL_ID,
            parameters={"role": "main"},
            timeout_sec=timeout_sec,
            executor_errors=executor_errors,
        )
        result = {
            "schema": "agentic_robot.real_perception_skill_probe.v1",
            "status": "passed",
            "profile": "camera-only-real-perception",
            "robot_backend": "absent",
            "control_effects": "forbidden",
            "hardware_reset": "forbidden",
            "wire_path": "bound_adapter_with_published_unix_server",
            "runtime_session_id": manifest.runtime_session_id,
            "route_manifest_hash": manifest.semantic_content_hash,
            "deployment_profile_hash": composition.deployment_profile_hash,
            "rgb": _validate_rgb_operation(
                adapter=adapter,
                client_id=client_id,
                runtime_session_id=manifest.runtime_session_id,
                operation=rgb_operation,
            ),
            "depth": _validate_depth_operation(
                adapter=adapter,
                client_id=client_id,
                runtime_session_id=manifest.runtime_session_id,
                operation=depth_operation,
            ),
            "camera_diagnostics": {
                source_id: {
                    key: info.get(key)
                    for key in (
                        "running",
                        "lifecycle_state",
                        "serial_number",
                        "generation",
                        "frame_count",
                        "timeout_count",
                        "black_frame_count",
                        "consecutive_black_frame_count",
                        "restart_count",
                        "last_error",
                        "last_runtime_error",
                    )
                }
                for source_id, source in source_binding.camera_sources.items()
                for info in (source.device_info(),)
            },
            "elapsed_sec": float(time.time() - started_at),
        }
    finally:
        try:
            if composition is not None:
                composition.application.close()
            else:
                close_errors = []
                for source_id, source in source_binding.camera_sources.items():
                    try:
                        source.close()
                    except Exception as error:  # noqa: BLE001 - close every source.
                        close_errors.append(
                            f"{source_id}:{type(error).__name__}:{error}"
                        )
                if close_errors:
                    raise RuntimeError(
                        "unassembled perception sources did not close: "
                        + ";".join(close_errors)
                    )
        finally:
            if endpoint is not None and endpoint.exists():
                endpoint.unlink()
            if owned_endpoint_dirs is not None:
                endpoint_dir, endpoint_root = owned_endpoint_dirs
                endpoint_dir.rmdir()
                endpoint_root.rmdir()
    assert result is not None
    if composition.application.state.value != "closed":
        raise RuntimeError("camera-only perception application did not close cleanly.")
    result = dict(result)
    result["application_state"] = composition.application.state.value
    evidence_path = root / "real_perception_skill_probe.json"
    result["evidence_path"] = str(evidence_path)
    evidence_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_probe(
            run_root=args.run_root,
            client_id=args.client_id,
            operation_timeout_sec=args.operation_timeout_sec,
        )
    except Exception as error:  # noqa: BLE001 - emit one bounded probe diagnostic.
        print(
            json.dumps(
                {
                    "schema": "agentic_robot.real_perception_skill_probe.v1",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "robot_backend": "absent",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_probe"]
