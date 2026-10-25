"""Same-grid proposals on a live v2 cut or an authenticated historical v3 JPEG."""

from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from agent_gateway.profiles.segmentation import (
    SEGMENT_MASK_OUTPUT_SLOT_ID_V1,
    SEGMENT_OVERLAY_OUTPUT_SLOT_ID_V1,
    SEGMENT_RGB_INPUT_SLOT_ID_V3,
    SEGMENT_MAX_RGB_BYTES_V3,
    LiveRgbSegmentationCapture,
    LiveSegmentationProvider,
    SegmentationRegistration,
    decode_segmentation_mask_npy,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.artifact_validation import validate_skill_artifact_bindings
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from agent_gateway.skills.perception.segmentation_schemas import (
    SEGMENT_GEOMETRY_MAX_HEIGHT_V3,
    SEGMENT_GEOMETRY_MAX_WIDTH_V3,
    SEGMENT_GEOMETRY_METRIC_V3,
    SEGMENT_RGB_PROVENANCE_FIELDS_V3,
)
from agent_gateway.skills.perception.definition import CAPTURE_RGB_SKILL_ID_V1
from dataset_contracts.perception_artifacts import (
    SEGMENTATION_MASK_COMPRESSION_V1,
    SEGMENTATION_MASK_ENCODING_V1,
    SEGMENTATION_MASK_MEDIA_TYPE_V1,
    SEGMENTATION_MASK_SCHEMA_ID_V1,
    SEGMENTATION_OVERLAY_COMPRESSION_V1,
    SEGMENTATION_OVERLAY_ENCODING_V1,
    SEGMENTATION_OVERLAY_MEDIA_TYPE_V1,
    SEGMENTATION_OVERLAY_SCHEMA_ID_V1,
)

_MASK_FIELDS = (
    "score",
    "label",
    "coordinate_format",
    "mask_bounds_xyxy",
    "mask_pixel_count",
    "mask_media_content_hash",
)
_COLORS = ((255, 85, 60), (40, 220, 110), (70, 150, 255))


def _check_pair(
    registration: SegmentationRegistration, provider: LiveSegmentationProvider
) -> None:
    if not isinstance(registration, SegmentationRegistration) or not isinstance(
        provider, LiveSegmentationProvider
    ):
        raise TypeError(
            "Batch segmentation requires the shared typed registration and provider."
        )
    if (
        registration.skill.definition.skill_version not in {"2", "3"}
        or provider.registration.skill.definition.skill_version != "1"
    ):
        raise ValueError(
            "Batch segmentation requires a v2/v3 registration and the original v1 provider."
        )
    original = provider.registration
    if (
        replace(registration.profile, profile_id=original.profile.profile_id,
                rgb_capture_route_hash=None)
        != original.profile
    ):
        raise ValueError(
            "Batch segmentation model, sources or byte bounds differ from its shared provider."
        )
    for key in (
        "runtime_session_id",
        "control_domain_id",
        "target_binding_id",
        "station_manifest_hash",
        "execution_manifest_hash",
        "target_kind",
    ):
        if getattr(registration.binding, key) != getattr(original.binding, key):
            raise ValueError(
                "Batch segmentation deployment differs from its shared provider."
            )


@dataclass(frozen=True)
class PreparedSegmentationBatch:
    execution_result: dict[str, Any]
    masks_npy: tuple[bytes, ...]
    overlay_jpeg: bytes


def resolve_segmentation_rgb_input(
    *, request: Any, registration: SegmentationRegistration,
    rgb_capture_registration: SkillRegistrationTemplate,
    rgb_capture_validator: AgentSkillPayloadValidatorSet,
    artifact_store: RecoverableAgentArtifactStore,
    lookup_operation: Callable, check_active: Callable[[], None],
) -> tuple[LiveRgbSegmentationCapture, dict[str, Any]]:
    """Authenticate a retained producer, then decode its exact managed JPEG bytes.

    The service lookup returns (request, operation), or None after eviction.
    """
    if registration.skill.definition.skill_version != "3":
        raise ValueError("Only geometry v3 consumes an authenticated RGB input.")
    registration.payload_validator.validate_parameters(request.parameters)
    validate_skill_artifact_bindings(io_contract=registration.skill.io_contract,
        direction="input", bindings=request.input_artifact_bindings)
    binding = request.input_artifact_bindings[0]
    if binding.slot_id != SEGMENT_RGB_INPUT_SLOT_ID_V3 or binding.ordinal != 0:
        raise ValueError("Exactly one input_image at ordinal zero is required.")
    artifact = binding.artifact
    check_active()
    operation_id = request.parameters["rgb_capture_operation_id"]
    retained = lookup_operation(operation_id)
    if retained is None:
        raise ValueError("RGB producer operation is no longer retained.")
    original, source = retained
    route = rgb_capture_registration.routes[0]
    if (source is None or original is None or source.operation_id != operation_id
            or source.status != "completed" or source.execution_outcome != "completed"
            or source.execution_result is None
            or original.skill_id != CAPTURE_RGB_SKILL_ID_V1
            or original.skill_version != rgb_capture_registration.definition.skill_version
            or original.skill_definition_hash != rgb_capture_registration.definition.semantic_content_hash
            or original.route_id != route.route_id or original.route_version != route.route_version
            or original.route_hash != route.semantic_content_hash
            or original.client_id != request.client_id
            or original.runtime_session_id != request.runtime_session_id
            or source.runtime_session_id != request.runtime_session_id
            or original.skill_route_manifest_hash != request.skill_route_manifest_hash
            or source.request_hash != original.semantic_content_hash
            or original.expected_station_revision != request.expected_station_revision):
        raise ValueError("RGB Artifact is not a completed authenticated capture of this client/session/revision.")
    rgb_capture_validator.validate_parameters(original.parameters)
    rgb_capture_validator.validate_execution_result(source.execution_result)
    rgb_capture_validator.validate_execution_result_for_parameters(original.parameters, source.execution_result)
    result = source.execution_result
    frames = [frame for frame in result["frames"] if frame["role"] == request.parameters["role"]]
    if len(frames) != 1:
        raise ValueError("RGB producer does not contain exactly one requested camera role.")
    frame = frames[0]
    if (type(frame["ordinal"]) is not int or not 0 <= frame["ordinal"] <= 7
            or result["station_revision"] != request.expected_station_revision
            or registration.profile.source_id_by_role.get(frame["role"]) != frame["source_id"]
            or frame["media_content_hash"] != artifact.sha256
            or not any(item.slot_id == "rgb_frames" and item.ordinal == frame["ordinal"]
                and item.artifact == artifact for item in source.output_artifact_bindings)):
        raise ValueError("RGB input differs from its exact producer frame/role/output Artifact.")
    record = artifact_store.artifact(artifact.artifact_id)
    if not record.available or record.artifact != artifact:
        raise ValueError("RGB input Artifact is unavailable or differs from the managed descriptor.")
    check_active()
    data = artifact_store.read_bytes(artifact, maximum=SEGMENT_MAX_RGB_BYTES_V3)
    if len(data) != artifact.byte_length or hashlib.sha256(data).hexdigest() != artifact.sha256:
        raise ValueError("RGB input bytes differ from their exact bounded content identity.")
    check_active()
    try:
        with Image.open(io.BytesIO(data)) as image:
            if (image.format != "JPEG" or image.mode != "RGB"
                    or image.size != (frame["width"], frame["height"])
                    or image.width > SEGMENT_GEOMETRY_MAX_WIDTH_V3
                    or image.height > SEGMENT_GEOMETRY_MAX_HEIGHT_V3):
                raise ValueError("RGB JPEG grid/encoding differs from its producer or reviewed bounds.")
            orientation = image.getexif().get(274)
            if orientation is not None and (type(orientation) is not int or orientation != 1):
                raise ValueError("RGB JPEG EXIF orientation would change the original pixel grid.")
            image.load()
            check_active()
            capture = LiveRgbSegmentationCapture(
                source_id=frame["source_id"], role=frame["role"],
                captured_at=result["captured_at"] if original.skill_version == "1" else frame["acquired_at"],
                station_revision=result["station_revision"], image=image, rgb_content_hash=artifact.sha256,
            )
    except OSError as error:
        raise ValueError("RGB input is not a valid complete JPEG.") from error
    check_active()
    return capture, {
        "rgb_capture_operation_id": operation_id, "rgb_capture_request_hash": source.request_hash,
        "input_rgb_artifact_id": artifact.artifact_id, "input_rgb_frame_ordinal": frame["ordinal"],
    }


def _mask_relations(
    masks: list[np.ndarray | None], *, check_active: Callable[[], None],
) -> list[dict[str, Any]]:
    """Measure canonical pairs only after their provider bytes were validated."""
    relations = []
    for one in range(len(masks)):
        for two in range(one + 1, len(masks)):
            check_active()
            first, second = masks[one], masks[two]
            relation = {
                "indices": [one, two],
                "metric": SEGMENT_GEOMETRY_METRIC_V3,
                "status": "unavailable",
                "unavailable_reason": "absent_mask",
                "distance_px": None,
                "closest_pixels_uv": None,
                "semantic_identity_verified": False,
                "physical_clearance_m": None,
            }
            if first is not None and second is not None:
                # Import only for an opted-in call with an available mask pair.
                from scipy.ndimage import distance_transform_edt

                check_active()
                distances = distance_transform_edt(~second)
                check_active()
                points = np.flatnonzero(first)
                chosen = int(points[np.argmin(distances.ravel()[points])])
                row, col = divmod(chosen, first.shape[1])
                del distances, points
                # Resolve equal-distance endpoints in row-major order explicitly,
                # rather than inheriting a library-specific feature-transform tie.
                rows, cols = np.nonzero(second)
                squared = (rows - row) ** 2 + (cols - col) ** 2
                closest = int(np.argmin(squared))
                other_row, other_col = int(rows[closest]), int(cols[closest])
                distance = float(np.sqrt(squared[closest]))
                relation.update(
                    status="overlap" if distance == 0 else "separated",
                    unavailable_reason=None,
                    distance_px=distance,
                    closest_pixels_uv=[[col, row], [other_col, other_row]],
                )
                del rows, cols, squared
            check_active()
            relations.append(relation)
    return relations


def _indexed_contour_overlay(
    image: Image.Image, masks: list[np.ndarray | None], *,
    check_active: Callable[[], None],
) -> Image.Image:
    """Annotate validated masks without tinting their interiors or empty boxes."""
    check_active()
    pixels = np.asarray(image).copy()
    labels: dict[tuple[int, int], list[int]] = {}
    for index, mask in enumerate(masks):
        check_active()
        if mask is None:
            continue
        height, width = mask.shape
        padded = np.pad(mask, 1, constant_values=False)
        interior = mask.copy()
        for dy in range(3):
            check_active()
            for dx in range(3):
                interior &= padded[dy:dy + height, dx:dx + width]
        check_active()
        pixels[mask & ~interior] = _COLORS[index]
        # Anchor on an occupied pixel, not an often-empty bounding-box corner.
        row, col = divmod(int(np.argmax(mask)), width)
        labels.setdefault((col, row), []).append(index)
    overlay = Image.fromarray(pixels)
    draw = ImageDraw.Draw(overlay)
    for (col, row), indices in labels.items():
        check_active()
        # Shared anchors do not merge targets: display their indices side by side.
        bounds = [draw.textbbox((0, 0), str(index)) for index in indices]
        total_width = sum(right - left for left, _, right, _ in bounds) + 2 * (len(indices) - 1)
        height = max(bottom - top for _, top, _, bottom in bounds)
        x = max(0, min(col, overlay.width - total_width))
        y = max(0, min(row, overlay.height - height))
        for index, (left, top, right, _) in zip(indices, bounds, strict=True):
            check_active()
            draw.text((x - left, y - top), str(index), fill=_COLORS[index])
            x += right - left + 2
    check_active()
    return overlay


def prepare_segmentation_batch(
    *,
    registration: SegmentationRegistration,
    provider: LiveSegmentationProvider,
    capture: LiveRgbSegmentationCapture,
    parameters: Mapping[str, Any],
    check_active: Callable[[], None],
    rgb_provenance: Mapping[str, Any] | None = None,
) -> PreparedSegmentationBatch:
    """Prepare ordered proposals from one cut; no store, device or control effects."""
    _check_pair(registration, provider)
    registration.payload_validator.validate_parameters(parameters)
    if not isinstance(capture, LiveRgbSegmentationCapture):
        raise TypeError("Batch capture must be typed.")
    if (
        capture.role != parameters["role"]
        or registration.profile.source_id_by_role.get(capture.role) != capture.source_id
    ):
        raise ValueError("Batch capture changed its registered camera identity.")
    geometry = registration.skill.definition.skill_version == "3"
    if geometry and (not isinstance(rgb_provenance, Mapping)
            or set(rgb_provenance) != set(SEGMENT_RGB_PROVENANCE_FIELDS_V3)
            or rgb_provenance["rgb_capture_operation_id"] != parameters["rgb_capture_operation_id"]):
        raise ValueError("Geometry requires the authenticated RGB producer provenance.")
    if geometry and (
        capture.image.width > SEGMENT_GEOMETRY_MAX_WIDTH_V3
        or capture.image.height > SEGMENT_GEOMETRY_MAX_HEIGHT_V3
    ):
        raise ValueError("Geometry mask grid exceeds its reviewed bound.")
    check_active()
    for target in parameters["targets"]:
        pixel = target["target_pixel_uv"]
        if pixel is not None and not (
            pixel[0] < capture.image.width and pixel[1] < capture.image.height
        ):
            raise ValueError("Batch point prompt is outside the image.")
    masks = []
    decoded_masks = []
    items = []
    pixels = None if geometry else np.asarray(capture.image).copy()
    for index, target in enumerate(parameters["targets"]):
        check_active()
        # Each invocation gets a copy of the same decoded cut, not a new camera frame.
        prepared = provider.try_segment_capture(
            capture=replace(capture, image=capture.image.copy()),
            parameters={"role": capture.role, **target},
        )
        check_active()
        item = {
            "index": index,
            **dict(target),
            "status": "no_mask",
            "mask_artifact_ordinal": None,
            "mask": None,
        }
        if prepared is not None:
            result = prepared.execution_result
            provider.registration.payload_validator.validate_execution_result(result)
            provider.registration.payload_validator.validate_execution_result_for_parameters(
                {"role": capture.role, **target}, result,
            )
            expected_capture = {
                "source_id": capture.source_id, "role": capture.role,
                "captured_at": float(capture.captured_at),
                "station_revision": capture.station_revision,
                "input_rgb_content_hash": capture.rgb_content_hash,
                "image_width": capture.image.width, "image_height": capture.image.height,
            }
            if any(result[key] != value for key, value in expected_capture.items()):
                raise RuntimeError("Batch provider result changed its same-cut capture identity.")
            if (len(prepared.mask_npy) > registration.profile.max_mask_bytes
                    or hashlib.sha256(prepared.mask_npy).hexdigest() != result["mask_media_content_hash"]):
                raise RuntimeError("Batch provider mask bytes differ from their bounded content hash.")
            check_active()
            mask = decode_segmentation_mask_npy(
                prepared.mask_npy, width=capture.image.width, height=capture.image.height,
                expected_count=result["mask_pixel_count"],
                expected_bounds=tuple(result["mask_bounds_xyxy"]),
            )
            check_active()
            item.update(
                status="completed",
                mask_artifact_ordinal=len(masks),
                mask={key: prepared.execution_result[key] for key in _MASK_FIELDS},
            )
            masks.append(prepared.mask_npy)
            if not geometry:
                color = np.asarray(_COLORS[index], dtype=np.uint16)
                pixels[mask] = (
                    (3 * pixels[mask].astype(np.uint16) + 2 * color) // 5
                ).astype(np.uint8)
            decoded_masks.append(mask)
        else:
            decoded_masks.append(None)
        items.append(item)
    if sum(map(len, masks)) > registration.profile.max_mask_bytes:
        raise RuntimeError("Batch masks exceed the aggregate deployment byte bound.")
    relations = _mask_relations(decoded_masks, check_active=check_active) if geometry else None
    check_active()
    if geometry:
        overlay = _indexed_contour_overlay(
            capture.image, decoded_masks, check_active=check_active,
        )
    else:
        overlay = capture.image.copy()
        overlay.frombytes(pixels.tobytes())
        draw = ImageDraw.Draw(overlay)
        for item in items:
            color = _COLORS[item["index"]]
            if item["mask"] is not None:
                x0, y0, x1, y1 = item["mask"]["mask_bounds_xyxy"]
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=2)
                draw.text(
                    (x0, min(overlay.height - 1, y0 + 12 * item["index"])),
                    str(item["index"]),
                    fill=color,
                )
            y = 12 * item["index"]
            draw.rectangle((0, y, min(overlay.width - 1, 150), y + 11), fill="black")
            draw.text((1, y), f"[{item['index']}] {item['status']}", fill=color)
    stream = io.BytesIO()
    overlay.save(stream, format="JPEG", quality=90)
    overlay_jpeg = stream.getvalue()
    check_active()
    if len(overlay_jpeg) > registration.profile.max_overlay_bytes:
        raise RuntimeError("Batch overlay exceeds its deployment byte bound.")
    body = {
        "source_id": capture.source_id,
        "role": capture.role,
        "captured_at": float(capture.captured_at),
        "station_revision": capture.station_revision,
        "input_rgb_content_hash": capture.rgb_content_hash,
        "image_width": capture.image.width,
        "image_height": capture.image.height,
        "model_profile_id": registration.profile.model_profile_id,
        "model_profile_hash": registration.profile.model_profile_hash,
        "overlay_media_content_hash": hashlib.sha256(overlay_jpeg).hexdigest(),
        "completed_count": len(masks),
        "no_mask_count": len(items) - len(masks),
        "items": items,
    }
    if geometry:
        body.update(rgb_provenance)
        body["relations"] = relations
    registration.payload_validator.validate_execution_result(body)
    registration.payload_validator.validate_execution_result_for_parameters(
        parameters, body
    )
    check_active()
    return PreparedSegmentationBatch(body, tuple(masks), overlay_jpeg)


def build_segmentation_batch_executor(
    *,
    registration: SegmentationRegistration,
    provider: LiveSegmentationProvider,
    artifact_store: RecoverableAgentArtifactStore,
    runtime_session_id: str,
    current_station_revision: Callable[[], int],
    clock: Callable[[], float] = time.monotonic,
    max_retained_operations: int = 32,
    rgb_capture_registration: SkillRegistrationTemplate | None = None,
    rgb_capture_validator: AgentSkillPayloadValidatorSet | None = None,
    lookup_operation: Callable | None = None,
) -> PassiveReadSkillExecutor:
    """Add a passive Route without creating another model/capture lifecycle."""
    _check_pair(registration, provider)
    if not isinstance(artifact_store, RecoverableAgentArtifactStore):
        raise TypeError("artifact_store must be typed.")
    if runtime_session_id != registration.binding.runtime_session_id:
        raise ValueError("Batch executor runtime session differs from its binding.")
    version = registration.skill.definition.skill_version
    if version == "3":
        if (not isinstance(rgb_capture_registration, SkillRegistrationTemplate)
                or not isinstance(rgb_capture_validator, AgentSkillPayloadValidatorSet)
                or not callable(lookup_operation)
                or rgb_capture_registration.definition.skill_id != CAPTURE_RGB_SKILL_ID_V1
                or rgb_capture_registration.definition.skill_version not in {"1", "2", "3"}
                or len(rgb_capture_registration.routes) != 1):
            raise ValueError("Geometry requires an exact RGB registration, validator and operation lookup.")
        rgb_capture_validator.validate_route(rgb_capture_registration.routes[0])
        producer_hash = rgb_capture_registration.routes[0].semantic_content_hash
        bound_hash = registration.profile.rgb_capture_route_hash
        if (bound_hash is not None and bound_hash != producer_hash
                or rgb_capture_registration.definition.skill_version == "3"
                and bound_hash != producer_hash):
            raise ValueError("Geometry RGB v3 producer differs from the published exact Route binding.")
        if max_retained_operations < 2:
            raise ValueError("Geometry requires retained RGB producer and consumer operations.")

    def read(request: Any, operation: Any) -> PassiveReadResult:
        registration.payload_validator.validate_parameters(request.parameters)

        def check_active() -> None:
            if (
                executor.cancel_requested(operation.operation_id)
                or float(clock()) >= operation.deadline_at
            ):
                raise RuntimeError("Batch segmentation cancelled or expired.")
            if current_station_revision() != request.expected_station_revision:
                raise RuntimeError(
                    "Station revision changed during batch segmentation."
                )

        check_active()
        rgb_provenance = None
        if version == "3":
            capture, rgb_provenance = resolve_segmentation_rgb_input(
                request=request, registration=registration, rgb_capture_registration=rgb_capture_registration,
                rgb_capture_validator=rgb_capture_validator, artifact_store=artifact_store,
                lookup_operation=lookup_operation, check_active=check_active,
            )
        else:
            capture = provider.capture_port.capture_segment_rgb(
                role=request.parameters["role"],
                expected_station_revision=request.expected_station_revision,
            )
        if capture.station_revision != request.expected_station_revision:
            raise RuntimeError("Batch capture returned another Station revision.")
        check_active()
        prepared = prepare_segmentation_batch(
            registration=registration,
            provider=provider,
            capture=capture,
            parameters=request.parameters,
            check_active=check_active,
            rgb_provenance=rgb_provenance,
        )
        check_active()
        artifacts = [
            (
                SEGMENT_MASK_OUTPUT_SLOT_ID_V1,
                ordinal,
                data,
                SEGMENTATION_MASK_MEDIA_TYPE_V1,
                SEGMENTATION_MASK_SCHEMA_ID_V1,
                SEGMENTATION_MASK_ENCODING_V1,
                SEGMENTATION_MASK_COMPRESSION_V1,
            )
            for ordinal, data in enumerate(prepared.masks_npy)
        ]
        artifacts.append(
            (
                SEGMENT_OVERLAY_OUTPUT_SLOT_ID_V1,
                0,
                prepared.overlay_jpeg,
                SEGMENTATION_OVERLAY_MEDIA_TYPE_V1,
                SEGMENTATION_OVERLAY_SCHEMA_ID_V1,
                SEGMENTATION_OVERLAY_ENCODING_V1,
                SEGMENTATION_OVERLAY_COMPRESSION_V1,
            )
        )
        bindings = []
        for slot_id, ordinal, data, media, schema, encoding, compression in artifacts:
            check_active()
            artifact = artifact_ref_for_bytes(
                data,
                media_type=media,
                schema_id=schema,
                encoding=encoding,
                compression=compression,
            )
            committed = commit_artifact_bytes(
                store=artifact_store,
                data=data,
                artifact=artifact,
                producer_id=f"agent_gateway.provider.{registration.profile.target_kind}_segmentation",
                runtime_session_id=runtime_session_id,
                idempotency_key=f"segmentation-v{version}-output:{operation.operation_id}:{slot_id}:{ordinal}",
                now=float(clock()),
            )
            bindings.append(
                AgentArtifactBinding(
                    slot_id=slot_id,
                    ordinal=ordinal,
                    artifact=committed,
                    production_receipt_ref=f"{registration.profile.target_kind}-segmentation:{committed.sha256}",
                )
            )
        check_active()
        receipt = canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "request_hash": request.semantic_content_hash,
                "execution_result": prepared.execution_result,
                "artifact_ref_hashes": [
                    binding.artifact.semantic_content_hash for binding in bindings
                ],
            },
            domain=f"dexteleop.agent_skill.same_cut_segmentation_receipt.v{version}",
        )
        return PassiveReadResult(
            execution_result=prepared.execution_result,
            evidence_refs=("same-cut-segmentation:" + receipt,),
            output_artifact_bindings=tuple(bindings),
            reason_code="passive_same_cut_segmentation_completed",
        )

    executor = PassiveReadSkillExecutor(
        port_id=f"agent-gateway:{registration.profile.target_kind}-passive-segmentation-v{version}:{runtime_session_id}",
        executor_id=registration.binding.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )
    return executor
