"""Supplied-image localization provider and passive Gateway executor."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
    LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    LOCALIZATION_ANNOTATED_IMAGE_SCHEMA_ID_V1,
    LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
    LOCALIZATION_INPUT_IMAGE_SCHEMA_IDS_V1,
    LOCALIZATION_INPUT_JPEG_ENCODING_V1,
    LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
    LOCALIZATION_INPUT_PNG_ENCODING_V1,
    LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1,
)

from .localization_deployment import (
    REAL_LOCALIZE_OBJECTS_INPUT_SLOT_ID_V1,
    REAL_LOCALIZE_OBJECTS_OUTPUT_SLOT_ID_V1,
    RealObjectLocalizationRegistration,
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def read_bounded_regular_file(path: Path, *, maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Object-localization input must be a regular file.")
        if info.st_size <= 0 or info.st_size > maximum:
            raise ValueError("Object-localization input file exceeds its byte bound.")
        chunks = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise OSError("Object-localization input file changed while reading.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.fstat(descriptor).st_size != info.st_size:
            raise OSError("Object-localization input file size changed while reading.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def safe_local_path(
    raw_path: object, roots: tuple[Path, ...], *, field_name: str = "local_image_path"
) -> Path:
    text = _required(raw_path, field_name=field_name)
    candidate = Path(text)
    if not candidate.is_absolute():
        raise ValueError("local_image_path must be absolute.")
    resolved = candidate.resolve(strict=True)
    if not roots:
        raise PermissionError("Local-path test input is disabled.")
    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise PermissionError("local_image_path is outside the configured test roots.")


def decode_bounded_image(
    data: bytes,
    *,
    expected_media_type: str | None,
    maximum_pixels: int,
) -> Image.Image:
    try:
        source = Image.open(io.BytesIO(data))
        source_format = source.format
        width, height = source.size
        if width <= 0 or height <= 0 or width * height > maximum_pixels:
            raise ValueError("Object-localization image dimensions exceed the bound.")
        expected_format = {
            LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1: "JPEG",
            LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1: "PNG",
        }.get(expected_media_type)
        if expected_format is not None and source_format != expected_format:
            raise ValueError(
                "Input Artifact bytes do not match their declared media type."
            )
        if source_format not in {"JPEG", "PNG"}:
            raise ValueError("Object localization accepts only JPEG or PNG images.")
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
        return image
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(
            "Object-localization input is not a valid bounded image."
        ) from exc


def _annotated_jpeg(image: Image.Image, detections: tuple[Any, ...]) -> bytes:
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)
    for ordinal, detection in enumerate(detections):
        box = tuple(int(value) for value in detection.box_xyxy)
        draw.rectangle(
            (box[0], box[1], box[2] - 1, box[3] - 1),
            outline=(255, 32, 32),
            width=3,
        )
        label = detection.label or "object"
        text = f"{ordinal}: {label} {float(detection.score):.3f}"
        text_box = draw.textbbox((box[0], box[1]), text)
        text_height = max(1, text_box[3] - text_box[1])
        text_top = max(0, box[1] - text_height - 4)
        draw.rectangle(
            (box[0], text_top, min(output_image.width - 1, text_box[2] + 4), box[1]),
            fill=(255, 32, 32),
        )
        draw.text((box[0] + 2, text_top + 1), text, fill=(255, 255, 255))
    output = io.BytesIO()
    output_image.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()


@dataclass(frozen=True)
class PreparedObjectLocalization:
    execution_result: dict[str, Any]
    annotated_jpeg: bytes


class ObjectLocalizationProvider:
    """Resolve one bounded input and invoke one lifecycle-owned localizer."""

    def __init__(
        self,
        *,
        localizer: object,
        registration: RealObjectLocalizationRegistration,
        allowed_local_image_roots: tuple[str | Path, ...] = (),
        max_image_pixels: int = 16_777_216,
    ) -> None:
        if not isinstance(registration, RealObjectLocalizationRegistration):
            raise TypeError("registration must be typed.")
        for name in ("start", "close", "localize"):
            if not callable(getattr(localizer, name, None)):
                raise TypeError(f"localizer must expose {name}.")
        implementation_hash = _required(
            getattr(localizer, "semantic_content_hash", None),
            field_name="localizer.semantic_content_hash",
        )
        if implementation_hash != registration.profile.inference_implementation_hash:
            raise ValueError("Localizer identity differs from the deployment profile.")
        roots = tuple(
            sorted(
                (
                    Path(value).expanduser().resolve(strict=True)
                    for value in allowed_local_image_roots
                ),
                key=str,
            )
        )
        if any(not root.is_dir() for root in roots):
            raise ValueError("Every allowed local image root must be a directory.")
        if bool(roots) != registration.profile.local_path_enabled:
            raise ValueError("Local-path roots disagree with the deployment profile.")
        if (
            isinstance(max_image_pixels, bool)
            or not isinstance(max_image_pixels, int)
            or max_image_pixels <= 0
        ):
            raise ValueError("max_image_pixels must be positive.")
        self.localizer = localizer
        self.registration = registration
        self.allowed_local_image_roots = roots
        self.max_image_pixels = max_image_pixels
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def semantic_content_hash(self) -> str:
        return self.registration.profile.inference_implementation_hash

    def start(self) -> None:
        self.localizer.start()
        self._started = True

    def close(self) -> None:
        try:
            self.localizer.close()
        finally:
            self._started = False

    def _input_bytes(
        self, request: Any, store: RecoverableAgentArtifactStore
    ) -> tuple[bytes, str, str]:
        source = request.parameters["input_source"]
        bindings = tuple(request.input_artifact_bindings)
        if source == "artifact":
            if len(bindings) != 1 or request.parameters["local_image_path"] is not None:
                raise ValueError(
                    "Artifact localization requires exactly one input Artifact."
                )
            binding = bindings[0]
            if binding.slot_id != REAL_LOCALIZE_OBJECTS_INPUT_SLOT_ID_V1:
                raise ValueError(
                    "Object-localization input Artifact uses another slot."
                )
            artifact = binding.artifact
            identity = (artifact.media_type, artifact.encoding, artifact.compression)
            if (
                identity
                not in {
                    (
                        LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
                        LOCALIZATION_INPUT_JPEG_ENCODING_V1,
                        LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
                    ),
                    (
                        LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1,
                        LOCALIZATION_INPUT_PNG_ENCODING_V1,
                        LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
                    ),
                }
                or artifact.schema_id not in LOCALIZATION_INPUT_IMAGE_SCHEMA_IDS_V1
            ):
                raise ValueError(
                    "Object-localization input Artifact identity is invalid."
                )
            return (
                store.read_bytes(artifact, maximum=self.registration.profile.max_image_bytes),
                artifact.media_type, artifact.sha256,
            )
        if source == "local_path":
            if bindings:
                raise ValueError(
                    "Local-path localization cannot also bind an Artifact."
                )
            path = safe_local_path(
                request.parameters["local_image_path"],
                self.allowed_local_image_roots,
            )
            data = read_bounded_regular_file(path, maximum=self.registration.profile.max_image_bytes)
            return data, "", hashlib.sha256(data).hexdigest()
        raise ValueError("Object-localization input source is unsupported.")

    def localize_request(
        self,
        request: Any,
        *,
        artifact_store: RecoverableAgentArtifactStore,
    ) -> PreparedObjectLocalization:
        data, declared_media_type, input_hash = self._input_bytes(request, artifact_store)
        image = decode_bounded_image(
            data,
            expected_media_type=declared_media_type or None,
            maximum_pixels=self.max_image_pixels,
        )
        prompt = request.parameters["language_prompt"]
        detections = self.localizer.localize(
            image,
            language_prompt=prompt,
            max_detections=request.parameters["max_detections"],
        )
        normalized = []
        for ordinal, detection in enumerate(detections):
            x_min, y_min, x_max, y_max = detection.box_xyxy
            if not (
                0 <= x_min < x_max <= image.width and 0 <= y_min < y_max <= image.height
            ):
                raise RuntimeError("Object localizer returned an out-of-image box.")
            normalized.append(
                {
                    "ordinal": ordinal,
                    "box_xyxy": [x_min, y_min, x_max, y_max],
                    "score": float(detection.score),
                    "label": detection.label,
                }
            )
        annotated = _annotated_jpeg(image, detections)
        if len(annotated) > self.registration.profile.max_annotated_bytes:
            raise RuntimeError("Annotated object-localization image exceeds its bound.")
        annotated_hash = hashlib.sha256(annotated).hexdigest()
        result = {
            "input_source": request.parameters["input_source"],
            "input_media_content_hash": input_hash,
            "mode": "text" if prompt is not None else "automatic",
            "language_prompt": prompt,
            "image_width": image.width,
            "image_height": image.height,
            "coordinate_format": "xyxy_pixel_0based_half_open",
            "semantic_identity_status": "unverified_proposal",
            "detections": normalized,
            "detection_count": len(normalized),
            "model_profile_id": self.registration.profile.model_profile_id,
            "model_profile_hash": self.registration.profile.model_profile_hash,
            "annotated_image_media_content_hash": annotated_hash,
        }
        return PreparedObjectLocalization(
            execution_result=result,
            annotated_jpeg=annotated,
        )


def build_real_object_localization_executor(
    *,
    registration: RealObjectLocalizationRegistration,
    provider: ObjectLocalizationProvider,
    artifact_store: RecoverableAgentArtifactStore,
    runtime_session_id: str,
    current_station_revision: Any,
    max_retained_operations: int,
    clock: Any = time.time,
) -> PassiveReadSkillExecutor:
    """Bind supplied-image model inference to one passive read executor."""

    if not isinstance(registration, RealObjectLocalizationRegistration):
        raise TypeError("registration must be typed.")
    if not isinstance(provider, ObjectLocalizationProvider):
        raise TypeError("provider must be typed.")
    if not isinstance(artifact_store, RecoverableAgentArtifactStore):
        raise TypeError("artifact_store must be typed.")
    session_id = _required(runtime_session_id, field_name="runtime_session_id")

    def read(request: Any, operation: Any) -> PassiveReadResult:
        prepared = provider.localize_request(request, artifact_store=artifact_store)
        artifact = artifact_ref_for_bytes(
            prepared.annotated_jpeg,
            media_type=LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
            schema_id=LOCALIZATION_ANNOTATED_IMAGE_SCHEMA_ID_V1,
            encoding=LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
            compression=LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
        )
        committed = commit_artifact_bytes(
            store=artifact_store,
            data=prepared.annotated_jpeg,
            artifact=artifact,
            producer_id="agent_gateway.provider.real_object_localization",
            runtime_session_id=session_id,
            idempotency_key="object-localization-output:" + operation.operation_id,
            now=float(clock()),
        )
        if (
            prepared.execution_result["annotated_image_media_content_hash"]
            != committed.sha256
        ):
            raise RuntimeError("Annotated image hash changed during commit.")
        identity = canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "request_hash": request.semantic_content_hash,
                "execution_result": prepared.execution_result,
                "artifact_ref_hash": committed.semantic_content_hash,
            },
            domain="dexteleop.agent_skill.real_object_localization_receipt.v1",
        )
        evidence_ref = "real-object-localization:" + identity
        return PassiveReadResult(
            execution_result=prepared.execution_result,
            evidence_refs=(evidence_ref,),
            output_artifact_bindings=(
                AgentArtifactBinding(
                    slot_id=REAL_LOCALIZE_OBJECTS_OUTPUT_SLOT_ID_V1,
                    ordinal=0,
                    artifact=committed,
                    production_receipt_ref=evidence_ref,
                ),
            ),
            reason_code="passive_object_localization_completed",
        )

    return PassiveReadSkillExecutor(
        port_id="agent-gateway:real-passive-object-localization:" + session_id,
        executor_id=registration.binding.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )


__all__ = [
    "ObjectLocalizationProvider",
    "PreparedObjectLocalization",
    "build_real_object_localization_executor",
    "decode_bounded_image",
    "read_bounded_regular_file",
    "safe_local_path",
]
