"""Shared live-camera segmentation registration, provider, and executor."""

from __future__ import annotations

import ast
import hashlib
import io
import math
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from PIL import Image

from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
)
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOContract,
)
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.definition import (
    segment_skill_definition_v1,
    segment_skill_definition_v2,
    segment_skill_definition_v3,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from agent_gateway.skills.perception.segmentation_schemas import (
    segment_schema_entry_v1,
    segment_schema_entry_v2,
    segment_schema_entry_v3,
    validate_segment_parameters_v1,
    validate_segment_parameters_v2,
    validate_segment_parameters_v3,
    validate_segment_result_for_parameters_v1,
    validate_segment_result_for_parameters_v2,
    validate_segment_result_for_parameters_v3,
    validate_segment_result_v1,
    validate_segment_result_v2,
    validate_segment_result_v3,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_MEDIA_TYPE_V1,
    RGB_FRAME_SCHEMA_ID_V1,
    SEGMENTATION_MASK_COMPRESSION_V1,
    SEGMENTATION_MASK_ENCODING_V1,
    SEGMENTATION_MASK_MEDIA_TYPE_V1,
    SEGMENTATION_MASK_SCHEMA_ID_V1,
    SEGMENTATION_OVERLAY_COMPRESSION_V1,
    SEGMENTATION_OVERLAY_ENCODING_V1,
    SEGMENTATION_OVERLAY_MEDIA_TYPE_V1,
    SEGMENTATION_OVERLAY_SCHEMA_ID_V1,
)

SEGMENT_MASK_OUTPUT_SLOT_ID_V1 = "segmentation_mask"
SEGMENT_OVERLAY_OUTPUT_SLOT_ID_V1 = "segmentation_overlay"
SEGMENT_MAX_MASK_BYTES_V1 = 16 * 1024 * 1024
SEGMENT_MAX_OVERLAY_BYTES_V1 = 16 * 1024 * 1024
SEGMENT_RGB_INPUT_SLOT_ID_V3 = "input_image"
SEGMENT_MAX_RGB_BYTES_V3 = 16 * 1024 * 1024

_SEGMENT_DEFINITIONS = {
    "1": segment_skill_definition_v1,
    "2": segment_skill_definition_v2,
    "3": segment_skill_definition_v3,
}
_SEGMENT_SCHEMAS = {
    "1": segment_schema_entry_v1,
    "2": segment_schema_entry_v2,
    "3": segment_schema_entry_v3,
}
_SEGMENT_RESULT_VALIDATORS = {
    "1": validate_segment_result_v1,
    "2": validate_segment_result_v2,
    "3": validate_segment_result_v3,
}


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _hash(value: object, *, field_name: str) -> str:
    text = _required(value, field_name=field_name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return text


@dataclass(frozen=True)
class LiveRgbSegmentationCapture:
    source_id: str
    role: str
    captured_at: float
    station_revision: int
    image: Image.Image
    rgb_content_hash: str

    def __post_init__(self) -> None:
        _required(self.source_id, field_name="source_id")
        _required(self.role, field_name="role")
        if not math.isfinite(float(self.captured_at)):
            raise ValueError("captured_at must be finite.")
        if (
            isinstance(self.station_revision, bool)
            or not isinstance(self.station_revision, int)
            or self.station_revision < 0
        ):
            raise ValueError("station_revision must be non-negative.")
        if (
            not isinstance(self.image, Image.Image)
            or self.image.width <= 0
            or self.image.height <= 0
        ):
            raise TypeError("image must be one non-empty PIL image.")
        _hash(self.rgb_content_hash, field_name="rgb_content_hash")
        image = self.image.convert("RGB").copy()
        image.load()
        object.__setattr__(self, "image", image)


class LiveRgbSegmentationCapturePort(Protocol):
    def capture_segment_rgb(
        self, *, role: str, expected_station_revision: int
    ) -> LiveRgbSegmentationCapture: ...


@dataclass(frozen=True)
class SegmentationProfile:
    target_kind: str
    current_mode: str
    source_id_by_role: Mapping[str, str]
    model_profile_id: str
    model_profile_hash: str
    inference_implementation_hash: str
    max_mask_bytes: int = SEGMENT_MAX_MASK_BYTES_V1
    max_overlay_bytes: int = SEGMENT_MAX_OVERLAY_BYTES_V1
    profile_id: str = "sam2-grounding-dino-live-segmentation-v1"
    rgb_capture_route_hash: str | None = None

    def __post_init__(self) -> None:
        if self.rgb_capture_route_hash is not None:
            _hash(self.rgb_capture_route_hash, field_name="rgb_capture_route_hash")
        for field_name in (
            "target_kind",
            "current_mode",
            "model_profile_id",
            "model_profile_hash",
            "inference_implementation_hash",
            "profile_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.target_kind not in {"real", "sim"}:
            raise ValueError("Segmentation target_kind is unsupported.")
        if self.current_mode not in {"real_observe", "real_active", "sim_active"}:
            raise ValueError("Segmentation current_mode is unsupported.")
        if (self.target_kind == "real") != self.current_mode.startswith("real_"):
            raise ValueError("Segmentation target kind and current mode disagree.")
        sources = dict(self.source_id_by_role)
        if not sources or len(set(sources.values())) != len(sources):
            raise ValueError("Segmentation roles must map one-to-one to sources.")
        for role, source_id in sources.items():
            _required(role, field_name="role")
            _required(source_id, field_name="source_id")
        object.__setattr__(
            self,
            "source_id_by_role",
            MappingProxyType(dict(sorted(sources.items()))),
        )
        for name in ("max_mask_bytes", "max_overlay_bytes"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 64 * 1024 * 1024
            ):
                raise ValueError(f"{name} is outside its reviewed bound.")

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self.source_id_by_role)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "target_kind": self.target_kind,
                "current_mode": self.current_mode,
                "source_id_by_role": dict(self.source_id_by_role),
                "model_profile_id": self.model_profile_id,
                "model_profile_hash": self.model_profile_hash,
                "inference_implementation_hash": self.inference_implementation_hash,
                "max_mask_bytes": self.max_mask_bytes,
                "max_overlay_bytes": self.max_overlay_bytes,
                "prompt_policy": "exactly_one_of_text_or_pixel",
                "downstream_control_effects": "forbidden",
                **({"rgb_capture_route_hash": self.rgb_capture_route_hash}
                   if self.rgb_capture_route_hash is not None else {}),
            },
            domain="dexteleop.agent_skill.live_segmentation_profile.v1",
        )


@dataclass(frozen=True)
class SegmentationDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    target_kind: str
    executor_id: str
    deployment_binding_version: str = "1"

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                field_name: getattr(self, field_name)
                for field_name in self.__dataclass_fields__
            }
            | {
                "control_effect_class": "read_only",
                "downstream_control_effects": "forbidden",
            },
            domain="dexteleop.agent_skill.live_segmentation_deployment.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind=self.target_kind,
            execution_manifest_hash=self.execution_manifest_hash,
        )


@dataclass(frozen=True)
class SegmentationRegistration:
    profile: SegmentationProfile
    skill: SkillRegistrationTemplate
    binding: SegmentationDeploymentBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile, SegmentationProfile):
            raise TypeError("profile must be a SegmentationProfile.")
        if not isinstance(self.skill, SkillRegistrationTemplate):
            raise TypeError("skill must be a SkillRegistrationTemplate.")
        if not isinstance(self.binding, SegmentationDeploymentBinding):
            raise TypeError("binding must be a SegmentationDeploymentBinding.")
        if not isinstance(self.payload_validator, AgentSkillPayloadValidatorSet):
            raise TypeError("payload_validator must be typed.")
        if len(self.skill.routes) != 1:
            raise ValueError(
                "Segmentation registration must contain exactly one Route."
            )
        route = self.skill.routes[0]
        version = self.skill.definition.skill_version
        if version not in _SEGMENT_DEFINITIONS:
            raise ValueError("Segmentation registration uses another Skill contract.")
        if self.profile.rgb_capture_route_hash is not None and version != "3":
            raise ValueError("An exact RGB producer binding requires segmentation v3.")
        definition = _SEGMENT_DEFINITIONS[version]()
        if self.skill.definition != definition or (
            route.skill_id,
            route.skill_version,
            route.control_effect_class,
            route.privilege_tier,
            route.may_move_robot,
            route.may_open_publication,
            route.requires_operator_confirmation,
            route.requires_deadman,
            route.max_operation_duration_ms,
            route.cancellation_policy,
            route.requires_task_evaluation,
        ) != (
            definition.skill_id,
            definition.skill_version,
            "read_only",
            "t0_read_only",
            False,
            False,
            False,
            False,
            60_000,
            "cancel_only",
            False,
        ):
            raise ValueError("Segmentation registration uses another Skill contract.")
        profile_ref = route.allowed_parameter_profiles
        if len(profile_ref) != 1 or (
            profile_ref[0].parameter_profile_id != self.profile.profile_id
            or profile_ref[0].parameter_profile_hash
            != self.profile.semantic_content_hash
        ):
            raise ValueError("Segmentation Route does not bind its exact profile.")
        if (
            self.binding.route_hash != route.semantic_content_hash
            or self.binding.profile_hash != self.profile.semantic_content_hash
            or self.binding.target_kind != self.profile.target_kind
            or route.target_kind != self.profile.target_kind
            or route.supported_modes != (self.profile.current_mode,)
        ):
            raise ValueError("Segmentation deployment contracts are inconsistent.")
        self.payload_validator.validate_route(route)
        expected_availability_hash = canonical_content_hash(
            {
                "status": ("configured_requires_authenticated_rgb_model_and_artifact_store" if version == "3"
                    else "configured_requires_live_rgb_model_and_artifact_store"),
                "route_hash": route.semantic_content_hash,
                "deployment_binding_hash": self.binding.semantic_content_hash,
                "profile_hash": self.profile.semantic_content_hash,
            },
            domain="dexteleop.agent_skill.live_segmentation_availability.v1",
        )
        if self.availability_hash != expected_availability_hash:
            raise ValueError("Segmentation availability does not bind its contracts.")


def build_segmentation_registration(
    *,
    target_kind: str,
    current_mode: str,
    source_id_by_role: Mapping[str, str],
    model_profile_id: str,
    model_profile_hash: str,
    inference_implementation_hash: str,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    skill_version: str = "1",
    rgb_capture_route_hash: str | None = None,
) -> SegmentationRegistration:
    if skill_version not in _SEGMENT_DEFINITIONS:
        raise ValueError("Segmentation skill_version must be 1, 2 or 3.")
    if rgb_capture_route_hash is not None and skill_version != "3":
        raise ValueError("An exact RGB producer binding requires segmentation v3.")
    profile = SegmentationProfile(
        target_kind=target_kind,
        current_mode=current_mode,
        source_id_by_role=source_id_by_role,
        model_profile_id=model_profile_id,
        model_profile_hash=model_profile_hash,
        inference_implementation_hash=inference_implementation_hash,
        rgb_capture_route_hash=rgb_capture_route_hash,
        **({"profile_id": (("sam2-grounding-dino-rgb-artifact-indexed-contour-segmentation-v3"
            + (".paired-rgbd" if rgb_capture_route_hash is not None else "")) if skill_version == "3"
            else f"sam2-grounding-dino-same-cut-segmentation-v{skill_version}")}
            if skill_version != "1" else {}),
    )
    definition = _SEGMENT_DEFINITIONS[skill_version]()
    schema_builder = _SEGMENT_SCHEMAS[skill_version]
    schema = schema_builder(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_roles=profile.roles,
    )
    io_contract = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        input_artifact_slots=(SkillArtifactSlotSpec(
            slot_id=SEGMENT_RGB_INPUT_SLOT_ID_V3, direction="input", min_count=1, max_count=1,
            allowed_media_types=(RGB_FRAME_MEDIA_TYPE_V1,), allowed_schema_ids=(RGB_FRAME_SCHEMA_ID_V1,),
            allowed_encodings=(RGB_FRAME_ENCODING_V1,), allowed_compressions=(RGB_FRAME_COMPRESSION_V1,),
            max_artifact_bytes=SEGMENT_MAX_RGB_BYTES_V3, max_total_bytes=SEGMENT_MAX_RGB_BYTES_V3,
        ),) if skill_version == "3" else (),
        output_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=SEGMENT_MASK_OUTPUT_SLOT_ID_V1,
                direction="output",
                min_count=1 if skill_version == "1" else 0,
                max_count=1 if skill_version == "1" else 3,
                allowed_media_types=(SEGMENTATION_MASK_MEDIA_TYPE_V1,),
                allowed_schema_ids=(SEGMENTATION_MASK_SCHEMA_ID_V1,),
                allowed_encodings=(SEGMENTATION_MASK_ENCODING_V1,),
                allowed_compressions=(SEGMENTATION_MASK_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_mask_bytes,
                max_total_bytes=profile.max_mask_bytes,
            ),
            SkillArtifactSlotSpec(
                slot_id=SEGMENT_OVERLAY_OUTPUT_SLOT_ID_V1,
                direction="output",
                min_count=1,
                max_count=1,
                allowed_media_types=(SEGMENTATION_OVERLAY_MEDIA_TYPE_V1,),
                allowed_schema_ids=(SEGMENTATION_OVERLAY_SCHEMA_ID_V1,),
                allowed_encodings=(SEGMENTATION_OVERLAY_ENCODING_V1,),
                allowed_compressions=(SEGMENTATION_OVERLAY_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_overlay_bytes,
                max_total_bytes=profile.max_overlay_bytes,
            ),
        ),
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=f"{target_kind}.local_model.perception.segment",
        route_version=skill_version,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=(current_mode,),
        target_kind=target_kind,
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=60_000,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
        skill_io_contract_hash=io_contract.semantic_content_hash,
    )
    skill = SkillRegistrationTemplate(
        definition=definition,
        schema_entry=schema,
        io_contract=io_contract,
        routes=(route,),
    )
    binding = SegmentationDeploymentBinding(
        deployment_binding_id=(f"{target_kind}.local_model.segmentation:{runtime_session_id}"
            if skill_version == "1" else f"{target_kind}.local_model.segmentation.v{skill_version}:{runtime_session_id}"),
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
        target_kind=target_kind,
        executor_id=f"{target_kind}-passive-live-segmentation.v{skill_version}",
    )
    parameter_validator = {"1": validate_segment_parameters_v1, "2": validate_segment_parameters_v2,
        "3": validate_segment_parameters_v3}[skill_version]
    result_validator = _SEGMENT_RESULT_VALIDATORS[skill_version]
    validator = AgentSkillPayloadValidatorSet(
        validator_id=f"dexteleop.agent_skill.validator.{target_kind}_segmentation:{runtime_session_id}",
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda payload: parameter_validator(
            payload, allowed_roles=profile.roles
        ),
        validate_execution_result=lambda payload: result_validator(
            payload,
            source_id_by_role=profile.source_id_by_role,
            model_profile_id=profile.model_profile_id,
            model_profile_hash=profile.model_profile_hash,
        ),
        validate_execution_result_for_parameters={"1": validate_segment_result_for_parameters_v1,
            "2": validate_segment_result_for_parameters_v2, "3": validate_segment_result_for_parameters_v3}[skill_version],
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    availability_hash = canonical_content_hash(
        {
            "status": ("configured_requires_authenticated_rgb_model_and_artifact_store" if skill_version == "3"
                else "configured_requires_live_rgb_model_and_artifact_store"),
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.live_segmentation_availability.v1",
    )
    return SegmentationRegistration(
        profile=profile,
        skill=skill,
        binding=binding,
        payload_validator=validator,
        availability_hash=availability_hash,
    )


@dataclass(frozen=True)
class PreparedSegmentation:
    execution_result: dict[str, Any]
    mask_npy: bytes
    overlay_jpeg: bytes


def _validate_encoded_mask_npy(
    data: bytes,
    *,
    width: int,
    height: int,
    expected_count: int,
    expected_bounds: tuple[int, int, int, int],
) -> None:
    if not data.startswith(b"\x93NUMPY\x01\x00") or len(data) < 10:
        raise RuntimeError("Segmentation mask Artifact is not NPY v1.")
    header_length = struct.unpack("<H", data[8:10])[0]
    header_end = 10 + header_length
    if header_end > len(data):
        raise RuntimeError("Segmentation mask NPY header is truncated.")
    header_bytes = data[10:header_end]
    if not header_bytes.endswith(b"\n") or header_end % 64 != 0:
        raise RuntimeError("Segmentation mask NPY v1 header framing is invalid.")
    try:
        header = ast.literal_eval(header_bytes.decode("latin1").strip())
    except (SyntaxError, UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("Segmentation mask NPY header is invalid.") from error
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise RuntimeError("Segmentation mask NPY header fields changed.")
    shape = header["shape"]
    if (
        header["descr"] != "|b1"
        or header["fortran_order"] is not False
        or not isinstance(shape, tuple)
        or len(shape) != 2
        or any(type(value) is not int for value in shape)
        or shape != (height, width)
    ):
        raise RuntimeError("Segmentation mask NPY dtype/order/shape changed.")
    payload = data[header_end:]
    if len(payload) != width * height:
        raise RuntimeError("Segmentation mask NPY payload is invalid.")
    selected_count = 0
    x_min = width
    y_min = height
    x_max = -1
    y_max = -1
    for index, value in enumerate(payload):
        if value not in (0, 1):
            raise RuntimeError("Segmentation mask NPY payload is invalid.")
        if value == 0:
            continue
        selected_count += 1
        x = index % width
        y = index // width
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
    if selected_count != expected_count:
        raise RuntimeError("Segmentation mask NPY count differs from metadata.")
    actual_bounds = (x_min, y_min, x_max + 1, y_max + 1)
    if actual_bounds != expected_bounds:
        raise RuntimeError("Segmentation mask NPY bounds differ from metadata.")


def decode_segmentation_mask_npy(
    data: bytes,
    *,
    width: int,
    height: int,
    expected_count: int,
    expected_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Decode one already-bounded mask after rechecking its exact metadata."""

    _validate_encoded_mask_npy(
        data,
        width=width,
        height=height,
        expected_count=expected_count,
        expected_bounds=expected_bounds,
    )
    try:
        mask = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError("Segmentation mask Artifact is not valid NPY.") from error
    if mask.dtype != np.bool_ or mask.shape != (height, width):
        raise RuntimeError("Segmentation mask decoded identity changed.")
    result = np.ascontiguousarray(mask)
    result.setflags(write=False)
    return result


def _validate_overlay_jpeg(data: bytes, *, width: int, height: int) -> None:
    try:
        with Image.open(io.BytesIO(data)) as overlay:
            if (
                overlay.format != "JPEG"
                or overlay.mode != "RGB"
                or overlay.size != (width, height)
            ):
                raise RuntimeError(
                    "Segmentation overlay JPEG identity differs from metadata."
                )
            overlay.load()
    except (OSError, ValueError) as error:
        raise RuntimeError("Segmentation overlay Artifact is invalid JPEG.") from error


class LiveSegmentationProvider:
    """Capture one target-specific RGB frame and run the shared localizer."""

    def __init__(
        self,
        *,
        localizer: object,
        capture_port: LiveRgbSegmentationCapturePort,
        registration: SegmentationRegistration,
    ) -> None:
        if not isinstance(registration, SegmentationRegistration):
            raise TypeError("registration must be typed.")
        if registration.skill.definition.skill_version != "1":
            raise ValueError("The shared single-target provider requires its v1 registration.")
        for method_name in ("segment", "encode_segment_artifacts"):
            if not callable(getattr(localizer, method_name, None)):
                raise TypeError(f"localizer must expose {method_name}().")
        if (
            getattr(localizer, "semantic_content_hash", None)
            != registration.profile.inference_implementation_hash
        ):
            raise ValueError(
                "Localizer identity differs from the segmentation profile."
            )
        model_spec = getattr(localizer, "model_spec", None)
        if (
            getattr(model_spec, "model_profile_id", None)
            != registration.profile.model_profile_id
            or getattr(model_spec, "semantic_content_hash", None)
            != registration.profile.model_profile_hash
        ):
            raise ValueError("Localizer model identity differs from the profile.")
        if not callable(getattr(capture_port, "capture_segment_rgb", None)):
            raise TypeError("capture_port must expose capture_segment_rgb().")
        self.localizer = localizer
        self.capture_port = capture_port
        self.registration = registration

    @property
    def started(self) -> bool:
        return bool(getattr(self.localizer, "started", False))

    def segment_request(self, request: Any) -> PreparedSegmentation:
        parameters = request.parameters
        requested_role = parameters["role"]
        expected_revision = request.expected_station_revision
        capture = self.capture_port.capture_segment_rgb(
            role=requested_role,
            expected_station_revision=expected_revision,
        )
        if capture.station_revision != expected_revision:
            raise RuntimeError(
                "Segmentation capture returned another Station revision."
            )
        return self.segment_capture(capture=capture, parameters=parameters)

    def segment_capture(
        self,
        *,
        capture: LiveRgbSegmentationCapture,
        parameters: Mapping[str, Any],
    ) -> PreparedSegmentation:
        """Run the registered localizer on one caller-owned immutable RGB cut."""

        result = self.try_segment_capture(capture=capture, parameters=parameters)
        if result is None:
            raise RuntimeError("segmentation_no_mask_above_threshold")
        return result

    def try_segment_capture(
        self, *, capture: LiveRgbSegmentationCapture, parameters: Mapping[str, Any],
    ) -> PreparedSegmentation | None:
        """Run the same validation/encoder while explicitly retaining a no-mask outcome."""

        if not isinstance(capture, LiveRgbSegmentationCapture):
            raise TypeError("capture must be a LiveRgbSegmentationCapture.")
        self.registration.payload_validator.validate_parameters(parameters)
        requested_role = parameters["role"]
        if capture.role != requested_role:
            raise RuntimeError("Segmentation capture returned another camera role.")
        if (
            self.registration.profile.source_id_by_role.get(requested_role)
            != capture.source_id
        ):
            raise RuntimeError(
                "Segmentation capture changed its registered role binding."
            )
        pixel_raw = parameters["target_pixel_uv"]
        pixel_uv = None if pixel_raw is None else (int(pixel_raw[0]), int(pixel_raw[1]))
        segment = self.localizer.segment(
            capture.image,
            language_prompt=parameters["language_prompt"],
            pixel_uv=pixel_uv,
            min_score=float(parameters["min_score"]),
        )
        if segment is None:
            return None
        encoded = self.localizer.encode_segment_artifacts(capture.image, segment)
        raw_bounds = getattr(encoded, "box_xyxy", None)
        if (
            not isinstance(raw_bounds, (list, tuple))
            or len(raw_bounds) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_bounds
            )
        ):
            raise RuntimeError("Segmentation encoder returned invalid mask bounds.")
        x0, y0, x1, y1 = raw_bounds
        if not (
            0 <= x0 < x1 <= capture.image.width and 0 <= y0 < y1 <= capture.image.height
        ):
            raise RuntimeError("Segmentation encoder returned out-of-frame bounds.")
        raw_score = getattr(encoded, "score", None)
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, Real)
            or not math.isfinite(float(raw_score))
            or not float(parameters["min_score"]) <= float(raw_score) <= 1.0
        ):
            raise RuntimeError("Segmentation model returned an invalid score.")
        label = getattr(encoded, "label", None)
        if label is not None and (
            not isinstance(label, str)
            or not label
            or label != label.strip()
            or len(label) > 256
        ):
            raise RuntimeError("Segmentation model returned an invalid label.")
        mask_pixel_count = getattr(encoded, "mask_pixel_count", None)
        if (
            isinstance(mask_pixel_count, bool)
            or not isinstance(mask_pixel_count, int)
            or not 1 <= mask_pixel_count <= capture.image.width * capture.image.height
        ):
            raise RuntimeError("Segmentation encoder returned an invalid mask count.")
        mask_npy = getattr(encoded, "mask_npy", None)
        overlay_jpeg = getattr(encoded, "overlay_jpeg", None)
        if not isinstance(mask_npy, bytes) or not isinstance(overlay_jpeg, bytes):
            raise TypeError("Segmentation encoder returned invalid Artifact bytes.")
        profile = self.registration.profile
        if (
            len(mask_npy) > profile.max_mask_bytes
            or len(overlay_jpeg) > profile.max_overlay_bytes
        ):
            raise RuntimeError("Segmentation output exceeds its deployment byte bound.")
        _validate_encoded_mask_npy(
            mask_npy,
            width=capture.image.width,
            height=capture.image.height,
            expected_count=mask_pixel_count,
            expected_bounds=tuple(raw_bounds),
        )
        _validate_overlay_jpeg(
            overlay_jpeg,
            width=capture.image.width,
            height=capture.image.height,
        )
        return PreparedSegmentation(
            execution_result={
                "source_id": capture.source_id,
                "role": capture.role,
                "captured_at": float(capture.captured_at),
                "station_revision": capture.station_revision,
                "input_rgb_content_hash": capture.rgb_content_hash,
                "image_width": capture.image.width,
                "image_height": capture.image.height,
                "mode": "text"
                if parameters["language_prompt"] is not None
                else "point",
                "language_prompt": parameters["language_prompt"],
                "target_pixel_uv": None if pixel_uv is None else list(pixel_uv),
                "min_score": float(parameters["min_score"]),
                "score": float(raw_score),
                "label": label,
                "coordinate_format": "xyxy_pixel_0based_half_open",
                "mask_bounds_xyxy": list(raw_bounds),
                "mask_pixel_count": mask_pixel_count,
                "model_profile_id": profile.model_profile_id,
                "model_profile_hash": profile.model_profile_hash,
                "mask_media_content_hash": hashlib.sha256(mask_npy).hexdigest(),
                "overlay_media_content_hash": hashlib.sha256(overlay_jpeg).hexdigest(),
            },
            mask_npy=mask_npy,
            overlay_jpeg=overlay_jpeg,
        )


def build_segmentation_executor(
    *,
    registration: SegmentationRegistration,
    provider: LiveSegmentationProvider,
    artifact_store: RecoverableAgentArtifactStore,
    runtime_session_id: str,
    current_station_revision: Callable[[], int],
    max_retained_operations: int,
    clock: Callable[[], float] = time.time,
) -> PassiveReadSkillExecutor:
    if not isinstance(registration, SegmentationRegistration) or not isinstance(
        provider, LiveSegmentationProvider
    ):
        raise TypeError("registration and provider must be typed.")
    if not isinstance(artifact_store, RecoverableAgentArtifactStore):
        raise TypeError("artifact_store must be typed.")
    session_id = _required(runtime_session_id, field_name="runtime_session_id")
    if provider.registration is not registration:
        raise ValueError("Segmentation provider and executor registrations differ.")
    if session_id != registration.binding.runtime_session_id:
        raise ValueError(
            "Segmentation executor runtime session differs from its binding."
        )

    def read(request: Any, operation: Any) -> PassiveReadResult:
        prepared = provider.segment_request(request)
        expected_revision = request.expected_station_revision
        if current_station_revision() != expected_revision:
            raise RuntimeError(
                "Station revision changed before segmentation Artifact commit."
            )
        artifacts = (
            (
                SEGMENT_MASK_OUTPUT_SLOT_ID_V1,
                prepared.mask_npy,
                SEGMENTATION_MASK_MEDIA_TYPE_V1,
                SEGMENTATION_MASK_SCHEMA_ID_V1,
                SEGMENTATION_MASK_ENCODING_V1,
                SEGMENTATION_MASK_COMPRESSION_V1,
            ),
            (
                SEGMENT_OVERLAY_OUTPUT_SLOT_ID_V1,
                prepared.overlay_jpeg,
                SEGMENTATION_OVERLAY_MEDIA_TYPE_V1,
                SEGMENTATION_OVERLAY_SCHEMA_ID_V1,
                SEGMENTATION_OVERLAY_ENCODING_V1,
                SEGMENTATION_OVERLAY_COMPRESSION_V1,
            ),
        )
        bindings = []
        refs = []
        for (
            slot_id,
            data,
            media_type,
            schema_id,
            encoding,
            compression,
        ) in artifacts:
            artifact = artifact_ref_for_bytes(
                data,
                media_type=media_type,
                schema_id=schema_id,
                encoding=encoding,
                compression=compression,
            )
            committed = commit_artifact_bytes(
                store=artifact_store,
                data=data,
                artifact=artifact,
                producer_id=f"agent_gateway.provider.{registration.profile.target_kind}_segmentation",
                runtime_session_id=session_id,
                idempotency_key=f"segmentation-output:{operation.operation_id}:{slot_id}",
                now=float(clock()),
            )
            evidence_ref = (
                f"{registration.profile.target_kind}-segmentation:{committed.sha256}"
            )
            refs.append(evidence_ref)
            bindings.append(
                AgentArtifactBinding(
                    slot_id=slot_id,
                    ordinal=0,
                    artifact=committed,
                    production_receipt_ref=evidence_ref,
                )
            )
        if current_station_revision() != expected_revision:
            raise RuntimeError(
                "Station revision changed during segmentation Artifact commit."
            )
        if (
            prepared.execution_result["mask_media_content_hash"]
            != bindings[0].artifact.sha256
            or prepared.execution_result["overlay_media_content_hash"]
            != bindings[1].artifact.sha256
        ):
            raise RuntimeError("Segmentation Artifact hash changed during commit.")
        receipt = canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "request_hash": request.semantic_content_hash,
                "execution_result": prepared.execution_result,
                "artifact_ref_hashes": [
                    binding.artifact.semantic_content_hash for binding in bindings
                ],
            },
            domain="dexteleop.agent_skill.live_segmentation_receipt.v1",
        )
        return PassiveReadResult(
            execution_result=prepared.execution_result,
            evidence_refs=("live-segmentation:" + receipt, *refs),
            output_artifact_bindings=tuple(bindings),
            reason_code="passive_object_segmentation_completed",
        )

    return PassiveReadSkillExecutor(
        port_id=f"agent-gateway:{registration.profile.target_kind}-passive-segmentation:{session_id}",
        executor_id=registration.binding.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )


__all__ = [
    "SEGMENT_MASK_OUTPUT_SLOT_ID_V1",
    "SEGMENT_OVERLAY_OUTPUT_SLOT_ID_V1",
    "LiveRgbSegmentationCapture",
    "LiveRgbSegmentationCapturePort",
    "LiveSegmentationProvider",
    "PreparedSegmentation",
    "SegmentationDeploymentBinding",
    "SegmentationProfile",
    "SegmentationRegistration",
    "build_segmentation_executor",
    "build_segmentation_registration",
    "decode_segmentation_mask_npy",
]
