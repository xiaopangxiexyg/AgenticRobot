"""Pure real-observe deployment binding for supplied-image localization."""

from __future__ import annotations

from dataclasses import dataclass

from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
)
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOContract,
)
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.definition import (
    localize_objects_skill_definition_v1,
)
from agent_gateway.skills.perception.schemas import (
    localize_objects_schema_entry_v1,
    validate_localize_objects_parameters_v1,
    validate_localize_objects_result_for_parameters_v1,
    validate_localize_objects_result_v1,
    validate_no_task_evaluation_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from perception_contracts.artifacts import (
    LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
    LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    LOCALIZATION_ANNOTATED_IMAGE_SCHEMA_ID_V1,
    LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
    LOCALIZATION_INPUT_IMAGE_SCHEMA_ID_V1,
    LOCALIZATION_INPUT_JPEG_ENCODING_V1,
    LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
    LOCALIZATION_INPUT_PNG_ENCODING_V1,
    LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1,
)

REAL_LOCALIZE_OBJECTS_ROUTE_ID_V1 = "real.local_model.perception.localize_objects"
REAL_LOCALIZE_OBJECTS_EXECUTOR_ID_V1 = "real-passive-object-localization.v1"
REAL_LOCALIZE_OBJECTS_INPUT_SLOT_ID_V1 = "input_image"
REAL_LOCALIZE_OBJECTS_OUTPUT_SLOT_ID_V1 = "annotated_image"
REAL_LOCALIZE_OBJECTS_MAX_IMAGE_BYTES_V1 = 16 * 1024 * 1024
REAL_LOCALIZE_OBJECTS_MAX_ANNOTATED_BYTES_V1 = 16 * 1024 * 1024
REAL_LOCALIZE_OBJECTS_MAX_DETECTIONS_V1 = 100


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class RealObjectLocalizationProfile:
    """Exact model identity, inference bounds, and test-path capability."""

    model_profile_id: str
    model_profile_hash: str
    inference_implementation_hash: str
    local_path_enabled: bool
    max_detections: int = REAL_LOCALIZE_OBJECTS_MAX_DETECTIONS_V1
    max_image_bytes: int = REAL_LOCALIZE_OBJECTS_MAX_IMAGE_BYTES_V1
    max_annotated_bytes: int = REAL_LOCALIZE_OBJECTS_MAX_ANNOTATED_BYTES_V1
    profile_id: str = "real-sam2-grounding-dino-object-localization-v1"

    def __post_init__(self) -> None:
        for field_name in (
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
        if not isinstance(self.local_path_enabled, bool):
            raise TypeError("local_path_enabled must be boolean.")
        for field_name, maximum in (
            ("max_detections", 256),
            ("max_image_bytes", 64 * 1024 * 1024),
            ("max_annotated_bytes", 64 * 1024 * 1024),
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"{field_name} is outside its reviewed bound.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "model_profile_id": self.model_profile_id,
                "model_profile_hash": self.model_profile_hash,
                "inference_implementation_hash": self.inference_implementation_hash,
                "local_path_test_input": (
                    "explicit_allowlisted_roots" if self.local_path_enabled else "disabled"
                ),
                "max_detections": self.max_detections,
                "max_image_bytes": self.max_image_bytes,
                "max_annotated_bytes": self.max_annotated_bytes,
                "coordinate_format": "xyxy_pixel_0based_half_open",
                "output_artifact": "annotated_jpeg",
                "downstream_control_effects": "forbidden",
            },
            domain="dexteleop.agent_skill.real_object_localization_profile.v1",
        )


@dataclass(frozen=True)
class RealObjectLocalizationDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    executor_id: str = REAL_LOCALIZE_OBJECTS_EXECUTOR_ID_V1
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
                "target_kind": "real",
                "control_effect_class": "read_only",
                "downstream_control_effects": "forbidden",
                "inference_owner": "local_sam2_grounding_dino_resource",
            },
            domain="dexteleop.agent_skill.real_object_localization_deployment.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="real",
            execution_manifest_hash=self.execution_manifest_hash,
        )


@dataclass(frozen=True)
class RealObjectLocalizationRegistration:
    profile: RealObjectLocalizationProfile
    skill: SkillRegistrationTemplate
    binding: RealObjectLocalizationDeploymentBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str


def build_real_object_localization_registration(
    *,
    model_profile_id: str,
    model_profile_hash: str,
    inference_implementation_hash: str,
    local_path_enabled: bool,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
) -> RealObjectLocalizationRegistration:
    """Build one closed read-only localization registration."""

    profile = RealObjectLocalizationProfile(
        model_profile_id=model_profile_id,
        model_profile_hash=model_profile_hash,
        inference_implementation_hash=inference_implementation_hash,
        local_path_enabled=local_path_enabled,
    )
    definition = localize_objects_skill_definition_v1()
    schema = localize_objects_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        max_detections=profile.max_detections,
    )
    io_contract = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        input_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=REAL_LOCALIZE_OBJECTS_INPUT_SLOT_ID_V1,
                direction="input",
                min_count=0,
                max_count=1,
                allowed_media_types=(
                    LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
                    LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1,
                ),
                allowed_schema_ids=(LOCALIZATION_INPUT_IMAGE_SCHEMA_ID_V1,),
                allowed_encodings=(
                    LOCALIZATION_INPUT_JPEG_ENCODING_V1,
                    LOCALIZATION_INPUT_PNG_ENCODING_V1,
                ),
                allowed_compressions=(LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_image_bytes,
                max_total_bytes=profile.max_image_bytes,
            ),
        ),
        output_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=REAL_LOCALIZE_OBJECTS_OUTPUT_SLOT_ID_V1,
                direction="output",
                min_count=1,
                max_count=1,
                allowed_media_types=(LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,),
                allowed_schema_ids=(LOCALIZATION_ANNOTATED_IMAGE_SCHEMA_ID_V1,),
                allowed_encodings=(LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,),
                allowed_compressions=(LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_annotated_bytes,
                max_total_bytes=profile.max_annotated_bytes,
            ),
        ),
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=REAL_LOCALIZE_OBJECTS_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=("real_observe",),
        target_kind="real",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=60_000,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
        skill_io_contract_hash=io_contract.semantic_content_hash,
    )
    registration = SkillRegistrationTemplate(
        definition=definition,
        schema_entry=schema,
        io_contract=io_contract,
        routes=(route,),
    )
    binding = RealObjectLocalizationDeploymentBinding(
        deployment_binding_id="real.local_model.object_localization:" + runtime_session_id,
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id="dexteleop.agent_skill.validator.real_object_localization:"
        + runtime_session_id,
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda payload: validate_localize_objects_parameters_v1(
            payload,
            max_detections=profile.max_detections,
            local_path_enabled=profile.local_path_enabled,
        ),
        validate_execution_result=lambda payload: validate_localize_objects_result_v1(
            payload,
            max_detections=profile.max_detections,
            model_profile_id=profile.model_profile_id,
            model_profile_hash=profile.model_profile_hash,
        ),
        validate_execution_result_for_parameters=(
            validate_localize_objects_result_for_parameters_v1
        ),
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    availability_hash = canonical_content_hash(
        {
            "status": "configured_requires_verified_model_assets_and_artifact_store",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.real_object_localization_availability.v1",
    )
    return RealObjectLocalizationRegistration(
        profile=profile,
        skill=registration,
        binding=binding,
        payload_validator=validator,
        availability_hash=availability_hash,
    )


__all__ = [
    "REAL_LOCALIZE_OBJECTS_EXECUTOR_ID_V1",
    "REAL_LOCALIZE_OBJECTS_INPUT_SLOT_ID_V1",
    "REAL_LOCALIZE_OBJECTS_MAX_ANNOTATED_BYTES_V1",
    "REAL_LOCALIZE_OBJECTS_MAX_DETECTIONS_V1",
    "REAL_LOCALIZE_OBJECTS_MAX_IMAGE_BYTES_V1",
    "REAL_LOCALIZE_OBJECTS_OUTPUT_SLOT_ID_V1",
    "REAL_LOCALIZE_OBJECTS_ROUTE_ID_V1",
    "RealObjectLocalizationDeploymentBinding",
    "RealObjectLocalizationProfile",
    "RealObjectLocalizationRegistration",
    "build_real_object_localization_registration",
]
