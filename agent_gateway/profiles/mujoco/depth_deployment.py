"""MuJoCo Route binding for the passive depth Artifact Skill."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
    capture_depth_skill_definition_v1,
)
from agent_gateway.skills.perception.schemas import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
    capture_depth_schema_entry_v1,
    validate_capture_depth_parameters_v1,
    validate_capture_depth_result_for_parameters_v1,
    validate_capture_depth_result_v1,
    validate_no_task_evaluation_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

MUJOCO_CAPTURE_DEPTH_ROUTE_ID_V1 = "sim.mujoco.perception.capture_depth"
MUJOCO_CAPTURE_DEPTH_EXECUTOR_ID_V1 = "control-station.mujoco-passive-depth-capture.v1"
MUJOCO_CAPTURE_DEPTH_PARAMETER_PROFILE_ID_V1 = "mujoco-registered-native-depth-views-v1"
MUJOCO_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1 = "depth_frame"
MUJOCO_CAPTURE_DEPTH_MAX_ARTIFACT_BYTES_V1 = 16 * 1024 * 1024


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class MujocoDepthCaptureProfile:
    """Exact registered view catalog and finite output bound."""

    dimensions_by_view: tuple[tuple[str, int, int], ...]
    max_artifact_bytes: int = MUJOCO_CAPTURE_DEPTH_MAX_ARTIFACT_BYTES_V1
    profile_id: str = MUJOCO_CAPTURE_DEPTH_PARAMETER_PROFILE_ID_V1

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_artifact_bytes, bool)
            or not isinstance(self.max_artifact_bytes, int)
            or self.max_artifact_bytes <= 0
        ):
            raise ValueError("max_artifact_bytes must be positive.")
        normalized: list[tuple[str, int, int]] = []
        for view_id, width, height in self.dimensions_by_view:
            view = _required(view_id, field_name="view_id")
            if (
                isinstance(width, bool)
                or not isinstance(width, int)
                or width <= 0
                or isinstance(height, bool)
                or not isinstance(height, int)
                or height <= 0
            ):
                raise ValueError("Depth view dimensions must be positive integers.")
            if width * height * 4 + 1024 > self.max_artifact_bytes:
                raise ValueError("Depth view exceeds the profile Artifact bound.")
            normalized.append((view, width, height))
        normalized.sort()
        if not normalized or len({item[0] for item in normalized}) != len(normalized):
            raise ValueError("Depth profile requires unique registered views.")
        object.__setattr__(self, "dimensions_by_view", tuple(normalized))
        object.__setattr__(
            self,
            "profile_id",
            _required(self.profile_id, field_name="profile_id"),
        )

    @property
    def view_ids(self) -> tuple[str, ...]:
        return tuple(item[0] for item in self.dimensions_by_view)

    @property
    def dimensions(self) -> Mapping[str, tuple[int, int]]:
        return {
            view_id: (width, height)
            for view_id, width, height in self.dimensions_by_view
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "dimensions_by_view": self.dimensions_by_view,
                "max_artifact_bytes": self.max_artifact_bytes,
                "depth_unit": "meter",
                "invalid_depth_value": 0.0,
                "dtype": "float32",
                "array_order": "C",
                "media_type": DEPTH_FRAME_MEDIA_TYPE_V1,
                "schema_id": DEPTH_FRAME_SCHEMA_ID_V1,
                "encoding": DEPTH_FRAME_ENCODING_V1,
                "compression": DEPTH_FRAME_COMPRESSION_V1,
                "capture_semantics": "latest_native_resolution_scene_cut",
            },
            domain="dexteleop.agent_skill.mujoco_depth_capture_profile.v1",
        )


@dataclass(frozen=True)
class MujocoDepthCaptureDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    executor_id: str = MUJOCO_CAPTURE_DEPTH_EXECUTOR_ID_V1
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
                "target_kind": "sim",
                "control_effect_class": "read_only",
                "downstream_control_effects": "forbidden",
                "render_owner": "mujoco_scene_capture_boundary",
            },
            domain="dexteleop.agent_skill.mujoco_depth_capture_binding.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="sim",
            execution_manifest_hash=self.execution_manifest_hash,
        )


@dataclass(frozen=True)
class MujocoDepthCaptureRegistration:
    registration: SkillRegistrationTemplate
    profile: MujocoDepthCaptureProfile
    deployment_binding: MujocoDepthCaptureDeploymentBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str


def build_mujoco_depth_capture_registration(
    *,
    source_catalog: tuple[Mapping[str, object], ...],
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
) -> MujocoDepthCaptureRegistration:
    dimensions: list[tuple[str, int, int]] = []
    for item in source_catalog:
        if not isinstance(item, Mapping) or set(item) != {
            "view_id",
            "native_width",
            "native_height",
        }:
            raise ValueError("Depth source catalog entry has unexpected fields.")
        dimensions.append(
            (
                item["view_id"],
                item["native_width"],
                item["native_height"],
            )
        )
    profile = MujocoDepthCaptureProfile(dimensions_by_view=tuple(dimensions))
    definition = capture_depth_skill_definition_v1()
    schema_entry = capture_depth_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_view_ids=profile.view_ids,
    )
    io_contract = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema_entry.parameter_schema_hash,
        execution_result_schema_hash=schema_entry.execution_result_schema_hash,
        task_evaluation_schema_hash=schema_entry.task_evaluation_schema_hash,
        output_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=MUJOCO_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
                direction="output",
                min_count=1,
                max_count=1,
                allowed_media_types=(DEPTH_FRAME_MEDIA_TYPE_V1,),
                allowed_schema_ids=(DEPTH_FRAME_SCHEMA_ID_V1,),
                allowed_encodings=(DEPTH_FRAME_ENCODING_V1,),
                allowed_compressions=(DEPTH_FRAME_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_artifact_bytes,
                max_total_bytes=profile.max_artifact_bytes,
            ),
        ),
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=MUJOCO_CAPTURE_DEPTH_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=schema_entry.parameter_schema_hash,
        execution_result_schema_hash=schema_entry.execution_result_schema_hash,
        task_evaluation_schema_hash=schema_entry.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema_entry.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=3_000,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
        skill_io_contract_hash=io_contract.semantic_content_hash,
    )
    registration = SkillRegistrationTemplate(
        definition=definition,
        schema_entry=schema_entry,
        io_contract=io_contract,
        routes=(route,),
    )
    binding = MujocoDepthCaptureDeploymentBinding(
        deployment_binding_id=(
            "sim.mujoco.passive_depth_capture:" + runtime_session_id
        ),
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
    )

    def validate_parameters(payload: Mapping[str, Any]) -> None:
        validate_capture_depth_parameters_v1(
            payload,
            allowed_view_ids=profile.view_ids,
        )

    def validate_result(payload: Mapping[str, Any]) -> None:
        validate_capture_depth_result_v1(
            payload,
            dimensions_by_view=profile.dimensions,
        )

    validator = AgentSkillPayloadValidatorSet(
        validator_id=(
            "dexteleop.agent_skill.validator.mujoco_depth_capture:" + runtime_session_id
        ),
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=validate_parameters,
        validate_execution_result=validate_result,
        validate_execution_result_for_parameters=(
            validate_capture_depth_result_for_parameters_v1
        ),
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    availability_hash = canonical_content_hash(
        {
            "status": "configured_requires_artifact_store",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.mujoco_depth_capture_availability.v1",
    )
    return MujocoDepthCaptureRegistration(
        registration=registration,
        profile=profile,
        deployment_binding=binding,
        payload_validator=validator,
        availability_hash=availability_hash,
    )


__all__ = [
    "MUJOCO_CAPTURE_DEPTH_EXECUTOR_ID_V1",
    "MUJOCO_CAPTURE_DEPTH_MAX_ARTIFACT_BYTES_V1",
    "MUJOCO_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1",
    "MUJOCO_CAPTURE_DEPTH_PARAMETER_PROFILE_ID_V1",
    "MUJOCO_CAPTURE_DEPTH_ROUTE_ID_V1",
    "MujocoDepthCaptureDeploymentBinding",
    "MujocoDepthCaptureProfile",
    "MujocoDepthCaptureRegistration",
    "build_mujoco_depth_capture_registration",
]
