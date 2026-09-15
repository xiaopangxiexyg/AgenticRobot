"""Deployment-bound, read-only live RGB-D point-grasp Route.

The Route accepts only a registered semantic camera role, a clicked pixel, and
one bounded candidate count.  Camera identity, RGB-D acquisition, intrinsics,
calibration, and Station-frame conversion remain deployment-owned inputs.
"""

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
    plan_point_grasp_skill_definition_v2,
    plan_point_grasp_skill_definition_v3,
)
from agent_gateway.skills.perception.point_grasp_schemas import (
    point_grasp_schema_entry_v2,
    point_grasp_schema_entry_v3,
    validate_point_grasp_parameters_v2,
    validate_point_grasp_parameters_v3,
    validate_point_grasp_result_for_parameters_v2,
    validate_point_grasp_result_for_parameters_v3,
    validate_point_grasp_result_v2,
    validate_point_grasp_result_v3,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.grasp_rgbd_snapshot import (
    GRASP_RGBD_SNAPSHOT_MEDIA_TYPE, GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
    GRASP_RGBD_SNAPSHOT_ENCODING, GRASP_RGBD_SNAPSHOT_MAX_BYTES,
)
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
    LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,
)

from .point_grasp_deployment import SUPPORTED_POINT_GRASP_BACKEND_IDS

REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID = "point_grasp_rgbd_snapshot"

REAL_STATION_POINT_GRASP_ROUTE_ID_V2 = (
    "real.realsense.local_model.perception.plan_point_grasp"
)
REAL_STATION_POINT_GRASP_EXECUTOR_ID_V2 = (
    "control-station.real-passive-station-point-grasp.v2"
)
REAL_STATION_POINT_GRASP_EXECUTOR_ID_V3 = (
    "control-station.real-passive-station-point-grasp.v3"
)
REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2 = "annotated_grasp_proposals"


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class StationPointGraspArmReachabilityProfile:
    """Trusted RM65 endpoint-IK classification bounds for optional v3 output."""

    arm_side: str
    evaluator_profile_id: str
    evaluator_profile_hash: str
    solver_reference_frame_id: str
    solver_reference_frame_spec_hash: str
    frame_graph_hash: str
    time_budget_ms: float = 1_000.0
    max_solve_cycles: int = 2_880
    max_measured_qpos_age_ms: float = 200.0
    standard_position_tolerance_m: float = 0.003
    standard_orientation_tolerance_deg: float = 1.0
    basic_position_tolerance_m: float = 0.05
    basic_orientation_tolerance_deg: float = 20.0

    def __post_init__(self) -> None:
        for field_name in (
            "evaluator_profile_id",
            "evaluator_profile_hash",
            "solver_reference_frame_id",
            "solver_reference_frame_spec_hash",
            "frame_graph_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        side = str(self.arm_side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("arm_side must be left or right.")
        object.__setattr__(self, "arm_side", side)
        for field_name in (
            "time_budget_ms",
            "max_measured_qpos_age_ms",
            "standard_position_tolerance_m",
            "standard_orientation_tolerance_deg",
            "basic_position_tolerance_m",
            "basic_orientation_tolerance_deg",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 < value < float("inf"):
                raise ValueError(f"{field_name} must be finite and positive.")
            object.__setattr__(self, field_name, value)
        cycles = int(self.max_solve_cycles)
        if isinstance(self.max_solve_cycles, bool) or cycles < 1:
            raise ValueError("max_solve_cycles must be positive.")
        object.__setattr__(self, "max_solve_cycles", cycles)
        if (
            self.standard_position_tolerance_m >= self.basic_position_tolerance_m
            or self.standard_orientation_tolerance_deg
            >= self.basic_orientation_tolerance_deg
        ):
            raise ValueError("Standard reachability tolerances must be stricter.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                field_name: getattr(self, field_name)
                for field_name in self.__dataclass_fields__
            }
            | {
                "evaluation_scope": (
                    "finite_time_endpoint_ik_not_path_or_grasp_success"
                ),
                "candidate_order_effect": "annotation_only_order_unchanged",
            },
            domain="dexteleop.agent_skill.station_point_grasp_arm_reachability.v1",
        )


@dataclass(frozen=True)
class StationBoundPointGraspProfile:
    """Exact live sources, frames, model identity, and bounded inference."""

    aligned_rgbd_sources: tuple[tuple[str, str, int, int], ...]
    source_binding_hash: str
    frame_binding_hash: str
    frame_graph_hash: str
    source_frame_id: str
    source_frame_spec_hash: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    backend_id: str
    model_profile_id: str
    model_profile_hash: str
    inference_implementation_hash: str
    arm_reachability_profile: StationPointGraspArmReachabilityProfile | None = None
    max_candidates: int = 5
    max_annotated_bytes: int = 16 * 1024 * 1024
    profile_id: str = "real-live-station-bound-dmtac-m-point-grasp-v2"

    def __post_init__(self) -> None:
        for field_name in (
            "source_binding_hash",
            "frame_binding_hash",
            "frame_graph_hash",
            "source_frame_id",
            "source_frame_spec_hash",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "backend_id",
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
        if self.backend_id not in SUPPORTED_POINT_GRASP_BACKEND_IDS:
            raise ValueError("Station point-grasp backend is unsupported.")
        if self.arm_reachability_profile is not None and not isinstance(
            self.arm_reachability_profile,
            StationPointGraspArmReachabilityProfile,
        ):
            raise TypeError("arm_reachability_profile must be typed or None.")
        if (
            self.arm_reachability_profile is not None
            and self.arm_reachability_profile.frame_graph_hash != self.frame_graph_hash
        ):
            raise ValueError("Arm reachability frame graph differs from point-grasp.")
        sources = tuple(self.aligned_rgbd_sources)
        if not sources:
            raise ValueError("At least one aligned RGB-D source is required.")
        normalized = []
        for source in sources:
            if not isinstance(source, tuple) or len(source) != 4:
                raise ValueError("Aligned RGB-D source entries must be 4-tuples.")
            source_id = _required(source[0], field_name="source_id")
            role = _required(source[1], field_name="role")
            width, height = source[2:]
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (width, height)
            ):
                raise ValueError("Aligned RGB-D dimensions must be positive integers.")
            normalized.append((source_id, role, width, height))
        normalized_tuple = tuple(sorted(normalized, key=lambda item: item[1]))
        if len({item[0] for item in normalized_tuple}) != len(normalized_tuple):
            raise ValueError("Aligned RGB-D source IDs must be unique.")
        if len({item[1] for item in normalized_tuple}) != len(normalized_tuple):
            raise ValueError("Aligned RGB-D roles must be unique.")
        object.__setattr__(self, "aligned_rgbd_sources", normalized_tuple)
        if not 1 <= self.max_candidates <= 5:
            raise ValueError("max_candidates must be in [1, 5].")
        if not 1 <= self.max_annotated_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_annotated_bytes exceeds the reviewed bound.")

    @property
    def source_by_role(self) -> dict[str, str]:
        return {
            role: source_id
            for source_id, role, _width, _height in self.aligned_rgbd_sources
        }

    @property
    def dimensions_by_role(self) -> dict[str, tuple[int, int]]:
        return {
            role: (width, height)
            for _source_id, role, width, height in self.aligned_rgbd_sources
        }

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "aligned_rgbd_sources": self.aligned_rgbd_sources,
            "source_binding_hash": self.source_binding_hash,
            "frame_binding_hash": self.frame_binding_hash,
            "frame_graph_hash": self.frame_graph_hash,
            "source_frame_id": self.source_frame_id,
            "source_frame_spec_hash": self.source_frame_spec_hash,
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "backend_id": self.backend_id,
            "model_profile_id": self.model_profile_id,
            "model_profile_hash": self.model_profile_hash,
            "inference_implementation_hash": self.inference_implementation_hash,
            "max_candidates": self.max_candidates,
            "max_annotated_bytes": self.max_annotated_bytes,
            "input": "deployment_owned_single_live_aligned_rgbd_frame",
            "output": "station_bound_ranked_unprivileged_tool_tcp_proposals",
            "collision_scope": "visible_scene_only_not_full_scene_or_robot_safety",
            "selection_authority": "none",
            "downstream_control_effects": "forbidden",
        }
        if self.arm_reachability_profile is None:
            return canonical_content_hash(
                payload,
                domain="dexteleop.agent_skill.station_point_grasp_profile.v2",
            )
        payload["arm_reachability_profile_hash"] = (
            self.arm_reachability_profile.semantic_content_hash
        )
        payload["output"] = (
            "station_bound_ranked_unprivileged_tool_tcp_proposals_with_optional_"
            "finite_time_arm_ik_annotations"
        )
        return canonical_content_hash(
            payload,
            domain="dexteleop.agent_skill.station_point_grasp_profile.v3",
        )


@dataclass(frozen=True)
class StationBoundPointGraspBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    executor_id: str = REAL_STATION_POINT_GRASP_EXECUTOR_ID_V2
    deployment_binding_version: str = "2"

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.deployment_binding_version not in {"2", "3"}:
            raise ValueError("deployment_binding_version must be 2 or 3.")

    @property
    def semantic_content_hash(self) -> str:
        domain = (
            "dexteleop.agent_skill.station_point_grasp_deployment.v3"
            if self.deployment_binding_version == "3"
            else "dexteleop.agent_skill.station_point_grasp_deployment.v2"
        )
        return canonical_content_hash(
            {
                field_name: getattr(self, field_name)
                for field_name in self.__dataclass_fields__
            }
            | {
                "target_kind": "real",
                "control_effect_class": "read_only",
                "capture_owner": "exclusive_realsense_resource_owner",
                "reference_frame_owner": "deployment_station_frame_binding",
                "selection_authority": "none",
                "downstream_control_effects": "forbidden",
            },
            domain=domain,
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
class StationBoundPointGraspRegistration:
    profile: StationBoundPointGraspProfile
    skill: SkillRegistrationTemplate
    binding: StationBoundPointGraspBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str


def build_station_bound_point_grasp_registration(
    *,
    aligned_rgbd_source_catalog: tuple[dict[str, object], ...],
    source_binding_hash: str,
    frame_binding_hash: str,
    frame_graph_hash: str,
    source_frame_id: str,
    source_frame_spec_hash: str,
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    backend_id: str,
    model_profile_id: str,
    model_profile_hash: str,
    inference_implementation_hash: str,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    current_mode: str = "real_observe",
    arm_reachability_profile: StationPointGraspArmReachabilityProfile | None = None,
) -> StationBoundPointGraspRegistration:
    """Build one closed, proposal-only live point-grasp registration."""

    aligned_sources = tuple(
        (
            str(item["source_id"]),
            str(item["role"]),
            int(item["native_width"]),
            int(item["native_height"]),
        )
        for item in aligned_rgbd_source_catalog
    )
    profile = StationBoundPointGraspProfile(
        aligned_rgbd_sources=aligned_sources,
        source_binding_hash=source_binding_hash,
        frame_binding_hash=frame_binding_hash,
        frame_graph_hash=frame_graph_hash,
        source_frame_id=source_frame_id,
        source_frame_spec_hash=source_frame_spec_hash,
        reference_frame_id=reference_frame_id,
        reference_frame_spec_hash=reference_frame_spec_hash,
        backend_id=backend_id,
        model_profile_id=model_profile_id,
        model_profile_hash=model_profile_hash,
        inference_implementation_hash=inference_implementation_hash,
        arm_reachability_profile=arm_reachability_profile,
        profile_id=(
            "real-live-station-bound-dmtac-m-point-grasp-v3"
            if arm_reachability_profile is not None
            else "real-live-station-bound-dmtac-m-point-grasp-v2"
        ),
    )
    if arm_reachability_profile is None:
        definition = plan_point_grasp_skill_definition_v2()
        schema = point_grasp_schema_entry_v2(
            skill_definition_hash=definition.semantic_content_hash,
            parameter_profile_id=profile.profile_id,
            parameter_profile_hash=profile.semantic_content_hash,
            roles=tuple(profile.source_by_role),
            max_candidates=profile.max_candidates,
            backend_id=profile.backend_id,
        )
    else:
        definition = plan_point_grasp_skill_definition_v3()
        reachability = arm_reachability_profile
        schema = point_grasp_schema_entry_v3(
            skill_definition_hash=definition.semantic_content_hash,
            parameter_profile_id=profile.profile_id,
            parameter_profile_hash=profile.semantic_content_hash,
            roles=tuple(profile.source_by_role),
            max_candidates=profile.max_candidates,
            backend_id=profile.backend_id,
            arm_side=reachability.arm_side,
            evaluator_profile_id=reachability.evaluator_profile_id,
            evaluator_profile_hash=reachability.evaluator_profile_hash,
            candidate_reference_frame_id=profile.reference_frame_id,
            solver_reference_frame_id=reachability.solver_reference_frame_id,
            solver_reference_frame_spec_hash=(
                reachability.solver_reference_frame_spec_hash
            ),
            frame_graph_hash=reachability.frame_graph_hash,
            time_budget_ms=reachability.time_budget_ms,
            max_solve_cycles=reachability.max_solve_cycles,
            max_measured_qpos_age_ms=reachability.max_measured_qpos_age_ms,
            standard_position_tolerance_m=(reachability.standard_position_tolerance_m),
            standard_orientation_tolerance_deg=(
                reachability.standard_orientation_tolerance_deg
            ),
            basic_position_tolerance_m=reachability.basic_position_tolerance_m,
            basic_orientation_tolerance_deg=(
                reachability.basic_orientation_tolerance_deg
            ),
        )
    io_contract = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        input_artifact_slots=(),
        output_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2,
                direction="output",
                min_count=1,
                max_count=1,
                allowed_media_types=(LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,),
                allowed_schema_ids=(POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,),
                allowed_encodings=(LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,),
                allowed_compressions=(LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_annotated_bytes,
                max_total_bytes=profile.max_annotated_bytes,
            ),
            SkillArtifactSlotSpec(
                slot_id=REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID,
                direction="output", min_count=1 if profile.backend_id == "graspgenx" else 0, max_count=1,
                allowed_media_types=(GRASP_RGBD_SNAPSHOT_MEDIA_TYPE,),
                allowed_schema_ids=(GRASP_RGBD_SNAPSHOT_SCHEMA_ID,),
                allowed_encodings=(GRASP_RGBD_SNAPSHOT_ENCODING,),
                allowed_compressions=("zip",),
                max_artifact_bytes=GRASP_RGBD_SNAPSHOT_MAX_BYTES,
                max_total_bytes=GRASP_RGBD_SNAPSHOT_MAX_BYTES,
            ),
        ),
    )
    if current_mode not in {"real_observe", "real_active"}:
        raise ValueError("point-grasp current_mode is unsupported")
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=REAL_STATION_POINT_GRASP_ROUTE_ID_V2,
        route_version=definition.skill_version,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=(current_mode,),
        target_kind="real",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=120_000,
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
    binding = StationBoundPointGraspBinding(
        deployment_binding_id="real.station.point_grasp:" + runtime_session_id,
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
        executor_id=(
            REAL_STATION_POINT_GRASP_EXECUTOR_ID_V3
            if arm_reachability_profile is not None
            else REAL_STATION_POINT_GRASP_EXECUTOR_ID_V2
        ),
        deployment_binding_version=definition.skill_version,
    )
    common_result_validation = {
        "source_by_role": profile.source_by_role,
        "dimensions_by_role": profile.dimensions_by_role,
        "source_binding_hash": profile.source_binding_hash,
        "frame_binding_hash": profile.frame_binding_hash,
        "frame_graph_hash": profile.frame_graph_hash,
        "source_frame_id": profile.source_frame_id,
        "source_frame_spec_hash": profile.source_frame_spec_hash,
        "reference_frame_id": profile.reference_frame_id,
        "reference_frame_spec_hash": profile.reference_frame_spec_hash,
        "backend_id": profile.backend_id,
        "model_profile_id": profile.model_profile_id,
        "model_profile_hash": profile.model_profile_hash,
        "inference_implementation_hash": profile.inference_implementation_hash,
        "max_candidates": profile.max_candidates,
    }
    if arm_reachability_profile is None:
        parameter_validator = lambda payload: validate_point_grasp_parameters_v2(
            payload,
            dimensions_by_role=profile.dimensions_by_role,
            max_candidates=profile.max_candidates,
        )
        result_validator = lambda payload: validate_point_grasp_result_v2(
            payload, **common_result_validation
        )
        result_for_parameters_validator = validate_point_grasp_result_for_parameters_v2
    else:
        reachability = arm_reachability_profile
        parameter_validator = lambda payload: validate_point_grasp_parameters_v3(
            payload,
            dimensions_by_role=profile.dimensions_by_role,
            max_candidates=profile.max_candidates,
        )
        result_validator = lambda payload: validate_point_grasp_result_v3(
            payload,
            **common_result_validation,
            arm_side=reachability.arm_side,
            evaluator_profile_id=reachability.evaluator_profile_id,
            evaluator_profile_hash=reachability.evaluator_profile_hash,
            solver_reference_frame_id=reachability.solver_reference_frame_id,
            solver_reference_frame_spec_hash=(
                reachability.solver_reference_frame_spec_hash
            ),
            time_budget_ms=reachability.time_budget_ms,
            max_solve_cycles=reachability.max_solve_cycles,
            max_measured_qpos_age_ms=reachability.max_measured_qpos_age_ms,
            standard_position_tolerance_m=(reachability.standard_position_tolerance_m),
            standard_orientation_tolerance_deg=(
                reachability.standard_orientation_tolerance_deg
            ),
            basic_position_tolerance_m=reachability.basic_position_tolerance_m,
            basic_orientation_tolerance_deg=(
                reachability.basic_orientation_tolerance_deg
            ),
        )
        result_for_parameters_validator = validate_point_grasp_result_for_parameters_v3
    validator = AgentSkillPayloadValidatorSet(
        validator_id="dexteleop.agent_skill.validator.station_point_grasp:"
        + runtime_session_id,
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=parameter_validator,
        validate_execution_result=result_validator,
        validate_execution_result_for_parameters=result_for_parameters_validator,
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    availability_hash = canonical_content_hash(
        {
            "status": "configured_requires_live_aligned_rgbd_and_verified_models",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "profile_hash": profile.semantic_content_hash,
        },
        domain=(
            "dexteleop.agent_skill.station_point_grasp_availability.v3"
            if arm_reachability_profile is not None
            else "dexteleop.agent_skill.station_point_grasp_availability.v2"
        ),
    )
    return StationBoundPointGraspRegistration(
        profile=profile,
        skill=skill,
        binding=binding,
        payload_validator=validator,
        availability_hash=availability_hash,
    )


__all__ = [
    "REAL_STATION_POINT_GRASP_EXECUTOR_ID_V2",
    "REAL_STATION_POINT_GRASP_EXECUTOR_ID_V3",
    "REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2",
    "REAL_STATION_POINT_GRASP_ROUTE_ID_V2",
    "StationBoundPointGraspBinding",
    "StationBoundPointGraspProfile",
    "StationBoundPointGraspRegistration",
    "StationPointGraspArmReachabilityProfile",
    "build_station_bound_point_grasp_registration",
]
