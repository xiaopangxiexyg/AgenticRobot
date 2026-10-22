"""Paired read-only planning and explicit execution contract for grasp approach.

The planning Route resolves one retained point-grasp proposal against the exact
lower target envelope, endpoint IK, production URDF/FCL scene, and full joint
path.  It cannot submit motion or HOLD.  A separate T2 motion Route may consume
the resulting fresh, session-local immutable plan exactly once.
"""

from __future__ import annotations

from contextlib import nullcontext
import numpy as np

import json
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from agent_gateway.protocol.contracts import AgentSkillOperation
from agent_gateway.protocol._contract_validation import thaw_json
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillDeploymentSafetyEnvelopeRef,
    SkillExecutionRoute,
)
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.service import AgentGatewayOperationOutcome
from agent_gateway.skills.motion.definition import (
    execute_grasp_approach_plan_skill_definition_v1,
)
from agent_gateway.skills.motion.schemas import (
    grasp_approach_plan_execution_schema_entry_v1,
    validate_agent_motion_route_schema_v1,
)
from agent_gateway.skills.perception.grasp_approach_plan_schemas import (
    grasp_approach_plan_schema_entry_v1,
    plan_grasp_approach_skill_definition_v1,
    validate_grasp_approach_plan_parameters_v1,
    validate_grasp_approach_plan_result_for_parameters_v1,
    validate_grasp_approach_plan_result_v1,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.grasp_rgbd_snapshot import GRASP_RGBD_SNAPSHOT_MAX_BYTES, decode_grasp_rgbd_snapshot
from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_MAX_BYTES, GRASP_MULTIVIEW_SCHEMA_ID, decode_grasp_multiview_snapshot_arrays,
)
from robot_motion.grasp_visible_scene import (
    PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER,
    ParallelJawSolidGeometry,
    VisibleGraspScene,
)
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilterResult,
)
from teleop_runtime.skill_deploy.motion_port import (
    REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD,
    RealAgentJointTrajectory,
    RealAgentMotionMeasuredQposCut,
    RealAgentMotionPort,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65_GRASP_PATH_PLANNING_TIME_BUDGET_SEC,
    RM65GraspTrajectoryPlanningError,
)

from .station_point_grasp_deployment import (
    StationBoundPointGraspRegistration, REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID,
)


def _bounded_start_check_value(value: object, *, depth: int = 0,
                               budget: list[int]) -> Any:
    """Copy bounded finite evidence before serialization or hashing."""
    budget[0] -= 1
    if budget[0] < 0 or depth > 6:
        raise ValueError("start-check evidence exceeds projection bounds")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > 512:
            raise ValueError("start-check evidence string is too long")
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("start-check evidence must be finite")
        return value
    if isinstance(value, np.ndarray):
        if value.size > 16 or value.ndim > 2:
            raise ValueError("start-check evidence array is too large")
        value = value.tolist()
    if isinstance(value, Mapping):
        if len(value) > 16:
            raise ValueError("start-check evidence object is too large")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError("start-check evidence key is invalid")
            result[key] = _bounded_start_check_value(item, depth=depth+1, budget=budget)
        return result
    if isinstance(value, (list, tuple)) and len(value) <= 16:
        return [_bounded_start_check_value(item, depth=depth+1, budget=budget) for item in value]
    raise ValueError("start-check evidence cannot be projected")


def _start_check_json(value: object) -> str | None:
    """Publish a projection, never let optional diagnostics replace a refusal.

    The digest covers the final projection including detail_scope, excluding
    detail_hash itself. It is not an identity for omitted lower metadata.
    """
    try:
        if not isinstance(value, Mapping):
            return None
        if any(not isinstance(value.get(key), bool) for key in ("safe", "collision_check_available")):
            return None
        projected = {key: _bounded_start_check_value(value.get(key), budget=[128]) for key in
                     ("safe", "collision_check_available", "reason", "minimum_distance_m")}
        if projected["reason"] is not None and not isinstance(projected["reason"], str):
            return None
        distance = projected["minimum_distance_m"]
        if distance is not None and (isinstance(distance, bool) or not isinstance(distance, (int, float))):
            return None
        metadata = value.get("metadata")
        if metadata is not None:
            try:
                if not isinstance(metadata, Mapping):
                    raise ValueError("start-check metadata must be an object")
                selected = {key: metadata[key] for key in (
                    "closest_geometry_names", "closest_geometry_ids", "nearest_observed",
                    "collision_check", "target_envelope") if key in metadata}
                projected["metadata"] = _bounded_start_check_value(selected, budget=[128])
                if len(selected) != len(metadata):
                    projected["details_omitted"] = True
            except Exception:
                projected["details_omitted"] = True
        projected["detail_scope"] = "published_projection"
        for _ in range(2):
            # Recompute after any trimming: only the final published preimage is hashed.
            digest = canonical_content_hash(projected,
                domain="dexteleop.grasp_start_check_projection.v1")
            encoded = json.dumps({**projected, "detail_hash": digest}, ensure_ascii=False,
                                 allow_nan=False, sort_keys=True, separators=(",", ":"))
            if len(encoded) <= 2048:
                return encoded
            projected.pop("metadata", None)
            projected["details_omitted"] = True
    except Exception:
        # Lower already decided the rejection. Encoding is optional evidence.
        return None
    return None


def build_grasp_visible_scene_reader(*, artifact_store: RecoverableAgentArtifactStore,
                                    frame_graph: CoordinateFrameGraph,
                                    execution_frame_id: str,
                                    captured_robot_observation_filter: (
                                        Callable[..., object] | None
                                    ) = None) -> Callable:
    """Resolve only the retained producer's hash-bound replay Artifact."""
    if (
        captured_robot_observation_filter is not None
        and not callable(captured_robot_observation_filter)
    ):
        raise TypeError("captured_robot_observation_filter must be callable or None")

    def read(operation: AgentSkillOperation, source: Mapping[str, Any]) -> VisibleGraspScene:
        bindings = [binding for binding in operation.output_artifact_bindings
                    if binding.slot_id == REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID]
        if len(bindings) != 1:
            raise ValueError("point-grasp has no unique RGB-D snapshot Artifact")
        artifact = bindings[0].artifact
        multiview = artifact.schema_id == GRASP_MULTIVIEW_SCHEMA_ID
        data = artifact_store.read_bytes(artifact, maximum=(
            GRASP_MULTIVIEW_MAX_BYTES if multiview else GRASP_RGBD_SNAPSHOT_MAX_BYTES))
        if multiview:
            views, metadata, scene_points, scene_labels = decode_grasp_multiview_snapshot_arrays(data)
            if scene_points is None or metadata.get("scene_frame") != "anchor_camera":
                raise ValueError("Multi-view plan requires its retained validated scene")
            selected_views = [a for a, m in views if m["role"] == source["role"]]
            if len(selected_views) != 1:
                raise ValueError("Multi-view scene anchor is missing")
            arrays = selected_views[0]
        else:
            arrays, metadata = decode_grasp_rgbd_snapshot(data)
            if not np.any(arrays["target_mask"] & (arrays["depth_m"] > 0)):
                raise ValueError("snapshot contains no observed metric target points")
        capture = metadata["capture"]
        # NPZ JSON arrays are lists; retained operations freeze them to tuples.
        # Compare JSON semantics here without changing the typed global encoder.
        if canonical_content_hash(thaw_json(capture), domain="grasp.snapshot.capture.v1") != canonical_content_hash(
            thaw_json(source), domain="grasp.snapshot.capture.v1"
        ):
            raise ValueError("snapshot and retained point-grasp capture differ")
        pose, _ = frame_graph.transform_pose_matrix(
            np.asarray(capture["T_reference_from_camera"]).reshape(4, 4),
            source_reference_frame_id=capture["reference_frame_id"],
            target_reference_frame_id=execution_frame_id,
        )
        capture_qpos = (
            None
            if arrays["left_qpos"].size == 0
            else {side: arrays[side + "_qpos"] for side in ("left", "right")}
        )
        if multiview:
            # Consume the exact filtered/labelled cloud that generated the
            # proposals. Rebuilding from anchor depth would lose wrist evidence.
            return VisibleGraspScene(
                points=scene_points @ pose[:3, :3].T + pose[:3, 3],
                target_mask=scene_labels, geometry=ParallelJawSolidGeometry(**metadata["solid_geometry"]),
                capture_qpos_by_side=capture_qpos)
        depth_m = arrays["depth_m"]
        capture_self_observation_evidence = None
        if captured_robot_observation_filter is not None and capture_qpos is not None:
            filtered = captured_robot_observation_filter(
                depth_m=depth_m,
                target_mask=arrays["target_mask"],
                intrinsics=capture["camera_intrinsics"],
                source_from_camera=np.asarray(
                    capture["T_source_from_camera"], dtype=float
                ).reshape(4, 4),
                capture_qpos_by_side=capture_qpos,
            )
            if not isinstance(filtered, RM65CapturedRobotObservationFilterResult):
                raise TypeError(
                    "captured_robot_observation_filter returned the wrong type"
                )
            depth_m = filtered.depth_m
            capture_self_observation_evidence = filtered.evidence
        return VisibleGraspScene.from_depth(
            depth_m=depth_m, target_mask=arrays["target_mask"],
            intrinsics=capture["camera_intrinsics"], reference_from_camera=pose,
            geometry=ParallelJawSolidGeometry(**metadata["solid_geometry"]),
            capture_qpos_by_side=capture_qpos,
            capture_self_observation_evidence=(
                capture_self_observation_evidence
            ),
            depth_filter=PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER,
        )
    return read

REAL_GRASP_APPROACH_PLAN_OPERATION_GRACE_SEC = 15.0
_GRASP_APPROACH_PLAN_OPERATION_GRACE_SEC = (
    REAL_GRASP_APPROACH_PLAN_OPERATION_GRACE_SEC
)
DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC = 120.0
_LOGGER = logging.getLogger(__name__)

REAL_AGENT_GRASP_APPROACH_PLAN_ROUTE_ID_V1 = "real.rm65.perception.plan_grasp_approach"
REAL_AGENT_GRASP_APPROACH_PLAN_EXECUTOR_ID_V1 = (
    "control-station.real-passive-grasp-approach-plan.v1"
)
REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1 = (
    "real.rm65.agent_skill_execute_grasp_approach_plan"
)
REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_EXECUTOR_ID_V1 = (
    "control-station.real-agent-execute-grasp-approach-plan.v1"
)
REAL_AGENT_GRASP_APPROACH_PLAN_PROFILE_ID_V1 = (
    "real-rm65-retained-point-grasp-approach-plan-v1"
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


PoseMatrix = tuple[tuple[float, ...], ...]


def _immutable_pose(value: object, *, field_name: str) -> PoseMatrix:
    to_list = getattr(value, "tolist", None)
    raw = to_list() if callable(to_list) else value
    if (
        isinstance(raw, (tuple, list))
        and len(raw) == 16
        and not any(isinstance(item, (tuple, list)) for item in raw)
    ):
        rows = tuple(tuple(raw[index : index + 4]) for index in range(0, 16, 4))
    elif (
        isinstance(raw, (tuple, list))
        and len(raw) == 4
        and all(isinstance(row, (tuple, list)) and len(row) == 4 for row in raw)
    ):
        rows = tuple(tuple(row) for row in raw)
    else:
        raise ValueError(f"{field_name} must be a finite 4x4 pose.")
    pose = tuple(
        tuple(_finite(item, field_name=field_name) for item in row) for row in rows
    )
    if any(
        abs(actual - expected) > 1e-9
        for actual, expected in zip(pose[3], (0.0, 0.0, 0.0, 1.0), strict=True)
    ):
        raise ValueError(f"{field_name} must be homogeneous.")
    return pose


@dataclass(frozen=True)
class RealGraspApproachPlanProfile:
    """Exact source, lower safety, frames, and lifetime for the paired Routes."""

    component_id: str
    side: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    execution_reference_frame_id: str
    execution_reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    convergence_profile_hash: str
    point_grasp_route_hash: str
    point_grasp_profile_hash: str
    point_grasp_deployment_binding_hash: str
    source_backend_id: str
    source_model_profile_id: str
    source_model_profile_hash: str
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...]
    max_candidates: int
    max_measured_qpos_age_ms: int
    min_speed_scale: float = 0.05
    max_speed_scale: float = 1.0
    plan_ttl_sec: float = 30.0
    parameter_profile_id: str = REAL_AGENT_GRASP_APPROACH_PLAN_PROFILE_ID_V1

    def __post_init__(self) -> None:
        for field_name in (
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "convergence_profile_hash",
            "point_grasp_route_hash",
            "point_grasp_profile_hash",
            "point_grasp_deployment_binding_hash",
            "source_backend_id",
            "source_model_profile_id",
            "source_model_profile_hash",
            "parameter_profile_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        refs = tuple(sorted(self.safety_envelope_refs, key=lambda item: item.object_id))
        if not refs or any(
            not isinstance(item, SkillDeploymentSafetyEnvelopeRef) for item in refs
        ):
            raise ValueError("Plan profile requires typed lower safety envelopes.")
        if not any(
            REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1 in item.allowed_route_ids
            for item in refs
        ):
            raise ValueError("No lower envelope authorizes the paired execution Route.")
        object.__setattr__(self, "safety_envelope_refs", refs)
        for field_name in ("max_candidates", "max_measured_qpos_age_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        minimum = _finite(self.min_speed_scale, field_name="min_speed_scale")
        maximum = _finite(self.max_speed_scale, field_name="max_speed_scale")
        ttl = _finite(self.plan_ttl_sec, field_name="plan_ttl_sec")
        if not 0.0 < minimum <= maximum <= 1.0:
            raise ValueError("speed scales must satisfy 0 < min <= max <= 1.")
        if not 0.0 < ttl <= 30.0:
            raise ValueError("plan_ttl_sec must be in (0, 30].")
        object.__setattr__(self, "min_speed_scale", minimum)
        object.__setattr__(self, "max_speed_scale", maximum)
        object.__setattr__(self, "plan_ttl_sec", ttl)

    @property
    def speed_scale_bounds(self) -> tuple[float, float]:
        return self.min_speed_scale, self.max_speed_scale

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **{
                    field_name: getattr(self, field_name)
                    for field_name in self.__dataclass_fields__
                    if field_name != "safety_envelope_refs"
                },
                "safety_envelope_refs": tuple(
                    item.to_wire() for item in self.safety_envelope_refs
                ),
                "planning_scope": (
                    "target_envelope_endpoint_ik_production_urdf_fcl_full_path"
                ),
                "planning_control_effects": "no_motion_no_hold",
                "execution_semantics": "explicit_single_plan_reference_only",
            },
            domain="dexteleop.agent_skill.real_grasp_approach_plan_profile.v1",
        )


@dataclass(frozen=True)
class RealGraspApproachPlanBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    point_grasp_deployment_binding_hash: str
    executor_id: str = REAL_AGENT_GRASP_APPROACH_PLAN_EXECUTOR_ID_V1
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
                "candidate_owner": "station_point_grasp_executor",
                "planning_owner": "rm65_production_urdf_fcl_planner",
                "downstream_motion_or_hold": "forbidden",
            },
            domain="dexteleop.agent_skill.real_grasp_approach_plan_deployment.v1",
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
class RealGraspApproachPlanRegistration:
    profile: RealGraspApproachPlanProfile
    skill: SkillRegistrationTemplate
    binding: RealGraspApproachPlanBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str
    execute_skill: SkillRegistrationTemplate


def build_real_grasp_approach_plan_registration(
    *,
    point_grasp_registration: StationBoundPointGraspRegistration,
    component_id: str,
    side: str,
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    execution_reference_frame_id: str,
    execution_reference_frame_spec_hash: str,
    frame_graph_hash: str,
    tool_config_hash: str,
    calibration_hash: str,
    convergence_profile_hash: str,
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...],
    max_measured_qpos_age_ms: int,
    min_speed_scale: float,
    max_speed_scale: float,
    execution_operation_timeout_sec: float = DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC,
) -> RealGraspApproachPlanRegistration:
    """Build both halves of the paired contract without runtime effects."""

    if not isinstance(point_grasp_registration, StationBoundPointGraspRegistration):
        raise TypeError("point_grasp_registration must be typed.")
    source_profile = point_grasp_registration.profile
    source_binding = point_grasp_registration.binding
    source_route = point_grasp_registration.skill.routes[0]
    execution_timeout_sec = _finite(
        execution_operation_timeout_sec,
        field_name="execution_operation_timeout_sec",
    )
    if execution_timeout_sec <= REAL_GRASP_APPROACH_PLAN_OPERATION_GRACE_SEC:
        raise ValueError(
            "execution_operation_timeout_sec must exceed the activation/cleanup grace."
        )
    profile = RealGraspApproachPlanProfile(
        component_id=component_id,
        side=side,
        reference_frame_id=reference_frame_id,
        reference_frame_spec_hash=reference_frame_spec_hash,
        execution_reference_frame_id=execution_reference_frame_id,
        execution_reference_frame_spec_hash=execution_reference_frame_spec_hash,
        frame_graph_hash=frame_graph_hash,
        tool_config_hash=tool_config_hash,
        calibration_hash=calibration_hash,
        convergence_profile_hash=convergence_profile_hash,
        point_grasp_route_hash=source_route.semantic_content_hash,
        point_grasp_profile_hash=source_profile.semantic_content_hash,
        point_grasp_deployment_binding_hash=source_binding.semantic_content_hash,
        source_backend_id=source_profile.backend_id,
        source_model_profile_id=source_profile.model_profile_id,
        source_model_profile_hash=source_profile.model_profile_hash,
        safety_envelope_refs=safety_envelope_refs,
        max_candidates=source_profile.max_candidates,
        max_measured_qpos_age_ms=max_measured_qpos_age_ms,
        min_speed_scale=min_speed_scale,
        max_speed_scale=max_speed_scale,
    )
    definition = plan_grasp_approach_skill_definition_v1(
        controlled_component_ids=(profile.component_id,)
    )
    schema = grasp_approach_plan_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        source_backend_id=profile.source_backend_id,
        source_model_profile_id=profile.source_model_profile_id,
        source_model_profile_hash=profile.source_model_profile_hash,
        max_candidates=profile.max_candidates,
        min_speed_scale=profile.min_speed_scale,
        max_speed_scale=profile.max_speed_scale,
    )
    io_contract = inline_only_skill_io_contract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=REAL_AGENT_GRASP_APPROACH_PLAN_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=source_route.supported_modes,
        target_kind="real",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        # The route deadline must outlive the planner's own bounded search so a
        # budget exhaustion is returned as a read-only candidate rejection.
        max_operation_duration_ms=int(
            math.ceil(
                (
                    RM65_GRASP_PATH_PLANNING_TIME_BUDGET_SEC
                    + _GRASP_APPROACH_PLAN_OPERATION_GRACE_SEC
                )
                * 1000.0
            )
        ),
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
    binding = RealGraspApproachPlanBinding(
        deployment_binding_id=(
            "real.rm65.grasp_approach_plan:" + source_binding.runtime_session_id
        ),
        route_hash=route.semantic_content_hash,
        runtime_session_id=source_binding.runtime_session_id,
        control_domain_id=source_binding.control_domain_id,
        target_binding_id=source_binding.target_binding_id,
        station_manifest_hash=source_binding.station_manifest_hash,
        execution_manifest_hash=source_binding.execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
        point_grasp_deployment_binding_hash=source_binding.semantic_content_hash,
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id=(
            "dexteleop.agent_skill.validator.real_grasp_approach_plan:"
            + source_binding.runtime_session_id
        ),
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda payload: validate_grasp_approach_plan_parameters_v1(
            payload,
            max_candidates=profile.max_candidates,
            speed_scale_bounds=profile.speed_scale_bounds,
        ),
        validate_execution_result=lambda payload: (
            validate_grasp_approach_plan_result_v1(
                payload,
                component_id=profile.component_id,
                side=profile.side,
                reference_frame_id=profile.reference_frame_id,
                reference_frame_spec_hash=profile.reference_frame_spec_hash,
                execution_reference_frame_id=profile.execution_reference_frame_id,
                execution_reference_frame_spec_hash=(
                    profile.execution_reference_frame_spec_hash
                ),
                frame_graph_hash=profile.frame_graph_hash,
                tool_config_hash=profile.tool_config_hash,
                calibration_hash=profile.calibration_hash,
                source_backend_id=profile.source_backend_id,
                source_model_profile_id=profile.source_model_profile_id,
                source_model_profile_hash=profile.source_model_profile_hash,
                max_candidates=profile.max_candidates,
                speed_scale_bounds=profile.speed_scale_bounds,
            )
        ),
        validate_execution_result_for_parameters=(
            validate_grasp_approach_plan_result_for_parameters_v1
        ),
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    availability_hash = canonical_content_hash(
        {
            "status": (
                "configured_requires_retained_point_grasp_fresh_qpos_and_"
                "production_urdf_fcl_planner"
            ),
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "profile_hash": profile.semantic_content_hash,
            "motion_or_hold_on_rejection": False,
        },
        domain="dexteleop.agent_skill.real_grasp_approach_plan_availability.v1",
    )

    execute_definition = execute_grasp_approach_plan_skill_definition_v1(
        controlled_component_ids=(profile.component_id,)
    )
    execute_schema = grasp_approach_plan_execution_schema_entry_v1(
        skill_definition_hash=execute_definition.semantic_content_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_component_ids=(profile.component_id,),
    )
    execute_io = inline_only_skill_io_contract(
        skill_id=execute_definition.skill_id,
        skill_version=execute_definition.skill_version,
        skill_definition_hash=execute_definition.semantic_content_hash,
        parameter_schema_hash=execute_schema.parameter_schema_hash,
        execution_result_schema_hash=execute_schema.execution_result_schema_hash,
        task_evaluation_schema_hash=execute_schema.task_evaluation_schema_hash,
    )
    execute_route = SkillExecutionRoute(
        skill_id=execute_definition.skill_id,
        skill_version=execute_definition.skill_version,
        skill_definition_hash=execute_definition.semantic_content_hash,
        route_id=REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=execute_schema.parameter_schema_hash,
        execution_result_schema_hash=execute_schema.execution_result_schema_hash,
        task_evaluation_schema_hash=execute_schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=execute_schema.parameter_profiles,
        control_effect_class="motion",
        privilege_tier="t2_real_motion",
        supported_modes=source_route.supported_modes,
        target_kind="real",
        may_move_robot=True,
        may_open_publication=False,
        requires_operator_confirmation=True,
        requires_deadman=False,
        max_operation_duration_ms=int(math.ceil(execution_timeout_sec * 1000.0)),
        cancellation_policy="hold_on_cancel_disconnect_or_timeout",
        requires_task_evaluation=False,
        skill_io_contract_hash=execute_io.semantic_content_hash,
    )
    validate_agent_motion_route_schema_v1(
        route=execute_route,
        schema_entry=execute_schema,
    )
    execute_skill = SkillRegistrationTemplate(
        definition=execute_definition,
        schema_entry=execute_schema,
        io_contract=execute_io,
        routes=(execute_route,),
    )
    return RealGraspApproachPlanRegistration(
        profile=profile,
        skill=skill,
        binding=binding,
        payload_validator=validator,
        availability_hash=availability_hash,
        execute_skill=execute_skill,
    )


@dataclass(frozen=True)
class RealGraspApproachPlan:
    """One content-addressed plan retained only inside its Gateway session."""

    runtime_session_id: str
    profile_hash: str
    planning_operation_id: str
    planning_request_hash: str
    point_grasp_operation_id: str
    point_grasp_request_hash: str
    point_grasp_result_hash: str
    point_grasp_candidate_hash: str
    source_station_revision: int
    component_id: str
    side: str
    target_pose_reference: PoseMatrix
    target_pose_execution: PoseMatrix
    speed_scale: float
    measured_qpos_revision: int
    measured_qpos_hash: str
    measured_qpos_at: float
    trajectory: RealAgentJointTrajectory
    created_at: float
    expires_at: float
    evidence_refs: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_session_id",
            "profile_hash",
            "planning_operation_id",
            "planning_request_hash",
            "point_grasp_operation_id",
            "point_grasp_request_hash",
            "point_grasp_result_hash",
            "point_grasp_candidate_hash",
            "component_id",
            "measured_qpos_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        for field_name in ("source_station_revision", "measured_qpos_revision"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
        if not isinstance(self.trajectory, RealAgentJointTrajectory):
            raise TypeError("trajectory must be a RealAgentJointTrajectory.")
        reference = _immutable_pose(
            self.target_pose_reference,
            field_name="target_pose_reference",
        )
        execution = _immutable_pose(
            self.target_pose_execution,
            field_name="target_pose_execution",
        )
        speed = _finite(self.speed_scale, field_name="speed_scale")
        measured = _finite(self.measured_qpos_at, field_name="measured_qpos_at")
        created = _finite(self.created_at, field_name="created_at")
        expires = _finite(self.expires_at, field_name="expires_at")
        if not 0.0 < speed <= 1.0:
            raise ValueError("speed_scale must satisfy 0 < speed_scale <= 1.")
        if expires <= created or measured > created:
            raise ValueError("Plan timestamps are inconsistent.")
        refs = tuple(
            sorted(
                {
                    _required(item, field_name="evidence_ref")
                    for item in self.evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("Plan requires immutable evidence refs.")
        object.__setattr__(self, "target_pose_reference", reference)
        object.__setattr__(self, "target_pose_execution", execution)
        object.__setattr__(self, "speed_scale", speed)
        object.__setattr__(self, "measured_qpos_at", measured)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "runtime_session_id": self.runtime_session_id,
                    "profile_hash": self.profile_hash,
                    "planning_operation_id": self.planning_operation_id,
                    "planning_request_hash": self.planning_request_hash,
                    "point_grasp_operation_id": self.point_grasp_operation_id,
                    "point_grasp_request_hash": self.point_grasp_request_hash,
                    "point_grasp_result_hash": self.point_grasp_result_hash,
                    "point_grasp_candidate_hash": self.point_grasp_candidate_hash,
                    "source_station_revision": self.source_station_revision,
                    "component_id": self.component_id,
                    "side": self.side,
                    "target_pose_reference": reference,
                    "target_pose_execution": execution,
                    "speed_scale": speed,
                    "measured_qpos_revision": self.measured_qpos_revision,
                    "measured_qpos_hash": self.measured_qpos_hash,
                    "measured_qpos_at": measured,
                    "trajectory_hash": self.trajectory.semantic_content_hash,
                    "created_at": created,
                    "expires_at": expires,
                    "evidence_refs": refs,
                },
                domain="dexteleop.real_agent_motion.grasp_approach_plan.v1",
            ),
        )

    @property
    def plan_id(self) -> str:
        return "real-grasp-approach-plan:" + self.semantic_content_hash

    @property
    def all_evidence_refs(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *self.evidence_refs,
                    *self.trajectory.evidence_refs,
                    self.plan_id,
                }
            )
        )


class RealGraspApproachPlanLookupError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = _required(reason_code, field_name="reason_code")
        super().__init__(self.reason_code)


@dataclass
class _StoredPlan:
    plan: RealGraspApproachPlan
    reserved_by_operation_id: str | None = None
    consumed: bool = False


class BoundedRealGraspApproachPlanStore:
    """Bounded one-session plan store with one-operation replay fencing."""

    def __init__(self, *, max_retained_plans: int) -> None:
        if (
            isinstance(max_retained_plans, bool)
            or not isinstance(max_retained_plans, int)
            or max_retained_plans <= 0
        ):
            raise ValueError("max_retained_plans must be positive.")
        self.max_retained_plans = max_retained_plans
        self._lock = threading.RLock()
        self._plans: OrderedDict[str, _StoredPlan] = OrderedDict()
        self._closed = False

    def _prune(self, *, now: float) -> None:
        expired = [
            plan_id
            for plan_id, stored in self._plans.items()
            if stored.plan.expires_at <= now
            and (stored.reserved_by_operation_id is None or stored.consumed)
        ]
        for plan_id in expired:
            self._plans.pop(plan_id, None)

    def put(self, plan: RealGraspApproachPlan, *, now: float) -> None:
        if not isinstance(plan, RealGraspApproachPlan):
            raise TypeError("plan must be a RealGraspApproachPlan.")
        timestamp = _finite(now, field_name="now")
        with self._lock:
            if self._closed:
                raise RuntimeError("Grasp-approach plan store is closed.")
            self._prune(now=timestamp)
            current = self._plans.get(plan.plan_id)
            if current is not None:
                if current.plan.semantic_content_hash != plan.semantic_content_hash:
                    raise RuntimeError("Plan ID collision detected.")
                return
            while len(self._plans) >= self.max_retained_plans:
                removable = next(
                    (
                        plan_id
                        for plan_id, stored in self._plans.items()
                        if stored.reserved_by_operation_id is None or stored.consumed
                    ),
                    None,
                )
                if removable is None:
                    raise RuntimeError(
                        "All retained grasp-approach plans are reserved."
                    )
                self._plans.pop(removable)
            self._plans[plan.plan_id] = _StoredPlan(plan=plan)

    def reserve(
        self,
        *,
        plan_id: str,
        plan_hash: str,
        operation_id: str,
        runtime_session_id: str,
        profile_hash: str,
        component_id: str,
        expected_station_revision: int,
        now: float,
    ) -> RealGraspApproachPlan:
        requested_id = _required(plan_id, field_name="plan_id")
        requested_hash = _required(plan_hash, field_name="plan_hash")
        requested_operation = _required(operation_id, field_name="operation_id")
        timestamp = _finite(now, field_name="now")
        with self._lock:
            if self._closed:
                raise RealGraspApproachPlanLookupError("grasp_path_plan_store_closed")
            self._prune(now=timestamp)
            stored = self._plans.get(requested_id)
            if stored is None:
                raise RealGraspApproachPlanLookupError("grasp_path_plan_not_found")
            plan = stored.plan
            if (
                requested_hash != plan.semantic_content_hash
                or requested_id != plan.plan_id
            ):
                raise RealGraspApproachPlanLookupError("grasp_path_plan_hash_mismatch")
            if timestamp >= plan.expires_at:
                raise RealGraspApproachPlanLookupError("grasp_path_plan_expired")
            if plan.runtime_session_id != runtime_session_id:
                raise RealGraspApproachPlanLookupError(
                    "grasp_path_plan_session_mismatch"
                )
            if plan.profile_hash != profile_hash:
                raise RealGraspApproachPlanLookupError(
                    "grasp_path_plan_profile_mismatch"
                )
            if plan.component_id != component_id:
                raise RealGraspApproachPlanLookupError(
                    "grasp_path_plan_component_mismatch"
                )
            if plan.source_station_revision != expected_station_revision:
                raise RealGraspApproachPlanLookupError(
                    "grasp_path_plan_station_changed"
                )
            if stored.consumed:
                raise RealGraspApproachPlanLookupError(
                    "grasp_path_plan_already_consumed"
                )
            if stored.reserved_by_operation_id not in {None, requested_operation}:
                raise RealGraspApproachPlanLookupError(
                    "grasp_path_plan_already_reserved"
                )
            stored.reserved_by_operation_id = requested_operation
            self._plans.move_to_end(requested_id)
            return plan

    def consume(self, *, plan_id: str, operation_id: str) -> None:
        requested_id = _required(plan_id, field_name="plan_id")
        requested_operation = _required(operation_id, field_name="operation_id")
        with self._lock:
            stored = self._plans.get(requested_id)
            if stored is None or stored.reserved_by_operation_id != requested_operation:
                raise RealGraspApproachPlanLookupError(
                    "grasp_path_plan_reservation_mismatch"
                )
            if stored.consumed:
                raise RealGraspApproachPlanLookupError(
                    "grasp_path_plan_already_consumed"
                )
            stored.consumed = True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._plans.clear()


class RealGraspApproachPlanProvider:
    """Resolve retained candidates through the exact production path planner."""

    def __init__(
        self,
        *,
        registration: RealGraspApproachPlanRegistration,
        point_grasp_registration: StationBoundPointGraspRegistration,
        point_grasp_executor: PassiveReadSkillExecutor,
        motion_port: RealAgentMotionPort,
        frame_graph: CoordinateFrameGraph,
        grasp_trajectory_planner: Callable[..., object],
        plan_store: BoundedRealGraspApproachPlanStore,
        runtime_session_id: str,
        current_station_revision: Callable[[], int],
        clock: Callable[[], float] = time.time,
        visible_scene_reader: Callable | None = None,
        non_motion_read_window: Callable[[], Any] | None = None,
    ) -> None:
        if not isinstance(registration, RealGraspApproachPlanRegistration):
            raise TypeError("registration must be typed.")
        if not isinstance(point_grasp_registration, StationBoundPointGraspRegistration):
            raise TypeError("point_grasp_registration must be typed.")
        if not isinstance(point_grasp_executor, PassiveReadSkillExecutor):
            raise TypeError("point_grasp_executor must be typed.")
        if not isinstance(motion_port, RealAgentMotionPort):
            raise TypeError("motion_port must be typed.")
        if not isinstance(frame_graph, CoordinateFrameGraph):
            raise TypeError("frame_graph must be typed.")
        if not callable(grasp_trajectory_planner):
            raise TypeError("grasp_trajectory_planner must be callable.")
        if not isinstance(plan_store, BoundedRealGraspApproachPlanStore):
            raise TypeError("plan_store must be typed.")
        if not callable(current_station_revision) or not callable(clock):
            raise TypeError("station revision and clock providers must be callable.")
        session_id = _required(runtime_session_id, field_name="runtime_session_id")
        profile = registration.profile
        source_route = point_grasp_registration.skill.routes[0]
        if session_id != registration.binding.runtime_session_id:
            raise ValueError("Plan runtime session differs from registration.")
        if (
            profile.point_grasp_route_hash != source_route.semantic_content_hash
            or profile.point_grasp_profile_hash
            != point_grasp_registration.profile.semantic_content_hash
            or profile.point_grasp_deployment_binding_hash
            != point_grasp_registration.binding.semantic_content_hash
        ):
            raise ValueError("Plan Route is bound to another point-grasp contract.")
        if (
            point_grasp_executor.executor_id
            != point_grasp_registration.binding.executor_id
        ):
            raise ValueError("Plan provider received another point-grasp executor.")
        if frame_graph.semantic_content_hash != profile.frame_graph_hash:
            raise ValueError("Plan provider frame graph differs from registration.")
        self.registration = registration
        self.point_grasp_registration = point_grasp_registration
        self.point_grasp_executor = point_grasp_executor
        if visible_scene_reader is not None and not callable(visible_scene_reader):
            raise TypeError("visible_scene_reader must be callable or None")
        if (
            non_motion_read_window is not None
            and not callable(non_motion_read_window)
        ):
            raise TypeError("non_motion_read_window must be callable or None")
        self.visible_scene_reader = visible_scene_reader
        self.non_motion_read_window = non_motion_read_window
        self.motion_port = motion_port
        self.frame_graph = frame_graph
        self.grasp_trajectory_planner = grasp_trajectory_planner
        self.plan_store = plan_store
        self.runtime_session_id = session_id
        self._current_station_revision = current_station_revision
        self._clock = clock

    def _retained_candidate(
        self,
        *,
        operation_id: str,
        ordinal: int,
        expected_station_revision: int,
    ) -> tuple[AgentSkillOperation, Mapping[str, Any], Mapping[str, Any]]:
        completion = self.point_grasp_executor.completion_for(operation_id)
        if completion is None:
            raise ValueError("Point-grasp operation is not retained by this session.")
        if not isinstance(completion, AgentGatewayOperationOutcome):
            raise TypeError("Retained point-grasp completion has an invalid type.")
        operation = completion.operation
        if not completion.accepted or not isinstance(operation, AgentSkillOperation):
            raise ValueError("Retained point-grasp completion was not accepted.")
        if operation.runtime_session_id != self.runtime_session_id:
            raise ValueError("Point-grasp operation belongs to another session.")
        if (
            operation.status != "completed"
            or operation.execution_outcome != "completed"
        ):
            raise ValueError("Point-grasp operation did not complete successfully.")
        result = operation.execution_result
        if not isinstance(result, Mapping):
            raise TypeError("Point-grasp operation has no retained result mapping.")
        self.point_grasp_registration.payload_validator.validate_execution_result(
            result
        )
        if result["station_revision"] != expected_station_revision:
            raise ValueError("Point-grasp result belongs to another Station revision.")
        candidates = result["candidates"]
        if not isinstance(candidates, (tuple, list)) or ordinal >= len(candidates):
            raise ValueError("candidate_ordinal is absent from the retained result.")
        candidate = candidates[ordinal]
        if not isinstance(candidate, Mapping) or candidate.get("ordinal") != ordinal:
            raise ValueError("Retained point-grasp candidate ordering is inconsistent.")
        return operation, result, candidate

    def _qpos_problem(
        self,
        cut: RealAgentMotionMeasuredQposCut | None,
        *,
        now: float,
    ) -> str | None:
        if cut is None:
            return "unavailable"
        if cut.measured_at > now or cut.available_at > now:
            return "from_future"
        if (
            now - cut.measured_at
        ) * 1_000.0 > self.registration.profile.max_measured_qpos_age_ms:
            return "stale"
        return None

    def _postplan_measured_qpos(
        self,
    ) -> tuple[RealAgentMotionMeasuredQposCut | None, float, str | None]:
        """Read the first fresh control-loop cut after a non-motion window.

        ``non_motion_read_window`` resumes the lower RM65 sampler before its
        context exits, but the control loop publishes that sample to the motion
        port asynchronously.  Reading the port immediately after a long
        planning call can therefore observe the same cut used at admission and
        misclassify the planning duration as measurement age.  Wait only for
        this publication hand-off; the ordinary freshness and start-state
        drift checks remain unchanged.
        """

        cut = self.motion_port.latest_measured_qpos()
        now = float(self._clock())
        problem = self._qpos_problem(cut, now=now)
        if (
            self.non_motion_read_window is None
            or problem not in {"unavailable", "stale"}
        ):
            return cut, now, problem

        deadline = time.monotonic() + (
            self.registration.profile.max_measured_qpos_age_ms / 1_000.0
        )
        while problem in {"unavailable", "stale"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.002, remaining))
            cut = self.motion_port.latest_measured_qpos()
            now = float(self._clock())
            problem = self._qpos_problem(cut, now=now)
        return cut, now, problem

    @staticmethod
    def _diagnostics(exc: RM65GraspTrajectoryPlanningError) -> dict[str, Any]:
        details = dict(exc.details)
        nested_failure = details.get("last_dense_revalidation_error")
        if isinstance(nested_failure, Mapping):
            nested_details = nested_failure.get("details")
            if isinstance(nested_details, Mapping):
                details.update(nested_details)
        for source_field, public_field in (
            ("solve_cycles", "cycles"),
            ("position_residual_m", "best_position_residual_m"),
            ("orientation_residual_deg", "best_orientation_residual_deg"),
        ):
            if public_field not in details and source_field in details:
                details[public_field] = details[source_field]
        diagnostics: dict[str, Any] = {}
        for field_name in (
            "cycles",
            "best_position_residual_m",
            "best_orientation_residual_deg",
            "modified_sides",
            "segment_index",
            "edge_sample_index",
            "exception_type",
            "envelope_projection_by_side",
            "envelope_ingress_rejection",
            "envelope_projection_subject",
            "maximum_path_envelope_ingress_distance_m",
            "path_envelope_ingress_distance_excess_m",
        ):
            if field_name in details:
                value = details[field_name]
                diagnostics[field_name] = (
                    list(value) if field_name == "modified_sides" else value
                )
        path = details.get("path")
        if isinstance(path, Mapping) and isinstance(path.get("status"), str):
            diagnostics["path_status"] = path["status"]
        source = details.get("start_qpos_source")
        if isinstance(source, str) and source in {"input_qpos", "envelope_ingress_endpoint"}:
            diagnostics["start_qpos_source"] = source
        encoded_start_check = _start_check_json(details.get("start_check"))
        if encoded_start_check is not None:
            diagnostics["start_check_json"] = encoded_start_check
        return diagnostics

    def _base_result(
        self,
        *,
        source_operation: AgentSkillOperation,
        source_result: Mapping[str, Any],
        candidate: Mapping[str, Any],
        ordinal: int,
        speed_scale: float,
        target_reference: PoseMatrix,
        target_execution: PoseMatrix,
        qpos_cut: RealAgentMotionMeasuredQposCut | None,
        now: float,
        planner_evidence_refs: tuple[str, ...],
    ) -> dict[str, Any]:
        profile = self.registration.profile
        result_hash = canonical_content_hash(
            source_result,
            domain="dexteleop.agent_gateway.passive_read_result.v1",
        )
        candidate_hash = canonical_content_hash(
            candidate,
            domain="dexteleop.agent_skill.station_point_grasp_candidate.v2",
        )
        return {
            "point_grasp_operation_id": source_operation.operation_id,
            "point_grasp_request_hash": source_operation.request_hash,
            "point_grasp_result_hash": result_hash,
            "point_grasp_candidate_hash": candidate_hash,
            "source_station_revision": source_result["station_revision"],
            "source_backend_id": source_result["backend_id"],
            "source_model_profile_id": source_result["model_profile_id"],
            "source_model_profile_hash": source_result["model_profile_hash"],
            "candidate_ordinal": ordinal,
            "candidate_score": float(candidate["score"]),
            "component_id": profile.component_id,
            "side": profile.side,
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "execution_reference_frame_id": profile.execution_reference_frame_id,
            "execution_reference_frame_spec_hash": (
                profile.execution_reference_frame_spec_hash
            ),
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            "T_reference_from_tool_tcp": [
                value for row in target_reference for value in row
            ],
            "T_execution_from_tool_tcp": [
                value for row in target_execution for value in row
            ],
            "speed_scale": speed_scale,
            "measured_qpos_revision": None if qpos_cut is None else qpos_cut.revision,
            "measured_qpos_hash": (
                None if qpos_cut is None else qpos_cut.semantic_content_hash
            ),
            "measured_qpos_at": None if qpos_cut is None else qpos_cut.measured_at,
            "measured_qpos_age_ms": (
                None
                if qpos_cut is None
                else max(0.0, (now - qpos_cut.measured_at) * 1_000.0)
            ),
            "planner_evidence_refs": sorted(set(planner_evidence_refs)),
            "execution_requires_explicit_motion_skill": True,
            "motion_or_hold_performed": False,
        }

    def _rejected_result(
        self,
        *,
        base: Mapping[str, Any],
        reason_code: str,
        diagnostics: Mapping[str, Any] = MappingProxyType({}),
    ) -> dict[str, Any]:
        return {
            **base,
            "planning_status": "rejected",
            "reason_code": reason_code,
            "grasp_approach_plan_id": None,
            "grasp_approach_plan_hash": None,
            "trajectory_hash": None,
            "trajectory_sample_count": None,
            "trajectory_duration_sec": None,
            "plan_created_at": None,
            "plan_expires_at": None,
            "eligible_for_execution": False,
            "diagnostics": dict(diagnostics),
        }

    def plan_request(
        self,
        request: Any,
        operation: AgentSkillOperation,
    ) -> tuple[dict[str, Any], RealGraspApproachPlan | None]:
        if request.input_artifact_bindings:
            raise ValueError("Grasp-approach planning accepts no input Artifacts.")
        expected_revision = request.expected_station_revision
        source_operation, source_result, candidate = self._retained_candidate(
            operation_id=str(request.parameters["point_grasp_operation_id"]),
            ordinal=int(request.parameters["candidate_ordinal"]),
            expected_station_revision=expected_revision,
        )
        profile = self.registration.profile
        ordinal = int(request.parameters["candidate_ordinal"])
        speed_scale = float(request.parameters["speed_scale"])
        target_reference = _immutable_pose(
            candidate["T_reference_from_tool_tcp"],
            field_name="T_reference_from_tool_tcp",
        )
        target_execution, resolution = self.frame_graph.transform_pose_matrix(
            target_reference,
            source_reference_frame_id=profile.reference_frame_id,
            target_reference_frame_id=profile.execution_reference_frame_id,
        )
        target_execution = _immutable_pose(
            target_execution,
            field_name="T_execution_from_tool_tcp",
        )
        qpos_cut = self.motion_port.latest_measured_qpos()
        # Timestamp the acquired cut, not the start of its potentially blocked
        # read.  A concurrent sampler can publish a newer cut during that read.
        now = float(self._clock())
        qpos_problem = self._qpos_problem(qpos_cut, now=now)
        source_evidence = tuple(source_operation.execution_receipt_refs)
        rejection_seed = {
            "source_operation_id": source_operation.operation_id,
            "candidate_ordinal": ordinal,
            "expected_station_revision": expected_revision,
            "qpos_problem": qpos_problem,
            "frame_resolution_hash": resolution.semantic_content_hash,
        }
        rejection_ref = "rm65-grasp-path-rejected:" + canonical_content_hash(
            rejection_seed,
            domain="dexteleop.real_agent_motion.grasp_path_rejection.v1",
        )
        base = self._base_result(
            source_operation=source_operation,
            source_result=source_result,
            candidate=candidate,
            ordinal=ordinal,
            speed_scale=speed_scale,
            target_reference=target_reference,
            target_execution=target_execution,
            qpos_cut=qpos_cut,
            now=now,
            planner_evidence_refs=tuple(sorted({*source_evidence, rejection_ref})),
        )
        if qpos_problem is not None:
            return (
                self._rejected_result(
                    base=base,
                    reason_code="grasp_path_measured_qpos_" + qpos_problem,
                ),
                None,
            )
        assert qpos_cut is not None
        if self._current_station_revision() != expected_revision:
            return (
                self._rejected_result(
                    base=base,
                    reason_code="grasp_path_station_revision_changed",
                ),
                None,
            )
        try:
            planning_window = (
                self.non_motion_read_window()
                if self.non_motion_read_window is not None
                else nullcontext()
            )
            with planning_window:
                # Scene reconstruction may run the captured-robot URDF/FCL
                # filter and is part of the same blocking, read-only planning
                # operation as trajectory search.  Enter the window before
                # this work; otherwise the force stale watchdog can trip while
                # the planner has not even received its scene yet.
                visible_scene = None
                if source_result["backend_id"] == "graspgenx":
                    if self.visible_scene_reader is None:
                        raise RM65GraspTrajectoryPlanningError(
                            "grasp_scene_snapshot_unavailable"
                        )
                    try:
                        visible_scene = self.visible_scene_reader(
                            source_operation, source_result
                        )
                        if not isinstance(visible_scene, VisibleGraspScene):
                            raise TypeError(
                                "snapshot reader did not return a typed visible scene"
                            )
                    except (ValueError, TypeError, KeyError, OSError) as error:
                        raise RM65GraspTrajectoryPlanningError(
                            "grasp_scene_snapshot_invalid",
                            details={
                                "exception_type": type(error).__name__,
                                # Retain a bounded cause in logs, not new v1 fields.
                                "exception_message": " ".join(
                                    str(error).split()
                                )[:512],
                            },
                        ) from error
                trajectory = self.grasp_trajectory_planner(
                    qpos_by_side=qpos_cut.qpos_by_side,
                    side=profile.side,
                    target_tool_tcp_world=target_execution,
                    speed_scale=speed_scale,
                    **(
                        {"visible_scene": visible_scene}
                        if visible_scene is not None
                        else {}
                    ),
                )
        except RM65GraspTrajectoryPlanningError as exc:
            # The published v1 diagnostics object is intentionally bounded.
            # Preserve every new lower-stage rejection detail in the retained
            # Gateway log without silently changing that public schema.
            _LOGGER.error(
                "Read-only grasp planning rejected: reason_code=%s details=%r",
                exc.reason_code,
                exc.details,
            )
            failed_base = {
                **base,
                "planner_evidence_refs": sorted({*source_evidence, rejection_ref}),
            }
            return (
                self._rejected_result(
                    base=failed_base,
                    reason_code=exc.reason_code,
                    diagnostics=self._diagnostics(exc),
                ),
                None,
            )
        if not isinstance(trajectory, RealAgentJointTrajectory):
            raise TypeError("RM65 planner returned an invalid trajectory type.")
        latest_qpos, now, post_problem = self._postplan_measured_qpos()
        if post_problem is not None:
            failed_base = self._base_result(
                source_operation=source_operation,
                source_result=source_result,
                candidate=candidate,
                ordinal=ordinal,
                speed_scale=speed_scale,
                target_reference=target_reference,
                target_execution=target_execution,
                qpos_cut=latest_qpos,
                now=now,
                planner_evidence_refs=trajectory.evidence_refs,
            )
            return (
                self._rejected_result(
                    base=failed_base,
                    reason_code="grasp_path_postplan_measured_qpos_" + post_problem,
                ),
                None,
            )
        assert latest_qpos is not None
        if any(
            max(
                abs(float(current) - float(planned))
                for current, planned in zip(
                    latest_qpos.qpos_by_side[arm_side],
                    trajectory.qpos_by_side[arm_side][0],
                    strict=True,
                )
            )
            > REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD
            for arm_side in ("left", "right")
        ):
            failed_base = self._base_result(
                source_operation=source_operation,
                source_result=source_result,
                candidate=candidate,
                ordinal=ordinal,
                speed_scale=speed_scale,
                target_reference=target_reference,
                target_execution=target_execution,
                qpos_cut=latest_qpos,
                now=now,
                planner_evidence_refs=trajectory.evidence_refs,
            )
            return (
                self._rejected_result(
                    base=failed_base,
                    reason_code="grasp_path_start_state_drift",
                ),
                None,
            )
        if self._current_station_revision() != expected_revision:
            failed_base = self._base_result(
                source_operation=source_operation,
                source_result=source_result,
                candidate=candidate,
                ordinal=ordinal,
                speed_scale=speed_scale,
                target_reference=target_reference,
                target_execution=target_execution,
                qpos_cut=latest_qpos,
                now=now,
                planner_evidence_refs=trajectory.evidence_refs,
            )
            return (
                self._rejected_result(
                    base=failed_base,
                    reason_code="grasp_path_station_revision_changed",
                ),
                None,
            )
        created_at = now
        expires_at = created_at + profile.plan_ttl_sec
        plan = RealGraspApproachPlan(
            runtime_session_id=self.runtime_session_id,
            profile_hash=profile.semantic_content_hash,
            planning_operation_id=operation.operation_id,
            planning_request_hash=request.semantic_content_hash,
            point_grasp_operation_id=source_operation.operation_id,
            point_grasp_request_hash=source_operation.request_hash,
            point_grasp_result_hash=base["point_grasp_result_hash"],
            point_grasp_candidate_hash=base["point_grasp_candidate_hash"],
            source_station_revision=expected_revision,
            component_id=profile.component_id,
            side=profile.side,
            target_pose_reference=target_reference,
            target_pose_execution=target_execution,
            speed_scale=speed_scale,
            measured_qpos_revision=qpos_cut.revision,
            measured_qpos_hash=qpos_cut.semantic_content_hash,
            measured_qpos_at=qpos_cut.measured_at,
            trajectory=trajectory,
            created_at=created_at,
            expires_at=expires_at,
            evidence_refs=tuple(
                sorted(
                    {
                        *source_evidence,
                        *qpos_cut.evidence_refs,
                        *latest_qpos.evidence_refs,
                        *trajectory.evidence_refs,
                        "grasp-approach-frame-resolution:"
                        + resolution.semantic_content_hash,
                    }
                )
            ),
        )
        # Keep the reported age tied to admission of the exact qpos cut used
        # to construct the path.  Recomputing it at completion made this field
        # grow by the entire planning duration even though the post-plan fresh
        # cut and start-state drift checks above had already passed.
        success_base = {
            **base,
            "planner_evidence_refs": sorted(set(plan.all_evidence_refs)),
        }
        result = {
            **success_base,
            "planning_status": "path_found",
            "reason_code": "path_found",
            "grasp_approach_plan_id": plan.plan_id,
            "grasp_approach_plan_hash": plan.semantic_content_hash,
            "trajectory_hash": trajectory.semantic_content_hash,
            "trajectory_sample_count": int(trajectory.time_from_start_sec.size),
            "trajectory_duration_sec": float(trajectory.time_from_start_sec[-1]),
            "plan_created_at": created_at,
            "plan_expires_at": expires_at,
            "eligible_for_execution": True,
            "diagnostics": {},
        }
        self.registration.payload_validator.validate_execution_result(result)
        self.registration.payload_validator.validate_execution_result_for_parameters(
            request.parameters,
            result,
        )
        self.plan_store.put(plan, now=created_at)
        return result, plan


def build_real_grasp_approach_plan_executor(
    *,
    registration: RealGraspApproachPlanRegistration,
    provider: RealGraspApproachPlanProvider,
    runtime_session_id: str,
    current_station_revision: Callable[[], int],
    max_retained_operations: int,
    clock: Callable[[], float] = time.time,
) -> PassiveReadSkillExecutor:
    """Bind full path assessment to a passive executor with no control port."""

    def read(request: Any, operation: AgentSkillOperation) -> PassiveReadResult:
        result, plan = provider.plan_request(request, operation)
        evidence_refs = (
            plan.all_evidence_refs
            if plan is not None
            else tuple(result["planner_evidence_refs"])
        )
        return PassiveReadResult(
            execution_result=result,
            evidence_refs=evidence_refs,
            reason_code="passive_grasp_approach_path_assessed",
        )

    return PassiveReadSkillExecutor(
        port_id="agent-gateway:real-passive-grasp-approach-plan:" + runtime_session_id,
        executor_id=registration.binding.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )


__all__ = [
    "REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_EXECUTOR_ID_V1",
    "REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1",
    "REAL_AGENT_GRASP_APPROACH_PLAN_EXECUTOR_ID_V1",
    "REAL_AGENT_GRASP_APPROACH_PLAN_PROFILE_ID_V1",
    "REAL_AGENT_GRASP_APPROACH_PLAN_ROUTE_ID_V1",
    "BoundedRealGraspApproachPlanStore",
    "RealGraspApproachPlan",
    "REAL_GRASP_APPROACH_PLAN_OPERATION_GRACE_SEC",
    "DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC",
    "RealGraspApproachPlanLookupError",
    "RealGraspApproachPlanProfile",
    "RealGraspApproachPlanProvider",
    "RealGraspApproachPlanRegistration",
    "build_real_grasp_approach_plan_executor",
    "build_real_grasp_approach_plan_registration",
]
