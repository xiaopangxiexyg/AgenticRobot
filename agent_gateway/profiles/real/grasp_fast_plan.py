"""Direct frozen-observation planning into the existing one-use grasp store."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import numpy as np

from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.protocol.skill_catalog import SkillDeploymentBindingRef
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import (
    ArtifactStoreCapacityError,
    RecoverableAgentArtifactStore,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadError,
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.grasp_fast_plan_schemas import (
    fast_grasp_plan_schema_entry_v1,
    fast_grasp_plan_skill_definition_v1,
    grasp_path_attempts_json,
    validate_fast_grasp_plan_parameters_v1,
    validate_fast_grasp_plan_result_for_parameters_v1,
    validate_fast_grasp_plan_result_v1,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_ENCODING,
    GRASP_MULTIVIEW_SCHEMA_ID,
)
from dataset_contracts.grasp_rgbd_snapshot import (
    GRASP_RGBD_SNAPSHOT_ENCODING,
    GRASP_RGBD_SNAPSHOT_MEDIA_TYPE,
    GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
)
from deployments.perception.grasp_first_path import (
    AXIAL_GRASP_SEARCH_ORDER,
    GRASP_SEARCH_PREFERENCE,
    FirstGraspPathPolicy,
)
from deployments.perception.point_grasp_failure import (
    bounded_point_grasp_failure_diagnostic,
)
from teleop_runtime.skill_deploy.motion_port import (
    REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspTrajectoryPlanningError,
)

from .grasp_approach_plan import (
    RealGraspApproachPlanProvider,
    RealGraspApproachPlanRegistration,
)
from .grasp_first_path import RM65FirstGraspPathSelector
from .point_grasp_gateway import (
    StationBoundPointGraspProvider,
    retain_point_grasp_failure,
)
from .station_point_grasp_deployment import StationBoundPointGraspRegistration

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealFastGraspPlanRegistration:
    skill: SkillRegistrationTemplate
    binding: SkillDeploymentBindingRef
    payload_validator: AgentSkillPayloadValidatorSet
    profile_hash: str
    availability_hash: str
    policy: FirstGraspPathPolicy
    executor_id: str = "real.passive.first_grasp_path.v1"


def build_real_fast_grasp_plan_registration(
    *,
    paired: RealGraspApproachPlanRegistration,
    point_grasp_registration: StationBoundPointGraspRegistration,
) -> RealFastGraspPlanRegistration:
    """Publish only for an exact frozen, embodied GraspGenX deployment."""
    source = point_grasp_registration
    if (
        not source.profile.frozen_observation
        or source.profile.backend_id != "graspgenx"
        or source.profile.arm_reachability_profile is None
    ):
        raise ValueError("Direct grasp planning requires frozen embodied GraspGenX.")
    profile = paired.profile
    policy = FirstGraspPathPolicy()
    profile_id = "real-rm65-first-acceptable-grasp-v1"
    profile_hash = canonical_content_hash(
        {
            "execution_profile": profile.semantic_content_hash,
            "observation_profile": source.profile.semantic_content_hash,
            "search_policy": asdict(policy),
            "search_preference": GRASP_SEARCH_PREFERENCE,
            "refine_grasp_centers": True,
            "refine_grasp_axial_depth": True,
            "axial_search_order": AXIAL_GRASP_SEARCH_ORDER,
        },
        domain="dexteleop.fast_grasp_plan.profile.v1",
    )
    definition = fast_grasp_plan_skill_definition_v1(
        controlled_component_ids=(profile.component_id,)
    )
    schema = fast_grasp_plan_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile_id,
        parameter_profile_hash=profile_hash,
        component_id=profile.component_id,
        side=profile.side,
        roles=tuple(source.profile.source_by_role),
        min_speed_scale=profile.min_speed_scale,
        max_speed_scale=profile.max_speed_scale,
        reference_frame_id=profile.reference_frame_id,
        reference_frame_spec_hash=profile.reference_frame_spec_hash,
        frame_graph_hash=profile.frame_graph_hash,
        tool_config_hash=profile.tool_config_hash,
        calibration_hash=profile.calibration_hash,
    )
    io = inline_only_skill_io_contract(
        skill_id=definition.skill_id,
        skill_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    route = replace(
        paired.skill.routes[0],
        skill_id=definition.skill_id,
        skill_definition_hash=definition.semantic_content_hash,
        route_id="real.rm65.perception.plan_grasp_fast",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io.semantic_content_hash,
        max_operation_duration_ms=45_000,
    )
    skill = SkillRegistrationTemplate(
        definition=definition, schema_entry=schema, io_contract=io, routes=(route,)
    )
    binding = replace(
        paired.binding.to_ref(),
        deployment_binding_id="real.grasp-fast:" + paired.binding.runtime_session_id,
        deployment_binding_hash=canonical_content_hash(
            {
                "route": route.semantic_content_hash,
                "profile": profile_hash,
                "paired_binding": paired.binding.semantic_content_hash,
            },
            domain="dexteleop.fast_grasp_plan.binding.v1",
        ),
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id="real.fast-grasp:" + profile_hash,
        skill_id=definition.skill_id,
        skill_version="1",
        route_id=route.route_id,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=lambda value: validate_fast_grasp_plan_parameters_v1(
            value, schema.parameter_schema
        ),
        validate_execution_result=lambda value: validate_fast_grasp_plan_result_v1(
            value, schema.execution_result_schema
        ),
        validate_execution_result_for_parameters=validate_fast_grasp_plan_result_for_parameters_v1,
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    return RealFastGraspPlanRegistration(
        skill,
        binding,
        validator,
        profile_hash,
        canonical_content_hash(
            {"route": route.semantic_content_hash, "binding": binding.to_wire()},
            domain="dexteleop.fast_grasp_plan.availability.v1",
        ),
        policy,
    )


class RealFastGraspPlanProvider:
    """One model invocation, no public candidate operation and no motion port writes."""

    def __init__(
        self,
        *,
        registration: RealFastGraspPlanRegistration,
        point_provider: StationBoundPointGraspProvider,
        paired_provider: RealGraspApproachPlanProvider,
        observation_executor: PassiveReadSkillExecutor,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        source = point_provider.registration
        paired_profile = paired_provider.registration.profile
        adapter = point_provider.arm_reachability_evaluator
        if (
            source.profile.semantic_content_hash
            != paired_profile.point_grasp_profile_hash
            or source.binding.semantic_content_hash
            != paired_profile.point_grasp_deployment_binding_hash
            or source.skill.routes[0].semantic_content_hash
            != paired_profile.point_grasp_route_hash
        ):
            raise ValueError(
                "Fast grasp source differs from the paired execution profile."
            )
        if (
            adapter is None
            or adapter.profile != source.profile.arm_reachability_profile
            or adapter.profile.arm_side != paired_profile.side
            or adapter.profile.solver_reference_frame_id
            != paired_profile.execution_reference_frame_id
            or adapter.profile.solver_reference_frame_spec_hash
            != paired_profile.execution_reference_frame_spec_hash
            or adapter.candidate_reference_frame_id != paired_profile.reference_frame_id
            or adapter.frame_graph.semantic_content_hash
            != paired_profile.frame_graph_hash
            or adapter.planner is not paired_provider.grasp_trajectory_planner
        ):
            raise ValueError(
                "Fast grasp native adapter differs in side, frames or canonical planner."
            )
        expected = build_real_fast_grasp_plan_registration(
            paired=paired_provider.registration, point_grasp_registration=source
        )
        if (
            registration.profile_hash != expected.profile_hash
            or registration.binding != expected.binding
            or registration.skill != expected.skill
            or registration.policy != expected.policy
            or registration.executor_id != expected.executor_id
        ):
            raise ValueError(
                "Fast grasp registration differs from the bound source/execution pair."
            )
        if (
            source.observation_registration is None
            or observation_executor.executor_id
            != source.observation_registration.executor_id
        ):
            raise ValueError(
                "Fast grasp requires the matching frozen observation executor."
            )
        self.registration = registration
        self.point_provider = point_provider
        self.paired_provider = paired_provider
        self.observation_executor = observation_executor
        self.monotonic = monotonic

    def plan_request(
        self,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        *,
        cancelled: Callable[[], bool] = lambda: False,
        failure_snapshot_sink: Callable[[bytes], Mapping] | None = None,
        replay_snapshot_sink: Callable[[bytes, bool, str], None] | None = None,
    ) -> PassiveReadResult:
        started = self.monotonic()
        paired = self.paired_provider
        p = request.parameters
        if request.input_artifact_bindings:
            raise ValueError(
                "Fast grasp planning accepts only its retained observation reference."
            )
        self.registration.payload_validator.validate_parameters(p)
        completion = self.observation_executor.completion_for(
            p["observation_operation_id"]
        )
        if (
            completion is None
            or not completion.accepted
            or completion.operation is None
            or completion.operation.status != "completed"
            or completion.operation.execution_outcome != "completed"
            or completion.operation.runtime_session_id != paired.runtime_session_id
            or not isinstance(completion.operation.execution_result, Mapping)
            or completion.operation.execution_result["station_revision"]
            != request.expected_station_revision
        ):
            raise ValueError(
                "Direct grasp source is not a completed observation in this session/revision."
            )
        source = completion.operation
        source_result = source.execution_result
        profile = paired.registration.profile
        budget = min(
            float(p["planning_budget_sec"]),
            operation.deadline_at - float(paired._clock()),
        )
        deadline = started + max(0.0, budget)
        result = {
            "planning_status": "rejected",
            "reason_code": "grasp_fast_search_budget_exhausted",
            "component_id": profile.component_id,
            "side": profile.side,
            "observation_operation_id": source.operation_id,
            "observation_content_hash": canonical_content_hash(
                source_result, domain="dexteleop.agent_gateway.passive_read_result.v1"
            ),
            "role": p["role"],
            "target_pixel_uv": list(p["target_pixel_uv"]),
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            "source_station_revision": request.expected_station_revision,
            **{
                key: None
                for key in (
                    "T_reference_from_tool_tcp",
                    "gripper_width_m",
                    "grasp_approach_plan_id",
                    "grasp_approach_plan_hash",
                    "trajectory_hash",
                    "trajectory_sample_count",
                    "trajectory_duration_sec",
                    "plan_created_at",
                    "plan_expires_at",
                )
            },
            "eligible_for_execution": False,
            "execution_requires_explicit_motion_skill": True,
            "motion_or_hold_performed": False,
            "planning_policy": "first_acceptable_v1",
            "diagnostics": {},
        }
        selected = None
        plan = None
        if budget > 0:
            cut = paired.motion_port.latest_measured_qpos()
            problem = paired._qpos_problem(cut, now=float(paired._clock()))
            if problem is not None:
                result["reason_code"] = "grasp_path_measured_qpos_" + problem
            else:
                policy = replace(
                    self.registration.policy, total_budget_sec=min(budget, 120.0)
                )

                def factory(capture_cut, scene):
                    nonlocal selected
                    if capture_cut is None or any(
                        np.max(
                            np.abs(
                                np.asarray(capture_cut.qpos_by_side[side])
                                - np.asarray(cut.qpos_by_side[side])
                            )
                        )
                        > REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD
                        for side in ("left", "right")
                    ):
                        raise RM65GraspTrajectoryPlanningError(
                            "grasp_fast_capture_start_state_drift"
                        )
                    selected = RM65FirstGraspPathSelector(
                        adapter=self.point_provider.arm_reachability_evaluator,
                        qpos_cut=cut,
                        visible_scene=scene,
                        speed_scale=p["speed_scale"],
                        deadline=deadline,
                        policy=policy,
                        monotonic=self.monotonic,
                        cancelled=cancelled,
                    )
                    return selected

                # The existing provider owns frozen decoding, filtering and
                # inference. This is not an invented AgentSkillRequest/operation.
                def preparation_guard():
                    if cancelled():
                        raise RuntimeError(
                            "Fast grasp planning was cancelled during proposal preparation."
                        )
                    if self.monotonic() >= deadline:
                        raise RM65GraspTrajectoryPlanningError(
                            "grasp_fast_search_budget_exhausted"
                        )

                point_input = SimpleNamespace(
                    input_artifact_bindings=(),
                    expected_station_revision=request.expected_station_revision,
                    request_id=request.request_id,
                    parameters={
                        key: p[key]
                        for key in (
                            "observation_operation_id",
                            "role",
                            "target_pixel_uv",
                            "view_policy",
                        )
                    }
                    | {"max_candidates": 1, "include_arm_reachability": True},
                )
                try:
                    prepared = self.point_provider.plan_request(
                        point_input,
                        preparation_guard=preparation_guard,
                        selection_factory=factory,
                        annotate_result=False,
                        refine_grasp_centers=True,
                        refine_grasp_axial_depth=True,
                        failure_snapshot_sink=failure_snapshot_sink,
                    )
                    candidate = prepared.execution_result["candidates"][0]
                    if (
                        selected is None
                        or selected.result is None
                        or selected.result.trajectory is None
                    ):
                        raise TypeError(
                            "Direct grasp selector did not retain its complete trajectory."
                        )
                    target_reference = np.asarray(
                        candidate["T_reference_from_tool_tcp"]
                    ).reshape(4, 4)
                    target_execution, resolution = (
                        paired.frame_graph.transform_pose_matrix(
                            target_reference,
                            source_reference_frame_id=profile.reference_frame_id,
                            target_reference_frame_id=profile.execution_reference_frame_id,
                        )
                    )
                    plan, _latest, _now, rejection = paired.prepare_plan(
                        request=request,
                        operation=operation,
                        source_operation=source,
                        source_result=source_result,
                        grasp=candidate,
                        trajectory=selected.result.trajectory,
                        qpos_cut=cut,
                        target_reference=target_reference,
                        target_execution=target_execution,
                        speed_scale=p["speed_scale"],
                        frame_resolution_hash=resolution.semantic_content_hash,
                        source_kind="frozen_observation_fast",
                    )
                    if rejection is not None:
                        result["reason_code"] = rejection
                    elif self.monotonic() >= deadline:
                        result["reason_code"] = "grasp_fast_search_budget_exhausted"
                        plan = None
                    elif cancelled():
                        raise RuntimeError("Fast grasp planning was cancelled.")
                    else:
                        result.update(
                            planning_status="path_found",
                            reason_code="path_found",
                            T_reference_from_tool_tcp=list(
                                candidate["T_reference_from_tool_tcp"]
                            ),
                            gripper_width_m=candidate["gripper_width_m"],
                            eligible_for_execution=True,
                            grasp_approach_plan_id=plan.plan_id,
                            grasp_approach_plan_hash=plan.semantic_content_hash,
                            trajectory_hash=plan.trajectory.semantic_content_hash,
                            trajectory_sample_count=len(
                                plan.trajectory.time_from_start_sec
                            ),
                            trajectory_duration_sec=float(
                                plan.trajectory.time_from_start_sec[-1]
                            ),
                            plan_created_at=plan.created_at,
                            plan_expires_at=plan.expires_at,
                        )
                except PassiveReadError as error:
                    if selected is not None and selected.result is not None:
                        raise PassiveReadError(
                            str(error),
                            reason_code=selected.result.reason_code,
                            diagnostic=bounded_point_grasp_failure_diagnostic({**error.diagnostic,
                                "elapsed_sec": max(0., self.monotonic()-started),
                                "budget_sec": budget,
                                "budget_scope": "complete_request_including_preparation"}),
                        ) from error
                    raise
                except RM65GraspTrajectoryPlanningError as error:
                    raise PassiveReadError(
                        str(error),
                        reason_code=error.reason_code,
                        diagnostic={"native_planning": paired._diagnostics(error)},
                    ) from error
        if selected is not None and selected.result is not None:
            result["diagnostics"].update(
                pool_count=selected.pool_count,
                path_query_count=len(selected.result.attempts),
                attempts_json=grasp_path_attempts_json(selected.result.attempts),
            )
            if selected.contact_evidence:
                result["diagnostics"]["contact_evidence"] = selected.contact_evidence
        result["diagnostics"]["elapsed_sec"] = max(0.0, self.monotonic() - started)
        self.registration.payload_validator.validate_execution_result(result)
        self.registration.payload_validator.validate_execution_result_for_parameters(
            p, result
        )
        if plan is not None and result["eligible_for_execution"]:
            if replay_snapshot_sink is not None and prepared.rgbd_snapshot is not None:
                replay_snapshot_sink(prepared.rgbd_snapshot, prepared.multiview_snapshot,
                                     plan.trajectory.semantic_content_hash)
            if cancelled():
                raise RuntimeError(
                    "Fast grasp planning was cancelled before plan publication."
                )
            if self.monotonic() >= deadline:
                raise RuntimeError(
                    "Fast grasp planning exceeded its publication deadline."
                )
            if paired._current_station_revision() != request.expected_station_revision:
                raise RuntimeError(
                    "Station revision changed before fast plan publication."
                )
            paired.plan_store.put(plan, now=float(paired._clock()))
            if cancelled() or self.monotonic() >= deadline:
                # The canonical operation remains unsuccessful; the store
                # fences pending entries against its authoritative completion.
                raise RuntimeError(
                    "Fast grasp publication was cancelled or exceeded its deadline."
                )
        return PassiveReadResult(
            execution_result=result,
            evidence_refs=plan.all_evidence_refs
            if plan is not None
            else tuple(source.execution_receipt_refs),
            reason_code="passive_first_grasp_path_assessed",
        )


def build_real_fast_grasp_plan_executor(
    *,
    registration,
    provider,
    runtime_session_id,
    current_station_revision,
    max_retained_operations,
    clock,
    artifact_store: RecoverableAgentArtifactStore,
) -> PassiveReadSkillExecutor:
    executor = None

    def read(request, operation):
        def retain_replay(data, multiview, trajectory_hash):
            ref = artifact_ref_for_bytes(data,
                media_type="application/zip" if multiview else GRASP_RGBD_SNAPSHOT_MEDIA_TYPE,
                schema_id=GRASP_MULTIVIEW_SCHEMA_ID if multiview else GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
                encoding=GRASP_MULTIVIEW_ENCODING if multiview else GRASP_RGBD_SNAPSHOT_ENCODING,
                compression="zip")
            try:
                ref = commit_artifact_bytes(store=artifact_store, data=data, artifact=ref,
                    producer_id="real-grasp-fast-producer", runtime_session_id=runtime_session_id,
                    now=float(clock()), idempotency_key="fast-grasp-replay:"+operation.operation_id)
            except ArtifactStoreCapacityError as error:
                # Optional diagnostics must not become an extra motion/planning
                # gate. Integrity, identity and unexpected I/O errors still fail.
                _LOGGER.warning(
                    "Fast-grasp replay retention unavailable: operation_id=%s trajectory_hash=%s reason=%s",
                    operation.operation_id, trajectory_hash, str(error))
                return
            _LOGGER.warning(
                "Fast-grasp segmented replay retained: operation_id=%s trajectory_hash=%s sha256=%s path=%s",
                operation.operation_id, trajectory_hash, ref.sha256,
                artifact_store.managed_read_path(ref.artifact_id))

        return provider.plan_request(
            request,
            operation,
            cancelled=lambda: executor.cancel_requested(operation.operation_id),
            replay_snapshot_sink=retain_replay,
            failure_snapshot_sink=lambda data: retain_point_grasp_failure(
                data=data,
                artifact_store=artifact_store,
                runtime_session_id=runtime_session_id,
                operation_id=operation.operation_id,
                now=float(clock()),
            ),
        )

    executor = PassiveReadSkillExecutor(
        port_id="real.grasp-fast:" + runtime_session_id,
        executor_id=registration.executor_id,
        read=read,
        clock=clock,
        current_station_revision=current_station_revision,
        max_retained_operations=max_retained_operations,
    )
    return executor
