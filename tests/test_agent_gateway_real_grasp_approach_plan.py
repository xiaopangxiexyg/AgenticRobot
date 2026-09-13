from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real.grasp_approach_plan import (
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1,
    BoundedRealGraspApproachPlanStore,
    RealGraspApproachPlanLookupError,
    RealGraspApproachPlanProvider,
    build_real_grasp_approach_plan_registration,
)
from agent_gateway.profiles.real.station_point_grasp_deployment import (
    StationBoundPointGraspRegistration,
    build_station_bound_point_grasp_registration,
)
from agent_gateway.protocol.contracts import AgentSkillOperation
from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from agent_gateway.runtime.passive_read_executor import PassiveReadSkillExecutor
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.runtime.service import AgentGatewayOperationOutcome
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from teleop_runtime.skill_deploy.motion_port import (
    RealAgentJointTrajectory,
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspTrajectoryPlanningError,
)


class _TargetBuilder:
    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side: str,
        tool_tcp_world_target: np.ndarray,
    ) -> np.ndarray:
        del side
        return np.asarray(tool_tcp_world_target, dtype=float).copy()

    def tool_tcp_world_targets_for_qpos_pair(
        self,
        *,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
    ) -> dict[str, np.ndarray]:
        del left_qpos, right_qpos
        return {
            "left": np.eye(4, dtype=float),
            "right": np.eye(4, dtype=float),
        }


class _RetainedPointGraspExecutor(PassiveReadSkillExecutor):
    def __init__(
        self,
        *,
        executor_id: str,
        retained: AgentGatewayOperationOutcome,
    ) -> None:
        super().__init__(
            port_id="retained-point-grasp:test",
            executor_id=executor_id,
            read=lambda _request, _operation: (_ for _ in ()).throw(
                AssertionError("The retained source must not be rerun.")
            ),
            current_station_revision=lambda: 7,
        )
        self.retained = retained

    def completion_for(self, operation_id: str) -> AgentGatewayOperationOutcome | None:
        operation = self.retained.operation
        return (
            self.retained
            if operation is not None and operation.operation_id == operation_id
            else None
        )


def _no_op_point_grasp_validator(
    registration: StationBoundPointGraspRegistration,
) -> AgentSkillPayloadValidatorSet:
    route = registration.skill.routes[0]
    return AgentSkillPayloadValidatorSet(
        validator_id="point-grasp-source-validator:test",
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda _payload: None,
        validate_execution_result=lambda _payload: None,
        validate_execution_result_for_parameters=lambda _parameters, _result: None,
        validate_task_evaluation=lambda _payload: None,
    )


def _planning_fixture(*, planner):
    execution_frame = CoordinateFrameSpec(frame_id="dual_arm_solver_world")
    station_frame = CoordinateFrameSpec(
        frame_id="canonical_station",
        semantic_parameters={"axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1},
    )
    graph = CoordinateFrameGraph(
        graph_id="grasp-approach-plan-test",
        frames=(execution_frame, station_frame),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="station-to-solver-world",
                source_frame_id=station_frame.frame_id,
                target_frame_id=execution_frame.frame_id,
                matrix_target_source=np.eye(4, dtype=float),
            ),
        ),
    )
    point_grasp = build_station_bound_point_grasp_registration(
        aligned_rgbd_source_catalog=(
            {
                "source_id": "front",
                "role": "main",
                "native_width": 640,
                "native_height": 480,
            },
        ),
        source_binding_hash="source-binding:test",
        frame_binding_hash="frame-binding:test",
        frame_graph_hash=graph.semantic_content_hash,
        source_frame_id="head-camera",
        source_frame_spec_hash="head-camera-frame:test",
        reference_frame_id=station_frame.frame_id,
        reference_frame_spec_hash=station_frame.semantic_content_hash,
        backend_id="graspgenx",
        model_profile_id="graspgenx:graspmoe-dmtac-m-test",
        model_profile_hash="graspgenx-model:test",
        inference_implementation_hash="graspgenx-service-client:test",
        runtime_session_id="real-session:grasp-plan-test",
        control_domain_id="real-domain:grasp-plan-test",
        target_binding_id="real-target:grasp-plan-test",
        station_manifest_hash="station-manifest:grasp-plan-test",
        execution_manifest_hash="execution-manifest:grasp-plan-test",
        current_mode="real_active",
    )
    point_grasp = replace(
        point_grasp,
        payload_validator=_no_op_point_grasp_validator(point_grasp),
    )
    safety_ref = SkillDeploymentSafetyEnvelopeRef(
        object_id="real-envelope:grasp-plan-test",
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash="real-envelope-hash:grasp-plan-test",
        constraint_set_hash="real-constraint-hash:grasp-plan-test",
        allowed_route_ids=(REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1,),
    )
    registration = build_real_grasp_approach_plan_registration(
        point_grasp_registration=point_grasp,
        component_id="right_arm",
        side="right",
        reference_frame_id=station_frame.frame_id,
        reference_frame_spec_hash=station_frame.semantic_content_hash,
        execution_reference_frame_id=execution_frame.frame_id,
        execution_reference_frame_spec_hash=execution_frame.semantic_content_hash,
        frame_graph_hash=graph.semantic_content_hash,
        tool_config_hash="tool-config:grasp-plan-test",
        calibration_hash="calibration:grasp-plan-test",
        convergence_profile_hash="convergence:grasp-plan-test",
        safety_envelope_refs=(safety_ref,),
        max_measured_qpos_age_ms=200,
        min_speed_scale=0.05,
        max_speed_scale=1.0,
    )
    candidate_pose = np.eye(4, dtype=float)
    candidate_pose[:3, 3] = (0.25, -0.10, 0.30)
    source_result = {
        "station_revision": 7,
        "backend_id": point_grasp.profile.backend_id,
        "model_profile_id": point_grasp.profile.model_profile_id,
        "model_profile_hash": point_grasp.profile.model_profile_hash,
        "candidates": [
            {
                "ordinal": 0,
                "score": 0.9,
                "T_reference_from_tool_tcp": list(candidate_pose.reshape(-1)),
            }
        ],
    }
    source_operation = AgentSkillOperation(
        operation_id="point-grasp-operation:test",
        request_hash="a" * 64,
        runtime_session_id=registration.binding.runtime_session_id,
        status="completed",
        reason_code="passive_read_completed",
        station_revision_at_admission=7,
        current_station_revision=7,
        started_at=9.0,
        updated_at=9.5,
        deadline_at=20.0,
        execution_outcome="completed",
        execution_receipt_refs=("point-grasp-receipt:test",),
        execution_result=source_result,
    )
    retained_executor = _RetainedPointGraspExecutor(
        executor_id=point_grasp.binding.executor_id,
        retained=AgentGatewayOperationOutcome(
            accepted=True,
            reason_code="passive_read_completed",
            operation=source_operation,
        ),
    )
    port = RealAgentMotionPort(
        config=RealAgentMotionPortConfig(
            active_sides=("right",),
            component_id_by_side={"right": "right_arm"},
            execution_frame_id=execution_frame.frame_id,
            execution_frame_spec_hash=execution_frame.semantic_content_hash,
        ),
        target_builder=_TargetBuilder(),
    )
    qpos_by_side = {
        "left": np.zeros(6, dtype=float),
        "right": np.zeros(6, dtype=float),
    }
    port.publish_measured_qpos_pair(
        qpos_by_side=qpos_by_side,
        measured_at=10.0,
        available_at=10.0,
        source_evidence_refs=("measured-qpos:grasp-plan-test",),
    )
    store = BoundedRealGraspApproachPlanStore(max_retained_plans=4)
    provider = RealGraspApproachPlanProvider(
        registration=registration,
        point_grasp_registration=point_grasp,
        point_grasp_executor=retained_executor,
        motion_port=port,
        frame_graph=graph,
        grasp_trajectory_planner=planner,
        plan_store=store,
        runtime_session_id=registration.binding.runtime_session_id,
        current_station_revision=lambda: 7,
        clock=lambda: 10.0,
    )
    request = SimpleNamespace(
        input_artifact_bindings=(),
        expected_station_revision=7,
        parameters={
            "point_grasp_operation_id": source_operation.operation_id,
            "candidate_ordinal": 0,
            "speed_scale": 0.2,
        },
        semantic_content_hash="b" * 64,
    )
    planning_operation = AgentSkillOperation(
        operation_id="grasp-path-planning-operation:test",
        request_hash=request.semantic_content_hash,
        runtime_session_id=registration.binding.runtime_session_id,
        status="admitted",
        reason_code="admitted",
        station_revision_at_admission=7,
        current_station_revision=7,
        started_at=10.0,
        updated_at=10.0,
        deadline_at=20.0,
    )
    return registration, provider, store, port, request, planning_operation


def _stationary_trajectory(qpos_by_side) -> RealAgentJointTrajectory:
    return RealAgentJointTrajectory(
        time_from_start_sec=np.array([0.0, 0.5], dtype=float),
        qpos_by_side={
            side: np.stack((qpos_by_side[side], qpos_by_side[side]))
            for side in ("left", "right")
        },
        planner_evidence_refs=("rm65-production-urdf-fcl-path:test",),
    )


def test_path_found_is_read_only_and_produces_one_use_immutable_plan() -> None:
    calls = []

    def planner(**kwargs):
        calls.append(kwargs)
        return _stationary_trajectory(kwargs["qpos_by_side"])

    registration, provider, store, port, request, operation = _planning_fixture(
        planner=planner
    )

    result, plan = provider.plan_request(request, operation)

    assert plan is not None
    assert result["planning_status"] == "path_found"
    assert result["eligible_for_execution"] is True
    assert result["execution_requires_explicit_motion_skill"] is True
    assert result["motion_or_hold_performed"] is False
    assert result["grasp_approach_plan_id"] == plan.plan_id
    assert result["grasp_approach_plan_hash"] == plan.semantic_content_hash
    assert calls[0]["side"] == "right"
    assert calls[0]["speed_scale"] == 0.2
    np.testing.assert_allclose(
        calls[0]["target_tool_tcp_world"],
        plan.target_pose_execution,
    )
    assert port.snapshot().phase == RealAgentMotionPhase.INACTIVE
    assert port.snapshot().dispatch_count == 0

    reserved = store.reserve(
        plan_id=plan.plan_id,
        plan_hash=plan.semantic_content_hash,
        operation_id="motion-operation:one",
        runtime_session_id=registration.binding.runtime_session_id,
        profile_hash=registration.profile.semantic_content_hash,
        component_id=registration.profile.component_id,
        expected_station_revision=7,
        now=10.1,
    )
    assert reserved is plan
    store.consume(plan_id=plan.plan_id, operation_id="motion-operation:one")
    with pytest.raises(
        RealGraspApproachPlanLookupError,
        match="grasp_path_plan_already_consumed",
    ):
        store.reserve(
            plan_id=plan.plan_id,
            plan_hash=plan.semantic_content_hash,
            operation_id="motion-operation:two",
            runtime_session_id=registration.binding.runtime_session_id,
            profile_hash=registration.profile.semantic_content_hash,
            component_id=registration.profile.component_id,
            expected_station_revision=7,
            now=10.2,
        )


def test_planner_rejection_returns_evidence_without_motion_or_hold() -> None:
    def planner(**_kwargs):
        raise RM65GraspTrajectoryPlanningError(
            "target_safety_envelope_blocked",
            details={
                "modified_sides": ("right",),
                "best_position_residual_m": 0.0964,
                "best_orientation_residual_deg": 38.4,
                "cycles": 84,
            },
        )

    registration, provider, _store, port, request, operation = _planning_fixture(
        planner=planner
    )

    result, plan = provider.plan_request(request, operation)

    assert plan is None
    assert result["planning_status"] == "rejected"
    assert result["reason_code"] == "target_safety_envelope_blocked"
    assert result["eligible_for_execution"] is False
    assert result["grasp_approach_plan_id"] is None
    assert result["diagnostics"] == {
        "cycles": 84,
        "best_position_residual_m": 0.0964,
        "best_orientation_residual_deg": 38.4,
        "modified_sides": ["right"],
    }
    assert result["motion_or_hold_performed"] is False
    registration.payload_validator.validate_execution_result(result)
    registration.payload_validator.validate_execution_result_for_parameters(
        request.parameters,
        result,
    )
    assert port.snapshot().phase == RealAgentMotionPhase.INACTIVE
    assert port.snapshot().dispatch_count == 0


def test_planner_rejection_surfaces_nested_dense_path_diagnostics() -> None:
    error = RM65GraspTrajectoryPlanningError(
        "path_target_safety_envelope_blocked",
        details={
            "endpoint_candidate_count": 3,
            "last_dense_revalidation_error": {
                "reason_code": "path_target_safety_envelope_blocked",
                "details": {
                    "modified_sides": ("right",),
                    "segment_index": 0,
                    "edge_sample_index": 0,
                },
            },
        },
    )

    assert RealGraspApproachPlanProvider._diagnostics(error) == {
        "modified_sides": ["right"],
        "segment_index": 0,
        "edge_sample_index": 0,
    }


@pytest.fixture
def stored_plan():
    _, provider, store, _, request, operation = _planning_fixture(
        planner=lambda **kwargs: _stationary_trajectory(kwargs["qpos_by_side"])
    )
    _, plan = provider.plan_request(request, operation)
    assert plan is not None
    reservation = {
        "plan_id": plan.plan_id,
        "plan_hash": plan.semantic_content_hash,
        "operation_id": "motion-operation:one",
        "runtime_session_id": plan.runtime_session_id,
        "profile_hash": plan.profile_hash,
        "component_id": plan.component_id,
        "expected_station_revision": plan.source_station_revision,
        "now": plan.created_at + 0.1,
    }
    return store, plan, reservation


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("plan_id", "missing-plan", "not_found"),
        ("plan_hash", "different-hash", "hash_mismatch"),
        ("runtime_session_id", "another-session", "session_mismatch"),
        ("profile_hash", "another-profile", "profile_mismatch"),
        ("component_id", "another-component", "component_mismatch"),
        ("expected_station_revision", 8, "station_changed"),
    ],
)
def test_plan_store_rejects_wrong_lineage_without_reserving(
    stored_plan, field, value, reason
) -> None:
    store, plan, reservation = stored_plan
    with pytest.raises(
        RealGraspApproachPlanLookupError, match=f"^grasp_path_plan_{reason}$"
    ):
        store.reserve(**(reservation | {field: value, "operation_id": "wrong-owner"}))
    assert store.reserve(**reservation) is plan


@pytest.mark.parametrize("previously_reserved", [False, True])
def test_plan_store_rejects_plan_at_expiry(stored_plan, previously_reserved) -> None:
    store, plan, reservation = stored_plan
    if previously_reserved:
        store.reserve(**reservation)
    # Unreserved expired plans are pruned; reservations retain replay evidence.
    reason = "expired" if previously_reserved else "not_found"
    with pytest.raises(
        RealGraspApproachPlanLookupError, match=f"^grasp_path_plan_{reason}$"
    ):
        store.reserve(**(reservation | {"now": plan.expires_at}))


def test_plan_store_preserves_reservation_under_capacity_pressure(stored_plan) -> None:
    _, plan, reservation = stored_plan
    store = BoundedRealGraspApproachPlanStore(max_retained_plans=1)
    store.put(plan, now=reservation["now"])
    assert store.reserve(**reservation) is plan
    assert store.reserve(**reservation) is plan
    with pytest.raises(
        RealGraspApproachPlanLookupError, match="^grasp_path_plan_already_reserved$"
    ):
        store.reserve(**(reservation | {"operation_id": "motion-operation:two"}))
    with pytest.raises(
        RealGraspApproachPlanLookupError, match="^grasp_path_plan_reservation_mismatch$"
    ):
        store.consume(plan_id=plan.plan_id, operation_id="motion-operation:two")

    next_plan = replace(plan, planning_operation_id="planning-operation:next")
    with pytest.raises(
        RuntimeError, match="All retained grasp-approach plans are reserved"
    ):
        store.put(next_plan, now=reservation["now"])
    assert store.reserve(**reservation) is plan
    store.consume(plan_id=plan.plan_id, operation_id=reservation["operation_id"])
    store.put(next_plan, now=reservation["now"])
    with pytest.raises(
        RealGraspApproachPlanLookupError, match="^grasp_path_plan_not_found$"
    ):
        store.reserve(**reservation)
    next_reservation = reservation | {
        "plan_id": next_plan.plan_id,
        "plan_hash": next_plan.semantic_content_hash,
        "operation_id": "motion-operation:two",
    }
    assert store.reserve(**next_reservation) is next_plan


def test_plan_store_close_revokes_retained_plan(stored_plan) -> None:
    store, plan, reservation = stored_plan
    store.reserve(**reservation)
    store.close()
    with pytest.raises(
        RealGraspApproachPlanLookupError, match="^grasp_path_plan_store_closed$"
    ):
        store.reserve(**reservation)
    with pytest.raises(
        RealGraspApproachPlanLookupError, match="^grasp_path_plan_reservation_mismatch$"
    ):
        store.consume(plan_id=plan.plan_id, operation_id=reservation["operation_id"])
    with pytest.raises(RuntimeError, match="Grasp-approach plan store is closed"):
        store.put(plan, now=reservation["now"])
