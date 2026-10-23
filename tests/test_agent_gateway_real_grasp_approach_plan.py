from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real.grasp_approach_plan import (
    DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC,
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1,
    BoundedRealGraspApproachPlanStore,
    RealGraspApproachPlanLookupError,
    RealGraspApproachPlanProvider,
    build_real_grasp_approach_plan_registration,
)
from agent_gateway.profiles.real.station_point_grasp_deployment import (
    StationBoundPointGraspRegistration,
    StationPointGraspArmReachabilityProfile,
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
    RM65_GRASP_PATH_PLANNING_TIME_BUDGET_SEC,
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


def _planning_fixture(
    *, planner, execution_operation_timeout_sec=DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC,
    finger_exchange=False, frozen_observation=False,
    plan_store_clock=None, planning_operation_lookup=None,
):
    from robot_motion.grasp_visible_scene import VisibleGraspScene, ParallelJawSolidGeometry
    # Explicit geometry fixture: route tests exercise control/plan lifecycles,
    # not physical grasp success. The real profile reads hash-bound Artifacts.
    scene = VisibleGraspScene(
        points=np.empty((0, 3)), target_mask=np.empty(0, dtype=bool),
        geometry=ParallelJawSolidGeometry(-0.043, 0.05, 0.01, 0.0255, 0.07),
    )
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
        finger_exchange=finger_exchange,
        frozen_observation=frozen_observation,
        arm_reachability_profile=(StationPointGraspArmReachabilityProfile(arm_side="right",
            evaluator_profile_id="fixture:reachability", evaluator_profile_hash="fixture:reachability-hash",
            solver_reference_frame_id=execution_frame.frame_id,
            solver_reference_frame_spec_hash=execution_frame.semantic_content_hash,
            frame_graph_hash=graph.semantic_content_hash) if finger_exchange else None),
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
        execution_operation_timeout_sec=execution_operation_timeout_sec,
    )
    candidate_pose = np.eye(4, dtype=float)
    candidate_pose[:3, 3] = (0.25, -0.10, 0.30)
    if finger_exchange:
        candidate_pose[:3, :3] = np.diag([-1., -1., 1.])
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
    store = BoundedRealGraspApproachPlanStore(max_retained_plans=4,
        clock=plan_store_clock, planning_operation_lookup=planning_operation_lookup)
    provider = RealGraspApproachPlanProvider(
        registration=registration,
        point_grasp_registration=point_grasp,
        point_grasp_executor=retained_executor,
        motion_port=port,
        frame_graph=graph,
        grasp_trajectory_planner=planner,
        visible_scene_reader=lambda _operation, _source: scene,
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


@pytest.mark.parametrize("publication_read", [1, 2, 3])
def test_planning_dates_each_measurement_after_reading_it(monkeypatch, publication_read) -> None:
    from contextlib import nullcontext

    _, provider, _, port, request, operation = _planning_fixture(
        planner=lambda **kwargs: _stationary_trajectory(kwargs["qpos_by_side"])
    )
    now = [10.0]
    reads = [0]
    original_read = port.latest_measured_qpos

    def read_with_publication():
        reads[0] += 1
        if reads[0] == publication_read:
            # A concurrent sampler publishes while the consumer acquires its cut.
            now[0] += 0.01
            port.publish_measured_qpos_pair(
                qpos_by_side={side: np.zeros(6) for side in ("left", "right")},
                measured_at=now[0], available_at=now[0],
                source_evidence_refs=("measurement-published-during-read:test",),
            )
        return original_read()

    provider._clock = lambda: now[0]
    provider.non_motion_read_window = nullcontext
    monkeypatch.setattr(port, "latest_measured_qpos", read_with_publication)
    if publication_read == 3:
        # Make the first post-plan cut stale to exercise the bounded hand-off retry.
        def planner(**kwargs):
            now[0] += 1.0
            return _stationary_trajectory(kwargs["qpos_by_side"])
        provider.grasp_trajectory_planner = planner

    result, plan = provider.plan_request(request, operation)
    assert result["planning_status"] == "path_found"
    assert plan is not None


@pytest.mark.parametrize("offset, reason", [(-0.3, "stale"), (0.01, "from_future")])
def test_planning_still_rejects_genuinely_invalid_measurement_time(offset, reason) -> None:
    _, provider, _, port, request, operation = _planning_fixture(
        planner=lambda **_kwargs: pytest.fail("Invalid measurements must not start planning.")
    )
    provider._clock = lambda: 10.0 - min(offset, 0.0)
    port.publish_measured_qpos_pair(
        qpos_by_side={side: np.zeros(6) for side in ("left", "right")},
        measured_at=10.0 + max(offset, 0.0), available_at=10.0 + max(offset, 0.0),
        source_evidence_refs=("invalid-measurement:test",),
    )
    result, plan = provider.plan_request(request, operation)
    assert result["reason_code"] == "grasp_path_measured_qpos_" + reason
    assert plan is None


def test_planning_route_deadline_outlives_internal_planner_budget() -> None:
    registration, *_ = _planning_fixture(planner=lambda **_kwargs: None)

    route = registration.skill.routes[0]

    assert route.max_operation_duration_ms == 135_000
    assert route.max_operation_duration_ms > int(
        RM65_GRASP_PATH_PLANNING_TIME_BUDGET_SEC * 1000.0
    )
    execute_route = registration.execute_skill.routes[0]
    assert execute_route.max_operation_duration_ms == int(
        DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC * 1000.0
    )
    assert route.control_effect_class == "read_only"
    assert route.may_move_robot is False

def test_execution_route_deadline_is_configurable() -> None:
    registration, *_ = _planning_fixture(
        planner=lambda **_kwargs: None,
        execution_operation_timeout_sec=180.0,
    )
    assert registration.execute_skill.routes[0].max_operation_duration_ms == 180_000

def test_graspgenx_missing_snapshot_cannot_publish_an_executable_plan() -> None:
    calls = []
    _, provider, _, _, request, operation = _planning_fixture(
        planner=lambda **kwargs: calls.append(kwargs)
    )
    provider.visible_scene_reader = None
    result, plan = provider.plan_request(request, operation)
    assert result["reason_code"] == "grasp_scene_snapshot_unavailable"
    assert result["planning_status"] == "rejected"
    assert plan is None and calls == []


@pytest.mark.parametrize("message", ["bad frame binding", "first line\nsecond line", "x" * 900])
def test_graspgenx_snapshot_failure_is_retained_as_read_only_rejection(caplog, message) -> None:
    calls = []
    _, provider, _, _, request, operation = _planning_fixture(
        planner=lambda **kwargs: calls.append(kwargs)
    )
    def broken_reader(*_args):
        raise ValueError(message)
    provider.visible_scene_reader = broken_reader
    result, plan = provider.plan_request(request, operation)
    assert result["reason_code"] == "grasp_scene_snapshot_invalid"
    assert result["diagnostics"] == {"exception_type": "ValueError"}
    assert plan is None and calls == []
    assert repr(" ".join(message.split())[:512]) in caplog.text
    if len(message) > 512:
        assert message not in caplog.text


def test_start_state_rejection_keeps_subject_and_unchanged_ingress_bound() -> None:
    def planner(**_kwargs):
        raise RM65GraspTrajectoryPlanningError("start_state_safety_envelope_blocked", details={
            "envelope_projection_subject": "current_measured_tool_tcp_in_envelope_reference_frame",
            "maximum_path_envelope_ingress_distance_m": 0.16,
            "path_envelope_ingress_distance_excess_m": 0.0007339041,
            "envelope_ingress_rejection": "initial_distance_exceeds_limit",
        })
    registration, provider, _, _, request, operation = _planning_fixture(planner=planner)
    result, plan = provider.plan_request(request, operation)
    registration.payload_validator.validate_execution_result(result)
    assert result["reason_code"] == "start_state_safety_envelope_blocked"
    assert result["diagnostics"]["maximum_path_envelope_ingress_distance_m"] == 0.16
    assert result["diagnostics"]["envelope_projection_subject"] == "current_measured_tool_tcp_in_envelope_reference_frame"
    assert plan is None


@pytest.mark.parametrize("finger_exchange", [False, True])
def test_path_found_is_read_only_and_produces_one_use_immutable_plan(finger_exchange) -> None:
    calls = []

    def planner(**kwargs):
        calls.append(kwargs)
        return _stationary_trajectory(kwargs["qpos_by_side"])

    registration, provider, store, port, request, operation = _planning_fixture(
        planner=planner, finger_exchange=finger_exchange,
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
    assert provider.point_grasp_registration.skill.definition.skill_version == ("5" if finger_exchange else "2")
    if finger_exchange:
        np.testing.assert_allclose(np.asarray(calls[0]["target_tool_tcp_world"]).reshape(4,4)[:3, :3],
            np.diag([-1., -1., 1.]))
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


def test_path_planning_uses_the_non_motion_read_window() -> None:
    events = []

    class Window:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            events.append("exit")
            return False

    registration, provider, _store, _port, request, operation = _planning_fixture(
        planner=lambda **kwargs: _stationary_trajectory(kwargs["qpos_by_side"])
    )
    provider.non_motion_read_window = lambda: Window()

    original_reader = provider.visible_scene_reader

    def reader(source_operation, source_result):
        events.append("reader")
        assert original_reader is not None
        return original_reader(source_operation, source_result)

    provider.visible_scene_reader = reader

    result, plan = provider.plan_request(request, operation)

    assert result["planning_status"] == "path_found"
    assert plan is not None
    assert events == ["enter", "reader", "exit"]
    registration.payload_validator.validate_execution_result(result)


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


def test_approach_rejection_retains_full_details_without_changing_v1_schema(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def planner(**_kwargs):
        raise RM65GraspTrajectoryPlanningError(
            "approach_lateral_deviation",
            details={
                "segment_index": 2,
                "edge_sample_index": 4,
                "lateral_error_m": 0.0034,
                "maximum_lateral_error_m": 0.002,
            },
        )

    registration, provider, _store, port, request, operation = _planning_fixture(
        planner=planner
    )

    result, plan = provider.plan_request(request, operation)

    assert plan is None
    assert result["reason_code"] == "approach_lateral_deviation"
    assert result["diagnostics"] == {"segment_index": 2, "edge_sample_index": 4}
    assert "lateral_error_m" in caplog.text
    assert "0.0034" in caplog.text
    registration.payload_validator.validate_execution_result(result)
    assert port.snapshot().phase == RealAgentMotionPhase.INACTIVE
    assert port.snapshot().dispatch_count == 0


def test_approach_ik_rejection_uses_existing_residual_diagnostic_fields() -> None:
    error = RM65GraspTrajectoryPlanningError(
        "approach_ik_search_exhausted",
        details={
            "solve_cycles": 180,
            "position_residual_m": 0.004,
            "orientation_residual_deg": 1.2,
            "approach_waypoint_index": 3,
        },
    )

    assert RealGraspApproachPlanProvider._diagnostics(error) == {
        "cycles": 180,
        "best_position_residual_m": 0.004,
        "best_orientation_residual_deg": 1.2,
    }


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
                    "envelope_projection_by_side": {
                        "right": {
                            "translation_modified": True,
                            "rotation_modified": False,
                            "translation_adjustment_m": 0.12014,
                            "rotation_adjustment_deg": 0.0,
                            "violated_translation_constraints": ["absolute:y:max"],
                            "active_translation_constraints": ["absolute:y:max"],
                            "raw_target_xyz_m": [-0.3324, 0.22014, 0.2012],
                            "effective_target_xyz_m": [-0.3324, 0.1, 0.2012],
                        }
                    },
                    "envelope_ingress_rejection": "path_moved_away_from_envelope",
                },
            },
        },
    )

    assert RealGraspApproachPlanProvider._diagnostics(error) == {
        "modified_sides": ["right"],
        "segment_index": 0,
        "edge_sample_index": 0,
        "envelope_projection_by_side": {
            "right": {
                "translation_modified": True,
                "rotation_modified": False,
                "translation_adjustment_m": 0.12014,
                "rotation_adjustment_deg": 0.0,
                "violated_translation_constraints": ["absolute:y:max"],
                "active_translation_constraints": ["absolute:y:max"],
                "raw_target_xyz_m": [-0.3324, 0.22014, 0.2012],
                "effective_target_xyz_m": [-0.3324, 0.1, 0.2012],
            }
        },
        "envelope_ingress_rejection": "path_moved_away_from_envelope",
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


@pytest.mark.parametrize("source", ("input_qpos", "envelope_ingress_endpoint"))
def test_start_rejection_preserves_composed_cause_without_claiming_physical_fcl_contact(source):
    details = {"planning_phase": "pregrasp_target", "start_qpos_source": source,
        "start_check": {"safe": False, "collision_check_available": True,
            "minimum_distance_m": .01, "reason": "path_target_safety_envelope_blocked",
            "metadata": {"closest_geometry_names": ["right_hand_proxy_0", "observed_voxels"],
                "nearest_observed": {"geometry_name": "right_hand_proxy_0",
                    "occupied_center_world_m": [.1, .2, .3]}}}}
    def planner(**_):
        raise RM65GraspTrajectoryPlanningError("path_start_collision_blocked", details=details)
    registration, provider, _, port, request, operation = _planning_fixture(planner=planner)
    result, plan = provider.plan_request(request, operation)
    assert plan is None and port.snapshot().dispatch_count == 0
    diagnostics = result["diagnostics"]
    assert diagnostics["start_qpos_source"] == source
    check = json.loads(diagnostics["start_check_json"])
    assert check["reason"] == "path_target_safety_envelope_blocked"
    assert len(check["detail_hash"]) == 64
    registration.payload_validator.validate_execution_result(result)
    for invalid in ("", "[]", "x"*2049):
        changed = {**result, "diagnostics": {**diagnostics, "start_check_json": invalid}}
        with pytest.raises(ValueError):
            registration.payload_validator.validate_execution_result(changed)



def _start_rejection_result(*, start_check=None, source="input_qpos"):
    def planner(**_):
        raise RM65GraspTrajectoryPlanningError("path_start_collision_blocked", details={
            "start_qpos_source": source, "path": {"status": "start_blocked"},
            "start_check": start_check if start_check is not None else {
                "safe": False, "collision_check_available": True,
                "minimum_distance_m": .01, "reason": "path_target_safety_envelope_blocked",
                "metadata": {"collision_check": {"safe": True, "collision_check_available": True,
                    "minimum_distance_m": .01, "reason": None, "metadata": {}},
                    "target_envelope": {"constraint": "fixture_envelope"}}}})
    registration, provider, store, port, request, operation = _planning_fixture(planner=planner)
    result, plan = provider.plan_request(request, operation)
    assert plan is None
    assert result["planning_status"] == "rejected"
    assert result["reason_code"] == "path_start_collision_blocked"
    assert result["eligible_for_execution"] is False
    assert result["grasp_approach_plan_id"] is None
    assert result["motion_or_hold_performed"] is False
    assert result["diagnostics"]["path_status"] == "start_blocked"
    assert port.snapshot().phase == RealAgentMotionPhase.INACTIVE
    assert port.snapshot().dispatch_count == 0
    registration.payload_validator.validate_execution_result(result)
    return registration, result


def test_start_projection_retains_composed_cause_and_hashes_exact_published_preimage():
    from dataset_contracts.canonical_encoding import canonical_content_hash
    _, result = _start_rejection_result()
    projected = json.loads(result["diagnostics"]["start_check_json"])
    digest = projected.pop("detail_hash")
    assert projected["detail_scope"] == "published_projection"
    assert projected["metadata"]["collision_check"]["safe"] is True
    assert projected["metadata"]["target_envelope"]["constraint"] == "fixture_envelope"
    assert digest == canonical_content_hash(projected, domain="dexteleop.grasp_start_check_projection.v1")


def test_start_projection_copies_bounded_numpy_metadata_scalars_and_array():
    _, result = _start_rejection_result(start_check={"safe": False,
        "collision_check_available": True, "minimum_distance_m": -.001,
        "reason": "fixture_collision", "metadata": {"nearest_observed": {
            "present": np.bool_(True), "distance_m": np.float64(.01),
            "occupied_center_world_m": np.array([.1, .2, .3])}}})
    projected = json.loads(result["diagnostics"]["start_check_json"])
    assert projected["metadata"]["nearest_observed"] == {
        "present": True, "distance_m": .01, "occupied_center_world_m": [.1, .2, .3]}
    assert "details_omitted" not in projected


@pytest.mark.parametrize("metadata", (
    {"nearest_observed": {"note": "x" * 513}},
    {"target_envelope": {"values": list(range(17))}},
    {"target_envelope": {"bad": float("nan")}},
    {"target_envelope": {"bad": object()}},
    {"target_envelope": {"a": {"b": {"c": {"d": {"e": {"f": {"g": 0}}}}}}}},
    {"nearest_observed": {str(i): "x" * 500 for i in range(6)}},
))
def test_defensive_start_projection_trims_unrepresentable_metadata_without_losing_refusal(metadata):
    # Direct malformed-error injection is defensive; typed production path checks
    # already require finite distance and canonical path metadata.
    from dataset_contracts.canonical_encoding import canonical_content_hash
    _, result = _start_rejection_result(start_check={"safe": False,
        "collision_check_available": True, "minimum_distance_m": -.001,
        "reason": "fixture_collision", "metadata": metadata})
    encoded = result["diagnostics"]["start_check_json"]
    assert len(encoded) <= 2048
    projected = json.loads(encoded)
    digest = projected.pop("detail_hash")
    assert projected["details_omitted"] is True
    assert "metadata" not in projected
    assert projected["reason"] == "fixture_collision"
    assert digest == canonical_content_hash(projected, domain="dexteleop.grasp_start_check_projection.v1")


@pytest.mark.parametrize("field,value", (("minimum_distance_m", float("inf")),
    ("minimum_distance_m", True), ("reason", object()), ("safe", "false")))
def test_defensive_invalid_start_core_omits_optional_json_and_preserves_refusal(field, value):
    check = {"safe": False, "collision_check_available": True,
             "minimum_distance_m": .01, "reason": "fixture_collision"}
    check[field] = value
    _, result = _start_rejection_result(start_check=check)
    assert "start_check_json" not in result["diagnostics"]
    assert result["diagnostics"]["start_qpos_source"] == "input_qpos"


@pytest.mark.parametrize("failure", ("hash", "encode"))
def test_start_diagnostic_encoder_failure_cannot_mask_lower_refusal(monkeypatch, failure):
    import agent_gateway.profiles.real.grasp_approach_plan as module
    original_hash = module.canonical_content_hash
    hash_calls, encode_calls = [], []
    def hashed(value, *, domain):
        if domain == "dexteleop.grasp_start_check_projection.v1":
            hash_calls.append(domain)
            if failure == "hash":
                raise ValueError("fixture diagnostic hash failure")
        return original_hash(value, domain=domain)
    def encode(*_, **__):
        encode_calls.append("encode")
        raise ValueError("fixture diagnostic encoding failure")
    monkeypatch.setattr(module, "canonical_content_hash", hashed)
    if failure == "encode":
        # Replace only the provider reference, keeping canonical encoding's
        # stdlib json.dumps intact so this reaches the post-hash failure.
        monkeypatch.setattr(module, "json", SimpleNamespace(dumps=encode))
    _, result = _start_rejection_result()
    assert result["diagnostics"] == {"path_status": "start_blocked", "start_qpos_source": "input_qpos"}
    assert len(hash_calls) == 1
    assert encode_calls == (["encode"] if failure == "encode" else [])


@pytest.mark.parametrize("encoded", ("", "[]", "null", "0", "{", "x" * 2049,
    '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e400}',
    '{"x":-1e400}', '{"x":' + '[' * 1000 + '0' + ']' * 1000 + '}'))
def test_start_check_json_validator_rejects_invalid_nonfinite_or_deep_payload(encoded):
    registration, result = _start_rejection_result()
    changed = {**result, "diagnostics": {**result["diagnostics"], "start_check_json": encoded}}
    with pytest.raises(ValueError):
        registration.payload_validator.validate_execution_result(changed)


@pytest.mark.parametrize("character", ("x", "\u4e2d"))
def test_start_check_json_limit_counts_decoded_unicode_characters(character):
    registration, result = _start_rejection_result()
    for length in (2048, 2049):
        encoded = json.dumps({"x": character * (length-8)}, ensure_ascii=False, separators=(",", ":"))
        assert len(encoded) == length
        changed = {**result, "diagnostics": {**result["diagnostics"], "start_check_json": encoded}}
        if length == 2048:
            registration.payload_validator.validate_execution_result(changed)
        else:
            with pytest.raises(ValueError):
                registration.payload_validator.validate_execution_result(changed)


@pytest.mark.parametrize("source", ("measured", "", None, [], 1))
def test_start_source_validator_rejects_unknown_or_untyped_provenance(source):
    registration, result = _start_rejection_result()
    changed = {**result, "diagnostics": {**result["diagnostics"], "start_qpos_source": source}}
    with pytest.raises(ValueError):
        registration.payload_validator.validate_execution_result(changed)


def test_grasp_start_diagnostic_contract_migration_requires_fresh_result_and_route_binding():
    from copy import deepcopy
    from agent_gateway.protocol._contract_validation import thaw_json
    from agent_gateway.protocol.skill_catalog import agent_skill_schema_document_hash
    registration, result = _start_rejection_result()
    schema = registration.skill.schema_entry
    current = thaw_json(schema.execution_result_schema)
    old = deepcopy(current)
    added_failure_fields = {
        "planning_phase", "target_collision_points", "background_collision_points",
        "terminal_target_contact_points", "terminal_contact_allowed",
        "path_gripper_proxy_padding_m", "terminal_position_error_m", "terminal_orientation_error_deg",
    }
    for field in added_failure_fields:
        del old["properties"]["diagnostics"]["properties"][field]
    for field in ("start_qpos_source", "start_check_json"):
        del old["properties"]["diagnostics"]["properties"][field]
    old_hash = agent_skill_schema_document_hash(old, schema_kind="execution_result")
    assert old_hash == "0096762d790e2a0c4ea53c1a661f0f7a413ea1c7f07936a6f707c87219216c7a"
    route = registration.skill.routes[0]
    assert route.execution_result_schema_hash == schema.execution_result_schema_hash != old_hash
    assert route.parameter_schema_hash == "9a3bc2e0194bb578ea1b5e15b3b551de69e21d438028e49ddad01b8bf7963a93"
    assert route.skill_definition_hash == "ecea8e7b2bd9b436131c050c4e740ddcfe479782020e243594a49985a2c1bded"
    assert route.skill_io_contract_hash != "c023e278a7fdb8910ad6e78b11ed02a2d5ebe533e0af61f70302a3fd380aeaf9"
    assert registration.binding.route_hash != "a595af276440972ef11b5a9c057091a78a88891e81f1fd11154e430ff9e2d5b7"
    assert route.control_effect_class == "read_only" and route.privilege_tier == "t0_read_only"
    assert not route.may_move_robot and not route.may_open_publication
    assert registration.binding.route_hash == route.semantic_content_hash
    registration.payload_validator.validate_execution_result(result)
    # Freeze the old strict diagnostic gate without introducing a second
    # schema engine dependency. Its closed property set forbids the new keys.
    old_diagnostics = old["properties"]["diagnostics"]
    assert old_diagnostics["additionalProperties"] is False
    original_start_diagnostics = {
        key: value for key, value in result["diagnostics"].items()
        if key not in added_failure_fields
    }
    assert set(original_start_diagnostics) - set(old_diagnostics["properties"]) == {
        "start_qpos_source", "start_check_json"}
    legacy_result = {**result, "diagnostics": {"path_status": "start_blocked"}}
    assert not set(legacy_result["diagnostics"]) - set(old_diagnostics["properties"])
    registration.payload_validator.validate_execution_result(legacy_result)
    # The separately authorized execution contract retains its old hashes.
    execute = registration.execute_skill.routes[0]
    assert execute.execution_result_schema_hash == "528c66fcead63efdf3ec16d4eda7545cef32104e68ec6720d712d11f6f2440b1"
    assert execute.skill_io_contract_hash == "9f5af1fa1e28933cd5dbcd47e92035ef95566098550c7e3a949c23669bb215f7"




def test_approach_rejection_retains_unpublished_details_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def planner(**_kwargs):
        raise RM65GraspTrajectoryPlanningError(
            "approach_lateral_deviation",
            details={
                "segment_index": 2,
                "edge_sample_index": 4,
                "lateral_error_m": 0.0034,
                "maximum_lateral_error_m": 0.002,
            },
        )

    registration, provider, _store, port, request, operation = _planning_fixture(
        planner=planner
    )

    result, plan = provider.plan_request(request, operation)

    assert plan is None
    assert result["reason_code"] == "approach_lateral_deviation"
    assert result["diagnostics"] == {"segment_index": 2, "edge_sample_index": 4}
    assert "lateral_error_m" in caplog.text
    assert "0.0034" in caplog.text
    registration.payload_validator.validate_execution_result(result)
    assert port.snapshot().phase == RealAgentMotionPhase.INACTIVE
    assert port.snapshot().dispatch_count == 0


def test_missing_collision_pose_metrics_do_not_invalidate_the_rejection():
    error = RM65GraspTrajectoryPlanningError("visible_gripper_path_collision", details={
        "planning_phase": "visible_scene_dense_revalidation", "target_collision_points": 1,
        "background_collision_points": 0, "terminal_position_error_m": None,
        "terminal_orientation_error_deg": None,
    })
    assert RealGraspApproachPlanProvider._diagnostics(error) == {
        "planning_phase": "visible_scene_dense_revalidation", "target_collision_points": 1,
        "background_collision_points": 0,
    }


@pytest.mark.parametrize("field,invalid", [
    (None, None), ("target_collision_points", True), ("background_collision_points", -1),
    ("terminal_contact_allowed", 1), ("path_gripper_proxy_padding_m", float("nan")),
    ("terminal_position_error_m", -1), ("terminal_orientation_error_deg", 181),
    ("planning_phase", ""),
])
def test_visible_collision_rejection_publishes_bounded_evidence_without_motion(field, invalid):
    details = {
        "planning_phase": "visible_scene_dense_revalidation", "target_collision_points": 1,
        "background_collision_points": 0, "terminal_target_contact_points": 0,
        "terminal_contact_allowed": False, "path_gripper_proxy_padding_m": .003,
        "terminal_position_error_m": .0184, "terminal_orientation_error_deg": .001,
    }
    def planner(**_kwargs):
        raise RM65GraspTrajectoryPlanningError("visible_gripper_path_collision", details=details)
    registration, provider, _, port, request, operation = _planning_fixture(planner=planner)
    result, plan = provider.plan_request(request, operation)
    assert plan is None and not result["motion_or_hold_performed"]
    assert result["diagnostics"] == details
    assert port.snapshot().dispatch_count == 0
    registration.payload_validator.validate_execution_result(result)
    if field is not None:
        result["diagnostics"][field] = invalid
        with pytest.raises((TypeError, ValueError)):
            registration.payload_validator.validate_execution_result(result)
