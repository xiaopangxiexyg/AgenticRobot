from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_station.agent_gateway import (
    AgentSkillOperation,
    AgentSkillRequest,
    CubeLiftSimDeploymentBinding,
    CubeLiftTerminalEvaluatorConfig,
    cube_lift_evaluation_plan_v1,
    cube_lift_schema_entry_v1,
    cube_lift_sim_route_v1,
    cube_lift_skill_definition_v1,
)
from control_station.sim_hil_profile.agent_skill import (
    CubeLiftAgentSkillLifecycle,
    CubeLiftCandidateObservation,
)
from control_station.sim_hil_profile.execution_evidence import (
    CubeLiftExecutionEvidenceCollector,
    CubeLiftExecutionEvidenceCut,
)


RUNTIME_SESSION_ID = "runtime:cube-lift-agent-skill-test"
CONTROL_DOMAIN_ID = "mujoco-sim-hil"
TRACE_PROFILE_ID = "trace-profile:cube-lift-agent-skill-test"
TASK_CONFIG_HASH = "task-config:cube-lift-agent-skill-test"


def _contracts():
    definition = cube_lift_skill_definition_v1()
    schema = cube_lift_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash
    )
    route = cube_lift_sim_route_v1(definition=definition, schema_entry=schema)
    evaluator_config = CubeLiftTerminalEvaluatorConfig(
        task_config_hash=TASK_CONFIG_HASH,
        object_body_name="debug_cube",
        left_finger_body_name="right_gripper_left_finger",
        right_finger_body_name="right_gripper_right_finger",
        table_body_name="table",
        lift_delta_m=0.04,
        stable_hold_physics_ticks=100,
        physics_hz=1000.0,
    )
    binding = CubeLiftSimDeploymentBinding(
        deployment_binding_id="sim.mujoco.cube_lift.policy_only.test",
        deployment_binding_version="1",
        route_hash=route.semantic_content_hash,
        runtime_session_id=RUNTIME_SESSION_ID,
        control_domain_id=CONTROL_DOMAIN_ID,
        target_binding_id="target:cube-lift-agent-skill-test",
        station_manifest_hash="station-manifest:cube-lift-agent-skill-test",
        execution_manifest_hash="execution-manifest:cube-lift-agent-skill-test",
        execution_domain_binding_hash="execution-domain:cube-lift-agent-skill-test",
        target_runtime_action_contract_hash="action-contract:test",
        component_target_capability_hash="component-capability:test",
        action_schema_hash="action-schema:test",
        observation_plan_hash="observation-plan:test",
        frame_graph_hash="frame-graph:test",
        tool_config_hash="tool:test",
        calibration_hash="calibration:test",
        task_config_hash=TASK_CONFIG_HASH,
        policy_checkpoint_hash="checkpoint:test",
        evaluator_configuration_hash=evaluator_config.semantic_content_hash,
        trace_requirement_profile_id=TRACE_PROFILE_ID,
        executor_id="control-station.cube-lift-worker:test",
        explicit_exclusions=(
            "hardware_backend",
            "human_input_attachment",
            "raw_action_submission",
            "second_dispatch_owner",
        ),
    )
    plan = cube_lift_evaluation_plan_v1(
        definition=definition,
        route=route,
        deployment_binding=binding,
        evaluator_config=evaluator_config,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    profile = route.allowed_parameter_profiles[0]
    request = AgentSkillRequest(
        client_id="rpent:cube-lift-agent-skill-test",
        request_id="request:cube-lift-agent-skill-test",
        idempotency_key="idempotency:cube-lift-agent-skill-test",
        client_sequence=1,
        runtime_session_id=RUNTIME_SESSION_ID,
        skill_route_manifest_hash="route-manifest:cube-lift-agent-skill-test",
        expected_station_revision=4,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=binding.semantic_content_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={},
        observation_snapshot_id="snapshot:cube-lift-agent-skill-test",
        requested_mode="sim_active",
        requested_timeout_ms=30_000,
    )
    return route, evaluator_config, binding, plan, request


def _operation(request: AgentSkillRequest, *, status: str) -> AgentSkillOperation:
    return AgentSkillOperation(
        operation_id="operation:cube-lift-agent-skill-test",
        request_hash=request.semantic_content_hash,
        runtime_session_id=request.runtime_session_id,
        status=status,
        reason_code=f"test_{status}",
        station_revision_at_admission=4,
        current_station_revision=4,
        started_at=10.0,
        updated_at=10.1,
        deadline_at=40.0,
        lower_operation_id=(
            None if status == "admitted" else "lower:cube-lift-agent-skill-test"
        ),
        execution_receipt_refs=("station-start:1",),
        progress_summary={"phase": status},
    )


def _hold_transition():
    return SimpleNamespace(
        event=SimpleNamespace(
            runtime_session_id=RUNTIME_SESSION_ID,
            control_domain_id=CONTROL_DOMAIN_ID,
            event_id="gate-event:cube-lift-agent-skill-test",
        ),
        snapshot=SimpleNamespace(
            control_domain_id=CONTROL_DOMAIN_ID,
            status="hold",
            semantic_content_hash="gate-snapshot:cube-lift-agent-skill-test",
            gate_epoch=7,
        ),
        cancellation=SimpleNamespace(
            cancellation_id="cancellation:cube-lift-agent-skill-test"
        ),
        cancellation_receipt=SimpleNamespace(
            cancellation_id="cancellation:cube-lift-agent-skill-test",
            required_ack_levels=("dispatch",),
            acknowledged_levels=("dispatch",),
            pending_levels=(),
            rejected_levels=(),
            acknowledgement_deadline_at=10.2,
            observed_at=10.15,
            complete=True,
            timed_out=False,
            requires_fault_escalation=False,
            late_levels=(),
        ),
        trace_facts=(
            SimpleNamespace(fact_id="trace-fact:cube-lift-agent-skill-hold"),
        ),
    )


def _execution_cut() -> CubeLiftExecutionEvidenceCut:
    return CubeLiftExecutionEvidenceCut(
        runtime_session_id=RUNTIME_SESSION_ID,
        control_domain_id=CONTROL_DOMAIN_ID,
        execution_evidence_chain_ref="execution-chain:test",
        source_proposal_chain_hash="source-chain:test",
        authority_decision_chain_hash="authority-chain:test",
        safety_decision_chain_hash="safety-chain:test",
        target_request_chain_hash="target-request-chain:test",
        target_submission_receipt_chain_hash="target-submission-chain:test",
        target_completion_receipt_chain_hash="target-completion-chain:test",
        resolved_component_target_chain_hash="resolved-target-chain:test",
        component_admission_receipt_chain_hash="target-admission-chain:test",
        dispatch_receipt_chain_hash="dispatch-chain:test",
        measurement_lineage_chain_hash="measurement-chain:test",
        first_cycle_receipt_ref="cycle-receipt:first",
        last_cycle_receipt_ref="cycle-receipt:last",
        terminal_state_hash="terminal-state:test",
        cycle_count=1,
        dispatched_cycle_count=1,
        measurement_count=1,
    )


class _FakeApplication:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.config = SimpleNamespace(
            initial_human_alpha=0.0,
            defer_human_input=True,
            realtime_worker=False,
            mujoco_viewer_enabled=False,
            execution_difference_enabled=True,
        )
        self.runtime = SimpleNamespace(
            config=SimpleNamespace(
                runtime_session_id=RUNTIME_SESSION_ID,
                control_domain_id=CONTROL_DOMAIN_ID,
            )
        )
        self.composition = SimpleNamespace(
            activation_scope=SimpleNamespace(
                activation_id="lower:cube-lift-agent-skill-test",
                semantic_content_hash="activation-scope:test",
            )
        )
        self.task_config = SimpleNamespace(
            object_body_name="debug_cube",
            scripted_pregrasp_reset=False,
            scripted_contact_reset=False,
        )
        self.task_initialization_info = {
            "owner": "sim.mujoco.tasks.cube_lift.CubeLiftKernel.reset",
            "seed": 0,
            "resting_center_z_m": 0.026,
            "scripted_pregrasp_reset": False,
            "scripted_contact_reset": False,
        }
        self.execution_difference_observer = SimpleNamespace(
            snapshot=self._measurement_snapshot
        )
        self.started = False
        self.next_decision_at = None

    @staticmethod
    def clock() -> float:
        return 10.0

    def start(self, *, cancel_requested):
        assert not cancel_requested()
        self.calls.append("application.start")
        self.started = True
        return SimpleNamespace(semantic_content_hash="preflight:test")

    def step(self):
        self.calls.append("application.step")
        return SimpleNamespace(cycle_index=0)

    def _measurement_snapshot(self):
        self.calls.append("measurement.snapshot")
        return SimpleNamespace()

    def hold(self, *, reason: str, force_cancellation: bool):
        assert reason
        assert force_cancellation is True
        self.calls.append("application.hold")
        return _hold_transition()

    def end_episode(self, *, reason: str, force_cancellation: bool):
        assert reason
        assert force_cancellation is True
        self.calls.append("application.end_episode")
        return _hold_transition()

    def close(self) -> None:
        self.calls.append("application.close")
        self.started = False


class _FakeTraceCapture:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.trace_runtime = SimpleNamespace(
            session=SimpleNamespace(
                runtime_session_id=RUNTIME_SESSION_ID,
                control_domain_id=CONTROL_DOMAIN_ID,
                profile_id=TRACE_PROFILE_ID,
            )
        )

    def prepare(self, *, now: float):
        assert now == 10.0
        self.calls.append("trace.prepare")
        return {"episode_uuid": "episode:cube-lift-agent-skill-test"}

    def finalize(self, *, now: float):
        assert now == 10.0
        self.calls.append("trace.finalize")

    def complete_runtime(self):
        self.calls.append("trace.complete_runtime")
        return SimpleNamespace(
            runtime_session_id=RUNTIME_SESSION_ID,
            control_domain_id=CONTROL_DOMAIN_ID,
            trace_terminal_receipt_ref="trace-terminal:test",
            trace_commit_receipt_ref="trace-commit:test",
            semantic_content_hash="trace-finalization:test",
        )

    def abort(self, *, now: float):
        assert now == 10.0
        self.calls.append("trace.abort")

    def close(self, *, now: float):
        assert now == 10.0
        self.calls.append("trace.close")


class _FakeCandidateObserver:
    runtime_session_id = RUNTIME_SESSION_ID
    control_domain_id = CONTROL_DOMAIN_ID
    baseline_resting_center_z_m = 0.026

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def capture(self) -> CubeLiftCandidateObservation:
        self.calls.append("candidate.capture")
        return CubeLiftCandidateObservation(
            observed_at=10.1,
            cube_center_z_m=0.07,
            left_finger_cube_contact=True,
            right_finger_cube_contact=True,
            table_cube_contact=False,
            finite_state=True,
            candidate_satisfied=True,
        )


class _FakeTerminalEvaluator:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def evaluate(self, **kwargs):
        assert kwargs["execution_lineage_complete"] is True
        assert kwargs["execution_evidence_chain_ref"] == "execution-chain:test"
        assert kwargs["terminal_state_hash"] == "terminal-state:test"
        self.calls.append("terminal.evaluate")
        return SimpleNamespace(
            evaluation_receipt=SimpleNamespace(semantic_content_hash="evaluation:test"),
            aggregate_receipt=SimpleNamespace(
                semantic_content_hash="aggregate:test",
                verdict="succeeded",
            ),
            task_evaluation={
                "schema_version": "cube_lift_task_evaluation.v1",
                "verdict": "succeeded",
            },
        )


def _lifecycle(calls: list[str]):
    route, evaluator_config, binding, plan, request = _contracts()
    app = _FakeApplication(calls)
    trace = _FakeTraceCapture(calls)
    collector = CubeLiftExecutionEvidenceCollector(
        runtime_session_id=RUNTIME_SESSION_ID,
        control_domain_id=CONTROL_DOMAIN_ID,
    )
    collector.observe_cycle = lambda cycle, measurement_snapshot: calls.append(
        "collector.observe_cycle"
    )
    collector.seal_for_hold = lambda proof: (
        calls.append("collector.seal_for_hold") or _execution_cut()
    )
    lifecycle = CubeLiftAgentSkillLifecycle(
        application=app,
        route=route,
        deployment_binding=binding,
        evaluation_plan=plan,
        evaluator_config=evaluator_config,
        station_manifest=SimpleNamespace(
            semantic_content_hash=binding.station_manifest_hash
        ),
        execution_manifest=SimpleNamespace(
            semantic_content_hash=binding.execution_manifest_hash
        ),
        trace_capture=trace,
        evidence_collector=collector,
        candidate_observer=_FakeCandidateObserver(calls),
        terminal_evaluator=_FakeTerminalEvaluator(calls),
        max_control_cycles=3,
        clock=lambda: 10.0,
        sleep=lambda _: None,
    )
    return lifecycle, app, trace, binding, request


def test_success_orders_hold_evaluation_episode_commit_and_runtime_terminal() -> None:
    calls: list[str] = []
    lifecycle, _, _, _, request = _lifecycle(calls)

    activation = lifecycle.activate(
        request=request,
        operation=_operation(request, status="admitted"),
        cancel_requested=lambda: False,
    )
    completion = lifecycle.run(
        request=request,
        operation=_operation(request, status="running"),
        cancel_requested=lambda: False,
    )

    assert activation.succeeded is True
    assert activation.lower_operation_id == "lower:cube-lift-agent-skill-test"
    assert completion.succeeded is True
    assert completion.terminal_hold_confirmed is True
    assert completion.evaluation is not None
    assert completion.evaluation.task_outcome == "succeeded"
    assert completion.execution_result["trace_terminal_receipt_ref"] == (
        "trace-terminal:test"
    )
    assert calls == [
        "application.start",
        "trace.prepare",
        "application.step",
        "measurement.snapshot",
        "collector.observe_cycle",
        "candidate.capture",
        "application.end_episode",
        "collector.seal_for_hold",
        "terminal.evaluate",
        "trace.finalize",
        "application.close",
        "trace.complete_runtime",
        "trace.close",
    ]

    lifecycle.close()
    assert calls[-1] == "trace.close"


def test_cancel_forces_lower_hold_then_close_aborts_uncommitted_episode() -> None:
    calls: list[str] = []
    lifecycle, _, _, _, request = _lifecycle(calls)
    lifecycle.activate(
        request=request,
        operation=_operation(request, status="admitted"),
        cancel_requested=lambda: False,
    )

    control = lifecycle.cancel(
        request=request,
        operation=_operation(request, status="running"),
        reason="agent_cancel",
    )
    lifecycle.close()

    assert control.succeeded is True
    assert control.lower_hold_confirmed is True
    assert calls == [
        "application.start",
        "trace.prepare",
        "application.hold",
        "trace.abort",
        "application.close",
        "trace.close",
    ]


def test_constructor_rejects_non_policy_only_application_profile() -> None:
    calls: list[str] = []
    lifecycle, app, trace, binding, request = _lifecycle(calls)
    assert lifecycle.executor_id == binding.executor_id
    app.config.initial_human_alpha = 0.5

    route, evaluator_config, _, plan, _ = _contracts()
    collector = CubeLiftExecutionEvidenceCollector(
        runtime_session_id=RUNTIME_SESSION_ID,
        control_domain_id=CONTROL_DOMAIN_ID,
    )
    with pytest.raises(ValueError, match="exact policy-only config"):
        CubeLiftAgentSkillLifecycle(
            application=app,
            route=route,
            deployment_binding=binding,
            evaluation_plan=plan,
            evaluator_config=evaluator_config,
            station_manifest=SimpleNamespace(
                semantic_content_hash=binding.station_manifest_hash
            ),
            execution_manifest=SimpleNamespace(
                semantic_content_hash=binding.execution_manifest_hash
            ),
            trace_capture=trace,
            evidence_collector=collector,
            candidate_observer=_FakeCandidateObserver(calls),
            terminal_evaluator=_FakeTerminalEvaluator(calls),
            max_control_cycles=3,
        )

    assert request.runtime_session_id == RUNTIME_SESSION_ID
