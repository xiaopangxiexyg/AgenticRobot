from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.protocol.contracts import (
    AgentSkillOperation,
    AgentSkillRequest,
)
from agent_gateway.profiles.mujoco.skills.cube_lift.terminal_evaluator import (
    CubeLiftLowerHoldProof,
    CubeLiftTerminalHoldSample,
    CubeLiftTerminalStableHoldEvaluator,
    MujocoCubeLiftTerminalHoldSampler,
    cube_lift_lower_hold_proof_from_transition,
)
from agent_gateway.skills.cube_lift.definition import (
    cube_lift_schema_entry_v1,
    cube_lift_skill_definition_v1,
)
from agent_gateway.profiles.mujoco.skills.cube_lift.sim_route import (
    CubeLiftSimDeploymentBinding,
    CubeLiftTerminalEvaluatorConfig,
    cube_lift_evaluation_plan_v1,
    cube_lift_sim_route_v1,
)

TASK_HASH = "task-config:cube-lift-test"


def _config() -> CubeLiftTerminalEvaluatorConfig:
    return CubeLiftTerminalEvaluatorConfig(
        task_config_hash=TASK_HASH,
        object_body_name="debug_cube",
        left_finger_body_name="right_gripper_left_finger",
        right_finger_body_name="right_gripper_right_finger",
        table_body_name="table",
        lift_delta_m=0.04,
        stable_hold_physics_ticks=100,
        physics_hz=1000.0,
    )


def _contracts(config: CubeLiftTerminalEvaluatorConfig):
    definition = cube_lift_skill_definition_v1()
    schema = cube_lift_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash
    )
    route = cube_lift_sim_route_v1(definition=definition, schema_entry=schema)
    binding = CubeLiftSimDeploymentBinding(
        deployment_binding_id="sim.mujoco.cube_lift.policy_only.test",
        deployment_binding_version="1",
        route_hash=route.semantic_content_hash,
        runtime_session_id="runtime:cube-lift-terminal-test",
        control_domain_id="mujoco-sim-hil",
        target_binding_id="target:cube-lift-terminal-test",
        station_manifest_hash="station-manifest:cube-lift-terminal-test",
        execution_manifest_hash="execution-manifest:cube-lift-terminal-test",
        execution_domain_binding_hash="execution-domain:cube-lift-terminal-test",
        target_runtime_action_contract_hash="action-contract:test",
        component_target_capability_hash="component-capability:test",
        action_schema_hash="action-schema:test",
        observation_plan_hash="observation-plan:test",
        frame_graph_hash="frame-graph:test",
        tool_config_hash="tool:test",
        calibration_hash="calibration:test",
        task_config_hash=TASK_HASH,
        policy_checkpoint_hash="checkpoint:test",
        evaluator_configuration_hash=config.semantic_content_hash,
        trace_requirement_profile_id="trace-profile:test",
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
        evaluator_config=config,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    profile = route.allowed_parameter_profiles[0]
    request = AgentSkillRequest(
        client_id="rpent:cube-lift-terminal-test",
        request_id="request:cube-lift-terminal-test",
        idempotency_key="idempotency:cube-lift-terminal-test",
        client_sequence=1,
        runtime_session_id=binding.runtime_session_id,
        skill_route_manifest_hash="route-manifest:cube-lift-terminal-test",
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
        observation_snapshot_id="snapshot:cube-lift-terminal-test",
        requested_mode="sim_active",
        requested_timeout_ms=30_000,
    )
    operation = AgentSkillOperation(
        operation_id="operation:cube-lift-terminal-test",
        request_hash=request.semantic_content_hash,
        runtime_session_id=request.runtime_session_id,
        status="running",
        reason_code="station_activation_accepted_not_completed",
        station_revision_at_admission=4,
        current_station_revision=4,
        started_at=10.0,
        updated_at=10.1,
        deadline_at=40.0,
        lower_operation_id="lower:cube-lift-terminal-test",
        execution_receipt_refs=("station-start:1",),
        progress_summary={"phase": "running"},
    )
    return route, binding, plan, request, operation


def _hold_proof() -> CubeLiftLowerHoldProof:
    return CubeLiftLowerHoldProof(
        runtime_session_id="runtime:cube-lift-terminal-test",
        control_domain_id="mujoco-sim-hil",
        hold_receipt_ref="mujoco-lower-hold:test",
        gate_snapshot_hash="gate-snapshot:test",
        gate_epoch=7,
        cancellation_receipt_ref="execution-cancellation-receipt:test",
        confirmed_at=10.2,
        evidence_refs=("gate-event:test", "execution-cancellation:test"),
    )


class _FakeSampler:
    runtime_session_id = "runtime:cube-lift-terminal-test"
    control_domain_id = "mujoco-sim-hil"
    task_config_hash = TASK_HASH
    realtime_worker = False

    def __init__(
        self,
        *,
        z_m: float = 0.07,
        bilateral: bool = True,
        table_contact: bool = False,
        tick_jump_at: int | None = None,
    ) -> None:
        self.z_m = z_m
        self.bilateral = bilateral
        self.table_contact = table_contact
        self.tick_jump_at = tick_jump_at
        self.count = 0
        self.validated = 0

    def validate_hold(self, proof):
        assert proof == _hold_proof()
        self.validated += 1

    def step_and_capture(self, *, proof):
        assert proof == _hold_proof()
        self.count += 1
        tick = 500 + self.count
        if self.tick_jump_at == self.count:
            tick += 1
        return CubeLiftTerminalHoldSample(
            physics_tick_sequence=tick,
            measured_at=10.2 + self.count / 1000.0,
            cube_center_z_m=self.z_m,
            left_finger_cube_contact=self.bilateral,
            right_finger_cube_contact=self.bilateral,
            table_cube_contact=self.table_contact,
            finite_state=True,
            source_measurement_ref=f"measurement:{self.count}",
        )


def _evaluate(sampler: _FakeSampler, *, lineage: bool = True):
    config = _config()
    route, binding, plan, request, operation = _contracts(config)
    evaluator = CubeLiftTerminalStableHoldEvaluator(
        config=config,
        sampler=sampler,
        clock=lambda: 10.5,
    )
    return evaluator.evaluate(
        operation=operation,
        request=request,
        route=route,
        deployment_binding=binding,
        evaluation_plan=plan,
        lower_hold_proof=_hold_proof(),
        baseline_resting_center_z_m=0.026,
        execution_evidence_chain_ref="execution-chain:test",
        terminal_state_hash="terminal-state:test",
        execution_lineage_complete=lineage,
    )


def test_exact_100_tick_stable_hold_produces_required_success_receipts() -> None:
    sampler = _FakeSampler()
    bundle = _evaluate(sampler)
    assert sampler.count == 100
    assert bundle.task_evaluation["verdict"] == "succeeded"
    assert bundle.task_evaluation["stable_hold_observed_physics_ticks"] == 100
    assert bundle.evaluation_receipt.verdict == "succeeded"
    assert bundle.aggregate_receipt.verdict == "succeeded"
    assert bundle.aggregate_receipt.consumed_evaluation_receipt_hashes == (
        bundle.evaluation_receipt.semantic_content_hash,
    )
    evidence_wire = bundle.evidence.to_wire()
    assert len(evidence_wire["samples"]) == 100
    assert "qpos" not in str(evidence_wire).lower()
    assert "action" not in str(evidence_wire).lower()


@pytest.mark.parametrize(
    ("sampler", "reason"),
    (
        (_FakeSampler(z_m=0.065), "cube_below_required_lift"),
        (_FakeSampler(bilateral=False), "bilateral_contact_not_stable"),
        (_FakeSampler(table_contact=True), "table_contact_observed"),
    ),
)
def test_terminal_predicate_failures_remain_task_failures(sampler, reason) -> None:
    bundle = _evaluate(sampler)
    assert bundle.task_evaluation["verdict"] == "failed"
    assert reason in bundle.task_evaluation["reason_codes"]
    assert bundle.evaluation_receipt.verdict == "failed"


def test_incomplete_execution_lineage_cannot_claim_task_success() -> None:
    bundle = _evaluate(_FakeSampler(), lineage=False)
    assert bundle.task_evaluation["verdict"] == "inconclusive"
    assert bundle.task_evaluation["execution_lineage_complete"] is False
    assert "execution_lineage_incomplete" in bundle.task_evaluation["reason_codes"]


def test_nonconsecutive_physics_ticks_abort_without_fabricating_evaluation() -> None:
    config = _config()
    route, binding, plan, request, operation = _contracts(config)
    evaluator = CubeLiftTerminalStableHoldEvaluator(
        config=config,
        sampler=_FakeSampler(tick_jump_at=23),
        clock=lambda: 10.5,
    )
    with pytest.raises(RuntimeError, match="not consecutive"):
        evaluator.evaluate(
            operation=operation,
            request=request,
            route=route,
            deployment_binding=binding,
            evaluation_plan=plan,
            lower_hold_proof=_hold_proof(),
            baseline_resting_center_z_m=0.026,
            execution_evidence_chain_ref="execution-chain:test",
            terminal_state_hash="terminal-state:test",
            execution_lineage_complete=True,
        )


def test_cancellation_preempts_evaluator_before_any_physics_tick() -> None:
    config = _config()
    route, binding, plan, request, operation = _contracts(config)
    sampler = _FakeSampler()
    evaluator = CubeLiftTerminalStableHoldEvaluator(
        config=config,
        sampler=sampler,
        clock=lambda: 10.5,
    )
    with pytest.raises(InterruptedError, match="cancelled"):
        evaluator.evaluate(
            operation=operation,
            request=request,
            route=route,
            deployment_binding=binding,
            evaluation_plan=plan,
            lower_hold_proof=_hold_proof(),
            baseline_resting_center_z_m=0.026,
            execution_evidence_chain_ref="execution-chain:test",
            terminal_state_hash="terminal-state:test",
            execution_lineage_complete=True,
            cancel_requested=lambda: True,
        )
    assert sampler.count == 0


class _FakeScene:
    def __init__(self) -> None:
        self.body_names = {
            "debug_cube",
            "right_gripper_left_finger",
            "right_gripper_right_finger",
            "table",
        }

    def has_body(self, body_name):
        return body_name in self.body_names

    def state_is_finite(self):
        return True

    def body_pose(self, body_name):
        assert body_name == "debug_cube"
        pose = np.eye(4)
        pose[2, 3] = 0.07
        return pose

    def contacts(self):
        return (
            SimpleNamespace(
                body1_name="debug_cube",
                body2_name="right_gripper_left_finger",
            ),
            SimpleNamespace(
                body1_name="right_gripper_right_finger",
                body2_name="debug_cube",
            ),
        )


class _FakeRig:
    def __init__(self) -> None:
        self.config = SimpleNamespace(realtime_worker=False)
        self.scene = _FakeScene()
        self.tick = 40

    def snapshot(self):
        return {"physics_ticks": self.tick}

    def step_physics(self):
        self.tick += 1

    def capture_scene(self, capture):
        return capture(self.scene)


def test_mujoco_sampler_uses_existing_held_rig_without_new_control_path() -> None:
    rig = _FakeRig()
    runtime = SimpleNamespace(
        rig=rig,
        config=SimpleNamespace(
            runtime_session_id="runtime:cube-lift-terminal-test",
            control_domain_id="mujoco-sim-hil",
        ),
        capture_execution_gate_snapshot=lambda **_kwargs: SimpleNamespace(
            status="hold",
            control_domain_id="mujoco-sim-hil",
            gate_epoch=7,
        ),
    )
    application = SimpleNamespace(
        runtime=runtime,
        started=True,
        held=True,
        closed=False,
    )
    task = SimpleNamespace(
        object_body_name="debug_cube",
        grasp_latch_left_finger_body_name="right_gripper_left_finger",
        grasp_latch_right_finger_body_name="right_gripper_right_finger",
        table_body_name="table",
        physics_hz=1000.0,
    )
    sampler = MujocoCubeLiftTerminalHoldSampler(
        application=application,
        task_config=task,
        task_config_hash=TASK_HASH,
        evaluator_config=_config(),
        clock=lambda: 10.3,
    )
    sample = sampler.step_and_capture(proof=_hold_proof())
    assert rig.tick == 41
    assert sample.physics_tick_sequence == 41
    assert sample.left_finger_cube_contact
    assert sample.right_finger_cube_contact
    assert not sample.table_cube_contact


def test_lower_hold_projection_requires_complete_cancellation_receipt() -> None:
    receipt = SimpleNamespace(
        cancellation_id="execution-cancellation:test",
        required_ack_levels=("backend_hold",),
        acknowledged_levels=("backend_hold",),
        pending_levels=(),
        rejected_levels=(),
        acknowledgement_deadline_at=10.3,
        observed_at=10.2,
        complete=True,
        timed_out=False,
        requires_fault_escalation=False,
        late_levels=(),
    )
    transition = SimpleNamespace(
        event=SimpleNamespace(
            event_id="gate-event:test",
            runtime_session_id="runtime:cube-lift-terminal-test",
            control_domain_id="mujoco-sim-hil",
        ),
        snapshot=SimpleNamespace(
            control_domain_id="mujoco-sim-hil",
            status="hold",
            gate_epoch=7,
            semantic_content_hash="gate-snapshot:test",
        ),
        cancellation=SimpleNamespace(
            cancellation_id="execution-cancellation:test"
        ),
        cancellation_receipt=receipt,
        trace_facts=(SimpleNamespace(fact_id="trace-fact:hold"),),
    )
    proof = cube_lift_lower_hold_proof_from_transition(
        transition,
        runtime_session_id="runtime:cube-lift-terminal-test",
        control_domain_id="mujoco-sim-hil",
        confirmed_at=10.2,
    )
    assert proof.gate_epoch == 7
    assert proof.hold_receipt_ref.startswith("mujoco-lower-hold:")

    receipt.complete = False
    with pytest.raises(ValueError, match="does not prove"):
        cube_lift_lower_hold_proof_from_transition(
            transition,
            runtime_session_id="runtime:cube-lift-terminal-test",
            control_domain_id="mujoco-sim-hil",
            confirmed_at=10.2,
        )
