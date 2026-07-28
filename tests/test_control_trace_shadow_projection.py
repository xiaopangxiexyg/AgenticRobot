from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    DEPLOY_ACTION_VECTOR_ACTION_SPACE,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    build_deployment_observation,
    nan_vector,
)
from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    ActionPersistencePolicy,
    ActiveExecutionSnapshot,
    AuthorityEventDraft,
    SharedCanonicalSafetyResult,
    build_robot_action_v1_runtime_schema,
    evaluate_shared_execution_shadow,
)
from robot_motion.action.authoritative_trace import (
    AuthoritativeTraceContext,
    project_shared_execution_shadow_evaluation,
)
from teleop_runtime.control_trace import ControlTraceRuntime, RuntimeTraceSession
from teleop_runtime.control_trace.projection import (
    make_runtime_trace_terminal,
    project_authority_event,
    project_execution_gate_event,
    project_legacy_execution,
    project_shadow_startup,
    project_shadow_step,
)
from teleop_runtime.deploy.arbiter_shadow import (
    PassivePolicyArbiterShadow,
    PolicyArbiterShadowConfig,
)


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class Policy:
    name = "projection-policy"

    def predict(self, _observation):
        value = nan_vector(DEPLOY_ACTION_DIM)
        value[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
            [0.12, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        return value


def test_m3_shadow_projects_to_complete_non_authoritative_m4a_trace() -> None:
    clock = Clock()
    deploy_runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
        ),
        monotonic_clock=clock,
    )
    catalog = ActionComponentCatalog(
        catalog_id="trace-shadow-right-arm-v1",
        action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
        components=(
            ActionComponentSpec(
                component_id="right_arm",
                selection_group_id="right_arm",
                effector_name="right_arm",
            ),
        ),
    )
    coordinator = PassivePolicyArbiterShadow(
        PolicyArbiterShadowConfig(
            runtime_session_id="shadow-session",
            control_domain_id="robot-0",
            embodiment_instance_id="rm65-pair-0",
            producer_id="policy-main",
            active_effectors=("right_arm",),
            action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
            decision_hz=15.0,
            component_catalog=catalog,
            action_schema=build_robot_action_v1_runtime_schema(
                component_catalog=catalog,
            ),
        ),
        clock=clock,
    )
    trace = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-shadow-1",
            runtime_session_id="shadow-session",
            control_domain_id="robot-0",
            profile_id="policy_shadow_dry_run_v1",
            started_at=clock.value,
            clock_domain_id="local_monotonic",
        ),
        capacity=64,
    )
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    observation = build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=0,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=100.0,
    )
    prediction = deploy_runtime.predict_and_queue(
        Policy(), observation, now=clock.value, action_keyframe_period_sec=1.0 / 15.0
    )
    assert prediction.accepted
    pipeline = CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.08,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )
    action_step = deploy_runtime.prepare_next_action(
        observation, preparer=pipeline, policy_name="projection-policy"
    )
    assert action_step.emission is not None
    shadow_step = coordinator.evaluate_emission(action_step.emission)
    assert shadow_step.computation is not None
    assert shadow_step.evaluation is not None
    shared_execution = evaluate_shared_execution_shadow(
        shadow_step.computation,
        active_execution=ActiveExecutionSnapshot(
            control_domain_id="robot-0",
            execution_epoch=0,
            active_segment_by_group={},
        ),
        persistence_policy=ActionPersistencePolicy(
            policy_id="projection-shadow-segment-v1",
            selected_segment_max_lifetime_sec=0.1,
        ),
        shared_safety=lambda selected, result_id: SharedCanonicalSafetyResult(
            input_action_id=result_id,
            safe_action=selected,
            status="safe",
        ),
        now=clock.value,
        supports_local_group_stop=False,
        source_arbitration_evaluation_id=shadow_step.evaluation.evaluation_id,
    )
    shared_execution_fact = project_shared_execution_shadow_evaluation(
        AuthoritativeTraceContext(
            runtime_session_id="shadow-session",
            control_domain_id="robot-0",
            embodiment_instance_id="rm65-pair-0",
            clock_domain_id="local_monotonic",
        ),
        shared_execution,
        occurred_at=clock.value,
    )
    legacy = coordinator.observe_legacy_execution(
        emission=action_step.emission,
        shadow_step=shadow_step,
        canonical_action=action_step.prepared_action,
        execution=SimpleNamespace(
            accepted=True,
            mode="dry_run",
            dispatch_attempted=False,
            dispatch_completed=False,
        ),
        dispatch_error=None,
        observed_at=clock.value,
    )

    facts = (
        *project_shadow_startup(coordinator),
        *project_shadow_step(coordinator, shadow_step),
        shared_execution_fact,
        project_legacy_execution(coordinator, legacy),
        make_runtime_trace_terminal(
            coordinator,
            terminal_status="completed",
            reason="fixture_complete",
            occurred_at=clock.value,
        ),
    )
    for item in facts:
        assert trace.issue(item).accepted

    event_types = {record.fact.event_type for record in trace.records()}
    report = trace.completeness()
    assert "arbitration_evaluation" in event_types
    assert "policy_action_emission" in event_types
    assert "source_validation" in event_types
    assert "legacy_execution_observation" in event_types
    assert "shared_execution_evaluation" in event_types
    assert "arbitration_decision" not in event_types
    assert report.in_memory_complete
    assert report.lineage_complete
    assert report.lineage_issues == ()
    assert not report.capture_commit_eligible
    evaluation = next(
        record.fact for record in trace.records() if record.fact.event_type == "arbitration_evaluation"
    )
    assert evaluation.payload["non_authoritative"] is True
    assert "would_select_action" in evaluation.payload
    assert "selected_action" not in evaluation.payload
    shared_fact = next(
        record.fact
        for record in trace.records()
        if record.fact.event_type == "shared_execution_evaluation"
    )
    assert shared_fact.payload["non_authoritative"] is True
    assert shared_fact.payload["command_intent"] is not None
    assert "decision_id" not in shared_fact.payload

    incomplete = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-shadow-missing-proposal",
            runtime_session_id="shadow-session",
            control_domain_id="robot-0",
            profile_id="policy_shadow_dry_run_v1",
            started_at=clock.value,
            clock_domain_id="local_monotonic",
        ),
        capacity=64,
    )
    for item in facts:
        if item.event_type != "action_proposal":
            assert incomplete.issue(item).accepted
    incomplete_report = incomplete.completeness()
    assert not incomplete_report.lineage_complete
    assert not incomplete_report.in_memory_complete
    assert {
        issue.code for issue in incomplete_report.lineage_issues
    } >= {
        "accepted_admission_missing_proposal",
        "snapshot_candidate_missing_proposal",
    }

    gate_event = coordinator.update_execution_gate(
        "hold",
        reason="projection_contract_fixture",
        produced_at=clock.value,
    )
    assert gate_event is not None
    gate_fact = project_execution_gate_event(coordinator, gate_event)
    assert gate_fact.fact_id == gate_event.event_id
    assert gate_fact.payload["status"] == "hold"
    assert "audit_metadata" not in gate_fact.payload

    authority_event = coordinator.authority_store.admit(
        AuthorityEventDraft(
            runtime_session_id="shadow-session",
            control_domain_id="robot-0",
            sequence=0,
            event_type="mode",
            produced_at=clock.value,
            produced_clock_domain_id="local_monotonic",
            control_mode="hold",
            reason="projection_contract_fixture",
        )
    )
    authority_fact = project_authority_event(coordinator, authority_event)
    assert authority_fact.fact_id == authority_event.event_id
    assert authority_fact.payload["authority_event_type"] == "mode"
    assert "audit_metadata" not in authority_fact.payload


def test_explicit_projection_does_not_follow_new_runtime_dataclass_fields() -> None:
    from dataset_contracts.control_trace import CONTROL_TRACE_RECORD_SPEC_BY_EVENT_V1

    expected = CONTROL_TRACE_RECORD_SPEC_BY_EVENT_V1[
        "arbitration_evaluation"
    ].payload_fields
    assert "would_select_action" in expected
    assert "selected_action" not in expected
