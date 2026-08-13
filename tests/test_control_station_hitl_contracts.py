from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_station.hitl import (
    EXCLUSIVE_HITL_POLICY_TO_HUMAN,
    ExclusiveHITLTransitionObservation,
    ExclusiveHITLTransitionPort,
    build_hitl_activation_prime_evidence,
    run_exclusive_hitl_switch_transaction,
    run_exclusive_hitl_switch_via_port,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_exclusive_switch_preserves_primary_failure_when_hold_observer_fails() -> None:
    recorded = []

    def fail_transition(_target_source: str):
        raise RuntimeError("primary-transition-failure")

    def fail_hold_observer() -> tuple[str, ...]:
        raise RuntimeError("hold-observer-failure")

    with pytest.raises(RuntimeError, match="primary-transition-failure"):
        run_exclusive_hitl_switch_transaction(
            trigger_intent_id="intent:failure-test",
            trigger_intent_hash="intent-hash:failure-test",
            direction=EXCLUSIVE_HITL_POLICY_TO_HUMAN,
            current_source="policy",
            requested_at=10.0,
            clock_domain_id="test.monotonic",
            transition=fail_transition,
            evidence_observer=recorded.append,
            hold_after_failure=lambda: None,
            failure_hold_evidence=fail_hold_observer,
            clock=lambda: 10.1,
        )

    assert len(recorded) == 1
    assert recorded[0].status == "failed"
    assert recorded[0].hold_evidence_refs == ()
    assert recorded[0].error == "RuntimeError:primary-transition-failure"


def test_exclusive_switch_observer_cannot_become_control_gate() -> None:
    hold_reasons: list[str] = []

    def transition(_target_source: str):
        return {"control_mode": "human_teleop"}, ExclusiveHITLTransitionObservation(
            target_source="human",
            completed_at=20.1,
            hold_evidence_refs=("hold:20",),
            home_capture_id="home:20",
            home_capture_hash="home-hash:20",
            home_capture={
                "capture_id": "home:20",
                "components": ("tool_a", "gripper_b", "auxiliary_c"),
            },
            source_selection_record_id="selection:20",
            source_selection_record_hash="selection-hash:20",
            activation_id="activation:20",
            activation_hash="activation-hash:20",
            target_emission_id="human-emission:20",
            target_emission_hash="human-emission-hash:20",
            target_id="target:20",
            target_hash="target-hash:20",
            target_admission_receipt_id="target-admission:20",
            target_completion_receipt_id="target-completion:20",
            command_id="command:20",
            dispatch_receipt_id="dispatch:20",
            source_reset_performed=False,
        )

    def fail_observer(_evidence) -> None:
        raise RuntimeError("trace-unavailable")

    result, evidence = run_exclusive_hitl_switch_transaction(
        trigger_intent_id="intent:observer-failure",
        trigger_intent_hash="intent-hash:observer-failure",
        direction=EXCLUSIVE_HITL_POLICY_TO_HUMAN,
        current_source="policy",
        requested_at=20.0,
        clock_domain_id="test.monotonic",
        transition=transition,
        evidence_observer=fail_observer,
        hold_after_failure=lambda: hold_reasons.append("hold"),
        failure_hold_evidence=lambda: (),
        clock=lambda: 20.2,
    )

    assert result == {"control_mode": "human_teleop"}
    assert evidence.status == "ready"
    assert hold_reasons == []


def test_activation_evidence_accepts_model_and_topology_opaque_identities() -> None:
    activation = SimpleNamespace(
        activation_id="activation:any-backend",
        semantic_content_hash="activation-hash:any-backend",
        input_barrier_at=30.0,
    )
    policy_emission = SimpleNamespace(
        emission_id="policy-emission:any-model",
        semantic_content_hash="policy-emission-hash:any-model",
        emitted_at=30.2,
    )
    policy_submission = SimpleNamespace(
        submission_id="submission:any-topology",
        semantic_content_hash="submission-hash:any-topology",
    )

    evidence = build_hitl_activation_prime_evidence(
        activation_scope=activation,
        primed_at=30.3,
        clock_domain_id="backend.monotonic",
        policy_required=True,
        human_required=False,
        policy_reset_performed=True,
        policy_reset_at=30.1,
        policy_emission=policy_emission,
        human_emission=None,
        policy_submission=policy_submission,
        human_submission=None,
    )

    assert evidence is not None
    assert evidence.target_emission_identity("policy") == (
        "policy-emission:any-model",
        "policy-emission-hash:any-model",
    )


def test_exclusive_switch_port_is_backend_neutral_and_preserves_evidence() -> None:
    class _Port:
        def transition(self, selected_source: str):
            assert selected_source == "human"
            return {"backend": "opaque"}, ExclusiveHITLTransitionObservation(
                target_source="human",
                completed_at=40.1,
                hold_evidence_refs=("lower-hold:40",),
                home_capture_id="home:40",
                home_capture_hash="home-hash:40",
                home_capture={"anchors": {"component-x": "opaque"}},
                source_selection_record_id="selection:40",
                source_selection_record_hash="selection-hash:40",
                activation_id="activation:40",
                activation_hash="activation-hash:40",
                target_emission_id="human-emission:40",
                target_emission_hash="human-emission-hash:40",
                target_id="target:40",
                target_hash="target-hash:40",
                target_admission_receipt_id="target-admission:40",
                target_completion_receipt_id="target-completion:40",
                command_id="command:40",
                dispatch_receipt_id="dispatch:40",
                source_reset_performed=False,
            )

        def hold_after_failure(self) -> None:
            raise AssertionError("successful transition must not request HOLD")

        def failure_hold_evidence(self) -> tuple[str, ...]:
            return ()

    port = _Port()
    assert isinstance(port, ExclusiveHITLTransitionPort)
    result, evidence = run_exclusive_hitl_switch_via_port(
        trigger_intent_id="intent:port",
        trigger_intent_hash="intent-hash:port",
        direction=EXCLUSIVE_HITL_POLICY_TO_HUMAN,
        current_source="policy",
        requested_at=40.0,
        clock_domain_id="opaque.monotonic",
        port=port,
        evidence_observer=None,
        clock=lambda: 40.2,
    )

    assert result == {"backend": "opaque"}
    assert evidence.status == "ready"
    assert evidence.home_capture["anchors"]["component-x"] == "opaque"
