from __future__ import annotations

import ast
import inspect

import pytest

from control_composition.lifecycle import (
    AuthorityReleaseReceipt,
    ExecutionHoldReceipt,
    ExecutionReadyReceipt,
    HumanAuthorityTakeoverReceipt,
    HumanReadinessReceipt,
    HumanTakeoverOperationEvidence,
    HumanTakeoverOperationSpec,
    HumanTakeoverRequest,
    MixedControlBoundaryPolicy,
    SourceInvalidationReceipt,
    apply_authority_release,
    apply_execution_hold,
    apply_execution_ready,
    apply_human_authority_takeover,
    apply_human_readiness,
    apply_source_invalidation,
    begin_human_takeover,
    fail_human_takeover,
)


def _policy(
    *,
    suspension: str = "hold_and_release_authority",
    episode: str = "capture_only",
) -> MixedControlBoundaryPolicy:
    return MixedControlBoundaryPolicy(
        policy_id=f"mixed-boundary:{suspension}:{episode}",
        policy_suspension_mode=suspension,
        episode_boundary_motion_policy=episode,
        human_emission_max_age_sec=0.1,
        measured_state_max_age_sec=0.1,
    )


def _request() -> HumanTakeoverRequest:
    return HumanTakeoverRequest(
        request_id="handoff:policy-to-human:0",
        runtime_session_id="session:sim",
        control_domain_id="domain:arms",
        selection_group_id="arms",
        previous_source_slot_id="policy-slot",
        target_human_source_slot_id="human-slot",
        previous_producer_id="policy-main",
        target_human_producer_id="human-main",
        previous_producer_epoch=3,
        authority_epoch=7,
        execution_epoch=11,
        requested_at=10.0,
        clock_domain_id="sim-monotonic",
        reason="explicit operator human takeover",
    )


def _hold(*, accepted: bool = True) -> ExecutionHoldReceipt:
    return ExecutionHoldReceipt(
        request_id=_request().request_id,
        prior_execution_epoch=11,
        execution_epoch=12,
        hold_acknowledged=accepted,
        completed_at=10.01,
        evidence_refs=("execution-cancellation:12", "backend-hold:12"),
    )


def _invalidation(
    *,
    invalidated: bool = True,
    worker_drain_status: str = "drained",
) -> SourceInvalidationReceipt:
    return SourceInvalidationReceipt(
        request_id=_request().request_id,
        producer_id="policy-main",
        prior_producer_epoch=3,
        producer_epoch=4,
        proposal_buffer_invalidated=invalidated,
        worker_drain_status=worker_drain_status,
        completed_at=10.02,
        evidence_refs=("producer-transition:4", "worker-drain:policy-main:3"),
    )


def _release(*, accepted: bool = True) -> AuthorityReleaseReceipt:
    return AuthorityReleaseReceipt(
        request_id=_request().request_id,
        released_producer_id="policy-main",
        prior_authority_epoch=7,
        authority_epoch=8,
        accepted=accepted,
        completed_at=10.03,
        evidence_refs=("authority-release:8",),
    )


def _takeover(*, accepted: bool = True) -> HumanAuthorityTakeoverReceipt:
    return HumanAuthorityTakeoverReceipt(
        request_id=_request().request_id,
        human_producer_id="human-main",
        prior_authority_epoch=8,
        authority_epoch=9,
        accepted=accepted,
        completed_at=10.04,
        evidence_refs=("authority-takeover:9",),
    )


def _readiness(
    *,
    emission_at: float = 10.05,
    measured_at: float = 10.045,
    deadman_at: float = 10.055,
    rebase_at: float = 10.05,
    completed_at: float = 10.06,
    safety_status: str = "safe",
) -> HumanReadinessReceipt:
    return HumanReadinessReceipt(
        request_id=_request().request_id,
        human_emission_id="human-emission:post-takeover:0",
        human_emission_available_at=emission_at,
        measured_state_id="measured-state:post-hold:0",
        measured_state_available_at=measured_at,
        deadman_rearm_event_id="deadman:release-press:0",
        deadman_rearmed_at=deadman_at,
        retargeter_rebase_receipt_id="retargeter-rebase:0",
        retargeter_rebased_at=rebase_at,
        canonical_safety_receipt_id="canonical-safety:0",
        canonical_safety_status=safety_status,
        completed_at=completed_at,
    )


def _ready(*, accepted: bool = True) -> ExecutionReadyReceipt:
    return ExecutionReadyReceipt(
        request_id=_request().request_id,
        execution_epoch=12,
        authority_epoch=9,
        accepted=accepted,
        completed_at=10.07,
        evidence_refs=("execution-gate-ready:12",),
    )


def _through_takeover():
    state = begin_human_takeover(_policy(), _request())
    state = apply_execution_hold(state, _hold())
    state = apply_source_invalidation(state, _invalidation())
    state = apply_authority_release(state, _release())
    return apply_human_authority_takeover(state, _takeover())


def _through_legacy_safety_takeover():
    state = begin_human_takeover(
        MixedControlBoundaryPolicy(
            policy_id="mixed-boundary:legacy-safety-preview",
            policy_suspension_mode="hold_and_release_authority",
            episode_boundary_motion_policy="capture_only",
            require_canonical_safety=True,
        ),
        _request(),
    )
    state = apply_execution_hold(state, _hold())
    state = apply_source_invalidation(state, _invalidation())
    state = apply_authority_release(state, _release())
    return apply_human_authority_takeover(state, _takeover())


def _through_mainline_source_readiness_takeover():
    state = begin_human_takeover(
        MixedControlBoundaryPolicy(
            policy_id="mixed-boundary:source-readiness-mainline",
            policy_suspension_mode="hold_and_release_authority",
            episode_boundary_motion_policy="capture_only",
            require_canonical_safety=False,
        ),
        _request(),
    )
    state = apply_execution_hold(state, _hold())
    state = apply_source_invalidation(state, _invalidation())
    state = apply_authority_release(state, _release())
    return apply_human_authority_takeover(state, _takeover())


def _operation_spec() -> HumanTakeoverOperationSpec:
    return HumanTakeoverOperationSpec(
        previous_source_slot_id="policy-slot",
        target_human_source_slot_id="human-slot",
        previous_source_action_contract_hash="policy-action-contract:sha256:one",
        target_human_action_contract_hash="human-action-contract:sha256:one",
        execution_binding_hash="execution-binding:sha256:one",
        readiness_observer_binding_id="human-observer:canonical-emission-v1",
        boundary_policy=_policy(episode="hold_before_unbind"),
        readiness_timeout_sec=0.25,
        capture_boundary_id="episode-boundary:one",
    )


def test_human_takeover_stays_closed_until_post_takeover_readiness_and_ready() -> None:
    state = begin_human_takeover(_policy(), _request())
    assert state.phase == "awaiting_hold"
    assert state.execution_gate_status == "transitioning"
    assert not state.old_source_publication_enabled
    assert not state.human_publication_enabled

    hold = _hold()
    state = apply_execution_hold(state, hold)
    assert state.phase == "awaiting_source_invalidation"
    assert state.execution_gate_status == "hold"
    assert state.execution_epoch == 12
    assert apply_execution_hold(state, hold) is state

    invalidation = _invalidation()
    state = apply_source_invalidation(state, invalidation)
    assert state.producer_epoch == 4
    assert apply_source_invalidation(state, invalidation) is state

    release = _release()
    state = apply_authority_release(state, release)
    assert state.authority_epoch == 8
    assert apply_authority_release(state, release) is state

    takeover = _takeover()
    state = apply_human_authority_takeover(state, takeover)
    assert state.phase == "awaiting_human_readiness"
    assert state.authority_epoch == 9
    assert state.execution_gate_status == "hold"
    assert not state.human_publication_enabled

    readiness = _readiness()
    state = apply_human_readiness(state, readiness)
    assert state.phase == "awaiting_ready"
    assert not state.human_publication_enabled
    assert apply_human_readiness(state, readiness) is state

    ready = _ready()
    state = apply_execution_ready(state, ready)
    assert state.phase == "ready"
    assert state.execution_gate_status == "ready"
    assert state.human_publication_enabled
    assert not state.old_source_publication_enabled
    assert apply_execution_ready(state, ready) is state
    assert state.semantic_content_hash == state.semantic_content_hash


def test_operation_spec_binds_compatibility_without_runtime_evidence() -> None:
    spec = _operation_spec()
    assert spec.operation_spec_id.startswith("human-takeover-spec:")
    assert spec.previous_source_slot_id == _request().previous_source_slot_id
    assert spec.target_human_source_slot_id == _request().target_human_source_slot_id
    assert not hasattr(spec, "human_emission_id")
    assert not hasattr(spec, "authority_epoch")

    with pytest.raises(ValueError, match="hold-before-unbind"):
        HumanTakeoverOperationSpec(
            previous_source_slot_id="policy-slot",
            target_human_source_slot_id="human-slot",
            previous_source_action_contract_hash="policy-action-contract:one",
            target_human_action_contract_hash="human-action-contract:one",
            execution_binding_hash="execution-binding:one",
            readiness_observer_binding_id="human-observer:one",
            boundary_policy=_policy(episode="capture_only"),
            readiness_timeout_sec=0.25,
        )


def test_operation_evidence_binds_ready_terminal_state() -> None:
    state = apply_execution_ready(
        apply_human_readiness(_through_takeover(), _readiness()),
        _ready(),
    )
    request = _request()
    evidence = HumanTakeoverOperationEvidence(
        operation_spec_id=_operation_spec().operation_spec_id,
        source_intent_id="station-intent:takeover:one",
        request_id=request.request_id,
        request_semantic_hash=request.semantic_content_hash,
        final_state_semantic_hash=state.semantic_content_hash,
        previous_source_slot_id=request.previous_source_slot_id,
        target_human_source_slot_id=request.target_human_source_slot_id,
        final_phase=state.phase,
        execution_gate_status=state.execution_gate_status,
        succeeded=True,
        completed_at=10.08,
        clock_domain_id=request.clock_domain_id,
        evidence_refs=(
            state.hold_receipt.semantic_content_hash,
            state.invalidation_receipt.semantic_content_hash,
            state.release_receipt.semantic_content_hash,
            state.takeover_receipt.semantic_content_hash,
            state.readiness_receipt.semantic_content_hash,
            state.ready_receipt.semantic_content_hash,
        ),
    )
    assert evidence.evidence_id.startswith("human-takeover-evidence:")
    with pytest.raises(ValueError, match="must bind READY"):
        HumanTakeoverOperationEvidence(
            **{
                **evidence.__dict__,
                "final_phase": "failed",
            }
        )


def test_keep_reference_suspension_cannot_be_used_for_handoff() -> None:
    with pytest.raises(ValueError, match="cannot hand off"):
        begin_human_takeover(
            _policy(suspension="suspend_policy_keep_reference"),
            _request(),
        )


def test_external_readiness_failure_is_terminal_and_fail_closed() -> None:
    state = _through_takeover()
    failed = fail_human_takeover(
        state,
        reason="human_readiness_timeout",
    )
    assert failed.phase == "failed"
    assert failed.execution_gate_status == "hold"
    assert not failed.human_publication_enabled
    assert (
        fail_human_takeover(failed, reason="human_readiness_timeout") is failed
    )


def test_hold_failure_faults_without_opening_human_publication() -> None:
    state = begin_human_takeover(_policy(), _request())
    failed = apply_execution_hold(state, _hold(accepted=False))
    assert failed.phase == "failed"
    assert failed.execution_gate_status == "fault"
    assert failed.failure_reason == "execution_hold_not_acknowledged"
    assert not failed.human_publication_enabled


@pytest.mark.parametrize(
    ("invalidated", "drain"),
    ((False, "drained"), (True, "failed")),
)
def test_old_source_must_be_invalidated_and_drained(
    invalidated: bool,
    drain: str,
) -> None:
    state = apply_execution_hold(
        begin_human_takeover(_policy(), _request()),
        _hold(),
    )
    failed = apply_source_invalidation(
        state,
        _invalidation(invalidated=invalidated, worker_drain_status=drain),
    )
    assert failed.phase == "failed"
    assert failed.execution_gate_status == "hold"
    assert not failed.human_publication_enabled


def test_authority_transfer_failure_remains_hold() -> None:
    state = apply_source_invalidation(
        apply_execution_hold(
            begin_human_takeover(_policy(), _request()),
            _hold(),
        ),
        _invalidation(),
    )
    release_failed = apply_authority_release(state, _release(accepted=False))
    assert release_failed.phase == "failed"
    assert release_failed.execution_gate_status == "hold"

    state = apply_authority_release(state, _release())
    takeover_failed = apply_human_authority_takeover(
        state,
        _takeover(accepted=False),
    )
    assert takeover_failed.phase == "failed"
    assert takeover_failed.execution_gate_status == "hold"
    assert not takeover_failed.human_publication_enabled


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"emission_at": 10.039}, "after authority takeover"),
        ({"deadman_at": 10.039}, "Deadman must be re-armed"),
        ({"rebase_at": 10.039}, "Retargeter must be re-based"),
        ({"rebase_at": 10.061}, "completion predates"),
        ({"measured_at": 10.0}, "fresh relative to the HOLD"),
        ({"emission_at": 10.05, "completed_at": 10.151}, "emission is stale"),
        ({"measured_at": 10.045, "completed_at": 10.146}, "measured state is stale"),
    ),
)
def test_stale_or_pre_takeover_human_readiness_is_rejected(
    kwargs: dict[str, float],
    message: str,
) -> None:
    state = _through_takeover()
    with pytest.raises(ValueError, match=message):
        apply_human_readiness(state, _readiness(**kwargs))
    assert state.phase == "awaiting_human_readiness"
    assert state.execution_gate_status == "hold"
    assert not state.human_publication_enabled


@pytest.mark.parametrize("safety_status", ("hold", "fault"))
def test_unsafe_first_human_command_fails_closed_in_hold(safety_status: str) -> None:
    failed = apply_human_readiness(
        _through_legacy_safety_takeover(),
        _readiness(safety_status=safety_status),
    )
    assert failed.phase == "failed"
    assert failed.execution_gate_status == "hold"
    assert failed.failure_reason == f"human_readiness_{safety_status}"
    assert not failed.human_publication_enabled


def test_canonical_safety_modified_is_still_an_accepted_safe_verdict() -> None:
    state = apply_human_readiness(
        _through_legacy_safety_takeover(),
        _readiness(safety_status="modified"),
    )
    assert state.phase == "awaiting_ready"
    assert not state.human_publication_enabled


def test_mainline_human_source_readiness_does_not_require_safety_preview() -> None:
    readiness = HumanReadinessReceipt(
        **{
            **_readiness().__dict__,
            "canonical_safety_receipt_id": None,
            "canonical_safety_status": None,
        }
    )
    state = apply_human_readiness(
        _through_mainline_source_readiness_takeover(),
        readiness,
    )
    assert state.phase == "awaiting_ready"
    assert state.readiness_receipt is readiness


def test_ready_receipt_cannot_change_epoch_or_bypass_readiness() -> None:
    state = _through_takeover()
    with pytest.raises(RuntimeError, match="awaiting_human_readiness"):
        apply_execution_ready(state, _ready())

    state = apply_human_readiness(state, _readiness())
    wrong_epoch = ExecutionReadyReceipt(
        request_id=_request().request_id,
        execution_epoch=13,
        authority_epoch=9,
        accepted=True,
        completed_at=10.07,
        evidence_refs=("execution-gate-ready:13",),
    )
    with pytest.raises(ValueError, match="cannot change"):
        apply_execution_ready(state, wrong_epoch)
    failed = apply_execution_ready(state, _ready(accepted=False))
    assert failed.phase == "failed"
    assert failed.execution_gate_status == "hold"


def test_episode_boundary_policy_is_explicit_and_handoff_contract_is_pure() -> None:
    legacy = _policy(episode="capture_only")
    mixed = _policy(episode="hold_before_unbind")
    assert legacy.semantic_content_hash != mixed.semantic_content_hash
    assert legacy.episode_boundary_motion_policy == "capture_only"
    assert mixed.episode_boundary_motion_policy == "hold_before_unbind"

    import control_composition.lifecycle.handoff as handoff

    tree = ast.parse(inspect.getsource(handoff))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        module.startswith(("deployments", "hardware", "sim", "teleop_runtime"))
        for module in imports
    )
