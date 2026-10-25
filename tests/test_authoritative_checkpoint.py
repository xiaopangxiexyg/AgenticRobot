"""Core recovery uses independent fake backends; no native target claim."""

from dataclasses import fields, replace

import pytest

from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.authoritative_checkpoint import capture_authoritative_control
from robot_motion.action.authoritative_control import AuthoritativeControlRuntime
from robot_motion.action.proposal import ActionProposalDraft
from robot_motion.runtime.dispatch_owner import DispatchOwnerRegistry
from tests.test_mujoco_deploy_runtime import ManualClock, action, build_runtime

pytestmark = pytest.mark.unit


def _restore(state, *, config=None):
    target, _, clock = build_runtime(clock=ManualClock(state.source.captured_at))
    initial = target.core
    bindings = {
        name: getattr(initial, name)
        for name in (
            "dispatcher",
            "backend_hold",
            "backend_name",
            "resolver",
            "shared_safety",
            "action_representation_adapter",
            "target_execution_port",
            "trace_sink",
            "contextual_dispatcher",
            "clock",
        )
    }
    restored = AuthoritativeControlRuntime(
        initial.config if config is None else config,
        **bindings,
        _checkpoint=state,
    )
    target.core = restored
    target.rig.start()
    return restored, target.rig, clock


def _moving():
    runtime, _, clock = build_runtime()
    runtime.start()
    runtime.submit_policy_action(action(), policy_output_id="source-0")
    assert runtime.step(now=clock.value).status == "dispatched"
    return runtime, clock


def test_active_history_survives_pruning_and_expiry_exactly_once():
    source, clock = _moving()
    clock.value = 10.1
    source.core.proposals.prune_expired(at_or_before=clock.value)
    assert not len(source.core.proposals)
    state = capture_authoritative_control(source.core)
    assert state.state["_active_segment_by_group"]
    branch, rig, branch_clock = _restore(state)
    try:
        for at in (10.1, 10.2, 10.3):
            clock.value = branch_clock.value = at
            old = source.core.run_decision(now=at)
            new = branch.run_decision(now=at)
            assert old.status == new.status
            assert (
                old.directive.semantic_content_hash
                == new.directive.semantic_content_hash
            )
            assert bool(old.cancellation) == bool(new.cancellation)
            assert (
                source.core.capture_receiver_state() == branch.capture_receiver_state()
            )
        assert source.core.execution_epoch == branch.execution_epoch == 1
        assert source.core._previous_decision_id == branch._previous_decision_id
        assert [f.semantic_content_hash for f in source.core.trace_facts] == [
            f.semantic_content_hash for f in branch.trace_facts
        ]
    finally:
        rig.close()
        source.close()


def test_noninitial_epoch_ready_with_hold_remains_and_next_gate_is_monotonic():
    source, _clock = _moving()
    source.transition_execution_gate("hold", reason="test:hold")
    source.transition_execution_gate("ready", reason="test:ready")
    assert source.core._backend_hold_active
    assert source.core.execution_epoch == 1
    state = capture_authoritative_control(source.core)
    branch, rig, _ = _restore(state)
    try:
        assert branch.config.initial_execution_epoch == 0
        assert branch.execution_epoch == 1
        assert branch._backend_hold_active
        assert branch.dispatch_owner_token == source.core.dispatch_owner_token
        assert branch.capture_execution_gate_snapshot().status == "ready"
        old = source.core.transition_execution_gate(status="hold", reason="test:second")
        new = branch.transition_execution_gate(status="hold", reason="test:second")
        assert old.event == new.event
        assert old.snapshot == new.snapshot
        assert (
            branch._next_execution_gate_sequence
            == source.core._next_execution_gate_sequence
        )
    finally:
        rig.close()
        source.close()


def test_constructor_restore_does_not_regrant_or_emit_old_facts(monkeypatch):
    source, _ = _moving()
    state = capture_authoritative_control(source.core)
    target, _, _ = build_runtime(clock=ManualClock(state.source.captured_at))
    prototype = target.core
    calls = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Restoration must not grant a new token.")

    monkeypatch.setattr(DispatchOwnerRegistry, "grant_initial_owner", forbidden)
    branch = AuthoritativeControlRuntime(
        prototype.config,
        dispatcher=prototype.dispatcher,
        backend_hold=prototype.backend_hold,
        backend_name=prototype.backend_name,
        resolver=prototype.resolver,
        action_representation_adapter=prototype.action_representation_adapter,
        trace_sink=calls.append,
        clock=prototype.clock,
        _checkpoint=state,
    )
    assert calls == []
    assert branch.dispatch_owner_token == source.core.dispatch_owner_token
    assert branch._facts is not source.core._facts
    assert branch.coordinator is not source.core.coordinator
    source.close()


def test_core_rejects_torn_cuts_configuration_trace_and_unknown_state():
    source, _ = _moving()
    state = capture_authoritative_control(source.core)
    first = state.facts[0]
    for bad in (
        replace(state, configuration_hash="wrong"),
        replace(state, backend_name="another-backend"),
        replace(state, decision=replace(state.decision, captured_at=11.0)),
        replace(
            state,
            facts=(replace(first, semantic_content_hash="0" * 64), *state.facts[1:]),
        ),
        replace(state, state={**state.state, "unknown_owner": {}}),
        replace(state, state={**state.state, "_next_authority_sequence": 10}),
        replace(state, state={**state.state, "_previous_selection_by_group": {}}),
        replace(state, state={**state.state, "_terminal": True}),
    ):
        with pytest.raises(ValueError):
            _restore(bad)
    original = source.core._fact_hash_by_id
    source.core._fact_hash_by_id = {}
    with pytest.raises(ValueError, match="index"):
        capture_authoritative_control(source.core)
    source.core._fact_hash_by_id = original
    source.close()


def test_unknown_arbiter_state_is_not_silently_discarded():
    source, _ = _moving()
    source.core.arbiter.unrecorded_state = 1
    with pytest.raises(ValueError, match="static owner"):
        capture_authoritative_control(source.core)
    del source.core.arbiter.unrecorded_state
    source.close()


@pytest.mark.parametrize("later", ["dispatch", "cancel"])
def test_earlier_valid_segment_cannot_replace_the_latest_execution_state(later):
    source, clock = _moving()
    old = capture_authoritative_control(source.core)
    if later == "dispatch":
        clock.value += 1.0 / source.config.decision_hz
        source.submit_policy_action(action(0.1), policy_output_id="second")
        assert source.step(now=clock.value).status == "dispatched"
    else:
        source.transition_execution_gate("hold", reason="after-first")
    current = capture_authoritative_control(source.core)
    forged = dict(current.state)
    for name in (
        "_active_segment_by_group",
        "_previous_decision_id",
        "_previous_selection_by_group",
    ):
        forged[name] = old.state[name]
    with pytest.raises(ValueError, match="history"):
        _restore(replace(current, state=forged))
    source.close()


def test_rehashed_fact_from_another_embodiment_is_rejected():
    source, _ = _moving()
    state = capture_authoritative_control(source.core)
    old = state.facts[0]
    wrong = ControlTraceFact.create(
        **{
            name: getattr(old, name)
            for name in (
                "fact_id",
                "event_type",
                "runtime_session_id",
                "control_domain_id",
                "occurred_at",
                "clock_domain_id",
                "payload",
            )
        },
        embodiment_instance_id="different-robot",
    )
    with pytest.raises(ValueError, match="scope"):
        _restore(replace(state, facts=(wrong, *state.facts[1:])))
    source.close()


def test_two_api_valid_branches_cannot_mix_source_history():
    first, _ = _moving()
    second, _, clock = build_runtime()
    second.start()
    second.submit_policy_action(action(0.15), policy_output_id="source-0")
    second.step(now=clock.value)
    a = capture_authoritative_control(first.core)
    b = capture_authoritative_control(second.core)
    assert (
        a.source.proposal_records[0].proposal_id
        == b.source.proposal_records[0].proposal_id
    )
    with pytest.raises(ValueError, match="history"):
        _restore(replace(a, source=b.source))
    first.close()
    second.close()


def test_rehashed_directive_cannot_rebind_its_decision_link():
    source, _ = _moving()
    state = capture_authoritative_control(source.core)
    old = next(f for f in state.facts if f.event_type == "execution_directive")
    wrong = ControlTraceFact.create(
        **{
            name: getattr(old, name)
            for name in (
                "fact_id",
                "event_type",
                "runtime_session_id",
                "control_domain_id",
                "embodiment_instance_id",
                "occurred_at",
                "clock_domain_id",
            )
        },
        payload={**old.payload, "decision_id": "another-decision"},
    )
    changed = tuple(wrong if f is old else f for f in state.facts)
    with pytest.raises(ValueError, match="lineage"):
        _restore(replace(state, facts=changed))
    source.close()


def test_store_configuration_and_terminal_or_hold_state_cannot_be_changed():
    source, _ = _moving()
    state = capture_authoritative_control(source.core)
    for settings in (
        (2048, -1, 1024, -1, "ready"),
        (1024, -1, 2048, -1, "ready"),
        (1024, -1, 1024, -1, "fault"),
    ):
        with pytest.raises(ValueError, match="settings"):
            _restore(replace(state, store_settings=settings))
    with pytest.raises(ValueError, match="history"):
        _restore(replace(state, state={**state.state, "_backend_hold_active": True}))
    source.close()
    with pytest.raises(ValueError, match="Terminal"):
        _restore(replace(state, facts=source.core.trace_facts))


def test_restored_active_core_accepts_next_proposal_and_advances_generation():
    source, clock = _moving()
    state = capture_authoritative_control(source.core)
    branch, rig, branch_clock = _restore(state)
    try:
        clock.value = branch_clock.value = 10.07
        receipt = source.submit_policy_action(action(0.1), policy_output_id="next")
        old = receipt.proposal
        draft = ActionProposalDraft(
            **{
                field.name: getattr(old, field.name)
                for field in fields(ActionProposalDraft)
            }
        )
        assert branch.admit_proposal(draft, source_event_id="next").accepted
        a, b = source.core.run_decision(), branch.run_decision()
        assert a.status == b.status == "dispatched"
        assert a.decision.decision_id == b.decision.decision_id
        assert a.directive.semantic_content_hash == b.directive.semantic_content_hash
        assert branch._active_segment_by_group == source.core._active_segment_by_group
        assert branch._active_segment_by_group["arms"].group_segment_generation == 2
        assert (
            branch._previous_selection_by_group
            == source.core._previous_selection_by_group
        )
    finally:
        rig.close()
        source.close()


def test_nonempty_authority_and_pruned_replay_authorization_survive_roundtrip():
    from robot_motion.action.checkpoint_codec import (
        decode_control_checkpoint,
        encode_control_checkpoint,
    )
    from sim.mujoco.deploy import MujocoDeployRuntime
    from tests.test_mujoco_deploy_runtime import replay_source

    base, _, clock = build_runtime()
    source = MujocoDeployRuntime(
        base.rig, replace(base.config, enable_replay_authority=True), clock=clock
    )
    source.start()
    submission = source.submit_replay_action(
        replay_source(source),
        action(0.1),
        source_event_id="replay:frame:0",
        source_path="/test-readonly-source",
        source_episode_index=0,
        group_key="replay",
        local_frame_index=0,
        source_action_index=0,
        source_timestamp=0.0,
    )
    assert submission.authorization_admission.accepted
    assert source.step(now=clock.value).status == "dispatched"
    source.transition_execution_gate("hold", reason="test:handoff")
    source.core.admit_authority_event(
        event_type="mode", control_mode="agent_skill", reason="test:mode"
    )
    source.transition_execution_gate("ready", reason="test:ready")
    clock.value = 11.0
    source.core.proposals.prune_expired(at_or_before=clock.value)
    assert len(source.core.proposals) == 0
    saved = capture_authoritative_control(source.core)
    blob = encode_control_checkpoint(saved)
    parsed = decode_control_checkpoint(blob, max_bytes=8 * 1024 * 1024)
    branch, rig, _ = _restore(parsed, config=source.core.config)
    try:
        assert branch.dispatch_owners.snapshot().current_authority_epoch == 1
        assert branch.dispatch_owner_token.authority_epoch_at_acquire == 0
        assert branch._backend_hold_active
        assert len(branch.authority_store._records) == 1
        assert len(branch._candidate_authorizations._records) == 1
        assert len(branch.proposals) == 0
        assert encode_control_checkpoint(capture_authoritative_control(branch)) == blob
    finally:
        rig.close()
        source.close()
