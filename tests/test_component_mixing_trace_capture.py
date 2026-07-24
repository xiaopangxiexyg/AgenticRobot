from __future__ import annotations

from pathlib import Path

import pytest

from control_composition.composition import (
    DiscreteOwnerPoseMixingStrategy,
    LinearHandQposInterpolationStrategy,
    evaluate_component_mixing_shadow,
)
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    HandActionSpec,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from control_composition.trace import project_component_mixing_shadow
from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    ControlTraceSpool,
    ControlTraceSpoolSink,
    ControlTraceWriterHealthLedger,
    EpisodeTraceRecorderSink,
    RuntimeTraceSession,
)

from tests.test_pose_mixing_strategy import (
    _compound_arbitration_input_with_left_hand,
)


def _shadow_decision(*, missing_hand_slot: bool):
    arbitration_input = _compound_arbitration_input_with_left_hand()
    hand_spec = HandActionSpec.dmtac_gripper("left")
    weights = (
        {"human-slot": 0.5, "missing-slot": 0.5}
        if missing_hand_slot
        else {"human-slot": 0.25, "replay-slot": 0.75}
    )
    return evaluate_component_mixing_shadow(
        arbitration_input,
        source_slot_by_producer={
            "human-main": "human-slot",
            "replay-main": "replay-slot",
        },
        clock_domain_id="control.monotonic",
        arm_strategy=DiscreteOwnerPoseMixingStrategy(),
        hand_strategy_by_side={
            "left": LinearHandQposInterpolationStrategy(
                hand_spec,
                action_space=BIMANUAL_EE_POSE_V1,
                weights_by_source_slot=weights,
            )
        },
        hand_spec_by_side={"left": hand_spec},
    )


def _terminal(*, status: str, occurred_at: float) -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id=f"runtime-terminal:{status}:{occurred_at}",
        event_type="runtime_trace_terminal",
        runtime_session_id="component-shadow-runtime",
        control_domain_id="robot-0",
        occurred_at=occurred_at,
        clock_domain_id="control.monotonic",
        payload={
            "terminal_status": status,
            "reason": "component_shadow_fixture",
            "occurred_at": occurred_at,
        },
    )


def _context() -> AuthoritativeTraceContext:
    return AuthoritativeTraceContext(
        runtime_session_id="component-shadow-runtime",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        clock_domain_id="control.monotonic",
    )


def test_component_shadow_projection_is_explicit_and_non_authoritative() -> None:
    decision = _shadow_decision(missing_hand_slot=False)
    fact = project_component_mixing_shadow(
        _context(),
        decision,
        evaluated_at=10.5,
    )

    assert fact.event_type == "component_mixing_shadow"
    assert fact.payload["non_authoritative"] is True
    assert fact.payload["status"] == "selected"
    assert fact.payload["output_action"]["hand_qpos_by_side"]["left"][0] == pytest.approx(
        0.65
    )
    assert fact.payload["arm_input_snapshot_id"] == decision.arm_decision.input_snapshot_id
    assert fact.payload["hand_decision_id_by_side"]["left"] == (
        decision.hand_decisions_by_side["left"].decision_id
    )
    assert fact.payload["semantic_content_hash"] == decision.semantic_content_hash


def test_component_shadow_formal_fixture_reuses_one_episode_owner_across_hold_abort_restart(
    tmp_path: Path,
) -> None:
    runtime = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="component-shadow-trace",
            runtime_session_id="component-shadow-runtime",
            control_domain_id="robot-0",
            profile_id="deployable_sim_component_mixing_shadow_v1",
            started_at=0.0,
            clock_domain_id="control.monotonic",
        ),
        capacity=256,
    )
    spool = ControlTraceSpool(
        tmp_path / "control_trace.wal",
        trace_session_id=runtime.session.trace_session_id,
        max_bytes=1 << 20,
        physical_preallocation=False,
    )
    health = ControlTraceWriterHealthLedger(
        tmp_path / "writer_health.ledger",
        trace_session_id=runtime.session.trace_session_id,
    )
    durable_sink = ControlTraceSpoolSink(
        sink_id="component-shadow-sink",
        runtime=runtime,
        spool=spool,
        health_ledger=health,
    )
    uuids = iter(
        ("episode-aborted", "episode-restarted", "episode-restarted-next")
    )
    recorder = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: next(uuids),
        control_clock=lambda: 1.0,
        commit_clock=lambda: 1.0,
    )

    try:
        prepared = recorder.prepare_episode(legacy_episode_index=0, prepared_at=1.0)
        recorder.cancel_prepared_episode(prepared)
        restarted = recorder.prepare_episode(legacy_episode_index=1, prepared_at=2.0)
        recorder.bind_prepared_episode(restarted, bound_at=2.0)

        hold_decision = _shadow_decision(missing_hand_slot=True)
        hold_fact = project_component_mixing_shadow(
            _context(),
            hold_decision,
            evaluated_at=2.5,
        )
        assert hold_decision.status == "no_action"
        assert runtime.issue(hold_fact).accepted

        released = recorder.release_episode(
            restarted,
            unbound_at=3.0,
            reason="operator_abort_during_hold",
        )
        assert released.seal.trace_sealed
        assert "component_mixing_shadow" in released.seal.observed_event_types
        assert released.seal.capture_commit_eligible is False

        second = recorder.prepare_episode(legacy_episode_index=2, prepared_at=4.0)
        recorder.bind_prepared_episode(second, bound_at=4.0)
        selected_decision = _shadow_decision(missing_hand_slot=False)
        selected_fact = project_component_mixing_shadow(
            _context(),
            selected_decision,
            evaluated_at=4.5,
        )
        assert runtime.issue(selected_fact).accepted
        released_again = recorder.release_episode(
            second,
            unbound_at=5.0,
            reason="restart_after_abort",
        )
        assert released_again.seal.trace_sealed

        assert runtime.issue(_terminal(status="aborted", occurred_at=6.0)).accepted
        assert durable_sink.flush_pending().failure_code is None
        report = runtime.completeness()
        assert report.in_memory_complete
        assert report.durable_complete
        assert report.lineage_complete
        assert report.capture_commit_eligible is False
        assert len(
            [
                item
                for item in runtime.records()
                if item.fact.event_type == "component_mixing_shadow"
            ]
        ) == 2
        memberships = [
            item.episode_membership.episode_uuid
            for item in runtime.journal_entries()
            if item.episode_membership is not None
        ]
        assert "episode-restarted" in memberships
        assert "episode-aborted" not in memberships
        assert "episode-restarted-next" in memberships
    finally:
        health.close()
        spool.close()


def test_component_shadow_writer_failure_never_becomes_durable_commit(
    tmp_path: Path,
) -> None:
    runtime = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="component-shadow-writer-failure",
            runtime_session_id="component-shadow-runtime",
            control_domain_id="robot-0",
            profile_id="deployable_sim_component_mixing_shadow_v1",
            started_at=0.0,
            clock_domain_id="control.monotonic",
        ),
        capacity=64,
    )
    spool = ControlTraceSpool(
        tmp_path / "bounded_trace.wal",
        trace_session_id=runtime.session.trace_session_id,
        max_bytes=4096,
        physical_preallocation=False,
    )
    health = ControlTraceWriterHealthLedger(
        tmp_path / "bounded_writer_health.ledger",
        trace_session_id=runtime.session.trace_session_id,
    )
    durable_sink = ControlTraceSpoolSink(
        sink_id="bounded-component-shadow-sink",
        runtime=runtime,
        spool=spool,
        health_ledger=health,
    )
    recorder = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: "writer-failure-episode",
    )

    try:
        prepared = recorder.prepare_episode(legacy_episode_index=0, prepared_at=1.0)
        recorder.bind_prepared_episode(prepared, bound_at=1.0)
        decision = _shadow_decision(missing_hand_slot=False)
        assert runtime.issue(
            project_component_mixing_shadow(_context(), decision, evaluated_at=1.5)
        ).accepted
        released = recorder.release_episode(
            prepared,
            unbound_at=2.0,
            reason="writer_failure_fixture",
        )
        assert released.seal.trace_sealed is False
        assert "episode_durability_gap" in released.seal.reason_codes
        assert runtime.issue(_terminal(status="aborted", occurred_at=3.0)).accepted
        report = durable_sink.flush_pending()
        assert report.failure_code == "spool_capacity_exhausted"
        assert runtime.completeness().durable_complete is False
        assert runtime.completeness().capture_commit_eligible is False
    finally:
        health.close()
        spool.close()
