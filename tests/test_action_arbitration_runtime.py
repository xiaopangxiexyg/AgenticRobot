from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    RobotAction,
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.arbitration_runtime import (
    MAX_PRODUCTION_ARBITER_HZ,
    ArbitrationRuntime,
    DecisionScheduleSpec,
    DecisionScheduler,
)


class Clock:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self):
        return self.value


def computation(
    *,
    result_id="result-0",
    deadline=100.0,
    target=0.0,
    gate_epoch=0,
    schedule_hash=None,
):
    action = project_robot_action_v1_to_component_action(
        RobotAction(
            action_space=BIMANUAL_QPOS_V1,
            arm_qpos_by_side={"left": np.zeros(6, dtype=np.float32)},
            source="test",
        )
    )
    return SimpleNamespace(
        result_id=result_id,
        output_status="selected",
        gate_epoch=gate_epoch,
        publication_deadline_at=deadline,
        target_effective_at=target,
        supported_until=deadline,
        arbitration_input_snapshot_id="input-0",
        selected_action_hash=action.semantic_content_hash,
        selected_by_group={"arms": "proposal-0"},
        composed_candidate=action,
        authority_epoch=0,
        decision_schedule_spec_hash=schedule_hash,
        target_clock_domain_id="control.monotonic",
        semantic_content_hash=f"{result_id}-hash",
    )


def test_production_decision_rate_accepts_60_and_rejects_above_60() -> None:
    assert DecisionScheduleSpec(spec_id="60hz", decision_hz=60).decision_hz == 60
    with pytest.raises(ValueError, match="60"):
        DecisionScheduleSpec(spec_id="too-fast", decision_hz=60.0001)
    assert MAX_PRODUCTION_ARBITER_HZ == 60.0


def test_event_burst_coalesces_without_catch_up_ticks() -> None:
    clock = Clock(1.0)
    scheduler = DecisionScheduler(
        DecisionScheduleSpec(
            spec_id="event-60", schedule_mode="event_limited", decision_hz=60
        ),
        clock=clock,
    )
    for index in range(100):
        scheduler.trigger(f"proposal-{index}", "proposal")

    first = scheduler.make_tick(cutoff_admission_sequence=100)
    assert first is not None
    assert first.coalesced_trigger_count == 99
    assert first.trigger_ids == ("proposal-99",)
    assert first.coalesced_trigger_ids[0] == "proposal-0"
    assert first.coalesced_trigger_ids[-1] == "proposal-98"
    assert scheduler.make_tick(cutoff_admission_sequence=100) is None

    clock.value = 10.0  # many nominal periods were missed
    scheduler.trigger("latest", "proposal")
    second = scheduler.make_tick(cutoff_admission_sequence=101)
    assert second is not None
    assert scheduler.make_tick(cutoff_admission_sequence=101) is None


def test_publication_gate_limits_only_authoritative_selected_decisions() -> None:
    spec = DecisionScheduleSpec(
        spec_id="60hz", decision_hz=60, max_target_lateness_sec=100
    )
    runtime = ArbitrationRuntime(spec)
    gate = SimpleNamespace(status="ready", gate_epoch=0)

    kwargs = {"schedule_hash": spec.semantic_content_hash}
    first_attempt, first = runtime.publish(computation(**kwargs), attempted_at=1.0, live_gate_snapshot=gate)
    early_attempt, early = runtime.publish(
        computation(result_id="result-1", **kwargs), attempted_at=1.005, live_gate_snapshot=gate
    )
    on_time_attempt, on_time = runtime.publish(
        computation(result_id="result-2", **kwargs),
        attempted_at=1.0 + 1.0 / 60.0,
        live_gate_snapshot=gate,
    )

    assert first_attempt.accepted and first is not None
    assert not early_attempt.accepted and early is None
    assert early_attempt.rejection_reason == "coalesced_before_publication"
    assert on_time_attempt.accepted and on_time is not None
    assert not hasattr(spec, "reference_hz")
    assert not hasattr(spec, "ik_hz")
    assert not hasattr(spec, "command_hz")


def test_live_gate_change_and_late_result_reject_publication() -> None:
    spec = DecisionScheduleSpec(
        spec_id="30hz",
        decision_hz=30,
        max_target_lateness_sec=0.01,
        late_decision_policy="hold",
    )
    changed_gate = SimpleNamespace(status="ready", gate_epoch=2)
    runtime = ArbitrationRuntime(spec)

    changed_attempt, changed = runtime.publish(
        computation(gate_epoch=1, schedule_hash=spec.semantic_content_hash),
        attempted_at=1.0,
        live_gate_snapshot=changed_gate,
    )
    late_attempt, late = runtime.publish(
        computation(
            result_id="late",
            target=0.0,
            deadline=2.0,
            gate_epoch=2,
            schedule_hash=spec.semantic_content_hash,
        ),
        attempted_at=1.0,
        live_gate_snapshot=changed_gate,
    )

    assert changed is None
    assert changed_attempt.rejection_reason == "execution_gate_epoch_changed"
    assert late is None
    assert late_attempt.rejection_reason == "target_late:hold"
    assert late_attempt.required_fallback == "hold"


def test_synthetic_computation_budget_is_recorded_outside_pure_result_hash() -> None:
    spec = DecisionScheduleSpec(
        spec_id="budget", decision_hz=30, decision_runtime_budget_sec=0.005
    )
    runtime = ArbitrationRuntime(spec)
    result = SimpleNamespace(result_id="result-0", publication_deadline_at=2.0)
    arbiter = SimpleNamespace(arbitrate=lambda snapshot: result)
    times = iter((1.0, 1.007))

    returned, metrics = runtime.compute(
        arbiter,
        SimpleNamespace(),
        clock=lambda: next(times),
    )

    assert returned is result
    assert metrics.runtime_sec == pytest.approx(0.007)
    assert metrics.runtime_budget_missed
    assert not metrics.publication_deadline_missed_at_completion
