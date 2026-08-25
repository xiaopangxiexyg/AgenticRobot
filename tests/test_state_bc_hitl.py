from __future__ import annotations

import numpy as np

from experiments.sim_learning.state_bc.hitl import (
    HITL_EVENT_ACTIVATION,
    HITL_EVENT_HOLD,
    HITL_EVENT_RTC_NO_ACTION,
    StateBCHITLRunner,
)


def test_hitl_switching_is_exclusive_and_rtc_no_action_is_not_a_sample():
    states = [np.full(16, float(index), dtype=np.float32) for index in range(6)]
    sources = ["policy", "policy", "human", "policy", "rtc", "rtc"]
    human_action = np.full(16, 9.0, dtype=np.float32)

    def policy(state):
        return np.asarray(state, dtype=np.float32) + 0.25

    def rtc_policy(state, nominal):
        if float(state[0]) == 4.0:
            return None
        return np.asarray(nominal, dtype=np.float32) + 0.5

    trace = StateBCHITLRunner(policy, rtc_policy=rtc_policy).run(
        states,
        sources,
        human_actions={2: human_action},
    )

    assert any(event.event_kind == HITL_EVENT_HOLD for event in trace.events)
    assert sum(event.event_kind == HITL_EVENT_ACTIVATION for event in trace.events) == 4
    assert sum(event.event_kind == HITL_EVENT_RTC_NO_ACTION for event in trace.events) == 1
    assert trace.source_sequence() == ("policy", "policy", "human", "policy", "rtc")
    assert len(trace.samples) == 5
    assert all(event.executed_source is not None for event in trace.action_events)
    assert all(event.action is not None for event in trace.action_events)
    assert not any(sample["source"] == "rtc" and sample["step_index"] == 4 for sample in trace.samples)


def test_hitl_runner_accepts_numpy_batch_sequences():
    states = np.stack(
        [np.full(16, float(index), dtype=np.float32) for index in range(3)],
        axis=0,
    )
    trace = StateBCHITLRunner(
        lambda state: np.asarray(state, dtype=np.float32),
    ).run(states, ["policy", "policy", "policy"])

    assert trace.source_sequence() == ("policy", "policy", "policy")
    assert len(trace.samples) == 3
