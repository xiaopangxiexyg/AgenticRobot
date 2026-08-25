from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.sim_learning.state_bc.stream import (
    STREAM_RUNTIME_ACTION_DIM,
    StateBCStreamingRunner,
)
from robot_motion.deploy import DeploymentObservation, NormalizedAction
from sim.mujoco.deploy.state_policy_adapter import (
    STATE_BC_LAYOUT_HASH_V1,
    StatePolicyAdapter,
    StatePolicyAdapterConfig,
)


def _observation() -> DeploymentObservation:
    state = np.full(38, np.nan, dtype=np.float32)
    state[6:13] = np.asarray(
        [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    state[25:32] = np.asarray(
        [0.40, 0.50, 0.60, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    state[13:19] = np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float32)
    state[32:38] = np.asarray([0.4, 0, 0, 0, 0, 0], dtype=np.float32)
    return DeploymentObservation(
        timestamp=1.5,
        step_index=0,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        state=state,
        state_valid=np.isfinite(state),
        arm_states={"left": {}, "right": {}},
        hand_states={
            "left": {"gripper_measured_scalar": 0.2},
            "right": {"gripper_measured_scalar": 0.4},
        },
        last_action=np.full(26, np.nan, dtype=np.float32),
    )


class _SequenceObservationSource:
    def __init__(self, observation: DeploymentObservation) -> None:
        self.observation = observation
        self.count = 0

    def capture(self) -> DeploymentObservation:
        result = replace(self.observation, step_index=self.count)
        self.count += 1
        return result


class _DummyStatePolicy:
    metadata = {"layout_hash": STATE_BC_LAYOUT_HASH_V1}
    artifact_hash = "sha256:stream-fixture"

    def predict(self, state):
        value = np.asarray(state, dtype=np.float32).copy()
        value[14] = 0.3
        value[15] = 0.7
        return value


def test_stream_runner_uses_26d_runtime_actions_and_exclusive_sources():
    adapter = StatePolicyAdapter(
        model=_DummyStatePolicy(),
        config=StatePolicyAdapterConfig(policy_absolute_frame="runtime_raw"),
    )
    source = _SequenceObservationSource(_observation())
    transitions: list[tuple[str, str]] = []
    submitted: list[tuple[str, int, np.ndarray]] = []

    def transition(event, _observation):
        transitions.append((event.event_kind, event.requested_source))
        return {"accepted": True, "event": event.event_kind}

    def execute(action, event, _observation):
        submitted.append(
            (event.executed_source or "", event.step_index, action.vector.copy())
        )
        return {"accepted": True, "source": event.executed_source}

    rtc_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def rtc_policy(state, nominal_model_action, nominal_runtime_action):
        rtc_calls.append(
            (
                np.asarray(state).copy(),
                np.asarray(nominal_model_action).copy(),
                nominal_runtime_action.vector.copy(),
            )
        )
        if len(rtc_calls) == 1:
            return None
        return np.asarray(nominal_model_action, dtype=np.float32) + 0.05

    human_action = np.zeros(STREAM_RUNTIME_ACTION_DIM, dtype=np.float32)
    human_action[0:7] = np.asarray(
        [0.11, 0.21, 0.31, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    human_action[7:14] = np.asarray(
        [0.41, 0.51, 0.61, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    human_action[14:20] = 0.55
    human_action[20:26] = 0.65

    trace = StateBCStreamingRunner(
        observation_source=source,
        policy_adapter=adapter,
        rtc_policy=rtc_policy,
        transition=transition,
        execute=execute,
    ).run(
        ["policy", "human", "policy", "rtc", "rtc"],
        human_actions={1: human_action},
    )

    assert trace.source_sequence() == ("policy", "human", "policy", "rtc")
    assert len(trace.action_events) == 4
    assert len(trace.runtime_actions) == 4
    assert len(trace.samples) == 4
    assert all(action.vector.shape == (STREAM_RUNTIME_ACTION_DIM,) for action in trace.runtime_actions)
    assert [source for source, _, _ in submitted] == [
        "policy",
        "human",
        "policy",
        "rtc",
    ]
    assert len(submitted) == 4
    assert transitions == [
        ("activation", "policy"),
        ("hold", "human"),
        ("activation", "human"),
        ("hold", "policy"),
        ("activation", "policy"),
        ("hold", "rtc"),
        ("activation", "rtc"),
    ]
    no_action = next(
        event
        for event in trace.events
        if event.event_kind == "rtc_no_action" and event.step_index == 3
    )
    assert no_action.runtime_action is None
    assert no_action.metadata["runtime_action_submitted"] is False
    assert trace.action_events[1].model_action is None
    assert trace.action_events[-1].model_action is not None
    assert len(rtc_calls) == 2
    assert rtc_calls[0][2].shape == (STREAM_RUNTIME_ACTION_DIM,)


def test_stream_runner_accepts_explicit_normalized_rtc_replacement():
    adapter = StatePolicyAdapter(
        model=_DummyStatePolicy(),
        config=StatePolicyAdapterConfig(policy_absolute_frame="runtime_raw"),
    )
    source = _SequenceObservationSource(_observation())
    replacement = np.zeros(STREAM_RUNTIME_ACTION_DIM, dtype=np.float32)
    replacement[0:7] = np.asarray(
        [0.12, 0.22, 0.32, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    replacement[7:14] = np.asarray(
        [0.42, 0.52, 0.62, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    replacement[14:20] = 0.25
    replacement[20:26] = 0.75

    trace = StateBCStreamingRunner(
        observation_source=source,
        policy_adapter=adapter,
        rtc_policy=lambda _state, _model, _runtime: NormalizedAction(
            vector=replacement,
            metadata={"source": "rtc_fixture"},
        ),
    ).run(["rtc"])

    assert trace.source_sequence() == ("rtc",)
    event = trace.action_events[0]
    assert event.metadata["rtc_output_space"] == "runtime_26d"
    assert event.runtime_action is not None
    np.testing.assert_allclose(event.runtime_action.vector, replacement)
