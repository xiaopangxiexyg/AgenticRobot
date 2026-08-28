from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from experiments.sim_learning.state_bc.evaluation import (
    STATE_BC_CONTROL_POLICY,
    STATE_BC_CONTROL_RTC,
    STATE_BC_CONTROL_TAKEOVER,
    TASK_ASSESSMENT_FAILURE,
    TASK_ASSESSMENT_SUCCESS,
    TASK_ASSESSMENT_UNKNOWN,
    StateBCEpisodeCondition,
    StateBCResetReceipt,
    StateBCTaskAssessment,
    run_matched_state_bc_evaluation,
)
from experiments.sim_learning.state_bc.stream import (
    StateBCStreamEvent,
    StateBCStreamTrace,
)
from robot_motion.deploy import NormalizedAction


def _runtime_action(source: str) -> NormalizedAction:
    return NormalizedAction(
        vector=np.zeros(26, dtype=np.float32),
        metadata={"source": source},
    )


def _conditions() -> tuple[StateBCEpisodeCondition, ...]:
    human_action = _runtime_action("fixture_human")
    return (
        StateBCEpisodeCondition(
            condition_id="policy",
            control_mode=STATE_BC_CONTROL_POLICY,
            requested_sources=("policy", "policy", "policy", "policy"),
        ),
        StateBCEpisodeCondition(
            condition_id="takeover",
            control_mode=STATE_BC_CONTROL_TAKEOVER,
            requested_sources=("policy", "human", "human", "policy"),
            human_actions={1: human_action, 2: human_action},
        ),
        StateBCEpisodeCondition(
            condition_id="rtc",
            control_mode=STATE_BC_CONTROL_RTC,
            requested_sources=("policy", "rtc", "rtc", "policy"),
        ),
    )


@dataclass
class _TraceRunner:
    seed: int
    condition: StateBCEpisodeCondition
    calls: list[tuple[int, str, tuple[str, ...]]]

    def run(self, requested_sources, *, human_actions=None):
        sources = tuple(requested_sources)
        assert sources == self.condition.requested_sources
        assert dict(human_actions or {}) == dict(self.condition.human_actions)
        self.calls.append((self.seed, self.condition.control_mode, sources))
        events = []
        for index, source in enumerate(sources):
            if self.condition.control_mode == STATE_BC_CONTROL_RTC and index == 1:
                events.append(
                    StateBCStreamEvent(
                        step_index=index,
                        event_kind="rtc_no_action",
                        requested_source=source,
                        activation_id="fixture:rtc",
                    )
                )
            else:
                events.append(
                    StateBCStreamEvent(
                        step_index=index,
                        event_kind="action",
                        requested_source=source,
                        executed_source=source,
                        activation_id=f"fixture:{source}",
                        runtime_action=_runtime_action(source),
                    )
                )
        return StateBCStreamTrace(events=tuple(events))


def test_matched_evaluation_runs_three_modes_per_seed_and_fails_unknown_closed():
    calls: list[tuple[int, str, tuple[str, ...]]] = []
    resets: list[tuple[int, str]] = []

    def reset(seed, condition):
        resets.append((seed, condition.control_mode))
        return StateBCResetReceipt(
            seed=seed,
            reset_id=f"reset:{seed}:{condition.control_mode}",
            initial_state_id=f"initial:{seed}",
        )

    def runner_factory(seed, condition, _reset):
        return _TraceRunner(seed=seed, condition=condition, calls=calls)

    def assess(_seed, condition, _reset, _trace):
        if condition.control_mode == STATE_BC_CONTROL_POLICY:
            return StateBCTaskAssessment(
                status=TASK_ASSESSMENT_SUCCESS,
                metrics={"cube_height_m": 0.10},
            )
        if condition.control_mode == STATE_BC_CONTROL_TAKEOVER:
            return StateBCTaskAssessment(status=TASK_ASSESSMENT_FAILURE)
        return StateBCTaskAssessment(
            status=TASK_ASSESSMENT_UNKNOWN,
            reason="task predicate unavailable in fixture",
        )

    report = run_matched_state_bc_evaluation(
        seeds=(3, 7),
        conditions=_conditions(),
        reset=reset,
        runner_factory=runner_factory,
        assess=assess,
    )

    assert len(report.records) == 6
    assert len(resets) == 6
    assert len(calls) == 6
    summary = report.summary()
    assert summary[STATE_BC_CONTROL_POLICY]["success_rate"] == 1.0
    assert summary[STATE_BC_CONTROL_TAKEOVER]["success_rate"] == 0.0
    assert summary[STATE_BC_CONTROL_RTC]["success_rate"] is None
    assert summary[STATE_BC_CONTROL_RTC]["rtc_no_action"] == 2
    assert summary[STATE_BC_CONTROL_RTC]["source_actions"] == {
        "policy": 4,
        "human": 0,
        "rtc": 2,
    }


def test_reset_identity_mismatch_stops_before_unmatched_episode_execution():
    calls = []

    def reset(seed, condition):
        return StateBCResetReceipt(
            seed=seed,
            reset_id=f"reset:{condition.control_mode}",
            initial_state_id=f"initial:{condition.control_mode}",
        )

    with pytest.raises(ValueError, match="matched initial_state_id"):
        run_matched_state_bc_evaluation(
            seeds=(0,),
            conditions=_conditions(),
            reset=reset,
            runner_factory=lambda seed, condition, _reset: _TraceRunner(
                seed=seed,
                condition=condition,
                calls=calls,
            ),
            assess=lambda *_args: StateBCTaskAssessment(
                status=TASK_ASSESSMENT_SUCCESS
            ),
        )

    assert len(calls) == 1


def test_condition_panel_rejects_missing_human_action_and_mismatched_horizon():
    with pytest.raises(ValueError, match="exactly cover"):
        StateBCEpisodeCondition(
            condition_id="bad-takeover",
            control_mode=STATE_BC_CONTROL_TAKEOVER,
            requested_sources=("policy", "human", "policy"),
        )

    conditions = list(_conditions())
    conditions[0] = StateBCEpisodeCondition(
        condition_id="short-policy",
        control_mode=STATE_BC_CONTROL_POLICY,
        requested_sources=("policy",),
    )
    with pytest.raises(ValueError, match="same step horizon"):
        run_matched_state_bc_evaluation(
            seeds=(0,),
            conditions=conditions,
            reset=lambda *_args: pytest.fail("reset must not run"),
            runner_factory=lambda *_args: pytest.fail("runner must not run"),
            assess=lambda *_args: pytest.fail("assess must not run"),
        )


def test_unknown_assessment_requires_an_explicit_reason():
    with pytest.raises(ValueError, match="requires a reason"):
        StateBCTaskAssessment(status=TASK_ASSESSMENT_UNKNOWN)


def test_evaluation_rejects_non_integer_seed_before_reset():
    with pytest.raises(TypeError, match="integers"):
        run_matched_state_bc_evaluation(
            seeds=(1.5,),
            conditions=_conditions(),
            reset=lambda *_args: pytest.fail("reset must not run"),
            runner_factory=lambda *_args: pytest.fail("runner must not run"),
            assess=lambda *_args: pytest.fail("assess must not run"),
        )
