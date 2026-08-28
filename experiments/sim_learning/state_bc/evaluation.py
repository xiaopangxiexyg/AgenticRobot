"""Matched, resettable episode evaluation for the state-BC HITL experiment.

This module owns comparison bookkeeping only.  Callers retain reset, simulator,
task-predicate, action execution, and resource-lifecycle ownership.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Protocol

from .hitl import (
    HITL_EVENT_RTC_NO_ACTION,
    HITL_SOURCE_HUMAN,
    HITL_SOURCE_POLICY,
    HITL_SOURCE_RTC,
)
from .stream import StateBCStreamTrace

STATE_BC_CONTROL_POLICY = "policy_only"
STATE_BC_CONTROL_TAKEOVER = "takeover"
STATE_BC_CONTROL_RTC = "rtc"
STATE_BC_CONTROL_MODES = frozenset(
    {STATE_BC_CONTROL_POLICY, STATE_BC_CONTROL_TAKEOVER, STATE_BC_CONTROL_RTC}
)

TASK_ASSESSMENT_SUCCESS = "success"
TASK_ASSESSMENT_FAILURE = "failure"
TASK_ASSESSMENT_UNKNOWN = "unknown"
TASK_ASSESSMENT_STATUSES = frozenset(
    {TASK_ASSESSMENT_SUCCESS, TASK_ASSESSMENT_FAILURE, TASK_ASSESSMENT_UNKNOWN}
)


def _identifier(value: Any, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


@dataclass(frozen=True)
class StateBCEpisodeCondition:
    """One exclusive control schedule in a matched three-condition panel."""

    condition_id: str
    control_mode: str
    requested_sources: tuple[str, ...]
    human_actions: Mapping[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        condition_id = _identifier(self.condition_id, name="condition_id")
        mode = str(self.control_mode).strip().lower()
        if mode not in STATE_BC_CONTROL_MODES:
            raise ValueError(f"unsupported state BC control mode {self.control_mode!r}.")
        sources = tuple(str(source).strip().lower() for source in self.requested_sources)
        if not sources:
            raise ValueError("an episode condition requires at least one step.")
        allowed_sources = {HITL_SOURCE_POLICY, HITL_SOURCE_HUMAN, HITL_SOURCE_RTC}
        if any(source not in allowed_sources for source in sources):
            raise ValueError("episode schedule contains an unsupported source.")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in self.human_actions
        ):
            raise TypeError("human action step indices must be integers.")
        humans = dict(self.human_actions)
        expected_human_steps = {
            index for index, source in enumerate(sources) if source == HITL_SOURCE_HUMAN
        }
        if set(humans) != expected_human_steps:
            raise ValueError("human_actions must exactly cover scheduled human steps.")
        source_set = set(sources)
        if mode == STATE_BC_CONTROL_POLICY and source_set != {HITL_SOURCE_POLICY}:
            raise ValueError("policy_only may schedule only policy actions.")
        if mode == STATE_BC_CONTROL_TAKEOVER and not (
            HITL_SOURCE_POLICY in source_set
            and HITL_SOURCE_HUMAN in source_set
            and HITL_SOURCE_RTC not in source_set
        ):
            raise ValueError("takeover must contain policy and human, but not rtc.")
        if mode == STATE_BC_CONTROL_RTC and not (
            HITL_SOURCE_POLICY in source_set
            and HITL_SOURCE_RTC in source_set
            and HITL_SOURCE_HUMAN not in source_set
        ):
            raise ValueError("rtc must contain policy and rtc, but not human.")
        object.__setattr__(self, "condition_id", condition_id)
        object.__setattr__(self, "control_mode", mode)
        object.__setattr__(self, "requested_sources", sources)
        object.__setattr__(self, "human_actions", humans)

    @property
    def horizon(self) -> int:
        return len(self.requested_sources)


@dataclass(frozen=True)
class StateBCResetReceipt:
    """Caller-owned reset evidence used to prove matched initial conditions."""

    seed: int
    reset_id: str
    initial_state_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seed = int(self.seed)
        if seed < 0:
            raise ValueError("reset seed must be non-negative.")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "reset_id", _identifier(self.reset_id, name="reset_id"))
        object.__setattr__(
            self,
            "initial_state_id",
            _identifier(self.initial_state_id, name="initial_state_id"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class StateBCTaskAssessment:
    """Fail-closed task predicate result; unknown is never counted as failure."""

    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in TASK_ASSESSMENT_STATUSES:
            raise ValueError(f"unsupported task assessment status {self.status!r}.")
        metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            name = _identifier(key, name="metric name")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"task metric {name!r} must be numeric.")
            number = float(value)
            if number != number or number in {float("inf"), float("-inf")}:
                raise ValueError(f"task metric {name!r} must be finite.")
            metrics[name] = number
        reason = None if self.reason is None else str(self.reason).strip()
        if status == TASK_ASSESSMENT_UNKNOWN and not reason:
            raise ValueError("unknown task assessment requires a reason.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "reason", reason or None)


@dataclass(frozen=True)
class StateBCEpisodeRecord:
    seed: int
    condition: StateBCEpisodeCondition
    reset: StateBCResetReceipt
    trace: StateBCStreamTrace
    assessment: StateBCTaskAssessment

    def __post_init__(self) -> None:
        if int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("episode seed must be a non-negative integer.")
        if not isinstance(self.condition, StateBCEpisodeCondition):
            raise TypeError("condition must be a StateBCEpisodeCondition.")
        if not isinstance(self.reset, StateBCResetReceipt) or self.reset.seed != self.seed:
            raise ValueError("reset receipt must match the episode seed.")
        if not isinstance(self.trace, StateBCStreamTrace):
            raise TypeError("trace must be a StateBCStreamTrace.")
        if not isinstance(self.assessment, StateBCTaskAssessment):
            raise TypeError("assessment must be a StateBCTaskAssessment.")


@dataclass(frozen=True)
class StateBCEvaluationReport:
    records: tuple[StateBCEpisodeRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records:
            raise ValueError("evaluation report requires at least one episode.")
        keys = {(record.seed, record.condition.control_mode) for record in records}
        if len(keys) != len(records):
            raise ValueError("evaluation contains a duplicate seed/control_mode episode.")
        modes_by_seed: dict[int, set[str]] = {}
        initial_by_seed: dict[int, set[str]] = {}
        for record in records:
            modes_by_seed.setdefault(record.seed, set()).add(record.condition.control_mode)
            initial_by_seed.setdefault(record.seed, set()).add(record.reset.initial_state_id)
        if any(modes != STATE_BC_CONTROL_MODES for modes in modes_by_seed.values()):
            raise ValueError("every seed must contain all three control modes.")
        if any(len(identities) != 1 for identities in initial_by_seed.values()):
            raise ValueError("control modes do not share a matched initial_state_id.")
        if len({record.condition.horizon for record in records}) != 1:
            raise ValueError("evaluation records do not share one matched horizon.")
        object.__setattr__(self, "records", records)

    def summary(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for mode in sorted(STATE_BC_CONTROL_MODES):
            selected = [record for record in self.records if record.condition.control_mode == mode]
            counts = {status: 0 for status in sorted(TASK_ASSESSMENT_STATUSES)}
            source_actions = {
                source: 0
                for source in (HITL_SOURCE_POLICY, HITL_SOURCE_HUMAN, HITL_SOURCE_RTC)
            }
            rtc_no_action = 0
            for record in selected:
                counts[record.assessment.status] += 1
                for source in record.trace.source_sequence():
                    source_actions[source] += 1
                rtc_no_action += sum(
                    event.event_kind == HITL_EVENT_RTC_NO_ACTION
                    for event in record.trace.events
                )
            known = counts[TASK_ASSESSMENT_UNKNOWN] == 0
            result[mode] = {
                "episodes": len(selected),
                "assessment_counts": counts,
                "success_rate": (
                    counts[TASK_ASSESSMENT_SUCCESS] / len(selected) if known else None
                ),
                "action_events": sum(source_actions.values()),
                "source_actions": source_actions,
                "rtc_no_action": rtc_no_action,
            }
        return result


class StateBCEpisodeRunner(Protocol):
    def run(
        self,
        requested_sources: Sequence[str],
        *,
        human_actions: Mapping[int, Any] | None = None,
    ) -> StateBCStreamTrace: ...


ResetFn = Callable[[int, StateBCEpisodeCondition], StateBCResetReceipt]
RunnerFactory = Callable[
    [int, StateBCEpisodeCondition, StateBCResetReceipt], StateBCEpisodeRunner
]
AssessFn = Callable[
    [int, StateBCEpisodeCondition, StateBCResetReceipt, StateBCStreamTrace],
    StateBCTaskAssessment,
]


def run_matched_state_bc_evaluation(
    *,
    seeds: Sequence[int],
    conditions: Sequence[StateBCEpisodeCondition],
    reset: ResetFn,
    runner_factory: RunnerFactory,
    assess: AssessFn,
) -> StateBCEvaluationReport:
    """Run all three conditions from a caller-proven matched reset per seed."""

    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("seeds must contain integers.")
    resolved_seeds = tuple(seeds)
    if not resolved_seeds or len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("seeds must be a non-empty unique sequence.")
    if any(seed < 0 for seed in resolved_seeds):
        raise ValueError("seeds must be non-negative.")
    resolved_conditions = tuple(conditions)
    modes = {condition.control_mode for condition in resolved_conditions}
    if len(resolved_conditions) != 3 or modes != STATE_BC_CONTROL_MODES:
        raise ValueError("conditions must contain exactly policy_only, takeover, and rtc.")
    if len({condition.condition_id for condition in resolved_conditions}) != 3:
        raise ValueError("condition_id values must be unique.")
    horizons = {condition.horizon for condition in resolved_conditions}
    if len(horizons) != 1:
        raise ValueError("matched conditions must use the same step horizon.")
    if not callable(reset) or not callable(runner_factory) or not callable(assess):
        raise TypeError("reset, runner_factory, and assess must be callable.")

    records: list[StateBCEpisodeRecord] = []
    for seed in resolved_seeds:
        matched_initial_state_id: str | None = None
        for condition in resolved_conditions:
            reset_receipt = reset(seed, condition)
            if not isinstance(reset_receipt, StateBCResetReceipt):
                raise TypeError("reset callback must return StateBCResetReceipt.")
            if reset_receipt.seed != seed:
                raise ValueError("reset callback returned the wrong seed.")
            if matched_initial_state_id is None:
                matched_initial_state_id = reset_receipt.initial_state_id
            elif reset_receipt.initial_state_id != matched_initial_state_id:
                raise ValueError("control modes do not share a matched initial_state_id.")
            runner = runner_factory(seed, condition, reset_receipt)
            trace = runner.run(
                condition.requested_sources,
                human_actions=condition.human_actions,
            )
            if not isinstance(trace, StateBCStreamTrace):
                raise TypeError("episode runner must return StateBCStreamTrace.")
            observed_steps: set[int] = set()
            for event in trace.events:
                if event.step_index >= condition.horizon:
                    raise ValueError("stream trace contains an out-of-horizon event.")
                if event.requested_source != condition.requested_sources[event.step_index]:
                    raise ValueError("stream trace does not match the requested schedule.")
                observed_steps.add(event.step_index)
            if observed_steps != set(range(condition.horizon)):
                raise ValueError("stream trace does not cover the requested horizon.")
            assessment = assess(seed, condition, reset_receipt, trace)
            records.append(
                StateBCEpisodeRecord(
                    seed=seed,
                    condition=condition,
                    reset=reset_receipt,
                    trace=trace,
                    assessment=assessment,
                )
            )
    return StateBCEvaluationReport(records=tuple(records))


__all__ = [
    "STATE_BC_CONTROL_MODES",
    "STATE_BC_CONTROL_POLICY",
    "STATE_BC_CONTROL_RTC",
    "STATE_BC_CONTROL_TAKEOVER",
    "TASK_ASSESSMENT_FAILURE",
    "TASK_ASSESSMENT_STATUSES",
    "TASK_ASSESSMENT_SUCCESS",
    "TASK_ASSESSMENT_UNKNOWN",
    "StateBCEpisodeCondition",
    "StateBCEpisodeRecord",
    "StateBCEpisodeRunner",
    "StateBCEvaluationReport",
    "StateBCResetReceipt",
    "StateBCTaskAssessment",
    "run_matched_state_bc_evaluation",
]
