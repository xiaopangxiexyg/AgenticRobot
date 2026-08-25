"""Offline exclusive-source HITL runner for state-policy validation.

This runner is intentionally below the authoritative MuJoCo target.  It
checks source switching, fresh activation, and causal sample lineage without
pretending to implement safety, IK, or dispatch.  A later integration test can
feed the same policy emissions to ``MujocoDeployRuntime``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import validate_action_vector, validate_state_vector


HITL_SOURCE_POLICY = "policy"
HITL_SOURCE_HUMAN = "human"
HITL_SOURCE_RTC = "rtc"
HITL_SOURCES = frozenset({HITL_SOURCE_POLICY, HITL_SOURCE_HUMAN, HITL_SOURCE_RTC})

HITL_EVENT_HOLD = "hold"
HITL_EVENT_ACTIVATION = "activation"
HITL_EVENT_ACTION = "action"
HITL_EVENT_RTC_NO_ACTION = "rtc_no_action"


def _source(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in HITL_SOURCES:
        raise ValueError(f"unsupported HITL source {value!r}.")
    return normalized


@dataclass(frozen=True)
class StateBCHITLEvent:
    step_index: int
    event_kind: str
    requested_source: str
    executed_source: str | None = None
    activation_id: str | None = None
    inference_mode: str | None = None
    action: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_index = int(self.step_index)
        if step_index < 0:
            raise ValueError("step_index must be non-negative.")
        event_kind = str(self.event_kind).strip().lower()
        if event_kind not in {HITL_EVENT_HOLD, HITL_EVENT_ACTIVATION, HITL_EVENT_ACTION, HITL_EVENT_RTC_NO_ACTION}:
            raise ValueError(f"unsupported HITL event kind {event_kind!r}.")
        requested = _source(self.requested_source)
        executed = None if self.executed_source is None else _source(self.executed_source)
        action = None if self.action is None else validate_action_vector(self.action, name="HITL action")
        if event_kind in {HITL_EVENT_HOLD, HITL_EVENT_ACTIVATION, HITL_EVENT_RTC_NO_ACTION} and action is not None:
            raise ValueError("hold/activation/rtc_no_action events cannot carry an executed action.")
        if event_kind == HITL_EVENT_ACTION and executed is None:
            raise ValueError("action events require one executed source.")
        if event_kind == HITL_EVENT_RTC_NO_ACTION and requested != HITL_SOURCE_RTC:
            raise ValueError("rtc_no_action must be requested by rtc.")
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "event_kind", event_kind)
        object.__setattr__(self, "requested_source", requested)
        object.__setattr__(self, "executed_source", executed)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class StateBCHITLTrace:
    events: tuple[StateBCHITLEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        self.validate_exclusive()

    @property
    def action_events(self) -> tuple[StateBCHITLEvent, ...]:
        return tuple(event for event in self.events if event.event_kind == HITL_EVENT_ACTION)

    @property
    def samples(self) -> tuple[dict[str, Any], ...]:
        """Return causal policy/human samples; holds and rtc_no_action are excluded."""

        output: list[dict[str, Any]] = []
        for event in self.action_events:
            assert event.action is not None
            output.append(
                {
                    "step_index": event.step_index,
                    "source": event.executed_source,
                    "inference_mode": event.inference_mode,
                    "activation_id": event.activation_id,
                    "action": event.action.copy(),
                    "metadata": dict(event.metadata),
                }
            )
        return tuple(output)

    def validate_exclusive(self) -> None:
        for event in self.events:
            if event.event_kind == HITL_EVENT_ACTION and event.executed_source not in HITL_SOURCES:
                raise AssertionError("an action event must have exactly one source.")
            if event.event_kind != HITL_EVENT_ACTION and event.action is not None:
                raise AssertionError("non-action event leaked an executable action.")
        by_step: dict[int, set[str]] = {}
        for event in self.action_events:
            assert event.executed_source is not None
            by_step.setdefault(event.step_index, set()).add(event.executed_source)
        if any(len(sources) != 1 for sources in by_step.values()):
            raise AssertionError("one step contains more than one executed source.")

    def source_sequence(self) -> tuple[str, ...]:
        return tuple(event.executed_source for event in self.action_events if event.executed_source is not None)


PolicyFn = Callable[[np.ndarray], Any]
RtcFn = Callable[[np.ndarray, np.ndarray], Any]


class StateBCHITLRunner:
    """Run one deterministic source schedule with HOLD/fresh activation."""

    def __init__(self, policy: PolicyFn, *, rtc_policy: RtcFn | None = None) -> None:
        if not callable(policy):
            raise TypeError("policy must be callable.")
        if rtc_policy is not None and not callable(rtc_policy):
            raise TypeError("rtc_policy must be callable or None.")
        self.policy = policy
        self.rtc_policy = rtc_policy

    def run(
        self,
        states: Sequence[Any],
        requested_sources: Sequence[str],
        *,
        human_actions: Mapping[int, Any] | None = None,
    ) -> StateBCHITLTrace:
        if len(states) != len(requested_sources):
            raise ValueError("states and requested_sources must have the same length.")
        # ``states`` is a generic sequence and the real dataset loader passes
        # a NumPy array.  Avoid NumPy's ambiguous truth-value semantics here;
        # emptiness is the only property this guard needs.
        if len(states) == 0:
            raise ValueError("HITL runner requires at least one step.")
        humans = {} if human_actions is None else dict(human_actions)
        events: list[StateBCHITLEvent] = []
        active_source: str | None = None
        activation_counter = 0
        activation_id: str | None = None
        for step_index, (raw_state, raw_source) in enumerate(zip(states, requested_sources)):
            state = validate_state_vector(raw_state, name=f"state[{step_index}]")
            source = _source(raw_source)
            if source != active_source:
                if active_source is not None:
                    events.append(
                        StateBCHITLEvent(
                            step_index=step_index,
                            event_kind=HITL_EVENT_HOLD,
                            requested_source=source,
                            activation_id=activation_id,
                            metadata={"previous_source": active_source},
                        )
                    )
                active_source = source
                activation_counter += 1
                activation_id = f"state-bc-activation:{activation_counter}"
                events.append(
                    StateBCHITLEvent(
                        step_index=step_index,
                        event_kind=HITL_EVENT_ACTIVATION,
                        requested_source=source,
                        activation_id=activation_id,
                        metadata={"fresh_emission_required": True},
                    )
                )

            if source == HITL_SOURCE_HUMAN:
                if step_index not in humans:
                    raise ValueError(f"human source has no action for step {step_index}.")
                action = validate_action_vector(humans[step_index], name=f"human_action[{step_index}]")
                events.append(
                    StateBCHITLEvent(
                        step_index=step_index,
                        event_kind=HITL_EVENT_ACTION,
                        requested_source=source,
                        executed_source=HITL_SOURCE_HUMAN,
                        activation_id=activation_id,
                        inference_mode="human",
                        action=action,
                    )
                )
                continue

            nominal = validate_action_vector(self.policy(state), name=f"policy_action[{step_index}]")
            if source == HITL_SOURCE_POLICY:
                action = nominal
                mode = "standard"
            else:
                if self.rtc_policy is None:
                    rtc_action = None
                else:
                    rtc_action = self.rtc_policy(state, nominal)
                if rtc_action is None:
                    events.append(
                        StateBCHITLEvent(
                            step_index=step_index,
                            event_kind=HITL_EVENT_RTC_NO_ACTION,
                            requested_source=source,
                            activation_id=activation_id,
                            inference_mode="rtc",
                            metadata={"nominal_policy_sample": False},
                        )
                    )
                    continue
                action = validate_action_vector(rtc_action, name=f"rtc_action[{step_index}]")
                mode = "rtc"
            events.append(
                StateBCHITLEvent(
                    step_index=step_index,
                    event_kind=HITL_EVENT_ACTION,
                    requested_source=source,
                    executed_source=source,
                    activation_id=activation_id,
                    inference_mode=mode,
                    action=action,
                    metadata={"nominal_policy_sample": source == HITL_SOURCE_RTC},
                )
            )
        return StateBCHITLTrace(events=tuple(events))


__all__ = [
    "HITL_EVENT_ACTION",
    "HITL_EVENT_ACTIVATION",
    "HITL_EVENT_HOLD",
    "HITL_EVENT_RTC_NO_ACTION",
    "HITL_SOURCE_HUMAN",
    "HITL_SOURCE_POLICY",
    "HITL_SOURCE_RTC",
    "StateBCHITLEvent",
    "StateBCHITLRunner",
    "StateBCHITLTrace",
]
