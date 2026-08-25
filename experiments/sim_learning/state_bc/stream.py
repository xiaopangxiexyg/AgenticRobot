"""Streaming state-BC source switching with a dual-space action trace."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from robot_motion.deploy import DeploymentObservation, NormalizedAction
from robot_motion.deploy.schema import DEPLOY_ACTION_DIM

from .contracts import validate_action_vector, validate_state_vector
from .hitl import (
    HITL_EVENT_ACTION,
    HITL_EVENT_ACTIVATION,
    HITL_EVENT_HOLD,
    HITL_EVENT_RTC_NO_ACTION,
    HITL_SOURCE_HUMAN,
    HITL_SOURCE_POLICY,
    HITL_SOURCE_RTC,
)


STREAM_RUNTIME_ACTION_DIM = DEPLOY_ACTION_DIM
STREAM_RUNTIME_ACTION_SPACE = "bimanual_ee_pose_v1"
STREAM_RUNTIME_ACTION_LAYOUT = "deploy_action_vector.ee_pose_q6.v1"
STREAM_SOURCES = frozenset(
    {HITL_SOURCE_POLICY, HITL_SOURCE_HUMAN, HITL_SOURCE_RTC}
)


def _source(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in STREAM_SOURCES:
        raise ValueError(f"unsupported state BC stream source {value!r}.")
    return normalized


def _runtime_action(value: Any, *, source: str) -> NormalizedAction:
    if isinstance(value, NormalizedAction):
        action = value
    else:
        try:
            action = NormalizedAction(
                vector=np.asarray(value, dtype=np.float32).reshape(-1)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("runtime action must be a finite 26D vector.") from exc
    if not isinstance(action, NormalizedAction):
        raise TypeError("runtime action must be a NormalizedAction or vector.")
    vector = np.asarray(action.vector, dtype=np.float32).reshape(-1)
    if vector.shape != (STREAM_RUNTIME_ACTION_DIM,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"runtime action must be finite with shape ({STREAM_RUNTIME_ACTION_DIM},).")
    metadata = dict(action.metadata)
    metadata.setdefault("source", f"state_bc_{source}")
    metadata.setdefault("action_space", STREAM_RUNTIME_ACTION_SPACE)
    metadata.setdefault("action_layout", STREAM_RUNTIME_ACTION_LAYOUT)
    metadata.setdefault("action_pose_mode", "absolute")
    return NormalizedAction(
        vector=vector.copy(),
        arm_joint_targets={
            str(side): np.asarray(target, dtype=np.float32).reshape(-1).copy()
            for side, target in action.arm_joint_targets.items()
        },
        metadata=metadata,
        arm_hold_prediction=action.arm_hold_prediction,
    )


class StateBCStreamPolicyAdapter(Protocol):
    def capture_model_state(self, observation: DeploymentObservation) -> np.ndarray:
        ...

    def predict(self, observation: DeploymentObservation) -> Any:
        ...

    def project_model_action(
        self,
        model_action: Any,
        *,
        source: str | None = None,
    ) -> NormalizedAction:
        ...


class StateBCStreamObservationSource(Protocol):
    def capture(self) -> DeploymentObservation:
        ...


StateBCStreamTransitionFn = Callable[
    ["StateBCStreamEvent", DeploymentObservation], Any
]
StateBCStreamExecuteFn = Callable[
    [NormalizedAction, "StateBCStreamEvent", DeploymentObservation], Any
]
StateBCStreamRtcFn = Callable[
    [np.ndarray, np.ndarray, NormalizedAction], Any
]


@dataclass(frozen=True)
class StateBCStreamEvent:
    step_index: int
    event_kind: str
    requested_source: str
    executed_source: str | None = None
    activation_id: str | None = None
    inference_mode: str | None = None
    model_state: np.ndarray | None = None
    model_action: np.ndarray | None = None
    runtime_action: NormalizedAction | None = None
    execution_receipt: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_index = int(self.step_index)
        if step_index < 0:
            raise ValueError("step_index must be non-negative.")
        event_kind = str(self.event_kind).strip().lower()
        if event_kind not in {
            HITL_EVENT_HOLD,
            HITL_EVENT_ACTIVATION,
            HITL_EVENT_ACTION,
            HITL_EVENT_RTC_NO_ACTION,
        }:
            raise ValueError(f"unsupported stream event kind {event_kind!r}.")
        requested_source = _source(self.requested_source)
        executed_source = (
            None if self.executed_source is None else _source(self.executed_source)
        )
        model_state = (
            None
            if self.model_state is None
            else validate_state_vector(self.model_state, name="stream.model_state")
        )
        model_action = (
            None
            if self.model_action is None
            else validate_action_vector(self.model_action, name="stream.model_action")
        )
        runtime_action = self.runtime_action
        if runtime_action is not None:
            runtime_action = _runtime_action(
                runtime_action,
                source=executed_source or requested_source,
            )
        if event_kind == HITL_EVENT_ACTION:
            if executed_source is None:
                raise ValueError("action events require one executed source.")
            if runtime_action is None:
                raise ValueError("action events require one runtime action.")
        elif runtime_action is not None:
            raise ValueError("non-action events cannot carry an executable action.")
        if event_kind == HITL_EVENT_RTC_NO_ACTION and requested_source != HITL_SOURCE_RTC:
            raise ValueError("rtc_no_action must be requested by rtc.")
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "event_kind", event_kind)
        object.__setattr__(self, "requested_source", requested_source)
        object.__setattr__(self, "executed_source", executed_source)
        object.__setattr__(self, "model_state", model_state)
        object.__setattr__(self, "model_action", model_action)
        object.__setattr__(self, "runtime_action", runtime_action)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class StateBCStreamTrace:
    events: tuple[StateBCStreamEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        self.validate_exclusive()

    @property
    def action_events(self) -> tuple[StateBCStreamEvent, ...]:
        return tuple(event for event in self.events if event.event_kind == HITL_EVENT_ACTION)

    @property
    def runtime_actions(self) -> tuple[NormalizedAction, ...]:
        return tuple(
            event.runtime_action
            for event in self.action_events
            if event.runtime_action is not None
        )

    def source_sequence(self) -> tuple[str, ...]:
        return tuple(
            event.executed_source
            for event in self.action_events
            if event.executed_source is not None
        )

    @property
    def samples(self) -> tuple[dict[str, Any], ...]:
        def sample(event: StateBCStreamEvent) -> dict[str, Any]:
            assert event.runtime_action is not None
            return {
                "step_index": event.step_index,
                "source": event.executed_source,
                "inference_mode": event.inference_mode,
                "activation_id": event.activation_id,
                "model_state": None if event.model_state is None else event.model_state.copy(),
                "model_action": None if event.model_action is None else event.model_action.copy(),
                "runtime_action": event.runtime_action.vector.copy(),
                "runtime_action_metadata": dict(event.runtime_action.metadata),
                "metadata": dict(event.metadata),
            }
        return tuple(sample(event) for event in self.action_events)

    def validate_exclusive(self) -> None:
        by_step: dict[int, set[str]] = {}
        for event in self.events:
            if event.event_kind != HITL_EVENT_ACTION:
                if event.runtime_action is not None:
                    raise AssertionError("non-action event leaked a runtime action.")
                continue
            if event.executed_source not in STREAM_SOURCES:
                raise AssertionError("an action event must have exactly one source.")
            by_step.setdefault(event.step_index, set()).add(event.executed_source)
        if any(len(sources) != 1 for sources in by_step.values()):
            raise AssertionError("one step contains more than one executed source.")


class StateBCStreamingRunner:
    def __init__(
        self,
        *,
        observation_source: StateBCStreamObservationSource,
        policy_adapter: StateBCStreamPolicyAdapter,
        rtc_policy: StateBCStreamRtcFn | None = None,
        transition: StateBCStreamTransitionFn | None = None,
        execute: StateBCStreamExecuteFn | None = None,
    ) -> None:
        if not callable(getattr(observation_source, "capture", None)):
            raise TypeError("observation_source must expose capture().")
        for name in ("capture_model_state", "predict", "project_model_action"):
            if not callable(getattr(policy_adapter, name, None)):
                raise TypeError(f"policy_adapter must expose {name}().")
        if rtc_policy is not None and not callable(rtc_policy):
            raise TypeError("rtc_policy must be callable or None.")
        if transition is not None and not callable(transition):
            raise TypeError("transition must be callable or None.")
        if execute is not None and not callable(execute):
            raise TypeError("execute must be callable or None.")
        self.observation_source = observation_source
        self.policy_adapter = policy_adapter
        self.rtc_policy = rtc_policy
        self.transition = transition
        self.execute = execute
        self.reset()

    def reset(self) -> None:
        self._active_source: str | None = None
        self._activation_counter = 0
        self._activation_id: str | None = None

    def _emit(
        self,
        event: StateBCStreamEvent,
        observation: DeploymentObservation,
        events: list[StateBCStreamEvent],
    ) -> None:
        if event.event_kind == HITL_EVENT_ACTION:
            assert event.runtime_action is not None
            receipt = None if self.execute is None else self.execute(
                event.runtime_action, event, observation
            )
        else:
            receipt = None if self.transition is None else self.transition(event, observation)
        events.append(replace(event, execution_receipt=receipt))

    def _event(
        self,
        *,
        step_index: int,
        source: str,
        kind: str,
        observation: DeploymentObservation,
        events: list[StateBCStreamEvent],
        activation_id: str | None = None,
        inference_mode: str | None = None,
        model_state: np.ndarray | None = None,
        model_action: np.ndarray | None = None,
        runtime_action: NormalizedAction | None = None,
        metadata: Mapping[str, Any] | None = None,
        notify: bool = True,
    ) -> None:
        event = StateBCStreamEvent(
            step_index=step_index, event_kind=kind, requested_source=source,
            executed_source=source if kind == HITL_EVENT_ACTION else None,
            activation_id=activation_id, inference_mode=inference_mode,
            model_state=model_state, model_action=model_action,
            runtime_action=runtime_action, metadata={} if metadata is None else metadata,
        )
        if notify:
            self._emit(event, observation, events)
        else:
            events.append(event)

    def _activate(
        self,
        *,
        source: str,
        step_index: int,
        model_state: np.ndarray,
        observation: DeploymentObservation,
        events: list[StateBCStreamEvent],
    ) -> None:
        if self._active_source is not None:
            self._event(
                step_index=step_index, kind=HITL_EVENT_HOLD, source=source,
                activation_id=self._activation_id, model_state=model_state,
                metadata={"previous_source": self._active_source},
                observation=observation, events=events,
            )
        self._active_source = source
        self._activation_counter += 1
        self._activation_id = f"state-bc-stream-activation:{self._activation_counter}"
        self._event(
            step_index=step_index, kind=HITL_EVENT_ACTIVATION, source=source,
            activation_id=self._activation_id, model_state=model_state,
            metadata={
                "fresh_observation_required": True,
                "runtime_action_dim": STREAM_RUNTIME_ACTION_DIM,
                "runtime_action_space": STREAM_RUNTIME_ACTION_SPACE,
            },
            observation=observation, events=events,
        )

    def step(
        self,
        requested_source: str,
        *,
        human_action: Any | None = None,
        step_index: int | None = None,
    ) -> tuple[StateBCStreamEvent, ...]:
        source = _source(requested_source)
        observation = self.observation_source.capture()
        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation_source.capture() must return DeploymentObservation.")
        resolved_step = int(observation.step_index if step_index is None else step_index)
        if resolved_step < 0:
            raise ValueError("step_index must be non-negative.")
        model_state = validate_state_vector(
            self.policy_adapter.capture_model_state(observation),
            name="stream.model_state",
        )
        human_runtime_action = None
        if source == HITL_SOURCE_HUMAN:
            if human_action is None:
                raise ValueError(f"human source has no action for step {resolved_step}.")
            human_runtime_action = _runtime_action(human_action, source=source)
        events: list[StateBCStreamEvent] = []
        if source != self._active_source:
            self._activate(
                source=source,
                step_index=resolved_step,
                model_state=model_state,
                observation=observation,
                events=events,
            )

        if source == HITL_SOURCE_HUMAN:
            self._event(
                step_index=resolved_step, source=source, kind=HITL_EVENT_ACTION,
                activation_id=self._activation_id, inference_mode="human",
                model_state=model_state, runtime_action=human_runtime_action,
                observation=observation, events=events,
            )
            return tuple(events)

        prediction = self.policy_adapter.predict(observation)
        nominal_model_action = validate_action_vector(
            prediction.model_action,
            name="stream.nominal_model_action",
        )
        nominal_runtime_action = _runtime_action(
            prediction.actions[0],
            source=HITL_SOURCE_POLICY,
        )
        if source == HITL_SOURCE_POLICY:
            self._event(
                step_index=resolved_step, source=source, kind=HITL_EVENT_ACTION,
                activation_id=self._activation_id, inference_mode="standard",
                model_state=model_state, model_action=nominal_model_action,
                runtime_action=nominal_runtime_action,
                metadata={"runtime_action_space": STREAM_RUNTIME_ACTION_SPACE},
                observation=observation, events=events,
            )
            return tuple(events)

        if self.rtc_policy is None:
            rtc_output = None
        else:
            rtc_output = self.rtc_policy(
                model_state,
                nominal_model_action.copy(),
                nominal_runtime_action,
            )
        if rtc_output is None:
            self._event(
                step_index=resolved_step, source=source, kind=HITL_EVENT_RTC_NO_ACTION,
                activation_id=self._activation_id, inference_mode="rtc",
                model_state=model_state,
                metadata={"nominal_policy_sample": False, "runtime_action_submitted": False},
                observation=observation, events=events, notify=False,
            )
            return tuple(events)

        replacement_model_action: np.ndarray | None = None
        if isinstance(rtc_output, NormalizedAction):
            replacement_runtime_action = _runtime_action(
                rtc_output,
                source=source,
            )
            output_space = "runtime_26d"
        else:
            candidate = np.asarray(rtc_output, dtype=np.float32).reshape(-1)
            if candidate.shape == (STREAM_RUNTIME_ACTION_DIM,):
                replacement_runtime_action = _runtime_action(
                    candidate,
                    source=source,
                )
                output_space = "runtime_26d"
            else:
                replacement_model_action = validate_action_vector(
                    candidate,
                    name="stream.rtc_model_action",
                )
                replacement_runtime_action = _runtime_action(
                    self.policy_adapter.project_model_action(
                        replacement_model_action,
                        source="state_bc_rtc",
                    ),
                    source=source,
                )
                output_space = "model_16d_projected_runtime_26d"
        self._event(
            step_index=resolved_step, source=source, kind=HITL_EVENT_ACTION,
            activation_id=self._activation_id, inference_mode="rtc",
            model_state=model_state, model_action=replacement_model_action,
            runtime_action=replacement_runtime_action,
            metadata={
                "nominal_policy_sample": True,
                "runtime_action_space": STREAM_RUNTIME_ACTION_SPACE,
                "rtc_output_space": output_space,
            },
            observation=observation, events=events,
        )
        return tuple(events)

    def run(
        self,
        requested_sources: Sequence[str],
        *,
        human_actions: Mapping[int, Any] | None = None,
    ) -> StateBCStreamTrace:
        if len(requested_sources) == 0:
            raise ValueError("state BC stream requires at least one source.")
        humans = {} if human_actions is None else dict(human_actions)
        self.reset()
        events: list[StateBCStreamEvent] = []
        for step_index, source in enumerate(requested_sources):
            step_events = self.step(
                source,
                human_action=humans.get(step_index),
                step_index=step_index,
            )
            events.extend(step_events)
        return StateBCStreamTrace(events=tuple(events))


__all__ = [
    "STREAM_RUNTIME_ACTION_DIM",
    "STREAM_RUNTIME_ACTION_LAYOUT",
    "STREAM_RUNTIME_ACTION_SPACE",
    "StateBCStreamEvent",
    "StateBCStreamExecuteFn",
    "StateBCStreamObservationSource",
    "StateBCStreamPolicyAdapter",
    "StateBCStreamRtcFn",
    "StateBCStreamTrace",
    "StateBCStreamTransitionFn",
    "StateBCStreamingRunner",
]
