from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from robot_motion.action import (
    ActionExecutionResult,
    ActionExecutor,
    ActionRecord,
    RobotAction,
    RobotActionSource,
)
from robot_motion.runtime import (
    LifecycleStartupError,
    TeleopCommand,
    collect_lifecycle_failures,
    run_best_effort_cleanup,
)

from ..diagnostics import RateTracker
from .observers import CompositeRuntimeObserver, NullRuntimeObserver, RuntimeObserver
from .timing import PhaseTimer


class ObservationProvider(Protocol):
    def start(self) -> None:
        ...

    def capture(self, context: "RuntimeStepContext") -> Any:
        ...

    def close(self) -> None:
        ...


class StaticObservationProvider:
    def __init__(self, observation: Any = None):
        self.observation = observation

    def start(self) -> None:
        pass

    def capture(self, context: "RuntimeStepContext") -> Any:
        return self.observation

    def close(self) -> None:
        pass


@dataclass
class RuntimeStepContext:
    timestamp: float
    step_index: int
    control_hz: float
    metadata: dict[str, Any] = field(default_factory=dict)
    _phase_timer: PhaseTimer | None = field(default=None, repr=False)

    def record_phase(self, name: str, elapsed_sec: float) -> None:
        if self._phase_timer is not None:
            self._phase_timer.record(name, elapsed_sec)


@dataclass
class RuntimeStepResult:
    timestamp: float
    step_index: int
    control_hz: float
    observation: Any = None
    action: RobotAction | None = None
    command: TeleopCommand | None = None
    execution: ActionExecutionResult | None = None
    record: ActionRecord | None = None
    faults: tuple[str, ...] = ()
    phases: tuple[tuple[str, float], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeKernelConfig:
    name: str = "runtime_kernel"
    auto_start: bool = True
    raise_observer_errors: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeKernel:
    """Small realtime kernel centered on the canonical RobotAction contract."""

    def __init__(
        self,
        *,
        observation_provider: ObservationProvider,
        action_source: RobotActionSource,
        executor: ActionExecutor,
        observer: RuntimeObserver | None = None,
        config: RuntimeKernelConfig | None = None,
    ):
        self.observation_provider = observation_provider
        self.action_source = action_source
        self.executor = executor
        self.observer = observer if observer is not None else NullRuntimeObserver()
        self.config = config if config is not None else RuntimeKernelConfig()
        self.rate_tracker = RateTracker()
        self._started = False
        self._step_index = 0

    @classmethod
    def with_observers(
        cls,
        *,
        observation_provider: ObservationProvider,
        action_source: RobotActionSource,
        executor: ActionExecutor,
        observers: list[RuntimeObserver] | tuple[RuntimeObserver, ...],
        config: RuntimeKernelConfig | None = None,
    ) -> "RuntimeKernel":
        return cls(
            observation_provider=observation_provider,
            action_source=action_source,
            executor=executor,
            observer=CompositeRuntimeObserver(observers),
            config=config,
        )

    def start(self) -> None:
        if self._started:
            return
        rollback_steps: list[tuple[str, Any]] = []
        try:
            rollback_steps.append(("observation_provider", self.observation_provider.close))
            self.observation_provider.start()
            start_action_source = getattr(self.action_source, "start", None)
            close_action_source = getattr(self.action_source, "close", None)
            if callable(start_action_source):
                if callable(close_action_source):
                    rollback_steps.append(("action_source", close_action_source))
                start_action_source()
            rollback_steps.append(("executor", self.executor.close))
            self.executor.start()
            start_observer = getattr(self.observer, "start", None)
            close_observer = getattr(self.observer, "close", None)
            if callable(start_observer):
                if callable(close_observer):
                    rollback_steps.append(("observer", close_observer))
                start_observer()
        except BaseException as start_error:  # noqa: BLE001 - rollback before propagating.
            self._started = False
            rollback_failures = collect_lifecycle_failures(reversed(rollback_steps))
            if rollback_failures:
                raise LifecycleStartupError(
                    "runtime_kernel.start",
                    start_error,
                    rollback_failures,
                ) from start_error
            raise
        self._started = True

    def step(
        self,
        *,
        now: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeStepResult:
        timer = PhaseTimer()
        if self.config.auto_start and not self._started:
            self.start()
            timer.mark("kernel.start")

        timestamp = time.time() if now is None else float(now)
        control_hz = self.rate_tracker.tick(timestamp)
        context = RuntimeStepContext(
            timestamp=timestamp,
            step_index=self._step_index,
            control_hz=control_hz,
            metadata={
                **dict(self.config.metadata),
                **({} if metadata is None else dict(metadata)),
            },
            _phase_timer=timer,
        )
        timer.mark("kernel.rate_tracker")

        observation = self.observation_provider.capture(context)
        timer.mark("observation.capture")

        action = self.action_source.produce_action(observation, context=context)
        timer.mark("action.produce")

        execution: ActionExecutionResult | None = None
        command: TeleopCommand | None = None
        faults: tuple[str, ...] = ()
        if action is not None:
            execution = self.executor.execute(action, observation=observation, context=context)
            command = execution.command
            faults = tuple(execution.faults)
            timer.mark("action.execute")
        else:
            timer.mark("action.skip_execute")

        pre_observer_phases = timer.as_tuples()
        record = ActionRecord(
            timestamp=timestamp,
            observation=observation,
            action=action,
            command=command,
            execution=execution,
            phases=pre_observer_phases,
            metadata=dict(context.metadata),
        )
        result = RuntimeStepResult(
            timestamp=timestamp,
            step_index=self._step_index,
            control_hz=control_hz,
            observation=observation,
            action=action,
            command=command,
            execution=execution,
            record=record,
            faults=faults,
            phases=pre_observer_phases,
            metadata=dict(context.metadata),
        )

        try:
            self.observer.on_step(result)
            timer.mark("observer.on_step")
        except Exception:
            timer.mark("observer.on_step_error")
            if self.config.raise_observer_errors:
                raise

        phases = timer.as_tuples()
        result.phases = phases
        result.record = ActionRecord(
            timestamp=timestamp,
            observation=observation,
            action=action,
            command=command,
            execution=execution,
            phases=phases,
            metadata=dict(context.metadata),
        )
        self._step_index += 1
        return result

    def close(self) -> None:
        steps = []
        close_observer = getattr(self.observer, "close", None)
        if callable(close_observer):
            steps.append(("observer", close_observer))
        steps.append(("executor", self.executor.close))
        close_action_source = getattr(self.action_source, "close", None)
        if callable(close_action_source):
            steps.append(("action_source", close_action_source))
        steps.append(("observation_provider", self.observation_provider.close))
        try:
            run_best_effort_cleanup("runtime_kernel.close", steps)
        finally:
            self._started = False

    def __enter__(self) -> "RuntimeKernel":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
