from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..core.kernel import RuntimeKernel, RuntimeKernelConfig, RuntimeStepContext, RuntimeStepResult
from ..core.observers import RuntimeObserver
from ..schemas import TeleopCommand
from .codec import ActionVectorEncoding, decode_robot_action
from .executor import ActionExecutor, ActionExecutorConfig
from .schema import RobotAction


@dataclass(frozen=True)
class SyntheticReplayFrame:
    action: RobotAction | ActionVectorEncoding | None
    observation: Any = None
    timestamp: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


class SequenceObservationProvider:
    def __init__(self, frames: Sequence[SyntheticReplayFrame]):
        self.frames = tuple(frames)

    def start(self) -> None:
        pass

    def capture(self, context: RuntimeStepContext) -> Any:
        return self.frames[int(context.step_index)].observation

    def close(self) -> None:
        pass


class SequenceActionSource:
    def __init__(self, frames: Sequence[SyntheticReplayFrame], *, source: str = "synthetic"):
        self.frames = tuple(frames)
        self.source = str(source)

    def start(self) -> None:
        pass

    def produce_action(self, observation: Any, *, context: RuntimeStepContext | None = None) -> RobotAction | None:
        if context is None:
            raise RuntimeError("SequenceActionSource requires RuntimeStepContext.")
        frame = self.frames[int(context.step_index)]
        action = frame.action
        if action is None:
            return None
        if isinstance(action, ActionVectorEncoding):
            return decode_robot_action(
                action,
                timestamp=frame.timestamp,
                source=self.source,
                metadata=frame.metadata,
            )
        return action

    def close(self) -> None:
        pass


class SyntheticCommandDispatcher:
    """Records resolved commands without touching hardware."""

    def __init__(self, *, per_dispatch_delay_sec: float = 0.0):
        self.per_dispatch_delay_sec = float(per_dispatch_delay_sec)
        if self.per_dispatch_delay_sec < 0.0:
            raise ValueError("per_dispatch_delay_sec must be >= 0.")
        self.commands: list[TeleopCommand] = []
        self.dispatch_timestamps: list[float] = []

    def start(self) -> None:
        pass

    def dispatch(self, command: TeleopCommand, *, context: RuntimeStepContext | None = None) -> None:
        if self.per_dispatch_delay_sec > 0.0:
            time.sleep(self.per_dispatch_delay_sec)
        self.commands.append(command)
        self.dispatch_timestamps.append(time.time())

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class PhaseStats:
    count: int
    mean_sec: float
    max_sec: float
    p95_sec: float


@dataclass(frozen=True)
class SyntheticReplayReport:
    results: tuple[RuntimeStepResult, ...]
    commands: tuple[TeleopCommand, ...]

    @property
    def step_count(self) -> int:
        return len(self.results)

    @property
    def command_count(self) -> int:
        return len(self.commands)

    @property
    def sent_command_count(self) -> int:
        return sum(1 for result in self.results if result.execution is not None and result.execution.sent)

    def step_elapsed_sec(self) -> tuple[float, ...]:
        return tuple(sum(elapsed for _, elapsed in result.phases) for result in self.results)

    def phase_durations(self, phase_name: str) -> tuple[float, ...]:
        values: list[float] = []
        for result in self.results:
            for name, elapsed in result.phases:
                if name == phase_name:
                    values.append(float(elapsed))
        return tuple(values)

    def phase_stats(self) -> dict[str, PhaseStats]:
        grouped: dict[str, list[float]] = {}
        for result in self.results:
            for name, elapsed in result.phases:
                grouped.setdefault(str(name), []).append(float(elapsed))
        return {name: _stats(values) for name, values in sorted(grouped.items())}


def _stats(values: Sequence[float]) -> PhaseStats:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return PhaseStats(count=0, mean_sec=0.0, max_sec=0.0, p95_sec=0.0)
    p95_index = min(len(sorted_values) - 1, int(round(0.95 * (len(sorted_values) - 1))))
    return PhaseStats(
        count=len(sorted_values),
        mean_sec=sum(sorted_values) / len(sorted_values),
        max_sec=max(sorted_values),
        p95_sec=sorted_values[p95_index],
    )


def run_synthetic_action_replay(
    frames: Sequence[SyntheticReplayFrame],
    *,
    observer: RuntimeObserver | None = None,
    dispatch_enabled: bool = True,
    per_dispatch_delay_sec: float = 0.0,
    start_time: float = 1.0,
    step_dt_sec: float = 1.0 / 30.0,
) -> SyntheticReplayReport:
    replay_frames = tuple(frames)
    dispatcher = SyntheticCommandDispatcher(per_dispatch_delay_sec=per_dispatch_delay_sec)
    kernel = RuntimeKernel(
        observation_provider=SequenceObservationProvider(replay_frames),
        action_source=SequenceActionSource(replay_frames),
        executor=ActionExecutor(
            dispatcher=dispatcher,
            config=ActionExecutorConfig(dispatch_enabled=dispatch_enabled),
        ),
        observer=observer,
        config=RuntimeKernelConfig(metadata={"runtime": "synthetic_action_replay"}),
    )
    results: list[RuntimeStepResult] = []
    try:
        for index, frame in enumerate(replay_frames):
            timestamp = (
                float(frame.timestamp)
                if frame.timestamp is not None
                else float(start_time) + int(index) * float(step_dt_sec)
            )
            results.append(kernel.step(now=timestamp, metadata=frame.metadata))
    finally:
        kernel.close()
    return SyntheticReplayReport(
        results=tuple(results),
        commands=tuple(dispatcher.commands),
    )
