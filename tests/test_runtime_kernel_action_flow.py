from __future__ import annotations

import time
import unittest

import numpy as np

from teleop_runtime.action import (
    ActionExecutor,
    NullCommandDispatcher,
    RobotAction,
    StaticActionSource,
)
from teleop_runtime.core import (
    DropOldestAsyncObserver,
    RuntimeKernel,
    RuntimeKernelConfig,
    StaticObservationProvider,
)


class _RecordingObserver:
    def __init__(self, events: list[str]):
        self.events = events
        self.results = []

    def start(self) -> None:
        self.events.append("observer.start")

    def on_step(self, result) -> None:
        self.events.append("observer.on_step")
        self.results.append(result)

    def close(self) -> None:
        self.events.append("observer.close")


class _RecordingDispatcher(NullCommandDispatcher):
    def __init__(self, events: list[str]):
        super().__init__()
        self.events = events

    def dispatch(self, command, *, context=None) -> None:
        self.events.append("dispatch")
        super().dispatch(command, context=context)


class RuntimeKernelActionFlowTest(unittest.TestCase):
    def test_kernel_runs_observation_action_execute_observer_sequence(self) -> None:
        events: list[str] = []
        observation = {"state": "pre_action"}
        action = RobotAction.bimanual_qpos(
            left_arm=np.arange(6, dtype=np.float32),
            timestamp=10.0,
            source="test",
        )
        dispatcher = _RecordingDispatcher(events)
        observer = _RecordingObserver(events)
        kernel = RuntimeKernel(
            observation_provider=StaticObservationProvider(observation),
            action_source=StaticActionSource(action),
            executor=ActionExecutor(dispatcher=dispatcher),
            observer=observer,
            config=RuntimeKernelConfig(metadata={"mode": "unit"}),
        )

        result = kernel.step(now=100.0)
        kernel.close()

        self.assertIn("dispatch", events)
        self.assertIn("observer.on_step", events)
        self.assertLess(events.index("dispatch"), events.index("observer.on_step"))
        self.assertEqual(result.observation, observation)
        self.assertIs(result.action, action)
        self.assertTrue(result.execution.sent)
        self.assertEqual(sorted(result.command.commands), ["left_arm"])
        self.assertEqual(result.record.action, action)
        self.assertEqual(result.record.observation, observation)
        self.assertEqual(result.record.metadata["mode"], "unit")
        self.assertIn("action.execute", [name for name, _ in result.phases])

    def test_kernel_can_skip_execute_when_no_action_is_available(self) -> None:
        dispatcher = NullCommandDispatcher()
        kernel = RuntimeKernel(
            observation_provider=StaticObservationProvider({"state": "hold"}),
            action_source=StaticActionSource(None),
            executor=ActionExecutor(dispatcher=dispatcher),
        )

        result = kernel.step(now=1.0)

        self.assertIsNone(result.action)
        self.assertIsNone(result.command)
        self.assertIsNone(result.execution)
        self.assertEqual(dispatcher.commands, [])
        self.assertIn("action.skip_execute", [name for name, _ in result.phases])

    def test_drop_oldest_async_observer_does_not_block_step_submission(self) -> None:
        class SlowObserver:
            def __init__(self):
                self.seen = []

            def start(self) -> None:
                pass

            def on_step(self, result) -> None:
                time.sleep(0.05)
                self.seen.append(result.step_index)

            def close(self) -> None:
                pass

        slow = SlowObserver()
        observer = DropOldestAsyncObserver(slow, max_queue_size=1)
        action = RobotAction.bimanual_qpos(left_arm=np.zeros((6,), dtype=np.float32))
        kernel = RuntimeKernel(
            observation_provider=StaticObservationProvider({}),
            action_source=StaticActionSource(action),
            executor=ActionExecutor(dispatcher=NullCommandDispatcher()),
            observer=observer,
        )

        started = time.perf_counter()
        for index in range(5):
            kernel.step(now=float(index))
        elapsed = time.perf_counter() - started
        kernel.close()

        self.assertLess(elapsed, 0.20)
        self.assertGreaterEqual(observer.dropped, 1)


if __name__ == "__main__":
    unittest.main()
