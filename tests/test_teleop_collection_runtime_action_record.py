from __future__ import annotations

import unittest

import numpy as np

from teleop_runtime.collection.build_collect_components import CollectorComponents
from teleop_runtime.collection.async_writer import WriteJob
from teleop_runtime.collection.runtime import CollectionRuntime
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.schemas import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.session import SessionStepResult


class _FakeSession:
    def __init__(self):
        self.started = False
        self.closed = False
        self.run_count = 0

    def start(self) -> None:
        self.started = True

    def run_step(self, now=None, phase_recorder=None) -> SessionStepResult:
        self.run_count += 1
        command = TeleopCommand(timestamp=1.0)
        command.add_command(
            EffectorCommand(
                effector_name="left_hand",
                side="left",
                command_type="joint_vector",
                value=np.arange(6, dtype=np.float32),
                frame_name=None,
            )
        )
        return SessionStepResult(
            timestamp=1.0,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=0.9),
            command=command,
            command_dispatched=True,
        )

    def close(self) -> None:
        self.closed = True


class _FakeWriterPool:
    def __init__(self):
        self.started = False
        self.closed = False
        self.submitted = []

    def start(self) -> None:
        self.started = True

    def submit(self, job) -> bool:
        self.submitted.append(job)
        return True

    def join(self) -> None:
        pass

    def close(self, *, wait=True) -> None:
        self.closed = True


class CollectionRuntimeActionRecordTest(unittest.TestCase):
    def test_runtime_attaches_action_record_before_sample_builder(self) -> None:
        seen = {}

        def sample_builder(result, components):
            seen["action_record"] = result.action_record
            return None

        runtime = CollectionRuntime(
            components=CollectorComponents(session=_FakeSession()),
            sample_builder=sample_builder,
            write_planner=lambda sample, components: (),
            writer_pool=_FakeWriterPool(),
        )

        result = runtime.step(now=1.0)

        self.assertIsNotNone(result.action_record)
        self.assertIs(seen["action_record"], result.action_record)
        self.assertTrue(result.action_record.execution.sent)
        self.assertEqual(result.action_record.action.active_effectors, ("left_hand",))

    def test_runtime_writer_observer_submits_planned_jobs_and_records_phases(self) -> None:
        seen = {}
        phases = []
        writer_pool = _FakeWriterPool()

        def sample_builder(result, components):
            seen["action_record"] = result.action_record
            return {"index": 1, "action": result.action_record.action}

        def write_planner(sample, components):
            seen["sample"] = sample
            return (WriteJob(kind="unit.row", payload=sample),)

        runtime = CollectionRuntime(
            components=CollectorComponents(session=_FakeSession()),
            sample_builder=sample_builder,
            write_planner=write_planner,
            writer_pool=writer_pool,
        )

        result = runtime.step(now=1.0, phase_recorder=lambda name, elapsed: phases.append(name))

        self.assertIs(seen["action_record"], result.action_record)
        self.assertEqual(len(writer_pool.submitted), 1)
        self.assertEqual(writer_pool.submitted[0].kind, "unit.row")
        self.assertEqual(runtime.submitted_samples, 1)
        self.assertIn("runtime.action_record", phases)
        self.assertIn("runtime.sample_builder", phases)
        self.assertIn("runtime.write_planner", phases)
        self.assertIn("runtime.writer_submit", phases)

    def test_runtime_last_sample_is_cleared_when_sample_builder_skips(self) -> None:
        writer_pool = _FakeWriterPool()
        calls = {"count": 0}

        def sample_builder(result, components):
            calls["count"] += 1
            if calls["count"] == 1:
                return {"index": 1, "action": result.action_record.action}
            return None

        runtime = CollectionRuntime(
            components=CollectorComponents(session=_FakeSession()),
            sample_builder=sample_builder,
            write_planner=lambda sample, components: (),
            writer_pool=writer_pool,
        )

        runtime.step(now=1.0)
        self.assertIsNotNone(runtime.last_sample)
        runtime.step(now=2.0)
        self.assertIsNone(runtime.last_sample)


if __name__ == "__main__":
    unittest.main()
