import unittest
from types import SimpleNamespace

from teleop_runtime.collection.diagnostics import (
    PhaseTimer,
    build_end_to_end_latency_breakdown,
    format_end_to_end_latency_breakdown,
    format_phase_timings,
    should_report_debug_frozen,
)


class _FakeClock:
    def __init__(self, value: float = 0.0):
        self.value = float(value)

    def __call__(self) -> float:
        return float(self.value)

    def advance(self, delta: float) -> None:
        self.value += float(delta)


class TeleopCollectionDiagnosticsTest(unittest.TestCase):
    def test_phase_timer_records_marks_and_nested_phases(self) -> None:
        clock = _FakeClock(10.0)
        timer = PhaseTimer(clock=clock)

        clock.advance(0.010)
        self.assertAlmostEqual(timer.mark("poll"), 0.010)
        timer.record("runtime.step", 0.025)
        clock.advance(0.005)
        self.assertAlmostEqual(timer.mark("status"), 0.005)

        self.assertEqual([name for name, _ in timer.phases], ["poll", "runtime.step", "status"])
        self.assertAlmostEqual(timer.phases[0][1], 0.010)
        self.assertAlmostEqual(timer.phases[1][1], 0.025)
        self.assertAlmostEqual(timer.phases[2][1], 0.005)
        self.assertAlmostEqual(timer.elapsed_sec(), 0.015)

    def test_format_phase_timings_sorts_by_elapsed_descending(self) -> None:
        text = format_phase_timings(
            (
                ("fast", 0.001),
                ("slow", 0.050),
                ("medium", 0.010),
            ),
            limit=2,
        )

        self.assertEqual(text, "phases=slow:50.0ms, medium:10.0ms")

    def test_format_phase_timings_handles_empty_input(self) -> None:
        self.assertEqual(format_phase_timings(()), "phases=none")

    def test_should_report_debug_frozen_uses_total_phase_and_force(self) -> None:
        self.assertTrue(
            should_report_debug_frozen(
                elapsed_sec=0.200,
                phases=(("fast", 0.010),),
                threshold_sec=0.100,
            )
        )

    def test_builds_end_to_end_latency_breakdown_from_result_timestamps(self) -> None:
        result = SimpleNamespace(
            timestamp=100.000,
            observation=SimpleNamespace(timestamp=99.960),
            command=SimpleNamespace(timestamp=99.950),
            channels={
                "left_wrist_pose": SimpleNamespace(
                    timestamp=99.970,
                    metadata={
                        "receiver_timestamp": 99.980,
                        "sender_timestamp": 99.900,
                    },
                ),
                "right_hand_q6": SimpleNamespace(
                    timestamp=99.990,
                    metadata={},
                ),
            },
        )

        breakdown = build_end_to_end_latency_breakdown(result, reference_time=100.100)

        self.assertAlmostEqual(breakdown.step_age_sec, 0.100)
        self.assertAlmostEqual(breakdown.observation_age_sec, 0.140)
        self.assertAlmostEqual(breakdown.command_age_sec, 0.150)
        self.assertEqual([item.name for item in breakdown.channel_age_sec], [
            "left_wrist_pose",
            "right_hand_q6",
        ])
        self.assertAlmostEqual(breakdown.channel_age_sec[0].age_sec, 0.130)
        self.assertEqual([item.name for item in breakdown.receiver_age_sec], ["left_wrist_pose"])
        self.assertAlmostEqual(breakdown.receiver_age_sec[0].age_sec, 0.120)
        self.assertEqual([item.name for item in breakdown.sender_age_sec], ["left_wrist_pose"])
        self.assertAlmostEqual(breakdown.sender_age_sec[0].age_sec, 0.200)

    def test_formats_end_to_end_latency_breakdown_with_limited_channels(self) -> None:
        result = SimpleNamespace(
            timestamp=10.0,
            observation=None,
            command=None,
            channels={
                "fast": SimpleNamespace(timestamp=9.990, metadata={}),
                "slow": SimpleNamespace(timestamp=9.900, metadata={}),
            },
        )
        breakdown = build_end_to_end_latency_breakdown(result, reference_time=10.100)

        text = format_end_to_end_latency_breakdown(breakdown, limit=1)

        self.assertIn("step_age:100.0ms", text)
        self.assertIn("obs_age:none", text)
        self.assertIn("cmd_age:none", text)
        self.assertIn("channel_age=slow:200.0ms", text)
        self.assertIn("receiver_age=none", text)
        self.assertIn("sender_age=none", text)
        self.assertTrue(
            should_report_debug_frozen(
                elapsed_sec=0.020,
                phases=(("slow_phase", 0.150),),
                threshold_sec=0.100,
            )
        )
        self.assertTrue(
            should_report_debug_frozen(
                elapsed_sec=0.020,
                phases=(),
                threshold_sec=0.100,
                force=True,
            )
        )
        self.assertFalse(
            should_report_debug_frozen(
                elapsed_sec=0.020,
                phases=(("fast", 0.030),),
                threshold_sec=0.100,
            )
        )


if __name__ == "__main__":
    unittest.main()
