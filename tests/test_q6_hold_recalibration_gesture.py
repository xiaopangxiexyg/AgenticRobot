from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.schemas import ObservationChannel


def _q6(tail: tuple[float, float, float]) -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, *tail], dtype=float)


def _channel(side: str, value: np.ndarray, *, valid: bool = True) -> ObservationChannel:
    return ObservationChannel(
        channel_name=f"{side}_hand_q6",
        capability="joint_vector",
        side=side,
        value=np.asarray(value, dtype=float),
        frame_name=None,
        timestamp=0.0,
        source_name="test",
        valid=valid,
    )


def _set_q6(source: real_teleop.HandQ6Source, left: np.ndarray, right: np.ndarray) -> None:
    with source._lock:
        source._latest_channels = {
            "left_hand_q6": _channel("left", left),
            "right_hand_q6": _channel("right", right),
        }


class Q6HoldRecalibrationGestureTests(unittest.TestCase):
    def test_state_machine_requires_close_release_close_sequence(self) -> None:
        gesture = real_teleop.Q6HoldRecalibrationGesture(
            real_teleop.Q6HoldRecalibrationGestureConfig(enabled=True, hold_sec=1.0)
        )
        closed = _q6((1.0, 0.99, 0.98))
        opened = _q6((0.6, 0.5, 0.4))

        self.assertIsNone(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=False,
                now=0.0,
            )
        )
        self.assertIsNone(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=False,
                now=0.5,
            )
        )
        self.assertEqual(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=False,
                now=1.01,
            ),
            real_teleop.Q6HoldRecalibrationGesture.EVENT_HOLD,
        )

        self.assertIsNone(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=True,
                now=2.5,
            )
        )
        self.assertIsNone(
            gesture.update(
                left_q6=opened,
                right_q6=opened,
                teleop_started=True,
                hold_active=True,
                now=3.0,
            )
        )
        self.assertEqual(
            gesture.update(
                left_q6=opened,
                right_q6=opened,
                teleop_started=True,
                hold_active=True,
                now=4.01,
            ),
            real_teleop.Q6HoldRecalibrationGesture.EVENT_REARMED,
        )
        self.assertIsNone(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=True,
                now=5.0,
            )
        )
        self.assertEqual(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=True,
                now=6.01,
            ),
            real_teleop.Q6HoldRecalibrationGesture.EVENT_RESTART,
        )

    def test_invalid_q6_resets_close_window(self) -> None:
        gesture = real_teleop.Q6HoldRecalibrationGesture(
            real_teleop.Q6HoldRecalibrationGestureConfig(enabled=True, hold_sec=1.0)
        )
        closed = _q6((1.0, 1.0, 1.0))

        self.assertIsNone(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=False,
                now=0.0,
            )
        )
        self.assertIsNone(
            gesture.update(
                left_q6=closed,
                right_q6=None,
                teleop_started=True,
                hold_active=False,
                now=0.8,
            )
        )
        self.assertIsNone(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=False,
                now=1.2,
            )
        )
        self.assertEqual(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=False,
                now=2.25,
            ),
            real_teleop.Q6HoldRecalibrationGesture.EVENT_HOLD,
        )

    def test_runtime_restart_requires_release_before_next_hold(self) -> None:
        gesture = real_teleop.Q6HoldRecalibrationGesture(
            real_teleop.Q6HoldRecalibrationGestureConfig(enabled=True, hold_sec=1.0)
        )
        closed = _q6((1.0, 1.0, 1.0))
        opened = _q6((0.6, 0.6, 0.6))

        gesture.mark_runtime_restarted()
        self.assertIsNone(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=False,
                now=0.0,
            )
        )
        self.assertIsNone(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=False,
                now=2.0,
            )
        )
        self.assertIsNone(
            gesture.update(
                left_q6=opened,
                right_q6=opened,
                teleop_started=True,
                hold_active=False,
                now=3.0,
            )
        )
        self.assertIsNone(
            gesture.update(
                left_q6=opened,
                right_q6=opened,
                teleop_started=True,
                hold_active=False,
                now=4.01,
            )
        )
        self.assertIsNone(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=False,
                now=5.0,
            )
        )
        self.assertEqual(
            gesture.update(
                left_q6=closed,
                right_q6=closed,
                teleop_started=True,
                hold_active=False,
                now=6.01,
            ),
            real_teleop.Q6HoldRecalibrationGesture.EVENT_HOLD,
        )

    def _make_control_with_q6_source(
        self,
        source: real_teleop.HandQ6Source,
        *,
        ui: object | None = None,
    ) -> real_teleop.RealTeleopControlInput:
        components = SimpleNamespace(
            hand_backend_mode=real_teleop.HAND_BACKEND_DMTAC_GRIPPER,
            active_sides=("left", "right"),
            left_hand_backend=None,
            right_hand_backend=None,
            left_gripper_retargeter=None,
            right_gripper_retargeter=None,
            session=SimpleNamespace(sources=[source]),
            gripper_policy={},
        )
        return real_teleop.RealTeleopControlInput(
            terminal=SimpleNamespace(
                enabled=False,
                should_quit=False,
                start=lambda: None,
                close=lambda: None,
                poll=lambda: None,
                consume_calibration_request=lambda: False,
                consume_episode_start_request=lambda: False,
                consume_episode_stop_request=lambda: False,
            ),
            ui=ui,
            args=SimpleNamespace(
                q6_hold_recalibration_gesture=True,
                q6_gesture_hold_sec=1.0,
                q6_gesture_close_threshold=0.95,
                q6_gesture_open_threshold=0.85,
                no_collection=True,
                no_episode_recording=True,
            ),
            components=components,
        )

    def test_control_input_maps_dual_q6_gesture_to_existing_hold_and_calibration_requests(self) -> None:
        source = real_teleop.HandQ6Source.from_bimanual(
            left_port="/dev/null",
            right_port="/dev/null",
        )
        closed = _q6((1.0, 1.0, 1.0))
        opened = _q6((0.6, 0.6, 0.6))
        _set_q6(source, closed, closed)

        control = self._make_control_with_q6_source(source)
        control.publish_teleop_running()

        control._poll_q6_hold_recalibration_gesture(now=0.0)
        control._poll_q6_hold_recalibration_gesture(now=1.01)
        self.assertTrue(control.consume_hold_request())
        self.assertEqual(control.hold_request_reason(), "q6_gesture")

        _set_q6(source, opened, opened)
        control._poll_q6_hold_recalibration_gesture(now=2.0)
        control._poll_q6_hold_recalibration_gesture(now=3.01)
        self.assertFalse(control.consume_calibration_request())

        control._tracker_side_check = {
            "status": "correct",
            "warning_required": False,
            "message": "correct",
        }
        _set_q6(source, closed, closed)
        control._poll_q6_hold_recalibration_gesture(now=4.0)
        control._poll_q6_hold_recalibration_gesture(now=5.01)
        self.assertFalse(control._two_pedal_fast_home_requested)
        self.assertTrue(control.consume_calibration_request())

        control.publish_teleop_running()
        control._poll_q6_hold_recalibration_gesture(now=6.0)
        control._poll_q6_hold_recalibration_gesture(now=7.5)
        self.assertFalse(control.consume_hold_request())

        _set_q6(source, opened, opened)
        control._poll_q6_hold_recalibration_gesture(now=8.0)
        control._poll_q6_hold_recalibration_gesture(now=9.01)
        self.assertFalse(control.consume_hold_request())

        _set_q6(source, closed, closed)
        control._poll_q6_hold_recalibration_gesture(now=10.0)
        control._poll_q6_hold_recalibration_gesture(now=11.01)
        self.assertTrue(control.consume_hold_request())
        self.assertEqual(control.hold_request_reason(), "q6_gesture")

    def test_manual_restart_after_hold_rearms_q6_after_release(self) -> None:
        source = real_teleop.HandQ6Source.from_bimanual(
            left_port="/dev/null",
            right_port="/dev/null",
        )
        closed = _q6((1.0, 1.0, 1.0))
        released = _q6((0.6, 0.6, 0.6))
        _set_q6(source, closed, closed)

        control = self._make_control_with_q6_source(source)
        control.publish_teleop_running()

        control._poll_q6_hold_recalibration_gesture(now=0.0)
        control._poll_q6_hold_recalibration_gesture(now=1.01)
        self.assertTrue(control.consume_hold_request())

        control._calibration_requested = True
        self.assertTrue(control.consume_calibration_request())
        control.publish_teleop_running()

        control._poll_q6_hold_recalibration_gesture(now=2.0)
        control._poll_q6_hold_recalibration_gesture(now=3.5)
        self.assertFalse(control.consume_hold_request())

        _set_q6(source, released, released)
        control._poll_q6_hold_recalibration_gesture(now=4.0)
        control._poll_q6_hold_recalibration_gesture(now=5.01)
        self.assertFalse(control.consume_hold_request())

        _set_q6(source, closed, closed)
        control._poll_q6_hold_recalibration_gesture(now=6.0)
        control._poll_q6_hold_recalibration_gesture(now=7.01)
        self.assertTrue(control.consume_hold_request())
        self.assertEqual(control.hold_request_reason(), "q6_gesture")

    def test_failed_recalibration_returns_to_hold_for_retry(self) -> None:
        source = real_teleop.HandQ6Source.from_bimanual(
            left_port="/dev/null",
            right_port="/dev/null",
        )
        closed = _q6((1.0, 1.0, 1.0))
        _set_q6(source, closed, closed)
        control = self._make_control_with_q6_source(source)
        control.publish_teleop_running()

        control.request_hold(reason="q6_gesture")
        self.assertTrue(control.consume_hold_request())
        control._calibration_requested = True
        self.assertTrue(control.consume_calibration_request())
        self.assertFalse(control.hold_active())

        control.publish_calibration_waiting("runtime home 标定失败")
        self.assertTrue(control.hold_active())
        control._calibration_requested = True
        self.assertTrue(control.consume_calibration_request())


if __name__ == "__main__":
    unittest.main()
