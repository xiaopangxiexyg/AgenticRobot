import unittest
from types import SimpleNamespace

from teleop_runtime.collection.status_snapshot import (
    camera_state_signature_text,
    camera_status_text,
    compact_log_value,
    fault_signature,
    fault_status_text,
    writer_debug_text,
)


class _CameraSource:
    def __init__(self, info):
        self._info = dict(info)

    def device_info(self):
        return dict(self._info)


class _BrokenCameraSource:
    def device_info(self):
        raise RuntimeError("camera offline")


class _WriterPool:
    def stats(self):
        return SimpleNamespace(
            queued=1,
            submitted=2,
            completed=3,
            dropped=4,
            failures=5,
            worker_count=6,
        )


class TeleopCollectionStatusSnapshotTest(unittest.TestCase):
    def test_compact_log_value_truncates_long_text(self) -> None:
        self.assertEqual(compact_log_value("abcdef", max_chars=4), "a...")
        self.assertEqual(compact_log_value("abc", max_chars=4), "abc")

    def test_writer_debug_text_formats_pool_stats(self) -> None:
        text = writer_debug_text(_WriterPool())

        self.assertEqual(
            text,
            "writer=queued:1 submitted:2 completed:3 dropped:4 failures:5 workers:6",
        )

    def test_camera_status_and_signature_cover_streaming_error_and_preview(self) -> None:
        previewer = SimpleNamespace(disabled_reason="no display")
        sources = {
            "front": _CameraSource({"frame_count": 3, "last_error": None}),
            "wrist": _BrokenCameraSource(),
        }

        status = camera_status_text(sources, camera_previewer=previewer)
        signature, state_text = camera_state_signature_text(sources, camera_previewer=previewer)

        self.assertIn("preview=disabled:no display", status)
        self.assertIn("front=frames:3", status)
        self.assertIn("wrist=status_error:RuntimeError", status)
        self.assertIn("preview:disabled:no display", signature)
        self.assertIn("front:streaming", signature)
        self.assertIn("wrist:status_error:RuntimeError", signature)
        self.assertIn("camera=preview=disabled:no display;front=streaming", state_text)

    def test_fault_signature_and_status_text_are_stable(self) -> None:
        faults = [
            SimpleNamespace(code="stale_channel", channel_name="left_wrist_pose", message="old"),
            SimpleNamespace(code="low_control_rate", channel_name=None, message=""),
        ]

        self.assertEqual(fault_signature(faults), ("low_control_rate", "stale_channel:left_wrist_pose"))
        self.assertEqual(
            fault_status_text(faults),
            "stale_channel:left_wrist_pose(old); low_control_rate",
        )


if __name__ == "__main__":
    unittest.main()
