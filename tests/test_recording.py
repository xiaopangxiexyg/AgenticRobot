from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

from teleop_runtime.recording import RecordingConfig, SessionRecorder


class SessionRecorderConfigTest(unittest.TestCase):
    @pytest.mark.integration
    def test_can_disable_command_stream_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = SessionRecorder(
                root_dir=tmp_dir,
                config=RecordingConfig(save_commands=False),
            )
            recorder.record_channels({"x": 1}, timestamp=123.0)
            recorder.record_command({"command": 1})
            recorder.close()

            root = Path(tmp_dir)
            self.assertTrue((root / "raw_inputs.jsonl").exists())
            self.assertFalse((root / "commands.jsonl").exists())

    @pytest.mark.integration
    def test_flush_every_buffers_until_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = SessionRecorder(
                root_dir=tmp_dir,
                config=RecordingConfig(flush_every=10, compact_json=True),
            )
            recorder.record_command({"a": 1, "b": [2, 3]})
            recorder.close()

            text = (Path(tmp_dir) / "commands.jsonl").read_text(encoding="utf-8")
            self.assertEqual(json.loads(text), {"a": 1, "b": [2, 3]})
            self.assertEqual(text, '{"a":1,"b":[2,3]}\n')

    def test_flush_every_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            RecordingConfig(flush_every=0)


if __name__ == "__main__":
    unittest.main()
