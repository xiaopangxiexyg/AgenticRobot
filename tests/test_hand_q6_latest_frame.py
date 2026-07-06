from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
import time
from pathlib import Path

import numpy as np

from hardware.teleop_hand import hand_q6_reader as runtime_hand_q6_reader
from hardware.teleop_hand.hand_q6_reader import (
    BufferedSerialLineReader as RuntimeBufferedSerialLineReader,
)


def _angles_line(base: int) -> bytes:
    values = [base + offset for offset in range(6)]
    body = ", ".join(f"a{index}={value}" for index, value in enumerate(values))
    return f"ANGLES: {body}\n".encode("utf-8")


class _FakeSerial:
    def __init__(self, payload: bytes = b"") -> None:
        self._buffer = bytearray(payload)
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return len(self._buffer)

    def read(self, size: int = 1) -> bytes:
        if size <= 0 or not self._buffer:
            return b""
        chunk = bytes(self._buffer[:size])
        del self._buffer[:size]
        return chunk

    def append(self, payload: bytes) -> None:
        self._buffer.extend(payload)

    def reset_input_buffer(self) -> None:
        self._buffer.clear()

    def close(self) -> None:
        self.closed = True


class _FakeCalibratingSerial:
    def __init__(self, line: bytes, count: int = 64) -> None:
        self._lines = [line for _ in range(count)]
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return 0

    def read(self, size: int = 1) -> bytes:
        del size
        return b""

    def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return _angles_line(200)

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _assert_list_close(testcase: unittest.TestCase, actual, expected) -> None:
    testcase.assertEqual(len(actual), len(expected))
    for left, right in zip(actual, expected):
        testcase.assertAlmostEqual(float(left), float(right), places=6)


class BufferedSerialLineReaderTests(unittest.TestCase):
    def test_runtime_reader_returns_latest_complete_angles_frame_from_backlog(self) -> None:
        serial_obj = _FakeSerial(
            _angles_line(0)
            + b"garbage\n"
            + _angles_line(10)
            + _angles_line(20)
        )
        reader = RuntimeBufferedSerialLineReader(serial_obj)

        angles = reader.read_latest_parsed(runtime_hand_q6_reader.parse_angles_line)

        self.assertEqual(angles, [20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
        self.assertEqual(reader.last_read_stats["parsed_frames"], 3)
        self.assertEqual(reader.last_read_stats["stale_frames_dropped"], 2)

    def test_runtime_reader_deadline_returns_none_without_frame(self) -> None:
        serial_obj = _FakeSerial()
        reader = RuntimeBufferedSerialLineReader(serial_obj)

        angles = reader.read_latest_parsed_until(
            runtime_hand_q6_reader.parse_angles_line,
            deadline=time.time() + 0.01,
        )

        self.assertIsNone(angles)
        self.assertEqual(reader.last_read_stats["parsed_frames"], 0)

    def test_single_hand_runtime_reader_drops_stale_backlog(self) -> None:
        serial_obj = _FakeSerial()
        with unittest.mock.patch.object(runtime_hand_q6_reader.serial, "Serial", return_value=serial_obj):
            reader = runtime_hand_q6_reader.SingleHandQ6Reader(
                port="/dev/test",
                side="right",
                raw_only=True,
            )
            serial_obj.append(_angles_line(0) + _angles_line(10) + _angles_line(20))
            packet = reader.read_frame()
            reader.close()

        np.testing.assert_allclose(
            packet["angles_deg"],
            np.asarray([20.0, 21.0, 22.0, 23.0, 24.0, 25.0], dtype=np.float32),
        )
        self.assertEqual(packet["serial_read_stats"]["stale_frames_dropped"], 2)
        self.assertTrue(serial_obj.closed)


class HandQ6NormalizationTests(unittest.TestCase):
    def test_runtime_calibration_prefers_current_port_specific_zero_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "exo_hand_zero_pose.json"
            path.write_text(
                json.dumps(
                    {
                        "calibrations": {
                            "right": {
                                "deg_at_0": [100.0] * 6,
                                "deg_at_1": [140.0, 30.0, 160.0, 160.0, 160.0, 160.0],
                            }
                        },
                        "port_calibrations": {
                            "right": {
                                "/dev/current-q6": {
                                    "deg_at_0": [200.0] * 6,
                                    "deg_at_1": [240.0, 130.0, 260.0, 260.0, 260.0, 260.0],
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            calib = runtime_hand_q6_reader.load_exo_hand_calibration(
                "right",
                path,
                port="/dev/current-q6",
            )

        self.assertEqual(calib["deg_at_0"].tolist(), [200.0] * 6)
        self.assertEqual(calib["deg_at_1"].tolist(), [240.0, 130.0, 260.0, 260.0, 260.0, 260.0])

    def test_runtime_calibration_rejects_saved_zero_pose_from_different_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "exo_hand_zero_pose.json"
            path.write_text(
                json.dumps(
                    {
                        "calibrations": {
                            "right": {
                                "deg_at_0": [100.0] * 6,
                                "deg_at_1": [140.0, 30.0, 160.0, 160.0, 160.0, 160.0],
                            }
                        },
                        "port_calibrations": {
                            "right": {
                                "/dev/other-q6": {
                                    "deg_at_0": [200.0] * 6,
                                    "deg_at_1": [240.0, 130.0, 260.0, 260.0, 260.0, 260.0],
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(KeyError, "Refusing to reuse"):
                runtime_hand_q6_reader.load_exo_hand_calibration(
                    "right",
                    path,
                    port="/dev/current-q6",
                )

    def test_runtime_calibration_update_records_current_port_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "exo_hand_zero_pose.json"
            runtime_hand_q6_reader._update_fixed_exo_hand_calibration(
                side="right",
                port="/dev/current-q6",
                baudrate=115200,
                deg_at_0=np.asarray([200.0] * 6, dtype=np.float32),
                record_path=None,
                calibration_path=path,
            )

            calib = runtime_hand_q6_reader.load_exo_hand_calibration(
                "right",
                path,
                port="/dev/current-q6",
            )

        self.assertEqual(calib["deg_at_0"].tolist(), [200.0] * 6)
        self.assertEqual(calib["deg_at_1"].tolist(), [240.0, 130.0, 260.0, 260.0, 260.0, 260.0])

    def test_single_hand_reader_auto_calibrates_missing_current_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "exo_hand_zero_pose.json"
            path.write_text(
                json.dumps(
                    {
                        "calibrations": {
                            "right": {
                                "deg_at_0": [100.0] * 6,
                                "deg_at_1": [140.0, 30.0, 160.0, 160.0, 160.0, 160.0],
                            }
                        },
                        "port_calibrations": {
                            "right": {
                                "/dev/other-q6": {
                                    "deg_at_0": [150.0] * 6,
                                    "deg_at_1": [190.0, 80.0, 210.0, 210.0, 210.0, 210.0],
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            serial_obj = _FakeCalibratingSerial(_angles_line(200))
            with unittest.mock.patch.object(runtime_hand_q6_reader.serial, "Serial", return_value=serial_obj):
                reader = runtime_hand_q6_reader.SingleHandQ6Reader(
                    port="/dev/current-q6",
                    side="right",
                    calibration_path=path,
                )
                reader.close()

            payload = json.loads(path.read_text(encoding="utf-8"))
            current_entry = payload["port_calibrations"]["right"]["/dev/current-q6"]

        self.assertTrue(serial_obj.closed)
        self.assertEqual(reader.calib["deg_at_0"].tolist(), [200.0, 201.0, 202.0, 203.0, 204.0, 205.0])
        self.assertEqual(current_entry["port"], "/dev/current-q6")
        self.assertEqual(current_entry["deg_at_0"], [200.0, 201.0, 202.0, 203.0, 204.0, 205.0])
        self.assertEqual(current_entry["deg_at_1"], [240.0, 131.0, 262.0, 263.0, 264.0, 265.0])


if __name__ == "__main__":
    unittest.main()
