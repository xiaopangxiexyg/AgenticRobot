from __future__ import annotations

import unittest

from inspire_data_collect.hand.serial_hand_q6_reader import (
    BufferedSerialLineReader,
    build_hand_packet,
    calibrate_runtime_zero,
    map_angles_to_norm,
    parse_angles_line,
)


def _angles_line(base: int) -> bytes:
    values = [base + offset for offset in range(6)]
    body = ", ".join(f"a{index}={value}" for index, value in enumerate(values))
    return f"ANGLES: {body}\n".encode("utf-8")


class _FakeSerial:
    def __init__(self, payload: bytes = b"") -> None:
        self._buffer = bytearray(payload)

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


class _FakeParsedReader:
    def __init__(self, frames: list[list[float]]) -> None:
        self._frames = list(frames)

    def read_latest_parsed(self, _parse_line):
        if not self._frames:
            raise RuntimeError("no more frames")
        return self._frames.pop(0)


def _assert_list_close(testcase: unittest.TestCase, actual, expected) -> None:
    testcase.assertEqual(len(actual), len(expected))
    for left, right in zip(actual, expected):
        testcase.assertAlmostEqual(float(left), float(right), places=6)


class BufferedSerialLineReaderTests(unittest.TestCase):
    def test_returns_latest_complete_angles_frame_from_backlog(self) -> None:
        serial_obj = _FakeSerial(
            _angles_line(0)
            + b"garbage\n"
            + _angles_line(10)
            + _angles_line(20)
        )
        reader = BufferedSerialLineReader(serial_obj)

        angles = reader.read_latest_parsed(parse_angles_line)

        self.assertEqual(angles, [20.0, 21.0, 22.0, 23.0, 24.0, 25.0])

    def test_keeps_partial_line_until_it_is_complete(self) -> None:
        serial_obj = _FakeSerial(_angles_line(0) + b"ANGLES: a0=10, a1=11")
        reader = BufferedSerialLineReader(serial_obj)

        first = reader.read_latest_parsed(parse_angles_line)
        self.assertEqual(first, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        serial_obj.append(b", a2=12, a3=13, a4=14, a5=15\n")
        second = reader.read_latest_parsed(parse_angles_line)
        self.assertEqual(second, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])


class HandQ6NormalizationTests(unittest.TestCase):
    def test_absolute_delta_runtime_zero_keeps_reverse_motion_nonzero(self) -> None:
        deg0 = [100.0] * 6
        deg1 = [140.0, 170.0, 160.0, 160.0, 160.0, 160.0]
        signed = map_angles_to_norm([70.0] * 6, deg0, deg1)
        absolute = map_angles_to_norm([70.0] * 6, deg0, deg1, absolute_delta=True)

        self.assertEqual(signed.tolist(), [0.0] * 6)
        _assert_list_close(self, absolute.tolist(), [0.75, 0.4285714328289032, 0.5, 0.5, 0.5, 0.5])

    def test_packet_records_runtime_zero_absolute_delta_metadata(self) -> None:
        packet = build_hand_packet(
            angles_deg=[70.0] * 6,
            side="right",
            port="/dev/test",
            baudrate=115200,
            calib={
                "deg_at_0": [100.0] * 6,
                "deg_at_1": [140.0, 170.0, 160.0, 160.0, 160.0, 160.0],
                "absolute_delta": True,
                "runtime_zero_on_start": True,
                "zero_scale": "absolute",
            },
        )

        self.assertTrue(packet["absolute_delta"])
        self.assertTrue(packet["runtime_zero_on_start"])
        self.assertEqual(packet["zero_scale"], "absolute")
        _assert_list_close(self, packet["norm"].tolist(), [0.75, 0.4285714328289032, 0.5, 0.5, 0.5, 0.5])

    def test_runtime_zero_calibration_uses_latest_stable_angles(self) -> None:
        reader = _FakeParsedReader([[100.0 + offset for offset in range(6)] for _ in range(10)])

        calib = calibrate_runtime_zero(
            reader,
            side="right",
            min_samples=3,
            stable_window=3,
            max_samples=10,
            zero_scale="absolute",
        )

        self.assertTrue(calib["absolute_delta"])
        self.assertTrue(calib["runtime_zero_on_start"])
        self.assertEqual(calib["zero_scale"], "absolute")
        self.assertEqual(calib["deg_at_0"].tolist(), [100.0, 101.0, 102.0, 103.0, 104.0, 105.0])


if __name__ == "__main__":
    unittest.main()
