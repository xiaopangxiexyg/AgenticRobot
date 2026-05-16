from __future__ import annotations

import unittest

from inspire_data_collect.hand.serial_hand_q6_reader import (
    BufferedSerialLineReader,
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


if __name__ == "__main__":
    unittest.main()
