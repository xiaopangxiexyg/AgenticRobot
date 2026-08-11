from __future__ import annotations

import re

from hardware.angle_trigger.exo_driver import (
    ExoDriver,
    ExoDriverCfg,
    send_command,
)


class _DelayedSerial:
    def __init__(self, responses: list[bytes]) -> None:
        self._responses = list(responses)
        self.writes: list[bytes] = []
        self.reset_count = 0
        self.flush_count = 0
        self.read_count = 0

    def reset_input_buffer(self) -> None:
        self.reset_count += 1

    def write(self, payload: bytes) -> None:
        self.writes.append(bytes(payload))

    def flush(self) -> None:
        self.flush_count += 1

    def read_all(self) -> bytes:
        self.read_count += 1
        return self._responses.pop(0) if self._responses else b""


def test_response_query_waits_for_delayed_complete_frame() -> None:
    serial = _DelayedSerial([b"", b"#001P", b"1500!"])

    response = send_command(
        serial,
        "#001PRAD!",
        delay_s=0.0001,
        clear_input=True,
        response_pattern=re.compile(r"#001P\d{4}"),
        timeout_s=0.05,
        poll_interval_s=0.0001,
    )

    assert response == "#001P1500!"
    assert serial.writes == [b"#001PRAD!"]
    assert serial.reset_count == 1
    assert serial.flush_count == 1
    assert serial.read_count == 3


def test_read_position_waits_for_requested_servo_and_ignores_other_id() -> None:
    serial = _DelayedSerial([b"#002P1200!", b"", b"#001P1600!"])
    driver = ExoDriver(
        ExoDriverCfg(
            servo_ids=[1],
            timeout_s=0.05,
            command_delay_s=0.0001,
            read_hz=30.0,
        )
    )
    driver.ser = serial

    reading = driver.read_position(1)

    assert reading.ok
    assert reading.raw_pos == 1600
    assert reading.angle_deg == 148.5
    assert serial.read_count == 3


def test_response_query_returns_bounded_partial_payload_on_timeout() -> None:
    serial = _DelayedSerial([b"#001P"])

    response = send_command(
        serial,
        "#001PRAD!",
        delay_s=0.0001,
        response_pattern=re.compile(r"#001P\d{4}"),
        timeout_s=0.002,
        poll_interval_s=0.0001,
    )

    assert response == "#001P"
    assert serial.read_count >= 2
