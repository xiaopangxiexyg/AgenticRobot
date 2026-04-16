from __future__ import annotations

from typing import Any

import numpy as np

from ...interfaces import ObservationSource
from ...schemas import ObservationChannel


class HandQ6Source(ObservationSource):
    def __init__(self, reader: Any, *, name: str = "hand_q6_serial"):
        self.reader = reader
        self._name = name
        self._expected: tuple[str, ...] | None = None

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_single_hand(
        cls,
        *,
        port: str,
        side: str,
        baudrate: int = 115200,
        timeout: float = 0.05,
        show_raw_lines: bool = False,
    ) -> "HandQ6Source":
        from hardware.teleop_hand.hand_q6_reader import SingleHandQ6Reader

        reader = SingleHandQ6Reader(
            port=port,
            side=side,
            baudrate=baudrate,
            timeout=timeout,
            raw_only=False,
            show_raw_lines=show_raw_lines,
        )
        source = cls(reader=reader, name=f"hand_q6_{side}")
        source._expected = (f"{side}_hand_q6",)
        return source

    @classmethod
    def from_bimanual(
        cls,
        *,
        left_port: str | None = None,
        right_port: str | None = None,
        baudrate: int = 115200,
        timeout: float = 0.05,
        show_raw_lines: bool = False,
    ) -> "HandQ6Source":
        from hardware.teleop_hand.hand_q6_reader import BimanualHandQ6Reader

        reader = BimanualHandQ6Reader(
            left_port=left_port,
            right_port=right_port,
            baudrate=baudrate,
            timeout=timeout,
            raw_only=False,
            show_raw_lines=show_raw_lines,
        )
        source = cls(reader=reader, name="hand_q6_bimanual")
        expected: list[str] = []
        if left_port is not None:
            expected.append("left_hand_q6")
        if right_port is not None:
            expected.append("right_hand_q6")
        source._expected = tuple(expected)
        return source

    def expected_channel_names(self) -> tuple[str, ...]:
        return self._expected or ()

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        packet = self.reader.read_frame()
        timestamp = float(np.asarray(packet.get("timestamp", [0.0]), dtype=float).ravel()[0])
        channels: dict[str, ObservationChannel] = {}

        if "norm" in packet:
            side = str(packet["side"])
            channel_name = f"{side}_hand_q6"
            channels[channel_name] = ObservationChannel(
                channel_name=channel_name,
                capability="joint_vector",
                side=side,
                value=np.asarray(packet["norm"], dtype=float).copy(),
                frame_name=None,
                timestamp=timestamp,
                source_name=self.name,
                valid=True,
                critical=True,
                metadata={"port": packet.get("port")},
            )
            return channels

        for side in ("left", "right"):
            side_packet = packet.get(side)
            if not isinstance(side_packet, dict) or "norm" not in side_packet:
                continue
            channel_name = f"{side}_hand_q6"
            side_timestamp = float(
                np.asarray(side_packet.get("timestamp", [timestamp]), dtype=float).ravel()[0]
            )
            channels[channel_name] = ObservationChannel(
                channel_name=channel_name,
                capability="joint_vector",
                side=side,
                value=np.asarray(side_packet["norm"], dtype=float).copy(),
                frame_name=None,
                timestamp=side_timestamp,
                source_name=self.name,
                valid=True,
                critical=True,
                metadata={"port": side_packet.get("port")},
            )
        return channels

    def close(self) -> None:
        self.reader.close()
