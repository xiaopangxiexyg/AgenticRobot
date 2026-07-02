from __future__ import annotations

import json
import socket
import time

import numpy as np

from teleop_runtime.sources.pose.vive import VivePoseSource, ViveTrackerConfig


def _packet(seq: int, x: float) -> bytes:
    return json.dumps(
        {
            "protocol": "teleop_runtime.vive/v1",
            "seq": seq,
            "sender_timestamp": 1000.0 + seq,
            "trackers": [
                {
                    "tracker_name": "tracker_1",
                    "tracker_role": "left_wrist",
                    "valid": True,
                    "pose_matrix_4x4": [
                        [1.0, 0.0, 0.0, x],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "error": None,
                }
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")


def test_vive_pose_source_drains_stale_udp_packets_to_latest() -> None:
    source = VivePoseSource(
        tracker_configs=[
            ViveTrackerConfig(
                name="tracker_1",
                side="left",
                role="left_wrist",
                channel_name="left_wrist_pose",
            )
        ],
        host="127.0.0.1",
        port=0,
        socket_timeout_sec=0.1,
    )
    source.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    source.sock.bind((source.host, 0))
    source.sock.settimeout(source.socket_timeout_sec)

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        address = source.sock.getsockname()
        sender.sendto(_packet(1, 0.1), address)
        sender.sendto(_packet(2, 0.2), address)
        sender.sendto(_packet(3, 0.3), address)

        received = source._recv_latest_packet()
        assert received is not None
        packet, received_at, stale_count = received
        parsed = source._parse_packet(packet, received_at)

        assert stale_count == 2
        assert parsed["left_wrist_pose"]["seq"] == 3
        np.testing.assert_allclose(
            parsed["left_wrist_pose"]["pose"][:3, 3],
            [0.3, 0.0, 0.0],
        )
    finally:
        sender.close()
        source.sock.close()


def test_vive_pose_source_limits_stale_udp_drain_batch() -> None:
    source = VivePoseSource(
        tracker_configs=[
            ViveTrackerConfig(
                name="tracker_1",
                side="left",
                role="left_wrist",
                channel_name="left_wrist_pose",
            )
        ],
        host="127.0.0.1",
        port=0,
        socket_timeout_sec=0.1,
        max_drain_packets=2,
    )
    source.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    source.sock.bind((source.host, 0))
    source.sock.settimeout(source.socket_timeout_sec)

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        address = source.sock.getsockname()
        for seq in range(1, 6):
            sender.sendto(_packet(seq, seq / 10.0), address)

        received = source._recv_latest_packet()
        assert received is not None
        packet, received_at, stale_count = received
        parsed = source._parse_packet(packet, received_at)

        assert stale_count == 2
        assert parsed["left_wrist_pose"]["seq"] == 3
        np.testing.assert_allclose(
            parsed["left_wrist_pose"]["pose"][:3, 3],
            [0.3, 0.0, 0.0],
        )
    finally:
        sender.close()
        source.sock.close()


def test_vive_pose_source_restarts_socket_after_recv_error() -> None:
    source = VivePoseSource(
        tracker_configs=[
            ViveTrackerConfig(
                name="tracker_1",
                side="left",
                role="left_wrist",
                channel_name="left_wrist_pose",
            )
        ],
        host="127.0.0.1",
        port=0,
        socket_timeout_sec=0.02,
    )
    source.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        assert source.sock is not None
        address = source.sock.getsockname()
        sender.sendto(_packet(1, 0.1), address)

        deadline = time.time() + 1.0
        while time.time() < deadline:
            channels = source.get_latest_channels()
            channel = channels.get("left_wrist_pose")
            if channel is not None and channel.metadata.get("seq") == 1:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("source did not receive initial packet")

        assert source.sock is not None
        source.sock.close()

        deadline = time.time() + 1.5
        while time.time() < deadline:
            stats = source.debug_stats()
            if (
                int(stats.get("socket_restart_count") or 0) >= 1
                and source.sock is not None
            ):
                address = source.sock.getsockname()
                sender.sendto(_packet(2, 0.2), address)
                break
            time.sleep(0.01)
        else:
            raise AssertionError("source did not restart receive socket")

        deadline = time.time() + 1.0
        while time.time() < deadline:
            channels = source.get_latest_channels()
            channel = channels.get("left_wrist_pose")
            if channel is not None and channel.metadata.get("seq") == 2:
                np.testing.assert_allclose(channel.value[:3, 3], [0.2, 0.0, 0.0])
                break
            time.sleep(0.01)
        else:
            raise AssertionError("source did not receive packet after socket restart")
    finally:
        sender.close()
        source.close()
