from __future__ import annotations

import json
import socket
import time

import numpy as np
import pytest

from hardware.VIVE.mocap.vive_data_sender import ClockSyncSenderClient
from teleop_runtime.sources.pose.vive import (
    DEFAULT_VIVE_CLOCK_SYNC_PORT,
    VivePoseSource,
    ViveTrackerConfig,
)


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


def _velocity_packet(
    *,
    velocity_valid: bool = True,
    linear_velocity: object = (0.1, -0.2, 0.3),
    angular_velocity: object = (-1.0, 2.0, -3.0),
) -> bytes:
    payload = json.loads(_packet(7, 0.7).decode("utf-8"))
    payload.update(
        {
            "pose_sample_timestamp": 1007.25,
            "pose_sample_monotonic_time": 12.5,
        }
    )
    payload["trackers"][0].update(
        {
            "linear_velocity_m_s": linear_velocity,
            "angular_velocity_rad_s": angular_velocity,
            "velocity_frame_name": "vive_world",
            "velocity_source": "openvr_tracked_device_pose",
            "velocity_valid": velocity_valid,
            "velocity_error": None,
        }
    )
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _dual_role_packet(seq: int = 1) -> bytes:
    payload = json.loads(_packet(seq, 0.25).decode("utf-8"))
    payload["trackers"].append(
        {
            "tracker_name": "synthetic-missing-right",
            "tracker_role": "right_wrist",
            "valid": False,
            "pose_matrix_4x4": np.eye(4).tolist(),
            "error": "single_tracker_mode:not_connected",
        }
    )
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _timed_packet(seq: int, x: float, sample_time: float) -> bytes:
    payload = json.loads(_packet(seq, x).decode("utf-8"))
    payload["pose_sample_monotonic_time"] = sample_time
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _timed_velocity_packet(seq: int, x: float, sample_time: float) -> bytes:
    payload = json.loads(_timed_packet(seq, x, sample_time).decode("utf-8"))
    payload["trackers"][0].update(
        {
            "linear_velocity_m_s": [0.01, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.05],
            "velocity_valid": True,
            "velocity_source": "openvr_tracked_device_pose",
        }
    )
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _source() -> VivePoseSource:
    return VivePoseSource(
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
        clock_sync_port=0,
    )


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def test_vive_pose_source_defaults_to_new_control_listener_and_reads_v1() -> None:
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
    )

    assert source.clock_sync_port == DEFAULT_VIVE_CLOCK_SYNC_PORT
    assert source.clock_sync_enabled is True
    source._process_received_batch([(_packet(1, 0.1), 1000.0, 20.0)])
    channel = source.get_latest_channels()["left_wrist_pose"]
    assert channel.valid is True
    assert channel.metadata["clock_sync_status"] == "unavailable"


def test_vive_source_role_swap_supports_single_logical_right_channel() -> None:
    source = VivePoseSource(
        tracker_configs=[
            ViveTrackerConfig(
                name="tracker_1",
                side="right",
                role="left_wrist",
                channel_name="right_wrist_pose",
                legacy_index=0,
            )
        ],
        host="127.0.0.1",
        port=0,
        clock_sync_port=0,
    )
    source._process_received_batch([(_packet(1, 0.1), 1000.0, 20.0)])
    assert source.get_latest_channels()["right_wrist_pose"].valid is True

    receipt = source.swap_left_right_source_roles()

    assert receipt["before_source_role_by_output_channel"] == {
        "right_wrist_pose": "left_wrist"
    }
    assert receipt["after_source_role_by_output_channel"] == {
        "right_wrist_pose": "right_wrist"
    }
    assert source.source_role_by_output_channel == {
        "right_wrist_pose": "right_wrist"
    }
    assert source.get_latest_channels() == {}


def test_upstream_role_diagnostics_survive_single_channel_mapping_and_swap() -> None:
    source = VivePoseSource(
        tracker_configs=[
            ViveTrackerConfig(
                name="tracker_1",
                side="right",
                role="left_wrist",
                channel_name="right_wrist_pose",
            )
        ],
        host="127.0.0.1",
        port=0,
        clock_sync_port=0,
    )

    source._process_received_batch(
        [(_dual_role_packet(9), 1000.0, 20.0, ("192.0.2.4", 5000))]
    )

    mapped = source.get_latest_channels()
    upstream = source.get_latest_upstream_role_channels()
    stats = source.debug_stats()
    assert mapped["right_wrist_pose"].metadata["tracker_role"] == "left_wrist"
    assert upstream["left_wrist"].valid is True
    assert upstream["left_wrist"].metadata["sender_host"] == "192.0.2.4"
    assert upstream["right_wrist"].valid is False
    assert (
        upstream["right_wrist"].metadata["sender_error"]
        == "single_tracker_mode:not_connected"
    )
    assert stats["latest_upstream_roles"] == ["left_wrist", "right_wrist"]
    assert stats["upstream_role_update_counts"] == {
        "left_wrist": 1,
        "right_wrist": 1,
    }

    source.swap_left_right_source_roles()

    # Mapping changes clear history but immediately expose the latest sender
    # snapshot under the new logical mapping.
    remapped = source.get_latest_channels()["right_wrist_pose"]
    assert remapped.metadata["tracker_role"] == "right_wrist"
    assert remapped.valid is False
    causal_remapped = source.get_latest_channels_at_or_before(1000.0)
    assert causal_remapped["right_wrist_pose"].valid is False
    assert causal_remapped["right_wrist_pose"].metadata[
        "source_mapping_generation"
    ] == 1
    upstream_after = source.get_latest_upstream_role_channels()
    assert upstream_after["left_wrist"].valid is True
    assert upstream_after["right_wrist"].valid is False

    source._process_received_batch(
        [(_dual_role_packet(10), 1000.1, 20.1, ("192.0.2.4", 5000))]
    )
    remapped = source.get_latest_channels()["right_wrist_pose"]
    assert remapped.metadata["tracker_role"] == "right_wrist"
    assert remapped.valid is False
    assert (
        remapped.metadata["sender_error"]
        == "single_tracker_mode:not_connected"
    )


def test_malformed_json_does_not_abort_batch_or_discard_last_valid_sample() -> None:
    source = _source()
    malformed = json.loads(_packet(2, 0.2).decode("utf-8"))
    malformed["sender_timestamp"] = "not-a-float"
    source._process_received_batch(
        [
            (_packet(1, 0.1), 1000.0, 20.0),
            (json.dumps(malformed).encode("utf-8"), 1000.01, 20.01),
            (_packet(3, 0.3), 1000.02, 20.02),
        ]
    )

    channel = source.get_latest_channels()["left_wrist_pose"]
    stats = source.debug_stats()
    assert channel.metadata["seq"] == 3
    assert stats["packet_count"] == 3
    assert stats["last_packet_received_monotonic_time"] == 20.02
    assert stats["last_process_finished_monotonic_time"] is not None
    assert stats["process_error_count"] == 1
    assert stats["last_process_parse_error_count"] == 1
    assert "could not convert string to float" in stats["last_process_error"]


@pytest.mark.integration
def test_vive_receive_worker_survives_malformed_datagram(
    require_udp_loopback: None,
) -> None:
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
        clock_sync_port=0,
    )
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    source.start()
    try:
        assert source.sock is not None
        address = source.sock.getsockname()
        malformed = json.loads(_packet(2, 0.2).decode("utf-8"))
        malformed["sender_timestamp"] = "not-a-float"
        sender.sendto(json.dumps(malformed).encode("utf-8"), address)

        deadline = time.time() + 1.0
        while time.time() < deadline:
            if source.debug_stats()["process_error_count"] >= 1:
                break
            time.sleep(0.01)
        stats = source.debug_stats()
        assert stats["worker_alive"] is True
        assert stats["process_error_count"] >= 1
        assert stats["datagram_count"] >= 1
        assert stats["last_datagram_received_monotonic_time"] is not None

        sender.sendto(_packet(3, 0.3), address)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            channel = source.get_latest_channels().get("left_wrist_pose")
            if channel is not None and channel.metadata.get("seq") == 3:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("receiver worker did not resume after malformed packet")
    finally:
        sender.close()
        source.close()


@pytest.mark.integration
def test_vive_receiver_default_does_not_share_udp_port(
    require_udp_loopback: None,
) -> None:
    first = VivePoseSource(
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
        clock_sync_port=0,
    )
    first.start()
    try:
        assert first.sock is not None
        port = int(first.sock.getsockname()[1])
        second = VivePoseSource(
            tracker_configs=[
                ViveTrackerConfig(
                    name="tracker_1",
                    side="left",
                    role="left_wrist",
                    channel_name="left_wrist_pose",
                )
            ],
            host="127.0.0.1",
            port=port,
            socket_timeout_sec=0.02,
            clock_sync_port=0,
        )
        try:
            with pytest.raises(OSError):
                second.start()
        finally:
            second.close()
    finally:
        first.close()


def test_declared_tracker_role_does_not_fall_back_to_matching_name() -> None:
    source = VivePoseSource(
        tracker_configs=[
            ViveTrackerConfig(
                name="tracker_1",
                side="right",
                role="right_wrist",
                channel_name="right_wrist_pose",
            )
        ],
        host="127.0.0.1",
        port=0,
        clock_sync_port=0,
    )

    # The name matches, but the packet explicitly declares left_wrist.
    assert source._parse_packet(_packet(1, 0.1), received_at=1000.0) == {}

    legacy_name_payload = json.loads(_packet(2, 0.2).decode("utf-8"))
    legacy_name_payload["trackers"][0].pop("tracker_role")
    parsed = source._parse_packet(
        json.dumps(legacy_name_payload).encode("utf-8"),
        received_at=1000.1,
    )
    assert parsed["right_wrist_pose"]["tracker_name"] == "tracker_1"


def test_vive_source_role_swap_exchanges_bilateral_sources_only() -> None:
    source = VivePoseSource(
        tracker_configs=[
            ViveTrackerConfig(
                name="left-tracker",
                side="left",
                role="left_wrist",
                channel_name="left_wrist_pose",
            ),
            ViveTrackerConfig(
                name="right-tracker",
                side="right",
                role="right_wrist",
                channel_name="right_wrist_pose",
            ),
            ViveTrackerConfig(
                name="aux",
                side=None,
                role="auxiliary",
                channel_name="auxiliary_tracker_pose",
                critical=False,
            ),
        ],
        host="127.0.0.1",
        port=0,
        clock_sync_port=0,
    )

    source.swap_left_right_source_roles()

    assert source.source_role_by_output_channel == {
        "left_wrist_pose": "right_wrist",
        "right_wrist_pose": "left_wrist",
        "auxiliary_tracker_pose": "auxiliary",
    }


def test_vive_pose_source_accepts_velocity_metadata() -> None:
    source = _source()

    parsed = source._parse_packet(_velocity_packet(), received_at=2000.0)
    source._latest_by_channel = parsed
    channel = source.get_latest_channels()["left_wrist_pose"]

    assert channel.valid is True
    assert channel.timestamp == 2000.0
    assert channel.metadata["velocity_valid"] is True
    assert channel.metadata["velocity_frame_name"] == "vive_world"
    assert channel.metadata["velocity_source"] == "openvr_tracked_device_pose"
    assert channel.metadata["linear_velocity_m_s"] == [0.1, -0.2, 0.3]
    assert channel.metadata["angular_velocity_rad_s"] == [-1.0, 2.0, -3.0]
    assert channel.metadata["pose_sample_timestamp"] == 1007.25
    assert channel.metadata["pose_sample_monotonic_time"] == 12.5


def test_vive_pose_source_rejects_bad_velocity_without_invalidating_pose() -> None:
    source = _source()

    parsed = source._parse_packet(
        _velocity_packet(linear_velocity=[1.0, 2.0]),
        received_at=2000.0,
    )
    payload = parsed["left_wrist_pose"]

    assert payload["valid"] is True
    assert payload["velocity_valid"] is False
    assert payload["linear_velocity_m_s"] is None
    assert payload["angular_velocity_rad_s"] is None
    assert "must have shape (3,)" in payload["velocity_error"]


def test_vive_pose_source_keeps_json_packets_without_velocity_compatible() -> None:
    source = _source()

    parsed = source._parse_packet(_packet(3, 0.3), received_at=2000.0)
    payload = parsed["left_wrist_pose"]

    assert payload["valid"] is True
    assert payload["velocity_valid"] is False
    assert payload["linear_velocity_m_s"] is None
    assert payload["angular_velocity_rad_s"] is None


@pytest.mark.integration
def test_vive_pose_source_drains_stale_udp_packets_to_latest(
    require_udp_loopback: None,
) -> None:
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

        batch = source._recv_packet_batch()
        source._process_received_batch(batch)
        channel = source.get_latest_channels()["left_wrist_pose"]
        history = source.get_pose_history("left_wrist_pose")

        assert len(batch) - 1 == 2
        assert channel.metadata["seq"] == 3
        assert [sample["seq"] for sample in history] == [1, 2, 3]
        np.testing.assert_allclose(
            channel.value[:3, 3],
            [0.3, 0.0, 0.0],
        )
    finally:
        sender.close()
        source.sock.close()


@pytest.mark.integration
def test_vive_pose_source_limits_stale_udp_drain_batch(
    require_udp_loopback: None,
) -> None:
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


def test_vive_pose_source_retains_drained_packets_in_time_history() -> None:
    source = _source()
    batch = [
        (_timed_packet(1, 0.1, 10.000), 1000.000, 20.000),
        (_timed_packet(2, 0.2, 10.004), 1000.004, 20.004),
        (_timed_packet(3, 0.3, 10.008), 1000.008, 20.008),
    ]

    source._process_received_batch(batch)

    channel = source.get_latest_channels()["left_wrist_pose"]
    history = source.get_pose_history("left_wrist_pose")
    stats = source.debug_stats()
    assert channel.metadata["seq"] == 3
    assert [sample["seq"] for sample in history] == [1, 2, 3]
    assert [sample["history_time"] for sample in history] == [20.000, 20.004, 20.008]
    assert [sample["availability_monotonic_time"] for sample in history] == [
        20.000,
        20.004,
        20.008,
    ]
    assert stats["packet_count"] == 3
    assert stats["last_drain_count"] == 2
    assert stats["channel_update_counts"]["left_wrist_pose"] == 1
    assert stats["history_sample_counts"]["left_wrist_pose"] == 3


def test_vive_pose_source_publishes_time_based_motion_window() -> None:
    source = _source()
    batch = [
        (
            _timed_velocity_packet(seq, 0.0001 * seq, 10.0 + 0.01 * seq),
            1000.0 + 0.01 * seq,
            20.0 + 0.01 * seq,
        )
        for seq in range(11)
    ]

    source._process_received_batch(batch)

    motion_window = source.get_latest_channels()["left_wrist_pose"].metadata[
        "tracker_motion_window"
    ]
    assert motion_window["quality_valid"] is True
    assert motion_window["sample_count"] == 11
    assert np.isclose(motion_window["window_span_sec"], 0.1)
    assert np.isclose(motion_window["linear_speed_p90_m_s"], 0.01)
    assert np.isclose(motion_window["angular_speed_p90_rad_s"], 0.05)


def test_vive_pose_source_history_retention_and_query_are_time_based() -> None:
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
        history_window_sec=0.10,
    )
    source._process_received_batch(
        [
            (_timed_packet(1, 0.1, 10.00), 1000.00, 20.00),
            (_timed_packet(2, 0.2, 10.04), 1000.04, 20.04),
            (_timed_packet(3, 0.3, 10.11), 1000.11, 20.11),
        ]
    )

    retained = source.get_pose_history("left_wrist_pose")
    last_50ms = source.get_pose_history("left_wrist_pose", window_sec=0.05)

    assert [sample["seq"] for sample in retained] == [2, 3]
    assert [sample["seq"] for sample in last_50ms] == [3]
    np.testing.assert_allclose(
        retained[-1]["history_time"] - retained[0]["history_time"],
        0.07,
    )


def test_vive_pose_source_snapshot_at_or_before_uses_receiver_availability() -> None:
    source = _source()
    source._process_received_batch(
        [
            (_timed_packet(1, 0.1, 1000.0), 100.00, 20.00),
            (_timed_packet(2, 0.2, 1.0), 100.01, 20.01),
            (_timed_packet(3, 0.3, 0.0), 100.02, 20.02),
        ]
    )

    before_second = source.get_pose_snapshot_at_or_before(
        "left_wrist_pose",
        20.01,
    )
    before_second_boundary = source.get_pose_snapshot_at_or_before(
        "left_wrist_pose",
        20.009,
    )

    assert before_second is not None
    assert before_second_boundary is not None
    assert before_second["seq"] == 2
    assert before_second_boundary["seq"] == 1
    before_second["pose"][0, 3] = 999.0
    frozen_again = source.get_pose_snapshot_at_or_before("left_wrist_pose", 20.01)
    assert frozen_again is not None
    assert frozen_again["pose"][0, 3] == 0.2


@pytest.mark.integration
def test_vive_pose_source_exposes_sync_capture_estimate_but_not_for_t0_query(
    require_udp_loopback: None,
) -> None:
    control_port = _free_udp_port()
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
        clock_sync_bind_host="127.0.0.1",
        clock_sync_port=control_port,
        clock_sync_min_samples=1,
        clock_sync_max_round_trip_sec=1.0,
        clock_sync_expected_sender_host="127.0.0.1",
    )
    sender = ClockSyncSenderClient(
        host="127.0.0.1",
        port=control_port,
        startup_probes=1,
        min_samples=1,
        timeout_sec=0.2,
        max_rtt_sec=1.0,
        startup_interval_sec=0.0,
        refresh_sec=10.0,
    )
    source.start()
    try:
        sender.start()
        assert sender.wait_until_ready(1.0)
        assert source.wait_for_clock_sync(1.0, min_samples=1)

        sender_capture_time = time.perf_counter()
        receiver_availability_time = time.perf_counter()
        source._process_received_batch(
            [
                (
                    _timed_packet(1, 0.1, sender_capture_time),
                    1000.0,
                    receiver_availability_time,
                    ("127.0.0.1", 34567),
                )
            ]
        )
        channel = source.get_latest_channels()["left_wrist_pose"]
        assert channel.metadata["clock_sync_status"] == "synchronized"
        assert channel.metadata["capture_receiver_monotonic_estimate"] is not None
        # The mapped capture time is diagnostic.  A made-up T0 in the receiver
        # domain still selects by availability rather than this value.
        assert (
            source.get_pose_snapshot_at_or_before(
                "left_wrist_pose",
                receiver_availability_time - 0.001,
            )
            is None
        )
        selected = source.get_pose_snapshot_at_or_before(
            "left_wrist_pose",
            receiver_availability_time,
        )
        assert selected is not None
        assert selected["seq"] == 1
    finally:
        sender.close()
        source.close()


def test_vive_pose_source_uses_receiver_availability_for_coarse_sender_clock() -> None:
    source = _source()
    source._process_received_batch(
        [
            (_timed_packet(1, 0.1, 10.000), 1000.000, 20.000),
            (_timed_packet(2, 0.2, 10.000), 1000.004, 20.004),
            (_timed_packet(3, 0.3, 10.016), 1000.008, 20.008),
        ]
    )

    history = source.get_pose_history("left_wrist_pose")
    stats = source.debug_stats()
    assert [sample["seq"] for sample in history] == [1, 2, 3]
    assert all(
        sample["history_clock_source"] == "receiver_monotonic_time"
        for sample in history
    )
    assert stats["history_reset_count"] == 0
    assert stats["history_last_reset_reason"] is None


def test_vive_pose_source_clears_time_history_on_sequence_reset() -> None:
    source = _source()
    source._process_received_batch(
        [
            (_timed_packet(10, 1.0, 30.000), 1000.000, 40.000),
            (_timed_packet(11, 1.1, 30.004), 1000.004, 40.004),
            (_timed_packet(0, 0.0, 1.000), 1000.008, 40.008),
        ]
    )

    history = source.get_pose_history("left_wrist_pose")
    stats = source.debug_stats()
    assert [sample["seq"] for sample in history] == [0]
    assert stats["history_reset_count"] == 1
    assert stats["history_last_reset_reason"] == "sender_session_restart"


def test_vive_pose_source_ignores_duplicate_and_out_of_order_history_packets() -> None:
    source = _source()
    source._process_received_batch(
        [
            (_timed_packet(10, 1.0, 30.000), 1000.000, 40.000),
            (_timed_packet(10, 9.0, 30.001), 1000.001, 40.001),
            (_timed_packet(9, 9.0, 30.002), 1000.002, 40.002),
            (_timed_packet(11, 1.1, 30.004), 1000.004, 40.004),
        ]
    )

    history = source.get_pose_history("left_wrist_pose")
    stats = source.debug_stats()
    assert [sample["seq"] for sample in history] == [10, 11]
    assert stats["history_duplicate_packet_count"] == 1
    assert stats["history_out_of_order_packet_count"] == 1
    assert stats["history_reset_count"] == 0


@pytest.mark.integration
def test_vive_pose_source_restarts_socket_after_recv_error(
    require_udp_loopback: None,
) -> None:
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
        clock_sync_port=0,
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
