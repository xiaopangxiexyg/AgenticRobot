from __future__ import annotations

import math
import socket
import sys
from pathlib import Path

import pytest

from hardware.VIVE.mocap import vive_data_sender, vive_data_sender_legacy
from teleop_runtime.sources.pose.vive_clock_sync import (
    ViveClockSyncConfig,
    ViveClockSyncServer,
)


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


class _OpenVRVector:
    def __init__(self, *values: float) -> None:
        self.v = values


class _VelocityDevice:
    def __init__(self, linear: object, angular: object) -> None:
        self.linear = linear
        self.angular = angular

    def get_velocity(self, pose_snapshot: object) -> object:
        assert pose_snapshot == "snapshot"
        return self.linear

    def get_angular_velocity(self, pose_snapshot: object) -> object:
        assert pose_snapshot == "snapshot"
        return self.angular


def test_vive_data_sender_defaults_to_300hz(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["vive_data_sender.py"])

    args = vive_data_sender.parse_args()

    assert args.hz == 300.0
    assert args.clock_sync_mode == "optional"
    assert args.clock_sync_port == 10001
    assert args.clock_sync_port != args.port


def test_legacy_sender_entrypoint_pins_clock_control_off(monkeypatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_main(argv: object, **kwargs: object) -> None:
        calls.append((argv, kwargs))

    monkeypatch.setattr(vive_data_sender_legacy._sender, "main", fake_main)

    vive_data_sender_legacy.main(["--clock-sync-mode", "required"])

    assert calls == [
        (
            ["--clock-sync-mode", "required"],
            {
                "clock_sync_mode_default": "off",
                "send_hz_default": 60.0,
                "force_clock_sync_mode": "off",
                "legacy_wire_format": True,
            },
        )
    ]


def test_v1_tracking_packet_has_no_clock_control_fields() -> None:
    trackers = [{"tracker_name": "tracker_1", "valid": True}]

    packet = vive_data_sender.build_v1_tracking_packet(
        seq=3,
        sender_timestamp=100.0,
        pose_sample_timestamp=100.1,
        pose_sample_monotonic_time=50.0,
        trackers=trackers,
    )

    assert list(packet) == [
        "protocol",
        "seq",
        "sender_timestamp",
        "pose_sample_timestamp",
        "pose_sample_monotonic_time",
        "trackers",
    ]
    assert packet["protocol"] == "teleop_runtime.vive/v1"
    assert packet["trackers"] is trackers
    assert "clock_sync" not in packet


def test_legacy_v1_tracking_packet_freezes_release_era_field_shape() -> None:
    packet = vive_data_sender.build_legacy_v1_tracking_packet(
        seq=3,
        sender_timestamp=100.0,
        trackers=[
            {
                "tracker_name": "tracker_1",
                "tracker_role": "left_wrist",
                "valid": True,
                "pose_matrix_4x4": [[1.0] * 4] * 4,
                "error": None,
                "read_elapsed_ms": 0.2,
                "velocity_valid": True,
                "linear_velocity_m_s": [1.0, 2.0, 3.0],
            }
        ],
    )

    assert list(packet) == ["protocol", "seq", "sender_timestamp", "trackers"]
    assert list(packet["trackers"][0]) == [
        "tracker_name",
        "tracker_role",
        "valid",
        "pose_matrix_4x4",
        "error",
        "read_elapsed_ms",
    ]
    assert "velocity_valid" not in packet["trackers"][0]
    assert vive_data_sender.parse_args([], send_hz_default=60.0).hz == 60.0


def test_sender_rejects_control_port_equal_to_legacy_pose_port() -> None:
    args = type(
        "Args",
        (),
        {
            "clock_sync_mode": "optional",
            "clock_sync_port": 9999,
            "port": 9999,
        },
    )()

    with pytest.raises(SystemExit, match="must differ from --port"):
        vive_data_sender.start_clock_sync_client(args)


def test_missing_clock_sync_helper_does_not_break_legacy_off_mode(monkeypatch) -> None:
    args = type("Args", (), {"clock_sync_mode": "off"})()
    monkeypatch.setattr(
        vive_data_sender,
        "_CLOCK_SYNC_IMPORT_ERROR",
        ImportError("clock_sync.py missing"),
    )

    assert vive_data_sender.start_clock_sync_client(args) is None


def test_missing_clock_sync_helper_fails_open_for_new_optional_mode(monkeypatch) -> None:
    args = type("Args", (), {"clock_sync_mode": "optional"})()
    monkeypatch.setattr(
        vive_data_sender,
        "_CLOCK_SYNC_IMPORT_ERROR",
        ImportError("clock_sync.py missing"),
    )

    assert vive_data_sender.start_clock_sync_client(args) is None


def test_sender_uses_only_canonical_triad_openvr_package_file() -> None:
    module_path = vive_data_sender.canonical_triad_openvr_module_path()

    assert module_path == (
        Path(vive_data_sender.__file__).resolve().parents[1]
        / "triad_openvr"
        / "triad_openvr.py"
    )
    assert module_path.is_file()
    assert module_path.parent.name == "triad_openvr"


def test_load_triad_openvr_supports_explicit_canonical_file(tmp_path) -> None:
    module_path = tmp_path / "triad_openvr.py"
    module_path.write_text("class triad_openvr:\n    pass\n", encoding="utf-8")

    module = vive_data_sender.load_triad_openvr(module_path)

    assert module.__file__ == str(module_path)
    assert module.triad_openvr.__name__ == "triad_openvr"


def test_pose_sample_clock_uses_high_resolution_perf_counter(monkeypatch) -> None:
    monkeypatch.setattr(vive_data_sender.time, "time", lambda: 123.5)
    monkeypatch.setattr(vive_data_sender.time, "perf_counter", lambda: 456.75)

    def fail_low_resolution_clock() -> float:
        raise AssertionError("pose sampling must not use time.monotonic()")

    monkeypatch.setattr(
        vive_data_sender.time,
        "monotonic",
        fail_low_resolution_clock,
    )

    assert vive_data_sender.pose_sample_clocks() == (123.5, 456.75)


def test_openvr_velocity_payload_uses_pose_snapshot_velocity() -> None:
    payload = vive_data_sender.openvr_velocity_payload(
        _VelocityDevice(
            _OpenVRVector(0.1, -0.2, 0.3),
            _OpenVRVector(-1.0, 2.0, -3.0),
        ),
        "snapshot",
    )

    assert payload == {
        "linear_velocity_m_s": [0.1, -0.2, 0.3],
        "angular_velocity_rad_s": [-1.0, 2.0, -3.0],
        "velocity_frame_name": "vive_world",
        "velocity_source": "openvr_tracked_device_pose",
        "velocity_valid": True,
        "velocity_error": None,
    }


def test_openvr_velocity_payload_does_not_invalidate_pose_on_bad_velocity() -> None:
    payload = vive_data_sender.openvr_velocity_payload(
        _VelocityDevice(
            _OpenVRVector(math.nan, 0.0, 0.0),
            _OpenVRVector(0.0, 0.0, 0.0),
        ),
        "snapshot",
    )

    assert payload["velocity_valid"] is False
    assert payload["linear_velocity_m_s"] is None
    assert payload["angular_velocity_rad_s"] is None
    assert "non-finite" in str(payload["velocity_error"])


def test_openvr_velocity_payload_supports_direct_tracked_pose_fallback() -> None:
    class DirectPose:
        vVelocity = [0.4, 0.5, 0.6]
        vAngularVelocity = [-0.4, -0.5, -0.6]

    class IndexedDevice:
        index = 1

    payload = vive_data_sender.openvr_velocity_payload(
        IndexedDevice(),
        [object(), DirectPose()],
    )

    assert payload["velocity_valid"] is True
    assert payload["linear_velocity_m_s"] == [0.4, 0.5, 0.6]
    assert payload["angular_velocity_rad_s"] == [-0.4, -0.5, -0.6]


class _FakeTracker:
    device_class = "Tracker"


class _FakeViveTrackers:
    def __init__(self, names: list[str]) -> None:
        self.devices = {name: _FakeTracker() for name in names}
        self.object_names = {"Tracker": list(names)}


def test_vive_data_sender_assigns_trackers_by_lexical_device_name() -> None:
    vive = _FakeViveTrackers(["tracker_z", "tracker_a"])

    specs = vive_data_sender.auto_tracker_specs(vive)

    assert [(spec.device_name, spec.tracker_role) for spec in specs] == [
        ("tracker_a", "left_wrist"),
        ("tracker_z", "right_wrist"),
    ]


@pytest.mark.parametrize("names", [[], ["tracker_a"], ["tracker_a", "tracker_b", "tracker_c"]])
def test_vive_data_sender_requires_exactly_two_trackers(names: list[str]) -> None:
    with pytest.raises(ValueError, match="exactly 2 GenericTrackers"):
        vive_data_sender.auto_tracker_specs(_FakeViveTrackers(names))


def test_send_error_backoff_is_bounded() -> None:
    assert vive_data_sender.send_error_backoff_seconds(0, 0.25, 2.0) == 0.0
    assert vive_data_sender.send_error_backoff_seconds(1, 0.25, 2.0) == 0.25
    assert vive_data_sender.send_error_backoff_seconds(4, 0.25, 2.0) == 2.0
    assert vive_data_sender.send_error_backoff_seconds(20, 0.25, 2.0) == 2.0


@pytest.mark.integration
def test_sender_clock_sync_client_uses_separate_control_port(
    require_udp_loopback: None,
) -> None:
    control_port = _free_udp_port()
    server = ViveClockSyncServer(
        ViveClockSyncConfig(
            bind_host="127.0.0.1",
            port=control_port,
            min_samples=1,
            max_round_trip_sec=1.0,
            expected_sender_host="127.0.0.1",
        )
    )
    server.start()
    client = vive_data_sender.ClockSyncSenderClient(
        host="127.0.0.1",
        port=control_port,
        startup_probes=1,
        min_samples=1,
        timeout_sec=0.2,
        max_rtt_sec=1.0,
        startup_interval_sec=0.0,
        refresh_sec=10.0,
    )
    try:
        client.start()
        assert client.wait_until_ready(1.0)
        assert server.wait_until_ready(1.0, min_samples=1)
        stats = client.debug_stats()
        assert stats["accepted_count"] >= 1
        assert stats["estimate"] is not None
    finally:
        client.close()
        server.close()


class _FakeViveRegistry:
    def __init__(self) -> None:
        self.devices = {}
        self.poll_count = 0
        self.refresh_count = 0

    def poll_vr_events(self) -> int:
        self.poll_count += 1
        return 0

    def refresh_discovered_devices(self) -> int:
        self.refresh_count += 1
        self.devices["tracker_1"] = object()
        return 1


def test_update_vive_device_registry_refreshes_missing_tracker() -> None:
    vive = _FakeViveRegistry()

    changes, error = vive_data_sender.update_vive_device_registry(vive, refresh=True)

    assert error is None
    assert changes == 1
    assert vive.poll_count == 1
    assert vive.refresh_count == 1
    assert "tracker_1" in vive.devices


def test_update_vive_device_registry_reports_poll_errors() -> None:
    class BrokenVive:
        def poll_vr_events(self) -> int:
            raise RuntimeError("openvr unavailable")

    changes, error = vive_data_sender.update_vive_device_registry(
        BrokenVive(),
        refresh=True,
    )

    assert changes == 0
    assert error == "RuntimeError: openvr unavailable"
