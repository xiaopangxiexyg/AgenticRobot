from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from hardware.VIVE.mocap import clock_sync as shared_clock_sync
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
    assert args.tracker_count == 2
    assert args.clock_sync_mode == "optional"
    assert args.clock_sync_port == 10001
    assert args.clock_sync_port != args.port
    assert args.allow_single_tracker is False


def test_vive_data_sender_single_tracker_compatibility_is_opt_in() -> None:
    args = vive_data_sender.parse_args(["--single"])

    assert args.allow_single_tracker is True


def test_embedded_clock_sync_wire_format_matches_shared_protocol() -> None:
    values = {
        "sender_instance_id": "sender-instance",
        "probe_id": 7,
        "t1_sender_monotonic_ns": 1_000,
    }
    assert vive_data_sender.make_probe(**values) == shared_clock_sync.make_probe(**values)

    reply_packet = shared_clock_sync.make_reply(
        **values,
        t2_collector_monotonic_ns=1_100,
        t3_collector_monotonic_ns=1_120,
    )
    reply = vive_data_sender.parse_reply(reply_packet)
    assert reply is not None
    assert vars(reply) == {
        **values,
        "t2_collector_monotonic_ns": 1_100,
        "t3_collector_monotonic_ns": 1_120,
    }

    result_values = {
        **vars(reply),
        "t4_sender_monotonic_ns": 1_250,
    }
    assert vive_data_sender.make_result(**result_values) == shared_clock_sync.make_result(
        **result_values
    )
    embedded_measurement = vive_data_sender.measurement_from_result(
        vive_data_sender.ClockSyncResult(**result_values)
    )
    shared_measurement = shared_clock_sync.measurement_from_result(
        shared_clock_sync.ClockSyncResult(**result_values)
    )
    assert embedded_measurement is not None
    assert shared_measurement is not None
    assert vars(embedded_measurement) == vars(shared_measurement)


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


def test_sender_preferred_triad_openvr_package_file_is_repository_copy() -> None:
    module_path = vive_data_sender.canonical_triad_openvr_module_path()

    assert module_path == (
        Path(vive_data_sender.__file__).resolve().parents[1]
        / "triad_openvr"
        / "triad_openvr.py"
    )
    assert module_path.is_file()
    assert module_path.parent.name == "triad_openvr"
    assert vive_data_sender.find_triad_openvr_module_path() == module_path


@pytest.mark.parametrize(
    "relative_driver_path",
    [
        Path("triad_openvr.py"),
        Path("triad_openvr") / "triad_openvr.py",
        Path("vendor") / "triad_openvr.py",
        Path("vendor") / "openvr_driver" / "triad_openvr.py",
        Path("..") / "triad_openvr" / "triad_openvr.py",
        Path("..") / "vendor" / "triad_openvr.py",
        Path("..") / "vendor" / "openvr_driver" / "triad_openvr.py",
    ],
)
def test_find_triad_openvr_supports_portable_relative_layouts(
    tmp_path,
    relative_driver_path: Path,
) -> None:
    sender_path = tmp_path / "bundle" / "mocap" / "vive_data_sender.py"
    sender_path.parent.mkdir(parents=True)
    sender_path.touch()
    driver_path = (sender_path.parent / relative_driver_path).resolve()
    driver_path.parent.mkdir(parents=True, exist_ok=True)
    driver_path.write_text("class triad_openvr:\n    pass\n", encoding="utf-8")

    assert vive_data_sender.find_triad_openvr_module_path(sender_path) == driver_path


def test_find_triad_openvr_prefers_sender_local_file(tmp_path) -> None:
    sender_path = tmp_path / "bundle" / "mocap" / "vive_data_sender.py"
    sender_path.parent.mkdir(parents=True)
    sender_path.touch()
    local_driver = sender_path.parent / "triad_openvr.py"
    sibling_driver = sender_path.parent.parent / "vendor" / "triad_openvr.py"
    local_driver.write_text("LOCAL = True\n", encoding="utf-8")
    sibling_driver.parent.mkdir()
    sibling_driver.write_text("LOCAL = False\n", encoding="utf-8")

    assert vive_data_sender.find_triad_openvr_module_path(sender_path) == local_driver


def test_find_triad_openvr_skips_hidden_repository_metadata(tmp_path) -> None:
    sender_path = tmp_path / "bundle" / "mocap" / "vive_data_sender.py"
    sender_path.parent.mkdir(parents=True)
    sender_path.touch()
    hidden_driver = sender_path.parent.parent / ".git" / "triad_openvr.py"
    hidden_driver.parent.mkdir()
    hidden_driver.write_text("SHOULD_NOT_LOAD = True\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="triad_openvr.py was not found"):
        vive_data_sender.find_triad_openvr_module_path(sender_path)


def test_load_triad_openvr_supports_explicit_canonical_file(tmp_path) -> None:
    module_path = tmp_path / "triad_openvr.py"
    module_path.write_text("class triad_openvr:\n    pass\n", encoding="utf-8")

    module = vive_data_sender.load_triad_openvr(module_path)

    assert module.__file__ == str(module_path)
    assert module.triad_openvr.__name__ == "triad_openvr"


def test_load_triad_openvr_searches_relative_to_sender(monkeypatch, tmp_path) -> None:
    sender_path = tmp_path / "bundle" / "mocap" / "vive_data_sender.py"
    sender_path.parent.mkdir(parents=True)
    sender_path.touch()
    module_path = (
        sender_path.parent.parent
        / "portable_driver"
        / "lib"
        / "triad_openvr.py"
    )
    module_path.parent.mkdir(parents=True)
    module_path.write_text("class triad_openvr:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(vive_data_sender, "__file__", str(sender_path))

    module = vive_data_sender.load_triad_openvr()

    assert module.__file__ == str(module_path)
    assert module.triad_openvr.__name__ == "triad_openvr"


def test_load_triad_openvr_falls_back_to_embedded_for_one_file_copy(
    monkeypatch,
    tmp_path,
) -> None:
    sender_path = tmp_path / "portable" / "vive_data_sender.py"
    sender_path.parent.mkdir()
    sender_path.touch()
    monkeypatch.setattr(vive_data_sender, "__file__", str(sender_path))

    module = vive_data_sender.load_triad_openvr()

    assert module.triad_openvr is vive_data_sender.EmbeddedTriadOpenVR


def test_one_file_sender_runs_with_only_openvr_dependency(tmp_path) -> None:
    sender_dir = tmp_path / "isolated_bundle" / "sender_host"
    sender_dir.mkdir(parents=True)
    sender_copy = sender_dir / "vive_data_sender.py"
    sender_copy.write_bytes(Path(vive_data_sender.__file__).read_bytes())
    (sender_dir / "openvr.py").write_text(
        """
VRApplication_Other = 3
TrackingUniverseStanding = 1
k_unMaxTrackedDeviceCount = 2
TrackedDeviceClass_Controller = 2
TrackedDeviceClass_HMD = 1
TrackedDeviceClass_GenericTracker = 3
TrackedDeviceClass_TrackingReference = 4
Prop_SerialNumber_String = 1000
Prop_ModelNumber_String = 1001
VREvent_TrackedDeviceActivated = 100
VREvent_TrackedDeviceDeactivated = 101

class _Pose:
    bDeviceIsConnected = True
    bPoseIsValid = True
    mDeviceToAbsoluteTracking = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    vVelocity = (0.0, 0.0, 0.0)
    vAngularVelocity = (0.0, 0.0, 0.0)

class _System:
    def getDeviceToAbsoluteTrackingPose(self, universe, prediction, count):
        return [_Pose(), _Pose()]
    def getTrackedDeviceClass(self, index):
        return TrackedDeviceClass_GenericTracker
    def getStringTrackedDeviceProperty(self, index, prop):
        return ("SERIAL-" if prop == Prop_SerialNumber_String else "MODEL-") + str(index)
    def pollNextEvent(self, event):
        return False

class VREvent_t:
    eventType = 0
    trackedDeviceIndex = 0

_SYSTEM = _System()
def init(application):
    return _SYSTEM
def VRSystem():
    return _SYSTEM
def shutdown():
    pass
""".lstrip(),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-S", str(sender_copy), "--list-devices"],
        cwd=sender_dir,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert "using embedded dependency-light OpenVR adapter" in completed.stdout
    assert "Found 2 Trackers" in completed.stdout
    assert "tracker_1 (SERIAL-0, MODEL-0)" in completed.stdout
    assert "tracker_2 (SERIAL-1, MODEL-1)" in completed.stdout


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


def test_single_tracker_maps_left_and_synthesizes_invalid_right() -> None:
    specs = vive_data_sender.auto_tracker_specs(
        _FakeViveTrackers(["tracker_only"]),
        allow_single_tracker=True,
    )

    assert [(spec.device_name, spec.tracker_role) for spec in specs] == [
        ("tracker_only", "left_wrist"),
        (vive_data_sender.SINGLE_TRACKER_COMPAT_RIGHT_NAME, "right_wrist"),
    ]


def test_vive_data_sender_single_tracker_wait_is_opt_in(monkeypatch) -> None:
    vive = _FakeViveTrackers(["tracker_only"])
    monkeypatch.setattr(vive_data_sender.time, "monotonic", lambda: 0.0)

    specs = vive_data_sender.wait_for_trackers(
        vive,
        1.0,
        allow_single_tracker=True,
    )

    assert specs is not None
    assert [(spec.device_name, spec.tracker_role) for spec in specs] == [
        ("tracker_only", "left_wrist"),
        (vive_data_sender.SINGLE_TRACKER_COMPAT_RIGHT_NAME, "right_wrist"),
    ]


def test_vive_data_sender_supports_explicit_additional_trackers() -> None:
    vive = _FakeViveTrackers(
        ["tracker_e", "tracker_c", "tracker_a", "tracker_d", "tracker_b"]
    )

    specs = vive_data_sender.auto_tracker_specs(vive, tracker_count=5)

    assert [(spec.device_name, spec.tracker_role) for spec in specs] == [
        ("tracker_a", "left_wrist"),
        ("tracker_b", "right_wrist"),
        ("tracker_c", "auxiliary"),
        ("tracker_d", "auxiliary_2"),
        ("tracker_e", "auxiliary_3"),
    ]


@pytest.mark.parametrize("names", [[], ["tracker_a"], ["tracker_a", "tracker_b", "tracker_c"]])
def test_vive_data_sender_requires_exactly_two_trackers(names: list[str]) -> None:
    with pytest.raises(ValueError, match="exactly 2 GenericTrackers"):
        vive_data_sender.auto_tracker_specs(_FakeViveTrackers(names))


def test_vive_data_sender_explicit_tracker_count_is_still_exact() -> None:
    with pytest.raises(ValueError, match="exactly 3 GenericTrackers"):
        vive_data_sender.auto_tracker_specs(
            _FakeViveTrackers(["tracker_a", "tracker_b", "tracker_c", "tracker_d"]),
            tracker_count=3,
        )


def test_vive_data_sender_waits_for_explicit_tracker_count() -> None:
    vive = _FakeViveTrackers(["tracker_c", "tracker_a", "tracker_b"])

    specs = vive_data_sender.wait_for_trackers(vive, 1.0, tracker_count=3)

    assert specs is not None
    assert [(spec.device_name, spec.tracker_role) for spec in specs] == [
        ("tracker_a", "left_wrist"),
        ("tracker_b", "right_wrist"),
        ("tracker_c", "auxiliary"),
    ]


def test_vive_data_sender_tracker_count_cli_is_opt_in() -> None:
    assert vive_data_sender.parse_args(["--tracker-count", "3"]).tracker_count == 3

    with pytest.raises(SystemExit):
        vive_data_sender.parse_args(["--tracker-count", "1"])


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
