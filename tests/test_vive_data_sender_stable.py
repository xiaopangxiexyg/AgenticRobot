from __future__ import annotations

import ast
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from hardware.VIVE.mocap import vive_data_sender_stable as stable
from teleop_runtime.sources.pose.vive import VivePoseSource, ViveTrackerConfig
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


def test_stable_sender_defaults_preserve_current_wire_contract() -> None:
    args = stable.parse_args([])

    assert args.hz == 300.0
    assert args.port == 9999
    assert args.tracker_count == 2
    assert args.allow_single_tracker is False
    assert args.clock_sync_mode == "optional"
    assert args.clock_sync_port == 10001
    assert args.print_hz == 1.0
    assert args.send_buffer_bytes == 0


def test_stable_sender_has_no_repository_runtime_imports() -> None:
    source_path = Path(stable.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        str(node.module).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert imported_roots <= {
        "__future__",
        "argparse",
        "json",
        "math",
        "openvr",
        "queue",
        "socket",
        "threading",
        "time",
        "uuid",
        "dataclasses",
        "pathlib",
        "typing",
    }
    assert "triad_openvr" not in source_path.read_text(encoding="utf-8")


def test_stable_packet_is_accepted_by_existing_receiver() -> None:
    source = VivePoseSource(
        [
            ViveTrackerConfig(
                name="tracker_1",
                side="right",
                role="left_wrist",
                channel_name="right_wrist_pose",
            )
        ],
        clock_sync_port=0,
    )
    packet = stable.build_tracking_packet(
        seq=17,
        sender_timestamp=100.0,
        pose_sample_timestamp=100.1,
        pose_sample_monotonic_time=50.25,
        trackers=[
            {
                "tracker_name": "tracker_1",
                "tracker_role": "left_wrist",
                "valid": True,
                "pose_matrix_4x4": np.eye(4).tolist(),
                "error": None,
                "read_elapsed_ms": 0.02,
                "linear_velocity_m_s": [0.1, 0.2, 0.3],
                "angular_velocity_rad_s": [0.4, 0.5, 0.6],
                "velocity_frame_name": "vive_world",
                "velocity_source": "openvr_tracked_device_pose",
                "velocity_valid": True,
                "velocity_error": None,
            }
        ],
    )

    parsed = source._parse_packet(stable._json_bytes(packet), received_at=101.0)

    channel = parsed["right_wrist_pose"]
    assert channel["seq"] == 17
    assert channel["valid"] is True
    assert channel["pose_sample_monotonic_time"] == 50.25
    assert channel["linear_velocity_m_s"] == [0.1, 0.2, 0.3]


def test_single_mode_enters_main_send_loop_and_emits_invalid_right(
    monkeypatch,
) -> None:
    class Pose:
        bDeviceIsConnected = True
        bPoseIsValid = True
        mDeviceToAbsoluteTracking = (
            (1.0, 0.0, 0.0, 0.1),
            (0.0, 1.0, 0.0, 0.2),
            (0.0, 0.0, 1.0, 0.3),
        )
        vVelocity = (0.0, 0.0, 0.0)
        vAngularVelocity = (0.0, 0.0, 0.0)

    class Registry:
        def __init__(self, _config: object) -> None:
            self.snapshot_count = 0
            self.tracker_names = ["tracker_1"]

        def snapshot(self) -> list[Pose]:
            self.snapshot_count += 1
            if self.snapshot_count >= 3:
                raise KeyboardInterrupt
            return [Pose()]

        def refresh(self, _snapshot: object) -> int:
            return 0

        def poll_events(self) -> bool:
            return False

        def describe(self) -> list[str]:
            return ["tracker_1 index=0 serial=SERIAL-1 model=MODEL-1"]

        def record(self, name: str) -> stable.DeviceRecord:
            assert name == "tracker_1"
            return stable.DeviceRecord(0, name, "SERIAL-1", "MODEL-1")

        def close(self) -> None:
            pass

    class CaptureTransport:
        payloads: list[bytes] = []

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def send(self, payload: bytes, *, now: float) -> tuple[bool, float]:
            assert now > 0.0
            self.payloads.append(payload)
            return True, 0.01

        def close(self) -> None:
            pass

    monkeypatch.setattr(stable, "OpenVRTrackerRegistry", Registry)
    monkeypatch.setattr(stable, "PoseDatagramSender", CaptureTransport)

    stable.main(
        [
            "--single",
            "--hz",
            "0",
            "--print-hz",
            "0",
            "--clock-sync-mode",
            "off",
        ]
    )

    assert len(CaptureTransport.payloads) == 1
    packet = json.loads(CaptureTransport.payloads[0])
    assert [(item["tracker_role"], item["valid"]) for item in packet["trackers"]] == [
        ("left_wrist", True),
        ("right_wrist", False),
    ]


class _Event:
    eventType = 0


class _EventSystem:
    def __init__(self, event_types: list[int]) -> None:
        self.event_types = list(event_types)

    def pollNextEvent(self, event: _Event) -> bool:
        if not self.event_types:
            return False
        event.eventType = self.event_types.pop(0)
        return True


def test_event_poll_only_marks_registry_dirty() -> None:
    registry = stable.OpenVRTrackerRegistry.__new__(stable.OpenVRTrackerRegistry)
    registry.openvr = type(
        "OpenVR",
        (),
        {
            "VREvent_t": _Event,
            "VREvent_TrackedDeviceActivated": 100,
            "VREvent_TrackedDeviceDeactivated": 101,
        },
    )()
    registry.system = _EventSystem([7, 100, 101])

    assert registry.poll_events() is True
    assert registry.system.event_types == []


def test_registry_refresh_reuses_the_provided_pose_snapshot() -> None:
    class Pose:
        bDeviceIsConnected = True

    class System:
        def getTrackedDeviceClass(self, index: int) -> int:
            return 3

        def getStringTrackedDeviceProperty(self, index: int, prop: int) -> str:
            return "SERIAL-A" if prop == 1000 else "MODEL-A"

        def getDeviceToAbsoluteTrackingPose(self, *_args: object) -> object:
            raise AssertionError("refresh must not fetch a second pose snapshot")

    registry = stable.OpenVRTrackerRegistry.__new__(stable.OpenVRTrackerRegistry)
    registry.openvr = type(
        "OpenVR",
        (),
        {
            "k_unMaxTrackedDeviceCount": 1,
            "TrackedDeviceClass_GenericTracker": 3,
            "Prop_SerialNumber_String": 1000,
            "Prop_ModelNumber_String": 1001,
        },
    )()
    registry.system = System()
    registry._configured_names = {}
    registry._name_by_serial = {}
    registry._records_by_name = {}

    registry.refresh([Pose()])

    assert registry.tracker_names == ["tracker_1"]
    assert registry.record("tracker_1").serial == "SERIAL-A"


def test_deadline_scheduler_skips_missed_ticks_without_catchup() -> None:
    scheduler = stable.DeadlineScheduler(1.0, spin_threshold_sec=0.0)
    scheduler.next_deadline = time.perf_counter() - 3.2

    lateness, missed = scheduler.wait()

    assert lateness >= 3.2
    assert missed == 3
    assert scheduler.next_deadline > time.perf_counter()


def test_udp_failure_drops_frame_without_sleeping(monkeypatch) -> None:
    class BrokenSocket:
        def setsockopt(self, *_args: object) -> None:
            pass

        def connect(self, _address: object) -> None:
            raise OSError("offline")

        def close(self) -> None:
            pass

    monkeypatch.setattr(stable.socket, "socket", lambda *_args: BrokenSocket())
    monkeypatch.setattr(
        stable.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("must not sleep")),
    )
    sender = stable.PoseDatagramSender(
        "127.0.0.1",
        9999,
        send_buffer_bytes=0,
        retry_sec=0.25,
    )

    sent, _elapsed_ms = sender.send(b"packet", now=10.0)

    assert sent is False
    assert sender.dropped == 1
    assert sender.errors == 1
    assert sender.next_retry_at == 10.25


@pytest.mark.integration
def test_stable_clock_sync_matches_existing_receiver(
    require_udp_loopback: None,
) -> None:
    port = _free_udp_port()
    server = ViveClockSyncServer(
        ViveClockSyncConfig(
            bind_host="127.0.0.1",
            port=port,
            min_samples=1,
            max_round_trip_sec=1.0,
            expected_sender_host="127.0.0.1",
        )
    )
    client = stable.ClockSyncClient(
        host="127.0.0.1",
        port=port,
        startup_probes=1,
        min_samples=1,
        timeout_sec=0.2,
        max_rtt_sec=1.0,
        refresh_sec=10.0,
    )
    server.start()
    try:
        client.start()
        assert client.wait_ready(1.0)
        assert server.wait_until_ready(1.0, min_samples=1)
        assert client.stats()["accepted"] == 1
    finally:
        client.close()
        server.close()


def test_stable_sender_runs_as_an_isolated_single_file(tmp_path: Path) -> None:
    sender = tmp_path / "vive_data_sender_stable.py"
    sender.write_bytes(Path(stable.__file__).read_bytes())
    (tmp_path / "openvr.py").write_text(
        """
VRApplication_Other = 3
TrackingUniverseStanding = 1
k_unMaxTrackedDeviceCount = 2
TrackedDeviceClass_GenericTracker = 3
Prop_SerialNumber_String = 1000
Prop_ModelNumber_String = 1001
VREvent_TrackedDeviceActivated = 100
VREvent_TrackedDeviceDeactivated = 101

class Pose:
    bDeviceIsConnected = True

class System:
    def getDeviceToAbsoluteTrackingPose(self, universe, prediction, count):
        return [Pose(), Pose()]
    def getTrackedDeviceClass(self, index):
        return TrackedDeviceClass_GenericTracker
    def getStringTrackedDeviceProperty(self, index, prop):
        return ("SERIAL-" if prop == Prop_SerialNumber_String else "MODEL-") + str(index)
    def pollNextEvent(self, event):
        return False

class VREvent_t:
    eventType = 0

_SYSTEM = System()
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

    result = subprocess.run(
        [sys.executable, "-S", str(sender), "--list-devices"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "found 2 GenericTracker(s)" in result.stdout
    assert "tracker_1 index=0 serial=SERIAL-0" in result.stdout
    assert "tracker_2 index=1 serial=SERIAL-1" in result.stdout
