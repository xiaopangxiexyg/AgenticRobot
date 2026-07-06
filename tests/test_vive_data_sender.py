from __future__ import annotations

from hardware.VIVE.mocap import vive_data_sender


def test_vive_data_sender_defaults_to_wrist_trackers_only() -> None:
    specs = vive_data_sender.parse_tracker_specs([])

    assert [(spec.device_name, spec.tracker_role) for spec in specs] == [
        ("tracker_1", "left_wrist"),
        ("tracker_2", "right_wrist"),
    ]


def test_vive_data_sender_accepts_explicit_auxiliary_tracker() -> None:
    specs = vive_data_sender.parse_tracker_specs(
        [
            "tracker_1:left_wrist",
            "tracker_2:right_wrist",
            "tracker_3:auxiliary",
        ]
    )

    assert [(spec.device_name, spec.tracker_role) for spec in specs] == [
        ("tracker_1", "left_wrist"),
        ("tracker_2", "right_wrist"),
        ("tracker_3", "auxiliary"),
    ]


def test_send_error_backoff_is_bounded() -> None:
    assert vive_data_sender.send_error_backoff_seconds(0, 0.25, 2.0) == 0.0
    assert vive_data_sender.send_error_backoff_seconds(1, 0.25, 2.0) == 0.25
    assert vive_data_sender.send_error_backoff_seconds(4, 0.25, 2.0) == 2.0
    assert vive_data_sender.send_error_backoff_seconds(20, 0.25, 2.0) == 2.0


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
