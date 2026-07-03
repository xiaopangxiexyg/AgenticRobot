from __future__ import annotations

from hardware.VIVE.VIVE_sender import data_sender_new


def test_data_sender_new_defaults_to_wrist_trackers_only() -> None:
    specs = data_sender_new.parse_tracker_specs([])

    assert [(spec.device_name, spec.tracker_role) for spec in specs] == [
        ("tracker_1", "left_wrist"),
        ("tracker_2", "right_wrist"),
    ]


def test_data_sender_new_accepts_explicit_auxiliary_tracker() -> None:
    specs = data_sender_new.parse_tracker_specs(
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
    assert data_sender_new.send_error_backoff_seconds(0, 0.25, 2.0) == 0.0
    assert data_sender_new.send_error_backoff_seconds(1, 0.25, 2.0) == 0.25
    assert data_sender_new.send_error_backoff_seconds(4, 0.25, 2.0) == 2.0
    assert data_sender_new.send_error_backoff_seconds(20, 0.25, 2.0) == 2.0
