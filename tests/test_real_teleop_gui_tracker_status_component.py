from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from teleop_runtime.real_teleop.gui_tracker_status import (
    publish_raw_tracker_pose_state,
    raw_tracker_pose_state_from_channels,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint


class _Publisher:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def publish(self, **payload: object) -> None:
        self.payloads.append(payload)


def test_extracted_tracker_status_payload_uses_receiver_time_and_marks_inactive_side() -> None:
    pose = np.eye(4)
    pose[:3, 3] = np.array([0.12345, -0.2, 0.3])
    channel = SimpleNamespace(
        value=pose,
        timestamp=1.0,
        valid=True,
        metadata={
            "receiver_timestamp": 9.8,
            "tracker_name": "tracker-left",
            "tracker_role": "left_wrist",
            "seq": 42,
        },
    )

    payload = raw_tracker_pose_state_from_channels(
        {"left_wrist_pose": channel},
        active_sides=("left",),
        now=10.0,
    )

    assert payload["left"] == {
        "enabled": True,
        "valid": True,
        "xyz_m": [0.1234, -0.2, 0.3],
        "rpy_deg": [0.0, 0.0, 0.0],
        "age_sec": 0.2,
        "stale": False,
        "status_level": "ok",
        "status_label": "有效",
        "stale_timeout_sec": 0.5,
        "tracker_name": "tracker-left",
        "tracker_role": "left_wrist",
        "seq": 42,
        "sender_host": "",
        "sender_error": None,
        "sender_read_elapsed_ms": None,
    }
    assert payload["right"]["status_level"] == "disabled"
    assert payload["right"]["enabled"] is False


def test_extracted_tracker_status_publication_preserves_stale_boundary() -> None:
    publisher = _Publisher()
    pose = np.eye(4)
    channel = SimpleNamespace(value=pose, timestamp=1.0, valid=True, metadata={})

    publish_raw_tracker_pose_state(
        publisher,
        {"left_wrist_pose": channel},
        active_sides=("left",),
        now=1.51,
    )

    assert publisher.payloads[-1]["raw_tracker_poses"]["left"]["status_level"] == "stale"
    assert publisher.payloads[-1]["raw_tracker_poses"]["left"]["stale"] is True


def test_legacy_tracker_status_wrapper_preserves_legacy_threshold_patch_point() -> None:
    channel = SimpleNamespace(value=np.eye(4), timestamp=1.0, valid=True, metadata={})

    with patch.object(legacy_entrypoint, "TRACKER_GUI_STALE_AGE_SEC", 0.1):
        payload = legacy_entrypoint._raw_tracker_pose_state_from_channels(
            {"left_wrist_pose": channel},
            active_sides=("left",),
            now=1.2,
        )

    assert payload["left"]["status_level"] == "stale"
