from __future__ import annotations

from types import SimpleNamespace

from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint
from teleop_runtime.real_teleop.wrist_depth_freshness import WristDepthFreshnessMonitor


def _frame(*, index: int, source_timestamp: float) -> SimpleNamespace:
    return SimpleNamespace(frame_index=index, source_timestamp=source_timestamp)


def test_extracted_wrist_depth_monitor_matches_legacy_repeated_frame_contract() -> None:
    assert legacy_entrypoint.WristDepthFreshnessMonitor is WristDepthFreshnessMonitor
    legacy = legacy_entrypoint.WristDepthFreshnessMonitor(stale_repeat_threshold=2)
    extracted = WristDepthFreshnessMonitor(stale_repeat_threshold=2)
    observations = [
        (_frame(index=1, source_timestamp=100.0), {"frame_number": 7}),
        (_frame(index=2, source_timestamp=100.1), {"frame_number": 7}),
        (_frame(index=3, source_timestamp=100.2), {"frame_number": 7}),
    ]

    for now, (frame, metadata) in enumerate(observations, start=100):
        assert extracted.update("left_wrist", frame, now=float(now), depth_metadata=metadata) == legacy.update(
            "left_wrist",
            frame,
            now=float(now),
            depth_metadata=metadata,
        )


def test_extracted_wrist_depth_monitor_matches_legacy_epoch_staleness_contract() -> None:
    legacy = legacy_entrypoint.WristDepthFreshnessMonitor()
    extracted = WristDepthFreshnessMonitor()
    frame = _frame(index=1, source_timestamp=1_700_000_004.5)
    metadata = {
        "frame_number": 3,
        "camera_timestamp_sec": 1_700_000_000.0,
        "sensor_timestamp_ms": 123.4567,
    }

    actual = extracted.update("right_wrist", frame, now=12.0, depth_metadata=metadata)
    expected = legacy.update("right_wrist", frame, now=12.0, depth_metadata=metadata)

    assert actual == expected
    assert actual["depth_freshness_status_level"] == "error"
    assert actual["depth_freshness_status_label"] == "Wrist Depth 停滞"


def test_extracted_wrist_depth_monitor_reset_is_scoped_to_requested_camera() -> None:
    monitor = WristDepthFreshnessMonitor()
    monitor.update("left_wrist", _frame(index=1, source_timestamp=1.0), now=1.0, depth_metadata={"frame_number": 1})
    monitor.update("right_wrist", _frame(index=1, source_timestamp=1.0), now=1.0, depth_metadata={"frame_number": 1})

    monitor.reset("left_wrist")

    left = monitor.update("left_wrist", _frame(index=2, source_timestamp=2.0), now=2.0, depth_metadata={"frame_number": 2})
    right = monitor.update("right_wrist", _frame(index=2, source_timestamp=2.0), now=2.0, depth_metadata={"frame_number": 1})
    assert left["depth_signature_total_seen"] == 1
    assert right["depth_signature_total_seen"] == 2
