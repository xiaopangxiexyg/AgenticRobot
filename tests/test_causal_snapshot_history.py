from __future__ import annotations

import time

import numpy as np
import pytest

from robot_motion.runtime.causal_history import CausalSnapshotHistory
from teleop_runtime.collection.hand_state_polling import (
    HandStateHistoryPoller,
    hand_state_tactile_arrays,
)


def test_history_freezes_numpy_payload_and_requires_both_timestamps() -> None:
    history = CausalSnapshotHistory(maxlen=4)
    payload = {"array": np.array([1.0, 2.0])}
    record = history.publish(
        payload,
        source_timestamp=10.0,
        available_timestamp=10.1,
    )
    payload["array"][0] = 99.0
    record["array"][1] = 88.0

    selected = history.latest_at_or_before(10.2)
    assert selected is not None
    frozen, metadata = selected
    np.testing.assert_allclose(frozen["array"], [1.0, 2.0])
    assert metadata.source_timestamp == 10.0
    assert metadata.available_timestamp == 10.1

    assert history.latest_at_or_before(10.05) is None  # available is in the future

    history.publish({"value": 3}, source_timestamp=11.0, available_timestamp=10.2)
    selected = history.latest_at_or_before(10.5)
    assert selected is not None
    assert selected[1].sequence == 1  # the future-source record is rejected


class _FakeHandBackend:
    def __init__(self) -> None:
        self.joint = np.array([0.1, 0.2], dtype=float)
        self.tactile = np.array([1.0, 2.0], dtype=float)
        self.joint_timestamp = 10.0
        self.tactile_timestamp = 10.0
        self.tactile_version = 1

    def get_state(
        self,
        *,
        include_tactile_arrays: bool = True,
        include_tactile_vector: bool = True,
        copy_tactile_arrays: bool = True,
    ) -> dict[str, object]:
        tactile = self.tactile.copy() if include_tactile_arrays else None
        return {
            "joint_pos": self.joint.copy(),
            "measured_joint_pos": self.joint.copy(),
            "joint_timestamp": self.joint_timestamp,
            "telemetry_timestamp": self.joint_timestamp,
            "tactile": tactile if include_tactile_vector else None,
            "tactile_dict": {"sensor": tactile} if tactile is not None else None,
            "tactile_timestamp": self.tactile_timestamp,
            "tactile_version": self.tactile_version,
        }


def test_hand_poller_freezes_telemetry_and_tactile_separately() -> None:
    backend = _FakeHandBackend()
    poller = HandStateHistoryPoller(backends_by_side={"left": backend})
    poller._poll_once("left", backend)

    # Mutating the backend after publication must not mutate either selected
    # stream.  A query cut after publication still returns the frozen frame.
    backend.joint[:] = 9.0
    backend.tactile[:] = 8.0
    selected = poller.latest_by_side(time.time() + 1.0)
    assert "left" in selected
    np.testing.assert_allclose(selected["left"]["joint_pos"], [0.1, 0.2])
    np.testing.assert_allclose(selected["left"]["tactile"], [1.0, 2.0])
    assert selected["left"]["telemetry_snapshot_sequence"] == 1
    assert selected["left"]["tactile_snapshot_sequence"] == 1
    assert selected["left"]["causal_snapshot_contract"] == (
        "source_timestamp<=T0_and_available_timestamp<=T0"
    )
    assert selected["left"]["telemetry_snapshot_source_clock"] == "local_wall_time"
    assert selected["left"]["telemetry_snapshot_available_clock"] == "local_wall_time"
    assert selected["left"]["tactile_snapshot_source_clock"] == "local_wall_time"
    assert selected["left"]["tactile_snapshot_available_clock"] == "local_wall_time"
    assert selected["left"]["telemetry_snapshot_source_age_sec"] > 1.0
    assert selected["left"]["telemetry_snapshot_available_age_sec"] < 2.0
    assert selected["left"]["telemetry_snapshot_age_sec"] == pytest.approx(
        selected["left"]["telemetry_snapshot_source_age_sec"]
    )
    assert selected["left"]["tactile_snapshot_age_sec"] == pytest.approx(
        selected["left"]["tactile_snapshot_source_age_sec"]
    )

    selected_again = poller.latest_by_side(time.time() + 1.0)
    first_array = selected["left"]["tactile_dict"]["sensor"]
    second_array = selected_again["left"]["tactile_dict"]["sensor"]
    assert first_array is second_array
    assert not first_array.flags.writeable
    with pytest.raises(ValueError):
        first_array.setflags(write=True)


def test_hand_poller_uses_joint_read_timestamp_not_newer_worker_loop_timestamp() -> None:
    now = time.time()
    backend = _FakeHandBackend()
    backend.joint_timestamp = now - 0.4
    poller = HandStateHistoryPoller(backends_by_side={"left": backend})
    original_get_state = backend.get_state

    def state_with_new_loop_timestamp(**kwargs):
        state = original_get_state(**kwargs)
        state["telemetry_timestamp"] = now
        return state

    backend.get_state = state_with_new_loop_timestamp
    poller._poll_once("left", backend)
    selected = poller.latest_by_side(now + 0.01)["left"]

    assert selected["telemetry_snapshot_source_timestamp"] == pytest.approx(now - 0.4)
    assert selected["telemetry_snapshot_source_age_sec"] == pytest.approx(0.41)
    assert selected["joint_age_sec"] == pytest.approx(0.41)


def test_tactile_metadata_without_arrays_is_not_writer_ready() -> None:
    assert hand_state_tactile_arrays(
        {
            "tactile_timestamp": time.time(),
            "tactile_version": 3,
            "tactile_devices": ["sensor"],
        }
    ) == {}
    arrays = hand_state_tactile_arrays(
        {"tactile_dict": {"sensor": np.ones((2, 3), dtype=np.float32)}}
    )
    assert set(arrays) == {"sensor"}
    assert arrays["sensor"].shape == (2, 3)
