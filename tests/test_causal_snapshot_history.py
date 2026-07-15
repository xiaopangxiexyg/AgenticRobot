from __future__ import annotations

import time

import numpy as np

from robot_motion.runtime.causal_history import CausalSnapshotHistory
from teleop_runtime.collection.hand_state_polling import HandStateHistoryPoller


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
