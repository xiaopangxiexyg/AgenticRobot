from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.deploy import NormalizedAction
from robot_motion.deploy.schema import ACTION_DIM
from teleop_runtime.real_teleop.policy_observation import (
    RealTeleopPolicyObservationCapture,
)


class _ArmPoller:
    def __init__(self, *, blocked: bool = False) -> None:
        self.blocked = blocked

    def readiness_by_side(self, *, required_sides, sample_timestamp):
        del sample_timestamp
        return {
            side: "feedback_stale" if self.blocked else None
            for side in required_sides
        }

    def latest_by_side(self, *, sample_timestamp):
        pose = np.eye(4, dtype=float)
        return {
            side: {
                "actual_qpos": np.zeros(6, dtype=float),
                "actual_tcp_pose": pose,
                "actual_hardware_result_code": 0,
                "actual_state_timestamp": sample_timestamp,
                "actual_state_stale": False,
                "causal_snapshot_contract": (
                    "source_timestamp<=T0_and_available_timestamp<=T0"
                ),
            }
            for side in ("left", "right")
        }


class _HandPoller:
    def latest_by_side(self, *, sample_timestamp):
        return {
            side: {
                "joint_pos": np.zeros(6, dtype=float),
                "joint_timestamp": sample_timestamp,
            }
            for side in ("left", "right")
        }


def _fixture(*, blocked: bool = False):
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_hand_backend=object(),
        right_hand_backend=object(),
    )
    recorder = SimpleNamespace(
        arm_state_poller=_ArmPoller(blocked=blocked),
        hand_state_poller=_HandPoller(),
    )
    return RealTeleopPolicyObservationCapture(
        components=components,
        dataset_recorder=recorder,
        camera_sources={},
        require_images=False,
        pose_frame_contract=None,
        wall_clock=lambda: 100.0,
    )


def test_policy_observation_reuses_causal_collection_snapshots() -> None:
    capture = _fixture()
    observation = capture(step_index=7)

    assert observation.step_index == 7
    assert observation.timestamp == 100.0
    assert observation.active_sides == ("left", "right")
    assert observation.end_effector_sides == ("left", "right")
    assert observation.faults == []
    assert observation.metadata["observation_source"] == (
        "real_teleop_collection_causal_snapshot"
    )


def test_policy_observation_last_action_is_explicit_and_resettable() -> None:
    capture = _fixture()
    vector = np.arange(ACTION_DIM, dtype=np.float32)
    capture.observe_last_action(NormalizedAction(vector=vector))
    np.testing.assert_array_equal(capture(step_index=1).last_action, vector)

    capture.reset()
    assert np.all(np.isnan(capture(step_index=2).last_action))


def test_policy_observation_fails_closed_on_stale_arm_feedback() -> None:
    capture = _fixture(blocked=True)
    with pytest.raises(RuntimeError, match="feedback_stale"):
        capture(step_index=3)


def test_policy_observation_follows_current_recorder_after_dataset_rebind() -> None:
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_hand_backend=object(),
        right_hand_backend=object(),
    )
    current = {
        "recorder": SimpleNamespace(
            arm_state_poller=_ArmPoller(blocked=False),
            hand_state_poller=_HandPoller(),
        )
    }
    capture = RealTeleopPolicyObservationCapture(
        components=components,
        dataset_recorder=lambda: current["recorder"],
        camera_sources={},
        require_images=False,
        pose_frame_contract=None,
        wall_clock=lambda: 100.0,
    )

    assert capture(step_index=1).faults == []
    current["recorder"] = SimpleNamespace(
        arm_state_poller=_ArmPoller(blocked=True),
        hand_state_poller=_HandPoller(),
    )
    with pytest.raises(RuntimeError, match="feedback_stale"):
        capture(step_index=2)
