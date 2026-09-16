from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts.state_history import PolicyStateHistorySpec
from dataset_contracts.v22 import PolicyIOLayout
from dataset_interface.state_history import PolicyStateHistoryBuffer
from deployments.openpi.client.state_history import (
    OpenPIStateHistoryBinding,
    OpenPIStateHistoryRecorder,
)
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.schema import (
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
    STATE_HAND_SLICES,
)
from robot_motion.deploy.v22 import V22DeploymentAdapter

_ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return {
        "schema": "dexteleop_history_state_input_v2",
        "input_key": "history_states",
        "age_input_key": "history_state_ages_sec",
        "frame_offsets": [4, 8, 16],
        "offset_semantics": "causal_control_frame_offset",
        "state_dim": 28,
        "state_layout": [
            "left_arm_qpos_6",
            "right_arm_qpos_6",
            "left_tcp_xyzquat_7",
            "right_tcp_xyzquat_7",
            "left_gripper_opening_1",
            "right_gripper_opening_1",
        ],
        "age_unit": "seconds",
        "selection": "same_episode_same_continuity_segment_exact_past_frame",
        "token_placement": "pi0_action_expert_prefix",
        "qpos_normalization": "divide_by_pi",
        "policy_state_normalization": "reuse_state_norm_stats",
    }


def _layout() -> PolicyIOLayout:
    payload = json.loads(
        (
            _ROOT
            / "teleop_runtime/configs/policy_io_layouts/dual_arm_bimanual_dmtac_gripper_v22.json"
        ).read_text(encoding="utf-8")
    )
    return PolicyIOLayout.from_dict(payload)


def _observation(step: int) -> DeploymentObservation:
    state = np.zeros((OBSERVATION_STATE_DIM,), dtype=np.float32)
    for side, base in (("left", 1.0), ("right", 11.0)):
        state[STATE_ARM_QPOS_SLICES[side]] = base + step
        pose = np.zeros((7,), dtype=np.float32)
        pose[:3] = base + step / 100.0
        pose[6] = 1.0
        state[STATE_ARM_POSE_SLICES[side]] = pose
        state[STATE_HAND_SLICES[side]] = 0.25 if side == "left" else 0.75
    return DeploymentObservation(
        timestamp=step / 30.0,
        step_index=step,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        state=state,
        state_valid=np.ones((OBSERVATION_STATE_DIM,), dtype=bool),
        hand_states={
            "left": {"gripper_measured_scalar": 0.25},
            "right": {"gripper_measured_scalar": 0.75},
        },
    )


def test_history_buffer_selects_exact_offsets_and_actual_ages() -> None:
    buffer = PolicyStateHistoryBuffer(PolicyStateHistorySpec())
    for step in range(17):
        buffer.append(
            timestamp=step / 30.0,
            step_index=step,
            state=np.full((28,), step, dtype=np.float32),
        )

    snapshot = buffer.snapshot(anchor_timestamp=16 / 30.0, anchor_step_index=16)

    assert snapshot is not None
    np.testing.assert_array_equal(np.asarray(snapshot.states)[:, 0], [12, 8, 0])
    np.testing.assert_allclose(snapshot.ages_sec, [4 / 30, 8 / 30, 16 / 30])


def test_history_buffer_resets_on_control_step_gap() -> None:
    buffer = PolicyStateHistoryBuffer(PolicyStateHistorySpec())
    for step in range(17):
        buffer.append(timestamp=step / 30.0, step_index=step, state=np.zeros((28,), dtype=np.float32))
    buffer.append(timestamp=20 / 30.0, step_index=20, state=np.zeros((28,), dtype=np.float32))

    assert buffer.snapshot(anchor_timestamp=20 / 30.0, anchor_step_index=20) is None


def test_openpi_history_binding_rejects_custom_hash_or_checkpoint_binding() -> None:
    payload = _contract()
    payload["contract_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="unknown=.*contract_hash"):
        OpenPIStateHistoryBinding.from_mapping(payload)

    payload = _contract()
    payload["checkpoint_step"] = 100_000
    with pytest.raises(ValueError, match="unknown=.*checkpoint_step"):
        OpenPIStateHistoryBinding.from_mapping(payload)


def test_openpi_history_recorder_builds_qpos_tcp_gripper_rows() -> None:
    binding = OpenPIStateHistoryBinding.from_mapping(_contract())
    recorder = OpenPIStateHistoryRecorder(
        binding=binding,
        adapter=V22DeploymentAdapter(_layout()),
    )
    for step in range(17):
        recorder.observe(_observation(step))

    anchor = _observation(16)
    assert recorder.is_ready(anchor)
    payload = recorder.request_payload(anchor)

    assert payload["history_states"].shape == (3, 28)
    assert payload["history_state_ages_sec"].shape == (3,)
    np.testing.assert_allclose(payload["history_states"][0, :6], 13.0)
    np.testing.assert_allclose(payload["history_states"][0, 6:12], 23.0)
    np.testing.assert_allclose(
        payload["history_state_ages_sec"],
        [4 / 30, 8 / 30, 16 / 30],
    )
