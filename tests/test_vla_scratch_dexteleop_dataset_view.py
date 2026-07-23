from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dataset_contracts import (
    DEXTELEOP_DATASET_SCHEMA_V23,
    PolicyArtifactContract,
    PolicyCameraInput,
    PolicyCameraInputContract,
    PoseActionChunkContract,
    PoseFrameContract,
    PoseSourceProfile,
    PoseTrainingDataContract,
)
from dataset_contracts.v22 import (
    DiscreteCategory,
    DiscreteTokenSpec,
    EndEffectorSpec,
    PolicyIOLayout,
)
from dataset_contracts.vla_scratch import (
    VLA_SCRATCH_ACTIONS_KEY,
    VLA_SCRATCH_IMAGE_MASKS_KEY,
    VLA_SCRATCH_IMAGES_KEY,
    VLA_SCRATCH_STATE_KEY,
    VLA_SCRATCH_TASK_KEY,
)
from dataset_interface.pose_chunk import pose_action_chunk_from_episode_rows
from dataset_interface.vla_scratch import (
    build_vla_scratch_dexteleop_processed_sample,
    vla_scratch_binding_from_policy_artifact,
)


CAMERAS = ("front", "left_wrist", "right_wrist")


def _pose(x: float, y: float) -> list[float]:
    return [x, y, 0.35, 0.0, 0.0, 0.0, 1.0]


def _layout() -> PolicyIOLayout:
    return PolicyIOLayout(
        name="dexteleop_v23_dual_arm_q6_fixture",
        end_effectors={
            side: EndEffectorSpec.continuous(
                side,
                adapter="canonical_joint_target_v1",
                target_dim=6,
                target_names=tuple(f"q{index}" for index in range(6)),
                state_names=tuple(f"q{index}" for index in range(6)),
                units="rad",
            )
            for side in ("left", "right")
        },
    )


def _frame_contract() -> PoseFrameContract:
    return PoseFrameContract.v23_forced_raw(
        PoseSourceProfile(
            source_id="vla_scratch_dual_arm_fixture_v1",
            source_family="test",
            source_description="full dual-arm canonical sample fixture",
            topology="dual_arm",
            raw_to_desk_transform_by_side={
                "left": np.eye(4),
                "right": np.eye(4),
            },
        )
    )


def _chunk_contract(
    *, horizon: int = 4, tail_policy: str = "drop"
) -> PoseActionChunkContract:
    return PoseActionChunkContract(
        action_horizon=horizon,
        action_pose_mode="absolute",
        tail_policy=tail_policy,
    )


def _artifact(
    *,
    layout: PolicyIOLayout | None = None,
    chunk: PoseActionChunkContract | None = None,
    cameras: tuple[str, ...] = CAMERAS,
) -> PolicyArtifactContract:
    resolved_layout = _layout() if layout is None else layout
    resolved_chunk = _chunk_contract() if chunk is None else chunk
    return PolicyArtifactContract(
        policy_io_layout=resolved_layout,
        model_id="vla-scratch-dexteleop-fixture",
        checkpoint_step="untrained-preflight",
        source_datasets=("local/full-dual-arm-v23-fixture",),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        camera_input=PolicyCameraInputContract(
            inputs=tuple(
                PolicyCameraInput(
                    source_key=key,
                    model_input_key=f"observation.images.{key}",
                    server_camera_slot=f"rgb_{index}",
                )
                for index, key in enumerate(cameras)
            )
        ),
        pose_action_chunk=resolved_chunk,
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(_frame_contract().contract_hash,),
    )


def _row(index: int, *, layout: PolicyIOLayout | None = None) -> dict[str, object]:
    resolved_layout = _layout() if layout is None else layout
    action = resolved_layout.pack_action(
        arm_pose_by_side={
            "left": _pose(0.30 + 0.01 * index, 0.20),
            "right": _pose(0.30 + 0.01 * index, -0.20),
        },
        end_effector_by_side={
            "left": np.linspace(0.10, 0.15, 6, dtype=np.float32) + index,
            "right": np.linspace(0.20, 0.25, 6, dtype=np.float32) + index,
        },
        strict=True,
    )
    state = resolved_layout.pack_observation_state(
        arm_pose_by_side={
            "left": _pose(0.20 + 0.01 * index, 0.20),
            "right": _pose(0.20 + 0.01 * index, -0.20),
        },
        end_effector_by_side={
            "left": np.linspace(0.01, 0.06, 6, dtype=np.float32) + index,
            "right": np.linspace(0.07, 0.12, 6, dtype=np.float32) + index,
        },
        strict=True,
    )
    last = resolved_layout.pack_action(
        arm_pose_by_side={
            "left": _pose(0.19 + 0.01 * index, 0.20),
            "right": _pose(0.19 + 0.01 * index, -0.20),
        },
        end_effector_by_side={
            "left": np.linspace(0.00, 0.05, 6, dtype=np.float32) + index,
            "right": np.linspace(0.06, 0.11, 6, dtype=np.float32) + index,
        },
        strict=True,
    )
    return {
        "action": action.continuous,
        "action.discrete": action.discrete,
        "observation.state": state.continuous,
        "observation.state.discrete": state.discrete,
        "observation.last_action": last.continuous,
        "observation.last_action.discrete": last.discrete,
        "observation.arm.left.actual_tcp_pose": _pose(0.20 + 0.01 * index, 0.20),
        "observation.arm.right.actual_tcp_pose": _pose(0.20 + 0.01 * index, -0.20),
        "policy_io_layout_hash": resolved_layout.layout_hash,
    }


def _pose_sample(
    *,
    rows: int = 4,
    chunk: PoseActionChunkContract | None = None,
    start_index: int = 0,
):
    resolved_chunk = _chunk_contract() if chunk is None else chunk
    return pose_action_chunk_from_episode_rows(
        [_row(index) for index in range(rows)],
        _layout(),
        pose_frame_contract=_frame_contract(),
        chunk_contract=resolved_chunk,
        start_index=start_index,
    )


def _images() -> dict[str, np.ndarray]:
    # Deliberately reverse insertion order. Artifact order remains authoritative.
    return {
        "right_wrist": np.full((6, 8, 3), 33, dtype=np.uint8),
        "left_wrist": np.full((6, 8, 3), 22, dtype=np.uint8),
        "front": np.full((6, 8, 3), 11, dtype=np.uint8),
    }


def test_full_dual_arm_three_camera_canonical_sample_uses_artifact_order() -> None:
    artifact = _artifact()
    binding = vla_scratch_binding_from_policy_artifact(
        artifact,
        source_revision="fa3cb37a6f3ba620bd2c167a97035b5431f31d19",
        state_history=1,
        required_camera_source_keys=CAMERAS,
    )
    pose_sample = _pose_sample()
    sample = build_vla_scratch_dexteleop_processed_sample(
        binding=binding,
        pose_sample=pose_sample,
        images_by_source=_images(),
        task="move the object with both arms",
    )
    payload = sample.to_processed_dict()

    assert binding.sample_spec.camera_source_keys == CAMERAS
    assert binding.camera_model_input_keys == tuple(
        f"observation.images.{key}" for key in CAMERAS
    )
    assert binding.camera_server_slots == ("rgb_0", "rgb_1", "rgb_2")
    assert binding.sample_spec.state_dim == 26
    assert binding.sample_spec.action_dim == 26
    assert binding.sample_spec.action_horizon == 4
    assert binding.required_active_sides == ("left", "right")
    assert set(pose_sample.actual_tcp_pose_by_side) == {"left", "right"}

    assert set(payload) == {
        VLA_SCRATCH_IMAGES_KEY,
        VLA_SCRATCH_IMAGE_MASKS_KEY,
        VLA_SCRATCH_STATE_KEY,
        VLA_SCRATCH_TASK_KEY,
        VLA_SCRATCH_ACTIONS_KEY,
    }
    assert payload[VLA_SCRATCH_IMAGES_KEY].shape == (3, 3, 6, 8)
    np.testing.assert_array_equal(
        payload[VLA_SCRATCH_IMAGES_KEY][:, 0, 0, 0],
        [11, 22, 33],
    )
    np.testing.assert_array_equal(
        payload[VLA_SCRATCH_IMAGE_MASKS_KEY],
        np.ones((3, 1), dtype=bool),
    )
    assert payload[VLA_SCRATCH_STATE_KEY].shape == (1, 26)
    assert payload[VLA_SCRATCH_ACTIONS_KEY].shape == (4, 26)
    np.testing.assert_array_equal(
        payload[VLA_SCRATCH_STATE_KEY][0],
        pose_sample.observation_state,
    )
    np.testing.assert_array_equal(
        payload[VLA_SCRATCH_ACTIONS_KEY],
        pose_sample.action,
    )
    assert np.all(np.isfinite(payload[VLA_SCRATCH_STATE_KEY]))
    assert np.all(np.isfinite(payload[VLA_SCRATCH_ACTIONS_KEY]))
    assert len(binding.semantic_content_hash) == 64
    assert set(binding.semantic_content_hash) <= set("0123456789abcdef")


def test_state_history_is_explicit_and_must_end_at_chunk_start() -> None:
    binding = vla_scratch_binding_from_policy_artifact(
        _artifact(),
        source_revision="fixture-revision",
        state_history=2,
        required_camera_source_keys=CAMERAS,
    )
    pose_sample = _pose_sample()
    history = np.stack((_row(0)["observation.state"], pose_sample.observation_state))
    sample = build_vla_scratch_dexteleop_processed_sample(
        binding=binding,
        pose_sample=pose_sample,
        images_by_source=_images(),
        task="history fixture",
        observation_state_history=history,
    )
    np.testing.assert_array_equal(sample.observation.state, history)

    wrong_history = history.copy()
    wrong_history[-1, 0] += 1.0
    with pytest.raises(ValueError, match="final state-history row"):
        build_vla_scratch_dexteleop_processed_sample(
            binding=binding,
            pose_sample=pose_sample,
            images_by_source=_images(),
            task="history fixture",
            observation_state_history=wrong_history,
        )


def test_binding_fails_closed_on_missing_camera_and_discrete_channels() -> None:
    with pytest.raises(ValueError, match="missing required camera"):
        vla_scratch_binding_from_policy_artifact(
            _artifact(cameras=("front", "left_wrist")),
            source_revision="fixture-revision",
            state_history=1,
            required_camera_source_keys=CAMERAS,
        )

    discrete_layout = PolicyIOLayout(
        name="discrete-right-gripper",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.discrete(
                "right",
                adapter="binary_v1",
                target_tokens=(
                    DiscreteTokenSpec(
                        name="mode",
                        categories=(
                            DiscreteCategory(id=0, label="open"),
                            DiscreteCategory(id=1, label="close"),
                        ),
                    ),
                ),
            ),
        },
    )
    with pytest.raises(ValueError, match="discrete policy channels"):
        vla_scratch_binding_from_policy_artifact(
            _artifact(layout=discrete_layout),
            source_revision="fixture-revision",
            state_history=1,
        )


def test_sample_rejects_unbound_frame_and_unmasked_padded_tail() -> None:
    binding = vla_scratch_binding_from_policy_artifact(
        _artifact(),
        source_revision="fixture-revision",
        state_history=1,
    )
    wrong_frame_sample = replace(
        _pose_sample(), pose_frame_contract_hash="sha256:wrong"
    )
    with pytest.raises(ValueError, match="not authorized"):
        build_vla_scratch_dexteleop_processed_sample(
            binding=binding,
            pose_sample=wrong_frame_sample,
            images_by_source=_images(),
            task="frame mismatch",
        )

    padded_chunk = _chunk_contract(horizon=4, tail_policy="pad_last")
    padded_binding = vla_scratch_binding_from_policy_artifact(
        _artifact(chunk=padded_chunk),
        source_revision="fixture-revision",
        state_history=1,
    )
    padded_sample = _pose_sample(rows=2, chunk=padded_chunk, start_index=0)
    with pytest.raises(ValueError, match="no action padding-mask"):
        build_vla_scratch_dexteleop_processed_sample(
            binding=padded_binding,
            pose_sample=padded_sample,
            images_by_source=_images(),
            task="padded tail",
        )


def test_binding_identity_changes_when_artifact_camera_order_changes() -> None:
    first = vla_scratch_binding_from_policy_artifact(
        _artifact(cameras=CAMERAS),
        source_revision="fixture-revision",
        state_history=1,
    )
    second = vla_scratch_binding_from_policy_artifact(
        _artifact(cameras=("left_wrist", "front", "right_wrist")),
        source_revision="fixture-revision",
        state_history=1,
    )
    assert first.sample_spec.camera_source_keys != second.sample_spec.camera_source_keys
    assert first.artifact_hash != second.artifact_hash
    assert first.semantic_content_hash != second.semantic_content_hash
