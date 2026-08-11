from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts import (
    DEXTELEOP_DATASET_SCHEMA_V23,
    POSE_CHUNK_TAIL_DROP,
    POSE_CHUNK_TAIL_PAD_LAST,
    PolicyArtifactContract,
    PoseActionChunkContract,
    PoseFrameContract,
    PoseSourceProfile,
    PoseTrainingDataContract,
)
from dataset_contracts.v22 import DatasetLayoutBindingError, EndEffectorSpec, PolicyIOLayout
from dataset_interface import (
    DexTeleopDatasetIndex,
    PosePolicyViewError,
    pose_action_chunk_from_dataset_metadata,
    pose_action_chunk_from_episode_rows,
    pose_action_chunk_start_indices,
)
from robot_motion.relative_action import absolute_pose, xyzquat_to_pose_matrix
from teleop_runtime.deploy.run_policy import _bind_policy_pose_mode_from_artifact


def _pose(x: float, y: float = 0.0) -> list[float]:
    return [x, y, 0.3, 0.0, 0.0, 0.0, 1.0]


def _layout() -> PolicyIOLayout:
    return PolicyIOLayout(
        name="pose_chunk_test",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.none("right"),
        },
    )


def _contract() -> PoseFrameContract:
    left = np.eye(4)
    left[0, 3] = 1.0
    right = np.eye(4)
    right[1, 3] = -0.5
    return PoseFrameContract.v23_forced_raw(
        PoseSourceProfile(
            source_id="pose_chunk_fixture_v1",
            source_family="test",
            source_description="fixture",
            topology="dual_arm",
            raw_to_desk_transform_by_side={"left": left, "right": right},
        )
    )


def _row(index: int) -> dict[str, object]:
    layout = _layout()
    return {
        "action": _pose(0.2 + 0.01 * index) + _pose(0.4 + 0.01 * index),
        "observation.state": _pose(0.1 + 0.01 * index) + _pose(0.3 + 0.01 * index),
        "observation.last_action": _pose(0.15 + 0.01 * index) + _pose(0.35 + 0.01 * index),
        "observation.arm.left.actual_tcp_pose": _pose(0.01 * index),
        "observation.arm.right.actual_tcp_pose": _pose(0.1 + 0.01 * index),
        "action.discrete": [],
        "observation.state.discrete": [],
        "observation.last_action.discrete": [],
        "policy_io_layout_hash": layout.layout_hash,
    }


def _metadata() -> dict[str, object]:
    layout = _layout()
    contract = _contract()
    return {
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        **layout.binding_metadata(),
        "pose_frame_contract": contract.to_dict(),
        "pose_storage_frame": contract.pose_storage_frame,
    }


def test_relative_pad_last_converts_real_actions_before_padding() -> None:
    rows = [_row(0), _row(1), _row(2)]
    original = list(rows[2]["action"])
    chunk_contract = PoseActionChunkContract(
        action_horizon=4,
        action_pose_mode="relative",
        last_action_pose_mode="relative",
        tail_policy=POSE_CHUNK_TAIL_PAD_LAST,
    )
    sample = pose_action_chunk_from_episode_rows(
        rows,
        _layout(),
        pose_frame_contract=_contract(),
        chunk_contract=chunk_contract,
        start_index=1,
    )

    assert sample.action.shape == (4, 14)
    np.testing.assert_array_equal(sample.action_is_pad, [False, False, True, True])
    assert sample.valid_action_count == 2
    assert sample.pose_frame_contract_hash == _contract().contract_hash
    np.testing.assert_allclose(
        sample.action[2:], np.repeat(sample.action[1:2], 2, axis=0), atol=0.0
    )
    np.testing.assert_allclose(sample.actions_for_normalization(), sample.action[:2])
    assert rows[2]["action"] == original

    # Only the current observation is returned and remains canonical desk absolute.
    assert sample.observation_state.shape == (14,)
    np.testing.assert_allclose(sample.observation_state[:3], [1.11, 0.0, 0.3], atol=1e-6)

    # last_action is relative to the current row's measured TCP, not another chunk.
    reconstructed_last = absolute_pose(
        sample.actual_tcp_pose_by_side["left"],
        xyzquat_to_pose_matrix(sample.last_action[:7]),
    )
    np.testing.assert_allclose(
        reconstructed_last,
        xyzquat_to_pose_matrix([1.16, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
    )


def test_drop_tail_exposes_only_complete_legal_starts() -> None:
    contract = PoseActionChunkContract(
        action_horizon=2,
        action_pose_mode="relative",
        tail_policy=POSE_CHUNK_TAIL_DROP,
    )
    assert pose_action_chunk_start_indices(3, contract) == (0, 1)
    with pytest.raises(PosePolicyViewError, match="tail_policy='drop'"):
        pose_action_chunk_from_episode_rows(
            [_row(0), _row(1), _row(2)],
            _layout(),
            pose_frame_contract=_contract(),
            chunk_contract=contract,
            start_index=2,
        )


def test_pose_chunk_default_tail_policy_is_drop_in_constructor_and_loader() -> None:
    contract = PoseActionChunkContract(action_horizon=2)
    loaded = PoseActionChunkContract.from_dict(
        {
            "schema": contract.schema,
            "action_horizon": 2,
        }
    )

    assert contract.tail_policy == POSE_CHUNK_TAIL_DROP
    assert loaded.tail_policy == POSE_CHUNK_TAIL_DROP
    assert pose_action_chunk_start_indices(3, contract) == (0, 1)


def test_pad_last_exposes_every_real_frame_as_a_start() -> None:
    contract = PoseActionChunkContract(
        action_horizon=4,
        tail_policy=POSE_CHUNK_TAIL_PAD_LAST,
    )
    assert pose_action_chunk_start_indices(3, contract) == (0, 1, 2)


def test_dataset_metadata_adapter_checks_episode_contract() -> None:
    chunk_contract = PoseActionChunkContract(action_horizon=2, action_pose_mode="relative")
    sample = pose_action_chunk_from_dataset_metadata(
        [_row(0), _row(1)],
        _metadata(),
        chunk_contract=chunk_contract,
        start_index=0,
        episode_metadata={"pose_frame_contract": _contract().to_dict()},
    )
    assert sample.chunk_contract_hash == chunk_contract.contract_hash

    other = PoseFrameContract.legacy_v22_raw()
    with pytest.raises(PosePolicyViewError, match="differs"):
        pose_action_chunk_from_dataset_metadata(
            [_row(0)],
            _metadata(),
            chunk_contract=chunk_contract,
            start_index=0,
            episode_metadata={"pose_frame_contract": other.to_dict()},
        )


@pytest.mark.integration
def test_dataset_index_enumerates_episode_local_chunk_locations(tmp_path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 0, "frame_count": 2}),
                json.dumps({"episode_index": 1, "frame_count": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    index = DexTeleopDatasetIndex(tmp_path)
    locations = index.pose_action_chunk_locations(
        PoseActionChunkContract(action_horizon=2)
    )
    assert [(item.episode_index, item.start_index, item.valid_action_count) for item in locations] == [
        (0, 0, 2),
        (1, 0, 2),
        (1, 1, 2),
    ]


def test_v23_policy_artifact_binds_pose_chunk_contract_and_hash() -> None:
    chunk_contract = PoseActionChunkContract(
        action_horizon=8,
        action_pose_mode="relative",
    )
    artifact = PolicyArtifactContract(
        policy_io_layout=_layout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=chunk_contract,
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(_contract().contract_hash,),
    )
    payload = artifact.to_dict()
    restored = PolicyArtifactContract.from_dict(payload)
    assert restored.pose_action_chunk == chunk_contract
    assert restored.pose_training_data == PoseTrainingDataContract.native_v23()
    assert payload["pose_action_chunk_contract_hash"] == chunk_contract.contract_hash
    assert payload["pose_training_data_contract"]["origin"] == "native_v23_dataset"

    payload["pose_action_chunk_contract_hash"] = "sha256:bad"
    with pytest.raises(DatasetLayoutBindingError, match="does not match"):
        PolicyArtifactContract.from_dict(payload)


def test_v23_pose_chunk_artifact_requires_pose_frame_provenance() -> None:
    with pytest.raises(DatasetLayoutBindingError, match="at least one"):
        PolicyArtifactContract(
            policy_io_layout=_layout(),
            dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
            pose_action_chunk=PoseActionChunkContract(action_horizon=2),
            pose_training_data=PoseTrainingDataContract.native_v23(),
        )


def test_v23_pose_chunk_artifact_requires_explicit_training_origin() -> None:
    with pytest.raises(DatasetLayoutBindingError, match="explicitly distinguish"):
        PolicyArtifactContract(
            policy_io_layout=_layout(),
            dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
            pose_action_chunk=PoseActionChunkContract(action_horizon=2),
            pose_frame_contract_hashes=(_contract().contract_hash,),
        )


def test_v22_artifact_rejects_pose_chunk_contract() -> None:
    with pytest.raises(DatasetLayoutBindingError, match="requires dexteleop_dataset_v2.3"):
        PolicyArtifactContract(
            policy_io_layout=_layout(),
            pose_action_chunk=PoseActionChunkContract(action_horizon=2),
            pose_frame_contract_hashes=(_contract().contract_hash,),
        )


def test_deploy_pose_mode_is_bound_from_structured_chunk_contract() -> None:
    artifact = PolicyArtifactContract(
        policy_io_layout=_layout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=4,
            action_pose_mode="relative",
        ),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(_contract().contract_hash,),
    )
    args = SimpleNamespace(policy_action_pose_mode="absolute")
    _bind_policy_pose_mode_from_artifact(args, artifact, raw_argv=[])
    assert args.policy_action_pose_mode == "relative"

    args = SimpleNamespace(policy_action_pose_mode="absolute")
    with pytest.raises(ValueError, match="CLI policy action pose mode disagrees"):
        _bind_policy_pose_mode_from_artifact(
            args,
            artifact,
            raw_argv=["--policy-action-pose-mode", "absolute"],
        )


def test_structured_and_legacy_artifact_pose_modes_cannot_disagree() -> None:
    artifact = PolicyArtifactContract(
        policy_io_layout=_layout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=4,
            action_pose_mode="relative",
        ),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(_contract().contract_hash,),
        metadata={"pose_representation": {"action": "absolute"}},
    )
    with pytest.raises(ValueError, match="disagrees with legacy"):
        _bind_policy_pose_mode_from_artifact(
            SimpleNamespace(policy_action_pose_mode="absolute"),
            artifact,
            raw_argv=[],
        )


def test_deploy_rejects_different_action_and_last_action_pose_modes() -> None:
    artifact = PolicyArtifactContract(
        policy_io_layout=_layout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=4,
            action_pose_mode="relative",
            last_action_pose_mode="absolute",
        ),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(_contract().contract_hash,),
    )
    with pytest.raises(ValueError, match="action_pose_mode and last_action_pose_mode to match"):
        _bind_policy_pose_mode_from_artifact(
            SimpleNamespace(policy_action_pose_mode="absolute"),
            artifact,
            raw_argv=[],
        )
