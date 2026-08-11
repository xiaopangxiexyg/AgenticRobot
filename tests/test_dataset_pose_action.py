from __future__ import annotations

import json

import numpy as np
import pytest

from dataset_contracts.pose_frames import PoseFrameContract, PoseSourceProfile
from dataset_contracts.v22 import EndEffectorSpec, PolicyIOLayout
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_interface.pose_action import (
    PosePolicyViewError,
    pose_policy_window_from_dataset_metadata,
    pose_policy_window_from_rows,
)
from robot_motion.relative_action import (
    ACTION_POSE_MODE_ABSOLUTE,
    ACTION_POSE_MODE_RELATIVE,
    absolute_pose,
    xyzquat_to_pose_matrix,
)
from teleop_runtime.replay.dataset import ReplayDatasetIndex


def _pose(x: float, y: float = 0.0) -> list[float]:
    return [x, y, 0.3, 0.0, 0.0, 0.0, 1.0]


def _layout() -> PolicyIOLayout:
    return PolicyIOLayout(
        name="pose_window_test",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.none("right"),
        },
    )


def _contract() -> PoseFrameContract:
    transform_left = np.eye(4)
    transform_left[:3, 3] = [1.0, 0.0, 0.0]
    transform_right = np.eye(4)
    transform_right[:3, 3] = [0.0, -0.5, 0.0]
    source = PoseSourceProfile(
        source_id="synthetic_dual_v1",
        source_family="test",
        source_description="synthetic pose source",
        topology="dual_arm",
        raw_to_desk_transform_by_side={"left": transform_left, "right": transform_right},
    )
    return PoseFrameContract(source=source, pose_storage_frame="raw_absolute")


def _row(index: int) -> dict[str, object]:
    layout = _layout()
    action = np.concatenate(
        [np.asarray(_pose(0.2 + 0.01 * index)), np.asarray(_pose(0.4 + 0.01 * index))]
    ).astype(np.float32)
    state = np.concatenate(
        [np.asarray(_pose(0.1 + 0.01 * index)), np.asarray(_pose(0.3 + 0.01 * index))]
    ).astype(np.float32)
    last_action = np.concatenate(
        [np.asarray(_pose(0.15 + 0.01 * index)), np.asarray(_pose(0.35 + 0.01 * index))]
    ).astype(np.float32)
    return {
        "action": action.tolist(),
        "observation.state": state.tolist(),
        "observation.last_action": last_action.tolist(),
        "observation.arm.left.actual_tcp_pose": _pose(0.0 + 0.01 * index),
        "observation.arm.right.actual_tcp_pose": _pose(0.1 + 0.01 * index),
        "action.discrete": [],
        "observation.state.discrete": [],
        "observation.last_action.discrete": [],
        "policy_io_layout_hash": layout.layout_hash,
    }


def test_pose_window_keeps_state_absolute_and_converts_action_last_action() -> None:
    rows = [_row(0), _row(1), _row(2)]
    original_action = np.asarray(rows[0]["action"], dtype=np.float32).copy()
    window = pose_policy_window_from_rows(
        rows,
        _layout(),
        pose_frame_contract=_contract(),
        action_pose_mode=ACTION_POSE_MODE_RELATIVE,
        last_action_pose_mode=ACTION_POSE_MODE_RELATIVE,
    )

    assert window.action_pose_mode == ACTION_POSE_MODE_RELATIVE
    assert window.last_action_pose_mode == ACTION_POSE_MODE_RELATIVE
    assert window.observation_state.shape == (3, 14)
    # The state is always desk absolute: raw x + left transform x translation.
    np.testing.assert_allclose(window.observation_state[0, :3], [1.1, 0.0, 0.3], atol=1e-6)
    # Source rows remain untouched by the derived view.
    np.testing.assert_array_equal(rows[0]["action"], original_action)

    # Relative action row 0 is expressed in the chunk-start left TCP frame.
    expected_left = np.linalg.inv(
        xyzquat_to_pose_matrix([1.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    ) @ xyzquat_to_pose_matrix([1.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        xyzquat_to_pose_matrix(window.action[0, :7]), expected_left, atol=1e-6
    )

    # last_action row 1 uses row 1's current TCP, not row 0's anchor.
    reconstructed = absolute_pose(
        window.actual_tcp_pose_by_side["left"][1],
        xyzquat_to_pose_matrix(window.last_action[1, :7]),
    )
    np.testing.assert_allclose(
        reconstructed,
        xyzquat_to_pose_matrix([1.16, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
    )


def test_absolute_pose_window_is_a_desk_view_without_relative_conversion() -> None:
    window = pose_policy_window_from_rows(
        [_row(0), _row(1)],
        _layout(),
        pose_frame_contract=_contract(),
        action_pose_mode=ACTION_POSE_MODE_ABSOLUTE,
    )
    np.testing.assert_allclose(window.action[0, :3], [1.2, 0.0, 0.3], atol=1e-6)
    np.testing.assert_allclose(window.action[0, 7:10], [0.4, -0.5, 0.3], atol=1e-6)


def test_pose_window_rejects_rows_across_pause_resume_continuity_segments() -> None:
    rows = [_row(0), _row(1)]
    rows[0]["collection.continuity_segment_index"] = 0
    rows[1]["collection.continuity_segment_index"] = 1
    with pytest.raises(PosePolicyViewError, match="pause/resume continuity segment"):
        pose_policy_window_from_rows(
            rows,
            _layout(),
            pose_frame_contract=_contract(),
        )


def test_dataset_metadata_resolves_contract_without_mutating_storage_rows() -> None:
    layout = _layout()
    contract = _contract()
    metadata = {
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        **layout.binding_metadata(),
        "pose_frame_contract": contract.to_dict(),
        "pose_storage_frame": contract.pose_storage_frame,
    }
    rows = [_row(0), _row(1)]
    source_action = list(rows[0]["action"])
    window = pose_policy_window_from_dataset_metadata(rows, metadata)
    np.testing.assert_allclose(window.action[0, :3], [1.2, 0.0, 0.3], atol=1e-6)
    assert rows[0]["action"] == source_action


def test_episode_contract_cannot_override_dataset_contract() -> None:
    layout = _layout()
    contract = _contract()
    metadata = {
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        **layout.binding_metadata(),
        "pose_frame_contract": contract.to_dict(),
        "pose_storage_frame": contract.pose_storage_frame,
    }
    conflicting_source = PoseSourceProfile(
        source_id="other_source",
        source_family="test",
        source_description="conflict",
        topology="dual_arm",
    )
    conflicting = PoseFrameContract.v23_forced_raw(conflicting_source)
    with pytest.raises(PosePolicyViewError, match="differs"):
        pose_policy_window_from_dataset_metadata(
            [_row(0)],
            metadata,
            episode_metadata={"pose_frame_contract": conflicting.to_dict()},
        )


@pytest.mark.optional
def test_replay_index_explicit_pose_window_keeps_loaded_storage_rows_raw(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    layout = _layout()
    contract = _contract()
    info = {
        "chunk_size": 1000,
        "nominal_fps": 10.0,
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        **layout.binding_metadata(),
        "pose_frame_contract": contract.to_dict(),
        "pose_storage_frame": contract.pose_storage_frame,
    }
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    rows = [_row(0), _row(1)]
    parquet_path = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    episode_record = {
        "episode_index": 0,
        "episode_id": "episode_000000",
        "frame_count": 2,
        "parquet_path": "data/chunk-000/episode_000000.parquet",
        "metadata": {"pose_frame_contract": contract.to_dict()},
    }
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps(episode_record) + "\n", encoding="utf-8"
    )

    index = ReplayDatasetIndex(tmp_path)
    loaded = index.load_episode(0)
    window = index.load_pose_policy_window(0)
    assert loaded.rows[0]["action"][0] == pytest.approx(0.2)
    assert window.action[0, 0] == pytest.approx(1.2)
