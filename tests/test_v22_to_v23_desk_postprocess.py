from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts.pose_training import PoseTrainingDataContract
from dataset_contracts.policy_artifact import PolicyArtifactContract
from dataset_contracts.pose_chunk import PoseActionChunkContract
from dataset_contracts.v22 import (
    DEXTELEOP_DATASET_SCHEMA_V22,
    PolicyIOLayout,
)
from dataset_contracts.v23 import (
    DEXTELEOP_DATASET_SCHEMA_V23,
    V23_COMPATIBILITY_V22_CANONICAL_DESK,
    resolve_v23_dataset_contract,
)
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract,
)
from robot_motion.deploy.session_contract import resolve_deploy_session_contract
from utils.dataset_postprocess.v21_to_v22 import load_v22_layout_json
from utils.dataset_postprocess.v22_to_v23_desk import (
    DEFAULT_TARGET_LAYOUT_PATH,
    convert_v22_dataset_to_v23_desk,
    plan_v22_to_v23_desk_conversion,
)


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _pose(x: float, y: float = 0.0) -> list[float]:
    return [x, y, 0.3, 0.0, 0.0, 0.0, 1.0]


def _source_layout() -> PolicyIOLayout:
    target = load_v22_layout_json(DEFAULT_TARGET_LAYOUT_PATH)
    return PolicyIOLayout(
        name="dual_arm_bimanual_dmtac_gripper_v22_base_fixture",
        arm_controlled_frame_by_side=target.arm_controlled_frame_by_side,
        arm_reference_frame_by_side={
            "left": "left_arm_base",
            "right": "right_arm_base",
        },
        end_effectors=target.end_effectors,
        metadata={"pose_storage_frame": "raw_absolute"},
    )


def _sample_metadata(action: list[float]) -> str:
    return json.dumps(
        {
            "canonical_action_record": {
                "action": {
                    "arm_pose_by_side": {
                        "left": action[:7],
                        "right": action[7:14],
                    },
                    "frame_by_effector": {
                        "left_arm": "left_arm_base",
                        "right_arm": "right_arm_base",
                    },
                    "metadata": {
                        "arm_reference_frame_by_side": {
                            "left": "left_arm_base",
                            "right": "right_arm_base",
                        }
                    },
                },
                "action_vector": {"vector": action[:14] + [0.0] * 12},
            }
        }
    )


def _write_dataset(root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    layout = _source_layout()
    left_state = _pose(0.1, 0.2)
    right_state = _pose(-0.1, -0.2)
    action0 = _pose(0.2, 0.1) + _pose(-0.2, -0.1) + [0.3, 0.4]
    action1 = [float("nan")] * 7 + _pose(-0.21, -0.11) + [0.31, 0.41]
    state = left_state + right_state + [0.5, 0.6]
    arm_specs = {
        side: {
            "side": side,
            "controlled_frame": "tool_tcp",
            "reference_frame": "arm_base",
        }
        for side in ("left", "right")
    }
    rows = []
    for index, (action, last_action) in enumerate(
        ((action0, [float("nan")] * 16), (action1, action0))
    ):
        rows.append(
            {
                "index": index,
                "episode_index": 0,
                "frame_index": index,
                "timestamp": float(index),
                "action": action,
                "action.arm.pose": action[:14],
                "action.left_arm.pose": action[:7],
                "action.right_arm.pose": action[7:14],
                "observation.state": state,
                "observation.arm.pose": state[:14],
                "observation.last_action": last_action,
                "observation.left_arm.pose": left_state,
                "observation.right_arm.pose": right_state,
                "observation.arm.left.actual_tcp_pose": left_state,
                "observation.arm.right.actual_tcp_pose": right_state,
                "teleop.action_source.left_wrist_pose_base": action[:7],
                "teleop.action_source.left_wrist_pose_desk": _pose(9.0),
                "arm_action_specs_json": json.dumps(arm_specs),
                "sample_metadata_json": _sample_metadata(action),
                "dexteleop.dataset_schema": DEXTELEOP_DATASET_SCHEMA_V22,
                "policy_io_layout_hash": layout.layout_hash,
            }
        )
    parquet = root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet, compression="zstd")
    (root / "videos/front/chunk-000").mkdir(parents=True)
    (root / "videos/front/chunk-000/episode_000000.mp4").write_bytes(b"video")
    info = {
        "dataset_name": "source",
        "dataset_dir_name": "source",
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V22,
        "episode_count": 1,
        "frame_count": 2,
        "active_sides": ["left", "right"],
        "arm_reference_frame_by_side": {
            "left": "left_arm_base",
            "right": "right_arm_base",
        },
        "arm_action_specs_by_side": arm_specs,
        **layout.binding_metadata(),
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 2,
                "parquet_path": "data/chunk-000/episode_000000.parquet",
                "metadata": layout.binding_metadata(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return info, rows


def test_v22_desk_conversion_is_dry_run_then_copy_on_write(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source_info, source_rows = _write_dataset(source)
    info_before = (source / "meta/info.json").read_bytes()
    parquet_before = (source / "data/chunk-000/episode_000000.parquet").read_bytes()

    plan = plan_v22_to_v23_desk_conversion(source, output)

    assert plan.invalid_action_rows == 1
    assert plan.invalid_last_action_rows == 1
    assert plan.first_row_invalid_last_action_episodes == 1
    assert not output.exists()
    assert (source / "meta/info.json").read_bytes() == info_before
    assert (source / "data/chunk-000/episode_000000.parquet").read_bytes() == parquet_before

    applied = convert_v22_dataset_to_v23_desk(source, output)

    assert applied.conversion_id == plan.conversion_id
    output_info = json.loads((output / "meta/info.json").read_text())
    assert output_info["dexteleop_dataset_schema"] == DEXTELEOP_DATASET_SCHEMA_V23
    assert output_info["pose_storage_frame"] == "desk_absolute"
    assert output_info["v23_compatibility_kind"] == V23_COMPATIBILITY_V22_CANONICAL_DESK
    assert output_info["native_v23_causal_snapshot"] is False
    assert output_info["policy_io_layout_hash"] == plan.target_layout_hash
    assert output_info["policy_io_layout_hash"] != source_info["policy_io_layout_hash"]
    assert resolve_v23_dataset_contract(output_info).compatibility_mode is True
    malformed = dict(output_info)
    malformed["native_v23_causal_snapshot"] = True
    with pytest.raises(ValueError, match="native_v23_causal_snapshot=false"):
        resolve_v23_dataset_contract(malformed)
    assert PoseTrainingDataContract.from_dict(
        output_info["pose_training_data_contract"]
    ) == PoseTrainingDataContract.v22_canonical_desk()

    converted = pq.read_table(
        output / "data/chunk-000/episode_000000.parquet"
    ).to_pylist()
    runtime = default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
    expected_left = runtime.storage_pose_vector_to_canonical_desk(
        source_rows[0]["action"][:7], side="left"
    )
    expected_right = runtime.storage_pose_vector_to_canonical_desk(
        source_rows[0]["action"][7:14], side="right"
    )
    np.testing.assert_allclose(converted[0]["action"][:7], expected_left, atol=1e-6)
    np.testing.assert_allclose(converted[0]["action"][7:14], expected_right, atol=1e-6)
    assert converted[0]["action"][14:] == source_rows[0]["action"][14:]
    assert converted[0]["teleop.action_source.left_wrist_pose_base"] == (
        source_rows[0]["teleop.action_source.left_wrist_pose_base"]
    )
    assert converted[0]["teleop.action_source.left_wrist_pose_desk"] == (
        source_rows[0]["teleop.action_source.left_wrist_pose_desk"]
    )
    assert converted[0]["training.action_finite"] is True
    assert converted[0]["training.relative_last_action_finite"] is False
    assert converted[1]["training.action_finite"] is False
    assert converted[1]["training.relative_last_action_finite"] is True
    assert all(math.isnan(value) for value in converted[0]["observation.last_action"])
    assert all(math.isnan(value) for value in converted[1]["action"][:7])
    embedded = json.loads(converted[0]["sample_metadata_json"])
    embedded_action = embedded["canonical_action_record"]["action"]
    np.testing.assert_allclose(embedded_action["arm_pose_by_side"]["left"], expected_left)
    assert embedded_action["frame_by_effector"]["left_arm"] == "desk"
    assert embedded_action["frame_by_effector"]["right_arm"] == "desk"

    source_video = source / "videos/front/chunk-000/episode_000000.mp4"
    output_video = output / "videos/front/chunk-000/episode_000000.mp4"
    assert (source_video.stat().st_dev, source_video.stat().st_ino) == (
        output_video.stat().st_dev,
        output_video.stat().st_ino,
    )


def test_v22_canonical_desk_training_origin_requires_explicit_binding() -> None:
    contract = PoseTrainingDataContract.v22_canonical_desk()
    assert contract.runtime_pose_binding == "explicit_runtime_raw_to_canonical_desk"
    assert contract.expected_runtime_source_id is None
    assert PoseTrainingDataContract.from_dict(contract.to_dict()) == contract

    runtime = default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
    artifact = PolicyArtifactContract(
        policy_io_layout=_source_layout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=PoseActionChunkContract(action_horizon=2),
        pose_training_data=contract,
        pose_frame_contract_hashes=(runtime.contract_hash,),
    )
    with pytest.raises(ValueError, match="requires an explicit"):
        resolve_deploy_session_contract(
            active_sides=("left", "right"),
            policy_artifact=artifact,
            hardware_output=False,
        )
    session = resolve_deploy_session_contract(
        active_sides=("left", "right"),
        policy_artifact=artifact,
        runtime_pose_frame_contract=runtime,
        hardware_output=False,
    )
    assert session.metadata["runtime_pose_identity_compatibility"] is False
