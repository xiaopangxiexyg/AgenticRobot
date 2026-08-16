from __future__ import annotations

import json

import numpy as np
import pytest

from dataset_contracts.pose_frames import (
    POSE_FRAME_CONTRACT_METADATA_KEY,
    PoseFrameContract,
    PoseFrameContractError,
    PoseSourceProfile,
)
from dataset_contracts.v22 import PolicyIOLayout
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_contracts.v23_validation import (
    V23RowValidationError,
    validate_v23_row_causal_snapshots,
)
from dataset_interface.collection_pose_storage import (
    bind_collection_row_pose_storage,
    validate_v23_collection_row_actual_tcp,
)
from dataset_interface.validation import validate_v23_dataset
from teleop_runtime.collection.pose_storage import (
    bind_collection_row_pose_storage as legacy_bind_pose_storage,
    validate_v23_collection_row_actual_tcp as legacy_validate_actual_tcp,
)
from teleop_runtime.collection.sample_assembler import CollectorSample
from teleop_runtime.collection.writer_v21 import (
    TACTILE_FRAME_JOB_KIND,
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
)


def test_legacy_pose_storage_imports_are_object_identical_data_exports() -> None:
    assert (
        legacy_bind_pose_storage
        is bind_collection_row_pose_storage
    )
    assert (
        legacy_validate_actual_tcp
        is validate_v23_collection_row_actual_tcp
    )
    assert (
        bind_collection_row_pose_storage.__module__
        == "dataset_interface.collection_pose_storage"
    )


def _translation(x: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = x
    return transform


def _pose7(x: float) -> list[float]:
    return [x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _source(*, left_x: float = 1.0, right_x: float = 2.0) -> PoseSourceProfile:
    return PoseSourceProfile(
        source_id="rm65_real_dual_arm_test_v1",
        source_family="real_robot",
        source_description="test fixture",
        topology="dual_arm",
        raw_absolute_frame_by_side={
            "left": "left_rm65_base",
            "right": "right_rm65_base",
        },
        raw_to_desk_transform_by_side={
            "left": _translation(left_x),
            "right": _translation(right_x),
        },
    )


def _row() -> dict[str, object]:
    arm_prefix = _pose7(0.1) + _pose7(0.2)
    return {
        "dexteleop.dataset_schema": "dexteleop_dataset_v2.2",
        "action": arm_prefix + [0.0] * 12,
        "action.arm.pose": list(arm_prefix),
        "action.left_arm.pose": _pose7(0.1),
        "action.right_arm.pose": _pose7(0.2),
        "action.reference.left_arm.pose_ref": _pose7(0.15),
        "observation.state": arm_prefix + [0.0] * 24,
        "observation.arm.pose": list(arm_prefix),
        "observation.last_action": arm_prefix + [0.0] * 12,
        "observation.left_arm.pose": _pose7(0.1),
        "observation.right_arm.pose": _pose7(0.2),
        "observation.arm.left.actual_tcp_pose": _pose7(0.3),
        "observation.arm.right.actual_tcp_pose": _pose7(0.4),
        "observation.arm.left.actual_hardware_pose_6d": [0.5, 0, 0, 0, 0, 0],
        "observation.wrist_camera.left.pose_base": _pose7(0.6),
        "teleop.action_source.left_wrist_pose_desk": _pose7(0.7),
        "teleop.action_source.left_wrist_pose_desk_frame": "vive_retarget_frame",
    }


def _valid_v23_feedback_row() -> dict[str, object]:
    row = _row()
    row["timestamp"] = 10.0
    for side, pose_x in (("left", 0.3), ("right", 0.4)):
        start = 0 if side == "left" else 7
        for key in ("observation.state", "observation.arm.pose"):
            values = list(row[key])
            values[start : start + 7] = _pose7(pose_x)
            row[key] = values
        prefix = f"observation.arm.{side}."
        row[f"observation.{side}_arm.pose"] = _pose7(pose_x)
        row[f"{prefix}actual_tcp_pose"] = _pose7(pose_x)
        row[f"{prefix}actual_tcp_pose_source"] = "hardware_qpos_fk"
        row[f"{prefix}actual_hardware_qpos"] = [0.0] * 6
        row[f"{prefix}actual_state_read_success"] = True
        row[f"{prefix}actual_state_valid"] = True
        row[f"{prefix}actual_state_stale"] = False
        row[f"{prefix}actual_state_sequence"] = 7
        row[f"{prefix}actual_state_timestamp"] = 9.98
        row[f"{prefix}source_timestamp"] = 9.98
        row[f"{prefix}available_timestamp"] = 9.99
        row[f"{prefix}source_clock"] = "local_wall_time"
        row[f"{prefix}available_clock"] = "local_wall_time"
        row[f"{prefix}causal_measured_state_eligible"] = True
        row[f"{prefix}causal_measured_state_rejection_reason"] = "accepted"
    return row


def _sample_with_feedback(*, available_timestamp: float = 9.99) -> CollectorSample:
    arm_states: dict[str, dict[str, object]] = {}
    observation_state = np.zeros(38, dtype=np.float32)
    action = np.zeros(26, dtype=np.float32)
    action[6] = 1.0
    action[13] = 1.0
    for side, pose_x in (("left", 0.3), ("right", 0.4)):
        pose = _translation(pose_x)
        arm_states[side] = {
            "current_qpos": np.zeros(6, dtype=float),
            "current_tcp_pose": pose,
            "actual_qpos": np.zeros(6, dtype=float),
            "actual_hardware_qpos": np.zeros(6, dtype=float),
            "actual_tcp_pose": pose,
            "actual_tcp_pose_source": "hardware_qpos_fk",
            "actual_state_read_success": True,
            "actual_state_valid": True,
            "actual_state_stale": False,
            "actual_state_sequence": 7,
            "actual_state_timestamp": 9.98,
            "source_timestamp": 9.98,
            "available_timestamp": available_timestamp,
            "source_clock": "local_wall_time",
            "available_clock": "local_wall_time",
            "causal_measured_state_eligible": True,
            "causal_measured_state_rejection_reason": "accepted",
        }
        state_slice = slice(6, 13) if side == "left" else slice(25, 32)
        observation_state[state_slice] = np.asarray(_pose7(pose_x), dtype=np.float32)
    return CollectorSample(
        index=0,
        episode_index=0,
        frame_index=0,
        timestamp=10.0,
        session_state="running",
        control_hz=15.0,
        action=action,
        observation_state=observation_state,
        observation_last_action=action.copy(),
        arm_state_source="causal_measured_hardware",
        active_sides=("left", "right"),
        arm_states=arm_states,
        hand_states={"left": {}, "right": {}},
    )


def test_writer_row_adapter_converts_only_detached_standard_arm_pose_copies() -> None:
    contract = PoseFrameContract.v23_default_desk(_source())
    source_row = _row()
    output = bind_collection_row_pose_storage(source_row, contract=contract)

    assert output is not source_row
    assert source_row["action"][0] == pytest.approx(0.1)
    assert output["action"][0] == pytest.approx(1.1)
    assert output["action"][7] == pytest.approx(2.2)
    assert output["action.left_arm.pose"][0] == pytest.approx(1.1)
    assert output["action.right_arm.pose"][0] == pytest.approx(2.2)
    assert output["observation.state"][0] == pytest.approx(1.1)
    assert output["observation.last_action"][7] == pytest.approx(2.2)
    assert output["observation.arm.left.actual_tcp_pose"][0] == pytest.approx(1.3)
    assert output["observation.arm.right.actual_tcp_pose"][0] == pytest.approx(2.4)

    # Controller/base audit fields intentionally retain their declared raw frame.
    assert output["observation.arm.left.actual_hardware_pose_6d"] == source_row[
        "observation.arm.left.actual_hardware_pose_6d"
    ]
    assert output["observation.wrist_camera.left.pose_base"] == source_row[
        "observation.wrist_camera.left.pose_base"
    ]
    assert output["teleop.action_source.left_wrist_pose_desk"] == source_row[
        "teleop.action_source.left_wrist_pose_desk"
    ]
    assert output["teleop.action_source.left_wrist_pose_desk_frame"] == (
        "vive_retarget_frame"
    )
    assert output["action.reference.left_arm.pose_ref"] == source_row[
        "action.reference.left_arm.pose_ref"
    ]
    assert output["action.reference.left_arm.pose_ref_frame"] == "left_rm65_base"
    assert output["dexteleop.dataset_schema"] == DEXTELEOP_DATASET_SCHEMA_V23
    assert output["dexteleop.pose_storage_frame"] == "desk_absolute"


def test_raw_storage_is_label_only_and_does_not_change_pose_values() -> None:
    contract = PoseFrameContract.v23_forced_raw(_source())
    source_row = _row()
    output = bind_collection_row_pose_storage(source_row, contract=contract)
    assert output["action"] == source_row["action"]
    assert output["observation.arm.left.actual_tcp_pose"] == source_row[
        "observation.arm.left.actual_tcp_pose"
    ]
    assert output["dexteleop.pose_storage_frame"] == "raw_absolute"


def test_writer_freezes_pose_contract_in_dataset_and_episode_metadata(tmp_path) -> None:
    contract = PoseFrameContract.v23_forced_raw(_source())
    writer = LeRobotV21Writer(
        LeRobotV21WriterConfig(
            root_dir=tmp_path,
            write_video=False,
            policy_io_layout=PolicyIOLayout(
                arm_reference_frame_by_side={
                    "left": "left_rm65_base",
                    "right": "right_rm65_base",
                }
            ),
            pose_frame_contract=contract,
        )
    )
    info = json.loads((tmp_path / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["dexteleop_dataset_schema"] == DEXTELEOP_DATASET_SCHEMA_V23
    assert info[POSE_FRAME_CONTRACT_METADATA_KEY] == contract.to_dict()

    writer.start_episode(metadata={"active_sides": ["left", "right"]})
    episode_metadata = writer._episodes[0].metadata
    assert episode_metadata[POSE_FRAME_CONTRACT_METADATA_KEY] == contract.to_dict()
    assert episode_metadata["pose_storage_frame"] == "raw_absolute"

    conflicting = PoseFrameContract.v23_forced_raw(_source(left_x=9.0))
    with pytest.raises(PoseFrameContractError, match="differs"):
        writer._ensure_episode_state(
            1,
            metadata={POSE_FRAME_CONTRACT_METADATA_KEY: conflicting.to_dict()},
        )
    with pytest.raises(PoseFrameContractError, match="causal_snapshot_contract_hash"):
        writer._ensure_episode_state(
            2,
            metadata={"causal_snapshot_contract_hash": "sha256:wrong"},
        )


def test_v23_writer_refuses_append_with_different_causal_contract(tmp_path) -> None:
    contract = PoseFrameContract.v23_forced_raw(_source())
    config = LeRobotV21WriterConfig(
        root_dir=tmp_path,
        write_video=False,
        policy_io_layout=PolicyIOLayout(
            arm_reference_frame_by_side={
                "left": "left_rm65_base",
                "right": "right_rm65_base",
            }
        ),
        pose_frame_contract=contract,
    )
    LeRobotV21Writer(config)
    info_path = tmp_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["frame_count"] = 1
    info["causal_snapshot_contract_hash"] = "sha256:wrong"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(PoseFrameContractError, match="causal_snapshot_contract_hash"):
        LeRobotV21Writer(config)


def test_v23_writer_requires_v22_compatible_layout(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires.*policy_io_layout"):
        LeRobotV21WriterConfig(
            root_dir=tmp_path,
            pose_frame_contract=PoseFrameContract.v23_forced_raw(_source()),
        )


def test_v23_actual_tcp_validator_accepts_complete_causal_feedback() -> None:
    validate_v23_collection_row_actual_tcp(
        _valid_v23_feedback_row(),
        contract=PoseFrameContract.v23_default_desk(_source()),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actual_tcp_pose", None, "missing_or_nonfinite_actual_tcp_pose"),
        ("actual_tcp_pose_source", "current_tcp_pose", "not_from_hardware_qpos_fk"),
        ("actual_state_stale", True, "actual_state_stale_or_unconfirmed"),
        ("available_timestamp", 10.01, "available_timestamp_after_T0"),
    ],
)
def test_v23_actual_tcp_validator_rejects_invalid_or_noncausal_feedback(
    field: str,
    value: object,
    message: str,
) -> None:
    row = _valid_v23_feedback_row()
    row[f"observation.arm.left.{field}"] = value

    with pytest.raises(PoseFrameContractError, match=message):
        validate_v23_collection_row_actual_tcp(
            row,
            contract=PoseFrameContract.v23_default_desk(_source()),
        )


def test_v23_writer_validates_feedback_before_planning_any_jobs(tmp_path) -> None:
    writer = LeRobotV21Writer(
        LeRobotV21WriterConfig(
            root_dir=tmp_path,
            write_parquet=False,
            write_video=False,
            policy_io_layout=PolicyIOLayout(
                arm_reference_frame_by_side={
                    "left": "left_rm65_base",
                    "right": "right_rm65_base",
                }
            ),
            pose_frame_contract=PoseFrameContract.v23_forced_raw(_source()),
        )
    )

    assert writer.plan_sample_jobs(_sample_with_feedback())
    with pytest.raises(PoseFrameContractError, match="available_timestamp_after_T0"):
        writer.plan_sample_jobs(_sample_with_feedback(available_timestamp=10.01))


def test_rejected_v23_row_does_not_commit_tactile_dedup_state(tmp_path) -> None:
    writer = LeRobotV21Writer(
        LeRobotV21WriterConfig(
            root_dir=tmp_path,
            write_parquet=False,
            write_video=False,
            policy_io_layout=PolicyIOLayout(
                arm_reference_frame_by_side={
                    "left": "left_rm65_base",
                    "right": "right_rm65_base",
                }
            ),
            pose_frame_contract=PoseFrameContract.v23_forced_raw(_source()),
        )
    )

    def with_tactile(sample: CollectorSample) -> CollectorSample:
        sample.hand_states["left"] = {
            "tactile_dict": {"raw": np.arange(4, dtype=np.uint8).reshape(2, 2)},
            "tactile_snapshot_sequence": 8,
            "tactile_snapshot_source_timestamp": 9.8,
            "tactile_snapshot_available_timestamp": 9.9,
            "tactile_snapshot_source_clock": "local_wall_time",
            "tactile_snapshot_available_clock": "local_wall_time",
        }
        return sample

    with pytest.raises(PoseFrameContractError):
        writer.plan_sample_jobs(
            with_tactile(_sample_with_feedback(available_timestamp=10.01))
        )
    jobs = writer.plan_sample_jobs(with_tactile(_sample_with_feedback()))
    assert sum(job.kind == TACTILE_FRAME_JOB_KIND for job in jobs) == 1


def test_v23_causal_validator_rejects_future_hand_snapshot() -> None:
    row = _valid_v23_feedback_row()
    prefix = "observation.hand.left.telemetry_snapshot_"
    row.update(
        {
            f"{prefix}sequence": 3,
            f"{prefix}source_timestamp": 10.01,
            f"{prefix}available_timestamp": 9.99,
            f"{prefix}source_clock": "local_wall_time",
            f"{prefix}available_clock": "local_wall_time",
        }
    )
    with pytest.raises(V23RowValidationError, match="source_timestamp_after_T0"):
        validate_v23_row_causal_snapshots(row)


@pytest.mark.optional
def test_native_v23_dataset_acceptance_scan(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    contract = PoseFrameContract.v23_forced_raw(_source())
    writer = LeRobotV21Writer(
        LeRobotV21WriterConfig(
            root_dir=tmp_path,
            write_video=False,
            policy_io_layout=PolicyIOLayout(
                arm_reference_frame_by_side={
                    "left": "left_rm65_base",
                    "right": "right_rm65_base",
                }
            ),
            pose_frame_contract=contract,
        )
    )
    writer.start_episode(metadata={"active_sides": ["left", "right"]})
    for job in writer.plan_sample_jobs(_sample_with_feedback()):
        writer.handle_job(job)
    writer.finalize_episode(0)

    report = validate_v23_dataset(tmp_path)
    assert report.ok, report.errors
    assert report.episode_count == 1
    assert report.frame_count == 1

    episodes_path = tmp_path / "meta" / "episodes.jsonl"
    episode = json.loads(episodes_path.read_text(encoding="utf-8"))
    video_relative = "videos/observation.images.front/chunk-000/episode_000000.mp4"
    (tmp_path / video_relative).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / video_relative).write_bytes(b"fixture")
    episode["video_paths"] = {"front": video_relative}
    episode["summary"] = {
        "video_frame_stats": {
            "front": {
                "path": "videos/observation.images.front/chunk-000/episode_000001.mp4"
            }
        }
    }
    episodes_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    stale_summary = validate_v23_dataset(tmp_path)
    assert not stale_summary.ok
    assert any("summary.video_frame_stats.front.path differs" in error for error in stale_summary.errors)

    info_path = tmp_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["causal_snapshot_contract_hash"] = "sha256:wrong"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    invalid = validate_v23_dataset(tmp_path)
    assert not invalid.ok
    assert any("causal_snapshot_contract_hash mismatch" in error for error in invalid.errors)
