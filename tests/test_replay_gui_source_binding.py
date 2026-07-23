from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_contracts.replay import (
    REPLAY_SOURCE_INTERPRETATION_UNSUPPORTED,
    REPLAY_SOURCE_INTERPRETATION_VERIFIED,
)
from dataset_contracts.v24 import DEXTELEOP_DATASET_SCHEMA_V24
from robot_motion.deploy import ACTION_ARM_POSE_SLICES, DEPLOY_ACTION_DIM, nan_vector
from robot_motion.replay import ReplayTrajectoryGroup
from teleop_runtime.replay.source_binding import build_replay_gui_source_binding


def _group(
    path: Path,
    *,
    component_schema_hash: str = "source-components",
    pose_frame_contract_hash: str | None = "source-frame",
) -> ReplayTrajectoryGroup:
    action = nan_vector(DEPLOY_ACTION_DIM)
    identity = np.asarray(
        [0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    action[ACTION_ARM_POSE_SLICES["left"]] = identity
    action[ACTION_ARM_POSE_SLICES["right"]] = identity
    dataset_info = {
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V24,
        "dataset_id": "checked-replay-source",
        "component_schema_hash": component_schema_hash,
        "action_spec_hash_by_component": {
            "left_arm": "rm65-tool-tcp-pose-v1",
            "right_arm": "rm65-tool-tcp-pose-v1",
        },
        "units_convention_id": "si_m_rad_xyzw_v1",
    }
    if pose_frame_contract_hash is not None:
        dataset_info["pose_frame_contract_hash"] = pose_frame_contract_hash
    return ReplayTrajectoryGroup(
        key="g",
        label="checked replay",
        source_path=str(path),
        source_episode_index=4,
        start_index=7,
        end_index=8,
        frame_count=1,
        action_hz=15.0,
        actions=np.asarray([action], dtype=np.float32),
        frame_times_sec=(0.0,),
        metadata={
            "source_metadata": {
                "dataset_info": dataset_info,
                "episode_metadata": {"episode_role": "fixture"},
            },
            "action_column": "selected_action",
            "action_metadata": {"pose_action_absolute_frame": "arm_base"},
            "frame_timing": {"source_timestamp_mode": "dataset"},
        },
    )


def test_checked_source_binding_uses_preflight_group_without_path_identity(
    tmp_path: Path,
) -> None:
    first = build_replay_gui_source_binding(
        _group(tmp_path / "first" / "episode.parquet"),
        episode_id="episode_000004",
        active_sides=("left", "right"),
        include_end_effectors=False,
        trajectory_source="action_pose",
    )
    second = build_replay_gui_source_binding(
        _group(tmp_path / "relocated" / "episode.parquet"),
        episode_id="episode_000004",
        active_sides=("left", "right"),
        include_end_effectors=False,
        trajectory_source="action_pose",
    )

    assert first.source_action_contract.interpretation_status == (
        REPLAY_SOURCE_INTERPRETATION_VERIFIED
    )
    assert first.source_action_contract.selection_group_by_component == {
        "left_arm": "arms",
        "right_arm": "arms",
    }
    assert first.source_action_contract.coupled_selection_groups == ("arms",)
    assert first.source_start_index == 7
    assert first.schedule.source_start_index == 7
    assert first.expected_source_path != second.expected_source_path
    assert first.semantic_content_hash == second.semantic_content_hash


def test_checked_source_binding_keeps_missing_source_frame_unresolved(
    tmp_path: Path,
) -> None:
    binding = build_replay_gui_source_binding(
        _group(tmp_path / "episode.parquet", pose_frame_contract_hash=None),
        episode_id="episode_000004",
        active_sides=("left", "right"),
        include_end_effectors=False,
        trajectory_source="action_pose",
    )

    assert binding.source_action_contract.interpretation_status == (
        REPLAY_SOURCE_INTERPRETATION_UNSUPPORTED
    )
    assert "pose_frame_contract" in (
        binding.source_action_contract.unresolved_dimensions
    )


def test_checked_source_binding_rejects_qpos_branch_without_emission_seam(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no source-frame emission seam"):
        build_replay_gui_source_binding(
            _group(tmp_path / "episode.parquet"),
            episode_id="episode_000004",
            active_sides=("left", "right"),
            include_end_effectors=False,
            trajectory_source="qpos_target",
        )


def test_checked_source_binding_metadata_is_detached(tmp_path: Path) -> None:
    group = _group(tmp_path / "episode.parquet")
    binding = build_replay_gui_source_binding(
        group,
        episode_id="episode_000004",
        active_sides=("left", "right"),
        include_end_effectors=False,
        trajectory_source="action_pose",
    )
    group.metadata["source_metadata"]["dataset_info"]["dataset_id"] = "mutated"

    assert binding.metadata_copy()["dataset_id"] == "checked-replay-source"
    with pytest.raises(TypeError):
        binding.dataset_metadata["dataset_id"] = "forbidden"  # type: ignore[index]
