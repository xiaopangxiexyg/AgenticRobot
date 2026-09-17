from __future__ import annotations

import numpy as np
import pytest

from teleop_runtime.skill_deploy.grasp_trajectory_record import (
    load_grasp_trajectory_record,
    write_grasp_trajectory_record,
)
from teleop_runtime.skill_deploy.motion_port import RealAgentJointTrajectory

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _trajectory() -> RealAgentJointTrajectory:
    return RealAgentJointTrajectory(
        time_from_start_sec=np.array([0.0, 0.2, 0.4]),
        qpos_by_side={
            "left": np.zeros((3, 6)),
            "right": np.arange(18, dtype=float).reshape(3, 6) * 0.01,
        },
        planner_evidence_refs=("rm65-grasp-path:test",),
    )


def _write(directory, trajectory):
    return write_grasp_trajectory_record(
        directory,
        trajectory,
        moving_side="right",
        scene_hash="scene:test",
        collision_scope="profile_sparse_proxy_pairs",
        collision_scene_profile="dual_arm_hand_proxy_strict",
        target_tool_tcp_world=np.eye(4),
        speed_scale=0.2,
        stage_boundaries=(
            {"planning_phase": "pregrasp", "end_sample_index": 1, "end_time_sec": 0.2},
            {"planning_phase": "local_approach", "end_sample_index": 2, "end_time_sec": 0.4},
        ),
    )


def test_exact_grasp_plan_samples_survive_restart_and_are_immutable(tmp_path):
    original = _trajectory()
    path = _write(tmp_path, original)
    stat_before = path.stat()
    restored, metadata = load_grasp_trajectory_record(path)

    assert restored.semantic_content_hash == original.semantic_content_hash
    np.testing.assert_array_equal(restored.time_from_start_sec, original.time_from_start_sec)
    for side in ("left", "right"):
        np.testing.assert_array_equal(restored.qpos_by_side[side], original.qpos_by_side[side])
    assert metadata["stage_boundaries"][-1]["end_sample_index"] == 2
    assert metadata["collision_scope"] == "profile_sparse_proxy_pairs"
    assert metadata["collision_scene_profile"] == "dual_arm_hand_proxy_strict"
    assert _write(tmp_path, original) == path
    assert path.stat().st_ino == stat_before.st_ino


def test_existing_corrupt_record_is_not_overwritten(tmp_path):
    original = _trajectory()
    path = tmp_path / f"{original.semantic_content_hash}.npz"
    path.write_bytes(b"broken")

    with pytest.raises(ValueError):
        _write(tmp_path, original)
    assert path.read_bytes() == b"broken"
    assert not list(tmp_path.glob(".grasp-plan-*.npz"))
