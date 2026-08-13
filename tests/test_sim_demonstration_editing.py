from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from sim.demonstration_editing import (
    DemonstrationEdit,
    DemonstrationTrajectory,
    EditedDemonstrationTrajectory,
    apply_demonstration_edit,
    preflight_zero_edit_replay,
    sample_edited_trajectory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "sim" / "demonstration_editing"


def _trajectory(*, hand_dim: int = 1) -> DemonstrationTrajectory:
    hand = np.stack(
        (
            np.zeros(hand_dim),
            np.full(hand_dim, 0.5),
            np.ones(hand_dim),
        ),
        axis=0,
    )
    return DemonstrationTrajectory(
        trajectory_id="nominal-grasp-1",
        reference_frame="object_initial",
        controlled_frame="right_tool_tcp",
        hand_component_id="right_gripper" if hand_dim == 1 else "right_hand",
        hand_target_representation=(
            "normalized_close" if hand_dim == 1 else "physical_joint_qpos"
        ),
        hand_target_names=(
            ("close",)
            if hand_dim == 1
            else tuple(f"joint_{index}" for index in range(hand_dim))
        ),
        hand_target_units=(
            ("unit_interval",)
            if hand_dim == 1
            else tuple("radian" for _ in range(hand_dim))
        ),
        timestamps_sec=np.asarray([0.0, 0.5, 1.0]),
        tool_pose_xyzquat_xyzw=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        hand_targets=hand,
        hand_edit_start_index=0,
        hand_edit_end_index=1,
        source_kind="sim_hil_human_teleop",
        source_id="control-trace-session-1",
        source_content_hash="1" * 64,
        metadata={"authority": "human_only"},
    )


def _edit(*, hand_dim: int = 1) -> DemonstrationEdit:
    return DemonstrationEdit(
        edit_id="edit-1",
        producer_id="demo-editor-policy-1",
        sample_id="ppo-sample-17",
        translation_m=np.asarray([0.0, 1.0, 0.0]),
        rotation_rotvec_rad=np.asarray([0.0, 0.0, np.pi / 2.0]),
        hand_final_delta=np.full(hand_dim, 0.2),
    )


def test_contract_roundtrip_is_hash_stable_and_arrays_are_read_only() -> None:
    trajectory = _trajectory()
    restored = DemonstrationTrajectory.from_dict(trajectory.to_dict())
    edit = _edit()
    restored_edit = DemonstrationEdit.from_dict(edit.to_dict())

    assert restored.semantic_content_hash == trajectory.semantic_content_hash
    assert restored_edit.semantic_content_hash == edit.semantic_content_hash
    assert restored.step_count == 3
    assert restored.hand_dim == 1
    with pytest.raises(ValueError):
        restored.timestamps_sec[0] = 1.0
    with pytest.raises(ValueError):
        restored_edit.translation_m[0] = 1.0


def test_edit_left_multiplies_pose_and_ramps_final_hand_delta() -> None:
    trajectory = _trajectory()
    edit = _edit()
    result = apply_demonstration_edit(trajectory, edit)

    np.testing.assert_allclose(
        result.tool_pose_xyzquat_xyzw[:, :3],
        np.asarray(
            [
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        ),
        atol=1e-12,
    )
    expected_quaternion = np.asarray([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    np.testing.assert_allclose(
        result.tool_pose_xyzquat_xyzw[:, 3:7],
        np.repeat(expected_quaternion[None, :], 3, axis=0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.hand_targets[:, 0],
        np.asarray([0.0, 0.7, 1.2]),
    )
    assert result.source_trajectory_hash == trajectory.semantic_content_hash
    assert result.edit_hash == edit.semantic_content_hash
    assert result.edit_sample_id == edit.sample_id


def test_identity_edit_is_nominal_but_keeps_distinct_lineage() -> None:
    trajectory = _trajectory(hand_dim=2)
    identity = DemonstrationEdit(
        edit_id="identity",
        producer_id="zero-edit-probe",
        sample_id="deterministic-zero",
        translation_m=np.zeros(3),
        rotation_rotvec_rad=np.zeros(3),
        hand_final_delta=np.zeros(2),
    )
    result = apply_demonstration_edit(trajectory, identity)

    np.testing.assert_allclose(
        result.tool_pose_xyzquat_xyzw,
        trajectory.tool_pose_xyzquat_xyzw,
    )
    np.testing.assert_allclose(
        result.hand_targets,
        trajectory.hand_targets,
    )
    assert result.semantic_content_hash not in {
        trajectory.semantic_content_hash,
        identity.semantic_content_hash,
    }


def test_zero_edit_preflight_replays_every_nominal_keyframe_only() -> None:
    trajectory = _trajectory(hand_dim=2)
    result = preflight_zero_edit_replay(trajectory, sample_id="zero-probe-1")

    assert result.accepted
    assert result.sample_id == "zero-probe-1"
    assert result.source_trajectory_hash == trajectory.semantic_content_hash
    assert len(result.references) == trajectory.step_count
    assert result.max_position_error_m == pytest.approx(0.0)
    assert result.max_rotation_error_rad == pytest.approx(0.0)
    assert result.max_hand_error == pytest.approx(0.0)
    for index, reference in enumerate(result.references):
        np.testing.assert_allclose(
            reference.tool_pose_xyzquat_xyzw,
            trajectory.tool_pose_xyzquat_xyzw[index],
        )
        np.testing.assert_allclose(
            reference.hand_target,
            trajectory.hand_targets[index],
        )


def test_time_sampling_interpolates_pose_hand_and_preserves_lineage() -> None:
    edited = apply_demonstration_edit(_trajectory(), _edit())
    reference = sample_edited_trajectory(edited, 0.25)

    assert reference.left_index == 0
    assert reference.right_index == 1
    assert reference.interpolation_alpha == pytest.approx(0.5)
    np.testing.assert_allclose(
        reference.tool_pose_xyzquat_xyzw[:3],
        np.asarray([0.0, 2.5, 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(reference.hand_target, np.asarray([0.35]))
    assert reference.source_trajectory_hash == edited.source_trajectory_hash
    assert reference.edit_hash == edited.edit_hash
    assert reference.edit_sample_id == edited.edit_sample_id
    assert reference.edited_trajectory_hash == edited.semantic_content_hash
    assert reference.reference_frame == "object_initial"
    assert reference.controlled_frame == "right_tool_tcp"
    assert reference.hand_component_id == "right_gripper"
    assert reference.hand_target_representation == "normalized_close"
    assert reference.hand_target_names == ("close",)
    assert reference.hand_target_units == ("unit_interval",)

    held = sample_edited_trajectory(edited, 5.0)
    assert held.requested_time_sec == 5.0
    assert held.resolved_time_sec == edited.duration_sec
    assert held.left_index == held.right_index == 2
    np.testing.assert_allclose(
        held.tool_pose_xyzquat_xyzw,
        edited.tool_pose_xyzquat_xyzw[-1],
    )


def test_edit_rejects_hand_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="hand dimension"):
        apply_demonstration_edit(_trajectory(hand_dim=1), _edit(hand_dim=2))


def test_hand_target_representation_is_hash_bound_and_dimension_checked() -> None:
    trajectory = _trajectory()
    changed_payload = trajectory.to_dict()
    changed_payload["hand_target_units"] = ["normalized"]
    changed = DemonstrationTrajectory.from_dict(changed_payload)

    assert changed.semantic_content_hash != trajectory.semantic_content_hash

    invalid_payload = trajectory.to_dict()
    invalid_payload["hand_target_names"] = ["close", "unused"]
    with pytest.raises(ValueError, match="must match the hand target dimension"):
        DemonstrationTrajectory.from_dict(invalid_payload)


def test_edited_contract_roundtrip_retains_lineage() -> None:
    edited = apply_demonstration_edit(_trajectory(), _edit())
    restored = EditedDemonstrationTrajectory.from_dict(edited.to_dict())

    assert restored.semantic_content_hash == edited.semantic_content_hash
    assert restored.source_trajectory_hash == edited.source_trajectory_hash
    assert restored.edit_hash == edited.edit_hash
    assert restored.edit_sample_id == edited.edit_sample_id


def test_package_does_not_import_engine_training_or_runtime_owners() -> None:
    forbidden_roots = {
        "control_station",
        "mujoco",
        "teleop_runtime",
        "torch",
    }
    forbidden_modules = {
        "sim.mujoco",
        "sim.mujoco_temp_pipeline",
    }
    violations: dict[str, list[str]] = {}
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        bad = sorted(
            name
            for name in imported
            if name.split(".", 1)[0] in forbidden_roots
            or any(name == module or name.startswith(module + ".") for module in forbidden_modules)
        )
        if bad:
            violations[str(path.relative_to(REPO_ROOT))] = bad
    assert violations == {}
