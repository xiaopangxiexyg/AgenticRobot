# Copyright (c) 2026 DexTeleop Contributors
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sim.mujoco.deploy.target_measurement import (
    build_mujoco_sim_hil_coordinate_profiles,
)


class _Scene:
    def __init__(self, urdf_path) -> None:
        self.conversion = SimpleNamespace(
            urdf_path=urdf_path,
            scene_profile_name="portable_scene_v1",
        )

    @staticmethod
    def has_body(name: str) -> bool:
        return name == "right_arm_base_link"

    @staticmethod
    def body_pose(_name: str) -> np.ndarray:
        return np.eye(4, dtype=np.float64)


def _profiles(root, *, urdf_text="<robot name='portable'/>"):
    urdf_path = root / "assets" / "robot.urdf"
    tool_path = root / "calibrations" / "right_tool.json"
    urdf_path.parent.mkdir(parents=True)
    tool_path.parent.mkdir(parents=True)
    urdf_path.write_text(urdf_text, encoding="utf-8")
    tool_path.write_text('{"tcp": [0.0, 0.0, 0.1]}', encoding="utf-8")
    task = SimpleNamespace(
        name="portable_task_v1",
        left_tool_profile_path=tool_path,
        right_tool_profile_path=tool_path,
    )
    return build_mujoco_sim_hil_coordinate_profiles(
        _Scene(urdf_path),
        task,
        active_sides=("right",),
    )


def test_coordinate_profile_identity_is_independent_of_checkout_path(
    tmp_path,
) -> None:
    first_target, first_tool = _profiles(tmp_path / "checkout-a")
    second_target, second_tool = _profiles(tmp_path / "checkout-b")

    assert first_target.artifact_id == (
        "dexteleop.mujoco.target_urdf.v1:portable_scene_v1"
    )
    assert first_tool.bindings[0].artifact_id == (
        "dexteleop.mujoco.tool_profile.v1:portable_task_v1:right"
    )
    assert first_target.artifact_hash == second_target.artifact_hash
    assert first_tool.bindings[0].artifact_hash == (
        second_tool.bindings[0].artifact_hash
    )
    assert first_target.semantic_content_hash == (
        second_target.semantic_content_hash
    )
    assert first_tool.semantic_content_hash == second_tool.semantic_content_hash
    assert str(tmp_path) not in repr(first_target.to_payload())
    assert str(tmp_path) not in repr(first_tool.to_payload())


def test_logical_artifact_identity_does_not_hide_content_drift(tmp_path) -> None:
    first_target, _first_tool = _profiles(
        tmp_path / "checkout-a",
        urdf_text="<robot name='portable-a'/>",
    )
    second_target, _second_tool = _profiles(
        tmp_path / "checkout-b",
        urdf_text="<robot name='portable-b'/>",
    )

    assert first_target.artifact_id == second_target.artifact_id
    assert first_target.artifact_hash != second_target.artifact_hash
    assert first_target.semantic_content_hash != second_target.semantic_content_hash
