# Copyright (c) 2026 DexTeleop Contributors
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dataset_contracts.pose_frames import load_pose_frame_contract
from sim.mujoco.deploy.target_config import DEFAULT_STATION_POSE_CONTRACT_PATH
from sim.mujoco.deploy.target_measurement import (
    MUJOCO_SIM_HIL_STATION_FRAME_ID_V1,
    MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1,
    build_mujoco_sim_hil_coordinate_profiles,
)


class _Scene:
    def __init__(self, urdf_path, *, world_from_base=None) -> None:
        self.conversion = SimpleNamespace(
            urdf_path=urdf_path,
            scene_profile_name="portable_scene_v1",
        )
        self._world_from_base = (
            np.eye(4, dtype=np.float64)
            if world_from_base is None
            else np.asarray(world_from_base, dtype=np.float64)
        )

    @staticmethod
    def has_body(name: str) -> bool:
        return name == "right_arm_base_link"

    def body_pose(self, _name: str) -> np.ndarray:
        return self._world_from_base.copy()


def _profiles(
    root,
    *,
    urdf_text="<robot name='portable'/>",
    world_from_base=None,
    station_pose_contract_path=None,
):
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
        _Scene(urdf_path, world_from_base=world_from_base),
        task,
        active_sides=("right",),
        station_pose_contract_path=station_pose_contract_path,
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


def test_coordinate_profile_exposes_world_to_canonical_station_adapter(
    tmp_path,
) -> None:
    contract = load_pose_frame_contract(DEFAULT_STATION_POSE_CONTRACT_PATH)
    station_from_base = (
        contract.source.effective_raw_to_canonical_desk_transform("right")
    )
    world_from_station = np.eye(4, dtype=np.float64)
    world_from_station[:3, :3] = np.diag((-1.0, -1.0, 1.0))
    world_from_station[:3, 3] = (-0.472, 0.0, 0.01)
    target, _tool = _profiles(
        tmp_path / "checkout",
        world_from_base=world_from_station @ station_from_base,
        station_pose_contract_path=DEFAULT_STATION_POSE_CONTRACT_PATH,
    )

    graph = target.frame_graph
    assert graph.frame_by_id[MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1].semantic_parameters[
        "frame_family"
    ] == "simulation_world"
    assert target.canonical_reference_frame_id == (
        MUJOCO_SIM_HIL_STATION_FRAME_ID_V1
    )
    station_from_world = graph.resolve(
        MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1,
        MUJOCO_SIM_HIL_STATION_FRAME_ID_V1,
    ).matrix_target_source
    np.testing.assert_allclose(
        station_from_world,
        np.linalg.inv(world_from_station),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        station_from_world[:3, :3],
        np.diag((-1.0, -1.0, 1.0)),
        rtol=0.0,
        atol=1e-12,
    )
