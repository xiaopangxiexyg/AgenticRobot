"""Native geometry-only qualification; no physics, execution or render owner."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from scripts.probes.qualify_mujoco_scene import inspect_prepared_scene
from sim.mujoco.deploy.target_config import (
    DEFAULT_STATION_POSE_CONTRACT_PATH, MujocoSimHILTargetConfig,
)
from sim.mujoco.deploy.target_measurement import MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1
from sim.mujoco.deploy.target_preparation import prepare_mujoco_sim_hil_target
from sim.mujoco.scene import MujocoCameraRenderer


def forbidden(*_args, **_kwargs):
    raise AssertionError("Geometry-only qualification cannot step or render.")


@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(MujocoCameraRenderer, "__init__", forbidden)
    return prepare_mujoco_sim_hil_target(MujocoSimHILTargetConfig(
        task_config_path=Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"),
        station_pose_contract_path=DEFAULT_STATION_POSE_CONTRACT_PATH,
        realtime_worker=False, allow_scripted_task_reset=False,
    ))


def test_initial_geometry_is_read_only_and_uses_original_oracle(prepared):
    report = inspect_prepared_scene(prepared)
    assert report["qualified"]
    assert report["time_qpos_qvel_unchanged"] and report["time_sec"] == 0
    assert report["forbidden_pair_count"] > 0
    assert report["minimum_clearance_m"] == .002
    assert report["hard_penetration_m"] == .001
    assert not report["motion_executed"] and not report["hold_proved"]
    assert not report["execution_owner_created"] and not report["renderer_created"]


def test_station_fixture_requires_shared_world_transform(prepared):
    graph = prepared.target_frame_profile.frame_graph
    station = prepared.target_frame_profile.canonical_reference_frame_id
    pose = np.eye(4)
    pose[:3, 3] = (-.06, -.23, .12)
    world_pose, _ = graph.transform_pose_matrix(
        pose, source_reference_frame_id=station,
        target_reference_frame_id=MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1,
    )
    assert world_pose[:3, 3] == pytest.approx((-.412, .23, .13))
    assert not np.allclose(world_pose, pose)
    roundtrip, _ = graph.transform_pose_matrix(
        world_pose, source_reference_frame_id=MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1,
        target_reference_frame_id=station,
    )
    assert roundtrip == pytest.approx(pose)


@pytest.mark.parametrize("position,category", [
    ((0., -.2625, .04), "robot_object"),
    ((-.272, 0., .01), "object_support"),
])
def test_invalid_fixture_overlap_rejected_without_physics(prepared, position, category):
    # Deliberately invalid fixture setup, never an action or runtime fallback.
    prepared.scene.set_freejoint_state(
        prepared.task_config.object_freejoint_name,
        qpos=(*position, 1., 0., 0., 0.), qvel=(0.,) * 6, forward=True,
    )
    report = inspect_prepared_scene(prepared)
    assert not report["qualified"]
    assert report["time_qpos_qvel_unchanged"]
    if category == "robot_object":
        assert report["robot_hard_contact"]["collision_category"] == category
    else:
        assert report["object_penetrations"]
        assert any(row["distance_m"] < -.001 for row in report["object_penetrations"])


def test_qualification_rejects_disabled_protection_and_scripted_reset(prepared):
    for setting in ({"collision_stop_enabled": False}, {"allow_scripted_task_reset": True}):
        with pytest.raises(ValueError):
            inspect_prepared_scene(replace(prepared, config=replace(prepared.config, **setting)))


def test_qualification_rejects_noninitial_time(prepared):
    prepared.scene.data.time = 0.1
    with pytest.raises(ValueError, match="before physics advances"):
        inspect_prepared_scene(prepared)


def test_cli_retains_preparation_failure_without_overwriting(monkeypatch, tmp_path):
    from scripts.probes import qualify_mujoco_scene as probe
    output = tmp_path / "failure.json"
    monkeypatch.setattr(sys, "argv", ["probe", "missing.json", "--output", str(output)])

    def fail_prepare(_config):
        raise FileNotFoundError("missing fixture")

    monkeypatch.setattr(probe, "prepare_mujoco_sim_hil_target", fail_prepare)
    with pytest.raises(FileNotFoundError, match="missing fixture"):
        probe.main()
    retained = output.read_bytes()
    report = json.loads(retained)
    assert report["qualified"] is False
    assert report["error"]["type"] == "FileNotFoundError"
    with pytest.raises(FileExistsError):
        probe.main()
    assert output.read_bytes() == retained
