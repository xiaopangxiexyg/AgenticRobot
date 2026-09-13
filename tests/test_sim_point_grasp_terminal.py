from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from experiments.sim_point_grasp.commands import run_terminal as terminal_module
from experiments.sim_point_grasp.commands.run_terminal import build_parser
from experiments.sim_point_grasp.evaluation import (
    TERMINAL_OBSERVATION_HOME_QPOS_BY_SIDE,
    SimPointGraspEvaluationConfig,
    build_sim_point_grasp_task_config,
    optical_camera_pose_world,
    project_world_point,
)
from sim.mujoco.pointcloud import render_intrinsics_from_fovy


def test_terminal_defaults_to_real_like_graspgenx_wrist_cycle() -> None:
    args = build_parser().parse_args([])
    assert args.arm_side == "right"
    assert args.backend == "graspgenx"
    assert args.view == "selected_wrist"
    assert args.graspgenx_mode == "service"
    assert args.max_candidates == 5


def test_terminal_config_resolves_selected_physical_wrist(tmp_path: Path) -> None:
    sam2 = tmp_path / "sam2"
    backend = tmp_path / "backend"
    sam2.mkdir()
    backend.mkdir()
    config = SimPointGraspEvaluationConfig(
        output_dir=tmp_path / "output",
        sam2_asset_bundle=sam2,
        graspnet_asset_bundle=backend,
        arm_side="left",
        point_grasp_backend="vgn",
        generation_view="selected_wrist",
        object_profile="cube",
        use_terminal_observation_home=True,
    )
    assert config.generation_camera_name == "left_wrist"


def test_terminal_run_builds_one_deterministic_cube_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sam2 = tmp_path / "sam2"
    backend = tmp_path / "backend"
    sam2.mkdir()
    backend.mkdir()
    captured = {}

    def _capture(config):
        captured["config"] = config
        return {"trials": []}

    monkeypatch.setattr(terminal_module, "evaluate_sim_point_grasps", _capture)
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "output"),
            "--sam2-asset-bundle",
            str(sam2),
            "--point-grasp-asset-bundle",
            str(backend),
            "--backend",
            "vgn",
            "--view",
            "front",
            "--arm-side",
            "left",
            "--seed",
            "7",
        ]
    )
    terminal_module.run(args)
    config = captured["config"]
    assert config.trials == 1
    assert config.seed_start == 7
    assert config.point_grasp_backend == "vgn"
    assert config.generation_camera_name == "front"
    assert config.object_profile == "cube"
    assert config.use_terminal_observation_home
    assert config.cube_spawn_xy_jitter_m == (0.0, 0.0)
    assert config.cube_spawn_yaw_jitter_rad == 0.0


@pytest.mark.parametrize("arm_side", ("left", "right"))
def test_cube_task_uses_40mm_contract_and_terminal_home(arm_side: str) -> None:
    config = build_sim_point_grasp_task_config(
        controlled_side=arm_side,
        cube_spawn_xy_jitter_m=(0.0, 0.0),
        cube_spawn_yaw_jitter_rad=0.0,
        object_profile="cube",
        controlled_home_qpos=TERMINAL_OBSERVATION_HOME_QPOS_BY_SIDE[arm_side],
    )
    contract = config.base_contract()
    assert config.object_size_m == (0.04, 0.04, 0.04)
    assert contract.scene.objects[0].size_m == (0.04, 0.04, 0.04)
    assert contract.scene.objects[0].rgba == (0.85, 0.18, 0.12, 1.0)
    assert config.right_home_qpos == TERMINAL_OBSERVATION_HOME_QPOS_BY_SIDE[arm_side]


@pytest.mark.optional
@pytest.mark.parametrize("arm_side", ("left", "right"))
def test_terminal_home_real_wrist_camera_faces_cube(arm_side: str) -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    config = build_sim_point_grasp_task_config(
        controlled_side=arm_side,
        cube_spawn_xy_jitter_m=(0.0, 0.0),
        cube_spawn_yaw_jitter_rad=0.0,
        object_profile="cube",
        controlled_home_qpos=TERMINAL_OBSERVATION_HOME_QPOS_BY_SIDE[arm_side],
    )
    env = FastCubeLiftEnv(config, include_training_reward_state=False)
    try:
        env.reset(seed=0)
        scene = env.kernel.scene
        camera_name = f"{arm_side}_wrist"
        camera_pose = optical_camera_pose_world(scene, camera_name)
        cube_center = scene.body_pose(config.object_body_name)[:3, 3]
        direction = cube_center - camera_pose[:3, 3]
        direction /= np.linalg.norm(direction)
        assert float(camera_pose[:3, 2] @ direction) > 0.98
        metadata = scene.camera_metadata()[camera_name]
        intrinsics = render_intrinsics_from_fovy(
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            fovy_deg=float(metadata["fovy_deg"]),
        )
        u, v, _depth = project_world_point(
            cube_center,
            world_from_camera=camera_pose,
            intrinsics=intrinsics,
        )
        assert abs(u - intrinsics.width // 2) < 80
        assert abs(v - intrinsics.height // 2) < 80
        assert math.isfinite(float(scene.data.time))
    finally:
        env.close()
