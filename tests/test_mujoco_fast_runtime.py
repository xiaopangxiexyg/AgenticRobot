from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pytest

TASK_CONFIG = Path("sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json")
ACTUATED_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_actuated_v1.json"
)
FULLY_ACTUATED_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_actuated_v2.json"
)


@pytest.fixture(scope="module")
def runtime_model_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Share compiled artifacts, never environments, MjModel/MjData or snapshots.
    # Cache-contract tests below deliberately keep their own cold tmp_path roots.
    return tmp_path_factory.mktemp("fast-runtime-model-cache")


def _load_shared_model_artifact(
    cache_root: str,
    start_at: float,
) -> tuple[str, bool, str, int, int]:
    """Load one artifact in a fresh process after a common start barrier."""

    from sim.mujoco.artifacts import ModelArtifactCache
    from sim.mujoco.tasks.cube_lift import CubeLiftTaskConfig

    time.sleep(max(0.0, start_at - time.monotonic()))
    contract = CubeLiftTaskConfig.load(TASK_CONFIG).base_contract()
    scene, artifact = ModelArtifactCache(cache_root).load_scene(contract)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    return (
        artifact.key,
        artifact.cache_hit,
        manifest["mjb_sha256"],
        int(scene.model.nq),
        int(scene.model.nv),
    )


@pytest.mark.parametrize("parent_value", (None, "/parent/runtime/lib"))
def test_worker_runtime_library_path_restores_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parent_value: str | None,
) -> None:
    import os
    import sys

    from sim.mujoco.runtime.cube_lift_vector import _worker_runtime_library_path

    environment_root = tmp_path / "worker-environment"
    runtime_lib = environment_root / "lib"
    runtime_lib.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(environment_root))
    if parent_value is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", parent_value)

    with _worker_runtime_library_path():
        inherited = os.environ["LD_LIBRARY_PATH"].split(":")
        assert inherited[0] == str(runtime_lib)
        assert inherited[1:] == ([] if parent_value is None else [parent_value])

    if parent_value is None:
        assert "LD_LIBRARY_PATH" not in os.environ
    else:
        assert os.environ["LD_LIBRARY_PATH"] == parent_value


def test_fast_kernel_matches_reference_action_trace(runtime_model_cache: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import CubeLiftEnv, FastCubeLiftEnv

    reference = CubeLiftEnv(TASK_CONFIG)
    fast = FastCubeLiftEnv(
        TASK_CONFIG,
        use_settled_snapshot=False,
        model_cache_dir=runtime_model_cache,
    )
    rng = np.random.default_rng(20260715)
    try:
        reference_observation, _ = reference.reset(seed=19)
        fast_observation, _ = fast.reset(seed=19)
        np.testing.assert_allclose(fast_observation, reference_observation, rtol=0.0, atol=0.0)
        for _ in range(12):
            action = rng.uniform(-0.25, 0.25, 7).astype(np.float32)
            ref = reference.step(action)
            candidate = fast.step(action)
            np.testing.assert_allclose(candidate[0], ref[0], rtol=0.0, atol=1e-7)
            assert candidate[1] == pytest.approx(ref[1], abs=1e-7)
            assert candidate[2:4] == ref[2:4]
            assert candidate[4]["termination_reason"] == ref[4]["termination_reason"]
            assert candidate[4]["physics_ticks"] == ref[4]["scheduler"]["physics_ticks"]
            assert candidate[4]["command_ticks"] == ref[4]["scheduler"]["command_ticks"]
    finally:
        fast.close()
        reference.close()


def test_fast_kernel_settled_snapshot_avoids_repeated_settle(runtime_model_cache: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv

    env = FastCubeLiftEnv(TASK_CONFIG, model_cache_dir=runtime_model_cache)
    try:
        first, first_info = env.reset(seed=1)
        second, second_info = env.reset(seed=2)
        np.testing.assert_allclose(second, first, rtol=0.0, atol=0.0)
        assert first_info["physics_ticks"] == env.config.settle_physics_ticks
        assert second_info["physics_ticks"] == 0
        assert env.kernel.physics_time == pytest.approx(
            env.config.settle_physics_ticks / env.config.physics_hz
        )
        after_snapshot = env.step(np.zeros(7, dtype=np.float32))[0]
        env.reset(seed=3)
        repeated = env.step(np.zeros(7, dtype=np.float32))[0]
        np.testing.assert_allclose(repeated, after_snapshot, rtol=0.0, atol=0.0)
    finally:
        env.close()


def test_model_artifact_cache_hits_and_recovers_from_corruption(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco.artifacts import ModelArtifactCache
    from sim.mujoco.tasks.cube_lift import CubeLiftTaskConfig

    contract = CubeLiftTaskConfig.load(TASK_CONFIG).base_contract()
    cache = ModelArtifactCache(tmp_path / "cache")
    first_scene, first = cache.load_scene(contract)
    assert not first.cache_hit
    assert first.mjb_path.is_file()
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["fingerprint"]["mujoco_python_version"]
    assert manifest["fingerprint"]["mujoco_native_version"]

    second_scene, second = cache.load_scene(contract)
    assert second.cache_hit
    assert second.key == first.key
    assert second_scene.data is not first_scene.data
    np.testing.assert_allclose(second_scene.model.opt.timestep, first_scene.model.opt.timestep)

    first.mjb_path.write_bytes(b"corrupt")
    _third_scene, third = cache.load_scene(contract)
    assert not third.cache_hit
    assert third.mjb_path.stat().st_size > len(b"corrupt")


def test_model_artifact_cache_publishes_one_valid_artifact_across_processes(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco.artifacts import ModelArtifactCache
    from sim.mujoco.tasks.cube_lift import CubeLiftTaskConfig

    cache_root = tmp_path / "shared-cache"
    start_at = time.monotonic() + 1.0
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=2,
        mp_context=context,
    ) as executor:
        futures = [
            executor.submit(_load_shared_model_artifact, str(cache_root), start_at)
            for _ in range(2)
        ]
        results = [future.result(timeout=90.0) for future in futures]

    keys = {result[0] for result in results}
    assert len(keys) == 1
    assert sorted(result[1] for result in results) == [False, True]
    assert len({result[2] for result in results}) == 1
    assert len({result[3:] for result in results}) == 1

    contract = CubeLiftTaskConfig.load(TASK_CONFIG).base_contract()
    scene, artifact = ModelArtifactCache(cache_root).load_scene(contract)
    assert artifact.cache_hit
    assert artifact.key == next(iter(keys))
    assert (int(scene.model.nq), int(scene.model.nv)) == results[0][3:]


def test_model_artifact_cache_separates_actuated_profile(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    from sim.mujoco.artifacts import ModelArtifactCache
    from sim.mujoco.tasks.cube_lift import CubeLiftTaskConfig

    direct = CubeLiftTaskConfig.load(TASK_CONFIG)
    actuated = CubeLiftTaskConfig.load(ACTUATED_TASK_CONFIG)
    cache = ModelArtifactCache(tmp_path / "cache")
    direct_scene, direct_artifact = cache.load_scene(direct.base_contract())
    actuated_scene, actuated_artifact = cache.load_scene(
        actuated.base_contract(),
        position_actuation_profile=actuated.position_actuation_profile(),
    )

    assert direct_artifact.key != actuated_artifact.key
    assert direct_scene.model.nu == 0
    assert actuated_scene.model.nu == 12
    assert actuated_scene.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    manifest = json.loads(actuated_artifact.manifest_path.read_text(encoding="utf-8"))
    assert (
        manifest["fingerprint"]["position_actuation_profile"]["name"]
        == actuated.position_actuation_profile_name
    )


def test_model_artifact_cache_separates_fully_actuated_gripper_profile(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco.artifacts import ModelArtifactCache
    from sim.mujoco.tasks.cube_lift import CubeLiftTaskConfig

    arm_only = CubeLiftTaskConfig.load(ACTUATED_TASK_CONFIG)
    fully_actuated = CubeLiftTaskConfig.load(FULLY_ACTUATED_TASK_CONFIG)
    cache = ModelArtifactCache(tmp_path / "cache")
    arm_scene, arm_artifact = cache.load_scene(
        arm_only.base_contract(),
        position_actuation_profile=arm_only.position_actuation_profile(),
    )
    full_scene, full_artifact = cache.load_scene(
        fully_actuated.base_contract(),
        position_actuation_profile=fully_actuated.position_actuation_profile(),
        gripper_actuation_profile=fully_actuated.gripper_actuation_profile(),
    )

    assert arm_artifact.key != full_artifact.key
    assert arm_scene.model.nu == 12
    assert full_scene.model.nu == 16
    manifest = json.loads(full_artifact.manifest_path.read_text(encoding="utf-8"))
    assert (
        manifest["fingerprint"]["gripper_actuation_profile"]["name"]
        == fully_actuated.gripper_actuation_profile_name
    )


def test_actuated_cube_lift_uses_measured_arm_state_and_shared_viewer_kernel(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import CubeLiftEnv

    env = CubeLiftEnv(ACTUATED_TASK_CONFIG)
    try:
        initial, reset_info = env.reset(seed=4)
        assert reset_info["execution_mode"] == "actuated_position_servo"
        assert reset_info["robot_dynamics"] == "mujoco_position_actuator"
        assert reset_info["gripper_dynamics"] == "prescribed_qpos_debug"
        assert reset_info["arm_tracking_error_rad_max"] < 0.01
        assert env.scene.model.nu == 12

        action = np.zeros(7, dtype=np.float32)
        action[0] = 0.5
        first = env.step(action)[0]
        for _ in range(6):
            final, _reward, terminated, truncated, info = env.step(
                np.zeros(7, dtype=np.float32)
            )
            assert not terminated and not truncated
        # Physical tracking is delayed: the first step has moved only part of
        # the 0.04 rad target, and subsequent 1000 Hz dynamics converge.
        first_delta = float(first[0] - initial[0])
        final_delta = float(final[0] - initial[0])
        assert 0.0 < first_delta < 0.04
        assert final_delta > first_delta
        assert final_delta == pytest.approx(0.04, abs=0.005)
        assert info["arm_tracking_error_rad_max"] < 0.01
    finally:
        env.close()


def test_pose_planner_recovers_only_small_mujoco_joint_limit_residual() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv
    from sim.mujoco.tasks.cube_lift.pose_action import (
        SIM_RM65_JOINT_LIMIT_RECOVERY_TOLERANCE_RAD,
    )

    env = FastCubeLiftEnv(
        "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v2.json",
        use_settled_snapshot=False,
    )
    try:
        env.reset(seed=11)
        controller = env.kernel._pose_action_controller
        assert controller is not None
        if controller._ik is None:
            pytest.skip("canonical RM65 planner is unavailable in this environment")
        upper = controller._ik.planner._joint_upper_by_side["right"]
        joint_index = 5
        residual = 0.5 * SIM_RM65_JOINT_LIMIT_RECOVERY_TOLERANCE_RAD
        env.kernel.data.qpos[env.kernel._right_qpos_idx[joint_index]] = (
            upper[joint_index] + residual
        )
        env.kernel.mujoco.mj_forward(env.kernel.model, env.kernel.data)

        status = controller.sync_planner_from_scene()

        assert status.safe is True
        assert status.recovered is True
        assert status.violating_side == "right"
        assert status.violating_joint_index == joint_index
        assert status.max_violation_rad == pytest.approx(residual, abs=1e-10)
        # Recovery only sanitizes the planner snapshot; it does not teleport or
        # rewrite the MuJoCo state.
        assert env.kernel.scene.get_arm_qpos("right")[joint_index] == pytest.approx(
            upper[joint_index] + residual,
            abs=1e-10,
        )
    finally:
        env.close()


def test_pose_planner_hard_limit_violation_terminates_only_current_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv
    from sim.mujoco.tasks.cube_lift.pose_action import CubeLiftPosePlannerSyncStatus

    env = FastCubeLiftEnv(
        "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v2.json",
        use_settled_snapshot=False,
    )
    try:
        env.reset(seed=12)
        controller = env.kernel._pose_action_controller
        assert controller is not None
        hard_status = CubeLiftPosePlannerSyncStatus(
            safe=False,
            recovered=False,
            max_violation_rad=2e-3,
            violating_side="right",
            violating_joint_index=5,
        )

        def hard_sync():
            controller._last_planner_sync = hard_status
            return hard_status

        monkeypatch.setattr(controller, "sync_planner_from_scene", hard_sync)

        _observation, _reward, terminated, truncated, info = env.step(
            np.zeros(7, dtype=np.float32)
        )

        assert terminated is True
        assert truncated is False
        assert info["termination_reason"] == "sim_joint_position_limit_violation"
        assert info["pose_ik_limit_violation_rad"] == pytest.approx(2e-3)
    finally:
        env.close()


def test_pose_planner_uses_mujoco_collision_clearance() -> None:
    """The sim IK anchor must use the sim policy's clearance, not real defaults."""

    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v2.json",
        use_settled_snapshot=False,
    )
    try:
        env.reset(seed=13)
        controller = env.kernel._pose_action_controller
        assert controller is not None
        expected = env.kernel.scene.scene_spec.collision.safety_minimum_clearance_m
        assert expected == pytest.approx(0.002)
        assert controller.planner_collision_min_distance_m == pytest.approx(expected)
        if controller._ik is not None:
            assert controller._ik.planner.config.collision_min_distance_m == pytest.approx(
                expected
            )
    finally:
        env.close()


def test_default_cube_lift_uses_measured_identified_gripper_state() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import CubeLiftEnv

    env = CubeLiftEnv()
    try:
        initial, reset_info = env.reset(seed=5)
        assert env.config.name == "rm65_right_cube_lift_rl_normal_v1"
        assert env.config.policy_class == "normal"
        assert initial.shape == (15,)
        assert reset_info["grasp_latched"] is False
        assert "training_reward_state" not in reset_info
        assert reset_info["gripper_dynamics"] == "mujoco_identified_position_actuator"
        assert reset_info["gripper_actuation_profile_name"] == (
            "dmtac_bimanual_position_20260722.v1"
        )
        assert env.scene.model.nu == 16

        close = np.zeros(7, dtype=np.float32)
        close[6] = 1.0
        first, _reward, terminated, truncated, first_info = env.step(close)
        assert not terminated and not truncated
        assert first_info["right_gripper_target_close"] == pytest.approx(0.2)
        # Transport delay plus the physical finger servo keeps the first
        # 66.7 ms policy observation effectively measured-open.
        assert first[6] < 1e-3
        assert first_info["right_gripper_measured_close"] < 1e-3

        for _ in range(3):
            _observation, _reward, terminated, truncated, _info = env.step(close)
            assert not terminated and not truncated
        for _ in range(8):
            final, _reward, terminated, truncated, info = env.step(
                np.zeros(7, dtype=np.float32)
            )
            assert not terminated and not truncated
        assert final[6] == pytest.approx(0.8, abs=0.01)
        assert info["right_gripper_measured_close"] == pytest.approx(0.8, abs=0.01)
        assert info["right_gripper_tracking_error"] == pytest.approx(0.0, abs=0.01)
    finally:
        env.close()


@pytest.mark.slow_sim
def test_vector_executor_is_deterministic_and_preserves_terminal_observation(
    runtime_model_cache: Path,
) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.runtime import CubeLiftVectorExecutor

    kwargs = {
        "num_envs": 2,
        "config_path": TASK_CONFIG,
        "model_cache_dir": runtime_model_cache,
        "master_seed": 31,
        "start_method": "spawn",
    }
    with CubeLiftVectorExecutor(**kwargs) as executor:
        initial_a, _ = executor.reset(seeds=[7, 8])
        next_a = executor.step(np.zeros((2, 7), dtype=np.float32))[0]
        initial_b, _ = executor.reset(seeds=[7, 8])
        next_b = executor.step(np.zeros((2, 7), dtype=np.float32))[0]
        np.testing.assert_allclose(initial_b, initial_a, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(next_b, next_a, rtol=0.0, atol=0.0)
        assert executor.shared_memory_bytes > 0

        selected, selected_infos = executor.reset(env_ids=[1], seeds=[99])
        assert selected.shape == (1, 31)
        assert len(selected_infos) == 1
        executor.reset(seeds=[7, 8])

        result = None
        for _ in range(150):
            result = executor.step(np.zeros((2, 7), dtype=np.float32))
        assert result is not None
        observations, _rewards, terminated, truncated, infos = result
        assert not np.any(terminated)
        assert np.all(truncated)
        assert observations.shape == (2, 31)
        assert all(info["auto_reset"] for info in infos)
        assert all(info["final_observation"].shape == (31,) for info in infos)


def test_fast_env_passes_gymnasium_checker(runtime_model_cache: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv

    env = FastCubeLiftEnv(TASK_CONFIG, model_cache_dir=runtime_model_cache)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_vector_env_exposes_batched_gymnasium_spaces(runtime_model_cache: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from gymnasium.vector import VectorEnv

    from sim.mujoco.runtime import CubeLiftVectorEnv

    env = CubeLiftVectorEnv(
        2,
        config_path=TASK_CONFIG,
        model_cache_dir=runtime_model_cache,
        start_method="spawn",
    )
    try:
        assert isinstance(env, VectorEnv)
        observations, _info = env.reset(seed=4)
        assert env.observation_space.contains(observations)
        result = env.step(np.zeros((2, 7), dtype=np.float32))
        assert result[0].shape == (2, 31)
        assert result[1].shape == (2,)
    finally:
        env.close()
