from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


TASK_CONFIG = Path("sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json")


def test_fast_kernel_matches_reference_action_trace(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import CubeLiftEnv, FastCubeLiftEnv

    reference = CubeLiftEnv(TASK_CONFIG)
    fast = FastCubeLiftEnv(
        TASK_CONFIG,
        use_settled_snapshot=False,
        model_cache_dir=tmp_path / "cache",
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


def test_fast_kernel_settled_snapshot_avoids_repeated_settle(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv

    env = FastCubeLiftEnv(TASK_CONFIG, model_cache_dir=tmp_path / "cache")
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
    np.testing.assert_allclose(second_scene.model.opt.timestep, first_scene.model.opt.timestep)

    first.mjb_path.write_bytes(b"corrupt")
    _third_scene, third = cache.load_scene(contract)
    assert not third.cache_hit
    assert third.mjb_path.stat().st_size > len(b"corrupt")


def test_vector_executor_is_deterministic_and_preserves_terminal_observation(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.runtime import CubeLiftVectorExecutor

    kwargs = {
        "num_envs": 2,
        "config_path": TASK_CONFIG,
        "model_cache_dir": tmp_path / "cache",
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


def test_fast_env_passes_gymnasium_checker(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env
    from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv

    env = FastCubeLiftEnv(TASK_CONFIG, model_cache_dir=tmp_path / "cache")
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_vector_env_exposes_batched_gymnasium_spaces(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from gymnasium.vector import VectorEnv
    from sim.mujoco.runtime import CubeLiftVectorEnv

    env = CubeLiftVectorEnv(
        2,
        config_path=TASK_CONFIG,
        model_cache_dir=tmp_path / "cache",
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
