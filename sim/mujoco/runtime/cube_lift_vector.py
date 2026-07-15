"""Shared-memory multiprocess vector execution for Cube-Lift."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from gymnasium.vector import VectorEnv as _VectorEnvBase
except ModuleNotFoundError:
    _VectorEnvBase = object


_ARRAY_LAYOUT = {
    "actions": ((7,), np.dtype(np.float32)),
    "observations": ((31,), np.dtype(np.float32)),
    "final_observations": ((31,), np.dtype(np.float32)),
    "rewards": ((), np.dtype(np.float32)),
    "terminated": ((), np.dtype(np.uint8)),
    "truncated": ((), np.dtype(np.uint8)),
    "final_valid": ((), np.dtype(np.uint8)),
    "physics_ticks": ((), np.dtype(np.int32)),
}


def _attach_arrays(
    names: dict[str, str],
    num_envs: int,
) -> tuple[dict[str, SharedMemory], dict[str, np.ndarray]]:
    segments: dict[str, SharedMemory] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, (tail, dtype) in _ARRAY_LAYOUT.items():
        segment = SharedMemory(name=names[name])
        segments[name] = segment
        arrays[name] = np.ndarray((num_envs, *tail), dtype=dtype, buffer=segment.buf)
    return segments, arrays


def _worker_main(
    env_id: int,
    num_envs: int,
    connection: Connection,
    shared_names: dict[str, str],
    config_path: str,
    cache_dir: str | None,
    auto_reset: bool,
    master_seed: int,
) -> None:
    segments: dict[str, SharedMemory] = {}
    env = None
    try:
        segments, arrays = _attach_arrays(shared_names, num_envs)
        from ..tasks.cube_lift.fast_env import FastCubeLiftEnv

        env = FastCubeLiftEnv(
            config_path,
            info_mode="compact",
            model_cache_dir=cache_dir,
        )
        episode_id = 0
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                if isinstance(payload, dict):
                    seed = payload.get("seed")
                    episode_id = int(payload.get("episode_id", 0))
                else:
                    seed = payload
                    episode_id = 0
                seed = None if seed is None else int(seed)
                observation, info = env.reset(seed=seed)
                arrays["observations"][env_id] = observation
                arrays["final_valid"][env_id] = 0
                arrays["physics_ticks"][env_id] = env.last_physics_ticks
                connection.send(("ok", info))
                continue
            if command != "step":
                raise ValueError(f"Unknown vector worker command {command!r}.")

            observation, reward, terminated, truncated, info = env.step(
                arrays["actions"][env_id]
            )
            arrays["rewards"][env_id] = reward
            arrays["terminated"][env_id] = terminated
            arrays["truncated"][env_id] = truncated
            arrays["physics_ticks"][env_id] = env.last_physics_ticks
            terminal = bool(terminated or truncated)
            if terminal:
                arrays["final_observations"][env_id] = observation
                arrays["final_valid"][env_id] = 1
                final_info = info
                if auto_reset:
                    episode_id += 1
                    seed = int(
                        np.random.SeedSequence([master_seed, env_id, episode_id]).generate_state(
                            1, dtype=np.uint32
                        )[0]
                    )
                    observation, reset_info = env.reset(seed=seed)
                    info = {
                        **info,
                        "final_info": final_info,
                        "reset_info": reset_info,
                        "auto_reset": True,
                        "episode_id": episode_id,
                    }
            else:
                arrays["final_valid"][env_id] = 0
            arrays["observations"][env_id] = observation
            connection.send(("ok", info))
    except BaseException as exc:
        try:
            connection.send(
                (
                    "error",
                    {
                        "env_id": env_id,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()
        for segment in segments.values():
            segment.close()


class CubeLiftVectorExecutor:
    """One fast scalar kernel per process with shared numeric buffers."""

    def __init__(
        self,
        num_envs: int,
        *,
        config_path: str | Path = "sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json",
        model_cache_dir: str | Path | None = None,
        auto_reset: bool = True,
        master_seed: int = 0,
        start_method: str = "spawn",
    ) -> None:
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"Unsupported multiprocessing start method {start_method!r}.")
        self.auto_reset = bool(auto_reset)
        self.master_seed = int(master_seed)
        self._closed = False
        self._segments: dict[str, SharedMemory] = {}
        self._arrays: dict[str, np.ndarray] = {}
        for name, (tail, dtype) in _ARRAY_LAYOUT.items():
            shape = (self.num_envs, *tail)
            size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            segment = SharedMemory(create=True, size=size)
            self._segments[name] = segment
            self._arrays[name] = np.ndarray(shape, dtype=dtype, buffer=segment.buf)
            self._arrays[name].fill(0)

        context = mp.get_context(start_method)
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        names = {name: segment.name for name, segment in self._segments.items()}
        try:
            for env_id in range(self.num_envs):
                parent, child = context.Pipe(duplex=True)
                process = context.Process(
                    target=_worker_main,
                    args=(
                        env_id,
                        self.num_envs,
                        child,
                        names,
                        str(config_path),
                        None if model_cache_dir is None else str(model_cache_dir),
                        self.auto_reset,
                        self.master_seed,
                    ),
                    name=f"cube-lift-worker-{env_id}",
                    daemon=True,
                )
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
            for connection in self._connections:
                status, payload = connection.recv()
                if status != "ready":
                    self._raise_worker(status, payload)
        except BaseException:
            self.close()
            raise

    @property
    def shared_memory_bytes(self) -> int:
        return sum(segment.size for segment in self._segments.values())

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return tuple(int(process.pid) for process in self._processes)

    def reset(
        self,
        *,
        env_ids: Sequence[int] | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        self._require_open()
        selected = (
            tuple(range(self.num_envs))
            if env_ids is None
            else tuple(int(env_id) for env_id in env_ids)
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("env_ids must be a non-empty unique sequence.")
        if any(env_id < 0 or env_id >= self.num_envs for env_id in selected):
            raise ValueError("env_ids contains an out-of-range environment index.")
        if seeds is None:
            seeds = [
                int(
                    np.random.SeedSequence([self.master_seed, env_id, 0]).generate_state(
                        1, dtype=np.uint32
                    )[0]
                )
                for env_id in selected
            ]
        if len(seeds) != len(selected):
            raise ValueError("seeds must have one entry per selected environment.")
        for env_id, seed in zip(selected, seeds):
            self._connections[env_id].send(
                ("reset", {"seed": seed, "episode_id": 0})
            )
        infos = self._receive_selected(selected)
        return self._arrays["observations"][list(selected)].copy(), infos

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        self._require_open()
        values = np.asarray(actions, dtype=np.float32)
        if values.shape != (self.num_envs, 7) or not np.all(np.isfinite(values)):
            raise ValueError(f"actions must be finite with shape ({self.num_envs}, 7).")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("actions must lie in [-1, 1].")
        self._arrays["actions"][:] = values
        for connection in self._connections:
            connection.send(("step", None))
        infos = self._receive_all()
        for env_id, info in enumerate(infos):
            if self._arrays["final_valid"][env_id]:
                info["final_observation"] = self._arrays["final_observations"][env_id].copy()
        return (
            self._arrays["observations"].copy(),
            self._arrays["rewards"].copy(),
            self._arrays["terminated"].astype(bool, copy=True),
            self._arrays["truncated"].astype(bool, copy=True),
            infos,
        )

    @property
    def last_physics_ticks(self) -> np.ndarray:
        return self._arrays["physics_ticks"].copy()

    def _receive_all(self) -> list[dict[str, Any]]:
        return self._receive_selected(tuple(range(self.num_envs)))

    def _receive_selected(self, env_ids: Sequence[int]) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        for env_id in env_ids:
            connection = self._connections[env_id]
            try:
                status, payload = connection.recv()
            except EOFError as exc:
                raise RuntimeError("Cube-Lift worker exited without a response.") from exc
            if status != "ok":
                self._raise_worker(status, payload)
            infos.append(payload)
        return infos

    @staticmethod
    def _raise_worker(status: str, payload: Any) -> None:
        if status == "error" and isinstance(payload, dict):
            raise RuntimeError(
                f"Cube-Lift worker {payload.get('env_id')} failed: "
                f"{payload.get('type')}: {payload.get('message')}\n{payload.get('traceback')}"
            )
        raise RuntimeError(f"Unexpected Cube-Lift worker response: {status!r}, {payload!r}")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CubeLiftVectorExecutor is closed.")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in getattr(self, "_connections", ()):
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in getattr(self, "_processes", ()):
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for connection in getattr(self, "_connections", ()):
            connection.close()
        for segment in getattr(self, "_segments", {}).values():
            segment.close()
            segment.unlink()

    def __enter__(self) -> "CubeLiftVectorExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _infos_to_vector_dict(infos: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = sorted({key for info in infos for key in info})
    result: dict[str, np.ndarray] = {}
    for key in keys:
        values = [info.get(key) for info in infos]
        result[key] = np.asarray(values, dtype=object)
        result[f"_{key}"] = np.asarray([key in info for info in infos], dtype=bool)
    return result


class CubeLiftVectorEnv(_VectorEnvBase):
    """Gymnasium-compatible vector adapter around CubeLiftVectorExecutor."""

    metadata = {"render_modes": [], "render_fps": 15}

    def __init__(self, num_envs: int, **executor_kwargs: Any) -> None:
        try:
            from gymnasium import spaces
            from gymnasium.vector.utils import batch_space
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("CubeLiftVectorEnv requires gymnasium.") from exc

        self.num_envs = int(num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.single_observation_space = spaces.Box(
            -1e6, 1e6, shape=(31,), dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.executor = CubeLiftVectorExecutor(self.num_envs, **executor_kwargs)

    def reset(
        self,
        *,
        seed: int | Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if options:
            raise ValueError("CubeLiftVectorEnv does not support reset options yet.")
        if seed is None:
            seeds = None
        elif isinstance(seed, (int, np.integer)):
            seeds = [int(seed) + env_id for env_id in range(self.num_envs)]
        else:
            seeds = [int(value) for value in seed]
        observations, infos = self.executor.reset(seeds=seeds)
        return observations, _infos_to_vector_dict(infos)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        observations, rewards, terminated, truncated, infos = self.executor.step(actions)
        return observations, rewards, terminated, truncated, _infos_to_vector_dict(infos)

    def close(self, **kwargs: Any) -> None:
        self.executor.close()


__all__ = ["CubeLiftVectorEnv", "CubeLiftVectorExecutor"]
