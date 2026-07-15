"""Benchmark the reference and fast Cube-Lift runtimes.

This command intentionally measures simulation throughput without rendering,
sleeping, training-framework work, or dataset output.  Results are development
machine baselines, not claims about the eventual sampling server.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..tasks.cube_lift import CubeLiftEnv


def _runtime_manifest() -> dict[str, Any]:
    import gymnasium
    import mujoco

    cpu_model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "mujoco_python_version": mujoco.__version__,
        "mujoco_native_version": mujoco.mj_versionString(),
        "numpy_version": np.__version__,
        "gymnasium_version": gymnasium.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }


def _measure(
    name: str,
    factory: Callable[[], Any],
    *,
    steps: int,
    warmup_steps: int,
    seed: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    env = factory()
    constructed = time.perf_counter()
    env.reset(seed=seed)
    reset_done = time.perf_counter()
    action = np.zeros(7, dtype=np.float32)
    for _ in range(warmup_steps):
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            env.reset()

    measured_start = time.perf_counter()
    physics_ticks = 0
    resets = 0
    for _ in range(steps):
        _obs, _reward, terminated, truncated, info = env.step(action)
        scheduler = info.get("scheduler") if isinstance(info, dict) else None
        if scheduler is not None:
            physics_ticks += int(scheduler["physics_ticks"])
        elif hasattr(env, "last_physics_ticks"):
            physics_ticks += int(env.last_physics_ticks)
        if terminated or truncated:
            env.reset()
            resets += 1
    measured_end = time.perf_counter()
    env.close()
    elapsed = measured_end - measured_start
    artifact = getattr(getattr(env, "kernel", None), "model_artifact", None)
    return {
        "runtime": name,
        "construct_seconds": constructed - start,
        "initial_reset_seconds": reset_done - constructed,
        "warmup_policy_steps": warmup_steps,
        "measured_policy_steps": steps,
        "measured_physics_steps": physics_ticks,
        "automatic_resets": resets,
        "elapsed_seconds": elapsed,
        "policy_steps_per_second": steps / elapsed,
        "physics_steps_per_second": physics_ticks / elapsed,
        "model_artifact_key": None if artifact is None else artifact.key,
        "model_cache_hit": None if artifact is None else artifact.cache_hit,
    }


def _linux_memory_kib(pids: Sequence[int]) -> dict[str, int] | None:
    totals = {"rss_kib": 0, "pss_kib": 0, "uss_kib": 0}
    try:
        for pid in pids:
            fields: dict[str, int] = {}
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text(
                encoding="utf-8"
            ).splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                parts = value.split()
                if parts:
                    fields[key] = int(parts[0])
            totals["rss_kib"] += fields.get("Rss", 0)
            totals["pss_kib"] += fields.get("Pss", 0)
            totals["uss_kib"] += fields.get("Private_Clean", 0) + fields.get(
                "Private_Dirty", 0
            )
    except (OSError, ValueError):
        return None
    return totals


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--runtime",
        choices=("reference", "fast", "both", "vector"),
        default="both",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--vector-envs",
        default="",
        help="Comma-separated worker counts, for example 1,4,8,16.",
    )
    return parser


def _measure_vector(
    num_envs: int,
    *,
    steps: int,
    warmup_steps: int,
    seed: int,
) -> dict[str, Any]:
    from ..runtime import CubeLiftVectorExecutor

    start = time.perf_counter()
    executor = CubeLiftVectorExecutor(num_envs, master_seed=seed)
    constructed = time.perf_counter()
    executor.reset()
    reset_done = time.perf_counter()
    process_ids = [os.getpid(), *executor.worker_pids]
    memory = _linux_memory_kib(process_ids)
    actions = np.zeros((num_envs, 7), dtype=np.float32)
    for _ in range(warmup_steps):
        executor.step(actions)
    measured_start = time.perf_counter()
    physics_ticks = 0
    for _ in range(steps):
        executor.step(actions)
        physics_ticks += int(np.sum(executor.last_physics_ticks))
    measured_end = time.perf_counter()
    final_memory = _linux_memory_kib(process_ids)
    shared_memory_bytes = executor.shared_memory_bytes
    executor.close()
    elapsed = measured_end - measured_start
    transitions = steps * num_envs
    return {
        "runtime": "multiprocess_vector",
        "num_envs": num_envs,
        "construct_seconds": constructed - start,
        "initial_reset_seconds": reset_done - constructed,
        "warmup_vector_steps": warmup_steps,
        "measured_vector_steps": steps,
        "measured_policy_transitions": transitions,
        "measured_physics_steps": physics_ticks,
        "elapsed_seconds": elapsed,
        "policy_transitions_per_second": transitions / elapsed,
        "physics_steps_per_second": physics_ticks / elapsed,
        "shared_memory_bytes": shared_memory_bytes,
        "process_count_including_parent": len(process_ids),
        "linux_memory": memory,
        "linux_memory_after": final_memory,
        "linux_memory_delta": (
            None
            if memory is None or final_memory is None
            else {key: final_memory[key] - memory[key] for key in memory}
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.steps <= 0 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup-steps must be non-negative.")
    factories: list[tuple[str, Callable[[], Any]]] = []
    if args.runtime in {"reference", "both"}:
        factories.append(("reference", CubeLiftEnv))
    if args.runtime in {"fast", "both"}:
        from ..tasks.cube_lift.fast_env import FastCubeLiftEnv

        factories.append(("fast", FastCubeLiftEnv))
    vector_counts = tuple(
        int(value) for value in args.vector_envs.split(",") if value.strip()
    )
    if any(value <= 0 for value in vector_counts):
        raise ValueError("vector-envs entries must be positive.")
    result = {
        "schema_version": "dexteleop.sim.mujoco.benchmark.cube_lift.v1",
        "manifest": _runtime_manifest(),
        "results": [
            _measure(
                name,
                factory,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                seed=args.seed,
            )
            for name, factory in factories
        ]
        + [
            _measure_vector(
                count,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                seed=args.seed,
            )
            for count in vector_counts
        ],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
