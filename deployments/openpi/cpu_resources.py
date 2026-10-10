"""Default-off CPU resources shared by OpenPI inference and training launchers.

Keep this module dependency-free: training must apply its environment before
NumPy, PyTorch, JAX, or their native thread pools are imported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from robot_motion.cpu_topology import cpu_last_level_cache_ids

NATIVE_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def non_negative_cpu_id(text: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("CPU id must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("CPU id must be non-negative")
    return value


def positive_thread_count(text: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("thread count must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("thread count must be positive")
    return value


def native_thread_environment(thread_count: int | None) -> dict[str, str]:
    if thread_count is None:
        return {}
    count = int(thread_count)
    if isinstance(thread_count, bool) or count != thread_count or count <= 0:
        raise ValueError("native thread count must be a positive integer")
    return {name: str(count) for name in NATIVE_THREAD_ENVIRONMENT_VARIABLES}


def cpu_exclusion_plan(
    excluded_cpu_ids: Sequence[int], exclude_llc_of_cpu_ids: Sequence[int]
) -> dict[str, object] | None:
    """Resolve topology without changing affinity or widening an inherited mask."""

    if not excluded_cpu_ids and not exclude_llc_of_cpu_ids:
        return None
    for cpu in (*excluded_cpu_ids, *exclude_llc_of_cpu_ids):
        if isinstance(cpu, bool) or int(cpu) != cpu or int(cpu) < 0:
            raise ValueError("CPU id must be a non-negative integer")
    getter = getattr(os, "sched_getaffinity", None)
    if not callable(getter):
        raise RuntimeError("OpenPI CPU affinity is unavailable on this platform")  # noqa: TRY004 - platform capability, not an input type error.
    allowed = set(getter(0))
    excluded = set(excluded_cpu_ids)
    if not excluded <= allowed:
        raise RuntimeError(
            "requested CPU exclusion is outside the launcher affinity mask"
        )
    groups = []
    for cpu in sorted(set(exclude_llc_of_cpu_ids)):
        shared = cpu_last_level_cache_ids(cpu)
        groups.append({"control_cpu": cpu, "shared_cpu_ids": list(shared)})
        excluded.update(allowed.intersection(shared))
    target = allowed.difference(excluded)
    if not target:
        raise RuntimeError("OpenPI CPU exclusion leaves no runnable CPU")
    return {
        "allowed_cpu_ids_before": sorted(allowed),
        "last_level_cache_groups": groups,
        "excluded_cpu_ids": sorted(excluded),
        "effective_cpu_ids": sorted(target),
    }


def exclude_cpus_before_exec(excluded_cpu_ids: Sequence[int]) -> dict[str, object]:
    """Bind only the launcher thread; exec replaces all its existing threads."""

    getter = getattr(os, "sched_getaffinity", None)
    setter = getattr(os, "sched_setaffinity", None)
    if not callable(getter) or not callable(setter):
        raise RuntimeError("OpenPI CPU affinity is unavailable on this platform")  # noqa: TRY004 - platform capability, not an input type error.
    for cpu in excluded_cpu_ids:
        if isinstance(cpu, bool) or int(cpu) != cpu or int(cpu) < 0:
            raise ValueError("CPU id must be a non-negative integer")
    before = {int(cpu) for cpu in getter(0)}
    excluded = {int(cpu) for cpu in excluded_cpu_ids}
    missing = excluded.difference(before)
    if missing:
        raise RuntimeError(
            "requested OpenPI CPU exclusion is outside the launcher affinity mask: "
            f"missing={sorted(missing)} allowed={sorted(before)}"
        )
    target = before.difference(excluded)
    if not target:
        raise RuntimeError("OpenPI CPU exclusion leaves no runnable CPU")
    setter(0, target)
    after = {int(cpu) for cpu in getter(0)}
    if after != target:
        raise RuntimeError(
            "failed to verify OpenPI CPU exclusion: "
            f"requested={sorted(target)} effective={sorted(after)}"
        )
    return {
        "excluded_cpu_ids": sorted(excluded),
        "effective_cpu_ids": sorted(after),
        "native_threads_migrated": 1,
    }


def add_cpu_resource_arguments(
    parser: argparse.ArgumentParser, *, check_only: bool = False
) -> None:
    parser.add_argument(
        "--exclude-cpu",
        type=non_negative_cpu_id,
        action="append",
        default=[],
        help="Default-off: exclude one control CPU from the inherited model mask; repeatable.",
    )
    parser.add_argument(
        "--exclude-llc-of-cpu",
        type=non_negative_cpu_id,
        action="append",
        default=[],
        help=(
            "Default-off: exclude this control CPU's full Linux last-level data/unified "
            "cache group, including SMT siblings; never grants system-wide exclusivity."
        ),
    )
    parser.add_argument(
        "--native-thread-count",
        type=positive_thread_count,
        default=None,
        help=(
            "Default-off: set OMP/MKL/OpenBLAS/NumExpr/BLIS/Accelerate environments "
            "before model imports; not a total JAX/XLA or process thread limit."
        ),
    )
    if check_only:
        parser.add_argument(
            "--cpu-resource-check-only",
            action="store_true",
            help=(
                "Print the resource plan and exit before importing training dependencies; "
                "does not validate a training config, change affinity, or load a model."
            ),
        )


def require_cpu_resource_cli(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """In-process calls cannot establish pre-import native pool environments."""

    if (
        args.exclude_cpu
        or args.exclude_llc_of_cpu
        or args.native_thread_count is not None
        or args.cpu_resource_check_only
    ):
        parser.error(
            "CPU resource options require launching this module with python -m, "
            "not calling main() in an already initialized Python process"
        )


def bootstrap_training_cpu_resources(module_name: str) -> None:
    """Re-exec a training CLI before heavy imports, stripping only resource flags.

    No environment marker can bypass admission. The new process no longer has
    resource options, and all future threads/children inherit its initial mask.
    Arguments after an OpenPI wrapper's `--` separator are preserved verbatim.
    """

    values = sys.argv[1:]
    separator = values.index("--") if "--" in values else len(values)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_cpu_resource_arguments(parser, check_only=True)
    args, remaining = parser.parse_known_args(values[:separator])
    if not (
        args.exclude_cpu
        or args.exclude_llc_of_cpu
        or args.native_thread_count is not None
        or args.cpu_resource_check_only
    ):
        return
    try:
        plan = cpu_exclusion_plan(args.exclude_cpu, args.exclude_llc_of_cpu)
        native = native_thread_environment(args.native_thread_count)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    report = {"cpu_resource_plan": plan, "native_thread_environment": native}
    if args.cpu_resource_check_only:
        print(
            json.dumps(
                {**report, "affinity_applied": False, "model_started": False}, indent=2
            )
        )
        raise SystemExit(0)
    if plan is not None:
        if sorted(os.sched_getaffinity(0)) != plan["allowed_cpu_ids_before"]:
            raise RuntimeError("launcher CPU affinity changed after resource planning")
        report["cpu_affinity_receipt"] = exclude_cpus_before_exec(
            plan["excluded_cpu_ids"]
        )
    environment = {**os.environ, **native}
    print(
        "[OpenPICPU] READY: " + json.dumps(report, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    os.execve(
        sys.executable,
        [sys.executable, "-m", module_name, *remaining, *values[separator:]],
        environment,
    )
    raise RuntimeError("training CPU bootstrap exec unexpectedly returned")
