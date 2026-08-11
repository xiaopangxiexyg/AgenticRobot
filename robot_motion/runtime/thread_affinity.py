"""Small fail-closed Linux thread-affinity helpers for control workers."""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Iterable


def available_cpu_ids() -> tuple[int, ...]:
    """Return CPUs currently available to the calling thread.

    The returned mask already reflects taskset/cpuset/cgroup restrictions and
    is therefore the only mask that deployment CLI validation should expose.
    """

    getter = getattr(os, "sched_getaffinity", None)
    if not callable(getter):
        raise RuntimeError("thread CPU affinity is unavailable on this platform")
    return tuple(sorted(int(cpu) for cpu in getter(0)))


def cpu_thread_sibling_ids(cpu_id: int) -> tuple[int, ...]:
    """Return Linux SMT siblings for one logical CPU, including itself."""

    if isinstance(cpu_id, bool) or int(cpu_id) != cpu_id or int(cpu_id) < 0:
        raise ValueError("cpu_id must be a non-negative integer")
    cpu = int(cpu_id)
    path = Path(
        f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    )
    try:
        payload = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(
            f"CPU topology is unavailable for logical CPU {cpu}"
        ) from exc
    siblings: set[int] = set()
    for part in payload.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" not in token:
            siblings.add(int(token))
            continue
        lower_text, upper_text = token.split("-", 1)
        lower = int(lower_text)
        upper = int(upper_text)
        if lower < 0 or upper < lower:
            raise RuntimeError(
                f"invalid CPU topology for logical CPU {cpu}: {payload!r}"
            )
        siblings.update(range(lower, upper + 1))
    if cpu not in siblings:
        raise RuntimeError(
            f"CPU topology for logical CPU {cpu} does not include itself: "
            f"{payload!r}"
        )
    return tuple(sorted(siblings))


def bind_current_thread_to_cpu(
    requested_cpu: int | None,
    *,
    role: str,
) -> dict[str, Any]:
    """Pin the calling native thread and return a serializable receipt.

    ``None`` is an explicit rollback/no-op. When a CPU is requested, inability
    to apply or verify the exact one-CPU mask raises instead of silently
    claiming isolation.
    """

    role_name = str(role).strip()
    if not role_name:
        raise ValueError("thread affinity role must be non-empty")
    if requested_cpu is not None and (
        isinstance(requested_cpu, bool) or int(requested_cpu) != requested_cpu
    ):
        raise ValueError("requested_cpu must be a non-negative integer or None")
    cpu = None if requested_cpu is None else int(requested_cpu)
    if cpu is not None and cpu < 0:
        raise ValueError("requested_cpu must be a non-negative integer or None")

    native_thread_id = int(threading.get_native_id())
    if cpu is None:
        return {
            "role": role_name,
            "requested": False,
            "requested_cpu": None,
            "applied": False,
            "native_thread_id": native_thread_id,
            "allowed_cpu_ids_before": None,
            "effective_cpu_ids": None,
        }
    before = available_cpu_ids()
    if cpu not in before:
        raise RuntimeError(
            f"requested CPU {cpu} for {role_name} is outside the calling "
            f"thread affinity mask {list(before)}"
        )

    setter = getattr(os, "sched_setaffinity", None)
    getter = getattr(os, "sched_getaffinity", None)
    if not callable(setter) or not callable(getter):
        raise RuntimeError("thread CPU affinity is unavailable on this platform")
    setter(native_thread_id, {cpu})
    after = tuple(sorted(int(value) for value in getter(native_thread_id)))
    if after != (cpu,):
        raise RuntimeError(
            f"failed to verify CPU affinity for {role_name}: requested={[cpu]} "
            f"effective={list(after)}"
        )
    return {
        "role": role_name,
        "requested": True,
        "requested_cpu": cpu,
        "applied": True,
        "native_thread_id": native_thread_id,
        "allowed_cpu_ids_before": list(before),
        "effective_cpu_ids": list(after),
    }


def _native_thread_ids() -> tuple[int, ...]:
    try:
        names = os.listdir("/proc/self/task")
    except OSError as exc:
        raise RuntimeError(
            "native process-thread enumeration is unavailable on this platform"
        ) from exc
    thread_ids = tuple(
        sorted(int(name) for name in names if str(name).isdigit())
    )
    if not thread_ids:
        raise RuntimeError("native process-thread enumeration returned no threads")
    return thread_ids


def _cpu_id_set(values: Iterable[int], *, name: str) -> set[int]:
    result: set[int] = set()
    for value in values:
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"{name} must contain only non-negative integers")
        result.add(int(value))
    return result


def exclude_cpu_ids_from_current_process(
    excluded_cpu_ids: Iterable[int],
    *,
    role: str,
    exempt_native_thread_ids: Iterable[int] = (),
    max_passes: int = 3,
) -> dict[str, Any]:
    """Remove CPUs from every non-exempt native thread in this process.

    This is a process-local scheduling reservation, not an OS-wide exclusive
    cpuset. Existing native threads are migrated without broadening their
    masks, and later threads inherit the mask of their creating thread. The
    caller must explicitly exempt any already-pinned authoritative worker.
    """

    role_name = str(role).strip()
    if not role_name:
        raise ValueError("process affinity role must be non-empty")
    excluded = _cpu_id_set(excluded_cpu_ids, name="excluded_cpu_ids")
    exempt = _cpu_id_set(
        exempt_native_thread_ids,
        name="exempt_native_thread_ids",
    )
    if isinstance(max_passes, bool) or int(max_passes) != max_passes:
        raise ValueError("max_passes must be a positive integer")
    pass_limit = int(max_passes)
    if pass_limit <= 0:
        raise ValueError("max_passes must be a positive integer")

    caller_before = available_cpu_ids()
    if not excluded:
        return {
            "role": role_name,
            "requested": False,
            "applied": False,
            "excluded_cpu_ids": [],
            "exempt_native_thread_ids": sorted(exempt),
            "caller_cpu_ids_before": list(caller_before),
            "caller_cpu_ids_after": list(caller_before),
            "native_thread_count": len(_native_thread_ids()),
            "native_threads_migrated": 0,
            "passes": 0,
        }
    missing = excluded.difference(caller_before)
    if missing:
        raise RuntimeError(
            f"requested CPU exclusion for {role_name} is outside the calling "
            f"thread affinity mask: missing={sorted(missing)} "
            f"allowed={list(caller_before)}"
        )

    setter = getattr(os, "sched_setaffinity", None)
    getter = getattr(os, "sched_getaffinity", None)
    if not callable(setter) or not callable(getter):
        raise RuntimeError("thread CPU affinity is unavailable on this platform")

    migrated: set[int] = set()
    final_thread_ids: tuple[int, ...] = ()
    completed_passes = 0
    for pass_index in range(1, pass_limit + 1):
        completed_passes = pass_index
        thread_ids_before = _native_thread_ids()
        for native_thread_id in thread_ids_before:
            if native_thread_id in exempt:
                continue
            try:
                before = set(int(cpu) for cpu in getter(native_thread_id))
            except ProcessLookupError:
                continue
            target = before.difference(excluded)
            if not target:
                raise RuntimeError(
                    f"cannot reserve CPUs {sorted(excluded)} for {role_name}: "
                    f"native thread {native_thread_id} has no remaining CPU "
                    f"without broadening its mask {sorted(before)}"
                )
            if target != before:
                try:
                    setter(native_thread_id, target)
                except ProcessLookupError:
                    continue
                migrated.add(native_thread_id)
            try:
                after = set(int(cpu) for cpu in getter(native_thread_id))
            except ProcessLookupError:
                continue
            overlap = after.intersection(excluded)
            if overlap:
                raise RuntimeError(
                    f"failed to exclude CPUs for {role_name}: native thread "
                    f"{native_thread_id} still allows {sorted(overlap)}"
                )

        final_thread_ids = _native_thread_ids()
        if final_thread_ids == thread_ids_before:
            break
    else:
        raise RuntimeError(
            f"native thread set did not stabilize while reserving CPUs for "
            f"{role_name} after {pass_limit} passes"
        )

    for native_thread_id in final_thread_ids:
        if native_thread_id in exempt:
            continue
        try:
            effective = set(int(cpu) for cpu in getter(native_thread_id))
        except ProcessLookupError:
            continue
        overlap = effective.intersection(excluded)
        if overlap:
            raise RuntimeError(
                f"process CPU reservation verification failed for {role_name}: "
                f"native thread {native_thread_id} allows {sorted(overlap)}"
            )

    caller_after = available_cpu_ids()
    return {
        "role": role_name,
        "requested": True,
        "applied": True,
        "excluded_cpu_ids": sorted(excluded),
        "exempt_native_thread_ids": sorted(exempt),
        "caller_cpu_ids_before": list(caller_before),
        "caller_cpu_ids_after": list(caller_after),
        "native_thread_count": len(final_thread_ids),
        "native_threads_migrated": len(migrated),
        "passes": completed_passes,
    }


__all__ = [
    "available_cpu_ids",
    "bind_current_thread_to_cpu",
    "cpu_thread_sibling_ids",
    "exclude_cpu_ids_from_current_process",
]
