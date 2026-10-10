"""Dependency-free Linux CPU topology for control and model launchers."""

from __future__ import annotations

import re
from pathlib import Path


def cpu_last_level_cache_ids(cpu_id: int) -> tuple[int, ...]:
    """Return the highest data/unified cache's complete sharing group.

    This is topology, not an allowed affinity mask. Callers must intersect it
    with their inherited mask and must not broaden that mask.
    """

    if isinstance(cpu_id, bool) or int(cpu_id) != cpu_id or int(cpu_id) < 0:
        raise ValueError("cpu_id must be a non-negative integer")
    cpu = int(cpu_id)
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache")
    caches: list[tuple[int, tuple[int, ...]]] = []
    try:
        for entry in root.glob("index*"):
            kind = (entry / "type").read_text(encoding="utf-8").strip()
            if kind == "Instruction":
                continue
            if kind not in {"Data", "Unified"}:
                raise ValueError(f"unknown cache type: {kind!r}")
            level = int((entry / "level").read_text(encoding="utf-8"))
            payload = (entry / "shared_cpu_list").read_text(encoding="utf-8").strip()
            if (
                level < 1
                or re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", payload)
                is None
            ):
                raise ValueError(
                    f"invalid cache level or CPU list: {level}, {payload!r}"
                )
            shared: set[int] = set()
            for token in payload.split(","):
                bounds = [int(value) for value in token.split("-")]
                lower, upper = bounds[0], bounds[-1]
                if upper < lower:
                    raise ValueError(f"reversed CPU range: {token!r}")
                shared.update(range(lower, upper + 1))
            if cpu not in shared:
                raise ValueError(f"cache CPU list does not include CPU {cpu}")
            caches.append((level, tuple(sorted(shared))))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"invalid or unavailable cache topology for CPU {cpu}: {exc}"
        ) from exc
    if not caches:
        raise RuntimeError(f"cache topology is unavailable for CPU {cpu}")
    highest = max(level for level, _ in caches)
    groups = {group for level, group in caches if level == highest}
    if len(groups) != 1:
        raise RuntimeError(f"ambiguous last-level cache topology for CPU {cpu}")
    return next(iter(groups))
