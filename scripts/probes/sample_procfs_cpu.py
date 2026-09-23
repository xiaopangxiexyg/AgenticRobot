"""Sample CPU execution and scheduler wait from procfs without sysstat."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _stat(path: Path) -> tuple[int, int, int] | None:
    try:
        raw = path.read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        return int(fields[1]), int(fields[11]) + int(fields[12]), int(fields[36])
    except (IndexError, OSError, ValueError):
        return None


def _descendants(roots: set[int]) -> tuple[set[int], dict[int, int]]:
    parents: dict[int, int] = {}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            stat = _stat(Path(entry.path) / "stat")
            if stat is not None:
                parents[pid] = stat[0]

    selected = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected, parents


def _threads(pid: int) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    try:
        tids = [int(entry.name) for entry in os.scandir(f"/proc/{pid}/task")]
    except OSError:
        return records
    for tid in sorted(tids):
        task = Path(f"/proc/{pid}/task/{tid}")
        stat = _stat(task / "stat")
        if stat is None:
            continue
        record = {"tid": tid, "cpu_ticks": stat[1], "last_cpu": stat[2]}
        try:
            sched = (task / "schedstat").read_text(encoding="utf-8").split()
            record["run_ns"] = int(sched[0])
            record["wait_ns"] = int(sched[1])
        except (IndexError, OSError, ValueError):
            pass
        records.append(record)
    return records


def _cpu_times() -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0].startswith("cpu") and (
            fields[0] == "cpu" or fields[0][3:].isdecimal()
        ):
            result[fields[0]] = [int(value) for value in fields[1:]]
    return result


def _pressure_cpu() -> str | None:
    try:
        return Path("/proc/pressure/cpu").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-sec", type=int, required=True)
    parser.add_argument("--pid", type=int, action="append", required=True)
    args = parser.parse_args()
    if args.duration_sec <= 0 or any(pid <= 0 for pid in args.pid):
        parser.error("duration and PIDs must be positive")

    roots = set(args.pid)
    for sample_index in range(args.duration_sec):
        sample_start = time.monotonic()
        selected, parents = _descendants(roots)
        processes = [
            {"pid": pid, "ppid": parents.get(pid), "threads": _threads(pid)}
            for pid in sorted(selected)
            if pid in parents
        ]
        print(
            json.dumps(
                {
                    "schema": "teleop_cpu_procfs/v1",
                    "sample_index": sample_index,
                    "epoch_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    "clock_ticks_per_sec": os.sysconf("SC_CLK_TCK"),
                    "root_pids": sorted(roots),
                    "loadavg": Path("/proc/loadavg").read_text(encoding="utf-8").strip(),
                    "pressure_cpu": _pressure_cpu(),
                    "cpu_times": _cpu_times(),
                    "processes": processes,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        time.sleep(max(0.0, sample_start + 1.0 - time.monotonic()))


if __name__ == "__main__":
    main()
