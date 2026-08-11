"""Shared process-level quiescence checks for collection maintenance.

Collection maintenance may compete with formal capture even when it targets a
different dataset.  This module owns the Linux process inspection used by
recompression, modality storage, and abandoned-capture recovery; individual
maintenance capabilities must not infer collection activity from arbitrary
command-line substrings.
"""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path


FORMAL_COLLECTION_ENTRY_MODULES = frozenset(
    {
        "teleop_runtime.collection",
        "teleop_runtime.collection.run_collect",
        "teleop_runtime.debug.run_vive_rm65_real_teleop",
    }
)


def is_formal_collection_argv(argv: Iterable[str]) -> bool:
    """Return whether *argv* launches one of the supported collection modules.

    Matching NUL-delimited argv tokens avoids treating a shell, test runner, or
    sandbox parent as collection merely because one of its string arguments
    contains a collection command. Wrappers such as ``conda run`` and
    ``taskset`` remain covered because they preserve the separate ``-m`` and
    module-name arguments; their Python child is covered independently too.
    """

    tokens = tuple(str(token) for token in argv)
    return any(
        token == "-m"
        and index + 1 < len(tokens)
        and tokens[index + 1] in FORMAL_COLLECTION_ENTRY_MODULES
        for index, token in enumerate(tokens)
    )


def active_formal_collection_processes() -> tuple[str, ...]:
    """Return active formal-collection command lines on Linux."""

    proc = Path("/proc")
    if not proc.is_dir():
        return ()
    current_pid = os.getpid()
    matches: list[str] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        try:
            payload = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = tuple(
            token.decode("utf-8", errors="replace")
            for token in payload.split(b"\0")
            if token
        )
        if not argv or not is_formal_collection_argv(argv):
            continue
        matches.append(f"pid={entry.name} {' '.join(argv)}")
    return tuple(sorted(matches))


__all__ = [
    "FORMAL_COLLECTION_ENTRY_MODULES",
    "active_formal_collection_processes",
    "is_formal_collection_argv",
]
