from __future__ import annotations

import os
import posixpath
import subprocess
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _tracked_modes() -> dict[str, str]:
    completed = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    modes: dict[str, str] = {}
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", maxsplit=1)
        mode = metadata.split(b" ", maxsplit=1)[0].decode("ascii")
        modes[os.fsdecode(encoded_path)] = mode
    return modes


def _escapes_repository(link_path: str, link_target: str) -> bool:
    target = PurePosixPath(link_target)
    if target.is_absolute():
        return True
    normalized = posixpath.normpath(
        str(PurePosixPath(link_path).parent.joinpath(target))
    )
    return normalized == ".." or normalized.startswith("../")


def test_tracked_symlinks_are_repository_relative_and_contained() -> None:
    tracked_symlinks = sorted(
        path for path, mode in _tracked_modes().items() if mode == "120000"
    )

    assert tracked_symlinks, "repository boundary audit found no tracked symlinks"
    for relative_path in tracked_symlinks:
        link_target = os.readlink(REPOSITORY_ROOT / relative_path)
        assert link_target, f"tracked symlink has an empty target: {relative_path}"
        assert not _escapes_repository(relative_path, link_target), (
            f"tracked symlink is host-bound or escapes the repository: "
            f"{relative_path} -> {link_target}"
        )


def test_openpi_compatibility_link_targets_the_managed_gitlink() -> None:
    modes = _tracked_modes()

    assert modes["model/openpi"] == "160000"
    assert modes["deployments/openpi/src"] == "120000"
    assert os.readlink(REPOSITORY_ROOT / "deployments/openpi/src") == (
        "../../model/openpi/src"
    )
