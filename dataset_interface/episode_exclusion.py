"""Reversible, episode-identity-bound training exclusion for formal datasets."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .episode_curation import source_episode_manifest_hash


DECISIONS_RELATIVE_PATH = Path("meta/curation/episode_decisions.jsonl")
SCHEMA = "dexteleop.episode_exclusion_decision.v1"


def read_episode_exclusions(dataset_root: str | Path) -> dict[int, bool]:
    """Return latest decisions; reject corrupt or stale episode bindings."""

    root = Path(dataset_root).expanduser().resolve()
    path = root / DECISIONS_RELATIVE_PATH
    if not path.is_file():
        return {}
    decisions: dict[int, bool] = {}
    with path.open("r", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        for number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
                index = record["episode_index"]
                if (
                    record["schema"] != SCHEMA
                    or type(index) is not int
                    or index < 0
                    or type(record["excluded"]) is not bool
                    or record["episode_manifest_hash"]
                    != source_episode_manifest_hash(root, index)
                ):
                    raise ValueError("invalid or stale episode decision")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid episode exclusion {path}:{number}: {exc}") from exc
            decisions[index] = record["excluded"]
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return decisions


def set_episode_excluded(
    dataset_root: str | Path, episode_index: int, *, excluded: bool
) -> None:
    """Append a durable exclusion/restoration decision without altering samples."""

    root = Path(dataset_root).expanduser().resolve()
    if type(episode_index) is not int or episode_index < 0 or type(excluded) is not bool:
        raise ValueError("episode_index and excluded must be a nonnegative int and bool")
    manifest_hash = source_episode_manifest_hash(root, episode_index)
    if manifest_hash is None:
        raise ValueError(f"episode {episode_index} is not finalized in {root}")
    path = root / DECISIONS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": SCHEMA,
        "episode_index": episode_index,
        "episode_manifest_hash": manifest_hash,
        "excluded": excluded,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    read_episode_exclusions(root)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if os.write(fd, encoded) != len(encoded):
            raise OSError("episode exclusion decision was only partially written")
        os.fsync(fd)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
