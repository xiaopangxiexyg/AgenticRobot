"""Atomic immutable canonical-file publication shared by collection sidecars."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_bytes, canonical_loads


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_immutable_canonical(
    path: Path,
    payload: Mapping[str, Any],
    *,
    domain: str,
) -> bool:
    """Publish immutable canonical content; return whether it was idempotent."""

    encoded = canonical_bytes(dict(payload), domain=domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != encoded:
            raise ValueError(f"Immutable canonical artifact differs at {path}.")
        canonical_loads(existing, expected_domain=domain)
        return True
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return False


__all__ = ["fsync_directory", "publish_immutable_canonical"]
