"""Small deterministic file and locator helpers for prepared views."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from dataset_contracts.canonical_encoding import (
    canonical_bytes,
    canonical_loads,
)

from .models import ALLOWED_MODALITIES, PreparedTrainingViewError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_cjsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    domain: str,
) -> None:
    payload = b"".join(
        canonical_bytes(dict(record), domain=domain) + b"\n"
        for record in records
    )
    write_bytes(path, payload)


def read_cjsonl(path: Path, *, domain: str) -> tuple[Mapping[str, Any], ...]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PreparedTrainingViewError(f"Cannot read {path.name}: {exc}") from exc
    if payload and not payload.endswith(b"\n"):
        raise PreparedTrainingViewError(f"{path.name} has a torn final row.")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise PreparedTrainingViewError(
                f"{path.name} contains an empty row at {line_number}."
            )
        value = canonical_loads(line, expected_domain=domain)
        if not isinstance(value, Mapping):
            raise PreparedTrainingViewError(
                f"{path.name}:{line_number} is not an object."
            )
        records.append(value)
    return tuple(records)


def relative_locator(*, root: Path, target: Path) -> str:
    value = Path(os.path.relpath(target.resolve(), start=root.resolve())).as_posix()
    locator = PurePosixPath(value)
    if locator.is_absolute() or value in {"", "."}:
        if target.resolve() == root.resolve():
            return "."
        raise PreparedTrainingViewError("Cannot derive a relative source locator.")
    return value


def source_relative_path(source_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(source_root.resolve())
    except ValueError as exc:
        raise PreparedTrainingViewError(
            f"Source artifact escapes dataset root: {resolved}."
        ) from exc
    return relative.as_posix()


def normalized_modalities(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(sorted({str(value).strip().lower() for value in values}))
    if not result or any(value not in ALLOWED_MODALITIES for value in result):
        raise PreparedTrainingViewError(
            f"modality_selection must be a non-empty subset of "
            f"{sorted(ALLOWED_MODALITIES)}."
        )
    return result


def normalized_projection(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if (
        not result
        or "timestamp" not in result
        or any(not value for value in result)
        or len(result) != len(set(result))
    ):
        raise PreparedTrainingViewError(
            "column_projection must contain unique fields including timestamp."
        )
    return result


__all__ = [
    "fsync_directory",
    "normalized_modalities",
    "normalized_projection",
    "read_cjsonl",
    "relative_locator",
    "sha256_file",
    "source_relative_path",
    "write_bytes",
    "write_cjsonl",
]
