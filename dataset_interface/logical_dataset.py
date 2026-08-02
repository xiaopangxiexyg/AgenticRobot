"""Read-only episode selection for versioned logical dataset manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


LOGICAL_EPISODE_REF_SCHEMA_V1 = "dexteleop.logical_episode_ref.v1"


@dataclass(frozen=True)
class LogicalEpisodeSelection:
    manifest_path: Path
    manifest_sha256: str
    source_dataset_root: Path
    episode_indices: frozenset[int]
    episode_identity_by_index: Mapping[int, str]


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _camera_slots(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("logical episode camera_slots must be a JSON list")
    slots = tuple(_nonempty(item, name="camera_slots item") for item in value)
    if not slots or len(set(slots)) != len(slots):
        raise ValueError("logical episode camera_slots must be non-empty and unique")
    return slots


def load_logical_episode_selection(
    manifest_path: str | Path,
    *,
    source_dataset_root: str | Path,
    camera_slots: Iterable[str],
) -> LogicalEpisodeSelection:
    """Load the subset for one source without modifying or materializing it."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"logical episode manifest is missing: {path}")
    payload = path.read_bytes()
    source_root = Path(source_dataset_root).expanduser().resolve()
    expected_slots = tuple(str(slot) for slot in camera_slots)
    selected: dict[int, str] = {}
    identities: set[str] = set()
    record_count = 0

    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line.strip():
            continue
        record_count += 1
        try:
            value: Any = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        if value.get("schema") != LOGICAL_EPISODE_REF_SCHEMA_V1:
            raise ValueError(f"{path}:{line_number} has an unsupported schema")
        record_root = Path(
            _nonempty(value.get("source_dataset_root"), name="source_dataset_root")
        ).expanduser().resolve()
        if record_root != source_root:
            continue
        index = value.get("source_episode_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(
                f"{path}:{line_number} source_episode_index must be non-negative"
            )
        identity = _nonempty(value.get("episode_identity"), name="episode_identity")
        slots = _camera_slots(value.get("camera_slots"))
        if slots != expected_slots:
            raise ValueError(
                f"{path}:{line_number} camera_slots {slots!r} differ from "
                f"configured slots {expected_slots!r}"
            )
        if index in selected:
            raise ValueError(
                f"{path}:{line_number} duplicates source episode {index}"
            )
        if identity in identities:
            raise ValueError(
                f"{path}:{line_number} duplicates episode identity {identity!r}"
            )
        selected[index] = identity
        identities.add(identity)

    if not record_count:
        raise ValueError(f"logical episode manifest is empty: {path}")
    if not selected:
        raise ValueError(
            f"logical episode manifest {path} has no records for {source_root}"
        )
    return LogicalEpisodeSelection(
        manifest_path=path,
        manifest_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        source_dataset_root=source_root,
        episode_indices=frozenset(selected),
        episode_identity_by_index=dict(selected),
    )


__all__ = [
    "LOGICAL_EPISODE_REF_SCHEMA_V1",
    "LogicalEpisodeSelection",
    "load_logical_episode_selection",
]
