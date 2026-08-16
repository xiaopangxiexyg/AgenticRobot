from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .workspace import LabError, validate_repository_name, validate_task_id

LEGACY_QUEUE_PROFILE_SCHEMA_ID = "lab.queue-profile.v1"
QUEUE_PROFILE_SCHEMA_ID = "lab.queue-profile.v3"
ALLOWED_RISK_LEVELS = frozenset({"R0", "R1", "R2", "R3", "R4", "R5"})
ALLOWED_SNAPSHOT_MODES = frozenset({"clean", "draft"})
_MAX_PROFILE_BYTES = 256 * 1024


@dataclass(frozen=True)
class QueueProfile:
    profile_id: str
    description: str
    target_id: str
    risk_level: str
    requires_clean: bool
    repositories: tuple[str, ...]
    command_ids: tuple[str, ...]
    capabilities: tuple[str, ...]
    snapshot_modes: tuple[str, ...]
    path: Path
    sha256: str


def _string_list(payload: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(value, str) and value for value in payload)
    ):
        raise LabError(f"{label} must be a non-empty string list")
    if len(payload) != len(set(payload)):
        raise LabError(f"{label} contains duplicates")
    return tuple(payload)


def load_queue_profile(profile_root: Path, profile_id: str) -> QueueProfile:
    validate_task_id(profile_id)
    root = profile_root.expanduser().resolve()
    candidate = root / f"{profile_id}.json"
    if candidate.parent != root:
        raise LabError(f"queue Profile path escapes trusted root: {profile_id}")
    if candidate.is_symlink() or not candidate.is_file():
        raise LabError(f"unknown or unsafe queue Profile: {profile_id} ({candidate})")
    path = candidate.resolve()
    if path.parent != root:
        raise LabError(f"queue Profile path escapes trusted root: {profile_id}")
    try:
        if path.stat().st_size > _MAX_PROFILE_BYTES:
            raise LabError(f"queue Profile exceeds its size limit: {path}")
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read queue Profile {path}: {exc}") from exc
    common = {
        "schema",
        "profile_id",
        "description",
        "target_id",
        "risk_level",
        "repositories",
        "command_ids",
        "capabilities",
    }
    if not isinstance(payload, dict):
        raise LabError(f"queue Profile has an unexpected shape: {path}")
    schema = payload.get("schema")
    if schema == LEGACY_QUEUE_PROFILE_SCHEMA_ID:
        expected = {*common, "requires_clean"}
    elif schema == QUEUE_PROFILE_SCHEMA_ID:
        expected = {*common, "snapshot_modes"}
    else:
        raise LabError(f"unsupported queue Profile schema: {path}")
    if set(payload) != expected:
        raise LabError(f"queue Profile has an unexpected shape: {path}")
    if payload.get("profile_id") != profile_id:
        raise LabError(f"queue Profile identity mismatch: {path}")
    target_id = validate_task_id(str(payload.get("target_id", "")))
    risk_level = payload.get("risk_level")
    if risk_level not in ALLOWED_RISK_LEVELS:
        raise LabError(f"queue Profile has invalid risk level: {path}")
    snapshot_modes: tuple[str, ...]
    if schema == LEGACY_QUEUE_PROFILE_SCHEMA_ID:
        if payload.get("requires_clean") is not True:
            raise LabError(
                f"legacy queue Profile must require a clean snapshot: {path}"
            )
        snapshot_modes = ("clean",)
    else:
        snapshot_modes = _string_list(
            payload.get("snapshot_modes"), label="queue Profile snapshot modes"
        )
        if not set(snapshot_modes).issubset(ALLOWED_SNAPSHOT_MODES):
            raise LabError(f"queue Profile has an invalid snapshot mode: {path}")
        if "clean" not in snapshot_modes:
            raise LabError(f"queue Profile must retain clean snapshot support: {path}")
    repositories = tuple(
        validate_repository_name(value)
        for value in _string_list(
            payload.get("repositories"), label="queue Profile repositories"
        )
    )
    command_ids = tuple(
        validate_task_id(value)
        for value in _string_list(
            payload.get("command_ids"), label="queue Profile command IDs"
        )
    )
    capabilities = tuple(
        validate_task_id(value)
        for value in _string_list(
            payload.get("capabilities"), label="queue Profile capabilities"
        )
    )
    description = payload.get("description")
    if not isinstance(description, str) or not description or len(description) > 512:
        raise LabError(f"queue Profile has invalid description: {path}")
    return QueueProfile(
        profile_id=profile_id,
        description=description,
        target_id=target_id,
        risk_level=str(risk_level),
        requires_clean=snapshot_modes == ("clean",),
        repositories=repositories,
        command_ids=command_ids,
        capabilities=capabilities,
        snapshot_modes=snapshot_modes,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
