from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import git
from .preparation_profile import PreparationProfile
from .remote_request import OID_PATTERN, SHA256_PATTERN
from .workspace import (
    LabError,
    _atomic_write_text,
    validate_repository_name,
    validate_task_id,
)

MAINTENANCE_RESERVATION_SCHEMA_ID = "lab.maintenance-reservation.v1"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise LabError(f"maintenance reservation has invalid {label}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError(f"maintenance reservation has invalid {label}") from exc
    if parsed.tzinfo is None:
        raise LabError(f"maintenance reservation {label} requires a timezone")
    return parsed


def _state_paths(
    profile: PreparationProfile,
    repository: str,
) -> tuple[Path, Path]:
    validate_repository_name(repository)
    install_root = profile.control_plane.install_root
    if (
        install_root.is_symlink()
        or not install_root.is_dir()
        or install_root.resolve() != install_root
    ):
        raise LabError("maintenance reservation install root is unavailable")
    state_root = install_root / "maintenance-reservations"
    if state_root.exists() or state_root.is_symlink():
        if state_root.is_symlink() or not state_root.is_dir():
            raise LabError("maintenance reservation state root is unsafe")
    else:
        state_root.mkdir(mode=0o700)
    return state_root / f"{repository}.json", state_root / f"{repository}.lock"


@contextmanager
def _locked_state(
    profile: PreparationProfile,
    repository: str,
) -> Iterator[Path]:
    state_path, lock_path = _state_paths(profile, repository)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield state_path
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _load_state(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise LabError("maintenance reservation state is unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read maintenance reservation state: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError("maintenance reservation state must be an object")
    return payload


def validate_maintenance_reservation(
    receipt: object,
    *,
    profile: PreparationProfile,
    repository: str,
    require_active: bool,
) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "preparation_profile_id",
        "preparation_profile_sha256",
        "host_id",
        "repository",
        "checkout_path",
        "local_commit",
        "reservation_id",
        "status",
        "managed_runtime_starts_blocked",
        "unmanaged_runtime_authorized",
        "device_access",
        "acquired_at",
        "expires_at",
        "released_at",
        "receipt_sha256",
    }:
        raise LabError("maintenance reservation has an unexpected shape")
    validate_repository_name(repository)
    placement = profile.stable_sync.get(repository)
    if placement is None:
        raise LabError(f"repository has no maintenance reservation policy: {repository}")
    reservation_id = receipt.get("reservation_id")
    if not isinstance(reservation_id, str):
        raise LabError("maintenance reservation has invalid ID")
    validate_task_id(reservation_id)
    status = receipt.get("status")
    released_at = receipt.get("released_at")
    if (
        receipt.get("schema") != MAINTENANCE_RESERVATION_SCHEMA_ID
        or receipt.get("preparation_profile_id") != profile.profile_id
        or receipt.get("preparation_profile_sha256") != profile.sha256
        or receipt.get("host_id") != profile.host_id
        or receipt.get("repository") != repository
        or receipt.get("checkout_path") != str(placement.path)
        or not OID_PATTERN.fullmatch(str(receipt.get("local_commit", "")))
        or status not in {"ACTIVE", "RELEASED"}
        or (status == "ACTIVE" and released_at is not None)
        or (status == "RELEASED" and released_at is None)
        or receipt.get("managed_runtime_starts_blocked") is not (status == "ACTIVE")
        or receipt.get("unmanaged_runtime_authorized") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("maintenance reservation identity or authority is unsafe")
    acquired = _timestamp(receipt.get("acquired_at"), label="acquired_at")
    expires = _timestamp(receipt.get("expires_at"), label="expires_at")
    if expires <= acquired or (released_at is not None and _timestamp(released_at, label="released_at") < acquired):
        raise LabError("maintenance reservation timestamps are invalid")
    if require_active and (
        status != "ACTIVE" or not datetime.now(timezone.utc) <= expires
    ):
        raise LabError("maintenance reservation is not active")
    digest = receipt.get("receipt_sha256")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        not SHA256_PATTERN.fullmatch(str(digest))
        or digest != hashlib.sha256(_canonical_json(content)).hexdigest()
    ):
        raise LabError("maintenance reservation hash mismatch")
    return receipt


def acquire_maintenance_reservation(
    profile: PreparationProfile,
    *,
    repository: str,
    reservation_id: str,
    validity_seconds: int = 60,
) -> dict[str, object]:
    validate_task_id(reservation_id)
    if not 1 <= validity_seconds <= 60:
        raise LabError("maintenance reservation validity must be between 1 and 60 seconds")
    placement = profile.stable_sync.get(repository)
    if placement is None:
        raise LabError(f"repository has no maintenance reservation policy: {repository}")
    if git.current_branch(placement.path) != placement.branch or git.status_short(
        placement.path
    ):
        raise LabError("maintenance reservation requires a clean protected checkout")
    with _locked_state(profile, repository) as state_path:
        existing = _load_state(state_path)
        if existing is not None:
            validated = validate_maintenance_reservation(
                existing,
                profile=profile,
                repository=repository,
                require_active=False,
            )
            if (
                validated.get("status") == "ACTIVE"
                and datetime.now(timezone.utc)
                <= _timestamp(validated.get("expires_at"), label="expires_at")
            ):
                raise LabError("maintenance reservation is already active")
        acquired = datetime.now(timezone.utc)
        content: dict[str, object] = {
            "schema": MAINTENANCE_RESERVATION_SCHEMA_ID,
            "preparation_profile_id": profile.profile_id,
            "preparation_profile_sha256": profile.sha256,
            "host_id": profile.host_id,
            "repository": repository,
            "checkout_path": str(placement.path),
            "local_commit": git.current_commit(placement.path),
            "reservation_id": reservation_id,
            "status": "ACTIVE",
            "managed_runtime_starts_blocked": True,
            "unmanaged_runtime_authorized": False,
            "device_access": False,
            "acquired_at": acquired.isoformat(),
            "expires_at": (acquired + timedelta(seconds=validity_seconds)).isoformat(),
            "released_at": None,
        }
        receipt = {
            **content,
            "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
        }
        _atomic_write_text(state_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt


def assert_managed_runtime_start_allowed(
    profile: PreparationProfile,
    *,
    repository: str,
) -> dict[str, object]:
    """Fail closed for launchers while an unexpired maintenance reservation is active."""

    with _locked_state(profile, repository) as state_path:
        state = _load_state(state_path)
        active_reservation: str | None = None
        if state is not None:
            validated = validate_maintenance_reservation(
                state,
                profile=profile,
                repository=repository,
                require_active=False,
            )
            expires = _timestamp(validated.get("expires_at"), label="expires_at")
            if validated.get("status") == "ACTIVE" and datetime.now(timezone.utc) <= expires:
                active_reservation = str(validated["reservation_id"])
        if active_reservation is not None:
            raise LabError(
                f"managed runtime start blocked by maintenance reservation: "
                f"{active_reservation}"
            )
        content: dict[str, object] = {
            "schema": "lab.managed-runtime-start-gate.v1",
            "preparation_profile_sha256": profile.sha256,
            "repository": repository,
            "allowed": True,
            "active_maintenance_reservation": None,
            "device_access": False,
        }
        return {
            **content,
            "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
        }


def release_maintenance_reservation(
    profile: PreparationProfile,
    *,
    repository: str,
    active_receipt: object,
) -> dict[str, object]:
    active = validate_maintenance_reservation(
        active_receipt,
        profile=profile,
        repository=repository,
        require_active=True,
    )
    with _locked_state(profile, repository) as state_path:
        state = _load_state(state_path)
        if state != active:
            raise LabError("maintenance reservation state changed before release")
        content = {
            key: value
            for key, value in active.items()
            if key not in {"status", "managed_runtime_starts_blocked", "released_at", "receipt_sha256"}
        }
        content.update(
            {
                "status": "RELEASED",
                "managed_runtime_starts_blocked": False,
                "released_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        released: dict[str, object] = {
            **content,
            "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
        }
        _atomic_write_text(state_path, json.dumps(released, indent=2, sort_keys=True) + "\n")
        return validate_maintenance_reservation(
            released,
            profile=profile,
            repository=repository,
            require_active=False,
        )
