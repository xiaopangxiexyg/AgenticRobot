from __future__ import annotations

import fcntl
import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import git
from .stable_sync import plan_stable_checkout_sync, validate_stable_sync_plan
from .workspace import (
    LabError,
    _atomic_write_text,
    validate_branch,
    validate_repository_name,
    validate_task_id,
)

MAINTENANCE_RESERVATION_SCHEMA_ID = "lab.maintenance-reservation.v1"
MANAGED_RUNTIME_GATE_SCHEMA_ID = "lab.managed-runtime-start-gate.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


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


def _safe_state_root(path: Path) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise LabError(f"maintenance state root is unsafe: {requested}")
    if requested.exists() and not requested.is_dir():
        raise LabError(f"maintenance state root is not a directory: {requested}")
    requested.mkdir(parents=True, exist_ok=True)
    resolved = requested.resolve()
    if requested.absolute() != resolved:
        raise LabError(f"maintenance state root is not canonical: {requested}")
    return resolved


@dataclass(frozen=True)
class StableCheckoutTarget:
    """Host-local protected checkout identity, independent of project payload code."""

    project_id: str
    host_id: str
    repository: str
    checkout_path: Path
    canonical_identity: str
    branch: str
    remote_ref: str
    state_root: Path

    def __post_init__(self) -> None:
        validate_task_id(self.project_id)
        validate_task_id(self.host_id)
        validate_repository_name(self.repository)
        validate_branch(self.branch)
        if self.remote_ref != f"refs/remotes/origin/{self.branch}":
            raise LabError("maintenance target requires the protected origin branch ref")
        requested = self.checkout_path.expanduser()
        root = requested.resolve()
        if (
            not requested.is_absolute()
            or requested.is_symlink()
            or requested.absolute() != root
            or git.repository_root(root) != root
        ):
            raise LabError(f"maintenance target is not a real repository root: {root}")
        if git.origin_identity(root) != self.canonical_identity:
            raise LabError("maintenance target canonical repository identity mismatch")
        object.__setattr__(self, "checkout_path", root)
        object.__setattr__(self, "state_root", _safe_state_root(self.state_root))

    def identity(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "host_id": self.host_id,
            "repository": self.repository,
            "checkout_path": str(self.checkout_path),
            "canonical_identity": self.canonical_identity,
            "branch": self.branch,
            "remote_ref": self.remote_ref,
        }

    @property
    def policy_sha256(self) -> str:
        return _hash(self.identity())


def _state_paths(target: StableCheckoutTarget) -> tuple[Path, Path]:
    root = target.state_root / "maintenance-reservations" / target.project_id
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise LabError(f"maintenance reservation root is unsafe: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not root.resolve().is_relative_to(target.state_root):
        raise LabError("maintenance reservation root escapes configured state")
    return root / f"{target.repository}.json", root / f"{target.repository}.lock"


@contextmanager
def _locked_state(target: StableCheckoutTarget) -> Iterator[Path]:
    state_path, lock_path = _state_paths(target)
    if lock_path.is_symlink():
        raise LabError(f"maintenance lock path is a symlink: {lock_path}")
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
        raise LabError(f"maintenance reservation state is unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read maintenance reservation state: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError("maintenance reservation state must be an object")
    return payload


def _validate_target_plan(
    target: StableCheckoutTarget,
    plan: object,
) -> dict[str, object]:
    validated = validate_stable_sync_plan(plan)
    if (
        validated.get("repository") != str(target.checkout_path)
        or validated.get("canonical_identity") != target.canonical_identity
        or validated.get("branch") != target.branch
        or validated.get("remote_ref") != target.remote_ref
    ):
        raise LabError("maintenance target does not match stable sync plan")
    return validated


def validate_maintenance_reservation(
    receipt: object,
    *,
    target: StableCheckoutTarget,
    require_unexpired: bool,
) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "target",
        "policy_sha256",
        "plan_sha256",
        "local_commit",
        "local_tree_oid",
        "checkout_status",
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
    status = receipt.get("status")
    released_at = receipt.get("released_at")
    if (
        receipt.get("schema") != MAINTENANCE_RESERVATION_SCHEMA_ID
        or receipt.get("target") != target.identity()
        or receipt.get("policy_sha256") != target.policy_sha256
        or not _SHA256_PATTERN.fullmatch(str(receipt.get("plan_sha256", "")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("local_commit", "")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("local_tree_oid", "")))
        or not isinstance(receipt.get("checkout_status"), list)
        or not all(isinstance(item, str) for item in receipt["checkout_status"])
        or status not in {"ACTIVE", "RELEASED"}
        or (status == "ACTIVE" and released_at is not None)
        or (status == "RELEASED" and released_at is None)
        or receipt.get("managed_runtime_starts_blocked") is not (status == "ACTIVE")
        or receipt.get("unmanaged_runtime_authorized") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("maintenance reservation identity or authority is unsafe")
    reservation_id = receipt.get("reservation_id")
    if not isinstance(reservation_id, str):
        raise LabError("maintenance reservation has invalid ID")
    validate_task_id(reservation_id)
    acquired = _timestamp(receipt.get("acquired_at"), label="acquired_at")
    expires = _timestamp(receipt.get("expires_at"), label="expires_at")
    if expires <= acquired or (
        released_at is not None
        and _timestamp(released_at, label="released_at") < acquired
    ):
        raise LabError("maintenance reservation timestamps are invalid")
    if require_unexpired and (
        status != "ACTIVE" or datetime.now(timezone.utc) > expires
    ):
        raise LabError("maintenance reservation is not active")
    digest = receipt.get("receipt_sha256")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not _SHA256_PATTERN.fullmatch(str(digest)) or digest != _hash(content):
        raise LabError("maintenance reservation hash mismatch")
    return receipt


def acquire_maintenance_reservation(
    target: StableCheckoutTarget,
    *,
    plan: object,
    reservation_id: str,
    validity_seconds: int = 120,
) -> dict[str, object]:
    validated_plan = _validate_target_plan(target, plan)
    if validated_plan.get("action") != "FAST_FORWARD_READY":
        raise LabError("maintenance reservation requires a fast-forward-ready plan")
    current = plan_stable_checkout_sync(
        target.checkout_path,
        expected_identity=target.canonical_identity,
        branch=target.branch,
        remote_ref=target.remote_ref,
    )
    if current != validated_plan:
        raise LabError("stable sync state changed before maintenance reservation")
    return acquire_checkout_maintenance_reservation(
        target,
        operation_plan_sha256=str(validated_plan["plan_sha256"]),
        local_commit=str(validated_plan["local_commit"]),
        local_tree_oid=str(validated_plan["local_tree_oid"]),
        expected_status=(),
        reservation_id=reservation_id,
        validity_seconds=validity_seconds,
    )


def acquire_checkout_maintenance_reservation(
    target: StableCheckoutTarget,
    *,
    operation_plan_sha256: str,
    local_commit: str,
    local_tree_oid: str,
    expected_status: tuple[str, ...],
    reservation_id: str,
    validity_seconds: int = 120,
) -> dict[str, object]:
    """Reserve one clean checkout for a trusted, content-addressed operation plan."""

    validate_task_id(reservation_id)
    if not 5 <= validity_seconds <= 300:
        raise LabError("maintenance reservation validity must be between 5 and 300 seconds")
    if not _SHA256_PATTERN.fullmatch(operation_plan_sha256):
        raise LabError("maintenance operation plan has an invalid hash")
    if not re.fullmatch(r"[0-9a-f]{40}", local_commit) or not re.fullmatch(
        r"[0-9a-f]{40}", local_tree_oid
    ):
        raise LabError("maintenance operation has an invalid checkout identity")
    if (
        git.current_commit(target.checkout_path) != local_commit
        or git.tree_oid(target.checkout_path) != local_tree_oid
        or git.status_short(target.checkout_path) != expected_status
    ):
        raise LabError("checkout changed before maintenance reservation")
    with _locked_state(target) as state_path:
        existing = _load_state(state_path)
        if existing is not None:
            prior = validate_maintenance_reservation(
                existing,
                target=target,
                require_unexpired=False,
            )
            if (
                prior.get("status") == "ACTIVE"
                and datetime.now(timezone.utc)
                <= _timestamp(prior.get("expires_at"), label="expires_at")
            ):
                raise LabError("maintenance reservation is already active")
        acquired = datetime.now(timezone.utc)
        content: dict[str, object] = {
            "schema": MAINTENANCE_RESERVATION_SCHEMA_ID,
            "target": target.identity(),
            "policy_sha256": target.policy_sha256,
            "plan_sha256": operation_plan_sha256,
            "local_commit": local_commit,
            "local_tree_oid": local_tree_oid,
            "checkout_status": list(expected_status),
            "reservation_id": reservation_id,
            "status": "ACTIVE",
            "managed_runtime_starts_blocked": True,
            "unmanaged_runtime_authorized": False,
            "device_access": False,
            "acquired_at": acquired.isoformat(),
            "expires_at": (acquired + timedelta(seconds=validity_seconds)).isoformat(),
            "released_at": None,
        }
        receipt = {**content, "receipt_sha256": _hash(content)}
        _atomic_write_text(state_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt


def assert_managed_runtime_start_allowed(
    target: StableCheckoutTarget,
) -> dict[str, object]:
    """Fail closed for cooperating launchers during an active maintenance window."""

    with _locked_state(target) as state_path:
        state = _load_state(state_path)
        active_reservation: str | None = None
        if state is not None:
            validated = validate_maintenance_reservation(
                state,
                target=target,
                require_unexpired=False,
            )
            expires = _timestamp(validated.get("expires_at"), label="expires_at")
            if validated.get("status") == "ACTIVE" and datetime.now(timezone.utc) <= expires:
                active_reservation = str(validated["reservation_id"])
        if active_reservation is not None:
            raise LabError(
                "managed runtime start blocked by maintenance reservation: "
                f"{active_reservation}"
            )
        content: dict[str, object] = {
            "schema": MANAGED_RUNTIME_GATE_SCHEMA_ID,
            "target": target.identity(),
            "policy_sha256": target.policy_sha256,
            "allowed": True,
            "active_maintenance_reservation": None,
            "device_access": False,
        }
        return {**content, "receipt_sha256": _hash(content)}


def release_maintenance_reservation(
    target: StableCheckoutTarget,
    *,
    active_receipt: object,
) -> dict[str, object]:
    active = validate_maintenance_reservation(
        active_receipt,
        target=target,
        require_unexpired=False,
    )
    if active.get("status") != "ACTIVE":
        raise LabError("maintenance reservation was already released")
    with _locked_state(target) as state_path:
        state = _load_state(state_path)
        if state != active:
            raise LabError("maintenance reservation state changed before release")
        content = {
            key: value
            for key, value in active.items()
            if key
            not in {
                "status",
                "managed_runtime_starts_blocked",
                "released_at",
                "receipt_sha256",
            }
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
            "receipt_sha256": _hash(content),
        }
        _atomic_write_text(
            state_path,
            json.dumps(released, indent=2, sort_keys=True) + "\n",
        )
        return validate_maintenance_reservation(
            released,
            target=target,
            require_unexpired=False,
        )
