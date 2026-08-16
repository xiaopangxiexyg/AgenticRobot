from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .workspace import (
    LabError,
    WorkspaceManager,
    validate_branch,
    validate_repository_name,
    validate_task_id,
)

WRITER_LEASE_SCHEMA_ID = "lab.writer-lease.v1"
WRITER_LEASE_RELEASE_SCHEMA_ID = "lab.writer-lease-release.v1"
WRITER_LEASE_STATUS_SCHEMA_ID = "lab.writer-lease-status.v1"
MIN_TTL_SECONDS = 60
MAX_TTL_SECONDS = 24 * 60 * 60


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _hash_document(content: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def _timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise LabError(f"writer lease has invalid {field}")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError(f"writer lease has invalid {field}") from exc
    if timestamp.tzinfo is None:
        raise LabError(f"writer lease {field} requires a timezone")
    return timestamp.astimezone(timezone.utc)


def _normalized_now(clock: Callable[[], datetime]) -> datetime:
    now = clock()
    if now.tzinfo is None:
        raise LabError("writer lease clock requires a timezone")
    return now.astimezone(timezone.utc)


def _validate_ttl(ttl_seconds: int) -> int:
    if (
        not isinstance(ttl_seconds, int)
        or isinstance(ttl_seconds, bool)
        or not MIN_TTL_SECONDS <= ttl_seconds <= MAX_TTL_SECONDS
    ):
        raise LabError(
            f"writer lease TTL must be between {MIN_TTL_SECONDS} and "
            f"{MAX_TTL_SECONDS} seconds"
        )
    return ttl_seconds


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class WriterLeaseManager:
    """Coordinate one active actor per Task repository without touching Git."""

    def __init__(
        self,
        workspace: WorkspaceManager,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.workspace = workspace
        self.clock = clock

    def _canonical_repository(self, value: str) -> str:
        validate_repository_name(value)
        repository = self.workspace.repository_aliases.get(value.casefold(), value)
        if repository not in self.workspace.catalog:
            available = ", ".join(sorted(self.workspace.catalog))
            raise LabError(
                f"unknown writer repository {value!r}; available: {available}"
            )
        return repository

    def _authority(
        self,
        task_id: str,
        repository: str,
        *,
        manifest: dict[str, object] | None = None,
    ) -> dict[str, object]:
        task_id = validate_task_id(task_id)
        repository = self._canonical_repository(repository)
        task = manifest if manifest is not None else self.workspace.load_task(task_id)
        if task.get("task_id") != task_id or task.get("status") != "ACTIVE":
            raise LabError(f"writer lease requires an ACTIVE Task: {task_id}")
        repositories = task.get("repositories")
        raw_repository = (
            repositories.get(repository) if isinstance(repositories, dict) else None
        )
        if not isinstance(raw_repository, dict):
            raise LabError(f"Task does not include repository: {task_id}/{repository}")
        writer = raw_repository.get("writer")
        if (
            not isinstance(writer, dict)
            or set(writer) != {"host", "epoch"}
            or not isinstance(writer.get("host"), str)
            or not isinstance(writer.get("epoch"), int)
            or isinstance(writer.get("epoch"), bool)
            or int(writer["epoch"]) < 1
        ):
            raise LabError(f"Task has invalid writer authority: {task_id}/{repository}")
        branch = validate_branch(str(raw_repository.get("branch", "")))
        worktree = Path(str(raw_repository.get("worktree_path", ""))).resolve()
        manifest_path = self.workspace.task_manifest_path(task_id)
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return {
            "task_id": task_id,
            "repository": repository,
            "host": writer["host"],
            "writer_epoch": writer["epoch"],
            "branch": branch,
            "worktree_path": str(worktree),
            "task_manifest_sha256": manifest_hash,
        }

    def _paths(
        self,
        task_id: str,
        repository: str,
        *,
        create: bool,
    ) -> tuple[Path, Path]:
        root = self.workspace.state_root
        if root.is_symlink():
            raise LabError(f"writer lease state root is a symlink: {root}")
        lease_root = root / "writer-leases"
        task_root = lease_root / task_id
        repository_root = task_root / repository
        for path in (root, lease_root, task_root, repository_root):
            if path.is_symlink():
                raise LabError(f"writer lease state path is a symlink: {path}")
        if create:
            root.mkdir(parents=True, exist_ok=True)
            lease_root.mkdir(exist_ok=True)
            task_root.mkdir(exist_ok=True)
            repository_root.mkdir(exist_ok=True)
        if not repository_root.resolve().is_relative_to(root.resolve()):
            raise LabError("writer lease state escapes configured root")
        events = repository_root / "events"
        if events.is_symlink():
            raise LabError(f"writer lease event path is a symlink: {events}")
        if create:
            events.mkdir(exist_ok=True)
        return repository_root / "active.json", events

    @staticmethod
    def _lock_key(task_id: str, repository: str) -> str:
        return f"{task_id}:{repository}"

    def _validate_lease(self, payload: object) -> dict[str, object]:
        expected_fields = {
            "schema",
            "lease_id",
            "task_id",
            "repository",
            "actor",
            "host",
            "writer_epoch",
            "branch",
            "worktree_path",
            "task_manifest_sha256",
            "claimed_at",
            "renewed_at",
            "expires_at",
            "revision",
            "replaces_lease_id",
            "status",
            "receipt_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise LabError("writer lease has an unexpected shape")
        if payload.get("schema") != WRITER_LEASE_SCHEMA_ID:
            raise LabError("unsupported writer lease schema")
        for field in ("lease_id", "task_id", "actor", "host"):
            validate_task_id(str(payload.get(field, "")))
        validate_repository_name(str(payload.get("repository", "")))
        validate_branch(str(payload.get("branch", "")))
        if (
            not isinstance(payload.get("worktree_path"), str)
            or not Path(str(payload["worktree_path"])).is_absolute()
            or not isinstance(payload.get("writer_epoch"), int)
            or isinstance(payload.get("writer_epoch"), bool)
            or int(payload["writer_epoch"]) < 1
            or not isinstance(payload.get("revision"), int)
            or isinstance(payload.get("revision"), bool)
            or int(payload["revision"]) < 1
            or payload.get("status") != "ACTIVE"
        ):
            raise LabError("writer lease has invalid authority/state")
        for field in ("task_manifest_sha256", "receipt_sha256"):
            value = payload.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise LabError(f"writer lease has invalid {field}")
        replacement = payload.get("replaces_lease_id")
        if replacement is not None:
            validate_task_id(str(replacement))
        claimed = _timestamp(payload.get("claimed_at"), field="claimed_at")
        renewed = _timestamp(payload.get("renewed_at"), field="renewed_at")
        expires = _timestamp(payload.get("expires_at"), field="expires_at")
        if not claimed <= renewed < expires:
            raise LabError("writer lease timestamps are inconsistent")
        content = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        if payload["receipt_sha256"] != _hash_document(content):
            raise LabError("writer lease receipt hash mismatch")
        return payload

    def _load_active(self, active_path: Path) -> dict[str, object] | None:
        if active_path.is_symlink():
            raise LabError(f"writer lease active path is a symlink: {active_path}")
        if not active_path.exists():
            return None
        if not active_path.is_file():
            raise LabError(f"writer lease active path is not a file: {active_path}")
        try:
            payload = json.loads(active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError(f"cannot read writer lease: {active_path}: {exc}") from exc
        return self._validate_lease(payload)

    @staticmethod
    def _authority_matches(
        lease: Mapping[str, object],
        authority: Mapping[str, object],
    ) -> bool:
        return all(
            lease.get(field) == authority.get(field)
            for field in (
                "task_id",
                "repository",
                "host",
                "writer_epoch",
                "branch",
                "worktree_path",
                "task_manifest_sha256",
            )
        )

    def inspect_repository(
        self,
        task_id: str,
        repository: str,
        *,
        manifest: dict[str, object] | None = None,
    ) -> dict[str, object]:
        authority = self._authority(task_id, repository, manifest=manifest)
        repository = str(authority["repository"])
        active_path, _ = self._paths(task_id, repository, create=False)
        now = _normalized_now(self.clock)
        lease = self._load_active(active_path)
        if lease is None:
            state = "AVAILABLE"
        elif not self._authority_matches(lease, authority):
            state = "STALE_AUTHORITY"
        elif _timestamp(lease["expires_at"], field="expires_at") <= now:
            state = "EXPIRED"
        else:
            state = "ACTIVE"
        return {
            "schema": WRITER_LEASE_STATUS_SCHEMA_ID,
            "observed_at": now.isoformat(),
            "state": state,
            "authority": authority,
            "lease": lease,
            "claim_allowed": state == "AVAILABLE",
            "replace_expired_required": state in {"EXPIRED", "STALE_AUTHORITY"},
        }

    def inspect_task(self, task_id: str) -> dict[str, object]:
        task_id = validate_task_id(task_id)
        manifest = self.workspace.load_task(task_id)
        repositories = manifest.get("repositories")
        if not isinstance(repositories, dict):
            raise LabError(f"Task manifest has no repositories: {task_id}")
        statuses = {
            name: self.inspect_repository(task_id, name, manifest=manifest)
            for name in sorted(repositories)
        }
        return {
            "schema": "lab.writer-lease-task-status.v1",
            "task_id": task_id,
            "repositories": statuses,
            "status": (
                "ATTENTION"
                if any(
                    status["state"] in {"EXPIRED", "STALE_AUTHORITY"}
                    for status in statuses.values()
                )
                else "READY"
            ),
        }

    def claim(
        self,
        task_id: str,
        repository: str,
        *,
        actor: str,
        ttl_seconds: int = 3600,
        replace_expired: str | None = None,
    ) -> dict[str, object]:
        actor = validate_task_id(actor)
        ttl_seconds = _validate_ttl(ttl_seconds)
        authority = self._authority(task_id, repository)
        repository = str(authority["repository"])
        active_path, events = self._paths(task_id, repository, create=True)
        now = _normalized_now(self.clock)
        lock_key = self._lock_key(task_id, repository)
        with self.workspace.advisory_locks("writer-leases", [lock_key]):
            existing = self._load_active(active_path)
            if existing is not None:
                expired = (
                    _timestamp(existing["expires_at"], field="expires_at") <= now
                    or not self._authority_matches(existing, authority)
                )
                if not expired:
                    if existing["actor"] == actor:
                        return existing
                    raise LabError(
                        f"writer lease is held by {existing['actor']}: "
                        f"{task_id}/{repository}"
                    )
                if replace_expired != existing["lease_id"]:
                    raise LabError(
                        "expired/stale writer lease requires exact --replace-expired "
                        f"{existing['lease_id']}"
                    )
            elif replace_expired is not None:
                raise LabError("--replace-expired supplied but no writer lease exists")
            lease_id = validate_task_id(f"WL-{secrets.token_hex(12)}")
            content: dict[str, object] = {
                "schema": WRITER_LEASE_SCHEMA_ID,
                "lease_id": lease_id,
                **authority,
                "actor": actor,
                "claimed_at": now.isoformat(),
                "renewed_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
                "revision": 1,
                "replaces_lease_id": (
                    existing["lease_id"] if existing is not None else None
                ),
                "status": "ACTIVE",
            }
            lease = {**content, "receipt_sha256": _hash_document(content)}
            event_path = events / f"{lease_id}-claim.json"
            if event_path.exists() or event_path.is_symlink():
                raise LabError(f"writer lease claim event collision: {event_path}")
            _atomic_write(event_path, lease)
            _atomic_write(active_path, lease)
            return lease

    def renew(
        self,
        task_id: str,
        repository: str,
        *,
        actor: str,
        lease_id: str,
        ttl_seconds: int = 3600,
    ) -> dict[str, object]:
        actor = validate_task_id(actor)
        lease_id = validate_task_id(lease_id)
        ttl_seconds = _validate_ttl(ttl_seconds)
        authority = self._authority(task_id, repository)
        repository = str(authority["repository"])
        active_path, events = self._paths(task_id, repository, create=True)
        now = _normalized_now(self.clock)
        lock_key = self._lock_key(task_id, repository)
        with self.workspace.advisory_locks("writer-leases", [lock_key]):
            existing = self._load_active(active_path)
            if existing is None:
                raise LabError(f"writer lease is not active: {task_id}/{repository}")
            if existing["lease_id"] != lease_id or existing["actor"] != actor:
                raise LabError("writer lease renew identity mismatch")
            if not self._authority_matches(existing, authority):
                raise LabError("writer lease authority changed; explicit replacement required")
            if _timestamp(existing["expires_at"], field="expires_at") <= now:
                raise LabError("expired writer lease cannot be renewed")
            existing_revision = existing["revision"]
            if not isinstance(existing_revision, int) or isinstance(
                existing_revision, bool
            ):
                raise LabError("writer lease has invalid revision")
            revision = existing_revision + 1
            content = {
                **{key: value for key, value in existing.items() if key != "receipt_sha256"},
                "renewed_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
                "revision": revision,
            }
            renewed = {**content, "receipt_sha256": _hash_document(content)}
            event_path = events / f"{lease_id}-renew-{revision}.json"
            if event_path.exists() or event_path.is_symlink():
                raise LabError(f"writer lease renew event collision: {event_path}")
            _atomic_write(event_path, renewed)
            _atomic_write(active_path, renewed)
            return renewed

    def release(
        self,
        task_id: str,
        repository: str,
        *,
        actor: str,
        lease_id: str,
    ) -> dict[str, object]:
        actor = validate_task_id(actor)
        lease_id = validate_task_id(lease_id)
        authority = self._authority(task_id, repository)
        repository = str(authority["repository"])
        active_path, events = self._paths(task_id, repository, create=True)
        release_path = events / f"{lease_id}-release.json"
        now = _normalized_now(self.clock)
        lock_key = self._lock_key(task_id, repository)
        with self.workspace.advisory_locks("writer-leases", [lock_key]):
            if release_path.is_symlink():
                raise LabError(f"writer lease release path is a symlink: {release_path}")
            if release_path.is_file() and not active_path.exists():
                try:
                    existing_release = json.loads(
                        release_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise LabError(f"cannot read writer release receipt: {exc}") from exc
                return self._validate_release(existing_release)
            existing = self._load_active(active_path)
            if existing is None:
                raise LabError(f"writer lease is not active: {task_id}/{repository}")
            if existing["lease_id"] != lease_id or existing["actor"] != actor:
                raise LabError("writer lease release identity mismatch")
            content: dict[str, object] = {
                "schema": WRITER_LEASE_RELEASE_SCHEMA_ID,
                "lease_id": lease_id,
                **authority,
                "actor": actor,
                "released_at": now.isoformat(),
                "released_revision": existing["revision"],
                "active_receipt_sha256": existing["receipt_sha256"],
                "status": "RELEASED",
            }
            released = {**content, "receipt_sha256": _hash_document(content)}
            if release_path.exists():
                raise LabError(f"writer lease release event collision: {release_path}")
            _atomic_write(release_path, released)
            active_path.unlink()
            return released

    def _validate_release(self, payload: object) -> dict[str, object]:
        expected_fields = {
            "schema",
            "lease_id",
            "task_id",
            "repository",
            "actor",
            "host",
            "writer_epoch",
            "branch",
            "worktree_path",
            "task_manifest_sha256",
            "released_at",
            "released_revision",
            "active_receipt_sha256",
            "status",
            "receipt_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise LabError("writer release receipt has an unexpected shape")
        if (
            payload.get("schema") != WRITER_LEASE_RELEASE_SCHEMA_ID
            or payload.get("status") != "RELEASED"
        ):
            raise LabError("writer release receipt has invalid identity/state")
        for field in ("lease_id", "task_id", "actor", "host"):
            validate_task_id(str(payload.get(field, "")))
        validate_repository_name(str(payload.get("repository", "")))
        validate_branch(str(payload.get("branch", "")))
        _timestamp(payload.get("released_at"), field="released_at")
        for field in (
            "task_manifest_sha256",
            "active_receipt_sha256",
            "receipt_sha256",
        ):
            value = payload.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise LabError(f"writer release receipt has invalid {field}")
        if (
            not isinstance(payload.get("writer_epoch"), int)
            or isinstance(payload.get("writer_epoch"), bool)
            or int(payload["writer_epoch"]) < 1
            or not isinstance(payload.get("released_revision"), int)
            or isinstance(payload.get("released_revision"), bool)
            or int(payload["released_revision"]) < 1
            or not isinstance(payload.get("worktree_path"), str)
            or not Path(str(payload["worktree_path"])).is_absolute()
        ):
            raise LabError("writer release receipt has invalid authority")
        content = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        if payload["receipt_sha256"] != _hash_document(content):
            raise LabError("writer release receipt hash mismatch")
        return payload


def writer_lease_fingerprint(task_status: Mapping[str, object]) -> dict[str, object]:
    repositories = task_status.get("repositories")
    if not isinstance(repositories, dict):
        raise LabError("writer lease Task status lacks repositories")
    fingerprint: dict[str, object] = {}
    for name, raw_status in repositories.items():
        if not isinstance(name, str) or not isinstance(raw_status, dict):
            raise LabError("writer lease Task status has invalid repository")
        authority = raw_status.get("authority")
        lease = raw_status.get("lease")
        if not isinstance(authority, dict):
            raise LabError("writer lease status lacks authority")
        fingerprint[name] = {
            "state": raw_status.get("state"),
            "task_manifest_sha256": authority.get("task_manifest_sha256"),
            "host": authority.get("host"),
            "writer_epoch": authority.get("writer_epoch"),
            "actor": lease.get("actor") if isinstance(lease, dict) else None,
            "lease_id": lease.get("lease_id") if isinstance(lease, dict) else None,
            "lease_receipt_sha256": (
                lease.get("receipt_sha256") if isinstance(lease, dict) else None
            ),
        }
    return fingerprint
