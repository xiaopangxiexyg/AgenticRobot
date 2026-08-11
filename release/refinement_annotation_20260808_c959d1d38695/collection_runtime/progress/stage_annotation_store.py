"""Crash-safe persistence for post-capture ordered-stage annotations.

The mutable draft is deliberately not a dataset contract.  Only a validated
``StageAnnotationBundleV1`` is published as an immutable annotation set.
Editable ownership is Episode-scoped.  The authoritative cross-machine lock is
an atomically-created directory so separate SSHFS/NFS clients do not depend on
the local kernel's advisory-lock implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from typing import Any, Mapping
import uuid

from dataset_contracts.progress_facts import ProgressEpisodeManifestV1
from dataset_contracts.retrospective_progress import (
    RetrospectiveProgressMigrationManifestV1,
)
from dataset_contracts.progress_stage_annotations import (
    STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1,
    STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1,
    STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
    StageAnnotationBundleV1,
    StageAnnotationManifestV1,
    StageAnnotationSetBindingV1,
)
from dataset_contracts.progress_stages import TaskStageProfileV1
from dataset_interface.progress_stage_annotation_artifacts import (
    LoadedStageAnnotationSet,
    STAGE_ANNOTATION_RELATIVE_ROOT,
    StageAnnotationArtifactReadError,
    load_stage_annotation_set as _load_stage_annotation_set,
)

from ..immutable_canonical_store import (
    fsync_directory,
    publish_immutable_canonical,
)


STAGE_ANNOTATION_DRAFT_RELATIVE_ROOT = Path(
    "meta/progress/stage_annotation_drafts"
)
STAGE_ANNOTATION_WRITER_LOCK_RELATIVE_PATH = (
    STAGE_ANNOTATION_RELATIVE_ROOT / ".writer.lock"
)
STAGE_ANNOTATION_EPISODE_LOCK_RELATIVE_ROOT = Path(
    "meta/progress/stage_annotation_locks"
)
_EPISODE_LOCK_OWNER_FILE = "owner.json"
_EPISODE_LOCK_SCHEMA = "dexteleop.stage_annotation_episode_lock.local.v1"
_UNSUPPORTED_FLOCK_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "ENOSYS", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    (*_UNSUPPORTED_FLOCK_ERRNOS, errno.EINVAL)
)


class StageAnnotationStoreError(RuntimeError):
    """Raised when an annotation set is incomplete or contradictory."""


@dataclass(frozen=True)
class StageAnnotationCommitReport:
    annotation_dir: Path
    manifest: StageAnnotationManifestV1
    idempotent: bool


@dataclass(frozen=True)
class _HeldEpisodeLock:
    episode_uuid: str
    lock_id: str
    lock_dir: Path
    owner: Mapping[str, Any]


def _fsync_lock_directory(path: Path) -> None:
    """Best-effort directory durability for mounts that support it.

    Atomic mkdir/rename is the ownership primitive. Some SSHFS servers reject
    directory fsync even though those operations are server-side atomic.
    """

    try:
        fsync_directory(path)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            raise


def _validate_set(
    *,
    source_progress_manifest: (
        ProgressEpisodeManifestV1
        | RetrospectiveProgressMigrationManifestV1
    ),
    stage_profile: TaskStageProfileV1,
    binding: StageAnnotationSetBindingV1,
    bundle: StageAnnotationBundleV1,
) -> None:
    if (
        binding.episode_capture_identity_hash
        != source_progress_manifest.episode_capture_identity_hash
    ):
        raise StageAnnotationStoreError(
            "Annotation binding and progress manifest Episode identity differ."
        )
    if (
        binding.source_progress_manifest_hash
        != source_progress_manifest.semantic_content_hash
    ):
        raise StageAnnotationStoreError(
            "Annotation binding does not reference the exact sealed progress "
            "manifest."
        )
    if (
        binding.task_stage_profile_hash
        != stage_profile.semantic_content_hash
    ):
        raise StageAnnotationStoreError(
            "Annotation binding and task stage profile differ."
        )
    clocks = tuple(
        item.clock_domain_id
        for item in (
            bundle.occurrences
            + bundle.abnormal_intervals
            + bundle.progress_boundary_overrides
        )
    )
    if any(clock != binding.bound_clock_domain_id for clock in clocks):
        raise StageAnnotationStoreError(
            "Every timeline annotation must use the bound Episode clock."
        )


def commit_stage_annotation_set(
    *,
    dataset_root: str | Path,
    episode_uuid: str,
    source_progress_manifest: (
        ProgressEpisodeManifestV1
        | RetrospectiveProgressMigrationManifestV1
    ),
    stage_profile: TaskStageProfileV1,
    binding: StageAnnotationSetBindingV1,
    bundle: StageAnnotationBundleV1,
) -> StageAnnotationCommitReport:
    """Publish one immutable set; the manifest is always written last."""

    normalized_episode_uuid = str(episode_uuid).strip()
    if (
        not normalized_episode_uuid
        or normalized_episode_uuid in {".", ".."}
        or Path(normalized_episode_uuid).name != normalized_episode_uuid
    ):
        raise ValueError("episode_uuid must be a path-safe string.")
    if binding.annotation_set_id.strip() in {".", ".."}:
        raise ValueError("annotation_set_id is not path-safe.")
    _validate_set(
        source_progress_manifest=source_progress_manifest,
        stage_profile=stage_profile,
        binding=binding,
        bundle=bundle,
    )
    root = Path(dataset_root).expanduser().resolve()
    annotation_set_id = binding.annotation_set_id.strip()
    if Path(annotation_set_id).name != annotation_set_id:
        raise ValueError("annotation_set_id is not path-safe.")
    annotation_dir = (
        root
        / STAGE_ANNOTATION_RELATIVE_ROOT
        / normalized_episode_uuid
        / annotation_set_id
    )
    expected_parent = (
        root / STAGE_ANNOTATION_RELATIVE_ROOT / normalized_episode_uuid
    )
    if annotation_dir.parent != expected_parent:
        raise ValueError("Stage annotation path escaped its owned root.")
    manifest = StageAnnotationManifestV1(
        annotation_set_binding_hash=binding.semantic_content_hash,
        annotation_bundle_hash=bundle.semantic_content_hash,
    )
    try:
        binding_idempotent = publish_immutable_canonical(
            annotation_dir / "binding.cjson",
            binding.to_payload(),
            domain=STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1,
            sync_parent=False,
        )
        bundle_idempotent = publish_immutable_canonical(
            annotation_dir / "annotations.cjson",
            bundle.to_payload(),
            domain=STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1,
            sync_parent=False,
        )
        # The manifest is the visibility boundary.  Make both referenced files
        # durable with one directory sync before publishing that final marker.
        fsync_directory(annotation_dir)
        manifest_idempotent = publish_immutable_canonical(
            annotation_dir / "manifest.cjson",
            manifest.to_payload(),
            domain=STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise StageAnnotationStoreError(
            f"Cannot publish immutable stage annotation set: {exc}"
        ) from exc
    return StageAnnotationCommitReport(
        annotation_dir=annotation_dir,
        manifest=manifest,
        idempotent=(
            binding_idempotent
            and bundle_idempotent
            and manifest_idempotent
        ),
    )


def load_stage_annotation_set(
    annotation_dir: str | Path,
) -> LoadedStageAnnotationSet:
    """Compatibility reader for the dataset-interface owned implementation."""

    try:
        return _load_stage_annotation_set(annotation_dir)
    except StageAnnotationArtifactReadError as exc:
        raise StageAnnotationStoreError(
            str(exc)
        ) from exc


class DatasetStageAnnotationWorkspace:
    """Coordinate Episode-scoped editors while dataset capture continues.

    ``acquire()`` retains a shared advisory lock only as a compatibility guard
    against an older annotation program that still requests the historical
    dataset-wide exclusive lock.  Correct cross-machine exclusion comes from
    ``acquire_episode()`` and its atomic lock directory.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        annotator_id: str = "",
        blocking: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.annotator_id = str(annotator_id).strip()
        self.blocking = bool(blocking)
        self._lock_file = None
        self._workspace_open = False
        self._mutex = threading.RLock()
        self._episode_locks: dict[str, _HeldEpisodeLock] = {}

    def acquire(self) -> None:
        if self._workspace_open:
            return
        lock_path = (
            self.dataset_root
            / STAGE_ANNOTATION_WRITER_LOCK_RELATIVE_PATH
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+b")
        # New clients share this legacy guard and then arbitrate individual
        # Episodes with atomic directories.  An old client still requesting
        # LOCK_EX cannot coexist on filesystems that propagate flock.
        operation = fcntl.LOCK_SH
        if not self.blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(lock_file.fileno(), operation)
        except BlockingIOError as exc:
            lock_file.close()
            raise StageAnnotationStoreError(
                "A legacy stage annotation program exclusively owns this "
                "dataset. Close or upgrade it before using Episode-scoped "
                "annotation."
            ) from exc
        except OSError as exc:
            if exc.errno not in _UNSUPPORTED_FLOCK_ERRNOS:
                lock_file.close()
                raise StageAnnotationStoreError(
                    f"Cannot acquire the legacy annotation compatibility "
                    f"guard: {exc}"
                ) from exc
            # Some SSHFS configurations do not implement flock.  The atomic
            # Episode directory remains authoritative; mixed old/new clients
            # are not safe on such a mount and are rejected by release policy.
            lock_file.close()
            lock_file = None
        self._lock_file = lock_file
        self._workspace_open = True

    @staticmethod
    def _normalized_episode_uuid(episode_uuid: str) -> str:
        normalized = str(episode_uuid).strip()
        if not normalized or normalized in {".", ".."}:
            raise ValueError("episode_uuid must be a path-safe string.")
        if Path(normalized).name != normalized:
            raise ValueError("episode_uuid must be a path-safe string.")
        return normalized

    def episode_lock_path(self, *, episode_uuid: str) -> Path:
        normalized = self._normalized_episode_uuid(episode_uuid)
        root = self.dataset_root / STAGE_ANNOTATION_EPISODE_LOCK_RELATIVE_ROOT
        path = root / f"{normalized}.lock"
        if path.parent != root:
            raise ValueError("Stage annotation Episode lock escaped its root.")
        return path

    @staticmethod
    def _read_lock_owner(lock_dir: Path) -> dict[str, Any] | None:
        path = lock_dir / _EPISODE_LOCK_OWNER_FILE
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeError, json.JSONDecodeError, OSError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _owner_description(owner: Mapping[str, Any] | None) -> str:
        if owner is None:
            return "owner metadata unavailable"
        values = []
        for key in ("annotator_id", "hostname", "pid", "acquired_at"):
            value = owner.get(key)
            if value not in (None, ""):
                values.append(f"{key}={value}")
        return ", ".join(values) or "owner metadata unavailable"

    @staticmethod
    def _write_lock_owner(lock_dir: Path, owner: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            dict(owner),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fd, name = tempfile.mkstemp(
            prefix=".owner.",
            suffix=".tmp",
            dir=str(lock_dir),
        )
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, lock_dir / _EPISODE_LOCK_OWNER_FILE)
            _fsync_lock_directory(lock_dir)
        finally:
            temporary.unlink(missing_ok=True)

    def acquire_episode(
        self,
        *,
        episode_uuid: str,
        annotator_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Acquire one fail-closed cross-machine Episode editing lock."""

        normalized = self._normalized_episode_uuid(episode_uuid)
        self._require_workspace()
        with self._mutex:
            existing = self._episode_locks.get(normalized)
            if existing is not None:
                self._require_episode_lock(normalized)
                return existing.owner

            lock_dir = self.episode_lock_path(episode_uuid=normalized)
            lock_dir.parent.mkdir(parents=True, exist_ok=True)
            lock_id = uuid.uuid4().hex
            owner = {
                "schema": _EPISODE_LOCK_SCHEMA,
                "episode_uuid": normalized,
                "lock_id": lock_id,
                "annotator_id": str(
                    self.annotator_id
                    if annotator_id is None
                    else annotator_id
                ).strip(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "acquired_at": time.time(),
            }
            try:
                lock_dir.mkdir()
            except FileExistsError as exc:
                observed = self._read_lock_owner(lock_dir)
                raise StageAnnotationStoreError(
                    f"Episode {normalized!r} is already being annotated by "
                    f"another program ({self._owner_description(observed)}). "
                    "The lock is never stolen automatically; verify the "
                    "recorded owner before removing a stale lock."
                ) from exc
            except OSError as exc:
                raise StageAnnotationStoreError(
                    f"Cannot acquire Episode annotation lock {lock_dir}: {exc}"
                ) from exc

            try:
                self._write_lock_owner(lock_dir, owner)
                _fsync_lock_directory(lock_dir.parent)
            except OSError as exc:
                # The mkdir succeeded, so only this process can own this new
                # directory.  Remove only the files created by this attempt.
                (lock_dir / _EPISODE_LOCK_OWNER_FILE).unlink(missing_ok=True)
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
                raise StageAnnotationStoreError(
                    f"Cannot publish Episode annotation lock owner: {exc}"
                ) from exc

            held = _HeldEpisodeLock(
                episode_uuid=normalized,
                lock_id=lock_id,
                lock_dir=lock_dir,
                owner=owner,
            )
            self._episode_locks[normalized] = held
            return owner

    def release_episode(self, *, episode_uuid: str) -> None:
        normalized = self._normalized_episode_uuid(episode_uuid)
        with self._mutex:
            held = self._episode_locks.get(normalized)
            if held is None:
                return
            self._require_episode_lock(normalized)
            releasing = held.lock_dir.with_name(
                f".{held.lock_dir.name}.releasing-{held.lock_id}"
            )
            try:
                os.replace(held.lock_dir, releasing)
            except OSError as exc:
                raise StageAnnotationStoreError(
                    f"Cannot release Episode annotation lock "
                    f"{held.lock_dir}: {exc}"
                ) from exc
            self._episode_locks.pop(normalized, None)
            try:
                (releasing / _EPISODE_LOCK_OWNER_FILE).unlink(missing_ok=True)
                releasing.rmdir()
                _fsync_lock_directory(releasing.parent)
            except OSError:
                # The canonical lock path was already atomically released.
                # A uniquely named tombstone is non-authoritative and can be
                # cleaned during maintenance without blocking another editor.
                pass

    def require_episode(self, *, episode_uuid: str) -> None:
        normalized = self._normalized_episode_uuid(episode_uuid)
        with self._mutex:
            self._require_episode_lock(normalized)

    def close(self) -> None:
        with self._mutex:
            episodes = tuple(self._episode_locks)
        for episode_uuid in episodes:
            try:
                self.release_episode(episode_uuid=episode_uuid)
            except StageAnnotationStoreError:
                # A failed remote release remains fail-closed at its canonical
                # path.  Closing the process must still release local handles.
                pass
        lock_file = self._lock_file
        self._lock_file = None
        self._workspace_open = False
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def draft_path(self, *, episode_uuid: str) -> Path:
        normalized = self._normalized_episode_uuid(episode_uuid)
        path = (
            self.dataset_root
            / STAGE_ANNOTATION_DRAFT_RELATIVE_ROOT
            / normalized
            / "draft.json"
        )
        expected_parent = (
            self.dataset_root
            / STAGE_ANNOTATION_DRAFT_RELATIVE_ROOT
            / normalized
        )
        if path.parent != expected_parent:
            raise ValueError("Stage annotation draft path escaped its root.")
        return path

    def load_draft(self, *, episode_uuid: str) -> dict[str, Any] | None:
        self.require_episode(episode_uuid=episode_uuid)
        path = self.draft_path(episode_uuid=episode_uuid)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StageAnnotationStoreError(
                f"Cannot load stage annotation draft {path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise StageAnnotationStoreError(
                "Stage annotation draft root must be a JSON object."
            )
        return value

    def save_draft(
        self,
        *,
        episode_uuid: str,
        draft: Mapping[str, Any],
        durable: bool = True,
    ) -> Path:
        self.require_episode(episode_uuid=episode_uuid)
        if not isinstance(draft, Mapping):
            raise TypeError("draft must be a mapping.")
        path = self.draft_path(episode_uuid=episode_uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            dict(draft),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fd, name = tempfile.mkstemp(
            prefix=".draft.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                if durable:
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(temporary, path)
            if durable:
                fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _require_workspace(self) -> None:
        if not self._workspace_open:
            raise StageAnnotationStoreError(
                "Stage annotation workspace is not open."
            )

    def _require_episode_lock(self, episode_uuid: str) -> None:
        held = self._episode_locks.get(episode_uuid)
        if held is None:
            raise StageAnnotationStoreError(
                f"Episode annotation lock is not held for {episode_uuid!r}."
            )
        observed = self._read_lock_owner(held.lock_dir)
        if (
            observed is None
            or observed.get("schema") != _EPISODE_LOCK_SCHEMA
            or observed.get("episode_uuid") != episode_uuid
            or observed.get("lock_id") != held.lock_id
        ):
            raise StageAnnotationStoreError(
                f"Episode annotation lock ownership changed or became "
                f"unreadable for {episode_uuid!r}; refusing to write."
            )

    def __enter__(self) -> "DatasetStageAnnotationWorkspace":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "STAGE_ANNOTATION_DRAFT_RELATIVE_ROOT",
    "STAGE_ANNOTATION_EPISODE_LOCK_RELATIVE_ROOT",
    "STAGE_ANNOTATION_RELATIVE_ROOT",
    "STAGE_ANNOTATION_WRITER_LOCK_RELATIVE_PATH",
    "DatasetStageAnnotationWorkspace",
    "LoadedStageAnnotationSet",
    "StageAnnotationCommitReport",
    "StageAnnotationStoreError",
    "commit_stage_annotation_set",
    "load_stage_annotation_set",
]
