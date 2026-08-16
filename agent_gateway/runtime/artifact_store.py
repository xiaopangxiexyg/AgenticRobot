"""Recoverable, quota-bounded transient Artifact store for Skill protocol v2.

The store owns only temporary Artifact availability and handoff.  It never
becomes a dataset, Experience, Memory, device, or Skill executor owner.  All
caller-visible identities are immutable :class:`AgentArtifactRef` values;
filesystem paths remain managed implementation details until a transport layer
issues a bounded access grant.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from agent_gateway.protocol._contract_validation import (
    finite_number,
    identifier,
    positive_integer,
)
from agent_gateway.protocol.artifact_contracts import (
    AgentArtifactMaterializationReceipt,
    AgentArtifactRef,
)

ARTIFACT_STORE_SCHEMA_VERSION_V1 = 1
ARTIFACT_STORE_STATES_V1 = (
    "staging",
    "verifying",
    "available",
    "quarantined",
    "aborted",
    "expired",
)
ARTIFACT_STORE_PIN_KINDS_V1 = (
    "operation_input",
    "operation_output",
    "handoff",
    "diagnostic",
)


class ArtifactStoreError(RuntimeError):
    """Base class for public, fail-closed Artifact store errors."""


class ArtifactStoreConflictError(ArtifactStoreError):
    """Raised when an idempotency key or immutable identity is reused."""


class ArtifactStoreCapacityError(ArtifactStoreError):
    """Raised before work when bounded store capacity is unavailable."""


class ArtifactStoreIntegrityError(ArtifactStoreError):
    """Raised when staged or committed bytes do not match their descriptor."""


@dataclass(frozen=True)
class ArtifactUploadLease:
    upload_id: str
    client_id: str
    runtime_session_id: str
    idempotency_key: str
    artifact: AgentArtifactRef
    state: str
    received_bytes: int
    created_at: float
    expires_at: float

    def __post_init__(self) -> None:
        if self.state not in ARTIFACT_STORE_STATES_V1:
            raise ValueError(f"Unsupported Artifact upload state: {self.state!r}.")

    @property
    def committed(self) -> bool:
        return self.state == "available"


@dataclass(frozen=True)
class ArtifactStoreRecord:
    artifact: AgentArtifactRef
    state: str
    created_at: float
    available_at: float | None
    minimum_retention_until: float
    handoff_deadline: float
    materialization_receipt_hash: str | None
    expired_at: float | None
    deletion_reason: str | None

    def __post_init__(self) -> None:
        if self.state not in ARTIFACT_STORE_STATES_V1:
            raise ValueError(f"Unsupported Artifact state: {self.state!r}.")

    @property
    def available(self) -> bool:
        return self.state == "available"


@dataclass(frozen=True)
class ArtifactGarbageCollectionReport:
    aborted_upload_ids: tuple[str, ...]
    released_pin_ids: tuple[str, ...]
    expired_artifact_ids: tuple[str, ...]
    reclaimed_bytes: int


def _canonical_ref_json(artifact: AgentArtifactRef) -> str:
    return json.dumps(
        artifact.to_wire(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _artifact_from_json(encoded: str) -> AgentArtifactRef:
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ArtifactStoreIntegrityError("Stored ArtifactRef is not an object.")
    return AgentArtifactRef.from_wire(payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactStoreIntegrityError("Artifact bytes are not a regular file.")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        return total, digest.hexdigest()
    finally:
        os.close(descriptor)


class RecoverableAgentArtifactStore:
    """Thread-safe local CAS with staging, pins, handoff, and tombstones."""

    def __init__(
        self,
        *,
        root: str | os.PathLike[str],
        quota_bytes: int,
        max_artifact_bytes: int,
        max_chunk_bytes: int = 4 * 1024 * 1024,
        staging_ttl_sec: float = 300.0,
        minimum_retention_sec: float = 300.0,
        handoff_retention_sec: float = 3600.0,
    ) -> None:
        raw_root = Path(os.fspath(root))
        if not raw_root.is_absolute():
            raise ValueError("Artifact store root must be an absolute path.")
        self.root = raw_root
        self.quota_bytes = positive_integer(quota_bytes, field_name="quota_bytes")
        self.max_artifact_bytes = positive_integer(
            max_artifact_bytes,
            field_name="max_artifact_bytes",
        )
        self.max_chunk_bytes = positive_integer(
            max_chunk_bytes,
            field_name="max_chunk_bytes",
        )
        if self.max_artifact_bytes > self.quota_bytes:
            raise ValueError("max_artifact_bytes cannot exceed quota_bytes.")
        self.staging_ttl_sec = self._positive_duration(
            staging_ttl_sec,
            field_name="staging_ttl_sec",
        )
        self.minimum_retention_sec = self._positive_duration(
            minimum_retention_sec,
            field_name="minimum_retention_sec",
        )
        self.handoff_retention_sec = self._positive_duration(
            handoff_retention_sec,
            field_name="handoff_retention_sec",
        )
        if self.handoff_retention_sec < self.minimum_retention_sec:
            raise ValueError(
                "handoff_retention_sec cannot be shorter than minimum_retention_sec."
            )

        self._lock = threading.RLock()
        self._closed = False
        self._prepare_managed_root()
        self._staging_root = self.root / "staging"
        self._blob_root = self.root / "blobs" / "sha256"
        self._staging_root.mkdir(mode=0o700, exist_ok=True)
        (self.root / "blobs").mkdir(mode=0o700, exist_ok=True)
        self._blob_root.mkdir(mode=0o700, exist_ok=True)
        for path in (self._staging_root, self.root / "blobs", self._blob_root):
            self._require_managed_directory(path)

        database_path = self.root / "artifact_store.sqlite3"
        if database_path.exists():
            info = os.lstat(database_path)
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ValueError("Artifact store database must be a regular file.")
        self._connection = sqlite3.connect(
            str(database_path),
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._initialize_schema()

    @staticmethod
    def _positive_duration(value: object, *, field_name: str) -> float:
        normalized = finite_number(value, field_name=field_name)
        if normalized <= 0:
            raise ValueError(f"{field_name} must be positive.")
        return normalized

    def _prepare_managed_root(self) -> None:
        parent = self.root.parent
        parent_info = os.lstat(parent)
        if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(
            parent_info.st_mode
        ):
            raise ValueError("Artifact store parent must be a real directory.")
        if parent_info.st_mode & stat.S_IWOTH:
            raise ValueError("Artifact store parent cannot be world-writable.")
        try:
            self.root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        self._require_managed_directory(self.root)

    @staticmethod
    def _require_managed_directory(path: Path) -> None:
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Managed Artifact path is not a real directory: {path}.")
        if info.st_mode & stat.S_IWOTH:
            raise ValueError(f"Managed Artifact directory is world-writable: {path}.")

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_ref_json TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    byte_length INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    available_at REAL,
                    minimum_retention_until REAL NOT NULL,
                    handoff_deadline REAL NOT NULL,
                    materialization_receipt_hash TEXT,
                    expired_at REAL,
                    deletion_reason TEXT
                );
                CREATE INDEX IF NOT EXISTS artifacts_sha256_idx
                    ON artifacts (sha256, state);
                CREATE TABLE IF NOT EXISTS uploads (
                    upload_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    runtime_session_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    artifact_ref_json TEXT NOT NULL,
                    staging_name TEXT,
                    state TEXT NOT NULL,
                    received_bytes INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    UNIQUE (client_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS artifact_pins (
                    pin_id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    pin_kind TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    FOREIGN KEY (artifact_id) REFERENCES artifacts (artifact_id)
                );
                CREATE INDEX IF NOT EXISTS artifact_pins_artifact_idx
                    ON artifact_pins (artifact_id);
                """
            )
            row = self._connection.execute(
                "SELECT schema_version FROM store_meta WHERE singleton = 1"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO store_meta (singleton, schema_version) VALUES (1, ?)",
                    (ARTIFACT_STORE_SCHEMA_VERSION_V1,),
                )
            elif row["schema_version"] != ARTIFACT_STORE_SCHEMA_VERSION_V1:
                raise RuntimeError(
                    "Unsupported Artifact store schema version: "
                    f"{row['schema_version']!r}."
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ArtifactStoreError("Artifact store is closed.")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "RecoverableAgentArtifactStore":
        self._ensure_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _blob_path(self, sha256: str) -> Path:
        directory = self._blob_root / sha256[:2]
        if not directory.exists():
            directory.mkdir(mode=0o700)
            _fsync_directory(self._blob_root)
        self._require_managed_directory(directory)
        return directory / sha256

    def _staging_path(self, staging_name: str) -> Path:
        if Path(staging_name).name != staging_name:
            raise ArtifactStoreIntegrityError("Invalid managed staging name.")
        return self._staging_root / staging_name

    def _reserved_bytes_locked(self) -> int:
        artifact_bytes = self._connection.execute(
            "SELECT COALESCE(SUM(byte_length), 0) AS total "
            "FROM artifacts WHERE state = 'available'"
        ).fetchone()["total"]
        staging_rows = self._connection.execute(
            "SELECT artifact_ref_json FROM uploads "
            "WHERE state IN ('staging', 'verifying')"
        ).fetchall()
        staging_bytes = sum(
            _artifact_from_json(row["artifact_ref_json"]).byte_length
            for row in staging_rows
        )
        return int(artifact_bytes) + staging_bytes

    def begin_upload(
        self,
        *,
        client_id: str,
        runtime_session_id: str,
        idempotency_key: str,
        artifact: AgentArtifactRef,
        now: float,
    ) -> ArtifactUploadLease:
        """Reserve quota and create one server-owned staging inode."""

        client = identifier(client_id, field_name="client_id")
        session = identifier(runtime_session_id, field_name="runtime_session_id")
        idempotency = identifier(idempotency_key, field_name="idempotency_key")
        timestamp = finite_number(now, field_name="now")
        if not isinstance(artifact, AgentArtifactRef):
            raise TypeError("artifact must be an AgentArtifactRef.")
        if artifact.byte_length > self.max_artifact_bytes:
            raise ArtifactStoreCapacityError("Artifact exceeds max_artifact_bytes.")
        encoded_ref = _canonical_ref_json(artifact)

        with self._lock:
            self._ensure_open()
            existing = self._connection.execute(
                "SELECT * FROM uploads WHERE client_id = ? AND idempotency_key = ?",
                (client, idempotency),
            ).fetchone()
            if existing is not None:
                if existing["runtime_session_id"] != session:
                    raise ArtifactStoreConflictError(
                        "Upload idempotency key belongs to another runtime session."
                    )
                if existing["artifact_ref_json"] != encoded_ref:
                    raise ArtifactStoreConflictError(
                        "Upload idempotency key is bound to another Artifact."
                    )
                return self._upload_lease(existing)

            available = self._connection.execute(
                "SELECT state FROM artifacts WHERE artifact_id = ?",
                (artifact.artifact_id,),
            ).fetchone()
            upload_id = f"upload:{uuid.uuid4().hex}"
            expires_at = timestamp + self.staging_ttl_sec
            if available is not None and available["state"] == "available":
                with self._connection:
                    self._connection.execute(
                        "UPDATE artifacts SET "
                        "minimum_retention_until = MAX(minimum_retention_until, ?), "
                        "handoff_deadline = MAX(handoff_deadline, ?) "
                        "WHERE artifact_id = ?",
                        (
                            timestamp + self.minimum_retention_sec,
                            timestamp + self.handoff_retention_sec,
                            artifact.artifact_id,
                        ),
                    )
                    self._connection.execute(
                        "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, NULL, "
                        "'available', ?, ?, ?)",
                        (
                            upload_id,
                            client,
                            session,
                            idempotency,
                            encoded_ref,
                            artifact.byte_length,
                            timestamp,
                            expires_at,
                        ),
                    )
                return self.upload(upload_id)

            if self._reserved_bytes_locked() + artifact.byte_length > self.quota_bytes:
                raise ArtifactStoreCapacityError("Artifact store quota is exhausted.")

            staging_name = upload_id.split(":", 1)[1] + ".part"
            staging_path = self._staging_path(staging_name)
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(staging_path, flags, 0o600)
            os.close(descriptor)
            _fsync_directory(self._staging_root)
            try:
                with self._connection:
                    self._connection.execute(
                        "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?, "
                        "'staging', 0, ?, ?)",
                        (
                            upload_id,
                            client,
                            session,
                            idempotency,
                            encoded_ref,
                            staging_name,
                            timestamp,
                            expires_at,
                        ),
                    )
            except BaseException:
                staging_path.unlink(missing_ok=True)
                raise
            return self.upload(upload_id)

    @staticmethod
    def _upload_lease(row: sqlite3.Row) -> ArtifactUploadLease:
        return ArtifactUploadLease(
            upload_id=row["upload_id"],
            client_id=row["client_id"],
            runtime_session_id=row["runtime_session_id"],
            idempotency_key=row["idempotency_key"],
            artifact=_artifact_from_json(row["artifact_ref_json"]),
            state=row["state"],
            received_bytes=row["received_bytes"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )

    def upload(self, upload_id: str) -> ArtifactUploadLease:
        normalized = identifier(upload_id, field_name="upload_id")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT * FROM uploads WHERE upload_id = ?",
                (normalized,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown Artifact upload: {normalized!r}.")
            return self._upload_lease(row)

    def _owned_upload_locked(
        self,
        *,
        upload_id: str,
        client_id: str,
        runtime_session_id: str,
    ) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM uploads WHERE upload_id = ?",
            (identifier(upload_id, field_name="upload_id"),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown Artifact upload: {upload_id!r}.")
        if row["client_id"] != identifier(client_id, field_name="client_id") or row[
            "runtime_session_id"
        ] != identifier(runtime_session_id, field_name="runtime_session_id"):
            raise PermissionError("Artifact upload does not belong to this session.")
        return row

    def write_upload_chunk(
        self,
        *,
        upload_id: str,
        client_id: str,
        runtime_session_id: str,
        offset: int,
        data: bytes | bytearray | memoryview,
        now: float,
    ) -> ArtifactUploadLease:
        """Append one bounded chunk, accepting only byte-identical replay."""

        timestamp = finite_number(now, field_name="now")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like.")
        chunk = bytes(data)
        if not chunk or len(chunk) > self.max_chunk_bytes:
            raise ValueError(
                "Artifact upload chunk is empty or exceeds max_chunk_bytes."
            )
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")

        with self._lock:
            self._ensure_open()
            row = self._owned_upload_locked(
                upload_id=upload_id,
                client_id=client_id,
                runtime_session_id=runtime_session_id,
            )
            if row["state"] != "staging":
                raise ArtifactStoreConflictError(
                    f"Upload is not writable in state {row['state']!r}."
                )
            if timestamp >= row["expires_at"]:
                raise ArtifactStoreConflictError("Artifact upload lease has expired.")
            artifact = _artifact_from_json(row["artifact_ref_json"])
            received = int(row["received_bytes"])
            if offset > received:
                raise ArtifactStoreConflictError("Upload chunk creates an offset gap.")
            if offset + len(chunk) > artifact.byte_length:
                raise ArtifactStoreCapacityError(
                    "Upload bytes exceed the immutable Artifact descriptor."
                )
            path = self._staging_path(row["staging_name"])
            flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_size != received:
                    raise ArtifactStoreIntegrityError(
                        "Staging inode size does not match committed upload progress."
                    )
                if offset < received:
                    if offset + len(chunk) > received:
                        raise ArtifactStoreConflictError(
                            "A replay chunk cannot overlap uncommitted bytes."
                        )
                    existing = os.pread(descriptor, len(chunk), offset)
                    if existing != chunk:
                        raise ArtifactStoreConflictError(
                            "Replayed upload chunk differs from committed bytes."
                        )
                    return self._upload_lease(row)
                written = os.pwrite(descriptor, chunk, offset)
                if written != len(chunk):
                    raise ArtifactStoreIntegrityError(
                        "Artifact chunk write was partial."
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            with self._connection:
                self._connection.execute(
                    "UPDATE uploads SET received_bytes = ? WHERE upload_id = ?",
                    (received + len(chunk), row["upload_id"]),
                )
            return self.upload(row["upload_id"])

    def commit_upload(
        self,
        *,
        upload_id: str,
        client_id: str,
        runtime_session_id: str,
        now: float,
    ) -> AgentArtifactRef:
        """Verify all bytes and atomically make one immutable Artifact available."""

        timestamp = finite_number(now, field_name="now")
        with self._lock:
            self._ensure_open()
            row = self._owned_upload_locked(
                upload_id=upload_id,
                client_id=client_id,
                runtime_session_id=runtime_session_id,
            )
            artifact = _artifact_from_json(row["artifact_ref_json"])
            if row["state"] == "available":
                return artifact
            if row["state"] not in {"staging", "verifying"}:
                raise ArtifactStoreConflictError(
                    f"Upload cannot commit from state {row['state']!r}."
                )
            if timestamp >= row["expires_at"]:
                raise ArtifactStoreConflictError("Artifact upload lease has expired.")
            if row["received_bytes"] != artifact.byte_length:
                raise ArtifactStoreConflictError("Artifact upload is incomplete.")
            with self._connection:
                self._connection.execute(
                    "UPDATE uploads SET state = 'verifying' WHERE upload_id = ?",
                    (row["upload_id"],),
                )

            staging_path = self._staging_path(row["staging_name"])
            blob_path = self._blob_path(artifact.sha256)
            candidate = staging_path if staging_path.exists() else blob_path
            try:
                byte_length, digest = _sha256_file(candidate)
                if byte_length != artifact.byte_length or digest != artifact.sha256:
                    raise ArtifactStoreIntegrityError(
                        "Staged bytes do not match Artifact size/hash."
                    )
                if blob_path.exists() and candidate != blob_path:
                    existing_size, existing_digest = _sha256_file(blob_path)
                    if (
                        existing_size != artifact.byte_length
                        or existing_digest != artifact.sha256
                    ):
                        os.chmod(staging_path, 0o400)
                        os.replace(staging_path, blob_path)
                    else:
                        staging_path.unlink()
                elif candidate != blob_path:
                    os.chmod(staging_path, 0o400)
                    os.replace(staging_path, blob_path)
                descriptor = os.open(
                    blob_path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                _fsync_directory(blob_path.parent)
            except (ArtifactStoreIntegrityError, OSError):
                with self._connection:
                    self._connection.execute(
                        "UPDATE uploads SET state = 'quarantined' WHERE upload_id = ?",
                        (row["upload_id"],),
                    )
                raise

            minimum_until = timestamp + self.minimum_retention_sec
            handoff_deadline = timestamp + self.handoff_retention_sec
            with self._connection:
                self._connection.execute(
                    "INSERT INTO artifacts (artifact_id, artifact_ref_json, sha256, "
                    "byte_length, state, created_at, available_at, "
                    "minimum_retention_until, handoff_deadline, "
                    "materialization_receipt_hash, expired_at, deletion_reason) "
                    "VALUES (?, ?, ?, ?, 'available', ?, ?, ?, ?, NULL, NULL, NULL) "
                    "ON CONFLICT(artifact_id) DO UPDATE SET "
                    "artifact_ref_json = excluded.artifact_ref_json, "
                    "sha256 = excluded.sha256, byte_length = excluded.byte_length, "
                    "state = 'available', available_at = excluded.available_at, "
                    "minimum_retention_until = MAX("
                    "artifacts.minimum_retention_until, "
                    "excluded.minimum_retention_until), "
                    "handoff_deadline = MAX(artifacts.handoff_deadline, "
                    "excluded.handoff_deadline), "
                    "expired_at = NULL, "
                    "deletion_reason = NULL",
                    (
                        artifact.artifact_id,
                        _canonical_ref_json(artifact),
                        artifact.sha256,
                        artifact.byte_length,
                        row["created_at"],
                        timestamp,
                        minimum_until,
                        handoff_deadline,
                    ),
                )
                self._connection.execute(
                    "UPDATE uploads SET state = 'available', received_bytes = ? "
                    "WHERE upload_id = ?",
                    (artifact.byte_length, row["upload_id"]),
                )
            return artifact

    def artifact(self, artifact_id: str) -> ArtifactStoreRecord:
        normalized = identifier(artifact_id, field_name="artifact_id")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (normalized,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown Artifact: {normalized!r}.")
            return ArtifactStoreRecord(
                artifact=_artifact_from_json(row["artifact_ref_json"]),
                state=row["state"],
                created_at=row["created_at"],
                available_at=row["available_at"],
                minimum_retention_until=row["minimum_retention_until"],
                handoff_deadline=row["handoff_deadline"],
                materialization_receipt_hash=row[
                    "materialization_receipt_hash"
                ],
                expired_at=row["expired_at"],
                deletion_reason=row["deletion_reason"],
            )

    def managed_read_path(self, artifact_id: str) -> Path:
        """Return an internal verified path for the transport grant issuer."""

        record = self.artifact(artifact_id)
        if not record.available:
            raise ArtifactStoreConflictError(
                f"Artifact is not available in state {record.state!r}."
            )
        path = self._blob_path(record.artifact.sha256)
        size, digest = _sha256_file(path)
        if size != record.artifact.byte_length or digest != record.artifact.sha256:
            with self._lock, self._connection:
                self._connection.execute(
                    "UPDATE artifacts SET state = 'quarantined', "
                    "deletion_reason = 'committed_blob_integrity_failure' "
                    "WHERE artifact_id = ?",
                    (record.artifact.artifact_id,),
                )
            raise ArtifactStoreIntegrityError("Committed Artifact failed verification.")
        return path

    def pin(
        self,
        *,
        pin_id: str,
        artifact_id: str,
        pin_kind: str,
        owner_id: str,
        now: float,
        expires_at: float | None = None,
    ) -> None:
        normalized_pin = identifier(pin_id, field_name="pin_id")
        normalized_artifact = identifier(artifact_id, field_name="artifact_id")
        normalized_owner = identifier(owner_id, field_name="owner_id")
        timestamp = finite_number(now, field_name="now")
        if pin_kind not in ARTIFACT_STORE_PIN_KINDS_V1:
            raise ValueError(f"Unsupported Artifact pin kind: {pin_kind!r}.")
        expiry = (
            None
            if expires_at is None
            else finite_number(expires_at, field_name="expires_at")
        )
        if expiry is not None and expiry <= timestamp:
            raise ValueError("expires_at must follow now.")
        with self._lock:
            self._ensure_open()
            record = self.artifact(normalized_artifact)
            if not record.available:
                raise ArtifactStoreConflictError(
                    "Only available Artifacts may be pinned."
                )
            existing = self._connection.execute(
                "SELECT * FROM artifact_pins WHERE pin_id = ?",
                (normalized_pin,),
            ).fetchone()
            expected = (
                normalized_artifact,
                pin_kind,
                normalized_owner,
                expiry,
            )
            if existing is not None:
                actual = (
                    existing["artifact_id"],
                    existing["pin_kind"],
                    existing["owner_id"],
                    existing["expires_at"],
                )
                if actual != expected:
                    raise ArtifactStoreConflictError(
                        "Artifact pin ID is bound to another claim."
                    )
                return
            with self._connection:
                self._connection.execute(
                    "INSERT INTO artifact_pins VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        normalized_pin,
                        normalized_artifact,
                        pin_kind,
                        normalized_owner,
                        timestamp,
                        expiry,
                    ),
                )

    def release_pin(self, *, pin_id: str, owner_id: str) -> bool:
        normalized_pin = identifier(pin_id, field_name="pin_id")
        normalized_owner = identifier(owner_id, field_name="owner_id")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT owner_id FROM artifact_pins WHERE pin_id = ?",
                (normalized_pin,),
            ).fetchone()
            if row is None:
                return False
            if row["owner_id"] != normalized_owner:
                raise PermissionError("Artifact pin belongs to another owner.")
            with self._connection:
                self._connection.execute(
                    "DELETE FROM artifact_pins WHERE pin_id = ?",
                    (normalized_pin,),
                )
            return True

    def record_materialization(
        self,
        receipt: AgentArtifactMaterializationReceipt,
    ) -> None:
        if not isinstance(receipt, AgentArtifactMaterializationReceipt):
            raise TypeError(
                "receipt must be an AgentArtifactMaterializationReceipt."
            )
        with self._lock:
            self._ensure_open()
            record = self.artifact(receipt.artifact_id)
            if not record.available:
                raise ArtifactStoreConflictError(
                    "Only an available Artifact can be materialized."
                )
            if receipt.verified_sha256 != record.artifact.sha256:
                raise ArtifactStoreIntegrityError(
                    "Materialization receipt hash does not match Artifact bytes."
                )
            existing = record.materialization_receipt_hash
            if existing is not None and existing != receipt.semantic_content_hash:
                raise ArtifactStoreConflictError(
                    "Artifact already has a different materialization receipt."
                )
            with self._connection:
                self._connection.execute(
                    "UPDATE artifacts SET materialization_receipt_hash = ? "
                    "WHERE artifact_id = ?",
                    (receipt.semantic_content_hash, receipt.artifact_id),
                )

    def recover(self, *, now: float) -> tuple[str, ...]:
        """Audit available bytes after restart and quarantine invalid records."""

        timestamp = finite_number(now, field_name="now")
        quarantined: list[str] = []
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT artifact_id, artifact_ref_json FROM artifacts "
                "WHERE state = 'available'"
            ).fetchall()
            for row in rows:
                artifact = _artifact_from_json(row["artifact_ref_json"])
                path = self._blob_path(artifact.sha256)
                try:
                    size, digest = _sha256_file(path)
                except (FileNotFoundError, ArtifactStoreIntegrityError, OSError):
                    size, digest = -1, ""
                if size != artifact.byte_length or digest != artifact.sha256:
                    with self._connection:
                        self._connection.execute(
                            "UPDATE artifacts SET state = 'quarantined', "
                            "expired_at = ?, deletion_reason = ? "
                            "WHERE artifact_id = ?",
                            (
                                timestamp,
                                "restart_integrity_check_failed",
                                artifact.artifact_id,
                            ),
                        )
                    quarantined.append(artifact.artifact_id)
            self.garbage_collect(now=timestamp)
        return tuple(sorted(quarantined))

    def garbage_collect(self, *, now: float) -> ArtifactGarbageCollectionReport:
        """Expire leases/pins and reclaim only unreferenced eligible bytes."""

        timestamp = finite_number(now, field_name="now")
        aborted_uploads: list[str] = []
        released_pins: list[str] = []
        expired_artifacts: list[str] = []
        reclaimed_bytes = 0
        with self._lock:
            self._ensure_open()
            expired_upload_rows = self._connection.execute(
                "SELECT upload_id, staging_name FROM uploads "
                "WHERE state IN ('staging', 'verifying') AND expires_at <= ?",
                (timestamp,),
            ).fetchall()
            for row in expired_upload_rows:
                if row["staging_name"] is not None:
                    self._staging_path(row["staging_name"]).unlink(missing_ok=True)
                aborted_uploads.append(row["upload_id"])
            if expired_upload_rows:
                with self._connection:
                    self._connection.executemany(
                        "UPDATE uploads SET state = 'aborted' WHERE upload_id = ?",
                        [(row["upload_id"],) for row in expired_upload_rows],
                    )

            expired_pin_rows = self._connection.execute(
                "SELECT pin_id FROM artifact_pins "
                "WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (timestamp,),
            ).fetchall()
            released_pins.extend(row["pin_id"] for row in expired_pin_rows)
            if expired_pin_rows:
                with self._connection:
                    self._connection.executemany(
                        "DELETE FROM artifact_pins WHERE pin_id = ?",
                        [(row["pin_id"],) for row in expired_pin_rows],
                    )

            candidates = self._connection.execute(
                "SELECT * FROM artifacts WHERE state = 'available' "
                "AND minimum_retention_until <= ?",
                (timestamp,),
            ).fetchall()
            for row in candidates:
                pin = self._connection.execute(
                    "SELECT 1 FROM artifact_pins WHERE artifact_id = ? LIMIT 1",
                    (row["artifact_id"],),
                ).fetchone()
                if pin is not None:
                    continue
                if (
                    row["materialization_receipt_hash"] is None
                    and timestamp < row["handoff_deadline"]
                ):
                    continue
                other = self._connection.execute(
                    "SELECT 1 FROM artifacts WHERE sha256 = ? AND state = 'available' "
                    "AND artifact_id != ? LIMIT 1",
                    (row["sha256"], row["artifact_id"]),
                ).fetchone()
                if other is None:
                    path = self._blob_path(row["sha256"])
                    if path.exists():
                        size = os.lstat(path).st_size
                        path.unlink()
                        reclaimed_bytes += size
                        _fsync_directory(path.parent)
                reason = (
                    "materialized_and_minimum_retention_elapsed"
                    if row["materialization_receipt_hash"] is not None
                    else "handoff_deadline_elapsed"
                )
                with self._connection:
                    self._connection.execute(
                        "UPDATE artifacts SET state = 'expired', expired_at = ?, "
                        "deletion_reason = ? WHERE artifact_id = ?",
                        (timestamp, reason, row["artifact_id"]),
                    )
                expired_artifacts.append(row["artifact_id"])
        return ArtifactGarbageCollectionReport(
            aborted_upload_ids=tuple(sorted(aborted_uploads)),
            released_pin_ids=tuple(sorted(released_pins)),
            expired_artifact_ids=tuple(sorted(expired_artifacts)),
            reclaimed_bytes=reclaimed_bytes,
        )


__all__ = [
    "ArtifactGarbageCollectionReport",
    "ArtifactStoreCapacityError",
    "ArtifactStoreConflictError",
    "ArtifactStoreError",
    "ArtifactStoreIntegrityError",
    "ArtifactStoreRecord",
    "ArtifactUploadLease",
    "RecoverableAgentArtifactStore",
]
