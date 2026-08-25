"""Transactional cold-storage operations for detachable dataset modalities.

The production CLI is deliberately dry-run first.  The strict detach/rehydrate
functions require the exact plan hash and source-binding confirmation emitted
by the preceding proposal; the managed move freezes the same plan inside one
explicit apply invocation.  Compatibility field names still say "dataset
fingerprint", but new tactile inventories bind only their tactile reference
projection.  None of these operations rewrite capture facts or manifests.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
import time
from typing import Any, Iterator, Mapping
import uuid

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.modality_artifacts import (
    ModalityArtifactInventoryV1,
    ModalityStorageTransferReceiptV1,
)
from dataset_contracts.tactile import ChunkedTactileStream, tactile_stream_compression
from dataset_interface.modality_artifacts import (
    DETACHMENT_SCOPES,
    DETACHED_VERIFIED,
    MODALITY_ARTIFACT_SIDECAR_RELATIVE,
    ModalityArtifactInventoryError,
    PAYLOAD_ONLY_DETACH,
    SELF_CONTAINED,
    WHOLE_PREFIX_DETACH,
    PAYLOAD_HASH_PREFER_SEALED,
    PAYLOAD_HASH_SCAN,
    build_modality_artifact_inventory,
    inventory_source_binding_matches,
    resolve_modality_availability,
    verified_detachment_scope,
    verify_modality_artifact_inventory,
)

from .dataset_layout import sanitize_dataset_name_for_path
from .modality_inventory import (
    load_modality_inventory as _load_published_modality_inventory,
    publish_modality_inventory,
)
from .process_quiescence import (
    active_formal_collection_processes as _active_collection_processes,
)


PLAN_HASH_DOMAIN_V1 = "dexteleop.modality_storage_plan.v1"
DEFAULT_MIN_FREE_BYTES = 1024**3
RESUMABLE_COPY_CHUNK_BYTES = 8 * 1024**2
FSYNC_MAX_ATTEMPTS = 12
FSYNC_INITIAL_RETRY_DELAY_SEC = 0.05
FSYNC_MAX_RETRY_DELAY_SEC = 1.0
_TRANSIENT_FSYNC_ERRNOS = frozenset(
    {
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.EINTR,
        errno.EBUSY,
    }
)
REPLACE_MAX_ATTEMPTS = 12
REPLACE_INITIAL_RETRY_DELAY_SEC = 0.05
REPLACE_MAX_RETRY_DELAY_SEC = 1.0
FULL_REVALIDATION = "full"
SAMPLED_REVALIDATION_V1 = "sampled_v1"
REVALIDATION_PROFILES = (FULL_REVALIDATION, SAMPLED_REVALIDATION_V1)
SAMPLED_REVALIDATION_MANIFEST_COUNT = 16


class ModalityStorageError(RuntimeError):
    """Raised when a cold-storage operation cannot prove safe completion."""


@dataclass(frozen=True)
class ModalityStoragePlan:
    operation: str
    detachment_scope: str
    dataset_root: str
    cold_root: str
    cold_artifact_root: str
    inventory_hash: str
    dataset_fingerprint: str
    episode_catalog_sha256: str
    source_device: int
    target_device: int
    require_distinct_device: bool
    file_count: int
    total_bytes: int
    payload_file_count: int
    payload_bytes: int
    restore_drill_peak_bytes: int
    source_validation_mode: str
    availability: str
    min_free_bytes: int
    ready: bool
    revalidation_profile: str = FULL_REVALIDATION
    revalidation_manifest_count: int = 0
    errors: tuple[str, ...] = ()

    def semantic_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("ready", None)
        payload.pop("errors", None)
        # Preserve the pre-sampled-feature plan hash for the default path.
        # Only an explicitly requested non-default profile extends the
        # confirmation contract.
        if self.revalidation_profile == FULL_REVALIDATION:
            payload.pop("revalidation_profile", None)
            payload.pop("revalidation_manifest_count", None)
        return payload

    @property
    def plan_hash(self) -> str:
        return canonical_content_hash(
            self.semantic_payload(),
            domain=PLAN_HASH_DOMAIN_V1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "plan_hash": self.plan_hash}


@dataclass(frozen=True)
class ModalityStorageResult:
    operation: str
    dataset_root: str
    cold_artifact_root: str
    inventory_hash: str
    plan_hash: str
    final_status: str
    receipt_paths: tuple[str, ...]
    released_hot_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _nearest_existing_parent(path: Path) -> Path:
    cursor = path
    while not cursor.exists():
        if cursor.parent == cursor:
            raise ModalityStorageError(f"no existing parent for {path}")
        cursor = cursor.parent
    if not cursor.is_dir():
        raise ModalityStorageError(f"storage parent is not a directory: {cursor}")
    return cursor


def _is_nested(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
    except ValueError:
        return False
    return True


def _validate_roots(dataset_root: Path, cold_root: Path) -> None:
    if dataset_root == cold_root or _is_nested(cold_root, dataset_root) or _is_nested(
        dataset_root, cold_root
    ):
        raise ModalityStorageError(
            "dataset_root and cold_root must be disjoint directory trees"
        )


def _sidecar_root(dataset_root: Path) -> Path:
    return dataset_root / MODALITY_ARTIFACT_SIDECAR_RELATIVE


def _receipt_path(dataset_root: Path, operation_id: str) -> Path:
    return _sidecar_root(dataset_root) / "receipts" / f"{operation_id}.json"


def _journal_path(dataset_root: Path) -> Path:
    return _sidecar_root(dataset_root) / "active_transaction.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModalityStorageError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ModalityStorageError(f"JSON value is not an object: {path}")
    return payload


def _fsync_with_retry(descriptor: int) -> None:
    """Retry only transient durability errors from network filesystems.

    CIFS can surface a short-lived ``EAGAIN`` from ``fsync`` even though the
    underlying file remains open and valid.  A bounded retry preserves the
    fail-closed contract: persistent or non-transient errors are still raised
    and the deterministic partial export remains available for a later run.
    """

    delay = FSYNC_INITIAL_RETRY_DELAY_SEC
    for attempt in range(FSYNC_MAX_ATTEMPTS):
        try:
            os.fsync(descriptor)
            return
        except OSError as exc:
            if (
                exc.errno not in _TRANSIENT_FSYNC_ERRNOS
                or attempt + 1 >= FSYNC_MAX_ATTEMPTS
            ):
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, FSYNC_MAX_RETRY_DELAY_SEC)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        _fsync_with_retry(descriptor)
    finally:
        os.close(descriptor)


def _replace_with_retry(source: Path, target: Path) -> None:
    """Publish one file despite bounded transient network-filesystem errors.

    CIFS may briefly return ``EBUSY`` for a same-directory rename after the
    payload has already been flushed.  Retrying the atomic rename is safe: the
    source partial remains the only unpublished name, and non-transient or
    persistent failures still fail closed for a later resumable invocation.
    """

    delay = REPLACE_INITIAL_RETRY_DELAY_SEC
    for attempt in range(REPLACE_MAX_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            if (
                exc.errno not in _TRANSIENT_FSYNC_ERRNOS
                or attempt + 1 >= REPLACE_MAX_ATTEMPTS
            ):
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, REPLACE_MAX_RETRY_DELAY_SEC)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ModalityStorageError(f"staging tree contains symlink: {path}")
        if path.is_file():
            with path.open("rb") as handle:
                _fsync_with_retry(handle.fileno())
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


def _copy_file_new_inode(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ModalityStorageError(f"source artifact is unsafe or absent: {source}")
    if int(source.stat().st_nlink) != 1:
        raise ModalityStorageError(f"source artifact is hardlinked: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, target.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=4 * 1024 * 1024)
        output_handle.flush()
        _fsync_with_retry(output_handle.fileno())
    if int(target.stat().st_nlink) != 1:
        raise ModalityStorageError(f"target artifact is not an independent inode: {target}")
    source_stat = source.stat()
    target_stat = target.stat()
    if source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino:
        raise ModalityStorageError(f"copy reused the source inode: {source}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(RESUMABLE_COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


_COLD_STAGING_NAMESPACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def _normalize_cold_staging_namespace(value: str) -> str:
    namespace = str(value or "").strip()
    if not namespace:
        return ""
    if not _COLD_STAGING_NAMESPACE_RE.fullmatch(namespace):
        raise ModalityStorageError(
            "cold staging namespace must be 1-64 path-safe ASCII characters "
            "starting with an alphanumeric character"
        )
    return namespace


def _cold_partial_root(
    artifact_root: Path,
    *,
    staging_namespace: str = "",
) -> Path:
    namespace = _normalize_cold_staging_namespace(staging_namespace)
    suffix = f".partial.{namespace}" if namespace else ".partial"
    return artifact_root.parent / f".{artifact_root.name}{suffix}"


def _partial_file_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.modality-part")


def _assert_regular_single_link(path: Path, *, label: str) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise ModalityStorageError(f"{label} is unsafe or absent: {path}")
    stat = path.stat()
    if int(stat.st_nlink) != 1:
        raise ModalityStorageError(f"{label} is hardlinked: {path}")
    return stat


def _assert_regular_source(
    path: Path,
    *,
    label: str,
    allow_hardlinks: bool,
) -> os.stat_result:
    if not allow_hardlinks:
        return _assert_regular_single_link(path, label=label)
    if path.is_symlink() or not path.is_file():
        raise ModalityStorageError(f"{label} is unsafe or absent: {path}")
    stat = path.stat()
    return stat


def _copy_file_resumable(
    source: Path,
    target: Path,
    artifact: Any,
    *,
    allow_hardlinked_source: bool = False,
) -> None:
    """Copy one inventory artifact with durable, prefix-verified resume.

    A completed target is immutable and reusable only when its exact inventory
    size/hash match.  An interrupted ``.modality-part`` file is compared byte
    for byte with the current source prefix before appending.  The completed
    name is published only after the entire source matches the frozen
    inventory record and the file has been fsynced.
    """

    source_stat = _assert_regular_source(
        source,
        label="source artifact",
        allow_hardlinks=bool(allow_hardlinked_source),
    )
    expected_size = int(artifact.size_bytes)
    expected_hash = str(artifact.sha256)
    if int(source_stat.st_size) != expected_size:
        raise ModalityStorageError(
            f"source artifact size differs from inventory: {source}"
        )
    if target.exists() or target.is_symlink():
        target_stat = _assert_regular_single_link(
            target,
            label="completed cold artifact",
        )
        if (
            int(target_stat.st_size) != expected_size
            or _sha256_file(target) != expected_hash
        ):
            raise ModalityStorageError(
                f"completed cold artifact differs from inventory: {target}"
            )
        if (
            source_stat.st_dev == target_stat.st_dev
            and source_stat.st_ino == target_stat.st_ino
        ):
            raise ModalityStorageError(f"cold artifact reused source inode: {source}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_file_path(target)
    if partial.exists() or partial.is_symlink():
        partial_stat = _assert_regular_single_link(
            partial,
            label="partial cold artifact",
        )
        if int(partial_stat.st_size) > expected_size:
            partial.unlink()
            _fsync_directory(partial.parent)
    if not partial.exists():
        with partial.open("xb") as handle:
            handle.flush()
            _fsync_with_retry(handle.fileno())

    initial_fields = (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_nlink,
        source_stat.st_size,
        source_stat.st_mtime_ns,
    )
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_handle, partial.open("r+b") as output_handle:
            prefix_size = int(partial.stat().st_size)
            prefix_valid = True
            remaining = prefix_size
            while remaining > 0:
                size = min(RESUMABLE_COPY_CHUNK_BYTES, remaining)
                source_chunk = input_handle.read(size)
                partial_chunk = output_handle.read(size)
                if source_chunk != partial_chunk or len(source_chunk) != size:
                    prefix_valid = False
                    break
                digest.update(source_chunk)
                remaining -= size
            if not prefix_valid:
                output_handle.seek(0)
                output_handle.truncate(0)
                input_handle.seek(0)
                digest = hashlib.sha256()
                prefix_size = 0
            else:
                output_handle.seek(prefix_size)
                input_handle.seek(prefix_size)
            while True:
                chunk = input_handle.read(RESUMABLE_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                output_handle.write(chunk)
                digest.update(chunk)
            output_handle.flush()
            _fsync_with_retry(output_handle.fileno())
        current_stat = source.stat()
        current_fields = (
            current_stat.st_dev,
            current_stat.st_ino,
            current_stat.st_nlink,
            current_stat.st_size,
            current_stat.st_mtime_ns,
        )
        if current_fields != initial_fields:
            raise ModalityStorageError(
                f"source artifact changed during cold copy: {source}"
            )
        if (
            partial.stat().st_size != expected_size
            or digest.hexdigest() != expected_hash
        ):
            raise ModalityStorageError(
                f"source artifact hash differs from frozen inventory: {source}"
            )
        _replace_with_retry(partial, target)
        _fsync_directory(target.parent)
        target_stat = _assert_regular_single_link(
            target,
            label="completed cold artifact",
        )
        if (
            source_stat.st_dev == target_stat.st_dev
            and source_stat.st_ino == target_stat.st_ino
        ):
            raise ModalityStorageError(f"cold artifact reused source inode: {source}")
    except BaseException:
        # Keep a size-bounded, known-path partial file for the next invocation.
        # A changed source/inventory cannot be accepted because the outer plan
        # hash will change and this prefix will be revalidated before reuse.
        raise


def _copy_inventory_artifacts_resumable(
    source_root: Path,
    target_root: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    allow_hardlinked_source: bool = False,
) -> None:
    for artifact in inventory.artifacts:
        if allow_hardlinked_source:
            _copy_file_resumable(
                source_root / artifact.logical_path,
                target_root / artifact.logical_path,
                artifact,
                allow_hardlinked_source=True,
            )
        else:
            _copy_file_resumable(
                source_root / artifact.logical_path,
                target_root / artifact.logical_path,
                artifact,
            )


def _copy_inventory_artifacts(
    source_root: Path,
    target_root: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    roles: frozenset[str] = frozenset({"manifest", "payload"}),
    allow_hardlinked_source: bool = False,
) -> None:
    for artifact in inventory.artifacts:
        if artifact.role not in roles:
            continue
        source = source_root / artifact.logical_path
        target = target_root / artifact.logical_path
        if allow_hardlinked_source:
            _copy_file_resumable(
                source,
                target,
                artifact,
                allow_hardlinked_source=True,
            )
        else:
            _copy_file_new_inode(source, target)


def _exact_artifact_tree(
    artifact_root: Path,
    inventory: ModalityArtifactInventoryV1,
) -> None:
    tactile_root = artifact_root / "tactile"
    actual = {
        path.relative_to(artifact_root).as_posix()
        for path in tactile_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected = {artifact.logical_path for artifact in inventory.artifacts}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ModalityStorageError(
            f"artifact tree differs from inventory; missing={missing[:5]} extra={extra[:5]}"
        )


def _publish_inventory(
    dataset_root: Path,
    inventory: ModalityArtifactInventoryV1,
) -> Path:
    try:
        return publish_modality_inventory(dataset_root, inventory)
    except Exception as exc:
        raise ModalityStorageError(str(exc)) from exc


def load_modality_inventory(
    dataset_root: str | Path,
    inventory_hash: str,
) -> ModalityArtifactInventoryV1:
    try:
        return _load_published_modality_inventory(dataset_root, inventory_hash)
    except Exception as exc:
        raise ModalityStorageError(str(exc)) from exc


def _publish_receipt(
    dataset_root: Path,
    receipt: ModalityStorageTransferReceiptV1,
) -> Path:
    path = _receipt_path(dataset_root, receipt.operation_id)
    if path.exists():
        existing = ModalityStorageTransferReceiptV1.from_dict(_read_json(path))
        if existing != receipt:
            raise ModalityStorageError(f"receipt operation_id collision: {path}")
        return path
    _atomic_write_json(path, receipt.to_dict())
    return path


def _complete_receipts(
    dataset_root: Path,
    inventory_hash: str,
) -> tuple[ModalityStorageTransferReceiptV1, ...]:
    directory = _sidecar_root(dataset_root) / "receipts"
    if not directory.is_dir():
        return ()
    receipts: list[ModalityStorageTransferReceiptV1] = []
    for path in sorted(directory.glob("*.json")):
        receipt = ModalityStorageTransferReceiptV1.from_dict(_read_json(path))
        if receipt.inventory_hash == inventory_hash and receipt.status == "complete":
            receipts.append(receipt)
    return tuple(receipts)


def _has_receipt(
    dataset_root: Path,
    inventory_hash: str,
    operation: str,
) -> bool:
    return any(
        receipt.operation == operation
        and receipt.independent_copy_verified
        for receipt in _complete_receipts(dataset_root, inventory_hash)
    )


def _receipt(
    *,
    operation: str,
    inventory: ModalityArtifactInventoryV1,
    source_storage_id: str,
    target_storage_id: str,
    file_count: int,
    total_bytes: int,
    independent_copy_verified: bool,
    operation_id: str | None = None,
) -> ModalityStorageTransferReceiptV1:
    return ModalityStorageTransferReceiptV1(
        operation_id=operation_id or f"{operation}.{uuid.uuid4().hex}",
        operation=operation,
        inventory_hash=inventory.semantic_content_hash,
        source_storage_id=source_storage_id,
        target_storage_id=target_storage_id,
        file_count=int(file_count),
        total_bytes=int(total_bytes),
        status="complete",
        independent_copy_verified=bool(independent_copy_verified),
        completed_at=_utc_now(),
    )


def _cold_artifact_root(cold_root: Path, inventory: ModalityArtifactInventoryV1) -> Path:
    return (
        cold_root
        / inventory.source_dataset_base_fingerprint
        / inventory.semantic_content_hash
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _active_collection_dataset_roots(command: str) -> tuple[Path, ...]:
    raw = command.split(" ", 1)[1] if command.startswith("pid=") else command
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return ()
    roots: list[Path] = []
    for index, token in enumerate(tokens[:-1]):
        if token == "--dataset-root":
            roots.append(
                Path(tokens[index + 1]).expanduser().resolve(strict=False)
            )
    return tuple(roots)


def _active_collection_task_names(command: str) -> tuple[str, ...]:
    raw = command.split(" ", 1)[1] if command.startswith("pid=") else command
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return ()
    names: list[str] = []
    accepted_options = {"--collection-task-name", "--task-name"}
    for index, token in enumerate(tokens[:-1]):
        if token in accepted_options:
            value = str(tokens[index + 1]).strip()
            if value:
                names.append(value)
    return tuple(names)


def active_collection_dataset_roots_for_date(date_prefix: str) -> tuple[Path, ...]:
    """Resolve exact live collection leaves without accepting a broad base root.

    Operator tools sometimes need to keep archival maintenance running while a
    collection process writes into the current dated task leaf.  The caller
    supplies only the date; the dataset base and task slug remain authoritative
    values parsed from the live process command line.  Missing or ambiguous
    leaves are not invented and the later quiescence check still revalidates
    every returned path against the live command.
    """

    value = str(date_prefix).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ModalityStorageError(
            f"active collection date must be YYYY-MM-DD, got {date_prefix!r}"
        )
    resolved: list[Path] = []
    for command in _active_collection_processes():
        bases = _active_collection_dataset_roots(command)
        tasks = _active_collection_task_names(command)
        for base in bases:
            for task in tasks:
                candidate = (
                    base / f"{value}_{sanitize_dataset_name_for_path(task)}"
                ).resolve(strict=False)
                if candidate.is_dir() and candidate.parent == base:
                    resolved.append(candidate)
    return tuple(dict.fromkeys(resolved))


def _verified_explicit_active_dataset_roots(
    command: str,
    *,
    advertised_roots: tuple[Path, ...],
    explicit_roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Narrow a CLI base root only when its concrete dated task leaf is proven.

    Real-teleop command lines retain the user-facing dataset base root while
    the collection layout writes into ``YYYY-MM-DD_<task>`` below it.  An
    operator-provided exact active root may narrow that base only when it is an
    existing direct child and its suffix matches the command's task name.
    Otherwise the global overlap check remains fail-closed.
    """

    task_slugs = {
        sanitize_dataset_name_for_path(name)
        for name in _active_collection_task_names(command)
    }
    if not task_slugs:
        return ()
    verified: list[Path] = []
    for explicit in explicit_roots:
        if not explicit.is_dir():
            continue
        for advertised in advertised_roots:
            if explicit.parent != advertised:
                continue
            match = re.fullmatch(r"\d{4}-\d{2}-\d{2}_(.+)", explicit.name)
            if match is not None and match.group(1) in task_slugs:
                verified.append(explicit)
                break
    return tuple(verified)


def _require_quiescent(
    dataset_root: Path | None = None,
    *,
    allow_disjoint_active_collection: bool = False,
    active_collection_dataset_roots: tuple[Path, ...] = (),
) -> None:
    active = _active_collection_processes()
    if not active:
        return
    if not allow_disjoint_active_collection or dataset_root is None:
        raise ModalityStorageError(
            "formal collection is active: " + "; ".join(active[:3])
        )
    root = dataset_root.resolve(strict=True)
    unresolved: list[str] = []
    overlapping: list[str] = []
    for command in active:
        active_roots = _active_collection_dataset_roots(command)
        if not active_roots:
            unresolved.append(command)
            continue
        if not any(_paths_overlap(root, active_root) for active_root in active_roots):
            continue
        narrowed_roots = _verified_explicit_active_dataset_roots(
            command,
            advertised_roots=active_roots,
            explicit_roots=active_collection_dataset_roots,
        )
        if not narrowed_roots:
            overlapping.append(command)
        elif any(_paths_overlap(root, active_root) for active_root in narrowed_roots):
            overlapping.append(command)
    if unresolved or overlapping:
        details = overlapping + unresolved
        raise ModalityStorageError(
            "cannot prove active formal collection is disjoint from export source: "
            + "; ".join(details[:3])
        )


@contextmanager
def _maintenance_lock(dataset_root: Path) -> Iterator[None]:
    lock_path = dataset_root.parent / f".{dataset_root.name}.modality-maintenance.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ModalityStorageError(
                f"another modality maintenance operation holds {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _cold_transfer_lock(
    artifact_root: Path,
    *,
    staging_namespace: str = "",
) -> Iterator[None]:
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    namespace = _normalize_cold_staging_namespace(staging_namespace)
    suffix = f".transfer.{namespace}.lock" if namespace else ".transfer.lock"
    lock_path = artifact_root.parent / f".{artifact_root.name}{suffix}"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ModalityStorageError(
                f"another cold transfer holds {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _partial_export_reusable_bytes(
    stage: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    revalidation_profile: str = FULL_REVALIDATION,
) -> int:
    """Validate one deterministic partial tree and report allocated bytes."""

    profile = _normalize_revalidation_profile(revalidation_profile)
    if not stage.exists():
        return 0
    if stage.is_symlink() or not stage.is_dir():
        raise ModalityStorageError(f"cold partial path is unsafe: {stage}")
    inventory_path = stage / "inventory.json"
    entries = tuple(stage.rglob("*"))
    if not inventory_path.is_file():
        if entries:
            raise ModalityStorageError(
                f"cold partial tree has no inventory binding: {stage}"
            )
        return 0
    bound = ModalityArtifactInventoryV1.from_dict(_read_json(inventory_path))
    if bound != inventory:
        raise ModalityStorageError(
            f"cold partial tree belongs to another inventory: {stage}"
        )
    expected: dict[str, Any] = {
        artifact.logical_path: artifact for artifact in inventory.artifacts
    }
    sampled_hash_paths = (
        {
            artifact.logical_path
            for artifact in inventory.artifacts
            if artifact.role == "manifest"
        }
        if profile == SAMPLED_REVALIDATION_V1
        else set(expected)
    )
    allowed = {"inventory.json"}
    for logical_path in expected:
        target = stage / logical_path
        allowed.add(logical_path)
        allowed.add(
            _partial_file_path(target).relative_to(stage).as_posix()
        )
    actual_files: set[str] = set()
    for path in entries:
        if path.is_symlink():
            raise ModalityStorageError(f"cold partial tree contains symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(stage).as_posix())
    extra = sorted(actual_files - allowed)
    if extra:
        raise ModalityStorageError(
            f"cold partial tree contains unowned files: {extra[:5]}"
        )
    reusable = 0
    for logical_path, artifact in expected.items():
        target = stage / logical_path
        partial = _partial_file_path(target)
        if target.exists() and partial.exists():
            raise ModalityStorageError(
                f"cold partial tree contains both completed and partial files: {target}"
            )
        candidate = target if target.exists() else partial
        if not candidate.exists():
            continue
        stat = _assert_regular_single_link(
            candidate,
            label="resumable cold artifact",
        )
        if int(stat.st_size) > int(artifact.size_bytes):
            raise ModalityStorageError(
                f"resumable cold artifact exceeds inventory size: {candidate}"
            )
        if candidate == target:
            if int(stat.st_size) != int(artifact.size_bytes):
                raise ModalityStorageError(
                    f"completed cold artifact has a short size: {candidate}"
                )
            if (
                logical_path in sampled_hash_paths
                and _sha256_file(candidate) != str(artifact.sha256)
            ):
                raise ModalityStorageError(
                    f"completed cold artifact differs from inventory: {candidate}"
                )
        reusable += int(stat.st_size)
    return reusable


def _prepare_partial_export_stage(
    stage: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    revalidation_profile: str = FULL_REVALIDATION,
) -> None:
    if not stage.exists():
        stage.mkdir(parents=True)
    inventory_path = stage / "inventory.json"
    if not inventory_path.exists():
        if any(stage.iterdir()):
            raise ModalityStorageError(
                f"cold partial tree has no inventory binding: {stage}"
            )
        _atomic_write_json(inventory_path, inventory.to_dict())
    _partial_export_reusable_bytes(
        stage,
        inventory,
        revalidation_profile=revalidation_profile,
    )


def _payload_summary(
    inventory: ModalityArtifactInventoryV1,
) -> tuple[int, int]:
    payloads = tuple(item for item in inventory.artifacts if item.role == "payload")
    return len(payloads), sum(item.size_bytes for item in payloads)


def _normalize_detachment_scope(value: str) -> str:
    scope = str(value or "").strip().lower().replace("-", "_")
    if scope not in DETACHMENT_SCOPES:
        raise ModalityStorageError(f"unsupported detachment scope: {value!r}")
    return scope


def _detachment_scope_for_inventory(
    dataset_root: Path,
    inventory_hash: str,
) -> str:
    return (
        verified_detachment_scope(dataset_root, inventory_hash)
        or PAYLOAD_ONLY_DETACH
    )


def _stored_inventory_availability(
    dataset_root: Path,
    inventory: ModalityArtifactInventoryV1,
):
    return resolve_modality_availability(
        dataset_root,
        inventory,
        detachment_scope=_detachment_scope_for_inventory(
            dataset_root,
            inventory.semantic_content_hash,
        ),
    )


def _manifest_groups(
    inventory: ModalityArtifactInventoryV1,
) -> tuple[tuple[Any, ...], ...]:
    groups: list[tuple[Any, ...]] = []
    payloads = tuple(item for item in inventory.artifacts if item.role == "payload")
    for manifest in (item for item in inventory.artifacts if item.role == "manifest"):
        group = (manifest,) + tuple(
            payload
            for payload in payloads
            if payload.episode_index == manifest.episode_index
            and payload.side == manifest.side
        )
        groups.append(tuple(sorted(group, key=lambda item: item.logical_path)))
    return tuple(groups)


def _normalize_revalidation_profile(value: str) -> str:
    profile = str(value or "").strip().lower()
    if profile not in REVALIDATION_PROFILES:
        raise ModalityStorageError(
            f"unsupported revalidation profile: {value!r}"
        )
    return profile


def _sampled_manifest_groups(
    inventory: ModalityArtifactInventoryV1,
    *,
    limit: int = SAMPLED_REVALIDATION_MANIFEST_COUNT,
) -> tuple[tuple[Any, ...], ...]:
    """Select deterministic, side-balanced restore/read groups.

    Every artifact byte remains covered by the exact copy hash verification.
    This selection only bounds expensive tail-frame decode and restore drills.
    """

    groups = _manifest_groups(inventory)
    if limit <= 0 or len(groups) <= limit:
        return groups
    buckets: dict[str, list[tuple[Any, ...]]] = {}
    for group in groups:
        manifest = next(item for item in group if item.role == "manifest")
        buckets.setdefault(str(manifest.side or ""), []).append(group)
    selected: dict[str, tuple[Any, ...]] = {}
    bucket_items = sorted(buckets.items())
    if len(bucket_items) > limit:
        raise ModalityStorageError(
            "sample limit cannot cover every tactile side: "
            f"sides={len(bucket_items)} limit={limit}"
        )
    base = max(
        1,
        (limit - len(bucket_items)) // max(1, len(bucket_items)),
    )
    for _side, bucket in bucket_items:
        ordered = sorted(
            bucket,
            key=lambda group: next(
                item.logical_path for item in group if item.role == "manifest"
            ),
        )
        take = min(len(ordered), base)
        if take == 1:
            indexes = (0,)
        else:
            indexes = tuple(
                round(position * (len(ordered) - 1) / (take - 1))
                for position in range(take)
            )
        largest = max(
            ordered,
            key=lambda group: sum(item.size_bytes for item in group),
        )
        for group in (ordered[index] for index in indexes):
            manifest_path = next(
                item.logical_path for item in group if item.role == "manifest"
            )
            selected[manifest_path] = group
        largest_manifest = next(
            item.logical_path for item in largest if item.role == "manifest"
        )
        selected[largest_manifest] = largest
    if len(selected) < limit:
        ranked = sorted(
            groups,
            key=lambda group: hashlib.sha256(
                (
                    inventory.semantic_content_hash
                    + ":"
                    + next(
                        item.logical_path
                        for item in group
                        if item.role == "manifest"
                    )
                ).encode("utf-8")
            ).hexdigest(),
        )
        for group in ranked:
            manifest_path = next(
                item.logical_path for item in group if item.role == "manifest"
            )
            selected.setdefault(manifest_path, group)
            if len(selected) >= limit:
                break
    if len(selected) > limit:
        # The normal left/right layout cannot reach this branch, but preserve
        # the per-side largest group if a future layout adds more side keys.
        required = {
            next(
                item.logical_path
                for item in max(
                    bucket,
                    key=lambda group: sum(item.size_bytes for item in group),
                )
                if item.role == "manifest"
            )
            for _side, bucket in bucket_items
        }
        keep = list(sorted(required))
        keep.extend(
            path
            for path in sorted(selected)
            if path not in required
        )
        selected = {path: selected[path] for path in keep[:limit]}
    return tuple(selected[path] for path in sorted(selected))


def _revalidation_groups(
    inventory: ModalityArtifactInventoryV1,
    profile: str,
) -> tuple[tuple[Any, ...], ...]:
    normalized = _normalize_revalidation_profile(profile)
    if normalized == FULL_REVALIDATION:
        return _manifest_groups(inventory)
    return _sampled_manifest_groups(inventory)


def _artifact_stat_snapshot(
    root: Path,
    inventory: ModalityArtifactInventoryV1,
) -> tuple[tuple[object, ...], ...]:
    snapshot: list[tuple[object, ...]] = []
    for artifact in inventory.artifacts:
        path = root / artifact.logical_path
        if path.is_symlink() or not path.is_file():
            raise ModalityStorageError(
                f"source artifact is unsafe or absent: {artifact.logical_path}"
            )
        stat = path.stat()
        snapshot.append(
            (
                artifact.logical_path,
                int(stat.st_dev),
                int(stat.st_ino),
                int(stat.st_nlink),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
            )
        )
    return tuple(snapshot)


def _restore_drill_peak_bytes(inventory: ModalityArtifactInventoryV1) -> int:
    return max(
        (sum(item.size_bytes for item in group) for group in _manifest_groups(inventory)),
        default=0,
    )


def _source_artifact_link_errors(
    dataset_root: Path,
    inventory: ModalityArtifactInventoryV1,
) -> tuple[str, ...]:
    """Return ownership errors that make whole-tree retirement ambiguous."""

    hardlinked: list[tuple[str, int]] = []
    unsafe: list[str] = []
    for artifact in inventory.artifacts:
        path = dataset_root / artifact.logical_path
        if path.is_symlink() or not path.is_file():
            unsafe.append(artifact.logical_path)
            continue
        link_count = int(path.stat().st_nlink)
        if link_count != 1:
            hardlinked.append((artifact.logical_path, link_count))
    errors: list[str] = []
    if unsafe:
        errors.append(
            "source tactile inventory contains unsafe/non-regular artifacts: "
            f"{unsafe[:5]}"
        )
    if hardlinked:
        errors.append(
            "source tactile inventory contains hardlinked artifacts; managed move "
            "requires independently owned source inodes: "
            f"count={len(hardlinked)} max_nlink="
            f"{max(link_count for _, link_count in hardlinked)} "
            f"examples={[path for path, _ in hardlinked[:5]]}"
        )
    return tuple(errors)


def _plan(
    operation: str,
    dataset_root: str | Path,
    cold_root: str | Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    detachment_scope: str,
    require_distinct_device: bool,
    min_free_bytes: int,
    source_validation_mode: str = "tactile_inventory",
    cold_staging_namespace: str = "",
    revalidation_profile: str = FULL_REVALIDATION,
) -> ModalityStoragePlan:
    root = _safe_resolved(dataset_root)
    cold = _safe_resolved(cold_root)
    scope = _normalize_detachment_scope(detachment_scope)
    profile = _normalize_revalidation_profile(revalidation_profile)
    errors: list[str] = []
    try:
        _validate_roots(root, cold)
    except ModalityStorageError as exc:
        errors.append(str(exc))
    cold_parent = _nearest_existing_parent(cold)
    hot_device = int(root.stat().st_dev)
    cold_device = int(cold_parent.stat().st_dev)
    source_is_hot = operation in {"detach", "export"}
    source_device = hot_device if source_is_hot else cold_device
    target_device = cold_device if source_is_hot else hot_device
    if require_distinct_device and source_device == target_device:
        errors.append("cold storage must be on a distinct filesystem/device")
    payload_count, payload_bytes = _payload_summary(inventory)
    drill_peak_bytes = _restore_drill_peak_bytes(inventory)
    artifact_root = _cold_artifact_root(cold, inventory)
    availability = (
        SELF_CONTAINED
        if operation == "export"
        or (operation == "detach" and source_validation_mode.startswith("legacy_"))
        else resolve_modality_availability(
            root,
            inventory,
            detachment_scope=scope,
        ).status
    )
    required_bytes = (
        inventory.total_bytes
        if source_is_hot or scope == WHOLE_PREFIX_DETACH
        else payload_bytes
    )
    reusable_export_bytes = 0
    if source_is_hot:
        if operation == "detach" and not source_validation_mode.startswith("legacy_"):
            errors.extend(_source_artifact_link_errors(root, inventory))
        partial_root = _cold_partial_root(
            artifact_root,
            staging_namespace=cold_staging_namespace,
        )
        try:
            if artifact_root.exists():
                cold_inventory = ModalityArtifactInventoryV1.from_dict(
                    _read_json(artifact_root / "inventory.json")
                )
                if cold_inventory != inventory:
                    raise ModalityStorageError(
                        "cold target contains another inventory"
                    )
                if profile == FULL_REVALIDATION:
                    verify_modality_artifact_inventory(
                        root,
                        inventory,
                        artifact_root=artifact_root,
                    )
                else:
                    _verify_sampled_inventory(root, artifact_root, inventory)
                _exact_artifact_tree(artifact_root, inventory)
                reusable_export_bytes = inventory.total_bytes
            else:
                reusable_export_bytes = _partial_export_reusable_bytes(
                    partial_root,
                    inventory,
                    revalidation_profile=profile,
                )
        except Exception as exc:
            errors.append(f"cold artifact root is not reusable: {exc}")
    required_bytes = max(0, int(required_bytes) - int(reusable_export_bytes))
    capacity_root = cold_parent if source_is_hot else root
    if shutil.disk_usage(capacity_root).free < required_bytes + int(min_free_bytes):
        errors.append("target filesystem has insufficient free space")
    if operation == "detach" and shutil.disk_usage(root).free < drill_peak_bytes + int(
        min_free_bytes
    ):
        errors.append("hot filesystem lacks space for one restore-drill group")
    pending = _journal_path(root)
    if pending.exists():
        errors.append(f"unresolved modality transaction: {pending}")
    if source_is_hot and availability != SELF_CONTAINED:
        errors.append(f"{operation} requires SELF_CONTAINED, got {availability}")
    if operation == "rehydrate" and availability != DETACHED_VERIFIED:
        errors.append(f"rehydrate requires DETACHED_VERIFIED, got {availability}")
    try:
        if source_is_hot:
            _verify_tactile_reader(
                root,
                inventory,
                revalidation_profile=profile,
            )
    except Exception as exc:
        errors.append(str(exc))
    if artifact_root.exists() and not source_is_hot:
        try:
            cold_inventory = ModalityArtifactInventoryV1.from_dict(
                _read_json(artifact_root / "inventory.json")
            )
            if cold_inventory != inventory:
                raise ModalityStorageError("cold target contains another inventory")
            verify_modality_artifact_inventory(
                root,
                inventory,
                artifact_root=artifact_root,
            )
            _exact_artifact_tree(artifact_root, inventory)
            _verify_tactile_reader(artifact_root, inventory)
        except Exception as exc:
            errors.append(f"cold artifact root is not reusable: {exc}")
    return ModalityStoragePlan(
        operation=operation,
        detachment_scope=scope,
        dataset_root=str(root),
        cold_root=str(cold),
        cold_artifact_root=str(artifact_root),
        inventory_hash=inventory.semantic_content_hash,
        dataset_fingerprint=inventory.source_dataset_base_fingerprint,
        episode_catalog_sha256=inventory.source_episode_catalog_sha256,
        source_device=source_device,
        target_device=target_device,
        require_distinct_device=bool(require_distinct_device),
        file_count=inventory.file_count,
        total_bytes=inventory.total_bytes,
        payload_file_count=payload_count,
        payload_bytes=payload_bytes,
        restore_drill_peak_bytes=drill_peak_bytes,
        source_validation_mode=str(source_validation_mode),
        availability=availability,
        min_free_bytes=int(min_free_bytes),
        ready=not errors,
        revalidation_profile=profile,
        revalidation_manifest_count=len(
            _revalidation_groups(inventory, profile)
        ),
        errors=tuple(errors),
    )


def plan_modality_detach(
    dataset_root: str | Path,
    cold_root: str | Path,
    *,
    detachment_scope: str = PAYLOAD_ONLY_DETACH,
    require_distinct_device: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    allow_legacy_hardlinked_source: bool = False,
    legacy_scan_payload_hashes: bool = False,
    revalidation_profile: str = FULL_REVALIDATION,
) -> ModalityStoragePlan:
    if legacy_scan_payload_hashes and not allow_legacy_hardlinked_source:
        raise ModalityStorageError(
            "legacy payload hash scan requires explicit legacy hardlink detach"
        )
    root = _safe_resolved(dataset_root)
    try:
        inventory = build_modality_artifact_inventory(
            root,
            payload_hash_mode=(
                PAYLOAD_HASH_SCAN
                if legacy_scan_payload_hashes
                else PAYLOAD_HASH_PREFER_SEALED
            ),
            require_single_link=not bool(allow_legacy_hardlinked_source),
        )
    except ModalityArtifactInventoryError as exc:
        raise ModalityStorageError(
            f"detach source is not inventory-safe: {exc}"
        ) from exc
    return _plan(
        "detach",
        root,
        cold_root,
        inventory,
        detachment_scope=detachment_scope,
        require_distinct_device=require_distinct_device,
        min_free_bytes=min_free_bytes,
        source_validation_mode=(
            "legacy_scan_inventory"
            if legacy_scan_payload_hashes
            else "legacy_inventory"
            if allow_legacy_hardlinked_source
            else "tactile_inventory"
        ),
        revalidation_profile=revalidation_profile,
    )


def plan_modality_export(
    dataset_root: str | Path,
    cold_root: str | Path,
    *,
    require_distinct_device: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    allow_legacy_source: bool = False,
    legacy_scan_payload_hashes: bool = False,
    cold_staging_namespace: str = "",
) -> ModalityStoragePlan:
    """Plan a source-preserving verified export of the tactile artifact graph.

    Unlike detach/move, export may read historical source files that share an
    inode through hardlinks.  The target is always copied to independent files
    and the hot dataset remains self-contained.
    """

    if legacy_scan_payload_hashes and not allow_legacy_source:
        raise ModalityStorageError(
            "legacy payload hash scan requires explicit legacy source mode"
        )
    root = _safe_resolved(dataset_root)
    try:
        inventory = build_modality_artifact_inventory(
            root,
            payload_hash_mode=(
                PAYLOAD_HASH_SCAN
                if legacy_scan_payload_hashes
                else PAYLOAD_HASH_PREFER_SEALED
            ),
            require_single_link=False,
        )
    except ModalityArtifactInventoryError as exc:
        raise ModalityStorageError(
            f"export source is not inventory-safe: {exc}"
        ) from exc
    return _plan(
        "export",
        root,
        cold_root,
        inventory,
        detachment_scope=PAYLOAD_ONLY_DETACH,
        require_distinct_device=require_distinct_device,
        min_free_bytes=min_free_bytes,
        source_validation_mode=(
            "legacy_scan_inventory"
            if legacy_scan_payload_hashes
            else "legacy_inventory"
            if allow_legacy_source
            else "tactile_inventory"
        ),
        cold_staging_namespace=cold_staging_namespace,
    )


def plan_modality_rehydrate(
    dataset_root: str | Path,
    cold_root: str | Path,
    inventory_hash: str,
    *,
    require_distinct_device: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> ModalityStoragePlan:
    root = _safe_resolved(dataset_root)
    inventory = load_modality_inventory(root, inventory_hash)
    detachment_scope = _detachment_scope_for_inventory(
        root,
        inventory.semantic_content_hash,
    )
    return _plan(
        "rehydrate",
        root,
        cold_root,
        inventory,
        detachment_scope=detachment_scope,
        require_distinct_device=require_distinct_device,
        min_free_bytes=min_free_bytes,
    )


def _assert_confirmation(
    plan: ModalityStoragePlan,
    *,
    expected_plan_hash: str,
    confirm_dataset_fingerprint: str,
) -> None:
    if not plan.ready:
        raise ModalityStorageError("plan is not ready: " + "; ".join(plan.errors))
    if plan.plan_hash != str(expected_plan_hash):
        raise ModalityStorageError("plan hash changed; generate a fresh proposal")
    if plan.dataset_fingerprint != str(confirm_dataset_fingerprint):
        raise ModalityStorageError("source binding confirmation mismatch")


def _verify_independent_copy(
    dataset_root: Path,
    artifact_root: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    require_distinct_device: bool,
) -> None:
    verify_modality_artifact_inventory(
        dataset_root,
        inventory,
        artifact_root=artifact_root,
    )
    _exact_artifact_tree(artifact_root, inventory)
    for artifact in inventory.artifacts:
        source_stat = (dataset_root / artifact.logical_path).stat()
        target_stat = (artifact_root / artifact.logical_path).stat()
        if source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino:
            raise ModalityStorageError(
                f"cold artifact shares source inode: {artifact.logical_path}"
            )
        if require_distinct_device and source_stat.st_dev == target_stat.st_dev:
            raise ModalityStorageError(
                f"cold artifact is on source device: {artifact.logical_path}"
            )


def _verify_sampled_inventory(
    dataset_root: Path,
    artifact_root: Path,
    inventory: ModalityArtifactInventoryV1,
) -> None:
    """Verify all metadata/manifests and deterministic payload samples.

    This is a planning/revalidation optimization only. The apply path still
    calls :func:`_verify_independent_copy`, which hashes every cold artifact.
    """

    if not inventory_source_binding_matches(dataset_root, inventory):
        raise ModalityStorageError("source modality reference binding changed")
    sampled_payloads = {
        item.logical_path
        for group in _sampled_manifest_groups(inventory)
        for item in group
        if item.role == "payload"
    }
    for artifact in inventory.artifacts:
        path = artifact_root / artifact.logical_path
        if path.is_symlink() or not path.is_file():
            raise ModalityStorageError(
                f"inventory artifact is unsafe or absent: {artifact.logical_path}"
            )
        if int(path.stat().st_size) != int(artifact.size_bytes):
            raise ModalityStorageError(
                f"artifact size changed: {artifact.logical_path}"
            )
        if artifact.role == "manifest" or artifact.logical_path in sampled_payloads:
            if _sha256_file(path) != artifact.sha256:
                raise ModalityStorageError(
                    f"artifact hash changed: {artifact.logical_path}"
                )


def _verify_tactile_reader(
    artifact_root: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    revalidation_profile: str = FULL_REVALIDATION,
) -> None:
    decode_manifests = {
        next(item.logical_path for item in group if item.role == "manifest")
        for group in _revalidation_groups(inventory, revalidation_profile)
    }
    payload_paths = {
        artifact.logical_path
        for artifact in inventory.artifacts
        if artifact.role == "payload"
    }
    for artifact in inventory.artifacts:
        if artifact.role != "manifest":
            continue
        manifest = _read_json(artifact_root / artifact.logical_path)
        streams = manifest.get("streams")
        if not isinstance(streams, Mapping) or not streams:
            raise ModalityStorageError(
                f"tactile manifest has no streams: {artifact.logical_path}"
            )
        for sensor_key, raw_spec in streams.items():
            if not isinstance(raw_spec, Mapping):
                raise ModalityStorageError(
                    f"invalid tactile stream {sensor_key!r}: {artifact.logical_path}"
                )
            spec = dict(raw_spec)
            relative = str(spec.get("path") or "")
            if relative not in payload_paths:
                raise ModalityStorageError(
                    f"stream is outside inventory: {artifact.logical_path}:{sensor_key}"
                )
            frame_count = int(spec.get("frame_count") or 0)
            shape = tuple(int(value) for value in spec.get("shape", ()) or ())
            dtype = np.dtype(str(spec.get("dtype") or ""))
            if frame_count < 0 or not shape or any(value <= 0 for value in shape):
                raise ModalityStorageError(
                    f"invalid tactile stream shape/count: {artifact.logical_path}:{sensor_key}"
                )
            if tactile_stream_compression(spec) == "none":
                expected = frame_count * math.prod(shape) * dtype.itemsize
                if (artifact_root / relative).stat().st_size != expected:
                    raise ModalityStorageError(
                        f"raw tactile byte count mismatch: {artifact.logical_path}:{sensor_key}"
                    )
                continue
            if artifact.logical_path not in decode_manifests:
                continue
            reader = ChunkedTactileStream(dataset_root=artifact_root, spec=spec)
            if frame_count:
                if reader.frame(0) is None or reader.frame(frame_count - 1) is None:
                    raise ModalityStorageError(
                        f"tactile reader failed first/tail decode: {artifact.logical_path}:{sensor_key}"
                    )


def _export_shadow(
    dataset_root: Path,
    artifact_root: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    require_distinct_device: bool,
    source_storage_id: str,
    target_storage_id: str,
    allow_hardlinked_source: bool = False,
    publish_source_receipt: bool = True,
    cold_staging_namespace: str = "",
    revalidation_profile: str = FULL_REVALIDATION,
) -> Path:
    profile = _normalize_revalidation_profile(revalidation_profile)
    with _cold_transfer_lock(
        artifact_root,
        staging_namespace=cold_staging_namespace,
    ):
        stage = _cold_partial_root(
            artifact_root,
            staging_namespace=cold_staging_namespace,
        )
        if artifact_root.exists():
            cold_inventory = ModalityArtifactInventoryV1.from_dict(
                _read_json(artifact_root / "inventory.json")
            )
            if cold_inventory != inventory:
                raise ModalityStorageError(
                    f"cold target contains another inventory: {artifact_root}"
                )
            _verify_independent_copy(
                dataset_root,
                artifact_root,
                inventory,
                require_distinct_device=require_distinct_device,
            )
            if stage.exists():
                _partial_export_reusable_bytes(
                    stage,
                    inventory,
                    revalidation_profile=profile,
                )
                shutil.rmtree(stage)
                _fsync_directory(stage.parent)
        else:
            _prepare_partial_export_stage(
                stage,
                inventory,
                revalidation_profile=profile,
            )
            _copy_inventory_artifacts_resumable(
                dataset_root,
                stage,
                inventory,
                allow_hardlinked_source=bool(allow_hardlinked_source),
            )
            _verify_independent_copy(
                dataset_root,
                stage,
                inventory,
                require_distinct_device=require_distinct_device,
            )
            _fsync_tree(stage)
            _replace_with_retry(stage, artifact_root)
            _fsync_directory(artifact_root.parent)
    cold_receipt_path = artifact_root / "export_receipt.json"
    if not publish_source_receipt and cold_receipt_path.exists():
        existing = ModalityStorageTransferReceiptV1.from_dict(
            _read_json(cold_receipt_path)
        )
        if (
            existing.operation != "export_shadow"
            or existing.inventory_hash != inventory.semantic_content_hash
            or existing.source_storage_id != source_storage_id
            or existing.target_storage_id != target_storage_id
            or existing.file_count != inventory.file_count
            or existing.total_bytes != inventory.total_bytes
            or existing.status != "complete"
            or not existing.independent_copy_verified
        ):
            raise ModalityStorageError(
                "cold export receipt differs from frozen inventory: "
                f"{cold_receipt_path}"
            )
        return cold_receipt_path
    receipt = _receipt(
        operation="export_shadow",
        inventory=inventory,
        source_storage_id=source_storage_id,
        target_storage_id=target_storage_id,
        file_count=inventory.file_count,
        total_bytes=inventory.total_bytes,
        independent_copy_verified=True,
        operation_id=(
            None
            if publish_source_receipt
            else f"export_shadow.{inventory.semantic_content_hash}"
        ),
    )
    if publish_source_receipt:
        return _publish_receipt(dataset_root, receipt)
    _atomic_write_json(cold_receipt_path, receipt.to_dict())
    return cold_receipt_path


def _restore_drill(
    dataset_root: Path,
    artifact_root: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    source_storage_id: str,
    target_storage_id: str,
    revalidation_profile: str = FULL_REVALIDATION,
) -> Path:
    profile = _normalize_revalidation_profile(revalidation_profile)
    drill = Path(
        tempfile.mkdtemp(
            prefix=f".{dataset_root.name}.tactile-restore-drill.",
            dir=dataset_root.parent,
        )
    )
    try:
        for group in _revalidation_groups(inventory, profile):
            group_inventory = ModalityArtifactInventoryV1(
                modality_key=inventory.modality_key,
                source_dataset_base_fingerprint=(
                    inventory.source_dataset_base_fingerprint
                ),
                source_episode_catalog_sha256=(
                    inventory.source_episode_catalog_sha256
                ),
                source_episode_count=inventory.source_episode_count,
                logical_prefixes=inventory.logical_prefixes,
                artifacts=tuple(group),
                source_binding_profile=inventory.source_binding_profile,
            )
            _copy_inventory_artifacts(artifact_root, drill, group_inventory)
            verify_modality_artifact_inventory(
                dataset_root,
                group_inventory,
                artifact_root=drill,
            )
            _exact_artifact_tree(drill, group_inventory)
            _verify_tactile_reader(drill, group_inventory)
            tactile = drill / "tactile"
            if tactile.exists():
                shutil.rmtree(tactile)
    finally:
        shutil.rmtree(drill, ignore_errors=True)
    receipt = _receipt(
        operation="restore_drill",
        operation_id=(
            None
            if profile == FULL_REVALIDATION
            else f"restore_drill.{profile}.{uuid.uuid4().hex}"
        ),
        inventory=inventory,
        source_storage_id=source_storage_id,
        target_storage_id=target_storage_id,
        file_count=inventory.file_count,
        total_bytes=inventory.total_bytes,
        independent_copy_verified=True,
    )
    return _publish_receipt(dataset_root, receipt)


def _swap_paths(dataset_root: Path, operation_id: str) -> tuple[Path, Path]:
    base = dataset_root.parent
    stage = base / f".{dataset_root.name}.{operation_id}.stage"
    quarantine = base / f".{dataset_root.name}.{operation_id}.quarantine"
    if stage.exists() or quarantine.exists():
        raise ModalityStorageError("transaction staging paths already exist")
    return stage, quarantine


def _write_journal(
    dataset_root: Path,
    *,
    operation_id: str,
    operation: str,
    inventory_hash: str,
    stage: Path,
    quarantine: Path,
    active_was_present: bool = True,
    artifact_root: Path | None = None,
) -> None:
    _atomic_write_json(
        _journal_path(dataset_root),
        {
            "schema": "dexteleop.modality_storage_transaction.v1",
            "operation_id": operation_id,
            "operation": operation,
            "inventory_hash": inventory_hash,
            "stage": str(stage),
            "quarantine": str(quarantine),
            "active_was_present": bool(active_was_present),
            "artifact_root": (
                str(_safe_resolved(artifact_root))
                if artifact_root is not None
                else None
            ),
        },
    )


def _rollback_swap(
    dataset_root: Path,
    stage: Path,
    quarantine: Path,
    *,
    active_was_present: bool = True,
) -> None:
    active = dataset_root / "tactile"
    displaced = stage.parent / f".{stage.name}.rollback-displaced"
    if active_was_present and not quarantine.exists() and not active.exists():
        raise ModalityStorageError(
            "cannot rollback modality transaction: both active and quarantine "
            "trees are absent"
        )
    for label, candidate in (
        ("active tactile tree", active),
        ("transaction quarantine", quarantine),
        ("transaction stage", stage),
        ("rollback displaced tree", displaced),
    ):
        if candidate.is_symlink() or (
            candidate.exists() and not candidate.is_dir()
        ):
            raise ModalityStorageError(f"{label} is not a safe directory: {candidate}")
    if active_was_present and quarantine.exists():
        if active.exists():
            if displaced.exists():
                shutil.rmtree(displaced)
            os.replace(active, displaced)
        os.replace(quarantine, active)
        _fsync_directory(dataset_root)
        _fsync_directory(dataset_root.parent)
    elif not active_was_present and active.exists():
        if displaced.exists():
            shutil.rmtree(displaced)
        os.replace(active, displaced)
        _fsync_directory(dataset_root)
        _fsync_directory(dataset_root.parent)
    if displaced.exists():
        shutil.rmtree(displaced)
    if stage.exists():
        shutil.rmtree(stage)
    _fsync_directory(dataset_root.parent)
    _journal_path(dataset_root).unlink(missing_ok=True)
    _fsync_directory(_journal_path(dataset_root).parent)


def recover_modality_storage_transaction(dataset_root: str | Path) -> bool:
    """Rollback an unreceipted directory swap; apply-mode recovery only."""

    root = _safe_resolved(dataset_root)
    journal_path = _journal_path(root)
    if not journal_path.is_file():
        return False
    journal = _read_json(journal_path)
    if journal.get("schema") != "dexteleop.modality_storage_transaction.v1":
        raise ModalityStorageError("transaction journal has an unsupported schema")
    operation_id = str(journal.get("operation_id") or "")
    operation = str(journal.get("operation") or "")
    if operation not in {
        "retire_hot_payloads",
        "retire_hot_module_tree",
        "rehydrate",
    }:
        raise ModalityStorageError("transaction journal has an unsupported operation")
    operation_suffix = operation_id.removeprefix(f"{operation}.")
    if (
        not operation_id.startswith(f"{operation}.")
        or not operation_suffix
        or not all(character.isalnum() or character in {"_", "-"} for character in operation_suffix)
    ):
        raise ModalityStorageError("transaction journal has an unsafe operation_id")
    stage = _safe_resolved(str(journal.get("stage") or ""))
    quarantine = _safe_resolved(str(journal.get("quarantine") or ""))
    expected_parent = root.parent
    expected_stage = expected_parent / f".{root.name}.{operation_id}.stage"
    expected_quarantine = expected_parent / f".{root.name}.{operation_id}.quarantine"
    if stage != expected_stage or quarantine != expected_quarantine:
        raise ModalityStorageError(
            "transaction journal paths do not match the dataset and operation_id"
        )
    active_was_present = bool(journal.get("active_was_present", True))
    inventory_hash = str(journal.get("inventory_hash") or "")
    receipt_path = _receipt_path(root, operation_id)
    if receipt_path.is_file():
        receipt = ModalityStorageTransferReceiptV1.from_dict(_read_json(receipt_path))
        if (
            receipt.operation != operation
            or receipt.status != "complete"
            or receipt.inventory_hash != inventory_hash
            or not receipt.independent_copy_verified
        ):
            raise ModalityStorageError("transaction receipt does not match journal")
        inventory = load_modality_inventory(root, inventory_hash)
        if operation in {"retire_hot_payloads", "retire_hot_module_tree"}:
            artifact_value = journal.get("artifact_root")
            if not isinstance(artifact_value, str) or not artifact_value:
                raise ModalityStorageError(
                    "committed retirement journal lacks its cold artifact root"
                )
            artifact_root = _safe_resolved(artifact_value)
            if len(artifact_root.parents) < 2:
                raise ModalityStorageError(
                    "committed retirement journal has an unsafe cold artifact root"
                )
            cold_root = artifact_root.parents[1]
            if _cold_artifact_root(cold_root, inventory) != artifact_root:
                raise ModalityStorageError(
                    "committed retirement journal cold artifact path does not "
                    "match its inventory"
                )
            _validate_roots(root, cold_root)
            verify_modality_artifact_inventory(
                root,
                inventory,
                artifact_root=artifact_root,
            )
            _exact_artifact_tree(artifact_root, inventory)
            _verify_tactile_reader(artifact_root, inventory)
            if not _has_receipt(root, inventory_hash, "export_shadow") or not _has_receipt(
                root,
                inventory_hash,
                "restore_drill",
            ):
                raise ModalityStorageError(
                    "committed retirement lacks export/restore verification receipts"
                )
            scope = (
                WHOLE_PREFIX_DETACH
                if operation == "retire_hot_module_tree"
                else PAYLOAD_ONLY_DETACH
            )
            availability = resolve_modality_availability(
                root,
                inventory,
                detachment_scope=scope,
            )
            if availability.status != DETACHED_VERIFIED:
                raise ModalityStorageError(
                    "committed retirement hot state is not detached and verified"
                )
        else:
            verify_modality_artifact_inventory(root, inventory)
            _exact_artifact_tree(root, inventory)
            _verify_tactile_reader(root, inventory)
            availability = resolve_modality_availability(root, inventory)
            if availability.status != SELF_CONTAINED:
                raise ModalityStorageError(
                    "committed rehydrate hot state is not self-contained"
                )
        if quarantine.exists():
            shutil.rmtree(quarantine)
        if stage.exists():
            shutil.rmtree(stage)
        _fsync_directory(root.parent)
        journal_path.unlink(missing_ok=True)
        _fsync_directory(journal_path.parent)
        return True
    _rollback_swap(
        root,
        stage,
        quarantine,
        active_was_present=active_was_present,
    )
    return True


def _retire_hot_payloads(
    dataset_root: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    artifact_root: Path,
    source_storage_id: str,
    target_storage_id: str,
    allow_hardlinked_source: bool = False,
    revalidation_profile: str = FULL_REVALIDATION,
) -> Path:
    if not _has_receipt(dataset_root, inventory.semantic_content_hash, "export_shadow"):
        raise ModalityStorageError("retirement requires a successful export receipt")
    if not _has_receipt(dataset_root, inventory.semantic_content_hash, "restore_drill"):
        raise ModalityStorageError("retirement requires a successful restore-drill receipt")
    profile = _normalize_revalidation_profile(revalidation_profile)
    operation_id = (
        f"retire_hot_payloads.{uuid.uuid4().hex}"
        if profile == FULL_REVALIDATION
        else f"retire_hot_payloads.{profile}.{uuid.uuid4().hex}"
    )
    stage, quarantine = _swap_paths(dataset_root, operation_id)
    stage.mkdir()
    committed = False
    try:
        _copy_inventory_artifacts(
            dataset_root,
            stage,
            inventory,
            roles=frozenset({"manifest"}),
            allow_hardlinked_source=allow_hardlinked_source,
        )
        _fsync_tree(stage)
        _write_journal(
            dataset_root,
            operation_id=operation_id,
            operation="retire_hot_payloads",
            inventory_hash=inventory.semantic_content_hash,
            stage=stage,
            quarantine=quarantine,
            artifact_root=artifact_root,
        )
        os.replace(dataset_root / "tactile", quarantine)
        os.replace(stage / "tactile", dataset_root / "tactile")
        _fsync_directory(dataset_root)
        _fsync_directory(dataset_root.parent)
        availability = resolve_modality_availability(dataset_root, inventory)
        if availability.status != DETACHED_VERIFIED:
            raise ModalityStorageError(
                f"retirement produced {availability.status}: {availability.reason}"
            )
        payload_count, payload_bytes = _payload_summary(inventory)
        receipt = _receipt(
            operation="retire_hot_payloads",
            operation_id=operation_id,
            inventory=inventory,
            source_storage_id=source_storage_id,
            target_storage_id=target_storage_id,
            file_count=payload_count,
            total_bytes=payload_bytes,
            independent_copy_verified=True,
        )
        path = _publish_receipt(dataset_root, receipt)
        committed = True
        shutil.rmtree(quarantine)
        shutil.rmtree(stage, ignore_errors=True)
        _fsync_directory(dataset_root.parent)
        _journal_path(dataset_root).unlink(missing_ok=True)
        _fsync_directory(_journal_path(dataset_root).parent)
        return path
    except BaseException:
        if not committed:
            _rollback_swap(dataset_root, stage, quarantine)
        raise


def _retire_hot_module_tree(
    dataset_root: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    artifact_root: Path,
    source_storage_id: str,
    target_storage_id: str,
    revalidation_profile: str = FULL_REVALIDATION,
) -> Path:
    if not _has_receipt(dataset_root, inventory.semantic_content_hash, "export_shadow"):
        raise ModalityStorageError("retirement requires a successful export receipt")
    if not _has_receipt(dataset_root, inventory.semantic_content_hash, "restore_drill"):
        raise ModalityStorageError("retirement requires a successful restore-drill receipt")
    profile = _normalize_revalidation_profile(revalidation_profile)
    operation_id = (
        f"retire_hot_module_tree.{uuid.uuid4().hex}"
        if profile == FULL_REVALIDATION
        else f"retire_hot_module_tree.{profile}.{uuid.uuid4().hex}"
    )
    stage, quarantine = _swap_paths(dataset_root, operation_id)
    stage.mkdir()
    committed = False
    try:
        _write_journal(
            dataset_root,
            operation_id=operation_id,
            operation="retire_hot_module_tree",
            inventory_hash=inventory.semantic_content_hash,
            stage=stage,
            quarantine=quarantine,
            active_was_present=True,
            artifact_root=artifact_root,
        )
        os.replace(dataset_root / "tactile", quarantine)
        _fsync_directory(dataset_root)
        _fsync_directory(dataset_root.parent)
        availability = resolve_modality_availability(
            dataset_root,
            inventory,
            detachment_scope=WHOLE_PREFIX_DETACH,
        )
        if availability.status != DETACHED_VERIFIED:
            raise ModalityStorageError(
                f"whole-tree retirement produced {availability.status}: "
                f"{availability.reason}"
            )
        receipt = _receipt(
            operation="retire_hot_module_tree",
            operation_id=operation_id,
            inventory=inventory,
            source_storage_id=source_storage_id,
            target_storage_id=target_storage_id,
            file_count=inventory.file_count,
            total_bytes=inventory.total_bytes,
            independent_copy_verified=True,
        )
        path = _publish_receipt(dataset_root, receipt)
        committed = True
        shutil.rmtree(quarantine)
        shutil.rmtree(stage, ignore_errors=True)
        _fsync_directory(dataset_root.parent)
        _journal_path(dataset_root).unlink(missing_ok=True)
        _fsync_directory(_journal_path(dataset_root).parent)
        return path
    except BaseException:
        if not committed:
            _rollback_swap(
                dataset_root,
                stage,
                quarantine,
                active_was_present=True,
            )
        raise


def export_modality(
    dataset_root: str | Path,
    cold_root: str | Path,
    *,
    expected_plan_hash: str,
    confirm_dataset_fingerprint: str,
    require_distinct_device: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    source_storage_id: str = "hot.local",
    target_storage_id: str = "cold.archive",
    confirm_no_active_readers: bool = False,
    allow_legacy_source: bool = False,
    legacy_scan_payload_hashes: bool = False,
    allow_disjoint_active_collection: bool = False,
    active_collection_dataset_roots: tuple[str | Path, ...] = (),
    cold_staging_namespace: str = "",
) -> ModalityStorageResult:
    """Copy and verify tactile artifacts without retiring any hot file."""

    root = _safe_resolved(dataset_root)
    if not confirm_no_active_readers:
        raise ModalityStorageError(
            "apply requires explicit confirmation that no dataset readers are active"
        )
    explicit_active_roots = tuple(
        Path(path).expanduser().resolve(strict=False)
        for path in active_collection_dataset_roots
    )
    with _maintenance_lock(root):
        _require_quiescent(
            root,
            allow_disjoint_active_collection=allow_disjoint_active_collection,
            active_collection_dataset_roots=explicit_active_roots,
        )
        recover_modality_storage_transaction(root)
        plan = plan_modality_export(
            root,
            cold_root,
            require_distinct_device=require_distinct_device,
            min_free_bytes=min_free_bytes,
            allow_legacy_source=allow_legacy_source,
            legacy_scan_payload_hashes=legacy_scan_payload_hashes,
            cold_staging_namespace=cold_staging_namespace,
        )
        _assert_confirmation(
            plan,
            expected_plan_hash=expected_plan_hash,
            confirm_dataset_fingerprint=confirm_dataset_fingerprint,
        )
        inventory = build_modality_artifact_inventory(
            root,
            payload_hash_mode=(
                PAYLOAD_HASH_SCAN
                if legacy_scan_payload_hashes
                else PAYLOAD_HASH_PREFER_SEALED
            ),
            require_single_link=False,
        )
        _verify_tactile_reader(root, inventory)
        artifact_root = Path(plan.cold_artifact_root)
        receipt_path = _export_shadow(
            root,
            artifact_root,
            inventory,
            require_distinct_device=require_distinct_device,
            source_storage_id=source_storage_id,
            target_storage_id=target_storage_id,
            allow_hardlinked_source=True,
            publish_source_receipt=False,
            cold_staging_namespace=cold_staging_namespace,
        )
        _verify_tactile_reader(artifact_root, inventory)
        _require_quiescent(
            root,
            allow_disjoint_active_collection=allow_disjoint_active_collection,
            active_collection_dataset_roots=explicit_active_roots,
        )
        current = build_modality_artifact_inventory(
            root,
            payload_hash_mode=(
                PAYLOAD_HASH_SCAN
                if legacy_scan_payload_hashes
                else PAYLOAD_HASH_PREFER_SEALED
            ),
            require_single_link=False,
        )
        if current != inventory:
            raise ModalityStorageError("dataset changed after verified export")
        _verify_tactile_reader(root, current)
        return ModalityStorageResult(
            operation="export",
            dataset_root=str(root),
            cold_artifact_root=str(artifact_root),
            inventory_hash=inventory.semantic_content_hash,
            plan_hash=plan.plan_hash,
            final_status=SELF_CONTAINED,
            receipt_paths=(str(receipt_path),),
            released_hot_bytes=0,
        )


def detach_modality(
    dataset_root: str | Path,
    cold_root: str | Path,
    *,
    expected_plan_hash: str,
    confirm_dataset_fingerprint: str,
    detachment_scope: str = PAYLOAD_ONLY_DETACH,
    require_distinct_device: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    source_storage_id: str = "hot.local",
    target_storage_id: str = "cold.archive",
    confirm_no_active_readers: bool = False,
    allow_legacy_hardlinked_source: bool = False,
    legacy_scan_payload_hashes: bool = False,
    allow_disjoint_active_collection: bool = False,
    active_collection_dataset_roots: tuple[str | Path, ...] = (),
    revalidation_profile: str = FULL_REVALIDATION,
) -> ModalityStorageResult:
    root = _safe_resolved(dataset_root)
    scope = _normalize_detachment_scope(detachment_scope)
    profile = _normalize_revalidation_profile(revalidation_profile)
    if not confirm_no_active_readers:
        raise ModalityStorageError(
            "apply requires explicit confirmation that no dataset readers are active"
        )
    if legacy_scan_payload_hashes and not allow_legacy_hardlinked_source:
        raise ModalityStorageError(
            "legacy payload hash scan requires explicit legacy hardlink detach"
        )
    payload_hash_mode = (
        PAYLOAD_HASH_SCAN
        if legacy_scan_payload_hashes
        else PAYLOAD_HASH_PREFER_SEALED
    )
    explicit_active_roots = tuple(
        Path(path).expanduser().resolve(strict=False)
        for path in active_collection_dataset_roots
    )
    with _maintenance_lock(root):
        _require_quiescent(
            root,
            allow_disjoint_active_collection=allow_disjoint_active_collection,
            active_collection_dataset_roots=explicit_active_roots,
        )
        recover_modality_storage_transaction(root)
        plan = plan_modality_detach(
            root,
            cold_root,
            detachment_scope=scope,
            require_distinct_device=require_distinct_device,
            min_free_bytes=min_free_bytes,
            allow_legacy_hardlinked_source=allow_legacy_hardlinked_source,
            legacy_scan_payload_hashes=legacy_scan_payload_hashes,
            revalidation_profile=profile,
        )
        _assert_confirmation(
            plan,
            expected_plan_hash=expected_plan_hash,
            confirm_dataset_fingerprint=confirm_dataset_fingerprint,
        )
        inventory = build_modality_artifact_inventory(
            root,
            payload_hash_mode=payload_hash_mode,
            require_single_link=not bool(allow_legacy_hardlinked_source),
        )
        _publish_inventory(root, inventory)
        artifact_root = Path(plan.cold_artifact_root)
        receipt_paths = [
            _export_shadow(
                root,
                artifact_root,
                inventory,
                require_distinct_device=require_distinct_device,
                source_storage_id=source_storage_id,
                target_storage_id=target_storage_id,
                allow_hardlinked_source=allow_legacy_hardlinked_source,
                revalidation_profile=profile,
            )
        ]
        source_snapshot = (
            _artifact_stat_snapshot(root, inventory)
            if profile == SAMPLED_REVALIDATION_V1
            else ()
        )
        _require_quiescent(
            root,
            allow_disjoint_active_collection=allow_disjoint_active_collection,
            active_collection_dataset_roots=explicit_active_roots,
        )
        _verify_tactile_reader(
            root,
            inventory,
            revalidation_profile=profile,
        )
        receipt_paths.append(
            _restore_drill(
                root,
                artifact_root,
                inventory,
                source_storage_id=target_storage_id,
                target_storage_id=source_storage_id,
                revalidation_profile=profile,
            )
        )
        _require_quiescent(
            root,
            allow_disjoint_active_collection=allow_disjoint_active_collection,
            active_collection_dataset_roots=explicit_active_roots,
        )
        # Rebuild before the destructive boundary. Only tactile references,
        # manifests, and payload identity are frozen by this plan; unrelated
        # dataset metadata and other modality references may evolve safely.
        if build_modality_artifact_inventory(
            root,
            payload_hash_mode=payload_hash_mode,
            require_single_link=not bool(allow_legacy_hardlinked_source),
        ) != inventory:
            raise ModalityStorageError("dataset changed after shadow verification")
        if source_snapshot and _artifact_stat_snapshot(root, inventory) != source_snapshot:
            raise ModalityStorageError(
                "source tactile artifacts changed after sampled revalidation"
            )
        # The inventory rebuild can be expensive for large tactile payloads.
        # Recheck process quiescence after it, immediately before the directory
        # retirement transaction, so a collector that started during hashing
        # cannot pass only the earlier check.
        _require_quiescent(
            root,
            allow_disjoint_active_collection=allow_disjoint_active_collection,
            active_collection_dataset_roots=explicit_active_roots,
        )
        if scope == WHOLE_PREFIX_DETACH:
            retirement_receipt = _retire_hot_module_tree(
                root,
                inventory,
                artifact_root=artifact_root,
                source_storage_id=source_storage_id,
                target_storage_id=target_storage_id,
                revalidation_profile=profile,
            )
        else:
            retirement_receipt = _retire_hot_payloads(
                root,
                inventory,
                artifact_root=artifact_root,
                source_storage_id=source_storage_id,
                target_storage_id=target_storage_id,
                allow_hardlinked_source=allow_legacy_hardlinked_source,
                revalidation_profile=profile,
            )
        receipt_paths.append(retirement_receipt)
        final = resolve_modality_availability(
            root,
            inventory,
            detachment_scope=scope,
        )
        if final.status != DETACHED_VERIFIED:
            raise ModalityStorageError(f"unexpected final availability: {final.status}")
        _, payload_bytes = _payload_summary(inventory)
        released_hot_bytes = (
            inventory.total_bytes
            if scope == WHOLE_PREFIX_DETACH
            else payload_bytes
        )
        return ModalityStorageResult(
            operation="detach",
            dataset_root=str(root),
            cold_artifact_root=str(artifact_root),
            inventory_hash=inventory.semantic_content_hash,
            plan_hash=plan.plan_hash,
            final_status=final.status,
            receipt_paths=tuple(path.relative_to(root).as_posix() for path in receipt_paths),
            released_hot_bytes=released_hot_bytes,
        )


def _completed_managed_move_result(
    dataset_root: Path,
    cold_root: str | Path,
    *,
    require_distinct_device: bool,
    min_free_bytes: int,
) -> ModalityStorageResult | None:
    """Return a verified result when a whole-prefix move already committed.

    This covers a retry after the retirement receipt became durable but before
    the original caller observed the result.  It is intentionally strict: the
    hot state, persisted receipts, requested cold root, cold inventory bytes,
    and core-only dataset view must all agree before the retry is considered
    successful.
    """

    inventory_directory = _sidecar_root(dataset_root) / "inventories"
    if not inventory_directory.is_dir():
        return None
    detached: list[ModalityArtifactInventoryV1] = []
    for path in sorted(inventory_directory.glob("*.json")):
        inventory = ModalityArtifactInventoryV1.from_dict(_read_json(path))
        scope = _detachment_scope_for_inventory(
            dataset_root,
            inventory.semantic_content_hash,
        )
        availability = resolve_modality_availability(
            dataset_root,
            inventory,
            detachment_scope=scope,
        )
        if availability.status == DETACHED_VERIFIED:
            detached.append(inventory)
    if not detached:
        return None
    if len(detached) != 1:
        raise ModalityStorageError(
            "managed move retry found multiple detached inventories"
        )

    inventory = detached[0]
    scope = _detachment_scope_for_inventory(
        dataset_root,
        inventory.semantic_content_hash,
    )
    if scope != WHOLE_PREFIX_DETACH:
        raise ModalityStorageError(
            "managed move retry found a payload-only detachment"
        )
    required_operations = {
        "export_shadow",
        "restore_drill",
        "retire_hot_module_tree",
    }
    receipts = tuple(
        receipt
        for receipt in _complete_receipts(
            dataset_root,
            inventory.semantic_content_hash,
        )
        if receipt.operation in required_operations
        and receipt.independent_copy_verified
    )
    operations = {receipt.operation for receipt in receipts}
    if operations != required_operations:
        missing = sorted(required_operations - operations)
        raise ModalityStorageError(
            f"managed move retry lacks committed receipts: {missing}"
        )

    cold = _safe_resolved(cold_root)
    _validate_roots(dataset_root, cold)
    cold_parent = _nearest_existing_parent(cold)
    hot_device = int(dataset_root.stat().st_dev)
    cold_device = int(cold_parent.stat().st_dev)
    if require_distinct_device and hot_device == cold_device:
        raise ModalityStorageError(
            "cold storage must be on a distinct filesystem/device"
        )
    artifact_root = _cold_artifact_root(cold, inventory)
    if not artifact_root.is_dir():
        raise ModalityStorageError(
            "managed move already committed, but the requested cold root does "
            f"not contain its artifact: {artifact_root}"
        )
    cold_inventory = ModalityArtifactInventoryV1.from_dict(
        _read_json(artifact_root / "inventory.json")
    )
    if cold_inventory != inventory:
        raise ModalityStorageError(
            f"cold target contains another inventory: {artifact_root}"
        )
    verify_modality_artifact_inventory(
        dataset_root,
        inventory,
        artifact_root=artifact_root,
    )
    _exact_artifact_tree(artifact_root, inventory)
    _verify_tactile_reader(artifact_root, inventory)
    payload_count, payload_bytes = _payload_summary(inventory)
    recovered_plan = ModalityStoragePlan(
        operation="detach",
        detachment_scope=WHOLE_PREFIX_DETACH,
        dataset_root=str(dataset_root),
        cold_root=str(cold),
        cold_artifact_root=str(artifact_root),
        inventory_hash=inventory.semantic_content_hash,
        dataset_fingerprint=inventory.source_dataset_base_fingerprint,
        episode_catalog_sha256=inventory.source_episode_catalog_sha256,
        source_device=hot_device,
        target_device=cold_device,
        require_distinct_device=bool(require_distinct_device),
        file_count=inventory.file_count,
        total_bytes=inventory.total_bytes,
        payload_file_count=payload_count,
        payload_bytes=payload_bytes,
        restore_drill_peak_bytes=_restore_drill_peak_bytes(inventory),
        source_validation_mode="tactile_inventory",
        availability=SELF_CONTAINED,
        min_free_bytes=int(min_free_bytes),
        ready=True,
    )
    receipt_paths = tuple(
        _receipt_path(dataset_root, receipt.operation_id)
        .relative_to(dataset_root)
        .as_posix()
        for receipt in receipts
    )
    return ModalityStorageResult(
        operation="detach",
        dataset_root=str(dataset_root),
        cold_artifact_root=str(artifact_root),
        inventory_hash=inventory.semantic_content_hash,
        plan_hash=recovered_plan.plan_hash,
        final_status=DETACHED_VERIFIED,
        receipt_paths=receipt_paths,
        released_hot_bytes=inventory.total_bytes,
    )


def move_modality(
    dataset_root: str | Path,
    cold_root: str | Path,
    *,
    require_distinct_device: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    source_storage_id: str = "hot.local",
    target_storage_id: str = "cold.archive",
    confirm_no_active_readers: bool = False,
) -> ModalityStorageResult:
    """Managed one-invocation whole-prefix move with resumable cold export.

    This is the safe user-facing equivalent of ``mv tactile/``.  It freezes a
    fresh plan inside the invocation, delegates the same verified detach
    transaction, and never retires the hot tree until the independently copied
    cold tree and restore drill are complete.  Repeating the call after an
    interruption resumes the deterministic cold partial tree.
    """

    root = _safe_resolved(dataset_root)
    if not confirm_no_active_readers:
        raise ModalityStorageError(
            "apply requires explicit confirmation that no dataset readers are active"
        )
    # Managed retries must recover a journal before attempting to rebuild an
    # inventory: during an unreceipted whole-prefix retirement, tactile/ may
    # temporarily live only in quarantine.  A receipted transaction is already
    # committed, so return its verified result idempotently.
    with _maintenance_lock(root):
        _require_quiescent()
        recover_modality_storage_transaction(root)
        completed = _completed_managed_move_result(
            root,
            cold_root,
            require_distinct_device=require_distinct_device,
            min_free_bytes=min_free_bytes,
        )
        if completed is not None:
            return completed

    plan = plan_modality_detach(
        root,
        cold_root,
        detachment_scope=WHOLE_PREFIX_DETACH,
        require_distinct_device=require_distinct_device,
        min_free_bytes=min_free_bytes,
    )
    if not plan.ready:
        raise ModalityStorageError("plan is not ready: " + "; ".join(plan.errors))
    return detach_modality(
        root,
        cold_root,
        expected_plan_hash=plan.plan_hash,
        confirm_dataset_fingerprint=plan.dataset_fingerprint,
        detachment_scope=WHOLE_PREFIX_DETACH,
        require_distinct_device=require_distinct_device,
        min_free_bytes=min_free_bytes,
        source_storage_id=source_storage_id,
        target_storage_id=target_storage_id,
        confirm_no_active_readers=confirm_no_active_readers,
    )


def _rehydrate_swap(
    dataset_root: Path,
    artifact_root: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    detachment_scope: str,
    source_storage_id: str,
    target_storage_id: str,
) -> Path:
    operation_id = f"rehydrate.{uuid.uuid4().hex}"
    stage, quarantine = _swap_paths(dataset_root, operation_id)
    scope = _normalize_detachment_scope(detachment_scope)
    active = dataset_root / "tactile"
    active_was_present = active.exists()
    if scope == WHOLE_PREFIX_DETACH and active_was_present:
        raise ModalityStorageError(
            "whole-prefix rehydrate requires the active tactile/ tree to be absent"
        )
    if scope == PAYLOAD_ONLY_DETACH and not active_was_present:
        raise ModalityStorageError(
            "payload-only rehydrate requires the manifest-only tactile/ tree"
        )
    stage.mkdir()
    committed = False
    try:
        _copy_inventory_artifacts(artifact_root, stage, inventory)
        verify_modality_artifact_inventory(
            dataset_root,
            inventory,
            artifact_root=stage,
        )
        _exact_artifact_tree(stage, inventory)
        _verify_tactile_reader(stage, inventory)
        _fsync_tree(stage)
        _write_journal(
            dataset_root,
            operation_id=operation_id,
            operation="rehydrate",
            inventory_hash=inventory.semantic_content_hash,
            stage=stage,
            quarantine=quarantine,
            active_was_present=active_was_present,
            artifact_root=artifact_root,
        )
        if active_was_present:
            os.replace(active, quarantine)
        os.replace(stage / "tactile", active)
        _fsync_directory(dataset_root)
        _fsync_directory(dataset_root.parent)
        verify_modality_artifact_inventory(dataset_root, inventory)
        _verify_tactile_reader(dataset_root, inventory)
        receipt = _receipt(
            operation="rehydrate",
            operation_id=operation_id,
            inventory=inventory,
            source_storage_id=source_storage_id,
            target_storage_id=target_storage_id,
            file_count=inventory.file_count,
            total_bytes=inventory.total_bytes,
            independent_copy_verified=True,
        )
        path = _publish_receipt(dataset_root, receipt)
        committed = True
        if quarantine.exists():
            shutil.rmtree(quarantine)
        shutil.rmtree(stage, ignore_errors=True)
        _fsync_directory(dataset_root.parent)
        _journal_path(dataset_root).unlink(missing_ok=True)
        _fsync_directory(_journal_path(dataset_root).parent)
        return path
    except BaseException:
        if not committed:
            _rollback_swap(
                dataset_root,
                stage,
                quarantine,
                active_was_present=active_was_present,
            )
        raise


def rehydrate_modality(
    dataset_root: str | Path,
    cold_root: str | Path,
    inventory_hash: str,
    *,
    expected_plan_hash: str,
    confirm_dataset_fingerprint: str,
    require_distinct_device: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    source_storage_id: str = "cold.archive",
    target_storage_id: str = "hot.local",
    confirm_no_active_readers: bool = False,
) -> ModalityStorageResult:
    root = _safe_resolved(dataset_root)
    if not confirm_no_active_readers:
        raise ModalityStorageError(
            "apply requires explicit confirmation that no dataset readers are active"
        )
    with _maintenance_lock(root):
        _require_quiescent()
        recover_modality_storage_transaction(root)
        plan = plan_modality_rehydrate(
            root,
            cold_root,
            inventory_hash,
            require_distinct_device=require_distinct_device,
            min_free_bytes=min_free_bytes,
        )
        _assert_confirmation(
            plan,
            expected_plan_hash=expected_plan_hash,
            confirm_dataset_fingerprint=confirm_dataset_fingerprint,
        )
        inventory = load_modality_inventory(root, inventory_hash)
        artifact_root = Path(plan.cold_artifact_root)
        verify_modality_artifact_inventory(
            root,
            inventory,
            artifact_root=artifact_root,
        )
        _exact_artifact_tree(artifact_root, inventory)
        _verify_tactile_reader(artifact_root, inventory)
        _require_quiescent()
        receipt_path = _rehydrate_swap(
            root,
            artifact_root,
            inventory,
            detachment_scope=plan.detachment_scope,
            source_storage_id=source_storage_id,
            target_storage_id=target_storage_id,
        )
        final = resolve_modality_availability(root, inventory)
        if final.status != SELF_CONTAINED:
            raise ModalityStorageError(f"unexpected final availability: {final.status}")
        return ModalityStorageResult(
            operation="rehydrate",
            dataset_root=str(root),
            cold_artifact_root=str(artifact_root),
            inventory_hash=inventory.semantic_content_hash,
            plan_hash=plan.plan_hash,
            final_status=final.status,
            receipt_paths=(receipt_path.relative_to(root).as_posix(),),
        )


def modality_storage_status(
    dataset_root: str | Path,
    *,
    inventory_hash: str | None = None,
) -> dict[str, Any]:
    """Read-only status; never recovers or removes an interrupted transaction."""

    root = _safe_resolved(dataset_root)
    pending = _journal_path(root)
    if inventory_hash is None:
        directory = _sidecar_root(root) / "inventories"
        paths = tuple(sorted(directory.glob("*.json"))) if directory.is_dir() else ()
        if not paths:
            try:
                inventory = build_modality_artifact_inventory(root)
            except Exception as exc:
                return {
                    "dataset_root": str(root),
                    "status": "INCOMPLETE_OR_CORRUPT",
                    "inventory_hash": None,
                    "pending_transaction": pending.is_file(),
                    "production_detached_verified": False,
                    "reason": f"no inventory sidecar and inventory build failed: {exc}",
                }
            availability = resolve_modality_availability(root, inventory)
            return {
                "dataset_root": str(root),
                "status": "SELF_CONTAINED_LEGACY",
                "structural_status": availability.status,
                "inventory_hash": inventory.semantic_content_hash,
                "pending_transaction": pending.is_file(),
                "production_detached_verified": False,
                "reason": "inventory is ephemeral; no storage operation sidecar exists",
            }
        inventories = [
            ModalityArtifactInventoryV1.from_dict(_read_json(path)) for path in paths
        ]
        current = [
            item
            for item in inventories
            if _stored_inventory_availability(root, item).status
            in {SELF_CONTAINED, DETACHED_VERIFIED}
        ]
        if len(current) != 1:
            return {
                "dataset_root": str(root),
                "status": "AMBIGUOUS_INVENTORY",
                "inventory_hash": None,
                "pending_transaction": pending.is_file(),
                "production_detached_verified": False,
                "reason": f"expected one current inventory, found {len(current)}",
            }
        inventory = current[0]
    else:
        inventory = load_modality_inventory(root, inventory_hash)
    detachment_scope = _detachment_scope_for_inventory(
        root,
        inventory.semantic_content_hash,
    )
    availability = resolve_modality_availability(
        root,
        inventory,
        detachment_scope=detachment_scope,
    )
    retirement_operation = (
        "retire_hot_module_tree"
        if detachment_scope == WHOLE_PREFIX_DETACH
        else "retire_hot_payloads"
    )
    retired = _has_receipt(
        root,
        inventory.semantic_content_hash,
        retirement_operation,
    )
    return {
        "dataset_root": str(root),
        "status": availability.status,
        "inventory_hash": inventory.semantic_content_hash,
        "detachment_scope": detachment_scope,
        "pending_transaction": pending.is_file(),
        "production_detached_verified": bool(
            availability.status == DETACHED_VERIFIED and retired and not pending.is_file()
        ),
        "core_training_eligible": bool(
            availability.core_training_eligible and not pending.is_file()
        ),
        "tactile_read_eligible": bool(
            availability.tactile_read_eligible and not pending.is_file()
        ),
        "receipt_operations": sorted(
            receipt.operation
            for receipt in _complete_receipts(root, inventory.semantic_content_hash)
        ),
        "reason": availability.reason,
    }


__all__ = [
    "DEFAULT_MIN_FREE_BYTES",
    "ModalityStorageError",
    "ModalityStoragePlan",
    "ModalityStorageResult",
    "detach_modality",
    "export_modality",
    "load_modality_inventory",
    "move_modality",
    "modality_storage_status",
    "plan_modality_detach",
    "plan_modality_export",
    "plan_modality_rehydrate",
    "recover_modality_storage_transaction",
    "rehydrate_modality",
]
