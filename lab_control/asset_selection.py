from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import git
from .asset_lock import AssetLock
from .asset_observation import AssetObservationSet
from .remote_request import OID_PATTERN, SHA256_PATTERN, _canonical_json
from .workspace import LabError, validate_task_id

ARTIFACT_MANIFEST_SCHEMA_ID = "lab.artifact-content-manifest.v1"
WORKLOAD_SELECTION_SCHEMA_ID = "lab.workload-selection.v1"
RUNTIME_ASSET_VERIFICATION_SCHEMA_ID = (
    "lab.runtime-asset-verification.v1"
)
MAX_ARTIFACT_FILES = 250_000
MAX_ARTIFACT_TOTAL_BYTES = 4 * 1024**4


@dataclass(frozen=True)
class WorkloadSelection:
    selection_id: str
    observation_id: str
    observation_sha256: str
    artifact_manifest_receipt_sha256: str
    workload_id: str
    purpose: str
    selected_at: str
    path: Path
    sha256: str


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _timestamp(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LabError(f"{label} has an invalid timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError(f"{label} has an invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise LabError(f"{label} timestamp requires a timezone")
    return value


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise LabError(f"{label} is invalid")
    path = PurePosixPath(value)
    if value.startswith(("/", "~")) or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise LabError(f"{label} is unsafe")
    return value


def _hash_open_regular_file(
    directory_descriptor: int,
    name: str,
    *,
    relative_path: str,
) -> tuple[str, int, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise LabError(
            f"artifact file is missing, escaped, or uses a symlink: {relative_path}"
        ) from exc
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LabError(f"artifact entry is not a regular file: {relative_path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                total += len(block)
                if total > MAX_ARTIFACT_TOTAL_BYTES:
                    raise LabError("artifact file exceeds the protected byte limit")
                digest.update(block)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ino != before.st_ino
            or after.st_dev != before.st_dev
        ):
            raise LabError(f"artifact file changed while hashing: {relative_path}")
        return digest.hexdigest(), total, stat.S_IMODE(before.st_mode)
    finally:
        os.close(descriptor)


def _artifact_entries(root: Path) -> tuple[dict[str, dict[str, object]], int]:
    entries: dict[str, dict[str, object]] = {}
    total_bytes = 0
    for directory, directory_names, file_names, descriptor in os.fwalk(
        root,
        topdown=True,
        follow_symlinks=False,
    ):
        current = Path(directory)
        try:
            relative_directory = current.relative_to(root)
        except ValueError as exc:
            raise LabError("artifact traversal escaped its root") from exc
        for directory_name in sorted(directory_names):
            metadata = os.stat(
                directory_name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                relative = (relative_directory / directory_name).as_posix()
                raise LabError(f"artifact directory entry is unsafe: {relative}")
        for file_name in sorted(file_names):
            relative = (relative_directory / file_name).as_posix()
            _safe_relative_path(relative, label="artifact file path")
            if len(entries) >= MAX_ARTIFACT_FILES:
                raise LabError("artifact exceeds the protected file-count limit")
            digest, size, mode = _hash_open_regular_file(
                descriptor,
                file_name,
                relative_path=relative,
            )
            total_bytes += size
            if total_bytes > MAX_ARTIFACT_TOTAL_BYTES:
                raise LabError("artifact exceeds the protected total-byte limit")
            entries[relative] = {
                "sha256": digest,
                "size_bytes": size,
                "mode": mode,
            }
    if not entries:
        raise LabError("artifact contains no regular files")
    return dict(sorted(entries.items())), total_bytes


def _hash_pinned_file(root: Path, relative_path: str) -> str:
    safe_relative = _safe_relative_path(relative_path, label="pinned asset path")
    requested = root / safe_relative
    resolved = requested.resolve()
    parent = resolved.parent
    if (
        requested.is_symlink()
        or resolved != requested.absolute()
        or not resolved.is_relative_to(root)
        or parent.is_symlink()
        or not parent.is_dir()
    ):
        raise LabError(f"pinned asset is missing, escaped, or unsafe: {safe_relative}")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        parent_descriptor = os.open(parent, flags)
    except OSError as exc:
        raise LabError(f"cannot safely open pinned asset: {safe_relative}") from exc
    try:
        digest, _, _ = _hash_open_regular_file(
            parent_descriptor,
            resolved.name,
            relative_path=safe_relative,
        )
        return digest
    finally:
        os.close(parent_descriptor)


def collect_artifact_manifest(
    observations: AssetObservationSet,
    *,
    workload_id: str,
    source_root: Path,
    expected_canonical_identity: str,
) -> dict[str, object]:
    """Hash one observed host-local artifact tree without network or activation."""

    validate_task_id(workload_id)
    candidate = observations.candidates.get(workload_id)
    if candidate is None:
        raise LabError(f"asset observations do not contain workload: {workload_id}")
    requested = source_root.expanduser()
    root = requested.resolve()
    if not requested.is_absolute() or requested.is_symlink() or requested.absolute() != root:
        raise LabError(f"artifact source root is unsafe: {source_root}")
    try:
        repository_root = git.repository_root(root)
        identity = git.origin_identity(root)
        commit = git.current_commit(root)
        tree_oid = git.tree_oid(root, commit)
        dirty = git.status_short(root)
    except git.GitError as exc:
        raise LabError(f"cannot identify artifact source repository: {exc}") from exc
    if (
        repository_root != root
        or identity != expected_canonical_identity
        or identity != observations.canonical_identity
        or commit != observations.commit
        or tree_oid != observations.tree_oid
        or dirty
    ):
        raise LabError("artifact source repository differs from its observation")
    for pin in candidate.companion_assets.values():
        if _hash_pinned_file(root, pin.relative_path) != pin.sha256:
            raise LabError(
                f"artifact companion differs from its observation: {pin.role}"
            )
    relative_content = _safe_relative_path(
        candidate.content_relative_path,
        label="artifact content relative path",
    )
    requested_content = root / relative_content
    content_root = requested_content.resolve()
    if (
        requested_content.is_symlink()
        or content_root != requested_content.absolute()
        or not content_root.is_relative_to(root)
        or not content_root.is_dir()
    ):
        raise LabError("artifact root is missing, escaped, or uses a symlink")
    entries, total_bytes = _artifact_entries(content_root)
    content_identity = {
        "domain": "lab.artifact-content.v1",
        "files": entries,
    }
    content: dict[str, object] = {
        "schema": ARTIFACT_MANIFEST_SCHEMA_ID,
        "observation_id": observations.observation_id,
        "observation_sha256": observations.sha256,
        "host_id": observations.host_id,
        "repository": {
            "canonical_identity": identity,
            "commit": commit,
            "tree_oid": tree_oid,
        },
        "workload_id": workload_id,
        "content_role": candidate.content_role,
        "content_relative_path": relative_content,
        "files": entries,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "content_sha256": _sha256(content_identity),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "network_used": False,
        "downloads_permitted": False,
        "large_file_downloaded": False,
        "selection_authorized": False,
        "activation_ready": False,
        "device_access": False,
    }
    manifest = {**content, "receipt_sha256": _sha256(content)}
    return validate_artifact_manifest(
        manifest,
        observations=observations,
        workload_id=workload_id,
    )


def validate_artifact_manifest(
    manifest: object,
    *,
    observations: AssetObservationSet,
    workload_id: str,
) -> dict[str, object]:
    validate_task_id(workload_id)
    candidate = observations.candidates.get(workload_id)
    if candidate is None:
        raise LabError(f"asset observations do not contain workload: {workload_id}")
    expected_keys = {
        "schema",
        "observation_id",
        "observation_sha256",
        "host_id",
        "repository",
        "workload_id",
        "content_role",
        "content_relative_path",
        "files",
        "file_count",
        "total_bytes",
        "content_sha256",
        "observed_at",
        "network_used",
        "downloads_permitted",
        "large_file_downloaded",
        "selection_authorized",
        "activation_ready",
        "device_access",
        "receipt_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise LabError("artifact manifest has an unexpected shape")
    if (
        manifest.get("schema") != ARTIFACT_MANIFEST_SCHEMA_ID
        or manifest.get("observation_id") != observations.observation_id
        or manifest.get("observation_sha256") != observations.sha256
        or manifest.get("host_id") != observations.host_id
        or manifest.get("workload_id") != workload_id
        or manifest.get("content_role") != candidate.content_role
        or manifest.get("content_relative_path") != candidate.content_relative_path
    ):
        raise LabError("artifact manifest identity mismatch")
    repository = manifest.get("repository")
    if not isinstance(repository, dict) or set(repository) != {
        "canonical_identity",
        "commit",
        "tree_oid",
    }:
        raise LabError("artifact manifest repository identity is invalid")
    if (
        repository.get("canonical_identity") != observations.canonical_identity
        or repository.get("commit") != observations.commit
        or repository.get("tree_oid") != observations.tree_oid
        or not OID_PATTERN.fullmatch(str(repository.get("commit", "")))
        or not OID_PATTERN.fullmatch(str(repository.get("tree_oid", "")))
    ):
        raise LabError("artifact manifest repository identity mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > MAX_ARTIFACT_FILES:
        raise LabError("artifact manifest has invalid files")
    total_bytes = 0
    normalized_files: dict[str, dict[str, object]] = {}
    for relative_path, identity in files.items():
        safe_path = _safe_relative_path(
            relative_path,
            label="artifact manifest file path",
        )
        size = identity.get("size_bytes") if isinstance(identity, dict) else None
        mode = identity.get("mode") if isinstance(identity, dict) else None
        if (
            not isinstance(identity, dict)
            or set(identity) != {"sha256", "size_bytes", "mode"}
            or not SHA256_PATTERN.fullmatch(str(identity.get("sha256", "")))
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(mode, int)
            or isinstance(mode, bool)
            or not 0 <= mode <= 0o7777
        ):
            raise LabError(f"artifact manifest file is invalid: {relative_path}")
        total_bytes += size
        if total_bytes > MAX_ARTIFACT_TOTAL_BYTES:
            raise LabError("artifact manifest exceeds the total-byte limit")
        normalized_files[safe_path] = dict(identity)
    if (
        manifest.get("file_count") != len(files)
        or manifest.get("total_bytes") != total_bytes
        or manifest.get("content_sha256")
        != _sha256(
            {
                "domain": "lab.artifact-content.v1",
                "files": dict(sorted(normalized_files.items())),
            }
        )
    ):
        raise LabError("artifact manifest content identity mismatch")
    for field in ("observation_sha256", "content_sha256", "receipt_sha256"):
        if not SHA256_PATTERN.fullmatch(str(manifest.get(field, ""))):
            raise LabError(f"artifact manifest has invalid {field}")
    _timestamp(manifest.get("observed_at"), label="artifact manifest")
    if (
        manifest.get("network_used") is not False
        or manifest.get("downloads_permitted") is not False
        or manifest.get("large_file_downloaded") is not False
        or manifest.get("selection_authorized") is not False
        or manifest.get("activation_ready") is not False
        or manifest.get("device_access") is not False
    ):
        raise LabError("artifact manifest has an unsafe authority envelope")
    content = {key: value for key, value in manifest.items() if key != "receipt_sha256"}
    if manifest.get("receipt_sha256") != _sha256(content):
        raise LabError("artifact manifest receipt hash mismatch")
    return manifest


def load_workload_selection(path: Path) -> WorkloadSelection:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read workload selection {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "selection_id",
        "observation_id",
        "observation_sha256",
        "artifact_manifest_receipt_sha256",
        "workload_id",
        "purpose",
        "selected_at",
        "device_access",
        "network_egress",
        "downloads_permitted",
    }:
        raise LabError(f"workload selection has an unexpected shape: {resolved}")
    if payload.get("schema") != WORKLOAD_SELECTION_SCHEMA_ID:
        raise LabError(f"unsupported workload selection: {resolved}")
    selection_id = payload.get("selection_id")
    observation_id = payload.get("observation_id")
    workload_id = payload.get("workload_id")
    if not all(
        isinstance(value, str)
        for value in (selection_id, observation_id, workload_id)
    ):
        raise LabError("workload selection has invalid identity")
    validate_task_id(str(selection_id))
    validate_task_id(str(observation_id))
    validate_task_id(str(workload_id))
    purpose = payload.get("purpose")
    if purpose != "gpu-only-validation":
        raise LabError("workload selection has an unsupported purpose")
    for field in ("observation_sha256", "artifact_manifest_receipt_sha256"):
        if not SHA256_PATTERN.fullmatch(str(payload.get(field, ""))):
            raise LabError(f"workload selection has invalid {field}")
    if (
        payload.get("device_access") is not False
        or payload.get("network_egress") != "loopback-only"
        or payload.get("downloads_permitted") is not False
    ):
        raise LabError("workload selection has an unsafe authority envelope")
    return WorkloadSelection(
        selection_id=str(selection_id),
        observation_id=str(observation_id),
        observation_sha256=str(payload["observation_sha256"]),
        artifact_manifest_receipt_sha256=str(
            payload["artifact_manifest_receipt_sha256"]
        ),
        workload_id=str(workload_id),
        purpose=str(purpose),
        selected_at=_timestamp(
            payload.get("selected_at"), label="workload selection"
        ),
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_selected_asset_lock(
    observations: AssetObservationSet,
    *,
    artifact_manifest: object,
    selection: WorkloadSelection,
) -> dict[str, object]:
    if (
        selection.observation_id != observations.observation_id
        or selection.observation_sha256 != observations.sha256
        or selection.workload_id not in observations.candidates
    ):
        raise LabError("workload selection does not match asset observations")
    manifest = validate_artifact_manifest(
        artifact_manifest,
        observations=observations,
        workload_id=selection.workload_id,
    )
    if (
        selection.artifact_manifest_receipt_sha256
        != manifest.get("receipt_sha256")
    ):
        raise LabError("workload selection does not match artifact manifest")
    candidate = observations.candidates[selection.workload_id]
    lock: dict[str, object] = {
        "schema": "lab.asset-lock.v1",
        "lock_id": f"remote-assets-{selection.sha256[:16]}",
        "workload_id": selection.workload_id,
        "assets": {
            candidate.content_role: {
                "asset_id": f"content-{str(manifest['content_sha256'])[:16]}",
                "sha256": manifest["content_sha256"],
            },
            **{
                role: {"asset_id": pin.asset_id, "sha256": pin.sha256}
                for role, pin in candidate.companion_assets.items()
            },
        },
        "unresolved": [],
    }
    return lock


def validate_runtime_asset_verification(receipt: object) -> dict[str, object]:
    expected_keys = {
        "schema",
        "observation_sha256",
        "artifact_manifest_receipt_sha256",
        "selection_sha256",
        "asset_lock_sha256",
        "asset_lock_id",
        "workload_id",
        "assets",
        "verified_at",
        "content_rehashed",
        "network_used",
        "downloads_permitted",
        "device_access",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise LabError("runtime asset verification has an unexpected shape")
    if receipt.get("schema") != RUNTIME_ASSET_VERIFICATION_SCHEMA_ID:
        raise LabError("unsupported runtime asset verification schema")
    for field in (
        "observation_sha256",
        "artifact_manifest_receipt_sha256",
        "selection_sha256",
        "asset_lock_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"runtime asset verification has invalid {field}")
    for field in ("asset_lock_id", "workload_id"):
        if not isinstance(receipt.get(field), str):
            raise LabError(f"runtime asset verification has invalid {field}")
        validate_task_id(str(receipt[field]))
    assets = receipt.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise LabError("runtime asset verification has invalid assets")
    for role, identity in assets.items():
        if not isinstance(role, str):
            raise LabError("runtime asset verification role must be a string")
        validate_task_id(role)
        if not isinstance(identity, dict) or set(identity) != {
            "asset_id",
            "sha256",
            "path",
        }:
            raise LabError(f"runtime asset verification is invalid: {role}")
        asset_id = identity.get("asset_id")
        raw_path = identity.get("path")
        if not isinstance(asset_id, str) or not isinstance(raw_path, str):
            raise LabError(f"runtime asset verification is invalid: {role}")
        validate_task_id(asset_id)
        path = Path(raw_path)
        if (
            not SHA256_PATTERN.fullmatch(str(identity.get("sha256", "")))
            or not path.is_absolute()
            or ".." in path.parts
        ):
            raise LabError(f"runtime asset verification is invalid: {role}")
    _timestamp(receipt.get("verified_at"), label="runtime asset verification")
    if (
        receipt.get("content_rehashed") is not True
        or receipt.get("network_used") is not False
        or receipt.get("downloads_permitted") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("runtime asset verification has an unsafe authority envelope")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("runtime asset verification hash mismatch")
    return receipt


def build_runtime_asset_verification(
    observations: AssetObservationSet,
    *,
    artifact_manifest: object,
    selection: WorkloadSelection,
    asset_lock: AssetLock,
    source_root: Path,
    expected_canonical_identity: str,
) -> dict[str, object]:
    """Rehash all selected host-local assets immediately before a remote Run."""

    manifest = validate_artifact_manifest(
        artifact_manifest,
        observations=observations,
        workload_id=selection.workload_id,
    )
    expected_lock = build_selected_asset_lock(
        observations,
        artifact_manifest=manifest,
        selection=selection,
    )
    observed_lock = {
        "schema": "lab.asset-lock.v1",
        "lock_id": asset_lock.lock_id,
        "workload_id": asset_lock.workload_id,
        "assets": {
            role: {
                "asset_id": pin.asset_id,
                "sha256": pin.sha256,
            }
            for role, pin in asset_lock.assets.items()
        },
        "unresolved": list(asset_lock.unresolved),
    }
    if observed_lock != expected_lock:
        raise LabError("active asset lock differs from the reviewed workload selection")
    fresh = collect_artifact_manifest(
        observations,
        workload_id=selection.workload_id,
        source_root=source_root,
        expected_canonical_identity=expected_canonical_identity,
    )
    for field in (
        "repository",
        "workload_id",
        "content_role",
        "content_relative_path",
        "files",
        "file_count",
        "total_bytes",
        "content_sha256",
    ):
        if fresh.get(field) != manifest.get(field):
            raise LabError(f"artifact changed after its reviewed manifest: {field}")
    candidate = observations.candidates[selection.workload_id]
    root = source_root.resolve()
    assets = {
        candidate.content_role: {
            "asset_id": asset_lock.assets[candidate.content_role].asset_id,
            "sha256": asset_lock.assets[candidate.content_role].sha256,
            "path": str((root / candidate.content_relative_path).resolve()),
        },
        **{
            role: {
                "asset_id": asset_lock.assets[role].asset_id,
                "sha256": asset_lock.assets[role].sha256,
                "path": str((root / pin.relative_path).resolve()),
            }
            for role, pin in candidate.companion_assets.items()
        },
    }
    content: dict[str, object] = {
        "schema": RUNTIME_ASSET_VERIFICATION_SCHEMA_ID,
        "observation_sha256": observations.sha256,
        "artifact_manifest_receipt_sha256": manifest["receipt_sha256"],
        "selection_sha256": selection.sha256,
        "asset_lock_sha256": asset_lock.sha256,
        "asset_lock_id": asset_lock.lock_id,
        "workload_id": selection.workload_id,
        "assets": assets,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "content_rehashed": True,
        "network_used": False,
        "downloads_permitted": False,
        "device_access": False,
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_runtime_asset_verification(receipt)
