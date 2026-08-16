from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import git
from .drafts import DRAFT_SCHEMA_ID, _canonical_json
from .workspace import (
    COMMIT_OID_PATTERN,
    LabError,
    WorkspaceManager,
    _atomic_write_text,
    validate_repository_name,
    validate_task_id,
)

BUNDLE_SCHEMA_ID = "lab.draft-bundle.v1"
BUNDLE_RECEIPT_SCHEMA_ID = "lab.draft-bundle-receipt.v1"
MAX_BUNDLE_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_BUNDLE_FILE_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_SET_BYTES = 32 * 1024 * 1024


def _regular_file_identity(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LabError(f"cannot safely open {label}: {path}") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise LabError(f"{label} exceeds the protected byte limit: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(block)
                if total > max_bytes:
                    raise LabError(f"{label} exceeds the protected byte limit: {path}")
                digest.update(block)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ino != before.st_ino
            or after.st_dev != before.st_dev
        ):
            raise LabError(f"{label} changed while it was inspected: {path}")
        return digest.hexdigest(), total
    finally:
        os.close(descriptor)


def _read_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LabError(f"cannot safely open {label}: {path}") from exc
    content = bytearray()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise LabError(f"{label} exceeds the protected byte limit: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                content.extend(block)
                if len(content) > max_bytes:
                    raise LabError(f"{label} exceeds the protected byte limit: {path}")
        after = os.fstat(descriptor)
        if (
            len(content) != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ino != before.st_ino
            or after.st_dev != before.st_dev
        ):
            raise LabError(f"{label} changed while it was inspected: {path}")
        return bytes(content)
    finally:
        os.close(descriptor)


def _git_command(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LabError(f"Git bundle operation failed in {repository}: {detail}")
    return completed.stdout.rstrip()


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise LabError(f"{label} is missing or a symlink: {path}")
    try:
        raw = _read_regular_file(
            path,
            max_bytes=MAX_BUNDLE_MANIFEST_BYTES,
            label=label,
        )
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError(f"{label} must contain an object: {path}")
    return payload, raw


def _validated_oid(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not COMMIT_OID_PATTERN.fullmatch(value):
        raise LabError(f"invalid {label}: {value!r}")
    return value


def _validated_submodules(value: object, *, repository: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise LabError(f"invalid draft submodule mapping: {repository}")
    validated: dict[str, str] = {}
    for raw_path, raw_commit in value.items():
        if not isinstance(raw_path, str):
            raise LabError(f"invalid draft submodule path: {repository}")
        path = PurePosixPath(raw_path)
        if (
            not raw_path
            or raw_path.startswith(("/", "~"))
            or "\\" in raw_path
            or "\x00" in raw_path
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise LabError(f"unsafe draft submodule path in {repository}: {raw_path!r}")
        validated[raw_path] = _validated_oid(
            raw_commit,
            label=f"{repository}/{raw_path} submodule commit",
        )
    return dict(sorted(validated.items()))


def _draft_manifest_path(
    workspace: WorkspaceManager,
    task_id: str,
    snapshot_id: str,
) -> Path:
    validate_task_id(task_id)
    validate_task_id(snapshot_id)
    root = (workspace.state_root / "drafts").resolve()
    path = root / task_id / snapshot_id / "snapshot.json"
    if not path.resolve().is_relative_to(root):
        raise LabError("draft manifest path escapes configured state root")
    return path


def _load_draft_manifest(
    workspace: WorkspaceManager,
    task_id: str,
    snapshot_id: str,
) -> tuple[dict[str, object], bytes]:
    path = _draft_manifest_path(workspace, task_id, snapshot_id)
    payload, raw = _read_json_object(path, label="draft manifest")
    if (
        payload.get("schema") != DRAFT_SCHEMA_ID
        or payload.get("task_id") != task_id
        or payload.get("snapshot_id") != snapshot_id
        or payload.get("snapshot_type") != "synthetic-draft-git"
        or payload.get("remote_eligible") is not False
        or payload.get("verify_eligible") is not False
        or payload.get("device_eligible") is not False
    ):
        raise LabError(f"invalid draft manifest identity or eligibility: {path}")
    return payload, raw


def _repository_entry(
    name: str,
    payload: object,
) -> dict[str, object]:
    validate_repository_name(name)
    if not isinstance(payload, dict):
        raise LabError(f"invalid draft repository entry: {name}")
    base_commit = _validated_oid(payload.get("base_commit"), label=f"{name} base")
    commit = _validated_oid(payload.get("commit"), label=f"{name} commit")
    tree_oid = _validated_oid(payload.get("tree_oid"), label=f"{name} tree")
    synthetic = payload.get("synthetic")
    ref = payload.get("ref")
    submodules = _validated_submodules(
        payload.get("submodules"),
        repository=name,
    )
    if not isinstance(synthetic, bool):
        raise LabError(f"invalid draft synthetic flag: {name}")
    if not isinstance(ref, str) or not ref.startswith("refs/lab/snapshots/"):
        raise LabError(f"invalid draft ref: {name}")
    if synthetic is False and commit != base_commit:
        raise LabError(f"clean draft repository commit/base mismatch: {name}")
    return {
        "base_commit": base_commit,
        "commit": commit,
        "tree_oid": tree_oid,
        "synthetic": synthetic,
        "ref": ref,
        "submodules": submodules,
    }


def _bundle_identity(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": payload.get("schema"),
        "task_id": payload.get("task_id"),
        "snapshot_id": payload.get("snapshot_id"),
        "source_manifest_sha256": payload.get("source_manifest_sha256"),
        "repositories": payload.get("repositories"),
        "transport_eligible": payload.get("transport_eligible"),
        "remote_run_eligible": payload.get("remote_run_eligible"),
        "verify_eligible": payload.get("verify_eligible"),
        "device_eligible": payload.get("device_eligible"),
    }


def _load_existing_export(path: Path) -> dict[str, object]:
    manifest, _ = _read_json_object(path / "bundle.json", label="bundle manifest")
    expected = (
        "bundle-"
        + hashlib.sha256(_canonical_json(_bundle_identity(manifest))).hexdigest()[:32]
    )
    if (
        manifest.get("schema") != BUNDLE_SCHEMA_ID
        or manifest.get("bundle_set_id") != expected
    ):
        raise LabError(f"invalid existing bundle export: {path}")
    return manifest


def export_draft_bundle(
    workspace: WorkspaceManager,
    task_id: str,
    snapshot_id: str,
    output_root: Path,
) -> dict[str, object]:
    draft, draft_raw = _load_draft_manifest(workspace, task_id, snapshot_id)
    draft_repositories = draft.get("repositories")
    if not isinstance(draft_repositories, dict) or not draft_repositories:
        raise LabError("draft manifest has no repositories")
    task = workspace.load_task(task_id)
    task_repositories = task.get("repositories")
    if not isinstance(task_repositories, dict) or set(task_repositories) != set(
        draft_repositories
    ):
        raise LabError("draft repositories do not match current Task manifest")
    repository_paths: dict[str, Path] = {}
    entries: dict[str, dict[str, object]] = {}
    for name, raw in draft_repositories.items():
        if not isinstance(name, str):
            raise LabError("draft repository name must be a string")
        entry = _repository_entry(name, raw)
        task_repository = task_repositories[name]
        if not isinstance(task_repository, dict):
            raise LabError(f"invalid Task repository entry: {name}")
        path = Path(str(task_repository.get("worktree_path", ""))).resolve()
        if git.repository_root(path) != path:
            raise LabError(f"Task repository path/root mismatch: {name}")
        if git.resolve_commit(path, str(entry["ref"])) != entry["commit"]:
            raise LabError(f"draft ref/commit mismatch: {name}")
        if git.tree_oid(path, str(entry["commit"])) != entry["tree_oid"]:
            raise LabError(f"draft commit/tree mismatch: {name}")
        if entry["synthetic"]:
            parents = _git_command(
                path, "rev-list", "--parents", "-n", "1", str(entry["commit"])
            ).split()
            if parents != [entry["commit"], entry["base_commit"]]:
                raise LabError(
                    f"synthetic draft must have exactly the recorded base parent: {name}"
                )
        repository_paths[name] = path
        entries[name] = entry

    requested_output_root = output_root.expanduser()
    if requested_output_root.is_symlink():
        raise LabError(f"bundle output root is a symlink: {requested_output_root}")
    output_root = requested_output_root.resolve()
    for name, repository in repository_paths.items():
        if output_root.is_relative_to(repository):
            raise LabError(f"bundle output root must be outside Task repository {name}")
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / snapshot_id
    if target.is_symlink():
        raise LabError(f"bundle export target is a symlink: {target}")
    if target.exists():
        existing_manifest = _load_existing_export(target)
        receipt = verify_draft_bundle_set(
            target / "bundle.json",
            repository_paths=repository_paths,
            require_synthetic_absent=False,
        )
        return {
            "path": str(target),
            "bundle": existing_manifest,
            "verification": receipt,
        }

    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=output_root))
    try:
        bundled: dict[str, dict[str, object]] = {}
        total_bundle_bytes = 0
        for name, entry in entries.items():
            record = dict(entry)
            if entry["synthetic"]:
                filename = (
                    f"{name}-{hashlib.sha256(name.encode()).hexdigest()[:12]}.bundle"
                )
                bundle_path = temporary / filename
                _git_command(
                    repository_paths[name],
                    "bundle",
                    "create",
                    str(bundle_path),
                    str(entry["ref"]),
                    f"^{entry['base_commit']}",
                )
                _git_command(
                    repository_paths[name], "bundle", "verify", str(bundle_path)
                )
                heads = _git_command(
                    repository_paths[name], "bundle", "list-heads", str(bundle_path)
                ).splitlines()
                if heads != [f"{entry['commit']} {entry['ref']}"]:
                    raise LabError(f"bundle advertised ref mismatch: {name}")
                bundle_sha256, bundle_size = _regular_file_identity(
                    bundle_path,
                    max_bytes=MAX_BUNDLE_FILE_BYTES,
                    label=f"draft bundle for {name}",
                )
                total_bundle_bytes += bundle_size
                if total_bundle_bytes > MAX_BUNDLE_SET_BYTES:
                    raise LabError("draft bundle set exceeds the protected byte limit")
                record.update(
                    {
                        "bundle_file": filename,
                        "bundle_sha256": bundle_sha256,
                        "bundle_size": bundle_size,
                        "prerequisites": [entry["base_commit"]],
                    }
                )
            else:
                record.update(
                    {
                        "bundle_file": None,
                        "bundle_sha256": None,
                        "bundle_size": 0,
                        "prerequisites": [entry["base_commit"]],
                    }
                )
            bundled[name] = record

        identity: dict[str, object] = {
            "schema": BUNDLE_SCHEMA_ID,
            "task_id": task_id,
            "snapshot_id": snapshot_id,
            "source_manifest_sha256": hashlib.sha256(draft_raw).hexdigest(),
            "repositories": bundled,
            "transport_eligible": True,
            "remote_run_eligible": False,
            "verify_eligible": False,
            "device_eligible": False,
        }
        generated_manifest: dict[str, object] = {
            **identity,
            "bundle_set_id": "bundle-"
            + hashlib.sha256(_canonical_json(identity)).hexdigest()[:32],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_text(
            temporary / "bundle.json",
            json.dumps(generated_manifest, indent=2, sort_keys=True) + "\n",
        )
        temporary.rename(target)
        receipt = verify_draft_bundle_set(
            target / "bundle.json",
            repository_paths=repository_paths,
            require_synthetic_absent=False,
        )
        _atomic_write_text(
            target / "verification.json",
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        )
        return {
            "path": str(target),
            "bundle": generated_manifest,
            "verification": receipt,
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _resolve_bundle_path(root: Path, value: object, *, repository: str) -> Path:
    if (
        not isinstance(value, str)
        or Path(value).name != value
        or not value.endswith(".bundle")
    ):
        raise LabError(f"invalid bundle filename: {repository}")
    path = root / value
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise LabError(
            f"bundle file is missing, a symlink, or escapes root: {repository}"
        )
    return path


def _verify_in_temporary_repository(
    repository: Path,
    bundle_path: Path,
    entry: Mapping[str, object],
) -> None:
    with tempfile.TemporaryDirectory(prefix="lab-bundle-verify-") as temporary:
        bare = Path(temporary) / "verify.git"
        bare.mkdir()
        _git_command(bare, "init", "--bare")
        objects_value = _git_command(repository, "rev-parse", "--git-path", "objects")
        objects = Path(objects_value)
        if not objects.is_absolute():
            objects = (repository / objects).resolve()
        alternates = bare / "objects" / "info" / "alternates"
        alternates.parent.mkdir(parents=True, exist_ok=True)
        alternates.write_text(str(objects.resolve()) + "\n", encoding="utf-8")
        _git_command(
            bare,
            "fetch",
            "--no-tags",
            "--no-write-fetch-head",
            str(bundle_path),
            f"{entry['ref']}:refs/verify/draft",
        )
        commit = _git_command(bare, "rev-parse", "refs/verify/draft^{commit}")
        tree_oid = _git_command(bare, "rev-parse", "refs/verify/draft^{tree}")
        parents = _git_command(bare, "rev-list", "--parents", "-n", "1", commit).split()
        if commit != entry["commit"] or tree_oid != entry["tree_oid"]:
            raise LabError("bundle materialized the wrong commit/tree")
        if parents != [entry["commit"], entry["base_commit"]]:
            raise LabError("bundle materialized a commit with the wrong parent")


def verify_draft_bundle_set(
    manifest_path: Path,
    *,
    repository_paths: Mapping[str, Path],
    require_synthetic_absent: bool,
) -> dict[str, object]:
    requested_manifest_path = manifest_path.expanduser()
    if requested_manifest_path.is_symlink():
        raise LabError(f"bundle manifest is a symlink: {requested_manifest_path}")
    manifest_path = requested_manifest_path.resolve()
    manifest, _ = _read_json_object(manifest_path, label="bundle manifest")
    root = manifest_path.parent
    if manifest.get("schema") != BUNDLE_SCHEMA_ID:
        raise LabError("unsupported draft bundle schema")
    task_id = str(manifest.get("task_id", ""))
    snapshot_id = str(manifest.get("snapshot_id", ""))
    validate_task_id(task_id)
    validate_task_id(snapshot_id)
    expected_id = (
        "bundle-"
        + hashlib.sha256(_canonical_json(_bundle_identity(manifest))).hexdigest()[:32]
    )
    if manifest.get("bundle_set_id") != expected_id:
        raise LabError("draft bundle content identity mismatch")
    source_manifest_sha256 = manifest.get("source_manifest_sha256")
    if (
        not isinstance(source_manifest_sha256, str)
        or len(source_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in source_manifest_sha256
        )
    ):
        raise LabError("draft bundle has an invalid source manifest hash")
    if (
        manifest.get("transport_eligible") is not True
        or manifest.get("remote_run_eligible") is not False
        or manifest.get("verify_eligible") is not False
        or manifest.get("device_eligible") is not False
    ):
        raise LabError("draft bundle has an unsafe eligibility envelope")
    repositories = manifest.get("repositories")
    if not isinstance(repositories, dict) or set(repositories) != set(repository_paths):
        raise LabError("bundle repository mapping mismatch")

    receipts: dict[str, object] = {}
    total_bundle_bytes = 0
    for name, raw_entry in repositories.items():
        if not isinstance(name, str) or not isinstance(raw_entry, dict):
            raise LabError("invalid bundle repository entry")
        entry = _repository_entry(name, raw_entry)
        repository = git.repository_root(repository_paths[name])
        _git_command(repository, "check-ref-format", str(entry["ref"]))
        if (
            git.resolve_commit(repository, str(entry["base_commit"]))
            != entry["base_commit"]
        ):
            raise LabError(f"bundle prerequisite base is unavailable: {name}")
        synthetic_present = (
            git.run_git(
                repository,
                ["cat-file", "-e", f"{entry['commit']}^{{commit}}"],
                check=False,
            ).returncode
            == 0
        )
        if entry["synthetic"]:
            if require_synthetic_absent and synthetic_present:
                raise LabError(f"target already contains synthetic commit: {name}")
            bundle_path = _resolve_bundle_path(
                root, raw_entry.get("bundle_file"), repository=name
            )
            expected_size = raw_entry.get("bundle_size")
            expected_sha = raw_entry.get("bundle_sha256")
            actual_sha, actual_size = _regular_file_identity(
                bundle_path,
                max_bytes=MAX_BUNDLE_FILE_BYTES,
                label=f"draft bundle for {name}",
            )
            if (
                not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size <= 0
                or actual_size != expected_size
                or not isinstance(expected_sha, str)
                or actual_sha != expected_sha
            ):
                raise LabError(f"bundle file identity mismatch: {name}")
            total_bundle_bytes += actual_size
            if total_bundle_bytes > MAX_BUNDLE_SET_BYTES:
                raise LabError("draft bundle set exceeds the protected byte limit")
            if raw_entry.get("prerequisites") != [entry["base_commit"]]:
                raise LabError(f"bundle prerequisites mismatch: {name}")
            _git_command(repository, "bundle", "verify", str(bundle_path))
            heads = _git_command(
                repository, "bundle", "list-heads", str(bundle_path)
            ).splitlines()
            if heads != [f"{entry['commit']} {entry['ref']}"]:
                raise LabError(f"bundle advertised ref mismatch: {name}")
            _verify_in_temporary_repository(repository, bundle_path, entry)
        else:
            if (
                raw_entry.get("bundle_file") is not None
                or raw_entry.get("bundle_sha256") is not None
                or raw_entry.get("bundle_size") != 0
                or raw_entry.get("prerequisites") != [entry["base_commit"]]
                or entry["commit"] != entry["base_commit"]
            ):
                raise LabError(f"clean bundle repository entry is inconsistent: {name}")
            if git.tree_oid(repository, str(entry["commit"])) != entry["tree_oid"]:
                raise LabError(f"clean bundle repository tree mismatch: {name}")
        receipts[name] = {
            "base_commit": entry["base_commit"],
            "commit": entry["commit"],
            "tree_oid": entry["tree_oid"],
            "synthetic": entry["synthetic"],
            "synthetic_preexisting": synthetic_present,
            "submodules": entry["submodules"],
            "verified": True,
        }

    receipt_content: dict[str, object] = {
        "schema": BUNDLE_RECEIPT_SCHEMA_ID,
        "bundle_set_id": manifest["bundle_set_id"],
        "task_id": task_id,
        "snapshot_id": snapshot_id,
        "repositories": receipts,
        "source_verified": True,
        "materialized_in_temporary_repository": True,
        "target_repository_mutated": False,
        "remote_run_eligible": False,
        "verify_eligible": False,
        "device_eligible": False,
    }
    return {
        **receipt_content,
        "receipt_sha256": hashlib.sha256(_canonical_json(receipt_content)).hexdigest(),
    }
