from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from . import git
from .remote_request import OID_PATTERN, SHA256_PATTERN
from .workspace import LabError

CONTROL_PLANE_PACKAGE_SCHEMA_ID = "lab.control-plane-package.v1"
PROTECTED_REF = "refs/remotes/origin/main"
PACKAGE_PATHS = (
    "projectctl",
    "lab_control",
    "integration/project.json",
    "integration/command_registries",
    "integration/host_profiles",
    "integration/preparation_profiles",
    "integration/remote_profiles",
    "integration/service_adapters",
    "stack/workload-selections",
    "stack/remote-assets.json",
    "stack/remote-asset-observations.json",
)
MAX_FILE_BYTES = 1024 * 1024
MAX_CONTENT_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _allowed_package_path(path: str) -> bool:
    return path in {
        "projectctl",
        "integration/project.json",
        "stack/remote-assets.json",
        "stack/remote-asset-observations.json",
    } or any(
        path.startswith(f"{prefix}/")
        for prefix in (
            "lab_control",
            "integration/command_registries",
            "integration/host_profiles",
            "integration/preparation_profiles",
            "integration/remote_profiles",
            "integration/service_adapters",
            "stack/workload-selections",
        )
    )


def _run_bytes(repository: Path, arguments: list[str]) -> bytes:
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
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise LabError(f"control-plane package Git operation failed: {detail}")
    return completed.stdout


def _package_entries(repository: Path, commit: str) -> list[dict[str, object]]:
    raw = _run_bytes(
        repository,
        ["ls-tree", "-r", "-z", "--long", commit, "--", *PACKAGE_PATHS],
    )
    entries: list[dict[str, object]] = []
    total = 0
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, raw_oid, raw_size = metadata.split(b" ", 3)
            path = raw_path.decode("utf-8")
            oid = raw_oid.decode("ascii")
            size = int(raw_size)
        except (ValueError, UnicodeDecodeError) as exc:
            raise LabError("cannot parse control-plane Git tree entry") from exc
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            raise LabError(f"control-plane package has unsafe tree entry: {path}")
        if size > MAX_FILE_BYTES:
            raise LabError(f"control-plane package file exceeds 1 MiB: {path}")
        content = _run_bytes(repository, ["cat-file", "blob", oid])
        if len(content) != size:
            raise LabError(f"control-plane package blob size mismatch: {path}")
        total += size
        if total > MAX_CONTENT_BYTES:
            raise LabError("control-plane package source exceeds 8 MiB")
        entries.append(
            {
                "path": path,
                "mode": mode.decode("ascii"),
                "git_blob_oid": oid,
                "size_bytes": size,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    required = {"projectctl", "lab_control/project_cli.py"}
    paths = {str(entry["path"]) for entry in entries}
    if not required <= paths:
        raise LabError(
            f"control-plane package lacks required files: {sorted(required - paths)}"
        )
    return sorted(entries, key=lambda entry: str(entry["path"]))


def build_control_plane_package(
    repository: Path,
    *,
    expected_identity: str,
    output_path: Path,
) -> dict[str, object]:
    """Build an inert small archive only from the exact protected main checkout."""

    requested = repository.expanduser()
    root = requested.resolve()
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or requested.absolute() != root
        or git.repository_root(root) != root
    ):
        raise LabError(f"control-plane package requires a real repository root: {root}")
    if git.origin_identity(root) != expected_identity:
        raise LabError("control-plane package canonical identity mismatch")
    if git.current_branch(root) != "main" or git.status_short(root):
        raise LabError("control-plane package requires a clean main checkout")
    if not git.ref_exists(root, PROTECTED_REF):
        raise LabError("control-plane package protected ref is missing")
    commit = git.current_commit(root)
    protected_commit = git.resolve_commit(root, PROTECTED_REF)
    if commit != protected_commit:
        raise LabError("control-plane package HEAD is not exact protected origin/main")
    tree_oid = git.tree_oid(root, commit)
    entries = _package_entries(root, commit)

    expanded_output = output_path.expanduser()
    output = expanded_output.resolve()
    if (
        not expanded_output.is_absolute()
        or expanded_output.is_symlink()
        or expanded_output.absolute() != output
        or output.exists()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
    ):
        raise LabError(f"control-plane package output is unsafe: {output}")
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "tar.umask=0022",
                "-C",
                str(root),
                "archive",
                "--format=tar",
                f"--output={temporary}",
                commit,
                "--",
                *PACKAGE_PATHS,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise LabError(f"cannot build control-plane package: {detail}")
        archive_size = temporary.stat().st_size
        if not 1 <= archive_size <= MAX_ARCHIVE_BYTES:
            raise LabError("control-plane package archive size is invalid")
        archive_hasher = hashlib.sha256()
        with temporary.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                archive_hasher.update(chunk)
        archive_sha = archive_hasher.hexdigest()
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    content: dict[str, object] = {
        "schema": CONTROL_PLANE_PACKAGE_SCHEMA_ID,
        "canonical_identity": expected_identity,
        "protected_ref": PROTECTED_REF,
        "commit": commit,
        "tree_oid": tree_oid,
        "entries": entries,
        "archive": {
            "name": output.name,
            "size_bytes": output.stat().st_size,
            "sha256": archive_sha,
        },
        "network_used": False,
        "working_tree_content_used": False,
        "task_payload_content_used": False,
        "lfs_smudge_enabled": False,
        "large_files_included": False,
        "execution_authorized": False,
        "device_access": False,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def validate_control_plane_package_receipt(
    receipt: object,
    *,
    expected_identity: str,
    expected_commit: str,
) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "canonical_identity",
        "protected_ref",
        "commit",
        "tree_oid",
        "entries",
        "archive",
        "network_used",
        "working_tree_content_used",
        "task_payload_content_used",
        "lfs_smudge_enabled",
        "large_files_included",
        "execution_authorized",
        "device_access",
        "receipt_sha256",
    }:
        raise LabError("control-plane package receipt has an unexpected shape")
    if (
        receipt.get("schema") != CONTROL_PLANE_PACKAGE_SCHEMA_ID
        or receipt.get("canonical_identity") != expected_identity
        or receipt.get("protected_ref") != PROTECTED_REF
        or receipt.get("commit") != expected_commit
        or not OID_PATTERN.fullmatch(str(receipt.get("commit", "")))
        or not OID_PATTERN.fullmatch(str(receipt.get("tree_oid", "")))
    ):
        raise LabError("control-plane package receipt identity mismatch")
    if (
        receipt.get("network_used") is not False
        or receipt.get("working_tree_content_used") is not False
        or receipt.get("task_payload_content_used") is not False
        or receipt.get("lfs_smudge_enabled") is not False
        or receipt.get("large_files_included") is not False
        or receipt.get("execution_authorized") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("control-plane package receipt has an unsafe authority envelope")
    archive = receipt.get("archive")
    if not isinstance(archive, dict) or set(archive) != {
        "name",
        "size_bytes",
        "sha256",
    }:
        raise LabError("control-plane package receipt has invalid archive identity")
    archive_name = archive.get("name")
    archive_size = archive.get("size_bytes")
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
        or not archive_name
        or not isinstance(archive_size, int)
        or isinstance(archive_size, bool)
        or not 1 <= archive_size <= MAX_ARCHIVE_BYTES
        or not SHA256_PATTERN.fullmatch(str(archive.get("sha256", "")))
    ):
        raise LabError("control-plane package receipt has invalid archive identity")
    entries = receipt.get("entries")
    if not isinstance(entries, list) or not entries:
        raise LabError("control-plane package receipt has invalid entries")
    seen: set[str] = set()
    ordered_paths: list[str] = []
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "mode",
            "git_blob_oid",
            "size_bytes",
            "sha256",
        }:
            raise LabError("control-plane package receipt has invalid entry")
        path = entry.get("path")
        size = entry.get("size_bytes")
        if (
            not isinstance(path, str)
            or not _allowed_package_path(path)
            or path in seen
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not 0 <= size <= MAX_FILE_BYTES
            or entry.get("mode") not in {"100644", "100755"}
            or not OID_PATTERN.fullmatch(str(entry.get("git_blob_oid", "")))
            or not SHA256_PATTERN.fullmatch(str(entry.get("sha256", "")))
        ):
            raise LabError(f"control-plane package receipt entry is invalid: {path}")
        seen.add(path)
        ordered_paths.append(path)
        total += size
    if total > MAX_CONTENT_BYTES or ordered_paths != sorted(ordered_paths):
        raise LabError("control-plane package receipt content is invalid")
    required = {"projectctl", "lab_control/project_cli.py"}
    if not required <= seen:
        raise LabError("control-plane package receipt lacks required files")
    digest = receipt.get("receipt_sha256")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        not SHA256_PATTERN.fullmatch(str(digest))
        or digest != hashlib.sha256(_canonical_json(content)).hexdigest()
    ):
        raise LabError("control-plane package receipt hash mismatch")
    return receipt


def verify_control_plane_package(
    archive_path: Path,
    receipt: object,
    *,
    expected_identity: str,
    expected_commit: str,
) -> dict[str, object]:
    """Verify a transferred package without extracting or executing it."""

    validated = validate_control_plane_package_receipt(
        receipt,
        expected_identity=expected_identity,
        expected_commit=expected_commit,
    )
    archive_record = validated["archive"]
    if not isinstance(archive_record, dict):
        raise LabError("control-plane package receipt lacks archive identity")
    expanded = archive_path.expanduser()
    archive = expanded.resolve()
    if (
        not expanded.is_absolute()
        or expanded.is_symlink()
        or expanded.absolute() != archive
        or archive.name != archive_record["name"]
    ):
        raise LabError("control-plane package archive identity mismatch")
    raw_entries = validated["entries"]
    if not isinstance(raw_entries, list):
        raise LabError("control-plane package receipt lacks entries")
    expected = {
        str(entry["path"]): entry for entry in raw_entries if isinstance(entry, dict)
    }
    observed: set[str] = set()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(archive, flags)
    except OSError as exc:
        raise LabError(f"cannot open control-plane package archive: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as archive_stream:
            metadata = os.fstat(archive_stream.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != archive_record["size_bytes"]
                or not 1 <= metadata.st_size <= MAX_ARCHIVE_BYTES
            ):
                raise LabError("control-plane package archive identity mismatch")
            archive_hasher = hashlib.sha256()
            for chunk in iter(lambda: archive_stream.read(1024 * 1024), b""):
                archive_hasher.update(chunk)
            if archive_hasher.hexdigest() != archive_record["sha256"]:
                raise LabError("control-plane package archive identity mismatch")
            archive_stream.seek(0)
            with tarfile.open(fileobj=archive_stream, mode="r:") as package:
                for member in package.getmembers():
                    pure = PurePosixPath(member.name)
                    if (
                        member.name.startswith(("/", "~"))
                        or "\\" in member.name
                        or any(part in {"", ".", ".."} for part in pure.parts)
                    ):
                        raise LabError(
                            f"control-plane package has unsafe archive path: "
                            f"{member.name}"
                        )
                    if member.isdir():
                        continue
                    if not member.isfile() or member.name not in expected:
                        raise LabError(
                            f"control-plane package has unexpected archive member: "
                            f"{member.name}"
                        )
                    if member.name in observed:
                        raise LabError(
                            f"control-plane package repeats archive member: {member.name}"
                        )
                    entry = expected[member.name]
                    expected_mode = 0o755 if entry["mode"] == "100755" else 0o644
                    if member.size != entry["size_bytes"] or member.mode != expected_mode:
                        raise LabError(
                            f"control-plane package member metadata mismatch: "
                            f"{member.name}"
                        )
                    stream = package.extractfile(member)
                    if stream is None:
                        raise LabError(
                            f"control-plane package cannot read member: {member.name}"
                        )
                    member_content = stream.read(MAX_FILE_BYTES + 1)
                    if (
                        len(member_content) != entry["size_bytes"]
                        or hashlib.sha256(member_content).hexdigest()
                        != entry["sha256"]
                    ):
                        raise LabError(
                            f"control-plane package member content mismatch: "
                            f"{member.name}"
                        )
                    observed.add(member.name)
    except (tarfile.TarError, OSError) as exc:
        raise LabError(f"cannot inspect control-plane package archive: {exc}") from exc
    if observed != set(expected):
        raise LabError(
            "control-plane package archive is incomplete: "
            f"missing={sorted(set(expected) - observed)}"
        )
    content = {
        "schema": "lab.control-plane-package-verification.v1",
        "package_receipt_sha256": validated["receipt_sha256"],
        "canonical_identity": expected_identity,
        "commit": expected_commit,
        "tree_oid": validated["tree_oid"],
        "archive_sha256": archive_record["sha256"],
        "entry_count": len(observed),
        "verified": True,
        "extracted": False,
        "executed": False,
        "device_access": False,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }
