from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import git
from .workspace import LabError, _atomic_write_text

CONTROL_INSTALL_PLAN_SCHEMA_ID = "lab.control-install-plan.v1"
CONTROL_INSTALL_MANIFEST_SCHEMA_ID = "lab.control-install-manifest.v1"
CONTROL_INSTALL_RECEIPT_SCHEMA_ID = "lab.control-install-receipt.v1"
PAYLOAD_PATHS = ("lab", "dev", "lab_control", "integration", "stack")
REQUIRED_PAYLOAD_FILES = (
    "lab",
    "lab_control/__init__.py",
    "integration/project.json",
    "stack/baseline.json",
)
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_PAYLOAD_MEMBERS = 5000
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _safe_install_root(path: Path, *, create: bool) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise LabError(f"control-plane install root is unsafe: {requested}")
    if create:
        requested.mkdir(parents=True, exist_ok=True, mode=0o700)
    if requested.exists() and not requested.is_dir():
        raise LabError(f"control-plane install root is not a directory: {requested}")
    resolved = requested.resolve()
    if requested.absolute() != resolved:
        raise LabError(f"control-plane install root is not canonical: {requested}")
    return resolved


def _validate_outside(repository: Path, install_root: Path) -> None:
    if install_root.is_relative_to(repository) or repository.is_relative_to(install_root):
        raise LabError("control-plane install root must be outside the managed checkout")


def _payload_names(repository: Path, commit: str) -> tuple[str, ...]:
    result = git.run_git(
        repository,
        ["ls-tree", "-r", "--name-only", commit, "--", *PAYLOAD_PATHS],
    )
    names = tuple(line for line in result.stdout.splitlines() if line)
    missing = sorted(set(REQUIRED_PAYLOAD_FILES) - set(names))
    if missing:
        raise LabError("protected control-plane payload is incomplete: " + ", ".join(missing))
    if not names:
        raise LabError("protected control-plane payload is empty")
    return names


def _current_commit(install_root: Path) -> str | None:
    current = install_root / "current"
    if not current.exists() and not current.is_symlink():
        return None
    if not current.is_symlink():
        raise LabError("control-plane current path is not a symlink")
    target = current.resolve()
    versions = (install_root / "versions").resolve()
    if not target.is_relative_to(versions) or target.parent != versions:
        raise LabError("control-plane current symlink escapes versions root")
    if not re.fullmatch(r"[0-9a-f]{40}", target.name):
        raise LabError("control-plane current symlink has an invalid version")
    return target.name


def plan_control_plane_install(
    repository: Path,
    *,
    managed_control_repository: Path,
    expected_identity: str,
    protected_branch: str,
    install_root: Path,
    python_executable: Path | None = None,
) -> dict[str, object]:
    """Plan an external control payload from an already-fetched protected ref."""

    root = git.repository_root(repository)
    managed = git.repository_root(managed_control_repository)
    if root != repository.expanduser().resolve() or managed != managed_control_repository.expanduser().resolve():
        raise LabError("control-plane install requires exact repository roots")
    if git.origin_identity(root) != expected_identity or git.origin_identity(managed) != expected_identity:
        raise LabError("control-plane install canonical repository identity mismatch")
    remote_ref = f"refs/remotes/origin/{protected_branch}"
    if not git.ref_exists(root, remote_ref):
        raise LabError("protected control-plane remote ref is unavailable")
    commit = git.resolve_commit(root, remote_ref)
    tree_oid = git.tree_oid(root, commit)
    names = _payload_names(root, commit)
    install = _safe_install_root(install_root, create=False)
    _validate_outside(managed, install)
    executable = (
        Path(sys.executable).resolve()
        if python_executable is None
        else python_executable.expanduser().resolve()
    )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise LabError(f"control-plane Python executable is unavailable: {executable}")
    version_path = install / "versions" / commit
    existing_manifest = version_path / "install.json"
    current_commit = _current_commit(install) if install.exists() else None
    if current_commit == commit and existing_manifest.is_file():
        action = "NOOP"
    elif version_path.exists():
        action = "ACTIVATE_READY"
    else:
        action = "INSTALL_READY"
    content: dict[str, object] = {
        "schema": CONTROL_INSTALL_PLAN_SCHEMA_ID,
        "canonical_identity": expected_identity,
        "source_repository": str(root),
        "managed_control_repository": str(managed),
        "protected_branch": protected_branch,
        "protected_remote_ref": remote_ref,
        "commit": commit,
        "tree_oid": tree_oid,
        "payload_paths": list(PAYLOAD_PATHS),
        "payload_file_count": len(names),
        "install_root": str(install),
        "version_path": str(version_path),
        "python_executable": str(executable),
        "current_commit": current_commit,
        "action": action,
        "network_used": False,
        "task_payload_authorized": False,
        "lfs_smudge_enabled": False,
        "large_assets_downloaded": False,
    }
    return {**content, "plan_sha256": _hash(content)}


def validate_control_install_plan(plan: object) -> dict[str, object]:
    if not isinstance(plan, dict) or set(plan) != {
        "schema",
        "canonical_identity",
        "source_repository",
        "managed_control_repository",
        "protected_branch",
        "protected_remote_ref",
        "commit",
        "tree_oid",
        "payload_paths",
        "payload_file_count",
        "install_root",
        "version_path",
        "python_executable",
        "current_commit",
        "action",
        "network_used",
        "task_payload_authorized",
        "lfs_smudge_enabled",
        "large_assets_downloaded",
        "plan_sha256",
    }:
        raise LabError("control-plane install plan has an unexpected shape")
    content = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema") != CONTROL_INSTALL_PLAN_SCHEMA_ID
        or plan.get("payload_paths") != list(PAYLOAD_PATHS)
        or plan.get("network_used") is not False
        or plan.get("task_payload_authorized") is not False
        or plan.get("lfs_smudge_enabled") is not False
        or plan.get("large_assets_downloaded") is not False
        or plan.get("plan_sha256") != _hash(content)
    ):
        raise LabError("control-plane install plan is invalid or unsafe")
    return plan


def _archive_payload(plan: dict[str, object], archive_path: Path) -> None:
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-C",
        str(plan["source_repository"]),
        "archive",
        "--format=tar",
        str(plan["commit"]),
        "--",
        *PAYLOAD_PATHS,
    ]
    environment = dict(os.environ)
    environment.update({"GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"})
    with archive_path.open("wb") as stream:
        completed = subprocess.run(
            command,
            check=False,
            stdout=stream,
            stderr=subprocess.PIPE,
            env=environment,
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()[:1000]
        raise LabError(f"control-plane Git archive failed: {detail}")
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise LabError("control-plane archive exceeds the size limit")


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if name.startswith(("/", "~")) or any(part in {"", ".", ".."} for part in path.parts):
        raise LabError(f"control-plane archive has an unsafe path: {name!r}")
    if path.parts[0] not in PAYLOAD_PATHS:
        raise LabError(f"control-plane archive escapes payload roots: {name!r}")
    return path


def _extract_payload(archive_path: Path, payload_root: Path) -> dict[str, object]:
    files: dict[str, object] = {}
    total_bytes = 0
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        if len(members) > MAX_PAYLOAD_MEMBERS:
            raise LabError("control-plane payload has too many archive members")
        for member in members:
            relative = _safe_member_path(member.name)
            target = payload_root.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o755)
                continue
            if not member.isfile():
                raise LabError("control-plane payload may contain only files/directories")
            total_bytes += member.size
            if total_bytes > MAX_PAYLOAD_BYTES:
                raise LabError("control-plane payload exceeds the extracted size limit")
            source = archive.extractfile(member)
            if source is None:
                raise LabError(f"cannot read control-plane payload member: {member.name}")
            data = source.read(MAX_PAYLOAD_BYTES + 1)
            if len(data) != member.size:
                raise LabError(f"control-plane payload member size mismatch: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            target.write_bytes(data)
            executable = bool(member.mode & 0o111)
            target.chmod(0o555 if executable else 0o444)
            files[relative.as_posix()] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
                "executable": executable,
            }
    missing = sorted(set(REQUIRED_PAYLOAD_FILES) - set(files))
    if missing:
        raise LabError("extracted control-plane payload is incomplete: " + ", ".join(missing))
    return {
        "files": dict(sorted(files.items())),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def _write_wrapper(install_root: Path, plan: dict[str, object]) -> Path:
    wrapper = install_root / "bin" / "lab"
    python = shlex.quote(str(plan["python_executable"]))
    root = shlex.quote(str(install_root))
    managed = shlex.quote(str(plan["managed_control_repository"]))
    content = f"""#!/bin/sh
set -eu
CONTROL_ROOT={root}
export LAB_CONTROL_REPOSITORY_ROOT={managed}
export LAB_PROJECT_MANIFEST="$CONTROL_ROOT/current/payload/integration/project.json"
export LAB_BASELINE_LOCK="$CONTROL_ROOT/current/payload/stack/baseline.json"
export LAB_PROFILE_ROOT="$CONTROL_ROOT/current/payload/integration/profiles"
export LAB_QUEUE_PROFILE_ROOT="$CONTROL_ROOT/current/payload/integration/queue-profiles"
export LAB_RUNNER_PROFILE_ROOT="$CONTROL_ROOT/current/payload/integration/runners"
export LAB_ENVIRONMENT_PROFILE_ROOT="$CONTROL_ROOT/current/payload/integration/environments"
export LAB_IMPACT_RULES="$CONTROL_ROOT/current/payload/stack/impact-rules.json"
export LAB_CONTROL_PLANE_INSTALL_MANIFEST="$CONTROL_ROOT/current/install.json"
export PYTHONDONTWRITEBYTECODE=1
exec {python} "$CONTROL_ROOT/current/payload/lab" "$@"
"""
    if wrapper.parent.exists() and (
        wrapper.parent.is_symlink() or not wrapper.parent.is_dir()
    ):
        raise LabError("control-plane wrapper directory is unsafe")
    wrapper.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if wrapper.is_symlink() or (wrapper.exists() and not wrapper.is_file()):
        raise LabError("control-plane wrapper path is unsafe")
    _atomic_write_text(wrapper, content)
    wrapper.chmod(0o555)
    return wrapper


def _activate(
    install_root: Path,
    commit: str,
    *,
    expected_current_commit: object,
) -> None:
    current = install_root / "current"
    if _current_commit(install_root) != expected_current_commit:
        raise LabError("control-plane current version changed before activation")
    temporary = install_root / f".current-{commit}"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(Path("versions") / commit)
    os.replace(temporary, current)


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read control-plane install manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError("control-plane install manifest must be an object")
    content = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if (
        payload.get("schema") != CONTROL_INSTALL_MANIFEST_SCHEMA_ID
        or payload.get("manifest_sha256") != _hash(content)
    ):
        raise LabError("control-plane install manifest hash mismatch")
    return payload


@contextmanager
def _install_lock(install_root: Path) -> Iterator[None]:
    lock_path = install_root / ".install.lock"
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise LabError("control-plane install lock path is unsafe")
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _verify_payload(version_path: Path, manifest: dict[str, object]) -> None:
    raw_payload = manifest.get("payload")
    if not isinstance(raw_payload, dict):
        raise LabError("control-plane install manifest lacks payload identity")
    raw_files = raw_payload.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise LabError("control-plane install manifest has invalid payload files")
    payload_root = version_path / "payload"
    if payload_root.is_symlink() or not payload_root.is_dir():
        raise LabError("control-plane installed payload root is unsafe")
    if payload_root.stat().st_mode & 0o222:
        raise LabError("control-plane installed payload root is writable")
    observed_files: set[str] = set()
    for path in payload_root.rglob("*"):
        if path.is_symlink():
            raise LabError(f"control-plane installed payload has a symlink: {path}")
        if path.is_file():
            observed_files.add(path.relative_to(payload_root).as_posix())
        elif path.is_dir():
            if path.stat().st_mode & 0o222:
                raise LabError(
                    f"control-plane installed payload directory is writable: {path}"
                )
        else:
            raise LabError(f"control-plane installed payload has an unsafe entry: {path}")
    if observed_files != set(raw_files):
        raise LabError("control-plane installed payload file set mismatch")
    total_bytes = 0
    for name, raw_identity in raw_files.items():
        if not isinstance(name, str) or not isinstance(raw_identity, dict):
            raise LabError("control-plane installed payload identity is invalid")
        relative = _safe_member_path(name)
        path = payload_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise LabError(f"control-plane installed payload file is unsafe: {name}")
        data = path.read_bytes()
        total_bytes += len(data)
        mode = path.stat().st_mode
        executable = bool(mode & 0o111)
        if (
            raw_identity.get("size") != len(data)
            or raw_identity.get("sha256") != hashlib.sha256(data).hexdigest()
            or raw_identity.get("executable") is not executable
            or bool(mode & 0o222)
        ):
            raise LabError(f"control-plane installed payload hash mismatch: {name}")
    if (
        raw_payload.get("file_count") != len(observed_files)
        or raw_payload.get("total_bytes") != total_bytes
    ):
        raise LabError("control-plane installed payload summary mismatch")


def install_control_plane(
    plan: object,
    *,
    confirmed_plan_sha256: str,
) -> dict[str, object]:
    validated = validate_control_install_plan(plan)
    if validated.get("plan_sha256") != confirmed_plan_sha256:
        raise LabError("confirmed control-plane install plan hash does not match")
    install_root = _safe_install_root(
        Path(str(validated["install_root"])),
        create=True,
    )
    with _install_lock(install_root):
        return _install_control_plane_locked(validated)


def _install_control_plane_locked(
    validated: dict[str, object],
) -> dict[str, object]:
    fresh = plan_control_plane_install(
        Path(str(validated["source_repository"])),
        managed_control_repository=Path(str(validated["managed_control_repository"])),
        expected_identity=str(validated["canonical_identity"]),
        protected_branch=str(validated["protected_branch"]),
        install_root=Path(str(validated["install_root"])),
        python_executable=Path(str(validated["python_executable"])),
    )
    if fresh != validated:
        raise LabError("control-plane install state changed after planning")
    install_root = _safe_install_root(Path(str(validated["install_root"])), create=True)
    versions = install_root / "versions"
    if versions.exists() and (versions.is_symlink() or not versions.is_dir()):
        raise LabError("control-plane versions root is unsafe")
    versions.mkdir(mode=0o700, exist_ok=True)
    commit = str(validated["commit"])
    version_path = versions / commit
    if version_path.is_symlink() or (
        version_path.exists() and not version_path.is_dir()
    ):
        raise LabError("control-plane version path is unsafe")
    installed_new = False
    if not version_path.exists():
        staging = Path(tempfile.mkdtemp(prefix=f".{commit}.", dir=versions))
        archive_path = staging / "payload.tar"
        payload_root = staging / "payload"
        payload_root.mkdir(mode=0o755)
        try:
            _archive_payload(validated, archive_path)
            payload = _extract_payload(archive_path, payload_root)
            archive_path.unlink()
            installed_at = datetime.now(timezone.utc).isoformat()
            manifest_content: dict[str, object] = {
                "schema": CONTROL_INSTALL_MANIFEST_SCHEMA_ID,
                "canonical_identity": validated["canonical_identity"],
                "protected_branch": validated["protected_branch"],
                "protected_remote_ref": validated["protected_remote_ref"],
                "commit": commit,
                "tree_oid": validated["tree_oid"],
                "managed_control_repository": validated["managed_control_repository"],
                "python_executable": validated["python_executable"],
                "payload": payload,
                "installed_at": installed_at,
                "network_used": False,
                "task_payload_authorized": False,
                "lfs_smudge_enabled": False,
                "large_assets_downloaded": False,
            }
            manifest = {
                **manifest_content,
                "manifest_sha256": _hash(manifest_content),
            }
            manifest_path = staging / "install.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_path.chmod(0o444)
            for directory in sorted(
                (path for path in payload_root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                directory.chmod(0o555)
            payload_root.chmod(0o555)
            os.replace(staging, version_path)
            version_path.chmod(0o555)
            installed_new = True
        finally:
            if staging.exists():
                for path in sorted(
                    staging.rglob("*"),
                    key=lambda item: len(item.parts),
                    reverse=True,
                ):
                    if path.is_file() or path.is_symlink():
                        path.chmod(0o600)
                        path.unlink()
                    elif path.is_dir():
                        path.chmod(0o700)
                        path.rmdir()
                staging.rmdir()
    manifest = _load_manifest(version_path / "install.json")
    if (
        manifest.get("commit") != commit
        or manifest.get("tree_oid") != validated.get("tree_oid")
        or manifest.get("canonical_identity") != validated.get("canonical_identity")
        or manifest.get("managed_control_repository")
        != validated.get("managed_control_repository")
    ):
        raise LabError("existing control-plane version identity mismatch")
    _verify_payload(version_path, manifest)
    wrapper = _write_wrapper(install_root, validated)
    _activate(
        install_root,
        commit,
        expected_current_commit=validated.get("current_commit"),
    )
    receipt_content: dict[str, object] = {
        "schema": CONTROL_INSTALL_RECEIPT_SCHEMA_ID,
        "plan_sha256": validated["plan_sha256"],
        "install_root": str(install_root),
        "version_path": str(version_path),
        "current_path": str(install_root / "current"),
        "wrapper_path": str(wrapper),
        "commit": commit,
        "tree_oid": validated["tree_oid"],
        "install_manifest_sha256": manifest["manifest_sha256"],
        "installed_new": installed_new,
        "activated": True,
        "network_used": False,
        "task_payload_authorized": False,
        "lfs_smudge_enabled": False,
        "large_assets_downloaded": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    return {**receipt_content, "receipt_sha256": _hash(receipt_content)}


def inspect_control_plane_install(install_root: Path) -> dict[str, object]:
    root = _safe_install_root(install_root, create=False)
    commit = _current_commit(root)
    if commit is None:
        return {
            "schema": "lab.control-install-status.v1",
            "install_root": str(root),
            "status": "NOT_INSTALLED",
            "current_commit": None,
        }
    manifest = _load_manifest(root / "versions" / commit / "install.json")
    _verify_payload(root / "versions" / commit, manifest)
    return {
        "schema": "lab.control-install-status.v1",
        "install_root": str(root),
        "status": "READY",
        "current_commit": commit,
        "manifest": manifest,
        "wrapper_path": str(root / "bin" / "lab"),
    }
