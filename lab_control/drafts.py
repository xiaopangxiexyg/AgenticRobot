from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from . import git
from .workspace import LabError, WorkspaceManager, _atomic_write_text
from .writer_lease import WriterLeaseManager, writer_lease_fingerprint

DRAFT_SCHEMA_ID = "lab.draft-snapshot.v1"
DEFAULT_MAX_FILE_BYTES = 1_048_576
DEFAULT_TTL_DAYS = 7
_ZERO_OID = "0" * 40
_DENIED_COMPONENTS = frozenset(
    {".ssh", ".gnupg", ".aws", ".kube", "secret", "secrets", "credentials"}
)
_DENIED_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _run_git(
    repository: Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    input_text: str | None = None,
) -> str:
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-C",
        str(repository),
        *arguments,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        input=input_text,
        env=None if environment is None else dict(environment),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LabError(f"Git draft operation failed in {repository}: {detail}")
    return completed.stdout.rstrip()


def _run_git_bytes(repository: Path, arguments: Sequence[str]) -> bytes:
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
        raise LabError(f"Git draft inspection failed in {repository}: {detail}")
    return completed.stdout


def _nul_paths(repository: Path, arguments: Sequence[str]) -> tuple[str, ...]:
    raw = _run_git_bytes(repository, arguments)
    return tuple(
        value.decode("utf-8", errors="surrogateescape")
        for value in raw.split(b"\0")
        if value
    )


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith(("/", "~"))
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise LabError(f"unsafe draft path: {value!r}")
    return path


def _denied_untracked_path(value: str) -> bool:
    path = _safe_relative_path(value)
    lower_parts = tuple(part.casefold() for part in path.parts)
    name = lower_parts[-1]
    return (
        any(part in _DENIED_COMPONENTS for part in lower_parts)
        or name == ".env"
        or name.startswith(".env.")
        or name.endswith(_DENIED_SUFFIXES)
        or name in {"token", "tokens", "credentials.json"}
        or name.endswith((".token", ".tokens"))
    )


def _matches_any(value: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _validate_file_size(
    repository: Path,
    value: str,
    *,
    max_file_bytes: int,
    untracked: bool,
) -> None:
    path = repository / value
    if not path.exists() and not path.is_symlink():
        return
    if untracked and path.is_symlink():
        raise LabError(f"draft snapshot refuses untracked symlink: {value}")
    if path.is_file() and path.lstat().st_size > max_file_bytes:
        kind = "untracked" if untracked else "changed tracked"
        raise LabError(
            f"draft snapshot refuses {kind} file over {max_file_bytes} bytes: {value}"
        )


def _reject_git_filters(repository: Path, paths: Sequence[str]) -> None:
    if not paths:
        return
    raw = _run_git_bytes(repository, ["check-attr", "-z", "filter", "--", *paths])
    fields = raw.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 3 != 0:
        raise LabError(f"cannot parse Git filter attributes in {repository}")
    for index in range(0, len(fields), 3):
        path = fields[index].decode("utf-8", errors="surrogateescape")
        value = fields[index + 2].decode("utf-8", errors="replace")
        if value not in {"unspecified", "unset"}:
            raise LabError(
                f"draft snapshot refuses clean/smudge filter {value!r}: {path}"
            )


def _path_hash(repository: Path, value: str) -> str:
    path = repository / value
    digest = hashlib.sha256()
    digest.update(value.encode("utf-8", errors="surrogateescape"))
    digest.update(b"\0")
    digest.update(str(path.lstat().st_mode).encode())
    digest.update(b"\0")
    if path.is_symlink():
        digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
    else:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(
    repository: Path,
    included_untracked: Sequence[str],
) -> dict[str, object]:
    return {
        "branch": git.current_branch(repository),
        "head": git.current_commit(repository),
        "status": list(git.status_short(repository)),
        "working_diff_sha256": hashlib.sha256(
            _run_git_bytes(
                repository,
                ["diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--"],
            )
        ).hexdigest(),
        "index_diff_sha256": hashlib.sha256(
            _run_git_bytes(repository, ["diff", "--cached", "--binary", "HEAD", "--"])
        ).hexdigest(),
        "index_entries_sha256": hashlib.sha256(
            _run_git_bytes(repository, ["ls-files", "--stage", "-z"])
        ).hexdigest(),
        "included_untracked": {
            path: _path_hash(repository, path) for path in included_untracked
        },
    }


def _capture_repository(
    name: str,
    repository: Path,
    *,
    include_untracked: Sequence[str],
    max_file_bytes: int,
) -> tuple[dict[str, object], dict[str, object]]:
    head = git.current_commit(repository)
    head_tree = git.tree_oid(repository)
    submodules, submodules_exact = git.submodule_status(repository)
    if not submodules_exact:
        raise LabError(f"draft snapshot requires exact initialized submodules: {name}")
    if _run_git_bytes(repository, ["ls-files", "--unmerged", "-z"]):
        raise LabError(f"draft snapshot refuses unmerged paths: {name}")
    changed_tracked = _nul_paths(
        repository,
        ["diff", "--name-only", "--diff-filter=ACDMRTUXB", "-z", "HEAD", "--"],
    )
    untracked = _nul_paths(
        repository,
        ["ls-files", "--others", "--exclude-standard", "-z"],
    )
    included = tuple(
        path for path in untracked if _matches_any(path, include_untracked)
    )
    excluded = tuple(path for path in untracked if path not in set(included))
    for path in changed_tracked:
        _safe_relative_path(path)
        _validate_file_size(
            repository,
            path,
            max_file_bytes=max_file_bytes,
            untracked=False,
        )
    for path in included:
        if _denied_untracked_path(path):
            raise LabError(f"draft snapshot denylist rejects untracked path: {path}")
        _validate_file_size(
            repository,
            path,
            max_file_bytes=max_file_bytes,
            untracked=True,
        )
    selected_paths = (*changed_tracked, *included)
    _reject_git_filters(repository, selected_paths)
    before = _source_fingerprint(repository, included)
    if not selected_paths:
        return (
            {
                "repository": name,
                "branch": git.current_branch(repository),
                "base_commit": head,
                "commit": head,
                "tree_oid": head_tree,
                "synthetic": False,
                "changed_tracked": [],
                "included_untracked": [],
                "excluded_untracked": list(excluded),
                "submodules": submodules,
            },
            before,
        )

    with tempfile.TemporaryDirectory(prefix="lab-draft-index-") as temporary:
        index_path = Path(temporary) / "index"
        environment = dict(os.environ)
        environment.update(
            {
                "GIT_INDEX_FILE": str(index_path),
                "GIT_AUTHOR_NAME": "Lab Snapshot",
                "GIT_AUTHOR_EMAIL": "lab-snapshot@example.invalid",
                "GIT_COMMITTER_NAME": "Lab Snapshot",
                "GIT_COMMITTER_EMAIL": "lab-snapshot@example.invalid",
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
            }
        )
        _run_git(repository, ["read-tree", "HEAD"], environment=environment)
        _run_git(repository, ["add", "-u", "--", "."], environment=environment)
        if included:
            _run_git(
                repository,
                ["add", "--", *included],
                environment=environment,
            )
        tree_oid = _run_git(repository, ["write-tree"], environment=environment)
        commit = _run_git(
            repository,
            ["commit-tree", tree_oid, "-p", head],
            environment=environment,
            input_text=f"lab synthetic draft for {name}\n",
        )
    if len(commit) != 40 or len(tree_oid) != 40:
        raise LabError(f"Git returned an invalid synthetic identity for {name}")
    return (
        {
            "repository": name,
            "branch": git.current_branch(repository),
            "base_commit": head,
            "commit": commit,
            "tree_oid": tree_oid,
            "synthetic": True,
            "changed_tracked": list(changed_tracked),
            "included_untracked": list(included),
            "excluded_untracked": list(excluded),
            "submodules": submodules,
        },
        before,
    )


class DraftSnapshotManager:
    """Capture Task working content as native Git commits without moving user state."""

    def __init__(self, workspace: WorkspaceManager) -> None:
        self.workspace = workspace

    def create(
        self,
        task_id: str,
        *,
        include_untracked: Sequence[str] = (),
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> dict[str, object]:
        if not isinstance(max_file_bytes, int) or isinstance(max_file_bytes, bool):
            raise LabError("draft max file bytes must be an integer")
        if not 1 <= max_file_bytes <= 100 * 1024 * 1024:
            raise LabError("draft max file bytes must be between 1 and 104857600")
        patterns = tuple(include_untracked)
        for pattern in patterns:
            pattern_path = PurePosixPath(pattern)
            if (
                not pattern
                or pattern.startswith(("/", "~"))
                or "\\" in pattern
                or "\x00" in pattern
                or ".." in pattern_path.parts
            ):
                raise LabError(f"unsafe untracked include pattern: {pattern!r}")
        inspection = self.workspace.inspect_task(task_id)
        if inspection.get("status") != "ACTIVE":
            raise LabError(f"draft snapshot rejects Task drift: {task_id}")
        raw_repositories = inspection.get("repositories")
        if not isinstance(raw_repositories, dict) or not raw_repositories:
            raise LabError(f"Task inspection has no repositories: {task_id}")
        lock_keys = [
            f"{name}:{raw_repository.get('path')}"
            for name, raw_repository in raw_repositories.items()
            if isinstance(raw_repository, dict)
        ]
        writer_lock_keys = [
            f"{task_id}:{name}"
            for name in raw_repositories
            if isinstance(name, str)
        ]
        writer_manager = WriterLeaseManager(self.workspace)
        with (
            self.workspace.advisory_locks("writer-leases", writer_lock_keys),
            self.workspace.advisory_locks("repositories", lock_keys),
        ):
            writer_leases = writer_lease_fingerprint(
                writer_manager.inspect_task(task_id)
            )
            current = self.workspace.inspect_task(task_id)
            if current.get("status") != "ACTIVE":
                raise LabError(f"Task drifted before draft barrier: {task_id}")
            repositories = current.get("repositories")
            if not isinstance(repositories, dict):
                raise LabError(f"Task inspection has no repositories: {task_id}")

            def has_active_writer(name: str) -> bool:
                writer = writer_leases.get(name)
                return isinstance(writer, dict) and writer.get("state") == "ACTIVE"

            unleased = sorted(
                name
                for name, raw_repository in repositories.items()
                if isinstance(name, str)
                and isinstance(raw_repository, dict)
                and raw_repository.get("dirty") is True
                and not has_active_writer(name)
            )
            if unleased:
                raise LabError(
                    "draft snapshot requires an ACTIVE writer lease for changed "
                    f"repositories: {', '.join(unleased)}"
                )
            captured: dict[str, dict[str, object]] = {}
            fingerprints: dict[str, dict[str, object]] = {}
            paths: dict[str, Path] = {}
            for name, raw_repository in repositories.items():
                if not isinstance(name, str) or not isinstance(raw_repository, dict):
                    raise LabError(f"invalid Task repository inspection: {task_id}")
                path = Path(str(raw_repository.get("path", ""))).resolve()
                paths[name] = path
                captured[name], fingerprints[name] = _capture_repository(
                    name,
                    path,
                    include_untracked=patterns,
                    max_file_bytes=max_file_bytes,
                )
            if not any(repository["synthetic"] for repository in captured.values()):
                raise LabError(
                    "draft snapshot has no captured changes; commit clean work or include an allowed untracked file"
                )
            for name, path in paths.items():
                raw_included = captured[name].get("included_untracked")
                if not isinstance(raw_included, list) or not all(
                    isinstance(value, str) for value in raw_included
                ):
                    raise LabError(f"invalid captured untracked paths: {name}")
                if (
                    _source_fingerprint(
                        path,
                        tuple(raw_included),
                    )
                    != fingerprints[name]
                ):
                    raise LabError(f"Task source changed during draft barrier: {name}")
            final_writer_leases = writer_lease_fingerprint(
                writer_manager.inspect_task(task_id)
            )
            if final_writer_leases != writer_leases:
                raise LabError(
                    f"Task writer lease changed during draft barrier: {task_id}"
                )

            identity = {
                "schema": DRAFT_SCHEMA_ID,
                "task_id": task_id,
                "writer_leases": writer_leases,
                "policy": {
                    "include_untracked": list(patterns),
                    "max_file_bytes": max_file_bytes,
                    "denylist_version": 1,
                },
                "repositories": {
                    name: {
                        key: repository[key]
                        for key in (
                            "base_commit",
                            "commit",
                            "tree_oid",
                            "synthetic",
                            "changed_tracked",
                            "included_untracked",
                            "excluded_untracked",
                            "submodules",
                        )
                    }
                    for name, repository in sorted(captured.items())
                },
            }
            snapshot_id = (
                "draft-" + hashlib.sha256(_canonical_json(identity)).hexdigest()[:32]
            )
            created_refs: list[tuple[Path, str, str]] = []
            try:
                for name, repository in captured.items():
                    repository_ref_id = hashlib.sha256(name.encode()).hexdigest()[:16]
                    ref = f"refs/lab/snapshots/{snapshot_id}/{repository_ref_id}"
                    repository["ref"] = ref
                    path = paths[name]
                    commit = str(repository["commit"])
                    if git.ref_exists(path, ref):
                        if git.resolve_commit(path, ref) != commit:
                            raise LabError(f"draft ref collision: {ref}")
                    else:
                        _run_git(path, ["update-ref", ref, commit, _ZERO_OID])
                        created_refs.append((path, ref, commit))

                drafts_root = (self.workspace.state_root / "drafts").resolve()
                drafts_root.mkdir(parents=True, exist_ok=True)
                task_root = drafts_root / task_id
                if task_root.is_symlink():
                    raise LabError(f"draft Task state path is a symlink: {task_root}")
                snapshot_root = task_root / snapshot_id
                if snapshot_root.is_symlink():
                    raise LabError(
                        f"draft snapshot state path is a symlink: {snapshot_root}"
                    )
                snapshot_root.mkdir(parents=True, exist_ok=True)
                if not snapshot_root.resolve().is_relative_to(drafts_root):
                    raise LabError("draft snapshot state escapes configured state root")
                manifest_path = snapshot_root / "snapshot.json"
                if manifest_path.exists():
                    try:
                        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise LabError(
                            f"cannot read existing draft manifest: {exc}"
                        ) from exc
                    if (
                        not isinstance(existing, dict)
                        or existing.get("schema") != DRAFT_SCHEMA_ID
                        or existing.get("snapshot_id") != snapshot_id
                    ):
                        raise LabError(f"draft manifest collision: {manifest_path}")
                    return existing
                created_at = datetime.now(timezone.utc)
                manifest: dict[str, object] = {
                    **identity,
                    "snapshot_id": snapshot_id,
                    "snapshot_type": "synthetic-draft-git",
                    "created_at": created_at.isoformat(),
                    "expires_at": (
                        created_at + timedelta(days=DEFAULT_TTL_DAYS)
                    ).isoformat(),
                    "repositories": captured,
                    "remote_eligible": False,
                    "verify_eligible": False,
                    "device_eligible": False,
                }
                _atomic_write_text(
                    manifest_path,
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                )
                return manifest
            except Exception:
                for path, ref, commit in reversed(created_refs):
                    _run_git(path, ["update-ref", "-d", ref, commit])
                raise
