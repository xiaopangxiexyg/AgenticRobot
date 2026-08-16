from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import git
from .queue_profiles import QueueProfile
from .snapshot_resolution import resolve_clean_snapshot_identity
from .workspace import (
    LabError,
    WorkspaceManager,
    scope_matches,
    validate_repository_name,
    validate_submodule_path,
)

_OID_PATTERN = re.compile(r"[0-9a-f]{40}")
_DENIED_UNTRACKED_NAMES = re.compile(
    r"(^|[._-])(secret|secrets|credential|credentials|token|private[._-]?key)"
    r"([._-]|$)",
    re.IGNORECASE,
)
_DENIED_UNTRACKED_SUFFIXES = frozenset(
    {".key", ".pem", ".p12", ".pfx", ".jks", ".keystore"}
)
_DENIED_UNTRACKED_PARTS = frozenset({".aws", ".gnupg", ".ssh"})
_DENIED_UNTRACKED_FILENAMES = frozenset(
    {"id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"}
)
_MAXIMUM_UNTRACKED_FILES = 256
_MAXIMUM_BLOB_BYTES = 16 * 1024 * 1024
_MAXIMUM_TOTAL_BYTES = 128 * 1024 * 1024


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _run_git_bytes(
    repository: Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> bytes:
    process_environment = dict(os.environ)
    if environment is not None:
        process_environment.update(environment)
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            *arguments,
        ],
        input=input_bytes,
        capture_output=True,
        check=False,
        env=process_environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise LabError(f"Git draft snapshot command failed in {repository}: {detail}")
    return completed.stdout


def _safe_relative_path(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith(("/", "~"))
        or "\\" in value
        or "\x00" in value
        or value != path.as_posix()
        or any(part in {"", ".", "..", ".git"} for part in path.parts)
    ):
        raise LabError(f"unsafe {label}: {value!r}")
    return value


def _index_path(repository: Path) -> Path:
    raw = git.run_git(repository, ["rev-parse", "--git-path", "index"]).stdout.strip()
    if not raw:
        raise LabError(f"cannot resolve Git index path: {repository}")
    path = Path(raw)
    if not path.is_absolute():
        path = repository / path
    return path.resolve()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_fingerprint(repository: Path) -> dict[str, object]:
    return {
        "branch": git.current_branch(repository),
        "commit": git.current_commit(repository),
        "status": list(git.status_short(repository)),
        "index_sha256": _sha256_file(_index_path(repository)),
    }


def _tree_entries(repository: Path, commit: str) -> dict[str, tuple[str, str]]:
    output = _run_git_bytes(repository, ["ls-tree", "-r", "-z", "--full-tree", commit])
    entries: dict[str, tuple[str, str]] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise LabError(f"cannot parse Git tree entry in {repository}")
        mode, _, oid = fields
        try:
            path = _safe_relative_path(
                raw_path.decode("utf-8"), label="tracked source path"
            )
            mode_text = mode.decode("ascii")
            oid_text = oid.decode("ascii")
        except UnicodeDecodeError as exc:
            raise LabError("draft snapshot rejects non-UTF-8 tracked paths") from exc
        entries[path] = (mode_text, oid_text)
    return entries


def _index_entries(repository: Path) -> dict[str, tuple[str, str]]:
    output = _run_git_bytes(repository, ["ls-files", "--stage", "-z"])
    entries: dict[str, tuple[str, str]] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise LabError(f"cannot parse Git index entry in {repository}")
        mode, oid, stage = fields
        if stage != b"0":
            raise LabError("draft snapshot rejects an index with unresolved conflicts")
        try:
            path = _safe_relative_path(
                raw_path.decode("utf-8"), label="indexed source path"
            )
            mode_text = mode.decode("ascii")
            oid_text = oid.decode("ascii")
        except UnicodeDecodeError as exc:
            raise LabError("draft snapshot rejects non-UTF-8 indexed paths") from exc
        entries[path] = (mode_text, oid_text)
    return entries


def _untracked_paths(repository: Path) -> tuple[str, ...]:
    output = _run_git_bytes(
        repository, ["ls-files", "--others", "--exclude-standard", "-z"]
    )
    paths: list[str] = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        try:
            path = _safe_relative_path(
                raw_path.decode("utf-8"), label="untracked source path"
            )
        except UnicodeDecodeError as exc:
            raise LabError("draft snapshot rejects non-UTF-8 untracked paths") from exc
        paths.append(path)
    return tuple(sorted(paths))


def _read_worktree_entry(
    repository: Path,
    relative_path: str,
    *,
    indexed_mode: str | None,
    deny_untracked_secret: bool,
) -> tuple[str, bytes]:
    path = repository.joinpath(*PurePosixPath(relative_path).parts)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise LabError(
            f"source disappeared during draft snapshot: {relative_path}"
        ) from exc
    if deny_untracked_secret:
        name = path.name
        if (
            name in {".env", ".netrc"}
            or name.casefold() in _DENIED_UNTRACKED_FILENAMES
            or any(part.casefold() in _DENIED_UNTRACKED_PARTS for part in path.parts)
            or _DENIED_UNTRACKED_NAMES.search(name)
            or path.suffix.casefold() in _DENIED_UNTRACKED_SUFFIXES
        ):
            raise LabError(
                f"draft snapshot denies sensitive untracked path: {relative_path}"
            )
    if stat.S_ISLNK(before.st_mode):
        link_target = os.readlink(path)
        if not link_target or Path(link_target).is_absolute():
            raise LabError(f"draft snapshot rejects unsafe symlink: {relative_path}")
        resolved_target = (path.parent / link_target).resolve(strict=False)
        if not resolved_target.is_relative_to(repository.resolve()):
            raise LabError(
                f"draft snapshot symlink escapes repository: {relative_path}"
            )
        content = link_target.encode("utf-8")
        mode = "120000"
    elif stat.S_ISREG(before.st_mode):
        if before.st_size > _MAXIMUM_BLOB_BYTES:
            raise LabError(
                f"draft source exceeds {_MAXIMUM_BLOB_BYTES} bytes: {relative_path}"
            )
        content = path.read_bytes()
        mode = indexed_mode or "100644"
        if mode not in {"100644", "100755"}:
            mode = "100644"
        file_mode = git.run_git(
            repository, ["config", "--bool", "core.fileMode"], check=False
        )
        if file_mode.returncode == 0 and file_mode.stdout.strip() == "true":
            mode = "100755" if before.st_mode & stat.S_IXUSR else "100644"
    else:
        raise LabError(f"draft snapshot rejects non-file source: {relative_path}")
    after = path.lstat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise LabError(f"source changed while being captured: {relative_path}")
    return mode, content


def _update_index_entry(
    repository: Path,
    *,
    environment: Mapping[str, str],
    path: str,
    mode: str,
    oid: str,
) -> None:
    if not _OID_PATTERN.fullmatch(oid):
        raise LabError(f"invalid draft Git object for {path}: {oid!r}")
    git.run_git(
        repository,
        ["update-index", "--add", "--cacheinfo", f"{mode},{oid},{path}"],
        environment=environment,
    )


def _gitlink_worktree_dirty(repository: Path, path: str) -> bool:
    completed = git.run_git(
        repository,
        ["diff-files", "--quiet", "--ignore-submodules=none", "--", path],
        check=False,
    )
    if completed.returncode not in {0, 1}:
        detail = completed.stderr.strip()
        raise LabError(f"cannot inspect draft gitlink {path}: {detail}")
    return completed.returncode == 1


@dataclass(frozen=True)
class DraftRepositoryCapture:
    repository: str
    base_commit: str
    commit: str
    tree_oid: str
    submodules: Mapping[str, str]
    synthetic: bool
    included_untracked: tuple[str, ...]
    excluded_untracked: tuple[str, ...]

    def identity(self) -> dict[str, object]:
        return {
            "commit": self.commit,
            "tree_oid": self.tree_oid,
            "submodules": dict(self.submodules),
            "base_commit": self.base_commit,
            "synthetic": self.synthetic,
            "included_untracked": list(self.included_untracked),
            "excluded_untracked": list(self.excluded_untracked),
        }


def capture_repository_draft(
    repository_name: str,
    repository: Path,
    *,
    scopes: Sequence[str],
    include_untracked: Sequence[str] = (),
    gitlink_overrides: Mapping[str, str] | None = None,
    expected_fingerprint: Mapping[str, object] | None = None,
) -> DraftRepositoryCapture:
    repository_name = validate_repository_name(repository_name)
    root = git.repository_root(repository)
    fingerprint = repository_fingerprint(root)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise LabError(f"repository changed before draft capture: {repository_name}")
    base_commit = str(fingerprint["commit"])
    base_tree = git.tree_oid(root, base_commit)
    head_entries = _tree_entries(root, base_commit)
    index_entries = _index_entries(root)
    untracked = _untracked_paths(root)
    requested_untracked = tuple(
        _safe_relative_path(path, label="draft include path")
        for path in include_untracked
    )
    if len(requested_untracked) != len(set(requested_untracked)):
        raise LabError(
            f"draft untracked allowlist contains duplicates: {repository_name}"
        )
    if len(requested_untracked) > _MAXIMUM_UNTRACKED_FILES:
        raise LabError(
            f"draft snapshot supports at most {_MAXIMUM_UNTRACKED_FILES} "
            f"untracked files per repository"
        )
    missing = set(requested_untracked) - set(untracked)
    if missing:
        raise LabError(
            f"draft include paths are not untracked source in {repository_name}: "
            + ", ".join(sorted(missing))
        )
    for path in requested_untracked:
        if not scope_matches(path, scopes):
            raise LabError(
                f"draft untracked path exceeds Task write scope in "
                f"{repository_name}: {path}"
            )
    overrides = {
        validate_submodule_path(path): oid
        for path, oid in (gitlink_overrides or {}).items()
    }
    for path, oid in overrides.items():
        if not _OID_PATTERN.fullmatch(oid):
            raise LabError(f"invalid draft gitlink override: {path}={oid}")
        indexed = index_entries.get(path) or head_entries.get(path)
        if indexed is None or indexed[0] != "160000":
            raise LabError(f"draft gitlink override has no tracked gitlink: {path}")

    with tempfile.TemporaryDirectory(prefix="lab-draft-index-") as raw:
        index_path = Path(raw) / "index"
        environment = {
            "GIT_INDEX_FILE": str(index_path),
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
        git.run_git(root, ["read-tree", base_commit], environment=environment)
        total_bytes = 0
        for path in sorted(set(head_entries) | set(index_entries)):
            indexed = index_entries.get(path)
            if indexed is None:
                git.run_git(
                    root,
                    ["update-index", "--force-remove", "--", path],
                    environment=environment,
                )
                continue
            indexed_mode, indexed_oid = indexed
            if indexed_mode == "160000":
                if path not in overrides and _gitlink_worktree_dirty(root, path):
                    raise LabError(
                        f"draft snapshot cannot omit dirty linked repository: "
                        f"{repository_name}/{path}"
                    )
                oid = overrides.get(path, indexed_oid)
                _update_index_entry(
                    root,
                    environment=environment,
                    path=path,
                    mode="160000",
                    oid=oid,
                )
                continue
            source = root.joinpath(*PurePosixPath(path).parts)
            if not source.exists() and not source.is_symlink():
                git.run_git(
                    root,
                    ["update-index", "--force-remove", "--", path],
                    environment=environment,
                )
                continue
            mode, content = _read_worktree_entry(
                root,
                path,
                indexed_mode=indexed_mode,
                deny_untracked_secret=False,
            )
            total_bytes += len(content)
            if total_bytes > _MAXIMUM_TOTAL_BYTES:
                raise LabError(
                    f"draft source exceeds {_MAXIMUM_TOTAL_BYTES} total bytes: "
                    f"{repository_name}"
                )
            oid = (
                _run_git_bytes(
                    root,
                    ["hash-object", "-w", "--stdin"],
                    input_bytes=content,
                )
                .decode("ascii")
                .strip()
            )
            _update_index_entry(
                root, environment=environment, path=path, mode=mode, oid=oid
            )
        for path in requested_untracked:
            mode, content = _read_worktree_entry(
                root,
                path,
                indexed_mode=None,
                deny_untracked_secret=True,
            )
            total_bytes += len(content)
            if total_bytes > _MAXIMUM_TOTAL_BYTES:
                raise LabError(
                    f"draft source exceeds {_MAXIMUM_TOTAL_BYTES} total bytes: "
                    f"{repository_name}"
                )
            oid = (
                _run_git_bytes(
                    root,
                    ["hash-object", "-w", "--stdin"],
                    input_bytes=content,
                )
                .decode("ascii")
                .strip()
            )
            _update_index_entry(
                root, environment=environment, path=path, mode=mode, oid=oid
            )
        tree_oid = git.run_git(
            root, ["write-tree"], environment=environment
        ).stdout.strip()
    if not _OID_PATTERN.fullmatch(tree_oid):
        raise LabError(f"draft snapshot produced an invalid tree: {repository_name}")
    synthetic = tree_oid != base_tree
    if synthetic:
        parent_timestamp = git.run_git(
            root, ["show", "-s", "--format=%ct", base_commit]
        ).stdout.strip()
        parent_date = f"{parent_timestamp} +0000"
        commit_environment = {
            "GIT_AUTHOR_NAME": "Lab Draft Snapshot",
            "GIT_AUTHOR_EMAIL": "lab-draft@example.invalid",
            "GIT_COMMITTER_NAME": "Lab Draft Snapshot",
            "GIT_COMMITTER_EMAIL": "lab-draft@example.invalid",
            "GIT_AUTHOR_DATE": parent_date,
            "GIT_COMMITTER_DATE": parent_date,
        }
        message = (
            f"lab synthetic draft snapshot\n\nbase: {base_commit}\ntree: {tree_oid}\n"
        ).encode()
        commit = (
            _run_git_bytes(
                root,
                ["commit-tree", tree_oid, "-p", base_commit],
                environment=commit_environment,
                input_bytes=message,
            )
            .decode("ascii")
            .strip()
        )
    else:
        commit = base_commit
    if not _OID_PATTERN.fullmatch(commit):
        raise LabError(f"draft snapshot produced an invalid commit: {repository_name}")
    if repository_fingerprint(root) != fingerprint:
        raise LabError(f"repository changed during draft capture: {repository_name}")
    return DraftRepositoryCapture(
        repository=repository_name,
        base_commit=base_commit,
        commit=commit,
        tree_oid=tree_oid,
        submodules=git.submodule_gitlinks(root, commit),
        synthetic=synthetic,
        included_untracked=tuple(sorted(requested_untracked)),
        excluded_untracked=tuple(sorted(set(untracked) - set(requested_untracked))),
    )


def _task_links(
    task_manifest: Mapping[str, object], repositories: set[str]
) -> tuple[tuple[str, str, str], ...]:
    raw_repositories = task_manifest.get("repositories")
    if not isinstance(raw_repositories, dict):
        raise LabError("Task manifest has no repositories")
    links: list[tuple[str, str, str]] = []
    for child, raw_repository in raw_repositories.items():
        if child not in repositories or not isinstance(raw_repository, dict):
            continue
        placement = raw_repository.get("workspace_placement")
        if placement is None:
            continue
        if not isinstance(placement, dict):
            raise LabError(f"invalid Task repository placement: {child}")
        parent = validate_repository_name(str(placement.get("parent_repository", "")))
        path = validate_submodule_path(str(placement.get("submodule_path", "")))
        if parent in repositories:
            links.append((parent, path, validate_repository_name(str(child))))
    return tuple(sorted(links))


def _child_first_order(
    repositories: set[str], links: Sequence[tuple[str, str, str]]
) -> tuple[str, ...]:
    children: dict[str, set[str]] = {}
    for parent, _, child in links:
        children.setdefault(parent, set()).add(child)
    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def visit(name: str) -> None:
        if name in visiting:
            raise LabError("Task repository links contain a cycle")
        if name in visited:
            return
        visiting.add(name)
        for child in sorted(children.get(name, set())):
            visit(child)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in sorted(repositories):
        visit(name)
    return tuple(order)


def create_draft_snapshot_identity(
    workspace_manager: WorkspaceManager,
    *,
    task_id: str,
    profile: QueueProfile,
    baseline_path: Path,
    include_untracked: Mapping[str, Sequence[str]],
) -> tuple[str, dict[str, object]]:
    if "draft" not in profile.snapshot_modes:
        raise LabError(
            f"queue Profile does not allow draft snapshots: {profile.profile_id}"
        )
    inspection = workspace_manager.inspect_task(task_id)
    if inspection.get("status") != "ACTIVE":
        raise LabError(f"Task {task_id} has branch or scope drift")
    raw_inspections = inspection.get("repositories")
    if not isinstance(raw_inspections, dict):
        raise LabError(f"Task {task_id} inspection has no repositories")
    task_manifest = workspace_manager.load_task(task_id)
    requested = set(profile.repositories)
    task_repositories = requested & set(raw_inspections)
    unknown_includes = set(include_untracked) - task_repositories
    if unknown_includes:
        raise LabError(
            "draft untracked allowlist references unavailable Task repositories: "
            + ", ".join(sorted(unknown_includes))
        )
    links = _task_links(task_manifest, task_repositories)
    order = _child_first_order(task_repositories, links)
    fingerprints: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    scopes: dict[str, tuple[str, ...]] = {}
    for name in task_repositories:
        raw = raw_inspections[name]
        if not isinstance(raw, dict):
            raise LabError(f"invalid Task repository inspection: {name}")
        paths[name] = Path(str(raw.get("path", "")))
        raw_scopes = raw.get("write_scope")
        if not isinstance(raw_scopes, list) or not all(
            isinstance(scope, str) for scope in raw_scopes
        ):
            raise LabError(f"invalid Task repository scope: {name}")
        scopes[name] = tuple(raw_scopes)
        fingerprints[name] = repository_fingerprint(paths[name])

    captures: dict[str, DraftRepositoryCapture] = {}
    with workspace_manager.advisory_locks(
        "snapshots", [str(paths[name]) for name in task_repositories]
    ):
        for name in order:
            link_overrides = {
                path: captures[child].commit
                for parent, path, child in links
                if parent == name
            }
            captures[name] = capture_repository_draft(
                name,
                paths[name],
                scopes=scopes[name],
                include_untracked=include_untracked.get(name, ()),
                gitlink_overrides=link_overrides,
                expected_fingerprint=fingerprints[name],
            )
        for name in task_repositories:
            if repository_fingerprint(paths[name]) != fingerprints[name]:
                raise LabError(
                    f"multi-repository workspace changed during draft capture: {name}"
                )
        snapshot_id, identity = resolve_clean_snapshot_identity(
            workspace_manager,
            task_id=task_id,
            profile_id=profile.profile_id,
            profile_sha256=profile.sha256,
            repositories=profile.repositories,
            baseline_path=baseline_path,
            task_repository_overrides={
                name: capture.identity() for name, capture in captures.items()
            },
            repository_link_overrides=[
                {
                    "parent_repository": parent,
                    "submodule_path": path,
                    "child_repository": child,
                }
                for parent, path, child in links
            ],
            snapshot_mode="draft",
        )
        for name, capture in captures.items():
            ref = f"refs/lab/snapshots/{snapshot_id}/{name}"
            if git.ref_exists(paths[name], ref):
                if git.resolve_commit(paths[name], ref) != capture.commit:
                    raise LabError(
                        f"draft snapshot ref collision: {name}/{snapshot_id}"
                    )
            else:
                git.run_git(paths[name], ["update-ref", ref, capture.commit])
        for name in task_repositories:
            if repository_fingerprint(paths[name]) != fingerprints[name]:
                raise LabError(
                    f"repository changed while publishing draft snapshot ref: {name}"
                )
    identity["draft"] = {
        "schema": "lab.synthetic-draft.v1",
        "ref_namespace": f"refs/lab/snapshots/{snapshot_id}",
        "capture_sha256": _sha256_json(
            {name: capture.identity() for name, capture in sorted(captures.items())}
        ),
    }
    return snapshot_id, identity
