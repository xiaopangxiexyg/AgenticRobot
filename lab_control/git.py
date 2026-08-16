from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class GitError(RuntimeError):
    """A Git operation failed without changing unrelated user state."""


@dataclass(frozen=True)
class GitResult:
    returncode: int
    stdout: str
    stderr: str


def run_git(
    repository: Path,
    arguments: Iterable[str],
    *,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
) -> GitResult:
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-C",
        str(repository),
        *arguments,
    ]
    process_environment = dict(os.environ)
    if environment is not None:
        process_environment.update(environment)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=process_environment,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitError(f"Git command failed in {repository}: {detail}")
    return GitResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def repository_root(path: Path) -> Path:
    if not path.is_dir():
        raise GitError(f"repository path does not exist: {path}")
    root = run_git(path, ["rev-parse", "--show-toplevel"]).stdout.strip()
    if not root:
        raise GitError(f"not a Git working tree: {path}")
    return Path(root).resolve()


def resolve_commit(repository: Path, ref: str) -> str:
    result = run_git(
        repository,
        ["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise GitError(f"ref did not resolve to a full commit in {repository}: {ref}")
    return commit


def ref_exists(repository: Path, ref: str) -> bool:
    result = run_git(repository, ["show-ref", "--verify", "--quiet", ref], check=False)
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    detail = result.stderr.strip() or result.stdout.strip()
    raise GitError(f"could not inspect ref {ref!r} in {repository}: {detail}")


def matching_remote_branches(repository: Path, branch: str) -> tuple[str, ...]:
    result = run_git(
        repository,
        ["for-each-ref", "--format=%(refname)", "refs/remotes"],
    )
    suffix = f"/{branch}"
    return tuple(
        ref
        for ref in result.stdout.splitlines()
        if ref.endswith(suffix) and not ref.endswith("/HEAD")
    )


def remote_tracking_refs_containing(
    repository: Path,
    commit: str,
) -> tuple[str, ...]:
    """Return fetched remote-tracking refs that contain an exact commit."""

    resolve_commit(repository, commit)
    result = run_git(
        repository,
        [
            "for-each-ref",
            "--contains",
            commit,
            "--format=%(refname)",
            "refs/remotes",
        ],
    )
    return tuple(
        sorted(
            ref
            for ref in result.stdout.splitlines()
            if ref and not ref.endswith("/HEAD")
        )
    )


def create_worktree(
    repository: Path,
    *,
    target: Path,
    branch: str,
    base_commit: str,
) -> None:
    run_git(
        repository,
        ["worktree", "add", "-b", branch, str(target), base_commit],
    )


def remove_worktree(repository: Path, target: Path) -> None:
    run_git(repository, ["worktree", "remove", "--force", str(target)])


def delete_branch(repository: Path, branch: str) -> None:
    run_git(repository, ["branch", "-D", branch])


def current_branch(repository: Path) -> str | None:
    branch = run_git(repository, ["branch", "--show-current"]).stdout.strip()
    return branch or None


def current_commit(repository: Path) -> str:
    return resolve_commit(repository, "HEAD")


def tree_oid(repository: Path, commit: str = "HEAD") -> str:
    result = run_git(
        repository,
        ["rev-parse", "--verify", "--end-of-options", f"{commit}^{{tree}}"],
    )
    oid = result.stdout.strip()
    if len(oid) != 40:
        raise GitError(f"commit did not resolve to a full tree OID: {commit}")
    return oid


def create_detached_worktree(
    repository: Path,
    *,
    target: Path,
    commit: str,
) -> None:
    run_git(repository, ["worktree", "add", "--detach", str(target), commit])


def status_short(repository: Path) -> tuple[str, ...]:
    result = run_git(repository, ["status", "--short", "--untracked-files=all"])
    return tuple(line for line in result.stdout.splitlines() if line)


def changed_paths(repository: Path, base_commit: str) -> tuple[str, ...]:
    tracked = run_git(
        repository,
        ["diff", "--name-only", "--diff-filter=ACDMRTUXB", base_commit],
    ).stdout.splitlines()
    untracked = run_git(
        repository,
        ["ls-files", "--others", "--exclude-standard"],
    ).stdout.splitlines()
    return tuple(sorted({path for path in (*tracked, *untracked) if path}))


def canonical_repository_identity(remote_url: str) -> str:
    """Return owner/repository independent of SSH aliases and URL transport."""

    value = remote_url.strip()
    if "://" in value:
        repository_path = urlparse(value).path
    elif ":" in value and not value.startswith("/"):
        repository_path = value.split(":", 1)[1]
    else:
        repository_path = value
    parts = [part for part in repository_path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise GitError(f"remote URL does not identify owner/repository: {remote_url}")
    repository = parts[1].removesuffix(".git")
    if not parts[0] or not repository:
        raise GitError(f"remote URL has an empty owner/repository: {remote_url}")
    return f"{parts[0]}/{repository}"


def origin_identity(repository: Path) -> str:
    url = run_git(repository, ["remote", "get-url", "origin"]).stdout.strip()
    if not url:
        raise GitError(f"repository has no origin URL: {repository}")
    return canonical_repository_identity(url)


def submodule_status(repository: Path) -> tuple[dict[str, str], bool]:
    result = run_git(repository, ["submodule", "status", "--recursive"])
    submodules: dict[str, str] = {}
    initialized_at_index = True
    for line in result.stdout.splitlines():
        if not line:
            continue
        state = line[0]
        fields = line[1:].strip().split()
        if len(fields) < 2 or len(fields[0]) != 40:
            raise GitError(f"cannot parse submodule status in {repository}: {line}")
        submodules[fields[1]] = fields[0]
        initialized_at_index = initialized_at_index and state == " "
    return dict(sorted(submodules.items())), initialized_at_index


def submodule_gitlinks(
    repository: Path,
    commit: str = "HEAD",
) -> dict[str, str]:
    """Read gitlinks from an exact tree without touching its checkout."""

    resolved = resolve_commit(repository, commit)
    result = run_git(repository, ["ls-tree", "-r", "-z", resolved])
    submodules: dict[str, str] = {}
    for record in result.stdout.split("\0"):
        if not record:
            continue
        metadata, separator, path = record.partition("\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise GitError(
                f"cannot parse tree entry for submodule identity in {repository}"
            )
        mode, object_type, oid = fields
        if mode != "160000":
            continue
        if object_type != "commit" or len(oid) != 40 or not path:
            raise GitError(f"invalid gitlink in {repository}: {record}")
        submodules[path] = oid
    return dict(sorted(submodules.items()))


def update_submodules_no_fetch(repository: Path) -> None:
    run_git(
        repository,
        ["submodule", "update", "--init", "--recursive", "--checkout", "--no-fetch"],
    )
