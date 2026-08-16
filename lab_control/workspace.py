from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import git

SCHEMA_ID = "lab.task.v1"
CHANGE_SCHEMA_ID = "lab.change.v1"
TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
REPOSITORY_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,63}")
BRANCH_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
COMMIT_OID_PATTERN = re.compile(r"[0-9a-f]{40}")


class LabError(RuntimeError):
    """A fail-closed validation or local workspace operation failed."""


@dataclass(frozen=True)
class RepositoryPlan:
    name: str
    management_path: Path
    worktree_path: Path
    branch: str
    base_ref: str
    base_commit: str
    scopes: tuple[str, ...]
    management_dirty: bool
    parent_repository: str | None = None
    submodule_path: str | None = None

    def to_manifest(
        self, *, development_host: str, writer_epoch: int
    ) -> dict[str, object]:
        manifest: dict[str, object] = {
            "management_path": str(self.management_path),
            "worktree_path": str(self.worktree_path),
            "branch": self.branch,
            "base_ref": self.base_ref,
            "base_commit": self.base_commit,
            "writer": {
                "host": development_host,
                "epoch": writer_epoch,
            },
            "write_scope": list(self.scopes),
            "management_dirty_at_create": self.management_dirty,
        }
        if self.parent_repository is not None:
            if self.submodule_path is None:
                raise LabError(
                    f"nested repository {self.name} lacks a submodule path"
                )
            manifest["workspace_placement"] = {
                "kind": "nested_gitlink",
                "parent_repository": self.parent_repository,
                "submodule_path": self.submodule_path,
            }
        return manifest


@dataclass(frozen=True)
class TaskPlan:
    task_id: str
    title: str
    change_id: str | None
    development_host: str
    task_root: Path
    repositories: tuple[RepositoryPlan, ...]


def _validate_component(value: str, *, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise LabError(f"invalid {label}: {value!r}")
    if value in {".", ".."} or ".." in value.split("/") or value.endswith(".lock"):
        raise LabError(f"unsafe {label}: {value!r}")
    return value


def validate_task_id(task_id: str) -> str:
    return _validate_component(task_id, label="task ID", pattern=TASK_ID_PATTERN)


def validate_repository_name(name: str) -> str:
    return _validate_component(
        name,
        label="repository name",
        pattern=REPOSITORY_NAME_PATTERN,
    )


def validate_branch(branch: str) -> str:
    validated = _validate_component(branch, label="branch", pattern=BRANCH_PATTERN)
    if validated.startswith("-") or validated.endswith(("/", ".")) or "//" in validated:
        raise LabError(f"unsafe branch: {branch!r}")
    return validated


def validate_scope(scope: str) -> str:
    if scope == "**":
        return scope
    if not scope or scope.startswith(("/", "~")) or "\\" in scope or "\x00" in scope:
        raise LabError(f"unsafe write scope: {scope!r}")
    path = PurePosixPath(scope)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise LabError(f"unsafe write scope: {scope!r}")
    return scope.rstrip("/")


def validate_submodule_path(path_value: str) -> str:
    if (
        not isinstance(path_value, str)
        or not path_value
        or path_value.startswith(("/", "~"))
        or "\\" in path_value
        or "\x00" in path_value
    ):
        raise LabError(f"unsafe submodule path: {path_value!r}")
    path = PurePosixPath(path_value)
    if path.as_posix() != path_value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise LabError(f"unsafe submodule path: {path_value!r}")
    return path_value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _canonical_repository_name(
    value: str,
    catalog: Mapping[str, Path],
    aliases: Mapping[str, str],
) -> str:
    canonical = aliases.get(value.casefold(), value)
    if canonical not in catalog:
        available = ", ".join(sorted(catalog))
        raise LabError(f"unknown repository {value!r}; available: {available}")
    return canonical


def _parse_assignments(
    assignments: Iterable[str],
    *,
    catalog: Mapping[str, Path],
    aliases: Mapping[str, str],
    label: str,
) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise LabError(f"{label} must use REPOSITORY=VALUE: {assignment!r}")
        raw_name, value = assignment.split("=", 1)
        name = _canonical_repository_name(raw_name, catalog, aliases)
        if not value:
            raise LabError(f"empty {label} value for {name}")
        parsed.setdefault(name, []).append(value)
    return parsed


def scope_matches(path: str, scopes: Sequence[str]) -> bool:
    for scope in scopes:
        if scope == "**":
            return True
        if scope.endswith("/**") and path.startswith(scope[:-3].rstrip("/") + "/"):
            return True
        if fnmatch.fnmatchcase(path, scope) or path == scope:
            return True
    return False


class WorkspaceManager:
    """Create and inspect local, multi-repository Task workspaces."""

    def __init__(
        self,
        *,
        control_repository: Path,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._environ = dict(os.environ if environ is None else environ)
        try:
            self.control_repository = git.repository_root(control_repository)
        except git.GitError as exc:
            raise LabError(str(exc)) from exc
        from .project_manifest import load_project_manifest

        configured_manifest = self._environ.get("LAB_PROJECT_MANIFEST")
        manifest_path = (
            Path(configured_manifest)
            if configured_manifest
            else self.control_repository / "integration" / "project.json"
        )
        self.project = load_project_manifest(
            manifest_path,
            control_repository_root=self.control_repository,
            environ=self._environ,
        )
        self.catalog = dict(self.project.repositories)
        self.repository_aliases = dict(self.project.aliases)
        configured_home = self._environ.get("LAB_HOME")
        self.lab_home = (
            Path(configured_home).expanduser().resolve()
            if configured_home
            else self.control_repository.parent.parent.resolve()
        )
        configured_worktrees = self._environ.get("LAB_WORKTREES_ROOT")
        self.worktrees_root = (
            Path(configured_worktrees).expanduser().resolve()
            if configured_worktrees
            else self.lab_home / "Worktrees"
        )
        configured_state = self._environ.get("LAB_STATE_ROOT")
        self.state_root = (
            Path(configured_state).expanduser().resolve()
            if configured_state
            else self.lab_home / "State"
        )

    @property
    def environ(self) -> Mapping[str, str]:
        return self._environ

    def plan_task(
        self,
        *,
        task_id: str,
        repository_names: Sequence[str],
        title: str = "",
        change_id: str | None = None,
        development_host: str | None = None,
        base_assignments: Sequence[str] = (),
        scope_assignments: Sequence[str] = (),
    ) -> TaskPlan:
        task_id = validate_task_id(task_id)
        resolved_development_host = (
            development_host or self.project.default_development_host
        )
        development_host = _validate_component(
            resolved_development_host,
            label="development host",
            pattern=TASK_ID_PATTERN,
        )
        if development_host != self.project.default_development_host:
            raise LabError(
                "remote-hosted Task creation is not implemented; use the project default development host"
            )
        if change_id is not None:
            change_id = validate_task_id(change_id)
            self.load_change(change_id)
        if not repository_names:
            raise LabError("at least one repository is required")

        selected: list[str] = []
        for raw_name in repository_names:
            name = _canonical_repository_name(
                raw_name, self.catalog, self.repository_aliases
            )
            if name in selected:
                raise LabError(f"duplicate repository: {name}")
            selected.append(name)

        bases = _parse_assignments(
            base_assignments,
            catalog=self.catalog,
            aliases=self.repository_aliases,
            label="base",
        )
        scopes = _parse_assignments(
            scope_assignments,
            catalog=self.catalog,
            aliases=self.repository_aliases,
            label="scope",
        )
        unselected = (set(bases) | set(scopes)) - set(selected)
        if unselected:
            raise LabError(
                "base/scope supplied for repositories not in this task: "
                + ", ".join(sorted(unselected))
            )

        task_root = self.worktrees_root / task_id
        if task_root.exists() or task_root.is_symlink():
            raise LabError(f"task workspace already exists: {task_root}")

        repository_facts: dict[str, dict[str, object]] = {}
        branch = validate_branch(f"task/{task_id}")
        for name in selected:
            configured_path = self.catalog[name]
            try:
                management_path = git.repository_root(configured_path)
                base_ref_values = bases.get(name, ["HEAD"])
                if len(base_ref_values) != 1:
                    raise LabError(f"multiple base refs supplied for {name}")
                base_ref = base_ref_values[0]
                if base_ref.startswith("-") or "\x00" in base_ref:
                    raise LabError(f"unsafe base ref for {name}: {base_ref!r}")
                base_commit = git.resolve_commit(management_path, base_ref)
                if git.ref_exists(management_path, f"refs/heads/{branch}"):
                    raise LabError(f"local branch already exists in {name}: {branch}")
                remote_refs = git.matching_remote_branches(management_path, branch)
                if remote_refs:
                    raise LabError(
                        f"known remote branch already exists in {name}: "
                        + ", ".join(remote_refs)
                    )
                repository_scopes = tuple(
                    validate_scope(value) for value in scopes.get(name, ["**"])
                )
                management_dirty = bool(git.status_short(management_path))
            except git.GitError as exc:
                raise LabError(str(exc)) from exc

            repository_facts[name] = {
                "management_path": management_path,
                "base_ref": base_ref,
                "base_commit": base_commit,
                "scopes": repository_scopes,
                "management_dirty": management_dirty,
            }

        # A selected repository that is already a gitlink checkout inside another
        # selected repository is placed at that same relative path in the Task.
        # This keeps one writable child checkout instead of creating a sibling
        # copy plus an unrelated nested submodule checkout.
        placements: dict[str, tuple[str, str]] = {}
        gitlinks_by_parent: dict[str, dict[str, str]] = {}
        for child_name, child_facts in repository_facts.items():
            child_management = Path(str(child_facts["management_path"]))
            candidates: list[tuple[int, str, str]] = []
            for parent_name, parent_facts in repository_facts.items():
                if parent_name == child_name:
                    continue
                parent_management = Path(str(parent_facts["management_path"]))
                try:
                    relative = child_management.relative_to(parent_management)
                except ValueError:
                    continue
                relative_path = validate_submodule_path(relative.as_posix())
                try:
                    parent_gitlinks = gitlinks_by_parent.get(parent_name)
                    if parent_gitlinks is None:
                        parent_gitlinks = git.submodule_gitlinks(
                            parent_management,
                            str(parent_facts["base_commit"]),
                        )
                        gitlinks_by_parent[parent_name] = parent_gitlinks
                except git.GitError as exc:
                    raise LabError(str(exc)) from exc
                gitlink_commit = parent_gitlinks.get(relative_path)
                if gitlink_commit is None:
                    continue
                if gitlink_commit != child_facts["base_commit"]:
                    raise LabError(
                        f"nested repository base differs from parent gitlink: "
                        f"{parent_name}/{relative_path}={gitlink_commit}, "
                        f"{child_name}={child_facts['base_commit']}"
                    )
                candidates.append(
                    (len(parent_management.parts), parent_name, relative_path)
                )
            if candidates:
                _, parent_name, relative_path = max(candidates)
                placements[child_name] = (parent_name, relative_path)

        worktree_paths: dict[str, Path] = {}
        visiting: set[str] = set()

        def task_path(name: str) -> Path:
            existing = worktree_paths.get(name)
            if existing is not None:
                return existing
            if name in visiting:
                raise LabError(f"nested repository placement cycle at: {name}")
            visiting.add(name)
            placement = placements.get(name)
            if placement is None:
                resolved = task_root / name
            else:
                parent_name, relative_path = placement
                resolved = task_path(parent_name).joinpath(
                    *PurePosixPath(relative_path).parts
                )
            visiting.remove(name)
            worktree_paths[name] = resolved
            return resolved

        plans_by_name: dict[str, RepositoryPlan] = {}
        for name in selected:
            facts = repository_facts[name]
            placement = placements.get(name)
            raw_scopes = facts["scopes"]
            if not isinstance(raw_scopes, tuple):
                raise LabError(f"invalid planned scopes for {name}")
            plans_by_name[name] = RepositoryPlan(
                name=name,
                management_path=Path(str(facts["management_path"])),
                worktree_path=task_path(name),
                branch=branch,
                base_ref=str(facts["base_ref"]),
                base_commit=str(facts["base_commit"]),
                scopes=tuple(str(scope) for scope in raw_scopes),
                management_dirty=bool(facts["management_dirty"]),
                parent_repository=placement[0] if placement else None,
                submodule_path=placement[1] if placement else None,
            )

        ordered: list[RepositoryPlan] = []
        added: set[str] = set()

        def add_with_parent(name: str) -> None:
            if name in added:
                return
            placement = placements.get(name)
            if placement is not None:
                add_with_parent(placement[0])
            ordered.append(plans_by_name[name])
            added.add(name)

        for name in selected:
            add_with_parent(name)

        return TaskPlan(
            task_id=task_id,
            title=title.strip(),
            change_id=change_id,
            development_host=development_host,
            task_root=task_root,
            repositories=tuple(ordered),
        )

    def create_task(self, plan: TaskPlan) -> dict[str, object]:
        lock_keys = [
            f"{repository.name}:{repository.management_path}"
            for repository in plan.repositories
        ]
        with self.advisory_locks("repositories", lock_keys):
            self._revalidate_task_plan(plan)
            created: list[RepositoryPlan] = []
            plan.task_root.mkdir(parents=True, exist_ok=False)
            try:
                for repository in plan.repositories:
                    if repository.parent_repository is not None:
                        target = repository.worktree_path
                        if target.is_dir() and not target.is_symlink():
                            try:
                                target.rmdir()
                            except OSError as exc:
                                raise LabError(
                                    f"nested gitlink checkout is not empty: {target}"
                                ) from exc
                    git.create_worktree(
                        repository.management_path,
                        target=repository.worktree_path,
                        branch=repository.branch,
                        base_commit=repository.base_commit,
                    )
                    created.append(repository)

                manifest = self._manifest_for_plan(plan)
                _atomic_write_text(
                    plan.task_root / "task.yaml",
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                )
                _atomic_write_text(
                    plan.task_root / f"{plan.task_id}.code-workspace",
                    self._workspace_document(plan),
                )
                _atomic_write_text(
                    plan.task_root / "task.env",
                    self._task_environment(plan),
                )
                return manifest
            except Exception as exc:
                cleanup_errors: list[str] = []
                for repository in reversed(created):
                    try:
                        git.remove_worktree(
                            repository.management_path,
                            repository.worktree_path,
                        )
                        git.delete_branch(
                            repository.management_path,
                            repository.branch,
                        )
                    except git.GitError as cleanup_exc:
                        cleanup_errors.append(str(cleanup_exc))
                if plan.task_root.exists() and not any(
                    repository.worktree_path.exists() for repository in created
                ):
                    shutil.rmtree(plan.task_root)
                detail = f"task creation failed: {exc}"
                if cleanup_errors:
                    detail += "; cleanup requires attention: " + "; ".join(
                        cleanup_errors
                    )
                raise LabError(detail) from exc

    def _revalidate_task_plan(self, plan: TaskPlan) -> None:
        if plan.task_root.exists() or plan.task_root.is_symlink():
            raise LabError(f"task workspace already exists: {plan.task_root}")
        for repository in plan.repositories:
            try:
                if (
                    git.repository_root(repository.management_path)
                    != repository.management_path
                ):
                    raise LabError(
                        f"management repository changed for {repository.name}: "
                        f"{repository.management_path}"
                    )
                if (
                    git.resolve_commit(
                        repository.management_path,
                        repository.base_commit,
                    )
                    != repository.base_commit
                ):
                    raise LabError(
                        f"base commit changed for {repository.name}: "
                        f"{repository.base_commit}"
                    )
                if git.ref_exists(
                    repository.management_path,
                    f"refs/heads/{repository.branch}",
                ):
                    raise LabError(
                        f"local branch appeared before create in {repository.name}: "
                        f"{repository.branch}"
                    )
                remote_refs = git.matching_remote_branches(
                    repository.management_path,
                    repository.branch,
                )
                if remote_refs:
                    raise LabError(
                        f"known remote branch appeared before create in {repository.name}: "
                        + ", ".join(remote_refs)
                    )
                if (
                    repository.worktree_path.exists()
                    or repository.worktree_path.is_symlink()
                ):
                    raise LabError(
                        f"worktree target appeared before create: {repository.worktree_path}"
                    )
            except git.GitError as exc:
                raise LabError(str(exc)) from exc

    @contextmanager
    def advisory_locks(self, category: str, keys: Sequence[str]) -> Iterator[None]:
        validate_repository_name(category)
        lock_root = self.state_root / "locks" / category
        lock_root.mkdir(parents=True, exist_ok=True)
        descriptors: list[int] = []
        try:
            for key in sorted(set(keys)):
                digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
                path = lock_root / f"{digest}.lock"
                flags = os.O_RDWR | os.O_CREAT
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(path, flags, 0o600)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    os.close(descriptor)
                    raise LabError(
                        f"another lab operation holds {category} lock for {key}"
                    ) from exc
                os.ftruncate(descriptor, 0)
                os.write(
                    descriptor,
                    f"pid={os.getpid()}\nkey={key}\n".encode(),
                )
                descriptors.append(descriptor)
            yield
        finally:
            for descriptor in reversed(descriptors):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _manifest_for_plan(self, plan: TaskPlan) -> dict[str, object]:
        return {
            "schema": SCHEMA_ID,
            "task_id": plan.task_id,
            "title": plan.title,
            "change_id": plan.change_id,
            "status": "ACTIVE",
            "development_host": plan.development_host,
            "writer_epoch": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(plan.task_root),
            "environment": {
                "status": "DEFERRED",
                "reason": "M1 creates source workspaces only; profiles resolve environments lazily",
            },
            "repositories": {
                repository.name: repository.to_manifest(
                    development_host=plan.development_host,
                    writer_epoch=1,
                )
                for repository in plan.repositories
            },
            "acceptance": [],
        }

    @staticmethod
    def _workspace_document(plan: TaskPlan) -> str:
        document = {
            "folders": [
                {
                    "name": repository.name,
                    "path": repository.worktree_path.relative_to(
                        plan.task_root
                    ).as_posix(),
                }
                for repository in plan.repositories
            ],
            "settings": {
                "window.title": f"[{plan.task_id}] ${{activeEditorShort}}",
            },
        }
        return json.dumps(document, indent=2, sort_keys=True) + "\n"

    @staticmethod
    def _task_environment(plan: TaskPlan) -> str:
        lines = [
            "# Generated by lab; source this file from the Task root.",
            f"export TASK_ID={shlex.quote(plan.task_id)}",
            f"export TASK_ROOT={shlex.quote(str(plan.task_root))}",
            f"export DEVELOPMENT_HOST={shlex.quote(plan.development_host)}",
        ]
        for repository in plan.repositories:
            key = re.sub(r"[^A-Za-z0-9]", "_", repository.name).upper()
            lines.append(
                f"export {key}_ROOT={shlex.quote(str(repository.worktree_path))}"
            )
        return "\n".join(lines) + "\n"

    def task_manifest_path(self, task_id: str) -> Path:
        return self.worktrees_root / validate_task_id(task_id) / "task.yaml"

    def change_manifest_path(self, change_id: str) -> Path:
        return self.state_root / "changes" / validate_task_id(change_id) / "change.json"

    def create_change(self, change_id: str, *, title: str = "") -> dict[str, object]:
        change_id = validate_task_id(change_id)
        changes_root = self.state_root / "changes"
        if changes_root.is_symlink():
            raise LabError(f"Change state root is a symlink: {changes_root}")
        target = changes_root / change_id
        with self.advisory_locks("changes", [change_id]):
            if target.exists() or target.is_symlink():
                raise LabError(f"Change already exists: {change_id}")
            target.mkdir(parents=True)
            manifest: dict[str, object] = {
                "schema": CHANGE_SCHEMA_ID,
                "change_id": change_id,
                "title": title.strip(),
                "status": "ACTIVE",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                _atomic_write_text(
                    target / "change.json",
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                )
                return manifest
            except Exception:
                if target.exists() and not any(target.iterdir()):
                    target.rmdir()
                raise

    def load_change(self, change_id: str) -> dict[str, object]:
        path = self.change_manifest_path(change_id)
        if path.is_symlink() or not path.is_file():
            raise LabError(f"Change does not exist: {change_id} ({path})")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError(f"cannot read Change manifest {path}: {exc}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != CHANGE_SCHEMA_ID
            or payload.get("change_id") != change_id
            or payload.get("status") != "ACTIVE"
        ):
            raise LabError(f"invalid Change manifest identity/state: {path}")
        return payload

    def inspect_change(self, change_id: str) -> dict[str, object]:
        change = self.load_change(change_id)
        tasks: dict[str, object] = {}
        if self.worktrees_root.exists():
            if self.worktrees_root.is_symlink():
                raise LabError(f"worktrees root is a symlink: {self.worktrees_root}")
            for task_root in sorted(self.worktrees_root.iterdir()):
                if task_root.is_symlink() or not task_root.is_dir():
                    continue
                manifest_path = task_root / "task.yaml"
                if not manifest_path.is_file() or manifest_path.is_symlink():
                    continue
                task_id = validate_task_id(task_root.name)
                manifest = self.load_task(task_id)
                if manifest.get("change_id") == change_id:
                    tasks[task_id] = self.inspect_task(task_id)
        attention = any(
            isinstance(task, dict) and task.get("status") != "ACTIVE"
            for task in tasks.values()
        )
        return {
            "schema": "lab.change-inspection.v1",
            "change_id": change_id,
            "title": change.get("title", ""),
            "status": "ATTENTION" if attention else "ACTIVE",
            "tasks": tasks,
            "task_count": len(tasks),
            "development_hosts": sorted(
                {
                    str(task.get("development_host"))
                    for task in tasks.values()
                    if isinstance(task, dict)
                }
            ),
            "next_action": (
                "Resolve drifted child Tasks before integration."
                if attention
                else "Develop child Tasks independently; integrate through exact snapshots."
            ),
        }

    def load_task(self, task_id: str) -> dict[str, object]:
        path = self.task_manifest_path(task_id)
        if not path.is_file():
            raise LabError(f"task does not exist: {task_id} ({path})")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError(f"cannot read task manifest {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA_ID:
            raise LabError(f"unsupported task manifest: {path}")
        if payload.get("task_id") != task_id:
            raise LabError(f"task manifest identity mismatch: {path}")
        change_id = payload.get("change_id")
        if change_id is not None:
            if not isinstance(change_id, str):
                raise LabError(f"task manifest has invalid Change ID: {path}")
            self.load_change(validate_task_id(change_id))
        expected_workspace_root = path.parent.resolve()
        configured_workspace_root = Path(
            str(payload.get("workspace_root", ""))
        ).resolve()
        if configured_workspace_root != expected_workspace_root:
            raise LabError(
                "task manifest workspace root mismatch: "
                f"{configured_workspace_root} != {expected_workspace_root}"
            )
        repositories = payload.get("repositories")
        if not isinstance(repositories, dict) or not repositories:
            raise LabError(f"task manifest has no repositories: {path}")
        management_paths: dict[str, Path] = {}
        for name, raw_repository in repositories.items():
            if not isinstance(name, str) or not isinstance(raw_repository, dict):
                raise LabError(f"invalid repository entry in task {task_id}")
            validate_repository_name(name)
            if name not in self.catalog:
                raise LabError(f"task manifest uses unknown repository: {name}")
            try:
                expected_management = git.repository_root(self.catalog[name])
            except git.GitError as exc:
                raise LabError(str(exc)) from exc
            configured_management = Path(
                str(raw_repository.get("management_path", ""))
            ).resolve()
            if configured_management != expected_management:
                raise LabError(
                    f"task manifest management path mismatch for {name}: "
                    f"{configured_management} != {expected_management}"
                )
            management_paths[name] = expected_management

        expected_worktrees: dict[str, Path] = {}
        visiting: set[str] = set()

        def expected_worktree_for(name: str) -> Path:
            existing = expected_worktrees.get(name)
            if existing is not None:
                return existing
            if name in visiting:
                raise LabError(
                    f"task manifest has a nested repository cycle at: {name}"
                )
            visiting.add(name)
            raw_repository = repositories.get(name)
            if not isinstance(raw_repository, dict):
                raise LabError(f"invalid repository entry in task {task_id}")
            raw_placement = raw_repository.get("workspace_placement")
            if raw_placement is None:
                expected = expected_workspace_root / name
            else:
                if not isinstance(raw_placement, dict) or set(raw_placement) != {
                    "kind",
                    "parent_repository",
                    "submodule_path",
                }:
                    raise LabError(
                        f"invalid workspace placement for {name} in task {task_id}"
                    )
                parent = str(raw_placement.get("parent_repository", ""))
                if (
                    raw_placement.get("kind") != "nested_gitlink"
                    or parent == name
                    or parent not in repositories
                ):
                    raise LabError(
                        f"invalid nested repository parent for {name} in task {task_id}"
                    )
                relative_path = validate_submodule_path(
                    str(raw_placement.get("submodule_path", ""))
                )
                expected_management = management_paths[parent].joinpath(
                    *PurePosixPath(relative_path).parts
                ).resolve()
                if management_paths[name] != expected_management:
                    raise LabError(
                        f"nested management path mismatch for {name}: "
                        f"{management_paths[name]} != {expected_management}"
                    )
                parent_repository = repositories[parent]
                if not isinstance(parent_repository, dict):
                    raise LabError(f"invalid repository entry in task {task_id}")
                parent_base = str(parent_repository.get("base_commit", ""))
                child_base = str(raw_repository.get("base_commit", ""))
                if (
                    not COMMIT_OID_PATTERN.fullmatch(parent_base)
                    or not COMMIT_OID_PATTERN.fullmatch(child_base)
                ):
                    raise LabError(
                        f"invalid nested repository base in task {task_id}"
                    )
                try:
                    gitlink = git.submodule_gitlinks(
                        management_paths[parent], parent_base
                    ).get(relative_path)
                except git.GitError as exc:
                    raise LabError(str(exc)) from exc
                if gitlink != child_base:
                    raise LabError(
                        f"nested repository base differs from task gitlink: "
                        f"{parent}/{relative_path}={gitlink}, {name}={child_base}"
                    )
                expected = expected_worktree_for(parent).joinpath(
                    *PurePosixPath(relative_path).parts
                )
            visiting.remove(name)
            expected_worktrees[name] = expected.resolve()
            return expected_worktrees[name]

        for name, raw_repository in repositories.items():
            if not isinstance(name, str) or not isinstance(raw_repository, dict):
                raise LabError(f"invalid repository entry in task {task_id}")
            expected_worktree = expected_worktree_for(name)
            configured_worktree = Path(
                str(raw_repository.get("worktree_path", ""))
            ).resolve()
            if configured_worktree != expected_worktree:
                raise LabError(
                    f"task manifest worktree path mismatch for {name}: "
                    f"{configured_worktree} != {expected_worktree}"
                )
        return payload

    def inspect_task(self, task_id: str) -> dict[str, object]:
        from .writer_lease import WriterLeaseManager

        manifest = self.load_task(task_id)
        repository_inspections: dict[str, object] = {}
        task_state = "ACTIVE"
        writer_manager = WriterLeaseManager(self)
        writer_attention = False
        repositories = manifest.get("repositories")
        if not isinstance(repositories, dict):
            raise LabError(f"task manifest has no repositories: {task_id}")
        for name, raw_repository in repositories.items():
            if not isinstance(name, str) or not isinstance(raw_repository, dict):
                raise LabError(f"invalid repository entry in task {task_id}")
            repository_path = Path(str(raw_repository.get("worktree_path", "")))
            expected_branch = str(raw_repository.get("branch", ""))
            base_commit = str(raw_repository.get("base_commit", ""))
            validate_branch(expected_branch)
            if not COMMIT_OID_PATTERN.fullmatch(base_commit):
                raise LabError(f"invalid base commit for {name} in task {task_id}")
            scopes_value = raw_repository.get("write_scope", [])
            if not isinstance(scopes_value, list) or not all(
                isinstance(value, str) for value in scopes_value
            ):
                raise LabError(f"invalid write scope for {name} in task {task_id}")
            scopes = tuple(scopes_value)
            for scope in scopes:
                validate_scope(scope)
            try:
                root = git.repository_root(repository_path)
                if root != repository_path.resolve():
                    raise LabError(
                        f"worktree path/root mismatch for {name}: {repository_path} -> {root}"
                    )
                branch = git.current_branch(repository_path)
                commit = git.current_commit(repository_path)
                tree_oid = git.tree_oid(repository_path, commit)
                status = git.status_short(repository_path)
                changed = git.changed_paths(repository_path, base_commit)
            except git.GitError as exc:
                raise LabError(str(exc)) from exc
            violations = tuple(
                path for path in changed if not scope_matches(path, scopes)
            )
            branch_matches = branch == expected_branch
            if not branch_matches or violations:
                task_state = "DRIFTED"
            writer_status = writer_manager.inspect_repository(
                task_id,
                name,
                manifest=manifest,
            )
            if writer_status["state"] in {"EXPIRED", "STALE_AUTHORITY"}:
                writer_attention = True
            repository_inspections[name] = {
                "path": str(repository_path),
                "expected_branch": expected_branch,
                "branch": branch,
                "branch_matches": branch_matches,
                "base_commit": base_commit,
                "commit": commit,
                "tree_oid": tree_oid,
                "dirty": bool(status),
                "status": list(status),
                "changed_paths": list(changed),
                "write_scope": list(scopes),
                "scope_violations": list(violations),
                "import_provenance": "DEFERRED_ENVIRONMENT",
                "writer_lease": writer_status,
            }

        if task_state == "DRIFTED":
            next_action = "Resolve branch/scope drift before checkpoint or test."
        elif writer_attention:
            next_action = (
                "Resolve expired/stale writer authority before the next edit."
            )
        else:
            next_action = "Claim the repository before editing; run recommended tests."
        return {
            "schema": "lab.inspection.v1",
            "task_id": task_id,
            "title": manifest.get("title", ""),
            "change_id": manifest.get("change_id"),
            "development_host": manifest.get("development_host"),
            "status": task_state,
            "workspace_root": manifest.get("workspace_root"),
            "environment": manifest.get("environment"),
            "repositories": repository_inspections,
            "writer_coordination": {
                "status": "ATTENTION" if writer_attention else "READY",
                "active_repositories": sorted(
                    name
                    for name, raw_repository in repository_inspections.items()
                    if isinstance(raw_repository, dict)
                    and isinstance(raw_repository.get("writer_lease"), dict)
                    and raw_repository["writer_lease"].get("state") == "ACTIVE"
                ),
            },
            "next_action": next_action,
        }

    def workspace_path(self, task_id: str) -> Path:
        manifest = self.load_task(task_id)
        task_root = Path(str(manifest["workspace_root"]))
        workspace = task_root / f"{task_id}.code-workspace"
        if not workspace.is_file():
            raise LabError(f"workspace file is missing: {workspace}")
        return workspace
