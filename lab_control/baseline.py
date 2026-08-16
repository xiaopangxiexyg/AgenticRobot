from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .workspace import LabError, validate_repository_name, validate_task_id

BASELINE_SCHEMA_ID = "lab.baseline.v2"
LEGACY_BASELINE_SCHEMA_ID = "lab.baseline.v1"
OID_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class BaselineRepository:
    name: str
    canonical_identity: str
    commit: str
    tree_oid: str
    submodules: Mapping[str, str]


@dataclass(frozen=True)
class BaselineRepositoryLink:
    parent_repository: str
    submodule_path: str
    child_repository: str


@dataclass(frozen=True)
class BaselineLock:
    schema: str
    baseline_id: str
    repositories: Mapping[str, BaselineRepository]
    repository_links: tuple[BaselineRepositoryLink, ...]
    path: Path
    sha256: str


def _submodule_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise LabError(f"invalid baseline submodule path: {label}")
    path = PurePosixPath(value)
    if (
        value.startswith(("/", "~"))
        or value != path.as_posix()
        or path in {PurePosixPath("."), PurePosixPath("")}
        or ".." in path.parts
    ):
        raise LabError(f"invalid baseline submodule path: {label}")
    return value


def load_baseline_lock(path: Path) -> BaselineLock:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read baseline lock {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") not in {
        LEGACY_BASELINE_SCHEMA_ID,
        BASELINE_SCHEMA_ID,
    }:
        raise LabError(f"unsupported baseline lock: {resolved}")
    schema = str(payload["schema"])
    baseline_id = str(payload.get("baseline_id", ""))
    validate_task_id(baseline_id)
    raw_repositories = payload.get("repositories")
    if not isinstance(raw_repositories, dict) or not raw_repositories:
        raise LabError(f"baseline lock has no repositories: {resolved}")

    repositories: dict[str, BaselineRepository] = {}
    for name, raw_repository in raw_repositories.items():
        validate_repository_name(name)
        if not isinstance(raw_repository, dict):
            raise LabError(f"invalid baseline repository: {name}")
        canonical_identity = raw_repository.get("canonical_identity")
        if (
            not isinstance(canonical_identity, str)
            or canonical_identity.count("/") != 1
        ):
            raise LabError(f"invalid baseline repository identity: {name}")
        commit = raw_repository.get("commit")
        tree_oid = raw_repository.get("tree_oid")
        if (
            not isinstance(commit, str)
            or not OID_PATTERN.fullmatch(commit)
            or not isinstance(tree_oid, str)
            or not OID_PATTERN.fullmatch(tree_oid)
        ):
            raise LabError(f"baseline repository lacks full OIDs: {name}")
        raw_submodules = raw_repository.get("submodules", {})
        if not isinstance(raw_submodules, dict):
            raise LabError(f"invalid baseline submodules: {name}")
        submodules: dict[str, str] = {}
        for submodule_path, oid in raw_submodules.items():
            if (
                not isinstance(oid, str)
                or not OID_PATTERN.fullmatch(oid)
            ):
                raise LabError(
                    f"invalid baseline submodule identity: {name}/{submodule_path}"
                )
            safe_path = _submodule_path(
                submodule_path,
                label=f"{name}/{submodule_path}",
            )
            submodules[safe_path] = oid
        repositories[name] = BaselineRepository(
            name=name,
            canonical_identity=canonical_identity,
            commit=commit,
            tree_oid=tree_oid,
            submodules=dict(sorted(submodules.items())),
        )

    raw_links = payload.get("repository_links")
    if schema == LEGACY_BASELINE_SCHEMA_ID:
        if raw_links is not None:
            raise LabError("baseline v1 cannot declare repository links")
        raw_links = []
    if not isinstance(raw_links, list):
        raise LabError(f"baseline v2 requires repository_links: {resolved}")
    repository_links: list[BaselineRepositoryLink] = []
    identities: set[tuple[str, str]] = set()
    for index, raw_link in enumerate(raw_links):
        if not isinstance(raw_link, dict) or set(raw_link) != {
            "parent_repository",
            "submodule_path",
            "child_repository",
        }:
            raise LabError(f"invalid baseline repository link: {index}")
        parent = str(raw_link.get("parent_repository", ""))
        child = str(raw_link.get("child_repository", ""))
        validate_repository_name(parent)
        validate_repository_name(child)
        if parent == child:
            raise LabError(f"baseline repository link cannot be recursive: {parent}")
        link_path = _submodule_path(
            raw_link.get("submodule_path"),
            label=f"repository_links[{index}]",
        )
        identity = (parent, link_path)
        if identity in identities:
            raise LabError(
                f"duplicate baseline repository link: {parent}/{link_path}"
            )
        identities.add(identity)
        parent_pin = repositories.get(parent)
        child_pin = repositories.get(child)
        if parent_pin is None or child_pin is None:
            raise LabError(
                f"baseline repository link references an unknown repository: "
                f"{parent}/{link_path} -> {child}"
            )
        if parent_pin.submodules.get(link_path) != child_pin.commit:
            raise LabError(
                f"baseline repository link gitlink mismatch: "
                f"{parent}/{link_path} != {child}@{child_pin.commit}"
            )
        repository_links.append(
            BaselineRepositoryLink(
                parent_repository=parent,
                submodule_path=link_path,
                child_repository=child,
            )
        )
    adjacency: dict[str, set[str]] = {}
    for link in repository_links:
        adjacency.setdefault(link.parent_repository, set()).add(
            link.child_repository
        )
    visiting: set[str] = set()
    visited: set[str] = set()
    postorder: list[str] = []

    def visit(repository: str) -> None:
        if repository in visiting:
            raise LabError(
                f"baseline repository links contain a cycle at: {repository}"
            )
        if repository in visited:
            return
        visiting.add(repository)
        for child in sorted(adjacency.get(repository, set())):
            visit(child)
        visiting.remove(repository)
        visited.add(repository)
        postorder.append(repository)

    for repository in sorted(repositories):
        visit(repository)
    repository_rank = {
        repository: rank
        for rank, repository in enumerate(reversed(postorder))
    }
    return BaselineLock(
        schema=schema,
        baseline_id=baseline_id,
        repositories=repositories,
        repository_links=tuple(
            sorted(
                repository_links,
                key=lambda link: (
                    repository_rank[link.parent_repository],
                    link.parent_repository,
                    link.submodule_path,
                    link.child_repository,
                ),
            )
        ),
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
