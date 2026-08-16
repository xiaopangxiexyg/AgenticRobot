from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .workspace import LabError, validate_repository_name, validate_task_id

PROJECT_MANIFEST_SCHEMA_ID = "lab.project.v3"
NESTED_PROJECT_MANIFEST_SCHEMA_ID = "lab.project.v2"
LEGACY_PROJECT_MANIFEST_SCHEMA_ID = "lab.project.v1"
_ENVIRONMENT_NAME = re.compile(r"LAB_REPOSITORY_[A-Z0-9_]{1,64}")
_BRANCH_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
_PATH_BASES = {"control_repository", "control_repository_parent"}
_CHECKOUT_MODES = {"stable_branch", "pinned_gitlink"}


@dataclass(frozen=True)
class ProjectManifest:
    project_id: str
    control_repository: str
    default_development_host: str
    repositories: Mapping[str, Path]
    stable_branches: Mapping[str, str]
    checkout_modes: Mapping[str, str]
    aliases: Mapping[str, str]
    path: Path
    sha256: str


def _relative_path(value: object, *, repository: str) -> str:
    if not isinstance(value, str) or not value:
        raise LabError(f"project manifest has invalid default path: {repository}")
    path = PurePosixPath(value)
    if value.startswith(("/", "~")) or "\\" in value or "\x00" in value:
        raise LabError(f"project manifest has unsafe default path: {repository}")
    if any(part in {"", ".."} for part in path.parts):
        raise LabError(f"project manifest has unsafe default path: {repository}")
    return value


def _stable_branch(value: object, *, repository: str) -> str:
    if (
        not isinstance(value, str)
        or not _BRANCH_NAME.fullmatch(value)
        or value.startswith(("/", "-"))
        or value.endswith("/")
        or ".." in value
        or "@{" in value
        or "//" in value
    ):
        raise LabError(f"project manifest has unsafe stable branch: {repository}")
    return value


def load_project_manifest(
    path: Path,
    *,
    control_repository_root: Path,
    environ: Mapping[str, str],
) -> ProjectManifest:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read project manifest {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "project_id",
        "control_repository",
        "default_development_host",
        "repositories",
    }:
        raise LabError(f"project manifest has an unexpected shape: {resolved}")
    project_id = payload.get("project_id")
    control_repository = payload.get("control_repository")
    development_host = payload.get("default_development_host")
    schema = payload.get("schema")
    if schema not in {
        LEGACY_PROJECT_MANIFEST_SCHEMA_ID,
        NESTED_PROJECT_MANIFEST_SCHEMA_ID,
        PROJECT_MANIFEST_SCHEMA_ID,
    } or not all(
        isinstance(value, str)
        for value in (project_id, control_repository, development_host)
    ):
        raise LabError(f"project manifest has invalid identity: {resolved}")
    validate_task_id(str(project_id))
    validate_repository_name(str(control_repository))
    validate_task_id(str(development_host))
    raw_repositories = payload.get("repositories")
    if not isinstance(raw_repositories, dict) or not raw_repositories:
        raise LabError("project manifest requires repositories")
    repositories: dict[str, Path] = {}
    stable_branches: dict[str, str] = {}
    checkout_modes: dict[str, str] = {}
    aliases: dict[str, str] = {}
    root = control_repository_root.resolve()
    for name, specification in raw_repositories.items():
        if not isinstance(name, str):
            raise LabError("project manifest has invalid repository name")
        validate_repository_name(name)
        expected_repository_fields = {
            "aliases",
            "default_path",
            "path_environment",
        }
        if schema in {
            NESTED_PROJECT_MANIFEST_SCHEMA_ID,
            PROJECT_MANIFEST_SCHEMA_ID,
        }:
            expected_repository_fields.add("path_base")
        if schema == PROJECT_MANIFEST_SCHEMA_ID:
            expected_repository_fields.update({"checkout_mode", "stable_branch"})
        if (
            not isinstance(specification, dict)
            or set(specification) != expected_repository_fields
        ):
            raise LabError(f"project manifest has invalid repository: {name}")
        path_environment = specification.get("path_environment")
        raw_aliases = specification.get("aliases")
        if (
            not isinstance(path_environment, str)
            or not _ENVIRONMENT_NAME.fullmatch(path_environment)
            or not isinstance(raw_aliases, list)
            or not all(isinstance(alias, str) for alias in raw_aliases)
        ):
            raise LabError(f"project manifest has invalid repository: {name}")
        default_path = _relative_path(
            specification.get("default_path"), repository=name
        )
        path_base = specification.get(
            "path_base", "control_repository_parent"
        )
        if path_base not in _PATH_BASES:
            raise LabError(
                f"project manifest has invalid path base for {name}: {path_base}"
            )
        configured = environ.get(path_environment)
        default_root = root if path_base == "control_repository" else root.parent
        repository_path = (
            Path(configured).expanduser().resolve()
            if configured
            else (default_root / default_path).resolve()
        )
        repositories[name] = repository_path
        stable_branches[name] = _stable_branch(
            specification.get("stable_branch", "main"),
            repository=name,
        )
        checkout_mode = specification.get("checkout_mode", "stable_branch")
        if checkout_mode not in _CHECKOUT_MODES:
            raise LabError(
                f"project manifest has invalid checkout mode: {name}"
            )
        checkout_modes[name] = str(checkout_mode)
        for alias in (name, *raw_aliases):
            validate_repository_name(alias)
            key = alias.casefold()
            previous = aliases.get(key)
            if previous is not None and previous != name:
                raise LabError(f"project manifest repeats repository alias: {alias}")
            aliases[key] = name
    if control_repository not in repositories:
        raise LabError("project manifest control repository is not declared")
    return ProjectManifest(
        project_id=str(project_id),
        control_repository=str(control_repository),
        default_development_host=str(development_host),
        repositories=dict(sorted(repositories.items())),
        stable_branches=dict(sorted(stable_branches.items())),
        checkout_modes=dict(sorted(checkout_modes.items())),
        aliases=dict(sorted(aliases.items())),
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
