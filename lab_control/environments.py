from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .workspace import LabError, validate_repository_name, validate_task_id

ENVIRONMENT_SCHEMA_ID = "lab.environment.v1"

_PROBE = """
import importlib.util
import json
import pathlib
import sys

origins = {}
for module in sys.argv[1:]:
    spec = importlib.util.find_spec(module)
    origin = spec.origin if spec is not None else None
    if origin is None and spec is not None and spec.submodule_search_locations:
        origin = next(iter(spec.submodule_search_locations), None)
    origins[module] = str(pathlib.Path(origin).resolve()) if origin else None
print(json.dumps({
    "python_executable": str(pathlib.Path(sys.executable).resolve()),
    "python_prefix": str(pathlib.Path(sys.prefix).resolve()),
    "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    "import_origins": origins,
}, sort_keys=True))
""".strip()


@dataclass(frozen=True)
class EnvironmentProfile:
    environment_id: str
    repository: str
    resolver_kind: str
    resolver_value: str
    python_major_minor: str
    import_modules: tuple[str, ...]
    source_roots: tuple[str, ...]
    path: Path
    sha256: str


@dataclass(frozen=True)
class ResolvedEnvironment:
    profile: EnvironmentProfile
    python_argv: tuple[str, ...]
    receipt: dict[str, object]


def inspect_environments(
    profile_root: Path,
    environment_ids: Sequence[str],
    *,
    repository_paths: Mapping[str, Path],
    environ: Mapping[str, str],
) -> dict[str, object]:
    profiles: list[EnvironmentProfile] = []
    errors: list[str] = []
    for environment_id in environment_ids:
        try:
            profiles.append(load_environment_profile(profile_root, environment_id))
        except LabError as exc:
            errors.append(str(exc))
    if errors:
        return {
            "schema": "lab.environment-inspection.v1",
            "status": "INVALID_ENVIRONMENT",
            "source_overlay": [],
            "environments": {},
            "errors": errors,
        }
    try:
        overlay = source_overlay(tuple(profiles), dict(repository_paths))
    except LabError as exc:
        return {
            "schema": "lab.environment-inspection.v1",
            "status": "INVALID_ENVIRONMENT",
            "source_overlay": [],
            "environments": {},
            "errors": [str(exc)],
        }
    environment = dict(environ)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(path) for path in overlay),
        }
    )
    receipts: dict[str, object] = {}
    for profile in profiles:
        repository_path = repository_paths.get(profile.repository)
        if repository_path is None:
            errors.append(
                f"environment {profile.environment_id} references unavailable "
                f"repository: {profile.repository}"
            )
            continue
        try:
            receipts[profile.environment_id] = resolve_environment(
                profile,
                repository_path=repository_path,
                environment=environment,
            ).receipt
        except LabError as exc:
            errors.append(str(exc))
    return {
        "schema": "lab.environment-inspection.v1",
        "status": "READY" if not errors else "INVALID_ENVIRONMENT",
        "source_overlay": [str(path) for path in overlay],
        "environments": receipts,
        "errors": errors,
    }


def _string_list(payload: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(value, str) and value for value in payload)
        or len(payload) != len(set(payload))
    ):
        raise LabError(f"{label} must be a unique non-empty string list")
    return tuple(payload)


def load_environment_profile(
    profile_root: Path,
    environment_id: str,
) -> EnvironmentProfile:
    validate_task_id(environment_id)
    root = profile_root.resolve()
    path = (root / f"{environment_id}.json").resolve()
    if path.parent != root or not path.is_file():
        raise LabError(f"unknown environment Profile: {environment_id} ({path})")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read environment Profile {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != ENVIRONMENT_SCHEMA_ID:
        raise LabError(f"unsupported environment Profile schema: {path}")
    if payload.get("environment_id") != environment_id:
        raise LabError(f"environment Profile identity mismatch: {path}")
    repository = validate_repository_name(str(payload.get("repository", "")))
    python_major_minor = payload.get("python_major_minor")
    if (
        not isinstance(python_major_minor, str)
        or python_major_minor.count(".") != 1
        or not python_major_minor.replace(".", "").isdigit()
    ):
        raise LabError(f"invalid environment Python identity: {path}")
    resolver = payload.get("resolver")
    if not isinstance(resolver, dict):
        raise LabError(f"environment Profile lacks resolver: {path}")
    resolver_kind = resolver.get("kind")
    if resolver_kind == "conda":
        resolver_value = str(resolver.get("environment", ""))
        validate_task_id(resolver_value)
    elif resolver_kind == "python":
        raw_executable = resolver.get("executable")
        if not isinstance(raw_executable, str):
            raise LabError(f"python resolver lacks executable: {path}")
        executable = Path(raw_executable)
        if not executable.is_absolute() or ".." in executable.parts:
            raise LabError(f"python resolver requires a safe absolute path: {path}")
        resolver_value = str(executable)
    else:
        raise LabError(f"unsupported environment resolver: {resolver_kind!r}")
    import_modules = _string_list(
        payload.get("import_modules"),
        label="import_modules",
    )
    source_roots = _string_list(payload.get("source_roots"), label="source_roots")
    for relative in source_roots:
        pure_path = PurePosixPath(relative)
        if relative.startswith(("/", "~")) or ".." in pure_path.parts:
            raise LabError(f"unsafe environment source root: {relative!r}")
    return EnvironmentProfile(
        environment_id=environment_id,
        repository=repository,
        resolver_kind=str(resolver_kind),
        resolver_value=resolver_value,
        python_major_minor=python_major_minor,
        import_modules=import_modules,
        source_roots=source_roots,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def source_overlay(
    profiles: tuple[EnvironmentProfile, ...],
    repository_paths: dict[str, Path],
) -> tuple[Path, ...]:
    roots: list[Path] = []
    for profile in profiles:
        repository = repository_paths.get(profile.repository)
        if repository is None:
            raise LabError(
                f"environment {profile.environment_id} references absent repository: "
                f"{profile.repository}"
            )
        for relative in profile.source_roots:
            root = (repository / relative).resolve()
            if not root.is_relative_to(repository) or not root.is_dir():
                raise LabError(
                    f"environment source root is missing/escapes snapshot: "
                    f"{profile.environment_id}/{relative}"
                )
            if root not in roots:
                roots.append(root)
    return tuple(roots)


def _python_commands(
    profile: EnvironmentProfile,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if profile.resolver_kind == "python":
        executable = Path(profile.resolver_value)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise LabError(
                f"environment Python is missing/not executable: {profile.environment_id}"
            )
        command = (str(executable.resolve()),)
        return command, command
    conda = shutil.which("conda")
    if conda is None:
        raise LabError(
            f"conda is unavailable for environment: {profile.environment_id}"
        )
    base = (str(Path(conda).resolve()), "run", "-n", profile.resolver_value)
    return (*base, "python"), (*base, "--no-capture-output", "python")


def resolve_environment(
    profile: EnvironmentProfile,
    *,
    repository_path: Path,
    environment: dict[str, str],
) -> ResolvedEnvironment:
    probe_python, runtime_python = _python_commands(profile)
    with tempfile.TemporaryDirectory(prefix="lab-env-probe-") as probe_cwd:
        completed = subprocess.run(
            [*probe_python, "-c", _PROBE, *profile.import_modules],
            cwd=probe_cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LabError(
            f"environment probe failed for {profile.environment_id}: {detail}"
        )
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LabError(
            f"environment probe returned invalid JSON for {profile.environment_id}: {exc}"
        ) from exc
    if not isinstance(receipt, dict):
        raise LabError(
            f"environment probe returned non-object: {profile.environment_id}"
        )
    if receipt.get("python_major_minor") != profile.python_major_minor:
        raise LabError(
            f"environment {profile.environment_id} requires Python "
            f"{profile.python_major_minor}, got {receipt.get('python_major_minor')}"
        )
    import_origins = receipt.get("import_origins")
    if not isinstance(import_origins, dict) or set(import_origins) != set(
        profile.import_modules
    ):
        raise LabError(
            f"environment import receipt is incomplete: {profile.environment_id}"
        )
    for module in profile.import_modules:
        raw_origin = import_origins[module]
        if not isinstance(raw_origin, str):
            raise LabError(
                f"environment cannot resolve {module}: {profile.environment_id}"
            )
        origin = Path(raw_origin)
        if not origin.is_relative_to(repository_path):
            raise LabError(
                f"environment {profile.environment_id} imports {module} from "
                f"the wrong checkout: {origin}"
            )
    return ResolvedEnvironment(
        profile=profile,
        python_argv=runtime_python,
        receipt={
            "environment_id": profile.environment_id,
            "profile_sha256": profile.sha256,
            "resolver_kind": profile.resolver_kind,
            "resolver_value": profile.resolver_value,
            **receipt,
        },
    )
