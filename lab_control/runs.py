from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from . import git
from .environments import (
    ResolvedEnvironment,
    load_environment_profile,
    resolve_environment,
    source_overlay,
)
from .profiles import load_profile
from .snapshot_resolution import resolve_clean_snapshot_identity
from .workspace import LabError, WorkspaceManager, _atomic_write_text, validate_task_id
from .writer_lease import WriterLeaseManager, writer_lease_fingerprint

RUN_SCHEMA_ID = "lab.run.v1"
SNAPSHOT_SCHEMA_ID = "lab.snapshot.v1"
CONTROL_PLANE_VERSION = "0.3.0-m3-bootstrap"
DEVELOPMENT_EVIDENCE_TIER = "Development"
CANDIDATE_EVIDENCE_TIER = "Candidate"
PROTECTED_CONTROL_PLANE_REF = "refs/remotes/origin/main"
MAX_CONTROL_PLANE_FILE_BYTES = 2 * 1024 * 1024
MAX_COMMAND_LOG_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class CommandOutcome:
    command_id: str
    repository: str
    environment_id: str | None
    argv: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    start_error: str | None
    duration_seconds: float
    log_path: Path

    @property
    def passed(self) -> bool:
        return self.start_error is None and not self.timed_out and self.exit_code == 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, max_bytes: int = MAX_COMMAND_LOG_BYTES) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LabError(f"cannot safely open evidence file {path}: {exc}") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise LabError(f"evidence file is unsafe or exceeds its limit: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(chunk)
                if total > max_bytes:
                    raise LabError(f"evidence file exceeds its limit: {path}")
                digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ino != before.st_ino
            or after.st_dev != before.st_dev
        ):
            raise LabError(f"evidence file changed while it was collected: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _run_id(snapshot_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"R-{timestamp}-{snapshot_id[:8]}-{secrets.token_hex(2)}"


def _control_plane_hash(control_repository: Path) -> str:
    digest = hashlib.sha256()
    paths = [control_repository / "lab"] + sorted(
        (control_repository / "lab_control").glob("*.py")
    )
    for path in paths:
        digest.update(path.relative_to(control_repository).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            bytes.fromhex(
                _sha256_file(path, max_bytes=MAX_CONTROL_PLANE_FILE_BYTES)
            )
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _isolated_payload_environment(run_root: Path) -> dict[str, str]:
    isolated_home = run_root / ".home"
    temporary_root = run_root / ".tmp"
    cache_root = run_root / ".cache"
    for path in (isolated_home, temporary_root, cache_root):
        path.mkdir(mode=0o700)
    return {
        "HOME": str(isolated_home),
        "PATH": os.defpath,
        "TMPDIR": str(temporary_root),
        "XDG_CACHE_HOME": str(cache_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _open_new_log(path: Path) -> TextIO:
    parent_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        parent_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(path.parent, parent_flags)
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _limited_payload_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    prlimit = Path("/usr/bin/prlimit")
    if prlimit.is_symlink() or not prlimit.is_file() or not os.access(prlimit, os.X_OK):
        raise LabError("protected payload launcher requires /usr/bin/prlimit")
    return (
        str(prlimit),
        f"--fsize={MAX_COMMAND_LOG_BYTES}:{MAX_COMMAND_LOG_BYTES}",
        "--",
        *argv,
    )


def _validate_payload_executable(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> None:
    executable = argv[0]
    if "/" in executable:
        candidate = Path(executable)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        available = candidate.is_file() and os.access(candidate, os.X_OK)
    else:
        available = shutil.which(executable, path=environment.get("PATH")) is not None
    if not available:
        raise LabError(f"payload executable is unavailable: {executable}")


def _expanded_argv(
    argv: Sequence[str],
    *,
    python_argv: Sequence[str],
    repository_paths: Mapping[str, Path],
    run_root: Path,
) -> tuple[str, ...]:
    replacements = {"{run_root}": str(run_root)}
    replacements.update(
        {f"{{repo:{name}}}": str(path) for name, path in repository_paths.items()}
    )
    expanded: list[str] = []
    for argument in argv:
        if argument == "{python}":
            expanded.extend(python_argv)
            continue
        if "{python}" in argument:
            raise LabError("{python} must be a standalone argv element")
        value = argument
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        if "{repo:" in value or value == "{run_root}":
            raise LabError(f"unresolved command placeholder: {argument!r}")
        expanded.append(value)
    return tuple(expanded)


def _execute_command(
    *,
    command_id: str,
    repository: str,
    environment_id: str | None,
    argv: tuple[str, ...],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    log_path: Path,
) -> CommandOutcome:
    started = time.monotonic()
    timed_out = False
    exit_code: int | None = None
    try:
        log_context = _open_new_log(log_path)
    except OSError as exc:
        return CommandOutcome(
            command_id=command_id,
            repository=repository,
            environment_id=environment_id,
            argv=argv,
            exit_code=None,
            timed_out=False,
            start_error=f"refused unsafe or occupied log path: {exc}",
            duration_seconds=round(time.monotonic() - started, 6),
            log_path=log_path,
        )
    with log_context as log_stream:
        log_stream.write(f"command_id: {command_id}\n")
        log_stream.write(f"repository: {repository}\n")
        log_stream.write(f"cwd: {cwd}\n")
        log_stream.write("argv: " + json.dumps(argv) + "\n\n")
        log_stream.flush()
        try:
            _validate_payload_executable(argv, cwd=cwd, environment=environment)
            limited_argv = _limited_payload_argv(argv)
            process = subprocess.Popen(
                limited_argv,
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except (LabError, OSError) as exc:
            log_stream.write(f"lab: command could not start: {exc}\n")
            return CommandOutcome(
                command_id=command_id,
                repository=repository,
                environment_id=environment_id,
                argv=argv,
                exit_code=None,
                timed_out=False,
                start_error=str(exc),
                duration_seconds=round(time.monotonic() - started, 6),
                log_path=log_path,
            )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                exit_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                exit_code = process.wait(timeout=5)
            log_stream.write(f"\nlab: command timed out after {timeout_seconds}s\n")
    return CommandOutcome(
        command_id=command_id,
        repository=repository,
        environment_id=environment_id,
        argv=argv,
        exit_code=exit_code,
        timed_out=timed_out,
        start_error=None,
        duration_seconds=round(time.monotonic() - started, 6),
        log_path=log_path,
    )


def _verify_snapshot_sources(
    repository_paths: Mapping[str, Path],
    expected_repositories: Mapping[str, Mapping[str, object]],
) -> tuple[bool, list[str]]:
    source_stable = True
    source_errors: list[str] = []
    for name, path in repository_paths.items():
        raw_expected = expected_repositories[name]
        try:
            actual_commit = git.current_commit(path)
            actual_tree = git.tree_oid(path, actual_commit)
            dirty = bool(git.status_short(path))
            actual_submodules, submodules_ready = git.submodule_status(path)
        except git.GitError as exc:
            source_stable = False
            source_errors.append(f"{name}: {exc}")
            continue
        if (
            actual_commit != raw_expected["commit"]
            or actual_tree != raw_expected["tree_oid"]
            or actual_submodules != raw_expected.get("submodules", {})
            or not submodules_ready
            or dirty
        ):
            source_stable = False
            source_errors.append(
                f"{name}: expected {raw_expected['commit']}/{raw_expected['tree_oid']}, "
                f"got {actual_commit}/{actual_tree}, dirty={dirty}"
            )
    return source_stable, source_errors


class RunManager:
    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        self.workspace_manager = workspace_manager
        configured_profiles = workspace_manager.environ.get("LAB_PROFILE_ROOT")
        self.profile_root = (
            Path(configured_profiles).expanduser().resolve()
            if configured_profiles
            else workspace_manager.control_repository / "integration" / "profiles"
        )
        configured_runtime = workspace_manager.environ.get("LAB_RUNTIME_ROOT")
        self.runtime_root = (
            Path(configured_runtime).expanduser().resolve()
            if configured_runtime
            else workspace_manager.lab_home / "Runtime"
        )
        self.runs_root = self.runtime_root / "tasks"
        self.reviews_root = workspace_manager.worktrees_root / "_review"
        configured_baseline = workspace_manager.environ.get(
            "LAB_BASELINE_LOCK"
        )
        self.baseline_path = (
            Path(configured_baseline).expanduser().resolve()
            if configured_baseline
            else workspace_manager.control_repository / "stack" / "baseline.json"
        )
        configured_environments = workspace_manager.environ.get(
            "LAB_ENVIRONMENT_PROFILE_ROOT"
        )
        self.environment_profile_root = (
            Path(configured_environments).expanduser().resolve()
            if configured_environments
            else workspace_manager.control_repository / "integration" / "environments"
        )

    def validate_evidence_tier(self, evidence_tier: str) -> None:
        if evidence_tier == DEVELOPMENT_EVIDENCE_TIER:
            return
        if evidence_tier != CANDIDATE_EVIDENCE_TIER:
            raise LabError(f"unsupported evidence tier: {evidence_tier}")
        control_repository = self.workspace_manager.control_repository
        try:
            branch = git.current_branch(control_repository)
            dirty = git.status_short(control_repository)
            protected_exists = git.ref_exists(
                control_repository,
                PROTECTED_CONTROL_PLANE_REF,
            )
            current_commit = git.current_commit(control_repository)
            protected_commit = (
                git.resolve_commit(control_repository, PROTECTED_CONTROL_PLANE_REF)
                if protected_exists
                else None
            )
        except git.GitError as exc:
            raise LabError(f"cannot validate Candidate control plane: {exc}") from exc
        if branch != "main" or dirty:
            raise LabError(
                "Candidate evidence requires a clean protected control-plane main"
            )
        if protected_commit is None or current_commit != protected_commit:
            raise LabError(
                "Candidate evidence requires control-plane HEAD to equal "
                "origin/main"
            )

    def run_clean_profile(
        self,
        task_id: str,
        profile_id: str,
        *,
        evidence_tier: str = DEVELOPMENT_EVIDENCE_TIER,
    ) -> dict[str, object]:
        self.validate_evidence_tier(evidence_tier)
        profile = load_profile(self.profile_root, profile_id)
        actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if (
            profile.python_major_minor is not None
            and profile.python_major_minor != actual_python
        ):
            raise LabError(
                f"profile {profile_id} requires Python {profile.python_major_minor}, "
                f"current interpreter is {actual_python} ({sys.executable})"
            )
        snapshot_id, snapshot_identity = resolve_clean_snapshot_identity(
            self.workspace_manager,
            task_id=task_id,
            profile_id=profile.profile_id,
            profile_sha256=profile.sha256,
            repositories=profile.repositories,
            baseline_path=self.baseline_path,
        )
        writer_leases = writer_lease_fingerprint(
            WriterLeaseManager(self.workspace_manager).inspect_task(task_id)
        )
        raw_snapshot_repositories = snapshot_identity.get("repositories")
        if not isinstance(raw_snapshot_repositories, dict):
            raise LabError("resolved snapshot has no repositories")
        snapshot_repositories = raw_snapshot_repositories
        run_id = _run_id(snapshot_id)
        snapshot_root, repository_paths = self._prepare_snapshot(
            task_id=task_id,
            snapshot_id=snapshot_id,
            run_id=run_id,
            identity=snapshot_identity,
        )

        run_root = self.runs_root / task_id / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        run_manifest_path = run_root / "run.yaml"
        control_commit = git.current_commit(self.workspace_manager.control_repository)
        control_tree = git.tree_oid(
            self.workspace_manager.control_repository, control_commit
        )
        control_dirty = bool(
            git.status_short(self.workspace_manager.control_repository)
        )
        run_manifest: dict[str, object] = {
            "schema": RUN_SCHEMA_ID,
            "run_id": run_id,
            "task_id": task_id,
            "snapshot_id": snapshot_id,
            "snapshot_root": str(snapshot_root),
            "run_root": str(run_root),
            "snapshot_workspace": str(
                snapshot_root / f"{task_id}-{snapshot_id}.code-workspace"
            ),
            "profile": {
                "profile_id": profile.profile_id,
                "sha256": profile.sha256,
                "risk_level": profile.risk_level,
                "execution": profile.execution,
                "python_major_minor": profile.python_major_minor,
            },
            "control_plane": {
                "version": CONTROL_PLANE_VERSION,
                "source_commit": control_commit,
                "source_tree_oid": control_tree,
                "source_dirty": control_dirty,
                "executor_sha256": _control_plane_hash(
                    self.workspace_manager.control_repository
                ),
            },
            "evidence_tier": evidence_tier,
            "baseline": snapshot_identity.get("baseline"),
            "repository_links": snapshot_identity.get("repository_links", []),
            "repositories": snapshot_repositories,
            "writer_leases": writer_leases,
            "environment": {
                "control_python_executable": sys.executable,
                "control_python_version": sys.version,
                "python_no_user_site": True,
                "python_no_bytecode": True,
                "source_overlay": [],
                "resolved": {},
                "errors": [],
            },
            "status": "PREPARING",
            "started_at": _utc_now(),
            "finished_at": None,
            "commands": [],
            "outcome": None,
            "artifacts": [],
        }
        _atomic_write_text(
            run_manifest_path,
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        )

        environment = _isolated_payload_environment(run_root)
        environment.update(
            {
                "TASK_ID": task_id,
                "RUN_ID": run_id,
                "RUN_ROOT": str(run_root),
            }
        )
        for name, path in repository_paths.items():
            key = "".join(
                character if character.isalnum() else "_" for character in name
            )
            environment[f"{key.upper()}_ROOT"] = str(path)

        environment_ids = tuple(
            dict.fromkeys(
                command.environment_id
                for command in profile.commands
                if command.environment_id is not None
            )
        )
        resolved_environments: dict[str, ResolvedEnvironment] = {}
        overlay_paths: tuple[Path, ...] = ()
        try:
            if environment_ids:
                environment_profiles = tuple(
                    load_environment_profile(
                        self.environment_profile_root,
                        environment_id,
                    )
                    for environment_id in environment_ids
                )
                command_repositories = {
                    command.environment_id: command.repository
                    for command in profile.commands
                    if command.environment_id is not None
                }
                for environment_profile in environment_profiles:
                    if (
                        command_repositories[environment_profile.environment_id]
                        != environment_profile.repository
                    ):
                        raise LabError(
                            f"environment {environment_profile.environment_id} is "
                            "bound to the wrong repository"
                        )
                overlay_paths = source_overlay(
                    environment_profiles,
                    repository_paths,
                )
                environment["PYTHONPATH"] = os.pathsep.join(
                    str(path) for path in overlay_paths
                )
                for environment_profile in environment_profiles:
                    resolved_environments[environment_profile.environment_id] = (
                        resolve_environment(
                            environment_profile,
                            repository_path=repository_paths[
                                environment_profile.repository
                            ],
                            environment=environment,
                        )
                    )
        except LabError as exc:
            source_stable, source_errors = _verify_snapshot_sources(
                repository_paths,
                snapshot_repositories,
            )
            environment_record = run_manifest["environment"]
            if isinstance(environment_record, dict):
                environment_record.update(
                    {
                        "source_overlay": [str(path) for path in overlay_paths],
                        "resolved": {
                            environment_id: resolved.receipt
                            for environment_id, resolved in resolved_environments.items()
                        },
                        "errors": [str(exc)],
                    }
                )
            run_manifest.update(
                {
                    "status": "FAILED",
                    "finished_at": _utc_now(),
                    "outcome": {
                        "classification": (
                            "INVALID_ENVIRONMENT" if source_stable else "INVALID_SOURCE"
                        ),
                        "execution_completed": False,
                        "assertions_passed": False,
                        "source_stable": source_stable,
                        "source_errors": source_errors,
                        "environment_valid": False,
                        "evidence_complete": True,
                    },
                }
            )
            _atomic_write_text(
                run_manifest_path,
                json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            )
            return run_manifest

        environment_record = run_manifest["environment"]
        if isinstance(environment_record, dict):
            environment_record.update(
                {
                    "payload_environment_keys": sorted(environment),
                    "isolated_home": environment["HOME"],
                    "source_overlay": [str(path) for path in overlay_paths],
                    "resolved": {
                        environment_id: resolved.receipt
                        for environment_id, resolved in resolved_environments.items()
                    },
                }
            )
        run_manifest["status"] = "RUNNING"
        _atomic_write_text(
            run_manifest_path,
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        )

        outcomes: list[CommandOutcome] = []
        for command in profile.commands:
            resolved_environment = (
                resolved_environments.get(command.environment_id)
                if command.environment_id is not None
                else None
            )
            argv = _expanded_argv(
                command.argv,
                python_argv=(
                    resolved_environment.python_argv
                    if resolved_environment is not None
                    else (sys.executable,)
                ),
                repository_paths=repository_paths,
                run_root=run_root,
            )
            log_path = run_root / f"{len(outcomes) + 1:02d}-{command.command_id}.log"
            outcome = _execute_command(
                command_id=command.command_id,
                repository=command.repository,
                environment_id=command.environment_id,
                argv=argv,
                cwd=repository_paths[command.repository],
                environment=environment,
                timeout_seconds=command.timeout_seconds,
                log_path=log_path,
            )
            outcomes.append(outcome)
            if not outcome.passed:
                break

        source_stable, source_errors = _verify_snapshot_sources(
            repository_paths,
            snapshot_repositories,
        )

        if not source_stable:
            classification = "INVALID_SOURCE"
        elif all(outcome.passed for outcome in outcomes) and len(outcomes) == len(
            profile.commands
        ):
            classification = "PASSED"
        elif any(outcome.timed_out or outcome.start_error for outcome in outcomes):
            classification = "INFRASTRUCTURE_FAILURE"
        else:
            classification = "FAILED_ASSERTION"

        command_records = [
            {
                "command_id": outcome.command_id,
                "repository": outcome.repository,
                "environment_id": outcome.environment_id,
                "argv": list(outcome.argv),
                "cwd": str(repository_paths[outcome.repository]),
                "exit_code": outcome.exit_code,
                "timed_out": outcome.timed_out,
                "start_error": outcome.start_error,
                "duration_seconds": outcome.duration_seconds,
                "log": outcome.log_path.name,
            }
            for outcome in outcomes
        ]
        artifacts: list[dict[str, object]] = []
        evidence_errors: list[str] = []
        for record in command_records:
            log = run_root / str(record["log"])
            try:
                digest = _sha256_file(log)
                size_bytes = os.lstat(log).st_size
            except (LabError, OSError) as exc:
                evidence_errors.append(str(exc))
                continue
            artifacts.append(
                {
                    "path": record["log"],
                    "sha256": digest,
                    "size_bytes": size_bytes,
                }
            )
        run_manifest.update(
            {
                "status": "COMPLETED" if classification == "PASSED" else "FAILED",
                "finished_at": _utc_now(),
                "commands": command_records,
                "outcome": {
                    "classification": classification,
                    "execution_completed": len(outcomes) == len(profile.commands)
                    and all(
                        not outcome.timed_out and outcome.start_error is None
                        for outcome in outcomes
                    ),
                    "assertions_passed": classification == "PASSED",
                    "source_stable": source_stable,
                    "source_errors": source_errors,
                    "environment_valid": True,
                    "evidence_complete": not evidence_errors,
                    "evidence_errors": evidence_errors,
                },
                "artifacts": artifacts,
            }
        )
        _atomic_write_text(
            run_manifest_path,
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        )
        return run_manifest

    def _prepare_snapshot(
        self,
        *,
        task_id: str,
        snapshot_id: str,
        run_id: str,
        identity: dict[str, object],
    ) -> tuple[Path, dict[str, Path]]:
        snapshot_root = self.reviews_root / task_id / snapshot_id / run_id
        snapshot_manifest_path = snapshot_root / "snapshot.yaml"
        raw_repositories = identity["repositories"]
        if not isinstance(raw_repositories, dict):
            raise LabError("snapshot identity has no repositories")

        if snapshot_root.exists():
            if not snapshot_manifest_path.is_file():
                raise LabError(f"existing snapshot is incomplete: {snapshot_root}")
            try:
                existing = json.loads(
                    snapshot_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise LabError(
                    f"cannot read existing snapshot {snapshot_root}: {exc}"
                ) from exc
            if not isinstance(existing, dict):
                raise LabError(f"invalid existing snapshot manifest: {snapshot_root}")
            if existing.get("identity") != identity:
                raise LabError(f"snapshot identity collision: {snapshot_root}")
            existing_paths = {name: snapshot_root / name for name in raw_repositories}
            for name, path in existing_paths.items():
                raw_repository = raw_repositories[name]
                if not isinstance(raw_repository, dict):
                    raise LabError(f"invalid snapshot repository: {name}")
                try:
                    if git.current_branch(path) is not None:
                        raise LabError(f"snapshot worktree is not detached: {path}")
                    if git.current_commit(path) != raw_repository["commit"]:
                        raise LabError(f"snapshot commit drift: {path}")
                    if git.tree_oid(path) != raw_repository["tree_oid"]:
                        raise LabError(f"snapshot tree drift: {path}")
                    actual_submodules, submodules_ready = git.submodule_status(path)
                    if (
                        actual_submodules != raw_repository.get("submodules", {})
                        or not submodules_ready
                    ):
                        raise LabError(f"snapshot submodule drift: {path}")
                    if git.status_short(path):
                        raise LabError(
                            f"snapshot worktree is dirty/quarantined: {path}"
                        )
                except git.GitError as exc:
                    raise LabError(str(exc)) from exc
            return snapshot_root, existing_paths

        snapshot_root.mkdir(parents=True, exist_ok=False)
        created: list[tuple[Path, Path]] = []
        paths: dict[str, Path] = {}
        try:
            for name, raw_repository in raw_repositories.items():
                if not isinstance(name, str) or not isinstance(raw_repository, dict):
                    raise LabError(f"invalid snapshot repository: {name}")
                management_path = Path(str(raw_repository["management_path"]))
                target = snapshot_root / name
                git.create_detached_worktree(
                    management_path,
                    target=target,
                    commit=str(raw_repository["commit"]),
                )
                created.append((management_path, target))
                raw_submodules = raw_repository.get("submodules", {})
                if not isinstance(raw_submodules, dict):
                    raise LabError(f"invalid snapshot submodules: {name}")
                if raw_submodules:
                    git.update_submodules_no_fetch(target)
                    actual_submodules, submodules_ready = git.submodule_status(target)
                    if actual_submodules != raw_submodules or not submodules_ready:
                        raise LabError(f"snapshot submodule mismatch: {name}")
                paths[name] = target
            snapshot_manifest = {
                "schema": SNAPSHOT_SCHEMA_ID,
                "snapshot_id": snapshot_id,
                "created_at": _utc_now(),
                "identity": identity,
            }
            _atomic_write_text(
                snapshot_manifest_path,
                json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n",
            )
            workspace = {
                "folders": [{"name": name, "path": name} for name in raw_repositories],
                "settings": {"window.title": f"[{task_id} Run snapshot]"},
            }
            _atomic_write_text(
                snapshot_root / f"{task_id}-{snapshot_id}.code-workspace",
                json.dumps(workspace, indent=2, sort_keys=True) + "\n",
            )
            return snapshot_root, paths
        except Exception as exc:
            cleanup_errors: list[str] = []
            for management_path, target in reversed(created):
                try:
                    git.remove_worktree(management_path, target)
                except git.GitError as cleanup_exc:
                    cleanup_errors.append(str(cleanup_exc))
            if snapshot_root.exists() and not any(
                target.exists() for _, target in created
            ):
                shutil.rmtree(snapshot_root)
            detail = f"snapshot preparation failed: {exc}"
            if cleanup_errors:
                detail += "; cleanup requires attention: " + "; ".join(cleanup_errors)
            raise LabError(detail) from exc

    def list_runs(self, task_id: str | None = None) -> list[dict[str, object]]:
        if task_id is not None:
            validate_task_id(task_id)
            task_roots = [self.runs_root / task_id]
        elif self.runs_root.is_dir():
            task_roots = sorted(
                path for path in self.runs_root.iterdir() if path.is_dir()
            )
        else:
            task_roots = []
        manifests: list[dict[str, object]] = []
        for task_root in task_roots:
            if not task_root.is_dir():
                continue
            for run_root in task_root.iterdir():
                manifest_path = run_root / "run.yaml"
                if not manifest_path.is_file():
                    continue
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise LabError(
                        f"cannot read Run manifest {manifest_path}: {exc}"
                    ) from exc
                self._validate_run_manifest(
                    payload,
                    manifest_path=manifest_path,
                    expected_task_id=task_root.name,
                    expected_run_id=run_root.name,
                )
                manifests.append(payload)
        return sorted(
            manifests,
            key=lambda payload: str(payload.get("started_at", "")),
            reverse=True,
        )

    def find_run(self, run_id: str) -> dict[str, object]:
        validate_task_id(run_id)
        matches = [
            payload for payload in self.list_runs() if payload.get("run_id") == run_id
        ]
        if not matches:
            raise LabError(f"Run does not exist: {run_id}")
        if len(matches) > 1:
            raise LabError(f"Run ID is not unique: {run_id}")
        return matches[0]

    def _validate_run_manifest(
        self,
        payload: object,
        *,
        manifest_path: Path,
        expected_task_id: str,
        expected_run_id: str,
    ) -> None:
        if not isinstance(payload, dict) or payload.get("schema") != RUN_SCHEMA_ID:
            raise LabError(f"unsupported Run manifest: {manifest_path}")
        if payload.get("task_id") != expected_task_id:
            raise LabError(f"Run task identity mismatch: {manifest_path}")
        if payload.get("run_id") != expected_run_id:
            raise LabError(f"Run identity mismatch: {manifest_path}")
        validate_task_id(expected_task_id)
        validate_task_id(expected_run_id)
        snapshot_id = str(payload.get("snapshot_id", ""))
        validate_task_id(snapshot_id)
        expected_run_root = manifest_path.parent.resolve()
        configured_run_root = Path(str(payload.get("run_root", ""))).resolve()
        if configured_run_root != expected_run_root:
            raise LabError(f"Run root mismatch: {manifest_path}")
        expected_snapshot_root = (
            self.reviews_root / expected_task_id / snapshot_id / expected_run_id
        ).resolve()
        configured_snapshot_root = Path(str(payload.get("snapshot_root", ""))).resolve()
        if configured_snapshot_root != expected_snapshot_root:
            raise LabError(f"Run snapshot root mismatch: {manifest_path}")
        expected_workspace = (
            expected_snapshot_root / f"{expected_task_id}-{snapshot_id}.code-workspace"
        )
        configured_workspace = Path(
            str(payload.get("snapshot_workspace", ""))
        ).resolve()
        if configured_workspace != expected_workspace:
            raise LabError(f"Run snapshot workspace mismatch: {manifest_path}")
