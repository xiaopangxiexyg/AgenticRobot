from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .workspace import LabError, validate_repository_name, validate_task_id

PROFILE_SCHEMA_ID = "lab.profile.v1"
ALLOWED_LOCAL_RISK_LEVELS = frozenset({"R0", "R1"})


@dataclass(frozen=True)
class ProfileCommand:
    command_id: str
    repository: str
    environment_id: str | None
    argv: tuple[str, ...]
    timeout_seconds: int


@dataclass(frozen=True)
class TestProfile:
    profile_id: str
    description: str
    execution: str
    risk_level: str
    requires_clean: bool
    python_major_minor: str | None
    commands: tuple[ProfileCommand, ...]
    path: Path
    sha256: str

    @property
    def repositories(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(command.repository for command in self.commands))


def load_profile(profile_root: Path, profile_id: str) -> TestProfile:
    validate_task_id(profile_id)
    root = profile_root.resolve()
    path = (root / f"{profile_id}.json").resolve()
    if path.parent != root:
        raise LabError(f"profile path escapes trusted root: {profile_id}")
    if not path.is_file():
        raise LabError(f"unknown test profile: {profile_id} ({path})")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read test profile {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != PROFILE_SCHEMA_ID:
        raise LabError(f"unsupported test profile schema: {path}")
    if payload.get("profile_id") != profile_id:
        raise LabError(f"test profile identity mismatch: {path}")

    execution = payload.get("execution")
    if execution != "local":
        raise LabError(
            f"M2 bootstrap only permits local profiles, got {execution!r}: {path}"
        )
    risk_level = payload.get("risk_level")
    if risk_level not in ALLOWED_LOCAL_RISK_LEVELS:
        raise LabError(
            f"M2 bootstrap only permits R0/R1 profiles, got {risk_level!r}: {path}"
        )
    if payload.get("requires_clean") is not True:
        raise LabError(f"M2 bootstrap requires clean profiles: {path}")
    python_major_minor = payload.get("python_major_minor")
    if python_major_minor is not None and (
        not isinstance(python_major_minor, str)
        or not python_major_minor.replace(".", "", 1).isdigit()
        or python_major_minor.count(".") != 1
    ):
        raise LabError(f"invalid python_major_minor in profile {path}")

    raw_commands = payload.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        raise LabError(f"test profile has no commands: {path}")
    commands: list[ProfileCommand] = []
    command_ids: set[str] = set()
    repository_environments: dict[str, str] = {}
    environment_binding_count = 0
    for index, raw_command in enumerate(raw_commands):
        if not isinstance(raw_command, dict):
            raise LabError(f"invalid command {index} in profile {path}")
        command_id = str(raw_command.get("command_id", ""))
        validate_task_id(command_id)
        if command_id in command_ids:
            raise LabError(f"duplicate command ID in profile {path}: {command_id}")
        command_ids.add(command_id)
        repository = validate_repository_name(str(raw_command.get("repository", "")))
        raw_environment_id = raw_command.get("environment_id")
        environment_id: str | None = None
        if raw_environment_id is not None:
            if not isinstance(raw_environment_id, str):
                raise LabError(f"invalid environment ID for {command_id} in {path}")
            environment_id = validate_task_id(raw_environment_id)
            environment_binding_count += 1
            previous = repository_environments.setdefault(repository, environment_id)
            if previous != environment_id:
                raise LabError(
                    f"repository {repository} uses multiple environments in {path}"
                )
        raw_argv = raw_command.get("argv")
        if (
            not isinstance(raw_argv, list)
            or not raw_argv
            or not all(isinstance(value, str) and value for value in raw_argv)
        ):
            raise LabError(f"invalid argv for {command_id} in profile {path}")
        timeout_seconds = raw_command.get("timeout_seconds", 300)
        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or not 1 <= timeout_seconds <= 3600
        ):
            raise LabError(f"invalid timeout for {command_id} in profile {path}")
        commands.append(
            ProfileCommand(
                command_id=command_id,
                repository=repository,
                environment_id=environment_id,
                argv=tuple(raw_argv),
                timeout_seconds=timeout_seconds,
            )
        )

    if environment_binding_count not in {0, len(commands)}:
        raise LabError(
            f"Profile must bind every command to an environment or none: {path}"
        )

    description = payload.get("description", "")
    if not isinstance(description, str):
        raise LabError(f"invalid profile description: {path}")
    return TestProfile(
        profile_id=profile_id,
        description=description,
        execution=execution,
        risk_level=risk_level,
        requires_clean=True,
        python_major_minor=python_major_minor,
        commands=tuple(commands),
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
