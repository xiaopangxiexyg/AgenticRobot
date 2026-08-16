from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .runner_protocol import RunnerCoordinator
from .runner_staging import ExactSnapshotStager
from .workspace import (
    LabError,
    _atomic_write_text,
    validate_repository_name,
    validate_task_id,
)

RUNNER_COMMAND_MAP_SCHEMA_ID = "lab.runner-command-map.v1"
COMMAND_EXECUTION_RECEIPT_SCHEMA_ID = "lab.command-execution-receipt.v1"
_MAXIMUM_COMMAND_MAP_BYTES = 256 * 1024
_MAXIMUM_ARGUMENTS = 256
_MAXIMUM_ARGUMENT_BYTES = 16 * 1024
_POLL_SECONDS = 0.05


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _open_new_text(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            yield stream
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_new_text(path: Path, content: str) -> None:
    with _open_new_text(path) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class TrustedCommand:
    command_id: str
    repository: str
    argv: tuple[str, ...]
    timeout_seconds: int


@dataclass(frozen=True)
class RunnerCommandMap:
    runner_id: str
    commands: tuple[TrustedCommand, ...]
    path: Path
    sha256: str

    @property
    def by_id(self) -> dict[str, TrustedCommand]:
        return {command.command_id: command for command in self.commands}


def load_runner_command_map(root: Path, runner_id: str) -> RunnerCommandMap:
    validate_task_id(runner_id)
    trusted_root = root.expanduser().resolve()
    candidate = trusted_root / f"{runner_id}.json"
    if (
        candidate.parent != trusted_root
        or candidate.is_symlink()
        or not candidate.is_file()
    ):
        raise LabError(f"unknown or unsafe runner command map: {runner_id}")
    path = candidate.resolve()
    if path.parent != trusted_root:
        raise LabError(f"runner command map path escapes trusted root: {runner_id}")
    try:
        if path.stat().st_size > _MAXIMUM_COMMAND_MAP_BYTES:
            raise LabError(f"runner command map exceeds its size limit: {path}")
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read runner command map {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "runner_id",
        "commands",
    }:
        raise LabError(f"runner command map has an unexpected shape: {path}")
    if (
        payload.get("schema") != RUNNER_COMMAND_MAP_SCHEMA_ID
        or payload.get("runner_id") != runner_id
    ):
        raise LabError(f"runner command map identity mismatch: {path}")
    raw_commands = payload.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        raise LabError(f"runner command map must contain commands: {path}")
    commands: list[TrustedCommand] = []
    command_ids: set[str] = set()
    for raw_command in raw_commands:
        if not isinstance(raw_command, dict) or set(raw_command) != {
            "command_id",
            "repository",
            "argv",
            "timeout_seconds",
        }:
            raise LabError(f"runner command has an unexpected shape: {path}")
        command_id = validate_task_id(str(raw_command.get("command_id", "")))
        if command_id in command_ids:
            raise LabError(f"runner command map contains duplicate ID: {command_id}")
        command_ids.add(command_id)
        repository = validate_repository_name(
            str(raw_command.get("repository", ""))
        )
        argv = raw_command.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or len(argv) > _MAXIMUM_ARGUMENTS
            or not all(isinstance(argument, str) and argument for argument in argv)
            or sum(len(argument.encode()) for argument in argv)
            > _MAXIMUM_ARGUMENT_BYTES
        ):
            raise LabError(f"runner command has invalid argv: {command_id}")
        timeout_seconds = raw_command.get("timeout_seconds")
        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or not 1 <= timeout_seconds <= 24 * 60 * 60
        ):
            raise LabError(f"runner command has invalid timeout: {command_id}")
        commands.append(
            TrustedCommand(
                command_id=command_id,
                repository=repository,
                argv=tuple(argv),
                timeout_seconds=timeout_seconds,
            )
        )
    return RunnerCommandMap(
        runner_id=runner_id,
        commands=tuple(commands),
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _expanded_argv(
    argv: tuple[str, ...],
    *,
    repository_paths: Mapping[str, Path],
    source_root: Path,
    run_root: Path,
    request_id: str,
) -> tuple[str, ...]:
    replacements = {
        "{source_root}": str(source_root),
        "{run_root}": str(run_root),
        "{request_id}": request_id,
    }
    replacements.update(
        {f"{{repo:{name}}}": str(path) for name, path in repository_paths.items()}
    )
    expanded: list[str] = []
    for argument in argv:
        value = argument
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        if "{repo:" in value or any(token in value for token in replacements):
            raise LabError(f"runner command contains an unresolved placeholder: {argument!r}")
        expanded.append(value)
    return tuple(expanded)


def _terminate(process: subprocess.Popen[str]) -> int | None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()
    deadline = time.monotonic() + 5
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(_POLL_SECONDS)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return None


class TrustedCommandExecutor:
    """Invoke protected argv in an exact staged source tree without shell parsing."""

    def __init__(
        self,
        coordinator: RunnerCoordinator,
        stager: ExactSnapshotStager,
        *,
        command_root: Path,
    ) -> None:
        self.coordinator = coordinator
        self.stager = stager
        self.command_root = command_root.expanduser().resolve()

    def _receipt(
        self,
        request_id: str,
        *,
        request: Mapping[str, object],
        claim: Mapping[str, object],
        command_map: RunnerCommandMap,
        stage_receipt: Mapping[str, object],
        started_at: str,
        commands: list[dict[str, object]],
        classification: str,
        source_stable: bool,
        source_error: str | None,
    ) -> dict[str, object]:
        content: dict[str, object] = {
            "schema": COMMAND_EXECUTION_RECEIPT_SCHEMA_ID,
            "request_id": request_id,
            "idempotency_key": request["idempotency_key"],
            "claim_id": claim["claim_id"],
            "runner_id": claim["runner_id"],
            "runner_profile_sha256": claim["runner_profile_sha256"],
            "slot_id": claim["slot_id"],
            "command_map_sha256": command_map.sha256,
            "stage_receipt_sha256": stage_receipt["receipt_sha256"],
            "started_at": started_at,
            "finished_at": _utc_now(),
            "commands": commands,
            "classification": classification,
            "source_stable": source_stable,
            "source_error": source_error,
        }
        return {**content, "receipt_sha256": _sha256_json(content)}

    @staticmethod
    def _validate_existing_receipt(
        receipt: object,
        *,
        request: Mapping[str, object],
        claim: Mapping[str, object],
        command_map: RunnerCommandMap,
        stage_receipt: Mapping[str, object],
    ) -> dict[str, object]:
        expected = {
            "schema",
            "request_id",
            "idempotency_key",
            "claim_id",
            "runner_id",
            "runner_profile_sha256",
            "slot_id",
            "command_map_sha256",
            "stage_receipt_sha256",
            "started_at",
            "finished_at",
            "commands",
            "classification",
            "source_stable",
            "source_error",
            "receipt_sha256",
        }
        if not isinstance(receipt, dict) or set(receipt) != expected:
            raise LabError("command execution receipt has an unexpected shape")
        if (
            receipt.get("schema") != COMMAND_EXECUTION_RECEIPT_SCHEMA_ID
            or receipt.get("request_id") != request.get("request_id")
            or receipt.get("idempotency_key") != request.get("idempotency_key")
            or receipt.get("claim_id") != claim.get("claim_id")
            or receipt.get("runner_id") != claim.get("runner_id")
            or receipt.get("runner_profile_sha256")
            != claim.get("runner_profile_sha256")
            or receipt.get("slot_id") != claim.get("slot_id")
            or receipt.get("command_map_sha256") != command_map.sha256
            or receipt.get("stage_receipt_sha256")
            != stage_receipt.get("receipt_sha256")
            or receipt.get("classification")
            not in {
                "PASSED",
                "FAILED_ASSERTION",
                "INFRASTRUCTURE_FAILURE",
                "INVALID_SOURCE",
                "CANCELLED",
            }
            or not isinstance(receipt.get("commands"), list)
            or not isinstance(receipt.get("source_stable"), bool)
            or (
                receipt.get("source_error") is not None
                and not isinstance(receipt.get("source_error"), str)
            )
        ):
            raise LabError("command execution receipt identity mismatch")
        content = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        if receipt.get("receipt_sha256") != _sha256_json(content):
            raise LabError("command execution receipt hash mismatch")
        return receipt

    def _run_command(
        self,
        command: TrustedCommand,
        *,
        request_id: str,
        repository_paths: Mapping[str, Path],
        source_root: Path,
        run_root: Path,
        environment: Mapping[str, str],
        index: int,
    ) -> tuple[dict[str, object], bool]:
        cwd = repository_paths[command.repository]
        argv = _expanded_argv(
            command.argv,
            repository_paths=repository_paths,
            source_root=source_root,
            run_root=run_root,
            request_id=request_id,
        )
        log_path = run_root / f"{index:02d}-{command.command_id}.log"
        started = time.monotonic()
        timed_out = False
        cancelled = False
        start_error: str | None = None
        exit_code: int | None = None
        with _open_new_text(log_path) as log_stream:
            log_stream.write(
                f"command_id: {command.command_id}\n"
                f"repository: {command.repository}\n"
                f"argv: {json.dumps(argv)}\n\n"
            )
            log_stream.flush()
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=dict(environment),
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                deadline = time.monotonic() + command.timeout_seconds
                while process.poll() is None:
                    state = self.coordinator.store.load(request_id)
                    if state.get("state") == "CANCEL_REQUESTED":
                        cancelled = True
                        exit_code = _terminate(process)
                        log_stream.write("\nlab: cancellation requested\n")
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        exit_code = _terminate(process)
                        log_stream.write(
                            f"\nlab: command timed out after "
                            f"{command.timeout_seconds}s\n"
                        )
                        break
                    time.sleep(_POLL_SECONDS)
                if exit_code is None:
                    exit_code = process.poll()
            except OSError as exc:
                start_error = str(exc)
                log_stream.write(f"lab: command could not start: {exc}\n")
        command_record: dict[str, object] = {
            "command_id": command.command_id,
            "repository": command.repository,
            "argv": list(argv),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "start_error": start_error,
            "duration_seconds": round(time.monotonic() - started, 6),
            "log": log_path.relative_to(run_root.parent).as_posix(),
            "log_sha256": _sha256_file(log_path),
            "log_byte_size": log_path.stat().st_size,
        }
        return command_record, cancelled

    def execute(self, request_id: str, runner_id: str) -> dict[str, object]:
        validate_task_id(request_id)
        validate_task_id(runner_id)
        claim, state, _ = self.coordinator.validated_claim(request_id, runner_id)
        request = self.coordinator.store.load_request(request_id)
        stage_receipt, request_root = self.stager.validate_staged(
            request_id, runner_id
        )
        command_map = load_runner_command_map(self.command_root, runner_id)
        raw_profile = request.get("profile")
        raw_stage_repositories = stage_receipt.get("repositories")
        if not isinstance(raw_profile, dict) or not isinstance(
            raw_stage_repositories, dict
        ):
            raise LabError("staged request lost its command or repository identity")
        raw_command_ids = raw_profile.get("command_ids")
        if not isinstance(raw_command_ids, list) or not all(
            isinstance(command_id, str) for command_id in raw_command_ids
        ):
            raise LabError("Run request has invalid command IDs")
        available = command_map.by_id
        missing = [command_id for command_id in raw_command_ids if command_id not in available]
        if missing:
            raise LabError(
                "runner command map does not authorize: " + ", ".join(missing)
            )
        selected = [available[command_id] for command_id in raw_command_ids]
        repository_paths: dict[str, Path] = {}
        for name, raw_repository in raw_stage_repositories.items():
            if not isinstance(raw_repository, dict) or not isinstance(
                raw_repository.get("relative_path"), str
            ):
                raise LabError(f"invalid staged repository path: {name}")
            path = request_root / str(raw_repository["relative_path"])
            if not path.resolve().is_relative_to(request_root.resolve()):
                raise LabError(f"staged repository path escapes request root: {name}")
            repository_paths[str(name)] = path
        missing_repositories = sorted(
            {command.repository for command in selected} - set(repository_paths)
        )
        if missing_repositories:
            raise LabError(
                "runner command references unstaged repositories: "
                + ", ".join(missing_repositories)
            )

        run_root = request_root / "run"
        receipt_path = run_root / "execution-receipt.json"
        if self.coordinator.store.has_attachment(
            request_id, "command-execution-receipt"
        ):
            attached = self.coordinator.store.load_attachment(
                request_id, "command-execution-receipt"
            )
            receipt = self._validate_existing_receipt(
                attached,
                request=request,
                claim=claim,
                command_map=command_map,
                stage_receipt=stage_receipt,
            )
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise LabError("command execution disk receipt is missing or changed")
            try:
                disk_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LabError(
                    f"cannot read command execution disk receipt: {exc}"
                ) from exc
            if disk_receipt != receipt:
                raise LabError("command execution disk receipt is missing or changed")
            return receipt
        if run_root.exists():
            raise LabError(f"command execution slot is incomplete/quarantined: {run_root}")
        run_root.mkdir(parents=False)
        source_root = request_root / "source"
        context = {
            "schema": "lab.command-execution-context.v1",
            "request_id": request_id,
            "runner_id": runner_id,
            "command_map_sha256": command_map.sha256,
            "repositories": {
                name: str(path) for name, path in sorted(repository_paths.items())
            },
            "run_root": str(run_root),
        }
        context_path = run_root / "context.json"
        _write_new_text(context_path, json.dumps(context, indent=2, sort_keys=True) + "\n")
        environment = dict(os.environ)
        environment.update(
            {
                "LAB_REQUEST_ID": request_id,
                "LAB_RUN_ROOT": str(run_root),
                "LAB_SOURCE_ROOT": str(source_root),
                "LAB_EXECUTION_CONTEXT": str(context_path),
            }
        )
        started_at = _utc_now()
        records: list[dict[str, object]] = []
        cancelled = state.get("state") == "CANCEL_REQUESTED"
        if not cancelled:
            if state.get("state") != "PREPARING":
                raise LabError("command execution requires a PREPARING request")
            self.coordinator.begin_running(request_id, runner_id)
            for index, command in enumerate(selected, start=1):
                if self.coordinator.store.load(request_id).get("state") == "CANCEL_REQUESTED":
                    cancelled = True
                    break
                record, command_cancelled = self._run_command(
                    command,
                    request_id=request_id,
                    repository_paths=repository_paths,
                    source_root=source_root,
                    run_root=run_root,
                    environment=environment,
                    index=index,
                )
                records.append(record)
                if command_cancelled:
                    cancelled = True
                    break
                if record["start_error"] is not None or record["exit_code"] != 0:
                    break
        if self.coordinator.store.load(request_id).get("state") == "CANCEL_REQUESTED":
            cancelled = True

        source_stable = True
        source_error: str | None = None
        try:
            self.stager.validate_staged(request_id, runner_id)
        except LabError as exc:
            source_stable = False
            source_error = str(exc)
        if cancelled:
            classification = "CANCELLED"
        elif not source_stable:
            classification = "INVALID_SOURCE"
        elif len(records) == len(selected) and all(
            record["exit_code"] == 0
            and record["start_error"] is None
            and record["timed_out"] is False
            for record in records
        ):
            classification = "PASSED"
        elif any(
            record["timed_out"] is True or record["start_error"] is not None
            for record in records
        ):
            classification = "INFRASTRUCTURE_FAILURE"
        else:
            classification = "FAILED_ASSERTION"
        receipt = self._receipt(
            request_id,
            request=request,
            claim=claim,
            command_map=command_map,
            stage_receipt=stage_receipt,
            started_at=started_at,
            commands=records,
            classification=classification,
            source_stable=source_stable,
            source_error=source_error,
        )
        self._validate_existing_receipt(
            receipt,
            request=request,
            claim=claim,
            command_map=command_map,
            stage_receipt=stage_receipt,
        )
        _atomic_write_text(
            receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        self.coordinator.store.put_attachment(
            request_id, "command-execution-receipt", receipt
        )
        if classification == "CANCELLED":
            self.coordinator.acknowledge_cancel(
                request_id, runner_id, detail="trusted command execution cancelled"
            )
            return receipt

        self.coordinator.begin_collecting(request_id, runner_id)
        artifacts = [
            {
                "artifact_id": f"command-{index:02d}-{record['command_id']}-log",
                "relative_path": record["log"],
                "media_type": "text/plain",
                "byte_size": record["log_byte_size"],
                "sha256": record["log_sha256"],
                "storage_id": "runner-slot",
            }
            for index, record in enumerate(records, start=1)
        ]
        self.coordinator.record_artifact_receipt(
            request_id,
            runner_id,
            artifacts=artifacts,
            evidence_complete=True,
        )
        if classification == "PASSED":
            self.coordinator.complete(
                request_id, runner_id, detail="trusted system command passed"
            )
        else:
            self.coordinator.store.fail(
                request_id,
                actor=runner_id,
                classification=classification,
                detail="trusted system command did not pass",
            )
        return receipt
