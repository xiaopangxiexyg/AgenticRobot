from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .execution_sandbox import validate_sandbox_for_attempt
from .remote_request import (
    SHA256_PATTERN,
    RemoteProfile,
    _sha256,
    validate_remote_request,
)
from .service_identity import (
    validate_service_launch_spec_identity,
    validate_service_receipt,
)
from .workspace import LabError, validate_task_id

ATTEMPT_START_SCHEMA_ID = "lab.attempt-start.v2"
ATTEMPT_RESULT_SCHEMA_ID = "lab.attempt-result.v1"
MAX_ATTEMPT_ARTIFACT_BYTES = 16 * 1024 * 1024


def _collect_regular_file(
    root_descriptor: int,
    relative: Path,
    *,
    max_bytes: int,
) -> tuple[str, int]:
    directory_descriptor = os.dup(root_descriptor)
    try:
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        for part in relative.parts[:-1]:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        descriptor = os.open(
            relative.parts[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        raise LabError(
            f"attempt artifact is missing, escaped, or uses a symlink: {relative}"
        ) from exc
    finally:
        os.close(directory_descriptor)
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise LabError(f"attempt artifact is unsafe or too large: {relative}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(block)
                if total > max_bytes:
                    raise LabError(f"attempt artifact is too large: {relative}")
                digest.update(block)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ino != before.st_ino
            or after.st_dev != before.st_dev
        ):
            raise LabError(f"attempt artifact changed while collected: {relative}")
        return digest.hexdigest(), total
    finally:
        os.close(descriptor)


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise LabError(f"{label} has an invalid timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError(f"{label} has an invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise LabError(f"{label} timestamp requires a timezone")
    return parsed


def _validate_service_binding(
    launch_spec: object,
    service_receipt: object,
) -> tuple[dict[str, object], dict[str, object]]:
    spec = validate_service_launch_spec_identity(launch_spec)
    receipt = validate_service_receipt(service_receipt)
    bindings = {
        "service_id": spec.get("service_id"),
        "request_id": spec.get("request_id"),
        "task_id": spec.get("task_id"),
        "snapshot_id": spec.get("snapshot_id"),
        "slot_id": spec.get("slot_id"),
        "launch_spec_sha256": spec.get("launch_spec_sha256"),
        "remote_profile_sha256": spec.get("remote_profile_sha256"),
        "readiness_receipt_sha256": spec.get("readiness_receipt_sha256"),
        "service_name": spec.get("service_name"),
        "protocol": spec.get("protocol"),
        "launch_command_id": spec.get("launch_command_id"),
        "source": spec.get("source"),
        "environment": spec.get("environment"),
        "asset_lock": spec.get("asset_lock"),
        "assets": spec.get("assets"),
        "gpu": spec.get("gpu"),
    }
    for field, expected in bindings.items():
        if receipt.get(field) != expected:
            raise LabError(f"service receipt does not match launch spec: {field}")
    return spec, receipt


def build_attempt_start(
    request: object,
    *,
    remote_profile: RemoteProfile,
    control_plane_version: str,
    launch_spec: object,
    service_receipt: object,
    execution_sandbox_receipt: object,
    command_id: str,
    command_adapter_id: str,
    command_registry_sha256: str,
    executor_nonce: str,
    attempt_number: int = 1,
) -> dict[str, object]:
    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    spec, service = _validate_service_binding(launch_spec, service_receipt)
    if command_id not in remote_profile.command_ids:
        raise LabError(f"command is not authorized by remote Profile: {command_id}")
    validate_task_id(command_id)
    validate_task_id(command_adapter_id)
    if not SHA256_PATTERN.fullmatch(command_registry_sha256):
        raise LabError("attempt command registry requires a full SHA-256")
    validate_task_id(executor_nonce)
    if (
        not isinstance(attempt_number, int)
        or isinstance(attempt_number, bool)
        or attempt_number < 1
    ):
        raise LabError("attempt number must be a positive integer")
    if (
        spec.get("request_id") != validated_request.get("request_id")
        or spec.get("task_id") != validated_request.get("task_id")
        or spec.get("snapshot_id") != validated_request.get("snapshot_id")
        or spec.get("remote_profile_sha256") != remote_profile.sha256
    ):
        raise LabError("attempt service identity does not match remote request")
    sandbox = validate_sandbox_for_attempt(
        execution_sandbox_receipt,
        request_id=str(validated_request["request_id"]),
        idempotency_key=str(validated_request["idempotency_key"]),
        slot_id=str(spec["slot_id"]),
        remote_profile_sha256=remote_profile.sha256,
        control_plane_version=control_plane_version,
        launch_spec_sha256=str(spec["launch_spec_sha256"]),
        service_receipt_sha256=str(service["receipt_sha256"]),
        gpu=service["gpu"],
    )
    content: dict[str, object] = {
        "schema": ATTEMPT_START_SCHEMA_ID,
        "request_id": validated_request["request_id"],
        "idempotency_key": validated_request["idempotency_key"],
        "task_id": validated_request["task_id"],
        "snapshot_id": validated_request["snapshot_id"],
        "slot_id": spec["slot_id"],
        "attempt_number": attempt_number,
        "remote_profile_sha256": remote_profile.sha256,
        "control_plane_version": control_plane_version,
        "command_id": command_id,
        "command_adapter_id": command_adapter_id,
        "command_registry_sha256": command_registry_sha256,
        "service_receipt_sha256": service["receipt_sha256"],
        "execution_sandbox_receipt_sha256": sandbox["receipt_sha256"],
        "asset_lock": validated_request["asset_lock"],
        "gpu": service["gpu"],
        "executor_nonce": executor_nonce,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "execution_authorized": True,
        "device_access": False,
        "network_egress": remote_profile.network_egress,
        "downloads_permitted": remote_profile.downloads_permitted,
    }
    identity_hash = _sha256(content)
    identified = {**content, "attempt_id": f"ATT-{identity_hash[:24]}"}
    receipt = {**identified, "receipt_sha256": _sha256(identified)}
    return validate_attempt_start(receipt)


def validate_attempt_start(receipt: object) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "attempt_id",
        "request_id",
        "idempotency_key",
        "task_id",
        "snapshot_id",
        "slot_id",
        "attempt_number",
        "remote_profile_sha256",
        "control_plane_version",
        "command_id",
        "command_adapter_id",
        "command_registry_sha256",
        "service_receipt_sha256",
        "execution_sandbox_receipt_sha256",
        "asset_lock",
        "gpu",
        "executor_nonce",
        "started_at",
        "execution_authorized",
        "device_access",
        "network_egress",
        "downloads_permitted",
        "receipt_sha256",
    }:
        raise LabError("attempt start receipt has an unexpected shape")
    if receipt.get("schema") != ATTEMPT_START_SCHEMA_ID:
        raise LabError("unsupported attempt start receipt schema")
    for field in (
        "attempt_id",
        "request_id",
        "task_id",
        "snapshot_id",
        "slot_id",
        "control_plane_version",
        "command_id",
        "command_adapter_id",
        "executor_nonce",
    ):
        if not isinstance(receipt.get(field), str):
            raise LabError(f"attempt start receipt has invalid {field}")
        validate_task_id(str(receipt[field]))
    for field in (
        "idempotency_key",
        "remote_profile_sha256",
        "command_registry_sha256",
        "service_receipt_sha256",
        "execution_sandbox_receipt_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"attempt start receipt has invalid {field}")
    attempt_number = receipt.get("attempt_number")
    if (
        not isinstance(attempt_number, int)
        or isinstance(attempt_number, bool)
        or attempt_number < 1
    ):
        raise LabError("attempt start receipt has invalid attempt_number")
    gpu = receipt.get("gpu")
    asset_lock = receipt.get("asset_lock")
    if (
        not isinstance(asset_lock, dict)
        or set(asset_lock) != {"lock_id", "workload_id", "sha256"}
        or not isinstance(asset_lock.get("lock_id"), str)
        or (
            asset_lock.get("workload_id") is not None
            and not isinstance(asset_lock.get("workload_id"), str)
        )
        or not SHA256_PATTERN.fullmatch(str(asset_lock.get("sha256", "")))
    ):
        raise LabError("attempt start receipt has invalid asset lock identity")
    validate_task_id(str(asset_lock["lock_id"]))
    if asset_lock["workload_id"] is not None:
        validate_task_id(str(asset_lock["workload_id"]))
    if not isinstance(gpu, dict) or set(gpu) != {
        "lease_id",
        "lease_receipt_sha256",
        "gpu_uuid",
    }:
        raise LabError("attempt start receipt has invalid GPU identity")
    if not all(isinstance(gpu.get(field), str) and gpu[field] for field in gpu):
        raise LabError("attempt start receipt has invalid GPU identity")
    if not SHA256_PATTERN.fullmatch(str(gpu.get("lease_receipt_sha256", ""))):
        raise LabError("attempt start receipt has invalid GPU lease receipt hash")
    if (
        receipt.get("execution_authorized") is not True
        or receipt.get("device_access") is not False
        or receipt.get("network_egress") != "loopback-only"
        or receipt.get("downloads_permitted") is not False
    ):
        raise LabError("attempt start receipt has an unsafe authority envelope")
    _timestamp(receipt.get("started_at"), label="attempt start receipt")
    identity_content = {
        key: value
        for key, value in receipt.items()
        if key not in {"attempt_id", "receipt_sha256"}
    }
    identity_hash = _sha256(identity_content)
    if receipt.get("attempt_id") != f"ATT-{identity_hash[:24]}":
        raise LabError("attempt start receipt ID mismatch")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("attempt start receipt hash mismatch")
    return receipt


def collect_attempt_artifacts(
    attempt_root: Path,
    relative_paths: tuple[str, ...],
    *,
    max_total_bytes: int = MAX_ATTEMPT_ARTIFACT_BYTES,
) -> dict[str, dict[str, object]]:
    root = attempt_root.resolve()
    if attempt_root.is_symlink() or not root.is_dir():
        raise LabError(f"attempt artifact root is missing or unsafe: {attempt_root}")
    if not 1 <= max_total_bytes <= MAX_ATTEMPT_ARTIFACT_BYTES:
        raise LabError("attempt artifact byte limit is outside the protected maximum")
    if len(relative_paths) != len(set(relative_paths)):
        raise LabError("attempt artifact paths contain duplicates")
    root_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        root_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        root_flags |= os.O_NOFOLLOW
    try:
        root_descriptor = os.open(root, root_flags)
    except OSError as exc:
        raise LabError(f"cannot safely open artifact root: {attempt_root}") from exc
    total = 0
    artifacts: dict[str, dict[str, object]] = {}
    try:
        for raw_path in sorted(relative_paths):
            relative = Path(raw_path)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or "\\" in raw_path
                or relative.as_posix() != raw_path
            ):
                raise LabError(f"attempt artifact path is unsafe: {raw_path}")
            digest, size = _collect_regular_file(
                root_descriptor,
                relative,
                max_bytes=max_total_bytes - total,
            )
            total += size
            artifacts[raw_path] = {"sha256": digest, "size_bytes": size}
        return artifacts
    finally:
        os.close(root_descriptor)


def build_attempt_result(
    start_receipt: object,
    *,
    exit_code: int,
    assertions_passed: bool,
    task_success: bool,
    service_acknowledged: bool,
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    start = validate_attempt_start(start_receipt)
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise LabError("attempt result exit_code must be an integer")
    if not all(
        isinstance(value, bool)
        for value in (assertions_passed, task_success, service_acknowledged)
    ):
        raise LabError("attempt result outcomes must be booleans")
    if task_success and not assertions_passed:
        raise LabError("attempt cannot succeed when assertions failed")
    if assertions_passed and exit_code != 0:
        raise LabError("passing assertions require exit code zero")
    if task_success and not service_acknowledged:
        raise LabError("attempt success requires service acknowledgement")
    normalized_artifacts: dict[str, dict[str, object]] = {}
    total_artifact_bytes = 0
    if not all(isinstance(path, str) for path in artifacts):
        raise LabError("attempt result artifact paths must be strings")
    for relative_path, identity in sorted(artifacts.items()):
        assert isinstance(relative_path, str)
        path = Path(relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or "\\" in relative_path
            or path.as_posix() != relative_path
        ):
            raise LabError(f"attempt result artifact path is unsafe: {relative_path}")
        size_bytes = (
            identity.get("size_bytes") if isinstance(identity, Mapping) else None
        )
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"sha256", "size_bytes"}
            or not SHA256_PATTERN.fullmatch(str(identity.get("sha256", "")))
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise LabError(
                f"attempt result artifact identity is invalid: {relative_path}"
            )
        total_artifact_bytes += size_bytes
        if total_artifact_bytes > MAX_ATTEMPT_ARTIFACT_BYTES:
            raise LabError("attempt result artifacts exceed the protected byte limit")
        normalized_artifacts[relative_path] = dict(identity)
    if not service_acknowledged:
        classification = "INFRASTRUCTURE_FAILURE"
    elif exit_code == 0 and assertions_passed and task_success:
        classification = "PASSED"
    else:
        classification = "FAILED_ASSERTION"
    content: dict[str, object] = {
        "schema": ATTEMPT_RESULT_SCHEMA_ID,
        "attempt_id": start["attempt_id"],
        "attempt_number": start["attempt_number"],
        "request_id": start["request_id"],
        "idempotency_key": start["idempotency_key"],
        "slot_id": start["slot_id"],
        "command_id": start["command_id"],
        "start_receipt_sha256": start["receipt_sha256"],
        "classification": classification,
        "exit_code": exit_code,
        "execution_completed": True,
        "service_acknowledged": service_acknowledged,
        "assertions_passed": assertions_passed,
        "task_success": task_success,
        "evidence_complete": True,
        "artifacts": normalized_artifacts,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "device_access": False,
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_attempt_result(receipt, start_receipt=start)


def validate_attempt_result(
    receipt: object,
    *,
    start_receipt: object,
) -> dict[str, object]:
    start = validate_attempt_start(start_receipt)
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "attempt_id",
        "attempt_number",
        "request_id",
        "idempotency_key",
        "slot_id",
        "command_id",
        "start_receipt_sha256",
        "classification",
        "exit_code",
        "execution_completed",
        "service_acknowledged",
        "assertions_passed",
        "task_success",
        "evidence_complete",
        "artifacts",
        "finished_at",
        "device_access",
        "receipt_sha256",
    }:
        raise LabError("attempt result receipt has an unexpected shape")
    if receipt.get("schema") != ATTEMPT_RESULT_SCHEMA_ID:
        raise LabError("unsupported attempt result receipt schema")
    bindings = {
        "attempt_id": start.get("attempt_id"),
        "attempt_number": start.get("attempt_number"),
        "request_id": start.get("request_id"),
        "idempotency_key": start.get("idempotency_key"),
        "slot_id": start.get("slot_id"),
        "command_id": start.get("command_id"),
        "start_receipt_sha256": start.get("receipt_sha256"),
    }
    for field, expected in bindings.items():
        if receipt.get(field) != expected:
            raise LabError(f"attempt result receipt binding mismatch: {field}")
    if receipt.get("classification") not in {
        "PASSED",
        "FAILED_ASSERTION",
        "INFRASTRUCTURE_FAILURE",
    }:
        raise LabError("attempt result receipt has invalid classification")
    exit_code = receipt.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise LabError("attempt result receipt has invalid exit_code")
    for field in (
        "execution_completed",
        "service_acknowledged",
        "assertions_passed",
        "task_success",
        "evidence_complete",
        "device_access",
    ):
        if not isinstance(receipt.get(field), bool):
            raise LabError(f"attempt result receipt has invalid {field}")
    if (
        receipt.get("execution_completed") is not True
        or receipt.get("evidence_complete") is not True
        or receipt.get("device_access") is not False
    ):
        raise LabError("attempt result receipt has an unsafe evidence envelope")
    passed = receipt.get("classification") == "PASSED"
    if passed != (
        exit_code == 0
        and receipt.get("service_acknowledged") is True
        and receipt.get("assertions_passed") is True
        and receipt.get("task_success") is True
    ):
        raise LabError("attempt result receipt outcome is inconsistent")
    if (
        receipt.get("classification") == "INFRASTRUCTURE_FAILURE"
        and receipt.get("service_acknowledged") is not False
    ):
        raise LabError(
            "infrastructure failure requires missing service acknowledgement"
        )
    if (
        receipt.get("classification") == "FAILED_ASSERTION"
        and receipt.get("service_acknowledged") is not True
    ):
        raise LabError("failed assertion requires a service acknowledgement")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise LabError("attempt result receipt has invalid artifacts")
    total_artifact_bytes = 0
    for relative_path, identity in artifacts.items():
        if not isinstance(relative_path, str):
            raise LabError("attempt result receipt artifact path must be a string")
        path = Path(relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or "\\" in relative_path
            or path.as_posix() != relative_path
            or not isinstance(identity, dict)
            or set(identity) != {"sha256", "size_bytes"}
            or not SHA256_PATTERN.fullmatch(str(identity.get("sha256", "")))
            or not isinstance(identity.get("size_bytes"), int)
            or isinstance(identity.get("size_bytes"), bool)
            or identity["size_bytes"] < 0
        ):
            raise LabError(
                f"attempt result receipt has invalid artifact: {relative_path}"
            )
        total_artifact_bytes += identity["size_bytes"]
        if total_artifact_bytes > MAX_ATTEMPT_ARTIFACT_BYTES:
            raise LabError(
                "attempt result receipt artifacts exceed the protected byte limit"
            )
    if not SHA256_PATTERN.fullmatch(str(receipt.get("receipt_sha256", ""))):
        raise LabError("attempt result receipt has invalid hash")
    finished = _timestamp(receipt.get("finished_at"), label="attempt result receipt")
    started = _timestamp(start.get("started_at"), label="attempt start receipt")
    if finished < started:
        raise LabError("attempt result predates its start")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("attempt result receipt hash mismatch")
    return receipt


@dataclass(frozen=True)
class TrustedCommandOutcome:
    exit_code: int
    assertions_passed: bool
    task_success: bool
    service_acknowledged: bool
    artifact_paths: tuple[str, ...]


TrustedCommandAdapter = Callable[[dict[str, object], Path], TrustedCommandOutcome]
