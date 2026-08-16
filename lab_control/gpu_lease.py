from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .remote_request import SHA256_PATTERN
from .workspace import LabError, _atomic_write_text, validate_task_id

GPU_LEASE_SCHEMA_ID = "lab.gpu-lease.v1"
GPU_CLEANUP_SCHEMA_ID = "lab.gpu-cleanup.v2"
GPU_RELEASE_SCHEMA_ID = "lab.gpu-release.v1"
GPU_STATE_SCHEMA_ID = "lab.gpu-lease-state.v1"
GPU_INVENTORY_SCHEMA_ID = "lab.gpu-inventory.v1"
GPU_PROCESS_INVENTORY_SCHEMA_ID = "lab.gpu-process-inventory.v1"
_GPU_UUID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,255}")
MAX_NVIDIA_SMI_OUTPUT_BYTES = 64 * 1024


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise LabError(f"GPU receipt has invalid {label}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError(f"GPU receipt has invalid {label}") from exc
    if parsed.tzinfo is None:
        raise LabError(f"GPU receipt {label} requires a timezone")
    return parsed


def _validate_gpu_uuid(gpu_uuid: str) -> str:
    if not _GPU_UUID_PATTERN.fullmatch(gpu_uuid):
        raise LabError(f"invalid GPU UUID: {gpu_uuid!r}")
    return gpu_uuid


def _receipt(content: dict[str, object]) -> dict[str, object]:
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def collect_nvidia_gpu_inventory(
    *,
    nvidia_smi: Path = Path("/usr/bin/nvidia-smi"),
) -> dict[str, object]:
    """Collect a bounded, read-only NVIDIA identity inventory."""

    requested = nvidia_smi.expanduser()
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or not requested.is_file()
        or not os.access(requested, os.X_OK)
    ):
        raise LabError(f"protected nvidia-smi is missing or unsafe: {requested}")
    executable = requested.resolve()
    try:
        completed = subprocess.run(
            [
                str(executable),
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LabError(f"GPU inventory probe failed: {exc}") from exc
    if (
        len(completed.stdout.encode()) > MAX_NVIDIA_SMI_OUTPUT_BYTES
        or len(completed.stderr.encode()) > MAX_NVIDIA_SMI_OUTPUT_BYTES
    ):
        raise LabError("GPU inventory probe output exceeds the protected byte limit")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LabError(f"GPU inventory probe failed: {detail}")
    gpus: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    seen_uuids: set[str] = set()
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2]:
            raise LabError("GPU inventory probe returned malformed output")
        index = int(fields[0])
        gpu_uuid = _validate_gpu_uuid(fields[1])
        if index in seen_indices or gpu_uuid in seen_uuids:
            raise LabError("GPU inventory probe returned duplicate identities")
        seen_indices.add(index)
        seen_uuids.add(gpu_uuid)
        gpus.append(
            {
                "gpu_id": f"gpu-{index}",
                "index": index,
                "gpu_uuid": gpu_uuid,
                "name": fields[2],
            }
        )
    if not gpus:
        raise LabError("GPU inventory probe found no GPUs")
    content: dict[str, object] = {
        "schema": GPU_INVENTORY_SCHEMA_ID,
        "executable": str(executable),
        "gpus": sorted(gpus, key=lambda gpu: str(gpu["gpu_id"])),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "probe_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "read_only": True,
        "device_control": False,
        "network_used": False,
    }
    return _receipt(content)


def validate_gpu_inventory(receipt: object) -> dict[str, object]:
    expected = {
        "schema",
        "executable",
        "gpus",
        "observed_at",
        "probe_stdout_sha256",
        "read_only",
        "device_control",
        "network_used",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("GPU inventory receipt has an unexpected shape")
    executable = receipt.get("executable")
    gpus = receipt.get("gpus")
    if (
        receipt.get("schema") != GPU_INVENTORY_SCHEMA_ID
        or not isinstance(executable, str)
        or not Path(executable).is_absolute()
        or not isinstance(gpus, list)
        or not gpus
        or receipt.get("read_only") is not True
        or receipt.get("device_control") is not False
        or receipt.get("network_used") is not False
        or not SHA256_PATTERN.fullmatch(
            str(receipt.get("probe_stdout_sha256", ""))
        )
    ):
        raise LabError("GPU inventory receipt identity or authority is invalid")
    indices: set[int] = set()
    uuids: set[str] = set()
    for gpu in gpus:
        if not isinstance(gpu, dict) or set(gpu) != {
            "gpu_id",
            "index",
            "gpu_uuid",
            "name",
        }:
            raise LabError("GPU inventory contains an invalid entry")
        index = gpu.get("index")
        gpu_id = gpu.get("gpu_id")
        gpu_uuid = gpu.get("gpu_uuid")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or gpu_id != f"gpu-{index}"
            or not isinstance(gpu_uuid, str)
            or not isinstance(gpu.get("name"), str)
            or not gpu["name"]
        ):
            raise LabError("GPU inventory contains an invalid identity")
        _validate_gpu_uuid(gpu_uuid)
        if index in indices or gpu_uuid in uuids:
            raise LabError("GPU inventory contains duplicate identities")
        indices.add(index)
        uuids.add(gpu_uuid)
    _timestamp(receipt.get("observed_at"), label="observed_at")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != hashlib.sha256(
        _canonical_json(content)
    ).hexdigest():
        raise LabError("GPU inventory receipt hash mismatch")
    return receipt


def collect_nvidia_compute_process_inventory(
    *,
    nvidia_smi: Path = Path("/usr/bin/nvidia-smi"),
) -> dict[str, object]:
    """Collect bounded compute-process identities without signalling processes."""

    requested = nvidia_smi.expanduser()
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or not requested.is_file()
        or not os.access(requested, os.X_OK)
    ):
        raise LabError(f"protected nvidia-smi is missing or unsafe: {requested}")
    executable = requested.resolve()
    try:
        completed = subprocess.run(
            [
                str(executable),
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LabError(f"GPU process inventory probe failed: {exc}") from exc
    if (
        len(completed.stdout.encode()) > MAX_NVIDIA_SMI_OUTPUT_BYTES
        or len(completed.stderr.encode()) > MAX_NVIDIA_SMI_OUTPUT_BYTES
    ):
        raise LabError("GPU process inventory exceeds the protected byte limit")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LabError(f"GPU process inventory probe failed: {detail}")
    processes: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if (
            len(fields) != 4
            or not fields[1].isdigit()
            or not fields[2]
            or len(fields[2]) > 4096
            or "\x00" in fields[2]
            or not fields[3]
        ):
            raise LabError("GPU process inventory probe returned malformed output")
        gpu_uuid = _validate_gpu_uuid(fields[0])
        pid = int(fields[1])
        if pid <= 0 or (gpu_uuid, pid) in seen:
            raise LabError("GPU process inventory contains an invalid identity")
        seen.add((gpu_uuid, pid))
        processes.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": pid,
                "process_name": fields[2],
                "used_gpu_memory_mib": fields[3],
            }
        )
    content: dict[str, object] = {
        "schema": GPU_PROCESS_INVENTORY_SCHEMA_ID,
        "executable": str(executable),
        "processes": sorted(
            processes,
            key=lambda process: (str(process["gpu_uuid"]), str(process["pid"])),
        ),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "probe_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "read_only": True,
        "process_control": False,
        "network_used": False,
    }
    return _receipt(content)


def validate_gpu_process_inventory(receipt: object) -> dict[str, object]:
    expected = {
        "schema",
        "executable",
        "processes",
        "observed_at",
        "probe_stdout_sha256",
        "read_only",
        "process_control",
        "network_used",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("GPU process inventory receipt has an unexpected shape")
    executable = receipt.get("executable")
    processes = receipt.get("processes")
    if (
        receipt.get("schema") != GPU_PROCESS_INVENTORY_SCHEMA_ID
        or not isinstance(executable, str)
        or not Path(executable).is_absolute()
        or not isinstance(processes, list)
        or receipt.get("read_only") is not True
        or receipt.get("process_control") is not False
        or receipt.get("network_used") is not False
        or not SHA256_PATTERN.fullmatch(
            str(receipt.get("probe_stdout_sha256", ""))
        )
    ):
        raise LabError("GPU process inventory identity or authority is invalid")
    seen: set[tuple[str, int]] = set()
    for process in processes:
        if not isinstance(process, dict) or set(process) != {
            "gpu_uuid",
            "pid",
            "process_name",
            "used_gpu_memory_mib",
        }:
            raise LabError("GPU process inventory contains an invalid entry")
        gpu_uuid = process.get("gpu_uuid")
        pid = process.get("pid")
        process_name = process.get("process_name")
        memory = process.get("used_gpu_memory_mib")
        if (
            not isinstance(gpu_uuid, str)
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(process_name, str)
            or not process_name
            or len(process_name) > 4096
            or "\x00" in process_name
            or not isinstance(memory, str)
            or not memory
        ):
            raise LabError("GPU process inventory contains an invalid identity")
        _validate_gpu_uuid(gpu_uuid)
        if (gpu_uuid, pid) in seen:
            raise LabError("GPU process inventory contains duplicate identities")
        seen.add((gpu_uuid, pid))
    _timestamp(receipt.get("observed_at"), label="observed_at")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != hashlib.sha256(
        _canonical_json(content)
    ).hexdigest():
        raise LabError("GPU process inventory receipt hash mismatch")
    return receipt


def require_gpu_compute_processes_absent(
    process_inventory_receipt: object,
    *,
    gpu_uuid: str,
    max_age_seconds: int | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    inventory = validate_gpu_process_inventory(process_inventory_receipt)
    if max_age_seconds is not None:
        if not 1 <= max_age_seconds <= 300:
            raise LabError("GPU process inventory max age must be between 1 and 300")
        observed = _timestamp(inventory.get("observed_at"), label="observed_at")
        current = now or datetime.now(timezone.utc)
        age = (current - observed).total_seconds()
        if age < -5 or age > max_age_seconds:
            raise LabError("GPU process inventory is stale")
    processes = inventory["processes"]
    assert isinstance(processes, list)
    matching = [
        process
        for process in processes
        if isinstance(process, dict) and process.get("gpu_uuid") == gpu_uuid
    ]
    if matching:
        pids = sorted(int(process["pid"]) for process in matching)
        raise LabError(f"GPU has existing compute processes: {pids}")
    return inventory


def select_single_gpu(inventory_receipt: object) -> tuple[str, str]:
    inventory = validate_gpu_inventory(inventory_receipt)
    gpus = inventory["gpus"]
    assert isinstance(gpus, list)
    if len(gpus) != 1:
        raise LabError("M3 GPU-only Profile requires exactly one observed GPU")
    gpu = gpus[0]
    assert isinstance(gpu, dict)
    return str(gpu["gpu_id"]), str(gpu["gpu_uuid"])


def validate_gpu_lease_receipt(
    receipt: object,
    *,
    host_id: str,
    gpu_uuid: str,
    require_active: bool,
    host_profile_sha256: str | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    expected = {
        "schema",
        "lease_id",
        "host_id",
        "host_profile_sha256",
        "gpu_id",
        "gpu_uuid",
        "request_id",
        "slot_id",
        "status",
        "acquired_at",
        "expires_at",
        "execution_eligible",
        "device_access",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("GPU lease receipt has an unexpected shape")
    validate_task_id(host_id)
    _validate_gpu_uuid(gpu_uuid)
    for field in ("lease_id", "gpu_id", "request_id", "slot_id"):
        value = receipt.get(field)
        if not isinstance(value, str):
            raise LabError(f"GPU lease receipt has invalid {field}")
        validate_task_id(value)
    if (
        receipt.get("schema") != GPU_LEASE_SCHEMA_ID
        or receipt.get("host_id") != host_id
        or receipt.get("gpu_uuid") != gpu_uuid
        or (
            host_profile_sha256 is not None
            and receipt.get("host_profile_sha256") != host_profile_sha256
        )
        or receipt.get("status") != "ACTIVE"
        or receipt.get("execution_eligible") is not False
        or receipt.get("device_access") is not False
        or not SHA256_PATTERN.fullmatch(
            str(receipt.get("host_profile_sha256", ""))
        )
    ):
        raise LabError("GPU lease receipt identity or authority is invalid")
    acquired = _timestamp(receipt.get("acquired_at"), label="acquired_at")
    expires = _timestamp(receipt.get("expires_at"), label="expires_at")
    if expires <= acquired:
        raise LabError("GPU lease receipt timestamps are invalid")
    if require_active and (now or datetime.now(timezone.utc)) > expires:
        raise LabError("GPU lease expired and requires reconciliation")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    digest = receipt.get("receipt_sha256")
    if (
        not SHA256_PATTERN.fullmatch(str(digest))
        or digest != hashlib.sha256(_canonical_json(content)).hexdigest()
    ):
        raise LabError("GPU lease receipt hash mismatch")
    return receipt


def build_gpu_cleanup_receipt(
    lease_receipt: object,
    *,
    host_id: str,
    gpu_uuid: str,
    service_started: bool,
    service_stopped: bool,
    process_inventory_receipt: object,
    host_profile_sha256: str | None = None,
) -> dict[str, object]:
    lease = validate_gpu_lease_receipt(
        lease_receipt,
        host_id=host_id,
        gpu_uuid=gpu_uuid,
        require_active=False,
        host_profile_sha256=host_profile_sha256,
    )
    if not all(isinstance(value, bool) for value in (service_started, service_stopped)):
        raise LabError("GPU cleanup facts must be booleans")
    process_inventory = require_gpu_compute_processes_absent(
        process_inventory_receipt,
        gpu_uuid=gpu_uuid,
        max_age_seconds=30,
    )
    safe = not service_started or service_stopped
    content: dict[str, object] = {
        "schema": GPU_CLEANUP_SCHEMA_ID,
        "lease_id": lease["lease_id"],
        "lease_receipt_sha256": lease["receipt_sha256"],
        "host_id": host_id,
        "gpu_uuid": gpu_uuid,
        "service_started": service_started,
        "service_stopped": service_stopped,
        "gpu_processes_absent": True,
        "gpu_process_inventory_receipt": process_inventory,
        "safe_to_release": safe,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    return _receipt(content)


def validate_gpu_cleanup_receipt(
    receipt: object,
    *,
    lease_receipt: object,
    host_id: str,
    gpu_uuid: str,
    host_profile_sha256: str | None = None,
) -> dict[str, object]:
    lease = validate_gpu_lease_receipt(
        lease_receipt,
        host_id=host_id,
        gpu_uuid=gpu_uuid,
        require_active=False,
        host_profile_sha256=host_profile_sha256,
    )
    expected = {
        "schema",
        "lease_id",
        "lease_receipt_sha256",
        "host_id",
        "gpu_uuid",
        "service_started",
        "service_stopped",
        "gpu_processes_absent",
        "gpu_process_inventory_receipt",
        "safe_to_release",
        "checked_at",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("GPU cleanup receipt has an unexpected shape")
    for field in ("service_started", "service_stopped", "gpu_processes_absent"):
        if not isinstance(receipt.get(field), bool):
            raise LabError(f"GPU cleanup receipt has invalid {field}")
    process_inventory = require_gpu_compute_processes_absent(
        receipt.get("gpu_process_inventory_receipt"),
        gpu_uuid=gpu_uuid,
    )
    safe = (
        receipt.get("service_started") is False
        or receipt.get("service_stopped") is True
    )
    if (
        receipt.get("schema") != GPU_CLEANUP_SCHEMA_ID
        or receipt.get("lease_id") != lease["lease_id"]
        or receipt.get("lease_receipt_sha256") != lease["receipt_sha256"]
        or receipt.get("host_id") != host_id
        or receipt.get("gpu_uuid") != gpu_uuid
        or receipt.get("safe_to_release") is not safe
        or receipt.get("gpu_processes_absent") is not True
        or not safe
    ):
        raise LabError("GPU cleanup receipt does not prove safe release")
    checked_at = _timestamp(receipt.get("checked_at"), label="checked_at")
    observed_at = _timestamp(
        process_inventory.get("observed_at"),
        label="observed_at",
    )
    if observed_at > checked_at:
        raise LabError("GPU cleanup process evidence is newer than its receipt")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != hashlib.sha256(
        _canonical_json(content)
    ).hexdigest():
        raise LabError("GPU cleanup receipt hash mismatch")
    return receipt


def validate_gpu_release_receipt(
    receipt: object,
    *,
    host_id: str,
    host_profile_sha256: str,
    gpu_uuid: str,
) -> dict[str, object]:
    expected = {
        "schema",
        "lease_id",
        "lease_receipt_sha256",
        "cleanup_receipt_sha256",
        "host_id",
        "host_profile_sha256",
        "gpu_id",
        "gpu_uuid",
        "request_id",
        "slot_id",
        "status",
        "released_at",
        "execution_eligible",
        "device_access",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("GPU release receipt has an unexpected shape")
    for field in ("lease_id", "gpu_id", "request_id", "slot_id"):
        value = receipt.get(field)
        if not isinstance(value, str):
            raise LabError(f"GPU release receipt has invalid {field}")
        validate_task_id(value)
    if (
        receipt.get("schema") != GPU_RELEASE_SCHEMA_ID
        or receipt.get("host_id") != host_id
        or receipt.get("host_profile_sha256") != host_profile_sha256
        or receipt.get("gpu_uuid") != gpu_uuid
        or receipt.get("status") != "RELEASED"
        or receipt.get("execution_eligible") is not False
        or receipt.get("device_access") is not False
        or not SHA256_PATTERN.fullmatch(
            str(receipt.get("lease_receipt_sha256", ""))
        )
        or not SHA256_PATTERN.fullmatch(
            str(receipt.get("cleanup_receipt_sha256", ""))
        )
    ):
        raise LabError("GPU release receipt identity or authority is invalid")
    _timestamp(receipt.get("released_at"), label="released_at")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != hashlib.sha256(
        _canonical_json(content)
    ).hexdigest():
        raise LabError("GPU release receipt hash mismatch")
    return receipt


class GpuLeaseStore:
    """Persistent, fail-closed single-GPU lease state for a protected host."""

    def __init__(
        self,
        state_root: Path,
        *,
        host_id: str,
        host_profile_sha256: str,
    ) -> None:
        validate_task_id(host_id)
        if not SHA256_PATTERN.fullmatch(host_profile_sha256):
            raise LabError("GPU lease store requires a host Profile SHA-256")
        requested = state_root.expanduser()
        if not requested.is_absolute() or requested.is_symlink():
            raise LabError(f"GPU lease state root is unsafe: {requested}")
        if (
            requested.parent.is_symlink()
            or requested.parent.resolve() != requested.parent.absolute()
        ):
            raise LabError(f"GPU lease state parent is unsafe: {requested.parent}")
        requested.mkdir(parents=True, exist_ok=True, mode=0o700)
        if requested.resolve() != requested.absolute():
            raise LabError(f"GPU lease state root escaped: {requested}")
        self.root = requested.absolute()
        self.host_id = host_id
        self.host_profile_sha256 = host_profile_sha256

    def _paths(self, gpu_uuid: str) -> tuple[Path, Path]:
        _validate_gpu_uuid(gpu_uuid)
        suffix = hashlib.sha256(gpu_uuid.encode()).hexdigest()
        return self.root / f"{suffix}.json", self.root / f"{suffix}.lock"

    @contextmanager
    def _locked(self, gpu_uuid: str) -> Iterator[Path]:
        state_path, lock_path = self._paths(gpu_uuid)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield state_path
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _load(self, state_path: Path, *, gpu_uuid: str) -> dict[str, object]:
        if not state_path.exists():
            return {
                "schema": GPU_STATE_SCHEMA_ID,
                "host_id": self.host_id,
                "host_profile_sha256": self.host_profile_sha256,
                "gpu_uuid": gpu_uuid,
                "generation": 0,
                "active": None,
                "last_release": None,
            }
        if state_path.is_symlink() or not state_path.is_file():
            raise LabError("GPU lease state is unsafe")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError(f"cannot read GPU lease state: {exc}") from exc
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "schema",
                "host_id",
                "host_profile_sha256",
                "gpu_uuid",
                "generation",
                "active",
                "last_release",
            }
            or state.get("schema") != GPU_STATE_SCHEMA_ID
            or state.get("host_id") != self.host_id
            or state.get("host_profile_sha256") != self.host_profile_sha256
            or state.get("gpu_uuid") != gpu_uuid
            or not isinstance(state.get("generation"), int)
        ):
            raise LabError("GPU lease state identity is invalid")
        active = state.get("active")
        if active is not None:
            validate_gpu_lease_receipt(
                active,
                host_id=self.host_id,
                gpu_uuid=gpu_uuid,
                require_active=False,
                host_profile_sha256=self.host_profile_sha256,
            )
        last_release = state.get("last_release")
        if last_release is not None:
            validate_gpu_release_receipt(
                last_release,
                host_id=self.host_id,
                host_profile_sha256=self.host_profile_sha256,
                gpu_uuid=gpu_uuid,
            )
        return state

    @staticmethod
    def _write(state_path: Path, state: dict[str, object]) -> None:
        _atomic_write_text(
            state_path,
            json.dumps(state, indent=2, sort_keys=True) + "\n",
        )

    def acquire(
        self,
        *,
        gpu_id: str,
        gpu_uuid: str,
        request_id: str,
        slot_id: str,
        validity_seconds: int = 900,
    ) -> dict[str, object]:
        for value in (gpu_id, request_id, slot_id):
            validate_task_id(value)
        _validate_gpu_uuid(gpu_uuid)
        if not 30 <= validity_seconds <= 3600:
            raise LabError("GPU lease validity must be between 30 and 3600 seconds")
        with self._locked(gpu_uuid) as state_path:
            state = self._load(state_path, gpu_uuid=gpu_uuid)
            active = state.get("active")
            if isinstance(active, dict):
                expires = _timestamp(active.get("expires_at"), label="expires_at")
                same_owner = (
                    active.get("request_id") == request_id
                    and active.get("slot_id") == slot_id
                    and active.get("gpu_id") == gpu_id
                )
                if datetime.now(timezone.utc) > expires:
                    raise LabError(
                        "expired GPU lease requires trusted cleanup/reconciliation"
                    )
                if same_owner:
                    return active
                raise LabError(
                    f"GPU is leased by request {active.get('request_id')}"
                )
            acquired = datetime.now(timezone.utc)
            identity = hashlib.sha256(
                _canonical_json(
                    {
                        "host_id": self.host_id,
                        "gpu_uuid": gpu_uuid,
                        "request_id": request_id,
                        "slot_id": slot_id,
                        "nonce": secrets.token_hex(16),
                    }
                )
            ).hexdigest()
            content: dict[str, object] = {
                "schema": GPU_LEASE_SCHEMA_ID,
                "lease_id": f"GL-{identity[:24]}",
                "host_id": self.host_id,
                "host_profile_sha256": self.host_profile_sha256,
                "gpu_id": gpu_id,
                "gpu_uuid": gpu_uuid,
                "request_id": request_id,
                "slot_id": slot_id,
                "status": "ACTIVE",
                "acquired_at": acquired.isoformat(),
                "expires_at": (
                    acquired + timedelta(seconds=validity_seconds)
                ).isoformat(),
                "execution_eligible": False,
                "device_access": False,
            }
            lease = _receipt(content)
            generation = state["generation"]
            assert isinstance(generation, int)
            state.update(
                {
                    "generation": generation + 1,
                    "active": lease,
                }
            )
            self._write(state_path, state)
            return lease

    def release(
        self,
        lease_receipt: object,
        *,
        gpu_uuid: str,
        cleanup_receipt: object,
    ) -> dict[str, object]:
        lease = validate_gpu_lease_receipt(
            lease_receipt,
            host_id=self.host_id,
            gpu_uuid=gpu_uuid,
            require_active=False,
            host_profile_sha256=self.host_profile_sha256,
        )
        cleanup = validate_gpu_cleanup_receipt(
            cleanup_receipt,
            lease_receipt=lease,
            host_id=self.host_id,
            gpu_uuid=gpu_uuid,
            host_profile_sha256=self.host_profile_sha256,
        )
        with self._locked(gpu_uuid) as state_path:
            state = self._load(state_path, gpu_uuid=gpu_uuid)
            active = state.get("active")
            last_release = state.get("last_release")
            if active is None and isinstance(last_release, dict):
                if last_release.get("lease_receipt_sha256") == lease["receipt_sha256"]:
                    return last_release
                raise LabError("GPU is no longer held by this lease")
            if (
                not isinstance(active, dict)
                or active.get("receipt_sha256") != lease["receipt_sha256"]
            ):
                raise LabError("GPU active lease changed before release")
            content: dict[str, object] = {
                "schema": GPU_RELEASE_SCHEMA_ID,
                "lease_id": lease["lease_id"],
                "lease_receipt_sha256": lease["receipt_sha256"],
                "cleanup_receipt_sha256": cleanup["receipt_sha256"],
                "host_id": self.host_id,
                "host_profile_sha256": self.host_profile_sha256,
                "gpu_id": lease["gpu_id"],
                "gpu_uuid": gpu_uuid,
                "request_id": lease["request_id"],
                "slot_id": lease["slot_id"],
                "status": "RELEASED",
                "released_at": datetime.now(timezone.utc).isoformat(),
                "execution_eligible": False,
                "device_access": False,
            }
            released = _receipt(content)
            validate_gpu_release_receipt(
                released,
                host_id=self.host_id,
                host_profile_sha256=self.host_profile_sha256,
                gpu_uuid=gpu_uuid,
            )
            generation = state["generation"]
            assert isinstance(generation, int)
            state.update(
                {
                    "generation": generation + 1,
                    "active": None,
                    "last_release": released,
                }
            )
            self._write(state_path, state)
            return released

    def active_lease(self, *, gpu_uuid: str) -> dict[str, object] | None:
        """Return the persistent active lease without changing resource state."""

        with self._locked(gpu_uuid) as state_path:
            state = self._load(state_path, gpu_uuid=gpu_uuid)
            active = state.get("active")
            if active is None:
                return None
            if not isinstance(active, dict):
                raise LabError("GPU active lease state is invalid")
            return validate_gpu_lease_receipt(
                active,
                host_id=self.host_id,
                gpu_uuid=gpu_uuid,
                require_active=False,
                host_profile_sha256=self.host_profile_sha256,
            )

    def observation(self, lease_receipt: object, *, gpu_uuid: str) -> dict[str, object]:
        lease = validate_gpu_lease_receipt(
            lease_receipt,
            host_id=self.host_id,
            gpu_uuid=gpu_uuid,
            require_active=True,
            host_profile_sha256=self.host_profile_sha256,
        )
        with self._locked(gpu_uuid) as state_path:
            state = self._load(state_path, gpu_uuid=gpu_uuid)
            active = state.get("active")
            if (
                not isinstance(active, dict)
                or active.get("receipt_sha256") != lease["receipt_sha256"]
            ):
                raise LabError("GPU lease is not the active persistent lease")
        return {
            "reserved_count": 1,
            "lease_id": lease["lease_id"],
            "gpu_id": lease["gpu_id"],
            "gpu_uuid": lease["gpu_uuid"],
            "lease_receipt_sha256": lease["receipt_sha256"],
            "lease_receipt": lease,
        }
