from __future__ import annotations

import os
import stat
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .execution_sandbox import (
    validate_service_sandbox_for_launch,
    validate_service_sandbox_receipt,
)
from .remote_preflight import HostProfile
from .remote_request import SHA256_PATTERN, _sha256
from .service_identity import (
    build_service_receipt,
    validate_service_health,
    validate_service_launch_spec_identity,
)
from .workspace import LabError, validate_task_id

MANAGED_SERVICE_PROCESS_SCHEMA_ID = "lab.managed-service-process.v1"
MANAGED_SERVICE_TEARDOWN_SCHEMA_ID = "lab.managed-service-teardown.v1"
_NAMESPACE_NAMES = ("user", "net", "pid", "mnt")


def _read_text(path: Path, *, label: str, max_bytes: int = 64 * 1024) -> str:
    try:
        metadata = path.stat(follow_symlinks=True)
        if not stat.S_ISREG(metadata.st_mode) or not 0 <= metadata.st_size <= max_bytes:
            raise LabError(f"{label} is not a bounded regular file")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LabError(f"cannot read {label}") from exc


def _process_stat(path: Path, *, expected_pid: int) -> dict[str, int]:
    raw = _read_text(path, label="managed service process stat")
    close = raw.rfind(")")
    if close < 2 or not raw.startswith(f"{expected_pid} ("):
        raise LabError("managed service process stat has an invalid identity")
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        raise LabError("managed service process stat is incomplete")
    try:
        return {
            "parent_pid": int(fields[1]),
            "process_group_id": int(fields[2]),
            "session_id": int(fields[3]),
            "proc_start_time_ticks": int(fields[19]),
        }
    except ValueError as exc:
        raise LabError("managed service process stat is invalid") from exc


def _no_new_privileges(path: Path) -> bool:
    status = _read_text(path, label="managed service process status")
    values = {
        key: value.strip()
        for line in status.splitlines()
        if ":" in line
        for key, value in (line.split(":", 1),)
    }
    return values.get("NoNewPrivs") == "1"


def _namespace_receipts(
    proc_root: Path,
    *,
    pid: int,
    collector_pid: int,
) -> dict[str, dict[str, object]]:
    namespaces: dict[str, dict[str, object]] = {}
    for name in _NAMESPACE_NAMES:
        parent_path = proc_root / str(collector_pid) / "ns" / name
        process_path = proc_root / str(pid) / "ns" / name
        try:
            parent_inode = parent_path.stat().st_ino
            process_inode = process_path.stat().st_ino
        except OSError as exc:
            raise LabError(f"cannot inspect managed service {name} namespace") from exc
        isolated = parent_inode != process_inode
        if not isolated:
            raise LabError(f"managed service did not isolate {name} namespace")
        namespaces[name] = {
            "collector_inode": parent_inode,
            "process_inode": process_inode,
            "isolated": True,
        }
    return namespaces


def collect_managed_service_process_receipt(
    *,
    launch_spec: object,
    service_sandbox_receipt: object,
    host_profile: HostProfile,
    pid: int,
    endpoint_port: int,
    control_plane_version: str,
    started_at: datetime,
    proc_root: Path = Path("/proc"),
    collector_pid: int | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Collect OS identity for a live, direct child launched in the service sandbox.

    Production callers use the fixed `/proc`. The injectable root exists only for
    hermetic verifier tests and is not exposed by `projectctl`.
    """

    spec = validate_service_launch_spec_identity(launch_spec)
    sandbox = validate_service_sandbox_for_launch(
        service_sandbox_receipt,
        launch_spec=spec,
        host_profile=host_profile,
        now=now,
    )
    validate_task_id(control_plane_version)
    if sandbox.get("control_plane_version") != control_plane_version:
        raise LabError("managed service uses a different control-plane version")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise LabError("managed service PID is invalid")
    if (
        not isinstance(endpoint_port, int)
        or isinstance(endpoint_port, bool)
        or not 1024 <= endpoint_port <= 65535
    ):
        raise LabError("managed service endpoint port is invalid")
    collector = os.getpid() if collector_pid is None else collector_pid
    if not isinstance(collector, int) or isinstance(collector, bool) or collector <= 0:
        raise LabError("managed service collector PID is invalid")
    requested_proc = proc_root.expanduser()
    resolved_proc = requested_proc.resolve()
    if (
        not requested_proc.is_absolute()
        or requested_proc.is_symlink()
        or requested_proc.absolute() != resolved_proc
        or not resolved_proc.is_dir()
    ):
        raise LabError("managed service proc root is unsafe")
    process_root = resolved_proc / str(pid)
    before = _process_stat(process_root / "stat", expected_pid=pid)
    if (
        before["parent_pid"] != collector
        or before["process_group_id"] != pid
        or before["session_id"] != pid
        or before["proc_start_time_ticks"] <= 0
    ):
        raise LabError(
            "managed service is not a direct isolated-session launcher child"
        )
    if not _no_new_privileges(process_root / "status"):
        raise LabError("managed service did not enable NoNewPrivs")
    namespaces = _namespace_receipts(
        resolved_proc,
        pid=pid,
        collector_pid=collector,
    )
    raw_environment = spec.get("environment")
    if not isinstance(raw_environment, dict):
        raise LabError("managed service environment identity is invalid")
    environment_id = raw_environment.get("environment_id")
    if not isinstance(environment_id, str):
        raise LabError("managed service environment ID is invalid")
    environment = host_profile.environments.get(environment_id)
    if environment is None:
        raise LabError("managed service environment is not in the host Profile")
    expected_executable = (environment.path / "bin" / "python").resolve()
    executable_link = process_root / "exe"
    try:
        executable = executable_link.resolve(strict=True)
        executable_metadata = executable.stat()
    except OSError as exc:
        raise LabError("cannot inspect managed service executable") from exc
    if (
        executable != expected_executable
        or not stat.S_ISREG(executable_metadata.st_mode)
        or not executable_metadata.st_mode & 0o111
    ):
        raise LabError("managed service executable differs from its environment")
    after = _process_stat(process_root / "stat", expected_pid=pid)
    if after != before:
        raise LabError("managed service process identity changed while observed")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or started_at.tzinfo is None:
        raise LabError("managed service process clocks require a timezone")
    if started_at > current:
        raise LabError("managed service process started_at is in the future")
    endpoint_policy = spec.get("endpoint_policy")
    if not isinstance(endpoint_policy, dict):
        raise LabError("managed service endpoint policy is invalid")
    content: dict[str, object] = {
        "schema": MANAGED_SERVICE_PROCESS_SCHEMA_ID,
        "service_id": spec["service_id"],
        "request_id": spec["request_id"],
        "slot_id": spec["slot_id"],
        "host_id": host_profile.host_id,
        "host_profile_sha256": host_profile.sha256,
        "control_plane_version": control_plane_version,
        "launch_spec_sha256": spec["launch_spec_sha256"],
        "service_sandbox_receipt_sha256": sandbox["receipt_sha256"],
        "pid": pid,
        **before,
        "executable": {
            "realpath": str(executable),
            "device": executable_metadata.st_dev,
            "inode": executable_metadata.st_ino,
            "size_bytes": executable_metadata.st_size,
            "mtime_ns": executable_metadata.st_mtime_ns,
        },
        "endpoint": {
            "scheme": endpoint_policy["scheme"],
            "host": "127.0.0.1",
            "port": endpoint_port,
        },
        "namespaces": namespaces,
        "no_new_privileges": True,
        "managed_direct_child": True,
        "process_group_isolated": True,
        "alive_at_observation": True,
        "started_at": started_at.isoformat(),
        "observed_at": current.isoformat(),
        "device_access": False,
    }
    return {**content, "receipt_sha256": _sha256(content)}


def validate_managed_service_process_receipt(
    receipt: object,
    *,
    launch_spec: object,
    service_sandbox_receipt: object,
) -> dict[str, object]:
    spec = validate_service_launch_spec_identity(launch_spec)
    if not isinstance(service_sandbox_receipt, dict):
        raise LabError("managed service sandbox receipt must be an object")
    expected = {
        "schema",
        "service_id",
        "request_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "control_plane_version",
        "launch_spec_sha256",
        "service_sandbox_receipt_sha256",
        "pid",
        "parent_pid",
        "process_group_id",
        "session_id",
        "proc_start_time_ticks",
        "executable",
        "endpoint",
        "namespaces",
        "no_new_privileges",
        "managed_direct_child",
        "process_group_isolated",
        "alive_at_observation",
        "started_at",
        "observed_at",
        "device_access",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("managed service process receipt has an unexpected shape")
    if receipt.get("schema") != MANAGED_SERVICE_PROCESS_SCHEMA_ID:
        raise LabError("unsupported managed service process receipt schema")
    bindings = {
        "service_id": spec["service_id"],
        "request_id": spec["request_id"],
        "slot_id": spec["slot_id"],
        "launch_spec_sha256": spec["launch_spec_sha256"],
        "service_sandbox_receipt_sha256": service_sandbox_receipt.get(
            "receipt_sha256"
        ),
    }
    for field, expected_value in bindings.items():
        if receipt.get(field) != expected_value:
            raise LabError(f"managed service process identity mismatch: {field}")
    for field in (
        "service_id",
        "request_id",
        "slot_id",
        "host_id",
        "control_plane_version",
    ):
        value = receipt.get(field)
        if not isinstance(value, str):
            raise LabError(f"managed service process has invalid {field}")
        validate_task_id(value)
    for field in (
        "host_profile_sha256",
        "launch_spec_sha256",
        "service_sandbox_receipt_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"managed service process has invalid {field}")
    pid = receipt.get("pid")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or receipt.get("process_group_id") != pid
        or receipt.get("session_id") != pid
        or not isinstance(receipt.get("parent_pid"), int)
        or not isinstance(receipt.get("proc_start_time_ticks"), int)
        or receipt["proc_start_time_ticks"] <= 0
    ):
        raise LabError("managed service process has invalid process identity")
    executable = receipt.get("executable")
    if not isinstance(executable, dict) or set(executable) != {
        "realpath",
        "device",
        "inode",
        "size_bytes",
        "mtime_ns",
    }:
        raise LabError("managed service process has invalid executable identity")
    if (
        not isinstance(executable.get("realpath"), str)
        or not Path(str(executable["realpath"])).is_absolute()
        or not all(
            isinstance(executable.get(field), int)
            and not isinstance(executable.get(field), bool)
            and executable[field] >= 0
            for field in ("device", "inode", "size_bytes", "mtime_ns")
        )
    ):
        raise LabError("managed service process has invalid executable identity")
    endpoint = receipt.get("endpoint")
    endpoint_policy = spec.get("endpoint_policy")
    if (
        not isinstance(endpoint, dict)
        or set(endpoint) != {"scheme", "host", "port"}
        or not isinstance(endpoint_policy, dict)
        or endpoint.get("scheme") != endpoint_policy.get("scheme")
        or endpoint.get("host") != "127.0.0.1"
        or not isinstance(endpoint.get("port"), int)
        or isinstance(endpoint.get("port"), bool)
        or not 1024 <= endpoint["port"] <= 65535
    ):
        raise LabError("managed service process has an invalid endpoint")
    namespaces = receipt.get("namespaces")
    if not isinstance(namespaces, dict) or set(namespaces) != set(_NAMESPACE_NAMES):
        raise LabError("managed service process has invalid namespaces")
    for name, identity in namespaces.items():
        if (
            not isinstance(identity, dict)
            or set(identity) != {"collector_inode", "process_inode", "isolated"}
            or not isinstance(identity.get("collector_inode"), int)
            or not isinstance(identity.get("process_inode"), int)
            or identity.get("collector_inode") == identity.get("process_inode")
            or identity.get("isolated") is not True
        ):
            raise LabError(f"managed service process has invalid {name} namespace")
    if (
        receipt.get("no_new_privileges") is not True
        or receipt.get("managed_direct_child") is not True
        or receipt.get("process_group_isolated") is not True
        or receipt.get("alive_at_observation") is not True
        or receipt.get("device_access") is not False
    ):
        raise LabError("managed service process has an unsafe authority envelope")
    timestamps: dict[str, datetime] = {}
    for field in ("started_at", "observed_at"):
        value = receipt.get(field)
        if not isinstance(value, str):
            raise LabError(f"managed service process has invalid {field}")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise LabError(f"managed service process has invalid {field}") from exc
        if parsed.tzinfo is None:
            raise LabError(f"managed service process {field} requires a timezone")
        timestamps[field] = parsed
    if timestamps["started_at"] > timestamps["observed_at"]:
        raise LabError("managed service process starts after its observation")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("managed service process receipt hash mismatch")
    return receipt


def build_attested_service_receipt(
    health: object,
    *,
    launch_spec: object,
    service_sandbox_receipt: object,
    managed_process_receipt: object,
    host_profile: HostProfile,
    now: datetime | None = None,
) -> dict[str, object]:
    """Bind service health to an independently observed managed process.

    The health endpoint is treated as task payload. It cannot nominate a PID,
    endpoint, or start time that differs from the protected launcher's receipt.
    """

    spec = validate_service_launch_spec_identity(launch_spec)
    sandbox = validate_service_sandbox_for_launch(
        service_sandbox_receipt,
        launch_spec=spec,
        host_profile=host_profile,
        now=now,
    )
    process = validate_managed_service_process_receipt(
        managed_process_receipt,
        launch_spec=spec,
        service_sandbox_receipt=sandbox,
    )
    if process.get("host_id") != host_profile.host_id or process.get(
        "host_profile_sha256"
    ) != host_profile.sha256:
        raise LabError("managed service process uses a different host Profile")
    validated_health = validate_service_health(health, launch_spec=spec)
    health_process = validated_health.get("process")
    if health_process != {
        "pid": process["pid"],
        "started_at": process["started_at"],
    }:
        raise LabError("service health process differs from launcher observation")
    if validated_health.get("endpoint") != process.get("endpoint"):
        raise LabError("service health endpoint differs from launcher observation")
    return build_service_receipt(
        validated_health,
        launch_spec=spec,
        managed_process_receipt=process,
    )


def _process_group_members(proc_root: Path, *, process_group_id: int) -> list[int]:
    members: list[int] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise LabError("cannot enumerate managed service process table") from exc
    for entry in entries:
        if not entry.name.isascii() or not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        if pid <= 0:
            continue
        try:
            identity = _process_stat(entry / "stat", expected_pid=pid)
        except LabError as exc:
            if not entry.exists():
                continue
            raise LabError(
                f"cannot prove managed service process-group absence at PID {pid}"
            ) from exc
        if identity["process_group_id"] == process_group_id:
            members.append(pid)
    return sorted(members)


def collect_managed_service_teardown_receipt(
    *,
    launch_spec: object,
    service_sandbox_receipt: object,
    managed_process_receipt: object,
    proc_root: Path = Path("/proc"),
    collector_pid: int | None = None,
    waitpid: Callable[[int, int], tuple[int, int]] = os.waitpid,
    now: datetime | None = None,
) -> dict[str, object]:
    """Prove that the protected launcher reaped its child and process group.

    Termination is deliberately outside this collector. The trusted launcher
    first performs reverse-order shutdown, then calls this function from the
    same parent process. Hermetic tests may inject a fake proc root and waitpid.
    """

    spec = validate_service_launch_spec_identity(launch_spec)
    sandbox = validate_service_sandbox_receipt(service_sandbox_receipt)
    process = validate_managed_service_process_receipt(
        managed_process_receipt,
        launch_spec=spec,
        service_sandbox_receipt=sandbox,
    )
    collector = os.getpid() if collector_pid is None else collector_pid
    if (
        not isinstance(collector, int)
        or isinstance(collector, bool)
        or collector <= 0
        or process.get("parent_pid") != collector
    ):
        raise LabError("managed service teardown must use the original launcher")
    requested_proc = proc_root.expanduser()
    resolved_proc = requested_proc.resolve()
    if (
        not requested_proc.is_absolute()
        or requested_proc.is_symlink()
        or requested_proc.absolute() != resolved_proc
        or not resolved_proc.is_dir()
    ):
        raise LabError("managed service teardown proc root is unsafe")
    pid = process.get("pid")
    start_ticks = process.get("proc_start_time_ticks")
    assert isinstance(pid, int)
    assert isinstance(start_ticks, int)
    try:
        waited_pid, wait_status = waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError) as exc:
        raise LabError("managed service was not reaped by its protected launcher") from exc
    if (
        not isinstance(waited_pid, int)
        or isinstance(waited_pid, bool)
        or waited_pid != pid
        or not isinstance(wait_status, int)
        or isinstance(wait_status, bool)
        or wait_status < 0
    ):
        raise LabError("managed service has not exited and been reaped")
    process_root = resolved_proc / str(pid)
    pid_reused = False
    if process_root.exists():
        observed = _process_stat(process_root / "stat", expected_pid=pid)
        if observed["proc_start_time_ticks"] == start_ticks:
            raise LabError("managed service process remains present after waitpid")
        pid_reused = True
    process_group_id = process.get("process_group_id")
    assert isinstance(process_group_id, int)
    members = _process_group_members(
        resolved_proc,
        process_group_id=process_group_id,
    )
    if members:
        raise LabError(f"managed service process group remains active: {members}")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("managed service teardown clock requires a timezone")
    observed_at = process.get("observed_at")
    assert isinstance(observed_at, str)
    observed_process = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    if current < observed_process:
        raise LabError("managed service teardown predates process observation")
    exited_normally = os.WIFEXITED(wait_status)
    if not exited_normally and not os.WIFSIGNALED(wait_status):
        raise LabError("managed service wait status does not prove termination")
    content: dict[str, object] = {
        "schema": MANAGED_SERVICE_TEARDOWN_SCHEMA_ID,
        "service_id": spec["service_id"],
        "request_id": spec["request_id"],
        "slot_id": spec["slot_id"],
        "launch_spec_sha256": spec["launch_spec_sha256"],
        "service_sandbox_receipt_sha256": sandbox["receipt_sha256"],
        "managed_process_receipt_sha256": process["receipt_sha256"],
        "pid": pid,
        "proc_start_time_ticks": start_ticks,
        "process_group_id": process_group_id,
        "wait_status": wait_status,
        "exited_normally": exited_normally,
        "exit_code": os.WEXITSTATUS(wait_status) if exited_normally else None,
        "term_signal": os.WTERMSIG(wait_status) if os.WIFSIGNALED(wait_status) else None,
        "original_process_absent": True,
        "process_group_absent": True,
        "pid_reused": pid_reused,
        "reaped_by_original_launcher": True,
        "teardown_verified": True,
        "observed_at": current.isoformat(),
        "device_access": False,
        "hardware_safety_claimed": False,
    }
    return {**content, "receipt_sha256": _sha256(content)}


def validate_managed_service_teardown_receipt(
    receipt: object,
    *,
    launch_spec: object,
    service_sandbox_receipt: object,
    managed_process_receipt: object,
) -> dict[str, object]:
    spec = validate_service_launch_spec_identity(launch_spec)
    sandbox = validate_service_sandbox_receipt(service_sandbox_receipt)
    process = validate_managed_service_process_receipt(
        managed_process_receipt,
        launch_spec=spec,
        service_sandbox_receipt=sandbox,
    )
    expected = {
        "schema",
        "service_id",
        "request_id",
        "slot_id",
        "launch_spec_sha256",
        "service_sandbox_receipt_sha256",
        "managed_process_receipt_sha256",
        "pid",
        "proc_start_time_ticks",
        "process_group_id",
        "wait_status",
        "exited_normally",
        "exit_code",
        "term_signal",
        "original_process_absent",
        "process_group_absent",
        "pid_reused",
        "reaped_by_original_launcher",
        "teardown_verified",
        "observed_at",
        "device_access",
        "hardware_safety_claimed",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("managed service teardown receipt has an unexpected shape")
    bindings = {
        "schema": MANAGED_SERVICE_TEARDOWN_SCHEMA_ID,
        "service_id": spec["service_id"],
        "request_id": spec["request_id"],
        "slot_id": spec["slot_id"],
        "launch_spec_sha256": spec["launch_spec_sha256"],
        "service_sandbox_receipt_sha256": sandbox["receipt_sha256"],
        "managed_process_receipt_sha256": process["receipt_sha256"],
        "pid": process["pid"],
        "proc_start_time_ticks": process["proc_start_time_ticks"],
        "process_group_id": process["process_group_id"],
    }
    for field, expected_value in bindings.items():
        if receipt.get(field) != expected_value:
            raise LabError(f"managed service teardown identity mismatch: {field}")
    for field in (
        "launch_spec_sha256",
        "service_sandbox_receipt_sha256",
        "managed_process_receipt_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"managed service teardown has invalid {field}")
    wait_status = receipt.get("wait_status")
    if (
        not isinstance(wait_status, int)
        or isinstance(wait_status, bool)
        or wait_status < 0
        or (not os.WIFEXITED(wait_status) and not os.WIFSIGNALED(wait_status))
        or receipt.get("exited_normally") is not os.WIFEXITED(wait_status)
        or receipt.get("exit_code")
        != (os.WEXITSTATUS(wait_status) if os.WIFEXITED(wait_status) else None)
        or receipt.get("term_signal")
        != (os.WTERMSIG(wait_status) if os.WIFSIGNALED(wait_status) else None)
    ):
        raise LabError("managed service teardown has invalid wait status")
    if (
        receipt.get("original_process_absent") is not True
        or receipt.get("process_group_absent") is not True
        or not isinstance(receipt.get("pid_reused"), bool)
        or receipt.get("reaped_by_original_launcher") is not True
        or receipt.get("teardown_verified") is not True
        or receipt.get("device_access") is not False
        or receipt.get("hardware_safety_claimed") is not False
    ):
        raise LabError("managed service teardown has an unsafe evidence envelope")
    observed_at = receipt.get("observed_at")
    if not isinstance(observed_at, str):
        raise LabError("managed service teardown has invalid observed_at")
    try:
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("managed service teardown has invalid observed_at") from exc
    if observed.tzinfo is None:
        raise LabError("managed service teardown observed_at requires a timezone")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("managed service teardown receipt hash mismatch")
    return receipt
