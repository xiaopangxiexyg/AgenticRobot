from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

from .remote_preflight import HostProfile, SandboxBackendPolicy
from .remote_request import SHA256_PATTERN, _sha256
from .service_identity import (
    validate_service_launch_spec_identity,
    validate_service_receipt,
)
from .slot_readiness import validate_slot_readiness_receipt
from .workspace import LabError, validate_task_id

SANDBOX_BACKEND_PROBE_SCHEMA_ID = "lab.sandbox-backend-probe.v1"
EXECUTION_SANDBOX_SCHEMA_ID = "lab.execution-sandbox.v1"
SERVICE_SANDBOX_SCHEMA_ID = "lab.service-sandbox.v1"
_GPU_DEVICE_PATTERN = re.compile(
    r"/dev/nvidia(?:[0-9]+|ctl|modeset|-uvm|-uvm-tools)"
)
_MAX_BACKEND_BYTES = 4 * 1024 * 1024
_MAX_PROBE_OUTPUT_BYTES = 64 * 1024
_PROBE_TIMEOUT_SECONDS = 15.0
_PROBE_SCRIPT = r"""
import glob
import json
import os
import socket
import stat
import sys

parents = json.loads(sys.argv[1])
allowed = json.loads(sys.argv[2])

def ns_changed(name):
    return os.stat(f"/proc/self/ns/{name}").st_ino != parents[name]

loopback_available = False
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client.settimeout(1.0)
    client.connect(server.getsockname())
    accepted, _ = server.accept()
    accepted.close()
    loopback_available = True
finally:
    client.close()
    server.close()

external_network_unreachable = False
probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    probe.settimeout(0.2)
    probe.connect(("1.1.1.1", 53))
except OSError:
    external_network_unreachable = True
finally:
    probe.close()

allowed_devices_present = all(
    os.path.exists(path)
    and not os.path.islink(path)
    and stat.S_ISCHR(os.stat(path, follow_symlinks=False).st_mode)
    for path in allowed
)
forbidden = []
for pattern in ("/dev/video*", "/dev/ttyUSB*", "/dev/ttyACM*"):
    forbidden.extend(glob.glob(pattern))
if os.path.exists("/dev/input"):
    forbidden.append("/dev/input")

no_new_privileges = False
with open("/proc/self/status", encoding="utf-8") as status:
    for line in status:
        if line.startswith("NoNewPrivs:"):
            no_new_privileges = line.split(":", 1)[1].strip() == "1"
            break

print(json.dumps({
    "allowed_devices": allowed,
    "allowed_devices_present": allowed_devices_present,
    "external_network_unreachable": external_network_unreachable,
    "filesystem_isolated": all(
        not os.path.exists(path) for path in ("/home", "/root", "/mnt")
    ),
    "forbidden_devices_absent": not forbidden,
    "loopback_available": loopback_available,
    "network_namespace": ns_changed("net"),
    "no_new_privileges": no_new_privileges,
    "process_namespace": ns_changed("pid"),
    "user_namespace": ns_changed("user"),
}, sort_keys=True))
""".strip()


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise LabError(f"{label} has an invalid timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError(f"{label} has an invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise LabError(f"{label} timestamp requires a timezone")
    return parsed


def build_sandbox_backend_probe_receipt(
    *,
    host_id: str,
    host_profile_sha256: str,
    backend_sha256: str,
    allowed_devices: tuple[str, ...],
    user_namespace: bool,
    network_namespace: bool,
    loopback_available: bool,
    external_network_unreachable: bool,
    filesystem_isolated: bool,
    forbidden_devices_absent: bool,
    process_namespace: bool,
    no_new_privileges: bool,
    allowed_devices_present: bool = True,
    backend_executable: str = "/usr/bin/bwrap",
    observed_at: datetime | None = None,
) -> dict[str, object]:
    """Create a probe receipt from a protected backend's measured facts.

    This is not itself a probe. Production may call it only after executing the
    fixed, installed bubblewrap capability probe.
    """

    validate_task_id(host_id)
    if not SHA256_PATTERN.fullmatch(host_profile_sha256) or not SHA256_PATTERN.fullmatch(
        backend_sha256
    ):
        raise LabError("sandbox probe requires full protected hashes")
    facts = (
        user_namespace,
        network_namespace,
        loopback_available,
        external_network_unreachable,
        filesystem_isolated,
        forbidden_devices_absent,
        process_namespace,
        no_new_privileges,
        allowed_devices_present,
    )
    if not all(isinstance(value, bool) for value in facts):
        raise LabError("sandbox probe facts must be booleans")
    content: dict[str, object] = {
        "schema": SANDBOX_BACKEND_PROBE_SCHEMA_ID,
        "host_id": host_id,
        "host_profile_sha256": host_profile_sha256,
        "backend_id": "bubblewrap-v1",
        "backend_executable": backend_executable,
        "backend_sha256": backend_sha256,
        "allowed_devices": list(allowed_devices),
        "allowed_devices_present": allowed_devices_present,
        "user_namespace": user_namespace,
        "network_namespace": network_namespace,
        "loopback_available": loopback_available,
        "external_network_unreachable": external_network_unreachable,
        "filesystem_isolated": filesystem_isolated,
        "forbidden_devices_absent": forbidden_devices_absent,
        "process_namespace": process_namespace,
        "no_new_privileges": no_new_privileges,
        "read_only": True,
        "device_control": False,
        "observed_at": (observed_at or datetime.now(timezone.utc)).isoformat(),
    }
    return {**content, "receipt_sha256": _sha256(content)}


def validate_sandbox_backend_probe_receipt(receipt: object) -> dict[str, object]:
    expected = {
        "schema",
        "host_id",
        "host_profile_sha256",
        "backend_id",
        "backend_executable",
        "backend_sha256",
        "allowed_devices",
        "allowed_devices_present",
        "user_namespace",
        "network_namespace",
        "loopback_available",
        "external_network_unreachable",
        "filesystem_isolated",
        "forbidden_devices_absent",
        "process_namespace",
        "no_new_privileges",
        "read_only",
        "device_control",
        "observed_at",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("sandbox backend probe receipt has an unexpected shape")
    host_id = receipt.get("host_id")
    allowed_devices = receipt.get("allowed_devices")
    backend_executable = receipt.get("backend_executable")
    if not isinstance(host_id, str):
        raise LabError("sandbox backend probe has an invalid host ID")
    validate_task_id(host_id)
    if (
        receipt.get("schema") != SANDBOX_BACKEND_PROBE_SCHEMA_ID
        or receipt.get("backend_id") != "bubblewrap-v1"
        or not isinstance(backend_executable, str)
        or not Path(backend_executable).is_absolute()
        or ".." in Path(backend_executable).parts
        or Path(backend_executable).name != "bwrap"
        or not SHA256_PATTERN.fullmatch(
            str(receipt.get("host_profile_sha256", ""))
        )
        or not SHA256_PATTERN.fullmatch(str(receipt.get("backend_sha256", "")))
        or not isinstance(allowed_devices, list)
        or not allowed_devices
        or len(allowed_devices) != len(set(allowed_devices))
        or not all(
            isinstance(device, str) and _GPU_DEVICE_PATTERN.fullmatch(device)
            for device in allowed_devices
        )
        or receipt.get("read_only") is not True
        or receipt.get("device_control") is not False
    ):
        raise LabError("sandbox backend probe identity or device policy is invalid")
    for field in (
        "user_namespace",
        "network_namespace",
        "loopback_available",
        "external_network_unreachable",
        "filesystem_isolated",
        "forbidden_devices_absent",
        "process_namespace",
        "no_new_privileges",
        "allowed_devices_present",
    ):
        if receipt.get(field) is not True:
            raise LabError(f"sandbox backend probe did not enforce {field}")
    _timestamp(receipt.get("observed_at"), label="sandbox backend probe")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("sandbox backend probe receipt hash mismatch")
    return receipt


def _open_protected_backend(path: Path) -> tuple[int, str, os.stat_result]:
    if not path.is_absolute() or ".." in path.parts or path.name != "bwrap":
        raise LabError("sandbox backend executable path is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LabError(f"cannot safely open sandbox backend: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not metadata.st_mode & 0o111
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_BACKEND_BYTES
        ):
            raise LabError("sandbox backend is not a bounded executable file")
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, digest.hexdigest(), metadata
    except Exception:
        os.close(descriptor)
        raise


def _validate_allowed_device_nodes(devices: tuple[Path, ...]) -> None:
    for device in devices:
        if not _GPU_DEVICE_PATTERN.fullmatch(str(device)):
            raise LabError(f"sandbox policy permits a non-GPU device: {device}")
        try:
            metadata = device.lstat()
        except OSError as exc:
            raise LabError(f"sandbox GPU device is unavailable: {device}") from exc
        if device.is_symlink() or not stat.S_ISCHR(metadata.st_mode):
            raise LabError(f"sandbox GPU device is not a real character device: {device}")


def _fixed_probe_argv(
    policy: SandboxBackendPolicy,
    *,
    parent_namespaces: dict[str, int],
) -> list[str]:
    argv = [
        str(policy.executable),
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for system_path in ("/bin", "/lib", "/lib64"):
        if Path(system_path).exists():
            argv.extend(("--ro-bind", system_path, system_path))
    argv.extend(("--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"))
    for device in policy.allowed_gpu_devices:
        argv.extend(("--dev-bind", str(device), str(device)))
    argv.extend(
        (
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "/usr/bin/python3",
            "-I",
            "-c",
            _PROBE_SCRIPT,
            json.dumps(parent_namespaces, sort_keys=True, separators=(",", ":")),
            json.dumps(
                [str(device) for device in policy.allowed_gpu_devices],
                separators=(",", ":"),
            ),
        )
    )
    return argv


def collect_sandbox_backend_probe(host_profile: HostProfile) -> dict[str, object]:
    """Execute only the protected, fixed bubblewrap capability probe.

    No task command, task environment, source directory, home directory, external
    network route, or device outside the host Profile can enter this invocation.
    """

    policy = host_profile.sandbox_backend
    if policy is None:
        raise LabError("host Profile lacks a sandbox backend policy")
    if policy.backend_id != "bubblewrap-v1":
        raise LabError("unsupported sandbox backend")
    _validate_allowed_device_nodes(policy.allowed_gpu_devices)
    parent_namespaces = {
        name: os.stat(f"/proc/self/ns/{name}").st_ino
        for name in ("user", "net", "pid")
    }
    descriptor, backend_sha256, before = _open_protected_backend(policy.executable)
    argv = _fixed_probe_argv(policy, parent_namespaces=parent_namespaces)
    try:
        try:
            completed = subprocess.run(
                argv,
                executable=f"/proc/self/fd/{descriptor}",
                pass_fds=(descriptor,),
                check=False,
                capture_output=True,
                timeout=_PROBE_TIMEOUT_SECONDS,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LabError("protected sandbox backend probe could not complete") from exc
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise LabError("sandbox backend changed while it was probed")
    finally:
        os.close(descriptor)
    if (
        len(completed.stdout) > _MAX_PROBE_OUTPUT_BYTES
        or len(completed.stderr) > _MAX_PROBE_OUTPUT_BYTES
    ):
        raise LabError("sandbox backend probe output exceeded the protected limit")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LabError(f"sandbox backend probe failed: {detail or completed.returncode}")
    try:
        facts = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabError("sandbox backend probe returned invalid JSON") from exc
    expected_facts = {
        "allowed_devices",
        "allowed_devices_present",
        "external_network_unreachable",
        "filesystem_isolated",
        "forbidden_devices_absent",
        "loopback_available",
        "network_namespace",
        "no_new_privileges",
        "process_namespace",
        "user_namespace",
    }
    if not isinstance(facts, dict) or set(facts) != expected_facts:
        raise LabError("sandbox backend probe returned an unexpected result")
    boolean_facts = expected_facts - {"allowed_devices"}
    if not all(isinstance(facts.get(field), bool) for field in boolean_facts):
        raise LabError("sandbox backend probe returned non-boolean facts")
    allowed_devices = tuple(str(device) for device in policy.allowed_gpu_devices)
    if facts.get("allowed_devices") != list(allowed_devices):
        raise LabError("sandbox backend probe changed the protected device policy")
    receipt = build_sandbox_backend_probe_receipt(
        host_id=host_profile.host_id,
        host_profile_sha256=host_profile.sha256,
        backend_sha256=backend_sha256,
        backend_executable=str(policy.executable),
        allowed_devices=allowed_devices,
        user_namespace=cast(bool, facts["user_namespace"]),
        network_namespace=cast(bool, facts["network_namespace"]),
        loopback_available=cast(bool, facts["loopback_available"]),
        external_network_unreachable=cast(
            bool, facts["external_network_unreachable"]
        ),
        filesystem_isolated=cast(bool, facts["filesystem_isolated"]),
        forbidden_devices_absent=cast(bool, facts["forbidden_devices_absent"]),
        process_namespace=cast(bool, facts["process_namespace"]),
        no_new_privileges=cast(bool, facts["no_new_privileges"]),
        allowed_devices_present=cast(bool, facts["allowed_devices_present"]),
    )
    return validate_sandbox_backend_probe_receipt(receipt)


def _live_canonical_path(
    value: object,
    *,
    label: str,
    allowed_roots: tuple[Path, ...] | None = None,
    directory: bool | None = None,
) -> Path:
    if not isinstance(value, str):
        raise LabError(f"{label} must be an absolute path")
    requested = Path(value).expanduser()
    resolved = requested.resolve()
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or requested.absolute() != resolved
        or not resolved.exists()
    ):
        raise LabError(f"{label} is missing, escaped, or uses a symlink")
    if directory is True and not resolved.is_dir():
        raise LabError(f"{label} must be a directory")
    if directory is False and not resolved.is_file():
        raise LabError(f"{label} must be a regular file")
    if allowed_roots is not None:
        if not allowed_roots or any(
            root.is_symlink() or root.resolve() != root or not root.is_dir()
            for root in allowed_roots
        ):
            raise LabError(f"{label} has unsafe protected roots")
        if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
            raise LabError(f"{label} is outside protected roots")
    return resolved


def _service_mount(
    *,
    kind: str,
    name: str,
    source: Path,
    target: str,
    read_only: bool,
) -> dict[str, object]:
    return {
        "kind": kind,
        "name": name,
        "source": str(source),
        "target": target,
        "read_only": read_only,
    }


def build_service_sandbox_receipt(
    *,
    launch_spec: object,
    readiness_receipt: object,
    backend_probe_receipt: object,
    host_profile: HostProfile,
    control_plane_version: str,
    runtime_root: Path,
    now: datetime | None = None,
) -> dict[str, object]:
    """Bind the exact read-only service inputs to a live sandbox probe.

    This receipt authorizes only the trusted service launcher. It does not start
    a process and cannot authorize a Task command or device access.
    """

    spec = validate_service_launch_spec_identity(launch_spec)
    readiness = validate_slot_readiness_receipt(
        readiness_receipt,
        request_id=str(spec["request_id"]),
        idempotency_key=str(spec["idempotency_key"]),
        slot_id=str(spec["slot_id"]),
    )
    if (
        readiness.get("receipt_sha256") != spec.get("readiness_receipt_sha256")
        or readiness.get("host_profile_sha256") != host_profile.sha256
        or readiness.get("remote_profile_sha256")
        != spec.get("remote_profile_sha256")
        or readiness.get("control_plane_version") != control_plane_version
    ):
        raise LabError("service sandbox readiness identity mismatch")
    validate_task_id(control_plane_version)
    policy = host_profile.sandbox_backend
    if policy is None:
        raise LabError("host Profile lacks a service sandbox backend")
    if (
        host_profile.device_access
        or host_profile.network_policy.task_payload_egress != "loopback-only"
        or host_profile.network_policy.implicit_downloads
        or host_profile.network_policy.large_file_downloads
    ):
        raise LabError("host Profile has an unsafe service authority envelope")
    probe = validate_sandbox_backend_probe_receipt(backend_probe_receipt)
    if (
        probe.get("host_id") != host_profile.host_id
        or probe.get("host_profile_sha256") != host_profile.sha256
        or probe.get("backend_id") != policy.backend_id
        or probe.get("backend_executable") != str(policy.executable)
        or probe.get("allowed_devices")
        != [str(device) for device in policy.allowed_gpu_devices]
    ):
        raise LabError("service sandbox probe does not match the host Profile")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("service sandbox clock requires a timezone")
    observed = _timestamp(probe.get("observed_at"), label="sandbox backend probe")
    age_seconds = (current - observed).total_seconds()
    if age_seconds < -5 or age_seconds > policy.max_probe_age_seconds:
        raise LabError("service sandbox backend probe is stale or from the future")

    raw_environment_identity = spec.get("environment")
    raw_environments = readiness.get("environments")
    raw_source = spec.get("source")
    if (
        not isinstance(raw_environment_identity, Mapping)
        or not isinstance(raw_environments, Mapping)
        or not isinstance(raw_source, Mapping)
    ):
        raise LabError("service sandbox source/environment identity is incomplete")
    environment_id = raw_environment_identity.get("environment_id")
    if not isinstance(environment_id, str):
        raise LabError("service sandbox environment ID is invalid")
    placement = host_profile.environments.get(environment_id)
    raw_environment = raw_environments.get(environment_id)
    if placement is None or not isinstance(raw_environment, Mapping):
        raise LabError("service sandbox environment is unavailable")
    if (
        raw_environment.get("path") != str(placement.path)
        or raw_environment.get("repository") != raw_source.get("repository")
        or raw_environment_identity.get("readiness_environment_sha256")
        != _sha256(raw_environment)
    ):
        raise LabError("service sandbox environment differs from readiness")
    environment_root = _live_canonical_path(
        raw_environment.get("path"),
        label="service environment root",
        directory=True,
    )
    raw_source_roots = raw_environment.get("source_roots")
    if (
        not isinstance(raw_source_roots, list)
        or not raw_source_roots
        or not all(isinstance(value, str) for value in raw_source_roots)
    ):
        raise LabError("service sandbox source roots are invalid")
    source_roots = tuple(
        _live_canonical_path(
            value,
            label="service staged source root",
            allowed_roots=(host_profile.slot_root,),
            directory=True,
        )
        for value in raw_source_roots
    )

    raw_assets = readiness.get("assets")
    spec_assets = spec.get("assets")
    if not isinstance(raw_assets, Mapping) or not isinstance(spec_assets, Mapping):
        raise LabError("service sandbox assets are incomplete")
    if set(raw_assets) != set(spec_assets):
        raise LabError("service sandbox asset role mapping mismatch")
    assets: dict[str, Path] = {}
    for role, raw_identity in sorted(raw_assets.items()):
        expected = spec_assets.get(role)
        if (
            not isinstance(role, str)
            or not isinstance(raw_identity, Mapping)
            or not isinstance(expected, Mapping)
            or raw_identity.get("asset_id") != expected.get("asset_id")
            or raw_identity.get("sha256") != expected.get("sha256")
            or raw_identity.get("identity_attested_by_runtime_receipt") is not True
        ):
            raise LabError(f"service sandbox asset identity mismatch: {role}")
        validate_task_id(role)
        assets[role] = _live_canonical_path(
            raw_identity.get("realpath"),
            label=f"service asset {role}",
            allowed_roots=host_profile.asset_roots,
        )

    requested_runtime = runtime_root.expanduser()
    resolved_runtime = _live_canonical_path(
        str(requested_runtime),
        label="service runtime root",
        allowed_roots=(host_profile.slot_root,),
        directory=True,
    )
    if not requested_runtime.is_absolute() or requested_runtime.absolute() != resolved_runtime:
        raise LabError("service runtime root is not canonical")

    mounts: list[dict[str, object]] = [
        _service_mount(
            kind="environment",
            name=environment_id,
            source=environment_root,
            target="/opt/lab/environment",
            read_only=True,
        )
    ]
    mounts.extend(
        _service_mount(
            kind="source",
            name=str(index),
            source=source_root,
            target=f"/workspace/source/{index}",
            read_only=True,
        )
        for index, source_root in enumerate(source_roots)
    )
    mounts.extend(
        _service_mount(
            kind="asset",
            name=role,
            source=path,
            target=f"/assets/{role}",
            read_only=True,
        )
        for role, path in sorted(assets.items())
    )
    mounts.append(
        _service_mount(
            kind="runtime",
            name="runtime",
            source=resolved_runtime,
            target="/run/lab",
            read_only=False,
        )
    )
    content: dict[str, object] = {
        "schema": SERVICE_SANDBOX_SCHEMA_ID,
        "service_id": spec["service_id"],
        "request_id": spec["request_id"],
        "slot_id": spec["slot_id"],
        "host_id": host_profile.host_id,
        "host_profile_sha256": host_profile.sha256,
        "remote_profile_sha256": spec["remote_profile_sha256"],
        "control_plane_version": control_plane_version,
        "launch_spec_sha256": spec["launch_spec_sha256"],
        "readiness_receipt_sha256": readiness["receipt_sha256"],
        "backend_probe_receipt": probe,
        "backend_probe_max_age_seconds": policy.max_probe_age_seconds,
        "backend_probe_valid_until": (
            observed + timedelta(seconds=policy.max_probe_age_seconds)
        ).isoformat(),
        "mounts": mounts,
        "allowed_gpu_devices": [
            str(device) for device in policy.allowed_gpu_devices
        ],
        "gpu": spec["gpu"],
        "network_egress": "loopback-only",
        "downloads_permitted": False,
        "device_access": False,
        "filesystem_isolated": True,
        "process_group_isolated": True,
        "no_new_privileges": True,
        "created_at": current.isoformat(),
        "service_launch_eligible": True,
        "task_command_eligible": False,
    }
    identity = _sha256(content)
    identified = {**content, "sandbox_id": f"SSBX-{identity[:24]}"}
    receipt = {**identified, "receipt_sha256": _sha256(identified)}
    return validate_service_sandbox_receipt(receipt)


def validate_service_sandbox_receipt(receipt: object) -> dict[str, object]:
    expected = {
        "schema",
        "sandbox_id",
        "service_id",
        "request_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "remote_profile_sha256",
        "control_plane_version",
        "launch_spec_sha256",
        "readiness_receipt_sha256",
        "backend_probe_receipt",
        "backend_probe_max_age_seconds",
        "backend_probe_valid_until",
        "mounts",
        "allowed_gpu_devices",
        "gpu",
        "network_egress",
        "downloads_permitted",
        "device_access",
        "filesystem_isolated",
        "process_group_isolated",
        "no_new_privileges",
        "created_at",
        "service_launch_eligible",
        "task_command_eligible",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("service sandbox receipt has an unexpected shape")
    if receipt.get("schema") != SERVICE_SANDBOX_SCHEMA_ID:
        raise LabError("unsupported service sandbox receipt schema")
    for field in (
        "sandbox_id",
        "service_id",
        "request_id",
        "slot_id",
        "host_id",
        "control_plane_version",
    ):
        value = receipt.get(field)
        if not isinstance(value, str):
            raise LabError(f"service sandbox has invalid {field}")
        validate_task_id(value)
    for field in (
        "host_profile_sha256",
        "remote_profile_sha256",
        "launch_spec_sha256",
        "readiness_receipt_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"service sandbox has invalid {field}")
    probe = validate_sandbox_backend_probe_receipt(
        receipt.get("backend_probe_receipt")
    )
    if (
        probe.get("host_id") != receipt.get("host_id")
        or probe.get("host_profile_sha256") != receipt.get("host_profile_sha256")
        or probe.get("allowed_devices") != receipt.get("allowed_gpu_devices")
    ):
        raise LabError("service sandbox probe identity mismatch")
    max_age = receipt.get("backend_probe_max_age_seconds")
    if (
        not isinstance(max_age, int)
        or isinstance(max_age, bool)
        or not 30 <= max_age <= 600
    ):
        raise LabError("service sandbox has an invalid probe lifetime")
    observed = _timestamp(probe.get("observed_at"), label="sandbox backend probe")
    valid_until = _timestamp(
        receipt.get("backend_probe_valid_until"),
        label="service sandbox probe expiry",
    )
    created = _timestamp(receipt.get("created_at"), label="service sandbox")
    if (
        valid_until != observed + timedelta(seconds=max_age)
        or created < observed - timedelta(seconds=5)
        or created > valid_until
    ):
        raise LabError("service sandbox backend probe lifetime mismatch")
    raw_mounts = receipt.get("mounts")
    if not isinstance(raw_mounts, list) or len(raw_mounts) < 4:
        raise LabError("service sandbox has invalid mounts")
    targets: set[str] = set()
    kinds: list[str] = []
    for mount in raw_mounts:
        if not isinstance(mount, dict) or set(mount) != {
            "kind",
            "name",
            "source",
            "target",
            "read_only",
        }:
            raise LabError("service sandbox has an invalid mount entry")
        kind = mount.get("kind")
        name = mount.get("name")
        source = mount.get("source")
        target = mount.get("target")
        if (
            kind not in {"environment", "source", "asset", "runtime"}
            or not isinstance(name, str)
            or not name
            or not isinstance(source, str)
            or not Path(source).is_absolute()
            or ".." in Path(source).parts
            or not isinstance(target, str)
            or not Path(target).is_absolute()
            or ".." in Path(target).parts
            or not isinstance(mount.get("read_only"), bool)
            or target in targets
        ):
            raise LabError("service sandbox has an unsafe mount entry")
        if kind == "runtime":
            if (
                name != "runtime"
                or target != "/run/lab"
                or mount.get("read_only") is not False
            ):
                raise LabError("service sandbox runtime mount is unsafe")
        elif mount.get("read_only") is not True:
            raise LabError("service sandbox input mounts must be read-only")
        targets.add(target)
        kinds.append(str(kind))
    if kinds.count("environment") != 1 or kinds.count("runtime") != 1:
        raise LabError("service sandbox mount roles are incomplete")
    allowed_devices = receipt.get("allowed_gpu_devices")
    if (
        not isinstance(allowed_devices, list)
        or not allowed_devices
        or len(allowed_devices) != len(set(allowed_devices))
        or not all(
            isinstance(device, str) and _GPU_DEVICE_PATTERN.fullmatch(device)
            for device in allowed_devices
        )
    ):
        raise LabError("service sandbox has invalid GPU devices")
    gpu = receipt.get("gpu")
    if not isinstance(gpu, dict) or set(gpu) != {
        "lease_id",
        "lease_receipt_sha256",
        "gpu_uuid",
    }:
        raise LabError("service sandbox has invalid GPU identity")
    if not SHA256_PATTERN.fullmatch(str(gpu.get("lease_receipt_sha256", ""))):
        raise LabError("service sandbox has invalid GPU lease hash")
    if (
        receipt.get("network_egress") != "loopback-only"
        or receipt.get("downloads_permitted") is not False
        or receipt.get("device_access") is not False
        or receipt.get("filesystem_isolated") is not True
        or receipt.get("process_group_isolated") is not True
        or receipt.get("no_new_privileges") is not True
        or receipt.get("service_launch_eligible") is not True
        or receipt.get("task_command_eligible") is not False
    ):
        raise LabError("service sandbox has an unsafe authority envelope")
    identity_content = {
        key: value
        for key, value in receipt.items()
        if key not in {"sandbox_id", "receipt_sha256"}
    }
    if receipt.get("sandbox_id") != f"SSBX-{_sha256(identity_content)[:24]}":
        raise LabError("service sandbox ID mismatch")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("service sandbox receipt hash mismatch")
    return receipt


def validate_service_sandbox_for_launch(
    receipt: object,
    *,
    launch_spec: object,
    host_profile: HostProfile,
    now: datetime | None = None,
) -> dict[str, object]:
    """Revalidate TTL, identity, paths, and host policy immediately before launch."""

    sandbox = validate_service_sandbox_receipt(receipt)
    spec = validate_service_launch_spec_identity(launch_spec)
    policy = host_profile.sandbox_backend
    if policy is None:
        raise LabError("host Profile lacks a service sandbox backend")
    bindings = {
        "service_id": spec["service_id"],
        "request_id": spec["request_id"],
        "slot_id": spec["slot_id"],
        "host_id": host_profile.host_id,
        "host_profile_sha256": host_profile.sha256,
        "remote_profile_sha256": spec["remote_profile_sha256"],
        "launch_spec_sha256": spec["launch_spec_sha256"],
        "readiness_receipt_sha256": spec["readiness_receipt_sha256"],
        "allowed_gpu_devices": [
            str(device) for device in policy.allowed_gpu_devices
        ],
        "gpu": spec["gpu"],
    }
    for field, expected in bindings.items():
        if sandbox.get(field) != expected:
            raise LabError(f"service sandbox launch binding mismatch: {field}")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("service sandbox launch clock requires a timezone")
    created = _timestamp(sandbox.get("created_at"), label="service sandbox")
    valid_until = _timestamp(
        sandbox.get("backend_probe_valid_until"),
        label="service sandbox probe expiry",
    )
    if current < created - timedelta(seconds=5) or current > valid_until:
        raise LabError("service sandbox backend probe is no longer launch-valid")
    raw_mounts = sandbox["mounts"]
    assert isinstance(raw_mounts, list)
    spec_assets = spec.get("assets")
    assert isinstance(spec_assets, dict)
    seen_assets: set[str] = set()
    for mount in raw_mounts:
        assert isinstance(mount, dict)
        kind = str(mount["kind"])
        name = str(mount["name"])
        source = str(mount["source"])
        if kind == "environment":
            placement = host_profile.environments.get(name)
            if placement is None or source != str(placement.path):
                raise LabError("service sandbox environment mount drifted")
            _live_canonical_path(
                source,
                label="service environment root",
                directory=True,
            )
        elif kind == "source":
            _live_canonical_path(
                source,
                label="service staged source root",
                allowed_roots=(host_profile.slot_root,),
                directory=True,
            )
        elif kind == "asset":
            if name not in spec_assets or name in seen_assets:
                raise LabError("service sandbox asset mount drifted")
            seen_assets.add(name)
            _live_canonical_path(
                source,
                label=f"service asset {name}",
                allowed_roots=host_profile.asset_roots,
            )
        else:
            _live_canonical_path(
                source,
                label="service runtime root",
                allowed_roots=(host_profile.slot_root,),
                directory=True,
            )
    if seen_assets != set(spec_assets):
        raise LabError("service sandbox asset mount set is incomplete")
    return sandbox


def build_execution_sandbox_receipt(
    *,
    launch_spec: object,
    service_receipt: object,
    backend_probe_receipt: object,
    host_profile_sha256: str,
    sandbox_backend_policy: SandboxBackendPolicy,
    control_plane_version: str,
    now: datetime | None = None,
) -> dict[str, object]:
    spec = validate_service_launch_spec_identity(launch_spec)
    service = validate_service_receipt(service_receipt)
    probe = validate_sandbox_backend_probe_receipt(backend_probe_receipt)
    if (
        probe.get("host_profile_sha256") != host_profile_sha256
        or probe.get("backend_id") != sandbox_backend_policy.backend_id
        or probe.get("backend_executable")
        != str(sandbox_backend_policy.executable)
        or probe.get("allowed_devices")
        != [str(device) for device in sandbox_backend_policy.allowed_gpu_devices]
    ):
        raise LabError("sandbox backend probe does not match the host policy")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("sandbox receipt clock requires a timezone")
    observed = _timestamp(probe.get("observed_at"), label="sandbox backend probe")
    age_seconds = (current - observed).total_seconds()
    if age_seconds < -5 or age_seconds > sandbox_backend_policy.max_probe_age_seconds:
        raise LabError("sandbox backend probe is stale or from the future")
    for field, expected in (
        ("request_id", spec.get("request_id")),
        ("task_id", spec.get("task_id")),
        ("slot_id", spec.get("slot_id")),
        ("launch_spec_sha256", spec.get("launch_spec_sha256")),
        ("remote_profile_sha256", spec.get("remote_profile_sha256")),
        ("gpu", spec.get("gpu")),
    ):
        if service.get(field) != expected:
            raise LabError(f"sandbox service identity mismatch: {field}")
    validate_task_id(control_plane_version)
    content: dict[str, object] = {
        "schema": EXECUTION_SANDBOX_SCHEMA_ID,
        "request_id": spec["request_id"],
        "idempotency_key": spec["idempotency_key"],
        "task_id": spec["task_id"],
        "slot_id": spec["slot_id"],
        "host_id": probe["host_id"],
        "host_profile_sha256": probe["host_profile_sha256"],
        "remote_profile_sha256": spec["remote_profile_sha256"],
        "control_plane_version": control_plane_version,
        "launch_spec_sha256": spec["launch_spec_sha256"],
        "service_receipt_sha256": service["receipt_sha256"],
        "gpu": spec["gpu"],
        "backend_probe_receipt": probe,
        "backend_probe_max_age_seconds": (
            sandbox_backend_policy.max_probe_age_seconds
        ),
        "backend_probe_valid_until": (
            observed
            + timedelta(seconds=sandbox_backend_policy.max_probe_age_seconds)
        ).isoformat(),
        "network_egress": "loopback-only",
        "downloads_permitted": False,
        "device_access": False,
        "filesystem_isolated": True,
        "process_group_isolated": True,
        "enforced": True,
        "created_at": current.isoformat(),
        "execution_eligible": True,
    }
    identity = _sha256(content)
    identified = {**content, "sandbox_id": f"SBX-{identity[:24]}"}
    return {**identified, "receipt_sha256": _sha256(identified)}


def validate_execution_sandbox_receipt(receipt: object) -> dict[str, object]:
    expected = {
        "schema",
        "sandbox_id",
        "request_id",
        "idempotency_key",
        "task_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "remote_profile_sha256",
        "control_plane_version",
        "launch_spec_sha256",
        "service_receipt_sha256",
        "gpu",
        "backend_probe_receipt",
        "backend_probe_max_age_seconds",
        "backend_probe_valid_until",
        "network_egress",
        "downloads_permitted",
        "device_access",
        "filesystem_isolated",
        "process_group_isolated",
        "enforced",
        "created_at",
        "execution_eligible",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("execution sandbox receipt has an unexpected shape")
    for field in (
        "sandbox_id",
        "request_id",
        "task_id",
        "slot_id",
        "host_id",
        "control_plane_version",
    ):
        value = receipt.get(field)
        if not isinstance(value, str):
            raise LabError(f"execution sandbox has invalid {field}")
        validate_task_id(value)
    for field in (
        "idempotency_key",
        "host_profile_sha256",
        "remote_profile_sha256",
        "launch_spec_sha256",
        "service_receipt_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"execution sandbox has invalid {field}")
    probe = validate_sandbox_backend_probe_receipt(
        receipt.get("backend_probe_receipt")
    )
    if (
        probe.get("host_id") != receipt.get("host_id")
        or probe.get("host_profile_sha256") != receipt.get("host_profile_sha256")
    ):
        raise LabError("execution sandbox probe identity mismatch")
    max_age = receipt.get("backend_probe_max_age_seconds")
    if (
        not isinstance(max_age, int)
        or isinstance(max_age, bool)
        or not 30 <= max_age <= 600
    ):
        raise LabError("execution sandbox has an invalid backend probe lifetime")
    observed = _timestamp(probe.get("observed_at"), label="sandbox backend probe")
    valid_until = _timestamp(
        receipt.get("backend_probe_valid_until"),
        label="execution sandbox probe expiry",
    )
    created = _timestamp(receipt.get("created_at"), label="execution sandbox")
    if (
        valid_until != observed + timedelta(seconds=max_age)
        or created < observed - timedelta(seconds=5)
        or created > valid_until
    ):
        raise LabError("execution sandbox backend probe lifetime mismatch")
    gpu = receipt.get("gpu")
    if not isinstance(gpu, dict) or set(gpu) != {
        "lease_id",
        "lease_receipt_sha256",
        "gpu_uuid",
    }:
        raise LabError("execution sandbox has invalid GPU identity")
    if not SHA256_PATTERN.fullmatch(str(gpu.get("lease_receipt_sha256", ""))):
        raise LabError("execution sandbox has invalid GPU lease hash")
    if (
        receipt.get("network_egress") != "loopback-only"
        or receipt.get("downloads_permitted") is not False
        or receipt.get("device_access") is not False
        or receipt.get("filesystem_isolated") is not True
        or receipt.get("process_group_isolated") is not True
        or receipt.get("enforced") is not True
        or receipt.get("execution_eligible") is not True
    ):
        raise LabError("execution sandbox has an unsafe authority envelope")
    identity_content = {
        key: value
        for key, value in receipt.items()
        if key not in {"sandbox_id", "receipt_sha256"}
    }
    if receipt.get("sandbox_id") != f"SBX-{_sha256(identity_content)[:24]}":
        raise LabError("execution sandbox ID mismatch")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("execution sandbox receipt hash mismatch")
    return receipt


def validate_execution_sandbox_for_launch(
    receipt: object,
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Require a structurally valid sandbox receipt with a live probe TTL."""

    validated = validate_execution_sandbox_receipt(receipt)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("sandbox launch clock requires a timezone")
    created = _timestamp(validated.get("created_at"), label="execution sandbox")
    valid_until = _timestamp(
        validated.get("backend_probe_valid_until"),
        label="execution sandbox probe expiry",
    )
    if current < created - timedelta(seconds=5) or current > valid_until:
        raise LabError("execution sandbox backend probe is no longer launch-valid")
    return validated


def validate_sandbox_for_attempt(
    receipt: object,
    *,
    request_id: str,
    idempotency_key: str,
    slot_id: str,
    remote_profile_sha256: str,
    control_plane_version: str,
    launch_spec_sha256: str,
    service_receipt_sha256: str,
    gpu: object,
) -> dict[str, object]:
    sandbox = validate_execution_sandbox_receipt(receipt)
    bindings = {
        "request_id": request_id,
        "idempotency_key": idempotency_key,
        "slot_id": slot_id,
        "remote_profile_sha256": remote_profile_sha256,
        "control_plane_version": control_plane_version,
        "launch_spec_sha256": launch_spec_sha256,
        "service_receipt_sha256": service_receipt_sha256,
        "gpu": gpu,
    }
    for field, expected in bindings.items():
        if sandbox.get(field) != expected:
            raise LabError(f"execution sandbox attempt binding mismatch: {field}")
    return sandbox
