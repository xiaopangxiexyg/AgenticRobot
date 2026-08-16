from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .endpoint_lease import validate_endpoint_lease_receipt
from .remote_preflight import HostProfile
from .remote_request import SHA256_PATTERN, _sha256
from .workspace import LabError, validate_task_id

NETWORK_NAMESPACE_PLAN_SCHEMA_ID = "lab.network-namespace-plan.v1"
NETWORK_NAMESPACE_ANCHOR_SCHEMA_ID = "lab.network-namespace-anchor-observation.v1"
NETWORK_NAMESPACE_MEMBER_SCHEMA_ID = "lab.network-namespace-member-observation.v1"
NETWORK_NAMESPACE_PROBE_SCHEMA_ID = "lab.network-namespace-probe.v1"
_TOPOLOGY = "shared-isolated-loopback"
_REQUIRED_MEMBERS = ("attempt", "service")
_MAX_EXECUTABLE_BYTES = 64 * 1024 * 1024
_MAX_PROBE_OUTPUT_BYTES = 64 * 1024
_PROBE_TIMEOUT_SECONDS = 5
_PROBE_SCRIPT = r'''
import json
import os
import socket
import sys

expected_inode = int(sys.argv[1])
namespace_inode = os.stat("/proc/self/ns/net").st_ino
interfaces = sorted(name for _index, name in socket.if_nameindex())
loopback_available = False
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        loopback_available = probe.getsockname()[0] == "127.0.0.1"
except OSError:
    pass

route_interfaces = []
for path in ("/proc/net/route", "/proc/net/ipv6_route"):
    try:
        lines = open(path, encoding="ascii").read().splitlines()
    except OSError:
        continue
    for line in lines:
        fields = line.split()
        if not fields or fields[0] in {"Iface", "Kernel"}:
            continue
        interface = fields[0] if path.endswith("/route") else fields[-1]
        route_interfaces.append(interface)

print(json.dumps({
    "external_routes_absent": all(name == "lo" for name in route_interfaces),
    "loopback_available": loopback_available,
    "network_namespace_inode": namespace_inode,
    "only_loopback_interfaces": interfaces == ["lo"],
    "requested_namespace_matched": namespace_inode == expected_inode,
}, sort_keys=True))
'''


@dataclass(frozen=True)
class NetworkNamespaceProbePolicy:
    """Installed control-plane executables for a fixed in-netns probe."""

    policy_id: str
    enter_executable: Path
    python_executable: Path
    max_probe_age_seconds: int = 300


def _read_text(path: Path, *, label: str, max_bytes: int = 64 * 1024) -> str:
    try:
        metadata = path.stat(follow_symlinks=True)
        if not stat.S_ISREG(metadata.st_mode) or not 0 <= metadata.st_size <= max_bytes:
            raise LabError(f"{label} is not a bounded regular file")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LabError(f"cannot read {label}") from exc


def _process_stat(path: Path, *, expected_pid: int) -> dict[str, int]:
    raw = _read_text(path, label="network namespace process stat")
    close = raw.rfind(")")
    if close < 2 or not raw.startswith(f"{expected_pid} ("):
        raise LabError("network namespace process stat has an invalid identity")
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        raise LabError("network namespace process stat is incomplete")
    try:
        return {
            "parent_pid": int(fields[1]),
            "process_group_id": int(fields[2]),
            "session_id": int(fields[3]),
            "proc_start_time_ticks": int(fields[19]),
        }
    except ValueError as exc:
        raise LabError("network namespace process stat is invalid") from exc


def _no_new_privileges(path: Path) -> bool:
    status = _read_text(path, label="network namespace process status")
    values = {
        key: value.strip()
        for line in status.splitlines()
        if ":" in line
        for key, value in (line.split(":", 1),)
    }
    return values.get("NoNewPrivs") == "1"


def _proc_root(path: Path) -> Path:
    requested = path.expanduser()
    resolved = requested.resolve()
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or requested.absolute() != resolved
        or not resolved.is_dir()
    ):
        raise LabError("network namespace proc root is unsafe")
    return resolved


def _namespace_inode(proc_root: Path, *, pid: int) -> int:
    try:
        inode = (proc_root / str(pid) / "ns" / "net").stat().st_ino
    except OSError as exc:
        raise LabError("cannot inspect process network namespace") from exc
    if not isinstance(inode, int) or inode <= 0:
        raise LabError("process network namespace inode is invalid")
    return inode


def _open_protected_executable(path: Path, *, label: str) -> tuple[int, str, os.stat_result]:
    if not path.is_absolute() or path.is_symlink() or ".." in path.parts:
        raise LabError(f"{label} path is unsafe")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LabError(f"cannot open {label}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not metadata.st_mode & 0o111
            or not 0 < metadata.st_size <= _MAX_EXECUTABLE_BYTES
        ):
            raise LabError(f"{label} is not a bounded executable file")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, digest.hexdigest(), metadata
    except Exception:
        os.close(descriptor)
        raise


def _validate_probe_policy(policy: NetworkNamespaceProbePolicy) -> None:
    validate_task_id(policy.policy_id)
    for path, label in (
        (policy.enter_executable, "network namespace enter executable"),
        (policy.python_executable, "network namespace probe Python"),
    ):
        if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
            raise LabError(f"{label} policy path is unsafe")
    if (
        not isinstance(policy.max_probe_age_seconds, int)
        or isinstance(policy.max_probe_age_seconds, bool)
        or not 30 <= policy.max_probe_age_seconds <= 600
    ):
        raise LabError("network namespace probe lifetime is outside policy")


def _probe_policy_identity(
    policy: NetworkNamespaceProbePolicy,
    *,
    enter_sha256: str,
    python_sha256: str,
) -> dict[str, object]:
    return {
        "policy_id": policy.policy_id,
        "enter_executable": str(policy.enter_executable),
        "enter_executable_sha256": enter_sha256,
        "python_executable": str(policy.python_executable),
        "python_executable_sha256": python_sha256,
        "max_probe_age_seconds": policy.max_probe_age_seconds,
    }


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


def build_network_namespace_plan(
    endpoint_lease_receipt: object,
    *,
    host_profile: HostProfile,
    now: datetime | None = None,
) -> dict[str, object]:
    """Describe the shared network boundary required by one request.

    This is deliberately a plan, not runtime evidence.  It neither creates a
    namespace nor authorizes a launcher.  A later trusted namespace anchor must
    attest that both the service and its attempt joined the same isolated
    loopback namespace before either process may be considered executable.
    """

    lease = validate_endpoint_lease_receipt(
        endpoint_lease_receipt,
        host_id=host_profile.host_id,
        host_profile_sha256=host_profile.sha256,
        now=now,
    )
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("network namespace plan clock requires a timezone")
    content: dict[str, object] = {
        "schema": NETWORK_NAMESPACE_PLAN_SCHEMA_ID,
        "request_id": lease["request_id"],
        "slot_id": lease["slot_id"],
        "host_id": lease["host_id"],
        "host_profile_sha256": lease["host_profile_sha256"],
        "endpoint_lease_receipt_sha256": lease["receipt_sha256"],
        "topology": _TOPOLOGY,
        "required_members": list(_REQUIRED_MEMBERS),
        "endpoint": {
            "scheme": lease["scheme"],
            "host": lease["host"],
            "port": lease["port"],
        },
        "loopback_required": True,
        "external_network_permitted": False,
        "namespace_anchor_required": True,
        "service_join_required": True,
        "attempt_join_required": True,
        "namespace_created": False,
        "loopback_ready": False,
        "service_joined": False,
        "attempt_joined": False,
        "launch_authorized": False,
        "device_access": False,
        # Tie the inert plan identity to the already persisted lease epoch so
        # retries produce the same plan instead of forking request state.
        "created_at": lease["acquired_at"],
    }
    identity = _sha256(content)
    identified = {**content, "namespace_id": f"NET-{identity[:24]}"}
    receipt = {**identified, "receipt_sha256": _sha256(identified)}
    return validate_network_namespace_plan(
        receipt,
        endpoint_lease_receipt=lease,
        host_profile=host_profile,
        now=current,
    )


def validate_network_namespace_plan(
    plan: object,
    *,
    endpoint_lease_receipt: object | None = None,
    host_profile: HostProfile | None = None,
    require_active: bool = True,
    now: datetime | None = None,
) -> dict[str, object]:
    if not isinstance(require_active, bool):
        raise LabError("network namespace plan active policy must be boolean")
    expected = {
        "schema",
        "namespace_id",
        "request_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "endpoint_lease_receipt_sha256",
        "topology",
        "required_members",
        "endpoint",
        "loopback_required",
        "external_network_permitted",
        "namespace_anchor_required",
        "service_join_required",
        "attempt_join_required",
        "namespace_created",
        "loopback_ready",
        "service_joined",
        "attempt_joined",
        "launch_authorized",
        "device_access",
        "created_at",
        "receipt_sha256",
    }
    if not isinstance(plan, dict) or set(plan) != expected:
        raise LabError("network namespace plan has an unexpected shape")
    if plan.get("schema") != NETWORK_NAMESPACE_PLAN_SCHEMA_ID:
        raise LabError("unsupported network namespace plan schema")
    for field in ("namespace_id", "request_id", "slot_id", "host_id"):
        value = plan.get(field)
        if not isinstance(value, str):
            raise LabError(f"network namespace plan has invalid {field}")
        validate_task_id(value)
    for field in (
        "host_profile_sha256",
        "endpoint_lease_receipt_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(plan.get(field, ""))):
            raise LabError(f"network namespace plan has invalid {field}")
    endpoint = plan.get("endpoint")
    if (
        not isinstance(endpoint, dict)
        or set(endpoint) != {"scheme", "host", "port"}
        or endpoint.get("scheme") not in {"http", "https", "ws", "wss"}
        or endpoint.get("host") != "127.0.0.1"
        or not isinstance(endpoint.get("port"), int)
        or isinstance(endpoint.get("port"), bool)
        or not 1024 <= endpoint["port"] <= 65535
    ):
        raise LabError("network namespace plan has an unsafe endpoint")
    if (
        plan.get("topology") != _TOPOLOGY
        or plan.get("required_members") != list(_REQUIRED_MEMBERS)
        or plan.get("loopback_required") is not True
        or plan.get("external_network_permitted") is not False
        or plan.get("namespace_anchor_required") is not True
        or plan.get("service_join_required") is not True
        or plan.get("attempt_join_required") is not True
        or plan.get("namespace_created") is not False
        or plan.get("loopback_ready") is not False
        or plan.get("service_joined") is not False
        or plan.get("attempt_joined") is not False
        or plan.get("launch_authorized") is not False
        or plan.get("device_access") is not False
    ):
        raise LabError("network namespace plan has an unsafe authority envelope")
    created = _timestamp(plan.get("created_at"), label="network namespace plan")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("network namespace plan validation clock requires a timezone")
    if created > current:
        raise LabError("network namespace plan was created in the future")
    identity_content = {
        key: value
        for key, value in plan.items()
        if key not in {"namespace_id", "receipt_sha256"}
    }
    if plan.get("namespace_id") != f"NET-{_sha256(identity_content)[:24]}":
        raise LabError("network namespace plan ID mismatch")
    content = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    if plan.get("receipt_sha256") != _sha256(content):
        raise LabError("network namespace plan receipt hash mismatch")

    if endpoint_lease_receipt is not None:
        lease = validate_endpoint_lease_receipt(
            endpoint_lease_receipt,
            host_id=host_profile.host_id if host_profile is not None else None,
            host_profile_sha256=(
                host_profile.sha256 if host_profile is not None else None
            ),
            request_id=str(plan["request_id"]),
            slot_id=str(plan["slot_id"]),
            require_active=require_active,
            now=current,
        )
        bindings = {
            "host_id": lease["host_id"],
            "host_profile_sha256": lease["host_profile_sha256"],
            "endpoint_lease_receipt_sha256": lease["receipt_sha256"],
            "endpoint": {
                "scheme": lease["scheme"],
                "host": lease["host"],
                "port": lease["port"],
            },
        }
        for field, expected_value in bindings.items():
            if plan.get(field) != expected_value:
                raise LabError(
                    f"network namespace plan binding mismatch: {field}"
                )
    elif host_profile is not None:
        if (
            plan.get("host_id") != host_profile.host_id
            or plan.get("host_profile_sha256") != host_profile.sha256
        ):
            raise LabError("network namespace plan host Profile mismatch")
    return plan


def collect_network_namespace_anchor_observation(
    network_namespace_plan: object,
    *,
    host_profile: HostProfile,
    pid: int,
    proc_root: Path = Path("/proc"),
    collector_pid: int | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Observe a direct-child namespace anchor without granting authority.

    The observation proves only stable process and namespace identity.  It does
    not prove that loopback is configured or that external routes are absent;
    those claims remain false until a later in-namespace trusted probe exists.
    """

    plan = validate_network_namespace_plan(
        network_namespace_plan,
        host_profile=host_profile,
        now=now,
    )
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise LabError("network namespace anchor PID is invalid")
    collector = os.getpid() if collector_pid is None else collector_pid
    if not isinstance(collector, int) or isinstance(collector, bool) or collector <= 0:
        raise LabError("network namespace collector PID is invalid")
    root = _proc_root(proc_root)
    process_root = root / str(pid)
    before = _process_stat(process_root / "stat", expected_pid=pid)
    if (
        before["parent_pid"] != collector
        or before["process_group_id"] != pid
        or before["session_id"] != pid
        or before["proc_start_time_ticks"] <= 0
    ):
        raise LabError("network namespace anchor is not a direct isolated session")
    if not _no_new_privileges(process_root / "status"):
        raise LabError("network namespace anchor did not enable NoNewPrivs")
    collector_inode = _namespace_inode(root, pid=collector)
    anchor_inode = _namespace_inode(root, pid=pid)
    if collector_inode == anchor_inode:
        raise LabError("network namespace anchor did not isolate its namespace")
    after = _process_stat(process_root / "stat", expected_pid=pid)
    if (
        after != before
        or _namespace_inode(root, pid=collector) != collector_inode
        or _namespace_inode(root, pid=pid) != anchor_inode
    ):
        raise LabError("network namespace anchor identity changed while observed")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("network namespace anchor clock requires a timezone")
    content: dict[str, object] = {
        "schema": NETWORK_NAMESPACE_ANCHOR_SCHEMA_ID,
        "namespace_id": plan["namespace_id"],
        "network_namespace_plan_sha256": plan["receipt_sha256"],
        "request_id": plan["request_id"],
        "slot_id": plan["slot_id"],
        "host_id": plan["host_id"],
        "host_profile_sha256": plan["host_profile_sha256"],
        "pid": pid,
        "collector_pid": collector,
        **before,
        "collector_network_namespace_inode": collector_inode,
        "anchor_network_namespace_inode": anchor_inode,
        "network_namespace_isolated": True,
        "managed_direct_child": True,
        "process_group_isolated": True,
        "no_new_privileges": True,
        "alive_at_observation": True,
        "loopback_verified": False,
        "external_network_absence_verified": False,
        "membership_eligible": False,
        "launch_authorized": False,
        "device_access": False,
        "observed_at": current.isoformat(),
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_network_namespace_anchor_observation(
        receipt,
        network_namespace_plan=plan,
        host_profile=host_profile,
    )


def validate_network_namespace_anchor_observation(
    observation: object,
    *,
    network_namespace_plan: object | None = None,
    host_profile: HostProfile | None = None,
) -> dict[str, object]:
    expected = {
        "schema",
        "namespace_id",
        "network_namespace_plan_sha256",
        "request_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "pid",
        "collector_pid",
        "parent_pid",
        "process_group_id",
        "session_id",
        "proc_start_time_ticks",
        "collector_network_namespace_inode",
        "anchor_network_namespace_inode",
        "network_namespace_isolated",
        "managed_direct_child",
        "process_group_isolated",
        "no_new_privileges",
        "alive_at_observation",
        "loopback_verified",
        "external_network_absence_verified",
        "membership_eligible",
        "launch_authorized",
        "device_access",
        "observed_at",
        "receipt_sha256",
    }
    if not isinstance(observation, dict) or set(observation) != expected:
        raise LabError("network namespace anchor observation has an unexpected shape")
    if observation.get("schema") != NETWORK_NAMESPACE_ANCHOR_SCHEMA_ID:
        raise LabError("unsupported network namespace anchor observation schema")
    for field in ("namespace_id", "request_id", "slot_id", "host_id"):
        value = observation.get(field)
        if not isinstance(value, str):
            raise LabError(f"network namespace anchor has invalid {field}")
        validate_task_id(value)
    for field in (
        "network_namespace_plan_sha256",
        "host_profile_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(observation.get(field, ""))):
            raise LabError(f"network namespace anchor has invalid {field}")
    pid = observation.get("pid")
    collector = observation.get("collector_pid")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or not isinstance(collector, int)
        or isinstance(collector, bool)
        or collector <= 0
        or observation.get("parent_pid") != collector
        or observation.get("process_group_id") != pid
        or observation.get("session_id") != pid
        or not isinstance(observation.get("proc_start_time_ticks"), int)
        or observation["proc_start_time_ticks"] <= 0
    ):
        raise LabError("network namespace anchor has invalid process identity")
    collector_inode = observation.get("collector_network_namespace_inode")
    anchor_inode = observation.get("anchor_network_namespace_inode")
    if (
        not isinstance(collector_inode, int)
        or isinstance(collector_inode, bool)
        or collector_inode <= 0
        or not isinstance(anchor_inode, int)
        or isinstance(anchor_inode, bool)
        or anchor_inode <= 0
        or collector_inode == anchor_inode
    ):
        raise LabError("network namespace anchor has invalid namespace identity")
    if (
        observation.get("network_namespace_isolated") is not True
        or observation.get("managed_direct_child") is not True
        or observation.get("process_group_isolated") is not True
        or observation.get("no_new_privileges") is not True
        or observation.get("alive_at_observation") is not True
        or observation.get("loopback_verified") is not False
        or observation.get("external_network_absence_verified") is not False
        or observation.get("membership_eligible") is not False
        or observation.get("launch_authorized") is not False
        or observation.get("device_access") is not False
    ):
        raise LabError("network namespace anchor has an unsafe authority envelope")
    _timestamp(
        observation.get("observed_at"),
        label="network namespace anchor observation",
    )
    content = {
        key: value for key, value in observation.items() if key != "receipt_sha256"
    }
    if observation.get("receipt_sha256") != _sha256(content):
        raise LabError("network namespace anchor observation hash mismatch")
    if network_namespace_plan is not None:
        plan = validate_network_namespace_plan(
            network_namespace_plan,
            host_profile=host_profile,
        )
        bindings = {
            "namespace_id": plan["namespace_id"],
            "network_namespace_plan_sha256": plan["receipt_sha256"],
            "request_id": plan["request_id"],
            "slot_id": plan["slot_id"],
            "host_id": plan["host_id"],
            "host_profile_sha256": plan["host_profile_sha256"],
        }
        for field, expected_value in bindings.items():
            if observation.get(field) != expected_value:
                raise LabError(
                    f"network namespace anchor binding mismatch: {field}"
                )
    elif host_profile is not None and (
        observation.get("host_id") != host_profile.host_id
        or observation.get("host_profile_sha256") != host_profile.sha256
    ):
        raise LabError("network namespace anchor host Profile mismatch")
    return observation


def collect_network_namespace_member_observation(
    network_namespace_plan: object,
    *,
    anchor_observation: object,
    role: str,
    pid: int,
    proc_root: Path = Path("/proc"),
    now: datetime | None = None,
) -> dict[str, object]:
    """Observe one process in the anchor netns without authorizing execution."""

    plan = validate_network_namespace_plan(network_namespace_plan)
    anchor = validate_network_namespace_anchor_observation(
        anchor_observation,
        network_namespace_plan=plan,
    )
    if role not in _REQUIRED_MEMBERS:
        raise LabError(f"unsupported network namespace member role: {role}")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or pid == anchor.get("pid")
    ):
        raise LabError("network namespace member PID is invalid")
    root = _proc_root(proc_root)
    process_root = root / str(pid)
    before = _process_stat(process_root / "stat", expected_pid=pid)
    if before["proc_start_time_ticks"] <= 0:
        raise LabError("network namespace member process identity is invalid")
    if not _no_new_privileges(process_root / "status"):
        raise LabError("network namespace member did not enable NoNewPrivs")
    member_inode = _namespace_inode(root, pid=pid)
    if member_inode != anchor.get("anchor_network_namespace_inode"):
        raise LabError("network namespace member did not join the shared namespace")
    after = _process_stat(process_root / "stat", expected_pid=pid)
    if (
        after != before
        or _namespace_inode(root, pid=pid) != member_inode
    ):
        raise LabError("network namespace member identity changed while observed")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("network namespace member clock requires a timezone")
    content: dict[str, object] = {
        "schema": NETWORK_NAMESPACE_MEMBER_SCHEMA_ID,
        "namespace_id": plan["namespace_id"],
        "network_namespace_plan_sha256": plan["receipt_sha256"],
        "anchor_observation_sha256": anchor["receipt_sha256"],
        "request_id": plan["request_id"],
        "slot_id": plan["slot_id"],
        "host_id": plan["host_id"],
        "host_profile_sha256": plan["host_profile_sha256"],
        "role": role,
        "pid": pid,
        **before,
        "network_namespace_inode": member_inode,
        "same_as_anchor": True,
        "no_new_privileges": True,
        "alive_at_observation": True,
        "loopback_verified": False,
        "external_network_absence_verified": False,
        "connectivity_authorized": False,
        "launch_authorized": False,
        "device_access": False,
        "observed_at": current.isoformat(),
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_network_namespace_member_observation(
        receipt,
        network_namespace_plan=plan,
        anchor_observation=anchor,
        expected_role=role,
    )


def validate_network_namespace_member_observation(
    observation: object,
    *,
    network_namespace_plan: object | None = None,
    anchor_observation: object | None = None,
    expected_role: str | None = None,
) -> dict[str, object]:
    expected = {
        "schema",
        "namespace_id",
        "network_namespace_plan_sha256",
        "anchor_observation_sha256",
        "request_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "role",
        "pid",
        "parent_pid",
        "process_group_id",
        "session_id",
        "proc_start_time_ticks",
        "network_namespace_inode",
        "same_as_anchor",
        "no_new_privileges",
        "alive_at_observation",
        "loopback_verified",
        "external_network_absence_verified",
        "connectivity_authorized",
        "launch_authorized",
        "device_access",
        "observed_at",
        "receipt_sha256",
    }
    if not isinstance(observation, dict) or set(observation) != expected:
        raise LabError("network namespace member observation has an unexpected shape")
    if observation.get("schema") != NETWORK_NAMESPACE_MEMBER_SCHEMA_ID:
        raise LabError("unsupported network namespace member observation schema")
    for field in ("namespace_id", "request_id", "slot_id", "host_id"):
        value = observation.get(field)
        if not isinstance(value, str):
            raise LabError(f"network namespace member has invalid {field}")
        validate_task_id(value)
    for field in (
        "network_namespace_plan_sha256",
        "anchor_observation_sha256",
        "host_profile_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(observation.get(field, ""))):
            raise LabError(f"network namespace member has invalid {field}")
    role = observation.get("role")
    if role not in _REQUIRED_MEMBERS or (
        expected_role is not None and role != expected_role
    ):
        raise LabError("network namespace member has invalid role")
    pid = observation.get("pid")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or not all(
            isinstance(observation.get(field), int)
            and not isinstance(observation.get(field), bool)
            and observation[field] > 0
            for field in (
                "parent_pid",
                "process_group_id",
                "session_id",
                "proc_start_time_ticks",
                "network_namespace_inode",
            )
        )
    ):
        raise LabError("network namespace member has invalid process identity")
    if (
        observation.get("same_as_anchor") is not True
        or observation.get("no_new_privileges") is not True
        or observation.get("alive_at_observation") is not True
        or observation.get("loopback_verified") is not False
        or observation.get("external_network_absence_verified") is not False
        or observation.get("connectivity_authorized") is not False
        or observation.get("launch_authorized") is not False
        or observation.get("device_access") is not False
    ):
        raise LabError("network namespace member has an unsafe authority envelope")
    _timestamp(
        observation.get("observed_at"),
        label="network namespace member observation",
    )
    content = {
        key: value for key, value in observation.items() if key != "receipt_sha256"
    }
    if observation.get("receipt_sha256") != _sha256(content):
        raise LabError("network namespace member observation hash mismatch")
    plan: dict[str, object] | None = None
    if network_namespace_plan is not None:
        plan = validate_network_namespace_plan(network_namespace_plan)
        bindings = {
            "namespace_id": plan["namespace_id"],
            "network_namespace_plan_sha256": plan["receipt_sha256"],
            "request_id": plan["request_id"],
            "slot_id": plan["slot_id"],
            "host_id": plan["host_id"],
            "host_profile_sha256": plan["host_profile_sha256"],
        }
        for field, expected_value in bindings.items():
            if observation.get(field) != expected_value:
                raise LabError(
                    f"network namespace member binding mismatch: {field}"
                )
    if anchor_observation is not None:
        anchor = validate_network_namespace_anchor_observation(
            anchor_observation,
            network_namespace_plan=plan,
        )
        if (
            observation.get("anchor_observation_sha256")
            != anchor.get("receipt_sha256")
            or observation.get("namespace_id") != anchor.get("namespace_id")
            or observation.get("network_namespace_inode")
            != anchor.get("anchor_network_namespace_inode")
            or observation.get("pid") == anchor.get("pid")
        ):
            raise LabError("network namespace member anchor binding mismatch")
    return observation


def collect_network_namespace_probe(
    network_namespace_plan: object,
    *,
    anchor_observation: object,
    host_profile: HostProfile,
    policy: NetworkNamespaceProbePolicy,
    proc_root: Path = Path("/proc"),
    now: datetime | None = None,
) -> dict[str, object]:
    """Run a fixed, protected probe inside the observed anchor netns.

    The only dynamic argument is the already observed namespace inode.  No Task
    argv, cwd, environment, source path, endpoint, or network destination is
    accepted.  The probe binds an ephemeral loopback socket and inspects local
    interfaces/routes; it performs no external connection.
    """

    plan = validate_network_namespace_plan(
        network_namespace_plan,
        host_profile=host_profile,
        now=now,
    )
    anchor = validate_network_namespace_anchor_observation(
        anchor_observation,
        network_namespace_plan=plan,
        host_profile=host_profile,
    )
    _validate_probe_policy(policy)
    root = _proc_root(proc_root)
    raw_anchor_pid = anchor["pid"]
    assert isinstance(raw_anchor_pid, int) and not isinstance(raw_anchor_pid, bool)
    anchor_pid = raw_anchor_pid
    anchor_root = root / str(anchor_pid)
    before = _process_stat(anchor_root / "stat", expected_pid=anchor_pid)
    expected_process = {
        "parent_pid": anchor["parent_pid"],
        "process_group_id": anchor["process_group_id"],
        "session_id": anchor["session_id"],
        "proc_start_time_ticks": anchor["proc_start_time_ticks"],
    }
    if before != expected_process or not _no_new_privileges(anchor_root / "status"):
        raise LabError("network namespace anchor is no longer probe-eligible")
    anchor_inode = _namespace_inode(root, pid=anchor_pid)
    if anchor_inode != anchor.get("anchor_network_namespace_inode"):
        raise LabError("network namespace anchor inode changed before probe")

    enter_fd, enter_sha256, enter_before = _open_protected_executable(
        policy.enter_executable,
        label="network namespace enter executable",
    )
    try:
        python_fd, python_sha256, python_before = _open_protected_executable(
            policy.python_executable,
            label="network namespace probe Python",
        )
    except Exception:
        os.close(enter_fd)
        raise
    argv = [
        str(policy.enter_executable),
        f"--net={anchor_root / 'ns' / 'net'}",
        "--",
        f"/proc/self/fd/{python_fd}",
        "-I",
        "-c",
        _PROBE_SCRIPT,
        str(anchor_inode),
    ]
    try:
        try:
            completed = subprocess.run(
                argv,
                executable=f"/proc/self/fd/{enter_fd}",
                pass_fds=(enter_fd, python_fd),
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=_PROBE_TIMEOUT_SECONDS,
                cwd="/",
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LabError("network namespace probe could not complete") from exc
        enter_after = os.fstat(enter_fd)
        python_after = os.fstat(python_fd)
        for label, previous, current_metadata in (
            ("enter executable", enter_before, enter_after),
            ("probe Python", python_before, python_after),
        ):
            if (
                previous.st_dev != current_metadata.st_dev
                or previous.st_ino != current_metadata.st_ino
                or previous.st_size != current_metadata.st_size
                or previous.st_mtime_ns != current_metadata.st_mtime_ns
            ):
                raise LabError(
                    f"network namespace {label} changed while it was used"
                )
    finally:
        os.close(python_fd)
        os.close(enter_fd)
    if (
        len(completed.stdout) > _MAX_PROBE_OUTPUT_BYTES
        or len(completed.stderr) > _MAX_PROBE_OUTPUT_BYTES
    ):
        raise LabError("network namespace probe output exceeded the protected limit")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LabError(
            f"network namespace probe failed: {detail or completed.returncode}"
        )
    try:
        facts = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabError("network namespace probe returned invalid JSON") from exc
    if not isinstance(facts, dict) or set(facts) != {
        "external_routes_absent",
        "loopback_available",
        "network_namespace_inode",
        "only_loopback_interfaces",
        "requested_namespace_matched",
    }:
        raise LabError("network namespace probe returned unexpected facts")
    if (
        facts.get("network_namespace_inode") != anchor_inode
        or facts.get("requested_namespace_matched") is not True
        or facts.get("loopback_available") is not True
        or facts.get("only_loopback_interfaces") is not True
        or facts.get("external_routes_absent") is not True
    ):
        raise LabError("network namespace probe did not prove loopback isolation")
    after = _process_stat(anchor_root / "stat", expected_pid=anchor_pid)
    if (
        after != before
        or _namespace_inode(root, pid=anchor_pid) != anchor_inode
        or not _no_new_privileges(anchor_root / "status")
    ):
        raise LabError("network namespace anchor changed while it was probed")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("network namespace probe clock requires a timezone")
    probe_policy = _probe_policy_identity(
        policy,
        enter_sha256=enter_sha256,
        python_sha256=python_sha256,
    )
    content: dict[str, object] = {
        "schema": NETWORK_NAMESPACE_PROBE_SCHEMA_ID,
        "namespace_id": plan["namespace_id"],
        "network_namespace_plan_sha256": plan["receipt_sha256"],
        "anchor_observation_sha256": anchor["receipt_sha256"],
        "request_id": plan["request_id"],
        "slot_id": plan["slot_id"],
        "host_id": plan["host_id"],
        "host_profile_sha256": plan["host_profile_sha256"],
        "probe_policy": probe_policy,
        "probe_policy_sha256": _sha256(probe_policy),
        "network_namespace_inode": anchor_inode,
        "loopback_verified": True,
        "only_loopback_interfaces": True,
        "external_network_absence_verified": True,
        "network_ready": True,
        "launch_authorized": False,
        "device_access": False,
        "observed_at": current.isoformat(),
        "valid_until": (
            current + timedelta(seconds=policy.max_probe_age_seconds)
        ).isoformat(),
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_network_namespace_probe_receipt(
        receipt,
        network_namespace_plan=plan,
        anchor_observation=anchor,
        policy=policy,
    )


def validate_network_namespace_probe_receipt(
    receipt: object,
    *,
    network_namespace_plan: object | None = None,
    anchor_observation: object | None = None,
    policy: NetworkNamespaceProbePolicy | None = None,
) -> dict[str, object]:
    expected = {
        "schema",
        "namespace_id",
        "network_namespace_plan_sha256",
        "anchor_observation_sha256",
        "request_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "probe_policy",
        "probe_policy_sha256",
        "network_namespace_inode",
        "loopback_verified",
        "only_loopback_interfaces",
        "external_network_absence_verified",
        "network_ready",
        "launch_authorized",
        "device_access",
        "observed_at",
        "valid_until",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("network namespace probe receipt has an unexpected shape")
    if receipt.get("schema") != NETWORK_NAMESPACE_PROBE_SCHEMA_ID:
        raise LabError("unsupported network namespace probe receipt schema")
    for field in ("namespace_id", "request_id", "slot_id", "host_id"):
        value = receipt.get(field)
        if not isinstance(value, str):
            raise LabError(f"network namespace probe has invalid {field}")
        validate_task_id(value)
    for field in (
        "network_namespace_plan_sha256",
        "anchor_observation_sha256",
        "host_profile_sha256",
        "probe_policy_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"network namespace probe has invalid {field}")
    raw_policy = receipt.get("probe_policy")
    if not isinstance(raw_policy, dict) or set(raw_policy) != {
        "policy_id",
        "enter_executable",
        "enter_executable_sha256",
        "python_executable",
        "python_executable_sha256",
        "max_probe_age_seconds",
    }:
        raise LabError("network namespace probe has invalid policy identity")
    policy_id = raw_policy.get("policy_id")
    if not isinstance(policy_id, str):
        raise LabError("network namespace probe has invalid policy identity")
    validate_task_id(policy_id)
    for field in ("enter_executable", "python_executable"):
        value = raw_policy.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise LabError("network namespace probe has invalid policy path")
    for field in ("enter_executable_sha256", "python_executable_sha256"):
        if not SHA256_PATTERN.fullmatch(str(raw_policy.get(field, ""))):
            raise LabError("network namespace probe has invalid executable identity")
    max_age = raw_policy.get("max_probe_age_seconds")
    if (
        not isinstance(max_age, int)
        or isinstance(max_age, bool)
        or not 30 <= max_age <= 600
        or receipt.get("probe_policy_sha256") != _sha256(raw_policy)
    ):
        raise LabError("network namespace probe has invalid policy identity")
    inode = receipt.get("network_namespace_inode")
    if not isinstance(inode, int) or isinstance(inode, bool) or inode <= 0:
        raise LabError("network namespace probe has invalid namespace identity")
    if (
        receipt.get("loopback_verified") is not True
        or receipt.get("only_loopback_interfaces") is not True
        or receipt.get("external_network_absence_verified") is not True
        or receipt.get("network_ready") is not True
        or receipt.get("launch_authorized") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("network namespace probe has an unsafe authority envelope")
    observed = _timestamp(receipt.get("observed_at"), label="network namespace probe")
    valid_until = _timestamp(
        receipt.get("valid_until"), label="network namespace probe expiry"
    )
    if valid_until != observed + timedelta(seconds=max_age):
        raise LabError("network namespace probe has an invalid validity window")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("network namespace probe receipt hash mismatch")
    plan: dict[str, object] | None = None
    if network_namespace_plan is not None:
        plan = validate_network_namespace_plan(network_namespace_plan)
        bindings = {
            "namespace_id": plan["namespace_id"],
            "network_namespace_plan_sha256": plan["receipt_sha256"],
            "request_id": plan["request_id"],
            "slot_id": plan["slot_id"],
            "host_id": plan["host_id"],
            "host_profile_sha256": plan["host_profile_sha256"],
        }
        for field, expected_value in bindings.items():
            if receipt.get(field) != expected_value:
                raise LabError(f"network namespace probe binding mismatch: {field}")
    if anchor_observation is not None:
        anchor = validate_network_namespace_anchor_observation(
            anchor_observation,
            network_namespace_plan=plan,
        )
        if (
            receipt.get("anchor_observation_sha256")
            != anchor.get("receipt_sha256")
            or receipt.get("network_namespace_inode")
            != anchor.get("anchor_network_namespace_inode")
        ):
            raise LabError("network namespace probe anchor binding mismatch")
    if policy is not None:
        _validate_probe_policy(policy)
        if (
            raw_policy.get("policy_id") != policy.policy_id
            or raw_policy.get("enter_executable") != str(policy.enter_executable)
            or raw_policy.get("python_executable") != str(policy.python_executable)
            or raw_policy.get("max_probe_age_seconds")
            != policy.max_probe_age_seconds
        ):
            raise LabError("network namespace probe policy binding mismatch")
    return receipt


def validate_network_namespace_probe_for_use(
    receipt: object,
    *,
    network_namespace_plan: object,
    anchor_observation: object,
    host_profile: HostProfile,
    policy: NetworkNamespaceProbePolicy,
    proc_root: Path = Path("/proc"),
    now: datetime | None = None,
) -> dict[str, object]:
    """Revalidate freshness, live executables, and anchor identity before use."""

    plan = validate_network_namespace_plan(
        network_namespace_plan,
        host_profile=host_profile,
        now=now,
    )
    anchor = validate_network_namespace_anchor_observation(
        anchor_observation,
        network_namespace_plan=plan,
        host_profile=host_profile,
    )
    validated = validate_network_namespace_probe_receipt(
        receipt,
        network_namespace_plan=plan,
        anchor_observation=anchor,
        policy=policy,
    )
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("network namespace probe use clock requires a timezone")
    observed = _timestamp(validated["observed_at"], label="network namespace probe")
    valid_until = _timestamp(
        validated["valid_until"], label="network namespace probe expiry"
    )
    if current < observed - timedelta(seconds=5) or current > valid_until:
        raise LabError("network namespace probe is stale or from the future")
    raw_policy = validated["probe_policy"]
    assert isinstance(raw_policy, dict)
    opened: list[int] = []
    try:
        for path, hash_field, label in (
            (
                policy.enter_executable,
                "enter_executable_sha256",
                "network namespace enter executable",
            ),
            (
                policy.python_executable,
                "python_executable_sha256",
                "network namespace probe Python",
            ),
        ):
            descriptor, digest, _metadata = _open_protected_executable(
                path,
                label=label,
            )
            opened.append(descriptor)
            if digest != raw_policy.get(hash_field):
                raise LabError(f"{label} changed after network namespace probe")
    finally:
        for descriptor in opened:
            os.close(descriptor)
    root = _proc_root(proc_root)
    raw_anchor_pid = anchor["pid"]
    assert isinstance(raw_anchor_pid, int) and not isinstance(raw_anchor_pid, bool)
    anchor_pid = raw_anchor_pid
    observed_process = _process_stat(
        root / str(anchor_pid) / "stat",
        expected_pid=anchor_pid,
    )
    expected_process = {
        "parent_pid": anchor["parent_pid"],
        "process_group_id": anchor["process_group_id"],
        "session_id": anchor["session_id"],
        "proc_start_time_ticks": anchor["proc_start_time_ticks"],
    }
    if (
        observed_process != expected_process
        or _namespace_inode(root, pid=anchor_pid)
        != anchor.get("anchor_network_namespace_inode")
        or not _no_new_privileges(root / str(anchor_pid) / "status")
    ):
        raise LabError("network namespace probe anchor is no longer live")
    return validated
