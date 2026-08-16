from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .endpoint_lease import validate_endpoint_lease_receipt
from .execution_sandbox import validate_service_sandbox_for_launch
from .network_namespace import validate_network_namespace_plan
from .remote_preflight import HostProfile
from .remote_request import SHA256_PATTERN, _sha256
from .service_identity import validate_service_launch_spec_identity
from .workspace import LabError, validate_task_id

SERVICE_LAUNCH_PLAN_SCHEMA_ID = "lab.service-launch-plan.v4"
_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_MAX_ARGUMENTS = 256
_MAX_ARGUMENT_BYTES = 4096
_FORBIDDEN_ENVIRONMENT_NAMES = frozenset(
    {
        "ALL_PROXY",
        "BASH_ENV",
        "ENV",
        "FTP_PROXY",
        "GIT_CONFIG_COUNT",
        "GIT_SSH_COMMAND",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NO_PROXY",
        "PYTHONHOME",
        "PYTHONPATH",
        "RSYNC_PROXY",
        "SSH_AUTH_SOCK",
    }
)
_FIXED_ENVIRONMENT_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "CUDA_VISIBLE_DEVICES",
    }
)


@dataclass(frozen=True)
class TrustedServiceAdapter:
    """Protected command template selected by launch_command_id.

    Instances belong to the installed control plane, never a Task manifest.
    Templates may reference only sandbox-local logical tokens.
    """

    adapter_id: str
    launch_command_id: str
    manifest_sha256: str
    argv_template: tuple[str, ...]
    working_directory: str = "/workspace/source/0"
    environment: tuple[tuple[str, str], ...] = ()
    supported_workload_ids: tuple[str, ...] = ()
    asset_projections: tuple[tuple[str, str, str], ...] = ()


def _validate_adapter(adapter: TrustedServiceAdapter) -> None:
    validate_task_id(adapter.adapter_id)
    validate_task_id(adapter.launch_command_id)
    if not SHA256_PATTERN.fullmatch(adapter.manifest_sha256):
        raise LabError("trusted service adapter has an invalid manifest identity")
    if (
        not adapter.argv_template
        or len(adapter.argv_template) > _MAX_ARGUMENTS
        or adapter.argv_template[0] != "{python}"
    ):
        raise LabError("trusted service adapter requires a bounded {python} command")
    for argument in adapter.argv_template:
        if (
            not isinstance(argument, str)
            or not argument
            or len(argument.encode()) > _MAX_ARGUMENT_BYTES
            or "\x00" in argument
            or "\n" in argument
            or "\r" in argument
        ):
            raise LabError("trusted service adapter has an unsafe argument")
    if (
        not adapter.working_directory.startswith("/workspace/source/")
        or ".." in Path(adapter.working_directory).parts
    ):
        raise LabError("trusted service adapter has an unsafe working directory")
    seen: set[str] = set()
    for name, value in adapter.environment:
        if (
            not _ENVIRONMENT_NAME.fullmatch(name)
            or name in seen
            or name in _FORBIDDEN_ENVIRONMENT_NAMES
            or name in _FIXED_ENVIRONMENT_NAMES
            or not isinstance(value, str)
            or len(value.encode()) > _MAX_ARGUMENT_BYTES
            or "\x00" in value
            or "\n" in value
            or "\r" in value
        ):
            raise LabError(f"trusted service adapter has unsafe environment: {name}")
        seen.add(name)
    if len(adapter.supported_workload_ids) != len(
        set(adapter.supported_workload_ids)
    ):
        raise LabError("trusted service adapter has duplicate workload identities")
    for workload_id in adapter.supported_workload_ids:
        validate_task_id(workload_id)
    projection_targets: set[tuple[str, str]] = set()
    for projection in adapter.asset_projections:
        if not isinstance(projection, tuple) or len(projection) != 3:
            raise LabError("trusted service adapter has an invalid asset projection")
        source_role, destination_role, relative_target = projection
        validate_task_id(source_role)
        validate_task_id(destination_role)
        relative = PurePosixPath(relative_target)
        if (
            not relative_target
            or relative_target.startswith(("/", "~"))
            or "\\" in relative_target
            or any(part in {"", ".", ".."} for part in relative.parts)
            or (destination_role, relative_target) in projection_targets
        ):
            raise LabError("trusted service adapter has an unsafe asset projection")
        projection_targets.add((destination_role, relative_target))


def _adapter_identity(adapter: TrustedServiceAdapter) -> dict[str, object]:
    return {
        "adapter_id": adapter.adapter_id,
        "launch_command_id": adapter.launch_command_id,
        "manifest_sha256": adapter.manifest_sha256,
        "supported_workload_ids": list(adapter.supported_workload_ids),
        "asset_projections": [
            {
                "source_role": source_role,
                "destination_role": destination_role,
                "relative_target": relative_target,
            }
            for source_role, destination_role, relative_target in adapter.asset_projections
        ],
    }


def _asset_projection_bindings(
    adapter: TrustedServiceAdapter,
    sandbox: dict[str, object],
) -> list[tuple[str, str]]:
    raw_mounts = sandbox.get("mounts")
    if not isinstance(raw_mounts, list):
        raise LabError("service launch sandbox lacks mounts")
    assets: dict[str, tuple[str, str]] = {}
    for mount in raw_mounts:
        if not isinstance(mount, dict) or mount.get("kind") != "asset":
            continue
        name = mount.get("name")
        source = mount.get("source")
        target = mount.get("target")
        if not all(isinstance(value, str) for value in (name, source, target)):
            raise LabError("service launch sandbox has an invalid asset mount")
        assets[str(name)] = (str(source), str(target))
    bindings: list[tuple[str, str]] = []
    for source_role, destination_role, relative_target in adapter.asset_projections:
        source_identity = assets.get(source_role)
        destination_identity = assets.get(destination_role)
        if source_identity is None or destination_identity is None:
            raise LabError("trusted service adapter projection asset is unavailable")
        source, _ = source_identity
        _, destination = destination_identity
        target = str(PurePosixPath(destination) / PurePosixPath(relative_target))
        bindings.append((source, target))
    return bindings


def _sandbox_tokens(
    sandbox: dict[str, object],
    *,
    endpoint_port: int,
    workload_id: str | None,
) -> dict[str, str]:
    raw_mounts = sandbox.get("mounts")
    if not isinstance(raw_mounts, list):
        raise LabError("service launch sandbox lacks mounts")
    tokens = {
        "{python}": "/opt/lab/environment/bin/python",
        "{port}": str(endpoint_port),
        "{runtime}": "/run/lab",
    }
    if workload_id is not None:
        tokens["{workload_id}"] = workload_id
    for mount in raw_mounts:
        if not isinstance(mount, dict):
            raise LabError("service launch sandbox has an invalid mount")
        kind = mount.get("kind")
        name = mount.get("name")
        target = mount.get("target")
        if not isinstance(name, str) or not isinstance(target, str):
            raise LabError("service launch sandbox has an invalid mount identity")
        if kind == "source":
            tokens[f"{{source:{name}}}"] = target
        elif kind == "asset":
            tokens[f"{{asset:{name}}}"] = target
    return tokens


def _expand_adapter_value(value: str, *, tokens: dict[str, str]) -> str:
    expanded = value
    for token, replacement in tokens.items():
        expanded = expanded.replace(token, replacement)
    if "{" in expanded or "}" in expanded:
        raise LabError(f"trusted service adapter has an unresolved token: {value}")
    if len(expanded.encode()) > _MAX_ARGUMENT_BYTES:
        raise LabError("expanded service adapter value is too large")
    return expanded


def _system_roots() -> tuple[str, ...]:
    roots = ["/usr"]
    roots.extend(path for path in ("/bin", "/lib", "/lib64") if Path(path).exists())
    return tuple(roots)


def build_service_launch_plan(
    *,
    launch_spec: object,
    service_sandbox_receipt: object,
    host_profile: HostProfile,
    adapter: TrustedServiceAdapter,
    endpoint_lease_receipt: object,
    network_namespace_plan: object,
    now: datetime | None = None,
) -> dict[str, object]:
    """Compile protected receipts into an inert inner-sandbox argv.

    The argv intentionally inherits its caller's network namespace.  It is not
    safe to execute directly: a future trusted launcher must first enter the
    request's attested shared namespace.  This plan remains non-executable.
    """

    spec = validate_service_launch_spec_identity(launch_spec)
    sandbox = validate_service_sandbox_for_launch(
        service_sandbox_receipt,
        launch_spec=spec,
        host_profile=host_profile,
        now=now,
    )
    _validate_adapter(adapter)
    if adapter.launch_command_id != spec.get("launch_command_id"):
        raise LabError("trusted service adapter does not match launch command")
    endpoint_lease = validate_endpoint_lease_receipt(
        endpoint_lease_receipt,
        host_id=host_profile.host_id,
        host_profile_sha256=host_profile.sha256,
        request_id=str(spec["request_id"]),
        slot_id=str(spec["slot_id"]),
        now=now,
    )
    network_plan = validate_network_namespace_plan(
        network_namespace_plan,
        endpoint_lease_receipt=endpoint_lease,
        host_profile=host_profile,
        now=now,
    )
    endpoint_port = endpoint_lease["port"]
    assert isinstance(endpoint_port, int)
    policy = host_profile.sandbox_backend
    if policy is None or policy.backend_id != "bubblewrap-v1":
        raise LabError("service launch plan requires protected bubblewrap")
    probe = sandbox.get("backend_probe_receipt")
    if not isinstance(probe, dict):
        raise LabError("service launch plan lacks a backend probe")
    raw_asset_lock = spec.get("asset_lock")
    if not isinstance(raw_asset_lock, dict):
        raise LabError("service launch plan lacks an asset lock identity")
    raw_workload_id = raw_asset_lock.get("workload_id")
    workload_id = str(raw_workload_id) if isinstance(raw_workload_id, str) else None
    if (
        adapter.supported_workload_ids
        and workload_id is None
    ):
        raise LabError("trusted service adapter requires a workload identity")
    if (
        workload_id is not None
        and adapter.supported_workload_ids
        and workload_id not in adapter.supported_workload_ids
    ):
        raise LabError("trusted service adapter does not support selected workload")
    tokens = _sandbox_tokens(
        sandbox,
        endpoint_port=endpoint_port,
        workload_id=workload_id,
    )
    source_targets = {
        target for token, target in tokens.items() if token.startswith("{source:")
    }
    if adapter.working_directory not in source_targets:
        raise LabError("trusted service adapter working directory is not mounted")
    payload_argv = tuple(
        _expand_adapter_value(argument, tokens=tokens)
        for argument in adapter.argv_template
    )
    if payload_argv[0] != tokens["{python}"]:
        raise LabError("trusted service adapter executable escaped its environment")
    raw_gpu = spec.get("gpu")
    raw_endpoint_policy = spec.get("endpoint_policy")
    raw_allowed_devices = sandbox.get("allowed_gpu_devices")
    if (
        not isinstance(raw_gpu, dict)
        or not isinstance(raw_endpoint_policy, dict)
        or not isinstance(raw_allowed_devices, list)
    ):
        raise LabError("service launch plan inputs are incomplete")
    if (
        endpoint_lease.get("scheme") != raw_endpoint_policy.get("scheme")
        or endpoint_lease.get("host") != raw_endpoint_policy.get("host")
    ):
        raise LabError("endpoint lease differs from protected endpoint policy")
    payload_environment: dict[str, str] = {
        "PATH": "/opt/lab/environment/bin:/usr/bin:/bin",
        "HOME": "/run/lab/home",
        "TMPDIR": "/run/lab/tmp",
        "XDG_CACHE_HOME": "/run/lab/cache",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": str(raw_gpu["gpu_uuid"]),
        **{
            name: _expand_adapter_value(value, tokens=tokens)
            for name, value in adapter.environment
        },
    }
    raw_mounts = sandbox.get("mounts")
    assert isinstance(raw_mounts, list)
    launcher_argv: list[str] = [
        str(policy.executable),
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--cap-drop",
        "ALL",
        "--clearenv",
    ]
    system_roots = _system_roots()
    for root in system_roots:
        launcher_argv.extend(("--ro-bind", root, root))
    launcher_argv.extend(("--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"))
    for device in raw_allowed_devices:
        launcher_argv.extend(("--dev-bind", str(device), str(device)))
    for mount in raw_mounts:
        assert isinstance(mount, dict)
        operation = "--ro-bind" if mount["read_only"] is True else "--bind"
        launcher_argv.extend(
            (operation, str(mount["source"]), str(mount["target"]))
        )
    projection_bindings = _asset_projection_bindings(adapter, sandbox)
    for source, target in projection_bindings:
        launcher_argv.extend(("--ro-bind", source, target))
    for name, value in sorted(payload_environment.items()):
        launcher_argv.extend(("--setenv", name, value))
    launcher_argv.extend(("--chdir", adapter.working_directory, "--", *payload_argv))
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("service launch plan clock requires a timezone")
    content: dict[str, object] = {
        "schema": SERVICE_LAUNCH_PLAN_SCHEMA_ID,
        "service_id": spec["service_id"],
        "request_id": spec["request_id"],
        "slot_id": spec["slot_id"],
        "host_id": host_profile.host_id,
        "host_profile_sha256": host_profile.sha256,
        "launch_spec_sha256": spec["launch_spec_sha256"],
        "service_sandbox_receipt_sha256": sandbox["receipt_sha256"],
        "endpoint_lease_receipt_sha256": endpoint_lease["receipt_sha256"],
        "network_namespace_plan_sha256": network_plan["receipt_sha256"],
        "backend": {
            "backend_id": policy.backend_id,
            "executable": str(policy.executable),
            "sha256": probe["backend_sha256"],
        },
        "adapter": _adapter_identity(adapter),
        "endpoint": {
            "scheme": endpoint_lease["scheme"],
            "host": endpoint_lease["host"],
            "port": endpoint_port,
        },
        "system_roots": list(system_roots),
        "working_directory": adapter.working_directory,
        "payload_environment": payload_environment,
        "payload_argv": list(payload_argv),
        "launcher_argv": launcher_argv,
        "launcher_argv_sha256": _sha256(launcher_argv),
        "network_topology": network_plan["topology"],
        "network_egress": "shared-isolated-loopback-only",
        "downloads_permitted": False,
        "device_access": False,
        "shell_used": False,
        "task_environment_inherited": False,
        "created_at": current.isoformat(),
        "endpoint_namespace_reserved": True,
        "endpoint_reservation_required": True,
        "outer_network_namespace_required": True,
        "outer_network_namespace_joined": False,
        "direct_launcher_execution_permitted": False,
        "launch_authorized": False,
        "service_launch_eligible": False,
        "task_command_eligible": False,
        "process_started": False,
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_service_launch_plan_receipt(receipt)


def validate_service_launch_plan_receipt(plan: object) -> dict[str, object]:
    expected = {
        "schema",
        "service_id",
        "request_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "launch_spec_sha256",
        "service_sandbox_receipt_sha256",
        "endpoint_lease_receipt_sha256",
        "network_namespace_plan_sha256",
        "backend",
        "adapter",
        "endpoint",
        "system_roots",
        "working_directory",
        "payload_environment",
        "payload_argv",
        "launcher_argv",
        "launcher_argv_sha256",
        "network_topology",
        "network_egress",
        "downloads_permitted",
        "device_access",
        "shell_used",
        "task_environment_inherited",
        "created_at",
        "endpoint_namespace_reserved",
        "endpoint_reservation_required",
        "outer_network_namespace_required",
        "outer_network_namespace_joined",
        "direct_launcher_execution_permitted",
        "launch_authorized",
        "service_launch_eligible",
        "task_command_eligible",
        "process_started",
        "receipt_sha256",
    }
    if not isinstance(plan, dict) or set(plan) != expected:
        raise LabError("service launch plan has an unexpected shape")
    if plan.get("schema") != SERVICE_LAUNCH_PLAN_SCHEMA_ID:
        raise LabError("unsupported service launch plan schema")
    for field in ("service_id", "request_id", "slot_id", "host_id"):
        value = plan.get(field)
        if not isinstance(value, str):
            raise LabError(f"service launch plan has invalid {field}")
        validate_task_id(value)
    for field in (
        "host_profile_sha256",
        "launch_spec_sha256",
        "service_sandbox_receipt_sha256",
        "endpoint_lease_receipt_sha256",
        "network_namespace_plan_sha256",
        "launcher_argv_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(plan.get(field, ""))):
            raise LabError(f"service launch plan has invalid {field}")
    backend = plan.get("backend")
    if (
        not isinstance(backend, dict)
        or set(backend) != {"backend_id", "executable", "sha256"}
        or backend.get("backend_id") != "bubblewrap-v1"
        or not isinstance(backend.get("executable"), str)
        or not Path(str(backend["executable"])).is_absolute()
        or Path(str(backend["executable"])).name != "bwrap"
        or not SHA256_PATTERN.fullmatch(str(backend.get("sha256", "")))
    ):
        raise LabError("service launch plan has invalid backend identity")
    adapter = plan.get("adapter")
    if not isinstance(adapter, dict) or set(adapter) != {
        "adapter_id",
        "launch_command_id",
        "manifest_sha256",
        "supported_workload_ids",
        "asset_projections",
    }:
        raise LabError("service launch plan has invalid adapter identity")
    for field in ("adapter_id", "launch_command_id"):
        value = adapter.get(field)
        if not isinstance(value, str):
            raise LabError("service launch plan has invalid adapter identity")
        validate_task_id(value)
    if not SHA256_PATTERN.fullmatch(str(adapter.get("manifest_sha256", ""))):
        raise LabError("service launch plan has invalid adapter identity")
    supported_workload_ids = adapter.get("supported_workload_ids")
    asset_projections = adapter.get("asset_projections")
    if (
        not isinstance(supported_workload_ids, list)
        or len(supported_workload_ids) != len(set(supported_workload_ids))
        or not all(isinstance(value, str) for value in supported_workload_ids)
        or not isinstance(asset_projections, list)
    ):
        raise LabError("service launch plan has invalid adapter identity")
    for workload_id in supported_workload_ids:
        validate_task_id(workload_id)
    for projection in asset_projections:
        if not isinstance(projection, dict) or set(projection) != {
            "source_role",
            "destination_role",
            "relative_target",
        }:
            raise LabError("service launch plan has invalid adapter projection")
        for role in ("source_role", "destination_role"):
            value = projection.get(role)
            if not isinstance(value, str):
                raise LabError("service launch plan has invalid adapter projection")
            validate_task_id(value)
        relative_target = projection.get("relative_target")
        if not isinstance(relative_target, str):
            raise LabError("service launch plan has invalid adapter projection")
        relative = PurePosixPath(relative_target)
        if (
            not relative_target
            or relative_target.startswith(("/", "~"))
            or "\\" in relative_target
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise LabError("service launch plan has invalid adapter projection")
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
        raise LabError("service launch plan has an unsafe endpoint")
    system_roots = plan.get("system_roots")
    if (
        not isinstance(system_roots, list)
        or not system_roots
        or system_roots[0] != "/usr"
        or len(system_roots) != len(set(system_roots))
        or not all(root in {"/usr", "/bin", "/lib", "/lib64"} for root in system_roots)
    ):
        raise LabError("service launch plan has invalid system roots")
    working_directory = plan.get("working_directory")
    if (
        not isinstance(working_directory, str)
        or not working_directory.startswith("/workspace/source/")
        or ".." in Path(working_directory).parts
    ):
        raise LabError("service launch plan has an unsafe working directory")
    environment = plan.get("payload_environment")
    if not isinstance(environment, dict) or not environment:
        raise LabError("service launch plan has invalid payload environment")
    if any(
        not isinstance(name, str)
        or not _ENVIRONMENT_NAME.fullmatch(name)
        or name in _FORBIDDEN_ENVIRONMENT_NAMES
        or not isinstance(value, str)
        or "\x00" in value
        or "\n" in value
        for name, value in environment.items()
    ):
        raise LabError("service launch plan has an unsafe payload environment")
    if (
        environment.get("HF_HUB_OFFLINE") != "1"
        or environment.get("TRANSFORMERS_OFFLINE") != "1"
        or environment.get("PYTHONNOUSERSITE") != "1"
        or not str(environment.get("HOME", "")).startswith("/run/lab/")
    ):
        raise LabError("service launch plan did not enforce offline isolation")
    payload_argv = plan.get("payload_argv")
    launcher_argv = plan.get("launcher_argv")
    if (
        not isinstance(payload_argv, list)
        or not payload_argv
        or len(payload_argv) > _MAX_ARGUMENTS
        or payload_argv[0] != "/opt/lab/environment/bin/python"
        or not all(
            isinstance(argument, str)
            and argument
            and len(argument.encode()) <= _MAX_ARGUMENT_BYTES
            and "\x00" not in argument
            and "\n" not in argument
            for argument in payload_argv
        )
        or not isinstance(launcher_argv, list)
        or not launcher_argv
        or launcher_argv[0] != backend["executable"]
        or launcher_argv[-len(payload_argv) :] != payload_argv
        or "--clearenv" not in launcher_argv
        or "--unshare-net" in launcher_argv
        or "--share-net" in launcher_argv
        or "--unshare-all" in launcher_argv
        or "--unshare-user" not in launcher_argv
        or "--unshare-pid" not in launcher_argv
        or "--dev" not in launcher_argv
        or "/dev" not in launcher_argv
        or any(value in launcher_argv for value in ("/home", "/root", "/mnt"))
        or plan.get("launcher_argv_sha256") != _sha256(launcher_argv)
    ):
        raise LabError("service launch plan has an unsafe launcher argv")
    if (
        plan.get("network_topology") != "shared-isolated-loopback"
        or plan.get("network_egress") != "shared-isolated-loopback-only"
        or plan.get("downloads_permitted") is not False
        or plan.get("device_access") is not False
        or plan.get("shell_used") is not False
        or plan.get("task_environment_inherited") is not False
        or plan.get("endpoint_namespace_reserved") is not True
        or plan.get("endpoint_reservation_required") is not True
        or plan.get("outer_network_namespace_required") is not True
        or plan.get("outer_network_namespace_joined") is not False
        or plan.get("direct_launcher_execution_permitted") is not False
        or plan.get("launch_authorized") is not False
        or plan.get("service_launch_eligible") is not False
        or plan.get("task_command_eligible") is not False
        or plan.get("process_started") is not False
    ):
        raise LabError("service launch plan has an unsafe authority envelope")
    created_at = plan.get("created_at")
    if not isinstance(created_at, str):
        raise LabError("service launch plan has invalid created_at")
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("service launch plan has invalid created_at") from exc
    if created.tzinfo is None:
        raise LabError("service launch plan created_at requires a timezone")
    content = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    if plan.get("receipt_sha256") != _sha256(content):
        raise LabError("service launch plan receipt hash mismatch")
    return plan


def validate_service_launch_plan_bindings(
    plan: object,
    *,
    launch_spec: object,
    service_sandbox_receipt: object,
    host_profile: HostProfile,
    adapter: TrustedServiceAdapter,
    endpoint_lease_receipt: object,
    network_namespace_plan: object,
    now: datetime | None = None,
) -> dict[str, object]:
    """Revalidate an inert plan against protected live inputs.

    A successful result proves identity and sandbox binding only. It does not
    bind the endpoint socket, authorize a launch, or start a process.
    """
    validated = validate_service_launch_plan_receipt(plan)
    spec = validate_service_launch_spec_identity(launch_spec)
    sandbox = validate_service_sandbox_for_launch(
        service_sandbox_receipt,
        launch_spec=spec,
        host_profile=host_profile,
        now=now,
    )
    policy = host_profile.sandbox_backend
    assert policy is not None
    _validate_adapter(adapter)
    endpoint_lease = validate_endpoint_lease_receipt(
        endpoint_lease_receipt,
        host_id=host_profile.host_id,
        host_profile_sha256=host_profile.sha256,
        request_id=str(spec["request_id"]),
        slot_id=str(spec["slot_id"]),
        now=now,
    )
    network_plan = validate_network_namespace_plan(
        network_namespace_plan,
        endpoint_lease_receipt=endpoint_lease,
        host_profile=host_profile,
        now=now,
    )
    if validated.get("adapter") != _adapter_identity(adapter):
        raise LabError("service launch plan adapter differs from protected adapter")
    raw_asset_lock = spec.get("asset_lock")
    assert isinstance(raw_asset_lock, dict)
    raw_workload_id = raw_asset_lock.get("workload_id")
    workload_id = raw_workload_id if isinstance(raw_workload_id, str) else None
    if (
        adapter.supported_workload_ids
        and workload_id is None
    ):
        raise LabError("trusted service adapter requires a workload identity")
    if (
        workload_id is not None
        and adapter.supported_workload_ids
        and workload_id not in adapter.supported_workload_ids
    ):
        raise LabError("trusted service adapter does not support selected workload")
    probe = sandbox["backend_probe_receipt"]
    assert isinstance(probe, dict)
    bindings = {
        "service_id": spec["service_id"],
        "request_id": spec["request_id"],
        "slot_id": spec["slot_id"],
        "host_id": host_profile.host_id,
        "host_profile_sha256": host_profile.sha256,
        "launch_spec_sha256": spec["launch_spec_sha256"],
        "service_sandbox_receipt_sha256": sandbox["receipt_sha256"],
        "endpoint_lease_receipt_sha256": endpoint_lease["receipt_sha256"],
        "network_namespace_plan_sha256": network_plan["receipt_sha256"],
        "backend": {
            "backend_id": policy.backend_id,
            "executable": str(policy.executable),
            "sha256": probe["backend_sha256"],
        },
    }
    for field, expected_value in bindings.items():
        if validated.get(field) != expected_value:
            raise LabError(f"service launch plan binding mismatch: {field}")
    if validated.get("system_roots") != list(_system_roots()):
        raise LabError("service launch plan system roots drifted")
    endpoint = validated["endpoint"]
    endpoint_policy = spec["endpoint_policy"]
    assert isinstance(endpoint, dict)
    assert isinstance(endpoint_policy, dict)
    if (
        endpoint.get("scheme") != endpoint_policy.get("scheme")
        or endpoint.get("host") != endpoint_policy.get("host")
        or endpoint.get("scheme") != endpoint_lease.get("scheme")
        or endpoint.get("host") != endpoint_lease.get("host")
        or endpoint.get("port") != endpoint_lease.get("port")
    ):
        raise LabError("service launch plan endpoint differs from protected policy")
    launcher_argv = validated["launcher_argv"]
    payload_environment = validated["payload_environment"]
    raw_mounts = sandbox["mounts"]
    raw_devices = sandbox["allowed_gpu_devices"]
    assert isinstance(launcher_argv, list)
    assert isinstance(payload_environment, dict)
    assert isinstance(raw_mounts, list)
    assert isinstance(raw_devices, list)

    def option_pairs(option: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for index, value in enumerate(launcher_argv):
            if value != option:
                continue
            if index + 2 >= len(launcher_argv):
                raise LabError(f"service launch plan has incomplete {option}")
            first = launcher_argv[index + 1]
            second = launcher_argv[index + 2]
            if not isinstance(first, str) or not isinstance(second, str):
                raise LabError(f"service launch plan has invalid {option}")
            pairs.append((first, second))
        return pairs

    expected_read_only = [(root, root) for root in _system_roots()]
    expected_read_only.extend(
        (str(mount["source"]), str(mount["target"]))
        for mount in raw_mounts
        if isinstance(mount, dict) and mount.get("read_only") is True
    )
    expected_read_only.extend(_asset_projection_bindings(adapter, sandbox))
    expected_writable = [
        (str(mount["source"]), str(mount["target"]))
        for mount in raw_mounts
        if isinstance(mount, dict) and mount.get("read_only") is False
    ]
    expected_devices = [(str(device), str(device)) for device in raw_devices]
    if (
        option_pairs("--ro-bind") != expected_read_only
        or option_pairs("--bind") != expected_writable
        or option_pairs("--dev-bind") != expected_devices
        or option_pairs("--setenv")
        != sorted((str(name), str(value)) for name, value in payload_environment.items())
    ):
        raise LabError("service launch plan mount, device, or environment drifted")
    return validated
