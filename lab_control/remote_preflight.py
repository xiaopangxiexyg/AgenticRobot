from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .gpu_lease import validate_gpu_lease_receipt
from .remote_request import SHA256_PATTERN, RemoteProfile, validate_remote_request
from .workspace import LabError, validate_repository_name, validate_task_id

HOST_PROFILE_SCHEMA_ID = "lab.host-profile.v1"
HOST_OBSERVATION_SCHEMA_ID = "lab.host-observation.v1"
PREFLIGHT_RECEIPT_SCHEMA_ID = "lab.preflight-receipt.v1"


@dataclass(frozen=True)
class RepositoryPlacement:
    name: str
    canonical_identity: str
    path: Path


@dataclass(frozen=True)
class EnvironmentPlacement:
    environment_id: str
    path: Path
    python_major_minor: str
    repository: str
    import_modules: tuple[str, ...]
    source_roots: tuple[str, ...] = (".",)


@dataclass(frozen=True)
class HostNetworkPolicy:
    task_payload_egress: str
    trusted_prepare_egress: str
    implicit_downloads: bool
    large_file_downloads: bool


@dataclass(frozen=True)
class SandboxBackendPolicy:
    backend_id: str
    executable: Path
    allowed_gpu_devices: tuple[Path, ...]
    max_probe_age_seconds: int


@dataclass(frozen=True)
class HostProfile:
    host_id: str
    hostname: str
    capabilities: tuple[str, ...]
    repositories: Mapping[str, RepositoryPlacement]
    environments: Mapping[str, EnvironmentPlacement]
    asset_roots: tuple[Path, ...]
    slots: tuple[str, ...]
    slot_root: Path
    control_plane_projectctl: Path
    device_access: bool
    network_policy: HostNetworkPolicy
    sha256: str
    sandbox_backend: SandboxBackendPolicy | None = None


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _load_object(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LabError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError(f"{label} must be an object: {path}")
    return payload, raw


def _absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise LabError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise LabError(f"{label} must be an absolute path without traversal")
    return path


def load_host_profile(profile_root: Path, host_id: str) -> HostProfile:
    validate_task_id(host_id)
    root = profile_root.resolve()
    path = (root / f"{host_id}.json").resolve()
    if path.parent != root or not path.is_file():
        raise LabError(f"unknown host Profile: {host_id} ({path})")
    payload, raw = _load_object(path, label="host Profile")
    if payload.get("schema") != HOST_PROFILE_SCHEMA_ID:
        raise LabError(f"unsupported host Profile schema: {path}")
    if payload.get("host_id") != host_id:
        raise LabError(f"host Profile identity mismatch: {path}")
    hostname = payload.get("hostname")
    if not isinstance(hostname, str) or not hostname:
        raise LabError(f"host Profile requires hostname: {path}")
    if payload.get("device_access") is not False:
        raise LabError(f"M3 host Profile must deny device access: {path}")
    raw_network = payload.get("network_policy")
    if not isinstance(raw_network, dict) or set(raw_network) != {
        "task_payload_egress",
        "trusted_prepare_egress",
        "implicit_downloads",
        "large_file_downloads",
    }:
        raise LabError(f"host Profile requires a protected network policy: {path}")
    if (
        raw_network.get("task_payload_egress") != "loopback-only"
        or raw_network.get("trusted_prepare_egress") != "explicit-only"
        or raw_network.get("implicit_downloads") is not False
        or raw_network.get("large_file_downloads") is not False
    ):
        raise LabError(f"host Profile has an unsafe network policy: {path}")
    network_policy = HostNetworkPolicy(
        task_payload_egress="loopback-only",
        trusted_prepare_egress="explicit-only",
        implicit_downloads=False,
        large_file_downloads=False,
    )
    raw_sandbox = payload.get("sandbox_backend")
    if not isinstance(raw_sandbox, dict) or set(raw_sandbox) != {
        "backend_id",
        "executable",
        "allowed_gpu_devices",
        "max_probe_age_seconds",
    }:
        raise LabError(f"host Profile requires a sandbox backend policy: {path}")
    executable = _absolute_path(
        raw_sandbox.get("executable"),
        label="sandbox backend executable",
    )
    raw_gpu_devices = raw_sandbox.get("allowed_gpu_devices")
    max_probe_age = raw_sandbox.get("max_probe_age_seconds")
    if (
        raw_sandbox.get("backend_id") != "bubblewrap-v1"
        or executable != Path("/usr/bin/bwrap")
        or not isinstance(raw_gpu_devices, list)
        or not raw_gpu_devices
        or not all(isinstance(device, str) for device in raw_gpu_devices)
        or len(raw_gpu_devices) != len(set(raw_gpu_devices))
        or not isinstance(max_probe_age, int)
        or isinstance(max_probe_age, bool)
        or not 30 <= max_probe_age <= 600
    ):
        raise LabError(f"host Profile has an unsafe sandbox backend policy: {path}")
    allowed_gpu_devices = tuple(
        _absolute_path(device, label="sandbox GPU device")
        for device in raw_gpu_devices
    )
    if not all(
        re.fullmatch(
            r"/dev/nvidia(?:[0-9]+|ctl|modeset|-uvm|-uvm-tools)",
            str(device),
        )
        for device in allowed_gpu_devices
    ):
        raise LabError(f"host Profile sandbox permits a non-GPU device: {path}")
    sandbox_backend = SandboxBackendPolicy(
        backend_id="bubblewrap-v1",
        executable=executable,
        allowed_gpu_devices=allowed_gpu_devices,
        max_probe_age_seconds=max_probe_age,
    )

    raw_capabilities = payload.get("capabilities")
    if (
        not isinstance(raw_capabilities, list)
        or not raw_capabilities
        or not all(isinstance(value, str) and value for value in raw_capabilities)
        or len(raw_capabilities) != len(set(raw_capabilities))
    ):
        raise LabError(f"host Profile requires unique capabilities: {path}")
    capabilities = tuple(raw_capabilities)

    raw_repositories = payload.get("repositories")
    if not isinstance(raw_repositories, dict) or not raw_repositories:
        raise LabError(f"host Profile has no repositories: {path}")
    repositories: dict[str, RepositoryPlacement] = {}
    for name, raw_repository in raw_repositories.items():
        validate_repository_name(name)
        if not isinstance(raw_repository, dict):
            raise LabError(f"invalid host repository placement: {name}")
        identity = raw_repository.get("canonical_identity")
        if not isinstance(identity, str) or identity.count("/") != 1:
            raise LabError(f"invalid canonical repository identity: {name}")
        repositories[name] = RepositoryPlacement(
            name=name,
            canonical_identity=identity,
            path=_absolute_path(raw_repository.get("path"), label=f"{name} path"),
        )

    raw_environments = payload.get("environments")
    if not isinstance(raw_environments, dict) or not raw_environments:
        raise LabError(f"host Profile has no environments: {path}")
    environments: dict[str, EnvironmentPlacement] = {}
    for environment_id, raw_environment in raw_environments.items():
        validate_task_id(environment_id)
        if not isinstance(raw_environment, dict):
            raise LabError(f"invalid environment placement: {environment_id}")
        repository = str(raw_environment.get("repository", ""))
        validate_repository_name(repository)
        if repository not in repositories:
            raise LabError(f"environment references unknown repository: {repository}")
        python_major_minor = raw_environment.get("python_major_minor")
        if not isinstance(python_major_minor, str):
            raise LabError(f"environment lacks Python identity: {environment_id}")
        raw_import_modules = raw_environment.get("import_modules")
        if (
            not isinstance(raw_import_modules, list)
            or not raw_import_modules
            or not all(
                isinstance(module, str) and module for module in raw_import_modules
            )
            or len(raw_import_modules) != len(set(raw_import_modules))
        ):
            raise LabError(f"environment lacks unique import modules: {environment_id}")
        raw_source_roots = raw_environment.get("source_roots")
        if (
            not isinstance(raw_source_roots, list)
            or not raw_source_roots
            or not all(isinstance(root, str) and root for root in raw_source_roots)
            or len(raw_source_roots) != len(set(raw_source_roots))
        ):
            raise LabError(f"environment lacks unique source roots: {environment_id}")
        source_roots: list[str] = []
        for source_root in raw_source_roots:
            pure = PurePosixPath(source_root)
            if (
                source_root.startswith(("/", "~"))
                or "\\" in source_root
                or ".." in pure.parts
            ):
                raise LabError(f"environment has unsafe source root: {environment_id}")
            source_roots.append(source_root)
        environments[environment_id] = EnvironmentPlacement(
            environment_id=environment_id,
            path=_absolute_path(
                raw_environment.get("path"),
                label=f"{environment_id} path",
            ),
            python_major_minor=python_major_minor,
            repository=repository,
            import_modules=tuple(raw_import_modules),
            source_roots=tuple(source_roots),
        )

    raw_asset_roots = payload.get("asset_roots")
    if not isinstance(raw_asset_roots, list) or not raw_asset_roots:
        raise LabError(f"host Profile has no asset roots: {path}")
    asset_roots = tuple(
        _absolute_path(value, label="asset root") for value in raw_asset_roots
    )
    raw_slots = payload.get("slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        raise LabError(f"host Profile has no slots: {path}")
    slots = tuple(str(slot) for slot in raw_slots)
    for slot in slots:
        validate_task_id(slot)
    if len(slots) != len(set(slots)):
        raise LabError(f"host Profile contains duplicate slots: {path}")
    return HostProfile(
        host_id=host_id,
        hostname=hostname,
        capabilities=capabilities,
        repositories=repositories,
        environments=environments,
        asset_roots=asset_roots,
        slots=slots,
        slot_root=_absolute_path(payload.get("slot_root"), label="slot root"),
        control_plane_projectctl=_absolute_path(
            payload.get("control_plane_projectctl"),
            label="control-plane projectctl",
        ),
        device_access=False,
        network_policy=network_policy,
        sha256=hashlib.sha256(raw).hexdigest(),
        sandbox_backend=sandbox_backend,
    )


def _require_observation_entry(
    mapping: object,
    key: str,
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(mapping, Mapping):
        raise LabError(f"host observation {label} must be an object")
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise LabError(f"host observation lacks {label} entry: {key}")
    return value


def preflight_remote_request(
    request: object,
    *,
    remote_profile: RemoteProfile,
    host_profile: HostProfile,
    observation: object,
    control_plane_version: str,
) -> dict[str, object]:
    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    if remote_profile.target_host != host_profile.host_id:
        raise LabError("remote Profile and host Profile identities do not match")
    if not isinstance(observation, Mapping):
        raise LabError("host observation must be an object")
    if observation.get("schema") != HOST_OBSERVATION_SCHEMA_ID:
        raise LabError("unsupported host observation schema")
    if observation.get("host_id") != host_profile.host_id:
        raise LabError("host observation identity mismatch")
    if observation.get("hostname") != host_profile.hostname:
        raise LabError("host observation hostname mismatch")
    if observation.get("device_groups") not in ([], ()):
        raise LabError("GPU-only preflight refuses device groups")
    if remote_profile.required_assets and not SHA256_PATTERN.fullmatch(
        str(observation.get("asset_verification_receipt_sha256", ""))
    ):
        raise LabError(
            "GPU-only preflight requires a content-rehashed asset verification"
        )
    if (
        remote_profile.network_egress
        != host_profile.network_policy.task_payload_egress
        or remote_profile.downloads_permitted
        or host_profile.network_policy.implicit_downloads
        or host_profile.network_policy.large_file_downloads
    ):
        raise LabError("remote and host network policies do not match safe execution")
    if not set(remote_profile.required_capabilities).issubset(
        host_profile.capabilities
    ):
        raise LabError("host Profile lacks required capabilities")
    observed_capabilities = observation.get("capabilities")
    if not isinstance(observed_capabilities, list) or not set(
        remote_profile.required_capabilities
    ).issubset(observed_capabilities):
        raise LabError("host observation lacks required capabilities")
    gpu = observation.get("gpu")
    if not isinstance(gpu, Mapping) or gpu.get("reserved_count") != 1:
        raise LabError("GPU-only preflight requires exactly one reserved GPU")
    if not isinstance(gpu.get("lease_id"), str) or not gpu["lease_id"]:
        raise LabError("GPU-only preflight requires a GPU lease receipt")
    if not isinstance(gpu.get("gpu_uuid"), str) or not gpu["gpu_uuid"]:
        raise LabError("GPU-only preflight requires a GPU UUID")
    if not SHA256_PATTERN.fullmatch(str(gpu.get("lease_receipt_sha256", ""))):
        raise LabError("GPU-only preflight requires a bound GPU lease receipt hash")
    lease = validate_gpu_lease_receipt(
        gpu.get("lease_receipt"),
        host_id=host_profile.host_id,
        gpu_uuid=str(gpu["gpu_uuid"]),
        require_active=True,
        host_profile_sha256=host_profile.sha256,
    )
    if (
        gpu.get("lease_id") != lease.get("lease_id")
        or gpu.get("gpu_id") != lease.get("gpu_id")
        or gpu.get("lease_receipt_sha256") != lease.get("receipt_sha256")
    ):
        raise LabError("GPU observation does not match its active lease receipt")

    request_repositories = validated_request["repositories"]
    observation_repositories = observation.get("repositories")
    assert isinstance(request_repositories, dict)
    for name in remote_profile.repositories:
        placement = host_profile.repositories.get(name)
        if placement is None:
            raise LabError(f"host Profile lacks repository placement: {name}")
        expected = request_repositories[name]
        actual = _require_observation_entry(
            observation_repositories,
            name,
            label="repositories",
        )
        if actual.get("canonical_identity") != placement.canonical_identity:
            raise LabError(f"repository identity mismatch: {name}")
        if actual.get("path") != str(placement.path):
            raise LabError(f"repository path mismatch: {name}")
        if actual.get("realpath") != str(placement.path) or actual.get("is_symlink"):
            raise LabError(f"repository anchor must be a real path: {name}")
        if actual.get("exists") is not True or actual.get("dirty") is not False:
            raise LabError(f"repository anchor is missing or dirty: {name}")
        if not isinstance(expected, dict):
            raise LabError(f"invalid request repository: {name}")
        for field in ("commit", "tree_oid", "submodules"):
            if actual.get(field) != expected.get(field):
                raise LabError(f"repository {field} mismatch: {name}")
        if actual.get("submodules_initialized") is not True:
            raise LabError(f"repository submodules are not initialized/exact: {name}")
        if actual.get("published_ref_contains") is not True:
            raise LabError(f"repository commit is not published/reachable: {name}")

    observation_environments = observation.get("environments")
    for environment_id in remote_profile.required_environments:
        environment_placement = host_profile.environments.get(environment_id)
        if environment_placement is None:
            raise LabError(f"host Profile lacks environment: {environment_id}")
        actual = _require_observation_entry(
            observation_environments,
            environment_id,
            label="environments",
        )
        repository_path = host_profile.repositories[
            environment_placement.repository
        ].path
        import_origins = actual.get("import_origins")
        if (
            actual.get("path") != str(environment_placement.path)
            or actual.get("realpath") != str(environment_placement.path)
            or actual.get("is_symlink") is not False
            or actual.get("exists") is not True
            or actual.get("python_major_minor")
            != environment_placement.python_major_minor
            or not isinstance(import_origins, Mapping)
            or set(import_origins) != set(environment_placement.import_modules)
        ):
            raise LabError(f"environment provenance mismatch: {environment_id}")
        for module in environment_placement.import_modules:
            origin = _absolute_path(
                import_origins[module],
                label=f"{environment_id}/{module} import origin",
            )
            if not origin.is_relative_to(repository_path):
                raise LabError(f"environment provenance mismatch: {environment_id}")

    request_assets = validated_request["assets"]
    observation_assets = observation.get("assets")
    assert isinstance(request_assets, dict)
    for role in remote_profile.required_assets:
        expected = request_assets[role]
        actual = _require_observation_entry(observation_assets, role, label="assets")
        if not isinstance(expected, dict):
            raise LabError(f"invalid request asset: {role}")
        if (
            actual.get("asset_id") != expected.get("asset_id")
            or actual.get("sha256") != expected.get("sha256")
            or actual.get("exists") is not True
            or actual.get("readable") is not True
        ):
            raise LabError(f"asset identity/availability mismatch: {role}")
        realpath = _absolute_path(actual.get("realpath"), label=f"{role} realpath")
        if not any(realpath.is_relative_to(root) for root in host_profile.asset_roots):
            raise LabError(f"asset escapes allowed roots: {role}")

    receipt_content: dict[str, object] = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA_ID,
        "request_id": validated_request["request_id"],
        "idempotency_key": validated_request["idempotency_key"],
        "host_id": host_profile.host_id,
        "host_profile_sha256": host_profile.sha256,
        "remote_profile_sha256": remote_profile.sha256,
        "control_plane_version": control_plane_version,
        "asset_lock": validated_request["asset_lock"],
        "observation_sha256": hashlib.sha256(_canonical_json(observation)).hexdigest(),
        "gpu_lease_id": gpu["lease_id"],
        "gpu_lease_receipt_sha256": gpu["lease_receipt_sha256"],
        "gpu_uuid": gpu["gpu_uuid"],
        "source_verified": True,
        "environment_verified": True,
        "assets_verified": True,
        "device_access": False,
    }
    return {
        **receipt_content,
        "receipt_sha256": hashlib.sha256(_canonical_json(receipt_content)).hexdigest(),
    }


def validate_preflight_receipt(
    receipt: object,
    *,
    request_id: str,
    idempotency_key: str,
) -> dict[str, object]:
    """Verify a preflight receipt's integrity and fail-closed safety assertions."""

    if not isinstance(receipt, dict):
        raise LabError("preflight receipt must be an object")
    expected_keys = {
        "schema",
        "request_id",
        "idempotency_key",
        "host_id",
        "host_profile_sha256",
        "remote_profile_sha256",
        "control_plane_version",
        "asset_lock",
        "observation_sha256",
        "gpu_lease_id",
        "gpu_lease_receipt_sha256",
        "gpu_uuid",
        "source_verified",
        "environment_verified",
        "assets_verified",
        "device_access",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise LabError("preflight receipt has an unexpected shape")
    if receipt.get("schema") != PREFLIGHT_RECEIPT_SCHEMA_ID:
        raise LabError("unsupported preflight receipt schema")
    if receipt.get("request_id") != request_id:
        raise LabError("preflight receipt belongs to a different request")
    if receipt.get("idempotency_key") != idempotency_key:
        raise LabError("preflight receipt idempotency mismatch")
    for field in (
        "host_profile_sha256",
        "remote_profile_sha256",
        "observation_sha256",
        "gpu_lease_receipt_sha256",
        "receipt_sha256",
    ):
        value = receipt.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LabError(f"preflight receipt has invalid {field}")
    for field in ("host_id", "control_plane_version", "gpu_lease_id", "gpu_uuid"):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise LabError(f"preflight receipt has invalid {field}")
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
        raise LabError("preflight receipt has invalid asset lock identity")
    validate_task_id(str(asset_lock["lock_id"]))
    if asset_lock["workload_id"] is not None:
        validate_task_id(str(asset_lock["workload_id"]))
    if (
        receipt.get("source_verified") is not True
        or receipt.get("environment_verified") is not True
        or receipt.get("assets_verified") is not True
        or receipt.get("device_access") is not False
    ):
        raise LabError("preflight receipt does not assert the safe execution envelope")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        hashlib.sha256(_canonical_json(content)).hexdigest()
        != receipt["receipt_sha256"]
    ):
        raise LabError("preflight receipt hash mismatch")
    return receipt
