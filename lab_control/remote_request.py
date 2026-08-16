from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .workspace import LabError, validate_repository_name, validate_task_id

REMOTE_PROFILE_SCHEMA_ID = "lab.remote-profile.v1"
REMOTE_REQUEST_SCHEMA_ID = "lab.remote-request.v2"
OID_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ServicePolicy:
    service_name: str
    repository: str
    environment_id: str
    protocol: str
    launch_mode: str
    launch_command_id: str
    endpoint_scheme: str


@dataclass(frozen=True)
class RemoteProfile:
    profile_id: str
    target_host: str
    risk_level: str
    repositories: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    required_environments: tuple[str, ...]
    required_assets: tuple[str, ...]
    command_ids: tuple[str, ...]
    gpu_count: int
    device_access: bool
    network_egress: str
    downloads_permitted: bool
    service_policy: ServicePolicy
    sha256: str


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _string_tuple(payload: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(value, str) and value for value in payload)
    ):
        raise LabError(f"{label} must be a non-empty string list")
    if len(payload) != len(set(payload)):
        raise LabError(f"{label} contains duplicates")
    return tuple(payload)


def _load_service_policy(
    payload: object,
    *,
    repositories: tuple[str, ...],
    environments: tuple[str, ...],
) -> ServicePolicy:
    if not isinstance(payload, dict):
        raise LabError("remote Profile requires a protected service policy")
    expected_keys = {
        "service_name",
        "repository",
        "environment_id",
        "protocol",
        "launch_mode",
        "launch_command_id",
        "endpoint_scheme",
    }
    if set(payload) != expected_keys:
        raise LabError("remote Profile service policy has an unexpected shape")
    for field in expected_keys:
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise LabError(f"remote Profile service policy has invalid {field}")
    service_name = str(payload["service_name"])
    environment_id = str(payload["environment_id"])
    launch_command_id = str(payload["launch_command_id"])
    validate_task_id(service_name)
    validate_task_id(environment_id)
    validate_task_id(launch_command_id)
    repository = str(payload["repository"])
    validate_repository_name(repository)
    if repository not in repositories:
        raise LabError("service repository is not required by remote Profile")
    if environment_id not in environments:
        raise LabError("service environment is not required by remote Profile")
    protocol = str(payload["protocol"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", protocol):
        raise LabError(f"invalid protected service protocol: {protocol}")
    launch_mode = str(payload["launch_mode"])
    if launch_mode not in {"ephemeral", "attach-existing"}:
        raise LabError(f"unsupported protected service launch mode: {launch_mode}")
    endpoint_scheme = str(payload["endpoint_scheme"])
    if endpoint_scheme not in {"http", "https", "ws", "wss"}:
        raise LabError(f"unsupported protected endpoint scheme: {endpoint_scheme}")
    return ServicePolicy(
        service_name=service_name,
        repository=repository,
        environment_id=environment_id,
        protocol=protocol,
        launch_mode=launch_mode,
        launch_command_id=launch_command_id,
        endpoint_scheme=endpoint_scheme,
    )


def load_remote_profile(profile_root: Path, profile_id: str) -> RemoteProfile:
    validate_task_id(profile_id)
    root = profile_root.resolve()
    path = (root / f"{profile_id}.json").resolve()
    if path.parent != root or not path.is_file():
        raise LabError(f"unknown remote Profile: {profile_id} ({path})")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LabError(f"cannot read remote Profile {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != REMOTE_PROFILE_SCHEMA_ID
    ):
        raise LabError(f"unsupported remote Profile schema: {path}")
    if payload.get("profile_id") != profile_id:
        raise LabError(f"remote Profile identity mismatch: {path}")
    if payload.get("execution") != "remote-gpu":
        raise LabError(f"M3 only permits remote-gpu Profiles: {path}")
    if payload.get("requires_clean") is not True:
        raise LabError(f"M3 remote Profile must require clean snapshots: {path}")
    if payload.get("risk_level") != "R1":
        raise LabError(f"M3 GPU-only Profile must be R1: {path}")
    target_host = str(payload.get("target_host", ""))
    validate_task_id(target_host)
    repositories = _string_tuple(payload.get("repositories"), label="repositories")
    for repository in repositories:
        validate_repository_name(repository)
    required_capabilities = _string_tuple(
        payload.get("required_capabilities"),
        label="required_capabilities",
    )
    for capability in required_capabilities:
        if not re.fullmatch(r"[a-z0-9][a-z0-9:._-]{0,63}", capability):
            raise LabError(f"invalid required capability: {capability}")
    required_environments = _string_tuple(
        payload.get("required_environments"),
        label="required_environments",
    )
    for environment in required_environments:
        validate_task_id(environment)
    required_assets = _string_tuple(
        payload.get("required_assets"),
        label="required_assets",
    )
    for asset in required_assets:
        validate_task_id(asset)
    command_ids = _string_tuple(payload.get("command_ids"), label="command_ids")
    for command_id in command_ids:
        validate_task_id(command_id)
    resources = payload.get("resources")
    if not isinstance(resources, dict):
        raise LabError(f"remote Profile has no resources: {path}")
    gpu_count = resources.get("gpu_count")
    device_access = resources.get("device_access")
    network_egress = resources.get("network_egress")
    downloads_permitted = resources.get("downloads_permitted")
    if (
        gpu_count != 1
        or device_access is not False
        or network_egress != "loopback-only"
        or downloads_permitted is not False
    ):
        raise LabError(
            "M3 Profile must request exactly one GPU, no device access, "
            f"loopback-only egress, and no downloads: {path}"
        )
    forbidden = set(resources) - {
        "gpu_count",
        "device_access",
        "network_egress",
        "downloads_permitted",
    }
    if forbidden:
        raise LabError(
            f"M3 Profile contains unsupported resource keys: {sorted(forbidden)}"
        )
    service_policy = _load_service_policy(
        payload.get("service"),
        repositories=repositories,
        environments=required_environments,
    )
    return RemoteProfile(
        profile_id=profile_id,
        target_host=target_host,
        risk_level="R1",
        repositories=repositories,
        required_capabilities=required_capabilities,
        required_environments=required_environments,
        required_assets=required_assets,
        command_ids=command_ids,
        gpu_count=1,
        device_access=False,
        network_egress="loopback-only",
        downloads_permitted=False,
        service_policy=service_policy,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_remote_request(
    *,
    task_id: str,
    snapshot_id: str,
    repositories: Mapping[str, Mapping[str, object]],
    assets: Mapping[str, Mapping[str, str]],
    asset_lock: Mapping[str, object],
    profile: RemoteProfile,
    control_plane_version: str,
    repository_links: object = (),
) -> dict[str, object]:
    validate_task_id(task_id)
    validate_task_id(snapshot_id)
    validate_task_id(control_plane_version)
    repository_payload: dict[str, object] = {}
    if set(repositories) != set(profile.repositories):
        raise LabError(
            "remote request repositories do not match protected Profile: "
            f"expected {profile.repositories}, got {tuple(repositories)}"
        )
    for name in profile.repositories:
        validate_repository_name(name)
        source = repositories[name]
        if not isinstance(source, Mapping):
            raise LabError(f"repository identity must be an object: {name}")
        commit = str(source.get("commit", ""))
        tree_oid = str(source.get("tree_oid", ""))
        if not OID_PATTERN.fullmatch(commit) or not OID_PATTERN.fullmatch(tree_oid):
            raise LabError(f"repository {name} requires full commit/tree OIDs")
        if source.get("dirty") is not False:
            raise LabError(f"remote GPU request rejects dirty repository: {name}")
        raw_submodules = source.get("submodules", {})
        if not isinstance(raw_submodules, dict):
            raise LabError(f"invalid submodule map for {name}")
        submodules: dict[str, str] = {}
        for submodule_path, oid in sorted(raw_submodules.items()):
            if (
                not isinstance(submodule_path, str)
                or submodule_path.startswith("/")
                or ".." in Path(submodule_path).parts
                or not isinstance(oid, str)
                or not OID_PATTERN.fullmatch(oid)
            ):
                raise LabError(
                    f"invalid submodule identity for {name}: {submodule_path}"
                )
            submodules[submodule_path] = oid
        repository_payload[name] = {
            "commit": commit,
            "tree_oid": tree_oid,
            "dirty": False,
            "submodules": submodules,
        }

    if not isinstance(repository_links, (list, tuple)):
        raise LabError("remote request repository links must be a list")
    link_payload: list[dict[str, str]] = []
    link_identities: set[tuple[str, str]] = set()
    for index, raw_link in enumerate(repository_links):
        if not isinstance(raw_link, Mapping) or set(raw_link) != {
            "parent_repository",
            "submodule_path",
            "child_repository",
            "commit",
        }:
            raise LabError(f"invalid remote repository link: {index}")
        parent = str(raw_link.get("parent_repository", ""))
        child = str(raw_link.get("child_repository", ""))
        validate_repository_name(parent)
        validate_repository_name(child)
        if (
            parent == child
            or parent not in repository_payload
            or child not in repository_payload
        ):
            raise LabError(
                f"remote repository link is outside the requested workspace: "
                f"{parent} -> {child}"
            )
        raw_path = raw_link.get("submodule_path")
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
            raise LabError(f"invalid remote repository link path: {index}")
        path = PurePosixPath(raw_path)
        if (
            raw_path.startswith(("/", "~"))
            or raw_path != path.as_posix()
            or ".." in path.parts
        ):
            raise LabError(f"invalid remote repository link path: {index}")
        link_commit = raw_link.get("commit")
        if not isinstance(link_commit, str) or not OID_PATTERN.fullmatch(
            link_commit
        ):
            raise LabError(f"remote repository link requires a full commit: {index}")
        identity = (parent, raw_path)
        if identity in link_identities:
            raise LabError(f"duplicate remote repository link: {parent}/{raw_path}")
        link_identities.add(identity)
        parent_repository = repository_payload[parent]
        child_repository = repository_payload[child]
        if not isinstance(parent_repository, dict) or not isinstance(
            child_repository, dict
        ):
            raise LabError("remote repository link identity is invalid")
        parent_submodules = parent_repository.get("submodules")
        if (
            not isinstance(parent_submodules, dict)
            or parent_submodules.get(raw_path) != link_commit
            or child_repository.get("commit") != link_commit
        ):
            raise LabError(
                f"remote repository link differs from repository identities: "
                f"{parent}/{raw_path} -> {child}"
            )
        link_payload.append(
            {
                "parent_repository": parent,
                "submodule_path": raw_path,
                "child_repository": child,
                "commit": link_commit,
            }
        )
    link_payload.sort(
        key=lambda link: (
            link["parent_repository"],
            link["submodule_path"],
            link["child_repository"],
        )
    )

    if set(assets) != set(profile.required_assets):
        raise LabError(
            "remote request assets do not match protected Profile: "
            f"expected {profile.required_assets}, got {tuple(assets)}"
        )
    asset_payload: dict[str, object] = {}
    for role in profile.required_assets:
        validate_task_id(role)
        source = assets[role]
        if not isinstance(source, Mapping):
            raise LabError(f"asset identity must be an object: {role}")
        asset_id = source.get("asset_id")
        sha256 = source.get("sha256")
        if not isinstance(asset_id, str):
            raise LabError(f"invalid asset ID for {role}")
        validate_task_id(asset_id)
        if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
            raise LabError(f"asset {role} requires a full SHA-256")
        asset_payload[role] = {"asset_id": asset_id, "sha256": sha256}

    if set(asset_lock) != {"lock_id", "workload_id", "sha256"}:
        raise LabError("remote request requires an exact asset lock identity")
    asset_lock_id = asset_lock.get("lock_id")
    asset_lock_workload_id = asset_lock.get("workload_id")
    asset_lock_sha256 = asset_lock.get("sha256")
    if not isinstance(asset_lock_id, str):
        raise LabError("remote request asset lock has an invalid ID")
    validate_task_id(asset_lock_id)
    if asset_lock_workload_id is not None:
        if not isinstance(asset_lock_workload_id, str):
            raise LabError("remote request asset lock has an invalid workload ID")
        validate_task_id(asset_lock_workload_id)
    if not isinstance(asset_lock_sha256, str) or not SHA256_PATTERN.fullmatch(
        asset_lock_sha256
    ):
        raise LabError("remote request asset lock requires a full SHA-256")

    content: dict[str, object] = {
        "schema": REMOTE_REQUEST_SCHEMA_ID,
        "task_id": task_id,
        "snapshot_id": snapshot_id,
        "snapshot_type": "clean-commit",
        "profile": {
            "profile_id": profile.profile_id,
            "sha256": profile.sha256,
            "target_host": profile.target_host,
            "risk_level": profile.risk_level,
        },
        "control_plane": {
            "minimum_version": control_plane_version,
        },
        "repositories": repository_payload,
        "repository_links": link_payload,
        "asset_lock": {
            "lock_id": asset_lock_id,
            "workload_id": asset_lock_workload_id,
            "sha256": asset_lock_sha256,
        },
        "assets": asset_payload,
        "resources": {
            "gpu_count": profile.gpu_count,
            "device_access": profile.device_access,
            "network_egress": profile.network_egress,
            "downloads_permitted": profile.downloads_permitted,
        },
        "service": {
            "service_name": profile.service_policy.service_name,
            "repository": profile.service_policy.repository,
            "environment_id": profile.service_policy.environment_id,
            "protocol": profile.service_policy.protocol,
            "launch_mode": profile.service_policy.launch_mode,
            "launch_command_id": profile.service_policy.launch_command_id,
            "endpoint_scheme": profile.service_policy.endpoint_scheme,
        },
        "command_ids": list(profile.command_ids),
    }
    idempotency_key = _sha256(content)
    return {
        **content,
        "request_id": f"Q-{idempotency_key[:24]}",
        "idempotency_key": idempotency_key,
    }


def validate_remote_request_identity(request: object) -> dict[str, object]:
    """Validate immutable request identity without granting execution authority."""

    if not isinstance(request, dict):
        raise LabError("remote request must be an object")
    if request.get("schema") != REMOTE_REQUEST_SCHEMA_ID:
        raise LabError("unsupported remote request schema")
    supplied_request_id = request.get("request_id")
    supplied_key = request.get("idempotency_key")
    if (
        not isinstance(supplied_request_id, str)
        or not isinstance(supplied_key, str)
        or not SHA256_PATTERN.fullmatch(supplied_key)
    ):
        raise LabError("remote request lacks request/idempotency identity")
    content = {
        key: value
        for key, value in request.items()
        if key not in {"request_id", "idempotency_key"}
    }
    if _sha256(content) != supplied_key:
        raise LabError("remote request idempotency hash mismatch")
    if supplied_request_id != f"Q-{supplied_key[:24]}":
        raise LabError("remote request ID mismatch")
    return request


def validate_remote_request(
    request: object,
    *,
    profile: RemoteProfile,
    control_plane_version: str,
) -> dict[str, object]:
    request = validate_remote_request_identity(request)
    repositories = request.get("repositories")
    assets = request.get("assets")
    asset_lock = request.get("asset_lock")
    if (
        not isinstance(repositories, Mapping)
        or not isinstance(assets, Mapping)
        or not isinstance(asset_lock, Mapping)
    ):
        raise LabError("remote request repositories/assets/asset lock must be objects")
    rebuilt = build_remote_request(
        task_id=str(request.get("task_id", "")),
        snapshot_id=str(request.get("snapshot_id", "")),
        repositories=repositories,
        repository_links=request.get("repository_links"),
        assets=assets,
        asset_lock=asset_lock,
        profile=profile,
        control_plane_version=control_plane_version,
    )
    if rebuilt != request:
        raise LabError("remote request differs from protected Profile/policy")
    return rebuilt
