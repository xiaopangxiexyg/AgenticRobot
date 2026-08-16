from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime

from .remote_request import (
    OID_PATTERN,
    SHA256_PATTERN,
    RemoteProfile,
    _sha256,
    validate_remote_request,
)
from .slot_readiness import validate_slot_readiness_receipt
from .workspace import LabError, validate_task_id

SERVICE_LAUNCH_SPEC_SCHEMA_ID = "lab.service-launch-spec.v1"
SERVICE_HEALTH_SCHEMA_ID = "lab.service-health.v1"
SERVICE_RECEIPT_SCHEMA_ID = "lab.service-receipt.v2"

_NONCE_PATTERN = re.compile(r"[A-Za-z0-9._-]{16,128}")


def _require_sha256(payload: Mapping[str, object], field: str, *, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise LabError(f"{label} has invalid {field}")
    return value


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise LabError("service health has invalid started_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("service health has invalid started_at") from exc
    if parsed.tzinfo is None:
        raise LabError("service health started_at must include a timezone")
    return value


def _launch_content(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in spec.items()
        if key not in {"service_id", "launch_spec_sha256"}
    }


def validate_service_launch_spec_identity(spec: object) -> dict[str, object]:
    """Validate the immutable identity of a launch spec without launching it."""

    if not isinstance(spec, dict):
        raise LabError("service launch spec must be an object")
    expected_keys = {
        "schema",
        "service_id",
        "request_id",
        "idempotency_key",
        "task_id",
        "snapshot_id",
        "slot_id",
        "remote_profile_sha256",
        "readiness_receipt_sha256",
        "service_name",
        "mode",
        "protocol",
        "launch_command_id",
        "source",
        "environment",
        "asset_lock",
        "assets",
        "gpu",
        "endpoint_policy",
        "launcher_nonce",
        "device_access",
        "execution_eligible",
        "launch_spec_sha256",
    }
    if set(spec) != expected_keys:
        raise LabError("service launch spec has an unexpected shape")
    if spec.get("schema") != SERVICE_LAUNCH_SPEC_SCHEMA_ID:
        raise LabError("unsupported service launch spec schema")
    for field in (
        "request_id",
        "task_id",
        "snapshot_id",
        "slot_id",
        "service_name",
        "launch_command_id",
    ):
        value = spec.get(field)
        if not isinstance(value, str):
            raise LabError(f"service launch spec has invalid {field}")
        validate_task_id(value)
    for field in (
        "idempotency_key",
        "remote_profile_sha256",
        "readiness_receipt_sha256",
        "launch_spec_sha256",
    ):
        _require_sha256(spec, field, label="service launch spec")
    nonce = spec.get("launcher_nonce")
    if not isinstance(nonce, str) or not _NONCE_PATTERN.fullmatch(nonce):
        raise LabError("service launch spec has invalid launcher nonce")
    if spec.get("mode") not in {"ephemeral", "attach-existing"}:
        raise LabError("service launch spec has unsupported mode")
    if not isinstance(spec.get("protocol"), str) or not spec["protocol"]:
        raise LabError("service launch spec has invalid protocol")
    if (
        spec.get("device_access") is not False
        or spec.get("execution_eligible") is not False
    ):
        raise LabError("service launch spec has an unsafe execution envelope")
    source = spec.get("source")
    if not isinstance(source, dict) or set(source) != {
        "repository",
        "commit",
        "tree_oid",
        "submodules",
    }:
        raise LabError("service launch spec has invalid source identity")
    if not OID_PATTERN.fullmatch(
        str(source.get("commit", ""))
    ) or not OID_PATTERN.fullmatch(str(source.get("tree_oid", ""))):
        raise LabError("service launch spec has invalid source OID")
    if not isinstance(source.get("submodules"), dict):
        raise LabError("service launch spec has invalid submodule identity")
    environment = spec.get("environment")
    if not isinstance(environment, dict) or set(environment) != {
        "environment_id",
        "readiness_environment_sha256",
    }:
        raise LabError("service launch spec has invalid environment identity")
    _require_sha256(
        environment,
        "readiness_environment_sha256",
        label="service launch environment",
    )
    assets = spec.get("assets")
    asset_lock = spec.get("asset_lock")
    if (
        not isinstance(asset_lock, dict)
        or set(asset_lock) != {"lock_id", "workload_id", "sha256"}
        or not isinstance(asset_lock.get("lock_id"), str)
        or (
            asset_lock.get("workload_id") is not None
            and not isinstance(asset_lock.get("workload_id"), str)
        )
    ):
        raise LabError("service launch spec has invalid asset lock identity")
    validate_task_id(str(asset_lock["lock_id"]))
    if asset_lock["workload_id"] is not None:
        validate_task_id(str(asset_lock["workload_id"]))
    _require_sha256(asset_lock, "sha256", label="service launch asset lock")
    if not isinstance(assets, dict) or not assets:
        raise LabError("service launch spec has invalid assets")
    for role, identity in assets.items():
        if (
            not isinstance(role, str)
            or not isinstance(identity, dict)
            or set(identity)
            != {
                "asset_id",
                "sha256",
            }
        ):
            raise LabError(f"service launch spec has invalid asset: {role}")
        _require_sha256(identity, "sha256", label=f"service launch asset {role}")
    gpu = spec.get("gpu")
    if not isinstance(gpu, dict) or set(gpu) != {
        "lease_id",
        "lease_receipt_sha256",
        "gpu_uuid",
    }:
        raise LabError("service launch spec has invalid GPU identity")
    if not all(isinstance(gpu.get(field), str) and gpu[field] for field in gpu):
        raise LabError("service launch spec has invalid GPU identity")
    if not SHA256_PATTERN.fullmatch(str(gpu.get("lease_receipt_sha256", ""))):
        raise LabError("service launch spec has invalid GPU lease receipt hash")
    endpoint_policy = spec.get("endpoint_policy")
    if not isinstance(endpoint_policy, dict) or set(endpoint_policy) != {
        "scheme",
        "host",
        "dynamic_port",
    }:
        raise LabError("service launch spec has invalid endpoint policy")
    if (
        endpoint_policy.get("scheme") not in {"http", "https", "ws", "wss"}
        or endpoint_policy.get("host") != "127.0.0.1"
        or endpoint_policy.get("dynamic_port") is not True
    ):
        raise LabError("service launch spec must use a dynamic loopback endpoint")
    content = _launch_content(spec)
    identity_hash = _sha256(content)
    if spec.get("service_id") != f"SVC-{identity_hash[:24]}":
        raise LabError("service launch spec ID mismatch")
    hashed = {**content, "service_id": spec["service_id"]}
    if spec.get("launch_spec_sha256") != _sha256(hashed):
        raise LabError("service launch spec hash mismatch")
    return spec


def build_service_launch_spec(
    request: object,
    *,
    remote_profile: RemoteProfile,
    readiness_receipt: object,
    control_plane_version: str,
    slot_id: str,
    launcher_nonce: str,
) -> dict[str, object]:
    """Build a protected, inert launch identity. This never starts a service."""

    validate_task_id(slot_id)
    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    readiness = validate_slot_readiness_receipt(
        readiness_receipt,
        request_id=str(validated_request["request_id"]),
        idempotency_key=str(validated_request["idempotency_key"]),
        slot_id=slot_id,
    )
    if readiness.get("remote_profile_sha256") != remote_profile.sha256:
        raise LabError("service launch readiness uses a different remote Profile")
    if readiness.get("control_plane_version") != control_plane_version:
        raise LabError("service launch readiness uses a different control plane")
    if not _NONCE_PATTERN.fullmatch(launcher_nonce):
        raise LabError("service launcher nonce must contain 16-128 safe characters")
    policy = remote_profile.service_policy
    request_service = validated_request.get("service")
    if not isinstance(request_service, dict):
        raise LabError("remote request lacks protected service policy")
    repositories = validated_request.get("repositories")
    environments = readiness.get("environments")
    assets = validated_request.get("assets")
    asset_lock = validated_request.get("asset_lock")
    if not isinstance(repositories, Mapping):
        raise LabError("service launch repository inputs are incomplete")
    if not isinstance(environments, Mapping):
        raise LabError("service launch environment inputs are incomplete")
    if not isinstance(assets, Mapping) or not isinstance(asset_lock, Mapping):
        raise LabError("service launch inputs are incomplete")
    source = repositories.get(policy.repository)
    environment = environments.get(policy.environment_id)
    if not isinstance(source, Mapping) or not isinstance(environment, Mapping):
        raise LabError("protected service source/environment is unavailable")
    content: dict[str, object] = {
        "schema": SERVICE_LAUNCH_SPEC_SCHEMA_ID,
        "request_id": validated_request["request_id"],
        "idempotency_key": validated_request["idempotency_key"],
        "task_id": validated_request["task_id"],
        "snapshot_id": validated_request["snapshot_id"],
        "slot_id": slot_id,
        "remote_profile_sha256": remote_profile.sha256,
        "readiness_receipt_sha256": readiness["receipt_sha256"],
        "service_name": policy.service_name,
        "mode": policy.launch_mode,
        "protocol": policy.protocol,
        "launch_command_id": policy.launch_command_id,
        "source": {
            "repository": policy.repository,
            "commit": source.get("commit"),
            "tree_oid": source.get("tree_oid"),
            "submodules": source.get("submodules"),
        },
        "environment": {
            "environment_id": policy.environment_id,
            "readiness_environment_sha256": _sha256(environment),
        },
        "asset_lock": dict(asset_lock),
        "assets": {
            role: {
                "asset_id": identity.get("asset_id"),
                "sha256": identity.get("sha256"),
            }
            for role, identity in assets.items()
            if isinstance(role, str) and isinstance(identity, Mapping)
        },
        "gpu": {
            "lease_id": readiness["gpu_lease_id"],
            "lease_receipt_sha256": readiness["gpu_lease_receipt_sha256"],
            "gpu_uuid": readiness["gpu_uuid"],
        },
        "endpoint_policy": {
            "scheme": policy.endpoint_scheme,
            "host": "127.0.0.1",
            "dynamic_port": True,
        },
        "launcher_nonce": launcher_nonce,
        "device_access": False,
        "execution_eligible": False,
    }
    identity_hash = _sha256(content)
    identified = {**content, "service_id": f"SVC-{identity_hash[:24]}"}
    spec = {**identified, "launch_spec_sha256": _sha256(identified)}
    return validate_service_launch_spec_identity(spec)


def validate_service_health(
    health: object,
    *,
    launch_spec: object,
) -> dict[str, object]:
    """Verify a launcher-authenticated service health identity against its spec."""

    spec = validate_service_launch_spec_identity(launch_spec)
    if not isinstance(health, dict):
        raise LabError("service health identity must be an object")
    expected_keys = {
        "schema",
        "service_id",
        "launcher_nonce",
        "managed",
        "mode",
        "protocol",
        "launch_command_id",
        "source",
        "environment",
        "asset_lock",
        "assets",
        "gpu",
        "endpoint",
        "process",
    }
    if set(health) != expected_keys:
        raise LabError("service health identity has an unexpected shape")
    if health.get("schema") != SERVICE_HEALTH_SCHEMA_ID:
        raise LabError("unsupported service health schema")
    if health.get("managed") is not True:
        raise LabError("unmanaged service is forbidden")
    for field in (
        "service_id",
        "launcher_nonce",
        "mode",
        "protocol",
        "launch_command_id",
        "source",
        "environment",
        "asset_lock",
        "assets",
        "gpu",
    ):
        if health.get(field) != spec.get(field):
            raise LabError(f"service health identity mismatch: {field}")
    endpoint = health.get("endpoint")
    policy = spec["endpoint_policy"]
    if not isinstance(endpoint, dict) or set(endpoint) != {"scheme", "host", "port"}:
        raise LabError("service health has invalid endpoint")
    if not isinstance(policy, dict):
        raise LabError("service launch spec has invalid endpoint policy")
    port = endpoint.get("port")
    if (
        endpoint.get("scheme") != policy["scheme"]
        or endpoint.get("host") != policy["host"]
        or not isinstance(port, int)
        or isinstance(port, bool)
        or not 1024 <= port <= 65535
    ):
        raise LabError("service health endpoint violates protected policy")
    process = health.get("process")
    if not isinstance(process, dict) or set(process) != {"pid", "started_at"}:
        raise LabError("service health has invalid process identity")
    pid = process.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise LabError("service health has invalid pid")
    _validate_timestamp(process.get("started_at"))
    return health


def build_service_receipt(
    health: object,
    *,
    launch_spec: object,
    managed_process_receipt: object,
) -> dict[str, object]:
    """Bind matching health to a protected launcher process attestation.

    Callers must validate the full managed process receipt against live OS state
    before invoking this identity-only builder. The public control-plane path does
    that through ``managed_service.build_attested_service_receipt``.
    """

    spec = validate_service_launch_spec_identity(launch_spec)
    validated_health = validate_service_health(health, launch_spec=spec)
    if not isinstance(managed_process_receipt, Mapping):
        raise LabError("service receipt requires a managed process attestation")
    process_receipt_sha256 = _require_sha256(
        managed_process_receipt,
        "receipt_sha256",
        label="managed service process receipt",
    )
    process_identity = {
        "pid": managed_process_receipt.get("pid"),
        "started_at": managed_process_receipt.get("started_at"),
        "proc_start_time_ticks": managed_process_receipt.get(
            "proc_start_time_ticks"
        ),
    }
    if validated_health.get("process") != {
        "pid": process_identity["pid"],
        "started_at": process_identity["started_at"],
    }:
        raise LabError("service health process differs from managed attestation")
    if validated_health.get("endpoint") != managed_process_receipt.get("endpoint"):
        raise LabError("service health endpoint differs from managed attestation")
    if (
        not isinstance(process_identity["pid"], int)
        or isinstance(process_identity["pid"], bool)
        or process_identity["pid"] <= 1
        or not isinstance(process_identity["proc_start_time_ticks"], int)
        or isinstance(process_identity["proc_start_time_ticks"], bool)
        or process_identity["proc_start_time_ticks"] <= 0
    ):
        raise LabError("managed service process attestation is invalid")
    _validate_timestamp(process_identity["started_at"])
    content: dict[str, object] = {
        "schema": SERVICE_RECEIPT_SCHEMA_ID,
        "service_id": spec["service_id"],
        "request_id": spec["request_id"],
        "task_id": spec["task_id"],
        "snapshot_id": spec["snapshot_id"],
        "slot_id": spec["slot_id"],
        "launch_spec_sha256": spec["launch_spec_sha256"],
        "remote_profile_sha256": spec["remote_profile_sha256"],
        "readiness_receipt_sha256": spec["readiness_receipt_sha256"],
        "service_name": spec["service_name"],
        "managed": True,
        "mode": spec["mode"],
        "protocol": spec["protocol"],
        "launch_command_id": spec["launch_command_id"],
        "source": spec["source"],
        "environment": spec["environment"],
        "asset_lock": spec["asset_lock"],
        "assets": spec["assets"],
        "gpu": spec["gpu"],
        "endpoint": validated_health["endpoint"],
        "process": process_identity,
        "managed_process_receipt_sha256": process_receipt_sha256,
        "health_observation_sha256": _sha256(validated_health),
        "service_identity_verified": True,
        "device_access": False,
        "execution_eligible": False,
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_service_receipt(receipt)


def validate_service_receipt(receipt: object) -> dict[str, object]:
    if not isinstance(receipt, dict):
        raise LabError("service receipt must be an object")
    expected_keys = {
        "schema",
        "service_id",
        "request_id",
        "task_id",
        "snapshot_id",
        "slot_id",
        "launch_spec_sha256",
        "remote_profile_sha256",
        "readiness_receipt_sha256",
        "service_name",
        "managed",
        "mode",
        "protocol",
        "launch_command_id",
        "source",
        "environment",
        "asset_lock",
        "assets",
        "gpu",
        "endpoint",
        "process",
        "managed_process_receipt_sha256",
        "health_observation_sha256",
        "service_identity_verified",
        "device_access",
        "execution_eligible",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise LabError("service receipt has an unexpected shape")
    if receipt.get("schema") != SERVICE_RECEIPT_SCHEMA_ID:
        raise LabError("unsupported service receipt schema")
    for field in (
        "launch_spec_sha256",
        "remote_profile_sha256",
        "readiness_receipt_sha256",
        "managed_process_receipt_sha256",
        "health_observation_sha256",
        "receipt_sha256",
    ):
        _require_sha256(receipt, field, label="service receipt")
    if (
        receipt.get("managed") is not True
        or receipt.get("service_identity_verified") is not True
        or receipt.get("device_access") is not False
        or receipt.get("execution_eligible") is not False
        or receipt.get("mode") not in {"ephemeral", "attach-existing"}
    ):
        raise LabError("service receipt has an unsafe identity envelope")
    for field in (
        "service_id",
        "request_id",
        "task_id",
        "snapshot_id",
        "slot_id",
        "service_name",
        "protocol",
        "launch_command_id",
    ):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise LabError(f"service receipt has invalid {field}")
    source = receipt.get("source")
    if not isinstance(source, dict) or set(source) != {
        "repository",
        "commit",
        "tree_oid",
        "submodules",
    }:
        raise LabError("service receipt has an invalid source identity")
    if not OID_PATTERN.fullmatch(
        str(source.get("commit", ""))
    ) or not OID_PATTERN.fullmatch(str(source.get("tree_oid", ""))):
        raise LabError("service receipt has an invalid source OID")
    environment = receipt.get("environment")
    if not isinstance(environment, dict) or set(environment) != {
        "environment_id",
        "readiness_environment_sha256",
    }:
        raise LabError("service receipt has an invalid environment identity")
    _require_sha256(
        environment,
        "readiness_environment_sha256",
        label="service receipt environment",
    )
    asset_lock = receipt.get("asset_lock")
    if (
        not isinstance(asset_lock, dict)
        or set(asset_lock) != {"lock_id", "workload_id", "sha256"}
        or not isinstance(asset_lock.get("lock_id"), str)
        or (
            asset_lock.get("workload_id") is not None
            and not isinstance(asset_lock.get("workload_id"), str)
        )
    ):
        raise LabError("service receipt has invalid asset lock identity")
    validate_task_id(str(asset_lock["lock_id"]))
    if asset_lock["workload_id"] is not None:
        validate_task_id(str(asset_lock["workload_id"]))
    _require_sha256(asset_lock, "sha256", label="service receipt asset lock")
    assets = receipt.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise LabError("service receipt has invalid assets")
    for role, identity in assets.items():
        if (
            not isinstance(role, str)
            or not isinstance(identity, dict)
            or set(identity)
            != {
                "asset_id",
                "sha256",
            }
        ):
            raise LabError(f"service receipt has invalid asset: {role}")
        _require_sha256(identity, "sha256", label=f"service receipt asset {role}")
    gpu = receipt.get("gpu")
    if not isinstance(gpu, dict) or set(gpu) != {
        "lease_id",
        "lease_receipt_sha256",
        "gpu_uuid",
    }:
        raise LabError("service receipt has invalid GPU identity")
    if not all(isinstance(gpu.get(field), str) and gpu[field] for field in gpu):
        raise LabError("service receipt has invalid GPU identity")
    if not SHA256_PATTERN.fullmatch(str(gpu.get("lease_receipt_sha256", ""))):
        raise LabError("service receipt has invalid GPU lease receipt hash")
    endpoint = receipt.get("endpoint")
    if (
        not isinstance(endpoint, dict)
        or set(endpoint) != {"scheme", "host", "port"}
        or endpoint.get("host") != "127.0.0.1"
        or endpoint.get("scheme") not in {"http", "https", "ws", "wss"}
        or not isinstance(endpoint.get("port"), int)
        or isinstance(endpoint.get("port"), bool)
        or not 1024 <= endpoint["port"] <= 65535
    ):
        raise LabError("service receipt has an unsafe endpoint")
    process = receipt.get("process")
    if not isinstance(process, dict) or set(process) != {
        "pid",
        "started_at",
        "proc_start_time_ticks",
    }:
        raise LabError("service receipt has an invalid process identity")
    if (
        not isinstance(process.get("pid"), int)
        or isinstance(process.get("pid"), bool)
        or process["pid"] <= 0
        or not isinstance(process.get("proc_start_time_ticks"), int)
        or isinstance(process.get("proc_start_time_ticks"), bool)
        or process["proc_start_time_ticks"] <= 0
    ):
        raise LabError("service receipt has an invalid pid")
    _validate_timestamp(process.get("started_at"))
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("service receipt hash mismatch")
    return receipt


def verify_attach_existing_service(
    launch_spec: object,
    existing_receipt: object,
) -> dict[str, object]:
    """Accept only a managed existing service with exact protected identity."""

    spec = validate_service_launch_spec_identity(launch_spec)
    if spec.get("mode") != "attach-existing":
        raise LabError("attach-existing requires a protected attach-existing Profile")
    receipt = validate_service_receipt(existing_receipt)
    if receipt.get("mode") != "attach-existing":
        raise LabError("existing service mode does not permit attachment")
    for field in (
        "service_name",
        "protocol",
        "launch_command_id",
        "source",
        "environment",
        "assets",
        "gpu",
    ):
        if receipt.get(field) != spec.get(field):
            raise LabError(f"existing service identity mismatch: {field}")
    endpoint = receipt["endpoint"]
    policy = spec["endpoint_policy"]
    if not isinstance(endpoint, dict) or not isinstance(policy, dict):
        raise LabError("existing service has invalid endpoint identity")
    if endpoint.get("host") != policy.get("host") or endpoint.get(
        "scheme"
    ) != policy.get("scheme"):
        raise LabError("existing service endpoint violates protected policy")
    return receipt
