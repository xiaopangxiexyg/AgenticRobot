from __future__ import annotations

import hashlib
import json

from .gpu_lease import (
    require_gpu_compute_processes_absent,
    validate_gpu_inventory,
    validate_gpu_lease_receipt,
)
from .remote_request import SHA256_PATTERN, validate_remote_request_identity
from .workspace import LabError, validate_task_id

RESOURCE_RESERVATION_SCHEMA_ID = "lab.resource-reservation.v2"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def build_gpu_resource_reservation(
    request: object,
    *,
    slot_id: str,
    host_id: str,
    host_profile_sha256: str,
    remote_profile_sha256: str,
    inventory_receipt: object,
    process_inventory_receipt: object,
    lease_receipt: object,
) -> dict[str, object]:
    """Bind one request slot and one persistent GPU lease without authorizing execution."""

    validated_request = validate_remote_request_identity(request)
    validate_task_id(slot_id)
    validate_task_id(host_id)
    if not SHA256_PATTERN.fullmatch(host_profile_sha256):
        raise LabError("resource reservation requires a host Profile SHA-256")
    if not SHA256_PATTERN.fullmatch(remote_profile_sha256):
        raise LabError("resource reservation requires a remote Profile SHA-256")
    inventory = validate_gpu_inventory(inventory_receipt)
    gpus = inventory["gpus"]
    assert isinstance(gpus, list)
    if len(gpus) != 1 or not isinstance(gpus[0], dict):
        raise LabError("resource reservation requires exactly one inventoried GPU")
    gpu = gpus[0]
    gpu_uuid = str(gpu.get("gpu_uuid", ""))
    process_inventory = require_gpu_compute_processes_absent(
        process_inventory_receipt,
        gpu_uuid=gpu_uuid,
    )
    lease = validate_gpu_lease_receipt(
        lease_receipt,
        host_id=host_id,
        gpu_uuid=gpu_uuid,
        require_active=True,
        host_profile_sha256=host_profile_sha256,
    )
    if (
        lease.get("gpu_id") != gpu.get("gpu_id")
        or lease.get("request_id") != validated_request.get("request_id")
        or lease.get("slot_id") != slot_id
    ):
        raise LabError("GPU lease does not match its request, slot, or inventory")
    content: dict[str, object] = {
        "schema": RESOURCE_RESERVATION_SCHEMA_ID,
        "request_id": validated_request["request_id"],
        "idempotency_key": validated_request["idempotency_key"],
        "task_id": validated_request["task_id"],
        "slot_id": slot_id,
        "host_id": host_id,
        "host_profile_sha256": host_profile_sha256,
        "remote_profile_sha256": remote_profile_sha256,
        "gpu_inventory_receipt": inventory,
        "gpu_process_inventory_receipt": process_inventory,
        "gpu_lease_receipt": lease,
        "execution_eligible": False,
        "device_access": False,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def validate_gpu_resource_reservation(
    receipt: object,
    *,
    request: object,
    slot_id: str,
    host_id: str | None = None,
    host_profile_sha256: str | None = None,
    remote_profile_sha256: str | None = None,
    require_active: bool = True,
) -> dict[str, object]:
    validated_request = validate_remote_request_identity(request)
    expected = {
        "schema",
        "request_id",
        "idempotency_key",
        "task_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "remote_profile_sha256",
        "gpu_inventory_receipt",
        "gpu_process_inventory_receipt",
        "gpu_lease_receipt",
        "execution_eligible",
        "device_access",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("resource reservation receipt has an unexpected shape")
    bindings = {
        "request_id": validated_request["request_id"],
        "idempotency_key": validated_request["idempotency_key"],
        "task_id": validated_request["task_id"],
        "slot_id": slot_id,
    }
    for field, expected_value in bindings.items():
        if receipt.get(field) != expected_value:
            raise LabError(f"resource reservation identity mismatch: {field}")
    if (
        receipt.get("schema") != RESOURCE_RESERVATION_SCHEMA_ID
        or receipt.get("execution_eligible") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("resource reservation has an unsafe authority envelope")
    for field, expected_value in (
        ("host_id", host_id),
        ("host_profile_sha256", host_profile_sha256),
        ("remote_profile_sha256", remote_profile_sha256),
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or not value:
            raise LabError(f"resource reservation has invalid {field}")
        if expected_value is not None and value != expected_value:
            raise LabError(f"resource reservation identity mismatch: {field}")
    for field in ("host_profile_sha256", "remote_profile_sha256"):
        if not SHA256_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"resource reservation has invalid {field}")
    inventory = validate_gpu_inventory(receipt.get("gpu_inventory_receipt"))
    gpus = inventory["gpus"]
    assert isinstance(gpus, list)
    if len(gpus) != 1 or not isinstance(gpus[0], dict):
        raise LabError("resource reservation requires exactly one inventoried GPU")
    gpu = gpus[0]
    require_gpu_compute_processes_absent(
        receipt.get("gpu_process_inventory_receipt"),
        gpu_uuid=str(gpu.get("gpu_uuid", "")),
    )
    lease = validate_gpu_lease_receipt(
        receipt.get("gpu_lease_receipt"),
        host_id=str(receipt["host_id"]),
        gpu_uuid=str(gpu.get("gpu_uuid", "")),
        require_active=require_active,
        host_profile_sha256=str(receipt["host_profile_sha256"]),
    )
    if (
        lease.get("gpu_id") != gpu.get("gpu_id")
        or lease.get("request_id") != validated_request.get("request_id")
        or lease.get("slot_id") != slot_id
    ):
        raise LabError("resource reservation GPU lease identity mismatch")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != hashlib.sha256(
        _canonical_json(content)
    ).hexdigest():
        raise LabError("resource reservation receipt hash mismatch")
    return receipt
