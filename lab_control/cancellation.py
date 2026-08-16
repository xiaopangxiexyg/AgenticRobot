from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from .execution_sandbox import validate_service_sandbox_receipt
from .gpu_lease import validate_gpu_release_receipt
from .managed_service import (
    validate_managed_service_process_receipt,
    validate_managed_service_teardown_receipt,
)
from .remote_request import SHA256_PATTERN, _sha256, validate_remote_request_identity
from .resource_reservation import validate_gpu_resource_reservation
from .service_identity import (
    validate_service_launch_spec_identity,
    validate_service_receipt,
)
from .workspace import LabError, validate_task_id

CANCEL_TEARDOWN_RECEIPT_SCHEMA_ID = "lab.cancel-teardown-receipt.v2"


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


def _validated_resource_chain(
    *,
    request: object,
    slot_id: str,
    resource_reservation_receipt: object | None,
    resource_release_receipt: object | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if resource_reservation_receipt is None:
        if resource_release_receipt is not None:
            raise LabError("GPU release cannot exist without a resource reservation")
        return None, None
    reservation = validate_gpu_resource_reservation(
        resource_reservation_receipt,
        request=request,
        slot_id=slot_id,
        require_active=False,
    )
    if resource_release_receipt is None:
        raise LabError("cancel teardown requires release of its reserved GPU")
    raw_lease = reservation.get("gpu_lease_receipt")
    if not isinstance(raw_lease, dict):
        raise LabError("cancel teardown resource reservation lacks a GPU lease")
    release = validate_gpu_release_receipt(
        resource_release_receipt,
        host_id=str(reservation["host_id"]),
        host_profile_sha256=str(reservation["host_profile_sha256"]),
        gpu_uuid=str(raw_lease.get("gpu_uuid", "")),
    )
    if (
        release.get("request_id") != reservation.get("request_id")
        or release.get("slot_id") != slot_id
        or release.get("lease_receipt_sha256") != raw_lease.get("receipt_sha256")
    ):
        raise LabError("cancel teardown GPU release differs from its reservation")
    return reservation, release


def _validated_service_chain(
    *,
    request: Mapping[str, object],
    slot_id: str,
    resource_reservation: dict[str, object] | None,
    service_launch_spec: object | None,
    service_receipt: object | None,
    service_sandbox_receipt: object | None,
    managed_process_receipt: object | None,
    managed_service_teardown_receipt: object | None,
) -> tuple[
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
]:
    supplied = (
        service_launch_spec,
        service_receipt,
        service_sandbox_receipt,
        managed_process_receipt,
        managed_service_teardown_receipt,
    )
    if all(value is None for value in supplied):
        return None, None, None, None, None
    if any(value is None for value in supplied):
        raise LabError(
            "partially started service cannot be cancelled without full teardown evidence"
        )
    if resource_reservation is None:
        raise LabError("managed service teardown requires its GPU reservation")
    spec = validate_service_launch_spec_identity(service_launch_spec)
    service = validate_service_receipt(service_receipt)
    sandbox = validate_service_sandbox_receipt(service_sandbox_receipt)
    process = validate_managed_service_process_receipt(
        managed_process_receipt,
        launch_spec=spec,
        service_sandbox_receipt=sandbox,
    )
    teardown = validate_managed_service_teardown_receipt(
        managed_service_teardown_receipt,
        launch_spec=spec,
        service_sandbox_receipt=sandbox,
        managed_process_receipt=process,
    )
    raw_lease = resource_reservation.get("gpu_lease_receipt")
    if not isinstance(raw_lease, dict):
        raise LabError("managed service teardown reservation lacks a GPU lease")
    service_gpu = service.get("gpu")
    spec_gpu = spec.get("gpu")
    if not isinstance(service_gpu, dict) or not isinstance(spec_gpu, dict):
        raise LabError("managed service teardown has invalid GPU identity")
    bindings = {
        "request_id": request["request_id"],
        "task_id": request["task_id"],
        "slot_id": slot_id,
        "launch_spec_sha256": spec["launch_spec_sha256"],
    }
    for field, expected in bindings.items():
        if service.get(field) != expected:
            raise LabError(f"cancel teardown service identity mismatch: {field}")
    if (
        spec.get("request_id") != request["request_id"]
        or spec.get("idempotency_key") != request["idempotency_key"]
        or spec.get("task_id") != request["task_id"]
        or spec.get("slot_id") != slot_id
        or service.get("managed_process_receipt_sha256")
        != process.get("receipt_sha256")
        or spec_gpu != service_gpu
        or spec_gpu.get("lease_receipt_sha256") != raw_lease.get("receipt_sha256")
        or spec_gpu.get("gpu_uuid") != raw_lease.get("gpu_uuid")
    ):
        raise LabError("cancel teardown service chain identity mismatch")
    return spec, service, sandbox, process, teardown


def validate_cancel_service_teardown_evidence(
    *,
    request: object,
    slot_id: str,
    resource_reservation_receipt: object,
    service_launch_spec: object,
    service_receipt: object,
    service_sandbox_receipt: object,
    managed_process_receipt: object,
    managed_service_teardown_receipt: object,
) -> dict[str, object]:
    """Validate full managed-service termination before releasing its GPU."""

    validated_request = validate_remote_request_identity(request)
    reservation = validate_gpu_resource_reservation(
        resource_reservation_receipt,
        request=validated_request,
        slot_id=slot_id,
        require_active=False,
    )
    _, service, _, _, teardown = _validated_service_chain(
        request=validated_request,
        slot_id=slot_id,
        resource_reservation=reservation,
        service_launch_spec=service_launch_spec,
        service_receipt=service_receipt,
        service_sandbox_receipt=service_sandbox_receipt,
        managed_process_receipt=managed_process_receipt,
        managed_service_teardown_receipt=managed_service_teardown_receipt,
    )
    if service is None or teardown is None:
        raise LabError("managed-service cancellation lacks teardown evidence")
    return teardown


def validate_cancel_teardown_receipt(
    receipt: object,
    *,
    request: object,
    slot_id: str,
    resource_reservation_receipt: object | None,
    resource_release_receipt: object | None,
    service_launch_spec: object | None,
    service_receipt: object | None,
) -> dict[str, object]:
    """Validate cancellation against exact process teardown and GPU release evidence."""

    validated_request = validate_remote_request_identity(request)
    validate_task_id(slot_id)
    reservation, release = _validated_resource_chain(
        request=validated_request,
        slot_id=slot_id,
        resource_reservation_receipt=resource_reservation_receipt,
        resource_release_receipt=resource_release_receipt,
    )
    if not isinstance(receipt, dict):
        raise LabError("cancel teardown receipt must be an object")
    expected_keys = {
        "schema",
        "request_id",
        "idempotency_key",
        "task_id",
        "slot_id",
        "resource_reservation_receipt_sha256",
        "gpu_release_receipt_sha256",
        "gpu_lease_acquired",
        "service_started",
        "service_receipt_sha256",
        "service_sandbox_receipt",
        "managed_process_receipt",
        "managed_service_teardown_receipt",
        "registered_process_group_terminated",
        "managed_service_absent",
        "gpu_lease_released",
        "device_access",
        "hardware_safety_claimed",
        "teardown_complete",
        "observed_at",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise LabError("cancel teardown receipt has an unexpected shape")
    resource_bindings = {
        "resource_reservation_receipt_sha256": (
            reservation.get("receipt_sha256") if reservation is not None else None
        ),
        "gpu_release_receipt_sha256": (
            release.get("receipt_sha256") if release is not None else None
        ),
        "gpu_lease_acquired": reservation is not None,
    }
    if (
        receipt.get("schema") != CANCEL_TEARDOWN_RECEIPT_SCHEMA_ID
        or receipt.get("request_id") != validated_request["request_id"]
        or receipt.get("idempotency_key") != validated_request["idempotency_key"]
        or receipt.get("task_id") != validated_request["task_id"]
        or receipt.get("slot_id") != slot_id
    ):
        raise LabError("cancel teardown receipt identity mismatch")
    for field, expected in resource_bindings.items():
        if receipt.get(field) != expected:
            raise LabError(f"cancel teardown resource identity mismatch: {field}")
    if not SHA256_PATTERN.fullmatch(str(receipt.get("receipt_sha256", ""))):
        raise LabError("cancel teardown receipt has an invalid hash")
    embedded_sandbox = receipt.get("service_sandbox_receipt")
    embedded_process = receipt.get("managed_process_receipt")
    embedded_teardown = receipt.get("managed_service_teardown_receipt")
    spec, service, _, _, teardown = _validated_service_chain(
        request=validated_request,
        slot_id=slot_id,
        resource_reservation=reservation,
        service_launch_spec=service_launch_spec,
        service_receipt=service_receipt,
        service_sandbox_receipt=embedded_sandbox,
        managed_process_receipt=embedded_process,
        managed_service_teardown_receipt=embedded_teardown,
    )
    service_started = service is not None
    if (
        receipt.get("service_started") is not service_started
        or receipt.get("service_receipt_sha256")
        != (service.get("receipt_sha256") if service is not None else None)
    ):
        raise LabError("cancel teardown service receipt identity mismatch")
    if service_started and spec is None:
        raise LabError("cancel teardown service launch identity is unavailable")
    if (
        receipt.get("registered_process_group_terminated") is not True
        or receipt.get("managed_service_absent") is not True
        or receipt.get("gpu_lease_released") is not True
        or receipt.get("device_access") is not False
        or receipt.get("hardware_safety_claimed") is not False
        or receipt.get("teardown_complete") is not True
    ):
        raise LabError("cancel teardown receipt is incomplete or overclaims safety")
    observed = _timestamp(receipt.get("observed_at"), label="cancel teardown")
    if release is not None:
        released_at = _timestamp(release.get("released_at"), label="GPU release")
        if released_at > observed:
            raise LabError("cancel teardown predates GPU release")
    if teardown is not None:
        teardown_at = _timestamp(
            teardown.get("observed_at"), label="managed service teardown"
        )
        if teardown_at > observed:
            raise LabError("cancel teardown predates managed service teardown")
        if release is not None and teardown_at > _timestamp(
            release.get("released_at"), label="GPU release"
        ):
            raise LabError("GPU release predates managed service teardown")
    content: Mapping[str, object] = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("cancel teardown receipt hash mismatch")
    return receipt


def build_cancel_teardown_receipt(
    request: object,
    *,
    slot_id: str,
    resource_reservation_receipt: object | None,
    resource_release_receipt: object | None,
    service_launch_spec: object | None = None,
    service_receipt: object | None = None,
    service_sandbox_receipt: object | None = None,
    managed_process_receipt: object | None = None,
    managed_service_teardown_receipt: object | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Build a non-authorizing cancellation receipt from already verified evidence."""

    validated_request = validate_remote_request_identity(request)
    validate_task_id(slot_id)
    reservation, release = _validated_resource_chain(
        request=validated_request,
        slot_id=slot_id,
        resource_reservation_receipt=resource_reservation_receipt,
        resource_release_receipt=resource_release_receipt,
    )
    _, service, sandbox, process, teardown = _validated_service_chain(
        request=validated_request,
        slot_id=slot_id,
        resource_reservation=reservation,
        service_launch_spec=service_launch_spec,
        service_receipt=service_receipt,
        service_sandbox_receipt=service_sandbox_receipt,
        managed_process_receipt=managed_process_receipt,
        managed_service_teardown_receipt=managed_service_teardown_receipt,
    )
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("cancel teardown clock requires a timezone")
    if release is not None and _timestamp(
        release.get("released_at"), label="GPU release"
    ) > current:
        raise LabError("cancel teardown predates GPU release")
    if teardown is not None and _timestamp(
        teardown.get("observed_at"), label="managed service teardown"
    ) > current:
        raise LabError("cancel teardown predates managed service teardown")
    content: dict[str, object] = {
        "schema": CANCEL_TEARDOWN_RECEIPT_SCHEMA_ID,
        "request_id": validated_request["request_id"],
        "idempotency_key": validated_request["idempotency_key"],
        "task_id": validated_request["task_id"],
        "slot_id": slot_id,
        "resource_reservation_receipt_sha256": (
            reservation.get("receipt_sha256") if reservation is not None else None
        ),
        "gpu_release_receipt_sha256": (
            release.get("receipt_sha256") if release is not None else None
        ),
        "gpu_lease_acquired": reservation is not None,
        "service_started": service is not None,
        "service_receipt_sha256": (
            service.get("receipt_sha256") if service is not None else None
        ),
        "service_sandbox_receipt": sandbox,
        "managed_process_receipt": process,
        "managed_service_teardown_receipt": teardown,
        "registered_process_group_terminated": True,
        "managed_service_absent": True,
        "gpu_lease_released": True,
        "device_access": False,
        "hardware_safety_claimed": False,
        "teardown_complete": True,
        "observed_at": current.isoformat(),
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_cancel_teardown_receipt(
        receipt,
        request=validated_request,
        slot_id=slot_id,
        resource_reservation_receipt=reservation,
        resource_release_receipt=release,
        service_launch_spec=service_launch_spec,
        service_receipt=service,
    )
