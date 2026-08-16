from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .remote_dispatch import validate_remote_outbox
from .remote_inbox import validate_remote_acceptance
from .remote_request import SHA256_PATTERN, _sha256
from .workspace import LabError, validate_task_id

TRANSPORT_PROFILE_SCHEMA_ID = "lab.transport-profile.v1"
TRANSPORT_RECEIPT_SCHEMA_ID = "lab.transport-receipt.v1"
_FINGERPRINT_PATTERN = re.compile(r"SHA256:[A-Za-z0-9+/]{43}")
_SSH_ALIAS_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class TransportProfile:
    transport_id: str
    target_host: str
    status: str
    ssh_alias: str
    ssh_host: str
    ssh_port: int
    ssh_user: str
    host_key_fingerprints: tuple[str, ...]
    remote_projectctl: Path
    remote_inbox_root: Path
    remote_state_root: Path
    path: Path
    sha256: str


def _absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise LabError(f"transport Profile {label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise LabError(f"transport Profile {label} is unsafe")
    return path


def load_transport_profile(profile_root: Path, transport_id: str) -> TransportProfile:
    validate_task_id(transport_id)
    root = profile_root.expanduser().resolve()
    path = (root / f"{transport_id}.json").resolve()
    if path.parent != root or not path.is_file():
        raise LabError(f"unknown transport Profile: {transport_id} ({path})")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read transport Profile {path}: {exc}") from exc
    expected_keys = {
        "schema",
        "transport_id",
        "target_host",
        "kind",
        "status",
        "ssh_alias",
        "ssh_host",
        "ssh_port",
        "ssh_user",
        "host_key_fingerprints",
        "remote_projectctl",
        "remote_inbox_root",
        "remote_state_root",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise LabError(f"transport Profile has an unexpected shape: {path}")
    if (
        payload.get("schema") != TRANSPORT_PROFILE_SCHEMA_ID
        or payload.get("transport_id") != transport_id
        or payload.get("kind") != "ssh-stdio"
    ):
        raise LabError(f"unsupported transport Profile: {path}")
    target_host = payload.get("target_host")
    if not isinstance(target_host, str):
        raise LabError(f"transport Profile has invalid target host: {path}")
    validate_task_id(target_host)
    status = payload.get("status")
    if status not in {"ACTIVE", "UNRESOLVED"}:
        raise LabError(f"transport Profile has invalid status: {path}")
    ssh_alias = payload.get("ssh_alias")
    if not isinstance(ssh_alias, str) or not _SSH_ALIAS_PATTERN.fullmatch(ssh_alias):
        raise LabError(f"transport Profile has invalid SSH alias: {path}")
    ssh_host = payload.get("ssh_host")
    ssh_user = payload.get("ssh_user")
    ssh_port = payload.get("ssh_port")
    if not isinstance(ssh_host, str) or not _SSH_ALIAS_PATTERN.fullmatch(ssh_host):
        raise LabError(f"transport Profile has invalid SSH host: {path}")
    if not isinstance(ssh_user, str) or not _SSH_ALIAS_PATTERN.fullmatch(ssh_user):
        raise LabError(f"transport Profile has invalid SSH user: {path}")
    if (
        not isinstance(ssh_port, int)
        or isinstance(ssh_port, bool)
        or not 1 <= ssh_port <= 65535
    ):
        raise LabError(f"transport Profile has invalid SSH port: {path}")
    fingerprints = payload.get("host_key_fingerprints")
    if not isinstance(fingerprints, list) or not all(
        isinstance(value, str) and _FINGERPRINT_PATTERN.fullmatch(value)
        for value in fingerprints
    ):
        raise LabError(f"transport Profile has invalid host-key fingerprints: {path}")
    if len(fingerprints) != len(set(fingerprints)):
        raise LabError(f"transport Profile repeats host-key fingerprints: {path}")
    if (status == "ACTIVE") != (len(fingerprints) == 1):
        raise LabError(
            "ACTIVE transport requires exactly one pin and UNRESOLVED transport "
            f"forbids pins: {path}"
        )
    return TransportProfile(
        transport_id=transport_id,
        target_host=target_host,
        status=status,
        ssh_alias=ssh_alias,
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        ssh_user=ssh_user,
        host_key_fingerprints=tuple(fingerprints),
        remote_projectctl=_absolute_path(
            payload.get("remote_projectctl"),
            label="remote_projectctl",
        ),
        remote_inbox_root=_absolute_path(
            payload.get("remote_inbox_root"),
            label="remote_inbox_root",
        ),
        remote_state_root=_absolute_path(
            payload.get("remote_state_root"),
            label="remote_state_root",
        ),
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def validate_transport_receipt(
    receipt: object,
    *,
    envelope: object,
    acceptance: object,
    profile: TransportProfile,
) -> dict[str, object]:
    validated_envelope = validate_remote_outbox(envelope)
    request = validated_envelope["request"]
    if not isinstance(request, dict):
        raise LabError("transport envelope lacks request")
    request_profile = request.get("profile")
    if not isinstance(request_profile, dict):
        raise LabError("transport request lacks Profile identity")
    validated_acceptance = validate_remote_acceptance(
        acceptance,
        request_id=str(request["request_id"]),
        idempotency_key=str(request["idempotency_key"]),
        target_host=profile.target_host,
    )
    if not isinstance(receipt, dict):
        raise LabError("authenticated transport receipt must be an object")
    expected_keys = {
        "schema",
        "request_id",
        "target_host",
        "transport_id",
        "transport_profile_sha256",
        "ssh_alias",
        "observed_host_key_fingerprint",
        "outbox_envelope_sha256",
        "acceptance_receipt_sha256",
        "transport_authenticated",
        "remote_acknowledged",
        "execution_eligible",
        "authenticated_at",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise LabError("authenticated transport receipt has an unexpected shape")
    if (
        receipt.get("schema") != TRANSPORT_RECEIPT_SCHEMA_ID
        or receipt.get("request_id") != request.get("request_id")
        or receipt.get("target_host") != profile.target_host
        or receipt.get("transport_id") != profile.transport_id
        or receipt.get("transport_profile_sha256") != profile.sha256
        or receipt.get("ssh_alias") != profile.ssh_alias
    ):
        raise LabError("authenticated transport receipt identity mismatch")
    observed = receipt.get("observed_host_key_fingerprint")
    if profile.status != "ACTIVE" or observed not in profile.host_key_fingerprints:
        raise LabError("transport host key is not pinned by an ACTIVE Profile")
    if (
        receipt.get("outbox_envelope_sha256")
        != validated_envelope.get("envelope_sha256")
        or receipt.get("acceptance_receipt_sha256")
        != validated_acceptance.get("receipt_sha256")
        or validated_acceptance.get("remote_profile_sha256")
        != request_profile.get("sha256")
    ):
        raise LabError("authenticated transport receipt binding mismatch")
    for field in (
        "transport_profile_sha256",
        "outbox_envelope_sha256",
        "acceptance_receipt_sha256",
        "receipt_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise LabError(f"authenticated transport receipt has invalid {field}")
    if (
        receipt.get("transport_authenticated") is not True
        or receipt.get("remote_acknowledged") is not True
        or receipt.get("execution_eligible") is not False
    ):
        raise LabError("authenticated transport receipt overclaims execution")
    authenticated_at = receipt.get("authenticated_at")
    if not isinstance(authenticated_at, str) or not authenticated_at:
        raise LabError("authenticated transport receipt has invalid timestamp")
    try:
        timestamp = datetime.fromisoformat(authenticated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("authenticated transport receipt has invalid timestamp") from exc
    if timestamp.tzinfo is None:
        raise LabError("authenticated transport receipt timestamp requires a timezone")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("authenticated transport receipt hash mismatch")
    return receipt


def build_transport_receipt(
    *,
    envelope: object,
    acceptance: object,
    profile: TransportProfile,
    observed_host_key_fingerprint: str,
) -> dict[str, object]:
    validated_envelope = validate_remote_outbox(envelope)
    request = validated_envelope["request"]
    if not isinstance(request, dict):
        raise LabError("transport envelope lacks request")
    validated_acceptance = validate_remote_acceptance(
        acceptance,
        request_id=str(request["request_id"]),
        idempotency_key=str(request["idempotency_key"]),
        target_host=profile.target_host,
    )
    if (
        profile.status != "ACTIVE"
        or observed_host_key_fingerprint not in profile.host_key_fingerprints
    ):
        raise LabError("transport host key is not pinned by an ACTIVE Profile")
    content: dict[str, object] = {
        "schema": TRANSPORT_RECEIPT_SCHEMA_ID,
        "request_id": request["request_id"],
        "target_host": profile.target_host,
        "transport_id": profile.transport_id,
        "transport_profile_sha256": profile.sha256,
        "ssh_alias": profile.ssh_alias,
        "observed_host_key_fingerprint": observed_host_key_fingerprint,
        "outbox_envelope_sha256": validated_envelope["envelope_sha256"],
        "acceptance_receipt_sha256": validated_acceptance["receipt_sha256"],
        "transport_authenticated": True,
        "remote_acknowledged": True,
        "execution_eligible": False,
        "authenticated_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_transport_receipt(
        receipt,
        envelope=validated_envelope,
        acceptance=validated_acceptance,
        profile=profile,
    )
