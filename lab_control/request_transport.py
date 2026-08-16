from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from .run_requests import RunRequestStore, validate_run_request
from .workspace import LabError, validate_task_id

REQUEST_ENVELOPE_SCHEMA_ID = "lab.request-envelope.v1"
REQUEST_INGRESS_RECEIPT_SCHEMA_ID = "lab.request-ingress-receipt.v1"
REQUEST_INGRESS_RESULT_SCHEMA_ID = "lab.request-ingress-result.v1"
AUTHENTICATION_SCHEME = "hmac-sha256"
_MINIMUM_SECRET_BYTES = 32
_MAXIMUM_TTL_SECONDS = 24 * 60 * 60
_MAXIMUM_CLOCK_SKEW_SECONDS = 5 * 60


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _parse_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise LabError(f"invalid {label}: {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LabError(f"invalid {label}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LabError(f"{label} must include a UTC offset")
    return parsed


def _secret(value: object, *, key_id: str) -> bytes:
    if not isinstance(value, bytes) or len(value) < _MINIMUM_SECRET_BYTES:
        raise LabError(
            f"transport key {key_id} must contain at least "
            f"{_MINIMUM_SECRET_BYTES} bytes"
        )
    return value


def build_request_envelope(
    request: Mapping[str, object],
    *,
    sender_id: str,
    recipient_id: str,
    key_id: str,
    secret: bytes,
    issued_at: datetime | None = None,
    ttl_seconds: int = 3600,
) -> dict[str, object]:
    """Seal one immutable Run request without choosing a transport mechanism."""

    sender_id = validate_task_id(sender_id)
    recipient_id = validate_task_id(recipient_id)
    key_id = validate_task_id(key_id)
    secret = _secret(secret, key_id=key_id)
    if (
        not isinstance(ttl_seconds, int)
        or isinstance(ttl_seconds, bool)
        or not 1 <= ttl_seconds <= _MAXIMUM_TTL_SECONDS
    ):
        raise LabError(
            f"transport envelope TTL must be between 1 and "
            f"{_MAXIMUM_TTL_SECONDS} seconds"
        )
    validated_request = validate_run_request(dict(request))
    issued = issued_at or _utc_now()
    if issued.tzinfo is None or issued.utcoffset() is None:
        raise LabError("transport envelope issued_at must include a UTC offset")
    expires = issued + timedelta(seconds=ttl_seconds)
    content: dict[str, object] = {
        "schema": REQUEST_ENVELOPE_SCHEMA_ID,
        "message_type": "run-request",
        "sender_id": sender_id,
        "recipient_id": recipient_id,
        "issued_at": issued.isoformat(),
        "expires_at": expires.isoformat(),
        "request_id": validated_request["request_id"],
        "idempotency_key": validated_request["idempotency_key"],
        "payload_sha256": _sha256_json(validated_request),
        "payload": validated_request,
    }
    envelope_sha256 = _sha256_json(content)
    authenticated = {
        **content,
        "envelope_id": f"E-{envelope_sha256[:24]}",
        "envelope_sha256": envelope_sha256,
        "authentication": {
            "scheme": AUTHENTICATION_SCHEME,
            "key_id": key_id,
        },
    }
    return {
        **authenticated,
        "authentication_mac": hmac.new(
            secret, _canonical_json(authenticated), hashlib.sha256
        ).hexdigest(),
    }


def validate_request_envelope(
    payload: object,
    *,
    recipient_id: str,
    trusted_sender_keys: Mapping[str, Mapping[str, bytes]],
    observed_at: datetime | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    """Authenticate, time-bound and canonicalize a transport envelope."""

    expected = {
        "schema",
        "message_type",
        "sender_id",
        "recipient_id",
        "issued_at",
        "expires_at",
        "request_id",
        "idempotency_key",
        "payload_sha256",
        "payload",
        "envelope_id",
        "envelope_sha256",
        "authentication",
        "authentication_mac",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LabError("request transport envelope has an unexpected shape")
    recipient_id = validate_task_id(recipient_id)
    sender_id = validate_task_id(str(payload.get("sender_id", "")))
    if (
        payload.get("schema") != REQUEST_ENVELOPE_SCHEMA_ID
        or payload.get("message_type") != "run-request"
        or payload.get("recipient_id") != recipient_id
    ):
        raise LabError("request transport envelope identity mismatch")
    authentication = payload.get("authentication")
    if not isinstance(authentication, dict) or set(authentication) != {
        "scheme",
        "key_id",
    }:
        raise LabError("request transport authentication has an unexpected shape")
    key_id = validate_task_id(str(authentication.get("key_id", "")))
    if authentication.get("scheme") != AUTHENTICATION_SCHEME:
        raise LabError("unsupported request transport authentication scheme")
    sender_keys = trusted_sender_keys.get(sender_id)
    if sender_keys is None or key_id not in sender_keys:
        raise LabError(f"request transport sender/key is not trusted: {sender_id}/{key_id}")
    secret = _secret(sender_keys[key_id], key_id=key_id)

    request = validate_run_request(payload.get("payload"))
    if (
        payload.get("request_id") != request.get("request_id")
        or payload.get("idempotency_key") != request.get("idempotency_key")
        or payload.get("payload_sha256") != _sha256_json(request)
    ):
        raise LabError("request transport payload identity mismatch")
    content = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "envelope_id",
            "envelope_sha256",
            "authentication",
            "authentication_mac",
        }
    }
    envelope_sha256 = _sha256_json(content)
    if (
        payload.get("envelope_sha256") != envelope_sha256
        or payload.get("envelope_id") != f"E-{envelope_sha256[:24]}"
    ):
        raise LabError("request transport envelope hash mismatch")
    authenticated = {
        **content,
        "envelope_id": payload["envelope_id"],
        "envelope_sha256": envelope_sha256,
        "authentication": authentication,
    }
    authentication_mac = payload.get("authentication_mac")
    if not isinstance(authentication_mac, str) or not hmac.compare_digest(
        authentication_mac,
        hmac.new(secret, _canonical_json(authenticated), hashlib.sha256).hexdigest(),
    ):
        raise LabError("request transport authentication failed")

    issued_at = _parse_timestamp(payload.get("issued_at"), label="envelope issued_at")
    expires_at = _parse_timestamp(
        payload.get("expires_at"), label="envelope expires_at"
    )
    if expires_at <= issued_at:
        raise LabError("request transport envelope has an invalid validity window")
    if (expires_at - issued_at).total_seconds() > _MAXIMUM_TTL_SECONDS:
        raise LabError("request transport envelope validity window is too long")
    observed = observed_at or _utc_now()
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise LabError("transport observed_at must include a UTC offset")
    if issued_at > observed + timedelta(seconds=_MAXIMUM_CLOCK_SKEW_SECONDS):
        raise LabError("request transport envelope is issued too far in the future")
    if observed > expires_at:
        raise LabError("request transport envelope has expired")
    return request, {
        "envelope_id": str(payload["envelope_id"]),
        "envelope_sha256": envelope_sha256,
        "sender_id": sender_id,
        "recipient_id": recipient_id,
        "authentication_scheme": AUTHENTICATION_SCHEME,
        "key_id": key_id,
    }


class RequestEnvelopeIngress:
    """Replay-safe ingress for an already authenticated portable envelope."""

    def __init__(
        self,
        store: RunRequestStore,
        *,
        recipient_id: str,
        trusted_sender_keys: Mapping[str, Mapping[str, bytes]],
    ) -> None:
        self.store = store
        self.recipient_id = validate_task_id(recipient_id)
        self.trusted_sender_keys = trusted_sender_keys

    def ingest(
        self,
        envelope: object,
        *,
        observed_at: datetime | None = None,
    ) -> dict[str, object]:
        request, transport = validate_request_envelope(
            envelope,
            recipient_id=self.recipient_id,
            trusted_sender_keys=self.trusted_sender_keys,
            observed_at=observed_at,
        )
        request_id = str(request["request_id"])
        state, created = self.store.submit(request)
        if self.store.has_attachment(request_id, "transport-ingress"):
            receipt = self.store.load_attachment(request_id, "transport-ingress")
            if (
                receipt.get("schema") != REQUEST_INGRESS_RECEIPT_SCHEMA_ID
                or receipt.get("request_id") != request_id
                or receipt.get("envelope_id") != transport["envelope_id"]
                or receipt.get("envelope_sha256") != transport["envelope_sha256"]
            ):
                raise LabError("Run request was previously ingested from another envelope")
        else:
            accepted = observed_at or _utc_now()
            content: dict[str, object] = {
                "schema": REQUEST_INGRESS_RECEIPT_SCHEMA_ID,
                "request_id": request_id,
                "idempotency_key": request["idempotency_key"],
                **transport,
                "accepted_at": accepted.isoformat(),
            }
            receipt = {
                **content,
                "receipt_sha256": _sha256_json(content),
            }
            receipt, _ = self.store.put_attachment(
                request_id, "transport-ingress", receipt
            )
        receipt_content = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        if receipt.get("receipt_sha256") != _sha256_json(receipt_content):
            raise LabError("request transport ingress receipt hash mismatch")
        return {
            "schema": REQUEST_INGRESS_RESULT_SCHEMA_ID,
            "created": created,
            "request": request,
            "state": state,
            "receipt": receipt,
        }
