from __future__ import annotations

from datetime import datetime, timezone

from .remote_dispatch import validate_remote_outbox
from .remote_request import SHA256_PATTERN, _sha256
from .runner_state import TERMINAL_STATES, validate_request_state_identity
from .transport import TransportProfile
from .workspace import LabError

REMOTE_STATE_OBSERVATION_SCHEMA_ID = "lab.remote-state-observation.v1"


def _latest_sequence(state: dict[str, object]) -> int:
    history = state.get("history")
    if (
        not isinstance(history, list)
        or not history
        or not isinstance(history[-1], dict)
    ):
        raise LabError("remote request state has no final history sequence")
    sequence = history[-1].get("sequence")
    if not isinstance(sequence, int):
        raise LabError("remote request state has an invalid final sequence")
    return sequence


def validate_remote_watch_result(
    payload: object,
    *,
    request: object,
    after_sequence: int,
) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "request_id",
        "after_sequence",
        "observed_sequence",
        "changed",
        "terminal",
        "request",
    }:
        raise LabError("remote watch result has an unexpected shape")
    state = validate_request_state_identity(payload.get("request"), request=request)
    sequence = _latest_sequence(state)
    if (
        payload.get("schema") != "lab.request-watch.v1"
        or payload.get("request_id") != state.get("request_id")
        or payload.get("after_sequence") != after_sequence
        or payload.get("observed_sequence") != sequence
        or payload.get("changed") is not (sequence > after_sequence)
        or payload.get("terminal") is not (state.get("state") in TERMINAL_STATES)
    ):
        raise LabError("remote watch result identity or sequence mismatch")
    return payload


def build_remote_state_observation(
    *,
    envelope: object,
    remote_state: object,
    profile: TransportProfile,
    observed_host_key_fingerprint: str,
) -> dict[str, object]:
    validated_envelope = validate_remote_outbox(envelope)
    request = validated_envelope.get("request")
    if not isinstance(request, dict):
        raise LabError("remote outbox lacks request")
    request_profile = request.get("profile")
    if not isinstance(request_profile, dict):
        raise LabError("remote request lacks Profile identity")
    if (
        profile.status != "ACTIVE"
        or profile.target_host != request_profile.get("target_host")
        or observed_host_key_fingerprint not in profile.host_key_fingerprints
    ):
        raise LabError("remote state observation requires the pinned request transport")
    state = validate_request_state_identity(remote_state, request=request)
    sequence = _latest_sequence(state)
    state_sha256 = _sha256(state)
    content: dict[str, object] = {
        "schema": REMOTE_STATE_OBSERVATION_SCHEMA_ID,
        "request_id": request["request_id"],
        "idempotency_key": request["idempotency_key"],
        "target_host": profile.target_host,
        "transport_id": profile.transport_id,
        "transport_profile_sha256": profile.sha256,
        "observed_host_key_fingerprint": observed_host_key_fingerprint,
        "outbox_envelope_sha256": validated_envelope["envelope_sha256"],
        "remote_state_sequence": sequence,
        "remote_state_sha256": state_sha256,
        "remote_state": state,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "execution_authorized": False,
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_remote_state_observation(
        receipt,
        envelope=validated_envelope,
        profile=profile,
    )


def validate_remote_state_observation(
    receipt: object,
    *,
    envelope: object,
    profile: TransportProfile,
) -> dict[str, object]:
    validated_envelope = validate_remote_outbox(envelope)
    request = validated_envelope.get("request")
    if not isinstance(request, dict):
        raise LabError("remote outbox lacks request")
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "request_id",
        "idempotency_key",
        "target_host",
        "transport_id",
        "transport_profile_sha256",
        "observed_host_key_fingerprint",
        "outbox_envelope_sha256",
        "remote_state_sequence",
        "remote_state_sha256",
        "remote_state",
        "observed_at",
        "execution_authorized",
        "receipt_sha256",
    }:
        raise LabError("remote state observation has an unexpected shape")
    if (
        receipt.get("schema") != REMOTE_STATE_OBSERVATION_SCHEMA_ID
        or receipt.get("request_id") != request.get("request_id")
        or receipt.get("idempotency_key") != request.get("idempotency_key")
        or receipt.get("target_host") != profile.target_host
        or receipt.get("transport_id") != profile.transport_id
        or receipt.get("transport_profile_sha256") != profile.sha256
        or receipt.get("outbox_envelope_sha256")
        != validated_envelope.get("envelope_sha256")
    ):
        raise LabError("remote state observation identity mismatch")
    if (
        profile.status != "ACTIVE"
        or receipt.get("observed_host_key_fingerprint")
        not in profile.host_key_fingerprints
        or receipt.get("execution_authorized") is not False
    ):
        raise LabError("remote state observation is not from the pinned transport")
    state = validate_request_state_identity(
        receipt.get("remote_state"), request=request
    )
    if receipt.get("remote_state_sequence") != _latest_sequence(state) or receipt.get(
        "remote_state_sha256"
    ) != _sha256(state):
        raise LabError("remote state observation content binding mismatch")
    for field in (
        "transport_profile_sha256",
        "outbox_envelope_sha256",
        "remote_state_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"remote state observation has invalid {field}")
    observed_at = receipt.get("observed_at")
    if not isinstance(observed_at, str) or not observed_at:
        raise LabError("remote state observation has invalid observed_at")
    try:
        timestamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("remote state observation has invalid observed_at") from exc
    if timestamp.tzinfo is None:
        raise LabError("remote state observation observed_at requires a timezone")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("remote state observation hash mismatch")
    return receipt
