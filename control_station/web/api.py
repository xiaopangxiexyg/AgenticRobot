"""CS-1b local typed-intent HTTP boundary.

The web access token authorizes use of this local API only.  It is not an
operator control lease, authority lease, dispatch token, or motion permit.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..contracts import StationIntentDraft


CS1B_SAFE_INTENT_KINDS = (
    "acquire_control_lease",
    "release_control_lease",
    "connect_source",
    "disconnect_source",
    "connect_target",
    "disconnect_target",
    "run_preflight",
    "start_shadow",
    "stop_shadow",
    "hold",
    "estop",
    "stop_runtime",
)
SCOPED_SIM_AUTHORITY_INTENT_KINDS_V1 = (
    "acquire_control_lease",
    "release_control_lease",
    "connect_source",
    "disconnect_source",
    "connect_target",
    "disconnect_target",
    "run_preflight",
    "request_candidate",
    "release_candidate",
    "request_motion_authority",
    "release_motion_authority",
    "hold",
    "stop_runtime",
)

_INTENT_FIELDS = frozenset(
    {
        "client_id",
        "client_sequence",
        "intent_kind",
        "produced_at",
        "produced_clock_domain_id",
        "expected_state_revision",
        "manifest_hash",
        "subject_id",
        "resource_binding_id",
        "operation_spec_id",
        "reason",
    }
)
_REQUIRED_INTENT_FIELDS = frozenset(
    {
        "client_id",
        "client_sequence",
        "intent_kind",
        "produced_at",
        "produced_clock_domain_id",
        "expected_state_revision",
        "manifest_hash",
    }
)
_STRING_FIELDS = frozenset(
    {
        "client_id",
        "intent_kind",
        "produced_clock_domain_id",
        "manifest_hash",
        "reason",
    }
)
_OPTIONAL_STRING_FIELDS = frozenset(
    {"subject_id", "resource_binding_id", "operation_spec_id"}
)
_INTEGER_FIELDS = frozenset({"client_sequence", "expected_state_revision"})


@dataclass(frozen=True)
class LocalIntentAccessPolicy:
    """Explicit local API configuration; never a control authorization fact."""

    access_policy_id: str
    bearer_token: str = field(repr=False)
    allowed_intent_kinds: tuple[str, ...] = CS1B_SAFE_INTENT_KINDS
    max_request_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        policy_id = str(self.access_policy_id).strip()
        if not policy_id:
            raise ValueError("access_policy_id must be non-empty.")
        if not isinstance(self.bearer_token, str):
            raise TypeError("bearer_token must be a string.")
        token = self.bearer_token
        if len(token) < 32:
            raise ValueError("bearer_token must contain at least 32 characters.")
        if token.strip() != token or any(character.isspace() for character in token):
            raise ValueError("bearer_token cannot contain whitespace.")
        try:
            token.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("bearer_token must be ASCII.") from error
        allowed = tuple(str(kind).strip() for kind in self.allowed_intent_kinds)
        if not allowed or any(not kind for kind in allowed):
            raise ValueError("allowed_intent_kinds must be non-empty.")
        if len(set(allowed)) != len(allowed):
            raise ValueError("allowed_intent_kinds cannot contain duplicates.")
        unsupported = sorted(set(allowed) - set(CS1B_SAFE_INTENT_KINDS))
        if unsupported:
            raise ValueError(
                f"CS-1b access policy contains unsafe intents: {unsupported}."
            )
        if (
            isinstance(self.max_request_bytes, bool)
            or not isinstance(self.max_request_bytes, int)
            or not 256 <= self.max_request_bytes <= 64 * 1024
        ):
            raise ValueError("max_request_bytes must be an integer in [256, 65536].")
        object.__setattr__(self, "access_policy_id", policy_id)
        object.__setattr__(self, "bearer_token", token)
        object.__setattr__(self, "allowed_intent_kinds", tuple(sorted(allowed)))
        object.__setattr__(self, "max_request_bytes", self.max_request_bytes)

    @property
    def token_fingerprint(self) -> str:
        return hashlib.sha256(self.bearer_token.encode("utf-8")).hexdigest()

    def accepts_authorization_header(self, authorization: str | None) -> bool:
        if authorization is None:
            return False
        scheme, separator, supplied = authorization.partition(" ")
        if separator != " " or scheme.lower() != "bearer":
            return False
        return hmac.compare_digest(supplied, self.bearer_token)


@dataclass(frozen=True)
class ScopedSimAuthorityAccessPolicy:
    """Local API access for the explicit scoped-sim authority builder.

    The bearer token authorizes HTTP use only. Candidate/authority still
    require the Station lease, typed adapter completion, publication barrier,
    and the existing sim runtime owner/epoch checks.
    """

    access_policy_id: str
    bearer_token: str = field(repr=False)
    allowed_intent_kinds: tuple[str, ...] = SCOPED_SIM_AUTHORITY_INTENT_KINDS_V1
    max_request_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        policy_id = str(self.access_policy_id).strip()
        if not policy_id:
            raise ValueError("access_policy_id must be non-empty.")
        if not isinstance(self.bearer_token, str):
            raise TypeError("bearer_token must be a string.")
        token = self.bearer_token
        if len(token) < 32:
            raise ValueError("bearer_token must contain at least 32 characters.")
        if token.strip() != token or any(character.isspace() for character in token):
            raise ValueError("bearer_token cannot contain whitespace.")
        try:
            token.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("bearer_token must be ASCII.") from error
        allowed = tuple(str(kind).strip() for kind in self.allowed_intent_kinds)
        if not allowed or any(not kind for kind in allowed):
            raise ValueError("allowed_intent_kinds must be non-empty.")
        if len(set(allowed)) != len(allowed):
            raise ValueError("allowed_intent_kinds cannot contain duplicates.")
        unsupported = sorted(
            set(allowed) - set(SCOPED_SIM_AUTHORITY_INTENT_KINDS_V1)
        )
        if unsupported:
            raise ValueError(
                "Scoped sim access policy contains unsupported intents: "
                f"{unsupported}."
            )
        if (
            isinstance(self.max_request_bytes, bool)
            or not isinstance(self.max_request_bytes, int)
            or not 256 <= self.max_request_bytes <= 64 * 1024
        ):
            raise ValueError("max_request_bytes must be an integer in [256, 65536].")
        object.__setattr__(self, "access_policy_id", policy_id)
        object.__setattr__(self, "bearer_token", token)
        object.__setattr__(self, "allowed_intent_kinds", tuple(sorted(allowed)))

    @property
    def token_fingerprint(self) -> str:
        return hashlib.sha256(self.bearer_token.encode("utf-8")).hexdigest()

    def accepts_authorization_header(self, authorization: str | None) -> bool:
        if authorization is None:
            return False
        scheme, separator, supplied = authorization.partition(" ")
        if separator != " " or scheme.lower() != "bearer":
            return False
        return hmac.compare_digest(supplied, self.bearer_token)


def station_intent_draft_from_json(
    payload: Mapping[str, Any],
    *,
    access_policy: LocalIntentAccessPolicy | ScopedSimAuthorityAccessPolicy,
) -> StationIntentDraft:
    """Validate an exact JSON shape before constructing the typed draft."""

    if not isinstance(payload, Mapping):
        raise TypeError("Intent payload must be a JSON object.")
    keys = frozenset(payload)
    unknown = sorted(keys - _INTENT_FIELDS)
    missing = sorted(_REQUIRED_INTENT_FIELDS - keys)
    if unknown:
        raise ValueError(f"Intent payload contains unknown fields: {unknown}.")
    if missing:
        raise ValueError(f"Intent payload is missing required fields: {missing}.")

    for field_name in _STRING_FIELDS:
        if field_name in payload and not isinstance(payload[field_name], str):
            raise TypeError(f"{field_name} must be a string.")
    for field_name in _OPTIONAL_STRING_FIELDS:
        value = payload.get(field_name)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string or null.")
    for field_name in _INTEGER_FIELDS:
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer.")
    produced_at = payload["produced_at"]
    if isinstance(produced_at, bool) or not isinstance(produced_at, (int, float)):
        raise TypeError("produced_at must be a number.")

    intent_kind = payload["intent_kind"]
    if intent_kind not in access_policy.allowed_intent_kinds:
        raise PermissionError(f"Intent kind is not enabled by CS-1b: {intent_kind!r}.")

    return StationIntentDraft(
        client_id=payload["client_id"],
        client_sequence=payload["client_sequence"],
        intent_kind=intent_kind,
        produced_at=produced_at,
        produced_clock_domain_id=payload["produced_clock_domain_id"],
        expected_state_revision=payload["expected_state_revision"],
        manifest_hash=payload["manifest_hash"],
        subject_id=payload.get("subject_id"),
        resource_binding_id=payload.get("resource_binding_id"),
        operation_spec_id=payload.get("operation_spec_id"),
        reason=payload.get("reason", "operator_request"),
    )


__all__ = [
    "CS1B_SAFE_INTENT_KINDS",
    "LocalIntentAccessPolicy",
    "SCOPED_SIM_AUTHORITY_INTENT_KINDS_V1",
    "ScopedSimAuthorityAccessPolicy",
    "station_intent_draft_from_json",
]
