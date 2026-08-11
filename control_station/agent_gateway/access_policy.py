"""Pure application authorization for Agent Skill Route requests.

An accepted decision means only that a request is eligible for later Station
and lower-runtime admission.  Neither an access policy nor a
``SkillExecutionGrant`` is an operator lease, authority lease, dispatch token,
motion permit, or evidence that a lower operation ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash

from .contracts import (
    AGENT_SKILL_ACCESS_DECISION_HASH_DOMAIN_V1,
    AGENT_SKILL_ACCESS_POLICY_HASH_DOMAIN_V1,
    AGENT_SKILL_EXECUTION_GRANT_HASH_DOMAIN_V1,
    AgentSkillRequest,
    _boolean,
    _enum,
    _finite,
    _nonnegative_integer,
    _optional,
    _positive_integer,
    _required,
    _unique_sorted,
)
from .skill_catalog import (
    AGENT_SKILL_CONTROL_EFFECT_CLASSES_V1,
    AGENT_SKILL_MODES_V1,
    AGENT_SKILL_PRIVILEGE_TIERS_V1,
    MAX_AGENT_SKILL_TIMEOUT_MS_V1,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    SkillExecutionRoute,
    validate_route_id,
    validate_skill_id,
    validate_version,
)


AGENT_SKILL_GRANT_REQUIRED_TIERS_V1 = (
    "t2_real_motion",
    "t3_lifecycle",
)


@dataclass(frozen=True)
class SkillExecutionGrant:
    """Server-side immutable scope created from operator authorization."""

    operator_authorization_receipt_id: str
    client_id: str
    runtime_session_id: str
    skill_route_manifest_hash: str
    control_domain_id: str
    target_binding_id: str
    skill_id: str
    skill_version: str
    skill_definition_hash: str
    route_id: str
    route_version: str
    route_hash: str
    deployment_binding_hash: str
    parameter_profile_id: str
    parameter_profile_hash: str
    authorized_request_hash: str | None
    allowed_mode: str
    control_effect_class: str
    may_move_robot: bool
    max_invocations: int
    issued_at: float
    expires_at: float
    grant_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "operator_authorization_receipt_id",
            "client_id",
            "runtime_session_id",
            "skill_route_manifest_hash",
            "control_domain_id",
            "target_binding_id",
            "skill_definition_hash",
            "route_hash",
            "deployment_binding_hash",
            "parameter_profile_id",
            "parameter_profile_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "skill_id", validate_skill_id(self.skill_id))
        object.__setattr__(
            self,
            "skill_version",
            validate_version(self.skill_version, field_name="skill_version"),
        )
        object.__setattr__(self, "route_id", validate_route_id(self.route_id))
        object.__setattr__(
            self,
            "route_version",
            validate_version(self.route_version, field_name="route_version"),
        )
        object.__setattr__(
            self,
            "authorized_request_hash",
            _optional(
                self.authorized_request_hash,
                field_name="authorized_request_hash",
            ),
        )
        object.__setattr__(
            self,
            "allowed_mode",
            _enum(
                self.allowed_mode,
                field_name="allowed_mode",
                allowed=AGENT_SKILL_MODES_V1,
            ),
        )
        object.__setattr__(
            self,
            "control_effect_class",
            _enum(
                self.control_effect_class,
                field_name="control_effect_class",
                allowed=AGENT_SKILL_CONTROL_EFFECT_CLASSES_V1,
            ),
        )
        object.__setattr__(
            self,
            "may_move_robot",
            _boolean(self.may_move_robot, field_name="may_move_robot"),
        )
        object.__setattr__(
            self,
            "max_invocations",
            _positive_integer(self.max_invocations, field_name="max_invocations"),
        )
        issued = _finite(self.issued_at, field_name="issued_at")
        expires = _finite(self.expires_at, field_name="expires_at")
        if expires <= issued:
            raise ValueError("Skill execution grant must expire after issuance.")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(
            self,
            "grant_id",
            "agent-skill-grant:" + self.semantic_content_hash,
        )

    def active_at(self, timestamp: float) -> bool:
        current = _finite(timestamp, field_name="timestamp")
        return self.issued_at <= current < self.expires_at

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self._semantic_payload(),
            domain=AGENT_SKILL_EXECUTION_GRANT_HASH_DOMAIN_V1,
        )

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "operator_authorization_receipt_id": (
                self.operator_authorization_receipt_id
            ),
            "client_id": self.client_id,
            "runtime_session_id": self.runtime_session_id,
            "skill_route_manifest_hash": self.skill_route_manifest_hash,
            "control_domain_id": self.control_domain_id,
            "target_binding_id": self.target_binding_id,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "skill_definition_hash": self.skill_definition_hash,
            "route_id": self.route_id,
            "route_version": self.route_version,
            "route_hash": self.route_hash,
            "deployment_binding_hash": self.deployment_binding_hash,
            "parameter_profile_id": self.parameter_profile_id,
            "parameter_profile_hash": self.parameter_profile_hash,
            "authorized_request_hash": self.authorized_request_hash,
            "allowed_mode": self.allowed_mode,
            "control_effect_class": self.control_effect_class,
            "may_move_robot": self.may_move_robot,
            "max_invocations": self.max_invocations,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    def to_wire(self) -> dict[str, object]:
        return {
            "grant_id": self.grant_id,
            **self._semantic_payload(),
            "grant_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class AgentSkillAccessPolicy:
    """Narrow client/mode/timeout policy, separate from transport auth."""

    access_policy_id: str
    allowed_client_ids: tuple[str, ...]
    allowed_modes: tuple[str, ...]
    max_request_timeout_ms: int
    max_grant_lifetime_sec: float
    max_grant_invocations: int = 1
    grant_required_tiers: tuple[str, ...] = AGENT_SKILL_GRANT_REQUIRED_TIERS_V1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "access_policy_id",
            _required(self.access_policy_id, field_name="access_policy_id"),
        )
        object.__setattr__(
            self,
            "allowed_client_ids",
            _unique_sorted(
                self.allowed_client_ids,
                field_name="allowed_client_ids",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "allowed_modes",
            _unique_sorted(
                self.allowed_modes,
                field_name="allowed_modes",
                allowed=AGENT_SKILL_MODES_V1,
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "max_request_timeout_ms",
            _positive_integer(
                self.max_request_timeout_ms,
                field_name="max_request_timeout_ms",
                maximum=MAX_AGENT_SKILL_TIMEOUT_MS_V1,
            ),
        )
        grant_lifetime = _finite(
            self.max_grant_lifetime_sec,
            field_name="max_grant_lifetime_sec",
        )
        if grant_lifetime <= 0.0:
            raise ValueError("max_grant_lifetime_sec must be positive.")
        object.__setattr__(self, "max_grant_lifetime_sec", grant_lifetime)
        object.__setattr__(
            self,
            "max_grant_invocations",
            _positive_integer(
                self.max_grant_invocations,
                field_name="max_grant_invocations",
            ),
        )
        grant_tiers = _unique_sorted(
            self.grant_required_tiers,
            field_name="grant_required_tiers",
            allowed=AGENT_SKILL_PRIVILEGE_TIERS_V1,
        )
        missing_required = sorted(
            set(AGENT_SKILL_GRANT_REQUIRED_TIERS_V1) - set(grant_tiers)
        )
        if missing_required:
            raise ValueError(
                "Agent skill protocol v1 cannot disable grants for tiers: "
                f"{missing_required}."
            )
        object.__setattr__(self, "grant_required_tiers", grant_tiers)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "access_policy_id": self.access_policy_id,
                "allowed_client_ids": self.allowed_client_ids,
                "allowed_modes": self.allowed_modes,
                "max_request_timeout_ms": self.max_request_timeout_ms,
                "max_grant_lifetime_sec": self.max_grant_lifetime_sec,
                "max_grant_invocations": self.max_grant_invocations,
                "grant_required_tiers": self.grant_required_tiers,
            },
            domain=AGENT_SKILL_ACCESS_POLICY_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class SkillAccessDecision:
    """Pure access result; acceptance is not operation admission."""

    accepted: bool
    reason_code: str
    request_hash: str
    manifest_hash: str
    access_policy_hash: str
    route_manifest_entry_hash: str | None = None
    grant_hash: str | None = None
    effective_timeout_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted",
            _boolean(self.accepted, field_name="accepted"),
        )
        for field_name in (
            "reason_code",
            "request_hash",
            "manifest_hash",
            "access_policy_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("route_manifest_entry_hash", "grant_hash"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _required(value, field_name=field_name),
                )
        if self.effective_timeout_ms is not None:
            object.__setattr__(
                self,
                "effective_timeout_ms",
                _positive_integer(
                    self.effective_timeout_ms,
                    field_name="effective_timeout_ms",
                    maximum=MAX_AGENT_SKILL_TIMEOUT_MS_V1,
                ),
            )
        if self.accepted:
            if self.reason_code != "accepted":
                raise ValueError("Accepted decisions require reason_code='accepted'.")
            if (
                self.route_manifest_entry_hash is None
                or self.effective_timeout_ms is None
            ):
                raise ValueError(
                    "Accepted decisions require Route and timeout evidence."
                )
        elif self.reason_code == "accepted":
            raise ValueError("Rejected decisions cannot use reason_code='accepted'.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "accepted": self.accepted,
                "reason_code": self.reason_code,
                "request_hash": self.request_hash,
                "manifest_hash": self.manifest_hash,
                "access_policy_hash": self.access_policy_hash,
                "route_manifest_entry_hash": self.route_manifest_entry_hash,
                "grant_hash": self.grant_hash,
                "effective_timeout_ms": self.effective_timeout_ms,
            },
            domain=AGENT_SKILL_ACCESS_DECISION_HASH_DOMAIN_V1,
        )


def _request_matches_entry(
    request: AgentSkillRequest,
    entry: SkillRouteManifestEntry,
) -> bool:
    route = entry.route
    binding = entry.deployment_binding
    return all(
        (
            request.skill_id == route.skill_id,
            request.skill_version == route.skill_version,
            request.skill_definition_hash == route.skill_definition_hash,
            request.route_id == route.route_id,
            request.route_version == route.route_version,
            request.route_hash == route.semantic_content_hash,
            request.deployment_binding_hash == binding.deployment_binding_hash,
        )
    )


def issue_skill_execution_grant(
    *,
    access_policy: AgentSkillAccessPolicy,
    manifest: SkillRouteManifest,
    route: SkillExecutionRoute,
    request: AgentSkillRequest | None,
    operator_authorization_receipt_id: str,
    client_id: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    max_invocations: int,
    issued_at: float,
    expires_at: float,
) -> SkillExecutionGrant:
    """Create an exact Route grant without changing Station or lower state."""

    if not isinstance(access_policy, AgentSkillAccessPolicy):
        raise TypeError("access_policy must be an AgentSkillAccessPolicy.")
    if not isinstance(manifest, SkillRouteManifest):
        raise TypeError("manifest must be a SkillRouteManifest.")
    if not isinstance(route, SkillExecutionRoute):
        raise TypeError("route must be a SkillExecutionRoute.")
    try:
        advertised = manifest.route(
            route.skill_id,
            route.skill_version,
            route.route_id,
            route.route_version,
        )
    except KeyError as error:
        raise ValueError("Route is not advertised by manifest.") from error
    if advertised.route.semantic_content_hash != route.semantic_content_hash:
        raise ValueError("Route content does not match the manifest.")
    route_entry = advertised
    binding = route_entry.deployment_binding
    profile_id = _required(
        parameter_profile_id,
        field_name="parameter_profile_id",
    )
    profile_hash = _required(
        parameter_profile_hash,
        field_name="parameter_profile_hash",
    )
    try:
        route.profile(profile_id, profile_hash)
    except KeyError as error:
        raise ValueError("Parameter profile is not allowed by Route.") from error
    normalized_client_id = _required(client_id, field_name="client_id")
    if normalized_client_id not in access_policy.allowed_client_ids:
        raise ValueError("client_id is not allowed by access policy.")
    if manifest.current_mode not in access_policy.allowed_modes:
        raise ValueError("manifest mode is not allowed by access policy.")
    normalized_max_invocations = _positive_integer(
        max_invocations,
        field_name="max_invocations",
    )
    if normalized_max_invocations > access_policy.max_grant_invocations:
        raise ValueError("max_invocations exceeds access policy.")
    normalized_issued_at = _finite(issued_at, field_name="issued_at")
    normalized_expires_at = _finite(expires_at, field_name="expires_at")
    if normalized_expires_at <= normalized_issued_at:
        raise ValueError("Skill execution grant must expire after issuance.")
    if (
        normalized_expires_at - normalized_issued_at
        > access_policy.max_grant_lifetime_sec
    ):
        raise ValueError("Grant lifetime exceeds access policy.")

    if request is not None and not isinstance(request, AgentSkillRequest):
        raise TypeError("request must be an AgentSkillRequest or None.")
    if route.requires_operator_confirmation and request is None:
        raise ValueError(
            "Operator-confirmed Routes require an exact request to bind."
        )
    authorized_request_hash: str | None = None
    if request is not None:
        if request.skill_execution_grant_id is not None:
            raise ValueError("Grant issuance requires a pre-grant request.")
        request_checks = (
            request.client_id == normalized_client_id,
            request.runtime_session_id == manifest.runtime_session_id,
            request.skill_route_manifest_hash == manifest.semantic_content_hash,
            _request_matches_entry(request, route_entry),
            request.parameter_profile_id == profile_id,
            request.parameter_profile_hash == profile_hash,
            request.requested_mode == manifest.current_mode,
        )
        if not all(request_checks):
            raise ValueError("Request does not match the proposed grant scope.")
        authorized_request_hash = request.authorization_content_hash
    return SkillExecutionGrant(
        operator_authorization_receipt_id=operator_authorization_receipt_id,
        client_id=normalized_client_id,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        control_domain_id=manifest.control_domain_id,
        target_binding_id=manifest.target_binding_id,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=binding.deployment_binding_hash,
        parameter_profile_id=profile_id,
        parameter_profile_hash=profile_hash,
        authorized_request_hash=authorized_request_hash,
        allowed_mode=manifest.current_mode,
        control_effect_class=route.control_effect_class,
        may_move_robot=route.may_move_robot,
        max_invocations=normalized_max_invocations,
        issued_at=normalized_issued_at,
        expires_at=normalized_expires_at,
    )


def _decision(
    *,
    accepted: bool,
    reason_code: str,
    request: AgentSkillRequest,
    manifest: SkillRouteManifest,
    policy: AgentSkillAccessPolicy,
    route_entry: SkillRouteManifestEntry | None = None,
    grant: SkillExecutionGrant | None = None,
    effective_timeout_ms: int | None = None,
) -> SkillAccessDecision:
    return SkillAccessDecision(
        accepted=accepted,
        reason_code=reason_code,
        request_hash=request.semantic_content_hash,
        manifest_hash=manifest.semantic_content_hash,
        access_policy_hash=policy.semantic_content_hash,
        route_manifest_entry_hash=(
            None if route_entry is None else route_entry.semantic_content_hash
        ),
        grant_hash=None if grant is None else grant.semantic_content_hash,
        effective_timeout_ms=effective_timeout_ms,
    )


def assess_skill_request_access(
    *,
    policy: AgentSkillAccessPolicy,
    manifest: SkillRouteManifest,
    request: AgentSkillRequest,
    current_station_revision: int,
    now: float,
    grant: SkillExecutionGrant | None = None,
    grant_invocations_used: int = 0,
) -> SkillAccessDecision:
    """Assess application access without admitting or executing an operation."""

    if not isinstance(policy, AgentSkillAccessPolicy):
        raise TypeError("policy must be an AgentSkillAccessPolicy.")
    if not isinstance(manifest, SkillRouteManifest):
        raise TypeError("manifest must be a SkillRouteManifest.")
    if not isinstance(request, AgentSkillRequest):
        raise TypeError("request must be an AgentSkillRequest.")
    revision = _nonnegative_integer(
        current_station_revision,
        field_name="current_station_revision",
    )
    assessed_at = _finite(now, field_name="now")
    invocations_used = _nonnegative_integer(
        grant_invocations_used,
        field_name="grant_invocations_used",
    )

    def reject(
        reason_code: str,
        *,
        route_entry: SkillRouteManifestEntry | None = None,
        include_grant: bool = False,
    ) -> SkillAccessDecision:
        return _decision(
            accepted=False,
            reason_code=reason_code,
            request=request,
            manifest=manifest,
            policy=policy,
            route_entry=route_entry,
            grant=grant if include_grant else None,
        )

    if request.client_id not in policy.allowed_client_ids:
        return reject("client_not_allowed")
    if request.runtime_session_id != manifest.runtime_session_id:
        return reject("runtime_session_mismatch")
    if request.skill_route_manifest_hash != manifest.semantic_content_hash:
        return reject("skill_route_manifest_hash_mismatch")
    if request.expected_station_revision != revision:
        return reject("stale_station_revision")
    if request.requested_mode != manifest.current_mode:
        return reject("runtime_mode_mismatch")
    if request.requested_mode not in policy.allowed_modes:
        return reject("mode_not_allowed_by_policy")

    try:
        route_entry = manifest.route(
            request.skill_id,
            request.skill_version,
            request.route_id,
            request.route_version,
        )
    except KeyError:
        return reject("skill_route_not_advertised")
    route = route_entry.route
    binding = route_entry.deployment_binding
    identity_checks = (
        (
            request.skill_definition_hash == route.skill_definition_hash,
            "skill_definition_hash_mismatch",
        ),
        (request.route_hash == route.semantic_content_hash, "route_hash_mismatch"),
        (
            request.deployment_binding_hash == binding.deployment_binding_hash,
            "deployment_binding_hash_mismatch",
        ),
    )
    for matches, reason_code in identity_checks:
        if not matches:
            return reject(reason_code, route_entry=route_entry)
    try:
        route.profile(
            request.parameter_profile_id,
            request.parameter_profile_hash,
        )
    except KeyError:
        return reject("parameter_profile_not_allowed", route_entry=route_entry)
    if request.requested_timeout_ms > policy.max_request_timeout_ms:
        return reject("timeout_exceeds_access_policy", route_entry=route_entry)
    if request.requested_timeout_ms > route.max_operation_duration_ms:
        return reject("timeout_exceeds_route", route_entry=route_entry)

    grant_required = (
        route.privilege_tier in policy.grant_required_tiers
        or route.requires_operator_confirmation
    )
    if grant_required and grant is None:
        return reject("skill_execution_grant_required", route_entry=route_entry)
    if grant is None:
        if request.skill_execution_grant_id is not None:
            return reject("unknown_skill_execution_grant", route_entry=route_entry)
        return _decision(
            accepted=True,
            reason_code="accepted",
            request=request,
            manifest=manifest,
            policy=policy,
            route_entry=route_entry,
            effective_timeout_ms=request.requested_timeout_ms,
        )

    if request.skill_execution_grant_id != grant.grant_id:
        return reject(
            "skill_execution_grant_id_mismatch",
            route_entry=route_entry,
            include_grant=True,
        )
    exact_grant_checks = (
        (grant.client_id == request.client_id, "grant_client_mismatch"),
        (
            grant.runtime_session_id == manifest.runtime_session_id,
            "grant_runtime_session_mismatch",
        ),
        (
            grant.skill_route_manifest_hash == manifest.semantic_content_hash,
            "grant_manifest_mismatch",
        ),
        (
            grant.control_domain_id == manifest.control_domain_id,
            "grant_control_domain_mismatch",
        ),
        (
            grant.target_binding_id == manifest.target_binding_id,
            "grant_target_binding_mismatch",
        ),
        (grant.skill_id == request.skill_id, "grant_skill_id_mismatch"),
        (
            grant.skill_version == request.skill_version,
            "grant_skill_version_mismatch",
        ),
        (
            grant.skill_definition_hash == request.skill_definition_hash,
            "grant_skill_definition_mismatch",
        ),
        (grant.route_id == request.route_id, "grant_route_id_mismatch"),
        (
            grant.route_version == request.route_version,
            "grant_route_version_mismatch",
        ),
        (grant.route_hash == request.route_hash, "grant_route_hash_mismatch"),
        (
            grant.deployment_binding_hash == request.deployment_binding_hash,
            "grant_deployment_binding_mismatch",
        ),
        (
            grant.parameter_profile_id == request.parameter_profile_id,
            "grant_parameter_profile_id_mismatch",
        ),
        (
            grant.parameter_profile_hash == request.parameter_profile_hash,
            "grant_parameter_profile_mismatch",
        ),
        (
            grant.authorized_request_hash is None
            or grant.authorized_request_hash == request.authorization_content_hash,
            "grant_authorized_request_mismatch",
        ),
        (grant.allowed_mode == request.requested_mode, "grant_mode_mismatch"),
        (
            grant.control_effect_class == route.control_effect_class,
            "grant_control_effect_scope_mismatch",
        ),
        (
            grant.may_move_robot == route.may_move_robot,
            "grant_motion_scope_mismatch",
        ),
    )
    for matches, reason_code in exact_grant_checks:
        if not matches:
            return reject(
                reason_code,
                route_entry=route_entry,
                include_grant=True,
            )
    if not grant.active_at(assessed_at):
        return reject(
            "skill_execution_grant_inactive",
            route_entry=route_entry,
            include_grant=True,
        )
    if route.requires_operator_confirmation and grant.authorized_request_hash is None:
        return reject(
            "grant_missing_authorized_request",
            route_entry=route_entry,
            include_grant=True,
        )
    if grant.expires_at - grant.issued_at > policy.max_grant_lifetime_sec:
        return reject(
            "grant_lifetime_exceeds_access_policy",
            route_entry=route_entry,
            include_grant=True,
        )
    if grant.max_invocations > policy.max_grant_invocations:
        return reject(
            "grant_invocations_exceed_access_policy",
            route_entry=route_entry,
            include_grant=True,
        )
    if invocations_used >= grant.max_invocations:
        return reject(
            "skill_execution_grant_exhausted",
            route_entry=route_entry,
            include_grant=True,
        )
    return _decision(
        accepted=True,
        reason_code="accepted",
        request=request,
        manifest=manifest,
        policy=policy,
        route_entry=route_entry,
        grant=grant,
        effective_timeout_ms=request.requested_timeout_ms,
    )


__all__ = [
    "AGENT_SKILL_GRANT_REQUIRED_TIERS_V1",
    "AgentSkillAccessPolicy",
    "SkillAccessDecision",
    "SkillExecutionGrant",
    "assess_skill_request_access",
    "issue_skill_execution_grant",
]
