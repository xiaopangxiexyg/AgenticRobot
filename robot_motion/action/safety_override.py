"""Typed, static admission for policy-requested safety constraints.

This contract replaces an eventual per-action metadata lookup with one
deployment/session preflight.  It is additive for now: the legacy
``ActionSafetyFilter`` still owns production behavior until its caller is
explicitly migrated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash


SAFETY_CONSTRAINT_OVERRIDE_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.safety_constraint_override_spec.v1"
)
SAFETY_CONSTRAINT_OVERRIDE_ADMISSION_HASH_DOMAIN_V1 = (
    "dexteleop.safety_constraint_override_admission.v1"
)
SAFETY_CONSTRAINT_EFFECTIVE_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.safety_constraint_effective_profile.v1"
)

SAFETY_CONSTRAINT_OVERRIDE_SCOPE_V1 = "deployment_session_static"
SAFETY_CONSTRAINT_BOUND_UNITS_V1 = MappingProxyType(
    {
        "max_arm_xyz_step_m": "m",
        "max_arm_rot_step_deg": "deg",
        "max_arm_joint_step_rad": "rad",
        "max_hand_step": "normalized_or_artifact_unit",
    }
)


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _identifier_tuple(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, name=name) for value in values)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be non-empty and unique.")
    return normalized


def _bound_mapping(values: Mapping[str, float], *, name: str) -> Mapping[str, float]:
    normalized: dict[str, float] = {}
    for raw_field, raw_value in dict(values).items():
        field = _identifier(raw_field, name=f"{name} field")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name}[{field!r}] must be finite and non-negative.")
        normalized[field] = value
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class SafetyConstraintOverrideSpec:
    spec_id: str
    spec_version: str
    source_artifact_id: str
    baseline_safety_profile_hash: str
    applicable_action_spaces: tuple[str, ...]
    applicable_selection_groups: tuple[str, ...]
    allowed_override_fields: tuple[str, ...]
    requested_bounds: Mapping[str, float]
    scope: str
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        spec_id: str,
        spec_version: str,
        source_artifact_id: str,
        baseline_safety_profile_hash: str,
        applicable_action_spaces: tuple[str, ...],
        applicable_selection_groups: tuple[str, ...],
        allowed_override_fields: tuple[str, ...],
        requested_bounds: Mapping[str, float],
        scope: str = SAFETY_CONSTRAINT_OVERRIDE_SCOPE_V1,
    ) -> "SafetyConstraintOverrideSpec":
        normalized_scope = _identifier(scope, name="scope")
        if normalized_scope != SAFETY_CONSTRAINT_OVERRIDE_SCOPE_V1:
            raise ValueError("Safety override v1 supports only static session scope.")
        allowed = _identifier_tuple(
            allowed_override_fields,
            name="allowed_override_fields",
        )
        unknown_allowed = set(allowed) - set(SAFETY_CONSTRAINT_BOUND_UNITS_V1)
        if unknown_allowed:
            raise ValueError(
                f"Unsupported safety override fields: {sorted(unknown_allowed)!r}."
            )
        requested = _bound_mapping(requested_bounds, name="requested_bounds")
        if set(requested) - set(allowed):
            raise ValueError("Requested safety bounds must be explicitly allowed.")
        payload = {
            "spec_id": _identifier(spec_id, name="spec_id"),
            "spec_version": _identifier(spec_version, name="spec_version"),
            "source_artifact_id": _identifier(
                source_artifact_id,
                name="source_artifact_id",
            ),
            "baseline_safety_profile_hash": _identifier(
                baseline_safety_profile_hash,
                name="baseline_safety_profile_hash",
            ),
            "applicable_action_spaces": _identifier_tuple(
                applicable_action_spaces,
                name="applicable_action_spaces",
            ),
            "applicable_selection_groups": _identifier_tuple(
                applicable_selection_groups,
                name="applicable_selection_groups",
            ),
            "allowed_override_fields": allowed,
            "requested_bounds": requested,
            "scope": normalized_scope,
        }
        semantic_payload = {**payload, "requested_bounds": dict(requested)}
        return cls(
            **payload,
            semantic_content_hash=canonical_content_hash(
                semantic_payload,
                domain=SAFETY_CONSTRAINT_OVERRIDE_SPEC_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class SafetyConstraintOverrideCheck:
    field_name: str
    unit: str
    baseline_value: float | None
    requested_value: float
    status: str
    reason: str


@dataclass(frozen=True)
class SafetyConstraintOverrideAdmissionReceipt:
    admission_id: str
    spec_id: str
    spec_hash: str
    baseline_safety_profile_hash: str
    accepted: bool
    checks: tuple[SafetyConstraintOverrideCheck, ...]
    effective_bounds: Mapping[str, float]
    effective_safety_profile_hash: str | None
    rejection_reasons: tuple[str, ...]
    admitted_at: float
    semantic_content_hash: str


def admit_safety_constraint_override(
    spec: SafetyConstraintOverrideSpec,
    *,
    baseline_safety_profile_hash: str,
    baseline_bounds: Mapping[str, float],
    action_space: str,
    selection_groups: tuple[str, ...],
    admitted_at: float,
) -> SafetyConstraintOverrideAdmissionReceipt:
    """Admit once; an accepted receipt may be referenced by later proposals."""

    if not isinstance(spec, SafetyConstraintOverrideSpec):
        raise TypeError("spec must be a SafetyConstraintOverrideSpec.")
    baseline_hash = _identifier(
        baseline_safety_profile_hash,
        name="baseline_safety_profile_hash",
    )
    baseline = _bound_mapping(baseline_bounds, name="baseline_bounds")
    target_action_space = _identifier(action_space, name="action_space")
    target_groups = _identifier_tuple(selection_groups, name="selection_groups")
    timestamp = float(admitted_at)
    if not math.isfinite(timestamp):
        raise ValueError("admitted_at must be finite.")

    rejection_reasons: list[str] = []
    if baseline_hash != spec.baseline_safety_profile_hash:
        rejection_reasons.append("baseline_profile_hash_mismatch")
    if target_action_space not in spec.applicable_action_spaces:
        rejection_reasons.append("action_space_not_applicable")
    if not set(target_groups).issubset(spec.applicable_selection_groups):
        rejection_reasons.append("selection_group_not_applicable")

    effective = dict(baseline)
    checks: list[SafetyConstraintOverrideCheck] = []
    for field, requested in spec.requested_bounds.items():
        baseline_value = baseline.get(field)
        if baseline_value is None:
            status = "baseline_field_missing"
            reason = "The baseline profile does not define this bound."
        elif baseline_value > 0.0 and not (0.0 < requested <= baseline_value + 1e-12):
            status = "not_tightening"
            reason = "An enabled bound may only be reduced to another positive value."
        elif baseline_value <= 0.0 and not math.isclose(
            requested,
            baseline_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            status = "disabled_baseline_cannot_change"
            reason = "A disabled baseline cannot be enabled or broadened by override."
        else:
            status = "exact" if math.isclose(
                requested,
                baseline_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ) else "tightened"
            reason = ""
            effective[field] = requested
        checks.append(
            SafetyConstraintOverrideCheck(
                field_name=field,
                unit=SAFETY_CONSTRAINT_BOUND_UNITS_V1[field],
                baseline_value=baseline_value,
                requested_value=requested,
                status=status,
                reason=reason,
            )
        )
        if status not in {"exact", "tightened"}:
            rejection_reasons.append(f"{status}:{field}")

    accepted = not rejection_reasons
    effective_profile_hash = None
    if accepted:
        effective_profile_hash = canonical_content_hash(
            {
                "baseline_safety_profile_hash": baseline_hash,
                "override_spec_hash": spec.semantic_content_hash,
                "effective_bounds": effective,
            },
            domain=SAFETY_CONSTRAINT_EFFECTIVE_PROFILE_HASH_DOMAIN_V1,
        )
    semantic_payload = {
        "spec_id": spec.spec_id,
        "spec_hash": spec.semantic_content_hash,
        "baseline_safety_profile_hash": baseline_hash,
        "accepted": accepted,
        "checks": tuple(
            {
                "field_name": check.field_name,
                "unit": check.unit,
                "baseline_value": check.baseline_value,
                "requested_value": check.requested_value,
                "status": check.status,
                "reason": check.reason,
            }
            for check in checks
        ),
        "effective_bounds": effective,
        "effective_safety_profile_hash": effective_profile_hash,
        "rejection_reasons": tuple(rejection_reasons),
        "admitted_at": timestamp,
    }
    semantic_hash = canonical_content_hash(
        semantic_payload,
        domain=SAFETY_CONSTRAINT_OVERRIDE_ADMISSION_HASH_DOMAIN_V1,
    )
    return SafetyConstraintOverrideAdmissionReceipt(
        admission_id=f"safety-override-admission:{semantic_hash}",
        spec_id=spec.spec_id,
        spec_hash=spec.semantic_content_hash,
        baseline_safety_profile_hash=baseline_hash,
        accepted=accepted,
        checks=tuple(checks),
        effective_bounds=MappingProxyType(dict(sorted(effective.items()))),
        effective_safety_profile_hash=effective_profile_hash,
        rejection_reasons=tuple(rejection_reasons),
        admitted_at=timestamp,
        semantic_content_hash=semantic_hash,
    )


__all__ = [
    "SAFETY_CONSTRAINT_BOUND_UNITS_V1",
    "SAFETY_CONSTRAINT_EFFECTIVE_PROFILE_HASH_DOMAIN_V1",
    "SAFETY_CONSTRAINT_OVERRIDE_ADMISSION_HASH_DOMAIN_V1",
    "SAFETY_CONSTRAINT_OVERRIDE_SCOPE_V1",
    "SAFETY_CONSTRAINT_OVERRIDE_SPEC_HASH_DOMAIN_V1",
    "SafetyConstraintOverrideAdmissionReceipt",
    "SafetyConstraintOverrideCheck",
    "SafetyConstraintOverrideSpec",
    "admit_safety_constraint_override",
]
