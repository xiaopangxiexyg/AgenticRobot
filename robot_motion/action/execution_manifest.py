"""Transport-neutral control execution manifests.

An execution manifest freezes *what one control domain intends to run*.  It is
not an execution authorization and it does not imply that a safety-migration
gate has passed.  Gate evidence remains in :mod:`safety_migration`, while the
final dispatch owner/permit remains a runtime fact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.position_control.capability import ExecutionCapabilityProfile

from .components import ActionComponentCatalog
from .reference_activation import ReferenceActivationSpec
from .schema import BIMANUAL_QPOS_V1


CONTROL_EXECUTION_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.control_execution_manifest.v1"
)
EXECUTION_PROFILE_EXCLUSION_HASH_DOMAIN_V1 = (
    "dexteleop.execution_profile_exclusion.v1"
)
M6_CANDIDATE_POLICY_ARM_QPOS_V1 = "m6_candidate_policy_arm_qpos_v1"

HUMAN_MOTION_PROPOSAL_STATUSES_V1 = frozenset(
    {"authorized", "not_applicable"}
)


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _unique_identifiers(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, name=name) for value in values)
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be unique.")
    return normalized


def _positive_rate(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


@dataclass(frozen=True)
class ExecutionProfileExclusion:
    """Explicitly excludes one capability from one narrow manifest."""

    capability_id: str
    reason: str
    evidence_ids: tuple[str, ...]
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        capability_id: str,
        reason: str,
        evidence_ids: tuple[str, ...],
    ) -> "ExecutionProfileExclusion":
        payload = {
            "capability_id": _identifier(capability_id, name="capability_id"),
            "reason": _identifier(reason, name="reason"),
            "evidence_ids": _unique_identifiers(
                evidence_ids,
                name="evidence_ids",
            ),
        }
        return cls(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=EXECUTION_PROFILE_EXCLUSION_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class ControlExecutionManifest:
    """Immutable logical execution configuration for one control domain.

    ``actual_reference_frame_producer`` is deliberately independent from
    ``reference_timing_semantics_hz``.  A profile may use 60 Hz timing for
    activation/limits without claiming that a 60 Hz reference producer exists.
    """

    manifest_id: str
    manifest_version: str
    action_space: str
    component_catalog_hash: str
    controlled_components: tuple[str, ...]
    selection_groups: tuple[str, ...]
    motion_authority_sources: tuple[str, ...]
    human_motion_proposal_status: str
    safety_control_sources: tuple[str, ...]
    safety_control_requests: tuple[str, ...]
    lifecycle_control_sources: tuple[str, ...]
    lifecycle_control_requests: tuple[str, ...]
    authoritative_decision_hz: float
    reference_activation_spec_id: str
    reference_activation_spec_hash: str
    execution_capability_profile_hash: str
    reference_timing_semantics_hz: float | None
    actual_reference_frame_producer: str
    planner_command_hz: float
    exclusions: tuple[ExecutionProfileExclusion, ...]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "manifest_id",
            "manifest_version",
            "action_space",
            "component_catalog_hash",
            "reference_activation_spec_id",
            "reference_activation_spec_hash",
            "execution_capability_profile_hash",
            "actual_reference_frame_producer",
            "semantic_content_hash",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))

        for name in (
            "controlled_components",
            "selection_groups",
            "motion_authority_sources",
            "safety_control_sources",
            "safety_control_requests",
            "lifecycle_control_sources",
            "lifecycle_control_requests",
        ):
            object.__setattr__(
                self,
                name,
                _unique_identifiers(tuple(getattr(self, name)), name=name),
            )

        decision_hz = _positive_rate(
            self.authoritative_decision_hz,
            name="authoritative_decision_hz",
        )
        if decision_hz > 60.0:
            raise ValueError("Authoritative ActionArbiter rate cannot exceed 60 Hz.")
        object.__setattr__(self, "authoritative_decision_hz", decision_hz)
        object.__setattr__(
            self,
            "planner_command_hz",
            _positive_rate(self.planner_command_hz, name="planner_command_hz"),
        )
        if self.reference_timing_semantics_hz is not None:
            object.__setattr__(
                self,
                "reference_timing_semantics_hz",
                _positive_rate(
                    self.reference_timing_semantics_hz,
                    name="reference_timing_semantics_hz",
                ),
            )

        human_status = str(self.human_motion_proposal_status).strip()
        if human_status not in HUMAN_MOTION_PROPOSAL_STATUSES_V1:
            raise ValueError("Unsupported human motion proposal status.")
        if human_status == "not_applicable" and any(
            source.startswith("human") for source in self.motion_authority_sources
        ):
            raise ValueError(
                "Human motion cannot be both not-applicable and an authority source."
            )
        object.__setattr__(self, "human_motion_proposal_status", human_status)

        exclusions = tuple(self.exclusions)
        if any(not isinstance(item, ExecutionProfileExclusion) for item in exclusions):
            raise TypeError("exclusions must contain ExecutionProfileExclusion values.")
        ids = tuple(item.capability_id for item in exclusions)
        if len(set(ids)) != len(ids):
            raise ValueError("Execution-profile exclusions must be unique.")
        object.__setattr__(
            self,
            "exclusions",
            tuple(sorted(exclusions, key=lambda item: item.capability_id)),
        )


def _manifest_payload(
    *,
    manifest_id: str,
    manifest_version: str,
    action_space: str,
    catalog: ActionComponentCatalog,
    motion_authority_sources: tuple[str, ...],
    human_motion_proposal_status: str,
    safety_control_sources: tuple[str, ...],
    safety_control_requests: tuple[str, ...],
    lifecycle_control_sources: tuple[str, ...],
    lifecycle_control_requests: tuple[str, ...],
    capability: ExecutionCapabilityProfile,
    activation: ReferenceActivationSpec,
    actual_reference_frame_producer: str,
    exclusions: tuple[ExecutionProfileExclusion, ...],
) -> dict[str, object]:
    return {
        "manifest_id": _identifier(manifest_id, name="manifest_id"),
        "manifest_version": _identifier(manifest_version, name="manifest_version"),
        "action_space": _identifier(action_space, name="action_space"),
        "component_catalog_hash": catalog.semantic_content_hash,
        "controlled_components": tuple(
            component.component_id for component in catalog.components
        ),
        "selection_groups": tuple(catalog.selection_group_ids),
        "motion_authority_sources": tuple(motion_authority_sources),
        "human_motion_proposal_status": human_motion_proposal_status,
        "safety_control_sources": tuple(safety_control_sources),
        "safety_control_requests": tuple(safety_control_requests),
        "lifecycle_control_sources": tuple(lifecycle_control_sources),
        "lifecycle_control_requests": tuple(lifecycle_control_requests),
        "authoritative_decision_hz": capability.configured_selected_keyframe_hz,
        "reference_activation_spec_id": activation.spec_id,
        "reference_activation_spec_hash": activation.semantic_content_hash,
        "execution_capability_profile_hash": capability.semantic_content_hash,
        "reference_timing_semantics_hz": capability.reference_resample_hz,
        "actual_reference_frame_producer": actual_reference_frame_producer,
        "planner_command_hz": capability.command_hz,
        "exclusions": tuple(exclusions),
    }


def build_control_execution_manifest(
    *,
    manifest_id: str,
    manifest_version: str,
    action_space: str,
    catalog: ActionComponentCatalog,
    motion_authority_sources: tuple[str, ...],
    human_motion_proposal_status: str,
    safety_control_sources: tuple[str, ...],
    safety_control_requests: tuple[str, ...],
    lifecycle_control_sources: tuple[str, ...],
    lifecycle_control_requests: tuple[str, ...],
    capability: ExecutionCapabilityProfile,
    activation: ReferenceActivationSpec,
    actual_reference_frame_producer: str,
    exclusions: tuple[ExecutionProfileExclusion, ...],
) -> ControlExecutionManifest:
    """Build one exact execution manifest without conferring authorization.

    Source- or application-specific builders may use this constructor after
    they have applied their narrower profile checks.  Numeric execution
    capability and runtime owner/permit evidence remain separate gates.
    """

    if not isinstance(catalog, ActionComponentCatalog):
        raise TypeError("catalog must be an ActionComponentCatalog.")
    if not isinstance(capability, ExecutionCapabilityProfile):
        raise TypeError("capability must be an ExecutionCapabilityProfile.")
    if not isinstance(activation, ReferenceActivationSpec):
        raise TypeError("activation must be a ReferenceActivationSpec.")
    if str(action_space) != catalog.action_space:
        raise ValueError("Execution manifest action space must match the catalog.")
    if action_space not in capability.validated_action_spaces:
        raise ValueError("Execution capability does not validate the action space.")
    if not set(catalog.selection_group_ids).issubset(
        capability.validated_selection_groups
    ):
        raise ValueError("Execution capability does not validate all selection groups.")
    if activation.execution_capability_profile_hash != capability.semantic_content_hash:
        raise ValueError("Reference activation is bound to another capability profile.")
    normalized_exclusions = tuple(
        sorted(tuple(exclusions), key=lambda item: item.capability_id)
    )
    payload = _manifest_payload(
        manifest_id=manifest_id,
        manifest_version=manifest_version,
        action_space=action_space,
        catalog=catalog,
        motion_authority_sources=motion_authority_sources,
        human_motion_proposal_status=human_motion_proposal_status,
        safety_control_sources=safety_control_sources,
        safety_control_requests=safety_control_requests,
        lifecycle_control_sources=lifecycle_control_sources,
        lifecycle_control_requests=lifecycle_control_requests,
        capability=capability,
        activation=activation,
        actual_reference_frame_producer=actual_reference_frame_producer,
        exclusions=normalized_exclusions,
    )
    semantic_payload = {
        **payload,
        "exclusions": tuple(
            exclusion.semantic_content_hash for exclusion in normalized_exclusions
        ),
    }
    return ControlExecutionManifest(
        **payload,
        semantic_content_hash=canonical_content_hash(
            semantic_payload,
            domain=CONTROL_EXECUTION_MANIFEST_HASH_DOMAIN_V1,
        ),
    )


def build_m6_candidate_policy_arm_qpos_manifest(
    *,
    catalog: ActionComponentCatalog,
    capability: ExecutionCapabilityProfile,
    activation: ReferenceActivationSpec,
) -> ControlExecutionManifest:
    """Freeze the first arms-only M6 candidate without authorizing motion."""

    if not isinstance(catalog, ActionComponentCatalog):
        raise TypeError("catalog must be an ActionComponentCatalog.")
    if not isinstance(capability, ExecutionCapabilityProfile):
        raise TypeError("capability must be an ExecutionCapabilityProfile.")
    if not isinstance(activation, ReferenceActivationSpec):
        raise TypeError("activation must be a ReferenceActivationSpec.")
    if catalog.action_space != BIMANUAL_QPOS_V1:
        raise ValueError("The first M6 candidate requires bimanual qpos actions.")
    expected_components = {"left_arm", "right_arm"}
    if {component.component_id for component in catalog.components} != expected_components:
        raise ValueError("The first M6 candidate requires exactly both arm components.")
    if catalog.selection_group_ids != ("arms",):
        raise ValueError("The first M6 candidate requires one coupled arms group.")
    if any(not component.coupled for component in catalog.components):
        raise ValueError("Both arm components must be atomically coupled.")
    if BIMANUAL_QPOS_V1 not in capability.validated_action_spaces:
        raise ValueError("Capability does not validate bimanual qpos execution.")
    if "arms" not in capability.validated_selection_groups:
        raise ValueError("Capability does not validate the coupled arms group.")
    if not math.isclose(
        capability.configured_selected_keyframe_hz,
        30.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("The first M6 candidate freezes a 30 Hz decision rate.")
    if not math.isclose(capability.command_hz, 120.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Current candidate evidence requires 120 Hz planner/qcmd.")
    if activation.activation_mode != "scheduled_segment":
        raise ValueError("The first M6 candidate requires scheduled segment activation.")
    if activation.execution_capability_profile_hash != capability.semantic_content_hash:
        raise ValueError("Reference activation is bound to another capability profile.")

    exclusion_evidence = (
        f"arms-only-component-catalog:{catalog.semantic_content_hash}",
    )
    exclusions = tuple(
        ExecutionProfileExclusion.create(
            capability_id=capability_id,
            reason="arms_only_execution_manifest",
            evidence_ids=exclusion_evidence,
        )
        for capability_id in (
            "hand_inspire_v1",
            "hand_dmtac_v1",
            "replay_authority",
            "pose_impedance",
            "fix_arm_human_motion",
            "human_motion_takeover",
        )
    )
    payload = _manifest_payload(
        manifest_id=M6_CANDIDATE_POLICY_ARM_QPOS_V1,
        manifest_version="v1",
        action_space=BIMANUAL_QPOS_V1,
        catalog=catalog,
        motion_authority_sources=("policy",),
        human_motion_proposal_status="not_applicable",
        safety_control_sources=("operator", "runtime_safety_supervisor"),
        safety_control_requests=("hold", "estop", "manual_stop"),
        lifecycle_control_sources=(
            "operator_authorized_recovery_state_machine",
        ),
        lifecycle_control_requests=("recovery", "home", "rearm"),
        capability=capability,
        activation=activation,
        actual_reference_frame_producer="not_applicable",
        exclusions=exclusions,
    )
    semantic_payload = {
        **payload,
        "exclusions": tuple(
            exclusion.semantic_content_hash for exclusion in exclusions
        ),
    }
    return ControlExecutionManifest(
        **payload,
        semantic_content_hash=canonical_content_hash(
            semantic_payload,
            domain=CONTROL_EXECUTION_MANIFEST_HASH_DOMAIN_V1,
        ),
    )


__all__ = [
    "CONTROL_EXECUTION_MANIFEST_HASH_DOMAIN_V1",
    "EXECUTION_PROFILE_EXCLUSION_HASH_DOMAIN_V1",
    "HUMAN_MOTION_PROPOSAL_STATUSES_V1",
    "M6_CANDIDATE_POLICY_ARM_QPOS_V1",
    "ControlExecutionManifest",
    "ExecutionProfileExclusion",
    "build_control_execution_manifest",
    "build_m6_candidate_policy_arm_qpos_manifest",
]
