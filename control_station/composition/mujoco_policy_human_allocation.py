"""Default-off MuJoCo profile for policy/human dynamic allocation.

This file chooses concrete role names and MuJoCo schemas.  Evidence windows,
allocation reduction, and component composition remain reusable
``control_composition`` owners.  The profile contains no canonical safety,
HOLD, target resolution, IK, reference, or dispatch policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from control_composition.allocation import (
    ActivityMetricProvider,
    AllocationEvidenceWindow,
    COMPOSITION_ALLOCATION_CONTEXT_CONTRACT_HASH_DEVELOPMENT,
    COMPOSITION_ALLOCATION_MODES_DEVELOPMENT,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    ComponentAllocationPolicy,
    CompositionAllocationSpec,
    AllocationTransitionContinuityPolicy,
    ResidualMagnitudeMetricProvider,
)
from control_composition.allocation.composition_profile import (
    AllocatedCompositionContracts,
    build_allocated_composition_contracts,
)
from control_composition.information import (
    CompositionContextRequirement,
    CompositionInformationPlan,
)
from robot_motion.action import (
    ComponentActionSchema,
    ConservativeCompoundAuthorityProfile,
)
from robot_motion.coordinate_frames import CoordinateFrameGraph

from .allocation_publication import (
    StationAllocationPublicationProfile,
)
from .derived_input_publication import (
    StationDerivedInputPublicationProfile,
)
from .mujoco_allocation_publication import (
    MujocoAllocationPublicationProfile,
)
from .mujoco_policy_human_residual import (
    MujocoPolicyHumanResidualInputContracts,
    MujocoPolicyHumanResidualProfileConfig,
    build_mujoco_policy_human_residual_input_contracts,
)


MUJOCO_POLICY_HUMAN_ALLOCATION_PROFILE_OFF = "off"
MUJOCO_POLICY_HUMAN_ALLOCATION_PROMOTABLE_MODES_SCHEME_A = (
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class MujocoPolicyHumanAllocationProfileConfig:
    """Input/allocation profile data, separate from proposal publication.

    ``send_actions`` remains false because this object configures contributor
    derivation and allocation evidence only.  The companion Station profile
    builder below is the sole place that enables proposal publication and
    carries the Scheme-A promotion allowlist into that gate.
    """

    profile_id: str = MUJOCO_POLICY_HUMAN_ALLOCATION_PROFILE_OFF
    enabled: bool = False
    send_actions: bool = False
    baseline_role: str = "policy"
    residual_role: str = "human_residual"
    absolute_overlay_role: str = "human_absolute"
    motion_activity_role: str = "human_activity"
    allocation_context_channel: str = "source_allocation"
    # Keep the generic profile conservative by default.  Applications that
    # support a stable deferred human-source port may opt into policy-only
    # activation and admit the human roles at a later typed boundary.
    require_human_roles_at_start: bool = True
    proposal_promotable_modes: tuple[str, ...] = (
        MUJOCO_POLICY_HUMAN_ALLOCATION_PROMOTABLE_MODES_SCHEME_A
    )

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "baseline_role",
            "residual_role",
            "absolute_overlay_role",
            "motion_activity_role",
            "allocation_context_channel",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        if any(
            not isinstance(value, bool)
            for value in (
                self.enabled,
                self.send_actions,
                self.require_human_roles_at_start,
            )
        ):
            raise TypeError(
                "enabled, send_actions, and require_human_roles_at_start "
                "must be booleans."
            )
        if self.send_actions:
            raise ValueError(
                "This profile is non-authoritative shadow; proposal publication "
                "uses the separate MuJoCo allocation publication gate."
            )
        if self.enabled and (
            self.profile_id == MUJOCO_POLICY_HUMAN_ALLOCATION_PROFILE_OFF
        ):
            raise ValueError("An enabled allocation profile requires an id.")
        promotable_modes = tuple(
            _required(mode, name="proposal_promotable_mode")
            for mode in self.proposal_promotable_modes
        )
        if (
            not promotable_modes
            or len(promotable_modes) != len(set(promotable_modes))
            or any(
                mode not in COMPOSITION_ALLOCATION_MODES_DEVELOPMENT
                for mode in promotable_modes
            )
        ):
            raise ValueError(
                "proposal_promotable_modes must be a non-empty unique subset "
                "of the allocation mode contract."
            )
        object.__setattr__(
            self,
            "proposal_promotable_modes",
            promotable_modes,
        )
        roles = (
            self.baseline_role,
            self.residual_role,
            self.absolute_overlay_role,
        )
        if len(set(roles)) != len(roles):
            raise ValueError("Allocation action roles must be distinct.")


@dataclass(frozen=True)
class MujocoPolicyHumanAllocationProfileBundle:
    config: MujocoPolicyHumanAllocationProfileConfig
    input_contracts: MujocoPolicyHumanResidualInputContracts
    allocation_spec: CompositionAllocationSpec
    composition_contracts: AllocatedCompositionContracts
    information_plan: CompositionInformationPlan
    authority_profile: ConservativeCompoundAuthorityProfile
    evidence_window: AllocationEvidenceWindow = field(
        repr=False,
        compare=False,
    )

    # Present the same read-only contract surface consumed by target/runtime
    # assembly without copying the generic allocated-composition contracts.
    @property
    def composition_spec(self):
        return self.composition_contracts.composition_spec

    @property
    def component_rule_set(self):
        return self.composition_contracts.component_rule_set

    @property
    def strategy_spec(self):
        return self.composition_contracts.strategy_spec

    @property
    def strategy(self):
        return self.composition_contracts.strategy

    @property
    def input_action_schema_by_role(
        self,
    ) -> Mapping[str, ComponentActionSchema]:
        output = self.composition_contracts.output_action_schema
        relative = self.composition_contracts.relative_action_schema
        return MappingProxyType(
            {
                self.config.baseline_role: output,
                self.config.residual_role: relative,
                self.config.absolute_overlay_role: output,
            }
        )


@dataclass(frozen=True)
class MujocoPolicyHumanAllocationStationProfiles:
    """Resource-free Station publication profiles for one allocation bundle."""

    derived_input: StationDerivedInputPublicationProfile
    allocation_context: StationAllocationPublicationProfile
    proposal: MujocoAllocationPublicationProfile

    def __post_init__(self) -> None:
        if not self.derived_input.enabled:
            raise ValueError("Allocation derived-input publication must be enabled.")
        if not self.allocation_context.enabled:
            raise ValueError("Allocation context publication must be enabled.")
        if not self.proposal.enabled or not self.proposal.send_actions:
            raise ValueError("Allocation proposal publication must be enabled.")


def build_mujoco_policy_human_allocation_profile(
    *,
    absolute_action_schema: ComponentActionSchema,
    frame_graph: CoordinateFrameGraph,
    concrete_reference_frame_by_component: Mapping[str, str],
    concrete_controlled_frame_by_component: Mapping[str, str],
    source_binding_id: str,
    source_binding_hash: str,
    composer_producer_id: str,
    component_policies: tuple[ComponentAllocationPolicy, ...],
    window_duration_sec: float,
    evidence_ttl_sec: float,
    activity_normalization_scale_by_component: Mapping[str, Mapping[str, float]],
    residual_normalization_scale_by_component: Mapping[str, Mapping[str, float]],
    config: MujocoPolicyHumanAllocationProfileConfig | None = None,
    pose_translation_gain_by_component: Mapping[str, object] | None = None,
    joint_gain_by_component: Mapping[str, object] | None = None,
    activity_metric_provider_by_component: (
        Mapping[str, ActivityMetricProvider] | None
    ) = None,
    activity_metric_provider_identity_by_component: (
        Mapping[str, tuple[str, str]] | None
    ) = None,
    residual_metric_provider_by_component: (
        Mapping[str, ResidualMagnitudeMetricProvider] | None
    ) = None,
    residual_metric_provider_identity_by_component: (
        Mapping[str, tuple[str, str]] | None
    ) = None,
    transition_continuity_policy: (
        AllocationTransitionContinuityPolicy | None
    ) = None,
) -> MujocoPolicyHumanAllocationProfileBundle:
    """Build an inert profile; callers must explicitly supply all thresholds."""

    profile = config or MujocoPolicyHumanAllocationProfileConfig()
    residual_config = MujocoPolicyHumanResidualProfileConfig(
        enabled=profile.enabled,
        profile_id=profile.profile_id,
        baseline_role=profile.baseline_role,
        residual_role=profile.residual_role,
        absolute_overlay_role=profile.absolute_overlay_role,
        require_residual_role=profile.require_human_roles_at_start,
    )
    input_contracts = build_mujoco_policy_human_residual_input_contracts(
        absolute_action_schema=absolute_action_schema,
        frame_graph=frame_graph,
        concrete_reference_frame_by_component=(concrete_reference_frame_by_component),
        concrete_controlled_frame_by_component=(concrete_controlled_frame_by_component),
        source_binding_id=source_binding_id,
        source_binding_hash=source_binding_hash,
        config=residual_config,
        pose_translation_gain_by_component=(pose_translation_gain_by_component),
        joint_gain_by_component=joint_gain_by_component,
    )
    policies = tuple(component_policies)
    if set(policy.component_id for policy in policies) != set(
        absolute_action_schema.component_ids
    ) or len(policies) != len(absolute_action_schema.component_ids):
        raise ValueError("Component policies must cover the action topology exactly.")
    if any(
        policy.motion_source_role != profile.motion_activity_role
        or policy.residual_source_role != profile.residual_role
        for policy in policies
    ):
        raise ValueError("Component evidence roles differ from the concrete profile.")
    allocation_spec = CompositionAllocationSpec(
        allocation_spec_id=f"{profile.profile_id}.allocation",
        base_absolute_role=profile.baseline_role,
        relative_overlay_role=profile.residual_role,
        absolute_overlay_role=profile.absolute_overlay_role,
        component_policies=policies,
        transition_continuity_policy_hash=(
            None
            if transition_continuity_policy is None
            else transition_continuity_policy.semantic_content_hash
        ),
    )
    composition_contracts = build_allocated_composition_contracts(
        contract_id=profile.profile_id,
        output_action_schema=absolute_action_schema,
        relative_action_schema=input_contracts.relative_action_schema,
        allocation_spec=allocation_spec,
        allocation_context_channel=profile.allocation_context_channel,
        required_roles=(
            (
                profile.baseline_role,
                profile.residual_role,
                profile.absolute_overlay_role,
            )
            if profile.require_human_roles_at_start
            else (profile.baseline_role,)
        ),
        allow_missing_allocation_context=(
            not profile.require_human_roles_at_start
        ),
        transition_continuity_policy=transition_continuity_policy,
    )
    information_plan = CompositionInformationPlan(
        plan_id=f"{profile.profile_id}.information",
        context_requirements=(
            CompositionContextRequirement(
                channel=profile.allocation_context_channel,
                accepted_payload_contract_hashes=(
                    COMPOSITION_ALLOCATION_CONTEXT_CONTRACT_HASH_DEVELOPMENT,
                ),
                payload_access="inline",
                maximum_age_sec=float(evidence_ttl_sec),
                required=profile.require_human_roles_at_start,
            ),
        ),
        diagnostic_level="core",
    )
    authority_profile = ConservativeCompoundAuthorityProfile(
        profile_id=f"{profile.profile_id}.compound-authority",
        producer_id=_required(
            composer_producer_id,
            name="composer_producer_id",
        ),
        component_coverage=absolute_action_schema.component_ids,
        selection_groups=tuple(
            absolute_action_schema.component_catalog.selection_group_ids
        ),
        composition_spec_hash=(
            composition_contracts.composition_spec.semantic_content_hash
        ),
        component_rule_set_hash=(
            composition_contracts.component_rule_set.semantic_content_hash
        ),
        formal_strategy_spec_hash=(
            composition_contracts.strategy_spec.semantic_content_hash
        ),
    )
    evidence_window = AllocationEvidenceWindow(
        allocation_spec=allocation_spec,
        window_duration_sec=window_duration_sec,
        evidence_ttl_sec=evidence_ttl_sec,
        activity_normalization_scale_by_component=(
            activity_normalization_scale_by_component
        ),
        residual_normalization_scale_by_component=(
            residual_normalization_scale_by_component
        ),
        activity_metric_provider_by_component=(activity_metric_provider_by_component),
        activity_metric_provider_identity_by_component=(
            activity_metric_provider_identity_by_component
        ),
        residual_metric_provider_by_component=(residual_metric_provider_by_component),
        residual_metric_provider_identity_by_component=(
            residual_metric_provider_identity_by_component
        ),
    )
    return MujocoPolicyHumanAllocationProfileBundle(
        config=profile,
        input_contracts=input_contracts,
        allocation_spec=allocation_spec,
        composition_contracts=composition_contracts,
        information_plan=information_plan,
        authority_profile=authority_profile,
        evidence_window=evidence_window,
    )


def build_mujoco_policy_human_allocation_station_profiles(
    bundle: MujocoPolicyHumanAllocationProfileBundle,
    *,
    proposal_policy_node_id: str = "allocation-composer",
) -> MujocoPolicyHumanAllocationStationProfiles:
    """Bind one enabled allocation profile to its three Station gates.

    The helper creates no worker, source, runtime, authority, or target.  Its
    purpose is to prevent an application factory from enabling allocation
    inputs/context while accidentally omitting the profile-selected proposal
    promotion allowlist.
    """

    if not isinstance(bundle, MujocoPolicyHumanAllocationProfileBundle):
        raise TypeError("bundle must be a typed allocation profile bundle.")
    config = bundle.config
    if not config.enabled:
        raise ValueError("Station publication requires an enabled allocation profile.")
    profile_id = config.profile_id
    return MujocoPolicyHumanAllocationStationProfiles(
        derived_input=StationDerivedInputPublicationProfile(
            profile_id=f"{profile_id}.derived-input",
            enabled=True,
        ),
        allocation_context=StationAllocationPublicationProfile(
            profile_id=f"{profile_id}.allocation-context",
            enabled=True,
        ),
        proposal=MujocoAllocationPublicationProfile(
            profile_id=f"{profile_id}.proposal",
            enabled=True,
            send_actions=True,
            allocation_context_channel=config.allocation_context_channel,
            policy_node_id=proposal_policy_node_id,
            promotable_allocation_modes=config.proposal_promotable_modes,
        ),
    )


__all__ = [
    "MUJOCO_POLICY_HUMAN_ALLOCATION_PROMOTABLE_MODES_SCHEME_A",
    "MUJOCO_POLICY_HUMAN_ALLOCATION_PROFILE_OFF",
    "MujocoPolicyHumanAllocationProfileBundle",
    "MujocoPolicyHumanAllocationProfileConfig",
    "MujocoPolicyHumanAllocationStationProfiles",
    "build_mujoco_policy_human_allocation_profile",
    "build_mujoco_policy_human_allocation_station_profiles",
]
