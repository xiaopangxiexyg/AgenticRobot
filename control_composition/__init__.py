"""Backend-neutral multi-source control composition.

This package owns composition semantics and orchestration seams.  It does not
own canonical motion authority, safety, execution, a model worker, a replay
cursor, a device lifecycle, or an episode writer.
"""

from __future__ import annotations

from .ports import (
    CompositionClock,
    CompositionEventEmitter,
    CompositionProposalAdmitter,
)
from .events import (
    CompositionActivationBinding,
    CompositionDomainEvent,
    CompositionSourceEligibilityBinding,
)
from .information import (
    CompositionContextRequirement,
    CompositionInformationPlan,
)
from .temporal import (
    COMPOSITION_ACTION_TIMESTAMP_SEMANTICS,
    COMPOSITION_OVERLAP_BEHAVIORS,
    COMPOSITION_TEMPORAL_SELECTION_SPEC_HASH_DOMAIN,
    DEFAULT_COMPOSITION_TEMPORAL_SELECTION_SPEC,
    CompositionTemporalSelectionSpec,
)
from .temporal_sampling import (
    COMPONENT_ACTION_INTERPOLATION_CONTRACT_HASH_DOMAIN,
    COMPONENT_ACTION_TEMPORAL_SAMPLE_HASH_DOMAIN,
    GUARANTEED_NEXT_COMPONENT_ACTION_HASH_DOMAIN,
    SOURCE_TEMPORAL_DERIVATION_KINDS,
    ComponentActionTemporalSample,
    GuaranteedNextComponentAction,
    sample_component_action_source,
)
from .activation import (
    ACTIVATION_LOCAL_INPUT_FRESHNESS_POLICY_V2,
    ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2,
    CompositionActivationInputVerdict,
    CompositionActivationScope,
    build_composition_activation_scope,
    evaluate_composition_input_activation,
)
from .residual_anchor import (
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    ResidualAnchorState,
)
from .eligibility import (
    COMPOSITION_SOURCE_STATUSES_V1,
    CompositionSourceEligibilityAdmissionError,
    CompositionSourceEligibilityAdmissionReceipt,
    CompositionSourceEligibilityDraft,
    CompositionSourceEligibilityProjection,
    CompositionSourceEligibilitySnapshot,
    CompositionSourceEligibilitySpec,
    CompositionSourceEligibilityState,
    CompositionSourceEligibilityStore,
    build_composition_source_eligibility_snapshot,
    project_composition_source_eligibility,
)
from .retention import TimeWindowRetentionSpec, TimeWindowStoreSnapshot
from .composition.input_adapter import (
    COMPOSITION_INPUT_DRAFT_ADAPTER_CONTRACT_ID_V1,
    TemporallySampledCompositionInput,
    build_composition_input_draft,
    build_robot_action_v1_composition_input_draft,
    build_temporally_sampled_composition_input,
)
__all__ = [
    "ACTIVATION_LOCAL_INPUT_FRESHNESS_POLICY_V2",
    "ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2",
    "CompositionClock",
    "CompositionActivationInputVerdict",
    "CompositionActivationScope",
    "ResidualAnchorComponentBinding",
    "ResidualAnchorReceipt",
    "ResidualAnchorRequest",
    "ResidualAnchorState",
    "CompositionActivationBinding",
    "CompositionContextRequirement",
    "CompositionDomainEvent",
    "CompositionEventEmitter",
    "CompositionInformationPlan",
    "COMPOSITION_ACTION_TIMESTAMP_SEMANTICS",
    "COMPOSITION_OVERLAP_BEHAVIORS",
    "COMPOSITION_TEMPORAL_SELECTION_SPEC_HASH_DOMAIN",
    "DEFAULT_COMPOSITION_TEMPORAL_SELECTION_SPEC",
    "CompositionTemporalSelectionSpec",
    "COMPONENT_ACTION_INTERPOLATION_CONTRACT_HASH_DOMAIN",
    "COMPONENT_ACTION_TEMPORAL_SAMPLE_HASH_DOMAIN",
    "GUARANTEED_NEXT_COMPONENT_ACTION_HASH_DOMAIN",
    "SOURCE_TEMPORAL_DERIVATION_KINDS",
    "ComponentActionTemporalSample",
    "GuaranteedNextComponentAction",
    "sample_component_action_source",
    "CompositionProposalAdmitter",
    "COMPOSITION_SOURCE_STATUSES_V1",
    "CompositionSourceEligibilityAdmissionError",
    "CompositionSourceEligibilityAdmissionReceipt",
    "CompositionSourceEligibilityBinding",
    "CompositionSourceEligibilityDraft",
    "CompositionSourceEligibilityProjection",
    "CompositionSourceEligibilitySnapshot",
    "CompositionSourceEligibilitySpec",
    "CompositionSourceEligibilityState",
    "CompositionSourceEligibilityStore",
    "TimeWindowRetentionSpec",
    "TimeWindowStoreSnapshot",
    "COMPOSITION_INPUT_DRAFT_ADAPTER_CONTRACT_ID_V1",
    "TemporallySampledCompositionInput",
    "build_composition_activation_scope",
    "build_composition_input_draft",
    "build_robot_action_v1_composition_input_draft",
    "build_temporally_sampled_composition_input",
    "evaluate_composition_input_activation",
    "build_composition_source_eligibility_snapshot",
    "project_composition_source_eligibility",
]
