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
from .activation import (
    ACTIVATION_LOCAL_INPUT_FRESHNESS_POLICY_V2,
    ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2,
    CompositionActivationInputVerdict,
    CompositionActivationScope,
    build_composition_activation_scope,
    evaluate_composition_input_activation,
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
__all__ = [
    "ACTIVATION_LOCAL_INPUT_FRESHNESS_POLICY_V2",
    "ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2",
    "CompositionClock",
    "CompositionActivationInputVerdict",
    "CompositionActivationScope",
    "CompositionActivationBinding",
    "CompositionContextRequirement",
    "CompositionDomainEvent",
    "CompositionEventEmitter",
    "CompositionInformationPlan",
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
    "build_composition_activation_scope",
    "evaluate_composition_input_activation",
    "build_composition_source_eligibility_snapshot",
    "project_composition_source_eligibility",
]
