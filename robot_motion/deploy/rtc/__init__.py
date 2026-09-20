"""Target-neutral RTC domain contracts.

This subpackage is the canonical home for new reusable RTC semantics.  It does
not own model transport, model-family adapters, target execution, safety,
reference generation, dispatch, teleoperation, replay, simulation, or
Control Station lifecycle.

Existing flat modules such as :mod:`robot_motion.deploy.rtc_runtime` and
:mod:`robot_motion.deploy.rtc_protocol` remain compatibility surfaces while
their stable responsibilities are migrated incrementally.
"""

from .admission import (
    RTC_LATENCY_ENVELOPE_STATUSES,
    RTC_LATENCY_LATE_BUT_USABLE,
    RTC_LATENCY_ON_TIME,
    RTC_LATENCY_STARVATION_RISK,
    RTC_REQUEST_BRIDGE_GUARD_STEPS,
    RtcExecutionCoverageAssessment,
    RtcLatencyEnvelopeAssessment,
    assess_rtc_execution_coverage,
    assess_rtc_latency_envelope,
    rtc_latency_budget_steps,
)
from .conditioning import (
    CANONICAL_RTC_ACTION_CONTENT_HASH_DOMAIN_V1,
    CANONICAL_RTC_CONDITIONING_BUNDLE_HASH_DOMAIN_V1,
    CANONICAL_RTC_CONDITIONING_BUNDLE_SCHEMA_V1,
    RTC_CONDITIONING_MODE_PREFIX_V1,
    RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1,
    RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1,
    RTC_PREFIX_SCHEDULE_HARD_DELAY_PREFIX_V1,
    CanonicalRtcConditioningBundle,
    RtcModelContinuationRuntimeSpec,
    build_canonical_rtc_conditioning_bundle,
)
from .continuation import (
    RTC_CANONICAL_CONTINUATION_PREFIX_HASH_DOMAIN_V1,
    RTC_CONTINUATION_ENCODING_RECEIPT_HASH_DOMAIN_V1,
    RTC_CONTINUATION_PREFIX_SOURCE_PRODUCER_LOCAL_OVERLAP_V1,
    RTC_CONTINUATION_SAMPLER_CONSTRAINT_CLAMPED_V1,
    RTC_CONTINUATION_SAMPLER_CONSTRAINT_WEIGHTED_V1,
    RTC_CONTINUATION_SELECTION_OUTCOME_HASH_DOMAIN_V1,
    RTC_CONTINUATION_SELECTION_SPEC_HASH_DOMAIN_V1,
    RTC_CONTINUATION_TIMELINE_HASH_DOMAIN_V1,
    RTC_PLAN_CONTINUATION_ENTRY_HASH_DOMAIN_V1,
    RTC_PLAN_CONTINUATION_SLICE_HASH_DOMAIN_V1,
    RtcContinuationEncodingReceipt,
    RtcContinuationSelectionOutcome,
    RtcContinuationSelectionSpec,
    RtcContinuationTimeline,
    RtcPlanContinuationEntry,
    RtcPlanContinuationSlice,
    rtc_continuation_timeline_from_request,
    select_rtc_plan_continuation,
)
from .profile import (
    RTC_PRODUCER_PROFILE_HASH_DOMAIN_V1,
    RtcProducerProfile,
)

__all__ = [
    "CANONICAL_RTC_ACTION_CONTENT_HASH_DOMAIN_V1",
    "CANONICAL_RTC_CONDITIONING_BUNDLE_HASH_DOMAIN_V1",
    "CANONICAL_RTC_CONDITIONING_BUNDLE_SCHEMA_V1",
    "RTC_CANONICAL_CONTINUATION_PREFIX_HASH_DOMAIN_V1",
    "RTC_CONDITIONING_MODE_PREFIX_V1",
    "RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1",
    "RTC_CONTINUATION_ENCODING_RECEIPT_HASH_DOMAIN_V1",
    "RTC_CONTINUATION_PREFIX_SOURCE_PRODUCER_LOCAL_OVERLAP_V1",
    "RTC_CONTINUATION_SAMPLER_CONSTRAINT_CLAMPED_V1",
    "RTC_CONTINUATION_SAMPLER_CONSTRAINT_WEIGHTED_V1",
    "RTC_CONTINUATION_SELECTION_OUTCOME_HASH_DOMAIN_V1",
    "RTC_CONTINUATION_SELECTION_SPEC_HASH_DOMAIN_V1",
    "RTC_CONTINUATION_TIMELINE_HASH_DOMAIN_V1",
    "RTC_LATENCY_ENVELOPE_STATUSES",
    "RTC_LATENCY_LATE_BUT_USABLE",
    "RTC_LATENCY_ON_TIME",
    "RTC_LATENCY_STARVATION_RISK",
    "RTC_PLAN_CONTINUATION_ENTRY_HASH_DOMAIN_V1",
    "RTC_PLAN_CONTINUATION_SLICE_HASH_DOMAIN_V1",
    "RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1",
    "RTC_PREFIX_SCHEDULE_HARD_DELAY_PREFIX_V1",
    "RTC_PRODUCER_PROFILE_HASH_DOMAIN_V1",
    "RTC_REQUEST_BRIDGE_GUARD_STEPS",
    "CanonicalRtcConditioningBundle",
    "RtcContinuationEncodingReceipt",
    "RtcContinuationSelectionOutcome",
    "RtcContinuationSelectionSpec",
    "RtcContinuationTimeline",
    "RtcExecutionCoverageAssessment",
    "RtcLatencyEnvelopeAssessment",
    "RtcModelContinuationRuntimeSpec",
    "RtcPlanContinuationEntry",
    "RtcPlanContinuationSlice",
    "RtcProducerProfile",
    "assess_rtc_execution_coverage",
    "assess_rtc_latency_envelope",
    "build_canonical_rtc_conditioning_bundle",
    "rtc_continuation_timeline_from_request",
    "rtc_latency_budget_steps",
    "select_rtc_plan_continuation",
]
