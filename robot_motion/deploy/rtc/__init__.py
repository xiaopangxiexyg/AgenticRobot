"""Target-neutral RTC domain contracts.

This subpackage is the canonical home for new reusable RTC semantics.  It does
not own model transport, model-family adapters, target execution, safety,
reference generation, dispatch, teleoperation, replay, simulation, or
Control Station lifecycle.

Existing flat modules such as :mod:`robot_motion.deploy.rtc_runtime` and
:mod:`robot_motion.deploy.rtc_protocol` remain compatibility surfaces while
their stable responsibilities are migrated incrementally.
"""

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
    "RTC_CANONICAL_CONTINUATION_PREFIX_HASH_DOMAIN_V1",
    "RTC_CONTINUATION_ENCODING_RECEIPT_HASH_DOMAIN_V1",
    "RTC_CONTINUATION_PREFIX_SOURCE_PRODUCER_LOCAL_OVERLAP_V1",
    "RTC_CONTINUATION_SAMPLER_CONSTRAINT_CLAMPED_V1",
    "RTC_CONTINUATION_SAMPLER_CONSTRAINT_WEIGHTED_V1",
    "RTC_CONTINUATION_SELECTION_OUTCOME_HASH_DOMAIN_V1",
    "RTC_CONTINUATION_SELECTION_SPEC_HASH_DOMAIN_V1",
    "RTC_CONTINUATION_TIMELINE_HASH_DOMAIN_V1",
    "RTC_PLAN_CONTINUATION_ENTRY_HASH_DOMAIN_V1",
    "RTC_PLAN_CONTINUATION_SLICE_HASH_DOMAIN_V1",
    "RTC_PRODUCER_PROFILE_HASH_DOMAIN_V1",
    "RtcContinuationEncodingReceipt",
    "RtcContinuationSelectionOutcome",
    "RtcContinuationSelectionSpec",
    "RtcContinuationTimeline",
    "RtcPlanContinuationEntry",
    "RtcPlanContinuationSlice",
    "RtcProducerProfile",
    "rtc_continuation_timeline_from_request",
    "select_rtc_plan_continuation",
]
