"""Non-authoritative source-difference and guidance projections."""

from __future__ import annotations

from .difference import (
    SOURCE_DIFFERENCE_OBSERVATION_HASH_DOMAIN_V1,
    SOURCE_DIFFERENCE_SPEC_HASH_DOMAIN_V1,
    ActionDifference,
    ComponentDifferenceMetric,
    SourceDifferenceObservation,
    SourceDifferenceSpec,
    action_difference_semantic_payload,
    evaluate_composition_source_differences,
)
from .feedback import (
    GUIDANCE_FEEDBACK_COMMAND_HASH_DOMAIN_V1,
    GUIDANCE_FEEDBACK_RECEIPT_HASH_DOMAIN_V1,
    GUIDANCE_FEEDBACK_SPEC_HASH_DOMAIN_V1,
    GuidanceFeedbackAdapter,
    GuidanceFeedbackCommand,
    GuidanceFeedbackCue,
    GuidanceFeedbackReceipt,
    GuidanceFeedbackSpec,
    NoIoGuidanceFeedbackAdapter,
    build_guidance_feedback_command,
)

__all__ = [
    "SOURCE_DIFFERENCE_OBSERVATION_HASH_DOMAIN_V1",
    "SOURCE_DIFFERENCE_SPEC_HASH_DOMAIN_V1",
    "ActionDifference",
    "ComponentDifferenceMetric",
    "SourceDifferenceObservation",
    "SourceDifferenceSpec",
    "action_difference_semantic_payload",
    "evaluate_composition_source_differences",
    "GUIDANCE_FEEDBACK_COMMAND_HASH_DOMAIN_V1",
    "GUIDANCE_FEEDBACK_RECEIPT_HASH_DOMAIN_V1",
    "GUIDANCE_FEEDBACK_SPEC_HASH_DOMAIN_V1",
    "GuidanceFeedbackAdapter",
    "GuidanceFeedbackCommand",
    "GuidanceFeedbackCue",
    "GuidanceFeedbackReceipt",
    "GuidanceFeedbackSpec",
    "NoIoGuidanceFeedbackAdapter",
    "build_guidance_feedback_command",
]
