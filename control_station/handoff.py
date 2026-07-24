"""Compatibility imports for the canonical control-composition handoff.

New code should import from ``control_composition.lifecycle``.  This explicit
surface preserves existing Station callers without duplicating contracts,
state, or reducer behavior.
"""

from __future__ import annotations

from control_composition.lifecycle.handoff import (
    EPISODE_BOUNDARY_MOTION_POLICIES_V1,
    HUMAN_TAKEOVER_PHASES_V1,
    POLICY_SUSPENSION_MODES_V1,
    AuthorityReleaseReceipt,
    ExecutionHoldReceipt,
    ExecutionReadyReceipt,
    HumanAuthorityTakeoverReceipt,
    HumanReadinessReceipt,
    HumanTakeoverOperationEvidence,
    HumanTakeoverOperationSpec,
    HumanTakeoverRequest,
    HumanTakeoverState,
    MixedControlBoundaryPolicy,
    SourceInvalidationReceipt,
    apply_authority_release,
    apply_execution_hold,
    apply_execution_ready,
    apply_human_authority_takeover,
    apply_human_readiness,
    apply_source_invalidation,
    begin_human_takeover,
    fail_human_takeover,
)

__all__ = [
    "EPISODE_BOUNDARY_MOTION_POLICIES_V1",
    "HUMAN_TAKEOVER_PHASES_V1",
    "POLICY_SUSPENSION_MODES_V1",
    "AuthorityReleaseReceipt",
    "ExecutionHoldReceipt",
    "ExecutionReadyReceipt",
    "HumanAuthorityTakeoverReceipt",
    "HumanReadinessReceipt",
    "HumanTakeoverOperationEvidence",
    "HumanTakeoverOperationSpec",
    "HumanTakeoverRequest",
    "HumanTakeoverState",
    "MixedControlBoundaryPolicy",
    "SourceInvalidationReceipt",
    "apply_authority_release",
    "apply_execution_hold",
    "apply_execution_ready",
    "apply_human_authority_takeover",
    "apply_human_readiness",
    "apply_source_invalidation",
    "begin_human_takeover",
    "fail_human_takeover",
]
