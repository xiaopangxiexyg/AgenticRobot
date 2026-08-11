"""Deployment-owned detection of position source-plan boundaries.

This adapter reads policy-emission or synchronous-queue lineage together with
acknowledged synchronous Hold events.  It does not interpolate, solve IK, or
choose a motion response; those semantics belong to
:mod:`robot_motion.position_control.plan_splice`.
"""

from __future__ import annotations

from dataclasses import dataclass

from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.runtime import PolicyActionEmission
from robot_motion.position_control.plan_splice import (
    POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
    POSITION_PLAN_GAP_NONE,
    POSITION_PLAN_RELATION_CONTIGUOUS,
    POSITION_PLAN_RELATION_REPLACEMENT,
    POSITION_PLAN_RELATION_SOURCE_GAP,
    POSITION_PLAN_RELATION_STREAM_START,
    PositionSourcePlanCut,
)


@dataclass(frozen=True)
class DeploymentPositionPlanCutProposal:
    token: int
    source_event_id: str
    source_plan_id: str
    source_step: int
    cut: PositionSourcePlanCut


class DeploymentPositionPlanCutDetector:
    """Transactionally classify only successfully submitted policy actions."""

    def __init__(self) -> None:
        self._last_source_plan_id: str | None = None
        self._last_source_step: int | None = None
        self._pending_hold_acknowledged = False
        self._last_hold_generation = 0
        self._pending: DeploymentPositionPlanCutProposal | None = None
        self._next_token = 0

    @staticmethod
    def _source_plan_id(emission: PolicyActionEmission) -> str:
        return (
            f"{emission.policy_node_id}:"
            f"{emission.inference_id}:"
            f"{int(emission.chunk_id)}"
        )

    @staticmethod
    def _sync_action_lineage(
        action: NormalizedAction,
        *,
        policy_node_id: str,
    ) -> tuple[str, str, int, dict[str, object]]:
        """Read the authoritative synchronous queue cursor from one action.

        ``PolicyActionEmission`` is a richer observation of the same source
        action, but emission construction is deliberately passive on the
        legacy synchronous path.  Position-plan ownership therefore cannot
        depend on that optional observer succeeding.  The queue-owned chunk
        tags are attached before admission and survive canonical preparation.
        """

        if not isinstance(action, NormalizedAction):
            raise TypeError("action must be a NormalizedAction.")
        normalized_policy_node_id = str(policy_node_id).strip()
        if not normalized_policy_node_id:
            raise ValueError("policy_node_id must be non-empty.")
        metadata = dict(action.metadata)
        try:
            chunk_id = int(metadata["deploy_action_chunk_id"])
            source_step = int(metadata["deploy_action_chunk_index"])
            chunk_size = int(metadata["deploy_action_chunk_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Synchronous position action is missing queue-owned chunk lineage."
            ) from exc
        if chunk_id < 0 or source_step < 0 or chunk_size <= 0:
            raise ValueError("Synchronous position chunk lineage must be non-negative.")
        if source_step >= chunk_size:
            raise ValueError(
                "Synchronous position chunk index is outside its chunk size."
            )
        source_plan_id = (
            f"{normalized_policy_node_id}:sync-chunk:{chunk_id}"
        )
        source_event_id = f"{source_plan_id}:step:{source_step}"
        return (
            source_event_id,
            source_plan_id,
            source_step,
            {
                "owner": "teleop_runtime.deploy",
                "lineage_source": "normalized_action_chunk_metadata",
                "policy_node_id": normalized_policy_node_id,
                "chunk_id": chunk_id,
                "chunk_size": chunk_size,
            },
        )

    def mark_hold_acknowledged(self) -> None:
        """Record an actual Hold boundary for the next admitted source step."""

        self._pending_hold_acknowledged = True

    def observe_hold_generation(self, generation: int) -> None:
        """Idempotently observe a deployment executor clear/Hold receipt."""

        if isinstance(generation, bool) or int(generation) != generation:
            raise ValueError("hold generation must be a non-negative integer.")
        normalized = int(generation)
        if normalized < self._last_hold_generation:
            raise ValueError("hold generation cannot move backwards.")
        if normalized > self._last_hold_generation:
            self._pending_hold_acknowledged = True
            self._last_hold_generation = normalized

    def prepare(
        self,
        emission: PolicyActionEmission,
        *,
        detected_at: float,
    ) -> DeploymentPositionPlanCutProposal:
        if not isinstance(emission, PolicyActionEmission):
            raise TypeError("emission must be a PolicyActionEmission.")
        if self._pending is not None:
            raise RuntimeError("a position plan-cut proposal is already pending.")
        return self._prepare_source_step(
            source_event_id=emission.emission_id,
            source_plan_id=self._source_plan_id(emission),
            source_step=int(emission.chunk_keyframe_index),
            detected_at=detected_at,
            evidence={
                "owner": "teleop_runtime.deploy",
                "lineage_source": "policy_action_emission",
                "emission_id": emission.emission_id,
                "policy_output_id": emission.policy_output_id,
                "target_effective_at": float(emission.target_effective_at),
            },
        )

    def prepare_synchronous_action(
        self,
        action: NormalizedAction,
        *,
        policy_node_id: str,
        detected_at: float,
    ) -> DeploymentPositionPlanCutProposal:
        """Classify a synchronous action without depending on passive emission."""

        (
            source_event_id,
            source_plan_id,
            source_step,
            evidence,
        ) = self._sync_action_lineage(
            action,
            policy_node_id=policy_node_id,
        )
        return self._prepare_source_step(
            source_event_id=source_event_id,
            source_plan_id=source_plan_id,
            source_step=source_step,
            detected_at=detected_at,
            evidence=evidence,
        )

    def _prepare_source_step(
        self,
        *,
        source_event_id: str,
        source_plan_id: str,
        source_step: int,
        detected_at: float,
        evidence: dict[str, object],
    ) -> DeploymentPositionPlanCutProposal:
        if self._pending is not None:
            raise RuntimeError("a position plan-cut proposal is already pending.")
        current_plan = str(source_plan_id)
        current_step = int(source_step)
        previous_plan = self._last_source_plan_id
        previous_step = self._last_source_step
        if previous_plan is None:
            relation = POSITION_PLAN_RELATION_STREAM_START
        elif current_plan != previous_plan:
            relation = POSITION_PLAN_RELATION_REPLACEMENT
        elif current_step == int(previous_step) + 1:
            relation = POSITION_PLAN_RELATION_CONTIGUOUS
        else:
            relation = POSITION_PLAN_RELATION_SOURCE_GAP
        # Before any source plan has been committed there is no stream whose
        # continuity a prior Hold could break.  Stream start already requires
        # zero dynamics and a measured anchor, so it dominates qpos-init or
        # startup clears that occurred before the first policy source step.
        if previous_plan is None:
            gap = POSITION_PLAN_GAP_NONE
        elif self._pending_hold_acknowledged:
            gap = POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED
        else:
            gap = POSITION_PLAN_GAP_NONE
        token = self._next_token
        self._next_token += 1
        cut = PositionSourcePlanCut(
            cut_id=f"deploy-position-plan-cut:{source_event_id}",
            current_source_plan_id=current_plan,
            current_source_step=current_step,
            previous_source_plan_id=previous_plan,
            previous_source_step=previous_step,
            relation=relation,
            execution_gap=gap,
            detected_at=float(detected_at),
            evidence=evidence,
        )
        proposal = DeploymentPositionPlanCutProposal(
            token=token,
            source_event_id=source_event_id,
            source_plan_id=current_plan,
            source_step=current_step,
            cut=cut,
        )
        self._pending = proposal
        return proposal

    def commit(self, proposal: DeploymentPositionPlanCutProposal) -> None:
        self._require_pending(proposal)
        self._last_source_plan_id = proposal.source_plan_id
        self._last_source_step = proposal.source_step
        self._pending_hold_acknowledged = False
        self._pending = None

    def reject(self, proposal: DeploymentPositionPlanCutProposal) -> None:
        self._require_pending(proposal)
        self._pending = None

    def reset(self) -> None:
        self._last_source_plan_id = None
        self._last_source_step = None
        self._pending_hold_acknowledged = False
        self._pending = None

    def _require_pending(
        self,
        proposal: DeploymentPositionPlanCutProposal,
    ) -> None:
        if not isinstance(proposal, DeploymentPositionPlanCutProposal):
            raise TypeError(
                "proposal must be a DeploymentPositionPlanCutProposal."
            )
        if self._pending is None or self._pending.token != proposal.token:
            raise RuntimeError("position plan-cut proposal is stale.")


__all__ = [
    "DeploymentPositionPlanCutDetector",
    "DeploymentPositionPlanCutProposal",
]
