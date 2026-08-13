"""Post-transition execution readiness for Sim-HIL source switches.

This application mixin does not resolve, admit, or dispatch targets itself. It
drives the already-selected control-cycle port until the lower runtime returns
one typed, admitted dispatch result for the new activation.  The exclusive
HITL transaction can therefore distinguish source priming from executable
handoff readiness without reconstructing lower-owned state.
"""

from __future__ import annotations

from dataclasses import dataclass

from .validation import _finite


@dataclass(frozen=True)
class SimHILActivationExecutionEvidence:
    target_source: str
    cycle_index: int
    proposal_status: str
    execution_status: str
    target_id: str
    target_hash: str
    target_admission_receipt_id: str
    target_completion_receipt_id: str
    command_id: str
    dispatch_receipt_id: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        source = str(self.target_source).strip()
        if source not in {"policy", "human"}:
            raise ValueError("target_source must be policy or human.")
        object.__setattr__(self, "target_source", source)
        if int(self.cycle_index) < 0:
            raise ValueError("cycle_index must be non-negative.")
        object.__setattr__(self, "cycle_index", int(self.cycle_index))
        for name in (
            "proposal_status",
            "execution_status",
            "target_id",
            "target_hash",
            "target_admission_receipt_id",
            "target_completion_receipt_id",
            "command_id",
            "dispatch_receipt_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        refs = tuple(str(value).strip() for value in self.evidence_refs)
        if not refs or any(not value for value in refs):
            raise ValueError("evidence_refs must contain non-empty IDs.")
        if len(set(refs)) != len(refs):
            raise ValueError("evidence_refs cannot contain duplicates.")
        object.__setattr__(self, "evidence_refs", refs)

    def to_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


class _SimHILApplicationTransitionMixin:
    def establish_activation_execution(
        self,
        *,
        target_source: str,
        timeout_sec: float | None = None,
    ) -> SimHILActivationExecutionEvidence:
        """Wait for one lower-admitted target from the new source activation."""

        source = str(target_source).strip()
        expected_mode = {
            "policy": "base_only",
            "human": "absolute_overlay_owner",
        }.get(source)
        if expected_mode is None:
            raise ValueError("target_source must be policy or human.")
        timeout = _finite(
            self.config.warmup_timeout_sec
            if timeout_sec is None
            else timeout_sec,
            name="activation_execution.timeout_sec",
        )
        if timeout <= 0.0:
            raise ValueError("activation execution timeout must be positive.")
        deadline = _finite(
            self.clock(),
            name="activation_execution.started_at",
        ) + timeout
        last_status = "not_started"
        while True:
            now = _finite(self.clock(), name="activation_execution.now")
            if now >= deadline:
                raise TimeoutError(
                    "New source activation produced no admitted lower target "
                    f"within {timeout:.3f}s; last_status={last_status}."
                )
            due_at = self.next_decision_at
            if due_at is not None and now < due_at:
                self.sleep(min(float(due_at) - now, 0.005))
                continue
            cycle = self.step(now=now)
            publication = cycle.composition_step
            execution = cycle.execution
            proposal_status = str(getattr(publication, "status", "") or "")
            execution_status = str(getattr(execution, "status", "") or "")
            last_status = f"{proposal_status}/{execution_status}"
            selected_modes = set(
                dict(
                    getattr(publication, "selected_mode_by_component", {})
                    or {}
                ).values()
            )
            if selected_modes and selected_modes != {expected_mode}:
                raise RuntimeError(
                    "Post-switch cycle selected another source mode: "
                    f"expected {expected_mode}, got {sorted(selected_modes)}."
                )
            if execution_status in {
                "compound_resolution_failure_hold",
                "execution_hold_active",
                "action_adaptation_rejected_keep_existing",
                "execution_cancellation_incomplete",
            }:
                raise RuntimeError(
                    "New source activation was not executable: "
                    f"{last_status}."
                )
            if execution_status != "dispatched":
                continue
            dispatch = getattr(execution, "target_dispatch_result", None)
            admission = (
                None if dispatch is None else dispatch.target_admission
            )
            completion = (
                None if dispatch is None else dispatch.completion_receipt
            )
            target = None if dispatch is None else dispatch.target
            if (
                proposal_status != "proposal_admitted"
                or dispatch is None
                or admission is None
                or not bool(admission.accepted)
                or completion is None
                or completion.status != "resolved"
                or target is None
            ):
                raise RuntimeError(
                    "Dispatched switch cycle lacks an admitted resolved target."
                )
            refs = tuple(
                dict.fromkeys(
                    value
                    for value in (
                        getattr(execution.target_execution_request, "request_id", None),
                        getattr(execution.target_submission_receipt, "receipt_id", None),
                        completion.receipt_id,
                        target.target_id,
                        admission.receipt_id,
                        execution.command_id,
                        execution.dispatch_receipt_id,
                    )
                    if value is not None and str(value).strip()
                )
            )
            return SimHILActivationExecutionEvidence(
                target_source=source,
                cycle_index=int(cycle.cycle_index),
                proposal_status=proposal_status,
                execution_status=execution_status,
                target_id=target.target_id,
                target_hash=target.semantic_content_hash,
                target_admission_receipt_id=admission.receipt_id,
                target_completion_receipt_id=completion.receipt_id,
                command_id=str(execution.command_id),
                dispatch_receipt_id=str(execution.dispatch_receipt_id),
                evidence_refs=refs,
            )


__all__ = ["SimHILActivationExecutionEvidence"]
