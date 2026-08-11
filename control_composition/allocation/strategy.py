"""Typed component-rule evaluation for allocation-selected composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_motion.action.component_action import ComponentActionSchema

from control_composition.composition.component_contract import (
    AllocatedPoseComponentCompositionRule,
    AllocatedQposComponentCompositionRule,
)
from control_composition.composition.component_operations import (
    component_record_available,
    record_matches_declared_schema,
    relative_pose_component_value,
    relative_qpos_component_value,
)
from control_composition.composition.formal_strategy import (
    FormalCompositionStrategyInput,
)
from control_composition.residual_anchor import (
    RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND,
)

from .context import (
    selected_allocation_anchor_from_context,
    selected_allocation_modes_from_context,
)
from .contracts import (
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
)


AllocatedComponentCompositionRule = (
    AllocatedPoseComponentCompositionRule | AllocatedQposComponentCompositionRule
)


@dataclass(frozen=True)
class AllocatedComponentRuleEvaluation:
    output_status: str
    reason: str
    component_id: str
    value: np.ndarray | None
    output_action_schema: ComponentActionSchema | None
    used_context_id: str | None
    selected_mode: str | None = None

    def __post_init__(self) -> None:
        if self.output_status not in {"ready", "no_action", "invalid"}:
            raise ValueError("Unsupported allocated-rule output status.")
        if (self.output_status == "ready") != (
            self.value is not None and self.output_action_schema is not None
        ):
            raise ValueError("A ready allocated-rule result requires value and schema.")
        if (self.output_status == "ready") != (self.selected_mode is not None):
            raise ValueError("A ready allocated-rule result requires a selected mode.")


def _blocked(
    rule: AllocatedComponentCompositionRule,
    *,
    reason: str,
    context_id: str | None = None,
) -> AllocatedComponentRuleEvaluation:
    return AllocatedComponentRuleEvaluation(
        output_status=rule.missing_behavior,
        reason=reason,
        component_id=rule.component_id,
        value=None,
        output_action_schema=None,
        used_context_id=context_id,
    )


def _baseline_only(
    strategy_input: FormalCompositionStrategyInput,
    *,
    rule: AllocatedComponentCompositionRule,
    reason: str,
    context_id: str | None = None,
) -> AllocatedComponentRuleEvaluation:
    baseline = strategy_input.input_snapshot.selected_input_by_role.get(
        rule.base_absolute_role
    )
    if baseline is None or not component_record_available(
        baseline,
        component_id=rule.component_id,
        representation=rule.representation,
    ):
        return _blocked(
            rule,
            reason="allocation_baseline_unavailable",
            context_id=context_id,
        )
    if not record_matches_declared_schema(strategy_input, baseline):
        return AllocatedComponentRuleEvaluation(
            output_status="invalid",
            reason="allocation_baseline_schema_mismatch",
            component_id=rule.component_id,
            value=None,
            output_action_schema=None,
            used_context_id=context_id,
        )
    output_schema = baseline.action.action_schema
    if (
        output_schema.semantic_content_hash
        != strategy_input.input_snapshot.action_schema_hash
    ):
        return AllocatedComponentRuleEvaluation(
            output_status="invalid",
            reason="allocation_baseline_output_schema_mismatch",
            component_id=rule.component_id,
            value=None,
            output_action_schema=None,
            used_context_id=context_id,
        )
    return AllocatedComponentRuleEvaluation(
        output_status="ready",
        reason=reason,
        component_id=rule.component_id,
        value=np.array(
            baseline.action.value(rule.component_id).values,
            copy=True,
        ),
        output_action_schema=output_schema,
        used_context_id=context_id,
        selected_mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    )


def evaluate_allocated_component_rule(
    strategy_input: FormalCompositionStrategyInput,
    *,
    rule: AllocatedComponentCompositionRule,
    allocation_component_ids: tuple[str, ...],
) -> AllocatedComponentRuleEvaluation:
    """Evaluate one mode using inputs and context from the same formal cut."""

    context_snapshot = strategy_input.context_snapshot
    if context_snapshot is None:
        if rule.context_missing_behavior == "use_baseline":
            return _baseline_only(
                strategy_input,
                rule=rule,
                reason="allocation_context_snapshot_missing_use_baseline",
            )
        return _blocked(rule, reason="allocation_context_snapshot_missing")
    context = context_snapshot.selected_context_by_channel.get(
        rule.allocation_context_channel
    )
    if context is None or context.availability != "available":
        if rule.context_missing_behavior == "use_baseline":
            return _baseline_only(
                strategy_input,
                rule=rule,
                reason="allocation_context_unavailable_use_baseline",
            )
        return _blocked(rule, reason="allocation_context_unavailable")

    lifecycle = strategy_input.residual_anchor_lifecycle_snapshot
    if lifecycle is None:
        return _blocked(rule, reason="residual_anchor_lifecycle_missing")
    if lifecycle.status != "ready" or lifecycle.ready_receipt is None:
        return _blocked(rule, reason=lifecycle.reason)
    active_receipt = lifecycle.ready_receipt
    try:
        modes = selected_allocation_modes_from_context(
            context,
            expected_allocation_spec_hash=rule.allocation_spec_hash,
            expected_component_ids=allocation_component_ids,
            target_effective_at=(strategy_input.input_snapshot.target_effective_at),
            expected_composition_activation_id=(
                strategy_input.input_snapshot.expected_activation_id
            ),
            expected_composition_activation_hash=(
                strategy_input.input_snapshot.expected_activation_hash
            ),
        )
        anchor_binding = selected_allocation_anchor_from_context(
            context,
            expected_allocation_spec_hash=rule.allocation_spec_hash,
            expected_component_ids=allocation_component_ids,
            target_effective_at=(
                strategy_input.input_snapshot.target_effective_at
            ),
            expected_composition_activation_id=(
                strategy_input.input_snapshot.expected_activation_id
            ),
            expected_composition_activation_hash=(
                strategy_input.input_snapshot.expected_activation_hash
            ),
        )
    except (TypeError, ValueError) as exc:
        return AllocatedComponentRuleEvaluation(
            output_status="invalid",
            reason=f"allocation_context_invalid:{exc}",
            component_id=rule.component_id,
            value=None,
            output_action_schema=None,
            used_context_id=context.context_id,
        )

    if context.semantic_payload.get("allocation_scope_id") != (
        lifecycle.lifecycle_scope_id
    ):
        return _blocked(
            rule,
            reason="allocation_anchor_lifecycle_scope_mismatch",
            context_id=context.context_id,
        )
    if (
        anchor_binding.receipt_id != active_receipt.receipt_id
        or anchor_binding.receipt_hash != active_receipt.semantic_content_hash
        or anchor_binding.request_generation != active_receipt.request_generation
    ):
        return _blocked(
            rule,
            reason="allocation_active_anchor_mismatch",
            context_id=context.context_id,
        )

    selected_inputs = strategy_input.input_snapshot.selected_input_by_role
    for role in (rule.relative_overlay_role, rule.absolute_overlay_role):
        selected = selected_inputs.get(role)
        if selected is None:
            continue
        if (
            selected.semantic_binding_id_by_kind.get(
                RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND
            )
            != active_receipt.receipt_id
            or selected.semantic_binding_hash_by_kind.get(
                RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND
            )
            != active_receipt.semantic_content_hash
        ):
            return _blocked(
                rule,
                reason=f"allocation_selected_input_anchor_mismatch:{role}",
                context_id=context.context_id,
            )
    baseline_evaluation = _baseline_only(
        strategy_input,
        rule=rule,
        reason="allocation_baseline_ready",
        context_id=context.context_id,
    )
    if baseline_evaluation.output_status != "ready":
        return baseline_evaluation
    baseline = selected_inputs[rule.base_absolute_role]
    assert baseline_evaluation.output_action_schema is not None
    output_schema = baseline_evaluation.output_action_schema

    mode = modes[rule.component_id]
    if mode == COMPOSITION_ALLOCATION_MODE_BASE_ONLY:
        value = np.array(
            baseline.action.value(rule.component_id).values,
            copy=True,
        )
    elif mode == COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY:
        delta = selected_inputs.get(rule.relative_overlay_role)
        if delta is None or not component_record_available(
            delta,
            component_id=rule.component_id,
            representation=rule.representation,
        ):
            return _blocked(
                rule,
                reason="allocation_relative_overlay_unavailable",
                context_id=context.context_id,
            )
        if (
            delta.semantic_binding_id_by_kind.get(
                RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND
            )
            != anchor_binding.receipt_id
            or delta.semantic_binding_hash_by_kind.get(
                RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND
            )
            != anchor_binding.receipt_hash
        ):
            return _blocked(
                rule,
                reason="allocation_relative_anchor_binding_mismatch",
                context_id=context.context_id,
            )
        if not record_matches_declared_schema(strategy_input, delta):
            return AllocatedComponentRuleEvaluation(
                output_status="invalid",
                reason="allocation_relative_overlay_schema_mismatch",
                component_id=rule.component_id,
                value=None,
                output_action_schema=None,
                used_context_id=context.context_id,
            )
        try:
            if isinstance(rule, AllocatedPoseComponentCompositionRule):
                value = relative_pose_component_value(
                    baseline,
                    delta,
                    component_id=rule.component_id,
                )
            else:
                value = relative_qpos_component_value(
                    baseline,
                    delta,
                    component_id=rule.component_id,
                    baseline_value_contract_hash=(rule.baseline_value_contract_hash),
                    delta_value_contract_hash=rule.delta_value_contract_hash,
                )
        except (KeyError, TypeError, ValueError) as exc:
            return AllocatedComponentRuleEvaluation(
                output_status="invalid",
                reason=f"allocation_relative_overlay_invalid:{exc}",
                component_id=rule.component_id,
                value=None,
                output_action_schema=None,
                used_context_id=context.context_id,
            )
    elif mode == COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER:
        absolute = selected_inputs.get(rule.absolute_overlay_role)
        if absolute is None or not component_record_available(
            absolute,
            component_id=rule.component_id,
            representation=rule.representation,
        ):
            return _blocked(
                rule,
                reason="allocation_absolute_overlay_unavailable",
                context_id=context.context_id,
            )
        if (
            absolute.semantic_binding_id_by_kind.get(
                RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND
            )
            != anchor_binding.receipt_id
            or absolute.semantic_binding_hash_by_kind.get(
                RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND
            )
            != anchor_binding.receipt_hash
        ):
            return _blocked(
                rule,
                reason="allocation_absolute_anchor_binding_mismatch",
                context_id=context.context_id,
            )
        if not record_matches_declared_schema(strategy_input, absolute):
            return AllocatedComponentRuleEvaluation(
                output_status="invalid",
                reason="allocation_absolute_overlay_schema_mismatch",
                component_id=rule.component_id,
                value=None,
                output_action_schema=None,
                used_context_id=context.context_id,
            )
        baseline_payload_schema = output_schema.payload_schema_by_component[
            rule.component_id
        ]
        absolute_payload_schema = (
            absolute.action.action_schema.payload_schema_by_component[rule.component_id]
        )
        if (
            absolute_payload_schema.semantic_content_hash
            != baseline_payload_schema.semantic_content_hash
        ):
            return AllocatedComponentRuleEvaluation(
                output_status="invalid",
                reason="allocation_absolute_payload_schema_mismatch",
                component_id=rule.component_id,
                value=None,
                output_action_schema=None,
                used_context_id=context.context_id,
            )
        value = np.array(
            absolute.action.value(rule.component_id).values,
            copy=True,
        )
    else:  # pragma: no cover - context parser rejects this first.
        raise AssertionError(f"Unhandled allocation mode {mode!r}.")

    return AllocatedComponentRuleEvaluation(
        output_status="ready",
        reason=f"allocation_mode:{mode}",
        component_id=rule.component_id,
        value=value,
        output_action_schema=output_schema,
        used_context_id=context.context_id,
        selected_mode=mode,
    )


__all__ = [
    "AllocatedComponentCompositionRule",
    "AllocatedComponentRuleEvaluation",
    "evaluate_allocated_component_rule",
]
