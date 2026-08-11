"""Formal, receiver-bound strategy seam for control composition.

Version 2 is additive to the legacy deterministic composer. A formal strategy
receives one immutable invocation containing the exact action/context cut and
returns a small decision. The framework, not the strategy implementation,
builds the canonical ``CompositionEvaluation`` and its promotion identity.

Only bounded inline strategies belong behind this protocol. Slow model or
remote strategy computation must arrive through an admitted context/advisory
producer so it cannot block the control loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction
from robot_motion.runtime.immutable import deep_freeze

from control_composition.information import CompositionInformationPlan
from control_composition.residual_anchor_lifecycle import (
    ResidualAnchorLifecycleSnapshot,
)

from .context import CompositionContextSnapshot
from .core import (
    COMPOSITION_EVALUATION_HASH_DOMAIN_V7,
    CompositionInputSnapshot,
    CompositionEvaluation,
    CompositionSpec,
    DeterministicComposer,
)


FORMAL_COMPOSITION_STRATEGY_SPEC_HASH_DOMAIN_V2 = (
    "dexteleop.formal_composition_strategy_spec.v2"
)
FORMAL_COMPOSITION_STRATEGY_SPEC_HASH_DOMAIN_V3 = (
    "dexteleop.formal_composition_strategy_spec.v3"
)
FORMAL_COMPOSITION_STRATEGY_SPEC_HASH_DOMAIN_V4 = (
    "dexteleop.formal_composition_strategy_spec.v4"
)
FORMAL_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_V2 = (
    "dexteleop.formal_composition_strategy_input.v2"
)
FORMAL_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_V3 = (
    "dexteleop.formal_composition_strategy_input.v3"
)
FORMAL_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_V4 = (
    "dexteleop.formal_composition_strategy_input.v4"
)
FORMAL_COMPOSITION_STRATEGY_RESULT_BINDING_HASH_DOMAIN_V2 = (
    "dexteleop.formal_composition_strategy_result_binding.v2"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True)
class FormalCompositionStrategySpec:
    """Immutable identity and declared inputs for one inline strategy.

    This spec identifies *which implementation* evaluates a composition
    contract. It must not redefine component weights, fallback behavior, or
    fusion/disagreement bounds owned by ``CompositionSpec``. Those bounds are
    composition semantics, not canonical motion safety. Optional
    implementation contract hashes only prove that the implementation binds
    the exact typed contracts already referenced by the composition semantics.
    """

    strategy_id: str
    strategy_version: str
    composition_spec_hash: str
    required_context_channels: tuple[str, ...] = ()
    optional_context_channels: tuple[str, ...] = ()
    required_residual_anchor_lifecycle_spec_hash: str | None = None
    implementation_contract_hashes: tuple[str, ...] = ()
    parameter_payload: Mapping[str, Any] = field(default_factory=dict)
    maximum_decision_latency_sec: float | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        strategy_id = _required(self.strategy_id, name="strategy_id")
        strategy_version = _required(
            self.strategy_version,
            name="strategy_version",
        )
        composition_spec_hash = _required(
            self.composition_spec_hash,
            name="composition_spec_hash",
        )
        channels = tuple(
            _required(value, name="required_context_channel")
            for value in self.required_context_channels
        )
        if len(channels) != len(set(channels)):
            raise ValueError("required_context_channels must be unique.")
        optional_channels = tuple(
            _required(value, name="optional_context_channel")
            for value in self.optional_context_channels
        )
        if len(optional_channels) != len(set(optional_channels)):
            raise ValueError("optional_context_channels must be unique.")
        if set(channels) & set(optional_channels):
            raise ValueError("Required and optional context channels must be disjoint.")
        implementation_contract_hashes = tuple(
            _required(value, name="implementation_contract_hash")
            for value in self.implementation_contract_hashes
        )
        if len(implementation_contract_hashes) != len(
            set(implementation_contract_hashes)
        ):
            raise ValueError("implementation_contract_hashes must be unique.")
        parameters = deep_freeze(dict(self.parameter_payload))
        maximum_latency = self.maximum_decision_latency_sec
        if maximum_latency is not None:
            maximum_latency = float(maximum_latency)
            if not math.isfinite(maximum_latency) or maximum_latency <= 0.0:
                raise ValueError(
                    "maximum_decision_latency_sec must be finite and positive."
                )
        lifecycle_spec_hash = self.required_residual_anchor_lifecycle_spec_hash
        if lifecycle_spec_hash is not None:
            lifecycle_spec_hash = _required(
                lifecycle_spec_hash,
                name="required_residual_anchor_lifecycle_spec_hash",
            )
        payload = {
            "strategy_id": strategy_id,
            "strategy_version": strategy_version,
            "execution_class": "inline_bounded",
            "composition_spec_hash": composition_spec_hash,
            "required_context_channels": channels,
            "parameter_payload": parameters,
            "maximum_decision_latency_sec": maximum_latency,
        }
        hash_domain = FORMAL_COMPOSITION_STRATEGY_SPEC_HASH_DOMAIN_V2
        if lifecycle_spec_hash is not None:
            payload["required_residual_anchor_lifecycle_spec_hash"] = (
                lifecycle_spec_hash
            )
            hash_domain = FORMAL_COMPOSITION_STRATEGY_SPEC_HASH_DOMAIN_V3
        if optional_channels:
            payload["optional_context_channels"] = optional_channels
            hash_domain = FORMAL_COMPOSITION_STRATEGY_SPEC_HASH_DOMAIN_V4
        if implementation_contract_hashes:
            payload["implementation_contract_hashes"] = (
                implementation_contract_hashes
            )
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "strategy_version", strategy_version)
        object.__setattr__(
            self,
            "composition_spec_hash",
            composition_spec_hash,
        )
        object.__setattr__(self, "required_context_channels", channels)
        object.__setattr__(self, "optional_context_channels", optional_channels)
        object.__setattr__(
            self,
            "required_residual_anchor_lifecycle_spec_hash",
            lifecycle_spec_hash,
        )
        object.__setattr__(
            self,
            "implementation_contract_hashes",
            implementation_contract_hashes,
        )
        object.__setattr__(self, "parameter_payload", parameters)
        object.__setattr__(
            self,
            "maximum_decision_latency_sec",
            maximum_latency,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=hash_domain,
            ),
        )

    @property
    def context_channels(self) -> tuple[str, ...]:
        return self.required_context_channels + self.optional_context_channels


@dataclass(frozen=True)
class FormalCompositionStrategyInput:
    """Exact immutable action/context cut presented to one strategy."""

    runtime_session_id: str
    control_domain_id: str
    clock_domain_id: str
    strategy_spec: FormalCompositionStrategySpec
    composition_spec: CompositionSpec
    input_snapshot: CompositionInputSnapshot
    context_snapshot: CompositionContextSnapshot | None
    information_plan: CompositionInformationPlan | None
    residual_anchor_lifecycle_snapshot: (
        ResidualAnchorLifecycleSnapshot | None
    ) = None
    strategy_control_binding_id: str | None = None
    strategy_control_binding_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        runtime_session_id = _required(
            self.runtime_session_id,
            name="runtime_session_id",
        )
        control_domain_id = _required(
            self.control_domain_id,
            name="control_domain_id",
        )
        clock_domain_id = _required(
            self.clock_domain_id,
            name="clock_domain_id",
        )
        if not isinstance(self.strategy_spec, FormalCompositionStrategySpec):
            raise TypeError(
                "strategy_spec must be a FormalCompositionStrategySpec."
            )
        if not isinstance(self.composition_spec, CompositionSpec):
            raise TypeError("composition_spec must be a CompositionSpec.")
        if not isinstance(self.input_snapshot, CompositionInputSnapshot):
            raise TypeError(
                "input_snapshot must be a CompositionInputSnapshot."
            )
        composition_spec_hash = self.composition_spec.semantic_content_hash
        if (
            self.strategy_spec.composition_spec_hash
            != composition_spec_hash
            or self.input_snapshot.composition_spec_hash
            != composition_spec_hash
        ):
            raise ValueError(
                "Formal strategy, composition spec, and input snapshot mismatch."
            )
        if self.input_snapshot.control_domain_id != control_domain_id:
            raise ValueError("Formal strategy input belongs to another domain.")
        channels = self.strategy_spec.context_channels
        context_snapshot_id: str | None = None
        context_snapshot_hash: str | None = None
        if channels:
            if self.information_plan is None:
                raise ValueError(
                    "Formal strategy context requires a CompositionInformationPlan."
                )
            if (
                self.information_plan.context_channels != channels
                or self.information_plan.required_context_channels
                != self.strategy_spec.required_context_channels
            ):
                raise ValueError(
                    "Formal strategy channels do not match the information plan."
                )
            if self.context_snapshot is None:
                raise ValueError(
                    "Formal strategy context requires one context snapshot."
                )
        if self.context_snapshot is not None:
            if not isinstance(
                self.context_snapshot,
                CompositionContextSnapshot,
            ):
                raise TypeError(
                    "context_snapshot must be a CompositionContextSnapshot."
                )
            if (
                self.context_snapshot.control_domain_id != control_domain_id
                or self.context_snapshot.cutoff_at
                != self.input_snapshot.cutoff_at
                or self.context_snapshot.cutoff_admission_sequence
                != self.input_snapshot.cutoff_admission_sequence
                or self.context_snapshot.target_effective_at
                != self.input_snapshot.target_effective_at
                or self.context_snapshot.required_channels != channels
            ):
                raise ValueError(
                    "Formal strategy action/context snapshots must share one cut."
                )
            context_snapshot_id = self.context_snapshot.context_snapshot_id
            context_snapshot_hash = (
                self.context_snapshot.semantic_content_hash
            )
        information_plan_hash = (
            None
            if self.information_plan is None
            else self.information_plan.semantic_content_hash
        )
        lifecycle_snapshot = self.residual_anchor_lifecycle_snapshot
        required_lifecycle_hash = (
            self.strategy_spec.required_residual_anchor_lifecycle_spec_hash
        )
        lifecycle_snapshot_id = None
        lifecycle_snapshot_hash = None
        if required_lifecycle_hash is not None:
            if not isinstance(
                lifecycle_snapshot,
                ResidualAnchorLifecycleSnapshot,
            ):
                raise ValueError(
                    "Formal strategy requires residual anchor lifecycle input."
                )
            if (
                lifecycle_snapshot.lifecycle_spec_hash
                != required_lifecycle_hash
                or lifecycle_snapshot.control_domain_id != control_domain_id
                or lifecycle_snapshot.cutoff_at != self.input_snapshot.cutoff_at
                or lifecycle_snapshot.cutoff_admission_sequence
                != self.input_snapshot.cutoff_admission_sequence
                or lifecycle_snapshot.target_effective_at
                != self.input_snapshot.target_effective_at
            ):
                raise ValueError(
                    "Residual anchor lifecycle snapshot must share the formal cut."
                )
            lifecycle_snapshot_id = lifecycle_snapshot.lifecycle_snapshot_id
            lifecycle_snapshot_hash = lifecycle_snapshot.semantic_content_hash
        elif lifecycle_snapshot is not None:
            raise ValueError(
                "Formal strategy did not declare residual anchor lifecycle input."
            )
        payload = {
            "runtime_session_id": runtime_session_id,
            "control_domain_id": control_domain_id,
            "clock_domain_id": clock_domain_id,
            "strategy_id": self.strategy_spec.strategy_id,
            "strategy_spec_hash": self.strategy_spec.semantic_content_hash,
            "composition_spec_hash": composition_spec_hash,
            "composition_input_snapshot_id": (
                self.input_snapshot.composition_input_snapshot_id
            ),
            "composition_input_snapshot_hash": (
                self.input_snapshot.semantic_content_hash
            ),
            "context_snapshot_id": context_snapshot_id,
            "context_snapshot_hash": context_snapshot_hash,
            "information_plan_hash": information_plan_hash,
        }
        if lifecycle_snapshot_id is not None:
            payload["residual_anchor_lifecycle_snapshot_id"] = (
                lifecycle_snapshot_id
            )
            payload["residual_anchor_lifecycle_snapshot_hash"] = (
                lifecycle_snapshot_hash
            )
        binding_id = self.strategy_control_binding_id
        binding_hash = self.strategy_control_binding_hash
        if (binding_id is None) != (binding_hash is None):
            raise ValueError(
                "Strategy control binding ID/hash must appear together."
            )
        hash_domain = FORMAL_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_V2
        if binding_id is not None:
            binding_id = _required(
                binding_id,
                name="strategy_control_binding_id",
            )
            binding_hash = _required(
                binding_hash,
                name="strategy_control_binding_hash",
            )
            payload["strategy_control_binding_id"] = binding_id
            payload["strategy_control_binding_hash"] = binding_hash
            hash_domain = FORMAL_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_V3
        if lifecycle_snapshot_id is not None:
            hash_domain = FORMAL_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_V4
        object.__setattr__(
            self,
            "strategy_control_binding_id",
            binding_id,
        )
        object.__setattr__(
            self,
            "strategy_control_binding_hash",
            binding_hash,
        )
        object.__setattr__(
            self,
            "runtime_session_id",
            runtime_session_id,
        )
        object.__setattr__(self, "control_domain_id", control_domain_id)
        object.__setattr__(self, "clock_domain_id", clock_domain_id)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=hash_domain,
            ),
        )

    @property
    def strategy_input_id(self) -> str:
        return f"formal-composition-strategy-input:{self.semantic_content_hash}"


@dataclass(frozen=True)
class CompositionStrategyDecision:
    """Minimal semantic output returned by a formal strategy."""

    output_status: str
    reason: str
    composed_candidate: ComponentAction | None
    used_context_ids: tuple[str, ...] = ()
    result_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = _required(self.output_status, name="output_status")
        if status not in {"ready", "no_action", "invalid"}:
            raise ValueError("Unsupported composition strategy output status.")
        reason = _required(self.reason, name="reason")
        if (status == "ready") != (self.composed_candidate is not None):
            raise ValueError(
                "Only a ready strategy decision may contain an action."
            )
        action = self.composed_candidate
        if action is not None and not isinstance(action, ComponentAction):
            raise TypeError(
                "Formal composition strategies must return ComponentAction."
            )
        used_context_ids = tuple(
            _required(value, name="used_context_id")
            for value in self.used_context_ids
        )
        if len(used_context_ids) != len(set(used_context_ids)):
            raise ValueError("used_context_ids must be unique.")
        result_data = deep_freeze(dict(self.result_data))
        object.__setattr__(self, "output_status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "composed_candidate", action)
        object.__setattr__(self, "used_context_ids", used_context_ids)
        object.__setattr__(self, "result_data", result_data)


class FormalCompositionStrategy(Protocol):
    @property
    def spec(self) -> FormalCompositionStrategySpec: ...

    def decide(
        self,
        strategy_input: FormalCompositionStrategyInput,
    ) -> CompositionStrategyDecision: ...


@runtime_checkable
class FormalCompositionPublicationTransaction(Protocol):
    """Optional two-phase state commit tied to exact proposal admission."""

    def validate_publication_transaction(
        self,
        evaluation: CompositionEvaluation,
    ) -> None: ...

    def commit_publication_transaction(
        self,
        evaluation: CompositionEvaluation,
    ) -> None: ...


class DeterministicCompositionStrategyV2:
    """Migration adapter that reproduces deterministic-v1 numerical behavior."""

    def __init__(self, spec: FormalCompositionStrategySpec) -> None:
        if not isinstance(spec, FormalCompositionStrategySpec):
            raise TypeError(
                "spec must be a FormalCompositionStrategySpec."
            )
        if spec.context_channels:
            raise ValueError(
                "The deterministic migration strategy does not consume context."
            )
        self._spec = spec
        self._composer = DeterministicComposer()

    @property
    def spec(self) -> FormalCompositionStrategySpec:
        return self._spec

    def decide(
        self,
        strategy_input: FormalCompositionStrategyInput,
    ) -> CompositionStrategyDecision:
        if strategy_input.strategy_spec != self.spec:
            raise ValueError("Formal strategy input uses another strategy spec.")
        result = self._composer.compose(
            strategy_input.input_snapshot,
            spec=strategy_input.composition_spec,
        )
        return CompositionStrategyDecision(
            output_status=result.output_status,
            reason=result.reason,
            composed_candidate=result.composed_candidate,
        )


def build_formal_strategy_composition_evaluation(
    strategy_input: FormalCompositionStrategyInput,
    decision: CompositionStrategyDecision,
    *,
    composition_supported_until: float | None = None,
) -> CompositionEvaluation:
    """Build the canonical v4 result for one exact formal strategy input."""

    if not isinstance(strategy_input, FormalCompositionStrategyInput):
        raise TypeError(
            "strategy_input must be a FormalCompositionStrategyInput."
        )
    if not isinstance(decision, CompositionStrategyDecision):
        raise TypeError("decision must be a CompositionStrategyDecision.")
    context_snapshot = strategy_input.context_snapshot
    available_context_ids = (
        set()
        if context_snapshot is None
        else set(context_snapshot.selected_context_id_by_channel.values())
    )
    if not set(decision.used_context_ids).issubset(available_context_ids):
        raise ValueError(
            "Strategy decision references context outside its frozen snapshot."
        )
    snapshot = strategy_input.input_snapshot
    spec = strategy_input.composition_spec
    missing_required = tuple(
        role
        for role in spec.required_roles
        if role not in snapshot.selected_input_by_role
    )
    if decision.output_status == "ready" and missing_required:
        raise ValueError(
            "A formal strategy cannot publish ready while required roles are "
            f"missing: {missing_required!r}."
        )
    ordered_inputs = tuple(
        snapshot.selected_input_by_role[role]
        for role in spec.input_roles
        if role in snapshot.selected_input_by_role
    )
    input_ids = tuple(item.composition_input_id for item in ordered_inputs)
    input_hashes = tuple(item.semantic_content_hash for item in ordered_inputs)
    context_ids = tuple(
        dict.fromkeys(
            context_id
            for item in ordered_inputs
            for context_id in item.context_ids
        )
    )
    action_hash = (
        None
        if decision.composed_candidate is None
        else decision.composed_candidate.semantic_content_hash
    )
    supported_until = None
    if decision.output_status == "ready":
        support_limits = [
            item.effective_until for item in ordered_inputs
        ]
        if context_snapshot is not None:
            support_limits.extend(
                record.effective_until
                for record in (
                    context_snapshot.selected_context_by_channel.values()
                )
            )
            if strategy_input.information_plan is not None:
                for requirement in (
                    strategy_input.information_plan.context_requirements
                ):
                    if requirement.maximum_age_sec is None:
                        continue
                    record = (
                        context_snapshot.selected_context_by_channel.get(
                            requirement.channel
                        )
                    )
                    if record is not None:
                        support_limits.append(
                            record.produced_at
                            + requirement.maximum_age_sec
                        )
        if composition_supported_until is not None:
            support_limits.append(float(composition_supported_until))
        if support_limits:
            supported_until = min(support_limits)
    context_snapshot_id = (
        None
        if context_snapshot is None
        else context_snapshot.context_snapshot_id
    )
    context_snapshot_hash = (
        None
        if context_snapshot is None
        else context_snapshot.semantic_content_hash
    )
    information_plan_hash = (
        None
        if strategy_input.information_plan is None
        else strategy_input.information_plan.semantic_content_hash
    )
    payload = {
        "composition_input_snapshot_id": (
            snapshot.composition_input_snapshot_id
        ),
        "composition_input_snapshot_hash": snapshot.semantic_content_hash,
        "composition_spec_hash": spec.semantic_content_hash,
        "action_schema_hash": snapshot.action_schema_hash,
        "target_effective_at": snapshot.target_effective_at,
        "target_clock_domain_id": snapshot.target_clock_domain_id,
        "output_status": decision.output_status,
        "reason": decision.reason,
        "composed_candidate_hash": action_hash,
        "input_ids": input_ids,
        "input_hashes": input_hashes,
        "context_ids": context_ids,
        "strategy_id": strategy_input.strategy_spec.strategy_id,
        "strategy_spec_hash": (
            strategy_input.strategy_spec.semantic_content_hash
        ),
        "strategy_input_id": strategy_input.strategy_input_id,
        "strategy_input_hash": strategy_input.semantic_content_hash,
        "strategy_context_snapshot_id": context_snapshot_id,
        "strategy_context_snapshot_hash": context_snapshot_hash,
        "strategy_information_plan_hash": information_plan_hash,
        "strategy_used_context_ids": decision.used_context_ids,
        "composition_supported_until": supported_until,
    }
    if decision.result_data:
        payload["strategy_result_data"] = decision.result_data
    evaluation_hash_domain = COMPOSITION_EVALUATION_HASH_DOMAIN_V7
    if strategy_input.strategy_control_binding_id is not None:
        payload["strategy_control_binding_id"] = (
            strategy_input.strategy_control_binding_id
        )
        payload["strategy_control_binding_hash"] = (
            strategy_input.strategy_control_binding_hash
        )
    if snapshot.expected_activation_id is not None:
        payload["composition_activation_id"] = (
            snapshot.expected_activation_id
        )
        payload["composition_activation_hash"] = (
            snapshot.expected_activation_hash
        )
    digest = canonical_content_hash(
        payload,
        domain=evaluation_hash_domain,
    )
    return CompositionEvaluation(
        composition_evaluation_id=f"composition-evaluation:{digest}",
        composed_candidate=decision.composed_candidate,
        semantic_content_hash=digest,
        **payload,
    )


@dataclass(frozen=True)
class FormalCompositionStrategyResultBinding:
    """Traceable relation between one strategy input and its canonical result."""

    strategy_input_id: str
    strategy_input_hash: str
    composition_evaluation_id: str
    composition_evaluation_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "strategy_input_id": _required(
                self.strategy_input_id,
                name="strategy_input_id",
            ),
            "strategy_input_hash": _required(
                self.strategy_input_hash,
                name="strategy_input_hash",
            ),
            "composition_evaluation_id": _required(
                self.composition_evaluation_id,
                name="composition_evaluation_id",
            ),
            "composition_evaluation_hash": _required(
                self.composition_evaluation_hash,
                name="composition_evaluation_hash",
            ),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    FORMAL_COMPOSITION_STRATEGY_RESULT_BINDING_HASH_DOMAIN_V2
                ),
            ),
        )

    @property
    def binding_id(self) -> str:
        return (
            "formal-composition-strategy-result-binding:"
            f"{self.semantic_content_hash}"
        )


def build_formal_strategy_result_binding(
    strategy_input: FormalCompositionStrategyInput,
    result: CompositionEvaluation,
) -> FormalCompositionStrategyResultBinding:
    if result.strategy_input_id != strategy_input.strategy_input_id or (
        result.strategy_input_hash != strategy_input.semantic_content_hash
    ):
        raise ValueError("Composition result does not match the strategy input.")
    return FormalCompositionStrategyResultBinding(
        strategy_input_id=strategy_input.strategy_input_id,
        strategy_input_hash=strategy_input.semantic_content_hash,
        composition_evaluation_id=result.composition_evaluation_id,
        composition_evaluation_hash=result.semantic_content_hash,
    )


__all__ = [
    "FORMAL_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_V2",
    "FORMAL_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_V3",
    "FORMAL_COMPOSITION_STRATEGY_RESULT_BINDING_HASH_DOMAIN_V2",
    "FORMAL_COMPOSITION_STRATEGY_SPEC_HASH_DOMAIN_V2",
    "CompositionStrategyDecision",
    "DeterministicCompositionStrategyV2",
    "FormalCompositionStrategy",
    "FormalCompositionPublicationTransaction",
    "FormalCompositionStrategyInput",
    "FormalCompositionStrategyResultBinding",
    "FormalCompositionStrategySpec",
    "build_formal_strategy_composition_evaluation",
    "build_formal_strategy_result_binding",
]
