"""Receiver-linearized hot control for one long-lived composition strategy.

The control path intentionally has only two mutable rates:

* an immutable, low-frequency strategy revision; and
* a latest-covering, high-frequency component alpha parameter record.

Topology, action representation, source roles, context contracts, strategy
implementation, authority, and execution remain outside this hot-control
surface.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
    admitted_at_or_before,
)
from robot_motion.runtime.producer_epoch import (
    ProducerEpochError,
    ProducerEpochRegistry,
    ProducerEpochSnapshot,
)

from control_composition.retention import TimeWindowRetentionSpec

from .component_contract import (
    ComponentCompositionRuleSet,
)
from .core import CompositionSpec
from .formal_strategy import FormalCompositionStrategySpec


COMPOSITION_DYNAMIC_PARAMETER_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.composition_dynamic_parameter_contract.v1"
)
COMPOSITION_DYNAMIC_PARAMETER_RECORD_HASH_DOMAIN_V1 = (
    "dexteleop.composition_dynamic_parameter_record.v1"
)
COMPOSITION_DYNAMIC_PARAMETER_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.composition_dynamic_parameter_snapshot.v1"
)
COMPOSITION_STRATEGY_REVISION_HASH_DOMAIN_V1 = (
    "dexteleop.composition_strategy_revision.v1"
)
COMPOSITION_STRATEGY_CONTROL_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.composition_strategy_control_binding.v1"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _alpha_map(
    values: Mapping[str, float],
    *,
    component_ids: tuple[str, ...],
    name: str,
) -> Mapping[str, float]:
    normalized = {
        _required(component_id, name=f"{name}.component_id"): _finite(
            value,
            name=f"{name}[{component_id!r}]",
        )
        for component_id, value in dict(values).items()
    }
    if set(normalized) != set(component_ids):
        raise ValueError(f"{name} must contain exactly the declared components.")
    if any(value < 0.0 or value > 1.0 for value in normalized.values()):
        raise ValueError(f"{name} values must be in [0, 1].")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class CompositionDynamicParameterContract:
    """Minimal high-rate parameter contract for policy/human-style blending."""

    parameter_set_id: str
    base_role: str
    overlay_role: str
    component_ids: tuple[str, ...]
    default_alpha_by_component: Mapping[str, float]
    lifecycle: str = "latched"
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        parameter_set_id = _required(
            self.parameter_set_id,
            name="parameter_set_id",
        )
        base_role = _required(self.base_role, name="base_role")
        overlay_role = _required(self.overlay_role, name="overlay_role")
        if base_role == overlay_role:
            raise ValueError("Dynamic alpha requires two different roles.")
        component_ids = tuple(
            _required(value, name="component_id")
            for value in self.component_ids
        )
        if not component_ids or len(component_ids) != len(set(component_ids)):
            raise ValueError("component_ids must be non-empty and unique.")
        lifecycle = _required(self.lifecycle, name="lifecycle")
        if lifecycle not in {"latched", "ttl"}:
            raise ValueError("Dynamic parameter lifecycle must be latched or ttl.")
        defaults = _alpha_map(
            self.default_alpha_by_component,
            component_ids=component_ids,
            name="default_alpha_by_component",
        )
        payload = {
            "parameter_set_id": parameter_set_id,
            "parameter_semantics": "component_base_overlay_alpha_v1",
            "base_role": base_role,
            "overlay_role": overlay_role,
            "component_ids": component_ids,
            "default_alpha_by_component": defaults,
            "lifecycle": lifecycle,
        }
        object.__setattr__(self, "parameter_set_id", parameter_set_id)
        object.__setattr__(self, "base_role", base_role)
        object.__setattr__(self, "overlay_role", overlay_role)
        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "default_alpha_by_component", defaults)
        object.__setattr__(self, "lifecycle", lifecycle)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    COMPOSITION_DYNAMIC_PARAMETER_CONTRACT_HASH_DOMAIN_V1
                ),
            ),
        )


@dataclass(frozen=True)
class CompositionStrategyRevisionDraft:
    request_id: str
    expected_previous_revision_hash: str
    strategy_mode: str
    composition_spec: CompositionSpec
    strategy_spec: FormalCompositionStrategySpec
    component_rule_set: ComponentCompositionRuleSet
    dynamic_parameter_contract: CompositionDynamicParameterContract | None
    initial_alpha_by_component: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CompositionStrategyRevision:
    runtime_session_id: str
    control_domain_id: str
    strategy_instance_id: str
    revision_sequence: int
    previous_revision_hash: str | None
    request_id: str
    strategy_mode: str
    composition_spec: CompositionSpec
    strategy_spec: FormalCompositionStrategySpec
    component_rule_set: ComponentCompositionRuleSet
    dynamic_parameter_contract: CompositionDynamicParameterContract | None
    initial_alpha_by_component: Mapping[str, float]
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
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
        instance_id = _required(
            self.strategy_instance_id,
            name="strategy_instance_id",
        )
        sequence = int(self.revision_sequence)
        if sequence < 0:
            raise ValueError("revision_sequence must be non-negative.")
        previous = self.previous_revision_hash
        if sequence == 0:
            if previous is not None:
                raise ValueError("Initial revision cannot have a predecessor.")
        else:
            previous = _required(previous, name="previous_revision_hash")
        request_id = _required(self.request_id, name="request_id")
        mode = _required(self.strategy_mode, name="strategy_mode")
        if not isinstance(self.composition_spec, CompositionSpec):
            raise TypeError("composition_spec must be a CompositionSpec.")
        if not isinstance(
            self.strategy_spec,
            FormalCompositionStrategySpec,
        ):
            raise TypeError(
                "strategy_spec must be a FormalCompositionStrategySpec."
            )
        if not isinstance(
            self.component_rule_set,
            ComponentCompositionRuleSet,
        ):
            raise TypeError(
                "component_rule_set must be a ComponentCompositionRuleSet."
            )
        if (
            self.composition_spec.component_rule_set_hash
            != self.component_rule_set.semantic_content_hash
            or self.strategy_spec.composition_spec_hash
            != self.composition_spec.semantic_content_hash
            or self.strategy_spec.implementation_contract_hashes
            != (self.component_rule_set.semantic_content_hash,)
        ):
            raise ValueError(
                "Revision composition, strategy, and component rules mismatch."
            )
        contract = self.dynamic_parameter_contract
        if contract is not None and not isinstance(
            contract,
            CompositionDynamicParameterContract,
        ):
            raise TypeError(
                "dynamic_parameter_contract must be a "
                "CompositionDynamicParameterContract or None."
            )
        if contract is None:
            if self.initial_alpha_by_component:
                raise ValueError(
                    "Initial alpha values require a dynamic parameter contract."
                )
            initial_alpha: Mapping[str, float] = MappingProxyType({})
        else:
            if not {
                contract.base_role,
                contract.overlay_role,
            }.issubset(self.composition_spec.input_roles):
                raise ValueError(
                    "Dynamic parameter roles must belong to CompositionSpec."
                )
            initial_alpha = _alpha_map(
                self.initial_alpha_by_component,
                component_ids=contract.component_ids,
                name="initial_alpha_by_component",
            )
            rule_by_component = {
                rule.component_id: rule
                for rule in self.component_rule_set.rules
            }
            for component_id in contract.component_ids:
                if component_id not in rule_by_component:
                    raise ValueError(
                        "Dynamic alpha references an unknown component."
                    )
                if set(
                    rule_by_component[component_id].contributing_roles
                ) != {
                    contract.base_role,
                    contract.overlay_role,
                }:
                    raise ValueError(
                        "Dynamic alpha currently requires exactly its base and "
                        "overlay roles on every controlled component."
                    )
        available_at = _finite(self.available_at, name="available_at")
        clock_domain_id = _required(
            self.available_clock_domain_id,
            name="available_clock_domain_id",
        )
        admission_sequence = int(self.control_admission_sequence)
        if admission_sequence < 0:
            raise ValueError(
                "control_admission_sequence must be non-negative."
            )
        payload = {
            "runtime_session_id": runtime_session_id,
            "control_domain_id": control_domain_id,
            "strategy_instance_id": instance_id,
            "revision_sequence": sequence,
            "previous_revision_hash": previous,
            "request_id": request_id,
            "strategy_mode": mode,
            "composition_spec_hash": (
                self.composition_spec.semantic_content_hash
            ),
            "strategy_spec_hash": self.strategy_spec.semantic_content_hash,
            "component_rule_set_hash": (
                self.component_rule_set.semantic_content_hash
            ),
            "dynamic_parameter_contract_hash": (
                None if contract is None else contract.semantic_content_hash
            ),
            "initial_alpha_by_component": initial_alpha,
            "available_at": available_at,
            "available_clock_domain_id": clock_domain_id,
            "control_admission_sequence": admission_sequence,
        }
        object.__setattr__(
            self,
            "runtime_session_id",
            runtime_session_id,
        )
        object.__setattr__(self, "control_domain_id", control_domain_id)
        object.__setattr__(self, "strategy_instance_id", instance_id)
        object.__setattr__(self, "revision_sequence", sequence)
        object.__setattr__(self, "previous_revision_hash", previous)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "strategy_mode", mode)
        object.__setattr__(
            self,
            "initial_alpha_by_component",
            initial_alpha,
        )
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(
            self,
            "available_clock_domain_id",
            clock_domain_id,
        )
        object.__setattr__(
            self,
            "control_admission_sequence",
            admission_sequence,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPOSITION_STRATEGY_REVISION_HASH_DOMAIN_V1,
            ),
        )

    @property
    def revision_id(self) -> str:
        return f"composition-strategy-revision:{self.semantic_content_hash}"


@dataclass(frozen=True)
class CompositionStrategyRevisionAdmissionReceipt:
    accepted: bool
    rejection_code: str | None = None
    message: str = ""
    revision: CompositionStrategyRevision | None = None


@dataclass(frozen=True)
class CompositionDynamicParameterDraft:
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_generation: int
    sequence: int
    strategy_revision_hash: str
    parameter_contract_hash: str
    alpha_by_component: Mapping[str, float]
    produced_at: float
    produced_clock_domain_id: str
    effective_from: float
    effective_until: float | None
    effective_clock_domain_id: str


@dataclass(frozen=True)
class CompositionDynamicParameterRecord:
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_generation: int
    sequence: int
    strategy_revision_hash: str
    parameter_contract_hash: str
    alpha_by_component: Mapping[str, float]
    produced_at: float
    produced_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    effective_from: float
    effective_until: float | None
    effective_clock_domain_id: str
    control_admission_sequence: int
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
        producer_id = _required(self.producer_id, name="producer_id")
        generation = int(self.producer_generation)
        sequence = int(self.sequence)
        admission_sequence = int(self.control_admission_sequence)
        if generation < 0 or sequence < 0 or admission_sequence < 0:
            raise ValueError(
                "Producer generation and sequences must be non-negative."
            )
        revision_hash = _required(
            self.strategy_revision_hash,
            name="strategy_revision_hash",
        )
        contract_hash = _required(
            self.parameter_contract_hash,
            name="parameter_contract_hash",
        )
        alpha = MappingProxyType(
            {
                _required(key, name="alpha component"): _finite(
                    value,
                    name=f"alpha_by_component[{key!r}]",
                )
                for key, value in dict(self.alpha_by_component).items()
            }
        )
        if any(value < 0.0 or value > 1.0 for value in alpha.values()):
            raise ValueError("Dynamic alpha values must be in [0, 1].")
        produced_at = _finite(self.produced_at, name="produced_at")
        available_at = _finite(self.available_at, name="available_at")
        effective_from = _finite(
            self.effective_from,
            name="effective_from",
        )
        effective_until = self.effective_until
        if effective_until is not None:
            effective_until = _finite(
                effective_until,
                name="effective_until",
            )
            if effective_until <= effective_from:
                raise ValueError(
                    "effective_until must follow effective_from."
                )
        payload = {
            "runtime_session_id": runtime_session_id,
            "control_domain_id": control_domain_id,
            "producer_id": producer_id,
            "producer_generation": generation,
            "sequence": sequence,
            "strategy_revision_hash": revision_hash,
            "parameter_contract_hash": contract_hash,
            "alpha_by_component": alpha,
            "produced_at": produced_at,
            "produced_clock_domain_id": _required(
                self.produced_clock_domain_id,
                name="produced_clock_domain_id",
            ),
            "available_at": available_at,
            "available_clock_domain_id": _required(
                self.available_clock_domain_id,
                name="available_clock_domain_id",
            ),
            "effective_from": effective_from,
            "effective_until": effective_until,
            "effective_clock_domain_id": _required(
                self.effective_clock_domain_id,
                name="effective_clock_domain_id",
            ),
            "control_admission_sequence": admission_sequence,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPOSITION_DYNAMIC_PARAMETER_RECORD_HASH_DOMAIN_V1,
            ),
        )

    @property
    def parameter_record_id(self) -> str:
        return f"composition-dynamic-parameter:{self.semantic_content_hash}"


@dataclass(frozen=True)
class CompositionDynamicParameterAdmissionReceipt:
    accepted: bool
    rejection_code: str | None = None
    message: str = ""
    parameter_record: CompositionDynamicParameterRecord | None = None


@dataclass(frozen=True)
class CompositionDynamicParameterSnapshot:
    strategy_revision_hash: str
    parameter_contract_hash: str | None
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    selected_parameter_record_id: str | None
    selected_parameter_record_hash: str | None
    alpha_by_component: Mapping[str, float]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        revision_hash = _required(
            self.strategy_revision_hash,
            name="strategy_revision_hash",
        )
        contract_hash = self.parameter_contract_hash
        if contract_hash is not None:
            contract_hash = _required(
                contract_hash,
                name="parameter_contract_hash",
            )
        record_id = self.selected_parameter_record_id
        record_hash = self.selected_parameter_record_hash
        if (record_id is None) != (record_hash is None):
            raise ValueError("Selected parameter record ID/hash must pair.")
        alpha = MappingProxyType(
            {
                _required(key, name="alpha component"): _finite(
                    value,
                    name=f"alpha_by_component[{key!r}]",
                )
                for key, value in dict(self.alpha_by_component).items()
            }
        )
        if any(value < 0.0 or value > 1.0 for value in alpha.values()):
            raise ValueError("Dynamic alpha values must be in [0, 1].")
        cutoff_sequence = int(self.cutoff_admission_sequence)
        if cutoff_sequence < 0:
            raise ValueError(
                "cutoff_admission_sequence must be non-negative."
            )
        payload = {
            "strategy_revision_hash": revision_hash,
            "parameter_contract_hash": contract_hash,
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "cutoff_at": _finite(self.cutoff_at, name="cutoff_at"),
            "cutoff_admission_sequence": cutoff_sequence,
            "target_effective_at": _finite(
                self.target_effective_at,
                name="target_effective_at",
            ),
            "selected_parameter_record_id": record_id,
            "selected_parameter_record_hash": record_hash,
            "alpha_by_component": alpha,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    COMPOSITION_DYNAMIC_PARAMETER_SNAPSHOT_HASH_DOMAIN_V1
                ),
            ),
        )

    @property
    def parameter_snapshot_id(self) -> str:
        return f"composition-dynamic-parameter-snapshot:{self.semantic_content_hash}"


@dataclass(frozen=True)
class CompositionStrategyControlBinding:
    revision: CompositionStrategyRevision
    parameter_snapshot: CompositionDynamicParameterSnapshot
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.parameter_snapshot.strategy_revision_hash
            != self.revision.semantic_content_hash
        ):
            raise ValueError(
                "Strategy revision and parameter snapshot do not match."
            )
        payload = {
            "strategy_revision_id": self.revision.revision_id,
            "strategy_revision_hash": self.revision.semantic_content_hash,
            "parameter_snapshot_id": (
                self.parameter_snapshot.parameter_snapshot_id
            ),
            "parameter_snapshot_hash": (
                self.parameter_snapshot.semantic_content_hash
            ),
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    COMPOSITION_STRATEGY_CONTROL_BINDING_HASH_DOMAIN_V1
                ),
            ),
        )

    @property
    def binding_id(self) -> str:
        return f"composition-strategy-control-binding:{self.semantic_content_hash}"


def _rule_signature(
    rule_set: ComponentCompositionRuleSet,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    return tuple(
        (
            rule.component_id,
            rule.representation,
            tuple(rule.contributing_roles),
        )
        for rule in rule_set.rules
    )


class CompositionStrategyControl:
    """One receiver-owned current/pending revision and parameter stream."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        control_domain_id: str,
        strategy_instance_id: str,
        coordinator: ControlAdmissionCoordinator,
        producer_epochs: ProducerEpochRegistry,
        retention_spec: TimeWindowRetentionSpec,
        supported_strategy_modes: tuple[str, ...],
        initial_revision: CompositionStrategyRevision,
    ) -> None:
        if initial_revision.revision_sequence != 0:
            raise ValueError("initial_revision must use revision_sequence zero.")
        if initial_revision.strategy_instance_id != strategy_instance_id:
            raise ValueError("Initial revision belongs to another strategy instance.")
        if (
            initial_revision.runtime_session_id != runtime_session_id
            or initial_revision.control_domain_id != control_domain_id
        ):
            raise ValueError("Initial revision belongs to another runtime.")
        if coordinator.control_domain_id != control_domain_id:
            raise ValueError("Strategy control and coordinator domains mismatch.")
        if coordinator.clock_domain.runtime_session_id != runtime_session_id:
            raise ValueError("Strategy control and coordinator sessions mismatch.")
        if producer_epochs.coordinator is not coordinator:
            raise ValueError("Strategy control must share producer epochs.")
        if not isinstance(retention_spec, TimeWindowRetentionSpec):
            raise TypeError(
                "retention_spec must be a TimeWindowRetentionSpec."
            )
        modes = tuple(
            _required(value, name="supported_strategy_mode")
            for value in supported_strategy_modes
        )
        if not modes or len(modes) != len(set(modes)):
            raise ValueError(
                "supported_strategy_modes must be non-empty and unique."
            )
        if initial_revision.strategy_mode not in modes:
            raise ValueError(
                "Initial revision uses an unsupported strategy mode."
            )
        self.runtime_session_id = _required(
            runtime_session_id,
            name="runtime_session_id",
        )
        self.control_domain_id = _required(
            control_domain_id,
            name="control_domain_id",
        )
        self.strategy_instance_id = _required(
            strategy_instance_id,
            name="strategy_instance_id",
        )
        self.coordinator = coordinator
        self.producer_epochs = producer_epochs
        self.retention_spec = retention_spec
        self.supported_strategy_modes = modes
        self._initial_revision = initial_revision
        self._revisions = [initial_revision]
        self._revision_by_hash = {
            initial_revision.semantic_content_hash: initial_revision
        }
        self._parameter_records: list[CompositionDynamicParameterRecord] = []
        self._last_parameter_sequence: dict[tuple[str, int], int] = {}
        self._applied_revision_sequence = initial_revision.revision_sequence
        self._closed_target_through: float | None = None
        self._lock = threading.RLock()

    @property
    def latest_revision(self) -> CompositionStrategyRevision:
        with self._lock:
            return self._revisions[-1]

    def revision_by_hash(
        self,
        revision_hash: str,
    ) -> CompositionStrategyRevision | None:
        with self._lock:
            return self._revision_by_hash.get(str(revision_hash))

    def _validate_revision_compatibility(
        self,
        draft: CompositionStrategyRevisionDraft,
    ) -> None:
        initial = self._initial_revision
        if (
            draft.strategy_spec.strategy_id
            != initial.strategy_spec.strategy_id
            or draft.strategy_spec.strategy_version
            != initial.strategy_spec.strategy_version
            or draft.strategy_spec.required_context_channels
            != initial.strategy_spec.required_context_channels
            or draft.strategy_spec.optional_context_channels
            != initial.strategy_spec.optional_context_channels
            or draft.composition_spec.algebra
            != initial.composition_spec.algebra
            or draft.composition_spec.input_roles
            != initial.composition_spec.input_roles
            or draft.composition_spec.required_roles
            != initial.composition_spec.required_roles
            or draft.composition_spec.depends_on_roles_by_role
            != initial.composition_spec.depends_on_roles_by_role
            or _rule_signature(draft.component_rule_set)
            != _rule_signature(initial.component_rule_set)
        ):
            raise ValueError(
                "Strategy revision crosses the fixed runtime compatibility boundary."
            )
        if draft.strategy_mode not in self.supported_strategy_modes:
            raise ValueError("unsupported_strategy_mode")
        if draft.dynamic_parameter_contract is not None:
            if set(draft.dynamic_parameter_contract.component_ids) != {
                rule.component_id for rule in draft.component_rule_set.rules
            }:
                raise ValueError(
                    "Dynamic parameter components must match the fixed rule topology."
                )

    def admit_revision(
        self,
        draft: CompositionStrategyRevisionDraft,
    ) -> CompositionStrategyRevisionAdmissionReceipt:
        if not isinstance(draft, CompositionStrategyRevisionDraft):
            raise TypeError("draft must be a CompositionStrategyRevisionDraft.")
        try:
            with self._lock:
                latest = self._revisions[-1]
                if (
                    latest.revision_sequence
                    > self._applied_revision_sequence
                ):
                    raise ValueError("pending_revision_exists")
                if (
                    draft.expected_previous_revision_hash
                    != latest.semantic_content_hash
                ):
                    raise ValueError("stale_previous_revision")
                self._validate_revision_compatibility(draft)

            def build(
                available_at: float,
                admission_sequence: int,
            ) -> CompositionStrategyRevision:
                with self._lock:
                    current = self._revisions[-1]
                    if (
                        draft.expected_previous_revision_hash
                        != current.semantic_content_hash
                    ):
                        raise ValueError("stale_previous_revision")
                    return CompositionStrategyRevision(
                        runtime_session_id=self.runtime_session_id,
                        control_domain_id=self.control_domain_id,
                        strategy_instance_id=self.strategy_instance_id,
                        revision_sequence=current.revision_sequence + 1,
                        previous_revision_hash=current.semantic_content_hash,
                        request_id=draft.request_id,
                        strategy_mode=draft.strategy_mode,
                        composition_spec=draft.composition_spec,
                        strategy_spec=draft.strategy_spec,
                        component_rule_set=draft.component_rule_set,
                        dynamic_parameter_contract=(
                            draft.dynamic_parameter_contract
                        ),
                        initial_alpha_by_component=(
                            draft.initial_alpha_by_component
                        ),
                        available_at=available_at,
                        available_clock_domain_id=(
                            self.coordinator.clock_domain.clock_domain_id
                        ),
                        control_admission_sequence=admission_sequence,
                    )

            revision = self.coordinator.commit(store=self, build=build)
            return CompositionStrategyRevisionAdmissionReceipt(
                accepted=True,
                revision=revision,
            )
        except (TypeError, ValueError) as exc:
            return CompositionStrategyRevisionAdmissionReceipt(
                accepted=False,
                rejection_code=str(exc),
                message=str(exc),
            )

    def admit_parameter(
        self,
        draft: CompositionDynamicParameterDraft,
    ) -> CompositionDynamicParameterAdmissionReceipt:
        if not isinstance(draft, CompositionDynamicParameterDraft):
            raise TypeError("draft must be a CompositionDynamicParameterDraft.")
        try:
            with self._lock:
                revision = self._revision_by_hash.get(
                    draft.strategy_revision_hash
                )
                if revision is None:
                    raise ValueError("unknown_strategy_revision")
                contract = revision.dynamic_parameter_contract
                if (
                    contract is None
                    or draft.parameter_contract_hash
                    != contract.semantic_content_hash
                ):
                    raise ValueError("parameter_contract_mismatch")
                alpha = _alpha_map(
                    draft.alpha_by_component,
                    component_ids=contract.component_ids,
                    name="alpha_by_component",
                )
                if (
                    contract.lifecycle == "latched"
                    and draft.effective_until is not None
                ):
                    raise ValueError(
                        "latched_parameter_must_not_expire"
                    )
                if (
                    contract.lifecycle == "ttl"
                    and draft.effective_until is None
                ):
                    raise ValueError("ttl_parameter_requires_expiry")
            if (
                draft.runtime_session_id != self.runtime_session_id
                or draft.control_domain_id != self.control_domain_id
            ):
                raise ValueError("parameter_runtime_mismatch")
            self.producer_epochs.assert_active(
                producer_id=draft.producer_id,
                producer_epoch=int(draft.producer_generation),
            )

            def build(
                available_at: float,
                admission_sequence: int,
            ) -> CompositionDynamicParameterRecord:
                return CompositionDynamicParameterRecord(
                    runtime_session_id=draft.runtime_session_id,
                    control_domain_id=draft.control_domain_id,
                    producer_id=draft.producer_id,
                    producer_generation=draft.producer_generation,
                    sequence=draft.sequence,
                    strategy_revision_hash=draft.strategy_revision_hash,
                    parameter_contract_hash=draft.parameter_contract_hash,
                    alpha_by_component=alpha,
                    produced_at=draft.produced_at,
                    produced_clock_domain_id=(
                        draft.produced_clock_domain_id
                    ),
                    available_at=available_at,
                    available_clock_domain_id=(
                        self.coordinator.clock_domain.clock_domain_id
                    ),
                    effective_from=draft.effective_from,
                    effective_until=draft.effective_until,
                    effective_clock_domain_id=(
                        draft.effective_clock_domain_id
                    ),
                    control_admission_sequence=admission_sequence,
                )

            record = self.coordinator.commit(store=self, build=build)
            return CompositionDynamicParameterAdmissionReceipt(
                accepted=True,
                parameter_record=record,
            )
        except (ProducerEpochError, TypeError, ValueError) as exc:
            return CompositionDynamicParameterAdmissionReceipt(
                accepted=False,
                rejection_code=str(exc),
                message=str(exc),
            )

    def _insert_admitted(
        self,
        record: CompositionStrategyRevision | CompositionDynamicParameterRecord,
    ) -> None:
        with self._lock:
            if isinstance(record, CompositionStrategyRevision):
                if (
                    record.previous_revision_hash
                    != self._revisions[-1].semantic_content_hash
                ):
                    raise ValueError("stale_previous_revision")
                self._revisions.append(record)
                self._revision_by_hash[record.semantic_content_hash] = record
                return
            if not isinstance(record, CompositionDynamicParameterRecord):
                raise TypeError("Unsupported strategy-control record.")
            key = (record.producer_id, record.producer_generation)
            previous = self._last_parameter_sequence.get(key)
            if previous is not None and record.sequence <= previous:
                raise ValueError("parameter_sequence_not_increasing")
            if record.parameter_record_id in {
                item.parameter_record_id for item in self._parameter_records
            }:
                raise ValueError("duplicate_parameter_record")
            self._parameter_records.append(record)
            self._last_parameter_sequence[key] = record.sequence

    def capture_binding(
        self,
        cut: ControlCut,
        *,
        producer_epochs: ProducerEpochSnapshot,
        target_effective_at: float,
    ) -> CompositionStrategyControlBinding:
        target = _finite(
            target_effective_at,
            name="target_effective_at",
        )
        with self._lock:
            if (
                self._closed_target_through is not None
                and target < self._closed_target_through
            ):
                raise ValueError(
                    "Strategy-control target cannot move behind the closed window."
                )
            eligible_revisions = tuple(
                revision
                for revision in self._revisions
                if admitted_at_or_before(revision, cut)
            )
            if not eligible_revisions:
                revision = self._initial_revision
            else:
                revision = max(
                    eligible_revisions,
                    key=lambda item: (
                        item.control_admission_sequence,
                        item.revision_sequence,
                    ),
                )
            contract = revision.dynamic_parameter_contract
            selected: CompositionDynamicParameterRecord | None = None
            if contract is not None:
                candidates = tuple(
                    record
                    for record in self._parameter_records
                    if (
                        record.strategy_revision_hash
                        == revision.semantic_content_hash
                        and record.parameter_contract_hash
                        == contract.semantic_content_hash
                        and admitted_at_or_before(record, cut)
                        and producer_epochs.active_epoch_by_producer.get(
                            record.producer_id
                        )
                        == record.producer_generation
                        and record.effective_from <= target
                        and (
                            record.effective_until is None
                            or target < record.effective_until
                        )
                    )
                )
                if candidates:
                    selected = max(
                        candidates,
                        key=lambda item: (
                            item.effective_from,
                            item.control_admission_sequence,
                            item.parameter_record_id,
                        ),
                    )
            alpha = (
                revision.initial_alpha_by_component
                if selected is None
                else selected.alpha_by_component
            )
            snapshot = CompositionDynamicParameterSnapshot(
                strategy_revision_hash=revision.semantic_content_hash,
                parameter_contract_hash=(
                    None
                    if contract is None
                    else contract.semantic_content_hash
                ),
                control_domain_id=cut.control_domain_id,
                cutoff_at=cut.cutoff_at,
                cutoff_admission_sequence=cut.cutoff_admission_sequence,
                target_effective_at=target,
                selected_parameter_record_id=(
                    None if selected is None else selected.parameter_record_id
                ),
                selected_parameter_record_hash=(
                    None if selected is None else selected.semantic_content_hash
                ),
                alpha_by_component=alpha,
            )
            if revision.revision_sequence > self._applied_revision_sequence:
                self._applied_revision_sequence = revision.revision_sequence
            self._closed_target_through = target
            self._prune_parameters_locked(cut=cut, target=target)
            return CompositionStrategyControlBinding(
                revision=revision,
                parameter_snapshot=snapshot,
            )

    def _prune_parameters_locked(
        self,
        *,
        cut: ControlCut,
        target: float,
    ) -> None:
        prune_before = self.retention_spec.prune_before(
            closed_target_through=target
        )
        retained: list[CompositionDynamicParameterRecord] = []
        anchors: dict[str, CompositionDynamicParameterRecord] = {}
        for record in self._parameter_records:
            if record.control_admission_sequence > cut.cutoff_admission_sequence:
                retained.append(record)
                continue
            if record.effective_until is not None and (
                record.effective_until <= prune_before
            ):
                continue
            if record.effective_from >= prune_before:
                retained.append(record)
                continue
            current = anchors.get(record.strategy_revision_hash)
            if current is None or (
                record.effective_from,
                record.control_admission_sequence,
            ) > (
                current.effective_from,
                current.control_admission_sequence,
            ):
                anchors[record.strategy_revision_hash] = record
        retained.extend(anchors.values())
        self._parameter_records = retained


def build_initial_strategy_revision(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    strategy_instance_id: str,
    request_id: str,
    strategy_mode: str,
    composition_spec: CompositionSpec,
    strategy_spec: FormalCompositionStrategySpec,
    component_rule_set: ComponentCompositionRuleSet,
    dynamic_parameter_contract: CompositionDynamicParameterContract | None,
    initial_alpha_by_component: Mapping[str, float],
    available_at: float,
    available_clock_domain_id: str,
) -> CompositionStrategyRevision:
    return CompositionStrategyRevision(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        strategy_instance_id=strategy_instance_id,
        revision_sequence=0,
        previous_revision_hash=None,
        request_id=request_id,
        strategy_mode=strategy_mode,
        composition_spec=composition_spec,
        strategy_spec=strategy_spec,
        component_rule_set=component_rule_set,
        dynamic_parameter_contract=dynamic_parameter_contract,
        initial_alpha_by_component=initial_alpha_by_component,
        available_at=available_at,
        available_clock_domain_id=available_clock_domain_id,
        control_admission_sequence=0,
    )


__all__ = [
    "COMPOSITION_DYNAMIC_PARAMETER_CONTRACT_HASH_DOMAIN_V1",
    "COMPOSITION_DYNAMIC_PARAMETER_RECORD_HASH_DOMAIN_V1",
    "COMPOSITION_DYNAMIC_PARAMETER_SNAPSHOT_HASH_DOMAIN_V1",
    "COMPOSITION_STRATEGY_CONTROL_BINDING_HASH_DOMAIN_V1",
    "COMPOSITION_STRATEGY_REVISION_HASH_DOMAIN_V1",
    "CompositionDynamicParameterAdmissionReceipt",
    "CompositionDynamicParameterContract",
    "CompositionDynamicParameterDraft",
    "CompositionDynamicParameterRecord",
    "CompositionDynamicParameterSnapshot",
    "CompositionStrategyControl",
    "CompositionStrategyControlBinding",
    "CompositionStrategyRevision",
    "CompositionStrategyRevisionAdmissionReceipt",
    "CompositionStrategyRevisionDraft",
    "build_initial_strategy_revision",
]
