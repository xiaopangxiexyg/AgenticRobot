"""Device-neutral human joint-position residual adaptation.

This module converts an already-retargeted absolute joint-position sample into
an additive ``relative_delta`` action.  The device-specific mapping from raw
sensor values to the target joint layout remains with the source adapter.  An
explicit anchor makes the operation state-valued rather than an integrated
per-frame delta:

``residual(t) = gain * (absolute_target(t) - absolute_target(anchor))``.

The adapter owns no device, composition authority, safety, or execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.action_schema import ActionPayloadSchema
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ComponentAction, ComponentActionSchema
from robot_motion.runtime.immutable import deep_freeze


HUMAN_JOINT_POSITION_RESIDUAL_ALGORITHM_DEVELOPMENT = (
    "human_joint_position_residual.anchor_subtract.development"
)
HUMAN_JOINT_POSITION_SAMPLE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_joint_position_sample.development"
)
HUMAN_JOINT_POSITION_RESIDUAL_ANCHOR_ENTRY_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_joint_position_residual_anchor_entry.development"
)
HUMAN_JOINT_POSITION_RESIDUAL_ANCHOR_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_joint_position_residual_anchor.development"
)
HUMAN_JOINT_POSITION_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_joint_position_residual_adapter.development"
)
HUMAN_JOINT_POSITION_RESIDUAL_RECEIPT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_joint_position_residual_receipt.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _source_schema(value: object) -> ActionPayloadSchema:
    if not isinstance(value, ActionPayloadSchema):
        raise TypeError("source_payload_schema must be ActionPayloadSchema.")
    if (
        value.representation_family != "joint_position"
        or value.value_semantics != "absolute_target"
    ):
        raise ValueError(
            "Human joint-position samples require an absolute-target "
            "joint_position payload schema."
        )
    return value


def _gain(
    value: object,
    *,
    component_id: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"gain for {component_id!r} must be numeric."
        ) from exc
    if raw.shape == ():
        raw = np.full(shape, float(raw), dtype=np.float64)
    else:
        raw = raw.reshape(-1)
    if raw.shape != shape or not np.all(np.isfinite(raw)):
        raise ValueError(
            f"gain for {component_id!r} must be one finite scalar or have "
            f"shape {shape}."
        )
    return deep_freeze(raw)


@dataclass(frozen=True)
class HumanJointPositionSample:
    """One absolute, already-retargeted human joint-position sample."""

    component_id: str
    sample_id: str
    measured_at: float
    measured_clock_domain_id: str
    source_value_binding_id: str
    source_value_binding_hash: str
    source_payload_schema: ActionPayloadSchema
    values: Any
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "component_id",
                "sample_id",
                "measured_clock_domain_id",
                "source_value_binding_id",
                "source_value_binding_hash",
            )
        }
        measured_at = _finite(self.measured_at, name="measured_at")
        schema = _source_schema(self.source_payload_schema)
        values = schema.validate_values(self.values)
        payload = {
            **text,
            "measured_at": measured_at,
            "source_payload_schema_hash": schema.semantic_content_hash,
            "values": values,
        }
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "measured_at", measured_at)
        object.__setattr__(self, "source_payload_schema", schema)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=HUMAN_JOINT_POSITION_SAMPLE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @classmethod
    def from_component_action(
        cls,
        action: ComponentAction,
        component_id: str,
        *,
        sample_id: str,
        measured_at: float,
        measured_clock_domain_id: str,
        source_value_binding_id: str,
        source_value_binding_hash: str,
    ) -> "HumanJointPositionSample":
        """Select one exact absolute joint component without copying siblings."""

        if not isinstance(action, ComponentAction):
            raise TypeError("action must be ComponentAction.")
        component = _required(component_id, name="component_id")
        try:
            value = action.value(component)
        except KeyError as exc:
            raise ValueError(
                f"Component action does not contain {component!r}."
            ) from exc
        return cls(
            component_id=component,
            sample_id=sample_id,
            measured_at=measured_at,
            measured_clock_domain_id=measured_clock_domain_id,
            source_value_binding_id=source_value_binding_id,
            source_value_binding_hash=source_value_binding_hash,
            source_payload_schema=value.payload_schema,
            values=value.values,
        )


@dataclass(frozen=True)
class HumanJointPositionResidualAnchorEntry:
    component_id: str
    source_sample_id: str
    source_sample_hash: str
    source_measured_at: float
    source_measured_clock_domain_id: str
    source_value_binding_id: str
    source_value_binding_hash: str
    source_payload_schema_hash: str
    values: np.ndarray
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "component_id",
                "source_sample_id",
                "source_sample_hash",
                "source_measured_clock_domain_id",
                "source_value_binding_id",
                "source_value_binding_hash",
                "source_payload_schema_hash",
            )
        }
        measured_at = _finite(
            self.source_measured_at,
            name="source_measured_at",
        )
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if values.size <= 0 or not np.all(np.isfinite(values)):
            raise ValueError("Joint-position residual anchor values must be finite.")
        values = deep_freeze(values)
        payload = {
            **text,
            "source_measured_at": measured_at,
            "values": values,
        }
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_measured_at", measured_at)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    HUMAN_JOINT_POSITION_RESIDUAL_ANCHOR_ENTRY_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


@dataclass(frozen=True)
class HumanJointPositionResidualAnchor:
    """Explicit activation/recenter anchor for one exact source mapping."""

    anchor_generation: int
    captured_at: float
    captured_clock_domain_id: str
    entries: tuple[HumanJointPositionResidualAnchorEntry, ...]
    semantic_content_hash: str = field(init=False)
    anchor_id: str = field(init=False)
    entry_by_component: Mapping[
        str,
        HumanJointPositionResidualAnchorEntry,
    ] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        generation = int(self.anchor_generation)
        if generation < 0:
            raise ValueError("anchor_generation must be non-negative.")
        captured_at = _finite(self.captured_at, name="captured_at")
        clock = _required(
            self.captured_clock_domain_id,
            name="captured_clock_domain_id",
        )
        entries = tuple(self.entries)
        if not entries or any(
            not isinstance(item, HumanJointPositionResidualAnchorEntry)
            for item in entries
        ):
            raise TypeError(
                "entries must contain at least one "
                "HumanJointPositionResidualAnchorEntry."
            )
        ordered = tuple(sorted(entries, key=lambda item: item.component_id))
        by_component = {item.component_id: item for item in ordered}
        if len(by_component) != len(ordered):
            raise ValueError(
                "Human joint-position residual anchor component IDs must be unique."
            )
        for item in ordered:
            if (
                item.source_measured_clock_domain_id == clock
                and item.source_measured_at > captured_at
            ):
                raise ValueError(
                    "Joint-position anchor capture predates its source sample."
                )
        payload = {
            "anchor_generation": generation,
            "captured_at": captured_at,
            "captured_clock_domain_id": clock,
            "entry_hash_by_component": {
                item.component_id: item.semantic_content_hash for item in ordered
            },
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=HUMAN_JOINT_POSITION_RESIDUAL_ANCHOR_HASH_DOMAIN_DEVELOPMENT,
        )
        object.__setattr__(self, "anchor_generation", generation)
        object.__setattr__(self, "captured_at", captured_at)
        object.__setattr__(self, "captured_clock_domain_id", clock)
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(
            self,
            "entry_by_component",
            MappingProxyType(by_component),
        )
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "anchor_id",
            f"human-joint-position-residual-anchor:{generation}:{semantic_hash}",
        )


@dataclass(frozen=True)
class HumanJointPositionResidualReceipt:
    adapter_contract_hash: str
    anchor_id: str
    anchor_hash: str
    anchor_generation: int
    adapted_at: float
    adapted_clock_domain_id: str
    source_sample_hash_by_component: Mapping[str, str]
    source_payload_schema_hash_by_component: Mapping[str, str]
    gain_by_component: Mapping[str, tuple[float, ...]]
    output_action_schema_hash: str
    output_action_hash: str
    semantic_content_hash: str = field(init=False)
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "adapter_contract_hash": _required(
                self.adapter_contract_hash,
                name="adapter_contract_hash",
            ),
            "anchor_id": _required(self.anchor_id, name="anchor_id"),
            "anchor_hash": _required(self.anchor_hash, name="anchor_hash"),
            "anchor_generation": int(self.anchor_generation),
            "adapted_at": _finite(self.adapted_at, name="adapted_at"),
            "adapted_clock_domain_id": _required(
                self.adapted_clock_domain_id,
                name="adapted_clock_domain_id",
            ),
            "source_sample_hash_by_component": dict(
                self.source_sample_hash_by_component
            ),
            "source_payload_schema_hash_by_component": dict(
                self.source_payload_schema_hash_by_component
            ),
            "gain_by_component": {
                str(component_id): tuple(float(item) for item in values)
                for component_id, values in self.gain_by_component.items()
            },
            "output_action_schema_hash": _required(
                self.output_action_schema_hash,
                name="output_action_schema_hash",
            ),
            "output_action_hash": _required(
                self.output_action_hash,
                name="output_action_hash",
            ),
            "algorithm_id": (
                HUMAN_JOINT_POSITION_RESIDUAL_ALGORITHM_DEVELOPMENT
            ),
        }
        key_sets = {
            frozenset(payload[name])
            for name in (
                "source_sample_hash_by_component",
                "source_payload_schema_hash_by_component",
                "gain_by_component",
            )
        }
        if len(key_sets) != 1 or not next(iter(key_sets), frozenset()):
            raise ValueError(
                "Joint-position residual receipt component mappings must "
                "have the same non-empty key set."
            )
        semantic_hash = canonical_content_hash(
            payload,
            domain=HUMAN_JOINT_POSITION_RESIDUAL_RECEIPT_HASH_DOMAIN_DEVELOPMENT,
        )
        for name in (
            "source_sample_hash_by_component",
            "source_payload_schema_hash_by_component",
            "gain_by_component",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(payload[name])),
            )
        for name in (
            "adapter_contract_hash",
            "anchor_id",
            "anchor_hash",
            "anchor_generation",
            "adapted_at",
            "adapted_clock_domain_id",
            "output_action_schema_hash",
            "output_action_hash",
        ):
            object.__setattr__(self, name, payload[name])
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "receipt_id",
            f"human-joint-position-residual:{semantic_hash}",
        )


@dataclass(frozen=True)
class HumanJointPositionResidualResult:
    relative_action: ComponentAction
    receipt: HumanJointPositionResidualReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.relative_action, ComponentAction):
            raise TypeError("relative_action must be ComponentAction.")
        if not isinstance(self.receipt, HumanJointPositionResidualReceipt):
            raise TypeError(
                "receipt must be HumanJointPositionResidualReceipt."
            )
        if (
            self.relative_action.semantic_content_hash
            != self.receipt.output_action_hash
        ):
            raise ValueError(
                "Joint-position residual action and receipt hashes disagree."
            )


class HumanJointPositionResidualAdapter:
    """Stateless absolute-target to additive joint-position residual adapter."""

    def __init__(
        self,
        *,
        output_action_schema: ComponentActionSchema,
        source_payload_schema_by_component: Mapping[str, ActionPayloadSchema],
        gain_by_component: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(output_action_schema, ComponentActionSchema):
            raise TypeError(
                "output_action_schema must be ComponentActionSchema."
            )
        components = tuple(
            component_id
            for component_id in output_action_schema.component_ids
            if (
                output_action_schema.payload_schema_by_component[
                    component_id
                ].representation_family
                == "joint_position"
                and output_action_schema.payload_schema_by_component[
                    component_id
                ].value_semantics
                == "relative_delta"
            )
        )
        if not components:
            raise ValueError(
                "Human joint-position residual adapter requires at least one "
                "relative-delta joint_position component."
            )
        source_schemas = dict(source_payload_schema_by_component)
        if set(source_schemas) != set(components):
            raise ValueError(
                "Source payload schemas must exactly cover residual "
                "joint-position components."
            )
        normalized_sources: dict[str, ActionPayloadSchema] = {}
        for component_id in components:
            source = _source_schema(source_schemas[component_id])
            output = output_action_schema.payload_schema_by_component[
                component_id
            ]
            if (
                source.shape != output.shape
                or source.coordinate_names != output.coordinate_names
                or source.units != output.units
            ):
                raise ValueError(
                    f"Source and residual layout disagree for {component_id!r}."
                )
            normalized_sources[component_id] = source
        raw_gains = dict(gain_by_component or {})
        unknown_gains = sorted(set(raw_gains) - set(components))
        if unknown_gains:
            raise ValueError(
                f"Gains reference non-residual components: {unknown_gains}."
            )
        gains = {
            component_id: _gain(
                raw_gains.get(component_id, 1.0),
                component_id=component_id,
                shape=output_action_schema.payload_schema_by_component[
                    component_id
                ].shape,
            )
            for component_id in components
        }
        contract_payload = {
            "algorithm_id": (
                HUMAN_JOINT_POSITION_RESIDUAL_ALGORITHM_DEVELOPMENT
            ),
            "output_action_schema_hash": (
                output_action_schema.semantic_content_hash
            ),
            "components": components,
            "source_payload_schema_hash_by_component": {
                component_id: normalized_sources[
                    component_id
                ].semantic_content_hash
                for component_id in components
            },
            "gain_by_component": gains,
        }
        self.output_action_schema = output_action_schema
        self.components = components
        self.source_payload_schema_by_component = MappingProxyType(
            normalized_sources
        )
        self.gain_by_component = MappingProxyType(gains)
        self.contract_hash = canonical_content_hash(
            contract_payload,
            domain=HUMAN_JOINT_POSITION_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT,
        )

    def capture_anchor(
        self,
        samples: Sequence[HumanJointPositionSample],
        *,
        anchor_generation: int,
        captured_at: float,
        captured_clock_domain_id: str,
    ) -> HumanJointPositionResidualAnchor:
        selected = self._samples_by_component(samples)
        entries = tuple(
            HumanJointPositionResidualAnchorEntry(
                component_id=component_id,
                source_sample_id=selected[component_id].sample_id,
                source_sample_hash=selected[
                    component_id
                ].semantic_content_hash,
                source_measured_at=selected[component_id].measured_at,
                source_measured_clock_domain_id=selected[
                    component_id
                ].measured_clock_domain_id,
                source_value_binding_id=selected[
                    component_id
                ].source_value_binding_id,
                source_value_binding_hash=selected[
                    component_id
                ].source_value_binding_hash,
                source_payload_schema_hash=selected[
                    component_id
                ].source_payload_schema.semantic_content_hash,
                values=selected[component_id].values,
            )
            for component_id in self.components
        )
        return HumanJointPositionResidualAnchor(
            anchor_generation=anchor_generation,
            captured_at=captured_at,
            captured_clock_domain_id=captured_clock_domain_id,
            entries=entries,
        )

    def adapt(
        self,
        samples: Sequence[HumanJointPositionSample],
        *,
        anchor: HumanJointPositionResidualAnchor,
        adapted_at: float,
        adapted_clock_domain_id: str,
    ) -> HumanJointPositionResidualResult:
        if not isinstance(anchor, HumanJointPositionResidualAnchor):
            raise TypeError(
                "anchor must be HumanJointPositionResidualAnchor."
            )
        adapted_time = _finite(adapted_at, name="adapted_at")
        adapted_clock = _required(
            adapted_clock_domain_id,
            name="adapted_clock_domain_id",
        )
        if (
            adapted_clock == anchor.captured_clock_domain_id
            and adapted_time < anchor.captured_at
        ):
            raise ValueError(
                "Joint-position residual adaptation predates its anchor capture."
            )
        if tuple(anchor.entry_by_component) != tuple(sorted(self.components)):
            raise ValueError(
                "Joint-position residual anchor coverage does not match adapter."
            )
        selected = self._samples_by_component(samples)
        values: dict[str, np.ndarray] = {}
        for component_id in self.components:
            sample = selected[component_id]
            anchor_entry = anchor.entry_by_component[component_id]
            self._validate_sample_against_anchor(
                sample=sample,
                anchor=anchor_entry,
            )
            output_schema = (
                self.output_action_schema.payload_schema_by_component[
                    component_id
                ]
            )
            values[component_id] = (
                self.gain_by_component[component_id]
                * (
                    np.asarray(sample.values, dtype=np.float64)
                    - np.asarray(anchor_entry.values, dtype=np.float64)
                )
            ).astype(output_schema.dtype)
        action = ComponentAction(
            action_schema=self.output_action_schema,
            values_by_component=values,
        )
        receipt = HumanJointPositionResidualReceipt(
            adapter_contract_hash=self.contract_hash,
            anchor_id=anchor.anchor_id,
            anchor_hash=anchor.semantic_content_hash,
            anchor_generation=anchor.anchor_generation,
            adapted_at=adapted_time,
            adapted_clock_domain_id=adapted_clock,
            source_sample_hash_by_component={
                component_id: selected[
                    component_id
                ].semantic_content_hash
                for component_id in self.components
            },
            source_payload_schema_hash_by_component={
                component_id: self.source_payload_schema_by_component[
                    component_id
                ].semantic_content_hash
                for component_id in self.components
            },
            gain_by_component={
                component_id: tuple(
                    float(value)
                    for value in self.gain_by_component[component_id]
                )
                for component_id in self.components
            },
            output_action_schema_hash=(
                self.output_action_schema.semantic_content_hash
            ),
            output_action_hash=action.semantic_content_hash,
        )
        return HumanJointPositionResidualResult(
            relative_action=action,
            receipt=receipt,
        )

    def _samples_by_component(
        self,
        samples: Sequence[HumanJointPositionSample],
    ) -> Mapping[str, HumanJointPositionSample]:
        values = tuple(samples)
        if any(
            not isinstance(item, HumanJointPositionSample) for item in values
        ):
            raise TypeError(
                "samples must contain HumanJointPositionSample values."
            )
        selected = {item.component_id: item for item in values}
        if len(selected) != len(values):
            raise ValueError(
                "Human joint-position sample component IDs must be unique."
            )
        if set(selected) != set(self.components):
            raise ValueError(
                "Human joint-position samples must exactly cover adapter "
                "components."
            )
        for component_id in self.components:
            expected = self.source_payload_schema_by_component[
                component_id
            ].semantic_content_hash
            observed = selected[
                component_id
            ].source_payload_schema.semantic_content_hash
            if observed != expected:
                raise ValueError(
                    f"Human joint-position sample {component_id!r} uses "
                    "another source payload schema."
                )
        return selected

    @staticmethod
    def _validate_sample_against_anchor(
        *,
        sample: HumanJointPositionSample,
        anchor: HumanJointPositionResidualAnchorEntry,
    ) -> None:
        for field_name in (
            "source_value_binding_id",
            "source_value_binding_hash",
        ):
            if getattr(sample, field_name) != getattr(anchor, field_name):
                raise ValueError(
                    "Human joint-position sample and anchor disagree on "
                    f"{field_name}."
                )
        if (
            sample.source_payload_schema.semantic_content_hash
            != anchor.source_payload_schema_hash
        ):
            raise ValueError(
                "Human joint-position sample and anchor use different "
                "payload schemas."
            )
        if (
            sample.measured_clock_domain_id
            != anchor.source_measured_clock_domain_id
        ):
            raise ValueError(
                "Human joint-position sample and anchor use different "
                "measured clocks."
            )
        if sample.measured_at < anchor.source_measured_at:
            raise ValueError(
                "Human joint-position sample predates its recenter anchor."
            )


__all__ = [
    "HUMAN_JOINT_POSITION_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_JOINT_POSITION_RESIDUAL_ALGORITHM_DEVELOPMENT",
    "HUMAN_JOINT_POSITION_RESIDUAL_ANCHOR_ENTRY_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_JOINT_POSITION_RESIDUAL_ANCHOR_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_JOINT_POSITION_RESIDUAL_RECEIPT_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_JOINT_POSITION_SAMPLE_HASH_DOMAIN_DEVELOPMENT",
    "HumanJointPositionResidualAdapter",
    "HumanJointPositionResidualAnchor",
    "HumanJointPositionResidualAnchorEntry",
    "HumanJointPositionResidualReceipt",
    "HumanJointPositionResidualResult",
    "HumanJointPositionSample",
]
