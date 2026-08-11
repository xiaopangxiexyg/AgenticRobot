"""Passive observations exported by a downstream execution owner.

These contracts describe facts that an execution capability has already
produced.  They do not require a backend to expose a reference stage, do not
run interpolation, and do not grant authority or motion permission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze


EXECUTION_OBSERVATION_PLAN_HASH_DOMAIN = (
    "dexteleop.execution_observation_plan.development"
)
EXECUTION_FACT_REFERENCE_HASH_DOMAIN = (
    "dexteleop.execution_fact_reference.development"
)
EXECUTION_REFERENCE_FRAME_HASH_DOMAIN = (
    "dexteleop.execution_reference_frame_observation.development"
)
EXECUTION_MEASUREMENT_OBSERVATION_HASH_DOMAIN = (
    "dexteleop.execution_measurement_observation.development"
)
EXECUTION_OBSERVATION_CAPABILITY_HASH_DOMAIN = (
    "dexteleop.execution_observation_capability.development"
)
EXECUTION_OBSERVATION_BINDING_HASH_DOMAIN = (
    "dexteleop.execution_observation_binding.development"
)
EXECUTION_OBSERVATION_WATERMARK_HASH_DOMAIN = (
    "dexteleop.execution_observation_watermark.development"
)
REFERENCE_OBSERVATION_MODES = frozenset(
    {"off", "decimated_frames", "full_frames"}
)
MEASUREMENT_OBSERVATION_MODES = frozenset(
    {"off", "decimated_frames", "full_frames"}
)
EXECUTION_OBSERVATION_STREAMS = frozenset({"reference", "measurement"})
EXECUTION_OBSERVATION_WATERMARK_STATUSES = frozenset(
    {"healthy", "violated", "terminated"}
)
REFERENCE_FRAME_TEMPORAL_SEMANTICS = frozenset(
    {"point_sample", "active_interval"}
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _sequence(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _component_values(
    values: Mapping[str, Sequence[float]],
    *,
    name: str,
) -> Mapping[str, tuple[float, ...]]:
    normalized: dict[str, tuple[float, ...]] = {}
    for raw_component_id, raw_values in values.items():
        component_id = _required(raw_component_id, name=f"{name} component_id")
        if component_id in normalized:
            raise ValueError(f"{name} contains duplicate component IDs.")
        component_values = tuple(
            _finite(value, name=f"{name}[{component_id!r}]")
            for value in raw_values
        )
        if not component_values:
            raise ValueError(
                f"{name}[{component_id!r}] must contain at least one value."
            )
        normalized[component_id] = component_values
    if not normalized:
        raise ValueError(f"{name} must contain at least one component.")
    return deep_freeze(normalized)


def _component_text_mapping(
    values: Mapping[str, str],
    *,
    component_ids: tuple[str, ...],
    name: str,
) -> Mapping[str, str]:
    normalized = {
        _required(component_id, name=f"{name} component_id"): _required(
            value,
            name=f"{name}[{component_id!r}]",
        )
        for component_id, value in values.items()
    }
    if set(normalized) != set(component_ids):
        raise ValueError(f"{name} must cover exactly the value components.")
    return deep_freeze(normalized)


@dataclass(frozen=True)
class ExecutionObservationPlan:
    """Build-time demand for downstream-owned execution observations.

    ``None`` at the application boundary, or both modes set to ``off``, creates
    no observation obligation.  Once a lower capability accepts a non-off plan
    through :func:`bind_execution_observation_plan`, every selected frame is a
    required fact: unsupported requests fail binding instead of degrading
    silently.  ``full_frames`` means every frame emitted by the capability's
    declared stage; it does not implicitly require access to a faster private
    servo or physics loop.
    """

    plan_id: str
    reference_mode: str = "off"
    measurement_mode: str = "off"
    component_ids: tuple[str, ...] = ()
    maximum_reference_publish_hz: float | None = None
    maximum_measurement_publish_hz: float | None = None
    include_debug_lineage: bool = False
    required_representation_by_component: Mapping[str, str] = field(
        default_factory=dict
    )
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        plan_id = _required(self.plan_id, name="plan_id")
        mode = _required(self.reference_mode, name="reference_mode")
        if mode not in REFERENCE_OBSERVATION_MODES:
            raise ValueError("Unsupported reference observation mode.")
        measurement_mode = _required(
            self.measurement_mode,
            name="measurement_mode",
        )
        if measurement_mode not in MEASUREMENT_OBSERVATION_MODES:
            raise ValueError("Unsupported measurement observation mode.")
        component_ids = tuple(
            _required(value, name="component_id") for value in self.component_ids
        )
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component_ids must be unique.")
        maximum_reference_hz = self.maximum_reference_publish_hz
        if maximum_reference_hz is not None:
            maximum_reference_hz = _finite(
                maximum_reference_hz,
                name="maximum_reference_publish_hz",
            )
            if maximum_reference_hz <= 0.0:
                raise ValueError(
                    "maximum_reference_publish_hz must be positive."
                )
        if mode == "decimated_frames" and maximum_reference_hz is None:
            raise ValueError(
                "decimated_frames requires maximum_reference_publish_hz."
            )
        if mode != "decimated_frames" and maximum_reference_hz is not None:
            raise ValueError(
                "maximum_reference_publish_hz is only valid for "
                "decimated_frames."
            )
        maximum_measurement_hz = self.maximum_measurement_publish_hz
        if maximum_measurement_hz is not None:
            maximum_measurement_hz = _finite(
                maximum_measurement_hz,
                name="maximum_measurement_publish_hz",
            )
            if maximum_measurement_hz <= 0.0:
                raise ValueError(
                    "maximum_measurement_publish_hz must be positive."
                )
        if (
            measurement_mode == "decimated_frames"
            and maximum_measurement_hz is None
        ):
            raise ValueError(
                "decimated measurement frames require "
                "maximum_measurement_publish_hz."
            )
        if (
            measurement_mode != "decimated_frames"
            and maximum_measurement_hz is not None
        ):
            raise ValueError(
                "maximum_measurement_publish_hz is only valid for "
                "decimated_frames."
            )
        if (
            mode in {"decimated_frames", "full_frames"}
            or measurement_mode != "off"
        ) and not component_ids:
            raise ValueError(
                "Frame observations require at least one component_id."
            )
        required_representations = {
            _required(component_id, name="required representation component_id"):
            _required(
                representation,
                name=(
                    "required_representation_by_component"
                    f"[{component_id!r}]"
                ),
            )
            for component_id, representation in dict(
                self.required_representation_by_component
            ).items()
        }
        if required_representations and set(required_representations) != set(
            component_ids
        ):
            raise ValueError(
                "required_representation_by_component must cover exactly the "
                "selected component_ids when supplied."
            )
        payload = {
            "plan_id": plan_id,
            "reference_mode": mode,
            "measurement_mode": measurement_mode,
            "component_ids": component_ids,
            "maximum_reference_publish_hz": maximum_reference_hz,
            "maximum_measurement_publish_hz": maximum_measurement_hz,
            "include_debug_lineage": bool(self.include_debug_lineage),
        }
        frozen_required_representations = deep_freeze(
            required_representations
        )
        # Keep the historical plan hash byte-for-byte when no representation
        # requirement is supplied.  The new requirement is additive and only
        # enters identity for callers that explicitly opt in.
        if required_representations:
            payload["required_representation_by_component"] = (
                frozen_required_representations
            )
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "required_representation_by_component",
            frozen_required_representations,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=EXECUTION_OBSERVATION_PLAN_HASH_DOMAIN,
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.reference_mode != "off" or self.measurement_mode != "off"


@dataclass(frozen=True)
class ExecutionObservationCapabilitySpec:
    """Exact observation supply contract owned by one target implementation."""

    capability_id: str
    target_handler_id: str
    clock_domain_id: str
    supported_reference_modes: tuple[str, ...]
    supported_measurement_modes: tuple[str, ...]
    representation_by_component: Mapping[str, str]
    schema_hash_by_component: Mapping[str, str]
    reference_stage_id: str | None = None
    measurement_stage_id: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        reference_modes = tuple(
            _required(value, name="supported_reference_mode")
            for value in self.supported_reference_modes
        )
        measurement_modes = tuple(
            _required(value, name="supported_measurement_mode")
            for value in self.supported_measurement_modes
        )
        if (
            not reference_modes
            or len(reference_modes) != len(set(reference_modes))
            or not set(reference_modes).issubset(REFERENCE_OBSERVATION_MODES)
        ):
            raise ValueError(
                "supported_reference_modes must be a unique non-empty subset."
            )
        if (
            not measurement_modes
            or len(measurement_modes) != len(set(measurement_modes))
            or not set(measurement_modes).issubset(MEASUREMENT_OBSERVATION_MODES)
        ):
            raise ValueError(
                "supported_measurement_modes must be a unique non-empty subset."
            )
        representations = {
            _required(component_id, name="component_id"): _required(
                representation,
                name=f"representation_by_component[{component_id!r}]",
            )
            for component_id, representation in self.representation_by_component.items()
        }
        schemas = {
            _required(component_id, name="component_id"): _required(
                schema_hash,
                name=f"schema_hash_by_component[{component_id!r}]",
            )
            for component_id, schema_hash in self.schema_hash_by_component.items()
        }
        if not representations or set(representations) != set(schemas):
            raise ValueError(
                "Observation capability representation/schema mappings must "
                "cover the same non-empty component set."
            )
        reference_stage = (
            None
            if self.reference_stage_id is None
            else _required(self.reference_stage_id, name="reference_stage_id")
        )
        measurement_stage = (
            None
            if self.measurement_stage_id is None
            else _required(self.measurement_stage_id, name="measurement_stage_id")
        )
        if any(
            mode in {"decimated_frames", "full_frames"}
            for mode in reference_modes
        ) and reference_stage is None:
            raise ValueError(
                "Frame-capable reference observations require reference_stage_id."
            )
        if (
            any(mode != "off" for mode in measurement_modes)
            and measurement_stage is None
        ):
            raise ValueError(
                "Frame-capable measurement observations require "
                "measurement_stage_id."
            )
        payload = {
            "capability_id": _required(
                self.capability_id,
                name="capability_id",
            ),
            "target_handler_id": _required(
                self.target_handler_id,
                name="target_handler_id",
            ),
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "supported_reference_modes": reference_modes,
            "supported_measurement_modes": measurement_modes,
            "representation_by_component": deep_freeze(representations),
            "schema_hash_by_component": deep_freeze(schemas),
            "reference_stage_id": reference_stage,
            "measurement_stage_id": measurement_stage,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=EXECUTION_OBSERVATION_CAPABILITY_HASH_DOMAIN,
            ),
        )


@dataclass(frozen=True)
class ExecutionObservationBinding:
    """Proof that one lower owner accepted an entire non-off plan."""

    plan_id: str
    plan_hash: str
    capability_id: str
    capability_hash: str
    target_handler_id: str
    clock_domain_id: str
    component_ids: tuple[str, ...]
    reference_mode: str
    measurement_mode: str
    bound_at: float
    binding_id: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        component_ids = tuple(
            _required(value, name="component_id") for value in self.component_ids
        )
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("Observation binding component_ids must be unique.")
        reference_mode = _required(self.reference_mode, name="reference_mode")
        measurement_mode = _required(
            self.measurement_mode,
            name="measurement_mode",
        )
        if reference_mode not in REFERENCE_OBSERVATION_MODES:
            raise ValueError("Unsupported bound reference mode.")
        if measurement_mode not in MEASUREMENT_OBSERVATION_MODES:
            raise ValueError("Unsupported bound measurement mode.")
        payload = {
            "plan_id": _required(self.plan_id, name="plan_id"),
            "plan_hash": _required(self.plan_hash, name="plan_hash"),
            "capability_id": _required(
                self.capability_id,
                name="capability_id",
            ),
            "capability_hash": _required(
                self.capability_hash,
                name="capability_hash",
            ),
            "target_handler_id": _required(
                self.target_handler_id,
                name="target_handler_id",
            ),
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "component_ids": component_ids,
            "reference_mode": reference_mode,
            "measurement_mode": measurement_mode,
            "bound_at": _finite(self.bound_at, name="bound_at"),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=EXECUTION_OBSERVATION_BINDING_HASH_DOMAIN,
        )
        for name, value in (
            *payload.items(),
            ("binding_id", f"execution-observation-binding:{semantic_hash}"),
            ("semantic_content_hash", semantic_hash),
        ):
            object.__setattr__(self, name, value)


def bind_execution_observation_plan(
    plan: ExecutionObservationPlan,
    capability: ExecutionObservationCapabilitySpec,
    *,
    bound_at: float,
) -> ExecutionObservationBinding:
    """Accept the entire plan or fail before target runtime construction."""

    if not isinstance(plan, ExecutionObservationPlan):
        raise TypeError("plan must be an ExecutionObservationPlan.")
    if not isinstance(capability, ExecutionObservationCapabilitySpec):
        raise TypeError(
            "capability must be an ExecutionObservationCapabilitySpec."
        )
    if not plan.enabled:
        raise ValueError("An off observation plan does not create a binding.")
    if plan.reference_mode not in capability.supported_reference_modes:
        raise ValueError(
            "Target observation capability does not support requested "
            "reference_mode."
        )
    if plan.measurement_mode not in capability.supported_measurement_modes:
        raise ValueError(
            "Target observation capability does not support requested "
            "measurement_mode."
        )
    unknown_components = sorted(
        set(plan.component_ids) - set(capability.representation_by_component)
    )
    if unknown_components:
        raise ValueError(
            "Target observation capability does not cover components: "
            f"{unknown_components!r}."
        )
    representation_mismatches = {
        component_id: {
            "required": required,
            "provided": capability.representation_by_component[component_id],
        }
        for component_id, required in (
            plan.required_representation_by_component.items()
        )
        if capability.representation_by_component[component_id] != required
    }
    if representation_mismatches:
        raise ValueError(
            "Target observation capability does not provide the requested "
            f"component representations: {representation_mismatches!r}."
        )
    return ExecutionObservationBinding(
        plan_id=plan.plan_id,
        plan_hash=plan.semantic_content_hash,
        capability_id=capability.capability_id,
        capability_hash=capability.semantic_content_hash,
        target_handler_id=capability.target_handler_id,
        clock_domain_id=capability.clock_domain_id,
        component_ids=plan.component_ids,
        reference_mode=plan.reference_mode,
        measurement_mode=plan.measurement_mode,
        bound_at=bound_at,
    )


@dataclass(frozen=True)
class ExecutionFactReference:
    """Opaque identity of one downstream fact used by an observation."""

    stage: str
    fact_id: str
    fact_hash: str
    observed_at: float
    clock_domain_id: str
    status: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "stage": _required(self.stage, name="stage"),
            "fact_id": _required(self.fact_id, name="fact_id"),
            "fact_hash": _required(self.fact_hash, name="fact_hash"),
            "observed_at": _finite(self.observed_at, name="observed_at"),
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "status": _required(self.status, name="status"),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=EXECUTION_FACT_REFERENCE_HASH_DOMAIN,
            ),
        )


def _fact_references(
    values: Sequence[ExecutionFactReference],
) -> tuple[ExecutionFactReference, ...]:
    references = tuple(values)
    if any(not isinstance(value, ExecutionFactReference) for value in references):
        raise TypeError("parent_facts must contain ExecutionFactReference values.")
    hashes = tuple(value.semantic_content_hash for value in references)
    if len(hashes) != len(set(hashes)):
        raise ValueError("parent_facts must be unique.")
    return references


@dataclass(frozen=True)
class ExecutionReferenceFrameObservation:
    """Exact reference value exported from the object used by execution."""

    reference_frame_id: str
    reference_stage_id: str
    source_reference_id: str
    source_reference_hash: str
    target_fact_id: str
    target_fact_hash: str
    control_domain_id: str
    selection_group_id: str
    execution_fence_sequence: int
    reference_revision: int
    reference_sequence: int
    temporal_semantics: str
    scheduled_for: float
    generated_at: float
    effective_from: float | None
    effective_until: float | None
    available_at: float
    clock_domain_id: str
    value_by_component: Mapping[str, Sequence[float]]
    representation_by_component: Mapping[str, str]
    schema_hash_by_component: Mapping[str, str]
    interpolation_contract_hash: str | None = None
    observation_binding_id: str | None = None
    observation_plan_hash: str | None = None
    parent_facts: tuple[ExecutionFactReference, ...] = ()
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = _component_values(
            self.value_by_component,
            name="value_by_component",
        )
        component_ids = tuple(values)
        representations = _component_text_mapping(
            self.representation_by_component,
            component_ids=component_ids,
            name="representation_by_component",
        )
        schemas = _component_text_mapping(
            self.schema_hash_by_component,
            component_ids=component_ids,
            name="schema_hash_by_component",
        )
        scheduled_for = _finite(self.scheduled_for, name="scheduled_for")
        generated_at = _finite(self.generated_at, name="generated_at")
        temporal_semantics = _required(
            self.temporal_semantics,
            name="temporal_semantics",
        )
        if temporal_semantics not in REFERENCE_FRAME_TEMPORAL_SEMANTICS:
            raise ValueError("Unsupported reference frame temporal semantics.")
        effective_from = self.effective_from
        if effective_from is not None:
            effective_from = _finite(effective_from, name="effective_from")
        effective_until = self.effective_until
        if effective_until is not None:
            effective_until = _finite(
                effective_until,
                name="effective_until",
            )
        if temporal_semantics == "point_sample":
            if effective_from is not None or effective_until is not None:
                raise ValueError(
                    "point_sample reference frames cannot claim an active interval."
                )
        else:
            if effective_from is None:
                raise ValueError(
                    "active_interval reference frames require effective_from."
                )
            if effective_until is not None and effective_until <= effective_from:
                raise ValueError(
                    "Reference interval must be [effective_from, "
                    "effective_until)."
                )
        available_at = _finite(self.available_at, name="available_at")
        if available_at + 1e-12 < generated_at:
            raise ValueError("Reference cannot be available before generation.")
        interpolation_hash = (
            None
            if self.interpolation_contract_hash is None
            else _required(
                self.interpolation_contract_hash,
                name="interpolation_contract_hash",
            )
        )
        binding_id = (
            None
            if self.observation_binding_id is None
            else _required(
                self.observation_binding_id,
                name="observation_binding_id",
            )
        )
        plan_hash = (
            None
            if self.observation_plan_hash is None
            else _required(
                self.observation_plan_hash,
                name="observation_plan_hash",
            )
        )
        if (binding_id is None) != (plan_hash is None):
            raise ValueError(
                "Observation binding ID and plan hash must appear together."
            )
        parents = _fact_references(self.parent_facts)
        payload = {
            "reference_frame_id": _required(
                self.reference_frame_id,
                name="reference_frame_id",
            ),
            "reference_stage_id": _required(
                self.reference_stage_id,
                name="reference_stage_id",
            ),
            "source_reference_id": _required(
                self.source_reference_id,
                name="source_reference_id",
            ),
            "source_reference_hash": _required(
                self.source_reference_hash,
                name="source_reference_hash",
            ),
            "target_fact_id": _required(
                self.target_fact_id,
                name="target_fact_id",
            ),
            "target_fact_hash": _required(
                self.target_fact_hash,
                name="target_fact_hash",
            ),
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "selection_group_id": _required(
                self.selection_group_id,
                name="selection_group_id",
            ),
            "execution_fence_sequence": _sequence(
                self.execution_fence_sequence,
                name="execution_fence_sequence",
            ),
            "reference_revision": _sequence(
                self.reference_revision,
                name="reference_revision",
            ),
            "reference_sequence": _sequence(
                self.reference_sequence,
                name="reference_sequence",
            ),
            "temporal_semantics": temporal_semantics,
            "scheduled_for": scheduled_for,
            "generated_at": generated_at,
            "effective_from": effective_from,
            "effective_until": effective_until,
            "available_at": available_at,
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "value_by_component": values,
            "representation_by_component": representations,
            "schema_hash_by_component": schemas,
            "interpolation_contract_hash": interpolation_hash,
            "observation_binding_id": binding_id,
            "observation_plan_hash": plan_hash,
            "parent_fact_hashes": tuple(
                value.semantic_content_hash for value in parents
            ),
        }
        for name, value in (
            *payload.items(),
            ("parent_facts", parents),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=EXECUTION_REFERENCE_FRAME_HASH_DOMAIN,
            ),
        )

    def covers(self, measured_at: float) -> bool:
        timestamp = _finite(measured_at, name="measured_at")
        if self.temporal_semantics != "active_interval":
            return False
        assert self.effective_from is not None
        return self.effective_from <= timestamp and (
            self.effective_until is None or timestamp < self.effective_until
        )


@dataclass(frozen=True)
class ExecutionMeasurementObservation:
    """One backend-owned measurement projected into component-keyed values."""

    measurement_id: str
    source_measurement_hash: str
    control_domain_id: str
    measured_at: float
    available_at: float
    clock_domain_id: str
    value_by_component: Mapping[str, Sequence[float]]
    representation_by_component: Mapping[str, str]
    schema_hash_by_component: Mapping[str, str]
    observation_binding_id: str | None = None
    observation_plan_hash: str | None = None
    parent_facts: tuple[ExecutionFactReference, ...] = ()
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = _component_values(
            self.value_by_component,
            name="value_by_component",
        )
        component_ids = tuple(values)
        representations = _component_text_mapping(
            self.representation_by_component,
            component_ids=component_ids,
            name="representation_by_component",
        )
        schemas = _component_text_mapping(
            self.schema_hash_by_component,
            component_ids=component_ids,
            name="schema_hash_by_component",
        )
        measured_at = _finite(self.measured_at, name="measured_at")
        available_at = _finite(self.available_at, name="available_at")
        if available_at + 1e-12 < measured_at:
            raise ValueError("Measurement cannot be available before measured_at.")
        binding_id = (
            None
            if self.observation_binding_id is None
            else _required(
                self.observation_binding_id,
                name="observation_binding_id",
            )
        )
        plan_hash = (
            None
            if self.observation_plan_hash is None
            else _required(
                self.observation_plan_hash,
                name="observation_plan_hash",
            )
        )
        if (binding_id is None) != (plan_hash is None):
            raise ValueError(
                "Observation binding ID and plan hash must appear together."
            )
        parents = _fact_references(self.parent_facts)
        payload = {
            "measurement_id": _required(
                self.measurement_id,
                name="measurement_id",
            ),
            "source_measurement_hash": _required(
                self.source_measurement_hash,
                name="source_measurement_hash",
            ),
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "measured_at": measured_at,
            "available_at": available_at,
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "value_by_component": values,
            "representation_by_component": representations,
            "schema_hash_by_component": schemas,
            "observation_binding_id": binding_id,
            "observation_plan_hash": plan_hash,
            "parent_fact_hashes": tuple(
                value.semantic_content_hash for value in parents
            ),
        }
        for name, value in (
            *payload.items(),
            ("parent_facts", parents),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=EXECUTION_MEASUREMENT_OBSERVATION_HASH_DOMAIN,
            ),
        )


def validate_execution_observation_binding(
    observation: ExecutionReferenceFrameObservation
    | ExecutionMeasurementObservation,
    binding: ExecutionObservationBinding,
    capability: ExecutionObservationCapabilitySpec,
) -> None:
    """Reject observations that do not fulfill their accepted build binding."""

    if not isinstance(binding, ExecutionObservationBinding):
        raise TypeError("binding must be an ExecutionObservationBinding.")
    if not isinstance(capability, ExecutionObservationCapabilitySpec):
        raise TypeError(
            "capability must be an ExecutionObservationCapabilitySpec."
        )
    if (
        binding.capability_id != capability.capability_id
        or binding.capability_hash != capability.semantic_content_hash
    ):
        raise ValueError("Observation binding references another capability.")
    if (
        observation.observation_binding_id != binding.binding_id
        or observation.observation_plan_hash != binding.plan_hash
    ):
        raise ValueError("Observation does not reference the accepted plan binding.")
    if observation.clock_domain_id != binding.clock_domain_id:
        raise ValueError("Observation and binding clock domains must match.")
    component_ids = tuple(observation.value_by_component)
    if set(component_ids) != set(binding.component_ids):
        raise ValueError(
            "Observation components must exactly cover the accepted plan."
        )
    for component_id in component_ids:
        if (
            observation.representation_by_component[component_id]
            != capability.representation_by_component[component_id]
            or observation.schema_hash_by_component[component_id]
            != capability.schema_hash_by_component[component_id]
        ):
            raise ValueError(
                f"Observation component {component_id!r} does not match "
                "the accepted capability contract."
            )
    if isinstance(observation, ExecutionReferenceFrameObservation):
        if binding.reference_mode == "off":
            raise ValueError("Accepted binding does not permit reference frames.")
        if observation.reference_stage_id != capability.reference_stage_id:
            raise ValueError("Reference observation uses another capability stage.")
    else:
        if binding.measurement_mode == "off":
            raise ValueError("Accepted binding does not permit measurement frames.")


@dataclass(frozen=True)
class ExecutionObservationWatermark:
    """Lower-owned proof of progress, loss, or termination for one stream."""

    observation_binding_id: str
    observation_plan_hash: str
    stream: str
    processed_through: float
    latest_sequence: int
    emitted_count: int
    dropped_count: int
    status: str
    reason: str
    available_at: float
    clock_domain_id: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        stream = _required(self.stream, name="stream")
        if stream not in EXECUTION_OBSERVATION_STREAMS:
            raise ValueError("Unsupported execution observation stream.")
        status = _required(self.status, name="status")
        if status not in EXECUTION_OBSERVATION_WATERMARK_STATUSES:
            raise ValueError("Unsupported execution observation watermark status.")
        emitted_count = _sequence(self.emitted_count, name="emitted_count")
        dropped_count = _sequence(self.dropped_count, name="dropped_count")
        if status == "healthy" and dropped_count:
            raise ValueError("A healthy observation watermark cannot report drops.")
        processed_through = _finite(
            self.processed_through,
            name="processed_through",
        )
        available_at = _finite(self.available_at, name="available_at")
        if available_at + 1e-12 < processed_through:
            raise ValueError(
                "Observation watermark cannot be available before its progress cut."
            )
        payload = {
            "observation_binding_id": _required(
                self.observation_binding_id,
                name="observation_binding_id",
            ),
            "observation_plan_hash": _required(
                self.observation_plan_hash,
                name="observation_plan_hash",
            ),
            "stream": stream,
            "processed_through": processed_through,
            "latest_sequence": _sequence(
                self.latest_sequence,
                name="latest_sequence",
            ),
            "emitted_count": emitted_count,
            "dropped_count": dropped_count,
            "status": status,
            "reason": _required(self.reason, name="reason"),
            "available_at": available_at,
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=EXECUTION_OBSERVATION_WATERMARK_HASH_DOMAIN,
            ),
        )


class ExecutionObservationProvider(Protocol):
    """Read-only lower-owned observation port for an accepted binding."""

    @property
    def execution_observation_binding(
        self,
    ) -> ExecutionObservationBinding | None:
        ...

    @property
    def execution_observation_capability(
        self,
    ) -> ExecutionObservationCapabilitySpec | None:
        ...


class ExecutionObservationSink(Protocol):
    """Nonblocking consumer handoff; ``False`` is an explicit delivery failure."""

    def publish_execution_observation(
        self,
        observation: ExecutionReferenceFrameObservation
        | ExecutionMeasurementObservation
        | ExecutionObservationWatermark,
    ) -> bool:
        ...


__all__ = [
    "EXECUTION_OBSERVATION_BINDING_HASH_DOMAIN",
    "EXECUTION_OBSERVATION_CAPABILITY_HASH_DOMAIN",
    "EXECUTION_FACT_REFERENCE_HASH_DOMAIN",
    "EXECUTION_MEASUREMENT_OBSERVATION_HASH_DOMAIN",
    "EXECUTION_OBSERVATION_PLAN_HASH_DOMAIN",
    "EXECUTION_OBSERVATION_STREAMS",
    "EXECUTION_OBSERVATION_WATERMARK_HASH_DOMAIN",
    "EXECUTION_OBSERVATION_WATERMARK_STATUSES",
    "EXECUTION_REFERENCE_FRAME_HASH_DOMAIN",
    "MEASUREMENT_OBSERVATION_MODES",
    "REFERENCE_OBSERVATION_MODES",
    "REFERENCE_FRAME_TEMPORAL_SEMANTICS",
    "ExecutionObservationBinding",
    "ExecutionObservationCapabilitySpec",
    "ExecutionFactReference",
    "ExecutionMeasurementObservation",
    "ExecutionObservationPlan",
    "ExecutionObservationProvider",
    "ExecutionObservationSink",
    "ExecutionObservationWatermark",
    "ExecutionReferenceFrameObservation",
    "bind_execution_observation_plan",
    "validate_execution_observation_binding",
]
