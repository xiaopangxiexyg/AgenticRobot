"""Pure model-RTC continuation timeline and encoding-lineage contracts.

These contracts stop before model invocation and before target execution.
They distinguish the model generation lattice from the executable suffix so
forecast delay is applied exactly once.  The first implementation recognizes
only producer-local overlap; downstream target acceptance is intentionally
outside this package and cannot be inferred from sampler constraint strength.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash

from ..chunk_plan import (
    POLICY_ACTION_PAYLOAD_TYPED_CANONICAL_V1,
    TimedPolicyActionKeyframe,
    TimedPolicyActionPlanSnapshot,
)
from ..inference import PolicyInferenceRequest


RTC_CONTINUATION_TIMELINE_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_continuation_timeline.v1"
)
RTC_CONTINUATION_ENCODING_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_continuation_encoding_receipt.v1"
)
RTC_CONTINUATION_SELECTION_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_continuation_selection_spec.v1"
)
RTC_PLAN_CONTINUATION_ENTRY_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_plan_continuation_entry.v1"
)
RTC_PLAN_CONTINUATION_SLICE_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_plan_continuation_slice.v1"
)
RTC_CANONICAL_CONTINUATION_PREFIX_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_canonical_continuation_prefix.v1"
)
RTC_CONTINUATION_SELECTION_OUTCOME_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_continuation_selection_outcome.v1"
)

# Frozen compatibility names used only by the deprecated pre-C1 encoding
# receipt below.  Canonical sampler/source identities now live in
# dataset_contracts.model_continuation and deployments.client.
RTC_CONTINUATION_PREFIX_SOURCE_PRODUCER_LOCAL_OVERLAP_V1 = (
    "producer_local_overlap_v1"
)
RTC_CONTINUATION_SAMPLER_CONSTRAINT_CLAMPED_V1 = "clamped_v1"
RTC_CONTINUATION_SAMPLER_CONSTRAINT_WEIGHTED_V1 = "weighted_v1"
RTC_CONTINUATION_SAMPLER_CONSTRAINTS_V1 = (
    RTC_CONTINUATION_SAMPLER_CONSTRAINT_CLAMPED_V1,
    RTC_CONTINUATION_SAMPLER_CONSTRAINT_WEIGHTED_V1,
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized != value or normalized <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _normalized_strings(
    values: tuple[str, ...],
    *,
    name: str,
    require_nonempty: bool,
) -> tuple[str, ...]:
    normalized = tuple(
        sorted({_nonempty(value, name=f"{name} item") for value in values})
    )
    if require_nonempty and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class RtcContinuationTimeline:
    """Exact relation between a generated chunk and its executable suffix."""

    clock_domain_id: str
    generation_start_at: float
    first_target_effective_at: float
    action_period_sec: float
    prediction_horizon_steps: int
    forecast_delay_steps: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clock_domain_id",
            _nonempty(self.clock_domain_id, name="clock_domain_id"),
        )
        generation_start = _finite(
            self.generation_start_at,
            name="generation_start_at",
        )
        first_target = _finite(
            self.first_target_effective_at,
            name="first_target_effective_at",
        )
        period = _finite(self.action_period_sec, name="action_period_sec")
        if period <= 0.0:
            raise ValueError("action_period_sec must be positive.")
        horizon = _positive_int(
            self.prediction_horizon_steps,
            name="prediction_horizon_steps",
        )
        delay = _nonnegative_int(
            self.forecast_delay_steps,
            name="forecast_delay_steps",
        )
        if delay >= horizon:
            raise ValueError(
                "forecast_delay_steps must leave a non-empty executable suffix."
            )
        expected_first_target = generation_start + delay * period
        if not math.isclose(
            first_target,
            expected_first_target,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "first_target_effective_at must equal generation_start_at plus "
                "forecast_delay_steps * action_period_sec."
            )
        object.__setattr__(self, "generation_start_at", generation_start)
        object.__setattr__(self, "first_target_effective_at", first_target)
        object.__setattr__(self, "action_period_sec", period)
        object.__setattr__(self, "prediction_horizon_steps", horizon)
        object.__setattr__(self, "forecast_delay_steps", delay)

    @property
    def maximum_execution_horizon_steps(self) -> int:
        return self.prediction_horizon_steps - self.forecast_delay_steps

    def generation_target_at(self, index: int) -> float:
        normalized = _nonnegative_int(index, name="generation index")
        if normalized >= self.prediction_horizon_steps:
            raise IndexError("generation index exceeds the prediction horizon.")
        return self.generation_start_at + normalized * self.action_period_sec

    def response_target_at(self, response_index: int) -> float:
        normalized = _nonnegative_int(response_index, name="response index")
        if normalized >= self.maximum_execution_horizon_steps:
            raise IndexError("response index exceeds the executable suffix.")
        return self.first_target_effective_at + normalized * self.action_period_sec

    def generation_index_for_response(self, response_index: int) -> int:
        normalized = _nonnegative_int(response_index, name="response index")
        if normalized >= self.maximum_execution_horizon_steps:
            raise IndexError("response index exceeds the executable suffix.")
        return self.forecast_delay_steps + normalized

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "clock_domain_id": self.clock_domain_id,
                "generation_start_at": self.generation_start_at,
                "first_target_effective_at": self.first_target_effective_at,
                "action_period_sec": self.action_period_sec,
                "prediction_horizon_steps": self.prediction_horizon_steps,
                "forecast_delay_steps": self.forecast_delay_steps,
            },
            domain=RTC_CONTINUATION_TIMELINE_HASH_DOMAIN_V1,
        )


def rtc_continuation_timeline_from_request(
    request: PolicyInferenceRequest,
) -> RtcContinuationTimeline:
    """Derive the full generation lattice from an admitted request."""

    if not isinstance(request, PolicyInferenceRequest):
        raise TypeError("request must be a PolicyInferenceRequest.")
    return RtcContinuationTimeline(
        clock_domain_id=request.clock_domain_id,
        generation_start_at=(
            request.first_target_effective_at
            - request.forecast_delay_steps * request.action_period_sec
        ),
        first_target_effective_at=request.first_target_effective_at,
        action_period_sec=request.action_period_sec,
        prediction_horizon_steps=request.prediction_horizon_steps,
        forecast_delay_steps=request.forecast_delay_steps,
    )


@dataclass(frozen=True)
class RtcContinuationSelectionSpec:
    """Fail-closed policy for selecting old-plan rows on a generation lattice."""

    prefix_horizon_steps: int
    minimum_selected_steps: int = 1
    lattice_tolerance_sec: float = 1e-9

    def __post_init__(self) -> None:
        horizon = _positive_int(
            self.prefix_horizon_steps,
            name="prefix_horizon_steps",
        )
        minimum = _positive_int(
            self.minimum_selected_steps,
            name="minimum_selected_steps",
        )
        if minimum > horizon:
            raise ValueError(
                "minimum_selected_steps cannot exceed prefix_horizon_steps."
            )
        tolerance = _finite(
            self.lattice_tolerance_sec,
            name="lattice_tolerance_sec",
        )
        if tolerance < 0.0:
            raise ValueError("lattice_tolerance_sec must be non-negative.")
        object.__setattr__(self, "prefix_horizon_steps", horizon)
        object.__setattr__(self, "minimum_selected_steps", minimum)
        object.__setattr__(self, "lattice_tolerance_sec", tolerance)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "prefix_horizon_steps": self.prefix_horizon_steps,
                "minimum_selected_steps": self.minimum_selected_steps,
                "lattice_tolerance_sec": self.lattice_tolerance_sec,
            },
            domain=RTC_CONTINUATION_SELECTION_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class RtcPlanContinuationEntry:
    """One exact old-plan keyframe aligned to a full generation-chunk index."""

    generation_index: int
    keyframe: TimedPolicyActionKeyframe

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_int(self.generation_index, name="generation_index"),
        )
        if not isinstance(self.keyframe, TimedPolicyActionKeyframe):
            raise TypeError("keyframe must be a TimedPolicyActionKeyframe.")
        if (
            self.keyframe.action_payload_kind
            != POLICY_ACTION_PAYLOAD_TYPED_CANONICAL_V1
        ):
            raise ValueError(
                "A model-RTC continuation entry requires a typed canonical action."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "generation_index": self.generation_index,
                "keyframe_hash": self.keyframe.semantic_content_hash,
            },
            domain=RTC_PLAN_CONTINUATION_ENTRY_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class RtcPlanContinuationSlice:
    """Immutable producer-local overlap selected before request admission."""

    source_plan_id: str
    source_plan_generation: int
    source_plan_hash: str
    producer_id: str
    producer_epoch: int
    policy_node_id: str
    timeline_hash: str
    selection_spec_hash: str
    prefix_horizon_steps: int
    entries: tuple[RtcPlanContinuationEntry, ...]

    def __post_init__(self) -> None:
        for name in (
            "source_plan_id",
            "source_plan_hash",
            "producer_id",
            "policy_node_id",
            "timeline_hash",
            "selection_spec_hash",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "source_plan_generation",
            _nonnegative_int(
                self.source_plan_generation,
                name="source_plan_generation",
            ),
        )
        object.__setattr__(
            self,
            "producer_epoch",
            _nonnegative_int(self.producer_epoch, name="producer_epoch"),
        )
        horizon = _positive_int(
            self.prefix_horizon_steps,
            name="prefix_horizon_steps",
        )
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("A continuation slice must contain at least one entry.")
        previous_index: int | None = None
        for entry in entries:
            if not isinstance(entry, RtcPlanContinuationEntry):
                raise TypeError(
                    "entries must contain RtcPlanContinuationEntry values."
                )
            if entry.generation_index >= horizon:
                raise ValueError(
                    "Continuation entry lies outside prefix_horizon_steps."
                )
            if (
                previous_index is not None
                and entry.generation_index != previous_index + 1
            ):
                raise ValueError(
                    "Continuation entry generation indices must be contiguous."
                )
            if entry.keyframe.producer_epoch != self.producer_epoch:
                raise ValueError("Continuation entry producer epoch mismatch.")
            if entry.keyframe.policy_node_id != self.policy_node_id:
                raise ValueError("Continuation entry policy node mismatch.")
            if (
                entry.keyframe.admitted_plan_generation
                > self.source_plan_generation
            ):
                raise ValueError(
                    "Continuation entry admission generation exceeds source plan."
                )
            previous_index = entry.generation_index
        object.__setattr__(self, "prefix_horizon_steps", horizon)
        object.__setattr__(self, "entries", entries)

    @property
    def first_generation_index(self) -> int:
        return self.entries[0].generation_index

    @property
    def valid_prefix_steps(self) -> int:
        """Exclusive upper generation index used by the later prefix mask."""

        return self.entries[-1].generation_index + 1

    @property
    def selected_generation_indices(self) -> tuple[int, ...]:
        return tuple(entry.generation_index for entry in self.entries)

    @property
    def canonical_prefix_hash(self) -> str:
        return canonical_content_hash(
            {
                "source_plan_hash": self.source_plan_hash,
                "timeline_hash": self.timeline_hash,
                "entry_hashes": tuple(
                    entry.semantic_content_hash for entry in self.entries
                ),
            },
            domain=RTC_CANONICAL_CONTINUATION_PREFIX_HASH_DOMAIN_V1,
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "source_plan_id": self.source_plan_id,
                "source_plan_generation": self.source_plan_generation,
                "source_plan_hash": self.source_plan_hash,
                "producer_id": self.producer_id,
                "producer_epoch": self.producer_epoch,
                "policy_node_id": self.policy_node_id,
                "timeline_hash": self.timeline_hash,
                "selection_spec_hash": self.selection_spec_hash,
                "prefix_horizon_steps": self.prefix_horizon_steps,
                "entry_hashes": tuple(
                    entry.semantic_content_hash for entry in self.entries
                ),
                "canonical_prefix_hash": self.canonical_prefix_hash,
            },
            domain=RTC_PLAN_CONTINUATION_SLICE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class RtcContinuationSelectionOutcome:
    """Accepted immutable slice or explicit no-prefix/bootstrap rejection."""

    accepted: bool
    source_plan_hash: str
    timeline_hash: str
    selection_spec_hash: str
    continuation_slice: RtcPlanContinuationSlice | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("source_plan_hash", "timeline_hash", "selection_spec_hash"):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        reasons = _normalized_strings(
            self.reasons,
            name="reasons",
            require_nonempty=not bool(self.accepted),
        )
        if self.accepted:
            if self.continuation_slice is None:
                raise ValueError("Accepted selection requires a continuation slice.")
            if reasons:
                raise ValueError("Accepted selection cannot carry rejection reasons.")
            if (
                self.continuation_slice.source_plan_hash
                != self.source_plan_hash
            ):
                raise ValueError("Selection/source plan hash mismatch.")
            if self.continuation_slice.timeline_hash != self.timeline_hash:
                raise ValueError("Selection/timeline hash mismatch.")
            if (
                self.continuation_slice.selection_spec_hash
                != self.selection_spec_hash
            ):
                raise ValueError("Selection/spec hash mismatch.")
        elif self.continuation_slice is not None:
            raise ValueError("Rejected selection cannot carry a continuation slice.")
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "reasons", reasons)

    @property
    def bootstrap_required(self) -> bool:
        return not self.accepted and any(
            reason
            in {
                "empty_source_plan",
                "insufficient_prefix_steps",
                "no_eligible_overlap",
            }
            for reason in self.reasons
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "accepted": self.accepted,
                "source_plan_hash": self.source_plan_hash,
                "timeline_hash": self.timeline_hash,
                "selection_spec_hash": self.selection_spec_hash,
                "continuation_slice_hash": (
                    None
                    if self.continuation_slice is None
                    else self.continuation_slice.semantic_content_hash
                ),
                "reasons": self.reasons,
            },
            domain=RTC_CONTINUATION_SELECTION_OUTCOME_HASH_DOMAIN_V1,
        )


def _selection_outcome(
    *,
    accepted: bool,
    plan: TimedPolicyActionPlanSnapshot,
    timeline: RtcContinuationTimeline,
    spec: RtcContinuationSelectionSpec,
    continuation_slice: RtcPlanContinuationSlice | None = None,
    reasons: tuple[str, ...] = (),
) -> RtcContinuationSelectionOutcome:
    return RtcContinuationSelectionOutcome(
        accepted=accepted,
        source_plan_hash=plan.semantic_content_hash,
        timeline_hash=timeline.semantic_content_hash,
        selection_spec_hash=spec.semantic_content_hash,
        continuation_slice=continuation_slice,
        reasons=reasons,
    )


def select_rtc_plan_continuation(
    plan: TimedPolicyActionPlanSnapshot,
    *,
    timeline: RtcContinuationTimeline,
    spec: RtcContinuationSelectionSpec,
) -> RtcContinuationSelectionOutcome:
    """Select exact, contiguous producer-local old-plan rows without resampling."""

    if not isinstance(plan, TimedPolicyActionPlanSnapshot):
        raise TypeError("plan must be a TimedPolicyActionPlanSnapshot.")
    if not isinstance(timeline, RtcContinuationTimeline):
        raise TypeError("timeline must be a RtcContinuationTimeline.")
    if not isinstance(spec, RtcContinuationSelectionSpec):
        raise TypeError("spec must be an RtcContinuationSelectionSpec.")
    if spec.prefix_horizon_steps > timeline.prediction_horizon_steps:
        return _selection_outcome(
            accepted=False,
            plan=plan,
            timeline=timeline,
            spec=spec,
            reasons=("prefix_horizon_exceeds_prediction_horizon",),
        )
    if spec.prefix_horizon_steps > timeline.forecast_delay_steps:
        return _selection_outcome(
            accepted=False,
            plan=plan,
            timeline=timeline,
            spec=spec,
            reasons=("prefix_horizon_exceeds_forecast_delay",),
        )
    if plan.in_flight_request_id is not None:
        return _selection_outcome(
            accepted=False,
            plan=plan,
            timeline=timeline,
            spec=spec,
            reasons=("source_plan_already_in_flight",),
        )
    if plan.clock_domain_id != timeline.clock_domain_id:
        return _selection_outcome(
            accepted=False,
            plan=plan,
            timeline=timeline,
            spec=spec,
            reasons=("clock_domain_mismatch",),
        )
    tolerance = spec.lattice_tolerance_sec
    if timeline.generation_start_at < plan.cursor_at - tolerance:
        return _selection_outcome(
            accepted=False,
            plan=plan,
            timeline=timeline,
            spec=spec,
            reasons=("generation_start_before_plan_cursor",),
        )
    if not plan.keyframes:
        return _selection_outcome(
            accepted=False,
            plan=plan,
            timeline=timeline,
            spec=spec,
            reasons=("empty_source_plan",),
        )

    entries_by_index: dict[int, RtcPlanContinuationEntry] = {}
    for keyframe in plan.keyframes:
        if keyframe.target_effective_at < timeline.generation_start_at - tolerance:
            continue
        offset = (
            keyframe.target_effective_at - timeline.generation_start_at
        ) / timeline.action_period_sec
        nearest_index = int(round(offset))
        expected_target = (
            timeline.generation_start_at
            + nearest_index * timeline.action_period_sec
        )
        if nearest_index < 0:
            continue
        if nearest_index >= spec.prefix_horizon_steps:
            break
        if not math.isclose(
            keyframe.target_effective_at,
            expected_target,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return _selection_outcome(
                accepted=False,
                plan=plan,
                timeline=timeline,
                spec=spec,
                reasons=("keyframe_off_generation_lattice",),
            )
        if not math.isclose(
            keyframe.action_period_sec,
            timeline.action_period_sec,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return _selection_outcome(
                accepted=False,
                plan=plan,
                timeline=timeline,
                spec=spec,
                reasons=("action_period_mismatch",),
            )
        if keyframe.effective_clock_domain_id != timeline.clock_domain_id:
            return _selection_outcome(
                accepted=False,
                plan=plan,
                timeline=timeline,
                spec=spec,
                reasons=("keyframe_clock_domain_mismatch",),
            )
        if (
            keyframe.action_payload_kind
            != POLICY_ACTION_PAYLOAD_TYPED_CANONICAL_V1
        ):
            return _selection_outcome(
                accepted=False,
                plan=plan,
                timeline=timeline,
                spec=spec,
                reasons=("typed_canonical_action_required",),
            )
        if nearest_index in entries_by_index:
            return _selection_outcome(
                accepted=False,
                plan=plan,
                timeline=timeline,
                spec=spec,
                reasons=("duplicate_generation_index",),
            )
        entries_by_index[nearest_index] = RtcPlanContinuationEntry(
            generation_index=nearest_index,
            keyframe=keyframe,
        )

    if not entries_by_index:
        return _selection_outcome(
            accepted=False,
            plan=plan,
            timeline=timeline,
            spec=spec,
            reasons=("no_eligible_overlap",),
        )
    indices = tuple(sorted(entries_by_index))
    if indices != tuple(
        range(indices[0], indices[-1] + 1)
    ):
        return _selection_outcome(
            accepted=False,
            plan=plan,
            timeline=timeline,
            spec=spec,
            reasons=("non_contiguous_overlap",),
        )
    if len(indices) < spec.minimum_selected_steps:
        return _selection_outcome(
            accepted=False,
            plan=plan,
            timeline=timeline,
            spec=spec,
            reasons=("insufficient_prefix_steps",),
        )
    continuation_slice = RtcPlanContinuationSlice(
        source_plan_id=plan.plan_id,
        source_plan_generation=plan.generation,
        source_plan_hash=plan.semantic_content_hash,
        producer_id=plan.producer_id,
        producer_epoch=plan.producer_epoch,
        policy_node_id=plan.policy_node_id,
        timeline_hash=timeline.semantic_content_hash,
        selection_spec_hash=spec.semantic_content_hash,
        prefix_horizon_steps=spec.prefix_horizon_steps,
        entries=tuple(entries_by_index[index] for index in indices),
    )
    return _selection_outcome(
        accepted=True,
        plan=plan,
        timeline=timeline,
        spec=spec,
        continuation_slice=continuation_slice,
    )


@dataclass(frozen=True)
class RtcContinuationEncodingReceipt:
    """Deprecated pre-C1 model-space encoding lineage.

    This frozen type remains import-compatible only.  New encoding and actual
    consumption evidence belongs to ``deployments.client``.  Do not add new
    callers or fields here.
    """

    source_plan_id: str
    source_plan_generation: int
    source_plan_hash: str
    timeline_hash: str
    prefix_source_role: str
    sampler_constraint: str
    valid_prefix_steps: int
    component_scope: tuple[str, ...]
    canonical_prefix_hash: str
    model_prefix_hash: str
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str

    def __post_init__(self) -> None:
        for name in (
            "source_plan_id",
            "source_plan_hash",
            "timeline_hash",
            "canonical_prefix_hash",
            "model_prefix_hash",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "source_plan_generation",
            _nonnegative_int(
                self.source_plan_generation,
                name="source_plan_generation",
            ),
        )
        object.__setattr__(
            self,
            "valid_prefix_steps",
            _positive_int(self.valid_prefix_steps, name="valid_prefix_steps"),
        )
        role = _nonempty(self.prefix_source_role, name="prefix_source_role")
        if role != RTC_CONTINUATION_PREFIX_SOURCE_PRODUCER_LOCAL_OVERLAP_V1:
            raise ValueError(
                "The first model-RTC contract accepts only producer-local overlap."
            )
        object.__setattr__(self, "prefix_source_role", role)
        constraint = _nonempty(
            self.sampler_constraint,
            name="sampler_constraint",
        )
        if constraint not in RTC_CONTINUATION_SAMPLER_CONSTRAINTS_V1:
            raise ValueError(f"Unsupported sampler constraint: {constraint!r}.")
        object.__setattr__(self, "sampler_constraint", constraint)
        scope = tuple(
            sorted(
                {
                    _nonempty(value, name="component_scope item")
                    for value in self.component_scope
                }
            )
        )
        if not scope:
            raise ValueError("component_scope must be non-empty.")
        object.__setattr__(self, "component_scope", scope)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "source_plan_id": self.source_plan_id,
                "source_plan_generation": self.source_plan_generation,
                "source_plan_hash": self.source_plan_hash,
                "timeline_hash": self.timeline_hash,
                "prefix_source_role": self.prefix_source_role,
                "sampler_constraint": self.sampler_constraint,
                "valid_prefix_steps": self.valid_prefix_steps,
                "component_scope": self.component_scope,
                "canonical_prefix_hash": self.canonical_prefix_hash,
                "model_prefix_hash": self.model_prefix_hash,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
            },
            domain=RTC_CONTINUATION_ENCODING_RECEIPT_HASH_DOMAIN_V1,
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
    "RtcContinuationEncodingReceipt",
    "RtcContinuationSelectionOutcome",
    "RtcContinuationSelectionSpec",
    "RtcContinuationTimeline",
    "RtcPlanContinuationEntry",
    "RtcPlanContinuationSlice",
    "rtc_continuation_timeline_from_request",
    "select_rtc_plan_continuation",
]
