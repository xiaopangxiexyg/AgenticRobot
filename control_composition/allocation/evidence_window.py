"""Causal time-window assembly for composition allocation evidence.

This module retains admitted source facts and turns one exact control cut into
an :class:`AllocationPublicationBasis`.  It does not choose an allocation
mode, publish context, request HOLD, or own downstream execution.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction
from robot_motion.runtime.control_admission import ControlCut

from ..activation import CompositionActivationScope
from ..residual_anchor import ResidualAnchorReceipt
from .contracts import CompositionAllocationSpec
from .metrics import (
    ActivityMetricProvider,
    ResidualMagnitudeMetricProvider,
    SourceActionActivitySample,
    build_component_residual_magnitude_observation,
    build_component_source_activity_observation,
)
from .publication_contracts import AllocationPublicationBasis


ALLOCATION_EVIDENCE_SAMPLE_PAIR_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.allocation_evidence_sample_pair.development"
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


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _normalization_map(
    value: Mapping[str, Mapping[str, float]],
    *,
    component_ids: tuple[str, ...],
    name: str,
) -> Mapping[str, Mapping[str, float]]:
    raw = dict(value)
    if set(raw) != set(component_ids):
        raise ValueError(f"{name} must cover every allocated component exactly.")
    normalized: dict[str, Mapping[str, float]] = {}
    for component_id in component_ids:
        metrics = {
            _required(metric, name=f"{name}.metric"): _positive(
                scale,
                name=f"{name}[{component_id!r}][{metric!r}]",
            )
            for metric, scale in dict(raw[component_id]).items()
        }
        if not metrics:
            raise ValueError(f"{name}[{component_id!r}] must not be empty.")
        normalized[component_id] = MappingProxyType(dict(sorted(metrics.items())))
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class AllocationEvidenceSamplePair:
    """One admitted absolute/residual view of the same physical source sample.

    ``last_control_admission_sequence`` is the atomic batch's final sequence.
    A control cut before that sequence therefore sees neither role.
    """

    runtime_session_id: str
    control_domain_id: str
    clock_domain_id: str
    clock_reset_epoch: int
    sampled_at: float
    available_at: float
    last_control_admission_sequence: int
    source_sample_id: str
    source_sample_hash: str
    absolute_action: ComponentAction
    residual_action: ComponentAction
    anchor_receipt_id: str
    anchor_receipt_hash: str
    anchor_request_generation: int
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "runtime_session_id",
                "control_domain_id",
                "clock_domain_id",
                "source_sample_id",
                "source_sample_hash",
                "anchor_receipt_id",
                "anchor_receipt_hash",
            )
        }
        reset_epoch = _nonnegative_int(
            self.clock_reset_epoch,
            name="clock_reset_epoch",
        )
        admission_sequence = _nonnegative_int(
            self.last_control_admission_sequence,
            name="last_control_admission_sequence",
        )
        if admission_sequence == 0:
            raise ValueError("last_control_admission_sequence must be positive.")
        sampled_at = _finite(self.sampled_at, name="sampled_at")
        available_at = _finite(self.available_at, name="available_at")
        if sampled_at > available_at:
            raise ValueError("sampled_at cannot follow receiver availability.")
        if not isinstance(self.absolute_action, ComponentAction):
            raise TypeError("absolute_action must be ComponentAction.")
        if not isinstance(self.residual_action, ComponentAction):
            raise TypeError("residual_action must be ComponentAction.")
        generation = _nonnegative_int(
            self.anchor_request_generation,
            name="anchor_request_generation",
        )
        payload = {
            **text,
            "clock_reset_epoch": reset_epoch,
            "sampled_at": sampled_at,
            "available_at": available_at,
            "last_control_admission_sequence": admission_sequence,
            "absolute_action_hash": self.absolute_action.semantic_content_hash,
            "residual_action_hash": self.residual_action.semantic_content_hash,
            "anchor_request_generation": generation,
        }
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "clock_reset_epoch", reset_epoch)
        object.__setattr__(self, "sampled_at", sampled_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(
            self,
            "last_control_admission_sequence",
            admission_sequence,
        )
        object.__setattr__(self, "anchor_request_generation", generation)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    ALLOCATION_EVIDENCE_SAMPLE_PAIR_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def pair_id(self) -> str:
        return f"allocation-evidence-sample-pair:{self.semantic_content_hash}"


@dataclass(frozen=True)
class AllocationEvidenceWindowSnapshot:
    anchor_receipt_id: str | None
    anchor_receipt_hash: str | None
    anchor_request_generation: int | None
    retained_pair_count: int
    dropped_pair_count: int
    latest_sampled_at: float | None
    latest_control_admission_sequence: int | None


class AllocationEvidenceWindow:
    """Retain causal sample pairs and build evidence without choosing a mode."""

    def __init__(
        self,
        *,
        allocation_spec: CompositionAllocationSpec,
        window_duration_sec: float,
        evidence_ttl_sec: float,
        activity_normalization_scale_by_component: Mapping[
            str, Mapping[str, float]
        ],
        residual_normalization_scale_by_component: Mapping[
            str, Mapping[str, float]
        ],
        activity_metric_provider_by_component: Mapping[
            str, ActivityMetricProvider
        ] | None = None,
        activity_metric_provider_identity_by_component: Mapping[
            str, tuple[str, str]
        ] | None = None,
        residual_metric_provider_by_component: Mapping[
            str, ResidualMagnitudeMetricProvider
        ] | None = None,
        residual_metric_provider_identity_by_component: Mapping[
            str, tuple[str, str]
        ] | None = None,
    ) -> None:
        if not isinstance(allocation_spec, CompositionAllocationSpec):
            raise TypeError("allocation_spec must be CompositionAllocationSpec.")
        self.allocation_spec = allocation_spec
        self.window_duration_sec = _positive(
            window_duration_sec,
            name="window_duration_sec",
        )
        self.evidence_ttl_sec = _positive(
            evidence_ttl_sec,
            name="evidence_ttl_sec",
        )
        component_ids = allocation_spec.component_ids
        self.activity_normalization_scale_by_component = _normalization_map(
            activity_normalization_scale_by_component,
            component_ids=component_ids,
            name="activity_normalization_scale_by_component",
        )
        self.residual_normalization_scale_by_component = _normalization_map(
            residual_normalization_scale_by_component,
            component_ids=component_ids,
            name="residual_normalization_scale_by_component",
        )
        self._activity_providers = self._provider_maps(
            activity_metric_provider_by_component,
            activity_metric_provider_identity_by_component,
            name="activity_metric_provider",
        )
        self._residual_providers = self._provider_maps(
            residual_metric_provider_by_component,
            residual_metric_provider_identity_by_component,
            name="residual_metric_provider",
        )
        self._anchor_receipt: ResidualAnchorReceipt | None = None
        self._pairs: list[AllocationEvidenceSamplePair] = []
        self._pair_hash_by_id: dict[str, str] = {}
        self._dropped_pair_count = 0
        self._lock = threading.RLock()

    def _provider_maps(
        self,
        providers: Mapping[str, object] | None,
        identities: Mapping[str, tuple[str, str]] | None,
        *,
        name: str,
    ) -> Mapping[str, tuple[object, str, str]]:
        provider_map = dict(providers or {})
        identity_map = dict(identities or {})
        if set(provider_map) != set(identity_map):
            raise ValueError(
                f"{name} providers and replayable identities must cover "
                "the same components."
            )
        unknown = set(provider_map) - set(self.allocation_spec.component_ids)
        if unknown:
            raise ValueError(f"{name} contains undeclared components: {unknown!r}.")
        normalized = {}
        for component_id, provider in provider_map.items():
            if not callable(provider):
                raise TypeError(f"{name}[{component_id!r}] must be callable.")
            provider_id, provider_hash = identity_map[component_id]
            normalized[component_id] = (
                provider,
                _required(provider_id, name=f"{name}_id"),
                _required(provider_hash, name=f"{name}_hash"),
            )
        return MappingProxyType(normalized)

    def bind_anchor(self, receipt: ResidualAnchorReceipt) -> None:
        """Bind a new anchor and atomically clear all pre-rebase influence."""

        if not isinstance(receipt, ResidualAnchorReceipt):
            raise TypeError("receipt must be ResidualAnchorReceipt.")
        if receipt.source_role != self.allocation_spec.relative_overlay_role:
            raise ValueError("Anchor source role differs from the allocation spec.")
        if set(receipt.component_binding_by_id) != set(
            self.allocation_spec.component_ids
        ):
            raise ValueError("Anchor must cover allocated components exactly.")
        with self._lock:
            self._anchor_receipt = receipt
            self._pairs.clear()
            self._pair_hash_by_id.clear()

    def clear(self) -> None:
        with self._lock:
            self._anchor_receipt = None
            self._pairs.clear()
            self._pair_hash_by_id.clear()

    def _validate_pair(
        self,
        pair: AllocationEvidenceSamplePair,
        receipt: ResidualAnchorReceipt,
    ) -> None:
        if pair.clock_domain_id != receipt.clock_domain_id:
            raise ValueError("Evidence pair and anchor use another mapped clock.")
        if (
            pair.anchor_receipt_id != receipt.receipt_id
            or pair.anchor_receipt_hash != receipt.semantic_content_hash
            or pair.anchor_request_generation != receipt.request_generation
        ):
            raise ValueError("Evidence pair belongs to another residual anchor.")
        if pair.absolute_action.action_schema.semantic_content_hash != (
            receipt.absolute_action_schema_hash
        ):
            raise ValueError("Evidence absolute action uses another anchor schema.")
        if pair.residual_action.action_schema.semantic_content_hash != (
            receipt.relative_action_schema_hash
        ):
            raise ValueError("Evidence residual action uses another anchor schema.")
        components = set(self.allocation_spec.component_ids)
        if not components.issubset(pair.absolute_action.component_coverage):
            raise ValueError("Evidence absolute action lacks allocated components.")
        if not components.issubset(pair.residual_action.component_coverage):
            raise ValueError("Evidence residual action lacks allocated components.")
        if pair.sampled_at < receipt.captured_at:
            raise ValueError("Evidence pair predates its active residual anchor.")

    def offer(self, pair: AllocationEvidenceSamplePair) -> bool:
        """Retain one newer pair; exact duplicate offers are idempotent."""

        if not isinstance(pair, AllocationEvidenceSamplePair):
            raise TypeError("pair must be AllocationEvidenceSamplePair.")
        with self._lock:
            receipt = self._anchor_receipt
            if receipt is None:
                raise RuntimeError("Allocation evidence has no active anchor.")
            self._validate_pair(pair, receipt)
            existing_hash = self._pair_hash_by_id.get(pair.pair_id)
            if existing_hash is not None:
                if existing_hash != pair.semantic_content_hash:
                    raise ValueError("Evidence pair ID was reused with another hash.")
                return False
            if self._pairs:
                previous = self._pairs[-1]
                if pair.runtime_session_id != previous.runtime_session_id:
                    raise ValueError("Evidence runtime session changed.")
                if pair.control_domain_id != previous.control_domain_id:
                    raise ValueError("Evidence control domain changed.")
                if pair.clock_reset_epoch != previous.clock_reset_epoch:
                    raise ValueError("Evidence clock reset epoch changed.")
                if pair.sampled_at <= previous.sampled_at:
                    raise ValueError("Evidence sample time must increase strictly.")
                if pair.available_at < previous.available_at:
                    raise ValueError("Evidence receiver availability moved backward.")
                if pair.last_control_admission_sequence <= (
                    previous.last_control_admission_sequence
                ):
                    raise ValueError("Evidence admission sequence must increase.")
            self._pairs.append(pair)
            self._pair_hash_by_id[pair.pair_id] = pair.semantic_content_hash
            self._prune_locked(reference_at=pair.sampled_at)
            return True

    def _prune_locked(self, *, reference_at: float) -> None:
        retain_from = reference_at - self.window_duration_sec
        first_retained = 0
        while (
            first_retained < len(self._pairs)
            and self._pairs[first_retained].sampled_at < retain_from
        ):
            first_retained += 1
        if not first_retained:
            return
        removed = self._pairs[:first_retained]
        self._pairs = self._pairs[first_retained:]
        for pair in removed:
            self._pair_hash_by_id.pop(pair.pair_id, None)
        self._dropped_pair_count += len(removed)

    def snapshot(self) -> AllocationEvidenceWindowSnapshot:
        with self._lock:
            receipt = self._anchor_receipt
            latest = self._pairs[-1] if self._pairs else None
            return AllocationEvidenceWindowSnapshot(
                anchor_receipt_id=None if receipt is None else receipt.receipt_id,
                anchor_receipt_hash=(
                    None if receipt is None else receipt.semantic_content_hash
                ),
                anchor_request_generation=(
                    None if receipt is None else receipt.request_generation
                ),
                retained_pair_count=len(self._pairs),
                dropped_pair_count=self._dropped_pair_count,
                latest_sampled_at=None if latest is None else latest.sampled_at,
                latest_control_admission_sequence=(
                    None
                    if latest is None
                    else latest.last_control_admission_sequence
                ),
            )

    def capture_basis(
        self,
        *,
        allocation_scope_id: str,
        control_cut: ControlCut,
        activation_scope: CompositionActivationScope,
        requested_effective_from: float,
    ) -> AllocationPublicationBasis | None:
        """Build facts visible at one exact cut, or return no current basis."""

        if not isinstance(control_cut, ControlCut):
            raise TypeError("control_cut must be ControlCut.")
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be CompositionActivationScope.")
        requested = _finite(
            requested_effective_from,
            name="requested_effective_from",
        )
        if requested < control_cut.cutoff_at:
            raise ValueError("requested_effective_from cannot predate the cut.")
        with self._lock:
            receipt = self._anchor_receipt
            if receipt is None:
                return None
            if (
                receipt.composition_activation_id != activation_scope.activation_id
                or receipt.composition_activation_hash
                != activation_scope.semantic_content_hash
            ):
                raise ValueError("Active anchor belongs to another activation.")
            retain_from = control_cut.cutoff_at - self.window_duration_sec
            pairs = tuple(
                pair
                for pair in self._pairs
                if pair.runtime_session_id == control_cut.runtime_session_id
                and pair.control_domain_id == control_cut.control_domain_id
                and pair.clock_domain_id == control_cut.clock_domain_id
                and pair.clock_reset_epoch == control_cut.clock_reset_epoch
                and retain_from <= pair.sampled_at <= control_cut.cutoff_at
                and pair.available_at <= control_cut.cutoff_at
                and pair.last_control_admission_sequence
                <= control_cut.cutoff_admission_sequence
            )
            if len(pairs) < 2:
                return None
            support_until = pairs[-1].sampled_at + self.evidence_ttl_sec
            if requested >= support_until:
                return None
            activity_samples = tuple(
                SourceActionActivitySample(
                    sampled_at=pair.sampled_at,
                    clock_domain_id=pair.clock_domain_id,
                    action=pair.absolute_action,
                    source_fact_id=pair.pair_id,
                    source_fact_hash=pair.semantic_content_hash,
                )
                for pair in pairs
            )
            latest_pair = pairs[-1]
            residual_sample = SourceActionActivitySample(
                sampled_at=latest_pair.sampled_at,
                clock_domain_id=latest_pair.clock_domain_id,
                action=latest_pair.residual_action,
                source_fact_id=latest_pair.pair_id,
                source_fact_hash=latest_pair.semantic_content_hash,
            )
            activity = {}
            magnitude = {}
            for policy in self.allocation_spec.component_policies:
                component_id = policy.component_id
                activity_provider = self._activity_providers.get(component_id)
                residual_provider = self._residual_providers.get(component_id)
                activity[component_id] = (
                    build_component_source_activity_observation(
                        activity_samples,
                        source_role=policy.motion_source_role,
                        component_id=component_id,
                        available_at=control_cut.cutoff_at,
                        valid_until=support_until,
                        normalization_scale_by_metric=(
                            self.activity_normalization_scale_by_component[
                                component_id
                            ]
                        ),
                        metric_provider=(
                            None
                            if activity_provider is None
                            else activity_provider[0]
                        ),
                        metric_provider_id=(
                            None
                            if activity_provider is None
                            else activity_provider[1]
                        ),
                        metric_provider_hash=(
                            None
                            if activity_provider is None
                            else activity_provider[2]
                        ),
                    )
                )
                magnitude[component_id] = (
                    build_component_residual_magnitude_observation(
                        residual_sample,
                        source_role=policy.residual_source_role,
                        component_id=component_id,
                        anchor_receipt=receipt,
                        available_at=control_cut.cutoff_at,
                        valid_until=support_until,
                        normalization_scale_by_metric=(
                            self.residual_normalization_scale_by_component[
                                component_id
                            ]
                        ),
                        metric_provider=(
                            None
                            if residual_provider is None
                            else residual_provider[0]
                        ),
                        metric_provider_id=(
                            None
                            if residual_provider is None
                            else residual_provider[1]
                        ),
                        metric_provider_hash=(
                            None
                            if residual_provider is None
                            else residual_provider[2]
                        ),
                    )
                )
            return AllocationPublicationBasis(
                allocation_scope_id=_required(
                    allocation_scope_id,
                    name="allocation_scope_id",
                ),
                control_cut=control_cut,
                activation_scope=activation_scope,
                activity_by_component=activity,
                residual_magnitude_by_component=magnitude,
                requested_effective_from=requested,
                support_until=support_until,
            )


__all__ = [
    "ALLOCATION_EVIDENCE_SAMPLE_PAIR_HASH_DOMAIN_DEVELOPMENT",
    "AllocationEvidenceSamplePair",
    "AllocationEvidenceWindow",
    "AllocationEvidenceWindowSnapshot",
]
