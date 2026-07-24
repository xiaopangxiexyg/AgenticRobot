"""Passive human-emission ingress and explicit takeover-readiness contract.

The ingress consumes an emission already produced by the lower teleoperation
runtime.  It owns only a bounded passive queue: no tracker, retargeter, device,
worker, deadman state, or publication authority.  In particular, observing a
fresh emission does not prove deadman re-arm or retargeter rebase.  A concrete
readiness provider must bind those independent lower-runtime receipts before
it can produce :class:`HumanTakeoverReadinessCandidate`.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.proposal import ProposalAdmissionReceipt
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.observation_cut import (
    ObservationBundle,
    ObservationCutRecord,
    ObservationSampleRef,
    ReceiverObservationCutBuilder,
)
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
    HumanProposalAdapterResult,
)
from teleop_runtime.control_trace.projection import (
    project_human_proposal_adapter_result,
)

from control_composition.lifecycle import (
    HumanTakeoverOperationSpec,
    HumanTakeoverState,
)


HUMAN_TAKEOVER_READINESS_CANDIDATE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.human_takeover_readiness_candidate.v1"
)


@runtime_checkable
class HumanProposalAdmissionCore(Protocol):
    """Minimal authoritative-core seam shared by human source profiles."""

    def record_external_facts(self, facts: tuple[Any, ...]) -> object: ...

    def admit_proposal(
        self,
        draft: object,
        *,
        source_event_id: str,
    ) -> ProposalAdmissionReceipt: ...


@runtime_checkable
class HumanObservationCutAdmissionCore(Protocol):
    """Receiver-side cut admission seam; independent from proposal ownership."""

    coordinator: ControlAdmissionCoordinator

    def record_observation_cut(self, record: ObservationCutRecord) -> object: ...


def admit_human_observation_cut(
    *,
    core: HumanObservationCutAdmissionCore,
    builder: ReceiverObservationCutBuilder,
    emissions: tuple[HumanCommandEmission, ...],
    payload_handles_by_stream: Mapping[str, Any],
    trigger_reason: str = "human_command_emission",
    built_at: float | None = None,
    context_dependencies: tuple[str, ...] = (),
) -> ObservationBundle:
    """Admit one causal input cut shared by one or more human proposals.

    Independent arm/hand emissions may reference subsets of the same cut.  The
    union is recorded exactly once, while every later proposal retains its own
    emission identity and selection-group coverage.
    """

    if not isinstance(core, HumanObservationCutAdmissionCore):
        raise TypeError("core does not satisfy HumanObservationCutAdmissionCore.")
    if not isinstance(builder, ReceiverObservationCutBuilder):
        raise TypeError("builder must be a ReceiverObservationCutBuilder.")
    normalized_emissions = tuple(emissions)
    if not normalized_emissions:
        raise ValueError("At least one human emission is required.")
    selected: dict[str, ObservationSampleRef] = {}
    for emission in normalized_emissions:
        if not isinstance(emission, HumanCommandEmission):
            raise TypeError("emissions must contain HumanCommandEmission values.")
        if not emission.observation_sample_refs:
            raise ValueError(
                "A receiver-admitted human cut requires source sample refs on "
                "every emission."
            )
        for stream_id, sample in emission.observation_sample_refs.items():
            existing = selected.get(stream_id)
            if existing is not None and existing != sample:
                raise ValueError(
                    f"Human emissions disagree on source sample {stream_id!r}."
                )
            selected[stream_id] = sample
    bundle = builder.capture(
        core.coordinator,
        selected_source_samples=selected,
        payload_handles_by_stream=payload_handles_by_stream,
        trigger_reason=trigger_reason,
        built_at=built_at,
        context_dependencies=context_dependencies,
    )
    core.record_observation_cut(bundle.record)
    return bundle


def admit_human_command_emission(
    *,
    core: HumanProposalAdmissionCore,
    adapter: HumanProposalAdapter,
    emission: HumanCommandEmission,
    effective_from: float,
    effective_until: float,
    target_effective_at: float,
    occurred_at: float,
    clock_domain_id: str,
    adaptation: HumanProposalAdapterResult | None = None,
    observation_cut: ObservationCutRecord | None = None,
) -> tuple[HumanProposalAdapterResult, ProposalAdmissionReceipt | None]:
    """Adapt, trace, and receiver-admit one human emission exactly once.

    The seam owns no queue, authority lease, safety stage, or dispatch right.
    Repeated submission is rejected by the existing ProposalBuffer identity,
    so this layer does not introduce another deduplication owner.
    """

    if not isinstance(adapter, HumanProposalAdapter):
        raise TypeError("adapter must be a HumanProposalAdapter.")
    if not isinstance(emission, HumanCommandEmission):
        raise TypeError("emission must be a HumanCommandEmission.")
    if not isinstance(core, HumanProposalAdmissionCore):
        raise TypeError("core does not satisfy HumanProposalAdmissionCore.")
    observation_cut_id: str | None = None
    if observation_cut is not None:
        if not isinstance(observation_cut, ObservationCutRecord):
            raise TypeError("observation_cut must be an ObservationCutRecord.")
        config = adapter.config
        if (
            observation_cut.runtime_session_id != config.runtime_session_id
            or observation_cut.control_domain_id != config.control_domain_id
            or observation_cut.policy_node_id != config.producer_id
            or observation_cut.policy_node_epoch != config.producer_epoch
        ):
            raise ValueError(
                "Human observation cut belongs to another producer/session."
            )
        if not emission.observation_sample_refs:
            raise ValueError(
                "An emission linked to an observation cut must carry source "
                "sample refs."
            )
        for stream_id, source_sample in emission.observation_sample_refs.items():
            admitted_sample = observation_cut.selected_sample_by_stream.get(
                stream_id
            )
            if admitted_sample is None:
                raise ValueError(
                    f"Human observation cut does not contain {stream_id!r}."
                )
            if (
                admitted_sample.sample_id != source_sample.sample_id
                or admitted_sample.payload_object_id
                != source_sample.payload_object_id
            ):
                raise ValueError(
                    "Human observation cut sample identity/content does not "
                    "match the emission."
                )
        observation_cut_id = observation_cut.observation_cut_id
    effective_start = _finite(effective_from, field_name="effective_from")
    effective_end = _finite(effective_until, field_name="effective_until")
    target_time = _finite(
        target_effective_at,
        field_name="target_effective_at",
    )
    if effective_end <= effective_start:
        raise ValueError("effective_until must be later than effective_from.")
    if not effective_start <= target_time < effective_end:
        raise ValueError("target_effective_at must be inside the effective interval.")
    if adaptation is None:
        result = adapter.adapt(
            emission,
            effective_from=effective_start,
            effective_until=effective_end,
            target_effective_at=target_time,
            observation_cut_id=observation_cut_id,
        )
    else:
        if not isinstance(adaptation, HumanProposalAdapterResult):
            raise TypeError("adaptation must be a HumanProposalAdapterResult.")
        receipt = adaptation.source_adapter_receipt
        if receipt.source_emission_semantic_hash != emission.semantic_content_hash:
            raise ValueError("Adaptation belongs to another human emission.")
        draft = adaptation.proposal_draft
        if draft is not None and observation_cut_id is not None:
            if draft.observation_cut_id not in {None, observation_cut_id}:
                raise ValueError("Adaptation belongs to another observation cut.")
            audit_metadata = dict(draft.audit_metadata)
            audit_metadata["source_observation_cut_id"] = observation_cut_id
            draft = replace(
                draft,
                observation_cut_id=observation_cut_id,
                audit_metadata=audit_metadata,
            )
            result = replace(adaptation, proposal_draft=draft)
        if draft is not None and (
            float(draft.effective_from) != effective_start
            or float(draft.effective_until) != effective_end
            or float(draft.target_effective_at) != target_time
        ):
            raise ValueError("Adaptation validity interval does not match submission.")
        result = adaptation
    config = adapter.config
    facts = project_human_proposal_adapter_result(
        emission,
        result,
        runtime_session_id=config.runtime_session_id,
        control_domain_id=config.control_domain_id,
        embodiment_instance_id=config.embodiment_instance_id,
        occurred_at=_finite(occurred_at, field_name="occurred_at"),
        clock_domain_id=_required(
            clock_domain_id,
            field_name="clock_domain_id",
        ),
    )
    core.record_external_facts(facts)
    receipt = None
    if (
        result.proposal_draft is not None
        and result.source_adapter_receipt.authoritative_eligible
    ):
        receipt = core.admit_proposal(
            result.proposal_draft,
            source_event_id=result.source_adapter_receipt.source_emission_id,
        )
    return result, receipt


@runtime_checkable
class HumanPoseSourceContract(Protocol):
    """Structural source contract shared by synthetic and live pose inputs."""

    source_id: str
    reference_frame_by_effector: tuple[tuple[str, str], ...]
    pose_frame_contract_hash: str
    action_spec_hash_by_effector: tuple[tuple[str, str], ...]
    units_convention_id: str

    @property
    def semantic_content_hash(self) -> str: ...


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


@dataclass(frozen=True)
class HumanTakeoverReadinessCandidate:
    observer_binding_id: str
    emission: HumanCommandEmission
    effective_from: float
    effective_until: float
    target_effective_at: float
    deadman_rearm_event_id: str
    deadman_rearmed_at: float
    retargeter_rebase_receipt_id: str
    retargeter_rebased_at: float

    def __post_init__(self) -> None:
        for field_name in (
            "observer_binding_id",
            "deadman_rearm_event_id",
            "retargeter_rebase_receipt_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if not isinstance(self.emission, HumanCommandEmission):
            raise TypeError("emission must be a HumanCommandEmission.")
        for field_name in (
            "effective_from",
            "effective_until",
            "target_effective_at",
            "deadman_rearmed_at",
            "retargeter_rebased_at",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), field_name=field_name),
            )
        if self.effective_until <= self.effective_from:
            raise ValueError("effective_until must be later than effective_from.")
        if not self.effective_from <= self.target_effective_at < self.effective_until:
            raise ValueError("target_effective_at must be inside the effective interval.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "observer_binding_id": self.observer_binding_id,
                "emission_hash": self.emission.semantic_content_hash,
                "effective_from": self.effective_from,
                "effective_until": self.effective_until,
                "target_effective_at": self.target_effective_at,
                "deadman_rearm_event_id": self.deadman_rearm_event_id,
                "deadman_rearmed_at": self.deadman_rearmed_at,
                "retargeter_rebase_receipt_id": (
                    self.retargeter_rebase_receipt_id
                ),
                "retargeter_rebased_at": self.retargeter_rebased_at,
            },
            domain=HUMAN_TAKEOVER_READINESS_CANDIDATE_HASH_DOMAIN_V1,
        )


class HumanTakeoverReadinessProvider(Protocol):
    """Bounded adapter over a lower-owned human-emission observer."""

    @property
    def observer_binding_id(self) -> str: ...

    def acquire(
        self,
        *,
        operation_spec: HumanTakeoverOperationSpec,
        takeover_state: HumanTakeoverState,
        timeout_sec: float,
    ) -> HumanTakeoverReadinessCandidate: ...


@dataclass(frozen=True)
class PassiveHumanCommandEmissionIngressStatus:
    observer_binding_id: str
    queue_size: int
    queue_capacity: int
    observed_count: int
    accepted_count: int
    evicted_count: int
    stale_count: int
    acquired_count: int


class PassiveHumanCommandEmissionIngress:
    """Bounded latest-preserving input for a lower-owned observer callback.

    ``observe_emission`` is nonblocking and preserves the exact immutable
    object.  When full, the oldest passive observation is evicted so a future
    readiness wait can still see the newest command.  Waiting is intended for
    the Station side-effect worker, never the teleoperation control loop.

    This class deliberately does not implement
    :class:`HumanTakeoverReadinessProvider`: emission freshness alone cannot
    satisfy the deadman/rebase gates.
    """

    nonblocking_bounded = True

    def __init__(
        self,
        observer_binding_id: str,
        *,
        queue_capacity: int = 8,
    ) -> None:
        self.observer_binding_id = _required(
            observer_binding_id,
            field_name="observer_binding_id",
        )
        capacity = int(queue_capacity)
        if capacity <= 0:
            raise ValueError("Human emission ingress queue_capacity must be positive.")
        self._queue: queue.Queue[HumanCommandEmission] = queue.Queue(
            maxsize=capacity
        )
        self._lock = threading.RLock()
        self._observed_count = 0
        self._accepted_count = 0
        self._evicted_count = 0
        self._stale_count = 0
        self._acquired_count = 0

    def observe_emission(self, emission: HumanCommandEmission) -> bool:
        """Offer one exact emission without propagating queue pressure."""

        if not isinstance(emission, HumanCommandEmission):
            raise TypeError("Human emission ingress accepts HumanCommandEmission only.")
        with self._lock:
            self._observed_count += 1
        try:
            self._queue.put_nowait(emission)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                # A concurrent waiter freed the slot between Full and get.
                pass
            else:
                self._queue.task_done()
                with self._lock:
                    self._evicted_count += 1
            try:
                self._queue.put_nowait(emission)
            except queue.Full:
                # Another producer won the bounded slot.  This observation is
                # best-effort and must never delay the lower control loop.
                return False
        with self._lock:
            self._accepted_count += 1
        return True

    def wait_for_post_boundary_emission(
        self,
        *,
        boundary_at: float,
        timeout_sec: float,
    ) -> HumanCommandEmission:
        """Return the first queued emission at/after an explicit time boundary."""

        boundary = _finite(boundary_at, field_name="boundary_at")
        timeout = _finite(timeout_sec, field_name="timeout_sec")
        if timeout <= 0.0:
            raise ValueError("timeout_sec must be positive.")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("Timed out waiting for a post-boundary human emission.")
            try:
                emission = self._queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    "Timed out waiting for a post-boundary human emission."
                ) from exc
            try:
                if emission.emitted_at < boundary:
                    with self._lock:
                        self._stale_count += 1
                    continue
                with self._lock:
                    self._acquired_count += 1
                return emission
            finally:
                self._queue.task_done()

    def status(self) -> PassiveHumanCommandEmissionIngressStatus:
        with self._lock:
            return PassiveHumanCommandEmissionIngressStatus(
                observer_binding_id=self.observer_binding_id,
                queue_size=self._queue.qsize(),
                queue_capacity=self._queue.maxsize,
                observed_count=self._observed_count,
                accepted_count=self._accepted_count,
                evicted_count=self._evicted_count,
                stale_count=self._stale_count,
                acquired_count=self._acquired_count,
            )


@dataclass
class _HumanObserverCounters:
    offer_count: int = 0
    rejection_count: int = 0
    error_count: int = 0
    last_error: str = ""


class PassiveHumanCommandObserverFanout:
    """Exception-isolated callback fanout assembled above ``TeleopSession``.

    Subscribers must themselves be passive and nonblocking.  The fanout does
    not create a second human source or alter the canonical teleop callback;
    it only lets an application root combine existing observer sinks.
    """

    def __init__(
        self,
        subscribers: Mapping[
            str,
            Callable[[HumanCommandEmission], object],
        ],
    ) -> None:
        normalized: dict[str, Callable[[HumanCommandEmission], object]] = {}
        for subscriber_id, callback in subscribers.items():
            name = _required(subscriber_id, field_name="subscriber_id")
            if name in normalized:
                raise ValueError(f"Duplicate human observer subscriber_id: {name!r}.")
            if not callable(callback):
                raise TypeError(f"Human observer subscriber {name!r} is not callable.")
            normalized[name] = callback
        if not normalized:
            raise ValueError("Human observer fanout requires at least one subscriber.")
        self._subscribers = normalized
        self._counters = {
            subscriber_id: _HumanObserverCounters()
            for subscriber_id in normalized
        }
        self._lock = threading.RLock()
        self._input_offer_count = 0

    def observe_emission(self, emission: HumanCommandEmission) -> bool:
        if not isinstance(emission, HumanCommandEmission):
            raise TypeError("Human observer fanout accepts HumanCommandEmission only.")
        with self._lock:
            self._input_offer_count += 1
            subscribers = tuple(self._subscribers.items())
        all_accepted = True
        for subscriber_id, callback in subscribers:
            try:
                accepted = callback(emission)
                with self._lock:
                    counters = self._counters[subscriber_id]
                    counters.offer_count += 1
                    if accepted is False:
                        counters.rejection_count += 1
                        all_accepted = False
            except Exception as exc:  # passive subscriber failure is isolated
                all_accepted = False
                with self._lock:
                    counters = self._counters[subscriber_id]
                    counters.error_count += 1
                    counters.last_error = f"{type(exc).__name__}: {exc}"
        return all_accepted

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "authoritative": False,
                "input_offer_count": self._input_offer_count,
                "subscriber_count": len(self._subscribers),
                "subscribers": {
                    subscriber_id: {
                        "offer_count": counters.offer_count,
                        "rejection_count": counters.rejection_count,
                        "error_count": counters.error_count,
                        "last_error": counters.last_error,
                    }
                    for subscriber_id, counters in self._counters.items()
                },
            }


__all__ = [
    "HUMAN_TAKEOVER_READINESS_CANDIDATE_HASH_DOMAIN_V1",
    "HumanProposalAdmissionCore",
    "HumanObservationCutAdmissionCore",
    "HumanTakeoverReadinessCandidate",
    "HumanTakeoverReadinessProvider",
    "HumanPoseSourceContract",
    "PassiveHumanCommandEmissionIngress",
    "PassiveHumanCommandEmissionIngressStatus",
    "PassiveHumanCommandObserverFanout",
    "admit_human_command_emission",
    "admit_human_observation_cut",
]
