"""Bounded passive fanout for the lower-owned RTC frontend step seam.

The deploy control loop still calls ``RtcPolicyShadowFrontend.step()`` exactly
once.  This adapter only offers that same immutable result to explicitly
bounded subscriber queues.  It owns no worker, timed plan, inference request,
due-pop, emission commit, legacy action execution, or motion authority.
"""

from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.deploy import (
    RTC_POLICY_EMISSION_SOURCE_BINDING_HASH_DOMAIN_V1,
    RTC_POLICY_RUNTIME_SOURCE_CONFIG_HASH_DOMAIN_V1,
    PolicyActionEmissionCommitReceipt,
    PolicyActionEmission,
    RtcPolicyEmissionSourceBinding,
    RtcProducerActivationReadinessReceipt,
    build_rtc_policy_emission_source_binding,
)

from .rtc_frontend import RtcFrontendStep, RtcPolicyShadowFrontend


# Import-compatible constant name for existing passive/debug callers.
RTC_FRONTEND_STEP_SOURCE_BINDING_HASH_DOMAIN_V1 = (
    RTC_POLICY_EMISSION_SOURCE_BINDING_HASH_DOMAIN_V1
)
RTC_POLICY_EMISSION_PUBLICATION_KEY_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_policy_emission_publication_key.v1"
)
RTC_POLICY_EMISSION_PUBLICATION_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_policy_emission_publication_receipt.v1"
)
RTC_POLICY_EMISSION_BINDING_ACK_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_policy_emission_binding_ack.v1"
)


# Import-compatible name for existing passive/debug callers.  The single
# implementation is emission-oriented; full-step validation is compatibility
# observability only.
RtcFrontendStepSourceBinding = RtcPolicyEmissionSourceBinding


def build_rtc_frontend_step_source_binding(
    frontend: RtcPolicyShadowFrontend,
) -> RtcPolicyEmissionSourceBinding:
    """Compatibility name for the canonical emission-source binding."""

    return build_rtc_policy_emission_source_binding(frontend)


@dataclass(frozen=True)
class RtcFrontendStepSubscriberStatus:
    subscriber_id: str
    queue_size: int
    queue_capacity: int
    offer_count: int
    drop_count: int
    error_count: int
    last_error: str


@dataclass
class _SubscriberCounters:
    offer_count: int = 0
    drop_count: int = 0
    error_count: int = 0
    last_error: str = ""


@dataclass(frozen=True)
class PolicyActionEmissionDeliveryOutcome:
    """One bounded delivery result, classified by subscriber role."""

    subscriber_id: str
    subscriber_role: str
    accepted: bool
    dropped: bool
    reason: str

    def to_payload(self) -> dict[str, object]:
        return {
            "subscriber_id": self.subscriber_id,
            "subscriber_role": self.subscriber_role,
            "accepted": self.accepted,
            "dropped": self.dropped,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PolicyActionEmissionPublicationReceipt:
    """Application proof of committed-emission publication, not target commit."""

    accepted: bool
    publication_key: str
    source_binding_hash: str
    emission_id: str
    emission_semantic_content_hash: str
    commit_receipt_hash: str
    published_at: float
    idempotent_replay: bool
    required_control_outcome: PolicyActionEmissionDeliveryOutcome | None
    observer_outcomes: tuple[PolicyActionEmissionDeliveryOutcome, ...]
    reasons: tuple[str, ...]
    semantic_content_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "publication_key": self.publication_key,
            "source_binding_hash": self.source_binding_hash,
            "emission_id": self.emission_id,
            "emission_semantic_content_hash": (
                self.emission_semantic_content_hash
            ),
            "commit_receipt_hash": self.commit_receipt_hash,
            "published_at": self.published_at,
            "idempotent_replay": self.idempotent_replay,
            "required_control_outcome": (
                None
                if self.required_control_outcome is None
                else self.required_control_outcome.to_payload()
            ),
            "observer_outcomes": tuple(
                outcome.to_payload() for outcome in self.observer_outcomes
            ),
            "reasons": self.reasons,
        }


@dataclass(frozen=True)
class RtcPolicyEmissionBindingAck:
    """Consumer proof that one epoch binding was explicitly installed."""

    accepted: bool
    subscriber_id: str
    source_binding_hash: str
    source_descriptor_hash: str
    binding_generation: int
    acknowledged_at: float
    reasons: tuple[str, ...]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "subscriber_id",
            "source_binding_hash",
            "source_descriptor_hash",
            "semantic_content_hash",
        ):
            normalized = str(getattr(self, name)).strip()
            if not normalized:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, normalized)
        generation = int(self.binding_generation)
        if generation < 0 or generation != self.binding_generation:
            raise ValueError("binding_generation must be a non-negative integer.")
        object.__setattr__(self, "binding_generation", generation)
        acknowledged_at = float(self.acknowledged_at)
        if not math.isfinite(acknowledged_at):
            raise ValueError("acknowledged_at must be finite.")
        object.__setattr__(self, "acknowledged_at", acknowledged_at)
        reasons = tuple(str(reason).strip() for reason in self.reasons)
        if any(not reason for reason in reasons):
            raise ValueError("Binding ACK reasons must be non-empty.")
        if self.accepted and reasons:
            raise ValueError("An accepted binding ACK cannot carry reasons.")
        if not self.accepted and not reasons:
            raise ValueError("A rejected binding ACK requires reasons.")
        object.__setattr__(self, "reasons", reasons)
        expected_hash = canonical_content_hash(
            {
                "accepted": self.accepted,
                "subscriber_id": self.subscriber_id,
                "source_binding_hash": self.source_binding_hash,
                "source_descriptor_hash": self.source_descriptor_hash,
                "binding_generation": self.binding_generation,
                "acknowledged_at": self.acknowledged_at,
                "reasons": self.reasons,
            },
            domain=RTC_POLICY_EMISSION_BINDING_ACK_HASH_DOMAIN_V1,
        )
        if self.semantic_content_hash != expected_hash:
            raise ValueError("Binding ACK semantic hash mismatch.")


def build_rtc_policy_emission_binding_ack(
    binding: RtcPolicyEmissionSourceBinding,
    *,
    subscriber_id: str,
    accepted: bool,
    acknowledged_at: float,
    reasons: tuple[str, ...] = (),
) -> RtcPolicyEmissionBindingAck:
    """Build one typed ACK; it does not activate or rearm the producer."""

    if not isinstance(binding, RtcPolicyEmissionSourceBinding):
        raise TypeError("binding must be an RtcPolicyEmissionSourceBinding.")
    payload = {
        "accepted": bool(accepted),
        "subscriber_id": str(subscriber_id).strip(),
        "source_binding_hash": binding.semantic_content_hash,
        "source_descriptor_hash": binding.descriptor_hash,
        "binding_generation": binding.binding_generation,
        "acknowledged_at": float(acknowledged_at),
        "reasons": tuple(reasons),
    }
    return RtcPolicyEmissionBindingAck(
        **payload,
        semantic_content_hash=canonical_content_hash(
            payload,
            domain=RTC_POLICY_EMISSION_BINDING_ACK_HASH_DOMAIN_V1,
        ),
    )


class RtcFrontendStepFanout:
    """Debug-only compatibility fanout for complete RTC frontend steps."""

    nonblocking_bounded = True

    def __init__(
        self,
        subscribers: Mapping[str, queue.Queue[RtcFrontendStep]],
    ) -> None:
        normalized: dict[str, queue.Queue[RtcFrontendStep]] = {}
        for raw_subscriber_id, subscriber_queue in subscribers.items():
            subscriber_id = str(raw_subscriber_id).strip()
            if not subscriber_id:
                raise ValueError("RTC step subscriber_id must be non-empty.")
            if subscriber_id in normalized:
                raise ValueError(f"Duplicate RTC step subscriber_id: {subscriber_id!r}.")
            if int(getattr(subscriber_queue, "maxsize", 0)) <= 0:
                raise ValueError(
                    f"RTC step subscriber {subscriber_id!r} must use a bounded queue."
                )
            normalized[subscriber_id] = subscriber_queue
        if not normalized:
            raise ValueError("RTC step fanout requires at least one subscriber.")
        self._subscribers = normalized
        self._counters = {
            subscriber_id: _SubscriberCounters()
            for subscriber_id in normalized
        }
        self._lock = threading.RLock()
        self._binding_condition = threading.Condition(self._lock)
        self._source_binding: RtcFrontendStepSourceBinding | None = None
        self._input_offer_count = 0
        self._emitting_step_count = 0

    def bind_frontend(
        self,
        frontend: RtcPolicyShadowFrontend,
    ) -> RtcFrontendStepSourceBinding:
        """Publish one exact lower-owned source binding before fanout use."""

        binding = build_rtc_frontend_step_source_binding(frontend)
        with self._binding_condition:
            if self._source_binding is not None:
                raise RuntimeError("RTC step fanout source is already bound.")
            self._source_binding = binding
            self._binding_condition.notify_all()
        return binding

    @property
    def source_binding(self) -> RtcFrontendStepSourceBinding | None:
        with self._lock:
            return self._source_binding

    def wait_for_source_binding(
        self,
        *,
        timeout_sec: float,
    ) -> RtcFrontendStepSourceBinding | None:
        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("timeout_sec must be finite and non-negative.")
        with self._binding_condition:
            self._binding_condition.wait_for(
                lambda: self._source_binding is not None,
                timeout=timeout,
            )
            return self._source_binding

    def put_nowait(self, step: RtcFrontendStep) -> None:
        """Offer one exact lower result; subscriber failures never propagate."""

        if not isinstance(step, RtcFrontendStep):
            raise TypeError("RTC step fanout accepts RtcFrontendStep only.")
        with self._lock:
            self._input_offer_count += 1
            if step.emission_result.emission is not None:
                self._emitting_step_count += 1
            subscribers = tuple(self._subscribers.items())
        for subscriber_id, subscriber_queue in subscribers:
            try:
                subscriber_queue.put_nowait(step)
                with self._lock:
                    self._counters[subscriber_id].offer_count += 1
            except queue.Full:
                with self._lock:
                    self._counters[subscriber_id].drop_count += 1
            except Exception as exc:  # passive observation cannot fault deploy
                with self._lock:
                    counters = self._counters[subscriber_id]
                    counters.error_count += 1
                    counters.last_error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, object]:
        with self._lock:
            statuses = tuple(
                RtcFrontendStepSubscriberStatus(
                    subscriber_id=subscriber_id,
                    queue_size=subscriber_queue.qsize(),
                    queue_capacity=int(subscriber_queue.maxsize),
                    offer_count=counters.offer_count,
                    drop_count=counters.drop_count,
                    error_count=counters.error_count,
                    last_error=counters.last_error,
                )
                for subscriber_id, subscriber_queue in self._subscribers.items()
                for counters in (self._counters[subscriber_id],)
            )
            return {
                "enabled": True,
                "authoritative": False,
                "producer_lifecycle_owned": False,
                "source_binding_hash": (
                    None
                    if self._source_binding is None
                    else self._source_binding.semantic_content_hash
                ),
                "input_offer_count": self._input_offer_count,
                "emitting_step_count": self._emitting_step_count,
                "subscriber_count": len(statuses),
                "subscribers": {
                    item.subscriber_id: {
                        "queue_size": item.queue_size,
                        "queue_capacity": item.queue_capacity,
                        "offer_count": item.offer_count,
                        "drop_count": item.drop_count,
                        "error_count": item.error_count,
                        "last_error": item.last_error,
                    }
                    for item in statuses
                },
            }


class PolicyActionEmissionFanout:
    """Compatibility observer fanout without producer-commit validation.

    New application paths must use :class:`RtcPolicyEmissionPublisher`.
    This class remains for passive/debug callers whose application already
    proved the commit at the call site.
    """

    nonblocking_bounded = True

    def __init__(
        self,
        subscribers: Mapping[str, queue.Queue[PolicyActionEmission]],
    ) -> None:
        normalized: dict[str, queue.Queue[PolicyActionEmission]] = {}
        for raw_subscriber_id, subscriber_queue in subscribers.items():
            subscriber_id = str(raw_subscriber_id).strip()
            if not subscriber_id:
                raise ValueError("Policy emission subscriber_id must be non-empty.")
            if subscriber_id in normalized:
                raise ValueError(
                    f"Duplicate policy emission subscriber_id: {subscriber_id!r}."
                )
            if int(getattr(subscriber_queue, "maxsize", 0)) <= 0:
                raise ValueError(
                    f"Policy emission subscriber {subscriber_id!r} "
                    "must use a bounded queue."
                )
            normalized[subscriber_id] = subscriber_queue
        if not normalized:
            raise ValueError("Policy emission fanout requires at least one subscriber.")
        self._subscribers = normalized
        self._counters = {
            subscriber_id: _SubscriberCounters()
            for subscriber_id in normalized
        }
        self._lock = threading.RLock()
        self._binding_condition = threading.Condition(self._lock)
        self._source_binding: RtcPolicyEmissionSourceBinding | None = None
        self._input_offer_count = 0

    def bind_frontend(
        self,
        frontend: RtcPolicyShadowFrontend,
    ) -> RtcPolicyEmissionSourceBinding:
        binding = build_rtc_policy_emission_source_binding(frontend)
        with self._binding_condition:
            if self._source_binding is not None:
                raise RuntimeError("Policy emission fanout source is already bound.")
            self._source_binding = binding
            self._binding_condition.notify_all()
        return binding

    @property
    def source_binding(self) -> RtcPolicyEmissionSourceBinding | None:
        with self._lock:
            return self._source_binding

    def wait_for_source_binding(
        self,
        *,
        timeout_sec: float,
    ) -> RtcPolicyEmissionSourceBinding | None:
        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("timeout_sec must be finite and non-negative.")
        with self._binding_condition:
            self._binding_condition.wait_for(
                lambda: self._source_binding is not None,
                timeout=timeout,
            )
            return self._source_binding

    def put_nowait(self, emission: PolicyActionEmission) -> None:
        """Offer one formal emission; subscriber failures never propagate."""

        with self._lock:
            binding = self._source_binding
        if binding is None:
            raise RuntimeError("Policy emission fanout source is not bound.")
        binding.validate_emission(emission)
        with self._lock:
            self._input_offer_count += 1
            subscribers = tuple(self._subscribers.items())
        for subscriber_id, subscriber_queue in subscribers:
            try:
                subscriber_queue.put_nowait(emission)
                with self._lock:
                    self._counters[subscriber_id].offer_count += 1
            except queue.Full:
                with self._lock:
                    self._counters[subscriber_id].drop_count += 1
            except Exception as exc:  # passive observation cannot fault deploy
                with self._lock:
                    counters = self._counters[subscriber_id]
                    counters.error_count += 1
                    counters.last_error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": True,
                "authoritative": False,
                "producer_lifecycle_owned": False,
                "source_binding_hash": (
                    None
                    if self._source_binding is None
                    else self._source_binding.semantic_content_hash
                ),
                "input_offer_count": self._input_offer_count,
                "subscriber_count": len(self._subscribers),
                "subscribers": {
                    subscriber_id: {
                        "queue_size": subscriber_queue.qsize(),
                        "queue_capacity": int(subscriber_queue.maxsize),
                        "offer_count": self._counters[subscriber_id].offer_count,
                        "drop_count": self._counters[subscriber_id].drop_count,
                        "error_count": self._counters[subscriber_id].error_count,
                        "last_error": self._counters[subscriber_id].last_error,
                    }
                    for subscriber_id, subscriber_queue in self._subscribers.items()
                },
            }


class RtcPolicyEmissionPublisher:
    """Validate and publish committed RTC emissions through bounded endpoints.

    Observers are best-effort and may drop.  The optional single control
    subscriber is required: a full or faulty endpoint rejects publication and
    suspends this publisher.  This façade owns application binding activation,
    but no producer lifecycle, target preparation, safety, or motion authority.
    """

    nonblocking_bounded = True

    def __init__(
        self,
        *,
        observers: Mapping[str, queue.Queue[PolicyActionEmission]] | None = None,
        required_control_subscriber: (
            tuple[str, queue.Queue[PolicyActionEmission]] | None
        ) = None,
        publication_receipt_capacity: int = 4096,
    ) -> None:
        normalized_observers = self._normalize_subscribers(
            {} if observers is None else observers,
            role="observer",
        )
        required: tuple[str, queue.Queue[PolicyActionEmission]] | None = None
        if required_control_subscriber is not None:
            if (
                not isinstance(required_control_subscriber, tuple)
                or len(required_control_subscriber) != 2
            ):
                raise TypeError(
                    "required_control_subscriber must be a "
                    "(subscriber_id, bounded_queue) tuple or None."
                )
            required_id, required_queue = required_control_subscriber
            normalized_required = self._normalize_subscribers(
                {required_id: required_queue},
                role="required_control",
            )
            required = next(iter(normalized_required.items()))
            if required[0] in normalized_observers:
                raise ValueError(
                    "A subscriber_id cannot be both observer and required control."
                )
        if not normalized_observers and required is None:
            raise ValueError("RTC emission publisher requires a subscriber.")
        capacity = int(publication_receipt_capacity)
        if capacity <= 0:
            raise ValueError("publication_receipt_capacity must be positive.")
        self._observers = normalized_observers
        self._required_control_subscriber = required
        self._publication_receipt_capacity = capacity
        self._lock = threading.RLock()
        self._binding_condition = threading.Condition(self._lock)
        self._active_binding: RtcPolicyEmissionSourceBinding | None = None
        self._suspended = False
        self._suspension_reason = ""
        self._accepted_receipts: dict[
            str,
            PolicyActionEmissionPublicationReceipt,
        ] = {}
        self._accepted_publication_count = 0
        self._rejected_publication_count = 0
        self._idempotent_replay_count = 0
        self._observer_drop_count = 0

    @staticmethod
    def _normalize_subscribers(
        subscribers: Mapping[str, queue.Queue[PolicyActionEmission]],
        *,
        role: str,
    ) -> dict[str, queue.Queue[PolicyActionEmission]]:
        normalized: dict[str, queue.Queue[PolicyActionEmission]] = {}
        for raw_subscriber_id, subscriber_queue in subscribers.items():
            subscriber_id = str(raw_subscriber_id).strip()
            if not subscriber_id:
                raise ValueError(f"{role} subscriber_id must be non-empty.")
            if subscriber_id in normalized:
                raise ValueError(f"Duplicate {role} subscriber_id: {subscriber_id!r}.")
            if not isinstance(subscriber_queue, queue.Queue):
                raise TypeError(f"{role} subscriber must be a queue.Queue.")
            if int(subscriber_queue.maxsize) <= 0:
                raise ValueError(f"{role} subscriber queue must be bounded.")
            normalized[subscriber_id] = subscriber_queue
        return normalized

    def bind_frontend(
        self,
        frontend: RtcPolicyShadowFrontend,
    ) -> RtcPolicyEmissionSourceBinding:
        """Activate the producer-owned initial binding candidate."""

        binding = build_rtc_policy_emission_source_binding(frontend)
        with self._binding_condition:
            if self._active_binding is not None:
                raise RuntimeError("RTC emission publisher is already bound.")
            self._active_binding = binding
            self._binding_condition.notify_all()
        return binding

    @property
    def source_binding(self) -> RtcPolicyEmissionSourceBinding | None:
        """Compatibility spelling for the active application binding."""

        with self._lock:
            return self._active_binding

    @property
    def active_binding(self) -> RtcPolicyEmissionSourceBinding | None:
        with self._lock:
            return self._active_binding

    def wait_for_source_binding(
        self,
        *,
        timeout_sec: float,
    ) -> RtcPolicyEmissionSourceBinding | None:
        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("timeout_sec must be finite and non-negative.")
        with self._binding_condition:
            self._binding_condition.wait_for(
                lambda: self._active_binding is not None,
                timeout=timeout,
            )
            return self._active_binding

    def suspend_publication(self, *, reason: str) -> RtcPolicyEmissionSourceBinding:
        """Stop publication before retiring or replacing the active binding."""

        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("suspension reason must be non-empty.")
        with self._lock:
            if self._active_binding is None:
                raise RuntimeError("RTC emission publisher is not bound.")
            self._suspended = True
            self._suspension_reason = normalized_reason
            return self._active_binding

    def activate_rebound_binding(
        self,
        binding: RtcPolicyEmissionSourceBinding,
        *,
        readiness_receipt: RtcProducerActivationReadinessReceipt,
        required_control_ack: RtcPolicyEmissionBindingAck | None,
    ) -> RtcPolicyEmissionSourceBinding:
        """Atomically install one rearmed epoch after readiness and required ACK."""

        if not isinstance(binding, RtcPolicyEmissionSourceBinding):
            raise TypeError("binding must be an RtcPolicyEmissionSourceBinding.")
        if not isinstance(
            readiness_receipt,
            RtcProducerActivationReadinessReceipt,
        ):
            raise TypeError(
                "readiness_receipt must be an "
                "RtcProducerActivationReadinessReceipt."
            )
        if required_control_ack is not None and not isinstance(
            required_control_ack,
            RtcPolicyEmissionBindingAck,
        ):
            raise TypeError(
                "required_control_ack must be an "
                "RtcPolicyEmissionBindingAck or None."
            )
        with self._binding_condition:
            old_binding = self._active_binding
            if old_binding is None:
                raise RuntimeError("RTC emission publisher is not bound.")
            if not self._suspended:
                raise RuntimeError(
                    "RTC emission publication must be suspended before rebind."
                )
            if binding.semantic_content_hash == old_binding.semantic_content_hash:
                raise ValueError("Rebind requires a new epoch binding.")
            if binding.descriptor_hash != old_binding.descriptor_hash:
                raise ValueError(
                    "V1 stop-and-rebind cannot change the stable source descriptor."
                )
            if binding.producer_id != old_binding.producer_id:
                raise ValueError("Rebind belongs to another producer.")
            if binding.producer_epoch <= old_binding.producer_epoch:
                raise ValueError("Rebind producer epoch must advance.")
            if not readiness_receipt.ready:
                raise ValueError("Producer activation readiness was rejected.")
            if (
                readiness_receipt.producer_id != binding.producer_id
                or readiness_receipt.old_producer_epoch
                != old_binding.producer_epoch
                or readiness_receipt.new_producer_epoch
                != binding.producer_epoch
            ):
                raise ValueError("Producer activation readiness/binding mismatch.")
            required = self._required_control_subscriber
            if required is None:
                if required_control_ack is not None:
                    raise ValueError(
                        "Observer-only publisher does not accept a required ACK."
                    )
            else:
                required_id, _required_queue = required
                if required_control_ack is None:
                    raise ValueError("Required control subscriber ACK is missing.")
                if not required_control_ack.accepted:
                    raise ValueError("Required control subscriber rejected binding.")
                if required_control_ack.subscriber_id != required_id:
                    raise ValueError("Binding ACK belongs to another subscriber.")
                if (
                    required_control_ack.source_binding_hash
                    != binding.semantic_content_hash
                    or required_control_ack.source_descriptor_hash
                    != binding.descriptor_hash
                    or required_control_ack.binding_generation
                    != binding.binding_generation
                ):
                    raise ValueError("Required control subscriber ACK/binding mismatch.")
            self._active_binding = binding
            self._accepted_receipts.clear()
            self._suspended = False
            self._suspension_reason = ""
            self._binding_condition.notify_all()
            return binding

    @staticmethod
    def _publication_key(emission: PolicyActionEmission) -> str:
        return canonical_content_hash(
            {
                "source_binding_hash": emission.source_binding_hash,
                "emission_id": emission.emission_id,
            },
            domain=RTC_POLICY_EMISSION_PUBLICATION_KEY_HASH_DOMAIN_V1,
        )

    @staticmethod
    def _build_receipt(
        *,
        accepted: bool,
        publication_key: str,
        source_binding_hash: str,
        emission: PolicyActionEmission,
        commit_receipt: PolicyActionEmissionCommitReceipt,
        published_at: float,
        idempotent_replay: bool,
        required_control_outcome: PolicyActionEmissionDeliveryOutcome | None,
        observer_outcomes: tuple[PolicyActionEmissionDeliveryOutcome, ...],
        reasons: tuple[str, ...],
    ) -> PolicyActionEmissionPublicationReceipt:
        payload = {
            "accepted": accepted,
            "publication_key": publication_key,
            "source_binding_hash": source_binding_hash,
            "emission_id": emission.emission_id,
            "emission_semantic_content_hash": emission.semantic_content_hash,
            "commit_receipt_hash": commit_receipt.semantic_content_hash,
            "published_at": published_at,
            "idempotent_replay": idempotent_replay,
            "required_control_outcome": (
                None
                if required_control_outcome is None
                else required_control_outcome.to_payload()
            ),
            "observer_outcomes": tuple(
                outcome.to_payload() for outcome in observer_outcomes
            ),
            "reasons": reasons,
        }
        return PolicyActionEmissionPublicationReceipt(
            accepted=accepted,
            publication_key=publication_key,
            source_binding_hash=source_binding_hash,
            emission_id=emission.emission_id,
            emission_semantic_content_hash=emission.semantic_content_hash,
            commit_receipt_hash=commit_receipt.semantic_content_hash,
            published_at=published_at,
            idempotent_replay=idempotent_replay,
            required_control_outcome=required_control_outcome,
            observer_outcomes=observer_outcomes,
            reasons=reasons,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=RTC_POLICY_EMISSION_PUBLICATION_RECEIPT_HASH_DOMAIN_V1,
            ),
        )

    def _validate_publication(
        self,
        emission: PolicyActionEmission,
        commit_receipt: PolicyActionEmissionCommitReceipt,
        *,
        published_at: float,
    ) -> tuple[RtcPolicyEmissionSourceBinding | None, tuple[str, ...]]:
        reasons: list[str] = []
        binding = self._active_binding
        if binding is None:
            reasons.append("publisher_unbound")
            return None, tuple(reasons)
        if self._suspended:
            reasons.append(f"publisher_suspended:{self._suspension_reason}")
        try:
            binding.validate_emission(emission)
        except (TypeError, ValueError) as exc:
            reasons.append(f"emission_binding_invalid:{type(exc).__name__}:{exc}")
        if not commit_receipt.accepted:
            reasons.append("producer_commit_not_accepted")
        if commit_receipt.emission_id != emission.emission_id:
            reasons.append("producer_commit_emission_id_mismatch")
        if (
            commit_receipt.emission_semantic_content_hash
            != emission.semantic_content_hash
        ):
            reasons.append("producer_commit_emission_hash_mismatch")
        if (
            commit_receipt.source_binding_hash
            != binding.semantic_content_hash
        ):
            reasons.append("producer_commit_binding_mismatch")
        if commit_receipt.producer_id != binding.producer_id:
            reasons.append("producer_commit_producer_id_mismatch")
        if commit_receipt.producer_epoch != binding.producer_epoch:
            reasons.append("producer_commit_epoch_mismatch")
        if published_at > float(emission.effective_until):
            reasons.append("emission_ttl_expired")
        return binding, tuple(reasons)

    def publish_committed_emission(
        self,
        emission: PolicyActionEmission,
        commit_receipt: PolicyActionEmissionCommitReceipt,
        *,
        now: float,
    ) -> PolicyActionEmissionPublicationReceipt:
        """Publish exactly one committed emission without target-side claims."""

        if not isinstance(emission, PolicyActionEmission):
            raise TypeError("emission must be a PolicyActionEmission.")
        if not isinstance(commit_receipt, PolicyActionEmissionCommitReceipt):
            raise TypeError(
                "commit_receipt must be a PolicyActionEmissionCommitReceipt."
            )
        published_at = float(now)
        if not math.isfinite(published_at):
            raise ValueError("now must be finite.")
        publication_key = self._publication_key(emission)
        with self._lock:
            binding, reasons = self._validate_publication(
                emission,
                commit_receipt,
                published_at=published_at,
            )
            binding_hash = (
                str(emission.source_binding_hash)
                if binding is None
                else binding.semantic_content_hash
            )
            cached = self._accepted_receipts.get(publication_key)
            if cached is not None:
                if (
                    cached.emission_semantic_content_hash
                    != emission.semantic_content_hash
                    or cached.commit_receipt_hash
                    != commit_receipt.semantic_content_hash
                ):
                    reasons = (*reasons, "publication_key_reuse_mismatch")
                elif not reasons or reasons == ("emission_ttl_expired",):
                    self._idempotent_replay_count += 1
                    return self._build_receipt(
                        accepted=True,
                        publication_key=cached.publication_key,
                        source_binding_hash=cached.source_binding_hash,
                        emission=emission,
                        commit_receipt=commit_receipt,
                        published_at=cached.published_at,
                        idempotent_replay=True,
                        required_control_outcome=(
                            cached.required_control_outcome
                        ),
                        observer_outcomes=cached.observer_outcomes,
                        reasons=(),
                    )
            if reasons:
                self._rejected_publication_count += 1
                return self._build_receipt(
                    accepted=False,
                    publication_key=publication_key,
                    source_binding_hash=binding_hash,
                    emission=emission,
                    commit_receipt=commit_receipt,
                    published_at=published_at,
                    idempotent_replay=False,
                    required_control_outcome=None,
                    observer_outcomes=(),
                    reasons=tuple(reasons),
                )
            if len(self._accepted_receipts) >= self._publication_receipt_capacity:
                self._rejected_publication_count += 1
                return self._build_receipt(
                    accepted=False,
                    publication_key=publication_key,
                    source_binding_hash=binding_hash,
                    emission=emission,
                    commit_receipt=commit_receipt,
                    published_at=published_at,
                    idempotent_replay=False,
                    required_control_outcome=None,
                    observer_outcomes=(),
                    reasons=("publication_receipt_capacity_exhausted",),
                )
            required_outcome = None
            if self._required_control_subscriber is not None:
                subscriber_id, subscriber_queue = (
                    self._required_control_subscriber
                )
                try:
                    subscriber_queue.put_nowait(emission)
                    required_outcome = PolicyActionEmissionDeliveryOutcome(
                        subscriber_id=subscriber_id,
                        subscriber_role="required_control",
                        accepted=True,
                        dropped=False,
                        reason="accepted",
                    )
                except queue.Full:
                    required_outcome = PolicyActionEmissionDeliveryOutcome(
                        subscriber_id=subscriber_id,
                        subscriber_role="required_control",
                        accepted=False,
                        dropped=False,
                        reason="queue_full",
                    )
                except Exception as exc:
                    required_outcome = PolicyActionEmissionDeliveryOutcome(
                        subscriber_id=subscriber_id,
                        subscriber_role="required_control",
                        accepted=False,
                        dropped=False,
                        reason=f"{type(exc).__name__}:{exc}",
                    )
                if not required_outcome.accepted:
                    self._suspended = True
                    self._suspension_reason = (
                        "required_control_subscriber_delivery_failed"
                    )
                    self._rejected_publication_count += 1
                    return self._build_receipt(
                        accepted=False,
                        publication_key=publication_key,
                        source_binding_hash=binding_hash,
                        emission=emission,
                        commit_receipt=commit_receipt,
                        published_at=published_at,
                        idempotent_replay=False,
                        required_control_outcome=required_outcome,
                        observer_outcomes=(),
                        reasons=(
                            "required_control_subscriber_delivery_failed",
                        ),
                    )
            observer_outcomes: list[PolicyActionEmissionDeliveryOutcome] = []
            for subscriber_id, subscriber_queue in self._observers.items():
                try:
                    subscriber_queue.put_nowait(emission)
                    outcome = PolicyActionEmissionDeliveryOutcome(
                        subscriber_id=subscriber_id,
                        subscriber_role="observer",
                        accepted=True,
                        dropped=False,
                        reason="accepted",
                    )
                except queue.Full:
                    self._observer_drop_count += 1
                    outcome = PolicyActionEmissionDeliveryOutcome(
                        subscriber_id=subscriber_id,
                        subscriber_role="observer",
                        accepted=False,
                        dropped=True,
                        reason="queue_full",
                    )
                except Exception as exc:
                    self._observer_drop_count += 1
                    outcome = PolicyActionEmissionDeliveryOutcome(
                        subscriber_id=subscriber_id,
                        subscriber_role="observer",
                        accepted=False,
                        dropped=True,
                        reason=f"{type(exc).__name__}:{exc}",
                    )
                observer_outcomes.append(outcome)
            receipt = self._build_receipt(
                accepted=True,
                publication_key=publication_key,
                source_binding_hash=binding_hash,
                emission=emission,
                commit_receipt=commit_receipt,
                published_at=published_at,
                idempotent_replay=False,
                required_control_outcome=required_outcome,
                observer_outcomes=tuple(observer_outcomes),
                reasons=(),
            )
            self._accepted_receipts[publication_key] = receipt
            self._accepted_publication_count += 1
            return receipt

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": True,
                "authoritative": False,
                "producer_lifecycle_owned": False,
                "target_commit_owned": False,
                "active": self._active_binding is not None and not self._suspended,
                "suspended": self._suspended,
                "suspension_reason": self._suspension_reason,
                "source_binding_hash": (
                    None
                    if self._active_binding is None
                    else self._active_binding.semantic_content_hash
                ),
                "source_descriptor_hash": (
                    None
                    if self._active_binding is None
                    else self._active_binding.descriptor_hash
                ),
                "required_control_subscriber_id": (
                    None
                    if self._required_control_subscriber is None
                    else self._required_control_subscriber[0]
                ),
                "observer_subscriber_count": len(self._observers),
                "accepted_publication_count": self._accepted_publication_count,
                "rejected_publication_count": self._rejected_publication_count,
                "idempotent_replay_count": self._idempotent_replay_count,
                "observer_drop_count": self._observer_drop_count,
            }


# Canonical functional name; the old class remains the same debug-only object.
RtcProducerDebugStepFanout = RtcFrontendStepFanout


__all__ = [
    "PolicyActionEmissionDeliveryOutcome",
    "PolicyActionEmissionFanout",
    "PolicyActionEmissionPublicationReceipt",
    "RTC_FRONTEND_STEP_SOURCE_BINDING_HASH_DOMAIN_V1",
    "RTC_POLICY_EMISSION_PUBLICATION_KEY_HASH_DOMAIN_V1",
    "RTC_POLICY_EMISSION_PUBLICATION_RECEIPT_HASH_DOMAIN_V1",
    "RTC_POLICY_EMISSION_BINDING_ACK_HASH_DOMAIN_V1",
    "RTC_POLICY_EMISSION_SOURCE_BINDING_HASH_DOMAIN_V1",
    "RTC_POLICY_RUNTIME_SOURCE_CONFIG_HASH_DOMAIN_V1",
    "RtcFrontendStepSourceBinding",
    "RtcFrontendStepFanout",
    "RtcFrontendStepSubscriberStatus",
    "RtcPolicyEmissionSourceBinding",
    "RtcPolicyEmissionBindingAck",
    "RtcPolicyEmissionPublisher",
    "RtcProducerDebugStepFanout",
    "build_rtc_frontend_step_source_binding",
    "build_rtc_policy_emission_binding_ack",
    "build_rtc_policy_emission_source_binding",
]
