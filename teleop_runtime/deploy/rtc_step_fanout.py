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
from robot_motion.deploy import CanonicalDeployActionPipeline, PolicyActionEmission

from .rtc_frontend import RtcFrontendStep, RtcPolicyShadowFrontend


RTC_FRONTEND_STEP_SOURCE_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_frontend_step_source_binding.v1"
)
RTC_POLICY_RUNTIME_SOURCE_CONFIG_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_policy_runtime_source_config.v1"
)


@dataclass(frozen=True)
class RtcFrontendStepSourceBinding:
    """Target-neutral identity for one lower-owned RTC step stream.

    The binding is captured from an existing frontend without starting or
    stepping it.  It proves which producer/configuration a passive subscriber
    expects; it says nothing about target compatibility, execution authority,
    or motion authorization.
    """

    source_slot_id: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    policy_node_id: str
    policy_clock_domain_id: str
    policy_runtime_config_hash: str
    action_adapter_contract_hash: str
    rtc_frontend_config_hash: str
    rtc_profile_hash: str
    artifact_hash: str
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "source_slot_id": self.source_slot_id,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "producer_id": self.producer_id,
                "producer_epoch": self.producer_epoch,
                "policy_node_id": self.policy_node_id,
                "policy_clock_domain_id": self.policy_clock_domain_id,
                "policy_runtime_config_hash": self.policy_runtime_config_hash,
                "action_adapter_contract_hash": self.action_adapter_contract_hash,
                "rtc_frontend_config_hash": self.rtc_frontend_config_hash,
                "rtc_profile_hash": self.rtc_profile_hash,
                "artifact_hash": self.artifact_hash,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
            },
            domain=RTC_FRONTEND_STEP_SOURCE_BINDING_HASH_DOMAIN_V1,
        )

    def validate_step(self, step: RtcFrontendStep) -> PolicyActionEmission | None:
        """Validate source identity and exact lower pop/commit lineage."""

        if not isinstance(step, RtcFrontendStep):
            raise TypeError("step must be an RtcFrontendStep.")
        plan = step.state.plan
        if (
            plan.session_id != self.runtime_session_id
            or plan.control_domain_id != self.control_domain_id
            or plan.producer_id != self.producer_id
            or plan.producer_epoch != self.producer_epoch
            or plan.policy_node_id != self.policy_node_id
            or plan.clock_domain_id != self.policy_clock_domain_id
        ):
            raise ValueError("RTC step producer state does not match source binding.")
        request = step.state.active_request
        if request is not None and (
            request.session_id != self.runtime_session_id
            or request.control_domain_id != self.control_domain_id
            or request.producer_id != self.producer_id
            or request.producer_epoch != self.producer_epoch
            or request.policy_node_id != self.policy_node_id
            or request.clock_domain_id != self.policy_clock_domain_id
            or request.artifact_hash != self.artifact_hash
            or request.action_codec_hash != self.action_codec_hash
            or request.frame_contract_hash != self.frame_contract_hash
            or request.layout_hash != self.layout_hash
            or request.normalization_hash != self.normalization_hash
        ):
            raise ValueError("RTC active request does not match source binding.")
        completion = step.completion
        if completion is not None:
            result = completion.result
            if (
                result.session_id != self.runtime_session_id
                or result.control_domain_id != self.control_domain_id
                or result.producer_id != self.producer_id
                or result.producer_epoch != self.producer_epoch
                or result.policy_node_id != self.policy_node_id
                or result.clock_domain_id != self.policy_clock_domain_id
                or result.artifact_hash != self.artifact_hash
                or result.action_codec_hash != self.action_codec_hash
                or result.frame_contract_hash != self.frame_contract_hash
                or result.layout_hash != self.layout_hash
                or result.normalization_hash != self.normalization_hash
            ):
                raise ValueError("RTC worker completion does not match source binding.")

        emission = step.action_preparation.emission
        if emission is None:
            if step.emission_commit_transition is not None:
                raise ValueError("RTC step committed an emission that is not present.")
            return None

        pop_transition = step.pop_transition
        commit_transition = step.emission_commit_transition
        if pop_transition is None or not pop_transition.receipt.accepted:
            raise ValueError("RTC emission lacks an accepted lower-owned due-pop.")
        pop = pop_transition.pop_outcome
        if pop is None or pop.keyframe is None:
            raise ValueError("RTC emission due-pop selected no keyframe.")
        if commit_transition is None or not commit_transition.receipt.accepted:
            raise ValueError("RTC emission lacks accepted commit-through evidence.")
        if commit_transition.receipt.emission_id != emission.emission_id:
            raise ValueError("RTC emission/commit identity mismatch.")
        keyframe = pop.keyframe
        if (
            keyframe.inference_id != emission.inference_id
            or keyframe.chunk_id != emission.chunk_id
            or keyframe.chunk_keyframe_index != emission.chunk_keyframe_index
            or keyframe.target_effective_at != emission.target_effective_at
        ):
            raise ValueError("RTC due-pop keyframe/emission lineage mismatch.")
        if emission.policy_node_id != self.policy_node_id:
            raise ValueError("RTC emission belongs to another policy node.")
        if (
            emission.produced_clock_domain_id != self.policy_clock_domain_id
            or emission.effective_clock_domain_id != self.policy_clock_domain_id
        ):
            raise ValueError("RTC emission clock domain does not match source binding.")
        if emission.queue_pop_count <= 0:
            raise ValueError("RTC emission has no lower-owned plan-pop evidence.")
        receipt = emission.representation_receipt
        if receipt is not None and (
            receipt.adapter_contract_hash != self.action_adapter_contract_hash
        ):
            raise ValueError("RTC emission used another action adapter contract.")
        return emission


def build_rtc_frontend_step_source_binding(
    frontend: RtcPolicyShadowFrontend,
) -> RtcFrontendStepSourceBinding:
    """Capture a target-neutral binding without owning RTC lifecycle."""

    if not isinstance(frontend, RtcPolicyShadowFrontend):
        raise TypeError("frontend must be an RtcPolicyShadowFrontend.")
    if not isinstance(frontend.preparer, CanonicalDeployActionPipeline):
        raise TypeError("RTC frontend preparer must be CanonicalDeployActionPipeline.")
    config = frontend.config
    runtime_config = frontend.deploy_runtime.config
    plan = frontend.state.plan
    if (
        runtime_config.policy_node_id != config.policy_node_id
        or runtime_config.policy_clock_domain_id != config.clock_domain_id
    ):
        raise ValueError("RTC frontend/runtime policy identity differs.")
    if (
        tuple(frontend.preparer.active_sides) != tuple(runtime_config.active_sides)
        or frontend.preparer.arms_enabled != runtime_config.arms_enabled
        or frontend.preparer.hands_enabled != runtime_config.hands_enabled
    ):
        raise ValueError("RTC frontend action preparer/runtime coverage differs.")
    if (
        plan.session_id != config.runtime_session_id
        or plan.control_domain_id != config.control_domain_id
        or plan.producer_id != config.producer_id
        or plan.policy_node_id != config.policy_node_id
        or plan.clock_domain_id != config.clock_domain_id
    ):
        raise ValueError("RTC frontend initial plan/config identity differs.")
    runtime_payload = {
        "active_sides": runtime_config.active_sides,
        "arms_enabled": runtime_config.arms_enabled,
        "hands_enabled": runtime_config.hands_enabled,
        "policy_action_pose_mode": runtime_config.policy_action_pose_mode,
        "policy_absolute_frame": runtime_config.policy_absolute_frame,
        "policy_node_id": runtime_config.policy_node_id,
        "policy_clock_domain_id": runtime_config.policy_clock_domain_id,
        "action_keyframe_period_sec": runtime_config.action_keyframe_period_sec,
        "action_maximum_hold_sec": runtime_config.action_maximum_hold_sec,
        "action_emission_admission_mode": (
            runtime_config.action_emission_admission_mode
        ),
        "action_chunk_execution_steps": runtime_config.action_chunk_execution_steps,
        "policy_observation_contract_hash": (
            runtime_config.policy_observation_contract.semantic_content_hash
        ),
    }
    return RtcFrontendStepSourceBinding(
        source_slot_id=f"rtc-policy-source:{config.producer_id}",
        runtime_session_id=config.runtime_session_id,
        control_domain_id=config.control_domain_id,
        producer_id=config.producer_id,
        producer_epoch=plan.producer_epoch,
        policy_node_id=config.policy_node_id,
        policy_clock_domain_id=config.clock_domain_id,
        policy_runtime_config_hash=canonical_content_hash(
            runtime_payload,
            domain=RTC_POLICY_RUNTIME_SOURCE_CONFIG_HASH_DOMAIN_V1,
        ),
        action_adapter_contract_hash=(
            frontend.preparer.action_adapter.contract_hash
        ),
        rtc_frontend_config_hash=config.semantic_content_hash,
        rtc_profile_hash=config.profile.semantic_content_hash,
        artifact_hash=config.artifact_hash,
        action_codec_hash=config.action_codec_hash,
        frame_contract_hash=config.frame_contract_hash,
        layout_hash=config.layout_hash,
        normalization_hash=config.normalization_hash,
    )


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


class RtcFrontendStepFanout:
    """Offer each exact RTC step to passive branches without blocking deploy."""

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
            if step.action_preparation.emission is not None:
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


__all__ = [
    "RTC_FRONTEND_STEP_SOURCE_BINDING_HASH_DOMAIN_V1",
    "RTC_POLICY_RUNTIME_SOURCE_CONFIG_HASH_DOMAIN_V1",
    "RtcFrontendStepSourceBinding",
    "RtcFrontendStepFanout",
    "RtcFrontendStepSubscriberStatus",
    "build_rtc_frontend_step_source_binding",
]
