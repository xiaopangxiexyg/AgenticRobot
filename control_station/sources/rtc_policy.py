"""Passive Control Station consumer for the lower-owned RTC step seam.

The deploy frontend remains the sole owner of inference, timed-plan merge,
due-pop, emission commit, legacy execution, and producer lifecycle.  This
adapter only validates and observes the exact immutable ``RtcFrontendStep``
offered through a bounded queue.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

from robot_motion.deploy import PolicyActionEmission
from teleop_runtime.deploy.rtc_frontend import RtcFrontendStep
from teleop_runtime.deploy.rtc_step_fanout import RtcFrontendStepSourceBinding


RTC_POLICY_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2d_rtc_policy_binding.v1"
)


@dataclass(frozen=True)
class RtcPolicyEmissionBinding:
    """Typed evidence for one lower-owned RTC producer and exact target."""

    source_slot_id: str
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
    semantic_content_hash: str

    def validate_frontend_step(self, step: RtcFrontendStep) -> PolicyActionEmission:
        if not isinstance(step, RtcFrontendStep):
            raise TypeError("step must be an RtcFrontendStep.")
        emission = step.action_preparation.emission
        if emission is None:
            raise ValueError("RTC frontend step has no formal emission.")
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
        plan = step.state.plan
        if (
            plan.producer_id != self.producer_id
            or plan.producer_epoch != self.producer_epoch
            or plan.policy_node_id != self.policy_node_id
            or plan.clock_domain_id != self.policy_clock_domain_id
        ):
            raise ValueError("RTC producer state does not match Station binding.")
        if emission.policy_node_id != self.policy_node_id:
            raise ValueError("RTC emission belongs to another policy node.")
        if (
            emission.produced_clock_domain_id != self.policy_clock_domain_id
            or emission.effective_clock_domain_id != self.policy_clock_domain_id
        ):
            raise ValueError("RTC emission clock domain does not match binding.")
        if emission.queue_pop_count <= 0:
            raise ValueError("RTC emission has no lower-owned plan-pop evidence.")
        receipt = emission.representation_receipt
        if receipt is not None and (
            receipt.adapter_contract_hash != self.action_adapter_contract_hash
        ):
            raise ValueError("RTC emission used another action adapter contract.")
        return emission


@dataclass(frozen=True)
class RtcPolicyStationConsumerStatus:
    started: bool
    closed: bool
    shadow_enabled: bool
    queue_size: int
    queue_capacity: int
    received_step_count: int
    observed_step_count: int
    observed_emission_count: int
    inactive_step_count: int
    rejected_step_count: int
    error_count: int
    fence_sequence: int
    last_error: str
    last_state_hash: str | None
    last_emission_id: str | None


@dataclass
class _RtcShadowFence:
    sequence: int
    enabled: bool
    applied: threading.Event = field(default_factory=threading.Event)


class RtcPolicyStationConsumer:
    """Consume exact RTC steps behind an ordered passive shadow fence.

    The start/stop fence travels through the same queue as source steps.  A
    step offered before ``start_shadow`` therefore cannot be observed after
    activation merely because it was waiting in the subscriber queue.
    """

    authoritative = False
    producer_lifecycle_owned = False
    motion_capable = False

    def __init__(
        self,
        binding: RtcFrontendStepSourceBinding,
        observe_step: Callable[[RtcFrontendStep], object],
        *,
        step_queue: queue.Queue[object] | None = None,
        queue_capacity: int = 256,
        fence_timeout_sec: float = 2.0,
    ) -> None:
        if not isinstance(binding, RtcFrontendStepSourceBinding):
            raise TypeError("binding must be an RtcFrontendStepSourceBinding.")
        if not callable(observe_step):
            raise TypeError("observe_step must be callable.")
        capacity = int(queue_capacity)
        if capacity <= 0:
            raise ValueError("RTC Station queue_capacity must be positive.")
        timeout = float(fence_timeout_sec)
        if timeout <= 0.0:
            raise ValueError("fence_timeout_sec must be positive.")
        self.binding = binding
        if step_queue is not None and not isinstance(step_queue, queue.Queue):
            raise TypeError("step_queue must be a queue.Queue or None.")
        if step_queue is not None and int(step_queue.maxsize) <= 0:
            raise ValueError("An injected RTC Station step_queue must be bounded.")
        self.step_queue: queue.Queue[object] = (
            queue.Queue(maxsize=capacity) if step_queue is None else step_queue
        )
        self._observe_step = observe_step
        self._fence_timeout_sec = timeout
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._shadow_enabled = False
        self._fence_sequence = 0
        self._received_step_count = 0
        self._observed_step_count = 0
        self._observed_emission_count = 0
        self._inactive_step_count = 0
        self._rejected_step_count = 0
        self._error_count = 0
        self._last_error = ""
        self._last_state_hash: str | None = None
        self._last_emission_id: str | None = None

    def start(self) -> None:
        """Start only the passive observer worker; never the RTC producer."""

        with self._lock:
            if self._closed:
                raise RuntimeError("RTC Station consumer is closed.")
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="control-station-rtc-policy-consumer",
                daemon=True,
            )
            self._thread.start()

    def set_shadow_enabled(self, enabled: bool) -> int:
        """Apply an ordered local observation fence without touching RTC."""

        self.start()
        with self._lock:
            if self._closed:
                raise RuntimeError("RTC Station consumer is closed.")
            sequence = self._fence_sequence + 1
            fence = _RtcShadowFence(sequence=sequence, enabled=bool(enabled))
            try:
                self.step_queue.put_nowait(fence)
            except queue.Full as exc:
                raise RuntimeError("RTC Station shadow fence queue is full.") from exc
        if not fence.applied.wait(timeout=self._fence_timeout_sec):
            raise TimeoutError("RTC Station shadow fence was not applied in time.")
        return sequence

    def _run(self) -> None:
        while not self._stop.is_set() or not self.step_queue.empty():
            try:
                item = self.step_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if isinstance(item, _RtcShadowFence):
                    with self._lock:
                        self._shadow_enabled = item.enabled
                        self._fence_sequence = item.sequence
                    item.applied.set()
                    continue
                if not isinstance(item, RtcFrontendStep):
                    raise TypeError("RTC Station queue accepts RtcFrontendStep only.")
                with self._lock:
                    self._received_step_count += 1
                    enabled = self._shadow_enabled and not self._closed
                if not enabled:
                    with self._lock:
                        self._inactive_step_count += 1
                    continue
                emission = self.binding.validate_step(item)
                self._observe_step(item)
                with self._lock:
                    self._observed_step_count += 1
                    self._last_state_hash = item.state.semantic_content_hash
                    if emission is not None:
                        self._observed_emission_count += 1
                        self._last_emission_id = emission.emission_id
            except (TypeError, ValueError, RuntimeError) as exc:
                with self._lock:
                    self._rejected_step_count += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # passive observation cannot fault deploy
                with self._lock:
                    self._error_count += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.step_queue.task_done()

    def status(self) -> RtcPolicyStationConsumerStatus:
        with self._lock:
            return RtcPolicyStationConsumerStatus(
                started=self._thread is not None,
                closed=self._closed,
                shadow_enabled=self._shadow_enabled,
                queue_size=self.step_queue.qsize(),
                queue_capacity=self.step_queue.maxsize,
                received_step_count=self._received_step_count,
                observed_step_count=self._observed_step_count,
                observed_emission_count=self._observed_emission_count,
                inactive_step_count=self._inactive_step_count,
                rejected_step_count=self._rejected_step_count,
                error_count=self._error_count,
                fence_sequence=self._fence_sequence,
                last_error=self._last_error,
                last_state_hash=self._last_state_hash,
                last_emission_id=self._last_emission_id,
            )

    def close(self, *, timeout_sec: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            thread = self._thread
        if thread is not None:
            self.set_shadow_enabled(False)
        with self._lock:
            self._closed = True
            self._shadow_enabled = False
        self._stop.set()
        if thread is not None:
            thread.join(timeout=float(timeout_sec))
            if thread.is_alive():
                raise TimeoutError("RTC Station consumer did not stop before deadline.")


__all__ = [
    "RTC_POLICY_BINDING_HASH_DOMAIN_V1",
    "RtcPolicyEmissionBinding",
    "RtcPolicyStationConsumer",
    "RtcPolicyStationConsumerStatus",
]
