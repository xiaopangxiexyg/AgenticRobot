"""Backend-neutral non-blocking policy-to-emission source.

The model call runs on one private worker thread.  The caller/control thread is
the only owner of prediction-result admission, the decoded action queue, and
formal ``PolicyActionEmission`` creation.  This keeps a slow remote model out
of a control cycle without giving the worker execution or composition
authority.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .action_adapter import DeployRobotActionAdapter
from .actions import NormalizedAction
from .observation import DeploymentObservation
from .runtime import (
    PolicyActionEmission,
    PolicyChunkPredictionCandidate,
    PolicyChunkPredictionResult,
    PolicyDeployRuntime,
)


class PolicyObservationCapture(Protocol):
    """Capture one immutable observation on the caller thread."""

    def __call__(self, *, step_index: int | None = None) -> DeploymentObservation: ...


class DeployActionPreparer:
    """Expose the shared representation lift without owning canonical safety."""

    def __init__(self, adapter: DeployRobotActionAdapter) -> None:
        if not isinstance(adapter, DeployRobotActionAdapter):
            raise TypeError("adapter must be a DeployRobotActionAdapter.")
        self.action_adapter = adapter

    def adapt_pre_safety(
        self,
        action: NormalizedAction,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> Any:
        return self.action_adapter.adapt(
            action,
            observation,
            policy_name=policy_name,
            input_stage="decoded_absolute_pre_safety",
        )

    def prepare(
        self,
        action: NormalizedAction,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> Any:
        return self.adapt_pre_safety(
            action,
            observation,
            policy_name=policy_name,
        ).robot_action


@dataclass(frozen=True)
class ThreadedPolicyEmissionSourceConfig:
    """Scheduling/lifecycle values independent of model and backend family."""

    source_id: str
    policy_name: str
    policy_interval_sec: float
    action_period_sec: float
    maximum_hold_sec: float
    prediction_request_queue_threshold: int = 0
    worker_join_timeout_sec: float = 2.0
    status_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source_id", "policy_name"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        for name in (
            "policy_interval_sec",
            "action_period_sec",
            "maximum_hold_sec",
            "worker_join_timeout_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        threshold = int(self.prediction_request_queue_threshold)
        if threshold < 0 or threshold != self.prediction_request_queue_threshold:
            raise ValueError(
                "prediction_request_queue_threshold must be a non-negative integer."
            )
        object.__setattr__(self, "prediction_request_queue_threshold", threshold)
        object.__setattr__(self, "status_metadata", dict(self.status_metadata or {}))


@dataclass(frozen=True)
class _PredictionTask:
    generation: int
    inference_id: str
    observation: DeploymentObservation


@dataclass(frozen=True)
class _PredictionCompletion:
    generation: int
    candidate: PolicyChunkPredictionCandidate


class ThreadedPolicyEmissionSource:
    """Produce sequential emissions while model inference remains non-blocking.

    The source never rewrites an already queued current chunk.  A prefetched
    result is appended behind remaining admitted keyframes; if no keyframe is
    left it replaces the empty queue.  Reset increments a generation before
    requesting transport cancellation, so a late completion cannot enter the
    new activation.
    """

    def __init__(
        self,
        config: ThreadedPolicyEmissionSourceConfig,
        *,
        runtime: PolicyDeployRuntime,
        policy: Any,
        observation_capture: PolicyObservationCapture,
        action_preparer: Any,
        observation_reset: Callable[[], None] | None = None,
        last_action_sink: Callable[[NormalizedAction], None] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, ThreadedPolicyEmissionSourceConfig):
            raise TypeError("config must be a ThreadedPolicyEmissionSourceConfig.")
        if not isinstance(runtime, PolicyDeployRuntime):
            raise TypeError("runtime must be a PolicyDeployRuntime.")
        if not callable(getattr(policy, "predict", None)):
            raise TypeError("policy must expose predict(observation).")
        if not callable(observation_capture):
            raise TypeError("observation_capture must be callable.")
        if not callable(getattr(action_preparer, "prepare", None)):
            raise TypeError("action_preparer must expose prepare(...).")
        if observation_reset is not None and not callable(observation_reset):
            raise TypeError("observation_reset must be callable or None.")
        if last_action_sink is not None and not callable(last_action_sink):
            raise TypeError("last_action_sink must be callable or None.")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable.")
        if not math.isclose(
            runtime.config.action_keyframe_period_sec,
            config.action_period_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "source action_period_sec must match PolicyDeployRuntime timing."
            )
        if not math.isclose(
            runtime.config.action_maximum_hold_sec,
            config.maximum_hold_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "source maximum_hold_sec must match PolicyDeployRuntime timing."
            )
        self.config = config
        self.runtime = runtime
        self.policy = policy
        self.observation_capture = observation_capture
        self.action_preparer = action_preparer
        self.observation_reset = observation_reset
        self.last_action_sink = last_action_sink
        self.clock = monotonic_clock

        self._tasks: queue.Queue[_PredictionTask | None] = queue.Queue(maxsize=1)
        self._completions: queue.Queue[_PredictionCompletion] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._started = False
        self._generation = 0
        self._in_flight = False
        self._reset_pending = False
        self._transport_cancelled = False
        self._last_submission_at: float | None = None
        self._next_emission_due_at: float | None = None
        self._last_emission: PolicyActionEmission | None = None
        self._last_prediction_result: PolicyChunkPredictionResult | None = None
        self._last_error: str | None = None
        self._prediction_failure_count = 0
        self._discarded_late_completion_count = 0

    @staticmethod
    def _finite_time(value: object, *, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite.")
        return result

    @property
    def started(self) -> bool:
        return self._started

    @property
    def ready(self) -> bool:
        return self._started and self._last_error is None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_emission(self) -> PolicyActionEmission | None:
        return self._last_emission

    @property
    def maximum_hold_sec(self) -> float:
        return self.config.maximum_hold_sec

    def _worker_loop(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            candidate = self.runtime.predict_chunk(
                self.policy,
                task.observation,
                inference_id=task.inference_id,
            )
            completion = _PredictionCompletion(
                generation=task.generation,
                candidate=candidate,
            )
            while not self._stop_event.is_set():
                try:
                    self._completions.put(completion, timeout=0.05)
                    break
                except queue.Full:
                    continue
            if self._stop_event.is_set():
                return

    def _request_policy_cancel(self) -> None:
        hook = getattr(self.policy, "request_cancel", None)
        if callable(hook):
            accepted = hook()
            if accepted is False:
                raise RuntimeError("policy request_cancel() was rejected.")
            self._transport_cancelled = True

    def _reset_policy_session(self) -> bool:
        if self._transport_cancelled:
            begin = getattr(self.policy, "begin_session", None)
            if not callable(begin):
                raise RuntimeError(
                    "cancelled policy transport cannot begin a fresh session."
                )
            if begin() is False:
                # Transport close is asynchronous.  Keep the activation
                # pending and retry from a later caller cycle; never wait on
                # socket cleanup in this method.
                self._last_error = "policy_session_rearm_pending"
                return False
            self._transport_cancelled = False
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        if self.observation_reset is not None:
            self.observation_reset()
        self._reset_pending = False
        self._last_error = None
        return True

    def start(self) -> None:
        if self._started:
            return
        if self._worker is not None:
            raise RuntimeError("A closed policy emission source cannot be restarted.")
        self.reset_for_activation()
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"dexteleop-policy-source:{self.config.source_id}",
            daemon=True,
        )
        self._worker.start()
        self._started = True

    def reset_for_activation(self) -> None:
        self._generation += 1
        self.runtime.clear_action_queue()
        self.runtime.reset_prediction_schedule()
        self._last_submission_at = None
        self._next_emission_due_at = None
        self._last_emission = None
        self._last_prediction_result = None
        self._last_error = None
        if self._in_flight:
            self._reset_pending = True
            self._request_policy_cancel()
        else:
            self._reset_policy_session()

    def _drain_completion(self, *, now: float) -> None:
        completion = None
        while True:
            try:
                completion = self._completions.get_nowait()
            except queue.Empty:
                break
        if completion is None:
            return
        self._in_flight = False
        if completion.generation != self._generation:
            self._discarded_late_completion_count += 1
            if self._reset_pending:
                self._reset_policy_session()
            return
        queue_len_before = self.runtime.action_queue_length()
        if completion.candidate.error is not None and queue_len_before > 0:
            # A failed prefetch is not allowed to revoke the already admitted
            # current chunk.  ``admit_prediction_result`` intentionally keeps
            # the older fail-closed replace behavior, so record the detached
            # failure here and let the existing tail finish in order.
            candidate = completion.candidate
            result = PolicyChunkPredictionResult(
                policy_observation=candidate.policy_observation,
                actor_observation=candidate.actor_observation,
                observation_parity=candidate.observation_parity,
                policy_output=candidate.policy_output,
                actions=candidate.actions,
                queue_len_before=queue_len_before,
                queue_len_after=queue_len_before,
                source_chunk_size=candidate.source_chunk_size,
                executed_chunk_size=0,
                action_chunk_execution_steps=(
                    candidate.action_chunk_execution_steps
                ),
                error=candidate.error,
            )
            self._last_prediction_result = result
            self._prediction_failure_count += 1
            self._last_error = (
                f"{type(candidate.error).__name__}:{candidate.error}"
            )
            return
        queue_mode = (
            "append" if queue_len_before > 0 else "replace"
        )
        result = self.runtime.admit_prediction_result(
            completion.candidate,
            now=now,
            queue_policy=queue_mode,
            action_keyframe_period_sec=self.config.action_period_sec,
        )
        self._last_prediction_result = result
        if result.error is not None or result.faults:
            self._prediction_failure_count += 1
            error = result.error or ",".join(result.faults)
            self._last_error = (
                f"{type(error).__name__}:{error}"
                if isinstance(error, BaseException)
                else str(error)
            )
            return
        if result.accepted and self._next_emission_due_at is None:
            self._next_emission_due_at = now

    def _maybe_submit_prediction(
        self,
        observation: DeploymentObservation,
        *,
        now: float,
    ) -> None:
        if self._in_flight or self._reset_pending:
            return
        if (
            self.runtime.action_queue_length()
            > self.config.prediction_request_queue_threshold
        ):
            return
        if (
            self._last_submission_at is not None
            and now - self._last_submission_at < self.config.policy_interval_sec
        ):
            return
        task = _PredictionTask(
            generation=self._generation,
            inference_id=self.runtime.reserve_prediction_inference_id(),
            observation=observation,
        )
        try:
            self._tasks.put_nowait(task)
        except queue.Full as exc:
            raise RuntimeError("policy prediction request mailbox is full.") from exc
        self._in_flight = True
        self._last_submission_at = now

    def step(
        self,
        *,
        now: float | None = None,
        step_index: int | None = None,
    ) -> PolicyActionEmission | None:
        if not self._started:
            raise RuntimeError("ThreadedPolicyEmissionSource.start() is required.")
        timestamp = self._finite_time(
            self.clock() if now is None else now,
            name="now",
        )
        self._drain_completion(now=timestamp)
        if self._reset_pending and not self._in_flight:
            self._reset_policy_session()
        observation = self.observation_capture(step_index=step_index)
        self._maybe_submit_prediction(observation, now=timestamp)

        due = self._next_emission_due_at
        if (
            due is None
            or timestamp + 1e-12 < due
            or self.runtime.action_queue_length() == 0
        ):
            return None
        preparation = self.runtime.prepare_next_action(
            observation,
            preparer=self.action_preparer,
            policy_name=self.config.policy_name,
            same_chunk_next_step_will_execute=True,
        )
        emission = preparation.emission
        if emission is None:
            if preparation.emission_error is not None:
                self._last_error = (
                    f"{type(preparation.emission_error).__name__}:"
                    f"{preparation.emission_error}"
                )
            return None
        if emission.pre_safety_faults or (
            emission.source_validation is None
            or not emission.source_validation.accepted
        ):
            self._last_error = (
                "policy pre-safety representation adapter rejected the action"
            )
            return None
        self._last_emission = emission
        self._last_error = None
        self._next_emission_due_at = max(timestamp, due) + self.config.action_period_sec
        if self.last_action_sink is not None and emission.decoded_absolute_action is not None:
            self.last_action_sink(emission.decoded_absolute_action)
        return emission

    def close(self) -> None:
        if self._worker is None:
            self._started = False
            return
        self._generation += 1
        if self._in_flight:
            self._request_policy_cancel()
        self._stop_event.set()
        try:
            self._tasks.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=self.config.worker_join_timeout_sec)
        alive = self._worker.is_alive()
        self._started = False
        self.runtime.clear_action_queue()
        self._last_emission = None
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()
        if alive:
            raise RuntimeError(
                "policy inference worker did not drain before close timeout."
            )

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        timestamp = self._finite_time(
            self.clock() if now is None else now,
            name="status.now",
        )
        lineage = (
            None
            if self._last_prediction_result is None
            else self._last_prediction_result.chunk_lineage
        )
        result = {
            "source_id": self.config.source_id,
            "policy_name": self.config.policy_name,
            "started": self._started,
            "ready": self.ready,
            "worker_alive": bool(self._worker and self._worker.is_alive()),
            "in_flight": self._in_flight,
            "activation_generation": self._generation,
            "queue_length": self.runtime.action_queue_length(),
            "policy_interval_sec": self.config.policy_interval_sec,
            "action_period_sec": self.config.action_period_sec,
            "last_emission_age_sec": (
                None
                if self._last_emission is None
                else timestamp - self._last_emission.emitted_at
            ),
            "last_inference_duration_sec": (
                None
                if lineage is None
                else lineage.inference_completed_at - lineage.inference_started_at
            ),
            "discarded_late_completion_count": (
                self._discarded_late_completion_count
            ),
            "prediction_failure_count": self._prediction_failure_count,
            "model_observation_sides": tuple(
                self.runtime.config.model_observation_sides
            ),
            "output_active_sides": tuple(self.runtime.config.active_sides),
            "last_error": self._last_error,
        }
        result.update(dict(self.config.status_metadata))
        return result


__all__ = [
    "DeployActionPreparer",
    "PolicyObservationCapture",
    "ThreadedPolicyEmissionSource",
    "ThreadedPolicyEmissionSourceConfig",
]
