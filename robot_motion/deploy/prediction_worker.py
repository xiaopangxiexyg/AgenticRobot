"""Bounded non-blocking policy prediction worker.

Only model inference runs on the private thread.  The caller remains the sole
owner of inference-id reservation, prediction-result admission, action queues,
formal emission, safety and dispatch.
"""

from __future__ import annotations

import math
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .observation import DeploymentObservation
from .runtime import PolicyChunkPredictionCandidate, PolicyDeployRuntime


@dataclass(frozen=True)
class _PredictionTask:
    generation: int
    inference_id: str
    observation: DeploymentObservation


@dataclass(frozen=True)
class _PredictionCompletion:
    generation: int
    candidate: PolicyChunkPredictionCandidate | None
    error: BaseException | None = None


class _PredictionOnlyPolicy:
    """Expose prediction without manufacturing transaction capabilities."""

    def __init__(self, predict: Callable[[Any], Any]) -> None:
        self._predict = predict

    def predict(self, observation: Any) -> Any:
        return self._predict(observation)


class ThreadedPolicyPredictionWorker:
    """Execute at most one detached policy prediction at a time.

    ``reset_for_activation()`` advances a generation before requesting
    transport cancellation.  A completion from the prior generation is never
    returned to its caller and therefore cannot be admitted into a new control
    activation.
    """

    def __init__(
        self,
        *,
        source_id: str,
        runtime: PolicyDeployRuntime,
        policy: Any,
        lifecycle_policy: Any | None = None,
        transaction_policy: Any | None = None,
        reset_callback: Callable[[], None] | None = None,
        join_timeout_sec: float = 2.0,
    ) -> None:
        source = str(source_id).strip()
        if not source:
            raise ValueError("source_id must be non-empty.")
        if not isinstance(runtime, PolicyDeployRuntime):
            raise TypeError("runtime must be a PolicyDeployRuntime.")
        if not callable(getattr(policy, "predict", None)):
            raise TypeError("policy must expose predict(observation).")
        transaction_owner = (
            policy if transaction_policy is None else transaction_policy
        )
        transaction_hooks = tuple(
            hook
            for hook in (
                "begin_prediction_transaction",
                "commit_prediction_transaction",
                "rollback_prediction_transaction",
            )
            if callable(getattr(transaction_owner, hook, None))
        )
        if transaction_hooks:
            raise TypeError(
                "threaded prediction is incompatible with policy transaction "
                "hooks; use the synchronous queue: "
                + ",".join(transaction_hooks)
            )
        if reset_callback is not None and not callable(reset_callback):
            raise TypeError("reset_callback must be callable or None.")
        timeout = float(join_timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("join_timeout_sec must be finite and positive.")

        self.source_id = source
        self.runtime = runtime
        self.policy = _PredictionOnlyPolicy(policy.predict)
        self.lifecycle_policy = (
            policy if lifecycle_policy is None else lifecycle_policy
        )
        self.reset_callback = reset_callback
        self.join_timeout_sec = timeout

        self._tasks: queue.Queue[_PredictionTask | None] = queue.Queue(maxsize=1)
        self._completions: queue.Queue[_PredictionCompletion] = queue.Queue(
            maxsize=1
        )
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._started = False
        self._generation = 0
        self._in_flight = False
        self._reset_pending = False
        self._transport_cancelled = False
        self._last_error: str | None = None
        self._discarded_late_completion_count = 0

    @property
    def started(self) -> bool:
        return self._started

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    @property
    def reset_pending(self) -> bool:
        return self._reset_pending

    @property
    def ready(self) -> bool:
        return (
            self._started
            and not self._reset_pending
            and self._last_error is None
            and bool(self._worker and self._worker.is_alive())
        )

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def discarded_late_completion_count(self) -> int:
        return self._discarded_late_completion_count

    def _worker_loop(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            try:
                candidate = self.runtime.predict_chunk(
                    self.policy,
                    task.observation,
                    inference_id=task.inference_id,
                )
                completion = _PredictionCompletion(
                    generation=task.generation,
                    candidate=candidate,
                )
            except Exception as exc:  # noqa: BLE001 - report to owner thread.
                completion = _PredictionCompletion(
                    generation=task.generation,
                    candidate=None,
                    error=exc,
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
        hook = getattr(self.lifecycle_policy, "request_cancel", None)
        if not callable(hook):
            return
        accepted = hook()
        if accepted is False:
            raise RuntimeError("policy request_cancel() was rejected.")
        self._transport_cancelled = True

    def _reset_policy_session(self) -> bool:
        if self._transport_cancelled:
            begin = getattr(self.lifecycle_policy, "begin_session", None)
            if not callable(begin):
                raise RuntimeError(
                    "cancelled policy transport cannot begin a fresh session."
                )
            if begin() is False:
                self._last_error = "policy_session_rearm_pending"
                return False
            self._transport_cancelled = False
        reset = getattr(self.lifecycle_policy, "reset", None)
        if callable(reset):
            reset()
        if self.reset_callback is not None:
            self.reset_callback()
        self._reset_pending = False
        self._last_error = None
        return True

    def start(self) -> None:
        if self._started:
            return
        if self._worker is not None:
            raise RuntimeError("A closed prediction worker cannot be restarted.")
        self.reset_for_activation()
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"dexteleop-policy-prediction:{self.source_id}",
            daemon=True,
        )
        self._worker.start()
        self._started = True

    def reset_for_activation(self) -> None:
        """Invalidate current work and reset/rearm the policy session."""

        self._generation += 1
        self._last_error = None
        drained_completion = False
        while True:
            try:
                self._completions.get_nowait()
                drained_completion = True
            except queue.Empty:
                break
        if drained_completion:
            self._in_flight = False
            self._discarded_late_completion_count += 1
        if self._in_flight:
            self._reset_pending = True
            self._request_policy_cancel()
        else:
            self._reset_policy_session()

    def submit(
        self,
        observation: DeploymentObservation,
        *,
        inference_id: str,
    ) -> bool:
        """Submit one immutable observation without blocking the caller."""

        if not self._started:
            raise RuntimeError("ThreadedPolicyPredictionWorker.start() is required.")
        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation.")
        if self._in_flight or self._reset_pending:
            return False
        task = _PredictionTask(
            generation=self._generation,
            inference_id=str(inference_id),
            observation=observation,
        )
        try:
            self._tasks.put_nowait(task)
        except queue.Full as exc:
            raise RuntimeError("policy prediction request mailbox is full.") from exc
        self._in_flight = True
        return True

    def poll_completion(self) -> PolicyChunkPredictionCandidate | None:
        """Return one current-generation completion on the caller thread."""

        if not self._started:
            raise RuntimeError("ThreadedPolicyPredictionWorker.start() is required.")
        try:
            completion = self._completions.get_nowait()
        except queue.Empty:
            if self._reset_pending and not self._in_flight:
                self._reset_policy_session()
            return None

        self._in_flight = False
        if completion.generation != self._generation:
            self._discarded_late_completion_count += 1
            if self._reset_pending:
                self._reset_policy_session()
            return None
        if completion.error is not None:
            self._last_error = (
                f"{type(completion.error).__name__}:{completion.error}"
            )
            raise RuntimeError("policy prediction worker failed") from completion.error
        assert completion.candidate is not None
        self._last_error = None
        return completion.candidate

    def close(self) -> None:
        if self._worker is None:
            self._started = False
            return
        self._generation += 1
        cancel_error: BaseException | None = None
        if self._in_flight:
            try:
                self._request_policy_cancel()
            except Exception as exc:  # noqa: BLE001 - still drain and close.
                cancel_error = exc
        self._stop_event.set()
        try:
            while True:
                self._tasks.get_nowait()
        except queue.Empty:
            pass
        try:
            self._tasks.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=self.join_timeout_sec)
        alive = self._worker.is_alive()
        self._started = False
        close_error: BaseException | None = None
        close = getattr(self.lifecycle_policy, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - report after drain proof.
                close_error = exc
        if alive:
            raise RuntimeError(
                "policy inference worker did not drain before close timeout."
            )
        if cancel_error is not None:
            raise RuntimeError("policy inference cancellation failed") from cancel_error
        if close_error is not None:
            raise RuntimeError("policy lifecycle close failed") from close_error

    def status(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "started": self._started,
            "ready": self.ready,
            "worker_alive": bool(self._worker and self._worker.is_alive()),
            "in_flight": self._in_flight,
            "reset_pending": self._reset_pending,
            "activation_generation": self._generation,
            "discarded_late_completion_count": (
                self._discarded_late_completion_count
            ),
            "last_error": self._last_error,
        }


__all__ = ["ThreadedPolicyPredictionWorker"]
