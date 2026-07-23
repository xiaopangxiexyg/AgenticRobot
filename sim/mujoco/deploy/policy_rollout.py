"""Model-neutral policy rollout composition for MuJoCo.

This module is deliberately a thin application seam between three already
versioned contracts:

* a read-only observation source (for example
  :class:`MujocoDeploymentObservationAdapter`),
* a model adapter (for example ``OpenPISimAdapter``), and
* the authoritative :class:`MujocoDeployRuntime`.

It does not import an OpenPI client, construct an IK profile, or know how a
particular model names its tensors.  The model output is decoded before it
reaches this class; every action in one inference chunk is adapted against the
same observation cut and submitted atomically.  A malformed action therefore
cannot result in a partially admitted chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from robot_motion.action import RobotAction
from robot_motion.deploy import DeployRobotActionAdapter, DeploymentObservation


class MujocoPolicyRolloutError(RuntimeError):
    """Raised when one simulation policy cycle cannot be admitted safely."""


class DeploymentObservationSource(Protocol):
    """Read-only source of one common deployment observation cut."""

    def capture(self, **kwargs: Any) -> DeploymentObservation:
        ...


class DecodedPolicy(Protocol):
    """Model adapter returning a decoded action chunk."""

    def predict(self, observation: DeploymentObservation) -> Any:
        ...

    def reset(self) -> None:
        ...


class DeployRuntime(Protocol):
    """Small runtime surface required by the simulation composition seam."""

    def start(self) -> None:
        ...

    def close(self) -> None:
        ...

    def submit_policy_action_chunk(
        self,
        actions: Sequence[RobotAction],
        *,
        policy_output_id: str,
        chunk_id: str | None = None,
        observation_cut_id: str | None = None,
    ) -> Any:
        ...

    def step(self, *, physics_ticks_after_dispatch: int = 0) -> Any:
        ...


@dataclass(frozen=True)
class MujocoPolicyRolloutConfig:
    """Lifecycle and trace defaults for one simulation deployment owner."""

    policy_output_prefix: str = "mujoco-openpi"
    physics_ticks_after_dispatch: int = 0
    fault_on_adapter_error: bool = True
    capture_last_action: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prefix = str(self.policy_output_prefix).strip()
        if not prefix:
            raise ValueError("policy_output_prefix must be non-empty.")
        ticks = int(self.physics_ticks_after_dispatch)
        if ticks < 0:
            raise ValueError("physics_ticks_after_dispatch must be non-negative.")
        object.__setattr__(self, "policy_output_prefix", prefix)
        object.__setattr__(self, "physics_ticks_after_dispatch", ticks)
        object.__setattr__(self, "fault_on_adapter_error", bool(self.fault_on_adapter_error))
        object.__setattr__(self, "capture_last_action", bool(self.capture_last_action))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class MujocoPolicyCycleResult:
    """Evidence from one observation→inference→admission→step cycle."""

    cycle_index: int
    observation: DeploymentObservation
    prediction: Any
    actions: tuple[RobotAction, ...]
    submission: Any
    execution: Any
    policy_output_id: str
    chunk_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cycle = int(self.cycle_index)
        if cycle < 0:
            raise ValueError("cycle_index must be non-negative.")
        actions = tuple(self.actions)
        if not actions or any(not isinstance(action, RobotAction) for action in actions):
            raise ValueError("actions must contain at least one RobotAction.")
        for name in ("policy_output_id", "chunk_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        object.__setattr__(self, "cycle_index", cycle)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "policy_output_id", str(self.policy_output_id))
        object.__setattr__(self, "chunk_id", str(self.chunk_id))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _prediction_actions(prediction: Any) -> tuple[Any, ...]:
    """Extract a model-neutral action chunk without importing a model adapter."""

    actions = getattr(prediction, "actions", prediction)
    if isinstance(actions, (str, bytes)):
        raise MujocoPolicyRolloutError("policy prediction actions must be a sequence.")
    try:
        values = tuple(actions)
    except TypeError as exc:
        raise MujocoPolicyRolloutError(
            "policy prediction must expose an iterable actions chunk."
        ) from exc
    if not values:
        raise MujocoPolicyRolloutError("policy prediction returned an empty action chunk.")
    return values


def _prediction_metadata(prediction: Any) -> dict[str, Any]:
    value = getattr(prediction, "metadata", {})
    return dict(value) if isinstance(value, Mapping) else {}


class MujocoPolicyRollout:
    """Own one sim-only model deployment lifecycle.

    ``MujocoPolicyRollout`` is intentionally synchronous.  The policy may
    return a chunk, but the chunk is admitted once and the authoritative
    runtime decides which keyframe is due.  This keeps timing/authority in the
    runtime rather than duplicating it in a model-specific client.
    """

    def __init__(
        self,
        observation_source: DeploymentObservationSource,
        policy: DecodedPolicy,
        action_adapter: DeployRobotActionAdapter,
        runtime: DeployRuntime,
        *,
        config: MujocoPolicyRolloutConfig | None = None,
    ) -> None:
        if not hasattr(observation_source, "capture") or not callable(observation_source.capture):
            raise TypeError("observation_source must expose capture(...).")
        if not hasattr(policy, "predict") or not callable(policy.predict):
            raise TypeError("policy must expose predict(observation).")
        if not isinstance(action_adapter, DeployRobotActionAdapter):
            raise TypeError("action_adapter must be DeployRobotActionAdapter.")
        for name in ("start", "close", "submit_policy_action_chunk", "step"):
            if not hasattr(runtime, name) or not callable(getattr(runtime, name)):
                raise TypeError(f"runtime must expose callable {name}(...).")
        self.observation_source = observation_source
        self.policy = policy
        self.action_adapter = action_adapter
        self.runtime = runtime
        self.config = config or MujocoPolicyRolloutConfig()
        self._started = False
        self._cycle_index = 0

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            return
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        self.runtime.start()
        self._started = True

    def _fault(self, message: str) -> MujocoPolicyRolloutError:
        if self.config.fault_on_adapter_error and self._started:
            transition = getattr(self.runtime, "transition_execution_gate", None)
            if callable(transition):
                try:
                    transition(
                        "fault",
                        reason=message,
                        force_cancellation=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive safety path
                    message = f"{message}; fault_transition_failed:{type(exc).__name__}:{exc}"
        return MujocoPolicyRolloutError(message)

    def run_cycle(self) -> MujocoPolicyCycleResult:
        """Capture, infer, adapt, admit, and advance one sim control cycle."""

        if not self._started:
            raise RuntimeError("MujocoPolicyRollout.start() is required.")
        cycle = self._cycle_index
        observation = self.observation_source.capture()
        if not isinstance(observation, DeploymentObservation):
            raise self._fault("observation source returned a non-DeploymentObservation value")

        try:
            prediction = self.policy.predict(observation)
            raw_actions = _prediction_actions(prediction)
        except MujocoPolicyRolloutError:
            raise
        except Exception as exc:
            raise self._fault(f"policy inference failed:{type(exc).__name__}:{exc}") from exc

        adapted: list[RobotAction] = []
        faults: list[str] = []
        for index, raw_action in enumerate(raw_actions):
            try:
                result = self.action_adapter.adapt(
                    raw_action,
                    observation,
                    policy_name=self.config.policy_output_prefix,
                )
            except Exception as exc:
                faults.append(f"action_{index}:adapter_exception:{type(exc).__name__}:{exc}")
                continue
            if result.robot_action is None or result.faults:
                reasons = ",".join(str(item) for item in result.faults) or "no_robot_action"
                faults.append(f"action_{index}:{reasons}")
                continue
            adapted.append(result.robot_action)

        if faults:
            raise self._fault("policy chunk rejected before admission: " + " | ".join(faults))
        if len(adapted) != len(raw_actions):
            raise self._fault("policy chunk adaptation was not lossless")

        policy_output_id = f"{self.config.policy_output_prefix}:{cycle}"
        prediction_metadata = _prediction_metadata(prediction)
        chunk_id = str(prediction_metadata.get("chunk_id") or f"{policy_output_id}:chunk")
        observation_cut_id = f"mujoco-observation:{observation.step_index}"
        try:
            submission = self.runtime.submit_policy_action_chunk(
                adapted,
                policy_output_id=policy_output_id,
                chunk_id=chunk_id,
                observation_cut_id=observation_cut_id,
            )
            execution = self.runtime.step(
                physics_ticks_after_dispatch=self.config.physics_ticks_after_dispatch,
            )
        except Exception as exc:
            raise self._fault(f"runtime admission/step failed:{type(exc).__name__}:{exc}") from exc

        # The common bridge accepts the final decoded 26D vector as the causal
        # last-action field.  It is optional and only updated after admission.
        if self.config.capture_last_action:
            set_last_action = getattr(self.observation_source, "set_last_action", None)
            final_decoded = getattr(raw_actions[-1], "vector", None)
            if final_decoded is None:
                try:
                    candidate = np.asarray(raw_actions[-1], dtype=np.float32).reshape(-1)
                except Exception:
                    candidate = np.empty((0,), dtype=np.float32)
                if candidate.shape == (26,) and np.all(np.isfinite(candidate)):
                    final_decoded = candidate
            if callable(set_last_action) and final_decoded is not None:
                try:
                    set_last_action(final_decoded)
                except Exception as exc:
                    raise self._fault(
                        f"observation last_action update failed:{type(exc).__name__}:{exc}"
                    ) from exc

        self._cycle_index += 1
        return MujocoPolicyCycleResult(
            cycle_index=cycle,
            observation=observation,
            prediction=prediction,
            actions=tuple(adapted),
            submission=submission,
            execution=execution,
            policy_output_id=policy_output_id,
            chunk_id=chunk_id,
            metadata={
                **self.config.metadata,
                **prediction_metadata,
                "observation_cut_id": observation_cut_id,
                "raw_chunk_size": len(raw_actions),
                "adapted_chunk_size": len(adapted),
            },
        )

    def close(self) -> None:
        if not self._started:
            return
        try:
            self.runtime.close()
        finally:
            self._started = False


__all__ = [
    "DecodedPolicy",
    "DeployRuntime",
    "DeploymentObservationSource",
    "MujocoPolicyCycleResult",
    "MujocoPolicyRollout",
    "MujocoPolicyRolloutConfig",
    "MujocoPolicyRolloutError",
]
