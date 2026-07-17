from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .actions import ActionChunkBuffer, NormalizedAction, normalize_policy_output
from .observation import DeploymentObservation
from .pose_adapter import (
    convert_normalized_action_chunk_to_absolute,
    prepare_observation_for_policy_pose_mode,
)
from .session_contract import POLICY_ABSOLUTE_FRAMES, POLICY_ABSOLUTE_FRAME_RUNTIME_RAW


_CHUNK_ID_METADATA_KEY = "deploy_action_chunk_id"
_CHUNK_INDEX_METADATA_KEY = "deploy_action_chunk_index"
_CHUNK_SIZE_METADATA_KEY = "deploy_action_chunk_size"


class ActionPreparer(Protocol):
    """Converts a policy action into a resolved command boundary."""

    def prepare(self, action: Any, observation: Any, *, policy_name: str = "") -> Any:
        ...


class PolicyLike(Protocol):
    def predict(self, observation: Any) -> Any:
        ...


@dataclass(frozen=True)
class PolicyDeployRuntimeConfig:
    """Configuration for the shared policy-output preparation boundary.

    This is intentionally smaller than a launcher configuration.  It contains
    only the values needed to turn a policy observation/output into the
    canonical absolute deployment action.  Device construction, clocks,
    startup/recovery and dispatch remain frontend/backend responsibilities.
    """

    active_sides: tuple[str, ...]
    arms_enabled: bool = True
    hands_enabled: bool = False
    policy_action_pose_mode: str = "absolute"
    policy_absolute_frame: str = POLICY_ABSOLUTE_FRAME_RUNTIME_RAW
    runtime_pose_frame_contract: Any | None = None

    def __post_init__(self) -> None:
        sides = tuple(str(side).strip().lower() for side in self.active_sides)
        if not sides:
            raise ValueError("active_sides must not be empty.")
        if any(side not in {"left", "right"} for side in sides):
            raise ValueError("active_sides must contain only 'left' and/or 'right'.")
        if len(set(sides)) != len(sides):
            raise ValueError("active_sides must not contain duplicates.")
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(
            self,
            "policy_action_pose_mode",
            str(self.policy_action_pose_mode).strip().lower(),
        )
        object.__setattr__(
            self,
            "policy_absolute_frame",
            str(self.policy_absolute_frame).strip().lower(),
        )
        if self.policy_action_pose_mode not in {"absolute", "relative"}:
            raise ValueError(
                "policy_action_pose_mode must be 'absolute' or 'relative'."
            )
        if self.policy_absolute_frame not in POLICY_ABSOLUTE_FRAMES:
            raise ValueError(
                "policy_absolute_frame must be one of "
                f"{POLICY_ABSOLUTE_FRAMES!r}."
            )


@dataclass(frozen=True)
class PolicyChunkPredictionResult:
    """Result of one policy prediction/transaction and queue update.

    The result deliberately carries model output and queue lineage, but no
    hardware, canonical safety, or dispatch objects.  A real frontend can use
    the result for logging and operator policy while the shared runtime keeps
    transaction rollback and action-chunk ownership in one place.
    """

    policy_observation: DeploymentObservation | None = None
    policy_output: Any = None
    actions: tuple[NormalizedAction, ...] = ()
    accepted: bool = False
    discarded_by_gate: bool = False
    faults: tuple[str, ...] = ()
    queue_len_before: int = 0
    queue_len_after: int = 0
    error: BaseException | None = None
    rollback_error: BaseException | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "faults", tuple(str(fault) for fault in self.faults))
        object.__setattr__(self, "queue_len_before", int(self.queue_len_before))
        object.__setattr__(self, "queue_len_after", int(self.queue_len_after))

    @property
    def has_actions(self) -> bool:
        return bool(self.actions)

    @property
    def empty(self) -> bool:
        return not self.actions and self.error is None


@dataclass(frozen=True)
class PolicyActionPreparationResult:
    """One queued action after the shared canonical preparation boundary."""

    raw_action: NormalizedAction | None = None
    raw_lookahead_action: NormalizedAction | None = None
    prepared_action: Any = None
    queue_len_before: int = 0
    queue_len_after: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "queue_len_before", int(self.queue_len_before))
        object.__setattr__(self, "queue_len_after", int(self.queue_len_after))


def resolve_action_queue_policy(
    *,
    action_queue_policy: object = "auto",
    append_action_chunks: bool = False,
    debug_replay: bool = False,
) -> str:
    """Resolve the policy chunk queue mode without CLI/frontend dependencies."""

    if bool(append_action_chunks):
        return "append"
    policy = str(action_queue_policy or "auto").strip().lower()
    if policy == "auto":
        return "append" if bool(debug_replay) else "replace"
    if policy not in {"replace", "append"}:
        raise ValueError(f"Unsupported action queue policy: {policy!r}.")
    return policy


class PolicyDeployRuntime:
    """Shared policy observation/action representation boundary.

    The real launcher owns device observation, lifecycle, canonical safety
    configuration and backend dispatch.  This class owns the policy stream
    state that must not be reimplemented by each frontend:

    ``schedule -> policy transaction -> normalized output -> absolute chunk
    -> queue``.

    Keeping this boundary small is deliberate.  It lets the real launcher use
    the class without moving device code into ``robot_motion``.  Canonical
    safety, IK/reference execution and dispatch stay outside this class.
    """

    def __init__(self, config: PolicyDeployRuntimeConfig):
        if not isinstance(config, PolicyDeployRuntimeConfig):
            raise TypeError("config must be a PolicyDeployRuntimeConfig.")
        self.config = config
        self._action_queue = ActionChunkBuffer()
        self._last_prediction_at: float | None = None
        self._next_action_chunk_id = 0

    def action_queue_length(self) -> int:
        """Return the number of decoded actions waiting for execution."""

        return len(self._action_queue)

    def clear_action_queue(self) -> None:
        """Discard all queued policy actions."""

        self._action_queue.clear()

    def pop_action(self, *, allow_output: bool = True) -> NormalizedAction | None:
        """Pop the next decoded action when the frontend allows output."""

        if not allow_output:
            return None
        return self._action_queue.pop()

    def peek_action(self, *, allow_output: bool = True) -> NormalizedAction | None:
        """Return the next decoded action without consuming it."""

        if not allow_output:
            return None
        return self._action_queue.peek()

    @staticmethod
    def _same_chunk_successor(
        current: NormalizedAction | None,
        candidate: NormalizedAction | None,
    ) -> bool:
        if current is None or candidate is None:
            return False
        current_metadata = dict(current.metadata)
        candidate_metadata = dict(candidate.metadata)
        try:
            current_chunk = int(current_metadata[_CHUNK_ID_METADATA_KEY])
            candidate_chunk = int(candidate_metadata[_CHUNK_ID_METADATA_KEY])
            current_index = int(current_metadata[_CHUNK_INDEX_METADATA_KEY])
            candidate_index = int(candidate_metadata[_CHUNK_INDEX_METADATA_KEY])
        except (KeyError, TypeError, ValueError):
            return False
        return current_chunk == candidate_chunk and candidate_index == current_index + 1

    def prepare_next_action(
        self,
        observation: DeploymentObservation,
        *,
        preparer: ActionPreparer,
        policy_name: str = "",
        allow_output: bool = True,
    ) -> PolicyActionPreparationResult:
        """Pop one decoded action and prepare it through the canonical boundary."""

        queue_len_before = self.action_queue_length()
        raw_action = self.pop_action(allow_output=allow_output)
        queue_len_after = self.action_queue_length()
        queued_candidate = self.peek_action(allow_output=allow_output)
        raw_lookahead_action = (
            queued_candidate
            if self._same_chunk_successor(raw_action, queued_candidate)
            else None
        )
        prepared_action = None
        if raw_action is not None:
            prepared_action = preparer.prepare(
                raw_action,
                observation,
                policy_name=policy_name,
            )
        return PolicyActionPreparationResult(
            raw_action=raw_action,
            raw_lookahead_action=raw_lookahead_action,
            prepared_action=prepared_action,
            queue_len_before=queue_len_before,
            queue_len_after=queue_len_after,
        )

    def reset_prediction_schedule(self) -> None:
        """Reset the policy-rate clock after hold/init/re-arm boundaries."""

        self._last_prediction_at = None

    def should_predict(
        self,
        *,
        now: float,
        policy_interval_s: float,
        min_action_queue: int = 0,
    ) -> bool:
        """Return whether a new policy chunk should be requested now."""

        if self.action_queue_length() > int(min_action_queue):
            return False
        if self._last_prediction_at is None:
            return True
        return float(now) - self._last_prediction_at >= float(policy_interval_s)

    def prepare_policy_observation(
        self,
        observation: DeploymentObservation,
    ) -> DeploymentObservation:
        """Build the model-facing observation using the shared v2.3 adapter."""

        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation.")
        return prepare_observation_for_policy_pose_mode(
            observation,
            target_mode=self.config.policy_action_pose_mode,
            active_sides=self.config.active_sides,
            policy_absolute_frame=self.config.policy_absolute_frame,
            runtime_pose_frame_contract=self.config.runtime_pose_frame_contract,
        )

    def normalize_and_decode(
        self,
        policy_output: Any,
        observation: DeploymentObservation,
    ) -> list[NormalizedAction]:
        """Normalize one policy output and decode it to runtime absolute poses.

        The input observation remains the raw runtime observation for safety and
        anchor purposes.  The model-facing observation is prepared separately;
        this preserves the existing contract that relative chunks are anchored
        to the immutable measured TCP snapshot from the control cut.
        """

        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation.")
        actions = normalize_policy_output(
            policy_output,
            observation,
            active_sides=self.config.active_sides,
            arms_enabled=bool(self.config.arms_enabled),
            hands_enabled=bool(self.config.hands_enabled),
        )
        return convert_normalized_action_chunk_to_absolute(
            actions,
            observation,
            source_mode=self.config.policy_action_pose_mode,
            active_sides=self.config.active_sides,
            policy_absolute_frame=self.config.policy_absolute_frame,
            runtime_pose_frame_contract=self.config.runtime_pose_frame_contract,
        )

    def prepare_policy_output(
        self,
        policy_output: Any,
        observation: DeploymentObservation,
    ) -> tuple[DeploymentObservation, list[NormalizedAction]]:
        """Return the model view and decoded absolute action chunk together."""

        policy_observation = self.prepare_policy_observation(observation)
        actions = self.normalize_and_decode(policy_output, observation)
        return policy_observation, actions

    def predict_and_queue(
        self,
        policy: PolicyLike,
        observation: DeploymentObservation,
        *,
        now: float,
        queue_policy: str = "replace",
        discard_policy_output: bool = False,
    ) -> PolicyChunkPredictionResult:
        """Predict, decode, transactionally update the policy action queue.

        ``discard_policy_output`` is used by the shared IK recovery gate.  The
        prediction still advances the policy-rate timestamp, but its chunk is
        rolled back and never becomes executable.  Any exception clears the
        queue and is returned to the frontend so it can apply its existing
        pause/hold policy without duplicating transaction handling.
        """

        queue_mode = str(queue_policy).strip().lower()
        if queue_mode not in {"replace", "append"}:
            raise ValueError(f"Unsupported queue_policy: {queue_policy!r}.")
        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation.")

        queue_len_before = self.action_queue_length()
        policy_observation: DeploymentObservation | None = None
        policy_output: Any = None
        actions: list[NormalizedAction] = []
        checkpoint: Any = None
        transaction_started = False
        transaction_resolved = False

        try:
            policy_observation = self.prepare_policy_observation(observation)
            begin = getattr(policy, "begin_prediction_transaction", None)
            if callable(begin):
                checkpoint = begin()
                transaction_started = True
            policy_output = policy.predict(policy_observation)
            actions = self.normalize_and_decode(policy_output, observation)
            if actions:
                chunk_id = int(self._next_action_chunk_id)
                self._next_action_chunk_id += 1
                chunk_size = len(actions)
                actions = [
                    NormalizedAction(
                        vector=action.vector,
                        arm_joint_targets=dict(action.arm_joint_targets),
                        metadata={
                            **dict(action.metadata),
                            _CHUNK_ID_METADATA_KEY: chunk_id,
                            _CHUNK_INDEX_METADATA_KEY: index,
                            _CHUNK_SIZE_METADATA_KEY: chunk_size,
                        },
                    )
                    for index, action in enumerate(actions)
                ]
                # Preserve the existing behavior: a chunk discarded because
                # of recovery still consumes this policy-rate opportunity.
                self._last_prediction_at = float(now)
                if discard_policy_output:
                    self.clear_action_queue()
                    if transaction_started:
                        rollback = getattr(policy, "rollback_prediction_transaction")
                        rollback(checkpoint)
                        transaction_resolved = True
                    return PolicyChunkPredictionResult(
                        policy_observation=policy_observation,
                        policy_output=policy_output,
                        actions=tuple(actions),
                        discarded_by_gate=True,
                        queue_len_before=queue_len_before,
                        queue_len_after=self.action_queue_length(),
                    )

                if queue_mode == "append":
                    self._action_queue.append(actions)
                else:
                    self._action_queue.replace(actions)
                if transaction_started:
                    commit = getattr(policy, "commit_prediction_transaction")
                    commit(checkpoint)
                    transaction_resolved = True
                return PolicyChunkPredictionResult(
                    policy_observation=policy_observation,
                    policy_output=policy_output,
                    actions=tuple(actions),
                    accepted=True,
                    queue_len_before=queue_len_before,
                    queue_len_after=self.action_queue_length(),
                )

            if transaction_started:
                rollback = getattr(policy, "rollback_prediction_transaction")
                rollback(checkpoint)
                transaction_resolved = True
            return PolicyChunkPredictionResult(
                policy_observation=policy_observation,
                policy_output=policy_output,
                faults=("policy_empty_action",),
                queue_len_before=queue_len_before,
                queue_len_after=self.action_queue_length(),
            )
        except Exception as exc:  # noqa: BLE001 - frontend decides pause/hold policy.
            rollback_error: BaseException | None = None
            if transaction_started and not transaction_resolved:
                try:
                    rollback = getattr(policy, "rollback_prediction_transaction")
                    rollback(checkpoint)
                except Exception as rollback_exc:  # noqa: BLE001 - report both failures.
                    rollback_error = rollback_exc
            self.clear_action_queue()
            faults = ()
            if rollback_error is not None:
                faults = (
                    "policy_prediction_rollback_error:"
                    f"{type(rollback_error).__name__}",
                )
            return PolicyChunkPredictionResult(
                policy_observation=policy_observation,
                policy_output=policy_output,
                faults=faults,
                queue_len_before=queue_len_before,
                queue_len_after=self.action_queue_length(),
                error=exc,
                rollback_error=rollback_error,
            )


@dataclass(frozen=True)
class DeployRuntimeStepResult:
    observation: Any
    raw_action: Any = None
    prepared_action: Any = None
    command: Any = None
    sent: bool = False
    faults: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "faults", tuple(str(fault) for fault in self.faults))
        object.__setattr__(self, "metadata", dict(self.metadata))


class DeployRuntime:
    """Minimal common single-step runtime glue for sim/real deployment.

    Migration note:
    The full real-time loop currently remains in ``teleop_runtime.deploy``.
    This common class is intentionally small so future sim and real launchers
    can share the same source -> policy -> canonical pipeline -> dispatcher
    shape without moving the existing production loop in one large edit.
    """

    def __init__(
        self,
        *,
        source: Any,
        policy: PolicyLike,
        preparer: ActionPreparer,
        dispatcher: Any,
        policy_name: str = "",
        send_actions: bool = False,
    ) -> None:
        self.source = source
        self.policy = policy
        self.preparer = preparer
        self.dispatcher = dispatcher
        self.policy_name = str(policy_name or getattr(policy, "name", "") or "policy")
        self.send_actions = bool(send_actions)

    def step(self, *, last_action: Any = None, now: float | None = None) -> DeployRuntimeStepResult:
        observation = self.source.get_observation(last_action=last_action, now=now)
        raw_action = self.policy.predict(observation)
        prepared = self.preparer.prepare(raw_action, observation, policy_name=self.policy_name)
        faults = tuple(getattr(prepared, "faults", ()) or ())
        command = getattr(prepared, "command", None)
        sent = False
        if command is not None and not faults and self.send_actions:
            dispatch = getattr(self.dispatcher, "dispatch", None)
            if dispatch is None:
                dispatch = self.dispatcher
            dispatch(command)
            sent = True
        return DeployRuntimeStepResult(
            observation=observation,
            raw_action=raw_action,
            prepared_action=prepared,
            command=command,
            sent=sent,
            faults=faults,
            metadata={"policy_name": self.policy_name},
        )
