from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import RobotAction, SourceValidationResult
from robot_motion.action.proposal import robot_action_semantic_payload
from robot_motion.action.snapshot import ActionEmissionScheduleSpec
from robot_motion.runtime.immutable import deep_freeze

from .actions import ActionChunkBuffer, NormalizedAction, normalize_policy_output
from .action_adapter import (
    DeployActionRepresentationReceipt,
    DeployRobotActionAdapterResult,
)
from .chunk import ActionChunkExecutionPolicy
from .normalization import (
    NormalizationContractError,
    NormalizationContract,
    explicit_identity_normalization_contract,
    normalization_contract_hash,
)
from .chunk_plan import TimedPolicyActionKeyframe
from .canonical_policy_action import CanonicalPolicyAction
from .observation import DeploymentObservation
from .observation_view import (
    PolicyObservationContract,
    PolicyObservationParityReceipt,
    PolicyObservationView,
    build_policy_observation_view,
    compare_policy_observation_view,
    default_deploy_policy_observation_contract,
)
from .pose_adapter import (
    convert_normalized_action_chunk_to_absolute,
    prepare_observation_for_policy_pose_mode,
)
from .session_contract import POLICY_ABSOLUTE_FRAMES, POLICY_ABSOLUTE_FRAME_RUNTIME_RAW


_CHUNK_ID_METADATA_KEY = "deploy_action_chunk_id"
_CHUNK_INDEX_METADATA_KEY = "deploy_action_chunk_index"
_CHUNK_SIZE_METADATA_KEY = "deploy_action_chunk_size"
_CHUNK_SOURCE_SIZE_METADATA_KEY = "deploy_action_chunk_source_size"
_CHUNK_EXECUTION_STEPS_METADATA_KEY = "deploy_action_chunk_execution_steps"
POLICY_ACTION_EMISSION_ADMISSION_MODE_V1 = "on_emission_pop_v1"
POLICY_CHUNK_LINEAGE_HASH_DOMAIN_V1 = "dexteleop.policy_chunk_lineage.v1"
POLICY_ACTION_EMISSION_HASH_DOMAIN_V1 = "dexteleop.policy_action_emission.v1"
POLICY_ACTION_EMISSION_HASH_DOMAIN_V2 = "dexteleop.policy_action_emission.v2"


class ActionPreparer(Protocol):
    """Converts a policy action into a resolved command boundary."""

    def prepare(self, action: Any, observation: Any, *, policy_name: str = "") -> Any:
        ...


class PolicyLike(Protocol):
    def predict(self, observation: Any) -> Any:
        ...


class PolicyPredictionTransactionRequiresSynchronousQueueError(RuntimeError):
    """Raised when a transactional policy is sent through the split seam."""


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
    # Numeric normalization is declared here so the generic runtime cannot
    # silently drop it when an adapter is converted into this config.  The
    # actual owner (client/server/none) remains explicit in the contract;
    # pose-frame conversion below is deliberately a separate concern.
    normalization_contract: Mapping[str, Any] = field(
        default_factory=explicit_identity_normalization_contract
    )
    normalization_hash: str | None = None
    policy_observation_contract: PolicyObservationContract | None = None
    policy_observation_binding_id: str = "legacy_deploy_v23"
    observation_firewall_mode: str = "strict"
    policy_node_id: str = "single_policy"
    policy_clock_domain_id: str = "local_monotonic"
    action_keyframe_period_sec: float = 1.0 / 15.0
    action_maximum_hold_sec: float = 1.0 / 15.0
    action_emission_admission_mode: str = POLICY_ACTION_EMISSION_ADMISSION_MODE_V1
    # ``None`` preserves the historical full-chunk behavior.  An explicit
    # positive value selects only the prefix before the next prediction.
    action_chunk_execution_steps: int | None = None

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
        if self.normalization_contract is None:
            raw_normalization = explicit_identity_normalization_contract()
        elif not isinstance(self.normalization_contract, Mapping):
            raise TypeError("normalization_contract must be a mapping or None.")
        else:
            raw_normalization = dict(self.normalization_contract)
        if not raw_normalization:
            raise ValueError("normalization_contract must not be empty.")
        try:
            parsed_normalization = NormalizationContract.from_optional_mapping(
                raw_normalization,
                allow_legacy=True,
            )
        except NormalizationContractError as exc:
            raise ValueError(str(exc)) from exc
        if parsed_normalization is not None:
            raw_normalization = parsed_normalization.to_dict()
            if parsed_normalization.has_client_owned_transform:
                raise ValueError(
                    "PolicyDeployRuntime does not execute client-owned numeric "
                    "normalization; apply it in the model adapter and pass the "
                    "post-transform contract instead."
                )
        expected_normalization_hash = normalization_contract_hash(raw_normalization)
        supplied_normalization_hash = self.normalization_hash
        if supplied_normalization_hash is not None:
            supplied_normalization_hash = str(supplied_normalization_hash).strip()
            if not supplied_normalization_hash:
                raise ValueError("normalization_hash must be non-empty when supplied.")
            if supplied_normalization_hash != expected_normalization_hash:
                raise ValueError(
                    "normalization_hash does not match normalization_contract contents."
                )
        else:
            supplied_normalization_hash = expected_normalization_hash
        object.__setattr__(self, "normalization_contract", raw_normalization)
        object.__setattr__(self, "normalization_hash", supplied_normalization_hash)
        mode = str(self.observation_firewall_mode).strip().lower().replace("-", "_")
        if mode not in {"strict", "dual_validate"}:
            raise ValueError("observation_firewall_mode must be strict or dual_validate.")
        object.__setattr__(self, "observation_firewall_mode", mode)
        policy_node_id = str(self.policy_node_id).strip()
        policy_clock_domain_id = str(self.policy_clock_domain_id).strip()
        if not policy_node_id or not policy_clock_domain_id:
            raise ValueError("policy_node_id and policy_clock_domain_id must be non-empty.")
        period = float(self.action_keyframe_period_sec)
        maximum_hold = float(self.action_maximum_hold_sec)
        if not math.isfinite(period) or period <= 0:
            raise ValueError("action_keyframe_period_sec must be finite and positive.")
        if not math.isfinite(maximum_hold) or maximum_hold <= 0:
            raise ValueError("action_maximum_hold_sec must be finite and positive.")
        if self.action_emission_admission_mode != POLICY_ACTION_EMISSION_ADMISSION_MODE_V1:
            raise ValueError("M2/M3 require on_emission_pop_v1 admission mode.")
        chunk_execution_policy = ActionChunkExecutionPolicy(
            self.action_chunk_execution_steps
        )
        object.__setattr__(self, "policy_node_id", policy_node_id)
        object.__setattr__(self, "policy_clock_domain_id", policy_clock_domain_id)
        object.__setattr__(self, "action_keyframe_period_sec", period)
        object.__setattr__(self, "action_maximum_hold_sec", maximum_hold)
        object.__setattr__(
            self,
            "action_chunk_execution_steps",
            chunk_execution_policy.execution_steps,
        )
        contract = self.policy_observation_contract
        if contract is None:
            contract = default_deploy_policy_observation_contract(
                binding_id=str(self.policy_observation_binding_id),
                last_action_pose_mode=self.policy_action_pose_mode,
            )
        elif not isinstance(contract, PolicyObservationContract):
            raise TypeError("policy_observation_contract must be a PolicyObservationContract.")
        if contract.last_action_contract is None:
            raise ValueError("Deploy actor contract must bind last_action semantics.")
        if contract.last_action_contract.pose_mode != self.policy_action_pose_mode:
            raise ValueError(
                "Policy observation last-action pose mode must match policy_action_pose_mode."
            )
        object.__setattr__(self, "policy_observation_contract", contract)


@dataclass(frozen=True)
class PolicyChunkLineage:
    inference_id: str
    policy_output_id: str
    policy_node_id: str
    chunk_id: int
    inference_started_at: float
    inference_completed_at: float
    produced_clock_domain_id: str
    observation_timestamp: float
    observation_step_index: int
    schedule: ActionEmissionScheduleSpec
    semantic_content_hash: str


@dataclass(frozen=True)
class PolicyActionEmission:
    emission_id: str
    admission_mode: str
    policy_node_id: str
    inference_id: str
    policy_output_id: str
    chunk_id: int
    chunk_keyframe_index: int
    chunk_size: int
    decoded_absolute_action: NormalizedAction | None
    pre_safety_robot_action: RobotAction | None
    source_validation: SourceValidationResult | None
    representation_receipt: DeployActionRepresentationReceipt | None
    pre_safety_faults: tuple[str, ...]
    produced_at: float
    produced_clock_domain_id: str
    emitted_at: float
    target_effective_at: float
    effective_from: float
    effective_until: float
    effective_clock_domain_id: str
    schedule_spec_id: str
    schedule_spec_hash: str
    queue_len_before: int
    queue_len_after: int
    queue_pop_count: int
    semantic_content_hash: str
    canonical_action: CanonicalPolicyAction | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pre_safety_faults", tuple(self.pre_safety_faults))
        if (self.decoded_absolute_action is None) == (self.canonical_action is None):
            raise ValueError(
                "A policy action emission must carry exactly one action payload."
            )
        if self.canonical_action is not None:
            if self.representation_receipt is not None:
                raise ValueError(
                    "A typed policy action emission cannot carry a legacy "
                    "representation receipt."
                )
            if self.pre_safety_robot_action is None:
                raise ValueError(
                    "A typed policy action emission requires its canonical "
                    "pre-safety RobotAction."
                )

    @property
    def action_payload_kind(self) -> str:
        return (
            "typed_canonical_policy_action_v1"
            if self.canonical_action is not None
            else "legacy_normalized_absolute_v1"
        )

    @property
    def representation_trace_id(self) -> str | None:
        if self.canonical_action is not None:
            return self.canonical_action.canonicalization_receipt.receipt_id
        if self.representation_receipt is None:
            return None
        return self.representation_receipt.representation_trace_id


def _freeze_normalized_action(action: NormalizedAction) -> NormalizedAction:
    vector = np.asarray(action.vector, dtype=np.float32).copy()
    vector.setflags(write=False)
    joints: dict[str, np.ndarray] = {}
    for side, value in action.arm_joint_targets.items():
        array = np.asarray(value, dtype=np.float32).copy()
        array.setflags(write=False)
        joints[str(side)] = array
    frozen = NormalizedAction(
        vector=vector,
        arm_joint_targets=joints,
        metadata=deep_freeze(dict(action.metadata)),
    )
    object.__setattr__(frozen, "arm_joint_targets", deep_freeze(joints))
    return frozen


class _ImmutableList(list[Any]):
    """List-shaped immutable value for canonical list/tuple parity."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Immutable prediction candidate metadata cannot be mutated.")

    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    __setitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def _freeze_candidate_metadata(value: Any) -> Any:
    """Freeze metadata while preserving canonical list versus tuple tags."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_candidate_metadata(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return _ImmutableList(_freeze_candidate_metadata(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_candidate_metadata(item) for item in value)
    return deep_freeze(value)


def _freeze_prediction_candidate_action(action: NormalizedAction) -> NormalizedAction:
    vector = np.asarray(action.vector, dtype=np.float32).copy()
    vector.setflags(write=False)
    joints: dict[str, np.ndarray] = {}
    for side, value in action.arm_joint_targets.items():
        array = np.asarray(value, dtype=np.float32).copy()
        array.setflags(write=False)
        joints[str(side)] = array
    frozen = NormalizedAction(
        vector=vector,
        arm_joint_targets=joints,
        metadata=_freeze_candidate_metadata(dict(action.metadata)),
    )
    object.__setattr__(frozen, "arm_joint_targets", deep_freeze(joints))
    return frozen


def _detach_normalized_action(
    action: NormalizedAction,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> NormalizedAction:
    """Copy array-backed action values before queue ownership transfers."""

    action_metadata = dict(action.metadata)
    if metadata is not None:
        action_metadata.update(dict(metadata))
    return NormalizedAction(
        vector=np.asarray(action.vector, dtype=np.float32).copy(),
        arm_joint_targets={
            str(side): np.asarray(value, dtype=np.float32).copy()
            for side, value in action.arm_joint_targets.items()
        },
        metadata=action_metadata,
    )


_EMISSION_SEMANTIC_METADATA_KEYS = (
    "arm_action_specs_by_side",
    "hand_action_specs_by_side",
    "hand_active_mask_by_side",
    "end_effector_enabled_sides",
    "pose_action_source_mode",
    "pose_action_mode",
    "pose_action_source_absolute_frame",
    "pose_action_output_absolute_frame",
    "pose_anchor_policy",
    "pose_anchor_sides",
)


def _emission_semantic_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in _EMISSION_SEMANTIC_METADATA_KEYS
        if key in metadata
    }


def _decoded_action_payload(action: NormalizedAction) -> dict[str, Any]:
    return {
        "vector": np.asarray(action.vector, dtype=np.float32),
        "arm_joint_targets": {
            str(side): np.asarray(value, dtype=np.float32)
            for side, value in action.arm_joint_targets.items()
        },
        "semantic_metadata": _emission_semantic_metadata(action.metadata),
    }


@dataclass(frozen=True)
class PolicyChunkPredictionCandidate:
    """Detached prediction output awaiting control-thread queue admission.

    This is a compatibility seam for proving that model execution/decode can
    be separated from the existing ``ActionChunkBuffer`` mutation.  It is not
    the RTC timed-plan result contract and it carries no execution authority.
    Policies with prediction transaction hooks must keep using the existing
    synchronous ``predict_and_queue()`` path.
    """

    inference_id: str
    policy_node_id: str
    produced_clock_domain_id: str
    observation_timestamp: float
    observation_step_index: int
    policy_observation: DeploymentObservation | None = None
    actor_observation: PolicyObservationView | None = None
    observation_parity: PolicyObservationParityReceipt | None = None
    policy_output: Any = None
    actions: tuple[NormalizedAction, ...] = ()
    source_chunk_size: int = 0
    action_chunk_execution_steps: int | None = None
    inference_started_at: float | None = None
    inference_completed_at: float | None = None
    error: BaseException | None = None

    def __post_init__(self) -> None:
        inference_id = str(self.inference_id).strip()
        policy_node_id = str(self.policy_node_id).strip()
        clock_domain_id = str(self.produced_clock_domain_id).strip()
        if not inference_id or not policy_node_id or not clock_domain_id:
            raise ValueError(
                "inference_id, policy_node_id and produced_clock_domain_id "
                "must be non-empty."
            )
        observation_timestamp = float(self.observation_timestamp)
        if not math.isfinite(observation_timestamp):
            raise ValueError("observation_timestamp must be finite.")
        observation_step_index = int(self.observation_step_index)
        if observation_step_index < 0:
            raise ValueError("observation_step_index must be non-negative.")
        inference_started_at = self.inference_started_at
        inference_completed_at = self.inference_completed_at
        if inference_started_at is not None:
            inference_started_at = float(inference_started_at)
            if not math.isfinite(inference_started_at):
                raise ValueError("inference_started_at must be finite when supplied.")
        if inference_completed_at is not None:
            inference_completed_at = float(inference_completed_at)
            if not math.isfinite(inference_completed_at):
                raise ValueError("inference_completed_at must be finite when supplied.")
        if self.error is None:
            if inference_started_at is None or inference_completed_at is None:
                raise ValueError(
                    "Successful prediction candidates require inference timestamps."
                )
            if inference_completed_at < inference_started_at:
                raise ValueError("Inference monotonic time moved backwards.")
        actions = tuple(
            _freeze_prediction_candidate_action(action) for action in self.actions
        )
        source_chunk_size = int(self.source_chunk_size)
        if source_chunk_size < len(actions):
            raise ValueError("source_chunk_size must cover the selected actions.")
        object.__setattr__(self, "inference_id", inference_id)
        object.__setattr__(self, "policy_node_id", policy_node_id)
        object.__setattr__(self, "produced_clock_domain_id", clock_domain_id)
        object.__setattr__(self, "observation_timestamp", observation_timestamp)
        object.__setattr__(self, "observation_step_index", observation_step_index)
        object.__setattr__(self, "inference_started_at", inference_started_at)
        object.__setattr__(self, "inference_completed_at", inference_completed_at)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "source_chunk_size", source_chunk_size)
        object.__setattr__(
            self,
            "action_chunk_execution_steps",
            ActionChunkExecutionPolicy(self.action_chunk_execution_steps).execution_steps,
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
    actor_observation: PolicyObservationView | None = None
    observation_parity: PolicyObservationParityReceipt | None = None
    policy_output: Any = None
    chunk_lineage: PolicyChunkLineage | None = None
    actions: tuple[NormalizedAction, ...] = ()
    accepted: bool = False
    discarded_by_gate: bool = False
    faults: tuple[str, ...] = ()
    queue_len_before: int = 0
    queue_len_after: int = 0
    source_chunk_size: int = 0
    executed_chunk_size: int = 0
    action_chunk_execution_steps: int | None = None
    error: BaseException | None = None
    rollback_error: BaseException | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "faults", tuple(str(fault) for fault in self.faults))
        object.__setattr__(self, "queue_len_before", int(self.queue_len_before))
        object.__setattr__(self, "queue_len_after", int(self.queue_len_after))
        object.__setattr__(self, "source_chunk_size", int(self.source_chunk_size))
        object.__setattr__(self, "executed_chunk_size", int(self.executed_chunk_size))
        object.__setattr__(
            self,
            "action_chunk_execution_steps",
            ActionChunkExecutionPolicy(self.action_chunk_execution_steps).execution_steps,
        )

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
    emission: PolicyActionEmission | None = None
    emission_error: BaseException | None = None
    queue_len_before: int = 0
    queue_len_after: int = 0
    canonical_action: CanonicalPolicyAction | None = None

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

    def __init__(
        self,
        config: PolicyDeployRuntimeConfig,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ):
        if not isinstance(config, PolicyDeployRuntimeConfig):
            raise TypeError("config must be a PolicyDeployRuntimeConfig.")
        self.config = config
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable.")
        self._clock = monotonic_clock
        self._action_queue = ActionChunkBuffer()
        self._chunk_execution_policy = ActionChunkExecutionPolicy(
            config.action_chunk_execution_steps
        )
        self._last_prediction_at: float | None = None
        self._next_action_chunk_id = 0
        self._next_inference_sequence = 0
        self._chunk_lineage_by_id: dict[int, PolicyChunkLineage] = {}
        self._next_scheduled_effective_at: float | None = None

    def action_queue_length(self) -> int:
        """Return the number of decoded actions waiting for execution."""

        return len(self._action_queue)

    def clear_action_queue(self) -> None:
        """Discard all queued policy actions."""

        self._action_queue.clear()
        self._chunk_lineage_by_id.clear()
        self._next_scheduled_effective_at = None

    def action_queue_pop_count(self) -> int:
        return int(self._action_queue.popped_count)

    def reserve_prediction_inference_id(self) -> str:
        """Reserve one inference identity on the control thread.

        The split prediction/admission seam deliberately leaves request
        scheduling with its caller.  Reservation consumes the same monotonic
        sequence used by ``predict_and_queue()`` so the two paths cannot
        silently reuse an inference identity on one runtime.
        """

        inference_sequence = self._next_inference_sequence
        self._next_inference_sequence += 1
        return f"{self.config.policy_node_id}:inference:{inference_sequence}"

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
        emission = None
        emission_error: BaseException | None = None
        if raw_action is not None:
            pre_safety: DeployRobotActionAdapterResult | None = None
            adapt_pre_safety = getattr(preparer, "adapt_pre_safety", None)
            if callable(adapt_pre_safety):
                try:
                    pre_safety = adapt_pre_safety(
                        raw_action,
                        observation,
                        policy_name=policy_name,
                    )
                except Exception as exc:  # noqa: BLE001 - passive tap must not alter legacy.
                    emission_error = exc
                    pre_safety = None
            try:
                emission = self._build_action_emission(
                    raw_action,
                    pre_safety=pre_safety,
                    queue_len_before=queue_len_before,
                    queue_len_after=queue_len_after,
                )
            except Exception as exc:  # noqa: BLE001 - passive tap must not alter legacy.
                if emission_error is None:
                    emission_error = exc
            prepared_action = preparer.prepare(
                raw_action,
                observation,
                policy_name=policy_name,
            )
        return PolicyActionPreparationResult(
            raw_action=raw_action,
            raw_lookahead_action=raw_lookahead_action,
            prepared_action=prepared_action,
            emission=emission,
            emission_error=emission_error,
            queue_len_before=queue_len_before,
            queue_len_after=queue_len_after,
        )

    def prepare_timed_action_keyframe(
        self,
        keyframe: TimedPolicyActionKeyframe,
        observation: DeploymentObservation,
        *,
        preparer: ActionPreparer,
        policy_name: str = "",
        plan_keyframe_count_before: int,
        plan_keyframe_count_after: int,
        plan_pop_count: int,
    ) -> PolicyActionPreparationResult:
        """Prepare one already-popped RTC keyframe through the formal emission seam.

        This method never reads or mutates the legacy ``ActionChunkBuffer`` and
        deliberately exposes no passive successor.  The returned
        ``PolicyActionEmission`` is the same typed admission fact used by the
        synchronous queue path.
        """

        if not isinstance(keyframe, TimedPolicyActionKeyframe):
            raise TypeError("keyframe must be TimedPolicyActionKeyframe.")
        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be DeploymentObservation.")
        before = int(plan_keyframe_count_before)
        after = int(plan_keyframe_count_after)
        pop_count = int(plan_pop_count)
        if before <= 0 or after < 0 or after >= before or pop_count <= 0:
            raise ValueError("Timed-plan pop counters are invalid.")
        typed_action = keyframe.canonical_action
        raw_action = keyframe.decoded_absolute_action
        if (raw_action is None) == (typed_action is None):
            raise RuntimeError("Timed keyframe action payload is ambiguous.")
        if typed_action is not None:
            emission = self._build_timed_action_emission(
                keyframe,
                pre_safety=None,
                plan_keyframe_count_before=before,
                plan_keyframe_count_after=after,
                plan_pop_count=pop_count,
            )
            prepare_canonical = getattr(preparer, "prepare_canonical", None)
            prepared_action = None
            emission_error: BaseException | None = None
            if callable(prepare_canonical):
                try:
                    prepared_action = prepare_canonical(
                        typed_action,
                        observation,
                        policy_name=policy_name,
                        input_action_id=emission.emission_id,
                        target_effective_at=keyframe.target_effective_at,
                    )
                except Exception as exc:  # noqa: BLE001 - typed path fails closed.
                    emission_error = exc
            else:
                emission_error = RuntimeError(
                    "Typed canonical preparation is unavailable; "
                    "the formal emission remains no-action."
                )
            return PolicyActionPreparationResult(
                raw_action=None,
                raw_lookahead_action=None,
                prepared_action=prepared_action,
                emission=emission,
                emission_error=emission_error,
                queue_len_before=before,
                queue_len_after=after,
                canonical_action=typed_action,
            )
        assert raw_action is not None
        pre_safety: DeployRobotActionAdapterResult | None = None
        emission_error: BaseException | None = None
        adapt_pre_safety = getattr(preparer, "adapt_pre_safety", None)
        if callable(adapt_pre_safety):
            try:
                pre_safety = adapt_pre_safety(
                    raw_action,
                    observation,
                    policy_name=policy_name,
                )
            except Exception as exc:  # noqa: BLE001 - passive tap matches sync seam.
                emission_error = exc
                pre_safety = None
        emission = None
        try:
            emission = self._build_timed_action_emission(
                keyframe,
                pre_safety=pre_safety,
                plan_keyframe_count_before=before,
                plan_keyframe_count_after=after,
                plan_pop_count=pop_count,
            )
        except Exception as exc:  # noqa: BLE001 - canonical prepare still reports faults.
            if emission_error is None:
                emission_error = exc
        prepared_action = preparer.prepare(
            raw_action,
            observation,
            policy_name=policy_name,
        )
        return PolicyActionPreparationResult(
            raw_action=raw_action,
            raw_lookahead_action=None,
            prepared_action=prepared_action,
            emission=emission,
            emission_error=emission_error,
            queue_len_before=before,
            queue_len_after=after,
        )

    def _build_timed_action_emission(
        self,
        keyframe: TimedPolicyActionKeyframe,
        *,
        pre_safety: DeployRobotActionAdapterResult | None,
        plan_keyframe_count_before: int,
        plan_keyframe_count_after: int,
        plan_pop_count: int,
    ) -> PolicyActionEmission:
        observed_at = float(self._clock())
        if not math.isfinite(observed_at):
            raise ValueError("monotonic_clock returned a non-finite emission time.")
        emitted_at = max(observed_at, keyframe.produced_at)
        typed_action = keyframe.canonical_action
        if typed_action is None:
            assert keyframe.decoded_absolute_action is not None
            frozen_action = _freeze_normalized_action(
                keyframe.decoded_absolute_action
            )
            robot_action = None if pre_safety is None else pre_safety.robot_action
            representation = (
                None if pre_safety is None else pre_safety.representation_receipt
            )
            validation = None if pre_safety is None else pre_safety.source_validation
            faults = () if pre_safety is None else pre_safety.faults
        else:
            frozen_action = None
            robot_action = typed_action.action
            representation = None
            validation = SourceValidationResult(accepted=True, reasons=())
            faults = ()
        payload = {
            "admission_mode": self.config.action_emission_admission_mode,
            "policy_node_id": keyframe.policy_node_id,
            "inference_id": keyframe.inference_id,
            "policy_output_id": keyframe.policy_output_id,
            "chunk_id": keyframe.chunk_id,
            "chunk_keyframe_index": keyframe.chunk_keyframe_index,
            "chunk_size": keyframe.chunk_size,
            "pre_safety_robot_action": (
                None
                if robot_action is None
                else robot_action_semantic_payload(robot_action)
            ),
            "representation_trace_id": (
                None
                if representation is None
                else representation.representation_trace_id
            ),
            "source_validation": (
                None
                if validation is None
                else {"accepted": validation.accepted, "reasons": validation.reasons}
            ),
            "pre_safety_faults": faults,
            "produced_at": keyframe.produced_at,
            "produced_clock_domain_id": keyframe.effective_clock_domain_id,
            "emitted_at": emitted_at,
            "target_effective_at": keyframe.target_effective_at,
            "effective_from": keyframe.target_effective_at,
            "effective_until": keyframe.effective_until,
            "effective_clock_domain_id": keyframe.effective_clock_domain_id,
            "schedule_spec_id": keyframe.schedule_spec_id,
            "schedule_spec_hash": keyframe.schedule_spec_hash,
            "queue_len_before": plan_keyframe_count_before,
            "queue_len_after": plan_keyframe_count_after,
            "queue_pop_count": plan_pop_count,
        }
        if typed_action is None:
            assert frozen_action is not None
            payload["decoded_absolute_action"] = {
                "vector": frozen_action.vector,
                "arm_joint_targets": frozen_action.arm_joint_targets,
                "semantic_metadata": _emission_semantic_metadata(
                    frozen_action.metadata
                ),
            }
            hash_domain = POLICY_ACTION_EMISSION_HASH_DOMAIN_V1
        else:
            payload["action_payload_kind"] = "typed_canonical_policy_action_v1"
            payload["canonical_action_hash"] = typed_action.semantic_content_hash
            payload["canonicalization_receipt_hash"] = (
                typed_action.canonicalization_receipt.semantic_content_hash
            )
            hash_domain = POLICY_ACTION_EMISSION_HASH_DOMAIN_V2
        semantic_hash = canonical_content_hash(
            payload,
            domain=hash_domain,
        )
        return PolicyActionEmission(
            emission_id=f"policy-action-emission:{semantic_hash}",
            admission_mode=self.config.action_emission_admission_mode,
            policy_node_id=keyframe.policy_node_id,
            inference_id=keyframe.inference_id,
            policy_output_id=keyframe.policy_output_id,
            chunk_id=keyframe.chunk_id,
            chunk_keyframe_index=keyframe.chunk_keyframe_index,
            chunk_size=keyframe.chunk_size,
            decoded_absolute_action=frozen_action,
            pre_safety_robot_action=robot_action,
            source_validation=validation,
            representation_receipt=representation,
            pre_safety_faults=faults,
            produced_at=keyframe.produced_at,
            produced_clock_domain_id=keyframe.effective_clock_domain_id,
            emitted_at=emitted_at,
            target_effective_at=keyframe.target_effective_at,
            effective_from=keyframe.target_effective_at,
            effective_until=keyframe.effective_until,
            effective_clock_domain_id=keyframe.effective_clock_domain_id,
            schedule_spec_id=keyframe.schedule_spec_id,
            schedule_spec_hash=keyframe.schedule_spec_hash,
            queue_len_before=plan_keyframe_count_before,
            queue_len_after=plan_keyframe_count_after,
            queue_pop_count=plan_pop_count,
            semantic_content_hash=semantic_hash,
            canonical_action=typed_action,
        )

    def _build_action_emission(
        self,
        action: NormalizedAction,
        *,
        pre_safety: DeployRobotActionAdapterResult | None,
        queue_len_before: int,
        queue_len_after: int,
    ) -> PolicyActionEmission:
        metadata = dict(action.metadata)
        try:
            chunk_id = int(metadata[_CHUNK_ID_METADATA_KEY])
            keyframe_index = int(metadata[_CHUNK_INDEX_METADATA_KEY])
            chunk_size = int(metadata[_CHUNK_SIZE_METADATA_KEY])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Popped policy action is missing chunk lineage.") from exc
        lineage = self._chunk_lineage_by_id.get(chunk_id)
        if lineage is None:
            raise RuntimeError("Popped policy action has no retained prediction lineage.")
        effective_from, effective_until = lineage.schedule.interval_for(keyframe_index)
        emitted_at = float(self._clock())
        if not math.isfinite(emitted_at):
            raise ValueError("monotonic_clock returned a non-finite emission time.")
        frozen_action = _freeze_normalized_action(action)
        robot_action = None if pre_safety is None else pre_safety.robot_action
        representation = None if pre_safety is None else pre_safety.representation_receipt
        validation = None if pre_safety is None else pre_safety.source_validation
        faults = () if pre_safety is None else pre_safety.faults
        payload = {
            "admission_mode": self.config.action_emission_admission_mode,
            "policy_node_id": lineage.policy_node_id,
            "inference_id": lineage.inference_id,
            "policy_output_id": lineage.policy_output_id,
            "chunk_id": chunk_id,
            "chunk_keyframe_index": keyframe_index,
            "chunk_size": chunk_size,
            "decoded_absolute_action": {
                "vector": frozen_action.vector,
                "arm_joint_targets": frozen_action.arm_joint_targets,
                "semantic_metadata": _emission_semantic_metadata(
                    frozen_action.metadata
                ),
            },
            "pre_safety_robot_action": (
                None if robot_action is None else robot_action_semantic_payload(robot_action)
            ),
            "representation_trace_id": (
                None if representation is None else representation.representation_trace_id
            ),
            "source_validation": (
                None
                if validation is None
                else {"accepted": validation.accepted, "reasons": validation.reasons}
            ),
            "pre_safety_faults": faults,
            "produced_at": lineage.inference_completed_at,
            "produced_clock_domain_id": lineage.produced_clock_domain_id,
            "emitted_at": emitted_at,
            "target_effective_at": effective_from,
            "effective_from": effective_from,
            "effective_until": effective_until,
            "effective_clock_domain_id": lineage.produced_clock_domain_id,
            "schedule_spec_id": lineage.schedule.spec_id,
            "schedule_spec_hash": lineage.schedule.semantic_content_hash,
            "queue_len_before": int(queue_len_before),
            "queue_len_after": int(queue_len_after),
            "queue_pop_count": self.action_queue_pop_count(),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=POLICY_ACTION_EMISSION_HASH_DOMAIN_V1,
        )
        emission = PolicyActionEmission(
            emission_id=f"policy-action-emission:{semantic_hash}",
            admission_mode=self.config.action_emission_admission_mode,
            policy_node_id=lineage.policy_node_id,
            inference_id=lineage.inference_id,
            policy_output_id=lineage.policy_output_id,
            chunk_id=chunk_id,
            chunk_keyframe_index=keyframe_index,
            chunk_size=chunk_size,
            decoded_absolute_action=frozen_action,
            pre_safety_robot_action=robot_action,
            source_validation=validation,
            representation_receipt=representation,
            pre_safety_faults=faults,
            produced_at=lineage.inference_completed_at,
            produced_clock_domain_id=lineage.produced_clock_domain_id,
            emitted_at=emitted_at,
            target_effective_at=effective_from,
            effective_from=effective_from,
            effective_until=effective_until,
            effective_clock_domain_id=lineage.produced_clock_domain_id,
            schedule_spec_id=lineage.schedule.spec_id,
            schedule_spec_hash=lineage.schedule.semantic_content_hash,
            queue_len_before=int(queue_len_before),
            queue_len_after=int(queue_len_after),
            queue_pop_count=self.action_queue_pop_count(),
            semantic_content_hash=semantic_hash,
        )
        if keyframe_index + 1 >= chunk_size:
            self._chunk_lineage_by_id.pop(chunk_id, None)
        return emission

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

    def prepare_actor_observation(
        self,
        policy_observation: DeploymentObservation,
    ) -> tuple[PolicyObservationView, PolicyObservationParityReceipt]:
        """Project and validate the only model-visible observation boundary."""

        contract = self.config.policy_observation_contract
        assert contract is not None
        view = build_policy_observation_view(policy_observation, contract=contract)
        parity = compare_policy_observation_view(policy_observation, view)
        if not parity.matched:
            raise ValueError(
                "Strict policy observation parity failed at: "
                + ", ".join(parity.difference_paths)
            )
        return view, parity

    def observation_firewall_manifest(self) -> dict[str, Any]:
        contract = self.config.policy_observation_contract
        assert contract is not None
        return {
            "schema": "dexteleop_policy_observation_firewall_v1",
            "mode": self.config.observation_firewall_mode,
            "contract_id": contract.contract_id,
            "contract_hash": contract.semantic_content_hash,
            "compatibility_adapter_id": contract.compatibility_adapter_id,
            "owner": "robot_motion.deploy.PolicyDeployRuntime",
            "legacy_unrestricted_input": False,
            "compatibility_deletion_condition": (
                "all supported policy adapters consume PolicyObservationView directly"
            ),
            "activation_deadline": "before_m6_authoritative_gate",
        }

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

    def _clock_time(self, *, name: str) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise ValueError(f"monotonic_clock returned non-finite {name}.")
        return value

    def _build_chunk_lineage(
        self,
        *,
        chunk_id: int,
        actions: list[NormalizedAction],
        observation: DeploymentObservation,
        inference_id: str,
        inference_started_at: float,
        inference_completed_at: float,
        schedule_start_hint: float,
        keyframe_period_sec: float,
        queue_mode: str,
    ) -> tuple[PolicyChunkLineage, float]:
        return self._build_chunk_lineage_from_observation_identity(
            chunk_id=chunk_id,
            actions=actions,
            observation_timestamp=observation.timestamp,
            observation_step_index=observation.step_index,
            inference_id=inference_id,
            inference_started_at=inference_started_at,
            inference_completed_at=inference_completed_at,
            schedule_start_hint=schedule_start_hint,
            keyframe_period_sec=keyframe_period_sec,
            queue_mode=queue_mode,
        )

    def _build_chunk_lineage_from_observation_identity(
        self,
        *,
        chunk_id: int,
        actions: list[NormalizedAction],
        observation_timestamp: float,
        observation_step_index: int,
        inference_id: str,
        inference_started_at: float,
        inference_completed_at: float,
        schedule_start_hint: float,
        keyframe_period_sec: float,
        queue_mode: str,
    ) -> tuple[PolicyChunkLineage, float]:
        period = float(keyframe_period_sec)
        if not math.isfinite(period) or period <= 0:
            raise ValueError("action_keyframe_period_sec must be finite and positive.")
        if queue_mode == "append" and self._next_scheduled_effective_at is not None:
            schedule_start = max(
                float(schedule_start_hint),
                self._next_scheduled_effective_at,
            )
        else:
            schedule_start = float(schedule_start_hint)
        chunk_size = len(actions)
        schedule = ActionEmissionScheduleSpec(
            spec_id=(
                f"{self.config.policy_node_id}:chunk:{chunk_id}:"
                f"period:{period:.17g}"
            ),
            start_at=schedule_start,
            keyframe_period_sec=period,
            keyframe_count=chunk_size,
            horizon_sec=period * chunk_size,
            maximum_hold_sec=max(
                period,
                self.config.action_maximum_hold_sec,
            ),
        )
        output_hash = canonical_content_hash(
            {
                "inference_id": inference_id,
                "actions": tuple(_decoded_action_payload(action) for action in actions),
            },
            domain="dexteleop.policy_output.v1",
        )
        policy_output_id = f"policy-output:{output_hash}"
        payload = {
            "inference_id": inference_id,
            "policy_output_id": policy_output_id,
            "policy_node_id": self.config.policy_node_id,
            "chunk_id": chunk_id,
            "inference_started_at": inference_started_at,
            "inference_completed_at": inference_completed_at,
            "produced_clock_domain_id": self.config.policy_clock_domain_id,
            "observation_timestamp": float(observation_timestamp),
            "observation_step_index": int(observation_step_index),
            "schedule_hash": schedule.semantic_content_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=POLICY_CHUNK_LINEAGE_HASH_DOMAIN_V1,
        )
        lineage = PolicyChunkLineage(
            inference_id=inference_id,
            policy_output_id=policy_output_id,
            policy_node_id=self.config.policy_node_id,
            chunk_id=chunk_id,
            inference_started_at=inference_started_at,
            inference_completed_at=inference_completed_at,
            produced_clock_domain_id=self.config.policy_clock_domain_id,
            observation_timestamp=float(observation_timestamp),
            observation_step_index=int(observation_step_index),
            schedule=schedule,
            semantic_content_hash=semantic_hash,
        )
        return lineage, schedule_start + period * chunk_size

    @staticmethod
    def _policy_uses_prediction_transaction(policy: PolicyLike) -> bool:
        return any(
            callable(getattr(policy, hook, None))
            for hook in (
                "begin_prediction_transaction",
                "commit_prediction_transaction",
                "rollback_prediction_transaction",
            )
        )

    def predict_chunk(
        self,
        policy: PolicyLike,
        observation: DeploymentObservation,
        *,
        inference_id: str,
    ) -> PolicyChunkPredictionCandidate:
        """Run prediction/decode without mutating the legacy action queue.

        This method is suitable for a worker executor only after the control
        thread has reserved/bound ``inference_id``.  Queue replacement,
        append, schedule lineage and the prediction-rate timestamp remain
        control-thread work in ``admit_prediction_result()``.
        """

        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation.")
        if self._policy_uses_prediction_transaction(policy):
            raise PolicyPredictionTransactionRequiresSynchronousQueueError(
                "Policies with prediction transaction hooks must use "
                "predict_and_queue()."
            )

        policy_observation: DeploymentObservation | None = None
        actor_observation: PolicyObservationView | None = None
        observation_parity: PolicyObservationParityReceipt | None = None
        policy_output: Any = None
        actions: list[NormalizedAction] = []
        source_chunk_size = 0
        inference_started_at: float | None = None
        inference_completed_at: float | None = None
        error: BaseException | None = None
        try:
            policy_observation = self.prepare_policy_observation(observation)
            actor_observation, observation_parity = self.prepare_actor_observation(
                policy_observation
            )
            prediction_input: Any = actor_observation
            if self.config.observation_firewall_mode == "dual_validate":
                prediction_input = actor_observation.to_deployment_observation()
            inference_started_at = self._clock_time(name="inference_started_at")
            policy_output = policy.predict(prediction_input)
            inference_completed_at = self._clock_time(name="inference_completed_at")
            if inference_completed_at < inference_started_at:
                raise ValueError("Inference monotonic time moved backwards.")
            actions = self.normalize_and_decode(policy_output, observation)
            if actions:
                source_chunk_size = len(actions)
                actions = list(self._chunk_execution_policy.select(actions))
                if not actions:
                    raise ValueError("Action chunk execution prefix is empty.")
        except Exception as exc:  # noqa: BLE001 - admission preserves legacy handling.
            error = exc

        return PolicyChunkPredictionCandidate(
            inference_id=inference_id,
            policy_node_id=self.config.policy_node_id,
            produced_clock_domain_id=self.config.policy_clock_domain_id,
            observation_timestamp=observation.timestamp,
            observation_step_index=observation.step_index,
            policy_observation=policy_observation,
            actor_observation=actor_observation,
            observation_parity=observation_parity,
            policy_output=policy_output,
            actions=tuple(actions),
            source_chunk_size=source_chunk_size,
            action_chunk_execution_steps=self._chunk_execution_policy.execution_steps,
            inference_started_at=inference_started_at,
            inference_completed_at=inference_completed_at,
            error=error,
        )

    def admit_prediction_result(
        self,
        candidate: PolicyChunkPredictionCandidate,
        *,
        now: float,
        queue_policy: str = "replace",
        discard_policy_output: bool = False,
        action_keyframe_period_sec: float | None = None,
    ) -> PolicyChunkPredictionResult:
        """Apply one split prediction result to the legacy queue in order.

        Admission is intentionally a control-thread compatibility operation;
        RTC stale-request/epoch validation and mutable-future replacement live
        in ``TimedPolicyActionPlan`` rather than in ``ActionChunkBuffer``.
        """

        if not isinstance(candidate, PolicyChunkPredictionCandidate):
            raise TypeError("candidate must be a PolicyChunkPredictionCandidate.")
        queue_mode = str(queue_policy).strip().lower()
        if queue_mode not in {"replace", "append"}:
            raise ValueError(f"Unsupported queue_policy: {queue_policy!r}.")
        if candidate.policy_node_id != self.config.policy_node_id:
            raise ValueError("Prediction candidate policy_node_id mismatch.")
        if candidate.produced_clock_domain_id != self.config.policy_clock_domain_id:
            raise ValueError("Prediction candidate clock-domain mismatch.")
        if (
            candidate.action_chunk_execution_steps
            != self._chunk_execution_policy.execution_steps
        ):
            raise ValueError("Prediction candidate execution-prefix mismatch.")

        queue_len_before = self.action_queue_length()
        common_result = {
            "policy_observation": candidate.policy_observation,
            "actor_observation": candidate.actor_observation,
            "observation_parity": candidate.observation_parity,
            "policy_output": candidate.policy_output,
            "queue_len_before": queue_len_before,
            "source_chunk_size": candidate.source_chunk_size,
            "action_chunk_execution_steps": (
                self._chunk_execution_policy.execution_steps
            ),
        }
        if candidate.error is not None:
            self.clear_action_queue()
            return PolicyChunkPredictionResult(
                **common_result,
                actions=candidate.actions,
                queue_len_after=self.action_queue_length(),
                executed_chunk_size=len(candidate.actions),
                error=candidate.error,
            )
        if not candidate.actions:
            return PolicyChunkPredictionResult(
                **common_result,
                faults=("policy_empty_action",),
                queue_len_after=self.action_queue_length(),
                executed_chunk_size=0,
            )

        actions: list[NormalizedAction] = []
        chunk_lineage: PolicyChunkLineage | None = None
        try:
            chunk_id = int(self._next_action_chunk_id)
            self._next_action_chunk_id += 1
            chunk_size = len(candidate.actions)
            actions = [
                _detach_normalized_action(
                    action,
                    metadata={
                        _CHUNK_ID_METADATA_KEY: chunk_id,
                        _CHUNK_INDEX_METADATA_KEY: index,
                        _CHUNK_SIZE_METADATA_KEY: chunk_size,
                        _CHUNK_SOURCE_SIZE_METADATA_KEY: candidate.source_chunk_size,
                        _CHUNK_EXECUTION_STEPS_METADATA_KEY: (
                            self._chunk_execution_policy.execution_steps
                        ),
                    },
                )
                for index, action in enumerate(candidate.actions)
            ]
            period = (
                self.config.action_keyframe_period_sec
                if action_keyframe_period_sec is None
                else float(action_keyframe_period_sec)
            )
            assert candidate.inference_started_at is not None
            assert candidate.inference_completed_at is not None
            chunk_lineage, next_scheduled_effective_at = (
                self._build_chunk_lineage_from_observation_identity(
                    chunk_id=chunk_id,
                    actions=actions,
                    observation_timestamp=candidate.observation_timestamp,
                    observation_step_index=candidate.observation_step_index,
                    inference_id=candidate.inference_id,
                    inference_started_at=candidate.inference_started_at,
                    inference_completed_at=candidate.inference_completed_at,
                    schedule_start_hint=max(
                        float(now),
                        candidate.inference_completed_at,
                    ),
                    keyframe_period_sec=period,
                    queue_mode=queue_mode,
                )
            )
            # Match the existing synchronous path: a gate-discarded chunk
            # still consumes this policy-rate opportunity.
            self._last_prediction_at = float(now)
            if discard_policy_output:
                self.clear_action_queue()
                return PolicyChunkPredictionResult(
                    **common_result,
                    chunk_lineage=chunk_lineage,
                    actions=tuple(actions),
                    discarded_by_gate=True,
                    queue_len_after=self.action_queue_length(),
                    executed_chunk_size=len(actions),
                )

            if queue_mode == "append":
                self._action_queue.append(actions)
            else:
                self._action_queue.replace(actions)
                self._chunk_lineage_by_id.clear()
            self._chunk_lineage_by_id[chunk_id] = chunk_lineage
            self._next_scheduled_effective_at = next_scheduled_effective_at
            return PolicyChunkPredictionResult(
                **common_result,
                chunk_lineage=chunk_lineage,
                actions=tuple(actions),
                accepted=True,
                queue_len_after=self.action_queue_length(),
                executed_chunk_size=len(actions),
            )
        except Exception as exc:  # noqa: BLE001 - preserve legacy queue fail-closed.
            self.clear_action_queue()
            return PolicyChunkPredictionResult(
                **common_result,
                chunk_lineage=chunk_lineage,
                actions=tuple(actions),
                queue_len_after=self.action_queue_length(),
                executed_chunk_size=len(actions),
                error=exc,
            )

    def predict_and_queue(
        self,
        policy: PolicyLike,
        observation: DeploymentObservation,
        *,
        now: float,
        queue_policy: str = "replace",
        discard_policy_output: bool = False,
        action_keyframe_period_sec: float | None = None,
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
        actor_observation: PolicyObservationView | None = None
        observation_parity: PolicyObservationParityReceipt | None = None
        policy_output: Any = None
        actions: list[NormalizedAction] = []
        source_chunk_size = 0
        checkpoint: Any = None
        transaction_started = False
        transaction_resolved = False
        inference_sequence = self._next_inference_sequence
        self._next_inference_sequence += 1
        inference_id = f"{self.config.policy_node_id}:inference:{inference_sequence}"
        inference_started_at: float | None = None
        inference_completed_at: float | None = None
        chunk_lineage: PolicyChunkLineage | None = None

        try:
            policy_observation = self.prepare_policy_observation(observation)
            actor_observation, observation_parity = self.prepare_actor_observation(
                policy_observation
            )
            begin = getattr(policy, "begin_prediction_transaction", None)
            if callable(begin):
                checkpoint = begin()
                transaction_started = True
            prediction_input: Any = actor_observation
            if self.config.observation_firewall_mode == "dual_validate":
                prediction_input = actor_observation.to_deployment_observation()
            inference_started_at = self._clock_time(name="inference_started_at")
            policy_output = policy.predict(prediction_input)
            inference_completed_at = self._clock_time(name="inference_completed_at")
            if inference_completed_at < inference_started_at:
                raise ValueError("Inference monotonic time moved backwards.")
            actions = self.normalize_and_decode(policy_output, observation)
            if actions:
                source_chunk_size = len(actions)
                actions = list(self._chunk_execution_policy.select(actions))
                if not actions:
                    raise ValueError("Action chunk execution prefix is empty.")
                chunk_id = int(self._next_action_chunk_id)
                self._next_action_chunk_id += 1
                chunk_size = len(actions)
                actions = [
                    _detach_normalized_action(
                        action,
                        metadata={
                            _CHUNK_ID_METADATA_KEY: chunk_id,
                            _CHUNK_INDEX_METADATA_KEY: index,
                            _CHUNK_SIZE_METADATA_KEY: chunk_size,
                            _CHUNK_SOURCE_SIZE_METADATA_KEY: source_chunk_size,
                            _CHUNK_EXECUTION_STEPS_METADATA_KEY: (
                                self._chunk_execution_policy.execution_steps
                            ),
                        },
                    )
                    for index, action in enumerate(actions)
                ]
                period = (
                    self.config.action_keyframe_period_sec
                    if action_keyframe_period_sec is None
                    else float(action_keyframe_period_sec)
                )
                assert inference_started_at is not None
                assert inference_completed_at is not None
                chunk_lineage, next_scheduled_effective_at = self._build_chunk_lineage(
                    chunk_id=chunk_id,
                    actions=actions,
                    observation=observation,
                    inference_id=inference_id,
                    inference_started_at=inference_started_at,
                    inference_completed_at=inference_completed_at,
                    # A newly replaced chunk cannot become effective before
                    # its output exists. Append mode may start even later when
                    # an older admitted chunk still owns the logical horizon.
                    schedule_start_hint=max(
                        float(now),
                        inference_completed_at,
                    ),
                    keyframe_period_sec=period,
                    queue_mode=queue_mode,
                )
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
                        actor_observation=actor_observation,
                        observation_parity=observation_parity,
                        policy_output=policy_output,
                        chunk_lineage=chunk_lineage,
                        actions=tuple(actions),
                        discarded_by_gate=True,
                        queue_len_before=queue_len_before,
                        queue_len_after=self.action_queue_length(),
                        source_chunk_size=source_chunk_size,
                        executed_chunk_size=len(actions),
                        action_chunk_execution_steps=(
                            self._chunk_execution_policy.execution_steps
                        ),
                    )

                if queue_mode == "append":
                    self._action_queue.append(actions)
                else:
                    self._action_queue.replace(actions)
                    self._chunk_lineage_by_id.clear()
                self._chunk_lineage_by_id[chunk_id] = chunk_lineage
                self._next_scheduled_effective_at = next_scheduled_effective_at
                if transaction_started:
                    commit = getattr(policy, "commit_prediction_transaction")
                    commit(checkpoint)
                    transaction_resolved = True
                return PolicyChunkPredictionResult(
                    policy_observation=policy_observation,
                    actor_observation=actor_observation,
                    observation_parity=observation_parity,
                    policy_output=policy_output,
                    chunk_lineage=chunk_lineage,
                    actions=tuple(actions),
                    accepted=True,
                    queue_len_before=queue_len_before,
                    queue_len_after=self.action_queue_length(),
                    source_chunk_size=source_chunk_size,
                    executed_chunk_size=len(actions),
                    action_chunk_execution_steps=(
                        self._chunk_execution_policy.execution_steps
                    ),
                )

            if transaction_started:
                rollback = getattr(policy, "rollback_prediction_transaction")
                rollback(checkpoint)
                transaction_resolved = True
            return PolicyChunkPredictionResult(
                policy_observation=policy_observation,
                actor_observation=actor_observation,
                observation_parity=observation_parity,
                policy_output=policy_output,
                faults=("policy_empty_action",),
                queue_len_before=queue_len_before,
                queue_len_after=self.action_queue_length(),
                source_chunk_size=source_chunk_size,
                executed_chunk_size=0,
                action_chunk_execution_steps=self._chunk_execution_policy.execution_steps,
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
                actor_observation=actor_observation,
                observation_parity=observation_parity,
                policy_output=policy_output,
                faults=faults,
                queue_len_before=queue_len_before,
                queue_len_after=self.action_queue_length(),
                source_chunk_size=source_chunk_size,
                executed_chunk_size=len(actions),
                action_chunk_execution_steps=self._chunk_execution_policy.execution_steps,
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
        action_chunk_execution_steps: int | None = None,
    ) -> None:
        self.source = source
        self.policy = policy
        self.preparer = preparer
        self.dispatcher = dispatcher
        self.policy_name = str(policy_name or getattr(policy, "name", "") or "policy")
        self.send_actions = bool(send_actions)
        self._chunk_execution_policy = ActionChunkExecutionPolicy(
            action_chunk_execution_steps
        )
        self._action_queue: deque[NormalizedAction] = deque()

    def action_queue_length(self) -> int:
        """Return the number of prefix actions waiting for the next step."""

        return len(self._action_queue)

    def clear_action_queue(self) -> None:
        self._action_queue.clear()

    def step(self, *, last_action: Any = None, now: float | None = None) -> DeployRuntimeStepResult:
        observation = self.source.get_observation(last_action=last_action, now=now)
        if self._chunk_execution_policy.enabled:
            if not isinstance(observation, DeploymentObservation):
                raise TypeError(
                    "action_chunk_execution_steps requires a DeploymentObservation source."
                )
            if self._action_queue:
                raw_action: Any = self._action_queue.popleft()
            else:
                policy_output = self.policy.predict(observation)
                actions = normalize_policy_output(policy_output, observation)
                selected = self._chunk_execution_policy.select(actions)
                if not selected:
                    return DeployRuntimeStepResult(
                        observation=observation,
                        raw_action=None,
                        faults=("policy_empty_action",),
                        metadata={
                            "policy_name": self.policy_name,
                            "action_chunk_execution_steps": (
                                self._chunk_execution_policy.execution_steps
                            ),
                        },
                    )
                raw_action = _detach_normalized_action(selected[0])
                self._action_queue.extend(
                    _detach_normalized_action(action) for action in selected[1:]
                )
        else:
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
        metadata = {"policy_name": self.policy_name}
        if self._chunk_execution_policy.enabled:
            metadata["action_chunk_execution_steps"] = (
                self._chunk_execution_policy.execution_steps
            )
        return DeployRuntimeStepResult(
            observation=observation,
            raw_action=raw_action,
            prepared_action=prepared,
            command=command,
            sent=sent,
            faults=faults,
            metadata=metadata,
        )
