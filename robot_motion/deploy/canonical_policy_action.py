"""Typed canonical policy actions shared by raw-RTC representation branches.

This module is intentionally pure.  It owns neither inference, timed-plan
admission, safety, IK, nor dispatch.  Absolute and relative model outputs must
already have been decoded to the same canonical ``RobotAction`` before they
enter this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import RobotAction, encode_robot_action
from robot_motion.action.proposal import (
    ProposalAdmissionError,
    freeze_robot_action,
    robot_action_semantic_payload,
)
from robot_motion.relative_action import (
    ACTION_POSE_MODE_ABSOLUTE,
    ACTION_POSE_MODE_RELATIVE,
    DEFAULT_ARM_POSE_SLICES,
    convert_arm_pose_chunk,
    normalize_action_pose_mode,
)

from .actions import NormalizedAction
from .schema import ACTION_DIM

if TYPE_CHECKING:
    from .inference import PolicyInferenceRequest


CANONICAL_POLICY_ACTION_SCHEMA_ID_V1 = "canonical_policy_action_v1"
CANONICAL_POLICY_ROBOT_ACTION_HASH_DOMAIN_V1 = (
    "dexteleop.canonical_policy_robot_action.v1"
)
POLICY_ACTION_CANONICALIZATION_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.policy_action_canonicalization_receipt.v1"
)
POLICY_ACTION_CANONICALIZATION_RECEIPT_HASH_DOMAIN_V2 = (
    "dexteleop.policy_action_canonicalization_receipt.v2"
)
CANONICAL_POLICY_ACTION_HASH_DOMAIN_V1 = "dexteleop.canonical_policy_action.v1"
ABSOLUTE_POLICY_SOURCE_CHUNK_HASH_DOMAIN_V1 = (
    "dexteleop.absolute_policy_source_chunk.v1"
)
ABSOLUTE_POLICY_SOURCE_ACTION_HASH_DOMAIN_V1 = (
    "dexteleop.absolute_policy_source_action.v1"
)
RELATIVE_POLICY_SOURCE_CHUNK_HASH_DOMAIN_V1 = (
    "dexteleop.relative_policy_source_chunk.v1"
)
RELATIVE_POLICY_SOURCE_ACTION_HASH_DOMAIN_V1 = (
    "dexteleop.relative_policy_source_action.v1"
)
CANONICAL_POLICY_ROBOT_ACTION_CHUNK_HASH_DOMAIN_V1 = (
    "dexteleop.canonical_policy_robot_action_chunk.v1"
)


def _nonempty(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def canonical_policy_robot_action_hash(action: RobotAction) -> str:
    """Hash the immutable execution-relevant RobotAction payload."""

    frozen = freeze_robot_action(action)
    return canonical_content_hash(
        robot_action_semantic_payload(frozen),
        domain=CANONICAL_POLICY_ROBOT_ACTION_HASH_DOMAIN_V1,
    )


@dataclass(frozen=True)
class PolicyActionCanonicalizationReceipt:
    """Representation lineage for one canonical policy-action row."""

    source_pose_mode: str
    observation_cut_id: str
    source_action_hash: str
    canonical_action_hash: str
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str
    conversion_contract_hash: str
    anchor_identity_hash: str | None = None
    source_chunk_hash: str | None = None
    canonical_chunk_hash: str | None = None
    source_index: int | None = None

    def __post_init__(self) -> None:
        mode = normalize_action_pose_mode(self.source_pose_mode)
        object.__setattr__(self, "source_pose_mode", mode)
        for name in (
            "observation_cut_id",
            "source_action_hash",
            "canonical_action_hash",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
            "conversion_contract_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        if mode == ACTION_POSE_MODE_RELATIVE:
            object.__setattr__(
                self,
                "anchor_identity_hash",
                _nonempty(self.anchor_identity_hash or "", name="anchor_identity_hash"),
            )
        elif self.anchor_identity_hash is not None:
            raise ValueError("Absolute canonicalization must not carry an anchor identity.")
        chunk_fields = (
            self.source_chunk_hash,
            self.canonical_chunk_hash,
            self.source_index,
        )
        if any(value is not None for value in chunk_fields):
            if any(value is None for value in chunk_fields):
                raise ValueError(
                    "Chunk-bound canonicalization requires source_chunk_hash, "
                    "canonical_chunk_hash and source_index together."
                )
            object.__setattr__(
                self,
                "source_chunk_hash",
                _nonempty(self.source_chunk_hash or "", name="source_chunk_hash"),
            )
            object.__setattr__(
                self,
                "canonical_chunk_hash",
                _nonempty(self.canonical_chunk_hash or "", name="canonical_chunk_hash"),
            )
            index = self.source_index
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("source_index must be a non-negative integer.")
            object.__setattr__(self, "source_index", int(index))

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "source_pose_mode": self.source_pose_mode,
            "observation_cut_id": self.observation_cut_id,
            "source_action_hash": self.source_action_hash,
            "canonical_action_hash": self.canonical_action_hash,
            "action_codec_hash": self.action_codec_hash,
            "frame_contract_hash": self.frame_contract_hash,
            "layout_hash": self.layout_hash,
            "normalization_hash": self.normalization_hash,
            "conversion_contract_hash": self.conversion_contract_hash,
            "anchor_identity_hash": self.anchor_identity_hash,
        }
        domain = POLICY_ACTION_CANONICALIZATION_RECEIPT_HASH_DOMAIN_V1
        if self.source_chunk_hash is not None:
            payload.update(
                {
                    "source_chunk_hash": self.source_chunk_hash,
                    "canonical_chunk_hash": self.canonical_chunk_hash,
                    "source_index": self.source_index,
                }
            )
            domain = POLICY_ACTION_CANONICALIZATION_RECEIPT_HASH_DOMAIN_V2
        return canonical_content_hash(payload, domain=domain)

    @property
    def receipt_id(self) -> str:
        return f"policy-action-canonicalization:{self.semantic_content_hash}"


@dataclass(frozen=True)
class CanonicalPolicyAction:
    """One immutable canonical action with a verified representation receipt."""

    action: RobotAction
    canonicalization_receipt: PolicyActionCanonicalizationReceipt
    schema_id: str = CANONICAL_POLICY_ACTION_SCHEMA_ID_V1

    def __post_init__(self) -> None:
        schema_id = _nonempty(self.schema_id, name="schema_id")
        if schema_id != CANONICAL_POLICY_ACTION_SCHEMA_ID_V1:
            raise ValueError(f"Unsupported canonical policy action schema: {schema_id!r}.")
        object.__setattr__(self, "schema_id", schema_id)
        try:
            frozen = freeze_robot_action(self.action)
        except ProposalAdmissionError as exc:
            raise ValueError(f"Invalid canonical RobotAction: {exc}") from exc
        if frozen.arm_qpos_by_side:
            raise ValueError(
                "Canonical production policy actions require arm pose; arm qpos is test/debug only."
            )
        canonical_hash = canonical_policy_robot_action_hash(frozen)
        if canonical_hash != self.canonicalization_receipt.canonical_action_hash:
            raise ValueError("Canonical action hash does not match canonicalization receipt.")
        object.__setattr__(self, "action", frozen)

    @property
    def canonical_action_hash(self) -> str:
        return self.canonicalization_receipt.canonical_action_hash

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema_id": self.schema_id,
                "canonical_action_hash": self.canonical_action_hash,
                "canonicalization_receipt_hash": (
                    self.canonicalization_receipt.semantic_content_hash
                ),
            },
            domain=CANONICAL_POLICY_ACTION_HASH_DOMAIN_V1,
        )

    def to_legacy_normalized_action(self) -> NormalizedAction:
        """Derive, but never store, the legacy 26D deploy compatibility view."""

        if self.action.end_effector_targets_by_side:
            raise ValueError(
                "Native typed end-effector targets have no lossless legacy 26D view."
            )
        encoding = encode_robot_action(self.action)
        vector = np.asarray(encoding.vector, dtype=np.float32).reshape(-1)
        if vector.shape != (ACTION_DIM,):
            raise ValueError(
                f"RobotAction {self.action.action_space!r} has no legacy {ACTION_DIM}D view."
            )
        return NormalizedAction(
            vector=vector.copy(),
            metadata={
                **dict(self.action.metadata),
                "source": self.action.source,
                "canonical_policy_action_hash": self.semantic_content_hash,
            },
        )


def build_canonical_policy_action(
    action: RobotAction,
    *,
    source_pose_mode: str,
    observation_cut_id: str,
    source_action_hash: str,
    action_codec_hash: str,
    frame_contract_hash: str,
    layout_hash: str,
    normalization_hash: str,
    conversion_contract_hash: str,
    anchor_identity_hash: str | None = None,
    source_chunk_hash: str | None = None,
    canonical_chunk_hash: str | None = None,
    source_index: int | None = None,
) -> CanonicalPolicyAction:
    """Freeze a RobotAction and bind its absolute/relative source lineage."""

    canonical_hash = canonical_policy_robot_action_hash(action)
    receipt = PolicyActionCanonicalizationReceipt(
        source_pose_mode=source_pose_mode,
        observation_cut_id=observation_cut_id,
        source_action_hash=source_action_hash,
        canonical_action_hash=canonical_hash,
        action_codec_hash=action_codec_hash,
        frame_contract_hash=frame_contract_hash,
        layout_hash=layout_hash,
        normalization_hash=normalization_hash,
        conversion_contract_hash=conversion_contract_hash,
        anchor_identity_hash=anchor_identity_hash,
        source_chunk_hash=source_chunk_hash,
        canonical_chunk_hash=canonical_chunk_hash,
        source_index=source_index,
    )
    return CanonicalPolicyAction(
        action=action,
        canonicalization_receipt=receipt,
    )


def canonical_policy_robot_action_chunk_hash(
    actions: Sequence[RobotAction],
) -> str:
    """Hash one ordered canonical RobotAction chunk without receipt recursion."""

    return canonical_content_hash(
        tuple(canonical_policy_robot_action_hash(action) for action in actions),
        domain=CANONICAL_POLICY_ROBOT_ACTION_CHUNK_HASH_DOMAIN_V1,
    )


class AbsolutePolicyActionChunkCanonicalizer:
    """Convert one declared-absolute model chunk into receipt-bound actions.

    ``decode_robot_actions`` owns only model-space codec/frame conversion.  It
    must return pre-safety canonical ``RobotAction`` rows and has no worker,
    plan, safety, target or dispatch authority.
    """

    def __init__(
        self,
        *,
        decode_robot_actions: Callable[
            [np.ndarray, np.ndarray | None, "PolicyInferenceRequest"],
            Sequence[RobotAction],
        ],
        conversion_contract_hash: str,
    ) -> None:
        if not callable(decode_robot_actions):
            raise TypeError("decode_robot_actions must be callable.")
        self._decode_robot_actions = decode_robot_actions
        self.conversion_contract_hash = _nonempty(
            conversion_contract_hash,
            name="conversion_contract_hash",
        )

    def __call__(
        self,
        model_space_actions: np.ndarray,
        model_space_discrete_actions: np.ndarray | None,
        request: "PolicyInferenceRequest",
    ) -> Sequence[CanonicalPolicyAction]:
        context = getattr(request, "action_representation_context", None)
        if context is not None and (
            context.source_pose_mode != ACTION_POSE_MODE_ABSOLUTE
        ):
            raise ValueError(
                "Absolute canonicalization cannot use a relative request context."
            )
        actions = np.asarray(model_space_actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] <= 0:
            raise ValueError("absolute model-space actions must be a non-empty rank-2 matrix.")
        if np.any(~np.isfinite(actions)):
            raise ValueError("absolute model-space actions must be finite.")
        discrete: np.ndarray | None
        if model_space_discrete_actions is None:
            discrete = None
        else:
            raw_discrete = np.asarray(model_space_discrete_actions)
            if raw_discrete.dtype.kind not in {"i", "u"}:
                raise TypeError("absolute model-space discrete actions must be integer.")
            discrete = np.asarray(raw_discrete, dtype=np.int64)
            if discrete.ndim != 2 or discrete.shape[0] != actions.shape[0]:
                raise ValueError(
                    "absolute model-space discrete actions must share the action horizon."
                )

        for name in (
            "semantic_content_hash",
            "observation_cut_id",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
        ):
            if not hasattr(request, name):
                raise TypeError(f"request is missing {name}.")
        source_chunk_hash = canonical_content_hash(
            {
                "request_hash": request.semantic_content_hash,
                "observation_cut_id": request.observation_cut_id,
                "action_codec_hash": request.action_codec_hash,
                "frame_contract_hash": request.frame_contract_hash,
                "layout_hash": request.layout_hash,
                "normalization_hash": request.normalization_hash,
                "source_pose_mode": "absolute",
                "model_space_actions": actions,
                "model_space_discrete_actions": discrete,
            },
            domain=ABSOLUTE_POLICY_SOURCE_CHUNK_HASH_DOMAIN_V1,
        )
        decoded = tuple(
            self._decode_robot_actions(actions, discrete, request)
        )
        if len(decoded) != actions.shape[0]:
            raise ValueError(
                "absolute RobotAction decoder output count differs from source horizon."
            )
        if any(not isinstance(action, RobotAction) for action in decoded):
            raise TypeError("absolute RobotAction decoder must return RobotAction rows.")
        canonical_chunk_hash = canonical_policy_robot_action_chunk_hash(decoded)
        canonical: list[CanonicalPolicyAction] = []
        for index, action in enumerate(decoded):
            source_action_hash = canonical_content_hash(
                {
                    "source_chunk_hash": source_chunk_hash,
                    "source_index": index,
                    "continuous_action": actions[index],
                    "discrete_action": (
                        None if discrete is None else discrete[index]
                    ),
                },
                domain=ABSOLUTE_POLICY_SOURCE_ACTION_HASH_DOMAIN_V1,
            )
            canonical.append(
                build_canonical_policy_action(
                    action,
                    source_pose_mode="absolute",
                    observation_cut_id=request.observation_cut_id,
                    source_action_hash=source_action_hash,
                    action_codec_hash=request.action_codec_hash,
                    frame_contract_hash=request.frame_contract_hash,
                    layout_hash=request.layout_hash,
                    normalization_hash=request.normalization_hash,
                    conversion_contract_hash=self.conversion_contract_hash,
                    source_chunk_hash=source_chunk_hash,
                    canonical_chunk_hash=canonical_chunk_hash,
                    source_index=index,
                )
            )
        return tuple(canonical)


class RelativePolicyActionChunkCanonicalizer:
    """Decode relative arm poses using only the immutable request context."""

    def __init__(
        self,
        *,
        decode_absolute_robot_actions: Callable[
            [np.ndarray, np.ndarray | None, "PolicyInferenceRequest"],
            Sequence[RobotAction],
        ],
        conversion_contract_hash: str,
        arm_pose_slices: Mapping[str, slice] = DEFAULT_ARM_POSE_SLICES,
    ) -> None:
        if not callable(decode_absolute_robot_actions):
            raise TypeError("decode_absolute_robot_actions must be callable.")
        self._decode_absolute_robot_actions = decode_absolute_robot_actions
        self.conversion_contract_hash = _nonempty(
            conversion_contract_hash,
            name="conversion_contract_hash",
        )
        self.arm_pose_slices = dict(arm_pose_slices)

    def __call__(
        self,
        model_space_actions: np.ndarray,
        model_space_discrete_actions: np.ndarray | None,
        request: "PolicyInferenceRequest",
    ) -> Sequence[CanonicalPolicyAction]:
        context = getattr(request, "action_representation_context", None)
        if context is None or context.source_pose_mode != ACTION_POSE_MODE_RELATIVE:
            raise ValueError(
                "Relative canonicalization requires a relative request context."
            )
        source = np.asarray(model_space_actions, dtype=np.float32)
        if source.ndim != 2 or source.shape[0] <= 0 or np.any(~np.isfinite(source)):
            raise ValueError(
                "relative model-space actions must be a non-empty finite rank-2 matrix."
            )
        discrete: np.ndarray | None
        if model_space_discrete_actions is None:
            discrete = None
        else:
            raw_discrete = np.asarray(model_space_discrete_actions)
            if raw_discrete.dtype.kind not in {"i", "u"}:
                raise TypeError("relative model-space discrete actions must be integer.")
            discrete = np.asarray(raw_discrete, dtype=np.int64)
            if discrete.ndim != 2 or discrete.shape[0] != source.shape[0]:
                raise ValueError(
                    "relative model-space discrete actions must share the action horizon."
                )
        absolute_rows = convert_arm_pose_chunk(
            source,
            anchors_by_side=context.arm_anchor_pose_by_side,
            source_mode=ACTION_POSE_MODE_RELATIVE,
            target_mode=ACTION_POSE_MODE_ABSOLUTE,
            arm_pose_slices=self.arm_pose_slices,
            active_sides=context.active_sides,
        )
        decoded = tuple(
            self._decode_absolute_robot_actions(absolute_rows, discrete, request)
        )
        if len(decoded) != source.shape[0]:
            raise ValueError(
                "relative RobotAction decoder output count differs from source horizon."
            )
        if any(not isinstance(action, RobotAction) for action in decoded):
            raise TypeError("relative RobotAction decoder must return RobotAction rows.")
        for action in decoded:
            if set(action.arm_pose_by_side) != set(context.active_sides):
                raise ValueError(
                    "Relative decoded arm coverage differs from request context."
                )
            for side in context.active_sides:
                if (
                    action.arm_frame(side)
                    != context.arm_reference_frame_by_side[side]
                    or action.arm_controlled_frame(side)
                    != context.arm_controlled_frame_by_side[side]
                ):
                    raise ValueError(
                        f"Relative decoded action frame differs from request context for {side!r}."
                    )

        source_chunk_hash = canonical_content_hash(
            {
                "request_hash": request.semantic_content_hash,
                "representation_context_hash": context.semantic_content_hash,
                "model_space_actions": source,
                "model_space_discrete_actions": discrete,
            },
            domain=RELATIVE_POLICY_SOURCE_CHUNK_HASH_DOMAIN_V1,
        )
        canonical_chunk_hash = canonical_policy_robot_action_chunk_hash(decoded)
        canonical: list[CanonicalPolicyAction] = []
        for index, action in enumerate(decoded):
            source_action_hash = canonical_content_hash(
                {
                    "source_chunk_hash": source_chunk_hash,
                    "source_index": index,
                    "continuous_action": source[index],
                    "discrete_action": (
                        None if discrete is None else discrete[index]
                    ),
                },
                domain=RELATIVE_POLICY_SOURCE_ACTION_HASH_DOMAIN_V1,
            )
            canonical.append(
                build_canonical_policy_action(
                    action,
                    source_pose_mode=ACTION_POSE_MODE_RELATIVE,
                    observation_cut_id=request.observation_cut_id,
                    source_action_hash=source_action_hash,
                    action_codec_hash=request.action_codec_hash,
                    frame_contract_hash=request.frame_contract_hash,
                    layout_hash=request.layout_hash,
                    normalization_hash=request.normalization_hash,
                    conversion_contract_hash=self.conversion_contract_hash,
                    anchor_identity_hash=context.semantic_content_hash,
                    source_chunk_hash=source_chunk_hash,
                    canonical_chunk_hash=canonical_chunk_hash,
                    source_index=index,
                )
            )
        return tuple(canonical)


__all__ = [
    "ABSOLUTE_POLICY_SOURCE_ACTION_HASH_DOMAIN_V1",
    "ABSOLUTE_POLICY_SOURCE_CHUNK_HASH_DOMAIN_V1",
    "CANONICAL_POLICY_ACTION_HASH_DOMAIN_V1",
    "CANONICAL_POLICY_ACTION_SCHEMA_ID_V1",
    "CANONICAL_POLICY_ROBOT_ACTION_HASH_DOMAIN_V1",
    "CANONICAL_POLICY_ROBOT_ACTION_CHUNK_HASH_DOMAIN_V1",
    "POLICY_ACTION_CANONICALIZATION_RECEIPT_HASH_DOMAIN_V1",
    "POLICY_ACTION_CANONICALIZATION_RECEIPT_HASH_DOMAIN_V2",
    "RELATIVE_POLICY_SOURCE_ACTION_HASH_DOMAIN_V1",
    "RELATIVE_POLICY_SOURCE_CHUNK_HASH_DOMAIN_V1",
    "AbsolutePolicyActionChunkCanonicalizer",
    "CanonicalPolicyAction",
    "PolicyActionCanonicalizationReceipt",
    "RelativePolicyActionChunkCanonicalizer",
    "build_canonical_policy_action",
    "canonical_policy_robot_action_chunk_hash",
    "canonical_policy_robot_action_hash",
]
