"""Deterministic model-continuation training views over artifact actions.

This module owns an offline projection only. It does not read or rewrite a
dataset, normalize model inputs, create framework tensors, call a model, or
participate in an RTC runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_LOSS_ALL_VALID_V1,
    MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    normalize_model_continuation_contract,
    require_trained_model_rtc_v1_contract,
)
from dataset_contracts.policy_artifact import PolicyArtifactContract
from dataset_contracts.pose_frames import PoseFrameContract
from dataset_contracts.v22 import DatasetLayoutBindingError
from robot_motion.relative_action import ACTION_POSE_MODE_ABSOLUTE

from .pose_chunk import (
    PoseActionChunkSample,
    pose_action_chunk_from_episode_rows,
)


MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1 = (
    "dexteleop_model_continuation_training_view_v1"
)
MODEL_CONTINUATION_TRAINING_VIEW_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_training_view.v1"
)
MODEL_CONTINUATION_ARTIFACT_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_artifact_binding.v1"
)
MODEL_CONTINUATION_TRAINING_SAMPLING_SCHEMA_V1 = (
    "dexteleop_model_continuation_training_sampling_v1"
)
MODEL_CONTINUATION_TRAINING_SAMPLING_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_training_sampling.v1"
)
MODEL_CONTINUATION_TRAINING_WINDOW_SCHEMA_V1 = (
    "dexteleop_model_continuation_training_window_v1"
)
MODEL_CONTINUATION_TRAINING_WINDOW_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_training_window.v1"
)
MODEL_CONTINUATION_TRAINING_CAUSAL_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_training_causal_binding.v1"
)
MODEL_CONTINUATION_SAMPLING_HASH_INDEXED_EXPLICIT_PAIRS_V1 = (
    "hash_indexed_uniform_explicit_pairs_v1"
)
TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1 = (
    "dataset_interface.trajectory_overlap_teacher_v1"
)
_TEACHER_OVERLAP_PROJECTION_SPEC_V1 = {
    "schema": MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1,
    "source_kind": TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    "action_representation": "artifact_action_continuous_absolute_v1",
    "delay_selection": "explicit_generation_steps_v1",
    "prefix_selection": "contiguous_generation_prefix_v1",
    "unselected_prefix_fill": "zero_v1",
    "prefix_mask": "float32_one_is_conditioned_v1",
    "target_padding_mask": "bool_true_is_padding_v1",
    "loss_mask": "bool_true_contributes_to_loss_v1",
}
TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1 = "sha256:" + (
    canonical_content_hash(
        _TEACHER_OVERLAP_PROJECTION_SPEC_V1,
        domain="dexteleop.model_continuation_teacher_projection_spec.v1",
    )
)


class ModelContinuationViewError(DatasetLayoutBindingError):
    """Raised when a continuation training view cannot be derived exactly."""


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ModelContinuationViewError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ModelContinuationViewError(f"{name} must be non-empty.")
    return normalized


def _exact_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ModelContinuationViewError(f"{name} must be an integer.")
    return int(value)


def _artifact_binding_hash(artifact: PolicyArtifactContract) -> str:
    assert artifact.pose_action_chunk is not None
    assert artifact.pose_training_data is not None
    assert artifact.model_continuation is not None
    return "sha256:" + canonical_content_hash(
        {
            "policy_io_layout": artifact.policy_io_layout.to_dict(),
            "pose_action_chunk": artifact.pose_action_chunk.to_dict(),
            "pose_training_data": artifact.pose_training_data.to_dict(),
            "pose_frame_contract_hashes": list(
                artifact.pose_frame_contract_hashes
            ),
            "model_continuation": artifact.model_continuation.to_dict(),
        },
        domain=MODEL_CONTINUATION_ARTIFACT_BINDING_HASH_DOMAIN_V1,
    )


@dataclass(frozen=True)
class ModelContinuationTrainingSamplingSpec:
    """Replayable selection over explicitly approved delay/prefix pairs."""

    delay_prefix_pairs: tuple[tuple[int, int], ...]
    selection_salt: str
    strategy: str = (
        MODEL_CONTINUATION_SAMPLING_HASH_INDEXED_EXPLICIT_PAIRS_V1
    )
    schema: str = MODEL_CONTINUATION_TRAINING_SAMPLING_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != MODEL_CONTINUATION_TRAINING_SAMPLING_SCHEMA_V1:
            raise ModelContinuationViewError(
                f"Unsupported continuation sampling schema {self.schema!r}."
            )
        if (
            self.strategy
            != MODEL_CONTINUATION_SAMPLING_HASH_INDEXED_EXPLICIT_PAIRS_V1
        ):
            raise ModelContinuationViewError(
                f"Unsupported continuation sampling strategy {self.strategy!r}."
            )
        salt = _nonempty(self.selection_salt, name="selection_salt")
        normalized: list[tuple[int, int]] = []
        for index, pair in enumerate(self.delay_prefix_pairs):
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, (str, bytes))
                or len(pair) != 2
            ):
                raise ModelContinuationViewError(
                    "Each delay_prefix_pairs entry must contain delay and "
                    f"prefix integers; entry {index} is invalid."
                )
            delay = _exact_int(pair[0], name=f"delay_prefix_pairs[{index}].delay")
            prefix = _exact_int(
                pair[1],
                name=f"delay_prefix_pairs[{index}].prefix",
            )
            if delay < 0 or prefix < 0 or prefix > delay:
                raise ModelContinuationViewError(
                    "Each delay/prefix pair must satisfy "
                    "0 <= prefix_steps <= forecast_delay_steps."
                )
            normalized.append((delay, prefix))
        if not normalized:
            raise ModelContinuationViewError(
                "delay_prefix_pairs must contain at least one pair."
            )
        if len(set(normalized)) != len(normalized):
            raise ModelContinuationViewError(
                "delay_prefix_pairs must not contain duplicates."
            )
        object.__setattr__(self, "delay_prefix_pairs", tuple(sorted(normalized)))
        object.__setattr__(self, "selection_salt", salt)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "strategy": self.strategy,
            "delay_prefix_pairs": [
                [delay, prefix]
                for delay, prefix in self.delay_prefix_pairs
            ],
            "selection_salt": self.selection_salt,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.semantic_payload()

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ModelContinuationTrainingSamplingSpec":
        expected = {
            "schema",
            "strategy",
            "delay_prefix_pairs",
            "selection_salt",
        }
        unknown = sorted(set(payload).difference(expected))
        if unknown:
            raise ModelContinuationViewError(
                f"Continuation sampling spec has unknown entries: {unknown}."
            )
        pairs = payload.get("delay_prefix_pairs")
        if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
            raise ModelContinuationViewError(
                "delay_prefix_pairs must be a sequence."
            )
        return cls(
            delay_prefix_pairs=tuple(pairs),
            selection_salt=str(payload.get("selection_salt", "")),
            strategy=str(payload.get("strategy", "")),
            schema=str(payload.get("schema", "")),
        )

    @property
    def semantic_content_hash(self) -> str:
        return "sha256:" + canonical_content_hash(
            self.semantic_payload(),
            domain=MODEL_CONTINUATION_TRAINING_SAMPLING_HASH_DOMAIN_V1,
        )


def select_model_continuation_training_delay_prefix(
    spec: ModelContinuationTrainingSamplingSpec,
    artifact: PolicyArtifactContract,
    *,
    source_episode_id: str,
    source_start_index: int,
    draw_index: int,
    valid_action_count: int,
) -> tuple[int, int]:
    """Select one legal pair without mutable RNG or worker-order dependence."""

    if not isinstance(spec, ModelContinuationTrainingSamplingSpec):
        raise TypeError(
            "spec must be a ModelContinuationTrainingSamplingSpec."
        )
    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be a PolicyArtifactContract.")
    if artifact.model_continuation is None:
        raise ModelContinuationViewError(
            "Trained model-RTC sampling requires an artifact continuation "
            "contract."
        )
    try:
        normalized = require_trained_model_rtc_v1_contract(
            artifact.model_continuation
        )
    except DatasetLayoutBindingError as exc:
        raise ModelContinuationViewError(str(exc)) from exc
    runtime = normalized.runtime
    valid_count = _exact_int(valid_action_count, name="valid_action_count")
    if valid_count < 1 or valid_count > runtime.prediction_horizon_steps:
        raise ModelContinuationViewError(
            "valid_action_count must be within the prediction horizon."
        )
    for delay, prefix in spec.delay_prefix_pairs:
        if not (
            runtime.minimum_forecast_delay_steps
            <= delay
            <= runtime.maximum_forecast_delay_steps
        ):
            raise ModelContinuationViewError(
                "Sampling spec delay is outside the artifact-supported range."
            )
        if not (
            runtime.minimum_prefix_steps
            <= prefix
            <= runtime.maximum_prefix_steps
        ):
            raise ModelContinuationViewError(
                "Sampling spec prefix is outside the artifact-supported range."
            )
        if prefix > delay:
            raise ModelContinuationViewError(
                "Sampling spec prefix cannot exceed its delay."
            )
    eligible = tuple(
        pair
        for pair in spec.delay_prefix_pairs
        if pair[0] < valid_count
    )
    if not eligible:
        raise ModelContinuationViewError(
            "Sampling spec has no pair with a real executable suffix."
        )
    episode = _nonempty(source_episode_id, name="source_episode_id")
    start = _exact_int(source_start_index, name="source_start_index")
    draw = _exact_int(draw_index, name="draw_index")
    if start < 0 or draw < 0:
        raise ModelContinuationViewError(
            "source_start_index and draw_index must be non-negative."
        )
    selection_digest = canonical_content_hash(
        {
            "sampling_spec_hash": spec.semantic_content_hash,
            "continuation_contract_hash": normalized.source_contract_hash,
            "source_episode_id": episode,
            "source_start_index": start,
            "draw_index": draw,
        },
        domain="dexteleop.model_continuation_training_pair_choice.v1",
    )
    return eligible[int(selection_digest[:16], 16) % len(eligible)]


def model_continuation_training_masks(
    artifact: PolicyArtifactContract,
    *,
    forecast_delay_steps: int,
    prefix_steps: int,
    valid_action_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical prefix, padding and loss masks for one trained window."""

    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be a PolicyArtifactContract.")
    continuation = artifact.model_continuation
    if continuation is None:
        raise ModelContinuationViewError(
            "Trained model-RTC masks require a continuation contract."
        )
    normalized = normalize_model_continuation_contract(continuation)
    runtime = normalized.runtime
    training = normalized.training
    if training is None:
        raise ModelContinuationViewError(
            "Inference-only continuation has no training masks."
        )
    delay = _exact_int(
        forecast_delay_steps,
        name="forecast_delay_steps",
    )
    prefix = _exact_int(prefix_steps, name="prefix_steps")
    valid_count = _exact_int(
        valid_action_count,
        name="valid_action_count",
    )
    horizon = runtime.prediction_horizon_steps
    if not (
        runtime.minimum_forecast_delay_steps
        <= delay
        <= runtime.maximum_forecast_delay_steps
    ):
        raise ModelContinuationViewError(
            "forecast_delay_steps is outside the artifact-supported range."
        )
    if not (
        runtime.minimum_prefix_steps
        <= prefix
        <= runtime.maximum_prefix_steps
    ):
        raise ModelContinuationViewError(
            "prefix_steps is outside the artifact-supported range."
        )
    if prefix > delay:
        raise ModelContinuationViewError(
            "prefix_steps cannot exceed forecast_delay_steps."
        )
    if not 1 <= valid_count <= horizon:
        raise ModelContinuationViewError(
            "valid_action_count must be within the prediction horizon."
        )
    if delay >= valid_count:
        raise ModelContinuationViewError(
            "forecast_delay_steps leaves no real executable target."
        )
    prefix_mask = np.arange(horizon) < prefix
    target_is_pad = np.arange(horizon) >= valid_count
    if (
        training.loss_mask_semantics
        == MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1
    ):
        loss_mask = ~target_is_pad & ~prefix_mask
    elif training.loss_mask_semantics == MODEL_CONTINUATION_LOSS_ALL_VALID_V1:
        loss_mask = ~target_is_pad
    else:
        raise ModelContinuationViewError(
            "Unsupported loss_mask_semantics."
        )
    return (
        prefix_mask.astype(np.float32),
        target_is_pad.astype(bool),
        loss_mask.astype(bool),
    )


@dataclass(frozen=True)
class ModelContinuationTrainingSample:
    """One artifact-action prefix/target/mask view.

    `prefix_mask == 1` means that generation step is conditioned.
    `target_is_pad == True` means the target step is padding.
    `loss_mask == True` means the target step contributes to loss.
    """

    conditioning_prefix: np.ndarray
    prefix_mask: np.ndarray
    target_action: np.ndarray
    target_is_pad: np.ndarray
    loss_mask: np.ndarray
    forecast_delay_steps: int
    prefix_steps: int
    training_prefix_source_kind: str
    source_episode_id: str
    source_start_index: int
    artifact_binding_hash: str
    continuation_contract_hash: str
    layout_hash: str
    chunk_contract_hash: str
    pose_frame_contract_hash: str
    projection_implementation_id: str
    projection_parameters_hash: str
    loss_mask_semantics: str
    schema: str = MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1:
            raise ModelContinuationViewError(
                f"Unsupported continuation training view schema {self.schema!r}."
            )
        target = np.asarray(self.target_action, dtype=np.float32)
        prefix = np.asarray(self.conditioning_prefix, dtype=np.float32)
        prefix_mask = np.asarray(self.prefix_mask, dtype=np.float32)
        target_is_pad = np.asarray(self.target_is_pad, dtype=bool)
        loss_mask = np.asarray(self.loss_mask, dtype=bool)
        if target.ndim != 2 or target.shape[0] < 2 or target.shape[1] < 1:
            raise ModelContinuationViewError(
                "target_action must be 2D [horizon, continuous_action_dim]."
            )
        horizon = target.shape[0]
        if prefix.shape != target.shape:
            raise ModelContinuationViewError(
                "conditioning_prefix must match target_action shape."
            )
        if (
            prefix_mask.shape != (horizon,)
            or target_is_pad.shape != (horizon,)
            or loss_mask.shape != (horizon,)
        ):
            raise ModelContinuationViewError(
                "Continuation masks must each have shape [horizon]."
            )
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prefix)):
            raise ModelContinuationViewError(
                "Continuation action values must be finite."
            )

        delay = _exact_int(
            self.forecast_delay_steps,
            name="forecast_delay_steps",
        )
        prefix_steps = _exact_int(self.prefix_steps, name="prefix_steps")
        start = _exact_int(self.source_start_index, name="source_start_index")
        if delay < 0 or delay >= horizon:
            raise ModelContinuationViewError(
                "forecast_delay_steps must be within the prediction horizon."
            )
        if prefix_steps < 0 or prefix_steps > delay:
            raise ModelContinuationViewError(
                "prefix_steps must be between zero and forecast_delay_steps."
            )
        if start < 0:
            raise ModelContinuationViewError(
                "source_start_index must be non-negative."
            )

        expected_prefix_mask = np.zeros((horizon,), dtype=np.float32)
        expected_prefix_mask[:prefix_steps] = 1.0
        if not np.array_equal(prefix_mask, expected_prefix_mask):
            raise ModelContinuationViewError(
                "prefix_mask must be a contiguous one-prefix/zero-suffix."
            )
        valid_count = int(np.count_nonzero(~target_is_pad))
        expected_padding = np.arange(horizon) >= valid_count
        if valid_count < 1 or not np.array_equal(
            target_is_pad,
            expected_padding,
        ):
            raise ModelContinuationViewError(
                "target_is_pad must be a contiguous false-prefix/true-tail."
            )
        if delay >= valid_count:
            raise ModelContinuationViewError(
                "forecast_delay_steps must leave a non-padding executable suffix."
            )
        if np.any((prefix_mask > 0) & target_is_pad):
            raise ModelContinuationViewError(
                "Conditioning prefix and target padding cannot overlap."
            )

        selected = prefix_mask > 0
        if not np.array_equal(prefix[selected], target[selected]):
            raise ModelContinuationViewError(
                "Teacher prefix must equal selected artifact-action targets."
            )
        if np.any(prefix[~selected] != 0):
            raise ModelContinuationViewError(
                "Unselected conditioning prefix entries must be zero-filled."
            )
        if (
            self.loss_mask_semantics
            == MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1
        ):
            expected_loss = ~target_is_pad & ~selected
        elif self.loss_mask_semantics == MODEL_CONTINUATION_LOSS_ALL_VALID_V1:
            expected_loss = ~target_is_pad
        else:
            raise ModelContinuationViewError(
                "Unsupported loss_mask_semantics."
            )
        if not np.array_equal(loss_mask, expected_loss):
            raise ModelContinuationViewError(
                "loss_mask does not match its declared semantics."
            )

        source_kind = _nonempty(
            self.training_prefix_source_kind,
            name="training_prefix_source_kind",
        )
        if source_kind != TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1:
            raise ModelContinuationViewError(
                "D2 supports only trajectory_overlap_teacher_v1 provenance."
            )
        for name in (
            "source_episode_id",
            "artifact_binding_hash",
            "continuation_contract_hash",
            "layout_hash",
            "chunk_contract_hash",
            "pose_frame_contract_hash",
            "projection_implementation_id",
            "projection_parameters_hash",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        if (
            self.projection_implementation_id
            != TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1
            or self.projection_parameters_hash
            != TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1
        ):
            raise ModelContinuationViewError(
                "Training sample projection identity does not match D2."
            )

        object.__setattr__(self, "conditioning_prefix", prefix.copy())
        object.__setattr__(self, "prefix_mask", prefix_mask.copy())
        object.__setattr__(self, "target_action", target.copy())
        object.__setattr__(self, "target_is_pad", target_is_pad.copy())
        object.__setattr__(self, "loss_mask", loss_mask.copy())
        object.__setattr__(self, "forecast_delay_steps", delay)
        object.__setattr__(self, "prefix_steps", prefix_steps)
        object.__setattr__(self, "source_start_index", start)
        object.__setattr__(self, "training_prefix_source_kind", source_kind)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "conditioning_prefix": self.conditioning_prefix,
            "prefix_mask": self.prefix_mask,
            "target_action": self.target_action,
            "target_is_pad": self.target_is_pad,
            "loss_mask": self.loss_mask,
            "forecast_delay_steps": self.forecast_delay_steps,
            "prefix_steps": self.prefix_steps,
            "training_prefix_source_kind": self.training_prefix_source_kind,
            "source_episode_id": self.source_episode_id,
            "source_start_index": self.source_start_index,
            "artifact_binding_hash": self.artifact_binding_hash,
            "continuation_contract_hash": self.continuation_contract_hash,
            "layout_hash": self.layout_hash,
            "chunk_contract_hash": self.chunk_contract_hash,
            "pose_frame_contract_hash": self.pose_frame_contract_hash,
            "projection_implementation_id": self.projection_implementation_id,
            "projection_parameters_hash": self.projection_parameters_hash,
            "loss_mask_semantics": self.loss_mask_semantics,
        }

    @property
    def sample_hash(self) -> str:
        return "sha256:" + canonical_content_hash(
            self.semantic_payload(),
            domain=MODEL_CONTINUATION_TRAINING_VIEW_HASH_DOMAIN_V1,
        )

    @property
    def valid_action_count(self) -> int:
        return int(np.count_nonzero(~self.target_is_pad))

    def causal_window_binding_payload(self) -> dict[str, Any]:
        """Name the observation/generation/executable indices explicitly."""

        start = self.source_start_index
        return {
            "schema": "dexteleop_model_continuation_causal_binding_v1",
            "source_episode_id": self.source_episode_id,
            "observation_source_index": start,
            "generation_start_source_index": start,
            "first_executable_source_index": (
                start + self.forecast_delay_steps
            ),
            "valid_action_stop_source_index_exclusive": (
                start + self.valid_action_count
            ),
            "forecast_delay_steps": self.forecast_delay_steps,
            "prefix_steps": self.prefix_steps,
            "training_sample_hash": self.sample_hash,
            "artifact_binding_hash": self.artifact_binding_hash,
            "continuation_contract_hash": self.continuation_contract_hash,
            "layout_hash": self.layout_hash,
            "chunk_contract_hash": self.chunk_contract_hash,
            "pose_frame_contract_hash": self.pose_frame_contract_hash,
        }

    @property
    def causal_window_binding_hash(self) -> str:
        return "sha256:" + canonical_content_hash(
            self.causal_window_binding_payload(),
            domain=(
                MODEL_CONTINUATION_TRAINING_CAUSAL_BINDING_HASH_DOMAIN_V1
            ),
        )

    def artifact_action_dict(self) -> dict[str, Any]:
        """Return copies without normalization or framework conversion."""

        return {
            "continuation.prefix": self.conditioning_prefix.copy(),
            "continuation.prefix_mask": self.prefix_mask.copy(),
            "action": self.target_action.copy(),
            "action_is_pad": self.target_is_pad.copy(),
            "action_loss_mask": self.loss_mask.copy(),
            "forecast_delay_steps": self.forecast_delay_steps,
            "training_prefix_source_kind": (
                self.training_prefix_source_kind
            ),
            "model_continuation_data_contract_hash": (
                self.continuation_contract_hash
            ),
            "model_continuation_training_sample_hash": self.sample_hash,
        }


@dataclass(frozen=True)
class ModelContinuationTrainingWindow:
    """One causally bound observation/action view for framework adapters."""

    pose_chunk: PoseActionChunkSample
    continuation: ModelContinuationTrainingSample
    sampling_spec_hash: str
    sampling_draw_index: int
    schema: str = MODEL_CONTINUATION_TRAINING_WINDOW_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != MODEL_CONTINUATION_TRAINING_WINDOW_SCHEMA_V1:
            raise ModelContinuationViewError(
                f"Unsupported continuation training window {self.schema!r}."
            )
        if not isinstance(self.pose_chunk, PoseActionChunkSample):
            raise TypeError("pose_chunk must be a PoseActionChunkSample.")
        if not isinstance(
            self.continuation,
            ModelContinuationTrainingSample,
        ):
            raise TypeError(
                "continuation must be a ModelContinuationTrainingSample."
            )
        sample = self.continuation
        sampling_spec_hash = _nonempty(
            self.sampling_spec_hash,
            name="sampling_spec_hash",
        )
        sampling_draw_index = _exact_int(
            self.sampling_draw_index,
            name="sampling_draw_index",
        )
        if sampling_draw_index < 0:
            raise ModelContinuationViewError(
                "sampling_draw_index must be non-negative."
            )
        metadata_start = _exact_int(
            self.pose_chunk.metadata.get("start_index"),
            name="pose_chunk.metadata.start_index",
        )
        if metadata_start != sample.source_start_index:
            raise ModelContinuationViewError(
                "Observation chunk and continuation source_start_index differ."
            )
        if (
            self.pose_chunk.valid_action_count
            != sample.valid_action_count
            or not np.array_equal(
                self.pose_chunk.action,
                sample.target_action,
            )
            or not np.array_equal(
                self.pose_chunk.action_is_pad,
                sample.target_is_pad,
            )
        ):
            raise ModelContinuationViewError(
                "Observation chunk action target differs from continuation."
            )
        expected_identities = {
            "layout_hash": self.pose_chunk.layout_hash,
            "chunk_contract_hash": self.pose_chunk.chunk_contract_hash,
            "pose_frame_contract_hash": (
                self.pose_chunk.pose_frame_contract_hash
            ),
        }
        for name, expected in expected_identities.items():
            if getattr(sample, name) != expected:
                raise ModelContinuationViewError(
                    f"Observation chunk {name} differs from continuation."
                )
        object.__setattr__(
            self,
            "sampling_spec_hash",
            sampling_spec_hash,
        )
        object.__setattr__(
            self,
            "sampling_draw_index",
            sampling_draw_index,
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "causal_window_binding_hash": (
                self.continuation.causal_window_binding_hash
            ),
            "training_sample_hash": self.continuation.sample_hash,
            "sampling_spec_hash": self.sampling_spec_hash,
            "sampling_draw_index": self.sampling_draw_index,
            "observation_state": self.pose_chunk.observation_state,
            "observation_state_discrete": (
                self.pose_chunk.observation_state_discrete
            ),
            "last_action": self.pose_chunk.last_action,
            "last_action_discrete": (
                self.pose_chunk.last_action_discrete
            ),
            "actual_tcp_pose_by_side": dict(
                self.pose_chunk.actual_tcp_pose_by_side
            ),
        }

    @property
    def semantic_content_hash(self) -> str:
        return "sha256:" + canonical_content_hash(
            self.semantic_payload(),
            domain=MODEL_CONTINUATION_TRAINING_WINDOW_HASH_DOMAIN_V1,
        )

    def training_dict(self) -> dict[str, Any]:
        """Return one framework-neutral observation/action training mapping."""

        payload = self.pose_chunk.model_dict()
        continuation = self.continuation.artifact_action_dict()
        if not np.array_equal(payload["action"], continuation["action"]):
            raise ModelContinuationViewError(
                "Training window action merge is inconsistent."
            )
        if not np.array_equal(
            payload["action_is_pad"],
            continuation["action_is_pad"],
        ):
            raise ModelContinuationViewError(
                "Training window padding merge is inconsistent."
            )
        payload.update(continuation)
        payload.update(
            {
                "model_continuation_sampling_spec_hash": (
                    self.sampling_spec_hash
                ),
                "model_continuation_sampling_draw_index": (
                    self.sampling_draw_index
                ),
                "model_continuation_training_window_hash": (
                    self.semantic_content_hash
                ),
                "model_continuation_causal_window_binding_hash": (
                    self.continuation.causal_window_binding_hash
                ),
                "observation_source_index": (
                    self.continuation.source_start_index
                ),
                "generation_start_source_index": (
                    self.continuation.source_start_index
                ),
                "first_executable_source_index": (
                    self.continuation.source_start_index
                    + self.continuation.forecast_delay_steps
                ),
            }
        )
        return payload


def model_continuation_training_sample_from_pose_chunk(
    chunk: PoseActionChunkSample,
    artifact: PolicyArtifactContract,
    *,
    source_episode_id: str,
    source_start_index: int,
    forecast_delay_steps: int,
    prefix_steps: int,
    training_prefix_source_kind: str = (
        TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1
    ),
) -> ModelContinuationTrainingSample:
    """Project one absolute continuous pose chunk into the D2 teacher view."""

    if not isinstance(chunk, PoseActionChunkSample):
        raise TypeError("chunk must be a PoseActionChunkSample.")
    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be a PolicyArtifactContract.")
    continuation = artifact.model_continuation
    chunk_contract = artifact.pose_action_chunk
    if continuation is None:
        raise ModelContinuationViewError(
            "Policy artifact has no model continuation contract."
        )
    normalized_continuation = normalize_model_continuation_contract(
        continuation
    )
    runtime_contract = normalized_continuation.runtime
    training_contract = normalized_continuation.training
    if training_contract is None:
        raise ModelContinuationViewError(
            "Inference-only continuation has no training projection."
        )
    if chunk_contract is None:
        raise ModelContinuationViewError(
            "D2 requires an artifact-bound pose action chunk contract."
        )
    if (
        training_contract.projection_implementation_id
        != TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1
        or training_contract.projection_parameters_hash
        != TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1
    ):
        raise ModelContinuationViewError(
            "Artifact continuation projection binding does not match the D2 "
            "teacher-overlap implementation."
        )
    source_kind = str(training_prefix_source_kind).strip()
    if source_kind not in training_contract.prefix_source_kinds:
        raise ModelContinuationViewError(
            "Requested training prefix source is not declared by the artifact."
        )
    if source_kind != TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1:
        raise ModelContinuationViewError(
            "D2 cannot derive recorded-policy or synthetic provenance."
        )
    if chunk_contract.action_pose_mode != ACTION_POSE_MODE_ABSOLUTE:
        raise ModelContinuationViewError(
            "D2 supports only absolute artifact-action chunks."
        )
    if artifact.policy_io_layout.action_discrete_dim != 0:
        raise ModelContinuationViewError(
            "D2 supports only continuous artifact actions."
        )
    if chunk.action_discrete.shape != (
        runtime_contract.prediction_horizon_steps,
        0,
    ):
        raise ModelContinuationViewError(
            "D2 requires an empty discrete action matrix."
        )
    if chunk.layout_hash != artifact.layout_hash:
        raise ModelContinuationViewError(
            "Chunk layout hash does not match the policy artifact."
        )
    if chunk.chunk_contract_hash != chunk_contract.contract_hash:
        raise ModelContinuationViewError(
            "Chunk contract hash does not match the policy artifact."
        )
    if chunk.pose_frame_contract_hash not in artifact.pose_frame_contract_hashes:
        raise ModelContinuationViewError(
            "Chunk pose frame contract is not bound by the policy artifact."
        )
    if chunk.action.shape != (
        runtime_contract.prediction_horizon_steps,
        runtime_contract.model_action_dim,
    ):
        raise ModelContinuationViewError(
            "Chunk action shape does not match the continuation contract."
        )
    if not np.all(np.isfinite(chunk.action)):
        raise ModelContinuationViewError(
            "Artifact-action target contains NaN or infinity."
        )

    delay = _exact_int(
        forecast_delay_steps,
        name="forecast_delay_steps",
    )
    selected_prefix_steps = _exact_int(prefix_steps, name="prefix_steps")
    if not (
        runtime_contract.minimum_forecast_delay_steps
        <= delay
        <= runtime_contract.maximum_forecast_delay_steps
    ):
        raise ModelContinuationViewError(
            "forecast_delay_steps is outside the artifact-supported range."
        )
    if not (
        runtime_contract.minimum_prefix_steps
        <= selected_prefix_steps
        <= runtime_contract.maximum_prefix_steps
    ):
        raise ModelContinuationViewError(
            "prefix_steps is outside the artifact-supported range."
        )
    if selected_prefix_steps > delay:
        raise ModelContinuationViewError(
            "prefix_steps cannot exceed forecast_delay_steps."
        )
    if delay >= chunk.valid_action_count:
        raise ModelContinuationViewError(
            "forecast_delay_steps leaves no real executable target."
        )

    prefix_mask, target_is_pad, loss_mask = (
        model_continuation_training_masks(
            artifact,
            forecast_delay_steps=delay,
            prefix_steps=selected_prefix_steps,
            valid_action_count=chunk.valid_action_count,
        )
    )
    conditioning_prefix = np.zeros_like(chunk.action, dtype=np.float32)
    conditioning_prefix[:selected_prefix_steps] = chunk.action[
        :selected_prefix_steps
    ]
    if not np.array_equal(target_is_pad, chunk.action_is_pad):
        raise ModelContinuationViewError(
            "Derived target padding differs from the pose chunk."
        )

    return ModelContinuationTrainingSample(
        conditioning_prefix=conditioning_prefix,
        prefix_mask=prefix_mask,
        target_action=chunk.action,
        target_is_pad=target_is_pad,
        loss_mask=loss_mask,
        forecast_delay_steps=delay,
        prefix_steps=selected_prefix_steps,
        training_prefix_source_kind=source_kind,
        source_episode_id=source_episode_id,
        source_start_index=source_start_index,
        artifact_binding_hash=_artifact_binding_hash(artifact),
        continuation_contract_hash=(
            normalized_continuation.source_contract_hash
        ),
        layout_hash=chunk.layout_hash,
        chunk_contract_hash=chunk.chunk_contract_hash,
        pose_frame_contract_hash=chunk.pose_frame_contract_hash,
        projection_implementation_id=(
            training_contract.projection_implementation_id
        ),
        projection_parameters_hash=(
            training_contract.projection_parameters_hash
        ),
        loss_mask_semantics=training_contract.loss_mask_semantics,
    )


def model_continuation_training_sample_from_episode_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    artifact: PolicyArtifactContract,
    *,
    pose_frame_contract: PoseFrameContract,
    source_episode_id: str,
    start_index: int,
    forecast_delay_steps: int,
    prefix_steps: int,
    active_sides: Sequence[str] = ("left", "right"),
    training_prefix_source_kind: str = (
        TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1
    ),
) -> ModelContinuationTrainingSample:
    """Build a D2 sample from rows already scoped to exactly one episode."""

    if artifact.pose_action_chunk is None:
        raise ModelContinuationViewError(
            "D2 requires an artifact-bound pose action chunk contract."
        )
    chunk = pose_action_chunk_from_episode_rows(
        episode_rows,
        artifact.policy_io_layout,
        pose_frame_contract=pose_frame_contract,
        chunk_contract=artifact.pose_action_chunk,
        start_index=start_index,
        active_sides=active_sides,
    )
    return model_continuation_training_sample_from_pose_chunk(
        chunk,
        artifact,
        source_episode_id=source_episode_id,
        source_start_index=start_index,
        forecast_delay_steps=forecast_delay_steps,
        prefix_steps=prefix_steps,
        training_prefix_source_kind=training_prefix_source_kind,
    )


def model_continuation_training_window_from_episode_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    artifact: PolicyArtifactContract,
    *,
    pose_frame_contract: PoseFrameContract,
    source_episode_id: str,
    start_index: int,
    sampling_spec: ModelContinuationTrainingSamplingSpec,
    draw_index: int,
    active_sides: Sequence[str] = ("left", "right"),
) -> ModelContinuationTrainingWindow:
    """Build the first trained model-RTC observation/action window."""

    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be a PolicyArtifactContract.")
    continuation_contract = artifact.model_continuation
    if continuation_contract is None:
        raise ModelContinuationViewError(
            "Trained model-RTC requires an artifact continuation contract."
        )
    try:
        require_trained_model_rtc_v1_contract(continuation_contract)
    except DatasetLayoutBindingError as exc:
        raise ModelContinuationViewError(str(exc)) from exc
    if artifact.pose_action_chunk is None:
        raise ModelContinuationViewError(
            "Trained model-RTC requires an artifact-bound pose action chunk."
        )
    chunk = pose_action_chunk_from_episode_rows(
        episode_rows,
        artifact.policy_io_layout,
        pose_frame_contract=pose_frame_contract,
        chunk_contract=artifact.pose_action_chunk,
        start_index=start_index,
        active_sides=active_sides,
    )
    delay, prefix = select_model_continuation_training_delay_prefix(
        sampling_spec,
        artifact,
        source_episode_id=source_episode_id,
        source_start_index=start_index,
        draw_index=draw_index,
        valid_action_count=chunk.valid_action_count,
    )
    sample = model_continuation_training_sample_from_pose_chunk(
        chunk,
        artifact,
        source_episode_id=source_episode_id,
        source_start_index=start_index,
        forecast_delay_steps=delay,
        prefix_steps=prefix,
        training_prefix_source_kind=(
            TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1
        ),
    )
    return ModelContinuationTrainingWindow(
        pose_chunk=chunk,
        continuation=sample,
        sampling_spec_hash=sampling_spec.semantic_content_hash,
        sampling_draw_index=draw_index,
    )


__all__ = [
    "MODEL_CONTINUATION_ARTIFACT_BINDING_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_SAMPLING_HASH_INDEXED_EXPLICIT_PAIRS_V1",
    "MODEL_CONTINUATION_TRAINING_CAUSAL_BINDING_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_TRAINING_SAMPLING_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_TRAINING_SAMPLING_SCHEMA_V1",
    "MODEL_CONTINUATION_TRAINING_VIEW_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1",
    "MODEL_CONTINUATION_TRAINING_WINDOW_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_TRAINING_WINDOW_SCHEMA_V1",
    "TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1",
    "TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1",
    "ModelContinuationTrainingSamplingSpec",
    "ModelContinuationTrainingSample",
    "ModelContinuationTrainingWindow",
    "ModelContinuationViewError",
    "model_continuation_training_masks",
    "model_continuation_training_window_from_episode_rows",
    "model_continuation_training_sample_from_episode_rows",
    "model_continuation_training_sample_from_pose_chunk",
    "select_model_continuation_training_delay_prefix",
]
