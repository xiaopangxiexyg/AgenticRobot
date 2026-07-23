"""Pure DexTeleop v2.3 view for the external ``vla-scratch`` baseline.

This module owns no model, transport, device, replay, or deployment runtime.
It binds an existing :class:`PolicyArtifactContract` to the processed sample
shape already consumed by vla-scratch, then projects one canonical pose chunk
and its ordered RGB views into that shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.policy_artifact import PolicyArtifactContract
from dataset_contracts.state_input import STATE_INPUT_MODE_RAW
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_contracts.vla_scratch import (
    VLA_SCRATCH_ACTIONS_KEY,
    VlaScratchProcessedObservation,
    VlaScratchProcessedSampleSpec,
    build_vla_scratch_processed_observation,
    validate_vla_scratch_action_chunk,
)

from .pose_chunk import PoseActionChunkSample


VLA_SCRATCH_DEXTELEOP_BINDING_SCHEMA_V1 = (
    "vla_scratch_dexteleop_binding_v1"
)
VLA_SCRATCH_DEXTELEOP_ARTIFACT_HASH_DOMAIN_V1 = (
    "dexteleop.vla_scratch.policy_artifact.v1"
)
VLA_SCRATCH_DEXTELEOP_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.vla_scratch.dataset_binding.v1"
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _unique_nonempty(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    normalized = tuple(_nonempty(value, name=name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} values must be unique.")
    return normalized


@dataclass(frozen=True)
class VlaScratchDexTeleopBinding:
    """Artifact-derived identity for one DexTeleop processed-sample view."""

    sample_spec: VlaScratchProcessedSampleSpec
    artifact_hash: str
    layout_hash: str
    pose_action_chunk_contract_hash: str
    pose_frame_contract_hashes: tuple[str, ...]
    camera_model_input_keys: tuple[str, ...]
    camera_server_slots: tuple[str, ...]
    required_active_sides: tuple[str, ...] = ("left", "right")
    schema: str = VLA_SCRATCH_DEXTELEOP_BINDING_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != VLA_SCRATCH_DEXTELEOP_BINDING_SCHEMA_V1:
            raise ValueError(
                f"Unsupported vla-scratch DexTeleop binding: {self.schema!r}."
            )
        if not isinstance(self.sample_spec, VlaScratchProcessedSampleSpec):
            raise TypeError("sample_spec must be VlaScratchProcessedSampleSpec.")
        for name in (
            "artifact_hash",
            "layout_hash",
            "pose_action_chunk_contract_hash",
        ):
            object.__setattr__(
                self, name, _nonempty(getattr(self, name), name=name)
            )
        frame_hashes = _unique_nonempty(
            self.pose_frame_contract_hashes,
            name="pose_frame_contract_hash",
        )
        if not frame_hashes:
            raise ValueError("pose_frame_contract_hashes must not be empty.")
        model_keys = _unique_nonempty(
            self.camera_model_input_keys,
            name="camera_model_input_key",
        )
        server_slots = _unique_nonempty(
            self.camera_server_slots,
            name="camera_server_slot",
        )
        camera_count = len(self.sample_spec.camera_source_keys)
        if len(model_keys) != camera_count or len(server_slots) != camera_count:
            raise ValueError(
                "Camera source, model-input, and server-slot bindings must have "
                "the same length."
            )
        sides = _unique_nonempty(
            self.required_active_sides,
            name="required_active_side",
        )
        if not sides or set(sides) - {"left", "right"}:
            raise ValueError("required_active_sides must use left/right.")
        object.__setattr__(self, "pose_frame_contract_hashes", frame_hashes)
        object.__setattr__(self, "camera_model_input_keys", model_keys)
        object.__setattr__(self, "camera_server_slots", server_slots)
        object.__setattr__(self, "required_active_sides", sides)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sample_spec": self.sample_spec.to_dict(),
            "sample_spec_hash": self.sample_spec.semantic_content_hash,
            "artifact_hash": self.artifact_hash,
            "layout_hash": self.layout_hash,
            "pose_action_chunk_contract_hash": (
                self.pose_action_chunk_contract_hash
            ),
            "pose_frame_contract_hashes": list(
                self.pose_frame_contract_hashes
            ),
            "camera_binding": [
                {
                    "source_key": source_key,
                    "model_input_key": model_key,
                    "server_camera_slot": server_slot,
                }
                for source_key, model_key, server_slot in zip(
                    self.sample_spec.camera_source_keys,
                    self.camera_model_input_keys,
                    self.camera_server_slots,
                    strict=True,
                )
            ],
            "required_active_sides": list(self.required_active_sides),
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=VLA_SCRATCH_DEXTELEOP_BINDING_HASH_DOMAIN_V1,
        )


def vla_scratch_binding_from_policy_artifact(
    artifact: PolicyArtifactContract,
    *,
    source_revision: str,
    state_history: int,
    required_camera_source_keys: Sequence[str] = (),
    required_active_sides: Sequence[str] = ("left", "right"),
) -> VlaScratchDexTeleopBinding:
    """Derive the VLA sample contract without guessing model dimensions.

    The current vla-scratch baseline has one dense floating state and one
    dense floating action output.  Artifacts with discrete channels or a
    transformed/masked state input therefore fail here instead of being
    silently flattened or dropped.
    """

    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be PolicyArtifactContract.")
    if artifact.dataset_schema != DEXTELEOP_DATASET_SCHEMA_V23:
        raise ValueError("vla-scratch canonical pose binding requires v2.3.")
    if artifact.pose_action_chunk is None or artifact.pose_training_data is None:
        raise ValueError(
            "vla-scratch canonical pose binding requires pose chunk and "
            "training-data contracts."
        )
    if artifact.camera_input is None or not artifact.camera_input.inputs:
        raise ValueError(
            "vla-scratch canonical pose binding requires ordered camera inputs."
        )
    if not artifact.camera_input.required:
        raise ValueError(
            "vla-scratch canonical pose binding requires its camera contract."
        )
    layout = artifact.policy_io_layout
    if layout.action_discrete_dim or layout.state_discrete_dim:
        raise ValueError(
            "The current vla-scratch processed sample cannot preserve "
            "DexTeleop discrete policy channels."
        )
    if artifact.state_input_mode != STATE_INPUT_MODE_RAW:
        raise ValueError(
            "The current canonical sample adapter only accepts raw artifact "
            "state input; transformed state needs an explicit shared transform."
        )

    required_cameras = _unique_nonempty(
        required_camera_source_keys,
        name="required_camera_source_key",
    )
    available_cameras = artifact.camera_input.source_keys
    missing_cameras = tuple(
        key for key in required_cameras if key not in available_cameras
    )
    if missing_cameras:
        raise ValueError(
            "Policy artifact is missing required camera sources: "
            f"{missing_cameras}."
        )
    sides = _unique_nonempty(required_active_sides, name="required_active_side")
    if not sides or set(sides) - {"left", "right"}:
        raise ValueError("required_active_sides must use left/right.")

    chunk = artifact.pose_action_chunk
    state_representation_id = (
        "dexteleop_v23/canonical_desk/observation_state/"
        f"{artifact.state_input_mode}/{layout.layout_hash}"
    )
    action_representation_id = (
        "dexteleop_v23/pose_action_chunk/"
        f"{chunk.action_pose_mode}/{chunk.contract_hash}/{layout.layout_hash}"
    )
    sample_spec = VlaScratchProcessedSampleSpec(
        source_revision=_nonempty(source_revision, name="source_revision"),
        camera_source_keys=available_cameras,
        state_history=state_history,
        state_dim=artifact.state_input_dim,
        action_horizon=chunk.action_horizon,
        action_dim=layout.action_dim,
        state_representation_id=state_representation_id,
        action_representation_id=action_representation_id,
    )
    return VlaScratchDexTeleopBinding(
        sample_spec=sample_spec,
        artifact_hash=canonical_content_hash(
            artifact.to_dict(),
            domain=VLA_SCRATCH_DEXTELEOP_ARTIFACT_HASH_DOMAIN_V1,
        ),
        layout_hash=layout.layout_hash,
        pose_action_chunk_contract_hash=chunk.contract_hash,
        pose_frame_contract_hashes=artifact.pose_frame_contract_hashes,
        camera_model_input_keys=tuple(
            item.model_input_key for item in artifact.camera_input.inputs
        ),
        camera_server_slots=artifact.camera_input.server_camera_slots,
        required_active_sides=sides,
    )


@dataclass(frozen=True)
class VlaScratchDexTeleopProcessedSample:
    """One fully validated training/serving sample and action target."""

    binding: VlaScratchDexTeleopBinding
    observation: VlaScratchProcessedObservation
    actions: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.binding, VlaScratchDexTeleopBinding):
            raise TypeError("binding must be VlaScratchDexTeleopBinding.")
        if not isinstance(self.observation, VlaScratchProcessedObservation):
            raise TypeError("observation must be VlaScratchProcessedObservation.")
        if (
            self.observation.spec.semantic_content_hash
            != self.binding.sample_spec.semantic_content_hash
        ):
            raise ValueError("Observation sample spec differs from its binding.")
        actions = validate_vla_scratch_action_chunk(
            self.actions,
            spec=self.binding.sample_spec,
        ).copy()
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)

    def to_processed_dict(self) -> dict[str, Any]:
        """Return exactly the baseline's existing processed training keys."""

        return {
            **self.observation.to_wire_payload(),
            VLA_SCRATCH_ACTIONS_KEY: self.actions.copy(),
        }


def build_vla_scratch_dexteleop_processed_sample(
    *,
    binding: VlaScratchDexTeleopBinding,
    pose_sample: PoseActionChunkSample,
    images_by_source: Mapping[str, Any],
    task: str,
    observation_state_history: Any | None = None,
) -> VlaScratchDexTeleopProcessedSample:
    """Project one canonical pose chunk into the unchanged baseline format."""

    if not isinstance(binding, VlaScratchDexTeleopBinding):
        raise TypeError("binding must be VlaScratchDexTeleopBinding.")
    if not isinstance(pose_sample, PoseActionChunkSample):
        raise TypeError("pose_sample must be PoseActionChunkSample.")
    if pose_sample.layout_hash != binding.layout_hash:
        raise ValueError("Pose sample layout hash differs from the artifact binding.")
    if (
        pose_sample.chunk_contract_hash
        != binding.pose_action_chunk_contract_hash
    ):
        raise ValueError(
            "Pose sample chunk contract hash differs from the artifact binding."
        )
    if (
        pose_sample.pose_frame_contract_hash
        not in binding.pose_frame_contract_hashes
    ):
        raise ValueError(
            "Pose sample frame contract is not authorized by the artifact."
        )
    missing_sides = tuple(
        side
        for side in binding.required_active_sides
        if side not in pose_sample.actual_tcp_pose_by_side
    )
    if missing_sides:
        raise ValueError(
            f"Pose sample is missing required active sides: {missing_sides}."
        )
    if np.any(pose_sample.action_is_pad):
        raise ValueError(
            "The current vla-scratch loss has no action padding-mask input; "
            "only complete canonical chunks are supported."
        )
    if (
        pose_sample.action_discrete.shape[1]
        or pose_sample.observation_state_discrete.size
    ):
        raise ValueError(
            "The current vla-scratch processed sample cannot carry discrete channels."
        )

    spec = binding.sample_spec
    if observation_state_history is None:
        if spec.state_history != 1:
            raise ValueError(
                "observation_state_history is required when state_history is not 1."
            )
        state = pose_sample.observation_state.reshape(1, -1)
    else:
        state = np.asarray(observation_state_history, dtype=np.float32)
        expected_shape = (spec.state_history, spec.state_dim)
        if state.shape != expected_shape:
            raise ValueError(
                f"observation_state_history must have shape {expected_shape}, "
                f"got {state.shape}."
            )
        if not np.array_equal(state[-1], pose_sample.observation_state):
            raise ValueError(
                "The final state-history row must be the chunk-start observation."
            )

    observation = build_vla_scratch_processed_observation(
        spec=spec,
        images_by_source=images_by_source,
        state=state,
        task=task,
    )
    actions = validate_vla_scratch_action_chunk(
        pose_sample.action,
        spec=spec,
    )
    return VlaScratchDexTeleopProcessedSample(
        binding=binding,
        observation=observation,
        actions=actions,
    )


__all__ = [
    "VLA_SCRATCH_DEXTELEOP_BINDING_SCHEMA_V1",
    "VlaScratchDexTeleopBinding",
    "VlaScratchDexTeleopProcessedSample",
    "build_vla_scratch_dexteleop_processed_sample",
    "vla_scratch_binding_from_policy_artifact",
]
