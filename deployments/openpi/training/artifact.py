"""Additive artifact derivation for trained OpenPI model-RTC."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    ModelContinuationDataContractV2,
    ModelContinuationTrainingContractV2,
    require_trained_model_rtc_v1_contract,
)
from dataset_contracts.policy_artifact import PolicyArtifactContract
from dataset_contracts.pose_chunk import PoseActionChunkContract
from dataset_contracts.pose_frames import PoseFrameContract
from dataset_contracts.pose_training import PoseTrainingDataContract
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_interface.model_continuation import (
    TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1,
    TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1,
)


def build_openpi_trained_model_rtc_artifact_template(
    source_artifact: PolicyArtifactContract,
    *,
    output_model_id: str,
    source_dataset_root: str | Path | None = None,
    pose_action_chunk: PoseActionChunkContract,
    pose_training_data: PoseTrainingDataContract,
    pose_frame_contract_hash: str,
) -> PolicyArtifactContract:
    """Derive one immutable training template without editing the source."""

    if not isinstance(source_artifact, PolicyArtifactContract):
        raise TypeError("source_artifact must be a PolicyArtifactContract.")
    source_payload = source_artifact.to_dict()
    source_continuation = source_artifact.model_continuation
    if not isinstance(source_continuation, ModelContinuationDataContractV2):
        raise ValueError(
            "Source artifact must have an explicit v2 continuation contract."
        )
    if (
        source_continuation.conditioning_origin
        != MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1
        or source_continuation.training is not None
    ):
        raise ValueError(
            "Source artifact must be inference-only; a trained artifact cannot "
            "be silently rebound."
        )
    if not isinstance(pose_action_chunk, PoseActionChunkContract):
        raise TypeError(
            "pose_action_chunk must be a PoseActionChunkContract."
        )
    if not isinstance(pose_training_data, PoseTrainingDataContract):
        raise TypeError(
            "pose_training_data must be a PoseTrainingDataContract."
        )
    if pose_action_chunk.action_pose_mode != "absolute":
        raise ValueError(
            "The first trained model-RTC profile requires absolute actions."
        )
    runtime = replace(
        source_continuation.runtime,
        prediction_horizon_steps=pose_action_chunk.action_horizon,
        frame_contract_hash=str(pose_frame_contract_hash),
    )
    training = ModelContinuationTrainingContractV2(
        prefix_source_kinds=(
            TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
        ),
        projection_implementation_id=(
            TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1
        ),
        projection_parameters_hash=(
            TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1
        ),
    )
    template = replace(
        source_artifact,
        model_id=str(output_model_id),
        checkpoint_step=None,
        source_datasets=(
            source_artifact.source_datasets
            if source_dataset_root is None
            else (
                str(Path(source_dataset_root).expanduser().resolve()),
            )
        ),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=pose_action_chunk,
        pose_training_data=pose_training_data,
        pose_frame_contract_hashes=(str(pose_frame_contract_hash),),
        model_continuation=ModelContinuationDataContractV2(
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
            runtime=runtime,
            training=training,
        ),
    )
    require_trained_model_rtc_v1_contract(template.model_continuation)
    if source_artifact.to_dict() != source_payload:
        raise AssertionError("Source artifact was mutated.")
    return template


def build_openpi_v22_absolute_model_rtc_artifact_template(
    source_artifact: PolicyArtifactContract,
    *,
    output_model_id: str,
    source_dataset_root: str | Path,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> PolicyArtifactContract:
    """First compatibility view: v2.2 raw identity, absolute, tail-drop."""

    continuation = source_artifact.model_continuation
    if not isinstance(continuation, ModelContinuationDataContractV2):
        raise ValueError(
            "Source artifact must have an explicit v2 continuation contract."
        )
    pose_frame = PoseFrameContract.legacy_v22_raw(
        active_sides=active_sides
    )
    return build_openpi_trained_model_rtc_artifact_template(
        source_artifact,
        output_model_id=output_model_id,
        source_dataset_root=source_dataset_root,
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=continuation.runtime.prediction_horizon_steps,
            action_pose_mode="absolute",
            last_action_pose_mode="absolute",
            tail_policy="drop",
        ),
        pose_training_data=PoseTrainingDataContract.v22_raw_identity(
            active_sides=active_sides
        ),
        pose_frame_contract_hash=pose_frame.contract_hash,
    )


__all__ = [
    "build_openpi_trained_model_rtc_artifact_template",
    "build_openpi_v22_absolute_model_rtc_artifact_template",
]
