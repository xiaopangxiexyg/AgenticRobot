from __future__ import annotations

from .capture_modalities import (
    CAPTURE_MODALITY_METADATA_KEY,
    CaptureModalityBindingError,
    bind_capture_modality_contract,
    dataset_is_blank,
    read_capture_modality_contract,
)
from .component_action_projection import (
    COMPONENT_ACTION_PROJECTION_SPEC_HASH_DOMAIN_V1,
    COMPONENT_ACTION_PROJECTION_SPEC_SCHEMA_V1,
    ComponentActionProjectionSpec,
    ProjectedComponentAction,
    ProjectedComponentActionValue,
    component_action_catalog_definition,
    project_component_action_to_v25p,
    validate_component_action_projection_binding,
)
from .v25p_component_profile import (
    ActionStreamBinding,
    V25pV2CaptureProfileBundle,
    V25pV2StreamDeclaration,
    build_v25p_v2_capture_profile,
    validate_v25p_embodiment_stream_binding,
    validate_v25p_v2_capture_profile_bundle,
)
from .gripper_force import (
    GRIPPER_FORCE_PROFILE_METADATA_KEY,
    bind_gripper_force_contract,
    read_gripper_force_contract,
)

from .config import DatasetInterfaceConfig, MaskConfig, MissingViewConfig, VideoConfig
from .control_trace import (
    ActualDecisionRow,
    ControlTraceReadError,
    EpisodeControlTrace,
    ShadowEvaluationRow,
    load_episode_control_trace,
    materialize_actual_decision_view,
    materialize_shadow_evaluation_view,
)
from .metadata import (
    MODALITY_VALIDITY_SCHEMA_VERSION,
    DexTeleopDatasetIndex,
    EpisodeRecord,
    InvalidTrainingDataError,
    TrainingEligibility,
)
from .model_continuation import (
    MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1,
    TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1,
    TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1,
    ModelContinuationTrainingSample,
    ModelContinuationViewError,
    model_continuation_training_sample_from_episode_rows,
    model_continuation_training_sample_from_pose_chunk,
)
from .episode_curation import (
    EpisodeCurationReader,
    source_dataset_base_fingerprint,
    source_episode_manifest_hash,
)
from .pipeline import MaskedFramePipeline
from .pose_action import (
    PosePolicyViewError,
    PosePolicyWindow,
    pose_policy_window_from_dataset_metadata,
    pose_policy_window_from_rows,
)
from .pose_chunk import (
    PoseActionChunkLocation,
    PoseActionChunkSample,
    pose_action_chunk_from_dataset_metadata,
    pose_action_chunk_from_episode_rows,
    pose_action_chunk_start_indices,
)
from .policy_io import (
    PolicyIOBatch,
    PolicyIOFrame,
    policy_io_batch_from_rows,
    policy_io_frame_from_row,
)
from .replay import (
    derive_replay_source_action_contract,
    replay_action_record_from_materialized_row,
    replay_action_stage_from_contract,
    replay_source_contract_from_metadata,
)
from .sample import FrameRequest, MaskedFrameSample
from .state_augmentation import (
    NormalizedStateInputAdapter,
    StateAugmentationConfig,
    StateAugmentationResult,
    augment_normalized_state,
    augment_normalized_state_torch,
    TorchStateAugmentationResult,
)
from .streams import (
    NativeStreamEpisode,
    NativeStreamExternalPayloadReference,
    NativeStreamReadError,
    load_native_stream_episode,
)
from .task_semantics import (
    TaskSemanticsFrameSnapshot,
    project_task_semantics_to_frames,
)
from .v25p import (
    V25P_ASSESSMENT_INVALID,
    V25P_ASSESSMENT_V23_PROVISIONAL,
    V25P_ASSESSMENT_V25_COMPATIBILITY,
    V25P_ASSESSMENT_V25_NATIVE_MERGE,
    V25pLoadedEpisode,
    V25pMergeReadiness,
    V25pReadError,
    assess_v25p_merge_readiness,
    load_v25p_episode,
)
from .v25p_native import (
    V25P_NATIVE_READINESS_PROVISIONAL,
    V25P_NATIVE_READINESS_WITH_COMPATIBILITY,
    V25P_NATIVE_READINESS_WITH_V24,
    V25pLoadedNativeEpisode,
    V25pNativeReadError,
    V25pNativeReadiness,
    load_v25p_native_episode,
)
from .v25p_parquet_storage import load_v25p_parquet_sample
from .v25p_v23_adapter import (
    V25pV23ProjectedRow,
    V25pV23ProjectionResult,
    project_native_episode_to_v23_rows,
)
from .v25p_dual_write import (
    V25pDualWriteEquivalenceReport,
    V25pFieldEquivalence,
    compare_projected_v23_row,
)
from .v25p_projection_plan import (
    V25pV23ProjectionError,
    V25pV23ProjectionPlan,
    plan_v23_compatibility_projection,
    verify_complete_v23_projection_receipt,
)
from .validation import V23DatasetValidationReport, validate_v23_dataset

__all__ = [
    "COMPONENT_ACTION_PROJECTION_SPEC_HASH_DOMAIN_V1",
    "COMPONENT_ACTION_PROJECTION_SPEC_SCHEMA_V1",
    "ActionStreamBinding",
    "ComponentActionProjectionSpec",
    "V25pV2CaptureProfileBundle",
    "V25pV2StreamDeclaration",
    "ProjectedComponentAction",
    "ProjectedComponentActionValue",
    "component_action_catalog_definition",
    "TaskSemanticsFrameSnapshot",
    "V25P_ASSESSMENT_INVALID",
    "V25P_ASSESSMENT_V23_PROVISIONAL",
    "V25P_ASSESSMENT_V25_COMPATIBILITY",
    "V25P_ASSESSMENT_V25_NATIVE_MERGE",
    "V25pMergeReadiness",
    "V25pLoadedEpisode",
    "V25pReadError",
    "V25P_NATIVE_READINESS_PROVISIONAL",
    "V25P_NATIVE_READINESS_WITH_COMPATIBILITY",
    "V25P_NATIVE_READINESS_WITH_V24",
    "V25pLoadedNativeEpisode",
    "V25pNativeReadError",
    "V25pNativeReadiness",
    "V25pV23ProjectedRow",
    "V25pV23ProjectionError",
    "V25pV23ProjectionPlan",
    "V25pV23ProjectionResult",
    "V25pDualWriteEquivalenceReport",
    "V25pFieldEquivalence",
    "CAPTURE_MODALITY_METADATA_KEY",
    "GRIPPER_FORCE_PROFILE_METADATA_KEY",
    "CaptureModalityBindingError",
    "DatasetInterfaceConfig",
    "ActualDecisionRow",
    "ControlTraceReadError",
    "EpisodeControlTrace",
    "DexTeleopDatasetIndex",
    "EpisodeRecord",
    "EpisodeCurationReader",
    "FrameRequest",
    "MaskConfig",
    "MaskedFramePipeline",
    "MaskedFrameSample",
    "MissingViewConfig",
    "MODEL_CONTINUATION_TRAINING_VIEW_SCHEMA_V1",
    "ModelContinuationTrainingSample",
    "ModelContinuationViewError",
    "PosePolicyViewError",
    "PosePolicyWindow",
    "PoseActionChunkLocation",
    "PoseActionChunkSample",
    "MODALITY_VALIDITY_SCHEMA_VERSION",
    "InvalidTrainingDataError",
    "PolicyIOFrame",
    "TrainingEligibility",
    "ShadowEvaluationRow",
    "PolicyIOBatch",
    "NormalizedStateInputAdapter",
    "NativeStreamEpisode",
    "NativeStreamExternalPayloadReference",
    "NativeStreamReadError",
    "StateAugmentationConfig",
    "StateAugmentationResult",
    "TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1",
    "TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1",
    "TorchStateAugmentationResult",
    "VideoConfig",
    "V23DatasetValidationReport",
    "bind_capture_modality_contract",
    "bind_gripper_force_contract",
    "dataset_is_blank",
    "policy_io_batch_from_rows",
    "policy_io_frame_from_row",
    "replay_action_record_from_materialized_row",
    "derive_replay_source_action_contract",
    "replay_action_stage_from_contract",
    "replay_source_contract_from_metadata",
    "pose_policy_window_from_dataset_metadata",
    "pose_policy_window_from_rows",
    "pose_action_chunk_from_dataset_metadata",
    "pose_action_chunk_from_episode_rows",
    "pose_action_chunk_start_indices",
    "augment_normalized_state",
    "augment_normalized_state_torch",
    "validate_v23_dataset",
    "load_episode_control_trace",
    "load_native_stream_episode",
    "materialize_actual_decision_view",
    "materialize_shadow_evaluation_view",
    "model_continuation_training_sample_from_episode_rows",
    "model_continuation_training_sample_from_pose_chunk",
    "read_capture_modality_contract",
    "read_gripper_force_contract",
    "source_dataset_base_fingerprint",
    "source_episode_manifest_hash",
    "assess_v25p_merge_readiness",
    "load_v25p_episode",
    "load_v25p_native_episode",
    "load_v25p_parquet_sample",
    "plan_v23_compatibility_projection",
    "project_native_episode_to_v23_rows",
    "compare_projected_v23_row",
    "verify_complete_v23_projection_receipt",
    "project_task_semantics_to_frames",
    "project_component_action_to_v25p",
    "validate_component_action_projection_binding",
    "build_v25p_v2_capture_profile",
    "validate_v25p_embodiment_stream_binding",
    "validate_v25p_v2_capture_profile_bundle",
]
