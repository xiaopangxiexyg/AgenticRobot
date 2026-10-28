"""Application-neutral online collection progress runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Keep package-root compatibility without importing unrelated runtimes.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    'PROGRESS_PHASES': ('.reducer', 'PROGRESS_PHASES'),
    'AbortEpisodeProgressRequest': ('.requests', 'AbortEpisodeProgressRequest'),
    'AdvanceProgressStepRequest': ('.requests', 'AdvanceProgressStepRequest'),
    'CompleteEpisodeProgressRequest': ('.requests', 'CompleteEpisodeProgressRequest'),
    'ProgressAppendResult': ('.store', 'ProgressAppendResult'),
    'ProgressController': ('.controller', 'ProgressController'),
    'ProgressControllerResult': ('.controller', 'ProgressControllerResult'),
    'ProgressEffectRequest': ('.controller', 'ProgressEffectRequest'),
    'FAILURE_TRIGGER_CAPTURE_RELATIVE_ROOT': ('.failure_trigger_capture_store', 'FAILURE_TRIGGER_CAPTURE_RELATIVE_ROOT'),
    'FailureTriggerCaptureCommitReport': ('.failure_trigger_capture_store', 'FailureTriggerCaptureCommitReport'),
    'FailureTriggerCaptureStoreError': ('.failure_trigger_capture_store', 'FailureTriggerCaptureStoreError'),
    'ProgressFactStore': ('.store', 'ProgressFactStore'),
    'ProgressReductionError': ('.reducer', 'ProgressReductionError'),
    'ProgressRequestContext': ('.requests', 'ProgressRequestContext'),
    'ProgressSealReport': ('.store', 'ProgressSealReport'),
    'ProgressState': ('.reducer', 'ProgressState'),
    'ProgressStoreError': ('.store', 'ProgressStoreError'),
    'ProgressStoreIncompleteError': ('.store', 'ProgressStoreIncompleteError'),
    'ReportAttemptFailureRequest': ('.requests', 'ReportAttemptFailureRequest'),
    'RefinementAnnotationAppendReport': ('.annotation_store', 'RefinementAnnotationAppendReport'),
    'RefinementAnnotationSealReport': ('.annotation_store', 'RefinementAnnotationSealReport'),
    'RefinementAnnotationStore': ('.annotation_store', 'RefinementAnnotationStore'),
    'RefinementAnnotationStoreError': ('.annotation_store', 'RefinementAnnotationStoreError'),
    'RefinementAnnotationStoreIncompleteError': ('.annotation_store', 'RefinementAnnotationStoreIncompleteError'),
    'ResumeFailedAttemptRequest': ('.requests', 'ResumeFailedAttemptRequest'),
    'StartEpisodeProgressRequest': ('.requests', 'StartEpisodeProgressRequest'),
    'STAGE_ANNOTATION_DRAFT_RELATIVE_ROOT': ('.stage_annotation_store', 'STAGE_ANNOTATION_DRAFT_RELATIVE_ROOT'),
    'STAGE_ANNOTATION_EPISODE_LOCK_RELATIVE_ROOT': ('.stage_annotation_store', 'STAGE_ANNOTATION_EPISODE_LOCK_RELATIVE_ROOT'),
    'STAGE_ANNOTATION_RELATIVE_ROOT': ('.stage_annotation_store', 'STAGE_ANNOTATION_RELATIVE_ROOT'),
    'STAGE_ANNOTATION_WRITER_LOCK_RELATIVE_PATH': ('.stage_annotation_store', 'STAGE_ANNOTATION_WRITER_LOCK_RELATIVE_PATH'),
    'DatasetStageAnnotationWorkspace': ('.stage_annotation_store', 'DatasetStageAnnotationWorkspace'),
    'LoadedStageAnnotationSet': ('.stage_annotation_store', 'LoadedStageAnnotationSet'),
    'StageAnnotationCommitReport': ('.stage_annotation_store', 'StageAnnotationCommitReport'),
    'StageAnnotationStoreError': ('.stage_annotation_store', 'StageAnnotationStoreError'),
    'UndoLastProgressAdvanceRequest': ('.requests', 'UndoLastProgressAdvanceRequest'),
    'initial_progress_state': ('.reducer', 'initial_progress_state'),
    'reduce_progress_fact': ('.reducer', 'reduce_progress_fact'),
    'replay_progress_facts': ('.reducer', 'replay_progress_facts'),
    'commit_stage_annotation_set': ('.stage_annotation_store', 'commit_stage_annotation_set'),
    'DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT': ('.detail_stage_annotation_store', 'DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT'),
    'DetailStageAnnotationCommitReport': ('.detail_stage_annotation_store', 'DetailStageAnnotationCommitReport'),
    'DetailStageAnnotationStoreError': ('.detail_stage_annotation_store', 'DetailStageAnnotationStoreError'),
    'commit_detail_stage_annotation_set': ('.detail_stage_annotation_store', 'commit_detail_stage_annotation_set'),
    'failure_trigger_capture_path': ('.failure_trigger_capture_store', 'failure_trigger_capture_path'),
    'failure_trigger_capture_root': ('.failure_trigger_capture_store', 'failure_trigger_capture_root'),
    'load_failure_trigger_capture_records': ('.failure_trigger_capture_store', 'load_failure_trigger_capture_records'),
    'load_stage_annotation_set': ('.stage_annotation_store', 'load_stage_annotation_set'),
    'load_detail_stage_annotation_set': ('.detail_stage_annotation_store', 'load_detail_stage_annotation_set'),
    'record_failure_trigger_capture': ('.failure_trigger_capture_store', 'record_failure_trigger_capture'),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
