from __future__ import annotations

import errno
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import quote

import pytest

from collection_runtime.progress import (
    DatasetStageAnnotationWorkspace,
    StageAnnotationStoreError,
    load_stage_annotation_set,
)
from collection_runtime.progress.failure_trigger_store import (
    load_failure_trigger_annotation_set,
)
from collection_runtime.progress.recoverable_failure_store import (
    load_recoverable_failure_annotation_set,
)
from collection_runtime.sample_acceptance import AcceptedSampleReceipt
from control_station.refinement_annotation import RefinementAnnotationGui
from control_station.refinement_annotation.application import (
    RefinementAnnotationApplication,
    RefinementAnnotationApplicationError,
)
from control_station.refinement_annotation.cli import build_parser
from dataset_interface.progress_annotation_read_cache import (
    ProgressAnnotationReadCache,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_profile import (
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stages import (
    ProgressStageDefinitionV1,
    TaskStageProfileV1,
)
from dataset_interface.progress_annotation_jobs import (
    list_progress_annotation_work_items,
    read_progress_annotation_work_item_batch,
)
from dataset_interface.progress_recoverable_failure_artifacts import (
    load_optional_recoverable_failure_annotation_set,
    load_recoverable_failure_annotation_set as load_analysis_recoverable_set,
)
from teleop_runtime.real_teleop.progress_adapter import (
    RealTeleopProgressEpisode,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _read_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=3.0) as response:
        return json.loads(response.read())


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urlopen(request, timeout=3.0) as response:
        return json.loads(response.read())


def test_gui_can_choose_dataset_and_annotator_after_start(
    tmp_path: Path,
) -> None:
    gui = RefinementAnnotationGui(open_browser=False)
    try:
        url = gui.start()
        workspace = _read_json(f"{url}api/workspace")["workspace"]
        assert workspace["configured"] is False
        with pytest.raises(HTTPError) as missing_workspace:
            _read_json(f"{url}api/episodes")
        assert missing_workspace.value.code == 400

        opened = _post_json(
            f"{url}api/workspace",
            {
                "dataset_root": str(tmp_path.resolve()),
                "annotator_id": "fixture",
                "second_review": True,
            },
        )["workspace"]
        assert opened["configured"] is True
        assert opened["dataset_root"] == str(tmp_path.resolve())
        assert opened["annotator_id"] == "fixture"
        assert opened["second_review"] is True
        assert opened["annotation_mode"] == "SECOND_REVIEW"
        assert _read_json(f"{url}api/episodes")["episodes"] == []

        with pytest.raises(HTTPError) as replacement:
            _post_json(
                f"{url}api/workspace",
                {
                    "dataset_root": str(tmp_path.resolve()),
                    "annotator_id": "other",
                },
            )
        assert replacement.value.code == 400
    finally:
        gui.close()


def test_gui_dataset_browser_and_workspace_switch(tmp_path: Path) -> None:
    first_root = tmp_path / "dataset-first"
    second_root = tmp_path / "dataset-second"
    first_root.mkdir()
    second_root.mkdir()
    gui = RefinementAnnotationGui(open_browser=False)
    try:
        url = gui.start()
        browser = _read_json(
            f"{url}api/dataset-browser?path={quote(str(tmp_path.resolve()))}",
        )
        assert browser["current"] == str(tmp_path.resolve())
        assert browser["parent"] == str(tmp_path.resolve().parent)
        assert {
            value["name"] for value in browser["directories"]
        } == {"dataset-first", "dataset-second"}

        opened = _post_json(
            f"{url}api/workspace",
            {
                "dataset_root": str(first_root),
                "annotator_id": "fixture",
            },
        )["workspace"]
        assert opened["dataset_root"] == str(first_root.resolve())

        switched = _post_json(
            f"{url}api/switch-workspace",
            {
                "dataset_root": str(second_root),
                "annotator_id": "fixture",
            },
        )["workspace"]
        assert switched["dataset_root"] == str(second_root.resolve())
        assert _read_json(f"{url}api/episodes")["dataset_root"] == str(
            second_root.resolve()
        )
    finally:
        gui.close()


def test_empty_dataset_gui_is_loopback_only_and_releases_lock(
    tmp_path: Path,
) -> None:
    gui = RefinementAnnotationGui(
        tmp_path,
        annotator_id="fixture",
        open_browser=False,
    )
    try:
        url = gui.start()
        assert url.startswith("http://127.0.0.1:")
        assert _read_json(f"{url}api/episodes")["episodes"] == []
        with urlopen(url, timeout=3.0) as response:
            body = response.read()
        assert b"episode-list" in body
        assert "可标注 Episode".encode() in body
        assert "这不表示二次阶段标注已经完成".encode() in body
        assert "每 4 秒自动发现新封存 Episode".encode() in body
        assert "完整时间范围会自动并入".encode() in body
        assert "撤销删除".encode() in body
        assert "强制重新编辑所选版本".encode() in body
        assert "收起 Episode 列表".encode() in body
        assert "上一条".encode() in body
        assert b"expand-episode-panel" in body
        assert b"change-dataset-root" in body
        assert b"dataset-browser-modal" in body
        assert "浏览目录".encode() in body
        assert "二次复核模式".encode() in body
        assert "需要二次复核".encode() in body
        with urlopen(f"{url}app.js", timeout=3.0) as response:
            app_source = response.read().decode("utf-8")
        with urlopen(f"{url}styles.css", timeout=3.0) as response:
            style_source = response.read().decode("utf-8")
        assert "occurrenceDeletionMergePlan" in app_source
        assert "STEP_COLOR_FAMILIES" in app_source
        assert "function stepForStage" in app_source
        assert "const tone = clamp(index / denominator" in app_source
        assert 'block.style.setProperty("--coarse-bg"' in app_source
        assert "mergePlan.target.start_at" in app_source
        assert "mergePlan.target.end_at" in app_source
        assert "function undoLastDeletion" in app_source
        assert "function reopenSealedAnnotation" in app_source
        assert "function setEpisodePanelCollapsed" in app_source
        assert "function navigateEpisode" in app_source
        assert "function openDatasetBrowser" in app_source
        assert "function switchWorkspace" in app_source
        assert "function hasUnsavedDraftChanges" in app_source
        assert "function ensureAnnotationsSealedBeforeTransition" in app_source
        assert "function publishCurrentAnnotations" in app_source
        assert "function requestJsonWithProgress" in app_source
        assert "function renderCatalogScanProgress" in app_source
        assert 'id="launcher-default-camera"' in body.decode("utf-8")
        assert "function preferredCameraForEpisode" in app_source
        assert "DEFAULT_CAMERA_STORAGE_KEY" in app_source
        assert "function fetchDatasetBrowser" in app_source
        assert "DATASET_BROWSER_RENDER_BATCH_SIZE" in app_source
        assert b'id="operation-progress"' in body
        assert ".operation-progress-fill.indeterminate" in style_source
        assert "payload.episode.draft_requires_seal ? null" in app_source
        assert "report.draft_cache_warning" in app_source
        assert "function renderFailedAttemptEditors" in app_source
        assert "/api/reflow-failed-stage/" in app_source
        assert "function renderCaptureEndStepEditor" in app_source
        assert "function annotationAttempts" in app_source
        assert "详细标注完成" in app_source
        assert "detail_annotation_complete" in app_source
        assert ".badge.detail-complete" in style_source
        assert "/api/reflow-capture-end-step/" in app_source
        assert "function timelineCoverageGaps" in app_source
        assert "function occurrenceReferenceIssues" in app_source
        assert "function reconcileAbnormalDraft" in app_source
        render_timeline_source = app_source.split(
            "function renderTimeline() {",
            1,
        )[1].split("function renderStageDescriptions", 1)[0]
        assert "normalizeOrderedOccurrences();" not in render_timeline_source
        assert "timeline-gap ${gap.kind}" in render_timeline_source
        assert ".abnormal-block" in style_source
        assert "height: 100%" in style_source
        assert "draft-warning-modal" in app_source
        assert "校验封存并切换" in body.decode("utf-8")
        assert b'id="save-draft"' not in body
        assert b'id="failed-attempt-editors"' in body
        assert b'id="capture-end-step-editor"' in body
        assert b'id="recoverable-failure-track"' in body
        assert b'id="detail-annotation-panel"' in body
        assert b'id="detail-stage-track"' in body
        assert b'id="publish-detail-annotation"' in body
        assert b'id="mark-recoverable-start"' in body
        assert b'id="cancel-recoverable-capture"' in body
        assert "中间失误后恢复".encode() in body
        assert "function renderRecoverableFailureTools" in app_source
        assert "function renderDetailAnnotation" in app_source
        assert "PROFILE_PREVIEW" in app_source or "profileStages" in app_source
        assert "detail-stage-preview-empty" in app_source
        assert "preserveViewport" in app_source
        assert "scrollbar-gutter: stable" in style_source
        assert "function confirmDetailAnnotation" in app_source
        assert "/api/commit-detail/" in app_source
        assert "当前有效分析仍使用粗粒度 fallback" in app_source
        assert "detail_overlay" in app_source
        assert "function currentTimelineSamplePoint" in app_source
        assert "function successOccurrenceOwningSample" in app_source
        assert "function recoverableFailureConflict" in app_source
        assert "function recoverableFailureAmbiguityIds" in app_source
        assert "function beginRecoverableFailureReannotation" in app_source
        assert "原区间会保留到新起止点通过校验" in app_source
        assert "recoverableFailureEditingId" in app_source
        assert ".recoverable-editing" in style_source
        assert ".recoverable-ambiguous" in style_source
        assert "/api/recoverable-failure/" in app_source
        assert "function browseDatasetPath" in app_source
        assert "function scheduleNextEpisodeWarmup" in app_source
        assert "function showVideoLoadPrompt" in app_source
        assert "function timelinePointCaptureReady" in app_source
        assert "function requireTimelinePointCaptureReady" in app_source
        assert "function occurrenceAtTimelineTime" in app_source
        assert "timestamp < value.end_at" in app_source
        assert "sealedPending && Boolean(request)" in app_source
        assert "function clearVideoLoadListeners" in app_source
        assert 'video.removeEventListener("error", onError)' in app_source
        assert (
            "if (episodesChanged) renderEpisodeList({preserveViewport: true})"
            in app_source
        )
        episodes_changed = app_source.split(
            "if (episodesChanged) {", 1
        )[1].split("\n    }", 1)[0]
        assert "state.episodeRenderLimit," in episodes_changed
        assert "inFlightEpisodeSummaryIds.has(episodeUuid)" in app_source
        assert "workspaceGeneration !== state.workspaceGeneration" in app_source
        assert "function cancelPendingSeek" in app_source
        assert "scheduleSeek(timestamp)" in app_source
        assert "function synchronizeFailureTriggerWindows" in app_source
        assert "function applyCoarseBoundaryCorrection" in app_source
        assert "if (fromTail) fromTail.end_at = corrected" in app_source
        assert "if (toHead) toHead.start_at = corrected" in app_source
        assert "function restoreOriginalCoarseBoundaries" in app_source
        assert 'kind: "coarse-boundary"' in app_source
        assert "function renderReviewControls" in app_source
        assert "function editableTailOutcome" in app_source
        assert "Math.abs(Number(occurrence.end_at) - attemptEnd)" in app_source
        tail_function = app_source.split(
            "function editableTailOutcome()", 1
        )[1].split("\nfunction ", 1)[0]
        assert "state.detail.end_timestamp" not in tail_function
        assert '"failure-trigger-track"' in app_source
        assert "POST_CAPTURE_REFINED" in app_source
        assert "nearestSampleIndex(proposed)" in app_source
        assert "EPISODE_PANEL_STORAGE_KEY" in app_source
        assert "var(--coarse-bg" in style_source
        assert ".dataset-browser-card" in style_source
        assert ".draft-warning-card" in style_source
        assert ".workspace.episode-panel-is-collapsed" in style_source
        concurrent = RefinementAnnotationGui(
            tmp_path,
            annotator_id="other",
            open_browser=False,
        )
        concurrent.close()
    finally:
        gui.close()

    replacement = RefinementAnnotationGui(
        tmp_path,
        annotator_id="replacement",
        open_browser=False,
    )
    replacement.close()


def test_cli_and_bind_validation_are_fail_closed(tmp_path: Path) -> None:
    launcher_args = build_parser().parse_args([])
    assert launcher_args.dataset_root is None
    assert launcher_args.annotator_id == ""
    assert launcher_args.second_review is False
    args = build_parser().parse_args(
        [
            str(tmp_path),
            "--annotator-id",
            "fixture",
            "--dry-run",
            "--second-review",
        ]
    )
    assert args.dry_run is True
    assert args.second_review is True
    with pytest.raises(ValueError, match="loopback"):
        RefinementAnnotationGui(
            tmp_path,
            annotator_id="fixture",
            host="0.0.0.0",
            open_browser=False,
        )


def _finalized_failure_retry_dataset(root: Path) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    progress = TaskProgressProfileV1(
        profile_id="annotation-fixture-progress",
        profile_version="1",
        task_id="annotation-fixture",
        steps=(
            ProgressStepDefinitionV1(
                progress_step_id="a",
                semantic_type="TASK_A",
                display_name="子任务 A",
                ordinal=1,
            ),
        ),
        initial_step_id="a",
        terminal_step_ids=("a",),
    )
    stages = TaskStageProfileV1(
        profile_id="annotation-fixture-stages",
        profile_version="1",
        task_progress_profile_hash=progress.semantic_content_hash,
        stages=(
            ProgressStageDefinitionV1("a", "a1", "A1", "阶段一", 1),
            ProgressStageDefinitionV1("a", "a2", "A2", "阶段二", 2),
            ProgressStageDefinitionV1("a", "a3", "A3", "阶段三", 3),
        ),
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid="annotation-fixture-episode",
        legacy_episode_index=0,
        lifecycle_owner_id="fixture",
        prepared_at=0.0,
        prepared_clock_domain_id="local_wall_time",
    )
    episode = RealTeleopProgressEpisode(
        dataset_root=root,
        identity=identity,
        profile=progress,
        stage_profile=stages,
    )
    episode.bind(started_at=0.0)
    effect = episode.report_failure(
        intent_id="failure-1",
        effective_at=3.0,
        available_at=3.0,
    )
    episode.record_applied_hold_pause(
        effect_request=effect,
        requested_at=3.0,
        application_hold_effective_at=3.0,
        formal_capture_pause_effective_at=3.0,
        available_at=3.1,
    )
    episode.record_applied_resume(
        resume_operation_id="resume-1",
        requested_at=3.5,
        application_resume_effective_at=4.0,
        formal_capture_resume_effective_at=4.0,
        available_at=4.1,
    )
    episode.retry_on_accepted_sample(
        AcceptedSampleReceipt(
            episode_index=0,
            sample_index=4,
            episode_frame_index=4,
            sample_timestamp=4.2,
        ),
        available_at=4.3,
    )
    episode.terminate(reason="manual_key_stop", effective_at=8.0)
    episode.seal(sealed_at=8.1)

    parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"timestamp": [float(value) for value in range(9)]}),
        parquet_path,
    )
    video_path = root / "videos" / "front" / "episode_000000.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fixture-video")
    catalog_path = root / "meta" / "episodes.jsonl"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 9,
                "start_timestamp": 0.0,
                "end_timestamp": 8.0,
                "parquet_path": str(parquet_path.relative_to(root)),
                "video_paths": {
                    "front": str(video_path.relative_to(root)),
                },
                "metadata": {
                    "progress_episode": episode.metadata_reference,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return identity.episode_uuid


def _finalized_manual_stop_dataset(
    root: Path,
    *,
    advance_to_second_step: bool = True,
    termination_reason: str = "manual_key_stop",
) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    progress = TaskProgressProfileV1(
        profile_id="manual-stop-progress",
        profile_version="1",
        task_id="manual-stop",
        steps=(
            ProgressStepDefinitionV1(
                "a",
                "TASK_A",
                "子任务 A",
                1,
                allowed_next_step_ids=("b",),
            ),
            ProgressStepDefinitionV1(
                "b",
                "TASK_B",
                "子任务 B",
                2,
                allowed_next_step_ids=("c",),
            ),
            ProgressStepDefinitionV1("c", "TASK_C", "子任务 C", 3),
        ),
        initial_step_id="a",
        terminal_step_ids=("c",),
    )
    stages = TaskStageProfileV1(
        profile_id="manual-stop-stages",
        profile_version="1",
        task_progress_profile_hash=progress.semantic_content_hash,
        stages=(
            ProgressStageDefinitionV1("a", "a1", "A1", "阶段 A1", 1),
            ProgressStageDefinitionV1("a", "a2", "A2", "阶段 A2", 2),
            ProgressStageDefinitionV1("b", "b1", "B1", "阶段 B1", 1),
            ProgressStageDefinitionV1("c", "c1", "C1", "阶段 C1", 1),
        ),
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid="manual-stop-episode",
        legacy_episode_index=0,
        lifecycle_owner_id="fixture",
        prepared_at=0.0,
        prepared_clock_domain_id="local_wall_time",
    )
    episode = RealTeleopProgressEpisode(
        dataset_root=root,
        identity=identity,
        profile=progress,
        stage_profile=stages,
    )
    episode.bind(started_at=0.0)
    if advance_to_second_step:
        episode.advance(intent_id="advance-to-b", effective_at=4.0)
    episode.terminate(reason=termination_reason, effective_at=6.0)
    episode.seal(sealed_at=6.1)

    parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"timestamp": [float(value) for value in range(7)]}),
        parquet_path,
    )
    video_path = root / "videos" / "front" / "episode_000000.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fixture-video")
    catalog_path = root / "meta" / "episodes.jsonl"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 7,
                "start_timestamp": 0.0,
                "end_timestamp": 6.0,
                "parquet_path": str(parquet_path.relative_to(root)),
                "video_paths": {
                    "front": str(video_path.relative_to(root)),
                },
                "metadata": {"progress_episode": episode.metadata_reference},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return identity.episode_uuid


def test_manual_stop_defaults_to_recorded_step_and_can_reflow_to_early_step(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_manual_stop_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        editor = detail["capture_end_step_editor"]
        assert editor["termination_cause"] == "MANUAL_STOP"
        assert editor["source_progress_step_id"] == "b"
        assert [value["progress_step_id"] for value in editor["options"]] == [
            "a",
            "b",
            "c",
        ]
        assert editor["default_end_progress_step_id"] == "b"
        assert detail["draft"]["capture_end_step"][
            "end_progress_step_id"
        ] == "b"
        assert [
            value["progress_step_id"]
            for value in detail["draft"]["occurrences"]
        ] == ["a", "a", "b"]

        original_draft = detail["draft"]
        early = application.reflow_capture_end_step(
            episode_uuid,
            original_draft,
            end_progress_step_id="a",
        )["draft"]
        assert early["capture_end_step"]["end_progress_step_id"] == "a"
        assert [
            value["stage_id"] for value in early["occurrences"]
        ] == ["a1", "a2"]
        report = application.commit(episode_uuid, early)
        assert report["coverage_status"] == "COMPLETE"
        sealed = load_stage_annotation_set(
            tmp_path
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
            / report["annotation_set_id"]
        )
        assert sealed.bundle.capture_end_step is not None
        assert sealed.bundle.capture_end_step.end_progress_step_id == "a"

        legacy = {
            **original_draft,
            "capture_end_step": None,
            "occurrences": original_draft["occurrences"][:3],
        }
        migrated = application.reflow_capture_end_step(
            episode_uuid,
            legacy,
            end_progress_step_id="b",
        )["draft"]
        assert migrated["capture_end_step"]["end_progress_step_id"] == "b"
        assert [
            value["progress_step_id"]
            for value in migrated["occurrences"]
        ] == ["a", "a", "b"]
    finally:
        application.close()


def test_manual_stop_in_first_step_defaults_to_first_and_can_extend(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_manual_stop_dataset(
        tmp_path,
        advance_to_second_step=False,
    )
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        assert detail["capture_end_step_editor"][
            "source_progress_step_id"
        ] == "a"
        assert detail["capture_end_step_editor"][
            "default_end_progress_step_id"
        ] == "a"
        assert detail["draft"]["capture_end_step"][
            "end_progress_step_id"
        ] == "a"
        assert [
            value["progress_step_id"]
            for value in detail["draft"]["occurrences"]
        ] == ["a", "a"]

        full_profile = application.reflow_capture_end_step(
            episode_uuid,
            detail["draft"],
            end_progress_step_id="c",
        )["draft"]
        assert [
            value["progress_step_id"]
            for value in full_profile["occurrences"]
        ] == ["a", "a", "b", "c"]
        assert application.commit(episode_uuid, full_profile)[
            "coverage_status"
        ] == "COMPLETE"
        reopened = application.episode_detail(episode_uuid)["draft"]
        assert reopened["capture_end_step"]["end_progress_step_id"] == "c"
        assert [
            value["progress_step_id"] for value in reopened["occurrences"]
        ] == ["a", "a", "b", "c"]
    finally:
        application.close()


def test_ambiguous_abort_requires_explicit_capture_end_selection(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_manual_stop_dataset(
        tmp_path,
        advance_to_second_step=False,
        termination_reason="unexpected_writer_exit",
    )
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        assert detail["capture_end_step_editor"]["termination_cause"] == "ABORTED"
        assert detail["draft"]["capture_end_step"] is None
        assert [
            value["stage_id"] for value in detail["draft"]["occurrences"]
        ] == ["a1"]

        explicitly_selected = application.reflow_capture_end_step(
            episode_uuid,
            detail["draft"],
            end_progress_step_id="a",
        )["draft"]
        assert explicitly_selected["capture_end_step"][
            "end_progress_step_id"
        ] == "a"
        assert [
            value["stage_id"]
            for value in explicitly_selected["occurrences"]
        ] == ["a1", "a2"]
    finally:
        application.close()


def test_http_capture_end_step_reflow_returns_valid_draft(tmp_path: Path) -> None:
    episode_uuid = _finalized_manual_stop_dataset(tmp_path)
    gui = RefinementAnnotationGui(
        tmp_path,
        annotator_id="fixture",
        open_browser=False,
    )
    try:
        url = gui.start()
        detail = _read_json(
            f"{url}api/episode/{quote(episode_uuid)}"
        )["episode"]
        result = _post_json(
            f"{url}api/reflow-capture-end-step/{quote(episode_uuid)}",
            {
                "draft": detail["draft"],
                "end_progress_step_id": "a",
            },
        )
        assert result["ok"] is True
        assert result["draft"]["capture_end_step"][
            "end_progress_step_id"
        ] == "a"
        assert [
            value["stage_id"] for value in result["draft"]["occurrences"]
        ] == ["a1", "a2"]
    finally:
        gui.close()


def test_two_applications_share_dataset_but_exclude_same_episode(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    first = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="first",
    )
    second = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="second",
    )
    try:
        assert first.episode_detail(episode_uuid)["episode_uuid"] == episode_uuid
        with pytest.raises(
            StageAnnotationStoreError,
            match="already being annotated.*annotator_id=first",
        ):
            second.episode_detail(episode_uuid)

        first.close()
        assert second.episode_detail(episode_uuid)["episode_uuid"] == episode_uuid
    finally:
        first.close()
        second.close()


def test_failed_episode_open_keeps_previous_lock_and_releases_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    target_uuid = "annotation-fixture-target"
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="first",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        with application._lock:
            application._items[target_uuid] = application._items[episode_uuid]

        def failed_target_detail(candidate: str) -> dict[str, object]:
            assert candidate == target_uuid
            raise RefinementAnnotationApplicationError(
                "simulated remote target-detail failure"
            )

        monkeypatch.setattr(
            application,
            "_episode_detail_owned",
            failed_target_detail,
        )
        with pytest.raises(
            RefinementAnnotationApplicationError,
            match="simulated remote target-detail failure",
        ):
            application.episode_detail(target_uuid)

        assert application._active_episode_uuid == episode_uuid
        application.workspace.require_episode(episode_uuid=episode_uuid)
        saved = application.save_draft(episode_uuid, detail["draft"])
        assert saved["episode_uuid"] == episode_uuid

        with DatasetStageAnnotationWorkspace(
            tmp_path,
            annotator_id="second",
        ) as concurrent:
            concurrent.acquire_episode(episode_uuid=target_uuid)
            with pytest.raises(
                StageAnnotationStoreError,
                match="already being annotated.*annotator_id=first",
            ):
                concurrent.acquire_episode(episode_uuid=episode_uuid)

        monkeypatch.setattr(
            application,
            "_episode_detail_owned",
            lambda candidate: {"episode_uuid": candidate},
        )
        assert application.episode_detail(target_uuid) == {
            "episode_uuid": target_uuid
        }
        assert application._active_episode_uuid == target_uuid
        with DatasetStageAnnotationWorkspace(
            tmp_path,
            annotator_id="second",
        ) as concurrent:
            concurrent.acquire_episode(episode_uuid=episode_uuid)
            with pytest.raises(
                StageAnnotationStoreError,
                match="already being annotated.*annotator_id=first",
            ):
                concurrent.acquire_episode(episode_uuid=target_uuid)
    finally:
        application.close()


def test_gui_workspace_open_defers_slow_initial_catalog_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_station.refinement_annotation.application as application_module

    started = threading.Event()
    release = threading.Event()
    original = application_module.read_progress_annotation_work_item_batch

    def blocked_discovery(root: object, **kwargs: object) -> object:
        started.set()
        assert release.wait(timeout=3.0)
        return original(root, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        application_module,
        "read_progress_annotation_work_item_batch",
        blocked_discovery,
    )
    began = time.monotonic()
    gui = RefinementAnnotationGui(
        tmp_path,
        annotator_id="fixture",
        open_browser=False,
    )
    elapsed = time.monotonic() - began
    try:
        assert elapsed < 0.5
        assert started.wait(timeout=1.0)
        assert gui.server.workspace_payload()["configured"] is True
    finally:
        release.set()
        gui.close()


def test_lightweight_episode_list_does_not_load_annotation_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_station.refinement_annotation.application as application_module

    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    writer = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="writer",
    )
    try:
        detail = writer.episode_detail(episode_uuid)
        writer.commit(episode_uuid, detail["draft"])
    finally:
        writer.close()

    def unexpected_load(_path: object) -> object:
        raise AssertionError("lightweight list loaded an annotation bundle")

    monkeypatch.setattr(
        application_module,
        "load_stage_annotation_set",
        unexpected_load,
    )
    reader = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="reader",
    )
    try:
        payload = reader.list_episodes(
            include_annotation_summaries=False,
        )
        # Startup intentionally leaves the count as an unloaded hint; the GUI
        # requests exact bundles only for rows that become visible.
        assert payload["episodes"][0]["annotation_set_count"] == 0
        assert payload["episodes"][0]["annotation_summary_loaded"] is False
        assert payload["episodes"][0]["whole_episode_abnormal"] is None
        assert payload["episodes"][0]["review_state"] is None
    finally:
        reader.close()


def test_failed_stage_reflow_adds_success_prefix_and_realigns_retry(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        failed_attempt = detail["attempts"][0]
        retry_attempt = detail["attempts"][1]

        result = application.reflow_failed_attempt_stage(
            episode_uuid,
            detail["draft"],
            attempt_id=failed_attempt["attempt_id"],
            failed_stage_id="a3",
        )

        assert result["changed_attempt_ids"] == (
            failed_attempt["attempt_id"],
            retry_attempt["attempt_id"],
        )
        by_attempt = {
            attempt_id: [
                value
                for value in result["draft"]["occurrences"]
                if value["coarse_attempt_id"] == attempt_id
            ]
            for attempt_id in result["changed_attempt_ids"]
        }
        assert [
            (value["stage_id"], value["outcome"])
            for value in by_attempt[failed_attempt["attempt_id"]]
        ] == [
            ("a1", "SUCCESS"),
            ("a2", "SUCCESS"),
            ("a3", "FAILURE"),
        ]
        assert by_attempt[failed_attempt["attempt_id"]][-1][
            "source_failure_fact_id"
        ] == failed_attempt["failure_fact_id"]
        assert [
            (value["stage_id"], value["outcome"])
            for value in by_attempt[retry_attempt["attempt_id"]]
        ] == [("a3", "SUCCESS")]

        report = application.commit(episode_uuid, result["draft"])
        assert report["coverage_status"] == "COMPLETE"
        assert report["exclusion_count"] == 1
    finally:
        application.close()


def test_detail_overlay_is_optional_until_explicitly_sealed(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    detail_profile_root = tmp_path / "detail-profiles"
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
        detail_profile_root=detail_profile_root,
    )
    try:
        item = application._item(episode_uuid)
        progress_profile = item.progress.profile
        detail_stage_profile = TaskStageProfileV1(
            profile_id="annotation-fixture-detail-stages",
            profile_version="1",
            task_progress_profile_hash=progress_profile.semantic_content_hash,
            stages=(
                ProgressStageDefinitionV1(
                    "a",
                    "a.start",
                    "开始",
                    "开始阶段",
                    1,
                ),
                ProgressStageDefinitionV1(
                    "a",
                    "a.body",
                    "主体",
                    "主体阶段",
                    2,
                ),
                ProgressStageDefinitionV1(
                    "a",
                    "a.finish",
                    "收尾",
                    "收尾阶段",
                    3,
                ),
            ),
        )
        detail_profile_root.mkdir()
        (detail_profile_root / "fixture.progress_detailed.json").write_text(
            json.dumps(progress_profile.to_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        (detail_profile_root / "fixture.stages_detailed.json").write_text(
            json.dumps(detail_stage_profile.to_payload(), ensure_ascii=False),
            encoding="utf-8",
        )

        initial = application.episode_detail(episode_uuid)
        assert initial["detail_annotation"]["available"] is False
        assert initial["detail_annotation"]["draft_state"] == (
            "PROFILE_PREVIEW"
        )
        assert [
            value["stage_id"]
            for value in initial["detail_annotation"]["profile"]["stages"]
        ] == ["a.start", "a.body", "a.finish"]

        coarse_report = application.commit(episode_uuid, initial["draft"])
        assert coarse_report["detail_annotation"]["available"] is True
        assert (
            coarse_report["detail_annotation"]["resolution_level"]
            == "COARSE_FALLBACK"
        )
        coarse_summary = application.episode_annotation_summaries(
            (episode_uuid,)
        )
        assert coarse_summary["summaries"][0][
            "detail_annotation_complete"
        ] is False

        detail = application.episode_detail(episode_uuid)
        detail_draft = json.loads(json.dumps(detail["draft"]))
        meta = detail["detail_annotation"]
        failure_attempt = meta["failure_attempts"][0]
        failure_attempt_id = failure_attempt["attempt_id"]
        failure_fact_id = failure_attempt["failure_fact_id"]
        overlay = detail_draft["detail_overlay"]
        overlay["occurrences"] = [
            {
                **value,
                **(
                    {
                        "outcome": "FAILURE",
                        "source_failure_fact_id": failure_fact_id,
                    }
                    if (
                        value["coarse_attempt_id"] == failure_attempt_id
                        and value["stage_id"] == "a.body"
                    )
                    else {}
                ),
            }
            for value in overlay["occurrences"]
            if not (
                value["coarse_attempt_id"] == failure_attempt_id
                and value["stage_id"] == "a.finish"
            )
        ]
        overlay["failure_stage_by_attempt"][failure_attempt_id] = "a.body"
        overlay["confirmed"] = True

        detail_report = application.commit_detail(
            episode_uuid,
            detail_draft,
        )
        assert detail_report["detail_annotation"]["resolution_level"] == (
            "DETAIL_COMPLETE"
        )
        reopened = application.episode_detail(episode_uuid)
        assert reopened["detail_annotation"]["resolution_level"] == (
            "DETAIL_COMPLETE"
        )
        assert reopened["detail_annotation"]["draft_state"] == "SEALED"
        complete_summary = application.episode_annotation_summaries(
            (episode_uuid,)
        )
        assert complete_summary["summaries"][0][
            "detail_annotation_complete"
        ] is True
    finally:
        application.close()


def test_builtin_writing_detail_profile_exposes_three_write_stages(
    tmp_path: Path,
) -> None:
    from collection_runtime.progress.profiles.catalog import (
        load_task_progress_profile_file,
        load_task_stage_profile_file,
    )

    profile_root = (
        Path(__file__).resolve().parents[1]
        / "collection_runtime"
        / "progress"
        / "profiles"
        / "linear"
    )
    progress = load_task_progress_profile_file(
        profile_root
        / "dmtac_rm65_bimanual_position_writing_four_stage.progress.json"
    )
    coarse_stages = load_task_stage_profile_file(
        profile_root
        / "dmtac_rm65_bimanual_position_writing_four_stage.stages.json",
        progress_profile=progress,
    )
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
        defer_initial_refresh=True,
    )
    try:
        selection, reason = application._detail_profile_selection(
            SimpleNamespace(
                progress=SimpleNamespace(
                    profile=progress,
                    stage_profile=coarse_stages,
                )
            )
        )
        assert reason is None
        assert selection is not None
        assert [
            value.stage_id
            for value in selection.stage_profile.stages_for_step("right_write")
        ] == [
            "right_write.pen_down",
            "right_write.write",
            "right_write.pen_up",
        ]
    finally:
        application.close()


def test_detail_overlay_without_target_step_falls_back_after_three_coarse_seals(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_manual_stop_dataset(
        tmp_path,
        advance_to_second_step=False,
    )
    detail_profile_root = tmp_path / "detail-profiles"
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
        detail_profile_root=detail_profile_root,
    )
    try:
        detail = application.episode_detail(episode_uuid)
        for _ in range(3):
            application.commit(episode_uuid, detail["draft"])
            detail = application.episode_detail(episode_uuid)

        item = application._item(episode_uuid)
        progress_profile = item.progress.profile
        detail_stage_profile = TaskStageProfileV1(
            profile_id="annotation-fixture-detail-missing-target",
            profile_version="1",
            task_progress_profile_hash=progress_profile.semantic_content_hash,
            stages=(
                ProgressStageDefinitionV1("a", "a1", "A1", "阶段 A1", 1),
                ProgressStageDefinitionV1("a", "a2", "A2", "阶段 A2", 2),
                ProgressStageDefinitionV1("b", "b1", "B1", "阶段 B1", 1),
                ProgressStageDefinitionV1("b", "b2", "B2", "阶段 B2", 2),
                ProgressStageDefinitionV1("c", "c1", "C1", "阶段 C1", 1),
            ),
        )
        detail_profile_root.mkdir()
        (detail_profile_root / "fixture.progress_detailed.json").write_text(
            json.dumps(progress_profile.to_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        (detail_profile_root / "fixture.stages_detailed.json").write_text(
            json.dumps(detail_stage_profile.to_payload(), ensure_ascii=False),
            encoding="utf-8",
        )

        reopened = application.episode_detail(episode_uuid)
        assert reopened["annotation_set_count"] == 3
        assert reopened["detail_annotation"]["available"] is False
        assert reopened["detail_annotation"]["resolution_level"] == (
            "COARSE_FALLBACK"
        )
        assert "没有覆盖详细 profile 的目标子任务" in (
            reopened["detail_annotation"]["reason"]
        )
        assert reopened["draft"]["detail_overlay"] is None
    finally:
        application.close()


def test_draft_navigation_state_distinguishes_bootstrap_legacy_and_sealed(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        initial = application.episode_detail(episode_uuid)
        assert initial["draft_persistence_state"] == "BOOTSTRAP_UNEDITED"
        assert initial["draft_requires_seal"] is False

        # Old GUI versions could leave a mutable draft without any immutable
        # annotation set. It must remain protected even before a new edit is
        # made in the current browser session.
        application.save_draft(episode_uuid, initial["draft"])
        legacy = application.episode_detail(episode_uuid)
        assert legacy["draft_persistence_state"] == "UNSEALED_DRAFT"
        assert legacy["draft_requires_seal"] is True

        application.commit(episode_uuid, legacy["draft"])
        sealed = application.episode_detail(episode_uuid)
        assert sealed["draft_persistence_state"] == "SEALED_SNAPSHOT"
        assert sealed["draft_requires_seal"] is False
    finally:
        application.close()


def test_stage_publish_retries_one_idempotent_transient_eperm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_station.refinement_annotation.application as application_module

    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    original_commit = application_module.commit_stage_annotation_set
    calls = 0

    def transient_commit(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                raise PermissionError(errno.EPERM, "transient sshfs mkdir")
            except PermissionError as cause:
                raise StageAnnotationStoreError("transient publish") from cause
        return original_commit(**kwargs)

    monkeypatch.setattr(
        application_module,
        "commit_stage_annotation_set",
        transient_commit,
    )
    try:
        draft = application.episode_detail(episode_uuid)["draft"]
        report = application.commit(episode_uuid, draft)
        assert calls == 2
        assert report["annotation_set_id"]
    finally:
        application.close()


def test_committed_set_survives_draft_cache_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    original_save = application.workspace.save_draft
    observed_cache_write: dict[str, object] = {}

    def rejected_cache_write(**kwargs: object) -> Path:
        observed_cache_write.update(kwargs)
        raise StageAnnotationStoreError("sshfs cache write rejected")

    try:
        draft = application.episode_detail(episode_uuid)["draft"]
        monkeypatch.setattr(
            application.workspace,
            "save_draft",
            rejected_cache_write,
        )
        report = application.commit(episode_uuid, draft)
        assert "was committed" in report["draft_cache_warning"]
        assert observed_cache_write["durable"] is False
        annotation_root = (
            tmp_path
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
            / report["annotation_set_id"]
        )
        assert (annotation_root / "manifest.cjson").is_file()

        monkeypatch.setattr(application.workspace, "save_draft", original_save)
        reopened = application.episode_detail(episode_uuid)
        assert reopened["draft_persistence_state"] == "SEALED_SNAPSHOT"
        assert reopened["draft_requires_seal"] is False
    finally:
        application.close()


def test_http_commit_returns_json_for_stage_store_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    gui = RefinementAnnotationGui(
        tmp_path,
        annotator_id="fixture",
        open_browser=False,
    )
    try:
        url = gui.start()
        draft = _read_json(
            f"{url}api/episode/{quote(episode_uuid)}"
        )["episode"]["draft"]

        def rejected_commit(*_args: object, **_kwargs: object) -> object:
            raise StageAnnotationStoreError("dataset storage unavailable")

        monkeypatch.setattr(
            gui.server.require_application(),
            "commit",
            rejected_commit,
        )
        with pytest.raises(HTTPError) as failure:
            _post_json(
                f"{url}api/commit/{quote(episode_uuid)}",
                draft,
            )
        assert failure.value.code == 400
        payload = json.loads(failure.value.read())
        assert payload == {
            "error": "dataset storage unavailable",
            "ok": False,
        }
    finally:
        gui.close()


def test_http_episode_open_reports_operation_progress(tmp_path: Path) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    gui = RefinementAnnotationGui(
        tmp_path,
        annotator_id="fixture",
        open_browser=False,
    )
    try:
        url = gui.start()
        operation_id = "fixture-open-progress"
        request = Request(
            f"{url}api/episode/{quote(episode_uuid)}",
            headers={"X-Annotation-Operation": operation_id},
        )
        with urlopen(request, timeout=3.0) as response:
            opened = json.loads(response.read())
        assert opened["episode"]["episode_uuid"] == episode_uuid

        progress = _read_json(
            f"{url}api/operation-progress?id={quote(operation_id)}"
        )["progress"]
        assert progress["kind"] == "OPEN_EPISODE"
        assert progress["percent"] == 100
        assert progress["active"] is False
        assert progress["error"] is None
    finally:
        gui.close()


def test_failure_retry_draft_commits_one_exact_failed_stage(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        assert detail["sample_timestamps"] == [
            float(value) for value in range(9)
        ]
        source_failure_fact_id = detail["attempts"][0]["failure_fact_id"]
        draft = detail["draft"]
        draft["occurrences"] = [
            {
                "annotation_id": "a1-success",
                "coarse_attempt_id": detail["attempts"][0]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a1",
                "occurrence_index_in_attempt": 1,
                "start_at": 0.0,
                "end_at": 1.0,
                "outcome": "SUCCESS",
                "confidence": 1.0,
                "source_failure_fact_id": None,
            },
            {
                "annotation_id": "a2-failure",
                "coarse_attempt_id": detail["attempts"][0]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a2",
                "occurrence_index_in_attempt": 2,
                "start_at": 1.0,
                "end_at": 3.0,
                "outcome": "FAILURE",
                "confidence": 1.0,
                "source_failure_fact_id": source_failure_fact_id,
            },
            {
                "annotation_id": "a2-retry-success",
                "coarse_attempt_id": detail["attempts"][1]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a2",
                "occurrence_index_in_attempt": 1,
                "start_at": 4.2,
                "end_at": 6.0,
                "outcome": "SUCCESS",
                "confidence": 1.0,
                "source_failure_fact_id": None,
            },
            {
                "annotation_id": "a3-success",
                "coarse_attempt_id": detail["attempts"][1]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a3",
                "occurrence_index_in_attempt": 2,
                "start_at": 6.0,
                "end_at": 8.0,
                "outcome": "SUCCESS",
                "confidence": 1.0,
                "source_failure_fact_id": None,
            },
        ]
        draft["deletion_history"] = [
            {
                "episodeUuid": episode_uuid,
                "deletedOccurrence": dict(draft["occurrences"][0]),
                "targetAnnotationId": "a2-failure",
                "targetStartBefore": 1.0,
                "targetEndBefore": 3.0,
                "targetStartAfter": 0.0,
                "targetEndAfter": 3.0,
            }
        ]

        report = application.commit(episode_uuid, draft)

        assert report["coverage_status"] == "COMPLETE"
        assert report["exclusion_count"] == 1
        assert report["draft"]["deletion_history"] == draft[
            "deletion_history"
        ]
        annotation_root = (
            tmp_path
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
            / report["annotation_set_id"]
        )
        assert (annotation_root / "binding.cjson").is_file()
        assert (annotation_root / "annotations.cjson").is_file()
        assert (annotation_root / "manifest.cjson").is_file()

        draft_path = application.workspace.draft_path(
            episode_uuid=episode_uuid
        )
        persisted_before_reopen = draft_path.read_bytes()
        reopened = application.reopen_sealed_annotation(
            episode_uuid,
            annotation_set_id=report["annotation_set_id"],
        )
        assert reopened["source_annotation_set"]["annotation_set_id"] == (
            report["annotation_set_id"]
        )
        assert reopened["draft"]["occurrences"] == draft["occurrences"]
        assert draft_path.read_bytes() == persisted_before_reopen

        second_report = application.commit(
            episode_uuid,
            reopened["draft"],
        )
        assert second_report["annotation_set_id"] != report["annotation_set_id"]
        assert (annotation_root / "manifest.cjson").is_file()
        second_root = (
            tmp_path
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
            / second_report["annotation_set_id"]
        )
        assert (second_root / "manifest.cjson").is_file()
        assert application.episode_detail(episode_uuid)[
            "annotation_set_count"
        ] == 2

        with pytest.raises(
            RefinementAnnotationApplicationError,
            match="unavailable or invalid",
        ):
            application.reopen_sealed_annotation(
                episode_uuid,
                annotation_set_id="missing-version",
            )
    finally:
        application.close()


def test_posthoc_tail_outcome_can_round_trip_between_success_and_failure(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        final_attempt = detail["attempts"][-1]
        draft = detail["draft"]
        tail = next(
            value
            for value in reversed(draft["occurrences"])
            if value["coarse_attempt_id"] == final_attempt["attempt_id"]
        )
        assert tail["end_at"] == final_attempt["end_at"]
        assert tail["source_failure_fact_id"] is None

        tail["outcome"] = "FAILURE"
        failed = application.commit(episode_uuid, draft)
        assert failed["draft"]["occurrences"][-1]["outcome"] == "FAILURE"
        assert failed["exclusion_count"] == 2

        restored_draft = failed["draft"]
        restored_draft["occurrences"][-1]["outcome"] = "SUCCESS"
        restored = application.commit(episode_uuid, restored_draft)
        assert restored["draft"]["occurrences"][-1]["outcome"] == "SUCCESS"
        assert restored["exclusion_count"] == 1
    finally:
        application.close()


def test_second_review_open_clears_pending_label_and_seals_review_mode(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    standard = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        draft = standard.episode_detail(episode_uuid)["draft"]
        draft["episode_review_request"] = {
            "annotation_id": "episode-review:fixture",
            "note": "complex occasional case",
        }
        pending_report = standard.commit(episode_uuid, draft)
        assert pending_report["annotation_mode"] == "STANDARD"
        assert pending_report["coverage_status"] == "EXCLUDED_EPISODE"
        pending_set = load_stage_annotation_set(
            tmp_path
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
            / pending_report["annotation_set_id"]
        )
        assert pending_set.binding.annotation_mode == "STANDARD"
        assert pending_set.bundle.episode_review_request is not None
    finally:
        standard.close()

    review = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
        second_review=True,
    )
    try:
        assert review.list_episodes()["episodes"][0][
            "needs_second_review"
        ] is True
        detail = review.episode_detail(episode_uuid)
        assert detail["annotation_mode"] == "SECOND_REVIEW"
        assert detail["review_label_auto_cleared"] is True
        assert detail["draft"]["episode_review_request"] is None
        assert detail["draft"]["draft_annotation_mode"] == "SECOND_REVIEW"
        assert detail["draft_persistence_state"] == "SECOND_REVIEW_UNSEALED"
        assert detail["draft_requires_seal"] is True

        reviewed_report = review.commit(episode_uuid, detail["draft"])
        assert reviewed_report["annotation_mode"] == "SECOND_REVIEW"
        assert reviewed_report["coverage_status"] == "COMPLETE"
        reviewed_set = load_stage_annotation_set(
            tmp_path
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
            / reviewed_report["annotation_set_id"]
        )
        assert reviewed_set.binding.annotation_mode == "SECOND_REVIEW"
        assert reviewed_set.bundle.episode_review_request is None
        assert review.list_episodes()["episodes"][0][
            "needs_second_review"
        ] is False
        reviewed_detail = review.episode_detail(episode_uuid)
        assert reviewed_detail["draft_persistence_state"] == "SEALED_SNAPSHOT"
        assert reviewed_detail["draft_requires_seal"] is False
    finally:
        review.close()


def test_standard_mode_cannot_clear_a_sealed_review_request(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        draft = application.episode_detail(episode_uuid)["draft"]
        draft["episode_review_request"] = {
            "annotation_id": "episode-review:sealed",
            "note": "requires an independent review",
        }
        pending_report = application.commit(episode_uuid, draft)
        annotation_dir = (
            tmp_path
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
            / pending_report["annotation_set_id"]
        )
        assert load_optional_recoverable_failure_annotation_set(
            annotation_dir
        ) is None

        reopened = application.episode_detail(episode_uuid)["draft"]
        reopened["episode_review_request"] = None
        with pytest.raises(
            RefinementAnnotationApplicationError,
            match="can only be cleared.*SECOND_REVIEW",
        ):
            application.commit(episode_uuid, reopened)

        summary = application.list_episodes()["episodes"][0]
        assert summary["annotation_set_count"] == 1
        assert summary["review_state"] == "PENDING"
        assert summary["needs_second_review"] is True
        assert (
            application.episode_detail(episode_uuid)["annotation_sets"][0][
                "annotation_set_id"
            ]
            == pending_report["annotation_set_id"]
        )
    finally:
        application.close()


def test_episode_summary_reports_whole_episode_abnormal_independently(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        draft = application.episode_detail(episode_uuid)["draft"]
        draft["privileged_tools_enabled"] = True
        draft["episode_abnormal"] = {
            "annotation_id": "episode-abnormal:fixture",
            "reason_code": "BAD_CAPTURE",
            "note": "whole trajectory is unusable",
        }
        application.commit(episode_uuid, draft)

        summary = application.list_episodes()["episodes"][0]

        assert summary["whole_episode_abnormal"] is True
        assert summary["whole_episode_abnormal_reason"] == "BAD_CAPTURE"
        assert summary["review_state"] == "NONE"
        assert summary["needs_second_review"] is False
    finally:
        application.close()


def test_failure_trigger_defaults_refines_and_is_sealed_as_sidecar(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        trigger = detail["draft"]["failure_triggers"][0]
        assert trigger["source"] == "CAPTURE_DEFAULT"
        assert trigger["trigger_sample_index"] == 3
        assert trigger["lower_bound_sample_index"] == 0
        assert trigger["upper_bound_sample_index"] == 3

        trigger["trigger_sample_index"] = 2
        trigger["trigger_at"] = 2.0
        trigger["source"] = "POST_CAPTURE_REFINED"
        saved = application.save_draft(episode_uuid, detail["draft"])
        assert saved["failure_triggers"][0]["trigger_at"] == 2.0

        report = application.commit(episode_uuid, saved)
        annotation_dir = (
            tmp_path
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
            / report["annotation_set_id"]
        )
        sidecar = load_failure_trigger_annotation_set(annotation_dir)
        assert report["failure_trigger_count"] == 1
        assert report["recoverable_failure_count"] == 0
        assert not (annotation_dir / "recoverable_failures.cjson").exists()
        assert sidecar.annotations[0].trigger_sample_index == 2
        assert sidecar.annotations[0].trigger_at == 2.0
        assert application.reopen_sealed_annotation(
            episode_uuid,
            annotation_set_id=report["annotation_set_id"],
        )["draft"]["recoverable_failures"] == []
    finally:
        application.close()


def test_recoverable_failure_is_stage_bound_optional_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_station.refinement_annotation.application as application_module

    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        draft = detail["draft"]
        source_failure_fact_id = detail["attempts"][0]["failure_fact_id"]
        draft["occurrences"] = [
            {
                "annotation_id": "a1-failure",
                "coarse_attempt_id": detail["attempts"][0]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a1",
                "occurrence_index_in_attempt": 1,
                "start_at": 0.0,
                "end_at": 3.0,
                "outcome": "FAILURE",
                "confidence": 1.0,
                "source_failure_fact_id": source_failure_fact_id,
            },
            {
                "annotation_id": "a1-retry-success",
                "coarse_attempt_id": detail["attempts"][1]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a1",
                "occurrence_index_in_attempt": 1,
                "start_at": 4.2,
                "end_at": 5.5,
                "outcome": "SUCCESS",
                "confidence": 1.0,
                "source_failure_fact_id": None,
            },
            {
                "annotation_id": "a2-retry-success",
                "coarse_attempt_id": detail["attempts"][1]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a2",
                "occurrence_index_in_attempt": 2,
                "start_at": 5.5,
                "end_at": 6.0,
                "outcome": "SUCCESS",
                "confidence": 1.0,
                "source_failure_fact_id": None,
            },
            {
                "annotation_id": "a3-retry-success",
                "coarse_attempt_id": detail["attempts"][1]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a3",
                "occurrence_index_in_attempt": 3,
                "start_at": 6.0,
                "end_at": 8.0,
                "outcome": "SUCCESS",
                "confidence": 1.0,
                "source_failure_fact_id": None,
            },
        ]
        added = application.add_recoverable_failure(
            episode_uuid,
            draft,
            onset_at=6.1,
            recovered_at=7.1,
            note="recovered without Retry",
        )["draft"]
        event = added["recoverable_failures"][0]
        assert event["stage_occurrence_annotation_id"] == (
            "a3-retry-success"
        )
        assert event["stage_id"] == "a3"
        assert event["onset_sample_index"] == 6
        assert event["recovered_sample_index"] == 7

        original_commit = application_module.commit_stage_annotation_set
        observed_visibility_boundary = False

        def observe_sidecars_before_manifest(**kwargs: object) -> object:
            nonlocal observed_visibility_boundary
            binding = kwargs["binding"]
            annotation_dir = (
                tmp_path
                / "meta"
                / "progress"
                / "stage_annotations"
                / episode_uuid
                / binding.annotation_set_id  # type: ignore[union-attr]
            )
            assert (annotation_dir / "failure_triggers.cjson").is_file()
            assert (annotation_dir / "recoverable_failures.cjson").is_file()
            assert not (annotation_dir / "manifest.cjson").exists()
            observed_visibility_boundary = True
            return original_commit(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            application_module,
            "commit_stage_annotation_set",
            observe_sidecars_before_manifest,
        )
        report = application.commit(episode_uuid, added)
        assert observed_visibility_boundary
        assert report["recoverable_failure_count"] == 1
        assert report["exclusion_count"] == 1
        annotation_dir = (
            tmp_path
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
            / report["annotation_set_id"]
        )
        sidecar = load_recoverable_failure_annotation_set(annotation_dir)
        analysis_sidecar = load_analysis_recoverable_set(annotation_dir)
        assert analysis_sidecar == sidecar
        assert sidecar.annotations[0].stage_id == "a3"
        assert sidecar.annotations[0].note == "recovered without Retry"
        reopened = application.reopen_sealed_annotation(
            episode_uuid,
            annotation_set_id=report["annotation_set_id"],
        )
        assert reopened["draft"]["recoverable_failures"] == added[
            "recoverable_failures"
        ]
    finally:
        application.close()


def test_recoverable_failure_rejects_cross_stage_points_without_writing(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        with pytest.raises(
            RefinementAnnotationApplicationError,
            match="both lie inside one successful stage",
        ):
            application.add_recoverable_failure(
                episode_uuid,
                detail["draft"],
                onset_at=2.0,
                recovered_at=5.0,
            )
        annotation_root = (
            tmp_path / "meta" / "progress" / "stage_annotations"
        )
        assert list(annotation_root.rglob("manifest.cjson")) == []
    finally:
        application.close()


@pytest.mark.parametrize("change_kind", ["ABNORMAL_INTERVAL", "STAGE_BOUNDARY"])
def test_recoverable_failure_fails_closed_after_owning_stage_changes(
    tmp_path: Path,
    change_kind: str,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        draft = detail["draft"]
        retry_occurrences = [
            value
            for value in draft["occurrences"]
            if value["source_failure_fact_id"] is None
        ]
        retry_occurrences[0]["end_at"] = 5.0
        retry_occurrences[1]["start_at"] = 5.0
        retry_occurrences[1]["end_at"] = 6.0
        retry_occurrences[2]["start_at"] = 6.0
        with_recoverable = application.add_recoverable_failure(
            episode_uuid,
            draft,
            onset_at=6.1,
            recovered_at=7.1,
            note="must remain in the final success stage",
        )["draft"]

        if change_kind == "ABNORMAL_INTERVAL":
            with_recoverable["privileged_tools_enabled"] = True
            with_recoverable["abnormal_intervals"] = [
                {
                    "annotation_id": "abnormal:cuts-recovery",
                    "start_at": 6.5,
                    "end_at": 7.5,
                    "reason_code": "BAD_CAPTURE",
                    "note": None,
                }
            ]
            expected = "both lie inside one successful stage"
        else:
            retry_occurrences = [
                value
                for value in with_recoverable["occurrences"]
                if value["source_failure_fact_id"] is None
            ]
            retry_occurrences[1]["end_at"] = 7.5
            retry_occurrences[2]["start_at"] = 7.5
            expected = "stage reference changed"

        with pytest.raises(
            RefinementAnnotationApplicationError,
            match=expected,
        ):
            application.commit(episode_uuid, with_recoverable)

        annotation_root = (
            tmp_path / "meta" / "progress" / "stage_annotations"
        )
        assert list(annotation_root.rglob("manifest.cjson")) == []
    finally:
        application.close()


def test_annotation_discovery_uses_sealed_snapshot_not_metadata_config_hash(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    catalog_path = tmp_path / "meta" / "episodes.jsonl"
    record = json.loads(catalog_path.read_text(encoding="utf-8"))
    progress_reference = record["metadata"]["progress_episode"]
    progress_reference["task_progress_profile_hash"] = "0" * 64
    progress_reference["progress_episode_binding_hash"] = "1" * 64
    catalog_path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    items = list_progress_annotation_work_items(tmp_path)

    assert tuple(item.episode_uuid for item in items) == (episode_uuid,)


def test_episode_detail_reuses_immutable_sample_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_station.refinement_annotation.application as application_module

    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    calls = 0
    original = application_module.load_episode_sample_axis

    def counted_load(item: object) -> object:
        nonlocal calls
        calls += 1
        return original(item)  # type: ignore[arg-type]

    monkeypatch.setattr(
        application_module,
        "load_episode_sample_axis",
        counted_load,
    )
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        application.episode_detail(episode_uuid)
        application.episode_detail(episode_uuid)
        assert calls == 1
    finally:
        application.close()


def test_application_discovers_newly_sealed_episode_while_open(
    tmp_path: Path,
) -> None:
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        assert application.list_episodes()["episodes"] == []

        episode_uuid = _finalized_failure_retry_dataset(tmp_path)
        payload = application.list_episodes()

        assert [item["episode_uuid"] for item in payload["episodes"]] == [
            episode_uuid
        ]
    finally:
        application.close()


def test_background_episode_refresh_does_not_block_cached_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_station.refinement_annotation.application as application_module

    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    started = threading.Event()
    release = threading.Event()
    original = application_module.read_progress_annotation_work_item_batch

    def blocked_discovery(root: object, **kwargs: object) -> object:
        started.set()
        assert release.wait(timeout=3.0)
        return original(root, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        application_module,
        "read_progress_annotation_work_item_batch",
        blocked_discovery,
    )
    catalog = tmp_path / "meta" / "episodes.jsonl"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("", encoding="utf-8")
    try:
        began = time.monotonic()
        assert application.list_episodes(wait_for_refresh=False)[
            "episodes"
        ] == []
        elapsed = time.monotonic() - began
        assert elapsed < 0.5
        assert started.wait(timeout=1.0)

        release.set()
        application.refresh()
        assert application.list_episodes()["episodes"] == []
    finally:
        release.set()
        application.close()


def test_episode_catalog_cursor_reads_only_appended_lines_and_falls_back(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "meta" / "episodes.jsonl"
    catalog.parent.mkdir(parents=True)
    catalog.write_text('{"metadata": null}\n', encoding="utf-8")

    first = read_progress_annotation_work_item_batch(tmp_path)
    assert first.full_scan is True
    assert first.cursor is not None
    assert first.cursor.line_number == 1

    with catalog.open("a", encoding="utf-8") as handle:
        handle.write('{"metadata": null}\n')
    appended = read_progress_annotation_work_item_batch(
        tmp_path,
        cursor=first.cursor,
    )
    assert appended.full_scan is False
    assert appended.cursor is not None
    assert appended.cursor.line_number == 2

    catalog.write_text('{"metadata": null}\n', encoding="utf-8")
    replaced = read_progress_annotation_work_item_batch(
        tmp_path,
        cursor=appended.cursor,
    )
    assert replaced.full_scan is True
    assert replaced.cursor is not None
    assert replaced.cursor.line_number == 1


def test_full_catalog_scan_hydrates_records_in_bounded_parallel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataset_interface.progress_annotation_jobs as jobs_module

    catalog = tmp_path / "meta" / "episodes.jsonl"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        "".join(
            json.dumps({"record_index": index}) + "\n"
            for index in range(12)
        ),
        encoding="utf-8",
    )
    lock = threading.Lock()
    active = 0
    max_active = 0
    progress: list[tuple[int, int]] = []

    def observed_hydration(
        _root: Path,
        _record: object,
        **_kwargs: object,
    ) -> None:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return None

    monkeypatch.setattr(
        jobs_module,
        "_work_item_from_record",
        observed_hydration,
    )
    batch = read_progress_annotation_work_item_batch(
        tmp_path,
        progress_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )

    assert batch.full_scan is True
    assert batch.items == ()
    assert max_active > 1
    assert progress[0] == (0, 12)
    assert progress[-1] == (12, 12)


def test_incremental_catalog_scan_remains_sequential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataset_interface.progress_annotation_jobs as jobs_module

    catalog = tmp_path / "meta" / "episodes.jsonl"
    catalog.parent.mkdir(parents=True)
    catalog.write_text('{"record_index":0}\n', encoding="utf-8")
    first = read_progress_annotation_work_item_batch(tmp_path)
    assert first.cursor is not None
    with catalog.open("a", encoding="utf-8") as handle:
        handle.write('{"record_index":1}\n')

    calls: list[int] = []

    def observed_hydration(
        _root: Path,
        record: object,
        **_kwargs: object,
    ) -> None:
        assert isinstance(record, dict)
        calls.append(int(record["record_index"]))
        return None

    monkeypatch.setattr(
        jobs_module,
        "_work_item_from_record",
        observed_hydration,
    )
    progress: list[tuple[int, int]] = []
    appended = read_progress_annotation_work_item_batch(
        tmp_path,
        cursor=first.cursor,
        progress_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )

    assert appended.full_scan is False
    assert calls == [1]
    assert progress == [(0, 1), (1, 1)]


def test_application_poll_uses_catalog_cursor_after_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_station.refinement_annotation.application as application_module

    catalog = tmp_path / "meta" / "episodes.jsonl"
    catalog.parent.mkdir(parents=True)
    catalog.write_text('{"metadata": null}\n', encoding="utf-8")
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    original = application_module.read_progress_annotation_work_item_batch
    cursors: list[object] = []

    def observed_discovery(root: object, **kwargs: object) -> object:
        cursors.append(kwargs.get("cursor"))
        return original(root, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        application_module,
        "read_progress_annotation_work_item_batch",
        observed_discovery,
    )
    try:
        with catalog.open("a", encoding="utf-8") as handle:
            handle.write('{"metadata": null}\n')

        application.list_episodes()

        assert len(cursors) == 1
        assert cursors[0] is not None
    finally:
        application.close()


def test_full_catalog_scan_publishes_verified_items_incrementally(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    observed: list[str] = []

    batch = read_progress_annotation_work_item_batch(
        tmp_path,
        item_callback=lambda item: observed.append(item.episode_uuid),
        load_annotation_state=False,
    )

    assert observed == [episode_uuid]
    assert tuple(item.episode_uuid for item in batch.items) == (episode_uuid,)


def test_persistent_read_cache_avoids_reloading_immutable_catalog_and_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_station.refinement_annotation.application as application_module

    dataset_root = tmp_path / "dataset"
    episode_uuid = _finalized_failure_retry_dataset(dataset_root)
    cache_root = tmp_path / "read-cache"
    first = RefinementAnnotationApplication(
        dataset_root,
        annotator_id="first",
        read_cache=ProgressAnnotationReadCache(cache_root),
    )
    try:
        first.episode_detail(episode_uuid)
    finally:
        first.close()

    def unexpected_catalog(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("immutable catalog was reloaded")

    def unexpected_sample_axis(_item: object) -> object:
        raise AssertionError("immutable sample axis was reloaded")

    monkeypatch.setattr(
        application_module,
        "read_progress_annotation_work_item_batch",
        unexpected_catalog,
    )
    monkeypatch.setattr(
        application_module,
        "load_episode_sample_axis",
        unexpected_sample_axis,
    )
    second = RefinementAnnotationApplication(
        dataset_root,
        annotator_id="second",
        read_cache=ProgressAnnotationReadCache(cache_root),
    )
    try:
        detail = second.episode_detail(episode_uuid)
        assert detail["episode_uuid"] == episode_uuid
        assert len(detail["sample_timestamps"]) == 9
    finally:
        second.close()


def test_persistent_catalog_prefix_hydrates_only_appended_records_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_station.refinement_annotation.application as application_module

    dataset_root = tmp_path / "dataset"
    episode_uuid = _finalized_failure_retry_dataset(dataset_root)
    cache_root = tmp_path / "read-cache"
    first = RefinementAnnotationApplication(
        dataset_root,
        annotator_id="first",
        read_cache=ProgressAnnotationReadCache(cache_root),
    )
    first.close()
    with (dataset_root / "meta" / "episodes.jsonl").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write('{"metadata":null}\n')

    original = application_module.read_progress_annotation_work_item_batch
    observed_cursors: list[object] = []

    def observed_read(root: object, **kwargs: object) -> object:
        observed_cursors.append(kwargs.get("cursor"))
        return original(root, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        application_module,
        "read_progress_annotation_work_item_batch",
        observed_read,
    )
    second = RefinementAnnotationApplication(
        dataset_root,
        annotator_id="second",
        read_cache=ProgressAnnotationReadCache(cache_root),
    )
    try:
        assert observed_cursors and observed_cursors[0] is not None
        assert [
            value["episode_uuid"]
            for value in second.list_episodes()["episodes"]
        ] == [episode_uuid]
    finally:
        second.close()


def test_cached_episode_refuses_escaped_source_and_releases_temporary_lock(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    episode_uuid = _finalized_failure_retry_dataset(dataset_root)
    cache_root = tmp_path / "read-cache"
    first = RefinementAnnotationApplication(
        dataset_root,
        annotator_id="first",
        read_cache=ProgressAnnotationReadCache(cache_root),
    )
    first.close()
    parquet = next(dataset_root.glob("data/**/*.parquet"))
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(parquet.read_bytes())
    parquet.unlink()
    parquet.symlink_to(outside)

    second = RefinementAnnotationApplication(
        dataset_root,
        annotator_id="second",
        read_cache=ProgressAnnotationReadCache(cache_root),
    )
    try:
        with pytest.raises(
            RefinementAnnotationApplicationError,
            match="sealed source files are no longer available",
        ):
            second.episode_detail(episode_uuid)
        with DatasetStageAnnotationWorkspace(
            dataset_root,
            annotator_id="probe",
        ) as probe:
            probe.acquire_episode(episode_uuid=episode_uuid)
            probe.release_episode(episode_uuid=episode_uuid)
    finally:
        second.close()


def test_visible_annotation_summary_reuses_unchanged_immutable_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import control_station.refinement_annotation.application as application_module

    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    writer = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="writer",
        read_cache=ProgressAnnotationReadCache(tmp_path / "writer-cache"),
    )
    try:
        detail = writer.episode_detail(episode_uuid)
        writer.commit(episode_uuid, detail["draft"])
    finally:
        writer.close()

    calls = 0
    original = application_module.load_stage_annotation_set

    def counted_load(path: object) -> object:
        nonlocal calls
        calls += 1
        return original(path)  # type: ignore[arg-type]

    monkeypatch.setattr(
        application_module,
        "load_stage_annotation_set",
        counted_load,
    )
    reader = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="reader",
        read_cache=ProgressAnnotationReadCache(tmp_path / "reader-cache"),
    )
    try:
        first_summary = reader.episode_annotation_summaries((episode_uuid,))
        second_summary = reader.episode_annotation_summaries((episode_uuid,))
        assert calls == 1
        assert first_summary["summaries"][0]["annotation_set_count"] == 1
        assert second_summary["summaries"][0]["annotation_set_count"] == 1

        other = RefinementAnnotationApplication(
            tmp_path,
            annotator_id="other",
            read_cache=ProgressAnnotationReadCache(tmp_path / "other-cache"),
        )
        try:
            detail = other.episode_detail(episode_uuid)
            other.commit(episode_uuid, detail["draft"])
        finally:
            other.close()

        calls = 0
        changed = reader.episode_annotation_summaries((episode_uuid,))
        assert changed["summaries"][0]["annotation_set_count"] == 2
        assert calls == 2
    finally:
        reader.close()


def test_annotation_set_object_cache_is_bounded_but_summaries_remain(
    tmp_path: Path,
) -> None:
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        for index in range(20):
            application._cache_annotation_sets(f"episode-{index}", ())

        assert len(application._annotation_sets_cache) == 16
        assert len(application._annotation_summary_cache) == 20
    finally:
        application.close()


def test_commit_does_not_rescan_episode_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)

        def unexpected_refresh(**_kwargs: object) -> None:
            raise AssertionError("commit unexpectedly rescanned the catalog")

        monkeypatch.setattr(application, "refresh", unexpected_refresh)
        report = application.commit(episode_uuid, detail["draft"])
        assert report["annotation_set_id"]
    finally:
        application.close()


def test_disabled_privileged_tools_reject_abnormal_payload(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        draft = application.episode_detail(episode_uuid)["draft"]
        draft["privileged_tools_enabled"] = False
        draft["abnormal_intervals"] = [
            {
                "annotation_id": "abnormal:must-be-rejected",
                "start_at": 1.0,
                "end_at": 2.0,
                "reason_code": "BAD_CAPTURE",
                "note": None,
            }
        ]

        with pytest.raises(
            ValueError,
            match="explicit privileged-tools switch",
        ):
            application.save_draft(episode_uuid, draft)
    finally:
        application.close()


def test_abnormal_interval_can_cover_fixed_failure_without_rewriting_it(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        draft = detail["draft"]
        draft["privileged_tools_enabled"] = True
        draft["abnormal_intervals"] = [
            {
                "annotation_id": "abnormal:failure-region",
                "start_at": 1.0,
                "end_at": 3.5,
                "reason_code": "BAD_CAPTURE",
                "note": "fixed capture Failure remains visible",
            }
        ]
        draft["occurrences"] = [
            {
                "annotation_id": "a1-success",
                "coarse_attempt_id": detail["attempts"][0]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a1",
                "occurrence_index_in_attempt": 1,
                "start_at": 0.0,
                "end_at": 1.0,
                "outcome": "SUCCESS",
                "confidence": 1.0,
                "source_failure_fact_id": None,
            },
            {
                "annotation_id": "a2-retry-success",
                "coarse_attempt_id": detail["attempts"][1]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a2",
                "occurrence_index_in_attempt": 1,
                "start_at": 4.2,
                "end_at": 6.0,
                "outcome": "SUCCESS",
                "confidence": 1.0,
                "source_failure_fact_id": None,
            },
            {
                "annotation_id": "a3-success",
                "coarse_attempt_id": detail["attempts"][1]["attempt_id"],
                "progress_step_id": "a",
                "stage_id": "a3",
                "occurrence_index_in_attempt": 2,
                "start_at": 6.0,
                "end_at": 8.0,
                "outcome": "SUCCESS",
                "confidence": 1.0,
                "source_failure_fact_id": None,
            },
        ]

        report = application.commit(episode_uuid, draft)

        assert report["coverage_status"] == "COMPLETE"
        assert report["exclusion_count"] == 1
        persisted = application.episode_detail(episode_uuid)["draft"]
        assert persisted["abnormal_intervals"][0]["start_at"] == 1.0
        assert detail["attempts"][0]["failure_reported_at"] == 3.0
    finally:
        application.close()


def test_abnormal_interval_inside_stage_is_split_at_commit(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        draft = detail["draft"]
        selected = next(
            value
            for value in draft["occurrences"]
            if value["outcome"] == "SUCCESS"
        )
        original_start = selected["start_at"]
        original_end = selected["end_at"]
        duration = original_end - original_start
        abnormal_start = original_start + duration * 0.4
        abnormal_end = original_start + duration * 0.6
        draft["privileged_tools_enabled"] = True
        draft["abnormal_intervals"] = [
            {
                "annotation_id": "abnormal:inside-stage",
                "start_at": abnormal_start,
                "end_at": abnormal_end,
                "reason_code": "BAD_CAPTURE",
                "note": None,
            }
        ]

        report = application.commit(episode_uuid, draft)

        committed = report["draft"]
        fragments = [
            value
            for value in committed["occurrences"]
            if (
                value["coarse_attempt_id"]
                == selected["coarse_attempt_id"]
                and value["stage_id"] == selected["stage_id"]
            )
        ]
        assert len(fragments) == 2
        assert fragments[0]["start_at"] == original_start
        assert fragments[0]["end_at"] == abnormal_start
        assert fragments[0]["outcome"] == "TRUNCATED_BY_ABNORMAL"
        assert fragments[0]["source_failure_fact_id"] is None
        assert fragments[1]["start_at"] == abnormal_end
        assert fragments[1]["end_at"] == original_end
        assert fragments[1]["outcome"] == selected["outcome"]
        assert all(
            not (
                value["start_at"] < abnormal_end
                and abnormal_start < value["end_at"]
            )
            for value in committed["occurrences"]
        )
    finally:
        application.close()


def test_abnormal_interval_does_not_implicitly_delete_success_stage(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        draft = detail["draft"]
        selected = next(
            value
            for value in draft["occurrences"]
            if value["outcome"] == "SUCCESS"
        )
        draft["privileged_tools_enabled"] = True
        draft["abnormal_intervals"] = [
            {
                "annotation_id": "abnormal:whole-stage",
                "start_at": selected["start_at"],
                "end_at": selected["end_at"],
                "reason_code": "BAD_CAPTURE",
                "note": None,
            }
        ]

        with pytest.raises(
            RefinementAnnotationApplicationError,
            match="fully covers a successful stage occurrence",
        ):
            application.commit(episode_uuid, draft)
    finally:
        application.close()


def test_removing_abnormal_interval_rejoins_derived_same_stage_fragments(
    tmp_path: Path,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        draft = detail["draft"]
        selected = next(
            value
            for value in draft["occurrences"]
            if value["outcome"] == "SUCCESS"
        )
        original_start = selected["start_at"]
        original_end = selected["end_at"]
        duration = original_end - original_start
        draft["privileged_tools_enabled"] = True
        draft["abnormal_intervals"] = [
            {
                "annotation_id": "abnormal:temporary",
                "start_at": original_start + duration * 0.4,
                "end_at": original_start + duration * 0.6,
                "reason_code": "BAD_CAPTURE",
                "note": None,
            }
        ]
        first_report = application.commit(episode_uuid, draft)
        reedited = first_report["draft"]
        reedited["abnormal_intervals"] = []

        reconciled = application.reconcile_abnormal_intervals(
            episode_uuid,
            reedited,
        )["draft"]
        immediate = [
            value
            for value in reconciled["occurrences"]
            if (
                value["coarse_attempt_id"]
                == selected["coarse_attempt_id"]
                and value["stage_id"] == selected["stage_id"]
            )
        ]
        assert len(immediate) == 1
        assert immediate[0]["start_at"] == original_start
        assert immediate[0]["end_at"] == original_end
        assert immediate[0]["outcome"] == selected["outcome"]

        second_report = application.commit(episode_uuid, reconciled)

        restored = [
            value
            for value in second_report["draft"]["occurrences"]
            if (
                value["coarse_attempt_id"]
                == selected["coarse_attempt_id"]
                and value["stage_id"] == selected["stage_id"]
            )
        ]
        assert len(restored) == 1
        assert restored[0]["start_at"] == original_start
        assert restored[0]["end_at"] == original_end
        assert restored[0]["outcome"] == selected["outcome"]
    finally:
        application.close()


@pytest.mark.parametrize("removed_index", [0, 1])
def test_removing_one_of_multiple_abnormal_intervals_rejoins_only_its_gap(
    tmp_path: Path,
    removed_index: int,
) -> None:
    episode_uuid = _finalized_failure_retry_dataset(tmp_path)
    application = RefinementAnnotationApplication(
        tmp_path,
        annotator_id="fixture",
    )
    try:
        detail = application.episode_detail(episode_uuid)
        draft = detail["draft"]
        selected = next(
            value
            for value in draft["occurrences"]
            if value["outcome"] == "SUCCESS"
        )
        original_start = selected["start_at"]
        original_end = selected["end_at"]
        duration = original_end - original_start
        intervals = [
            {
                "annotation_id": "abnormal:first",
                "start_at": original_start + duration * 0.2,
                "end_at": original_start + duration * 0.3,
                "reason_code": "BAD_CAPTURE",
                "note": None,
            },
            {
                "annotation_id": "abnormal:second",
                "start_at": original_start + duration * 0.6,
                "end_at": original_start + duration * 0.7,
                "reason_code": "BAD_CAPTURE",
                "note": None,
            },
        ]
        draft["privileged_tools_enabled"] = True
        draft["abnormal_intervals"] = intervals
        committed = application.commit(episode_uuid, draft)["draft"]

        remaining = intervals[1 - removed_index]
        committed["abnormal_intervals"] = [remaining]
        reconciled = application.reconcile_abnormal_intervals(
            episode_uuid,
            committed,
        )["draft"]
        fragments = sorted(
            (
                value
                for value in reconciled["occurrences"]
                if (
                    value["coarse_attempt_id"]
                    == selected["coarse_attempt_id"]
                    and value["stage_id"] == selected["stage_id"]
                )
            ),
            key=lambda value: value["start_at"],
        )

        assert len(fragments) == 2
        assert fragments[0]["start_at"] == pytest.approx(original_start)
        assert fragments[0]["end_at"] == pytest.approx(
            remaining["start_at"]
        )
        assert fragments[0]["outcome"] == "TRUNCATED_BY_ABNORMAL"
        assert fragments[1]["start_at"] == pytest.approx(
            remaining["end_at"]
        )
        assert fragments[1]["end_at"] == pytest.approx(original_end)
        assert fragments[1]["outcome"] == selected["outcome"]
    finally:
        application.close()
