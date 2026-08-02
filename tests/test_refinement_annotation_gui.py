from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import quote

import pytest

from collection_runtime.progress import (
    StageAnnotationStoreError,
    load_stage_annotation_set,
)
from collection_runtime.progress.failure_trigger_store import (
    load_failure_trigger_annotation_set,
)
from collection_runtime.sample_acceptance import AcceptedSampleReceipt
from control_station.refinement_annotation import RefinementAnnotationGui
from control_station.refinement_annotation.application import (
    RefinementAnnotationApplication,
    RefinementAnnotationApplicationError,
)
from control_station.refinement_annotation.cli import build_parser
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
        assert "function browseDatasetPath" in app_source
        assert "function scheduleNextEpisodeWarmup" in app_source
        assert "function showVideoLoadPrompt" in app_source
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
        assert ".workspace.episode-panel-is-collapsed" in style_source
        with pytest.raises(StageAnnotationStoreError, match="already owns"):
            RefinementAnnotationGui(
                tmp_path,
                annotator_id="other",
                open_browser=False,
            )
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
    finally:
        review.close()


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
        sidecar = load_failure_trigger_annotation_set(
            tmp_path
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
            / report["annotation_set_id"]
        )
        assert report["failure_trigger_count"] == 1
        assert sidecar.annotations[0].trigger_sample_index == 2
        assert sidecar.annotations[0].trigger_at == 2.0
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

        second_report = application.commit(episode_uuid, reedited)

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
