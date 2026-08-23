from __future__ import annotations

import json
from pathlib import Path

import pytest

from collection_runtime.progress import (
    CompleteEpisodeProgressRequest,
    ProgressController,
    ProgressFactStore,
    ProgressRequestContext,
    StartEpisodeProgressRequest,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_interface.progress_artifacts import load_sealed_progress_episode
from utils.dataset_postprocess.organize_episodes import (
    DatasetOrganizationError,
    _materialize_file,
    _output_relative_path,
    _passes_minimum_duration,
    _source_task_binding_snapshot,
    build_datasets,
    plan_dataset_build,
)


def _hand_vector(value: float) -> list[float]:
    return [0.0, float(value), 0.0, 0.0, 0.0, 0.0]


def _require_pyarrow():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    return pa, pq


def _make_source(root: Path) -> None:
    pa, pq = _require_pyarrow()
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.front/chunk-000").mkdir(parents=True)
    action = [0.0] * 26
    action[14:20] = _hand_vector(0.7)
    action[20:26] = _hand_vector(0.4)
    state = [0.0] * 38
    state[13:19] = _hand_vector(0.6)
    state[32:38] = _hand_vector(0.3)
    table = pa.table(
        {
            "index": pa.array([40, 41]),
            "episode_index": pa.array([5, 5]),
            "frame_index": pa.array([0, 1]),
            "action": pa.array([action, action]),
            "observation.last_action": pa.array([action, action]),
            "observation.state": pa.array([state, state]),
            "action.left_hand.joint_target": pa.array([_hand_vector(0.7), _hand_vector(0.7)]),
            "action.right_hand.joint_target": pa.array([_hand_vector(0.4), _hand_vector(0.4)]),
            "observation.left_hand.qpos": pa.array([_hand_vector(0.6), _hand_vector(0.6)]),
            "observation.right_hand.qpos": pa.array([_hand_vector(0.3), _hand_vector(0.3)]),
            "observation.images.front.path": pa.array(
                [
                    "videos/observation.images.front/chunk-000/episode_000005.mp4",
                    "videos/observation.images.front/chunk-000/episode_000005.mp4",
                ]
            ),
            "hand_action_specs_json": pa.array([json.dumps({"left": {}}), json.dumps({"left": {}})]),
        }
    )
    pq.write_table(table, root / "data/chunk-000/episode_000005.parquet")
    source_video = root / "videos/observation.images.front/chunk-000/episode_000005.mp4"
    source_video.write_bytes(b"synthetic-video")
    info = {
        "format": "lerobot_v2.1_style",
        "dataset_name": "source",
        "episode_count": 1,
        "frame_count": 2,
        "nominal_fps": 15.0,
        "chunk_size": 1000,
        "task_id": "cube",
        "task_description": "cube",
        "features": {"action": {"shape": [26], "dtype": "float32"}},
        "hand_action_specs_by_side": {"left": {"hand_model": "inspire_tac"}},
        "hand_active_mask_by_side": {"left": [True] * 6},
    }
    record = {
        "episode_index": 5,
        "episode_id": "episode_000005",
        "frame_count": 2,
        "duration_sec": 2.0,
        "parquet_path": "data/chunk-000/episode_000005.parquet",
        "video_paths": {"front": "videos/observation.images.front/chunk-000/episode_000005.mp4"},
        "summary": {
            "video_frame_stats": {
                "front": {
                    "path": "videos/observation.images.front/chunk-000/episode_000005.mp4",
                    "written": 2,
                }
            }
        },
        "metadata": {
            "hand_action_specs_by_side": {"left": {"hand_model": "inspire_tac"}},
            "gripper_policy": {"output_dim": 1},
        },
    }
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_id": "cube", "description": "cube"}) + "\n",
        encoding="utf-8",
    )
    (root / "meta/episodes.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "meta/episodes_stats.jsonl").write_text(json.dumps({"episode_index": 5, "frame_count": 2}) + "\n", encoding="utf-8")


def _add_progress(root: Path, *, episode_uuid: str) -> str:
    identity = EpisodeCaptureIdentity(
        episode_uuid=episode_uuid,
        legacy_episode_index=5,
        lifecycle_owner_id="organize-fixture",
        prepared_at=1.0,
        prepared_clock_domain_id="fixture-clock",
    )
    profile = TaskProgressProfileV1(
        profile_id="organize-profile",
        profile_version="1",
        task_id="cube",
        steps=(
            ProgressStepDefinitionV1(
                progress_step_id="execute",
                semantic_type="EXECUTE",
                display_name="Execute",
                ordinal=1,
            ),
        ),
        initial_step_id="execute",
        terminal_step_ids=("execute",),
    )
    binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash=identity.semantic_content_hash,
        task_progress_profile_hash=profile.semantic_content_hash,
        bound_clock_domain_id="episode-clock",
        recording_scope="FULL_TASK",
        entry_step_id="execute",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
    )
    episode_dir = root / "meta/progress/episodes" / episode_uuid
    store = ProgressFactStore(
        episode_dir=episode_dir,
        identity=identity,
        profile=profile,
        binding=binding,
    )
    controller = ProgressController(store=store)

    def context(request_id: str, at: float) -> ProgressRequestContext:
        return ProgressRequestContext(
            request_id=request_id,
            effective_at=at,
            effective_clock_domain_id="episode-clock",
            available_at=at,
            available_clock_domain_id="episode-clock",
            actor_id="fixture",
        )

    controller.start(
        StartEpisodeProgressRequest("attempt-1"),
        context=context("start", 1.0),
    )
    controller.complete(
        CompleteEpisodeProgressRequest(),
        context=context("complete", 2.0),
    )
    report = controller.seal(
        sealed_at=3.0,
        sealed_clock_domain_id="episode-clock",
    )
    dataset_profile = root / "meta/progress/profile.cjson"
    dataset_profile.parent.mkdir(parents=True, exist_ok=True)
    dataset_profile.write_bytes((episode_dir / "profile.cjson").read_bytes())
    catalog = root / "meta/episodes.jsonl"
    record = json.loads(catalog.read_text(encoding="utf-8"))
    record["metadata"]["progress_episode"] = {
        "schema": "dexteleop.real_teleop_progress_episode_reference.v1",
        "episode_uuid": episode_uuid,
        "episode_identity_hash": identity.semantic_content_hash,
        "manifest_ref": (
            episode_dir / "manifest.cjson"
        ).relative_to(root).as_posix(),
        "binding_ref": (
            episode_dir / "binding.cjson"
        ).relative_to(root).as_posix(),
        "profile_ref": (
            episode_dir / "profile.cjson"
        ).relative_to(root).as_posix(),
        "progress_episode_binding_hash": binding.semantic_content_hash,
        "task_progress_profile_hash": profile.semantic_content_hash,
        "progress_manifest_hash": report.manifest.semantic_content_hash,
    }
    catalog.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return identity.semantic_content_hash


@pytest.mark.optional
def test_build_reindexes_and_migrates_legacy_dmtac_slot(tmp_path: Path) -> None:
    _pa, pq = _require_pyarrow()
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_source(source)
    manifest = {
        "min_duration_sec": 2.0,
        "datasets": [
            {
                "name": "derived",
                "output_root": "output",
                "task": {"id": "cube", "description": "cube"},
                "sources": [
                    {
                        "dataset_root": "source",
                        "episode_indices": [5],
                        "transforms": ["dmtac_slot1_to_slot0"],
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = plan_dataset_build(manifest_path)
    assert plan[0].episode_count == 1
    assert plan[0].minimum_duration_sec == 2.0
    assert plan[0].filtered_by_minimum_duration == ()
    assert plan[0].episodes[0].source_episode_index == 5
    build_datasets(manifest_path)

    output_record = json.loads((output / "meta/episodes.jsonl").read_text().strip())
    assert output_record["episode_index"] == 0
    assert output_record["episode_id"] == "episode_000000"
    assert output_record["parquet_path"] == "data/chunk-000/episode_000000.parquet"
    assert output_record["video_paths"]["front"].endswith("episode_000000.mp4")
    assert output_record["summary"]["video_frame_stats"]["front"]["path"].endswith(
        "episode_000000.mp4"
    )
    source_video = source / "videos/observation.images.front/chunk-000/episode_000005.mp4"
    output_video = output / "videos/observation.images.front/chunk-000/episode_000000.mp4"
    assert source_video.stat().st_ino == output_video.stat().st_ino

    output_table = pq.read_table(output / output_record["parquet_path"])
    assert output_table["episode_index"].to_pylist() == [0, 0]
    assert output_table["index"].to_pylist() == [0, 1]
    assert output_table["observation.images.front.path"].to_pylist() == [
        "videos/observation.images.front/chunk-000/episode_000000.mp4",
        "videos/observation.images.front/chunk-000/episode_000000.mp4",
    ]
    assert output_table["action"].to_pylist()[0][14:20] == pytest.approx([0.7, 0, 0, 0, 0, 0])
    assert output_table["action"].to_pylist()[0][20:26] == pytest.approx([0.4, 0, 0, 0, 0, 0])
    assert output_table["observation.state"].to_pylist()[0][13:19] == pytest.approx([0.6, 0, 0, 0, 0, 0])

    source_table = pq.read_table(source / "data/chunk-000/episode_000005.parquet")
    assert source_table["action"].to_pylist()[0][14:20] == pytest.approx([0, 0.7, 0, 0, 0, 0])
    output_info = json.loads((output / "meta/info.json").read_text())
    assert output_info["dataset_name"] == "derived"
    assert output_info["task_id"] == "cube"
    assert output_info["task_description"] == "cube"
    assert [
        json.loads(line)
        for line in (output / "meta/tasks.jsonl").read_text().splitlines()
    ] == [{"task_id": "cube", "description": "cube"}]
    assert output_info["hand_action_specs_by_side"]["left"]["hand_model"] == "dmtac_gripper"
    assert output_info["hand_active_mask_by_side"]["left"] == [True, False, False, False, False, False]


def test_reindex_path_accepts_an_already_zero_episode_name() -> None:
    assert _output_relative_path(
        Path("data/chunk-000/episode_000000.parquet"),
        output_episode_index=0,
        chunk_size=1000,
    ) == Path("data/chunk-000/episode_000000.parquet")


def test_rewrite_paths_does_not_treat_episode_event_names_as_paths() -> None:
    from utils.dataset_postprocess.organize_episodes import _rewrite_paths_in_value

    payload = {
        "event": "episode_start",
        "path": "data/chunk-000/episode_000005.parquet",
    }

    assert _rewrite_paths_in_value(
        payload,
        output_episode_index=2,
        chunk_size=1000,
    ) == {
        "event": "episode_start",
        "path": "data/chunk-000/episode_000002.parquet",
    }


def test_minimum_duration_filter_is_inclusive_at_the_threshold() -> None:
    assert not _passes_minimum_duration({"episode_index": 7, "duration_sec": 1.999}, 2.0)
    assert _passes_minimum_duration({"episode_index": 8, "duration_sec": 2.0}, 2.0)
    assert _passes_minimum_duration({"episode_index": 9, "duration_sec": 2.001}, 2.0)


@pytest.mark.optional
def test_plan_rejects_source_task_id_that_differs_from_canonical_task(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _make_source(source)
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["task_id"] = "another-task"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    (source / "meta/tasks.jsonl").write_text(
        json.dumps({"task_id": "another-task", "description": "cube"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "physical",
                        "output_root": "physical",
                        "task": {"id": "cube", "description": "cube"},
                        "sources": [{"dataset_root": "source", "episode_indices": "all"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetOrganizationError, match="source task_id differs"):
        plan_dataset_build(manifest_path)


@pytest.mark.optional
def test_reviewed_metadata_only_task_rebinding_is_audited(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_source(source)
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["task_id"] = "legacy-writing"
    info["task_description"] = "legacy display"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    (source / "meta/tasks.jsonl").write_text(
        json.dumps({"task_id": "legacy-writing", "description": "legacy display"})
        + "\n"
        + json.dumps({"task_id": "legacy-alias", "description": "legacy alias"})
        + "\n",
        encoding="utf-8",
    )
    _snapshot, snapshot_sha256 = _source_task_binding_snapshot(source, info)
    manifest = {
        "datasets": [
            {
                "name": "canonical-writing",
                "output_root": "output",
                "task": {
                    "id": "canonical-writing",
                    "description": "canonical writing",
                },
                "sources": [
                    {
                        "dataset_root": "source",
                        "episode_indices": "all",
                        "reviewed_task_rebinding": {
                            "schema": "dexteleop.dataset_source_task_rebinding.v1",
                            "source_binding_sha256": snapshot_sha256,
                            "target_task_id": "canonical-writing",
                            "target_task_description": "canonical writing",
                            "reason": "Reviewed legacy metadata aliases; frames own no task column.",
                        },
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = plan_dataset_build(manifest_path)[0]
    assert len(plan.task_rebindings) == 1
    assert plan.task_rebindings[0].source_binding_sha256 == snapshot_sha256
    build_datasets(manifest_path)

    output_info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert output_info["task_id"] == "canonical-writing"
    assert output_info["task_description"] == "canonical writing"
    assert [
        json.loads(line)
        for line in (output / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
    ] == [{"task_id": "canonical-writing", "description": "canonical writing"}]
    audit = json.loads((output / "meta/build_audit.json").read_text(encoding="utf-8"))
    assert audit["plan"]["task_rebindings"][0]["source_binding_sha256"] == snapshot_sha256
    assert json.loads(info_path.read_text(encoding="utf-8"))["task_id"] == "legacy-writing"


@pytest.mark.optional
def test_reviewed_task_rebinding_rejects_stale_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    manifest = {
        "datasets": [
            {
                "name": "canonical-writing",
                "output_root": "output",
                "task": {
                    "id": "canonical-writing",
                    "description": "canonical writing",
                },
                "sources": [
                    {
                        "dataset_root": "source",
                        "episode_indices": "all",
                        "reviewed_task_rebinding": {
                            "schema": "dexteleop.dataset_source_task_rebinding.v1",
                            "source_binding_sha256": "0" * 64,
                            "target_task_id": "canonical-writing",
                            "target_task_description": "canonical writing",
                            "reason": "Reviewed legacy source.",
                        },
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetOrganizationError, match="source snapshot changed"):
        plan_dataset_build(manifest_path)


@pytest.mark.optional
def test_reviewed_task_rebinding_rejects_frame_task_columns(tmp_path: Path) -> None:
    pa, pq = _require_pyarrow()
    source = tmp_path / "source"
    _make_source(source)
    parquet_path = source / "data/chunk-000/episode_000005.parquet"
    table = pq.read_table(parquet_path).append_column(
        "task_index",
        pa.array([0, 0]),
    )
    pq.write_table(table, parquet_path)
    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    _snapshot, snapshot_sha256 = _source_task_binding_snapshot(source, info)
    manifest = {
        "datasets": [
            {
                "name": "canonical-writing",
                "output_root": "output",
                "task": {
                    "id": "canonical-writing",
                    "description": "canonical writing",
                },
                "sources": [
                    {
                        "dataset_root": "source",
                        "episode_indices": "all",
                        "reviewed_task_rebinding": {
                            "schema": "dexteleop.dataset_source_task_rebinding.v1",
                            "source_binding_sha256": snapshot_sha256,
                            "target_task_id": "canonical-writing",
                            "target_task_description": "canonical writing",
                            "reason": "Reviewed legacy source.",
                        },
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetOrganizationError, match="frame-level task columns"):
        plan_dataset_build(manifest_path)


@pytest.mark.optional
def test_plan_rejects_source_with_internally_inconsistent_task_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _make_source(source)
    (source / "meta/tasks.jsonl").write_text(
        json.dumps({"task_id": "cube", "description": "wrong description"})
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "physical",
                        "output_root": "physical",
                        "task": {"id": "cube", "description": "canonical description"},
                        "sources": [{"dataset_root": "source", "episode_indices": "all"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetOrganizationError, match="invalid task binding"):
        plan_dataset_build(manifest_path)


@pytest.mark.optional
def test_physical_merge_preserves_progress_and_uses_independent_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _make_source(first)
    _make_source(second)
    first_identity = _add_progress(first, episode_uuid="episode-a")
    second_identity = _add_progress(second, episode_uuid="episode-b")
    stale_progress = first / "meta/progress/episodes/unsealed-stale"
    source_progress = first / "meta/progress/episodes/episode-a"
    stale_progress.mkdir(parents=True)
    for name in ("profile.cjson", "binding.cjson"):
        (stale_progress / name).write_bytes((source_progress / name).read_bytes())
    (stale_progress / "facts.cjsonl").write_text("", encoding="utf-8")
    manifest = {
        "datasets": [
            {
                "name": "physical",
                "output_root": "physical",
                "task": {"id": "cube", "description": "cube"},
                "artifact_materialization_policy": "copy",
                "progress_artifact_policy": "preserve",
                "sources": [
                    {"dataset_root": "first", "episode_indices": [5]},
                    {"dataset_root": "second", "episode_indices": [5]},
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = plan_dataset_build(manifest_path)[0]
    assert plan.artifact_materialization_policy == "copy"
    assert plan.progress_artifact_policy == "preserve"
    assert plan.episode_count == 2
    build_datasets(manifest_path)

    output = tmp_path / "physical"
    records = [
        json.loads(line)
        for line in (output / "meta/episodes.jsonl").read_text().splitlines()
    ]
    assert [record["episode_index"] for record in records] == [0, 1]
    assert [
        record["metadata"]["progress_episode"]["episode_identity_hash"]
        for record in records
    ] == [first_identity, second_identity]
    for record in records:
        reference = record["metadata"]["progress_episode"]
        loaded = load_sealed_progress_episode(
            output / Path(reference["manifest_ref"]).parent,
            expected_episode_identity_hash=reference["episode_identity_hash"],
        )
        assert loaded.timeline.full_task_outcome == "SUCCESS"

    source_video = first / "videos/observation.images.front/chunk-000/episode_000005.mp4"
    output_video = output / records[0]["video_paths"]["front"]
    assert source_video.stat().st_ino != output_video.stat().st_ino
    assert output_video.stat().st_nlink == 1
    source_fact_log = first / "meta/progress/episodes/episode-a/facts.cjsonl"
    output_fact_log = output / "meta/progress/episodes/episode-a/facts.cjsonl"
    assert source_fact_log.stat().st_ino != output_fact_log.stat().st_ino
    assert not (output / "meta/progress/episodes/unsealed-stale").exists()


def test_reflink_required_never_falls_back_to_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"immutable-payload")

    def fail_reflink(*_args, **_kwargs):
        raise OSError("filesystem has no reflink support")

    monkeypatch.setattr(
        "utils.dataset_postprocess.organize_episodes._reflink_required",
        fail_reflink,
    )

    with pytest.raises(DatasetOrganizationError, match="implicit full-copy fallback"):
        _materialize_file(
            source,
            destination,
            policy="reflink_required",
        )
    assert not destination.exists()


@pytest.mark.optional
def test_plan_can_select_one_named_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "first",
                        "output_root": "first",
                        "task": {"id": "cube", "description": "cube"},
                        "sources": [{"dataset_root": "source", "episode_indices": [5]}],
                    },
                    {
                        "name": "second",
                        "output_root": "second",
                        "task": {"id": "cube", "description": "cube"},
                        "sources": [{"dataset_root": "source", "episode_indices": [5]}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    plan = plan_dataset_build(manifest_path, dataset_names=["second"])
    assert [item.name for item in plan] == ["second"]
