from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_interface.episode_curation import source_dataset_base_fingerprint
from dataset_interface.logical_task_dataset import (
    LOGICAL_TASK_DATASET_POLICY_V1,
    LOGICAL_TASK_DATASET_SCHEMA_V1,
    LogicalTaskDatasetError,
    audit_logical_task_dataset_sources,
    load_logical_task_dataset_manifest,
    logical_task_dataset_hash,
)


def _write_manifest(
    tmp_path: Path,
    *,
    task_scope: str = "stage1_only",
    task_aliases: tuple[str, ...] = (),
) -> Path:
    source = tmp_path / "source"
    meta = source / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "task_id": "historical-capture-name",
                "episode_count": 2,
                "frame_count": 31,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (meta / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_id": task_id}) + "\n"
            for task_id in ("historical-capture-name", *task_aliases)
        ),
        encoding="utf-8",
    )
    payload = {
        "schema": LOGICAL_TASK_DATASET_SCHEMA_V1,
        "logical_dataset_id": "writing-stage1-test",
        "logical_dataset_version": "1",
        "canonical_task_id": "canonical-writing-task",
        "canonical_task_description": "Canonical writing task.",
        "progress_step_id": "right_acquire_brush",
        "source_identity_policy": LOGICAL_TASK_DATASET_POLICY_V1,
        "scenario_classes": [
            {
                "scenario_class": "initial_multi_choice",
                "description": "Choose the correct brush slot at the start.",
            }
        ],
        "sources": [
            {
                "source_dataset_id": "source-a",
                "dataset_root": str(source),
                "source_task_id": "historical-capture-name",
                "source_task_id_aliases": list(task_aliases),
                "source_dataset_fingerprint": source_dataset_base_fingerprint(source),
                "scenario_class": "initial_multi_choice",
                "task_scope": task_scope,
                "raw_episode_count": 2,
                "raw_frame_count": 31,
            }
        ],
        "totals": {
            "source_count": 1,
            "raw_episode_count": 2,
            "raw_frame_count": 31,
            "by_scenario_class": {
                "initial_multi_choice": {
                    "source_count": 1,
                    "raw_episode_count": 2,
                    "raw_frame_count": 31,
                }
            },
        },
    }
    payload["semantic_content_hash"] = logical_task_dataset_hash(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_logical_task_dataset_keeps_source_task_as_provenance(
    tmp_path: Path,
) -> None:
    manifest = load_logical_task_dataset_manifest(_write_manifest(tmp_path))
    assert manifest.canonical_task_id == "canonical-writing-task"
    assert manifest.sources[0].source_task_id == "historical-capture-name"
    assert manifest.sources[0].source_task_id_aliases == ()
    assert audit_logical_task_dataset_sources(manifest)["ok"] is True


def test_logical_task_dataset_pins_explicit_historical_task_aliases(
    tmp_path: Path,
) -> None:
    manifest = load_logical_task_dataset_manifest(
        _write_manifest(tmp_path, task_aliases=("legacy-task-row",))
    )
    assert manifest.sources[0].source_task_id_aliases == ("legacy-task-row",)
    assert audit_logical_task_dataset_sources(manifest)["ok"] is True

    tasks_path = manifest.sources[0].dataset_root / "meta" / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps({"task_id": "historical-capture-name"}) + "\n",
        encoding="utf-8",
    )
    report = audit_logical_task_dataset_sources(manifest)
    assert report["ok"] is False
    assert "TASK_MANIFEST_ID_MISMATCH" in report["sources"][0]["errors"]


def test_logical_task_dataset_audit_detects_source_identity_change(
    tmp_path: Path,
) -> None:
    manifest = load_logical_task_dataset_manifest(_write_manifest(tmp_path))
    info_path = manifest.sources[0].dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["task_id"] = "rewritten-task"
    info_path.write_text(json.dumps(info) + "\n", encoding="utf-8")

    report = audit_logical_task_dataset_sources(manifest)
    assert report["ok"] is False
    assert report["sources"][0]["errors"] == [
        "SOURCE_TASK_ID_MISMATCH",
        "SOURCE_DATASET_FINGERPRINT_MISMATCH",
    ]


def test_stage1_only_scenario_rejects_broader_scope(tmp_path: Path) -> None:
    with pytest.raises(LogicalTaskDatasetError, match="must be stage1_only"):
        load_logical_task_dataset_manifest(
            _write_manifest(tmp_path, task_scope="full_writing_sequence")
        )


def test_stage1_training_snapshot_matches_logical_organization() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = load_logical_task_dataset_manifest(
        repository
        / "dataset_interface/logical_task_manifests/"
        "writing_stage1_scenarios_through_2026_08_11_v1.json"
    )
    profile = json.loads(
        (
            repository
            / "deployments/openpi/configs/training/"
            "writing_stage1_scenarios_0804_0811_front_h16_absolute_repeat146_v1.json"
        ).read_text(encoding="utf-8")
    )
    manifest_sources = {
        item.source_dataset_id: item for item in manifest.sources
    }
    assert set(manifest_sources) == {
        str(item["source_dataset_id"]) for item in profile["sources"]
    }
    expected_weights = {
        "fixed_middle": 1,
        "initial_multi_choice": 4,
        "midcourse_interrupt": 6,
    }
    for item in profile["sources"]:
        source = manifest_sources[str(item["source_dataset_id"])]
        assert Path(item["dataset_root"]).resolve() == source.dataset_root
        assert item["expected_task_id"] == source.source_task_id
        assert item["scenario_class"] == source.scenario_class
        assert item["task_scope"] == source.task_scope
        assert item["expected_raw_episode_count"] == source.raw_episode_count
        assert item["expected_raw_frame_count"] == source.raw_frame_count
        assert item["repeat_weight"] == expected_weights[source.scenario_class]


def test_stage3_fulltask_profiles_match_logical_source_organization() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = load_logical_task_dataset_manifest(
        repository
        / "dataset_interface/logical_task_manifests/"
        "writing_stage3_fulltask_sources_2026_08_04_through_08_07_v1.json"
    )
    profile_root = repository / "deployments/openpi/configs/training"
    baseline = json.loads(
        (
            profile_root
            / "writing_stage3_0804_0806_front_h16_absolute_v1.json"
        ).read_text(encoding="utf-8")
    )
    replay = json.loads(
        (
            profile_root
            / "writing_stage3_0804_0806_front_h16_absolute_replay_v2.json"
        ).read_text(encoding="utf-8")
    )
    extended_profiles = tuple(
        json.loads(
            (
                profile_root
                / f"writing_stage{stage}_0804_0807_front_h16_absolute_v1.json"
            ).read_text(encoding="utf-8")
        )
        for stage in (2, 3, 4)
    )
    extended = extended_profiles[1]

    assert manifest.progress_step_id == "right_write"
    assert {source.task_scope for source in manifest.sources} == {
        "full_writing_sequence"
    }
    assert sum(source.raw_episode_count for source in manifest.sources) == 1241
    assert sum(source.raw_frame_count for source in manifest.sources) == 471797

    manifest_sources = {
        source.source_dataset_id: source for source in manifest.sources
    }
    logical_source_root = Path(
        "/home/user/zc/datasets/RealmanDMTAC/LogicalCollections/"
        "writing_four_stage_fulltask_2026_08_04_through_08_07_v1/sources"
    )
    baseline_ids = tuple(item["source_dataset_id"] for item in baseline["sources"])
    extended_ids = tuple(item["source_dataset_id"] for item in extended["sources"])
    assert baseline_ids == tuple(manifest_sources)[:5]
    assert all(
        tuple(item["source_dataset_id"] for item in profile["sources"])
        == tuple(manifest_sources)
        for profile in extended_profiles
    )
    assert extended_ids == tuple(manifest_sources)
    assert baseline["expected_total_eligible_episode_count"] == 879
    assert extended["expected_total_eligible_episode_count"] == 1200
    assert replay["sources"] == baseline["sources"]
    assert replay["expected_prepared_binding_hash"] == (
        "28c9d7693324fe738f635e6da11ab8875dd793302af6e4dcf3081687b838f39e"
    )
    assert replay["annotation_selection"][
        "a4f276be5948bb7c1900b047c31f69c5ea7c4695e7aa3d2e7e05864d8bf90f22"
    ] == "stage-1786011149701-e78dad71"

    for profile in (baseline, replay, *extended_profiles):
        assert profile["expected_task_id"] == manifest.canonical_task_id
        for item in profile["sources"]:
            source = manifest_sources[item["source_dataset_id"]]
            assert Path(item["dataset_root"]) == (
                logical_source_root / source.source_dataset_id
            )
    assert baseline["target_skill_id"] == "writing.write_with_brush"
    assert replay["target_skill_id"] == "writing.write_with_brush"
    assert tuple(profile["target_skill_id"] for profile in extended_profiles) == (
        "writing.remove_brush_cap_and_return_cap",
        "writing.write_with_brush",
        "writing.return_brush",
    )


def test_all_openpi_profiles_use_the_fulltask_logical_source_view() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = load_logical_task_dataset_manifest(
        repository
        / "dataset_interface/logical_task_manifests/"
        "writing_stage3_fulltask_sources_2026_08_04_through_08_07_v1.json"
    )
    logical_source_root = Path(
        "/home/user/zc/datasets/RealmanDMTAC/LogicalCollections/"
        "writing_four_stage_fulltask_2026_08_04_through_08_07_v1/sources"
    )
    known_source_ids = {
        source.source_dataset_id for source in manifest.sources
    }
    checked_bindings = 0
    profile_root = repository / "deployments/openpi/configs/training"
    for profile_path in sorted(profile_root.glob("*.json")):
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        sources = profile.get("sources", ())
        if not isinstance(sources, list):
            continue
        for item in sources:
            source_id = str(item.get("source_dataset_id", ""))
            if source_id not in known_source_ids:
                continue
            assert Path(str(item["dataset_root"])) == (
                logical_source_root / source_id
            ), profile_path.name
            checked_bindings += 1
    assert checked_bindings > len(known_source_ids)
