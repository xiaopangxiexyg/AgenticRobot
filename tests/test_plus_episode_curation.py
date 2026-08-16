from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from collection_runtime.episode_curation import (
    EpisodeCurationStore as CollectionRuntimeEpisodeCurationStore,
    load_task_variant_profile as load_collection_runtime_task_variant_profile,
)
from dataset_contracts import (
    EpisodeCurationContractError,
    EpisodeCurationRecord,
    TaskVariant,
    TaskVariantProfile,
    annotation_episode_id,
)
from dataset_interface import DexTeleopDatasetIndex, EpisodeCurationReader, source_dataset_base_fingerprint
from teleop_runtime.collection.episode_curation import (
    EpisodeCurationSession,
    EpisodeCurationStore,
    load_task_variant_profile,
)


def test_legacy_store_imports_are_object_identical_lower_owned_exports() -> None:
    assert EpisodeCurationStore is CollectionRuntimeEpisodeCurationStore
    assert (
        load_task_variant_profile
        is load_collection_runtime_task_variant_profile
    )
    assert EpisodeCurationStore.__module__ == "collection_runtime.episode_curation"
    assert (
        EpisodeCurationSession.__module__
        == "teleop_runtime.collection.episode_curation"
    )


def _legacy_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.3", "dataset_name": "fixture"}),
        encoding="utf-8",
    )
    (root / "meta" / "tasks.jsonl").write_text('{"task_index":0,"task":"fixture"}\n', encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text(
        '{"episode_index":0,"frame_count":3}\n{"episode_index":1,"frame_count":2}\n',
        encoding="utf-8",
    )


def test_old_dataset_without_sidecar_is_readable(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _legacy_dataset(dataset)
    reader = EpisodeCurationReader(tmp_path / "missing-overlay")
    assert reader.for_episode(dataset, 0) is None
    assert source_dataset_base_fingerprint(dataset) == source_dataset_base_fingerprint(dataset)
    index = DexTeleopDatasetIndex(dataset)
    assert index.plus_episode_curation(0, None) is None
    assert index.plus_episode_curation(0, tmp_path / "missing-overlay") is None


def test_store_writes_external_append_only_overlay_and_is_idempotent(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _legacy_dataset(dataset)
    source_before = {
        path.relative_to(dataset).as_posix(): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    profile = TaskVariantProfile(
        task_profile_id="fixture_v1",
        task_id="fixture",
        variants=(TaskVariant("left", "左侧"), TaskVariant("right", "右侧")),
    )
    store = EpisodeCurationStore(
        annotation_root=tmp_path / "annotations",
        source_dataset_root=dataset,
        source_binding="v23_compat_content_locator_v1",
        task_profile=profile,
        source_schema="dexteleop_dataset_v2.3",
    )
    record = store.append_for_episode(
        episode_index=0,
        disposition="keep_recovery",
        reason_code="operator_recovery",
        variant_id="left",
    )
    same = store.append(record)
    assert same == record
    sidecar = tmp_path / "annotations" / "episode_curation.jsonl"
    assert sidecar.exists()
    assert not (dataset / "meta" / "annotations").exists()
    assert source_before == {
        path.relative_to(dataset).as_posix(): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    assert len(sidecar.read_text(encoding="utf-8").splitlines()) == 1
    reader = EpisodeCurationReader(tmp_path / "annotations")
    assert reader.for_episode(dataset, 0) == record
    assert reader.status == "draft"
    assert not reader.usable_for_training
    store.set_status("complete")
    reader = EpisodeCurationReader(tmp_path / "annotations")
    assert reader.usable_for_training
    with (dataset / "meta" / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"episode_index":2,"frame_count":1}\n')
    assert reader.for_episode(dataset, 0) == record


def test_base_fingerprint_and_store_remain_stable_across_episode_counter_updates(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _legacy_dataset(dataset)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update({"episode_count": 2, "frame_count": 5})
    info_path.write_text(json.dumps(info), encoding="utf-8")
    initial_fingerprint = source_dataset_base_fingerprint(dataset)

    annotations = tmp_path / "annotations"
    first = EpisodeCurationStore(
        annotation_root=annotations,
        source_dataset_root=dataset,
    ).append_for_episode(episode_index=0, disposition="keep_nominal")
    info.update({"episode_count": 3, "frame_count": 6})
    info_path.write_text(json.dumps(info), encoding="utf-8")

    assert source_dataset_base_fingerprint(dataset) == initial_fingerprint
    second = EpisodeCurationStore(
        annotation_root=annotations,
        source_dataset_root=dataset,
    ).append_for_episode(episode_index=1, disposition="keep_nominal")
    reader = EpisodeCurationReader(annotations)
    assert reader.for_episode(dataset, 0) == first
    assert reader.for_episode(dataset, 1) == second


def test_persisted_plus_records_lock_live_source_reindexing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _legacy_dataset(dataset)
    session = EpisodeCurationSession(
        enabled=True,
        annotation_root=tmp_path / "annotations",
        source_binding="v23_compat_content_locator_v1",
        source_schema="dexteleop_dataset_v2.3",
        status="ready",
    )
    assert session.persisted_binding_lock_reason() == ""
    session.mark_pending(dataset_root=dataset, episode_index=0)
    session.save(
        disposition="keep_nominal",
        pending_token=session.pending_token,
    )

    assert "禁止切换数据集或原地删除" in session.persisted_binding_lock_reason()


def test_v24_compatibility_binding_reuses_the_same_overlay_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "v24_dataset"
    _legacy_dataset(dataset)
    (dataset / "meta" / "info.json").write_text(
        json.dumps({"dexteleop_dataset_schema": "dexteleop_dataset_v2.4"}),
        encoding="utf-8",
    )
    store = EpisodeCurationStore(
        annotation_root=tmp_path / "v24_annotations",
        source_dataset_root=dataset,
        source_binding="v24_compatibility_content_locator_v1",
        source_schema="dexteleop_dataset_v2.4",
    )
    record = store.append_for_episode(episode_index=0, disposition="keep_nominal")
    assert record.source_binding == "v24_compatibility_content_locator_v1"
    assert record.source_schema == "dexteleop_dataset_v2.4"
    assert EpisodeCurationReader(tmp_path / "v24_annotations").for_episode(dataset, 0) == record


def test_conflicting_record_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _legacy_dataset(dataset)
    store = EpisodeCurationStore(
        annotation_root=tmp_path / "annotations",
        source_dataset_root=dataset,
    )
    record = store.append_for_episode(episode_index=0, disposition="keep_nominal")
    conflicting = EpisodeCurationRecord.from_dict(
        {**record.to_dict(), "disposition": "exclude", "reason_code": "task_failure"}
    )
    with pytest.raises(ValueError, match="conflicting"):
        store.append(conflicting)


def test_store_rejects_missing_source_episode(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _legacy_dataset(dataset)
    store = EpisodeCurationStore(
        annotation_root=tmp_path / "annotations",
        source_dataset_root=dataset,
    )
    with pytest.raises(ValueError, match="missing source episode"):
        store.append_for_episode(episode_index=99, disposition="unknown")


def test_reason_code_and_variant_are_validated() -> None:
    with pytest.raises(EpisodeCurationContractError):
        EpisodeCurationRecord(
            annotation_episode_id="0" * 64,
            legacy_episode_index=0,
            source_dataset_base_fingerprint="1" * 64,
            source_episode_manifest_hash=None,
            source_schema="v2.3",
            source_binding="v23_compat_content_locator_v1",
            disposition="keep_nominal",
            reason_code="task_failure",
        )
    profile = TaskVariantProfile(
        task_profile_id="fixture_v1",
        variants=(TaskVariant("left", "左侧"),),
    )
    with pytest.raises(EpisodeCurationContractError):
        profile.require_variant("free form operator note")
    assert annotation_episode_id(
        source_dataset_base_fingerprint="1" * 64,
        source_episode_manifest_hash=None,
        legacy_episode_index=0,
    ) == annotation_episode_id(
        source_dataset_base_fingerprint="1" * 64,
        source_episode_manifest_hash=None,
        legacy_episode_index=0,
    )


def test_runtime_session_is_default_off_and_requires_explicit_root(tmp_path: Path) -> None:
    disabled = EpisodeCurationSession.from_args(
        SimpleNamespace(plus_episode_curation=False),
        source_binding="v23_compat_content_locator_v1",
    )
    assert disabled.status == "off"
    with pytest.raises(ValueError, match="annotation-root"):
        EpisodeCurationSession.from_args(
            SimpleNamespace(plus_episode_curation=True, plus_annotation_root=""),
            source_binding="v23_compat_content_locator_v1",
        )
    enabled = EpisodeCurationSession.from_args(
        SimpleNamespace(
            plus_episode_curation=True,
            plus_annotation_root=str(tmp_path / "annotations"),
            plus_task_profile="",
        ),
        source_binding="v24_compatibility_content_locator_v1",
    )
    assert enabled.enabled
    assert enabled.source_binding == "v24_compatibility_content_locator_v1"


def test_runtime_session_binds_review_to_one_pending_episode(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _legacy_dataset(dataset)
    session = EpisodeCurationSession(
        enabled=True,
        annotation_root=tmp_path / "annotations",
        source_binding="v23_compat_content_locator_v1",
        source_schema="dexteleop_dataset_v2.3",
        status="ready",
    )
    session.mark_pending(dataset_root=dataset, episode_index=0)
    token = session.pending_token
    assert token
    assert session.matches_pending(episode_index=0, pending_token=token)
    assert not session.matches_pending(episode_index=1, pending_token=token)

    session.set_review(disposition="exclude", reason_code="task_failure")
    assert session.ui_state()["plus_curation_disposition"] == "exclude"
    record = session.save(
        disposition="exclude",
        reason_code="task_failure",
        pending_token=token,
    )
    assert record.legacy_episode_index == 0
    assert session.pending_episode_index is None
    assert session.ui_state()["plus_curation_disposition"] == "unknown"


def test_runtime_session_rejects_stale_pending_token(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _legacy_dataset(dataset)
    session = EpisodeCurationSession(
        enabled=True,
        annotation_root=tmp_path / "annotations",
        source_binding="v23_compat_content_locator_v1",
        source_schema="dexteleop_dataset_v2.3",
        status="ready",
    )
    session.mark_pending(dataset_root=dataset, episode_index=0)
    with pytest.raises(RuntimeError, match="token"):
        session.save(
            disposition="unknown",
            reason_code="unknown",
            pending_token="stale-token",
        )


def test_runtime_session_can_finish_snapshot_after_next_episode_begins(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _legacy_dataset(dataset)
    session = EpisodeCurationSession(
        enabled=True,
        annotation_root=tmp_path / "annotations",
        source_binding="v23_compat_content_locator_v1",
        source_schema="dexteleop_dataset_v2.3",
        status="ready",
    )
    session.mark_pending(dataset_root=dataset, episode_index=0)
    token = session.pending_token
    session.begin_episode(episode_index=1)
    record = session.save(
        disposition="keep_nominal",
        reason_code="unknown",
        pending_token=token,
        source_episode_index=0,
        source_dataset_root=dataset,
    )
    assert record.legacy_episode_index == 0
    assert session.pending_episode_index is None
