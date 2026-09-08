from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from collection_runtime.progress.profiles import (
    DATASET_PROGRESS_PROFILE_RELATIVE_PATH,
    DatasetProgressProfileBindingError,
    bind_dataset_progress_profile,
    common_progress_semantic_type,
    inspect_dataset_progress_profile,
    list_managed_task_profiles,
    load_managed_task_profile,
    load_task_progress_profile_file,
    load_task_stage_profile_file,
    read_dataset_progress_profile,
    resolve_task_progress_profile_reference,
)
from collection_runtime.immutable_canonical_store import publish_immutable_canonical
from dataset_contracts.progress_facts import PROGRESS_PROFILE_FILE_DOMAIN_V1
from dataset_contracts.progress_refinement import (
    validate_task_refinement_profile,
)


def test_catalog_separates_linear_and_tree_namespaces() -> None:
    linear = list_managed_task_profiles(topology_kind="LINEAR")
    tree = list_managed_task_profiles(topology_kind="ROOTED_TREE")

    assert tuple(item.profile_id for item in linear) == (
        "dmtac_bimanual_writing_acquire_present_window_v1",
        "dmtac_rm65_bimanual_position_v1_progress",
        "dmtac_rm65_bimanual_position_writing_five_stage_progress",
        "dmtac_rm65_bimanual_position_writing_four_stage_progress",
        "dmtac_rm65_bimanual_position_writing_three_stage_progress",
        "dmtac_rm65_full_trajectory_replay_v1_progress",
        "dmtac_rm65_right_gripper_rule_writing_progress",
        "example_three_stage_pick_place_progress",
        "test_dmtac_rm65_bimanual_position_v1_three_stage_progress",
    )
    assert all(item.topology_kind == "LINEAR" for item in linear)
    assert tree == ()


def test_dmtac_three_stage_gui_test_profile_is_explicit_and_linear() -> None:
    progress = load_managed_task_profile(
        "test_dmtac_rm65_bimanual_position_v1_three_stage_progress"
    ).progress_profile

    assert progress.task_id == "dmtac_rm65_bimanual_position_v1"
    assert progress.topology_kind == "LINEAR"
    assert tuple(step.progress_step_id for step in progress.steps) == (
        "test_prepare",
        "test_manipulate",
        "test_finish",
    )
    assert all(
        step.display_name.startswith("【测试】")
        for step in progress.steps
    )
    assert progress.allowed_next_step_ids("test_prepare") == (
        "test_manipulate",
    )
    assert progress.allowed_next_step_ids("test_manipulate") == (
        "test_finish",
    )


def test_writing_profiles_preserve_exact_coarse_and_fine_stage_semantics() -> None:
    four = load_managed_task_profile(
        "dmtac_rm65_bimanual_position_writing_four_stage_progress"
    )
    three = load_managed_task_profile(
        "dmtac_rm65_bimanual_position_writing_three_stage_progress"
    )

    assert four.progress_profile.task_id == (
        "dmtac_rm65_bimanual_position_v1"
    )
    assert tuple(
        step.progress_step_id for step in four.progress_profile.steps
    ) == (
        "right_acquire_brush",
        "left_remove_brush_cap",
        "right_write",
        "right_return_brush",
    )
    assert four.stage_profile is not None
    assert tuple(
        stage.display_name
        for stage in four.stage_profile.stages_for_step(
            "right_acquire_brush"
        )
    ) == (
        "靠近毛笔",
        "正确抓住毛笔",
        "提起毛笔并送至左手",
    )
    assert tuple(
        stage.display_name
        for stage in four.stage_profile.stages_for_step(
            "left_remove_brush_cap"
        )
    ) == ("取下笔盖", "放回笔盖")
    assert tuple(
        stage.display_name
        for stage in four.stage_profile.stages_for_step("right_write")
    ) == ("右手书写",)
    assert tuple(
        stage.display_name
        for stage in four.stage_profile.stages_for_step(
            "right_return_brush"
        )
    ) == ("调整放回位置", "放下归位")

    assert tuple(
        step.progress_step_id for step in three.progress_profile.steps
    ) == (
        "right_acquire_brush",
        "right_write",
        "right_return_brush",
    )
    assert three.progress_profile.allowed_next_step_ids(
        "right_acquire_brush"
    ) == ("right_write",)
    assert three.stage_profile is not None
    assert tuple(
        stage.display_name
        for stage in three.stage_profile.stages_for_step(
            "right_acquire_brush"
        )
    ) == (
        "靠近毛笔",
        "正确抓住毛笔",
        "提起毛笔并进入书写准备",
    )
    assert not three.stage_profile.stages_for_step(
        "left_remove_brush_cap"
    )
    assert four.stage_profile.task_progress_profile_hash == (
        four.progress_profile.semantic_content_hash
    )
    assert three.stage_profile.task_progress_profile_hash == (
        three.progress_profile.semantic_content_hash
    )


def test_detailed_writing_profile_is_an_explicit_additive_selection() -> None:
    coarse = load_managed_task_profile(
        "dmtac_rm65_bimanual_position_writing_four_stage_progress"
    )
    detailed = load_managed_task_profile(
        "dmtac_rm65_bimanual_position_writing_four_stage_progress_detailed"
    )

    assert detailed.is_detailed is True
    assert detailed.profile_id == coarse.profile_id
    assert detailed.selection_id.endswith("_detailed")
    assert detailed.progress_profile == coarse.progress_profile
    assert detailed.stage_profile is not None
    assert coarse.stage_profile is not None
    assert tuple(
        stage.stage_id
        for stage in detailed.stage_profile.stages_for_step("right_write")
    ) == (
        "right_write.pen_down",
        "right_write.write",
        "right_write.pen_up",
    )
    assert tuple(
        stage.stage_id
        for stage in coarse.stage_profile.stages_for_step("right_write")
    ) == ("right_write.write",)
    assert tuple(
        item.selection_id
        for item in list_managed_task_profiles(topology_kind="LINEAR")
    ) == tuple(
        item.profile_id
        for item in list_managed_task_profiles(topology_kind="LINEAR")
    )
    assert any(
        item.selection_id == detailed.selection_id
        for item in list_managed_task_profiles(
            topology_kind="LINEAR",
            include_detailed=True,
        )
    )


def test_right_gripper_rule_profile_is_arm_only_and_three_stage_detailed() -> None:
    bundle = load_managed_task_profile(
        "dmtac_rm65_right_gripper_rule_writing_progress_detailed"
    )

    assert bundle.progress_profile.task_id == "dmtac_rm65_rule_based_writing_v1"
    assert tuple(bundle.progress_profile.step_by_id) == ("right_write",)
    arm_control = bundle.progress_profile.step_by_id["right_write"].arm_control
    assert arm_control is not None
    assert arm_control.left == "fixed_hold"
    assert arm_control.right == "teleop"
    assert arm_control.fixed_scope == "arm_only"
    assert bundle.stage_profile is not None
    assert tuple(
        stage.stage_id
        for stage in bundle.stage_profile.stages_for_step("right_write")
    ) == (
        "right_write.pen_down",
        "right_write.write",
        "right_write.pen_up",
    )


def test_writing_five_stage_profile_adds_only_the_ink_dipping_stage() -> None:
    five = load_managed_task_profile(
        "dmtac_rm65_bimanual_position_writing_five_stage_progress"
    )

    assert tuple(
        step.progress_step_id for step in five.progress_profile.steps
    ) == (
        "right_acquire_brush",
        "left_remove_brush_cap",
        "right_dip_ink",
        "right_write",
        "right_return_brush",
    )
    assert five.progress_profile.allowed_next_step_ids(
        "left_remove_brush_cap"
    ) == ("right_dip_ink",)
    assert five.progress_profile.allowed_next_step_ids("right_dip_ink") == (
        "right_write",
    )
    assert five.stage_profile is not None
    assert tuple(
        stage.display_name
        for stage in five.stage_profile.stages_for_step("right_dip_ink")
    ) == ("沾墨水",)
    assert five.stage_profile.task_progress_profile_hash == (
        five.progress_profile.semantic_content_hash
    )
    assert all(
        step.arm_control is not None for step in five.progress_profile.steps
    )
    progress_source = (
        Path(__file__).resolve().parents[1]
        / "collection_runtime"
        / "progress"
        / "profiles"
        / "linear"
        / "dmtac_rm65_bimanual_position_writing_five_stage.progress.json"
    )
    assert load_task_progress_profile_file(
        progress_source
    ) == five.progress_profile


def test_five_stage_detailed_profile_splits_dipping_and_writing() -> None:
    coarse = load_managed_task_profile(
        "dmtac_rm65_bimanual_position_writing_five_stage_progress"
    )
    detailed = load_managed_task_profile(
        "dmtac_rm65_bimanual_position_writing_five_stage_progress_detailed"
    )

    assert detailed.is_detailed is True
    assert detailed.progress_profile == coarse.progress_profile
    assert detailed.stage_profile is not None
    assert tuple(
        stage.stage_id
        for stage in detailed.stage_profile.stages_for_step("right_dip_ink")
    ) == (
        "right_dip_ink.approach",
        "right_dip_ink.dip",
        "right_dip_ink.depart",
    )
    assert tuple(
        stage.display_name
        for stage in detailed.stage_profile.stages_for_step("right_dip_ink")
    ) == ("靠近蘸水", "蘸水", "离开蘸水")
    assert tuple(
        stage.stage_id
        for stage in detailed.stage_profile.stages_for_step("right_write")
    ) == (
        "right_write.pen_down",
        "right_write.write",
        "right_write.pen_up",
    )


def test_three_stage_sample_owns_coarse_and_internal_stage_profiles() -> None:
    bundle = load_managed_task_profile(
        "example_three_stage_pick_place_progress"
    )
    progress = bundle.progress_profile
    refinement = bundle.refinement_profile
    stages = bundle.stage_profile

    assert tuple(step.progress_step_id for step in progress.steps) == (
        "approach_target",
        "acquire_object",
        "place_object",
    )
    assert tuple(step.semantic_type for step in progress.steps) == (
        "APPROACH_TARGET",
        "ACQUIRE_OBJECT",
        "PLACE_OBJECT",
    )
    assert progress.allowed_next_step_ids("approach_target") == (
        "acquire_object",
    )
    assert progress.allowed_next_step_ids("acquire_object") == (
        "place_object",
    )
    assert refinement is not None
    validate_task_refinement_profile(
        progress_profile=progress,
        refinement_profile=refinement,
    )
    assert tuple(
        item.refinement_boundary_id
        for item in refinement.boundaries_for_step("acquire_object")
    ) == (
        "acquire_object.contact_established",
        "acquire_object.lift_verified",
    )
    assert not any(
        item.eligible_as_exclusion_anchor
        for item in refinement.boundaries
    )
    assert stages is not None
    assert stages.task_progress_profile_hash == (
        progress.semantic_content_hash
    )
    assert tuple(
        item.stage_id
        for item in stages.stages_for_step("acquire_object")
    ) == (
        "acquire_object.contact",
        "acquire_object.secure",
    )


def test_catalog_reference_and_file_reference_are_exactly_equivalent() -> None:
    bundle = load_managed_task_profile(
        "dmtac_rm65_bimanual_position_v1_progress"
    )
    from_catalog = resolve_task_progress_profile_reference(
        "catalog:dmtac_rm65_bimanual_position_v1_progress"
    )
    resource_path = (
        Path(__file__).resolve().parents[1]
        / "collection_runtime"
        / "progress"
        / "profiles"
        / "linear"
        / "dmtac_rm65_bimanual_position_v1.progress.json"
    )
    from_file = resolve_task_progress_profile_reference(resource_path)

    assert from_catalog == bundle.progress_profile
    assert from_file.semantic_content_hash == from_catalog.semantic_content_hash


def test_editable_profile_sources_recompute_hashes_and_companion_binding(
    tmp_path: Path,
) -> None:
    source_root = (
        Path(__file__).resolve().parents[1]
        / "collection_runtime"
        / "progress"
        / "profiles"
        / "linear"
    )
    progress_payload = json.loads(
        (
            source_root
            / "dmtac_rm65_bimanual_position_writing_five_stage.progress.json"
        ).read_text(encoding="utf-8")
    )
    declared_progress_hash = progress_payload["semantic_content_hash"]
    progress_payload["steps"][0]["display_name"] += "（可编辑）"
    progress_path = tmp_path / "edited.progress.json"
    progress_path.write_text(
        json.dumps(progress_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    progress = load_task_progress_profile_file(progress_path)

    assert progress.semantic_content_hash != declared_progress_hash

    stage_payload = json.loads(
        (
            source_root
            / "dmtac_rm65_bimanual_position_writing_five_stage.stages.json"
        ).read_text(encoding="utf-8")
    )
    declared_stage_hash = stage_payload["semantic_content_hash"]
    declared_progress_binding = stage_payload["task_progress_profile_hash"]
    stage_path = tmp_path / "edited.stages.json"
    stage_path.write_text(
        json.dumps(stage_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    stages = load_task_stage_profile_file(
        stage_path,
        progress_profile=progress,
    )

    assert stages.task_progress_profile_hash == progress.semantic_content_hash
    assert stages.task_progress_profile_hash != declared_progress_binding
    assert stages.semantic_content_hash != declared_stage_hash


def test_common_semantics_are_optional_cross_profile_vocabulary() -> None:
    acquire = common_progress_semantic_type("acquire_object")

    assert acquire is not None
    assert acquire.semantic_type == "ACQUIRE_OBJECT"
    assert common_progress_semantic_type("TASK_SPECIFIC_CUSTOM_STEP") is None


def test_blank_dataset_selection_does_not_lock_until_bind(
    tmp_path: Path,
) -> None:
    profile = load_managed_task_profile(
        "example_three_stage_pick_place_progress"
    ).progress_profile

    admission = inspect_dataset_progress_profile(
        tmp_path,
        requested_profile=profile,
    )

    assert admission.profile == profile
    assert admission.locked is False
    assert admission.source == "blank_dataset"
    assert read_dataset_progress_profile(tmp_path) is None

    path = bind_dataset_progress_profile(tmp_path, profile)

    assert path == tmp_path / DATASET_PROGRESS_PROFILE_RELATIVE_PATH
    assert read_dataset_progress_profile(tmp_path) == profile
    assert inspect_dataset_progress_profile(tmp_path).locked is True


def test_populated_profileless_dataset_cannot_be_partially_upgraded(
    tmp_path: Path,
) -> None:
    profile = load_managed_task_profile(
        "example_three_stage_pick_place_progress"
    ).progress_profile
    meta = tmp_path / "meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text(
        '{"episode_index":0}\n',
        encoding="utf-8",
    )

    admission = inspect_dataset_progress_profile(tmp_path)

    assert admission.profile is None
    assert admission.locked is True
    assert admission.source == "populated_profileless_dataset"
    with pytest.raises(
        DatasetProgressProfileBindingError,
        match="populated profile-less",
    ):
        bind_dataset_progress_profile(tmp_path, profile)


def test_existing_dataset_rejects_same_id_with_different_hash(
    tmp_path: Path,
) -> None:
    original = load_managed_task_profile(
        "example_three_stage_pick_place_progress"
    ).progress_profile
    changed = type(original)(
        profile_id=original.profile_id,
        profile_version=original.profile_version,
        task_id=original.task_id,
        steps=original.steps,
        initial_step_id=original.initial_step_id,
        terminal_step_ids=original.terminal_step_ids,
        topology_kind=original.topology_kind,
        allow_retry_after_failure=False,
    )
    bind_dataset_progress_profile(tmp_path, original)

    with pytest.raises(
        DatasetProgressProfileBindingError,
        match="differs from the profile already used",
    ):
        inspect_dataset_progress_profile(
            tmp_path,
            requested_profile=changed,
        )


def test_legacy_episode_profiles_backfill_only_when_identical(
    tmp_path: Path,
) -> None:
    profile = load_managed_task_profile(
        "example_three_stage_pick_place_progress"
    ).progress_profile
    for episode_uuid in ("episode-a", "episode-b"):
        publish_immutable_canonical(
            tmp_path
            / "meta"
            / "progress"
            / "episodes"
            / episode_uuid
            / "profile.cjson",
            profile.to_payload(),
            domain=PROGRESS_PROFILE_FILE_DOMAIN_V1,
        )

    admission = inspect_dataset_progress_profile(tmp_path)
    assert admission.profile == profile
    assert admission.source == "legacy_episode_profiles"
    assert not (
        tmp_path / DATASET_PROGRESS_PROFILE_RELATIVE_PATH
    ).exists()

    bind_dataset_progress_profile(tmp_path, profile)
    assert read_dataset_progress_profile(tmp_path) == profile


def test_concurrent_different_profile_binding_admits_exactly_one(
    tmp_path: Path,
) -> None:
    profiles = tuple(
        bundle.progress_profile
        for bundle in list_managed_task_profiles(topology_kind="LINEAR")
    )[:2]
    barrier = threading.Barrier(2)
    accepted: list[str] = []
    rejected: list[Exception] = []
    result_lock = threading.Lock()

    def bind(profile) -> None:
        barrier.wait()
        try:
            bind_dataset_progress_profile(tmp_path, profile)
        except Exception as exc:  # noqa: BLE001 - concurrent rejection is asserted.
            with result_lock:
                rejected.append(exc)
        else:
            with result_lock:
                accepted.append(profile.semantic_content_hash)

    workers = [
        threading.Thread(target=bind, args=(profile,))
        for profile in profiles
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2.0)

    assert all(not worker.is_alive() for worker in workers)
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert isinstance(rejected[0], DatasetProgressProfileBindingError)
    assert (
        read_dataset_progress_profile(tmp_path).semantic_content_hash
        == accepted[0]
    )
