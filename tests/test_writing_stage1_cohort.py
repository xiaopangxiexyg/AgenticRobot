from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dataset_interface.writing_stage1_cohort import (
    WritingStage1CohortError,
    build_writing_stage1_cohort_plan,
    select_writing_stage1_pilot,
)
from dataset_interface.writing_stage1_v23 import (
    WritingStage1V23LayerError,
    resolve_writing_stage1_v23_campaign_batches,
)


_ROOT_0724 = "/datasets/2026-07-24_vive_rm65_real_teleop_v23"
_ROOT_0712 = "/datasets/2026-07-12_vive_rm65_real_teleop_v23_clean"


def _row(
    identity_number: int,
    *,
    view_profile: str,
    source_index: int,
    frame_count: int,
    provenance_root: str,
) -> dict[str, object]:
    three_view = view_profile == "three_view_ready"
    return {
        "schema": "dexteleop.logical_episode_ref.v1",
        "episode_identity": f"sha256:{identity_number:064x}",
        "task": "writing",
        "view_profile": view_profile,
        "camera_slots": (
            ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
            if three_view
            else ["base_0_rgb"]
        ),
        "frame_count": frame_count,
        "media_integrity": {
            "status": "valid",
            "checks": [
                "declared_frame_count",
                "decoder_frame_count",
                "tail_decode",
            ],
            "camera_keys": (
                ["front", "left_wrist", "right_wrist"]
                if three_view
                else ["front"]
            ),
        },
        "source_dataset_root": f"/logical/{view_profile}",
        "source_episode_index": source_index,
        "source_provenance": {
            "source_dataset_root": provenance_root,
            "source_episode_index": source_index,
            "frame_count": frame_count,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path):
    front_path = tmp_path / "front.jsonl"
    three_path = tmp_path / "three.jsonl"
    front_rows = [
        _row(
            index + 1,
            view_profile="front_only",
            source_index=index,
            frame_count=value,
            provenance_root="/datasets/2026-07-23",
        )
        for index, value in enumerate((100, 200, 300))
    ]
    three_rows = [
        _row(
            index + 100,
            view_profile="three_view_ready",
            source_index=index,
            frame_count=value,
            provenance_root=(
                _ROOT_0724 if index in {1, 3} else _ROOT_0712
            ),
        )
        for index, value in enumerate((80, 120, 160, 240, 320))
    ]
    _write_jsonl(front_path, front_rows)
    _write_jsonl(three_path, three_rows)
    return front_path, three_path, front_rows, three_rows


def test_cohort_filters_other_provenance_and_preserves_view_capabilities(
    tmp_path: Path,
) -> None:
    front_path, three_path, _, _ = _fixture(tmp_path)
    plan = build_writing_stage1_cohort_plan(
        front_manifest_path=front_path,
        three_view_manifest_path=three_path,
        three_view_required_provenance_root=_ROOT_0724,
        expected_front_count=3,
        expected_three_view_manifest_count=5,
        expected_three_view_0724_count=2,
        expected_total_count=5,
    )

    assert plan.counts == {
        "front_0723": 3,
        "three_view_0724": 2,
        "front_training_candidates": 5,
        "three_view_training_candidates": 2,
        "excluded_three_view_other_provenance": 3,
    }
    assert all(not item.three_view_valid for item in plan.front_0723)
    assert all(item.three_view_valid for item in plan.three_view_0724)
    assert {
        item.provenance_dataset_root for item in plan.three_view_0724
    } == {_ROOT_0724}
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    relocated_front = relocated / front_path.name
    relocated_three = relocated / three_path.name
    relocated_front.write_bytes(front_path.read_bytes())
    relocated_three.write_bytes(three_path.read_bytes())
    assert plan.semantic_content_hash == build_writing_stage1_cohort_plan(
        front_manifest_path=relocated_front,
        three_view_manifest_path=relocated_three,
        three_view_required_provenance_root=_ROOT_0724,
        expected_front_count=3,
        expected_three_view_manifest_count=5,
        expected_three_view_0724_count=2,
        expected_total_count=5,
    ).semantic_content_hash


def test_pilot_selection_covers_short_and_long_in_each_batch(
    tmp_path: Path,
) -> None:
    front_path, three_path, _, _ = _fixture(tmp_path)
    plan = build_writing_stage1_cohort_plan(
        front_manifest_path=front_path,
        three_view_manifest_path=three_path,
        three_view_required_provenance_root=_ROOT_0724,
        expected_front_count=3,
        expected_three_view_manifest_count=5,
        expected_three_view_0724_count=2,
        expected_total_count=5,
    )

    pilot = select_writing_stage1_pilot(plan, per_batch=2)
    assert tuple(item.batch_id for item in pilot) == (
        "2026-07-23",
        "2026-07-23",
        "2026-07-24",
        "2026-07-24",
    )
    assert tuple(item.frame_count for item in pilot) == (100, 300, 120, 240)


def test_cohort_fails_closed_on_identity_overlap(tmp_path: Path) -> None:
    front_path, three_path, front_rows, three_rows = _fixture(tmp_path)
    three_rows[1]["episode_identity"] = front_rows[0]["episode_identity"]
    _write_jsonl(three_path, three_rows)

    with pytest.raises(WritingStage1CohortError, match="duplicate Episode"):
        build_writing_stage1_cohort_plan(
            front_manifest_path=front_path,
            three_view_manifest_path=three_path,
            three_view_required_provenance_root=_ROOT_0724,
            expected_front_count=3,
            expected_three_view_manifest_count=5,
            expected_three_view_0724_count=2,
            expected_total_count=5,
        )


def test_dataset_local_batches_require_two_distinct_single_roots(
    tmp_path: Path,
) -> None:
    front_path, three_path, _, _ = _fixture(tmp_path)
    plan = build_writing_stage1_cohort_plan(
        front_manifest_path=front_path,
        three_view_manifest_path=three_path,
        three_view_required_provenance_root=_ROOT_0724,
        expected_front_count=3,
        expected_three_view_manifest_count=5,
        expected_three_view_0724_count=2,
        expected_total_count=5,
    )
    pilot = select_writing_stage1_pilot(plan, per_batch=1)
    batches = resolve_writing_stage1_v23_campaign_batches(
        plan,
        pilot,
        expected_per_batch=1,
    )
    assert tuple(len(value.eligible) for value in batches) == (3, 2)
    assert tuple(len(value.selected) for value in batches) == (1, 1)
    assert batches[0].source_dataset_root != batches[1].source_dataset_root

    shared_root = plan.front_0723[0].source_dataset_root
    merged_three = tuple(
        replace(value, source_dataset_root=shared_root)
        for value in plan.three_view_0724
    )
    merged_plan = replace(plan, three_view_0724=merged_three)
    merged_pilot = (
        pilot[0],
        replace(pilot[1], source_dataset_root=shared_root),
    )
    with pytest.raises(WritingStage1V23LayerError, match="two distinct"):
        resolve_writing_stage1_v23_campaign_batches(
            merged_plan,
            merged_pilot,
            expected_per_batch=1,
        )


def test_cohort_requires_explicit_decoder_and_tail_evidence(
    tmp_path: Path,
) -> None:
    front_path, three_path, front_rows, _ = _fixture(tmp_path)
    media = dict(front_rows[0]["media_integrity"])
    media["checks"] = ["declared_frame_count"]
    front_rows[0]["media_integrity"] = media
    _write_jsonl(front_path, front_rows)
    with pytest.raises(WritingStage1CohortError, match="tail-decode"):
        build_writing_stage1_cohort_plan(
            front_manifest_path=front_path,
            three_view_manifest_path=three_path,
            three_view_required_provenance_root=_ROOT_0724,
            expected_front_count=3,
            expected_three_view_manifest_count=5,
            expected_three_view_0724_count=2,
            expected_total_count=5,
        )


def test_cohort_fails_closed_on_media_invalid_row(tmp_path: Path) -> None:
    front_path, three_path, front_rows, _ = _fixture(tmp_path)
    front_rows[0]["media_integrity"] = {"status": "invalid"}
    _write_jsonl(front_path, front_rows)

    with pytest.raises(WritingStage1CohortError, match="valid media"):
        build_writing_stage1_cohort_plan(
            front_manifest_path=front_path,
            three_view_manifest_path=three_path,
            three_view_required_provenance_root=_ROOT_0724,
            expected_front_count=3,
            expected_three_view_manifest_count=5,
            expected_three_view_0724_count=2,
            expected_total_count=5,
        )
