from __future__ import annotations

import json
from pathlib import Path

from utils.dataset_postprocess import camera_semantic_promotion as promotion


def _plan(source: Path, corrected: Path, retirement: Path) -> promotion.CameraSemanticPromotionPlan:
    return promotion.CameraSemanticPromotionPlan(
        source_root=str(source),
        corrected_root=str(corrected),
        retirement_root=str(retirement),
        retired_dataset_root=str(retirement / "dataset.swapped-receipt"),
        registry_path=str(source.parent / "registry.json"),
        binding_hash="binding",
        materialization_receipt_hash="receipt",
        materialization_plan_hash="materialization-plan",
        tactile_inventory_hash="tactile",
        source_inventory_hash="source-inventory",
        corrected_inventory_hash="corrected-inventory",
        source_root_device=int(source.stat().st_dev),
        source_root_inode=int(source.stat().st_ino),
        corrected_root_device=int(corrected.stat().st_dev),
        corrected_root_inode=int(corrected.stat().st_ino),
        source_file_count=1,
        corrected_file_count=1,
        source_hardlinked_file_count=0,
        corrected_hardlinked_file_count=0,
        shared_file_inode_count=0,
        episode_count=1,
        frame_count=3,
        source_video_count=2,
        corrected_video_count=2,
        source_decoded_video_frames=6,
        corrected_decoded_video_frames=6,
    )


def test_atomic_promotion_and_explicit_rollback_preserve_both_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "dataset"
    corrected = tmp_path / "corrected" / "dataset"
    retirement = tmp_path / "retired"
    source.mkdir()
    corrected.mkdir(parents=True)
    (source / "marker.txt").write_text("old-swapped", encoding="utf-8")
    (corrected / "marker.txt").write_text("new-canonical", encoding="utf-8")
    plan = _plan(source, corrected, retirement)
    monkeypatch.setattr(
        promotion,
        "plan_camera_semantic_promotion",
        lambda *_args, **_kwargs: plan,
    )

    applied = promotion.promote_camera_semantic_dataset(
        source,
        corrected,
        retirement,
        plan.registry_path,
        expected_plan_hash=plan.plan_hash,
        confirm_no_active_readers=True,
    )

    retired = Path(plan.retired_dataset_root)
    transaction = Path(applied["transaction_path"])
    assert (source / "marker.txt").read_text(encoding="utf-8") == "new-canonical"
    assert (retired / "marker.txt").read_text(encoding="utf-8") == "old-swapped"
    assert not corrected.exists()
    assert json.loads(transaction.read_text(encoding="utf-8"))["state"] == "completed"

    rollback_plan = promotion.plan_camera_semantic_promotion_rollback(transaction)
    rolled_back = promotion.rollback_camera_semantic_promotion(
        transaction,
        expected_plan_hash=rollback_plan.plan_hash,
        confirm_no_active_readers=True,
    )

    assert rolled_back["state"] == "rolled_back"
    assert (source / "marker.txt").read_text(encoding="utf-8") == "old-swapped"
    assert (corrected / "marker.txt").read_text(encoding="utf-8") == "new-canonical"
    assert not retired.exists()
    assert json.loads(transaction.read_text(encoding="utf-8"))["state"] == "rolled_back"


def test_apply_requires_exact_plan_and_reader_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "dataset"
    corrected = tmp_path / "corrected"
    retirement = tmp_path / "retired"
    source.mkdir()
    corrected.mkdir()
    plan = _plan(source, corrected, retirement)
    monkeypatch.setattr(
        promotion,
        "plan_camera_semantic_promotion",
        lambda *_args, **_kwargs: plan,
    )

    try:
        promotion.promote_camera_semantic_dataset(
            source,
            corrected,
            retirement,
            plan.registry_path,
            expected_plan_hash="wrong",
            confirm_no_active_readers=True,
        )
    except promotion.CameraSemanticPromotionError as exc:
        assert "plan hash changed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("stale plan was accepted")

    try:
        promotion.promote_camera_semantic_dataset(
            source,
            corrected,
            retirement,
            plan.registry_path,
            expected_plan_hash=plan.plan_hash,
            confirm_no_active_readers=False,
        )
    except promotion.CameraSemanticPromotionError as exc:
        assert "no active readers" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("reader confirmation was not required")

    assert source.is_dir()
    assert corrected.is_dir()
    assert not retirement.exists()
