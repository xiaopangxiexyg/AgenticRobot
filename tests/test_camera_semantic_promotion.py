from __future__ import annotations

import json
from dataclasses import replace
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
        tactile_detachment_scope="whole_prefix",
        source_tactile_status="SELF_CONTAINED",
        corrected_tactile_status="SELF_CONTAINED",
        dataset_validation_mode=promotion.NATIVE_V23_VALIDATION_MODE,
        source_dataset_format="dexteleop_v2.3",
        corrected_dataset_format="dexteleop_v2.3",
        legacy_format_preserving_contract_hash="",
        source_inventory_hash="source-inventory",
        corrected_inventory_hash="corrected-inventory",
        source_rollback_guard_hash=promotion._rollback_guard_inventory(source),
        corrected_rollback_guard_hash=promotion._rollback_guard_inventory(corrected),
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


def test_modality_pair_accepts_detached_source_and_self_contained_corrected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    corrected = tmp_path / "corrected"
    source.mkdir()
    corrected.mkdir()
    statuses = {
        source: {
            "status": "DETACHED_VERIFIED",
            "inventory_hash": "inventory",
            "detachment_scope": "whole_prefix",
            "pending_transaction": False,
            "production_detached_verified": True,
            "tactile_read_eligible": False,
        },
        corrected: {
            "status": "SELF_CONTAINED",
            "inventory_hash": "inventory",
            # A completed rehydrate may report its default self-contained
            # scope; only the detached source scope drives core validation.
            "detachment_scope": "payload_only",
            "pending_transaction": False,
            "production_detached_verified": False,
            "tactile_read_eligible": True,
        },
    }
    monkeypatch.setattr(
        promotion,
        "modality_storage_status",
        lambda root: statuses[Path(root)],
    )

    assert promotion._require_modality_pair(source, corrected) == (
        "inventory",
        "whole_prefix",
        "DETACHED_VERIFIED",
        "SELF_CONTAINED",
    )


def test_modality_pair_accepts_matching_detached_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    corrected = tmp_path / "corrected"
    source.mkdir()
    corrected.mkdir()
    detached = {
        "status": "DETACHED_VERIFIED",
        "inventory_hash": "inventory",
        "detachment_scope": "whole_prefix",
        "pending_transaction": False,
        "production_detached_verified": True,
        "tactile_read_eligible": False,
    }
    monkeypatch.setattr(promotion, "modality_storage_status", lambda _root: detached)

    assert promotion._require_modality_pair(source, corrected) == (
        "inventory",
        "whole_prefix",
        "DETACHED_VERIFIED",
        "DETACHED_VERIFIED",
    )


def test_modality_pair_rejects_self_contained_source_and_detached_corrected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    corrected = tmp_path / "corrected"
    source.mkdir()
    corrected.mkdir()
    statuses = {
        source: {
            "status": "SELF_CONTAINED",
            "inventory_hash": "inventory",
            "detachment_scope": "payload_only",
            "pending_transaction": False,
            "production_detached_verified": False,
            "tactile_read_eligible": True,
        },
        corrected: {
            "status": "DETACHED_VERIFIED",
            "inventory_hash": "inventory",
            "detachment_scope": "whole_prefix",
            "pending_transaction": False,
            "production_detached_verified": True,
            "tactile_read_eligible": False,
        },
    }
    monkeypatch.setattr(
        promotion,
        "modality_storage_status",
        lambda root: statuses[Path(root)],
    )

    try:
        promotion._require_modality_pair(source, corrected)
    except promotion.CameraSemanticPromotionError as exc:
        assert "refuses to replace self-contained tactile" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("hot-to-detached promotion was accepted")


def test_modality_pair_rejects_unverified_or_scope_mismatched_detached_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    corrected = tmp_path / "corrected"
    source.mkdir()
    corrected.mkdir()
    statuses = {
        source: {
            "status": "DETACHED_VERIFIED",
            "inventory_hash": "inventory",
            "detachment_scope": "whole_prefix",
            "pending_transaction": False,
            "production_detached_verified": True,
            "tactile_read_eligible": False,
        },
        corrected: {
            "status": "DETACHED_VERIFIED",
            "inventory_hash": "inventory",
            "detachment_scope": "whole_prefix",
            "pending_transaction": False,
            "production_detached_verified": False,
            "tactile_read_eligible": False,
        },
    }
    monkeypatch.setattr(
        promotion,
        "modality_storage_status",
        lambda root: statuses[Path(root)],
    )

    try:
        promotion._require_modality_pair(source, corrected)
    except promotion.CameraSemanticPromotionError as exc:
        assert "corrected tactile detach lacks production retirement evidence" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unverified detached corrected dataset was accepted")

    statuses[corrected] = {
        **statuses[corrected],
        "production_detached_verified": True,
        "detachment_scope": "payload_only",
    }
    try:
        promotion._require_modality_pair(source, corrected)
    except promotion.CameraSemanticPromotionError as exc:
        assert "different tactile detachment scopes" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("scope-mismatched detached pair was accepted")


def test_detached_pair_promotion_is_immediately_rollback_eligible(
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
    plan = replace(
        _plan(source, corrected, retirement),
        source_tactile_status="DETACHED_VERIFIED",
        corrected_tactile_status="DETACHED_VERIFIED",
    )
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
    statuses = {
        source: {
            "status": "DETACHED_VERIFIED",
            "inventory_hash": "tactile",
            "pending_transaction": False,
            "production_detached_verified": True,
        },
        retired: {
            "status": "DETACHED_VERIFIED",
            "inventory_hash": "tactile",
            "pending_transaction": False,
            "production_detached_verified": True,
        },
    }
    monkeypatch.setattr(
        promotion,
        "modality_storage_status",
        lambda root: statuses[Path(root)],
    )

    assert applied["rollback_requires_canonical_detach"] is False
    rollback_plan = promotion.plan_camera_semantic_promotion_rollback(
        applied["transaction_path"]
    )
    rolled_back = promotion.rollback_camera_semantic_promotion(
        applied["transaction_path"],
        expected_plan_hash=rollback_plan.plan_hash,
        confirm_no_active_readers=True,
    )
    assert rolled_back["state"] == "rolled_back"
    assert (source / "marker.txt").read_text(encoding="utf-8") == "old-swapped"
    assert (corrected / "marker.txt").read_text(encoding="utf-8") == "new-canonical"


def test_detached_source_rollback_requires_detached_promoted_root(
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
    plan = replace(
        _plan(source, corrected, retirement),
        source_tactile_status="DETACHED_VERIFIED",
    )
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
    transaction = Path(applied["transaction_path"])
    retired = Path(plan.retired_dataset_root)
    statuses = {
        source: {
            "status": "SELF_CONTAINED",
            "inventory_hash": "tactile",
            "pending_transaction": False,
            "production_detached_verified": False,
        },
        retired: {
            "status": "DETACHED_VERIFIED",
            "inventory_hash": "tactile",
            "pending_transaction": False,
            "production_detached_verified": True,
        },
    }
    monkeypatch.setattr(
        promotion,
        "modality_storage_status",
        lambda root: statuses[Path(root)],
    )
    try:
        promotion.plan_camera_semantic_promotion_rollback(transaction)
    except promotion.CameraSemanticPromotionError as exc:
        assert "canonical tactile must be DETACHED_VERIFIED" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("rollback accepted a hot promoted tactile tree")

    receipt = source / "meta/modality_artifacts/tactile/receipts/new-cycle.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    statuses[source] = {
        "status": "DETACHED_VERIFIED",
        "inventory_hash": "tactile",
        "pending_transaction": False,
        "production_detached_verified": True,
    }
    rollback_plan = promotion.plan_camera_semantic_promotion_rollback(transaction)
    assert rollback_plan.canonical_root == str(source)


def test_rollback_rejects_non_tactile_core_drift(
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
    (source / "marker.txt").write_text("changed", encoding="utf-8")

    try:
        promotion.plan_camera_semantic_promotion_rollback(applied["transaction_path"])
    except promotion.CameraSemanticPromotionError as exc:
        assert "Canonical core files changed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("rollback accepted changed canonical content")
