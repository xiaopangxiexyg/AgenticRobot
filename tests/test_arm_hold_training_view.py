from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from dataset_contracts import (
    ARM_HOLD_TRAINING_PROJECTION_ID_V1,
    ARM_MODE_IDS,
    ARM_MODE_SCHEMA_V2,
    DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1,
    ArmModeContractError,
    arm_hold_training_projection_metadata,
    resolve_arm_hold_training_contract,
)
from dataset_interface import (
    ArmHoldProjectionError,
    project_arm_hold_training_view,
    summarize_arm_hold_coverage,
)
from teleop_runtime.collection.arm_mode_labels import (
    ARM_MODE_IDS as COLLECTION_ARM_MODE_IDS,
)
from teleop_runtime.collection.arm_mode_labels import (
    ARM_MODE_SCHEMA as COLLECTION_ARM_MODE_SCHEMA,
)
from teleop_runtime.collection.writer_v21 import (
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
)


def _old_v2_metadata() -> dict[str, Any]:
    return {
        "label_specs": {
            "teleop.arm_mode": {
                "schema": ARM_MODE_SCHEMA_V2,
                "sides": ["left", "right"],
                "mode_ids": dict(ARM_MODE_IDS),
            }
        }
    }


def _metadata_with_projection() -> dict[str, Any]:
    metadata = _old_v2_metadata()
    metadata["label_specs"]["teleop.arm_mode"]["training_projections"] = {
        ARM_HOLD_TRAINING_PROJECTION_ID_V1: (
            arm_hold_training_projection_metadata()
        )
    }
    return metadata


def _columns(
    pairs: list[tuple[str, str]],
    *,
    continuity: list[int] | None = None,
) -> dict[str, Sequence[object]]:
    columns: dict[str, Sequence[object]] = {
        "teleop.arm_mode.schema": [ARM_MODE_SCHEMA_V2] * len(pairs),
        "teleop.arm_mode.available": [1] * len(pairs),
    }
    for side_index, side in enumerate(("left", "right")):
        labels = [pair[side_index] for pair in pairs]
        columns[f"teleop.arm_mode.{side}.label"] = labels
        columns[f"teleop.arm_mode.{side}.id"] = [
            ARM_MODE_IDS[label] for label in labels
        ]
        columns[f"teleop.arm_mode.{side}.scope"] = [
            "arm_and_end_effector" if label == "fixed_hold" else ""
            for label in labels
        ]
    if continuity is not None:
        columns["collection.continuity_segment_index"] = continuity
    return columns


def test_known_old_v2_label_spec_uses_frozen_compatibility_projection() -> None:
    resolved = resolve_arm_hold_training_contract(_old_v2_metadata())

    assert resolved.contract == DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1
    assert resolved.compatibility_mode is True
    assert resolved.source_kind == "implicit_v2_arm_mode_projection"


def test_explicit_projection_is_hash_bound_and_not_compatibility_mode() -> None:
    metadata = _metadata_with_projection()
    resolved = resolve_arm_hold_training_contract(metadata)

    assert resolved.contract == DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1
    assert resolved.compatibility_mode is False
    assert resolved.contract is not None
    assert (
        resolved.contract.semantic_content_hash
        == arm_hold_training_projection_metadata()["contract_hash"]
    )

    payload = metadata["label_specs"]["teleop.arm_mode"][
        "training_projections"
    ][ARM_HOLD_TRAINING_PROJECTION_ID_V1]
    payload["contract_hash"] = "sha256:not-the-contract"
    with pytest.raises(ArmModeContractError, match="hash"):
        resolve_arm_hold_training_contract(metadata)


def test_dataset_without_arm_mode_protocol_remains_action_compatible() -> None:
    resolved = resolve_arm_hold_training_contract({})
    view = project_arm_hold_training_view(
        {"action": [[0.0], [1.0], [2.0]]},
        metadata={},
    )

    assert resolved.contract is None
    assert view.targets.shape == (3, 2)
    assert not view.valid_mask.any()
    assert not view.loss_mask.any()
    assert view.receipt.invalid_reason_counts == {"label_spec_missing": 6}


def test_projection_builds_per_side_targets_and_masks_ignored_and_padding() -> None:
    columns = _columns(
        [
            ("fixed_hold", "teleop"),
            ("teleop", "fixed_hold"),
            ("init_pose", "teleop"),
        ]
    )
    view = project_arm_hold_training_view(
        columns,
        metadata=_old_v2_metadata(),
        target_is_pad=np.array([False, True, False]),
    )

    np.testing.assert_array_equal(
        view.targets,
        np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        view.valid_mask,
        np.array([[1, 1], [1, 1], [0, 1]], dtype=bool),
    )
    np.testing.assert_array_equal(
        view.loss_mask,
        np.array([[1, 1], [0, 0], [0, 1]], dtype=bool),
    )
    assert view.receipt.invalid_reason_counts == {"ignored_mode:init_pose": 1}


def test_transition_mask_respects_continuity_indices_and_padding() -> None:
    columns = _columns(
        [
            ("teleop", "fixed_hold"),
            ("fixed_hold", "teleop"),
            ("teleop", "fixed_hold"),
            ("fixed_hold", "teleop"),
        ],
        continuity=[0, 0, 1, 1],
    )
    view = project_arm_hold_training_view(
        columns,
        metadata=_old_v2_metadata(),
        target_is_pad=np.array([False, False, False, True]),
    )

    np.testing.assert_array_equal(
        view.transition_mask,
        np.array(
            [[0, 0], [1, 1], [0, 0], [0, 0]],
            dtype=bool,
        ),
    )

    skipped = project_arm_hold_training_view(
        columns,
        metadata=_old_v2_metadata(),
        source_indices=[0, 2],
    )
    assert not skipped.transition_mask.any()


def test_malformed_present_labels_mask_or_fail_strictly() -> None:
    columns = _columns([("fixed_hold", "teleop")])
    columns["teleop.arm_mode.left.scope"] = [""]

    view = project_arm_hold_training_view(
        columns,
        metadata=_old_v2_metadata(),
    )
    assert view.valid_mask.tolist() == [[False, True]]
    assert view.receipt.invalid_reason_counts == {"positive_scope_mismatch": 1}

    with pytest.raises(ArmHoldProjectionError, match="positive_scope_mismatch"):
        project_arm_hold_training_view(
            columns,
            metadata=_old_v2_metadata(),
            strict=True,
        )


def test_coverage_reports_complementary_pair_bias() -> None:
    view = project_arm_hold_training_view(
        _columns(
            [
                ("fixed_hold", "teleop"),
                ("teleop", "fixed_hold"),
                ("fixed_hold", "teleop"),
            ]
        ),
        metadata=_old_v2_metadata(),
    )
    report = summarize_arm_hold_coverage(view)

    assert report.pair_counts == {"00": 0, "01": 1, "10": 2, "11": 0}
    assert report.missing_pair_states == ("00", "11")
    assert report.perfectly_complementary is True


def test_collection_constants_and_writer_metadata_use_pure_contract(tmp_path) -> None:
    assert COLLECTION_ARM_MODE_SCHEMA == ARM_MODE_SCHEMA_V2
    assert COLLECTION_ARM_MODE_IDS is ARM_MODE_IDS

    LeRobotV21Writer(
        LeRobotV21WriterConfig(
            root_dir=tmp_path,
            write_video=False,
            write_parquet=False,
        )
    )
    info = json.loads(
        (tmp_path / "meta" / "info.json").read_text(encoding="utf-8")
    )
    projection = info["label_specs"]["teleop.arm_mode"][
        "training_projections"
    ][ARM_HOLD_TRAINING_PROJECTION_ID_V1]
    assert projection == arm_hold_training_projection_metadata()
