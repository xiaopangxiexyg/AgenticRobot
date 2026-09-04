from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from dataset_contracts.v22 import EndEffectorSpec, PolicyIOLayout
from deployments.openpi.model_rtc_profile import (
    load_prepared_model_rtc_profile,
    prepared_model_rtc_profile_hash,
)
from deployments.openpi.training.prepared_skill_launch import (
    OPENPI_QUANTILE_SEMANTIC_BOUNDS_V1,
    _apply_openpi_quantile_semantic_bounds_to_stats,
    _load_profile,
    _model_rtc_profile_matches_prepared_profile,
    _openpi_quantile_semantic_bounds_spec,
    _prepared_profile_for_model_rtc,
)


_REPOSITORY = Path(__file__).resolve().parents[1]
_PREPARED_PROFILE = (
    _REPOSITORY
    / "deployments/openpi/configs/training/"
    / "writing_stage2_all_valid_all_view_h32_absolute_non_rtc_100k_v1.json"
)
_MODEL_RTC_PROFILE = (
    _REPOSITORY
    / "deployments/openpi/configs/model_rtc/"
    / "writing_stage2_all_valid_all_view_h32_absolute_prepared_model_rtc_v1.json"
)


@dataclasses.dataclass(frozen=True)
class _NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray


def _continuous_gripper_layout() -> PolicyIOLayout:
    return PolicyIOLayout(
        name="semantic-quantile-bounds-test",
        end_effectors={
            side: EndEffectorSpec.continuous(
                side,
                adapter="dmtac_gripper_v1",
                target_dim=1,
                target_names=("gripper_close",),
                state_names=("gripper_close",),
                units="normalized_0_1",
                normalization={"minimum": 0.0, "maximum": 1.0},
            )
            for side in ("left", "right")
        },
    )


def test_stage2_all_view_h32_model_rtc_profile_matches_prepared_profile() -> None:
    prepared = _load_profile(_PREPARED_PROFILE)
    model_rtc = load_prepared_model_rtc_profile(_MODEL_RTC_PROFILE)

    assert prepared["profile_id"] == (
        "writing_stage2_all_valid_all_view_h32_absolute_non_rtc_100k_v1"
    )
    assert model_rtc["prepared_skill_profile_id"] == prepared["profile_id"]
    assert prepared["camera_selection"] == ["front", "left_wrist", "right_wrist"]
    assert prepared["camera_slot_to_dataset_key"] == {
        "base_0_rgb": "front",
        "left_wrist_0_rgb": "left_wrist",
        "right_wrist_0_rgb": "right_wrist",
    }
    assert prepared["action_horizon"] == 32
    assert prepared["action_pose_mode"] == "absolute"
    assert prepared["openpi_quantile_semantic_bounds"] == {
        "schema": OPENPI_QUANTILE_SEMANTIC_BOUNDS_V1,
        "components": {"actions": ["right.end_effector.target"]},
    }
    assert prepared["expected_total_raw_episode_count"] == 1504
    assert prepared["expected_total_eligible_episode_count"] == 1424
    assert prepared["expected_openpi_quality_valid_window_count"] == 41578

    assert model_rtc["minimum_forecast_delay_steps"] == 1
    assert model_rtc["maximum_forecast_delay_steps"] == 12
    assert model_rtc["minimum_prefix_steps"] == 0
    assert model_rtc["maximum_prefix_steps"] == 12
    assert model_rtc["maximum_prefix_steps"] <= (
        model_rtc["maximum_forecast_delay_steps"]
    ) < prepared["action_horizon"]


def test_stage2_all_view_h32_model_rtc_sampling_is_explicit_and_hashable() -> None:
    model_rtc = load_prepared_model_rtc_profile(_MODEL_RTC_PROFILE)
    sampling = model_rtc["sampling_spec"]

    assert sampling["schema"] == "dexteleop_model_continuation_training_sampling_v1"
    assert sampling["strategy"] == "hash_indexed_uniform_explicit_pairs_v1"
    assert sampling["selection_salt"] == model_rtc["profile_id"].replace(
        "_", "-"
    )
    assert sampling["delay_prefix_pairs"] == [
        [1, 1],
        [2, 1],
        [2, 2],
        [4, 2],
        [4, 4],
        [8, 4],
        [8, 8],
        [12, 8],
        [12, 12],
    ]
    assert prepared_model_rtc_profile_hash(model_rtc).startswith("sha256:")


def test_stage2_model_rtc_profile_has_no_checkpoint_or_mask_rebinding() -> None:
    prepared = _load_profile(_PREPARED_PROFILE)
    payload = json.loads(_MODEL_RTC_PROFILE.read_text(encoding="utf-8"))

    assert "checkpoint_dir" not in payload
    assert "base_checkpoint_dir" not in payload
    assert "continuation_contract_hash" not in payload
    assert "prepared_final_mask_root" not in payload
    assert payload["output_config_name"] != prepared["openpi_config_name"]
    assert _prepared_profile_for_model_rtc(prepared, payload) is prepared


def test_stage2_model_rtc_overlay_requires_exact_prepared_profile_identity() -> None:
    prepared = _load_profile(_PREPARED_PROFILE)
    model_rtc = load_prepared_model_rtc_profile(_MODEL_RTC_PROFILE)

    assert _model_rtc_profile_matches_prepared_profile(prepared, model_rtc)

    mismatched = dict(model_rtc)
    mismatched["prepared_skill_profile_id"] = "another-prepared-profile"
    assert not _model_rtc_profile_matches_prepared_profile(prepared, mismatched)


def test_semantic_quantile_bounds_replace_only_declared_component() -> None:
    layout = _continuous_gripper_layout()
    original_q01 = np.arange(layout.action_dim, dtype=np.float64) / 100.0
    original_q99 = original_q01 + 0.5
    original_q01[-1] = 0.999994
    original_q99[-1] = 0.999994
    original = _NormStats(
        mean=np.zeros(layout.action_dim),
        std=np.ones(layout.action_dim),
        q01=original_q01,
        q99=original_q99,
    )
    profile = {
        "openpi_quantile_semantic_bounds": {
            "schema": OPENPI_QUANTILE_SEMANTIC_BOUNDS_V1,
            "components": {"actions": ["right.end_effector.target"]},
        }
    }

    component_spec = _openpi_quantile_semantic_bounds_spec(profile)
    assert component_spec is not None
    bounded, replacements = _apply_openpi_quantile_semantic_bounds_to_stats(
        component_spec=component_spec,
        layout=layout,
        norm_stats={"actions": original},
    )

    assert original.q01[-1] == 0.999994
    assert original.q99[-1] == 0.999994
    assert np.array_equal(bounded["actions"].q01[:-1], original.q01[:-1])
    assert np.array_equal(bounded["actions"].q99[:-1], original.q99[:-1])
    assert bounded["actions"].q01[-1] == 0.0
    assert bounded["actions"].q99[-1] == 1.0
    assert replacements == [
        {
            "vector": "actions",
            "component": "right.end_effector.target",
            "offset": 15,
            "stop": 16,
            "layout_units": "normalized_0_1",
            "before_q01": [0.999994],
            "before_q99": [0.999994],
            "semantic_q01": [0.0],
            "semantic_q99": [1.0],
        }
    ]


def test_semantic_quantile_bounds_reject_non_end_effector_component() -> None:
    layout = _continuous_gripper_layout()
    stats = _NormStats(
        mean=np.zeros(layout.action_dim),
        std=np.ones(layout.action_dim),
        q01=np.zeros(layout.action_dim),
        q99=np.ones(layout.action_dim),
    )

    try:
        _apply_openpi_quantile_semantic_bounds_to_stats(
            component_spec={"actions": ("right.arm.pose",)},
            layout=layout,
            norm_stats={"actions": stats},
        )
    except ValueError as exc:
        assert "end-effector components" in str(exc)
    else:
        raise AssertionError("arm pose component unexpectedly accepted")
