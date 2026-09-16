from __future__ import annotations

import hashlib
import json
from pathlib import Path

from deployments.openpi.model_rtc_profile import load_prepared_model_rtc_profile
from deployments.openpi.training.prepared_skill_launch import (
    _load_profile,
    _model_rtc_profile_matches_prepared_profile,
    _prepared_profile_for_model_rtc,
)

_ROOT = Path(__file__).resolve().parents[1]
_BASE_PROFILE = (
    _ROOT
    / "deployments/openpi/configs/training/"
    "writing_stage3_all_valid_all_view_h32_absolute_non_rtc_100k_by_date_v1.json"
)
_HISTORY_PROFILE = (
    _ROOT
    / "deployments/openpi/configs/training/"
    "writing_stage3_all_valid_all_view_h32_absolute_state_history_non_rtc_100k_by_date_v1.json"
)
_MODEL_RTC_PROFILE = (
    _ROOT
    / "deployments/openpi/configs/model_rtc/"
    "writing_stage3_all_valid_all_view_h32_absolute_state_history_prepared_model_rtc_v1.json"
)


def test_stage3_history_profile_changes_only_model_identity_and_history_contract() -> None:
    base = _load_profile(_BASE_PROFILE)
    history = _load_profile(_HISTORY_PROFILE)

    for key in (
        "organized_dataset_root",
        "sources",
        "annotation_selection",
        "source_snapshot",
        "camera_selection",
        "camera_slot_to_dataset_key",
        "action_horizon",
        "action_pose_mode",
        "tail_policy",
        "openpi_final_mask_root",
        "openpi_final_mask_inventory_receipt",
        "openpi_final_mask_inventory_receipt_sha256",
        "expected_total_eligible_episode_count",
        "expected_openpi_quality_valid_window_count",
        "expected_openpi_repeated_epoch_sample_count",
    ):
        assert history[key] == base[key]

    assert history["openpi_base_config"] == (
        "writing-all-view-h32-absolute-state-history-4-8-16-v1"
    )
    assert history["openpi_num_train_steps"] == 100_000
    assert history["openpi_training_mode"] == "standard_non_rtc_v1"
    assert history["history_state_input"] == {
        "schema": "dexteleop_history_state_input_v2",
        "input_key": "history_states",
        "age_input_key": "history_state_ages_sec",
        "frame_offsets": [4, 8, 16],
        "offset_semantics": "causal_control_frame_offset",
        "state_dim": 28,
        "state_layout": [
            "left_arm_qpos_6",
            "right_arm_qpos_6",
            "left_tcp_xyzquat_7",
            "right_tcp_xyzquat_7",
            "left_gripper_opening_1",
            "right_gripper_opening_1",
        ],
        "age_unit": "seconds",
        "selection": "same_episode_same_continuity_segment_exact_past_frame",
        "token_placement": "pi0_action_expert_prefix",
        "qpos_normalization": "divide_by_pi",
        "policy_state_normalization": "reuse_state_norm_stats",
    }
    assert "contract_hash" not in history["history_state_input"]
    assert "checkpoint_step" not in history["history_state_input"]


def test_stage3_history_profile_pins_reproducible_openpi_source_without_model_binding() -> None:
    profile = _load_profile(_HISTORY_PROFILE)
    identity = {
        "schema": "dexteleop.openpi_source_identity.v1",
        "revision": "060d1fb86d1fd4acc87584e31d548c0518cae1e5",
        "workspace_source_roots": ["src", "packages/openpi-client/src"],
        "training_config_sha256": (
            "c148b3b10dbe9749170fac449c5acb83ae532590d28efc78714b0ca180260cc7"
        ),
        "openpi_client_history_state_contract_sha256": (
            "39fde45e6c744f0f7fd42b88b9c86f0c288b27f6ad431bcf47c9ca470e800028"
        ),
    }
    identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode()

    assert profile["openpi_source_revision"] == identity["revision"]
    assert profile["openpi_source_identity_sha256"] == hashlib.sha256(
        identity_bytes
    ).hexdigest()
    assert profile["runtime_derivation"]["code_openpi"] == identity["revision"]
    assert profile["runtime_derivation"]["code_agenticrobot"] == (
        "0337360c4daf7b8308570e77f3591634523321b3"
    )
    assert not any(
        "checkpoint" in key
        for key in profile["history_state_input"]
    )


def test_stage3_history_model_rtc_overlay_preserves_history_base_config() -> None:
    prepared = _load_profile(_HISTORY_PROFILE)
    model_rtc = load_prepared_model_rtc_profile(_MODEL_RTC_PROFILE)

    assert _model_rtc_profile_matches_prepared_profile(prepared, model_rtc)
    effective = _prepared_profile_for_model_rtc(prepared, model_rtc)
    assert effective["openpi_base_config"] == prepared["openpi_base_config"]
    assert model_rtc["training_implementation"] == "mask_aware_hard_flow_prefix_v2"
    assert model_rtc["rtc_prefix_mask_conditioning"] is True
    assert model_rtc["sampling_spec"]["delay_prefix_pairs"] == [
        [1, 0],
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
    assert "checkpoint_dir" not in model_rtc
    assert "checkpoint_step" not in model_rtc
    assert "continuation_contract_hash" not in model_rtc
