from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from teleop_runtime.deploy import debug_payload, run_policy


def test_run_policy_reexports_canonical_debug_payload_helpers() -> None:
    for name in (
        "DEPLOY_DEBUG_PAYLOAD_COMPACT",
        "DEPLOY_DEBUG_PAYLOAD_FULL",
        "DEPLOY_DEBUG_PAYLOAD_MODES",
        "_DEBUG_EFFECTOR_METADATA_KEYS",
        "_DEBUG_LAST_COMMAND_METADATA_KEYS",
        "_debug_action_summary",
        "_debug_action_vector",
        "_debug_arm_joint_targets",
        "_debug_arm_state_summary",
        "_debug_args_payload",
        "_debug_backend_arm_states",
        "_debug_backend_dispatch_by_effector_summary",
        "_debug_backend_group_health_summary",
        "_debug_backend_limits",
        "_debug_background_streaming_summary",
        "_debug_compact_value",
        "_debug_command_arm_pose_by_side",
        "_debug_command_summary",
        "_debug_hand_by_side_from_vector",
        "_debug_metadata_summary",
        "_debug_pose_by_side_from_vector",
        "_debug_pose_delta",
        "_debug_pose_delta_by_side",
        "_debug_position_executor_event_summary",
        "_debug_quat_delta_deg",
        "_debug_redact",
        "_debug_remote_policy_summary",
        "_debug_state_pose_by_side",
        "_debug_vector",
        "_debug_xyzquat",
    ):
        assert getattr(run_policy, name) is getattr(debug_payload, name)


def test_debug_args_payload_recursively_redacts_policy_credentials() -> None:
    args = argparse.Namespace(
        policy_kwargs_json=json.dumps(
            {
                "api_key": "top-secret",
                "nested": {"password": "also-secret", "model": "test"},
            }
        ),
        active_sides=("left",),
    )

    payload = debug_payload._debug_args_payload(args)

    assert payload["policy_kwargs_json"] == {
        "api_key": "<redacted>",
        "nested": {"password": "<redacted>", "model": "test"},
    }
    assert payload["active_sides"] == ["left"]


def test_compact_debug_value_bounds_large_arrays_without_changing_small_arrays() -> None:
    small = np.arange(6, dtype=np.float32)
    large = np.arange(65, dtype=np.float32)

    compact_small = debug_payload._debug_compact_value(small)
    compact_large = debug_payload._debug_compact_value(large)

    np.testing.assert_array_equal(compact_small, small)
    assert compact_small is not small
    assert compact_large == {
        "omitted": True,
        "reason": "compact_deploy_debug_array",
        "shape": (65,),
        "dtype": "float32",
        "size": 65,
    }


def test_position_executor_payload_keeps_bounded_diagnostic_identity() -> None:
    payload = debug_payload._debug_position_executor_event_summary(
        {
            "position_collision_debug": {
                "q_ik_raw_by_side": {"left": [0.0] * 6},
                "q_ref_by_side": {"left": [0.1] * 6},
                "large_internal_detail": list(range(100)),
            },
            "position_arm_backend_group_health": {
                "ready": True,
                "sides": {
                    "left": {
                        "ready": True,
                        "hardware_ready": True,
                        "unbounded_internal_detail": list(range(100)),
                    }
                },
            },
        }
    )

    collision = payload["position_collision_debug"]
    assert collision["omitted"] is True
    assert collision["reason"] == "compact_deploy_debug_redundant_detail"
    assert collision["q_ref_by_side"] == {"left": [0.1] * 6}
    assert payload["position_arm_backend_group_health"] == {
        "ready": True,
        "sides": {"left": {"ready": True, "hardware_ready": True}},
    }


def test_compact_position_event_bounds_ik_detail_but_keeps_exact_certificate() -> None:
    certificate = {
        "sequence": 12,
        "qref_fingerprint_by_side": {"left": "left-qref", "right": "right-qref"},
        "singularity_aware_wrist_traversal_by_side": {
            "right": {"mode": "crossing", "certificate": {"accepted": True}},
        },
    }
    nested = {"by_side": {"right": {"candidate_detail": list(range(1000))}}}
    selection = {
        "candidate_count": 4,
        "collision_checked_count": 4,
        "selected": True,
        "selected_source": "coupled_pink",
        "selected_score": 0.25,
        "candidate_eval_elapsed_ms": 8.0,
        "selected_metadata": nested,
        "candidate_certificate": certificate,
    }
    payload = {"ik_candidate_selection": selection, "position_reference_sequence": 19}

    summary = debug_payload._debug_position_executor_event_summary(payload)

    compact = summary["ik_candidate_selection"]
    assert compact is not selection
    assert compact["candidate_certificate"] is certificate
    assert compact["selected_metadata"]["by_side"]["right"] == {
        "omitted": True,
        "reason": "compact_deploy_debug_depth",
        "type": "mapping",
        "size": 1,
    }
    for key in ("candidate_count", "collision_checked_count", "selected", "selected_source",
                "selected_score", "candidate_eval_elapsed_ms"):
        assert compact[key] == selection[key]
    assert summary["position_reference_sequence"] == 19
    assert selection["selected_metadata"] is nested
    assert nested["by_side"]["right"]["candidate_detail"] == list(range(1000))


def test_compact_position_event_preserves_absent_or_scalar_ik_diagnostic() -> None:
    assert debug_payload._debug_position_executor_event_summary({"event": "test"}) == {"event": "test"}
    assert debug_payload._debug_position_executor_event_summary({"ik_candidate_selection": None}) == {
        "ik_candidate_selection": None,
    }


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_position_ik_debug_projection_is_confined_to_compact_sink(tmp_path, mode) -> None:
    from teleop_runtime.deploy.debug_logger import DeploymentDebugLogger

    selection = {
        "selected": True,
        "candidate_certificate": {"sequence": 7, "qref_fingerprint_by_side": {"right": "qref"}},
        "selected_metadata": {"by_side": {"right": {"detail": list(range(1000))}}},
    }
    logger = DeploymentDebugLogger(enabled=True, root_dir=tmp_path, active_sides=("right",),
                                   payload_mode=mode)
    try:
        logger.log_event("position_executor_arm_reference_dispatch", {"ik_candidate_selection": selection})
    finally:
        logger.close()
    record = json.loads(logger.path.read_text())
    observed = record["ik_candidate_selection"]
    assert observed["selected"] is True
    assert observed["candidate_certificate"] == selection["candidate_certificate"]
    if mode == "full":
        assert observed == selection
    else:
        assert observed["selected_metadata"]["by_side"]["right"]["omitted"] is True
    assert selection["selected_metadata"]["by_side"]["right"]["detail"] == list(range(1000))
