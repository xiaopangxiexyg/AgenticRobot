from __future__ import annotations

import argparse
import json

import numpy as np

from teleop_runtime.deploy import debug_payload, run_policy


def test_run_policy_reexports_canonical_debug_payload_helpers() -> None:
    for name in (
        "DEPLOY_DEBUG_PAYLOAD_COMPACT",
        "DEPLOY_DEBUG_PAYLOAD_FULL",
        "DEPLOY_DEBUG_PAYLOAD_MODES",
        "_DEBUG_EFFECTOR_METADATA_KEYS",
        "_DEBUG_LAST_COMMAND_METADATA_KEYS",
        "_debug_args_payload",
        "_debug_backend_dispatch_by_effector_summary",
        "_debug_backend_group_health_summary",
        "_debug_background_streaming_summary",
        "_debug_compact_value",
        "_debug_metadata_summary",
        "_debug_position_executor_event_summary",
        "_debug_redact",
        "_debug_remote_policy_summary",
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
