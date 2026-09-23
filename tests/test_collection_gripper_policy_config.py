from __future__ import annotations

import argparse
import json

import pytest

from teleop_runtime.collection import gripper_policy_config, run_collect


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "gripper_task_profile": "",
        "gripper_task": "",
        "task_id": "",
        "task_name": "",
        "task": "",
        "gripper_source_dim": None,
        "gripper_output_dim": None,
        "gripper_invert": None,
        "gripper_alpha": None,
        "gripper_input_min": None,
        "gripper_input_max": None,
        "gripper_speed": None,
        "gripper_torque": None,
        "gripper_max_action": None,
        "gripper_send_threshold": None,
        "gripper_command_hz": None,
        "gripper_continuous_grasp": None,
        "gripper_telemetry_hz": None,
        "gripper_telemetry_command_grace_s": None,
        "dmtac_tactile": None,
        "with_tactile": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_collect_preserves_gripper_policy_compatibility_exports() -> None:
    assert run_collect.GRIPPER_POLICY_DEFAULTS is gripper_policy_config.GRIPPER_POLICY_DEFAULTS
    assert (
        run_collect._load_gripper_task_profile
        is gripper_policy_config.load_gripper_task_profile
    )
    assert run_collect._profile_value is gripper_policy_config.profile_value
    assert (
        run_collect._resolved_gripper_policy
        is gripper_policy_config.resolved_gripper_policy
    )
    assert (
        run_collect._dmtac_tactile_enabled
        is gripper_policy_config.dmtac_tactile_enabled
    )


def test_default_policy_is_cached_as_an_independent_copy() -> None:
    args = _args()

    first = gripper_policy_config.resolved_gripper_policy(args)
    first["speed"] = 0.9
    second = gripper_policy_config.resolved_gripper_policy(args)

    assert second == gripper_policy_config.GRIPPER_POLICY_DEFAULTS
    assert second is not args._resolved_gripper_policy
    assert second["speed"] == 0.2


def test_task_profile_precedes_cli_defaults_and_clamps_action_space(tmp_path) -> None:
    profile_path = tmp_path / "gripper_tasks.json"
    profile_path.write_text(
        json.dumps(
            {
                "gripper": {"force": 0.20, "speed": 0.20, "max_action": 0.77},
                "tasks": {
                    "soft_grasp": {
                        "gripper": {
                            "force": 0.17,
                            "speed": 0.09,
                            "max_close": 0.55,
                            "continuous": True,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = _args(
        gripper_task_profile=str(profile_path),
        task_id="soft_grasp",
    )

    policy = gripper_policy_config.resolved_gripper_policy(args)

    assert args._resolved_gripper_task == "soft_grasp"
    assert policy["torque"] == pytest.approx(0.17)
    assert policy["speed"] == pytest.approx(0.09)
    assert policy["max_action"] == pytest.approx(1.0)
    assert policy["continuous_grasp"] is True


def test_collection_profile_root_must_remain_a_json_object(tmp_path) -> None:
    profile_path = tmp_path / "gripper_tasks.json"
    profile_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        gripper_policy_config.load_gripper_task_profile(
            _args(gripper_task_profile=str(profile_path))
        )


def test_cli_overrides_are_normalized_and_tactile_fields_are_canonical() -> None:
    args = _args(
        gripper_source_dim=3,
        gripper_invert=True,
        gripper_alpha=0.75,
        gripper_speed=0.12,
        gripper_torque=0.34,
        gripper_send_threshold=0.01,
        gripper_command_hz=12.0,
        gripper_continuous_grasp=False,
        gripper_telemetry_hz=20.0,
        gripper_telemetry_command_grace_s=0.3,
    )

    policy = gripper_policy_config.resolved_gripper_policy(args)

    assert policy["source_dim"] == 3
    assert policy["invert"] is True
    assert policy["alpha"] == pytest.approx(0.75)
    assert policy["speed"] == pytest.approx(0.12)
    assert policy["torque"] == pytest.approx(0.34)
    assert policy["send_threshold"] == pytest.approx(0.01)
    assert policy["command_hz"] == pytest.approx(12.0)
    assert policy["continuous_grasp"] is False
    assert policy["telemetry_hz"] == pytest.approx(20.0)
    assert policy["telemetry_command_grace_s"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("gripper_source_dim", 6, r"source_dim must be in \[0, 5\]"),
        ("gripper_output_dim", 1, r"output_dim/gripper_slot must be 0"),
        ("gripper_alpha", 0.0, r"alpha must be in \(0, 1\]"),
        ("gripper_command_hz", 0.0, "command_hz must be > 0"),
    ),
)
def test_policy_rejects_invalid_cli_ranges(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gripper_policy_config.resolved_gripper_policy(_args(**{field: value}))


def test_dmtac_tactile_enablement_prefers_explicit_value() -> None:
    assert gripper_policy_config.dmtac_tactile_enabled(_args(with_tactile=False)) is False
    assert (
        gripper_policy_config.dmtac_tactile_enabled(
            _args(with_tactile=False, dmtac_tactile=True)
        )
        is True
    )
    assert (
        gripper_policy_config.dmtac_tactile_enabled(
            _args(with_tactile=True, dmtac_tactile=False)
        )
        is False
    )
