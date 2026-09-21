from __future__ import annotations

from types import SimpleNamespace

import pytest

from robot_motion.deploy.async_producer import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
)
from teleop_runtime.deploy import rtc_cli_validation, run_policy


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "debug": False,
        "rtc_debug_level": None,
        "send_actions": False,
        "debug_replay_dataset": None,
        "debug_replay_parquet": None,
        "action_hz_dataset": None,
        "action_hz_parquet": None,
        "append_action_chunks": False,
        "action_queue_policy": "replace",
        "policy_spec": "hold",
        "policy_artifact_json": None,
        "policy_kwargs_json": "",
        "left_arm_ip": "",
        "right_arm_ip": "",
        "left_hand_ip": "",
        "right_hand_ip": "",
        "hands_enabled": False,
        "action_hz": 0.0,
        "policy_hz": None,
        "rtc_action_hz": None,
        "position_chunk_lookahead": True,
        "rtc_request_validity_steps": 10,
        "rtc_worker_stop_timeout_sec": 0.0,
        "deploy_fixed_init_dataset": None,
        "deploy_fixed_init_parquet": None,
        "policy_qpos_init_dataset": None,
        "policy_qpos_init_parquet": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate(
    args: SimpleNamespace,
    *,
    profile_id: str = "off",
    profile_enabled: bool = False,
    real_shadow: bool = False,
) -> None:
    rtc_cli_validation.validate_rtc_cli_args(
        args,
        rtc_profile_id=profile_id,
        rtc_profile_enabled=profile_enabled,
        rtc_real_shadow=real_shadow,
        hand_backend_mode_for_side=lambda _args, _side: "inspire",
    )


def test_run_policy_keeps_the_extracted_validator_facade() -> None:
    assert run_policy.validate_rtc_cli_args is rtc_cli_validation.validate_rtc_cli_args


def test_active_rtc_max_steps_counts_accepted_dispatches_not_warmup_ticks() -> None:
    assert not run_policy._bounded_policy_step_limit_reached(
        max_steps=10,
        control_loop_steps=57,
        rtc_active_dispatch_counting=True,
        accepted_rtc_dispatch_steps=0,
    )
    assert not run_policy._bounded_policy_step_limit_reached(
        max_steps=10,
        control_loop_steps=66,
        rtc_active_dispatch_counting=True,
        accepted_rtc_dispatch_steps=9,
    )
    assert run_policy._bounded_policy_step_limit_reached(
        max_steps=10,
        control_loop_steps=67,
        rtc_active_dispatch_counting=True,
        accepted_rtc_dispatch_steps=10,
    )


def test_non_active_modes_keep_max_steps_control_loop_semantics() -> None:
    assert not run_policy._bounded_policy_step_limit_reached(
        max_steps=10,
        control_loop_steps=9,
        rtc_active_dispatch_counting=False,
        accepted_rtc_dispatch_steps=100,
    )
    assert run_policy._bounded_policy_step_limit_reached(
        max_steps=10,
        control_loop_steps=10,
        rtc_active_dispatch_counting=False,
        accepted_rtc_dispatch_steps=0,
    )


def test_off_profile_defaults_debug_level_and_rejects_overrides() -> None:
    args = _args()
    _validate(args)
    assert args._rtc_debug_level == "off"

    with pytest.raises(ValueError, match="RTC validity/worker-stop overrides"):
        _validate(_args(rtc_request_validity_steps=11))


def test_debug_level_requires_debug_and_an_enabled_profile() -> None:
    with pytest.raises(ValueError, match="requires --debug"):
        _validate(
            _args(rtc_debug_level="summary"), profile_id="rtc", profile_enabled=True
        )

    with pytest.raises(ValueError, match="requires an RTC profile"):
        _validate(_args(debug=True, rtc_debug_level="summary"))


def test_rtc3c_only_accepts_builtin_hold_without_artifact_or_kwargs() -> None:
    with pytest.raises(ValueError, match="builtin hold virtual policy"):
        _validate(
            _args(policy_spec="custom"),
            profile_id=RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            profile_enabled=True,
        )
    with pytest.raises(ValueError, match="does not bind a model artifact"):
        _validate(
            _args(policy_artifact_json="artifact.json"),
            profile_id=RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            profile_enabled=True,
        )
    with pytest.raises(ValueError, match="accepts no policy kwargs"):
        _validate(
            _args(policy_kwargs_json='{"temperature": 0.1}'),
            profile_id=RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            profile_enabled=True,
        )


def test_model_profile_requires_artifact_and_shadow_for_live_endpoints() -> None:
    with pytest.raises(ValueError, match="requires an explicit --policy-artifact-json"):
        _validate(
            _args(),
            profile_id=RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
            profile_enabled=True,
        )
    with pytest.raises(ValueError, match="requires explicit --rtc-real-shadow"):
        _validate(
            _args(policy_artifact_json="artifact.json", left_arm_ip="192.0.2.10"),
            profile_id=RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
            profile_enabled=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("debug_replay_dataset", "dataset", "do not modify or consume debug replay"),
        ("action_hz_dataset", "dataset", "do not consume replay/reference rate inputs"),
        ("append_action_chunks", True, "timed mutable-future merge"),
    ),
)
def test_enabled_rtc_profile_rejects_legacy_queue_inputs(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(
            _args(**{field: value}),
            profile_id=RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            profile_enabled=True,
    )


def test_rtc3c_keeps_virtual_profile_without_action_dispatch() -> None:
    with pytest.raises(ValueError, match="does not accept --send-actions"):
        _validate(
            _args(send_actions=True),
            profile_id=RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            profile_enabled=True,
        )


@pytest.mark.parametrize(
    "profile_id",
    (RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1, MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1),
)
def test_model_rtc_profiles_allow_explicit_actions_and_qpos_init(
    profile_id: str,
) -> None:
    args = _args(
        policy_artifact_json="artifact.json",
        policy_qpos_init_dataset="dataset",
        send_actions=True,
    )
    _validate(
        args,
        profile_id=profile_id,
        profile_enabled=True,
        real_shadow=True,
    )
    assert args.send_actions is True
    assert args.policy_qpos_init_dataset == "dataset"
    assert args.policy_qpos_init_parquet is None


def test_conditioned_model_rtc_allows_explicit_15hz_runtime_cadence() -> None:
    args = _args(
        policy_artifact_json="artifact.json",
        policy_qpos_init_dataset="dataset",
        rtc_action_hz=15.0,
    )
    _validate(
        args,
        profile_id=MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
        profile_enabled=True,
        real_shadow=True,
    )
    assert args.action_hz == 15.0
    assert args.policy_hz == 30.0


def test_rtc_action_hz_override_is_model_only_and_conflict_checked() -> None:
    with pytest.raises(ValueError, match="only supported with the conditioned"):
        _validate(
            _args(rtc_action_hz=15.0),
            profile_id=RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
            profile_enabled=True,
        )
    with pytest.raises(ValueError, match="conflicts with --rtc-action-hz"):
        _validate(
            _args(
                policy_artifact_json="artifact.json",
                rtc_action_hz=15.0,
                action_hz=30.0,
            ),
            profile_id=MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
            profile_enabled=True,
        )


def test_model_rtc_explicit_actions_without_live_endpoints_remain_synthetic() -> None:
    args = _args(policy_artifact_json="artifact.json", send_actions=True)
    _validate(
        args,
        profile_id=RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
        profile_enabled=True,
    )
    assert args.send_actions is True


def test_rtc3c_applies_fixed_rates_and_disables_chunk_lookahead() -> None:
    args = _args(debug=True)
    _validate(
        args,
        profile_id=RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
        profile_enabled=True,
    )
    assert args._rtc_debug_level == "summary"
    assert args.action_hz == 30.0
    assert args.policy_hz == 30.0
    assert args.position_chunk_lookahead is False


def test_rtc3c_rejects_non_fixed_rates() -> None:
    with pytest.raises(ValueError, match="action rate is fixed"):
        _validate(
            _args(action_hz=15.0),
            profile_id=RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            profile_enabled=True,
        )
    with pytest.raises(ValueError, match="decision rate is fixed"):
        _validate(
            _args(policy_hz=15.0),
            profile_id=RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            profile_enabled=True,
        )


def test_real_shadow_requires_model_profile_and_preserves_init_inputs() -> None:
    with pytest.raises(ValueError, match="explicit model-backed RTC profile"):
        _validate(_args(), real_shadow=True)

    policy_qpos_args = _args(
        policy_artifact_json="artifact.json",
        policy_qpos_init_dataset="dataset",
    )
    _validate(
        policy_qpos_args,
        profile_id=RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
        profile_enabled=True,
        real_shadow=True,
    )
    assert policy_qpos_args.policy_qpos_init_dataset == "dataset"

    fixed_init_args = _args(
        policy_artifact_json="artifact.json",
        deploy_fixed_init_dataset="dataset",
    )
    _validate(
        fixed_init_args,
        profile_id=RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
        profile_enabled=True,
        real_shadow=True,
    )
    assert fixed_init_args.deploy_fixed_init_dataset == "dataset"
