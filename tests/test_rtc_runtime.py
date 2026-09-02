from __future__ import annotations

from dataclasses import replace

import pytest

from robot_motion.deploy.async_producer import PolicyAsyncExecutionProfile
from robot_motion.deploy.inference import (
    InferenceDelayEstimatorSpec,
    RTC_FLOW_INPAINT_MODE_V1,
)
from robot_motion.deploy.rtc_runtime import (
    RTC_MODEL_WORKER_STOP_DISABLED_POLICY_V1,
    RTC_REQUEST_VALIDITY_POLICY_V1,
    RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V4,
    RtcPolicyShadowConfig,
)


def _config(*, request_validity_steps: int = 2) -> RtcPolicyShadowConfig:
    profile = PolicyAsyncExecutionProfile(
        profile_id="rtc-runtime-contract-test",
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
    )
    delay_spec = InferenceDelayEstimatorSpec(
        spec_id="rtc-runtime-delay-test",
        action_period_sec=1.0 / 30.0,
        window_size=4,
        warmup_latency_sec=1.0 / 30.0,
        network_guard_sec=0.0,
        control_jitter_guard_steps=0,
        minimum_delay_steps=1,
        maximum_delay_steps=4,
    )
    return RtcPolicyShadowConfig(
        runtime_session_id="rtc-runtime-session",
        control_domain_id="rtc-runtime-control",
        producer_id="rtc-runtime-producer",
        policy_node_id="rtc-runtime-policy",
        artifact_hash="artifact:test",
        action_codec_hash="codec:test",
        frame_contract_hash="frame:test",
        layout_hash="layout:test",
        normalization_hash="normalization:test",
        profile=profile,
        prediction_horizon_steps=8,
        minimum_execution_horizon_steps=1,
        request_validity_steps=request_validity_steps,
        delay_estimator_spec=delay_spec,
    )


def test_delay_spec_and_runtime_derive_one_ttl_policy() -> None:
    config = _config()

    assert config.minimum_safe_request_validity_steps == 5
    assert config.effective_request_validity_steps(1) == 2
    assert config.effective_request_validity_steps(4) == 5
    assert config.timing_contract["request_validity_policy"] == (
        RTC_REQUEST_VALIDITY_POLICY_V1
    )
    assert config.timing_contract["minimum_safe_request_validity_steps"] == 5
    assert config.timing_contract["configured_request_validity_steps"] == 2
    assert config.timing_contract["worker_stop_policy"] == (
        RTC_MODEL_WORKER_STOP_DISABLED_POLICY_V1
    )


def test_runtime_rejects_forecast_outside_the_pinned_delay_range() -> None:
    config = _config()

    with pytest.raises(ValueError, match="forecast_delay_steps"):
        config.effective_request_validity_steps(0)
    with pytest.raises(ValueError, match="forecast_delay_steps"):
        config.effective_request_validity_steps(5)


def test_timing_contract_is_part_of_the_current_config_identity() -> None:
    config = _config()
    changed = replace(config, request_validity_steps=3)

    assert config.timing_contract_hash != changed.timing_contract_hash
    assert config.semantic_content_hash != changed.semantic_content_hash
    assert RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V4 not in {
        "",
    }
