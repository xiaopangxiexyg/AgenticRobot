from __future__ import annotations

from dataclasses import replace

import pytest

from robot_motion.deploy.inference import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    RTC_FLOW_INPAINT_MODE_V1,
    PolicyInferencePrefix,
    PolicyInferenceRequest,
)


RAW_REQUEST_HASH_GOLDEN_V1 = (
    "1270deb12533367938f001868d26edb1eff4c81f1e618e956ac35a002f97e9b6"
)
LEGACY_PREFIX_HASH_GOLDEN_V1 = (
    "25ebee3adf4dc6c856d6afb7b3ea0ee845eed38b5ad8f8aa1f099bf37e9761d6"
)
LEGACY_FLOW_REQUEST_HASH_GOLDEN_V1 = (
    "b77281661992674d5ebcc5faf81fa7e48f04e9e159efe6e572f60b709973ab4c"
)


def _request_kwargs() -> dict[str, object]:
    return {
        "request_id": "request:r1-golden",
        "inference_id": "inference:r1-golden",
        "session_id": "session:r1",
        "control_domain_id": "domain:r1",
        "producer_id": "producer:r1",
        "producer_epoch": 2,
        "policy_node_id": "policy:r1",
        "artifact_hash": "artifact:r1",
        "observation_cut_id": "cut:r1",
        "observation_timestamp": 10.0,
        "observation_step_index": 100,
        "triggered_at": 10.1,
        "admitted_at": 10.1,
        "first_target_effective_at": 10.3,
        "clock_domain_id": "clock:r1",
        "action_period_sec": 0.1,
        "prediction_horizon_steps": 4,
        "minimum_execution_horizon_steps": 1,
        "source_plan_id": "plan:r1",
        "source_plan_generation": 3,
        "source_plan_hash": "plan-hash:r1",
        "source_plan_cursor_at": 10.1,
        "producer_plan_committed_through": 10.0,
        "forecast_delay_steps": 2,
        "delay_estimator_spec_hash": "delay:r1",
        "action_codec_hash": "codec:r1",
        "frame_contract_hash": "frame:r1",
        "layout_hash": "layout:r1",
        "normalization_hash": "normalization:r1",
        "valid_until": 10.8,
    }


def _legacy_prefix() -> PolicyInferencePrefix:
    return PolicyInferencePrefix(
        model_space_actions=[[1.0, 2.0], [3.0, 4.0]],
        valid_steps=2,
        source_plan_id="plan:r1",
        source_plan_generation=3,
        source_plan_hash="plan-hash:r1",
    )


def test_raw_and_legacy_model_prefix_hashes_remain_exactly_unchanged() -> None:
    raw = PolicyInferenceRequest(
        rtc_mode=ASYNC_ALIGNED_CHUNK_MODE_V1,
        **_request_kwargs(),
    )
    prefix = _legacy_prefix()
    legacy_flow = PolicyInferenceRequest(
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
        conditioning_prefix=prefix,
        **_request_kwargs(),
    )

    assert raw.semantic_content_hash == RAW_REQUEST_HASH_GOLDEN_V1
    assert prefix.semantic_content_hash == LEGACY_PREFIX_HASH_GOLDEN_V1
    assert (
        legacy_flow.semantic_content_hash
        == LEGACY_FLOW_REQUEST_HASH_GOLDEN_V1
    )


def test_causal_binding_allows_model_rtc_request_without_model_arrays() -> None:
    request = PolicyInferenceRequest(
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
        conditioning_binding_hash="sha256:conditioning-bundle-v1",
        **_request_kwargs(),
    )

    assert request.conditioning_prefix is None
    assert request.conditioning_prefix_hash is None
    assert (
        request.conditioning_binding_hash
        == "sha256:conditioning-bundle-v1"
    )
    assert request.semantic_content_hash not in {
        RAW_REQUEST_HASH_GOLDEN_V1,
        LEGACY_FLOW_REQUEST_HASH_GOLDEN_V1,
    }
    assert request.semantic_content_hash != replace(
        request,
        conditioning_binding_hash="sha256:conditioning-bundle-v2",
    ).semantic_content_hash


def test_legacy_prefix_and_new_binding_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="cannot carry both"):
        PolicyInferenceRequest(
            rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
            conditioning_prefix=_legacy_prefix(),
            conditioning_binding_hash="sha256:conditioning-bundle-v1",
            **_request_kwargs(),
        )


def test_new_binding_is_rejected_on_raw_rtc_and_missing_on_model_rtc() -> None:
    with pytest.raises(ValueError, match="requires rtc_flow_inpaint_v1"):
        PolicyInferenceRequest(
            rtc_mode=ASYNC_ALIGNED_CHUNK_MODE_V1,
            conditioning_binding_hash="sha256:conditioning-bundle-v1",
            **_request_kwargs(),
        )
    with pytest.raises(ValueError, match="requires a legacy prefix or causal"):
        PolicyInferenceRequest(
            rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
            **_request_kwargs(),
        )
