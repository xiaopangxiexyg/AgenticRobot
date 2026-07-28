from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from deployments.client.rtc_frontend import (
    build_rtc_model_producer_runtime,
    build_rtc_model_shadow_frontend,
    rtc_producer_config_from_model_adapter,
    rtc_shadow_config_from_model_adapter,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.deploy.async_producer import (
    POLICY_ASYNC_EXECUTION_PROFILE_HASH_DOMAIN_V1,
    POLICY_DUE_POP_LATEST_VALID_NO_CATCHUP_V1,
    POLICY_REQUEST_TRIGGER_COVERAGE_SINGLE_IN_FLIGHT_V1,
    PolicyAsyncExecutionProfile,
)
from robot_motion.deploy.inference import (
    RTC_FLOW_INPAINT_MODE_V1,
    PolicyInferencePrefix,
    PolicyInferenceRequest,
)
from robot_motion.deploy.rtc import (
    RTC_CONTINUATION_PREFIX_SOURCE_PRODUCER_LOCAL_OVERLAP_V1,
    RTC_CONTINUATION_SAMPLER_CONSTRAINT_CLAMPED_V1,
    RTC_CONTINUATION_SAMPLER_CONSTRAINT_WEIGHTED_V1,
    RtcContinuationEncodingReceipt,
    RtcContinuationTimeline,
    RtcProducerProfile,
    rtc_continuation_timeline_from_request,
)
from robot_motion.position_control.config import PositionControlRates


def _legacy_profile(
    *,
    position_rates: PositionControlRates,
    position_lookahead_enabled: bool,
    send_actions: bool,
) -> PolicyAsyncExecutionProfile:
    return PolicyAsyncExecutionProfile(
        profile_id="rtc-package-architecture-test",
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
        observation_hz=13.0,
        decision_hz=29.0,
        action_hz=29.0,
        position_rates=position_rates,
        request_trigger_mode=(
            POLICY_REQUEST_TRIGGER_COVERAGE_SINGLE_IN_FLIGHT_V1
        ),
        due_pop_mode=POLICY_DUE_POP_LATEST_VALID_NO_CATCHUP_V1,
        position_lookahead_enabled=position_lookahead_enabled,
        send_actions=send_actions,
    )


def _request() -> PolicyInferenceRequest:
    prefix = PolicyInferencePrefix(
        model_space_actions=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        valid_steps=2,
        source_plan_id="plan:test",
        source_plan_generation=3,
        source_plan_hash="plan-hash:test",
    )
    return PolicyInferenceRequest(
        request_id="request:test",
        inference_id="inference:test",
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
        session_id="session:test",
        control_domain_id="domain:test",
        producer_id="producer:test",
        producer_epoch=2,
        policy_node_id="policy:test",
        artifact_hash="artifact:test",
        observation_cut_id="observation:test",
        observation_timestamp=9.9,
        observation_step_index=99,
        triggered_at=10.0,
        admitted_at=10.0,
        first_target_effective_at=10.2,
        clock_domain_id="clock:test",
        action_period_sec=0.1,
        prediction_horizon_steps=4,
        minimum_execution_horizon_steps=1,
        source_plan_id="plan:test",
        source_plan_generation=3,
        source_plan_hash="plan-hash:test",
        source_plan_cursor_at=10.0,
        producer_plan_committed_through=9.9,
        forecast_delay_steps=2,
        delay_estimator_spec_hash="delay:test",
        action_codec_hash="codec:test",
        frame_contract_hash="frame:test",
        layout_hash="layout:test",
        normalization_hash="normalization:test",
        valid_until=10.6,
        conditioning_prefix=prefix,
    )


def test_legacy_profile_hash_is_unchanged_by_target_neutral_projection() -> None:
    profile = _legacy_profile(
        position_rates=PositionControlRates(
            sample_hz=13.0,
            reference_hz=58.0,
            ik_hz=58.0,
            command_hz=116.0,
        ),
        position_lookahead_enabled=True,
        send_actions=True,
    )
    before = profile.semantic_content_hash
    producer = profile.producer_profile
    assert isinstance(producer, RtcProducerProfile)
    assert producer.action_period_sec == pytest.approx(1.0 / 29.0)
    assert profile.semantic_content_hash == before
    assert before == canonical_content_hash(
        {
            "profile_id": profile.profile_id,
            "rtc_mode": profile.rtc_mode,
            "observation_hz": profile.observation_hz,
            "decision_hz": profile.decision_hz,
            "action_hz": profile.action_hz,
            "position_rates": profile.position_rates.to_metadata(),
            "request_trigger_mode": profile.request_trigger_mode,
            "due_pop_mode": profile.due_pop_mode,
            "position_lookahead_enabled": profile.position_lookahead_enabled,
            "send_actions": profile.send_actions,
        },
        domain=POLICY_ASYNC_EXECUTION_PROFILE_HASH_DOMAIN_V1,
    )


def test_producer_profile_identity_excludes_target_execution_settings() -> None:
    first = _legacy_profile(
        position_rates=PositionControlRates(
            sample_hz=13.0,
            reference_hz=58.0,
            ik_hz=58.0,
            command_hz=116.0,
        ),
        position_lookahead_enabled=False,
        send_actions=False,
    )
    second = _legacy_profile(
        position_rates=PositionControlRates(
            sample_hz=29.0,
            reference_hz=87.0,
            ik_hz=87.0,
            command_hz=174.0,
        ),
        position_lookahead_enabled=True,
        send_actions=True,
    )
    assert first.semantic_content_hash != second.semantic_content_hash
    assert (
        first.producer_profile.semantic_content_hash
        == second.producer_profile.semantic_content_hash
    )


def test_continuation_timeline_applies_forecast_delay_exactly_once() -> None:
    timeline = rtc_continuation_timeline_from_request(_request())
    assert timeline.generation_start_at == pytest.approx(10.0)
    assert timeline.first_target_effective_at == pytest.approx(10.2)
    assert timeline.maximum_execution_horizon_steps == 2
    assert timeline.generation_index_for_response(0) == 2
    assert timeline.generation_target_at(2) == pytest.approx(
        timeline.response_target_at(0)
    )
    assert timeline.generation_index_for_response(1) == 3


def test_continuation_timeline_rejects_double_delay_or_empty_suffix() -> None:
    with pytest.raises(ValueError, match="must equal generation_start_at"):
        RtcContinuationTimeline(
            clock_domain_id="clock:test",
            generation_start_at=10.0,
            first_target_effective_at=10.4,
            action_period_sec=0.1,
            prediction_horizon_steps=4,
            forecast_delay_steps=2,
        )
    with pytest.raises(ValueError, match="non-empty executable suffix"):
        RtcContinuationTimeline(
            clock_domain_id="clock:test",
            generation_start_at=10.0,
            first_target_effective_at=10.4,
            action_period_sec=0.1,
            prediction_horizon_steps=4,
            forecast_delay_steps=4,
        )


def test_encoding_receipt_separates_source_authority_from_sampler_strength() -> None:
    timeline = rtc_continuation_timeline_from_request(_request())
    base = RtcContinuationEncodingReceipt(
        source_plan_id="plan:test",
        source_plan_generation=3,
        source_plan_hash="plan-hash:test",
        timeline_hash=timeline.semantic_content_hash,
        prefix_source_role=(
            RTC_CONTINUATION_PREFIX_SOURCE_PRODUCER_LOCAL_OVERLAP_V1
        ),
        sampler_constraint=RTC_CONTINUATION_SAMPLER_CONSTRAINT_CLAMPED_V1,
        valid_prefix_steps=2,
        component_scope=("right_arm", "left_arm", "left_arm"),
        canonical_prefix_hash="canonical-prefix:test",
        model_prefix_hash="model-prefix:test",
        action_codec_hash="codec:test",
        frame_contract_hash="frame:test",
        layout_hash="layout:test",
        normalization_hash="normalization:test",
    )
    assert base.component_scope == ("left_arm", "right_arm")
    weighted = replace(
        base,
        sampler_constraint=RTC_CONTINUATION_SAMPLER_CONSTRAINT_WEIGHTED_V1,
    )
    assert base.prefix_source_role == weighted.prefix_source_role
    assert base.semantic_content_hash != weighted.semantic_content_hash
    with pytest.raises(ValueError, match="producer-local overlap"):
        replace(base, prefix_source_role="downstream_accepted")


def test_rtc_core_subpackage_has_no_model_target_or_application_imports() -> None:
    root = Path(__file__).resolve().parents[1] / "robot_motion/deploy/rtc"
    forbidden_roots = {
        "control_composition",
        "control_station",
        "deployments",
        "hardware",
        "model",
        "sim",
        "teleop_runtime",
    }
    violations: dict[str, list[str]] = {}
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                imported.add(node.module.split(".", 1)[0])
        forbidden = sorted(imported & forbidden_roots)
        if forbidden:
            violations[path.name] = forbidden
    assert violations == {}


def test_model_rtc_cross_package_dependencies_point_downward() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden_by_path = {
        "dataset_contracts/model_continuation.py": {
            "control_composition",
            "control_station",
            "dataset_interface",
            "deployments",
            "hardware",
            "model",
            "robot_motion",
            "sim",
            "teleop_runtime",
        },
        "dataset_interface/model_continuation.py": {
            "control_composition",
            "control_station",
            "deployments",
            "hardware",
            "model",
            "sim",
            "teleop_runtime",
        },
        "deployments/client/rtc_conditioning.py": {
            "control_composition",
            "control_station",
            "hardware",
            "model",
            "sim",
            "teleop_runtime",
        },
        "deployments/client/rtc_conditioning_protocol.py": {
            "control_composition",
            "control_station",
            "hardware",
            "model",
            "sim",
            "teleop_runtime",
        },
        "deployments/openpi/client/rtc_conditioned.py": {
            "control_composition",
            "control_station",
            "hardware",
            "model",
            "sim",
            "teleop_runtime",
        },
    }
    violations: dict[str, list[str]] = {}
    for relative, forbidden_roots in forbidden_by_path.items():
        path = root / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                imported.add(node.module.split(".", 1)[0])
        forbidden = sorted(imported & forbidden_roots)
        if forbidden:
            violations[relative] = forbidden
    assert violations == {}


def test_canonical_model_frontend_names_are_compatibility_aliases() -> None:
    assert build_rtc_model_producer_runtime is build_rtc_model_shadow_frontend
    assert (
        rtc_producer_config_from_model_adapter
        is rtc_shadow_config_from_model_adapter
    )
