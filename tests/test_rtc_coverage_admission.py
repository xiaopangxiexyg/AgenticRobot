from __future__ import annotations

from types import SimpleNamespace

import pytest

from robot_motion.deploy.inference import (
    InferenceDelayEstimatorSpec,
    InferenceDelayEstimatorState,
)
from robot_motion.deploy.rtc import (
    RTC_LATENCY_LATE_BUT_USABLE,
    RTC_LATENCY_ON_TIME,
    RTC_LATENCY_STARVATION_RISK,
    assess_rtc_execution_coverage,
    assess_rtc_latency_envelope,
)
from robot_motion.deploy.rtc_runtime import RtcPolicyShadowFrontend
from teleop_runtime.deploy.run_policy import (
    _rtc_active_latency_admission_result,
    _rtc_recoverable_coverage_reason,
)

ACTION_PERIOD_SEC = 1.0 / 30.0


def _coverage(latency_sec: float):
    return assess_rtc_execution_coverage(
        latency_sec,
        action_period_sec=ACTION_PERIOD_SEC,
        prediction_horizon_steps=32,
        model_forecast_delay_steps=12,
        configured_trigger_coverage_steps=4,
        minimum_execution_horizon_steps=4,
        control_jitter_guard_steps=1,
        bridge_guard_steps=1,
    )


@pytest.mark.parametrize(
    ("latency_sec", "budget", "trigger", "suffix", "usable"),
    (
        (0.390, 13, 14, 19, True),
        (0.466, 15, 16, 17, True),
        (0.500, 16, 17, 16, False),
        (0.623, 20, 21, 12, False),
    ),
)
def test_conditioned_coverage_accounts_for_both_prefix_crops(
    latency_sec: float,
    budget: int,
    trigger: int,
    suffix: int,
    usable: bool,
) -> None:
    assessment = _coverage(latency_sec)

    assert assessment.latency_budget_steps == budget
    assert assessment.effective_trigger_coverage_steps == trigger
    assert assessment.model_prefix_crop_steps == 12
    assert assessment.additional_obsolete_steps == max(0, budget - 12)
    assert assessment.usable_result_suffix_steps == suffix
    assert assessment.execution_usable is usable
    assert (
        "result_suffix_cannot_sustain_next_request_cycle"
        in assessment.reasons
    ) is (not usable)


@pytest.mark.parametrize(
    ("latency_sec", "status", "slo_met", "usable"),
    (
        (0.390, RTC_LATENCY_ON_TIME, True, True),
        (0.466, RTC_LATENCY_LATE_BUT_USABLE, False, True),
        (0.500, RTC_LATENCY_STARVATION_RISK, False, False),
        (0.623, RTC_LATENCY_STARVATION_RISK, False, False),
    ),
)
def test_latency_slo_is_reported_independently_from_execution_coverage(
    latency_sec: float,
    status: str,
    slo_met: bool,
    usable: bool,
) -> None:
    assessment = assess_rtc_latency_envelope(
        latency_sec,
        performance_slo_sec=0.4,
        action_period_sec=ACTION_PERIOD_SEC,
        prediction_horizon_steps=32,
        model_forecast_delay_steps=12,
        configured_trigger_coverage_steps=4,
        minimum_execution_horizon_steps=4,
        control_jitter_guard_steps=1,
        bridge_guard_steps=1,
    )

    assert assessment.status == status
    assert assessment.performance_slo_met is slo_met
    assert assessment.execution_usable is usable


def test_completion_checks_the_actual_bridge_recorded_at_request_time() -> None:
    bridged = assess_rtc_execution_coverage(
        0.466,
        action_period_sec=ACTION_PERIOD_SEC,
        prediction_horizon_steps=32,
        model_forecast_delay_steps=12,
        configured_trigger_coverage_steps=4,
        minimum_execution_horizon_steps=4,
        control_jitter_guard_steps=1,
        request_plan_coverage_steps=16,
    )
    assert bridged.bridge_steps_at_completion == 1
    assert bridged.execution_usable

    gap_already_occurred = assess_rtc_execution_coverage(
        0.466,
        action_period_sec=ACTION_PERIOD_SEC,
        prediction_horizon_steps=32,
        model_forecast_delay_steps=12,
        configured_trigger_coverage_steps=4,
        minimum_execution_horizon_steps=4,
        control_jitter_guard_steps=1,
        request_plan_coverage_steps=15,
    )
    assert gap_already_occurred.bridge_steps_at_completion == 0
    assert not gap_already_occurred.execution_usable
    assert "old_plan_bridge_below_guard" in gap_already_occurred.reasons


def test_active_warmup_keeps_raw_rtc_strict_but_uses_coverage_for_model_rtc() -> None:
    raw = _rtc_active_latency_admission_result(
        [0.466],
        discard_completions=0,
        required_completions=1,
        performance_slo_sec=0.4,
    )
    assert raw == (True, False, 0.466, None)

    def assess(latency_sec: float):
        return assess_rtc_latency_envelope(
            latency_sec,
            performance_slo_sec=0.4,
            action_period_sec=ACTION_PERIOD_SEC,
            prediction_horizon_steps=32,
            model_forecast_delay_steps=12,
            configured_trigger_coverage_steps=4,
            minimum_execution_horizon_steps=4,
            control_jitter_guard_steps=1,
            bridge_guard_steps=1,
        )

    conditioned_usable = _rtc_active_latency_admission_result(
        [0.466],
        discard_completions=0,
        required_completions=1,
        performance_slo_sec=0.4,
        coverage_assessments=[assess(0.466)],
    )
    assert conditioned_usable[:3] == (True, True, 0.466)
    assert conditioned_usable[3].status == RTC_LATENCY_LATE_BUT_USABLE

    conditioned_starved = _rtc_active_latency_admission_result(
        [0.500],
        discard_completions=0,
        required_completions=1,
        performance_slo_sec=0.4,
        coverage_assessments=[assess(0.500)],
    )
    assert conditioned_starved[:3] == (True, False, 0.500)
    assert conditioned_starved[3].status == RTC_LATENCY_STARVATION_RISK


def test_warmup_coverage_stays_bound_to_each_requests_forecast_crop() -> None:
    slower_but_usable = assess_rtc_latency_envelope(
        0.390,
        performance_slo_sec=0.4,
        action_period_sec=ACTION_PERIOD_SEC,
        prediction_horizon_steps=32,
        model_forecast_delay_steps=12,
        configured_trigger_coverage_steps=4,
        minimum_execution_horizon_steps=4,
        control_jitter_guard_steps=1,
    )
    faster_but_starved = assess_rtc_latency_envelope(
        0.300,
        performance_slo_sec=0.4,
        action_period_sec=ACTION_PERIOD_SEC,
        prediction_horizon_steps=32,
        model_forecast_delay_steps=22,
        configured_trigger_coverage_steps=4,
        minimum_execution_horizon_steps=4,
        control_jitter_guard_steps=1,
    )

    result = _rtc_active_latency_admission_result(
        [0.390, 0.300],
        discard_completions=0,
        required_completions=2,
        performance_slo_sec=0.4,
        coverage_assessments=[slower_but_usable, faster_but_starved],
    )

    assert result[:3] == (True, False, 0.390)
    assert result[3] is faster_but_starved


def test_frontend_request_trigger_tracks_measured_latency_budget() -> None:
    spec = InferenceDelayEstimatorSpec(
        spec_id="rtc-coverage:test",
        action_period_sec=ACTION_PERIOD_SEC,
        window_size=4,
        warmup_latency_sec=0.1,
        network_guard_sec=0.0,
        control_jitter_guard_steps=1,
        minimum_delay_steps=1,
        maximum_delay_steps=32,
    )
    frontend = object.__new__(RtcPolicyShadowFrontend)
    frontend.config = SimpleNamespace(
        delay_estimator_spec=spec,
        model_continuation=object(),
        request_trigger_coverage_steps=4,
    )
    frontend._delay_state = InferenceDelayEstimatorState.empty(spec).record(
        spec,
        0.466,
    )

    assert frontend.effective_request_trigger_coverage_steps == 16


def test_only_typed_starvation_hold_is_recoverable_as_coverage_pause() -> None:
    assessment = _coverage(0.5)
    typed_step = SimpleNamespace(
        completion_coverage_assessment=assessment,
        lifecycle_transition=SimpleNamespace(
            producer_transition=SimpleNamespace(
                state=SimpleNamespace(
                    last_transition_reason="rtc_latency_starvation_risk"
                )
            )
        ),
    )
    assert (
        _rtc_recoverable_coverage_reason(typed_step)
        == "rtc_latency_starvation_risk"
    )

    arbitrary_hold = SimpleNamespace(
        completion_coverage_assessment=assessment,
        lifecycle_transition=SimpleNamespace(
            producer_transition=SimpleNamespace(
                state=SimpleNamespace(last_transition_reason="operator_hold")
            )
        ),
    )
    assert _rtc_recoverable_coverage_reason(arbitrary_hold) is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("latency_sec", -0.1),
        ("action_period_sec", 0.0),
        ("prediction_horizon_steps", 0),
        ("model_forecast_delay_steps", 33),
        ("minimum_execution_horizon_steps", 33),
    ),
)
def test_coverage_rejects_invalid_dimensions(field: str, value: float) -> None:
    kwargs = {
        "latency_sec": 0.1,
        "action_period_sec": ACTION_PERIOD_SEC,
        "prediction_horizon_steps": 32,
        "model_forecast_delay_steps": 12,
        "configured_trigger_coverage_steps": 4,
        "minimum_execution_horizon_steps": 4,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        assess_rtc_execution_coverage(**kwargs)
