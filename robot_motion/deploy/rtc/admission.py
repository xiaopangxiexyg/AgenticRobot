"""Pure execution-coverage and latency-SLO assessment for RTC results.

The model-RTC client removes the forecast prefix from every generated chunk.
Timed-plan admission may then discard more rows that are already obsolete at
the actual completion cursor. This module accounts for both operations. It
owns no worker, lifecycle, queue, dispatch, or hardware state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

RTC_LATENCY_ON_TIME = "on_time"
RTC_LATENCY_LATE_BUT_USABLE = "late_but_usable"
RTC_LATENCY_STARVATION_RISK = "starvation_risk"
RTC_REQUEST_BRIDGE_GUARD_STEPS = 1
RTC_LATENCY_ENVELOPE_STATUSES = (
    RTC_LATENCY_ON_TIME,
    RTC_LATENCY_LATE_BUT_USABLE,
    RTC_LATENCY_STARVATION_RISK,
)


def _finite_nonnegative(value: Any, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return normalized


def _positive(value: Any, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive.")
    return normalized


def _nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return normalized


def _positive_int(value: Any, *, field_name: str) -> int:
    normalized = _nonnegative_int(value, field_name=field_name)
    if normalized == 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return normalized


def rtc_latency_budget_steps(
    latency_sec: float,
    *,
    action_period_sec: float,
    network_guard_sec: float = 0.0,
    control_jitter_guard_steps: int = 0,
) -> int:
    """Convert end-to-end latency to an unclamped control-step budget."""

    latency = _finite_nonnegative(latency_sec, field_name="latency_sec")
    period = _positive(action_period_sec, field_name="action_period_sec")
    network_guard = _finite_nonnegative(
        network_guard_sec,
        field_name="network_guard_sec",
    )
    jitter_guard = _nonnegative_int(
        control_jitter_guard_steps,
        field_name="control_jitter_guard_steps",
    )
    return math.ceil((latency + network_guard) / period) + jitter_guard


@dataclass(frozen=True)
class RtcExecutionCoverageAssessment:
    """Evidence that one completion can sustain the next request cycle."""

    observed_latency_sec: float
    action_period_sec: float
    observed_latency_steps: int
    latency_budget_steps: int
    model_forecast_delay_steps: int
    configured_trigger_coverage_steps: int
    effective_trigger_coverage_steps: int
    bridge_guard_steps: int
    request_plan_coverage_steps: int | None
    bridge_steps_at_completion: int | None
    model_prefix_crop_steps: int
    additional_obsolete_steps: int
    usable_result_suffix_steps: int
    minimum_execution_horizon_steps: int
    stable_cycle_usable: bool
    reasons: tuple[str, ...]

    @property
    def execution_usable(self) -> bool:
        return self.stable_cycle_usable

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "dexteleop.rtc_execution_coverage_assessment.v1",
            "observed_latency_sec": self.observed_latency_sec,
            "action_period_sec": self.action_period_sec,
            "observed_latency_steps": self.observed_latency_steps,
            "latency_budget_steps": self.latency_budget_steps,
            "model_forecast_delay_steps": self.model_forecast_delay_steps,
            "configured_trigger_coverage_steps": (
                self.configured_trigger_coverage_steps
            ),
            "effective_trigger_coverage_steps": (
                self.effective_trigger_coverage_steps
            ),
            "bridge_guard_steps": self.bridge_guard_steps,
            "request_plan_coverage_steps": self.request_plan_coverage_steps,
            "bridge_steps_at_completion": self.bridge_steps_at_completion,
            "model_prefix_crop_steps": self.model_prefix_crop_steps,
            "additional_obsolete_steps": self.additional_obsolete_steps,
            "usable_result_suffix_steps": self.usable_result_suffix_steps,
            "minimum_execution_horizon_steps": (
                self.minimum_execution_horizon_steps
            ),
            "stable_cycle_usable": self.stable_cycle_usable,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RtcLatencyEnvelopeAssessment(RtcExecutionCoverageAssessment):
    """Independent performance-SLO and execution-coverage evidence."""

    status: str
    performance_slo_sec: float
    performance_slo_met: bool

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload.update(
            {
                "schema": "dexteleop.rtc_latency_envelope_assessment.v2",
                "status": self.status,
                "performance_slo_sec": self.performance_slo_sec,
                "performance_slo_met": self.performance_slo_met,
            }
        )
        return payload


def assess_rtc_execution_coverage(
    latency_sec: float,
    *,
    action_period_sec: float,
    prediction_horizon_steps: int,
    model_forecast_delay_steps: int,
    configured_trigger_coverage_steps: int,
    minimum_execution_horizon_steps: int,
    network_guard_sec: float = 0.0,
    control_jitter_guard_steps: int = 0,
    bridge_guard_steps: int = RTC_REQUEST_BRIDGE_GUARD_STEPS,
    request_plan_coverage_steps: int | None = None,
) -> RtcExecutionCoverageAssessment:
    """Assess both model-prefix crop and actual-cursor obsolescence.

    A conditioned H-step generation is first cropped by F forecast steps in
    the model client. At completion B budget steps have elapsed, so timed-plan
    admission discards a further ``max(0, B - F)`` rows. The executable suffix
    is therefore ``H - max(F, B)``, not ``H - max(0, B - F)``.
    """

    latency = _finite_nonnegative(latency_sec, field_name="latency_sec")
    period = _positive(action_period_sec, field_name="action_period_sec")
    horizon = _positive_int(
        prediction_horizon_steps,
        field_name="prediction_horizon_steps",
    )
    forecast = _nonnegative_int(
        model_forecast_delay_steps,
        field_name="model_forecast_delay_steps",
    )
    configured_trigger = _nonnegative_int(
        configured_trigger_coverage_steps,
        field_name="configured_trigger_coverage_steps",
    )
    minimum_horizon = _positive_int(
        minimum_execution_horizon_steps,
        field_name="minimum_execution_horizon_steps",
    )
    bridge_guard = _nonnegative_int(
        bridge_guard_steps,
        field_name="bridge_guard_steps",
    )
    request_coverage = (
        None
        if request_plan_coverage_steps is None
        else _nonnegative_int(
            request_plan_coverage_steps,
            field_name="request_plan_coverage_steps",
        )
    )
    if forecast > horizon:
        raise ValueError("model_forecast_delay_steps exceeds prediction horizon.")
    if minimum_horizon > horizon:
        raise ValueError("minimum execution horizon exceeds prediction horizon.")

    observed_steps = math.ceil(latency / period)
    budget_steps = rtc_latency_budget_steps(
        latency,
        action_period_sec=period,
        network_guard_sec=network_guard_sec,
        control_jitter_guard_steps=control_jitter_guard_steps,
    )
    effective_trigger = max(configured_trigger, budget_steps + bridge_guard)
    bridge_at_completion = (
        None
        if request_coverage is None
        else max(0, request_coverage - budget_steps)
    )
    additional_obsolete = max(0, budget_steps - forecast)
    usable_suffix = max(0, horizon - forecast - additional_obsolete)

    reasons: list[str] = []
    if (
        bridge_at_completion is not None
        and bridge_at_completion < bridge_guard
    ):
        reasons.append("old_plan_bridge_below_guard")
    if usable_suffix < minimum_horizon:
        reasons.append("result_suffix_below_minimum_execution_horizon")
    if effective_trigger > usable_suffix:
        reasons.append("result_suffix_cannot_sustain_next_request_cycle")

    return RtcExecutionCoverageAssessment(
        observed_latency_sec=latency,
        action_period_sec=period,
        observed_latency_steps=observed_steps,
        latency_budget_steps=budget_steps,
        model_forecast_delay_steps=forecast,
        configured_trigger_coverage_steps=configured_trigger,
        effective_trigger_coverage_steps=effective_trigger,
        bridge_guard_steps=bridge_guard,
        request_plan_coverage_steps=request_coverage,
        bridge_steps_at_completion=bridge_at_completion,
        model_prefix_crop_steps=forecast,
        additional_obsolete_steps=additional_obsolete,
        usable_result_suffix_steps=usable_suffix,
        minimum_execution_horizon_steps=minimum_horizon,
        stable_cycle_usable=not reasons,
        reasons=tuple(reasons),
    )


def assess_rtc_latency_envelope(
    latency_sec: float,
    *,
    performance_slo_sec: float,
    action_period_sec: float,
    prediction_horizon_steps: int,
    model_forecast_delay_steps: int,
    configured_trigger_coverage_steps: int,
    minimum_execution_horizon_steps: int,
    network_guard_sec: float = 0.0,
    control_jitter_guard_steps: int = 0,
    bridge_guard_steps: int = RTC_REQUEST_BRIDGE_GUARD_STEPS,
    request_plan_coverage_steps: int | None = None,
) -> RtcLatencyEnvelopeAssessment:
    """Report the strict performance SLO independently from plan safety."""

    slo = _positive(performance_slo_sec, field_name="performance_slo_sec")
    coverage = assess_rtc_execution_coverage(
        latency_sec,
        action_period_sec=action_period_sec,
        prediction_horizon_steps=prediction_horizon_steps,
        model_forecast_delay_steps=model_forecast_delay_steps,
        configured_trigger_coverage_steps=configured_trigger_coverage_steps,
        minimum_execution_horizon_steps=minimum_execution_horizon_steps,
        network_guard_sec=network_guard_sec,
        control_jitter_guard_steps=control_jitter_guard_steps,
        bridge_guard_steps=bridge_guard_steps,
        request_plan_coverage_steps=request_plan_coverage_steps,
    )
    performance_slo_met = coverage.observed_latency_sec < slo
    if not coverage.execution_usable:
        status = RTC_LATENCY_STARVATION_RISK
    elif performance_slo_met:
        status = RTC_LATENCY_ON_TIME
    else:
        status = RTC_LATENCY_LATE_BUT_USABLE
    return RtcLatencyEnvelopeAssessment(
        **coverage.__dict__,
        status=status,
        performance_slo_sec=slo,
        performance_slo_met=performance_slo_met,
    )
