"""Pure binding between decision timing and reference receiver semantics.

For ``scheduled_segment`` profiles, admission is a strict two-owner protocol:
the action scheduler owns the *candidate tick* and must not publish a tick whose
START is no longer in the future; the receiver owns the final START deadline
check.  A late scheduler observation skips the invalid grid point and records
the next legal fixed-grid tick.  A segment submitted directly after its START
is rejected with ``reject_and_hold``; neither path backdates or catches up a
segment, and the control-station layer does not recompute either timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.position_control.capability import ExecutionCapabilityProfile

from .arbitration_runtime import DecisionScheduleSpec


REFERENCE_ACTIVATION_HASH_DOMAIN_V1 = "dexteleop.reference_activation_spec.v1"

IMMEDIATE_QREF_SIM_V1 = "immediate_qref_sim_v1"
SCHEDULED_REFERENCE_SEGMENT_SIM_V1 = "scheduled_reference_segment_sim_v1"
SCHEDULED_REFERENCE_LATE_TICK_RECOVERY_V1 = (
    "skip_to_next_legal_reference_tick"
)
SCHEDULED_REFERENCE_LATE_ADMISSION_REJECTION_V1 = "reject_and_hold"

REFERENCE_ACTIVATION_MODES_V1 = frozenset(
    {"immediate_target", "scheduled_segment"}
)


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite_non_negative(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


@dataclass(frozen=True)
class ReferenceActivationSpec:
    """Versioned receiver timing semantics; it owns no runtime state."""

    spec_id: str
    activation_mode: str
    decision_schedule_spec_hash: str
    execution_capability_profile_hash: str
    target_lead_time_sec: float
    selected_keyframe_period_sec: float
    reference_period_sec: float
    minimum_activation_margin_sec: float
    activation_margin_evidence_id: str
    required_admission_deadline: str
    first_segment_policy: str
    late_admission_policy: str
    reference_tick_alignment_policy: str
    anchor_commit_policy: str
    replacement_policy: str
    semantic_content_hash: str
    late_tick_recovery_policy: str = "not_applicable"

    def __post_init__(self) -> None:
        for name in (
            "spec_id",
            "decision_schedule_spec_hash",
            "execution_capability_profile_hash",
            "activation_margin_evidence_id",
            "required_admission_deadline",
            "first_segment_policy",
            "late_admission_policy",
            "reference_tick_alignment_policy",
            "anchor_commit_policy",
            "replacement_policy",
            "semantic_content_hash",
            "late_tick_recovery_policy",
        ):
            _required_text(getattr(self, name), name=name)
        if self.activation_mode not in REFERENCE_ACTIVATION_MODES_V1:
            raise ValueError("Unsupported reference activation mode.")
        for name in (
            "target_lead_time_sec",
            "selected_keyframe_period_sec",
            "reference_period_sec",
            "minimum_activation_margin_sec",
        ):
            value = _finite_non_negative(getattr(self, name), name=name)
            if name in {"selected_keyframe_period_sec", "reference_period_sec"} and (
                value <= 0.0
            ):
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)

        if self.activation_mode == "immediate_target":
            expected = {
                "required_admission_deadline": "not_applicable",
                "first_segment_policy": "not_applicable",
                "late_admission_policy": "reject_expired_target",
                "reference_tick_alignment_policy": "not_applicable",
                "anchor_commit_policy": "receiver_acceptance",
                "replacement_policy": "latest_mailbox_target",
                "late_tick_recovery_policy": "not_applicable",
            }
            if not math.isclose(self.target_lead_time_sec, 0.0, abs_tol=1e-12):
                raise ValueError("Immediate activation requires zero target lead.")
            if not math.isclose(
                self.minimum_activation_margin_sec,
                0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("Immediate activation cannot declare a future margin.")
        else:
            expected = {
                "required_admission_deadline": "before_segment_start",
                "first_segment_policy": "fresh_measured_rebase_next_reference_tick",
                "late_admission_policy": (
                    SCHEDULED_REFERENCE_LATE_ADMISSION_REJECTION_V1
                ),
                "reference_tick_alignment_policy": "next_reference_tick_strict",
                "anchor_commit_policy": "endpoint_effective_time",
                "replacement_policy": "explicit_supersede",
                "late_tick_recovery_policy": (
                    SCHEDULED_REFERENCE_LATE_TICK_RECOVERY_V1
                ),
            }
            minimum_lead = (
                self.selected_keyframe_period_sec
                + self.minimum_activation_margin_sec
            )
            if self.target_lead_time_sec + 1e-12 < minimum_lead:
                raise ValueError(
                    "Scheduled activation target lead is below keyframe period plus margin."
                )
            if self.minimum_activation_margin_sec + 1e-12 < (
                self.reference_period_sec
            ):
                raise ValueError(
                    "Scheduled activation margin must cover at least one reference period."
                )
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"{self.activation_mode} requires {name}={value}."
                )


def _build_reference_activation_spec(
    *,
    spec_id: str,
    activation_mode: str,
    schedule: DecisionScheduleSpec,
    capability: ExecutionCapabilityProfile,
    minimum_activation_margin_sec: float,
    activation_margin_evidence_id: str,
) -> ReferenceActivationSpec:
    if not isinstance(schedule, DecisionScheduleSpec):
        raise TypeError("schedule must be a DecisionScheduleSpec.")
    if not isinstance(capability, ExecutionCapabilityProfile):
        raise TypeError("capability must be an ExecutionCapabilityProfile.")
    immediate = activation_mode == "immediate_target"
    payload = {
        "spec_id": _required_text(spec_id, name="spec_id"),
        "activation_mode": activation_mode,
        "decision_schedule_spec_hash": schedule.semantic_content_hash,
        "execution_capability_profile_hash": capability.semantic_content_hash,
        "target_lead_time_sec": float(schedule.target_lead_time_sec),
        "selected_keyframe_period_sec": float(schedule.period_sec),
        "reference_period_sec": 1.0 / float(capability.reference_resample_hz),
        "minimum_activation_margin_sec": float(minimum_activation_margin_sec),
        "activation_margin_evidence_id": _required_text(
            activation_margin_evidence_id,
            name="activation_margin_evidence_id",
        ),
        "required_admission_deadline": (
            "not_applicable" if immediate else "before_segment_start"
        ),
        "first_segment_policy": (
            "not_applicable"
            if immediate
            else "fresh_measured_rebase_next_reference_tick"
        ),
        "late_admission_policy": (
            "reject_expired_target"
            if immediate
            else SCHEDULED_REFERENCE_LATE_ADMISSION_REJECTION_V1
        ),
        "reference_tick_alignment_policy": (
            "not_applicable" if immediate else "next_reference_tick_strict"
        ),
        "anchor_commit_policy": (
            "receiver_acceptance" if immediate else "endpoint_effective_time"
        ),
        "replacement_policy": (
            "latest_mailbox_target" if immediate else "explicit_supersede"
        ),
        "late_tick_recovery_policy": (
            "not_applicable"
            if immediate
            else SCHEDULED_REFERENCE_LATE_TICK_RECOVERY_V1
        ),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=REFERENCE_ACTIVATION_HASH_DOMAIN_V1,
    )
    return ReferenceActivationSpec(
        **payload,
        semantic_content_hash=semantic_hash,
    )


def build_immediate_qref_activation_spec(
    schedule: DecisionScheduleSpec,
    capability: ExecutionCapabilityProfile,
) -> ReferenceActivationSpec:
    return _build_reference_activation_spec(
        spec_id=IMMEDIATE_QREF_SIM_V1,
        activation_mode="immediate_target",
        schedule=schedule,
        capability=capability,
        minimum_activation_margin_sec=0.0,
        activation_margin_evidence_id="immediate-target-no-future-margin-v1",
    )


def build_scheduled_reference_segment_activation_spec(
    schedule: DecisionScheduleSpec,
    capability: ExecutionCapabilityProfile,
    *,
    minimum_activation_margin_sec: float,
    activation_margin_evidence_id: str,
) -> ReferenceActivationSpec:
    return _build_reference_activation_spec(
        spec_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        activation_mode="scheduled_segment",
        schedule=schedule,
        capability=capability,
        minimum_activation_margin_sec=minimum_activation_margin_sec,
        activation_margin_evidence_id=activation_margin_evidence_id,
    )


def validate_reference_activation_binding(
    spec: ReferenceActivationSpec,
    *,
    schedule: DecisionScheduleSpec,
    capability: ExecutionCapabilityProfile,
) -> None:
    """Fail closed when a runtime tries to reuse a spec with another profile."""

    if not isinstance(spec, ReferenceActivationSpec):
        raise TypeError("spec must be a ReferenceActivationSpec.")
    if spec.decision_schedule_spec_hash != schedule.semantic_content_hash:
        raise ValueError("Reference activation schedule binding mismatch.")
    if spec.execution_capability_profile_hash != capability.semantic_content_hash:
        raise ValueError("Reference activation capability binding mismatch.")
    if not math.isclose(
        spec.target_lead_time_sec,
        schedule.target_lead_time_sec,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Reference activation target lead mismatch.")
    if not math.isclose(
        spec.reference_period_sec,
        1.0 / capability.reference_resample_hz,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Reference activation reference-rate mismatch.")


__all__ = [
    "IMMEDIATE_QREF_SIM_V1",
    "REFERENCE_ACTIVATION_HASH_DOMAIN_V1",
    "REFERENCE_ACTIVATION_MODES_V1",
    "SCHEDULED_REFERENCE_SEGMENT_SIM_V1",
    "SCHEDULED_REFERENCE_LATE_ADMISSION_REJECTION_V1",
    "SCHEDULED_REFERENCE_LATE_TICK_RECOVERY_V1",
    "ReferenceActivationSpec",
    "build_immediate_qref_activation_spec",
    "build_scheduled_reference_segment_activation_spec",
    "validate_reference_activation_binding",
]
