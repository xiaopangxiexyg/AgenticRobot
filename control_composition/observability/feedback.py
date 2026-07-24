"""Bounded non-authoritative guidance feedback derived from source differences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash

from .difference import SourceDifferenceObservation


GUIDANCE_FEEDBACK_SPEC_HASH_DOMAIN_V1 = "dexteleop.guidance_feedback_spec.v1"
GUIDANCE_FEEDBACK_COMMAND_HASH_DOMAIN_V1 = (
    "dexteleop.guidance_feedback_command.v1"
)
GUIDANCE_FEEDBACK_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.guidance_feedback_receipt.v1"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite_positive(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True)
class GuidanceFeedbackSpec:
    spec_id: str
    compared_role: str
    reference_role: str
    translation_full_scale_m: float
    rotation_full_scale_rad: float
    max_intensity: float = 1.0
    command_ttl_sec: float = 0.1

    def __post_init__(self) -> None:
        for name in ("spec_id", "compared_role", "reference_role"):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        if self.compared_role == self.reference_role:
            raise ValueError("Guidance feedback roles must be different.")
        object.__setattr__(
            self,
            "translation_full_scale_m",
            _finite_positive(
                self.translation_full_scale_m,
                name="translation_full_scale_m",
            ),
        )
        object.__setattr__(
            self,
            "rotation_full_scale_rad",
            _finite_positive(
                self.rotation_full_scale_rad,
                name="rotation_full_scale_rad",
            ),
        )
        intensity = float(self.max_intensity)
        if not math.isfinite(intensity) or not 0.0 < intensity <= 1.0:
            raise ValueError("max_intensity must be in (0, 1].")
        object.__setattr__(self, "max_intensity", intensity)
        object.__setattr__(
            self,
            "command_ttl_sec",
            _finite_positive(self.command_ttl_sec, name="command_ttl_sec"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "spec_id": self.spec_id,
                "compared_role": self.compared_role,
                "reference_role": self.reference_role,
                "translation_full_scale_m": self.translation_full_scale_m,
                "rotation_full_scale_rad": self.rotation_full_scale_rad,
                "max_intensity": self.max_intensity,
                "command_ttl_sec": self.command_ttl_sec,
                "feedback_kind": "guidance_haptic_non_force",
            },
            domain=GUIDANCE_FEEDBACK_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class GuidanceFeedbackCue:
    component_id: str
    translation_direction_xyz: tuple[float, ...]
    translation_intensity: float
    rotation_axis_xyz: tuple[float, ...]
    rotation_intensity: float


@dataclass(frozen=True)
class GuidanceFeedbackCommand:
    command_id: str
    spec_hash: str
    source_observation_id: str
    source_observation_hash: str
    sequence: int
    status: str
    reason: str
    produced_at: float
    valid_until: float
    clock_domain_id: str
    cues: tuple[GuidanceFeedbackCue, ...]
    feedback_kind: str
    authoritative: bool
    semantic_content_hash: str


@dataclass(frozen=True)
class GuidanceFeedbackReceipt:
    receipt_id: str
    command_id: str
    command_hash: str
    adapter_id: str
    accepted: bool
    status: str
    observed_at: float
    clock_domain_id: str
    semantic_content_hash: str


class GuidanceFeedbackAdapter(Protocol):
    """Device-neutral sink boundary; implementations must return a receipt."""

    adapter_id: str

    def submit(
        self,
        command: GuidanceFeedbackCommand,
        *,
        observed_at: float,
        clock_domain_id: str,
    ) -> GuidanceFeedbackReceipt: ...


def _unit_or_zero(values: tuple[float, ...]) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return tuple(0.0 for _ in values)
    return tuple(float(value) for value in vector / norm)


def build_guidance_feedback_command(
    observation: SourceDifferenceObservation,
    *,
    spec: GuidanceFeedbackSpec,
    sequence: int,
    produced_at: float,
    clock_domain_id: str,
    clear: bool = False,
    reason: str = "source_difference_guidance",
) -> GuidanceFeedbackCommand:
    """Map one difference pair to bounded dimensionless guidance cues."""

    if isinstance(sequence, bool) or int(sequence) != sequence or int(sequence) < 0:
        raise ValueError("sequence must be a non-negative integer.")
    produced = float(produced_at)
    if not math.isfinite(produced):
        raise ValueError("produced_at must be finite.")
    clock = _required(clock_domain_id, name="clock_domain_id")
    if observation.clock_domain_id != clock:
        raise ValueError("Guidance command/observation clock mismatch.")
    normalized_reason = _required(reason, name="reason")
    cues: tuple[GuidanceFeedbackCue, ...] = ()
    status = "clear" if clear else "active"
    if not clear:
        matches = tuple(
            difference
            for difference in observation.pairwise_differences
            if difference.compared_role == spec.compared_role
            and difference.reference_role == spec.reference_role
        )
        if len(matches) != 1:
            raise ValueError(
                "Guidance feedback requires exactly one configured source pair."
            )
        built: list[GuidanceFeedbackCue] = []
        for metric in matches[0].component_metrics:
            translation_intensity = 0.0
            rotation_intensity = 0.0
            translation_direction: tuple[float, ...] = ()
            rotation_axis: tuple[float, ...] = ()
            if metric.representation == "qpos":
                raise ValueError(
                    "guidance_qpos_mapping_unsupported_v1:"
                    f"{metric.component_id}"
                )
            if metric.translation_norm_m is not None:
                translation_intensity = min(
                    spec.max_intensity,
                    metric.translation_norm_m
                    / spec.translation_full_scale_m,
                )
                translation_direction = _unit_or_zero(
                    metric.translation_delta_xyz_m
                )
            if metric.rotation_angle_rad is not None:
                rotation_intensity = min(
                    spec.max_intensity,
                    metric.rotation_angle_rad / spec.rotation_full_scale_rad,
                )
                rotation_axis = _unit_or_zero(
                    metric.rotation_delta_rotvec_rad
                )
            built.append(
                GuidanceFeedbackCue(
                    component_id=metric.component_id,
                    translation_direction_xyz=translation_direction,
                    translation_intensity=translation_intensity,
                    rotation_axis_xyz=rotation_axis,
                    rotation_intensity=rotation_intensity,
                )
            )
        cues = tuple(built)
    valid_until = (
        produced + spec.command_ttl_sec
        if clear
        else min(
            observation.valid_until,
            produced + spec.command_ttl_sec,
        )
    )
    if not clear and valid_until <= produced:
        raise ValueError("Cannot build active guidance from a stale observation.")
    payload = {
        "spec_hash": spec.semantic_content_hash,
        "source_observation_id": observation.observation_id,
        "source_observation_hash": observation.semantic_content_hash,
        "sequence": int(sequence),
        "status": status,
        "reason": normalized_reason,
        "produced_at": produced,
        "valid_until": valid_until,
        "clock_domain_id": clock,
        "cues": tuple(
            {
                "component_id": cue.component_id,
                "translation_direction_xyz": cue.translation_direction_xyz,
                "translation_intensity": cue.translation_intensity,
                "rotation_axis_xyz": cue.rotation_axis_xyz,
                "rotation_intensity": cue.rotation_intensity,
            }
            for cue in cues
        ),
        "feedback_kind": "guidance_haptic_non_force",
        "authoritative": False,
    }
    digest = canonical_content_hash(
        payload,
        domain=GUIDANCE_FEEDBACK_COMMAND_HASH_DOMAIN_V1,
    )
    return GuidanceFeedbackCommand(
        command_id=f"guidance-feedback-command:{digest}",
        cues=cues,
        semantic_content_hash=digest,
        **{key: value for key, value in payload.items() if key != "cues"},
    )


class NoIoGuidanceFeedbackAdapter:
    """Deterministic validation sink; performs no device I/O."""

    def __init__(self, *, adapter_id: str = "guidance-feedback-no-io-v1") -> None:
        self.adapter_id = _required(adapter_id, name="adapter_id")

    def submit(
        self,
        command: GuidanceFeedbackCommand,
        *,
        observed_at: float,
        clock_domain_id: str,
    ) -> GuidanceFeedbackReceipt:
        observed = float(observed_at)
        if not math.isfinite(observed):
            raise ValueError("observed_at must be finite.")
        clock = _required(clock_domain_id, name="clock_domain_id")
        if command.clock_domain_id != clock:
            accepted = False
            status = "clock_domain_mismatch"
        elif observed >= command.valid_until:
            accepted = False
            status = "stale"
        else:
            accepted = True
            status = "validated_no_io"
        payload = {
            "command_id": command.command_id,
            "command_hash": command.semantic_content_hash,
            "adapter_id": self.adapter_id,
            "accepted": accepted,
            "status": status,
            "observed_at": observed,
            "clock_domain_id": clock,
        }
        digest = canonical_content_hash(
            payload,
            domain=GUIDANCE_FEEDBACK_RECEIPT_HASH_DOMAIN_V1,
        )
        return GuidanceFeedbackReceipt(
            receipt_id=f"guidance-feedback-receipt:{digest}",
            semantic_content_hash=digest,
            **payload,
        )


__all__ = [
    "GUIDANCE_FEEDBACK_COMMAND_HASH_DOMAIN_V1",
    "GUIDANCE_FEEDBACK_RECEIPT_HASH_DOMAIN_V1",
    "GUIDANCE_FEEDBACK_SPEC_HASH_DOMAIN_V1",
    "GuidanceFeedbackCommand",
    "GuidanceFeedbackAdapter",
    "GuidanceFeedbackCue",
    "GuidanceFeedbackReceipt",
    "GuidanceFeedbackSpec",
    "NoIoGuidanceFeedbackAdapter",
    "build_guidance_feedback_command",
]
