"""Explicit same-target temporal selection for composition inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash


COMPOSITION_TEMPORAL_SELECTION_SPEC_HASH_DOMAIN = (
    "dexteleop.composition_temporal_selection_spec.development"
)
COMPOSITION_OVERLAP_BEHAVIORS = frozenset(
    {
        "latest_effective_start_then_receiver_admission",
        "reject_multiple_covering",
    }
)
COMPOSITION_ACTION_TIMESTAMP_SEMANTICS = frozenset({"lineage_only"})


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _optional_nonnegative(
    value: float | None,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative or None.")
    return result


@dataclass(frozen=True)
class CompositionTemporalSelectionSpec:
    """How one cut selects source facts for one target-effective time.

    This is selection, not interpolation. ``RobotAction.timestamp`` remains
    source lineage and never participates in ranking.
    """

    spec_id: str
    overlap_behavior: str = (
        "latest_effective_start_then_receiver_admission"
    )
    maximum_target_lead_sec: float | None = None
    maximum_target_lag_sec: float | None = None
    action_timestamp_semantics: str = "lineage_only"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "spec_id",
            _required(self.spec_id, name="spec_id"),
        )
        overlap = _required(
            self.overlap_behavior,
            name="overlap_behavior",
        )
        if overlap not in COMPOSITION_OVERLAP_BEHAVIORS:
            raise ValueError("Unsupported composition overlap behavior.")
        timestamp_semantics = _required(
            self.action_timestamp_semantics,
            name="action_timestamp_semantics",
        )
        if timestamp_semantics not in COMPOSITION_ACTION_TIMESTAMP_SEMANTICS:
            raise ValueError("Unsupported action timestamp semantics.")
        object.__setattr__(self, "overlap_behavior", overlap)
        object.__setattr__(
            self,
            "maximum_target_lead_sec",
            _optional_nonnegative(
                self.maximum_target_lead_sec,
                name="maximum_target_lead_sec",
            ),
        )
        object.__setattr__(
            self,
            "maximum_target_lag_sec",
            _optional_nonnegative(
                self.maximum_target_lag_sec,
                name="maximum_target_lag_sec",
            ),
        )
        object.__setattr__(
            self,
            "action_timestamp_semantics",
            timestamp_semantics,
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "spec_id": self.spec_id,
                "selection": "same_target_effective_interval_cover",
                "overlap_behavior": self.overlap_behavior,
                "maximum_target_lead_sec": self.maximum_target_lead_sec,
                "maximum_target_lag_sec": self.maximum_target_lag_sec,
                "action_timestamp_semantics": (
                    self.action_timestamp_semantics
                ),
            },
            domain=COMPOSITION_TEMPORAL_SELECTION_SPEC_HASH_DOMAIN,
        )

    def validate_target(self, *, cutoff_at: float, target_effective_at: float) -> None:
        cutoff = float(cutoff_at)
        target = float(target_effective_at)
        if not math.isfinite(cutoff) or not math.isfinite(target):
            raise ValueError("Composition cut and target times must be finite.")
        if (
            self.maximum_target_lead_sec is not None
            and target > cutoff + self.maximum_target_lead_sec + 1e-12
        ):
            raise ValueError(
                "target_effective_at exceeds the temporal selection lead bound."
            )
        if (
            self.maximum_target_lag_sec is not None
            and target < cutoff - self.maximum_target_lag_sec - 1e-12
        ):
            raise ValueError(
                "target_effective_at exceeds the temporal selection lag bound."
            )


DEFAULT_COMPOSITION_TEMPORAL_SELECTION_SPEC = CompositionTemporalSelectionSpec(
    spec_id="same-target-latest-effective-start-development",
)


__all__ = [
    "COMPOSITION_ACTION_TIMESTAMP_SEMANTICS",
    "COMPOSITION_OVERLAP_BEHAVIORS",
    "COMPOSITION_TEMPORAL_SELECTION_SPEC_HASH_DOMAIN",
    "CompositionTemporalSelectionSpec",
    "DEFAULT_COMPOSITION_TEMPORAL_SELECTION_SPEC",
]
