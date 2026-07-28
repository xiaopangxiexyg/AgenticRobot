"""Immutable canonical conditioning bundles for model-RTC.

The bundle is the causal handoff between the RTC plan owner and a later model
client.  It contains canonical actions and identities only: no model-space
array, framework tensor, sampler, target receipt, safety verdict, or dispatch
state belongs here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_PREFIX_MASK_SEMANTICS_V1,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    ModelContinuationContract,
    normalize_model_continuation_contract,
)
from robot_motion.relative_action import ACTION_POSE_MODE_ABSOLUTE

from ..action_representation import PolicyActionRepresentationContext
from ..canonical_policy_action import CanonicalPolicyAction
from .continuation import (
    RtcContinuationSelectionOutcome,
    RtcContinuationTimeline,
)


CANONICAL_RTC_CONDITIONING_BUNDLE_SCHEMA_V1 = (
    "canonical_rtc_conditioning_bundle_v1"
)
CANONICAL_RTC_ACTION_CONTENT_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_canonical_action_content.v1"
)
CANONICAL_RTC_CONDITIONING_BUNDLE_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_canonical_conditioning_bundle.v1"
)

RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1 = (
    "explicit_selected_generation_steps_v1"
)
RTC_PREFIX_SCHEDULE_HARD_DELAY_PREFIX_V1 = "hard_delay_prefix_v1"
RTC_PREFIX_SCHEDULE_SEMANTICS_V1 = (
    RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1,
    RTC_PREFIX_SCHEDULE_HARD_DELAY_PREFIX_V1,
)

RTC_CONDITIONING_MODE_PREFIX_V1 = "conditioned_prefix_v1"
RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1 = (
    "unconditioned_bootstrap_v1"
)
RTC_CONDITIONING_MODES_V1 = (
    RTC_CONDITIONING_MODE_PREFIX_V1,
    RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1,
)
RTC_MODEL_CONTINUATION_RUNTIME_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_model_continuation_runtime_spec.v1"
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _binary_mask(
    values: tuple[float, ...],
    *,
    horizon: int,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) != horizon:
        raise ValueError(
            "prefix_mask must cover the complete prediction horizon."
        )
    if any(
        not math.isfinite(value) or value not in {0.0, 1.0}
        for value in normalized
    ):
        raise ValueError("prefix_mask must contain only finite 0.0/1.0 values.")
    return normalized


@dataclass(frozen=True)
class CanonicalRtcConditioningBundle:
    """One immutable producer-overlap fact bound before model encoding."""

    runtime_session_id: str
    control_domain_id: str
    artifact_hash: str
    timeline: RtcContinuationTimeline
    selection: RtcContinuationSelectionOutcome
    action_representation_context: PolicyActionRepresentationContext
    continuation_contract: ModelContinuationContract
    prefix_mask: tuple[float, ...]
    conditioning_mode: str
    prefix_schedule_semantics: str = (
        RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
    )
    prefix_source_role: str = RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1
    schema: str = CANONICAL_RTC_CONDITIONING_BUNDLE_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != CANONICAL_RTC_CONDITIONING_BUNDLE_SCHEMA_V1:
            raise ValueError(
                f"Unsupported canonical RTC conditioning schema {self.schema!r}."
            )
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "artifact_hash",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        if not isinstance(self.timeline, RtcContinuationTimeline):
            raise TypeError("timeline must be an RtcContinuationTimeline.")
        if not isinstance(self.selection, RtcContinuationSelectionOutcome):
            raise TypeError(
                "selection must be an RtcContinuationSelectionOutcome."
            )
        if not isinstance(
            self.action_representation_context,
            PolicyActionRepresentationContext,
        ):
            raise TypeError(
                "action_representation_context must be a "
                "PolicyActionRepresentationContext."
            )
        normalized_contract = normalize_model_continuation_contract(
            self.continuation_contract
        )
        if (
            normalized_contract.runtime.continuation_mode
            != MODEL_CONTINUATION_FLOW_INPAINT_V1
        ):
            raise ValueError(
                "Canonical RTC conditioning v1 requires flow_inpaint."
            )

        mode = _nonempty(self.conditioning_mode, name="conditioning_mode")
        if mode not in RTC_CONDITIONING_MODES_V1:
            raise ValueError(f"Unsupported conditioning_mode {mode!r}.")
        schedule = _nonempty(
            self.prefix_schedule_semantics,
            name="prefix_schedule_semantics",
        )
        if schedule not in RTC_PREFIX_SCHEDULE_SEMANTICS_V1:
            raise ValueError(
                f"Unsupported prefix_schedule_semantics {schedule!r}."
            )
        role = _nonempty(self.prefix_source_role, name="prefix_source_role")
        contract = normalized_contract.runtime
        timeline = self.timeline
        context = self.action_representation_context
        selection = self.selection

        if role != RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1:
            raise ValueError(
                "Canonical RTC conditioning v1 only accepts "
                "producer_local_overlap_v1."
            )
        if role not in contract.runtime_prefix_source_roles:
            raise ValueError(
                "prefix_source_role is not supported by the continuation contract."
            )
        if (
            contract.prefix_mask_semantics
            != MODEL_CONTINUATION_PREFIX_MASK_SEMANTICS_V1
        ):
            raise ValueError("Continuation prefix-mask semantics mismatch.")
        if (
            timeline.prediction_horizon_steps
            != contract.prediction_horizon_steps
        ):
            raise ValueError(
                "Timeline prediction horizon does not match the continuation "
                "contract."
            )
        if not (
            contract.minimum_forecast_delay_steps
            <= timeline.forecast_delay_steps
            <= contract.maximum_forecast_delay_steps
        ):
            raise ValueError(
                "Timeline forecast delay is outside the continuation contract."
            )
        if selection.timeline_hash != timeline.semantic_content_hash:
            raise ValueError("Selection/timeline identity mismatch.")

        # C1 v1 is deliberately narrower than the schema reserved for future
        # representations. Relative continuation needs an independent codec and
        # round-trip gate before it can be admitted.
        if context.source_pose_mode != ACTION_POSE_MODE_ABSOLUTE:
            raise ValueError(
                "Canonical RTC conditioning v1 supports absolute actions only."
            )
        if context.source_pose_mode not in contract.supported_action_pose_modes:
            raise ValueError(
                "Action representation is not supported by the continuation "
                "contract."
            )

        prefix_mask = _binary_mask(
            self.prefix_mask,
            horizon=timeline.prediction_horizon_steps,
        )
        selected_indices: tuple[int, ...] = ()
        if selection.accepted:
            if mode != RTC_CONDITIONING_MODE_PREFIX_V1:
                raise ValueError(
                    "An accepted continuation slice requires conditioned_prefix_v1."
                )
            continuation_slice = selection.continuation_slice
            assert continuation_slice is not None
            selected_indices = continuation_slice.selected_generation_indices
            expected_mask = tuple(
                1.0 if index in selected_indices else 0.0
                for index in range(timeline.prediction_horizon_steps)
            )
            if prefix_mask != expected_mask:
                raise ValueError(
                    "prefix_mask must identify exactly the selected canonical "
                    "generation steps."
                )
            selected_count = len(selected_indices)
            if not (
                contract.minimum_prefix_steps
                <= selected_count
                <= contract.maximum_prefix_steps
            ):
                raise ValueError(
                    "Selected prefix length is outside the continuation contract."
                )
            if selected_indices[-1] >= timeline.forecast_delay_steps:
                raise ValueError(
                    "Conditioning prefix cannot enter the executable suffix."
                )
            if schedule == RTC_PREFIX_SCHEDULE_HARD_DELAY_PREFIX_V1:
                expected_indices = tuple(
                    range(timeline.forecast_delay_steps)
                )
                if selected_indices != expected_indices:
                    raise ValueError(
                        "hard_delay_prefix_v1 requires every delayed generation "
                        "step in order."
                    )
            self._validate_selected_actions()
        else:
            if not selection.bootstrap_required:
                raise ValueError(
                    "Only an explicit selector bootstrap outcome may omit a prefix."
                )
            if mode != RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1:
                raise ValueError(
                    "A bootstrap selection requires "
                    "unconditioned_bootstrap_v1."
                )
            if not contract.supports_unconditioned_bootstrap:
                raise ValueError(
                    "The continuation contract does not support unconditioned "
                    "bootstrap."
                )
            if any(prefix_mask):
                raise ValueError(
                    "Unconditioned bootstrap requires an all-zero full-horizon "
                    "prefix mask."
                )

        object.__setattr__(self, "conditioning_mode", mode)
        object.__setattr__(self, "prefix_schedule_semantics", schedule)
        object.__setattr__(self, "prefix_source_role", role)
        object.__setattr__(self, "prefix_mask", prefix_mask)
        object.__setattr__(
            self,
            "schema",
            CANONICAL_RTC_CONDITIONING_BUNDLE_SCHEMA_V1,
        )

    def _validate_selected_actions(self) -> None:
        continuation_slice = self.selection.continuation_slice
        assert continuation_slice is not None
        contract = normalize_model_continuation_contract(
            self.continuation_contract
        ).runtime
        for entry in continuation_slice.entries:
            keyframe = entry.keyframe
            if keyframe.artifact_hash != self.artifact_hash:
                raise ValueError(
                    "Selected continuation action artifact identity mismatch."
                )
            action = keyframe.canonical_action
            assert action is not None
            receipt = action.canonicalization_receipt
            expected = {
                "action_codec_hash": contract.action_codec_hash,
                "frame_contract_hash": contract.frame_contract_hash,
                "layout_hash": contract.layout_hash,
                "normalization_hash": contract.normalization_hash,
            }
            for name, expected_value in expected.items():
                if getattr(receipt, name) != expected_value:
                    raise ValueError(
                        f"Selected continuation action {name} mismatch."
                    )
            if receipt.source_pose_mode != ACTION_POSE_MODE_ABSOLUTE:
                raise ValueError(
                    "Canonical RTC conditioning v1 cannot encode a selected "
                    "relative-source action."
                )

    @property
    def selected_generation_indices(self) -> tuple[int, ...]:
        continuation_slice = self.selection.continuation_slice
        if continuation_slice is None:
            return ()
        return continuation_slice.selected_generation_indices

    @property
    def canonical_actions(self) -> tuple[CanonicalPolicyAction, ...]:
        continuation_slice = self.selection.continuation_slice
        if continuation_slice is None:
            return ()
        actions = tuple(
            entry.keyframe.canonical_action
            for entry in continuation_slice.entries
        )
        assert all(action is not None for action in actions)
        return actions  # type: ignore[return-value]

    @property
    def canonical_action_content_hash(self) -> str:
        return canonical_content_hash(
            tuple(
                action.canonical_action_hash
                for action in self.canonical_actions
            ),
            domain=CANONICAL_RTC_ACTION_CONTENT_HASH_DOMAIN_V1,
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "artifact_hash": self.artifact_hash,
                "timeline_hash": self.timeline.semantic_content_hash,
                "selection_hash": self.selection.semantic_content_hash,
                "action_representation_context_hash": (
                    self.action_representation_context.semantic_content_hash
                ),
                "continuation_contract_hash": (
                    self.continuation_contract.contract_hash
                ),
                "prefix_mask": self.prefix_mask,
                "conditioning_mode": self.conditioning_mode,
                "prefix_schedule_semantics": (
                    self.prefix_schedule_semantics
                ),
                "prefix_source_role": self.prefix_source_role,
                "canonical_action_content_hash": (
                    self.canonical_action_content_hash
                ),
            },
            domain=CANONICAL_RTC_CONDITIONING_BUNDLE_HASH_DOMAIN_V1,
        )

    @property
    def conditioning_binding_hash(self) -> str:
        return f"sha256:{self.semantic_content_hash}"


@dataclass(frozen=True)
class RtcModelContinuationRuntimeSpec:
    """Target-neutral online selection policy bound to an artifact contract."""

    continuation_contract: ModelContinuationContract
    prefix_schedule_semantics: str = (
        RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
    )
    lattice_tolerance_sec: float = 1e-9

    def __post_init__(self) -> None:
        contract = normalize_model_continuation_contract(
            self.continuation_contract
        ).runtime
        if contract.continuation_mode != MODEL_CONTINUATION_FLOW_INPAINT_V1:
            raise ValueError(
                "Model-RTC C1 runtime requires flow_inpaint."
            )
        schedule = _nonempty(
            self.prefix_schedule_semantics,
            name="prefix_schedule_semantics",
        )
        if schedule not in RTC_PREFIX_SCHEDULE_SEMANTICS_V1:
            raise ValueError(
                f"Unsupported prefix_schedule_semantics {schedule!r}."
            )
        tolerance = float(self.lattice_tolerance_sec)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "lattice_tolerance_sec must be finite and non-negative."
            )
        object.__setattr__(self, "prefix_schedule_semantics", schedule)
        object.__setattr__(self, "lattice_tolerance_sec", tolerance)

    def selection_spec_for_timeline(
        self,
        timeline: RtcContinuationTimeline,
    ):
        from .continuation import RtcContinuationSelectionSpec

        if not isinstance(timeline, RtcContinuationTimeline):
            raise TypeError("timeline must be an RtcContinuationTimeline.")
        contract = normalize_model_continuation_contract(
            self.continuation_contract
        ).runtime
        if (
            self.prefix_schedule_semantics
            == RTC_PREFIX_SCHEDULE_HARD_DELAY_PREFIX_V1
        ):
            prefix_horizon = timeline.forecast_delay_steps
        else:
            prefix_horizon = min(
                timeline.forecast_delay_steps,
                contract.maximum_prefix_steps,
            )
        if prefix_horizon <= 0:
            raise ValueError(
                "Model-RTC continuation requires a positive forecast delay."
            )
        minimum = max(1, contract.minimum_prefix_steps)
        if minimum > prefix_horizon:
            minimum = prefix_horizon
        return RtcContinuationSelectionSpec(
            prefix_horizon_steps=prefix_horizon,
            minimum_selected_steps=minimum,
            lattice_tolerance_sec=self.lattice_tolerance_sec,
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "continuation_contract_hash": (
                    self.continuation_contract.contract_hash
                ),
                "prefix_schedule_semantics": (
                    self.prefix_schedule_semantics
                ),
                "lattice_tolerance_sec": self.lattice_tolerance_sec,
            },
            domain=RTC_MODEL_CONTINUATION_RUNTIME_SPEC_HASH_DOMAIN_V1,
        )


def build_canonical_rtc_conditioning_bundle(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    artifact_hash: str,
    timeline: RtcContinuationTimeline,
    selection: RtcContinuationSelectionOutcome,
    action_representation_context: PolicyActionRepresentationContext,
    continuation_contract: ModelContinuationContract,
    prefix_schedule_semantics: str = (
        RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
    ),
) -> CanonicalRtcConditioningBundle:
    """Derive the only valid full-horizon mask from a selector outcome."""

    if selection.accepted:
        continuation_slice = selection.continuation_slice
        assert continuation_slice is not None
        selected = set(continuation_slice.selected_generation_indices)
        prefix_mask = tuple(
            1.0 if index in selected else 0.0
            for index in range(timeline.prediction_horizon_steps)
        )
        mode = RTC_CONDITIONING_MODE_PREFIX_V1
    else:
        prefix_mask = (0.0,) * timeline.prediction_horizon_steps
        mode = RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1
    return CanonicalRtcConditioningBundle(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        artifact_hash=artifact_hash,
        timeline=timeline,
        selection=selection,
        action_representation_context=action_representation_context,
        continuation_contract=continuation_contract,
        prefix_mask=prefix_mask,
        conditioning_mode=mode,
        prefix_schedule_semantics=prefix_schedule_semantics,
    )
