"""Pure policy-output contracts for deployment action semantics.

These declarations are additive to the continuous/discrete action tensors in
``PolicyArtifactContract``.  They must never be encoded by extending the
continuous action dimension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .arm_mode import (
    ARM_HOLD_POSITIVE_SCOPE_ARM_AND_END_EFFECTOR_V1,
    ARM_MODE_SIDE_ORDER_V1,
    DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1,
)
from .canonical_encoding import canonical_content_hash
from .v22 import DatasetLayoutBindingError


POLICY_ARM_HOLD_OUTPUT_SCHEMA_V1 = "dexteleop_policy_arm_hold_output_v1"
POLICY_ARM_HOLD_OUTPUT_METADATA_KEY = "arm_hold_output_contract"
POLICY_ARM_HOLD_OUTPUT_HASH_KEY = "arm_hold_output_contract_hash"
POLICY_ARM_HOLD_OUTPUT_HASH_DOMAIN_V1 = (
    "dexteleop.policy_arm_hold_output.semantic.v1"
)
POLICY_ARM_HOLD_SCORE_SEMANTICS_PROBABILITY_V1 = "hold_probability_v1"
POLICY_ARM_HOLD_ALIGNMENT_ACTION_CHUNK_V1 = "same_action_keyframe_v1"
POLICY_ARM_HOLD_SUPPORTED_COMBINATIONS_V1 = ("00", "01", "10", "11")


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise DatasetLayoutBindingError(f"{name} must be non-empty.")
    return text


def _required_probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise DatasetLayoutBindingError(
            f"{name} must be a finite probability."
        )
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetLayoutBindingError(
            f"{name} must be a finite probability."
        ) from exc
    if not 0.0 <= probability <= 1.0:
        raise DatasetLayoutBindingError(f"{name} must be in [0, 1].")
    return probability


def _required_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise DatasetLayoutBindingError(f"{name} must be a positive integer.")
    try:
        count = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetLayoutBindingError(
            f"{name} must be a positive integer."
        ) from exc
    if not math.isfinite(numeric) or numeric != float(count) or count <= 0:
        raise DatasetLayoutBindingError(
            f"{name} must be a positive integer."
        )
    return count


@dataclass(frozen=True)
class PolicyArmHoldOutputContract:
    """Artifact-bound output and state-resolution contract for arm hold.

    Thresholds and debounce counts are deliberately required.  A deployment
    must calibrate them for its checkpoint instead of inheriting a hidden
    runtime default.
    """

    output_key: str
    training_contract_hash: str
    enter_threshold: float
    exit_threshold: float
    enter_consecutive_steps: int
    exit_consecutive_steps: int
    side_order: tuple[str, ...] = ARM_MODE_SIDE_ORDER_V1
    score_semantics: str = POLICY_ARM_HOLD_SCORE_SEMANTICS_PROBABILITY_V1
    target_alignment: str = POLICY_ARM_HOLD_ALIGNMENT_ACTION_CHUNK_V1
    positive_scope: str = ARM_HOLD_POSITIVE_SCOPE_ARM_AND_END_EFFECTOR_V1
    supported_combinations: tuple[str, ...] = (
        POLICY_ARM_HOLD_SUPPORTED_COMBINATIONS_V1
    )
    schema: str = POLICY_ARM_HOLD_OUTPUT_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != POLICY_ARM_HOLD_OUTPUT_SCHEMA_V1:
            raise DatasetLayoutBindingError(
                f"Unsupported arm-hold output schema {self.schema!r}."
            )
        output_key = _required_text(self.output_key, name="output_key")
        training_hash = _required_text(
            self.training_contract_hash,
            name="training_contract_hash",
        )
        digest = (
            training_hash.removeprefix("sha256:")
            if training_hash.startswith("sha256:")
            else training_hash
        )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise DatasetLayoutBindingError(
                "training_contract_hash must be a canonical sha256 hash."
            )
        if (
            training_hash
            != DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1.semantic_content_hash
        ):
            raise DatasetLayoutBindingError(
                "arm-hold output v1 must bind the stable arm_hold_intent_v1 "
                "training contract."
            )
        enter = _required_probability(
            self.enter_threshold,
            name="enter_threshold",
        )
        exit_ = _required_probability(
            self.exit_threshold,
            name="exit_threshold",
        )
        if enter <= exit_:
            raise DatasetLayoutBindingError(
                "enter_threshold must be greater than exit_threshold."
            )
        enter_steps = _required_positive_int(
            self.enter_consecutive_steps,
            name="enter_consecutive_steps",
        )
        exit_steps = _required_positive_int(
            self.exit_consecutive_steps,
            name="exit_consecutive_steps",
        )
        sides = tuple(str(side).strip().lower() for side in self.side_order)
        if sides != ARM_MODE_SIDE_ORDER_V1:
            raise DatasetLayoutBindingError(
                f"arm-hold side_order must be {ARM_MODE_SIDE_ORDER_V1!r}."
            )
        combinations = tuple(str(value) for value in self.supported_combinations)
        if combinations != POLICY_ARM_HOLD_SUPPORTED_COMBINATIONS_V1:
            raise DatasetLayoutBindingError(
                "arm-hold v1 must explicitly support 00, 01, 10, and 11."
            )
        if (
            self.score_semantics
            != POLICY_ARM_HOLD_SCORE_SEMANTICS_PROBABILITY_V1
        ):
            raise DatasetLayoutBindingError(
                "arm-hold v1 accepts probabilities, not logits."
            )
        if self.target_alignment != POLICY_ARM_HOLD_ALIGNMENT_ACTION_CHUNK_V1:
            raise DatasetLayoutBindingError(
                "arm-hold v1 scores must align with action keyframes."
            )
        if (
            self.positive_scope
            != ARM_HOLD_POSITIVE_SCOPE_ARM_AND_END_EFFECTOR_V1
        ):
            raise DatasetLayoutBindingError(
                "arm-hold v1 positive scope is arm_and_end_effector."
            )
        object.__setattr__(self, "output_key", output_key)
        object.__setattr__(self, "training_contract_hash", training_hash)
        object.__setattr__(self, "enter_threshold", enter)
        object.__setattr__(self, "exit_threshold", exit_)
        object.__setattr__(self, "enter_consecutive_steps", enter_steps)
        object.__setattr__(self, "exit_consecutive_steps", exit_steps)
        object.__setattr__(self, "side_order", sides)
        object.__setattr__(self, "supported_combinations", combinations)

    def to_dict(self, *, include_hash: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "output_key": self.output_key,
            "training_contract_hash": self.training_contract_hash,
            "side_order": list(self.side_order),
            "score_semantics": self.score_semantics,
            "target_alignment": self.target_alignment,
            "positive_scope": self.positive_scope,
            "supported_combinations": list(self.supported_combinations),
            "enter_threshold": self.enter_threshold,
            "exit_threshold": self.exit_threshold,
            "enter_consecutive_steps": self.enter_consecutive_steps,
            "exit_consecutive_steps": self.exit_consecutive_steps,
        }
        if include_hash:
            payload["contract_hash"] = self.contract_hash
        return payload

    @property
    def contract_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=POLICY_ARM_HOLD_OUTPUT_HASH_DOMAIN_V1,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PolicyArmHoldOutputContract":
        if not isinstance(payload, Mapping):
            raise DatasetLayoutBindingError(
                "arm_hold_output_contract must be a mapping."
            )
        required = (
            "output_key",
            "training_contract_hash",
            "enter_threshold",
            "exit_threshold",
            "enter_consecutive_steps",
            "exit_consecutive_steps",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise DatasetLayoutBindingError(
                "arm_hold_output_contract is missing required fields: "
                + ", ".join(missing)
            )
        contract = cls(
            schema=str(payload.get("schema", "")),
            output_key=str(payload["output_key"]),
            training_contract_hash=str(payload["training_contract_hash"]),
            side_order=tuple(payload.get("side_order", ())),
            score_semantics=str(payload.get("score_semantics", "")),
            target_alignment=str(payload.get("target_alignment", "")),
            positive_scope=str(payload.get("positive_scope", "")),
            supported_combinations=tuple(
                payload.get("supported_combinations", ())
            ),
            enter_threshold=payload["enter_threshold"],
            exit_threshold=payload["exit_threshold"],
            enter_consecutive_steps=payload["enter_consecutive_steps"],
            exit_consecutive_steps=payload["exit_consecutive_steps"],
        )
        declared_hash = payload.get("contract_hash")
        if declared_hash is not None and str(declared_hash) != contract.contract_hash:
            raise DatasetLayoutBindingError(
                "arm_hold_output_contract contract_hash does not match."
            )
        return contract


__all__ = [
    "POLICY_ARM_HOLD_ALIGNMENT_ACTION_CHUNK_V1",
    "POLICY_ARM_HOLD_OUTPUT_HASH_DOMAIN_V1",
    "POLICY_ARM_HOLD_OUTPUT_HASH_KEY",
    "POLICY_ARM_HOLD_OUTPUT_METADATA_KEY",
    "POLICY_ARM_HOLD_OUTPUT_SCHEMA_V1",
    "POLICY_ARM_HOLD_SCORE_SEMANTICS_PROBABILITY_V1",
    "POLICY_ARM_HOLD_SUPPORTED_COMBINATIONS_V1",
    "PolicyArmHoldOutputContract",
]
