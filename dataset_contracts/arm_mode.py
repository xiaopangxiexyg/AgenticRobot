"""Pure arm-mode storage and hold-training projection contracts.

The recorded ``teleop.arm_mode`` values describe operator/control modes.  They
are not a measured-velocity classifier: ``teleop`` may contain physically
still frames, while ``fixed_hold`` is an explicit per-side hold intent.

This module owns only stable identifiers, metadata validation, and the
declaration of the derived training target.  Reading rows and materializing
NumPy training arrays belongs to :mod:`dataset_interface.arm_hold`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


ARM_MODE_SCHEMA_V2 = "dexteleop_arm_mode_v2"
ARM_MODE_SCHEMA = ARM_MODE_SCHEMA_V2
ARM_MODE_SIDE_ORDER_V1 = ("left", "right")
ARM_MODE_LABEL_SPEC_KEY = "teleop.arm_mode"

ARM_MODE_IDS: dict[str, int] = {
    "unknown": 0,
    "teleop": 1,
    "fixed_hold": 2,
    "init_pose": 3,
    "stopped": 4,
}
ARM_MODE_ID_TO_LABEL = {
    mode_id: label for label, mode_id in ARM_MODE_IDS.items()
}

ARM_HOLD_TRAINING_PROJECTION_ID_V1 = "arm_hold_intent_v1"
ARM_HOLD_TRAINING_CONTRACT_SCHEMA_V1 = (
    "dexteleop.arm_hold_training_contract.v1"
)
ARM_HOLD_TRAINING_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.arm_hold_training_contract.semantic.v1"
)
ARM_HOLD_TARGET_ALIGNMENT_ACTION_INDICES_V1 = (
    "same_indices_as_action_target_chunk_v1"
)
ARM_HOLD_MISSING_LABEL_MASK_INVALID_V1 = "mask_invalid_v1"
ARM_HOLD_POSITIVE_SCOPE_ARM_AND_END_EFFECTOR_V1 = "arm_and_end_effector"


class ArmModeContractError(ValueError):
    """Raised when declared arm-mode metadata contradicts the stable schema."""


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ArmModeContractError(f"{name} must be non-empty.")
    return normalized


def _string_tuple(
    value: object,
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ArmModeContractError(f"{name} must be a list or tuple of strings.")
    normalized = tuple(_nonempty(item, name=name) for item in value)
    if len(normalized) != len(set(normalized)):
        raise ArmModeContractError(f"{name} must not contain duplicates.")
    return normalized


@dataclass(frozen=True)
class ArmHoldTrainingContract:
    """Stable binary projection from recorded arm modes to a hold-intent target."""

    source_schema: str = ARM_MODE_SCHEMA_V2
    side_order: tuple[str, ...] = ARM_MODE_SIDE_ORDER_V1
    positive_modes: tuple[str, ...] = ("fixed_hold",)
    negative_modes: tuple[str, ...] = ("teleop",)
    ignored_modes: tuple[str, ...] = ("unknown", "init_pose", "stopped")
    required_positive_scope: str = (
        ARM_HOLD_POSITIVE_SCOPE_ARM_AND_END_EFFECTOR_V1
    )
    missing_label_policy: str = ARM_HOLD_MISSING_LABEL_MASK_INVALID_V1
    target_alignment: str = ARM_HOLD_TARGET_ALIGNMENT_ACTION_INDICES_V1
    target_dtype: str = "float32"
    schema: str = ARM_HOLD_TRAINING_CONTRACT_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != ARM_HOLD_TRAINING_CONTRACT_SCHEMA_V1:
            raise ArmModeContractError(
                f"Unsupported arm-hold training contract schema {self.schema!r}."
            )
        source_schema = _nonempty(self.source_schema, name="source_schema")
        side_order = _string_tuple(self.side_order, name="side_order")
        positive = _string_tuple(self.positive_modes, name="positive_modes")
        negative = _string_tuple(self.negative_modes, name="negative_modes")
        ignored = _string_tuple(self.ignored_modes, name="ignored_modes")
        if side_order != ARM_MODE_SIDE_ORDER_V1:
            raise ArmModeContractError(
                f"side_order must be {ARM_MODE_SIDE_ORDER_V1!r}."
            )
        known_modes = set(ARM_MODE_IDS)
        declared_modes = set(positive) | set(negative) | set(ignored)
        if declared_modes != known_modes:
            raise ArmModeContractError(
                "positive, negative, and ignored modes must partition "
                f"{sorted(known_modes)!r}."
            )
        if (
            set(positive) & set(negative)
            or set(positive) & set(ignored)
            or set(negative) & set(ignored)
        ):
            raise ArmModeContractError(
                "positive, negative, and ignored modes must be disjoint."
            )
        if self.missing_label_policy != ARM_HOLD_MISSING_LABEL_MASK_INVALID_V1:
            raise ArmModeContractError(
                "The v1 arm-hold contract requires mask_invalid_v1 for "
                "missing labels."
            )
        if self.target_alignment != ARM_HOLD_TARGET_ALIGNMENT_ACTION_INDICES_V1:
            raise ArmModeContractError(
                "The v1 arm-hold contract requires same-index action alignment."
            )
        if self.target_dtype != "float32":
            raise ArmModeContractError("The v1 arm-hold target dtype is float32.")
        object.__setattr__(self, "source_schema", source_schema)
        object.__setattr__(self, "side_order", side_order)
        object.__setattr__(self, "positive_modes", positive)
        object.__setattr__(self, "negative_modes", negative)
        object.__setattr__(self, "ignored_modes", ignored)
        object.__setattr__(
            self,
            "required_positive_scope",
            _nonempty(
                self.required_positive_scope,
                name="required_positive_scope",
            ),
        )

    def to_dict(self, *, include_hash: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "source_schema": self.source_schema,
            "side_order": list(self.side_order),
            "positive_modes": list(self.positive_modes),
            "negative_modes": list(self.negative_modes),
            "ignored_modes": list(self.ignored_modes),
            "required_positive_scope": self.required_positive_scope,
            "missing_label_policy": self.missing_label_policy,
            "target_alignment": self.target_alignment,
            "target_dtype": self.target_dtype,
        }
        if include_hash:
            payload["contract_hash"] = self.semantic_content_hash
        return payload

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=ARM_HOLD_TRAINING_CONTRACT_HASH_DOMAIN_V1,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArmHoldTrainingContract":
        if not isinstance(payload, Mapping):
            raise ArmModeContractError(
                "Arm-hold training contract payload must be a mapping."
            )
        contract = cls(
            schema=str(payload.get("schema", "")),
            source_schema=str(payload.get("source_schema", "")),
            side_order=_string_tuple(
                payload.get("side_order"),
                name="side_order",
            ),
            positive_modes=_string_tuple(
                payload.get("positive_modes"),
                name="positive_modes",
            ),
            negative_modes=_string_tuple(
                payload.get("negative_modes"),
                name="negative_modes",
            ),
            ignored_modes=_string_tuple(
                payload.get("ignored_modes"),
                name="ignored_modes",
            ),
            required_positive_scope=str(
                payload.get("required_positive_scope", "")
            ),
            missing_label_policy=str(payload.get("missing_label_policy", "")),
            target_alignment=str(payload.get("target_alignment", "")),
            target_dtype=str(payload.get("target_dtype", "")),
        )
        supplied_hash = payload.get("contract_hash")
        if supplied_hash is not None and str(supplied_hash) != (
            contract.semantic_content_hash
        ):
            raise ArmModeContractError(
                "Arm-hold training contract hash does not match its contents."
            )
        return contract


DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1 = ArmHoldTrainingContract()


@dataclass(frozen=True)
class ResolvedArmHoldTrainingContract:
    """Result of binding optional dataset metadata to the training projection."""

    contract: ArmHoldTrainingContract | None
    source_kind: str
    compatibility_mode: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if self.contract is not None and not isinstance(
            self.contract,
            ArmHoldTrainingContract,
        ):
            raise TypeError("contract must be ArmHoldTrainingContract or None.")
        object.__setattr__(
            self,
            "source_kind",
            _nonempty(self.source_kind, name="source_kind"),
        )
        object.__setattr__(self, "compatibility_mode", bool(self.compatibility_mode))
        object.__setattr__(self, "reason", str(self.reason))


def arm_hold_training_projection_metadata(
    contract: ArmHoldTrainingContract = DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1,
) -> dict[str, Any]:
    """Return the additive ``label_specs`` training-projection declaration."""

    if not isinstance(contract, ArmHoldTrainingContract):
        raise TypeError("contract must be an ArmHoldTrainingContract.")
    return contract.to_dict(include_hash=True)


def _arm_mode_spec(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    label_specs = metadata.get("label_specs")
    if not isinstance(label_specs, Mapping):
        return None
    spec = label_specs.get(ARM_MODE_LABEL_SPEC_KEY)
    return spec if isinstance(spec, Mapping) else None


def resolve_arm_hold_training_contract(
    metadata: Mapping[str, Any],
    *,
    require_labels: bool = False,
) -> ResolvedArmHoldTrainingContract:
    """Resolve explicit new metadata or the frozen v2 compatibility declaration.

    Existing v2.3 datasets declare ``dexteleop_arm_mode_v2`` and its mode IDs
    but do not contain ``training_projections``.  They bind to the frozen v1
    projection in compatibility mode.  Datasets without arm-mode labels remain
    valid for ordinary action training and resolve to ``contract=None``.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    spec = _arm_mode_spec(metadata)
    if spec is None:
        if require_labels:
            raise ArmModeContractError(
                f"Dataset metadata is missing label_specs.{ARM_MODE_LABEL_SPEC_KEY}."
            )
        return ResolvedArmHoldTrainingContract(
            contract=None,
            source_kind="labels_missing",
            compatibility_mode=True,
            reason="dataset_has_no_arm_mode_label_spec",
        )
    if str(spec.get("schema", "")) != ARM_MODE_SCHEMA_V2:
        raise ArmModeContractError(
            "Arm-mode label spec schema is not dexteleop_arm_mode_v2."
        )
    sides = tuple(str(side) for side in spec.get("sides", ()))
    if sides != ARM_MODE_SIDE_ORDER_V1:
        raise ArmModeContractError(
            f"Arm-mode label sides must be {ARM_MODE_SIDE_ORDER_V1!r}."
        )
    raw_mode_ids = spec.get("mode_ids")
    if not isinstance(raw_mode_ids, Mapping):
        raise ArmModeContractError("Arm-mode label spec is missing mode_ids.")
    mode_ids = {str(label): int(mode_id) for label, mode_id in raw_mode_ids.items()}
    if mode_ids != ARM_MODE_IDS:
        raise ArmModeContractError(
            "Arm-mode label spec mode_ids disagree with dexteleop_arm_mode_v2."
        )
    projections = spec.get("training_projections")
    if isinstance(projections, Mapping):
        payload = projections.get(ARM_HOLD_TRAINING_PROJECTION_ID_V1)
        if payload is not None:
            if not isinstance(payload, Mapping):
                raise ArmModeContractError(
                    "arm_hold_intent_v1 training projection must be a mapping."
                )
            return ResolvedArmHoldTrainingContract(
                contract=ArmHoldTrainingContract.from_dict(payload),
                source_kind="explicit_training_projection",
                compatibility_mode=False,
            )
    return ResolvedArmHoldTrainingContract(
        contract=DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1,
        source_kind="implicit_v2_arm_mode_projection",
        compatibility_mode=True,
        reason="known_v2_label_spec_without_training_projection",
    )


__all__ = [
    "ARM_HOLD_MISSING_LABEL_MASK_INVALID_V1",
    "ARM_HOLD_POSITIVE_SCOPE_ARM_AND_END_EFFECTOR_V1",
    "ARM_HOLD_TARGET_ALIGNMENT_ACTION_INDICES_V1",
    "ARM_HOLD_TRAINING_CONTRACT_HASH_DOMAIN_V1",
    "ARM_HOLD_TRAINING_CONTRACT_SCHEMA_V1",
    "ARM_HOLD_TRAINING_PROJECTION_ID_V1",
    "ARM_MODE_IDS",
    "ARM_MODE_ID_TO_LABEL",
    "ARM_MODE_LABEL_SPEC_KEY",
    "ARM_MODE_SCHEMA",
    "ARM_MODE_SCHEMA_V2",
    "ARM_MODE_SIDE_ORDER_V1",
    "DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1",
    "ArmHoldTrainingContract",
    "ArmModeContractError",
    "ResolvedArmHoldTrainingContract",
    "arm_hold_training_projection_metadata",
    "resolve_arm_hold_training_contract",
]
