"""Derived per-side hold-intent targets for model training.

This module reads the existing flat v2.3 arm-mode columns.  It never rewrites
source rows, guesses missing labels, or derives physical motion from pose
velocity.  The output target is aligned to the same source indices as the
continuous action target.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.arm_mode import (
    ARM_MODE_IDS,
    DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1,
    ArmHoldTrainingContract,
    ResolvedArmHoldTrainingContract,
    resolve_arm_hold_training_contract,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


ARM_HOLD_TRAINING_VIEW_SCHEMA_V1 = "dexteleop.arm_hold_training_view.v1"
ARM_HOLD_TRAINING_VIEW_HASH_DOMAIN_V1 = (
    "dexteleop.arm_hold_training_view.semantic.v1"
)


class ArmHoldProjectionError(ValueError):
    """Raised when present arm-mode data contradicts its declared contract."""


def _freeze_array(value: Any, *, dtype: Any, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ArmHoldProjectionError(
            f"Expected array shape [N, {', '.join(map(str, shape_tail))}], "
            f"got {array.shape!r}."
        )
    output = np.ascontiguousarray(array).copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class ArmHoldProjectionReceipt:
    contract_hash: str
    source_kind: str
    compatibility_mode: bool
    source_row_count: int
    projected_row_count: int
    missing_columns: tuple[str, ...]
    invalid_reason_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_hash", str(self.contract_hash))
        object.__setattr__(self, "source_kind", str(self.source_kind))
        object.__setattr__(self, "compatibility_mode", bool(self.compatibility_mode))
        object.__setattr__(self, "source_row_count", int(self.source_row_count))
        object.__setattr__(self, "projected_row_count", int(self.projected_row_count))
        object.__setattr__(
            self,
            "missing_columns",
            tuple(str(name) for name in self.missing_columns),
        )
        object.__setattr__(
            self,
            "invalid_reason_counts",
            MappingProxyType(
                {
                    str(reason): int(count)
                    for reason, count in self.invalid_reason_counts.items()
                    if int(count) > 0
                }
            ),
        )


@dataclass(frozen=True)
class ArmHoldTrainingView:
    """One immutable, episode-local hold target window."""

    source_indices: np.ndarray
    targets: np.ndarray
    valid_mask: np.ndarray
    loss_mask: np.ndarray
    transition_mask: np.ndarray
    source_mode_ids: np.ndarray
    receipt: ArmHoldProjectionReceipt
    schema: str = ARM_HOLD_TRAINING_VIEW_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != ARM_HOLD_TRAINING_VIEW_SCHEMA_V1:
            raise ArmHoldProjectionError(
                f"Unsupported arm-hold view schema {self.schema!r}."
            )
        indices = np.asarray(self.source_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0):
            raise ArmHoldProjectionError(
                "source_indices must be a non-negative one-dimensional array."
            )
        frozen_indices = np.ascontiguousarray(indices).copy()
        frozen_indices.setflags(write=False)
        targets = _freeze_array(
            self.targets,
            dtype=np.float32,
            shape_tail=(2,),
        )
        valid = _freeze_array(
            self.valid_mask,
            dtype=bool,
            shape_tail=(2,),
        )
        loss = _freeze_array(
            self.loss_mask,
            dtype=bool,
            shape_tail=(2,),
        )
        transitions = _freeze_array(
            self.transition_mask,
            dtype=bool,
            shape_tail=(2,),
        )
        mode_ids = _freeze_array(
            self.source_mode_ids,
            dtype=np.int64,
            shape_tail=(2,),
        )
        row_count = frozen_indices.shape[0]
        if any(
            array.shape[0] != row_count
            for array in (targets, valid, loss, transitions, mode_ids)
        ):
            raise ArmHoldProjectionError(
                "All arm-hold view arrays must have the same leading dimension."
            )
        if np.any(loss & ~valid):
            raise ArmHoldProjectionError("loss_mask cannot include invalid labels.")
        if np.any(transitions & ~valid):
            raise ArmHoldProjectionError(
                "transition_mask cannot include invalid labels."
            )
        if np.any((targets[valid] != 0.0) & (targets[valid] != 1.0)):
            raise ArmHoldProjectionError("Valid arm-hold targets must be binary.")
        object.__setattr__(self, "source_indices", frozen_indices)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "loss_mask", loss)
        object.__setattr__(self, "transition_mask", transitions)
        object.__setattr__(self, "source_mode_ids", mode_ids)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "contract_hash": self.receipt.contract_hash,
                "source_indices": self.source_indices,
                "targets": self.targets,
                "valid_mask": self.valid_mask,
                "loss_mask": self.loss_mask,
                "transition_mask": self.transition_mask,
                "source_mode_ids": self.source_mode_ids,
            },
            domain=ARM_HOLD_TRAINING_VIEW_HASH_DOMAIN_V1,
        )

    def to_training_dict(self) -> dict[str, np.ndarray]:
        return {
            "arm_hold_target": self.targets.copy(),
            "arm_hold_valid_mask": self.valid_mask.copy(),
            "arm_hold_loss_mask": self.loss_mask.copy(),
            "arm_hold_transition_mask": self.transition_mask.copy(),
        }


@dataclass(frozen=True)
class ArmHoldCoverageReport:
    total_rows: int
    valid_counts_by_side: Mapping[str, int]
    positive_counts_by_side: Mapping[str, int]
    negative_counts_by_side: Mapping[str, int]
    transition_counts_by_side: Mapping[str, int]
    pair_counts: Mapping[str, int]
    missing_pair_states: tuple[str, ...]
    perfectly_complementary: bool
    invalid_reason_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        for field_name in (
            "valid_counts_by_side",
            "positive_counts_by_side",
            "negative_counts_by_side",
            "transition_counts_by_side",
            "pair_counts",
            "invalid_reason_counts",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(
                    {
                        str(key): int(value)
                        for key, value in getattr(self, field_name).items()
                    }
                ),
            )
        object.__setattr__(self, "total_rows", int(self.total_rows))
        object.__setattr__(
            self,
            "missing_pair_states",
            tuple(str(value) for value in self.missing_pair_states),
        )
        object.__setattr__(
            self,
            "perfectly_complementary",
            bool(self.perfectly_complementary),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "valid_counts_by_side": dict(self.valid_counts_by_side),
            "positive_counts_by_side": dict(self.positive_counts_by_side),
            "negative_counts_by_side": dict(self.negative_counts_by_side),
            "transition_counts_by_side": dict(self.transition_counts_by_side),
            "pair_counts": dict(self.pair_counts),
            "missing_pair_states": list(self.missing_pair_states),
            "perfectly_complementary": self.perfectly_complementary,
            "invalid_reason_counts": dict(self.invalid_reason_counts),
        }


def _as_columnar(
    rows_or_columns: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
) -> dict[str, Sequence[Any]]:
    if isinstance(rows_or_columns, Mapping):
        return {str(key): value for key, value in rows_or_columns.items()}
    if isinstance(rows_or_columns, (str, bytes)) or not isinstance(
        rows_or_columns,
        Sequence,
    ):
        raise TypeError("rows_or_columns must be a column mapping or row sequence.")
    rows = tuple(rows_or_columns)
    names = {
        str(name)
        for row in rows
        if isinstance(row, Mapping)
        for name in row
    }
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("Every row must be a mapping.")
    return {
        name: [row.get(name) for row in rows]
        for name in names
    }


def _source_row_count(columns: Mapping[str, Sequence[Any]]) -> int:
    lengths = {
        len(value)
        for value in columns.values()
        if not isinstance(value, (str, bytes)) and hasattr(value, "__len__")
    }
    if not lengths:
        return 0
    if len(lengths) != 1:
        raise ArmHoldProjectionError(
            f"Arm-mode source columns have inconsistent lengths: {sorted(lengths)!r}."
        )
    return int(next(iter(lengths)))


def _selected_padding_mask(
    value: Any,
    *,
    source_indices: np.ndarray,
    source_row_count: int,
) -> np.ndarray:
    count = int(source_indices.size)
    if value is None:
        return np.zeros((count, 2), dtype=bool)
    array = np.asarray(value, dtype=bool)
    if array.ndim == 1:
        array = np.repeat(array[:, None], 2, axis=1)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ArmHoldProjectionError(
            "target_is_pad must have shape [N] or [N, 2]."
        )
    if array.shape[0] == source_row_count:
        return np.asarray(array[source_indices], dtype=bool)
    if array.shape[0] == count:
        return np.asarray(array, dtype=bool)
    raise ArmHoldProjectionError(
        "target_is_pad leading dimension must match the source or projected rows."
    )


def _resolved_contract(
    *,
    metadata: Mapping[str, Any] | None,
    contract: ArmHoldTrainingContract | None,
) -> ResolvedArmHoldTrainingContract:
    if contract is not None:
        if not isinstance(contract, ArmHoldTrainingContract):
            raise TypeError("contract must be ArmHoldTrainingContract or None.")
        return ResolvedArmHoldTrainingContract(
            contract=contract,
            source_kind="explicit_call_contract",
            compatibility_mode=False,
        )
    if metadata is None:
        return ResolvedArmHoldTrainingContract(
            contract=DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1,
            source_kind="default_known_v2_projection",
            compatibility_mode=True,
            reason="metadata_not_supplied",
        )
    return resolve_arm_hold_training_contract(metadata)


def project_arm_hold_training_view(
    rows_or_columns: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    contract: ArmHoldTrainingContract | None = None,
    source_indices: Sequence[int] | np.ndarray | None = None,
    target_is_pad: Any = None,
    strict: bool = False,
) -> ArmHoldTrainingView:
    """Project existing arm-mode columns to binary targets and explicit masks."""

    columns = _as_columnar(rows_or_columns)
    row_count = _source_row_count(columns)
    if source_indices is None:
        indices = np.arange(row_count, dtype=np.int64)
    else:
        indices = np.asarray(source_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0):
            raise ArmHoldProjectionError(
                "source_indices must be a non-negative one-dimensional sequence."
            )
        if row_count and np.any(indices >= row_count):
            raise ArmHoldProjectionError("source_indices exceed source row count.")
    resolved = _resolved_contract(metadata=metadata, contract=contract)
    projection_contract = resolved.contract
    count = int(indices.size)
    targets = np.zeros((count, 2), dtype=np.float32)
    valid = np.zeros((count, 2), dtype=bool)
    transitions = np.zeros((count, 2), dtype=bool)
    mode_ids = np.full((count, 2), -1, dtype=np.int64)
    invalid_reasons: Counter[str] = Counter()
    required_columns = (
        "teleop.arm_mode.schema",
        "teleop.arm_mode.available",
        *(
            name
            for side in ("left", "right")
            for name in (
                f"teleop.arm_mode.{side}.id",
                f"teleop.arm_mode.{side}.label",
                f"teleop.arm_mode.{side}.scope",
            )
        ),
    )
    missing_columns = tuple(name for name in required_columns if name not in columns)
    if projection_contract is None:
        invalid_reasons["label_spec_missing"] += count * 2
    elif row_count == 0 and count:
        invalid_reasons["source_rows_missing"] += count * 2
    else:
        continuity_column = columns.get("collection.continuity_segment_index")
        previous_valid = [False, False]
        previous_target = [0.0, 0.0]
        previous_continuity: Any = None
        previous_source_index: int | None = None
        structural_errors: list[str] = []
        for output_index, source_index_value in enumerate(indices):
            source_index = int(source_index_value)
            schema_value = (
                columns["teleop.arm_mode.schema"][source_index]
                if "teleop.arm_mode.schema" in columns
                else None
            )
            available_value = (
                columns["teleop.arm_mode.available"][source_index]
                if "teleop.arm_mode.available" in columns
                else None
            )
            continuity_value = (
                continuity_column[source_index]
                if continuity_column is not None
                else None
            )
            same_continuity = (
                output_index > 0
                and previous_source_index is not None
                and source_index == previous_source_index + 1
                and (
                    continuity_column is None
                    or continuity_value == previous_continuity
                )
            )
            for side_index, side in enumerate(projection_contract.side_order):
                reason: str | None = None
                id_name = f"teleop.arm_mode.{side}.id"
                label_name = f"teleop.arm_mode.{side}.label"
                scope_name = f"teleop.arm_mode.{side}.scope"
                raw_id = (
                    columns[id_name][source_index]
                    if id_name in columns
                    else None
                )
                raw_label = (
                    columns[label_name][source_index]
                    if label_name in columns
                    else None
                )
                raw_scope = (
                    columns[scope_name][source_index]
                    if scope_name in columns
                    else None
                )
                label = "" if raw_label is None else str(raw_label)
                try:
                    mode_id = int(raw_id)
                except (TypeError, ValueError):
                    mode_id = -1
                mode_ids[output_index, side_index] = mode_id
                if schema_value is None:
                    reason = "schema_missing"
                elif str(schema_value) != projection_contract.source_schema:
                    reason = "schema_mismatch"
                elif available_value not in (1, True):
                    reason = "label_unavailable"
                elif not label:
                    reason = "label_missing"
                elif label not in ARM_MODE_IDS:
                    reason = "unknown_label"
                elif mode_id != ARM_MODE_IDS[label]:
                    reason = "id_label_mismatch"
                elif label in projection_contract.positive_modes:
                    if str(raw_scope or "") != (
                        projection_contract.required_positive_scope
                    ):
                        reason = "positive_scope_mismatch"
                    else:
                        targets[output_index, side_index] = 1.0
                        valid[output_index, side_index] = True
                elif label in projection_contract.negative_modes:
                    targets[output_index, side_index] = 0.0
                    valid[output_index, side_index] = True
                elif label in projection_contract.ignored_modes:
                    reason = f"ignored_mode:{label}"
                else:
                    reason = "unmapped_label"
                if reason is not None:
                    invalid_reasons[reason] += 1
                    if reason in {
                        "schema_mismatch",
                        "unknown_label",
                        "id_label_mismatch",
                        "positive_scope_mismatch",
                    }:
                        structural_errors.append(
                            f"row {source_index} {side}: {reason}"
                        )
                if (
                    valid[output_index, side_index]
                    and previous_valid[side_index]
                    and same_continuity
                    and targets[output_index, side_index]
                    != previous_target[side_index]
                ):
                    transitions[output_index, side_index] = True
                previous_valid[side_index] = bool(
                    valid[output_index, side_index]
                )
                previous_target[side_index] = float(
                    targets[output_index, side_index]
                )
            previous_continuity = continuity_value
            previous_source_index = source_index
        if strict and structural_errors:
            preview = "; ".join(structural_errors[:5])
            suffix = (
                ""
                if len(structural_errors) <= 5
                else f"; ... {len(structural_errors) - 5} more"
            )
            raise ArmHoldProjectionError(preview + suffix)
    padding = _selected_padding_mask(
        target_is_pad,
        source_indices=indices,
        source_row_count=row_count,
    )
    loss = valid & ~padding
    transitions &= ~padding
    contract_hash = (
        ""
        if projection_contract is None
        else projection_contract.semantic_content_hash
    )
    receipt = ArmHoldProjectionReceipt(
        contract_hash=contract_hash,
        source_kind=resolved.source_kind,
        compatibility_mode=resolved.compatibility_mode,
        source_row_count=row_count,
        projected_row_count=count,
        missing_columns=missing_columns,
        invalid_reason_counts=invalid_reasons,
    )
    return ArmHoldTrainingView(
        source_indices=indices,
        targets=targets,
        valid_mask=valid,
        loss_mask=loss,
        transition_mask=transitions,
        source_mode_ids=mode_ids,
        receipt=receipt,
    )


def summarize_arm_hold_coverage(
    view: ArmHoldTrainingView,
) -> ArmHoldCoverageReport:
    """Summarize side balance, pair coverage, and transition coverage."""

    if not isinstance(view, ArmHoldTrainingView):
        raise TypeError("view must be an ArmHoldTrainingView.")
    sides = ("left", "right")
    valid_counts = {
        side: int(np.sum(view.valid_mask[:, index]))
        for index, side in enumerate(sides)
    }
    positive_counts = {
        side: int(
            np.sum(view.valid_mask[:, index] & (view.targets[:, index] == 1.0))
        )
        for index, side in enumerate(sides)
    }
    negative_counts = {
        side: int(
            np.sum(view.valid_mask[:, index] & (view.targets[:, index] == 0.0))
        )
        for index, side in enumerate(sides)
    }
    transition_counts = {
        side: int(np.sum(view.transition_mask[:, index]))
        for index, side in enumerate(sides)
    }
    pair_counter: Counter[str] = Counter()
    both_valid = np.all(view.valid_mask, axis=1)
    for target in view.targets[both_valid]:
        pair_counter[f"{int(target[0])}{int(target[1])}"] += 1
    pair_states = ("00", "01", "10", "11")
    missing = tuple(state for state in pair_states if pair_counter[state] == 0)
    pair_total = int(np.sum(both_valid))
    complementary = bool(
        pair_total > 0
        and pair_counter["01"] + pair_counter["10"] == pair_total
    )
    return ArmHoldCoverageReport(
        total_rows=int(view.targets.shape[0]),
        valid_counts_by_side=valid_counts,
        positive_counts_by_side=positive_counts,
        negative_counts_by_side=negative_counts,
        transition_counts_by_side=transition_counts,
        pair_counts={state: int(pair_counter[state]) for state in pair_states},
        missing_pair_states=missing,
        perfectly_complementary=complementary,
        invalid_reason_counts=view.receipt.invalid_reason_counts,
    )


__all__ = [
    "ARM_HOLD_TRAINING_VIEW_HASH_DOMAIN_V1",
    "ARM_HOLD_TRAINING_VIEW_SCHEMA_V1",
    "ArmHoldCoverageReport",
    "ArmHoldProjectionError",
    "ArmHoldProjectionReceipt",
    "ArmHoldTrainingView",
    "project_arm_hold_training_view",
    "summarize_arm_hold_coverage",
]
