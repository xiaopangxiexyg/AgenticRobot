"""Model-neutral history state driven only by formal arm-hold resolutions.

This module does not inspect future chunks, prepare targets, dispatch commands,
or claim hardware execution.  An embedding runtime may construct an update only
after the authoritative ``ArmHoldSemanticResolver`` state transition has been
committed.  That semantic commit is the compatibility seam shared by sync and
RTC even though their later emission/target paths differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dataset_contracts.arm_hold_history import (
    ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1,
    ArmHoldHistoryContext,
)
from dataset_contracts.arm_mode import ARM_MODE_SIDE_ORDER_V1
from dataset_contracts.canonical_encoding import canonical_content_hash

from .arm_hold import (
    ARM_HOLD_STATE_HOLD,
    ARM_HOLD_STATE_MOVING,
    ArmHoldResolutionReceipt,
)


ARM_HOLD_RESOLVED_HISTORY_BINDING_SCHEMA_V1 = (
    "dexteleop.arm_hold_resolved_history_binding.v1"
)
ARM_HOLD_RESOLVED_HISTORY_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.arm_hold_resolved_history_binding.semantic.v1"
)
ARM_HOLD_RESOLVED_HISTORY_UPDATE_SCHEMA_V1 = (
    "dexteleop.arm_hold_resolved_history_update.v1"
)
ARM_HOLD_RESOLVED_HISTORY_UPDATE_HASH_DOMAIN_V1 = (
    "dexteleop.arm_hold_resolved_history_update.semantic.v1"
)

ARM_HOLD_HISTORY_ADVANCED = "advanced"
ARM_HOLD_HISTORY_DUPLICATE = "duplicate"
ARM_HOLD_HISTORY_BINDING_MISMATCH = "binding_mismatch"
ARM_HOLD_HISTORY_STALE_SEQUENCE = "stale_sequence"
ARM_HOLD_HISTORY_SEQUENCE_CONFLICT = "sequence_conflict"
ARM_HOLD_HISTORY_SEQUENCE_GAP = "sequence_gap"


class ArmHoldResolvedHistoryError(ValueError):
    """Raised when a resolved-history identity or payload is malformed."""


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ArmHoldResolvedHistoryError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ArmHoldResolvedHistoryError(f"{name} must be an integer.")
    resolved = int(value)
    if resolved < 0:
        raise ArmHoldResolvedHistoryError(f"{name} must be nonnegative.")
    return resolved


@dataclass(frozen=True)
class ArmHoldResolvedHistoryBinding:
    """Identity of one resettable formal-resolution stream."""

    binding_id: str
    epoch: int
    contract_hash: str
    side_order: tuple[str, str] = ARM_MODE_SIDE_ORDER_V1
    schema: str = ARM_HOLD_RESOLVED_HISTORY_BINDING_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != ARM_HOLD_RESOLVED_HISTORY_BINDING_SCHEMA_V1:
            raise ArmHoldResolvedHistoryError(
                "Unknown resolved-history binding schema."
            )
        sides = tuple(str(side).strip().lower() for side in self.side_order)
        if sides != ARM_MODE_SIDE_ORDER_V1:
            raise ArmHoldResolvedHistoryError(
                f"side_order must be {ARM_MODE_SIDE_ORDER_V1!r}."
            )
        object.__setattr__(
            self, "binding_id", _nonempty(self.binding_id, name="binding_id")
        )
        object.__setattr__(
            self, "epoch", _nonnegative_integer(self.epoch, name="epoch")
        )
        object.__setattr__(
            self,
            "contract_hash",
            _nonempty(self.contract_hash, name="contract_hash"),
        )
        object.__setattr__(self, "side_order", sides)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding_id": self.binding_id,
            "epoch": self.epoch,
            "contract_hash": self.contract_hash,
            "side_order": list(self.side_order),
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=ARM_HOLD_RESOLVED_HISTORY_BINDING_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class ArmHoldResolvedHistoryUpdate:
    """One committed resolution with caller-owned stream ordering evidence."""

    binding_hash: str
    resolution_sequence: int
    resolution_id: str
    resolution: ArmHoldResolutionReceipt
    schema: str = ARM_HOLD_RESOLVED_HISTORY_UPDATE_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != ARM_HOLD_RESOLVED_HISTORY_UPDATE_SCHEMA_V1:
            raise ArmHoldResolvedHistoryError(
                "Unknown resolved-history update schema."
            )
        if not isinstance(self.resolution, ArmHoldResolutionReceipt):
            raise ArmHoldResolvedHistoryError(
                "resolution must be an ArmHoldResolutionReceipt."
            )
        sequence = _nonnegative_integer(
            self.resolution_sequence,
            name="resolution_sequence",
        )
        if sequence == 0:
            raise ArmHoldResolvedHistoryError(
                "resolution_sequence must be positive."
            )
        side_resolutions = self.resolution.side_resolutions
        sides = tuple(item.side for item in side_resolutions)
        if sides != ARM_MODE_SIDE_ORDER_V1:
            raise ArmHoldResolvedHistoryError(
                "Resolution must contain left/right sides in canonical order."
            )
        resolved_states = tuple(item.resolved_state for item in side_resolutions)
        if any(
            state not in {ARM_HOLD_STATE_MOVING, ARM_HOLD_STATE_HOLD}
            for state in resolved_states
        ):
            raise ArmHoldResolvedHistoryError(
                "Resolution contains an unknown resolved state."
            )
        combination = "".join(
            "1" if state == ARM_HOLD_STATE_HOLD else "0"
            for state in resolved_states
        )
        if combination != self.resolution.resolved_combination:
            raise ArmHoldResolvedHistoryError(
                "Resolution combination disagrees with its side states."
            )
        object.__setattr__(
            self,
            "binding_hash",
            _nonempty(self.binding_hash, name="binding_hash"),
        )
        object.__setattr__(self, "resolution_sequence", sequence)
        object.__setattr__(
            self,
            "resolution_id",
            _nonempty(self.resolution_id, name="resolution_id"),
        )

    @property
    def resolved_modes(self) -> tuple[int, int]:
        values = tuple(
            1 if item.resolved_state == ARM_HOLD_STATE_HOLD else 0
            for item in self.resolution.side_resolutions
        )
        return values[0], values[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding_hash": self.binding_hash,
            "resolution_sequence": self.resolution_sequence,
            "resolution_id": self.resolution_id,
            "resolution": self.resolution.to_dict(include_hash=True),
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=ARM_HOLD_RESOLVED_HISTORY_UPDATE_HASH_DOMAIN_V1,
        )


def build_arm_hold_resolved_history_update(
    *,
    binding: ArmHoldResolvedHistoryBinding,
    resolution_sequence: int,
    resolution_id: str,
    resolution: ArmHoldResolutionReceipt,
) -> ArmHoldResolvedHistoryUpdate:
    """Bind an authoritative resolver commit to its stream identity."""

    if not isinstance(binding, ArmHoldResolvedHistoryBinding):
        raise TypeError("binding must be an ArmHoldResolvedHistoryBinding.")
    if not isinstance(resolution, ArmHoldResolutionReceipt):
        raise TypeError("resolution must be an ArmHoldResolutionReceipt.")
    if resolution.contract_hash != binding.contract_hash:
        raise ArmHoldResolvedHistoryError(
            "Resolution contract hash does not match the active binding."
        )
    return ArmHoldResolvedHistoryUpdate(
        binding_hash=binding.semantic_content_hash,
        resolution_sequence=resolution_sequence,
        resolution_id=resolution_id,
        resolution=resolution,
    )


@dataclass(frozen=True)
class ArmHoldResolvedHistoryAdvanceReceipt:
    status: str
    accepted: bool
    mutated: bool
    resolution_sequence: int
    last_resolution_sequence: int | None


@dataclass(frozen=True)
class ArmHoldResolvedHistorySnapshot:
    binding: ArmHoldResolvedHistoryBinding
    context: ArmHoldHistoryContext
    last_resolution_sequence: int | None
    last_resolution_id: str | None
    last_update_hash: str | None


class ArmHoldResolvedHistoryState:
    """Bounded state for the next model request; owns no execution semantics."""

    def __init__(self) -> None:
        self._binding: ArmHoldResolvedHistoryBinding | None = None
        self._modes = (0, 0)
        self._dwell = (0, 0)
        self._last_resolution_sequence: int | None = None
        self._last_resolution_id: str | None = None
        self._last_update_hash: str | None = None

    def activate(
        self,
        binding: ArmHoldResolvedHistoryBinding,
    ) -> ArmHoldResolvedHistorySnapshot:
        if not isinstance(binding, ArmHoldResolvedHistoryBinding):
            raise TypeError("binding must be an ArmHoldResolvedHistoryBinding.")
        current = self._binding
        if current is not None:
            if binding.semantic_content_hash == current.semantic_content_hash:
                return self.snapshot()
            if binding.binding_id == current.binding_id:
                if binding.epoch < current.epoch:
                    raise ArmHoldResolvedHistoryError(
                        "Cannot rotate a binding to an older epoch."
                    )
                if binding.epoch == current.epoch:
                    raise ArmHoldResolvedHistoryError(
                        "One binding epoch cannot change its contract."
                    )
        self._binding = binding
        self._modes = (0, 0)
        self._dwell = (0, 0)
        self._last_resolution_sequence = None
        self._last_resolution_id = None
        self._last_update_hash = None
        return self.snapshot()

    def deactivate(self) -> None:
        self._binding = None
        self._modes = (0, 0)
        self._dwell = (0, 0)
        self._last_resolution_sequence = None
        self._last_resolution_id = None
        self._last_update_hash = None

    def snapshot(self) -> ArmHoldResolvedHistorySnapshot:
        binding = self._binding
        if binding is None:
            raise RuntimeError("Resolved arm-hold history has no active binding.")
        available = self._last_resolution_sequence is not None
        return ArmHoldResolvedHistorySnapshot(
            binding=binding,
            context=ArmHoldHistoryContext(
                previous_mode=np.asarray(self._modes, dtype=np.float32),
                available_mask=np.asarray([available, available], dtype=bool),
                dwell_steps=np.asarray(self._dwell, dtype=np.int64),
                provenance=ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1,
            ),
            last_resolution_sequence=self._last_resolution_sequence,
            last_resolution_id=self._last_resolution_id,
            last_update_hash=self._last_update_hash,
        )

    def advance(
        self,
        update: ArmHoldResolvedHistoryUpdate,
    ) -> ArmHoldResolvedHistoryAdvanceReceipt:
        if not isinstance(update, ArmHoldResolvedHistoryUpdate):
            raise TypeError("update must be an ArmHoldResolvedHistoryUpdate.")
        binding = self._binding
        if binding is None:
            raise RuntimeError("Resolved arm-hold history has no active binding.")
        last = self._last_resolution_sequence
        if update.binding_hash != binding.semantic_content_hash:
            return ArmHoldResolvedHistoryAdvanceReceipt(
                status=ARM_HOLD_HISTORY_BINDING_MISMATCH,
                accepted=False,
                mutated=False,
                resolution_sequence=update.resolution_sequence,
                last_resolution_sequence=last,
            )
        if update.resolution.contract_hash != binding.contract_hash:
            return ArmHoldResolvedHistoryAdvanceReceipt(
                status=ARM_HOLD_HISTORY_BINDING_MISMATCH,
                accepted=False,
                mutated=False,
                resolution_sequence=update.resolution_sequence,
                last_resolution_sequence=last,
            )
        update_hash = update.semantic_content_hash
        if last is not None and update.resolution_sequence == last:
            duplicate = (
                update.resolution_id == self._last_resolution_id
                and update_hash == self._last_update_hash
            )
            return ArmHoldResolvedHistoryAdvanceReceipt(
                status=(
                    ARM_HOLD_HISTORY_DUPLICATE
                    if duplicate
                    else ARM_HOLD_HISTORY_SEQUENCE_CONFLICT
                ),
                accepted=duplicate,
                mutated=False,
                resolution_sequence=update.resolution_sequence,
                last_resolution_sequence=last,
            )
        if last is not None and update.resolution_sequence < last:
            return ArmHoldResolvedHistoryAdvanceReceipt(
                status=ARM_HOLD_HISTORY_STALE_SEQUENCE,
                accepted=False,
                mutated=False,
                resolution_sequence=update.resolution_sequence,
                last_resolution_sequence=last,
            )
        if last is not None and update.resolution_sequence != last + 1:
            return ArmHoldResolvedHistoryAdvanceReceipt(
                status=ARM_HOLD_HISTORY_SEQUENCE_GAP,
                accepted=False,
                mutated=False,
                resolution_sequence=update.resolution_sequence,
                last_resolution_sequence=last,
            )

        modes = update.resolved_modes
        dwell_values = tuple(
            self._dwell[index] + 1
            if last is not None and modes[index] == self._modes[index]
            else 1
            for index in range(2)
        )
        self._modes = modes
        self._dwell = dwell_values[0], dwell_values[1]
        self._last_resolution_sequence = update.resolution_sequence
        self._last_resolution_id = update.resolution_id
        self._last_update_hash = update_hash
        return ArmHoldResolvedHistoryAdvanceReceipt(
            status=ARM_HOLD_HISTORY_ADVANCED,
            accepted=True,
            mutated=True,
            resolution_sequence=update.resolution_sequence,
            last_resolution_sequence=update.resolution_sequence,
        )


__all__ = [
    "ARM_HOLD_HISTORY_ADVANCED",
    "ARM_HOLD_HISTORY_BINDING_MISMATCH",
    "ARM_HOLD_HISTORY_DUPLICATE",
    "ARM_HOLD_HISTORY_SEQUENCE_CONFLICT",
    "ARM_HOLD_HISTORY_SEQUENCE_GAP",
    "ARM_HOLD_HISTORY_STALE_SEQUENCE",
    "ARM_HOLD_RESOLVED_HISTORY_BINDING_HASH_DOMAIN_V1",
    "ARM_HOLD_RESOLVED_HISTORY_BINDING_SCHEMA_V1",
    "ARM_HOLD_RESOLVED_HISTORY_UPDATE_HASH_DOMAIN_V1",
    "ARM_HOLD_RESOLVED_HISTORY_UPDATE_SCHEMA_V1",
    "ArmHoldResolvedHistoryAdvanceReceipt",
    "ArmHoldResolvedHistoryBinding",
    "ArmHoldResolvedHistoryError",
    "ArmHoldResolvedHistorySnapshot",
    "ArmHoldResolvedHistoryState",
    "ArmHoldResolvedHistoryUpdate",
    "build_arm_hold_resolved_history_update",
]
