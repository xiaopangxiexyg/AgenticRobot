"""Lower-owned application policy port consumed by the UMI controller."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable


UmiOperatorEventTuple = tuple[str, str, float]


@runtime_checkable
class UmiOperatorPolicyPort(Protocol):
    """Station-owned operator policy required by the lower UMI runtime."""

    def active_controls(
        self,
        *,
        calibration_expired: bool,
        recording: bool,
        collection_blocked: bool,
        plus_save_available: bool,
        plus_save_in_flight: bool,
        foot_pad_online: bool,
        foot_pad_test_active: bool,
        dataset_reconfiguration_available: bool,
        episode_deletion_available: bool,
    ) -> list[str]: ...

    def merge_operator_events(
        self,
        web_events: Iterable[str],
        foot_pad_events: Iterable[str],
        *,
        timestamp: float,
    ) -> tuple[UmiOperatorEventTuple, ...]: ...

    def foot_pad_test_batch_active(
        self,
        events: Iterable[UmiOperatorEventTuple],
        *,
        test_already_active: bool,
    ) -> bool: ...


__all__ = ["UmiOperatorEventTuple", "UmiOperatorPolicyPort"]
