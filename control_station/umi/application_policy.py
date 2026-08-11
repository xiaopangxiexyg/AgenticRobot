"""Control Station implementation of the lower UMI operator-policy port."""

from __future__ import annotations

from collections.abc import Iterable

from .control_availability import UmiControlState, active_controls
from .input_mapping import (
    foot_pad_test_batch_active,
    merge_operator_events,
)


class StationUmiOperatorPolicy:
    """Project UMI runtime state and sampled inputs into operator policy."""

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
    ) -> list[str]:
        return active_controls(
            UmiControlState(
                calibration_expired=calibration_expired,
                recording=recording,
                collection_blocked=collection_blocked,
                plus_save_available=plus_save_available,
                plus_save_in_flight=plus_save_in_flight,
                foot_pad_online=foot_pad_online,
                foot_pad_test_active=foot_pad_test_active,
                dataset_reconfiguration_available=(
                    dataset_reconfiguration_available
                ),
                episode_deletion_available=episode_deletion_available,
            )
        )

    def merge_operator_events(
        self,
        web_events: Iterable[str],
        foot_pad_events: Iterable[str],
        *,
        timestamp: float,
    ) -> tuple[tuple[str, str, float], ...]:
        return tuple(
            (event.name, event.source, event.timestamp)
            for event in merge_operator_events(
                web_events,
                foot_pad_events,
                timestamp=timestamp,
            )
        )

    def foot_pad_test_batch_active(
        self,
        events: Iterable[tuple[str, str, float]],
        *,
        test_already_active: bool,
    ) -> bool:
        return foot_pad_test_batch_active(
            events,
            test_already_active=test_already_active,
        )


DEFAULT_STATION_UMI_OPERATOR_POLICY = StationUmiOperatorPolicy()


__all__ = [
    "DEFAULT_STATION_UMI_OPERATOR_POLICY",
    "StationUmiOperatorPolicy",
]
