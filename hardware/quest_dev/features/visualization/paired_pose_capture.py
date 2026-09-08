"""Lower-owned, passive capture of operator-declared static pose windows.

The reducer in this module consumes already-admitted Quest and robot pose
events.  It performs no I/O, starts no worker, advances no source, fits no
calibration, and installs no frame transform.  Its only output is replayable
candidate evidence plus the existing R0 paired-pose dataset projection.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_observation import (
    ExecutionFactReference,
    ExecutionMeasurementObservation,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.observation_cut import ObservationSampleRef
from robot_motion.runtime.pose_observation import (
    PoseCoordinateBinding,
    PoseMeasurementAcceptedEvent,
    PoseMeasurementInvalidationEvent,
    PoseMeasurementJournalCut,
    PoseMeasurementJournalEvent,
    PoseMeasurementPublication,
    pose_receiver_clock_descriptor_hash,
)

from ..controller_pose.protocol import (
    REFERENCE_SPACE_TYPES,
    ControllerGripPose,
    ControllerPoseSample,
    validate_column_major_rigid_matrix,
)
from ..controller_pose.runtime import (
    ControllerPoseAcceptedEvent,
    ControllerPoseActiveSessionCut,
    ControllerPoseInvalidationEvent,
    ControllerPoseObserverCloseCut,
)
from .paired_pose_calibration import (
    PairedPoseDataset,
    PairedPoseSample,
    PoseObservation,
    controller_wire_matrix_to_transform,
    parse_paired_pose_dataset,
)
from .transforms import pose_payload_to_matrix

CAPTURE_EVIDENCE_SCHEMA = "quest_dev.visualization.paired_pose_capture_evidence.v1"
CAPTURE_PROFILE_SCHEMA = "quest_dev.visualization.static_window_profile.v1"
CAPTURE_BINDING_SCHEMA = "quest_dev.visualization.paired_pose_capture_binding.v1"
STATIC_WINDOW_SCHEMA = "quest_dev.visualization.static_window_receipt.v1"
CAPTURE_FINALIZATION_SCHEMA = (
    "quest_dev.visualization.paired_pose_capture_finalization.v1"
)
CAPTURE_PROFILE_HASH_DOMAIN = "quest_dev.visualization.static_window_profile.v1"
CAPTURE_BINDING_HASH_DOMAIN = "quest_dev.visualization.paired_pose_capture_binding.v1"
QUEST_EVENT_HASH_DOMAIN = "quest_dev.visualization.quest_pose_event.v1"
STATIC_WINDOW_HASH_DOMAIN = "quest_dev.visualization.static_window_receipt.v1"
CAPTURE_FINALIZATION_HASH_DOMAIN = (
    "quest_dev.visualization.paired_pose_capture_finalization.v1"
)
QUEST_OBSERVER_CLOSE_CUT_HASH_DOMAIN = (
    "quest_dev.visualization.controller_pose_observer_close_cut.v1"
)
STATIONARY_EVIDENCE_HASH_DOMAIN = (
    "quest_dev.visualization.static_window_stationarity.v1"
)
CAPTURE_EVIDENCE_HASH_DOMAIN = "quest_dev.visualization.paired_pose_capture_evidence.v1"
SELECTION_METHOD = "minimum_receiver_monotonic_skew_then_robot_source_quest_sample_and_event_sequence.v1"


class CaptureTerminalError(RuntimeError):
    """The exact bound source can no longer produce trustworthy evidence."""


class StaticWindowRejectedError(ValueError):
    """One operator-declared window failed its explicit capture profile."""


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) > 240:
        raise ValueError(f"{name} must be at most 240 characters")
    return normalized


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    normalized = _non_negative_int(value, name=name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _finite_non_negative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _require_exact_keys(
    value: Mapping[Any, Any],
    required: set[str],
    *,
    name: str,
) -> None:
    missing = required.difference(value)
    unexpected = set(value).difference(required)
    if missing or unexpected:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(str(item) for item in unexpected)}"
        )


def _semantic_hash(payload: Mapping[str, Any], *, domain: str) -> str:
    return canonical_content_hash(payload, domain=domain)


def _clock_payload(clock: ClockDomainDescriptor) -> dict[str, Any]:
    if not isinstance(clock, ClockDomainDescriptor):
        raise TypeError("receiver_clock must be a ClockDomainDescriptor")
    if clock.monotonic is not True:
        raise ValueError("receiver_clock.monotonic must be exactly True")
    return {
        "clock_domain_id": clock.clock_domain_id,
        "runtime_session_id": clock.runtime_session_id,
        "boot_id": clock.boot_id,
        "reset_epoch": clock.reset_epoch,
        "kind": clock.kind,
        "resolution_sec": clock.resolution_sec,
        "monotonic": clock.monotonic,
        "mapped_clock_domain_id": clock.mapped_clock_domain_id,
        "offset_estimate_sec": clock.offset_estimate_sec,
        "offset_uncertainty_sec": clock.offset_uncertainty_sec,
        "mapping_sampled_from": clock.mapping_sampled_from,
        "mapping_sampled_until": clock.mapping_sampled_until,
    }


_CLOCK_KEYS = {
    "clock_domain_id",
    "runtime_session_id",
    "boot_id",
    "reset_epoch",
    "kind",
    "resolution_sec",
    "monotonic",
    "mapped_clock_domain_id",
    "offset_estimate_sec",
    "offset_uncertainty_sec",
    "mapping_sampled_from",
    "mapping_sampled_until",
}


def _parse_clock(value: object, *, name: str) -> ClockDomainDescriptor:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _require_exact_keys(value, _CLOCK_KEYS, name=name)
    return ClockDomainDescriptor(
        clock_domain_id=value["clock_domain_id"],
        runtime_session_id=value["runtime_session_id"],
        boot_id=value["boot_id"],
        reset_epoch=value["reset_epoch"],
        kind=value["kind"],
        resolution_sec=value["resolution_sec"],
        monotonic=value["monotonic"],
        mapped_clock_domain_id=value["mapped_clock_domain_id"],
        offset_estimate_sec=value["offset_estimate_sec"],
        offset_uncertainty_sec=value["offset_uncertainty_sec"],
        mapping_sampled_from=value["mapping_sampled_from"],
        mapping_sampled_until=value["mapping_sampled_until"],
    )


@dataclass(frozen=True)
class StaticWindowCaptureProfile:
    """All admission limits are explicit; there are no production defaults."""

    profile_id: str
    min_window_duration_ns: int
    max_window_duration_ns: int
    min_source_span_ns: int
    max_pair_skew_ns: int
    min_robot_samples: int
    max_robot_samples: int
    min_quest_samples: int
    max_quest_samples: int
    max_robot_translation_excursion_m: float
    max_robot_rotation_excursion_deg: float
    max_quest_translation_excursion_m: float
    max_quest_rotation_excursion_deg: float
    max_quest_reorder_events: int
    max_event_history: int
    max_completed_windows: int
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = {
            "profile_id": _required(self.profile_id, name="profile_id"),
            "min_window_duration_ns": _non_negative_int(
                self.min_window_duration_ns,
                name="min_window_duration_ns",
            ),
            "max_window_duration_ns": _positive_int(
                self.max_window_duration_ns,
                name="max_window_duration_ns",
            ),
            "min_source_span_ns": _non_negative_int(
                self.min_source_span_ns,
                name="min_source_span_ns",
            ),
            "max_pair_skew_ns": _non_negative_int(
                self.max_pair_skew_ns,
                name="max_pair_skew_ns",
            ),
            "min_robot_samples": _positive_int(
                self.min_robot_samples,
                name="min_robot_samples",
            ),
            "max_robot_samples": _positive_int(
                self.max_robot_samples,
                name="max_robot_samples",
            ),
            "min_quest_samples": _positive_int(
                self.min_quest_samples,
                name="min_quest_samples",
            ),
            "max_quest_samples": _positive_int(
                self.max_quest_samples,
                name="max_quest_samples",
            ),
            "max_robot_translation_excursion_m": _finite_non_negative(
                self.max_robot_translation_excursion_m,
                name="max_robot_translation_excursion_m",
            ),
            "max_robot_rotation_excursion_deg": _finite_non_negative(
                self.max_robot_rotation_excursion_deg,
                name="max_robot_rotation_excursion_deg",
            ),
            "max_quest_translation_excursion_m": _finite_non_negative(
                self.max_quest_translation_excursion_m,
                name="max_quest_translation_excursion_m",
            ),
            "max_quest_rotation_excursion_deg": _finite_non_negative(
                self.max_quest_rotation_excursion_deg,
                name="max_quest_rotation_excursion_deg",
            ),
            "max_quest_reorder_events": _positive_int(
                self.max_quest_reorder_events,
                name="max_quest_reorder_events",
            ),
            "max_event_history": _positive_int(
                self.max_event_history,
                name="max_event_history",
            ),
            "max_completed_windows": _positive_int(
                self.max_completed_windows,
                name="max_completed_windows",
            ),
        }
        if values["min_window_duration_ns"] > values["max_window_duration_ns"]:
            raise ValueError("window duration limits are reversed")
        if values["min_source_span_ns"] > values["max_window_duration_ns"]:
            raise ValueError("min_source_span_ns exceeds max_window_duration_ns")
        if values["min_robot_samples"] > values["max_robot_samples"]:
            raise ValueError("robot sample count limits are reversed")
        if values["min_quest_samples"] > values["max_quest_samples"]:
            raise ValueError("Quest sample count limits are reversed")
        if values["max_event_history"] < max(
            values["max_robot_samples"],
            values["max_quest_samples"],
            values["max_quest_reorder_events"],
        ):
            raise ValueError("max_event_history is too small for profile limits")
        for key, normalized in values.items():
            object.__setattr__(self, key, normalized)
        payload = self._semantic_payload()
        object.__setattr__(
            self,
            "semantic_content_hash",
            _semantic_hash(payload, domain=CAPTURE_PROFILE_HASH_DOMAIN),
        )

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": CAPTURE_PROFILE_SCHEMA,
            "profile_id": self.profile_id,
            "min_window_duration_ns": self.min_window_duration_ns,
            "max_window_duration_ns": self.max_window_duration_ns,
            "min_source_span_ns": self.min_source_span_ns,
            "max_pair_skew_ns": self.max_pair_skew_ns,
            "min_robot_samples": self.min_robot_samples,
            "max_robot_samples": self.max_robot_samples,
            "min_quest_samples": self.min_quest_samples,
            "max_quest_samples": self.max_quest_samples,
            "max_robot_translation_excursion_m": (
                self.max_robot_translation_excursion_m
            ),
            "max_robot_rotation_excursion_deg": (self.max_robot_rotation_excursion_deg),
            "max_quest_translation_excursion_m": (
                self.max_quest_translation_excursion_m
            ),
            "max_quest_rotation_excursion_deg": (self.max_quest_rotation_excursion_deg),
            "max_quest_reorder_events": self.max_quest_reorder_events,
            "max_event_history": self.max_event_history,
            "max_completed_windows": self.max_completed_windows,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._semantic_payload(),
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class QuestControllerCaptureBinding:
    server_instance_id: str
    subscription_token: int
    client_id: int
    client_instance_id: str
    xr_session_id: str
    reference_space_type: str
    reference_space_generation: int
    controller_source_session_id: str
    handedness: str
    target_ray_mode: str
    profiles: tuple[str, ...]
    reference_frame_id: str
    moving_frame_id: str
    xr_frame_clock_domain_id: str
    receipt_clock_domain_id: str
    next_event_sequence: int
    baseline_sample_sequence: int

    def __post_init__(self) -> None:
        for key in (
            "server_instance_id",
            "client_instance_id",
            "xr_session_id",
            "reference_space_type",
            "controller_source_session_id",
            "handedness",
            "target_ray_mode",
            "reference_frame_id",
            "moving_frame_id",
            "xr_frame_clock_domain_id",
            "receipt_clock_domain_id",
        ):
            object.__setattr__(self, key, _required(getattr(self, key), name=key))
        if isinstance(self.client_id, bool) or not isinstance(self.client_id, int):
            raise TypeError("client_id must be an integer")
        if self.client_id < 0:
            raise ValueError("client_id must be non-negative")
        generation = _non_negative_int(
            self.reference_space_generation,
            name="reference_space_generation",
        )
        next_sequence = _non_negative_int(
            self.next_event_sequence,
            name="next_event_sequence",
        )
        baseline = self.baseline_sample_sequence
        subscription_token = _positive_int(
            self.subscription_token,
            name="subscription_token",
        )
        if isinstance(baseline, bool) or not isinstance(baseline, int) or baseline < -1:
            raise ValueError("baseline_sample_sequence must be an integer >= -1")
        profiles = tuple(
            _required(value, name="profiles item") for value in self.profiles
        )
        if len(profiles) != len(set(profiles)):
            raise ValueError("profiles must be unique and preserve source order")
        if self.target_ray_mode != "tracked-pointer":
            raise ValueError(
                "trusted capture requires target_ray_mode='tracked-pointer'"
            )
        if self.handedness not in {"left", "right", "none"}:
            raise ValueError("handedness is unsupported")
        if self.reference_space_type not in REFERENCE_SPACE_TYPES:
            raise ValueError("reference_space_type is unsupported")
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "subscription_token", subscription_token)
        object.__setattr__(self, "reference_space_generation", generation)
        object.__setattr__(self, "next_event_sequence", next_sequence)
        object.__setattr__(self, "baseline_sample_sequence", baseline)

    @property
    def reference_epoch_id(self) -> str:
        return (
            f"{self.xr_session_id}:{self.reference_space_type}:"
            f"{self.reference_space_generation}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "server_instance_id": self.server_instance_id,
            "subscription_token": self.subscription_token,
            "client_id": self.client_id,
            "client_instance_id": self.client_instance_id,
            "xr_session_id": self.xr_session_id,
            "reference_space_type": self.reference_space_type,
            "reference_space_generation": self.reference_space_generation,
            "controller_source_session_id": self.controller_source_session_id,
            "handedness": self.handedness,
            "target_ray_mode": self.target_ray_mode,
            "profiles": list(self.profiles),
            "reference_frame_id": self.reference_frame_id,
            "moving_frame_id": self.moving_frame_id,
            "xr_frame_clock_domain_id": self.xr_frame_clock_domain_id,
            "receipt_clock_domain_id": self.receipt_clock_domain_id,
            "next_event_sequence": self.next_event_sequence,
            "baseline_sample_sequence": self.baseline_sample_sequence,
            "reference_epoch_id": self.reference_epoch_id,
        }


@dataclass(frozen=True)
class PairedPoseCaptureBinding:
    capture_id: str
    receiver_clock: ClockDomainDescriptor
    robot: PoseCoordinateBinding
    robot_next_event_sequence: int
    quest: QuestControllerCaptureBinding
    receiver_clock_descriptor_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        capture_id = _required(self.capture_id, name="capture_id")
        if not isinstance(self.robot, PoseCoordinateBinding):
            raise TypeError("robot must be a PoseCoordinateBinding")
        if not isinstance(self.quest, QuestControllerCaptureBinding):
            raise TypeError("quest must be a QuestControllerCaptureBinding")
        robot_next = _positive_int(
            self.robot_next_event_sequence,
            name="robot_next_event_sequence",
        )
        clock_payload = _clock_payload(self.receiver_clock)
        clock_hash = pose_receiver_clock_descriptor_hash(self.receiver_clock)
        if self.robot.receipt_clock_domain_id != self.receiver_clock.clock_domain_id:
            raise ValueError("robot receipt clock differs from receiver_clock")
        if self.quest.receipt_clock_domain_id != self.receiver_clock.clock_domain_id:
            raise ValueError("Quest receipt clock differs from receiver_clock")
        object.__setattr__(self, "capture_id", capture_id)
        object.__setattr__(self, "robot_next_event_sequence", robot_next)
        object.__setattr__(self, "receiver_clock_descriptor_hash", clock_hash)
        payload = {
            "schema": CAPTURE_BINDING_SCHEMA,
            "capture_id": capture_id,
            "receiver_clock": clock_payload,
            "receiver_clock_descriptor_hash": clock_hash,
            "robot": self.robot.as_dict(),
            "robot_next_event_sequence": robot_next,
            "quest": self.quest.as_dict(),
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            _semantic_hash(payload, domain=CAPTURE_BINDING_HASH_DOMAIN),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CAPTURE_BINDING_SCHEMA,
            "capture_id": self.capture_id,
            "receiver_clock": _clock_payload(self.receiver_clock),
            "receiver_clock_descriptor_hash": self.receiver_clock_descriptor_hash,
            "robot": self.robot.as_dict(),
            "robot_next_event_sequence": self.robot_next_event_sequence,
            "quest": self.quest.as_dict(),
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class _WindowBoundary:
    window_id: str
    start_received_monotonic_ns: int
    robot_start_next_event_sequence: int
    quest_start_next_event_sequence: int
    end_received_monotonic_ns: int | None = None
    robot_end_next_event_sequence: int | None = None
    quest_end_next_event_sequence: int | None = None


@dataclass(frozen=True)
class StaticWindowReceipt:
    window_id: str
    start_received_monotonic_ns: int
    end_received_monotonic_ns: int
    robot_start_next_event_sequence: int
    robot_end_next_event_sequence: int
    quest_start_next_event_sequence: int
    quest_end_next_event_sequence: int
    robot_events: tuple[PoseMeasurementJournalEvent, ...]
    quest_events: tuple[
        ControllerPoseAcceptedEvent | ControllerPoseInvalidationEvent, ...
    ]
    selected_robot_event_sequence: int
    selected_quest_event_sequence: int
    metrics: Mapping[str, int | float]
    stationary_evidence_id: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        window_id = _required(self.window_id, name="window_id")
        robot_events = tuple(self.robot_events)
        quest_events = tuple(self.quest_events)
        metrics = dict(self.metrics)
        payload = _receipt_payload(
            window_id=window_id,
            start_received_monotonic_ns=_non_negative_int(
                self.start_received_monotonic_ns,
                name="start_received_monotonic_ns",
            ),
            end_received_monotonic_ns=_non_negative_int(
                self.end_received_monotonic_ns,
                name="end_received_monotonic_ns",
            ),
            robot_start_next_event_sequence=_positive_int(
                self.robot_start_next_event_sequence,
                name="robot_start_next_event_sequence",
            ),
            robot_end_next_event_sequence=_positive_int(
                self.robot_end_next_event_sequence,
                name="robot_end_next_event_sequence",
            ),
            quest_start_next_event_sequence=_non_negative_int(
                self.quest_start_next_event_sequence,
                name="quest_start_next_event_sequence",
            ),
            quest_end_next_event_sequence=_non_negative_int(
                self.quest_end_next_event_sequence,
                name="quest_end_next_event_sequence",
            ),
            robot_events=robot_events,
            quest_events=quest_events,
            selected_robot_event_sequence=_positive_int(
                self.selected_robot_event_sequence,
                name="selected_robot_event_sequence",
            ),
            selected_quest_event_sequence=_non_negative_int(
                self.selected_quest_event_sequence,
                name="selected_quest_event_sequence",
            ),
            metrics=metrics,
            stationary_evidence_id=_required(
                self.stationary_evidence_id,
                name="stationary_evidence_id",
            ),
        )
        if _semantic_hash(payload, domain=STATIC_WINDOW_HASH_DOMAIN) != (
            self.semantic_content_hash
        ):
            raise ValueError("static-window receipt semantic hash mismatch")
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "robot_events", robot_events)
        object.__setattr__(self, "quest_events", quest_events)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STATIC_WINDOW_SCHEMA,
            "window_id": self.window_id,
            "boundary": {
                "start_received_monotonic_ns": self.start_received_monotonic_ns,
                "end_received_monotonic_ns": self.end_received_monotonic_ns,
                "robot_start_next_event_sequence": (
                    self.robot_start_next_event_sequence
                ),
                "robot_end_next_event_sequence": self.robot_end_next_event_sequence,
                "quest_start_next_event_sequence": (
                    self.quest_start_next_event_sequence
                ),
                "quest_end_next_event_sequence": self.quest_end_next_event_sequence,
            },
            "robot_events": [event.as_evidence_dict() for event in self.robot_events],
            "quest_events": [
                _quest_event_evidence(event) for event in self.quest_events
            ],
            "selection": {
                "method": SELECTION_METHOD,
                "robot_event_sequence": self.selected_robot_event_sequence,
                "quest_event_sequence": self.selected_quest_event_sequence,
            },
            "metrics": dict(self.metrics),
            "stationary_evidence_id": self.stationary_evidence_id,
            "semantic_content_hash": self.semantic_content_hash,
        }


def _robot_cut_payload(cut: PoseMeasurementJournalCut) -> dict[str, Any]:
    if not isinstance(cut, PoseMeasurementJournalCut):
        raise TypeError("robot_cut must be a PoseMeasurementJournalCut")
    return {
        "event_sequence": cut.event_sequence,
        "oldest_retained_event_sequence": cut.oldest_retained_event_sequence,
        "receiver_clock": {
            **_clock_payload(cut.receiver_clock),
            "semantic_content_hash": cut.receiver_clock_descriptor_hash,
        },
        "latest_by_stream": {
            stream_id: publication.as_evidence_dict()
            for stream_id, publication in cut.latest_by_stream.items()
        },
        "binding_hash_by_stream": dict(cut.binding_hash_by_stream),
        "invalidated_stream_ids": list(cut.invalidated_stream_ids),
        "expected_next_binding_hash_by_stream": dict(
            cut.expected_next_binding_hash_by_stream
        ),
        "receiver_clock_descriptor_hash": cut.receiver_clock_descriptor_hash,
        "semantic_content_hash": cut.semantic_content_hash,
    }


def _quest_active_session_payload(
    session: ControllerPoseActiveSessionCut,
) -> dict[str, Any]:
    if not isinstance(session, ControllerPoseActiveSessionCut):
        raise TypeError(
            "quest_cut.active_sessions must contain "
            "ControllerPoseActiveSessionCut values"
        )
    last_sequence = session.last_accepted_sample_sequence
    if last_sequence is not None:
        last_sequence = _non_negative_int(
            last_sequence,
            name="quest active session last_accepted_sample_sequence",
        )
    reference_space_type = _required(
        session.reference_space_type,
        name="quest active session reference_space_type",
    )
    if reference_space_type not in REFERENCE_SPACE_TYPES:
        raise ValueError("Quest active session reference_space_type is unsupported")
    return {
        "client_id": _non_negative_int(
            session.client_id,
            name="quest active session client_id",
        ),
        "client_instance_id": _required(
            session.client_instance_id,
            name="quest active session client_instance_id",
        ),
        "xr_session_id": _required(
            session.xr_session_id,
            name="quest active session xr_session_id",
        ),
        "reference_space_type": reference_space_type,
        "reference_space_generation": _non_negative_int(
            session.reference_space_generation,
            name="quest active session reference_space_generation",
        ),
        "last_accepted_sample_sequence": last_sequence,
    }


def _quest_close_cut_semantic_payload(
    cut: ControllerPoseObserverCloseCut,
) -> dict[str, Any]:
    if not isinstance(cut, ControllerPoseObserverCloseCut):
        raise TypeError("quest_cut must be a ControllerPoseObserverCloseCut")
    if not isinstance(cut.stopped, bool):
        raise TypeError("quest_cut.stopped must be boolean")
    receipt_clock = cut.receipt_clock_domain_id
    if receipt_clock is not None:
        receipt_clock = _required(
            receipt_clock,
            name="quest_cut.receipt_clock_domain_id",
        )
    sessions = tuple(
        _quest_active_session_payload(session) for session in cut.active_sessions
    )
    client_ids = tuple(session["client_id"] for session in sessions)
    if len(client_ids) != len(set(client_ids)):
        raise ValueError("quest_cut.active_sessions client IDs must be unique")
    if client_ids != tuple(sorted(client_ids)):
        raise ValueError("quest_cut.active_sessions must be ordered by client_id")
    return {
        "subscription_token": _positive_int(
            cut.subscription_token,
            name="quest_cut.subscription_token",
        ),
        "server_instance_id": _required(
            cut.server_instance_id,
            name="quest_cut.server_instance_id",
        ),
        "next_event_sequence": _non_negative_int(
            cut.next_event_sequence,
            name="quest_cut.next_event_sequence",
        ),
        "active_sessions": list(sessions),
        "receipt_clock_domain_id": receipt_clock,
        "stopped": cut.stopped,
    }


def _quest_close_cut_payload(cut: ControllerPoseObserverCloseCut) -> dict[str, Any]:
    payload = _quest_close_cut_semantic_payload(cut)
    return {
        **payload,
        "semantic_content_hash": _semantic_hash(
            payload,
            domain=QUEST_OBSERVER_CLOSE_CUT_HASH_DOMAIN,
        ),
    }


def _validate_finalization_against_binding(
    *,
    binding: PairedPoseCaptureBinding,
    robot_cut: PoseMeasurementJournalCut,
    quest_cut: ControllerPoseObserverCloseCut,
) -> None:
    _robot_cut_payload(robot_cut)
    _quest_close_cut_payload(quest_cut)
    robot_end = robot_cut.event_sequence + 1
    if robot_end < binding.robot_next_event_sequence:
        raise ValueError("robot final cut predates the capture binding")
    if robot_cut.receiver_clock != binding.receiver_clock:
        raise ValueError("robot final cut receiver clock descriptor drifted")
    stream_id = binding.robot.stream_id
    if (
        robot_cut.binding_hash_by_stream.get(stream_id)
        != binding.robot.semantic_content_hash
    ):
        raise ValueError("robot final cut binding hash drifted")
    if stream_id in robot_cut.invalidated_stream_ids:
        raise ValueError("robot final cut marks the selected stream invalid")
    if stream_id in robot_cut.expected_next_binding_hash_by_stream:
        raise ValueError("robot final cut has a pending selected-stream transition")
    latest_robot = robot_cut.latest_by_stream.get(stream_id)
    if latest_robot is None:
        raise ValueError("robot final cut lacks the selected stream's latest pose")
    if latest_robot.coordinate_binding != binding.robot:
        raise ValueError("robot final cut latest pose binding drifted")

    quest = binding.quest
    if quest_cut.subscription_token != quest.subscription_token:
        raise ValueError("Quest final cut subscription token drifted")
    if quest_cut.server_instance_id != quest.server_instance_id:
        raise ValueError("Quest final cut server instance drifted")
    if quest_cut.receipt_clock_domain_id != quest.receipt_clock_domain_id:
        raise ValueError("Quest final cut receipt clock domain drifted")
    if quest_cut.stopped:
        raise ValueError("Quest final cut is from a stopped runtime")
    if quest_cut.next_event_sequence < quest.next_event_sequence:
        raise ValueError("Quest final cut predates the capture binding")
    sessions = tuple(
        session
        for session in quest_cut.active_sessions
        if session.client_id == quest.client_id
    )
    if len(sessions) != 1:
        raise ValueError("Quest final cut must contain the bound active client once")
    session = sessions[0]
    if (
        session.client_instance_id != quest.client_instance_id
        or session.xr_session_id != quest.xr_session_id
        or session.reference_space_type != quest.reference_space_type
        or session.reference_space_generation != quest.reference_space_generation
    ):
        raise ValueError("Quest final cut session/reference identity drifted")
    last_sequence = session.last_accepted_sample_sequence
    if last_sequence is None or last_sequence < quest.baseline_sample_sequence:
        raise ValueError("Quest final cut lacks the bound session sample baseline")


@dataclass(frozen=True)
class PairedPoseCaptureFinalization:
    """Typed, pose-redacted close fence plus exact latest-pose evidence."""

    binding: PairedPoseCaptureBinding
    robot_cut: PoseMeasurementJournalCut
    quest_cut: ControllerPoseObserverCloseCut
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, PairedPoseCaptureBinding):
            raise TypeError("binding must be a PairedPoseCaptureBinding")
        _validate_finalization_against_binding(
            binding=self.binding,
            robot_cut=self.robot_cut,
            quest_cut=self.quest_cut,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            _semantic_hash(
                self._semantic_payload(),
                domain=CAPTURE_FINALIZATION_HASH_DOMAIN,
            ),
        )

    @property
    def robot_end_next_event_sequence(self) -> int:
        return self.robot_cut.event_sequence + 1

    @property
    def quest_end_next_event_sequence(self) -> int:
        return self.quest_cut.next_event_sequence

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": CAPTURE_FINALIZATION_SCHEMA,
            "binding_hash": self.binding.semantic_content_hash,
            "robot_end_next_event_sequence": (self.robot_end_next_event_sequence),
            "quest_end_next_event_sequence": (self.quest_end_next_event_sequence),
            "robot_cut": _robot_cut_payload(self.robot_cut),
            "quest_cut": _quest_close_cut_payload(self.quest_cut),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._semantic_payload(),
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class PairedPoseCaptureEvidence:
    profile: StaticWindowCaptureProfile
    binding: PairedPoseCaptureBinding
    windows: tuple[StaticWindowReceipt, ...]
    finalization: PairedPoseCaptureFinalization
    dataset: PairedPoseDataset
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile, StaticWindowCaptureProfile):
            raise TypeError("profile must be a StaticWindowCaptureProfile")
        if not isinstance(self.binding, PairedPoseCaptureBinding):
            raise TypeError("binding must be a PairedPoseCaptureBinding")
        if not isinstance(self.finalization, PairedPoseCaptureFinalization):
            raise TypeError("finalization must be a PairedPoseCaptureFinalization")
        if self.finalization.binding != self.binding:
            raise ValueError("finalization binding differs from capture binding")
        windows = tuple(self.windows)
        if (
            not windows
            or len(windows) > self.profile.max_completed_windows
            or any(not isinstance(value, StaticWindowReceipt) for value in windows)
        ):
            raise ValueError("windows must contain at least one StaticWindowReceipt")
        windows = tuple(
            _parse_window_receipt(
                window.as_dict(),
                profile=self.profile,
                binding=self.binding,
                name=f"windows[{index}]",
            )
            for index, window in enumerate(windows)
        )
        _validate_window_chain(
            windows=windows,
            binding=self.binding,
            finalization=self.finalization,
        )
        if not isinstance(self.dataset, PairedPoseDataset):
            raise TypeError("dataset must be a PairedPoseDataset")
        reparsed = parse_paired_pose_dataset(self.dataset.as_dict())
        if reparsed.semantic_sha256 != self.dataset.semantic_sha256:
            raise ValueError("dataset semantic hash is inconsistent")
        expected_dataset = ManualStaticPairedPoseCapture(
            profile=self.profile,
            binding=self.binding,
        )._build_dataset(windows)
        if expected_dataset.as_dict() != reparsed.as_dict():
            raise ValueError("dataset does not reproduce from captured windows")
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "dataset", reparsed)
        payload = self._semantic_payload()
        object.__setattr__(
            self,
            "semantic_content_hash",
            _semantic_hash(payload, domain=CAPTURE_EVIDENCE_HASH_DOMAIN),
        )

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": CAPTURE_EVIDENCE_SCHEMA,
            "profile": self.profile.as_dict(),
            "binding": self.binding.as_dict(),
            "windows": [window.as_dict() for window in self.windows],
            "finalization": self.finalization.as_dict(),
            "dataset": self.dataset.as_dict(),
            "dataset_semantic_sha256": self.dataset.semantic_sha256,
            "candidate_only": True,
            "calibration_accepted": None,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._semantic_payload(),
            "semantic_content_hash": self.semantic_content_hash,
        }


QuestPoseEvent = ControllerPoseAcceptedEvent | ControllerPoseInvalidationEvent


def _controller_payload(controller: ControllerGripPose) -> dict[str, Any]:
    if not isinstance(controller, ControllerGripPose):
        raise TypeError("Quest controllers must be ControllerGripPose values")
    matrix = validate_column_major_rigid_matrix(controller.matrix_reference_from_grip)
    if controller.handedness not in {"left", "right", "none"}:
        raise ValueError("controller.handedness is unsupported")
    if controller.target_ray_mode not in {
        "gaze",
        "tracked-pointer",
        "screen",
        "transient-pointer",
    }:
        raise ValueError("controller.target_ray_mode is unsupported")
    if not isinstance(controller.emulated_position, bool):
        raise TypeError("controller.emulated_position must be boolean")
    return {
        "source_session_id": _required(
            controller.source_session_id,
            name="controller.source_session_id",
        ),
        "handedness": _required(controller.handedness, name="controller.handedness"),
        "target_ray_mode": _required(
            controller.target_ray_mode,
            name="controller.target_ray_mode",
        ),
        "profiles": [
            _required(value, name="controller profile") for value in controller.profiles
        ],
        "emulated_position": controller.emulated_position,
        "matrix_reference_from_grip": list(matrix),
    }


def _sample_payload(sample: ControllerPoseSample) -> dict[str, Any]:
    if not isinstance(sample, ControllerPoseSample):
        raise TypeError("sample must be a ControllerPoseSample")
    received_monotonic_ns = _non_negative_int(
        sample.received_monotonic_ns,
        name="sample.received_monotonic_ns",
    )
    received_wall_time_s = _finite_non_negative(
        sample.received_wall_time_s,
        name="sample.received_wall_time_s",
    )
    xr_frame_time_ms = _finite_non_negative(
        sample.xr_frame_time_ms,
        name="sample.xr_frame_time_ms",
    )
    if sample.reference_space_type not in REFERENCE_SPACE_TYPES:
        raise ValueError("sample.reference_space_type is unsupported")
    controllers = [_controller_payload(value) for value in sample.controllers]
    source_ids = tuple(value["source_session_id"] for value in controllers)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("controller source IDs must be unique within a sample")
    return {
        "client_instance_id": _required(
            sample.client_instance_id,
            name="sample.client_instance_id",
        ),
        "xr_session_id": _required(sample.xr_session_id, name="sample.xr_session_id"),
        "reference_space_type": _required(
            sample.reference_space_type,
            name="sample.reference_space_type",
        ),
        "reference_space_generation": _non_negative_int(
            sample.reference_space_generation,
            name="sample.reference_space_generation",
        ),
        "sample_sequence": _non_negative_int(
            sample.sample_sequence,
            name="sample.sample_sequence",
        ),
        "xr_frame_time_ms": xr_frame_time_ms,
        "controllers": controllers,
        "received_wall_time_s": received_wall_time_s,
        "received_monotonic_ns": received_monotonic_ns,
    }


def _quest_event_semantic_payload(event: QuestPoseEvent) -> dict[str, Any]:
    if isinstance(event, ControllerPoseAcceptedEvent):
        return {
            "event_kind": "accepted",
            "event_sequence": _non_negative_int(
                event.event_sequence,
                name="event.event_sequence",
            ),
            "server_instance_id": _required(
                event.server_instance_id,
                name="event.server_instance_id",
            ),
            "client_id": _non_negative_int(event.client_id, name="event.client_id"),
            "receipt_clock_domain_id": _required(
                event.receipt_clock_domain_id,
                name="event.receipt_clock_domain_id",
            ),
            "sample": _sample_payload(event.sample),
        }
    if isinstance(event, ControllerPoseInvalidationEvent):
        if event.reference_space_type not in REFERENCE_SPACE_TYPES:
            raise ValueError("event.reference_space_type is unsupported")
        if event.reason not in {
            "reference_space_reset",
            "session_ended",
            "client_disconnected",
            "runtime_stopped",
        }:
            raise ValueError("event.reason is unsupported")
        return {
            "event_kind": "invalidation",
            "event_sequence": _non_negative_int(
                event.event_sequence,
                name="event.event_sequence",
            ),
            "server_instance_id": _required(
                event.server_instance_id,
                name="event.server_instance_id",
            ),
            "client_id": _non_negative_int(event.client_id, name="event.client_id"),
            "client_instance_id": _required(
                event.client_instance_id,
                name="event.client_instance_id",
            ),
            "xr_session_id": _required(
                event.xr_session_id,
                name="event.xr_session_id",
            ),
            "reference_space_type": _required(
                event.reference_space_type,
                name="event.reference_space_type",
            ),
            "previous_reference_space_generation": _non_negative_int(
                event.previous_reference_space_generation,
                name="event.previous_reference_space_generation",
            ),
            "next_reference_space_generation": (
                None
                if event.next_reference_space_generation is None
                else _non_negative_int(
                    event.next_reference_space_generation,
                    name="event.next_reference_space_generation",
                )
            ),
            "last_accepted_sample_sequence": (
                None
                if event.last_accepted_sample_sequence is None
                else _non_negative_int(
                    event.last_accepted_sample_sequence,
                    name="event.last_accepted_sample_sequence",
                )
            ),
            "reason": _required(event.reason, name="event.reason"),
            "received_monotonic_ns": _non_negative_int(
                event.received_monotonic_ns,
                name="event.received_monotonic_ns",
            ),
            "receipt_clock_domain_id": _required(
                event.receipt_clock_domain_id,
                name="event.receipt_clock_domain_id",
            ),
        }
    raise TypeError("event must be a typed Quest pose event")


def _quest_event_hash(event: QuestPoseEvent) -> str:
    return _semantic_hash(
        _quest_event_semantic_payload(event),
        domain=QUEST_EVENT_HASH_DOMAIN,
    )


def _quest_event_evidence(event: QuestPoseEvent) -> dict[str, Any]:
    payload = _quest_event_semantic_payload(event)
    return {
        **payload,
        "semantic_content_hash": _semantic_hash(
            payload,
            domain=QUEST_EVENT_HASH_DOMAIN,
        ),
    }


def _robot_event_hash(event: PoseMeasurementJournalEvent) -> str:
    if not isinstance(
        event,
        (PoseMeasurementAcceptedEvent, PoseMeasurementInvalidationEvent),
    ):
        raise TypeError("event must be a typed robot pose journal event")
    return event.semantic_content_hash


def _event_received_monotonic_ns(
    event: PoseMeasurementJournalEvent | QuestPoseEvent,
) -> int:
    if isinstance(event, PoseMeasurementAcceptedEvent):
        return event.publication.received_monotonic_ns
    if isinstance(event, PoseMeasurementInvalidationEvent):
        return event.received_monotonic_ns
    if isinstance(event, ControllerPoseAcceptedEvent):
        return event.sample.received_monotonic_ns
    if isinstance(event, ControllerPoseInvalidationEvent):
        return event.received_monotonic_ns
    raise TypeError("unsupported pose event")


def _robot_matrix(event: PoseMeasurementAcceptedEvent) -> np.ndarray:
    binding = event.publication.coordinate_binding
    values = tuple(
        event.publication.measurement.value_by_component[binding.component_id]
    )
    return pose_payload_to_matrix(
        {
            "position": values[:3],
            "orientation_quat_xyzw": values[3:],
        }
    )


def _selected_controller(
    event: ControllerPoseAcceptedEvent,
    binding: QuestControllerCaptureBinding,
) -> ControllerGripPose | None:
    matches = tuple(
        value
        for value in event.sample.controllers
        if value.source_session_id == binding.controller_source_session_id
    )
    if len(matches) > 1:
        raise ValueError("selected controller source appears more than once")
    return None if not matches else matches[0]


def _selected_valid_controller(
    event: ControllerPoseAcceptedEvent,
    binding: QuestControllerCaptureBinding,
) -> ControllerGripPose | None:
    """Return only a selected controller pose with real tracked position.

    WebXR can keep reporting a controller while its position is temporarily
    emulated.  That sample is useful as lifecycle evidence, but it must not
    enter the paired-pose dataset because its grip translation is not a
    measured controller position.
    """

    controller = _selected_controller(event, binding)
    if controller is None or controller.emulated_position:
        return None
    return controller


def _quest_matrix(
    event: ControllerPoseAcceptedEvent,
    binding: QuestControllerCaptureBinding,
) -> np.ndarray:
    controller = _selected_valid_controller(event, binding)
    if controller is None:
        raise ValueError("selected controller is absent or emulated")
    return controller_wire_matrix_to_transform(controller.matrix_reference_from_grip)


def _max_pairwise_excursion(
    matrices: Sequence[np.ndarray],
) -> tuple[float, float]:
    maximum_translation = 0.0
    maximum_rotation_deg = 0.0
    for first_index, first in enumerate(matrices):
        for second in matrices[first_index + 1 :]:
            maximum_translation = max(
                maximum_translation,
                float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
            )
            relative_rotation = first[:3, :3].T @ second[:3, :3]
            cosine = float(
                np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
            )
            maximum_rotation_deg = max(
                maximum_rotation_deg,
                math.degrees(math.acos(cosine)),
            )
    return maximum_translation, maximum_rotation_deg


def _receipt_payload(
    *,
    window_id: str,
    start_received_monotonic_ns: int,
    end_received_monotonic_ns: int,
    robot_start_next_event_sequence: int,
    robot_end_next_event_sequence: int,
    quest_start_next_event_sequence: int,
    quest_end_next_event_sequence: int,
    robot_events: Sequence[PoseMeasurementJournalEvent],
    quest_events: Sequence[QuestPoseEvent],
    selected_robot_event_sequence: int,
    selected_quest_event_sequence: int,
    metrics: Mapping[str, int | float],
    stationary_evidence_id: str,
) -> dict[str, Any]:
    return {
        "schema": STATIC_WINDOW_SCHEMA,
        "window_id": window_id,
        "boundary": {
            "start_received_monotonic_ns": start_received_monotonic_ns,
            "end_received_monotonic_ns": end_received_monotonic_ns,
            "robot_start_next_event_sequence": robot_start_next_event_sequence,
            "robot_end_next_event_sequence": robot_end_next_event_sequence,
            "quest_start_next_event_sequence": quest_start_next_event_sequence,
            "quest_end_next_event_sequence": quest_end_next_event_sequence,
        },
        "robot_events": [event.as_evidence_dict() for event in robot_events],
        "quest_events": [_quest_event_evidence(event) for event in quest_events],
        "selection": {
            "method": SELECTION_METHOD,
            "robot_event_sequence": selected_robot_event_sequence,
            "quest_event_sequence": selected_quest_event_sequence,
        },
        "metrics": dict(metrics),
        "stationary_evidence_id": stationary_evidence_id,
    }


class ManualStaticPairedPoseCapture:
    """Pure reducer for one explicitly bound manual calibration capture."""

    def __init__(
        self,
        *,
        profile: StaticWindowCaptureProfile,
        binding: PairedPoseCaptureBinding,
    ) -> None:
        if not isinstance(profile, StaticWindowCaptureProfile):
            raise TypeError("profile must be a StaticWindowCaptureProfile")
        if not isinstance(binding, PairedPoseCaptureBinding):
            raise TypeError("binding must be a PairedPoseCaptureBinding")
        self.profile = profile
        self.binding = binding
        self._lock = threading.RLock()
        self._robot_next_event_sequence = binding.robot_next_event_sequence
        self._quest_next_event_sequence = binding.quest.next_event_sequence
        self._robot_history: deque[PoseMeasurementJournalEvent] = deque()
        self._quest_history: deque[QuestPoseEvent] = deque()
        self._robot_hash_by_sequence: dict[int, str] = {}
        self._quest_hash_by_sequence: dict[int, str] = {}
        self._quest_pending: dict[int, QuestPoseEvent] = {}
        self._quest_pending_hash: dict[int, str] = {}
        self._last_robot_received_monotonic_ns: int | None = None
        self._last_selected_robot_source_sequence: int | None = None
        self._last_selected_quest_sample_sequence = (
            binding.quest.baseline_sample_sequence
        )
        self._last_selected_quest_xr_frame_time_ms: float | None = None
        self._last_selected_quest_received_monotonic_ns: int | None = None
        self._active: _WindowBoundary | None = None
        self._completed: list[StaticWindowReceipt] = []
        self._finalization: PairedPoseCaptureFinalization | None = None
        self._terminal_reason: str | None = None
        self._sealed = False

    def _assert_live(self) -> None:
        if self._terminal_reason is not None:
            raise CaptureTerminalError(self._terminal_reason)
        if self._sealed:
            raise RuntimeError("capture is already sealed")

    def _terminal(self, reason: str) -> None:
        self._terminal_reason = _required(reason, name="terminal reason")
        self._active = None
        raise CaptureTerminalError(self._terminal_reason)

    def _append_robot_history(self, event: PoseMeasurementJournalEvent) -> None:
        if len(self._robot_history) >= self.profile.max_event_history:
            evicted = self._robot_history[0]
            active = self._active
            if (
                active is not None
                and evicted.event_sequence >= active.robot_start_next_event_sequence
                and (
                    active.robot_end_next_event_sequence is None
                    or evicted.event_sequence < active.robot_end_next_event_sequence
                )
            ):
                self._terminal("robot event history overflowed an active window")
            evicted = self._robot_history.popleft()
            self._robot_hash_by_sequence.pop(evicted.event_sequence, None)
        self._robot_history.append(event)
        self._robot_hash_by_sequence[event.event_sequence] = _robot_event_hash(event)

    def _append_quest_history(self, event: QuestPoseEvent) -> None:
        if len(self._quest_history) >= self.profile.max_event_history:
            evicted = self._quest_history[0]
            active = self._active
            if (
                active is not None
                and evicted.event_sequence >= active.quest_start_next_event_sequence
                and (
                    active.quest_end_next_event_sequence is None
                    or evicted.event_sequence < active.quest_end_next_event_sequence
                )
            ):
                self._terminal("Quest event history overflowed an active window")
            evicted = self._quest_history.popleft()
            self._quest_hash_by_sequence.pop(evicted.event_sequence, None)
        self._quest_history.append(event)
        self._quest_hash_by_sequence[event.event_sequence] = _quest_event_hash(event)

    def ingest_robot_event(self, event: PoseMeasurementJournalEvent) -> None:
        with self._lock:
            self._ingest_robot_event_locked(event)

    def _ingest_robot_event_locked(
        self,
        event: PoseMeasurementJournalEvent,
    ) -> None:
        self._assert_live()
        if (
            self._finalization is not None
            and event.event_sequence >= self._finalization.robot_end_next_event_sequence
        ):
            return
        event_hash = _robot_event_hash(event)
        sequence = event.event_sequence
        if sequence < self._robot_next_event_sequence:
            previous = self._robot_hash_by_sequence.get(sequence)
            if previous == event_hash:
                return
            self._terminal("robot event sequence was reused or fell outside history")
        if sequence > self._robot_next_event_sequence:
            self._terminal("robot pose journal event gap")
        event_clock = (
            event.publication.receiver_clock
            if isinstance(event, PoseMeasurementAcceptedEvent)
            else event.receiver_clock
        )
        if event_clock != self.binding.receiver_clock:
            self._terminal("robot event receiver clock descriptor drifted")
        received_monotonic_ns = _event_received_monotonic_ns(event)
        if (
            self._last_robot_received_monotonic_ns is not None
            and received_monotonic_ns < self._last_robot_received_monotonic_ns
        ):
            self._terminal("robot receiver-monotonic time regressed")
        self._append_robot_history(event)
        self._robot_next_event_sequence = sequence + 1
        self._last_robot_received_monotonic_ns = received_monotonic_ns
        if isinstance(event, PoseMeasurementAcceptedEvent):
            publication = event.publication
            robot_binding = publication.coordinate_binding
            if robot_binding.stream_id != self.binding.robot.stream_id:
                return
            if robot_binding != self.binding.robot:
                self._terminal("robot coordinate binding drifted")
            source_sequence = publication.source_sequence
            previous_sequence = self._last_selected_robot_source_sequence
            if previous_sequence is not None and source_sequence <= previous_sequence:
                self._terminal("robot source sequence did not increase")
            self._last_selected_robot_source_sequence = source_sequence
            return
        if event.coordinate_binding.stream_id == self.binding.robot.stream_id:
            self._terminal("selected robot pose binding was invalidated")

    def ingest_quest_event(self, event: QuestPoseEvent) -> None:
        with self._lock:
            self._ingest_quest_event_locked(event)

    def _ingest_quest_event_locked(self, event: QuestPoseEvent) -> None:
        self._assert_live()
        if (
            self._finalization is not None
            and event.event_sequence >= self._finalization.quest_end_next_event_sequence
        ):
            return
        event_hash = _quest_event_hash(event)
        sequence = event.event_sequence
        if sequence < self.binding.quest.next_event_sequence:
            self._terminal("Quest event predates the atomic subscription cut")
        if sequence < self._quest_next_event_sequence:
            previous = self._quest_hash_by_sequence.get(sequence)
            if previous == event_hash:
                return
            self._terminal("Quest event sequence was reused or fell outside history")
        existing = self._quest_pending_hash.get(sequence)
        if existing is not None:
            if existing == event_hash:
                return
            self._terminal("Quest event sequence was reused with different content")
        if (
            len(self._quest_pending) >= self.profile.max_quest_reorder_events
            and sequence != self._quest_next_event_sequence
        ):
            self._terminal("Quest event reorder buffer overflow")
        self._quest_pending[sequence] = event
        self._quest_pending_hash[sequence] = event_hash
        while self._quest_next_event_sequence in self._quest_pending:
            next_sequence = self._quest_next_event_sequence
            ordered = self._quest_pending.pop(next_sequence)
            self._quest_pending_hash.pop(next_sequence)
            self._process_ordered_quest_event(ordered)
            self._quest_next_event_sequence = next_sequence + 1

    def _process_ordered_quest_event(self, event: QuestPoseEvent) -> None:
        quest = self.binding.quest
        if event.server_instance_id != quest.server_instance_id:
            self._terminal("Quest server instance drifted")
        if event.receipt_clock_domain_id != quest.receipt_clock_domain_id:
            self._terminal("Quest receipt clock domain drifted")
        self._append_quest_history(event)
        if isinstance(event, ControllerPoseInvalidationEvent):
            if (
                event.client_id == quest.client_id
                and event.client_instance_id == quest.client_instance_id
                and event.xr_session_id == quest.xr_session_id
                and event.reference_space_type == quest.reference_space_type
                and event.previous_reference_space_generation
                == quest.reference_space_generation
            ):
                self._terminal("selected Quest reference epoch was invalidated")
            return
        if event.client_id != quest.client_id:
            return
        sample = event.sample
        if (
            sample.client_instance_id != quest.client_instance_id
            or sample.xr_session_id != quest.xr_session_id
            or sample.reference_space_type != quest.reference_space_type
            or sample.reference_space_generation != quest.reference_space_generation
        ):
            self._terminal("selected Quest session/reference identity drifted")
        if sample.sample_sequence <= self._last_selected_quest_sample_sequence:
            self._terminal("Quest sample sequence did not increase")
        if (
            self._last_selected_quest_xr_frame_time_ms is not None
            and sample.xr_frame_time_ms < self._last_selected_quest_xr_frame_time_ms
        ):
            self._terminal("Quest XR frame time regressed")
        if (
            self._last_selected_quest_received_monotonic_ns is not None
            and sample.received_monotonic_ns
            < self._last_selected_quest_received_monotonic_ns
        ):
            self._terminal("Quest receiver-monotonic time regressed")
        self._last_selected_quest_sample_sequence = sample.sample_sequence
        self._last_selected_quest_xr_frame_time_ms = sample.xr_frame_time_ms
        self._last_selected_quest_received_monotonic_ns = sample.received_monotonic_ns
        controller = _selected_controller(event, quest)
        if controller is None:
            return
        if (
            controller.handedness != quest.handedness
            or controller.target_ray_mode != quest.target_ray_mode
        ):
            self._terminal("selected Quest controller identity metadata drifted")
        # WebXR profile strings describe current input capabilities and can
        # change after the same XRInputSource is first exposed. They remain in
        # event/evidence metadata, but are not source identity and therefore do
        # not terminate an otherwise consistent selected source.
        if len(controller.profiles) != len(set(controller.profiles)):
            self._terminal("selected Quest controller profiles are invalid")
        # A temporary WebXR tracking loss may expose an emulated controller
        # position.  Keep the ordered event for evidence and sequence fences,
        # but exclude its pose from the calibration dataset.  The window-level
        # sample-count/span checks below remain the fail-closed acceptance gate.
        if controller.emulated_position:
            return

    def begin_window(
        self,
        *,
        window_id: str,
        start_received_monotonic_ns: int,
        robot_start_next_event_sequence: int,
        quest_start_next_event_sequence: int,
    ) -> None:
        with self._lock:
            self._begin_window_locked(
                window_id=window_id,
                start_received_monotonic_ns=start_received_monotonic_ns,
                robot_start_next_event_sequence=robot_start_next_event_sequence,
                quest_start_next_event_sequence=quest_start_next_event_sequence,
            )

    def _begin_window_locked(
        self,
        *,
        window_id: str,
        start_received_monotonic_ns: int,
        robot_start_next_event_sequence: int,
        quest_start_next_event_sequence: int,
    ) -> None:
        self._assert_live()
        if self._finalization is not None:
            raise RuntimeError("cannot begin a window after finalization is pinned")
        if self._active is not None:
            raise RuntimeError("another static window is active")
        if len(self._completed) >= self.profile.max_completed_windows:
            raise RuntimeError("completed-window limit reached")
        start_ns = _non_negative_int(
            start_received_monotonic_ns,
            name="start_received_monotonic_ns",
        )
        robot_start = _positive_int(
            robot_start_next_event_sequence,
            name="robot_start_next_event_sequence",
        )
        quest_start = _non_negative_int(
            quest_start_next_event_sequence,
            name="quest_start_next_event_sequence",
        )
        if robot_start < self.binding.robot_next_event_sequence:
            raise ValueError("robot start cut predates the capture binding")
        if quest_start < self.binding.quest.next_event_sequence:
            raise ValueError("Quest start cut predates the capture binding")
        if self._completed:
            previous = self._completed[-1]
            if (
                start_ns < previous.end_received_monotonic_ns
                or robot_start < previous.robot_end_next_event_sequence
                or quest_start < previous.quest_end_next_event_sequence
            ):
                raise ValueError("static windows must not overlap or move backwards")
        self._assert_history_covers_start(
            robot_start=robot_start, quest_start=quest_start
        )
        self._active = _WindowBoundary(
            window_id=_required(window_id, name="window_id"),
            start_received_monotonic_ns=start_ns,
            robot_start_next_event_sequence=robot_start,
            quest_start_next_event_sequence=quest_start,
        )

    def _assert_history_covers_start(
        self, *, robot_start: int, quest_start: int
    ) -> None:
        if self._robot_next_event_sequence > robot_start and (
            not self._robot_history
            or self._robot_history[0].event_sequence > robot_start
        ):
            self._terminal("robot history no longer covers the requested start cut")
        if self._quest_next_event_sequence > quest_start and (
            not self._quest_history
            or self._quest_history[0].event_sequence > quest_start
        ):
            self._terminal("Quest history no longer covers the requested start cut")

    def complete_window(
        self,
        *,
        end_received_monotonic_ns: int,
        robot_end_next_event_sequence: int,
        quest_end_next_event_sequence: int,
    ) -> StaticWindowReceipt | None:
        with self._lock:
            return self._complete_window_locked(
                end_received_monotonic_ns=end_received_monotonic_ns,
                robot_end_next_event_sequence=robot_end_next_event_sequence,
                quest_end_next_event_sequence=quest_end_next_event_sequence,
            )

    def _complete_window_locked(
        self,
        *,
        end_received_monotonic_ns: int,
        robot_end_next_event_sequence: int,
        quest_end_next_event_sequence: int,
    ) -> StaticWindowReceipt | None:
        self._assert_live()
        active = self._active
        if active is None:
            raise RuntimeError("no static window is active")
        end_ns = _non_negative_int(
            end_received_monotonic_ns,
            name="end_received_monotonic_ns",
        )
        robot_end = _positive_int(
            robot_end_next_event_sequence,
            name="robot_end_next_event_sequence",
        )
        quest_end = _non_negative_int(
            quest_end_next_event_sequence,
            name="quest_end_next_event_sequence",
        )
        if active.end_received_monotonic_ns is None:
            if end_ns < active.start_received_monotonic_ns:
                raise ValueError("static window time boundary is reversed")
            if robot_end < active.robot_start_next_event_sequence:
                raise ValueError("robot event boundary is reversed")
            if quest_end < active.quest_start_next_event_sequence:
                raise ValueError("Quest event boundary is reversed")
            active = _WindowBoundary(
                window_id=active.window_id,
                start_received_monotonic_ns=active.start_received_monotonic_ns,
                robot_start_next_event_sequence=(
                    active.robot_start_next_event_sequence
                ),
                quest_start_next_event_sequence=(
                    active.quest_start_next_event_sequence
                ),
                end_received_monotonic_ns=end_ns,
                robot_end_next_event_sequence=robot_end,
                quest_end_next_event_sequence=quest_end,
            )
            self._active = active
        elif (
            end_ns != active.end_received_monotonic_ns
            or robot_end != active.robot_end_next_event_sequence
            or quest_end != active.quest_end_next_event_sequence
        ):
            raise ValueError("a draining window's end boundary is immutable")
        assert active.robot_end_next_event_sequence is not None
        assert active.quest_end_next_event_sequence is not None
        if (
            self._robot_next_event_sequence < active.robot_end_next_event_sequence
            or self._quest_next_event_sequence < active.quest_end_next_event_sequence
        ):
            return None
        try:
            receipt = self._build_window_receipt(active)
        except StaticWindowRejectedError:
            self._active = None
            raise
        self._completed.append(receipt)
        self._active = None
        return receipt

    def _build_window_receipt(self, active: _WindowBoundary) -> StaticWindowReceipt:
        assert active.end_received_monotonic_ns is not None
        assert active.robot_end_next_event_sequence is not None
        assert active.quest_end_next_event_sequence is not None
        duration_ns = (
            active.end_received_monotonic_ns - active.start_received_monotonic_ns
        )
        if not (
            self.profile.min_window_duration_ns
            <= duration_ns
            <= self.profile.max_window_duration_ns
        ):
            raise StaticWindowRejectedError("static window duration is outside profile")
        robot_events = tuple(
            event
            for event in self._robot_history
            if active.robot_start_next_event_sequence
            <= event.event_sequence
            < active.robot_end_next_event_sequence
        )
        quest_events = tuple(
            event
            for event in self._quest_history
            if active.quest_start_next_event_sequence
            <= event.event_sequence
            < active.quest_end_next_event_sequence
        )
        robot_samples = tuple(
            event
            for event in robot_events
            if isinstance(event, PoseMeasurementAcceptedEvent)
            and event.publication.coordinate_binding == self.binding.robot
            and active.start_received_monotonic_ns
            <= event.publication.received_monotonic_ns
            <= active.end_received_monotonic_ns
        )
        quest_samples = tuple(
            event
            for event in quest_events
            if isinstance(event, ControllerPoseAcceptedEvent)
            and event.client_id == self.binding.quest.client_id
            and _selected_valid_controller(event, self.binding.quest) is not None
            and active.start_received_monotonic_ns
            <= event.sample.received_monotonic_ns
            <= active.end_received_monotonic_ns
        )
        self._validate_sample_counts(robot_samples, quest_samples)
        robot_source_span_ns = (
            robot_samples[-1].publication.received_monotonic_ns
            - robot_samples[0].publication.received_monotonic_ns
        )
        quest_source_span_ns = (
            quest_samples[-1].sample.received_monotonic_ns
            - quest_samples[0].sample.received_monotonic_ns
        )
        if (
            robot_source_span_ns < self.profile.min_source_span_ns
            or quest_source_span_ns < self.profile.min_source_span_ns
        ):
            raise StaticWindowRejectedError("source time span is below profile")
        robot_matrices = tuple(_robot_matrix(event) for event in robot_samples)
        quest_matrices = tuple(
            _quest_matrix(event, self.binding.quest) for event in quest_samples
        )
        robot_translation, robot_rotation = _max_pairwise_excursion(robot_matrices)
        quest_translation, quest_rotation = _max_pairwise_excursion(quest_matrices)
        if (
            robot_translation > self.profile.max_robot_translation_excursion_m
            or robot_rotation > self.profile.max_robot_rotation_excursion_deg
            or quest_translation > self.profile.max_quest_translation_excursion_m
            or quest_rotation > self.profile.max_quest_rotation_excursion_deg
        ):
            raise StaticWindowRejectedError(
                "source motion exceeds static-window profile"
            )
        selected_robot, selected_quest = min(
            ((robot, quest) for robot in robot_samples for quest in quest_samples),
            key=lambda pair: (
                abs(
                    pair[0].publication.received_monotonic_ns
                    - pair[1].sample.received_monotonic_ns
                ),
                pair[0].publication.source_sequence,
                pair[1].sample.sample_sequence,
                pair[0].event_sequence,
                pair[1].event_sequence,
            ),
        )
        pair_skew_ns = abs(
            selected_robot.publication.received_monotonic_ns
            - selected_quest.sample.received_monotonic_ns
        )
        if pair_skew_ns > self.profile.max_pair_skew_ns:
            raise StaticWindowRejectedError("best pose pair exceeds receipt skew limit")
        metrics: dict[str, int | float] = {
            "duration_ns": duration_ns,
            "robot_sample_count": len(robot_samples),
            "quest_sample_count": len(quest_samples),
            "robot_source_span_ns": robot_source_span_ns,
            "quest_source_span_ns": quest_source_span_ns,
            "robot_translation_excursion_m": robot_translation,
            "robot_rotation_excursion_deg": robot_rotation,
            "quest_translation_excursion_m": quest_translation,
            "quest_rotation_excursion_deg": quest_rotation,
            "pair_skew_ns": pair_skew_ns,
        }
        stationarity_payload = {
            "window_id": active.window_id,
            "boundary": {
                "start_received_monotonic_ns": (active.start_received_monotonic_ns),
                "end_received_monotonic_ns": active.end_received_monotonic_ns,
                "robot_start_next_event_sequence": (
                    active.robot_start_next_event_sequence
                ),
                "robot_end_next_event_sequence": (active.robot_end_next_event_sequence),
                "quest_start_next_event_sequence": (
                    active.quest_start_next_event_sequence
                ),
                "quest_end_next_event_sequence": (active.quest_end_next_event_sequence),
            },
            "robot_event_hashes": [
                event.semantic_content_hash for event in robot_events
            ],
            "quest_event_hashes": [_quest_event_hash(event) for event in quest_events],
            "selection": {
                "method": SELECTION_METHOD,
                "robot_event_sequence": selected_robot.event_sequence,
                "quest_event_sequence": selected_quest.event_sequence,
            },
            "metrics": metrics,
            "profile_hash": self.profile.semantic_content_hash,
            "binding_hash": self.binding.semantic_content_hash,
        }
        stationary_evidence_id = _semantic_hash(
            stationarity_payload,
            domain=STATIONARY_EVIDENCE_HASH_DOMAIN,
        )
        payload = _receipt_payload(
            window_id=active.window_id,
            start_received_monotonic_ns=active.start_received_monotonic_ns,
            end_received_monotonic_ns=active.end_received_monotonic_ns,
            robot_start_next_event_sequence=(active.robot_start_next_event_sequence),
            robot_end_next_event_sequence=active.robot_end_next_event_sequence,
            quest_start_next_event_sequence=(active.quest_start_next_event_sequence),
            quest_end_next_event_sequence=active.quest_end_next_event_sequence,
            robot_events=robot_events,
            quest_events=quest_events,
            selected_robot_event_sequence=selected_robot.event_sequence,
            selected_quest_event_sequence=selected_quest.event_sequence,
            metrics=metrics,
            stationary_evidence_id=stationary_evidence_id,
        )
        return StaticWindowReceipt(
            window_id=active.window_id,
            start_received_monotonic_ns=active.start_received_monotonic_ns,
            end_received_monotonic_ns=active.end_received_monotonic_ns,
            robot_start_next_event_sequence=(active.robot_start_next_event_sequence),
            robot_end_next_event_sequence=active.robot_end_next_event_sequence,
            quest_start_next_event_sequence=(active.quest_start_next_event_sequence),
            quest_end_next_event_sequence=active.quest_end_next_event_sequence,
            robot_events=robot_events,
            quest_events=quest_events,
            selected_robot_event_sequence=selected_robot.event_sequence,
            selected_quest_event_sequence=selected_quest.event_sequence,
            metrics=metrics,
            stationary_evidence_id=stationary_evidence_id,
            semantic_content_hash=_semantic_hash(
                payload,
                domain=STATIC_WINDOW_HASH_DOMAIN,
            ),
        )

    def _validate_sample_counts(
        self,
        robot_samples: Sequence[PoseMeasurementAcceptedEvent],
        quest_samples: Sequence[ControllerPoseAcceptedEvent],
    ) -> None:
        if not (
            self.profile.min_robot_samples
            <= len(robot_samples)
            <= self.profile.max_robot_samples
        ):
            raise StaticWindowRejectedError("robot sample count is outside profile")
        if not (
            self.profile.min_quest_samples
            <= len(quest_samples)
            <= self.profile.max_quest_samples
        ):
            raise StaticWindowRejectedError("Quest sample count is outside profile")

    def seal_capture(
        self,
        *,
        robot_cut: PoseMeasurementJournalCut,
        quest_cut: ControllerPoseObserverCloseCut,
    ) -> PairedPoseCaptureEvidence | None:
        with self._lock:
            return self._seal_capture_locked(
                robot_cut=robot_cut,
                quest_cut=quest_cut,
            )

    def _seal_capture_locked(
        self,
        *,
        robot_cut: PoseMeasurementJournalCut,
        quest_cut: ControllerPoseObserverCloseCut,
    ) -> PairedPoseCaptureEvidence | None:
        self._assert_live()
        if self._active is not None:
            raise RuntimeError("cannot seal while a static window is active")
        if not self._completed:
            raise RuntimeError("at least one accepted static window is required")
        candidate = PairedPoseCaptureFinalization(
            binding=self.binding,
            robot_cut=robot_cut,
            quest_cut=quest_cut,
        )
        _validate_window_chain(
            windows=tuple(self._completed),
            binding=self.binding,
            finalization=candidate,
        )
        if self._finalization is None:
            if (
                self._robot_next_event_sequence
                > candidate.robot_end_next_event_sequence
                or self._quest_next_event_sequence
                > candidate.quest_end_next_event_sequence
            ):
                raise ValueError("final cut predates already-consumed capture events")
            if any(
                sequence >= candidate.quest_end_next_event_sequence
                for sequence in self._quest_pending
            ):
                raise ValueError("Quest pending events cross the proposed final cut")
            self._finalization = candidate
        elif candidate.as_dict() != self._finalization.as_dict():
            raise ValueError("a pinned capture finalization cut is immutable")
        finalization = self._finalization
        if (
            self._robot_next_event_sequence < finalization.robot_end_next_event_sequence
            or self._quest_next_event_sequence
            < finalization.quest_end_next_event_sequence
        ):
            return None
        dataset = self._build_dataset(tuple(self._completed))
        evidence = PairedPoseCaptureEvidence(
            profile=self.profile,
            binding=self.binding,
            windows=tuple(self._completed),
            finalization=finalization,
            dataset=dataset,
        )
        self._sealed = True
        return evidence

    def _build_dataset(
        self,
        windows: Sequence[StaticWindowReceipt],
    ) -> PairedPoseDataset:
        samples: list[PairedPoseSample] = []
        for window in windows:
            robot_event = next(
                event
                for event in window.robot_events
                if event.event_sequence == window.selected_robot_event_sequence
            )
            quest_event = next(
                event
                for event in window.quest_events
                if event.event_sequence == window.selected_quest_event_sequence
            )
            assert isinstance(robot_event, PoseMeasurementAcceptedEvent)
            assert isinstance(quest_event, ControllerPoseAcceptedEvent)
            robot_publication = robot_event.publication
            quest_hash = _quest_event_hash(quest_event)
            first = PoseObservation(
                observation_id=robot_publication.measurement.measurement_id,
                source_instance_id=self.binding.robot.source_instance_id,
                source_epoch_id=self.binding.robot.source_epoch_id,
                sequence=robot_publication.source_sequence,
                observed_at=robot_publication.measurement.measured_at,
                clock_domain_id=robot_publication.measurement.clock_domain_id,
                stationary_evidence_id=window.stationary_evidence_id,
                matrix_reference_from_moving=_robot_matrix(robot_event),
            )
            second = PoseObservation(
                observation_id=f"quest-{quest_hash[:32]}",
                source_instance_id=(self.binding.quest.controller_source_session_id),
                source_epoch_id=self.binding.quest.reference_epoch_id,
                sequence=quest_event.sample.sample_sequence,
                observed_at=quest_event.sample.xr_frame_time_ms / 1_000.0,
                clock_domain_id=self.binding.quest.xr_frame_clock_domain_id,
                stationary_evidence_id=window.stationary_evidence_id,
                matrix_reference_from_moving=_quest_matrix(
                    quest_event,
                    self.binding.quest,
                ),
            )
            samples.append(
                PairedPoseSample(
                    sample_id=f"pair-{window.stationary_evidence_id[:32]}",
                    first=first,
                    second=second,
                )
            )
        dataset = PairedPoseDataset(
            frames={
                "first_reference": self.binding.robot.reference_frame_id,
                "first_moving": self.binding.robot.moving_frame_id,
                "second_reference": self.binding.quest.reference_frame_id,
                "second_moving": self.binding.quest.moving_frame_id,
            },
            pairing_method="operator_static_window",
            quest_session_id=self.binding.quest.xr_session_id,
            reference_space_type=self.binding.quest.reference_space_type,
            reference_space_generation=(self.binding.quest.reference_space_generation),
            controller_source_session_id=(
                self.binding.quest.controller_source_session_id
            ),
            robot_source_instance_id=self.binding.robot.source_instance_id,
            quest_reference_epoch_id=self.binding.quest.reference_epoch_id,
            robot_source_epoch_id=self.binding.robot.source_epoch_id,
            tool_profile_hash=self.binding.robot.tool_profile_hash,
            samples=tuple(samples),
        )
        return parse_paired_pose_dataset(dataset.as_dict())

    def candidate_dataset(self) -> PairedPoseDataset:
        """Return an immutable fit candidate without pinning final source cuts."""

        with self._lock:
            self._assert_live()
            if self._active is not None:
                raise RuntimeError(
                    "cannot build a candidate dataset while a window is active"
                )
            if not self._completed:
                raise RuntimeError("at least one accepted static window is required")
            return self._build_dataset(tuple(self._completed))

    def snapshot_status(self) -> dict[str, Any]:
        """Return bounded, pose-redacted status only."""

        with self._lock:
            return self._snapshot_status_locked()

    def _snapshot_status_locked(self) -> dict[str, Any]:
        active = self._active
        finalization = self._finalization
        return {
            "mode": "passive_manual_static_window",
            "capture_id": self.binding.capture_id,
            "profile_id": self.profile.profile_id,
            "binding_hash": self.binding.semantic_content_hash,
            "receiver_clock_descriptor_hash": (
                self.binding.receiver_clock_descriptor_hash
            ),
            "robot_next_event_sequence": self._robot_next_event_sequence,
            "quest_next_event_sequence": self._quest_next_event_sequence,
            "quest_pending_event_count": len(self._quest_pending),
            "completed_window_count": len(self._completed),
            "active_window_id": None if active is None else active.window_id,
            "draining": bool(
                active is not None and active.end_received_monotonic_ns is not None
            ),
            "finalizing": finalization is not None and not self._sealed,
            "final_robot_end_next_event_sequence": (
                None
                if finalization is None
                else finalization.robot_end_next_event_sequence
            ),
            "final_quest_end_next_event_sequence": (
                None
                if finalization is None
                else finalization.quest_end_next_event_sequence
            ),
            "terminal_reason": self._terminal_reason,
            "sealed": self._sealed,
        }


_PROFILE_KEYS = {
    "schema",
    "profile_id",
    "min_window_duration_ns",
    "max_window_duration_ns",
    "min_source_span_ns",
    "max_pair_skew_ns",
    "min_robot_samples",
    "max_robot_samples",
    "min_quest_samples",
    "max_quest_samples",
    "max_robot_translation_excursion_m",
    "max_robot_rotation_excursion_deg",
    "max_quest_translation_excursion_m",
    "max_quest_rotation_excursion_deg",
    "max_quest_reorder_events",
    "max_event_history",
    "max_completed_windows",
    "semantic_content_hash",
}


def _parse_profile(value: object) -> StaticWindowCaptureProfile:
    if not isinstance(value, Mapping):
        raise TypeError("profile must be a mapping")
    _require_exact_keys(value, _PROFILE_KEYS, name="profile")
    if value["schema"] != CAPTURE_PROFILE_SCHEMA:
        raise ValueError("capture profile schema mismatch")
    profile = StaticWindowCaptureProfile(
        profile_id=value["profile_id"],
        min_window_duration_ns=value["min_window_duration_ns"],
        max_window_duration_ns=value["max_window_duration_ns"],
        min_source_span_ns=value["min_source_span_ns"],
        max_pair_skew_ns=value["max_pair_skew_ns"],
        min_robot_samples=value["min_robot_samples"],
        max_robot_samples=value["max_robot_samples"],
        min_quest_samples=value["min_quest_samples"],
        max_quest_samples=value["max_quest_samples"],
        max_robot_translation_excursion_m=(value["max_robot_translation_excursion_m"]),
        max_robot_rotation_excursion_deg=(value["max_robot_rotation_excursion_deg"]),
        max_quest_translation_excursion_m=(value["max_quest_translation_excursion_m"]),
        max_quest_rotation_excursion_deg=(value["max_quest_rotation_excursion_deg"]),
        max_quest_reorder_events=value["max_quest_reorder_events"],
        max_event_history=value["max_event_history"],
        max_completed_windows=value["max_completed_windows"],
    )
    if value["semantic_content_hash"] != profile.semantic_content_hash:
        raise ValueError("capture profile semantic hash mismatch")
    return profile


_QUEST_BINDING_KEYS = {
    "server_instance_id",
    "subscription_token",
    "client_id",
    "client_instance_id",
    "xr_session_id",
    "reference_space_type",
    "reference_space_generation",
    "controller_source_session_id",
    "handedness",
    "target_ray_mode",
    "profiles",
    "reference_frame_id",
    "moving_frame_id",
    "xr_frame_clock_domain_id",
    "receipt_clock_domain_id",
    "next_event_sequence",
    "baseline_sample_sequence",
    "reference_epoch_id",
}


def _parse_quest_binding(value: object) -> QuestControllerCaptureBinding:
    if not isinstance(value, Mapping):
        raise TypeError("binding.quest must be a mapping")
    _require_exact_keys(value, _QUEST_BINDING_KEYS, name="binding.quest")
    profiles = value["profiles"]
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
        raise TypeError("binding.quest.profiles must be a sequence")
    binding = QuestControllerCaptureBinding(
        server_instance_id=value["server_instance_id"],
        subscription_token=value["subscription_token"],
        client_id=value["client_id"],
        client_instance_id=value["client_instance_id"],
        xr_session_id=value["xr_session_id"],
        reference_space_type=value["reference_space_type"],
        reference_space_generation=value["reference_space_generation"],
        controller_source_session_id=value["controller_source_session_id"],
        handedness=value["handedness"],
        target_ray_mode=value["target_ray_mode"],
        profiles=tuple(profiles),
        reference_frame_id=value["reference_frame_id"],
        moving_frame_id=value["moving_frame_id"],
        xr_frame_clock_domain_id=value["xr_frame_clock_domain_id"],
        receipt_clock_domain_id=value["receipt_clock_domain_id"],
        next_event_sequence=value["next_event_sequence"],
        baseline_sample_sequence=value["baseline_sample_sequence"],
    )
    if value["reference_epoch_id"] != binding.reference_epoch_id:
        raise ValueError("Quest reference epoch ID mismatch")
    return binding


_POSE_BINDING_KEYS = {
    "binding_id",
    "source_instance_id",
    "source_epoch_id",
    "source_binding_hash",
    "control_domain_id",
    "component_id",
    "stream_id",
    "reference_frame_id",
    "reference_frame_spec_hash",
    "moving_frame_id",
    "tool_profile_hash",
    "pose_schema_hash",
    "fk_contract_id",
    "fk_contract_hash",
    "measurement_clock_domain_id",
    "receipt_clock_domain_id",
    "observation_binding_id",
    "observation_binding_hash",
    "observation_plan_hash",
    "source_kind",
    "semantic_content_hash",
}


def _parse_pose_binding(value: object, *, name: str) -> PoseCoordinateBinding:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _require_exact_keys(value, _POSE_BINDING_KEYS, name=name)
    binding = PoseCoordinateBinding(
        binding_id=value["binding_id"],
        source_instance_id=value["source_instance_id"],
        source_epoch_id=value["source_epoch_id"],
        source_binding_hash=value["source_binding_hash"],
        control_domain_id=value["control_domain_id"],
        component_id=value["component_id"],
        stream_id=value["stream_id"],
        reference_frame_id=value["reference_frame_id"],
        reference_frame_spec_hash=value["reference_frame_spec_hash"],
        moving_frame_id=value["moving_frame_id"],
        tool_profile_hash=value["tool_profile_hash"],
        pose_schema_hash=value["pose_schema_hash"],
        fk_contract_id=value["fk_contract_id"],
        fk_contract_hash=value["fk_contract_hash"],
        measurement_clock_domain_id=value["measurement_clock_domain_id"],
        receipt_clock_domain_id=value["receipt_clock_domain_id"],
        observation_binding_id=value["observation_binding_id"],
        observation_binding_hash=value["observation_binding_hash"],
        observation_plan_hash=value["observation_plan_hash"],
        source_kind=value["source_kind"],
    )
    if value["semantic_content_hash"] != binding.semantic_content_hash:
        raise ValueError(f"{name} semantic hash mismatch")
    return binding


_CAPTURE_BINDING_KEYS = {
    "schema",
    "capture_id",
    "receiver_clock",
    "receiver_clock_descriptor_hash",
    "robot",
    "robot_next_event_sequence",
    "quest",
    "semantic_content_hash",
}


def _parse_capture_binding(value: object) -> PairedPoseCaptureBinding:
    if not isinstance(value, Mapping):
        raise TypeError("binding must be a mapping")
    _require_exact_keys(value, _CAPTURE_BINDING_KEYS, name="binding")
    if value["schema"] != CAPTURE_BINDING_SCHEMA:
        raise ValueError("capture binding schema mismatch")
    binding = PairedPoseCaptureBinding(
        capture_id=value["capture_id"],
        receiver_clock=_parse_clock(value["receiver_clock"], name="receiver_clock"),
        robot=_parse_pose_binding(value["robot"], name="binding.robot"),
        robot_next_event_sequence=value["robot_next_event_sequence"],
        quest=_parse_quest_binding(value["quest"]),
    )
    if (
        value["receiver_clock_descriptor_hash"]
        != binding.receiver_clock_descriptor_hash
    ):
        raise ValueError("capture receiver clock descriptor hash mismatch")
    if value["semantic_content_hash"] != binding.semantic_content_hash:
        raise ValueError("capture binding semantic hash mismatch")
    return binding


_FACT_REFERENCE_KEYS = {
    "stage",
    "fact_id",
    "fact_hash",
    "observed_at",
    "clock_domain_id",
    "status",
    "semantic_content_hash",
}


def _parse_fact_reference(value: object, *, name: str) -> ExecutionFactReference:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _require_exact_keys(value, _FACT_REFERENCE_KEYS, name=name)
    fact = ExecutionFactReference(
        stage=value["stage"],
        fact_id=value["fact_id"],
        fact_hash=value["fact_hash"],
        observed_at=value["observed_at"],
        clock_domain_id=value["clock_domain_id"],
        status=value["status"],
    )
    if value["semantic_content_hash"] != fact.semantic_content_hash:
        raise ValueError(f"{name} semantic hash mismatch")
    return fact


_MEASUREMENT_KEYS = {
    "measurement_id",
    "source_measurement_hash",
    "control_domain_id",
    "measured_at",
    "available_at",
    "clock_domain_id",
    "value_by_component",
    "representation_by_component",
    "schema_hash_by_component",
    "observation_binding_id",
    "observation_plan_hash",
    "parent_facts",
    "semantic_content_hash",
}


def _parse_measurement(value: object) -> ExecutionMeasurementObservation:
    if not isinstance(value, Mapping):
        raise TypeError("publication.measurement must be a mapping")
    _require_exact_keys(value, _MEASUREMENT_KEYS, name="publication.measurement")
    parents = value["parent_facts"]
    if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes)):
        raise TypeError("publication.measurement.parent_facts must be a sequence")
    values = value["value_by_component"]
    representations = value["representation_by_component"]
    schemas = value["schema_hash_by_component"]
    if not all(
        isinstance(item, Mapping) for item in (values, representations, schemas)
    ):
        raise TypeError("measurement component maps must be mappings")
    measurement = ExecutionMeasurementObservation(
        measurement_id=value["measurement_id"],
        source_measurement_hash=value["source_measurement_hash"],
        control_domain_id=value["control_domain_id"],
        measured_at=value["measured_at"],
        available_at=value["available_at"],
        clock_domain_id=value["clock_domain_id"],
        value_by_component=values,
        representation_by_component=representations,
        schema_hash_by_component=schemas,
        observation_binding_id=value["observation_binding_id"],
        observation_plan_hash=value["observation_plan_hash"],
        parent_facts=tuple(
            _parse_fact_reference(item, name=f"parent_facts[{index}]")
            for index, item in enumerate(parents)
        ),
    )
    if value["semantic_content_hash"] != measurement.semantic_content_hash:
        raise ValueError("measurement semantic hash mismatch")
    return measurement


_SAMPLE_REF_KEYS = {
    "stream_id",
    "sample_id",
    "sequence",
    "measured_at",
    "measured_clock_domain_id",
    "available_at",
    "available_clock_domain_id",
    "payload_object_id",
    "valid",
    "stale",
    "control_admission_sequence",
    "content_metadata",
}


def _parse_sample_ref(value: object) -> ObservationSampleRef:
    if not isinstance(value, Mapping):
        raise TypeError("publication.sample_ref must be a mapping")
    _require_exact_keys(value, _SAMPLE_REF_KEYS, name="publication.sample_ref")
    metadata = value["content_metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("publication.sample_ref.content_metadata must be a mapping")
    return ObservationSampleRef(
        stream_id=value["stream_id"],
        sample_id=value["sample_id"],
        sequence=value["sequence"],
        measured_at=value["measured_at"],
        measured_clock_domain_id=value["measured_clock_domain_id"],
        available_at=value["available_at"],
        available_clock_domain_id=value["available_clock_domain_id"],
        payload_object_id=value["payload_object_id"],
        valid=value["valid"],
        stale=value["stale"],
        control_admission_sequence=value["control_admission_sequence"],
        content_metadata=metadata,
    )


_PUBLICATION_KEYS = {
    "coordinate_binding",
    "source_sequence",
    "received_monotonic_ns",
    "receiver_clock",
    "measurement",
    "sample_ref",
    "semantic_content_hash",
}


def _parse_publication(value: object) -> PoseMeasurementPublication:
    if not isinstance(value, Mapping):
        raise TypeError("publication must be a mapping")
    _require_exact_keys(value, _PUBLICATION_KEYS, name="publication")
    raw_clock = value["receiver_clock"]
    if not isinstance(raw_clock, Mapping):
        raise TypeError("publication.receiver_clock must be a mapping")
    _require_exact_keys(
        raw_clock,
        _CLOCK_KEYS | {"semantic_content_hash"},
        name="publication.receiver_clock",
    )
    clock = _parse_clock(
        {key: raw_clock[key] for key in _CLOCK_KEYS},
        name="publication.receiver_clock",
    )
    clock_hash = pose_receiver_clock_descriptor_hash(clock)
    if raw_clock["semantic_content_hash"] != clock_hash:
        raise ValueError("publication receiver clock hash mismatch")
    publication = PoseMeasurementPublication(
        coordinate_binding=_parse_pose_binding(
            value["coordinate_binding"],
            name="publication.coordinate_binding",
        ),
        source_sequence=value["source_sequence"],
        measurement=_parse_measurement(value["measurement"]),
        sample_ref=_parse_sample_ref(value["sample_ref"]),
        received_monotonic_ns=value["received_monotonic_ns"],
        receiver_clock=clock,
    )
    if value["semantic_content_hash"] != publication.semantic_content_hash:
        raise ValueError("pose publication semantic hash mismatch")
    return publication


def _parse_robot_event(value: object, *, name: str) -> PoseMeasurementJournalEvent:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    kind = value.get("event_kind")
    if kind == "accepted":
        keys = {
            "event_kind",
            "event_sequence",
            "publication",
            "receiver_clock_descriptor_hash",
            "semantic_content_hash",
        }
        _require_exact_keys(value, keys, name=name)
        event: PoseMeasurementJournalEvent = PoseMeasurementAcceptedEvent(
            event_sequence=value["event_sequence"],
            publication=_parse_publication(value["publication"]),
        )
    elif kind == "invalidation":
        keys = {
            "event_kind",
            "event_sequence",
            "coordinate_binding",
            "reason",
            "received_monotonic_ns",
            "receiver_clock",
            "invalidated_source_sequence",
            "next_coordinate_binding",
            "semantic_content_hash",
        }
        _require_exact_keys(value, keys, name=name)
        raw_clock = value["receiver_clock"]
        if not isinstance(raw_clock, Mapping):
            raise TypeError(f"{name}.receiver_clock must be a mapping")
        _require_exact_keys(
            raw_clock,
            _CLOCK_KEYS | {"semantic_content_hash"},
            name=f"{name}.receiver_clock",
        )
        clock = _parse_clock(
            {key: raw_clock[key] for key in _CLOCK_KEYS},
            name=f"{name}.receiver_clock",
        )
        if raw_clock["semantic_content_hash"] != pose_receiver_clock_descriptor_hash(
            clock
        ):
            raise ValueError(f"{name} receiver clock hash mismatch")
        next_binding = value["next_coordinate_binding"]
        event = PoseMeasurementInvalidationEvent(
            event_sequence=value["event_sequence"],
            coordinate_binding=_parse_pose_binding(
                value["coordinate_binding"],
                name=f"{name}.coordinate_binding",
            ),
            reason=value["reason"],
            received_monotonic_ns=value["received_monotonic_ns"],
            receiver_clock=clock,
            invalidated_source_sequence=value["invalidated_source_sequence"],
            next_coordinate_binding=(
                None
                if next_binding is None
                else _parse_pose_binding(
                    next_binding,
                    name=f"{name}.next_coordinate_binding",
                )
            ),
        )
    else:
        raise ValueError(f"{name}.event_kind is unsupported")
    if (
        value.get(
            "receiver_clock_descriptor_hash", event.receiver_clock_descriptor_hash
        )
        != event.receiver_clock_descriptor_hash
    ):
        raise ValueError(f"{name} receiver clock descriptor hash mismatch")
    if value["semantic_content_hash"] != event.semantic_content_hash:
        raise ValueError(f"{name} semantic hash mismatch")
    return event


_CONTROLLER_KEYS = {
    "source_session_id",
    "handedness",
    "target_ray_mode",
    "profiles",
    "emulated_position",
    "matrix_reference_from_grip",
}


def _parse_controller(value: object, *, name: str) -> ControllerGripPose:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _require_exact_keys(value, _CONTROLLER_KEYS, name=name)
    profiles = value["profiles"]
    matrix = value["matrix_reference_from_grip"]
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
        raise TypeError(f"{name}.profiles must be a sequence")
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
        raise TypeError(f"{name}.matrix_reference_from_grip must be a sequence")
    emulated = value["emulated_position"]
    if not isinstance(emulated, bool):
        raise TypeError(f"{name}.emulated_position must be boolean")
    controller = ControllerGripPose(
        source_session_id=_required(
            value["source_session_id"],
            name=f"{name}.source_session_id",
        ),
        handedness=_required(value["handedness"], name=f"{name}.handedness"),
        target_ray_mode=_required(
            value["target_ray_mode"],
            name=f"{name}.target_ray_mode",
        ),
        profiles=tuple(_required(item, name=f"{name}.profile") for item in profiles),
        emulated_position=emulated,
        matrix_reference_from_grip=tuple(matrix),
    )
    _controller_payload(controller)
    return controller


_QUEST_SAMPLE_KEYS = {
    "client_instance_id",
    "xr_session_id",
    "reference_space_type",
    "reference_space_generation",
    "sample_sequence",
    "xr_frame_time_ms",
    "controllers",
    "received_wall_time_s",
    "received_monotonic_ns",
}


def _parse_quest_sample(value: object, *, name: str) -> ControllerPoseSample:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _require_exact_keys(value, _QUEST_SAMPLE_KEYS, name=name)
    raw_controllers = value["controllers"]
    if not isinstance(raw_controllers, Sequence) or isinstance(
        raw_controllers,
        (str, bytes),
    ):
        raise TypeError(f"{name}.controllers must be a sequence")
    controllers = tuple(
        _parse_controller(item, name=f"{name}.controllers[{index}]")
        for index, item in enumerate(raw_controllers)
    )
    source_ids = tuple(value.source_session_id for value in controllers)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{name} controller source IDs must be unique")
    sample = ControllerPoseSample(
        client_instance_id=value["client_instance_id"],
        xr_session_id=value["xr_session_id"],
        reference_space_type=value["reference_space_type"],
        reference_space_generation=value["reference_space_generation"],
        sample_sequence=value["sample_sequence"],
        xr_frame_time_ms=value["xr_frame_time_ms"],
        controllers=controllers,
        received_wall_time_s=value["received_wall_time_s"],
        received_monotonic_ns=value["received_monotonic_ns"],
    )
    _sample_payload(sample)
    return sample


def _parse_quest_event(value: object, *, name: str) -> QuestPoseEvent:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    kind = value.get("event_kind")
    if kind == "accepted":
        keys = {
            "event_kind",
            "event_sequence",
            "server_instance_id",
            "client_id",
            "receipt_clock_domain_id",
            "sample",
            "semantic_content_hash",
        }
        _require_exact_keys(value, keys, name=name)
        event: QuestPoseEvent = ControllerPoseAcceptedEvent(
            event_sequence=value["event_sequence"],
            server_instance_id=value["server_instance_id"],
            client_id=value["client_id"],
            sample=_parse_quest_sample(value["sample"], name=f"{name}.sample"),
            receipt_clock_domain_id=value["receipt_clock_domain_id"],
        )
    elif kind == "invalidation":
        keys = {
            "event_kind",
            "event_sequence",
            "server_instance_id",
            "client_id",
            "client_instance_id",
            "xr_session_id",
            "reference_space_type",
            "previous_reference_space_generation",
            "next_reference_space_generation",
            "last_accepted_sample_sequence",
            "reason",
            "received_monotonic_ns",
            "receipt_clock_domain_id",
            "semantic_content_hash",
        }
        _require_exact_keys(value, keys, name=name)
        event = ControllerPoseInvalidationEvent(
            event_sequence=value["event_sequence"],
            server_instance_id=value["server_instance_id"],
            client_id=value["client_id"],
            client_instance_id=value["client_instance_id"],
            xr_session_id=value["xr_session_id"],
            reference_space_type=value["reference_space_type"],
            previous_reference_space_generation=(
                value["previous_reference_space_generation"]
            ),
            next_reference_space_generation=(value["next_reference_space_generation"]),
            last_accepted_sample_sequence=value["last_accepted_sample_sequence"],
            reason=value["reason"],
            received_monotonic_ns=value["received_monotonic_ns"],
            receipt_clock_domain_id=value["receipt_clock_domain_id"],
        )
    else:
        raise ValueError(f"{name}.event_kind is unsupported")
    expected_hash = _quest_event_hash(event)
    if value["semantic_content_hash"] != expected_hash:
        raise ValueError(f"{name} semantic hash mismatch")
    return event


_ROBOT_CUT_KEYS = {
    "event_sequence",
    "oldest_retained_event_sequence",
    "receiver_clock",
    "latest_by_stream",
    "binding_hash_by_stream",
    "invalidated_stream_ids",
    "expected_next_binding_hash_by_stream",
    "receiver_clock_descriptor_hash",
    "semantic_content_hash",
}


def _parse_robot_cut(value: object, *, name: str) -> PoseMeasurementJournalCut:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _require_exact_keys(value, _ROBOT_CUT_KEYS, name=name)
    raw_clock = value["receiver_clock"]
    if not isinstance(raw_clock, Mapping):
        raise TypeError(f"{name}.receiver_clock must be a mapping")
    _require_exact_keys(
        raw_clock,
        _CLOCK_KEYS | {"semantic_content_hash"},
        name=f"{name}.receiver_clock",
    )
    clock = _parse_clock(
        {key: raw_clock[key] for key in _CLOCK_KEYS},
        name=f"{name}.receiver_clock",
    )
    clock_hash = pose_receiver_clock_descriptor_hash(clock)
    if raw_clock["semantic_content_hash"] != clock_hash:
        raise ValueError(f"{name} receiver clock semantic hash mismatch")
    latest = value["latest_by_stream"]
    binding_hashes = value["binding_hash_by_stream"]
    invalidated = value["invalidated_stream_ids"]
    expected_next = value["expected_next_binding_hash_by_stream"]
    if not isinstance(latest, Mapping):
        raise TypeError(f"{name}.latest_by_stream must be a mapping")
    if not isinstance(binding_hashes, Mapping):
        raise TypeError(f"{name}.binding_hash_by_stream must be a mapping")
    if not isinstance(expected_next, Mapping):
        raise TypeError(
            f"{name}.expected_next_binding_hash_by_stream must be a mapping"
        )
    if not isinstance(invalidated, Sequence) or isinstance(
        invalidated,
        (str, bytes),
    ):
        raise TypeError(f"{name}.invalidated_stream_ids must be a sequence")
    cut = PoseMeasurementJournalCut(
        event_sequence=value["event_sequence"],
        oldest_retained_event_sequence=value["oldest_retained_event_sequence"],
        receiver_clock=clock,
        latest_by_stream={
            _required(stream_id, name=f"{name} latest stream_id"): (
                _parse_publication(publication)
            )
            for stream_id, publication in latest.items()
        },
        binding_hash_by_stream=binding_hashes,
        invalidated_stream_ids=tuple(invalidated),
        expected_next_binding_hash_by_stream=expected_next,
    )
    if value["receiver_clock_descriptor_hash"] != cut.receiver_clock_descriptor_hash:
        raise ValueError(f"{name} receiver clock descriptor hash mismatch")
    if value["semantic_content_hash"] != cut.semantic_content_hash:
        raise ValueError(f"{name} semantic hash mismatch")
    if _robot_cut_payload(cut) != dict(value):
        raise ValueError(f"{name} does not reproduce from typed cut evidence")
    return cut


_QUEST_ACTIVE_SESSION_KEYS = {
    "client_id",
    "client_instance_id",
    "xr_session_id",
    "reference_space_type",
    "reference_space_generation",
    "last_accepted_sample_sequence",
}
_QUEST_CLOSE_CUT_KEYS = {
    "subscription_token",
    "server_instance_id",
    "next_event_sequence",
    "active_sessions",
    "receipt_clock_domain_id",
    "stopped",
    "semantic_content_hash",
}


def _parse_quest_close_cut(
    value: object,
    *,
    name: str,
) -> ControllerPoseObserverCloseCut:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _require_exact_keys(value, _QUEST_CLOSE_CUT_KEYS, name=name)
    raw_sessions = value["active_sessions"]
    if not isinstance(raw_sessions, Sequence) or isinstance(
        raw_sessions,
        (str, bytes),
    ):
        raise TypeError(f"{name}.active_sessions must be a sequence")
    sessions: list[ControllerPoseActiveSessionCut] = []
    for index, raw_session in enumerate(raw_sessions):
        session_name = f"{name}.active_sessions[{index}]"
        if not isinstance(raw_session, Mapping):
            raise TypeError(f"{session_name} must be a mapping")
        _require_exact_keys(
            raw_session,
            _QUEST_ACTIVE_SESSION_KEYS,
            name=session_name,
        )
        sessions.append(
            ControllerPoseActiveSessionCut(
                client_id=raw_session["client_id"],
                client_instance_id=raw_session["client_instance_id"],
                xr_session_id=raw_session["xr_session_id"],
                reference_space_type=raw_session["reference_space_type"],
                reference_space_generation=(raw_session["reference_space_generation"]),
                last_accepted_sample_sequence=(
                    raw_session["last_accepted_sample_sequence"]
                ),
            )
        )
    cut = ControllerPoseObserverCloseCut(
        subscription_token=value["subscription_token"],
        server_instance_id=value["server_instance_id"],
        next_event_sequence=value["next_event_sequence"],
        active_sessions=tuple(sessions),
        receipt_clock_domain_id=value["receipt_clock_domain_id"],
        stopped=value["stopped"],
    )
    expected = _quest_close_cut_payload(cut)
    if expected != dict(value):
        raise ValueError(f"{name} does not reproduce from typed cut evidence")
    return cut


_FINALIZATION_KEYS = {
    "schema",
    "binding_hash",
    "robot_end_next_event_sequence",
    "quest_end_next_event_sequence",
    "robot_cut",
    "quest_cut",
    "semantic_content_hash",
}


def _parse_finalization(
    value: object,
    *,
    binding: PairedPoseCaptureBinding,
) -> PairedPoseCaptureFinalization:
    if not isinstance(value, Mapping):
        raise TypeError("capture finalization must be a mapping")
    _require_exact_keys(value, _FINALIZATION_KEYS, name="capture finalization")
    if value["schema"] != CAPTURE_FINALIZATION_SCHEMA:
        raise ValueError("capture finalization schema mismatch")
    if value["binding_hash"] != binding.semantic_content_hash:
        raise ValueError("capture finalization binding hash mismatch")
    finalization = PairedPoseCaptureFinalization(
        binding=binding,
        robot_cut=_parse_robot_cut(value["robot_cut"], name="finalization.robot_cut"),
        quest_cut=_parse_quest_close_cut(
            value["quest_cut"],
            name="finalization.quest_cut",
        ),
    )
    if finalization.as_dict() != dict(value):
        raise ValueError("capture finalization does not reproduce from typed cuts")
    return finalization


_WINDOW_KEYS = {
    "schema",
    "window_id",
    "boundary",
    "robot_events",
    "quest_events",
    "selection",
    "metrics",
    "stationary_evidence_id",
    "semantic_content_hash",
}
_BOUNDARY_KEYS = {
    "start_received_monotonic_ns",
    "end_received_monotonic_ns",
    "robot_start_next_event_sequence",
    "robot_end_next_event_sequence",
    "quest_start_next_event_sequence",
    "quest_end_next_event_sequence",
}
_SELECTION_KEYS = {"method", "robot_event_sequence", "quest_event_sequence"}
_METRIC_KEYS = {
    "duration_ns",
    "robot_sample_count",
    "quest_sample_count",
    "robot_source_span_ns",
    "quest_source_span_ns",
    "robot_translation_excursion_m",
    "robot_rotation_excursion_deg",
    "quest_translation_excursion_m",
    "quest_rotation_excursion_deg",
    "pair_skew_ns",
}


def _parse_window_receipt(
    value: object,
    *,
    profile: StaticWindowCaptureProfile,
    binding: PairedPoseCaptureBinding,
    name: str,
) -> StaticWindowReceipt:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _require_exact_keys(value, _WINDOW_KEYS, name=name)
    if value["schema"] != STATIC_WINDOW_SCHEMA:
        raise ValueError(f"{name} schema mismatch")
    boundary = value["boundary"]
    selection = value["selection"]
    metrics = value["metrics"]
    if not isinstance(boundary, Mapping):
        raise TypeError(f"{name}.boundary must be a mapping")
    if not isinstance(selection, Mapping):
        raise TypeError(f"{name}.selection must be a mapping")
    if not isinstance(metrics, Mapping):
        raise TypeError(f"{name}.metrics must be a mapping")
    _require_exact_keys(boundary, _BOUNDARY_KEYS, name=f"{name}.boundary")
    _require_exact_keys(selection, _SELECTION_KEYS, name=f"{name}.selection")
    _require_exact_keys(metrics, _METRIC_KEYS, name=f"{name}.metrics")
    if selection["method"] != SELECTION_METHOD:
        raise ValueError(f"{name} selection method mismatch")
    raw_robot_events = value["robot_events"]
    raw_quest_events = value["quest_events"]
    if not isinstance(raw_robot_events, Sequence) or isinstance(
        raw_robot_events,
        (str, bytes),
    ):
        raise TypeError(f"{name}.robot_events must be a sequence")
    if not isinstance(raw_quest_events, Sequence) or isinstance(
        raw_quest_events,
        (str, bytes),
    ):
        raise TypeError(f"{name}.quest_events must be a sequence")
    if (
        len(raw_robot_events) > profile.max_event_history
        or len(raw_quest_events) > profile.max_event_history
    ):
        raise ValueError(f"{name} exceeds bounded evidence history")
    robot_events = tuple(
        _parse_robot_event(item, name=f"{name}.robot_events[{index}]")
        for index, item in enumerate(raw_robot_events)
    )
    quest_events = tuple(
        _parse_quest_event(item, name=f"{name}.quest_events[{index}]")
        for index, item in enumerate(raw_quest_events)
    )
    robot_start = _positive_int(
        boundary["robot_start_next_event_sequence"],
        name=f"{name}.robot_start_next_event_sequence",
    )
    robot_end = _positive_int(
        boundary["robot_end_next_event_sequence"],
        name=f"{name}.robot_end_next_event_sequence",
    )
    quest_start = _non_negative_int(
        boundary["quest_start_next_event_sequence"],
        name=f"{name}.quest_start_next_event_sequence",
    )
    quest_end = _non_negative_int(
        boundary["quest_end_next_event_sequence"],
        name=f"{name}.quest_end_next_event_sequence",
    )
    start_ns = _non_negative_int(
        boundary["start_received_monotonic_ns"],
        name=f"{name}.start_received_monotonic_ns",
    )
    end_ns = _non_negative_int(
        boundary["end_received_monotonic_ns"],
        name=f"{name}.end_received_monotonic_ns",
    )
    window_id = _required(value["window_id"], name=f"{name}.window_id")
    if end_ns < start_ns or robot_end < robot_start or quest_end < quest_start:
        raise ValueError(f"{name} boundary is reversed")
    if (
        robot_start < binding.robot_next_event_sequence
        or quest_start < binding.quest.next_event_sequence
    ):
        raise ValueError(f"{name} starts before the capture binding")
    if tuple(event.event_sequence for event in robot_events) != tuple(
        range(robot_start, robot_end)
    ):
        raise ValueError(f"{name} robot evidence does not exactly cover its cuts")
    if tuple(event.event_sequence for event in quest_events) != tuple(
        range(quest_start, quest_end)
    ):
        raise ValueError(f"{name} Quest evidence does not exactly cover its cuts")
    validator = ManualStaticPairedPoseCapture(profile=profile, binding=binding)
    validator._robot_next_event_sequence = robot_start
    validator._quest_next_event_sequence = quest_start
    for event in robot_events:
        validator.ingest_robot_event(event)
    for event in quest_events:
        validator.ingest_quest_event(event)
    active = _WindowBoundary(
        window_id=window_id,
        start_received_monotonic_ns=start_ns,
        robot_start_next_event_sequence=robot_start,
        quest_start_next_event_sequence=quest_start,
        end_received_monotonic_ns=end_ns,
        robot_end_next_event_sequence=robot_end,
        quest_end_next_event_sequence=quest_end,
    )
    expected = validator._build_window_receipt(active)
    if expected.as_dict() != dict(value):
        raise ValueError(f"{name} does not reproduce from its typed event evidence")
    return expected


def _validate_window_chain(
    *,
    windows: Sequence[StaticWindowReceipt],
    binding: PairedPoseCaptureBinding,
    finalization: PairedPoseCaptureFinalization,
) -> None:
    for previous, current in pairwise(windows):
        if (
            current.start_received_monotonic_ns < previous.end_received_monotonic_ns
            or current.robot_start_next_event_sequence
            < previous.robot_end_next_event_sequence
            or current.quest_start_next_event_sequence
            < previous.quest_end_next_event_sequence
        ):
            raise ValueError("capture evidence windows overlap or move backwards")
    last_robot_receipt_ns: int | None = None
    last_robot_source_sequence: int | None = None
    last_robot_publication: PoseMeasurementPublication | None = None
    last_quest_sample_sequence = binding.quest.baseline_sample_sequence
    last_quest_xr_frame_time_ms: float | None = None
    last_quest_receipt_ns: int | None = None
    for window in windows:
        for event in window.robot_events:
            received_ns = _event_received_monotonic_ns(event)
            if (
                last_robot_receipt_ns is not None
                and received_ns < last_robot_receipt_ns
            ):
                raise ValueError("robot receipt time regressed across windows")
            last_robot_receipt_ns = received_ns
            if (
                isinstance(event, PoseMeasurementAcceptedEvent)
                and event.publication.coordinate_binding == binding.robot
            ):
                source_sequence = event.publication.source_sequence
                if (
                    last_robot_source_sequence is not None
                    and source_sequence <= last_robot_source_sequence
                ):
                    raise ValueError("robot source sequence regressed across windows")
                last_robot_source_sequence = source_sequence
                last_robot_publication = event.publication
        for event in window.quest_events:
            if not isinstance(event, ControllerPoseAcceptedEvent):
                continue
            if event.client_id != binding.quest.client_id:
                continue
            sample = event.sample
            if sample.sample_sequence <= last_quest_sample_sequence:
                raise ValueError("Quest sample sequence regressed across windows")
            if (
                last_quest_xr_frame_time_ms is not None
                and sample.xr_frame_time_ms < last_quest_xr_frame_time_ms
            ):
                raise ValueError("Quest XR frame time regressed across windows")
            if (
                last_quest_receipt_ns is not None
                and sample.received_monotonic_ns < last_quest_receipt_ns
            ):
                raise ValueError("Quest receipt time regressed across windows")
            last_quest_sample_sequence = sample.sample_sequence
            last_quest_xr_frame_time_ms = sample.xr_frame_time_ms
            last_quest_receipt_ns = sample.received_monotonic_ns

    if not windows:
        raise ValueError("capture evidence requires at least one static window")
    last_window = windows[-1]
    if (
        finalization.robot_end_next_event_sequence
        < last_window.robot_end_next_event_sequence
        or finalization.quest_end_next_event_sequence
        < last_window.quest_end_next_event_sequence
    ):
        raise ValueError("capture finalization predates the last static window")
    latest_robot = finalization.robot_cut.latest_by_stream[binding.robot.stream_id]
    if last_robot_publication is None:
        raise ValueError("capture evidence lacks a selected robot publication")
    if latest_robot.source_sequence < last_robot_publication.source_sequence:
        raise ValueError("robot final cut latest pose predates captured evidence")
    if (
        latest_robot.source_sequence == last_robot_publication.source_sequence
        and latest_robot.semantic_content_hash
        != last_robot_publication.semantic_content_hash
    ):
        raise ValueError("robot final cut reused a source sequence with new content")
    if (
        latest_robot.source_sequence > last_robot_publication.source_sequence
        and latest_robot.received_monotonic_ns
        < last_robot_publication.received_monotonic_ns
    ):
        raise ValueError("robot final cut latest receipt time regressed")
    if (
        latest_robot.source_sequence > last_robot_publication.source_sequence
        and finalization.robot_end_next_event_sequence
        == last_window.robot_end_next_event_sequence
    ):
        raise ValueError("robot final cut advanced without a journal event slot")
    bound_sessions = tuple(
        session
        for session in finalization.quest_cut.active_sessions
        if session.client_id == binding.quest.client_id
    )
    if len(bound_sessions) != 1:
        raise ValueError("Quest final cut lost the bound active session")
    final_sample_sequence = bound_sessions[0].last_accepted_sample_sequence
    if (
        final_sample_sequence is None
        or final_sample_sequence < last_quest_sample_sequence
    ):
        raise ValueError("Quest final cut sample baseline predates captured evidence")
    if (
        final_sample_sequence > last_quest_sample_sequence
        and finalization.quest_end_next_event_sequence
        == last_window.quest_end_next_event_sequence
    ):
        raise ValueError("Quest final cut advanced without a runtime event slot")


_CAPTURE_EVIDENCE_KEYS = {
    "schema",
    "profile",
    "binding",
    "windows",
    "finalization",
    "dataset",
    "dataset_semantic_sha256",
    "candidate_only",
    "calibration_accepted",
    "semantic_content_hash",
}


def parse_paired_pose_capture_evidence(
    payload: Mapping[str, Any],
) -> PairedPoseCaptureEvidence:
    """Strictly replay typed event evidence and rebuild the R0 dataset."""

    if not isinstance(payload, Mapping):
        raise TypeError("paired-pose capture evidence must be a mapping")
    _require_exact_keys(payload, _CAPTURE_EVIDENCE_KEYS, name="capture evidence")
    if payload["schema"] != CAPTURE_EVIDENCE_SCHEMA:
        raise ValueError("paired-pose capture evidence schema mismatch")
    if payload["candidate_only"] is not True:
        raise ValueError("paired-pose capture evidence must remain candidate-only")
    if payload["calibration_accepted"] is not None:
        raise ValueError("capture evidence cannot assert calibration acceptance")
    profile = _parse_profile(payload["profile"])
    binding = _parse_capture_binding(payload["binding"])
    raw_windows = payload["windows"]
    if not isinstance(raw_windows, Sequence) or isinstance(raw_windows, (str, bytes)):
        raise TypeError("capture evidence windows must be a sequence")
    if not raw_windows or len(raw_windows) > profile.max_completed_windows:
        raise ValueError("capture evidence window count is outside profile")
    windows = tuple(
        _parse_window_receipt(
            value,
            profile=profile,
            binding=binding,
            name=f"windows[{index}]",
        )
        for index, value in enumerate(raw_windows)
    )
    finalization = _parse_finalization(
        payload["finalization"],
        binding=binding,
    )
    _validate_window_chain(
        windows=windows,
        binding=binding,
        finalization=finalization,
    )
    rebuilder = ManualStaticPairedPoseCapture(profile=profile, binding=binding)
    expected_dataset = rebuilder._build_dataset(windows)
    raw_dataset = payload["dataset"]
    if not isinstance(raw_dataset, Mapping):
        raise TypeError("capture evidence dataset must be a mapping")
    dataset = parse_paired_pose_dataset(raw_dataset)
    if dataset.as_dict() != expected_dataset.as_dict():
        raise ValueError("R0 dataset does not reproduce from captured windows")
    if payload["dataset_semantic_sha256"] != dataset.semantic_sha256:
        raise ValueError("R0 dataset semantic hash mismatch")
    raw_semantic_payload = {
        key: payload[key]
        for key in _CAPTURE_EVIDENCE_KEYS
        if key != "semantic_content_hash"
    }
    if payload["semantic_content_hash"] != _semantic_hash(
        raw_semantic_payload,
        domain=CAPTURE_EVIDENCE_HASH_DOMAIN,
    ):
        raise ValueError("raw paired-pose capture evidence hash mismatch")
    evidence = PairedPoseCaptureEvidence(
        profile=profile,
        binding=binding,
        windows=windows,
        finalization=finalization,
        dataset=dataset,
    )
    if payload["semantic_content_hash"] != evidence.semantic_content_hash:
        raise ValueError("paired-pose capture evidence semantic hash mismatch")
    return evidence


__all__ = [
    "CAPTURE_BINDING_SCHEMA",
    "CAPTURE_EVIDENCE_SCHEMA",
    "CAPTURE_FINALIZATION_SCHEMA",
    "CAPTURE_PROFILE_SCHEMA",
    "STATIC_WINDOW_SCHEMA",
    "CaptureTerminalError",
    "ManualStaticPairedPoseCapture",
    "PairedPoseCaptureBinding",
    "PairedPoseCaptureEvidence",
    "PairedPoseCaptureFinalization",
    "QuestControllerCaptureBinding",
    "StaticWindowCaptureProfile",
    "StaticWindowReceipt",
    "StaticWindowRejectedError",
    "parse_paired_pose_capture_evidence",
]
