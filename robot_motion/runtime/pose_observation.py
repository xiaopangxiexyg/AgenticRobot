"""Passive pose-measurement bindings and receiver-local publication evidence.

The canonical measured value remains :class:`ExecutionMeasurementObservation`.
This module only binds that value to explicit frame/tool/source identities and
records when the immutable fact became visible on a receiver-monotonic clock.
It owns no device reads, FK, control authority, or calibration acceptance.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_observation import (
    ExecutionMeasurementObservation,
)

from .identity import ClockDomainDescriptor
from .observation_cut import ObservationSampleRef

POSE_COORDINATE_BINDING_HASH_DOMAIN_V1 = "dexteleop.pose_coordinate_binding.v1"
POSE_MEASUREMENT_PUBLICATION_HASH_DOMAIN_V1 = (
    "dexteleop.pose_measurement_publication.v1"
)
POSE_RECEIVER_CLOCK_DESCRIPTOR_HASH_DOMAIN_V1 = (
    "dexteleop.pose_receiver_clock_descriptor.v1"
)
POSE_MEASUREMENT_ACCEPTED_EVENT_HASH_DOMAIN_V1 = (
    "dexteleop.pose_measurement_accepted_event.v1"
)
POSE_MEASUREMENT_INVALIDATION_EVENT_HASH_DOMAIN_V1 = (
    "dexteleop.pose_measurement_invalidation_event.v1"
)
POSE_MEASUREMENT_JOURNAL_CUT_HASH_DOMAIN_V1 = (
    "dexteleop.pose_measurement_journal_cut.v1"
)
POSE_COORDINATE_PARENT_STAGE_V1 = "real_robot.pose_coordinate_binding"


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _sequence(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _positive_sequence(value: object, *, name: str) -> int:
    normalized = _sequence(value, name=name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _clock_descriptor_payload(
    descriptor: ClockDomainDescriptor,
) -> dict[str, Any]:
    if not isinstance(descriptor, ClockDomainDescriptor):
        raise TypeError("receiver_clock must be a ClockDomainDescriptor.")
    if descriptor.monotonic is not True:
        raise ValueError("receiver_clock.monotonic must be exactly True.")
    return {
        "clock_domain_id": descriptor.clock_domain_id,
        "runtime_session_id": descriptor.runtime_session_id,
        "boot_id": descriptor.boot_id,
        "reset_epoch": descriptor.reset_epoch,
        "kind": descriptor.kind,
        "resolution_sec": descriptor.resolution_sec,
        "monotonic": descriptor.monotonic,
        "mapped_clock_domain_id": descriptor.mapped_clock_domain_id,
        "offset_estimate_sec": descriptor.offset_estimate_sec,
        "offset_uncertainty_sec": descriptor.offset_uncertainty_sec,
        "mapping_sampled_from": descriptor.mapping_sampled_from,
        "mapping_sampled_until": descriptor.mapping_sampled_until,
    }


def pose_receiver_clock_descriptor_hash(
    descriptor: ClockDomainDescriptor,
) -> str:
    """Hash every field of one exact receiver-monotonic clock descriptor."""

    return canonical_content_hash(
        _clock_descriptor_payload(descriptor),
        domain=POSE_RECEIVER_CLOCK_DESCRIPTOR_HASH_DOMAIN_V1,
    )


def _sha256_reference(value: object, *, name: str) -> str:
    normalized = _required(value, name=name)
    digest = normalized.removeprefix("sha256:")
    if (
        not normalized.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must use sha256:<64 lowercase hex chars>.")
    return normalized


@dataclass(frozen=True)
class PoseCoordinateBinding:
    """Immutable source, frame, tool, FK, and observation-plan binding."""

    binding_id: str
    source_instance_id: str
    source_epoch_id: str
    source_binding_hash: str
    control_domain_id: str
    component_id: str
    stream_id: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    moving_frame_id: str
    tool_profile_hash: str
    pose_schema_hash: str
    fk_contract_id: str
    fk_contract_hash: str
    measurement_clock_domain_id: str
    receipt_clock_domain_id: str
    observation_binding_id: str
    observation_binding_hash: str
    observation_plan_hash: str
    source_kind: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "binding_id": _required(self.binding_id, name="binding_id"),
            "source_instance_id": _required(
                self.source_instance_id,
                name="source_instance_id",
            ),
            "source_epoch_id": _required(
                self.source_epoch_id,
                name="source_epoch_id",
            ),
            "source_binding_hash": _required(
                self.source_binding_hash,
                name="source_binding_hash",
            ),
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "component_id": _required(self.component_id, name="component_id"),
            "stream_id": _required(self.stream_id, name="stream_id"),
            "reference_frame_id": _required(
                self.reference_frame_id,
                name="reference_frame_id",
            ),
            "reference_frame_spec_hash": _required(
                self.reference_frame_spec_hash,
                name="reference_frame_spec_hash",
            ),
            "moving_frame_id": _required(
                self.moving_frame_id,
                name="moving_frame_id",
            ),
            "tool_profile_hash": _sha256_reference(
                self.tool_profile_hash,
                name="tool_profile_hash",
            ),
            "pose_schema_hash": _required(
                self.pose_schema_hash,
                name="pose_schema_hash",
            ),
            "fk_contract_id": _required(
                self.fk_contract_id,
                name="fk_contract_id",
            ),
            "fk_contract_hash": _required(
                self.fk_contract_hash,
                name="fk_contract_hash",
            ),
            "measurement_clock_domain_id": _required(
                self.measurement_clock_domain_id,
                name="measurement_clock_domain_id",
            ),
            "receipt_clock_domain_id": _required(
                self.receipt_clock_domain_id,
                name="receipt_clock_domain_id",
            ),
            "observation_binding_id": _required(
                self.observation_binding_id,
                name="observation_binding_id",
            ),
            "observation_binding_hash": _required(
                self.observation_binding_hash,
                name="observation_binding_hash",
            ),
            "observation_plan_hash": _required(
                self.observation_plan_hash,
                name="observation_plan_hash",
            ),
            "source_kind": _required(self.source_kind, name="source_kind"),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=POSE_COORDINATE_BINDING_HASH_DOMAIN_V1,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "source_instance_id": self.source_instance_id,
            "source_epoch_id": self.source_epoch_id,
            "source_binding_hash": self.source_binding_hash,
            "control_domain_id": self.control_domain_id,
            "component_id": self.component_id,
            "stream_id": self.stream_id,
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "moving_frame_id": self.moving_frame_id,
            "tool_profile_hash": self.tool_profile_hash,
            "pose_schema_hash": self.pose_schema_hash,
            "fk_contract_id": self.fk_contract_id,
            "fk_contract_hash": self.fk_contract_hash,
            "measurement_clock_domain_id": self.measurement_clock_domain_id,
            "receipt_clock_domain_id": self.receipt_clock_domain_id,
            "observation_binding_id": self.observation_binding_id,
            "observation_binding_hash": self.observation_binding_hash,
            "observation_plan_hash": self.observation_plan_hash,
            "source_kind": self.source_kind,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class PoseMeasurementPublication:
    """One canonical measurement plus receiver-local sample admission."""

    coordinate_binding: PoseCoordinateBinding
    source_sequence: int
    measurement: ExecutionMeasurementObservation
    sample_ref: ObservationSampleRef
    received_monotonic_ns: int
    receiver_clock: ClockDomainDescriptor
    receiver_clock_descriptor_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_binding, PoseCoordinateBinding):
            raise TypeError("coordinate_binding must be a PoseCoordinateBinding.")
        if not isinstance(self.measurement, ExecutionMeasurementObservation):
            raise TypeError("measurement must be an ExecutionMeasurementObservation.")
        if not isinstance(self.sample_ref, ObservationSampleRef):
            raise TypeError("sample_ref must be an ObservationSampleRef.")
        source_sequence = _sequence(
            self.source_sequence,
            name="source_sequence",
        )
        received_monotonic_ns = _sequence(
            self.received_monotonic_ns,
            name="received_monotonic_ns",
        )
        receiver_clock_payload = _clock_descriptor_payload(self.receiver_clock)
        receiver_clock_hash = pose_receiver_clock_descriptor_hash(self.receiver_clock)
        binding = self.coordinate_binding
        measurement = self.measurement
        sample = self.sample_ref
        if set(measurement.value_by_component) != {binding.component_id}:
            raise ValueError(
                "Pose measurement must contain exactly its bound component."
            )
        if (
            measurement.representation_by_component[binding.component_id]
            != "cartesian_pose"
            or measurement.schema_hash_by_component[binding.component_id]
            != binding.pose_schema_hash
        ):
            raise ValueError("Pose representation/schema does not match its binding.")
        pose_values = tuple(measurement.value_by_component[binding.component_id])
        if len(pose_values) != 7 or any(
            not math.isfinite(float(value)) for value in pose_values
        ):
            raise ValueError("Cartesian pose must be exactly seven finite values.")
        quaternion_norm = math.hypot(*(float(value) for value in pose_values[3:]))
        if not math.isfinite(quaternion_norm) or not math.isclose(
            quaternion_norm,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("Cartesian pose quaternion must be normalized.")
        if (
            measurement.control_domain_id != binding.control_domain_id
            or measurement.clock_domain_id != binding.measurement_clock_domain_id
            or measurement.observation_binding_id != binding.observation_binding_id
            or measurement.observation_plan_hash != binding.observation_plan_hash
        ):
            raise ValueError("Pose measurement observation binding is inconsistent.")
        coordinate_parents = tuple(
            parent
            for parent in measurement.parent_facts
            if parent.stage == POSE_COORDINATE_PARENT_STAGE_V1
        )
        if len(coordinate_parents) != 1:
            raise ValueError(
                "Pose measurement must contain exactly one coordinate binding fact."
            )
        coordinate_parent = coordinate_parents[0]
        if (
            coordinate_parent.fact_id != binding.binding_id
            or coordinate_parent.fact_hash != binding.semantic_content_hash
            or coordinate_parent.status != "applied"
            or coordinate_parent.clock_domain_id != binding.measurement_clock_domain_id
            or coordinate_parent.observed_at != measurement.available_at
            or any(
                parent.fact_id == binding.binding_id and parent is not coordinate_parent
                for parent in measurement.parent_facts
            )
        ):
            raise ValueError("Pose coordinate binding parent fact is inconsistent.")
        if self.receiver_clock.clock_domain_id != binding.receipt_clock_domain_id:
            raise ValueError(
                "Receiver clock descriptor does not match the coordinate binding."
            )
        received_at = received_monotonic_ns / 1_000_000_000.0
        if (
            sample.stream_id != binding.stream_id
            or sample.sample_id != measurement.measurement_id
            or sample.sequence != source_sequence
            or sample.measured_at != measurement.measured_at
            or sample.measured_clock_domain_id != measurement.clock_domain_id
            or sample.available_clock_domain_id != binding.receipt_clock_domain_id
            or sample.available_at != received_at
            or sample.payload_object_id != measurement.semantic_content_hash
            or not sample.valid
            or sample.stale
        ):
            raise ValueError("Pose sample reference does not match its measurement.")
        metadata = sample.content_metadata
        if (
            metadata.get("coordinate_binding_hash") != binding.semantic_content_hash
            or metadata.get("source_instance_id") != binding.source_instance_id
            or metadata.get("source_epoch_id") != binding.source_epoch_id
            or metadata.get("source_kind") != binding.source_kind
            or metadata.get("source_sequence") != source_sequence
            or metadata.get("reference_frame_id") != binding.reference_frame_id
            or metadata.get("reference_frame_spec_hash")
            != binding.reference_frame_spec_hash
            or metadata.get("moving_frame_id") != binding.moving_frame_id
            or metadata.get("tool_profile_hash") != binding.tool_profile_hash
            or metadata.get("measurement_available_at") != measurement.available_at
            or metadata.get("measurement_available_clock_domain_id")
            != measurement.clock_domain_id
            or metadata.get("received_monotonic_ns") != received_monotonic_ns
        ):
            raise ValueError("Pose sample reference lineage is incomplete.")
        sample_payload = {
            "stream_id": sample.stream_id,
            "sample_id": sample.sample_id,
            "sequence": sample.sequence,
            "measured_at": sample.measured_at,
            "measured_clock_domain_id": sample.measured_clock_domain_id,
            "available_at": sample.available_at,
            "available_clock_domain_id": sample.available_clock_domain_id,
            "payload_object_id": sample.payload_object_id,
            "valid": sample.valid,
            "stale": sample.stale,
            "control_admission_sequence": sample.control_admission_sequence,
            "content_metadata": dict(sample.content_metadata),
        }
        payload = {
            "coordinate_binding_hash": binding.semantic_content_hash,
            "source_sequence": source_sequence,
            "measurement_hash": measurement.semantic_content_hash,
            "received_monotonic_ns": received_monotonic_ns,
            "receiver_clock": receiver_clock_payload,
            "receiver_clock_descriptor_hash": receiver_clock_hash,
            "sample_ref": sample_payload,
        }
        object.__setattr__(self, "source_sequence", source_sequence)
        object.__setattr__(
            self,
            "received_monotonic_ns",
            received_monotonic_ns,
        )
        object.__setattr__(
            self,
            "receiver_clock_descriptor_hash",
            receiver_clock_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=POSE_MEASUREMENT_PUBLICATION_HASH_DOMAIN_V1,
            ),
        )

    def as_evidence_dict(self) -> dict[str, Any]:
        measurement = self.measurement
        sample = self.sample_ref
        return {
            "coordinate_binding": self.coordinate_binding.as_dict(),
            "source_sequence": self.source_sequence,
            "received_monotonic_ns": self.received_monotonic_ns,
            "receiver_clock": {
                **_clock_descriptor_payload(self.receiver_clock),
                "semantic_content_hash": self.receiver_clock_descriptor_hash,
            },
            "measurement": {
                "measurement_id": measurement.measurement_id,
                "source_measurement_hash": measurement.source_measurement_hash,
                "control_domain_id": measurement.control_domain_id,
                "measured_at": measurement.measured_at,
                "available_at": measurement.available_at,
                "clock_domain_id": measurement.clock_domain_id,
                "value_by_component": {
                    key: list(value)
                    for key, value in measurement.value_by_component.items()
                },
                "representation_by_component": dict(
                    measurement.representation_by_component
                ),
                "schema_hash_by_component": dict(measurement.schema_hash_by_component),
                "observation_binding_id": measurement.observation_binding_id,
                "observation_plan_hash": measurement.observation_plan_hash,
                "parent_facts": [
                    {
                        "stage": parent.stage,
                        "fact_id": parent.fact_id,
                        "fact_hash": parent.fact_hash,
                        "observed_at": parent.observed_at,
                        "clock_domain_id": parent.clock_domain_id,
                        "status": parent.status,
                        "semantic_content_hash": parent.semantic_content_hash,
                    }
                    for parent in measurement.parent_facts
                ],
                "semantic_content_hash": measurement.semantic_content_hash,
            },
            "sample_ref": {
                "stream_id": sample.stream_id,
                "sample_id": sample.sample_id,
                "sequence": sample.sequence,
                "measured_at": sample.measured_at,
                "measured_clock_domain_id": sample.measured_clock_domain_id,
                "available_at": sample.available_at,
                "available_clock_domain_id": sample.available_clock_domain_id,
                "payload_object_id": sample.payload_object_id,
                "valid": sample.valid,
                "stale": sample.stale,
                "control_admission_sequence": sample.control_admission_sequence,
                "content_metadata": dict(sample.content_metadata),
            },
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class PoseMeasurementAcceptedEvent:
    """One publication admitted into the bounded receiver-local journal."""

    event_sequence: int
    publication: PoseMeasurementPublication
    receiver_clock_descriptor_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        event_sequence = _positive_sequence(
            self.event_sequence,
            name="event_sequence",
        )
        if not isinstance(self.publication, PoseMeasurementPublication):
            raise TypeError("publication must be a PoseMeasurementPublication.")
        clock_hash = self.publication.receiver_clock_descriptor_hash
        payload = {
            "event_kind": "accepted",
            "event_sequence": event_sequence,
            "publication_hash": self.publication.semantic_content_hash,
            "receiver_clock_descriptor_hash": clock_hash,
        }
        object.__setattr__(self, "event_sequence", event_sequence)
        object.__setattr__(
            self,
            "receiver_clock_descriptor_hash",
            clock_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=POSE_MEASUREMENT_ACCEPTED_EVENT_HASH_DOMAIN_V1,
            ),
        )

    def as_evidence_dict(self) -> dict[str, Any]:
        return {
            "event_kind": "accepted",
            "event_sequence": self.event_sequence,
            "publication": self.publication.as_evidence_dict(),
            "receiver_clock_descriptor_hash": (self.receiver_clock_descriptor_hash),
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class PoseMeasurementInvalidationEvent:
    """Invalidation of one exact binding, optionally naming its recovery binding."""

    event_sequence: int
    coordinate_binding: PoseCoordinateBinding
    reason: str
    received_monotonic_ns: int
    receiver_clock: ClockDomainDescriptor
    invalidated_source_sequence: int | None = None
    next_coordinate_binding: PoseCoordinateBinding | None = None
    receiver_clock_descriptor_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        event_sequence = _positive_sequence(
            self.event_sequence,
            name="event_sequence",
        )
        if not isinstance(self.coordinate_binding, PoseCoordinateBinding):
            raise TypeError("coordinate_binding must be a PoseCoordinateBinding.")
        reason = _required(self.reason, name="reason")
        received_monotonic_ns = _sequence(
            self.received_monotonic_ns,
            name="received_monotonic_ns",
        )
        clock_payload = _clock_descriptor_payload(self.receiver_clock)
        if (
            self.receiver_clock.clock_domain_id
            != self.coordinate_binding.receipt_clock_domain_id
        ):
            raise ValueError("Invalidation receiver clock does not match its binding.")
        invalidated_source_sequence = self.invalidated_source_sequence
        if invalidated_source_sequence is not None:
            invalidated_source_sequence = _sequence(
                invalidated_source_sequence,
                name="invalidated_source_sequence",
            )
        next_binding = self.next_coordinate_binding
        if next_binding is not None:
            if not isinstance(next_binding, PoseCoordinateBinding):
                raise TypeError(
                    "next_coordinate_binding must be a PoseCoordinateBinding."
                )
            if next_binding.stream_id != self.coordinate_binding.stream_id:
                raise ValueError(
                    "A pose binding transition must remain on the same stream."
                )
            if (
                next_binding.receipt_clock_domain_id
                != self.receiver_clock.clock_domain_id
            ):
                raise ValueError(
                    "Next binding must use the same receiver clock domain."
                )
            current_source = (
                self.coordinate_binding.source_instance_id,
                self.coordinate_binding.source_epoch_id,
            )
            next_source = (
                next_binding.source_instance_id,
                next_binding.source_epoch_id,
            )
            if (
                next_source == current_source
                and next_binding != self.coordinate_binding
            ):
                raise ValueError(
                    "Binding drift within one source instance/epoch is forbidden."
                )
        clock_hash = pose_receiver_clock_descriptor_hash(self.receiver_clock)
        payload = {
            "event_kind": "invalidation",
            "event_sequence": event_sequence,
            "coordinate_binding_hash": (self.coordinate_binding.semantic_content_hash),
            "reason": reason,
            "received_monotonic_ns": received_monotonic_ns,
            "receiver_clock": clock_payload,
            "receiver_clock_descriptor_hash": clock_hash,
            "invalidated_source_sequence": invalidated_source_sequence,
            "next_coordinate_binding_hash": (
                None if next_binding is None else next_binding.semantic_content_hash
            ),
        }
        object.__setattr__(self, "event_sequence", event_sequence)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "received_monotonic_ns",
            received_monotonic_ns,
        )
        object.__setattr__(
            self,
            "invalidated_source_sequence",
            invalidated_source_sequence,
        )
        object.__setattr__(
            self,
            "receiver_clock_descriptor_hash",
            clock_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=POSE_MEASUREMENT_INVALIDATION_EVENT_HASH_DOMAIN_V1,
            ),
        )

    def as_evidence_dict(self) -> dict[str, Any]:
        return {
            "event_kind": "invalidation",
            "event_sequence": self.event_sequence,
            "coordinate_binding": self.coordinate_binding.as_dict(),
            "reason": self.reason,
            "received_monotonic_ns": self.received_monotonic_ns,
            "receiver_clock": {
                **_clock_descriptor_payload(self.receiver_clock),
                "semantic_content_hash": self.receiver_clock_descriptor_hash,
            },
            "invalidated_source_sequence": self.invalidated_source_sequence,
            "next_coordinate_binding": (
                None
                if self.next_coordinate_binding is None
                else self.next_coordinate_binding.as_dict()
            ),
            "semantic_content_hash": self.semantic_content_hash,
        }


PoseMeasurementJournalEvent = (
    PoseMeasurementAcceptedEvent | PoseMeasurementInvalidationEvent
)


@dataclass(frozen=True)
class PoseMeasurementJournalCut:
    """Atomic journal cursor and latest-value snapshot at one lock boundary."""

    event_sequence: int
    oldest_retained_event_sequence: int | None
    receiver_clock: ClockDomainDescriptor
    latest_by_stream: Mapping[str, PoseMeasurementPublication]
    binding_hash_by_stream: Mapping[str, str]
    invalidated_stream_ids: tuple[str, ...]
    expected_next_binding_hash_by_stream: Mapping[str, str]
    receiver_clock_descriptor_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        event_sequence = _sequence(self.event_sequence, name="event_sequence")
        oldest = self.oldest_retained_event_sequence
        if oldest is not None:
            oldest = _positive_sequence(
                oldest,
                name="oldest_retained_event_sequence",
            )
            if oldest > event_sequence:
                raise ValueError(
                    "oldest retained event cannot exceed the cut sequence."
                )
        clock_payload = _clock_descriptor_payload(self.receiver_clock)
        clock_hash = pose_receiver_clock_descriptor_hash(self.receiver_clock)
        latest: dict[str, PoseMeasurementPublication] = {}
        for raw_stream_id, publication in self.latest_by_stream.items():
            stream_id = _required(raw_stream_id, name="latest stream_id")
            if not isinstance(publication, PoseMeasurementPublication):
                raise TypeError(
                    "latest_by_stream values must be PoseMeasurementPublication."
                )
            if publication.coordinate_binding.stream_id != stream_id:
                raise ValueError("Latest publication stream key is inconsistent.")
            if publication.receiver_clock != self.receiver_clock:
                raise ValueError("Latest publication uses another receiver clock.")
            latest[stream_id] = publication
        binding_hashes = {
            _required(stream_id, name="binding stream_id"): _required(
                binding_hash,
                name="binding hash",
            )
            for stream_id, binding_hash in self.binding_hash_by_stream.items()
        }
        if set(latest) - set(binding_hashes):
            raise ValueError("Every latest stream must have an active binding hash.")
        invalidated_stream_ids = tuple(
            sorted(
                {
                    _required(stream_id, name="invalidated stream_id")
                    for stream_id in self.invalidated_stream_ids
                }
            )
        )
        if len(invalidated_stream_ids) != len(self.invalidated_stream_ids):
            raise ValueError("invalidated_stream_ids must be unique.")
        if set(invalidated_stream_ids) & set(latest):
            raise ValueError("An invalidated stream cannot retain a latest pose.")
        expected_next = {
            _required(stream_id, name="expected-next stream_id"): _required(
                binding_hash,
                name="expected-next binding hash",
            )
            for stream_id, binding_hash in (
                self.expected_next_binding_hash_by_stream.items()
            )
        }
        if set(expected_next) - set(invalidated_stream_ids):
            raise ValueError(
                "Only invalidated streams may name an expected next binding."
            )
        latest_hashes = {
            stream_id: publication.semantic_content_hash
            for stream_id, publication in sorted(latest.items())
        }
        payload = {
            "event_sequence": event_sequence,
            "oldest_retained_event_sequence": oldest,
            "receiver_clock": clock_payload,
            "receiver_clock_descriptor_hash": clock_hash,
            "latest_publication_hash_by_stream": latest_hashes,
            "binding_hash_by_stream": dict(sorted(binding_hashes.items())),
            "invalidated_stream_ids": invalidated_stream_ids,
            "expected_next_binding_hash_by_stream": dict(sorted(expected_next.items())),
        }
        object.__setattr__(self, "event_sequence", event_sequence)
        object.__setattr__(self, "oldest_retained_event_sequence", oldest)
        object.__setattr__(
            self,
            "latest_by_stream",
            MappingProxyType(dict(sorted(latest.items()))),
        )
        object.__setattr__(
            self,
            "binding_hash_by_stream",
            MappingProxyType(dict(sorted(binding_hashes.items()))),
        )
        object.__setattr__(
            self,
            "invalidated_stream_ids",
            invalidated_stream_ids,
        )
        object.__setattr__(
            self,
            "expected_next_binding_hash_by_stream",
            MappingProxyType(dict(sorted(expected_next.items()))),
        )
        object.__setattr__(
            self,
            "receiver_clock_descriptor_hash",
            clock_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=POSE_MEASUREMENT_JOURNAL_CUT_HASH_DOMAIN_V1,
            ),
        )


class PoseMeasurementJournalGapError(RuntimeError):
    """Requested cursor predates the bounded journal's retained prefix."""

    def __init__(
        self,
        *,
        requested_after: int,
        oldest_retained: int,
        latest: int,
    ) -> None:
        self.requested_after = _sequence(
            requested_after,
            name="requested_after",
        )
        self.oldest_retained = _positive_sequence(
            oldest_retained,
            name="oldest_retained",
        )
        self.latest = _sequence(latest, name="latest")
        super().__init__(
            "pose measurement journal gap: requested events after "
            f"{self.requested_after}, but retained events start at "
            f"{self.oldest_retained} and latest is {self.latest}"
        )


@dataclass
class _PoseMeasurementStreamState:
    coordinate_binding: PoseCoordinateBinding
    latest: PoseMeasurementPublication | None = None
    last_source_sequence: int | None = None
    last_publication_hash: str | None = None
    invalidated: bool = False
    expected_next_binding: PoseCoordinateBinding | None = None


class PassivePoseMeasurementJournal:
    """Bounded producer-nonblocking journal of passive pose facts.

    Producers only validate immutable values and append under a bounded local
    critical section.  The journal invokes no external callback and starts no
    worker thread.  Consumers take an atomic cut and later call
    :meth:`events_after`; falling behind the retained deque fails closed.
    """

    def __init__(
        self,
        *,
        receiver_clock: ClockDomainDescriptor,
        max_events: int = 256,
        max_streams: int = 8,
    ) -> None:
        _clock_descriptor_payload(receiver_clock)
        if isinstance(max_events, bool) or not isinstance(max_events, int):
            raise TypeError("max_events must be an integer.")
        if max_events <= 0:
            raise ValueError("max_events must be positive.")
        if isinstance(max_streams, bool) or not isinstance(max_streams, int):
            raise TypeError("max_streams must be an integer.")
        if max_streams <= 0:
            raise ValueError("max_streams must be positive.")
        self.receiver_clock = receiver_clock
        self.receiver_clock_descriptor_hash = pose_receiver_clock_descriptor_hash(
            receiver_clock
        )
        self.max_events = max_events
        self.max_streams = max_streams
        self._lock = threading.RLock()
        self._event_sequence = 0
        self._events: deque[PoseMeasurementJournalEvent] = deque(maxlen=max_events)
        self._state_by_stream: dict[str, _PoseMeasurementStreamState] = {}
        self._last_received_monotonic_ns: int | None = None
        self.accepted_count = 0
        self.invalidation_count = 0

    def _validate_publication_clock(
        self,
        publication: PoseMeasurementPublication,
    ) -> None:
        if publication.receiver_clock != self.receiver_clock:
            raise ValueError(
                "Pose publication receiver clock differs from the journal."
            )
        if (
            publication.receiver_clock_descriptor_hash
            != self.receiver_clock_descriptor_hash
        ):
            raise ValueError(
                "Pose publication receiver clock hash differs from the journal."
            )

    def _validate_receipt_order_locked(
        self,
        received_monotonic_ns: int,
    ) -> None:
        previous = self._last_received_monotonic_ns
        if previous is not None and received_monotonic_ns < previous:
            raise ValueError("pose measurement receiver-monotonic timestamp regressed.")

    @staticmethod
    def _raise_binding_transition_error(
        current: PoseCoordinateBinding,
        incoming: PoseCoordinateBinding,
    ) -> None:
        current_source = (
            current.source_instance_id,
            current.source_epoch_id,
        )
        incoming_source = (
            incoming.source_instance_id,
            incoming.source_epoch_id,
        )
        if incoming_source == current_source:
            raise ValueError(
                "Pose coordinate binding drifted within one source instance/epoch."
            )
        raise ValueError(
            "Pose source instance/epoch changed without explicit invalidation."
        )

    def publish(
        self,
        publication: PoseMeasurementPublication,
    ) -> PoseMeasurementAcceptedEvent | None:
        """Admit one new source sequence; exact duplicates are idempotent."""

        if not isinstance(publication, PoseMeasurementPublication):
            raise TypeError("publication must be a PoseMeasurementPublication.")
        self._validate_publication_clock(publication)
        binding = publication.coordinate_binding
        stream_id = binding.stream_id
        with self._lock:
            state = self._state_by_stream.get(stream_id)
            replacement_state: _PoseMeasurementStreamState | None = None
            if state is None:
                if len(self._state_by_stream) >= self.max_streams:
                    raise RuntimeError("pose measurement stream limit reached.")
                state = _PoseMeasurementStreamState(
                    coordinate_binding=binding,
                )
                replacement_state = state
            elif state.invalidated:
                expected = state.expected_next_binding
                if expected is None:
                    raise RuntimeError(
                        "Invalidated pose stream requires an explicit next binding."
                    )
                if binding != expected:
                    raise ValueError(
                        "Pose publication does not match the explicit next binding."
                    )
                if binding != state.coordinate_binding:
                    state = _PoseMeasurementStreamState(
                        coordinate_binding=binding,
                    )
                    replacement_state = state
            elif binding != state.coordinate_binding:
                self._raise_binding_transition_error(
                    state.coordinate_binding,
                    binding,
                )
            previous_sequence = state.last_source_sequence
            if previous_sequence is not None:
                if publication.source_sequence < previous_sequence:
                    raise ValueError("pose measurement source sequence regressed.")
                if publication.source_sequence == previous_sequence:
                    if publication.semantic_content_hash != state.last_publication_hash:
                        raise ValueError(
                            "pose source sequence was reused with different content."
                        )
                    return None
            self._validate_receipt_order_locked(
                publication.received_monotonic_ns,
            )
            event_sequence = self._event_sequence + 1
            event = PoseMeasurementAcceptedEvent(
                event_sequence=event_sequence,
                publication=publication,
            )
            if replacement_state is not None:
                self._state_by_stream[stream_id] = replacement_state
                state = replacement_state
            state.latest = publication
            state.last_source_sequence = publication.source_sequence
            state.last_publication_hash = publication.semantic_content_hash
            state.invalidated = False
            state.expected_next_binding = None
            self._event_sequence = event_sequence
            self._events.append(event)
            self._last_received_monotonic_ns = publication.received_monotonic_ns
            self.accepted_count += 1
            return event

    def invalidate(
        self,
        *,
        coordinate_binding: PoseCoordinateBinding,
        reason: str,
        received_monotonic_ns: int,
        next_coordinate_binding: PoseCoordinateBinding | None = None,
    ) -> PoseMeasurementInvalidationEvent | None:
        """Clear latest and optionally authorize one exact next source binding."""

        if not isinstance(coordinate_binding, PoseCoordinateBinding):
            raise TypeError("coordinate_binding must be a PoseCoordinateBinding.")
        reason = _required(reason, name="reason")
        received_monotonic_ns = _sequence(
            received_monotonic_ns,
            name="received_monotonic_ns",
        )
        if (
            coordinate_binding.receipt_clock_domain_id
            != self.receiver_clock.clock_domain_id
        ):
            raise ValueError("Invalidation binding uses another receiver clock domain.")
        if next_coordinate_binding is not None:
            if not isinstance(next_coordinate_binding, PoseCoordinateBinding):
                raise TypeError(
                    "next_coordinate_binding must be a PoseCoordinateBinding."
                )
            if next_coordinate_binding.stream_id != coordinate_binding.stream_id:
                raise ValueError(
                    "A pose binding transition must remain on the same stream."
                )
            if (
                next_coordinate_binding.receipt_clock_domain_id
                != self.receiver_clock.clock_domain_id
            ):
                raise ValueError("Next binding uses another receiver clock domain.")
            if (
                next_coordinate_binding.source_instance_id,
                next_coordinate_binding.source_epoch_id,
            ) == (
                coordinate_binding.source_instance_id,
                coordinate_binding.source_epoch_id,
            ) and next_coordinate_binding != coordinate_binding:
                raise ValueError(
                    "Binding drift within one source instance/epoch is forbidden."
                )
        stream_id = coordinate_binding.stream_id
        with self._lock:
            state = self._state_by_stream.get(stream_id)
            if state is None:
                if len(self._state_by_stream) >= self.max_streams:
                    raise RuntimeError("pose measurement stream limit reached.")
                state = _PoseMeasurementStreamState(
                    coordinate_binding=coordinate_binding,
                )
            elif coordinate_binding != state.coordinate_binding:
                self._raise_binding_transition_error(
                    state.coordinate_binding,
                    coordinate_binding,
                )
            elif state.invalidated:
                if next_coordinate_binding == state.expected_next_binding:
                    return None
                if not (
                    next_coordinate_binding is None
                    and state.expected_next_binding is not None
                ):
                    raise ValueError(
                        "Cannot change an invalidated stream's expected next binding."
                    )
            self._validate_receipt_order_locked(received_monotonic_ns)
            event_sequence = self._event_sequence + 1
            event = PoseMeasurementInvalidationEvent(
                event_sequence=event_sequence,
                coordinate_binding=coordinate_binding,
                reason=reason,
                received_monotonic_ns=received_monotonic_ns,
                receiver_clock=self.receiver_clock,
                invalidated_source_sequence=state.last_source_sequence,
                next_coordinate_binding=next_coordinate_binding,
            )
            state.latest = None
            state.invalidated = True
            state.expected_next_binding = next_coordinate_binding
            self._state_by_stream[stream_id] = state
            self._event_sequence = event_sequence
            self._events.append(event)
            self._last_received_monotonic_ns = received_monotonic_ns
            self.invalidation_count += 1
            return event

    def latest(self, *, stream_id: str) -> PoseMeasurementPublication | None:
        stream_id = _required(stream_id, name="stream_id")
        with self._lock:
            state = self._state_by_stream.get(stream_id)
            return None if state is None else state.latest

    def atomic_cut(self) -> PoseMeasurementJournalCut:
        with self._lock:
            oldest = None if not self._events else self._events[0].event_sequence
            latest = {
                stream_id: state.latest
                for stream_id, state in self._state_by_stream.items()
                if state.latest is not None
            }
            return PoseMeasurementJournalCut(
                event_sequence=self._event_sequence,
                oldest_retained_event_sequence=oldest,
                receiver_clock=self.receiver_clock,
                latest_by_stream=latest,
                binding_hash_by_stream={
                    stream_id: state.coordinate_binding.semantic_content_hash
                    for stream_id, state in self._state_by_stream.items()
                },
                invalidated_stream_ids=tuple(
                    stream_id
                    for stream_id, state in self._state_by_stream.items()
                    if state.invalidated
                ),
                expected_next_binding_hash_by_stream={
                    stream_id: state.expected_next_binding.semantic_content_hash
                    for stream_id, state in self._state_by_stream.items()
                    if state.expected_next_binding is not None
                },
            )

    def events_after(
        self,
        event_sequence: int,
    ) -> tuple[PoseMeasurementJournalEvent, ...]:
        """Return the retained suffix, or fail closed if its prefix was lost."""

        event_sequence = _sequence(event_sequence, name="event_sequence")
        with self._lock:
            latest = self._event_sequence
            if event_sequence > latest:
                raise ValueError("event_sequence cannot be newer than the journal.")
            if event_sequence == latest:
                return ()
            if not self._events:
                raise RuntimeError(
                    "pose measurement journal lost all requested events."
                )
            oldest = self._events[0].event_sequence
            if event_sequence < oldest - 1:
                raise PoseMeasurementJournalGapError(
                    requested_after=event_sequence,
                    oldest_retained=oldest,
                    latest=latest,
                )
            return tuple(
                event for event in self._events if event.event_sequence > event_sequence
            )

    def snapshot_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": "passive_bounded_event_journal",
                "receiver_clock_descriptor_hash": (self.receiver_clock_descriptor_hash),
                "event_sequence": self._event_sequence,
                "oldest_retained_event_sequence": (
                    None if not self._events else self._events[0].event_sequence
                ),
                "last_received_monotonic_ns": self._last_received_monotonic_ns,
                "retained_event_count": len(self._events),
                "max_events": self.max_events,
                "stream_count": len(self._state_by_stream),
                "max_streams": self.max_streams,
                "accepted_count": self.accepted_count,
                "invalidation_count": self.invalidation_count,
                "streams": {
                    stream_id: {
                        "binding_hash": (
                            state.coordinate_binding.semantic_content_hash
                        ),
                        "source_instance_id": (
                            state.coordinate_binding.source_instance_id
                        ),
                        "source_epoch_id": (state.coordinate_binding.source_epoch_id),
                        "last_source_sequence": state.last_source_sequence,
                        "invalidated": state.invalidated,
                        "has_latest": state.latest is not None,
                        "expected_next_binding_hash": (
                            None
                            if state.expected_next_binding is None
                            else state.expected_next_binding.semantic_content_hash
                        ),
                    }
                    for stream_id, state in self._state_by_stream.items()
                },
            }


__all__ = [
    "POSE_COORDINATE_BINDING_HASH_DOMAIN_V1",
    "POSE_COORDINATE_PARENT_STAGE_V1",
    "POSE_MEASUREMENT_ACCEPTED_EVENT_HASH_DOMAIN_V1",
    "POSE_MEASUREMENT_INVALIDATION_EVENT_HASH_DOMAIN_V1",
    "POSE_MEASUREMENT_JOURNAL_CUT_HASH_DOMAIN_V1",
    "POSE_MEASUREMENT_PUBLICATION_HASH_DOMAIN_V1",
    "POSE_RECEIVER_CLOCK_DESCRIPTOR_HASH_DOMAIN_V1",
    "PassivePoseMeasurementJournal",
    "PoseCoordinateBinding",
    "PoseMeasurementAcceptedEvent",
    "PoseMeasurementInvalidationEvent",
    "PoseMeasurementJournalCut",
    "PoseMeasurementJournalEvent",
    "PoseMeasurementJournalGapError",
    "PoseMeasurementPublication",
    "pose_receiver_clock_descriptor_hash",
]
