"""Serializable observation-cut records and process-local payload bundles."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from .control_admission import ControlCut
from .identity import RuntimeFactIdentity
from .immutable import deep_freeze
from .contracts import ObservationChannel


OBSERVATION_CUT_HASH_DOMAIN_V1 = "dexteleop.observation_cut.semantic.v1"
OBSERVATION_SAMPLE_PAYLOAD_HASH_DOMAIN_V1 = (
    "dexteleop.observation_sample.payload.v1"
)
OBSERVATION_SAMPLE_ID_HASH_DOMAIN_V1 = "dexteleop.observation_sample.id.v1"
RECEIVER_OBSERVATION_SAMPLE_ADMISSION_HASH_DOMAIN_V1 = (
    "dexteleop.observation_sample.receiver_admission.v1"
)


def _finite(value: float, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


@dataclass(frozen=True)
class ObservationSampleRef:
    stream_id: str
    sample_id: str
    sequence: int
    measured_at: float
    measured_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    payload_object_id: str
    valid: bool = True
    stale: bool = False
    control_admission_sequence: int | None = None
    content_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "stream_id",
            "sample_id",
            "payload_object_id",
            "measured_clock_domain_id",
            "available_clock_domain_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        if int(self.sequence) < 0:
            raise ValueError("sequence must be non-negative.")
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "measured_at", _finite(self.measured_at, field_name="measured_at"))
        object.__setattr__(self, "available_at", _finite(self.available_at, field_name="available_at"))
        if self.control_admission_sequence is not None:
            sequence = int(self.control_admission_sequence)
            if sequence < 0:
                raise ValueError("control_admission_sequence must be non-negative.")
            object.__setattr__(self, "control_admission_sequence", sequence)
        object.__setattr__(self, "content_metadata", deep_freeze(dict(self.content_metadata)))

    def semantic_payload(self, *, cutoff_at: float, cutoff_clock_domain_id: str) -> dict[str, Any]:
        measured_age = (
            cutoff_at - self.measured_at
            if self.measured_clock_domain_id == cutoff_clock_domain_id
            else None
        )
        return {
            "stream_id": self.stream_id,
            "sample_id": self.sample_id,
            "sequence": self.sequence,
            "measured_at": self.measured_at,
            "measured_clock_domain_id": self.measured_clock_domain_id,
            "available_at": self.available_at,
            "available_clock_domain_id": self.available_clock_domain_id,
            "measured_age_sec": measured_age,
            "available_age_sec": cutoff_at - self.available_at,
            "payload_object_id": self.payload_object_id,
            "valid": bool(self.valid),
            "stale": bool(self.stale),
            "control_admission_sequence": self.control_admission_sequence,
            "content_metadata": self.content_metadata,
        }


class ObservationSampleTracker:
    """Assign stable receiver-local identities to latest-value channels.

    Device sources may expose their own sequence and clock metadata, but the
    receiver cannot assume every device does so.  This tracker therefore
    assigns one monotonic sequence per observed stream and reuses the same
    ``ObservationSampleRef`` while an unchanged source sample is selected on
    multiple control steps.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last_fingerprint_by_stream: dict[str, str] = {}
        self._last_ref_by_stream: dict[str, ObservationSampleRef] = {}
        self._next_sequence_by_stream: dict[str, int] = {}

    def observe(
        self,
        channels: Mapping[str, ObservationChannel],
        *,
        available_at: float,
        available_clock_domain_id: str,
    ) -> Mapping[str, ObservationSampleRef]:
        available = _finite(available_at, field_name="available_at")
        available_clock = str(available_clock_domain_id).strip()
        if not available_clock:
            raise ValueError("available_clock_domain_id must be non-empty.")
        selected: dict[str, ObservationSampleRef] = {}
        with self._lock:
            for raw_stream_id, channel in sorted(channels.items()):
                stream_id = str(raw_stream_id)
                if not isinstance(channel, ObservationChannel):
                    raise TypeError(
                        f"Channel {stream_id!r} must be an ObservationChannel."
                    )
                if stream_id != str(channel.channel_name):
                    raise ValueError(
                        "Observation channel mapping key must match channel_name."
                    )
                payload_object_id = canonical_content_hash(
                    {
                        "capability": str(channel.capability),
                        "side": (
                            None if channel.side is None else str(channel.side)
                        ),
                        "value": channel.value,
                        "frame_name": (
                            None
                            if channel.frame_name is None
                            else str(channel.frame_name)
                        ),
                    },
                    domain=OBSERVATION_SAMPLE_PAYLOAD_HASH_DOMAIN_V1,
                )
                upstream_sequence = channel.metadata.get(
                    "sample_sequence",
                    channel.metadata.get("seq"),
                )
                fingerprint = canonical_content_hash(
                    {
                        "stream_id": stream_id,
                        "source_name": str(channel.source_name),
                        "timestamp": float(channel.timestamp),
                        "valid": bool(channel.valid),
                        "payload_object_id": payload_object_id,
                        "upstream_sequence": (
                            None
                            if upstream_sequence is None
                            else str(upstream_sequence)
                        ),
                    },
                    domain=OBSERVATION_SAMPLE_ID_HASH_DOMAIN_V1,
                )
                if (
                    self._last_fingerprint_by_stream.get(stream_id)
                    == fingerprint
                ):
                    selected[stream_id] = self._last_ref_by_stream[stream_id]
                    continue
                sequence = self._next_sequence_by_stream.get(stream_id, 0)
                self._next_sequence_by_stream[stream_id] = sequence + 1
                measured_at = channel.metadata.get(
                    "measured_at",
                    channel.metadata.get(
                        "source_timestamp",
                        channel.timestamp,
                    ),
                )
                measured_clock = str(
                    channel.metadata.get(
                        "measured_clock_domain_id",
                        channel.metadata.get(
                            "source_clock",
                            f"{channel.source_name}.source_time",
                        ),
                    )
                ).strip()
                if not measured_clock:
                    measured_clock = f"{channel.source_name}.source_time"
                sample_id = str(channel.metadata.get("sample_id") or "").strip()
                if not sample_id:
                    sample_id = canonical_content_hash(
                        {
                            "stream_id": stream_id,
                            "source_name": str(channel.source_name),
                            "sequence": sequence,
                            "fingerprint": fingerprint,
                        },
                        domain=OBSERVATION_SAMPLE_ID_HASH_DOMAIN_V1,
                    )
                ref = ObservationSampleRef(
                    stream_id=stream_id,
                    sample_id=sample_id,
                    sequence=sequence,
                    measured_at=float(measured_at),
                    measured_clock_domain_id=measured_clock,
                    available_at=available,
                    available_clock_domain_id=available_clock,
                    payload_object_id=payload_object_id,
                    valid=bool(channel.valid),
                    stale=bool(channel.metadata.get("stale", False)),
                    content_metadata={
                        "source_name": str(channel.source_name),
                        "capability": str(channel.capability),
                        "side": (
                            None if channel.side is None else str(channel.side)
                        ),
                        "frame_name": (
                            None
                            if channel.frame_name is None
                            else str(channel.frame_name)
                        ),
                        "critical": bool(channel.critical),
                        "upstream_sequence": (
                            None
                            if upstream_sequence is None
                            else str(upstream_sequence)
                        ),
                    },
                )
                self._last_fingerprint_by_stream[stream_id] = fingerprint
                self._last_ref_by_stream[stream_id] = ref
                selected[stream_id] = ref
        return MappingProxyType(selected)


class ReceiverObservationCutBuilder:
    """Build causal cuts from samples observed in another runtime boundary.

    The lower source retains ownership of its sample identity and measured
    clock.  This builder adds the receiver-local availability evidence needed
    by :class:`ObservationCutRecord` without mutating the upstream ref.  An
    unchanged ``(stream_id, sample_id, payload_object_id)`` is admitted once
    and reused by later cuts, so polling a held device value does not create a
    second native sample.
    """

    def __init__(
        self,
        *,
        producer_id: str,
        producer_epoch: int,
        observation_contract_hash: str,
        io_layout_hash: str,
    ) -> None:
        producer = str(producer_id).strip()
        contract_hash = str(observation_contract_hash).strip()
        layout_hash = str(io_layout_hash).strip()
        if not producer or not contract_hash or not layout_hash:
            raise ValueError(
                "Receiver observation producer/contract/layout identity is incomplete."
            )
        epoch = int(producer_epoch)
        if epoch < 0:
            raise ValueError("producer_epoch must be non-negative.")
        self.producer_id = producer
        self.producer_epoch = epoch
        self.observation_contract_hash = contract_hash
        self.io_layout_hash = layout_hash
        self._next_cut_sequence = 0
        self._admitted_ref_by_source_key: dict[
            tuple[str, str, str], ObservationSampleRef
        ] = {}
        self._lock = threading.RLock()

    def capture(
        self,
        coordinator: "ControlAdmissionCoordinator",
        *,
        selected_source_samples: Mapping[str, ObservationSampleRef],
        payload_handles_by_stream: Mapping[str, Any],
        trigger_reason: str,
        built_at: float | None = None,
        context_dependencies: tuple[str, ...] = (),
    ) -> "ObservationBundle":
        """Freeze one receiver-local cut and its process-local payload handles."""

        # Import locally to keep this module's public data contracts lightweight
        # and avoid a runtime import cycle in type-checking environments.
        from .control_admission import ControlAdmissionCoordinator

        if not isinstance(coordinator, ControlAdmissionCoordinator):
            raise TypeError("coordinator must be a ControlAdmissionCoordinator.")
        source_samples = dict(selected_source_samples)
        handles = dict(payload_handles_by_stream)
        if not source_samples:
            raise ValueError("Receiver observation cut requires at least one sample.")
        if set(handles) != set(source_samples):
            raise ValueError(
                "payload_handles_by_stream must exactly cover selected source samples."
            )
        for stream_id, sample in source_samples.items():
            if not isinstance(sample, ObservationSampleRef):
                raise TypeError(
                    f"selected_source_samples[{stream_id!r}] must be "
                    "ObservationSampleRef."
                )
            if str(stream_id) != sample.stream_id:
                raise ValueError(
                    "selected_source_samples key must match sample.stream_id."
                )

        with self._lock:
            capture = coordinator.capture_cut(cutoff_at=built_at)
            cut = capture.cut
            admitted: dict[str, ObservationSampleRef] = {}
            for stream_id, source in sorted(source_samples.items()):
                source_key = (
                    source.stream_id,
                    source.sample_id,
                    source.payload_object_id,
                )
                existing = self._admitted_ref_by_source_key.get(source_key)
                if existing is None:
                    admission_hash = canonical_content_hash(
                        {
                            "receiver_runtime_session_id": cut.runtime_session_id,
                            "receiver_control_domain_id": cut.control_domain_id,
                            "receiver_producer_id": self.producer_id,
                            "stream_id": source.stream_id,
                            "sample_id": source.sample_id,
                            "payload_object_id": source.payload_object_id,
                            "available_at": cut.cutoff_at,
                            "available_clock_domain_id": cut.clock_domain_id,
                            "control_admission_sequence": (
                                cut.cutoff_admission_sequence
                            ),
                        },
                        domain=(
                            RECEIVER_OBSERVATION_SAMPLE_ADMISSION_HASH_DOMAIN_V1
                        ),
                    )
                    existing = ObservationSampleRef(
                        stream_id=source.stream_id,
                        sample_id=source.sample_id,
                        sequence=source.sequence,
                        measured_at=source.measured_at,
                        measured_clock_domain_id=(
                            source.measured_clock_domain_id
                        ),
                        available_at=cut.cutoff_at,
                        available_clock_domain_id=cut.clock_domain_id,
                        payload_object_id=source.payload_object_id,
                        valid=source.valid,
                        stale=source.stale,
                        control_admission_sequence=(
                            cut.cutoff_admission_sequence
                        ),
                        content_metadata={
                            **dict(source.content_metadata),
                            "receiver_admission_hash": admission_hash,
                            "upstream_available_at": source.available_at,
                            "upstream_available_clock_domain_id": (
                                source.available_clock_domain_id
                            ),
                            "upstream_control_admission_sequence": (
                                source.control_admission_sequence
                            ),
                        },
                    )
                    self._admitted_ref_by_source_key[source_key] = existing
                elif (
                    existing.sequence != source.sequence
                    or existing.measured_at != source.measured_at
                    or existing.measured_clock_domain_id
                    != source.measured_clock_domain_id
                ):
                    raise ValueError(
                        "A source sample identity was reused with different timing "
                        "or sequence evidence."
                    )
                if existing.available_at > cut.cutoff_at:
                    raise ValueError(
                        "A cached receiver sample cannot be selected before admission."
                    )
                admitted[stream_id] = existing

            record = ObservationCutRecord.build(
                runtime_session_id=cut.runtime_session_id,
                control_cut=cut,
                policy_node_id=self.producer_id,
                policy_node_epoch=self.producer_epoch,
                sequence=self._next_cut_sequence,
                built_at=cut.cutoff_at,
                built_clock_domain_id=cut.clock_domain_id,
                selected_sample_by_stream=admitted,
                context_dependencies=context_dependencies,
                observation_contract_hash=self.observation_contract_hash,
                policy_io_layout_hash=self.io_layout_hash,
                trigger_reason=trigger_reason,
            )
            self._next_cut_sequence += 1
        return ObservationBundle(
            record=record,
            payload_handles_by_stream=handles,
        )


@dataclass(frozen=True)
class ObservationCutRecord:
    observation_cut_id: str
    runtime_session_id: str
    control_domain_id: str
    policy_node_id: str
    policy_node_epoch: int
    sequence: int
    cutoff_at: float
    cutoff_clock_domain_id: str
    cutoff_clock_reset_epoch: int
    cutoff_admission_sequence: int
    built_at: float
    built_clock_domain_id: str
    selected_sample_by_stream: Mapping[str, ObservationSampleRef]
    context_dependencies: tuple[str, ...]
    observation_contract_hash: str
    policy_io_layout_hash: str
    trigger_reason: str
    semantic_content_hash: str

    @classmethod
    def build(
        cls,
        *,
        runtime_session_id: str,
        control_cut: ControlCut,
        policy_node_id: str,
        policy_node_epoch: int,
        sequence: int,
        built_at: float,
        built_clock_domain_id: str,
        selected_sample_by_stream: Mapping[str, ObservationSampleRef],
        context_dependencies: tuple[str, ...] = (),
        observation_contract_hash: str,
        policy_io_layout_hash: str,
        trigger_reason: str,
    ) -> "ObservationCutRecord":
        if str(runtime_session_id) != control_cut.runtime_session_id:
            raise ValueError("Observation cut session must match the control cut session.")
        identity = RuntimeFactIdentity(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_cut.control_domain_id,
            producer_id=policy_node_id,
            producer_epoch=policy_node_epoch,
            sequence=sequence,
            namespace="observation_cut",
        )
        built = _finite(built_at, field_name="built_at")
        built_clock = str(built_clock_domain_id).strip()
        contract_hash = str(observation_contract_hash).strip()
        layout_hash = str(policy_io_layout_hash).strip()
        trigger = str(trigger_reason).strip()
        if not contract_hash or not layout_hash or not trigger or not built_clock:
            raise ValueError(
                "Observation contract/layout hashes, built clock, and trigger_reason must be non-empty."
            )
        if built_clock != control_cut.clock_domain_id:
            raise ValueError("Observation built_at must use the control cut clock domain.")
        if built < control_cut.cutoff_at:
            raise ValueError("Observation built_at cannot precede cutoff_at on the same clock.")
        selected: dict[str, ObservationSampleRef] = {}
        for stream_id, sample in selected_sample_by_stream.items():
            if str(stream_id) != sample.stream_id:
                raise ValueError("selected_sample_by_stream key must match sample.stream_id.")
            if sample.available_clock_domain_id != control_cut.clock_domain_id:
                raise ValueError("Sample available_at must use the control cut clock domain.")
            if sample.available_at > control_cut.cutoff_at:
                raise ValueError("Observation cut cannot select a future-available sample.")
            if (
                sample.measured_clock_domain_id == control_cut.clock_domain_id
                and sample.measured_at > control_cut.cutoff_at
            ):
                raise ValueError("Observation cut cannot select a future-measured sample.")
            if (
                sample.control_admission_sequence is not None
                and sample.control_admission_sequence > control_cut.cutoff_admission_sequence
            ):
                raise ValueError("Observation cut cannot select an uncommitted sample.")
            selected[sample.stream_id] = sample
        dependencies = tuple(str(value).strip() for value in context_dependencies)
        if any(not value for value in dependencies):
            raise ValueError("context_dependencies must contain non-empty IDs.")
        payload = {
            "observation_cut_id": identity.fact_id,
            "runtime_session_id": identity.runtime_session_id,
            "control_domain_id": identity.control_domain_id,
            "policy_node_id": identity.producer_id,
            "policy_node_epoch": identity.producer_epoch,
            "sequence": identity.sequence,
            "cutoff_at": control_cut.cutoff_at,
            "cutoff_clock_domain_id": control_cut.clock_domain_id,
            "cutoff_clock_reset_epoch": control_cut.clock_reset_epoch,
            "cutoff_admission_sequence": control_cut.cutoff_admission_sequence,
            "built_at": built,
            "built_clock_domain_id": built_clock,
            "selected_sample_by_stream": {
                stream_id: sample.semantic_payload(
                    cutoff_at=control_cut.cutoff_at,
                    cutoff_clock_domain_id=control_cut.clock_domain_id,
                )
                for stream_id, sample in selected.items()
            },
            "context_dependencies": dependencies,
            "observation_contract_hash": contract_hash,
            "policy_io_layout_hash": layout_hash,
            "trigger_reason": trigger,
        }
        semantic_hash = canonical_content_hash(payload, domain=OBSERVATION_CUT_HASH_DOMAIN_V1)
        return cls(
            observation_cut_id=identity.fact_id,
            runtime_session_id=identity.runtime_session_id,
            control_domain_id=identity.control_domain_id,
            policy_node_id=identity.producer_id,
            policy_node_epoch=identity.producer_epoch,
            sequence=identity.sequence,
            cutoff_at=control_cut.cutoff_at,
            cutoff_clock_domain_id=control_cut.clock_domain_id,
            cutoff_clock_reset_epoch=control_cut.clock_reset_epoch,
            cutoff_admission_sequence=control_cut.cutoff_admission_sequence,
            built_at=built,
            built_clock_domain_id=built_clock,
            selected_sample_by_stream=MappingProxyType(selected),
            context_dependencies=dependencies,
            observation_contract_hash=contract_hash,
            policy_io_layout_hash=layout_hash,
            trigger_reason=trigger,
            semantic_content_hash=semantic_hash,
        )


@dataclass(frozen=True)
class ObservationBundle:
    """Process-local payload handles paired with a stable cut record."""

    record: ObservationCutRecord
    payload_handles_by_stream: Mapping[str, Any]

    def __post_init__(self) -> None:
        handles = dict(self.payload_handles_by_stream)
        unknown = set(handles) - set(self.record.selected_sample_by_stream)
        if unknown:
            raise ValueError(f"Payload handles have no selected sample refs: {sorted(unknown)}.")
        object.__setattr__(self, "payload_handles_by_stream", MappingProxyType(handles))


__all__ = [
    "OBSERVATION_CUT_HASH_DOMAIN_V1",
    "OBSERVATION_SAMPLE_ID_HASH_DOMAIN_V1",
    "OBSERVATION_SAMPLE_PAYLOAD_HASH_DOMAIN_V1",
    "RECEIVER_OBSERVATION_SAMPLE_ADMISSION_HASH_DOMAIN_V1",
    "ObservationBundle",
    "ObservationCutRecord",
    "ObservationSampleRef",
    "ObservationSampleTracker",
    "ReceiverObservationCutBuilder",
]
