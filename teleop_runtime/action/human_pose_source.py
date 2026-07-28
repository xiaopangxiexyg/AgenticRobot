"""Selective observation-channel adaptation for human pose residual sources.

The adapter reads only predeclared pose channels and pairs each value with the
receiver-issued ``ObservationSampleRef`` that identifies it. It is independent
of VIVE, a simulator, robot topology, and the later residual/composition logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime import ObservationChannel
from robot_motion.runtime.observation_cut import (
    OBSERVATION_SAMPLE_PAYLOAD_HASH_DOMAIN_V1,
    ObservationSampleRef,
)

from .human_residual import HumanPoseSample


HUMAN_POSE_CHANNEL_ADAPTER_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_pose_channel_adapter.development"
)
HUMAN_POSE_CHANNEL_ADAPTATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_pose_channel_adaptation_receipt.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class HumanPoseChannelBinding:
    component_id: str
    channel_name: str
    source_controlled_frame_id: str
    expected_reference_frame_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "component_id",
            "channel_name",
            "source_controlled_frame_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        if self.expected_reference_frame_id is not None:
            object.__setattr__(
                self,
                "expected_reference_frame_id",
                _required(
                    self.expected_reference_frame_id,
                    name="expected_reference_frame_id",
                ),
            )

    def semantic_payload(self) -> dict[str, str | None]:
        return {
            "component_id": self.component_id,
            "channel_name": self.channel_name,
            "source_controlled_frame_id": self.source_controlled_frame_id,
            "expected_reference_frame_id": self.expected_reference_frame_id,
        }


@dataclass(frozen=True)
class HumanPoseChannelAdapterConfig:
    adapter_id: str
    source_coordinate_binding_id: str
    source_coordinate_binding_hash: str
    component_bindings: tuple[HumanPoseChannelBinding, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "adapter_id",
                "source_coordinate_binding_id",
                "source_coordinate_binding_hash",
            )
        }
        bindings = tuple(self.component_bindings)
        if not bindings or any(
            not isinstance(item, HumanPoseChannelBinding)
            for item in bindings
        ):
            raise TypeError(
                "component_bindings must contain at least one "
                "HumanPoseChannelBinding."
            )
        ordered = tuple(sorted(bindings, key=lambda item: item.component_id))
        component_ids = tuple(item.component_id for item in ordered)
        channel_names = tuple(item.channel_name for item in ordered)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("Human pose component IDs must be unique.")
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("One pose channel cannot feed multiple components.")
        payload = {
            **values,
            "component_bindings": tuple(
                item.semantic_payload() for item in ordered
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "component_bindings", ordered)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=HUMAN_POSE_CHANNEL_ADAPTER_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


@dataclass(frozen=True)
class HumanPoseChannelAdaptationReceipt:
    adapter_contract_hash: str
    adapted_at: float
    adapted_clock_domain_id: str
    source_coordinate_binding_id: str
    source_coordinate_binding_hash: str
    source_sample_id_by_component: Mapping[str, str]
    source_payload_object_id_by_component: Mapping[str, str]
    output_pose_sample_hash_by_component: Mapping[str, str]
    semantic_content_hash: str = field(init=False)
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "adapter_contract_hash": _required(
                self.adapter_contract_hash,
                name="adapter_contract_hash",
            ),
            "adapted_at": _finite(self.adapted_at, name="adapted_at"),
            "adapted_clock_domain_id": _required(
                self.adapted_clock_domain_id,
                name="adapted_clock_domain_id",
            ),
            "source_coordinate_binding_id": _required(
                self.source_coordinate_binding_id,
                name="source_coordinate_binding_id",
            ),
            "source_coordinate_binding_hash": _required(
                self.source_coordinate_binding_hash,
                name="source_coordinate_binding_hash",
            ),
            "source_sample_id_by_component": dict(
                self.source_sample_id_by_component
            ),
            "source_payload_object_id_by_component": dict(
                self.source_payload_object_id_by_component
            ),
            "output_pose_sample_hash_by_component": dict(
                self.output_pose_sample_hash_by_component
            ),
        }
        key_sets = {
            frozenset(payload[name])
            for name in (
                "source_sample_id_by_component",
                "source_payload_object_id_by_component",
                "output_pose_sample_hash_by_component",
            )
        }
        if len(key_sets) != 1 or not next(iter(key_sets), frozenset()):
            raise ValueError(
                "Human pose adaptation receipt component mappings must have "
                "the same non-empty key set."
            )
        semantic_hash = canonical_content_hash(
            payload,
            domain=HUMAN_POSE_CHANNEL_ADAPTATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT,
        )
        for name, value in payload.items():
            object.__setattr__(
                self,
                name,
                (
                    MappingProxyType(value)
                    if name.endswith("_by_component")
                    else value
                ),
            )
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "receipt_id",
            f"human-pose-channel-adaptation:{semantic_hash}",
        )


@dataclass(frozen=True)
class HumanPoseChannelAdaptationResult:
    samples: tuple[HumanPoseSample, ...]
    receipt: HumanPoseChannelAdaptationReceipt

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples or any(
            not isinstance(item, HumanPoseSample) for item in samples
        ):
            raise TypeError(
                "samples must contain at least one HumanPoseSample."
            )
        if not isinstance(self.receipt, HumanPoseChannelAdaptationReceipt):
            raise TypeError(
                "receipt must be HumanPoseChannelAdaptationReceipt."
            )
        expected = {
            item.component_id: item.semantic_content_hash for item in samples
        }
        if expected != dict(
            self.receipt.output_pose_sample_hash_by_component
        ):
            raise ValueError(
                "Human pose samples and adaptation receipt hashes disagree."
            )
        object.__setattr__(
            self,
            "samples",
            tuple(sorted(samples, key=lambda item: item.component_id)),
        )


class HumanPoseChannelAdapter:
    """Adapt only declared observation channels into immutable pose samples."""

    def __init__(self, config: HumanPoseChannelAdapterConfig) -> None:
        if not isinstance(config, HumanPoseChannelAdapterConfig):
            raise TypeError("config must be HumanPoseChannelAdapterConfig.")
        self.config = config

    def adapt(
        self,
        channels: Mapping[str, ObservationChannel],
        *,
        sample_refs: Mapping[str, ObservationSampleRef],
        adapted_at: float,
        adapted_clock_domain_id: str,
    ) -> HumanPoseChannelAdaptationResult:
        channel_values = dict(channels)
        refs = dict(sample_refs)
        samples: list[HumanPoseSample] = []
        payload_ids: dict[str, str] = {}
        sample_ids: dict[str, str] = {}
        for binding in self.config.component_bindings:
            try:
                channel = channel_values[binding.channel_name]
            except KeyError as exc:
                raise ValueError(
                    f"Missing declared human pose channel "
                    f"{binding.channel_name!r}."
                ) from exc
            try:
                sample_ref = refs[binding.channel_name]
            except KeyError as exc:
                raise ValueError(
                    f"Missing ObservationSampleRef for human pose channel "
                    f"{binding.channel_name!r}."
                ) from exc
            if not isinstance(channel, ObservationChannel):
                raise TypeError(
                    f"Channel {binding.channel_name!r} must be "
                    "ObservationChannel."
                )
            if not isinstance(sample_ref, ObservationSampleRef):
                raise TypeError(
                    f"Sample ref {binding.channel_name!r} must be "
                    "ObservationSampleRef."
                )
            self._validate_channel_sample_pair(
                channel=channel,
                sample_ref=sample_ref,
                binding=binding,
            )
            sample = HumanPoseSample(
                component_id=binding.component_id,
                sample_id=sample_ref.sample_id,
                measured_at=sample_ref.measured_at,
                measured_clock_domain_id=(
                    sample_ref.measured_clock_domain_id
                ),
                source_reference_frame_id=str(channel.frame_name),
                source_controlled_frame_id=(
                    binding.source_controlled_frame_id
                ),
                source_coordinate_binding_id=(
                    self.config.source_coordinate_binding_id
                ),
                source_coordinate_binding_hash=(
                    self.config.source_coordinate_binding_hash
                ),
                pose_source_reference_source_controlled=channel.value,
            )
            samples.append(sample)
            sample_ids[binding.component_id] = sample_ref.sample_id
            payload_ids[binding.component_id] = sample_ref.payload_object_id
        receipt = HumanPoseChannelAdaptationReceipt(
            adapter_contract_hash=self.config.semantic_content_hash,
            adapted_at=adapted_at,
            adapted_clock_domain_id=adapted_clock_domain_id,
            source_coordinate_binding_id=(
                self.config.source_coordinate_binding_id
            ),
            source_coordinate_binding_hash=(
                self.config.source_coordinate_binding_hash
            ),
            source_sample_id_by_component=sample_ids,
            source_payload_object_id_by_component=payload_ids,
            output_pose_sample_hash_by_component={
                item.component_id: item.semantic_content_hash
                for item in samples
            },
        )
        return HumanPoseChannelAdaptationResult(
            samples=tuple(samples),
            receipt=receipt,
        )

    @staticmethod
    def _validate_channel_sample_pair(
        *,
        channel: ObservationChannel,
        sample_ref: ObservationSampleRef,
        binding: HumanPoseChannelBinding,
    ) -> None:
        if channel.channel_name != binding.channel_name:
            raise ValueError(
                "Human pose channel mapping key and channel_name disagree."
            )
        if sample_ref.stream_id != binding.channel_name:
            raise ValueError(
                "Human pose sample stream and channel binding disagree."
            )
        if not channel.valid or not sample_ref.valid or sample_ref.stale:
            raise ValueError(
                f"Human pose channel {binding.channel_name!r} is invalid or stale."
            )
        frame = _required(channel.frame_name, name="channel.frame_name")
        if (
            binding.expected_reference_frame_id is not None
            and frame != binding.expected_reference_frame_id
        ):
            raise ValueError(
                f"Human pose channel {binding.channel_name!r} frame {frame!r} "
                f"does not match {binding.expected_reference_frame_id!r}."
            )
        metadata_frame = sample_ref.content_metadata.get("frame_name")
        if metadata_frame is not None and str(metadata_frame) != frame:
            raise ValueError(
                "Human pose channel frame disagrees with its sample reference."
            )
        payload_object_id = canonical_content_hash(
            {
                "capability": str(channel.capability),
                "side": None if channel.side is None else str(channel.side),
                "value": channel.value,
                "frame_name": frame,
            },
            domain=OBSERVATION_SAMPLE_PAYLOAD_HASH_DOMAIN_V1,
        )
        if payload_object_id != sample_ref.payload_object_id:
            raise ValueError(
                "Human pose channel payload does not match its sample reference."
            )


__all__ = [
    "HUMAN_POSE_CHANNEL_ADAPTATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_POSE_CHANNEL_ADAPTER_HASH_DOMAIN_DEVELOPMENT",
    "HumanPoseChannelAdaptationReceipt",
    "HumanPoseChannelAdaptationResult",
    "HumanPoseChannelAdapter",
    "HumanPoseChannelAdapterConfig",
    "HumanPoseChannelBinding",
]
