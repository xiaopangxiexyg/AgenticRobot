"""Default-off native v2.5-p producer for real-teleop collector samples."""

from __future__ import annotations

from dataset_contracts.v22 import PolicyIOLayout
from dataset_contracts.v25p_native import (
    V25pComponentStreamProfileLike,
    V25pComponentStreamProfileV2,
)
from dataset_interface.component_action_projection import (
    ComponentActionProjectionSpec,
    project_component_action_to_v25p,
)
from robot_motion.action import ComponentAction, ComponentActionSchema

from .sample_assembler import CollectorSample
from .schema_v22 import extract_collector_policy_io_component_values
from .v25p_component_action import attach_projected_component_action_timing
from .v25p_native_store import V25pComponentInput
from .v25p_producer import (
    V25pProducedSample,
    V25pProducerBindings,
    validate_produced_values,
    validate_producer_bindings,
)


class RealTeleopV25pProducer:
    """Translate one accepted ``CollectorSample`` without owning its lifecycle."""

    def __init__(
        self,
        *,
        layout: PolicyIOLayout,
        stream_profile: V25pComponentStreamProfileLike,
        bindings: V25pProducerBindings,
        component_action_schema: ComponentActionSchema | None = None,
        component_action_projection: ComponentActionProjectionSpec | None = None,
        clock_domain_id: str = "local_wall_time",
        strict: bool = True,
    ) -> None:
        self.layout = layout
        self.stream_profile = stream_profile
        self.bindings = bindings
        self.component_action_schema = component_action_schema
        self.component_action_projection = component_action_projection
        self.clock_domain_id = str(clock_domain_id).strip()
        self.strict = bool(strict)
        if not self.clock_domain_id:
            raise ValueError("clock_domain_id must be non-empty.")
        validate_producer_bindings(
            profile=stream_profile,
            bindings=bindings,
            expected_stage_by_value_key={
                key: (
                    "accepted_reference"
                    if key.startswith("action.")
                    else "causal_measured"
                )
                for key in bindings.stream_id_by_value_key
            },
        )
        v2 = isinstance(stream_profile, V25pComponentStreamProfileV2)
        if v2 != (
            component_action_schema is not None
            and component_action_projection is not None
        ):
            raise ValueError(
                "Canonical v2 stream profiles require exactly one ComponentAction "
                "schema and projection spec; frozen v1 profiles require neither."
            )
        if v2 and (
            component_action_schema.semantic_content_hash
            != stream_profile.component_action_space_schema.semantic_content_hash
            or component_action_projection.component_action_space_schema_hash
            != component_action_schema.semantic_content_hash
        ):
            raise ValueError("RealTeleop ComponentAction/profile binding mismatch.")

    def produce(self, sample: CollectorSample) -> V25pProducedSample:
        timestamp = float(sample.timestamp)
        source = extract_collector_policy_io_component_values(
            sample,
            layout=self.layout,
            strict=self.strict,
        )
        inputs: dict[str, V25pComponentInput] = {}
        if isinstance(self.stream_profile, V25pComponentStreamProfileV2):
            assert self.component_action_schema is not None
            assert self.component_action_projection is not None
            source_key_by_stream = {
                stream_id: value_key
                for value_key, stream_id in self.bindings.stream_id_by_value_key.items()
            }
            action = ComponentAction(
                action_schema=self.component_action_schema,
                values_by_component={
                    component_id: source[source_key_by_stream[stream_id]]
                    for component_id, stream_id in (
                        self.component_action_projection.stream_id_by_action_component.items()
                    )
                },
            )
            projected = project_component_action_to_v25p(
                action=action,
                profile=self.stream_profile,
                spec=self.component_action_projection,
            )
            inputs.update(
                attach_projected_component_action_timing(
                    projected=projected,
                    profile=self.stream_profile,
                    effective_at=timestamp,
                    available_at=timestamp,
                    effective_clock_domain_id=self.clock_domain_id,
                    available_clock_domain_id=self.clock_domain_id,
                )
            )
        for value_key, stream_id in self.bindings.stream_id_by_value_key.items():
            if value_key not in source:
                raise ValueError(
                    f"Real-teleop sample does not provide bound native value {value_key!r}."
                )
            if stream_id in inputs:
                continue
            inputs[stream_id] = V25pComponentInput(
                values=source[value_key],
                effective_at=timestamp,
                available_at=timestamp,
                effective_clock_domain_id=self.clock_domain_id,
                available_clock_domain_id=self.clock_domain_id,
                source_record_id=(
                    f"collector_sample:{int(sample.episode_index)}:"
                    f"{int(sample.frame_index)}:{value_key}"
                ),
            )
        produced = V25pProducedSample(
            sample_index=int(sample.frame_index),
            cut_at=timestamp,
            cut_clock_domain_id=self.clock_domain_id,
            values_by_stream=inputs,
        )
        validate_produced_values(profile=self.stream_profile, produced=produced)
        return produced


__all__ = ["RealTeleopV25pProducer"]
