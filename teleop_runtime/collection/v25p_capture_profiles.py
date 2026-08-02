"""Profile construction for the default-off v2.5-p capture adapters.

The native contracts are topology-neutral.  This module describes only the
streams that the existing RealTeleop and UMI source adapters can produce.  It
does not own episode lifecycle, persistence, source production, or devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dataset_contracts.action_schema import ActionPayloadSchema
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.causal_snapshot import DEFAULT_CAUSAL_SNAPSHOT_CONTRACT
from dataset_contracts.embodiment import (
    CaptureProjectionComponentV2,
    CaptureProjectionProfileV2,
    EmbodimentCaptureProfileV2,
    EmbodimentComponentCaptureSpec,
)
from dataset_contracts.v22 import EndEffectorChannelSpec, PolicyIOLayout, SIDES
from dataset_contracts.v25p_native import (
    V25pComponentStreamProfileV2,
    V25pComponentStreamSpecV2,
)
from dataset_interface.component_action_projection import (
    ComponentActionProjectionSpec,
)
from dataset_interface.v25p_component_profile import (
    ActionStreamBinding,
    build_v25p_v2_capture_profile,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ComponentActionSchema,
    ControlComponentSpecV2,
)

from collection_runtime.component_capture import ComponentStreamBindings


V25P_REAL_TELEOP_SOURCE = "real_teleop"
V25P_UMI_SOURCE = "umi"


@dataclass(frozen=True)
class V25pCaptureProfileBundle:
    """Runtime composition of persisted profiles and adapter bindings.

    The bundle is not another persisted profile.  The action schema and
    projection spec are embedded by hash/definition in the v2 stream profile.
    """

    embodiment: EmbodimentCaptureProfileV2
    streams: V25pComponentStreamProfileV2
    v23_projection: CaptureProjectionProfileV2
    producer_bindings: ComponentStreamBindings
    component_action_schema: ComponentActionSchema
    component_action_projection: ComponentActionProjectionSpec


def _digest(value: object, *, domain: str) -> str:
    return canonical_content_hash(value, domain=domain)


def _plain_hash(value: str, *, name: str) -> str:
    result = str(value).removeprefix("sha256:").strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a SHA-256 digest.")
    return result


def _payload_schema_hash(
    *,
    representation_family: str,
    dtype: str,
    shape: tuple[int, ...],
    coordinate_names: tuple[str, ...],
    units: tuple[str, ...],
    value_semantics: str,
    semantic_parameters: object = (),
    reference_frame: str | None = None,
    controlled_frame: str | None = None,
    expressed_in_frame: str | None = None,
) -> str:
    return _digest(
        {
            "representation_family": representation_family,
            "dtype": dtype,
            "shape": shape,
            "coordinate_names": coordinate_names,
            "units": units,
            "value_semantics": value_semantics,
            "semantic_parameters": semantic_parameters,
            "reference_frame": reference_frame,
            "controlled_frame": controlled_frame,
            "expressed_in_frame": expressed_in_frame,
        },
        domain="dexteleop.v25p.non_action_payload_schema.semantic.v2",
    )


def _pose_action_schema(
    *,
    source: str,
    side: str,
    component_id: str,
    layout: PolicyIOLayout,
) -> ActionPayloadSchema:
    reference_frame = layout.arm_reference_frame_by_side[side]
    controlled_frame = layout.arm_controlled_frame_by_side[side]
    return ActionPayloadSchema(
        schema_id=f"dexteleop.{source}.action.pose.{side}",
        schema_version="1",
        representation_family="cartesian_pose",
        dtype="float32",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="absolute_target",
        reference_frame=reference_frame,
        controlled_frame=controlled_frame,
        expressed_in_frame=reference_frame,
        semantic_parameters={"rotation_representation": "quaternion_xyzw"},
    )


def _pose_state_stream(
    *,
    source: str,
    side: str,
    component_id: str,
    layout: PolicyIOLayout,
) -> V25pComponentStreamSpecV2:
    names = ("x", "y", "z", "qx", "qy", "qz", "qw")
    units = ("m", "m", "m", "1", "1", "1", "1")
    reference_frame = layout.arm_reference_frame_by_side[side]
    controlled_frame = layout.arm_controlled_frame_by_side[side]
    parameters = {"rotation_representation": "quaternion_xyzw"}
    schema_hash = _payload_schema_hash(
        representation_family="cartesian_pose",
        dtype="float32",
        shape=(7,),
        coordinate_names=names,
        units=units,
        value_semantics="measured_state",
        semantic_parameters=parameters,
        reference_frame=reference_frame,
        controlled_frame=controlled_frame,
        expressed_in_frame=reference_frame,
    )
    return V25pComponentStreamSpecV2(
        stream_id=f"{source}.state.pose.{side}",
        component_id=component_id,
        dataset_role="state",
        value_stage="causal_measured",
        payload_schema_hash=schema_hash,
        representation_family="cartesian_pose",
        dtype="float32",
        shape=(7,),
        coordinate_names=names,
        units=units,
        reference_frame=reference_frame,
        controlled_frame=controlled_frame,
        expressed_in_frame=reference_frame,
        value_semantics="measured_state",
        semantic_parameters=parameters,
    )


def _qpos_stream(
    *,
    source: str,
    side: str,
    component_id: str,
) -> V25pComponentStreamSpecV2:
    names = tuple(f"joint_{index}" for index in range(1, 7))
    schema_hash = _payload_schema_hash(
        representation_family="joint_position",
        dtype="float32",
        shape=(6,),
        coordinate_names=names,
        units=("rad",),
        value_semantics="measured_state",
    )
    return V25pComponentStreamSpecV2(
        stream_id=f"{source}.state.qpos.{side}",
        component_id=component_id,
        dataset_role="state",
        value_stage="causal_measured",
        payload_schema_hash=schema_hash,
        representation_family="joint_position",
        dtype="float32",
        shape=(6,),
        coordinate_names=names,
        units=("rad",),
        value_semantics="measured_state",
    )


def _end_effector_semantics(
    channel: EndEffectorChannelSpec,
) -> tuple[str, str, tuple[str, ...], dict[str, object]]:
    if channel.value_type == "discrete":
        return (
            "discrete_mode",
            "mode",
            ("category_id",),
            {"tokens": tuple(token.to_dict() for token in channel.tokens)},
        )
    return (
        "continuous_native",
        "native_target",
        (str(channel.units or "1"),),
        {},
    )


def _end_effector_action_schema(
    *,
    source: str,
    side: str,
    component_id: str,
    layout: PolicyIOLayout,
) -> ActionPayloadSchema:
    channel = layout.channel("action", side)
    if channel.value_type == "none":
        raise ValueError(f"Cannot create a stream for absent {side} end effector.")
    representation, semantics, units, parameters = _end_effector_semantics(channel)
    return ActionPayloadSchema(
        schema_id=f"dexteleop.{source}.action.end_effector.{side}",
        schema_version="1",
        representation_family=representation,
        dtype="int64" if channel.value_type == "discrete" else "float32",
        shape=(channel.dim,),
        coordinate_names=tuple(channel.names),
        units=units if len(units) == channel.dim else units * channel.dim,
        value_semantics=semantics,
        semantic_parameters=parameters,
    )


def _end_effector_state_stream(
    *,
    source: str,
    side: str,
    component_id: str,
    layout: PolicyIOLayout,
) -> V25pComponentStreamSpecV2:
    channel = layout.channel("state", side)
    if channel.value_type == "none":
        raise ValueError(f"Cannot create a stream for absent {side} end effector.")
    representation, _, units, parameters = _end_effector_semantics(
        channel
    )
    dtype = "int64" if channel.value_type == "discrete" else "float32"
    normalized_units = units if len(units) == channel.dim else units * channel.dim
    value_semantics = "measured_state"
    schema_hash = _payload_schema_hash(
        representation_family=representation,
        dtype=dtype,
        shape=(channel.dim,),
        coordinate_names=tuple(channel.names),
        units=normalized_units,
        value_semantics=value_semantics,
        semantic_parameters=parameters,
    )
    return V25pComponentStreamSpecV2(
        stream_id=f"{source}.state.end_effector.{side}",
        component_id=component_id,
        dataset_role="state",
        value_stage="causal_measured",
        payload_schema_hash=schema_hash,
        representation_family=representation,
        dtype=dtype,
        shape=(channel.dim,),
        coordinate_names=tuple(channel.names),
        units=normalized_units,
        value_semantics=value_semantics,
        semantic_parameters=parameters,
    )


def build_policy_io_v25p_capture_profiles(
    *,
    source: Literal["real_teleop", "umi"],
    layout: PolicyIOLayout,
    pose_frame_contract_hash: str,
    component_sides: tuple[str, ...] | None = None,
) -> V25pCaptureProfileBundle:
    """Build canonical v2 action and native stream profiles for one adapter."""

    if source not in {V25P_REAL_TELEOP_SOURCE, V25P_UMI_SOURCE}:
        raise ValueError(f"Unsupported built-in v2.5-p source {source!r}.")
    sides = SIDES if component_sides is None else tuple(component_sides)
    if (
        not sides
        or len(sides) != len(set(sides))
        or any(side not in SIDES for side in sides)
    ):
        raise ValueError(
            "The built-in v2.3 producer adapter requires a non-empty unique "
            "subset of the legacy left/right sides."
        )
    pose_hash = _plain_hash(pose_frame_contract_hash, name="pose_frame_contract_hash")
    arm_kind = "manipulator" if source == V25P_REAL_TELEOP_SOURCE else "tracked_rigid_body"
    arm_entity = "robot_arm" if source == V25P_REAL_TELEOP_SOURCE else "operator_wrist"

    components: list[EmbodimentComponentCaptureSpec] = []
    catalog_components: list[ControlComponentSpecV2] = []
    action_schemas: dict[str, ActionPayloadSchema] = {}
    stream_declarations: list[
        ActionStreamBinding | V25pComponentStreamSpecV2
    ] = []
    bindings: dict[str, str] = {}
    projection: list[CaptureProjectionComponentV2] = []

    for side in sides:
        arm_component_id = (
            f"arm.{side}" if source == V25P_REAL_TELEOP_SOURCE else f"wrist_tracker.{side}"
        )
        arm_controlled_entity = f"{arm_entity}.{side}"
        components.append(
            EmbodimentComponentCaptureSpec(
                component_id=arm_component_id,
                component_kind=arm_kind,
                controlled_entity_id=arm_controlled_entity,
            )
        )
        arm_schema = _pose_action_schema(
            source=source,
            side=side,
            component_id=arm_component_id,
            layout=layout,
        )
        action_schemas[arm_component_id] = arm_schema
        catalog_components.append(
            ControlComponentSpecV2(
                component_id=arm_component_id,
                controlled_entity_id=arm_controlled_entity,
                command_channel_id=f"{source}.action.pose.{side}",
                selection_group_id="manipulators",
                resource_claims=(arm_controlled_entity,),
                accepted_payload_schema_hashes=(arm_schema.semantic_content_hash,),
            )
        )
        pose_action_stage = (
            "accepted_reference"
            if source == V25P_REAL_TELEOP_SOURCE
            else "next_accepted_observation"
        )
        pose_action_stream_id = f"{source}.action.pose.{side}"
        stream_declarations.append(
            ActionStreamBinding(
                action_component_id=arm_component_id,
                dataset_component_id=arm_component_id,
                stream_id=pose_action_stream_id,
                value_stage=pose_action_stage,
            )
        )
        bindings[f"action.pose.{side}"] = pose_action_stream_id
        projection.append(
            CaptureProjectionComponentV2(
                stream_id=pose_action_stream_id,
                dataset_field=f"action.{side}_arm.pose",
            )
        )
        pose_state = _pose_state_stream(
            source=source,
            side=side,
            component_id=arm_component_id,
            layout=layout,
        )
        stream_declarations.append(pose_state)
        bindings[f"state.pose.{side}"] = pose_state.stream_id
        projection.append(
            CaptureProjectionComponentV2(
                stream_id=pose_state.stream_id,
                dataset_field=f"observation.{side}_arm.pose",
            )
        )
        if source == V25P_REAL_TELEOP_SOURCE:
            qpos = _qpos_stream(
                source=source,
                side=side,
                component_id=arm_component_id,
            )
            stream_declarations.append(qpos)
            bindings[f"state.qpos.{side}"] = qpos.stream_id

        end_effector = layout.end_effector(side)
        if not end_effector.present:
            continue
        end_effector_component_id = f"end_effector.{side}"
        components.append(
            EmbodimentComponentCaptureSpec(
                component_id=end_effector_component_id,
                component_kind="end_effector",
                controlled_entity_id=end_effector_component_id,
            )
        )
        end_effector_schema = _end_effector_action_schema(
            source=source,
            side=side,
            component_id=end_effector_component_id,
            layout=layout,
        )
        action_schemas[end_effector_component_id] = end_effector_schema
        catalog_components.append(
            ControlComponentSpecV2(
                component_id=end_effector_component_id,
                controlled_entity_id=end_effector_component_id,
                command_channel_id=f"{source}.action.end_effector.{side}",
                selection_group_id="end_effectors",
                resource_claims=(end_effector_component_id,),
                accepted_payload_schema_hashes=(
                    end_effector_schema.semantic_content_hash,
                ),
            )
        )
        end_effector_action_stream_id = (
            f"{source}.action.end_effector.{side}"
        )
        stream_declarations.append(
            ActionStreamBinding(
                action_component_id=end_effector_component_id,
                dataset_component_id=end_effector_component_id,
                stream_id=end_effector_action_stream_id,
                value_stage="accepted_reference",
            )
        )
        bindings[f"action.end_effector.{side}"] = end_effector_action_stream_id
        action_token_suffix = (
            "_token"
            if layout.channel("action", side).value_type == "discrete"
            else ""
        )
        projection.append(
            CaptureProjectionComponentV2(
                stream_id=end_effector_action_stream_id,
                dataset_field=(
                    f"action.end_effector.{side}.target{action_token_suffix}"
                ),
            )
        )
        end_effector_state = _end_effector_state_stream(
            source=source,
            side=side,
            component_id=end_effector_component_id,
            layout=layout,
        )
        stream_declarations.append(end_effector_state)
        bindings[f"state.end_effector.{side}"] = end_effector_state.stream_id
        state_token_suffix = (
            "_token"
            if layout.channel("state", side).value_type == "discrete"
            else ""
        )
        projection.append(
            CaptureProjectionComponentV2(
                stream_id=end_effector_state.stream_id,
                dataset_field=(
                    f"observation.end_effector.{side}.state{state_token_suffix}"
                ),
            )
        )

    catalog = ActionComponentCatalogV2(
        catalog_id=f"dexteleop.{source}.{layout.name}.capture_components",
        catalog_version="2",
        components=tuple(catalog_components),
    )
    component_action_schema = ComponentActionSchema(
        schema_id=f"dexteleop.{source}.{layout.name}.component_action",
        schema_version="2",
        component_catalog=catalog,
        payload_schema_by_component=action_schemas,
        semantic_parameters={"capture_source": source},
    )
    native_profiles = build_v25p_v2_capture_profile(
        component_action_schema=component_action_schema,
        embodiment_profile_id=f"{source}.{layout.name}",
        embodiment_profile_version="2",
        topology_id=(
            ("single_manipulator" if len(sides) == 1 else "multi_manipulator")
            if source == V25P_REAL_TELEOP_SOURCE
            else ("single_tracked_wrist" if len(sides) == 1 else "multi_tracked_wrist")
        ),
        embodiment_components=tuple(components),
        stream_profile_id=f"{source}.{layout.name}.native_streams",
        stream_profile_version="2",
        stream_declarations=tuple(stream_declarations),
        projection_id=f"{source}.{layout.name}.component_action_to_v25p",
        projection_version="1",
    )
    embodiment = native_profiles.embodiment_profile
    stream_profile = native_profiles.component_stream_profile
    component_action_projection = (
        native_profiles.component_action_projection_spec
    )
    temporal_payload = tuple(
        (item.stream_id, item.dataset_role, item.value_stage)
        for item in stream_profile.streams
    )
    row_encoder_id = (
        "dexteleop.real_teleop.v23_row_encoder.v1"
        if source == V25P_REAL_TELEOP_SOURCE
        else "dexteleop.umi.v23_row_encoder.v1"
    )
    v23_projection = CaptureProjectionProfileV2(
        projection_id=f"{source}.{layout.name}.to_v23",
        projection_version="2",
        temporal_alignment_contract_id="dexteleop.v25p.stream_local_alignment.v1",
        temporal_alignment_contract_hash=_digest(
            temporal_payload,
            domain="dexteleop.v25p.stream_local_alignment.semantic.v1",
        ),
        row_encoder_contract_id=row_encoder_id,
        row_encoder_contract_hash=_digest(
            {"row_encoder_contract_id": row_encoder_id},
            domain="dexteleop.v25p.row_encoder_binding.semantic.v1",
        ),
        policy_io_layout_hash=_plain_hash(layout.layout_hash, name="policy_io_layout_hash"),
        pose_frame_contract_hash=pose_hash,
        causal_snapshot_contract_hash=_plain_hash(
            DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.contract_hash,
            name="causal_snapshot_contract_hash",
        ),
        components=tuple(projection),
    )
    return V25pCaptureProfileBundle(
        embodiment=embodiment,
        streams=stream_profile,
        v23_projection=v23_projection,
        producer_bindings=ComponentStreamBindings(bindings),
        component_action_schema=component_action_schema,
        component_action_projection=component_action_projection,
    )


__all__ = [
    "V25P_REAL_TELEOP_SOURCE",
    "V25P_UMI_SOURCE",
    "V25pCaptureProfileBundle",
    "build_policy_io_v25p_capture_profiles",
]
