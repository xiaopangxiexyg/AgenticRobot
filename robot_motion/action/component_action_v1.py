"""Explicit RobotAction v1 compatibility bridge for component actions."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from robot_motion.runtime.immutable import deep_thaw

from .arm_spec import ArmActionSpec
from .component_action import (
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
)
from .components import (
    ActionComponentCatalog,
    ActionComponentCatalogV2,
    ControlComponentSpecV2,
)
from .hand_spec import HandActionSpec
from .schema import (
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_MIXED_HAND16_V1,
    BIMANUAL_MIXED_V1,
    BIMANUAL_QPOS_HAND16_V1,
    BIMANUAL_QPOS_V1,
    EndEffectorTarget,
    RobotAction,
    hand_dof_for_action_space,
    split_effector_name,
)


ROBOT_ACTION_V1_COMPONENT_BRIDGE_CONTRACT_ID = (
    "robot_action_v1_component_action_v2_bridge_v1"
)
ROBOT_ACTION_V1_SEMANTIC_METADATA_KEYS = frozenset(
    {"arm_action_specs_by_side", "hand_action_specs_by_side"}
)
_POSE_ACTION_SPACES = frozenset(
    {BIMANUAL_EE_POSE_V1, BIMANUAL_EE_POSE_HAND16_V1}
)
_QPOS_ACTION_SPACES = frozenset(
    {BIMANUAL_QPOS_V1, BIMANUAL_QPOS_HAND16_V1}
)
_MIXED_ACTION_SPACES = frozenset(
    {BIMANUAL_MIXED_V1, BIMANUAL_MIXED_HAND16_V1}
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _arm_spec_for_action(action: RobotAction, side: str) -> ArmActionSpec:
    payload = action.metadata.get("arm_action_specs_by_side")
    if isinstance(payload, Mapping) and isinstance(payload.get(side), Mapping):
        return ArmActionSpec.from_metadata(payload[side])
    return ArmActionSpec(
        side=side,
        reference_frame=action.arm_frame(side),
        controlled_frame=action.arm_controlled_frame(side),
    )


def _hand_spec_for_action(action: RobotAction, side: str) -> HandActionSpec:
    payload = action.metadata.get("hand_action_specs_by_side")
    if isinstance(payload, Mapping) and isinstance(payload.get(side), Mapping):
        return HandActionSpec.from_metadata(payload[side])
    slot_count = hand_dof_for_action_space(action.action_space)
    if slot_count == 16:
        return HandActionSpec.q16(side, active_dof=slot_count)
    return HandActionSpec(side=side, slot_count=slot_count, active_dof=slot_count)


def _canonical_v1_execution_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = sorted(set(metadata) - ROBOT_ACTION_V1_SEMANTIC_METADATA_KEYS)
    if unknown:
        raise ValueError(
            "RobotAction v1 bridge accepts only typed execution metadata; "
            f"unsupported keys: {unknown}."
        )
    canonical: dict[str, Any] = {}
    arm_payload = metadata.get("arm_action_specs_by_side")
    if arm_payload is not None:
        if not isinstance(arm_payload, Mapping):
            raise ValueError("arm_action_specs_by_side must be a mapping.")
        canonical["arm_action_specs_by_side"] = {
            str(side): ArmActionSpec.from_metadata(
                deep_thaw(payload)
            ).to_metadata()
            for side, payload in arm_payload.items()
        }
    hand_payload = metadata.get("hand_action_specs_by_side")
    if hand_payload is not None:
        if not isinstance(hand_payload, Mapping):
            raise ValueError("hand_action_specs_by_side must be a mapping.")
        canonical["hand_action_specs_by_side"] = {
            str(side): HandActionSpec.from_metadata(
                deep_thaw(payload)
            ).to_metadata()
            for side, payload in hand_payload.items()
        }
    return canonical


def _payload_schema_for_v1_component(
    action: RobotAction,
    component_id: str,
) -> tuple[ActionPayloadSchema, np.ndarray]:
    side, kind = split_effector_name(component_id)
    common_parameters: dict[str, Any] = {
        "compatibility_contract_id": (
            ROBOT_ACTION_V1_COMPONENT_BRIDGE_CONTRACT_ID
        ),
        "v1_action_space": action.action_space,
    }
    if kind == "arm" and side in action.arm_pose_by_side:
        spec = _arm_spec_for_action(action, side)
        common_parameters.update(
            {"v1_field": "arm_pose_by_side", "arm_action_spec": spec.to_metadata()}
        )
        return (
            ActionPayloadSchema(
                schema_id=f"robot-action-v1:{action.action_space}:{component_id}:pose",
                schema_version="1",
                representation_family="cartesian_pose",
                dtype="float32",
                shape=(7,),
                coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
                units=("m", "m", "m", "1", "1", "1", "1"),
                value_semantics="absolute_target",
                reference_frame=action.arm_frame(side),
                controlled_frame=action.arm_controlled_frame(side),
                semantic_parameters=common_parameters,
            ),
            np.asarray(action.arm_pose_by_side[side], dtype=np.float32),
        )
    if kind == "arm" and side in action.arm_qpos_by_side:
        values = np.asarray(action.arm_qpos_by_side[side], dtype=np.float32)
        common_parameters["v1_field"] = "arm_qpos_by_side"
        return (
            ActionPayloadSchema(
                schema_id=f"robot-action-v1:{action.action_space}:{component_id}:qpos",
                schema_version="1",
                representation_family="joint_position",
                dtype="float32",
                shape=(int(values.size),),
                coordinate_names=tuple(
                    f"joint_{index}" for index in range(int(values.size))
                ),
                units=tuple("rad" for _ in range(int(values.size))),
                value_semantics="absolute_target",
                semantic_parameters=common_parameters,
            ),
            values,
        )
    if kind == "hand" and side in action.hand_qpos_by_side:
        spec = _hand_spec_for_action(action, side)
        values = spec.active_values(action.hand_qpos_by_side[side])
        common_parameters.update(
            {
                "v1_field": "hand_qpos_by_side",
                "hand_action_spec": spec.to_metadata(),
            }
        )
        return (
            ActionPayloadSchema(
                schema_id=f"robot-action-v1:{action.action_space}:{component_id}:qpos",
                schema_version="1",
                representation_family="joint_position",
                dtype="float32",
                shape=(spec.active_dof,),
                coordinate_names=spec.joint_names,
                units=tuple("1" for _ in range(spec.active_dof)),
                value_semantics="absolute_target",
                semantic_parameters=common_parameters,
            ),
            values,
        )
    if kind == "hand" and side in action.end_effector_targets_by_side:
        target = action.end_effector_targets_by_side[side]
        values = np.asarray(target.values)
        names = target.names or tuple(
            f"value_{index}" for index in range(int(values.size))
        )
        common_parameters.update(
            {
                "v1_field": "end_effector_targets_by_side",
                "end_effector_adapter": target.adapter,
                "end_effector_names": names,
                "end_effector_metadata": dict(target.metadata),
            }
        )
        family = (
            "continuous_native"
            if target.value_type == "continuous"
            else "discrete_mode"
        )
        return (
            ActionPayloadSchema(
                schema_id=(
                    f"robot-action-v1:{action.action_space}:{component_id}:"
                    f"{target.value_type}"
                ),
                schema_version="1",
                representation_family=family,
                dtype="float32" if target.value_type == "continuous" else "int64",
                shape=(int(values.size),),
                coordinate_names=names,
                units=tuple("native" for _ in range(int(values.size))),
                value_semantics=(
                    "native_target"
                    if target.value_type == "continuous"
                    else "mode"
                ),
                semantic_parameters=common_parameters,
            ),
            values,
        )
    raise ValueError(
        f"Active RobotAction v1 component {component_id!r} has no value."
    )


def _payload_representation_matches(
    source: ActionPayloadSchema,
    target: ActionPayloadSchema,
) -> bool:
    """Compare value semantics, not the source carrier's v1 identity.

    A target activation schema names the canonical component representation.
    Different RobotAction-v1 action-space envelopes may still carry the same
    pose/qpos component value, so their schema IDs and v1 provenance metadata
    must not become an accidental compatibility barrier.
    """

    return all(
        getattr(source, name) == getattr(target, name)
        for name in (
            "representation_family",
            "dtype",
            "shape",
            "coordinate_names",
            "units",
            "value_semantics",
            "reference_frame",
            "controlled_frame",
            "expressed_in_frame",
        )
    )


def build_robot_action_v1_runtime_schema(
    *,
    component_catalog: ActionComponentCatalog,
    arm_action_specs_by_side: Mapping[str, ArmActionSpec] | None = None,
    hand_action_specs_by_side: Mapping[str, HandActionSpec] | None = None,
    arm_representation_by_side: Mapping[str, str] | None = None,
) -> ComponentActionSchema:
    """Build one complete activation schema for a declared v1 target profile.

    This is a compatibility-profile builder, not the generic topology API.  It
    never infers a representation from a live vector length.  Mixed v1 action
    spaces must declare each active arm as ``pose`` or ``qpos`` explicitly.
    Native/token end-effectors require a separately authored component schema
    and are intentionally not guessed here.
    """

    if not isinstance(component_catalog, ActionComponentCatalog):
        raise TypeError("component_catalog must be an ActionComponentCatalog.")
    action_space = component_catalog.action_space
    arm_specs = dict(arm_action_specs_by_side or {})
    hand_specs = dict(hand_action_specs_by_side or {})
    arm_representations = {
        str(side): str(value).strip().lower()
        for side, value in dict(arm_representation_by_side or {}).items()
    }
    arm_pose_by_side: dict[str, np.ndarray] = {}
    arm_qpos_by_side: dict[str, np.ndarray] = {}
    hand_qpos_by_side: dict[str, np.ndarray] = {}
    active_effectors: list[str] = []
    metadata: dict[str, Any] = {}
    for component in component_catalog.components:
        side, kind = split_effector_name(component.effector_name)
        active_effectors.append(component.effector_name)
        if kind == "arm":
            if action_space in _POSE_ACTION_SPACES:
                representation = "pose"
            elif action_space in _QPOS_ACTION_SPACES:
                representation = "qpos"
            elif action_space in _MIXED_ACTION_SPACES:
                representation = arm_representations.get(side, "")
                if representation not in {"pose", "qpos"}:
                    raise ValueError(
                        "Mixed RobotAction v1 schemas require explicit "
                        "arm_representation_by_side."
                    )
            else:
                raise ValueError(
                    f"Unsupported RobotAction v1 action space {action_space!r}."
                )
            if representation == "pose":
                arm_pose_by_side[side] = np.asarray(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                )
            else:
                arm_qpos_by_side[side] = np.zeros(6, dtype=np.float32)
        else:
            spec = hand_specs.get(side)
            if spec is None:
                slot_count = hand_dof_for_action_space(action_space)
                spec = (
                    HandActionSpec.q16(side, active_dof=slot_count)
                    if slot_count == 16
                    else HandActionSpec(
                        side=side,
                        slot_count=slot_count,
                        active_dof=slot_count,
                    )
                )
            fill = (
                float("nan")
                if spec.invalid_slot_value.strip().lower() == "nan"
                else 0.0
            )
            values = np.full((spec.slot_count,), fill, dtype=np.float32)
            values[np.asarray(spec.active_mask, dtype=bool)] = 0.0
            hand_qpos_by_side[side] = values
            hand_specs[side] = spec
    if arm_specs:
        metadata["arm_action_specs_by_side"] = {
            side: spec.to_metadata() for side, spec in arm_specs.items()
        }
    if hand_specs:
        metadata["hand_action_specs_by_side"] = {
            side: spec.to_metadata() for side, spec in hand_specs.items()
        }
    frame_by_effector = {
        f"{side}_arm": spec.reference_frame
        for side, spec in arm_specs.items()
        if side in arm_pose_by_side
    }
    controlled_frame_by_effector = {
        f"{side}_arm": spec.controlled_frame
        for side, spec in arm_specs.items()
        if side in arm_pose_by_side
    }
    template = RobotAction(
        action_space=action_space,
        arm_pose_by_side=arm_pose_by_side,
        arm_qpos_by_side=arm_qpos_by_side,
        hand_qpos_by_side=hand_qpos_by_side,
        active_effectors=tuple(active_effectors),
        frame_by_effector=frame_by_effector,
        controlled_frame_by_effector=controlled_frame_by_effector,
        source="runtime_schema_template",
        metadata=metadata,
    )
    local_schema = project_robot_action_v1_to_component_action(
        template
    ).action_schema
    converted_components = tuple(
        ControlComponentSpecV2(
            component_id=component.component_id,
            controlled_entity_id=split_effector_name(
                component.effector_name
            )[0],
            command_channel_id=component.effector_name,
            selection_group_id=component.selection_group_id,
            resource_claims=(component.component_id,),
            accepted_payload_schema_hashes=(
                local_schema.payload_schema_by_component[
                    component.effector_name
                ].semantic_content_hash,
            ),
            atomic_group_id=(
                component.selection_group_id if component.coupled else None
            ),
        )
        for component in component_catalog.components
    )
    return ComponentActionSchema(
        schema_id=(
            f"robot-action-v1:{component_catalog.catalog_id}:runtime-action"
        ),
        schema_version="1",
        component_catalog=ActionComponentCatalogV2(
            catalog_id=f"{component_catalog.catalog_id}:component-v2",
            catalog_version="1",
            components=converted_components,
        ),
        payload_schema_by_component={
            component.component_id: local_schema.payload_schema_by_component[
                component.effector_name
            ]
            for component in component_catalog.components
        },
        semantic_parameters={
            **dict(local_schema.semantic_parameters),
            # Preserve the exact v1 catalog identity only for compatibility
            # manifests and trace observers.  Production admission binds the
            # complete ComponentActionSchema hash above this legacy view.
            "legacy_component_catalog_hash": (
                component_catalog.semantic_content_hash
            ),
        },
    )


def project_robot_action_v1_to_component_action(
    action: RobotAction,
    *,
    action_schema: ComponentActionSchema | None = None,
    component_id_by_effector: Mapping[str, str] | None = None,
) -> ComponentAction:
    """Project one v1 action into a component-keyed action.

    Production callers must pass the complete schema bound by the current
    runtime activation.  ``action.active_effectors`` may cover only a subset
    of that schema; this is how single-arm or end-effector-only sources share
    one proposal carrier without inventing source-local topology hashes.

    Omitting ``action_schema`` retains a self-contained compatibility
    projection for offline conversion and tests.  Such a locally derived
    schema is not a production activation binding.

    Source and timestamp intentionally remain outside the component carrier;
    callers keep them in proposal/emission lineage and may pass them back when
    materializing a legacy view.
    """

    if not isinstance(action, RobotAction):
        raise TypeError("action must be a RobotAction.")
    requested_mapping = {
        _required(effector, name="effector_name"): _required(
            component_id,
            name="component_id",
        )
        for effector, component_id in dict(
            component_id_by_effector or {}
        ).items()
    }
    payload_schemas: dict[str, ActionPayloadSchema] = {}
    raw_values: dict[str, np.ndarray] = {}
    for effector_name in action.active_effectors:
        payload_schema, values = _payload_schema_for_v1_component(
            action,
            effector_name,
        )
        component_id = requested_mapping.get(effector_name, effector_name)
        if component_id in raw_values:
            raise ValueError(
                "RobotAction v1 effector mapping produced duplicate component "
                f"{component_id!r}."
            )
        payload_schemas[component_id] = payload_schema
        raw_values[component_id] = values
    if action_schema is not None:
        if not isinstance(action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")
        unknown = sorted(
            set(raw_values) - set(action_schema.component_ids)
        )
        if unknown:
            raise ValueError(
                "RobotAction v1 contains components outside the runtime "
                f"action schema: {unknown}."
            )
        for component_id, payload_schema in payload_schemas.items():
            expected = action_schema.payload_schema_by_component[component_id]
            if not _payload_representation_matches(payload_schema, expected):
                raise ValueError(
                    f"RobotAction v1 component {component_id!r} does not match "
                    "the runtime payload schema."
                )
        return ComponentAction(
            action_schema=action_schema,
            values_by_component=raw_values,
        )

    arm_components = tuple(
        component_id
        for component_id in raw_values
        if split_effector_name(component_id)[1] == "arm"
    )
    component_specs = []
    for component_id in raw_values:
        side, kind = split_effector_name(component_id)
        payload_hash = payload_schemas[component_id].semantic_content_hash
        component_specs.append(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=side,
                command_channel_id=component_id,
                selection_group_id="arms" if kind == "arm" else component_id,
                resource_claims=(component_id,),
                accepted_payload_schema_hashes=(payload_hash,),
                atomic_group_id=(
                    "arms"
                    if kind == "arm" and len(arm_components) > 1
                    else None
                ),
            )
        )
    catalog = ActionComponentCatalogV2(
        catalog_id=f"robot-action-v1:{action.action_space}:catalog",
        catalog_version="1",
        components=tuple(component_specs),
    )
    action_schema = ComponentActionSchema(
        schema_id=f"robot-action-v1:{action.action_space}:component-action",
        schema_version="1",
        component_catalog=catalog,
        payload_schema_by_component=payload_schemas,
        semantic_parameters={
            "compatibility_contract_id": (
                ROBOT_ACTION_V1_COMPONENT_BRIDGE_CONTRACT_ID
            ),
            "v1_action_space": action.action_space,
            "v1_execution_metadata": _canonical_v1_execution_metadata(
                action.metadata
            ),
            "v1_frame_by_effector": dict(action.frame_by_effector),
            "v1_controlled_frame_by_effector": dict(
                action.controlled_frame_by_effector
            ),
        },
    )
    return ComponentAction(
        action_schema=action_schema,
        values_by_component=raw_values,
    )


def materialize_robot_action_v1(
    action: ComponentAction,
    *,
    timestamp: float | None = None,
    source: str = "",
) -> RobotAction:
    """Materialize a v1 compatibility view when every component is lossless."""

    if not isinstance(action, ComponentAction):
        raise TypeError("action must be a ComponentAction.")
    parameters = action.action_schema.semantic_parameters
    if parameters.get("compatibility_contract_id") != (
        ROBOT_ACTION_V1_COMPONENT_BRIDGE_CONTRACT_ID
    ):
        raise ValueError(
            "Component action was not produced by the RobotAction v1 bridge."
        )
    action_space = _required(
        parameters.get("v1_action_space"),
        name="v1_action_space",
    )
    metadata = _canonical_v1_execution_metadata(
        parameters.get("v1_execution_metadata", {})
    )
    arm_pose_by_side: dict[str, np.ndarray] = {}
    arm_qpos_by_side: dict[str, np.ndarray] = {}
    hand_qpos_by_side: dict[str, np.ndarray] = {}
    native_targets_by_side: dict[str, EndEffectorTarget] = {}
    for component_id, value in action.values_by_component.items():
        component_spec = (
            action.action_schema.component_catalog.component_by_id[
                component_id
            ]
        )
        effector_name = component_spec.command_channel_id
        try:
            side, kind = split_effector_name(effector_name)
        except ValueError as exc:
            raise ValueError(
                f"Component {component_id!r} has no RobotAction v1 command "
                "channel."
            ) from exc
        schema_parameters = value.payload_schema.semantic_parameters
        field_name = schema_parameters.get("v1_field")
        values = np.asarray(value.values).copy()
        if field_name == "arm_pose_by_side" and kind == "arm":
            arm_pose_by_side[side] = values.astype(np.float32)
        elif field_name == "arm_qpos_by_side" and kind == "arm":
            arm_qpos_by_side[side] = values.astype(np.float32)
        elif field_name == "hand_qpos_by_side" and kind == "hand":
            spec_payload = schema_parameters.get("hand_action_spec")
            if not isinstance(spec_payload, Mapping):
                raise ValueError("V1 hand bridge is missing HandActionSpec.")
            spec = HandActionSpec.from_metadata(deep_thaw(spec_payload))
            if values.shape != (spec.active_dof,):
                raise ValueError("V1 hand active value dimension changed.")
            padding = spec.invalid_slot_value.strip().lower()
            fill = float("nan") if padding == "nan" else 0.0
            full = np.full((spec.slot_count,), fill, dtype=np.float32)
            full[np.asarray(spec.active_mask, dtype=bool)] = values
            hand_qpos_by_side[side] = full
        elif field_name == "end_effector_targets_by_side" and kind == "hand":
            adapter = _required(
                schema_parameters.get("end_effector_adapter"),
                name="end_effector_adapter",
            )
            names = tuple(
                str(item)
                for item in deep_thaw(
                    schema_parameters.get("end_effector_names", ())
                )
            )
            target_metadata = dict(
                deep_thaw(
                    schema_parameters.get("end_effector_metadata", {})
                )
            )
            native_targets_by_side[side] = EndEffectorTarget(
                value_type=(
                    "continuous"
                    if value.payload_schema.representation_family
                    == "continuous_native"
                    else "discrete"
                ),
                values=values,
                adapter=adapter,
                names=names,
                metadata=target_metadata,
            )
        else:
            raise ValueError(
                f"Component {component_id!r} cannot be materialized as "
                "RobotAction v1."
            )
    return RobotAction(
        action_space=action_space,
        timestamp=timestamp,
        arm_pose_by_side=arm_pose_by_side,
        arm_qpos_by_side=arm_qpos_by_side,
        hand_qpos_by_side=hand_qpos_by_side,
        end_effector_targets_by_side=native_targets_by_side,
        active_effectors=tuple(
            action.action_schema.component_catalog.component_by_id[
                component_id
            ].command_channel_id
            for component_id in action.component_coverage
        ),
        frame_by_effector=dict(
            deep_thaw(parameters.get("v1_frame_by_effector", {}))
        ),
        controlled_frame_by_effector=dict(
            deep_thaw(
                parameters.get("v1_controlled_frame_by_effector", {})
            )
        ),
        source=str(source),
        metadata=metadata,
    )



__all__ = [
    "ROBOT_ACTION_V1_COMPONENT_BRIDGE_CONTRACT_ID",
    "ROBOT_ACTION_V1_SEMANTIC_METADATA_KEYS",
    "build_robot_action_v1_runtime_schema",
    "materialize_robot_action_v1",
    "project_robot_action_v1_to_component_action",
]
