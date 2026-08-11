"""MuJoCo target bindings for canonical component-keyed actions.

The generic action carrier and topology contracts live in ``robot_motion``.
This module only binds those contracts to concrete MuJoCo scene entities.  It
does not own source adaptation, composition, authority, safety, IK, reference
generation, or dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    ActionComponentCatalog,
    ActionComponentCatalogV2,
    ActionComponentSpec,
    ActionPayloadSchema,
    ArmActionSpec,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
    HandActionSpec,
    RobotAction,
    arm_action_specs_metadata,
    hand_action_specs_metadata,
    hand_action_specs_from_metadata,
)
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.rm65.constants import TOOL_TCP_FRAME


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _unique_mapping(
    values: Mapping[str, str],
    *,
    key_name: str,
    value_name: str,
    allow_empty: bool = False,
) -> Mapping[str, str]:
    normalized = {
        _required(key, name=key_name): _required(value, name=value_name)
        for key, value in dict(values).items()
    }
    if not normalized and not allow_empty:
        raise ValueError(f"{key_name} mapping must not be empty.")
    if len(normalized.values()) != len(set(normalized.values())):
        raise ValueError(f"{value_name} values must be unique.")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class MujocoComponentTargetProfile:
    """Bind canonical components to concrete MuJoCo target entities.

    The action schema remains backend-neutral.  The two maps are target-private
    routing data used only after authoritative composition has selected one
    ``ComponentAction``.
    """

    profile_id: str
    action_schema: ComponentActionSchema
    arm_side_by_component: Mapping[str, str]
    gripper_name_by_component: Mapping[str, str]

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        if not isinstance(self.action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")
        arms = _unique_mapping(
            self.arm_side_by_component,
            key_name="arm component id",
            value_name="MuJoCo arm side",
        )
        grippers = _unique_mapping(
            self.gripper_name_by_component,
            key_name="gripper component id",
            value_name="MuJoCo gripper name",
            allow_empty=True,
        )
        if set(arms) & set(grippers):
            raise ValueError("One component cannot route to both an arm and a gripper.")
        if set(arms) | set(grippers) != set(self.action_schema.component_ids):
            raise ValueError(
                "MuJoCo component routes must exactly cover the action schema."
            )
        for component_id in arms:
            payload = self.action_schema.payload_schema_by_component[component_id]
            if payload.representation_family != "cartesian_pose":
                raise ValueError(
                    f"Arm component {component_id!r} must use cartesian_pose."
                )
        for component_id in grippers:
            payload = self.action_schema.payload_schema_by_component[component_id]
            if (
                payload.representation_family != "joint_position"
                or payload.value_semantics != "absolute_target"
            ):
                raise ValueError(
                    f"Gripper component {component_id!r} must use absolute "
                    "joint_position."
                )
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "arm_side_by_component", arms)
        object.__setattr__(self, "gripper_name_by_component", grippers)


class MujocoRobotActionV1ComponentIngressAdapter:
    """Compatibility ingress from existing source envelopes to native values.

    The adapter is intentionally source-agnostic and non-authoritative.  It
    performs only representation conversion required by the selected MuJoCo
    target profile.  It does not clamp, run safety, own freshness, or dispatch.
    New sources may bypass it by emitting the exact ``ComponentAction`` schema.
    """

    def __init__(
        self,
        scene: Any,
        profile: MujocoComponentTargetProfile,
        *,
        component_id_by_effector: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(profile, MujocoComponentTargetProfile):
            raise TypeError("profile must be a MujocoComponentTargetProfile.")
        self.scene = scene
        self.profile = profile
        self.component_id_by_effector = MappingProxyType(
            {
                _required(effector, name="effector name"): _required(
                    component_id,
                    name="component id",
                )
                for effector, component_id in dict(
                    component_id_by_effector or {}
                ).items()
            }
        )
        gripper_derivation = {
            component_id: {
                "gripper_name": gripper_name,
                "joint_names": tuple(
                    scene.gripper_specs[gripper_name].joint_names
                ),
                "open_qpos": tuple(
                    float(value)
                    for value in scene.gripper_specs[gripper_name].open_qpos
                ),
                "closed_qpos": tuple(
                    float(value)
                    for value in scene.gripper_specs[gripper_name].closed_qpos
                ),
            }
            for component_id, gripper_name in (
                profile.gripper_name_by_component.items()
            )
        }
        self.semantic_content_hash = canonical_content_hash(
            {
                "contract_id": (
                    "mujoco.robot_action_v1_to_component_action_ingress.v1"
                ),
                "output_action_schema_hash": (
                    profile.action_schema.semantic_content_hash
                ),
                "component_id_by_effector": self.component_id_by_effector,
                "gripper_close_derivation": gripper_derivation,
                "clamping": "none",
                "safety": "none",
            },
            domain="dexteleop.mujoco.component_ingress_adapter.v1",
        )

    def project(self, action: RobotAction) -> ComponentAction:
        """Project using the immutable source-effector routing contract."""

        if not self.component_id_by_effector:
            raise ValueError(
                "The ingress adapter has no bound component_id_by_effector."
            )
        return self.adapt(
            action,
            component_id_by_effector=self.component_id_by_effector,
        )

    def adapt(
        self,
        action: RobotAction,
        *,
        component_id_by_effector: Mapping[str, str] | None = None,
    ) -> ComponentAction:
        if not isinstance(action, RobotAction):
            raise TypeError("action must be a RobotAction.")
        mapping = {
            _required(effector, name="effector name"): _required(
                component_id,
                name="component id",
            )
            for effector, component_id in dict(
                (
                    self.component_id_by_effector
                    if component_id_by_effector is None
                    else component_id_by_effector
                )
            ).items()
        }
        if not mapping:
            raise ValueError("component_id_by_effector must not be empty.")
        values_by_component: dict[str, np.ndarray] = {}
        hand_specs = (
            hand_action_specs_from_metadata(action.metadata)
            if action.hand_qpos_by_side
            else {}
        )
        for effector in action.active_effectors:
            try:
                component_id = mapping[effector]
            except KeyError as exc:
                raise ValueError(
                    f"No component route for active effector {effector!r}."
                ) from exc
            if component_id in values_by_component:
                raise ValueError(
                    f"Multiple effectors route to component {component_id!r}."
                )
            if component_id in self.profile.arm_side_by_component:
                side = self.profile.arm_side_by_component[component_id]
                if effector != f"{side}_arm" or side not in action.arm_pose_by_side:
                    raise ValueError(
                        f"Component {component_id!r} requires {side!r} arm pose."
                    )
                payload = self.profile.action_schema.payload_schema_by_component[
                    component_id
                ]
                if (
                    action.arm_frame(side) != payload.reference_frame
                    or action.arm_controlled_frame(side)
                    != payload.controlled_frame
                ):
                    raise ValueError(
                        f"Arm pose for {component_id!r} does not match the "
                        "target profile frame contract."
                    )
                values_by_component[component_id] = np.asarray(
                    action.arm_pose_by_side[side],
                    dtype=np.float32,
                )
                continue

            gripper_name = self.profile.gripper_name_by_component[component_id]
            gripper_spec = self.scene.gripper_specs[gripper_name]
            side = str(gripper_spec.side or "").strip()
            if not side or effector != f"{side}_hand":
                raise ValueError(
                    f"Component {component_id!r} requires the declared "
                    f"{side!r} hand source."
                )
            if side not in action.hand_qpos_by_side:
                raise ValueError(
                    f"Active gripper effector {effector!r} has no qpos value."
                )
            source_spec = hand_specs.get(side)
            if source_spec is None:
                raise ValueError(
                    f"Gripper source {effector!r} lacks an exact HandActionSpec."
                )
            source_values = np.asarray(
                source_spec.active_values(action.hand_qpos_by_side[side]),
                dtype=np.float64,
            ).reshape(-1)
            target_joint_names = tuple(str(name) for name in gripper_spec.joint_names)
            if (
                source_values.shape == (len(target_joint_names),)
                and tuple(source_spec.joint_names) == target_joint_names
            ):
                target_qpos = source_values
            elif (
                source_values.shape == (1,)
                and tuple(source_spec.joint_names) == ("gripper_close",)
            ):
                # Preserve the raw source scalar.  The target's open/closed
                # geometry defines a linear representation conversion only;
                # downstream target safety may decide whether the result is
                # executable.
                opened = np.asarray(gripper_spec.open_qpos, dtype=np.float64)
                closed = np.asarray(
                    gripper_spec.closed_qpos,
                    dtype=np.float64,
                )
                target_qpos = opened + source_values[0] * (closed - opened)
            else:
                raise ValueError(
                    f"Gripper source {effector!r} does not provide either the "
                    "declared physical joint qpos or one gripper_close scalar."
                )
            values_by_component[component_id] = target_qpos.astype(
                np.float32,
                copy=False,
            )
        return ComponentAction(
            action_schema=self.profile.action_schema,
            values_by_component=values_by_component,
        )


class MujocoComponentActionRobotActionV1Materializer:
    """Target-private compatibility view for the current coupled RM65 executor.

    Authoritative values remain in the native component schema.  The legacy
    view is created only after target capability admission.  Missing arm poses
    may be supplied from a measured-state callback when the concrete downstream
    solver is coupled; those support-only poses never become source,
    composition, or authority components.
    """

    def __init__(
        self,
        scene: Any,
        profile: MujocoComponentTargetProfile,
        *,
        hand_action_spec_by_component: Mapping[str, HandActionSpec],
        support_arm_pose_provider: Callable[[], Mapping[str, Any]],
        support_arm_sides: tuple[str, ...] = (),
        support_pose_contract_id: str,
    ) -> None:
        if not isinstance(profile, MujocoComponentTargetProfile):
            raise TypeError("profile must be a MujocoComponentTargetProfile.")
        if not callable(support_arm_pose_provider):
            raise TypeError("support_arm_pose_provider must be callable.")
        support_sides = tuple(
            _required(side, name="support arm side") for side in support_arm_sides
        )
        if len(support_sides) != len(set(support_sides)):
            raise ValueError("support_arm_sides must be unique.")
        authoritative_arm_sides = set(profile.arm_side_by_component.values())
        if set(support_sides) & authoritative_arm_sides:
            raise ValueError(
                "Support-only arm sides must not overlap authoritative arms."
            )
        hand_specs = dict(hand_action_spec_by_component)
        if set(hand_specs) != set(profile.gripper_name_by_component):
            raise ValueError(
                "hand_action_spec_by_component must cover every gripper "
                "component exactly."
            )
        for component_id, spec in hand_specs.items():
            if not isinstance(spec, HandActionSpec):
                raise TypeError("hand_action_spec_by_component values must be typed.")
            gripper_name = profile.gripper_name_by_component[component_id]
            gripper_side = str(scene.gripper_specs[gripper_name].side or "").strip()
            if spec.side != gripper_side or spec.active_dof != 1:
                raise ValueError(
                    "Current RobotAction-v1 compatibility view requires one "
                    "active DM-Tac close value on the routed gripper side."
                )
        all_arm_sides = authoritative_arm_sides | set(support_sides)
        if all_arm_sides != {"left", "right"}:
            raise ValueError(
                "The current coupled RM65 compatibility executor requires "
                "left and right arm poses after measured support is added."
            )
        self.scene = scene
        self.profile = profile
        self.hand_action_spec_by_component = MappingProxyType(hand_specs)
        self.support_arm_pose_provider = support_arm_pose_provider
        self.support_arm_sides = support_sides
        self.support_pose_contract_id = _required(
            support_pose_contract_id,
            name="support_pose_contract_id",
        )
        self.output_effector_by_component = MappingProxyType(
            {
                **{
                    component_id: f"{side}_arm"
                    for component_id, side in profile.arm_side_by_component.items()
                },
                **{
                    component_id: (
                        f"{self.hand_action_spec_by_component[component_id].side}_hand"
                    )
                    for component_id in profile.gripper_name_by_component
                },
            }
        )
        self.support_only_output_effectors = tuple(
            f"{side}_arm" for side in support_sides
        )
        gripper_geometry = {
            component_id: {
                "gripper_name": gripper_name,
                "joint_names": tuple(
                    scene.gripper_specs[gripper_name].joint_names
                ),
                "open_qpos": tuple(
                    float(value)
                    for value in scene.gripper_specs[gripper_name].open_qpos
                ),
                "closed_qpos": tuple(
                    float(value)
                    for value in scene.gripper_specs[gripper_name].closed_qpos
                ),
                "compatibility_hand_spec": (
                    self.hand_action_spec_by_component[component_id].to_metadata()
                ),
            }
            for component_id, gripper_name in (
                profile.gripper_name_by_component.items()
            )
        }
        self.contract_id = (
            "mujoco.component_action_to_coupled_robot_action_v1.development"
        )
        self.semantic_content_hash = canonical_content_hash(
            {
                "contract_id": self.contract_id,
                "input_action_schema_hash": (
                    profile.action_schema.semantic_content_hash
                ),
                "output": "target_private_robot_action_v1",
                "output_effector_by_component": (
                    self.output_effector_by_component
                ),
                "support_only_output_effectors": (
                    self.support_only_output_effectors
                ),
                "support_pose_contract_id": self.support_pose_contract_id,
                "gripper_physical_qpos_to_close": gripper_geometry,
                "clamping": "none",
                "authority": "none",
                "safety": "none",
            },
            domain="dexteleop.mujoco.component_execution_adapter.development",
        )

    @staticmethod
    def _physical_qpos_to_close(
        values: Any,
        *,
        open_qpos: Any,
        closed_qpos: Any,
    ) -> float:
        qpos = np.asarray(values, dtype=np.float64).reshape(-1)
        opened = np.asarray(open_qpos, dtype=np.float64).reshape(-1)
        closed = np.asarray(closed_qpos, dtype=np.float64).reshape(-1)
        if qpos.shape != opened.shape or opened.shape != closed.shape:
            raise ValueError("Physical gripper qpos geometry shape mismatch.")
        delta = closed - opened
        moving = np.abs(delta) > 1e-12
        if not np.any(moving):
            raise ValueError("Gripper open/closed geometry has no moving joint.")
        if not np.allclose(
            qpos[~moving],
            opened[~moving],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "Physical gripper qpos changes a non-moving compatibility joint."
            )
        close_values = (qpos[moving] - opened[moving]) / delta[moving]
        close = float(np.mean(close_values))
        if not np.allclose(close_values, close, rtol=0.0, atol=1e-6):
            raise ValueError(
                "Physical gripper qpos is outside the target's scalar-close "
                "compatibility manifold."
            )
        # Deliberately do not clamp.  Canonical target safety owns executable
        # limits after this representation conversion.
        return close

    def __call__(
        self,
        action: ComponentAction,
        *,
        timestamp: float,
        source: str,
    ) -> RobotAction:
        if not isinstance(action, ComponentAction):
            raise TypeError("action must be a ComponentAction.")
        if (
            action.action_schema.semantic_content_hash
            != self.profile.action_schema.semantic_content_hash
            or action.component_coverage
            != self.profile.action_schema.component_ids
        ):
            raise ValueError(
                "Execution materializer requires the exact complete target schema."
            )
        arm_pose_by_side = {
            side: np.asarray(
                action.value(component_id).values,
                dtype=np.float32,
            )
            for component_id, side in self.profile.arm_side_by_component.items()
        }
        support: dict[str, np.ndarray] = {}
        for side, values in dict(self.support_arm_pose_provider()).items():
            support_side = str(side)
            measured = np.asarray(values, dtype=np.float32)
            if measured.shape == (4, 4):
                measured = pose_matrix_to_xyzquat(measured)
            measured = np.asarray(measured, dtype=np.float32).reshape(-1)
            if measured.shape != (7,) or not np.isfinite(measured).all():
                raise ValueError(
                    "Measured support-arm poses must be finite xyz+quaternion "
                    "vectors or 4x4 pose matrices."
                )
            support[support_side] = measured
        if set(support) != set(self.support_arm_sides):
            raise ValueError(
                "Measured support-arm provider returned another support scope."
            )
        arm_pose_by_side.update(support)
        arm_specs = {
            side: ArmActionSpec.tool_tcp(
                side,
                reference_frame=f"{side}_arm_base",
            )
            for side in arm_pose_by_side
        }
        hand_qpos_by_side: dict[str, np.ndarray] = {}
        hand_specs_by_side: dict[str, HandActionSpec] = {}
        for component_id, gripper_name in (
            self.profile.gripper_name_by_component.items()
        ):
            spec = self.hand_action_spec_by_component[component_id]
            gripper_spec = self.scene.gripper_specs[gripper_name]
            close = self._physical_qpos_to_close(
                action.value(component_id).values,
                open_qpos=gripper_spec.open_qpos,
                closed_qpos=gripper_spec.closed_qpos,
            )
            vector = np.zeros(spec.slot_count, dtype=np.float32)
            vector[int(spec.slot_map["gripper_close"])] = close
            hand_qpos_by_side[spec.side] = vector
            hand_specs_by_side[spec.side] = spec
        active_effectors = tuple(
            (
                *(f"{side}_arm" for side in ("left", "right")),
                *(f"{side}_hand" for side in sorted(hand_qpos_by_side)),
            )
        )
        return RobotAction(
            action_space=BIMANUAL_EE_POSE_V1,
            arm_pose_by_side=arm_pose_by_side,
            hand_qpos_by_side=hand_qpos_by_side,
            active_effectors=active_effectors,
            timestamp=float(timestamp),
            source=str(source),
            metadata={
                "arm_action_specs_by_side": arm_action_specs_metadata(arm_specs),
                "hand_action_specs_by_side": hand_action_specs_metadata(
                    hand_specs_by_side
                ),
            },
        )


def build_mujoco_component_execution_binding(
    profile: MujocoComponentTargetProfile,
    materializer: MujocoComponentActionRobotActionV1Materializer,
) -> Any:
    """Bind a native profile to the existing lower MuJoCo execution owner."""

    if not isinstance(profile, MujocoComponentTargetProfile):
        raise TypeError("profile must be a MujocoComponentTargetProfile.")
    if not isinstance(
        materializer,
        MujocoComponentActionRobotActionV1Materializer,
    ):
        raise TypeError(
            "materializer must be MujocoComponentActionRobotActionV1Materializer."
        )
    if (
        materializer.profile.action_schema.semantic_content_hash
        != profile.action_schema.semantic_content_hash
    ):
        raise ValueError("Execution materializer binds another component profile.")
    from .runtime import MujocoComponentExecutionBinding

    catalog = ActionComponentCatalog(
        catalog_id=f"{profile.profile_id}:execution-catalog",
        action_space=profile.action_schema.schema_id,
        components=tuple(
            ActionComponentSpec(
                component_id=component.component_id,
                selection_group_id=component.selection_group_id,
                effector_name=(
                    materializer.output_effector_by_component[
                        component.component_id
                    ]
                ),
                coupled=False,
            )
            for component in profile.action_schema.component_catalog.components
        ),
    )
    return MujocoComponentExecutionBinding(
        binding_id=f"{profile.profile_id}:robot-action-v1-private-edge",
        action_schema=profile.action_schema,
        execution_catalog=catalog,
        handler_contract_id=materializer.contract_id,
        handler_contract_hash=materializer.semantic_content_hash,
        materializer=materializer,
        output_effector_by_component=(
            materializer.output_effector_by_component
        ),
        support_only_output_effectors=(
            materializer.support_only_output_effectors
        ),
    )


def _build_mujoco_pose_component_target_profile(
    scene: Any,
    *,
    profile_id: str,
    arm_side_by_component: Mapping[str, str],
    gripper_name_by_component: Mapping[str, str],
    arm_reference_frame_by_component: Mapping[str, str] | None = None,
    arm_controlled_frame_by_component: Mapping[str, str] | None = None,
    selection_group_by_component: Mapping[str, str] | None = None,
    require_gripper: bool,
) -> MujocoComponentTargetProfile:
    """Build the shared arm-pose plus optional gripper-qpos profile body.

    Arm values are absolute ``xyz + quat_xyzw`` tool poses.  Gripper values are
    the actual MuJoCo joint positions in the scene-declared joint order; a
    normalized close scalar is deliberately not the canonical component value.
    """

    profile = _required(profile_id, name="profile_id")
    arms = {
        _required(component_id, name="arm component id"): _required(
            side,
            name="MuJoCo arm side",
        )
        for component_id, side in dict(arm_side_by_component).items()
    }
    grippers = {
        _required(component_id, name="gripper component id"): _required(
            gripper_name,
            name="MuJoCo gripper name",
        )
        for component_id, gripper_name in dict(
            gripper_name_by_component
        ).items()
    }
    if not arms:
        raise ValueError("A pose+qpos target profile requires at least one arm.")
    if require_gripper and not grippers:
        raise ValueError("A pose+qpos target profile requires at least one gripper.")
    if set(arms) & set(grippers):
        raise ValueError("Arm and gripper component ids must be disjoint.")
    reference_frames = {
        str(key): str(value)
        for key, value in dict(
            arm_reference_frame_by_component or {}
        ).items()
    }
    controlled_frames = {
        str(key): str(value)
        for key, value in dict(
            arm_controlled_frame_by_component or {}
        ).items()
    }
    selection_groups = {
        str(key): str(value)
        for key, value in dict(selection_group_by_component or {}).items()
    }
    known_components = set(arms) | set(grippers)
    for mapping_name, mapping in (
        ("arm_reference_frame_by_component", reference_frames),
        ("arm_controlled_frame_by_component", controlled_frames),
        ("selection_group_by_component", selection_groups),
    ):
        unknown = sorted(set(mapping) - known_components)
        if unknown:
            raise ValueError(f"{mapping_name} contains unknown components: {unknown}.")

    payloads: dict[str, ActionPayloadSchema] = {}
    components: list[ControlComponentSpecV2] = []
    for component_id, side in arms.items():
        if side not in scene.arm_specs:
            raise KeyError(f"MuJoCo scene has no arm side {side!r}.")
        reference_frame = _required(
            reference_frames.get(component_id, f"{side}_arm_base"),
            name=f"{component_id} reference frame",
        )
        controlled_frame = _required(
            controlled_frames.get(component_id, TOOL_TCP_FRAME),
            name=f"{component_id} controlled frame",
        )
        payload = ActionPayloadSchema(
            schema_id=f"{profile}:{component_id}:tool-pose",
            schema_version="1",
            representation_family="cartesian_pose",
            dtype="float32",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="absolute_target",
            reference_frame=reference_frame,
            controlled_frame=controlled_frame,
            semantic_parameters={
                "target_backend": "mujoco",
                "target_entity_kind": "arm",
                "target_arm_side": side,
            },
        )
        payloads[component_id] = payload
        components.append(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=f"mujoco.arm:{side}",
                command_channel_id=f"mujoco.arm_pose:{side}:{controlled_frame}",
                selection_group_id=selection_groups.get(
                    component_id,
                    component_id,
                ),
                resource_claims=(f"mujoco.arm_kinematic_chain:{side}",),
                accepted_payload_schema_hashes=(payload.semantic_content_hash,),
            )
        )

    for component_id, gripper_name in grippers.items():
        try:
            gripper_spec = scene.gripper_specs[gripper_name]
        except KeyError as exc:
            raise KeyError(
                f"MuJoCo scene has no gripper {gripper_name!r}."
            ) from exc
        joint_names = tuple(str(name) for name in gripper_spec.joint_names)
        payload = ActionPayloadSchema(
            schema_id=f"{profile}:{component_id}:joint-position",
            schema_version="1",
            representation_family="joint_position",
            dtype="float32",
            shape=(len(joint_names),),
            coordinate_names=joint_names,
            # The current DMTac gripper uses slide joints measured in metres.
            # Another target profile must declare its own per-joint units.
            units=tuple("m" for _ in joint_names),
            value_semantics="absolute_target",
            semantic_parameters={
                "target_backend": "mujoco",
                "target_entity_kind": "gripper",
                "target_gripper_name": gripper_name,
                "canonical_value": "physical_joint_qpos",
                "normalized_close_is_derived": True,
            },
        )
        payloads[component_id] = payload
        components.append(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=f"mujoco.gripper:{gripper_name}",
                command_channel_id=f"mujoco.gripper_qpos:{gripper_name}",
                selection_group_id=selection_groups.get(
                    component_id,
                    component_id,
                ),
                resource_claims=tuple(
                    f"mujoco.joint:{joint_name}" for joint_name in joint_names
                ),
                accepted_payload_schema_hashes=(payload.semantic_content_hash,),
            )
        )

    action_schema = ComponentActionSchema(
        schema_id=f"{profile}:component-action",
        schema_version="1",
        component_catalog=ActionComponentCatalogV2(
            catalog_id=f"{profile}:catalog",
            catalog_version="1",
            components=tuple(components),
        ),
        payload_schema_by_component=payloads,
        semantic_parameters={
            "target_profile_id": profile,
            "canonical_arm_representation": "cartesian_pose",
            "canonical_end_effector_representation": (
                "physical_joint_qpos" if grippers else "none"
            ),
            "source_model_device_independent": True,
        },
    )
    return MujocoComponentTargetProfile(
        profile_id=profile,
        action_schema=action_schema,
        arm_side_by_component=arms,
        gripper_name_by_component=grippers,
    )


def build_mujoco_pose_component_target_profile(
    scene: Any,
    *,
    profile_id: str,
    arm_side_by_component: Mapping[str, str],
    arm_reference_frame_by_component: Mapping[str, str] | None = None,
    arm_controlled_frame_by_component: Mapping[str, str] | None = None,
    selection_group_by_component: Mapping[str, str] | None = None,
) -> MujocoComponentTargetProfile:
    """Build an arms-only absolute tool-pose target profile.

    This is the narrow ingress used by generic Agent motion.  It deliberately
    exposes no hand or gripper component; a coupled solver may still receive a
    measured support-arm pose through the target-private materializer.
    """

    return _build_mujoco_pose_component_target_profile(
        scene,
        profile_id=profile_id,
        arm_side_by_component=arm_side_by_component,
        gripper_name_by_component={},
        arm_reference_frame_by_component=arm_reference_frame_by_component,
        arm_controlled_frame_by_component=arm_controlled_frame_by_component,
        selection_group_by_component=selection_group_by_component,
        require_gripper=False,
    )


def build_mujoco_pose_qpos_component_target_profile(
    scene: Any,
    *,
    profile_id: str,
    arm_side_by_component: Mapping[str, str],
    gripper_name_by_component: Mapping[str, str],
    arm_reference_frame_by_component: Mapping[str, str] | None = None,
    arm_controlled_frame_by_component: Mapping[str, str] | None = None,
    selection_group_by_component: Mapping[str, str] | None = None,
) -> MujocoComponentTargetProfile:
    """Build one arbitrary-cardinality pose+qpos target profile from a scene."""

    return _build_mujoco_pose_component_target_profile(
        scene,
        profile_id=profile_id,
        arm_side_by_component=arm_side_by_component,
        gripper_name_by_component=gripper_name_by_component,
        arm_reference_frame_by_component=arm_reference_frame_by_component,
        arm_controlled_frame_by_component=arm_controlled_frame_by_component,
        selection_group_by_component=selection_group_by_component,
        require_gripper=True,
    )


def build_rm65_right_cube_lift_component_target_profile(
    scene: Any,
    task_config: Any,
) -> MujocoComponentTargetProfile:
    """Instantiate the generic MuJoCo profile for the current right-side task."""

    return build_mujoco_pose_qpos_component_target_profile(
        scene,
        profile_id="mujoco.rm65.right_cube_lift.pose_qpos.v1",
        arm_side_by_component={"right_arm": str(task_config.controlled_side)},
        gripper_name_by_component={
            "right_gripper": str(task_config.right_gripper_name)
        },
        arm_reference_frame_by_component={"right_arm": "right_arm_base"},
        arm_controlled_frame_by_component={"right_arm": TOOL_TCP_FRAME},
        selection_group_by_component={
            "right_arm": "arms",
            "right_gripper": "right_gripper",
        },
    )


__all__ = [
    "MujocoComponentActionRobotActionV1Materializer",
    "MujocoComponentTargetProfile",
    "MujocoRobotActionV1ComponentIngressAdapter",
    "build_mujoco_component_execution_binding",
    "build_mujoco_pose_component_target_profile",
    "build_mujoco_pose_qpos_component_target_profile",
    "build_rm65_right_cube_lift_component_target_profile",
]
