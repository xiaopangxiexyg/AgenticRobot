"""Pure MuJoCo kinematics bindings for passive execution observation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ComponentActionSchema
from robot_motion.deploy.schema import pose_matrix_to_xyzquat


MUJOCO_POSE_KINEMATICS_BINDING_HASH_DOMAIN = (
    "dexteleop.sim.mujoco.pose_kinematics_binding.development"
)
MUJOCO_POSE_PROJECTOR_HASH_DOMAIN = (
    "dexteleop.sim.mujoco.pose_observation_projector.development"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _nonnegative_indices(
    values: Sequence[int],
    *,
    name: str,
) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if (
        not normalized
        or any(value < 0 for value in normalized)
        or len(normalized) != len(set(normalized))
    ):
        raise ValueError(f"{name} must contain unique non-negative indices.")
    return normalized


def _transform_tuple(value: object) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("endpoint_to_controlled must be one finite 4x4 matrix.")
    if not np.allclose(
        matrix[3],
        np.asarray((0.0, 0.0, 0.0, 1.0)),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("endpoint_to_controlled must be a rigid transform.")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=1e-7,
    ) or not np.isclose(
        np.linalg.det(rotation),
        1.0,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("endpoint_to_controlled rotation must be orthonormal.")
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True)
class MujocoPoseKinematicsBinding:
    """Bind one public pose component to one MuJoCo kinematic chain."""

    component_id: str
    qpos_source_id: str
    joint_names: tuple[str, ...]
    joint_qpos_addresses: tuple[int, ...]
    reference_frame_id: str
    reference_body_name: str
    reference_body_id: int
    controlled_frame_id: str
    endpoint_body_name: str
    endpoint_body_id: int
    endpoint_to_controlled: Sequence[Sequence[float]]
    pose_schema_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, name="component_id")
        source_id = _required(self.qpos_source_id, name="qpos_source_id")
        joint_names = tuple(
            _required(value, name="joint_name") for value in self.joint_names
        )
        if not joint_names or len(joint_names) != len(set(joint_names)):
            raise ValueError("joint_names must be unique and non-empty.")
        addresses = _nonnegative_indices(
            self.joint_qpos_addresses,
            name="joint_qpos_addresses",
        )
        if len(addresses) != len(joint_names):
            raise ValueError("joint_names and qpos addresses must have equal length.")
        reference_body_id = int(self.reference_body_id)
        endpoint_body_id = int(self.endpoint_body_id)
        if reference_body_id < 0 or endpoint_body_id < 0:
            raise ValueError("MuJoCo body IDs must be non-negative.")
        transform = _transform_tuple(self.endpoint_to_controlled)
        payload = {
            "component_id": component_id,
            "qpos_source_id": source_id,
            "joint_names": joint_names,
            "joint_qpos_addresses": addresses,
            "reference_frame_id": _required(
                self.reference_frame_id,
                name="reference_frame_id",
            ),
            "reference_body_name": _required(
                self.reference_body_name,
                name="reference_body_name",
            ),
            "reference_body_id": reference_body_id,
            "controlled_frame_id": _required(
                self.controlled_frame_id,
                name="controlled_frame_id",
            ),
            "endpoint_body_name": _required(
                self.endpoint_body_name,
                name="endpoint_body_name",
            ),
            "endpoint_body_id": endpoint_body_id,
            "endpoint_to_controlled": transform,
            "pose_schema_hash": _required(
                self.pose_schema_hash,
                name="pose_schema_hash",
            ),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=MUJOCO_POSE_KINEMATICS_BINDING_HASH_DOMAIN,
            ),
        )


class MujocoPoseObservationProjector:
    """Evaluate component-keyed FK poses without touching live ``MjData``."""

    def __init__(
        self,
        *,
        mujoco_module: Any,
        model: Any,
        bindings: Sequence[MujocoPoseKinematicsBinding],
        model_contract_id: str,
        model_contract_hash: str,
    ) -> None:
        values = tuple(bindings)
        if (
            not values
            or any(
                not isinstance(value, MujocoPoseKinematicsBinding)
                for value in values
            )
            or len({value.component_id for value in values}) != len(values)
        ):
            raise ValueError("bindings must contain unique pose components.")
        if not callable(getattr(mujoco_module, "MjData", None)) or not callable(
            getattr(mujoco_module, "mj_forward", None)
        ):
            raise TypeError("mujoco_module must expose MjData and mj_forward.")
        nq = int(getattr(model, "nq"))
        if nq <= 0:
            raise ValueError("MuJoCo model must expose a positive nq.")
        for binding in values:
            if max(binding.joint_qpos_addresses) >= nq:
                raise ValueError(
                    f"Binding {binding.component_id!r} exceeds model nq."
                )
        self._mujoco = mujoco_module
        self._model = model
        self._nq = nq
        self._bindings = MappingProxyType(
            {value.component_id: value for value in values}
        )
        self._scratch = mujoco_module.MjData(model)
        self._lock = threading.Lock()
        self.model_contract_id = _required(
            model_contract_id,
            name="model_contract_id",
        )
        self.model_contract_hash = _required(
            model_contract_hash,
            name="model_contract_hash",
        )
        self.semantic_content_hash = canonical_content_hash(
            {
                "model_contract_id": self.model_contract_id,
                "model_contract_hash": self.model_contract_hash,
                "binding_hash_by_component": {
                    value.component_id: value.semantic_content_hash
                    for value in values
                },
                "live_state_mutation": False,
                "fk_engine": "mujoco.private_mjdata",
            },
            domain=MUJOCO_POSE_PROJECTOR_HASH_DOMAIN,
        )
        self.contract_id = (
            f"mujoco-pose-observation-projector:{self.semantic_content_hash}"
        )

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(self._bindings)

    @property
    def schema_hash_by_component(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                component_id: binding.pose_schema_hash
                for component_id, binding in self._bindings.items()
            }
        )

    def project(
        self,
        *,
        model_qpos: Sequence[float],
        qpos_by_source: Mapping[str, Sequence[float]],
        component_ids: Sequence[str],
    ) -> Mapping[str, tuple[float, ...]]:
        requested = tuple(
            _required(value, name="component_id") for value in component_ids
        )
        if (
            not requested
            or len(requested) != len(set(requested))
            or not set(requested).issubset(self._bindings)
        ):
            raise ValueError(
                "component_ids must be a unique non-empty projector subset."
            )
        background = np.asarray(model_qpos, dtype=float).reshape(-1)
        if background.shape != (self._nq,) or not np.all(np.isfinite(background)):
            raise ValueError("model_qpos must match the finite MuJoCo model qpos.")
        source_values = {
            _required(source_id, name="qpos source id"): np.asarray(
                qpos,
                dtype=float,
            ).reshape(-1)
            for source_id, qpos in qpos_by_source.items()
        }
        output: dict[str, tuple[float, ...]] = {}
        with self._lock:
            self._scratch.qpos[:] = background
            applied_by_source: dict[str, tuple[float, ...]] = {}
            for component_id in requested:
                binding = self._bindings[component_id]
                try:
                    qpos = source_values[binding.qpos_source_id]
                except KeyError as exc:
                    raise ValueError(
                        f"Missing qpos source {binding.qpos_source_id!r}."
                    ) from exc
                if qpos.shape != (len(binding.joint_qpos_addresses),) or not np.all(
                    np.isfinite(qpos)
                ):
                    raise ValueError(
                        f"Qpos source {binding.qpos_source_id!r} has another layout."
                    )
                normalized = tuple(float(value) for value in qpos)
                previous = applied_by_source.get(binding.qpos_source_id)
                if previous is not None and previous != normalized:
                    raise ValueError(
                        "One qpos source cannot carry conflicting values in one cut."
                    )
                applied_by_source[binding.qpos_source_id] = normalized
                for address, value in zip(binding.joint_qpos_addresses, qpos):
                    self._scratch.qpos[address] = float(value)
            self._mujoco.mj_forward(self._model, self._scratch)
            for component_id in requested:
                binding = self._bindings[component_id]
                world_from_reference = _body_pose(
                    self._scratch,
                    binding.reference_body_id,
                )
                world_from_endpoint = _body_pose(
                    self._scratch,
                    binding.endpoint_body_id,
                )
                reference_from_controlled = (
                    np.linalg.inv(world_from_reference)
                    @ world_from_endpoint
                    @ np.asarray(binding.endpoint_to_controlled, dtype=float)
                )
                pose = pose_matrix_to_xyzquat(reference_from_controlled)
                if pose is None:
                    raise ValueError(
                        f"FK pose for {component_id!r} is not representable."
                    )
                output[component_id] = tuple(
                    float(value) for value in np.asarray(pose).reshape(7)
                )
        return MappingProxyType(output)


def build_mujoco_pose_observation_projector(
    scene: Any,
    *,
    action_schema: ComponentActionSchema,
    qpos_source_by_component: Mapping[str, str],
    joint_names_by_source: Mapping[str, Sequence[str]],
    reference_body_by_component: Mapping[str, str],
    endpoint_body_by_component: Mapping[str, str],
    endpoint_to_controlled_by_component: Mapping[
        str,
        Sequence[Sequence[float]],
    ],
    model_contract_id: str,
    model_contract_hash: str,
) -> MujocoPoseObservationProjector:
    """Build a topology-neutral projector from an explicit target binding."""

    if not isinstance(action_schema, ComponentActionSchema):
        raise TypeError("action_schema must be a ComponentActionSchema.")
    component_sources = {
        _required(component_id, name="component_id"): _required(
            source_id,
            name="qpos_source_id",
        )
        for component_id, source_id in qpos_source_by_component.items()
    }
    if not component_sources:
        raise ValueError("qpos_source_by_component must not be empty.")
    expected_components = set(component_sources)
    for name, values in (
        ("reference_body_by_component", reference_body_by_component),
        ("endpoint_body_by_component", endpoint_body_by_component),
        (
            "endpoint_to_controlled_by_component",
            endpoint_to_controlled_by_component,
        ),
    ):
        if set(values) != expected_components:
            raise ValueError(f"{name} must cover every pose component exactly.")
    bindings: list[MujocoPoseKinematicsBinding] = []
    for component_id, source_id in component_sources.items():
        try:
            payload_schema = action_schema.payload_schema_by_component[
                component_id
            ]
        except KeyError as exc:
            raise ValueError(
                f"Unknown action component {component_id!r}."
            ) from exc
        if (
            payload_schema.representation_family != "cartesian_pose"
            or payload_schema.shape != (7,)
            or payload_schema.reference_frame is None
            or payload_schema.controlled_frame is None
        ):
            raise ValueError(
                f"Component {component_id!r} must declare one framed pose schema."
            )
        try:
            joint_names = tuple(joint_names_by_source[source_id])
        except KeyError as exc:
            raise ValueError(
                f"Missing joint layout for qpos source {source_id!r}."
            ) from exc
        reference_body = _required(
            reference_body_by_component[component_id],
            name="reference body",
        )
        endpoint_body = _required(
            endpoint_body_by_component[component_id],
            name="endpoint body",
        )
        bindings.append(
            MujocoPoseKinematicsBinding(
                component_id=component_id,
                qpos_source_id=source_id,
                joint_names=joint_names,
                joint_qpos_addresses=tuple(
                    int(scene.joint_qpos_addr(name)) for name in joint_names
                ),
                reference_frame_id=payload_schema.reference_frame,
                reference_body_name=reference_body,
                reference_body_id=int(scene.body_id(reference_body)),
                controlled_frame_id=payload_schema.controlled_frame,
                endpoint_body_name=endpoint_body,
                endpoint_body_id=int(scene.body_id(endpoint_body)),
                endpoint_to_controlled=(
                    endpoint_to_controlled_by_component[component_id]
                ),
                pose_schema_hash=payload_schema.semantic_content_hash,
            )
        )
    return MujocoPoseObservationProjector(
        mujoco_module=scene.mujoco,
        model=scene.model,
        bindings=bindings,
        model_contract_id=model_contract_id,
        model_contract_hash=model_contract_hash,
    )


def _body_pose(data: Any, body_id: int) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray(data.xpos[body_id], dtype=float)
    pose[:3, :3] = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    return pose


__all__ = [
    "MUJOCO_POSE_KINEMATICS_BINDING_HASH_DOMAIN",
    "MUJOCO_POSE_PROJECTOR_HASH_DOMAIN",
    "MujocoPoseKinematicsBinding",
    "MujocoPoseObservationProjector",
    "build_mujoco_pose_observation_projector",
]
