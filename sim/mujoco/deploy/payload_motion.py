"""Scoped attached-object queries; no plant, grasp, or dispatch authority.

Capture under the existing rig scene lock. Trusted read-only callbacks receive
private native data, and must return calibrated world-from-TCP and *measured*
closed state. A close receipt is not a measurement. Capture requires both named
sensor contacts; this establishes contact_secured, never a verified hold.

tool_from_object is the native body frame (not COM or an enclosing box). Pass
the original transform on later captures to validate it without re-anchoring.
Planning moves the complete native collision geometry in private MjData only.
check_measured and the additional canonical contact oracle always inspect the
actual body relation and contacts, including after the planner was cleared.
These checks do not authenticate callbacks or replace the runtime's TTL,
authority, held-evidence provider, physics owner, or canonical safety checks.
The assembly must keep model geometry immutable for a planned/active operation.
Full native geometry hashes are validated at capture, plan and activation, not
on every candidate or physics tick. Revoke the operation before model edits.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.object_transport import rigid_pose
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.pose import pose_delta_metrics
from sim.mujoco.collision_safety import KinematicCollisionProjector
from sim.mujoco.collision_safety.oracle import CollisionSafetyOracle, _ForbiddenGeomPair
from sim.mujoco.collision_safety.policy import NonContactObjectMovePolicy
from sim.mujoco.collision_safety.result import CollisionCheckResult
from sim.mujoco.deploy.contact_timing import is_single_physics_step_elapsed

_MAX_RELATIVE_ROTATION_RAD = math.radians(2)


def _vector(value, size, name):
    if isinstance(value, (tuple, list)) and any(isinstance(x, (bool, np.bool_)) for x in value):
        raise ValueError(f"{name} must not contain boolean values")
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} requires {size} finite real values")
    return tuple(float(x) for x in array)


def _names(value, name):
    if isinstance(value, str):
        raise TypeError(f"{name} requires a sequence of native names")
    values = tuple(value)
    if (not values or len(set(values)) != len(values)
            or any(not isinstance(n, str) or not n or n.strip() != n for n in values)):
        raise ValueError(f"{name} requires explicit unique native names")
    return values


def _body_pose(data, body):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = data.xmat[body].reshape(3, 3), data.xpos[body]
    return rigid_pose(pose, name="native world_from_body")


def _model_signature(model):
    digest = hashlib.sha256(model.names)
    digest.update(np.asarray([model.opt.timestep], dtype=np.float64).tobytes())
    # Compiled native mesh vertices/faces and offsets are part of geometry too.
    for name in ("body_parentid", "body_pos", "body_quat", "body_mocapid", "body_jntadr", "body_jntnum",
                 "body_dofnum", "jnt_type", "jnt_qposadr", "jnt_axis", "jnt_pos",
                 "geom_type", "geom_bodyid", "geom_contype", "geom_conaffinity", "geom_size",
                 "geom_pos", "geom_quat", "geom_dataid", "geom_aabb", "exclude_signature",
                 "mesh_vertadr", "mesh_vertnum", "mesh_vert", "mesh_faceadr", "mesh_facenum",
                 "mesh_face", "eq_type", "eq_objtype", "eq_obj1id", "eq_obj2id", "eq_data"):
        digest.update(getattr(model, name).tobytes())
    return digest.hexdigest()


def _require_current_cut(source, actual):
    """Accept the live cut or its exact kinematic/contact-input copy only."""
    if source.model is not actual.model:
        raise ValueError("Actual payload cut belongs to another model")
    if source is actual:
        return
    if source.time != actual.time or any(not np.array_equal(getattr(source, name), getattr(actual, name))
            for name in ("qpos", "qvel", "mocap_pos", "mocap_quat", "eq_active")):
        raise ValueError("Payload measurement requires the current live native cut")


@dataclass(frozen=True)
class MujocoPayloadSupport:
    """Explicit fixed native plane/box face and finite projected contact region.

    normal_world points out of the named support face. region bounds contain
    the complete object's footprint projected onto that face. Lift may leave
    initial support only along this normal (within lateral_tolerance_m while
    near support); place permits approach into this region. Neither permits
    another object/support pair or penetration beyond the scene hard bound.
    Actual startup tracking may undershoot by max_penetration_m, with native
    penetration checked independently. Commanded lift candidates stay strict.
    """

    geom_names: tuple[str, ...]
    normal_world: tuple[float, float, float]
    region_min_world: tuple[float, float, float]
    region_max_world: tuple[float, float, float]
    phase: str
    max_penetration_m: float
    lateral_tolerance_m: float = .003
    dynamic_body_name: str | None = None
    frozen_pose_xyz_quat_xyzw: tuple[float, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "geom_names", _names(self.geom_names, "Support geoms"))
        for name in ("normal_world", "region_min_world", "region_max_world"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name))
        if (self.phase not in {"lift", "place"}
                or not math.isclose(math.hypot(*self.normal_world), 1., abs_tol=1e-8, rel_tol=0)
                or np.any(np.asarray(self.region_min_world) > self.region_max_world)):
            raise ValueError("Support needs lift/place, a unit outward normal and ordered finite region")
        for name in ("max_penetration_m", "lateral_tolerance_m"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.lateral_tolerance_m <= .003:
            raise ValueError("Lift lateral tolerance must retain the 3mm bound")
        if (self.dynamic_body_name is None) != (self.frozen_pose_xyz_quat_xyzw is None):
            raise ValueError("Dynamic support requires both body identity and frozen pose")
        if self.dynamic_body_name is not None:
            _names((self.dynamic_body_name,), "Dynamic support body")
            pose = _vector(self.frozen_pose_xyz_quat_xyzw, 7, "Frozen support pose")
            rigid_pose(xyzquat_to_pose_matrix(pose), name="frozen support pose")
            object.__setattr__(self, "frozen_pose_xyz_quat_xyzw", pose)

    def to_payload(self):
        payload = dict(vars(self))
        if self.dynamic_body_name is None:
            payload.pop("dynamic_body_name")
            payload.pop("frozen_pose_xyz_quat_xyzw")
        return payload


@dataclass(frozen=True, init=False, eq=False)
class MujocoPayloadBinding:
    """Frozen same-cut binding, with measured checks separate from prediction.

    capture is the public constructor. source_model_qpos is a complete immutable
    cut for planning. Later operations capture a new cut but pass the original
    tool_from_object. Actual checks retain at most the existing 3mm/2deg follow
    tolerances. No static contact or geometric plan establishes held_verified.
    """

    target_body_name: str
    sensor_geom_names: tuple[str, str]
    tool_from_object: tuple[tuple[float, ...], ...]
    source_model_qpos: tuple[float, ...]
    source_mocap_pos: tuple[tuple[float, ...], ...]
    source_mocap_quat: tuple[tuple[float, ...], ...]
    source_eq_active: tuple[bool, ...]
    support: MujocoPayloadSupport | None
    translation_tolerance_m: float
    rotation_tolerance_rad: float
    _model: Any = field(repr=False)
    _live_data: Any = field(repr=False)
    _physics_timestep_sec: float = field(repr=False)
    _signature: str = field(repr=False)
    _tcp: Callable = field(repr=False)
    _closed: Callable = field(repr=False)
    _anchor_tool: Any = field(repr=False)
    _policy: Any = field(repr=False)
    _measured_guard: Callable | None = field(repr=False)
    _measured_oracle: Any = field(repr=False)
    guard_identity: str | None

    @classmethod
    def capture(cls, scene, source_data, *, target_body_name, sensor_geom_names,
                tcp_pose_from_data, gripper_closed_from_data, tool_from_object=None,
                support=None, translation_tolerance_m=.003,
                rotation_tolerance_rad=_MAX_RELATIVE_ROTATION_RAD, measured_guard=None,
                guard_identity=None):
        """Caller holds the scene lock; all derivations use one private native cut."""
        if source_data.model is not scene.model:
            raise ValueError("Payload source must belong to this exact native scene model")
        _require_current_cut(source_data, scene.data)
        timestep = float(scene.model.opt.timestep)
        if not math.isfinite(timestep) or timestep <= 0:
            raise ValueError("Payload physics timestep must be positive and finite")
        sensors = _names(sensor_geom_names, "Payload sensors")
        _names((target_body_name,), "Payload target")
        if len(sensors) != 2:
            raise ValueError("Payload requires exactly two selected sensor geoms")
        if not callable(tcp_pose_from_data) or not callable(gripper_closed_from_data):
            raise TypeError("Trusted canonical TCP and measured-closed callbacks are required")
        if support is not None and not isinstance(support, MujocoPayloadSupport):
            raise TypeError("support must be MujocoPayloadSupport or None")
        if (measured_guard is None) != (guard_identity is None) or (measured_guard is not None and (
                not callable(measured_guard) or not isinstance(guard_identity, str)
                or not guard_identity.strip() or guard_identity.strip() != guard_identity)):
            raise ValueError("Measured guard requires a read-only callable and explicit guard_identity together")
        for value, maximum in ((translation_tolerance_m, .003),
                               (rotation_tolerance_rad, _MAX_RELATIVE_ROTATION_RAD)):
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError("Payload follow tolerance must retain the positive 3mm/2deg bounds")
        original = scene.collision_projector()
        if original is None:
            raise ValueError("Payload requires the canonical scene collision projector")
        binding = object.__new__(cls)
        fields = {"target_body_name": target_body_name, "sensor_geom_names": sensors,
            "source_model_qpos": _vector(source_data.qpos, scene.model.nq, "source qpos"),
            "source_mocap_pos": tuple(tuple(float(v) for v in row) for row in source_data.mocap_pos),
            "source_mocap_quat": tuple(tuple(float(v) for v in row) for row in source_data.mocap_quat),
            "source_eq_active": tuple(bool(v) for v in source_data.eq_active),
            "support": support, "translation_tolerance_m": float(translation_tolerance_m),
            "rotation_tolerance_rad": float(rotation_tolerance_rad), "_model": scene.model,
            "_live_data": scene.data, "_physics_timestep_sec": timestep,
            "_signature": _model_signature(scene.model), "_tcp": tcp_pose_from_data,
            "_closed": gripper_closed_from_data, "_policy": MappingProxyType(dict(vars(original.policy))),
            "_measured_guard": measured_guard, "guard_identity": guard_identity}
        for name, value in fields.items():
            object.__setattr__(binding, name, value)
        data = binding._copy_actual(scene, source_data)
        tcp = rigid_pose(tcp_pose_from_data(data), name="canonical world_from_tcp")
        relative = np.linalg.inv(tcp) @ _body_pose(data, scene.model.body(target_body_name).id)
        frozen = rigid_pose(relative if tool_from_object is None else tool_from_object,
                            name="frozen tool_from_object")
        object.__setattr__(binding, "tool_from_object", tuple(tuple(float(x) for x in row) for row in frozen))
        object.__setattr__(binding, "_anchor_tool", tuple(tuple(float(x) for x in row) for row in tcp))
        object.__setattr__(binding, "_measured_oracle", _PayloadSceneOracle(
            scene.mujoco, scene.model, NonContactObjectMovePolicy(**binding._policy), binding,
            actual_scene=scene))
        check = binding.check_measured(scene, source_data=data)
        if not check.safe:
            raise ValueError(f"Payload capture rejected: {check.reason}: {dict(check.metadata)}")
        return binding

    def to_payload(self):
        return {"target_body_name": self.target_body_name, "sensor_geom_names": self.sensor_geom_names,
                "tool_from_object": self.tool_from_object, "source_model_qpos": self.source_model_qpos,
                "source_mocap_pos": self.source_mocap_pos, "source_mocap_quat": self.source_mocap_quat,
                "source_eq_active": self.source_eq_active,
                "support": None if self.support is None else self.support.to_payload(),
                "translation_tolerance_m": self.translation_tolerance_m,
                "rotation_tolerance_rad": self.rotation_tolerance_rad,
                "native_geometry_hash": self._signature, "scene_policy": dict(self._policy),
                "guard_identity": self.guard_identity}

    @property
    def semantic_content_hash(self):
        return canonical_content_hash(self.to_payload(), domain="mujoco.attached_payload.v1")

    def validate_scene(self, scene, *, geometry=True):
        self._validate_timestep()
        if (scene.model is not self._model or (geometry and _model_signature(scene.model) != self._signature)
                or dict(vars(scene.collision_projector().policy)) != self._policy):
            raise ValueError("Payload native geometry/model/policy changed; capture a new binding")

    def _validate_timestep(self):
        if float(self._model.opt.timestep) != self._physics_timestep_sec:
            raise ValueError("Payload physics timestep changed; the binding is invalid")

    def validate_planning_source(self, scene, model_qpos, *, preview=False):
        self.validate_scene(scene)
        if not preview and not np.array_equal(model_qpos, self.source_model_qpos):
            raise ValueError("Payload planning requires the binding's exact same-cut source qpos")

    def _restore_planning_auxiliary_state(self, data):
        """Copy the frozen non-qpos geometry inputs into caller-owned scratch."""
        if data.model is not self._model or data is self._live_data:
            raise ValueError("Payload planning state requires private same-model native data")
        data.mocap_pos[:] = np.asarray(self.source_mocap_pos).reshape(self._model.nmocap, 3)
        data.mocap_quat[:] = np.asarray(self.source_mocap_quat).reshape(self._model.nmocap, 4)
        data.eq_active[:] = self.source_eq_active

    def _require_planning_cut(self, data):
        self._validate_timestep()
        if data.model is not self._model or not np.array_equal(data.qpos, self.source_model_qpos) or any(
                not np.array_equal(getattr(data, name), np.asarray(frozen).reshape(getattr(data, name).shape))
                for name, frozen in (("mocap_pos", self.source_mocap_pos),
                                     ("mocap_quat", self.source_mocap_quat),
                                     ("eq_active", self.source_eq_active))):
            raise ValueError("Payload query requires its frozen same-cut geometry source")

    def _require_actual_cut(self, scene, source):
        self.validate_scene(scene, geometry=False)
        if scene.data is not self._live_data:
            raise ValueError("Payload live native data owner changed")
        _require_current_cut(source, self._live_data)

    def _copy_actual(self, scene, source):
        self._require_actual_cut(scene, source)
        if source.model is not self._model:
            raise ValueError("Actual payload cut must belong to the bound model")
        if (not np.isfinite(source.qpos).all() or not np.isfinite(source.qvel).all()
                or not np.isfinite(source.mocap_pos).all() or not np.isfinite(source.mocap_quat).all()
                or not math.isfinite(source.time) or source.time < 0):
            raise ValueError("Actual payload state/time must be finite and nonnegative")
        for joint, kind in enumerate(scene.model.jnt_type):
            offset = int(scene.model.jnt_qposadr[joint])
            if kind == scene.mujoco.mjtJoint.mjJNT_FREE:
                offset += 3
            elif kind != scene.mujoco.mjtJoint.mjJNT_BALL:
                continue
            if not math.isclose(math.hypot(*source.qpos[offset:offset+4]), 1., abs_tol=1e-6, rel_tol=0):
                raise ValueError("Actual payload state requires unit native quaternions")
        if not np.allclose(np.linalg.norm(source.mocap_quat, axis=1), 1., atol=1e-6, rtol=0):
            raise ValueError("Actual payload state requires unit mocap quaternions")
        scratch = scene.mujoco.MjData(scene.model)
        scene.mujoco.mj_copyData(scratch, scene.model, source)
        scene.mujoco.mj_kinematics(scene.model, scratch)
        scene.mujoco.mj_collision(scene.model, scratch)
        return scratch

    def contact_oracle(self, scene):
        """One actual-state oracle per binding; caller holds the rig scene lock.

        Physics and pull measurements must share the same last-bilateral cut.
        Prediction stays isolated in MujocoPayloadProjector's own oracle.
        """
        self.validate_scene(scene)
        return self._measured_oracle

    def check_measured(self, scene, *, source_data=None, _oracle=None) -> JointConfigurationCheck:
        """Read actual relation, closed/bilateral contacts and canonical geometry.

        This survives HOLD/clear. Caller must hold the existing rig scene lock.
        Invalid input raises; measured safety failures return an unsafe check.
        """
        data = self._copy_actual(scene, scene.data if source_data is None else source_data)
        oracle = self._measured_oracle if _oracle is None else _oracle
        if oracle is not self._measured_oracle or oracle.model is not scene.model:
            raise ValueError("Measured payload oracle must belong to this exact binding/model")
        answer = oracle.check_contacts(data)
        return JointConfigurationCheck(safe=answer.safe,
            minimum_distance_m=answer.minimum_distance_m if math.isfinite(answer.minimum_distance_m) else None,
            reason=None if answer.safe else "mujoco_payload_measured_guard_failed",
            metadata={**answer.to_dict(), **oracle.last_measured_details})


class _PayloadSceneOracle(CollisionSafetyOracle):
    """Canonical pair checks plus only this payload's full native swept volume."""

    def __init__(self, mujoco, model, policy, binding, *, actual_scene=None):
        self.binding = binding
        self._actual_scene = actual_scene
        self._body = int(model.body(binding.target_body_name).id)
        joint = int(model.body_jntadr[self._body])
        if (self._body == 0 or model.body_parentid[self._body] != 0
                or model.body_jntnum[self._body] != 1
                or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
                or np.any(model.body_parentid[1:] == self._body)):
            raise ValueError("Payload must be a top-level free rigid body without descendants")
        self.object_qpos_address = int(model.jnt_qposadr[joint])
        self._geoms = frozenset(int(g) for g in np.flatnonzero((model.geom_bodyid == self._body)
            & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))))
        if not self._geoms or any(model.geom_contype[g] != 8 or model.geom_type[g] not in (
                mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH) for g in self._geoms):
            raise ValueError("Complete payload requires all native target collision geoms as object boxes/meshes")
        self._sensors = tuple(int(model.geom(n).id) for n in binding.sensor_geom_names)
        if (any(model.geom_contype[g] != 4 for g in self._sensors)
                or len({int(model.geom_bodyid[g]) for g in self._sensors}) != 2):
            raise ValueError("Payload requires two distinct native hand sensor bodies")
        self._supports = () if binding.support is None else tuple(int(model.geom(n).id) for n in binding.support.geom_names)
        self._dynamic_support_body = None
        self._frozen_support_pose = None
        if binding.support is not None:
            if binding.support.max_penetration_m > policy.hard_penetration_m:
                raise ValueError("Payload support cannot exceed the scene hard penetration bound")
            if binding.support.dynamic_body_name is not None:
                body = int(model.body(binding.support.dynamic_body_name).id)
                joint = int(model.body_jntadr[body])
                if (body == self._body or model.body_parentid[body] != 0
                        or model.body_jntnum[body] != 1
                        or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
                        or np.any(model.body_parentid[1:] == body)):
                    raise ValueError("Dynamic support requires another top-level free rigid body")
                self._dynamic_support_body = body
                self._frozen_support_pose = xyzquat_to_pose_matrix(
                    binding.support.frozen_pose_xyz_quat_xyzw)
                collision_geoms = {int(g) for g in np.flatnonzero(
                    (model.geom_bodyid == body)
                    & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))}
                if set(self._supports) != collision_geoms:
                    raise ValueError("Dynamic support must bind its complete collision geometry")
            for geom in self._supports:
                body = int(model.geom_bodyid[geom])
                if self._dynamic_support_body is not None:
                    if (body != self._dynamic_support_body
                            or model.geom_contype[geom] != 8
                            or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX):
                        raise ValueError("Dynamic support must use its complete native box identity")
                    continue
                while body and model.body_dofnum[body] == 0:
                    if model.body_mocapid[body] >= 0:
                        raise ValueError("Payload support must be fixed, not mocap-driven")
                    body = int(model.body_parentid[body])
                if body or model.geom_contype[geom] != 1 or model.geom_type[geom] not in (
                        mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_BOX):
                    raise ValueError("Payload support must name fixed native environment planes/boxes")
        super().__init__(mujoco, model, policy, pair_clearance_m={
            tuple(sorted((sensor, geom))): -policy.hard_penetration_m
            for sensor in self._sensors for geom in self._geoms})
        self._actual_scratch = mujoco.MjData(model)
        self._last_bilateral_contact_time = None
        # Pre/post checks may inspect the same cut; elapsed simulation time owns this one-step window.
        self._contact_dropout_grace_sec = float(model.opt.timestep)
        self.last_measured_details = {}

    def _build_pairs(self):
        pairs = list(super()._build_pairs())
        keys = {(p.geom1, p.geom2) for p in pairs}
        for geom in self._geoms:
            for other in range(self.model.ngeom):
                if other in self._geoms or not (self.model.geom_contype[other] or self.model.geom_conaffinity[other]):
                    continue
                a, b = sorted((geom, other))
                if (a, b) in keys:
                    continue
                bodies = [int(self.model.geom_bodyid[g]) for g in (a, b)]
                groups = sorted(int(self.model.geom_contype[g]) for g in (a, b))
                pairs.append(_ForbiddenGeomPair(a, b,
                    *(self._name(self.mujoco.mjtObj.mjOBJ_GEOM, g, f"<geom:{g}>") for g in (a, b)),
                    *(self._name(self.mujoco.mjtObj.mjOBJ_BODY, body, f"<body:{body}>") for body in bodies),
                    f"{groups[0]}:{groups[1]}"))
                keys.add((a, b))
        return tuple(pairs)

    def _support_allowed(self, data, geom, support_geom, *, measured=False):
        support, model = self.binding.support, self.model
        normal = np.asarray(support.normal_world)
        rotation = data.geom_xmat[support_geom].reshape(3, 3)
        local = rotation.T @ normal
        axis = int(np.argmax(np.abs(local)))
        if model.geom_type[support_geom] == self.mujoco.mjtGeom.mjGEOM_PLANE:
            aligned = np.allclose(local, (0, 0, 1), atol=1e-7, rtol=0)
            face = data.geom_xpos[support_geom]
        else:
            expected = np.zeros(3)
            expected[axis] = np.sign(local[axis])
            aligned = np.allclose(local, expected, atol=1e-7, rtol=0)
            face = data.geom_xpos[support_geom] + normal * model.geom_size[support_geom, axis]
        if not aligned:
            raise ValueError("Explicit support normal must match an outward native support face")
        # Native geometry, including rotated/offset boxes and convex mesh vertices.
        if model.geom_type[geom] == self.mujoco.mjtGeom.mjGEOM_BOX:
            vertices = np.asarray(tuple(itertools.product((-1., 1.), repeat=3))) * model.geom_size[geom]
        else:
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start:start+count]
        points = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        height = (points - face) @ normal
        footprint = points - height[:, None] * normal
        if (not np.isfinite(points).all() or np.min(height) < -support.max_penetration_m
                or np.any(footprint < np.asarray(support.region_min_world) - 1e-9)
                or np.any(footprint > np.asarray(support.region_max_world) + 1e-9)):
            return False
        if model.geom_type[support_geom] == self.mujoco.mjtGeom.mjGEOM_BOX:
            local_points = (footprint - data.geom_xpos[support_geom]) @ rotation
            tangents = [i for i in range(3) if i != axis]
            if np.any(np.abs(local_points[:, tangents]) > model.geom_size[support_geom, tangents] + 1e-9):
                return False
        tcp = rigid_pose(self.binding._tcp(data), name="canonical world_from_tcp")
        delta = tcp[:3, 3] - np.asarray(self.binding._anchor_tool)[:3, 3]
        rise = float(delta @ normal)
        # Intent is strictly non-descending. Actual tracking may undershoot
        # within this support's existing hard-depth budget; full native object
        # penetration, footprint, bilateral contact and rigid relation still
        # have independent checks. Never tune this to one observed jitter.
        minimum_lift_rise = -support.max_penetration_m if measured else -1e-9
        return not (np.linalg.norm(delta - rise * normal) > support.lateral_tolerance_m
                or (support.phase == "lift" and rise < minimum_lift_rise)
                or (support.phase == "place" and rise > .003))

    def check(self, data, *, _measured_support=False, **kwargs):
        if self._dynamic_support_body is not None:
            translation, angle = pose_delta_metrics(
                self._frozen_support_pose, _body_pose(data, self._dynamic_support_body))
            if translation > .003 or angle > 2.:
                return CollisionCheckResult(
                    safe=False, minimum_distance_m=0.,
                    body1_name=self.binding.support.dynamic_body_name,
                    pair_class="payload_support_moved")
        with np.errstate(over="raise", invalid="raise"):
            rotation = np.asarray(data.geom_xmat).reshape((-1, 3, 3))
            position = np.asarray(data.geom_xpos)
            center = position + np.einsum("nij,nj->ni", rotation, self.model.geom_aabb[:, :3])
            half = np.einsum("nij,nj->ni", np.abs(rotation), self.model.geom_aabb[:, 3:])
            # einsum/native FK do not reliably signal overflow via errstate.
            if (not np.isfinite(position).all() or not np.isfinite(rotation).all()
                    or not np.isfinite(center).all() or not np.isfinite(half).all()):
                raise ValueError("Payload checked native geometry/AABBs must be finite")
            for geom in self._geoms:
                for support in self._supports:
                    key = tuple(sorted((geom, support)))
                    self._pair_clearance_m.pop(key, None)
                    if self._support_allowed(data, geom, support, measured=_measured_support):
                        # Instance-local, phase/region-specific pair threshold only.
                        self._pair_clearance_m[key] = -self.binding.support.max_penetration_m
            return super().check(data, **kwargs)

    def check_contacts(self, data):
        return self._check_contacts(data, publish=True)

    def _check_source_without_publication(self, data):
        # A sweep validates its actual source, not speculative candidates.
        # Do not publish a new history cut or invoke the external observer.
        previous = self.last_measured_details
        try:
            return self._check_contacts(data, publish=False)
        finally:
            self.last_measured_details = previous

    def _check_contacts(self, data, *, publish):
        self.binding._validate_timestep()
        if self._actual_scene is not None:
            self.binding.validate_scene(self._actual_scene, geometry=False)
            if self._actual_scene.data is not self.binding._live_data:
                raise ValueError("Payload live native data owner changed")
            _require_current_cut(data, self.binding._live_data)
        if data.model is not self.model:
            raise ValueError("Payload guard native model changed")
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                or not np.isfinite(data.mocap_pos).all() or not np.isfinite(data.mocap_quat).all()
                or not math.isfinite(data.time) or data.time < 0):
            raise ValueError("Payload guard actual state/time must be finite")
        for joint, kind in enumerate(self.model.jnt_type):
            address = int(self.model.jnt_qposadr[joint])
            if kind == self.mujoco.mjtJoint.mjJNT_FREE:
                address += 3
            elif kind != self.mujoco.mjtJoint.mjJNT_BALL:
                continue
            if not math.isclose(math.hypot(*data.qpos[address:address+4]), 1., abs_tol=1e-6, rel_tol=0):
                raise ValueError("Payload guard requires unit native quaternions")
        if not np.allclose(np.linalg.norm(data.mocap_quat, axis=1), 1., atol=1e-6, rtol=0):
            raise ValueError("Payload guard requires unit mocap quaternions")
        # mj_step may have integrated qpos after producing cached xpos/contact.
        # Recompute only in private scratch so actual relation and contacts use
        # the same current cut at both existing pre/post-step call sites.
        self.mujoco.mj_copyData(self._actual_scratch, self.model, data)
        data = self._actual_scratch
        self.mujoco.mj_kinematics(self.model, data)
        self.mujoco.mj_collision(self.model, data)
        self.last_measured_details = {}
        obj = _body_pose(data, self._body)
        tcp = rigid_pose(self.binding._tcp(data), name="actual canonical world_from_tcp")
        relative = np.linalg.inv(tcp) @ obj
        slip, angle = pose_delta_metrics(np.asarray(self.binding.tool_from_object), relative)
        closed = self.binding._closed(data)
        if not isinstance(closed, (bool, np.bool_)):
            raise TypeError("Measured closed callback must return a boolean")
        touched = set()
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            if not math.isfinite(contact.dist) or not (0 <= a < self.model.ngeom and 0 <= b < self.model.ngeom):
                raise ValueError("Invalid native contact distance/identity")
            if contact.dist <= 0:
                if a in self._geoms:
                    touched.add(b)
                if b in self._geoms:
                    touched.add(a)
        for index in range(self.model.neq):
            if data.eq_active[index] and self.model.eq_type[index] in (
                    self.mujoco.mjtEq.mjEQ_WELD, self.mujoco.mjtEq.mjEQ_CONNECT):
                bodies = (self.model.eq_obj1id[index], self.model.eq_obj2id[index])
                if self.model.eq_objtype[index] == self.mujoco.mjtObj.mjOBJ_SITE:
                    bodies = tuple(int(self.model.site_bodyid[s]) for s in bodies)
                if self._body in bodies:
                    raise ValueError("Payload guard requires an unconstrained free object")
        sides = tuple(sensor in touched for sensor in self._sensors)
        unsupported_touch = any(g not in self._sensors and g not in self._supports for g in touched)
        previous_bilateral_time = self._last_bilateral_contact_time
        contact_age = (None if previous_bilateral_time is None else float(data.time) - previous_bilateral_time)
        relation_valid = (closed and slip <= self.binding.translation_tolerance_m
                          and math.radians(angle) <= self.binding.rotation_tolerance_rad)
        dropout_grace = bool(
            not all(sides) and relation_valid and not unsupported_touch
            and contact_age is not None
            and is_single_physics_step_elapsed(
                previous_bilateral_time, float(data.time), self._contact_dropout_grace_sec))
        self.last_measured_details = {"sensor_contacts": sides, "measured_closed": bool(closed),
            "relative_translation_error_m": slip, "relative_rotation_error_rad": math.radians(angle),
            "unsupported_target_contact": unsupported_touch,
            "sensor_contact_dropout_grace": {
                "accepted": dropout_grace,
                "age_sec": contact_age,
                "maximum_sec": self._contact_dropout_grace_sec,
                "scope": "one MuJoCo physics step after a same-session bilateral contact cut",
            }}
        # Always run canonical contacts and actual full native scene geometry.
        contacts = super().check_contacts(data)
        geometry = self.check(data, _measured_support=True)
        if not contacts.safe:
            return contacts
        if not geometry.safe:
            return geometry
        if not relation_valid or (not all(sides) and not dropout_grace):
            return CollisionCheckResult(safe=False, minimum_distance_m=0.,
                body1_name=self.binding.target_body_name, pair_class="payload_relation")
        if publish and self.binding._measured_guard is not None:
            guard = self.binding._measured_guard(data)
            if not isinstance(guard, JointConfigurationCheck):
                raise TypeError("Measured guard must return JointConfigurationCheck")
            self.last_measured_details["measured_guard"] = guard.to_dict()
            if not guard.safe or not guard.collision_check_available:
                return CollisionCheckResult(safe=False, minimum_distance_m=0.,
                    body1_name=self.binding.target_body_name, pair_class="payload_measured_guard")
        if publish and all(sides):
            self._last_bilateral_contact_time = float(data.time)
        return geometry


class MujocoPayloadProjector(KinematicCollisionProjector):
    """The existing sweep/project algorithm with only candidate scratch attached."""

    def __init__(self, scene, controlled_qpos_indices, policy, binding):
        binding.validate_scene(scene)
        indices = np.asarray(controlled_qpos_indices)
        scalar = {int(scene.model.jnt_qposadr[j]) for j, kind in enumerate(scene.model.jnt_type)
                  if kind in (scene.mujoco.mjtJoint.mjJNT_HINGE, scene.mujoco.mjtJoint.mjJNT_SLIDE)}
        if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu"
                or len(set(indices.tolist())) != indices.size or not set(indices.tolist()) <= scalar):
            raise ValueError("Payload controlled indices must be distinct scalar arm coordinates")
        super().__init__(scene.mujoco, scene.model, controlled_qpos_indices, policy)
        self.binding = binding
        self._scene = scene
        self.oracle = _PayloadSceneOracle(scene.mujoco, scene.model, policy, binding)
        self._realized_start_qpos = None

    def project(self, source_data, *, start_qpos, requested_qpos, **kwargs):
        """Reuse the canonical sweep, recognizing only its actual source start.

        The runtime separately checks the measured relation/contacts. A tiny
        tracking undershoot at the exact current cut must not invalidate a
        sweep back up to its strict planned reference. All later candidates,
        including downward command endpoints, retain intent checks.
        """
        self.binding._require_actual_cut(self._scene, source_data)
        source_check = self.binding._measured_oracle._check_source_without_publication(source_data)
        if not source_check.safe:
            raise ValueError(f"Payload sweep actual source is unsafe: {source_check.to_dict()}")
        start = np.asarray(start_qpos, dtype=float).reshape(-1)
        if not np.array_equal(start, source_data.qpos[self.controlled_qpos_indices]):
            raise ValueError("Payload sweep must start at the current measured arm cut")
        self._realized_start_qpos = start.copy()
        try:
            return super().project(source_data, start_qpos=start_qpos,
                                   requested_qpos=requested_qpos, **kwargs)
        finally:
            self._realized_start_qpos = None

    def _check_candidate(self, source_data, controlled_qpos, *, minimum_clearance_m=None,
                         distance_query_limit_m=None):
        if self._realized_start_qpos is None:
            self.binding._require_planning_cut(source_data)
        else:
            _require_current_cut(source_data, self.binding._live_data)
        if source_data.model is not self.model:
            raise ValueError("Payload candidate source must belong to the exact model")
        if not np.isfinite(source_data.qpos).all() or not np.isfinite(controlled_qpos).all():
            raise ValueError("Payload candidate/source qpos must be finite")
        self.scratch.qpos[:] = source_data.qpos
        self.scratch.mocap_pos[:] = source_data.mocap_pos
        self.scratch.mocap_quat[:] = source_data.mocap_quat
        self.scratch.eq_active[:] = source_data.eq_active
        self.scratch.qpos[self.controlled_qpos_indices] = controlled_qpos
        self.mujoco.mj_kinematics(self.model, self.scratch)
        tcp = rigid_pose(self.binding._tcp(self.scratch), name="candidate canonical world_from_tcp")
        world_from_object = rigid_pose(tcp @ np.asarray(self.binding.tool_from_object),
                                      name="candidate world_from_object")
        address = self.oracle.object_qpos_address
        self.scratch.qpos[address:address+3] = world_from_object[:3, 3]
        self.mujoco.mju_mat2Quat(self.scratch.qpos[address+3:address+7],
            world_from_object[:3, :3].copy().reshape(-1))
        self.mujoco.mj_kinematics(self.model, self.scratch)
        if not np.isfinite(self.scratch.geom_xpos).all() or not np.isfinite(self.scratch.geom_xmat).all():
            raise ValueError("Payload candidate native world geometry must be finite")
        actual_start = (self._realized_start_qpos is not None
                        and np.array_equal(controlled_qpos, self._realized_start_qpos))
        return self.oracle.check(self.scratch, minimum_clearance_m=minimum_clearance_m,
            distance_query_limit_m=distance_query_limit_m, _measured_support=actual_start)
