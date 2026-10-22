"""Independent Pinocchio/FCL collision oracle for RM65 obstacle planning.

The oracle clones the supplied collision model before adding scene geometry.
It never mutates a live IK solver's model or data and it owns no execution or
hardware authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Any

import numpy as np

from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from robot_motion.observed_obstacles import ObservedObstacleCloud, ObservedBoxDistance
from dataset_contracts.canonical_encoding import canonical_content_hash

_OBSTACLE_GEOMETRY_PREFIX = "aabb_obstacle::"


class RM65InheritedStartBlocked(ValueError):
    """A measured start has actual overlap, not an unavailable query."""

    def __init__(self, *, source: str, distance_m: float, geometry_name: str):
        self.details = {"source": source, "distance_m": float(distance_m),
                        "geometry_name": geometry_name}
        super().__init__("inherited start is in contact or penetration")


@dataclass(frozen=True)
class AttachedBoxCollisionProxy:
    """One conservative box rigidly attached to an existing model frame."""

    geometry_name: str
    parent_frame_name: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    rotation_row_major: tuple[float, ...] = (1., 0., 0., 0., 1., 0., 0., 0., 1.)
    collision_geometry_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.geometry_name.strip() or not self.parent_frame_name.strip():
            raise ValueError("attached collision proxy names must not be empty.")
        center = np.asarray(self.center_xyz_m, dtype=float)
        size = np.asarray(self.size_xyz_m, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("attached collision proxy center must be finite XYZ.")
        if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ValueError(
                "attached collision proxy size must be positive finite XYZ."
            )
        rotation = np.asarray(self.rotation_row_major, dtype=float)
        if (rotation.shape != (9,) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.reshape(3, 3).T @ rotation.reshape(3, 3), np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(rotation.reshape(3, 3)), 1., atol=1e-6, rtol=0)):
            raise ValueError("attached collision proxy requires a rigid rotation")
        if (len(set(self.collision_geometry_names)) != len(self.collision_geometry_names)
                or any(not name or name.strip() != name for name in self.collision_geometry_names)):
            raise ValueError("attached collision pair names must be distinct nonempty names")


# Conservative wrist-camera bounds shared by real URDF planning and MuJoCo
# replay. They are attached to the selected tool assembly, not the bare arm.
RM65_D405_ATTACHED_COLLISION_PROXIES = (
    AttachedBoxCollisionProxy(
        geometry_name="left_d405_safety_proxy",
        parent_frame_name="arm_link6",
        center_xyz_m=(-0.0351, 0.0, 0.02015),
        size_xyz_m=(0.104, 0.103, 0.087),
    ),
    AttachedBoxCollisionProxy(
        geometry_name="right_d405_safety_proxy",
        parent_frame_name="right_arm_link6",
        center_xyz_m=(-0.0351, 0.0, 0.02015),
        size_xyz_m=(0.104, 0.103, 0.087),
    ),
)


def _load_geometry_modules() -> tuple[Any, Any]:
    try:
        import hppfcl  # type: ignore
        import pinocchio as pin  # type: ignore
    except Exception as exc:
        raise RuntimeError("Pinocchio/FCL collision checking is unavailable.") from exc
    return pin, hppfcl


def _default_rm65_component_indices(
    nq: int,
    *,
    solution_layout: str | None,
) -> Mapping[str, tuple[int, ...]]:
    if nq == 12:
        return {"left": tuple(range(6)), "right": tuple(range(6, 12))}
    if nq == 14:
        return {"left": tuple(range(1, 7)), "right": tuple(range(8, 14))}
    if solution_layout == "dual_arm_hand" and nq >= 24:
        return {"left": tuple(range(6)), "right": tuple(range(18, 24))}
    raise ValueError(
        f"cannot infer RM65 arm indices for nq={nq} and "
        f"solution_layout={solution_layout!r}; supply qpos_indices_by_component"
    )


class RM65ObstacleCollisionOracle:
    """Full collision-model oracle augmented with immutable AABB obstacles."""

    def __init__(
        self,
        *,
        model: Any,
        base_collision_model: Any,
        reference_full_q: Sequence[float],
        qpos_indices_by_component: Mapping[str, Sequence[int]],
        scene: ObstacleSceneSnapshot,
        expected_reference_frame_id: str | None = None,
        expected_reference_frame_spec_hash: str | None = None,
        attached_box_proxies: Sequence[AttachedBoxCollisionProxy] = (),
        obstacle_moving_geometry_names: Sequence[str] | None = None,
        stop_at_first_clearance_violation: bool = False,
        collision_scope: str = "full_pinocchio_collision_model",
        pin_module: Any | None = None,
        hppfcl_module: Any | None = None,
    ) -> None:
        if not isinstance(scene, ObstacleSceneSnapshot):
            raise TypeError("scene must be an ObstacleSceneSnapshot.")
        if not isinstance(stop_at_first_clearance_violation, bool):
            raise TypeError(
                "stop_at_first_clearance_violation must be boolean."
            )
        if expected_reference_frame_id is not None and scene.reference_frame_id != str(expected_reference_frame_id):
            raise ValueError("scene reference_frame_id does not match the collision model frame.")
        if expected_reference_frame_spec_hash is not None and scene.reference_frame_spec_hash != str(expected_reference_frame_spec_hash):
            raise ValueError("scene reference_frame_spec_hash does not match the collision model frame contract.")
        pin, hppfcl = (
            _load_geometry_modules()
            if pin_module is None or hppfcl_module is None
            else (pin_module, hppfcl_module)
        )
        reference = np.asarray(reference_full_q, dtype=float)
        nq = int(model.nq)
        if reference.shape != (nq,) or not np.all(np.isfinite(reference)):
            raise ValueError(f"reference_full_q must be finite with shape ({nq},).")
        index_map: dict[str, tuple[int, ...]] = {}
        occupied: set[int] = set()
        for raw_component, raw_indices in qpos_indices_by_component.items():
            component = str(raw_component)
            indices = tuple(int(item) for item in raw_indices)
            if not component or component.strip() != component or not indices:
                raise ValueError("qpos index mappings require named non-empty components.")
            if len(set(indices)) != len(indices) or any(item < 0 or item >= nq for item in indices):
                raise ValueError(f"qpos indices for {component!r} are invalid.")
            overlap = occupied.intersection(indices)
            if overlap:
                raise ValueError(f"qpos index mappings overlap at {sorted(overlap)}.")
            occupied.update(indices)
            index_map[component] = indices
        if not index_map:
            raise ValueError("qpos_indices_by_component must not be empty.")

        collision_model = deepcopy(base_collision_model)
        existing_geometry_names = {
            str(geometry.name) for geometry in collision_model.geometryObjects
        }
        for proxy in attached_box_proxies:
            if not isinstance(proxy, AttachedBoxCollisionProxy):
                raise TypeError(
                    "attached_box_proxies must contain AttachedBoxCollisionProxy values."
                )
            if proxy.geometry_name in existing_geometry_names:
                raise ValueError(
                    f"attached collision proxy geometry {proxy.geometry_name!r} already exists."
                )
            frame_id = int(model.getFrameId(proxy.parent_frame_name))
            if (
                frame_id >= len(model.frames)
                or str(model.frames[frame_id].name) != proxy.parent_frame_name
            ):
                raise ValueError(
                    f"attached collision proxy parent frame {proxy.parent_frame_name!r} "
                    "does not exist."
                )
            frame = model.frames[frame_id]
            placement = frame.placement * pin.SE3(
                np.asarray(proxy.rotation_row_major).reshape(3, 3), np.asarray(proxy.center_xyz_m, dtype=float)
            )
            collision_model.addGeometryObject(
                pin.GeometryObject(
                    proxy.geometry_name,
                    int(frame.parentJoint),
                    frame_id,
                    placement,
                    hppfcl.Box(*proxy.size_xyz_m),
                )
            )
            existing_geometry_names.add(proxy.geometry_name)
        attached_proxy_geometry_count = len(attached_box_proxies)
        base_geometry_count = len(collision_model.geometryObjects)
        moving_geometry_ids = tuple(
            geometry_id
            for geometry_id, geometry in enumerate(collision_model.geometryObjects)
            if int(getattr(geometry, "parentJoint", 0)) != 0
        )
        obstacle_pair_geometry_ids = moving_geometry_ids
        if obstacle_moving_geometry_names is not None:
            requested_names = tuple(
                dict.fromkeys(str(name) for name in obstacle_moving_geometry_names)
            )
            geometry_id_by_name = {
                str(geometry.name): geometry_id
                for geometry_id, geometry in enumerate(
                    collision_model.geometryObjects
                )
            }
            missing = tuple(
                name for name in requested_names if name not in geometry_id_by_name
            )
            if missing:
                raise ValueError(
                    f"obstacle collision geometries do not exist: {missing}"
                )
            obstacle_pair_geometry_ids = tuple(
                geometry_id_by_name[name] for name in requested_names
            )
            non_moving = tuple(
                name
                for name, geometry_id in zip(
                    requested_names, obstacle_pair_geometry_ids, strict=True
                )
                if geometry_id not in moving_geometry_ids
            )
            if non_moving:
                raise ValueError(
                    f"obstacle collision geometries must move with the robot: {non_moving}"
                )
        obstacle_ids: dict[int, str] = {}
        for obstacle in scene.obstacles:
            placement = pin.SE3(np.eye(3), np.asarray(obstacle.center_xyz_m, dtype=float))
            geometry = pin.GeometryObject(
                f"{_OBSTACLE_GEOMETRY_PREFIX}{obstacle.obstacle_id}",
                0,
                0,
                placement,
                hppfcl.Box(*obstacle.size_xyz_m),
            )
            obstacle_geometry_id = int(collision_model.addGeometryObject(geometry))
            obstacle_ids[obstacle_geometry_id] = obstacle.obstacle_id

        existing_pairs = {
            tuple(sorted((int(pair.first), int(pair.second))))
            for pair in collision_model.collisionPairs
        }
        # Carried objects need explicit pairs with robot proxies too, not only
        # the static world. The application binds the reviewed contact pairs;
        # default camera proxies retain their existing collision scope.
        geometry_ids = {str(g.name): i for i, g in enumerate(collision_model.geometryObjects)}
        for proxy in attached_box_proxies:
            for name in proxy.collision_geometry_names:
                if name not in geometry_ids or name == proxy.geometry_name:
                    raise ValueError(f"Invalid attached collision pair: {proxy.geometry_name}, {name}")
                key = tuple(sorted((geometry_ids[proxy.geometry_name], geometry_ids[name])))
                if key not in existing_pairs:
                    collision_model.addCollisionPair(pin.CollisionPair(*key))
                    existing_pairs.add(key)
        for obstacle_geometry_id in obstacle_ids:
            for moving_geometry_id in obstacle_pair_geometry_ids:
                key = tuple(sorted((obstacle_geometry_id, moving_geometry_id)))
                if key not in existing_pairs:
                    collision_model.addCollisionPair(pin.CollisionPair(*key))
                    existing_pairs.add(key)

        self._pin = pin
        self._model = model
        self._model_data = model.createData()
        self._collision_model = collision_model
        self._collision_data = pin.GeometryData(collision_model)
        self._stop_at_first_clearance_violation = (
            stop_at_first_clearance_violation
        )
        self._collision_scope = str(collision_scope)
        self._collision_query_margin_m: float | None = None
        if self._stop_at_first_clearance_violation:
            for request in self._collision_data.collisionRequests:
                request.security_margin = float(scene.minimum_clearance_m)
            self._collision_query_margin_m = float(scene.minimum_clearance_m)
        self._reference_full_q = reference.copy()
        self._indices = MappingProxyType(index_map)
        self._scene = scene
        self._obstacle_ids = MappingProxyType(obstacle_ids)
        self._base_geometry_count = base_geometry_count
        self._attached_proxy_geometry_count = attached_proxy_geometry_count
        self._moving_geometry_ids = moving_geometry_ids
        self._obstacle_pair_geometry_ids = obstacle_pair_geometry_ids
        self._check_cache: dict[bytes, JointConfigurationCheck] = {}
        self._clearance_check_cache: dict[bytes, JointConfigurationCheck] = {}
        self._search_check_cache: dict[tuple[float, bytes], JointConfigurationCheck] = {}
        self._observed_distance: ObservedBoxDistance | None = None
        self._observed_scene_hash: str | None = None

    def bind_observed_cloud(self, cloud: ObservedObstacleCloud, *, minimum_clearance_m: float | None = None) -> None:
        """Bind once, before any query, to this request-owned oracle."""
        if (self._observed_distance is not None or self._check_cache
                or self._clearance_check_cache or self._search_check_cache):
            raise ValueError("Observed cloud must be bound before collision queries.")
        if cloud.reference_frame_id != self._scene.reference_frame_id:
            raise ValueError("Observed cloud and collision model reference frames differ.")
        required = self._scene.minimum_clearance_m if minimum_clearance_m is None else minimum_clearance_m
        if (isinstance(required, bool) or not np.isfinite(required) or required < 0
                or (minimum_clearance_m is not None and required == 0)):
            raise ValueError("Observed minimum clearance must be finite and positive.")
        self.world_obstacle_proxy_boxes({
            name: self._reference_full_q[list(indices)] for name, indices in self._indices.items()})
        self._observed_distance = ObservedBoxDistance(cloud)
        self._observed_minimum_clearance_m = float(required)
        self._observed_scene_hash = canonical_content_hash({
            "fixed_scene": self._scene.semantic_content_hash,
            "observed_scene": cloud.semantic_content_hash,
            **({"observed_minimum_clearance_m": float(required)} if minimum_clearance_m is not None else {}),
        }, domain="dexteleop.rm65.combined_obstacle_scene.v1")

    def for_observed_cloud(self, cloud: ObservedObstacleCloud, *, minimum_clearance_m: float | None = None) -> RM65ObstacleCollisionOracle:
        """Reuse static models with new request-owned FK, queries and obstacles."""
        if self._observed_distance is not None:
            raise ValueError("Observed requests must originate from a static oracle.")
        request = copy(self)
        request._model_data = self._model.createData()
        request._collision_data = self._pin.GeometryData(self._collision_model)
        request._collision_query_margin_m = None
        if self._stop_at_first_clearance_violation:
            for query in request._collision_data.collisionRequests:
                query.security_margin = float(self._scene.minimum_clearance_m)
            request._collision_query_margin_m = float(self._scene.minimum_clearance_m)
        request._check_cache = {}
        request._clearance_check_cache = {}
        request._search_check_cache = {}
        request.bind_observed_cloud(cloud, minimum_clearance_m=minimum_clearance_m)
        return request

    @classmethod
    def from_solver(
        cls,
        solver: Any,
        *,
        scene: ObstacleSceneSnapshot,
        qpos_indices_by_component: Mapping[str, Sequence[int]] | None = None,
        solution_layout: str | None = None,
        expected_reference_frame_id: str | None = None,
        expected_reference_frame_spec_hash: str | None = None,
        attached_box_proxies: Sequence[AttachedBoxCollisionProxy] = (),
        obstacle_moving_geometry_names: Sequence[str] | None = None,
        stop_at_first_clearance_violation: bool = False,
        collision_scope: str = "full_pinocchio_collision_model",
    ) -> RM65ObstacleCollisionOracle:
        if (
            expected_reference_frame_id is None
            or not str(expected_reference_frame_id).strip()
            or expected_reference_frame_spec_hash is None
            or not str(expected_reference_frame_spec_hash).strip()
        ):
            raise ValueError(
                "from_solver requires expected_reference_frame_id and "
                "expected_reference_frame_spec_hash from a reviewed frame contract."
            )
        robot = getattr(solver, "robot", None)
        if robot is None:
            raise ValueError("solver does not expose robot model/collision_model.")
        model = getattr(robot, "model", None)
        collision_model = getattr(robot, "collision_model", None)
        if model is None or collision_model is None:
            raise ValueError("solver robot lacks a model or collision_model.")
        reference = np.asarray(getattr(solver, "current_q", ()), dtype=float)
        if reference.shape != (int(model.nq),):
            raise ValueError("solver current_q does not match its kinematic model.")
        indices = (
            _default_rm65_component_indices(
                int(model.nq),
                solution_layout=solution_layout,
            )
            if qpos_indices_by_component is None
            else qpos_indices_by_component
        )
        return cls(
            model=model,
            base_collision_model=collision_model,
            reference_full_q=reference,
            qpos_indices_by_component=indices,
            scene=scene,
            expected_reference_frame_id=expected_reference_frame_id,
            expected_reference_frame_spec_hash=expected_reference_frame_spec_hash,
            attached_box_proxies=attached_box_proxies,
            obstacle_moving_geometry_names=obstacle_moving_geometry_names,
            stop_at_first_clearance_violation=(
                stop_at_first_clearance_violation
            ),
            collision_scope=collision_scope,
        )

    @property
    def scene_hash(self) -> str:
        if self._observed_distance is not None:
            return self._observed_scene_hash
        return self._scene.semantic_content_hash

    def world_obstacle_proxy_boxes(self, qpos_by_component):
        """Query the reviewed obstacle proxies, independent of cached checks."""
        full_q = self._embed(qpos_by_component)
        self._pin.updateGeometryPlacements(
            self._model, self._model_data, self._collision_model, self._collision_data, full_q)
        boxes = []
        for index in self._obstacle_pair_geometry_ids:
            geometry = self._collision_model.geometryObjects[index]
            half = getattr(geometry.geometry, "halfSide", None)
            if half is None:
                raise ValueError("Observed obstacle coverage requires explicit sparse box proxies.")
            pose = self._collision_data.oMg[index]
            transform = np.eye(4)
            transform[:3, :3] = pose.rotation
            transform[:3, 3] = pose.translation
            boxes.append((str(geometry.name), transform, np.asarray(half).copy()))
        if not boxes:
            raise ValueError("Observed obstacle proxy set is empty.")
        return tuple(boxes)

    def _with_observed(self, qpos, result, *, distance_bound_m):
        if self._observed_distance is None or not result.safe:
            return result
        cloud = self._observed_distance.cloud
        try:
            distance, nearest = self._observed_distance.check_boxes(
                self.world_obstacle_proxy_boxes(qpos), clearance_m=max(distance_bound_m, self._observed_minimum_clearance_m))
        except (ValueError, RuntimeError) as error:
            return JointConfigurationCheck(safe=False, collision_check_available=False,
                reason="observed_obstacle_collision_check_unavailable", metadata={"error": str(error)})
        safe = distance >= self._observed_minimum_clearance_m
        closest = {}
        if nearest is not None and (result.minimum_distance_m is None or distance < result.minimum_distance_m):
            closest = {"closest_geometry_names": (nearest["geometry_name"], "observed_voxels"),
                       "closest_obstacle_id": "observed-rgbd:" + cloud.semantic_content_hash}
        return replace(result, safe=safe, reason=result.reason if safe else "observed_obstacle_clearance_blocked",
            minimum_distance_m=min(distance, result.minimum_distance_m)
            if result.minimum_distance_m is not None else distance,
            metadata={**result.metadata, **closest, "scene_hash": self.scene_hash,
                      "observed_scene_hash": cloud.semantic_content_hash,
                      "observed_minimum_clearance_m": self._observed_minimum_clearance_m,
                      "observed_scope": "occupied_voxels_against_profile_obstacle_box_proxies",
                      "observed_distance_is_lower_bound": True, "nearest_observed": nearest})

    @property
    def collision_model_is_independent(self) -> bool:
        return True

    def check_rigid_endpoint_pose(
        self, qpos_by_component: Mapping[str, np.ndarray], *, active_component: str,
        current_tool_tcp_world: np.ndarray, target_tool_tcp_world: np.ndarray,
    ) -> JointConfigurationCheck:
        """Advisory terminal-tool clearance without solving arm IK.

        Move only the terminal joint's rigid subtree to the requested TCP pose.
        Query existing pairs against fixed geometry and the inactive component,
        in independent scratch data. Forearm geometry, pose tolerances and path
        continuity are deliberately NOT certified by this query. A result may
        order IK requests, never admit or remove a grasp candidate.
        """
        metadata = {"scene_hash": self.scene_hash, "active_component": active_component,
            "query_mode": "advisory_rigid_endpoint_vs_static_pairs",
            "admission_authority": False}
        try:
            full_q = self._embed(qpos_by_component)
            active_indices = set(self._indices[active_component])
        except (KeyError, TypeError, ValueError) as exc:
            return JointConfigurationCheck(safe=False, collision_check_available=False,
                reason="invalid_rigid_endpoint_query", metadata={**metadata, "error": str(exc)})
        poses = [np.asarray(p, dtype=float) for p in (current_tool_tcp_world, target_tool_tcp_world)]
        for pose in poses:
            if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                    or not np.allclose(pose[3], (0., 0., 0., 1.), atol=1e-9, rtol=0.)
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5, rtol=0.)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5, rtol=0.)):
                raise ValueError("Rigid endpoint query requires two finite rigid world poses.")

        def ancestors(joint_id):
            result = set()
            while joint_id:
                result.add(joint_id)
                joint_id = int(self._model.parents[joint_id])
            return result

        active_joints = {i for i, j in enumerate(self._model.joints)
            if i and active_indices.intersection(range(j.idx_q, j.idx_q + j.nq))}
        terminal_joints = active_joints - set().union(
            *(ancestors(int(self._model.parents[j])) for j in active_joints))
        if len(terminal_joints) != 1:
            return JointConfigurationCheck(safe=False, collision_check_available=False,
                reason="rigid_endpoint_requires_serial_component", metadata=metadata)
        terminal_joint = next(iter(terminal_joints))
        ancestry = [ancestors(int(g.parentJoint)) for g in self._collision_model.geometryObjects]
        rigid = {i for i, parents in enumerate(ancestry) if terminal_joint in parents}
        affected = {i for i, parents in enumerate(ancestry) if active_joints.intersection(parents)}
        pairs = tuple(i for i, p in enumerate(self._collision_model.collisionPairs)
            if (int(p.first) in rigid and int(p.second) not in affected)
            or (int(p.second) in rigid and int(p.first) not in affected))
        if not pairs:
            return JointConfigurationCheck(safe=False, collision_check_available=False,
                reason="no_rigid_endpoint_static_pairs", metadata=metadata)
        try:
            # Do not alter ordinary FK/GeometryData or any qpos-keyed cache.
            data = self._model.createData()
            geometry_data = self._pin.GeometryData(self._collision_model)
            self._pin.updateGeometryPlacements(self._model, data, self._collision_model, geometry_data, full_q)
            delta = poses[1] @ np.linalg.inv(poses[0])
            transform = self._pin.SE3(delta[:3, :3], delta[:3, 3])
            for i in rigid:
                geometry_data.oMg[i] = transform * geometry_data.oMg[i]
            distances = [float(self._pin.computeDistance(self._collision_model, geometry_data, i).min_distance)
                for i in pairs]
        except Exception as exc:  # noqa: BLE001 - optional external geometry query is unavailable
            return JointConfigurationCheck(safe=False, collision_check_available=False,
                reason="rigid_endpoint_distance_query_failed",
                metadata={**metadata, "exception_type": type(exc).__name__, "exception": str(exc)})
        if not np.isfinite(distances).all():
            return JointConfigurationCheck(safe=False, collision_check_available=False,
                reason="rigid_endpoint_distance_invalid", metadata=metadata)
        closest = int(np.argmin(distances))
        pair = self._collision_model.collisionPairs[pairs[closest]]
        distance = distances[closest]
        return JointConfigurationCheck(safe=distance > self._scene.minimum_clearance_m,
            collision_check_available=True, reason="advisory_rigid_endpoint_clearance",
            minimum_distance_m=distance, metadata={**metadata,
                "rigid_geometry_count": len(rigid), "queried_pair_count": len(pairs),
                "minimum_clearance_m": self._scene.minimum_clearance_m,
                "closest_geometry_names": tuple(str(self._collision_model.geometryObjects[i].name)
                    for i in (int(pair.first), int(pair.second)))})

    def check(self, qpos_by_component: Mapping[str, np.ndarray]) -> JointConfigurationCheck:
        try:
            full_q = self._embed(qpos_by_component)
        except (TypeError, ValueError) as exc:
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="invalid_collision_query",
                metadata={"error": str(exc)},
            )
        key = full_q.tobytes()
        if key not in self._check_cache:
            self._check_cache[key] = self._with_observed(qpos_by_component,
                self._check_embedded(full_q), distance_bound_m=self._scene.minimum_clearance_m)
        return self._check_cache[key]

    def check_static_scene(
        self, qpos_by_component: Mapping[str, np.ndarray], *, active_component: str,
    ) -> JointConfigurationCheck:
        """Check pairs whose placement the selected arm cannot change.

        This is an early scene-readiness check, not a collision exemption.
        Ordinary endpoint and path queries still check every configured pair.
        Dependencies come from joint ancestry, never geometry-name prefixes.
        """
        try:
            full_q = self._embed(qpos_by_component)
            active_indices = set(self._indices[active_component])
        except (KeyError, TypeError, ValueError) as exc:
            return JointConfigurationCheck(safe=False, collision_check_available=False,
                reason="invalid_static_collision_query", metadata={"error": str(exc)})

        def affected(geometry_id: int) -> bool:
            joint_id = int(self._collision_model.geometryObjects[geometry_id].parentJoint)
            while joint_id:
                joint = self._model.joints[joint_id]
                if active_indices.intersection(range(joint.idx_q, joint.idx_q + joint.nq)):
                    return True
                joint_id = int(self._model.parents[joint_id])
            return False

        pairs = tuple(
            i for i, pair in enumerate(self._collision_model.collisionPairs)
            if not affected(int(pair.first)) and not affected(int(pair.second))
        )
        if not pairs:
            return JointConfigurationCheck(safe=True, collision_check_available=True,
                reason="no_static_collision_pairs", metadata={"active_component": active_component})
        check = self._check_embedded_distances(full_q,
            pair_count=len(self._collision_model.collisionPairs), pair_indices=pairs)
        return replace(check, metadata={**check.metadata,
            "query_mode": "inactive_component_minimum_distance",
            "active_component": active_component, "static_pair_count": len(pairs)})

    def inherited_start_clearance(self, qpos_by_component, *, active_component: str):
        """Bind a request-local, positive-distance start allowance for safe egress.

        This does not alter this oracle, the scene margin, or its collision
        pairs. The returned oracle still checks every pair and the observed
        cloud. Its moving-pair progress is separately certified on the dense
        final trajectory by the caller.
        """
        return RM65InheritedStartClearanceOracle(
            self, qpos_by_component=qpos_by_component,
            active_component=active_component)


    def check_clearance_only(
        self, qpos_by_component: Mapping[str, np.ndarray]
    ) -> JointConfigurationCheck:
        """Check the full collision model at the scene's clearance threshold.

        A safe fast result proves the requested clearance but reports only a
        lower distance bound. Borderline or blocked guard results are resolved
        with an exact query so a conservative guard cannot reject a safe path.
        """

        try:
            full_q = self._embed(qpos_by_component)
        except (TypeError, ValueError):
            return self.check(qpos_by_component)
        key = full_q.tobytes()
        if key not in self._clearance_check_cache:
            pair_count = len(self._collision_model.collisionPairs)
            if not self._moving_geometry_ids or pair_count == 0:
                result = self._check_embedded(full_q)
            else:
                result = self._check_embedded_early_exit(
                    full_q,
                    pair_count=pair_count,
                    clearance_m=self._scene.minimum_clearance_m,
                )
                if result.collision_check_available and not result.safe:
                    result = self.check(qpos_by_component)
            self._clearance_check_cache[key] = self._with_observed(
                qpos_by_component, result, distance_bound_m=self._scene.minimum_clearance_m)
        return self._clearance_check_cache[key]

    def _check_embedded(self, full_q: np.ndarray) -> JointConfigurationCheck:
        pair_count = len(self._collision_model.collisionPairs)
        if not self._moving_geometry_ids:
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="collision_model_has_no_moving_geometry",
                metadata=self._base_metadata(pair_count),
            )
        if pair_count == 0:
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="collision_model_has_no_collision_pairs",
                metadata=self._base_metadata(pair_count),
            )
        if self._stop_at_first_clearance_violation:
            return self._check_embedded_early_exit(
                full_q, pair_count=pair_count, clearance_m=self._scene.minimum_clearance_m,
            )
        return self._check_embedded_distances(full_q, pair_count=pair_count)

    def check_for_search(
        self, qpos_by_component: Mapping[str, np.ndarray], *, refinement_distance_m: float,
    ) -> JointConfigurationCheck:
        """Prove clearance beyond refinement, computing exact distances near it.

        A broad-phase lower bound at the safety margin alone can needlessly
        trigger fine edge sampling. Query at the refinement boundary instead;
        a failed guard is not a rejection until the exact safety query runs.
        Search lower bounds never enter the ordinary check cache.
        """
        refinement = float(refinement_distance_m)
        if not np.isfinite(refinement) or refinement < 0.0:
            raise ValueError("refinement_distance_m must be finite and nonnegative.")
        try:
            full_q = self._embed(qpos_by_component)
        except (TypeError, ValueError):
            return self.check(qpos_by_component)
        margin = max(float(self._scene.minimum_clearance_m), refinement)
        key = (margin, full_q.tobytes())
        if key not in self._search_check_cache:
            pair_count = len(self._collision_model.collisionPairs)
            if not self._moving_geometry_ids or pair_count == 0:
                return self._check_embedded(full_q)
            guard = self._check_embedded_early_exit(
                full_q, pair_count=pair_count, clearance_m=margin,
            )
            if guard.collision_check_available and (
                not guard.safe or guard.minimum_distance_m <= margin
            ):
                # The 20--30 mm band remains safe but must refine normally.
                guard = (
                    self._check_embedded_distances(full_q, pair_count=pair_count)
                    if self._stop_at_first_clearance_violation
                    else self.check(qpos_by_component)
                )
            guard = self._with_observed(qpos_by_component, guard, distance_bound_m=margin)
            self._search_check_cache[key] = replace(
                guard, metadata={**guard.metadata, "search_guard_clearance_m": margin},
            )
        return self._search_check_cache[key]

    def _check_embedded_distances(
        self, full_q: np.ndarray, *, pair_count: int,
        pair_indices: tuple[int, ...] | None = None,
    ) -> JointConfigurationCheck:
        metadata = {**self._base_metadata(pair_count), "query_mode": "all_pair_minimum_distance"}
        try:
            model_data, collision_data = self._model_data, self._collision_data
            if pair_indices is not None:
                # Readiness subsets may overlap ordinary cached queries.
                # Keep mutable native buffers private through result extraction.
                model_data = self._model.createData()
                collision_data = self._pin.GeometryData(self._collision_model)
            self._pin.computeDistances(
                self._model,
                model_data,
                self._collision_model,
                collision_data,
                full_q,
            )
            distance_results = tuple(collision_data.distanceResults)
        except Exception as exc:  # noqa: BLE001 - external collision query fails closed
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="pinocchio_fcl_distance_query_failed",
                metadata={
                    **metadata,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                },
            )
        if len(distance_results) != pair_count:
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="collision_distance_result_count_mismatch",
                metadata={
                    **metadata,
                    "distance_result_count": len(distance_results),
                },
            )
        distances = np.asarray(
            [float(result.min_distance) for result in distance_results], dtype=float
        )
        if distances.size == 0 or not np.all(np.isfinite(distances)):
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="collision_distance_invalid",
                metadata=metadata,
            )
        if pair_indices is None:
            closest_index = int(np.argmin(distances))
        else:
            closest_index = pair_indices[int(np.argmin(distances[list(pair_indices)]))]
        closest_pair = self._collision_model.collisionPairs[closest_index]
        first_id = int(closest_pair.first)
        second_id = int(closest_pair.second)
        first_name = str(self._collision_model.geometryObjects[first_id].name)
        second_name = str(self._collision_model.geometryObjects[second_id].name)
        obstacle_id = self._obstacle_ids.get(first_id, self._obstacle_ids.get(second_id))
        minimum_distance = float(distances[closest_index])
        # Contact at exactly the requested clearance is blocked.  This keeps a
        # solid AABB boundary fail-closed even when the requested margin is 0.
        safe = minimum_distance > self._scene.minimum_clearance_m
        return JointConfigurationCheck(
            safe=safe,
            collision_check_available=True,
            reason="collision_clearance_satisfied" if safe else "collision_clearance_violated",
            minimum_distance_m=minimum_distance,
            metadata={
                **metadata,
                "closest_geometry_names": (first_name, second_name),
                "closest_obstacle_id": obstacle_id,
            },
        )

    def _check_embedded_early_exit(
        self,
        full_q: np.ndarray,
        *,
        pair_count: int,
        clearance_m: float,
    ) -> JointConfigurationCheck:
        metadata = {
            **self._base_metadata(pair_count),
            "query_mode": "first_clearance_violation",
            "distance_is_lower_bound": True,
            "query_clearance_m": clearance_m,
        }
        try:
            if self._collision_query_margin_m != clearance_m:
                for request in self._collision_data.collisionRequests:
                    request.security_margin = float(clearance_m)
                self._collision_query_margin_m = float(clearance_m)
            blocked = bool(
                self._pin.computeCollisions(
                    self._model,
                    self._model_data,
                    self._collision_model,
                    self._collision_data,
                    full_q,
                    True,
                )
            )
            collision_results = tuple(self._collision_data.collisionResults)
        except Exception as exc:  # noqa: BLE001 - external query fails closed
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="pinocchio_fcl_collision_query_failed",
                metadata={
                    **metadata,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                },
            )
        if len(collision_results) != pair_count:
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="collision_result_count_mismatch",
                metadata={
                    **metadata,
                    "collision_result_count": len(collision_results),
                },
            )
        if blocked:
            result_index = next(
                (
                    index
                    for index, result in enumerate(collision_results)
                    if result.isCollision()
                ),
                None,
            )
            if result_index is None:
                return JointConfigurationCheck(
                    safe=False,
                    collision_check_available=False,
                    reason="collision_result_missing_blocking_pair",
                    metadata=metadata,
                )
        else:
            lower_bounds = np.asarray(
                [float(result.distance_lower_bound) for result in collision_results],
                dtype=float,
            )
            if lower_bounds.size == 0 or not np.all(np.isfinite(lower_bounds)):
                return JointConfigurationCheck(
                    safe=False,
                    collision_check_available=False,
                    reason="collision_distance_lower_bound_invalid",
                    metadata=metadata,
                )
            result_index = int(np.argmin(lower_bounds))
        pair = self._collision_model.collisionPairs[result_index]
        first_id = int(pair.first)
        second_id = int(pair.second)
        result = collision_results[result_index]
        minimum_distance = float(result.distance_lower_bound) + float(clearance_m)
        return JointConfigurationCheck(
            safe=not blocked,
            collision_check_available=True,
            reason=(
                "collision_clearance_satisfied"
                if not blocked
                else "collision_clearance_violated"
            ),
            minimum_distance_m=(
                minimum_distance if np.isfinite(minimum_distance) else None
            ),
            metadata={
                **metadata,
                "closest_geometry_names": (
                    str(self._collision_model.geometryObjects[first_id].name),
                    str(self._collision_model.geometryObjects[second_id].name),
                ),
                "closest_obstacle_id": self._obstacle_ids.get(
                    first_id, self._obstacle_ids.get(second_id)
                ),
            },
        )

    def _embed(self, qpos_by_component: Mapping[str, np.ndarray]) -> np.ndarray:
        if set(qpos_by_component) != set(self._indices):
            raise ValueError(
                "collision query components must exactly match "
                f"{tuple(self._indices)}"
            )
        full_q = self._reference_full_q.copy()
        for component, indices in self._indices.items():
            values = np.asarray(qpos_by_component[component], dtype=float)
            if values.shape != (len(indices),) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"collision query component {component!r} must have finite shape "
                    f"({len(indices)},)"
                )
            full_q[np.asarray(indices, dtype=int)] = values
        return full_q

    def _base_metadata(self, pair_count: int) -> dict[str, Any]:
        return {
            "scene_hash": self.scene_hash,
            "scene_revision": self._scene.revision,
            "collision_model_is_independent": True,
            "collision_scope": self._collision_scope,
            "query_mode": (
                "first_clearance_violation"
                if self._stop_at_first_clearance_violation
                else "all_pair_minimum_distance"
            ),
            "base_geometry_count": self._base_geometry_count,
            "attached_proxy_geometry_count": self._attached_proxy_geometry_count,
            "obstacle_geometry_count": len(self._obstacle_ids),
            "moving_geometry_count": len(self._moving_geometry_ids),
            "obstacle_pair_geometry_count": len(
                self._obstacle_pair_geometry_ids
            ),
            "collision_pair_count": int(pair_count),
            "minimum_clearance_m": self._scene.minimum_clearance_m,
        }


class RM65StartClearanceProgress:
    """Ordered dense-path proof for inherited active collision pairs."""

    def __init__(self, *, pair_indices: tuple[int | str, ...], initial_distances: tuple[float, ...],
                 ordinary_clearance_m: float, tolerance_m: float):
        self.pair_indices = pair_indices
        self.previous = list(initial_distances)
        self.exited = [False] * len(pair_indices)
        self.ordinary_clearance_m = ordinary_clearance_m
        self.tolerance_m = tolerance_m

    def advance(self, distances: tuple[float, ...]) -> dict[str, object] | None:
        if len(distances) != len(self.previous):
            return {"reason": "inherited_start_clearance_distance_count_changed"}
        for index, distance in enumerate(distances):
            previous = self.previous[index]
            if (not np.isfinite(distance) or distance <= 0.0
                    or (self.exited[index] and distance <= self.ordinary_clearance_m)
                    or (not self.exited[index] and distance < previous - self.tolerance_m)):
                return {"reason": "inherited_start_clearance_regressed",
                    "inherited_pair_index": self.pair_indices[index],
                    "previous_distance_m": previous, "distance_m": distance}
            # Compare to the high-water mark. A per-step tolerance must not
            # accumulate into a centimetre-scale retreat over many samples.
            self.previous[index] = max(previous, distance)
            if distance > self.ordinary_clearance_m:
                self.exited[index] = True
        return None

    @property
    def all_exited(self) -> bool:
        return all(self.exited)


class RM65InheritedStartClearanceOracle:
    """Request-owned clearance oracle for an already non-contact start state.

    Only fixed collision pairs below the ordinary margin at the exact start
    receive their measured positive-distance floor. All other pairs keep the
    original margin. The inactive component must remain at the measured qpos.
    A dense, ordered path check must additionally prove that active near pairs
    move monotonically out of the margin and never re-enter it.
    """

    _DISTANCE_TOLERANCE_M = 1e-6

    def __init__(self, source: RM65ObstacleCollisionOracle, *,
                 qpos_by_component: Mapping[str, np.ndarray], active_component: str):
        if not isinstance(source, RM65ObstacleCollisionOracle):
            raise TypeError("Inherited start clearance requires the native RM65 oracle.")
        if active_component not in source._indices:
            raise ValueError("Active component is absent from the collision model.")
        self._source = source
        self._active_component = active_component
        self._start = {name: np.asarray(values, dtype=float).copy()
                       for name, values in qpos_by_component.items()}
        source._embed(self._start)
        self._model_data = source._model.createData()
        self._collision_data = source._pin.GeometryData(source._collision_model)
        self._margin = float(source._scene.minimum_clearance_m)
        # Observed voxels use the request's observed clearance (explicit or the
        # fixed-scene margin); fixed pairs keep the ordinary scene margin.
        self._observed_margin = (float(source._observed_minimum_clearance_m)
                                 if source._observed_distance is not None else self._margin)
        self._baseline = self._distances(self._start)
        if np.any(self._baseline <= 0.0):
            index = int(np.argmin(self._baseline))
            pair = source._collision_model.collisionPairs[index]
            names = tuple(str(source._collision_model.geometryObjects[part].name)
                          for part in (int(pair.first), int(pair.second)))
            raise RM65InheritedStartBlocked(source="fixed_scene",
                distance_m=float(self._baseline[index]), geometry_name=" / ".join(names))
        self._inherited = tuple(int(index) for index in np.flatnonzero(self._baseline <= self._margin))
        active_indices = set(source._indices[active_component])

        def affected(geometry_id: int) -> bool:
            joint_id = int(source._collision_model.geometryObjects[geometry_id].parentJoint)
            while joint_id:
                joint = source._model.joints[joint_id]
                if active_indices.intersection(range(joint.idx_q, joint.idx_q + joint.nq)):
                    return True
                joint_id = int(source._model.parents[joint_id])
            return False

        self.active_inherited_pairs = tuple(index for index in self._inherited
            if affected(int(source._collision_model.collisionPairs[index].first))
            or affected(int(source._collision_model.collisionPairs[index].second)))
        self.static_inherited_pairs = tuple(index for index in self._inherited
            if index not in self.active_inherited_pairs)
        self._observed_baseline = self._observed_distances(self._start)
        for name, (distance, _) in self._observed_baseline.items():
            if not np.isfinite(distance):
                raise RuntimeError("Inherited observed distance is nonfinite.")
            if distance <= 0.0:
                raise RM65InheritedStartBlocked(source="observed_rgbd",
                    distance_m=distance, geometry_name=name)
        self._observed_inherited = tuple(name for name, (distance, _) in self._observed_baseline.items()
            if distance <= self._observed_margin)
        if not self._inherited and not self._observed_inherited:
            raise ValueError("The start already satisfies the ordinary clearance margin.")
        geometry_by_name = {str(geometry.name): index
            for index, geometry in enumerate(source._collision_model.geometryObjects)}
        self.observed_active_names = tuple(name for name in self._observed_inherited
            if affected(geometry_by_name[name]))
        self.observed_static_names = tuple(name for name in self._observed_inherited
            if name not in self.observed_active_names)
        self.scene_hash = canonical_content_hash({
            "scene_hash": source.scene_hash,
            "active_component": active_component,
            "start_qpos": self._start,
            "inherited_pairs": [(index, float(self._baseline[index])) for index in self._inherited],
            "inherited_observed_boxes": [(name, float(self._observed_baseline[name][0]))
                for name in self._observed_inherited],
            "ordinary_margin_m": self._margin,
            **({} if self._observed_margin == self._margin
               else {"observed_margin_m": self._observed_margin}),
        }, domain="dexteleop.rm65.inherited_start_clearance.v1")

    @property
    def ordinary_clearance_m(self) -> float:
        return self._margin

    def _distances(self, qpos_by_component: Mapping[str, np.ndarray]) -> np.ndarray:
        source = self._source
        full_q = source._embed(qpos_by_component)
        source._pin.computeDistances(source._model, self._model_data,
            source._collision_model, self._collision_data, full_q)
        results = tuple(self._collision_data.distanceResults)
        if len(results) != len(source._collision_model.collisionPairs):
            raise RuntimeError("Inherited clearance distance result count mismatch.")
        values = np.asarray([float(item.min_distance) for item in results], dtype=float)
        if not values.size or not np.isfinite(values).all():
            raise RuntimeError("Inherited clearance distance query is unavailable.")
        return values

    def inherited_active_distances(self, qpos_by_component) -> tuple[float, ...]:
        """Exact ordered-pair distances for dense egress progress validation."""
        values = self._distances(qpos_by_component)
        return tuple(float(values[index]) for index in self.active_inherited_pairs)

    def observed_active_distances(self, qpos_by_component) -> tuple[float, ...]:
        values = self._observed_distances(qpos_by_component)
        return tuple(values[name][0] for name in self.observed_active_names)

    def _observed_distances(self, qpos_by_component):
        distance_query = self._source._observed_distance
        if distance_query is None:
            return {}
        boxes = self._source.world_obstacle_proxy_boxes(qpos_by_component)
        values = {}
        for box in boxes:
            name = box[0]
            if name in values:
                raise RuntimeError("Observed proxy names must be unique.")
            values[name] = distance_query.check_boxes((box,), clearance_m=self._observed_margin)
        return values

    def new_dense_progress(self) -> RM65StartClearanceProgress:
        return RM65StartClearanceProgress(
            pair_indices=self.active_inherited_pairs,
            initial_distances=tuple(float(self._baseline[index])
                for index in self.active_inherited_pairs),
            ordinary_clearance_m=self._margin,
            tolerance_m=self._DISTANCE_TOLERANCE_M)

    def new_observed_dense_progress(self) -> RM65StartClearanceProgress:
        return RM65StartClearanceProgress(
            pair_indices=self.observed_active_names,
            initial_distances=tuple(self._observed_baseline[name][0]
                for name in self.observed_active_names),
            ordinary_clearance_m=self._observed_margin,
            tolerance_m=self._DISTANCE_TOLERANCE_M)

    def check(self, qpos_by_component: Mapping[str, np.ndarray]) -> JointConfigurationCheck:
        if any(not np.array_equal(np.asarray(qpos_by_component.get(name)), start)
               for name, start in self._start.items() if name != self._active_component):
            return JointConfigurationCheck(safe=False, collision_check_available=False,
                reason="inherited_clearance_inactive_component_changed")
        try:
            distances = self._distances(qpos_by_component)
        except Exception as error:  # noqa: BLE001 - native distance faults fail closed
            return JointConfigurationCheck(safe=False, collision_check_available=False,
                reason="inherited_clearance_collision_check_unavailable",
                metadata={"exception_type": type(error).__name__})
        thresholds = np.full(len(distances), self._margin, dtype=float)
        thresholds[list(self._inherited)] = np.maximum(
            0.0, self._baseline[list(self._inherited)] - self._DISTANCE_TOLERANCE_M)
        violated = np.flatnonzero((distances <= 0.0) | (distances <= thresholds))
        index = (int(violated[0]) if violated.size else int(np.argmin(distances)))
        pair = self._source._collision_model.collisionPairs[index]
        names = tuple(str(self._source._collision_model.geometryObjects[part].name)
                      for part in (int(pair.first), int(pair.second)))
        observed = {}
        if not violated.size:
            try:
                observed = self._observed_distances(qpos_by_component)
            except Exception as error:  # noqa: BLE001 - observed geometry faults fail closed
                return JointConfigurationCheck(safe=False, collision_check_available=False,
                    reason="inherited_observed_collision_check_unavailable",
                    metadata={"exception_type": type(error).__name__})
            if set(observed) != set(self._observed_baseline):
                return JointConfigurationCheck(safe=False, collision_check_available=False,
                    reason="inherited_observed_proxy_set_changed")
        observed_blocked = None
        for name, (distance, nearest) in observed.items():
            baseline = self._observed_baseline.get(name)
            if baseline is None:
                return JointConfigurationCheck(safe=False, collision_check_available=False,
                    reason="inherited_observed_proxy_set_changed")
            threshold = (max(0.0, baseline[0] - self._DISTANCE_TOLERANCE_M)
                         if name in self._observed_inherited else self._observed_margin)
            if distance <= 0.0 or distance <= threshold:
                observed_blocked = (name, distance, nearest)
                break
        result = JointConfigurationCheck(
            safe=not bool(violated.size) and observed_blocked is None,
            collision_check_available=True,
            reason="inherited_start_clearance_satisfied" if not violated.size
                   and observed_blocked is None else
                   "observed_obstacle_clearance_blocked" if observed_blocked is not None
                   else "inherited_start_clearance_blocked",
            minimum_distance_m=min([float(np.min(distances))]
                + [distance for distance, _ in observed.values()]),
            metadata={"scene_hash": self.scene_hash,
                      "closest_geometry_names": (
                          (observed_blocked[0], "observed_voxels")
                          if observed_blocked is not None else names),
                      "blocking_pair_index": index if violated.size else None,
                      "inherited_pair_count": len(self._inherited),
                      "active_inherited_pair_count": len(self.active_inherited_pairs),
                      "static_inherited_pair_count": len(self.static_inherited_pairs),
                      "observed_inherited_box_count": len(self._observed_inherited),
                      "observed_active_box_count": len(self.observed_active_names),
                      "nearest_observed": None if observed_blocked is None else observed_blocked[2],
                      "ordinary_clearance_m": self._margin,
                      "observed_clearance_m": self._observed_margin})
        return result

    def check_clearance_only(self, qpos_by_component):
        return self.check(qpos_by_component)

    def check_rigid_endpoint_pose(self, *args, **kwargs):
        return self._source.check_rigid_endpoint_pose(*args, **kwargs)


@lru_cache(maxsize=2)
def _profile_collision_solver_bundle(
    scene_profile: str, model_content_hash: str | None = None,
    validate_model_resources: Callable[[], None] | None = None,
):
    """Load one immutable source model for independent obstacle oracles."""

    from .ik import (
        RM65DualArmCollisionIKConfig,
        build_rm65_dual_arm_collision_ik_solver,
    )

    if validate_model_resources is not None:
        validate_model_resources()
    bundle = build_rm65_dual_arm_collision_ik_solver(
        RM65DualArmCollisionIKConfig(scene_profile=scene_profile)
    )
    # An exception here must escape the cached function, before lru_cache can
    # publish a model built while its verified resources were changing.
    if validate_model_resources is not None:
        validate_model_resources()
    return bundle


def build_rm65_profile_obstacle_collision_oracle(
    *,
    scene_profile: str,
    scene: ObstacleSceneSnapshot,
    expected_reference_frame_id: str,
    expected_reference_frame_spec_hash: str,
    attached_box_proxies: Sequence[AttachedBoxCollisionProxy] = (),
    use_profile_proxy_geometry: bool = False,
    stop_at_first_clearance_violation: bool = False,
    model_content_hash: str | None = None,
    validate_model_resources: Callable[[], None] | None = None,
) -> RM65ObstacleCollisionOracle:
    """Build an independent obstacle oracle from an explicit RM65 profile.

    Verified resource identities require a validator inside the cache miss
    boundary. Failed materialization cannot publish a model under that identity.
    """

    if model_content_hash is not None:
        if (not isinstance(model_content_hash, str) or len(model_content_hash) != 64
                or any(c not in "0123456789abcdef" for c in model_content_hash)):
            raise ValueError("model_content_hash must be a verified resource SHA-256.")
        if not callable(validate_model_resources):
            raise ValueError("Verified model resources require a validation callback.")
        validate_model_resources()
        bundle = _profile_collision_solver_bundle(
            str(scene_profile), model_content_hash, validate_model_resources)
    else:
        if validate_model_resources is not None:
            raise ValueError("Resource validation requires an explicit model content hash.")
        bundle = _profile_collision_solver_bundle(str(scene_profile))
    return RM65ObstacleCollisionOracle.from_solver(
        bundle.solver,
        scene=scene,
        solution_layout=bundle.scene_profile.solution_layout,
        expected_reference_frame_id=expected_reference_frame_id,
        expected_reference_frame_spec_hash=expected_reference_frame_spec_hash,
        attached_box_proxies=attached_box_proxies,
        obstacle_moving_geometry_names=(
            (
                *(box.geometry_name for box in bundle.scene_profile.hand_proxy_boxes),
                *(proxy.geometry_name for proxy in attached_box_proxies),
            )
            if use_profile_proxy_geometry
            else None
        ),
        stop_at_first_clearance_violation=stop_at_first_clearance_violation,
        collision_scope=(
            "profile_sparse_proxy_pairs"
            if use_profile_proxy_geometry else "profile_collision_pairs"
        ),
    )


__all__ = [
    "AttachedBoxCollisionProxy",
    "RM65_D405_ATTACHED_COLLISION_PROXIES",
    "RM65ObstacleCollisionOracle",
    "build_rm65_profile_obstacle_collision_oracle",
]
