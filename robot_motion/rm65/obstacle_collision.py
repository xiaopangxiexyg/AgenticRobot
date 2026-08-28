"""Independent Pinocchio/FCL collision oracle for RM65 obstacle planning.

The oracle clones the supplied collision model before adding scene geometry.
It never mutates a live IK solver's model or data and it owns no execution or
hardware authority.
"""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.scene_obstacles import ObstacleSceneSnapshot


_OBSTACLE_GEOMETRY_PREFIX = "aabb_obstacle::"


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
        return {"left": tuple(range(0, 6)), "right": tuple(range(6, 12))}
    if nq == 14:
        return {"left": tuple(range(1, 7)), "right": tuple(range(8, 14))}
    if solution_layout == "dual_arm_hand" and nq >= 24:
        return {"left": tuple(range(0, 6)), "right": tuple(range(18, 24))}
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
        pin_module: Any | None = None,
        hppfcl_module: Any | None = None,
    ) -> None:
        if not isinstance(scene, ObstacleSceneSnapshot):
            raise TypeError("scene must be an ObstacleSceneSnapshot.")
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
        nq = int(getattr(model, "nq"))
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
        base_geometry_count = len(collision_model.geometryObjects)
        moving_geometry_ids = tuple(
            geometry_id
            for geometry_id, geometry in enumerate(collision_model.geometryObjects)
            if int(getattr(geometry, "parentJoint", 0)) != 0
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
        for obstacle_geometry_id in obstacle_ids:
            for moving_geometry_id in moving_geometry_ids:
                key = tuple(sorted((obstacle_geometry_id, moving_geometry_id)))
                if key not in existing_pairs:
                    collision_model.addCollisionPair(pin.CollisionPair(*key))
                    existing_pairs.add(key)

        self._pin = pin
        self._model = model
        self._model_data = model.createData()
        self._collision_model = collision_model
        self._collision_data = pin.GeometryData(collision_model)
        self._reference_full_q = reference.copy()
        self._indices = MappingProxyType(index_map)
        self._scene = scene
        self._obstacle_ids = MappingProxyType(obstacle_ids)
        self._base_geometry_count = base_geometry_count
        self._moving_geometry_ids = moving_geometry_ids

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
    ) -> "RM65ObstacleCollisionOracle":
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
        )

    @property
    def scene_hash(self) -> str:
        return self._scene.semantic_content_hash

    @property
    def collision_model_is_independent(self) -> bool:
        return True

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
        try:
            self._pin.computeDistances(
                self._model,
                self._model_data,
                self._collision_model,
                self._collision_data,
                full_q,
            )
            distance_results = tuple(self._collision_data.distanceResults)
        except Exception as exc:
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="pinocchio_fcl_distance_query_failed",
                metadata={
                    **self._base_metadata(pair_count),
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
                    **self._base_metadata(pair_count),
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
                metadata=self._base_metadata(pair_count),
            )
        closest_index = int(np.argmin(distances))
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
                **self._base_metadata(pair_count),
                "closest_geometry_names": (first_name, second_name),
                "closest_obstacle_id": obstacle_id,
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
            "scene_hash": self._scene.semantic_content_hash,
            "scene_revision": self._scene.revision,
            "collision_model_is_independent": True,
            "collision_scope": "full_pinocchio_collision_model",
            "base_geometry_count": self._base_geometry_count,
            "obstacle_geometry_count": len(self._obstacle_ids),
            "moving_geometry_count": len(self._moving_geometry_ids),
            "collision_pair_count": int(pair_count),
            "minimum_clearance_m": self._scene.minimum_clearance_m,
        }


__all__ = ["RM65ObstacleCollisionOracle"]
