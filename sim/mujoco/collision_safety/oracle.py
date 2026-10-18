from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .policy import CollisionSafetyPolicy
from .result import CollisionCheckResult


@dataclass(frozen=True)
class _ForbiddenGeomPair:
    geom1: int
    geom2: int
    geom1_name: str
    geom2_name: str
    body1_name: str
    body2_name: str
    pair_class: str


class CollisionSafetyOracle:
    """Precomputed forbidden-pair distance oracle over one MuJoCo model.

    Explicit hand/object pairs may have a contact threshold within the scene's
    hard bound. Other pairs keep the preventive margin. Report an actual
    blocking pair even if a permitted contact has a smaller signed distance.
    """

    def __init__(
        self, mujoco, model, policy: CollisionSafetyPolicy | None = None, *,
        pair_clearance_m: Mapping[tuple[int, int], float] | None = None,
    ) -> None:
        self.mujoco = mujoco
        self.model = model
        self.policy = policy or CollisionSafetyPolicy()
        self._fromto = np.empty(6, dtype=np.float64)
        self._pairs = self._build_pairs()
        self._geom1_ids = np.asarray([pair.geom1 for pair in self._pairs], dtype=np.int32)
        self._geom2_ids = np.asarray([pair.geom2 for pair in self._pairs], dtype=np.int32)
        self._local_aabb_center = np.asarray(self.model.geom_aabb[:, :3])
        self._local_aabb_half = np.asarray(self.model.geom_aabb[:, 3:])
        self._pair_by_geom_ids = {
            (pair.geom1, pair.geom2): pair for pair in self._pairs
        }
        self._pair_clearance_m = {}
        for pair, value in (pair_clearance_m or {}).items():
            key = tuple(sorted(pair))
            if key not in self._pair_by_geom_ids:
                raise ValueError("Contact clearance must bind an existing checked geom pair.")
            if self._pair_by_geom_ids[key].pair_class != "4:8":
                raise ValueError("Contact clearance is limited to hand/object pairs.")
            value = float(value)
            if not np.isfinite(value) or not -self.policy.hard_penetration_m <= value <= 0:
                raise ValueError("Contact clearance must retain the hard penetration bound.")
            self._pair_clearance_m[key] = value

    @property
    def forbidden_pair_count(self) -> int:
        return len(self._pairs)

    def _name(self, object_type, object_id: int, fallback: str) -> str:
        value = self.mujoco.mj_id2name(self.model, object_type, int(object_id))
        return fallback if value is None else str(value)

    def _build_pairs(self) -> tuple[_ForbiddenGeomPair, ...]:
        def welded_root(body_id: int) -> int:
            current = int(body_id)
            while current != 0 and int(self.model.body_dofnum[current]) == 0:
                current = int(self.model.body_parentid[current])
            return current

        excluded_roots = {
            tuple(sorted((welded_root(int(value) >> 16), welded_root(int(value) & 0xFFFF))))
            for value in self.model.exclude_signature
        }
        pairs: list[_ForbiddenGeomPair] = []
        for geom1 in range(int(self.model.ngeom)):
            group1 = int(self.model.geom_contype[geom1])
            if group1 not in {1, 2, 4, 8}:
                continue
            body1 = int(self.model.geom_bodyid[geom1])
            for geom2 in range(geom1 + 1, int(self.model.ngeom)):
                group2 = int(self.model.geom_contype[geom2])
                if not self.policy.pair_is_forbidden(group1, group2):
                    continue
                body2 = int(self.model.geom_bodyid[geom2])
                if body1 == body2:
                    continue
                root_pair = tuple(sorted((welded_root(body1), welded_root(body2))))
                # Fixed descendants are part of their nearest articulated body.
                # MuJoCo excludes are authored on URDF link bodies, so extend
                # those exclusions to welded proxy/contact-frame descendants.
                if root_pair[0] == root_pair[1] or root_pair in excluded_roots:
                    continue
                body1_name = self._name(
                    self.mujoco.mjtObj.mjOBJ_BODY, body1, f"<body:{body1}>"
                )
                body2_name = self._name(
                    self.mujoco.mjtObj.mjOBJ_BODY, body2, f"<body:{body2}>"
                )
                pairs.append(
                    _ForbiddenGeomPair(
                        geom1=geom1,
                        geom2=geom2,
                        geom1_name=self._name(
                            self.mujoco.mjtObj.mjOBJ_GEOM, geom1, f"<geom:{geom1}>"
                        ),
                        geom2_name=self._name(
                            self.mujoco.mjtObj.mjOBJ_GEOM, geom2, f"<geom:{geom2}>"
                        ),
                        body1_name=body1_name,
                        body2_name=body2_name,
                        pair_class=f"{min(group1, group2)}:{max(group1, group2)}",
                    )
                )
        return tuple(pairs)

    def check(
        self,
        data,
        *,
        minimum_clearance_m: float | None = None,
        distance_query_limit_m: float | None = None,
    ) -> CollisionCheckResult:
        """Check preventive clearance, optionally with a recovery threshold.

        The normal threshold is the configured preventive margin.  Actuated
        position control can leave the measured state a small distance inside
        that margin (or just inside the hard penetration bound) even when the
        command target itself is valid.  Callers may therefore request a
        temporary, explicit threshold for that recovery projection; active
        contact checks remain governed by the configured ``hard_penetration_m``
        bound.

        Preserve the policy's reporting radius for general simulation. A
        caller may explicitly use the admission threshold as the query radius
        to avoid broad-query mesh artifacts; that result is a clipped lower
        bound, not measured clearance. Neither mode changes the safety margin.
        """

        clearance = (
            self.policy.minimum_clearance_m
            if minimum_clearance_m is None
            else float(minimum_clearance_m)
        )
        if not np.isfinite(clearance):
            raise ValueError("minimum_clearance_m must be finite.")
        minimum = float("inf")
        closest: _ForbiddenGeomPair | None = None
        minimum_is_lower_bound = False
        minimum_raw_distance = None
        distance_query_disagreement_count = 0
        blocking = None
        blocking_deficit = 0.0
        required_limit = max(0.0, clearance)
        limit = max(self.policy.distance_query_limit_m, required_limit)
        if distance_query_limit_m is not None:
            requested_limit = float(distance_query_limit_m)
            if not np.isfinite(requested_limit) or requested_limit < required_limit:
                raise ValueError(
                    "distance_query_limit_m must be finite and cover the clearance threshold."
                )
            limit = requested_limit
        if not self._pairs:
            return CollisionCheckResult(safe=True, minimum_distance_m=float("inf"))
        rotation = np.asarray(data.geom_xmat).reshape((-1, 3, 3))
        world_center = np.asarray(data.geom_xpos) + np.einsum(
            "nij,nj->ni", rotation, self._local_aabb_center
        )
        world_half = np.einsum(
            "nij,nj->ni", np.abs(rotation), self._local_aabb_half
        )
        separation = (
            np.abs(world_center[self._geom1_ids] - world_center[self._geom2_ids])
            - world_half[self._geom1_ids]
            - world_half[self._geom2_ids]
        )
        aabb_lower_bound = np.linalg.norm(np.maximum(separation, 0.0), axis=1)
        candidate_indices = np.flatnonzero(aabb_lower_bound <= limit)
        for pair_index in candidate_indices:
            pair = self._pairs[int(pair_index)]
            distance = float(
                self.mujoco.mj_geomDistance(
                    self.model,
                    data,
                    pair.geom1,
                    pair.geom2,
                    limit,
                    self._fromto,
                )
            )
            if not np.isfinite(distance):
                raise RuntimeError("MuJoCo geom distance query returned a non-finite result.")
            # Disjoint world AABBs prove a positive separation independently
            # of CCD. Native CCD can return a spurious negative distance for
            # nearly planar CAD meshes at broad query radii. Preserve the
            # conservative geometric bound, never an invented exact distance;
            # overlapping AABBs still require the signed narrow-phase result.
            lower_bound = max(0.0, float(aabb_lower_bound[pair_index]) - 1e-9)
            bounded = lower_bound > 0.0 and distance < lower_bound
            raw_distance = distance if bounded else None
            if bounded:
                distance = lower_bound
                distance_query_disagreement_count += 1
            if distance < minimum:
                minimum = distance
                closest = pair
                minimum_is_lower_bound = bounded or distance >= limit
                minimum_raw_distance = raw_distance
            pair_clearance = self._pair_clearance_m.get((pair.geom1, pair.geom2), clearance)
            if distance - pair_clearance < blocking_deficit:
                blocking_deficit = distance - pair_clearance
                blocking = (
                    pair, distance, bounded or distance >= limit,
                    pair_clearance, raw_distance,
                )
        if closest is None:
            return CollisionCheckResult(
                safe=True, minimum_distance_m=limit,
                minimum_distance_is_lower_bound=True,
            )
        if blocking is not None:
            (
                closest, minimum, minimum_is_lower_bound,
                required_clearance, minimum_raw_distance,
            ) = blocking
        else:
            required_clearance = self._pair_clearance_m.get(
                (closest.geom1, closest.geom2), clearance
            )
        return CollisionCheckResult(
            safe=blocking is None,
            minimum_distance_m=minimum,
            geom1_id=closest.geom1,
            geom2_id=closest.geom2,
            geom1_name=closest.geom1_name,
            geom2_name=closest.geom2_name,
            body1_name=closest.body1_name,
            body2_name=closest.body2_name,
            pair_class=closest.pair_class,
            minimum_distance_is_lower_bound=minimum_is_lower_bound,
            distance_query_disagreement_count=distance_query_disagreement_count,
            distance_query_raw_distance_m=minimum_raw_distance,
            required_clearance_m=required_clearance if self._pair_clearance_m else None,
        )

    def check_contacts(self, data) -> CollisionCheckResult:
        """Low-cost hard-penetration check over contacts already computed."""

        minimum = float("inf")
        closest: _ForbiddenGeomPair | None = None
        blocking = None
        deficit = 0.0
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            key = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            pair = self._pair_by_geom_ids.get(key)
            if pair is not None:
                distance = float(contact.dist)
                threshold = self._pair_clearance_m.get(
                    key, 0.0 if self._pair_clearance_m and pair.pair_class == "4:8"
                    else -self.policy.hard_penetration_m)
                if distance < minimum:
                    minimum, closest = distance, pair
                if distance - threshold < deficit:
                    blocking = (pair, distance, threshold)
                    deficit = distance - threshold
        if closest is None:
            return CollisionCheckResult(safe=True, minimum_distance_m=float("inf"))
        threshold = None
        if blocking is not None:
            closest, minimum, threshold = blocking
        return CollisionCheckResult(
            safe=blocking is None,
            minimum_distance_m=minimum,
            geom1_id=closest.geom1,
            geom2_id=closest.geom2,
            geom1_name=closest.geom1_name,
            geom2_name=closest.geom2_name,
            body1_name=closest.body1_name,
            body2_name=closest.body2_name,
            pair_class=closest.pair_class,
            required_clearance_m=threshold if self._pair_clearance_m else None,
        )
