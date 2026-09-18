"""Visible-only parallel-jaw solid and sweep checks, independent of inference.

This is a conservative nominal proxy check, not contact mechanics, unseen-space
clearance, or whole-arm safety. The free space between fingers is never filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import math
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from scipy.spatial import cKDTree
from dataset_contracts.canonical_encoding import canonical_content_hash


@dataclass(frozen=True)
class ParallelJawSolidGeometry:
    palm_z_m: float
    finger_length_m: float
    finger_thickness_m: float
    pad_height_m: float
    maximum_opening_m: float
    tool_from_solid_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    solid_yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.palm_z_m):
            raise ValueError("palm_z_m must be finite")
        for value in (self.finger_length_m, self.finger_thickness_m,
                      self.pad_height_m, self.maximum_opening_m):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("gripper dimensions must be finite and positive")
        offset = np.asarray(self.tool_from_solid_xyz_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("solid frame offset must contain three finite meters")
        if not math.isfinite(self.solid_yaw_rad):
            raise ValueError("solid frame yaw must be finite")
        object.__setattr__(self, "tool_from_solid_xyz_m", tuple(float(x) for x in offset))

    def solid_pose(self, tool_pose: np.ndarray) -> np.ndarray:
        """Locate +X-closing solids without changing the controller TCP.

        Only local yaw/translation are supported: +Z remains the approach axis,
        so the analytic approach sweep uses the same direction in both frames.
        """
        transform = np.eye(4)
        c, s = math.cos(self.solid_yaw_rad), math.sin(self.solid_yaw_rad)
        transform[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
        transform[:3, 3] = self.tool_from_solid_xyz_m
        return np.asarray(tool_pose) @ transform


class VisibleGraspScene:
    def __init__(self, *, points: np.ndarray, target_mask: np.ndarray,
                 geometry: ParallelJawSolidGeometry,
                 capture_qpos_by_side: Mapping[str, np.ndarray] | None = None,
                 capture_self_observation_evidence: (
                     Mapping[str, object] | None
                 ) = None) -> None:
        xyz = np.asarray(points, dtype=float)
        labels = np.asarray(target_mask)
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("scene must contain finite Nx3 XYZ points")
        if labels.dtype != np.bool_ or labels.shape != (len(xyz),):
            raise ValueError("scene target labels must be a boolean N vector")
        self.points = xyz.copy()
        self.target_mask = labels.copy()
        self.geometry = geometry
        self.capture_qpos_by_side = None
        if capture_qpos_by_side is not None:
            if set(capture_qpos_by_side) != {"left", "right"}:
                raise ValueError("capture qpos must retain both arms")
            qpos = {side: np.asarray(q, dtype=float).copy()
                    for side, q in capture_qpos_by_side.items()}
            if any(q.shape != (6,) or not np.all(np.isfinite(q)) for q in qpos.values()):
                raise ValueError("capture qpos must contain six finite radians per arm")
            for q in qpos.values():
                q.setflags(write=False)
            self.capture_qpos_by_side = MappingProxyType(qpos)
        self.capture_self_observation_evidence = None
        if capture_self_observation_evidence is not None:
            evidence = dict(capture_self_observation_evidence)
            removed_background = evidence.get("removed_background_points")
            if (
                isinstance(removed_background, bool)
                or not isinstance(removed_background, int)
                or removed_background < 0
                or evidence.get("removed_target_points") != 0
            ):
                raise ValueError(
                    "capture self-observation evidence must retain targets and "
                    "contain a nonnegative background count"
                )
            canonical_content_hash(
                evidence,
                domain="dexteleop.motion.capture_self_observation_evidence.v1",
            )
            self.capture_self_observation_evidence = MappingProxyType(evidence)
        self.points.setflags(write=False)
        self.target_mask.setflags(write=False)
        self._tree = cKDTree(self.points)
        self.semantic_content_hash = canonical_content_hash({
            "points_sha256": hashlib.sha256(self.points.tobytes()).hexdigest(),
            "target_labels_sha256": hashlib.sha256(self.target_mask.tobytes()).hexdigest(),
            "geometry": asdict(geometry),
            "capture_self_observation_evidence": (
                None
                if self.capture_self_observation_evidence is None
                else dict(self.capture_self_observation_evidence)
            ),
        }, domain="dexteleop.motion.visible_grasp_scene.v1")

    def _local_points(self, pose: np.ndarray, *, opening_m: float,
                      distance_m: float = 0) -> tuple[np.ndarray, np.ndarray]:
        g = self.geometry
        radius = math.sqrt((opening_m/2+g.finger_thickness_m)**2 +
                           (g.pad_height_m/2)**2 +
                           max(abs(g.palm_z_m-g.finger_thickness_m-distance_m),
                               abs(g.palm_z_m+g.finger_length_m))**2)
        indices = self._tree.query_ball_point(pose[:3, 3], radius+1e-9)
        return ((self.points[indices]-pose[:3, 3]) @ pose[:3, :3],
                self.target_mask[indices])

    @classmethod
    def from_depth(cls, *, depth_m: np.ndarray, target_mask: np.ndarray,
                   intrinsics: Mapping[str, float], reference_from_camera: np.ndarray,
                   geometry: ParallelJawSolidGeometry,
                   capture_qpos_by_side: Mapping[str, np.ndarray] | None = None,
                   capture_self_observation_evidence: (
                       Mapping[str, object] | None
                   ) = None) -> VisibleGraspScene:
        depth = np.asarray(depth_m)
        mask = np.asarray(target_mask)
        if (depth.ndim != 2 or mask.shape != depth.shape or mask.dtype != np.bool_
                or not np.all(np.isfinite(depth)) or np.any(depth < 0)):
            raise ValueError("depth and boolean target mask must be aligned")
        fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
        if not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
            raise ValueError("invalid intrinsics")
        transform = np.asarray(reference_from_camera, dtype=float)
        if (transform.shape != (4, 4) or not np.all(np.isfinite(transform))
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1)):
            raise ValueError("camera transform must be rigid")
        rows, columns = np.nonzero(depth > 0)
        z = depth[rows, columns]
        xyz = np.column_stack(((columns-cx)*z/fx, (rows-cy)*z/fy, z))
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        return cls(points=xyz, target_mask=mask[rows, columns], geometry=geometry,
                   capture_qpos_by_side=capture_qpos_by_side,
                   capture_self_observation_evidence=(
                       capture_self_observation_evidence
                   ))

    def _solid_mask(self, local: np.ndarray, *, opening_m: float,
                    distance_m: float = 0) -> np.ndarray:
        """Use the same separate palm/finger solids for filtering and checks."""
        g = self.geometry
        x, y, z = local.T
        fingers = self._finger_mask(local, opening_m=opening_m, distance_m=distance_m)
        palm = ((z > g.palm_z_m-g.finger_thickness_m-distance_m)
                & (z <= g.palm_z_m) & (np.abs(x) < opening_m/2+g.finger_thickness_m))
        return (np.abs(y) < g.pad_height_m/2) & (fingers | palm)

    def _finger_mask(self, local: np.ndarray, *, opening_m: float,
                     distance_m: float = 0) -> np.ndarray:
        g = self.geometry
        x, _y, z = local.T
        return (
            (z > g.palm_z_m - distance_m)
            & (z < g.palm_z_m + g.finger_length_m)
            & (np.abs(x) > opening_m / 2)
            & (np.abs(x) < opening_m / 2 + g.finger_thickness_m)
        )

    def _terminal_target_contact_mask(
        self,
        local: np.ndarray,
        *,
        opening_m: float,
        contact_depth_m: float,
    ) -> np.ndarray:
        """Return target points allowed to touch the fingertip at the goal."""
        if (
            not math.isfinite(contact_depth_m)
            or contact_depth_m <= 0.0
            or contact_depth_m > self.geometry.finger_length_m
        ):
            raise ValueError(
                "contact_depth_m must be finite and within finger length"
            )
        _x, y, z = local.T
        fingertip_z = self.geometry.palm_z_m + self.geometry.finger_length_m
        return (
            self._finger_mask(local, opening_m=opening_m)
            & (z >= fingertip_z - contact_depth_m)
            & (z <= fingertip_z)
            & (np.abs(y) < self.geometry.pad_height_m / 2)
        )

    def without_captured_gripper_observations(
        self,
        tool_tcp_poses: Mapping[str, np.ndarray],
        *,
        surface_margin_m: float = 0.0,
    ) -> tuple[VisibleGraspScene, int]:
        """Remove background attributable to open gripper solids at capture time.

        This is robot self-observation filtering, not collision-pair permission.
        Capture opening is approximated by maximum_opening_m, as in the path
        checks. A bounded surface margin accounts for depth samples on the
        observed solid boundary; it never fills the finger gap, extends an
        approach sweep, or removes target-labelled points. The resulting scene
        stays fixed while the robot moves in the planner.
        """
        margin = float(surface_margin_m)
        if (
            isinstance(surface_margin_m, bool)
            or not math.isfinite(margin)
            or not 0.0 <= margin <= self.geometry.finger_thickness_m
        ):
            raise ValueError(
                "surface_margin_m must be finite and within finger thickness"
            )
        occupied = np.zeros(len(self.points), dtype=bool)
        g = self.geometry
        for pose in tool_tcp_poses.values():
            self.collision_counts(pose, opening_m=g.maximum_opening_m)
            pose = g.solid_pose(np.asarray(pose, dtype=float))
            local = (self.points-pose[:3, 3]) @ pose[:3, :3]
            if margin == 0.0:
                occupied |= self._solid_mask(
                    local, opening_m=g.maximum_opening_m
                )
                continue
            x, y, z = local.T
            fingers = (
                (z >= g.palm_z_m - margin)
                & (z <= g.palm_z_m + g.finger_length_m + margin)
                & (np.abs(x) >= max(0.0, g.maximum_opening_m / 2.0 - margin))
                & (
                    np.abs(x)
                    <= g.maximum_opening_m / 2.0 + g.finger_thickness_m + margin
                )
            )
            palm = (
                (z >= g.palm_z_m - g.finger_thickness_m - margin)
                & (z <= g.palm_z_m + margin)
                & (
                    np.abs(x)
                    <= g.maximum_opening_m / 2.0 + g.finger_thickness_m + margin
                )
            )
            occupied |= (np.abs(y) <= g.pad_height_m / 2.0 + margin) & (
                fingers | palm
            )
        remove = occupied & ~self.target_mask
        return (VisibleGraspScene(points=self.points[~remove],
                                  target_mask=self.target_mask[~remove],
                                  geometry=g,
                                  capture_qpos_by_side=self.capture_qpos_by_side,
                                  capture_self_observation_evidence=(
                                      self.capture_self_observation_evidence
                                  )),
                int(np.count_nonzero(remove)))

    def _collision_local_points(
        self, pose: np.ndarray, *, opening_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        g = self.geometry
        if not math.isfinite(opening_m) or not 0 < opening_m <= g.maximum_opening_m:
            raise ValueError("opening is outside the geometry bound")
        pose = np.asarray(pose, dtype=float)
        if (pose.shape != (4, 4) or not np.all(np.isfinite(pose))
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1)):
            raise ValueError("grasp pose must be rigid")
        pose = g.solid_pose(pose)
        return self._local_points(pose, opening_m=opening_m)

    def collision_counts(self, pose: np.ndarray, *, opening_m: float) -> tuple[int, int]:
        """Count observed target/background points inside the nominal solids."""
        local, labels = self._collision_local_points(pose, opening_m=opening_m)
        occupied = self._solid_mask(local, opening_m=opening_m)
        return (int(np.count_nonzero(occupied & labels)),
                int(np.count_nonzero(occupied & ~labels)))

    def collision_counts_with_terminal_target_contact(
        self,
        pose: np.ndarray,
        *,
        opening_m: float,
        contact_depth_m: float,
    ) -> tuple[int, int, int]:
        """Separate allowed fingertip contact from target and background hits.

        The target is still treated as an obstacle everywhere except the
        terminal fingertip band.  This is the grasp-goal contact rule; callers
        that need a strict no-contact check should continue using
        :meth:`collision_counts`.
        """
        local, labels = self._collision_local_points(pose, opening_m=opening_m)
        occupied = self._solid_mask(local, opening_m=opening_m)
        target = occupied & labels
        terminal_contact = self._terminal_target_contact_mask(
            local, opening_m=opening_m, contact_depth_m=contact_depth_m
        )
        return (
            int(np.count_nonzero(target & ~terminal_contact)),
            int(np.count_nonzero(target & terminal_contact)),
            int(np.count_nonzero(occupied & ~labels)),
        )

    def approach_collision_counts(
        self,
        grasp_pose: np.ndarray,
        *,
        opening_m: float,
        distance_m: float,
        allow_terminal_target_contact: bool = False,
        terminal_target_contact_depth_m: float = 0.006,
    ) -> tuple[int, int]:
        """Check the entire fixed-orientation +Z approach analytically.

        In the final TCP frame, a stationary point moves only along local Z;
        expanding each solid backwards by the approach distance is therefore
        its exact swept volume. No waypoint sampling can skip a thin obstacle.
        """
        if not math.isfinite(distance_m) or distance_m < 0:
            raise ValueError("approach distance must be finite and nonnegative")
        # Validate opening and transform using the independently callable check.
        self.collision_counts(grasp_pose, opening_m=opening_m)
        pose = self.geometry.solid_pose(np.asarray(grasp_pose))
        local, labels = self._local_points(pose, opening_m=opening_m, distance_m=distance_m)
        occupied = self._solid_mask(local, opening_m=opening_m, distance_m=distance_m)
        target = occupied & labels
        if allow_terminal_target_contact:
            target &= ~self._terminal_target_contact_mask(
                local,
                opening_m=opening_m,
                contact_depth_m=terminal_target_contact_depth_m,
            )
        return (int(np.count_nonzero(target)),
                int(np.count_nonzero(occupied & ~labels)))
