"""Object inclination geometry; no observation, planning or control authority.

The material axis is selected once by the grasp owner. Rotating the object
about world up is unconstrained; changing the material axis is not yaw freedom.
"""

from dataclasses import dataclass
import math

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.object_transport import rigid_pose


_ANGLE_EPSILON_RAD = 1e-9


def _unit_direction(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite three-dimensional direction")
    length = math.hypot(*vector)
    if not math.isfinite(length) or length <= 0.:
        raise ValueError(f"{name} must be nonzero")
    # Preserve an already normalized basis across successive immutable bands.
    if not math.isclose(length, 1., rel_tol=0., abs_tol=1e-12):
        vector = vector / length
    return tuple(float(v) for v in vector)


def _inclination_rad(world_from_object, axis_object_xyz, up_world_xyz):
    # Evaluation consumes an already validated rigid pose from the pose owner.
    axis_world = np.asarray(world_from_object)[:3, :3] @ axis_object_xyz
    cosine = float(np.dot(axis_world, up_world_xyz))
    sine = math.hypot(*np.cross(axis_world, up_world_xyz))
    angle = math.atan2(sine, cosine)
    if not math.isfinite(angle):
        raise ValueError("Object inclination is not finite")
    return angle


def up_facing_box_axis(anchor_world_from_object, object_from_box, up_world_xyz):
    """Select one signed box axis at the grasp anchor, expressed in object axes.

    The caller retains this result for the grasp generation; it must not call
    this selector again on every moving observation.
    """
    anchor = rigid_pose(anchor_world_from_object, name="grasp attitude anchor")
    box = rigid_pose(object_from_box, name="object_from_box")
    up = _unit_direction(up_world_xyz, "world up")
    projections = (anchor[:3, :3] @ box[:3, :3]).T @ up
    index = int(np.argmax(np.abs(projections)))
    sign = 1. if projections[index] >= 0. else -1.
    return _unit_direction(sign * box[:3, index], "up-facing box axis")


@dataclass(frozen=True)
class ObjectInclinationBand:
    """A scalar inclination band, allowing world yaw and tilt-direction changes.

    Without a goal, inclination stays near the supplied anchor angle. With a
    goal, it stays near the closed interval between start and goal angles.
    This geometric interval proves neither monotonic leveling nor a slew rate.
    """

    axis_object_xyz: tuple[float, float, float]
    up_world_xyz: tuple[float, float, float]
    anchor_inclination_rad: float
    max_deviation_rad: float
    goal_inclination_rad: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "axis_object_xyz", _unit_direction(self.axis_object_xyz, "material axis"))
        object.__setattr__(self, "up_world_xyz", _unit_direction(self.up_world_xyz, "world up"))
        angles = (self.anchor_inclination_rad,) + (() if self.goal_inclination_rad is None
                                                 else (self.goal_inclination_rad,))
        if any(isinstance(value, (bool, np.bool_)) or not math.isfinite(value)
               or not 0. <= value <= math.pi for value in angles):
            raise ValueError("Inclination endpoints must be finite radians in [0, pi]")
        if (isinstance(self.max_deviation_rad, (bool, np.bool_))
                or not math.isfinite(self.max_deviation_rad) or not 0. < self.max_deviation_rad <= math.pi):
            raise ValueError("Inclination deviation must be finite radians in (0, pi]")
        object.__setattr__(self, "anchor_inclination_rad", float(self.anchor_inclination_rad))
        object.__setattr__(self, "max_deviation_rad", float(self.max_deviation_rad))
        if self.goal_inclination_rad is not None:
            object.__setattr__(self, "goal_inclination_rad", float(self.goal_inclination_rad))

    @classmethod
    def from_poses(cls, anchor_world_from_object, *, axis_object_xyz, up_world_xyz,
                   max_deviation_rad, goal_world_from_object=None):
        """Build from the grasp anchor, or an explicitly admitted release start.

        Supplying a release start does not replace the observer's grasp anchor.
        That caller must first validate its current cut against the old policy.
        """
        axis = _unit_direction(axis_object_xyz, "material axis")
        up = _unit_direction(up_world_xyz, "world up")
        start = rigid_pose(anchor_world_from_object, name="inclination band start")
        goal = (None if goal_world_from_object is None else
                rigid_pose(goal_world_from_object, name="inclination band goal"))
        return cls(axis, up, _inclination_rad(start, axis, up), max_deviation_rad,
                   None if goal is None else _inclination_rad(goal, axis, up))

    def inclination_rad(self, world_from_object):
        """Read one true/predicted rigid object pose; never a wrist rotation."""
        return _inclination_rad(world_from_object, self.axis_object_xyz, self.up_world_xyz)

    def deviation_rad(self, world_from_object):
        angle = self.inclination_rad(world_from_object)
        goal = self.anchor_inclination_rad if self.goal_inclination_rad is None else self.goal_inclination_rad
        return max(min(self.anchor_inclination_rad, goal) - angle,
                   angle - max(self.anchor_inclination_rad, goal), 0.)

    def accepts(self, world_from_object):
        return self.deviation_rad(world_from_object) <= self.max_deviation_rad + _ANGLE_EPSILON_RAD

    def to_payload(self):
        goal = self.anchor_inclination_rad if self.goal_inclination_rad is None else self.goal_inclination_rad
        return {"axis_object_xyz": list(self.axis_object_xyz), "up_world_xyz": list(self.up_world_xyz),
                "anchor_inclination_rad": self.anchor_inclination_rad,
                "goal_inclination_rad": self.goal_inclination_rad,
                "max_deviation_rad": self.max_deviation_rad,
                "minimum_inclination_rad": max(0., min(self.anchor_inclination_rad, goal) - self.max_deviation_rad),
                "maximum_inclination_rad": min(math.pi, max(self.anchor_inclination_rad, goal) + self.max_deviation_rad),
                "comparison_epsilon_rad": _ANGLE_EPSILON_RAD}

    @property
    def semantic_content_hash(self):
        return canonical_content_hash(self.to_payload(), domain="robot_motion.object_inclination_band.v1")
