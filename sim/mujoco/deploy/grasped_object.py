"""Measured native hand-object relation, observed under the existing rig lock.

This observer never advances physics, modifies live data, creates a weld or
grants execution authority. The first supported bilateral cut is contact-secured,
not a verified grasp. Following an actual lift must be observed before carry.
The original relative transform is never re-anchored to hide cumulative slip.
"""

import copy
from dataclasses import dataclass

import numpy as np

from robot_motion.object_transport import rigid_pose
from robot_motion.object_release import (
    ObjectReleaseMeasurement,
    ObjectReleasePolicy,
    ObjectReleaseWindow,
)
from robot_motion.pose import pose_delta_metrics
from sim.mujoco.deploy.contact_timing import is_single_physics_step_elapsed
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw


@dataclass(frozen=True)
class NativeGraspedObjectCut:
    sim_time_s: float
    world_from_object: np.ndarray
    world_from_tcp: np.ndarray
    tool_from_object: np.ndarray
    object_com_world_m: tuple[float, float, float]
    sensor_contacts: tuple[bool, bool]
    support_contact: bool
    other_contact: bool
    measured_close: float
    command_generation: int
    status: str
    relative_translation_error_m: float
    relative_rotation_error_deg: float


class ObjectMotionPreconditionError(ValueError):
    """A phase is unavailable; no physical relation failure is inferred."""


class ObjectRelationInvalidated(ValueError):
    """Measured relation loss, latched until a new gripper generation."""


class MujocoGraspedObjectObserver:
    """One scene/object/gripper; generation changes revoke the prior relation.

    At physics-owner cadence, one sensor-contact dropout may reuse the prior
    fully bilateral cut for exactly one native timestep, if the closed command
    and frozen relation remain valid. Lower-rate or sustained losses fail.
    """

    def __init__(self, *, rig, object_body_name, sensor_geom_names, support_body_names,
                 gripper_name, command_state, world_tcp_from_scene, track_close_contacts=False,
                 contact_verified_close=False, gripper_settled=None):
        self.rig = rig
        self.object_body_name = object_body_name
        self.sensor_geom_names = tuple(sensor_geom_names)
        self.support_body_names = tuple(support_body_names)
        if len(set(self.support_body_names)) != len(self.support_body_names):
            raise ValueError("Object supports must have unique native identities.")
        self.gripper_name = gripper_name
        self.command_state = command_state
        self.world_tcp_from_scene = world_tcp_from_scene
        if type(track_close_contacts) is not bool:
            raise TypeError("Contact tracking must be explicitly boolean.")
        self.track_close_contacts = track_close_contacts
        if type(contact_verified_close) is not bool or (
            contact_verified_close and not callable(gripper_settled)
        ):
            raise ValueError("Contact-verified close requires an explicit settled gripper producer.")
        self.contact_verified_close = contact_verified_close
        self.gripper_settled = gripper_settled
        if (len(self.sensor_geom_names) != 2 or len(set(self.sensor_geom_names)) != 2
                or not self.support_body_names or not callable(command_state)
                or not callable(world_tcp_from_scene)):
            raise ValueError("Object observation requires two sensors, supports and measured callbacks.")
        self._active_support_body_name = self.support_body_names[0]
        self._generation = None
        self._anchor = None
        self._initial_z = self._initial_tcp_z = None
        self._window_start = None
        self._window_count = 0
        self._verified = False
        self._last = None
        self._failure = None
        self._scratch = None
        self._model = None
        self._tracking_phase = None
        self._last_physics_cut_time = None
        self._release_window = None

    def measured_closed(self, measured):
        """Keep the legacy floor unless the deployed hand has settled target contact."""
        return bool(
            np.isfinite(measured) and (
                measured >= .05 or (
                    self.contact_verified_close and measured > 0.
                    and (self._anchor is not None or self.gripper_settled())
                )
            )
        )

    def _reset_generation(self, generation, target):
        # Opening revokes the grasp as before. Only a previously verified grasp
        # arms release diagnostics; initial open/closed commands are not proof.
        released_verified_grasp = target == 0. and self._failure is None and (
            self._verified or self._release_window is not None)
        self._release_window = (ObjectReleaseWindow(ObjectReleasePolicy(
            duration_s=.6, min_measurements=60, max_sample_gap_s=.05,
            position_tolerance_m=.002, rotation_tolerance_rad=.05))
            if released_verified_grasp else None)
        self._generation = generation
        # Preserve the selected place support across opening so the release
        # window cannot call a fall onto the table a supported placement.
        if target == 1.:
            self._active_support_body_name = self.support_body_names[0]
        self._anchor = self._last = self._window_start = None
        self._verified = False
        self._window_count = 0
        self._failure = None
        self._tracking_phase = None
        self._last_physics_cut_time = None

    @property
    def active_support_body_name(self):
        return self._active_support_body_name

    def _capture(self, scene, source_data=None, *, support_body_name=None):
        model, mj = scene.model, scene.mujoco
        if self._scratch is None:
            self._scratch, self._model = mj.MjData(model), model
        if self._model is not model or (source_data is not None and source_data.model is not model):
            raise ValueError("Object observation model changed.")
        data = self._scratch
        mj.mj_copyData(data, model, scene.data if source_data is None else source_data)
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        mj.mj_collision(model, data)
        view = copy.copy(scene)
        view.data = data
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.time):
            raise ValueError("Object observation has nonfinite native state.")
        body = int(model.body(self.object_body_name).id)
        sensors = tuple(int(model.geom(name).id) for name in self.sensor_geom_names)
        selected_support = self._active_support_body_name if support_body_name is None else support_body_name
        if selected_support not in self.support_body_names:
            raise ValueError("Object support is not in the deployed allowlist.")
        supports = {int(model.body(selected_support).id)}
        target_geoms = set(np.flatnonzero((model.geom_bodyid == body)
                           & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))).tolist())
        touching = set()
        for contact in data.contact[:int(data.ncon)]:
            if contact.dist > 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            if a in target_geoms and b not in target_geoms:
                touching.add(b)
            elif b in target_geoms and a not in target_geoms:
                touching.add(a)
        supported = any(int(model.geom_bodyid[g]) in supports for g in touching)
        other = any(g not in sensors and int(model.geom_bodyid[g]) not in supports for g in touching)
        obj = rigid_pose(view.body_pose(self.object_body_name), name="native world_from_object")
        tcp = rigid_pose(self.world_tcp_from_scene(view), name="calibrated native world_from_tcp")
        return (float(data.time), obj, tcp, tuple(g in touching for g in sensors),
                supported, other, float(view.get_gripper_close(self.gripper_name)),
                tuple(float(value) for value in data.xipos[body]))

    def observe(self, *, phase, establish=False, source_data=None, place_support_body_name=None):
        """Consume a same-cut measurement. A new close is not a grasp proof."""
        if phase not in {"lift", "carry", "place"}:
            raise ValueError("Object observation requires a lift, carry or place phase.")
        if place_support_body_name is not None and (
            phase != "place" or place_support_body_name not in self.support_body_names
        ):
            raise ObjectMotionPreconditionError("Place support is not an admitted object identity.")
        return self.rig.capture_scene(lambda scene: self._observe(
            scene, phase=phase, establish=establish, source_data=source_data,
            place_support_body_name=place_support_body_name))

    def _observe(self, scene, *, phase, establish=False, source_data=None,
                 place_support_body_name=None):
        # State updates and native capture share the original physics-owner lock.
        at_physics_cut = (source_data is not None and scene is not None
                          and source_data is scene.data)
        generation, target = self.command_state()
        if generation != self._generation:
            self._reset_generation(generation, target)
        if self._failure is not None:
            raise ObjectRelationInvalidated("Object relation invalidated: " + self._failure)
        try:
            selected = ({} if place_support_body_name is None else
                        {"support_body_name": place_support_body_name})
            stamp, obj, tcp, sides, support, other, measured, com = self._capture(
                scene, source_data, **selected)
            if target != 1.0 or not self.measured_closed(measured):
                raise ObjectRelationInvalidated("Actual closed command and measured closure are required.")
            if support and phase == "carry" and not self._verified:
                raise ObjectMotionPreconditionError("Carry requires an unsupported verified lift first.")
            if other or (support and phase == "carry"):
                raise ObjectRelationInvalidated("Object has contact outside the selected phase support policy.")
            relative = rigid_pose(np.linalg.inv(tcp) @ obj, name="measured tool_from_object")
            if not all(sides):
                previous = self._last
                timestep = (None if self._model is None else float(self._model.opt.timestep))
                if (sum(sides) == 1 and previous is not None
                        and previous.sensor_contacts == (True, True) and self._anchor is not None
                        and timestep is not None and not other
                        and is_single_physics_step_elapsed(previous.sim_time_s, stamp, timestep)):
                    slip, angle = pose_delta_metrics(self._anchor, relative)
                    if slip <= .003 and angle <= 2.0:
                        # The measured payload oracle independently enforces the same one-step
                        # window; keep the last fully bilateral cut as observer evidence.
                        if at_physics_cut:
                            self._tracking_phase = phase
                            self._last_physics_cut_time = stamp
                        return previous
                raise ObjectRelationInvalidated(
                    "Both target sensor contacts are required outside one physics-step recovery.")
            if self._anchor is None:
                if not establish or phase != "lift":
                    raise ObjectMotionPreconditionError("A new grasp must first establish the measured lift relation.")
                self._anchor = relative
                self._initial_z, self._initial_tcp_z = com[2], float(tcp[2, 3])
            slip, angle = pose_delta_metrics(self._anchor, relative)
            if slip > .003 or angle > 2.0:
                raise ObjectRelationInvalidated("Object slipped beyond the frozen 3mm/2deg hand-object relation.")
            if self._last is not None and stamp < self._last.sim_time_s:
                raise ObjectRelationInvalidated("Native object observation time went backwards.")
            if self._last is not None and stamp - self._last.sim_time_s > .05 + 1e-9:
                if not establish:
                    raise ObjectRelationInvalidated("Object follow observation gap exceeds 50ms of simulation time.")
                self._verified = False
                self._window_start, self._window_count = None, 0
            lifted = (com[2] - self._initial_z >= .03
                      and tcp[2, 3] - self._initial_tcp_z >= .03 and not support)
            fresh = self._last is None or stamp > self._last.sim_time_s
            if lifted and fresh:
                if self._window_start is None:
                    self._window_start = stamp
                    self._window_count = 0
                self._window_count += 1
                if stamp - self._window_start >= .5 and self._window_count >= 3:
                    self._verified = True
            elif not lifted and not self._verified:
                self._window_start, self._window_count = None, 0
            if establish and phase != "lift" and not self._verified:
                raise ObjectMotionPreconditionError("Carry/place requires observed unsupported lift and rigid following.")
            self._last = NativeGraspedObjectCut(
                stamp, obj, tcp, self._anchor, com, sides, support, other, measured, generation,
                "held_verified" if self._verified else "contact_secured", float(slip), float(angle))
            # Only the initial lift or a guard on the live physics-owned MjData
            # may change policy. Binding/activation preflights use private copies.
            if self._tracking_phase is None or at_physics_cut:
                self._tracking_phase = phase
            if at_physics_cut:
                self._last_physics_cut_time = stamp
                if phase == "place" and place_support_body_name is not None:
                    self._active_support_body_name = place_support_body_name
            return self._last
        except ObjectMotionPreconditionError:
            raise
        except (ValueError, FloatingPointError) as error:
            self._failure = str(error)
            raise

    def observe_physics_step(self, scene):
        """Continue an established relation on the sole physics owner's tick.

        Dormant before an explicit lift unless contact tracking is bound. That
        mode freezes a contact-secured relation after an explicit close while
        supported; it still requires an actual observed lift to verify holding.
        Expected relation losses remain latched
        for the next object operation; they do not invent a global arm fault.
        Active payload guards retain their original stop behavior. Unexpected
        observer failures propagate to the rig's existing fault/hold boundary.
        """
        generation, target = self.command_state()
        if generation != self._generation:
            self._reset_generation(generation, target)
        if self._release_window is not None:
            stamp, obj, _, sides, support, other, _, _ = self._capture(scene)
            self._release_window.observe(ObjectReleaseMeasurement(
                measured_at=stamp,
                pose_xyz_quat_xyzw=tuple(pose_to_xyz_quat_xyzw(obj)),
                support_contact=support, gripper_contact=any(sides), other_contact=other))
            return
        if self._failure is not None:
            return
        if self._anchor is None:
            if not self.track_close_contacts or target != 1.:
                return
            _, _, _, sides, support, other, measured, _ = self._capture(scene)
            if all(sides) and support and not other and self.measured_closed(measured):
                try:
                    self._observe(scene, phase="lift", establish=True, source_data=scene.data)
                except ObjectRelationInvalidated:
                    return
            return
        if float(scene.data.time) == self._last_physics_cut_time:
            # Also deduplicate a guarded one-tick dropout. Its historical fully
            # bilateral evidence deliberately keeps the preceding timestamp.
            return
        try:
            self._observe(scene, phase=self._tracking_phase, source_data=scene.data)
        except ObjectRelationInvalidated:
            # _observe retained the first measured failure; no retry/re-anchoring.
            return

    @property
    def latest(self):
        """Return historical proof, not a fresh reading or a renewed holding claim."""
        return self._last

    def _release_snapshot(self, stamp, generation):
        if self._release_window is None or generation != self._generation:
            return None
        release = self._release_window.snapshot(now=stamp)
        if release["measured_at"] != stamp and release["evidence_fresh"]:
            return {**release, "status": "current_cut_unobserved",
                    "stable_supported_release": False}
        return release

    def release_snapshot(self):
        """Read existing history under the owner lock; never sample or renew it."""
        return self.rig.capture_scene(lambda scene: self._release_snapshot(
            float(scene.data.time), self.command_state()[0]))

    def diagnostic_snapshot(self):
        """Measure current relation without advancing or renewing grasp evidence.

        This is diagnostic data, not an execution precondition or a new proof of
        rigid following. In particular, a fresh kinematic reading cannot repair
        a gap in the observation window or silently re-anchor the object.
        """
        def capture(scene):
            generation, target = self.command_state()
            stamp, obj, tcp, sides, support, other, measured, _ = self._capture(scene)
            last = self._last
            age = None if last is None else stamp - last.sim_time_s
            if age is not None and age < 0:
                raise ValueError("Native object diagnostic time went backwards.")
            relative = rigid_pose(np.linalg.inv(tcp) @ obj, name="diagnostic tool_from_object")
            slip, angle = ((None, None) if self._anchor is None
                           else pose_delta_metrics(self._anchor, relative))
            command_matches = generation == self._generation and target == 1.0
            relation_valid = bool(
                command_matches and self.measured_closed(measured)
                and self._anchor is not None and slip <= .003 and angle <= 2.0
                and all(sides) and not other and self._failure is None)
            fresh = age is not None and age <= .05 + 1e-9
            release = self._release_snapshot(stamp, generation)
            return {
                "sim_time_s": stamp,
                "command_generation": generation,
                "relation_generation": self._generation,
                "command_matches_relation": command_matches,
                "current_relation_within_bounds": relation_valid,
                "sensor_contacts": sides,
                "support_contact": support,
                "other_contact": other,
                "measured_close": measured,
                "relative_translation_error_m": slip,
                "relative_rotation_error_deg": angle,
                "last_evidence_sim_time_s": None if last is None else last.sim_time_s,
                "evidence_age_sec": age,
                "evidence_fresh": fresh,
                "historical_status": None if last is None else last.status,
                "holding_verified": bool(
                    last is not None and last.status == "held_verified"
                    and fresh and relation_valid and not support),
                "failure_reason": self._failure,
                "tracking_phase": self._tracking_phase,
                "release_observation": release,
                "scope": "current native diagnostic; does not renew the frozen grasp evidence",
            }

        return self.rig.capture_scene(capture)
