"""RealMan held-object observation for carried-object motion and grip verification.

One observation captures stationary aligned RGB-D views (each with its
capture-time dual-arm qpos) and measures the object in the holding tool frame
(``robot_motion.held_object_measurement``). The holding arm's own wrist camera
is the primary view: it is segmented with the deployment's text segmentation
(or, without a text prompt, from a point prompt at the pixel showing
something between the pads) and alone decides whether something is between
the fingers. Only then is every other view segmented, from a point prompt
where it sees the primary surface (or by the same text), and allowed to
extend the primary box by a bounded amount. Other cameras add their station
calibration error to the gripper geometry; on the saved real samples their
own "between the pads" pixels were mostly the gripper's fingers.
The gripper sample adds the measured closure and the last commanded target;
holding is verified only with that sample. Nothing here moves the
robot, commands a gripper or grants motion authority; a planner only accepts
the resulting box through ``RM65CarriedObjectBox``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from agent_gateway.profiles.segmentation import decode_segmentation_mask_npy
from agent_gateway.skills.gripper.runtime import GripperConvergenceProfile
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.held_object_measurement import (
    HeldObjectBox,
    HeldObjectView,
    SupportPlane,
    grip_seed_pixel,
    measure_held_object_box,
    projected_seed_pixel,
)
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.tool_entity_scene import (
    RELATION_MAXIMUM_DEPTH_M,
    object_surface_from_mask,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65CarriedObjectBox

from .rm65_camera_geometry import REAL_WRIST_CAMERA_SIDE_BY_ROLE

REAL_HELD_OBJECT_SEMANTICS_V2 = "agenticrobot.real.held_object_observation.v2"
# The primary (own wrist) camera role of each arm.
REAL_HELD_OBJECT_PRIMARY_ROLE_BY_SIDE = MappingProxyType(
    {side: role for role, side in REAL_WRIST_CAMERA_SIDE_BY_ROLE.items()})
# A closed gripper that stopped clearly before its full-close position holds
# something between the pads; a fully closed one holds nothing graspable.
REAL_HELD_OBJECT_FULL_CLOSE_MARGIN = 0.03
# An open DMTAC reads about 0.0. Below the gripper Skill's minimum closed
# position the pads do not grip (the Skill itself never settles a close there).
REAL_HELD_OBJECT_MIN_CLOSE = GripperConvergenceProfile().minimum_closed_position


@dataclass(frozen=True)
class RealHeldObjectObservation:
    """One measured held-object cut of one arm."""

    side: str
    label: str
    language_prompt: str | None
    station_revision: int
    capture_qpos_by_side: Mapping[str, tuple[float, ...]]
    world_from_tcp: np.ndarray
    measurement: HeldObjectBox
    segments: tuple[dict, ...]
    unmeasured: tuple[dict, ...]
    gripper: Mapping[str, object] | None = None
    evidence_ref: str = field(init=False)

    def __post_init__(self) -> None:
        pose = np.array(self.world_from_tcp, dtype=float).reshape(4, 4)
        pose.setflags(write=False)
        object.__setattr__(self, "world_from_tcp", pose)
        object.__setattr__(self, "evidence_ref", "real-held-object:" + canonical_content_hash(
            self._payload(), domain="agenticrobot.real.held_object_observation.v2"))

    def _payload(self) -> dict:
        return {"semantics": REAL_HELD_OBJECT_SEMANTICS_V2, "side": self.side, "label": self.label,
                "language_prompt": self.language_prompt, "station_revision": self.station_revision,
                "capture_qpos_by_side": {k: list(v) for k, v in self.capture_qpos_by_side.items()},
                "world_from_tcp": self.world_from_tcp.tolist(), "measurement": self.measurement.as_dict(),
                "segments": [dict(s) for s in self.segments], "unmeasured": [dict(u) for u in self.unmeasured],
                "gripper": None if self.gripper is None else dict(self.gripper)}

    @property
    def gripper_closed_on_object(self) -> bool | None:
        """The DMTAC grips something (None without a gripper sample).

        The gripper must report no fault, its last commanded target (when this
        session commanded it) must be a close, and the measured closure must
        have stopped between the minimum closed position and full close.
        """
        if self.gripper is None or self.gripper.get("measured_close") is None:
            return None
        if self.gripper.get("fault_code") is not None:
            return False
        target = self.gripper.get("last_command_target_close")
        if target is not None and float(target) < 0.5:
            return False
        close = float(self.gripper["measured_close"])
        return REAL_HELD_OBJECT_MIN_CLOSE <= close < 1.0 - REAL_HELD_OBJECT_FULL_CLOSE_MARGIN

    @property
    def holding_verified(self) -> bool:
        """Between the fingers in a measured view and the sampled gripper closed on it."""
        return self.measurement.held and self.gripper_closed_on_object is True

    def carried_object_box(self, *, object_id: str | None = None) -> RM65CarriedObjectBox:
        if not self.measurement.held or self.measurement.tool_from_box is None:
            raise ValueError("Only a measured held object can be carried.")
        return RM65CarriedObjectBox(
            object_id=object_id or f"{self.side}_held_object", side=self.side,
            tool_from_box=self.measurement.tool_from_box, size_xyz_m=self.measurement.size_xyz_m,
            evidence_ref=self.evidence_ref)

    def as_dict(self) -> dict:
        return {**self._payload(), "evidence_ref": self.evidence_ref,
                "gripper_closed_on_object": self.gripper_closed_on_object,
                "holding_verified": self.holding_verified}


class RealHeldObjectObserver:
    """Measure one arm's held object from fresh stationary RGB-D views.

    ``scene_port`` is the tool-entity scene port (stationary captures, request-
    safe tool FK); ``gripper_state_by_side`` optionally reads one fresh DMTAC
    sample per side without commanding it, and ``gripper_command_by_side``
    the port's last commanded ``(target_close, attempt_ref)`` (both None
    before any command). ``roles`` must include every measured arm's own wrist
    camera (``REAL_HELD_OBJECT_PRIMARY_ROLE_BY_SIDE``).
    """

    def __init__(self, *, scene_port, segmentation_provider, tool_entity_by_side: Mapping[str, object],
                 roles: Sequence[str], gripper_state_by_side: Mapping[str, Callable[[], object]] | None = None,
                 gripper_command_by_side: Mapping[str, Callable[[], tuple]] | None = None,
                 maximum_depth_m: float = RELATION_MAXIMUM_DEPTH_M):
        if not callable(getattr(scene_port, "capture_views", None)):
            raise TypeError("scene_port must capture stationary views.")
        if not callable(getattr(segmentation_provider, "try_segment_capture", None)):
            raise TypeError("segmentation_provider must segment captures.")
        if not tool_entity_by_side or not roles:
            raise ValueError("Held-object observation needs a gripper entity and camera roles.")
        if any(REAL_HELD_OBJECT_PRIMARY_ROLE_BY_SIDE.get(side) not in tuple(roles) for side in tool_entity_by_side):
            raise ValueError("Held-object observation needs each measured arm's own wrist camera.")
        if not math.isfinite(maximum_depth_m) or maximum_depth_m <= 0:
            raise ValueError("maximum_depth_m must be positive.")
        self.scene_port = scene_port
        self.segmentation_provider = segmentation_provider
        self.tool_entity_by_side = dict(tool_entity_by_side)
        self.roles = tuple(roles)
        self.gripper_state_by_side = dict(gripper_state_by_side or {})
        self.gripper_command_by_side = dict(gripper_command_by_side or {})
        self.maximum_depth_m = float(maximum_depth_m)

    def _segment_view(self, capture, *, language_prompt, seed, seed_evidence, prompt, use, min_score,
                      segments, unmeasured):
        """Segment one view (text, or a point prompt at ``seed``); its measured surface points or None."""
        prepared = self.segmentation_provider.try_segment_capture(
            capture=self.scene_port.segmentation_input(capture),
            parameters={"role": capture.role, "language_prompt": language_prompt,
                        "target_pixel_uv": None if seed is None else list(seed),
                        "min_score": float(min_score)})
        if prepared is None:
            # None: no mask above min_score or, for a point prompt, SAM3's best
            # mask misses the seed (try_segment_capture). This view measures
            # nothing; the other views still run.
            unmeasured.append({"role": capture.role, "use": use,
                               "reason": "no_mask_above_threshold" if seed is None else "no_mask_at_seed"})
            return None
        evidence = prepared.execution_result
        mask = decode_segmentation_mask_npy(
            prepared.mask_npy, width=int(evidence["image_width"]), height=int(evidence["image_height"]),
            expected_count=int(evidence["mask_pixel_count"]),
            expected_bounds=tuple(int(v) for v in evidence["mask_bounds_xyxy"]))
        points, surface = object_surface_from_mask(mask, capture, maximum_depth_m=self.maximum_depth_m)
        segments.append({"role": capture.role, "use": use, "score": float(evidence["score"]),
                         "prompt": "text" if seed is None else prompt,
                         **({} if seed is None else {"seed_pixel_uv": list(seed), **seed_evidence}),
                         "mask_pixel_count": int(evidence["mask_pixel_count"]),
                         "mask_bounds_xyxy": [int(v) for v in evidence["mask_bounds_xyxy"]],
                         "valid_pixel_count": int(surface["valid_pixel_count"])})
        if points is None:
            unmeasured.append({"role": capture.role, "use": use, "reason": "no_valid_depth"})
        return points

    def observe(self, *, side: str, label: str, language_prompt: str | None, min_score: float,
                expected_station_revision: int, cancel_requested: Callable[[], bool],
                support_plane: SupportPlane | None = None, roles: Sequence[str] | None = None,
                captures=None) -> RealHeldObjectObservation:
        """Measure ``side``'s held object.

        The holding arm's own wrist view is primary. With ``language_prompt``
        it is segmented by that text; without it, from a point prompt at the
        pixel that shows something inside the closing region between the pads
        (``grip_seed_pixel``). No such pixel means nothing is between the
        fingers (``not_between_fingers``, reason
        ``no_object_between_fingers_in_wrist_view``). Only when the primary
        surface is between the fingers is every other view segmented, by the
        same text or from a point prompt where that view sees the primary
        surface (``projected_seed_pixel``); those views only extend the box.
        """
        if side not in self.tool_entity_by_side:
            raise ValueError("Held-object observation needs this arm's reviewed gripper entity.")
        if language_prompt is not None and (not isinstance(language_prompt, str) or not language_prompt.strip()):
            raise ValueError("A held-object text prompt must be nonempty.")
        selected = self.roles if roles is None else tuple(roles)
        primary_role = REAL_HELD_OBJECT_PRIMARY_ROLE_BY_SIDE[side]
        if primary_role not in selected:
            raise ValueError("Held-object observation needs this arm's own wrist camera.")
        if captures is None:
            captures = tuple(self.scene_port.capture_views(
                roles=selected, expected_station_revision=expected_station_revision,
                cancel_requested=cancel_requested))
        if tuple(capture.role for capture in captures) != selected:
            raise RuntimeError("Held-object capture returned other camera roles.")
        for capture in captures:
            if capture.station_revision != expected_station_revision or capture.capture_qpos_by_side is None:
                raise RuntimeError("Held-object capture needs this revision's capture-time joint state.")
        qpos = {s: tuple(float(v) for v in np.asarray(q, dtype=float).reshape(6))
                for s, q in captures[0].capture_qpos_by_side.items()}
        entity = self.tool_entity_by_side[side]

        def tcp_of(capture):
            return xyzquat_to_pose_matrix(self.scene_port.tcp_pose_by_side(capture.capture_qpos_by_side)[side])

        def pose_of(capture):
            return np.asarray(capture.source_from_camera, dtype=float).reshape(4, 4)

        views, segments, unmeasured = [], [], []
        primary = next(capture for capture in captures if capture.role == primary_role)
        others = tuple(capture for capture in captures if capture.role != primary_role)
        if cancel_requested():
            raise InterruptedError("Held-object observation cancelled.")
        tcp = tcp_of(primary)
        seed, seed_evidence, points = None, {}, None
        if language_prompt is None:
            seed, seed_evidence = grip_seed_pixel(
                depth_m=primary.depth_m, intrinsics=primary.intrinsics, world_from_camera=pose_of(primary),
                world_from_tcp=tcp, entity=entity, maximum_depth_m=self.maximum_depth_m)
            if seed is None:
                unmeasured.append({"role": primary_role, "use": "primary",
                                   "reason": "no_object_between_fingers_in_view", **seed_evidence})
        if language_prompt is not None or seed is not None:
            points = self._segment_view(
                primary, language_prompt=language_prompt, seed=seed, seed_evidence=seed_evidence,
                prompt="closing_region_point", use="primary", min_score=min_score,
                segments=segments, unmeasured=unmeasured)
        if points is not None:
            views.append(HeldObjectView(role=primary_role, points_world_m=points, world_from_tcp=tcp))
        confirmed = bool(views) and measure_held_object_box(
            views=views, entity=entity, primary_role=primary_role, support_plane=support_plane).held
        for capture in others:
            if not confirmed:
                unmeasured.append({"role": capture.role, "use": "extension",
                                   "reason": "not_used_without_primary_object"})
                continue
            if cancel_requested():
                raise InterruptedError("Held-object observation cancelled.")
            seed, seed_evidence = None, {}
            if language_prompt is None:
                seed, seed_evidence = projected_seed_pixel(
                    depth_m=capture.depth_m, intrinsics=capture.intrinsics, world_from_camera=pose_of(capture),
                    points_world_m=views[0].points_world_m, maximum_depth_m=self.maximum_depth_m)
                if seed is None:
                    unmeasured.append({"role": capture.role, "use": "extension",
                                       "reason": "primary_object_not_seen_in_view", **seed_evidence})
                    continue
            extension = self._segment_view(
                capture, language_prompt=language_prompt, seed=seed, seed_evidence=seed_evidence,
                prompt="primary_object_point", use="extension", min_score=min_score,
                segments=segments, unmeasured=unmeasured)
            if extension is not None:
                views.append(HeldObjectView(role=capture.role, points_world_m=extension, world_from_tcp=tcp_of(capture)))
        world_from_tcp = xyzquat_to_pose_matrix(self.scene_port.tcp_pose_by_side(qpos)[side])
        measurement = measure_held_object_box(views=views, entity=entity, primary_role=primary_role,
                                              support_plane=support_plane)
        if language_prompt is None and not views and any(
                item["role"] == primary_role and item["reason"] == "no_object_between_fingers_in_view"
                for item in unmeasured):
            # The holding arm's own camera sees the closing region empty: nothing is held there.
            measurement = replace(measurement, status="not_between_fingers",
                                  reason="no_object_between_fingers_in_wrist_view")
        gripper = None
        reader = self.gripper_state_by_side.get(side)
        if reader is not None:
            sample = reader()
            gripper = {"measured_close": float(sample.measured_close), "moving": sample.moving,
                       "fault_code": sample.fault_code, "observed_at": float(sample.observed_at),
                       "evidence_ref": sample.evidence_ref}
            command = self.gripper_command_by_side.get(side)
            if command is not None:
                target, attempt_ref = command()
                gripper["last_command_target_close"] = None if target is None else float(target)
                gripper["last_command_attempt_ref"] = attempt_ref
        return RealHeldObjectObservation(
            side=side, label=label, language_prompt=language_prompt, station_revision=expected_station_revision,
            capture_qpos_by_side=qpos, world_from_tcp=world_from_tcp, measurement=measurement,
            segments=tuple(segments), unmeasured=tuple(unmeasured), gripper=gripper)


__all__ = ["REAL_HELD_OBJECT_PRIMARY_ROLE_BY_SIDE", "REAL_HELD_OBJECT_SEMANTICS_V2", "RealHeldObjectObservation",
           "RealHeldObjectObserver"]
