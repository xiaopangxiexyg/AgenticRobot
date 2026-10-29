"""RealMan held-object observation for carried-object motion and grip verification.

One observation captures stationary aligned RGB-D views (each with its
capture-time dual-arm qpos), segments the named object with the deployment's
text segmentation in every view, measures each view's object surface and
fuses them in the holding tool frame (``robot_motion.held_object_measurement``).
The optional gripper sample adds the measured closure. Nothing here moves the
robot, commands a gripper or grants motion authority; a planner only accepts
the resulting box through ``RM65CarriedObjectBox``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from agent_gateway.profiles.segmentation import decode_segmentation_mask_npy
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.held_object_measurement import (
    HeldObjectBox,
    HeldObjectView,
    SupportPlane,
    measure_held_object_box,
)
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.tool_entity_scene import (
    RELATION_MAXIMUM_DEPTH_M,
    object_surface_from_mask,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65CarriedObjectBox

REAL_HELD_OBJECT_SEMANTICS_V1 = "agenticrobot.real.held_object_observation.v1"
# A closed gripper that stopped clearly before its full-close position holds
# something between the pads; a fully closed one holds nothing graspable.
REAL_HELD_OBJECT_FULL_CLOSE_MARGIN = 0.03


@dataclass(frozen=True)
class RealHeldObjectObservation:
    """One measured held-object cut of one arm."""

    side: str
    label: str
    language_prompt: str
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
            self._payload(), domain="agenticrobot.real.held_object_observation.v1"))

    def _payload(self) -> dict:
        return {"semantics": REAL_HELD_OBJECT_SEMANTICS_V1, "side": self.side, "label": self.label,
                "language_prompt": self.language_prompt, "station_revision": self.station_revision,
                "capture_qpos_by_side": {k: list(v) for k, v in self.capture_qpos_by_side.items()},
                "world_from_tcp": self.world_from_tcp.tolist(), "measurement": self.measurement.as_dict(),
                "segments": [dict(s) for s in self.segments], "unmeasured": [dict(u) for u in self.unmeasured],
                "gripper": None if self.gripper is None else dict(self.gripper)}

    @property
    def gripper_closed_on_object(self) -> bool | None:
        """Measured closure stopped before full close (None without a gripper sample)."""
        if self.gripper is None or self.gripper.get("measured_close") is None:
            return None
        return float(self.gripper["measured_close"]) < 1.0 - REAL_HELD_OBJECT_FULL_CLOSE_MARGIN

    @property
    def holding_verified(self) -> bool:
        """Between the fingers in a measured view and, when sampled, closed on something."""
        return self.measurement.held and self.gripper_closed_on_object is not False

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
    sample per side without commanding it.
    """

    def __init__(self, *, scene_port, segmentation_provider, tool_entity_by_side: Mapping[str, object],
                 roles: Sequence[str], gripper_state_by_side: Mapping[str, Callable[[], object]] | None = None,
                 maximum_depth_m: float = RELATION_MAXIMUM_DEPTH_M):
        if not callable(getattr(scene_port, "capture_views", None)):
            raise TypeError("scene_port must capture stationary views.")
        if not callable(getattr(segmentation_provider, "try_segment_capture", None)):
            raise TypeError("segmentation_provider must segment captures.")
        if not tool_entity_by_side or not roles:
            raise ValueError("Held-object observation needs a gripper entity and camera roles.")
        if not math.isfinite(maximum_depth_m) or maximum_depth_m <= 0:
            raise ValueError("maximum_depth_m must be positive.")
        self.scene_port = scene_port
        self.segmentation_provider = segmentation_provider
        self.tool_entity_by_side = dict(tool_entity_by_side)
        self.roles = tuple(roles)
        self.gripper_state_by_side = dict(gripper_state_by_side or {})
        self.maximum_depth_m = float(maximum_depth_m)

    def observe(self, *, side: str, label: str, language_prompt: str, min_score: float,
                expected_station_revision: int, cancel_requested: Callable[[], bool],
                support_plane: SupportPlane | None = None, roles: Sequence[str] | None = None,
                captures=None) -> RealHeldObjectObservation:
        if side not in self.tool_entity_by_side:
            raise ValueError("Held-object observation needs this arm's reviewed gripper entity.")
        if not isinstance(language_prompt, str) or not language_prompt.strip():
            raise ValueError("Held-object observation needs the object's text prompt.")
        selected = self.roles if roles is None else tuple(roles)
        if captures is None:
            captures = tuple(self.scene_port.capture_views(
                roles=selected, expected_station_revision=expected_station_revision,
                cancel_requested=cancel_requested))
        if tuple(capture.role for capture in captures) != selected:
            raise RuntimeError("Held-object capture returned other camera roles.")
        entity = self.tool_entity_by_side[side]
        views, segments, unmeasured = [], [], []
        qpos = None
        for capture in captures:
            if capture.station_revision != expected_station_revision or capture.capture_qpos_by_side is None:
                raise RuntimeError("Held-object capture needs this revision's capture-time joint state.")
            if qpos is None:
                qpos = {s: tuple(float(v) for v in np.asarray(q, dtype=float).reshape(6))
                        for s, q in capture.capture_qpos_by_side.items()}
            if cancel_requested():
                raise InterruptedError("Held-object observation cancelled.")
            prepared = self.segmentation_provider.try_segment_capture(
                capture=self.scene_port.segmentation_input(capture),
                parameters={"role": capture.role, "language_prompt": language_prompt,
                            "target_pixel_uv": None, "min_score": float(min_score)})
            if prepared is None:
                unmeasured.append({"role": capture.role, "reason": "no_mask_above_threshold"})
                continue
            evidence = prepared.execution_result
            mask = decode_segmentation_mask_npy(
                prepared.mask_npy, width=int(evidence["image_width"]), height=int(evidence["image_height"]),
                expected_count=int(evidence["mask_pixel_count"]),
                expected_bounds=tuple(int(v) for v in evidence["mask_bounds_xyxy"]))
            points, surface = object_surface_from_mask(mask, capture, maximum_depth_m=self.maximum_depth_m)
            segments.append({"role": capture.role, "score": float(evidence["score"]),
                             "mask_pixel_count": int(evidence["mask_pixel_count"]),
                             "mask_bounds_xyxy": [int(v) for v in evidence["mask_bounds_xyxy"]],
                             "valid_pixel_count": int(surface["valid_pixel_count"])})
            if points is None:
                unmeasured.append({"role": capture.role, "reason": "no_valid_depth"})
                continue
            tcp = self.scene_port.tcp_pose_by_side(capture.capture_qpos_by_side)[side]
            views.append(HeldObjectView(role=capture.role, points_world_m=points,
                                        world_from_tcp=xyzquat_to_pose_matrix(tcp)))
        if qpos is None:
            raise RuntimeError("Held-object observation captured no view.")
        world_from_tcp = xyzquat_to_pose_matrix(self.scene_port.tcp_pose_by_side(qpos)[side])
        measurement = measure_held_object_box(views=views, entity=entity, support_plane=support_plane)
        gripper = None
        reader = self.gripper_state_by_side.get(side)
        if reader is not None:
            sample = reader()
            gripper = {"measured_close": float(sample.measured_close), "moving": sample.moving,
                       "fault_code": sample.fault_code, "observed_at": float(sample.observed_at),
                       "evidence_ref": sample.evidence_ref}
        return RealHeldObjectObservation(
            side=side, label=label, language_prompt=language_prompt, station_revision=expected_station_revision,
            capture_qpos_by_side=qpos, world_from_tcp=world_from_tcp, measurement=measurement,
            segments=tuple(segments), unmeasured=tuple(unmeasured), gripper=gripper)


__all__ = ["REAL_HELD_OBJECT_SEMANTICS_V1", "RealHeldObjectObservation", "RealHeldObjectObserver"]
