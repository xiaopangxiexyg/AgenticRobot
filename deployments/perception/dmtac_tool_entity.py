"""Bind the reviewed DMTAC gripper as a TCP-attached tool entity.

The RealMan Gateway, the MuJoCo Gateway (same station URDF and tool
calibrations) and the offline saved-scene probe all bind gripper geometry
here, so relation and hover answers use one entity definition.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from deployments.perception.dmtac_grasp_overlay import (
    DMTAC_M_REVIEWED_TOOL_TCP_POSE_FLANGE_M_RAD,
    dmtac_graspgenx_alignment_profile,
)
from robot_motion.tool_entity_goals import ToolEntityBox


def _reviewed_tcp(tcp) -> bool:
    return bool(np.allclose(np.asarray(tcp, dtype=float), DMTAC_M_REVIEWED_TOOL_TCP_POSE_FLANGE_M_RAD,
                            atol=1e-9, rtol=0.))


def dmtac_open_sensor_contact_solids(urdf_path, tool_tcp_pose_flange_m_rad_by_side):
    """URDF DMTAC sensor collision boxes at the neutral (open) gripper, per TCP frame.

    The grasp-screening finger solids are 19.9 mm thick; the station URDF's
    sensor collision boxes are 48 mm along the closing axis and reach about
    30 mm beyond them when open (identical in the MuJoCo twin). They are
    returned as contact-only solids for arms with the reviewed TCP; an arm
    without exactly two such boxes gets none.
    """
    import pinocchio as pin

    path = Path(urdf_path).resolve(strict=True)
    model = pin.buildModelFromUrdf(str(path))
    geometry = pin.buildGeomFromUrdf(model, str(path), pin.GeometryType.COLLISION,
                                     package_dirs=[str(path.parent), str(path.parent.parent)])
    data, geometry_data = model.createData(), pin.GeometryData(geometry)
    q = pin.neutral(model)
    # The v5 screening alignment is defined at gripper_master_q_rad == 0 (its
    # loader enforces it); require q=0 to be the open end of every gripper joint.
    for joint_id in range(1, model.njoints):
        name = model.names[joint_id]
        if "_eg2_4b2_cad_" in name and model.joints[joint_id].nq == 1:
            index = model.joints[joint_id].idx_q
            if not np.isclose(model.lowerPositionLimit[index], 0.) or not np.isclose(q[index], 0.):
                raise ValueError("Open DMTAC sensor boxes require q=0 at the open joint limit.")
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.updateGeometryPlacements(model, data, geometry, geometry_data, q)
    result = {}
    for side, tcp in tool_tcp_pose_flange_m_rad_by_side.items():
        if not _reviewed_tcp(tcp):
            continue
        frame = side + "_arm_link6"
        if not model.existFrame(frame):
            continue
        flange_from_tcp = np.eye(4)
        flange_from_tcp[:3, :3] = Rotation.from_euler("xyz", np.asarray(tcp[3:], dtype=float)).as_matrix()
        flange_from_tcp[:3, 3] = np.asarray(tcp[:3], dtype=float)
        world_from_tcp = data.oMf[model.getFrameId(frame)].homogeneous @ flange_from_tcp
        solids = []
        for index, item in enumerate(geometry.geometryObjects):
            half = getattr(item.geometry, "halfSide", None)
            if half is None or not item.name.startswith(side + "_gripper_") or "dm_tac_w2m_sensor" not in item.name:
                continue
            pose = np.linalg.inv(world_from_tcp) @ geometry_data.oMg[index].homogeneous
            solids.append((tuple(pose.reshape(-1)), tuple(float(x) for x in np.asarray(half))))
        if len(solids) == 2:
            result[side] = tuple(solids)
    return result


def default_tool_entity_by_side(tool_tcp_pose_flange_m_rad_by_side, tool_profile_by_side, *, urdf_path=None):
    """Bind the nominal grasp-screening box only for a matching tool profile.

    The DMTAC GraspGenX alignment v5 solid frame is defined relative to the
    reviewed 140 mm tool TCP. The frozen profile must identify DMTAC M on the
    same side as well; this nominal match does not prove physical installation.
    Any other arm gets no entity, so entity queries fail instead of
    reinterpreting the box. With ``urdf_path`` the URDF's open DMTAC sensor
    boxes join as contact-only solids for tool-object relations; hover
    geometry is unchanged.
    """
    alignment = dmtac_graspgenx_alignment_profile()
    geometry = alignment.collision_geometry(maximum_opening_m=alignment.nominal_geometry.maximum_opening_m)
    source = {"geometry": "graspgenx_screening_solids_maximum_opening_envelope",
              "alignment_profile_id": alignment.profile_id,
              "alignment_source_sha256": alignment.source_content_hash,
              "collision_tip_extension_m": alignment.collision_tip_extension_m,
              "tool_tcp_pose_flange_m_rad": list(DMTAC_M_REVIEWED_TOOL_TCP_POSE_FLANGE_M_RAD),
              "opening_policy": "maximum_opening_envelope_v1"}
    result = {}
    contact = ({} if urdf_path is None
               else dmtac_open_sensor_contact_solids(urdf_path, tool_tcp_pose_flange_m_rad_by_side))
    for side, tcp in tool_tcp_pose_flange_m_rad_by_side.items():
        profile = tool_profile_by_side.get(side)
        if (isinstance(profile, Mapping) and profile.get("arm") == side
                and profile.get("tool_name") == "DMTAC_M" and _reviewed_tcp(tcp)):
            profile_identity = canonical_content_hash(
                {key: value for key, value in profile.items() if not str(key).startswith("_")},
                domain="agenticrobot.saved_tool_profile.v1")
            entity = ToolEntityBox.from_parallel_jaw_geometry(geometry, source={
                **source, "side": side, "tool_name": "DMTAC_M", "tool_profile_hash": profile_identity})
            if side in contact:
                entity = dataclasses.replace(entity, contact_solids=contact[side], source={
                    **entity.source, "contact_solids": "urdf_dmtac_sensor_collision_boxes_open_v1",
                    "contact_solids_urdf": Path(urdf_path).name})
            result[side] = entity
    return result


__all__ = ["default_tool_entity_by_side", "dmtac_open_sensor_contact_solids"]
