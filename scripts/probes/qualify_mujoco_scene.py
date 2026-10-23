"""Zero-motion scene qualification using the original prepared target/oracle.

This probe creates no execution owner, renderer, observation or action target.
Its geometry report is not a motion, HOLD, grasp or task-success receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.mujoco.deploy.target_config import (
    DEFAULT_STATION_POSE_CONTRACT_PATH,
    MujocoSimHILTargetConfig,
)
from sim.mujoco.deploy.target_preparation import (
    MujocoSimHILPreparedTarget,
    prepare_mujoco_sim_hil_target,
)


def inspect_prepared_scene(prepared: MujocoSimHILPreparedTarget) -> dict:
    """Read the initial full-robot geometry; do not advance or edit MjData."""

    if prepared.config.allow_scripted_task_reset or any(
        prepared.task_initialization_info.get(key, False)
        for key in ("scripted_pregrasp_reset", "scripted_contact_reset")
    ):
        raise ValueError("Zero-motion qualification requires scripted reset disabled.")
    if not prepared.config.collision_stop_enabled:
        raise ValueError("Zero-motion qualification requires collision protection.")
    scene = prepared.scene
    if not scene.state_is_finite():
        raise ValueError("Initial simulator state is non-finite.")
    projector = scene.collision_projector()
    if projector is None:
        raise ValueError("Scene has no collision oracle.")
    before = (float(scene.data.time), scene.data.qpos.copy(), scene.data.qvel.copy())
    if before[0] != 0.0:
        raise ValueError("Initial geometry must be inspected before physics advances.")
    # Use the original preventive threshold as the distance query radius;
    # returned clearance can be clipped and must not be called exact distance.
    preventive = projector.oracle.check(
        scene.data, distance_query_limit_m=projector.policy.minimum_clearance_m,
    )
    hard_contact = projector.oracle.check_contacts(scene.data)
    # Physical object/support contact is intentionally not a forbidden robot
    # pair. Still reject an invalid fixture spawned beyond the same hard bound.
    object_penetrations = []
    for contact in scene.contacts():
        geom_ids = [scene.mujoco.mj_name2id(
            scene.model, scene.mujoco.mjtObj.mjOBJ_GEOM, name,
        ) for name in (contact.geom1_name, contact.geom2_name)]
        first, second = (int(scene.model.geom_contype[index]) for index in geom_ids)
        if 8 not in (first, second) or first not in (1, 8) or second not in (1, 8):
            continue
        if contact.distance_m < -projector.policy.hard_penetration_m:
            object_penetrations.append({
                "geom1": contact.geom1_name, "geom2": contact.geom2_name,
                "distance_m": float(contact.distance_m),
            })
    unchanged = (
        before[0] == float(scene.data.time)
        and np.array_equal(before[1], scene.data.qpos)
        and np.array_equal(before[2], scene.data.qvel)
    )
    if not unchanged:
        raise RuntimeError("Geometry inspection changed the initial simulator state.")
    return {
        "qualified": bool(preventive.safe and hard_contact.safe and not object_penetrations),
        "scope": "initial_geometry_only",
        "robot_preventive": preventive.to_dict(),
        "robot_hard_contact": hard_contact.to_dict(),
        "minimum_clearance_m": projector.policy.minimum_clearance_m,
        "hard_penetration_m": projector.policy.hard_penetration_m,
        "object_penetrations": object_penetrations,
        "forbidden_pair_count": projector.oracle.forbidden_pair_count,
        "time_sec": before[0],
        "time_qpos_qvel_unchanged": unchanged,
        "execution_owner_created": False,
        "renderer_created": False,
        "motion_executed": False,
        "hold_proved": False,
        "physical_task_success": "not_assessed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_config", type=Path)
    parser.add_argument("--station-pose-contract", type=Path, default=DEFAULT_STATION_POSE_CONTRACT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve the output first. Never overwrite a retained qualification.
    with args.output.open("x") as stream:
        try:
            prepared = prepare_mujoco_sim_hil_target(MujocoSimHILTargetConfig(
                task_config_path=args.task_config,
                station_pose_contract_path=args.station_pose_contract,
                realtime_worker=False,
                allow_scripted_task_reset=False,
            ))
            report = inspect_prepared_scene(prepared)
        except Exception as error:
            json.dump({"qualified": False, "scope": "initial_geometry_only",
                "error": {"type": type(error).__name__, "message": str(error)}}, stream,
                indent=2, allow_nan=False)
            stream.flush()
            raise
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
