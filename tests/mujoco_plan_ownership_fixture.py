"""Real planner transactions with an in-memory scene, without geometry or IK."""

import threading
from types import SimpleNamespace

import numpy as np

from sim.mujoco.deploy.planned_tcp_motion import (
    MujocoPlannedTcpMotion,
    MujocoTcpMotionPlan,
)


def pending_chain_planner():
    """Keep real scene/planner locks and qpos copies for ownership-only tests."""
    scene = SimpleNamespace(data=SimpleNamespace(qpos=np.array([0., .1, .2])))
    scene_lock = threading.RLock()

    def capture_scene(project):
        with scene_lock:
            return project(scene)

    rig = SimpleNamespace(scene=scene, capture_scene=capture_scene, scene_lock=scene_lock)
    planner = MujocoPlannedTcpMotion(
        rig=rig, task_config=None, pose_projector=None, reference_hz=60.,
    )
    plan = MujocoTcpMotionPlan(
        path=None, trajectory=None, source_model_qpos=planner._model_qpos(),
        reference_poses=(), trajectory_hash="a" * 64,
    )
    planner._pending_plan = plan
    return planner, plan
