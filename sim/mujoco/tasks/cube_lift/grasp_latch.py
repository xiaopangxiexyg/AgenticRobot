"""Temporary contact-gated grasp constraint for the debug Cube-Lift task."""

from __future__ import annotations

import numpy as np

from ...scene import MujocoScene
from ...transforms import matrix_to_quat_wxyz, quat_wxyz_to_matrix


class ContactGraspLatch:
    """Carry the free cube after a bilateral, aligned gripper contact.

    This is intentionally task-local scaffolding for the non-identified
    direct-qpos environment. It is not a physical grasp model and must never be
    shared with real/deployment code.
    """

    def __init__(
        self,
        scene: MujocoScene,
        *,
        object_body_name: str,
        object_freejoint_name: str,
        grasp_body_name: str,
        left_finger_body_name: str,
        right_finger_body_name: str,
        acquire_close: float,
        release_close: float,
        acquire_distance_m: float,
    ) -> None:
        self.scene = scene
        self.object_freejoint_name = str(object_freejoint_name)
        self.object_body_id = scene.body_id(object_body_name)
        self.grasp_body_id = scene.body_id(grasp_body_name)
        self.left_finger_body_id = scene.body_id(left_finger_body_name)
        self.right_finger_body_id = scene.body_id(right_finger_body_name)
        self.acquire_close = float(acquire_close)
        self.release_close = float(release_close)
        self.acquire_distance_m = float(acquire_distance_m)
        self.active = False
        self.acquisition_count = 0
        self._grasp_to_cube = np.eye(4, dtype=np.float64)

    def reset(self) -> None:
        self.active = False
        self.acquisition_count = 0
        self._grasp_to_cube[:] = np.eye(4, dtype=np.float64)

    def update(self, gripper_close: float) -> bool:
        close = float(gripper_close)
        if self.active:
            if close <= self.release_close:
                self.active = False
                return False
            self._apply()
            return True
        if close < self.acquire_close:
            return False
        cube_pose = self._cube_pose()
        grasp_pose = self._grasp_pose()
        if np.linalg.norm(cube_pose[:3, 3] - grasp_pose[:3, 3]) > self.acquire_distance_m:
            return False
        left_contact = False
        right_contact = False
        body_ids = self.scene.model.geom_bodyid
        for index in range(int(self.scene.data.ncon)):
            contact = self.scene.data.contact[index]
            first = int(body_ids[int(contact.geom1)])
            second = int(body_ids[int(contact.geom2)])
            pair = {first, second}
            if self.object_body_id not in pair:
                continue
            left_contact = left_contact or self.left_finger_body_id in pair
            right_contact = right_contact or self.right_finger_body_id in pair
        if not (left_contact and right_contact):
            return False
        self._grasp_to_cube[:] = np.linalg.inv(grasp_pose) @ cube_pose
        self.active = True
        self.acquisition_count += 1
        self._apply()
        return True

    def _grasp_pose(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = self.scene.data.xpos[self.grasp_body_id]
        pose[:3, :3] = self.scene.data.xmat[self.grasp_body_id].reshape(3, 3)
        return pose

    def _cube_pose(self) -> np.ndarray:
        qpos = self.scene.freejoint_qpos(self.object_freejoint_name)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = qpos[:3]
        pose[:3, :3] = quat_wxyz_to_matrix(qpos[3:])
        return pose

    def _apply(self) -> None:
        pose = self._grasp_pose() @ self._grasp_to_cube
        qpos = np.concatenate(
            (pose[:3, 3], matrix_to_quat_wxyz(pose[:3, :3]))
        )
        self.scene.set_freejoint_state(
            self.object_freejoint_name,
            qpos=qpos,
            qvel=np.zeros(6, dtype=np.float64),
            forward=True,
        )


__all__ = ["ContactGraspLatch"]
