from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .paths import require_asset_path


def _to_pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (7,):
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
        matrix[:3, 3] = pose[:3]
        return matrix
    raise ValueError(f"Unsupported pose shape: {pose.shape}")


class SingleArmIKSolver:
    def __init__(self, urdf_path: str | Path | None = None, eef_name: str = "hand_base"):
        try:
            import pinocchio as pin
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SingleArmIKSolver requires pinocchio to be importable."
            ) from exc

        if urdf_path is None:
            urdf_path = require_asset_path("rm65_arm.urdf")

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.eef_name = eef_name
        self.frame_names = tuple(frame.name for frame in self.model.frames)
        self.eef_frame_id = self.model.getFrameId(eef_name)
        if self.eef_frame_id >= len(self.model.frames):
            raise ValueError(
                f"Frame '{eef_name}' was not found in URDF '{urdf_path}'. "
                f"Available frames include: {self.frame_names[:20]}"
            )
        self.num_dofs = self.model.nv
        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()

    def compute_pose(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=float)
        self.pin.forwardKinematics(self.model, self.data, qpos)
        self.pin.updateFramePlacements(self.model, self.data)
        transform = self.data.oMf[self.eef_frame_id]
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = transform.rotation
        pose[:3, 3] = transform.translation
        return pose

    def compute_jacobian(self, qpos: np.ndarray, reference_frame: int | None = None) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=float)
        if reference_frame is None:
            reference_frame = self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        self.pin.computeJointJacobians(self.model, self.data, qpos)
        return self.pin.computeFrameJacobian(
            self.model,
            self.data,
            qpos,
            self.eef_frame_id,
            reference_frame,
        )

    def solve_step(
        self,
        target_pose: np.ndarray,
        current_qpos: np.ndarray,
        max_joint_step: float | np.ndarray = 0.05,
        damping: float = 0.1,
    ) -> np.ndarray:
        target_pose = _to_pose_matrix(target_pose)
        current_qpos = np.asarray(current_qpos, dtype=float)

        current_pose = self.compute_pose(current_qpos)
        pos_err = target_pose[:3, 3] - current_pose[:3, 3]
        rot_err = self.pin.log3(target_pose[:3, :3] @ current_pose[:3, :3].T)
        error = np.concatenate([pos_err, rot_err])

        jacobian = self.compute_jacobian(current_qpos)
        regularizer = (damping**2) * np.eye(6, dtype=float)
        joint_update = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + regularizer, error)

        max_joint_step = np.asarray(max_joint_step, dtype=float)
        if max_joint_step.shape == ():
            max_joint_step = np.full(self.num_dofs, float(max_joint_step), dtype=float)
        target_qpos = current_qpos + np.clip(joint_update, -max_joint_step, max_joint_step)
        return np.clip(target_qpos, self.q_min, self.q_max)

    def solve_iterative(
        self,
        target_pose: np.ndarray,
        q_init: np.ndarray | None = None,
        max_iterations: int = 100,
        max_joint_step: float | np.ndarray = 0.1,
        damping: float = 0.1,
        tolerance: float = 1e-4,
    ) -> np.ndarray:
        target_pose = _to_pose_matrix(target_pose)
        q = np.zeros(self.num_dofs, dtype=float) if q_init is None else np.asarray(q_init, dtype=float)

        for _ in range(max_iterations):
            current_pose = self.compute_pose(q)
            pos_err = target_pose[:3, 3] - current_pose[:3, 3]
            rot_err = self.pin.log3(target_pose[:3, :3] @ current_pose[:3, :3].T)
            error = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(error) < tolerance:
                break
            q = self.solve_step(
                target_pose=target_pose,
                current_qpos=q,
                max_joint_step=max_joint_step,
                damping=damping,
            )
        return q
