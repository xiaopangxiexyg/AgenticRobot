from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .transforms import pose6_xyz_rpy_to_matrix


@dataclass
class RobotPoseSample:
    T_gripper_to_base: np.ndarray
    pose_xyz_rpy: list[float]
    joint_deg: list[float]
    timestamp_sec: float
    arm_err: int
    sys_err: int
    tool_frame: str = ""


class RM65PoseReader:
    """Read-only RM65 pose reader for calibration."""

    def __init__(self, ip: str, *, retry: int = 2, expected_tool: str = ""):
        from hardware.rm65.rm_api2_adapter import Arm, RM65

        self.ip = str(ip)
        self.retry = int(retry)
        self.expected_tool = str(expected_tool or "")
        self._arm = Arm(RM65, self.ip)
        self._closed = False
        self.current_tool = self._read_tool_frame()
        if self.expected_tool and self.expected_tool not in self.current_tool:
            self.close()
            raise RuntimeError(
                f"expected active RM65 tool containing {self.expected_tool!r}, "
                f"got {self.current_tool!r}."
            )

    def read(self) -> RobotPoseSample:
        code, joint_deg, pose, arm_err, sys_err = self._arm.Get_Current_Arm_State(
            retry=self.retry,
        )
        if int(code) != 0:
            raise RuntimeError(f"RM65 Get_Current_Arm_State failed with code {code}.")
        pose_values = _finite_float_list(pose, expected_len=6, label="pose")
        joint_values = _finite_float_list(joint_deg, expected_len=6, label="joint")
        if int(arm_err) != 0 or int(sys_err) != 0:
            raise RuntimeError(f"RM65 reports arm_err={arm_err}, sys_err={sys_err}.")
        return RobotPoseSample(
            T_gripper_to_base=pose6_xyz_rpy_to_matrix(pose_values),
            pose_xyz_rpy=pose_values,
            joint_deg=joint_values,
            timestamp_sec=time.time(),
            arm_err=int(arm_err),
            sys_err=int(sys_err),
            tool_frame=self.current_tool,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._arm.RM_API_UnInit()
        finally:
            self._arm.Arm_Socket_Close()

    def __enter__(self) -> "RM65PoseReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _read_tool_frame(self) -> str:
        try:
            code, frame = self._arm.Get_Current_Tool_Frame()
        except Exception:
            return ""
        if int(code) != 0:
            if self.expected_tool:
                raise RuntimeError(f"RM65 Get_Current_Tool_Frame failed with code {code}.")
            return ""
        return str(frame).strip("\x00")


def _finite_float_list(values: Iterable[float], *, expected_len: int, label: str) -> list[float]:
    out = [float(value) for value in values]
    if len(out) != expected_len:
        raise RuntimeError(f"RM65 {label} expected {expected_len} values, got {len(out)}.")
    if not all(math.isfinite(value) for value in out):
        raise RuntimeError(f"RM65 {label} contains non-finite values: {out}")
    return out
