import threading
from enum import IntEnum
from dataclasses import dataclass
import numpy as np

num_motors = 35
@dataclass
class MotorCommand:
    q_ref = np.zeros(num_motors)
    dq_ref = np.zeros(num_motors)
    tau_ff = np.zeros(num_motors)
    kp = np.zeros(num_motors)
    kd = np.zeros(num_motors)


@dataclass
class MotorState:
    q = np.zeros(num_motors)
    dq = np.zeros(num_motors)


@dataclass
class BaseState:
    omega = np.zeros(3)
    rpy = np.zeros(3)


class DataBuffer:
    def __init__(self):
        self._data = None
        self.lock = threading.Lock()
    
    @property
    def data(self):
        with self.lock:
            return self._data

    @data.setter
    def data(self, data):
        with self.lock:
            self._data = data
        
class ArmJointIndex(IntEnum):
    # Left arm
    left_shoulder_pitch = 13
    left_shoulder_roll = 14
    left_shoulder_yaw = 15
    left_elbow_pitch = 16
    left_elbow_roll = 17
    left_wrist_pitch = 18
    left_wrist_yaw = 19

    # Right arm
    right_shoulder_pitch = 20
    right_shoulder_roll = 21
    right_shoulder_yaw = 22
    right_elbow_pitch = 23
    right_elbow_roll = 24
    right_wrist_pitch = 25
    right_wrist_yaw = 26


class JointIndex(IntEnum):
    # Left leg
    left_hip_yaw = 0
    left_hip_roll = 1
    left_hip_pitch = 2
    left_knee = 3
    left_ankle = 4
    left_ankle_roll = 5

    # Right leg
    right_hip_yaw = 6
    right_hip_roll = 7
    right_hip_pitch = 8
    right_knee = 9
    right_ankle = 10
    right_ankle_roll = 11

    waist_yaw = 12

    # Left arm
    left_shoulder_pitch = 13
    left_shoulder_roll = 14
    left_shoulder_yaw = 15
    left_elbow_pitch = 16
    left_elbow_roll = 17
    left_wrist_pitch = 18
    left_wrist_yaw = 19

    # Right arm
    right_shoulder_pitch = 20
    right_shoulder_roll = 21
    right_shoulder_yaw = 22
    right_elbow_pitch = 23
    right_elbow_roll = 24
    right_wrist_pitch = 25
    right_wrist_yaw = 26

    not_used_joint = 27
    not_used_joint1 = 28
    not_used_joint2 = 29
    not_used_joint3 = 30
    not_used_joint4 = 31
    not_used_joint5 = 32
    not_used_joint6 = 33
    not_used_joint7 = 34

arm_sdk_used_joint_index = 27