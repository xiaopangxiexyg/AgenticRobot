from __future__ import annotations

from typing import Iterable

from .Robotic_Arm.rm_robot_interface import (
    RM_MOVE_MULTI_BLOCK,
    RM_MOVE_NBLOCK,
    RoboticArm,
    rm_force_position_t,
    rm_force_position_move_t,
    rm_multi_drag_teach_t,
    rm_api_version,
    rm_frame_t,
    rm_inverse_kinematics_params_t,
    rm_thread_mode_e,
)


RM65 = 65
RM65_DOF = 6
LEFT_ARM_IP = "192.168.101.21"
RIGHT_ARM_IP = "192.168.1.20"


def _as_float_list(values: Iterable[float]) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return [float(values)]
    out: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out.extend(_as_float_list(value))
        else:
            out.append(float(value))
    return out


def _as_int_list(values: Iterable[float]) -> list[int]:
    return [int(value) for value in _as_float_list(values)]


def _force_list(values: object) -> list[float]:
    if values is None:
        return []
    return [float(value) for value in list(values)]


def _fixed_float_list(values: Iterable[float], length: int, label: str) -> list[float]:
    out = _as_float_list(values)
    if len(out) != length:
        raise ValueError(f"{label} must contain exactly {length} values, got {len(out)}")
    return out


def _six_int_list(values: Iterable[float], label: str) -> list[int]:
    out = _as_int_list(values)
    if len(out) != 6:
        raise ValueError(f"{label} must contain exactly 6 values, got {len(out)}")
    return out


class Arm:
    """Compatibility wrapper over RealMan API2's ``RoboticArm``.

    The old code in this repo imports ``Arm(RM65, ip)`` and calls API1-style
    method names.  API2 uses ``RoboticArm`` plus ``rm_*`` methods, so this class
    preserves the local call surface while loading the newer SDK and libraries.
    """

    def __init__(
        self,
        dev_mode: int,
        ip: str,
        *,
        port: int = 8080,
        log_level: int = 3,
        thread_mode: rm_thread_mode_e = rm_thread_mode_e.RM_TRIPLE_MODE_E,
    ):
        self.dev_mode = dev_mode
        self.ip = ip
        self.port = port
        self._closed = False
        self._robot = RoboticArm(thread_mode)
        self._handle = self._robot.rm_create_robot_arm(ip, port, level=log_level)
        self.nSocket = getattr(self._handle, "id", -1)
        if getattr(self._robot, "arm_dof", 0) == 0 and dev_mode == RM65:
            self._robot.arm_dof = RM65_DOF

    @property
    def arm_dof(self) -> int:
        dof = int(getattr(self._robot, "arm_dof", 0) or 0)
        return dof if dof > 0 else RM65_DOF

    def _joint_for_dof(self, joint: Iterable[float]) -> list[float]:
        values = _as_float_list(joint)
        if len(values) < self.arm_dof:
            values = values + [0.0] * (self.arm_dof - len(values))
        return values[: self.arm_dof]

    def _joint_for_canfd(self, joint: Iterable[float]) -> list[float]:
        values = _as_float_list(joint)
        if len(values) < 7:
            values = values + [0.0] * (7 - len(values))
        return values[:7]

    def API_Version(self) -> str:
        api_version = rm_api_version()
        if isinstance(api_version, bytes):
            return api_version.decode("utf-8")
        return str(api_version)

    def Arm_Socket_State(self) -> int:
        return 0 if self.nSocket != -1 else 5

    def Get_Current_Arm_State(self, retry: int = 0):
        ret, state = self._robot.rm_get_current_arm_state()
        while ret and retry:
            ret, state = self._robot.rm_get_current_arm_state()
            retry -= 1
        joint = _as_float_list(state.get("joint", []))[: self.arm_dof]
        pose = _as_float_list(state.get("pose", []))[:6]
        err = state.get("err", {}) or {}
        err_values = err.get("err", []) or []
        arm_err = int(err_values[0]) if err_values else 0
        sys_err = 0
        return ret, joint, pose, arm_err, sys_err

    def Get_Force_Data(self):
        ret, data = self._robot.rm_get_force_data()
        return (
            ret,
            _force_list(data.get("force_data")),
            _force_list(data.get("zero_force_data")),
            _force_list(data.get("work_zero_force_data")),
            _force_list(data.get("tool_zero_force_data")),
        )

    def Get_Install_Pose(self):
        return self._robot.rm_get_install_pose()

    def Set_Install_Pose(self, x, y, z):
        return self._robot.rm_set_install_pose(float(x), float(y), float(z))

    def Get_Current_Tool_Frame(self):
        return self._robot.rm_get_current_tool_frame()

    def Get_Total_Tool_Frame(self):
        return self._robot.rm_get_total_tool_frame()

    def Get_Given_Tool_Frame(self, name):
        return self._robot.rm_get_given_tool_frame(str(name))

    def Manual_Set_Tool_Frame(self, name, pose, payload, x, y, z):
        frame = rm_frame_t(
            str(name),
            tuple(_as_float_list(pose)[:6]),
            float(payload),
            float(x),
            float(y),
            float(z),
        )
        return self._robot.rm_set_manual_tool_frame(frame)

    def Update_Tool_Frame(self, name, pose, payload, x, y, z):
        frame = rm_frame_t(
            str(name),
            tuple(_as_float_list(pose)[:6]),
            float(payload),
            float(x),
            float(y),
            float(z),
        )
        return self._robot.rm_update_tool_frame(frame)

    def Change_Tool_Frame(self, name):
        return self._robot.rm_change_tool_frame(str(name))

    def Set_Force_Sensor(self, block=True):
        return self._robot.rm_set_force_sensor(bool(block))

    def Manual_Set_Force(self, point_num, joint, block=True):
        return self._robot.rm_manual_set_force(
            int(point_num),
            self._joint_for_dof(joint),
            bool(block),
        )

    def Stop_Set_Force_Sensor(self):
        return self._robot.rm_stop_set_force_sensor()

    def Movej_CANFD(self, joint, follow, expand=0, trajectory_mode=0, radio=0):
        return self._robot.rm_movej_canfd(
            self._joint_for_canfd(joint),
            bool(follow),
            float(expand),
            int(trajectory_mode),
            int(radio),
        )

    def Algo_Inverse_Kinematics(self, q_in, pose, flag=1):
        """Solve IK with RealMan's local algorithm library.

        Args:
            q_in: Seed joint angles in degrees.
            pose: Target pose. Use [x, y, z, rx, ry, rz] when flag=1.
            flag: 1 for Euler pose, 0 for quaternion pose.
        """
        q_seed = self._joint_for_canfd(q_in)
        pose_values = _as_float_list(pose)
        pose_len = 7 if int(flag) == 0 else 6
        params = rm_inverse_kinematics_params_t(q_seed, pose_values[:pose_len], int(flag))
        return self._robot.rm_algo_inverse_kinematics(params)

    def Algo_Forward_Kinematics(self, joint, flag=1):
        return self._robot.rm_algo_forward_kinematics(self._joint_for_dof(joint), int(flag))

    def Algo_Kin_Singularity_Analyse(self, joint):
        return self._robot.rm_algo_kin_robot_singularity_analyse(self._joint_for_dof(joint))

    def Set_MoveV_CANFD_Init(self, avoid_singularity_flag=1, frame_type=1, dt=20):
        return self._robot.rm_set_movev_canfd_init(
            int(avoid_singularity_flag),
            int(frame_type),
            int(dt),
        )

    def MoveV_CANFD(self, cartesian_velocity, follow=False, trajectory_mode=0, radio=0):
        return self._robot.rm_movev_canfd(
            _fixed_float_list(cartesian_velocity, 6, "cartesian_velocity"),
            bool(follow),
            int(trajectory_mode),
            int(radio),
        )

    def Movep_CANFD(self, pose, follow=False, trajectory_mode=0, radio=0):
        pose_values = _as_float_list(pose)
        if len(pose_values) not in (6, 7):
            raise ValueError(f"pose must contain 6 euler or 7 quaternion values, got {len(pose_values)}")
        return self._robot.rm_movep_canfd(
            pose_values,
            bool(follow),
            int(trajectory_mode),
            int(radio),
        )

    def Movej_Cmd(self, joint, v, r=0, connect=0, block=True):
        block_mode = RM_MOVE_MULTI_BLOCK if block else RM_MOVE_NBLOCK
        return self._robot.rm_movej(
            self._joint_for_dof(joint),
            int(v),
            int(r),
            int(connect),
            block_mode,
        )

    def Movel_Cmd(self, pose, v, r=0, connect=0, block=True):
        block_mode = RM_MOVE_MULTI_BLOCK if block else RM_MOVE_NBLOCK
        return self._robot.rm_movel(
            _as_float_list(pose)[:6],
            int(v),
            int(r),
            int(connect),
            block_mode,
        )

    def Movel_Offset_Cmd(self, offset, v, r=0, connect=0, frame_type=1, block=True):
        """Cartesian offset move.

        RealMan API2 documents this as a fourth-generation-controller feature;
        prefer absolute Movel_Cmd on RM65 third-generation controllers.
        """
        block_mode = RM_MOVE_MULTI_BLOCK if block else RM_MOVE_NBLOCK
        return self._robot.rm_movel_offset(
            _as_float_list(offset)[:6],
            int(v),
            int(r),
            int(connect),
            int(frame_type),
            block_mode,
        )

    def Arm_Slow_Stop(self):
        return self._robot.rm_set_arm_slow_stop()

    def Arm_Stop(self):
        return self._robot.rm_set_arm_stop()

    def Set_Hand_Angle(self, hand_angle, block=True, timeout=10):
        return self._robot.rm_set_hand_angle(_as_int_list(hand_angle), bool(block), int(timeout))

    def Clear_Force_Data(self, block=True):
        return self._robot.rm_clear_force_data()

    def Start_Force_Position_Move(self, block=True):
        return self._robot.rm_start_force_position_move()

    def Stop_Force_Position_Move(self, block=True):
        return self._robot.rm_stop_force_position_move()

    def Set_Force_Position(self, sensor, mode, direction, force):
        return self._robot.rm_set_force_position(
            int(sensor),
            int(mode),
            int(direction),
            float(force),
        )

    def Set_Force_Position_New(
        self,
        *,
        sensor=1,
        mode=1,
        control_mode=None,
        desired_force=None,
        limit_vel=None,
    ):
        if control_mode is None:
            control_mode = [0, 0, 0, 0, 0, 0]
        if desired_force is None:
            desired_force = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if limit_vel is None:
            limit_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        param = rm_force_position_t()
        param.sensor = int(sensor)
        param.mode = int(mode)
        param.control_mode = type(param.control_mode)(*_six_int_list(control_mode, "control_mode"))
        param.desired_force = type(param.desired_force)(
            *_fixed_float_list(desired_force, 6, "desired_force")
        )
        param.limit_vel = type(param.limit_vel)(*_fixed_float_list(limit_vel, 6, "limit_vel"))
        return self._robot.rm_set_force_position_new(param)

    def Stop_Force_Position(self):
        return self._robot.rm_stop_force_position()

    def Set_Force_Drag_Mode(self, mode):
        return self._robot.rm_set_force_drag_mode(int(mode))

    def Get_Force_Drag_Mode(self):
        return self._robot.rm_get_force_drag_mode()

    def Start_Multi_Drag_Teach(self, mode, singular_wall=1):
        return self._robot.rm_start_multi_drag_teach(int(mode), int(singular_wall))

    def Start_Multi_Drag_Teach_New(self, free_axes, frame=1, singular_wall=1):
        axes = _as_int_list(free_axes)
        if len(axes) != 6:
            raise ValueError("free_axes must contain exactly 6 values")
        param = rm_multi_drag_teach_t()
        param.free_axes = (type(param.free_axes))(*axes)
        param.frame = int(frame)
        param.singular_wall = int(singular_wall)
        return self._robot.rm_start_multi_drag_teach_new(param)

    def Stop_Drag_Teach(self):
        return self._robot.rm_stop_drag_teach()

    def Force_Position_Move_Joint(self, joint, sensor, mode, dir, force, follow):
        return self._robot.rm_force_position_move_joint(
            self._joint_for_dof(joint),
            int(sensor),
            int(mode),
            int(dir),
            float(force),
            bool(follow),
        )

    def Force_Position_Move_Pose(self, pose, sensor, mode, dir, force, follow):
        return self._robot.rm_force_position_move_pose(
            _as_float_list(pose),
            int(sensor),
            int(mode),
            int(dir),
            float(force),
            bool(follow),
        )

    def Force_Position_Move_New(
        self,
        *,
        flag,
        pose=None,
        joint=None,
        sensor=1,
        mode=1,
        follow=False,
        control_mode=None,
        desired_force=None,
        limit_vel=None,
        trajectory_mode=0,
        radio=0,
    ):
        if control_mode is None:
            control_mode = [0, 0, 0, 0, 0, 0]
        if desired_force is None:
            desired_force = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if limit_vel is None:
            limit_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if int(flag) == 0:
            joint_values = self._joint_for_canfd(joint or [])
            pose_values = None
        else:
            pose_values = _as_float_list(pose or [])[:6]
            joint_values = None
        param = rm_force_position_move_t(
            flag=int(flag),
            pose=pose_values,
            joint=joint_values,
            sensor=int(sensor),
            mode=int(mode),
            follow=bool(follow),
            control_mode=_as_int_list(control_mode)[:6],
            desired_force=_as_float_list(desired_force)[:6],
            limit_vel=_as_float_list(limit_vel)[:6],
            trajectory_mode=int(trajectory_mode),
            radio=int(radio),
        )
        return self._robot.rm_force_position_move(param)

    def Get_Arm_Software_Version(self, retry=0):
        ret, info = self._robot.rm_get_arm_software_info()
        return (
            ret,
            info.get("plan_info", {}).get("version", ""),
            info.get("ctrl_info", {}).get("version", ""),
            "",
            "",
            info.get("product_version", ""),
        )

    def RM_API_UnInit(self):
        return 0

    def Arm_Socket_Close(self):
        if self._closed:
            return 0
        self._closed = True
        return self._robot.rm_delete_robot_arm()

    def close(self):
        return self.Arm_Socket_Close()
