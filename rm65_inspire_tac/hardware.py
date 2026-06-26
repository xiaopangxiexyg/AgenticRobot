from __future__ import annotations

import multiprocessing
import signal
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import BimanualExecutionResult


def interpolate_vectors_linear(
    start: np.ndarray, end: np.ndarray, max_distance: np.ndarray | float
) -> list[np.ndarray]:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    max_distance = np.asarray(max_distance, dtype=float)
    if max_distance.shape == ():
        max_distance = np.full_like(start, float(max_distance))
    diff = np.abs(end - start)
    num_steps = int(np.max(np.ceil(diff / max_distance)))
    if num_steps < 1:
        return []
    return [start + (end - start) * i / num_steps for i in range(1, num_steps + 1)]


def _load_rm65_dependencies() -> tuple[Any, Any, Any]:
    try:
        from hardware.inspire_tac.inspire_tac_modbus import InspireTacModbus
        from hardware.rm65.rm_api2_adapter import Arm, RM65
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Portable RM65 hardware control requires the RM65 SDK and pymodbus stack."
        ) from exc
    return Arm, RM65, InspireTacModbus


def _run_arm_hand_controller(
    arm_ip: str,
    hand_ip: str,
    max_fingertip_force: int,
    target_now: multiprocessing.Array,
    target_changed: multiprocessing.Value,
    current_target_done: multiprocessing.Value,
    lock: multiprocessing.Lock,
    max_joint_delta: np.ndarray,
    control_dt: float,
    stop_event: multiprocessing.Event,
    ready_event: multiprocessing.Event,
    force_position_control: bool = False,
    force_sensor: int = 1,
    force_mode: int = 0,
    force_dir: int = 0,
    force: float = 0.0,
    force_follow: bool = False,
    force_clear_on_start: bool = True,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    Arm = RM65 = InspireTacModbus = None
    robot = None
    hand = None
    force_position_active = False
    try:
        Arm, RM65, InspireTacModbus = _load_rm65_dependencies()
        robot = Arm(RM65, arm_ip)
        result_code, joint_now, _, _, _ = robot.Get_Current_Arm_State()
        if result_code == 0:
            target_now[:6] = [joint / 180.0 * np.pi for joint in joint_now]

        hand = InspireTacModbus(ip=hand_ip, max_fingertip_force=max_fingertip_force)
        if force_position_control:
            if force_clear_on_start:
                robot.Clear_Force_Data(True)
            robot.Start_Force_Position_Move(True)
            force_position_active = True

        last_waypoint = np.array(target_now[:6], dtype=float)
        waypoints: list[np.ndarray] = []
        waypoint_index = 0
        ready_event.set()

        while not stop_event.is_set():
            loop_start = time.time()
            with lock:
                if target_changed.value:
                    waypoints = interpolate_vectors_linear(
                        start=last_waypoint,
                        end=np.array(target_now[:6], dtype=float),
                        max_distance=max_joint_delta,
                    )
                    waypoint_index = 0
                    target_changed.value = False
                    current_target_done.value = False
                    hand.set_joint_pos(np.array(target_now[6:], dtype=float))

            if waypoint_index < len(waypoints):
                joint_deg = waypoints[waypoint_index] / np.pi * 180.0
                if force_position_control:
                    robot.Force_Position_Move_Joint(
                        joint_deg,
                        sensor=force_sensor,
                        mode=force_mode,
                        dir=force_dir,
                        force=force,
                        follow=force_follow,
                    )
                else:
                    robot.Movej_CANFD(joint_deg, False)
                last_waypoint = waypoints[waypoint_index]
                waypoint_index += 1
            else:
                current_target_done.value = True

            remaining = control_dt - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if robot is not None:
            if force_position_active:
                robot.Stop_Force_Position_Move(True)
            robot.RM_API_UnInit()
            robot.Arm_Socket_Close()
        if hand is not None:
            hand.close()


class RM65InspireTacStateListener:
    def __init__(self, arm_ip: str, hand_ip: str, max_fingertip_force: int):
        Arm, RM65, InspireTacModbus = _load_rm65_dependencies()
        self.arm = Arm(RM65, arm_ip)
        self.hand = InspireTacModbus(ip=hand_ip, max_fingertip_force=max_fingertip_force)

    def get_state(
        self, require_arm_6d_force: bool = False, tactile_regions: list[str] | None = None
    ) -> dict[str, Any]:
        state: dict[str, Any] = {}
        result_code, joint_now, pose, _, _ = self.arm.Get_Current_Arm_State()
        if result_code == 0:
            state["arm_joint_pos"] = np.asarray(joint_now, dtype=float) / 180.0 * np.pi
            state["eef_6d_pose"] = pose

        hand_joint_pos = self.hand.read_joint_pos()
        tactile_vec, tactile_dict = self.hand.read_tac_data(target_regions=tactile_regions)
        state["hand_joint_pos"] = hand_joint_pos
        state["hand_tac_vec"] = tactile_vec
        state["hand_tac_dict"] = tactile_dict

        if require_arm_6d_force:
            force_code, force, _, _, _ = self.arm.Get_Force_Data()
            if force_code == 0:
                state["arm_6d_force"] = force
        return state

    def close(self) -> None:
        self.arm.RM_API_UnInit()
        self.arm.Arm_Socket_Close()
        self.hand.close()


class RM65InspireTacController:
    def __init__(
        self,
        arm_ip: str,
        hand_ip: str,
        max_fingertip_force: int = 800,
        max_angvel: list[float] | tuple[float, ...] = (0.2,) * 6,
        control_dt: float = 0.02,
        start_listener: bool = True,
        startup_timeout: float = 10.0,
        force_position_control: bool = False,
        force_sensor: int = 1,
        force_mode: int = 0,
        force_dir: int = 0,
        force: float = 0.0,
        force_follow: bool = False,
        force_clear_on_start: bool = True,
    ):
        max_angvel_array = np.asarray(max_angvel, dtype=float)
        if np.max(max_angvel_array) >= 1.58:
            raise ValueError("max_angvel must stay below 1.58 rad/s for safety.")

        self.max_joint_delta = max_angvel_array * control_dt
        self.control_dt = control_dt
        self.target_now = multiprocessing.Array("f", [0.0] * 6 + [1.0] * 6)
        self.target_changed = multiprocessing.Value("b", False)
        self.current_target_done = multiprocessing.Value("b", True)
        self.lock = multiprocessing.Lock()
        self.stop_event = multiprocessing.Event()
        self.ready_event = multiprocessing.Event()
        self.closed = False

        self.controller_process = multiprocessing.Process(
            target=_run_arm_hand_controller,
            args=(
                arm_ip,
                hand_ip,
                max_fingertip_force,
                self.target_now,
                self.target_changed,
                self.current_target_done,
                self.lock,
                self.max_joint_delta,
                self.control_dt,
                self.stop_event,
                self.ready_event,
                force_position_control,
                force_sensor,
                force_mode,
                force_dir,
                force,
                force_follow,
                force_clear_on_start,
            ),
        )
        self.controller_process.start()

        start_time = time.time()
        while not self.ready_event.is_set():
            if not self.controller_process.is_alive():
                raise RuntimeError("RM65 control process exited before becoming ready.")
            if time.time() - start_time > startup_timeout:
                raise TimeoutError("Timed out while waiting for the RM65 control process.")
            time.sleep(0.05)

        self.listener = (
            RM65InspireTacStateListener(arm_ip, hand_ip, max_fingertip_force)
            if start_listener
            else None
        )

    def set_target(
        self, arm_qpos: np.ndarray | list[float], hand_qpos: np.ndarray | list[float] | None = None
    ) -> None:
        if hand_qpos is None:
            target = np.asarray(arm_qpos, dtype=float)
            if target.shape != (12,):
                raise ValueError("set_target expects shape (12,) when hand_qpos is omitted.")
        else:
            target = np.concatenate(
                [np.asarray(arm_qpos, dtype=float), np.asarray(hand_qpos, dtype=float)]
            )
        if target.shape != (12,):
            raise ValueError(f"Expected a 12-dim target, got {target.shape}.")

        with self.lock:
            self.target_now[:] = target.tolist()
            self.target_changed.value = True
            self.current_target_done.value = False

    def move_to(
        self,
        arm_qpos: np.ndarray | list[float],
        hand_qpos: np.ndarray | list[float] | None = None,
        block: bool = True,
    ) -> None:
        self.set_target(arm_qpos, hand_qpos)
        if block:
            time.sleep(0.5)
            while not self.current_target_done.value:
                time.sleep(0.05)

    def get_state(self, **kwargs: Any) -> dict[str, Any]:
        if self.listener is None:
            raise RuntimeError("This controller was created with start_listener=False.")
        return self.listener.get_state(**kwargs)

    def close(self) -> None:
        if self.closed:
            return
        self.stop_event.set()
        if self.listener is not None:
            self.listener.close()
        self.controller_process.join(timeout=3.0)
        if self.controller_process.is_alive():
            self.controller_process.terminate()
            self.controller_process.join(timeout=1.0)
        self.closed = True

    def __enter__(self) -> "RM65InspireTacController":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


@dataclass
class ArmHardwareConfig:
    arm_ip: str
    hand_ip: str
    max_fingertip_force: int = 800
    max_angvel: tuple[float, ...] = (0.2,) * 6
    control_dt: float = 0.02
    force_position_control: bool = False
    force_sensor: int = 1
    force_mode: int = 0
    force_dir: int = 0
    force: float = 0.0
    force_follow: bool = False
    force_clear_on_start: bool = True


class BimanualRM65InspireTacSystem:
    def __init__(
        self,
        left: RM65InspireTacController | None = None,
        right: RM65InspireTacController | None = None,
    ):
        self.left = left
        self.right = right

    @classmethod
    def from_configs(
        cls,
        left: ArmHardwareConfig | None = None,
        right: ArmHardwareConfig | None = None,
    ) -> "BimanualRM65InspireTacSystem":
        left_controller = None
        right_controller = None
        if left is not None:
            left_controller = RM65InspireTacController(
                arm_ip=left.arm_ip,
                hand_ip=left.hand_ip,
                max_fingertip_force=left.max_fingertip_force,
                max_angvel=left.max_angvel,
                control_dt=left.control_dt,
                force_position_control=left.force_position_control,
                force_sensor=left.force_sensor,
                force_mode=left.force_mode,
                force_dir=left.force_dir,
                force=left.force,
                force_follow=left.force_follow,
                force_clear_on_start=left.force_clear_on_start,
            )
        if right is not None:
            right_controller = RM65InspireTacController(
                arm_ip=right.arm_ip,
                hand_ip=right.hand_ip,
                max_fingertip_force=right.max_fingertip_force,
                max_angvel=right.max_angvel,
                control_dt=right.control_dt,
                force_position_control=right.force_position_control,
                force_sensor=right.force_sensor,
                force_mode=right.force_mode,
                force_dir=right.force_dir,
                force=right.force,
                force_follow=right.force_follow,
                force_clear_on_start=right.force_clear_on_start,
            )
        return cls(left=left_controller, right=right_controller)

    def send_execution(self, execution: BimanualExecutionResult) -> None:
        if execution.left is not None and self.left is not None:
            self.left.set_target(execution.left.arm_qpos, execution.left.hardware_hand_qpos)
        if execution.right is not None and self.right is not None:
            self.right.set_target(execution.right.arm_qpos, execution.right.hardware_hand_qpos)

    def get_state(
        self, require_arm_6d_force: bool = False, tactile_regions: list[str] | None = None
    ) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.left is not None:
            state["left"] = self.left.get_state(
                require_arm_6d_force=require_arm_6d_force,
                tactile_regions=tactile_regions,
            )
        if self.right is not None:
            state["right"] = self.right.get_state(
                require_arm_6d_force=require_arm_6d_force,
                tactile_regions=tactile_regions,
            )
        return state

    def close(self) -> None:
        if self.left is not None:
            self.left.close()
        if self.right is not None:
            self.right.close()

    def __enter__(self) -> "BimanualRM65InspireTacSystem":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
