import os
import sys
import time
import threading
import subprocess
from enum import Enum

import can
import numpy as np


class FrameProperty(Enum):
    INVALID_FRAME_PROPERTY = 0x00
    JOINT_POSITION_RCO = 0x01
    MAX_PRESS_RCO = 0x02
    MAX_PRESS_RCO2 = 0x03
    JOINT_POSITION2_RCO = 0x04
    JOINT_SPEED = 0x05
    JOINT_SPEED2 = 0x06
    REQUEST_DATA_RETURN = 0x09
    JOINT_POSITION_N = 0x11
    MAX_PRESS_N = 0x12
    HAND_NORMAL_FORCE = 0x20
    HAND_TANGENTIAL_FORCE = 0x21
    HAND_TANGENTIAL_FORCE_DIR = 0x22
    HAND_APPROACH_INC = 0x23
    MOTOR_TEMPERATURE_1 = 0x33
    MOTOR_TEMPERATURE_2 = 0x34


class HandCan:
    def __init__(self, can_id, can_channel, baudrate):
        self.can_id = int(can_id)
        self.can_channel = can_channel
        self.baudrate = int(baudrate)

        # -------------------- 原有关节/状态缓存 --------------------
        # 0x01 返回前6关节，0x04 返回后4关节
        self.x01 = [-1] * 6
        self.x02 = [-1] * 6
        self.x03 = [-1] * 6
        self.x04 = [-1] * 4
        self.x05 = [-1] * 6
        self.x06 = [-1] * 6

        self.x33 = [0] * 5
        self.x34 = [0] * 5
        self.x35 = [0] * 5
        self.x36 = [0] * 5

        self.normal_force = [0.0] * 5
        self.tangential_force = [0.0] * 5
        self.tangential_force_dir = [0.0] * 5
        self.approach_inc = [0.0] * 5

        self.version = None
        self.joint_angles = [0] * 10

        # 最近一次有效状态
        self.last_valid_state_6 = None
        self.last_valid_state_10 = None
        self.last_state_ts = 0.0

        # 用于同步等待状态回包
        self._evt_x01 = threading.Event()
        self._evt_x04 = threading.Event()

        # -------------------- tactile: 4x10 矩阵缓存 --------------------
        self.matrix_rows = 10
        self.matrix_cols = 4

        self.thumb_matrix = np.full((self.matrix_rows, self.matrix_cols), -1, dtype=np.int16)
        self.index_matrix = np.full((self.matrix_rows, self.matrix_cols), -1, dtype=np.int16)
        self.middle_matrix = np.full((self.matrix_rows, self.matrix_cols), -1, dtype=np.int16)
        self.ring_matrix = np.full((self.matrix_rows, self.matrix_cols), -1, dtype=np.int16)
        self.little_matrix = np.full((self.matrix_rows, self.matrix_cols), -1, dtype=np.int16)

        # 完整帧快照（只有 5指×10行都齐了才更新）
        self.thumb_frame = self.thumb_matrix.copy()
        self.index_frame = self.index_matrix.copy()
        self.middle_frame = self.middle_matrix.copy()
        self.ring_frame = self.ring_matrix.copy()
        self.little_frame = self.little_matrix.copy()
        self.touch_stamp = 0.0

        # 兼容 row_flag: 0..9 或 0,16,32,...,144
        self.matrix_map = {}
        for i in range(self.matrix_rows):
            self.matrix_map[i] = i
            self.matrix_map[i * 16] = i

        # tactile 状态帧/请求响应缓存
        self.xb0 = [-1] * 5
        self.xb1 = [-1] * 2
        self.xb2 = [-1] * 2
        self.xb3 = [-1] * 2
        self.xb4 = [-1] * 2
        self.xb5 = [-1] * 2

        # 完整帧判定：每根手指已收齐的行号集合
        self._row_seen = {
            "thumb": set(),
            "index": set(),
            "middle": set(),
            "ring": set(),
            "little": set(),
        }

        # 若设备矩阵回包 arbitration_id 与 can_id 不同，可改为 True 后锁定
        self.rx_id = None
        self.rx_id_lock = False
        self.touch_code = 0xA4  # 强制请求 4x10

        # 锁（发送/接收缓存并发安全）
        self._rx_lock = threading.Lock()
        self._tx_lock = threading.Lock()

        self._dbg_matrix_once = False

        self.bus = self.init_can_bus(can_channel, baudrate)

        # Start receiving thread
        self.running = True
        self.receive_thread = threading.Thread(target=self.receive_response, daemon=True)
        self.receive_thread.start()

    def init_can_bus(self, channel, baudrate):
        try:
            if sys.platform == "linux":
                return can.interface.Bus(channel=channel, interface="socketcan", bitrate=baudrate)
            elif sys.platform == "win32":
                return can.interface.Bus(channel=channel, interface="pcan", bitrate=baudrate)
            else:
                raise EnvironmentError("Unsupported platform for CAN interface")
        except Exception:
            print("Please insert CAN device")
            return None

    def send_frame(self, frame_property, data_list, sleep=0.003):
        """Send a single CAN frame with specified properties and data."""
        if self.bus is None:
            print("CAN bus is not initialized.")
            if sleep > 0:
                time.sleep(sleep)
            return

        frame_property_value = int(frame_property.value) if hasattr(frame_property, "value") else int(frame_property)
        data = [frame_property_value] + [int(val) for val in data_list]
        msg = can.Message(arbitration_id=self.can_id, data=data, is_extended_id=False)

        try:
            with self._tx_lock:
                self.bus.send(msg)
        except can.CanError as e:
            print(f"Failed to send message: {e}")
        if sleep > 0:
            time.sleep(sleep)

    def _is_valid_joint_block(self, vals, expect_len):
        if not isinstance(vals, (list, tuple)):
            return False
        if len(vals) < expect_len:
            return False
        for v in vals[:expect_len]:
            if not isinstance(v, (int, float)):
                return False
            if v < 0 or v > 255:
                return False
        return True

    # -------------------- 原关节控制接口 --------------------

    def set_joint_positions(self, joint_angles):
        """Set joint positions. Supports 6-joint or 10-joint input."""
        if joint_angles is None:
            return
        self.joint_angles = list(joint_angles)

        # 兼容原协议习惯：先发后4，再发前6
        if len(self.joint_angles) > 6:
            self.send_frame(FrameProperty.JOINT_POSITION2_RCO, self.joint_angles[6:], sleep=0.0)
            time.sleep(0.001)
        self.send_frame(FrameProperty.JOINT_POSITION_RCO, self.joint_angles[:6], sleep=0.0)

    def get_current_status(self, need_all_10=False, timeout=0.03, retries=2, return_last_valid=True):
        """
        同步读取当前关节状态。
        need_all_10=False: 只读前6关节
        need_all_10=True : 读前6 + 后4
        """
        for _ in range(int(retries)):
            self._evt_x01.clear()
            self._evt_x04.clear()

            # 发请求
            self.send_frame(0x01, [], sleep=0.0)
            if need_all_10:
                self.send_frame(0x04, [], sleep=0.0)

            # 等待回包
            got01 = self._evt_x01.wait(timeout)
            got04 = True
            if need_all_10:
                got04 = self._evt_x04.wait(timeout)

            with self._rx_lock:
                x01 = list(self.x01)
                x04 = list(self.x04)

            valid01 = self._is_valid_joint_block(x01, 6)
            valid04 = self._is_valid_joint_block(x04, 4) if need_all_10 else True

            if got01 and valid01 and got04 and valid04:
                state6 = x01[:6]
                state10 = x01[:6] + x04[:4]

                with self._rx_lock:
                    self.last_valid_state_6 = list(state6)
                    self.last_valid_state_10 = list(state10)
                    self.last_state_ts = time.time()

                return state10 if need_all_10 else state6

            time.sleep(0.001)

        # 超时后优先返回最近一次有效状态
        with self._rx_lock:
            if return_last_valid:
                if need_all_10 and self.last_valid_state_10 is not None:
                    return list(self.last_valid_state_10)
                if (not need_all_10) and self.last_valid_state_6 is not None:
                    return list(self.last_valid_state_6)

        raise TimeoutError("Failed to read valid joint status from hand CAN bus.")

    def set_torque(self, torque=[180] * 6):
        """Set L6 maximum torque limits."""
        if len(torque) != 6:
            raise ValueError("Torque list must have 6 elements.")
        self.send_frame(0x02, torque, sleep=0.0)

    def set_speed(self, speed=[180] * 6):
        """Set L6 speed."""
        if len(speed) != 6:
            raise ValueError("Speed list must have 6 elements.")
        with self._rx_lock:
            self.x05 = list(speed)
        for _ in range(2):
            time.sleep(0.001)
            self.send_frame(0x05, speed, sleep=0.0)

    def get_speed(self):
        # L6 暂不支持读取实时速度（按原逻辑）
        return [0] * 6

    def get_torque(self):
        """Not supported yet; returns cache after request."""
        self.send_frame(0x02, [], sleep=0.01)
        with self._rx_lock:
            return list(self.x02)

    # -------------------- tactile 请求/读取接口 --------------------

    def request_matrix_all(self, touch_code=None):
        """
        请求 5 指 tactile 矩阵数据。
        默认使用 0xA4（4x10）
        """
        code = int(self.touch_code if touch_code is None else touch_code)
        self.send_frame(0xB1, [code], sleep=0.0)
        self.send_frame(0xB2, [code], sleep=0.0)
        self.send_frame(0xB3, [code], sleep=0.0)
        self.send_frame(0xB4, [code], sleep=0.0)
        self.send_frame(0xB5, [code], sleep=0.0)

    def get_matrix_cached(self):
        """
        返回最近一次“完整帧快照”:
        thumb, index, middle, ring, little, touch_stamp
        每个矩阵 shape=(10,4), dtype=int16, 未更新位置为 -1
        """
        with self._rx_lock:
            return (
                self.thumb_frame.copy(),
                self.index_frame.copy(),
                self.middle_frame.copy(),
                self.ring_frame.copy(),
                self.little_frame.copy(),
                float(self.touch_stamp),
            )

    def read_matrix_touch_4x10(self, request=True, wait_complete=False, timeout=0.05, poll_interval=0.002):
        """
        便捷接口：
        - request=True: 先发一次 5指请求
        - wait_complete=True: 等待 touch_stamp 更新（最多 timeout 秒）
        """
        prev_ts = self.get_matrix_cached()[-1]
        if request:
            self.request_matrix_all(touch_code=0xA4)

        if wait_complete:
            t0 = time.time()
            while time.time() - t0 < float(timeout):
                mats = self.get_matrix_cached()
                ts = mats[-1]
                if ts > prev_ts:
                    return mats
                time.sleep(float(poll_interval))

        return self.get_matrix_cached()

    # -------------------- 接收线程 --------------------

    def receive_response(self):
        while self.running:
            try:
                if self.bus is None:
                    time.sleep(0.1)
                    continue

                msg = self.bus.recv(timeout=1.0)
                if not self.running:
                    break
                if msg:
                    self.process_response(msg)
            except OSError as e:
                if self.running:
                    print(f"Socket error on CAN recv: {e}")
                break
            except can.CanError as e:
                if self.running:
                    print(f"python-can recv error: {e}")
                break
            except Exception as e:
                if self.running:
                    print(f"Unknown recv error: {e}")
                break

    def _looks_like_matrix_row(self, msg):
        if msg is None or msg.data is None or len(msg.data) < 2 + self.matrix_cols:
            return False
        ft = int(msg.data[0])
        if ft not in (0xB1, 0xB2, 0xB3, 0xB4, 0xB5):
            return False
        row_flag = int(msg.data[1])
        return row_flag in self.matrix_map

    def _process_matrix_row(self, msg):
        """
        处理 tactile row 帧：
        msg.data = [0xBx, row_flag, v0, v1, v2, v3, ...]
        仅取前 4 个值作为 4x10 数据
        """
        if not self._looks_like_matrix_row(msg):
            return False

        if self.rx_id_lock:
            if self.rx_id is None:
                self.rx_id = int(msg.arbitration_id)
            if int(msg.arbitration_id) != int(self.rx_id):
                return False

        if not self._dbg_matrix_once:
            raw = " ".join(f"{int(x):02X}" for x in msg.data)
            print(f"[DBG] MATRIX raw id=0x{int(msg.arbitration_id):X} data=[{raw}]")
            self._dbg_matrix_once = True

        ft = int(msg.data[0])
        row_flag = int(msg.data[1])
        row = self.matrix_map.get(row_flag, None)
        if row is None or not (0 <= row < self.matrix_rows):
            return False

        vals = [int(x) for x in msg.data[2:2 + self.matrix_cols]]

        if ft == 0xB1:
            key = "thumb"
            mat = self.thumb_matrix
        elif ft == 0xB2:
            key = "index"
            mat = self.index_matrix
        elif ft == 0xB3:
            key = "middle"
            mat = self.middle_matrix
        elif ft == 0xB4:
            key = "ring"
            mat = self.ring_matrix
        else:
            key = "little"
            mat = self.little_matrix

        with self._rx_lock:
            mat[row, :] = np.array(vals, dtype=np.int16)
            self._row_seen[key].add(row)

            # 只有 5 指各 10 行都齐了，才更新“完整帧快照”
            full = all(len(self._row_seen[k]) >= self.matrix_rows for k in self._row_seen.keys())
            if full:
                self.thumb_frame = self.thumb_matrix.copy()
                self.index_frame = self.index_matrix.copy()
                self.middle_frame = self.middle_matrix.copy()
                self.ring_frame = self.ring_matrix.copy()
                self.little_frame = self.little_matrix.copy()
                self.touch_stamp = time.time()
                for k in self._row_seen.keys():
                    self._row_seen[k].clear()
        return True

    def process_response(self, msg):
        """Process received CAN messages (已融合 tactile 4x10)."""
        if msg is None or msg.data is None or len(msg.data) == 0:
            return

        frame_type = int(msg.data[0])

        # ---------- 先处理 tactile（矩阵帧可能不走 self.can_id） ----------
        if frame_type in (0xB1, 0xB2, 0xB3, 0xB4, 0xB5):
            if self._process_matrix_row(msg):
                return

            response_data = list(msg.data[1:])
            with self._rx_lock:
                if frame_type == 0xB1:
                    self.xb1 = list(response_data)
                elif frame_type == 0xB2:
                    self.xb2 = list(response_data)
                elif frame_type == 0xB3:
                    self.xb3 = list(response_data)
                elif frame_type == 0xB4:
                    self.xb4 = list(response_data)
                elif frame_type == 0xB5:
                    self.xb5 = list(response_data)
            return

        # ---------- 非 tactile 帧按原来的 hand_id 过滤 ----------
        if int(msg.arbitration_id) != int(self.can_id):
            return

        response_data = list(msg.data[1:])

        with self._rx_lock:
            if frame_type == FrameProperty.JOINT_POSITION_RCO.value:  # 0x01
                if self._is_valid_joint_block(response_data, 6):
                    self.x01 = list(response_data[:6])
                    self._evt_x01.set()

            elif frame_type == FrameProperty.MAX_PRESS_RCO.value:  # 0x02
                self.x02 = list(response_data[:6])

            elif frame_type == FrameProperty.MAX_PRESS_RCO2.value:  # 0x03
                self.x03 = list(response_data[:6])

            elif frame_type == FrameProperty.JOINT_POSITION2_RCO.value:  # 0x04
                if self._is_valid_joint_block(response_data, 4):
                    self.x04 = list(response_data[:4])
                    self._evt_x04.set()

            elif frame_type == 0x05:
                self.x05 = list(response_data[:6])

            elif frame_type == 0x06:
                self.x06 = list(response_data[:6])

            elif frame_type == 0x20:
                self.normal_force = [float(i) for i in response_data[:5]]

            elif frame_type == 0x21:
                self.tangential_force = [float(i) for i in response_data[:5]]

            elif frame_type == 0x22:
                self.tangential_force_dir = [float(i) for i in response_data[:5]]

            elif frame_type == 0x23:
                self.approach_inc = [float(i) for i in response_data[:5]]

            elif frame_type == 0x33:
                self.x33 = list(response_data[:5])

            elif frame_type == 0x34:
                self.x34 = list(response_data[:5])

            elif frame_type == 0x35:
                self.x35 = list(response_data[:5])

            elif frame_type == 0x36:
                self.x36 = list(response_data[:5])

            elif frame_type == 0x64:
                self.version = list(response_data)

    def close_can_interface(self):
        """Stop the CAN communication."""
        self.running = False
        try:
            if self.receive_thread.is_alive():
                self.receive_thread.join(timeout=2.0)
        finally:
            if self.bus:
                try:
                    self.bus.shutdown()
                except Exception:
                    pass


class HandApi:
    def __init__(self, hand_type, can, baudrate):  # Ubuntu: can0   win: PCAN_USBBUS1
        self.last_position = []
        self.can = can
        if hand_type == "left":
            self.hand_id = 0x28
        elif hand_type == "right":
            self.hand_id = 0x27
        else:
            raise ValueError("hand_type must be 'left' or 'right'")

        self.hand = HandCan(can_id=self.hand_id, can_channel=self.can, baudrate=baudrate)

    def set_speed(self, speed=[100] * 6):
        """Set speed"""
        has_non_int = any(not isinstance(x, (int, float)) or x < 0 or x > 255 for x in speed)
        if has_non_int:
            print(
                "Set Speed The numerical range can only be positive integers or floating-point numbers between 0 and 255",
                flush=True,
            )
            return
        if len(speed) != 6:
            print("Data length is wrong, must be 6 elements", flush=True)
            return
        self.hand.set_speed(speed=speed)

    def set_torque(self, torque=[180] * 6):
        """Set maximum torque"""
        has_non_int = any(not isinstance(x, (int, float)) or x < 0 or x > 255 for x in torque)
        if has_non_int:
            print(
                "Set Torque The numerical range can only be positive integers or floating-point numbers between 0 and 255",
                flush=True,
            )
            return
        if len(torque) != 6:
            print("Data length is wrong, must be 6 elements", flush=True)
            return
        return self.hand.set_torque(torque=torque)

    def get_speed(self):
        """Get speed"""
        return self.hand.get_speed()

    def get_torque(self):
        """Get current maximum torque"""
        return self.hand.get_torque()

    # Five-finger movement
    def finger_move(self, pose=None):
        if pose is None:
            return
        if pose == self.last_position:
            return

        self._ensure_0_255_list(pose, name="pose", min_len=6)
        if len(pose) != 6:
            raise ValueError(f"Expected 6 joint values in pose, got {len(pose)}")

        self.hand.set_joint_positions(pose)
        self.last_position = list(pose)

    def get_state(self, need_all_10=False, timeout=0.03, retries=2, return_last_valid=True):
        """Get current joint state"""
        return self.hand.get_current_status(
            need_all_10=need_all_10,
            timeout=timeout,
            retries=retries,
            return_last_valid=return_last_valid,
        )

    # -------------------- tactile 接口 --------------------

    def request_matrix_touch_4x10(self):
        """主动请求一次 5指 tactile 4x10 数据。"""
        self.hand.request_matrix_all(touch_code=0xA4)

    def get_matrix_touch_4x10(self):
        """
        获取最近一次完整触觉快照（不主动请求）：
        return: thumb, index, middle, ring, little, ts
        """
        return self.hand.get_matrix_cached()

    def read_matrix_touch_4x10(self, request=True, wait_complete=False, timeout=0.05, poll_interval=0.002):
        """
        便捷接口：
        - request=True: 先发请求
        - wait_complete=True: 等待一帧完整更新
        """
        return self.hand.read_matrix_touch_4x10(
            request=request,
            wait_complete=wait_complete,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def close_can(self):
        if hasattr(self, "hand") and hasattr(self.hand, "close_can_interface"):
            self.hand.close_can_interface()

    def _ensure_0_255_list(self, vals, name: str, min_len: int = None):
        """Validate a list whose elements must be in [0, 255]."""
        if not isinstance(vals, (list, tuple)):
            raise TypeError(f"{name} must be a list or tuple, got {type(vals)}")
        for x in vals:
            if not isinstance(x, (int, float)) or x < 0 or x > 255:
                raise ValueError(f"{name} values must be numbers in [0, 255], got {vals}")
        if min_len is not None and len(vals) < min_len:
            raise ValueError(f"{name} length must be at least {min_len}, got {len(vals)}")


def _run(cmd: str, stdin_text=None):
    import shlex
    p = subprocess.run(
        shlex.split(cmd),
        input=stdin_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return p.returncode, (p.stdout or "").strip()


def _sudo(cmd: str):
    pw = os.environ.get("PREFLIGHT_SUDO_PASS", "")
    if pw:
        return _run(f"sudo -S {cmd}", stdin_text=pw + "\n")
    return _run(f"sudo {cmd}")


def _can_state(iface: str):
    p = f"/sys/class/net/{iface}/operstate"
    try:
        with open(p, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def _ensure_can_up(iface: str, bitrate: int, restart_ms: int = 100, require_root: bool = False) -> str:
    st = _can_state(iface)
    if st not in ("up", "unknown"):
        cmd_down = f"/usr/sbin/ip link set {iface} down"
        # cmd_type = f"/usr/sbin/ip link set {iface} type can bitrate {bitrate} restart-ms {restart_ms}"
        cmd_type = f"/usr/sbin/ip link set {iface} type can bitrate {bitrate}"
        cmd_up = f"/usr/sbin/ip link set {iface} up"
        run = _run if (os.geteuid() == 0 and not require_root) else _sudo
        run(cmd_down)
        rc2, out2 = run(cmd_type)
        if rc2 != 0:
            raise RuntimeError(f"CAN config failed: {out2}")
        rc3, out3 = run(cmd_up)
        if rc3 != 0:
            raise RuntimeError(f"CAN up failed: {out3}")
        time.sleep(0.2)
    return iface


def _clip(v, lo, hi):
    return max(lo, min(v, hi))


class DexHand:
    def __init__(self, hand_type, can_iface, baudrate=1000000):
        iface = _ensure_can_up(iface=can_iface, bitrate=baudrate, restart_ms=100)
        self.hand = HandApi(hand_type=hand_type, can=iface, baudrate=baudrate)

        self.joint_names = [
            "thumb_cmc_pitch",
            "thumb_cmc_yaw",
            "index_mcp_pitch",
            "middle_mcp_pitch",
            "ring_mcp_pitch",
            "pinky_mcp_pitch",
        ]
        self.num_joints = len(self.joint_names)

        self.hardware_joint_limit_low = [0, 0, 0, 0, 0, 0]
        self.hardware_joint_limit_high = [255, 255, 255, 255, 255, 255]

    def read_joint_pos(self):
        """
        return:
            joint_pos: (D,) list in [0,1]
            1 = closed, 0 = open
        """
        data = self.hand.get_state(
            need_all_10=False,
            timeout=0.03,
            retries=2,
            return_last_valid=True,
        )

        if len(data) < self.num_joints:
            raise RuntimeError(f"Invalid hand state length: got {len(data)}, expected {self.num_joints}")

        joint_pos = []
        for i in range(self.num_joints):
            d = data[i]
            lo = self.hardware_joint_limit_low[i]
            hi = self.hardware_joint_limit_high[i]

            if not isinstance(d, (int, float)) or d < lo or d > hi:
                raise RuntimeError(f"Invalid joint raw value at index {i}: {d}")

            val = (d - lo) / (hi - lo) if hi != lo else 0.0
            val = _clip(val, 0.0, 1.0)
            joint_pos.append(1.0 - val)
        return joint_pos

    def set_joint_pos(self, joint_pos):
        """
        joint_pos: (D,) list in [0,1]
        """
        data = []
        for i, jp in enumerate(joint_pos):
            lo = self.hardware_joint_limit_low[i]
            hi = self.hardware_joint_limit_high[i]
            inv = 1.0 - _clip(jp, 0.0, 1.0)
            val = inv * (hi - lo) + lo
            data.append(int(_clip(val, 0, 255)))
        self.hand.finger_move(pose=data)

    def set_speed(self, speed):
        """
        speed: (D,) list in [0,1]
            1 = max speed, 0 = min speed
        """
        data = []
        for i, sp in enumerate(speed):
            val = _clip(sp, 0.0, 1.0)
            val = val * 255.0
            val = int(round(_clip(val, 0.0, 255.0)))
            data.append(val)
        self.hand.set_speed(speed=data)

    def get_speed(self):
        """
        return:
            speed: (D,) list in [0,1]
            1 = max speed, 0 = min speed
        """
        raw = self.hand.get_speed()[:self.num_joints]
        speed = []
        for i, d in enumerate(raw):
            lo, hi = 0.0, 255.0
            val = (d - lo) / (hi - lo) if hi != lo else 0.0
            val = _clip(val, 0.0, 1.0)
            speed.append(val)
        return speed

    def set_torque(self, torque):
        """
        torque: (D,) list in [0,1]
            1 = max torque, 0 = min torque
        """
        data = []
        for i, tq in enumerate(torque):
            val = _clip(tq, 0.0, 1.0)
            val = val * 255.0
            val = int(round(_clip(val, 0.0, 255.0)))
            data.append(val)
        self.hand.set_torque(torque=data)

    def get_torque(self):
        """
        return:
            torque: (D,) list in [0,1]
            1 = max torque, 0 = min torque
        """
        raw = self.hand.get_torque()[:self.num_joints]
        torque = []
        for i, d in enumerate(raw):
            lo, hi = 0.0, 255.0
            val = (d - lo) / (hi - lo) if hi != lo else 0.0
            val = _clip(val, 0.0, 1.0)
            torque.append(val)
        return torque

    # -------------------- tactile 接口 --------------------

    def request_matrix_touch_4x10(self):
        """
        主动请求一次 tactile 数据（5指 4x10）
        """
        self.hand.request_matrix_touch_4x10()

    def read_matrix_touch_4x10(self, request=True, wait_complete=False, timeout=0.05, poll_interval=0.002):
        """
        返回:
            thumb, index, middle, ring, little, ts
        其中每个矩阵 shape=(10,4)，未更新位置为 -1。
        ts 为最近一次“完整帧”更新时间戳（5指×10行都收齐才更新）
        """
        return self.hand.read_matrix_touch_4x10(
            request=request,
            wait_complete=wait_complete,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def close_can(self):
        self.hand.close_can()

    def open_hand(self):
        n = getattr(self, "num_joints", 6)
        p = [0.0] * n
        self.set_joint_pos(p)
        time.sleep(0.1)

    def half_close_hand(self):
        n = getattr(self, "num_joints", 6)
        p = [0.5] * n
        self.set_joint_pos(p)
        time.sleep(0.1)

    def gesture_dance(self, duration_s: float, m: float = 0.1, interval_s: float = 0.1, start=None):
        if start is None:
            start = [0.0, 0.4, 0.4, 0.6, 0.8, 1.0]

        if len(start) != self.num_joints:
            raise ValueError(f"start length must be {self.num_joints}, got {len(start)}")

        lo = [0.0] * 6
        hi = [0.4, 0.4, 1.0, 1.0, 1.0, 1.0]

        cur = []
        for i, v in enumerate(start):
            if v < lo[i]:
                v = lo[i]
            elif v > hi[i]:
                v = hi[i]
            cur.append(v)

        direction = [-1.0] * len(cur)
        t_end = time.time() + duration_s

        while time.time() < t_end:
            self.set_joint_pos(cur)
            time.sleep(interval_s)
            print(self.read_joint_pos())
            for i in range(len(cur)):
                cur[i] += direction[i] * m
                if cur[i] <= lo[i]:
                    cur[i] = lo[i]
                    direction[i] = 1.0
                elif cur[i] >= hi[i]:
                    cur[i] = hi[i]
                    direction[i] = -1.0


if __name__ == "__main__":
    right = DexHand(hand_type="right", can_iface="can0")
    # right = DexHand(hand_type="left", can_iface="can1")

    try:
        # # debug tactile sensor
        # for _ in range(500):
        #     thumb, index, middle, ring, little, ts = right.read_matrix_touch_4x10(
        #         request=True,
        #         wait_complete=False,
        #     )
        #     print("touch ts:", ts, "thumb shape:", thumb.shape)
        #     time.sleep(0.02)
        #
        # thumb, index, middle, ring, little, ts = right.read_matrix_touch_4x10(
        #     request=True,
        #     wait_complete=True,
        #     timeout=0.05
        # )

        # right.gesture_dance(duration_s=3, m=0.1, interval_s=0.1)

        right.open_hand()

        # right.set_speed(speed=[1, 1, 1, 1, 1, 1])
        # right.set_torque(torque=[1, 1, 1, 1, 1, 1])
        #
        right.set_joint_pos([0.55, 0.8, 0.42, 0.45, 0, 0])
        time.sleep(1)
        # right.open_hand()
        # time.sleep(1)
        #
        # right.get_torque()
        # right.get_speed()
        #
        # # right.set_speed(speed=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # # right.set_torque(torque=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        #
        print(right.read_joint_pos())
        time.sleep(1)
        # right.half_close_hand()
        print(right.read_joint_pos())

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")

    finally:
        try:
            safe_open = [0.0] * 6
            right.set_joint_pos(safe_open)
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed to send safe-open: {e}")
        try:
            right.close_can()
        except Exception as e:
            print(f"Failed to close: {e}")