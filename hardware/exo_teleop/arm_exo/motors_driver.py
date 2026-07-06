import time
import json
from threading import Event, Lock, Thread
from typing import Protocol, Sequence, Dict, Any, Iterable, Optional, Union

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

from .motors_config import MotorRobotParams
from .motors_config import (
    ModelSpec, XM430_W350, XM540_W270, XL330_M288, XC330_M288,
)

ScalarOrSeq = Union[int, Sequence[int]]

# ========================= Control table / constants =========================

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4

ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
LEN_PROFILE_VELOCITY = 4

ADDR_OPERATING_MODE = 11
ADDR_GOAL_CURRENT = 102

CURRENT_LIMIT = 800  # max 1250 mA, XL330-M288 ±651 mA

ADDR_BAUDRATE = 8
ADDR_MODEL_NUMBER = 0
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146
ADDR_BUS_WATCHDOG = 98

PROTOCOL_VERSION = 2.0

MODEL_NUMBER_TO_NAME: Dict[int, str] = {
    1020: "XM430-W350",
    1030: "XM430-W210",
    1120: "XM540-W270",
    1060: "XL330-M288",
    1240: "XC330-M288-T",
    360: "XL430-W250",
    321: "XL320",
    2000: "MX-28(2.0)",
}

NAME_KEY_TO_SPEC: Dict[str, ModelSpec] = {
    "XM430": XM430_W350,
    "XM540": XM540_W270,
    "XL330": XL330_M288,
    "XC330": XC330_M288,
}

OPERATING_MODES = {
    0: "Current",
    1: "Velocity",
    3: "Position",
    4: "Extended Position",
    5: "Current-based Position",
    16: "PWM",
}

BAUDRATE_MAP = {
    0: 9600,
    1: 57600,
    2: 115200,
    3: 1000000,
    4: 2000000,
    5: 3000000,
    6: 4000000,
}

TICKS_PER_REV = 4096
TICKS2RAD = 2 * np.pi / TICKS_PER_REV


# ========================= Helper functions =========================

def _infer_spec(model_number: int):
    name = MODEL_NUMBER_TO_NAME.get(model_number)
    if name:
        for key, spec in NAME_KEY_TO_SPEC.items():
            if key in name:
                return spec
    return None


def _current_lsb_to_mA(lsb: int, spec: Optional[ModelSpec]):
    if spec is None:
        return None
    return float(lsb) * spec.current_unit_mA_per_lsb


def _unwrap_ticks_step(prev: int, now: int) -> int:
    # 把 (now - prev) 约束到 [-2048, 2047]
    return ((now - prev + 2048) % 4096) - 2048


def rad_to_ticks(rad: float) -> int:
    return int(np.round((rad % (2 * np.pi)) * TICKS_PER_REV / (2 * np.pi)))


def ticks_to_rad(ticks: int) -> float:
    return (int(ticks) % TICKS_PER_REV) * (2 * np.pi) / TICKS_PER_REV


def _fmt_list(nums, nd=8):
    arr = np.around(np.asarray(nums, dtype=float), nd).tolist()
    return json.dumps(arr, separators=(',', ':'))


def scan_dynamixels(
    port_name: str,
    id_range: Iterable[int],
    baudrate_list: Optional[Iterable[int]] = None,
):
    if baudrate_list is None:
        baudrate_list = BAUDRATE_MAP.values()

    devices: list[Dict[str, Any]] = []

    for baudrate in baudrate_list:
        port = PortHandler(port_name)
        pkt = PacketHandler(PROTOCOL_VERSION)

        print(f"\n--- Scanning on {port_name} @ {baudrate} bps ---")
        opened = False
        try:
            if not port.openPort():
                print(f"[ERROR] openPort failed: {port_name}")
                continue
            opened = True

            if not port.setBaudRate(int(baudrate)):
                print(f"[ERROR] setBaudRate failed: {baudrate}")
                continue

            for dxl_id in id_range:
                model_number, r, e = pkt.read2ByteTxRx(port, dxl_id, ADDR_MODEL_NUMBER)
                if r != COMM_SUCCESS or e != 0:
                    time.sleep(0.001)
                    continue

                spec = _infer_spec(model_number)
                model_name = MODEL_NUMBER_TO_NAME.get(model_number, f"Unknown({model_number})")

                pos_raw, r1, e1 = pkt.read4ByteTxRx(port, dxl_id, ADDR_PRESENT_POSITION)
                pos_raw = pos_raw if (r1 == COMM_SUCCESS and e1 == 0) else None
                pos_rad = ticks_to_rad(pos_raw) if pos_raw is not None else None
                pos_deg = float(np.degrees(pos_rad)) if pos_rad is not None else None

                mode, r2, e2 = pkt.read1ByteTxRx(port, dxl_id, ADDR_OPERATING_MODE)
                mode = mode if (r2 == COMM_SUCCESS and e2 == 0) else None

                baudnum, r3, e3 = pkt.read1ByteTxRx(port, dxl_id, ADDR_BAUDRATE)
                baudnum = baudnum if (r3 == COMM_SUCCESS and e3 == 0) else None
                baud_actual = BAUDRATE_MAP.get(baudnum, None) if baudnum is not None else None

                hes, r4, e4 = pkt.read1ByteTxRx(port, dxl_id, ADDR_HARDWARE_ERROR_STATUS)
                hes = hes if (r4 == COMM_SUCCESS and e4 == 0) else None

                cur_lsb, r5, e5 = pkt.read2ByteTxRx(port, dxl_id, ADDR_PRESENT_CURRENT)
                if r5 == COMM_SUCCESS and e5 == 0:
                    cur_lsb = np.int16(cur_lsb).item()
                    cur_mA = _current_lsb_to_mA(cur_lsb, spec)
                else:
                    cur_lsb = None
                    cur_mA = None

                volt_10, r6, e6 = pkt.read2ByteTxRx(port, dxl_id, ADDR_PRESENT_VOLTAGE)
                volt_V = (volt_10 / 10.0) if (r6 == COMM_SUCCESS and e6 == 0) else None

                temp_C, r7, e7 = pkt.read1ByteTxRx(port, dxl_id, ADDR_PRESENT_TEMPERATURE)
                temp_C = temp_C if (r7 == COMM_SUCCESS and e7 == 0) else None

                if spec is not None:
                    unit = spec.current_unit_mA_per_lsb
                    limit_mA = int(spec.current_limit_max_lsb * spec.current_unit_mA_per_lsb)
                else:
                    unit = None
                    limit_mA = None

                dev = {
                    "id": dxl_id,
                    "baudrate": baudrate,
                    "model_number": model_number,
                    "model_name": model_name,
                    "spec": spec,
                    "current_unit_mA_per_lsb": unit,
                    "current_limit_max_mA": limit_mA,
                    "present_position_ticks": pos_raw,
                    "present_position_rad": pos_rad,
                    "present_position_deg": pos_deg,
                    "operating_mode": mode,
                    "baudnum": baudnum,
                    "baudrate_actual": baud_actual,
                    "hardware_error": hes,
                    "present_current_lsb": cur_lsb,
                    "present_current_mA": cur_mA,
                    "voltage_V": volt_V,
                    "temperature_C": temp_C,
                }
                devices.append(dev)

                unit_str = f"{unit} mA/LSB" if unit is not None else "Unknown"
                limit_str = f"{limit_mA} mA" if limit_mA is not None else "Unknown"
                cur_str = f"{cur_mA:.0f} mA" if cur_mA is not None else "N/A"
                hes_str = f"0x{hes:02X}" if hes is not None else "N/A"
                pos_str = (
                    f"{pos_raw} ticks | {pos_rad:.4f} rad | {pos_deg:.2f}°"
                    if pos_raw is not None else "N/A"
                )
                mode_str = OPERATING_MODES.get(mode, str(mode)) if mode is not None else "N/A"
                v_str = f"{volt_V:.1f}V" if volt_V is not None else "N/A"
                t_str = f"{temp_C}C" if temp_C is not None else "N/A"

                print(
                    f"[ID:{dxl_id:03d}] {model_name} "
                    f"| Mode:{mode_str} | BaudNum:{baudnum}({baud_actual}) "
                    f"| Pos:{pos_str} "
                    f"| CurrUnit:{unit_str} | CurrLimit:{limit_str} "
                    f"| I:{cur_str} | V:{v_str} | T:{t_str} | HES:{hes_str}"
                )
                time.sleep(0.001)

            if devices:
                break

        finally:
            if opened:
                port.closePort()

    if not devices:
        print("[RESULT] No Dynamixel devices found.")
    return devices


# ========================= Protocol interface =========================

class DynamixelDriverProtocol(Protocol):
    def set_control_mode(self, mode: int):
        ...

    def set_joints(self, joint_angles: Sequence[float]):
        ...

    def torque_enabled(self) -> bool:
        ...

    def set_torque_mode(self, enable: bool):
        ...

    def get_joints(self) -> np.ndarray:
        ...

    def close(self):
        ...


class FakeDynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._joint_angles = np.zeros(len(ids), dtype=float)
        self._torque_enabled = False

    def set_control_mode(self, mode: int):
        if mode not in [0, 1, 3, 5, 11, 16]:
            raise ValueError("Invalid control mode.")
        print(f"Control mode set to {mode} for fake Dynamixel driver.")

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError("The length of joint_angles must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        self._joint_angles = np.array(joint_angles, dtype=float)

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        return self._joint_angles.copy()

    def close(self):
        pass


# ========================= Real driver =========================

class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, cfg: MotorRobotParams, read_only: bool = False):
        self._ids = cfg.ids
        self._spec_by_id = cfg.id2spec
        self._profile_velocity = cfg.control_velocitys
        self._profile_acc = cfg.accelerations
        self._read_only = bool(read_only)

        # 串口 I/O 锁：只保护 tx/rx
        self._io_lock = Lock()
        # 状态缓存锁：只保护内存缓存
        self._state_lock = Lock()

        self._joint_angles: Optional[np.ndarray] = None
        self._joint_ts: float = 0.0

        self._last_ticks: dict[int, Optional[int]] = {i: None for i in self._ids}
        self._angle_cont: dict[int, float] = {i: 0.0 for i in self._ids}

        self._closed = False
        self._torque_enabled = False
        self._stop_thread = Event()

        # 固定后台读取频率
        self._read_hz = 50.0

        # ---------------- 必须先创建底层通信对象 ----------------
        self._portHandler = PortHandler(cfg.port)
        self._packetHandler = PacketHandler(cfg.protocol_version)

        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        # ---------------- 打开串口 ----------------
        if not self._portHandler.openPort():
            raise RuntimeError(f"Failed to open the port: {cfg.port}")

        if not self._portHandler.setBaudRate(cfg.baudrate):
            self._portHandler.closePort()
            raise RuntimeError(f"Failed to change the baudrate to {cfg.baudrate}")

        # ---------------- 注册 sync read ----------------
        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                self._portHandler.closePort()
                raise RuntimeError(f"Failed to add parameter for Dynamixel with ID {dxl_id}")

        # ---------------- read_only / normal 模式分流 ----------------
        if self._read_only:
            # 只读模式：不要改 operating mode / profile / watchdog
            # 只尝试确保扭矩关闭，失败也不致命
            try:
                self.set_torque_mode(False)
            except Exception as e:
                print(f"[WARN] read_only init torque off failed: {e}")
            self._torque_enabled = False
        else:
            # 正常控制模式：允许配置模式、profile、watchdog
            self.set_control_mode(cfg.operating_mode, enable_torque_after=cfg.torque_enable_on_boot)

            self.bus_watchdog_lsb = cfg.bus_watchdog_lsb
            if cfg.torque_enable_on_boot and self.bus_watchdog_lsb > 0:
                self._apply_bus_watchdog(self.bus_watchdog_lsb)

            if not cfg.torque_enable_on_boot:
                try:
                    self.set_torque_mode(False)
                except Exception as e:
                    print(f"[WARN] torque disable after init failed: {e}")
                self._torque_enabled = False

        # ---------------- 启动后台读取线程 ----------------
        self._start_reading_thread()

    # ------------------ basic io helpers ------------------

    def _check_result(self, txrx_code: int, dxl_err: int, msg: str = "") -> None:
        txrx_msg = self._packetHandler.getTxRxResult(txrx_code)
        dxl_msg = self._packetHandler.getRxPacketError(dxl_err)

        if txrx_code != COMM_SUCCESS and dxl_err != 0:
            raise RuntimeError(f"{msg} TX/RX={txrx_msg} (code={txrx_code}); DXL={dxl_msg} (code={dxl_err})")
        if txrx_code != COMM_SUCCESS:
            raise RuntimeError(f"{msg} TX/RX={txrx_msg} (code={txrx_code})")
        if dxl_err != 0:
            raise RuntimeError(f"{msg} DXL={dxl_msg} (code={dxl_err})")

    def _try_write(self, fn, *args):
        try:
            r, e = fn(*args)
            if r != COMM_SUCCESS or e != 0:
                txrx_msg = self._packetHandler.getTxRxResult(r)
                dxl_msg = self._packetHandler.getRxPacketError(e)
                print(f"[CLOSE] write fail: args={args} TXRX={txrx_msg} DXL={dxl_msg}")
        except Exception as ex:
            print(f"[CLOSE] exception on write {args}: {ex}")

    def _expand_per_id(self, value: ScalarOrSeq, name: str):
        if isinstance(value, (bytes, bytearray, str)):
            raise TypeError(f"{name} must be int or sequence of int")
        try:
            if hasattr(value, "__iter__") and not isinstance(value, int):
                lst = [int(v) for v in value]
                if len(lst) != len(self._ids):
                    raise ValueError(f"{name} length {len(lst)} != ids length {len(self._ids)}")
                return lst
            return [int(value)] * len(self._ids)
        except TypeError:
            return [int(value)] * len(self._ids)

    def _set_profile_locked(self, velocities: ScalarOrSeq, accelerations: ScalarOrSeq):
        vs = self._expand_per_id(velocities, "velocity")
        acc = self._expand_per_id(accelerations, "acceleration")
        for dxl_id, v in zip(self._ids, vs):
            r, e = self._packetHandler.write4ByteTxRx(
                self._portHandler, dxl_id, ADDR_PROFILE_VELOCITY, int(v)
            )
            self._check_result(r, e, f"profile velocity id={dxl_id}")
        for dxl_id, a in zip(self._ids, acc):
            r, e = self._packetHandler.write4ByteTxRx(
                self._portHandler, dxl_id, ADDR_PROFILE_ACCELERATION, int(a)
            )
            self._check_result(r, e, f"profile accel id={dxl_id}")

    def _set_profile_locked_safe(self, velocities: ScalarOrSeq, accelerations: ScalarOrSeq):
        vs = self._expand_per_id(velocities, "velocity")
        acc = self._expand_per_id(accelerations, "acceleration")
        with self._io_lock:
            for dxl_id, v in zip(self._ids, vs):
                self._try_write(
                    self._packetHandler.write4ByteTxRx,
                    self._portHandler, dxl_id, ADDR_PROFILE_VELOCITY, int(v)
                )
            for dxl_id, a in zip(self._ids, acc):
                self._try_write(
                    self._packetHandler.write4ByteTxRx,
                    self._portHandler, dxl_id, ADDR_PROFILE_ACCELERATION, int(a)
                )

    def _apply_bus_watchdog(self, lsb: int):
        with self._io_lock:
            for dxl_id in self._ids:
                r, e = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_BUS_WATCHDOG, int(lsb)
                )
                self._check_result(r, e, f"watchdog id={dxl_id}")

    # ------------------ state cache ------------------

    def _snapshot_command_state(self):
        with self._state_lock:
            angle_cont = dict(self._angle_cont)
            last_ticks = dict(self._last_ticks)
        return angle_cont, last_ticks

    def get_joints_cached(self):
        while self._joint_angles is None and not self._stop_thread.is_set():
            time.sleep(0.002)

        with self._state_lock:
            if self._joint_angles is None:
                raise RuntimeError("No joint cache available.")
            q = self._joint_angles.copy()
            ts = self._joint_ts

        age = time.monotonic() - ts
        return q, age

    def get_joints(self) -> np.ndarray:
        q, _ = self.get_joints_cached()
        return q

    def get_joints_dict(self) -> dict:
        q, _ = self.get_joints_cached()
        return {dxl_id: float(angle) for dxl_id, angle in zip(self._ids, q)}

    # ------------------ read thread ------------------

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._read_joint_angles, daemon=True)
        self._reading_thread.start()

    def _read_joint_angles(self, retry: int = 5):
        fail = 0
        dt = 1.0 / float(self._read_hz)
        next_t = time.monotonic()

        while not self._stop_thread.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t += dt

            try:
                with self._io_lock:
                    r = self._groupSyncRead.txRxPacket()
                    if r != COMM_SUCCESS:
                        fail += 1
                        if fail > retry:
                            print("[WARN] read failed repeatedly")
                            fail = 0
                        continue

                    q = np.empty(len(self._ids), dtype=np.float64)

                    for i, dxl_id in enumerate(self._ids):
                        if not self._groupSyncRead.isAvailable(
                            dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                        ):
                            # 不可用时沿用旧值
                            with self._state_lock:
                                q[i] = self._angle_cont[dxl_id]
                            continue

                        raw = int(
                            self._groupSyncRead.getData(
                                dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                            )
                        ) & 0xFFFFFFFF
                        raw16 = raw % TICKS_PER_REV

                        with self._state_lock:
                            if self._last_ticks[dxl_id] is None:
                                self._last_ticks[dxl_id] = raw16
                                self._angle_cont[dxl_id] = raw16 * TICKS2RAD
                            else:
                                step = _unwrap_ticks_step(self._last_ticks[dxl_id], raw16)
                                self._last_ticks[dxl_id] = raw16
                                self._angle_cont[dxl_id] += step * TICKS2RAD

                            q[i] = self._angle_cont[dxl_id]

                with self._state_lock:
                    self._joint_angles = q
                    self._joint_ts = time.monotonic()

            except Exception as ex:
                fail += 1
                if fail > retry:
                    print(f"[WARN] read thread exception repeatedly: {ex}")
                    fail = 0

    # ------------------ control APIs ------------------

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool, retry: int = 3, delay: float = 0.1):
        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE

        with self._io_lock:
            for dxl_id in self._ids:
                success = False
                for attempt in range(retry):
                    dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                        self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                    )

                    if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                        success = True
                        print(f"Torque {'enabled' if enable else 'disabled'} for ID {dxl_id}")
                        break
                    else:
                        print(
                            f"[Retry {attempt + 1}/{retry}] Failed to set torque for ID {dxl_id}: "
                            f"COMM={dxl_comm_result}, ERROR={dxl_error}"
                        )
                        time.sleep(delay)

                if not success:
                    raise RuntimeError(
                        f"Failed to set torque mode for Dynamixel ID {dxl_id} after {retry} attempts."
                    )

        self._torque_enabled = enable

    def set_control_mode(self, mode: int, enable_torque_after: bool = True):
        if mode not in (0, 1, 3, 4, 5, 16):
            raise ValueError("Invalid control mode")

        with self._io_lock:
            for dxl_id in self._ids:
                r, e = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
                )
                self._check_result(r, e, f"torque off id={dxl_id}")

            for dxl_id in self._ids:
                r, e = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_OPERATING_MODE, int(mode)
                )
                self._check_result(r, e, f"set mode id={dxl_id}")

            self._set_profile_locked(self._profile_velocity, self._profile_acc)

            if enable_torque_after:
                for dxl_id in self._ids:
                    r, e = self._packetHandler.write1ByteTxRx(
                        self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
                    )
                    self._check_result(r, e, f"torque on id={dxl_id}")

        self._torque_enabled = bool(enable_torque_after)

    def _goal_current_mA_to_lsb(self, dxl_id: int, mA: float) -> int:
        spec = self._spec_by_id[dxl_id]
        lsb = int(np.round(mA / spec.current_unit_mA_per_lsb))
        lsb = max(-spec.current_limit_max_lsb, min(lsb, spec.current_limit_max_lsb))
        return lsb

    def set_goal_currents_mA(self, currents):
        if len(currents) != len(self._ids):
            raise ValueError("currents length mismatch")

        with self._io_lock:
            for dxl_id, mA in zip(self._ids, currents):
                lsb = self._goal_current_mA_to_lsb(dxl_id, float(mA))
                r, e = self._packetHandler.write2ByteTxRx(
                    self._portHandler, dxl_id, ADDR_GOAL_CURRENT, int(lsb)
                )
                self._check_result(r, e, f"goal current id={dxl_id}")

    def set_joints_current_based_position(self, joint_angles, currents):
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled")
        if len(joint_angles) != len(self._ids) or len(currents) != len(self._ids):
            raise ValueError("len mismatch")

        with self._io_lock:
            for dxl_id, mA in zip(self._ids, currents):
                lsb = self._goal_current_mA_to_lsb(dxl_id, float(mA))
                r, e = self._packetHandler.write2ByteTxRx(
                    self._portHandler, dxl_id, ADDR_GOAL_CURRENT, int(lsb)
                )
                self._check_result(r, e, f"goal current id={dxl_id}")

            for dxl_id, angle in zip(self._ids, joint_angles):
                pos = rad_to_ticks(float(angle)) & 0xFFFFFFFF
                param = [
                    DXL_LOBYTE(DXL_LOWORD(pos)),
                    DXL_HIBYTE(DXL_LOWORD(pos)),
                    DXL_LOBYTE(DXL_HIWORD(pos)),
                    DXL_HIBYTE(DXL_HIWORD(pos)),
                ]
                if not self._groupSyncWrite.addParam(dxl_id, param):
                    raise RuntimeError(f"sync addParam id={dxl_id} failed")

            r = self._groupSyncWrite.txPacket()
            if r != COMM_SUCCESS:
                self._groupSyncWrite.clearParam()
                raise RuntimeError("sync write goal positions failed")

            self._groupSyncWrite.clearParam()

    def _unwrap_near(self, q_des, q_now):
        two_pi = 2 * np.pi
        k = int(np.round((q_now - q_des) / two_pi))
        return q_des + k * two_pi

    def set_joints(self, joint_angles):
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled")
        if len(joint_angles) != len(self._ids):
            raise ValueError("len mismatch")

        angle_cont, last_ticks = self._snapshot_command_state()

        params_to_send = []
        for dxl_id, q_des_in in zip(self._ids, map(float, joint_angles)):
            q_now = angle_cont[dxl_id]
            last_raw = last_ticks[dxl_id]

            q_des_u = self._unwrap_near(q_des_in, q_now)
            d_tick = int(np.round((q_des_u - q_now) / TICKS2RAD))

            if d_tick == 0 and last_raw is not None:
                continue

            if last_raw is None:
                pos_mod = int(
                    np.round((q_des_u % (2 * np.pi)) / (2 * np.pi) * TICKS_PER_REV)
                ) & 0xFFFFFFFF
            else:
                pos_mod = (last_raw + d_tick) % TICKS_PER_REV

            param = [
                DXL_LOBYTE(DXL_LOWORD(pos_mod)),
                DXL_HIBYTE(DXL_LOWORD(pos_mod)),
                DXL_LOBYTE(DXL_HIWORD(pos_mod)),
                DXL_HIBYTE(DXL_HIWORD(pos_mod)),
            ]
            params_to_send.append((dxl_id, param))

        if not params_to_send:
            return

        with self._io_lock:
            for dxl_id, param in params_to_send:
                if not self._groupSyncWrite.addParam(dxl_id, param):
                    self._groupSyncWrite.clearParam()
                    raise RuntimeError(f"sync addParam id={dxl_id} failed")

            r = self._groupSyncWrite.txPacket()
            self._groupSyncWrite.clearParam()

            if r != COMM_SUCCESS:
                raise RuntimeError("sync write goal positions failed")

    # ------------------ close / home helpers ------------------

    def _wrap_pi(self, x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2 * np.pi) - np.pi

    def _majority_cbp(self, default_mode: int = 3) -> bool:
        counts = {3: 0, 5: 0}
        with self._io_lock:
            for dxl_id in self._ids:
                try:
                    mode, r, e = self._packetHandler.read1ByteTxRx(
                        self._portHandler, dxl_id, ADDR_OPERATING_MODE
                    )
                    if r == COMM_SUCCESS and e == 0 and mode in (3, 5):
                        counts[mode] += 1
                except Exception as ex:
                    print(f"[HOME] read mode fail id={dxl_id}: {ex}")

        if counts[3] == counts[5] == 0:
            return (default_mode == 5)
        return counts[5] >= counts[3]

    def _pack_goal_from_near(self, dxl_id: int, q_des_in: float):
        try:
            with self._state_lock:
                q_now = self._angle_cont[dxl_id]
                last_raw = self._last_ticks[dxl_id]

            two_pi = 2 * np.pi
            k = int(np.round((q_now - q_des_in) / two_pi))
            q_des_u = q_des_in + k * two_pi

            d_tick = int(np.round((q_des_u - q_now) / TICKS2RAD))
            if d_tick == 0 and last_raw is not None:
                return None

            if last_raw is None:
                pos_mod = int(
                    np.round((q_des_u % (2 * np.pi)) / (2 * np.pi) * TICKS_PER_REV)
                ) & 0xFFFFFFFF
            else:
                pos_mod = (last_raw + d_tick) % TICKS_PER_REV

            return [
                DXL_LOBYTE(DXL_LOWORD(pos_mod)),
                DXL_HIBYTE(DXL_LOWORD(pos_mod)),
                DXL_LOBYTE(DXL_HIWORD(pos_mod)),
                DXL_HIBYTE(DXL_HIWORD(pos_mod)),
            ]
        except Exception as ex:
            print(f"[HOME] pack goal fail id={dxl_id}: {ex}")
            return None

    def _safe_set_pos(self, q_rad: np.ndarray):
        with self._io_lock:
            any_added = False
            for dxl_id, q in zip(self._ids, map(float, q_rad)):
                param = self._pack_goal_from_near(dxl_id, q)
                if param is None:
                    continue
                ok = self._groupSyncWrite.addParam(dxl_id, param)
                if not ok:
                    print(f"[HOME] addParam fail id={dxl_id}")
                else:
                    any_added = True

            if any_added:
                r = self._groupSyncWrite.txPacket()
                if r != COMM_SUCCESS:
                    print("[HOME] sync write goal positions failed")

            try:
                self._groupSyncWrite.clearParam()
            except Exception:
                pass

    def _safe_set_cbp(self, q_rad: np.ndarray, currents_mA: Sequence[float]):
        with self._io_lock:
            for dxl_id, mA in zip(self._ids, map(float, currents_mA)):
                try:
                    lsb = self._goal_current_mA_to_lsb(dxl_id, mA)
                    self._try_write(
                        self._packetHandler.write2ByteTxRx,
                        self._portHandler, dxl_id, ADDR_GOAL_CURRENT, int(lsb)
                    )
                except Exception as ex:
                    print(f"[HOME] set current fail id={dxl_id}: {ex}")

            any_added = False
            for dxl_id, q in zip(self._ids, map(float, q_rad)):
                param = self._pack_goal_from_near(dxl_id, q)
                if param is None:
                    continue
                ok = self._groupSyncWrite.addParam(dxl_id, param)
                if not ok:
                    print(f"[HOME] addParam fail id={dxl_id}")
                else:
                    any_added = True

            if any_added:
                r = self._groupSyncWrite.txPacket()
                if r != COMM_SUCCESS:
                    print("[HOME] sync write goal positions failed")

            try:
                self._groupSyncWrite.clearParam()
            except Exception:
                pass

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        self._stop_thread.set()
        if hasattr(self, "_reading_thread") and self._reading_thread.is_alive():
            self._reading_thread.join(timeout=1.0)

        with self._io_lock:
            for dxl_id in self._ids:
                self._try_write(
                    self._packetHandler.write2ByteTxRx,
                    self._portHandler, dxl_id, ADDR_GOAL_CURRENT, 0
                )

            for dxl_id in self._ids:
                self._try_write(
                    self._packetHandler.write1ByteTxRx,
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
                )

            for dxl_id in self._ids:
                self._try_write(
                    self._packetHandler.write1ByteTxRx,
                    self._portHandler, dxl_id, ADDR_BUS_WATCHDOG, 0
                )

            try:
                self._groupSyncRead.clearParam()
            except Exception:
                pass
            try:
                self._groupSyncWrite.clearParam()
            except Exception:
                pass

        try:
            self._portHandler.closePort()
        finally:
            print("[INFO] Dynamixel driver closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.home_then_close()
        return False

    def home_then_close(
        self,
        target_deg=(180, 95, 265, 180, 190, 180),
        currents_mA=(500, 900, 900, 500, 500, 500),
        tol_deg: float = 3.0,
        timeout_s: float = 15.0,
        *,
        pre_sleep_s: float = 0.6,
        hold_s: float = 1.0,
        profile_vel: int = 20,
        profile_acc: int = 10,
        refresh_period_s: float = 0.25,
        disable_watchdog: bool = True,
    ):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        homed = False
        try:
            if getattr(self, "_torque_enabled", False):
                if disable_watchdog:
                    with self._io_lock:
                        for dxl_id in self._ids:
                            self._try_write(
                                self._packetHandler.write1ByteTxRx,
                                self._portHandler, dxl_id, ADDR_BUS_WATCHDOG, 0
                            )

                self._set_profile_locked_safe(profile_vel, profile_acc)
                time.sleep(max(0.0, float(pre_sleep_s)))

                use_cbp = self._majority_cbp(default_mode=3)
                target_rad = np.deg2rad(target_deg)

                if use_cbp:
                    self._safe_set_cbp(target_rad, currents_mA)
                else:
                    self._safe_set_pos(target_rad)

                tol = np.deg2rad(tol_deg)
                t0 = time.time()
                last_refresh = t0

                while time.time() - t0 < float(timeout_s):
                    q = self.get_joints()
                    err = np.max(np.abs(self._wrap_pi(target_rad - q)))
                    if err <= tol:
                        homed = True
                        break

                    now = time.time()
                    if (now - last_refresh) >= float(refresh_period_s):
                        if use_cbp:
                            self._safe_set_cbp(target_rad, currents_mA)
                        else:
                            self._safe_set_pos(target_rad)
                        last_refresh = now
                    time.sleep(0.02)

                if homed:
                    t_hold = time.time()
                    while time.time() - t_hold < float(hold_s):
                        now = time.time()
                        if (now - last_refresh) >= float(refresh_period_s):
                            if use_cbp:
                                self._safe_set_cbp(target_rad, currents_mA)
                            else:
                                self._safe_set_pos(target_rad)
                            last_refresh = now
                        time.sleep(0.05)
                else:
                    print("[CLOSE] homing timeout -> fallback hold current pose 3s")
                    hold_q = self.get_joints()
                    t_hold = time.time()
                    while time.time() - t_hold < 3.0:
                        if use_cbp:
                            self._safe_set_cbp(hold_q, currents_mA)
                        else:
                            self._safe_set_pos(hold_q)
                        time.sleep(0.1)
            else:
                print("[CLOSE] torque is off, skip homing.")
        finally:
            self._stop_thread.set()
            if hasattr(self, "_reading_thread") and self._reading_thread.is_alive():
                self._reading_thread.join(timeout=1.0)

            with self._io_lock:
                for dxl_id in self._ids:
                    try:
                        val, r, e = self._packetHandler.read1ByteTxRx(
                            self._portHandler, dxl_id, ADDR_HARDWARE_ERROR_STATUS
                        )
                        if r == COMM_SUCCESS and e == 0:
                            print(f"[DEBUG] ID {dxl_id} Hardware Error Status = 0x{val:02X}")
                    except Exception:
                        pass

                for dxl_id in self._ids:
                    self._try_write(
                        self._packetHandler.write2ByteTxRx,
                        self._portHandler, dxl_id, ADDR_GOAL_CURRENT, 0
                    )
                for dxl_id in self._ids:
                    self._try_write(
                        self._packetHandler.write1ByteTxRx,
                        self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
                    )
                for dxl_id in self._ids:
                    self._try_write(
                        self._packetHandler.write1ByteTxRx,
                        self._portHandler, dxl_id, ADDR_BUS_WATCHDOG, 0
                    )

                try:
                    self._groupSyncRead.clearParam()
                except Exception:
                    pass
                try:
                    self._groupSyncWrite.clearParam()
                except Exception:
                    pass

            try:
                self._portHandler.closePort()
            finally:
                print(f"[INFO] Dynamixel driver closed (homed={homed})")


# ========================= Simple test =========================

def read_position(config):
    from motors_to_robot import dynamixel_to_robot_arm
    print(config.port)
    print(config.ids)
    try:
        with DynamixelDriver(config) as driver:
            driver.set_torque_mode(False)
            for _ in range(10):
                q, age = driver.get_joints_cached()
                q_deg = np.rad2deg(q)
                robot_arm_joint = dynamixel_to_robot_arm(q)
                robot_arm_deg = np.rad2deg(robot_arm_joint)

                print("=====================================")
                print(f"Cache age: {age:.4f}s")
                print(f"Dynamixel joint angles (rad): {q}")
                print(f"Dynamixel joint angles (deg): {_fmt_list(q_deg)}")
                print(f"Robot arm joint angles (rad): {robot_arm_joint}")
                print(f"Robot arm joint angles (deg): {_fmt_list(robot_arm_deg)}")
                time.sleep(1.0)
    except FileNotFoundError:
        print("Dynamixel driver not found")


def current_posi_ctrl(config):
    with DynamixelDriver(config) as driver:
        target_angles_deg = [206.54296875, 161.59179687, 203.76953125, 203.11523437, 239.67773438, 169.36523438]
        target_angles = np.deg2rad(target_angles_deg)
        goal_currents_mA = config.control_currents_mA

        driver.set_joints_current_based_position(target_angles, currents=goal_currents_mA)

        try:
            while True:
                joint_angles, age = driver.get_joints_cached()
                joint_degrees = np.rad2deg(joint_angles)
                print("============ STATE ============")
                print(f"Cache age: {age:.4f}s")
                for idx, deg in zip(config.ids, joint_degrees):
                    print(f"Motor ID {idx}: {deg:.2f}°")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user.")


if __name__ == "__main__":
    from motors_config import make_default_arm_params
    cfg = make_default_arm_params()

    # baudrate_list = [57600, 1000000, 4000000]
    # id_scan_range = range(0, 30)
    # devices = scan_dynamixels(cfg.port, id_scan_range, baudrate_list)

    read_position(cfg)
    # current_posi_ctrl(cfg)