from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional

@dataclass
class ModelSpec:
    name: str
    current_unit_mA_per_lsb: float
    current_limit_max_lsb: int
    velocity_limit_max_lsb: int = 1023      # Velocity Limit(44)
    pos_ticks_per_rev: int = 4096           # 0.088°/tick
    profile_vel_max_lsb: int = 32767        # Profile Velocity(112)  0~32767速度型Profile）
    profile_acc_max_lsb: int = 32767        # Profile Acceleration(108)  0~32767（速度型Profile，单位≈214.577 rpm/min^2）

XM430_W350 = ModelSpec("XM430-W350-T", current_unit_mA_per_lsb=2.69, current_limit_max_lsb=1193, velocity_limit_max_lsb=1023)
XM540_W270 = ModelSpec("XM540-W270-T", current_unit_mA_per_lsb=2.69, current_limit_max_lsb=2047, velocity_limit_max_lsb=1023)
XL330_M288 = ModelSpec("XL330-M288-T", current_unit_mA_per_lsb=1.00, current_limit_max_lsb=1750, velocity_limit_max_lsb=2047)
XC330_M288 = ModelSpec("XC330-M288-T", current_unit_mA_per_lsb=1.00, current_limit_max_lsb=2352, velocity_limit_max_lsb=2047)

# 通用单位换算常量（X 系列一致
RPM_PER_LSB = 0.229        # 速度寄存器单位：≈0.229 rpm/LSB（Goal/Limit/Profile Velocity）
ACC_RPM_PER_MIN2 = 214.577 # Profile Acceleration 单位：≈214.577 rev/min^2/LSB
DEG_PER_TICK = 0.088       # 角度分辨率：≈0.088°/tick（4096 tick/rev）

# 换算函数
def current_mA_to_lsb(mA: float, spec: ModelSpec) -> int:
    return int(round(mA / spec.current_unit_mA_per_lsb))

def velocity_rpm_to_lsb(rpm: float) -> int:
    return int(round(rpm / RPM_PER_LSB))

def ticks_to_deg(ticks: int) -> float:
    return ticks * DEG_PER_TICK

# 参数对象
@dataclass
class MotorRobotParams:
    # 通信
    port: str
    baudrate: int
    # ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    ids: List[int]
    id2spec: Dict[int, ModelSpec] = field(default_factory=dict)  # {id: ModelSpec}

    protocol_version: float = 2.0
    operating_mode: int = 5  # 0=Current,1=Velocity,3=Position,4=Extended,5=Current-based Position,16=PWM
    torque_enable_on_boot: bool = False
    # Bus Watchdog(98)：1~127 启用，超时=值×20 ms；0 关闭并清错。超时停止并将 Goal* 改为只读（读数=-1）。
    bus_watchdog_ms: int = 2000

    # 轨迹 Profile（位置/电流位置模式生效；纯速度模式仅 Accel 生效）
    # Profile Velocity(112)：0~32767 LSB（≈0.229 rpm/LSB），但实际速度受 Velocity Limit(44) 限制
    control_velocitys: List[int] = field(default_factory=list)   # 每关节 Profile Velocity(112) LSB
    # Profile Acceleration(108)：0~32767 LSB（≈214.577 rev/min^2/LSB 或由 Drive Mode 决定时间基准）
    accelerations: List[int] = field(default_factory=list)       # 每关节 Profile Accel(108) LSB

    # 目标电流（仅在 Current / Current-based Position 模式使用）
    # Goal Current(102)：单位见型号；不得超过 Current Limit(38)
    control_currents_mA: List[int] = field(default_factory=list) # 以 mA 存储，便于直观设定

    # 便捷常量（只读）
    units: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(
        rpm_per_lsb=RPM_PER_LSB,
        acc_rpm_per_min2_per_lsb=ACC_RPM_PER_MIN2,
        deg_per_tick=DEG_PER_TICK
    ))

    def _ensure_lengths(self):
        n = len(self.ids)
        if not self.id2spec or any(i not in self.id2spec for i in self.ids):
            raise ValueError("id2spec 必须覆盖所有 ids。")
        def _fill(lst: List[int], fill):
            return lst if len(lst) == n else (lst + [fill] * (n - len(lst)))[:n]
        # 缺省 Profile：Vel=100LSB(~22.9rpm)，Accel=50LSB；电流默认按各型号上限的 30%
        defaults_curr = [int(0.3 * self.id2spec[i].current_limit_max_lsb * self.id2spec[i].current_unit_mA_per_lsb) for i in self.ids]
        self.control_velocitys = _fill(self.control_velocitys, 100)
        self.accelerations     = _fill(self.accelerations, 50)
        self.control_currents_mA = _fill(self.control_currents_mA, None)
        self.control_currents_mA = [defaults_curr[k] if v is None else v for k, v in enumerate(self.control_currents_mA)]

    def _clip_to_hw(self):
        """将 Profile/Current 裁剪到各型号硬范围内。"""
        self._ensure_lengths()
        for idx, i in enumerate(self.ids):
            spec = self.id2spec[i]
            # Profile Vel/Accel
            self.control_velocitys[idx] = max(0, min(self.control_velocitys[idx], spec.velocity_limit_max_lsb))
            self.accelerations[idx]     = max(0, min(self.accelerations[idx],     spec.profile_acc_max_lsb))
            # Current
            max_mA = int(spec.current_limit_max_lsb * spec.current_unit_mA_per_lsb)
            self.control_currents_mA[idx] = max(-max_mA, min(self.control_currents_mA[idx], max_mA))

    @property
    def bus_watchdog_lsb(self) -> int:
        """Bus Watchdog(98) 的寄存器值：0 关闭；1~127 每步 20ms。"""
        if self.bus_watchdog_ms <= 0:
            return 0
        return max(1, min(127, round(self.bus_watchdog_ms / 20)))

    def by_id(self, i: int) -> SimpleNamespace:
        """按 ID 查看该关节的型号、范围与当前设定。"""
        self._clip_to_hw()
        idx = self.ids.index(i)
        spec = self.id2spec[i]
        max_mA = int(spec.current_limit_max_lsb * spec.current_unit_mA_per_lsb)
        return SimpleNamespace(
            id=i,
            model=spec.name,
            current_unit_mA_per_lsb=spec.current_unit_mA_per_lsb,                 # XM≈2.69；XL/XC=1.0
            current_limit_max_mA=max_mA,                                          # XM430≈3210；XM540≈5500；XL330≈1750；XC330≈2352
            velocity_limit_max_rpm=int(spec.velocity_limit_max_lsb * RPM_PER_LSB),# XM*≈234；XL/XC≈469（理论上限）
            pos_ticks_per_rev=spec.pos_ticks_per_rev,                              # 0~4095
            profile_velocity_lsb=self.control_velocitys[idx],
            profile_accel_lsb=self.accelerations[idx],
            goal_current_mA=self.control_currents_mA[idx],
            goal_current_lsb=current_mA_to_lsb(self.control_currents_mA[idx], spec)
        )


def make_default_arm_params(
    model_by_id: Optional[Dict[int, ModelSpec]] = None,  # 传 {3: XL330_M288, 4: XC330_M288, ...} 覆盖
    ) -> MotorRobotParams:

    default_id2spec: Dict[int, ModelSpec] = {
        21: XM430_W350,
        22: XM540_W270,
        23: XM430_W350,
        24: XM540_W270,
        25: XM430_W350,
        26: XM430_W350,
        27: XM430_W350,
    }
    id2spec = default_id2spec.copy()
    if model_by_id:
        for k, v in model_by_id.items():
            if k in id2spec:
                id2spec[k] = v

    ap = MotorRobotParams(
        port="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0",
        baudrate=57600,
        protocol_version=2.0,
        ids=[21, 22, 23, 24, 25, 26, 27],
        id2spec=id2spec,
        operating_mode=5,                 # Current-based Position
        torque_enable_on_boot=False,
        bus_watchdog_ms=200,#2000
        # control_velocitys=[80, 100, 80, 80, 60, 60],
        # accelerations = [50, 60, 50, 50, 40, 40],
        # control_currents_mA = [500, 900, 500, 500, 500, 500], # 留空→按各型上限 30% 自动给默认值
        control_velocitys=[30, 25, 30, 25, 30, 30, 30],
        accelerations=[50, 40, 50, 40, 50, 40, 40],
        control_currents_mA=[1000, 2000, 1000, 2000, 1000, 1000, 1000],  # 留空→按各型上限 30% 自动给默认值
    )
    ap._clip_to_hw()
    return ap


if __name__ == "__main__":
    cfg = make_default_arm_params()
    print("port:", cfg.port)
    print("ids:", cfg.ids)
    print("bus_watchdog_lsb:", cfg.bus_watchdog_lsb)

    # 查看某关节参数
    j1 = cfg.by_id(1)
    print("id1:", j1.model, "goal_current_mA:", j1.goal_current_mA, "vel_limit_rpm:", j1.velocity_limit_max_rpm)

    # 覆盖某关节型号为 XL330/XC330
    cfg2 = make_default_arm_params(model_by_id={3: XL330_M288, 4: XC330_M288})
    print("id3 model:", cfg2.by_id(3).model, "id4 model:", cfg2.by_id(4).model)