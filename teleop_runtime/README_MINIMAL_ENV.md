# Realman RM65 / InspireTac 最小真机环境

这份说明面向把模型环境补齐为 `teleop_runtime` 的 RM65 真机部署/采集环境。`teleop_runtime` 不是单目录包，运行时还依赖仓库里的 RM65 SDK、InspireTac 驱动、IK 资产和可选 VIVE/q6 输入代码。

## 先确认的阻塞点

- RM65 arm backend 必须能导入 `hardware.rm65.robotic_arm_package.robotic_arm`，并能加载 `hardware/rm65/robotic_arm_package/libRM_Base.so`。
- RM65 笛卡尔 pose 策略会调用 `rm65_inspire_tac.ik.SingleArmIKSolver`，因此 arm 部署需要 `pinocchio`。当前可用 pip 包组合是 `pin==3.7.0`，导入名是 `pinocchio`。
- 双臂 VIVE 采集/遥操作默认会启用 joint-space 碰撞规避，需要 `pin-pink`、`qpsolvers` 和一个 QP backend，例如 `osqp`。如果只做策略部署到 RM65 pose，可以先不装 `pin-pink`。
- 当前 `teleop_runtime` 的 InspireTac backend 导入 `hardware.inspire_tac.inspire_tac_modbus`，该文件顶层导入 `cv2`；所以只连真手也需要 `opencv-python`，除非后续改成无 `cv2` 的轻量驱动。
- `pymodbus` 必须是 `2.5.3`，`pymodbus 3.x` 不兼容当前 Modbus import 路径。

## 本地代码依赖

从仓库根目录运行，或把仓库根目录放进 `PYTHONPATH`：

```bash
export REPO=$(pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
```

最小文件/目录：

- `teleop_runtime/`
- `rm65_inspire_tac/`
- `assets/rm65_inspire_tac/`，至少包含 `rm65_arm.urdf` 和相关 meshes
- `hardware/rm65/robotic_arm_package/`，包含 `robotic_arm.py` 和 `libRM_Base.so*`
- `hardware/inspire_tac/`，连接 InspireTac 真手时需要
- `hardware/teleop_hand/`，q6 串口手套输入时需要
- `hardware/VIVE/`，只在 VIVE 发送端或相关调试脚本上需要
- `teleop_records/calibration_profiles/`，加载历史 VIVE/RM65 标定时需要

## Python 包

所有 `teleop_runtime` 部署入口的基础包：

```bash
python -m pip install "numpy<2" scipy
```

RM65 arm 部署：

```bash
python -m pip install "pin==3.7.0"
```

InspireTac 真手和 q6 串口输入：

```bash
python -m pip install "pymodbus==2.5.3" opencv-python pyserial
```

双臂 VIVE 采集/遥操作的碰撞规避 IK：

```bash
python -m pip install "pin-pink==4.1.0" "qpsolvers==4.11.0" osqp loop-rate-limiters
```

采集/回放 parquet、相机、VIVE 发送端按需安装：

```bash
python -m pip install pyarrow pyrealsense2
python -m pip install triad-openvr  # only on the VIVE sender machine
```

说明：`pin-pink` 的 Python import 名是 `pink`；`pin` 的 Python import 名是 `pinocchio`。

## 导入检查

基础 RM65 + InspireTac 部署：

```bash
python - <<'PY'
import importlib.util as u

required = [
    "numpy",
    "scipy",
    "pinocchio",
    "pymodbus",
    "cv2",
    "teleop_runtime",
    "rm65_inspire_tac",
    "hardware",
]
optional = ["serial", "pyarrow", "pyrealsense2", "pink", "qpsolvers"]
for name in required + optional:
    print(f"{name}: {bool(u.find_spec(name))}")

from hardware.rm65.robotic_arm_package.robotic_arm import RM65, dllPath
from teleop_runtime.backends.arms import RM65ArmBackend
from teleop_runtime.backends.hands import InspireTacHandBackend
print("rm65 sdk ok", RM65, dllPath)
print("teleop_runtime hardware imports ok")
PY
```

双臂碰撞规避 IK 检查：

```bash
python - <<'PY'
import pink
import qpsolvers
print("pink", getattr(pink, "__version__", "unknown"))
print("qpsolvers", getattr(qpsolvers, "__version__", "unknown"))
print("available_solvers", qpsolvers.available_solvers)
assert qpsolvers.available_solvers, "no QP solver backend installed"
PY
```

## 硬件权限与网络

- RM65 需要部署机能连通左右臂 IP，典型参数是 `--left-arm-ip`、`--right-arm-ip`。
- InspireTac 默认 Modbus 端口是 `6000`，手 IP 由 `--left-hand-ip/--right-hand-ip` 指定。
- q6 串口优先使用 `/dev/serial/by-id/...`；没有权限时把当前用户加入 `dialout` 后重新登录。
- VIVE receiver 不需要 `triad_openvr`；只有读取 SteamVR tracker 并发送 UDP 的机器需要 `triad-openvr`。
- RealSense 需要系统已安装 librealsense udev 规则，且相机没有被其他进程占用。

## 最小 smoke test

纯 dry-run，不连接真机、不发送动作：

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy hold \
  --active-sides right \
  --camera-mode none \
  --max-steps 5 \
  --auto-start
```

只连接单臂 RM65 并读取状态，仍不发送动作：

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy hold \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --camera-mode none \
  --max-steps 30 \
  --auto-start
```

单臂 + InspireTac 手 dry-run：

```bash
python -m teleop_runtime.deploy.run_policy \
  --policy hold \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --enable-hands \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-tactile-regions "" \
  --camera-mode none \
  --max-steps 30 \
  --auto-start
```

真实下发必须显式加 `--send-actions`；未加 `--auto-start` 时还需要按 `g` 才会执行策略。
