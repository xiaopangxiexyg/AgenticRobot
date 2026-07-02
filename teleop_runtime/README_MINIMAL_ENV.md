# DexTeleop 真机环境安装指南

本文用于在一台机器上安装 `teleop_runtime` 真机采集、遥操作和部署环境。默认环境名为 `dexteleop310`，所有命令都在仓库根目录执行。

## 1. 创建环境

```bash
conda create -n dexteleop310 python=3.10 -y
conda activate dexteleop310

export REPO=/home/zc/HandMotion/Developing/DexTeleop
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
```


```bash
conda activate dexteleop310
conda config --env --set channel_priority strict
conda install -c conda-forge -y \
  "numpy=1.26.4" \
  "scipy=1.13.1" \
  "pinocchio=3.7.0"

cat > /tmp/dexteleop_constraints.txt <<'EOF'
numpy==1.26.4
scipy==1.13.1
EOF
```

不要安装 pip 包 `pin`。本项目使用 conda-forge 的 `pinocchio`。

## 2. 安装基础依赖

```bash
python -m pip install --no-cache-dir \
  -c /tmp/dexteleop_constraints.txt \
  "sapien==3.0.3" \
  "transforms3d==0.4.1" \
  "pymodbus==2.5.3" \
  opencv-python \
  pyserial \
  pyarrow \
  Pillow \
  pyrealsense2 \
  pandas \
  pytest \
  scipy
python -m pip install --no-cache-dir \
  -c /tmp/dexteleop_constraints.txt \
  "qpsolvers==4.11.0" \
  osqp \
  loop-rate-limiters \
  typing-extensions

python -m pip install --no-cache-dir \
  -c /tmp/dexteleop_constraints.txt \
  --no-deps \
  "pin-pink==4.2.0"

conda activate dexteleop310
export REPO=/home/zc/HandMotion/Developing/DexTeleop
cd "$REPO"

export DMTAC_SDK_ROOT="$REPO/assets/DM_Tac_SDK_Publish_1.2.10"
cd "$DMTAC_SDK_ROOT"
python -m pip install -e .

cd "$REPO/hardware/DMTAC_gripper"
python -m pip install --no-cache-dir \
  -c /tmp/dexteleop_constraints.txt \
  -r requirements.txt
```

`pip check` 可能提示 `pin-pink requires pin`，这是预期的 metadata 提示。不要为了消除它安装 pip 包 `pin`。

VIVE 发送端才需要：

```bash
python -m pip install --no-cache-dir \
  -c /tmp/dexteleop_constraints.txt \
  triad-openvr
```


运行遥操作时，DMTAC gripper 默认只连接夹爪，不读取触觉相机。需要触觉数据或触觉可视化时再显式加：

```bash
--dmtac-with-tactile
```

## 4. 验证安装

基础数值栈：

```bash
python - <<'PY'
import importlib.metadata as metadata
import numpy as np
import scipy
import pinocchio as pin

assert np.__version__ == "1.26.4", np.__version__
assert scipy.__version__ == "1.13.1", scipy.__version__
try:
    metadata.version("pin")
except metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError("Remove pip package 'pin'. Use conda-forge pinocchio only.")

print("numpy", np.__version__)
print("scipy", scipy.__version__)
print("pinocchio", pin.__version__)
PY
```

项目和硬件 backend：

```bash
python - <<'PY'
from hardware.rm65.robotic_arm_package.robotic_arm import RM65, dllPath
from teleop_runtime.backends.arms import RM65ArmBackend, RM65ImpedanceArmBackend
from teleop_runtime.backends.grippers import DMTacGripperBackend
from teleop_runtime.backends.hands import InspireTacHandBackend

print("RM65 SDK", RM65, dllPath)
print("teleop_runtime imports ok")
PY
```

双臂防碰撞 IK：

```bash
python - <<'PY'
import pink
import qpsolvers
from teleop_runtime.retargeters.dual_arm_collision_aware import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
)

assert "osqp" in qpsolvers.available_solvers, qpsolvers.available_solvers
retargeter = DualArmCollisionAwareRetargeter(
    DualArmCollisionAwareRetargeterConfig(scene_profile="dual_arm_hand_proxy_open")
)
print("dual-arm IK ok", retargeter.profile.name, len(retargeter.profile.collision_pairs))
PY
```

DM-Tac 夹爪：

```bash
cd /home/password-1/DexTeleop/hardware/DMTAC_gripper
python run_gripper.py --help
```

RealSense：

```bash
python - <<'PY'
import pyrealsense2 as rs
print("pyrealsense2", getattr(rs, "__version__", "unknown"))
PY
```

## 5. 设备权限

运行 GUI 遥操作前先检查 USB、相机、串口和配置：

```bash
python -m teleop_runtime.configure_devices --check
```

如果提示权限不足，执行：

```bash
sudo -E /home/password-1/miniconda3/envs/dexteleop310/bin/python \
  -m teleop_runtime.configure_devices --fix-permissions
```

如果新接 DM-Tac gripper 后 `/dev/ttyUSB*` 不出现，或 `lsusb -t` 看到 CH340/CP210x 是
`Driver=usbfs`，优先按 USB-Serial 抢占处理。若这台机器不需要盲文终端，可执行：

```bash
sudo -E /home/password-1/miniconda3/envs/dexteleop310/bin/python \
  -m teleop_runtime.configure_devices --fix-serial-conflicts
```

之后重新拔插 gripper USB/HUB，再运行 `--fix-permissions`。如果机器需要盲文终端，不要禁用
`brltty`，改用设备专用 udev 例外。

长期建议安装 udev rules，或把当前用户加入 `dialout`、`video` 后重新登录并重新插拔设备。

## 6. 常用运行方式

双臂 RM65 + VIVE + DMTAC gripper：

```bash
export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20

python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --load-profile teleop_records/calibration_profiles/20260627_190203_vive_rm65_virtual_desk.json \
  --gui \
  --hand-backend dmtac-gripper \
  --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json
```

如果要同时读取 DM-Tac 触觉：

```bash
  --dmtac-with-tactile
```

只做无相机调试：

```bash
  --no-camera
```

跳过 GUI 设备预检：

```bash
  --no-device-preflight
```

## 7. 故障速查

- `ModuleNotFoundError: pinocchio`：确认已用 conda 安装 `pinocchio=3.7.0`。
- `cmeel-boost requires numpy>=2`：环境被 pip `pin` 污染，建议重建环境。
- `pymodbus` import 路径错误：确认版本是 `pymodbus==2.5.3`。
- RealSense 找不到：检查 librealsense/udev 规则和相机是否被其他进程占用。
- 串口不可读写：先运行 `configure_devices --fix-permissions`。
- `/dev/ttyUSB*` 不出现或 `Driver=usbfs`：先运行 `configure_devices --fix-serial-conflicts`，重新拔插后再修权限。
- GUI 卡顿且日志出现多个 DM-Tac reset：确认没有误加 `--dmtac-with-tactile` 或触觉可视化。
