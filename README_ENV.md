# DexTeleop 真机与 real2sim 环境安装指南

本文用于安装 `teleop_runtime` 真机采集、遥操作、部署和 real2sim 检查环境。默认真机环境名为
`dexteleop310`。除特别说明外，所有命令都在仓库根目录执行。

## 1. 创建环境

```bash
conda create -n dexteleop310 python=3.10 -y
conda activate dexteleop310

cd <DexTeleop仓库根目录>
export REPO="$(pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export DEXTELEOP_CONSTRAINTS="$REPO/.cache/dexteleop_constraints.txt"
```


```bash
conda activate dexteleop310
conda config --env --set channel_priority strict
conda install -c conda-forge -y \
  "numpy=1.26.4" \
  "scipy=1.13.1" \
  "pinocchio=3.7.0"

mkdir -p "$REPO/.cache"
cat > "$DEXTELEOP_CONSTRAINTS" <<'EOF'
numpy==1.26.4
scipy==1.13.1
EOF
```

不要安装 pip 包 `pin`。本项目使用 conda-forge 的 `pinocchio`。

## 2. 安装基础依赖

```bash
python -m pip install --no-cache-dir \
  -c "$DEXTELEOP_CONSTRAINTS" \
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
  pyarrow \
  scipy \
  pyyaml \
  websockets \
  msgpack
python -m pip install --no-cache-dir \
  -c "$DEXTELEOP_CONSTRAINTS" \
  "qpsolvers==4.11.0" \
  osqp \
  loop-rate-limiters \
  typing-extensions

python -m pip install --no-cache-dir \
  -c "$DEXTELEOP_CONSTRAINTS" \
  --no-deps \
  "pin-pink==4.2.0"

conda activate dexteleop310
cd "$REPO"

export DMTAC_SDK_ROOT="$REPO/assets/DM_Tac_SDK_Publish_1.2.10"
cd "$DMTAC_SDK_ROOT"
python -m pip install -e .

cd "$REPO/hardware/DMTAC_gripper"
python -m pip install --no-cache-dir \
  -c "$DEXTELEOP_CONSTRAINTS" \
  -r requirements.txt
```

`pip check` 可能提示 `pin-pink requires pin`，这是预期的 metadata 提示。不要为了消除它安装 pip 包 `pin`。

VIVE 发送端才需要：

```bash
python -m pip install --no-cache-dir \
  -c "$DEXTELEOP_CONSTRAINTS" \
  triad-openvr
```


运行遥操作时，DMTAC gripper 默认只连接夹爪，不读取触觉相机。需要触觉数据或触觉可视化时再显式加：

```bash
--dmtac-with-tactile
```

## 3. 安装 Isaac Gym 环境

Isaac Gym Preview 4 用于后续 real2sim 和相机位姿检查。不要安装到 `dexteleop310`：包内
`setup.py` 声明只支持 Python `>=3.6,<3.9`，而 `dexteleop310` 是 Python 3.10。建议单独创建
`isaacgym38` 环境，和真机采集环境隔离。

Isaac Gym Preview 4 需要从 NVIDIA Developer 下载离线包。下载后设置包路径；若已经解压过，直接设置
`ISAACGYM_ROOT` 即可：

```bash
export ISAACGYM_ARCHIVE=<IsaacGym_Preview_4_Package.tar.gz>
export ISAACGYM_ROOT=<isaacgym解压目录>
```

如果 `ISAACGYM_ROOT` 不存在，先解压：

```bash
mkdir -p "$(dirname "$ISAACGYM_ROOT")"
tar -xzf "$ISAACGYM_ARCHIVE" -C "$(dirname "$ISAACGYM_ROOT")"
```

基础要求：

- Ubuntu 18.04/20.04 风格系统；新系统通常也可运行，但要以实际 smoke test 为准。
- NVIDIA driver 版本不低于 470。
- Python 3.6/3.7/3.8。推荐 Python 3.8，因为包内有 `gym_38.so`。

检查 GPU/驱动：

```bash
nvidia-smi
```

创建独立环境并安装：

```bash
conda create -n isaacgym38 python=3.8 -y
conda activate isaacgym38

python -m pip install --upgrade pip setuptools wheel

# 使用 Isaac Gym Preview 4 官方 rlgpu yml 对应的老版本栈，避免 pip 拉到不兼容 Python 3.8 的新版 Torch。
conda install -c pytorch -c conda-forge -y \
  "pytorch=1.8.1" \
  "torchvision=0.9.1" \
  "cudatoolkit=11.1" \
  "numpy>=1.16.4" \
  "scipy>=1.5.0" \
  "pyyaml>=5.3.1" \
  "tensorboard>=2.2.1"

python -m pip install --no-cache-dir pillow imageio ninja

cd "$ISAACGYM_ROOT/python"
python -m pip install --no-deps -e .
```

验证 Isaac Gym Python package：

```bash
conda activate isaacgym38

python - <<'PY'
from isaacgym import gymapi
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
print("isaacgym import ok; sim created:", sim is not None)
if sim is not None:
    gym.destroy_sim(sim)
PY
```

验证 viewer 和 asset 加载。该示例必须在 Isaac Gym 的 examples 目录运行，因为示例使用相对资产路径：

```bash
cd "$ISAACGYM_ROOT/python/examples"
python joint_monkey.py
```

如果 viewer 找不到 NVIDIA Vulkan 设备，可先指定 NVIDIA ICD：

```bash
export VK_ICD_FILENAMES=<path-to-nvidia_icd.json>
```

官方脚本可作为替代方案：

```bash
cd "$ISAACGYM_ROOT"
./create_conda_env_rlgpu.sh
conda activate rlgpu
```

注意：官方脚本会删除并重建名为 `rlgpu` 的环境；已有同名环境时不要直接运行。

后续 real2sim 相机检查优先使用这里生成的双臂 URDF：

```bash
export DEXTELEOP_REAL2SIM_URDF="$REPO/assets/rm65_inspire_gripper_v1/rm65_dual_arm_gripper_v1.urdf"
```

该 URDF 是自包含相对 mesh 路径，已包含：

- `world`、`table`、`camera_pole`
- 左右 RM65 基座固定外参
- 左右 `arm_link6` 末端挂载的法兰、D405、`left_camera_rgb` / `right_camera_rgb`
- 左右 gripper 几何

## 4. 安装 MuJoCo 环境

MuJoCo 用于更轻量的相机、几何和坐标系检查。官方 Python binding 发布为 `mujoco` 包，包内已经包含
MuJoCo library，不需要单独下载 MuJoCo 二进制。它可以直接安装到 `dexteleop310`，方便复用项目代码和
URDF 资产。

```bash
conda activate dexteleop310
cd "$REPO"

python -m pip install --no-cache-dir \
  -c "$DEXTELEOP_CONSTRAINTS" \
  mujoco
```

验证 MuJoCo Python package：

```bash
python - <<'PY'
import mujoco
print("mujoco", mujoco.__version__)
PY
```

验证 viewer：

```bash
python -m mujoco.viewer
```

后续 real2sim 相机检查仍然优先使用同一个双臂 URDF：

```bash
export DEXTELEOP_REAL2SIM_URDF="$REPO/assets/rm65_inspire_gripper_v1/rm65_dual_arm_gripper_v1.urdf"
```

先做最小加载检查：

```bash
python - <<'PY'
import os
import mujoco

model_path = os.environ["DEXTELEOP_REAL2SIM_URDF"]
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("mujoco model ok")
print("nbody", model.nbody, "ngeom", model.ngeom, "ncam", model.ncam)
for name in ("left_camera_rgb", "right_camera_rgb"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    print(name, "body_id", body_id)
PY
```

如果该 URDF 不能被 MuJoCo 直接编译，下一步应保留 URDF 作为真实装配基准，再生成一个 MJCF wrapper 或转换后的
MJCF 用于 MuJoCo 检查，避免为了仿真破坏原始坐标定义。

参考：[MuJoCo 官方 Python 文档](https://mujoco.readthedocs.io/en/stable/python.html)推荐通过
`pip install mujoco` 安装，并说明 package 内置 MuJoCo library。

## 5. 验证安装

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
cd "$REPO/hardware/DMTAC_gripper"
python run_gripper.py --help
```

RealSense：

```bash
python - <<'PY'
import pyrealsense2 as rs
print("pyrealsense2", getattr(rs, "__version__", "unknown"))
PY
```

Isaac Gym real2sim 环境：

```bash
conda activate isaacgym38
cd "$ISAACGYM_ROOT/python/examples"
python joint_monkey.py
```

MuJoCo real2sim 环境：

```bash
conda activate dexteleop310
python - <<'PY'
import mujoco
print("mujoco", mujoco.__version__)
PY
```

## 6. 设备权限

运行 GUI 遥操作前先检查 USB、相机、串口和配置：

```bash
python -m teleop_runtime.configure_devices --check
```

如果提示权限不足，执行：

```bash
sudo -E "$CONDA_PREFIX/bin/python" \
  -m teleop_runtime.configure_devices --fix-permissions
```

如果新接 DM-Tac gripper 后 `/dev/ttyUSB*` 不出现，或 `lsusb -t` 看到 CH340/CP210x 是
`Driver=usbfs`，优先按 USB-Serial 抢占处理。若这台机器不需要盲文终端，可执行：

```bash
sudo -E "$CONDA_PREFIX/bin/python" \
  -m teleop_runtime.configure_devices --fix-serial-conflicts
```

之后重新拔插 gripper USB/HUB，再运行 `--fix-permissions`。如果机器需要盲文终端，不要禁用
`brltty`，改用设备专用 udev 例外。

长期建议安装 udev rules，或把当前用户加入 `dialout`、`video` 后重新登录并重新插拔设备。

## 7. 常用运行方式

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

## 8. 故障速查

- `ModuleNotFoundError: pinocchio`：确认已用 conda 安装 `pinocchio=3.7.0`。
- `cmeel-boost requires numpy>=2`：环境被 pip `pin` 污染，建议重建环境。
- `pymodbus` import 路径错误：确认版本是 `pymodbus==2.5.3`。
- RealSense 找不到：检查 librealsense/udev 规则和相机是否被其他进程占用。
- Isaac Gym import 失败：确认在 `isaacgym38` 或 `rlgpu` 环境中，且 Python 版本是 3.6/3.7/3.8。
- Isaac Gym viewer 黑屏或无法创建设备：先检查 `nvidia-smi`，再尝试设置 `VK_ICD_FILENAMES=<path-to-nvidia_icd.json>`。
- Isaac Gym 报 `libpython3.x` 或 `libstdc++`：优先设置 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`，再重新运行 smoke test。
- MuJoCo import 失败：确认在 `dexteleop310` 环境中，并重新执行 `python -m pip install mujoco`。
- MuJoCo viewer 无法打开：先确认当前会话有桌面显示权限；远程或无显示环境下优先做 import 和模型加载检查。
- MuJoCo 直接加载 URDF 失败：保留 URDF 作为装配真值，单独生成 MJCF 检查文件，不要在 URDF 里混入 MuJoCo 专用 workaround。
- 串口不可读写：先运行 `configure_devices --fix-permissions`。
- `/dev/ttyUSB*` 不出现或 `Driver=usbfs`：先运行 `configure_devices --fix-serial-conflicts`，重新拔插后再修权限。
- GUI 卡顿且日志出现多个 DM-Tac reset：确认没有误加 `--dmtac-with-tactile` 或触觉可视化。
