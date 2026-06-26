# Realman RM65 / InspireTac 最小真机环境

这份说明面向把模型环境补齐为 `teleop_runtime` 的 RM65 真机部署/采集环境。`teleop_runtime` 不是单目录包，运行时还依赖仓库里的 RM65 SDK、InspireTac 驱动、IK 资产和可选 VIVE/q6 输入代码。

## 先确认的阻塞点

- RM65 arm backend 必须能导入 `hardware.rm65.robotic_arm_package.robotic_arm`，并能加载 `hardware/rm65/robotic_arm_package/libRM_Base.so`。
- RM65 笛卡尔 pose 策略会调用 `rm65_inspire_tac.ik.SingleArmIKSolver`，因此 arm 部署需要 `pinocchio`。当前推荐使用 conda-forge `pinocchio=3.7.0`，不要安装 pip 包 `pin`。
- 双臂 VIVE 采集/遥操作的 joint-space 碰撞规避需要 `pin-pink`、`qpsolvers` 和一个 QP backend，例如 `osqp`。当前最小 viewer/部署环境先用 `--no-collision`，不要默认安装 `pin-pink`。
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
- `hardware/rm65/utils/configs/calibrations/`，可选；真机 arm backend 会读取
  当前 active RealMan tool name，并自动加载匹配的 `rm65_{arm}_{tool}.json`
  TCP profile。
- `hardware/inspire_tac/`，连接 InspireTac 真手时需要
- `hardware/teleop_hand/`，q6 串口手套输入时需要
- `hardware/VIVE/`，只在 VIVE 发送端或相关调试脚本上需要
- `teleop_records/calibration_profiles/`，加载历史 VIVE/RM65 标定时需要

## Python 包

### 版本锁定原则

优先使用一个干净的 Python 3.10 环境。不要在已经混装过 `numpy>=2`、`pin/cmeel`、`scipy` 的环境里继续覆盖安装；一旦发现 `pip check` 报 `cmeel-boost requires numpy>=2.0` 或同一包存在多个 `.dist-info`，建议重建环境。

当前 SAPIEN viewer、`transforms3d` 和项目里的 RM65/Pinocchio 路线以 NumPy 1.x 为基准。下面版本必须严格保持：

- `numpy==1.26.4`
- `scipy==1.13.1`
- conda-forge `pinocchio=3.7.0`

安装包时统一使用 `python -m pip`，不要混用裸 `pip`，避免写进错误的 conda/env。
不要在这个环境里安装 pip 包 `pin`；当前 `pin` 会拉取需要 NumPy 2.x 的 `cmeel-boost`，会和 SAPIEN viewer 的 NumPy 1.x 路线冲突。
后续每条 `python -m pip install ...` 都使用 constraints 文件，防止 resolver 因可选包把 NumPy/SciPy 升级到不兼容版本。

### 推荐新建环境

如果已经在同名环境里执行过失败的 `python -m pip install pin...` 或出现过 `cmeel-boost requires numpy>=2`，先删除环境再重建：

```bash
conda deactivate
conda remove -n dexteleop310 --all -y
```

```bash
conda create -n dexteleop310 python=3.10 -y
conda activate dexteleop310

cd /home/password-1/DexTeleop
export REPO=$(pwd)
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
```

所有 `teleop_runtime` 部署入口的基础包：

```bash
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

RM65 arm 部署依赖上面的 conda-forge `pinocchio`。不要再运行 `python -m pip install pin==...`。

SAPIEN 坐标系/双臂可视化：

```bash
python -m pip install --no-cache-dir \
  -c /tmp/dexteleop_constraints.txt \
  "sapien==3.0.3" \
  "transforms3d==0.4.1"
```

InspireTac 真手和 q6 串口输入：

```bash
python -m pip install --no-cache-dir \
  -c /tmp/dexteleop_constraints.txt \
  "pymodbus==2.5.3" \
  opencv-python \
  pyserial
```

双臂 VIVE 采集/遥操作的碰撞规避 IK 先默认关闭，运行入口加 `--no-collision`。先不要把 `pin-pink` 和 SAPIEN viewer 装进同一个环境；当前 pip resolver 容易重新拉入 `pin/cmeel` 并把 NumPy 推到 2.x。

如果只想准备 QP 相关包，可以先装不触碰 Pinocchio 的部分：

```bash
python -m pip install --no-cache-dir \
  -c /tmp/dexteleop_constraints.txt \
  "qpsolvers==4.11.0" \
  osqp \
  loop-rate-limiters
```

需要恢复碰撞规避时，再单独验证 `pin-pink` 与 conda-forge `pinocchio` 的组合；验证通过前不要放松 `numpy==1.26.4` / `scipy==1.13.1`。

采集/回放 parquet、相机、VIVE 发送端按需安装：

```bash
python -m pip install --no-cache-dir \
  -c /tmp/dexteleop_constraints.txt \
  pyarrow \
  pyrealsense2 \
  Pillow
python -m pip install --no-cache-dir \
  -c /tmp/dexteleop_constraints.txt \
  triad-openvr  # only on the VIVE sender machine
```

说明：`pinocchio` 的 Python import 名是 `pinocchio`。`pin-pink` 的 Python import 名是 `pink`，但当前不要在基础 viewer 环境中默认安装。
每装完一组可选包后，重新运行下面的基础数值栈检查；只有检查通过，才继续装下一组。

## 导入检查

先检查基础数值栈和 Pinocchio 绑定。这个检查必须通过，否则不要继续装其他可选包：

```bash
python - <<'PY'
import importlib.metadata as metadata
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

assert np.__version__ == "1.26.4", np.__version__
assert metadata.version("scipy") == "1.13.1", metadata.version("scipy")
try:
    metadata.version("pin")
except metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError("Do not install the pip package 'pin' in this NumPy 1.x SAPIEN environment.")

model = pin.Model()
joint_id = model.addJoint(0, pin.JointModelRY(), pin.SE3.Identity(), "joint")
model.appendBodyToJoint(joint_id, pin.Inertia.Random(), pin.SE3.Identity())
data = model.createData()
q = np.zeros(model.nq, dtype=np.float64)
pin.forwardKinematics(model, data, q)

print("numpy/scipy/pinocchio ABI ok")
print("numpy", np.__version__)
print("scipy", metadata.version("scipy"))
print("pinocchio", pin.__version__)
PY
```

如果需要 SAPIEN viewer，再检查：

```bash
python - <<'PY'
import sapien
import transforms3d
print("sapien ok")
print("transforms3d", getattr(transforms3d, "__version__", "unknown"))
PY
```

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
python -m teleop_runtime.deploy \
  --policy hold \
  --active-sides right \
  --camera-mode none \
  --max-steps 5 \
  --auto-start
```

只连接单臂 RM65 并读取状态，仍不发送动作：

```bash
python -m teleop_runtime.deploy \
  --policy hold \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --camera-mode none \
  --max-steps 30 \
  --auto-start
```

单臂 + InspireTac 手 dry-run：

```bash
python -m teleop_runtime.deploy \
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

## VIVE RM65 真机遥操默认流程

VIVE 真机入口：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --host 0.0.0.0 \
  --port 9999 \
  --no-collision
```

当前默认控制模式是 `--arm-control-mode impedance --impedance-mode pose`：

- VIVE retarget 生成统一 arm action：`T_arm_base_tool_tcp`。
- `tool_tcp` 是 tracker retarget、数据集记录、policy 输出和真机执行的主 frame。
- TCP、payload、重力补偿和 impedance 参数由 RM65 tool profile 提供，并按左右臂分别加载。
- 只控制机械臂且没有末端 profile 时，`tool_tcp` 默认为机械臂 `arm_endpoint`。
- profile TCP 距离 `arm_endpoint` 超过 `0.20m` 或旋转偏差超过 `90deg` 时会打印 warning。
- 默认 `--strict-tool-profile`，启动时会要求 active tool 能匹配
  `hardware/rm65/utils/configs/calibrations/rm65_{arm}_{tool}.json`，且 profile 含确认过的
  `tcp.pose_flange_m_rad`、payload 和 `impedance.pose`。
- runtime home 捕获完成后，入口会强制提示保持末端静止，并执行 `--force-tare-sec 3.0`
  的 force tare；tare 成功后才会 `arm_hardware` 并开始下发。
- 内置 `Force_Position_Move_Joint` 不在默认路径上；需要旧位置控制时显式传
  `--arm-control-mode position`。

如果 profile 不是 active tool 名，可以显式传：

```bash
--right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_empty.json
```
