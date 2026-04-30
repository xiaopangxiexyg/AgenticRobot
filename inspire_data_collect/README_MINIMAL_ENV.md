# FR3 / InspireTac 最小真机环境

这份说明面向把模型环境补齐为 `inspire_data_collect` 的真机部署环境。不要只拷贝本目录；FR3 路径还依赖仓库根目录下的 `exo_teleop`，手部标定/串口检查会用到 `hardware/teleop_hand`。

## 先确认的阻塞点

- `pymodbus` 必须是 `2.5.3`，当前代码使用 `pymodbus.client.sync.ModbusTcpClient`，`pymodbus 3.x` 不兼容。
- 启用 `--enable-arm` 时必须能导入 `exo_teleop.fr3_joint_control_client.FR3JointClient`，并且 FR3 joint control server 已在 `$FR3_HOST:$FR3_PORT` 运行。
- 采集外骨骼 FR3 随动时还必须能导入 `dynamixel_sdk`；纯策略部署到 FR3 不需要外骨骼串口。
- 如果策略需要 RealSense 图像，`pyrealsense2` 是硬依赖；不用相机时保持 `--camera-mode none`。

## 本地代码依赖

从仓库根目录运行，或把仓库根目录放进 `PYTHONPATH`：

```bash
export REPO=/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop_new
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
```

最小文件/目录：

- `inspire_data_collect/`
- `exo_teleop/fr3_joint_control_client.py`
- `exo_teleop/arm_exo/`，只在外骨骼采集/映射时需要
- `hardware/teleop_hand/exo_hand_zero_pose.json`，只在 q6 手套串口输入或重新标定时需要

`inspire_data_collect` 的 InspireTac backend 使用本目录内的轻量 Modbus 驱动，不依赖 `hardware/inspire_tac/inspire_tac_modbus.py`。

## Python 包

基础部署，手 + 可选 FR3：

```bash
python -m pip install "numpy<2" "pymodbus==2.5.3"
```

如果要读取 q6 串口手套或外骨骼：

```bash
python -m pip install pyserial dynamixel-sdk
```

如果要采集/回放 LeRobot parquet 或 debug replay：

```bash
python -m pip install pyarrow
```

如果要 RealSense 或 OpenCV 预览/视频：

```bash
python -m pip install pyrealsense2 opencv-python
```

当前已验证的参考环境是 Python 3.10，`grasp_py310_copy` 中这些关键包可导入。

## 导入检查

```bash
python - <<'PY'
import importlib.util as u

required = [
    "numpy",
    "pymodbus",
    "inspire_data_collect",
    "exo_teleop",
]
optional = [
    "serial",
    "dynamixel_sdk",
    "pyarrow",
    "cv2",
    "pyrealsense2",
]
for name in required + optional:
    print(f"{name}: {bool(u.find_spec(name))}")

from inspire_data_collect.hand.hand_backend import InspireTacHandBackend
from inspire_data_collect.arm.fr3_backend import FR3ArmBackend
print("inspire_data_collect hardware imports ok")
PY
```

## 硬件权限与网络

- 串口优先使用 `/dev/serial/by-id/...`；没有权限时把当前用户加入 `dialout` 后重新登录。
- InspireTac 默认 Modbus 端口是 `6000`，手 IP 由 `--left-hand-ip/--right-hand-ip` 指定。
- FR3 默认端口来自代码参数 `--fr3-port 5558`，部署机需要能连到 FR3 server。
- RealSense 需要系统已安装 librealsense udev 规则，且相机没有被其他进程占用。

## 最小 smoke test

先做纯 dry-run，不连 FR3、不连相机、不发送动作：

```bash
python -m inspire_data_collect.deploy.run_policy \
  --policy hold \
  --active-sides right \
  --camera-mode none \
  --max-steps 5 \
  --auto-start
```

只读连接 FR3 状态，仍不发送策略动作：

```bash
python -m inspire_data_collect.deploy.run_policy \
  --policy hold \
  --active-sides right \
  --enable-arm \
  --arm-side right \
  --fr3-host "$FR3_HOST" \
  --fr3-port "${FR3_PORT:-5558}" \
  --camera-mode none \
  --max-steps 30 \
  --auto-start
```

真实下发必须显式加 `--send-actions`；未加 `--auto-start` 时还需要按 `g` 才会执行策略。
