# inspire_data_collect

Inspire Tac 手数据采集便携包，已接入可选的外骨骼单臂 FR3 随动与记录，不依赖 `teleop_runtime`。保留原工程的 `assets` 不复制；运行时只需要本目录、硬件可访问、以及 Python 依赖。

realsense_gui:conda activate tactile_grasp
python -m inspire_data_collect.utils.realsense_gui --view-mode overlay


## 首先下载解压assets到项目路径: https://disk.pku.edu.cn/link/AA669C31FE27A74A01A2A3BADD61ADF559
文件名：assets.zip
有效期限：2026-06-08 14:10
提取码：YIlq

## 依赖

硬件与导出所需：

```bash
pip install pyserial pymodbus==2.5.3 pyarrow numpy dynamixel_sdk
```

触觉可视化和视频导出需要：

```bash
pip install opencv-python
```

RealSense 需要：

```bash
pip install pyrealsense2
```


## 模块

- `hand/serial_hand_q6_reader.py`: 读取遥操作手串口 `ANGLES:`，默认加载固定 zero pose 标定，输出归一化 6DoF；需要更新固定值时用 `--exo-hand-calibration`。
- `hand/hand_source.py`: 并行读取左/右手串口，产出 `left_hand_q6`、`right_hand_q6` 通道。
- `hand/retargeter.py`: 将遥操作手 6DoF 映射为 Inspire 手 6DoF 指令，支持关节重排、反向和 EMA。
- `hand/inspire_tac_modbus.py`: 包内 Inspire Tac Modbus 驱动，负责关节控制、关节读取、触觉读取。
- `hand/hand_backend.py`: 封装 Inspire Tac 控制和后台遥测线程。
- `utils/realsense_rgb.py`: 单路 RealSense RGB 后台采集，维护 nearest-frame buffer。
- `video_frame.py`: 视频帧结构和按时间戳取最近帧的缓存。
- `utils/tactile_visualizer.py`: OpenCV 触觉 UV 可视化，显示 1062 维触觉。
- `arm/exo_joint_source.py`: 读取 7DoF Dynamixel 外骨骼关节缓存。
- `arm/retargeter.py`: 复用 `exo_teleop` 的 `dynamixel_to_robot_arm` 映射，并做 FR3 安全关节裁剪。
- `arm/fr3_backend.py`: 封装 `FR3JointClient`，后台轮询 FR3 状态缓存，下发关节目标，并在启动阶段做 FR3 当前姿态到外骨骼目标的慢速插值对齐。
- `utils/writer.py`: LeRobot 风格 parquet/meta/video 导出，手数据完整写入，并支持单臂 7DoF 动作和状态位。
- `run_collect.py`: 完整采集入口，默认启用静默模式，只打印操作信息和状态变化。

## 代码结构

采集主流程在 `run_collect.py`，核心路径是：

```text
HandQ6Source -> HandRetargeter -> InspireTacHandBackend -> HandSampleBuilder -> HandOnlyLeRobotWriter
ExoJointSource -> FR3ArmRetargeter -> FR3ArmBackend --------^                    parquet/meta/video
RealSenseRGBSource -----------------------------------------^
```

关键类：

- `HandQ6Source`: 每只遥操作手一个后台线程，持续读串口帧，并缓存最新 `ObservationChannel`。
- `HandRetargeter`: 从 `left_hand_q6` / `right_hand_q6` 取源数据，按 `DEFAULT_INSPIRE_HAND_JOINT_ORDER=(5,4,3,2,0,1)` 重排，再按需取反，输出 `EffectorCommand`。
- `InspireTacHandBackend`: 负责 `set_joint_pos()` 下发，后台遥测线程读 `read_joint_pos()` 和 `read_tac_data()`。
- `ExoJointSource`: 从外骨骼 Dynamixel 读取 7DoF 关节角，输出 `arm_exo_joint_qpos`。
- `FR3ArmRetargeter`: 将外骨骼关节映射到 FR3 7DoF 目标，写入裁剪和映射元数据。
- `FR3ArmBackend`: 使用两条 TCP 连接的 `FR3JointClient` 下发 `send_q()`，通过后台 poller 获取 FR3 状态；第一次收到外骨骼目标时默认先从 FR3 当前 qpos 插值到外骨骼映射目标，避免启动瞬间跳变。
- `RealSenseRGBSource`: 启动 RealSense color stream，后台线程将 RGB 帧推入 `NearestFrameBuffer`。
- `HandSampleBuilder`: 将当前 commands、arm state、hand states、faults 合成一帧 `HandSample`。
- `HandOnlyLeRobotWriter`: 接收异步写盘任务，写 `data/*.parquet`、`meta/*.jsonl/json` 和可选 `videos/*.mp4`。

最小代码调用示例：

```python
from inspire_data_collect.hand.hand_source import HandQ6Source
from inspire_data_collect.hand.retargeter import HandRetargeter, HandRetargeterConfig
from inspire_data_collect.hand.hand_backend import InspireTacHandBackend, InspireTacHandBackendConfig

source = HandQ6Source.from_ports(right_port="/dev/ttyACM0", critical=True)
retargeter = HandRetargeter(
    HandRetargeterConfig(
        side="right",
        input_channel_name="right_hand_q6",
        effector_name="right_hand",
    )
)
backend = InspireTacHandBackend(
    InspireTacHandBackendConfig(
        side="right",
        hand_ip="192.168.12.210",
        enable_telemetry=True,
        # tactile_regions defaults to ("all",); pass () only when you need to disable tactile reads.
    )
)

source.start()
backend.start()
channels = source.get_latest_channels()
command = retargeter.retarget(channels, timestamp=0.0)
if command is not None:
    backend.send_command(command)
hand_state = backend.get_state()
```

## 运行前硬件测试流程

以下命令默认在包根目录执行，并假设当前 Python 环境已经安装本包所需依赖：

```bash
cd /home/user/zc/dex_inspire_teleop_new
```

先确认 Python 依赖能导入。RealSense 只在启用相机时需要：

```bash
python - <<'PY'
for name in ("serial", "pymodbus", "pyarrow", "numpy", "dynamixel_sdk"):
    __import__(name)
    print("ok", name)
try:
    __import__("pyrealsense2")
    print("ok pyrealsense2")
except ModuleNotFoundError:
    print("skip pyrealsense2: only required for --camera-mode single/multi")
PY
```

检查串口设备名是否稳定。正式命令里优先使用 `/dev/serial/by-id/...`，不要依赖每次可能变化的 `/dev/ttyACM*`：

```bash
ls -l /dev/serial/by-id/
```

### 1. 遥操作手串口 q6

右手示例；左手把变量和通道名改成 `LEFT_HAND_PORT` / `left_hand_q6`。

默认会读取仓库内标定注册表：

```text
hardware/teleop_hand/exo_hand_zero_pose.json
```

采集启动时会按当前 `side + port` 查找 `port_calibrations`。如果注册表还没有当前串口，但 `teleop_records/hand_zero_pose_calibrations/` 里已有匹配的历史记录，程序会自动导入最新记录；如果两处都没有，程序会要求保持 zero pose 并自动重新标定，然后把结果写回注册表和时间戳记录。若更换外骨骼、重新装配手指连杆，或需要强制更新当前串口记录，在采集命令中追加 `--exo-hand-calibration`。

也可以用下面的检查工具手动更新：

```bash
python hardware/teleop_hand/check_zero_pose_calibration.py \
  --left-port "$LEFT_HAND_PORT" \
  --right-port "$RIGHT_HAND_PORT" \
  --rounds 1 \
  --exo-hand-calibration
```

去掉 `--exo-hand-calibration` 时只保存每次检查结果，不更新固定标定 JSON。正式采集建议使用 `/dev/serial/by-id/...` 这类稳定 port 名称；程序也会记录当前 port 的 realpath 和 by-id/by-path alias，后续同一设备换成等价 alias 时仍可匹配。

```bash
export RIGHT_HAND_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130699-if00

python - <<'PY'
import os
import time
import numpy as np
from inspire_data_collect.hand.hand_source import HandQ6Source

source = HandQ6Source.from_ports(
    right_port=os.environ["RIGHT_HAND_PORT"],
    baudrate=115200,
    timeout=0.05,
    critical=True,
)
source.start()
try:
    for _ in range(80):
        channels = source.get_latest_channels()
        ch = channels.get("right_hand_q6")
        if ch is not None and ch.valid:
            age = time.time() - ch.timestamp
            print("right_hand_q6", np.round(ch.value, 3).tolist(), "age_sec", round(age, 3))
            break
        time.sleep(0.05)
    else:
        raise RuntimeError(f"no valid right_hand_q6 frame: {channels}")
finally:
    source.close()
PY
```

期望结果：能持续读到 6 维归一化 q6，移动遥操作手时数值变化；如果失败，先检查串口名、波特率和串口权限。

### 2. InspireTac 手本体 Modbus

这一步只读关节位置，并把当前关节位置原样发回一次，用于确认读写接口可用。左右手分别测试，避免 IP 或侧别接反。

```bash
export RIGHT_HAND_IP=192.168.11.210

python - <<'PY'
import os
import time
import numpy as np
from inspire_data_collect.common import EffectorCommand
from inspire_data_collect.hand.hand_backend import InspireTacHandBackend, InspireTacHandBackendConfig

backend = InspireTacHandBackend(
    InspireTacHandBackendConfig(
        side="right",
        hand_ip=os.environ["RIGHT_HAND_IP"],
        port=6000,
        connect_hardware=True,
        enable_telemetry=False,
    )
)
backend.start()
try:
    if not backend.is_ready():
        raise RuntimeError(f"hand not ready: {backend.get_state()}")
    q = backend.read_joint_pos_now()
    print("read_joint_pos", np.round(q, 3).tolist())
    backend.send_command(
        EffectorCommand(
            effector_name="right_hand",
            side="right",
            value=q,
            timestamp=time.time(),
        )
    )
    print("send_current_q ok")
finally:
    backend.close()
PY
```

期望结果：`read_joint_pos` 返回 6 维 `[0, 1]` 范围内的值，`send_current_q ok` 无异常。

### 3. InspireTac 触觉读取

默认会读取全部触觉区域；建议先用较低 `telemetry_hz` 确认链路正常。命令行也支持别名 `--tactile-regions` 和 `--tactile-region`。

```bash
export RIGHT_HAND_IP=192.168.11.210

python - <<'PY'
import os
import time
from inspire_data_collect.hand.hand_backend import InspireTacHandBackend, InspireTacHandBackendConfig

backend = InspireTacHandBackend(
    InspireTacHandBackendConfig(
        side="right",
        hand_ip=os.environ["RIGHT_HAND_IP"],
        port=6000,
        connect_hardware=True,
        enable_telemetry=True,
        telemetry_hz=5.0,
        tactile_regions=("all",),
    )
)
backend.start()
try:
    time.sleep(0.6)
    state = backend.get_state()
    if state["telemetry_error"]:
        raise RuntimeError(state["telemetry_error"])
    tactile = state["tactile"]
    print("joint_pos", None if state["joint_pos"] is None else state["joint_pos"].round(3).tolist())
    print("tactile_len", None if tactile is None else len(tactile))
    print("regions", state["tactile_regions"])
finally:
    backend.close()
PY
```

期望结果：`telemetry_error` 为空，`tactile_len` 非空；如果只需要手关节采集，可以在正式运行中传 `--hand-tactile-regions ""` 或 `--tactile-region ""` 关闭触觉读取。

### 4. RealSense 单相机

先列出系统可见的相机序列号：

```bash
python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
for dev in ctx.query_devices():
    print(dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
PY
```

再用采集包接口取一帧 RGB：

```bash
export CAMERA_SERIAL=135122070754

python - <<'PY'
import os
import time
from inspire_data_collect.utils.realsense_rgb import RealSenseRGBCameraConfig, RealSenseRGBSource

source = RealSenseRGBSource(
    RealSenseRGBCameraConfig(
        serial=os.environ["CAMERA_SERIAL"],
        camera_key="front",
        width=640,
        height=480,
        fps=30,
        keep_last=4,
        wait_timeout_ms=1000,
        stale_after_sec=2.0,
    )
)
source.start()
try:
    for _ in range(80):
        frame = source.get_latest_frame()
        if frame is not None:
            print("frame", frame.frame_index, frame.image.shape, frame.metadata)
            break
        time.sleep(0.05)
    else:
        raise RuntimeError(f"no RealSense frame: {source.device_info()}")
finally:
    source.close()
PY
```

期望结果：能打印 `frame`、图像 shape 和 metadata；如果失败，先确认序列号、USB 连接和是否有其他程序占用相机。

正式采集时可加 `--vis_cam` 实时显示 RealSense RGB 画面。预览窗口直接使用当前采集帧，因此分辨率和 `--camera-width/--camera-height` 或配置文件一致，并且窗口默认可拖拽调整大小；如果 OpenCV 后端限制窗口不能大于图像尺寸，默认 `--vis-cam-render-scale 2.0` 会把预览 backing image 放大到 2 倍，不够时可调到 `3.0` 或更高。如果希望窗口严格等于图像尺寸，传 `--vis-cam-window-mode autosize`。`--vis_cam` 支持 `--camera-mode single` 和 `--camera-mode multi`；多相机时用 `--vis-cam-key <camera_key>` 选择一个视角预览，不影响其他视角写盘。程序启动时会先做 OpenCV HighGUI 预检；如果显示环境不可用，会在终端打印 `preview is disabled` 并继续采集，视频写盘不受影响。

### 5. 外骨骼 Dynamixel 读取

这一步使用 `read_only=True` 读取外骨骼 7DoF 缓存，不向电机发送目标位置。测试时手动小幅移动外骨骼，确认读数变化。

```bash
export EXO_PORT=/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0

python - <<'PY'
import os
import time
import numpy as np
from inspire_data_collect.arm.exo_joint_source import ExoJointSource, ExoJointSourceConfig

source = ExoJointSource(
    ExoJointSourceConfig(
        side="right",
        port=os.environ["EXO_PORT"],
        baudrate=57600,
    )
)
source.start()
try:
    info = source.device_info()
    print("device_info", info)
    if not source.is_ready():
        raise RuntimeError(info)
    for _ in range(80):
        ch = source.get_latest_channels().get("arm_exo_joint_qpos")
        if ch is not None and ch.valid:
            print("arm_exo_joint_qpos", np.round(ch.value, 4).tolist(), "metadata", ch.metadata)
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("no valid arm_exo_joint_qpos")
finally:
    source.close()
PY
```

期望结果：能读到 7 维关节角，`cache_age_sec` 较小；如果失败，先检查外骨骼供电、串口名、Dynamixel ID 和波特率。

### 6. FR3 TCP 状态接口

这一步只 `ping/get` 和读取缓存状态，不发送关节目标，适合先确认 FR3 控制服务可达。

```bash
export FR3_HOST=192.168.20.172
export FR3_PORT=5558

python - <<'PY'
import os
from exo_teleop.fr3_joint_control_client import FR3JointClient

with FR3JointClient(
    os.environ["FR3_HOST"],
    int(os.environ["FR3_PORT"]),
    timeout_s=2.0,
) as client:
    print("ping", client.ping())
    state = client.get_state()
    print("sync_state", {"t": state.get("t"), "age_s": state.get("age_s"), "q": state.get("q"), "dq": state.get("dq")})
    client.start_state_poller(poll_hz=20.0, warn_stale_after_s=0.2)
    if not client.wait_until_state_ready(timeout_s=2.0):
        raise RuntimeError("FR3 state poller did not become ready")
    cached, age = client.get_state_cached(max_age_s=0.5)
    print("cached_state", {"client_cache_age": age, "server_age_s": cached.get("age_s"), "q": cached.get("q")})
PY
```

期望结果：`ping` 返回 ok，`q` 是 7 维，缓存状态年龄小于阈值；如果失败，先检查 FR3 server 是否启动、主机 IP、端口和网络连通。

### 7. 外骨骼到 FR3 映射接口

这一步只从外骨骼读取并生成 FR3 目标，不下发给 FR3。用它确认 `dynamixel_to_robot_arm` 映射、FR3 安全裁剪和元数据。

```bash
export EXO_PORT=/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0

python - <<'PY'
import os
import time
import numpy as np
from inspire_data_collect.arm.exo_joint_source import ExoJointSource, ExoJointSourceConfig
from inspire_data_collect.arm.retargeter import FR3ArmRetargeter, FR3ArmRetargeterConfig

source = ExoJointSource(
    ExoJointSourceConfig(side="right", port=os.environ["EXO_PORT"], baudrate=57600)
)
retargeter = FR3ArmRetargeter(
    FR3ArmRetargeterConfig(side="right", shrink_deg=10.0)
)
source.start()
try:
    for _ in range(80):
        channels = source.get_latest_channels()
        command = retargeter.retarget(channels, timestamp=time.time())
        if command is not None:
            print("fr3_target", np.round(command.value, 4).tolist())
            print("metadata", command.metadata)
            break
        time.sleep(0.05)
    else:
        raise RuntimeError(f"no arm command from channels: {channels}")
finally:
    source.close()
PY
```

期望结果：`fr3_target` 是 7 维，`clip_mask` 说明是否触发安全限位；若频繁裁剪，先回到安全姿态或调整外骨骼映射零位。

### 8. 最小闭环短测

前面所有单项测试都通过后，再做不录制数据的短时闭环。这个命令会按外骨骼目标下发 FR3，请确认 FR3 和外骨骼都处于安全姿态，并先用较低主循环频率短测。默认会先用 `--arm-startup-align-duration-s 5.0` 从 FR3 当前姿态慢速插值到外骨骼映射目标，插值结束后才进入直接随动。

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets_debug \
  --task-name smoke_fr3_exo \
  --active-sides right \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --enable-arm \
  --arm-side right \
  --exo-port "$EXO_PORT" \
  --fr3-host "$FR3_HOST" \
  --fr3-port "$FR3_PORT" \
  --arm-startup-align-duration-s 5.0 \
  --hz 10 \
  --max-steps 100 \
  --no-parquet \
  --no-video
```

期望结果：启动后无 `state_error` / `telemetry_error`，FR3 按外骨骼小幅随动；确认安全和方向正确后再进入正式采集。

## 运行

以下命令默认在包根目录执行：

```bash
cd /home/user/zc/dex_inspire_teleop_new
```

### 开启触觉可视化采集

触觉可视化采集只需要在运行命令中加 `--visualize-tactile`。该开关显示 Inspire Tac 的 1062 维触觉 UV，并要求对应手配置 `--left-hand-ip` / `--right-hand-ip`，因为触觉数据来自 Inspire Tac 遥测。

双手触觉可视化采集：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets \
  --task-name hand_bimanual \
  --active-sides both \
  --left-hand-port /dev/ttyACM0 \
  --right-hand-port /dev/ttyACM1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.12.211 \
  --visualize-tactile
```

单右手触觉可视化采集：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets \
  --task-name right_hand_grasp_test \
  --active-sides right \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --visualize-tactile
```

可视化窗口可用参数：

```bash
--tactile-vis-size 640
--tactile-vis-value-scale 0.02
--tactile-vis-render-scale 2.0
--tactile-vis-window-mode resizable
--tactile-region thumb_1,index_1,middle_1,ring_1,little_1,palm
```

默认 `--tactile-vis-window-mode resizable` 会创建可拖拽调整大小的 OpenCV 窗口，`--tactile-vis-render-scale 2.0` 会把触觉 backing canvas 放大到 2 倍，避免部分 OpenCV 后端限制窗口不能大于原始图像；如果希望窗口严格等于渲染图尺寸，传 `--tactile-vis-window-mode autosize`。`--tactile-vis-size` 控制每只手面板的基础渲染分辨率，调大后窗口初始尺寸和细节都会增加。

按键：`z` 开始 episode，`c` 保存当前 episode，`d` 丢弃当前 episode，`q` 退出。加 `--record-auto-start` 可启动后直接采集。

输出目录会按 `--dataset-root/YYYY-MM-DD_task-name` 自动创建，例如：

```text
./datasets/2026-04-24_pick_cube
```

其中 `--task-name` 是必填参数，会同时写入 `meta/info.json` 和 `meta/tasks.jsonl`，表示该数据集采集的任务。

无相机默认模式：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets \
  --task-name hand_bimanual \
  --active-sides both \
  --left-hand-port /dev/ttyACM0 \
  --right-hand-port /dev/ttyACM1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.12.211 \
  --visualize-tactile
```

单右手 + 外骨骼单臂 FR3 随动采集：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets \
  --task-name beer_bottle_open \
  --active-sides right \
  --right-hand-port "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130699-if00" \
  --right-hand-ip 192.168.11.210 \
  --enable-arm \
  --arm-side right \
  --exo-port /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0 \
  --exo-baudrate 57600 \
  --fr3-host 192.168.20.172 \
  --fr3-port 5558 \
  --fr3-state-poll-hz 30 \
  --arm-startup-align-duration-s 5.0 \
  --arm-shrink-deg 10 \
  --camera-mode multi \
  --camera-config inspire_data_collect/realsense.json \
  --vis_cam \
  --visualize-tactile \
  --vis-cam-window-mode resizable \
  --tactile-vis-window-mode resizable \
  --record-hand-only \
  --record-hand-fixed-pos "0.096,0.057,0.022,0.056,0.495,0.0"
```

如果只想采集 Inspire 手的 action/state/tactile，但不向 Inspire 手下发遥操作动作，在采集命令中追加 `--record-hand-only`。该模式仍会读取遥操作手串口、计算并记录 `action.<side>_hand.joint_target`，也会连接 Inspire 手读取关节状态和触觉；主循环里的手部 `set_joint_pos` 下发会关闭。作为采集间复位，程序会在脚本启动后、以及每次 episode 成功保存后，向 active hands 下发一次固定 q6，默认 `--record-hand-fixed-pos "0.096,0.057,0.022,0.056,0.495,0.0"`，可按需要覆盖。FR3 手臂、相机和其他设备流程保持不变。

### DMTAC gripper collect mode

`--collect-mode gripper` 使用 `hardware/DMTAC_gripper` 内的夹爪和 DM-Tac 触觉代码。遥操作输入仍然读取外骨骼手的 6DoF 串口，但只使用第二维，也就是 q6 index `1`，作为夹爪开合动作。为保持当前 LeRobot writer 和下游 action shape 不变，夹爪 action/state 仍写在对应 side 的 6 维 hand slot 中，只有 index `1` 有意义，其余维度为 0。

触觉不再读取 InspireTac Modbus 的 1062 维区域数据，而是读取 DM-Tac SDK `getDistributeForce()`，每个传感器原始帧为 `(288, 384, 3)` dense force。保存端会对最终触觉帧做 `kernel_size=2, stride=2` 的空间 mean pooling，写盘形状变为 `(144, 192, 3)`，再按传感器顺序 flatten 后写入 `observation.hand.<side>.tactile`。`observation.hand.<side>.tactile_dict_json` 只保存每个传感器已压缩 force array 的 shape/dtype，避免把 dense force 数据重复写进 JSON。

夹爪 action 有硬保护：`--gripper-max-action` 默认 `0.9`，命令行不允许设置超过 `0.9`，后端在下发 `move()` 前也会再次裁剪到 `[0, 0.9]`。不要用底层脚本直接执行 `move 1.0`；超过 `0.9` 可能损坏夹爪。

夹爪模式下 `--visualize-tactile` 会打开 DM-Tac `getDistributeForce()` 可视化窗口。热图显示 XY force magnitude，箭头显示每个区域的 `(Fx, Fy)` 方向，底部显示 mean/max/dXYmax。常用缩放参数仍是 `--tactile-vis-value-scale`，数值越小颜色和箭头越敏感。

### DMTAC gripper startup testing

以下步骤从单设备到完整采集逐步增加硬件，默认在项目根目录执行：

```bash
cd /home/user/zc/dex_inspire_teleop_new
export RIGHT_HAND_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130699-if00
export EXO_PORT=/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0
export FR3_HOST=192.168.20.172
export FR3_PORT=5558
export GRIPPER_PORT="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
export DMTAC_IDS="L25480055 L25480053"
```

1. 检查夹爪串口和 DM-Tac 设备枚举：

```bash
python hardware/DMTAC_gripper/run_gripper.py list-ports
python hardware/DMTAC_gripper/run_gripper.py list-tactile
```

2. 只读夹爪状态，不运动：

```bash
python hardware/DMTAC_gripper/run_gripper.py verify-gripper \
  --port "$GRIPPER_PORT" \
  --id 1
```

3. 低速低力矩做小行程动作测试。这里最多只移动到 `0.2`，不要测试 `1.0`：

```bash
python hardware/DMTAC_gripper/run_gripper.py set-speed 0.2 --port "$GRIPPER_PORT" --id 1
python hardware/DMTAC_gripper/run_gripper.py set-torque 0.2 --port "$GRIPPER_PORT" --id 1
python hardware/DMTAC_gripper/run_gripper.py open --port "$GRIPPER_PORT" --id 1 --wait
python hardware/DMTAC_gripper/run_gripper.py move 0.2 --port "$GRIPPER_PORT" --id 1 --speed 0.2 --torque 0.2 --wait
python hardware/DMTAC_gripper/run_gripper.py stop --port "$GRIPPER_PORT" --id 1
```

4. 只读 `getDistributeForce()`，确认两个触觉传感器都有帧：

```bash
python hardware/DMTAC_gripper/run_gripper.py verify-tactile \
  --tactile_ids $DMTAC_IDS \
  --backend cpu \
  --count 10
```

5. 单独打开 DM-Tac force 可视化。启动和 baseline 期间保持两个传感器无接触：

```bash
python hardware/DMTAC_gripper/run_gripper.py visualize-force \
  --tactile_ids $DMTAC_IDS \
  --backend cpu \
  --reset_tactile \
  --warmup_frames 30 \
  --viz_mode delta \
  --baseline_frames 20 \
  --fixed_ref 0.0005 \
  --deadband_abs 5e-5 \
  --temporal_alpha 0.35 \
  --spatial_blur_ksize 5 \
  --arrow_pool 16 \
  --stride 12 \
  --scale 2 \
  --max_hz 30
```

6. 检查外骨骼手 q6，并重点观察第二维 `q6[1]`。夹爪采集只使用这一维：

```bash
python - <<'PY'
import os
import time
import numpy as np
from inspire_data_collect.hand.hand_source import HandQ6Source

source = HandQ6Source.from_ports(
    right_port=os.environ["RIGHT_HAND_PORT"],
    baudrate=115200,
    timeout=0.05,
    critical=True,
)
source.start()
try:
    for _ in range(120):
        ch = source.get_latest_channels().get("right_hand_q6")
        if ch is not None and ch.valid:
            q6 = np.asarray(ch.value, dtype=float)
            print("right_hand_q6", np.round(q6, 3).tolist(), "gripper_dim_1", round(float(q6[1]), 3))
        time.sleep(0.05)
finally:
    source.close()
PY
```

7. 不连接夹爪、不连接触觉，先跑采集管线和 writer：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets_debug \
  --task-name gripper_smoke_no_hw \
  --collect-mode gripper \
  --active-sides right \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --gripper-no-hardware \
  --gripper-no-tactile \
  --gripper-max-action 0.9 \
  --hz 10 \
  --max-steps 100 \
  --no-parquet \
  --no-video
```

8. 只连接外骨骼手、夹爪和触觉，打开采集内置的 gripper tactile visualizer，不连接 FR3 或相机：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets_debug \
  --task-name gripper_tactile_smoke \
  --collect-mode gripper \
  --active-sides right \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --gripper-port "$GRIPPER_PORT" \
  --gripper-id 1 \
  --gripper-speed 0.2 \
  --gripper-torque 0.2 \
  --gripper-max-action 0.9 \
  --gripper-tactile-ids $DMTAC_IDS \
  --gripper-tactile-backend cpu \
  --visualize-tactile \
  --tactile-vis-value-scale 0.0005 \
  --hz 10 \
  --max-steps 200 \
  --no-parquet \
  --no-video
```

9. 上面全部通过后，再接入 FR3 和相机做完整短测：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets_debug \
  --task-name gripper_full_smoke \
  --collect-mode gripper \
  --active-sides right \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --gripper-port "$GRIPPER_PORT" \
  --gripper-id 1 \
  --gripper-speed 0.2 \
  --gripper-torque 0.2 \
  --gripper-max-action 0.9 \
  --gripper-tactile-ids $DMTAC_IDS \
  --gripper-tactile-backend cpu \
  --enable-arm \
  --arm-side right \
  --exo-port "$EXO_PORT" \
  --fr3-host "$FR3_HOST" \
  --fr3-port "$FR3_PORT" \
  --camera-mode single \
  --camera-config inspire_data_collect/realsense.json \
  --arm-startup-align-duration-s 5.0 \
  --vis_cam \
  --visualize-tactile \
  --tactile-vis-value-scale 0.0005 \
  --hz 10 \
  --max-steps 200
```

单右侧夹爪 + FR3 + 多相机示例：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets \
  --task-name gripper_collect_peeling \
  --collect-mode gripper \
  --active-sides right \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --gripper-port "$GRIPPER_PORT" \
  --gripper-id 1 \
  --gripper-speed 0.7 \
  --gripper-torque 1.0 \
  --gripper-max-action 0.9 \
  --gripper-tactile-ids L25480055 L25480053 \
  --gripper-tactile-backend cpu \
  --enable-arm \
  --arm-side right \
  --exo-port "$EXO_PORT" \
  --fr3-host "$FR3_HOST" \
  --fr3-port "$FR3_PORT" \
  --arm-startup-align-duration-s 5.0 \
  --camera-mode single \
  --camera-config inspire_data_collect/realsense.json \
  --vis_cam \
  --visualize-tactile \
  --tactile-vis-value-scale 0.0005 \
  --hz 10
```

如果只想先检查采集管线和数据写出，不连接夹爪或 DM-Tac 触觉，可加：

```bash
--gripper-no-hardware --gripper-no-tactile
```

### DexHand Q6 right-slot collect mode

`--collect-mode dexhand_q6` 是一个 DexHand 左手真机接入的单手本体补充。它只支持当前约定：

- 数据集逻辑槽位固定为 `right_hand`，即必须使用 `--active-sides right`。
- 物理硬件固定为左手 DexHand，即 `--dexhand-hardware-hand left`。
- 保存和对外暴露的 6DoF 顺序沿用现有外骨骼手 reader：`[thumb_yaw, thumb_flex, index, middle, ring, pinky]`。
- deploy/replay 也读取 `right_hand` 槽位，但实际发送给左手 DexHand。

后端内部会把上述 q6 转成 DexHand 12 关节命令。写入 observation 时不会直接保存最后一次命令，而是从 DexHand 实测关节反馈反算回 q6：`thumb_yaw` 来自 `th_rot`，`thumb_flex` 取 `th_dip/th_mcp` 平均；四指分别用 DIP 直接估计值和 MCP 按 profile 比例换算后的估计值取平均，因此最后两组耦合关节不一致时会得到一个折中 q6。

DexHand 触觉保存为 15 维 fingertip xyz force，排列维持后端现状：

```text
[th_fx, th_fy, th_fz,
 ff_fx, ff_fy, ff_fz,
 mf_fx, mf_fy, mf_fz,
 rf_fx, rf_fy, rf_fz,
 lf_fx, lf_fy, lf_fz]
```

finger order 为 `["th", "ff", "mf", "rf", "lf"]`，axis order 为 `["force_x", "force_y", "force_z"]`。采集和 deploy 的内置 tactile 可视化暂未接入 DexHand；调试触觉时使用 `hardware/portable_dexhand021` 自带工具。

#### DexHand Q6 startup testing

以下步骤从 q6 串口、DexHand 单体控制到完整采集逐步增加硬件：

```bash
cd /home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop_new
export RIGHT_HAND_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_XXXXXXXX-if00
```

1. 不连接 DexHand，只验证右手槽位外骨骼 q6 reader 和 writer：

```bash
python inspire_data_collect/run_collect.py \
  --dataset-root ./datasets_debug \
  --task-name dexhand_probe_serial \
  --collect-mode dexhand_q6 \
  --active-sides right \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --dexhand-no-hardware \
  --record-hand-only \
  --record-auto-start \
  --max-steps 200 \
  --no-silent
```

确认 `right_hand_q6` 有效，并且 `action.right_hand.joint_target` 已写入 q6。

2. 在采集框架外单独验证 DexHand 左手硬件控制。注意串口源仍是 `--side right`，真机是 `--hand left`：

```bash
python hardware/portable_dexhand021/6DOF_control/run_q6_to_dexhand.py \
  --port "$RIGHT_HAND_PORT" \
  --side right \
  --hand left \
  --hz 30 \
  --print-hz 5
```

确认开合方向正确、动作幅度安全。触觉可单独用 portable DexHand 工具查看。

3. 连接 DexHand，先用 record-only 短测 telemetry、q6 反算和 15 维触觉写盘：

```bash
python inspire_data_collect/run_collect.py \
  --dataset-root ./datasets_debug \
  --task-name dexhand_record_only \
  --collect-mode dexhand_q6 \
  --active-sides right \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --record-hand-only \
  --record-auto-start \
  --max-steps 300 \
  --no-silent
```

重点检查 `observation.right_hand.qpos`、`observation.hand.right.tactile` 长度 15，以及 `observation.hand.right.tactile_dict_json` 中的 fingertip xyz。

4. 完整采集短测：

```bash
python inspire_data_collect/run_collect.py \
  --dataset-root ./datasets_debug \
  --task-name dexhand_collect \
  --collect-mode dexhand_q6 \
  --active-sides right \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --record-auto-start \
  --no-silent
```

5. 对同一 right-slot 数据做 deploy/replay，实际仍执行到左手 DexHand：

```bash
python -m inspire_data_collect.deploy.run_policy \
  --hand-mode dexhand_q6 \
  --active-sides right \
  --send-actions \
  --auto-start \
  --debug-replay-dataset /path/to/dataset_root \
  --debug-replay-episode 0
```

deploy 特殊处理和采集保持一致：replay 读取 `action.right_hand.joint_target`，startup init 也使用 right-slot action vector，然后由 DexHand backend 发给左手真机。

常用 DexHand 参数：

- `--dexhand-hardware-hand left`: 物理 DexHand 侧，目前仅支持 left。
- `--dexhand-device-type auto|config|100U|200U`: USBCANFD 类型选择。
- `--dexhand-device-index N`: USBCANFD 设备序号。
- `--dexhand-tactile-period-ms 20`: 周期反馈/触觉请求周期。
- `--dexhand-no-hardware`: 只跑 reader/writer，不连接 DexHand。
- `--dexhand-mock`: 使用 portable DexHand mock 后端。

外骨骼到 FR3 是直接关节映射，程序刚启动时 FR3 当前关节和外骨骼映射目标通常不一致。默认启用启动对齐：FR3 后端第一次收到 arm 指令时，会以 FR3 当前 `qpos` 为起点，在 `--arm-startup-align-duration-s` 秒内线性插值到外骨骼当前目标；最大关节差低于 `--arm-startup-align-threshold-rad` 时会跳过插值。启动对齐期间操作者应尽量保持外骨骼不动，等插值结束后再进入正常随动。确认二者已经物理对齐且需要完全直接随动时，才使用 `--disable-arm-startup-align`。

单 RealSense 相机，使用包内 `realsense.json`：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets \
  --task-name hand_camera_test \
  --active-sides right\
  --right-hand-port "$RIGHT_HAND_PORT" \
  --right-hand-ip 192.168.11.210 \
  --camera-mode single \
  --camera-config inspire_data_collect/realsense.json \
  --vis_cam \
  --visualize-tactile
```

单 RealSense 相机，直接用序列号配置：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets \
  --task-name right_hand_camera \
  --active-sides right \
  --right-hand-port /dev/ttyACM0 \
  --right-hand-ip 192.168.12.210 \
  --camera-mode single \
  --camera-key front \
  --camera-serial 213222079798 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --vis_cam
```

多 RealSense 视角采集，使用一个 JSON 配置所有相机，并选择 `front` 视角预览：

```json
{
  "front": {
    "serial": "135122070754",
    "width": 640,
    "height": 480,
    "fps": 30,
    "wait_timeout_ms": 1000,
    "stale_after_sec": 2.0,
    "metadata": {"mount": "front"}
  },
  "wrist": {
    "serial": "213222079798",
    "width": 640,
    "height": 480,
    "fps": 30,
    "wait_timeout_ms": 1000,
    "stale_after_sec": 2.0,
    "metadata": {"mount": "wrist"}
  }
}
```

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets \
  --task-name right_hand_multi_camera \
  --active-sides right \
  --right-hand-port /dev/ttyACM0 \
  --right-hand-ip 192.168.12.210 \
  --camera-mode multi \
  --camera-config inspire_data_collect/realsense.json \
  --vis_cam \
  --vis-cam-key front
```

```bash
export FR3_HOST=192.168.20.172
export FR3_PORT=5558

python -m inspire_data_collect \
  --dataset-root ./datasets \
  --task-name chips_addtion \
  --active-sides right \
  --right-hand-port "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130699-if00" \
  --right-hand-ip 192.168.11.210 \
  --camera-mode single \
  --camera-config inspire_data_collect/realsense.json \
  --vis_cam \
  --visualize-tactile \
  --enable-arm \
  --tactile-vis-value-scale 0.002 \
  --tactile-regions all \
  --arm-side right \
  --exo-port "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0" \
  --fr3-host "$FR3_HOST" \
  --fr3-port "$FR3_PORT" \
  --arm-startup-align-duration-s 5.0 \
  --hz 20 \
  --max-force 600
```


`--camera-mode multi` 会记录配置文件中的全部相机，每个 `camera_key` 单独写入一个 `videos/observation.images.<camera_key>/...mp4`。`--vis-cam-key` 只控制预览窗口；不传时优先预览 `front`，否则预览配置中按名称排序的第一个相机。

多视角时偶发 `Frame didn't arrive` 通常是某一路 RealSense 在当前 timeout 内没有返回新帧，可能来自 USB 带宽、启动抖动或分辨率/帧率过高。采集端默认用 `wait_timeout_ms=1000`，并且只有连续超过 `stale_after_sec=2.0` 没有收到新帧时才报告相机 error；短暂超时不会反复刷屏，也不会影响已收到帧的预览和写盘。如果仍持续报 `frame_timeout_stale`，优先降低多路相机的 `width/height/fps`，并确认相机分散到足够的 USB 控制器上。

默认启用静默模式，不再持续打印 `[Status] ...`。采集过程中仍会保留开始/保存/丢弃/退出、标定、相机 preview 启停、fault 出现/恢复、相机从 waiting 到 streaming 等状态变化信息。需要恢复周期状态行时追加 `--no-silent` 或 `--verbose-status`，输出间隔由 `--status-sec` 控制。

如果当前终端没有 `DISPLAY` 或 `WAYLAND_DISPLAY`，可以删除 `--vis_cam`；即使保留该参数，程序也会禁用预览窗口并继续采集。需要持续观察 RealSense 帧计数时使用 `--no-silent`，运行中可看 `[Status] ... camera=front=frames:<N>`。

手部外骨骼 zero pose：默认读取固定标定；需要重新标定并更新固定值时，在任意采集命令中追加 `--exo-hand-calibration`。也可以用 `--exo-hand-calibration-path PATH` 指定另一份固定标定 JSON。

触觉读取：默认就是 `--hand-tactile-regions all`，也可以显式传 `--tactile-regions all` 或 `--tactile-region all` 表示读取全部触觉区域；传空字符串 `--tactile-region ""` 表示不读取触觉。也可以传逗号分隔区域，例如 `--tactile-region thumb_1,index_1,middle_1,ring_1,little_1,palm`。有效区域名为 `little_1..little_3`、`ring_1..ring_3`、`middle_1..middle_3`、`index_1..index_3`、`thumb_1..thumb_4`、`palm`；非法区域会在启动时直接报错。Python API 中同理，`tactile_regions` 默认读全部，`tactile_regions=()` 不读触觉。

InspireTac max force：`--inspire-hand-max-force N` 会在手 backend 连接时把 `N` 写到每个手指的 force，默认 `800`；兼容旧参数 `--max-fingertip-force`，也可用简写 `--max-force`。

## 输出

目录结构：

```text
dataset_root/YYYY-MM-DD_task-name/
  data/chunk-000/episode_000000.parquet
  meta/info.json
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/tasks.jsonl
```

启用相机后额外生成：

```text
dataset_root/YYYY-MM-DD_task-name/
  meta/devices/realsense.json
  videos/observation.images.front/chunk-000/episode_000000.mp4
  videos/observation.images.wrist/chunk-000/episode_000000.mp4
```

单相机时只会生成对应一个 `observation.images.<camera_key>` 目录；多相机时每个配置的 `camera_key` 都会生成独立视频目录。

`action` 和 `observation.state` 均为 19 维：

- `[0:7]`: 单臂 7DoF；未传 `--enable-arm` 时填 `NaN`，启用后写入 FR3 目标/状态。
- `[7:13]`: right hand q6。
- `[13:19]`: left hand q6。

也就是说 LeRobot 主向量保存顺序是：`arm(7) -> right_hand(6) -> left_hand(6)`。这个顺序也写入 `meta/info.json` 的 `vector_layout.hand_vector_order`。

同时单独写出：

- `action.arm.joint_target`: 7 维 FR3 关节目标；未启用 arm 时为 `NaN`。
- `observation.arm.qpos`: 7 维 FR3 当前关节状态；未启用 arm 或状态暂不可用时为 `NaN`。
- `observation.arm.dq`: 7 维 FR3 关节速度，若服务端未返回则为 null。
- `observation.arm.startup_alignment_*`: FR3 启动对齐插值状态、进度、起点和目标。
- `action.right_hand.joint_target`、`action.left_hand.joint_target`。
- `observation.right_hand.qpos`、`observation.left_hand.qpos`。
- `observation.hand.<side>.tactile`: InspireTac 手模式下是 1062 维触觉向量；gripper 模式下是 DM-Tac `getDistributeForce()` 原始 `(288, 384, 3)` force 经保存端 `kernel_size=2, stride=2` mean pooling 后的 `(144, 192, 3)` flatten 向量，默认两个传感器时为 `2 * 144 * 192 * 3 = 165888` 维；DexHand Q6 模式下只写 `observation.hand.right.tactile`，为 15 维 `[th_xyz, ff_xyz, mf_xyz, rf_xyz, lf_xyz]` fingertip force。
- `observation.hand.<side>.tactile_timestamp`: 本帧触觉读取请求时间。
- `observation.hand.<side>.telemetry_timestamp`: 手部 telemetry 最新更新时间。
- `observation.hand.<side>.tactile_regions`: 本次采集启用的触觉区域名；DexHand Q6 模式下为 `("th", "ff", "mf", "rf", "lf")`。
- `observation.hand.<side>.tactile_dict_json`: InspireTac 手模式下是选中触觉区域的矩阵数据 JSON；gripper 模式下只记录每个 DM-Tac 传感器的 shape/dtype；DexHand Q6 模式下记录 finger order、axis order、每个指尖的 `force_x/force_y/force_z`，以及底层 raw tactile 摘要。
- `teleop.action_source.arm_exo_qpos`: 外骨骼原始 7DoF 关节角。
- `teleop.action_source.arm_preclip_joint_target`: 外骨骼映射到 FR3 后、裁剪前的 7DoF 目标。
- `teleop.action_source.arm_clip_mask`: 每个 FR3 关节是否被安全限位裁剪。
- `teleop.action_source.<side>_hand_q6`: 遥操作源手 6DoF。

`meta/info.json` 中的 `reserved_spaces` 明确记录单臂 7DoF 的 shape 和 slice；当前 writer 会在启用 arm 时直接填充前 7 维。

DexHand Q6 模式还会在 episode metadata 中写入 `logical_hand_side="right"`、`hardware_hand_side="left"`、`dexhand_q6_order`、`dexhand_joint_feedback_saved_as_q6=true`、`dexhand_tactile_format="fingertip_xyz"`、`dexhand_tactile_fingers`、`dexhand_tactile_axes`、`dexhand_tactile_dim=15`。设备配置另存为 `meta/devices/dexhand_q6.json`，用于记录 right-slot 到 left hardware 的映射。

启用相机后，每帧还会写出：

- `camera_keys`: 当前帧包含的 camera key。
- `observation.images.<camera_key>.source_timestamp`: 被选中的相机帧时间戳。
- `observation.images.<camera_key>.receive_timestamp`: 收到该帧的本机时间。
- `observation.images.<camera_key>.age_sec`: 样本时间与相机帧时间差。
- `observation.images.<camera_key>.source_frame_index`: RealSense 源帧序号。
- `observation.images.<camera_key>.video_frame_index`: 写入 mp4 后的视频帧序号。
- `observation.images.<camera_key>.path`: 相对 dataset root 的视频路径。

### 拆分 LeRobot episode 数据集

`utils/extract_lerobot_episodes.py` 用于把已有 LeRobot 风格数据集中的指定 episode 拆到一个新数据集，同时把原数据集改写为只包含剩余 episode。脚本会重写 `meta/*.jsonl/json`、parquet 里的 `index/episode_index/frame_index`，并复制对应视频文件；不要手动复制 `data/` 和 `videos/` 目录来替代该脚本。

依赖：

```bash
pip install pyarrow
```

先 dry-run 查看计划，不会改动数据：

```bash
python /home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop_new/inspire_data_collect/utils/extract_lerobot_episodes.py \
  --dataset-root /home/zc/HandMotion/real_robot_dataset/datasets/2026-05-06_soft_hard \
  --output-root /home/zc/HandMotion/real_robot_dataset/datasets/2026-05-06_soft_hard_wasted \
  --episodes 44,46,99
```

确认 `extracted` 和 `remaining source` 数量正确后，加 `--yes` 正式执行：

```bash
python /home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop_new/inspire_data_collect/utils/extract_lerobot_episodes.py \
  --dataset-root /home/zc/HandMotion/real_robot_dataset/datasets/2026-05-06_soft_hard \
  --output-root /home/zc/HandMotion/real_robot_dataset/datasets/2026-05-06_soft_hard_wasted \
  --episodes 44,46,99 \
  --yes
```

`--episodes` 支持逗号和闭区间：

```bash
--episodes 0
--episodes 0,2,5
--episodes 0-3,8
```

常用参数：

- `--dataset-root`: 被拆分的原数据集目录，必须包含 `meta/episodes.jsonl`。
- `--output-root`: 拆出的 episode 写入的新数据集目录，不能等于或嵌套在 `--dataset-root` 中。
- `--overwrite-output`: 如果 `--output-root` 已存在，允许先删除再写入。
- `--output-dataset-name`: 写入拆出数据集 `meta/info.json` 的 `dataset_name`；默认使用 `--output-root` 目录名。
- `--remaining-dataset-name`: 写回剩余原数据集的 `dataset_name`；默认保留原 `info.json`。
- `--allow-missing-videos`: `meta/episodes.jsonl` 中记录的视频不存在时跳过视频复制，否则默认报错。
- `--no-backup`: 成功拆分后删除原始数据集备份；不建议首次使用时开启。
- `--parquet-compression`: 重写 parquet 时使用的压缩格式，默认 `zstd`。

执行成功后会得到：

```text
--output-root                         # 指定 episodes 组成的新数据集
--dataset-root                        # 原路径被改写为剩余 episodes 数据集
--dataset-root 的同级隐藏 backup        # 默认保留拆分前的完整原始数据集
```

backup 目录名类似：

```text
.2026-05-06_soft_hard.before_extract-<timestamp>-<pid>
```

如果需要恢复到拆分前的原样数据，优先使用这个 backup；它比重新合并两个已重编号的数据集更可靠。

同一个脚本也支持把两个数据类型一致的 LeRobot 数据集合并到一个新目录。合并不会修改两个输入数据集，会按 `--dataset-root` 的 episode 在前、`--merge-root` 的 episode 在后的顺序重新编号，并重写 parquet/meta/video 路径。

先 dry-run：

```bash
python /home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop_new/inspire_data_collect/utils/extract_lerobot_episodes.py \
  --dataset-root /home/zc/HandMotion/real_robot_dataset/datasets/2026-05-06_soft_hard \
  --merge-root /home/zc/HandMotion/real_robot_dataset/datasets/2026-05-06_soft_hard_random \
  --output-root /home/zc/HandMotion/real_robot_dataset/datasets/2026-05-06_soft_hard_merged
```

确认计划后正式合并：

```bash
python /home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop_new/inspire_data_collect/utils/extract_lerobot_episodes.py \
  --dataset-root /home/zc/HandMotion/real_robot_dataset/datasets/2026-05-06_soft_hard \
  --merge-root /home/zc/HandMotion/real_robot_dataset/datasets/2026-05-06_soft_hard_random \
  --output-root /home/zc/HandMotion/real_robot_dataset/datasets/2026-05-06_soft_hard_merged \
  --yes
```

合并模式会校验两个输入的 `meta/info.json` 关键结构、视频 camera key 和首个 parquet schema。若输出目录已存在，需要显式加 `--overwrite-output`。如果视频文件缺失但仍想保留 parquet/meta 合并，可加 `--allow-missing-videos`。

导出的最小示例：

```python
import time
from inspire_data_collect.async_writer import AsyncWriterPool
from inspire_data_collect.utils.writer import HandOnlyLeRobotWriter, HandSampleBuilder, WriterConfig

writer = HandOnlyLeRobotWriter(WriterConfig(root_dir="./hand_only_dataset"))
pool = AsyncWriterPool(writer.handle_job)
pool.start()

episode_index = writer.start_episode()
builder = HandSampleBuilder(active_sides=("right", "left"), episode_index=episode_index)

sample = builder.build(
    timestamp=time.time(),
    session_state="running",
    control_hz=30.0,
    channels={},
    commands={},
    hand_states={},
)
for job in writer.plan_sample_jobs(sample):
    pool.submit(job)
for job in writer.plan_finalize_episode(episode_index):
    pool.submit(job)
pool.close(wait=True)
```
