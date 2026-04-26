# inspire_data_collect

Inspire Tac 手数据采集便携包，已接入可选的外骨骼单臂 FR3 随动与记录，不依赖 `teleop_runtime`。保留原工程的 `assets` 不复制；运行时只需要本目录、硬件可访问、以及 Python 依赖。

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
- `run_collect.py`: 完整采集入口。

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
        tactile_regions=("all",),  # "all" reads all tactile regions; () disables tactile reads
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
cd /home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop_new
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
    print("skip pyrealsense2: only required for --camera-mode single")
PY
```

检查串口设备名是否稳定。正式命令里优先使用 `/dev/serial/by-id/...`，不要依赖每次可能变化的 `/dev/ttyACM*`：

```bash
ls -l /dev/serial/by-id/
```

### 1. 遥操作手串口 q6

右手示例；左手把变量和通道名改成 `LEFT_HAND_PORT` / `left_hand_q6`。

默认会读取仓库内固定标定：

```text
hardware/teleop_hand/exo_hand_zero_pose.json
```

这份标定由 18 次 zero pose 记录汇总得到，当前最大 span 为 left `0.0315 deg`、right `0.2700 deg`。因此普通采集不需要每次重新保持 zero pose。若更换外骨骼、重新装配手指连杆，或需要更新固定值，再显式运行 zero pose 标定：

```bash
python hardware/teleop_hand/check_zero_pose_calibration.py \
  --left-port "$LEFT_HAND_PORT" \
  --right-port "$RIGHT_HAND_PORT" \
  --rounds 1 \
  --exo-hand-calibration
```

去掉 `--exo-hand-calibration` 时只保存每次检查结果，不更新固定标定 JSON。

```bash
export RIGHT_HAND_PORT=/dev/ttyACM0

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
export RIGHT_HAND_IP=192.168.12.210

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

先用少量区域低频读取，确认链路正常后再在正式采集中用 `--hand-tactile-regions all`。

```bash
export RIGHT_HAND_IP=192.168.12.210

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
        tactile_regions=("thumb_1", "index_1", "palm"),
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

期望结果：`telemetry_error` 为空，`tactile_len` 非空；如果只需要手关节采集，可以在正式运行中传 `--hand-tactile-regions ""` 关闭触觉读取。

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
export CAMERA_SERIAL=213222079798

python - <<'PY'
import os
import time
from inspire_data_collect.utils.realsense_rgb import RealSenseRGBCameraConfig, RealSenseRGBSource

source = RealSenseRGBSource(
    RealSenseRGBCameraConfig(
        serial=os.environ["CAMERA_SERIAL"],
        camera_key="front",
        width=1280,
        height=720,
        fps=30,
        keep_last=4,
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

正式采集时可加 `--vis_cam` 实时显示 RealSense RGB 画面。预览窗口直接使用当前采集帧，因此分辨率和 `--camera-width/--camera-height` 或配置文件一致。`--vis_cam` 只支持 `--camera-mode single`，无显示环境时不要传该参数。

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
cd /home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop_new
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
  --task-name right_hand_grasp \
  --active-sides right \
  --right-hand-port /dev/ttyACM0 \
  --right-hand-ip 192.168.12.210 \
  --visualize-tactile
```

可视化窗口可用参数：

```bash
--tactile-vis-size 640
--tactile-vis-value-scale 0.05
--tactile-vis-window-mode resizable
--hand-tactile-regions all
```

默认 `--tactile-vis-window-mode resizable` 会创建可拖拽调整大小的 OpenCV 窗口；如果希望窗口严格等于渲染图尺寸，传 `--tactile-vis-window-mode autosize`。`--tactile-vis-size` 控制每只手面板的基础渲染分辨率，调大后窗口初始尺寸和细节都会增加。

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
  --task-name right_hand_fr3_exo \
  --active-sides right \
  --right-hand-port /dev/ttyACM0 \
  --right-hand-ip 192.168.12.210 \
  --enable-arm \
  --arm-side right \
  --exo-port /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0 \
  --exo-baudrate 57600 \
  --fr3-host 192.168.20.172 \
  --fr3-port 5558 \
  --fr3-state-poll-hz 20 \
  --arm-startup-align-duration-s 5.0 \
  --arm-shrink-deg 10
```

外骨骼到 FR3 是直接关节映射，程序刚启动时 FR3 当前关节和外骨骼映射目标通常不一致。默认启用启动对齐：FR3 后端第一次收到 arm 指令时，会以 FR3 当前 `qpos` 为起点，在 `--arm-startup-align-duration-s` 秒内线性插值到外骨骼当前目标；最大关节差低于 `--arm-startup-align-threshold-rad` 时会跳过插值。启动对齐期间操作者应尽量保持外骨骼不动，等插值结束后再进入正常随动。确认二者已经物理对齐且需要完全直接随动时，才使用 `--disable-arm-startup-align`。

单 RealSense 相机，使用包内 `realsense.json`：

```bash
python -m inspire_data_collect \
  --dataset-root ./datasets \
  --task-name hand_camera \
  --active-sides both \
  --left-hand-port /dev/ttyACM0 \
  --right-hand-port /dev/ttyACM1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.12.211 \
  --camera-mode single \
  --camera-config inspire_data_collect/realsense.json \
  --vis_cam
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

如果当前终端没有 `DISPLAY` 或 `WAYLAND_DISPLAY`，删除 `--vis_cam`，不影响相机采集和视频写盘。

手部外骨骼 zero pose：默认读取固定标定；需要重新标定并更新固定值时，在任意采集命令中追加 `--exo-hand-calibration`。也可以用 `--exo-hand-calibration-path PATH` 指定另一份固定标定 JSON。

触觉读取：`--hand-tactile-regions all` 表示读取全部触觉区域；传空字符串 `--hand-tactile-regions ""` 表示不读取触觉。Python API 中同理，`tactile_regions=("all",)` 读全部，`tactile_regions=()` 不读触觉。

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
```

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
- `observation.hand.<side>.tactile`: 1062 维触觉向量。
- `teleop.action_source.arm_exo_qpos`: 外骨骼原始 7DoF 关节角。
- `teleop.action_source.arm_preclip_joint_target`: 外骨骼映射到 FR3 后、裁剪前的 7DoF 目标。
- `teleop.action_source.arm_clip_mask`: 每个 FR3 关节是否被安全限位裁剪。
- `teleop.action_source.<side>_hand_q6`: 遥操作源手 6DoF。

`meta/info.json` 中的 `reserved_spaces` 明确记录单臂 7DoF 的 shape 和 slice；当前 writer 会在启用 arm 时直接填充前 7 维。

启用相机后，每帧还会写出：

- `camera_keys`: 当前帧包含的 camera key。
- `observation.images.<camera_key>.source_timestamp`: 被选中的相机帧时间戳。
- `observation.images.<camera_key>.receive_timestamp`: 收到该帧的本机时间。
- `observation.images.<camera_key>.age_sec`: 样本时间与相机帧时间差。
- `observation.images.<camera_key>.source_frame_index`: RealSense 源帧序号。
- `observation.images.<camera_key>.video_frame_index`: 写入 mp4 后的视频帧序号。
- `observation.images.<camera_key>.path`: 相对 dataset root 的视频路径。

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
