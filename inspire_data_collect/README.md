# inspire_data_collect

只含 Inspire Tac 手的数据采集便携包，不依赖 `teleop_runtime`。保留原工程的 `assets` 不复制；运行时只需要本目录、硬件可访问、以及 Python 依赖。

## 首先下载解压assets到项目路径: https://disk.pku.edu.cn/link/AA669C31FE27A74A01A2A3BADD61ADF559
文件名：assets.zip
有效期限：2026-06-08 14:10
提取码：YIlq

## 依赖

硬件与导出所需：

```bash
pip install pyserial pymodbus==2.5.3 pyarrow numpy
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

- `hand/serial_hand_q6_reader.py`: 读取遥操作手串口 `ANGLES:`，自动零位标定，输出归一化 6DoF。
- `hand/hand_source.py`: 并行读取左/右手串口，产出 `left_hand_q6`、`right_hand_q6` 通道。
- `hand/retargeter.py`: 将遥操作手 6DoF 映射为 Inspire 手 6DoF 指令，支持关节重排、反向和 EMA。
- `hand/inspire_tac_modbus.py`: 包内 Inspire Tac Modbus 驱动，负责关节控制、关节读取、触觉读取。
- `hand/hand_backend.py`: 封装 Inspire Tac 控制和后台遥测线程。
- `utils/realsense_rgb.py`: 单路 RealSense RGB 后台采集，维护 nearest-frame buffer。
- `video_frame.py`: 视频帧结构和按时间戳取最近帧的缓存。
- `utils/tactile_visualizer.py`: OpenCV 触觉 UV 可视化，显示 1062 维触觉。
- `utils/writer.py`: LeRobot 风格 parquet/meta/video 导出，手数据完整写入，并预留单臂 7DoF 扩展位。
- `run_collect.py`: 完整采集入口。

## 代码结构

采集主流程在 `run_collect.py`，核心路径是：

```text
HandQ6Source -> HandRetargeter -> InspireTacHandBackend -> HandSampleBuilder -> HandOnlyLeRobotWriter
RealSenseRGBSource ------------------------------------------^                    parquet/meta/video
```

关键类：

- `HandQ6Source`: 每只遥操作手一个后台线程，持续读串口帧，并缓存最新 `ObservationChannel`。
- `HandRetargeter`: 从 `left_hand_q6` / `right_hand_q6` 取源数据，按 `DEFAULT_INSPIRE_HAND_JOINT_ORDER=(5,4,3,2,0,1)` 重排，再按需取反，输出 `EffectorCommand`。
- `InspireTacHandBackend`: 负责 `set_joint_pos()` 下发，后台遥测线程读 `read_joint_pos()` 和 `read_tac_data()`。
- `RealSenseRGBSource`: 启动 RealSense color stream，后台线程将 RGB 帧推入 `NearestFrameBuffer`。
- `HandSampleBuilder`: 将当前 commands、hand states、faults 合成一帧 `HandSample`。
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

## 运行

### 开启触觉可视化采集

触觉可视化采集只需要在运行命令中加 `--visualize-tactile`。该开关显示 Inspire Tac 的 1062 维触觉 UV，并要求对应手配置 `--left-hand-ip` / `--right-hand-ip`，因为触觉数据来自 Inspire Tac 遥测。

双手触觉可视化采集：

```bash
python -m inspire_data_collect \
  --dataset-root ./hand_only_dataset \
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
  --dataset-root ./right_hand_dataset \
  --active-sides right \
  --right-hand-port /dev/ttyACM0 \
  --right-hand-ip 192.168.12.210 \
  --visualize-tactile
```

可视化窗口可用参数：

```bash
--tactile-vis-size 640
--tactile-vis-value-scale 0.05
--hand-tactile-regions all
```

按键：`z` 开始 episode，`c` 保存当前 episode，`d` 丢弃当前 episode，`q` 退出。加 `--record-auto-start` 可启动后直接采集。

无相机默认模式：

```bash
conda activate grasp_py310_copy

python -m inspire_data_collect \
  --dataset-root ./hand_only_dataset \
  --active-sides both \
  --left-hand-port /dev/ttyACM0 \
  --right-hand-port /dev/ttyACM1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.12.211 \
  --visualize-tactile
```

单 RealSense 相机，使用包内 `realsense.json`：

```bash
python -m inspire_data_collect \
  --dataset-root ./hand_camera_dataset \
  --active-sides both \
  --left-hand-port /dev/ttyACM0 \
  --right-hand-port /dev/ttyACM1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.12.211 \
  --camera-mode single \
  --camera-config inspire_data_collect/realsense.json
```

单 RealSense 相机，直接用序列号配置：

```bash
python -m inspire_data_collect \
  --dataset-root ./hand_camera_dataset \
  --active-sides right \
  --right-hand-port /dev/ttyACM0 \
  --right-hand-ip 192.168.12.210 \
  --camera-mode single \
  --camera-key front \
  --camera-serial 213222079798 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30
```

触觉读取：`--hand-tactile-regions all` 表示读取全部触觉区域；传空字符串 `--hand-tactile-regions ""` 表示不读取触觉。Python API 中同理，`tactile_regions=("all",)` 读全部，`tactile_regions=()` 不读触觉。

## 输出

目录结构：

```text
dataset_root/
  data/chunk-000/episode_000000.parquet
  meta/info.json
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/tasks.jsonl
```

启用相机后额外生成：

```text
dataset_root/
  meta/devices/realsense.json
  videos/observation.images.front/chunk-000/episode_000000.mp4
```

`action` 和 `observation.state` 均为 19 维：

- `[0:7]`: 单臂 7DoF 预留位，当前填 `NaN`。
- `[7:13]`: right hand q6。
- `[13:19]`: left hand q6。

也就是说 LeRobot 主向量保存顺序是：`arm(7) -> right_hand(6) -> left_hand(6)`。这个顺序也写入 `meta/info.json` 的 `vector_layout.hand_vector_order`。

同时单独写出：

- `action.arm.joint_target`: 7 维预留动作，当前 `NaN`。
- `observation.arm.qpos`: 7 维预留状态，当前 `NaN`。
- `action.right_hand.joint_target`、`action.left_hand.joint_target`。
- `observation.right_hand.qpos`、`observation.left_hand.qpos`。
- `observation.hand.<side>.tactile`: 1062 维触觉向量。
- `teleop.action_source.<side>_hand_q6`: 遥操作源手 6DoF。

`meta/info.json` 中的 `reserved_spaces` 明确记录单臂 7DoF 的 shape 和 slice，后续接入单臂时直接填充前 7 维即可。

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
