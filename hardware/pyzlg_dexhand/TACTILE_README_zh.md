# DexHand 手指触觉读取接口与数据类型

本文档基于当前仓库代码梳理 `pyzlg_dexhand` 的触觉反馈来源、Python 读取接口、ROS 数组格式和底层 CANFD 字段布局。硬件和环境已经可用时，建议先用 Python 接口直接读取，再按需要接入 ROS。

## 触觉信息是什么

触觉信息来自每个指尖控制板的运动反馈帧扩展字段。SDK 解码后会把 5 个指尖传感器聚合到 `HandFeedback.touch`，键名固定为：

| 键名 | 手指 | 来源板号 |
| --- | --- | --- |
| `th` | 拇指 thumb | board 0 |
| `ff` | 食指 index/first finger | board 2 |
| `mf` | 中指 middle finger | board 3 |
| `rf` | 无名指 ring finger | board 4 |
| `lf` | 小指 little finger | board 5 |

`board 1` 控制 `th_rot` 和 `ff_spr`，代码中没有映射为指尖触觉，因此不会出现在 `touch` 字典里。

每个指尖触觉数据类型是 `StampedTouchFeedback`：

| 字段 | 类型 | 单位/含义 |
| --- | --- | --- |
| `timestamp` | `float/int` | SDK 处理该反馈时的主机时间戳；当前实现为 `time.time_ns()` 纳秒值 |
| `normal_force` | `float` | 法向力，单位 N |
| `normal_force_delta` | `int` | 法向力变化量，原始单位 |
| `tangential_force` | `float` | 切向力，单位 N |
| `tangential_force_delta` | `int` | 切向力变化量，原始单位 |
| `direction` | `int` | 切向力方向，0-359 度，指尖方向为 0；`65535` 或大于 359 可按无效值处理 |
| `proximity` | `int` | 接近度，原始单位 |
| `temperature` | `int` | 指尖温度，单位摄氏度 |

如果 `feedback.touch` 为空，通常表示当前没有收到带触觉扩展字段的反馈帧，或者尚未开启周期反馈。

## Python 读取接口

推荐在 `DexHand021` 环境中运行：

```bash
conda activate DexHand021
cd /home/zc/HandMotion/DexHand021/pyzlg_dexhand
pip install -e .
```

### 方式 1：直接读取触觉

新增的 `get_tactile_feedback()` 只返回触觉字典：

```python
from pyzlg_dexhand.dexhand_interface import LeftDexHand, FeedbackMode

hand = LeftDexHand()

# 如果只是读取而不持续发运动命令，先开启周期反馈。
# period_ms 的硬件编码粒度是 10 ms，常用 20/50/100 ms。
hand.set_feedback_mode(FeedbackMode.PERIODIC, period_ms=20, enable=True)

touch = hand.get_tactile_feedback()

for finger, data in touch.items():
    print(
        finger,
        data.normal_force,
        data.tangential_force,
        data.direction,
        data.proximity,
        data.temperature,
    )

hand.close()
```

### 方式 2：读取完整反馈

原有接口 `get_feedback()` 会同时返回关节和触觉：

```python
from pyzlg_dexhand import LeftDexHand

hand = LeftDexHand()
feedback = hand.get_feedback()

thumb_touch = feedback.touch.get("th")
if thumb_touch is not None:
    print("thumb normal force:", thumb_touch.normal_force)
```

`HandFeedback` 的结构：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `query_timestamp` | `float` | 调用 `get_feedback()` 时的主机时间，单位秒 |
| `joints` | `Dict[str, JointFeedback]` | 12 个关节反馈 |
| `touch` | `Dict[str, StampedTouchFeedback]` | 5 个指尖触觉反馈，可能为空或缺少部分手指 |

### 方式 3：命令行采样脚本

仓库新增了只读采样脚本：

```bash
conda activate DexHand021
cd /home/zc/HandMotion/DexHand021/pyzlg_dexhand

# 左手，50 Hz 轮询，持续打印
python examples/read_tactile_feedback.py --hands left --rate 50

# 只看 5 指接近觉，20 Hz 持续打印
python examples/read_proximity_feedback.py --hands left --rate 20

# 同时精细控制关节，并可视化 5 个指尖触觉点的详细力觉信息
python examples/finger_joint_control_gui.py --hands left --tactile-rate 20

# 只读可视化 5 个指尖触觉点的详细力觉信息
python examples/tactile_force_visualizer.py --hands left --rate 20

# 不连接硬件，使用模拟数据预览可视化窗口
python examples/tactile_force_visualizer.py --demo

# 如果自动打开失败，可显式指定适配器型号
python examples/read_proximity_feedback.py --hands left --device-type 200U

# 右手，采样 10 秒，并保存 JSONL
python examples/read_tactile_feedback.py --hands right --rate 100 --duration 10 --jsonl tactile_right.jsonl

# 同时读取左右手
python examples/read_tactile_feedback.py --hands left right --rate 50

# 如果别的控制循环已经开启反馈，只轮询现有反馈，不发送反馈配置命令
python examples/read_tactile_feedback.py --hands left --no-configure-feedback
```

脚本默认会调用：

```python
hand.set_feedback_mode(FeedbackMode.PERIODIC, period_ms=20, enable=True)
```

该命令只配置反馈模式，不发送关节运动命令。若正在运行其他控制程序，建议使用 `--no-configure-feedback`，避免多个程序同时配置 CAN 反馈。

USBCANFD 设备通常不能被两个 Python 进程同时打开。需要一边控制关节一边看力觉时，请运行合并后的 `examples/finger_joint_control_gui.py`；`examples/tactile_force_visualizer.py` 保留给只读观察使用，不要和控制 GUI 同时运行。

JSONL 每行示例：

```json
{
  "timestamp": 1710000000.123,
  "hand": "left",
  "touch": {
    "th": {
      "timestamp": 1710000000.120,
      "normal_force": 0.12,
      "normal_force_delta": 3,
      "tangential_force": 0.02,
      "tangential_force_delta": 1,
      "direction": 18,
      "proximity": 450,
      "temperature": 27
    }
  }
}
```

## ROS 触觉话题格式

ROS 节点在 `config/config.yaml` 中配置触觉话题：

| 手 | 默认话题 | 类型 |
| --- | --- | --- |
| 左手 | `/left_hand/touch_sensors` | `std_msgs/Float64MultiArray` |
| 右手 | `/right_hand/touch_sensors` | `std_msgs/Float64MultiArray` |

数组长度为 `5 * 8 = 40`。手指顺序固定：

| finger_index | 手指键名 |
| --- | --- |
| 0 | `th` |
| 1 | `ff` |
| 2 | `mf` |
| 3 | `rf` |
| 4 | `lf` |

每个手指 8 个值，偏移计算：

```python
base = finger_index * 8
timestamp = data[base + 0]
normal_force = data[base + 1]
normal_force_delta = data[base + 2]
tangential_force = data[base + 3]
tangential_force_delta = data[base + 4]
direction = data[base + 5]
proximity = data[base + 6]
temperature = data[base + 7]
```

ROS 发布层会把 `direction == 65535` 或 `direction > 359` 转成 `-1`，表示方向无效。

## 底层 CANFD 触觉字段

运动反馈帧的 CAN ID 类型是 `MessageType.MOTION_FEEDBACK = 0x180`，实际 ID 为：

```text
feedback_can_id = 0x180 + hand_base_id + board_idx
left hand base_id  = 0x01
right hand base_id = 0x07
```

触觉字段从反馈帧第 16 字节开始，完整触觉数据需要帧长度至少 42 字节：

| 字节范围 | 解码 | SDK 字段 |
| --- | --- | --- |
| `16:20` | little-endian `float32` | `normal_force` |
| `20:24` | little-endian unsigned int | `normal_force_delta` |
| `24:28` | little-endian `float32` | `tangential_force` |
| `28:32` | little-endian unsigned int | `tangential_force_delta` |
| `32:34` | little-endian unsigned short | `direction` |
| `34:38` | little-endian unsigned int | `proximity` |
| `38:42` | little-endian unsigned int | `temperature` |

相关代码位置：

- 协议解码：`pyzlg_dexhand/dexhand_protocol/messages.py::_decode_feedback`
- 板号到手指映射：`pyzlg_dexhand/dexhand_interface.py::DexHandBase.finger_map`
- Python 聚合接口：`pyzlg_dexhand/dexhand_interface.py::get_feedback` 和 `get_tactile_feedback`
- ROS 数组发布：`examples/ros_node/dexhand_ros.py::process_and_publish_feedback`

## 常见注意点

- `get_tactile_feedback()` 读取的是最近接收到的反馈状态；要持续更新，需要硬件处于周期反馈模式，或控制命令中请求反馈。
- 如果只读触觉，优先用 `set_feedback_mode(FeedbackMode.PERIODIC, period_ms=20, enable=True)` 后循环调用 `get_tactile_feedback()`。
- `period_ms` 的底层编码以 10 ms 为单位，建议使用 10 的倍数。
- 多个程序不要同时打开同一个 ZCAN 设备。需要控制和力觉可视化同步使用时，运行 `finger_joint_control_gui.py` 这个合并入口；若 ROS 节点已经占用设备，应通过 ROS 话题读取触觉，不要再启动 Python 直连脚本。
- `normal_force_delta`、`tangential_force_delta`、`proximity` 是硬件原始量，适合做阈值、趋势或标定后的特征；绝对物理量解释应以传感器标定为准。
