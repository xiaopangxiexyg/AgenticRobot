# Minimal Teleop Readers

当前代码框架里，手臂遥操作主链路是：

1. `teleop.py` 或 `teleop_arm_only_to_redis.py` 从外骨骼读 14 个电机角。
2. 通过 `dynamixel_to_robot_arm(...)` 映射成 `q14`。
3. `q14` 直接送给 `G1LowController`，或者写入 Redis 的 `teleop:arm`。

也就是说：

- 手臂链路默认暴露的是 `q14`，不是末端 6DoF 位姿
- 手部链路默认暴露的是 `q6`，不是末端 `xyz+rpy`

这个目录里现在有两条最小读取器：

- [teleop_6dof_reader.py](/home/zc/HandMotion/RealRobotExperiments/g1_o6_teleop/minimal_control/teleop_6dof_reader.py)
  作用：外骨骼手臂 `q14` -> 左右手末端 6DoF pose
- [hand_q6_reader.py](/home/zc/HandMotion/RealRobotExperiments/g1_o6_teleop/minimal_control/hand_q6_reader.py)
  作用：手部串口设备 -> 6 个手指关节值

如果你现在只想拿到遥操作的六维信息，最小做法就是：

1. 复用现有外骨骼读数和 `q14` 标定映射。
2. 不启动 `G1LowController`，不接 Unitree DDS，不下发机器人。
3. 用本地 `assets/g1_o6/robot.urdf` 做前向运动学，把 `q14` 转成左右末端 pose。

如果你接的是“只有手部 6 关节”的设备，不要用 `teleop_6dof_reader.py`，应该用 `hand_q6_reader.py`。

## 手部 q6 读取

适用场景：

- 设备只提供单手 6 个关节
- 串口看起来像 `/dev/ttyACM0`
- 设备不是 Dynamixel 总线

仓库里手部输入链路对应的是 [drivers/glove_reader.py](/home/zc/HandMotion/RealRobotExperiments/g1_o6_teleop/drivers/glove_reader.py)，默认波特率就是 `115200`，而不是臂部那套 `57600` 的 Dynamixel 总线。

最小读取命令：

```bash
python minimal_control/hand_q6_reader.py \
  --port /dev/ttyACM0 \
  --baudrate 115200 \
  --side right
```

如果你只想先看串口里到底发了什么原始文本：

```bash
python minimal_control/hand_q6_reader.py \
  --port /dev/ttyACM0 \
  --baudrate 115200 \
  --raw-only \
  --show-raw-lines
```

这个脚本会解析形如 `ANGLES: ...` 的串口文本，并输出：

- `angles_deg`: 原始 6 维角度
- `norm`: 按当前零位自动标定后的 6 维 `0..1`，可直接给 `DexHand.set_joint_pos(...)`
- `norm_swap01`: 把前两个维度交换后的版本，主要给仿真链路
- `sim_q6_rad`: 仿真用的弧度 `q6`

为什么会同时给 `norm` 和 `norm_swap01`：

- 仓库里的手套链路前两个维度是否需要交换，代码里本来就保留了可配置逻辑，见 [teleop_sim.py](/home/zc/HandMotion/RealRobotExperiments/g1_o6_teleop/teleop_sim.py:84)
- 当前约定的 `q6` 语义是 `[thumb_yaw, thumb_flex, index, middle, ring, pinky]`，见 [teleop_sim.py](/home/zc/HandMotion/RealRobotExperiments/g1_o6_teleop/teleop_sim.py:63)

如果你想把单手数据发布到 Redis：

```bash
python minimal_control/hand_q6_reader.py \
  --port /dev/ttyACM0 \
  --baudrate 115200 \
  --side right \
  --redis-key teleop:hand_single
```

## 直接用于灵巧手控制

单手最关键的接口是：

- `packet["norm"]`
- 或者 `reader.get_norm_values()`

它们都是 6 维 `0..1`，顺序是：

```python
["thumb_yaw", "thumb_flex", "index", "middle", "ring", "pinky"]
```

这就是最适合直接喂给灵巧手控制器的量。

最小 Python 用法：

```python
from minimal_control.hand_q6_reader import SingleHandQ6Reader
from drivers.hand_driver import DexHand

reader = SingleHandQ6Reader(
    port="/dev/ttyACM0",
    side="right",
    baudrate=115200,
)

hand = DexHand(hand_type="right", can_iface="can0")

try:
    while True:
        norm6 = reader.get_norm_values()   # shape=(6,), 数值范围 0..1
        hand.set_joint_pos(norm6.tolist())
finally:
    reader.close()
    hand.close_can()
```

如果你已经在 CLI 下调试通了，也可以先看打印出来的这一行：

```text
[right] norm(用于DexHand.set_joint_pos): [...]
```

## 便捷支持左右手

`hand_q6_reader.py` 现在支持双手模式：

```bash
python minimal_control/hand_q6_reader.py \
  --left-port /dev/ttyACM0 \
  --right-port /dev/ttyACM1 \
  --baudrate 115200
```

双手模式下会额外输出：

- `norm_action12`: `[left6, right6]`

也就是和仓库里 `teleop_hand_only_to_redis.py` 的拼接顺序一致，先左手后右手。

Python 里对应的便捷接口：

```python
from minimal_control.hand_q6_reader import BimanualHandQ6Reader

reader = BimanualHandQ6Reader(
    left_port="/dev/ttyACM0",
    right_port="/dev/ttyACM1",
    baudrate=115200,
)

try:
    norm_dict = reader.get_norm_values()
    left6 = norm_dict["left"]
    right6 = norm_dict["right"]

    action12 = reader.get_norm_action12()  # [left6, right6]
finally:
    reader.close()
```

如果你只接了单手设备，就继续用：

- `SingleHandQ6Reader`
- `--port ... --side left/right`

## 输出定义

- 左右各输出一个 6 维向量：`[x, y, z, roll, pitch, yaw]`
- 单位：
  - 位置 `x, y, z` 是米
  - 姿态 `roll, pitch, yaw` 是弧度
- 坐标系：
  - 默认相对于 `torso_link`
- 末端 link：
  - 默认取 `left_hand_base_link`
  - 默认取 `right_hand_base_link`

如果你只想看 wrist，不想包含手掌安装的固定外参，可以把目标 link 改成：

- `--left-link left_wrist_yaw_link`
- `--right-link right_wrist_yaw_link`

## 最常用的几种用法

先做一次纯 FK 自检，不碰硬件：

```bash
python minimal_control/teleop_6dof_reader.py --dry-run --print-q14
```

先查看当前机器能看到哪些串口：

```bash
python minimal_control/teleop_6dof_reader.py --list-ports
```

直接读取外骨骼并在终端打印左右 6DoF：

```bash
python minimal_control/teleop_6dof_reader.py --hz 20
```

如果你希望和现有 Redis 流程对齐，把结果发布出去：

```bash
python minimal_control/teleop_6dof_reader.py \
  --hz 20 \
  --redis-key teleop:arm6d
```

如果当前外骨骼开机后需要和 `teleop.py` 一样先做复位：

```bash
python minimal_control/teleop_6dof_reader.py \
  --hz 20 \
  --reset-exo
```

如果端口能打开，但你怀疑它不是正确的 Dynamixel 总线，可以先做低层探测：

```bash
python minimal_control/dxl_bus_probe.py \
  --port /dev/ttyUSB0 \
  --baudrates 57600 115200 1000000 \
  --id-start 0 \
  --id-end 13
```

## 新设备接入后如何设置端口

这一步最关键的是：

- 优先使用稳定别名 `/dev/serial/by-id/...`
- 其次使用 `/dev/serial/by-path/...`
- 最后才用 `/dev/ttyUSB0` / `/dev/ttyACM0`

原因很简单：

- `ttyUSB0` 这类名字会随着插拔顺序变化
- `by-id` 一般和 USB 转串口芯片的序列号绑定，更稳定
- `by-path` 和物理 USB 口位置绑定，适合固定插在某个口位

### 场景 1: 只接入 1 台新设备

推荐流程：

1. 先拔掉目标设备。
2. 执行一次：

```bash
python minimal_control/teleop_6dof_reader.py --list-ports
```

3. 插上设备，再执行一次同样命令。
4. 新出现的那一项，就是这个设备当前对应的串口。
5. 运行脚本时显式传入：

```bash
python minimal_control/teleop_6dof_reader.py \
  --port /dev/ttyUSB0 \
  --hz 20
```

如果你看到了 `/dev/serial/by-id/...`，更推荐直接用它：

```bash
python minimal_control/teleop_6dof_reader.py \
  --port /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_FTA7NOIE-if00-port0 \
  --hz 20
```

### 场景 2: 同时接入 2 台新设备

推荐流程：

1. 先只插第一台，记录它的 `device`、`serial_number`、`location`。
2. 再只插第二台，也记录同样信息。
3. 最后两台都插上，再执行：

```bash
python minimal_control/teleop_6dof_reader.py --list-ports
```

4. 用 `serial_number` 或 `location` 区分哪一台是哪一台。

判断规则一般是：

- 如果两台设备有不同的 `serial_number`，优先按 `serial_number` 认设备
- 如果没有稳定序列号，但设备永远插在固定 USB 口，按 `location` 或 `by-path` 认设备
- 不要依赖 `ttyUSB0`/`ttyUSB1` 的先后顺序

### 场景 3: 机器上没有 `/dev/serial/by-id`

如果你看到的报错像这次这样：

```text
No such file or directory: /dev/serial/by-id/...
```

通常表示两种情况之一：

- 当前机器没有这个历史端口对应的设备
- 当前系统没有生成 `by-id` 别名，只能先用 `ttyUSB*` 或 `ttyACM*`

这时先执行：

```bash
python minimal_control/teleop_6dof_reader.py --list-ports
```

如果能看到设备，例如 `/dev/ttyUSB0`，就先这样运行：

```bash
python minimal_control/teleop_6dof_reader.py \
  --port /dev/ttyUSB0 \
  --hz 20
```

### 场景 5: 端口能打开，但报 `Incorrect status packet`

这类报错和“端口不存在”不同，它通常意味着：

- 这个口后面不是 Dynamixel 2.0 设备
- 或者口是对的，但波特率/电平/转换器类型不对
- 或者总线上有别的设备在输出非 Dynamixel 数据

结合这个仓库的现有配置，优先级最高的怀疑项是：

1. 你选到的是 `ttyACM0` 这类 CDC/串口设备，而不是外骨骼那根 FTDI 转串口线
2. 设备确实是外骨骼，但新适配器的名字变了，需要换成 `ttyUSB*` 或新的 `by-id`
3. 设备接法不对，导致收到的字节流不是标准 Dynamixel 包

建议排查顺序：

1. 先跑 `python minimal_control/dxl_bus_probe.py --port <你的端口> --baudrates 57600 115200 1000000 --id-start 0 --id-end 13`
2. 如果 3 个波特率都主要报 `Incorrect status packet`，基本可以判定这个端口不是目标总线
3. 如果变成 `There is no status packet`，再继续怀疑波特率、ID、接线
4. 如果某个波特率下能扫出 `id=0..13` 中的一部分或全部，就把这个波特率和端口用到 `teleop_6dof_reader.py`

### 场景 4: 想长期固定名称

如果你有两台相同的 USB 串口设备，建议最终做 udev 规则，给它们固定别名。

先查设备属性：

```bash
python -m serial.tools.list_ports -v
```

或者：

```bash
udevadm info -a -n /dev/ttyUSB0 | rg 'idVendor|idProduct|serial'
```

然后写规则，例如：

```bash
sudo tee /etc/udev/rules.d/99-teleop-serial.rules >/dev/null <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", ATTRS{serial}=="FTA7NOIE", SYMLINK+="teleop_exo"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", ATTRS{serial}=="FTXXXXXX", SYMLINK+="teleop_exo_backup"
EOF
```

重载规则：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

之后就可以固定使用：

```bash
python minimal_control/teleop_6dof_reader.py \
  --port /dev/teleop_exo \
  --hz 20
```

## 这个仓库里哪些地方需要端口

如果你只跑这个最小脚本：

- 只需要设置 1 个端口：外骨骼 Dynamixel 总线 `--port`

如果你跑完整 teleop 链路：

- `teleop.py` 里有 3 个串口/路径要确认
- `EXO_PORT`: 外骨骼电机总线
- `RIGHT_GLOVE_PORT`: 右手手套串口
- `LEFT_GLOVE_PORT`: 左手手套串口

其中：

- 手套通常更适合用 `/dev/serial/by-path/...`，因为左右可能接在固定 USB 口
- 外骨骼总线更适合用 `/dev/serial/by-id/...`，因为同一根 USB 转串口线最好绑到序列号

## Redis 包格式

脚本在 `--redis-key` 打开时会用 `pickle.dumps(...)` 发布一个字典，字段如下：

- `timestamp`: shape `(1,)`
- `q14`: shape `(14,)`
- `left_xyz_rpy`: shape `(6,)`
- `right_xyz_rpy`: shape `(6,)`
- `pose6d`: shape `(12,)`，即 `left_xyz_rpy + right_xyz_rpy`
- `base_link`
- `left_link`
- `right_link`

读取示例：

```python
import pickle
import redis

r = redis.Redis(host="localhost", port=6379)
data = pickle.loads(r.get("teleop:arm6d"))

left_pose = data["left_xyz_rpy"]     # [x, y, z, roll, pitch, yaw]
right_pose = data["right_xyz_rpy"]   # [x, y, z, roll, pitch, yaw]
q14 = data["q14"]
```

## 什么时候该用这个最小脚本

适合：

- 只想获取遥操作输入，不想驱动 G1
- 想把当前外骨骼输入转成末端 6DoF 给别的模块用
- 想先确认当前 teleop 输入范围、方向、姿态定义

不适合：

- 需要机器人真实状态 `state`
- 需要机器人 torso/world 坐标系下的末端位姿
- 需要手指 6 维输入，那个应该继续看 `teleop_hand_only_to_redis.py`

## 代码上和现有框架的关系

- 外骨骼读数和标定映射沿用 `teleop.py` / `teleop_arm_only_to_redis.py`
- 没有复用那两个脚本的 import，是因为它们会顺带拉起更重的依赖
- 这里单独做成最小脚本，避免只想读 6DoF 时还接入机器人控制链路

如果后面你希望把这个 6DoF 直接并进现有 `teleop:arm` 数据流，最直接的改法是在 [teleop_arm_only_to_redis.py](/home/zc/HandMotion/RealRobotExperiments/g1_o6_teleop/teleop_arm_only_to_redis.py) 里，`q_exo = read_exo_q14(exo)` 后面插入同样的 FK 计算，再一起 publish。
