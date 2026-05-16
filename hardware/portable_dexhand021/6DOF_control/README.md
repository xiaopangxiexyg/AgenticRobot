# DexHand 6DOF Control Wrapper

这个目录是额外的 6 维控制包装，不修改 `teleop_hand`、`minimal_hand_teleop` 或现有 `portable_dexhand021` 核心代码。

## 0. 先测试 hand_q6 设备连接

在接 DexHand 真机前，先只连接 hand_q6 串口设备，确认能读到实际 6 维控制信号。启动命令前保持手完全张开，脚本会默认采集当前姿态作为 `q6=0` 的零位基准。这个命令不会打开 USBCANFD，也不会驱动 DexHand：

```bash
python portable_dexhand021/6DOF_control/run_q6_to_dexhand.py \
  --port "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00" \
  --side right \
  --dry-run \
  --frames 20 \
  --print-hz 10
```

启动后会先看到类似：

```text
[Zero:right] keep hand_q6 fully open; waiting for a stable zero pose...
[Zero:right] finished with ... samples, used ...
[Zero:right] deg_at_0=[...]
```

之后输出里的 `q6=[...]` 就是实际送入本包装映射后的 6 维控制信号。完全张开时它应接近：

```text
q6=[0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
```

顺序为：

```text
[thumb_flex, thumb_rot, index, middle, ring, little]
```

同时会打印 `action={...}`，表示这组 6 维控制量将被映射成哪些 DexHand 12 关节角度。若想确认串口原始文本，加：

```bash
--show-raw-lines
```

如果端口不确定，先查看当前系统串口：

```bash
python -m serial.tools.list_ports -v
```

如果要跳过启动零位校准，改用 `teleop_hand/exo_hand_zero_pose.json` 里的固定标定，加：

```bash
--no-zero-on-start
```

启动零位校准默认参照 `teleop_hand/hand_q6_reader.py` 的稳定窗口逻辑：最多采 `--zero-max-samples 200` 帧，至少 `--zero-min-samples 30` 帧，并用最近 `--zero-stable-window 20` 帧判断是否稳定。由于启动时只有张开零位、没有闭合端点，默认 `--zero-scale absolute` 会把相对零位的角度变化量视为收缩量，避免左右手符号或端口侧别不匹配时数值一直被裁剪到 0。如果确认 `--side` 和设备方向完全一致，可以改用：

```bash
--zero-scale signed
```

## 6 维约定

本包装内部使用：

```text
[thumb_flex, thumb_rot, index, middle, ring, little]
```

映射到 DexHand 12 个关节：

```text
q6[0] thumb_flex -> th_mcp = th_dip
q6[1] thumb_rot  -> th_rot
q6[2] index      -> ff_mcp = ff_dip
q6[3] middle     -> mf_mcp = mf_dip
q6[4] ring       -> rf_mcp = rf_dip
q6[5] little     -> lf_mcp = lf_dip
ff_spr 固定为 0，不控制伸展/侧展维度
```

默认开闭方向：

- `0.0` 表示张开，对应最小角度
- `1.0` 表示闭合，对应最大角度

这和 `teleop_hand/hand_q6_reader.py` 的 `norm` 语义一致：手指越弯曲数值越大。若输入源使用 Inspire 风格的 `1=张开, 0=闭合`，运行时加 `--invert`。

## 和现有 6 维 reader 的顺序差异

`teleop_hand/hand_q6_reader.py` 当前输出顺序是：

```text
[thumb_yaw, thumb_flex, index, middle, ring, pinky]
```

本包装要求大拇指收缩是第 1 维、大拇指旋转是第 2 维，所以桥接串口 reader 时默认使用 `--source-order auto`，会按 `teleop` 顺序自动交换前两维。

如果你直接传入本包装约定的 6 维动作，用 `--q6` 即可。

## 纯映射测试

只打印映射，不连接 DexHand：

```bash
python portable_dexhand021/6DOF_control/run_q6_to_dexhand.py \
  --q6 0.5,0.2,0.8,0.8,0.8,0.8 \
  --dry-run
```

## 发送单帧到 DexHand

```bash
python portable_dexhand021/6DOF_control/run_q6_to_dexhand.py \
  --q6 0.2,0.1,0.5,0.5,0.5,0.5 \
  --hand left \
  --device-type auto
```

## 四指比例耦合模式

默认四指使用比例耦合模式，即等价于：

```bash
--four-finger-mode profile
```

它会按 [current_joint_angles_lh2tsvng.json](current_joint_angles_lh2tsvng.json) 里的四指姿态比例控制，不改变大拇指。

如果需要退回简单同值映射，可以显式加：

```bash
--four-finger-mode same
```

同值映射为：

```text
ff_dip = ff_mcp = index_angle
mf_dip = mf_mcp = middle_angle
rf_dip = rf_mcp = ring_angle
lf_dip = lf_mcp = little_angle
```

默认 profile 模式下：

```text
*_dip = q6_finger * 90
*_mcp = *_dip * (json[*_mcp] / json[*_dip])
```

所有角度都会限制在 `0..90` 度，所以比例较大的 `*_mcp` 可能提前到 90 度，后续继续收缩时保持 90 度。完全张开时仍然是 `*_dip = *_mcp = 0`。

使用示例：

```bash
python portable_dexhand021/6DOF_control/run_q6_to_dexhand.py \
  --q6 0.2,0.1,0.5,0.5,0.5,0.5 \
  --dry-run
```

如需指定其他姿态 JSON：

```bash
--four-finger-profile /path/to/joint_angles.json
```

## 使用手部串口 reader 控制 DexHand

```bash
python portable_dexhand021/6DOF_control/run_q6_to_dexhand.py \
  --port "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00" \
  --side right \
  --hand left \
  --hz 50 \
  --print-hz 5
```

串口输入兼容形如 `ANGLES: ...` 的 6 维角度行。默认会在启动时用当前完全张开姿态做零位；如果加 `--no-zero-on-start`，才会读取固定标定文件：

```text
teleop_hand/exo_hand_zero_pose.json
```

## 同时控制、可视化和读状态

控制、状态读取和触觉可视化都复用同一个 `DexHandController` 连接，不会额外打开第二个 USBCANFD 进程。

控制时打开触觉窗口：

```bash
python portable_dexhand021/6DOF_control/run_q6_to_dexhand.py \
  --port "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00" \
  --side right \
  --hand left \
  --visualize-tactile
```

控制时定期打印关节状态和指尖法向力：

```bash
--state-print-hz 5
```

触觉窗口刷新率可调：

```bash
--tactile-viewer-hz 20
```

## 运行中自动清错

控制 DexHand 真机时脚本默认会：

- 连接后先执行一次 `clear_errors()`
- 运行中以 `--error-check-hz 20` 的频率刷新反馈并检测 board error
- 一旦发现错误，立即执行 `clear_errors()`，并强制下一帧动作重新下发

如果需要降低 CAN 反馈轮询压力，可以把检查频率调低：

```bash
--error-check-hz 5
```

如需关闭运行期检测：

```bash
--error-check-hz 0
```

如需跳过启动清错：

```bash
--no-clear-errors-on-start
```

## Python 用法

```python
from pathlib import Path
import sys

root = Path("/home/zc/HandMotion/DexHand021")
sys.path.insert(0, str(root / "portable_dexhand021" / "6DOF_control"))

from q6_to_dexhand import DexHandQ6Mapper

mapper = DexHandQ6Mapper()
action = mapper.q6_to_dexhand_action([0.2, 0.1, 0.5, 0.5, 0.5, 0.5])
print(action)
```

如果要像控制类包装一样读取 6 维动作并复现发送，可以给 `DexHandQ6Controller` 传入带 `read_q6()` 方法的 reader：

```python
from q6_to_dexhand import DexHandQ6Controller

q6_controller = DexHandQ6Controller(hand="left", q6_reader=reader)
with q6_controller:
    q6 = q6_controller.read_q6(source_order="teleop")
    action = q6_controller.reproduce_q6()
    state = q6_controller.read_state()
```
