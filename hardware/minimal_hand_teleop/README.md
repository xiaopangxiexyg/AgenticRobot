# Minimal Inspire Hand Teleop

这个目录提供一个最简化的桥接脚本，把：

- `teleop_hardware/hand_q6_reader.py` 读到的 6 维手部动作
- `inspire_tac/inspire_tac_modbus.py` 的 Modbus 控制接口

直接连起来，用于遥操作当前的 Inspire 灵巧手。

当前实现默认面向“单一右手 Inspire 手”，但输入端可以是：

- 单手 reader
- 双手 reader，再通过 `--source-side left/right` 选一只手来驱动

## 环境

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy
```

如果 `pymodbus`、`pyserial`、`numpy` 还没装，需要先装好对应依赖。

## 最小启动

假设你的手部输入设备接在 `/dev/ttyACM0`，当前控制的是默认右手 Inspire：

```bash
python minimal_hand_teleop/inspire_hand_teleop.py \
  --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00 \
  --side right \
  --hand-ip 192.168.12.210
```

如果你是左手输入设备驱动这只右手 Inspire：

```bash
python minimal_hand_teleop/inspire_hand_teleop.py \
  --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00 \
  --side left \
  --hand-ip 192.168.11.210
```

如果同时接了左右两个输入设备，但只想用右手输入来控制当前 Inspire：

```bash
python minimal_teleop/inspire_hand_teleop.py \
  --left-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00 \
  --right-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00 \
  --source-side right \
  --hand-ip 192.168.11.210
```

## 动作方向

脚本默认启用了 `--invert`。

原因是当前 `hand_q6_reader.py` 的 `norm` 一般可理解为“手指越弯曲数值越大”，而仓库里 `InspireTacModbus` 的常用约定是：

- `1.0` 更偏张开
- `0.0` 更偏闭合

所以默认会做一次 `target = 1 - source`。

如果你上电后发现动作方向正好反了，直接改成：

```bash
python minimal_teleop/inspire_hand_teleop.py \
  --port /dev/ttyACM0 \
  --side right \
  --hand-ip 192.168.11.210 \
  --no-invert
```

## 常用调参

### 1. 限制初始力度

默认 `--max-fingertip-force 300`，比原始驱动默认值更保守。

如果手指动作太弱，可以逐步提高：

```bash
--max-fingertip-force 500
```

### 2. 平滑输入

串口输入比较抖时，可以加一个简单 EMA：

```bash
--alpha 0.4
```

- `1.0` 表示不平滑
- 越小越平滑，但跟手性越弱

### 3. 调整关节顺序

当前 reader 输出的 `q6` 语义是：

```text
[thumb_yaw, thumb_flex, index, middle, ring, pinky]
```

默认 `--joint-order` 不是恒等映射，而是：

```text
[5, 4, 3, 2, 0, 1]
```

也就是默认会把 reader 的 `q6` 重排成 InspireTac 使用的顺序：

```text
[pinky, ring, middle, index, thumb_yaw, thumb_flex]
```

如果后续硬件接线或语义顺序不同，可以用：

```bash
--joint-order 0,1,2,3,4,5
```

例如要交换前两个关节：

```bash
--joint-order 1,0,2,3,4,5
```

### 4. 做线性微调

可以对每个关节做线性缩放和偏置：

```bash
--gain 1,1,1,0.9,0.9,0.9
--bias 0,0,0,0,0,0
```

也支持标量写法：

```bash
--gain 0.9
```

## 安全调试建议

先用 dry-run 看输入和映射后的目标值，不连 Inspire 手：

```bash
python minimal_teleop/inspire_hand_teleop.py \
  --port /dev/ttyACM0 \
  --side right \
  --dry-run
```

连接真机时，建议先：

1. 保持手附近无障碍物
2. 从较小 `--max-fingertip-force` 开始
3. 先观察张开/闭合方向是否正确
4. 必要时再切换 `--invert` / `--no-invert`

如果希望脚本启动或退出时把手张开，可以加：

```bash
--open-on-start
--open-on-exit
```

## 输出说明

脚本会周期性打印：

- `source`: 从 hand reader 读到的归一化 `q6`
- `target`: 映射后真正发给 Inspire 的 `q6`
- `sent`: 当前这一帧是否真的下发了 Modbus 命令

其中 `sent=False` 表示变化量没有超过 `--send-threshold`，因此被判重跳过了。
