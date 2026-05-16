# portable_dexhand021

`portable_dexhand021` 是一个不修改 `pyzlg_dexhand` 源码的轻量末端控制器封装。它假设 `pyzlg_dexhand` 已经安装好，只在运行时导入 SDK 的公开接口。

## 功能

- 单手统一控制类：`DexHandController(hand="left" | "right")`
- 初始化 / 关闭 ZCAN 和 DexHand
- 发送关节动作：支持关节名 dict、12 维动作序列、关键字参数
- 读取统一状态：关节反馈 + 指尖力觉反馈
- 单独读取力觉：5 个指尖 `th/ff/mf/rf/lf`
- 同一进程内可视化力觉，避免两个 Python 进程同时占用 USBCANFD

## 基本用法

在 `/home/zc/HandMotion/DexHand021` 下运行：

```python
from portable_dexhand021 import DexHandController

with DexHandController(hand="left", device_type="auto") as hand:
    hand.send_action({
        "th_rot": 30,
        "ff_mcp": 45,
        "ff_dip": {"position": 45, "current": 200, "velocity": 1000},
    })

    state = hand.read_state()
    print(state["joints"]["ff_mcp"]["angle"])
    print(state["tactile"].get("ff"))
```

12 维序列顺序固定为：

```python
from portable_dexhand021 import JOINT_NAMES

print(JOINT_NAMES)
# ('th_dip', 'th_mcp', 'th_rot', 'ff_spr', 'ff_dip', 'ff_mcp',
#  'mf_dip', 'mf_mcp', 'rf_dip', 'rf_mcp', 'lf_dip', 'lf_mcp')
```

示例：

```python
with DexHandController("left") as hand:
    action = [0, 0, 30, 0, 45, 45, 45, 45, 45, 45, 45, 45]
    hand.send_action(action)
```

## 状态格式

`read_state()` 返回普通 dict：

```python
{
    "timestamp": 1710000000.123,
    "hand": "left",
    "joints": {
        "ff_mcp": {
            "timestamp": 1710000000.120,
            "angle": 42.0,
            "encoder_position": 1234,
            "current": 100,
            "velocity": 20,
            "error_code": None,
            "impedance": None,
        }
    },
    "tactile": {
        "ff": {
            "timestamp": 1710000000.121,
            "force_x": 0.00618,
            "force_y": 0.01902,
            "force_z": 0.12,
            "force_direction_valid": True,
            "force_z_delta_raw": 3,
            "force_xy_delta_raw": 1,
            "proximity": 450,
            "temperature": 27,
        }
    },
}
```

## 指尖触觉坐标系

`portable_dexhand021` 对外输出局部三轴力，不直接暴露 SDK 的 `normal_force/tangential_force/direction`。每个指尖都有自己的局部触觉坐标系：

- `+Z`: 垂直于指尖触觉面，正方向定义为物体压入指腹/传感器的压缩方向。因此原始 `normal_force` 会成为 `force_z`。
- `+Y`: 指尖表面内，从指根指向指尖末端的方向。原始 `direction=0°` 对应 `+Y`。
- `+X`: 指尖表面内，与 `+Y` 垂直。原始 `direction` 按从 `+Y` 朝 `+X` 的顺时针角度解释。

分解公式：

```python
force_z = normal_force
force_x = tangential_force * sin(direction_rad)
force_y = tangential_force * cos(direction_rad)
```

例如 `normal_force=0.12`、`tangential_force=0.02`、`direction=18°` 会输出约：

```python
force_x = 0.02 * sin(18°) = 0.00618 N
force_y = 0.02 * cos(18°) = 0.01902 N
force_z = 0.12 N
```

如果原始方向无效，`force_x=0`、`force_y=0`，且 `force_direction_valid=False`。`force_z_delta_raw` 和 `force_xy_delta_raw` 是原始变化量指标，不是牛顿单位。

## 力觉可视化

直接启动窗口：

```bash
python -m portable_dexhand021.tactile_viewer --hand left --rate 20
```

在控制脚本中复用同一个连接：

```python
from portable_dexhand021 import DexHandController

with DexHandController("left") as hand:
    hand.visualize_tactile(rate_hz=20)
```

窗口中圆的大小和颜色表示 `force_z`，箭头表示 `force_x/force_y` 合成的平面力向量。

## 常用参数

- `hand`: `"left"` 或 `"right"`
- `device_type`: `"auto"`、`"config"`、`"100U"`、`"200U"`
- `device_index`: 默认 `0`
- `mock`: 使用 SDK mock ZCAN，不连接真实硬件
- `tactile_period_ms`: 默认 `20`，初始化时自动配置周期反馈
- `default_control_mode`: 默认 `impedance_grasp`
- `use_broadcast`: 默认 `True`
- `current`: 当前 SDK 帧编码实际可用 `10-255 mA`，封装会提前校验

## 注意

- USBCANFD 通常不能被两个 Python 进程同时打开。控制和力觉可视化应放在同一个 `DexHandController` 进程内。
- 被动读取力觉前需要开启周期反馈；本封装默认在初始化时执行。
- 清错使用 SDK 推荐的逐板方式：`hand.clear_errors()`。
