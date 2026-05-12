# portable_DMTAC_gripper

`portable_DMTAC_gripper` 是一个可直接运行的本地代码库，用于控制 Inspire 夹爪，并同步读取夹爪末端的两个 DM-Tac 视触觉传感器。

本目录后续可能继续改底层代码，因此不要把 `portable_DMTAC_gripper` 自己做成 pip 包安装。运行方式是进入本目录后直接执行脚本；修改源码后重新运行即可生效。`dmrobotics` SDK 可以单独安装到当前 conda 环境中，本代码会优先使用环境里的 `dmrobotics`，导入失败时才尝试使用父目录中的本地源码。

## 环境

```bash
conda activate grasp_py310
cd /home/zc/HandMotion/DM_Tac_SDK_Publish_1.2.10
python -m pip install -e .

cd /home/zc/HandMotion/DM_Tac_SDK_Publish_1.2.10/portable_DMTAC_gripper
python -m pip install -r requirements.txt
python run_gripper.py --help
```

说明：

- 如果实际环境名不是 `grasp_py310`，用当前调试环境名替换。
- `python -m pip install -e .` 在 SDK 根目录执行，用于安装 `dmrobotics`；不要在 `portable_DMTAC_gripper/` 内执行。
- `requirements.txt` 只包含本目录运行脚本需要的外部依赖，例如 `numpy`、`pyserial`、`opencv-python`。

## 硬件检查

先确认 USB 设备在线：

```bash
lsusb
```

通常应看到一个 CH340 串口设备和两个 Realtek N160MU2 触觉相机。然后在 `portable_DMTAC_gripper/` 内检查枚举：

```bash
python run_gripper.py list-ports
python run_gripper.py list-tactile
```

常用夹爪串口是 `/dev/ttyUSB0`，夹爪 ID 通常是 `1`。如果串口权限不足，把用户加入 `dialout` 后重新登录：

```bash
sudo usermod -aG dialout $USER
```

## 基础调试

只读夹爪状态：

```bash
python run_gripper.py verify-gripper --port /dev/ttyUSB0 --id 1
```

只读两个触觉传感器，默认读取 `distributed_force`：

```bash
python run_gripper.py verify-tactile --tactile_ids L25480055 L25480053 --backend cpu --count 10
```

低速低力矩验证夹爪动作：

```bash
python run_gripper.py set-speed 0.2 --port /dev/ttyUSB0 --id 1
python run_gripper.py set-torque 0.2 --port /dev/ttyUSB0 --id 1
python run_gripper.py open --port /dev/ttyUSB0 --id 1 --wait
python run_gripper.py move 0.2 --port /dev/ttyUSB0 --id 1 --speed 0.2 --torque 0.2 --wait
python run_gripper.py stop --port /dev/ttyUSB0 --id 1
```

夹爪和触觉联合读取：

```bash
python run_gripper.py read \
  --port /dev/ttyUSB0 --id 1 \
  --tactile_ids L25480055 L25480053 \
  --backend cpu \
  --count 100
```

## 触觉可视化

实时可视化使用 `getDistributeForce()` 的 XY 两个通道：

```text
箭头方向 = (Fx, Fy)
颜色强度 = sqrt(Fx^2 + Fy^2)
```

Z 轴不参与当前可视化。推荐启动命令：

```bash
python run_gripper.py visualize-force \
  --tactile_ids L25480055 L25480053 \
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

启动时保持两个传感器无接触，尤其是 `--reset_tactile`、`--warmup_frames` 和 `--baseline_frames` 期间。当前实现会把这些去噪参数下发到读取层，因此 `DMTacGripperDevice.read_tactile()` 返回的 `frame.distributed_force` 已经是去噪后的数据；可视化层只负责显示。

如果需要更快响应，可以先尝试：

```bash
python run_gripper.py visualize-force \
  --tactile_ids L25480055 L25480053 \
  --backend cpu \
  --reset_tactile \
  --warmup_frames 30 \
  --viz_mode delta \
  --baseline_frames 20 \
  --fixed_ref 0.0005 \
  --deadband_abs 5e-5 \
  --temporal_alpha 0.45 \
  --spatial_blur_ksize 3 \
  --arrow_pool 12 \
  --stride 12 \
  --scale 2 \
  --max_hz 30
```

数值诊断：

```bash
python run_gripper.py probe-force \
  --tactile_ids L25480055 L25480053 \
  --backend cpu \
  --reset_tactile \
  --warmup_frames 30 \
  --count 60
```

诊断输出中主要看 `xy_max`、`dxy_max`、`z_abs_max` 和 `dxyz_max`。如果 `z_abs_max` 明显变化但 `xy_max` 很小，说明接触主要在法向 Z 方向，当前 XY 可视化不会很明显。

如果怀疑读取层去噪或 baseline 有问题，先关闭去噪看 SDK 原始值：

```bash
python run_gripper.py probe-force \
  --tactile_ids L25480055 L25480053 \
  --backend cpu \
  --reset_tactile \
  --warmup_frames 30 \
  --no_denoise \
  --count 30
```

`--warmup_frames` 当前会先丢弃 reset 后的原始过渡帧，再重新建立 `distributed_force` 的去噪基线。若某个传感器仍有固定方向初始偏置，优先增大 `--warmup_frames`，例如 `60` 或 `90`。

## Python 调用

建议把调试脚本放在 `portable_DMTAC_gripper/` 内运行。若脚本在其他目录，需要自行把本目录加入 `sys.path`。

最小示例：

```python
from portable_dmtac_gripper import (
    DMTacGripperDevice,
    GripperConfig,
    TactileConfig,
    TactileDenoiseConfig,
)

gripper = GripperConfig(port="/dev/ttyUSB0", gripper_id=1)
tactile = TactileConfig(
    dev_ids=("L25480055", "L25480053"),
    backend="cpu",
    fields=("distributed_force",),
    denoise_distributed_force=True,
    denoise_config=TactileDenoiseConfig(
        mode="delta",
        baseline_frames=20,
        deadband_abs=5e-5,
        temporal_alpha=0.35,
        spatial_blur_ksize=5,
    ),
)

with DMTacGripperDevice(gripper=gripper, tactile=tactile) as dev:
    dev.clear_fault()
    dev.set_speed(0.2)
    dev.set_torque(0.2)

    dev.reset_tactile()
    dev.warmup_tactile_filter(frames=30, timeout_ms=100)

    snapshot = dev.read(tactile_timeout_ms=100)
    gripper_state = snapshot.gripper
    tactile_frames = snapshot.tactile
    force_55 = tactile_frames["L25480055"].distributed_force
```

如果需要 SDK 原始 `distributed_force`，关闭读取层去噪：

```python
TactileConfig(..., denoise_distributed_force=False)
```

常用夹爪接口：

| 函数 | 作用 |
| --- | --- |
| `set_speed(speed)` | 设置归一化速度，范围 `0.0-1.0` |
| `set_torque(torque)` | 设置归一化力矩，范围 `0.0-1.0` |
| `clear_fault()` | 清除夹爪错误 |
| `open(wait=False)` | 张开夹爪 |
| `close_gripper(speed=None, torque=None, continuous=False, wait=False)` | 闭合夹爪 |
| `move(joint_pos, speed=None, torque=None, continuous=None, wait=False)` | 移动到归一化位置 |
| `stop()` | 停止夹爪 |
| `read_gripper_state()` | 读取夹爪状态 |

夹爪位置约定：

```text
joint_pos = 0.0  完全张开
joint_pos = 1.0  完全闭合
```

常用触觉接口：

| 函数 | 作用 |
| --- | --- |
| `wait_tactile_ready(timeout_s=10)` | 等待触觉传感器 ready |
| `reset_tactile()` | 调用 SDK reset，并重置读取层去噪状态 |
| `warmup_tactile_filter(frames=30, timeout_ms=100)` | 读取若干帧，用于稳定输出和建立基线 |
| `read_tactile(timeout_ms=100)` | 只读取触觉数据 |
| `read(read_gripper=True, read_tactile=True)` | 同时读取夹爪和触觉 |

## 触觉数据字段

`TactileConfig(fields=...)` 在初始化时决定每帧读取哪些字段。默认只读取 `distributed_force`，避免全量读取拖慢频率。

| 字段 | SDK 函数 | 数据格式 | 含义 |
| --- | --- | --- | --- |
| `raw` | `getRawImg()` | `DMTacImage.img: (480, 640), uint8` | 原始灰度触觉图像 |
| `infer` | `getInferImg()` | `DMTacImage.img: (288, 384), uint8` | SDK 预处理/推理图像 |
| `deformation2d` | `getDeformation2D()` | `(288, 384, 2), float32` | 2D 位移场 `(dx, dy)` |
| `shear` | `getShear()` | `(288, 384, 2), float32` | 2D 剪切/切向场 |
| `depth` | `getDepth()` | `(288, 384), float32` | 法向深度/压入估计 |
| `force` | `getForce()` | `(6,), float32` | `Fx, Fy, Fz, Mx, My, Mz` |
| `distributed_force` | `getDistributeForce()` | `(288, 384, 3), float32` | 每像素三维力分布 |
| `contact_area` | `getContactArea()` | `float` | SDK 聚合接触面积 |
| `base_frame` | `getBaseFrame()` | `DMTacImage` | 当前基准帧 |
| `events` | `getEvents()` | `(events, last_seq)` | SDK 事件日志 |

示例：

```python
TactileConfig(fields=("distributed_force",))
TactileConfig(fields=("distributed_force", "depth", "shear"))
TactileConfig(fields=("all",))
```

注意：`distributed_force` 依赖 SDK 内部的 deformation、shear、depth、force 推理链路。本封装会自动打开这些计算，但只返回 `fields` 中要求的字段。

## 去噪参数

`TactileDenoiseConfig` 只作用于 `distributed_force`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `mode` | `"delta"` | `"delta"` 为当前帧减启动基线；`"absolute"` 为原始绝对场；`"frame_delta"` 为相邻帧差分 |
| `baseline_frames` | `20` | `delta` 模式建立基线的帧数 |
| `deadband_abs` | `5e-5` | XY 力幅值低于该阈值时置零 |
| `temporal_alpha` | `0.35` | 时域低通系数，越大响应越快，越小越稳 |
| `spatial_blur_ksize` | `5` | 空间高斯平滑核大小，`0` 或 `1` 表示关闭 |

## 示例脚本

| 脚本 | 用途 |
| --- | --- |
| `examples/verify_gripper.py` | 验证夹爪状态读取 |
| `examples/verify_tactile.py` | 验证 DM-Tac 读取 |
| `examples/minimal_gripper_tactile_loop.py` | 低速夹爪动作 + 触觉读取 |
| `examples/visualize_distributed_force.py` | 实时可视化 `distributed_force` XY |

## 安全

- 第一次动作使用低速度、低力矩，例如 `0.2`。
- 闭合前确认夹爪运动范围内没有手、线缆或硬物。
- 触觉 reset、warmup、baseline 时保持传感器无接触。
- 如果夹爪 `error_code != 0`，先停机、清故障并检查机械和供电。
