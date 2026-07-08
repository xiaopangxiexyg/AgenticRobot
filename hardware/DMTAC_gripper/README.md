# portable_DMTAC_gripper

`portable_DMTAC_gripper` 是一个可直接运行的本地代码库，用于控制 Inspire 夹爪，并同步读取夹爪末端的两个 DM-Tac 视触觉传感器。

进入本目录后直接执行脚本；修改源码后重新运行即可生效。`dmrobotics` SDK 可以单独安装到当前 conda 环境中；本仓库内的 SDK 放在 `assets/DM_Tac_SDK_Publish_1.2.10`。

## 环境

```bash
conda activate XXXXXXXX

# 在 DexTeleop 仓库根目录执行
export REPO=$(pwd)
cd "$REPO/assets/DM_Tac_SDK_Publish_1.2.10"
python -m pip install -e .

cd "$REPO/hardware/DMTAC_gripper"
python -m pip install -r requirements.txt
python run_gripper.py --help
```

说明：

- 如果实际环境名不是 `XXXXXXXX`，用当前调试环境名替换。
- `python -m pip install -e .` 在 SDK 根目录执行，用于安装 `dmrobotics`；不要在 `hardware/DMTAC_gripper/` 内执行。
- `requirements.txt` 只包含本目录运行脚本需要的外部依赖，例如 `numpy`、`pyserial`、`opencv-python`。

## 硬件检查

先确认 USB 设备在线：

```bash
lsusb
lsusb -t
ls /dev/video*
```

通常应看到一个 CH340 串口设备和两个 Realtek N160MU2 触觉相机。然后在 `hardware/DMTAC_gripper/` 内检查枚举：

```bash
python run_gripper.py list-ports
python run_gripper.py list-tactile
```

如果 `lsusb` 能看到 `1a86:7523 QinHeng Electronics CH340 serial converter`，但
`lsusb -t` 里对应设备显示 `Driver=` 为空，并且没有 `/dev/ttyUSB*`，说明 CH340 已被 USB
枚举但没有绑定 Linux 串口驱动。先加载 `ch341`：

```bash
sudo modprobe ch341
lsmod | grep ch341
```

然后拔插 gripper 的 USB-RS485 转接器，重新检查：

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
lsusb -t
python run_gripper.py list-ports
```

正常情况下 `lsusb -t` 中 CH340 那一行应显示 `Driver=ch341`，并出现 `/dev/ttyUSB0` /
`/dev/ttyUSB1`。如果 `sudo modprobe ch341` 报模块不存在，需要安装当前内核对应的额外模块包，
例如 Ubuntu/Debian 常见为：

```bash
sudo apt update
sudo apt install "linux-modules-extra-$(uname -r)"
sudo modprobe ch341
```

如果只出现一个 `/dev/ttyUSB*`，而另一只 CH340 在 `lsusb -t` 里显示 `Driver=usbfs`，
说明该 USB 接口被用户态程序占用，内核 `ch341` 驱动没有接管。先看是谁占用，例如设备为
Bus 003 Device 071 时：

```bash
sudo fuser -v /dev/bus/usb/003/071
sudo dmesg -T | grep -E "ch341|ttyUSB|brltty|1a86|usbfs" | tail -n 80
```

Ubuntu 上常见原因是 `brltty` 抢占 USB 串口。若不需要盲文终端，优先使用 DexTeleop 的设备脚本
停止/禁用相关服务，然后重新拔插 gripper 的 USB-RS485 转接器：

```bash
sudo -E "$CONDA_PREFIX/bin/python" -m teleop_runtime.configure_devices --fix-serial-conflicts
sudo -E "$CONDA_PREFIX/bin/python" -m teleop_runtime.configure_devices --fix-permissions
sudo modprobe ch341
```

如果仍然被抢占，再考虑 `sudo apt remove brltty`。如果这台机器需要盲文终端，不要禁用
`brltty`，改用设备专用 udev 例外或换一台遥操机器。

常用夹爪串口是 `/dev/ttyUSB0`，夹爪 ID 通常是 `1`。如果夹爪通过 USB HUB 接入，
`list-ports` 里优先看 `usb_aliases` 的 `/dev/serial/by-path/...`，它能体现 HUB 拓扑；
同时确认 `permissions.readable/writable` 都是 `true`。如果串口权限不足，把用户加入 `dialout` 后重新登录：

```bash
sudo usermod -aG dialout $USER
```

临时调试可直接放开当前枚举到的 USB 串口权限：

```bash
sudo chmod a+rw /dev/ttyUSB0 /dev/ttyACM0
```

如果 `lsusb -t` 能看到 N160MU2，但 `configure-usb --scan-only` 只显示
`tactile candidates not usable by SDK`，且 reason 为 `missing_device_node` 或 `not_readable`，
优先检查当前进程是否能访问 `/dev/video*`，以及用户是否在 `video` 组中：

```bash
groups
sudo usermod -aG video $USER
```

重新登录后再运行扫描。

## USB 配置保存

夹爪和触觉传感器不一定成对出现：可以只有夹爪、只有触觉，也可以左右两套夹爪各带
inner/outer 两个触觉传感器。首次上机建议先扫描、测试、分配并保存配置：

```bash
python run_gripper.py configure-usb --scan-only
python run_gripper.py configure-usb --scan-only --test-gripper 0 --motion-amplitude 0.3
python run_gripper.py configure-usb --scan-only --visualize-tactile 1 --reset_tactile --warmup_frames 30
python run_gripper.py configure-usb --assign-tactile-gui --interactive --timeout_ms 20 --max_fps 60 --tactile-gui-max-hz 30 --save
```

`configure-usb` 默认会在疑似 gripper 的 USB 串口上自动扫描 Modbus id `1..247`，并尝试
`115200/57600/19200` 三种波特率；HUB 后面如果线缆或供电不稳，常见表现是所有组合都
`Serial read timeout`。这种情况可先加长超时，再看 `probes_tried` 和 `baudrates`：

```bash
python run_gripper.py configure-usb --scan-only --timeout 0.1
```

如果已知 id/波特率，可跳过自动扫描：

```bash
python run_gripper.py configure-usb --scan-only \
  --id 32 \
  --baudrate 115200 \
  --no-scan-gripper-ids \
  --no-scan-baudrates
```

`--assign-tactile-gui` 会自动取本次扫描到的最多 4 个 DM-Tac 触觉传感器，实时显示
轻量 20x30x3 力场：每个传感器聚合成 20 行 x 30 列，每格保留 `(Fx, Fy, Fz)` 平均力，
颜色显示正压力 `max(sign * Fy, 0)`，抽样箭头显示 `X/Z` 平面向量，
然后在窗口里分配。GUI 默认不等待新帧、
不做触觉去噪/时间平滑，以便排查延迟；窗口底部会显示 `read=...ms render=...ms display=...ms loop=...ms`，
每个传感器标题区显示 `fid_age=...ms df=...ms force=...ms`，其中 `df` 是
`getDistributeForce()` getter 耗时。
开始可视化后会自动记录读数到
`teleop_records/dmtac_tactile_debug/<timestamp>_tactile_gui/`：`metadata.json` 保存本次配置，
`frames.jsonl` 逐帧保存 fid、getter 耗时和力场摘要；默认不写完整 20x30 网格，以免文件 IO 拖慢下一帧。

```text
1-4 选择传感器
A   left inner
S   left outer
K   right inner
L   right outer
U   取消当前传感器分配
C   清空全部触觉分配
Enter 确认
Esc   取消
```

如果扫描到超过 4 个触觉传感器，默认只显示前 4 个；也可以显式指定：

```bash
python run_gripper.py configure-usb \
  --assign-tactile-gui \
  --tactile-gui-devices L25480055 L25480053 L25480056 L25480057 \
  --tactile-gui-max-hz 30 \
  --timeout_ms 20 \
  --interactive \
  --save
```

如果按压后没有正压力响应，但 `y` 数值是负的，传感器法向符号可能相反，追加
`--tactile-gui-normal-sign -1`。如果需要旧的完整 XY 力图，可加
`--tactile-gui-render-mode xy --tactile-gui-decimation 2`；如果只想看平滑后的力场，再加
`--tactile-gui-denoise`。默认每 2 个网格画一个 `X/Z` 箭头；如果要每个 20x30 网格都画箭头，
追加 `--tactile-gui-vector-stride 1`。
空载时 DM-Tac 会有 `1e-6` 量级微小噪声；GUI 默认用显示层 deadband 和固定缩放抑制闪烁，
默认 `Y+` 正压力 deadband 为 `5e-6`、`X/Z` 向量 deadband 为 `2e-5`，不会改 SDK 标定。若需要看更弱信号，可调低
`--tactile-gui-pressure-deadband` / `--tactile-gui-vector-deadband`；若空载仍闪烁，可调高
deadband 或 `--tactile-gui-pressure-ref` / `--tactile-gui-vector-ref`。
如果需要离线分析完整网格，可加 `--tactile-gui-record-grid`；如果记录仍过大，可加
`--tactile-gui-record-every-n 3` 降低记录频率。临时不记录则加 `--tactile-gui-no-record`。
多传感器默认并行读取，以降低 2-4 个触觉传感器顺序 `getDistributeForce()` 叠加出的延迟。
GUI 默认使用 `--tactile-gui-compute-profile dist-only`，只为 `distributed_force` 打开最小
force 计算链路；如果这个 SDK/固件版本下没有力场输出，可加
`--tactile-gui-compute-profile legacy` 回到旧的 deformation/shear/depth/force 全链路。
如果要和顺序读取做 A/B 对比，可加 `--tactile-gui-sequential-read`。

只有在需要确认每次显示都是新触觉帧时才加 `--tactile-gui-wait-for-new`；该模式会增加延迟，
多个传感器顺序读取时尤其明显。

也可以非交互保存，参数既可以写扫描 index，也可以写实际 port/dev_id：

```bash
python run_gripper.py configure-usb \
  --left-gripper 0 \
  --left-inner L25480055 \
  --left-outer L25480053 \
  --right-gripper 1 \
  --right-inner L25480056 \
  --right-outer L25480057 \
  --save
```

配置默认保存到仓库的 `teleop_runtime/configs/hardware/dmtac/`，同时更新
`latest.json` 并记录 `hostname`。后续独立调试可以直接加载：

```bash
python run_gripper.py verify-gripper --config latest --side left
python run_gripper.py verify-tactile --config latest --side left --tactile-slots inner outer --count 10
python run_gripper.py visualize-force --config latest --side right --tactile-slots inner --reset_tactile --warmup_frames 30
```

加载配置时会重新扫描 USB。配置中的设备缺失时默认拒绝运行；当前机器多出未分配设备时
会打印 warning，并要求输入 `YES` 确认。脚本化运行时只有在人工确认过后才使用
`--yes`。

## 基础调试

只读夹爪状态：

```bash
python run_gripper.py verify-gripper --port /dev/ttyUSB0 --id 1
```

只读两个触觉传感器，默认读取 `raw` 原始图像；需要实时力图时显式传 `--tactile_fields distributed_force`：

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

普通 gripper 读写命令默认会在 transient Modbus I/O 错误后重开串口并重试一次，
可用 `--no-gripper-reconnect` 关闭，或用 `--gripper-reconnect-attempts` /
`--gripper-reconnect-delay` 调整。`configure-usb` 的 gripper id/baudrate probe
会关闭这层重试，避免扫描大量错误 id 时被额外重连拖慢。

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
  --temporal_alpha 0.85 \
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
  --temporal_alpha 0.85 \
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
  --max_fps 60 \
  --tactile-compute-profile dist-only \
  --reset_tactile \
  --warmup_frames 30 \
  --count 60
```

诊断输出会打印整体 `read=...ms`、`profile=...`、并行状态，以及每个传感器的
`distributed_force=...ms` 等 getter 耗时。数值上主要看 `xz_max`、`dxz_max`、`y_abs_max`、
`dy_abs_max` 和 `dxyz_max`。
当前约定中正压力在 Y 方向；如果 `y_abs_max` 明显变化但平面向量很小，说明接触主要在法向 Y 方向。
如果 `dist-only` 没有 `distributed_force` 输出，改用 `--tactile-compute-profile legacy`。

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

建议把调试脚本放在 `hardware/DMTAC_gripper/` 内运行。若脚本在其他目录，需要自行把本目录加入 `sys.path`。

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
        temporal_alpha=0.85,
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

`TactileConfig(fields=...)` 在初始化时决定每帧读取哪些字段。默认只读取 `raw` 原始图像，避免实时力解算拖慢频率；需要实时力图时显式配置 `distributed_force`。

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
TactileConfig(fields=("raw",))
TactileConfig(fields=("distributed_force",))
TactileConfig(fields=("distributed_force", "depth", "shear"))
TactileConfig(fields=("all",))
```

`TactileConfig(compute_profile=...)` 控制 SDK 初始化时启用哪些计算通道。默认
`"dist-only"` 对 `raw` 默认字段不会启用 force 链路；配合 `distributed_force` 时会打开最小 force 链路；
`"legacy"` 回到旧的 deformation/shear/depth/force 全链路；`"shear-depth"` 和
`"requested"` 主要用于定位不同 SDK/固件版本的 getter 延迟。

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
