# portable_DMTAC_gripper

> 本页维护 DMTAC 单设备 CLI。整机设备检查与标定顺序统一从
> [`docs/hardware/README.md`](../../docs/hardware/README.md) 开始。

`portable_DMTAC_gripper` 是一个可直接运行的本地代码库，用于控制 Inspire 夹爪，并同步读取夹爪末端的两个 DM-Tac 视触觉传感器。

进入本目录后直接执行脚本；修改源码后重新运行即可生效。`dmrobotics` SDK 可以单独安装到当前 conda 环境中；本仓库内的 SDK 放在 `assets/DM_Tac_SDK_Publish_1.2.10`。

## 环境

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"

# 在 DexTeleop 仓库根目录执行
export REPO=$(pwd)
cd "$REPO/assets/DM_Tac_SDK_Publish_1.2.10"
python -m pip install -e .

cd "$REPO/hardware/DMTAC_gripper"
python -m pip install -r requirements.txt
python run_gripper.py --help
```

说明：

- 如果安装完整项目环境时使用了 `--env-name NAME`，先设置
  `DEXTELEOP_ENV_NAME=NAME`；不要写死 Conda 安装目录。
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
`list-ports` 里的 `usb_aliases` 会同时保留 `/dev/ttyUSB*` 和
`/dev/serial/by-path/...`。数采运行时优先使用 `/dev/ttyUSB*`，避免夹爪成对
断电后 HUB topology path 变化导致重新分配；`by-path` 仍保留用于身份审计和诊断。
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
inner/outer 两个触觉传感器。首次上机建议先扫描，再按要修改的范围分配并保存：

```bash
python run_gripper.py configure-usb --scan-only
python run_gripper.py configure-usb --grippers-only --save
python run_gripper.py configure-usb --tactile-only --tactile-timeout-ms 20 --tactile-max-fps 60 --save
python run_gripper.py configure-usb --assign-tactile-gui --tactile-timeout-ms 20 --tactile-max-fps 60 --save
```

`--scan-only` 严格只扫描，不能和保存、分配、运动测试或可视化参数组合。
`--grippers-only` 和 `--tactile-only` 默认合并 `latest`，分别保留未修改的 tactile 或
gripper。完整重新分配时不使用这两个模式。

`configure-usb` 默认会在疑似 gripper 的 USB 串口上自动扫描 Modbus id `1..247`，并尝试
`115200/57600/19200` 三种波特率；HUB 后面如果线缆或供电不稳，常见表现是所有组合都
`Serial read timeout`。这种情况可先加长超时，再看 `probes_tried` 和 `baudrates`：

默认找到 2 个可达夹爪后立即停止探测剩余候选；失败候选不计入这 2 个。需要排查第三个
设备或穷举所有端口时，显式传 `--max-reachable-grippers 0`。

```bash
python run_gripper.py configure-usb --scan-only --gripper-timeout 0.1
```

如果已知 id/波特率，可跳过自动扫描：

```bash
python run_gripper.py configure-usb --scan-only \
  --gripper-id 32 \
  --gripper-baudrate 115200 \
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
开始可视化后默认仍会自动记录读数到
`teleop_records/dmtac_tactile_debug/<timestamp>_tactile_gui/`；显式传入
`--tactile-gui-artifact-layout runtime-artifacts-v1` 时改写到
`runtime_artifacts/runs/dmtac_tactile_gui/<date>/<timestamp>_tactile_gui/`，而
`--tactile-gui-record-dir` 始终优先。`metadata.json` 保存本次配置与实际存储选择，
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
  --tactile-only \
  --tactile-gui-devices L25480055 L25480053 L25480056 L25480057 \
  --tactile-gui-max-hz 30 \
  --tactile-timeout-ms 20 \
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
python run_gripper.py visualize-force --config latest --side right --tactile-slots inner --reset-tactile --warmup-frames 30
```

独立读取/运动命令默认直接使用 assignment 中保存的稳定设备路径，不重新扫描 USB；
需要在执行前核对当前接线时显式添加 `--validate-current-usb`。启用校验后，配置中的设备
缺失会默认拒绝运行；当前机器多出未分配设备时会打印 warning 并要求输入 `YES`。
脚本化运行时只有在人工确认过后才使用 `--yes`。

## 基础调试

### 独立夹爪调试（不加载触觉/机械臂/遥操）

`run_gripper_only.py` 是新增的 gripper-only 入口，只导入串口夹爪驱动，并从现有
left/right USB assignment 读取端口。它不改动原有 `run_gripper.py` 或遥操调用路径。

只读左右夹爪状态、当前力度和速度设置：

```bash
python hardware/DMTAC_gripper/run_gripper_only.py --config latest status --side both
```

分别预览左右夹爪的开度和力度。这里 `opening=0` 表示完全闭合，`opening=1` 表示完全
张开；`force` 是设备归一化设置 `[0,1]`，不是牛顿：

```bash
python hardware/DMTAC_gripper/run_gripper_only.py --config latest set \
  --left-opening 0.8 --left-force 0.20 \
  --right-opening 0.5 --right-force 0.15
```

`set` 默认只输出 preview，不连接硬件。确认端口、周围空间和数值无误后，单独执行左夹爪：

```bash
python hardware/DMTAC_gripper/run_gripper_only.py --config latest set \
  --side left --opening 0.8 --force 0.20 --speed 0.20 \
  --execute --confirmation GRIPPER_ONLY --wait
```

只修改力度寄存器而不发送位置命令，可省略 `--opening`：

```bash
python hardware/DMTAC_gripper/run_gripper_only.py --config latest set \
  --side right --force 0.20 \
  --execute --confirmation GRIPPER_ONLY
```

大于 `0.5` 的力度还必须同时使用 `--allow-high-force --high-force-confirmation HIGH_FORCE`。
也可用 `--config none --left-port /dev/ttyUSB0` 绕过 assignment，其他串口参数有对应的
`--left/right-id`、`--left/right-baudrate` 和 `--left/right-timeout` 覆盖项。

只读夹爪状态：

```bash
python run_gripper.py verify-gripper --port /dev/ttyUSB0 --gripper-id 1
```

只读两个触觉传感器，默认读取 `raw` 原始图像；需要实时力图时显式传
`--tactile-fields distributed_force`：

```bash
python run_gripper.py verify-tactile --tactile-ids L25480055 L25480053 --tactile-backend cpu --count 10
```

低速低力矩验证夹爪动作：

```bash
python run_gripper.py set-speed 0.2 --port /dev/ttyUSB0 --gripper-id 1
python run_gripper.py set-torque 0.2 --port /dev/ttyUSB0 --gripper-id 1
python run_gripper.py open --port /dev/ttyUSB0 --gripper-id 1 --wait
python run_gripper.py move 0.2 --port /dev/ttyUSB0 --gripper-id 1 --speed 0.2 --torque 0.2 --wait
python run_gripper.py stop --port /dev/ttyUSB0 --gripper-id 1
```

普通 gripper 读写命令默认会在 transient Modbus I/O 错误后重开串口并重试一次，
可用 `--no-gripper-reconnect` 关闭，或用 `--gripper-reconnect-attempts` /
`--gripper-reconnect-delay` 调整。`configure-usb` 的 gripper id/baudrate probe
会关闭这层重试，避免扫描大量错误 id 时被额外重连拖慢。

### 遥操时序约定

共享 DMTac 遥操后端默认以 20 Hz 接收命令、20 Hz 读取位置反馈。一次写命令完成后仅保留
20 ms 的串口读保护窗口；保护窗口按“写完成时间”计算，连续命令不会再反复把状态读取
推迟 150 ms。若连续命令仍让每次遥测恰好落入保护窗口，后端会在两个遥测周期后强制
执行一次串行化位置读取；默认 20 Hz 遥测下最大延期为 100 ms。`joint_timestamp` /
`gripper_measurement_timestamp` 是夹爪位置的权威采样
时间；`telemetry_timestamp` 只表示后台循环运行过，不能用于刷新被跳过读取时的旧位置。

独立执行器辨识表明，命令调用本身约 37 ms，命令开始到首次可见运动约 0.13 s，而全行程
约 0.85 s。因此 action 与目标位置完全重合所需的多帧偏移包含物理运动过程，不能全部
解释成通信死区。遥操数据会额外记录读写耗时、跳读原因、命令开始/完成时间和位置年龄，
用于区分通信排队、采样陈旧和执行器运动时间。

夹爪和触觉联合读取：

```bash
python run_gripper.py read \
  --port /dev/ttyUSB0 --gripper-id 1 \
  --tactile-ids L25480055 L25480053 \
  --tactile-backend cpu \
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
  --tactile-ids L25480055 L25480053 \
  --tactile-backend cpu \
  --reset-tactile \
  --warmup-frames 30 \
  --viz-mode delta \
  --baseline-frames 20 \
  --fixed-ref 0.0005 \
  --deadband-abs 5e-5 \
  --temporal-alpha 0.85 \
  --spatial-blur-ksize 5 \
  --arrow-pool 16 \
  --stride 12 \
  --scale 2 \
  --max-hz 30
```

启动时保持两个传感器无接触，尤其是 `--reset-tactile`、`--warmup-frames` 和
`--baseline-frames` 期间。当前实现会把这些去噪参数下发到读取层，因此
`DMTacGripperDevice.read_tactile()` 返回的 `frame.distributed_force` 已经是去噪后的数据；
可视化层只负责显示。

如果需要更快响应，可以先尝试：

```bash
python run_gripper.py visualize-force \
  --tactile-ids L25480055 L25480053 \
  --tactile-backend cpu \
  --reset-tactile \
  --warmup-frames 30 \
  --viz-mode delta \
  --baseline-frames 20 \
  --fixed-ref 0.0005 \
  --deadband-abs 5e-5 \
  --temporal-alpha 0.85 \
  --spatial-blur-ksize 3 \
  --arrow-pool 12 \
  --stride 12 \
  --scale 2 \
  --max-hz 30
```

数值诊断：

```bash
python run_gripper.py probe-force \
  --tactile-ids L25480055 L25480053 \
  --tactile-backend cpu \
  --tactile-max-fps 60 \
  --tactile-compute-profile dist-only \
  --denoise \
  --reset-tactile \
  --warmup-frames 30 \
  --count 60
```

诊断输出会打印整体 `read=...ms`、`profile=...`、并行状态，以及每个传感器的
`distributed_force=...ms` 等 getter 耗时。数值上主要看 `xz_max`、`dxz_max`、`y_abs_max`、
`dy_abs_max` 和 `dxyz_max`。
当前约定中正压力在 Y 方向；如果 `y_abs_max` 明显变化但平面向量很小，说明接触主要在法向 Y 方向。
如果 `dist-only` 没有 `distributed_force` 输出，改用 `--tactile-compute-profile legacy`。

默认不执行新的 reset，也不建立读取层去噪 baseline。若当前传感器序列号已有固定
base-frame 配置，会自动加载；尚无配置时才直接显示 SDK 原始解算值：

```bash
python run_gripper.py probe-force \
  --tactile-ids L25480055 L25480053 \
  --tactile-backend cpu \
  --count 30
```

只有显式传入 `--reset-tactile` 才会在确认无接触的状态下重置传感器，并将 SDK 返回的
base frame 按序列号持久化到 `configs/tactile_calibration/`。以后真机运行和离线后处理
默认自动复用。显式传入 `--denoise` 才会启用读取层去噪/baseline；它与固定 SDK 校准是
两层不同处理。`--warmup-frames` 只在明确需要丢弃过渡帧或预填去噪状态时使用。

### 独立真机校准

推荐用独立脚本直接连接真实 DM-Tac。它不连接夹爪或机械臂，也不会执行任何运动：

```bash
python hardware/DMTAC_gripper/examples/calibrate_tactile_base_frames.py
```

脚本会自动发现全部 DM-Tac 序列号，要求操作者输入 `CALIBRATE` 确认表面完全无接触，
丢弃 10 帧预热数据，再采集 30 帧 raw 图像。最终 base frame 是逐像素中位数，而不是某一
张偶然帧；保存前会检查时间稳定性，随后将校准安装到当前 SDK 会话、写入固定配置并回读
SHA-256 校验。已有相同序列号配置默认拒绝覆盖，确认需要更新时使用：

```bash
python hardware/DMTAC_gripper/examples/calibrate_tactile_base_frames.py \
  --dev-ids M26190018 M26190023 M26190034 M26190173 \
  --overwrite
```

`--yes` 只用于已经由外部流程确认无接触的非交互运行。稳定性阈值可通过
`--max-mean-deviation` 和 `--max-p99-deviation` 调整；设为 `0` 表示只记录、不据此拒绝。

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
        enabled=True,
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

默认配置不会重新 reset，也不做读取层去噪，但会按序列号自动加载已有固定校准：

```python
TactileConfig(...)
```

可用 `TactileConfig(calibration_policy="required")` 强制所有传感器必须已有配置，或用
`calibration_policy="off"` 做明确的未校准诊断。`dev.reset_tactile()` 是会改变 SDK 状态的
显式操作，必须保证传感器无接触；成功后默认保存当前 base frame。

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
| `examples/calibrate_tactile_base_frames.py` | 直接连接真实传感器并生成固定 base-frame 校准 |
| `examples/inspect_dmtac_offline_api.py` | 只反射 SDK 符号和签名，不构建设备或调用 SDK 函数 |

从仓库根目录执行离线 SDK 接口检查：

```bash
python hardware/DMTAC_gripper/examples/inspect_dmtac_offline_api.py
```

## 安全

- 第一次动作使用低速度、低力矩，例如 `0.2`。
- 闭合前确认夹爪运动范围内没有手、线缆或硬物。
- 触觉 reset、warmup、baseline 时保持传感器无接触。
- 如果夹爪 `error_code != 0`，先停机、清故障并检查机械和供电。
