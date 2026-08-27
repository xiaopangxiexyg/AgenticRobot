# DMTAC / Gripper / Q6 配置说明

| 你要做什么 | 阅读位置 |
|---|---|
| 理解 assignment 和覆盖优先级 | 第 2–3 节 |
| 只读枚举/验证设备 | 第 4.1–4.2 节 |
| 重新分配 gripper 或 tactile | 第 4.3–4.5 节 |
| 在正式入口中使用 DMTAC | 第 5 节（仅参数片段） |
| 区分 Q6、AngleTrigger、UMI hand-eye | 第 6–7 节 |

## 1. Assignment 与统一规则

正式运行入口共同使用以下 DMTAC assignment：

```text
teleop_runtime/configs/hardware/dmtac/latest.json
```

该文件是左右 DMTAC gripper transport 与四个 tactile slot 的事实源，保存：

- gripper：稳定串口、Modbus ID、baudrate、timeout、设备 identity/aliases；
- tactile：`left/right × inner/outer` 的 DM-Tac `dev_id`；
- 保存时的扫描证据。时间戳快照与 `latest.json` 位于同一目录。

运行时规则：

- GUI、collection、deploy、Replay Web GUI 不再各自复制一套 DMTAC assignment 参数；
- 正式参数统一为一套可读名称，旧名称只作为隐藏兼容别名；
- 每侧手工 gripper override 补齐 `port/id/baudrate/timeout`，不再隐含固定 transport；
- `--require-dmtac-config` 会校验每个 active DMTAC side 确实解析成功；
- Replay Web GUI 与其他正式入口使用同一 assignment precedence；Replay 只连接 gripper，
  不连接 assignment 中的 tactile；
- 独立维护 CLI 的参数统一使用 kebab-case，并为子命令补齐说明；
- `configure-usb --scan-only` 现在拒绝保存、分配、运动测试和可视化选项，避免“只扫描”
  名称与实际副作用不符；
- `configure-usb` 显式区分 `--grippers-only`、`--tactile-only` 和完整分配；
- 原 `--interactive` 改名为 `--manual-all-slots`，避免与默认的 micro-jog 交互确认混淆。
- gripper USB 探测默认找到 2 个可达夹爪后立即停止，不再继续扫描剩余串口候选；
  `--max-reachable-grippers 0` 可显式恢复穷举扫描。

`latest.json` 不会被 `list-*`、`--scan-only` 或 `verify-*` 改写；只有明确的
`configure-usb ... --save` 会更新 assignment。

## 2. 配置边界

| 配置 | 保存位置 | 含义 | 不负责 |
|---|---|---|---|
| DMTAC assignment | [`../../teleop_runtime/configs/hardware/dmtac/latest.json`](../../teleop_runtime/configs/hardware/dmtac/latest.json) | 左右 gripper transport 与四个 tactile slot | 动作速度、力矩、TCP |
| DMTAC task profile | [`../../teleop_runtime/configs/gripper_task_profile.example.json`](../../teleop_runtime/configs/gripper_task_profile.example.json) | 输入映射、范围、速度、力矩、阈值与运行频率 | USB 左右身份 |
| Policy I/O layout | `teleop_runtime/configs/policy_io_layouts/` | 模型 action/observation slot 语义 | 物理串口 |
| Q6 assignment | [`../../teleop_runtime/configs/hardware/q6/latest.json`](../../teleop_runtime/configs/hardware/q6/latest.json) | 左右遥操输入设备 | DMTAC 输出夹爪 |
| AngleTrigger | `teleop_runtime/configs/hardware/angle_trigger/` | UMI/可选 GUI 的开合输入及角度端点 | DMTAC 输出夹爪 |
| RM65 tool/TCP | `hardware/rm65/utils/configs/calibrations/rm65_*_DMTAC_M.json` | 对应侧工具 TCP、payload 等几何/动力学配置 | USB assignment |
| UMI runtime config | [`../../teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json`](../../teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json) | wrist camera、tracker、AngleTrigger、DMTAC assignment、标定引用 | 独立的第二份 DMTAC 映射 |

`latest.json` 当前仓库快照为：

| side | gripper stable port | ID / baudrate / timeout | inner / outer tactile |
|---|---|---|---|
| left | `/dev/serial/by-path/pci-0000:10:00.4-usb-0:2.4.4:1.0-port0` | `1 / 115200 / 0.03 s` | `M26190023 / M26190018` |
| right | `/dev/serial/by-path/pci-0000:0e:00.0-usb-0:4.1.4.4:1.0-port0` | `1 / 115200 / 0.03 s` | `M26190173 / M26190034` |

这是审计时快照，不应复制成启动脚本常量。运行时应读取 assignment。`/dev/ttyUSB*`
只是易变化别名，不应用作长期左右身份。

## 3. 正式入口的统一参数

GUI、collection、deploy、Replay Web GUI 现在共享以下 assignment 参数：

| 参数 | 语义 |
|---|---|
| `--dmtac-config PATH_OR_LATEST` | assignment 路径，默认 `latest` |
| `--dmtac-use-config` / `--no-dmtac-use-config` | 是否先加载 assignment，默认加载 |
| `--require-dmtac-config` | active DMTAC side 解析不完整时失败退出 |
| `--dmtac-validate-usb` / `--no-dmtac-validate-usb` | 是否用当前 USB 状态校验保存的 assignment，默认校验 |
| `--dmtac-allow-missing-devices` | 仅放宽 USB 校验中的设备缺失，不修复映射 |
| `--dmtac-prompt-on-extra` | 检测到额外未分配设备时要求终端确认 |
| `--SIDE-gripper-port` | 逐侧手工覆盖，`SIDE` 为 `left/right` |
| `--SIDE-gripper-id` | 与该侧手工 port 配套的 Modbus ID |
| `--SIDE-gripper-baudrate` | 与该侧手工 port 配套的 baudrate |
| `--SIDE-gripper-timeout` | 与该侧手工 port 配套的串口 timeout，单位秒 |

优先级为：先加载 assignment，再由非空的 per-side manual port 覆盖对应侧。手工覆盖会生成
gripper-only backend，不会附带该侧 tactile；需要 gripper+tactile 时应修正 assignment。
Replay 本身不读取 tactile，因此它的手工覆盖只影响 gripper。

若要完全不用 assignment，必须同时明确 transport，例如：

```bash
python -m teleop_runtime.deploy \
  --hand-backend dmtac-gripper \
  --no-dmtac-use-config \
  --left-gripper-port /dev/serial/by-path/LEFT \
  --left-gripper-id 1 \
  --left-gripper-baudrate 115200 \
  --left-gripper-timeout 0.03 \
  --require-dmtac-config
```

不建议把该形式作为长期启动配置，因为它丢失 tactile identity 和 assignment 扫描证据。

### 3.1 兼容别名

以下旧参数仍可解析，但不再出现在 `--help`，新脚本不得继续使用：

| 旧名称 | 正式名称 |
|---|---|
| `--dmtac-config-required` | `--require-dmtac-config` |
| `--no-dmtac-default-config` | `--no-dmtac-use-config` |
| `--dmtac-allow-missing` | `--dmtac-allow-missing-devices` |
| `--dmtac-no-validate-current-usb` | `--no-dmtac-validate-usb` |
| `--dmtac-confirm-extra` | `--dmtac-prompt-on-extra` |

独立 `run_gripper.py` 中的旧 `--id`、`--baudrate`、`--timeout`、`--max_fps`、
`--timeout_ms` 及其他 underscore 参数也只保留为兼容别名；正式名称分别使用
`--gripper-*`、`--tactile-*` 和 kebab-case。

## 4. DMTAC 维护 CLI

若从 DexTeleop 仓库根运行独立 CLI，先让 Python 同时看到仓库包和 DMTAC 子包：

```bash
export PYTHONPATH="$PWD:$PWD/hardware/DMTAC_gripper:${PYTHONPATH:-}"
```

入口：

```bash
python hardware/DMTAC_gripper/run_gripper.py --help
python hardware/DMTAC_gripper/run_gripper.py configure-usb --help
```

### 4.1 命令分类

| 类别 | 命令 | 副作用 |
|---|---|---|
| 枚举 | `list-ports`、`list-tactile` | 不发送夹爪运动；tactile 枚举会初始化 SDK |
| 扫描 | `configure-usb --scan-only` | 打开候选串口并读取/probe；不保存、不分配、不运动 |
| 读取验证 | `verify-gripper`、`verify-tactile`、`read`、`probe-force` | 读取设备；不发送夹爪目标 |
| 可视化 | `visualize-force` | 持续读取 tactile 并打开窗口 |
| 基线变更 | `reset-tactile` 或 `--reset-tactile` | 修改 tactile baseline，必须完全卸载传感器 |
| assignment 写入 | `configure-usb ... --save` | 写时间戳快照并更新 `latest.json` |
| 运动/参数命令 | `open`、`close`、`move`、`stop`、`set-speed`、`set-torque` | 访问并改变真实夹爪状态 |

`--scan-only` 不能再和 `--save`、`--motion-test-gripper`、`--visualize-tactile`、
`--assign-tactile-gui` 或 assignment token 混用。需要运动测试应使用明确的运动命令；
需要 tactile 可视化应使用 `visualize-force`。

### 4.2 只读检查

```bash
python hardware/DMTAC_gripper/run_gripper.py list-ports
python hardware/DMTAC_gripper/run_gripper.py list-tactile
python hardware/DMTAC_gripper/run_gripper.py configure-usb --scan-only
python hardware/DMTAC_gripper/run_gripper.py verify-gripper --config latest --side left
python hardware/DMTAC_gripper/run_gripper.py verify-gripper --config latest --side right
python hardware/DMTAC_gripper/run_gripper.py verify-tactile \
  --config latest --side left --tactile-slots inner outer --count 10
```

夹爪扫描默认上限为 2 个**可达**设备：失败或超时的候选不计数，找到第 2 个夹爪后跳过
剩余候选。终端会打印 early-stop 提示。只有排查额外夹爪、端口排序或疑似漏检时才执行：

```bash
python hardware/DMTAC_gripper/run_gripper.py configure-usb \
  --scan-only \
  --max-reachable-grippers 0
```

`0` 表示不设上限，会探测所有候选，因此可能重新变慢。`--probe-all-serial` 只扩大候选
类型，本身不会取消“找到两个即停止”；若两者都要穷举，需同时传
`--probe-all-serial --max-reachable-grippers 0`。

这些命令仍会占用设备。运行前必须停止 GUI、collection、UMI、deploy 和 replay。

### 4.2.1 串口占用与运行时故障恢复

夹爪串口以进程独占方式打开。同一个物理串口已经被 GUI、collection、deploy、replay
或另一条维护命令占用时，新进程应在打开阶段直接失败，而不应与现有进程交错消费
Modbus 响应。不要在正式遥操运行中并行执行 `verify-gripper`。尤其不要用易漂移的
`/dev/ttyUSB*` 猜测左右夹爪；读取验证应使用 `--config latest --side left/right`，让命令
复用 assignment 中的稳定 by-path、ID、baudrate 和 timeout。

正式遥操中的夹爪恢复遵循以下边界：

- 控制循环不会等待正在进行的夹爪 I/O；锁忙时该次夹爪命令直接记为 `io_busy`，机械臂
  控制循环继续运行；
- 串口读写失败后，当前夹爪连接立即隔离，该次命令失败且不会在控制线程内重发；
- 后台 telemetry 按 `reconnect_cooldown_sec` 重试打开连接。只有重新读取到与配置 ID
  一致的完整状态后，才清除连接错误并增加 `reconnect_count`；
- 重连后若设备仍报告非零 `gripper_error_code`，命令继续按硬件故障拒绝，不会因“串口
  已重新打开”而绕过故障门；
- 夹爪断电重启期间不需要退出整个遥操终端。恢复前夹爪命令保持 fail-closed，状态可从
  `gripper_is_connected`、`telemetry_error`、`gripper_last_reconnect_error` 和
  `reconnect_count`
  判断。

若电源恢复后仍长期出现 `Unexpected slave id`，应先停止所有可能占用该串口的进程，
再核对 assignment 的稳定路径和 `gripper_id`。该错误表示响应身份与当前配置不一致，
不能通过盲目重发或放宽校验处理。

### 4.3 只重新确认左右 gripper，保留 tactile

推荐交互流程会对可达夹爪进行逐个 micro-jog；每次确认前必须清空运动范围：

```bash
python hardware/DMTAC_gripper/run_gripper.py configure-usb \
  --grippers-only \
  --gripper-jog-amplitude 0.1 \
  --save
```

`--grippers-only` 默认从 `latest` 合并未改动的 tactile slot，不必再写重复的
`--merge-latest`。若已经通过独立证据确认稳定路径，可不用 micro-jog，显式指定：

```bash
python hardware/DMTAC_gripper/run_gripper.py configure-usb \
  --grippers-only \
  --left-gripper /dev/serial/by-path/LEFT \
  --right-gripper /dev/serial/by-path/RIGHT \
  --save
```

### 4.4 只重新分配 tactile，保留 gripper

```bash
python hardware/DMTAC_gripper/run_gripper.py configure-usb \
  --tactile-only \
  --tactile-timeout-ms 20 \
  --tactile-max-fps 60 \
  --tactile-gui-max-hz 30 \
  --save
```

`--tactile-only` 在没有显式 slot token 时自动打开 tactile assignment GUI，并从 `latest`
保留两个 gripper。GUI 中按键：`1-4` 选设备，`A/S/K/L` 分别分配 left-inner、
left-outer、right-inner、right-outer，`U` 取消当前分配，`C` 清空，Enter 确认，Esc 取消。

只有在所有 tactile 表面完全卸载且确实要重建 baseline 时才加 `--reset-tactile`。

### 4.5 完整重新分配

推荐同时使用 gripper micro-jog 与 tactile GUI：

```bash
python hardware/DMTAC_gripper/run_gripper.py configure-usb \
  --assign-tactile-gui \
  --tactile-timeout-ms 20 \
  --tactile-max-fps 60 \
  --save
```

若自动 gripper 探测不可用，可使用 `--manual-all-slots` 进行终端 token 输入；它是旧
`--interactive` 的明确名称：

```bash
python hardware/DMTAC_gripper/run_gripper.py configure-usb \
  --manual-all-slots \
  --save
```

完整重分配默认不继承未指定槽位。只有明确希望保留旧槽位时才额外使用
`--merge-config latest`，并在保存前核对打印的完整 assignment。

## 5. 正式入口示例

以下仅展示 DMTAC 参数如何附加到不同入口，**都不是可直接运行的完整命令**。真实启动仍需
从 [真实系统入口](../real/README.md) 选择完整 arm、camera、dataset 或 policy 配方。

| 入口 | DMTAC 参数片段 |
|---|---|
| GUI / headless collection / deploy | `--hand-backend dmtac-gripper --dmtac-config latest --require-dmtac-config` |
| Replay Web GUI | 上述片段之外还必须显式提供 `--end-effectors-enabled` |
| UMI collection | 不追加该片段；由 UMI JSON 顶层 `dmtac_assignment` 引用同一 assignment |

Replay 的 DMTAC 解析规则与其他入口一致，并把 assignment 中的 `port/id/baudrate/timeout`
写入 `ReplayHardwareConfig`；它不初始化 DM-Tac tactile。`deployments.openpi.client.real`
最终把 runtime 参数转给 `teleop_runtime.deploy`，不拥有另一套 DMTAC assignment。

UMI 的完整命令见 [UMI 数采](../real/UMI.md)。UMI side 节点中的空 `gripper_port` 不是第二
事实源。

## 6. Q6、AngleTrigger 与统一设备预检

Q6 是遥操输入，DMTAC gripper 是输出，两个串口 namespace 不可混用：

```bash
python -m hardware.teleop_hand.q6_assignment scan --probe-all-serial
python -m hardware.teleop_hand.q6_assignment configure \
  --left "${LEFT_Q6_BY_ID:?set LEFT_Q6_BY_ID}" \
  --right "${RIGHT_Q6_BY_ID:?set RIGHT_Q6_BY_ID}"
```

AngleTrigger 是 UMI/可选 GUI 的开合输入：

```bash
python -m hardware.angle_trigger --list
python -m hardware.angle_trigger --reassign
python -m hardware.angle_trigger --recalibrate-angles
```

统一设备预检的 `--force-reconfigure` 范围比夹爪更大，会涉及 Q6 与 DMTAC；只修 DMTAC
时应使用 `run_gripper.py configure-usb`，不要把 broad preflight 当成 gripper-only 命令。

## 7. `utils/umi_hand_eye` 是否涉及左右夹爪

结论：涉及左右工具几何，不涉及左右夹爪 USB assignment，也不控制夹爪。

`utils/umi_hand_eye` 按 side 绑定 wrist camera、Vive tracker、URDF camera/arm frame，并将
默认 `tcp_profile=dmgripper` 分别解析到 left/right RM65 DMTAC tool profile。代码中的
`T_gripper_to_base` / `T_camera_to_gripper` 是手眼算法对移动刚体的传统命名；UMI 路径中
移动刚体实际是 tracker/腕部装配，不代表 DMTAC serial backend。

因此：

- 只改变 `dmtac/latest.json` 的 USB 左右映射，不要求重做 UMI hand-eye；
- 改变左右 wrist camera/tracker 绑定、腕部机械装配、URDF 几何或对应侧 tool/TCP profile，
  必须重新标定受影响 side；
- 左右 tool profile 不得互换，即使两个 gripper 的串口参数相同。

## 8. 安全与审计要求

- 扫描或验证前停止所有占用 gripper/tactile 的进程；不要并发打开同一串口。
- `verify-gripper` 是读状态；`open/close/move/set-speed/set-torque/stop` 是真实设备命令。
- gripper micro-jog、AngleTrigger/Q6 重分配、tactile reset 均需要明确硬件范围和现场确认。
- 不要为绕过错误长期使用 `--no-dmtac-validate-usb` 或
  `--dmtac-allow-missing-devices`；它们不修复左右 identity。
- 不建议直接编辑 `latest.json`。CLI 保存会保留 identity、aliases、probe 证据和时间戳快照。
- 生产启动建议使用 `--require-dmtac-config`，防止 assignment 错误后静默变成缺失 side。
