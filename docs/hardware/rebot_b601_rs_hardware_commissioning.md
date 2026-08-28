# B601-RS 有线后端与力控频率 commissioning

本文是 `hardware/rebot_b601_rs_haptic_acceptance.md` 的硬件后端补充。它固定有线
SocketCAN/MotorBridge 的接入边界和频率验收顺序，不替代急停、机械支撑或硬件负责人
批准。离线、被动监听和 MuJoCo 结果都不是 `HARDWARE_PASS`，也不构成使能或运动授权。

## 当前实现边界

`hardware/rebot_b601_rs` 是 AgenticRobot 中的薄适配层：

- 从显式的 `rebotarm_rs.yaml` 读取 `can0`、六个 arm 电机 ID、反馈 ID、型号和 POS_VEL
  默认速度上限；夹爪 ID7 不在第一版 arm backend 内；
- `connect()` 只打开本地 controller 并注册 RS 电机，不请求反馈、不切模式、不使能；
- `poll_feedback()` 默认只调用 `poll_feedback_once()` 消费已有接收队列；MotorBridge 0.4.9
  对 RobStride 的通用 `request_feedback()` 是 no-op。显式传入 `request_feedback=True` 的
  active commissioning loop 仍会调用该兼容入口，然后消费接收队列，并对六个 arm 电机
  串行读取 RobStride `mechPos` 参数 `0x7019`。MotorBridge 的连续状态路径要求另行启用
  active report；本 backend 当前不会隐式改变该设备配置。两条路径对缺失/错误
  ID/非零 vendor status/非有限值都生成 `invalid/stale` 状态，绝不补成零；active position
  fallback 可能没有 velocity/torque 字段，力控调用必须单独检查对应字段存在；
- `prepare_position_velocity_mode()` 对照 pinned reBotArm 的
  `JointGroup.mode_pos_vel()`，在进入发送循环前按 `0x7017`（limit_spd）、`0x701F`
  （速度环 Kp）、`0x7020`（速度环 Ki）、`0x701E`（位置环 Kp）的顺序写入 YAML 中的
  POS_VEL 参数，并保留上游的短暂 settling delay 后再调用 `ensure_mode(POS_VEL)`；若任一
  参数写入或模式切换失败，receipt 为 incomplete，probe 不会继续使能。POS_VEL 参数写入只
  发生在一次性 setup，不在高频循环内重复；本次 probe 的有效速度上限同时写入 `0x7017`，
  因而不会只限制软件命令而留下更宽的驱动器内部限幅；
  YAML 中的 `pos_ki` 仅按上游 `JointCfg` 保留以保持配置语义，不自行映射到 RobStride
  未被 reBotArm `_write_pv_params` 使用的寄存器；
- `prepare_mit_mode()` 对照同一 upstream `JointGroup.mode_mit()`，只在高频循环外逐关节调用
  `ensure_mode(MIT)` 并保留相同 settling delay；它不使能电机，也不选择运行增益。当前
  backend 的 `send_mit()` 参数顺序与 upstream 一致，均为
  `(position, velocity, kp, kd, feedforward_tau)`；
- MotorBridge 0.4.9 的 `MotorState` 没有源时间戳或接收序号，所以反馈状态明确标为
  `freshness_verified=false`、`poll_observation_unverified`，报告的 `vel/torq` 也保留
  `vendor_reported_unverified` 语义；
- 位置、MIT、模式设置、使能和失能均要求构造 backend 时显式 `allow_motion=True`，每次
  命令返回逐关节 receipt；部分发送不会被报告成六关节成功；
- `safe_stop()` 是单独的、可在未授权启动运动的 backend 上调用的逐关节失能入口；它仍然
  需要现场机械支撑和物理急停配合，`close()` 不隐式失能；
- 低增益 probe 的 active set 默认是 `--joint` 指定的一个目标关节加
  `joint2`/`joint3` 支撑关节；其余 arm 关节不会被使能。异常路径只有在
  last-safe 快照通过新鲜度、限位、速度和小 rebase 检查后才锁存 J2/J3，否则直接全部
  arm `safe_stop`。这个保护保持不包含 ID7 夹爪；
- 任何 motion-capable caller 都必须先从同一 pinned URDF 提供六关节 position limits；本轮
  位置频率 probe 会在 enable 前从 `--rebot-source-root`（默认由 config 路径推导）解析并
  锁定这些 limits，解析失败则不允许进入位置命令循环；
- `feedback_timeout_sec`/`watchdog_timeout_sec` 目前只是显式配置声明；因为 MotorBridge
  缓存没有源时间戳，适配器不会伪造 freshness，也不会后台自动 watchdog。H2 仍须由后续
  经评审的控制循环完成并取得 stop receipt；
- POS_VEL 参数/模式设置必须在循环外完成，高频循环只发送已审查的参考。`close()` 不隐式
  去使能，避免把“释放负载”的行为藏在析构路径中。
- `position-tx-disabled` 先取得一次完整 warmup snapshot，再对所选关节取得成功的
  `safe_stop` 收据，随后只测保持该参考的 POS_VEL host-to-vendor dispatch clock；该模式
  拒绝 `--enable`，循环内不读取反馈，退出时再次 `safe_stop`。它用于把约 50 ms 的六关节
  同步 `mechPos` 参数读取与 TX 上限分开，仍会发送位置命令帧，必须由操作者运行；它不证明
  电机已运动、线上 CAN 帧率、闭环状态 freshness 或 enabled control frequency。
- `mit-tx-disabled` 在 warmup 后先失能，切到 MIT mode 后再次失能，再以固定 warmup q
  发送 `qvel=kp=kd=tau=0`；退出时第三次 `safe_stop`。该模式只测 MotorBridge MIT 调用
  上限，不发送驱动力矩、不调用 `enable()`，也不证明实际 CAN 帧、MIT 扭矩、重力补偿或
  阻尼控制有效；
- `gravity-offline` 复用 `B601PinocchioModel.arm_generalized_gravity()` 与
  `compute_b601_mit_torque()` 的 zero-PD 路径，验证固定合法 q 下 `requested_tau == g(q)`
  并测量 compute tick。它不构造硬件 backend，不打开 CAN。

数学、重力和 haptic 组合仍由 `robot_motion.rebot_b601_rs` 负责，仿真 effort backend
仍由 `sim.mujoco.rebot_b601_rs` 负责；本模块不复制 Pinocchio 或 MuJoCo 实现。

## 频率指标契约

频率不能从一条混合 CAN 计数推断。`hardware.rebot_b601_rs.frequency.FrequencyLedger`
必须分别记录：

| event | 含义 |
| --- | --- |
| `feedback` | 收到/消费的反馈观察 |
| `compute` | 本地计算完成、即将发送的控制 tick |
| `tx` | vendor send 调用返回的发送完成观察 |
| `haptic` | haptic wrench/torque 计算完成 |

每个 event 独立输出 sample count、observed Hz、p50/p95/p99/max interval 和 deadline
misses。被动 CAN 探针还会按 arbitration ID、RobStride communication type 分别统计 type-2
`operation_status` 与 type-24 `active_report` 候选的 device/host ID 和接收间隔，但不会解码
payload，也不会声称 command 或 haptic 频率。MotorBridge 以 type 24 切换 active report，
不同 firmware 的周期流实际可能表现为 type 2 或 type 24；两者必须分别观察。当前
MotorBridge 缺少源时间戳/序号，故被动或 backend 反馈频率只能作为接收时间证据；即使观察
到周期流，`freshness_verified` 仍为 false。要关闭 freshness 缺口，需要 vendor 层为实际
firmware 的周期帧提供解码、接收序号/时间戳并保持同一时钟域，而不是在应用层复制一套状态
payload 解码器。

active probe 还会输出独立的 `execution_status` 和 `acceptance_status`。默认频率门要求每个
required event 的 observed rate 不低于目标的 95%，且超过 `2 * period` 的 interval 不多于
1%；阈值会写入 JSON，也可通过 `--minimum-rate-ratio` 和
`--maximum-deadline-miss-ratio` 显式覆盖。设备调用和 cleanup 完成、但频率未达标时，顶层
`status` 是 `FAIL` 且进程返回 1；前置条件或调用异常返回 2。不能再把旧结果中的
`status: OK` 解读为 125 Hz 通过。

## 阶段与停止条件

1. **P0 离线契约**：运行 mock backend、配置解析和 `FrequencyLedger` 测试；确认 import
   不打开 CAN。
2. **P1 被动反馈**：电机保持当前安全状态，只运行 read-only SocketCAN 监听，核对 ID、
   总线状态、每 ID 帧率、错误/丢包和时间间隔。任何第二个控制 owner、未知发送源、帧
   解释不清或急停不可用都停止。
3. **P2 backend 只读**：使用明确 RS YAML 构造 backend，确认六个 arm ID 顺序、位置符号、
   空反馈/错误 status 的 fail-closed 行为；默认 `poll_feedback()` 不调用
   `request_feedback`、mode、enable 或 send。
4. **P2.5 失能 TX 上限**：仅由操作者运行 `position-tx-disabled`。确认 preflight/final
   `safe_stop` 均完整、`motors_enabled_by_probe=false`，再检查 `compute/tx` 是否通过；该
   阶段不要求 warmup `feedback` 达到目标，也不构成运动或闭环通过。
5. **P2.6 离线重力与失能 MIT TX**：先离线运行 `gravity-offline`，再仅由操作者运行
   `mit-tx-disabled`。后者必须同时具有 preflight、post-mode 和 final 三份完整
   `safe_stop` receipt、`motors_enabled_by_probe=false`，且命令语义必须保持
   `qvel=kp=kd=tau=0`。这仍不构成已使能 MIT/重力/阻尼通过。
6. **P3 高频 POS_VEL**：仅由操作者运行下方 enabled guarded command。先 125 Hz，再 250 Hz，
   最后才评估 500 Hz；每档均须机械支撑、物理急停、唯一控制 owner、`--auto-disable`，
   并保存 `feedback/compute/tx` 三个 event 的独立统计。只保持启动时反馈位置，不引入
   trajectory 或 haptic overlay。
7. **P4 单关节 MIT/重力阻尼**：由硬件负责人另行批准，先单关节、低幅、当前 q 参考和零
   overlay；确认重力方向、阻尼耗散、最终 torque/slew limit 和 stop receipt 后才扩大到
   多关节。P3 的位置频率通过不等于 P4 的力控通过。
8. **P5 haptic 频率**：接入 `compute_b601_mit_torque()` 与 `J.T @ wrench` 后，单独记录
   `haptic`，并用低幅 virtual spring/wall 或已审查的 synthetic follower wrench 验证
   frame、功率不变量、反馈延迟和限幅。只有 `feedback/compute/tx/haptic` 全部可对齐，
   才能讨论 500 Hz 目标；不能用 aggregate bus rate 代替。

任一阶段出现方向反转、跳变/振荡、反馈陈旧或非有限、部分发送、多个控制 owner、错误
frame、watchdog 超时、日志/commit/config 身份缺失，立即停止该阶段并保留原始日志；不得
通过放宽 watchdog 或重复运行掩盖原因。

## 无运动检查

以下命令不打开设备，适合开发机和 CI：

```bash
bash scripts/run_pytest.sh -q tests/test_rebot_b601_hardware_backend.py
python scripts/hardware/rebot_b601_rs_frequency_probe.py \
  --mode offline --duration 0.2 --target-hz 500
python scripts/hardware/rebot_b601_rs_frequency_probe.py \
  --mode gravity-offline \
  --rebot-source-root /path/to/reBotArm_control_py \
  --duration 0.2 \
  --target-hz 500
```

被动硬件检查只接收，不发送 CAN 帧，也不构造 MotorBridge：

```bash
python scripts/hardware/rebot_b601_rs_frequency_probe.py \
  --mode read-only --channel can0 --duration 3 \
  --output /tmp/b601-rs-commissioning/b601_passive_feedback.json
```

运行前仍须确认 CAN 上没有其他控制 owner；监听输出是 `NOT_RUN`/只读证据，不是硬件运动
验收。输出建议写入 Task/Run 外的操作者日志目录，不要写回源 worktree。重点检查：

- `frames_by_robstride_communication_type` 中 type 2/type 24 的实际分布；
- 对应的 `*_frames_by_device_id` 是否覆盖预期 arm ID 1--6（ID7 是夹爪）；
- `robstride_short_operation_status_frame_count` 和
  `robstride_short_active_report_frame_count` 是否为零；
- 对应 `*_per_device_feedback` 的 `observed_hz`，以及 host ID 是否与配置的 `0xFD` 一致。

没有 type 2/type 24 周期候选只表示监听期间没有观察到对应帧，可能是 active report 尚未
启用或总线上没有生产者，不能推断硬件不支持该路径。任一类型计数非零也只建立 host 接收
节拍；在检查周期性、ID 方向和 payload 契约前不能区分一次性切换命令与连续设备上报，更
不能证明源端新鲜度或 MotorBridge cache 新鲜度。

## 操作者手动失能 TX 频率入口（本轮不执行）

先运行这一档来区分 MotorBridge 发送调用上限与同步反馈读取上限。该命令不会调用
`enable()`，但会发送 POS_VEL 命令帧，因此仍要求现场机械支撑、急停、唯一 owner 和人工
执行。默认只测单关节，先从 125 Hz 开始：

```bash
python scripts/hardware/rebot_b601_rs_frequency_probe.py \
  --mode position-tx-disabled \
  --config "$REBOT_RS_CONFIG" \
  --rebot-source-root /path/to/reBotArm_control_py \
  --target-hz 125 \
  --duration 3 \
  --velocity-limit-rad-s 0.1 \
  --joint joint1 \
  --allow-motion \
  --confirmation B601_RS_DISABLED_TX_FREQUENCY \
  --auto-disable \
  --output /path/to/operator-log/b601_position_tx_disabled_125hz.json
```

预期结构条件是 `preflight_disable_receipt.completed=true`、
`cleanup_receipt.completed=true`、`motors_enabled_by_probe=false`，然后才读取
`acceptance.events.compute/tx`。这里的 PASS 仅表示 host-to-vendor 调用时钟达标。

## 操作者手动失能 MIT TX 频率入口（本轮不执行）

该入口不调用 `enable()`，并在 MIT mode 前后都确认所选电机已失能；循环中仅发送
`qvel=kp=kd=tau=0` 的 MIT frame。即使输出 PASS，也只能说明 host-to-vendor MIT 调用
时钟达到目标，不能解释为电机侧力控、重力补偿或阻尼已验收：

```bash
python scripts/hardware/rebot_b601_rs_frequency_probe.py \
  --mode mit-tx-disabled \
  --config "$REBOT_RS_CONFIG" \
  --rebot-source-root /path/to/reBotArm_control_py \
  --target-hz 125 \
  --duration 3 \
  --joint joint1 \
  --allow-motion \
  --confirmation B601_RS_DISABLED_MIT_TX_FREQUENCY \
  --auto-disable \
  --output /path/to/operator-log/b601_mit_tx_disabled_125hz.json
```

只有 `preflight_disable_receipt.completed=true`、
`post_mode_disable_receipt.completed=true`、`cleanup_receipt.completed=true`、
`motors_enabled_by_probe=false` 和 `mit_command_semantics.final_commanded_effort_nm` 全零时，
才检查 `acceptance.events.compute/tx`。

## P4.1 低增益软件 torque-owner probe（操作者手动运行）

这是 P4 单关节 MIT/重力阻尼验证的 guarded 入口。它仍然是 motion-capable
命令，只能由现场操作者运行；自动化测试不会打开 CAN、使能电机或发送 MIT 帧。
探针默认只把目标关节和强制支撑关节 `joint2`、`joint3` 放入 active set，明确排除
ID7 夹爪以及未选中的 arm 关节；因此第一档应从 `--joint joint1` 开始，且必须为
`joint2`/`joint3` 和未选中的腕部关节提供机械支撑。`hold` 从使能后的当前姿态开始，
不会跳到预设绝对姿态，`slow-step` 只在 hold 通过后以相对、有界步长运行。

### 异常时的 protective hold（锁存，不是普通回零）

这条路径借鉴 reBot 官方的故障处置边界：通信异常不尝试回零而立即失能；普通退出才
允许平滑回零。我们的特殊需求是在满足更严格的 admission 条件时，把 `joint2`/`joint3`
锁在故障前最后一个安全姿态，同时把 `joint1`/`joint4`/`joint5`/`joint6` 失能。
夹爪 ID7 不在当前 backend 的注册范围内，不能把下面的“全臂失能”理解成夹爪也已失能。

状态转换固定为：

```text
ACTIVE
  ├─ 正常结束 ──> active-set 短暂保持 ──> 全部 arm safe_stop
  ├─ Ctrl+C/异常且 admission 通过
  │                 └─> PROTECTIVE_HOLD（只发 joint2/joint3）
  │                       ├─ SIGUSR1 ──> 全部 arm safe_stop
  │                       └─ hold dispatch/非保持失能失败
  │                                      └─> 全部 arm safe_stop
  └─ 无法证明当前 q 新鲜/合法 ─────────────> 全部 arm safe_stop
```

进入 `PROTECTIVE_HOLD` 前，程序必须同时证明：

1. `last_safe_q` 是六关节完整、有限、在 pinned URDF position limits 内的快照；
2. 快照年龄是本进程单调时钟测得且不超过 `--fault-hold-max-age-sec`（默认
   `0.15 s`）；MotorBridge 0.4.9 没有设备源时间戳，所以 JSON 仍明确标记
   `freshness_verified=false`，不能把它宣传成 vendor freshness；
3. 快照的六关节有限差分速度均不超过 `--fault-hold-max-velocity-rad-s`（默认
   `0.02 rad/s`），保持参考的 `qvel` 永远是全零；
4. 如果故障样本仍有完整 q，则故障样本本身也必须在 limits 内、速度不超过同一阈值，
   且 joint2/joint3 相对 `last_safe_q` 的 rebase 不超过
   `--fault-hold-max-rebase-rad`（默认 `0.05 rad`）。超出即禁止保持；
5. 只有通过上述检查后才会发送第一帧；发送的是固定 `last_safe_q`、零 qvel、vendor
   `kp=kd=0`，并继续经过软件 torque absolute limit 和 torque slew limit。q 绝不由
   错误/陈旧样本补零，也不使用最后一次 command 代替最后一次合法状态。

第一帧保持成功后，程序先对非保持 arm 关节发逐关节 `safe_stop`；该 receipt 不完整时
立即回退到全部 arm `safe_stop`，不继续假设“部分失能”已经成立。之后每一个保持周期都会
重新读取完整反馈：反馈不完整/失效、主机观测不再推进、任一关节超出 limits、有限差分速度
超过阈值，或 joint2/joint3 偏离固定参考超过 `max-rebase`，都会停止继续发保持帧并回退到
全部 arm `safe_stop`。通过监视时，固定参考仍是故障前的 `last_safe_q`；当前测量 q 只用于
计算受限的重力/阻尼力矩，绝不用于更新参考。保持没有时间自动释放，也不会因为 EOF、普通
cleanup 或 `close()` 释放；操作者必须在另一终端显式发送 `SIGUSR1`，并确认返回的
`full_disable_receipt.completed=true`。如果没有可证明的新鲜安全 q，程序直接采用官方通信
故障同类的“无回零立即失能”fallback。

释放命令示例（不会让交接 shell 退出）：

```bash
PID=<probe 输出的 pid>
if kill -USR1 "$PID"; then
  printf 'full-disable requested: pid=%s\n' "$PID"
else
  printf 'ERROR: full-disable request failed: pid=%s\n' "$PID"
fi
printf '%s\n' 'shell remains active'
```

本入口固定单一 torque owner：软件按
`tau = gravity_scale*g(q) + Kp*(q_ref-q) + Kd*(dq_ref-dq)` 组合重力前馈、低增益
位置项和基于新鲜 `mechPos (0x7019)` 的有限差分阻尼，然后向 upstream
MotorBridge 发送 `kp=kd=0` 的 MIT 帧。最终总力矩先做 URDF effort ceiling 的
`torque-limit-scale`（默认 0.25）绝对限幅，再做 torque slew 限制；超出速度、
位置误差、非有限差分、部分发送或 cleanup 不完整会立即结束本次运行。由于当前
`mechPos` 是同步参数读取且 MotorBridge 没有源序号/时间戳，本探针的默认 20 Hz
只是低速 commissioning 节拍，不是高频 haptic loop，也不声明 freshness、力传感器
或真实重力参数已验收。Pinocchio 重力模型使用中性辅助夹爪坐标，实际夹爪负载和
摩擦仍须后续标定。

运行前必须确认：机械臂处于可支撑的 clearance/L 姿态（建议 j2≈0.7、j3≈1.1），
操作者手扶或有机械支撑，物理急停可立即触达，`can0` 为 1 Mbit/s，现场没有第二个
CAN 控制 owner，且本次只批准一个关节的低幅验证。不要在夹爪接触桌面或其他物体时
运行；`safe_stop` 失能后负载可能下落，停止过程中仍需保持支撑。下面命令使用
状态分支和显式 status 打印，不使用会让交接终端提前退出的 `set -euo pipefail`；
命令由操作者在目标 worktree 中运行，我们不在自动流程中代运行。版本化文档保留
可移植变量；实际交接时必须替换为当前 Task 的绝对路径，并先核对固定的 commit/tree：

```bash
WT=/path/to/isolated/AgenticRobot-worktree
PY=/path/to/dexteleop310/bin/python3.10
CFG=/path/to/reBotArm_control_py/config/rebotarm_rs.yaml
SRC=/path/to/reBotArm_control_py
OUT_DIR=/tmp/b601-rs-commissioning

mkdir -p "$OUT_DIR"
if cd "$WT"; then
  if PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/hardware/rebot_b601_rs_low_gain_probe.py \
      --mode hold \
      --config "$CFG" \
      --rebot-source-root "$SRC" \
      --duration-sec 8 \
      --warmup-sec 2 \
      --freeze-sec 1 \
      --rate-hz 20 \
      --joint joint1 \
      --kp 0.5 \
      --kd 0.5 \
      --gravity-scale 1.0 \
      --torque-limit-scale 0.25 \
      --torque-rise-time-sec 1.0 \
      --velocity-abort-rad-s 0.10 \
      --max-error-rad 0.15 \
      --allow-motion \
      --auto-disable \
      --confirmation B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING \
      --fault-hold-max-age-sec 0.15 \
      --fault-hold-max-velocity-rad-s 0.02 \
      --fault-hold-max-rebase-rad 0.05 \
      --output "$OUT_DIR/b601_low_gain_hold_joint1.json"; then
    STATUS=0
  else
    STATUS=$?
  fi
  printf 'probe exit code: %s\n' "$STATUS"
else
  printf '%s\n' 'ERROR: worktree not found'
fi
printf '%s\n' 'shell remains active'
```

第一档只验收 `hold`：观察 j1 是否按当前状态工作、j2/j3 是否作为支撑保持、重力方向
是否合理、阻尼是否耗散而非振荡，
并在 JSON 中确认 `status=OK`、`acceptance_status=PASS`、
`cleanup_receipt.completed=true`、`cleanup_errors=[]`、`stop_reason=null`，以及
`frequency_acceptance.feedback/compute/tx.*.passed=true`。输出中的
`hardware_acceptance` 必须仍为 `NOT_CLAIMED`；这是一份低增益软件路径和现场观察
记录，不是 `HARDWARE_PASS`。

只有 hold 通过且机械支撑未改变，才可单独运行一档相对慢步；每次只批准一个命令，
例如 j1 正方向 0.05 rad、0.02 rad/s：

```bash
# 复用上面的 WT/PY/CFG/SRC/OUT_DIR 变量；先确认上一档已停止且 cleanup 完整
if cd "$WT"; then
  if PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/hardware/rebot_b601_rs_low_gain_probe.py \
      --mode slow-step \
      --config "$CFG" \
      --rebot-source-root "$SRC" \
      --duration-sec 8 \
      --warmup-sec 2 \
      --freeze-sec 1 \
      --rate-hz 20 \
      --joint joint1 \
      --step-direction 1 \
      --step-rad 0.05 \
      --step-speed-rad-s 0.02 \
      --kp 0.5 \
      --kd 0.5 \
      --gravity-scale 1.0 \
      --torque-limit-scale 0.25 \
      --torque-rise-time-sec 1.0 \
      --velocity-abort-rad-s 0.10 \
      --max-error-rad 0.15 \
      --allow-motion \
      --auto-disable \
      --confirmation B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING \
      --fault-hold-max-age-sec 0.15 \
      --fault-hold-max-velocity-rad-s 0.02 \
      --fault-hold-max-rebase-rad 0.05 \
      --output "$OUT_DIR/b601_low_gain_slow_step_joint1.json"; then
    STATUS=0
  else
    STATUS=$?
  fi
  printf 'probe exit code: %s\n' "$STATUS"
else
  printf '%s\n' 'ERROR: worktree not found'
fi
printf '%s\n' 'shell remains active'
```

任一方向出现反向、跳变、振荡、超速、超误差、非有限反馈或未完成 stop receipt，
立即按急停/支撑规程停止并保留 JSON；不得通过提高 `max-error`、放宽速度/力矩限幅、
降低目标频率或重复运行来掩盖原因。

## 已使能力控仍然阻塞的契约

当前实测的六关节同步 `mechPos (0x7019)` 反馈约为 20 Hz，而且 MotorBridge 缓存没有源
序号/时间戳。MotorBridge 参数表明确 `0x701A` 是滤波电流 `iqf`，真正的 `mechVel` 是
`0x701B`；因此既有把 `0x701A` 当速度得到的量纲/符号观察无效，不能据此评价速度反馈。
在 `0x701B` 的真实设备量纲、符号和 freshness 经独立验证前，控制侧仍应使用新鲜位置的
有界差分速度。当前不能用约 20 Hz 的同步参数读取直接闭合 125 Hz 重力/阻尼控制。

另外，`compute_b601_mit_torque()` 已经在软件中组合 position、damping、gravity 和 overlay；
若把该总力矩作为 `tau` 发送，同时给驱动器非零 `kp/kd`，会重复计算 PD。反过来，若使用
upstream 的驱动器 PD 加 `g(q)` 前馈，当前 backend 仅能限制前馈 `tau`，不能证明驱动器
内部 PD 与前馈相加后的最终总力矩满足项目的 final-output bound。进入 P4 前必须先确定并
实现唯一的一种 torque ownership，并取得高频、可验证 freshness 的 q 与差分 dq，再补齐
watchdog、最终 torque/slew 限制和单关节安全验收。本文不提供已使能 MIT 命令。

## 操作者手动 enabled 位置频率入口（本轮不执行）

这是 P3 的 motion-capable handoff，必须由操作者在同一有线主机上运行。先把
`REBOT_RS_CONFIG` 指向已审查的 `rebotarm_rs.yaml`，并确认现场支撑/急停/唯一 owner；命令
会在每次 active poll 前对六个 arm 电机请求反馈并读取 `mechPos (0x7019)`，设置 POS_VEL、
使能所选 arm 电机、发送保持当前位置的命令，并在退出时显式失能。反馈/参数读取不是运动
命令；由于 MotorBridge 缺少源序号，输出仍会标记 feedback freshness 未验证。

```bash
export REBOT_RS_CONFIG=/path/to/reBotArm_control_py/config/rebotarm_rs.yaml
python scripts/hardware/rebot_b601_rs_frequency_probe.py \
  --mode position \
  --config "$REBOT_RS_CONFIG" \
  --target-hz 125 \
  --duration 10 \
  --velocity-limit-rad-s 0.1 \
  --joint joint1 \
  --allow-motion \
  --confirmation B601_RS_POSITION_FREQUENCY \
  --enable \
  --auto-disable \
  --output /path/to/operator-log/b601_position_125hz.json
```

`--joint` 可用逗号分隔的六个 arm 名称；脚本默认从 config 的上两级目录寻找 pinned URDF，
也可显式传入 `--rebot-source-root`。没有通过上一档频率和硬件负责人复核，不得直接提高到
250/500 Hz。`--velocity-limit-rad-s` 是本次 POS_VEL 参考的保守速度上限（默认
`0.1 rad/s`），最终还会取不高于 YAML 中每关节 `vlim` 的值；它不是频率参数。该命令的
`motion_commands_sent=true` 仅表示发送尝试发生，不能
替代现场运动观察或 `HARDWARE_PASS` 收据。

该 enabled 命令当前有意保留“每 tick 完整同步反馈”的严格组合语义。若它与此次实测一样
稳定在约 20 Hz，结论是 combined feedback+compute+TX gate 为 FAIL，而不是调整
`--target-hz` 或把 TX 声称为 125 Hz。只有失能 TX 上限先独立通过，且后续获得可验证 freshness
的异步/解码反馈路径后，才应设计真正的高频 enabled/haptic 双时钟循环；不得在没有
MotorBridge 线程安全契约时并发调用同一个 controller。

## 通过记录最小字段

保存 AgenticRobot commit/tree、reBotArm gitlink/config SHA-256、motor model/ID、CAN
bitrate、操作者/观察者、唯一 owner、每 event 的统计、每关节状态/命令/错误、限幅、
watchdog/fault transition、物理急停和 cleanup 结果。每个阶段标记为 `NOT_RUN`、`FAIL`
或 `HARDWARE_PASS`；`OFFLINE_PASS` 只能描述 P0，不得升级为真机结论。
