# B601-RS 有线后端与力控频率 commissioning

本文是 `hardware/rebot_b601_rs_haptic_acceptance.md` 的硬件后端补充。它固定有线
SocketCAN/MotorBridge 的接入边界和频率验收顺序，不替代急停、机械支撑或硬件负责人
批准。离线、被动监听和 MuJoCo 结果都不是 `HARDWARE_PASS`，也不构成使能或运动授权。

## 当前实现边界

`hardware/rebot_b601_rs` 是 AgenticRobot 中的薄适配层：

- 从显式的 `rebotarm_rs.yaml` 读取 `can0`、六个 arm 电机 ID、反馈 ID、型号和 POS_VEL
  默认速度上限；夹爪 ID7 不在第一版 arm backend 内；
- `connect()` 只打开本地 controller 并注册 RS 电机，不请求反馈、不切模式、不使能；
- `poll_feedback()` 默认只调用 `poll_feedback_once()` 消费已有接收队列；显式传入
  `request_feedback=True` 的 active commissioning loop 会先对六个 arm 电机各发送一次
  `request_feedback()`，再消费接收队列，并读取 RobStride `mechPos` 参数 `0x7019`（该
  firmware 的紧凑 type-0x18 位置流不由 MotorBridge 0.4.9 解码）。两条路径对缺失/错误
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
- MotorBridge 0.4.9 的 `MotorState` 没有源时间戳或接收序号，所以反馈状态明确标为
  `freshness_verified=false`、`poll_observation_unverified`，报告的 `vel/torq` 也保留
  `vendor_reported_unverified` 语义；
- 位置、MIT、模式设置、使能和失能均要求构造 backend 时显式 `allow_motion=True`，每次
  命令返回逐关节 receipt；部分发送不会被报告成六关节成功；
- `safe_stop()` 是单独的、可在未授权启动运动的 backend 上调用的逐关节失能入口；它仍然
  需要现场机械支撑和物理急停配合，`close()` 不隐式失能；
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
misses。被动 CAN 探针还会按 arbitration ID 报告帧数和间隔，但不会声称 command 或 haptic
频率。当前 MotorBridge 缺少源时间戳/序号，故被动或 backend 反馈频率只能作为接收时间
证据；要关闭 freshness 缺口，需要后续 raw SocketCAN 解码/接口扩展并保持同一时钟域。

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
5. **P3 高频 POS_VEL**：仅由操作者运行下方 enabled guarded command。先 125 Hz，再 250 Hz，
   最后才评估 500 Hz；每档均须机械支撑、物理急停、唯一控制 owner、`--auto-disable`，
   并保存 `feedback/compute/tx` 三个 event 的独立统计。只保持启动时反馈位置，不引入
   trajectory 或 haptic overlay。
6. **P4 单关节 MIT/重力阻尼**：由硬件负责人另行批准，先单关节、低幅、当前 q 参考和零
   overlay；确认重力方向、阻尼耗散、最终 torque/slew limit 和 stop receipt 后才扩大到
   多关节。P3 的位置频率通过不等于 P4 的力控通过。
7. **P5 haptic 频率**：接入 `compute_b601_mit_torque()` 与 `J.T @ wrench` 后，单独记录
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
```

被动硬件检查只接收，不发送 CAN 帧，也不构造 MotorBridge：

```bash
python scripts/hardware/rebot_b601_rs_frequency_probe.py \
  --mode read-only --channel can0 --duration 3
```

运行前仍须确认 CAN 上没有其他控制 owner；监听输出是 `NOT_RUN`/只读证据，不是硬件运动
验收。输出建议写入 Task/Run 外的操作者日志目录，不要写回源 worktree。

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
