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
  active report；backend 仅在独立 `allow_active_reporting=True` 授权下提供显式、可回滚的
  `set_active_reporting()`，不会由 `connect()`、`poll_feedback()` 或运动接口隐式改变。两条路径对缺失/错误
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
- backend 可显式读写 RobStride `limit_torque (0x700B)`，但只提供临时、易失事务：当前
  session 必须先取得所选关节成功的 `disable/safe_stop` receipt，随后才会读取全部原值、
  写入新值并逐项回读；任一失败会回滚全部已触碰关节。原值未全部恢复并回读前，`close()`
  会拒绝释放 backend。该接口不写 flash、不替代软件总力矩/slew envelope，也不授权使能；
- MotorBridge 0.4.9 的 `MotorState` 没有源时间戳或接收序号，所以反馈状态明确标为
  `freshness_verified=false`、`poll_observation_unverified`，报告的 `vel/torq` 也保留
  `vendor_reported_unverified` 语义；
- 位置、MIT、模式设置、使能和失能均要求构造 backend 时显式 `allow_motion=True`，每次
  命令返回逐关节 receipt；部分发送不会被报告成六关节成功；
- `safe_stop()` 是单独的、可在未授权启动运动的 backend 上调用的逐关节失能入口；它仍然
  需要现场机械支撑和物理急停配合，`close()` 不隐式失能；
- active-report 使用与 `allow_motion` 分离的授权，且不切模式、不使能、不发送 POS_VEL/MIT
  参考。多关节启用部分失败时会对所有已尝试关节反向发送关闭请求；任何未确认关闭的关节会
  保持锁存，`close()` 拒绝释放 backend，避免把不确定的设备上报状态隐藏在析构中；
- 低增益 probe 的所有正常阻抗/重力模式都固定使用 joint1..joint6；`--joint` 只决定
  slow-step 目标或单关节增益覆盖，不缩小使能和 MIT dispatch 集合。异常路径只有在
  last-safe 快照通过新鲜度、限位、速度和小 rebase 检查后才锁存六个 arm 关节，否则直接全部
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
- `feedback-stability` 是本轮低增益试验前的只读噪声标定入口：它用
  `allow_motion=False` 构造同一个 backend，主动请求并读取六个 arm 的 `mechPos`，按与
  active probe 相同的时间戳，对通过反馈边界归一化后的 q 计算有限差分速度，输出每关节
  步进/速度的 p50/p95/p99/max 以及 `0.02 rad/s` stationarity gate 统计；每个样本仍保留
  原始 q 供复核。它不切模式、不使能、不失能、不发送
  MIT/POS_VEL 控制帧，也不改变运行前已经存在的电机状态；`CAPTURED` 只表示采样完成，
  不能解读为力控或硬件通过。该结果用于判断 `last-safe` 速度门是否被反馈量化噪声触发，
  不允许据此直接放宽保护阈值。
- 反馈进入限位/速度/阻抗计算前，J2 的已观测零位/启动边界
  `-1e-2 <= q < 0 rad`、J3 的已观测边界 `-3e-3 <= q < 0 rad` 会被表示为合法的
  `0 rad`。J4 的运行坐标额定边界按权威 `+/-pi/2` 表示（源 URDF 的
  `-1.57/1.57` 仅是两位小数的记录），并额外接纳
  `-pi/2-0.001 <= q < -pi/2 rad`，投影到精确的 `-pi/2`。这些都是反馈状态的
  有界投影，不修改源 URDF 或真实机械硬限位；反馈窗口本身不改变出站参考或
  torque/slew 包络。超过各自窗口的读数仍 fail-closed。JSON 样本同时保留 `raw_qpos_rad`、归一化后的
  `qpos_rad` 和 `boundary_normalized_joint_names`，便于复核实际读数。
- J2 的 `-1e-2 rad` 是只读接纳下限，不是常规工作范围：两次零位只读捕获曾观察到
  `-4.943/-4.517 mrad`，2026-09-08 使用 host-id 限定反馈的启动锁/重力交接又观察到
  连贯的 `-7.627 mrad` 边界瞬态，因此仅对 J2 保留到 `-10 mrad` 的整值余量。J3/J4
  窗口和全部出站命令限位保持不变；比该下限更负的 J2 反馈仍 fail-closed。
- active low-gain probe 用相邻 normalized q 与精确 host receive `dt` 计算 qvel，再用连续
  qvel 与 `dt` 计算 qacc。首个控制样本没有前一 qvel 时不伪造 qacc；`dt <= 0` 和非有限值
  立即拒绝。有限但 qvel 或 qacc 越限的第一帧在任何 MIT dispatch 前隔离：不更新
  `previous_q`、feedback timestamp、qvel history、`final_q`、fault snapshot 或 `last-safe`，
  电机继续保持上一份完整命令和已配置的 driver Kd。下一帧以最后可信位置和跨过异常帧的
  总 `dt` 重新计算；恢复正常则继续 ACTIVE/标定，连续第二帧仍越限才进入 protective hold。
  这不是把 `12.6 rad/s` 当成合法拖动，也不改变 CLI 配置的速度/加速度阈值。JSON 的
  `motion_guard.rejected_sample` 独立保存 offending joint、raw/normalized q、previous/current
  qvel、qacc、dt、seq/timestamp，并标记 `command_dispatched=false`、
  `eligible_for_last_safe=false`；确认策略和隔离/恢复/连续中止计数也一并保存。默认
  `--acceleration-abort-rad-s2 4.0` 只是待真机整定的 commissioning seed，不是已认证上限。
  位置限位和 motion guard 都先于更新 `final_q`、当前故障快照和 `last-safe`；被拒绝的
  反馈只进入 evidence，不会被用于重力、阻抗或保护保持计算。
- 反馈接纳现在在 wired backend 入口再次执行：`feedback_position_limits_rad` 是只读的
  接纳包络，默认回退到严格的 `position_limits_rad`，由 probe 显式传入仅扩展 J2/J3/J4
  已批准的边界窗口。这样即使 `mechPos(0x7019)` 返回一个有限但物理上不可能的巨大数值，
  backend 也会将该关节标为 `valid=false, stale=true`，并在错误中保留关节名、motor/feedback
  ID、参数 ID、原始值、可用时的 IEEE-754 单精度位模式和接纳区间；坏帧不会被任何 backend 消费者误当成完整状态。出站 MIT/
  POS_VEL 仍只使用严格 URDF 包络，读侧窗口不能放宽命令限位。active probe 会把每关节
  backend 诊断写入 `motion_guard.rejected_sample.backend_joint_diagnostics`，随后复用上一份
  已验证的六轴重力缓存进入 `PROTECTIVE_HOLD`；只有已配置 driver Kd 时，反馈不可信的
  降级阶段才仍有电机本地阻尼。终端会用完整分隔线分别显示 `PROTECTIVE HOLD DEGRADED`、
  `LATCHED` 和 `RECOVERED`，并明确当前六轴仍使能、控制能力和再次 Ctrl-C/SIGUSR1 的
  全失能含义。只有完整缓存发送也失败或操作者第二次
  Ctrl-C/SIGUSR1 才允许全臂失能。
- 若安装的 MotorBridge 提供 `robstride_get_param_f32_host_id()`，backend 会优先使用带
  `feedback_id/host_id` 的参数读取 ABI；旧版本没有该入口时才回退到未限定 host 的兼容调用。
  这缩小了共享 CAN 队列中把其他设备的参数响应当成当前关节的机会，但仍不把返回值视为已验证
  的设备时间戳/序列；读侧数值包络和上一份安全缓存仍是必须的第二道边界。
- 2026-09-04 的人工单轴扰动被原 `0.03 rad/s` 正常控制门在 `0.098028 rad/s` 中止；
  操作者于 2026-09-06 补充确认中止前的单轴人工扰动已经验收。正常控制 qvel 默认值和
  后续人工单轴 commissioning 值据此调整为 `0.12 rad/s`，为已观察值保留约 22% 裕量。
  no-touch 命令仍显式使用更保守的 `0.05 rad/s`。
- `0.12 rad/s` 只调整正常控制的 qvel abort。`4.0 rad/s^2` qacc seed、`0.02 rad/s`
  fault-hold stationarity、总力矩、slew、位姿、wrench 和持续时间门保持不变。原始运行没有
  qacc/rejected-sample 完整字段，不能用同一条观察推导其他门的放宽值。

数学、重力和 haptic 组合仍由 `robot_motion.rebot_b601_rs` 负责，仿真 effort backend
仍由 `sim.mujoco.rebot_b601_rs` 负责；本模块不复制 Pinocchio 或 MuJoCo 实现。
离线 `haptic_demo` 的 `pose-tx/ty/tz` 从扰动前当前 FK 锚定后施加 world-axis 平移，
`pose-rx/ry/rz` 施加 tool-local 旋转，`pose-6d` 同时施加两类扰动；输出分别判定
translation/rotation 收敛，并记录 force/torque clip、joint torque/slew、qvel/qacc 和
driver fault。该 ideal-effort 结果只验证控制数学与稳定趋势，不映射为真机力值或手感。

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
4. **P2.1 主动上报传输**：仅由操作者在前台运行下方 no-motion active-report probe。它只
   切换 type-24 上报配置并被动统计 type-2 完整帧，必须自动关闭且逐设备通过。该 PASS 仍
   固定为 `control_feedback_admission=NOT_ADMITTED`，不能进入阻抗/力控状态反馈。
5. **P2.5 失能 TX 上限**：仅由操作者运行 `position-tx-disabled`。确认 preflight/final
   `safe_stop` 均完整、`motors_enabled_by_probe=false`，再检查 `compute/tx` 是否通过；该
   阶段不要求 warmup `feedback` 达到目标，也不构成运动或闭环通过。
6. **P2.6 离线重力与失能 MIT TX**：先离线运行 `gravity-offline`，再仅由操作者运行
   `mit-tx-disabled`。后者必须同时具有 preflight、post-mode 和 final 三份完整
   `safe_stop` receipt、`motors_enabled_by_probe=false`，且命令语义必须保持
   `qvel=kp=kd=tau=0`。这仍不构成已使能 MIT/重力/阻尼通过。
7. **P3 高频 POS_VEL**：仅由操作者运行下方 enabled guarded command。先 125 Hz，再 250 Hz，
   最后才评估 500 Hz；每档均须机械支撑、物理急停、唯一控制 owner、`--auto-disable`，
   并保存 `feedback/compute/tx` 三个 event 的独立统计。只保持启动时反馈位置，不引入
   trajectory 或 haptic overlay。
8. **P3.5 反馈稳定性**：在 P4 前先由操作者运行 `feedback-stability`，电机保持既有安全
   状态。必须得到至少两个完整快照并保存原始 JSON；查看各关节 `p95/p99` 与
   `stationarity_exceed_count`，若有超限，先分析读取周期、编码器量化和机械实际运动，
   不得通过提高 active probe 的 `--fault-hold-max-velocity-rad-s` 绕过该证据。
9. **P4 单关节 MIT/重力阻尼**：由硬件负责人另行批准，先单关节、低幅、当前 q 参考和零
   overlay；确认重力方向、阻尼耗散、最终 torque/slew limit 和 stop receipt 后才扩大到
   多关节。P3 的位置频率通过不等于 P4 的力控通过。
10. **P5 haptic 频率**：接入 `compute_b601_mit_torque()` 与 `J.T @ wrench` 后，单独记录
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

## P2.1 主动上报传输探针（无使能、无运动，操作者手动运行）

该探针复用上面的被动 SocketCAN 解析，只额外通过 MotorBridge 对所选电机发送 type-24
active-report 开/关配置。它不调用 motor `enable()`、不切 MIT/POS_VEL mode，也不发送位置或
力矩参考。因为会改变设备上报配置，所以仍须现场唯一 CAN owner、前台运行和显式自动关闭；
Ctrl+C 会进入同一个 `finally` 清理路径。若任一关闭请求失败，JSON 和终端都会保留
`cleanup_errors`，backend 也会拒绝正常 close，不能把这次运行视为通过。

下面模板不会使用 `set -e`，不会后台运行；命令和完整输出同时追加到固定 log，Python 退出后
当前 shell 保持存活：

```bash
WT=/home/zc/HandMotion/Developing/HarnessLocal/agentic-robot/Worktrees/AR-20260831-b601-active-report-telemetry-r0/AgenticRobot
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
CFG=/home/zc/HandMotion/Developing/reBotArm_control_py/config/rebotarm_rs.yaml
OUT_DIR=/tmp/b601-rs-commissioning/AR-20260831-b601-active-report-telemetry-r0
LOG="$OUT_DIR/command_and_output.log"
JSON="$OUT_DIR/b601_active_report_all_arm_125hz.json"

mkdir -p "$OUT_DIR"
CMD=(
  "$PY" "$WT/scripts/hardware/rebot_b601_rs_active_report_probe.py"
  --config "$CFG"
  --channel can0
  --duration-sec 10
  --target-hz 125
  --minimum-rate-ratio 0.90
  --maximum-p99-factor 2.0
  --minimum-sample-count 100
  --joint joint1,joint2,joint3,joint4,joint5,joint6
  --allow-active-report
  --confirmation B601_RS_ACTIVE_REPORT_TELEMETRY
  --auto-disable-active-report
  --output "$JSON"
)

printf 'command:' | tee -a "$LOG"
printf ' %q' "${CMD[@]}" | tee -a "$LOG"
printf '\n' | tee -a "$LOG"
"${CMD[@]}" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
printf 'probe exit code: %s\n' "$STATUS" | tee -a "$LOG"
printf 'JSON: %s\n' "$JSON" | tee -a "$LOG"
printf 'full command/output log: %s\n' "$LOG"
printf '%s\n' 'shell remains active'
```

运行时会先显示 `[REPORT] 3/2/1/ENABLE STREAM`；这里的 `ENABLE STREAM` 只表示开启遥测上报，
不是电机使能。只有以下条件全部满足，`stream_transport_acceptance` 才是 `PASS`：

- ID1--6 都收到不少于 `--minimum-sample-count` 个完整 type-2 operation-status frame；
- 每个设备的 observed Hz 不低于目标的 90%，且 p99 interval 不超过 `2 / target_hz`；
- host/feedback ID 包含 YAML 声明的 `0xFD`，且没有短 type-2 frame；
- enable 和 cleanup receipt 都完整，没有 cleanup error。

即使上述全部通过，输出仍必须是 `freshness_verified=false`、
`control_feedback_admission=NOT_ADMITTED`、`hardware_acceptance=NOT_CLAIMED`。本探针没有把
payload 解码并绑定到源序号/源时间戳，也没有验证 `q/dq/tau` 的量纲、符号和同帧一致性；因此
它只回答“周期帧是否能到达”，不回答“该状态是否可用于高频阻抗或力反馈”。

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

这是 P4 低增益 MIT/重力阻尼验证的 guarded 入口。它仍然是 motion-capable
命令，只能由现场操作者运行；自动化测试不会打开 CAN、使能电机或发送 MIT 帧。
所有正常模式都使能并调节 joint1..joint6，明确排除 ID7 夹爪；`--joint` 仅决定
slow-step 目标或单关节增益覆盖。`hold` 从使能后的当前姿态开始，不会跳到预设绝对姿态；
末端模式的最终位置恢复由末端阻抗经完整 Jacobian 转置完成。使能后的启动状态机为：

```text
ENABLED_UNARMED → LOCKING → LOCKED → GRAVITY_RAMP (total-torque slew) → HANDOFF → ACTIVE
```

`LOCKING` 立即捕获新鲜的 `q0`，用 pinned YAML 的 MIT Kp/Kd 构建软件侧全六关节位置锁，
首帧和后续锁帧都经过最终总力矩/变化率包络；`LOCKED` 阶段在四个连续反馈样本期间继续
发送固定 `q_ref=q0` 的锁帧。完整六轴的目标重力从首个锁帧开始就进入力矩组合，启动阶段
唯一的渐变是最终总力矩的 slew 包络；不再把重力先乘一个从零开始的独立比例后再做一次
slew，避免补偿空窗造成末端缓慢下坠。重力与临时关节锁保持连续后，再用同样的
`--torque-rise-time-sec` 把临时关节锁平滑交给最终控制器，`pose-hold/cartesian-hold`
的末端弹簧在 `HANDOFF` 中逐步成为位置恢复主体。启动阶段不依赖未受软件包络约束的
vendor Kp；MIT 帧的 vendor Kp 保持为零。`LOCKED` 的过渡检查使用本次运行的
`--velocity-abort-rad-s`，以容纳使能瞬态中仍在正常运动包络内的反馈变化；它不复用
仅用于首次 protective-hold admission 的严格 `--fault-hold-max-velocity-rad-s`。

### 异常时的 protective hold（全臂可拖动重力保持）

这条路径不是把某几个关节强制锁死，也不是回到预设姿态。第一次 Ctrl-C 或可处理的控制
错误会让 `joint1..joint6` 继续保持 MIT 使能，在当前姿态持续计算完整六轴重力补偿，并叠加
一档可拖动的移动阻尼；夹爪 ID7 不在当前 backend 注册范围内，因此不参与这条控制链。软件
Kp 为零，不存在固定位置弹簧。现场人员可以在保持重力平衡的同时拖动整条机械臂。

状态转换固定为：

```text
ACTIVE
  ├─ --duration-sec 到期 ──> PROTECTIVE_HOLD（不会定时失能）
  ├─ 第一次 Ctrl+C ──> PROTECTIVE_HOLD（joint1..joint6 重力+阻尼，允许拖动）
  ├─ 控制/位置错误且当前 q 完整合法 ──> PROTECTIVE_HOLD（同一全臂控制）
  └─ 当前 RX/计算已不可用但存在已验证缓存 ──> DEGRADED_HOLD（缓存重力+电机阻尼）

PROTECTIVE_HOLD
  ├─ RX/数值/运动监视/模型异常 ──> DEGRADED_HOLD
  ├─ 完整六轴发送失败 ──> 同一有界命令立即重试一次
  └─ 再次 Ctrl-C/SIGUSR1 ──> 全部 arm safe_stop

DEGRADED_HOLD
  ├─ 完整合法反馈和实时重力发送恢复 ──> PROTECTIVE_HOLD
  ├─ 缓存六轴发送失败 ──> 立即重试一次
  └─ live 与缓存命令都持续无法完整发送 ──> 全部 arm safe_stop
```

进入 `PROTECTIVE_HOLD` 前，程序必须同时证明：

1. 对明确的运行中 operator Ctrl-C，优先使用六关节完整、有限、在 pinned URDF position
   limits 内且未过期的 `last_safe_q`；若它已过期但当前故障帧完整合法，则使用当前故障 q。
   对普通控制/位置错误，直接使用该错误循环刚收到的完整故障快照，避免过期的
   `last_safe_q` 把可处理的错误升级为全臂失能。如果异常发生时当前帧已经缺失或不可用，
   但此前至少有一帧完整六轴发送成功，则使用与该完整发送配对的最后合法 q 和纯重力项
   直接进入 `DEGRADED_HOLD`；此路径不把过期 q 当成实时测量值，也不启用位置弹簧；
2. 首帧快照年龄按本进程单调时钟测量，并受 `--fault-hold-max-age-sec` 约束；MotorBridge
   0.4.9 没有设备源时间戳，所以 JSON 仍明确标记 `freshness_verified=false`，不能宣传成
   vendor freshness；
3. 首帧 admission 仍使用 `--fault-hold-max-velocity-rad-s` 的静止候选门，普通故障入口
   则使用本次运行的 `--velocity-abort-rad-s`。启动 `LOCKED` settle 也使用后者，以区分
   使能瞬态与首次 Ctrl-C 的静止快照要求。这些门只决定是否能安全进入保持，不限制进入
   保持后的正常人工拖动；保持阶段按 `--velocity-abort-rad-s` 和
   `--acceleration-abort-rad-s2` 监视真正不可处理的运动；
4. 只有通过上述检查后才发送第一帧。发送集始终是完整 `joint1..joint6`，vendor `kp=0`。
   protective-hold profile 的软件 Kp=0；未选择非零 driver Kd 时，软件使用
   `--protective-hold-kd`（默认 `1.0 N·m·s/rad`）作为中等阻尼；显式选择 driver Kd 时，
   该关节由 driver 单独负责阻尼，软件 Kd 自动为零。两条路径都继续经过现有软件绝对力矩和
   slew 包络，以及已回读的 `0x700B` 总力矩上限（如有）；
5. 保持循环每次读取完整反馈，检查有限性、URDF 位置限位、时间推进、运动监视和 MIT 发送
   回执。当前反馈 q 成为下一帧参考并重新计算 `g(q)`；它不会与首次 q 比较，也不会触发
   `max-rebase` 漂移失能。这样既能抑制重力下坠，又不会因为有意拖动而与操作者对抗。

`DEGRADED_HOLD` 的缓存只在一帧 `joint1..joint6` 命令完整成功后更新，并且只保存该帧对应
的合法 q 与纯重力项。它不保存或重放关节位置项、末端 `J^T F` overlay，也不重放基于旧
速度的软件阻尼；因此一个坏反馈、越限值或异常 overlay 不会污染降级命令。缓存重力仍走
原有绝对力矩和变化率包络，vendor Kp 始终为零。由于主机无法在 RX 不可信时重新计算速度，
此阶段只有显式配置且已通过驱动器力矩上限回读的非零 driver Kd 能继续提供“大阻尼”；若
driver Kd 为零，程序会明确记录 `motor_local_damping_available=false`，此时只能诚实声称缓存
重力保持，不能声称仍有软件大阻尼。

以下异常本身都只是进入或维持 `DEGRADED_HOLD` 的理由，不再单独授权失能：feedback 再次
缺失、invalid/stale 或 vendor 状态异常；当前观测越过 URDF 限位；主机观测时间不推进或
`dt<=0`；有限差分速度/加速度非有限或超过配置监视值；FK、Jacobian、gravity、阻抗、功率
一致性或总力矩组合异常。坏样本不进入缓存，程序继续轮询恢复。MIT 部分发送、驱动器拒绝
或发送异常则先对同一有界完整六轴命令立即重试一次；若 live 命令仍失败，再尝试纯重力
缓存命令。只有缓存命令也无法在重试后完整覆盖 J1--J6，才说明主机已经无法可靠维持这个
控制状态，允许全臂失能。这里的“一次重试”是发送完整性判定策略，不是现场调参阈值。

保持没有时间自动释放，也不会因为 EOF、普通 cleanup 或 `close()` 释放。操作者必须在另一
终端显式发送 `SIGUSR1`，或在前台再次按 Ctrl-C，并确认返回的
`full_disable_receipt.completed=true`。除此之外，全臂失能只允许发生在从未形成任何完整
命令缓存，或 live 与缓存六轴命令都经过立即重试仍无法完整发送的情况下。需要注意，
`safe_stop()` 仍是 vendor 的逐关节 disable，不是原子六轴动作；失能后重载机械臂可能下坠，
所以机械承托和物理急停仍是最终安全边界，软件不能宣称总线失联后仍可维持重力补偿。

`--duration-sec` 只限定 `GRAVITY_RAMP/HANDOFF/ACTIVE` 主控制与采集阶段，不是失能倒计时。
到期时程序输出 `CONTROL DURATION COMPLETE`，将这次到期视为第一次保护事件并进入上述全臂重力加阻尼
保持；随后一次 Ctrl-C 或 SIGUSR1 才是最终失能请求。历史参数 `--freeze-sec` 仍为命令行
兼容项，但 duration 到期不再执行“短暂 freeze 后 safe_stop”。若到期时正处于一个标定
记录窗口，该窗口会立即标为 incomplete，整次标定 acceptance 为 FAIL；此前已完整结束的
窗口仍保留为独立证据。

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

本入口默认由软件按
`tau = gravity_scale*g(q) + Kp*(q_ref-q) + Kd*(dq_ref-dq) + J^T F` 组合重力前馈、
位置项、基于 `mechPos (0x7019)` 的有限差分阻尼和可选末端 overlay。启动时 YAML MIT
Kp/Kd 只作为临时软件位置锁；完整六轴重力从首个锁帧进入，并由最终总力矩 slew 包络
平滑达到可用幅值，避免重复衰减造成 J2/J3/J4 补偿比例失衡。重力与临时锁连续后，
位置、阻尼和末端阻抗项一起交给最终 profile。`--torque-rise-time-sec` 决定总力矩 slew
以及锁到最终控制器的交接时间；现场可按机械支撑和急停条件配置，不写成固定验收数值。
因此，要求“重力补偿”时不能把 `--gravity-scale` 或 `--torque-limit-scale` 当成阻抗首测的
低增益参数：前者为模型重力的比例，后者同时限制软件输出和可选的 `0x700B` 驱动器总力矩。
它们必须由当前姿态的模型力矩、负载和机械支撑共同确定；例如此前有效记录使用过
`--gravity-scale 0.95`、`--torque-limit-scale 0.25`，而 `0.5/0.15` 只提供半重力并把
J1--J3 限到约 5.4 N·m，不能宣称完整重力平衡。这是现场命令配置，不是代码或测试中的
固定数值限制。
然后向 upstream MotorBridge 发送 vendor `kp=0` 的 MIT 帧；vendor `kd` 默认也是零。
为避免 20 Hz 有限差分阻尼产生阶梯，可显式令软件 `--kd 0` 并用 `--driver-kd` 把纯速度
耗散交给电机内部环；driver `kp` 仍固定为零，因此末端 `J^T wrench` 仍是唯一最终位置
恢复项。非零 driver Kd 只有在
`--apply-driver-torque-limits` 成功完成六轴写入和回读后才允许使能。最终总力矩先做 URDF effort ceiling 的
`torque-limit-scale`（默认 0.25）绝对限幅，再做 torque slew 限制；超出速度、
位置误差、非有限差分或部分发送会结束 ACTIVE 阶段并优先进入 protective/degraded hold；
单个有限的速度/加速度越限帧按上述两帧确认策略先隔离。上述路径都不直接失能；只有明确
最终停止或 live 与缓存完整六轴发送都失败才允许
safe_stop。由于当前
`mechPos` 是同步参数读取且 MotorBridge 没有源序号/时间戳，本探针的默认 20 Hz
只是低速 commissioning 节拍，不是高频 haptic loop，也不声明 freshness、力传感器
或真实重力参数已验收。Pinocchio 重力模型使用中性辅助夹爪坐标，实际夹爪负载和
摩擦仍须后续标定。

### 交互式多姿态重力残差采集

`gravity-calibration` 独立于已经验收的 `hold`、`pose-hold` 和 protective-hold。
它完成同一套 `LOCKING → GRAVITY_RAMP → HANDOFF → ACTIVE` 启动后，首先进入
`FREE_DRAG`：六个 arm 关节保持实时模型重力补偿和命令指定的单一阻尼 owner，每周期
使用当前反馈作为参考，因此软件位置力矩为零。操作者可在此阶段安全拖动到待测姿态。

拖到目标姿态后应先完全松手，确认机械臂和反馈已经静止，然后按一次 `R`（无需 Enter）。
程序用收到 `R` 按键的同一合法反馈帧捕获六关节参考并进入 `SETTLING`；此时参考已经固定，
但尚未计入正式样本。`--calibration-settling-sec`（默认 2 秒）让非零 `--kp` 在固定
参考下自然建立可观测的位置残差；稳定阶段结束后才进入 `RECORDING`，其长度由
`--calibration-duration-sec`（默认 20 秒）决定。窗口结束后自动回到 `FREE_DRAG`，不失能、
不重新执行启动过程，操作者可换到下一姿态再次按 `R`。记录期间再次按 `R` 不重置窗口。
`R` 只控制采集，不是安全按钮；任意阶段第一次 Ctrl-C 仍进入全臂重力加阻尼
protective-hold，第二次 Ctrl-C 或 SIGUSR1 才请求全臂失能。
如果命令使用 `--kp 0`，捕获的参考位置不会产生位置残差力矩：这类窗口可用于检查姿态、
漂移和模型重力输出，但不能据此反推缺失的静态重力力矩，也不构成已经求解出的重力参数。
当前样本同样不含经过标定的关节实测力矩或电流，因此不能单独完成质量/质心辨识。

### 离线归约、姿态去重和拟合

真机采集完成后，不要直接把所有 20 Hz 帧当作独立姿态，也不要把第 1～4 组 `--kp 0`
窗口送入拟合。使用同一探针的无硬件模式 `gravity-calibration-fit` 完成以下步骤：

1. 只提取完整 `RECORDING` 区间；每个区间用 q、速度、重力项和位置项的中位数归约为一个
   带 MAD 的姿态观测。含非有限值、绝对力矩限幅或 slew 限幅的窗口会被标为无效。
2. 每次完整的 `R` 记录直接归约成一个静态残差观测。命令提供的姿态距离容差只用于识别
   近似重复姿态并平均，操作者不需要判断或标记接近方向，也不需要反复返回同一姿态。
   旧文件中没有方向标签的完整 `R` 窗口同样可用，只要非零 Kp、稳定性和限幅检查通过。
3. 以 `r = position_tau`、`g0 = gravity_tau / gravity_scale` 拟合有效重力修正。第一轮使用
   `global-scale`，直接求出可回填到 `--gravity-scale` 的有效重力比例。若独立验证
   仍有明显姿态相关残差且确认末端存在未建模负载，再显式选择
   `global-scale-payload`；它用 FK/Jacobian 回归非负负载质量和末端坐标系下的一阶矩，
   不修改 URDF。所有模型的输出都是“有效重力 overlay”，不是未经力矩比例/电流标定的真实
   质量/质心测量。
4. 对姿态候选按回归矩阵秩、奇异值和对数体积贪心选择；达到所需信息量后停止，不再用
   同一姿态追加大量重复窗口。至少保留一组未参与拟合的独立验证姿态。

拟合只写入带 `b601.gravity-calibration-overlay.v1` schema 的候选 JSON，并绑定数据集 SHA、
URDF SHA、配置 SHA、选中姿态、秩/条件数和训练/验证残差。`activation` 明确为
`candidate_only_until_independent_validation`；在验证通过前不会自动改变 URDF 或运行时模型。
没有非零软件 Kp、没有有效姿态或没有独立验证数据时，报告为 `CANDIDATE/FAIL`，即使采集
文件本身的控制状态为 `OK` 也不能宣称完成标定。

下面是正式的两批采集和离线拟合命令模板。先完成驱动器限幅 round-trip，再分别采集训练
姿态和独立验证姿态；`duration-sec` 只是本批次的最长人工操作时间，到达时会进入
protective-hold，不会自动全臂失能。每个窗口的操作顺序为：拖到姿态、完全松手、按 `R`、
等待终端提示 `RECORDING` 后保持不触碰。建议训练批次使用 6～8 个不同姿态各记录一次，
验证批次使用 3～4 个全新姿态各记录一次；
J2 约 1.9～2.1 rad 的区域必须覆盖一次，但不得为了覆盖它主动越过机械安全范围。

```bash
WT=/home/zc/HandMotion/Developing/HarnessLocal/agentic-robot/Worktrees/AR-20260906-b601-gravity-startup-handoff-r1/AgenticRobot
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
CFG=/home/zc/HandMotion/Developing/reBotArm_control_py/config/rebotarm_rs.yaml
SRC=/home/zc/HandMotion/Developing/reBotArm_control_py
OUT_DIR=/tmp/b601-rs-gravity-calibration-r1
mkdir -p "$OUT_DIR"

cd "$WT" || exit 86
env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py \
  --mode gravity-calibration \
  --config "$CFG" \
  --rebot-source-root "$SRC" \
  --duration-sec 900 \
  --warmup-sec 2 \
  --freeze-sec 1 \
  --rate-hz 20 \
  --calibration-settling-sec 2 \
  --calibration-duration-sec 20 \
  --kp 5.0 \
  --kd 0 \
  --driver-kd 1.0 \
  --gravity-scale 1.0 \
  --torque-limit-scale 0.40 \
  --apply-driver-torque-limits \
  --torque-rise-time-sec 0.5 \
  --velocity-abort-rad-s 5 \
  --acceleration-abort-rad-s2 40 \
  --max-error-rad 2 \
  --allow-motion \
  --auto-disable \
  --confirmation B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING \
  --fault-hold-max-age-sec 0.15 \
  --fault-hold-max-velocity-rad-s 0.02 \
  --fault-hold-max-rebase-rad 0.05 \
  --output "$OUT_DIR/b601_gravity_calibration_train.json" \
  --log "$OUT_DIR/b601_gravity_calibration_train.log"
```

运行验证批次时保持完全相同的参数，只把输出改为
`b601_gravity_calibration_validation.json` 和对应 `.log`；不要把训练姿态复制到验证批次。
完成两批采集并执行第二次 Ctrl-C 让 protective-hold 正常收尾后，运行以下无硬件拟合：

```bash
env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py \
  --mode gravity-calibration-fit \
  --config "$CFG" \
  --rebot-source-root "$SRC" \
  --calibration-input "$OUT_DIR/b601_gravity_calibration_train.json" \
  --calibration-validation-input "$OUT_DIR/b601_gravity_calibration_validation.json" \
  --calibration-model global-scale \
  --calibration-pose-tolerance-rad 0.08 \
  --calibration-max-poses 12 \
  --calibration-max-velocity-rad-s 0.03 \
  --calibration-max-position-span-rad 0.02 \
  --calibration-overlay-output "$OUT_DIR/b601_gravity_overlay_candidate.json" \
  --calibration-report-output "$OUT_DIR/b601_gravity_calibration_report.json" \
  --output "$OUT_DIR/b601_gravity_calibration_fit.json" \
  --log "$OUT_DIR/b601_gravity_calibration_fit.log"
```

只有拟合报告显示完整秩、没有无效窗口、训练与独立验证残差均下降，且人工复核无启动前倾、
反向抬升、饱和或异常停机，才可另行批准 overlay 的运行时接入。该命令不会自动写回 URDF
或配置；`activation` 保持 candidate-only，失败结果仍会写报告供排查。

终端只显示 `FREE_DRAG`、`SETTLING`、`RECORDING #N`、`COMPLETE #N` 和原有安全状态。
JSON 的 `gravity_calibration.records` 保存触发索引、稳定阶段索引、正式记录起止 sample
index、实际持续时间和完成原因；每帧额外保存 `calibration_state`、
`calibration_record_index` 和窗口内时间。完整 q、有限差分速度/加速度、重力项、位置残差
项、请求/限幅后力矩仍保存在 `samples`，因此可以按每个 record 的
`[sample_start_index, sample_end_index)` 精确提取稳定窗口，而无需把临时残差阈值写进代码
或测试。

`gravity_calibration.acceptance_status` 只有在至少存在一个完整窗口、且所有已触发窗口都
完整结束时才为 `PASS`；零完整窗口或任何 incomplete 窗口都会使标定整体验收失败。这里的
失败不会丢弃完整窗口，也不会授权全臂失能。

该模式必须在具有 controlling terminal 的前台运行；程序在连接 backend 前建立单键输入，
无法取得终端时不会打开 CAN 或改变电机状态。退出正常控制路径时会先恢复终端属性，再进入
既有 protective-hold/cleanup。不要使用后台运行、管道吞掉控制终端或自动发送按键。

运行前必须确认：机械臂处于当前安全、可支撑且有足够间隙的姿态；不得为了探针主动
移动到示例姿态。操作者手扶或有机械支撑，物理急停可立即触达，`can0` 为 1 Mbit/s，
现场没有第二个 CAN 控制 owner。`hold/slow-step` 每次只批准一个目标关节的低幅验证；当前
J6 方案显式使能六个 arm 关节并锚定各自的使能后当前位置，但只有 J6 的 reference 可以
变化。后述 `cartesian-hold` 和 `pose-hold` 固定由六个 arm 关节共同使能和调节。不要在夹爪接触桌面或其他物体时
运行；`safe_stop` 失能后负载可能下落，停止过程中仍需保持支撑。下面命令使用
状态分支和显式 status 打印，不使用会让交接终端提前退出的 `set -euo pipefail`；
命令由操作者在目标 worktree 中运行，我们不在自动流程中代运行。版本化文档保留
可移植变量；实际交接时必须替换为当前 Task 的绝对路径，并先核对固定的 commit/tree：

硬件探针必须在前台同步运行，以便操作者可立即用 Ctrl+C 或物理急停介入。探针把驱动器
stdout/stderr 和详细诊断写入 `--log`；未显式给出时，日志默认与 `--output` 同名并使用
`.log` 后缀。终端只保留启动/急停说明、使能倒计时、保护状态和结束摘要。不得使用 `&`、
`nohup` 或其他后台/脱离终端方式。下面的 `run_logged` 只把命令身份和退出码追加到固定的
`command_and_output.log`，探针本身仍在前台输出核心状态。
所有会调用 `enable()` 的模式在反馈预热、位置/速度边界、preflight `safe_stop` 和 MIT
模式准备全部通过后，都会把简短的 `[DRIVE] 3/2/1/ENABLE` 同时写入固定日志并镜像到
当前控制终端；倒计时期间按 Ctrl+C 会输出 `[DRIVE] CANCELLED` 并在调用 `enable()` 前
结束。运行中第一次 Ctrl+C 请求保护保持，第二次 Ctrl+C 请求六关节全停；`SIGUSR1` 仍可
直接请求六关节全停。只读 `feedback-stability` 不显示使能倒计时：

```bash
WT=/path/to/isolated/AgenticRobot-worktree
PY=/path/to/dexteleop310/bin/python3.10
CFG=/path/to/reBotArm_control_py/config/rebotarm_rs.yaml
SRC=/path/to/reBotArm_control_py
OUT_DIR=/tmp/b601-rs-commissioning
LOG="$OUT_DIR/command_and_output.log"

run_logged() {
  {
    printf '\n===== command start =====\n'
    printf 'started_at=%s\n' "$(date -Is)"
    printf 'cwd=%s\n' "$PWD"
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
  } >>"$LOG"
  "$@"
  STATUS=$?
  {
    printf 'exit_code=%s\n' "$STATUS"
    printf 'finished_at=%s\n' "$(date -Is)"
    printf '===== command end =====\n'
  } >>"$LOG"
  return "$STATUS"
}
```

在任何新的 active 阻抗试验前，先单独验证六轴驱动器临时总力矩限幅事务。下面模式对
J1..J6 逐轴读取一份完整合法反馈、`safe_stop`、把 scaled URDF effort 写入易失的
`0x700B` 并回读，
再次 `safe_stop`，最后恢复本次运行实际读到的原值并回读。它不切模式、不调用 `enable()`、
不发 MIT/POS_VEL 命令，因此不会显示 `[DRIVE]` 倒计时；但它会写驱动器参数，仍须前台运行、
保持机械支撑和可触达急停：

```bash
# 复用上面的 WT/PY/CFG/SRC/OUT_DIR/LOG 和 run_logged
mkdir -p "$OUT_DIR"
if cd "$WT"; then
  if run_logged env PYTHONDONTWRITEBYTECODE=1 "$PY" \
      scripts/hardware/rebot_b601_rs_low_gain_probe.py \
      --mode driver-torque-limit-roundtrip \
      --config "$CFG" \
      --rebot-source-root "$SRC" \
      --torque-limit-scale 0.15 \
      --allow-motion \
      --auto-disable \
      --confirmation B601_RS_DRIVER_TORQUE_LIMIT_ROUNDTRIP \
      --output "$OUT_DIR/b601_driver_torque_limit_roundtrip_all_arm.json"; then
    STATUS=0
  else
    STATUS=$?
  fi
  printf 'driver-limit roundtrip exit code: %s\n' "$STATUS"
  printf 'command and output log: %s\n' "$LOG"
else
  printf '%s\n' 'ERROR: worktree not found'
fi
printf '%s\n' 'shell remains active'
```

只有 `status=OK`、两份 `safe_stop` receipt 完整、`apply_receipt.completed=true`、
`restore_receipt.completed=true`，且 apply 的 `prior_limits_nm` 与 restore 的
`verified_limits_nm` 逐轴一致，才允许在后续 active 命令中加入
`--apply-driver-torque-limits`。active 路径会对 J1..J6 使用同一完整 active set，逐轴应用、
回读并恢复限制。这只验收六轴参数事务，不证明电机会运动或力控有效。

完成该往返验收后，末端阻抗验收使用 `pose-hold`。它在使能后稳定采样的任意当前姿态建立
末端 6D 零误差锚点，以末端位姿/速度误差生成 wrench，再通过完整 Jacobian 转置映射为
J1..J6 力矩。关节位置 Kp 为零；六轴使用统一 Kd 抑制首测下坠速度，但不产生关节位置
平衡点。命令仍会在前台显示 `[DRIVE] 3/2/1/ENABLE`：

```bash

mkdir -p "$OUT_DIR"
if cd "$WT"; then
  if run_logged env PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/hardware/rebot_b601_rs_low_gain_probe.py \
      --mode pose-hold \
      --config "$CFG" \
      --rebot-source-root "$SRC" \
      --duration-sec 60 \
      --warmup-sec 2 \
      --freeze-sec 1 \
      --rate-hz 20 \
      --kp 0 \
      --kd 0 \
      --driver-kd 1.0 \
      --gravity-scale 0.5 \
      --torque-limit-scale 0.15 \
      --apply-driver-torque-limits \
      --torque-rise-time-sec 0.5 \
      --velocity-abort-rad-s 0.5 \
      --acceleration-abort-rad-s2 4.0 \
      --max-error-rad 0.2 \
      --cartesian-stiffness-n-per-m 15 \
      --cartesian-damping-n-s-per-m 0 \
      --cartesian-max-force-n 30.0 \
      --cartesian-max-displacement-m 0.3 \
      --cartesian-overlay-limit-scale 1 \
      --rotational-stiffness-nm-per-rad 1.0 \
      --rotational-damping-nm-s-per-rad 0 \
      --rotational-max-torque-nm 20 \
      --rotational-max-displacement-rad 1 \
      --allow-motion \
      --auto-disable \
      --confirmation B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING \
      --fault-hold-max-age-sec 0.15 \
      --fault-hold-max-velocity-rad-s 0.02 \
      --fault-hold-max-rebase-rad 0.05 \
      --output "$OUT_DIR/b601_ee_pose_impedance_all_arm.json"; then
    STATUS=0
  else
    STATUS=$?
  fi
  printf 'probe exit code: %s\n' "$STATUS"
  printf 'command and output log: %s\n' "$LOG"
else
  printf '%s\n' 'ERROR: worktree not found'
fi
printf '%s\n' 'shell remains active'
```

这一档把所有 host 有限差分阻尼设为零，由六轴 driver Kd 提供电机局部连续耗散；这些数值
只是针对已观察到的 20 Hz 阶梯和腕部静摩擦死区的下一档 commissioning 配置，不是代码
默认值或测试契约。启动锁仍暂时使用 YAML MIT Kp/Kd，但只在软件总力矩包络内生效；交接
完成后 driver Kp 始终为零，旋转刚度仍在末端 6D spring 中，经完整 `J^T` 调节所有关节。

第一档只验收 `pose-hold`：观察任意当前姿态是否成为末端零误差锚点，轻微的末端平移或旋转
扰动是否产生反向、耗散且受限的恢复 wrench；ID7 夹爪不参与。`active_joint_names` 必须正好
是 joint1..joint6，`joint_impedance.software_kp_nm_per_rad` 必须全零，统一的
`pose_impedance.enabled=true`，且 `vendor_kp` 全零。软件 Kd 与 driver Kd 只能按运行命令
明确分配；为改善连续性而启用 driver Kd 时，软件 Kd 应为零，driver Kd 只提供速度耗散，
不创建六个关节位置弹簧。末端平衡点和恢复刚度仍只来自 6D 阻抗及 `J^T wrench`。
正常控制的 `--velocity-abort-rad-s` 与故障保持候选门是两个独立安全层：某一帧低于正常
运动中止阈值、但高于更严格的 `--fault-hold-max-velocity-rad-s` 时，该帧不会替换
`last_safe`，但也不会单独中止正常 hold。JSON 的 `fault_hold_candidate_admission` 会记录
拒绝次数与最后原因。若随后发生真实故障，保护保持仍会重新验证当前帧、旧 `last_safe`
的年龄和反馈边界 rebase；任一条件不满足仍立即执行六关节 `safe_stop`，不得通过提高故障
保持速度阈值绕过。
`--acceleration-abort-rad-s2` 又是独立于 qvel 和 fault-hold stationarity 的第三层；不得因
放宽正常人工推动的 qvel 阈值而同步放宽 qacc 或 `0.02 rad/s` 的保护保持静止门。首次恢复
人工扰动前必须先审查 `motion_guard`，并为该硬件、负载和采样噪声重新批准 qacc 阈值。
并在 JSON 中确认 `status=OK`、`acceptance_status=PASS`、
`cleanup_receipt.completed=true`、`cleanup_errors=[]`、`stop_reason=null`，以及
`frequency_acceptance.feedback/compute/tx.*.passed=true`。输出中的
`hardware_acceptance` 必须仍为 `NOT_CLAIMED`；这是一份低增益软件路径和现场观察
记录，不是 `HARDWARE_PASS`。

在重新尝试任何使能或 MIT 之前，建议先运行下面的 P3.5 反馈稳定性采样。它不会改变电机
的当前状态；如果上一轮异常路径已经完成全臂 `safe_stop`，可直接采样。命令不使用
`set -euo pipefail`，即使探针返回非零，交接终端仍保持可用：

```bash
# 复用上面的 WT/PY/CFG/SRC/OUT_DIR/LOG 和 run_logged
mkdir -p "$OUT_DIR"
if cd "$WT"; then
  if run_logged env PYTHONDONTWRITEBYTECODE=1 "$PY" \
      scripts/hardware/rebot_b601_rs_low_gain_probe.py \
      --mode feedback-stability \
      --config "$CFG" \
      --rebot-source-root "$SRC" \
      --duration-sec 20 \
      --rate-hz 20 \
      --fault-hold-max-velocity-rad-s 0.02 \
      --confirmation B601_RS_FEEDBACK_STABILITY_READ_ONLY \
      --output "$OUT_DIR/b601_feedback_stability_20s.json"; then
    STATUS=0
  else
    STATUS=$?
  fi
  printf 'feedback-stability exit code: %s\n' "$STATUS"
  printf 'result: %s\n' "$OUT_DIR/b601_feedback_stability_20s.json"
  printf 'command and output log: %s\n' "$LOG"
else
  printf '%s\n' 'ERROR: worktree not found'
fi
printf '%s\n' 'shell remains active'
```

此模式的 `status=OK`/`acceptance_status=CAPTURED` 只表示收到了至少两个完整、合法的
快照；是否越过保护速度门要看 `stability.per_joint.*.stationarity_exceed_count` 及
`p95/p99`。任何越过都应先保留 JSON、确认是否有可见机械运动并分析采样量化，再决定后续
实现；不能把它当成放宽 `last-safe` 门槛的理由。

只有上述六关节 hold 通过且机械支撑未改变，才可单独运行一档 J6 相对慢步。实现对 command
和实测相对锚点 excursion 都设有 `20 deg`（`0.349066 rad`）硬上限，但首轮只批准 J6
正方向 `0.02 rad`（约 `1.15 deg`）、`0.005 rad/s`，远低于该上限：

```bash
# 复用上面的 WT/PY/CFG/SRC/OUT_DIR 变量；先确认上一档已停止且 cleanup 完整
if cd "$WT"; then
  if run_logged env PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/hardware/rebot_b601_rs_low_gain_probe.py \
      --mode slow-step \
      --config "$CFG" \
      --rebot-source-root "$SRC" \
      --duration-sec 8 \
      --warmup-sec 2 \
      --freeze-sec 1 \
      --rate-hz 20 \
      --joint joint6 \
      --step-direction 1 \
      --step-rad 0.02 \
      --step-speed-rad-s 0.005 \
      --kp 0.25 \
      --kd 0.5 \
      --target-kp 1.0 \
      --target-kd 0.75 \
      --gravity-scale 0.1 \
      --torque-limit-scale 0.1 \
      --apply-driver-torque-limits \
      --torque-rise-time-sec 2.0 \
      --velocity-abort-rad-s 0.12 \
      --acceleration-abort-rad-s2 4.0 \
      --max-error-rad 0.05 \
      --allow-motion \
      --auto-disable \
      --confirmation B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING \
      --fault-hold-max-age-sec 0.15 \
      --fault-hold-max-velocity-rad-s 0.02 \
      --fault-hold-max-rebase-rad 0.05 \
      --output "$OUT_DIR/b601_low_gain_slow_step_joint6_pos002_all_arm.json"; then
    STATUS=0
  else
    STATUS=$?
  fi
  printf 'probe exit code: %s\n' "$STATUS"
  printf 'command and output log: %s\n' "$LOG"
else
  printf '%s\n' 'ERROR: worktree not found'
fi
printf '%s\n' 'shell remains active'
```

slow-step 的 `acceptance_status=PASS` 现在同时要求
`control_acceptance_status=PASS` 和 `motion_response_status=PASS`。后者使用末尾五份反馈的
均值，要求 reference 已完成、J6 沿命令方向产生至少 `max(0.1 deg, 10% step)` 的响应、
不超过 `150% step`，且任一已使能支撑关节相对锚点的峰值漂移不超过 `0.5 deg`。因此
“命令循环和 cleanup 成功但 J6 没有实际响应”会返回 `status=FAIL` 和
`motion_response_status=INSUFFICIENT_RESPONSE`，不能再误判为动作通过。

参数升级不得采用无依据的逐次试增。官方 RS 的无接触标定只测得 J4/J5 Coulomb 摩擦约
`0.30/0.21 N·m`，并明确指出腕部 `0.2–0.5 N·m` 摩擦可与重力相当或更大；该记录没有
独立测量 J6，因此不能把相邻关节数值当作 J6 标定值。当前 `0.02 rad` 步长、
`target-kp=2.0` 的软件位置项峰值仅 `0.04 N·m`，正反方向都不足响应与这一量级相容。
官方重力锁定示例使用驱动器内部 `kp=8.0, kd=1.0`，配置文件中 J6 常规 MIT 参数为
`kp=50.0, kd=4.0`；二者只是上游实现依据，不是本 commissioning 探针可直接照搬的安全
参数。若后续引入非零驱动器内部增益，准入必须在发送前对每帧证明
`|tau_ff| + kp*|q_ref-q| + kd*|dq_ref-dq|` 不超过探针最终力矩 envelope，同时保留当前位置
锚定、速度/误差 abort、slew limit 和完整 cleanup。新增的 `0x700B` 临时限幅可作为驱动器
侧独立绝对上限，但它不提供软件 slew-rate 约束，也不能单独证明内部 PD 与前馈的精确合成
语义；在非零驱动器 PD 契约实现并离线验证前，继续保持当前单一软件力矩 owner，不运行
更高增益真机试验。

任一方向出现反向、跳变、振荡、超速、超误差、非有限反馈或未完成 stop receipt，
立即按急停/支撑规程停止并保留 JSON；不得通过提高 `max-error`、放宽速度/力矩限幅、
降低目标频率或重复运行来掩盖原因。

### P4.2 当前 TCP 锚定的低增益平移阻抗（操作者手动运行）

只有上面的六关节 `hold` 和 J6 `slow-step` 都通过后，才允许进入这一档。该模式复用已有
`B601TranslationalVirtualSpring`、Pinocchio `LOCAL_WORLD_ALIGNED` Jacobian 和统一的
`J^T F` 映射；锚点是六关节使能后第一份完整、合法、近静止反馈对应的
`base_link_T_gripper_end` 平移，不读取或追踪任何预设绝对 pose。第一版不控制姿态/转动，
因此必须 `--kp 0 --kd 0`，全部六个 arm 关节固定参与调节；ID7 夹爪仍不参与。

笛卡尔力先按轴限幅，再由 MIT profile 对 joint overlay 独立限幅，最后与重力和关节阻尼
相加并通过总力矩绝对限幅和 slew-rate 限幅。每帧 JSON 记录 anchor、TCP displacement、
linear velocity、raw/effective force、Jacobian translational singular values、overlay
requested/effective torque 和 `F^T V == tau^T dq` 的 power error。TCP displacement norm
超过阈值会触发异常 stop；它不会扩大工作空间或把当前测量位置重新设成新锚点。

第一档只验证“使能后不触碰、保持 4 秒”，采用比默认值更小的参数。机械臂应持续由操作者
支撑；这一步只能证明当前姿态的符号、限幅、重力/阻尼组合没有立即异常：

```bash
# 复用上面的 WT/PY/CFG/SRC/OUT_DIR；先确认 joint hold/slow-step 已完整 cleanup
if cd "$WT"; then
  if run_logged env PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/hardware/rebot_b601_rs_low_gain_probe.py \
      --mode cartesian-hold \
      --config "$CFG" \
      --rebot-source-root "$SRC" \
      --duration-sec 4 \
      --warmup-sec 2 \
      --freeze-sec 1 \
      --rate-hz 20 \
      --joint joint1 \
      --kp 0 \
      --kd 0 \
      --gravity-scale 0.5 \
      --torque-limit-scale 0.15 \
      --torque-rise-time-sec 2.0 \
      --velocity-abort-rad-s 0.12 \
      --acceleration-abort-rad-s2 4.0 \
      --max-error-rad 0.15 \
      --cartesian-stiffness-n-per-m 10 \
      --cartesian-damping-n-s-per-m 0.5 \
      --cartesian-max-force-n 1.0 \
      --cartesian-max-displacement-m 0.01 \
      --cartesian-overlay-limit-scale 0.10 \
      --allow-motion \
      --auto-disable \
      --confirmation B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING \
      --fault-hold-max-age-sec 0.15 \
      --fault-hold-max-velocity-rad-s 0.02 \
      --fault-hold-max-rebase-rad 0.05 \
      --output "$OUT_DIR/b601_cartesian_hold_no_touch.json"; then
    STATUS=0
  else
    STATUS=$?
  fi
  printf 'probe exit code: %s\n' "$STATUS"
  printf 'command and output log: %s\n' "$LOG"
else
  printf '%s\n' 'ERROR: worktree not found'
fi
printf '%s\n' 'shell remains active'
```

无触碰档通过后，才可用完全相同的小参数再运行一次，并以单指缓慢推动 TCP，位移严格小于
`5 mm`，随后放手观察恢复方向。不要敲击、快速推拉、扭转末端或用夹爪接触环境。验收时除
P4.1 的通用字段外，还应确认 `active_joint_names` 覆盖 joint1..joint6、
`cartesian_impedance.anchor_source=fresh_post_enable_fk`、`position_tau_nm` 全程为零、
`power_error_w` 接近数值零，且恢复力始终与 displacement 反向。20 Hz 下只能进行这种慢速
commissioning；不得据此声明触觉带宽、接触稳定性或最终 impedance gains 已验收。
人工单轴档只把上方命令的 `--velocity-abort-rad-s` 从 no-touch 的 `0.05` 改为 `0.12`；
不得同时改变 qacc、fault-hold、力矩/slew、`5 mm` 位移或 wrench 参数。每次运行只允许一个
平移轴，并在日志中明确写出轴名，避免再次产生无法拆分逐轴结论的聚合 JSON。

### P4.3 当前末端 SE(3) 锚定的低增益位姿阻抗（操作者手动运行）

`pose-hold` 是 P4.2 之后的独立模式，不改变旧 `cartesian-hold`。它在同一个四样本
arming-settle 门后固定完整的 `base_link_T_gripper_end`，用
`R_current R_anchor^T` 的最短 world-aligned rotation vector 表示姿态误差，并计算
`[f, torque] = -K [translation, rotation] - D [linear velocity, angular velocity]`。
完整 6D wrench 仍只经过现有 `LOCAL_WORLD_ALIGNED J^T` 映射、joint overlay 限幅和最终
总力矩/slew 包络。该模式固定由六个 arm 关节共同调节；首测命令选择 `--kp 0` 和统一关节
Kd 作为速度耗散，但这些调参值不属于代码契约；ID7 仍不参与。

首测统一使用 P4.1 的一条 `pose-hold` 命令，不再维护另一组写死的 no-touch 参数模板。
结果应确认 `pose_impedance.enabled=true`、anchor 来自 fresh post-enable FK、J1..J6 全部
参与、最终关节力矩和 slew 包络生效，并且 `power_error_w` 保持在数值误差范围内。现场调参
保留在运行命令和输出 JSON 中，不进入代码或单元测试契约。

## 已使能力控仍然阻塞的契约

当前实测的六关节同步 `mechPos (0x7019)` 反馈约为 20 Hz，而且 MotorBridge 缓存没有源
序号/时间戳。MotorBridge 参数表明确 `0x701A` 是滤波电流 `iqf`，真正的 `mechVel` 是
`0x701B`；因此既有把 `0x701A` 当速度得到的量纲/符号观察无效，不能据此评价速度反馈。
在 `0x701B` 的真实设备量纲、符号和 freshness 经独立验证前，控制侧仍应使用新鲜位置的
有界差分速度。当前不能用约 20 Hz 的同步参数读取直接闭合 125 Hz 重力/阻尼控制。

另外，`compute_b601_mit_torque()` 若已经组合软件 damping，再同时给驱动器非零 Kd，会重复
计算阻尼。当前只允许一种明确拆分：软件 Kd 为零，driver Kp 为零，driver Kd 单独拥有
电机局部速度耗散，软件拥有 gravity 和 `J^T wrench`，驱动器 `0x700B` 拥有最终总力矩
绝对上限。非零 driver Kp、软件与驱动器双重 Kd、以及把约 20 Hz host loop 宣称为高频
haptic 仍然禁止。真正高带宽 task-space impedance 仍需可验证 freshness 的异步 q/dq
解码路径。

## 操作者手动 enabled 位置频率入口（本轮不执行）

这是 P3 的 motion-capable handoff，必须由操作者在同一有线主机上运行。先把
`REBOT_RS_CONFIG` 指向已审查的 `rebotarm_rs.yaml`，并确认现场支撑/急停/唯一 owner；命令
会在每次 active poll 前对六个 arm 电机请求反馈并读取 `mechPos (0x7019)`，设置 POS_VEL、
使能所选 arm 电机、发送保持当前位置的命令，并在退出时显式失能。反馈/参数读取不是运动
命令；由于 MotorBridge 缺少源序号，输出仍会标记 feedback freshness 未验证。

该独立 frequency probe 当前不复用 low-gain probe 的重力缓存或 protective-hold，duration/
异常退出仍会直接对所选关节执行 `safe_stop`。因此它不能用于无机械承托的负载机械臂；本节
继续标记“本轮不执行”，直到另一个明确范围为它设计并验证等价的全臂保持退出链。

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
