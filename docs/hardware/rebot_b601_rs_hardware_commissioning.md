# B601-RS 有线后端与力控频率 commissioning

本文是 `hardware/rebot_b601_rs_haptic_acceptance.md` 的硬件后端补充。它固定有线
SocketCAN/MotorBridge 的接入边界和频率验收顺序，不替代急停、机械支撑或硬件负责人
批准。离线、被动监听和 MuJoCo 结果都不是 `HARDWARE_PASS`，也不构成使能或运动授权。

## 本地关节助力 GUI（待真机验收）

`rebot_b601_rs_owner.py --torque-console` 复用现有 J1–J6 重力/阻尼 owner，网页
仅在 `127.0.0.1` 提交短时附加力矩意图，不直接操作 CAN。未准备、附加助力归零时
可切换 J1–J6，切换后清空旧关节的锚点、曲线与发送显示，必须重新从当前静止位姿
准备。界面区分 HTTP 已接收、owner 限速后数值、六轴完整 MIT 发送回执和实测
关节位移；终端近期状态与保护原因也可在网页查看。心跳不覆盖结果中的操作者事件。

“进入保护保持”只请求原 owner 的保护保持，重复点击不会升级；“软件急停：全失能”
经确认后请求同一 owner 的六轴全失能，保护保持期间仍可使用。网页/进程/CAN
异常时不能保证软件请求送达，物理急停始终优先。此次界面改动不放宽位置、速度、
支撑关节漂移或力矩边界；命令回执不等于实测电机输出力矩。

助力期间还对所选关节使用 `mechPos` 有限差分速度作独立保护，默认阈值
`0.05 rad/s`（`--torque-console-max-joint-velocity-rad-s`）。越界时本周期不再
发送助力，转入原有保护保持；网页显示当前速度和阈值。此阈值不用于未加力的
手动摆位，也不替代原有速度/位移门禁。保护保持只是撤销助力并继续重力补偿，
不是位置锁。显式 motion profile 的保护保持沿用该 profile 的六轴驱动器阻尼，
不会因调试基线为零而撤掉阻尼；没有 motion profile 的纯重力调试仍可保持零阻尼。
阻尼不保证消除重力残差引起的漂移，操作者仍需保持支撑和物理急停可用。

### GUI 助力试次的离线归约

运行结束后，可对 `--debug-dynamics-trace` 的逐帧 JSONL 做只读归约，无需连接 CAN：

```bash
python -m control_station.b601_torque_console.report --run-dir /path/to/b601-owner-RUN
```

脚本逐行读取 `b601_owner.debug.jsonl`，在相邻目标值变化处分段，输出新建的
`b601-torque-report-*/report.json`；输入记录保持不变。每段包含关节、目标与
实际确认发送的最大附加助力、起始六轴姿态、模型重力、位移范围、首次连续两帧
达到 `1 mrad` 位移的观测点、限幅计数和保护原因。目标归零后还区分撤力斜率期
与**完整发送回执确认助力为零**的时刻，后者起算默认 `0.5 s` 的姿态观察窗口。
这些阈值仅定义离线报告指标，不改变 owner 保护。`active_dispatch` 表示软件发送
已获完整回执，不是实测电机力矩；“未观察到持续位移”也不是静摩擦阈值的下界。
人为接触、传动弹性、反馈量化或未校准的驱动力矩比例仍可能解释同一记录，须与
操作者标注一同判断，不可据此自动放宽力矩或速度限制。

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
  有界投影，不修改源 URDF 或真实机械硬限位。J4/J5/J6 另有独立的反馈工作包络：
  在 nominal URDF 上下界外各接纳 `0.1 rad`，真实反馈不做投影，仍进入 FK 和重力计算；
  这不是自动规划或位置命令范围。超过工作包络的读数仍 fail-closed。JSON 样本同时保留 `raw_qpos_rad`、归一化后的
  `qpos_rad` 和 `boundary_normalized_joint_names`，便于复核实际读数。
- J2 的 `-1e-2 rad` 是只读接纳下限，不是常规工作范围：两次零位只读捕获曾观察到
  `-4.943/-4.517 mrad`，2026-09-08 使用 host-id 限定反馈的启动锁/重力交接又观察到
  连贯的 `-7.627 mrad` 边界瞬态，因此仅对 J2 保留到 `-10 mrad` 的整值余量。J2/J3
  不额外扩展一般工作包络；比各自下限更负的反馈仍 fail-closed。
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
  接纳包络，默认回退到严格的 `position_limits_rad`；probe 显式加入 J2/J3/J4
  已批准的边界窗口，以及 J4/J5/J6 各 `0.1 rad` 的反馈工作余量。这样即使
  `mechPos(0x7019)` 返回一个有限但物理上不可能的巨大数值，
  backend 也会将该关节标为 `valid=false, stale=true`，并在错误中保留关节名、motor/feedback
  ID、参数 ID、原始值、可用时的 IEEE-754 单精度位模式和接纳区间；坏帧不会被任何 backend 消费者误当成完整状态。出站 MIT/
  POS_VEL 仍只使用严格 URDF 包络。临时启动/回放锁之外的 active probe vendor Kp 固定为零；当真实反馈位于
  J4/J5/J6 工作余量时，软件位置项和 `g(q)` 使用真实接纳姿态，编码进 MIT 帧的位置参考
  则夹在 nominal 命令包络内。临时锁若遇到这种夹紧，必须把对应关节的 vendor Kp 置零，只保留
  重力前馈和本地 Kd，不能让夹紧后的参考主动把关节拉向 URDF 边界。读侧余量不能用于放宽
  slow-step 或其他规划命令。active probe 会把每关节
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
   切换 type-24 上报配置并被动统计 type-24 active-report 完整帧，必须自动关闭且逐设备通过。该 PASS 仍
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

- ID1--6 都收到不少于 `--minimum-sample-count` 个完整 type-24 active-report frame；
- 每个设备的 observed Hz 不低于目标的 90%，且 p99 interval 不超过 `2 / target_hz`；
- host/feedback ID 包含 YAML 声明的 `0xFD`，且没有短 type-24 active-report frame；
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

`LOCKING` 立即捕获新鲜的 `q0`，把 pinned YAML 的 MIT Kp/Kd 交给电机本地 MIT 闭环建立
六轴同步发送的临时位置锁；主机只发送经过最终总力矩/变化率包络的重力和任务前馈。若某
关节的接纳反馈位于 nominal URDF 命令包络外，该关节仍同步发送，但本地 Kp 被置零，只保留
重力和本地 Kd，避免编码参考夹回边界后产生主动运动。`LOCKED` 阶段
在四个连续反馈样本期间继续发送固定 `q_ref=q0` 的锁帧。临时位置与阻尼项不再经过约
20 Hz 的主机有限差分和力矩 slew，避免所需制动力矩反转时实际命令仍滞后并向机械臂注入
能量。使用该锁止路径必须先通过 `--apply-driver-torque-limits` 把缩放后的 URDF effort
写入并回读 `0x700B`，由驱动器约束完整 MIT 位置、阻尼和前馈合力矩。

完整六轴的目标重力从首个锁帧开始就进入力矩组合；不再把重力先乘一个从零开始的独立
比例后再做一次 slew。重力与临时关节锁保持连续后，再用同样的
`--torque-rise-time-sec` 把本地 Kp 平滑撤到零、把本地 Kd 交接到 active driver Kd，
`pose-hold/cartesian-hold` 的末端弹簧在 `HANDOFF` 中逐步成为位置恢复主体。交接完成后
vendor Kp 严格为零。`LOCKED` 的过渡检查使用本次运行的
`--velocity-abort-rad-s`，以容纳使能瞬态中仍在正常运动包络内的反馈变化；它不复用
仅用于首次 protective-hold admission 的严格 `--fault-hold-max-velocity-rad-s`。

每次运行还会把启动专项证据写入结果 JSON 的 `startup_diagnostics`。它保存使能前反馈、
`ENABLED_UNARMED`、首个 `LOCKING` 命令、各个 `LOCKED_SETTLE` 样本以及首次
`ACTIVE` 的位置差，并在普通逐周期 `samples[].vendor_status` 中保留 MotorBridge 已经接收的
operation-status 缓存速度/力矩。该观测完全被动，不增加 CAN 请求，不参与控制或安全判定；
由于 pinned ABI 没有源时间戳和命令-响应关联，它明确标为 unverified。若同时使用包含
`robstride-startup` 调试模式的 MotorBridge，原始日志会在既有 type-17 帧之外记录 type-1
operation-control 和 type-2 operation-status 帧，便于离线核对命令与解码。这里不在 20 Hz 控制循环中同步追加
`iqf (0x701A)`/`torque_fdb (0x302C)` 读取，避免把每周期六次位置读取扩成十八次并降低控制节拍。
`startup_diagnostics.displacement_gate_rad=null`：位移只作为测量证据，不新增脱离实际硬件依据的
数值型通过阈值。

### 连续使能的启动交接回放

`startup-replay` 用于区分“从零位边界进入自由空间平衡”和“非边界姿态下仍有重力模型偏差”。
它先完整执行普通启动状态机，然后保持 gravity+damping，让机械臂在软承托和物理急停保护下
自行稳定。操作者确认机械臂已经停止后，只按一次 `R`，不拖动机械臂。程序捕获当前六轴
姿态，并在同一次使能会话内自动执行：

```text
FIRST_ACTIVE → WAITING_FOR_R → RAMP_IN → RELOCKED → HANDOFF → SECOND_ACTIVE
```

`RAMP_IN` 使用 `--torque-rise-time-sec` 把 YAML-derived 电机本地 MIT Kp/Kd 从 active
`Kp=0`/所选 driver Kd 平滑加入；`RELOCKED` 使用既有 `--freeze-sec` 保持；`HANDOFF` 再用
同一 rise time 平滑回到 active `Kp=0`/所选 driver Kd。完整重力向量在全部阶段保持比例
`1.0`，主机临时 Kp/Kd 始终为零，前馈力矩继续经过同一个绝对值和变化率 conditioner，
完整电机输出由已回读的 `0x700B` 约束。该模式不自动判断“已经稳定”，避免增加会被
静摩擦误导的速度/持续时间阈值；`R` 只选择回放参考，不是安全按钮。
每个样本同时记录实际 vendor Kp/Kd、参与位置锁的关节和因 nominal 命令边界而取消 Kp 的
关节；边界取消不是失败，也不会改变该姿态下的 `g(q)`。

结果 JSON 的 `startup_replay` 保存 R 时参考、每次状态转换的 q/有限差分速度、第二次 ACTIVE
和最终位置相对参考的六轴差值；`samples[]` 同时保存 `startup_replay_state` 和
`startup_replay_lock_fraction`。`displacement_gate_rad` 仍为 `null`，完成回放只代表证据完整，
不代表重力模型已经通过验收。持续时间到期和首次 Ctrl-C 仍进入既有六轴 protective hold。

2026-09-09 的首个回放候选曾把 YAML Kp/Kd 作为约 20 Hz 主机软件 PD，再把位置/有限差分
阻尼力矩送入持续 slew conditioner。实测所需制动力矩反向后，conditioned torque 多帧仍保留
旧符号，J2 在 `RELOCKED` 内从约 `0.36 rad` 发散并往复运动；完整 type-1/type-17 证据排除
了部分发送和浮点坏帧。当前实现禁止这条结构：临时 host Kp/Kd 为零，YAML MIT 增益只在
驱动器本地执行，完整输出必须由回读后的 `0x700B` 约束。该修复不修改 gravity vector、
标定窗口或离线拟合语义；回放真机复验通过前仍不能声称重力模型已经验收。

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
5. 保持循环每次读取完整反馈，检查有限性、反馈工作包络、时间推进、运动监视和 MIT 发送
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
缺失、invalid/stale 或 vendor 状态异常；当前观测越过反馈工作包络；主机观测时间不推进或
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

本入口正常 active 控制默认由软件按
`tau_i = gravity_scale*gravity_joint_scales[i]*g_i(q+offset) + bias_i + friction_i*direction(dq_i) + Kp_i*(q_ref_i-q_i) + Kd_i*(dq_ref_i-dq_i) + (J^T F)_i`
组合重力前馈、
位置项、基于 `mechPos (0x7019)` 的有限差分阻尼和可选末端 overlay。启动和
`startup-replay` 的临时锁例外地使用驱动器本地 MIT Kp/Kd：主机前馈不重复临时位置或
阻尼项，完整六轴重力从首个锁帧进入。交接时本地 Kp 回到零、本地 Kd 回到 active driver
Kd，软件最终 profile 同步接管。`--torque-rise-time-sec` 决定主机前馈 slew 以及本地锁到
最终控制器的交接时间；现场可按机械支撑和急停条件配置，不写成固定验收数值。
因此，要求“重力补偿”时不能把 `--gravity-scale` 或 `--torque-limit-scale` 当成阻抗首测的
低增益参数：前者为模型重力的比例，后者同时限制软件输出和可选的 `0x700B` 驱动器总力矩。
它们必须由当前姿态的模型力矩、负载和机械支撑共同确定；例如此前有效记录使用过
`--gravity-scale 0.95`、`--torque-limit-scale 0.25`，而 `0.5/0.15` 只提供半重力并把
J1--J3 限到约 5.4 N·m，不能宣称完整重力平衡。这是现场命令配置，不是代码或测试中的
固定数值限制。
`--gravity-joint-scales J1 J2 J3 J4 J5 J6` 是显式、可回退的逐关节重力比例。操作者最新
确认后，默认值为 `1 0.96 1 1 1 1`。该向量在全局 `--gravity-scale` 之后生效，并贯穿
启动锁、ACTIVE、正常 protective hold 以及从最后完整命令恢复的 degraded hold；不能只在
ACTIVE 中临时修正，否则第一次 Ctrl-C 或一般异常会产生力矩跳变。每个值只要求有限且
非负，不把常规调参范围写成代码硬限制；实际总输出仍受现有 URDF effort 比例、驱动器
`0x700B` 回读限矩和 slew conditioner 约束。结果 JSON 的 `profile.gravity_joint_scales` 与
`profile.effective_gravity_multipliers` 会记录本次真实配置。

`--gravity-joint-offsets-rad`、`--gravity-joint-bias-nm` 和
`--gravity-coulomb-friction-nm` 同样按 J1--J6 顺序显式给出。默认 offset 和 friction 均为
六个零，默认 bias 为 `0 -0.22 -0.11 -0.08 0 0 N·m`。offset 只改变 Pinocchio 重力求值
姿态 `q+offset`；bias 是有符号常值力矩；Coulomb
项的幅值必须非负，方向来自 `mechPos` 有限差分速度。为防止零速量化噪声让非零摩擦项
逐帧换向，`--gravity-coulomb-velocity-threshold-rad-s` 定义方向为零的速度区间，默认
`0.02 rad/s`，且只有启用非零摩擦项时才有实际影响。这些配置和比例一样贯穿启动、ACTIVE 与 protective
hold，并完整写入结果 `profile`；省略参数时使用上述默认值。

### 重力参数调试模式的保持契约

当现场需要比较重力比例、offset、bias、Coulomb 或 driver Kd 对手感的影响时，
canonical `rebot_b601_rs_owner.py` 默认启用 `--gravity-debug-mode`；直接调用
low-gain probe 或其他模式仍需显式传入该开关，owner 可用
`--no-gravity-debug-mode` 恢复普通保护保持语义。该开关只改变参数
来源的可审计语义，不放宽任何安全门禁：程序在使能前把本次命令中的
`gravity_scale`、六轴 scales/offsets/bias/friction、Coulomb 速度阈值和
`driver_kd` 复制为不可变的 run snapshot。重力标定参数贯穿 ACTIVE、首次 protective
hold、反馈恢复后的 live hold，以及无可信反馈时的 degraded cached hold。
普通重力调试的 driver Kd 也使用该 snapshot；显式 motion profile 的保护阻尼
使用已配置的 `drive_driver_kd_nm_s_per_rad` 与基线 Kd 的逐轴较大值。这样自动标定
发生故障时不会从运行阻尼切换成零阻尼，也不引入高于已有 profile 的新增益。
结果 JSON 的 `gravity_debug` 保留调试基线，`protective_hold` 和动态轨迹记录实际
保护阻尼及其来源；离线 preflight 同样显示保护阻尼。两者不再混为同一参数组。

因此，调试期间第一次 Ctrl-C、普通故障或 duration 到期不会偷偷切回默认重力
profile，操作者可以把姿态变化归因于当前正在测试的参数。degraded hold 仍只
重放最后一帧已验证的 gravity torque（不伪造软件速度阻尼），这是反馈不可信时
的既有安全语义。`--gravity-debug-mode` 不改变反馈年龄/速度/加速度/位置/力矩
限制、active-report 门禁、完整六轴发送重试、protective hold 的二次失能规则或
物理急停要求；显式取消该开关的普通 `hold`、遥操和其他模式完全沿用原路径。

调试模式仍关闭 protective hold 的默认主机阻尼 fallback：没有 motion profile 时，冻结快照中的某个
`driver Kd` 为零，live protective hold 和 degraded cached hold 都保持该关节的
`Kd=0`，不会在故障边界偷偷加入 `protective_hold_kd`。这样保护阶段不会改变正在
调试的手感。普通模式仍保持原有语义：`driver Kd=0` 的关节使用
`protective_hold_kd` 作为保护保持阻尼；这条差异只由显式 `--gravity-debug-mode`
触发；显式 motion profile 的保护阻尼按上面的规则独立确定，并记录在 `protective_hold` 的 `damping_owner`、
`software_damping_fallback_enabled` 和 `preserve_zero_driver_damping` 字段中。

典型的直接 probe 调试命令需要加入：

```text
--gravity-debug-mode
```

每次改变参数都应重新启动一次调试 run，不能在同一 run 中修改命令行列表或
依赖运行时外部文件热改；这样 live 与 protective/degraded 证据始终属于同一
组冻结参数。

需要排查速度异常、力矩方向或 protective hold 切换时，可给 owner 加
`--debug-dynamics-trace`。它默认关闭，开启后在每次唯一 run 目录写入
`b601_owner.debug.jsonl`；终端仍只显示简短状态。`active_dispatch` 记录含单调时间、
六轴位置、由相邻 `mechPos` 样本差分得到的速度、命令重力/阻尼/助力/最终力矩、
限幅标记以及 protective hold 的 live/cached-gravity 来源。`commanded_power_w`
是“发送力矩 × 差分速度”的诊断量，**不是实测电机输出功率**；vendor 状态字段只作
旁证，不参与控制速度计算。该开关不增加 CAN 读取、不更改控制参数或安全限值。
写盘发生故障时，诊断停止，结果 JSON 的 `debug_dynamics_trace.error` 会说明原因，
控制与保护路径不因日志故障改动。鉴于该模式每周期写本地文件，真机再运行前仍须
离线核对本机日志开销与控制频率；不能用它替代急停或解释尚未证实的反驱动根因。

这些参数现在组成操作者确认的默认重力补偿 profile；原始 URDF 模型不再是省略参数时的
默认行为。Seeed 的 RS 实机报告曾得到 J2/J3 约
`0.948/0.952` 的有效重力比例；本机第 10 个无接触窗口也观察到 J2 在正重力方向发生延迟
breakaway，位置锁最终提供约 `-0.268 N·m` 修正，方向和量级均支持先测试 J2 `0.95`。
当前默认比例为 `1 0.96 1 1 1 1`，配合新的 J2--J4 bias 使用。
要回退原始模型，需显式传入六个 scale `1`、六个 offset/bias/friction `0`。
进入最终 active 后向 upstream MotorBridge 发送 vendor `kp=0` 的 MIT 帧；只有上述有明确
起止状态的临时启动锁允许非零 vendor Kp。
为避免 20 Hz 有限差分阻尼产生阶梯，可显式令软件 `--kd 0` 并用 `--driver-kd` 把纯速度
耗散交给电机内部环。当前正常重力补偿的默认 driver Kd 按 J1--J6 为
`0 0 0 0 0 0`；显式单值仍表示六轴共用。driver `kp` 仍固定为零，
因此末端 `J^T wrench` 仍是唯一最终位置
恢复项。正常重力补偿的软件默认值为 `--kp 0 --kd 0`；`gravity-calibration` 若要在按下
`R` 后用位置残差采样，必须为该采集显式选择经过审核的非零 `--kp`。非零 driver Kd 只有在
`--apply-driver-torque-limits` 成功完成六轴写入和回读后才允许使能。本次默认值变更限定在
B601 硬件 owner/probe 及其 protective/degraded hold；已有 MuJoCo
`B601Real2SimControllerProfile` 保持独立的仿真契约，避免未经单独验收就改变既有仿真动力学
和回归结果。最终总力矩先做 URDF effort ceiling 的
`torque-limit-scale`（默认 0.25）绝对限幅，再做 torque slew 限制；超出速度、
位置误差、非有限差分或部分发送会结束 ACTIVE 阶段并优先进入 protective/degraded hold；
单个有限的速度/加速度越限帧按上述两帧确认策略先隔离。上述路径都不直接失能；只有明确
最终停止或 live 与缓存完整六轴发送都失败才允许
safe_stop。由于当前
`mechPos` 是同步参数读取且 MotorBridge 没有源序号/时间戳，本探针的默认 20 Hz
只是低速 commissioning 节拍，不是高频 haptic loop，也不声明 freshness、力传感器
或真实重力参数已验收。Pinocchio 重力模型使用中性辅助夹爪坐标，实际夹爪负载和
摩擦仍须后续标定。

### 手动确认活动范围时显示六轴角度

前台 owner 可加 `--show-joint-angles`，每 0.5 秒显示一行 J1–J6 带符号的反馈角度，单位为度。
显示复用同一 owner 已取得并通过本周期检查的六轴反馈，不另开 CAN、不产生助力，
也不改变参考、重力、阻尼或保护阈值。角度保留边界投影前的关节实测值，使用模型关节
坐标零点，不是相对启动位置的位移；终端和 owner 日志均可查看。只在收到有效反馈并
完成本周期发送后刷新，正常 protective hold 继续显示；反馈不可信的 degraded hold
不重复旧角度冒充新测量。该选项默认关闭，显示频率独立于控制频率。

使用 GUI 时不附加 `--show-joint-angles`，避免终端刷屏。页面上方的“实时六轴角度与命令”
表同时显示 J1–J6 的度、弧度、速度及命令；复用既有状态快照，不增加 CAN 查询或发送。
角度列使用 owner 控制反馈（小幅边界外读数可能已投影，原始读数在 debug trace），
不是相对起点位移；表头显示反馈年龄。过期值置灰并明确标记“最近值”，连接中断时清空。
终端角度选项仍保留给无 GUI 的活动范围检查，不改变显式启用该选项时的行为。

仅检查手动活动范围时使用普通 `--mode hold --kp 0 --kd 0` 及已确认的重力参数，
不要附加 torque-console、自动标定或 policy-assist 选项。等待 `STARTUP ACTIVE` 后手动
缓慢移动，记录边界姿态的完整六轴角度以及接触／托架限制；各轴的独立最小、最大值不能
证明它们的任意组合都可达或无碰撞。原有第一次 Ctrl-C 保持重力、再次 Ctrl-C 请求
最终失能的协议不变。

### J2/J3 多位姿自动采集（重力优先，离线实现，待真机验收）

新增 `docs/hardware/profiles/b601_j2_j3_workspace_calibration_v1.json`。仍由原有
`rebot_b601_rs_owner.py --torque-console-auto-calibration-profile <profile>` 启动；
GUI 保留“当前位置小范围测试”和“完整流程”两个旧按钮，并增加“J2/J3 多位姿自动采集”。
默认仍选旧的定点模式，选择新按钮本身不会驱动；声明无人介入并点击开始才执行本批次。
本模式只负责硬件采集，不接入 replay、policy 或 RM65。

操作者确认的**目标范围**为 J2 `[0,120]°`、J3 `[0,90]°`。J2 偶尔出现的
`−0.3°` 是弹性反馈，不是负方向目标授权；继续沿用原 owner 的反馈边界处理，原始读数
仍保存。连接前核对 pinned URDF 范围能覆盖该目标范围，不扩大 URDF 或 owner 的硬边界。
关节区间本身不是自碰撞或环境碰撞证明，开始前必须清空各组合位姿及过渡路径的运动空间。

首轮仅从 J2/J3 近零位开始（各不超过 `2°`），使用 J2 `3/6°` × J3 `3/6°` 四个中心，
本批实际反馈和目标范围限定为两轴 `[0,10]°`。**撤回此前未经碰撞审查的九姿态宽网格测试方案**；
机械区间合规不代表组合位姿无碰撞。后续扩大范围必须先检查组合及路径，不能直接做笛卡尔积。
每个中心依次执行：

1. 先 J3、后 J2 分别换位，每次最多 `0.06 rad`，不同时给两轴添加定位助力；
   避免保持 J3 折叠零位时先搬动 J2。
2. 撤去附加驱动力，观察五秒重力补偿下的六轴姿态范围和漂移；不使用位置锁掩盖漂移。
3. 先 J2、后 J3，复用原有局部双向、双速度采集。每轴十二段，四姿态共八组、96 段；
   记录实际六轴姿态，不把计划中心当作真实姿态。
4. 正常完成后先 J2、再 J3 分段返回本批起点，确认零助力、基础阻尼和原有停稳门；
   保持重力补偿，等 GUI 人工确认机械支撑后才全失能。

换位和局部采集都复用原 staged controller、运动 session、完整六轴发送回执与最终
conditioner。J2/J3 使用各自显式配置的六轴 driver Kd；本 profile 两组暂均为
`0.4/0.6/0.4/0.6/0.2/0.1`，基础重力模式和重力参数不变。既有速度、单段位移、
支撑轴、最终力矩和 `40 ms` watchdog 继续生效。首轮 profile 有 `2400 s` 时限和
`800 N·m·s` 软件命令暴露预算；子段仍受原时限和暴露预算约束，不无限恢复或重试。
这些是命令预算，不是实测力矩冲量、人体受力或实测能量。

J2/J3 均可使用同一显式解析运动配置中的定力起步观察和 TRACK 交接。所有 J2/J3 的
换位／采集子控制器配置均在 CAN 连接前构造校验，避免完成 J3 换位后才遇到旧 J3 专用
校验异常。该通路不开放给 J1/J4–J6，且不改变各阶段数值包络、重力参数或最终停止规则。

后台继续，关闭/刷新页面、人工介入声明或撤销会停止本批并撤力，**不自动返回**。
停止后不能自动恢复旧目标。完成和故障的已采数据分别保留，后续停止不能回溯删除前面段。
全程无独立接触传感器，人工介入仍来自已有 GUI 声明，不把它冒充自动人体检测。

已确认六轴零助力、基础阻尼和停稳之后的 `stopped_short_of_target` 属于采集未完成告警：
停止本批，不自动重试或返回，GUI 和原 owner 的重力补偿继续运行。它不会记作到位，
也不再仅因采集状态为 `failed` 就升级为硬件故障。反馈失效、越限、部分发送、回执缺失、
撤力未确认等仍保持故障处理。结束的子任务在最后回执后立即归档，不再产生新待发送帧。
控制中断时，部分阶段保存实际状态及 `pending_dispatch`，不会补造回执或停稳结论。

显式连续调速将正常 TRACK 的有界维持项与 PD 推进量分开：连续速度缩放只削减
同向 PD 推进量；比例速度误差反馈则必须能够削减完整同向合成输出，不能被
较小或已经反向的 PD 项封顶。两项取较严格的减力，只应用一次，不新增力矩反向。
积分记忆不被这项瞬时输出减力递归扣除，但仍接受原有速度误差积分、派发短缺和上限约束。
维持项仍由窗口速度误差积分、原有积分变化率、最终力矩上限和派发短缺约束更新，
不是固定力矩下限，也不是已辨识的重力／摩擦补偿。反向 PD 制动保留，积分仍能衰减。
原始速度硬门、窗口／近期速度软制动、力矩上限、变化率、撤销和 watchdog 保留。
trace 的 `tracking_velocity_correction_nm` 和 `continuous_speed_drive_cut_nm` 仍分别记录
速度反馈减力及额外调速减力；新增 `continuous_speed_bound_cut_nm` 是第二项独立约束，
`applied_drive_cut_nm` 是两项约束合成后的总减力；`unscaled_maintenance_nm` 和
`regulatable_propulsion_nm` 分别显示本周期单独合成的维持项和可调推进量。它们均非实测力矩。
`velocity_feedback_baseline_mode=continuous_pd_scaling_with_net_velocity_feedback`
说明“PD 缩放、净输出速度反馈”的区别；维持项不是必须发送的最低驱动力。

workspace 的定位／回位使用独立的 `workspace.positioning_support_soft_speed_recovery`，
默认关闭，当前近零显式 profile 开启；原 `cycle` 同名配置仍只属于完整启动模式。
它只复用已有的支撑轴软速度制动恢复：先撤力，六轴停稳及完整零发送确认后，
在原目标、原段时限和原命令暴露预算内恢复。扫掠阶段不隐式开启这个定位选项。
持续支撑轴运动、硬速度、位移、反馈／回执异常、人工撤销均不能由此恢复。

本轮 `w2f7j86f` 在第一段 J3 定位的 2.325 秒附近，J4 窗口速度约 0.0393 rad/s
越过 0.039 rad/s 软门，继而达到约 0.057 rad/s；定位子控制器未接入上述恢复选项，
因此停在约 1.362°，未达到 3° 目标。这是实际支撑轴运动证据，不标为噪声误报。
同段 2.204 秒附近 J3 窗口速度约 0.0252 rad/s，目标速度约 0.0092 rad/s，
维持偏置约 0.619 N·m；旧合成仍请求约 0.607 N·m。
修复净输出反馈后，同一固定输入约请求 0.447 N·m；这只是命令反事实，
不是闭环运动预测或真机平顺性验收。不改变重力、阻尼及任何门限来通过本例。

同一有效目标内的选定轴软速度制动，会先把维持记忆限制到此前实际完整发送过的
同向附加命令幅值，只冻结记忆，不在 BRAKE 输出这项力，也不继续积分。
`tracking_bias_frozen_for_governed_stop` 标记该状态。仍须原有六轴零发送／停稳确认，
才能沿原 TRACK 上升率恢复；不重置目标、段时限或命令暴露预算。
人工介入、撤销、权限失效、支撑轴制动、反馈／硬保护和到位卸载不保留这份记忆。
起步交接也不把大脉冲当成已验证维持负载：首周期仍执行原有快速 DROP，
被撤掉的起步尾量不会存成潜在维持项。默认非连续路径保持原行为。

连续模式中的速度误差减力与维持助力积分统一使用同一份 80 ms 因果窗口速度，
不再把单帧差分或更短近期窗口的最大值混入该正常调节回路。否则一次编码器计数
可能在窗口显示低速时触发较大瞬时减力，并与下一周期恢复驱动形成往复。
`tracking_cut_velocity_source=causal_window_same_as_maintenance` 标明这一来源；
近期位移／速度仍独立约束连续速度缩放、快速撤力和软制动，原始差分仍执行硬速度保护。
默认非连续模式、补偿参数、力矩／位移／预算上限及退出收尾语义不变。

连续到位稳定阶段沿用 TRACK 的派发短缺计算，不把已计划的速度减力反复扣入维持记忆。
只在首次进入到位阶段时，以最近完整发送总量为上界并预留当期同向 PD，裁掉隐藏记忆；
随后只扣“上一 owner 请求与其实际完整回执之间的额外差额”。进入最终卸载后再次约束
到实际发送值并按原卸载率减力，不能从隐藏记忆恢复输出。`dispatch_credit_mode` 分别为
`arrival_entry_reserve_pd_from_sent_total`、`receipt_shortfall_from_previous_owner_request`、
`arrival_unload_sent_total`。到位稳定中的缓慢修正不是最终卸载；人工介入、权限撤销、
硬保护、原参考偏差门、段时限、暴露预算和六轴零后停稳确认仍保持原行为。

`w36ya827` 已完成 J3/J2 定位及首组 5 秒重力观察；首段 J2 负向扫掠进入到位区后，
维持记忆连续由约 -0.485 降为 -0.271、再降为零，随后回退并触发参考偏差制动。
这些是软件命令和编码器证据，不是独立输出测力。修复重复扣减不等于放行参考偏差越限，
也不能证明重力残差与传动弹性已经分离；原始报告和诊断窗口保留，不自动安装拟合值。

两轮实际 39.94 Hz 的主要时间仍在反馈准入重试。每周期 `feedback_read_diagnostics`
新增固定三类 `rejection_counts`：`backend_poll`、`host_freshness_admission`、
`baseline_not_advanced`；`first_rejection`／`last_rejection` 只保存首末阶段、序号及
最多 400 字符的原因，不保存异常对象或全部尝试，不额外读取控制时钟或请求 CAN 帧。
这些字段不改变准入、1 ms 重试等待、超时和派发锚点。现有记录缺少具体拒绝原因，
本候选只补齐证据，尚不能声称 40 Hz 已恢复为 100 Hz；不得降低验收线掩盖该问题。

采集进入失败／撤销时，现有终端日志立即显示停止原因及“不自动重试／返回”。
GUI 说明批次已经结束推进，并区分已确认仅重力补偿与尚需查看保护／撤力回执的状态，
不再让“采集停止”看起来像等待下一段；它不表示六轴已失能。

退出时先原子保存主结果检查点，样本归档状态为 `PENDING`；归档完成后写为 `COMPLETE`，
再开始离线报告。归档中断会保留主结果、原轨迹及已有部分归档，明确标记 `INTERRUPTED`，
不会声称全部样本已归档。报告汇总中断不覆盖控制结果及首次停止原因，
以 `automatic_calibration_report_status=INTERRUPTED` 和退出码 130 表示后处理被中断。
正常报告状态为 `COMPLETE`；已知报告数据／序列化异常仍作为显式 ERROR 写入主结果。
主结果先于大轨迹扫描落盘，不再依赖报告完成才能拿到退出证据。
控制节拍拒绝记录补充当前阶段耗时、反馈读取和上一条 trace 写入耗时，便于定位长间隔；
不在原因未知时延长 40 ms watchdog，也不把调度／写盘假设当成已确认原因。
若旧 run 缺少最终结果记录，离线报告只从已落盘的阶段、重力观察和运动窗口恢复证据，
保留 `terminal_record_missing`，绝不将其改成采集完成或自动安装补偿。

结果中的 `workspace` 保存配置组标签、实际完整姿态、换位、重力观察、完成段及未完成段。
`automatic_calibration_report.json` 新增 `workspace_motion`：逐轴保存带完整回执的运动窗口，
包括命令变化范围、实际速度、线性残差及 driver Kd；变化中的闭环命令不会再因不满足
旧“近恒定命令”条件而整段丢弃，也不会直接当成纯摩擦。此阶段不另做静摩擦阈值扫描，
不自动拟合或安装重力/摩擦参数。后续先做相近完整姿态下的正反方向、双速度分析，得到
有效重力修正候选并重新采集，再进入起步摩擦优化。没有外部测力时不能承诺绝对力矩校准。

`setup_label` 默认 `temporary_end_effector`，与每次 run ID、config/URDF 哈希及冻结补偿参数
一起保存。最终末端确定后改用新的配置组重采，不能把不同负载默认为同一拟合数据集。
中心网格及两组阻尼都只通过连接前 profile 调整，批次运行时不改配置。

离线几何检查入口（不执行真机、不覆盖已有输出）：

```bash
python -m scripts.probes.b601_workspace_collision_audit \
  --source-root /path/to/reBotArm_control_py \
  --profile docs/hardware/profiles/b601_j2_j3_workspace_calibration_v1.json \
  --origin-deg 0 0 0 0 0 0 \
  --compare-previous-grid \
  --output /path/to/new-workspace-collision-report.json
```

该工具使用启用碰撞几何的 MuJoCo 场景，复用 owner 的任务顺序和局部往返计划，按不超过
`0.5°` 采样整条名义路径；另用已安装 FCL 核对原始三角网格表面，不复制碰撞过滤规则。
MuJoCo STL 碰撞使用凸包，不能把凸包近似重叠与真实网格表面相交混为一谈；两种结果均保留。
默认不虚构现场地面；只有提供 `--ground-z-m` 才加入该高度的示例平面，不能当成已测量环境。
零位可能无表面相交但小于既有 `2 mm` 预防间距，报告会标为 `CLEARANCE_REVIEW_REQUIRED`，
不会伪称无碰撞余量充足。名义路径检查不覆盖动力学跟踪误差、支撑轴漂移、线缆和最终末端。
该诊断不在硬件控制循环内，不授权运动，也没有关闭或放宽既有碰撞保护。

新模式的完整 `samples` 在停止和后端清理之后逐帧压缩为
`b601_owner.samples.jsonl.gz`，结果 JSON 通过 `sample_storage` 指向它，避免结束时将全历史
展开成巨大的重复 JSON；完整 debug JSONL 仍保留。旧模式输出不变。长批次仍需足够的
内存、磁盘和可靠供电，压缩归档不代表控制期间没有资源开销。

### 四中心 J2/J3 保持采集（显式扩展，不改变近零默认配置）

`b601_j2_j3_four_centres_holding_v1.json` 复用近零已运行的 coordinated holding
控制器，只将中心从 `(3°,3°)` 扩为蛇形 `(3°,3°) → (3°,8°) → (8°,8°) → (8°,3°)`。
每中心两个关节各 12 段，总计 8 组关节采集、96 段；完成后返回本轮实测原位，
确认零附加助力和基础阻尼，并等待 GUI 人工确认最终失能。失败不自动重试或返回。
GUI 应选择「J2/J3 位姿采集」，不是「固定位置」或另一项 20° 完整流程。

整 workspace 时限由 3120 扩为 12480 秒，累计命令暴露由 1040 扩为
4160 N·m·s，仅按四倍中心任务量扩展。每组 1200 秒／480 N·m·s、每段
180 秒／120 N·m·s不变；时间／暴露不会因换候选、制动或重到位而自动续期。
命令暴露不是实测能量。两轴共享 L1 附加助力上限仍为 1.0 N·m，原速度、
位移、阻尼、加减力、保持搜索、40 ms watchdog 和反馈／派发保护不变。
该配置是独立显式 profile；原单中心 profile 和所有程序默认值不变。
采集边界显式设为 0–12°，给 8° 中心保留原 `global_excursion_rad=0.04`（约 2.29°）
扫掠／撤力准入余量；不删除这项连接前校验，也不生成 12° 目标。局部往返仍约 ±0.573°。
启动 owner 的 `--duration-sec` 应覆盖整批时限，例如 14400，避免沿用 3600 秒
导致长批次被外层时限截断；14400 是上限，不是预期采集时间。

几何检查必须使用实际起始六轴姿态，并同时覆盖中心间路径和局部往返。
上一轮 `117zswqu` 起点约 `[-1.428°,0.461°,0.374°,0.901°,1.121°,-2.285°]`；
对应四中心名义路径检查了 74 个唯一配置，无三角表面相交或低于 2 mm 的三角表面间距，
但保留起点／返回近零处既有凸包间距警告，状态为 `CLEARANCE_REVIEW_REQUIRED`。
这不是任意起始姿态、跟踪误差、线缆、支架或最终末端的碰撞保证，也没有放宽严格碰撞模型。
不得据此直接扫描 J2 0–120° × J3 0–90° 整个矩形：较大范围存在基座／夹爪自碰撞。
换末端或其余四轴姿态后需重新检查，不把旧起点的名义报告当成新路径验收。

采集保持原物理配置组标签，不把 profile 文件名当成不同负载。96 条完成记录
仍是完整姿态与软件保持命令的观测，不是 96 个独立中心、唯一重力残差或实测输出力；
不自动拟合／安装补偿。后续以方向配对和跨姿态分析生成候选，再显式启用重采。

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
当前样本可包含未验证 freshness 的 vendor operation-status 缓存力矩，但仍不含经过标定并与
命令严格关联的关节实测力矩或电流，因此不能单独完成质量/质心辨识。

### 离线归约、姿态去重和拟合

真机采集完成后，不要直接把所有 20 Hz 帧当作独立姿态，也不要把第 1～4 组 `--kp 0`
窗口送入拟合。使用同一探针的无硬件模式 `gravity-calibration-fit` 完成以下步骤：

1. 只提取完整 `RECORDING` 区间；每个区间用 q、速度、重力项和位置项的中位数归约为一个
   带 MAD 的姿态观测。含非有限值、绝对力矩限幅或 slew 限幅的窗口会被标为无效。
2. 每次完整的 `R` 记录直接归约成一个静态残差观测。命令提供的姿态距离容差只用于识别
   近似重复姿态并平均，操作者不需要判断或标记接近方向，也不需要反复返回同一姿态。
   旧文件中没有方向标签的完整 `R` 窗口同样可用，只要非零 Kp、稳定性和限幅检查通过。
3. 以 `r = position_tau`、`g0 = gravity_tau / gravity_scale` 拟合有效重力修正；这里的
   `g0` 已包含本次显式 `gravity_joint_scales`，拟合产物会原样记录该向量以及最终六轴有效
   multiplier。第一轮使用 `global-scale`，直接求出可与同一逐关节向量共同回填的
   `--gravity-scale` 有效比例。若独立验证
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
可移植变量；实际交接时必须替换为当前 Task 的绝对路径，并先核对固定的 commit/tree。
不要把 `git -C "$WT" rev-parse HEAD 2>/dev/null` 的输出直接拿去比较：路径不存在、目录
不是 Git worktree、`.git` 不可读时，该写法会隐藏真正错误并把 `ACTUAL_HEAD` 变成空串，
最终产生没有诊断价值的 `actual=`。统一使用下面的入口检查；它把“路径/仓库无法解析”和
“已解析但 immutable pin 不匹配”分开报告，任一失败都只跳过探针并保留当前 shell：

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
EXPECTED_HEAD='replace-with-full-reviewed-commit-sha'
EXPECTED_TREE='replace-with-full-reviewed-tree-sha'

enter_pinned_worktree() {
  if ! test -d "$WT"; then
    printf 'ERROR: worktree directory not found: %s\n' "$WT"
    return 80
  fi

  ACTUAL_HEAD="$(git -C "$WT" rev-parse --verify HEAD 2>&1)"
  STATUS=$?
  if test "$STATUS" -ne 0; then
    printf 'ERROR: cannot resolve worktree HEAD: path=%s status=%s\n' "$WT" "$STATUS"
    printf '%s\n' "$ACTUAL_HEAD"
    return 81
  fi

  ACTUAL_TREE="$(git -C "$WT" rev-parse --verify 'HEAD^{tree}' 2>&1)"
  STATUS=$?
  if test "$STATUS" -ne 0; then
    printf 'ERROR: cannot resolve worktree tree: path=%s head=%s\n' "$WT" "$ACTUAL_HEAD"
    printf '%s\n' "$ACTUAL_TREE"
    return 82
  fi
  if test "$ACTUAL_HEAD" != "$EXPECTED_HEAD"; then
    printf 'ERROR: worktree commit mismatch: expected=%s actual=%s\n' \
      "$EXPECTED_HEAD" "$ACTUAL_HEAD"
    return 83
  fi
  if test "$ACTUAL_TREE" != "$EXPECTED_TREE"; then
    printf 'ERROR: worktree tree mismatch: expected=%s actual=%s\n' \
      "$EXPECTED_TREE" "$ACTUAL_TREE"
    return 84
  fi

  SOURCE_STATUS="$(git -C "$WT" status --porcelain=v1 --untracked-files=all)" || return 85
  if test -n "$SOURCE_STATUS"; then
    printf '%s\n' 'ERROR: worktree contains source changes'
    printf '%s\n' "$SOURCE_STATUS"
    return 86
  fi
  cd "$WT" || {
    printf 'ERROR: cannot enter worktree: %s\n' "$WT"
    return 87
  }
}

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
if enter_pinned_worktree; then
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
  STATUS=$?
  printf '[B601] preflight status=%s\n' "$STATUS"
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
if enter_pinned_worktree; then
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
  STATUS=$?
  printf '[B601] preflight status=%s\n' "$STATUS"
fi
printf '%s\n' 'shell remains active'
```

这一档把所有 host 有限差分阻尼设为零，由六轴 driver Kd 提供电机局部连续耗散；这些数值
只是针对已观察到的 20 Hz 阶梯和腕部静摩擦死区的下一档 commissioning 配置，不是代码
默认值或测试契约。启动锁仍暂时使用 YAML MIT Kp/Kd，但在电机本地闭环内生效并由
`0x700B` 约束完整输出；交接完成后 driver Kp 始终为零，旋转刚度仍在末端 6D spring 中，
经完整 `J^T` 调节所有关节。

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
if enter_pinned_worktree; then
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
  STATUS=$?
  printf '[B601] preflight status=%s\n' "$STATUS"
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
if enter_pinned_worktree; then
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
  STATUS=$?
  printf '[B601] preflight status=%s\n' "$STATUS"
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
if enter_pinned_worktree; then
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
  STATUS=$?
  printf '[B601] preflight status=%s\n' "$STATUS"
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

#### J2/J3 启动静摩擦补偿（默认关闭）

若重力参数已经在无外力状态下验收，但同一 Cartesian assist 在 X 方向持续出现“有明确
`J^T F` 力矩、J2/J3 仍停在静摩擦死区，而 Y/Z 方向能够响应”，可以在原有单次
Cartesian assist 内显式加入 J2/J3 breakaway 补偿。它不是新的控制 owner，也不改变重力
模型；其方向只取自当帧未限幅的 `J^T F` 任务力矩，不使用速度符号猜测运动方向。

对应参数为：

```text
--cartesian-assist-breakaway-positive-nm J1 J2 J3 J4 J5 J6
--cartesian-assist-breakaway-negative-nm J1 J2 J3 J4 J5 J6
--cartesian-assist-breakaway-velocity-fade-rad-s VALUE
--cartesian-assist-breakaway-intent-deadband-nm VALUE
--cartesian-assist-breakaway-intent-ramp-nm VALUE
```

所有关节的正、负方向补偿默认都严格为 `0 N·m`。当前只允许 J2/J3 为非零，其他关节非零
会在使能前失败。正负方向分别配置是为了容纳减速器、线束与装配造成的方向不对称；数值
必须来自独立的现场 breakaway 标定，不能从官方相邻关节摩擦值或本次 Cartesian force
反推。任务力矩绝对值低于 deadband 时补偿为零，随后在 intent ramp 内线性增大；实测有限
差分速度从静止增至 velocity fade 时，补偿线性衰减到零。因此它只帮助脱离静摩擦，不应在
已经运动后继续充当恒定推力。

组合顺序固定为：

```text
J^T F task torque
  + bounded J2/J3 breakaway torque
  -> existing assist authority ramp
  -> existing per-joint absolute overlay cap
  -> existing torque slew limit
  -> gravity and final hardware torque envelope
```

任一 breakaway 标定值大于既有 `--cartesian-assist-max-joint-torque-nm` 会在使能前失败；不能
通过 breakaway 参数绕过原有绝对力矩、slew、速度/加速度、位移或 protective-hold 边界。
JSON 会同时记录 `raw_task_joint_torque_overlay_nm`、breakaway 配置、逐轴 velocity/intent
权重、实际 breakaway 力矩以及合成后力矩，便于区分“任务力不足”“静摩擦补偿生效但被总
包络截断”和“机械臂已经运动、补偿正常衰减”。在完成 J2/J3 正反方向独立标定前，真机命令
必须继续使用全零默认值。

#### Cartesian 助力的非对称力矩变化率

Cartesian assist 的启动推力仍由
`--cartesian-assist-joint-torque-slew-nm-s` 缓慢加入，限制主动助力注入能量的速度。现场反向
试验表明，若同一个很小的 slew 同时用于撤销旧力矩，TCP 越过目标后即使 `J^T F` 已经要求
反向制动，实际发送的关节 overlay 仍可能在数个 20 Hz 周期内保持旧符号。为避免把安全的
慢启动错误地复用成迟滞制动，可显式配置：

```text
--cartesian-assist-joint-torque-release-slew-nm-s VALUE
```

该参数只控制逐关节的快速撤销与耗散制动。请求力矩与可信有限差分关节速度方向相反、且速度
绝对值超过既有 `--cartesian-assist-stationarity-limit-rad-s` 时，才允许用 release rate 接近
制动力矩；这避免把静止附近的位置差分噪声误判为可快速施加的新力矩。请求只是在非耗散条件
下换符号时，先用 release rate 把旧力矩撤到零，新方向的加力仍回到慢 rise rate。未传入该
参数时，它严格等于 rise rate，输出与历史对称 slew 行为一致。

这不是第二套限幅器：实现仍复用现有 `condition_torque()`，只是为每个关节选择 conditioner
目标和 rate。原有 assist 绝对力矩上限、最终总力矩包络、速度/加速度和 Cartesian 位移/姿态
guard 均保持不变。每帧审计新增 `conditioning_target_joint_torque_nm`、
`joint_torque_slew_mode` 和 `joint_torque_selected_rate_nm_s`，必须结合
`authority_joint_torque_overlay_nm` 与最终 `joint_torque_overlay_nm` 判断制动是否及时；不能通过
提高所有方向的 rise rate 来掩盖旧力矩撤销迟滞。

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

这条约束也适用于调参和保护路径：当前 B601 owner 的 Kd、Coulomb friction、速度/加速度
守护、反转判断和 protective-hold admission 均只接受
`normalized_qpos_finite_difference_over_feedback_dt`。MotorBridge operation-status cache
中的 vendor velocity 只作为带有 `diagnostic_only_not_used_for_control` 标记的被动证据，不能
被解释为已经验证的 rad/s。Pinned `reBotArm_control_py` 子模块保持原始 gitlink 不变；其中
旧标定脚本的 `mechVel/0x701A` 兼容性字段不属于 B601 控制契约，不能作为本项目控制或验收依据。

启动诊断可设置 `MOTORBRIDGE_DEBUG_MODE=robstride-startup` 和
`MOTORBRIDGE_DEBUG_LOG=<path>`，被动保存 type-1 operation-control、type-2 operation-status
与 type-17 参数帧；
它不读取 `0x701A/0x302C`，也不改变控制总线负载。原始 type-2 与 JSON 中的 vendor cache
可用于检查解码和时间邻近关系，但在缺少源序号时仍不能声称逐命令严格配对。

另外，`compute_b601_mit_torque()` 若已经组合软件 damping，再同时给驱动器非零 Kd，会重复
计算阻尼。正常 active 只允许一种明确拆分：软件 Kd 为零，driver Kp 为零，driver Kd 单独
拥有电机局部速度耗散，软件拥有 gravity 和 `J^T wrench`，驱动器 `0x700B` 拥有最终总力矩
绝对上限。只有上述临时启动/回放锁可使用非零 driver Kp；其 host Kp/Kd 必须为零，并在
退出时连续恢复 active 增益。除此之外的非零 driver Kp、软件与驱动器双重 Kd、以及把约
20 Hz host loop 宣称为高频 haptic 仍然禁止。真正高带宽 task-space impedance 仍需可验证
freshness 的异步 q/dq 解码路径。

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
