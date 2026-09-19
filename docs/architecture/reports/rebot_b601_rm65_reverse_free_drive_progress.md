# RM65 free-drive → B601 MuJoCo 反向遥操进度

## 当前切片：P0 只读观测、纯 retarget、P1 仿真装配与 P3 前台运行环

本切片面向“RM65 真机手动拖动/free-drive，B601 在 MuJoCo 中跟随”的第一段
链路。它只增加适配层，不改变既有 B601 真机控制、RM65 真机执行、MuJoCo
位置驱动或控制组合 authority。

### 已落地

- `control_station.sources.rm65_free_drive.RM65FreeDriveSource`
  - 复用现有 `RM65ArmBackend.refresh_state_from_hardware()`。
  - 只接受 `observation_only`、`hardware_observation_only`、完整且成功的
    hardware qpos/FK pose、递增 measurement sequence 和 `local_wall_time`。
  - 同时检查 wall-time measurement age 与 monotonic receive age；坏帧、重复
    sequence、未来时间、过期状态和非刚体 pose 均当前 tick fail-closed，不回退
    到上一帧。
  - 不启动后台流、不调用 RM65 命令、不拥有 GUI、dispatch 或 episode authority。

- `robot_motion.rebot_b601_rs.rm65_retarget.RM65ToB601Retargeter`
  - 复用已有 B601/RM65 `CoordinateFrameGraph`，通过图的逆向解析实现
    `RM65 right_arm_base/tool_tcp → B601 base_link/gripper_end`，没有复制矩阵
    约定或另建坐标系。
  - 采用显式 source/target anchor、平移/旋转 gain、freshness 和连续 seed；连续
    seed 由桥接层保存的上一条已接受 IK qpos 提供，不使用可能因位置伺服滞后而落后的
    MuJoCo 实测 qpos。
  - 复用 `B601DampedLeastSquaresIK`；IK 失败、过期样本、重复 sequence 或
    frame/clock 不匹配只返回 rejection，不产生旧 qpos fallback。
  - 输出仍是纯 qpos candidate；不直接调用硬件或 MuJoCo。后续应用层应把成功
    candidate 交给现有 `B601PositionDriver`/target runtime 的唯一执行 owner。

- `control_station.sim_hil_profile.rm65_b601_free_drive.RM65ToB601FreeDriveBridge`
  - 复用现有 `B601PositionDriver` 的物理步、500 Hz 命令节拍、位置 slew、限位
    和 fault latch；桥接层不写 `MjData`，不复制 position servo。
  - 只负责 `inactive → active → clutched → active/held` 的 anchor、clutch 和
    recenter 状态；激活/重定位都要求一帧新的 RM65 观测。
  - 目标 qpos 只在新样本通过 frame/age/IK 后排队；source、IK 或 driver 拒绝
    会撤销 pending target 并进入 `held`，不会重放上一条 qpos。
  - `step()` 是一个 MuJoCo physics tick；调用方仍负责按现有 driver 的 1 kHz
    physics 时钟推进，不把 RM65 20 Hz 观测误当成新的仿真执行频率。
- `build_rm65_to_b601_free_drive_bridge()` 直接加载现有 B601 模型、现有
  mapping profile、现有 Pinocchio FK/IK 和现有 MuJoCo scene/driver。反向平移
  与旋转 gain 在 `RM65ToB601FreeDriveSimConfig` 中显式给出，不隐式反用正向
  150% profile。

- `control_station.sim_hil_profile.rm65_b601_free_drive_foreground`
  - 复用既有 `B601RM65TerminalCommands` 的无 Enter 单键读取语义，并由新的
    反向前台协调器统一管理 `MujocoPassiveViewerSession` 生命周期。
  - `a/c/r/s/q` 只改变桥接层的 inactive/active/clutched/held 状态；物理环仍按
    B601 既有节拍推进，native viewer 默认独立以 60 Hz 刷新，不再让 GUI vsync
    把物理环压到约 185 Hz。
  - live application 的 RM65 source poll 按 monotonic wall clock 调度（默认
    20 Hz），不再使用被 viewer 拖慢的仿真时间调度；synthetic tests 默认保留
    确定性的仿真时钟调度。
  - 当前候选增加可选的 `source_poll_mode=background`：RM65 只读刷新由一个可关闭的
    source worker 负责，桥接/IK/position-driver 主循环只按顺序消费完整结果，不再被
    SDK 读取长尾阻塞。worker 使用固定 start-to-start deadline；坏帧、worker 异常或
    mailbox overflow 都发布为显式 rejection 并进入 `held`，不会静默复用旧 qpos。
    `source_poll_mode=synchronous` 仍保留用于离线对照和故障复现。
  - viewer 关闭、Ctrl-C、EOF 和 source/IK admission rejection 都会撤销
    pending target 并退出/保持 held；退出只关闭仿真 viewer，不触碰 RM65 或
    B601 真机。

- `python -m control_station rm65-b601-free-drive`
  - 在同一应用入口中组装 `RM65ArmBackend(observation_only=True)`、只读
    `RM65FreeDriveSource`、反向 B601 MuJoCo bridge 和 native passive viewer。
  - `--validate-only` 只加载 URDF、mapping、MuJoCo 和 Pinocchio；
    `--connect-read-only` 才建立 RM65 观测连接，并要求显式 confirmation、IP
    和 URDF。该模式不启用 RM65、不发送位置/力矩/停止命令，MuJoCo 目标也不
    是真实 B601 执行路径。
  - 连接、freshness、anchor、clutch/recenter、held 和退出事件统一写入传入
    的 JSON/log；不再依赖 shell 维护交互状态。

### 自动化测试证据（不是现场验收）

```text
tests/test_control_station_rm65_free_drive.py     5 passed
tests/test_robot_motion_rm65_retarget.py          5 passed
tests/test_control_station_rm65_b601_free_drive.py 6 passed
tests/test_control_station_rm65_b601_free_drive_foreground.py 3 passed
tests/test_control_station_rm65_b601_free_drive_application.py 2 passed
tests/test_rebot_b601_rm65_free_drive_integration.py 3 passed
ruff (changed files)                               PASS
git diff --check                                   PASS
```

这些证据是 fake backend/纯数学测试，不代表 RM65 真机连接、free-drive 状态或
MuJoCo GUI 已验收。

当前候选的合并前相关切片（包含上述应用入口测试、集成、前台和边界测试）为
`66 passed`；该数字仍只表示离线自动化证据，不表示现场硬件验收。

## 现场延迟与 held 诊断（2026-09-19）

现场一次运行中，MuJoCo 日志约 49 秒只推进了 9075 个 physics step（约
185 Hz），而命令请求的是 1000 Hz。原因是旧前台循环在每个 physics step 都调用
native viewer `sync()`，并且每条 JSON 事件都立即 flush；同时 source poll 以
仿真时间计时，使 20 Hz source 在该现场实际只有约 3--4 Hz。RM65 读取和连续 seed
IK 也在同一前台 tick 同步执行，因此会进一步造成可见延迟。

当前实现已将 viewer 刷新和 source poll 时钟解耦，并将日志 flush 批量化；每个
`bridge_step` 仍保留 source/retarget/position/total elapsed 字段，所以不会以
降低证据粒度换取速度。下次现场日志应重点统计这些字段，不能仅凭终端刷新速度
判断 IK 性能。

同一现场日志还显示先有一条 `source_admitted=true`、`retarget_accepted=false`
的 tick，之后桥接器进入 `held`，最后操作者才按 Ctrl-C。它不是自动断开或 RM65
网络断线，而是当前目标被 retarget/IK 门拒绝后按 fail-closed 契约撤销 pending
target。此前前台只打印了泛化的 `bridge rejected the target`，无法知道是
`ik_rejected:*`、source freshness 还是 driver rejection。现在日志和首次/周期性
held 提示会记录 `bridge_rejection_reason`、IK status/iterations/errors、IK seed/结果
qpos、source measurement flags/sequence 以及各阶段耗时；在拿到这一证据前，不应放宽
workspace、freshness 或 IK 安全门。

### 现场延迟测量契约（RM65 qpos → B601 MuJoCo target）

仿真目标限速的默认值已按现场观察调整为 `5.0 rad/s`。它是 MuJoCo
`B601PositionDriver` 的可调、仿真专用 target slew limit，不改变 RM65 或 B601
真机的任何速度/力矩限制；现场仍可通过
`--target-velocity-limit-rad-s` 显式覆盖。

每个 `bridge_step` 事件现在额外记录：

- `source_receive_monotonic_sec`：本次被接受的 RM65 硬件读状态完成时间戳；
- `source_step_started_monotonic_sec` / `source_step_completed_monotonic_sec`：一次
  source refresh 从进入到完成 freshness/frame 校验的时间边界；
- `source_admission_monotonic_sec`：该 source sample 完成校验并准入桥接器的时间戳；
- `target_dispatch_monotonic_sec`：该 retarget qpos 被 B601 MuJoCo position
  driver 接受的时间戳；
- `source_to_target_dispatch_latency_sec`：上述两者之差；只有真实发生
  `command_applied=true` 时才有值；
- `source_admission_to_target_dispatch_latency_sec`：从 source step 完成准入到
  target driver 接受的延迟。它与前一个字段同时保留，用来区分硬件 read 时间戳到
  source 状态组装/校验的耗时；
- `source_poll_deadline_sec` / `source_poll_lateness_sec`：source 固定 deadline
  及本次实际开始读取时的迟到量；`source_poll_schedule_skipped=true` 表示读取跨过
  了至少一个 deadline，调度器跳过 missed period 后重新建立 cadence；
- `source_state_read_phase_timing`：RM65 backend 的 SDK、解析/校验、重试等读状态
  阶段耗时，用于把 source 读取成本与 retarget/driver 成本分开。

因此原主指标测量的是“RM65 backend read timestamp → B601 MuJoCo target driver 接受”，不把
RM65 反馈在读之前已经积累的 measurement age、MuJoCo viewer 刷新或仿真机械臂的
物理响应混入。它不是 B601 真机执行延迟；真机阶段还必须独立记录 CAN/driver
ack 与实际反馈。新增的 `source_admission_to_target_dispatch_latency_sec` 则把
freshness/frame 校验和 source 状态组装之后的准入点作为起点，避免将两个边界混称为
单一“SDK 返回后延迟”。

作为新字段上线前的基线，2026-09-19 的现场日志（尚未记录主指标）显示：RM65
读状态间隔平均 `57.04 ms`、p95 `60.76 ms`；source read+校验平均 `6.33 ms`、
p95 `8.97 ms`；连续 seed IK 平均 `0.919 ms`、p95 `1.081 ms`；position step
p95 `0.137 ms`。因此在该日志中“读完成后到目标接受”只能估算为约 `7--10 ms`
（不能当作精确验收值），而整体操作者动作到下一次目标的主要等待来自约 `57 ms`
的 source 间隔。新日志必须使用 `source_to_target_dispatch_latency_sec` 取代
这个估算。

现场日志可用下面的离线报告得到 p50/p95/p99/max，并同时查看 source read、IK
retarget 和 position-driver 各阶段：

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/probes/report_rm65_b601_reverse_latency.py \
  "$RUN/gui.log" \
  --output "$RUN/reverse_latency_report.json"
```

判读顺序固定为：先比较 `source_to_target_dispatch_latency_sec` 与
`source_admission_to_target_dispatch_latency_sec`，再看 `source_poll_lateness_sec`、
`source_step_elapsed_sec` 与 `retarget_elapsed_sec`。source 调度现在使用固定
start-to-start deadline；读取跨过 deadline 时只跳过 missed period，不追赶 burst。
若主指标稳定而 `sim_slew_limited` 比例高，瓶颈是 target slew；若 source read 或
retarget p95 接近/超过 source period，才考虑 latest-only 读取/IK worker。任何 worker
仍必须在主线程重新检查 sequence、freshness、frame 和 fail-closed rejection，不能用
旧 qpos 或静默重试掩盖延迟。

当前 P0/P1 候选已经落地但尚未做新的现场采集：P0 将 read、source admission
和 target dispatch 三个时间边界同时写入 `bridge_step`；P1 将 source deadline
从“本次读取完成时间”改为固定 start-to-start cadence。若 SDK 读取跨过一个或多个
deadline，候选只跳过 missed period 并从当前时钟切片重新建立下一周期，不会为了追赶
而连续发送读取请求。该调度改变不放宽 freshness、sequence、frame、IK 或 target
driver 的任何安全门。

这次具体拒绝是一次有效样本上的 `ik_rejected:line_search_stalled`：source 已准入、
Jacobian 条件数约为 10，随后才进入 `held`。首个假设是上一版每次用 MuJoCo 实测
qpos 作为 IK seed；位置伺服的动态/目标 slew 滞后使 seed 落后于最近已接受目标，连续
拖动时容易跨入不利的局部数值 basin。桥接器已改为“首次激活或 recenter 使用实测
qpos，之后使用上一条已接受 IK qpos”，不改变 source freshness、joint limits、
driver slew 或 fail-closed rejection 契约。

随后对 2026-09-19 的最新现场日志进行了精确复现：新 seed 已经是上一条已接受目标，
source sequence/时间戳也完整新鲜；失败发生在 IK 第 0 次迭代。该目标的初始位置误差
约 0.0107 m、姿态误差约 0.0681 rad、Jacobian 条件数约 9.6，但 DLS 的主导 J4 步约
为 -0.110 rad。旧的 0.05 rad 数值步长上限把它截断后，全部回溯候选都不能降低 merit，
因此错误地进入 `line_search_stalled`。默认数值步长现调整为 0.10 rad；这不是硬件命令
或 MuJoCo 执行速度限制，物理关节限位、位置 driver slew 和 fail-closed 仍保持不变。
该精确 seed/target 已加入 IK 回归测试。后续若仍出现相同状态，应根据每条事件中的
`ik_seed_qpos_rad`、目标误差和 `ik_iterations` 判断是否是新的数值条件，而不能再把它
解释成 RM65 网络断线。

当前候选只把 RM65 source read 移出物理主循环；IK、目标选择、position-driver
admission 和 dispatch 仍由同一个桥接主循环按顺序完成，避免引入第二个 IK 或目标
authority。新增 `source_completion_to_consumption_latency_sec` 和
`source_completion_to_target_dispatch_latency_sec` 用于识别 worker mailbox 等待，
并继续保留 freshness、sequence、frame、IK 和 fail-closed admission。报告中完成→消费
只在 worker 结果实际被消费的低频 tick 计数，完成→目标派发只在 position driver 接受
命令的 tick 计数，避免同一源样本重复统计。该 worker 不是旧样本回退机制，也不改变
RM65 的 observation-only 约束。

### 2026-09-19 40 Hz 完整反向桥接基线

在远端 `user@192.168.70.116` 的隔离候选工作树中完成了 30 秒、40 Hz source / 500 Hz
MuJoCo physics 的完整桥接运行（`null_viewer`，因此不包含 GUI presentation latency）。
RM65 处于只读观测，B601 仅为 MuJoCo position driver；没有 RM65/B601 真机运动命令。

```text
source samples:                  1190
target dispatches:               1187
rejection reasons:               none
source interval mean/p95/p99:    25.20 / 31.62 / 35.88 ms
read completion -> dispatch:     2.58 ms mean, 4.00 ms p95, 4.88 ms max
retarget/IK:                     0.837 ms mean, 1.078 ms p95, 1.472 ms max
SDK read:                        7.28 ms mean, 12.38 ms p95, 18.53 ms p99, 45.22 ms max
source schedule skips:           5
maximum source interval:         107.97 ms
```

该基线确认 IK 不是主要延迟来源；异常长尾来自 RM65 SDK read。同步主循环在这些
长尾上会丢 source deadline，因此当前候选默认现场应用改用 background source poll。
随后在同一远端硬件、同一 500 Hz MuJoCo null viewer 上以 background source poll 重测
30 秒：source samples 1199、target dispatches 1199、拒绝 0、schedule skips 0；源间隔
mean/p95/p99 为 25.002/29.074/31.157 ms，SDK read wall mean/p95/p99/max 为
4.488/8.254/10.309/14.427 ms，完成→消费 mean/p95/p99/max 为
0.975/1.567/1.598/1.944 ms，完成→派发 mean/p95/max 为 2.540/3.662/3.735 ms。
该轮仍然不包含 GUI presentation latency 或物理执行器响应，但表明把 SDK read 移出
物理主循环后，主循环不再承受 4–14 ms 的同步读阻塞，且 40 Hz 源调度没有丢期。
候选仍未合并或安装。

### 2026-09-19 GUI 40 Hz 只读验收

在同一远端候选上以 `DISPLAY=:1` 启动 MuJoCo passive viewer，自动执行一次激活，运行
约 30 秒后自动退出。RM65 使用 observation-only backend，B601 只运行 MuJoCo
position driver；`rm65_commands_sent=false`、`b601_hardware_commands_sent=false`，
没有任何真实执行器命令。

```text
execution status:                COMPLETED
source samples / dispatches:     1170 / 1170
rejection reasons:               none
source interval mean/p95/p99:   25.002 / 29.100 / 31.180 ms
source poll lateness p99:        0.109 ms
read completion -> dispatch:     3.842 ms mean, 8.334 ms p95, 10.470 ms max
worker completion -> consumption:
                                  1.623 ms mean, 5.712 ms p95, 7.044 ms max
retarget/IK:                     0.947 ms mean, 1.272 ms p95, 1.411 ms p99
viewer bridge steps:             10213 in 30.287 s
```

GUI 验收证明 viewer、后台 40 Hz source poll、连续 IK 和 MuJoCo target dispatch 可以
共同工作；但 passive viewer 的同步/呈现负载使实际 physics loop 低于名义 500 Hz，
因此 GUI 结果不能替代 null-viewer 的时序基准，也不能代表真实 B601 执行器频率。
后续若需要量化 viewer 性能，应单独测量 `viewer_sync_hz`，不要把 GUI 步频当成
RM65 观测频率或硬件控制频率。

### 2026-09-19 请求-响应延迟定位与可选 UDP 观测路径

对远端 40 Hz GUI 运行的原始 `source_read_phase_timing` 记录进行了逐条分析。1480 次
有效读取中，锁等待和解析验证都在亚毫秒量级，重试次数始终为 1，`retry_sleep` 始终为
零；而 RealMan SDK 请求-响应阶段的 wall time 为：

```text
sdk_call_wall_ms: mean 6.545 / p95 10.357 / p99 43.189 / max 49.335
total_wall_ms:    mean 6.585 / p95 10.386 / p99 43.225 / max 49.379
```

22 次 source schedule skip 全部对应 SDK wall time 超过 15 ms。它们不是 IK 或锁竞争
造成的：retarget/IK 的 mean/p95/p99 约为 `0.979/1.287/1.464 ms`，source poll
lateness 的 p99 约为 `0.13 ms`。因此 background poll 已经解决了主循环被同步 SDK
调用阻塞的问题，但请求-响应协议本身仍然决定 RM65 新鲜观测的延迟尾部；单纯提高
MuJoCo 频率、放宽目标速度或继续微调 IK 不能消除这部分延迟。

候选现在增加了一个**可选** `realtime-push` source transport：

- `RM65FreeDriveRealtimePushSource` 只保留 vendor UDP callback 的最新完整六轴样本；
  callback 不做 FK、不运行 IK，也不持有第二个 target authority。
- FK、freshness、错误码、sequence 递增和 fail-closed admission 仍在原有 bridge
  owner 线程完成，保持现有 source/target 安全边界。
- `RM65ArmBackend.start_realtime_push_observation()` 默认只查询已有 UDP 配置并注册
  callback；不会写控制器配置、使能电机或发送运动命令。
- 只有同时传入 `--configure-realtime-push`、目标 IP 和专用确认字符串
  `RM65_B601_READ_ONLY_CONFIGURE_TELEMETRY` 才会写 UDP telemetry 配置。该操作仍是
  观测配置，不是运动命令；在真机上必须由操作者单独确认并保存前台日志。
- CLI 的 `--realtime-push-cycle-ms` 使用毫秒表达；适配器会按官方协议转换为
  控制器 `cycle` 字段的 `5 ms` 单位计数，并在显式重配置时保留原有 custom telemetry
  字段，避免把无关状态项恢复成默认值。
- 默认 `request-response` 路径保持不变，可直接回退。新路径的故障（无样本、重复
  sequence、vendor error、回调字段不完整）均进入 source rejection，不使用旧样本
  冒险跟随。

该接口复用 RealMan API2 已有的 UDP 状态主动上报、三线程模式和实时状态 callback，
不是重新实现控制协议。官方 API2 文档也要求三线程模式和正确的 UDP 目标/防火墙配置；
因此现场验收前先做只读配置查询，再决定是否进行显式 telemetry 配置。当前只完成离线
source/application 测试，尚未在真机上启用或配置 UDP push。

### 下一轮延迟验收

先在远端只读运行 `--source-transport realtime-push`、不带
`--configure-realtime-push`，确认控制器已有 push endpoint 且结果中的
`source_transport=realtime-push`、`realtime_push_config`、source interval 和
completion-to-dispatch 均连续。如果配置未启用，命令应明确失败并保持无运动命令；
不得把失败伪装成 request-response 回退。只有在操作者确认后，才单独运行带专用确认的
telemetry 配置命令，再重复同一 40 Hz 只读验收并比较 p95/p99。比较对象必须包含：

```text
request-response: sdk_call_wall_ms, source_to_target_dispatch_latency_sec
realtime-push:    callback arrival interval, source_to_target_dispatch_latency_sec
```

两者都不包含物理执行器响应和 viewer 呈现延迟；这些仍需在后续双向真机阶段单独测量。

### P2 synthetic MuJoCo 自动化测试

已增加 `tests/test_rebot_b601_rm65_free_drive_integration.py`。该用例使用真实
`reBotArm_control_py` URDF、真实 Pinocchio B601 模型、真实 MuJoCo scene 和
现有 `B601PositionDriver`，但 RM65 输入是确定性的 synthetic observation，不
打开 CAN。它验证：

- 通过现有 mapping profile 的反向 frame graph 生成 RM65 6D 样本；
- anchor 后连续 seed IK 和 position-driver command timing；
- 目标 qpos 最终跟踪误差小于 `5e-4 rad`；
- source rejection 后进入 `held`，撤销 pending command，不重发旧目标。

现场有 vendor 源目录时的显式命令为：

```bash
DEXTELEOP_REBOT_SOURCE_ROOT=/absolute/path/to/reBotArm_control_py \
PYTHONDONTWRITEBYTECODE=1 bash scripts/run_pytest.sh -q -p no:cacheprovider \
  tests/test_rebot_b601_rm65_free_drive_integration.py
```

本候选已用 `/home/zc/HandMotion/Developing/reBotArm_control_py` 作为只读输入，
桥接集成结果为 `2 passed in 1.65s`；加入前台 runner 的组合回归后，
`tests/test_rebot_b601_rm65_free_drive_integration.py` 为 `3 passed in 1.98s`。
该路径没有写入代码或测试默认值。

## P3 自动化测试

新增前台运行环的离线测试：

```bash
PYTHONDONTWRITEBYTECODE=1 bash scripts/run_pytest.sh -q -p no:cacheprovider \
  tests/test_control_station_rm65_b601_free_drive_foreground.py \
  tests/test_control_station_boundaries.py
```

该测试用假的 bridge/viewer，只验证生命周期、无 Enter 命令语义、held/retry
和 viewer 关闭；它是测试，不代表 MuJoCo native GUI 或 RM65 真机已验收。

## 现场验收（尚未执行）

自动化测试通过后，现场验收必须另行执行并保存前台日志、JSON 结果和操作者观察：

1. 在 `user@192.168.70.116` 的桌面会话中，使用
   `python -m control_station rm65-b601-free-drive --connect-read-only`，明确
   传入 RM65 右臂 IP、`rm65_arm.urdf`、B601 source root、task config 和 mapping
   profile；不得猜测现场网络配置，也不得把 RM65 command/stream owner 放进此
   模块。SSH 非交互会话没有 `DISPLAY` 时只可做 validate-only/诊断，不能冒充
   native GUI 验收。
2. 现场操作者先确认 RM65 失能/free-drive、无负载和急停可用，再启动只读观测；
   通过 `a` 激活、手动拖动、`c/r` 离合重定位、`q` 或 Ctrl-C 退出。
3. 验收记录必须包括：RM65 实际反馈连续性、source age、anchor/recenter、
   MuJoCo 末端位移/旋转方向、viewer 关闭、异常/过期帧进入 held，以及退出后
   RM65 未收到运动命令。自动化测试通过不能替代这些现场验收项。

## 真机边界

当前提交没有打开 CAN、没有启动 RM65 free-drive、没有发送 B601 或 RM65 运动
命令。远端隔离 worktree 已完成同一候选的自动化测试，但尚未进行现场验收。后续
真机验收必须由现场操作者前台执行，并单独记录：硬件读状态、observation-only
配置、source age、anchor/recenter、MuJoCo target 轨迹和急停/退出行为；离线通过
不能替代这些证据。

## 当前 frame calibration 固定值

正向 B601→RM65 与本反向 RM65→B601 共用同一份 canonical profile：
`docs/hardware/profiles/b601_rm65_mapping_150_v2.json`。现场已确认并固定：

```text
controlled RPY (roll pitch yaw): 0 90 0
reference RPY (roll pitch yaw): 0 0 90
```

profile 对应的旋转矩阵是 `R_y(+90°)` 与 `R_z(+90°)`，base 平移仍是 URDF-derived
`(-0.2625, 0, 0)`，正向平移 gain 仍为 `1.5 1.5 1.5`。反向 retargeter 通过同一
`CoordinateFrameGraph` 解析严格的逆向边，不把正向矩阵直接套到 RM65 source；反向
translation/rotation gain 由 `RM65ToB601FreeDriveSimConfig` 独立提供。

该固定值只改变 frame calibration，不改变 B601 真机 owner、RM65 真机执行器或任何
CAN/位置/力矩命令。离线反向集成测试必须验证 round-trip、连续 seed 和 rejection
撤销；现场 `--connect-read-only` 仍然只读 RM65，并只驱动 B601 MuJoCo target。

## 2026-09-19 realtime-push `max_iterations` 诊断与修复候选

远端 `user@192.168.70.116` 的只读 realtime-push 运行中，首次拒绝发生在
`/tmp/rm65-b601-field/realtime-push-Qffh2X/gui.log` 的 source sequence `5023`：

```text
source_state_measurement_valid = true
source_completion_to_consumption = 0.337 ms
target_seed_translation_delta = 0.409 mm
target_seed_orientation_delta = 5.244 mrad
ik_iterations = 120
ik_position_error = 0.146 mm
ik_orientation_error = 0.534 mrad
ik_sigma_min = 2.75e-6
ik_condition_number = 1.084e6
ik_limit_projection_applied = false
```

因此这次不是 UDP 丢包、source freshness、关节越限或目标突变；它是 B601
Jacobian 进入近奇异状态后，误差已经很小但无法继续降到严格 `10 um / 0.1 mrad`
阈值。重复的 `held` 行是 bridge 的 fail-closed latch，不是每个 physics tick
重新执行一次 IK。把 `max_iterations` 提高到 3000 只会增加耗时，残差仍约为
`0.146 mm / 0.534 mrad`。

当前候选在 `robot_motion/rebot_b601_rs/ik.py` 增加了明确的有界近似收敛契约：
当状态为 `max_iterations` 或 `line_search_stalled`、没有发生 joint-limit
projection，且残差不超过 `0.5 mm / 1 mrad` 时，结果为
`converged_approximate` 并可继续 MuJoCo following；数值失败、越限投影和更大残差
仍然拒绝。`rm65_retarget.py` 同时记录失败候选 qpos，便于区分奇异残差与真正不可达。
该阈值是仿真桥接的明确数值边界，不是放宽 B601 真机硬件关节限位或发送未验证的
硬件命令。

离线检查结果：

```text
tests/test_rebot_b601_ik.py + tests/test_control_station_rm65_b601_free_drive.py
12 passed, 9 skipped
reverse-latency related suite
30 passed, 1 warning
exact realtime-push target regression: converged_approximate
```

`9` 个 skip 仅表示当前候选 checkout 的 gitlink vendor URDF 未初始化；精确现场
目标已用只读 `/home/zc/HandMotion/Developing/reBotArm_control_py` 独立复现通过。
候选尚未合并、push 或安装；合并前仍需在包含 vendor checkout 的环境重跑 IK/P2
集成测试，并用前台只读 GUI 验证连续跟随不会因该近奇异帧立即进入 held。

## 2026-09-19 延迟主路径诊断与 P1 优化

本轮优先处理延迟，不把近奇异 IK 拒绝当作主要延迟来源。远端 realtime-push
日志 `/tmp/rm65-b601-field/realtime-push-Qffh2X/gui.log` 的离线重算结果为：

```text
source_poll_hz=100 Hz
push callback -> source admission: mean 3.701 ms, p95 4.477 ms, max 5.267 ms
source admission -> target dispatch: mean 3.511 ms, p95 7.650 ms
source callback -> target dispatch: mean 7.212 ms, p95 11.472 ms
retarget/IK: p95 about 0.871 ms (accepted samples)
```

因此此前的主要瓶颈是 realtime-push latest mailbox 以 100 Hz 被消费，平均等待下一次
消费，而不是 IK 计算本身。报告脚本现在新增
`source_receive_to_admission_latency_sec`，并分别记录 `viewer_sync` 和
`control_tick_lateness_sec`，可以把 push 消费、MuJoCo viewer 阻塞和控制线程抖动分开。

P1 候选的改动保持单一控制所有者和 fail-closed 语义：

- realtime-push 的重复但仍新鲜的 latest packet 变为显式 `unchanged` no-op，不再被
  当成 `measurement_not_advancing` 故障，也不会撤销当前 MuJoCo 目标；真正的错误码、
  非法帧和过期 packet 仍立即进入 held；
- realtime-push 的后台 poller 使用 latest-only 消费，避免消费队列追赶旧样本；
- CLI 未显式指定 `--source-poll-hz` 时，request-response 仍为 20 Hz，realtime-push
  默认改为 500 Hz；显式值仍完全尊重；
- latency report 增加 push mailbox、viewer sync 和 foreground tick lateness 分段，
  便于用同一份现场日志判断优化是否生效。

现场复测应优先比较 `source_receive_to_admission_latency_sec` 和
`source_to_target_dispatch_latency_sec`，而不是只看 viewer 画面。预期在 5 ms
realtime-push 周期、500 Hz 消费下，前者应降到约 0--2 ms；若仍高，才继续检查
RM65 UDP 接收线程或操作系统调度。该候选目前仍未合并、push 或安装。
