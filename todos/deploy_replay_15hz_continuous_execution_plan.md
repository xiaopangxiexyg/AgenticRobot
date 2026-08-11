# Deploy / Replay 15 Hz 连续执行计划

## 1. 目标

15 Hz 只定义 source action / keyframe 的时间格点，不定义机械臂每 66.7 ms
停车一次。目标执行链应当满足：

```text
15 Hz source keyframes
  -> 多帧 Pose→QPOS path lift（Pose 路径）或直接 qpos waypoints
  -> 连续时间 q / qd / qdd trajectory
  -> 120 Hz command cut
  -> collision / joint / dynamic admission
```

连续平移或旋转输入下，15 Hz 边界前后的 `qd`、`qdd` 应连续；只有真正的 source
停止、计划覆盖耗尽、安全故障或人工 Hold 才允许进入零速度。

本计划不修改现有真机默认配置。新实现必须是独立 default-off profile，旧 deploy 和
replay 路径保留为原子回退。

## 2. 当前实现与问题定位

### 2.1 已有的正确基础

- `PositionDeployExecutor` 已区分 15 Hz source、reference clock 和 command clock；
- `TimestampedMotionTarget` 已区分 `timestamp` 与 `available_timestamp`，可以表达
  “未来逻辑目标已经由同一 action chunk 提前给出”；
- `CausalMotionResampler` 支持：
  - 无未来输入时延迟一个 source period 后插值；
  - 同 chunk future 已可用时零额外 source-period 延迟插值；
- 当前默认 deploy 的 official all-IK governor 已经在 source Pose 上枚举候选、选择
  collision-safe branch，并在 120 Hz 使用有限 horizon 的 jerk 约束；
- replay 已能按真实 action timestamp 把 15 Hz qpos 线性重采样到 60 Hz，再通过共享
  command interpolator 产生 120 Hz qcmd。

这些组件说明不需要新增生产入口，也不需要把时间律放进 `teleop_runtime` 或
`control_station`。共享轨迹语义应继续归 `robot_motion` 所有，`teleop_runtime` 只绑定
policy/replay source、真机 dispatch 和 GUI。

### 2.2 仍会产生 15 Hz 走停的原因

当前链路仍有四个结构问题：

1. **源 qref 是分段线性的。**
   `RM65OfficialAllIKGovernorPlanner._qref_at()` 对相邻 15 Hz all-IK qpos 做线性插值。
   位置连续，但 knot 处速度一般不连续；jerk governor 只能追踪并缓和这个折线，不能
   把它变成天然连续的源路径。
2. **有限 future 覆盖末端会变成 flat tail。**
   当前 executor 只显式携带 current、lookahead、second lookahead。若下一 chunk 未到，
   horizon 后段只能保持最后目标。优化器会合理地提前制动；如果这种覆盖不足持续发生，
   主观表现就是 15 Hz 加速—减速。
3. **每个新 source interval 重新规划。**
   虽然内部状态保留 `q/qd/qdd`，但新的目标折线和 horizon 终端假设会周期性改变，可能
   在 source cut 处形成 jerk 或加速度包络。
4. **现有 qpos replay 是双重分段线性。**
   15→60 Hz 先线性插值，60→120 Hz 又从上一已发 command 向新 reference 线性追赶。
   它保持位置连续，但不保证 knot 处速度、加速度和 jerk 连续。

因此，“把 reference_hz 从 15 提到 60/120”本身不能解决问题。要改变的是 trajectory
合同，而不是只增加采样点。

## 3. 不可绕过的因果边界

对于只在当前时刻输出单个 action 的 15 Hz policy，未来 endpoint 尚不存在。以下三项
不能同时严格满足：

1. 零额外延迟；
2. 精确经过每个未知的 15 Hz endpoint；
3. endpoint 处速度和加速度连续。

因此新 profile 的平滑保证分为两档：

- **chunk / replay 模式：** 至少已有两个 future keyframe，可以提供严格的连续规划保证；
- **single-step 模式：** 只能选择一个 source-period 延迟，或使用有界预测并接受新目标
  到达后的误差。不得把后者描述成“精确零延迟插值”。

部署 policy 当前本来就产生 action chunk，replay 更拥有完整未来，因此主线应使用
future-aware 但仍满足 `available_timestamp` 的因果规划。single-step 只作为兼容回退。

## 4. 目标合同

### 4.1 统一时钟

```text
source target lattice       15 Hz（真实 timestamp，可有 jitter）
trajectory planning window  >= current + 2 future source keyframes
command evaluation          120 Hz
hardware feedback           保持现有后端频率与权威 owner
```

不再把 60 Hz 作为必须存在的第二套关节插值层。Pose path lift 若仍需 60 Hz 数值求解，
它是 planner 内部离散化，不是另一个会抢占 qcmd 的权威 reference owner。

### 4.2 15 Hz keyframe 语义

- keyframe 是带时间戳的 waypoint，不隐含 `qd=0`、`qdd=0`；
- 连续运动时 knot derivative 来自邻近 keyframes 和当前 admitted execution state；
- 明确停止只能由连续近零 source displacement、Hold/stop intent、计划覆盖耗尽或安全
  状态触发；
- 15 Hz canonical safety 只执行一次；高率轨迹不得重复改变 action 的语义；
- dataset action 仍是 source timestamp 上的 action，不因执行插值而改写 label。

### 4.3 执行连续性

每次 plan replacement 必须从最后一条正式 admitted command 状态开始：

```text
(q0, qd0, qdd0) = last admitted command state
```

不能从 raw model target、未实际执行的理论轨迹点或异步 hardware snapshot 猜测动态
anchor。新旧 plan 的切口至少保持 `q/qd/qdd` 连续；jerk 有硬界，并记录切口 jump。

## 5. 推荐实现

### 5.1 Source Pose → 多分支 qpos waypoints

部署 Pose 与 pose replay 共享同一 source-rate path-lift：

1. 每个 15 Hz Pose 使用 official all-IK 枚举合法候选；
2. 对当前 committed branch 和未来至少两个 keyframes 建立小型候选图；
3. 硬筛选 joint limit、FK residual、collision 和未经授权的大 branch jump；
4. 用 `q/qd/qdd/jerk`、branch hysteresis、奇异裕度选择窗口路径；
5. 只 commit 短 prefix，mutable suffix 可被下一 chunk 安全替换；
6. Pink 保留为全解不存在或需要小 Pose slack 时的显式 fallback，不再作为主要 branch
   owner。

关键变化是先得到一条多帧 joint path，再生成连续时间轨迹；不能逐帧 IK 后再试图用
滤波器掩盖错误 branch 或尖角。

### 5.2 连续 joint trajectory

建议实现共享的 `JointWaypointTrajectoryGovernor`（名称可在落地时按现有模块命名），
其输入为：

```text
current admitted (q, qd, qdd)
timestamped q waypoints
immutable/available horizon
joint q/v/a/jerk limits
collision validation callback
```

其输出为可事务提交的 120 Hz control cut 和完整 receipt。

轨迹生成分两种算法模式，但使用相同输入/输出合同：

#### Offline replay

- 对整段或较长滑窗解 constrained minimum-jerk QP；
- waypoint 位置在固定时间可行时作为等式；
- knot `qd/qdd` 是全局联合变量，不设为零；
- `q`、`qd`、`qdd` 至少 C2 连续，jerk 分段有界；
- 若固定时间下违反动态约束，按配置选择：
  1. `strict_timing`：拒绝并报告不可行；
  2. `safe_retime`：只延长违规区间并生成 time-warp receipt。

不得静默 clip qpos 或在输出后再裁速度，这会破坏末端路径和连续性。

#### Online deploy chunk

- 使用 receding-horizon minimum-jerk / tracking QP；
- current admitted `q/qd/qdd` 为硬初值；
- 已知 source waypoints 是 tracking 目标，默认不是每个 knot 的停车终态；
- 当前 chunk 后还有 future coverage 时，终端速度由末两/三段的受限切线估计；
- 只在 coverage 即将耗尽、没有新 chunk 时逐渐转入 braking tail；
- 新 chunk 到达时从当前 admitted state 重规划，不能重置速度或加速度；
- commit 1–2 个 120 Hz tick 或一个明确的短 control cut，而不是把整个 mutable
  horizon 提前宣告已执行。

### 5.3 Knot derivative

首版不使用普通 unconstrained cubic spline，避免关节限位附近 overshoot。推荐：

- 内部 waypoint：受限 centered difference / shape-preserving Hermite 初值；
- 窗口首端：使用 last admitted `qd/qdd`；
- 窗口末端：有 future 时使用受限 one-sided tangent；future 不足时使用可达 braking
  terminal state；
- 最终由 joint q/v/a/jerk constrained QP 投影，而不是逐关节事后裁剪。

对于连续匀速 source path，理想结果是所有内部 knot 具有近似相同非零速度，而不是
每 66.7 ms 回到零。

## 6. Deploy、qpos replay、pose replay 的具体语义

### 6.1 Deploy（推荐主线）

```text
policy action chunk
  -> canonical 15 Hz keyframe safety
  -> immutable future coverage / mutable suffix receipt
  -> source all-IK candidate graph
  -> continuous joint horizon governor
  -> 120 Hz qcmd
```

平滑 profile 必须要求至少 `current + 2 future` 已安全 staged。覆盖低于阈值时不继续
假装匀速，而是发出 `coverage_degrading` 并平滑制动。新 chunk 恢复后从当前动态状态
接续。

现有 `chunk_lookahead_enabled` 只能表达少量 successor；落地时应消费 typed
`TimedPolicyActionPlanSnapshot` / `ComponentActionTrajectory` 的连续 future，而不是继续
扩展多个 `lookahead_*` 参数。

### 6.2 QPOS replay

- 第一帧 qpos 仍用于归位；
- 余下已记录 qpos 按真实 source timestamp 作为 waypoints；
- 默认 `strict_timing`：只有满足 q/v/a/jerk 才执行原 15 Hz 时间；
- 不可行时禁止“线性插值后直接下发”，由用户选择拒绝或 `safe_retime`；
- qpos replay 的目标是复现 joint path，不经过 Pose→QPOS；
- 若要求逐样本严格命中，就必须接受 retime 或拒绝，不以停车换“命中”。

### 6.3 Pose replay

- 第一帧 qpos 仍用于归位；
- source Pose 进入与 deploy 完全相同的 official all-IK candidate graph、branch selector
  和 joint governor；
- 默认 `deploy_parity` 只暴露与线上相同的 bounded lookahead，作为策略部署复现检查；
- 另设 `offline_full_lookahead` 作为诊断上界，不得把其结果宣称为 runtime-causal
  deploy 能力；
- 不能先对 Cartesian Pose 做任意高阶 spline 再逐点 IK，因为中间 Pose 可能不可达或
  诱发不同 branch。首版先对 source Pose 做 all-IK path lift，再在 joint path 上连续化。

## 7. 停止、迟到与故障语义

### 正常停止

连续 source displacement 低于阈值并保持多个 keyframe 后，规划 bounded deceleration；
不能因单帧重复 action 立即把 terminal velocity 清零。

### action/chunk 迟到

- future coverage 高：照常执行 immutable prefix；
- coverage 进入 warning：减少 terminal tangent，开始可逆制动；
- coverage 耗尽：到达 safe hold，禁止外推；
- 新 chunk 到达：从当前 admitted `q/qd/qdd` 重规划。

### Hold / collision / force guard / execution fault

保持现有权威安全语义：取消旧 plan epoch、停止新 dispatch、执行安全 Hold。恢复后必须
从新测得/正式接受的动态 anchor 创建新 plan，不跨故障复用旧 derivative。

## 8. “平滑保证”的可验证定义

新 profile 只有同时满足以下条件，才可以称为解决 15 Hz 走停：

### 硬条件

- joint position / velocity / acceleration / jerk violation = 0；
- collision endpoint 与 segment certificate 通过；
- plan replacement 的 q/qd/qdd continuity violation = 0；
- 未授权 branch jump = 0；
- late/fallback/retime 都有显式 receipt，不静默改变时间或目标；
- 真机主环不得因轨迹求解超过 deadline。

### 走停指标

- 持续单向 source 区间中，内部 15 Hz knot 的速度谷值不得周期性接近零；
- `velocity_min_at_knot / local_velocity_median` 应高于设定阈值，首版建议离线门禁
  `>= 0.6`，再由真机 A/B 校准；
- 15 Hz 及其谐波的 qd/qdd/jerk 能量相对当前基线显著下降；
- knot 邻域 jerk p95 不高于非 knot 区域的固定倍数；
- 持续运动区的 stop-duty-cycle、速度反转和 catch-up 峰值不增加。

### 跟随指标

- source waypoint 的 joint/FK Pose error；
- qref→qcmd 延迟和 steady ramp lag；
- 停止后的 0.1/0.25/0.5 s 收敛；
- 不允许用永久 lag、幅值缩小或大量 retime 换取平滑。

## 9. 离线测试矩阵

### 合成轨迹

1. 单关节恒速斜坡：内部 knot 不得减速到零；
2. 多关节恒速和恒曲率轨迹；
3. 平滑换向；
4. 真实停止与单帧重复 target；
5. chunk 边界连续、chunk 迟到、coverage 耗尽后恢复；
6. joint limit、奇异、合法 branch change；
7. fixed timing 不可行，验证 strict reject 和 safe retime；
8. 非整数/抖动 source timestamp。

### 真实数据

- 0726、0727、0731、0802 及当前最新有效数据；
- 单臂/双臂、慢速/快速、持续匀速/换向/停止；
- 已知 branch、J3 boundary、奇异和高频微抖窗口；
- qpos replay 和 pose replay 使用同一 episode 做 paired A/B；
- pose replay 分别跑 `deploy_parity` 与 `offline_full_lookahead`，明确量化 future
  信息带来的差异。

对照至少包括：

```text
A. 当前 qpos replay 双线性链路
B. 当前 official all-IK causal governor
C. 新 continuous waypoint governor（bounded lookahead）
D. 新 continuous waypoint governor（offline full lookahead，仅上界）
```

## 10. 分阶段落地与回退

### Phase 0：冻结基线和只读 evaluator

- 固定当前 deploy/replay profile 名称和关键参数 SHA；
- 增加走停、knot derivative、15 Hz spectral energy、coverage 指标；
- 不改变任何 command。

### Phase 1：QPOS replay 离线连续轨迹

- 在 `robot_motion` 实现纯 joint waypoint governor；
- 接入 qpos replay dry-run / sim；
- 支持 `strict_timing` 和显式 `safe_retime`；
- 当前 replay linear path 保持可选、默认不变。

### Phase 2：Pose replay parity

- official all-IK 生成 source qpos candidate path；
- 接入相同 continuous governor；
- 加 `deploy_parity` 和 `offline_full_lookahead` 两种明确模式；
- MuJoCo 对代表性轨迹并排可视化。

### Phase 3：Deploy chunk shadow

- 从 typed timed plan 获取全部已 available future，不再依赖固定两个 lookahead 参数；
- 新 planner 只产生 shadow qcmd/receipt，与当前真机 command 对比；
- 验证 deadline、plan replacement、coverage braking 和安全证书。

### Phase 4：Default-off 真机 A/B

- 单臂低速空载开始，再双臂；
- 先持续直线/旋转，再换向、停止、chunk delay；
- 任一 safety/continuity/deadline gate 失败立即回退旧 profile；
- 用户确认后才讨论默认提升。

### 原子回退

- 不修改当前 default deploy backend；
- 新 profile 单独命名和注册；
- 旧 qpos replay 与 pose replay mode 保留；
- planner state、OSQP workspace 和 trace schema 不与旧 profile共享可变状态；
- profile 切换后必须 reset epoch 和动态 anchor。

## 11. 推荐决策

1. 不把当前 15 Hz teleop 的 sample-aligned “每 cut endpoint 严格对齐”直接复用于
   deploy/replay；该合同天然容易把每个 sample 变成停车点。
2. 优先从 qpos replay 落地全序列 constrained minimum-jerk，再接 pose replay；这两条
   都可完全离线验证。
3. deploy 只在 action chunk 提供足够 future coverage 时启用连续 profile；single-step
   policy 保留延迟插值/安全 governor 兼容模式。
4. deploy 与 pose replay 共享 all-IK candidate/path selector 和 joint governor；差别只在
   future 可见范围，避免“replay 很平滑但 deploy 无法复现”。
5. 15 Hz keyframe 是 waypoint，不是 zero-velocity endpoint；这是消除周期性走停的核心
   合同。

## 12. 2026-08-04 Phase 1 落地状态

已完成一个 default-off、仅用于离线 / MuJoCo QPOS replay 的安全切片：

- 新增 `offline_c2_quintic_waypoint_v1`：15 Hz qpos 作为带时间戳 waypoint，内部 knot
  使用 shape-preserving 非零速度，所有分段共享 knot acceleration，因此 `q/qd/qdd`
  连续；
- 对每个分段多项式解析求取 `q/qd/qdd/jerk` 的连续时间极值，不只检查 120 Hz 样点；
- `strict_fixed_timing` 在任一硬约束不满足时直接拒绝；
- `safe_retime_global` 保持全部 qpos waypoint 不变，只统一延长执行时间，并记录
  `time_scale`、source / execution duration 和完整证书；
- MuJoCo debug QPOS replay 通过 `--qpos-execution-mode continuous_quintic` 显式启用，
  原 `legacy_linear` 仍为默认与原子回退；
- 当前尚未接入真实 replay executor、Pose replay 或 deploy command owner，也未启动真机。

合成恒速轨迹验证结果：内部两个 15 Hz knot 的速度均为 `0.15 rad/s`，没有周期性归零；
越限轨迹被拒绝，安全 retime 后 waypoint 保持精确且连续时间证书通过。

对本地只读副本 `/tmp/remote_0802_latest_data/episode_000002.parquet` 的 1084 帧审计结果：

- source duration `72.9225 s`；
- 原时间戳下位置、速度、加速度可满足当前 RM65 合同；
- jerk 有 5 个关节不满足，最大约为合同的 `6.18x`，因此 strict 模式正确拒绝；
- `safe_retime_global` 在 `time_scale=1.83557` 时通过全部 q/v/a/jerk 连续时间证书，
  execution duration 为 `133.8542 s`。

这说明“原时间、逐点精确、动态硬约束、无走停”在该记录上不能同时满足。下一阶段不能
把 `1.83557x` 全局慢放直接用于 deploy；应实现局部 retime（qpos replay）和允许小 tracking
误差的 causal receding-horizon governor（Pose replay / deploy shadow）。

## 13. 2026-08-04 bounded-lookahead deploy 实验切片

已在现有 official source all-IK governor 内加入两级、可独立回退的能力：

- `--position-continuous-waypoint-shadow` 只计算并记录 C2 候选，绝不改变 qcmd；
- `--position-continuous-waypoint-experimental-admission` 允许候选在一个完整 15 Hz source
  cut 内原子接管，但仍为 default-off、debug-only；不加该参数即完整回到正式 governor；
- 候选必须有两个已 staged 的 causal successor，并通过解析连续时间 q/v/a/jerk 证书；
- 由于真实 position backend 只接收 q sample，另对整段 120 Hz 实际位置序列计算离散
  qd/qdd/jerk 证书，下一 cut 也使用这一离散 emitted state 作为 anchor；
- 接管前一次性检查完整 committed cut 的全部 8 个 endpoint 和每段 25/50/75% 内插碰撞；
  任一动态、coverage 或碰撞检查失败时整段使用正式 governor，不允许 cut 内逐 tick 切换；
- 碰撞合同仍是 sampled interval，不是 continuous swept-volume certificate；metadata 明确记录
  selection、fallback reason、collision/dynamic certificate 和 state semantics。

同时修复正式 governor 的一个独立数值合同问题：reused batched workspace 默认 tolerance
由 `1e-4` 收紧为与硬证书匹配的 `1e-6`；旧值仍可通过
`--position-all-ik-batched-solver-tolerance 1e-4` 显式回退。只有 batched candidate 已求出但
未通过硬证书时，才使用同限制的 non-batched certified fallback；QP failure/infeasible 等其他
错误继续 fail-closed。

### closed-loop 离线 A/B

评估使用固定 deploy source clock 15 Hz 和 command clock 120 Hz，不使用 collection wall
timestamp；每个新 cut 从上一条实际选中路径的 emitted q/qd/qdd 继续，而不是始终 teacher-force
回正式路径。

0727 的 115 条最长 continuity segment（115,912 个 command frame）：

- 实验路径接管 `96.56%` command frame，115/115 segment 至少接管一次；
- velocity p95 `23.82 -> 23.89 deg/s`、p99 `48.05 -> 48.19 deg/s`，轻微增加约 0.3%，max
  保持 `108.75 deg/s`；
- acceleration p95 `111.93 -> 99.16 deg/s^2`、p99 `339.65 -> 236.66 deg/s^2`；
- jerk p95 `17,759.90 -> 6,544.75 deg/s^3`、p99
  `47,899.47 -> 17,763.87 deg/s^3`；
- 默认 `1e-6` batched tolerance 下正式路径没有 certificate fallback；实验 closed-loop 中有
  3 个 source cut 使用 certified non-batched fallback（metadata 在各 8 个 tick 重复，因此统计
  为 24 tick）；
- 实验 source-cut 计算耗时 p50/p95/p99/max 为
  `5.92/6.82/8.34/41.30 ms`。p99 已贴近 120 Hz 单 tick budget，max 明显超限。

另外只读拉取远端
`DexTeleop_0803/datasets/2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage`
最后 5 条 episode（238--242）扩充验证：

- 无真实碰撞模型桩时，3,968 个 command frame 的接管率为 `97.58%`；acceleration p99
  `258.77 -> 166.40 deg/s^2`，jerk p99 `39,419.85 -> 11,941.11 deg/s^3`；
- 使用项目真实 collision model 后，episode 238/239/240/242 仍接管 `97.45%`，acceleration
  p99 `247.37 -> 152.12 deg/s^2`，jerk p99 `34,970.06 -> 10,877.76 deg/s^3`；
- episode 241 在 source selection 前因记录的右 J6 为 `2.0943951607 rad`，比绝对上限
  `2.0943951024 rad` 高 `5.83e-8 rad` 而 fail-closed；这是数据 float32 序列化残差，不是
  连续候选或碰撞回退，暂未在本切片中放宽；
- 启用真实碰撞检查后的 source-cut 计算耗时 p50/p95/p99/max 为
  `9.22/10.15/11.52/20.35 ms`，持续超过 120 Hz 同步 hot-path budget。

### 当前权限结论

运动质量在 0727 全量和新增 0804 样本上整体正向，但真实碰撞预检的同步耗时尚不满足
120 Hz deadline。因此：

- 当前正式 governor（含 `1e-6` 数值修复）继续是 deploy 默认；
- continuous waypoint shadow 可用于 debug 采集；
- experimental admission 保持 default-off，不作为真机推荐配置；
- 下一步应把 complete-cut 构建与碰撞预检移到有 epoch/anchor 校验的 15 Hz bounded worker，
  120 Hz owner 只消费已认证 plan；planner 迟到或 anchor 变化时继续正式 governor，不能阻塞
  command hot path。

## 14. 2026-08-05 bounded next-cut precompute

在不改变任何默认配置、且保留 Phase 13 同步实现作为原子回退的前提下，增加了第三个独立
实验开关：

```text
--position-continuous-waypoint-async-precompute
```

该开关必须与 `--debug` 和
`--position-continuous-waypoint-experimental-admission` 同时使用。实现合同为：

- 只有一个 daemon worker、一个 pending slot 和最多两个已完成结果；新请求只覆盖尚未开始的
  speculative 请求，不建立无界队列；
- 当前 cut 决定正式 / C2 整段执行模式后，立即从该整段末行推导下一 cut 的精确
  `q/qd/qdd` anchor；当下一次 15 Hz staging 补齐两个 causal successor 后才提交预计算；
- request key 同时覆盖 reset epoch、source interval、source timestamp、全部 waypoint、anchor
  `q/qd/qdd`、terminal tangent 和 120 Hz sample timestamps；120 Hz owner 只消费完全相同的 key；
- 首 cut、worker 迟到、pending、reset、source replacement 或 emitted anchor 不一致时不等待，
  整个 cut 使用正式 governor；禁止逐 tick 混切；
- C2 构建、离散 qd/qdd/jerk 证书及完整 cut 的 8 endpoint + 24 interior collision precheck
  在独立、配置相同的 proposal collision planner 上完成；该 planner 不拥有最终安全权限；
- canonical collision planner 仍在每一个实际 120 Hz tick 对本 tick endpoint 和三个 interior
  sample 重新验证。proposal 与 canonical 不一致时 fail-closed/hold，不在 cut 中途切回 formal；
- executor close 会显式停止该 bounded worker；reset 增加 epoch 并丢弃 pending/completed 结果。

### 0804 真实 collision model 离线结果

对远端只读副本 episode 238/239/240/242，以真实 15/120 Hz 间隔运行；episode 241 仍因既有
右 J6 float32 上限残差在 source selection 前 fail-closed，未混入统计。共 2,824 command frame：

- C2 接管率 `97.17%`，四条轨迹全部至少接管一次；
- acceleration p95 `94.59 -> 71.44 deg/s^2`，p99
  `247.37 -> 152.12 deg/s^2`；
- jerk p95 `15,643.76 -> 4,606.16 deg/s^3`，p99
  `34,970.06 -> 10,877.76 deg/s^3`；
- velocity p99 `35.08 -> 34.82 deg/s`，max 保持 `62.14 deg/s`；
- async candidate source-cut owner 耗时 p50/p95/p99/max 为
  `2.78/3.82/5.41/15.32 ms`。相较同步真实碰撞版本的
  `9.22/10.15/11.52/20.35 ms`，p99 已低于 8.33 ms budget；最大的两个离群点
  `15.32/14.32 ms` 都位于各 episode 的首 cut，非首 cut 最大为 `5.54 ms`。这里还专门把 worker
  提交移到当前 tick 的 canonical collision/FK 之后，避免 proposal worker 与当前权威 hot path
  同时争用 CPU / Pinocchio cache。

### 当前权限结论

- 默认 deploy 仍为正式 official all-IK governor；
- Phase 13 同步 admission 开关保持原样，可单独回退和对照；
- Phase 14 async admission 仍为 default-off、debug-only，不直接提升为真机推荐；
- 在真机权限提升前还需扩大真实 executor deadline soak，并验证 submission replacement、
  clear/reset 和 Ctrl+C close trace；首 cut 固定使用 formal，当前离线首 cut 初始化超时不会选择尚未
  存在的 C2 candidate，但仍需在真实 scheduler trace 中确认不会传播到后续 tick；
- 所有这些后续验证不得削弱 canonical per-tick collision validation。

## 15. 2026-08-05 real-executor no-hardware deadline soak

为避免 Phase 14 只验证 planner 函数、没有覆盖真实 executor 调度与生命周期，新增长时间探针：

```text
todos/probes/deploy_continuous_waypoint_deadline_soak.py
```

探针运行真实 `PositionDeployExecutor`、official all-IK governor、canonical collision planner、
proposal collision planner 和 bounded async worker；唯一替换的边界是机械臂 dispatch/hold，使用
内存 callback 返回 accepted receipt，不打开网络、SDK 或设备。输入用记录的 accepted `q_ref`
经共享 RM65 FK 生成可达 Pose，随后仍由真实 deploy Pose→QPOS 链路重新求解。

### soak 暴露并修复的两个集成合同

1. async worker 为下一 cut 构造两步 source horizon 时，需要当前 source 之外三个同 chunk
   successor。原 deploy typed path 只传递到第二 successor，因此直接 planner evaluator 能运行，
   真实 executor 却永远无法提交 next-cut request。现只在 async 实验开关启用时透传第三个
   speculative same-chunk lookahead；它不推进 source cursor，也不获得执行权限。
2. 相邻提交会以不同的浮点结合顺序重复 stage 同一个未来 source timestamp，差值约
   `4.66e-10 s`。`stage_source_pose` 既把该值判定为等价，又覆盖原 bytes，导致 exact async
   admission hash 每 cut 漂移。现对现有 `1e-9 s / 1e-8 pose` 等价合同作真正幂等处理：等价
   restage 保留首次接受的 bytes；超出容差的真实 replacement 仍执行原有 invalidation。

### 真实 executor 基线结果

初版使用 episode 238--242、与真机命令相同的 `taskset -c 0-7`，运行 10 秒：

- 1,196 次 mock arm dispatch，1,169 次选择完整 C2 cut，接管率 `97.74%`；
- 149/149 async request 完成，exact-anchor mismatch 为 0；
- 稳态（每次 reset 排除前 16 command frame）planner p50/p95/p99/max 为
  `0.637/2.236/2.547/3.079 ms`；
- 稳态 deadline lateness p50/p95/p99/max 为
  `2.058/4.975/6.229/7.001 ms`，1,148 个稳态 tick 中没有一次超过 `8.333 ms`；
- 无 executor fault、无 dispatch failure、无 queue full，close 后 reference worker 已停止；
- 全程仍有 3 个启动阶段 missed ticks 和一次首 cut planner budget exceed，因此该结果明确拆分为
  `deadline_clean=false`、`steady_state_deadline_clean=true`，不能隐去冷启动成本。

继续定位确认 reset 已在 reference clock 启动前完成，首 tick 峰值实际来自一次性为当前及初始
successor 枚举/碰撞筛选全部 IK 分支，以及 batched QP workspace 的首次 solve。现增加
`prepare_staged_source_horizon()`：在 first/reset submission 已有完整 same-chunk lookahead 后、
reference clock 启动前，用完全相同的 authoritative selector 选择 initial source qpos，并构建、缓存
首个正式 horizon。它不下发命令、不 commit source、不推进 emitted state；真实 future replacement
仍会使缓存失效。定向测试确认预热与未预热的首条 formal qcmd 在 `1e-9 rad` 内一致。

修复后再次运行相同 10 秒 / 3 次 reset：

- 1,196 次 dispatch，`missed_reference_ticks=0`；
- 全量（不再排除 warmup）deadline lateness p50/p95/p99/max 为
  `2.089/4.841/6.058/7.040 ms`；
- 全量 planner p50/p95/p99/max 为 `0.646/2.617/3.023/3.899 ms`；
- `time_budget_exceeded_count=0`，`deadline_clean=true`，C2 接管率 `97.83%`；
- 无 executor fault、dispatch failure、queue full、anchor mismatch 或 close 后线程残留。

因此无硬件 Python executor 中的冷启动 miss 已消除；真实硬件边界仍需按本节末尾权限限制验证。

当时默认 profile 落地后的最后一次同规格复验仍为 1,194 次 dispatch、C2 接管率 `97.82%`，planner
p99/max 为 `3.190/3.919 ms`，Pose 相对 formal q-reference 的平移 p99/max 为
`0.343/0.506 mm`、旋转 p99/max 为 `0.067/0.159 deg`；无 fault、dispatch failure、queue full、
anchor mismatch 或线程残留。当前并行开发负载下出现 2 个 lateness `8.333 ms` 以上的 tick，最大
`8.686 ms`，而同期 planner 最大仍仅 `3.919 ms`。因此该次尾部归类为主机调度证据，不归因于 C2
planner；`taskset` affinity 不能替代后续遥操/模型/真机进程的 CPU 隔离或实时调度验收。

未固定 CPU affinity 的 20 秒扩展运行覆盖 4 条轨迹、2,366 次 dispatch：C2 接管率
`98.27%`，planner 稳态 p99/max 为 `2.625/3.762 ms`；但 OS 调度尾部造成 2,302 个稳态 tick
中 8 个 lateness 超过周期。说明探针能实际测出 scheduler/affinity 差异，不能只用 planner
函数耗时替代 executor deadline。

### replacement / reset / late-worker / dispatch-spike 注入

8 秒组合故障注入包含：每 4 个 request 注入 25 ms proposal delay、每 9 个 source 替换一次
speculative successor、每 31 个 source clear/reset、每 100 次 dispatch 注入 12 ms delay。结果：

- 916 次 dispatch，674 次 C2、242 次完整 formal fallback；
- 13 次 replacement、3 次主动 clear/reset 均未触发 executor fault、dispatch failure 或
  queue full；worker 和 reference thread 均在 close 后退出；
- C2 迟到或 anchor 改变时只增加 `precompute_not_ready` / formal fallback，没有等待 worker，
  也没有 cut 内部分切换；
- 注入本身按预期造成 deadline miss，故该运行的 deadline verdict 不应通过，但 lifecycle /
  fallback verdict 通过。

### 可重复命令

普通无硬件 soak：

```bash
taskset -c 0-7 conda run -n dexteleop310 python \
  todos/probes/deploy_continuous_waypoint_deadline_soak.py \
  --glob '/tmp/dexteleop_0804_waypoint_ab/episode_000*.parquet' \
  --repeat 100 --duration-sec 600 \
  --output-json /tmp/deploy_deadline_soak_10min.json
```

组合故障注入：

```bash
taskset -c 0-7 conda run -n dexteleop310 python \
  todos/probes/deploy_continuous_waypoint_deadline_soak.py \
  --glob '/tmp/dexteleop_0804_waypoint_ab/episode_000*.parquet' \
  --repeat 100 --duration-sec 600 \
  --precompute-spike-every 4 --precompute-spike-ms 25 \
  --replace-every-source-frames 9 --clear-every-source-frames 31 \
  --dispatch-spike-every 100 --dispatch-spike-ms 12 \
  --output-json /tmp/deploy_deadline_soak_faults_10min.json
```

### 权限边界

该探针已经能够模拟真实 Python executor 的 15/120 Hz wall-clock scheduling、planner contention、
bounded worker、source replacement、reset、close 和 fallback；它不能模拟 RM SDK/CAN 延迟、真机
进程优先级、USB/相机/模型 worker 竞争、内核 IRQ 或执行器反馈。经上述动态、Pose、deadline、
replacement 和 lifecycle 门禁曾支持将
`--position-continuous-waypoint-profile async_c2` 提升为默认。但该权限一直未覆盖
RM SDK/CAN、整机调度、相机/模型竞争和实际反馈。

2026-08-08 首轮完整真实部署 trace 否定了这个默认权限：C2 worker 单次预计算约
`10.96 ms`，与权威 60 Hz reference worker 竞争后，C2 tick 端到端达到
`33.02/34.10 ms`，最终触发 `skipped=2` 的 deadline fail-closed。一次 miss 还会使后续
candidate anchor 失效，造成预计算占用算力却无法接管。因此当前默认已恢复为
`--position-continuous-waypoint-profile formal_third_order`；`async_c2` 保留为原子可回退的
隔离研究 profile，只有在 CPU 隔离与真实 executor deadline soak 重新通过后才能重新
评估真机权限。历史 shadow、同步 admission 和 async-precompute 布尔参数仍为
`--debug` 研究接口。
