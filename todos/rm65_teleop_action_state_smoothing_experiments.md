# RM65 双臂遥操 Action–State 平滑实验计划

## 目标与边界

目标是在保持双臂真机安全合同和可原子回退的前提下，找到同时满足以下条件的
遥操执行时间律：

1. 120 Hz `qcmd` 平稳，无异常关节跳变、15 Hz 周期性速度脉动或高频反向；
2. 位置、速度、加速度、jerk、关节限位和碰撞门禁不弱化；
3. 人手持续运动时跟随性良好，停止后能收敛，不产生长期静差；
4. 数据集 15 Hz 主 action 尽量接近 `state(t+1)`；
5. 双臂分别通过，不能以一侧静止掩盖另一侧回退；
6. 所有研究实现默认关闭，当前 V1/V2 代码路径和真机命令保持可回退。

本轮先使用 actuator trace 和既有数据集做离线 replay，不启动设备、不执行运动。
只有通过离线门禁的候选才允许注册为新的 default-off profile。

## 已确认的 V2 失败机制

最新 `20260802_233918` V2 trace 表明，主要问题不是 VIVE、奇异 IK、SDK 拒绝或
120 Hz 调度，而是每个 15 Hz source cut 被当作一个终端停车点：

- 15 Hz qref 关节向量步长中位数约 `2.58°`，p95 约 `4.32°`；
- planner 在前 8 个 120 Hz tick 插值到当前 target，之后把 horizon reference
  保持为常数；
- 左臂速度在单个 8-tick bucket 内呈明显的加速—减速包络；
- 重规划边界 jerk p95 明显高于 horizon 内部；
- 多个关节频繁达到 98% 以上 jerk 硬上限。

V2 的 full target 与 0.70 source fast path 缓解了延迟，同时放大了上述 stop-go
时间律缺陷。现有 `jerk_delta_regularization` 只约束单个 horizon 内的相邻 jerk，
不约束上一已执行 jerk 与新 horizon 首 jerk 的连续性。

## 统一输入与对照集

### Actuator trace

优先使用以下真实输入：

- `20260802_233918`：最新 V2，左臂主动运动、右臂基本静止；
- `20260802_233741`：第二条 V2；
- `20260802_230006`：同日晚间 V1；
- `20260802_151813`：较长 V2-responsive 双臂运行；
- `20260802_142716`：标准 V1；
- `20260802_142432`：标准 V2-responsive。

trace 中的 60 Hz `ik_shadow.admitted_q_rad` 作为统一的上游 IK/qref 输入；这样首先
隔离并比较 qref→qcmd 时间律，不把不同 IK 分支混入结论。后续再用
`target_world_pose`/VIVE 输入重跑不同 IK rate。

### 历史数据集

对 actuator trace 筛出的少量 Pareto 候选，再扩展到已有 0726、0727、0731、0802 数据集。
数据集中的 accepted qref 与实际 state/qcmd 用于验证：

- action 与 next state 的关节/FK Pose 差；
- 静止、缓慢、快速、反向、单臂和双臂区间；
- 奇异/关节限位/branch guard 区域；
- pause、hold、切臂和连续性 segment 边界。

## 候选路线

### A. 15 Hz action-authoritative 时间律

这是满足严格 `action(t)≈state(t+1)` 最自然的主线。

1. `15_hold`：复现当前 flat-tail reference，作为离线基线；
2. `15_velocity_tail`：用过去两个或三个 15 Hz qref 因果估计目标速度，前 8 tick
   仍追踪当前 action endpoint，后半 horizon 按受限速度延拓，不再假设每个 action
   都要求停车；
3. `15_hermite`：使用当前 q/qd 与因果目标速度构造 C1/C2 参考；
4. source velocity EMA 扫描，并限制为逐关节 production 速度的一定比例；
5. jerk ramp 扫描 `0.05/0.075/0.10/0.15 s`；
6. 新 horizon 首 jerk 相对上一条实际 jerk 的软约束与可选 snap 硬限制；
7. 停止检测和速度衰减，避免旧速度外推造成 overshoot 或长期静差。

该路线的正式 action 始终是下一 15 Hz 边界实际会执行的 admitted endpoint，中间
60 Hz IK 帧只能作为安全/观测输入，不得抢占当前 control cut。

### B. 30 Hz IK/target 辅助

比较两种不同合同，不混为一谈：

1. `30_preemptive`：每 33.3 ms 重规划，直接测量其平滑收益以及对已经发布的
   15 Hz action forecast 的破坏；若 action→next-state 回退则不能用于正式采集；
2. `30_delayed_velocity_observer`：30 Hz 中间帧只更新下一 bucket 的因果速度估计，
   不抢占当前 15 Hz control cut。它会引入有限信息延迟，但保留 action 权威性。

只有第二种天然符合现有数据合同。第一种保留为“平滑上界/语义反例”实验。

### C. 60 Hz 重规划与 checkpoint action

1. 复现当前 60→120 Hz 链路；
2. 对 60 Hz qref 使用时间戳感知滤波、速度/加速度前馈和 jerk-limited OTG；
3. 每个 15 Hz checkpoint 预测下一 sample 的 qcmd，并计算后续三次 60 Hz 抢占后
   的 forecast error；
4. 若必须通过延迟一帧、time dilation 或 action 重定义才能达到一致性，必须明确
   标记，不能声称与现有 15 Hz action 合同相同。

60 Hz 路线可能有最佳即时跟手性，但只有在 checkpoint forecast 仍稳定时才有资格
进入正式数采候选。

### D. 独立的上游抖动抑制

在下游时间律收敛后再单独验证：

- VIVE SE(3) timestamp-aware One-Euro / critically damped filter；
- translation 与 rotation 分轴阈值和速度相关截止频率；
- IK 仍以前一正式 admitted qref 为连续性 anchor；
- 只抑制亚毫米/亚度静止噪声，不削弱快速离开奇异或关节限位方向。

上游滤波与下游 governor 必须分别做 A/B，避免用双重平滑掩盖根因。

## 评价指标

每条 trace、每侧分别报告，不只输出全局 worst：

### 安全硬门禁

- joint position、velocity、acceleration、jerk violation 必须为 0；
- qcmd 单步异常跳变必须为 0；
- 后续生产候选必须通过 endpoint 与 segment collision 门禁；
- 非有限值、QP failure、deadline/fallback 必须为 0 或显式 fail-closed。

### 平滑性

- qcmd step、velocity、acceleration、jerk 的 p50/p95/p99/max；
- jerk 达到 90%/98% 硬上限的帧数和持续时间；
- 15 Hz cut 附近与 horizon 内部 jerk 的比值；
- 每关节有意义的方向反转次数；
- 每个 8-tick bucket 的归一化速度包络，检测 stop-go；
- 静止窗口 RMS/峰峰值和频谱中的 15/30/60 Hz 成分。

### 跟随性

- requested qref→admitted action 的关节/FK Pose 误差；
- qref→qcmd 的时间延迟与动态误差；
- 停止后的 0.1/0.25/0.5 s 收敛误差；
- 持续斜坡、变向和快速动作下的峰值落后；
- 不允许以永久静差换取平滑。

### 训练语义

- `action(t) - state(t+1)` 的关节和 FK Pose p50/p95/max；
- action forecast 被中间 30/60 Hz 重规划改写的次数和幅度；
- action lineage 必须对应实际执行 control cut，不能使用未来输入；
- hold/pause/切臂后重新建立 epoch，不跨 segment 差分。

## 选择与上线门禁

候选按字典序筛选：

1. 先通过所有安全硬门禁；
2. 排除异常跳变、15 Hz 明显 stop-go 和高 jerk 饱和候选；
3. 排除 action→next-state 明显回退的 30/60 Hz 抢占候选；
4. 在剩余候选中比较跟随延迟、终态误差和双臂 worst；
5. 只有跨 trace、跨数据集大部分为正向且无新 corner-case 的候选，才注册成
   `default-off` 新 profile；
6. V1/V2 保持不变，profile 选择即为原子回退；默认真机命令不变，直到用户明确
   授权提升权限。

## 当前执行顺序

- [x] 建立 actuator-trace 统一 replay/evaluator；
- [x] 复现实际 V1/V2 qcmd 指标与最新 V2 stop-go；
- [x] 扫描 15 Hz flat/velocity/Hermite 与 jerk 参数；
- [x] 扫描 30/60 Hz preemptive，量化 action forecast 回退；
- [x] 实现并否决会污染 V2 fallback 的跨 horizon 首 jerk 软约束；
- [x] 对 Pareto 候选扩展到 0726/0727/0731/0802 与 corner cases；
- [x] 将通过当前离线门禁的零空间方案注册为 default-off V3 并补测试；
- [ ] 真机验证必须由用户另行明确授权。

## 2026-08-03 离线实验结果

### 基线合同修正

早期探针有两项偏乐观的合同偏差，相关旧数字不再作为上线依据：

1. 最初使用 RM65 名义关节速度上限，未同时施加 `0.675° × 120 Hz = 81°/s`
   的实际单步上限；现已统一使用逐关节两者较小值。
2. 手写 `15_hold` 参考把 target 预先裁进 `1e-4 rad`，而生产 V2 是保留原 target、
   再由 QP 位置约束投影。靠近 J3/J6 边界时二者会分叉。

最终比较全部直接实例化生产
`SampleAlignedJerkLimitedJointTrajectoryPlanner`：

- `15_runtime_v2`：生产 `linear_hold`；
- `15_runtime_v2_nullspace`：生产 `linear_hold_nullspace_smooth`。

探针和注册 profile 不再维护两套候选算法。

### 已否决路线

- **30/60 Hz 抢占式重规划**：执行更及时，但会在同一个 15 Hz action 已发布后改写
  其未来，代表轨迹出现超过 `1°` 的 action forecast 偏差；除非重新定义数据合同，
  不能用于正式数采。
- **无条件 causal Hermite**：requested→action 误差略降，但生产本体跨日期高运动窗口
  的 98% jerk 饱和从 V2 对照的约 `1799` 次升至 `2604` 次；降低 EMA/cap 后仍升至
  `3070–3189` 次，属于负优化。
- **同 workspace 的 Hold/Hermite 候选竞争**：即使回退 Hold，候选求解也会污染
  OSQP warm start；在关节边界附近产生后续数值分叉。
- **独立 workspace + 单 cut 选择**：修复 warm-start 污染后，仍需同时保持终端
  `q/qd/qdd`，否则候选会改变下一 cut 的动态 anchor；把 anchor 收紧到数值等价后，
  Hermite 基本退化为 V2，收益不足。
- **额外 `initial_jerk_delta_regularization=1e-5`**：它改变 V2 权威 Hold 目标函数，
  部分日期 jerk p95 回退；V3 已恢复为 `0`。

### 通过当前门禁的方案

最终 V3 不再预测未来 qref，而是在每个 V2 control cut 内处理已经 admission 的
8 个 120 Hz jerk 样本：

1. 先用生产 V2 QP 得到权威 plan；
2. 在保持 cut 末端 `q/qd/qdd` 完全相同的三维线性约束零空间内，最小化归一化 jerk
   能量及相邻 jerk 变化；
3. 从原 plan 到零空间最优解做 `1.0/0.75/0.5/0.25/0.125` 回溯；
4. 重新检查位置、速度、加速度、jerk 硬界；
5. 候选已执行 8 tick 的平滑分数至少改善 2%；
6. `0.002°` 以上有效步进的激活集合与方向符号必须和 V2 完全相同；
7. 将终端 `q/qd/qdd` 按位恢复为 V2 anchor，避免浮点残差改变下一次 OSQP active set；
8. 任一条件不满足就逐关节保留 V2，不影响其他关节。

因此它不会用“跟踪变慢”换平滑，也不依赖未来 VIVE 帧；15 Hz action 和
`state(t+1)` 的定义保持不变。

### 跨日期结果

验证集包含 11 条真实 actuator trace，按日期分层：

- 0726：2 条；
- 0727：2 条；
- 0731：2 条；
- 0802：5 条（含最新 `20260802_233918`）。

每条取 240 个 60 Hz 最高运动窗口，左右臂独立评价；合计 22 个 side-window。

| 指标 | 生产 V2 | 零空间 V3 | 结果 |
|---|---:|---:|---:|
| action→next-state p95 | `0°` | `0°` | 完全保持 |
| requested→action worst p95 | `2.5958°` | `2.5958°` | 完全保持 |
| jerk p95 worst | `177553.8°/s³` | `153114.3°/s³` | 约 `-13.8%` |
| cut 邻域 jerk p95 worst | `177621.6°/s³` | `153129.3°/s³` | 约 `-13.8%` |
| 98% jerk 饱和总数 | `1830` | `1582` | `-248`，约 `-13.6%` |
| `>0.05°` 相邻反向 | `22` | `20` | 不增加 |
| 有效方向反转总数 | `761` | `761` | 完全保持 |
| q/v/a/jerk 硬违规 | `0` | `0` | 完全保持 |

最新 `20260802_233918` 主动左臂单独比较：

- requested→action p95：两者均约 `0.1019°`；
- jerk p95：`137372.4→136781.4°/s³`；
- 98% jerk 饱和：`216→194`；
- 有效反转：`39→39`；
- V3 与 V2 的 120 Hz qcmd 差异 p95 约 `0.0515°`，最大约 `0.0841°`；
  右侧静止臂最大差异约 `0.00071°`。

0726 两条全长轨迹也完成验证：action/state 和有效反转完全相同，98% jerk 饱和
`6310→5460`（约 `-13.5%`）。0727 单条全长计算超过当前离线命令时间预算，未把
进程中止当作通过；该日期结论来自多条最高运动窗口。

### 碰撞与耗时

生产 V3 对上述 11 个双臂窗口做 endpoint 及每段 `0.25/0.5/0.75` 插值碰撞抽样：

- 检查 5269 个双臂 endpoint；
- endpoint collision：0；
- sampled segment collision interval：0；
- validation error：0；
- 最小抽样碰撞距离约 `0.06499 m`。

在最新 trace 左臂 434 个 60 Hz 输入、约 109 个 15 Hz cut 上，三次离线整段耗时均值：

- V2：约 `0.255 s`；
- V3：约 `0.347 s`；
- 新增成本约 `0.85 ms / 15 Hz cut`，远低于 `66.7 ms` cut 预算。

### 当前权限

- 默认 profile 未改变；
- 新 profile：`rm65_bimanual_position_sample_aligned_jerk_v3`；
- shell 参数：`POSITION_FORMAL_SAMPLE_ALIGNED_JERK_V3_ARGS`；
- 原子回退：`rm65_bimanual_position_sample_aligned_jerk_v2` /
  `POSITION_FORMAL_SAMPLE_ALIGNED_JERK_V2_ARGS`；
- 尚未启动真机、相机或任何设备，也未执行运动。

## V3 与历史 60 Hz 位控 V2 的离线 A/B

对照不是探针中的近似 `60_hold`，而是直接实例化历史生产
`rm65_bimanual_position_v2_responsive` 的完整 qref→qcmd 合同：60 Hz target
抢占、`AccelerationLimitedJointTrajectoryPlanner`、0.03 s target-velocity filter、
0.10 s velocity ramp、0.675° command step 和最终 `follow_alpha=0.9`。V3 使用当前
生产 `linear_hold_nullspace_smooth`。两者读取完全相同的 60 Hz
`ik_shadow.admitted_q_rad`。

跨 0726/0727/0731/0802 的 11 条全长 trace、22 个 side 的结果：

| 指标 | 历史 60 Hz V2 | 当前 15 Hz V3 | 解释 |
|---|---:|---:|---|
| 瞬时 qref tracking p95 worst | `1.9345°` | `2.9386°` | V2 更跟手 |
| 最佳跟随滞后均值 | `22.7 ms` | `62.5 ms` | V3 约多 40 ms |
| 15 Hz action→next-state p95 worst | `1.9393°` | `0°` | V3 保持训练合同 |
| jerk p95 worst | `722457.7°/s³` | `137372.4°/s³` | V3 约低 81% |
| jerk max worst | `1501710.3°/s³` | `183163.2°/s³` | V3 有硬 jerk 上界 |
| 有效方向反转总数 | `37146` | `23553` | V3 约低 36.6% |
| 相邻反向幅度 `>0.05°` | `1324` | `32` | V3 显著抑制大反向 |
| 相邻反向幅度 `>0.10°` | `480` | `0` | V3 未出现 |

历史 V2 没有 jerk 硬约束，因此其 `jerk saturation` 不能与 V3 的饱和次数直接比较。
离线固定 120 Hz 调度下，V2 最终 qcmd 数值二阶差分还出现 229 帧超过现行逐关节
qdd 门禁；这主要来自可抢占 acceleration planner 后的最终 0.9 follow adjustment，
应视为旧链路的时间律风险，而不是硬件测量。

高运动 240-row 窗口给出相同方向：V2/V3 的瞬时 qref tracking p95 worst 分别为
`2.2528°/5.3992°`，平均最佳滞后为 `26.9/59.8 ms`；jerk p95 worst 分别为
`794918.5/153114.3°/s³`，`>0.05°` 相邻反向为 `278/20`。

因此两者不是单调支配关系：历史 60 Hz V2 明显更及时，但更容易把 60 Hz qref
微反向和抢占边界直接传给 120 Hz qcmd，并且无法保证已发布 15 Hz action 的
next-state；V3 明显更平滑且 action 对齐严格，但相对最新 qref 多约 40 ms 跟随滞后。
本 A/B 只验证指令轨迹，不模拟机械臂结构柔性、负载、SDK/网络 jitter 和真实 state
闭环，真机主观抖动仍需受控 A/B 才能最终确认。

探针的 60 Hz V2 数值也得到历史真机 qcmd 侧证：实际
`20260802_142432` V2-responsive 左右臂 jerk p95 分别约
`707388/709723°/s³`，与固定时钟离线 V2 的跨日期 worst `722458°/s³`
同量级；该实际 trace 的 jerk max 达约 `1.27e6/1.13e6°/s³`。因此 V2/V3
平滑性差异并非只由离线仿真器人为制造。

## 60 Hz VIVE 速度辅助的 15/30 Hz 研究

### 数据与因果合同

新增只读离线探针
`todos/probes/rm65_vive_velocity_execution_probe.py`。它不初始化设备、不下发命令，
使用 actuator trace 中的 60 Hz `ik_shadow.target_world_pose`；新 trace 同时使用
`vive_source.pose_sample_monotonic_time_ns` 和 validity，旧 trace 使用同频
`ik_shadow.monotonic_time_ns`。每一行速度只使用当前及历史行：

1. 对相邻 SE(3) 位姿计算世界系平移速度和 rotation-vector 角速度；
2. 在最近 5 个 60 Hz 样本内取逐轴中位数；
3. 用净位移/路径长度抑制往返抖动；
4. 通过当前 RM65 Jacobian 和阻尼最小二乘映射为关节速度；
5. 估计不足、VIVE 无效、时钟停顿或映射门禁失败时显式标记 invalid。

invalid 不是零速度命令。生产 planner 的研究接口新增显式 validity：无效时严格回退
原 `linear_hold`/V3 路径，并清空速度滤波状态。早期实验把 invalid 写成零速度后仍然
生成 Hermite 曲线，导致有效/无效切换产生额外 jerk；该合同已由单测覆盖。

### 15 Hz 候选结果

跨 0727、0731、0802 共 6 条 trace 的最高运动窗口测试。原 V3 与较保守的
VIVE-DLS 候选都保持：

- `action→state(t+1) p95 = 0°`；
- 位置、速度、加速度、jerk 硬违规均为 0；
- 15 Hz action 仍由 accepted endpoint 定义。

直接采用速度的候选呈混合结果。以 `a25/c20` 为例，在不过滤 DLS frame 时：

- aggregate 最佳滞后约 `75.7→73.6 ms`；
- 98% jerk 饱和约 `1522→1316`；
- `>0.05°` 相邻反向约 `27→25`；
- 但 jerk p95 worst 约 `137372→148641°/s³`；
- 有效方向反转总数没有改善，部分单轨迹增加；
- 历史 0727 的约 63° IK branch discontinuity 仍主导最坏 tracking，速度时间律不能
  修复上游 qref 分支跳变。

增加跨 cut 初始 jerk regularization 可以进一步降低饱和，例如 `1e-4` 时约
`1522→892`，但有效小反转增至约 `879→924`，因此不是整体正优化。对 DLS residual
做逐帧硬拒绝也不是正确答案：虽然会安全回退，但高运动区可能频繁切换 baseline/
velocity mode。下一版应使用带滞回的连续置信度或多 cut 锁存，而不是逐帧开关。

严格 selective 候选只在局部 jerk、反转和预测误差同时不回退时采用速度。它与 V3
几乎完全一致，仅少约 2 次小反转，说明门禁安全但收益不足，暂不值得真机 profile。

### 30 Hz 候选结果

30 Hz midpoint refresh 能降低部分瞬时 qref 误差，但当前普通 QP 只把原 15 Hz
action endpoint 当 reference，没有把它作为硬等式。最新活动窗口中：

- `action→state(t+1) p95 ≈ 0.0916°`，不再是 0；
- qref tracking p95 `2.2657→2.0974°`；
- 98% jerk 饱和 `194→497`，反而明显增加。

因此该 30 Hz 实现被拒绝。后续若继续 30 Hz，必须采用“同一 15 Hz cut 内的
remaining-horizon 重规划 + 末端 q 硬等式”，不能只增加 midpoint reference。

### 当前结论与权限

- 60 Hz VIVE 速度估计和 validity/fallback 合同已实现并通过离线测试；
- 15 Hz 直接速度辅助证明了可改善滞后和部分大反向，但尚未整体支配 V3；
- 30 Hz 当前实现违反 action/state 硬合同；
- 不注册真机 profile，不修改默认参数；
- 下一研究点是连续置信度/模式锁存，以及 30 Hz endpoint equality；在这些问题解决并
  通过跨日期门禁前，真机继续使用现有 V3 命令。

### 后续结构候选与最终离线结论

继续测试没有停留在 Hermite 增益扫描，而是分别验证了以下结构：

1. **直接 15 Hz VIVE-DLS terminal velocity**：最佳保守参数将 aggregate 跟随滞后
   `75.7→73.6 ms`，98% jerk 饱和 `1522→1165`，`>0.05°` 反向
   `27→24`；但 jerk p95 worst `137372→146266°/s³`，有效小反转
   `879→888`，拒绝。
2. **30 Hz midpoint，严格保持末端 q/qd/qdd**：后半个 cut 只有很小零空间，结果
   数值上几乎等于 V3，没有实际收益。
3. **30 Hz midpoint，只保持末端 q**：局部最新轨迹可将 jerk p95
   `136781→112809°/s³`、饱和 `194→143`，但跨日期有效小反转增加约 4–12 次；
   即使限制末端 qd/qdd 偏差仍不满足整体非回退。
4. **15 Hz accepted-action 小幅预测**：0.25 个 60 Hz tick 的预测在排除历史 0727
   上游约 63° IK 分支跳变后，平均滞后约 `64.17→60.83 ms`，但全组有效小反转
   `879→887`；0.5 tick 虽更及时，反转增加更多，拒绝。
5. **延长因果速度窗口**：5→9 行并增加路径效率抑制，直接速度候选饱和进一步降至
   `1146`，但有效小反转升至 `901`。这证明回退不是单纯的 VIVE 速度估计噪声，
   而是 terminal dynamics/accepted anchor 改变后的跨 cut 时间律。

最后新增了 **30 Hz midpoint exact-q+qdd** 研究候选：在半 cut 处使用新的 VIVE
速度，只重排剩余 4 个 120 Hz jerk 样本；正式 action 位置和 cut 末加速度保持精确，
末速度变化限制在 `2°/s` 内，并要求：

- q/v/a/jerk 全部通过硬界；
- 本 cut 不增加 98% jerk 饱和；
- 有效步进激活集合和方向与 V3 一致；
- 当前 cut 归一化平滑分数不得高于 V3；
- VIVE 速度误差至少改善 2%；否则逐关节保留 V3。

六条跨 0727/0731/0802 的 600-row 高运动窗口结果：

| 指标 | V3 | exact-q+qdd | 结果 |
|---|---:|---:|---|
| action→state(t+1) p95 | `0°` | `0°` | 完全保持 |
| requested→action worst p95 | `63.1656°` | `63.1657°` | 被旧 0727 分支跳变主导，数值等价 |
| qref tracking worst p95 | `70.3159°` | `70.3159°` | 数值等价 |
| 最佳跟随滞后 worst | `75.7 ms` | `75.7 ms` | 无改善 |
| jerk p95 worst | `137372.4°/s³` | `137372.4°/s³` | 不变 |
| 98% jerk 饱和总数 | `1522` | `1479` | `-43`，约 `-2.8%` |
| `>0.05°` 相邻反向 | `27` | `26` | `-1` |
| 有效方向反转总数 | `879` | `879` | 完全保持 |
| 硬违规/求解失败 | `0` | `0` | 完全保持 |

这是目前唯一同时保持 action/state、硬界和反转合同的 VIVE 速度候选，但收益仅是小幅
减少饱和，**没有改善跟随滞后**。逐轨迹 p95 仍可能因分位点重排出现很小的正负变化，
且当前证据只有 6 条窗口，不足以注册真机 profile。它保留在离线探针中，默认配置和
真机参数均未改变。

综合判断：在 15 Hz action endpoint 必须严格对齐的合同下，60 Hz VIVE 速度可以用于
重排 cut 内时间律并小幅降饱和，但不能单靠下游插值消除约 60–75 ms 的 endpoint
等待。显著提升跟随必须预测/提前移动 accepted action，现有因果候选又会增加小反转。
下一步需要含原始 60 Hz VIVE、实际 120 Hz qcmd 与同步 robot state 的新 trace，用
状态观测器和带滞回的模式控制区分“持续运动”和“真实换向”；在此之前继续扫 EMA、
cap 或窗口长度不再有工程价值。

## 2026-08-03 V2 responsive 普通区 EMA 真机 A/B

这一轮与上面的 15 Hz action-aligned 时间律研究不同：正式链路仍是
15/60/60/120 Hz 的 V2 responsive，只调整普通、非 guard 区域的 post-IK fast-path
EMA。奇异/边界路径、60 Hz qref admission、120 Hz acceleration planner 和受力急停
均未改变。

对比的预分配 trace 为：

- responsive `alpha=0.70`：`20260803_231952_682329_vive_rm65_actuator_trace`，
  69.91 秒；
- balanced `alpha=0.55`：`20260803_235753_226935_vive_rm65_actuator_trace`，
  108.78 秒，profile SHA
  `99f4441d0eacd0e2321cd2175326a71488859c81d2cb656f3d3a1f4c4c254b1d`。

两段操作者运动并不完全相同，因此不能把所有绝对 RMS 当成严格配对实验；但频谱、
分支证据和用户体感方向一致：

- 左 J4 的 120 Hz `>10 Hz` RMS 从约 `0.01818°` 降到 `0.01067°`；
- 右 J4 从约 `0.02000°` 降到 `0.01437°`；
- balanced 左 J4 qcmd jerk p95 约 `330043→146708°/s³`，右 J4约
  `258797→208807°/s³`；
- balanced 全段 singularity gate 记录均为 `pass`；canonical 大于 `0.5°` 的修改仅
  左 11 帧、右 6 帧，残余振动不是 guard/escape 分支来回切换；
- 排除启动首 1 秒后，108.8 秒内相邻两步均超过 `0.02°` 的 qcmd 真反向仅左 4 次、
  右 7 次；原来的大尖刺问题已基本被隔离；
- IK 慢帧仍有 5 次超过 25 ms、最大约 63.2 ms，可以解释偶发停顿/追赶，但不能解释
  持续的细小高频振动。

balanced 残余频谱在约 30 Hz 仍有明显峰值，来源是 60 Hz qref 的小幅交替变化；
120 Hz planner 会继续衰减该分量，并没有产生新的分支振动。用户确认 alpha=0.55
跟手性没有问题、高频振动有缓解且整体为正优化，因此：

- `rm65_bimanual_position_v2_balanced` 固定为稳定基线，不再原位调参；
- 后继 `rm65_bimanual_position_v2_smooth` 默认关闭，仅把普通区 alpha 改为 0.45；
- 对纯 30 Hz 交替输入，EMA 理论增益从 `0.55/(2-0.55)` 降为
  `0.45/(2-0.45)`，再衰减约 23%；
- 低频群延迟增加约 0.4 个 60 Hz reference frame（约 6.7 ms），90% step settling
  增加约一个 reference frame；
- smooth 仍需独立真机 A/B，不能与 balanced episode 写入同一个 task。切回 balanced
  是完整回退。
