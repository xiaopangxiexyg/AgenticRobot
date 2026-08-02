# RM65 Pose→QPOS 离线方案与结果汇总

更新时间：2026-07-30

本文记录当前 Pose replay / deploy 的 Pose→QPOS 尝试、0727 真机数据上的离线结果，以及
每个方案的回退边界。所有结果均来自只读 evaluator，不连接设备、不下发动作。

## 1. 评估对象与数据

本轮使用本地拉取的 0727 数据集：

```text
remote_data/pose_lift_0727_full_v2
来源：2026-07-27_dmtac_rm65_bimanual_position_v2_responsive
规模：115 episodes，28,987 frames
```

数据集保存的 accepted `q_ref` 是 teacher；`q_ik_raw` 不作为真值。Pose replay 的首帧仍用
保存的 accepted `q_ref` 初始化，之后按 deploy reference clock（默认 60 Hz）执行 Pose→QPOS。
评估器比较三种状态合同：

| 模式 | planner 状态 | 用途 |
| --- | --- | --- |
| `teacher_forced` | 每帧由 teacher qref 重置 | 分离单帧 Pose→QPOS 误差 |
| `closed_loop` | 上一帧 candidate 作为下一帧 anchor | 检查误差累积与奇异逃逸 |
| `runtime_causal` | 与运行时因果顺序一致 | 最接近部署/回放行为 |

门禁同时检查 joint/FK parity、末端 pose、归一化 Jacobian `sigma_min`、q3/q5 结构裕度、
速度/加速度/jerk、hold、branch jump、碰撞与独立 motion certificate；不能只看关节误差。

### 1.1 复现环境与源码边界

0727 全量计算在远端
`/home/user/zc/DexTeleop/datasets/2026-07-27_dmtac_rm65_bimanual_position_v2_responsive`
上执行；`remote_data/pose_lift_0727_full_v2` 是用于本地检查的只读副本。本文补录时复核的
远端环境为：

```text
project:    /home/user/zc/DexTeleop
conda env:  /home/user/miniconda3/envs/dexteleop310
Python:     3.10.20
NumPy:      1.26.4
Pink:       4.2.0
Pinocchio:  3.7.0
qpsolvers:  4.11.0
```

实验时仓库基准提交为 `2f0bc9611b664233b4126925ce56a82c5b442db6`，但 Pose Lift
研究代码位于同步中的 dirty working tree，不能只凭该 commit 复现。与本文结果直接相关的
文件摘要如下：

```text
robot_motion/rm65/dual_arm_ik.py
  5bc866d68ae33c17a712bc3433d71f618c2e118a3b2cc3299047bb43ddee1066
robot_motion/rm65/pose_lift_profile.py
  583d60a9edbd624f8e0aa86773c9fa0ea8a69d6e06a4e4bc9cb4f10f946d077c
robot_motion/rm65/position_collision.py
  4f1c165775620fa2be2d4477438e2fa0e0aafcc1095e432fd1b779e39c829c55
teleop_runtime/debug/evaluate_rm65_pose_lift_parity.py
  9b31a4b40f4f277356ef2909470e1f2eafedc52639f1fd80bd71eec2fcd07f81
teleop_runtime/debug/evaluate_rm65_singularity_wind_tunnel.py
  39a341422ba9287818a10903330f206e4853282949ef9fe5b82aeda52eb3357a
```

左右工具 profile 的 SHA256 分别为
`b15df082549d3d5227f3f9c7318edab3a6e71161314110d289edb48e75eb4074` 和
`71a4c9967ed7634d643da251ea34160dcbdfc51b8d4a4260123f2f658ea772d8`。
后续如果继续修改上述文件，应同时更新本文的结果和摘要，不能把旧结果归因到新实现。

### 1.2 门禁口径

`production` comparison 是严格非回退合同，而不是一个可以由 sigma 收益抵消其他回退的
加权总分。默认绝对数值容差为 `1e-6`；guard band 为 `sigma_min<0.04`，其中 candidate
最小 sigma 不得低于 baseline 超过 `1e-6`。joint error、末端 translation/rotation 的
mean 必须不劣；p95 仅分别允许 `0.25 mm` 和 `0.25°` 的额外余量。最大关节 step、velocity、
acceleration、jerk，Hold、branch jump、motion-certificate violation 计数均不得增加；
guard band 内最大 condition number 也不得变差，且不能出现非有限 condition number。

`sigma_escape_research` 是单独的离线研究合同。它要求 guarded side 的 sigma 至少改善
`max(0.1%, 1e-5)`，允许 joint error mean/p95 各最多回退 `0.15°`、translation
mean/p95 各 `0.02 mm`、rotation mean/p95 各 `0.005°`、jerk 最多回退 `0.1%`；
guard/deep sample 数、mean guard deficit 和硬 motion violation 仍不得回退。该合同通过
只表示“值得继续研究”，从不授予 deploy 或真机权限。

阅读本文结果时遵守三个原则：

1. candidate 未触发而与 V4 完全相同，只能算非回退，不能算收益；
2. 单个最小 sigma 改善不能抵消 Pose、动态、Hold 或 certificate 回退；
3. production 的数值级失败需要结合其他指标判断，但不能擅自忽略。例如 ep96 的 H8
   acceleration max 只发生极小数值回退，同时 `<0.001` 深尾暴露也恶化，因此结论并非只由
   浮点噪声造成。

### 1.3 基础复现命令

单 episode 的 profile 对比可用：

```bash
python -m teleop_runtime.debug.evaluate_rm65_pose_lift_parity \
  --no-motion \
  --dataset-root remote_data/pose_lift_0727_full_v2 \
  --episode 96 \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --pose-lift-profile rm65_pose_lift_v6_structural_branch \
  --compare-baseline-profile rm65_pose_lift_v4_qp_constrained \
  --comparison-contract sigma_escape_research \
  --offline-fixed-solve-iterations 4 \
  --output-json /tmp/rm65_pose_lift_result.json
```

## 2. Profile 演进

| Profile | 主要机制 | 离线结论 | 当前权限 |
| --- | --- | --- | --- |
| V1 `rm65_pose_lift_v1_baseline` | 结构调整前的 deploy 快照 | 原子回退基线；不提供奇异保护 | 可部署 |
| V2 `rm65_pose_lift_v2_continuous` | 合法关节 unwrap、腕部等价解规范化 | 降低部分分支跳变，但不是完整奇异方案 | 显式 opt-in |
| V3 `rm65_pose_lift_v3_bounded_continuous` | V2 + 腕部有界局部搜索，posture cost=0 | parity 常改善，但个别近奇异样本 `sigma_min` 下降、condition 上升 | 显式 opt-in |
| V4 `rm65_pose_lift_v4_qp_constrained` | admitted-anchor Pink QP；速度/加速度约束在求解层 | 0727 代表轨迹中目标追踪总体持平/改善，动态峰值下降，未见奇异指标回退 | **部署显式回退；Pose replay 默认；可真机** |
| V5 `rm65_pose_lift_v5_sigma_escape` | `sigma_min<0.04` 时沿有限差分梯度增加 sigma | fixed2 多数 guarded 轨迹改善；fixed4 的 ep96 极深尾部恶化 | 离线 only |
| V6 `rm65_pose_lift_v6_structural_branch` | V5 + q3/q5 单侧分支锁，默认零点裕度 0.75° | 8 条实际进入 guard 的 guarded 轨迹通过研究门禁；ep96 深尾显著改善，但存在单侧平台 | 离线 only |
| V7 `rm65_pose_lift_v7_structural_escape` | V6 + q3/q5 朝 4° 的软外移目标 | ep96 局部改善，8 条全局复核失败 | 离线失败探针 |
| V8 `rm65_pose_lift_v8_nonlinear_best_candidate` | V6 + 每帧 exact nonlinear best-so-far | ep96 单帧指标改善，但改变后续 anchor 后多条 guarded 轨迹回退 | 离线失败探针 |
| V9 `rm65_pose_lift_v9_predictive_branch_barrier` | V6 + 仅深尾启用预测式平滑单边 Barrier | ep96 深尾改善；其余 guarded/正常轨迹保持 V6；尚未覆盖更多深尾样本 | 离线候选 |
| V10 `rm65_pose_lift_v10_directional_feasibility_guard` | Pink task-conditioned 方向缩放 + q3/q5 平滑 Barrier | 独立风洞消除跨支且无 Hold；真实轨迹阻止必要换支并造成 parity/动态回退 | 离线局部控制探针，禁止部署 |

V11 目前不是已注册的 deploy profile，而是 evaluator 内的 H8 selector 原型；它只组合
已注册 profile 的 detached rollout，不应通过 profile 名称传入真机入口。

## 3. 各阶段详细结果

### 3.1 数据与时间合同复核

在继续调整 IK 前，先复核了三个会影响全部结论的合同：

- 0727 全部 115 个 episode、28,987 帧的 stored Pose 与同帧 accepted `q_ref` FK
  一致。左右最大平移误差分别约 `7.01e-8/7.59e-8 m`，最大旋转误差约
  `1.32e-5/1.55e-5 deg`；错开一帧后误差立即增至毫米/约 0.5° 量级。因此 ep96 是真实
  path-lift 分支漂移，不是 Pose/qref 错帧或保存语义不一致。
- fixed2/fixed4 的多次 Pink 迭代共享同一 outer admitted anchor、同一个上一条正式速度和
  一个物理 `dt=1/60 s`；previous velocity 只在最终 command admitted 后提交。它不是把
  4 个完整控制周期累计到一帧。
- evaluator 原先对非网格末帧存在时间合同错误：resampler 会附加最终 source timestamp，
  但动态证书误用了该短 source interval，而 planner 始终按固定 reference period 积分。
  现已统一按 `1/reference_hz` 计算速度、加速度和 jerk，并增加非网格末帧回归测试。

评估器现在并列保留两套有意不同的观察量：旧的 geometric Jacobian 使用 `0.5 m`
characteristic length 归一化，用于回看 V5/V6 结构实验；新的 task-conditioned
Jacobian 直接使用与 Pink FrameTask 一致的 local SE(3) Log Jacobian、position/orientation
task cost 和关节速度能力归一化，用于 V10 directional demand。前者仍是结构指标，不能
直接解释为 Pink objective 的统一阈值；后续 V10/V11 触发判断应优先使用 task-conditioned
指标、directional demand、active constraint cause 和结构裕度的组合。

### 3.2 V1/V2/V3：连续性与腕部规范化

- V2 的 unwrap/canonicalization 解决的是等价腕部解和 `2π` 连续性，不等价于结构奇异保护。
- V3 的有界局部搜索可降低部分 qref/FK 误差，但在近奇异 episode 中出现 `sigma_min` 略降或
  condition number 上升；因此不能仅凭 parity 改善上线。
- `posture_cost=0/0.025/0.05/0.075/0.1` 不存在跨 episode 支配值。

### 3.3 V4：原部署默认、当前显式回退的下层约束基线

V4 将正式 admitted qref 的速度/加速度区间放入 Pink QP，仍保留 QP 后的关节、步长、碰撞
admission。固定 4 次迭代复核 episode 16/35/80/81：

- raw/admitted 速度和加速度越界计数为 0；
- qref 加速度和 jerk 总体下降；
- 最小实际 `sigma_min` 未低于对照；
- QP 后处理最大额外改变量仅为数值级（约 `0.000445°`，其余轨迹为 0 或噪声）。

在 2026-07-31 新后端提升权限之前，V4 是唯一默认/可部署的 Pose Lift；现在它保留为
Pink 部署显式回退和 Pose replay 默认。V1 仍可作为 Pink 路径的完整原子回退。

### 3.4 速度追随与 posture 候选：否决

尝试过两类 V4 上的软连续性：

1. 关节 qdot 连续性：可降低部分 acceleration/jerk，但会延续上一帧朝奇异区的内部关节速度，
   16/35/80/81 的右臂最小 sigma 均下降。
2. Cartesian qdot 连续性，并在 guarded band 中禁用继续降低 sigma 的预测速度：episode 16
   不再回退，但 episode 81 的最小 sigma 仍下降且 jerk 上升。

另有双 posture 候选和有限时域 H8/L8 选择器。严格终点 rejoin、整段 sigma、pose、动态和
certificate 门禁下，候选选择次数为 0；放宽条件时只有数值级收益并伴随 parity 回退。该路径
没有进入 production。

### 3.5 V5：sigma escape

对全部 115 个 episode 的 teacher qref 做 Jacobian 排名，只有 9 个进入 `sigma_min<0.04`：
`80, 81, 37, 96, 16, 35, 59, 68, 53`。另取 8 个正常 episode，V5 与 V4 的 qref、pose、
Jacobian、hold、jerk 逐项相同，说明 guard 未泄漏到抽样正常工作区。

- fixed2：9 条中 8 条实际进入 guard 的轨迹改善；累计 `<0.01` 样本 `66→58`，`<0.001`
  样本 `10→8`，guard deficit 约 `26.3083→25.6854`。
- fixed4：多数轨迹改善，但 ep96 最小 sigma 约
  `1.557e-4→7.828e-5`，`<0.001` 样本 `5→10`，出现极深尾部回退。
- ep96 的 teacher qref 最小 sigma 约 `0.01055`，而 V4 closed-loop Pose Lift 可降到约
  `1e-4`；说明该问题主要由 Pose→QPOS 重建路径产生。

结论：V5 是有价值的研究方向，但不能作为稳定 profile；cost 增大不能直接解决问题。

### 3.6 V6：结构 q3/q5 单侧分支锁

V6 在 sigma guard 内选择 q3/q5 中更接近零的结构关节，锁存当前正/负侧，直到 release band
外；Pink QP 内禁止穿越零点，并保留 V4/V5 的硬约束。

- ep96 fixed2/fixed4 最小 sigma 约 `0.002132/0.002136`，V4 对应约
  `8.70e-6/1.56e-4`；fixed4 `<0.001` 样本由 `5→0`，无 hold。
- 9 条 teacher-guarded 轨迹中，8 条实际进入 guard 的轨迹均通过研究门禁；ep53 没有有效
  guard 证据，单独标记为无证据样本。
- 8 条正常轨迹与 V4 的 qref、sigma、误差、hold、jerk 逐项相同。
- 裕度 `1.5/2.0°` 虽提高 ep96 sigma，但 jerk 增加约 `3.7%/2.7%`；裕度 `0.5°` 安全余量
  更窄，因此仍采用 `0.75°`。
- V5 cost 离线提高到 `0.3/0.4/0.5` 时，ep96 sigma 约为
  `0.00234/0.00658/0.01144`，但全量复核出现 parity、guard 暴露或 certificate 回退。

V6 的代价是深奇异区存在约 `0.75°` 的单侧安全平台，不能宣称完全跟手；但它是目前最可靠
的离线结构奇异保护基线。

### 3.7 V7：软外移目标

V7 在 V6 上增加 q3/q5 朝 4° 的软目标，尝试减少单侧平台粘滞。

- ep96 单独 fixed4：平均最大关节误差约 `1.696°→1.562°`，最小 sigma 约
  `1.56e-4→6.7e-3`（参数扫描版本），局部效果明显。
- 8 条 guarded episode 全局复核失败：episode 80/16/68 出现速度或加速度 certificate
  回退，episode 35/59 sigma 改善不足，并在其他组合出现 pose/parity 回退。
- 提高 cost 到 `0.20` 或把激活范围收窄到 `sigma<0.005`，仍不能通过全局门禁。

因此 V7 只保留为可复现的离线失败探针，禁止部署和真机 dispatch。

### 3.8 V8：单帧 nonlinear best-so-far

V8 对 fixed4 的第 2–4 次 nonlinear iterate 做真实 FK、真实归一化 Jacobian、joint/step
certificate 和 endpoint collision 复核，不再机械返回最后一次迭代。ep96 上它把最大关节误差
从约 `14.84°` 降到 `13.76°`，最小 sigma 保持 V6 水平。

但 closed-loop 中选中不同单帧 candidate 会改变下一帧 anchor。扩展到全部 guarded 轨迹后，
episode 80/16/68 出现动态证书回退，episode 16/81/35/59 还出现 parity、pose 或 guard
暴露回退。说明“当前帧 exact 最优”不等于“关节路径提升更优”。V8 因此保留为失败探针，
不作为 V6 的改进。

### 3.9 V9：预测式深尾混合 Barrier

初版 V9 对整个 guard band 使用上一条 admitted 速度预测 150 ms，并以连续单边 CBF

```text
sign*qdot + gain*(sign*q - margin) >= 0
```

替代 V6 的一步硬 margin。统一 gain 扫描不存在跨轨迹支配值：更早/更强的 Barrier 可改善
episode 80/81/37/16，但会提高 ep96 jerk；继续增强又会损害 ep16 pose。

因此注册的 V9 收窄为混合模式：

- `sigma>=0.01` 的普通 guard 区完全沿用 V6 hard margin；
- 仅当前或 150 ms 预测 `sigma<0.01` 时启用 gain=`6` 的平滑 Barrier；
- 所有 V4/V5/V6 的速度、加速度、碰撞和 admission 证书保持不变；
- `hardware_dispatch_allowed=false`。

0727 fixed4 closed-loop 结果：

- ep96：右臂最小 sigma `0.002136→0.002363`，`<0.01` 样本 `52→51`；最大关节误差
  `14.844°→14.746°`，平均最大关节误差 `1.674°→1.667°`；最大速度
  `137.08→136.39 deg/s`，最大 jerk `368499→368169 deg/s³`；pose p95 不变，无
  velocity/acceleration certificate 违规，深尾研究门禁通过。
- ep80/81/37/35/59/68 与 V6 的最小 sigma 完全相同；ep16 仅有 `7.5e-9` 数值差，没有
  pose/parity/dynamic/hold/branch 回退。
- 8 条正常轨迹 `88,102,100,46,94,38,2,39` 的完整 compact mode report 与 V6 JSON
  逐项完全相等。

现有 `sigma_escape_research` 合同要求每个 guarded side 都必须得到实质 sigma 增益，因此
选择性深尾 V9 的整组 verdict 仍为 false；这不是把失败改成通过。当前证据只支持：
V9 是对已观察到的 ep96 深尾有效、对其余样本中性的离线候选。需要更多独立深尾 episode 和
局部奇异风洞验证后，才可讨论替换 V6，更不能直接进入部署。

### 3.10 跨日期与旧 profile 扩大验证

为避免结论只来自 0727 responsive 数据，进一步盘点了远端历史数据。数据准入仍严格要求：
Pose replay 的每一帧必须同时具有左右 accepted `q_ref`；不能用 observation qpos 或
`q_ik_raw` 替代 teacher。

盘点结果：

- `2026-07-23_dmtac_rm65_bimanual_position_v1`：2 episodes、326 帧，teacher 完整；
- `2026-07-24_dmtac_rm65_bimanual_position_v1`：1 episode、55 帧，teacher 完整；
- `2026-07-27_dmtac_rm65_bimanual_position_v1_old`：35 episodes、17,422 帧，teacher 完整；
- writing 合并集 503 episodes/82,266 帧、0724 v2.2/filtered 241 episodes/35,667 帧、
  0725 filtered 310 episodes/68,408 帧虽然 schema 包含 replay-qpos 字段，但这些字段在
  全部 episode 中均为空。它们不能用于 accepted-qref parity，evaluator 按设计 fail-closed，
  没有退回 observation qpos。

由此，accepted-qref/Jacobian 可审计总覆盖从原来的 115 episodes/28,987 帧扩大为
153 episodes/46,790 帧。新增 38 个 episode 的 full V9↔V6 fixed4 closed-loop 覆盖
17,803 source frames，经 60 Hz 重采样后共 72,758 个 reference：

- 0723/0724 的 3 条跨日期正常轨迹最小 teacher sigma 为 `0.195–0.213`，V9 与 V6 的
  compact report 逐项完全相同；
- old 35 条中只有 episode 13/17 进入 teacher guard，最小 sigma 分别约
  `0.02168/0.03065`，其余正常轨迹最低约 `0.06161`；
- old 全 35 条 V9 与 V6 的 compact report 全部逐项完全相同，35/35 的 production
  comparison 均非回退；closed-loop 没有任何 episode 进入 `<0.01` 深尾；
- ep13/17 的研究 verdict 为 false，只因为 V9 没有达到“必须实质提高 sigma”的研究收益
  条件，不是出现回退；
- 单独比较 V6↔V4 时，ep13/17 均通过 23 项研究门禁：最小 sigma 分别
  `0.022089→0.022132`、`0.031528→0.031579`，pose p95、最大 jerk 和 motion
  certificate 无回退，平均关节误差仅增加约 `0.0033°/0.0013°`。

扩大结果强化了两个边界：

1. V9 的 `<0.01` 混合开关没有泄漏到新增的正常或软 guard 数据；
2. 新增数据没有提供第二个 ep96 级深尾，因此不能据此提升 V9 权限。V9 的正向收益证据
   仍主要来自 ep96，下一步仍需独立深尾数据或局部奇异风洞。

### 3.11 路径提升主线与统一任务度量

后续研究不再继续扫描 V9 gain/margin/posture，也不再以单帧 candidate 最优作为
closed-loop 路径选择依据。问题正式拆成：

1. 当前结构分支内的方向性可实现性下降；
2. 局部 IK 的长期 anchor 漂移和错误 branch lift；
3. stay-in-branch 确实不可行时的显式授权 rebranch。

第一阶段已加入纯观测诊断，所有默认 command、profile、QP 和 admission 均未改变：

- 保留原来的 0.5 m characteristic-length geometric Jacobian 指标；
- 新增直接调用 Pink `FrameTask.compute_error/compute_jacobian` 的 local SE(3) Log
  Jacobian；
- 使用 Pink 的 position/orientation cost 左乘任务行，并以正式 joint velocity limit
  缩放列，得到 `sigma_task_velocity`；
- 记录每个左奇异方向的任务需求
  `|u_i^T xi_desired|/(sigma_i+epsilon)`；
- 可选 `--include-constrained-feasibility` 运行 detached LP/QP，报告当前
  position/velocity/acceleration/branch/collision inequality 下的 strict directional
  `alpha`，以及 alpha=1 时的最佳约束跟踪残差方向；
- evaluator 同时记录真正可因果使用的 `pre_solve` 指标和 post-admission
  truth/raw/admitted 指标。detached 求解不积分配置、不更新 motion-limit history。

episode 96 V4 fixed4 closed-loop 的首轮只读结果表明新指标能提前描述困难方向：

- 右臂 `sigma_task_velocity_min` 最低约 `2.64e-4`；
- pre-solve 弱方向需求峰值约为正式关节速度能力的 `58.81x`；
- pre-solve strict directional alpha 最低约 `1.45e-5`；
- strict alpha 首次低于 `0.5` 在 sequence 252，而 geometric sigma 首次低于 `0.04`
  在 sequence 365，提前约 113 个 60 Hz reference（约 1.88 s）；
- directional demand 首次超过 `1x` 在 sequence 249；当时 geometric sigma 仍约
  `0.113`，说明只靠固定 geometric sigma 阈值会明显晚于真实约束冲突。
- 右臂第一次发生明确结构分支事件在 sequence 435：V4 q3 已跨到约 `-0.087°`，
  同帧 teacher q3 仍为约 `+3.652°`。teacher 整段未跨 q3 零点；V4 在 sequence 450
  最接近零点时约 `+0.054°`，随后继续沿错误 lift。`alpha<0.5` / demand `>1x`
  分别比首次未经授权的 q3 符号跨越提前约 183/186 个 reference（约 3.05/3.10 s）。

这些数值尚不构成 V10 阈值。`alpha` 也会受到正常的速度/加速度约束影响，必须在独立正常集、
guard 集和局部奇异风洞上测 false positive/negative 后才可用于状态机。当前权限不变。

第一批非深尾对照进一步明确了这一边界：

- episode 0/1/2 共 3549 个 reference，两臂 strict directional alpha 全程为 `1.0`，
  `sigma_task_velocity_min>=0.248`，方向速度需求峰值不超过 `0.561x`；
- 扩展到 episode 38/39/46/88/94/100/102 的 5516 个 reference 后，
  `sigma_task_velocity_min>=0.240`，方向需求峰值仍不超过 `0.862x`；
- 但 episode 100 左臂 strict alpha 一度降至 `0.0172`。该帧
  `sigma_task_velocity≈0.337`、方向需求约 `0.084x`、最佳约束跟踪残差仅约
  `5.4e-4`，并不是奇异方向能力不足；
- 新增的 active-constraint 分类确认，此处由 `ConfigurationLimit` 和
  `_AdmittedReferenceMotionLimit` 共同约束。episode 94/102 的轻微 alpha 下降也属于
  需要按约束来源解释的 feasibility 事件，而不能自动归类为奇异。

因此 V10 的触发合同必须联合使用 `sigma_task`、directional demand、strict alpha、
tracking residual、q3/q5 structural margin 和 active constraint source。strict alpha
只能回答“原方向在全部当前约束下能严格实现多少”，不能单独回答“是否进入结构奇异”。

同口径的 7 条 guard episode（16/35/37/59/68/80/81，共 9065 个 reference）提供了
进一步区分：

- 每条 guard episode 都至少有一侧 directional required utilization 峰值超过 `1x`
  （最低为 episode 59 的 `1.08x`），而上述 10 条非深尾对照的全局峰值为 `0.862x`；
- guard 侧的峰值分别可达 `12.12x/2.42x/2.28x/2.07x/9.72x/3.53x/12.71x`；
- episode 59 右臂 `sigma_task` 最低仍约 `0.132`，但方向需求达到 `2.07x`，证明
  单独使用 sigma 阈值仍会漏掉“构型尚可、当前目标方向困难”的情况；
- guard 集 strict alpha 最低落在约 `0.044–0.711`，但非深尾 episode 100 因普通
  joint limit 可更低至 `0.017`，再次说明 active constraint cause 和 directional
  demand 必须共同进入状态判断。

这只是由已有 geometric-guard 数据挑选出的首轮辨识结果，不是独立 held-out 风洞，
因此不能据此固定 `1x` 等生产阈值。

远端扩展扫描随后覆盖了 0727 responsive 的全部 115 个 episode、117030 个 60 Hz
closed-loop reference：

- 39 个 episode 的 pre-solve directional required utilization 峰值超过 `1x`，
  20 个超过 `2x`，4 个超过 `5x`；
- 只有 11 个 episode 的 `sigma_task<0.1`，只有 8 个 geometric sigma `<0.04`；
  directional demand 确实能覆盖“构型尚未进入传统 guard、但当前目标方向已经困难”的
  中间区域；
- 以 episode 最大关节 parity error `>3°` 作为只读结果标签时，demand `>1.5x`
  在本集合得到 `TP=26, FP=1, FN=0, TN=88`；demand `<=1x` 的 episode 没有一个
  超过 `3°`；
- 这个列联结果来自同一 0727 集，只能说明辨识力，不能把 `1.5x` 直接注册成生产阈值。

对风险层做逐帧复核：

- 12 个 demand `>2x`、但不在原 deep-guard 组的 episode 全部没有 q3/q5 符号跨越；
- 19 个 demand `1–2x` 的边界 episode 也全部没有 q3/q5 符号跨越；
- 7 个原 deep-guard episode（16/35/37/59/68/80/81）同样没有跨越；
- 因而 39 个 demand `>1x` episode 中，目前只有 episode 96 出现明确的错误结构
  crossing。其余样本主要是 stay-in-branch 的方向需求/动态可行性问题。

跨日期 held-out 只能获得 0723 的 2 条和 0724 的 1 条正式 DMTAC episode，共 1877
references。三条的 demand 峰值不超过 `0.504x`，最大关节误差不超过 `0.981°`，
无 branch crossing。0723 episode 0 的 strict alpha 曾低至 `7.9e-5`，但 demand、
`sigma_task` 和 parity 均健康，进一步证明 alpha 不能独立触发奇异保护。0725 raw/v2.2
数据缺少 accepted `action.replay.*_arm_joint_target`，因此被排除，不能作为 parity 真值。

这批扩展数据支持如下职责划分：

- V10 应处理广泛的 stay-in-branch directional feasibility，触发必须包含 demand、
  persistence、tracking residual、structural margin 和 constraint cause；
- V11 应处理少数跨帧 branch path-lift 失败，不能因单帧 demand 高就请求 rebranch；
- 真正独立的危险 held-out 仍不足，局部奇异风洞仍是进入 V10 控制实现前的必要门禁。

下一主线固定为：

```text
统一任务度量与独立奇异风洞
        ↓
V10 directional feasibility stay-in-branch guard
        ↓
V11 short-horizon discrete branch path lifting
        ↓
V12 explicit authorized rebranch
```

V11 的 terminal contract 必须是 task-space tube、合法 branch 和可继续执行的动态状态，不得
再要求回到 V4 的同一个关节解。离线 Pose replay 可以使用真实 future Pose；实时 deploy
只能使用因果 buffer/外推，并且二者必须注册成不同能力。

### 3.12 局部奇异风洞与 V10 directional feasibility guard

新增只读入口：

```bash
python -m teleop_runtime.debug.evaluate_rm65_singularity_wind_tunnel \
  --no-motion \
  --dataset-root /home/user/zc/DexTeleop/datasets/2026-07-27_dmtac_rm65_bimanual_position_v2_responsive \
  --all-episodes \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --profile rm65_pose_lift_v4_qp_constrained \
  --profile rm65_pose_lift_v10_directional_feasibility_guard \
  --split heldout \
  --max-seeds 16 \
  --offset-deg -0.75 --offset-deg 0.75 \
  --initial-motion stationary \
  --direction weakest \
  --direction-sign both \
  --scenario persistent_push \
  --duration-sec 0.5 \
  --compact \
  --output-json /tmp/rm65_pose_lift_v10_wind_heldout.json
```

风洞从保存的 accepted `q_ref` 取真实其余关节构型，只替换一个 q3/q5 结构坐标；目标沿
同一 Pink weighted local-SE(3) Jacobian 的固定奇异方向生成。它执行连续 closed-loop
rollout，而不是互不相关的单帧 probe。tune/held-out 按 seed identity 确定性拆分；两组
不能合并后重新调参，否则 held-out 失去意义。上面的命令是最终 held-out 口径；把
`--split heldout` 改成 `--split tune` 可复核调参集。

V10 当前实现：

- 仅在结构关节进入 8° band 且目标方向所需速度能力超过 enter 阈值时，按奇异方向分别
  缩放 local SE(3) error；离开结构边界的 modal direction 使用更宽松上限；
- q3/q5 使用 0.25° margin、gain=6 的单边平滑 Barrier；
- V4 的 position/velocity/acceleration/collision 和最终 admission 证书保持；
- `hardware_dispatch_allowed=false`，没有真机权限。

风洞首次运行暴露出一个独立状态错误：Planner 构造时的零配置会先锁存正分支；随后
`reset()` 到负分支近奇异配置时，旧 latch 因仍在 release band 内而未清除，导致首帧被
错误推过零点。`reset()` 现在被明确视为新的 authoritative motion epoch：先清除旧
structural latch，再从本次 measured/seed q 重新锁存。该修复不改变同一 epoch 内的分支
连续性。

修复后，0727 responsive 风洞结果为：

| 集合 | 有效 case | V4 跨支/Hold | V10 跨支/Hold | V4 最坏 `sigma_task` | V10 最坏 `sigma_task` |
| --- | ---: | ---: | ---: | ---: | ---: |
| tune | 36 | 10 / 0 | **0 / 0** | `5.14e-4` | `2.75e-3` |
| held-out | 52 | 10 / 0 | **0 / 0** | `2.96e-4` | `2.93e-3` |

这证明方向选择性 stay-in-branch 控制在“目标应留在当前分支”的局部问题上有效，也避免了
V6 的全程 Hold。它仍不构成 production 通过：

- 合成持续推压中，V10 为维持当前分支会把运动转移到其他关节；部分 q5 case 的最大
  加速度接近已声明的 QP 上限，held-out 最坏约 `6808 deg/s²`；
- 与 V4 的逐 case 比较中，held-out 的最小 sigma 为 `30` 改善、`16` 相同、`6` 回退，
  即改善不是逐 case 支配；
- 最大 translation error 为 `7` 改善、`45` 回退；最大 rotation error 为 `13` 改善、
  `39` 回退。多数回退幅度不大，但方向非常一致，说明 V10 为守住结构分支系统性牺牲了
  task tracking；
- 这说明单纯 stay-in-branch 仍会在持续不可实现目标下积累残差或使用高动态绕行。

随后对真实 guard episodes `16/35/37/59/68/80/81/96` 执行 V10↔V4、
fixed4、runtime-causal production comparison。结果为 **8/8 verdict false**：

- episode 16/68/81 出现明显 joint parity、Pose 与动态回退；ep68 最大 acceleration
  约 `1524→8472 deg/s²`，且新增 motion-certificate 失败帧；
- 其余 episode 也至少存在严格 production 指标回退；
- 真实 Pose 路径中存在合法或必要的结构换支。一个没有 future/branch plan 的
  stay-in-branch controller 无法同时保证复现和避奇异。

风洞和真实轨迹并不矛盾。风洞的问题定义是“目标持续推压时仍留在当前 branch”，所以
零 crossing 是明确收益；真实 replay 的目标是复现 recorded Pose path，其中一部分路径
允许或需要结构换支。V10 将“所有 crossing 都危险”当成了控制合同，因此在合成局部任务上
有效，在完整 path lift 上却会把正确换支也拦住。它解决了问题 A（同一分支内的方向可行性），
没有解决问题 B/C（跨帧分支选择与授权换支）。

因此 V10 的最终定位是：

> 可解释、有效的局部 stay-in-branch 控制探针；不是整体 Pose Lift，也不优于默认 V4。

下一步必须进入 V11：在短时域内同时比较 stay-q3、stay-q5、双 Barrier 和 authorized
rebranch 路径，并用 task-space terminal tube 接回；不能继续通过 V10 gain、margin 或
min-scale 扫描来掩盖“目标实际需要换支”的离散规划问题。

### 3.13 V11 H8 task-space terminal 原型

`evaluate_rm65_pose_lift_parity` 现支持一个仅离线的有限时域原型：

- primary 与 alternate profile 从同一 authoritative anchor 出发；
- 8 个 reference 使用 fixed4 detached rollout，只提交一个完整认证过的 plan；
- terminal 可选择 `task_space_tube`，允许合法的不同关节 lift，不再要求回到 V4 的
  同一 q；
- 输出逐窗口 `decision_checks`、guard-only 拒绝原因和候选路线统计；
- 默认动态相对 slack 为 `0`。额外的 `--adaptive-posture-dynamic-relative-slack`
  只用于离线敏感性检查，不改变生产门禁或 profile。

严格动态非回退的 ep96 复现命令如下。它不是新 deploy profile，而是 evaluator 内的
detached horizon selector：

```bash
python -m teleop_runtime.debug.evaluate_rm65_pose_lift_parity \
  --no-motion \
  --dataset-root /home/user/zc/DexTeleop/datasets/2026-07-27_dmtac_rm65_bimanual_position_v2_responsive \
  --episode 96 \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --pose-lift-profile rm65_pose_lift_v4_qp_constrained \
  --alternate-pose-lift-profile rm65_pose_lift_v6_structural_branch \
  --compare-baseline-profile rm65_pose_lift_v4_qp_constrained \
  --comparison-contract production \
  --mode runtime_causal \
  --offline-fixed-solve-iterations 4 \
  --adaptive-posture-horizon-steps 8 \
  --adaptive-posture-lock-steps 8 \
  --adaptive-posture-terminal-mode task_space_tube \
  --adaptive-posture-terminal-translation-m 0.002 \
  --adaptive-posture-terminal-rotation-deg 0.5 \
  --compact \
  --output-json /tmp/rm65_pose_lift_v11_ep96_strict.json
```

2% 敏感性检查只是在同一命令中增加：

```bash
--adaptive-posture-dynamic-relative-slack 0.02
```

该参数允许 selector 在候选窗口内部接受最多 2% 的 velocity/acceleration/jerk
max/mean 相对回退，但最终 `production` comparison 仍按零回退合同评估，因此不能把
“候选被选择”误读为“候选获得生产权限”。

在 episode 96 上，以 V4 为 primary、V10 为 alternate 时，165 次重规划中只有 17 次
进入 guard。V10 只有 3 个窗口同时满足 sigma 改善、Pose 有界和 terminal tube，但这些
窗口均未通过动态非回退，因此最终 `0/1317` reference 选择 V10。把 alternate 换成 V6
后，有 10 个 guard 窗口满足 sigma/Pose/terminal 条件，但严格动态门禁仍全部拒绝。

窗口明细说明这种拒绝不是单一问题：

- 浅 guard 中一部分 V6 候选只增加约 0.5–1.3% 的速度，但也常增加 2–9% jerk 均值；
- 深尾候选可出现约 40% 速度增幅或数倍 acceleration/jerk；
- 因此不能通过无界放宽 selector 来换取 sigma。

为验证浅区的边界，只执行了 1% 与 2% 两个预先有明细依据的离线检查。1% 仍为 0 次
接管；2% 在 episode 96 产生 4 个 H8 plan、32 个 reference 接管，局部指标接近正向，
最小 sigma 从约 `1.557e-4` 提升到 `1.821e-4`，但 `<0.001` 暴露从 5 帧增至 8 帧。
同时 acceleration max 从约 `4124.391573` 数值级增至 `4124.391656 deg/s²`。即使将后者
视作数值边界，深尾持续时间也明确恶化，所以严格 production verdict 未通过不是单纯的
浮点误判。

随后冻结同一参数，对真实 guard episodes `16/35/37/59/68/80/81/96` 做扩展验证：

| episode | V6 接管 reference / plan | verdict | 主要结果 |
| ---: | ---: | --- | --- |
| 16 / 35 / 68 | `0 / 0` | pass | 与 V4 完全相同，不构成收益 |
| 37 | `136 / 17` | fail | sigma 改善；joint parity 与 Pose 回退 |
| 59 | `40 / 5` | fail | joint/dynamics 略改善；rotation mean 回退 |
| 80 | `112 / 14` | fail | sigma 改善；joint parity 与 Pose 回退 |
| 81 | `208 / 26` | fail | sigma 改善；joint parity 与 Pose 回退 |
| 96 | `32 / 4` | fail | 多数均值略改善；深尾暴露和严格 acceleration max 未通过 |

实际数值进一步说明“sigma 改善”为什么不能直接等于整体改善：

- episode 37：最小 sigma `0.0191946→0.0203221`，但 joint error mean
  `1.8491→1.8972°`、translation mean `0.00057726→0.00058327 m`、rotation mean
  `0.044774→0.045125°`，复现质量一致回退；
- episode 59：最小 sigma 基本保持 `0.0215612`；joint error mean
  `0.89855→0.89702°`、acceleration p95 `454.54→454.47 deg/s²`、jerk p95
  `39040→38870 deg/s³` 略有改善，但 rotation mean
  `0.0507633→0.0507775°` 回退，未形成支配；
- episode 80：最小 sigma `0.0240489→0.0251826`，同时 joint error mean
  `2.08122→2.10394°`、translation mean `0.00078825→0.00079048 m`、rotation mean
  `0.0673018→0.0674407°`；
- episode 81：最小 sigma `0.0113451→0.0129969`，但 joint error mean
  `2.19458→2.27348°`、translation mean `0.00078049→0.00078723 m`、rotation mean
  `0.0589532→0.0594004°`；
- episodes 16/35/68 完全没有接管，因此 pass 仅表示 selector 没有污染 V4，并不提供
  V11 收益证据。

因此 H8 selector 本身已经能保持未触发区原样，也证明 task-space terminal 不会强制候选
回到 V4 q；但 V4/V6/V10 现有整条 profile rollout 不是足够的 V11 候选集合。下一步必须
生成显式 stay-q3、stay-q5、双 Barrier 和可授权 rebranch 的多分支路径，而不是继续放宽
dynamic slack。当前没有整体优于 V4 的选择。

### 3.14 官方 RM65 全解 IK 与分层控制探针

在 V4–V11 的局部 Pink 路线没有得到整体支配解后，新增了两个 `todos/probes/` 下的
纯离线探针：

- `rm65_official_all_ik_probe.py`：核查官方全解、全部候选、SDK/URDF FK、求解延迟、
  60 Hz closed-loop 和共享 Cartesian alpha governor；
- `rm65_official_branch_anchor_probe.py`：核查 soft Pink anchor、source-rate 全解路径、
  60 Hz 中间帧投影、已有 qcmd planner 和官方连续遥操 IK。

两者均只初始化空 handle 的算法库，不连接机械臂、不枚举设备、不 dispatch，也不拥有
任何 control authority。SDK 算法版本为 `1.6.0-8a02e510`。默认 deploy、Pose replay
profile 和真机命令没有改变。

#### SDK/模型合同与 source-rate 全解能力

先把 recorded TCP Pose 通过现有 tool profile 转回 flange Pose，再比较官方 FK 与项目
Pinocchio FK。0727 responsive 全部 115 episodes、28,987 source frames 的结果为：

- 左右 source all-IK 均为 `0` 失败；
- 最坏 per-episode source solve p99 为左 `9.941 us`、右 `10.736 us`；
- 官方 FK 对 stored target 的最坏误差为左 `3.892 um / 0.000472°`、右
  `4.200 um / 0.000660°`；
- 每帧都能找到与 recorded accepted `q_ref` 同支、数值接近的精确候选。

这证明 bundled RealMan 全解库、当前 RM65 型号、base/flange/TCP 变换和项目 joint
约定已经对齐。至少在这批真实数据上，官方全解可作为 branch oracle；没有必要先引入
EAIK/IKFast 才能验证多解路线。

#### 不能对每个 60 Hz Cartesian 插值点强制精确全解

deploy 的现有因果重采样是 `15 Hz source Pose → 60 Hz xyz-linear + shortest-slerp`。
直接对每个 60 Hz 插值 Pose 调用全解，完整数据出现：

- 左臂 `53`、右臂 `223` 个无 project-valid exact candidate 的 reference；
- q3/q5 branch sign mismatch 为 `0`，说明主要失败不是选错符号分支；
- episode 73 的两个 15 Hz 端点均精确可达，但右臂有 `81` 个中间 Cartesian 插值点
  无合法精确 IK。这是可达集合非凸的真实反例；
- 全量最坏 joint deviation 为左 `0.436°`、右 `2.793°`；
- 最坏 acceleration 达左 `8010°/s²`、右 `34663°/s²`，右侧主要来自 exact/no-exact
  边界上的关节路径不连续。

因此不能把 `all-IK(T_interpolated)` 直接当作 60 Hz qcmd，也不能在 exact IK 暂时消失时
简单 Hold。Cartesian 线段的两个合法端点不保证线段内部也存在合法 RM65 解。

#### 当前最强正结果：只解 materialized source Pose，再插值 qref

另一条路径只在真实 policy/source 帧上枚举全解，使用上一 source qref 选择最近的
collision-safe 分支，再在两个 source qref 间做 60 Hz joint interpolation。完整 115
episodes 的结果为：

- `0` source failure、`0` runtime endpoint collision failure；
- 对 recorded qref 的最坏关节差为左 `0.042641°`、右 `0.030419°`；
- 最小 geometric sigma 为左 `0.0105325`、右 `0.0102807`，保持 teacher 级别，没有
  复现 V4 在 episode 96 中跌到 `1e-4` 的错误 branch drift；
- 最坏 Pose 路径误差为左 `0.296 mm / 0.0926°`、右 `13.626 mm / 0.4827°`；
- 最坏速度为左 `86.14°/s`、右 `93.37°/s`；
- 最坏 acceleration 为左 `6344.7°/s²`、右 `9108.4°/s²`，基本等于 recorded qref
  本身的拐点动态，而不是新的求解器放大。

episode 96 的右臂最小 sigma 保持约 `0.01056`，最大关节差约 `0.03042°`；episode 73
不再要求那 81 个不可达中间 Pose，最大 Pose 误差只有约 `0.107 mm / 0.0356°`。episode
34 右臂的 `13.626 mm` 是 joint-linear 路径与 SE(3)-linear 路径之间的真实几何差，
不是 branch/index 错误；同一 episode 的 V4 右臂最大 Pose error 约为
`57.98 mm / 1.18°`，仍明显更差。

这个结果第一次在完整真实集上同时获得：

- teacher 关节路径近似复现；
- 深奇异 branch drift 消失；
- source 端全解零失败；
- endpoint collision 零失败。

它目前是比 V6–V11 更有价值的下一候选，但还不是 production backend。它改变了
`Pose interpolation → IK` 为 `source Pose IK → q interpolation`，必须作为显式的新
interpolation/path-lift contract 注册，不能悄悄替换 V4。

全解也明确了能力边界。episode 96 每个 source Pose 的 SDK raw candidate 恒为 8 组，
project-valid candidate 为 1–8 组；但右臂“同一 Pose 下最佳 exact candidate”的
最小 sigma 仍只有约 `0.010552`，相对 teacher 的最大 sigma gain 仅
`0.000635`。所以全解能防止 V4 额外漂移到 `1e-4`，却不能把目标本身从
`~0.0105` 的奇异邻域搬走。若以后需要进一步提高结构裕度，仍必须显式接受 bounded
Pose slack、目标限速或 rebranch，不能宣称“枚举 8 解自然消除所有奇异”。

#### 中间帧 exact projection 的收益与代价

以 source q 路径作为无漂移 seed，再仅在有 exact solution 时投影 60 Hz Cartesian
target，可以把完整集最坏 Pose error 压到约 `0.107 mm / 0.0832°`，并保持最小 sigma
在 teacher 级别，endpoint collision 仍为 0。但代价是：

- 最坏关节差回到左 `0.436°`、右 `2.793°`；
- 右臂速度峰值升到 `117.65°/s`；
- episode 73 仍有 81 个右臂中间帧必须退回 source-q seed；
- exact/fallback 切换需要额外的跨帧连续性合同。

因此该投影更适合作为离线诊断或带迟滞的可选修正，不应成为每帧默认路径。

#### 已明确否决的组合

以下路径均保持可复现，但不继续调参：

- **Pink soft branch anchor**：小 cost 无法阻止 episode 96 漂移；大 cost 虽提高 sigma，
  却引入约 55° joint deviation 和 11–28° Pose error。它不能把局部 Pink 变成全局
  branch solver。
- **source-q seed + frame-local Pink residual**：没有累积 branch drift，但 episode 34
  右臂 translation 只从 `13.6 mm` 降到 `11.6 mm`，acceleration 却从约
  `1623→3691°/s²`，其他压力轨迹也回退。
- **共享 Cartesian alpha governor**：在 episode 73 的非凸可达集合上失效，
  alpha mean 约 `0.113`，479/524 帧小于 1，最大误差扩大到约 `0.20 m / 47°`。
  对可行性做普通二分隐含了单调/星形可达假设，该假设不成立。
- **现有 AccelerationLimitedJointTrajectoryPlanner 直接追 source-q path**：
  ep34/73/96 出现约 18°、24°、55° 的跟踪滞后，ep96 还有 116 次 command endpoint
  collision rejection。它是现有高率执行器，不等价于“给定任意 15 Hz 大目标即可
  无误差完成”的 Ruckig/reference governor。
- **官方 `rm_algo_ik_remote` 直接替换 V4**：ep73 的 1048 次双臂调用中有 481 次
  `IK LIMITED`，产生 13 个碰撞拒绝帧；右臂最大关节差约 `162°`、Pose error 约
  `194 mm / 50.8°`。官方文档也限定它用于连续 Cartesian Pose，并要求 command
  difference/dt 不超过安全速度；当前不可达插值反例不满足直接替换前提。

#### 新的分层边界

后续不再让一个 IK profile 同时承担几何分支、时间缩放和 qcmd 平滑。研究链路调整为：

```text
15 Hz materialized source Pose
        ↓
official all-solution IK
        ↓
causal branch selector + collision endpoint certificate
        ↓
explicit 60 Hz qref interpolation contract
        ↓
独立的 velocity / acceleration / jerk reference governor
        ↓
现有最终 admission 与高率 qcmd 执行
```

#### qref→qcmd 动态可行性：离线全量投影结果

`source Pose all-IK → joint interpolation` 解决的是几何分支，不应把线性
qref 直接解释成可下发 qcmd。为隔离时间律问题，新增纯离线探针
`todos/probes/rm65_qref_dynamics_probe.py`。它读取 parquet 中：

- `action.reference.*.q_ref`；
- `position.transport.*.last_accepted_qpos`；
- `collection.continuity_segment_index`。

探针先将 15 Hz materialized qref 插值为 120 Hz reference，再求“与该 reference
平方误差最小”的整段关节轨迹，同时硬约束：

- 项目 RM65 绝对关节限位；
- production position profile 的逐关节 velocity；
- production position profile 的逐关节 acceleration；
- `jerk_max = acceleration_max / 0.10 s`。

这是 future-aware 离线可行性 oracle，不是 runtime 实现。它不访问设备、不初始化
控制 authority、不 dispatch。必须按 continuity segment 分段；跨数秒 pause/gap
直接差分会把新 segment 的首帧误报成巨大 acceleration/jerk。

0727 全部 115 episodes、28,987 source frames 的结果：

- 原 120 Hz linear-qref 离散 acceleration 最坏为左 `15891.6°/s²`、右
  `15824.0°/s²`；
- 原离散 jerk 最坏为左 `1.907e6°/s³`、右 `1.899e6°/s³`；
- 约束投影后 acceleration 最坏为左 `3030.9°/s²`、右 `3002.2°/s²`；
- 约束投影后 jerk 最坏为左右约 `9.167e4°/s³`，等于所选逐关节硬上限中的最大值；
- 相对 linear qref 的全量最坏关节偏差为左 `0.743°`、右 `0.757°`；
- “每个 episode 内最大偏差”的均值为左 `0.048°`、右 `0.049°`，p95 为左
  `0.119°`、右 `0.119°`；
- 历史 transport-accepted qcmd 在 15 Hz 数据集抽样点相对 qref 的最坏偏差为左
  `5.60°`、右 `5.52°`。该值受抽样相位影响，不能当作 120 Hz 峰值证书，但说明
  新投影所需的 reference 让步并不大于现有执行链的历史跟踪偏差。

压力集 ep34/73/96 中，0.10 s jerk ramp 的最坏偏差仅左 `0.032°`、右
`0.190°`；右臂 acceleration 从 `7611.9→2248.1°/s²`，jerk 从
`9.13e5→9.17e4°/s³`。0.05 s ramp 偏差更小但 jerk 上限翻倍；0.20 s ramp
动态更平滑但右臂最坏偏差约 `0.349°`。因此 0.10 s 是当前合理的离线中点，不是
未经验证的 production 默认值。

这个结果证明“all-IK qref 不能直接作为 qcmd”不是阻断项，而是明确要求一个独立
时间律层。推荐 runtime 形态是 bounded-lookahead receding-horizon governor：

```text
selected source-q branch path
        ↓
120 Hz desired qref horizon
        ↓
以 last admitted qcmd / qdot / qddot 为初态
        ↓
最小 qref tracking error
subject to q / qdot / qddot / jerk
        ↓
只提交首个短 control cut
        ↓
连续碰撞/admission 后成为 qcmd
```

#### 因果两帧前视 governor

同一探针还实现了 source-rate receding-horizon 版本。每个 15 Hz source tick
重新规划一次，初态严格继承上一 control cut 的 q/qdot/qddot；规划窗口为两个
source step（约 133 ms），但每次只提交接下来的 8 个 120 Hz tick。QP 中的第一步
acceleration/jerk 也相对继承状态计算，不把内部求解迭代误当成已执行控制帧。
QP 核心已抽为
`robot_motion.position_control.plan_jerk_limited_joint_horizon`：它是 backend-neutral
纯候选生成器，不持有 clock、authority、dispatch 或 committed state。探针已改为直接
调用该共享实现，下面的全量数字因此不是一份另行复制的实验算法结果。

ep96 上比较 2/3/4 个 source step 后，3/4 步没有改善最终误差或动态峰值，只增加了
当前探针的求解时间，因此全量验证采用两步。0727 全部 115 episodes 的结果为：

- acceleration 最坏为左 `2862.5°/s²`、右 `2842.5°/s²`；
- jerk 最坏为左 `91581.6°/s³`、右 `91581.6°/s³`，均在所设
  `91673.2°/s³` 最大关节 jerk 上限内；
- 相对 linear qref 的逐 episode 最大关节误差：左 mean `0.217°`、p95
  `0.971°`、worst `1.295°`；右 mean `0.667°`、p95 `1.462°`、worst
  `1.817°`；
- 在 15 Hz source keyframe 处的最坏误差为左 `0.759°`、右 `1.403°`；
- 当前研究探针每个 episode 每侧的 wall-clock solve time 均值约
  `68–72 ms`，最大约 `117–141 ms`。探针每次重建 OSQP，且运行了统计与
  Python 数据装配，因此该数字不是 production solver latency；它只说明上线前必须
  缓存稀疏结构/因子或接入专用 OTG，不能直接复制探针热路径。

FK 与碰撞验证共检查 `229131` 个 120 Hz qcmd endpoint、`228736` 个相邻
qcmd interval。结果：

- endpoint 的关节限位/双臂 collision failure：`0`；
- 每个 interval 在 25%/50%/75% 三个内点采样的 collision failure：`0`；
- 左臂 FK Pose 误差逐 episode 最大值的 mean 为 `0.597 mm / 0.172°`，
  p95 为 `1.874 mm / 0.757°`，全量 worst 为 `3.213 mm / 1.069°`；
- 右臂对应 mean 为 `1.602 mm / 0.454°`，p95 为
  `3.251 mm / 1.014°`，全量 worst 为 `4.771 mm / 1.340°`。

最后一次使用共享 governor、linear reference、两 source-step lookahead 和
`0.10 s` jerk ramp 的全量重跑进一步得到：

- 左右 velocity/acceleration/jerk violation 均为 `0`；
- source all-IK failure、qcmd endpoint joint-limit/collision failure、25%/50%/75%
  interval sample failure 均为 `0`；
- 全量最坏 qref deviation 为左 `1.29535°`、右 `1.81729°`；
- 全量最坏 Pose error 为左 `3.194 mm / 1.0872°`、右
  `4.745 mm / 1.3371°`；
- 最小 sigma 为左 `0.0105198`、右 `0.0102805`。

此前 ep24/ep99 曾各出现一个 endpoint `joint_position_limit` 失败。详细诊断证明它们
不是碰撞，也不是实际越限路径，而是 q3 位于硬上限时 OSQP 归一化位置不等式留下的
微小正残差。共享 governor 已将位置约束改为 radian 尺度，并在未来节点保留
`1e-6 rad` 数值内缩；exact-limit 初态只允许向安全域内移动。修复后完整 115 episodes
重跑为上述 `0` endpoint failure，且单元测试额外断言整段 q path 均严格位于硬限位内。

#### 与 V4 的同集直接 A/B

为避免“V4 压力集”与“新链路全量最坏值”错位，另外固定同一组 16 episodes：

```text
0–7, 16, 35, 37, 59, 68, 80, 81, 96
```

V4 使用 `closed_loop`、60 Hz reference、`offline_fixed_solve_iterations=2`；
新链路使用 source all-IK、linear qref、两 source-step causal horizon 和
`0.10 s` jerk ramp。V4 共评估 `18,580` 个 reference。结果如下：

| 同集最坏指标 | V4 | official source all-IK + horizon governor |
|---|---:|---:|
| joint deviation | `32.354°` | `1.455°` |
| target translation error | `29.137 mm` | `3.180 mm` |
| target rotation error | `3.566°` | `1.337°` |
| velocity | `197.909°/s` | `106.717°/s` |
| acceleration | `6922.45°/s²` | `2862.54°/s²` |
| jerk | `443532.9°/s³` | `91581.6°/s³` |
| 双臂最小 sigma | `8.6958e-6` | `0.0102805` |
| velocity violation side-frame | `0` | `0` |
| acceleration violation side-frame | `1` | `0` |
| source all-IK failure | 不适用 | `0` |
| qcmd endpoint/采样内点失败 | 未作为该 evaluator 的同类指标输出 | `0 / 0` |

这不是仅在 episode 96 提高 sigma 的局部收益：在同一真实子集上，新链路同时改善了
最坏 Pose、joint deviation、velocity、acceleration、jerk 和双臂最小 sigma，并消除了
V4 的一个 acceleration violation side-frame。它因此是目前第一个在现有离线证据上
**整体优于 V4** 的 deploy 候选。

随后把 V4 压力集与新链路全量最坏值所在的 episodes 合并为 23 条逐 episode 对照。
`0.10 s` ramp 的改善/回退计数为：

| 指标 | 优于 V4 | 差于 V4 |
|---|---:|---:|
| joint deviation | `22/23` | `1/23` |
| translation | `20/23` | `3/23` |
| rotation | `15/23` | `8/23` |
| velocity peak | `12/23` | `11/23` |
| acceleration peak | `23/23` | `0/23` |
| jerk peak | `23/23` | `0/23` |
| sigma | `11/23` | `12/23` |

sigma 的逐 episode 回退不能简单解释为 selector 选错分支。ep37/80 的 recorded action
本身最低约为 `0.01053/0.01029`，V4 通过偏离 qref 得到了约 `0.0225/0.0277`；
官方全解在同一精确 Pose 下的最佳 sigma gain 只有约 `1e-6`，不存在一个“换 exact
candidate 就既复现 Pose 又得到 V4 sigma”的隐藏分支。新链路解决的是 ep96 式的
错误 closed-loop drift，不会凭空消除 source action 自身的奇异邻域。

#### 时间律参数扫描：最终离线候选采用 0.05 s

在固定 23 episodes 上只调整 `jerk_max = acceleration_max / ramp_sec`，不改变 source
all-IK、branch selector、qref 或碰撞逻辑：

| ramp | joint worst | translation worst | rotation worst | acceleration worst | jerk worst |
|---|---:|---:|---:|---:|---:|
| `0.10 s` | `1.817°` | `4.745 mm` | `1.337°` | `2862.5°/s²` | `91581.6°/s³` |
| `0.075 s` | `1.484°` | `3.821 mm` | `1.096°` | `3597.5°/s²` | `122108.8°/s³` |
| `0.05 s` | `1.113°` | `2.700 mm` | `0.830°` | `4586.3°/s²` | `183163.2°/s³` |

`0.05 s` 在这 23 条上相对 V4 达到：

- joint、translation、rotation、acceleration、jerk 均为 `23/23` 改善；
- velocity peak 为 `12/23` 改善、`11/23` 回退，但新链路同集全局 worst
  `109.3°/s` 仍低于 V4 的 `197.9°/s`，且每帧均通过逐关节硬限速；
- source、endpoint、采样 interval failure 均为 `0`。

因此 `0.05 s` 是当前更偏向跟手性的 Pareto 点；`0.10 s` 仍是更平滑但 Pose lag
更大的可回退研究参数。没有继续把 ramp 压到更小，因为 `0.05 s` 已经在固定集上消除
全部 Pose/joint 回退，继续放大 jerk 只会降低平滑裕度。

最后对 `0.05 s` 重跑完整 115 episodes：

- source failure、velocity/acceleration/jerk violation、endpoint
  joint-limit/collision failure、25%/50%/75% interval sample failure 全部为 `0`；
- 最坏 joint deviation：左 `0.7728°`、右 `1.1127°`；
- 最坏 Pose error：左 `1.947 mm / 0.6723°`、右
  `2.756 mm / 0.8302°`；
- 最坏 velocity：左 `93.38°/s`、右 `111.06°/s`；
- 最坏 acceleration：左 `4586.26°/s²`、右 `4575.32°/s²`；
- 最坏 jerk：左右约 `183163.2°/s³`；
- 最小 sigma：左 `0.0105279`、右 `0.0102804`。

所以当前离线最优链路的完整参数是：

```text
official source all-IK
+ collision-safe causal nearest-branch selector
+ linear source-q reference
+ two-source-step receding horizon
+ 120 Hz third-order governor
+ 0.05 s jerk ramp
```

其中候选配对逻辑已从 `todos/probes/` 抽为
`robot_motion.rm65.select_nearest_collision_safe_rm65_ik_pair`。它只比较已产生的
双臂候选并调用 canonical validation callback，不初始化 SDK、不持有 planner state、
clock、cursor、authority 或 dispatch；探针已经改为调用这份共享实现并在 ep96 复核
结果不变。

下面是完成 runtime 接入**之前**的权限判断。V4 evaluator 和新链路探针当时并非
同一份 runtime 实现；新链路还需完成连续 swept collision、runtime admitted-state
恢复、source backlog/epoch、SDK 隔离和 solver deadline 门禁。权限应表述为：

```text
接入前 production 默认：V4
当前离线整体最优候选：official source all-IK + causal jerk horizon governor
下一权限：default-off deploy shadow / no-dispatch 集成
```

这些数字证明：已有离线 qpos 足以建立一个动态可行的因果 qref→qcmd 方案，
而且不需要先接触真机。它同时说明不能把 qref 线性插值直接下发，也不能要求
qcmd 与每个 qref tick 完全相同；正确合同是保留 qref 作为动作语义，由独立时间律层
产生受 q/qdot/qddot/jerk 约束的 qcmd。

仍未完成的门禁：

- 25%/50%/75% 内点无碰撞不是严格的连续 segment collision certificate；
  production 仍需保守细分、swept-volume 或等价的连续证明；
- 因果探针从记录 segment 首 qref 以零速度/零加速度开始；接入 runtime 时必须从
  last admitted qcmd/qdot/qddot 和 epoch 一致的 source backlog 恢复；
- 若 policy chunk 不提供足够 lookahead，必须显式选择 backlog/time dilation、
  有界跟踪误差或 fail-closed，不能假装三者可以同时为零；
- 当前 `AccelerationLimitedJointTrajectoryPlanner` 没有 jerk 状态，也按活动 endpoint
  构造 braking envelope；它在 ep34/73/96 已表现出 18–55° 滞后和 collision reject，
  不能仅增加一个 jerk clamp 后复用为新 governor。

仍需完成的生产门禁包括：

- 60 Hz joint segment 的连续碰撞证书，而不只是 endpoint；
- selector 使用 admitted q/qdot 的动态滞回，并记录 candidate count/选择原因；
- source action 丢失、全解失败、分支消失和重建 epoch 的明确 fallback；
- 独立比较现有 planner 与真正 jerk-limited OTG，不能把依赖名称当作效果证明；
- bundled SDK 算法全局状态的线程/进程隔离；
- replay lookahead graph 与实时 causal selector 分开注册。

上述条目记录的是接入前门禁；后面的 2026-07-31 权限变更覆盖其中的“默认
deploy/default-off”结论，但不删除尚未完成的连续 collision 和真机验收边界。

## 4. 接入前结论与回退边界

```text
接入前 deploy / Pose replay 默认：V4
显式原子回退：V1
离线结构研究：V5、V6、V9
离线失败探针：V7、V8
离线局部控制探针：V10
离线时域 selector 原型：V11 H8（尚无合格多分支候选）
离线整体最优候选：official source all-IK + causal jerk horizon governor
严格 QPOS replay：直接回放 accepted q_ref，不经过 Pose Lift
```

2026-07-31 在用户明确授权后，新链路已接入同一
`PositionDeployExecutor` 并成为正式双臂 position deploy 默认后端：

```text
--position-path-lift-backend official_source_all_ik_causal_governor
--position-all-ik-lookahead-source-steps 2
--position-all-ik-jerk-ramp-sec 0.05
```

执行器现在以 120 Hz 产生 governor control cut；source all-IK 只在新的 15 Hz
Pose keyframe 出现时求解。固定 12 关节、16 transition QP 的稀疏矩阵和 OSQP
factorization 在 planner 构造期完成，运行期 source cut 只更新 reference/state
向量并 warm-start；第二个 same-chunk successor 只作为不推进 policy queue
的因果 horizon 输入，实际 source cursor/episode authority 没有复制。旧 V4 保留为：

```text
--position-path-lift-backend pink_pose_lift
--pose-lift-profile rm65_pose_lift_v4_qp_constrained
```

该权限提升不改变证据边界：完整 115 episodes 的离线结果仍是主要效果证据；当前代码
只完成无设备、mock 和 contract 验证，尚未声称完成新后端真机验收。每个 qcmd endpoint
和相邻区间 25%/50%/75% 内点会继续 fail-closed 检查，但 metadata 明确标为 sampled
interval，**不冒充连续 swept collision certificate**。solver deadline 也会记录
`solve_time_budget_exceeded`，首次真机前仍应先跑 dry-run/debug 并检查 p95/p99。

在本地 `dexteleop310`、无设备 mock collision/candidate 条件下，120 个连续 control
cut 的 source-boundary 重规划为 p50 `1.29 ms`、p95 `1.66 ms`、max `2.18 ms`；
非重规划 cut 的 p95 为 `0.08 ms`，没有超过 120 Hz 的 `8.33 ms` 预算。独立无设备
官方 SDK all-IK 100 次调用为 p50 `0.192 ms`、p95 `0.202 ms`、max `0.377 ms`。
这些数据证明固定 QP setup 不再进入热路径，但不替代带真实 collision scene、系统负载
和真机 SDK 环境下的端到端 p95/p99 验收。

```text
当前双臂 position deploy 默认：official source all-IK + causal governor 0.05 s
当前部署回退：Pink Pose Lift V4
Pink 原子回退：Pose Lift V1
live teleop 默认：POSITION_FORMAL_V1_ARGS（独立版本空间）
Pose replay 默认：仍为 Pose Lift V4
```

局部奇异风洞和 Pink task-conditioned directional demand 已落地。V10 已证明局部
stay-in-branch 有效，但真实轨迹 production gate 失败。V11 H8 也没有产生足够好的候选。
后续主线已经转为 source-rate official all-IK 的显式多解选择与独立时间律层；V10/V11
保留为失败证据和 fallback 研究，不再作为近期 production 候选。

## 5. 验证记录

- 本次部署接入相关回归：`151 passed`（共享 horizon governor、official all-IK adapter、
  deploy contracts/defaults、pose-lift profile/evaluator/sim replay）。
- `py_compile`：通过。
- `git diff --check`：通过。
- 本次文档补录重新执行了两个 evaluator 的 `--help`，确认本文使用的 profile、H8、
  task-space terminal、dynamic slack、wind-tunnel split 和 `--output-json` 参数仍存在。
- 所有实验 evaluator 均使用 `--no-motion`，未连接或驱动真实设备。

上述命令把完整 JSON 写入 `/tmp`，这些文件是可重新生成的临时产物，并未作为仓库内的长期
实验资产提交；本文保留的是决定权限所需的数值、参数和否决原因。若后续要用结果作为发布或
论文证据，应在固定源码快照下重新运行，并把原始 JSON、命令行和环境清单一并归档。
