# Pose Action 到 QPOS 的训测一致规划

## 状态与范围

本文件记录规划与分阶段实施状态。近期目标是在保持现有 pose action 主合同和
可回退性的前提下，明确训练标签、部署 Pose→QPOS 提升、奇异保护与实际 admission 之间
的边界。不得为了复现演示关节分支而绕过关节、速度、加速度、碰撞或反馈有效性等硬约束。

## 已确认的数据事实

- position 数采的模型 action pose 来自正式 accepted `q_ref` 的 FK，而不是未经 admission
  的原始 IK 候选。
- 数据集同时保存 `action.reference.<side>_arm.q_ref`、`q_ik_raw` 和 `pose_ref`，因此离线
  回放、审计和一致性评估必须直接使用保存的 `q_ref`，不得通过 pose 反解来猜原关节值。
- pose 到 qpos 并非一一映射：普通区域存在离散肩/肘/腕分支，腕部共轴奇异附近还存在
  连续或近连续的等价关节分配。因此 pose-only 策略不能保证唯一还原演示 `q_ref`。
- 训测一致需要拆成两层：模型 action 的 SE(3) 语义一致，以及部署端从 pose 到正式
  admitted qpos 的有状态、确定性、连续提升一致。

## 推荐的近期方案

保留 pose action 作为模型主输出，不把完整 qpos 直接替换为新的主 action。部署端引入
独立、可按 profile 回退的 Pose→QPOS 提升策略：

1. 始终以上一条正式 admitted/last-safe qpos 为主初值和连续性锚点，并对旋转关节做合法
   unwrap。
2. 将“粘滞”定义为软连续性正则或 trust region，不定义为跨阈值突然开关的冻结状态。
3. 连续性权重同时依赖归一化 Cartesian 残差和可信的奇异严重度：正常区不得附加可感知
   的跟手阻力；奇异且目标接近时增强连续性；残差持续增大且局部解停滞时平滑降低软权重。
4. 只削弱 Jacobian 的退化任务方向，尽量保留平移和其他可控旋转方向；不得用一个全局
   标量把完整 6D 目标统一粘住。
5. 大残差只触发有限多种子/多分支搜索，不直接解除安全连续性。只有候选显著改善目标
   残差、满足所有硬约束、存在有界连续路径，并在迟滞比较中稳定胜出时才允许换支。
6. 分支选择以一段可执行路径为单位锁定，禁止逐帧在多个候选间重新投票造成来回抖动。
7. 机器人侧 reference 只由正式 admitted qpos/FK 推进。保留 requested pose、admitted
   qpos、admitted pose 与 realized feedback 的区别，不静默声称被安全修改的目标已实现。

硬约束在任何模式下都不得弱化：关节绝对限位、每周期步长、速度、加速度、碰撞、有限值
检查和反馈有效性。可放宽的只有连续性、posture 偏好、退化任务方向精度、候选数量和搜索
半径等软项。

## 离线验证门槛

先在历史 position 轨迹上构建只读评估，再决定运行时改动：

1. 使用每个 episode 的真实起始 qpos 初始化部署提升器，逐帧输入记录的 pose action。
2. 与数据集保存的 accepted `q_ref` 比较，而不是与 `q_ik_raw` 比较。
3. 分开统计正常帧与 `pose_quality.score=-1` 的明显奇异帧。
4. 至少报告 FK 平移/旋转误差、unwrap 后关节偏差、肩肘腕分支变化、单周期 `dq`、速度、
   加速度、hold 次数、候选换支次数和从奇异区恢复时间。
5. 奇异腕部除了逐关节误差，还应按经 RM65 轴定义验证后的 `q4±q6` 不变量和 q5 分支
   比较，避免把等价腕部解错误记为完整失败。
6. 将“正常区跟手性”设为与“奇异区稳定性”同等权重的验收项：正常区不得出现额外
   local-step hold、目标残差累积、持续 step scaling 或可感知相位滞后。

## 何时扩展模型合同

如果排除 `score=-1` 的明显奇异帧后，部署提升器仍频繁无法复现演示分支，则 pose 信息
不足已经被实证确认。届时优先增加软分支/posture 辅助目标，而不是立刻把主 action 改成
完整 qpos：

- 肩、肘分支类别；
- 腕部 q5 分支/区域；
- 共轴腕部组合角或 q4/q6 分配提示；
- 必要时才增加完整 qpos 辅助 head。

辅助输出只作为 IK 候选打分偏好，不直接绕过 admission。若业务要求逐关节精确复现演示
qpos，则 pose-only 合同从信息论上无法满足，必须显式携带完整或部分关节分支信息。

## 实施顺序与回退

1. 建立离线 Pose→QPOS parity evaluator，不连接真机、不下发运动。
2. 固化现有部署行为为 baseline profile 和回归样本。
3. 在新 profile 中实现连续自适应正则和正常区旁路，默认保留旧 profile 可一键回退。
4. 增加“持续残差 + 局部停滞 + 目标时间一致”触发的有限候选路径搜索。
5. 用历史轨迹比较 baseline/new profile，并设置正常区跟手性不能回退的验收门槛。
6. 只有离线结果通过后，才制定受控真机 shadow/no-motion 与低风险验证步骤。
7. 若正常帧仍存在显著分支漂移，再单独规划辅助分支模型合同；不得与本次 solver 调参
   混在一个不可回退变更中。

## 2026-07-28 Phase 0/1 落地状态

- replay 已显式拆成 `pose` 和 `qpos` 两种模式。两者首帧都使用保存的 accepted q_ref
  做 QPOS init；QPOS replay 后续严格复现 q_ref，不调用 IK、不修改目标，但保留硬限位、
  reference 速度/步长和双臂 endpoint collision 准入。
- 当前 deploy 行为保存为默认 `rm65_pose_lift_v1_baseline`；新增 opt-in
  `rm65_pose_lift_v2_continuous`。baseline 默认不改变 solver 候选；V2 当前仅做经 pose
  tolerance/绝对限位认证的合法 2π unwrap 与腕部共轴等价解规范化。认证失败原子回退 raw
  solver candidate，未实现 Phase 2 differential IK 或 Phase 3 多路径搜索。
- Pose replay 与 deploy 都将同一 Pose Lift profile 传入共享
  `RM65PositionCollisionPlanner`，并使用同名 objective identity。严格 QPOS replay 不受
  Pose Lift profile 影响。
- 已增加只读 `teleop_runtime.debug.evaluate_rm65_pose_lift_parity`，提供
  teacher-forced、closed-loop、15→60 Hz runtime-causal 三种评估；首帧以 q_ref 初始化，
  真值只使用保存的 accepted q_ref，不使用 q_ik_raw。
- 后续 Phase 2/3 只能继续新增显式 profile，不得静默改变 V1 baseline 或当前 V2 语义。

## 当前待回答的问题

- 当前遥操“粘滞”是否只由可信奇异状态触发，还是 local-step-first、joint posture、输出
  smoothing、step limiter 等普通路径也在形成相同体感。
- 腕、肘、肩奇异阈值与 release hysteresis 是否覆盖了正常工作区域。
- 正常区是否仍优先选择 local step、是否周期性跳过 full solve，以及失败时是否直接
  hold last qref。
- 输出层的奇异专用 smoothing 是否严格限制在窄奇异带内，并能在退出后立即释放。
- `score=-1` 的记录判定与运行时保护判定是否使用同一事实来源但保持不同职责，避免标签
  阈值反向扩大控制保护范围。

## 2026-07-27 历史轨迹只读审计

本节只记录诊断结论；尚未修改运行时参数或控制语义。审计覆盖 6 次具有有效 IK shadow 的
历史运行：`20260726_135629_225603`、`20260726_161727_980633`、
`20260726_171912_485544`、`20260726_190731_287140`、
`20260726_214120_469078`、`20260726_231517_858602`。最近的
`20260726_233756_353471` 没有有效记录，未纳入统计。

为避免把奇异保护误算成普通区粘滞，审计中的“普通帧”采用保守定义：双臂 q3/q5 在
raw、canonical、admitted 三阶段均离奇异阈值超过 6 度，且腕部 null scale 为 1。即使
如此，各有效运行中仍有大量普通帧，且其响应规律跨版本一致：

- 最新有效运行共有 11,390 帧，普通帧 8,522 帧，占 74.8%。
- 普通帧 `raw IK -> canonical` 最大关节差中位数为 0 度、p95 为 0.257 度；候选选择或
  腕部规范化不是此运行中广泛粘滞的首要来源。
- 普通帧 `canonical -> admitted qref` 最大关节差中位数为 1.913 度、p95 为
  2.467 度；99.7% 普通帧超过 0.1 度，97.9% 超过 0.5 度。
- 对存在明确候选运动的普通帧，`admitted step / canonical step` 的 p10、p50、p90
  分别约为 0.166、0.347、0.448，且全部小于 0.5。这与当前 adaptive post-IK qpos
  smoothing 的 0.14 到 0.45 alpha 精确对应。
- 6 次历史有效运行都出现了同样的普通区响应上限，而不仅是最新奇异逃逸实现之后才出现。
  因此它是正式 position source shaping 的长期行为，不应归因于最近新增的奇异搜索。
- 最新运行的腕部 null attenuation 只覆盖约 4.25% 帧；历史运行虽因姿态不同在 0% 到
  54.87% 间变化，但仍无法解释最新运行 74.8% 普通帧上的普遍衰减。
- 下游 `admitted qref -> qcmd` 在最新普通帧中的最大关节滞后中位数约 0.176 度、p95
  约 0.938 度，存在附加相位滞后，但显著小于上游 `canonical -> admitted` 的衰减。

### 已定位的普通区 shaping 层

1. position 遥操始终启用 target pose smoothing。默认时间常数为 0.025 秒，在 60 Hz
   下单周期 alpha 约 0.487；该层发生在 shadow 中记录的 `target_world_pose` 之前，现有
   历史记录无法单独量化 raw tracker target 到 smoothed target 的损失。
2. post-IK qpos smoothing 对 live human position teleop 的全部六个关节始终生效。正式
   profile 的 slow/fast alpha 为 0.14/0.45，速度自适应只在两者之间插值，不会在健康
   普通区趋近 1 或旁路。这是当前广泛非奇异区粘滞最直接且被轨迹数值验证的来源。
3. 60 Hz qref 还有正式 joint-velocity step limit；随后 120 Hz execution planner 继续
   执行 command-step、速度和加速度限制。它们需要分别计量，不应与奇异连续性正则混为
   一个“粘滞”旋钮。

### 已定位的局部奇异保护过宽点

q5 hysteresis 当前并未保存“确实完成过一次奇异逃逸”的 latch。实现会把 q5 在
3.25 到 6 度区间内的任意同侧向内运动保持在上一值；只有目标残差达到 0.10 m 或 20 度
时才放松。它因此可能冻结形式上已经位于 SDK 3 度奇异门外的细小正常腕部指令。最新运行
该区只占约 0.69% side-frame，但历史姿态中曾达到约 10.78% 和 20.79%；这是应单独修正的
窄区语义问题，而不是解释所有普通区不跟手的主因。

### 修改前必须补齐的观测与验收

后续任何调参或新 profile 都必须把以下链路分段记录并分别验收：

1. raw tracker target 到 smoothed pose target 的延迟、deadband 与响应比例；
2. smoothed target 到 raw IK candidate；
3. raw candidate 到 canonical/local-step candidate；
4. canonical candidate 到 admitted qref；
5. admitted qref 到 qcmd；
6. qcmd 到 measured qpos/末端 realized pose。

统计必须同时按正常区、GUARDED/DEGRADED/RECOVERY、关节/碰撞限幅和单臂 fix/双臂模式
分桶。正常区验收至少包含 10/50/90 百分位 step response、Cartesian 跟踪误差、端到端
group delay、目标残差是否积累；安全验收继续包含最大 dq/qdot/qddot、branch flip、hold、
奇异逃逸成功率和碰撞约束。不能用奇异区稳定性改善掩盖正常区跟手性退化。

### 后续变更边界

- 第一优先级不是继续放宽奇异判定，而是为健康普通区设计明确 fast path，使 post-IK
  qpos smoothing 能平滑趋近 1 或安全旁路；硬 qref 速度、加速度、关节和碰撞限制保持。
- target pose smoothing 需要在补齐 raw/smoothed 观测后再决定是否缩短时间常数或仅保留
  静止噪声 deadband，不能与 qpos smoothing 同时盲调。
- q5 hysteresis 应改为由“实际 admitted 的逃逸已发生”显式置位的状态，而不是仅凭当前
  q5 落在 3.25 到 6 度区间触发；其修改与普通区 fast path 分开提交、分开回退。
- 12 度 wrist branch guard 主要阻止离散翻腕候选，最新普通帧中不是广泛衰减首因，不作为
  第一调参对象。
- 改变 source shaping 会改变未来数据中由 accepted qref FK 得到的 action 分布。现有稳定
  profile 必须保留；行为变更应使用新版本 profile，并在数据元信息中可区分，不能静默
  覆盖当前稳定版本。

## 2026-07-27 响应性 V2 实施状态

已新增 opt-in `rm65_bimanual_position_v2_responsive`，V1 保持原行为和原 source metadata
字段集合。V2 当前只改变 live-human source shaping：

- 单侧状态为 `tracking`、tracker 运动证据有效、前后 q3/q5 均在 4 度 guard 外、腕部
  null projection 未介入且 q5 escape latch 未置位时，post-IK alpha 使用 0.70；tracker
  运动质量允许 0.05 秒软宽限并同步实际 alpha 状态。关节、null projection、q5 latch、
  admission 等硬门仍立即生效，硬 qref step、速度、加速度、碰撞和后续 120 Hz planner
  不变。
- q5 release hysteresis 在 V2 中只有当显式 escape 修正确实进入最终 admitted qref 后才
  置位；候选探测被 collision/jump/final admission 拒绝时不会置位。V1 继续使用原固定
  release-band 行为。
- V2 通过 `POSITION_FORMAL_V2_RESPONSIVE_ARGS` 选择，切回
  `POSITION_FORMAL_V1_ARGS` 即原子回退。V1/V2 必须使用不同 collection task/profile
  名称，避免两种 source action 分布混写。

本阶段未修改 target pose smoother、joint velocity profile、command acceleration planner、
collision admission 或数据 action 合同。后续仍需用新增运行记录分层确认 raw target 到
realized motion 的端到端响应，不得仅凭 IK shadow 中间量宣布真机跟手性验收完成。

## 2026-07-27 训测一致重新规划

本节取代前文“推荐的近期方案”和“实施顺序与回退”作为后续执行依据；前文保留为决策
历史，不再据其直接修改运行时。

### 重新规划结论

不再把“部署端使用一套较弱粘滞”作为首选实现。粘滞会主动制造 pose residual，而它既不
保证选回演示分支，也会把控制稳定性与模型 action 语义耦合。新的主方案是：

> **以最后正式 admitted qpos 为状态锚点，先做保持 FK 的等价解规范化，再做连续的局部
> Pose Lift；只有局部解被证实停滞时才启动有限路径搜索。**

遥操 V2 source profile 与部署 Pose Lift profile 必须是两个独立合同：前者定义人类输入
如何生成数据 action，后者定义已存在的 pose action 如何被提升为可执行 qpos。部署不得
重放 tracker deadband、tracker motion-window、post-IK source smoothing 或遥操 escape
手势语义。

当前代码事实也要求二者分离：部署端虽然接受 `--position-control-profile`，但以
`include_source_shaping=False` 解析；V1/V2 目前共享相同的 `ik_execution`，所以仅把部署
命令从 V1 改成 V2 不会改变 Pose→QPOS 提升结果，也不能构成训测一致修复。

### 一致性分成三个层次

1. **Action 语义一致**：模型输出的 pose 与数据集中由 accepted `q_ref` FK 得到的
   `pose_ref` 使用相同 TCP、坐标系、四元数约定和时序定义。
2. **Lift 路径一致**：在 pose 足以区分关节分支时，部署提升器应从上一条正式 admitted
   qpos 连续地还原与演示相同的肩、肘、腕分支，而不是每帧独立选解。
3. **执行一致**：requested pose、lifted qpos、admitted qpos、admitted FK pose、qcmd 和
   measured state 必须分别记录。安全修正后的 admitted pose 可以偏离 requested pose，
   但不得把偏差静默隐藏成“策略已实现目标”。

在精确或近精确共轴腕部，pose-only 无法唯一确定 q4/q6 分配。此时验收首先比较 FK pose、
q5 分支和经 RM65 轴约定验证的 q4±q6 等价类；不能把同一末端姿态的另一等价关节分配误判
为 action 语义不一致。若下游动力学确实要求逐关节复现原演示，则必须增加分支提示，不能
继续声称 pose-only 可以精确恢复完整 qpos。

### Phase 0：建立只读 parity evaluator

先建立离线 evaluator。`dataset_interface/` 只负责数据读取、投影和可选质量分桶；调用
`robot_motion` Pose Lift 的编排放在 `teleop_runtime/debug/` 或测试/probe 层，避免
`dataset_interface` 反向拥有控制规划。evaluator 不连接设备，不拥有 dispatch、episode
或 canonical safety。使用今天不带 `old` 的 V2 数据为主集合，再用历史有效记录做回归
集合。每个 episode 同时运行三种模式：

1. **Teacher-forced one-step**：第 k 帧以前一帧保存的 `q_ref` 为锚点，输入第 k 帧
   `pose_ref`。它隔离“单步 inverse map 是否能选回演示分支”。
2. **Closed-loop lift**：只在首帧使用真实 qpos，之后以前一帧 evaluator 自己 admitted 的
   qpos 为锚点。它暴露分支漂移、误差积累和不可恢复 hold。
3. **Runtime-causal**：按部署实际 15→60 Hz 插值、当前 TCP/frame、step/velocity limits 和
   collision planner 运行。它衡量真实部署拓扑，而不是把 15 Hz 表格直接逐行反解。

三种模式都必须先跑当前 `RM65PositionCollisionPlanner`，固化为
`rm65_pose_lift_v1_baseline`。报告至少包含：

实施补充（2026-07-28）：evaluator 已补齐归一化 Jacobian `sigma_min`/condition number、
q3/q5 结构性裕度、奇异方向变化、腕部 branch/null redistribution 与 qdot/qddot/jerk。
0727 的代表 episode 证明“关节/FK 误差更小”与“奇异裕度更大”不是同一结论：当前 V3 在
80/81 的右臂最小 `sigma_min` 略低于 baseline。因此 V3 继续保持 opt-in，Phase 1 的验收
必须同时满足 parity、分支连续性和奇异裕度，不能只优化关节误差。

- requested pose 到 lifted/admitted FK 的平移和旋转残差；
- unwrap 后逐关节误差，以及肩、肘、q5 分支 mismatch；
- 共轴腕部的 q4±q6 等价类误差；
- 每周期 dq、qdot、qddot、branch switch、backoff、hold 和连续停滞长度；
- teacher-forced 与 closed-loop 的误差差值，用于区分单步不可辨识和累计漂移；
- 按正常帧、`pose_quality.score=-1`、奇异前后窗口、joint limited、collision limited 分桶。

不能用 `q_ik_raw` 作为真值；它只是候选。比较真值始终是保存的 accepted `q_ref` 及其
`pose_ref`。首帧没有可靠锚点、帧/TCP/profile fingerprint 不匹配的 episode 必须单独标记
为不可评估，不能混入失败率。

### Phase 1：无 pose 误差的连续规范化

在 `robot_motion/rm65/` 增加独立且可测试的 Pose Lift 策略，当前 planner 保持不变作为
回退。第一阶段只做不应改变目标 pose 的处理：

1. 所有关节候选相对上一条正式 admitted qpos 做合法 unwrap，并在绝对限位内选择最近
   等价表示。
2. 球腕进入共轴带时，用已验证的 q4±q6 组合量规范化等价解；q4/q6 分配以 admitted
   anchor 为主，关节限位余量为次级代价。
3. 对求解器产生的多个合法离散分支先做 endpoint pose tolerance 和硬约束过滤，再按
   “分支连续、关节路径短、限位余量”排序。连续性只是同等 pose 质量候选之间的
   tie-breaker，不允许用它接受更差的 Cartesian 解。
4. 分支选择使用上一条正式 admitted qpos，而不是 measured 抖动、未经 admission 的 raw
   IK 或下一帧预测值。测量 qpos 只负责首次 reset、fault/rebase 和 realized difference。

如果这一阶段已经使 teacher-forced 正常帧稳定复现演示分支，就不应再引入任何“弱粘滞”。

### Phase 2：连续局部 Pose Lift

对没有等价闭式规范化结果的普通帧，从 admitted anchor 做 bounded differential IK/QP：

- Cartesian residual 是主目标，平移与旋转分别加权；
- joint continuity、posture 和 joint centering 是软次目标；
- 每周期 step、速度、加速度、绝对限位、collision 和有限值检查仍是硬约束；
- 自适应 DLS 只衰减 Jacobian 的低奇异值方向，不给整个 6D action 乘统一粘滞系数；
- 正常区不启用奇异 escape cost，也不增加 source smoothing；
- reference 仅由最终 admitted qpos 的 FK 推进，拒绝的 raw candidate 不改变 anchor。

连续性权重只能随“可信奇异严重度”平滑增加，并随 Cartesian residual 增大而平滑降低；
不得使用一个 residual 阈值在 sticky/free 两个状态间跳变。局部步骤若持续改善 pose residual，
即使尚未到达目标也继续执行，不应因为一次未收敛就 hold。

### Phase 3：有证据的有限路径搜索

只有同时满足以下条件才启动较宽搜索：目标在一个短时间窗口内保持一致、局部求解连续
多帧未改善、当前确有可信奇异/分支阻塞证据，且目标 residual 显著。搜索应：

1. 生成有限个肩/肘/腕种子和共轴腕部规范化种子；
2. 对每个候选验证一段受 step/velocity/acceleration/collision 约束的有界路径，而不只看
   终点 IK；
3. 只有候选在 pose residual 上显著优于当前路径并通过迟滞比较时才换支；
4. 一旦选中，以 plan id 锁定该路径，除非失效或出现明显更优且稳定的替代路径，禁止
   每帧重新投票造成左右犹豫；
5. 搜索失败时保持最后安全状态并报告明确 failure reason，不积累待追赶的隐藏 pose 误差。

Phase 3 不是正常执行必经路径，也不能通过扩大搜索半径绕过硬安全约束。

### Phase 4：证明 pose-only 的信息边界后再扩展模型

若 Phase 1/2 在 teacher-forced 正常帧上通过，而 `score=-1` 或特定分支帧仍无法复现演示
qpos，应先把它判定为信息不足，而不是继续调 IK。按改动从小到大选择：

1. 训练/评估视图显式 mask `score=-1` 帧或包含该帧的动作窗口，让基础 pose-only 策略
   避免学习不可唯一提升的 action；原始数据不删除。
2. 增加离散 shoulder/elbow/wrist/q5 branch hint，作为 IK 候选排序提示。
3. 增加共轴腕部组合角或局部 qpos delta 辅助 head。
4. 只有业务明确要求逐关节精确复现时，才增加完整 qpos 辅助输出。

辅助输出不得直接下发，仍需经过同一个 Pose Lift、硬约束、collision 和 admission。任何
新增模型字段都需要新的 dataset/artifact/deploy contract 版本；在离线证据证明必要之前，
不修改当前主 action 维度。

### Profile、所有权与回退

- `rm65_pose_lift_v1_baseline`：封装当前部署 planner 行为，保持默认和回退入口。
- `rm65_pose_lift_v2_continuous`：新策略，先只允许 offline/shadow，验收后才 opt-in 真机。
- Pose Lift profile 需要独立 fingerprint，至少绑定 solver 版本、TCP/frame、joint-limit
  profile、collision scene、分支规范化版本和 hard-bound 版本；不要借用 teleop source
  profile 名称表达这些语义。
- 当前 `_position_deploy_ik_tuning()` 将 objective mode 名称和 position weight 绑定到 V1；
  在引入独立 Pose Lift profile 前必须解除这一硬编码，并增加“解析后的 profile 与实际
  planner tuning fingerprint 相同”的合同测试。当前 V1/V2 参数相同不代表该写法可继续
  扩展。
- `robot_motion/` 拥有可复用 Pose Lift、状态锚点、IK 和安全规划语义；
  `teleop_runtime/position_control/` 只绑定 real deploy executor；`dataset_interface/` 只
  提供只读数据投影/分桶，offline evaluator 编排属于 debug/verification；只有新增模型
  字段时才修改 `dataset_contracts/` 和 artifact 合同。
- shadow evaluator/planner 只能被动观察，不得推进 authoritative planner 的 cursor、
  last-safe state、reference 或 dispatch。

### 分阶段验收门

1. **Phase 0 完整性**：相同 episode/profile 重跑结果确定性一致；缺失 TCP/frame/anchor 的
   数据明确拒绝评估；baseline 报告可复现。
2. **正常区**：新 profile 的 pose residual、hold、branch switch 和 closed-loop drift 均
   不得差于 baseline；目标门槛使用 planner 已有 2 mm/0.5 度 tolerance，并在 baseline
   报告后冻结 percentile 和 mismatch 阈值。
3. **奇异区**：比较 task-space residual、q5 分支、q4±q6 等价类、路径连续性和恢复时间；
   不以逐关节完全相等作为 pose-only 的虚假硬门槛。
4. **安全**：零 NaN/Inf、零绝对限位突破、零 step/velocity/acceleration 违规、零未经
   endpoint collision certification 的 admission；失败必须 hold last-safe。
5. **回退**：同一输入显式选择 V1 时，结果和 baseline fixture/hash 一致；V2 状态不得污染
   V1 reset 或下一 execution epoch。
6. **上线顺序**：offline replay → passive shadow/no-motion → 有界低速空载 → 正式部署。
   每一步必须保存 profile fingerprint 和差异报告，上一门未通过不得进入下一步。

### 下一步唯一优先事项

先实现 Phase 0 evaluator 和 baseline fixture，不改部署运行时。它会直接回答三个决定后续
设计的问题：正常区当前 solver 能否从前一 accepted qref 选回演示分支、闭环误差从哪里
开始累积、`score=-1` 中有多少失败属于等价腕部解而非真正 pose 失配。在得到这三项证据
前，不继续调“粘滞强度”，也不扩展模型 action 合同。

## 2026-07-28 Pose Lift 多目标门禁与负结果

Phase 0 evaluator 已落地，且当前结论已超过上节“先实现 evaluator”的状态：

- 增加 V1 对照门禁，同时检查 joint/FK parity、Hold、branch jump、qdot/qddot/jerk、
  归一化实际 Jacobian `sigma_min`/condition number；只在相同 episode、相同模式、相同
  TCP/profile 下比较。
- `--include-frames` 现可按帧输出 truth/raw/admitted qpos、两侧 pose residual、实际
  Jacobian 指标和动态量，用于定位路径分歧，默认聚合报告仍保持有界。
- 0727 episode 16/35/80/81 的 closed-loop 与 runtime-causal 结果证明 V3 的主要行为差异
  不是腕部规范化：35/80/81 的 `pose_lift_applied_count=0`，轨迹仍变化，根因是 V3 将
  Pink posture cost 从 baseline `0.1` 改成 `0.0`。
- posture cost `0/0.025/0.05/0.075/0.1` 扫值不存在跨 episode 支配解。降低 cost 通常改善
  pose/joint parity，但在 35/80 等轨迹降低右臂最差 `sigma_min` 或提高动态峰值；不能选一个
  折中常数宣称问题解决。
- “恢复 baseline posture cost + 保留 V3 有界腕部规范化”只在 episode 16 产生数值级变化，
  其余代表 episode 与 V1 完全一致且无实质收益；该实验 profile 已撤销，未暴露为 deploy
  选项。当前可选 profile 仍仅 V1/V2/V3。
- 两种已否决的奇异约束实验也不得恢复：最终 joint-limit/collision 之前的 post-IK
  certification 会被后续逐关节限幅破坏；q3/q5 结构裕度不等价于实际 Jacobian 退化方向。
  episode 81 已实际出现“结构裕度受控但 `sigma_min` 仍降到约 0.01”的反例。

evaluator 中另有 opt-in、离线专用的双 posture 候选探针：每帧从同一 admitted anchor 分别
完成 baseline/alternate IK、硬关节与碰撞认证，再以实际 Jacobian、pose 和动态量做 Pareto
选择；第三条纯 V1 shadow 用于防止本地 baseline 随已选路径漂移。它不进入 deploy，不拥有
dispatch。代表结果：

- episode 81 runtime-causal：最差右臂 `sigma_min` 约从 `0.013398` 提高到 `0.013576`，
  最大速度约从 `168.5` 降到 `135.1 deg/s`，joint mean 也改善；说明额外候选确实包含有
  价值的逃逸路径。
- episode 16：pose/joint 与速度改善，但最大加速度约从 `7659` 上升到 `7801 deg/s²`，
  jerk 约从 `460473` 上升到 `468452 deg/s³`。
- episode 35：局部每步 sigma 不劣仍不足以阻止闭环累积，最差右臂 `sigma_min` 约从
  `0.019068` 降到 `0.019044`。
- episode 80：sigma 和 pose 有改善，但 joint-error p95 从约 `7.407` 增至 `7.456 deg`。

因此单帧 Pareto 重新投票不满足上线条件。下一步唯一允许的控制算法原型是 Phase 3 的
有限时域路径评价：对有限候选各自向前滚动若干参考步，整段验证 pose residual、实际
Jacobian 最小裕度、qdot/qddot/jerk、关节和碰撞；选中后绑定 plan id 并保持迟滞，直到路径
失效或出现经过整段证明的显著更优路径。该原型仍须先留在 offline evaluator；上述多目标
门禁全部通过前，不增加 production Pose Lift profile，不改变 deploy 默认 V1。

### 2026-07-28 有限时域验证结果

有限时域原型已仅在 offline evaluator 中实现并验证，没有进入 deploy/teleop：

- 初版 H8/L4 锁定的是 posture 求解器标签，而不是已经完成硬约束和 collision certification
  的 qpos 路径。它在后续锁期重新解 IK，因此 episode 35/81 仍出现 Jacobian 裕度回退；该
  语义已否决。
- 即使缓存 qpos 路径，默认 wall-clock adaptive iteration 会使额外 observer workload 改变
  IK 迭代数，导致“baseline selected”也不再等于独立 V1。有限时域模式现在强制显式
  `--offline-fixed-solve-iterations N`，并经 episode 80 的 V1 对 V1、closed-loop 与
  runtime-causal 共 36 项门禁证明确定性一致。这个固定迭代选项只用于离线算法比较，不能
  被表述成当前 deploy parity。
- 可执行 alternate 必须锁定并执行完整 horizon，终点回接独立 V1 shadow 的关节邻域，
  整段最小实际 Jacobian 裕度不退化，pose 改善达到显著阈值，窗口动态均值和累计峰值均不
  高于 shadow；未选择 alternate 时直接沿用主 V1 planner 实例，不缓存、不替换、不 reset
  baseline 输出。
- 固定 4 次迭代、H8/L8、终点离 shadow 不超过 `0.5 deg`、归一化 pose score 至少改善
  `0.05` 的最终探针，在 0727 episode 16/35/80/81 的 closed-loop 与 runtime-causal
  全部通过非回退门禁，但八个 episode×mode 组合的 alternate 选择次数均为 **0**。通过原因
  是严格保持 baseline，不是候选产生了改善。
- 放宽显著改善阈值时，episode 81 曾仅选择一段 8-step alternate：右臂最小 `sigma_min`
  从约 `0.0121359` 微升到 `0.0121406`，但平均 joint parity 回退约 `0.000113 deg`；收益
  属数值级且未通过严格门禁，因此没有理由放宽验收标准。

由此确认：`posture_cost=0.0` 与 baseline `0.1` 这两个候选不足以生成“既改善 pose/parity、
又提高实际奇异裕度、还可平滑回接”的路径，继续调 horizon、lock 或 Pareto 阈值没有价值。
下一步若继续 Phase 3，应更换**候选生成机制**，优先实现有方向的 bounded differential IK
或明确 shoulder/elbow/wrist 分支种子，并要求每条候选自带完整受限路径和 terminal rejoin；
不要再把 posture cost 扫值包装成奇异逃逸。新候选仍先走 fixed-iteration offline replay，
随后才讨论 cloneable planner state、确定性算力预算和 passive shadow。production profile
仍仅保留当前 V1/V2/V3，默认 V1 不变。

## 2026-07-28 约束 IK 部署探针 V4

已增加可原子回退的 `rm65_pose_lift_v4_qp_constrained`，但其验证范围明确为 offline、Pose
replay dry-run 和 deploy dry-run，`--send-actions` 会被硬门禁拒绝。V1 默认、V2/V3 语义与既有
fingerprint 保持不变。V4 当前只做一件事：把正式 qref 逐关节速度/加速度边界放进 Pink QP；同一
60 Hz reference 内的 2–4 次 Pink 调用共享相对上一条 admitted qref 的同一运动区间，不能
把数值迭代误当成多个物理控制周期。QP 后的绝对关节、步长和 collision admission 仍作为
独立证书保留。

速度/加速度约束都以 admitted qref/velocity 为锚，并包含接近绝对关节限位时的制动距离；
现有 120 Hz qcmd planner 的速度/加速度安全合同保持独立，V4 不替代下发层安全。

本轮还修复了 evaluator 的 reference-clock 合同错误：旧 teacher-forced/closed-loop 直接
按保存的 15 Hz 帧调用 60 Hz planner，使 time-aware QP 在 66.7 ms 的帧间隔里只获得一次
16.7 ms 运动预算，虚构出厘米级滞后。现在三种模式都先重采样到 deploy reference clock，
差异只来自 planner state 合同。修复后重新启用 admitted-anchor acceleration 验证，0727
episode 16/35/80/81 的 V4 对 V3 比较中，所有最小 `sigma_min` 均不下降；runtime-causal
最大 qref 加速度分别从约 `7991/4152/5911/4606` 降到 `6875/2698/4125/2729 deg/s²`，
最大 jerk 也全部下降。16/35/81 的 joint mean 改善，80 仅回退约 `0.000043 deg`。16/81
的最大速度高于 V3 实际轨迹峰值，但仍低于发生该峰值关节的正式硬速度上限；现有“不得
高于 baseline 实际峰值”门禁因此拒绝它，不能把该拒绝误写成越界。V4 已证明是更合理的
下层约束基线，但四条轨迹仍不足以升级为 hardware production profile。

## 2026-07-29 V4 默认部署与遥操 opt-in

经后续人工确认，V4 已提升为 deploy 与 Pose replay 的默认 Pose Lift，并允许通过既有
preflight、qpos-init、collision admission 和人工 ready gate 后执行真机输出。冻结的
`rm65_pose_lift_v1_baseline` 保持不变，显式选择 V1 即可原子回退；严格 QPOS replay 仍不经过
Pose Lift。

遥操没有静默切换到 V4。新增 default-off 的
`--position-ik-admitted-motion-limit`，只允许与
`single_step_singularity_aware` 同时使用；它把正式 qref 速度/加速度区间放进 Pink QP，
并继续保留 QP 后的候选、碰撞、关节和 120 Hz qcmd 安全层。测试数组
`POSITION_FORMAL_V3_QP_CONSTRAINED_ARGS` 基于稳定 V1，仅增加该开关；恢复
`POSITION_FORMAL_V1_ARGS` 即完整回退。

## 2026-07-29 V4 后续平滑实验与约束证书

在 V4 上继续验证了两条软连续性路径，均未进入 production profile：

- 关节 qdot 连续性可明显降低部分轨迹的 acceleration/jerk，但会延续上一帧朝奇异区运动
  的关节速度。固定 4 次 IK 迭代时，0727 episode 16/35/80/81 的右臂最小实际
  `sigma_min` 均下降，因此否决。
- 改为 Cartesian qdot 连续性并在 guarded band 内对“预测继续降低 sigma”的上一帧速度
  禁用后，episode 16 的最小 sigma 不再回退，但 episode 81 仍从约 `0.01135` 降到
  `0.01115`，且最大 jerk 上升；说明只平滑末端速度仍可能诱导更差的内部关节姿态，因此
  同样否决。

上述两种“追随上一帧速度”的实验代码及其临时 V5 选项均已撤回；deploy/Pose replay 默认
仍为 V4，遥操仍为显式 opt-in，真机命令不变。下一种候选必须把“方向选择性奇异退化/
显式 manipulability 或分支路径”直接放入同一个受限优化问题，不能再单独增加速度追随项。

离线 parity evaluator 现增加逐帧 `motion_constraint_certificate`，同时统计 QP raw candidate
与最终 admitted qref 的速度/加速度利用率、越界 side-frame 数，以及 QP 后腕部规范化、硬
关节证书和 collision backoff 的最大关节改变量。固定 4 次迭代复核 0727 episode
16/35/80/81：raw 与 admitted 的速度/加速度越界计数全部为 0；QP 后最大改变量只在 episode
16 非零，最大约 `0.000445 deg`，其余 episode 为零或数值噪声。这排除了“QP 后处理重新破坏
运动约束”作为当前主要问题，并把该证书加入 profile 非回退比较。

## 2026-07-29 V5 方向性 sigma escape（离线候选）

新增独立 profile `rm65_pose_lift_v5_sigma_escape`，但明确设置
`hardware_dispatch_allowed=false`，因此不能用于真机 action dispatch；V4 默认、遥操默认和
已有真机命令均未改变。V5 在 V4 的 qpos/qdot/qddot/collision 硬约束内增加一个低权重软任务：

- 使用 `LOCAL_WORLD_ALIGNED` geometric Jacobian，线速度行按 `0.5 m` characteristic length
  归一化；
- 只在 `sigma_min < 0.04` 时启用；
- 每个 60 Hz admitted anchor 只计算一次 active-chain central finite-difference 梯度，Pink
  的 2--4 次数值迭代共享同一逃逸方向，避免每次迭代重新选方向；
- 目标只是在本周期沿梯度把 sigma 提高最多 `0.03/s`，软任务 cost 为 `0.15`；所有硬运动
  和碰撞约束拥有更高权威。

同时保留原 `compare_profile_reports` 生产非回退门禁不变，另增
`sigma_escape_research` 离线 Pareto 验收。后者只有在 guarded side 的 sigma 有实质提升、
速度/加速度证书越界为 0、hold/branch jump 不增加，并且 teacher-qpos、末端 pose 和 jerk
回退处于明确小上限内才通过；它不具有任何硬件准入权威。

0727 episode 16/35/80/81 的 `runtime_causal` 回放结果：

- 固定 4 次迭代时，右臂最小 sigma 分别提高约 `0.16% / 0.15% / 3.43% / 16.67%`；
- 固定 2 次迭代时，分别提高约 `0.18% / 0.14% / 2.31% / 35.30%`；
- 8 个 episode/迭代组合的 raw/admitted qdot/qddot 越界计数均为 0，hold 和腕部分支跳变
  均未增加，独立 research gate 全部通过；
- 最困难的 episode 81 固定 4 次迭代中，平均 teacher 关节误差增加约 `0.071 deg`，平均末端
  平移误差增加约 `4.9 um`，平均旋转误差增加约 `0.00034 deg`。更强的 `cost=0.2` 虽能把
  sigma 提高约 `28.5%`，但关节误差代价约 `0.124 deg`，因此未选用。

当前结论仅支持“V5 是值得继续扩大离线数据覆盖的候选”，不支持替换 V4 或真机启用。离线
评估器提供 `--offline-sigma-escape-cost` 与
`--offline-sigma-escape-rate-per-sec` 做参数扫描，覆盖只作用于 candidate，registry profile 和
baseline 均不变。

### 0727 全量覆盖复核

随后将同一天、非 `old` 的
`2026-07-27_dmtac_rm65_bimanual_position_v2_responsive` 全部 115 个 episode 的
Parquet/qref 纳入只读扫描，而不是继续只观察最初 4 条轨迹：

- 以 Pose Lift 同模型、同 TCP、同 `0.5 m` 归一化 Jacobian 扫描保存的 teacher qref，只有
  9 个 episode 进入 `sigma_min < 0.04`：80、81、37、96、16、35、59、68、53；
- 另取 8 个正常 episode 做固定 2 次迭代对照，V5 与 V4 的 qref、pose、Jacobian 指标逐项
  相同，说明 V5 的 guard 没有泄漏到抽样的正常工作区；
- 固定 2 次迭代时，9 条 teacher-guarded 轨迹中实际进入 V4 guard 的 8 条均改善，累计
  `<0.01` 样本从 66 降到 58，`<0.001` 从 10 降到 8，累计 guard deficit 从约
  `26.3083` 降到 `25.6854`；
- 固定 4 次迭代时，7 条改善、episode 53 的 V4/V5 rollout 均未进入 guard，episode 96
  则出现反例：最小 sigma 从约 `1.557e-4` 降到 `7.828e-5`，`<0.001` 样本从 5 增到
  10。虽然其 `<0.01` 样本和平均 guard deficit 略有改善，但极深奇异尾部明确恶化；
- episode 96 保存的 teacher qref 最小 sigma 约为 `0.01055`，而 V4 Pose-to-QPOS
  closed-loop 会把它带到约 `1e-4`。因此该极深奇异并非演示 qref 原本经过的路径，而是
  部署 Pose Lift 重建分支造成的训测偏差；
- 逐帧检查表明 q3 在零点两侧换号，fixed2/fixed4 的单点最小值还会受参考时钟采样相位影响。
  因此后续判断不能只看最小值，也必须同时看 guard 暴露时间、`<0.01`/`<0.001` 尾部计数
  和 guard deficit。

离线 evaluator 已补充 `guarded_exposure` 与全 episode
`--rank-jacobian-episodes`。`sigma_escape_research` 门禁也增加同参考时钟样本数一致、guard
样本数、深尾部样本数及平均 deficit 非回退要求。新门禁复核 episode 96：fixed2 通过；
fixed4 同时因最小 sigma 和 `<0.001` 计数回退而失败。

结论更新为：V5 在已观察到的大多数软奇异轨迹上是正向候选，但其逃逸方向在穿越精确结构
奇异时对数值迭代敏感，尚不具备稳定的极深尾部安全性。它继续保持
`hardware_dispatch_allowed=false`，V4 默认和真机命令均不改变。下一候选应优先研究
q3/q5 零点附近的单侧分支保持或显式 barrier/trust-region，而不是提高现有 V5 cost。

## 2026-07-29 V6 结构奇异单侧分支锁（离线候选）

针对 episode 96 已确认的根因，新增独立 profile
`rm65_pose_lift_v6_structural_branch`，仍设置 `hardware_dispatch_allowed=false`：

- 在 V5 的 sigma guard 内，按当前 admitted anchor 选择 RM65 q3/q5 中更接近零的结构关节；
- 锁存该关节当前正/负侧，直到 sigma 离开 release band；
- 在 Pink QP 内加入单侧硬不等式，保留默认 `0.75 deg` 零点裕度，禁止数值迭代穿越 q3/q5=0；
- V4 的 qpos/qdot/qddot/collision 约束及 V5 的 sigma soft objective 仍保持原有权威关系。

0727 离线结果：

- episode 96 fixed2/fixed4 均不再穿越 q3=0；最小 sigma 分别约为 `0.002132/0.002136`，
  V4 的对应值为 `8.70e-6/1.56e-4`；`<0.001` 样本 fixed4 从 5 降为 0；无 hold，研究门禁通过；
- 9 条 teacher-guarded 轨迹 fixed4 中，8 条实际进入 guard 的轨迹全部通过研究门禁；episode 96
  是收益最大的一条，其余轨迹的 qref/pose/jerk 变化很小；episode 53 的 V4/V6 rollout
  均未进入 guard，按“无证据”单独标记，不作为收益样本；
- 8 条正常轨迹 fixed4 与 V4 的 sigma、qref 误差、hold、jerk 均逐项相同；
- 将裕度提高到 `1.5/2.0 deg` 可把 episode 96 最小 sigma 提升到约 `0.00426/0.00568`，但
  fixed4 最大 jerk 增加约 3.7%/2.7%，超过当前研究门禁，因此不采用更大裕度；
- `0.75 deg` 会在极深奇异区形成单侧安全平台（episode 96 q3 约停在 `+0.75 deg`），能避免
  分支抖动和精确穿零，但不能宣称完全恢复 teacher q3。后续若要进一步跟手，应研究带
  admitted-motion 方向的软 barrier/局部 trust-region，而不是直接扩大硬裕度。

补充参数扫描显示：把裕度降到 `0.5 deg` 时 episode 96 最小 sigma 约为 `0.00144`，仍无
`<0.001` 样本且 fixed4 jerk 基本不变；但安全余量更窄，因此暂不替换默认的 `0.75 deg`。
保持 `0.75 deg` 并将 V5 sigma escape cost 离线提高到 `0.3/0.4/0.5` 时，最小 sigma 约为
`0.00234/0.00658/0.01144`；`0.3` 通过当前研究门禁，`0.4` 因最大 jerk 回退失败，
`0.5` 还出现加速度证书越界。说明“加大逃逸权重”确实能更快离开平台，但会把新抖动/约束
冲突重新引入，当前不采纳。

对 cost=`0.3` 扩展到全部 9 条 guarded episode 后，episode 81/37 的 qref/pose parity
回退、episode 16 的 guard 暴露帧增加，故不能只凭 episode 96 选择更强权重；V6 默认仍保留
cost=`0.15`。

V6 只作为可回退离线研究 profile 保留；V4 默认、V5 实验 profile、部署端默认和所有真机命令
均未改变。

## 2026-07-29 V7 结构分支软外移探针（未通过全局门禁）

为缓解 V6 在 `0.75 deg` 单侧平台上的局部跟手损失，增加了仅离线可用的
`rm65_pose_lift_v7_structural_escape`：在 q3/q5 单侧硬约束外，按深度 sigma 逐渐增加
一个朝向 4° 的软外移目标。它不是独立安全 authority，V6 的分支锁、V5 的 sigma 目标和
V4 的 qpos/qdot/qddot/collision 证书仍优先。

0727 fixed4 结果：episode 96 单独运行时，runtime-causal 的平均最大关节误差约从
`1.696°` 降到 `1.562°`，最小 sigma 从约 `1.56e-4` 提升到 `6.7e-3`，但 8 条
guarded episode 的整体门禁失败：episode 80/16/68 出现速度或加速度证书回退，episode
35/59 的 sigma 改善不足，其他组合还出现 pose/parity 回退。把软外移 cost 从 `0.10` 提高到
`0.20` 会扩大多条轨迹的 parity 回退；把激活 sigma 收窄到 `0.005` 仍会在 episode 96
引入 jerk 回退。因此 V7 保持 `hardware_dispatch_allowed=false`，不进入 deploy、Pose
replay 默认或真机命令。

随后尝试的两种“减少 V6 粘滞”离线覆盖也未通过：首次锁定增加 q3/q5 触发角（4°）以及
普通区小边界、深 sigma 恢复 0.75° 的分段边界，均在 8 条轨迹中产生 motion certificate
回退。它们的临时参数没有写入 profile，V6 的 fingerprint 和默认语义保持不变。

当前可复现结论：V6 是结构奇异离线保护基线，V4 仍是唯一默认/可部署 profile；任何新的
“减小平台”方案必须同时通过 0727 全部 guarded episode 的 pose/parity、sigma 深尾、
qdot/qddot/jerk、hold 和独立 motion certificate 门禁后才可继续推进。

## 2026-07-30 Path-lift 合同复核与 V8/V9

在继续优化奇异逃逸前，已先核查会影响全部离线结论的三个合同：

- 对 0727 全部 115 个 episode、28,987 帧逐帧验证，stored Pose 与同帧 accepted `q_ref`
  的 FK 一致。左右最大平移误差约 `7.01e-8/7.59e-8 m`，最大旋转误差约
  `1.32e-5/1.55e-5 deg`；错开一帧后立即变成毫米/约 0.5° 量级。因此 episode 96 的
  closed-loop 深奇异是 Pose→QPOS path lifting 分支漂移，不是 Pose/qref 保存语义或时钟
  对齐错误。
- fixed2/fixed4 的 Pink 调用共享同一 admitted anchor、上一条正式 admitted velocity 和
  一个物理 `dt=1/60 s`。内部重新线性化不会提交 previous velocity，也不会把 4 个完整
  控制周期累计为一帧。
- evaluator 原先会为非 reference-grid 的最终 source timestamp 使用极短尾间隔计算速度和
  加速度，而 planner 仍按固定 reference period 积分。这会虚构数百倍动态越界。现已统一
  以 `1/reference_hz` 计算 planner emission 的 qdot/qddot/jerk，并增加非网格末帧回归
  测试。

还存在一个尚未统一、但已明确记录的观察口径：evaluator/V5/V6 使用线速度按 `0.5 m`
归一化的 geometric Jacobian；Pink FrameTask 使用 local SE(3) Log Jacobian 和自身的
position/orientation task cost。后续 directional-demand guard 必须先统一这套任务度量，
不能把 evaluator 的固定 `sigma` 阈值直接当成 Pink objective 的同一物理量。

### V8：exact nonlinear best-so-far 未通过 closed-loop 门禁

新增仅离线 profile `rm65_pose_lift_v8_nonlinear_best_candidate`，对 fixed4 的 nonlinear
iterate 用真实 FK、真实 Jacobian、joint/step certificate 和 endpoint collision 评价，
并返回当前帧 best-so-far，而不是无条件返回最后一次迭代。

episode 96 的当前帧指标有所改善：最大关节误差约 `14.84°→13.76°`，最小 sigma 保持 V6
水平。但候选一旦被选中，就会改变后续 closed-loop anchor；扩展到全部 guarded episode
后，episode 80/16/68 出现动态证书回退，episode 16/81/35/59 还出现 parity、pose 或 guard
暴露回退。结论是：

> 单帧 exact 最优不等价于关节路径提升最优。

V8 保留为可复现失败探针，`hardware_dispatch_allowed=false`；不得替换 V4 或 V6。

### V9：仅深尾启用的预测式平滑 Barrier

新增仅离线 profile `rm65_pose_lift_v9_predictive_branch_barrier`。初版曾在整个
`sigma<0.04` guard band 使用上一条 admitted velocity 预测 150 ms，并用单边 CBF

```text
sign*qdot + gain*(sign*q - margin) >= 0
```

替代 V6 的一步硬 margin。gain/horizon 扫描没有跨 episode 支配解：提前介入可以改善部分
软 guard 轨迹，却会提高 episode 96 jerk；进一步增强还会损害 episode 16 pose。

因此注册 V9 收窄为 `hybrid_deep_barrier`：

- `sigma>=0.01` 时完全保留 V6 hard margin；
- 仅当前或 150 ms 预测 `sigma<0.01` 时启用 gain=`6` 的平滑 Barrier；
- V4 的 qpos/qdot/qddot/collision admission 和 V5/V6 的已有机制不变；
- `hardware_dispatch_allowed=false`。

0727 fixed4 closed-loop 结果：

- episode 96 右臂最小 sigma `0.002136→0.002363`，`<0.01` 样本 `52→51`；
  最大关节误差 `14.844°→14.746°`，平均最大关节误差 `1.674°→1.667°`；
  最大速度 `137.08→136.39 deg/s`，最大 jerk `368499→368169 deg/s³`；
  pose p95 不变，无 velocity/acceleration certificate、hold 或 branch 回退。
- episode 80/81/37/35/59/68 与 V6 的最小 sigma 完全相同；episode 16 仅存在
  `7.5e-9` 数值差，其他门禁不回退。
- 正常 episode `88,102,100,46,94,38,2,39` 的完整 compact report 与 V6 逐项相同。

现有 `sigma_escape_research` 要求每个 guarded side 都必须获得实质 sigma 增益，所以整组
V9 verdict 仍为 false。不能把“其余轨迹中性”改写成“整组通过”。当前只证明 V9 对已观察
到的 episode 96 深尾有小幅正收益，不能证明它可替换 V6，更不能提升为部署选项。

### 当前权限与下一阶段

```text
deploy / Pose replay 默认：V4
显式原子回退：V1
离线结构基线：V6
离线深尾候选：V9
离线失败探针：V7、V8
严格复现：QPOS replay 直接回放 accepted q_ref
```

真机命令和默认 profile 均不改变。下一阶段不再扫描单一 gain 或 posture cost，优先级为：

1. 构建独立的 q3/q5 局部奇异风洞，覆盖分支、进入/离开方向、初速度、reference 频率和
   小扰动，避免继续围绕 episode 96 过拟合。
2. 统一 evaluator guard 与 Pink task 的位置/姿态尺度，再记录弱奇异方向需求
   `|u_i^T xi_desired|/(sigma_i+epsilon)`。
3. 比较 baseline、q3 Barrier、q5 Barrier、双 Barrier 和 directional-relaxed 候选，使用
   跨帧 rollout 与字典序证书选择，不再使用单帧加权总分。
4. 将有限时域 terminal 条件改成 task-space tube、合法分支和动态约束，而不是强制回到
   V4 的同一个关节解。
5. 只有当目标确实要求跨越结构分支时，才单独设计受控 `REBRANCH`；它必须显式接受有限
   姿态 slack，不能伪装成普通单帧 IK。

完整实验表与复现边界见 `docs/real/POSE_LIFT_EXPERIMENTS_0727.md`。

## 2026-07-30 跨日期扩大验证

在不改变任何 profile、阈值或 evaluator 代码的前提下，继续盘点并验证远端历史数据：

- 新增可用 teacher 数据为 0723 的 2 episodes/326 帧、0724 的 1 episode/55 帧，以及
  单独标注为旧遥操 profile 的 0727 `v1_old` 35 episodes/17,422 帧；
- 加上 responsive 0727，accepted-qref/Jacobian 可审计总覆盖达到
  153 episodes/46,790 帧；
- writing 503 episodes、0724 v2.2/filtered 241 episodes、0725 filtered 310 episodes
  的 replay-qpos 字段全为空，不能将 observation qpos 冒充 accepted-qref teacher。

新增 38 个 episode 已完成 V9↔V6 fixed4 closed-loop 全长对比，共 17,803 source frames、
72,758 个 60 Hz reference：

- 38/38 的 candidate 与 baseline compact report 逐项完全相同；
- 38/38 的 production non-regression 不变；
- old 只有 episode 13/17 进入软 guard，没有任何 episode 进入 `<0.01` 深尾；
- 0723/0724 跨日期正常轨迹也没有 guard 泄漏。

另对 old episode 13/17 执行 V6↔V4 比较，两条均通过 23 项研究门禁；最小 sigma 分别由
`0.022089→0.022132`、`0.031528→0.031579`，没有 pose、动态或 certificate 回退。

该结果提高了 V9 “正常/软 guard 中性”的可信度，但没有增加独立深尾收益样本。因此权限
保持不变：V4 默认、V1 回退、V6 离线结构基线、V9 离线深尾候选。下一步优先获取或合成
`sigma<0.01` 的独立 path-lift 样本，而不是继续扩大只有正常区的数据量。

## 2026-07-30 branch-aware short-horizon path-lift 规划

已停止的路线：

- [x] 停止继续扫描 V5 sigma cost；
- [x] 停止继续扫描 V9 gain/margin；
- [x] 不再使用每帧 nonlinear best candidate 作为 closed-loop 路径选择；
- [x] 不再使用 exact V4 joint rejoin 作为 horizon terminal hard gate。

### 阶段 1：任务度量与因果诊断统一

- [x] 保留 `sigma_geometric`，不删除已有历史比较合同；
- [x] evaluator 调用与 Pink 相同的 local SE(3) Log error/Jacobian；
- [x] 使用 Pink task cost 和正式 joint velocity capability 构造
  `sigma_task_velocity`；
- [x] 输出 singular-direction demand 和 required velocity utilization；
- [x] 增加可选 detached strict directional alpha 与 best constrained residual；
- [x] strict alpha 改用确定性 LP，避免病态 OSQP 把 max-iteration 当作风险；
- [x] 区分可用于因果触发的 `pre_solve` 与 post-admission stream；
- [x] 输出 active inequality source，区分结构 Barrier、普通 joint limit 与
  admitted motion limit；
- [x] 完成 10 条、9065 reference 的首轮非深尾对照；确认 strict alpha 在正常自由区
  可保持 1.0，但会被普通关节限位/动态约束拉低，不能单独作为奇异触发器；
- [x] 完成 7 条、9065 reference 的首轮 guard 对照；所有 episode 至少一侧
  directional demand 峰值超过 `1x`，但该集合由已有 guard 筛选，尚不能作为独立阈值；
- [x] 全量完成 0727 responsive 115 episode、117030 reference 的 V4 closed-loop
  task metric；demand `>1/2/5x` 分别覆盖 39/20/4 个 episode；
- [x] 完成 demand `>1x` 全部风险层的逐帧 branch 分类；39 条中只有 ep96 明确跨
  q3/q5 结构边界；
- [x] 完成 0723/0724 三条、1877 reference 的跨日期健康 held-out；0725 因缺 accepted
  q_ref 明确排除；
- [x] 在现有真实集统计初步 trigger 辨识：demand `>1.5x` 对 episode
  `max_joint_error>3°` 为 `TP=26/FP=1/FN=0/TN=88`；
- [ ] 在独立危险风洞上统计触发提前量、false positive 和 false negative，禁止用同集
  `1.5x` 直接定生产阈值；
- [x] 定位 ep96 第一次明确结构事件：sequence 435 V4 右 q3 跨负，而 teacher
  仍为 `+3.652°` 且整段不跨零；新指标在约 3.1 s 前已预警；
- [ ] 在 q3 符号跨越前进一步区分“可恢复连续偏差起点”和“不可逆 branch commitment”。

阶段 1 不拥有控制权限：不得改变 command、QP、admission、reference advancement 或
dispatch。`--include-constrained-feasibility` 默认关闭。

### 阶段 2：局部奇异风洞与 branch oracle

- [x] 新增 `evaluate_rm65_singularity_wind_tunnel --no-motion`，从真实 accepted
  q_ref 生成 q3/q5 结构种子并执行连续 closed-loop rollout；
- [x] 支持任意 repeatable offset；首轮已覆盖 `±0.75°`，其余
  `±0.1/0.25/0.5/1/2/4°` 尚待矩阵扩展；
- [x] 支持左右臂、正负 branch、静止/向内/向外初速度；
- [x] 支持最弱/次弱/强奇异方向和正负方向；混合 twist 尚未实现；
- [x] 支持进入—离开和持续推压连续轨迹；可绕行/horizon 目标尚未实现；
- [x] 按 seed identity 确定性分离 tune 与 held-out 风洞；
- [x] 完成首轮 0727 `±0.75° × weakest × persistent_push`：
  tune 36、held-out 52 个有效 case；
- [x] 定位并修复 structural latch 跨 planner reset 泄漏：reset 新 epoch 必须从当前
  authoritative q 重新锁存分支；
- [ ] 只读验证 IK-Geo/EAIK 对 RM65 结构的兼容性；
- [ ] 将解析多解先作为 branch oracle/seed，不替换在线 Pink 主环。

### 阶段 3：V10 directional feasibility guard

- [x] 注册 default-off、`hardware_dispatch_allowed=false` 的 V10 research profile；
- [x] 实现 enter/release 双阈值、方向选择性 task scaling、离开方向放宽和平滑 q3/q5
  Barrier；没有逐帧 V4/V6/V9 candidate 贪心切换；
- [x] 风洞 tune/held-out 中把 V4 的 `10/10` 个 crossing 均降为 `0/0`，且 V10
  `0` Hold；最坏 `sigma_task` 分别提升到约 `0.00275/0.00293`；
- [x] 对真实 guard episode 16/35/37/59/68/80/81/96 完成 runtime-causal
  production comparison；
- [x] 明确否决 V10 production：8/8 episode verdict false，ep16/68/81 出现显著
  parity/动态回退，证明真实路径包含合法/必要 rebranch；
- [ ] PREVENT/PROTECT/RECOVER 的完整 dwell/subspace tracking 尚未实现；在 V11
  branch path 判断前暂停继续强化 V10；
- [ ] 独立 trigger correctness 与 conditional benefit verdict 仍需正式拆分。

### 阶段 4：V11 short-horizon branch path lift

- [x] evaluator 已支持 primary/alternate/shadow 三路 fixed-iteration H8 rollout、
  plan lock、逐窗口门禁原因和 `task_space_tube` terminal；该原型没有部署权限；
- [x] 验证 V4→V10 H8：episode 96 的 17 个 guard decision 中只有 3 个同时满足
  sigma/Pose/terminal，但均动态回退，最终 0 次接管；
- [x] 验证 V4→V6 H8：严格动态门禁下 0 次接管；只读 1%/2% sensitivity 中，2%
  在 guard8 的 5 条轨迹实际接管，但 5/5 production verdict false，另外 3 条 pass
  只是因为 0 次接管；
- [x] 明确停止继续放宽 selector dynamic slack：现有 V4/V6/V10 profile rollout
  不是合格的多分支候选生成器；
- [ ] 离线 Pose replay 实现 all-IK graph 或显式 stay-q3/stay-q5/双 Barrier/
  authorized-rebranch 多 seed horizon SQP；
- [ ] 优化完整 `q[1:H]`，不是当前一帧；
- [x] terminal 已从 exact shadow q 扩展为 task-space tube；
- [ ] terminal 继续加入合法 branch、release risk 和动态可继续条件；
- [ ] 候选按 hard certificate → branch → horizon alpha/sigma → pose → dynamics →
  parity 的字典序选择；
- [ ] 离线 lookahead 与 runtime-causal predictive capability 分开注册；
- [ ] 实时版本异步运行，带 epoch/anchor hash/deadline，超时回退 V10。

### 阶段 5：V12 authorized rebranch

- [ ] 只有所有 stay-branch horizon 均不可行时才能请求；
- [ ] 显式记录 requested/admitted/active/completed/aborted；
- [ ] 根据 task importance 和弱奇异子空间选择临时 slack 方向；
- [ ] 规划完整受限 crossing/rejoin 轨迹；
- [ ] 普通 V4/V10 不允许隐式 branch crossing。

当前权限继续保持：

```text
默认 deploy / Pose replay：V4
原子回退：V1
局部结构离线基线：V6
深尾 shadow：V9
下一研究 profile：V10
中期核心：V11
最终显式换支：V12
```

2026-07-30 更新：V10 已完成局部可行性验证，但没有获得 production 权限；它只表示
V11 的局部 fallback 组件，不是下一部署候选。V11 H8 selector/task-space terminal 原型
也已落地，但 V4/V6/V10 候选集在 guard8 上没有整体正收益。任何后续 V11 比较仍以 V4
为 production baseline，V1 为原子回退。

## 2026-07-30 官方全解 IK 路线复核

在不改变默认 deploy 和真机命令的前提下，已完成 bundled RealMan
`rm_algo_inverse_kinematics_all` 的纯离线合同与全量真实数据验证：

- [x] 空 handle 初始化，无设备枚举、连接、motion 或 dispatch；
- [x] SDK RM65 FK 与项目 Pinocchio flange/TCP 合同对齐；
- [x] 0727 responsive 全部 115 episodes、28,987 source frames 双臂 all-IK
  `0` 失败；
- [x] 最坏 per-episode source solve p99 低于 `11 us`；
- [x] source Pose 的官方候选中持续存在 recorded accepted-qref 所属分支；
- [x] 证实 60 Hz Cartesian 插值不能假设逐帧精确可达：完整集左/右分别有
  `53/223` 个无 project-valid candidate reference；
- [x] 证实 source-rate all-IK 后插值 qref 可在完整集保持 teacher 分支：
  source/runtime endpoint collision failure 均为 0，最坏关节差约
  `0.043°/0.030°`，最小 sigma 保持 teacher 级别；
- [x] 否决 shared Cartesian alpha governor；
- [x] 否决 official remote IK 直接替换 V4；
- [x] 否决 Pink soft branch anchor 和 frame-local Pink residual；
- [x] 证实现有 acceleration-limited qcmd planner 不能被误当作任意 15 Hz
  大目标的无滞后 jerk-limited reference governor。

新的研究职责拆分为：

```text
source Pose all-IK / branch selection
        ↓
explicit qref interpolation
        ↓
independent time law / jerk-limited governor
        ↓
canonical collision + admission + qcmd execution
```

下一步只推进 default-off、可原子回退的实现：

- [x] `robot_motion.types.IKCandidate` 继续作为 backend-neutral all-solution
  candidate；新增 `RM65SourceIKPairSelection` 与
  `select_nearest_collision_safe_rm65_ik_pair`，vendor 调用和进程隔离仍归
  `hardware`；
- [x] source-rate selector 已使用 admitted source qref 距离、canonical collision
  rejection reason 和 deterministic enumeration tie-break；qdot-aware hysteresis
  留到 runtime backlog 接入时完成；
- [ ] 将 `source Pose IK → qref interpolation` 注册为显式 capability/profile，
  不覆盖现有 `Pose interpolation → V4 IK`；
- [ ] 对每个 60 Hz qref endpoint 和 joint segment 做 canonical collision certificate；
- [ ] 为 all-IK failure、合法分支暂时消失、source gap、epoch reset 定义 Hold/
  Pink fallback，不允许 silent branch switch；
- [ ] 建立真正 jerk-limited OTG 与现有 planner 的同输入 A/B；依赖未选定前不修改
  `pyproject.toml`、环境或默认参数；
- [ ] 离线 Pose replay 使用 future-aware all-IK graph/DP；实时 deploy 只使用
  causal source candidate 与显式有限 lookahead，能力名称分开；
- [ ] 全量 production gate 不再把 joint parity 当唯一硬真值，但继续报告 parity，
  并以 Pose、限位、碰撞、qdot/qddot/jerk、branch continuity 和 latency 为主门禁；
- [ ] 通过 dry-run replay 后再讨论 hardware dispatch 权限；当前禁止真机启用。

权限保持：

```text
默认 deploy / Pose replay：V4
原子回退：V1
新 default-off 主线：official source all-IK + qref interpolation
```

### 2026-07-30 qref→qcmd 动态层离线可行性

- [x] 新增 `todos/probes/rm65_qref_dynamics_probe.py`，只读 0727 parquet，
  不访问设备、不 dispatch。
- [x] 动态差分按 `collection.continuity_segment_index` 分段，禁止把 pause/gap
  两侧误当作一个连续控制区间。
- [x] 在 115 episodes / 28,987 source frames 上验证 future-aware 120 Hz
  constrained projection：
  - qdot/qddot 使用 production position limits；
  - jerk 暂按 `qddot_max / 0.10 s`；
  - 最坏 qref 偏差左/右 `0.743°/0.757°`；
  - acceleration 最坏降至 `3031°/s²` / `3002°/s²`；
  - jerk 最坏降至约 `9.17e4°/s³`。
- [x] 将完整未来 oracle 改为 2/3/4 个 source step lookahead 的因果
  receding-horizon 对照；每个 control cut 继承上一 cut 的 q/qdot/qddot。
  ep96 上 3/4 步无可见收益，全量采用两步（约 133 ms）。
- [x] 对两步因果候选完成全量 FK Pose、双臂 endpoint collision 和相邻
  qcmd interval 的 25%/50%/75% 内点采样：
  - 115 episodes / 229,131 endpoints / 228,736 intervals；
  - endpoint 与内点采样 failure 均为 0；
  - FK worst 左 `3.213 mm / 1.069°`，右 `4.771 mm / 1.340°`。
- [ ] 将 interval 内点采样升级为严格连续 segment collision certificate；
  当前结果明确标记 `continuous_segment_collision_certified=false`。
- [x] 全量两步因果动态对照：
  - acceleration worst 左/右 `2862.5/2842.5°/s²`；
  - jerk worst 左/右 `91629.5/91598.1°/s³`，未越过约 `91673°/s³` 上限；
  - qref error worst 左/右 `1.295°/1.817°`，p95 `0.971°/1.462°`；
  - source keyframe error worst 左/右 `0.759°/1.403°`。
- [x] 将因果 QP 抽为
  `robot_motion.position_control.plan_jerk_limited_joint_horizon`：
  - backend-neutral、无 clock/authority/dispatch/committed-state；
  - 输入 last-admitted q/qdot/qddot 与有限 qref horizon；
  - 输出仅是待 canonical admission 的 control-cut candidate；
  - 离线探针已改为直接调用共享实现，并在 115 episodes 上复现全部结果。
- [ ] 明确 action backlog/time dilation 合同：若持续目标超出动态能力，选择
  有界延迟、丢弃旧目标或 fail-closed，不能静默积累。
- [ ] 将共享 governor 接成独立 default-off execution profile；不得塞入 Pink
  pose-lift profile，也不得覆盖现有 V4/default qcmd planner。

```text
60 Hz intermediate exact projection：仅诊断
official remote IK：已否决直接替换
```

### 2026-07-31 同集 V4 对照与 governor 参数收敛

- [x] 保存固定 16 episodes 的 V4 closed-loop JSON，避免终端截断；
- [x] 加入新链路全量 worst 所在的 7 episodes，形成 23 条反向压力集；
- [x] 对同一 23 条比较 `0.10/0.075/0.05 s` jerk ramp；
- [x] `0.05 s` 相对 V4：
  - joint/translation/rotation/acceleration/jerk 均 `23/23` 改善；
  - velocity peak `12/23` 改善、`11/23` 回退，但整体 worst
    `109.3°/s < 197.9°/s`，且无逐关节速度违规；
  - 双臂整体最小 sigma 保持约 `0.01028`，避免 V4 ep96 的
    `8.70e-6` 深尾；
- [x] `0.05 s` 重跑完整 115 episodes：
  - source/dynamic/endpoint/采样 interval failure 全部 `0`；
  - joint worst 左/右 `0.773°/1.113°`；
  - Pose worst 左 `1.947 mm / 0.672°`，右
    `2.756 mm / 0.830°`；
  - velocity worst 左/右 `93.38/111.06°/s`；
  - acceleration worst 左/右 `4586.3/4575.3°/s²`；
  - jerk worst 约 `1.832e5°/s³`；
- [x] 修复 exact-limit OSQP 数值残差：位置不等式改为 radian 尺度并保留
  `1e-6 rad` 内缩；完整集 endpoint failure 由 ep24/ep99 的两个数值假阳性降为 0；
- [x] governor 单元测试断言整个 q path 严格位于硬关节限位内。

当前研究参数固定为：

```text
source selector: official all-IK + causal nearest collision-safe branch
qref: linear source-q interpolation
governor lookahead: 2 source steps
governor command clock: 120 Hz
jerk ramp: 0.05 s
```

`0.10 s` 保留为更平滑但 lag 更大的原子研究回退；不继续缩短 ramp，因为 0.05 s
已经消除固定 23 条中的 Pose/joint 回退，继续放大 jerk 没有必要。

### 下一步 default-off deploy shadow 的所有权边界

不要把新链路注册成另一个 Pink `pose_lift_profile`。它跨越 source-rate geometry 与
qcmd time-law 两个阶段，必须使用独立 execution capability：

```text
rm65_source_pose_lift = v4_reference_ik | official_all_ik_v1
rm65_qref_governor = existing | jerk_horizon_v1
```

按以下顺序接入，任何一步失败都保持 V4 默认不变：

1. [ ] 在 `teleop_runtime` source-keyframe acceptance 处调用 backend 提供的
   all-solution candidate port；不在 `robot_motion` 初始化 vendor SDK；
2. [ ] selector 只读取上一条 admitted source qref、候选和 canonical collision
   certificate，输出带 generation/source-index/anchor-hash 的 qref receipt；
3. [ ] source Pose 成功 lift 后写入独立 qref buffer，不复用或同时推进现有 pose
   resampler cursor；
4. [ ] horizon governor 读取同一 epoch 的至少两个 source-step backlog，并以 last
   admitted qcmd/qdot/qddot 初始化；输出仍只是 candidate；
5. [ ] final qcmd 继续经过 canonical joint-limit/collision admission，只有 receipt
   接受后才更新 governor state；
6. [ ] shadow 模式只记录 V4 与新链路 difference，不影响 authority、dispatch、
   source cursor 或 episode；
7. [ ] source all-IK 无解、deadline miss、branch disappearance、anchor hash
   mismatch 或 backlog 不足时，shadow 只标记失败；未来 active 模式必须整段回退
   V4/hold，禁止在一个 control cut 内混合两条路径；
8. [ ] 先完成 dry-run replay 的 p50/p95/p99 latency、fallback rate、candidate
   count、endpoint 与严格 continuous segment collision，再讨论
   `hardware_dispatch_allowed=true`。

原子回退要求：

```text
feature flag 默认 off
V4 代码路径不改语义
新 worker/buffer/state 可整体不构造
一次 generation cut 只允许一个 authoritative path
异常不得把 shadow qref 写回正式 resampler
```
