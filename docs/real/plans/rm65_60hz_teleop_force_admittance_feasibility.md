# RM65 60 Hz 遥操作力控阻抗可行性方案

## 结论

可以为当前 60 Hz 位控遥操作增加接触柔顺，但第一阶段不应把现有
`impedance_40hz` 后端强行提高到 60 Hz，也不应同时运行位控和阻抗两个后端。

推荐的低风险方案是：

```text
VIVE pose intent @ 60 Hz
             +
单一 wrench producer @ 20/40 Hz
             -> 滤波、deadband、外环导纳状态
             -> 60 Hz 有界 Cartesian offset / reference rebase
             -> 现有双臂 IK、碰撞、关节限位和 admission @ 60 Hz
             -> 现有 qref planner 与 qcmd transport @ 120 Hz
```

这属于“60 Hz 位控遥操作上的主机侧 Cartesian 导纳/柔顺覆盖层”，不是底层关节
力矩阻抗。当前 RM65 SDK 也提供控制器侧透传力位混合接口，可作为后续独立候选，
但现有项目只做过低频 smoke，尚不能直接证明它与当前 60/120 Hz 正式链路兼容。

## 当前实现事实

### 60 Hz 位控链路

当前正式 position profile 的核心链路为：

- VIVE/参考与双臂 IK：60 Hz；
- qref 规划及最终 qcmd：120 Hz；
- 位控受力急停：独立约 20 Hz wrench 读取，保持 fail-closed、双臂 Arm_Stop 锁存；
- IK、碰撞、关节动态限制、reference 与 dispatch 均已有唯一权威 owner。

### 现有 `impedance_40hz`

当前 impedance 实现是主机侧外环导纳：读取六维力、形成有界 Cartesian pose
offset、再次求 IK，最后仍下发位置命令。它不是关节力矩环意义上的真实阻抗。

其控制器与 backend 明确限制 `control_hz <= 40`，而且拥有自己的一套状态、IK 和
command thread。因此：

- 去掉 40 Hz 上限并不能自动得到可信的 60 Hz 阻抗；
- 与现有 position backend 并行会产生两个 IK/reference/dispatch owner；
- 直接混合两条链路违反单一运动资源只有一个权威 owner 的架构合同。

### RM65 控制器侧力位混合

SDK 与现有 adapter 已提供：

- `Start_Force_Position_Move` / `Stop_Force_Position_Move`；
- joint 或 pose 透传力位混合；
- 新接口支持 Fx/Fy/Fz/Mx/My/Mz 分轴固定、浮动、弹簧、运动、力跟踪等模式；
- 最大线/角速度、期望力和传输平滑参数。

普通 RM65 backend 也已有 `force_position_control` 分支，可把 qpos 改由
`Force_Position_Move_Joint` 下发。不过正式遥操作 builder 当前没有启用它，现有
smoke 的循环约为 10 Hz，尚未覆盖 60 Hz qref、120 Hz qcmd、双臂同步、deadline、
停止恢复、奇异保护与数据合同。因此它是“有潜力但需要独立验证”的方案，而不是
现成安全开关。

#### 2026-08-06 隔离真机验证结论

左臂 RM65-6FB + DMTAC_M 的 mode-1 `float` 已确认会沿指定工具 Y 轴产生真实运动，
Ctrl+C、力/位移/关节故障和新增实测速率保护也都能触发并取得至少一条成功的
Arm_Stop receipt。但当前控制器参数呈现明显死区和陡峭增益：3 N 以下接近刚性，
7--9 N 时中位速度升至约 10.7--13.6 mm/s，并出现约 20 mm/s 的短时测得峰值。

因此 Phase 4 当前结论为“能力存在，但默认参数下不进入正式遥操”。禁止继续通过增加
人工施力、`limit_vel`、位移或关节漂移上限来寻找柔顺区间。若未来重启这条路线，必须
在 teach pendant 中一次只调整一个 M/B/K 参数，并保留实测速度和独立 Arm_Stop
保护。完整证据见
[`docs/real/reports/RM65_NATIVE_FORCE_POSITION_PROBE_20260806.md`](../reports/RM65_NATIVE_FORCE_POSITION_PROBE_20260806.md)。

## 推荐方案：有界外环导纳覆盖层

### 所有权与数据流

1. 保留当前 position profile 为唯一 motion/reference/IK/admission/dispatch owner。
2. wrench producer 只生产带时间戳的力状态；安全 guard 与导纳 modifier 都是其有界
   被动消费者，不允许各自并发调用同一个 SDK 读接口。
3. 导纳 modifier 只生成 Cartesian feedback offset 或 rebase intent，不直接下发
   joint target，也不维护第二份机械臂权威状态。
4. 原始 VIVE intent 与力反馈合成后的目标仍必须通过现有 IK、碰撞、关节限位、
   速度/加速度/jerk 和 admission。
5. 禁止在 IK 后叠加 joint offset；否则会绕过末端一致性和碰撞/动态证书。

### 多速率合同

- 仅在收到新的 wrench sample 时按真实 force `dt` 更新导纳状态；不得在 60 Hz
  重复积分同一条力样本。
- 60 Hz 只读取最新导纳状态，并进行 bounded hold、插值或 slew limit。
- 导纳闭环带宽应明显低于 wrench producer 采样率；首版以低频接触让步为目标，
  不追求高带宽力跟踪。
- 每条 wrench 必须带 source timestamp、receive timestamp、frame、tare generation
  和 freshness。

### 初始行为语义

首版建议只启用平移方向的“零力让步”，暂不启用旋转柔顺或指定力跟踪：

- 接触力超过 deadband 后，目标沿卸载方向产生小幅 pose offset；
- offset、offset velocity 和积分状态均有限幅；
- Hold、单臂 fix、双臂切换时按 side 独立冻结/重置，inactive side 不积分；
- 从 Hold 或切换恢复时，以最近 admitted/realized pose 初始化，不能产生跳变。

仅做弹簧回到原始 VIVE 绝对目标会产生一个实际问题：操作者继续把目标推入物体时，
力释放后系统会重新向物体追赶。推荐加入 contact rebase：机器人侧 reference 仅按
admitted/realized motion 推进，或者动态更新 hand-to-robot offset。这样接触让步后
不会积累大量未执行误差，也不要求操作者先把手退回旧位置。

### 坐标与状态

- wrench、offset、VIVE target 和 TCP 必须明确在 tool/work frame 中转换；
- 姿态合成使用 SE(3)/rotation vector，不做 Euler 分量直接相加；
- 导纳 anti-windup 应以最终 admitted qref 的 FK/realized pose 反向同步；
- force tare 应在 runtime Home 完成后建立，并在人工调整/重启后重新建立 generation。

## 安全边界

现有 position force guard 是独立的 canonical safety，不由柔顺层替代或放宽：

- 超过硬力/力矩阈值仍立即进入锁存 Arm_Stop；
- wrench stale/read error 时不得让 offset 自动回零并重新推向接触面；应冻结 modifier，
  并由现有 fail-closed watchdog 决定 Hold/Stop；
- 首版 translation-only，旋转/力矩轴关闭；
- offset、offset velocity、导纳积分、单周期 pose step 均使用保守上限；
- 具体 compliance/deadband 不直接凭经验写入正式 profile，先由 shadow trace 标定；
- 不修改碰撞检查，合成目标继续经过既有 collision admission；
- debug/trace 必须异步有界，不允许在控制热路径写盘或等待 GUI。

## 数据与训测合同

为了可解释和可复现，需要分别记录：

- raw VIVE pose intent；
- raw/tared/filtered wrench、frame、timestamp、freshness；
- requested admittance offset；
- 经 IK/动态/碰撞后实际 admitted 的 offset；
- composed pose target；
- accepted qref 及其 FK action pose；
- compliance state、contact/rebase 状态、stale/limit/hard-stop reason。

正式 action 仍应是 accepted qref 的 FK/admitted action，而不是 raw VIVE target。
如果策略训练要复现柔顺行为，应明确选择：

1. 把力反馈 modifier 作为部署时同样存在的 runtime feedback governor；或
2. 让策略显式观察 wrench/contact state 并学习该行为。

不能把被柔顺修正过的 action 与未记录的 raw target 混为同一合同。

## 候选方案比较

### A. Position + 主机侧外环导纳（推荐首选）

优点：

- 保留已稳定的 60/120 Hz IK、规划、碰撞和 dispatch；
- 容易 default-off、逐侧启用和原子回退；
- 能离线 shadow，风险与输出均可完整记录；
- 不需要宣称或依赖底层 torque control。

缺点：

- 接触带宽受 wrench 采样、网络和 position transport 限制；
- 需要正确设计 rebase，否则会反复推压/追赶；
- 不是高性能装配意义的真实阻抗。

### B. RM65 控制器侧透传力位混合（第二阶段研究候选）

优点：

- 力位补偿在机器人控制器内部完成，理论上接触响应可能优于主机外环；
- SDK 原生支持分轴模式、期望力及速度限制。

缺点与未知项：

- 未验证当前控制器/SDK 在 120 Hz joint passthrough 下的 deadline 和稳定性；
- 控制器内部补偿后的 realized pose/qpos 与上游 qref 可能不一致，reference、碰撞预测、
  数据 action 及重启恢复合同都必须重新定义；
- mode start/stop、Arm_Stop、失联、force stale 和双臂同步的失败语义更复杂；
- 不能只打开现存 bool 就进入正式数采。

### C. 把现有 `impedance_40hz` 强行升至 60 Hz（不推荐）

原因：现有硬上限和验证边界明确为 40 Hz；它自带第二套 IK/状态/command owner，
对稳定 position 链路侵入最大，且不能解决力数据本身低频、stale 与 reference rebase。

## 分阶段实施与验收

## 2026-08-06 主线收敛与代码落点

隔离原生 float 试验到此结束，后续回到主机侧 Cartesian admittance 主线。当前代码
审计补充确认了两个必须遵守的事实：

1. `PositionForceGuardMonitor` 已经是左右 RM65 wrench 的单一 20 Hz 生产者，并提供带
   `monotonic_time_sec` 的原子 snapshot。柔顺层只能消费该 snapshot，禁止增加第二个
   SDK force reader。
2. `DualArmCollisionAwareRetargeter.target_override_filter` 正好位于 VIVE 目标成形之后、
   双臂 IK/碰撞/动态 admission 之前，但当前 InitArm 已占用这个单一回调。不能用新
   callback 覆盖旧 callback；必须先建立确定顺序的组合器，并明确 Init/Hold/Fix 优先级。

### 权威链路与组合顺序

首版运行顺序固定为：

```text
VIVE target shaping
  -> InitArm target override（若 active，独占目标且导纳不积分）
  -> per-side Cartesian admittance modifier（仅普通 RUNNING/live side）
  -> existing dual-arm IK
  -> existing collision/joint-limit/dynamics admission
  -> accepted qref / 120 Hz qcmd
```

- InitArm active 时直接 bypass admittance，并清除该侧未实现 offset；不能对归位目标施力
  偏置。
- Hold 时两侧导纳冻结，不能自动回零后重新推向接触面；重新采集 runtime Home 后以零
  offset 和新 tare generation 开始。
- Fix 时 fixed side 不积分、不输出 offset；live side 可独立工作。单/双臂切换只要求
  新 live side 建立自己的有效状态，不得让另一侧缓存阻塞切换。
- force guard fault 永远先于 modifier；现有硬阈值、stale watchdog、Arm_Stop 及 collision
  检查保持不变。

### Slice A：纯数学与状态合同（下一步，完全离线）

状态：2026-08-06 已实现并通过离线合同测试；随后已由 Slice B/C 薄适配器接入 CLI，
纯模块本身仍不拥有 target、IK、collision、dispatch 或真机资源。

归属放在 `robot_motion/position_control/`，因为它是共享运动语义，不是 RM65 SDK 生命周期。
该模块只接收数值和显式时间，不导入 `hardware`、`teleop_runtime`、GUI、数据集或线程：

- `CartesianAdmittanceConfig`：逐轴 enable、deadband、M/B/K、offset/velocity/slew 上限；
- `CartesianAdmittanceState`：offset、velocity、最后消费的 sample identity、tare generation；
- `WrenchSample`/等价纯输入：values、frame、source/receive time、generation；
- `step_new_sample(...)`：只有 sample identity 变化时按 force dt 积分一次；
- `sample_output(...)`：60 Hz 只读取/限速当前状态，不重复积分旧的 20 Hz sample；
- `freeze/reset/rebase(...)`：明确 Hold、Fix、Home 和 admission anti-windup 的状态转移；
- 首版仅 translation-only，旋转三轴必须严格为零。

本切片不加 CLI、不接真机、不改变任何 target 或 command。

最低测试矩阵：

- disabled、deadband 内及零力输入严格 identity；
- 同一 20 Hz sample 被 60 Hz 循环读取三次时只积分一次；
- sample stale 时冻结，不回零、不追赶；
- timestamp 回退、generation 改变、过大 dt 均 fail-closed reset/freeze；
- offset、offset velocity、slew 和单步位移始终受限；
- tool/base frame 变换在左右臂上符号正确；
- Hold/Fix/Init transitions 不遗留旧 offset；
- 模块不产生 qpos，也不拥有 IK、collision 或 dispatch。

### Slice B：Position teleop shadow adapter（default-off，不改 command）

状态：2026-08-06 已实现并通过离线/入口回归；同日已完成首轮双臂真机 shadow
采集（DMTAC_M，约 26.8 s，含人工外力，不改 command）。

在 `teleop_runtime/real_teleop/` 增加薄适配器：

- 从现有 `PositionForceGuardMonitor.snapshot()` 读取，不调用 SDK；
- runtime Home 完成后建立 per-side tare generation；
- 在 60 Hz loop 中计算 requested offset/composed target，但 shadow 模式始终返回原 target；
- 通过有界 debug trace 记录 sample age、是否为新 sample、wrench frame、requested offset、
  clamp/freeze/reset reason 和计算耗时；
- 不把 trace 写盘放在控制热路径，不引入 GUI 或 dataset owner。

当前实现使用现有 `--debug-action` 有界后台队列，以可配置采样率记录
`position_admittance`；shadow callback 原样返回旧 InitArm filter 的返回对象，
因此保留历史 `None`/override 语义并保证目标 identity。未开启该 flag 时不会安装 callback。

InitArm 与 shadow modifier 通过一个组合回调接入现有 `target_override_filter`。顺序固定为
InitArm first；InitArm 返回 override 或 active 时，shadow 只能观察并 reset/bypass，不能
再次改写目标。这个组合逻辑必须有独立 corner-case 测试，不能直接覆盖现有 callback。

首轮证据：395/395 条 15 Hz debug frame 完整封口，内部 modifier 计数 1577 帧，
无 dropped frame/event、无 fault、无 stale/freeze；force sample age 的 p95 为左右
47.3/48.5 ms、最大 54.0/56.0 ms。主循环耗时 p50/p95/p99/max 为
5.00/6.11/6.54/6.74 ms。所有帧均满足 `shadow_only=true`、
`target_identity_preserved=true`、`applied_offset=[0,0,0]`，因此该次运行没有把假想
offset 写入 IK target。静止期 tare 后三轴合力范数 p95 左右为 0.333/0.304 N，
0.45 N 逐轴 deadband 后输出严格为零；人工外力窗口观察到左右最大 tare 后合力约
14.27/12.41 N，证明传感器与坐标变换链路确实响应了外力。

该次 shadow 同时发现原 C0 将 5 mm 和 10 mm/s 按三轴分别 clip，组合方向可达到
7.67 mm 和 17.32 mm/s。2026-08-06 已在纯数学模块内修成“分量上限 + 总平移范数
上限”，与旧 15→40 Hz pose-admittance 的线性范数合同一致；未改 force guard、IK、
碰撞、reference 或 dispatch。相关 targeted regression 为 141 passed。

验收：关闭或 shadow 开启时，最终 qref 与当前稳定 position profile 在确定性离线重放中
逐帧一致；60 Hz deadline、20 Hz force guard freshness 和 Ctrl+C/Arm_Stop 无回退。

### Slice C：单臂 active modifier（仍为独立 default-off profile）

状态：2026-08-06 已实现 C0 可回退代码路径并通过离线回归，默认关闭，尚未授权或执行
主动真机运动。C0 要求显式选择唯一 `active-side`，仅把该侧有界 translation offset
施加到拷贝的 pre-IK target；InitArm
override 保持对象 identity，Hold/Fix/inactive side 不叠加，Home tare reset 失败会请求 Hold。
开启后仍完整经过现有 IK、碰撞、joint limit、qref dynamics 与 120 Hz executor。

C0 参数参考已验证的 DMTAC_M 15→40 Hz pose impedance：`0.004 m/N` 对应等效
`250 N/m`。旧 profile 的 `lowpass_alpha=0.9` 实际运行在 40 Hz 控制周期，其等效连续
时间常数是约 `0.010857362 s`；先前写成 `0.022 s` 是把采样率和滤波权重语义混淆，
现已更正。主动包络仍收窄为
总平移范数 `5 mm` offset 与 `10 mm/s` velocity/slew，同时保留逐轴分量上限，不继承
原链路 70--80 mm 的大偏移范围。

#### 2026-08-06 离线动态 parity 结果

进一步审计确认，现有 DMTAC_M 40 Hz 行为不是二阶 M/B/K 积分器，而是：

```text
tare-corrected wrench
  -> 40 Hz low-pass（0.9 新样本权重）
  -> deadband（左 0.45 N，右 0.35 N）
  -> compliance target（0.004 m/N）
  -> active velocity / passive return velocity 限制
  -> 1.0 /s 离散回零
```

因此保留原 `second_order` 实现作为原子回退，同时新增显式
`profile_compliance` 响应模式。该模式：

- 先滤波再 deadband，与旧 `WrenchProcessor` 顺序一致；
- 将旧 40 Hz 离散滤波极点换算为连续时间常数，20 Hz 只消费一次新 wrench；
- 将旧 40 Hz 两个回零周期折算到一个 20 Hz 周期，避免采样率改变回零手感；
- 60 Hz 输出只做有界 slew，不把同一 20 Hz wrench 重复积分三次；
- compliance/return 动态与最终安全 envelope 分离，后者仍维持 5 mm、10 mm/s 的保守值。

合成阶跃、持续力和释放序列以旧 `WrenchProcessor + PoseAdmittance` 为 oracle：在共同的
50 ms 边界上，filtered wrench、deadband 后 wrench 和 requested offset 均达到逐点数值
一致；60 Hz 输出在一个 50 ms wrench 周期内形成三段有界斜坡。相关 targeted regression
目前为 `166 passed`。功能总开关仍默认关闭，`response_model` 默认仍是
`second_order`，只有显式选择 `profile_compliance` 才启用新控制律，因此不会改变现有
遥操、旧 impedance backend 或已保存的 shadow 基线。

#### 2026-08-06 V2 balanced shadow 复核与 active 状态闭环

随后采集的 baseline/shadow 均实际解析为
`rm65_bimanual_position_v2_balanced` 和 15/60/60/120 时钟。shadow 共 562 帧，左右分别
观察到 118/100 个 contact frame、193/189 个新 wrench sample，无 force stale/freeze、
send failure、runtime fault 或 IK budget exceed；目标 identity 与零 applied offset 也保持成立。

新 analyzer 在最终 60 Hz 输出边界加入 5 mm/10 mm/s 硬检查后，历史 shadow 只失败于
output envelope：左 8 帧、右 4 帧，最大分别为 5.0667/5.0393 mm。该运行是 shadow，
因此越界未进入 IK target。纯导纳模块现已在 `sample_output()` 最终边界重新施加总平移范数
clamp；旧日志会按预期失败，修复后的新日志必须零 violation 才能继续。

active 默认关闭路径同时补齐了两个离线状态合同：

- session command filter、Fix 和 qref admission 后的最终双臂 joint qref 通过只读 FK 转为
  solver-world tool-TCP pose；仅有实际 deadbanded force 的轴回写 anti-windup，零力轴的普通
  position 跟踪滞后不会重写 hand mapping；
- 接触轴释放时，只把最后 admitted transient offset 提交到平移 anchor，并把 transient
  state 置零，保证 composed target 连续，不回追接触前的绝对 VIVE 目标；FK 回执缺失会锁存
  modifier 并请求 Hold，只有新的 runtime Home/tare 才能解除。

本轮本地相关回归为 217 passed，加上完整双臂 IK/碰撞回归 110 passed；远端无设备回归为
45 passed，accepted-qref FK 的 focused 回归为 2 passed。下一次真机仍只运行 shadow：
左臂 live、右臂 Fix、`POSITION_FORMAL_V2_BALANCED_ARGS`，命令保存在
`scripts/temp_cmd/20260806_221054.md`。

尚未闭合的边界：

- 左右 DMTAC_M profile 的 deadband、active velocity、最大 offset 不同；下一次真机测试
  必须按 active side 显式取值，不能用一个双臂公共配置冒充完全 parity；
- 旧 40 Hz 的 `work` frame 与新路径的 `tool -> solver_world` 显式变换需要结合固定姿态和
  多姿态 shadow 验证方向一致性；不能只比较数值轴；
- 5 mm/10 mm/s 保守 envelope 会有意弱于旧 70--80 mm、50--60 mm/s profile，当前只
  声明“局部力-位移动态一致且更严格限幅”，不声明全幅手感已一致；
- accepted-qref FK anti-windup/contact rebase 已完成离线合同，但尚无主动真机证据；active
  仍不进入正式数采。

当前边界：active 状态闭环已有离线实现，但历史 shadow 暴露过最终 output envelope 越界；
必须先用修复后的“左 live + 右 Fix + V2 balanced”shadow 确认零 violation、力方向、tare
漂移和 sample age。C0 仍只作为受限研究开关，不进入正式数采 profile；正式 active 真机
命令需在新 shadow 通过后另行交付。

#### 2026-08-06 修复后 60 Hz shadow 结论与 Active 时钟修正

修复后的 V2 balanced shadow 共记录 876 帧、约 59.46 s。右臂 Fix 是否激活不作为本轮
60 Hz 导纳数值链的晋级门禁；本轮对左臂得到的直接证据为：

- 306 个新 wrench sample，sample age p50/p95/max 为 25.29/47.56/60.38 ms，未 stale、
  未 freeze；
- requested/output 总平移范数最大均为 5.000 mm，最终 output envelope 零 violation；
- 输出速度范数最大 10.000 mm/s，120 Hz stream 无 failed send/stale/fault，60 Hz 主循环
  p50/p95/p99/max 为 5.46/8.53/9.96/11.88 ms，无 IK budget exceed；
- 首尾 2 s 静止窗口逐轴 tared force 均保持在 0.45 N deadband 内，deadbanded force 与
  output 严格为零或接近零；
- 大于 1 N 的分轴窗口中，x/y/z 的力与输出同号分别为 107/108、173/173、206/206；
  唯一 x 反号帧发生在旧偏移未回零而外力刚换向的瞬态；
- 最后一段接触释放后，4.673 mm offset 单调回落，3.810 s 内低于 0.1 mm，结束时为
  0.005 mm，无大于 1 um 的反向增长。

该证据关闭了历史 5 mm 最终边界越界、freshness、静止 tare 漂移和被动释放稳定性问题。
它仍是 shadow 证据，不能替代 Active 的 admitted-qref FK、contact rebase 与真实接触卸载
验证。

进入 Active 前的逐周期审计又修正了两个只影响 Active feedback 的 60 Hz 时间/状态合同：

1. accepted-qref FK 回写不再使用早于 pre-IK modifier 的 `loop_start`，而在实际回写时读取
   monotonic time；纯导纳状态拒绝任何倒退的 realized-offset 时间，避免下一周期放大 slew
   `dt`；
2. `profile_compliance` 的 anti-windup 只把 60 Hz realized output 对齐 admitted FK，保留
   有界 requested force target，使一个 20 Hz wrench 周期内的后续两个 60 Hz duplicate-
   sample 帧仍能完成 slew；旧 `second_order` 继续使用完整状态 rebase 作为原子回退。

新增合同测试曾覆盖 backdated feedback 拒绝和 Active admitted feedback 后的 60 Hz 连续
slew。这里的“Active 尚未执行”和晋级判断只记录事故前的历史状态，已被下面的真机事故证据
推翻，不能再作为运行依据。

#### 2026-08-06 Active 事故与 2026-08-07 C0 离线修复

首次左臂 Active 在约 0.526 N 的轻微 deadbanded force 下触发了持续错误运动。2 mm
`output_offset` 限幅本身生效，但事故前已有约 140.1 mm 的普通 base target 到 admitted-qref
FK 残差；Active feedback 把这个绝对残差当成“未实现导纳”写入无总包络的持久 anchor。
最终记录到的总 modifier 达 417.3 mm，左臂 q2/q3/q5 qref 分别累计约
+61.2/-125.7/+64.5 deg。force、stale、collision、jump 和 per-step dynamics guard 均未触发，
因为它们不拥有 modifier 的累计范围。进程最终以 `reason=finally` 退出，不是自动安全停止。

事故后的 C0 合同改为：

- 删除 admitted-qref FK 到 anchor/controller 的状态回写；FK 只记录绝对残差和 modifier
  tracking residual，不改变控制状态；
- 删除 contact-release anchor commit；释放只通过有界 `profile_compliance` transient 按
  `max_return_velocity` 回零；
- pre-IK 最终 modifier 只能等于 bounded transient，持久 offset 恒为零；在应用前独立复核
  component、总平移范数、component slew 和总平移 slew，任何异常拒绝当前帧并进入需要新
  Home 的 Hold，不再静默退回可能很远的 base target；
- 非零 modifier 还必须使用上一周期新鲜 admitted-qref FK 证明 base target 已对齐：总平移
  residual 不超过当前 max offset，姿态 residual 不超过 3 deg；缺失、stale 或超限时在
  pre-IK 直接拒绝，防止微小overlay把 `current_pose_hold` 重新激活到远距离raw target；
  同时复核最终 `raw target + modifier` 到 admitted-qref 的平移残差不超过“nominal
  limit + 当前 modifier 范数”的三角上界；nominal residual 和 modifier 仍各自独立限制为
  2 mm，避免把普通约0.3 mm qref滞后与合法2 mm modifier误判为事故，也不允许隐藏的第三项；
- schema 升为 `position-admittance-v4`，明确
  `active_feedback_policy=diagnostic_only_no_state_update`；
- 新增事故型闭环回归：140 mm nominal target-to-admitted 残差必须在首次非零overlay前
  fail-closed；对齐后的20 Hz wrench/60 Hz output、接触/释放切换和长时运行逐 tick 断言
  总 modifier 不超过2 mm、slew不超过5 mm/s、anchor始终为零；另有姿态基线、总包络与
  slew越界的主动fail-closed测试。

2026-08-07 又闭合了两道无设备门禁：

- 对保存的234帧事故日志执行15 Hz采样证据重放。第一次非零 transient 出现在
  `step_index=656`，上一 admitted-qref 到当前 raw target 的残差为139.932 mm，而允许值为
  2 mm；新 translation-baseline gate 会在同一帧、旧 anchor 首次造成总包络越界之前拒绝。
  重放还复核到旧 anchor/总 modifier 最大值分别为417.410/417.304 mm；
- 运行630个60 Hz tick 的确定性闭环，wrench 以20 Hz更新，admitted pose 以3 mm/s受限
  跟随，包含正/负接触、重复sample和长时间释放。每个tick的最终 modifier 均不超过2 mm、
  slew不超过5 mm/s、anchor恒为零，admitted pose也保持在2 mm内；6 s释放后的 modifier
  与滞后pose均低于10 um。该测试是位姿接纳滞后模型，不冒充真实RM65/MuJoCo接触证据。

随后把保存的受力数据直接送入当前C0核心，而不是等待新的在线采集。输入使用日志中的
`filtered_force_control_n`，即已经完成tool到solver-world变换和运行时低通后的受力；以60 Hz
重采样，统一施加2 mm/5 mm/s包络，并追加6 s零力释放：

- 59.464 s良性shadow轨迹含876个记录受力点、重放3932 tick，输入力最大21.085 N；当前
  C0输出最大2.000 mm、requested velocity和output slew最大5.000 mm/s，零freeze、零包络
  violation，释放尾差0.0049 um；
- 15.872 s事故轨迹含234个记录受力点、重放1315 tick，输入力最大2.885 N；删除anchor后
  当前C0输出最大仅1.802 mm、slew最大5.000 mm/s，零freeze、零包络violation，释放尾差
  2.566 um。这直接证明旧417 mm持续运动不能由该受力序列和bounded transient产生；
- 力换向后旧位移尚未过零的最长短暂反号分别为83.3/133.3 ms，均小于由2 mm/5 mm/s
  推导的416.7 ms最坏清零边界；所有大于1 N的分轴反号点都位于受力换向后的200 ms内。

证据等级是 **offline sampled control-force replay**：action-debug只有15 Hz快照，并非完整
20 Hz producer流；它也不重新计算原始tool-frame变换，不能证明IK、接触物理或硬件运动。
因此它足以把“受力响应/回零/总包络”从下一次在线门禁中前移并关闭，但不能把Active解冻。

#### 2026-08-07 shared RM65 IK 与 60 Hz qref 闭环反事实

新增纯离线回放器把保存的受力序列送入当前C0，再使用正式
`rm65_bimanual_position_v2_balanced` 的共享双臂模型、DMTAC_M左右TCP、同一IK/碰撞/
关节步进与post-IK shaping，分别执行：

1. 以每条历史accepted qref为种子的teacher-forced baseline/candidate单步反事实；
2. 60 Hz ZOH raw target连续闭环；
3. 60 Hz线性重建raw target连续闭环；
4. 6 s零力释放尾段。

回放同时补出一个此前被离线对比遗漏的控制语义：操作者手静止时，tracker会发布
`INACTIVE/hold_weight=1`，这会让求解器的inactive-side hold吞掉已经授权的力控target变化。
现在只有Active、live side且最终modifier实际非零时，modifier才通过通用
`pose_action_motion_intent`声明该侧有运动意图；shadow不发布，Fix权威仍优先，普通position
路径不变。为了让下一份shadow不再依赖平移差值猜测，action-debug也新增完整的
`raw/would_apply/applied_target_pose_solver_world`。

良性876帧shadow日志的结果：

- 774个非零candidate全部通过pre-IK组合门控，并全部完成共享RM65 IK；
- teacher-forced candidate相对baseline的左臂qref增量最大0.0201 deg、TCP平移最大
  0.1053 mm，对侧增量为零，无jump/collision状态回退；
- ZOH与线性闭环均完成3933/3933 tick，最终modifier最大2.000 mm；左臂相对baseline的
  TCP增量最大分别1.4702/1.4673 mm，旋转增量最大0.1800/0.1749 deg，对侧为零；
- 正式关节步进上限利用率最大仅0.00307/0.00300；所有7866个per-side tick均为
  `tracking`；6 s释放后candidate与baseline的TCP差分别0.7068/0.7613 mm，已进入既有
  1 mm `current_pose_hold`平移容差。纯C0 modifier自身的释放尾差仍为0.0049 um，因此未为
  该求解器容差增加新的release-pending状态。

234帧事故日志的结果：

- 70个非零candidate中69个因nominal target-to-recorded-qref残差超2 mm而在IK前拒绝；
  历史后段残差p95/max为276.9/278.2 mm；唯一eligible帧未产生qref/TCP增量；
- ZOH与线性连续反事实均完成1316/1316 tick，力控candidate相对同一raw-target baseline的
  TCP增量最大0.8544/0.8547 mm，对侧为零，无模型错误。这里baseline与candidate都会跟随
  历史raw target，因此该连续模式只证明“删除anchor后力控增量保持局部”，不能替代上述
  recorded-qref门控证据，也不声称复现事故中的qref冻结。

旧日志没有新加的完整post-smoother pose，只能使用source pose rotation proxy；受力也是
15 Hz采样后的已变换/已滤波值。至此离线数据已覆盖最终C0包络、释放、pre-IK组合门控、
共享IK、碰撞/跳变状态、正式关节步进以及60 Hz ZOH/线性qref闭环。尚需真机shadow才能
获得完整20 Hz原始tool-frame流、新增精确pose字段、真实调度时序和真实接触方向证据。
同一日志的右侧输出全程为零，只能作为fixed/no-op一致性证据，不能冒充右侧force-to-IK
覆盖；回放器现已把“至少一个非零candidate进入共享IK”设为独立覆盖门禁。

#### 2026-08-07 生产 Active observer 与 shared IK 同帧闭环

继续按真实运行顺序把生产 Active observer、正式 V2 balanced shared IK、最终 joint qref
接纳反馈串在一起后，首次非零受力被 fail-closed 门禁拒绝，报告 nominal target-to-admitted
残差约140 mm。该残差不是机械臂真实跳变，而是反馈 frame 混用：pre-IK modifier 作用于
`solver_world` target，接纳反馈却使用了 `tool_tcp_world` FK；当前 DMTAC tool transform
自身就超过100 mm，因此这条 Active 链路即使不失控，也无法正常进入闭环。

修复后，最终 joint command 统一投影回 `solver_world`，并在 receipt 中显式记录组合式
`p_applied = p_raw + output_offset`、无 persistent term、最终2 mm与5 mm/s包络。新增测试还
故意断言 tool-TCP 与 solver-frame 平移差大于100 mm，保证旧实现会失败；再以20 Hz虚构
tool-frame受力驱动720个60 Hz tick，确认左臂实际产生非Hold qref运动、右臂保持、正式关节
步进不变、6 s释放回到1 mm内，且anchor和admission-driven state update始终为零。

另新增原始受力生产链回放器：从action-debug读取`raw_wrench_n_nm`并由首帧raw/tared差值
恢复tare，经过当前tool到solver变换、生产Active observer、共享IK、正式接纳qref FK反馈，
同时跑60 Hz ZOH和线性target重建，再追加10 s、20 Hz新鲜零力释放。normal shadow保留
80 mm/60 mm/s参数只用于15--40 Hz兼容性证据；生产Active安装器现独立拒绝超过2 mm/
5 mm/s的配置，回放也在不改变filter/compliance动态的前提下收紧到该谨慎包络。

良性normal受力日志在谨慎Active包络下两种重建均通过：1254/1254 tick完成，最大modifier
严格为2.000 mm，最大slew利用率约1.0，正式关节步进利用率低于0.0031，分别有84/81个
真实Active qref运动tick；释放后modifier低于0.1 um，raw nominal残差低于0.373 mm，
旋转残差低于0.074 deg，对侧最终qref变化为零。谨慎日志虽然有1405个非零modifier tick，
但实际qref运动tick为零，因此被新的覆盖门禁正确判为“激励不足”，不能冒充Active行为验证。

这一级证据仍是离线命令闭环：记录的raw wrench与target只有15 Hz action-debug快照，零力尾
段是虚构的20 Hz新鲜样本，admitted qref是命令回执而非机械臂实测pose，也没有执行真实
force guard/Arm_Stop。因此它关闭了frame、生产observer、共享IK、接纳反馈和释放门禁，
但不提供接触物理、20 Hz完整producer时序或硬件停止证明。

当前状态仍是 **Active rejected / 不得真机重跑**。完整相邻回归、采样日志重放和确定性
位姿接纳闭环、两条历史受力轨迹的当前C0重放已经通过。在线shadow不再是检查C0响应包络
的唯一手段；15 Hz采样点上的原始frame变换和真实shared IK也已覆盖。剩余未被历史数据
覆盖的是完整20 Hz producer流与调度、真实接触方向、实测机械臂pose以及force guard/
Arm_Stop停止链。在这些剩余边界关闭并再次审计前，不开放Active，也不生成Active真机命令。

只在 Slice B 证据通过后开放，且最初限制为一侧 live、另一侧 Fix、formal collection 关闭：

- 将有界平移 offset 应用于 pre-IK target；
- 在 pre-IK 最终组合边界独立检查完整 modifier 的总幅值和slew；
- 完整通过既有 IK、collision、joint-limit、动态约束和 admission；
- accepted qref 的 FK 只读观测，不回写 anchor、integrator 或 controller state；
- C0 不拥有 contact rebase；释放由同一个有界 transient 状态按 return velocity 回零；
- stale、mode transition、Home generation 变化均冻结或清零，不允许瞬时 snap-back。

参数不从原生 float 试验的 3--9 N 响应反推。先用 shadow trace 标定噪声、静态线缆预载、
人手输入与接触事件，再确定很小的 active envelope。

### Slice D：双臂与正式数据合同

最后才扩展到双臂和 formal collection，并补齐 raw/tared/filtered wrench、requested/admitted
offset、composed target、accepted qref/FK action、tare generation 和 contact/rebase 状态。
数据 action 仍以 accepted qref 的 FK 为准。

### 原子回退合同

- 功能默认关闭；不带 admittance profile/flag 时必须走当前稳定 position 路径。
- shadow 与 active profile 分离，不能用一个模糊 bool 同时表示“计算”和“应用”。
- 删除/停用组合器中的 admittance stage 即回到现有 InitArm-only callback 行为。
- 不修改已有 impedance backend、原生 force-position probe、force guard 阈值、碰撞检查、
  IK profile 或 120 Hz transport。
- 每个切片独立提交/验证；Slice A/B 不具备运动权限，Slice C 需重新经过明确真机授权。

### Phase 0：离线合同与 shadow 数学

- 将纯 wrench filter/admittance 数学放到拥有共享运动语义的层，避免从硬件 backend
  复制实现；hardware 仍只负责 SDK 生命周期和采样。
- 使用已有数据或新增只读 trace 重放 raw VIVE、wrench、qref，计算但不应用 offset。
- 校验力方向、frame、tare drift、噪声谱、stale 分布以及潜在 qref/IK 变化。

验收：未触发时输出严格为零；正常运动无明显误触发；不能造成 branch/joint-limit、
collision 或 dynamics 指标回退。

### Phase 1：真机 shadow（不影响 command）

- 运行现有 position teleop；modifier 只 shadow 计算。
- 验证单一 wrench producer 对 120 Hz CANFD、60 Hz IK 和 force guard deadline 无影响。
- 对比 raw/tared/filtered force 与预期卸载方向。

验收：控制周期与 force guard 无回退，无 stale 风暴，无重复 SDK reader 竞争。

### Phase 2：单臂低风险闭环

- 另一臂 fix，关闭正式 collection；
- translation-only、小 offset、低速度、低 compliance；
- 覆盖无接触、轻推、持续推压、释放、stale、Hold、人工调整与 Arm_Stop。

验收：接触后单调卸载，无振荡、追赶、跳变；硬阈值仍独立触发；恢复不产生旧 offset。

### Phase 3：双臂与数采

- per-side state；覆盖双臂、左右单臂、Hold 和快速切换；
- 完成 action/wrench/compliance lineage 和 dataset validator；
- 再建立 default-off 正式 profile。

### Phase 4：控制器侧力位混合 A/B

- 先做无运动/极小位移、单轴、单臂的专用 probe；
- 逐级测试 10/20/40/60/120 Hz 与 controller acknowledgement/deadline；
- 对比主机导纳在接触响应、realized qpos 偏差、停止可靠性和数据合同上的收益。

只有控制器侧方案显著更好且所有失败语义闭合，才考虑将其纳入正式 profile。

当前状态：隔离 capability probe 已完成，默认 M/B/K 下未达到接入标准。Phase 4 暂停，
不继续扫描人工力和速度上限；后续资源优先投入 Phase 0--2 的主机侧有界导纳。

## 建议决策

1. 把目标表述为“为 60 Hz position teleop 增加可回退的接触柔顺”，而非直接把
   `impedance_40hz` 改成 60 Hz。
2. 首版采用单一 wrench producer + 低带宽 translation-only 有界 transient，不引入
   admitted-qref/realized-pose anchor 或 contact rebase。
3. 保留现有 force guard、碰撞、IK、qref planner 与 120 Hz transport 的权威性。
4. RM65 原生力位混合保留为隔离 A/B 证据；默认 M/B/K 已验证不接入正式遥操，只有
   新的单参数 M/B/K 试验显著改善后才重新评审。
5. 整个功能 default-off，并保留一项配置即可回到当前稳定 position profile 的原子回退。
