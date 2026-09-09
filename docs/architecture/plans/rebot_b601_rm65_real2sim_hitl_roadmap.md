# ReBot B601-RS 与 RM65 双向 Real2Sim / HITL 开发路线图

> 状态：长期共享规划，2026-09-09 首版。本文记录跨阶段目标、依赖顺序、所有权和验收门禁，
> 不声明未落地功能已经完成，也不授予真机使能、运动、合并、push 或安装权限。当前代码事实
> 以测试和对应实现为准；B601 真机候选的精确状态、参数和现场流程见
> [`../../hardware/rebot_b601_rs_task_handoff.md`](../../hardware/rebot_b601_rs_task_handoff.md)。

## 1. 总目标

系统最终要支持 ReBot B601-RS 与 RealMan RM65 之间的双向状态和动作传导，并在唯一的
policy/safety/dispatch 主链上加入低延迟人工实时干预。目标不是建立两套相互竞争的控制环，
而是按阶段验证下列闭环：

```text
ReBot real motion
    -> cross-embodiment mapping
    -> RM65 sim target

RM65 policy/state
    -> independently bounded reverse mapping
    -> ReBot sim shadow/target

policy emission + same-cut human residual
    -> one canonical safety/dispatch owner
    -> sim target
    -> real target after separate hardware acceptance
```

当前规划暂以 RM65 action space 作为 policy 的规范动作空间。ReBot 到 RM65 与 RM65 到
ReBot 是两个有各自 frame、anchor、scale、limit 和 freshness 契约的有向适配器；不得因为
两端都是六自由度机械臂就假定两个映射互为逆变换。

## 2. 长期不变量

以下原则跨所有阶段保持不变：

1. **唯一控制权。** `control_composition` 只组合 source 并产生 candidate；target runtime
   继续拥有 canonical safety、authority、dispatch、watchdog 和设备生命周期。human、policy、
   replay 或 shadow 不能各自建立第二套执行环。
2. **运动与触觉分离。** 人类运动 residual 是 target motion source；接触 wrench 到触觉力矩
   是反向 feedback channel。触觉导致的 master 位移不能未经区分再次作为人工动作意图输入。
3. **同一控制切面。** 对 action chunk 的人工干预只修改当前实际执行切面，不回写、污染或
   整段替换原 policy chunk；下一 tick 重新形成执行候选。
4. **相对而非绝对映射。** 跨机器人遥操默认使用 activation/clutch 时记录的 anchor-relative
   pose delta；两台机器人 base、workspace 和 joint morphology 不同，禁止直接令绝对 pose 相等。
5. **单一 torque owner。** 重力、位置/速度反馈、末端 `J.T @ wrench`、限幅和 slew 必须在
   一个明确的最终 torque composition 边界合成，不能在多个层重复补偿。
6. **模型与补偿分层。** URDF mass/COM/inertia、被动摩擦和电机等效惯量属于 plant；gravity
   scale/offset/bias、driver Kd、控制限幅属于 controller。运行时补偿不得反写为未经辨识的
   URDF 物理参数。
7. **多时钟显式化。** physics、驱动器本地环、host command、feedback、policy 与 human source
   使用各自 cadence、timestamp、sequence/freshness 和 ZOH/delay 语义。约 20 Hz 的 host 完整
   反馈观测不等于电机本地控制环只有 20 Hz。
8. **证据不越级。** offline、simulation、read-only hardware、motion-capable commissioning 和
   system-level hardware acceptance 分开记录；任一层 PASS 不自动证明下一层。
9. **失败关闭且保持可控。** wrong-frame、非有限值、过期或不完整 source 不进入组合；真机普通
   异常优先进入经验证的 gravity+damping protective hold，只有明确最终停止或无法生成完整安全
   命令时才全臂失能。
10. **J7 独立。** B601 的 J1-J6 是 arm 动力学和 protective hold 对象；J7 是夹爪，不得隐式
    混入六轴 FK、Jacobian、gravity 或整臂 receipt。

## 3. 当前基线（2026-09-09）

### 3.1 已有能力

- 厂商 `reBotArm_control_py` 以固定 gitlink 提供 B601-RS URDF、mesh 和参考实现；
  AgenticRobot adapter 不把 vendor 项目变成共享控制包的隐式依赖。
- `robot_motion.rebot_b601_rs` 已提供固定 J1-J6 顺序下的 Pinocchio FK、Jacobian、gravity、
  wrench/pose spring 和 MIT torque composition。
- `hardware.rebot_b601_rs` 已提供本地有线 MotorBridge/SocketCAN、反馈 admission、逐关节
  receipt、临时 torque-limit 事务和显式 safe stop。
- 当前真机候选具有 J1-J6 同步启动、driver-local locking、完整重力接管、全臂
  gravity+damping protective hold、反馈坏帧隔离和 calibration/replay 支持。
- `sim.mujoco.rebot_b601_rs` 已能从同一 URDF 构造 B601 ideal-effort 模型，并运行 gravity、
  virtual spring、virtual wall 和 6D pose impedance 的离线数学验收。
- RM65 已有可复用的 real2sim profile、delay/filter/force-limit 模式和成熟 MuJoCo target；
  `control_composition` 已有 backend-neutral 的相对 residual、source admission、handoff、lineage
  和 observability 语义。

### 3.2 当前 B601 重力补偿基线

当前操作者确认的运行时 profile 是：

```text
software Kp/Kd       = 0 / 0
driver Kd J1..J6     = 0.20, 0.30, 0.20, 0.15, 0.10, 0.05
global gravity scale = 1.0
joint gravity scale  = 1.0, 0.95, 1.0, 1.0, 1.0, 1.0
joint offset [rad]   = 0, 0, 0, 0, 0, 0
joint bias [N.m]     = 0, -0.105, -0.1, -0.05, 0, 0
Coulomb [N.m]        = 0, 0, 0, 0, 0, 0
```

它是 runtime compensation/手感基线，不是独立的 mass、COM、friction 或 payload 物理辨识。
现有 held-out fit 没有证明修改 URDF 物理参数能够稳定改善预测，因此 Real2Sim 第一版必须保留
原始 URDF plant，并把上述参数放在 controller profile。

### 3.3 当前错误兜底基线

- 启动按 `ENABLED_UNARMED -> LOCKING -> LOCKED -> GRAVITY_RAMP -> HANDOFF -> ACTIVE`
  推进；完整重力从首个锁帧参与，避免交接空窗。
- 反馈必须经过完整性、有限值、边界归一化、qpos、真实 `dt` 下的 qvel/qacc guard，才可更新
  `last_safe` 或生成发送命令。
- 单个有限异常帧被隔离；连续异常升级为 protective hold。拒绝帧不得成为新的安全参考。
- 第一次 Ctrl-C、普通异常或 duration 完成进入 J1-J6 全臂 gravity+damping protective hold；
  第二次 Ctrl-C/SIGUSR1 或 live/cache 都无法生成完整六轴命令时才全臂失能。
- MotorBridge 当前不能提供设备源 timestamp/sequence，因此 `freshness_verified=false`；接收时刻
  和轮询顺序不能冒充电机端采样 freshness。
- CAN 顺序发送可检测 partial receipt，但不是总线原子事务。这一剩余风险必须在真实系统验收中
  保留，不能由仿真 PASS 消除。

### 3.4 当前 Real2Sim 缺口

当前 B601 MuJoCo 是理想 generalized-effort 验收模型，不是已标定数字孪生：

- passive damping、friction loss、armature 和 joint stiffness 当前为零；
- 仿真 damping 与真机 driver-local Kd 的位置、频率和语义尚未对齐；
- 真机 gravity scale/offset/bias/Coulomb 尚未进入仿真 controller profile；
- command/feedback delay、host ZOH、量化、噪声、dropout 和坏帧尚未建模；
- B601 当前 scene 尚未形成真实接触场景，virtual wall 只是数学 fixture；
- 尚无同一初态、同一 reference/history 下的 real/sim paired replay 和 held-out 验证。

## 4. 阶段路线图

### M0：B601 Real2Sim 契约与理想基线冻结

**目标**

在不改变当前 ideal-effort 默认行为的前提下，建立一个版本化、可追溯的 B601 Real2Sim
profile 契约。

**交付物**

- 在现有 `sim/mujoco/rebot_b601_rs` 所有权内定义集中 profile；先避免创建第二套 simulator
  或生产入口。
- profile 明确区分 `plant`、`controller`、`observation/transport` 和 `evidence`。
- 每个参数标注 `identified`、`operator_validated`、`provisional` 或 `unmodeled`，并记录 URDF、
  config、vendor commit/tree 和数据 artifact identity。
- 将现有 spring/wall/pose-6d 结果固化为 ideal baseline regression，而不是 real-fidelity 证据。

**进入下一阶段的门禁**

- 当前 ideal profile 全部回归测试不变；
- 无隐式绝对路径、复制的 vendor mesh/URDF 或 RM65-specific 反向依赖；
- profile 缺少必要身份或出现非有限/越界参数时 fail closed。

### M1：控制器等价与多频率执行

**目标**

先复现已经有真机证据的控制器语义，而不是猜测未知 plant 参数。

**交付物**

- 仿真使用与真机一致的 gravity scale/offset/bias/Coulomb 公式；
- driver-local Kd 在 motor/physics cadence 上执行，software Kd 与 MuJoCo passive damping 分开；
- physics、local damping、host compute/command 与 feedback cadence 独立配置；host tick 之间明确
  ZOH，并记录 delay/jitter/dropout；
- 复用 torque absolute limit、torque slew、qpos/qvel/qacc guard；
- 离线复现 locking、gravity ramp、handoff 和 protective-hold 状态转换。

**门禁**

- 证明任一 damping/gravity term 只计算一次；
- 静态姿态不因切换 profile 产生未限幅 torque step；
- source stale、wrong-frame、非有限或丢帧时不推进 reference/last-safe；
- timing trace 能区分 physics tick、host command 和 feedback sample。

### M2：B601 plant 参数辨识与 Paired Replay

**目标**

使用真机安全采集的无接触数据辨识可观察参数，并用独立 held-out 数据判断是否采用。

**优先数据**

- 多姿态静态 gravity residual；
- 小范围、低速、低增益的单关节和全臂 free-drag/settling；
- startup replay、hold/release 和 protective-hold 切换；
- 逐关节 command/feedback cadence、delay 和稳定性统计；
- 明确负载、温度、配置、零点和操作者动作说明。

**候选参数**

- 电机/传动等效 armature；
- passive viscous damping；
- Coulomb friction 和 dead-zone；
- command/feedback delay 与低通带宽；
- 仅在有证据时才扩展 payload 或 link dynamics。

**门禁**

- fit 与 held-out 数据分离；
- 比原始 URDF/controller baseline 有稳定且有意义的 held-out 改善；
- 静态 drift、settling、q/qvel/qacc、torque/slew 和启动瞬态都有误差报告；
- 无法独立辨识的项保持 `unmodeled`，不通过多组互相补偿的参数制造表面拟合。

### M3：B601 MuJoCo 接触和触觉通道

**目标**

在已验证的无接触 plant/controller 上加入 MuJoCo physical contact，为后续 RealMan contact
到 ReBot haptic feedback 建立独立通道。

**交付物**

- 接入已有 scene/contact 机制，提供桌面、墙体和简单物体；
- 明确 self-collision filter、末端 contact site 和 wrench 表达 frame/origin；
- 分开报告 virtual-fixture wrench、MuJoCo contact wrench 和 estimated external wrench；
- 用完整 spatial adjoint 变换 wrench，并验证 `F.T @ V == tau.T @ dq` 功率不变量；
- force clip、slew/LPF、deadband 和延迟注入；必要时再引入 passivity observer/energy tank。

**门禁**

- 接触方向、frame、原点和功率不变量通过；
- 无接触时无伪接触力，接触断开后力命令按限制消退；
- 延迟/jitter/dropout sweep 下不产生未限制能量；
- 没有独立传感器证据前，不宣称仿真接触绝对力值等于真机手感。

### M4：ReBot 真机遥操 RM65 Sim

**目标**

以 ReBot 真机状态作为 human source，先驱动 RM65 MuJoCo target，完成第一条跨 embodiment
动作链；此阶段不需要 RM65 真机。

**交付物**

- ReBot state source 只发布经过 admission 的完整 J1-J6 state、FK pose、timestamp 和 validity；
- activation/clutch 时记录 ReBot master anchor 与 RM65 target anchor；
- 显式 frame transform、translation/rotation scale、workspace、singularity 和 speed/acceleration
  limits；
- ReBot pose delta 转换为 bounded RM65 Cartesian residual/reference，并提交给 RM65 target 的
  唯一 canonical safety/dispatch owner；
- 延迟、jitter、dropout、clutch/recenter、wrong-frame 和 stale-source 测试；
- 可选触觉通道只接收 RM65 sim contact wrench，不参与动作 authority。

**门禁**

- ReBot 不动时 RM65 sim 不漂移；
- clutch 期间 motion residual 为零，释放后无 pose jump；
- 单轴正负方向、旋转方向和尺度全部有可视化与数值断言；
- source 失效后 RM65 target 进入既有 HOLD/stop 语义，不保持未经复核的无限期旧参考；
- 既有 RM65 sim target、IK、canonical safety 和 dispatch 不被复制。

### M5：RM65 Policy 反向传导到 ReBot Sim

**目标**

把 RM65 canonical policy emission 及必要状态映射到 ReBot MuJoCo shadow/target，先验证
反向映射和 ReBot 动力学响应，不触发 ReBot 真机。

**交付物**

- 消费既有 immutable policy emission，不第二次 pop/replan action chunk；
- 独立的 RM65-to-ReBot anchor/frame/scale/limit 映射；
- policy target、ReBot sim accepted reference 和 measured q/pose 三层 trace；
- qpos/Cartesian 两种必要表示只在一个明确 adapter 边界转换；
- policy chunk 每 tick 推进，ReBot sim 按自己的 plant/transport profile 执行。

**门禁**

- 反向映射不改变原 policy emission；
- ReBot sim 不可达、奇异或越界时 fail closed，而不是裁剪后伪装完全执行；
- execution difference 可区分 policy、映射、canonical safety 和 plant dynamics 的误差；
- shadow PASS 不授予 ReBot 真机运动权限。

### M6：Policy + 人工实时干预，先在 Sim 闭环验收

**目标**

在 RM65 canonical action cut 上形成 `policy base + bounded human residual`，并先让所有 target
保持为仿真或 shadow。

**交付物**

- 显式 composition profile 才加载 `control_composition`；单源 policy/teleop 入口保持原行为；
- human、policy、replay 的 admission、eligibility、activation、handoff 和 lineage；
- `u_exec = u_policy + alpha * u_human`，再进入同一个 target safety/dispatch；
- intervention/clutch/recenter 状态机和 alpha/saturation/singularity gate；
- action chunk 仅在当前 execution cut 叠加 residual；
- synthetic expert residual、delay/dropout 和 policy-human conflict 评估。

**门禁**

- composition 不持有 target lifecycle 或第二个 dispatch token；
- policy-only profile 与当前单源部署行为等价；
- human residual 激活/释放无 discontinuity 或隐式旧 anchor 复用；
- stale/wrong-lineage human input 立即撤销参与资格；
- 日志可还原 policy candidate、human residual、composed candidate、accepted command 和 measured state。

### M7：ReBot 真机作为执行目标的受控验收

**目标**

在 M5/M6 的仿真和 shadow 证据稳定后，将同一 accepted ReBot reference 接到真实 B601 backend。
本阶段每次运动都需要单独现场授权。

**推进顺序**

1. read-only state/source 与 sim shadow 同步；
2. enabled gravity+damping、零 residual；
3. 当前姿态小范围 qpos impedance；
4. 当前 TCP 锚定的小范围 Cartesian/SE(3) impedance；
5. 单轴/单方向 policy reference，优先低风险末端关节；
6. 有界多轴 policy reference；
7. policy + human residual，最后才接入触觉反馈。

**门禁**

- exact installed commit/config/vendor pin 与离线候选一致；
- 真机命令前台运行，终端保留倒计时、PID、第一次/第二次停止语义和固定 log；
- 每次升级幅度前检查完整 cleanup receipt、速度/加速度/力矩/slew 和 protective hold；
- 普通退出不直接掉能造成 J2/J3 下坠；最终失能是明确操作；
- 任何意外运动立即冻结 Active path，保留命令、配置、日志和进程证据，禁止直接调参重试。

### M8：RealMan 与 ReBot 双真机 Policy/HITL

**目标**

在两台机械臂分别通过自身真机验收后，完成 RealMan 真机、ReBot 真机和 policy/human
composition 的系统闭环。

**交付物**

- 两端独立 preflight、watchdog、safe stop 和设备 lifecycle；
- 跨设备统一 monotonic timing/sequence、延迟预算和 stale policy；
- target 侧唯一 canonical safety/dispatch；另一台机械臂只能作为 source 或独立 haptic sink；
- 网络不是默认依赖；同机有线控制优先，跨进程只使用已有 typed transport；
- system-level emergency procedure 和操作者角色分工；
- 双真机 evidence 记录两端 exact identity、配置、时钟、receipt 和 intervention timeline。

**门禁**

- 任意一端 stale/fault 都不会在另一端继续放大旧动作；
- 单侧 safe stop 不被 composition/haptic 通道重新使能；
- delay、方向、workspace、力矩和能量限制有系统级验收；
- 只有双端现场结果才可标记 system-level `HARDWARE_PASS`。

## 5. 模块所有权

| 领域 | 应负责 | 不应负责 |
| --- | --- | --- |
| `hardware/rebot_b601_rs` | CAN/MotorBridge、反馈、mode、enable/disable、receipt、watchdog、最终设备安全 | policy、跨机器人 pose mapping、HITL blending |
| `robot_motion/rebot_b601_rs` | FK/Jacobian/gravity、wrench/pose 数学、torque composition 与 conditioning | 真机初始化、MuJoCo scene、policy lifecycle |
| `sim/mujoco/rebot_b601_rs` | B601 MuJoCo plant、actuation/timing profile、contact 和 sim state | 真实 CAN、RM65-specific policy、第二个 composition runtime |
| `sim.rm65` / `sim.mujoco` | 既有 RM65 scene、real2sim、target safety/dispatch | B601-specific 补偿或 vendor 参数 |
| `control_composition` | backend-neutral source admission、relative residual、handoff、lineage、observability | B601/RM65 特判、设备 lifecycle、最终 dispatch |
| `control_station` | 显式 source/target/profile 装配 | 重建 lower package 的 authority 或状态 |
| `teleop_runtime` | 单源真实 teleop/deploy/replay 应用入口 | 默认加载多源 composition、第二套 target executor |
| RPent / Agent Skill | 后续离散高层任务和语义操作 | 连续 residual、触觉 servo、真机 torque owner |

如果为了两个 embodiment 的映射出现可复用 contract，应先放到 backend-neutral motion/application
边界；在只有一个实际消费者前，不提前建立庞大的通用框架。

## 6. 统一状态、命令和证据字段

跨阶段样本至少携带：

```text
source_id / target_id
seq
source_timestamp
receive_timestamp
frame_id
representation
valid / stale / freshness_verified
operation_id / epoch / authority identity
config, model and profile identity
```

trace 必须能区分：

```text
raw source
admitted source
policy emission
human residual
composed candidate
canonical-safety accepted command
transported command
measured state
haptic wrench / haptic torque
```

没有这些层次时，不能将“机器人没到目标”直接归因于 policy、mapping、transport 或 plant。

## 7. 开发和验收顺序

每个里程碑遵循同一证据阶梯：

```text
pure contract/unit tests
    -> deterministic offline replay
    -> MuJoCo ideal baseline
    -> MuJoCo real2sim profile
    -> read-only hardware
    -> explicitly authorized foreground motion commissioning
    -> held-out/system acceptance
```

实现时优先选择最小完整切片。第一批应只覆盖 M0 和 M1：集中 profile、控制器等价、多时钟
执行和回归测试；不应同时引入接触、跨机器人 mapping、policy composition 和真机入口。

每批交接至少报告：

- 候选 commit/tree、merge base、声明路径和依赖 pin；
- 新增/改变的 authority、state owner、dispatch 和 lifecycle；
- offline/sim/read-only/motion-capable 证据分别是什么；
- 未识别参数、未测试工作空间和尚未关闭的安全风险；
- 是否仍在等待合并。若等待合并，必须显式写明，不能让接手者误以为应继续等待运行中的 Agent。

## 8. 当前开放决策

以下问题不阻塞 M0-M3 的 B601 Real2Sim，但必须在 M4 之前闭合：

1. 阶段 M6-M8 中“主臂/从臂”是严格指命令 source/target，还是任务机器人与触觉设备的产品
   角色名称；现场人工物理输入最终发生在 RealMan 还是 ReBot。
2. ReBot 到 RM65 的首个规范映射采用末端 pose/twist residual 还是 joint-space residual；当前
   建议优先 anchor-relative Cartesian pose/twist，并只在 target 侧做 RM65 IK。
3. RM65 policy 反向传导给 ReBot 时，ReBot 是纯 shadow、物理任务执行器还是 haptic follower；
   三者的安全、接触和数据语义不同，不能共用一个模糊 profile。
4. system-level stale/wrong-frame 行为应由 source admission 立即撤销资格，再由 target watchdog
   决定 HOLD/stop；不得用“继续保持上一合法动作直到无限期 watchdog”作为隐式默认。

这些决定闭合后，应更新本文而不是另建一份相互冲突的临时路线图。

开发批次、验证结果和剩余工作统一记录在 [B601 Real2Sim 开发进度](../reports/rebot_b601_real2sim_progress.md)。
