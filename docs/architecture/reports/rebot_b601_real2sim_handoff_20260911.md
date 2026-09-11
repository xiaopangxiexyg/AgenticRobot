# B601-RS Real2Sim / HITL 开发交接文档

文档日期：2026-09-11
文档性质：开发交接与状态基线；不是硬件通过证书，也不是新的总路线图
对应候选任务：AR-20260911-b601-handoff-doc-r0
本候选状态：文档候选，提交后等待合并；未 push、未安装、未执行真机动作

## 0. 阅读顺序与使用方式

建议接手者按以下顺序阅读：

1. 先读“当前结论”和“不可违反的边界”，确认当前工作不是把 B601 单独做成一套新的 HITL 框架。
2. 再读“代码所有权”和“当前实现”，确定每个问题应在哪一层解决。
3. 读“证据与限制”，区分已验证的数学、仿真、软件行为、已观察的真机行为和仍未证明的动力学结论。
4. 最后按“下一步动作”和“Harness/Git 交接流程”继续工作。

本文件只描述当前工程状态和交接规则。既有总路线图
docs/architecture/plans/rebot_b601_rm65_real2sim_hitl_roadmap.md 是方向性契约，除非总目标发生明确变化，不应在日常进度中反复改写；阶段进度应写到独立的 progress/report 文档中。

---

## 1. 总目标：从 B601 真机到跨机器人 HITL

最终目标不是“提前学会控制 B601 的某个固定姿态”，而是建立一条可审计、可逐步替换真实设备的控制链：

    B601-RS real arm
          │  feedback / human motion / haptic path
          ▼
    B601 model + calibrated-ish controller/plant shadow
          │  Real2Sim mapping
          ▼
    RM65 MuJoCo simulation
          │  policy inference + same-cut human residual
          ▼
    RM65 real / B601 real, only after independent hardware acceptance

需要保持的四个方向性阶段是：

1. B601 real -> RM65 sim 遥操：B601 作为人工输入设备，使用 anchor-relative 的末端增量或受限 twist，经显式 frame/mapping 变换驱动 RM65 仿真。
2. RM65 policy -> B601 sim：policy 的 RM65 action/state 经过独立、有限、可审计的 cross-embodiment mapping，先作用于 B601 MuJoCo，不能把 RM65 的动作空间直接当成 B601 的关节命令。
3. 仿真中的 policy + 实时人工残差：每个 execution cut 读取 policy 当前动作，在末端执行前叠加同一 cut 的人工 residual；不改写整段 action chunk，也不增加第二个安全/dispatch owner。
4. B601 从臂 + RM65 主臂真机 HITL：只有仿真、mapping、策略残差、安全与两套硬件后端分别通过后，才进入跨真机闭环。

这四个阶段的共同原则是：B601 的运动通道和力反馈通道独立；policy 是基线，人工输入是有界 residual；安全、IK、参考生成和最终 dispatch 各自只有一个明确 owner。

---

## 2. 当前快照与结论

### 2.1 精确源码快照

| 组件 | 当前精确身份 | 说明 |
| --- | --- | --- |
| AgenticRobot development | 82e1facb250412219736c7c54ada017cb7c96e88 | 当前开发主线，clean；相对 origin/development 本地领先 11 个提交，尚未 push |
| AgenticRobot tree | 6641d723642dbd8c985981ecda80c48924df261a | 当前集成结果树身份 |
| MotorBridge main/development | 218a160b73a63eb7c48752bd5bc35206ce3a8bfb | 两个本地分支均指向该提交；tree 63f2f7ef0a304bc7c19f7181c9aac06cb693bcaa |
| reBotArm vendor submodule | 5ba28acef46237eb6a7560658bbc43b06cf8a259 | hardware/reBotArm_control_py 的固定 gitlink；不可把 vendor 源码复制进框架 |
| OpenPI project link | df9ed0fff25d31d4b94e5001b73cd21111061b0d | 当前集成证据中的 project link；使用前仍需按目标任务验证来源和语义 |

当前 canonical AgenticRobot development checkout 为
/home/zc/HandMotion/Developing/AgenticRobot-development-mainline。
管理 checkout /home/zc/HandMotion/Developing/AgenticRobot 有预先存在的 remote_results/ 未跟踪内容，不能因为本任务清理或重置；另一个 recovery worktree 也应保持原状。源码任务应在 Harness 专用 worktree 中完成。

本交接候选 worktree 保留了正确的 vendor gitlink，但没有自动检出
hardware/reBotArm_control_py 的工作树。这样可以避免文档任务隐式改变资产环境；不依赖 URDF/mesh 的 135 项相关回归已通过、14 项按资产条件跳过。需要 MuJoCo/Pinocchio 真实资产的检查，应由后续集成者在明确的集成环境中检出同一 gitlink 后重新运行，并把结果绑定到集成后的 exact SHA/tree。

### 2.2 当前结论

- B601 的 vendor 资产、Pinocchio 模型、FK/IK/Jacobian、MIT 参考/力矩数学、haptic wrench 变换、MuJoCo 机械臂场景、基础 position driver、effort/Real2Sim controller 和 protective-hold 语义已经进入当前 development 主线。
- MotorBridge 的 RobStride 参数回复关联能力已经在对应集成结果中通过源码测试和 clippy；但运行时仍须确认实际 Python binding/动态库是否来自新的构建，不能把“源码已集成”当作“现场安装已升级”。
- 真机已证明：CAN 连接、六轴扫描、参数读写、前台启动状态机、重力补偿/阻尼 protective hold 的基本行为可以运行；尚未证明 125 Hz 真机闭环、完整源时间戳新鲜度、绝对力矩标定、接触力尺度或最终 haptic passivity。
- Real2Sim 当前的 plant profile 是明确标注的 provisional profile，不是由独立负载、温度、双向 sweep 拟合出的真实动力学。它用于把控制接口和可重复仿真先跑通，不能用于宣称 sim-to-real fidelity。
- 当前工作的主线应从 M1 控制/时序/生命周期等价性进入 M2 成对真实数据与动力学辨识；跨 embodiment、policy 和最终 HITL 仍是后续阶段，不应被一个 B601 demo 代替。

---

## 3. 目标阶段与验收门槛

总路线图的阶段可按下面的工程含义理解。阶段之间有依赖，不应跳过证据层级。

| 阶段 | 内容 | 当前状态 | 完成条件 |
| --- | --- | --- | --- |
| M0 | 模型、坐标、关节/执行器 contract，冻结理想参考 | 基本完成 | vendor identity、URDF、frame、joint set、单位和零位约定可复现 |
| M1 | controller/timing/lifecycle 等价 | 正在收敛 | sim 中 qpos/qvel/qacc、ZOH、delay/dropout、startup/protective hold 与安全 receipt 有测试 |
| M2 | real paired capture 与 plant identification | 尚未完成 | command/feedback/send/receive 的可配对数据、标注、拟合/held-out 误差和不确定性报告 |
| M3 | contact/haptic/virtual fixture | 数学和仿真骨架已有 | wrench frame、J.T @ F、power invariant、延迟下限幅/限斜率/passivity 测试 |
| M4 | B601 real -> RM65 sim 遥操 | 尚未形成系统验收 | anchor-relative mapping、clutch/recenter、workspace/IK admission 与 sim demo 可重复 |
| M5 | RM65 policy -> B601 sim | 尚未完成 | 明确 action/state contract、独立 mapping、B601 sim 的安全 dispatch 和离线 replay |
| M6 | sim policy + same-cut human residual | 组合接口已有，系统闭环待做 | residual bounded、source epoch/TTL、同一 execution cut、policy chunk 不被污染 |
| M7 | B601 real + RM65 sim | 尚未完成 | B601 硬件验收通过且 real input 只进入已验证的 sim target |
| M8 | 两侧真机 HITL | 最后阶段 | 两侧硬件、力反馈、延迟、急停和端到端安全证据均独立通过 |

任何“仿真通过”“软件测试通过”只能关闭相应层级的门槛，不能自动提升到硬件或系统级 HARDWARE_PASS。

---

## 4. 代码边界与所有权

### 4.1 目录职责

| 目录/组件 | 负责什么 | 明确不负责什么 |
| --- | --- | --- |
| hardware/rebot_b601_rs/ | 有线 SocketCAN/MotorBridge 后端、反馈 admission、状态/receipt、使能/停止、能力门控 | policy、cross-embodiment mapping、HITL arbitration、通用 IK |
| robot_motion/rebot_b601_rs/ | Pinocchio 模型、FK/IK/Jacobian、MIT reference/torque breakdown、haptic 数学 | 直接打开 CAN、选择 policy、决定系统 authority |
| sim/mujoco/rebot_b601_rs/ | B601 MJCF/URDF 载入、ideal/effort/Real2Sim driver、startup/protective hold、仿真 fixtures | 真机写入、真实动力学声明、RM65 业务策略 |
| control_composition/ | source facts、receiver cut、TTL、epoch、relative residual anchor、candidate lineage/evaluation | RobotAction canonical owner、IK、最终安全、硬件/sim dispatch、急停 |
| teleop_runtime/ / control station | 输入采集、policy 调度、目标选择、端到端执行编排 | 绕过 B601 backend 直接写电机 |
| RM65 sim/real target | RM65 私有 frame、IK、安全 envelope、dispatch | 接管 B601 的设备状态或全局 policy authority |
| hardware/reBotArm_control_py | 上游 vendor 代码和模型资产，作为固定 submodule 使用 | 本地业务逻辑、HITL 逻辑、框架代码复制 |
| MotorBridge | vendor transport/binding、参数请求回复关联、原始反馈读取 | 上层 gravity policy、cross-robot mapping、应用级安全状态机 |
| RPent | 后续 skill/任务语义接入 | 低层 B601 驱动、实时 haptic servo |

规范的单路径应保持为：

    source facts
      -> control_composition candidate
      -> RobotMotion proposal admission / authority
      -> target-private safety + IK + smoothing + limits
      -> one dispatch owner
      -> B601 wired backend or MuJoCo backend

不得因为增加一个实验模式而再复制一套 q -> torque -> send 主循环。若确实需要新行为，优先扩展已有 profile/driver contract，并给出独立的 receipt 和证据标签。

### 4.2 真机与仿真的边界

MujocoMasterBackend、B601RSBackend 和 RM65 target 都应实现同一层级的接口，但不应假装具有相同的物理可观测性。仿真可以提供完整 source timestamp、确定性 state 和 contact wrench；MotorBridge 当前只给出 vendor cache 语义，源时间戳/序列号仍未验证。因此上层必须保留 valid、frame_id、source_timestamp、receive_timestamp、seq（可用时）和 freshness/evidence 状态，不能将 sim 的理想时间字段冒充真机证据。

---

## 5. B601 模型、坐标系和 vendor 资产

### 5.1 固定资产与关节集合

hardware/reBotArm_control_py 当前固定到上游提交
5ba28acef46237eb6a7560658bbc43b06cf8a259。模型中包含 B601 的六个 arm revolute joints 以及 gripper 相关坐标；本 backend 的动力学、重力补偿、protective hold 和硬件验收只处理 joint1..joint6。J7/gripper 是独立通道，不能在“六轴完成”或“六轴 safe stop”中被误报为已处理。

### 5.2 内部 frame/单位规范

- 右手坐标系：X=forward, Y=left, Z=up。
- SI 单位：m、rad、N、N·m、s。
- transform 记号 {^A}T_B 表示 B frame expressed in A frame。
- 末端 pose 使用 base_link_T_gripper_end 的 SE(3) 表达，不在核心接口中散落 xyz_rpy。
- Pinocchio Jacobian 使用 LOCAL_WORLD_ALIGNED；线速度在前、角速度在后；奇异性指标要显式 characteristic length。
- 官方 Web/Three.js 资产可能是 Y-up；不能把该显示约定带进内部 ROS/Pinocchio/MuJoCo 核心。

FK/Jacobian 的正确性先用 Pinocchio 与 numerical test 验证，再用 MuJoCo 交叉检查；不要用“GUI 模型看起来正确”替代数值证据。

### 5.3 J2/J3/J4 的零位边界

J2/J3 的 [-5e-4, 0) 极小负读数可在反馈证据归一化中解释为 0 rad，以吸收已知的极微小编码器漂移。该规则不修改 URDF、不放宽 outgoing reference、不改变真实位置极限。J4 的精确 ±pi/2 边界、J4–J6 的只读余量和 outgoing command limit 仍按现有后端实现执行；不能用扩大阈值掩盖坏帧或越界。

---

## 6. 控制语义与重力补偿

### 6.1 关节 MIT 与重力通道

当前 B601 控制数学的概念形式是：

    tau_cmd = gravity(q)
            + Kp * (q_ref - q)
            + Kd * (dq_ref - dq)
            + tau_overlay

最终发送前还必须通过每关节 torque limit、velocity/acceleration abort、torque rise/slew 和 finite/reference validity admission。controller profile 负责产生有界参考；plant profile 负责描述 MuJoCo 的 damping、frictionloss、armature 等物理参数；不要把二者混成一组“真机参数”。

### 6.2 操作者确认的当前 gravity/damping profile

这组数值是当前的控制补偿 baseline，不是独立的真实动力学辨识结果：

    software Kp = 0
    software Kd = 0
    driver Kd (J1..J6) = 0.20, 0.30, 0.20, 0.15, 0.10, 0.05
    global gravity scale = 1.0
    gravity joint scale = 1, 0.95, 1, 1, 1, 1
    gravity joint offset = 0, 0, 0, 0, 0, 0 rad
    gravity joint bias = 0, -0.105, -0.1, -0.05, 0, 0 N.m
    Coulomb friction = 0, 0, 0, 0, 0, 0 N.m
    Coulomb velocity threshold = 0.02 rad/s

省略 CLI profile 参数时，默认采用这组补偿；若要实验原始 URDF 重力模型，必须显式恢复 scale=1、offset/bias/friction 全零和 driver-kd=0。不要在代码中悄悄替换默认 profile，也不要把操作者对手感的确认写成“参数已经辨识”。

driver-local damping 的所有权属于驱动器命令；软件的重力更新和 overlay 不能再叠加一个未声明的第二阻尼 owner。MuJoCo 中可以用等效 local damping 做 Real2Sim 对照，但要在结果中标明它是 shadow model。

### 6.3 Haptic wrench

Cartesian wrench 到关节的基本关系为 tau_h = J(q)^T F。不同 frame 之间必须用 spatial adjoint 的逆转置变换：

    F_target = Ad(T)^(-T) F_source

同时用 F.T @ V 做 power-invariance 检查。若力方向、力矩方向或旋转反馈异常，先检查 frame convention、adjoint 和 Jacobian，不能先调大增益。

---

## 7. 当前 Real2Sim 实现

### 7.1 已有组件

- sim/mujoco/rebot_b601_rs/scene.py：使用 pinned vendor URDF/mesh 构建 B601 MuJoCo scene，六轴 arm 加固定/open gripper；plant profile 在 MJCF 编译前应用。
- driver.py：简单高频 qpos/position driver，具备 host command validity、sequence/time stale 检查和 zero-order hold；仅仿真。
- mit_driver.py：effort driver，支持 ideal 与 opt-in Real2Sim controller、startup profile、gravity/local damping、protective hold、torque composition、clip/slew。
- real2sim.py：B601Real2SimPlantProfile、B601SimStartupProfile、B601Real2SimControllerProfile、probe trace audit、raw model baseline、operator gravity profile 和 unmodeled flags。
- demo.py、haptic_demo.py、axis_demo.py：仿真-only 的基本运动、虚拟 spring/wall、pose-6D 和方向验证。

当前模型/控制 profile 的重要设计是 opt-in：默认 ideal behavior 保持既有仿真调用兼容；显式启用 Real2Sim controller/plant profile 才引入 provisional damping、frictionloss、armature、host rate 或 startup lifecycle。这样不会影响已有 GUI/static model 或旧的仿真逻辑。

### 7.2 正确的仿真用途

当前仿真可以验证：

- qpos/qvel/qacc、FK/Jacobian、IK seed continuity 和 singularity admission；
- startup/protective-hold 的生命周期；
- host-ZOH、有限 rate、延迟/jitter/dropout 的控制逻辑；
- gravity + local damping、virtual spring/wall、wrench direction 和 torque saturation；
- policy action 经过 bounded residual overlay 后的执行链和 receipt lineage。

当前仿真不能单独证明：

- MotorBridge 真机是否有相同的 source timestamp、反馈新鲜度或 bus 原子性；
- 真实 RobStride friction、gear compliance、摩擦/温升/负载变化；
- contact wrench 的绝对数值和最终 haptic gain；
- 125 Hz 或更高的真机端到端闭环；
- 跨机器人 mapping 在真实 RM65/B601 上的成功率。

一次历史上“20 Hz fixture 失败、500 Hz fixture 通过”的结果只能说明对应 host loop/profile 的时序行为，不能推出“B601 电机只能 20 Hz”。真实设备的 vendor cache、CAN 事务和控制带宽必须分层测量。

### 7.3 仿真验证原则

仿真命令应从干净的候选/集成 worktree 前台执行，并把完整 command/output 写入固定日志；不在后台启动长时间仿真进程，不让 set -e 造成终端退出。若需要展示交互 GUI，GUI 只是观察层，验收仍应以 JSON、receipt、状态和数值检查为准。

---

## 8. 真机后端、安全状态机与已知异常

### 8.1 后端与能力门控

真机后端是有线 SocketCAN + MotorBridge 的 local wired path。激活式控制前必须确认 parameter_reply_correlation_v1 能力已经由当前运行时绑定验证；仅有历史源码通过或 CLI 存在不够。RobStride 参数读写中，motor/device id 与 host/feedback id 的语义不同：典型设备 ID 为 1..7，host responder 可能是 0xFE，参数回复 correlation 仍需使用当前 MotorBridge contract，不得自行把 responder 当 motor id。

MotorBridge 当前没有可靠的 source timestamp/sequence contract；现有 freshness_semantics 只能标为 vendor cache/unverified。顺序发送、部分发送失败和反馈快照完整性必须在 backend receipt 中显式保留。

### 8.2 启动与停止

六轴启动状态机为：

    ENABLED_UNARMED -> LOCKING -> LOCKED -> GRAVITY_RAMP -> HANDOFF -> ACTIVE

首个 lock frame 就应有重力参与，避免在重力爬升期间下坠。J1–J6 的完整、有限、边界合法反馈才可形成 safe reference；坏帧、缺轴、非有限值、突变 qvel/qacc 或 stale/wrong-frame 数据不得晋升为参考。

第一枚 Ctrl-C、普通异常和正常 duration 完成进入六轴 gravity+damping protective hold；第二枚 Ctrl-C、SIGUSR1，或已无法产生完整合法的六轴安全命令，才允许最终全臂 disable。J7/gripper 不属于这一 backend。物理 E-stop 始终是更高优先级。

protective hold 的核心语义是“保持最后一个完整合法且新鲜的参考，同时继续 live gravity+damping”，不是把所有电机立刻掉能。若进入 degraded hold，只能重复最后一次 verified gravity command，并提示保持物理支撑/E-stop ready；绝不能用错误 q、巨大浮点或越界值重新基准化。

### 8.3 stale/wrong-frame 契约

当前工程采用以下保守边界：

- wrong frame、invalid/non-finite 或被 admission 拒绝的当前指令：立即撤销该 overlay/candidate，不能沿用错误 frame 的内容。
- stale sequence、过期 source epoch、超过 TTL 的 candidate：不得进入新的 execution cut。
- 正常控制短暂丢帧时，后端可以在 watchdog 的明确上限内保持“上一合法参考/最后验证的 gravity command”，并持续报告 stale/degraded 状态；这不是无限保持，也不是自动重启。
- watchdog 到期、safe command 连续失败或 operator 第二次停止：执行全六轴 disable；不自动重新 enable。

任何改变上述语义的改动都必须同时更新状态机、receipt、测试和真机操作说明，不能只调整一个 timeout 常量。

### 8.4 J6 巨大 mechPos 的处理

曾观察到 J6 的 mechPos(0x7019) 返回类似 2231254908928.0、raw bits 0x5401e04a 的异常大值。该值超出当前反馈接受范围 [-3.24, 3.24]，必须被拒绝、隔离并记录 raw bits、motor id、feedback id、来源和原因；绝不能进入 gravity、IK、qpos hold 或 protective reference。再次出现时优先保留原始 type-17/parameter reply correlation 日志，排查 binding/帧解析/host-id 关联；不能用放大阈值“修复”。

### 8.5 已有真机观察与证据边界

已观察到并可作为定性证据的内容包括：

- CAN 接口恢复为 ERROR-ACTIVE、1 Mbit/s 后，motor scan 能找到 J1–J3 rs-06 和 J4–J7 rs-00；参数读写可返回 0x7019/0x701B 等值。
- 零位后 J1–J6 的静态 mechPos 可读；J7/gripper 的位置量级与六轴关节不同，不能混入六轴 qpos。
- 前台启动完整经过 ENABLED_UNARMED、LOCKING、LOCKED、GRAVITY_RAMP、HANDOFF、ACTIVE；普通 duration/第一次 Ctrl-C 后 protective hold 能继续输出 gravity+damping。
- 操作者确认无触碰、无外力支撑时，长时间 hold/replay 没有明显下坠或振荡；该结论是定性现场观察，不是完整力矩/频率标定。
- 一次 60 s active gravity/damping soak 有 1197 commands、约 19.936 Hz 的 compute/feedback/tx，零 deadline misses；长时间 protective hold 也完成安全停机。这个约 20 Hz 是当前顺序 vendor transaction/host loop 的观测值，不是电机内部可用控制带宽的结论。

历史证据文件和 SHA-256 见 docs/hardware/rebot_b601_rs_task_handoff.md、docs/hardware/rebot_b601_rs_haptic_acceptance.md 以及对应 /tmp/b601-rs-commissioning/ 目录。接手者必须保留原始 JSON、完整 command/output log 和 raw MotorBridge log；不能只复制“exit code 0”。

---

## 9. control-composition 与 HITL 合同

### 9.1 composition 的边界

control_composition 是 backend-neutral 的组合库，不是生产 dispatch 入口。它负责：

- immutable source facts、receiver execution cut、source epoch、TTL 和 role dependencies；
- policy candidate、human residual candidate 的选择与 lineage/evaluation；
- relative residual anchor/recenter 的事务语义；
- handoff/session 记录和同一 cut 的候选产出。

它不负责：canonical RobotAction 的最终定义、系统 authority/arbitration、cancellation、safety、IK、参考生成、设备/sim target 或物理 E-stop。

### 9.2 policy + human residual

目标控制形式是：

    u_exec = u_policy + alpha(t) * u_human_residual

但 alpha、residual envelope、slew/rate、frame 和 target safety 必须在最终执行 owner 中再次 admission。policy action chunk 不被整体替换或污染；每个 control tick 只把当前 policy action 与当前合法 residual 组合后提交，下一 tick 重新计算。

运动通道和 haptic 通道必须独立：

    master motion -> human residual -> policy + residual -> target safety/dispatch
    slave wrench  -> wrench/frame/Jacobian -> master haptic command

反力引起的 master 位移不能未经 observer/deadband/低通就被重新当成人类意图。

### 9.3 anchor-relative mapping

intervention/clutch 激活时记录 master 与 slave 的有效 anchor。后续使用：

    Delta_T_master = inverse(T_master_0) * T_master_t
    Delta_T_slave  = M(Delta_T_master)

而不是把两个机械臂绝对 pose 直接相等。clutch 时 human residual 置零，允许 master recenter，释放时事务性更新两侧 anchor；anchor 变更必须带 source epoch，旧 candidate 立即失效。

---

## 10. 证据阶梯与验收含义

| 层级 | 典型检查 | 可以证明 | 不能证明 |
| --- | --- | --- | --- |
| L0 静态源码 | gitlink、diff、类型、单元测试 | contract/边界没有明显回归 | 运行时设备可用 |
| L1 离线数学 | FK/IK/Jacobian、wrench、power invariant | 数学约定和数值实现 | vendor feedback/真实摩擦 |
| L2 MuJoCo | gravity、impedance、virtual wall、delay/dropout | 仿真控制结构和稳定性趋势 | 真机绝对力、带宽、温升 |
| L3 read-only hardware | scan、参数读取、passive raw CAN、完整反馈 | 总线、ID、帧解析和静态可读性 | 运动安全、力控和 source freshness |
| L4 bounded motion | 单关节/当前位置 hold/小幅 pose test | 命令链、限幅、停止语义和定性手感 | 系统级 haptic 或长时标定 |
| L5 paired identification | command/feedback 对齐、负载/温度/方向 sweep | provisional plant 的参数证据和 held-out 误差 | 未覆盖条件下的普适性 |
| L6 integrated hardware | real B601 + target + policy/HITL | 端到端目标在明确条件下通过 | 永久/无条件安全保证 |

当前已达到 L0/L1，MuJoCo 多数结构位于 L2；真机已得到部分 L3/L4 定性证据；L5 尚未完成；L6 不得宣称通过。

MotorBridge 集成结果的 94 项 workspace tests 和严格 cargo clippy --workspace --all-targets -- -D warnings 已通过；AgenticRobot 当前集成结果的 fast-feedback、motion-feedback、full-acceptance 三个 R0 profile 均通过。harness project validate 为 VALID，但仍会报告预期的 baseline/impact-rules 警告；警告需记录，不应被静默抹除。

---

## 11. 真机操作规则与建议测试顺序

### 11.1 操作规则

- Agent 不自行运行会使机械臂运动的命令；只提供前台命令、预期行为、停止条件和日志路径，由现场操作者运行。
- 任何 motion-capable 命令都必须有很短的终端 countdown（3、2、1、ENABLE），并保持前台；不得后台运行，以便现场立即 Ctrl-C/物理 E-stop。
- 完整 command/output 写入固定 log，JSON 单独保存；命令失败时终端仍保持可用，不用 set -e 让 shell 退出。
- 运行前确认 CAN 接口、vendor commit/config SHA、MotorBridge capability 和单一 CAN owner；运行中保持物理支撑和 E-stop ready。
- 先当前位置、低增益、低速度、低力矩；不要用大范围运动“证明代码正常”。J6 或末端小范围验证优先于基座关节大动作。
- 普通退出/第一次 Ctrl-C 进入 protective hold；只有操作者明确第二次 Ctrl-C/SIGUSR1 才做全臂失能。若反馈 malformed、速度/加速度越界或安全命令不完整，应立即停止该路径并按现场安全规程处置。

### 11.2 推荐 H1–H8 顺序

1. H1 read-only feedback：J1–J6 完整 mechPos、多轮有限性/范围检查，J7 单独记录；保存 raw frames。
2. H2 transport/safety：interface 状态、scan、capability probe、watchdog/急停语义；不发运动命令。
3. H3 单关节低增益：从当前 q 开始，优先 J6，再做其他单关节；限速度、限加速度、限力矩，先 hold 再极小 slow-step。
4. H4 gravity/damping：当前已确认 profile，当前位置静置；观察下坠、振荡、温升和 protective hold。
5. H5 virtual spring：先在 MuJoCo 验证方向/饱和，再在真机做小范围当前位置 overlay。
6. H6 virtual wall/pose impedance：仿真先完成平移和旋转分量；真机分别验证 translation 和 rotation，再组合 6D。
7. H7 timing/fault：前台注入 stale、坏帧、短暂丢反馈（只在安全条件下），确认保持、degraded、最终 disable 的顺序。
8. H8 integrated：只有 H1–H7 和 paired evidence 完整后，才接 policy、human residual、target mapping。

---

## 12. 当前未完成工作与优先级

### P0：运行时能力与证据闭环

1. 在受控、明确授权的环境中构建/选择包含 parameter_reply_correlation_v1 的 MotorBridge runtime，记录动态库/绑定来源；不要仅因源码合并就认为运行时已升级。
2. 先做 read-only capability/parameter correlation probe，再做任何 motion-capable 验收；保留精确 commit、config SHA、binding path 和 raw log。
3. 如果需要安装或替换运行时，必须单独获得安装授权；dexteleop310 仍视为只读环境。

### P1：M1 时序与 lifecycle

1. 将 host compute、feedback request、send、driver command 的时间线写成可比较 receipt；明确 vendor cache 与真实 source time 的差异。
2. 在 MuJoCo 中系统注入 0/10/20/40/80/120 ms delay、jitter 和 packet drop，观察 residual stability、contact overshoot、haptic energy。
3. 增加 qacc、same-frame baseline、final composed modifier envelope 的测试；组件级限幅不等于最终 action 已限幅。
4. 对 startup/protective hold、wrong-frame/stale overlay revoke、watchdog expiry 和 second-stop 做 deterministic replay。

### P2：M2 paired capture/identification

1. 在无人手、无负载、明确零位/姿态/温度条件下采集 paired command + feedback；每条记录包含 seq、source/receive timestamp、frame、valid、motor/host id、配置 SHA。
2. 覆盖正反方向、小/中速度、多个姿态和必要的静态 hold；如要拟合摩擦/重力，保留温度和负载标签。
3. 先独立拆分 gravity scale/offset/bias、driver damping、Coulomb/frictionloss，再给出 fit/held-out split 和误差；旧的 passive/read-only traces 只能做质量检查，不能冒充辨识数据。
4. 只有 held-out 误差、边界条件和不确定性报告齐全后，才更新 Real2Sim plant profile；更新后保留旧 profile 的可复现版本和回退路径。

### P3：M3/M4/M5/M6 系统闭环

1. 完成 contact wrench -> master haptic -> J.T @ F 的延迟、限幅、slew/passivity 测试。
2. 完成 B601 real -> RM65 sim 的 anchor/clutch/recenter 与 workspace/IK admission。
3. 明确 RM65 policy action/state 的 frame、单位、时间切片和 mapping；先离线 replay，再 B601 sim。
4. 将 policy + human residual 接入一个 execution cut，验证 policy chunk 不被污染、residual 不越界、stale candidate 不进入 dispatch。

---

## 13. 需要保留的开放决策

下列事项已经有保守默认，但若要改变会影响语义/验收，必须在实现前明确记录：

| 决策 | 当前默认 | 何时需要重新确认 |
| --- | --- | --- |
| wrong-frame | 当前 candidate/overlay 立即撤销 | 想改成沿用旧值时，必须同时重新审查安全和证据 |
| stale command | 新 cut 拒绝；watchdog 短窗口内可保持上一合法参考，超时全停 | 想延长 hold 或自动恢复时 |
| source timestamp | MotorBridge 未验证，标 unverified | 需要高带宽 haptic 或严格 freshness 时 |
| 真机 loop rate | 先按实际 transaction 测量，不把 20 Hz 当电机上限 | 需要宣称 125 Hz/高带宽时 |
| policy action | 现有 policy 主要是 RM65 action | 接入 B601 前必须冻结 mapping、frame、单位、chunk/cut |
| force scale/passivity | 先仿真验证结构，真机只做受限定性实验 | 需要接触或长时 haptic 时 |
| J7/gripper | 独立、非六轴 backend | 需要夹爪参与任务时单独建立 contract |

若这些决策尚未冻结，可以继续做不依赖它们的离线数学、仿真和只读证据；不要用临时默认值推进依赖它们的真机闭环。

---

## 14. Harness/Git 开发与集成规则

### 14.1 编码前

1. 读取 canonical Harness 配置、项目 AGENTS.md、task.env 和当前 worktree 状态。
2. 先执行 inspect，确认 exact SHA/tree、dirty entries、project/host binding。
3. 再执行 task conflicts，检查 exact path 与声明 scope；有重叠时先 coordinate，不得靠猜测写入。
4. 只在真实 writer-capable context claim writer lease；lease 释放后不得继续改 branch/tree。若 Harness 报 TASK_ENVIRONMENT_ATTENTION，先刷新 task.env，再重新 inspect/conflicts/claim，不能绕过门禁。

### 14.2 编码中

- 只写 task 声明的 worktree 和 write scope；共享 development、main、recovery/dirty worktree 不得顺手清理。
- 复用已有接口和依赖；不要把 vendor 文件复制到框架，不要引入大资产或生成文件。
- 安全代码改动要同时更新 receipt、状态机、测试和操作文档；不能只改变阈值或错误文案。
- 任何硬件动作都留给操作者前台执行；Agent 只做源码、离线、仿真和只读诊断。

### 14.3 提交与交接

在提交前重新检查 scope/conflicts、git diff --check、测试和 dirty 状态；一次任务至少形成一个 clean、可复现的 candidate commit。之后按顺序：

    recheck -> commit exact scope -> release writer lease
            -> quiesce/handoff-for-integration
            -> integration claim/prepare
            -> exact merge (不改写候选内容)
            -> fresh integration R0 profiles
            -> promote/close receipts

候选提交等待合并时，必须在交接和最终汇报中明确写“等待合并”；不能让合并者根据 ACTIVE/clean 状态猜测。集成通过不代表已经 push；push、安装、刷新 baseline、删除 worktree 都需要独立授权。

### 14.4 证据与 provenance

- 测试结果必须绑定 exact commit/tree；stale artifact 不能替代当前结果树的 fresh R0。
- 记录 source/config/binding SHA、完整命令、完整输出、JSON、raw log 和环境限制。
- exit 0 只说明进程返回成功；还要核对 observed behavior、receipt、sample count、deadline/freshness、cleanup 和 operator notes。
- CAN/AF_UNIX/socket 权限失败首先标为 execution-context failure；不能自动归咎于代码，也不能为了通过而放宽安全门槛。

---

## 15. 接手者的最短路线

1. 确认基线：在 canonical development 与本候选/集成树上记录 exact SHA/tree、submodule SHA、MotorBridge runtime 来源和 Harness 状态；不修改共享 dirty worktree。
2. 复现离线层：跑 B601 model/IK/Jacobian/haptics、MuJoCo Real2Sim 与现有 control-composition 边界测试；确认没有影响旧 GUI/static model。
3. 完成 M1 小切片：先写 deterministic timing/lifecycle replay 和 final envelope 测试，再扩展 delay/dropout；所有新增行为默认 opt-in。
4. 准备 M2 采集：先设计 paired record schema 和采集目录，不写入旧数据；得到现场授权后由操作者前台运行 read-only/静态采集命令。
5. 建立 RM65 sim target：只在 B601 Real2Sim 和 mapping contract 有离线证据后接入 B601 real -> RM65 sim，不把 B601 后端改造成上层 teleop/policy owner。
6. 最后才接 policy/HITL：冻结 action/state frame、cut、TTL、anchor 和 residual envelope，使用一个 canonical safety/dispatch owner。

如遇到模糊问题，优先提出会改变语义、数据含义、硬件安全或阶段依赖的问题；普通实现细节在既有边界内自行决定并留下可审计记录。

---

## 16. 相关文档索引

- 总路线图（方向契约，非日常进度）：docs/architecture/plans/rebot_b601_rm65_real2sim_hitl_roadmap.md
- Real2Sim 阶段进度：docs/architecture/reports/rebot_b601_real2sim_progress.md
- B601/HITL 拓扑与边界：docs/architecture/rebot_b601_hitl_topology.md
- B601 真机任务交接与 gravity profile：docs/hardware/rebot_b601_rs_task_handoff.md
- B601 haptic acceptance：docs/hardware/rebot_b601_rs_haptic_acceptance.md
- MuJoCo 使用约定：docs/sim/MUJOCO.md
- Git/Harness 工作流：docs/development/git-workflow.md
- control-composition 合同：control_composition/README.md

最终提醒：当前最重要的产出不是扩大真机运动范围，而是把 B601 的模型、反馈 admission、控制生命周期、时序证据和 provisional Real2Sim plant 变成可复现的底层能力；在此基础上再逐步接入 RM65 sim、policy 和 human residual。任何真机动作都必须保持当前位置起步、低增益、前台、可立即停止，并保留重力补偿 protective hold 回退。
