# robot_motion 定位与收敛规划评审（临时稿）

> 状态：架构评审草案，供后续拆分和实现前讨论。  
> 范围：robot_motion/、sim/、teleop_runtime/ 以及 replay/deploy 的运动控制边界。  
> 本次只新增本文档，没有修改控制代码、硬件接口或传输逻辑。

## 1. 结论先行

当前规划的主方向是合理的：robot_motion 应成为 sim/real/replay/deploy 共用的运动控制内核，real 和 sim 只在最终执行端分叉。当前代码实际上已经形成了这个方向的雏形，特别是：

- robot_motion/position_control/ 已经抽出了时间重采样、75 Hz 参考、150 Hz 关节命令插值和相关契约；
- robot_motion/rm65/position_collision.py 已经是 sim/real 中立的双臂位姿到关节参考规划器；
- robot_motion/replay/ 已经开始复用在线位置控制的时间表；
- sim/mujoco/ik.py 已经把 MuJoCo 侧的 IK 适配到共享 RM65 规划器，而不是在仿真器中另写一套 IK；
- teleop_runtime/deploy/README.md 和 robot_motion/deploy/README.md 已经表达了“策略动作 → 共享规划 → real/sim sink”的目标边界。

但目前仍有四个必须明确的修正：

1. **单臂 IK、双臂耦合 IK 和防碰撞规划不应排除在 robot_motion 之外。** 它们只要不依赖 VIVE、会话状态、硬件 SDK 和实际传输，就属于共享运动规划内核。通用接口可以放在 kinematics/planning，RM65 的实现放在 robot_motion/rm65。
2. **15 Hz 主 action 应定义为“运动意图/策略动作契约”，而不是整条控制链路的唯一接口。** 它要与 75 Hz 参考、150 Hz 位置命令、40 Hz 左右的力/阻抗控制命令分层。
3. **sim 的直接位控应被明确标记为 debug execution mode。** 它可以复用共享的 action、IK、碰撞、限制和时间调度，但不能被误认为已经实现了真实的动力学位置伺服链。
4. **当前 DualArmCollisionAwareRetargeter 和 real RM65 backend 仍然混合了多个职责。** 这不说明目标方向错了，而是说明需要先抽取共享规划层，再逐步瘦身 runtime/retargeter/backend。

因此，建议的目标不是“把所有东西都搬进 robot_motion”，而是：

> 将跨 sim/real 的语义、状态机前的纯运动规划和控制链路编排收敛到 robot_motion；将输入设备、校准会话、硬件/仿真执行、进程和安全壳留在各自的运行时目录。

## 2. 当前系统的真实形态

### 2.1 已存在的几条链路

| 链路 | 当前主要形态 | 应有的定位 |
|---|---|---|
| real 位控 | 15 Hz 动作/目标，经过位姿时间重采样、双臂耦合 IK/碰撞与限制，形成约 75 Hz 关节参考，再生成约 150 Hz qcmd，最后交给 RM65 实际发送 | 共享位置控制链，最终 sink 属于 teleop_runtime |
| real 力控/阻抗 | 位姿目标进入 RM65 的力/阻抗控制器，以约 30/40 Hz 运行，读取真实状态/力传感器并调用硬件控制接口 | 共享目标和控制协议可以抽象，真实控制器和 SDK 属于 teleop_runtime |
| sim 直接位控 | 直接使用高频 qpos 命令驱动 MuJoCo 场景，当前主要是 kinematic_direct_qpos_debug | 共享位置链的仿真 debug sink，不是动力学伺服器 |
| replay | 既有 action/pose replay，也有 qpos/轨迹 replay；已经开始复用 robot_motion.replay 和位置控制时间表 | 必须区分“重放动作后重新规划”和“重放已测关节轨迹” |
| deploy | 策略输出经过动作归一化、过滤、位置规划或力控后端，再进入 real/sim | 策略和启动壳可以不同，但控制链选择和运动语义应共用 |

这里最需要避免的是把这些链路都抽象成同一个“发送 command”。它们的反馈、频率、可验证性和安全含义不同：

- 位控链的核心输出是具有时间戳和有效性的关节位置参考/命令；
- 力控链的核心输出可能是笛卡尔目标、阻抗参数、力/位混合目标或厂商特定命令；
- 直接 qpos 是仿真中的执行方式，不代表机器人真实执行时已经经过位置伺服；
- replay 的输入可能已经是关节轨迹，也可能仍然是高层动作，二者不能在元数据上混为一谈。

### 2.2 当前已经合理的部分

以下几处设计基本符合目标方向：

1. robot_motion/position_control/config.py 将 action/sample、reference、IK 和 command 的频率拆开，而不是假定所有频率存在固定整数关系。15→75→150 和兼容的 15→45→150 都可以作为 profile。
2. robot_motion/position_control/resampling.py 和 interpolation.py 使用时间戳、因果策略和四元数 SLERP/关节插值，适合在线、回放和仿真共用。
3. robot_motion/position_control/contracts.py 已经把参考帧、命令 tick、hold/stale 等概念从具体发送器中抽离。
4. robot_motion/rm65/position_collision.py 已经明确：它接收高频参考位姿，负责耦合规划、关节限制、末端碰撞复核和 last-safe hold，不负责 150 Hz 发送。
5. MuJoCo 侧的 DirectQposDebugController 明确要求调用者提供完整的高频 qpos，且不在 sink 中重复 IK 或插值。这是正确的责任方向。
6. replay/deploy 文档已经把共享动作构造、轨迹契约和 15→75→150 调度集中到 robot_motion，而把 real backend、CLI 和设备生命周期留在 teleop_runtime。

## 3. 建议采用的三层边界

建议把“运动控制”拆成三个层次，而不是按 real/sim 复制两棵代码树。

### 第一层：语义契约层

描述机器人想做什么，不描述如何通过 UDP、CAN、MuJoCo API 或厂商 SDK 做到。

应包括：

- ActionIntent/RobotAction：动作空间、版本、时间戳、序列号、控制的 side/effector、参考坐标系、单位和有效性；
- MotionState/Observation：关节状态、末端状态、力/扭矩、时间戳、有效性和来源；
- PositionReference：经过重采样和规划后的原子双臂关节参考；
- ControlCommand：面向某种控制模式的最终控制语义，例如关节位置、笛卡尔阻抗目标；
- 版本、frame、tool profile、capability 和故障原因。

这一层是 sim、real、replay、deploy 共同依赖的公共语言。

### 第二层：执行无关的运动规划/控制层

只使用输入状态、目标、机器人模型和约束，生成确定、可测试的参考或控制输出。

应包括：

- 位姿/关节目标的时间重采样；
- 单臂 FK/IK 接口和 RM65 单臂 IK 实现；
- 双臂耦合 IK、姿态/关节限制、碰撞约束和 last-safe hold；
- 速度、步长、workspace、有限值、stale 输入等纯约束；
- 75 Hz 参考生成、75→150 插值和 preemption；
- 力/阻抗控制器的纯接口、目标整形、限制和诊断协议；
- replay/deploy 共用的时间调度和控制链 runner。

这一层应尽量是纯函数、显式状态对象或可注入的模型接口，不能直接调用硬件 SDK、读 VIVE、操作 MuJoCo scene。

### 第三层：运行时和最终执行端

负责把第二层的结果送到实际环境。

- teleop_runtime：VIVE/手部输入、校准、clutch、session、real-time worker、RM65 SDK/CANFD、真实力传感器、硬件状态和 real safety shell；
- sim：MuJoCo scene、physics stepping、camera、task/reward、仿真 observation、直接 qpos debug sink 和未来的理想/动力学 actuator sink；
- replay/deploy launcher：命令行、策略进程、数据读取、UI、日志和运行生命周期。具体归属可以按现有目录保留，但不能反向把这些 launcher 依赖塞入 robot_motion。

目标数据流可以概括为：

~~~text
teleop / policy / replay / task source
                │
                ▼
      ActionIntent / RobotAction
                │
        选择控制模式和 profile
          ┌─────┴─────────────┐
          │                   │
          ▼                   ▼
  position chain        force/impedance chain
  resample              target + measured state/force
  IK + limits            controller at 30/40 Hz
  collision + hold       force/impedance command
  qref 75 Hz             real force sink
  qcmd 150 Hz
          │
     ┌────┴────┐
     ▼         ▼
 real position sink   sim position sink
 teleop_runtime       MuJoCo direct/debug or actuator
~~~

反馈从 real/sim sink 返回到共享 planner/controller，但输入设备和进程生命周期不应成为 planner 的依赖。

## 4. robot_motion 应该负责什么

### 4.1 应明确归入 robot_motion

#### A. 运动语义和公共契约

当前 robot_motion/action、position_control/contracts.py、replay 中已有不少正确内容。建议最终由 robot_motion 统一维护：

- 动作空间和版本；
- pose、qpos、hand target 的 frame/单位/布局；
- 状态和反馈的最小公共字段；
- 位置参考、命令 tick、hold/stale/fault；
- 控制模式和能力声明；
- 规划结果的 status、error、collision metric、solver timing。

#### B. 单臂 IK

单臂 IK 不属于 VIVE retargeter，也不应长期属于真实 RM65 backend。只要求解本身不发送运动、不读取 SDK 状态，它就是共享运动规划能力：

- 通用层定义 SingleArmIKSolver、输入/输出和诊断；
- RM65 的模型、tool profile、solver adapter 放在 robot_motion/rm65；
- rm65_inspire_tac.ik.SingleArmIKSolver 可以作为底层求解实现，但通过纯适配器接入；
- 真实 backend 最终只接收已经生成的 joint target；
- 如果某条旧兼容路径仍需要 backend-side IK，应明确标为 legacy/compatibility，并设置迁移边界。

#### C. 双臂 IK、防碰撞和原子规划

双臂防碰撞 IK 更应该属于共享内核，因为它的价值正是保证 sim/real/replay/deploy 使用同一套约束语义。它应：

- 同时接收双臂目标和两侧当前状态；
- 对被动侧也保留完整状态，而不是把两侧拆成两个独立 backend；
- 一次性输出两侧的 qpos reference 和统一的 result status；
- 统一处理关节边界、步长、workspace、碰撞 barrier、末端复核和 last-safe hold；
- 把 collision scene/profile/version 作为显式输入或元数据；
- 不能隐式做 transport、硬件 enable、清错或动作发送。

当前 RM65PositionCollisionPlanner 的方向已经符合这一要求。需要保留并显式记录的限制是：当前主要是末端/终点碰撞复核，150 Hz 插值段的连续碰撞还没有被认证。不能因为 endpoint recheck 存在，就把整条轨迹宣称为连续无碰撞。

#### D. 位控链的时间和参考生成

建议把下列对象视作共享控制链，而不是 real 专属：

~~~text
ActionIntent
  → causal timestamp resampling
  → 75 Hz pose/qpos reference
  → single/coupled IK
  → limits/collision/hold
  → atomic PositionReferenceFrame
  → 75→150 qpos interpolation
  → PositionCommandTick
  → final sink
~~~

PositionReferenceFrame、PositionCommandTick 和调度 profile 应继续保持 sim/real/replay 一致；只有最后的 sink 不同。

#### E. 力/阻抗链的控制协议，而非硬件实现

力控可以在 robot_motion 定义：

- 力/阻抗控制器的输入状态、目标、参数和输出协议；
- 控制模式 capability；
- 目标限幅、stale/hold、故障状态和诊断；
- 采样和控制频率的时间语义；
- 可替换的纯控制器或 controller interface。

但以下内容不应放入 robot_motion：

- RM65 厂商 force-position API；
- CANFD/串口/SDK 线程；
- 真实力传感器 tare、硬件模式切换和 enable；
- 某台设备的 transport retry 或清错。

换言之，robot_motion 可以定义“一个力控链需要什么以及返回什么”，但不应假装自己已经拥有硬件力控执行能力。

## 5. robot_motion 不应负责什么

以下内容即使与运动结果有关，也不应该成为共享运动内核的职责：

| 内容 | 建议归属 | 原因 |
|---|---|---|
| VIVE/OpenVR 设备发现、tracker 绑定、UDP 接收 | teleop_runtime 的 input/source | 是输入设备协议，不是机器人运动语义 |
| tracker 字符串排序、左右映射、clutch、桌面标定 | teleop_runtime | 是会话和输入重定向策略 |
| home recalibration、输入丢失、session hold、录制 | teleop_runtime | 依赖操作者和运行生命周期 |
| RM65 SDK、CANFD、厂商 Movej/力控接口 | teleop_runtime | 是 real hardware sink |
| MuJoCo scene、physics step、camera、task reward | sim | 是仿真环境职责 |
| CubeLift 等任务专用 7D action | sim task 或 task/policy adapter | 不是通用机器人运动动作 |
| policy server、CLI、数据路径、日志进程 | deploy/replay/runtime shell | 不应污染纯运动规划 |
| 训练专用 observation/action tensor 布局 | dataset/task/deploy adapter | 可以映射成公共 RobotAction，但不应让核心绑定单一模型 |

这里有一个重要的边界判断：**上层 action 的“机器人运动语义”应在 robot_motion，但具体模型或任务的 tensor 格式不应全部归入 robot_motion。**

例如，当前 26D 的“左臂 pose7 + 右臂 pose7 + 双手 q6”如果是跨 teleop、replay、deploy、sim 共用的机器人动作，就应由 robot_motion 维护版本和语义；而 CubeLift 的 7D task action、某个策略网络的特殊归一化输出，则应在 task/policy adapter 中转换一次后再进入 RobotAction。

## 6. 15 Hz 主 action 的建议定义

### 6.1 15 Hz 是节拍，不是完整接口层

15 Hz 应记录为一个可配置的 action_hz 或 source/policy cadence，而不应成为所有控制对象的名字。建议明确区分：

| 名称 | 示例频率 | 含义 |
|---|---:|---|
| policy/action cadence | 15 Hz | 策略或上层动作意图产生频率 |
| reference/IK cadence | 75 Hz | 位控链重新规划和产生可执行关节参考的频率 |
| position command cadence | 150 Hz | 位控 sink 接收的 qcmd 频率 |
| force/impedance cadence | 30/40 Hz | 真实力/阻抗控制器的运行频率 |
| physics cadence | 1000 Hz | 仿真物理积分频率 |
| dataset sample cadence | 可不同 | 采集/回放记录频率，不等于 policy cadence |

所有这些频率都应该在 profile 和记录元数据中单独出现。时间戳应是调度的依据，不能依赖“15 能整除 75/150”这一隐含假设。

### 6.2 建议保留当前 26D v1，但冻结其解释

当前 26D action 作为兼容格式可以继续使用，但需要：

- 固定 layout/version、frame、单位和 quaternion 顺序；
- 明确它是 pose action，不是 qpos action，也不是力控命令；
- 不允许下游根据维度猜测控制模式；
- 由显式的 action_space/schema_version 选择 pose、qpos 或其他模式；
- 旧数据继续通过 adapter 进入新契约，不能静默改变旧数据的含义；
- 对单臂、双臂、手部缺失和 inactive side 的行为写入 capability/validation，而不是靠默认值推断。

RobotAction 当前同时容纳 pose、qpos、mixed 和 hand target，这对迁移和兼容是有价值的；但它越来越像一个“动作联合体”。下一阶段应给不同控制链加严格的 mode validation：

- position-pose：允许 pose → IK → qpos；
- position-qpos：允许明确的 qpos reference/replay，但不能绕过要求的双臂安全规划；
- force/impedance：进入独立的 target/controller schema，不把力控参数塞进 pose 26D；
- task action：必须先通过 task adapter 转成上述一种语义动作。

### 6.3 建议的公共动作字段

不要求一次性重写全部类型，但目标契约至少应能表达：

~~~text
ActionIntent
  schema/version
  timestamp + sequence
  action_space / control_mode
  active effectors and side policy
  values
  controlled frame + reference frame
  units / quaternion convention
  source and provenance
  validity / stale / hold intent
  tool profile / robot model profile
  optional capability requirements
~~~

其中 transport、socket、CAN、线程、设备 index 等字段不属于这个契约。

## 7. 两条控制链的目标形态

### 7.1 位控链：作为第一优先级收敛

建议把位控链固定为如下单向结构：

~~~text
Policy/teleop/replay ActionIntent
  → action validation and safety filter
  → causal pose/qpos resampling
  → single-arm or coupled-bimanual planner
  → joint limits + step/velocity limits
  → collision check + last-safe hold
  → atomic qpos reference at reference_hz
  → qpos interpolation/rate limiting at command_hz
  → JointPositionSink
~~~

对应关系：

- planner、reference frame、interpolation 和 schedule：robot_motion；
- real JointPositionSink、worker、SDK/CANFD：teleop_runtime；
- MuJoCo DirectQposDebugController 或未来 actuator controller：sim；
- replay/deploy 的 source/launcher：各自 runtime，但中间链路复用 robot_motion。

当前默认 15→75→150 是合理的；15→45→150 可以作为兼容 profile，但不要让 45 Hz 的历史路径继续成为隐式默认。

对单臂有一个必须显式决定的能力问题：若启用了双臂碰撞约束，单侧目标不能假设另一侧不存在。当前 real 位置主链要求两臂都具有权威状态是合理的；若未来要支持单臂，应增加明确的 single-arm scene/state profile，或者继续 fail-closed，不能把缺失的被动侧状态默认为零位。

### 7.2 力控/阻抗链：保持独立，不强行伪装成位控

建议目标形态为：

~~~text
ActionIntent / CartesianTarget
  → target validation, frame/profile conversion
  → measured joint/cartesian/force state
  → force/impedance controller at controller_hz
  → Force/ImpedanceCommand
  → RealForceSink
~~~

这条链与位控链有共同的 action/frame/limit/fault 语义，但不是：

~~~text
pose → qpos → direct qpos
~~~

当前 RM65ImpedanceArmBackend 保留在 real runtime 是合理的。后续可以把它拆为：

1. robot_motion 中的控制器接口、输入输出契约、纯目标限制和诊断；
2. teleop_runtime 中的传感器、tare、SDK controller、线程和硬件模式切换。

在 sim 尚未具备相同的力传感器、接触动力学和控制器验证条件前，应把 force/impedance 标为 unsupported capability 或 separate experimental path，而不是把它自动降级成 direct qpos。否则 replay/deploy 看起来“通过了”，实际验证的是完全不同的控制语义。

## 8. replay/deploy 应如何复用控制链

### 8.1 两种 replay 必须分开命名

建议至少区分：

1. **semantic action replay**：回放 15 Hz 的 ActionIntent/RobotAction，重新经过当前选定的 position 或 force chain。它验证的是动作契约和控制链。
2. **executed trajectory replay**：回放已经记录的 qpos/reference/command trajectory。它验证的是时间表、执行 sink 或仿真场景，不重新执行 IK。

第二种不能只叫 replay，否则用户会误以为 replay 和 deploy 使用了同样的规划。轨迹中应记录 source_kind、frame、profile、control_mode 和是否已经经过 IK。

### 8.2 deploy 的建议边界

deploy 可以有自己的策略输入、进程、观测预处理和故障处理，但在动作落到执行端前应进入同一个共享链：

~~~text
policy output
  → policy/task adapter
  → RobotAction / ActionIntent
  → selected control-chain runner
  → reference/command
  → sim or real sink
~~~

因此：

- deploy 不应另写一套独立的双臂 IK；
- sim replay 不应因为是仿真就跳过公共 frame、limits 和版本校验；
- real deploy 不应把 IK 再塞回 arm backend；
- qpos passthrough 必须由显式 profile 选择，并标记安全前提。

## 9. 当前最需要收敛的职责混合

### 9.1 DualArmCollisionAwareRetargeter

当前它同时包含：

- tracker/teleop target 的处理；
- clutch、输入有效性和 pose smoothing；
- 双臂 IK；
- 碰撞、跳变、关节步长和 hold；
- 输出是 pose 还是 joint vector 的选择。

建议拆成两部分：

1. teleop_runtime 保留 Tracker/TeleopTargetBuilder：把 VIVE 输入、校准、clutch 变成带 frame 和 timestamp 的双臂目标；
2. robot_motion 提供 CoupledBimanualMotionPlanner：接收目标和 measured state，输出原子 qpos reference/result。

迁移期间可以保留原 retargeter 作为 facade，内部调用新 planner，避免一次性改动 session、rig 和 transport。

### 9.2 RM65ArmBackend

当前 RM65 real backend 既包含：

- 硬件生命周期和 SDK/CANFD；
- 位置命令流；
- 力/位置厂商模式；
- 单臂 IK；
- 一些 qpos smoothing/limit/状态转换。

目标是让 backend 成为 sink/facade：

- position sink 接收已规划 qpos/qcmd；
- force sink 接收已定义的 force/impedance command；
- FK、状态读取和硬件 readiness 作为 runtime adapter；
- 单臂 IK 迁移到 robot_motion/rm65，旧入口暂时 re-export/委托。

### 9.3 合同的重复和命名漂移

当前 robot_motion/action/schema.py、robot_motion/deploy/schema.py、sim/contracts/action.py 以及 teleop_runtime 的兼容层都涉及动作定义。建议遵循：

- robot_motion 是语义合同的单一来源；
- sim/contracts 只声明仿真 profile 如何 pin 某个公共版本，并补充 sim-only observation/task contract；
- teleop_runtime 的旧路径只做兼容 re-export；
- deploy/task 的 schema 只能描述输入如何映射到公共动作，不能复制公共动作的 frame/单位解释。

同样，RobotAction、NormalizedAction、TeleopCommand、EffectorCommand、PositionReferenceFrame、DirectQposCommand 应在文档和类型上明确处于不同层，避免调用者把任意一个对象直接传给 backend。

## 10. 推荐的目标目录（概念结构）

不建议为了目录好看立即大规模搬文件。可以先通过接口和 facade 收敛，等依赖稳定后再整理。概念上可以逐步靠近：

~~~text
robot_motion/
  contracts/
    action.py          # ActionIntent/RobotAction、版本和 frame
    state.py           # MotionState/feedback
    reference.py       # PositionReference
    command.py         # position/force command semantics
    capability.py
  kinematics/
    interfaces.py      # FK/IK protocols and result diagnostics
    single_arm.py
    coupled.py
    collision.py
  planning/
    position_pipeline.py
    limits.py
    hold.py
    resampling.py
    interpolation.py
  controllers/
    position.py
    impedance.py       # pure protocol/controller pieces
  replay/
  deploy/
  embodiments/
    rm65/              # model, profile, IK/collision adapters

teleop_runtime/
  sources/             # VIVE, hand, camera and other inputs
  retargeters/         # device/input → target intent
  sessions/
  position_control/    # real-time worker and reference publisher
  backends/            # real hardware sinks

sim/
  contracts/           # sim profile pins and observation/task contracts
  mujoco/
    scene/
    controllers/       # sim sinks/adapters
    tasks/
~~~

按现有代码，robot_motion/position_control、robot_motion/replay、robot_motion/rm65/position_collision.py 已经接近这个目标；runtime 和 deploy 仍处于迁移期，不应在没有兼容层的情况下突然改名。

## 11. 建议的收敛顺序

### M0：先冻结契约和能力矩阵

先不改控制行为，写清楚每条链的：

- action space/version；
- frame、单位、tool profile；
- source/action/reference/command/physics 频率；
- 是否支持单臂；
- 是否需要双臂权威状态；
- 是否有 endpoint collision、segment collision；
- 是否支持 position、force/impedance；
- replay 是 action replay 还是 qpos replay。

这一步的产物应是可机器校验的 capability/profile，而不是散落在 README 中的默认值。

### M1：统一公共类型，保留兼容入口

优先定义或补齐：

- ActionIntent；
- MotionState；
- PositionReference；
- JointPositionCommand；
- Force/ImpedanceCommand；
- ControlChainResult/fault/status；
- ControlCapability。

让现有 RobotAction、TeleopCommand 等逐步实现或适配这些类型。此阶段不改 VIVE 传输和 RM65 发送逻辑。

### M2：抽取共享位置 planner

把双臂 retargeter 中的 IK/碰撞/限制/hold 逻辑迁到共享 planner，先由旧 retargeter facade 调用。然后让：

- real teleop；
- real deploy；
- sim one-shot；
- sim replay；
- real/sim dry-run；

都生成相同格式的 PositionReferenceFrame。

验收重点是同一组 state/target/profile 得到相同的 q_ik_raw、q_ref、status 和 reason，差异只能出现在 final sink。

### M3：拆分 real backend

在不改变 transport 行为的前提下，将 RM65 backend 的职责分成：

- position qcmd sink；
- force/impedance sink；
- state/FK/readiness adapter；
- legacy compatibility facade。

把单臂 IK 从 backend 改为委托到共享 robot_motion。旧调用点先保留，逐步删除重复实现。

### M4：统一 replay/deploy/sim runner

使 deploy 和 semantic replay 复用同一个 selected control-chain runner；使 MuJoCo direct-qpos 只是一个明确的 position sink。增加 golden trace，比较 sink 之前的参考和命令序列。

### M5：再做连续碰撞和动力学仿真

当前 endpoint collision recheck 是有用但不完整的安全边界。只有在以下条件满足后再推进：

- 75→150 插值段的连续碰撞检查或安全轨迹生成；
- sim 和 real 使用可追溯的同版本机器人/工具/碰撞 profile；
- 仿真具有真实 actuator、延迟、限速和接触语义；
- force/impedance controller 有可验证的 sim capability。

不要为了让 sim 看起来和 real “都有同名接口”，而提前伪造力控或动力学等价性。

## 12. 验证门槛

每次职责迁移都应有以下类型的测试：

### 公共契约

- action encode/decode、版本和 legacy adapter round-trip；
- pose/qpos/hand/mixed mode 的显式校验；
- frame、单位、quaternion 顺序；
- stale、invalid、缺 side、缺 tool profile 时的 fail-closed 行为。

### 规划确定性

- 相同 state/target/profile/时间戳得到相同结果；
- 单臂 IK 的成功、无解、越界、跳变和 hold；
- 双臂 IK 的被动侧保持、碰撞 backoff、last-safe 和原子输出；
- endpoint collision 明确通过/拒绝，不能被误报为 segment certification。

### 时间链

- 15→75→150 和 15→45→150；
- 非整数时间间隔、延迟、乱序、重复 timestamp；
- qcmd target preemption 以最后已发/已接受值为锚；
- replay 和 online resampling 的结果一致。

### sim/real 等价边界

- 在 final sink 之前比较同一 action 的 reference/qcmd golden trace；
- MuJoCo sink 只执行 joint command，不偷偷执行另一套 IK；
- real fake sink 验证 dispatch/hold/partial failure；
- direct qpos debug 明确带 execution mode；
- force capability 不支持时必须报不支持，不能静默切到 direct qpos。

### 运行时和安全

- real worker 停止、hold、stale、backend not-ready；
- 双臂部分 dispatch 失败后的 fail-stop/hold；
- session 输入丢失与 planner 无输入的区分；
- transport 层行为保持原有测试和接口不变。

## 13. 最终建议清单

建议按以下原则作为后续实现的判断标准：

1. **跨 sim/real 的动作、状态、参考和控制语义，放 robot_motion。**
2. **单臂 IK、RM65 IK adapter、双臂耦合防碰撞规划，放 robot_motion；VIVE 到目标 pose 的映射留在 teleop_runtime。**
3. **15 Hz 是上层 cadence；75/150/40 Hz 是不同控制阶段的 cadence，必须分别建模。**
4. **position 和 force/impedance 是两条明确的控制链，共享契约和安全原则，但不共用一个含糊的 command。**
5. **replay/deploy 复用共享链；semantic action replay 和 executed qpos replay 必须显式区分。**
6. **sim direct qpos 继续保留，但只能作为 debug/kinematic sink；未来动力学 actuator 是另一个实现。**
7. **硬件 transport、VIVE transport、MuJoCo scene 和进程生命周期不进入 robot_motion。**
8. **先通过 facade、capability、golden trace 收敛边界，再做目录重构。**
9. **当前 endpoint collision 的限制必须保留在结果和文档中，不能提前宣称连续无碰撞。**
10. **本轮架构评审不需要改变现有 VIVE 或 RM65 的传输逻辑；传输仍是最终执行端的实现细节。**

总体判断：当前方向可以继续推进，最优先的工作不是再增加新的 sim controller，而是先把共享 action/状态/参考/命令契约冻结，并从 DualArmCollisionAwareRetargeter 与 RM65ArmBackend 中抽出已经存在但尚未归位的 IK、碰撞和位置规划逻辑。这样 real 位控、sim 位控、replay 和 deploy 才会真正共享同一条运动控制链；力控则以独立 capability 的方式逐步接入，而不是被迫套进位控模型。
