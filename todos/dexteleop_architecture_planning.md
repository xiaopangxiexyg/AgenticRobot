# DexTeleop：模型输出、人类数采与 sim/真机控制流架构规划

## 结论

需要调整项目结构，但不建议现在进行大规模目录搬迁或重写。当前项目已经具备不少关键基础：

- robot_motion 中已有统一动作语义、Rig、因果历史、安全过滤、位置控制。
- dataset_contracts 已有 v2.2/v2.3 的策略输入输出和时间因果约束。
- teleop_runtime 已有真机数采、部署和新的 RuntimeKernel 原型。
- sim 已有 MuJoCo、CubeLift、快速执行器和向量化雏形。

但目前还没有形成一个真正统一的：

> 人类动作 / 模型动作 / replay 动作 → 统一仲裁 → 统一安全与控制 → sim/真机执行 → 统一轨迹记录

这正是 DAgger、human-in-the-loop RL、真机 RL 后续会遇到的核心问题。

本次检查只读，没有修改代码。当前工作树存在大量未提交修改、删除和未跟踪文件，因此正式实施前应该先冻结一个架构基线或创建独立分支。

---

## 一、当前项目结构判断

| 模块 | 当前状态 | 判断 |
|---|---|---|
| robot_motion | 已有 RobotAction、Rig、安全过滤、位置控制、部署观测 | 应继续作为 sim/真机共享的控制语义层 |
| teleop_runtime | 真机数采、TeleopSession、部署循环比较完整，但入口文件很大 | 应保留真机编排职责，不应成为 sim 依赖 |
| sim | MuJoCo/CubeLift/VectorExecutor 已存在，但任务和动作维度强绑定 | 目前更像仿真验证和 RL embryo，还不是完整 deployable sim |
| deployments | 有 OpenPI client、模型握手和 artifact 元数据 | 可以作为模型服务/策略客户端层 |
| dataset_contracts | schema 较完整，有 v2.2 PolicyIOLayout、v2.3 因果快照 | 还缺统一控制时间线和跨域 provenance |
| dataset_interface | 有 PolicyIO batch、图像和 state pipeline | 还缺多模态、时间窗口化的统一轨迹接口 |
| model | 当前没有真正的训练/策略/rollout 包 | 必须新增 |
| umi_runtime | 当前基本为空 | 不建议直接作为通用学习框架入口，除非明确其职责 |

已有设计文档其实已经指出了正确方向，尤其是：

- [sim/STRUCTURE_PLAN.md](/home/zc/HandMotion/Developing/DexTeleop/sim/STRUCTURE_PLAN.md)
- [sim/SIM_ENV_TODO.md](/home/zc/HandMotion/Developing/DexTeleop/sim/SIM_ENV_TODO.md)
- [sim/REAL2SIM_SIM2REAL_TODO_TMP.md](/home/zc/HandMotion/Developing/DexTeleop/sim/REAL2SIM_SIM2REAL_TODO_TMP.md)

当前最重要的不是再设计一套完全不同的结构，而是把这些文档中的边界真正落地。

---

## 二、目前最关键的结构性问题

### 1. 现在存在多套控制入口

主要有：

- [teleop_runtime/session.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/session.py)
- [teleop_runtime/collection/runtime.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/collection/runtime.py)
- [teleop_runtime/core/kernel.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/core/kernel.py)
- [teleop_runtime/deploy/run_policy.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/deploy/run_policy.py)
- robot_motion/deploy/DeployRuntime

它们各自拥有部分 observation、action、dispatch、fault、record 逻辑。

现在接入 DAgger 后，很容易出现：

~~~text
人类控制循环 ────────┐
                      ├── 同时写入真机 backend
模型控制循环 ────────┘
~~~

这会带来动作竞争、时间错位、双重安全过滤和无法解释的训练数据。

必须改成：

~~~text
人类动作 ────────┐
模型动作 ────────┼── ActionArbiter ── Safety ── Planner ── Dispatcher
Replay 动作 ────┘
~~~

整个系统只能有一个最终动作拥有者和一个最终 dispatcher。

### 2. RuntimeKernel 的位置和能力还不够通用

[teleop_runtime/core/kernel.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/core/kernel.py) 的思路是对的，但它现在有三个限制：

- 位于 teleop_runtime，sim 无法自然依赖它。
- 一个 step 只有一个 RobotActionSource。
- 没有明确区分 policy rate、human input rate、reference rate、command rate 和 physics rate。

建议把它的共享部分提升到：

~~~text
robot_motion/runtime/
~~~

因为 robot_motion 按项目规则本来就是 sim/真机共享的运动控制层。

建议逐步形成：

~~~text
robot_motion/runtime/
├── kernel.py
├── proposal.py
├── authority.py
├── arbitration.py
├── scheduler.py
├── episode.py
├── trace.py
├── causal_history.py
├── interfaces.py
└── rig.py
~~~

teleop_runtime/core 可以先保留兼容层并 re-export，但 sim 不应该反向依赖 teleop_runtime。

### 3. 当前 sim 控制流还不是 deployable policy 控制流

当前 [sim/mujoco/tasks/cube_lift/env.py](/home/zc/HandMotion/Developing/DexTeleop/sim/mujoco/tasks/cube_lift/env.py) 大致是：

~~~text
7D task action
    → DirectQposDebugController
    → MuJoCo scene
    → 31D privileged observation
~~~

这对于 CubeLift 的任务调试和快速 RL 是合理的，但它与真机部署仍不是同一条语义链路：

- action 是 task-local 7D。
- observation 包含 privileged state。
- 控制器是 direct-qpos。
- [sim/mujoco/runtime/cube_lift_vector.py](/home/zc/HandMotion/Developing/DexTeleop/sim/mujoco/runtime/cube_lift_vector.py) 仍然硬编码动作和观测维度。
- 没有统一使用 DeploymentObservation、PolicyIOLayout 和完整的 CanonicalDeployActionPipeline。

建议保留当前 CubeLift 快速路径，但明确标记为：

~~~text
task_native / privileged / non_deployable
~~~

同时新增一条 deployable sim 路径：

~~~text
MujocoObservationProvider
    → DeploymentObservation
    → PolicyIOLayout adapter
    → RobotAction
    → CanonicalActionPipeline
    → MuJoCo Rig/backend
~~~

这样才能真正验证“同一个 policy artifact 是否可以切换 sim 和真机”。

---

## 三、推荐的统一运行时模型

核心对象不要混在一起，建议分成以下几层。

### 1. PolicyIOLayout

负责模型张量语义：

- 输入字段和维度
- 相机顺序
- 图像预处理
- state layout
- action layout
- 单位、坐标系、TCP
- absolute/relative action
- action horizon
- control frequency
- normalization
- layout hash

它是模型接口契约，不应直接承担硬件 dispatch。

### 2. DeploymentObservation

负责控制时刻的观测快照：

- 机器人状态
- 相机帧
- tactile/depth
- 时间戳
- valid mask
- frame age
- fault 状态
- last action

现有 [robot_motion/deploy/observation.py](/home/zc/HandMotion/Developing/DexTeleop/robot_motion/deploy/observation.py) 可以继续演进。

### 3. ActionProposal

人类、模型、replay 都先产生 proposal，而不是直接发命令：

~~~text
ActionProposal
├── source: human / policy / replay / scripted
├── RobotAction
├── source_timestamp
├── observation_timestamp
├── valid_until
├── effector_mask
├── confidence
├── inference_latency
├── source_metadata
└── validity
~~~

人类动作和模型动作必须拥有同样的 RobotAction 语义。

### 4. ActionArbiter

负责动作拥有权和干预策略：

~~~text
ActionProposal[]
    → authority decision
    → per-effector selection
    → handoff smoothing
    → selected RobotAction
~~~

第一阶段建议只实现：

- policy_only
- human_only
- policy_shadow
- human_override
- per_effector_override

暂时不要直接做任意的 pose 线性 blend。对于位姿，应使用 SE(3) 插值或在 reference trajectory 层做平滑；四元数不能直接逐元素线性混合。

控制权可以建模成：

~~~text
POLICY
HUMAN
HANDOFF
HOLD
FAULT
~~~

人类输入过期、模型推理失败、硬件 fault 时，必须进入明确的 hold/fallback 状态，不能让旧 action 无限复用。

### 5. CanonicalActionPipeline

仲裁之后统一经过：

~~~text
RobotAction
    → action validation
    → safety filter
    → IK / coupled planner
    → reference trajectory
    → command interpolation
    → real Rig 或 MuJoCo Rig
~~~

现有 [robot_motion/deploy/canonical_action.py](/home/zc/HandMotion/Developing/DexTeleop/robot_motion/deploy/canonical_action.py) 和 robot_motion/position_control 可以复用。

尤其要保留现在已经形成的多频率架构：

~~~text
human input       高频
policy inference  约 15 Hz
reference / IK    约 45 Hz
command           约 150 Hz
physics           约 1000 Hz
~~~

模型推理不能放在 150 Hz command loop 中。模型只产生新 action chunk，高频控制线程消费最近一个有效 chunk。

---

## 四、DAgger 和真机 RL 应如何共用一套框架

### DAgger

每个 policy tick 最少记录：

~~~text
observation_t
policy_action_t
expert_action_t
selected_action_t
safety_filtered_action_t
executed_action_t
authority
intervention_mask
takeover_reason
policy_latency
observation_timestamp
action_timestamp
dispatch_timestamp
measured_state_{t+1}
~~~

其中：

- expert_action_t 是 DAgger 的训练标签。
- executed_action_t 是实际执行动作。
- selected_action_t 是仲裁结果。
- safety_filtered_action_t 用于判断安全层改变了多少。
- intervention_mask 可以按左臂、右臂、左手、右手分别记录。

不能只记录最终执行动作，否则无法区分：

~~~text
模型主动控制
人类接管
安全层修改
人类只控制某一个末端
~~~

现有 [robot_motion/runtime/causal_history.py](/home/zc/HandMotion/Developing/DexTeleop/robot_motion/runtime/causal_history.py) 的 source_timestamp <= cutoff 机制非常适合作为 DAgger 的时间对齐基础，可以避免把未来的人类动作泄漏到当前 observation。

### 真机 RL

真机 RL 的 rollout 应记录：

~~~text
observation
executed_action
reward
terminated
truncated
next_observation
faults
latency
intervention
policy_artifact_id
environment_id
episode_id
~~~

真机 RL 使用的 action 应是实际执行的安全动作，而不是未经安全层处理的 policy intent。

视频、tactile 原始数据和大对象不应该阻塞控制线程。建议采用：

~~~text
ControlLoop
    → lightweight RolloutSink
    → bounded queue
    → asynchronous media/dataset writer
~~~

这与现有 DropOldestAsyncObserver 思路一致。

---

## 五、建议新增或调整的目录

推荐目标结构如下：

~~~text
robot_motion/
├── action/
├── runtime/
│   ├── kernel.py
│   ├── proposal.py
│   ├── authority.py
│   ├── arbitration.py
│   ├── scheduler.py
│   ├── episode.py
│   ├── trace.py
│   ├── causal_history.py
│   └── rig.py
├── deploy/
└── position_control/

sim/
├── core/
│   ├── protocols.py
│   ├── task_spec.py
│   ├── env_adapter.py
│   └── vector_env.py
├── adapters/
├── tasks/
└── mujoco/
    ├── adapters/
    ├── backend.py
    ├── runtime/
    └── tasks/

model/
├── core/
│   ├── policy.py
│   ├── learner.py
│   ├── rollout.py
│   └── artifacts.py
├── policies/
├── data/
│   ├── trajectory.py
│   ├── rollout_buffer.py
│   └── replay_buffer.py
├── algorithms/
│   ├── il/
│   │   ├── bc/
│   │   └── dagger/
│   └── rl/
│       ├── ppo/
│       └── sac/
├── sampling/
├── artifacts/
└── runners/

dataset_contracts/
├── policy_io/
├── control_trace.py
├── cross_domain.py
└── io/

dataset_interface/
├── policy_io.py
├── trajectory.py
└── windows.py

teleop_runtime/
├── sources/
├── collection/
├── deploy/
└── adapters/

deployments/
├── client/
├── openpi/
└── artifacts/
~~~

几个边界需要明确：

- model/algorithms 不得依赖 MuJoCo、硬件或 teleop_runtime。
- dataset_contracts 不得依赖 Torch、MuJoCo、硬件和训练器。
- sim 不得依赖 teleop_runtime。
- teleop_runtime 可以依赖 robot_motion、硬件和部署 client。
- deployments 负责策略调用和 artifact，不负责训练循环。
- robot_motion/runtime 只提供通用运行时和控制语义，不放真机设备初始化。

---

## 六、数据结构方面必须补齐的内容

目前 dataset_contracts 已有 Policy I/O，但还缺一个跨域控制轨迹契约。建议新增：

### CrossDomainManifest

至少包含：

- domain：sim / real / replay
- policy_io_layout_hash
- simulation_contract_hash
- action/observation frame
- TCP、tool、calibration 信息
- 相机预处理和相机顺序
- policy/reference/command/physics frequency
- timestamp policy
- policy artifact id
- source dataset
- code revision
- random seed
- environment/task profile

### ControlTrace 或 InterventionFrame

用于描述一个控制时刻，不建议继续把所有字段都塞进现有 ActionRecord。这样可以避免破坏已有 v2.1/v2.2 数据。

### dataset_interface/trajectory.py

建议形成统一的时间窗口读取接口：

~~~text
TrajectoryWindow
├── observation
├── images / depth / tactile
├── robot_state
├── policy_action
├── expert_action
├── executed_action
├── intervention_mask
├── valid_mask
├── timestamps
├── domain
└── provenance
~~~

新的 DAgger 和 RL 数据应以此为入口。v2.1 保留兼容读取，新数据建议以 v2.2 typed layout 和因果时间线为默认。

---

## 七、现有代码建议如何迁移

不建议第一步就重写 run_collect.py 或 run_policy.py。它们目前很大，但仍可以作为旧入口。

推荐渐进式包裹：

1. 将现有 OpenPI 或其他模型包装成 PolicyActionSource。
2. 将现有 retargeter 包装成 HumanActionSource。
3. 将 replay/script 包装成 ReplayActionSource。
4. 三者都只产生 ActionProposal。
5. 由唯一的 ActionArbiter 选择动作。
6. 复用现有 safety、position control 和 Rig。
7. 通过统一 ControlTraceObserver 写数据。
8. 最后再把 run_collect.py、run_policy.py 简化成组装入口。

尤其要避免让 TeleopSession 和新的 RuntimeKernel 各自保留一套独立的：

- armed/fault 状态机
- hold/recovery 逻辑
- action dispatch
- safety filter

最终应该只有一套控制权和故障状态源。

---

## 八、实现前建议冻结的几个决策

建议默认采用以下方案：

1. 模型输出先映射为 PolicyIOLayout，再转换成 RobotAction，不要直接输出 TeleopCommand 或硬件 qpos。
2. 第一版人机协同使用 priority override 和 per-effector mask，不做任意连续 blend。
3. policy observation 禁止 privileged state；privileged state 只提供给 critic、reward 或 debug info。
4. 新数据以 v2.2 typed action 为主，v2.1 只作为兼容适配层。
5. sim 和真机必须共享 RobotAction、Safety、Planner 和 artifact compatibility 检查。
6. 每个 episode 必须有明确的 episode_id、domain、task profile、policy artifact 和 seed。
7. 所有动作都必须区分 intent、selected、safe、commanded、measured。

---

## 九、推荐实施顺序

### Phase 0：冻结基线

先整理当前工作树，形成一个可追踪的 baseline commit，并记录：

- 当前 v2.1/v2.2/v2.3 状态
- 哪些文件属于正在进行的迁移
- 目标 policy action profile
- deployable sim 与 task-native sim 的区别

### Phase 1：共享控制内核

先不接真机、不训练模型，完成：

- ActionProposal
- ActionArbiter
- authority state machine
- 多频率 scheduler
- ControlTrace
- policy/human/replay 三种 synthetic source

重点测试时间因果、接管、超时、动作 mask 和安全层修改。

### Phase 2：统一 sim 路径

新增 MuJoCo observation/action adapter，使一个 deployable policy 可以走：

~~~text
model → RobotAction → common pipeline → MuJoCo backend
~~~

同时保留现有 CubeLift fast/native RL 路径，避免为了 deploy parity 破坏快速实验效率。

### Phase 3：真机 shadow 与人工接管

先运行：

- policy shadow：模型只预测和记录，真人实际控制
- human override：模型持续预测，人类可以接管
- per-effector override：只接管某一侧

之后再考虑 policy-only 和真机 RL。

### Phase 4：统一 DAgger / RL 数据

实现：

- TrajectoryWindow
- DAgger expert label export
- RL rollout buffer
- 异步视频/tactile writer
- episode manifest
- policy artifact 与数据集绑定

### Phase 5：新增 model/

最后再逐步加入：

- BC
- DAgger
- PPO/SAC
- sim rollout collector
- real rollout collector
- checkpoint/export/eval runner

训练依赖应放到独立 optional dependency，不要把 Torch、训练器和 RL 库加入 robot_motion 或 dataset_contracts 的基础依赖。

---

## 最终建议

当前最值得先做的不是“接入某个具体模型”，而是建立一个共享的 robot_motion/runtime 控制内核，并补齐：

1. 多来源 ActionProposal
2. 人机 ActionArbiter
3. 多频率控制调度
4. 跨域 ControlTrace
5. deployable sim adapter
6. model/ 学习包边界
7. 统一 trajectory/rollout 数据接口

完成这几个基础后，DAgger、模型 shadow、人工接管、sim RL 和真机 RL 都可以成为同一框架下的不同运行模式，而不会继续产生多套互相不兼容的控制流。
