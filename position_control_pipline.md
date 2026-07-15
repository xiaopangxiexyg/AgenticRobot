# 机械臂遥操数采与策略部署系统结构

## 1. 设计目标

系统同时支持：

1. 人类遥操与真机数据采集；
2. 基于采集数据训练 15 Hz 策略模型；
3. 将模型输出转换为安全、连续的真机控制指令；
4. 保持数采动作语义与部署动作语义尽可能一致。

系统采用三层频率：

| 层级     |    名义频率 | 主要职责                            |
| ------ | ------: | ------------------------------- |
| 模型与数据层 |   15 Hz | 采集 observation、记录 action、执行模型推理 |
| 动作参考层  | 约 75 Hz | 遥操 IK、动作约束、部署动作插值与 IK           |
| 真机指令层  |  150 Hz | 生成并下发连续 qpos 指令                 |

其中，75 Hz只是`reference_hz`的当前默认值，可通过统一参数修改；算法、metadata和schema
不以45/75命名或分支。默认15:75为1:5，但实现不依赖固定帧计数，各模块仍保留独立时间戳。

---

## 2. 核心动作定义

系统中不同阶段的动作必须使用明确、固定的语义。

| 变量                   | 含义                                   |
| -------------------- | ------------------------------------ |
| `teleop_pose_raw`    | 人类遥操设备产生的原始末端 pose                   |
| `q_ik_raw`           | 原始 pose 经过 IK 后得到的未经处理 qpos          |
| `q_ref`           | 经过遥操限位和平滑后的准下发 qpos                  |
| `action_pose_ref`    | 对 `q_ref` 执行 FK 后得到的末端 pose       |
| `action_label_15`    | 15 Hz 数据中记录的策略训练标签                   |
| `policy_pose_15`     | 模型以 15 Hz 输出的末端动作 pose               |
| `policy_pose_safe_15` | 经过离散关键帧 finite/workspace/步长限制的 15 Hz pose |
| `policy_pose_ref` | 由 15 Hz 模型动作插值得到的约 75 Hz pose        |
| `q_policy_raw`    | `policy_pose_ref` 经过 IK 后得到的 qpos |
| `q_safe_endpoint` | replay/deploy 经绝对/步长限位并完成 75 Hz endpoint collision recheck 的 qpos；不代表 75→150 整段已认证 |
| `q_cmd_150`          | 插值后实际发送给机器人的 150 Hz qpos             |
| `q_measured`         | 机器人实际返回的关节状态                         |

模型训练的 action 语义为：

> 经过数采遥操控制器限位和平滑后的可执行参考动作所对应的末端 pose。

即：

```text
action_label_15 = FK(q_ref)
```

---

## 3. 模块定义

### 3.1 `ObservationSampler`

负责以 15 Hz 采集模型输入。

输入：

* 机器人 state；
* 相机 image；
* 其他传感器信息。

输出：

```text
observation_15
```

要求：

* state、image 和其他 observation 带有时间戳；
* 不负责生成或修改 action。

---

### 3.2 `TeleopInputProvider`

负责读取人类遥操输入。

输出：

```text
teleop_pose_raw
```

其运行频率独立于 15 Hz 数据采集频率。

---

### 3.3 `IKSolver`

负责将末端 pose 转换为机器人关节位置。

数采输入：

```text
teleop_pose_raw
```

部署输入：

```text
policy_pose_ref
```

输出：

```text
q_ik_raw
```

或：

```text
q_policy_raw
```

数采和部署应尽量复用相同的 IK 实现、坐标系定义和机器人配置。

---

### 3.4 `TeleopActionProcessor`

这是现有数采遥操系统中的动作处理模块。

输入：

```text
q_ik_raw
```

输出：

```text
q_ref
```

职责包括已有的：

* 关节限位；
* 动态约束；
* 平滑处理；
* 无效 IK 结果处理。

`q_ref` 是数采阶段的标准动作参考，也是生成训练 action label 的来源。

---

### 3.5 `ActionLabelBuilder`

负责生成 15 Hz 训练标签。

处理流程：

```text
q_ref
    -> FK
    -> action_pose_ref
    -> 记录为 action_label_15
```

每次记录 15 Hz observation 时，读取该时刻之前最近一次已经生成的有效 `action_pose_ref`。

不要求 15 Hz 和约 75 Hz 循环严格按照固定比例同步。

---

### 3.6 `PolicyInference`

负责以 15 Hz 执行模型推理。

输入：

```text
observation_15
```

输出：

```text
policy_pose_15
```

如果模型输出 action chunk，该模块同时负责维护当前 chunk 和新 chunk，但对下游统一输出当前应执行的 15 Hz action pose。

---

### 3.7 `PolicyPoseInterpolator`

负责将 15 Hz 模型动作转换为约 75 Hz 动作参考。

输入：

```text
policy_pose_safe_15
```

输出：

```text
policy_pose_ref
```

当前阶段采用线性插值，不要求模型直接预测 75 Hz 动作。

这里的“线性”是：xyz 线性、quaternion 最短弧 SLERP；qpos 模式为逐关节线性。它是明确的时间戳轨迹生成，不是 low-pass/EMA 时序平滑。

该模块应按照实际时间推进插值，而不是依赖固定的三个子步骤，因此在中间控制循环频率出现波动时仍可正常工作。

---

offline replay 已知整段后继关键帧，可直接重采样；online deploy 不允许读取尚未到达的未来动作，因此用一个 15 Hz action period 的显式因果延迟换取真实插值。零延迟且没有右括号时只能 hold，不能伪称完成插值。

### 3.8 `PolicyKeyframeSafetyGuard` 与 `DeploymentExecutionSafetyGuard`

第一层在插值前处理 15 Hz 离散关键帧：

```text
policy_pose_15
    -> finite / quaternion / workspace / pose-or-joint step gate
    -> policy_pose_safe_15
```

这层只做最小必要的显式安全限制，不做时序滤波；限制后的关键帧才允许进入 75 Hz 插值器。

第二层对每个 75 Hz reference 执行 IK 与执行安全检查：

负责部署阶段的最终动作安全检查。

输入：

```text
q_policy_raw
```

输出：

```text
q_safe_endpoint（当前 real replay/deploy）
```

部署阶段必须保留：

* 绝对关节位置限制；
* 工作空间和碰撞相关限制；
* IK 有效性检查；
* 必要的速度、加速度或单步变化限制；
* 异常动作、通信异常和失效保护。

15 Hz keyframe gate 不得在每个 75 Hz 插值点上重复调用，否则会把单步安全限制误变成隐式时序平滑。完整目标 contract 中，75 Hz 仍需保留 IK、绝对关节限位、奇异/碰撞端点、执行步长和 transport 边界检查。

当前实现边界必须单独说明：real replay/deploy 已在 75 Hz pose 插值之后接入共享
coupled dual-arm collision IK，并按“绝对关节限位 -> 每 reference 步长限位 -> 最终
endpoint collision recheck”的顺序得到 `q_safe_endpoint`。real backend 只接收 joint
qpos，不再分别执行单臂 Cartesian IK。planner 从 scene URDF 解析完整双臂基座变换，并要求
left+right 权威当前 qpos；缺少被动臂状态会 fail-closed。

真机 reset 的“权威”还包括新鲜性：本轮 hardware read 必须成功，hardware/measured qpos、
sequence 与 monotonic timestamp 必须一致，默认 age 不超过 0.25 s、双臂 timestamp skew
不超过 0.05 s。读取失败时不得用 cached target/current qpos 建立新的 last-safe 状态。
planner 输出进入 backend 时必须标记为 coupled/prelimited；若 backend 又做单臂改写或
singularity hold，则整组 fault/hold，且不得继续同 tick hand。

尚未完成的是 `q_safe_endpoint` 之间 75→150 关节插值线段的连续碰撞认证。所选 scene
profile 的固定 hand/link proxy 也不能泛化为任意自定义工具几何已完整认证。因此
`q_safe_endpoint` 只表达 endpoint-safe，不表达 continuous-path-safe；这一个剩余项归入
本轮明确暂缓的问题 2。

默认情况下，不再对所有模型动作重复执行数采阶段持续生效的低通或平滑处理。

设计原则是：

```text
合法且连续的模型动作 -> 尽量原样通过
违反安全约束的动作 -> 进行最小必要修改
```

若未来增加额外平滑，应作为独立、可配置模块，并记录平滑前后的动作，不能隐式包含在安全限位逻辑中。

---

### 3.9 `CommandInterpolator`

负责将约 75 Hz 的 endpoint-safe qpos 参考转换为约 150 Hz qcmd。该 planner 属于
`robot_motion`，sim/real 共用相同的 timestamp、preemption、stale 与 segment-anchor 语义；
只有最终 sink 不同。

输入：

```text
q_safe_endpoint
```

输出：

```text
q_cmd_150
```

数采阶段对应输入为：

```text
q_ref
```

该模块只负责高频轨迹生成和指令连续化，不应重新改变上层动作目标的语义。sim 最终由
仿真 qpos controller 消费，real 最终由 RM65 `Movej_CANFD` 消费。

---

### 3.10 `RobotCommandInterface`

负责：

* 以 150 Hz 向机器人下发 `q_cmd_150`；
* 接收机器人实际关节状态 `q_measured`；
* 提供通信状态和执行状态。

---

### 3.11 `TrajectoryLogger`

负责记录完整链路中的关键数据。

至少记录：

```text
timestamp
observation_15
teleop_pose_raw
q_ik_raw
q_ref
action_pose_ref
policy_pose_15
policy_pose_ref
q_policy_raw
q_safe_endpoint
q_cmd_150
q_measured
```

同时记录动作是否被安全模块修改以及修改原因。

---

## 4. 数采链路

```text
TeleopInputProvider
        |
        v
teleop_pose_raw
        |
        v
IKSolver
        |
        v
q_ik_raw
        |
        v
TeleopActionProcessor
        |
        v
q_ref
     /       \
    v         v
FK        CommandInterpolator
    |         |
    v         v
action_pose_ref    q_cmd_150
    |                 |
    v                 v
ActionLabelBuilder   RobotCommandInterface
    |
    v
action_label_15
```

15 Hz 数据样本为：

```text
sample = {
    observation: observation_15,
    action: latest_valid(action_pose_ref)
}
```

其中 `latest_valid` 表示读取采样时间之前最近生成的有效动作参考，而不是假设固定的 1:3 周期关系。

---

## 5. 部署链路

```text
ObservationSampler
        |
        v
observation_15
        |
        v
PolicyInference
        |
        v
policy_pose_15
        |
        v
PolicyKeyframeSafetyGuard
        |
        v
policy_pose_safe_15
        |
        v
PolicyPoseInterpolator
        |
        v
policy_pose_ref
        |
        v
IKSolver
        |
        v
q_policy_raw
        |
        v
DeploymentExecutionSafetyGuard
        |
        v
q_safe_endpoint
        |
        v
CommandInterpolator
        |
        v
q_cmd_150
        |
        v
RobotCommandInterface
```

---

## 6. 数采与部署对应关系

| 数采阶段                    | 部署阶段                    |
| ----------------------- | ----------------------- |
| 人类输出 `teleop_pose_raw`  | 模型输出 `policy_pose_15`   |
| 人类输入本身约 75 Hz           | 15 Hz 模型动作插值到约 75 Hz    |
| `IKSolver`              | `IKSolver`              |
| `TeleopActionProcessor` | 15 Hz `PolicyKeyframeSafetyGuard` + 75 Hz `DeploymentExecutionSafetyGuard` |
| 得到 `q_ref`           | 得到 endpoint collision checked 的 `q_safe_endpoint` |
| 插值到 `q_cmd_150`         | 插值到 `q_cmd_150`         |

两条链路的关键区别是：

* 数采阶段需要将人类原始输入处理为平滑、可执行的示教动作；
* 部署阶段的模型已经学习了处理后的示教动作，因此不默认重复进行相同的持续平滑；
* 部署阶段仍必须保留安全约束和异常保护。

---

## 7. 系统不变量

实现和修改代码时必须保持以下规则：

1. `action_label_15` 始终来源于 `FK(q_ref)`。
2. 模型输出 action 的定义始终是“可执行参考动作 pose”。
3. 数采与部署使用一致的机器人坐标系、pose 定义和 IK 配置。
4. 15 Hz 与约 75 Hz 不依赖固定整数倍同步。
5. 部署通过插值将 15 Hz pose 转换为约 75 Hz pose。
6. 75→150 qpos planner 为 sim/real 通用；150 Hz 最终 qpos sink 才区分 sim/real。
7. 安全限位始终开启。
8. 部署安全模块以最小干预为原则，不默认重复数采阶段的持续平滑。
9. 所有跨频率数据必须携带时间戳。
10. 安全模块对动作的任何修改都必须可记录、可统计、可回放。
11. 15 Hz keyframe safety 先于 75 Hz 插值，且不会在每个插值点重复施加。
12. online deploy 的插值必须因果；当前实现明确记录一个 action period 的延迟、parent index 和 alpha。
13. 手/夹爪保持 15 Hz 离散语义，不由机械臂 75 Hz 插值器生成中间动作；online deploy 中混合 arm+hand action 共用一个 action-period 因果延迟，并在对应 75 Hz reference tick 做同相位逻辑调度。hand I/O 使用独立的一槽 latest-wins worker，避免阻塞 arm reference；若 reference worker 落后，不做追赶 burst，只保留最新到期值。这里的同相位只指调度语义，不承诺两个物理设备同时生效；实际 callback 时间、backend receipt 和内部 retry 必须从 debug event 判断。hand fault 是 fail-stop：阻止后续 arm callback，但一个已经进入 backend 的 arm callback 无法撤销，可能完成后再 hold，并记录为 `accepted_before_async_abort`；真实 callback 必须具有有限 I/O timeout。
14. replay/deploy 的 position 真机输出必须经过双臂 75 Hz endpoint collision checker；缺少 left/right 权威 qpos/TCP 时 fail-closed，不能猜被动臂状态。
15. metadata 必须同时记录 `endpoint_collision_checked` 和 `segment_collision_certified`；当前后者固定为 `false`，不得将 endpoint-safe 表述成连续轨迹已认证。
16. 真机 collision planner reset 不得使用失败/过期的缓存 qpos；双臂测量 freshness/skew 和来源必须进入 debug metadata。
17. 75 Hz coupled qpos 进入 150 Hz planner 前后都必须验证 backend final target 与 arm-stream health；已知 fault/单臂改写必须先于 hand fail-closed。左右臂顺序 queue 与跨设备 hand I/O 仍非原子事务，剩余竞态必须显式记录。
