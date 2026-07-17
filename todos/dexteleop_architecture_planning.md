# DexTeleop：模型输出、人类数采与 sim/真机控制流架构规划

## 结论

需要调整项目结构，但不建议现在进行大规模目录搬迁或重写。当前项目已经具备不少关键基础：

- robot_motion 中已有统一动作语义、Rig、因果历史、安全过滤、位置控制。
- dataset_contracts 已有 v2.2/v2.3 的策略输入输出和时间因果约束。
- teleop_runtime 已有正式 GUI 真机 teleop、headless 兼容数采和 policy deployment；
  `RuntimeKernel` 目前仍是 experimental，不作为生产入口。
- sim 已有 MuJoCo、CubeLift、快速执行器和向量化雏形。

第一批收敛已经完成了 record 生命周期和 human/headless 的最终 dispatch seam；随后又把
policy loader、v2.3 表示适配，以及更完整的 policy stream 状态继续抽到了
`robot_motion`。当前仍没有形成一个真正统一的：

> 人类动作 / 模型动作 / replay 动作 → 统一仲裁 → 统一安全与控制 → sim/真机执行 → 统一轨迹记录

这里的“统一轨迹记录”不能只保存最终发送给 Rig 的一个 action 向量。数采时必须在控制 cut内实时保存每个 action component 的来源、proposal、选择结果、安全修改、command lineage和 measured 对齐关系；否则后续无法可靠区分 DAgger 的 expert label、policy intent、人工接管和真正执行的行为。

这正是 DAgger、human-in-the-loop RL、真机 RL 后续会遇到的核心问题。

因此当前应继续做边界收敛，而不是再增加新的控制入口。正式接入模型前，仍需要冻结控制语义和数据契约；工作树中的其他并行修改不属于本次架构收敛范围。本轮暂不实现 sim frontend，sim 只保留为后续复用约束。

---

## 一、当前项目结构判断

| 模块 | 当前状态 | 判断 |
|---|---|---|
| robot_motion | 已抽出 `action`、`runtime`、`deploy`、`position_control` 和 RM65 耦合规划；拥有 RobotAction、通用 contracts、policy stream、canonical action、位置/IK/安全语义 | 继续作为 sim/真机共享的控制语义层；下一步只补 proposal/arbiter/trace，不重复抽 control loop |
| teleop_runtime | 仍负责真机数采、TeleopSession、硬件观测、启动/恢复、fault/hold、position executor 和部署 frontend；入口文件仍较大 | 保留真机编排职责，不应成为 sim 依赖；逐步改成共享 core 的 composition layer |
| sim | MuJoCo/CubeLift/VectorExecutor 已存在，但任务和动作维度强绑定 | 目前更像仿真验证和 RL embryo，还不是完整 deployable sim |
| deployments | 有 OpenPI client、模型握手和 artifact 元数据 | 可以作为模型服务/策略客户端层 |
| dataset_contracts | schema 较完整，有 v2.2 PolicyIOLayout、v2.3 因果快照 | 还缺统一控制时间线、跨 source provenance 和按 component 的 action attribution |
| dataset_interface | 有 PolicyIO batch、图像和 state pipeline | 还缺多模态、时间窗口化的统一轨迹接口 |
| model | 当前没有真正的训练/策略/rollout 包 | 必须新增 |
| umi_runtime | 当前基本为空 | 不建议直接作为通用学习框架入口，除非明确其职责 |

已有设计文档其实已经指出了正确方向，尤其是：

- [sim/STRUCTURE_PLAN.md](/home/zc/HandMotion/Developing/DexTeleop/sim/STRUCTURE_PLAN.md)
- [sim/SIM_ENV_TODO.md](/home/zc/HandMotion/Developing/DexTeleop/sim/SIM_ENV_TODO.md)
- [sim/REAL2SIM_SIM2REAL_TODO_TMP.md](/home/zc/HandMotion/Developing/DexTeleop/sim/REAL2SIM_SIM2REAL_TODO_TMP.md)

当前最重要的不是再设计一套完全不同的结构，而是把这些文档中的边界真正落地。

---

## 二、结构性问题与当前收敛状态

### 1. 控制入口已经分层，但还没有统一 proposal/arbiter

当前审计后应将入口分为四类，而不是把它们都视为等价的生产控制循环：

- **正式 human/GUI 控制**：[teleop_runtime/debug/run_vive_rm65_real_teleop.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/debug/run_vive_rm65_real_teleop.py)，其控制 step 由
  [teleop_runtime/session.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/session.py) 统一承载。
- **headless/兼容数采**：[teleop_runtime/collection](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/collection)，复用 `TeleopSession`，不再发展为第二套正式 GUI 控制语义。
- **正式 policy deployment**：[teleop_runtime/deploy/run_policy.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/deploy/run_policy.py)，当前是带真机资源编排的正式 frontend；policy rate、action chunk、v2.3 decode、canonical action 和位置控制的共享语义已有较多部分抽到 `robot_motion`，入口仍负责真机资源与最终执行，不能让 sim 直接复用整个真机入口。
- **实验/脚手架**：`teleop_runtime/core/kernel.py`、`robot_motion/deploy/DeployRuntime` 和 `deployments/client/DeploymentClientRuntime`，目前不能当作正式入口。

原问题不是入口数量本身，而是多个入口分别持有 observation、action、dispatch、fault、record 的部分所有权。
前一批已将 GUI/headless 的 writer 生命周期和 `TeleopSession -> Rig` 最终 dispatch 收敛；本轮
`robot_motion` 又进一步抽出了跨入口可复用的 action/runtime/deploy/position 边界，但
observation/fault/authority 的跨 human、policy、replay 统一仍未完成。尤其是当前
`TeleopSession` 中各 retargeter 的 `TeleopCommand` 会在合并后覆盖同名 effector；如果在合并后
才记录 action，就无法知道某一侧或某个 hand slot 原本来自 human、policy 还是 replay。

当前抽取边界应按下面的事实理解：

- `robot_motion.action` 已有 canonical `RobotAction`、vector codec/layout、source protocol、
  `RobotActionResolver` 和通用 `ActionExecutor`/dispatcher seam；它们可以服务 synthetic/replay
  和后续 proposal adapter，但还不是 `ActionArbiter`，`run_policy.py` 也没有被强行改成通用
  `ActionExecutor`。
- `robot_motion.runtime` 已有 `TeleopObservation`/`TeleopCommand`、source/backend/retargeter/Rig
  interfaces、dispatch receipt、lifecycle cleanup、`BimanualRig` 和 causal history；
  `teleop_runtime` 中相应路径主要负责兼容和真机 composition。
- `robot_motion.deploy` 已有 `DeploymentObservation`、pose/frame adapter、artifact/session
  contract、`CanonicalDeployActionPipeline` 和 `PolicyDeployRuntime`。其中后者已经拥有
  policy-rate、prediction transaction、chunk queue、IK discard、queue pop，并通过注入的
  preparer 调用 canonical action boundary。
- `robot_motion.position_control` 和 `robot_motion.rm65` 已承载 reference/command contract、
  resampling/interpolation、trajectory policy、position audit、coupled IK/collision/hold 等共享
  语义；`teleop_runtime` 的 `PositionDeployExecutor` 仍是面向真实硬件的 adapter。

所以现在需要补的是“候选 proposal 的并行保留、authority/selection、per-component attribution
和 ControlTrace”，不是再抽一个 `PolicyRuntime`、`ActionExecutor` 或 position loop。

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
Replay 动作 ────┘             │
                             └── per-component ControlTrace / lineage
~~~

目标仍然是整个系统只能有一个最终动作拥有者和一个最终 dispatcher；当前只完成了
human/headless 路径的 dispatch seam，尚未实现 `ActionProposal`、`ActionArbiter` 和实时
per-component attribution。

#### 1.1 正式 policy deployment 适配 sim：复用控制核心，不复用真机入口

`teleop_runtime/deploy/run_policy.py` 目前是正式真机应用入口，不是一个可以直接被 sim import
的通用 runtime。它的 `main()` 仍包含 policy artifact/config 绑定、模型加载、真实机械臂/手/相机
初始化、ready/fault/hold、startup init、键盘交互、position executor、真实 dispatch 和 debug
记录；不过 policy observation view、v2.3 decode、policy-rate、prediction transaction、chunk
queue 和 action preparation 的共享边界已经抽到 `robot_motion.deploy`。即使增加一个 `--sim`
分支，也会让入口拥有两套资源生命周期和 fault 语义，最终形成另一种“入口内分叉”，而不是复用。

最佳路径是：**继续沿现有 `robot_motion` shared policy boundary 收敛，保留 `run_policy.py` 为
真机 frontend；未来 sim frontend 只注入自己的 observation source 和 execution backend。** sim 不应直接
依赖 `teleop_runtime`，因此不能让 `sim/...` 反向 import `teleop_runtime.deploy.run_policy`。

目标结构应是：

~~~text
real CLI / run_policy.py ───────┐
                                ├─ PolicyDeployRuntime（robot_motion）
sim CLI / sim/mujoco/deploy ────┘      ├─ policy schedule / chunk transaction
                                       ├─ observation pose-mode preparation
                                       ├─ policy.predict
                                       ├─ normalize + v2.3 pose decode
                                       ├─ ActionProposal / ActionArbiter
                                       ├─ CanonicalDeployActionPipeline
                                       ├─ shared position/reference clocks
                                       └─ backend adapter + ControlTrace
~~~

这里的 shared policy stream 以现有 `robot_motion/deploy/runtime.py` 的
`PolicyDeployRuntime` 为边界，而不是再新增第三个 runtime。旧的
`robot_motion.deploy.DeployRuntime` 与 `deployments.client.DeploymentClientRuntime` 仍只是窄
scaffold/兼容 seam；后续应合并职责或保留兼容 re-export，不能继续各自发展一份 single-step loop。

##### 1.1.1 `run_policy.py` 中哪些逻辑可以直接复用

| 当前逻辑 | sim 处理方式 | 复用判断 |
|---|---|---|
| `LoadedPolicy`/`load_policy` 通用 wrapper | 已迁移到 `deployments/client/policy.py` 的无硬件 policy loader，real/sim 注入同一 `PolicyLike` | 复用 loader 契约，不让 sim import teleop_runtime |
| policy artifact/layout/frame contract 绑定 | 由 shared runtime 接收已解析的 artifact binding；real/sim 使用同一校验 | 复用契约和校验，不复制 v2.3 分支 |
| `DeploymentObservation` | sim 直接生成同类型 observation，保持 state/image/tactile/last_action layout | 原样复用类型 |
| `prepare_observation_for_policy_pose_mode` | 使用 sim measured TCP 和同一 `PoseFrameContract` | 原样复用 |
| `normalize_policy_output` | 同一 policy 输出、chunk shape、layout、padding 规则 | 原样复用 |
| `convert_normalized_action_chunk_to_absolute` | 使用 sim 当前实测 TCP，固定 chunk anchor | 原样复用 |
| policy chunk 的预测、替换、padding、sequence | `robot_motion.deploy.runtime.PolicyDeployRuntime`、`robot_motion.deploy.actions.ActionChunkBuffer` | `run_policy` 只注入 policy、clock 和 queue policy；后续 source adapter 提交已解码 proposal |
| `CanonicalDeployActionPipeline` | safety/filter/resolver 输入仍是 absolute runtime action | 原样复用 |
| `PositionControlRates`/position trajectory/audit | `robot_motion.position_control`、`robot_motion.rm65`；真机 executor 只做硬件适配 | 原样复用共享语义，不复用硬件 callback |
| `ArmStateReader`、`build_arm_backends` | 改为 `MujocoDeploymentObservationSource` | 不复用真机实现 |
| `build_camera_sources` | 改为 scene render/frame provider，使用 sim time 戳 | 不复用 RealSense 初始化 |
| `PositionDeployExecutor` 的真实 callback | 拆出共享 reference/interpolation 语义，sim 使用自己的 sink | 不直接复用硬件 executor |
| `dispatch_teleop_command`、CAN/串口/SDK ready 检查 | 由 `MujocoDeploymentExecutor` 生成 sim receipt | 不复用 |
| startup init/键盘/真实 recovery | 由 frontend/backend 生命周期适配 | 只复用状态语义，不复用实现 |
| `DeploymentDebugLogger` | 仍由 real frontend 持有；后续抽取 typed event/trace sink | 复用事件契约，不绑定硬件 |

这意味着真正需要复制的只有两类东西：

1. sim 如何从 `MujocoScene` 读取 measured state、TCP、图像和 tactile；
2. sim 如何把 canonical command 变成 qpos/reference，并返回 receipt。

policy 的输入、relative decode、action chunk、Arbiter、canonical safety 和数据 lineage 都不应
复制。

其中无硬件的 `LoadedPolicy`、`HoldPolicy` 和 module/file loader 已经迁移到
`deployments/client/policy.py`；`teleop_runtime/deploy/policy.py` 现在只保留兼容 re-export，
因此 sim 可以直接依赖 `deployments.client.policy`，也可以注入外部 `PolicyLike`，不需要依赖
`teleop_runtime` 或 `run_policy.py`。这次迁移保留了现有 replay 所需的 prediction transaction
方法，没有改变 real deployment 的模型调用语义。artifact pose-mode binding 也已提升到
`robot_motion.deploy.session_contract.resolve_policy_action_pose_mode_from_artifact()`；
`run_policy.py` 只保留 raw argv 的“CLI 是否显式指定”判断和兼容 wrapper。real frontend 仍负责
把 argparse/config 转成输入，不再持有 artifact agreement 的业务规则。

##### 1.1.2 Shared `PolicyDeployRuntime` 的当前边界

当前不再把 `PolicyDeployRuntime` 描述成只完成单步表示适配的“未来 runner”。它已经承载完整
的 policy stream 状态：policy-rate 判定、prediction transaction、chunk queue、恢复期丢弃、
queue pop/lookahead，以及通过注入的 `ActionPreparer` 调用 canonical action boundary。它仍然
不是一个独立生产入口，也不负责 observation source、fault gate、硬件执行或 trace writer。

~~~text
real frontend observation/lifecycle
  → DeploymentObservation (immutable control cut)
  → PolicyDeployRuntime.should_predict(now, interval, min_queue)
  → PolicyDeployRuntime.predict_and_queue(...)
       → prepare policy observation
       → prediction transaction
       → policy.predict
       → normalize output
       → v2.3 relative → absolute decode
       → chunk queue replace/append
  → PolicyDeployRuntime.prepare_next_action(
       preparer=CanonicalDeployActionPipeline)
       → queued absolute action
       → injected canonical preparation result
  → real frontend fault/hold/position/reference/dispatch
~~~

当前已经落地的共享边界是：`robot_motion/deploy/runtime.py` 中的
`PolicyDeployRuntime`/`PolicyDeployRuntimeConfig` 统一封装
`prepare_policy_observation()` 和 `normalize_and_decode()`，前者调用现有
`prepare_observation_for_policy_pose_mode()`，后者调用现有 `normalize_policy_output()` 和
`convert_normalized_action_chunk_to_absolute()`。`run_policy.py` 已经通过这个边界完成
model-view preparation 与 v2.3 decode；同时 `PolicyDeployRuntime` 已接管 policy-rate decision、
prediction transaction、`ActionChunkBuffer` 生命周期、IK recovery discard、queue pop/lookahead
和调用注入的 canonical preparer。canonical filter/resolver 的共享实现位于
`robot_motion.deploy.canonical_action`，但其 real 配置、fault latch、position/reference sink 和
最终 hardware dispatch 仍由 `run_policy.py` 编排。

因此当前已统一“policy 输入/输出 → 绝对 action → canonical preparation seam”的共享部分，
但还没有实现 `ActionProposal`、`ActionArbiter`、per-component attribution、统一 observation/
fault gate、reference sink 和 `ControlTrace` writer。

`PolicyDeployRuntime` 的职责必须限定为 policy stream 与 injected preparation seam，不拥有：

- 真实设备对象、MuJoCo `model/data` 或 GUI/键盘；
- wall-clock、sleep 或整数频率比假设；clock 由 frontend/backend 注入；
- 真机 enable/disable、home、servo、emergency stop 或硬件 fault 清除；
- episode writer 的具体线程实现；未来只向 typed trace sink 提交 event；
- raw relative action 的几何实现；只调用已有 `pose_adapter`；
- policy artifact 文件路径的 CLI 解析；只接收已验证的 artifact binding。

旧 `DeployRuntime.step()` 仍只有“单次 predict → prepare → dispatch”的最小 scaffold，缺少
action chunk、v2.3 decode、prediction rollback、fault/hold、position/reference 和 trace；不能
把它当作最终 runtime。它已有的 `source/policy/preparer/dispatcher` seam 可继续作为兼容 API，
但生产 policy stream 统一以 `PolicyDeployRuntime` 为唯一共享实现，不能让两个 runtime 并行演进。

##### 1.1.3 真机 frontend 如何收敛

`teleop_runtime/deploy/run_policy.py` 的最终目标只保留真机 composition 职责；当前已经把 policy
observation preparation、prediction/queue transaction、output decode 和 queued-action preparation
委托给 `PolicyDeployRuntime`，而共享 canonical pipeline 的实现也已位于 `robot_motion.deploy`。
入口仍负责真实配置、生命周期、安全锁存和最终执行，便于保持现有真机行为：

~~~text
CLI/config/artifact parsing
  → build real policy client
  → build RealDeploymentObservationSource
  → build canonical safety/preparer and real action sink
  → build real startup/recovery/keyboard/debug observers
  → PolicyDeployRuntime.run(...)
~~~

现有真实链路的行为保持不变：

- `ArmStateReader`、hand polling、RealSense source 继续由 real frontend 构造；
- startup init、ready confirmation、IK recovery 和 manual rearm 仍由真机 lifecycle/adapter
  提供，并在 real frontend 内形成 readiness/fault/hold gate；不能假设这些状态已经被
  `PolicyDeployRuntime` 统一拥有；
- direct/position 两种执行模式仍可存在，但它们都是 `DeploymentActionSink` 的实现；
- 真机 `PositionDeployExecutor` 中的 arm backend callback、CAN 状态检查和 hold callback 留在
  real sink，不进入 shared policy loop；
- 当前 policy output 的 queue、prediction transaction、v2.3 decode 和 queued-action preparation
  已由 shared runtime 承载；`CanonicalDeployActionPipeline` 的实现由 `robot_motion.deploy`
  复用，但真实入口仍负责 safety 配置、fault latch、position executor、debug 和 dispatch，
  不再维护另一份 predict/normalize/queue 代码。
- `robot_motion.action.ActionExecutor`/`RigCommandDispatcher` 已可用于通用或 synthetic/replay
  seam，但不应为了表面统一而直接替换当前 `run_policy` 的 real execute 分支；position mode、
  hand/arm 组合 receipt、hold 和硬件故障语义仍属于 real frontend adapter。

##### 1.1.4 sim frontend 应怎样实现（暂缓）

本阶段暂不实现 sim frontend；以下内容只保留为后续设计约束，不纳入本次迁移和验收范围。

建议新增的是 `sim/mujoco/deploy/` 或 `sim/mujoco/adapters/deployment.py`，而不是
`sim` 直接调用 `run_policy.py`：

~~~text
MujocoDeploymentObservationSource
  ├─ scene.data.time 作为 deterministic observation timestamp
  ├─ MuJoCo measured qpos/qvel 与 FK TCP pose
  ├─ scene.render/render_depth 作为带 sim timestamp 的 image provider
  ├─ simulated tactile/contact（没有则显式 invalid）
  ├─ last_action 与 state_valid
  └─ finite/collision/scene readiness faults

MujocoDeploymentActionSink
  ├─ 接收 canonical absolute RobotAction/TeleopCommand
  ├─ coupled pose → qpos planner（复用 robot_motion RM65 planner）
  ├─ PositionReferenceTimeline + JointCommandInterpolator
  ├─ MuJoCo arm/gripper joint sink
  ├─ physics tick / command tick scheduling
  └─ SimDispatchReceipt + measured_state_id
~~~

sim observation 必须与 real 使用相同的 `DeploymentObservation` 和 observation contract：

- arm `current_qpos` 和 `actual_tcp_pose` 都来自当前 scene measured state；relative policy 的
  anchor 只能来自该 state，不能使用上一条 target/reference；
- TCP 必须使用显式 side-specific tool profile 和 `tool_tcp`/arm-base frame，不能因为是 sim
  就把 controlled body 或 arm_link6 默认为 TCP；
- camera frame 的 timestamp 使用 `scene.data.time` 或明确的 sim clock，并遵守
  `latest_not_after_observation`，不能使用真实 wall-clock；
- 没有 tactile/hand sensor 时必须写 `valid=false`/fault 或按 artifact contract 禁用该 modality，
  不能填零后伪装成有效 observation；
- `state_valid`、frame contract hash、policy artifact hash、simulation contract hash 都进入
  observation metadata 和 ControlTrace。

sim action sink 的关键是复用已有语义、替换 backend：

1. `CanonicalDeployActionPipeline` 仍先产生带 safety metadata 的 absolute `RobotAction` 和
   `TeleopCommand`；它不直接写 MuJoCo qpos。
2. Cartesian arm command 进入 group-level `RM65CanonicalIK`/
   `RM65PositionCollisionPlanner`，保持与真机 position path 相同的 coupled、frame、joint-limit
   和 hold 语义；不能让左右臂各自调用一个独立 IK 后再拼接。
3. planner 的 `q_ref` 进入已有 `PositionReferenceFrame`/`ReferenceTimeline`，由
   `JointCommandInterpolator` 生成 profile-defined command tick；sim physics 只消费最终
   qcmd/reference，不在 physics loop 重新做 policy、relative decode 或 authority decision。
4. `MuJoCoArmBackend` 可以作为已经规划好的 joint-vector leaf sink；它不能接收 raw
   `cartesian_pose`，仍由 group planner 先完成 pose→qpos。
5. 现有 `DirectQposDebugController` 只作为 `task_native/debug` 或早期 kinematic bridge，不能
   被标成 deployable sim 的默认 policy executor。它不模拟真实 servo/target tracking，且会绕过
   deployable position-control 语义。

##### 1.1.5 sim 与 real 的复用验收标准

不能以“两个入口都能调用 `policy.predict`”作为复用完成。至少要验证以下等价关系：

| 阶段 | 要求 |
|---|---|
| artifact binding | 同一 policy artifact、layout hash、pose chunk/frame contract 在 real/sim 都通过同一校验 |
| observation | 同一 state layout、frame、last_action、causal cutoff 和 validity 语义 |
| v2.3 decode | 给定同一 observation/anchor/chunk，relative→absolute 的结果逐元素或按 SE(3) tolerance 等价 |
| chunk scheduling | 同一 policy/reference profile 下，keyframe、padding、queue replace/append 结果一致 |
| safety | 给定相同 absolute observation/action，`CanonicalDeployActionPipeline` 的 filter/result 一致 |
| arbitration | 同一 proposal snapshot 的 selected source、decision reason、component mask 一致 |
| position | reference/IK 的 contract、epoch/sequence、lineage 一致；qcmd 允许 backend 数值差异但必须可解释 |
| execution | real 产生 hardware receipt，sim 产生 `SimDispatchReceipt`，二者都能关联 command/reference/measured id |
| data | ControlTrace 字段一致；sim 不因没有真实 SDK 而退回只写一个 action vector |

推荐的 golden test 是：同一 `DeploymentObservation` 和同一 policy output 分别经过 real/sim
前端的 shared runner，比较 `policy model view`、decoded absolute action、safety result、
proposal/decision lineage；只有最后的 backend receipt 和 measured state 允许不同。之后再用
实际 MuJoCo observation 做 deterministic replay，验证不存在“sim 一套相对 anchor、real 一套
相对 anchor”的隐性分叉。

##### 1.1.6 分阶段落地顺序

1. **[已完成] 抽出 policy representation 与 artifact binding。** 已从 `run_policy.py` 抽出
   model-view preparation、normalize/decode 和 artifact pose-mode agreement；由
   `robot_motion.deploy.runtime.PolicyDeployRuntime` 与
   `robot_motion.deploy.session_contract` 调用现有实现。无硬件 policy loader 也已移到
   `deployments/client/policy.py`，旧 import path 仅作兼容 re-export。
2. **[已完成] 抽出真机 policy stream 状态。** `PolicyDeployRuntime` 已接管 policy-rate decision、
   prediction transaction、chunk queue replace/append、IK recovery discard、schedule reset、
   queue pop/lookahead 和注入的 canonical-preparation seam；`robot_motion.deploy.canonical_action`
   提供可复用的 filter/resolver 实现。`run_policy.py` 继续持有真机 lifecycle、fault/hold、safety
   fault latch、position executor、debug 和 hardware dispatch。
3. **[下一步] 收敛 real frontend 的剩余编排。** 把 observation source snapshot、readiness/fault
   gate、startup/recovery 状态和 position/reference sink 逐一整理成真机 frontend adapter；不把
   硬件对象、wall-clock 或 operator keyboard 放入 `robot_motion`，也不在本阶段实现 sim。
4. **[下一步] 接入 ActionArbiter 与 attribution。** 在已解码的 absolute `RobotAction` 和
   canonical preparation 之间生成 proposal；先记录 raw model-view、decode anchor、selected
   source 和 per-component lineage，再切换执行权。
5. **[后续] 真机 shadow 与人工接管。** 先 policy shadow，再显式 human override；policy-only
   需要在 ActionArbiter、ControlTrace 和安全验收完成后另行开启。

因此，sim 不是 `run_policy.py` 的一个新 backend flag，而是同一个 shared policy-control core
的另一个 frontend/backend 组合。复用比例最高、边界最清晰，也符合 `sim` 不依赖
`teleop_runtime` 的目录约束。

### 2. RuntimeKernel 仍不能直接晋升为共享内核

[teleop_runtime/core/kernel.py](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/core/kernel.py) 的思路可以作为实验参考，但它现在有三个限制：

- 位于 teleop_runtime，sim 无法自然依赖它。
- 一个 step 只有一个 RobotActionSource，不能表达 human/policy/replay 同时存在时的 authority。
- 没有明确区分 policy rate、human input rate、reference rate、command rate 和 physics rate。

因此当前不把它接入正式 GUI、headless 或 deploy loop，也不把它与 `TeleopSession` 并列为第二套生产状态机。

共享部分确实应继续位于 `robot_motion`，但当前不应把 `RuntimeKernel` 整体搬过去，也不应
因为 `robot_motion/runtime` 已经存在就立即新增一套 kernel/scheduler。当前已经抽出的基础边界是：

~~~text
robot_motion/runtime/
├── contracts.py       # TeleopObservation/TeleopCommand/effector contracts
├── interfaces.py      # source/backend/retargeter/Rig protocols
├── dispatch.py        # receipt/error contracts
├── lifecycle.py       # startup/cleanup failure aggregation
├── causal_history.py  # causal observation history
└── rig.py             # shared BimanualRig/CompositeRig
~~~

后续只有在 `ActionProposal`、authority、scheduler、ControlTrace 契约冻结后，才评估增加
`scheduler.py` 或 `trace.py`；proposal/authority/arbiter 属于 `robot_motion.action`，不放回
`teleop_runtime/core/kernel.py`。`teleop_runtime/core` 继续作为 experimental/synthetic 兼容层，
不能与 `TeleopSession` 或 policy stream 并列为第二套生产状态机，也不能让 sim 反向依赖
`teleop_runtime`。

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
- observation/sensor validity；内部 fault 原因默认不进入 model view
- artifact 明确定义的 history/last-action 字段

现有 [robot_motion/deploy/observation.py](/home/zc/HandMotion/Developing/DexTeleop/robot_motion/deploy/observation.py) 可以继续演进。

### 2.1 Policy view 与 control plane 的可见性边界

必须把“模型需要什么输入”与“控制系统为了安全、执行和训练记录产生什么状态”分成两条数据面。
模型输出的是 proposal/intent，不是对 `ActionArbiter` 以下链路的直接控制；`ActionArbiter` 以下
的实现细节对 actor 模型不可见，也不应成为 actor 的输入契约：

~~~text
PolicyIOLayout + allowlisted DeploymentObservation
  → model-facing observation view
  → policy output
  → existing normalize / v2.3 decode
  → absolute RobotAction proposal
  ║  visibility firewall：不把 control-plane metadata 回灌给 policy
  → ActionArbiter
  → selected action
  → safety / IK / collision / trajectory / reference
  → command / dispatch / measured alignment
  → ControlTrace + reward/label adapter（训练侧读取）
~~~

模型可以看到的内容只由 `PolicyIOLayout` 和已冻结的 observation contract 白名单决定，通常包括
当前 measured state、图像、tactile、显式 validity/age，以及 artifact 已声明的 policy history。
它不能直接看到或依赖：

- `ActionArbiter` 的 `control_mode`、authority epoch、source kind、proposal id、selected owner、
  rejected/expired proposal、human takeover 或 per-component mask；
- safety filter、IK/collision/planner 的 clamp、hold、降级、失败原因和内部 solver 状态；
- `PositionReferenceFrame`、`PositionCommandTick`、command/dispatch receipt、hardware status、
  measured-state id、writer queue 或 `ControlTrace` id；
- `expert_label_mask`、`policy_behavior_mask`、reward、intervention reason 和训练标签。

下一控制 cut 的 measured state 可以自然反映实际执行结果，这是环境反馈，不等于把 Arbiter 或
safety 的内部决策暴露给模型。尤其是 `last_action` 必须保持现有 artifact contract 的明确语义：
不能在未声明的情况下把 selected、safe、commanded 或 measured action 任意替换进去；如果未来
需要提供实际执行反馈，必须作为独立、显式版本化的 observation field，而不是复用含义不清的
`last_action`。

当前实现有一个需要在下一步修正的边界风险：
`robot_motion.deploy.observation.DeploymentObservation.model_input()` 目前会直接带出 `faults`
和完整 `metadata`。这会让 runtime-private 信息有机会穿过 model wrapper。后续应改为
`PolicyObservationView`/allowlist projection，只复制 PolicyIO 声明的 tensor、validity 和 history
字段；`ActionAttributionFrame`、ArbitrationResult、SafetyResult 和 trace metadata 必须走旁路
ControlTrace，禁止放入 policy observation 的任意 metadata。

这条隔离不妨碍训练：actor 只消费 policy view，critic/reward/DAgger exporter 在训练侧读取
ControlTrace；若某个算法确实需要 owner/mask 作为 auxiliary input，必须另立 training-only
contract，不能悄悄污染部署 actor 的 observation schema。

### 2.5 v2.3 相对控制链路：复用表示适配层，不扩展仲裁层

v2.3 的 relative action 不能被理解为另一种需要重新实现的 runtime 控制语义。当前代码已经
形成了一条可复用的表示转换链路，最佳方案是把它固定为 **model-facing representation adapter**：
模型可以输出 relative，数据集可以提供 relative 视图，但进入 `ActionArbiter` 之前必须已经
还原为带有明确 frame、单位和 active component 的绝对 `RobotAction`。这样 human、policy、replay
和 scripted source 才能在同一个语义层竞争控制权。

#### 2.5.1 当前 v2.3 链路和唯一事实来源

当前应把下面这条链路视为 v2.3 的基线，不要另起一个 `RelativeActionPipeline`：

~~~text
v2.3 dataset rows
  (absolute pose + actual TCP anchor + PoseFrameContract)
    → dataset_interface.pose_action / pose_chunk
    → PoseActionChunkContract
    → model-facing action chunk
       (absolute or relative, with action_is_pad)
    → policy.predict
    → PolicyDeployRuntime.prepare_policy_observation
    → PolicyDeployRuntime.normalize_and_decode
       → normalize_policy_output
       → robot_motion.deploy.pose_adapter
          (measured TCP anchor, frame conversion, relative → absolute)
    → absolute runtime RobotAction
    → CanonicalDeployActionPipeline
    → TeleopCommand / PositionReferencePublisher
    → ReferenceTimeline / JointCommandInterpolator
    → dispatch receipt → measured state
~~~

这条链路中各部分的职责应冻结如下：

| 语义 | 现有唯一 owner | ActionArbiter 的关系 |
|---|---|---|
| v2.3 物理存储 | `dataset_contracts.v23`、现有 writer | 不改变 parquet 的 absolute storage 约定 |
| 绝对/相对模型视图 | `PoseActionChunkContract`、`dataset_interface.pose_action`、`dataset_interface.pose_chunk` | 作为 policy/replay source 的输入契约 |
| SE(3) 相对变换 | `robot_motion.relative_action` | Arbiter 不实现、不调用 anchor math |
| runtime raw/canonical desk frame 转换 | `robot_motion.deploy.pose_adapter`、`PoseFrameContract` | 在 proposal 生成前完成 |
| policy chunk 的预测、替换、padding、sequence | `robot_motion.deploy.runtime.PolicyDeployRuntime`、`robot_motion.deploy.actions.ActionChunkBuffer` | `run_policy` 负责注入真实 policy/clock/gate；source adapter 后续只提交已解码 proposal |
| 绝对动作语义 | `RobotAction`、`PolicyIOLayout`、`ArmActionSpec` | Arbiter 只接收这一层 |
| safety、IK、coupled planner | `CanonicalDeployActionPipeline` 及现有 resolver/planner | 在 selected proposal 之后统一执行 |
| reference/command 时序 | `PositionControlRates`、`PositionReferenceFrame`、`ReferenceTimeline`、`PositionCommandTick` | 传递 decision/lineage，不新建 timeline |
| dispatch 和 measured 对齐 | 现有 `TeleopCommand`、receipt、`PositionAuditEvent`、sample assembler | 作为 trace 的后续阶段补全 |

特别要保留三个容易被混淆的边界：

1. **v2.3 dataset canonical `desk` 不等于历史 VIVE virtual-desk retarget frame。**
   policy 的 canonical desk 与真机 runtime raw/base frame 的转换必须由
   `PoseFrameContract` 明确声明，不能通过字段名或矩阵形状猜测。
2. **policy relative 与 human `delta_clutch` 不是同一种 relative。** policy chunk 使用
   chunk 起点的实测 TCP anchor，human retargeter 使用 tracker/home/clutch/session anchor。
   二者都应在 source adapter 内还原为绝对 `RobotAction`，但不得共用一个含义模糊的
   `relative_mode` 分支或相互替换 anchor。
3. **`last_action` 不等于上一条 command。** 现有 v2.3 contract 对 action chunk 和
   `last_action` 有独立的 anchor/转换语义；在 artifact 尚未明确支持 mixed mode 之前，保留
   `run_policy` 当前要求 `action_pose_mode == last_action_pose_mode` 的 fail-closed gate，
   不在 Arbiter 中临时放宽，也不通过条件分支伪造兼容。

#### 2.5.2 ActionArbiter 的准确插入点

推荐的最终 source flow 是：

~~~text
human PoseToArmRetargeter / delta_clutch
  → absolute runtime RobotAction
  → human ActionProposal

policy v2.3 relative chunk
  → existing pose chunk contract
  → existing pose_adapter decode
  → absolute runtime RobotAction
  → policy ActionProposal

v2.3 replay / scripted source
  → same dataset/model-view adapter as policy
  → absolute runtime RobotAction
  → replay/scripted ActionProposal

all ActionProposal
  → ActionArbiter
  → selected absolute RobotAction
  → existing safety / IK / planner / reference / interpolation
  → existing TeleopCommand / Rig or MuJoCo backend
~~~

因此 `ActionProposal.action` 的类型仍是绝对的 typed `RobotAction`。proposal 可以携带一份
representation/decode provenance，但不能把 raw relative vector、anchor 或归一化参数作为
Arbiter 的输入，让 Arbiter 现场推导 absolute action。这样可以避免以下重复实现：

- 在 `robot_motion/action/arbiter.py` 再实现一套 SE(3) relative conversion；
- 在 safety 层分别为 absolute 和 relative action 做两次约束；
- 在 150 Hz command loop 里重新解码或重新 anchor；
- 在 sim 里另建一个 relative-to-qpos executor；
- 在数据写入侧从最终 command 数值反推 source。

Arbiter 只负责“选择哪个已经具有相同语义的 proposal”，不负责“把不同表示变成相同语义”。
这也是 human、policy 和 replay 可以共享同一个 `component_schema_hash`、selection group 和
lineage contract 的前提。

#### 2.5.3 不新增数学模块，只增加一个轻量 decode receipt

现有 `pose_adapter` 已经能完成部署所需的关键工作：从 immutable observation 读取
`actual_tcp_pose`，按 `PoseFrameContract` 做 frame 转换，并调用
`robot_motion.relative_action.convert_arm_pose_chunk`。建议只增加一个很薄的 typed
`ActionRepresentationTrace`（可放在已有 `robot_motion/action/attribution.py`，不单独创建
runner/executor），由 source adapter 产生并由 `ActionProposal` 引用：

~~~text
ActionRepresentationTrace
├── representation_trace_id
├── input_action_mode: absolute / relative
├── input_last_action_mode
├── output_action_mode: absolute_runtime
├── input_frame / output_frame
├── pose_chunk_contract_hash
├── pose_frame_contract_hash
├── chunk_id / keyframe_index / valid_action_count
├── action_is_pad / valid_mask
├── anchor_id / anchor_source: current_actual_tcp
├── anchor_observation_id / anchor_measurement_timestamp
├── anchor_pose_by_side
├── anchor_frame
├── conversion_status: decoded / invalid / stale / blocked
├── conversion_error_code
└── metadata: normalization/layout/source artifact ids
~~~

其中 raw model output、decoded absolute action 和 anchor 不一定都复制进每一个高频 command；
它们应在 policy keyframe 的 proposal/trace event 中保存，后续 `reference_segment_id`、
`command_id` 和 `dispatch_receipt_id` 只引用该 `representation_trace_id`。

必须遵守以下 anchor 规则：

- policy relative action 的一个 chunk 使用 chunk 起点的 **实测** TCP anchor，并对该 chunk 的
  所有有效 keyframe 固定；不能用上一条 commanded pose，也不能在每个 150 Hz tick 重新 anchor；
- anchor 缺失、过期、frame hash 不匹配或非 finite 时，decode 结果为 invalid/hold，不能回退
  到当前 commanded target 或把零 relative action 当作安全动作；
- `last_action` 按当前已有 contract 的 `current_actual_tcp` 规则独立转换，并在 trace 中明确
  `last_action_anchor_id`，不能与 action chunk anchor 混为一个字段；
- `action_is_pad` 的 padded row 不生成新的 proposal，也不能作为 DAgger expert label 或 RL
  behavior action；`valid_action_count` 和 padding mask 必须沿 chunk、proposal、trace 传递；
- 解码只做 representation/frame conversion，不在这里做第二次 safety、IK 或 position
  interpolation。解码后的绝对动作只经过一次 canonical safety/planner pipeline。

#### 2.5.4 最少新增、最大复用的接入方式

建议不是新建一套 `PolicyActionRuntime`，而是继续使用现有 `PolicyDeployRuntime` 作为 policy
stream 的共享边界，在 `run_policy` 和 human source adapter 外面补 proposal/arbiter 编排。
当前 `PolicyDeployRuntime` 已经不是只包住两个函数的薄 facade：它拥有 policy schedule、prediction
transaction、chunk queue、recovery discard、queue pop/lookahead，并通过 `ActionPreparer` 接入
canonical preparation；它仍不拥有 safety 配置、reference、hardware dispatch 或 authority：

~~~text
PolicyDeployRuntime (shared policy stream boundary)
  ├── bind artifact → PoseActionChunkContract + frame hashes
  ├── should_predict(...) / predict_and_queue(...)
  │      ├── prepare_policy_observation(...)
  │      ├── policy.predict + prediction transaction
  │      ├── normalize_and_decode(...)
  │      │      └── existing normalize + pose_adapter conversion
  │      └── ActionChunkBuffer replace/append
  ├── prepare_next_action(preparer=CanonicalDeployActionPipeline)
  │      └── queued absolute action → injected canonical preparation result
  ├── attach ActionRepresentationTrace (下一阶段)
  └── emit one absolute RobotAction proposal per valid keyframe (下一阶段)
~~~

它不是新的控制 loop，也不拥有 authority、safety、reference 或 dispatch。当前 policy stream
boundary 已完成；`ActionRepresentationTrace`、proposal emission 和 outer shared runner 仍按
以下顺序落地：

1. **冻结 v2.3 contract。** 继续使用现有 v2.3 absolute storage、`PoseFrameContract`、
   `PoseActionChunkContract` 和 artifact hash；先完成 dataset row → model chunk → runtime
   decode 的 golden round-trip。对 absolute→relative→absolute 做 SE(3) 误差、padding、episode
   boundary 和 frame mismatch 测试。
2. **[已完成] 把 policy stream 和 pose adapter 包成共享 boundary。**
   `PolicyDeployRuntime` 复用 `relative_action`、`pose_adapter` 和现有 normalization，已经接管
   chunk/keyframe queue、prediction transaction、policy rate 和 recovery discard；不再由
   `run_policy` 维护另一份这些状态。尚未完成的是把 chunk/keyframe 标识、anchor 和
   `ActionRepresentationTrace` 作为 typed proposal lineage 接出。
3. **先生成 proposal，再改变执行权。** policy decoder 和 human retarget adapter 都输出
   absolute `RobotAction` proposal；第一阶段以 observe-only 方式把 proposal/decision/trace
   接到现有 GUI 主线，实际 selected/dispatch 仍走当前 `TeleopSession` seam。
4. **把 selected action 接入已有 canonical pipeline。** Arbiter 输出的 absolute action 进入
   已抽到 `robot_motion.deploy.canonical_action.CanonicalDeployActionPipeline`，再进入现有
   `RobotActionResolver`、`PositionReferencePublisher`、`ReferenceTimeline` 和
   `JointCommandInterpolator`。不要让 Arbiter 直接调用 `BimanualRig`，也不要把 position
   executor 复制一份到 `robot_motion/action`。
5. **让现有 position/audit seam 承载 lineage。** `robot_motion.position_control` 已拥有
   reference/command contracts、resampling/interpolation、trajectory policy 和 position audit；
   将 `control_tick_id`、`decision_id`、
   `proposal_id_by_component`、`representation_trace_id`、anchor id、contract hashes 和
   source/mask 作为 typed metadata/contract 从 selected action 传到
   `PositionReferenceFrame`、`PositionCommandTick`、`PositionReferenceAuditEvent` 和
   `CommandDispatchAuditEvent`。现有 `PositionReferencePublisher` 是唯一推荐的 reference
   边界；它目前已有 causal timeline、sequence、epoch 和 audit observer，应扩展而不是旁路。
6. **最后才切换真机 owner。** 先做 policy shadow，再做显式 human override 和允许的 group
   override；policy-only 先在 sim/dry-run 验证。每一步都必须能够从 trace 重建 raw relative
   intent → decoded absolute → selected → safe → reference → command → receipt → measured。

#### 2.5.5 为什么这是当前的最佳路径

它同时满足三个目标：

- **最少重复：** v2.3 已有的 contract、几何转换、anchor 读取、chunk transaction、位置控制
  和 writer lifecycle 都继续作为唯一 owner；新增的只是 proposal provenance 和跨层 lineage；
- **最小风险：** relative decode 在进入 Arbiter 前完成，Arbiter 仍是纯 absolute-action 选择器，
  不改变现有 safety/IK/dispatch 的输入语义；可以先 observe-only，随时回退到当前 GUI 主线；
- **可训练、可重放：** raw policy intent、decode anchor、component selected source、safe/
  commanded/executed 值都在同一个 causal tick 下可关联，DAgger 与 RL 不需要从最终向量猜测
  人类介入或 policy behavior。

不推荐的方案及原因：

| 方案 | 问题 |
|---|---|
| Arbiter 直接接收 relative vector 并自行找 anchor | 侵入 runtime ownership，容易用错 observation/commanded anchor，且 sim/real 会出现两套解码 |
| 在 `CanonicalDeployActionPipeline` 之外另建 relative safety pipeline | 同一动作被重复过滤，safe 结果和 attribution 语义不一致 |
| 每个 150 Hz tick 重新把 relative 转 absolute | 改变 chunk 语义，产生漂移和不可重放的 anchor |
| 为 v2.3 新建 relative parquet layout | 与当前 absolute storage + model view contract 冲突，增加 writer/reader 分叉 |
| 新建 `RelativeReferenceTimeline` 或 policy 专属 position executor | 重复已有 `ReferenceTimeline`、interpolator 和 receipt/audit 链路 |
| 通过最终 command 与 policy/human 数值差异推断来源 | 对 clamp、IK、coupling、hold 和插值不可逆，不能生成可靠 DAgger label |

### 3. ActionProposal：统一在 RobotAction 语义层，不在 TeleopCommand 层仲裁

当前代码有两个不同层级的动作对象：

- `RobotAction` 是跨 collection、replay、policy、sim、real 的语义动作，包含
  `action_space`、arm pose/qpos、hand qpos/end-effector target、active effectors、frame 和 source metadata。
- `TeleopCommand` 是已经接近 Rig/backend 的 transport command，按 effector 携带具体的
  `cartesian_pose`、`joint_vector` 或 end-effector command。

因此 `ActionArbiter` 不应接收原始 policy vector，也不应直接合并多个
`TeleopCommand`。推荐的边界是：

~~~text
human retargeter / policy / replay
    → source adapter
    → RobotAction
    → ActionProposal
    → ActionArbiter
    → selected RobotAction
    → safety / resolver / reference executor
    → TeleopCommand
    → Rig 或 MuJoCo backend
~~~

当前 human retargeter 仍直接产生 `TeleopCommand`，而 policy 的 v2.3 链路应明确为
`policy output -> existing pose_adapter decode -> absolute RobotAction ->
CanonicalDeployActionPipeline -> TeleopCommand`。所以第一阶段不能强行把旧 human command
当成 canonical proposal；应增加一个窄的 `TeleopCommand -> RobotAction` 适配器，并用 golden
round-trip 测试确认语义不变。policy source 则复用 2.5 节的 decoder，不应在 Arbiter 或
`CanonicalDeployActionPipeline` 中再次解析 relative action。

适配器至少要明确支持：

- `cartesian_pose`：矩阵/pose 转为 `RobotAction.arm_pose_by_side`，保留 reference frame、
  controlled frame 和 `ArmActionSpec`；
- arm `joint_vector`：转为 `arm_qpos_by_side`；
- hand `joint_vector`：转为 `hand_qpos_by_side`，保留 active mask 和 hand spec；
- `end_effector_continuous/discrete`：转为 `EndEffectorTarget`；
- 未识别的 command type：生成 invalid proposal 并 fail closed，不能绕过 Arbiter 继续直发。

`RobotAction` 当前虽然是 frozen dataclass，但内部 mapping/array 仍可能被 producer 修改。
在 proposal 边界必须做 snapshot/copy，并为 proposal 分配稳定的 `proposal_id`；不能让异步
policy chunk 或 GUI 临时 buffer 在仲裁之后改变已选动作。

建议的最小不可变契约为：

~~~text
ActionProposal
├── proposal_id
├── action: absolute runtime RobotAction
├── source_kind: human / policy / replay / scripted / recovery
├── source_id
├── source_timestamp
├── received_timestamp
├── observation_timestamp
├── valid_from
├── valid_until
├── sequence
├── causal_observation_id / causal_cutoff
├── requested_effectors
├── component_schema_hash / component_coverage
├── representation_trace_id
│   # raw relative/absolute intent、frame、contract hash、measured anchor 的引用
├── confidence                  # 只记录，第一版不用于抢占
├── inference_latency
├── authority_request           # optional explicit takeover request
├── validity: valid / stale / invalid / blocked
└── metadata
~~~

时间字段必须在同一个 control clock 语义下比较。设备原始时间、VIVE 时间和网络接收时间
可以保留在 metadata，但不能未经 clock-sync 转换就互相比较。`valid_until` 是硬边界：
过期 proposal 不得因为“上一帧可用”而无限复用。

`requested_effectors` 必须与 `RobotAction.active_effectors` 一致；每个 proposal 还要声明
其动作是否要求成组选择。双臂 collision/coupled planner 场景应把左右臂作为一个
`selection_group`，不能在 Arbiter 中独立选择后再把不相容的两侧拼起来。

### 4. ActionArbiter：只做候选选择和 authority decision

`ActionArbiter` 的职责是确定当前 control cut 中“哪些 source 的哪些 effector/group
proposal 可以成为 selected action”。它不负责：

- 轮询相机、VIVE、手或 policy；
- 解析原始 policy output；
- 执行 IK、workspace/joint safety、collision planner；
- 访问真实 backend、调用 `Rig.send_commands`；
- 启动/停止 episode writer；
- 自己拥有 fault/recovery 状态机。

建议接口保持纯进程内、无硬件依赖：

~~~text
result = arbiter.arbitrate(
    proposals_by_source,
    context=ArbitrationContext(
        now,
        control_mode,
        safety_status,
        active_groups,
        previous_selection,
    ),
)
~~~

其中：

~~~text
ArbitrationContext
├── now / clock_domain
├── control_mode
├── safety_status: READY / HOLD / FAULT / RECOVERING
├── active_selection_groups
├── previous_selection / authority_epoch
├── explicit authority events
└── metadata

ArbitrationResult
├── selected_by_group
├── selected_by_component
├── proposal_id_by_component
├── source_kind_by_component
├── composed_action: RobotAction | None
├── decision_by_group
├── decision_by_component
├── rejected_proposals
├── expired_proposals
├── authority_epoch
├── handoff_event | None
├── output_status: selected / no_action / hold / fault
└── trace_metadata
~~~

`ArbitrationResult` 必须有两个明确的出口，不能把它作为一个无边界的大对象继续向下游和
模型传递：

1. **control port**：只向后续 safety/IK/reference/dispatch 提供 selected absolute
   `RobotAction` 及其最小 `decision_id`/lineage 引用；这些下游可以消费 selected action，但不
   需要知道 policy 的 raw vector 或其他 source 的完整 proposal。
2. **attribution port**：向 `ControlTrace`/reward/label adapter 提供每个 component 的
   owner、proposal、decision reason、mask 和依赖关系；该端口只写训练/审计数据，禁止回流到
   actor 的 `PolicyObservationView`。

Arbiter 本身不实现 reward，也不根据 task reward 改变 selection。reward adapter 应读取
`component_id` 对齐的 attribution 和执行结果，按任务定义计算 credit；这样动作选择与奖励定义
解耦，DAgger、RL 和 offline analysis 可以使用同一份事实记录。

`ArbitrationResult` 要保留“为什么选中/拒绝”的结构化原因，不能只返回一个 action；至少要能
对 `ActionComponentCatalog` 中的每个 `component_id` 给出一条 decision：

~~~text
ProposalDecision
├── group
├── component_id | component_set
├── selected_proposal_id | None
├── owner: human / policy / replay / none
├── reason: priority / takeover / stale / invalid / fault_veto / mode_blocked / incompatible
├── age_sec
├── valid_until
└── metadata
~~~

#### 4.1 选择顺序

每个 control cut 按固定顺序执行，保证结果可重放：

1. **结构验证**：检查 action space、active effectors、frame、shape、finite value、source、
   component schema/coverage 和 proposal 时间字段；失败直接标记 invalid。component coverage
   与 `RobotAction` 的 active fields 不一致时，不能只按 effector 名称继续发送。
2. **因果/新鲜度验证**：检查 `observation_timestamp <= source_timestamp <= valid_until`、
   clock skew、sequence 单调性和 observation age；过期 proposal 不进入候选集。
3. **安全状态 veto**：`FAULT`、manual hold、tracker hold、startup not-ready 等状态不允许
   普通 source 获得执行权；Arbiter 输出 `hold/fault`，不能用旧 action 填充。
4. **控制模式过滤**：根据 `human_only`、`policy_only`、`policy_shadow`、`human_override`
   或 `per_effector_override` 过滤 source。
5. **显式 authority**：处理 GUI/foot-pad/脚本产生的 takeover/release event。第一版不根据
   confidence、动作幅度或“人类是否移动”隐式抢占控制权。
6. **确定性优先级**：在同一 group 中按 mode policy、authority lease、proposal sequence、
   source priority 和 timestamp 做确定性选择；所有 tie-break 规则写入 config 和 trace。
7. **组完整性检查**：coupled group 必须整体选中同一 proposal 或明确进入 hold，不能静默拼接
   不同 frame/action space 的左右臂。
8. **组合与输出**：不同 group 可以组合为一个 `RobotAction`；同一 effector 只能有一个 owner。
   组合失败时返回 fault/hold，不回退到任意 raw `TeleopCommand`。

第一版不做任意 pose blend。Arbiter 只选择 owner；handoff 的平滑应由后续 reference/trajectory
layer 在已知 frame、当前 measured state 和安全约束下完成。四元数不能直接逐元素线性混合。

#### 4.2 ControlMode 与 SafetyStatus 分离

不要把 `POLICY`、`HUMAN`、`HOLD`、`FAULT` 混成一个 enum。建议分成两层：

~~~text
ControlMode:
  HUMAN_ONLY
  POLICY_ONLY
  POLICY_SHADOW
  HUMAN_OVERRIDE
  PER_EFFECTOR_OVERRIDE

SafetyStatus:
  READY
  HOLD
  FAULT
  RECOVERING
  NOT_READY
~~~

`ControlMode` 表达“允许谁提出/获得动作”，`SafetyStatus` 表达“当前是否允许执行”。
SafetyStatus 是上游 session/backend 的 veto，不能由普通 policy proposal 清除；恢复必须
经过现有 `TeleopSession` 或 deployment recovery 流程，并生成明确的 authority epoch。

推荐的初始模式行为：

| 模式 | policy proposal | human proposal | 实际执行 | 用途 |
|---|---|---|---|---|
| `HUMAN_ONLY` | 可不产生/只记录 | 选中 | human | 当前 GUI/headless 默认兼容行为 |
| `POLICY_SHADOW` | 产生并记录 | 选中 | human | 真机第一阶段，验证时序和数据 |
| `HUMAN_OVERRIDE` | 持续产生 | 显式接管 group | human selected group | DAgger/human intervention |
| `PER_EFFECTOR_OVERRIDE` | 选未接管 group | 选接管 group | 按 group 组合 | 仅在 coupling 规则允许时启用 |
| `POLICY_ONLY` | 选中 | 只作 emergency override | policy | 先在 sim/dry-run，最后才考虑真机 |

当显式要求 human override 但 human proposal 已过期或无效时，默认进入 `HOLD`，不能静默
回退到 policy。fallback policy 必须是单独、可审计的模式，而不是 Arbiter 的隐式行为。

#### 4.3 Handoff、hold 和 action chunk

- authority 变化产生 `handoff_event` 和新的 `authority_epoch`，并清除旧 owner 的未消费
  proposal；不能让旧 chunk 跨 owner 继续执行。
- handoff 第一版只做“切 owner + reference anchor reset/hold”，不在 Arbiter 中做数值 blend。
- policy action chunk 由 `PolicyProposalSource` 按 action/reference clock 拆成带 sequence 和
  `valid_until` 的 proposal；policy prediction、reference/IK 和 command 使用
  `PositionControlRates` 的 profile-defined clocks，不应混成一个 Arbiter 频率。当前默认 profile
  是约 `15 → 75 → 150 Hz`，另有显式的 `15 → 45 → 150 Hz` compatibility profile；Arbiter 不应
  写死 45 或 75。
- 对 v2.3 relative policy，proposal 的有效边界是已解码 chunk/keyframe 的边界；同一 chunk 的
  measured TCP anchor 不能因为 reference tick 或 command tick 变快而重新计算。
- Arbiter 在 keyframe/reference boundary 运行；150 Hz command worker 或 position executor
  只消费已选且仍有效的结果，不重新做 authority decision。
- queue empty、policy error、tracker loss、backend not-ready 都要产生显式 no-action/hold
  decision；不能把最后一个 proposal 当成无限期有效。

#### 4.4 fault 与安全层的落位

Arbiter 不是新的 SafetyMonitor。推荐顺序为：

~~~text
observation/source fault
  → source proposal validation
  → ActionArbiter selection
  → backend-specific safety filter / command guard
  → RobotActionResolver / reference executor
  → TeleopCommand / Rig
~~~

当前 policy 的 `ActionSafetyFilter` 主要接收 `NormalizedAction + DeploymentObservation`，
当前 human 的 `TeleopSession.command_filter/command_guard` 主要接收 `TeleopCommand +
TeleopObservation`。第一阶段不应把 policy filter 直接套到 human command 上造成双重或错误
参考；应先抽取统一的 safety result/trace contract，再分别提供 human/policy/sim adapter。

安全层应至少返回：

~~~text
SafetyResult
├── input_action / selected_proposal_id(s) by component
├── safe_action | None
├── modified_effectors / modified_components
├── dependency_proposal_ids_by_component
├── faults
├── blocked: bool
├── reference_timestamp
└── metadata
~~~

这样可以区分“Arbiter 没选到动作”“安全层修改了动作”“backend 拒绝了 command”和“实际
dispatch 成功”。真机 RL/DAgger 训练标签只能按数据契约选择 safe/executed action，不能把
未通过安全层的 policy intent 误当成实际动作。

#### 4.5 Action attribution：数采时实时区分每个 action 的对应部分

统一发送和执行只解决“最后由谁把 command 发给 Rig”的问题；对 DAgger、human-in-the-loop
RL 和 offline RL 来说，还必须解决“当前控制结果的每一部分究竟来自哪个 action source”的问题。
这个归因必须在 Arbiter 做 selection 的同一个 control cut 中生成，不能在 episode 结束后只看
最终 command 向量、再用数值差异猜测来源。

当前 `build_collection_action_record()` 从最终 `TeleopCommand` 反推一个 `RobotAction`，
`CollectorSample` 也主要保存单一 action vector。这些对象可以继续作为 v2.1/v2.2 兼容数据，
但不能作为新 DAgger/RL 数据的唯一事实来源，因为 merge、safety、IK 和 interpolation 之后，
source identity 已经不可逆。

##### 4.5.1 先定义 action component，而不是只定义一个整 action

需要引入稳定的 `ActionComponentCatalog`。它是 action layout 的版本化清单，把 canonical
`RobotAction`、`PolicyIOLayout`、dataset feature 和 backend command 的字段对应起来。
component 至少采用以下层级：

~~~text
component group / selection group
  → effector
    → typed field
      → optional slot

left_arm_coupled
  → left_arm
    → pose / qpos
      → position.x, position.y, ...       # 只有 layout 定义时才下沉到 scalar slot
left_hand
  → left_hand
    → qpos / end_effector_target
      → q0, q1, ...
~~~

第一版的 authority 和 safety granularity 默认是 effector 或 coupled group；dataset 可以同时
保存更细的 field/slot mask。不能把任意向量 slice 自动当成独立可接管 component：例如四元数
不能被任意 source 分别占用，双臂 coupled planner 也不能把左右两侧拆成互不相关的 scalar。

每个 component key 必须是稳定的 typed name，而不是“当前 Python 实现里的第几个数组元素”。
例如 `left_arm.pose`、`right_hand.qpos[3]`；实际可用 component 由
`PolicyIOLayout`/`ActionComponentCatalog` 的 `schema_hash` 冻结。不同 action space、frame
或单位不能直接合并；必须先通过 adapter 投影到同一个 component profile，否则记录为
`incompatible`，不生成虚假的 attribution。

需要明确区分两个概念：

- `selection_group`：Arbiter 是否必须原子地选择一组动作，例如双臂 coupled planner。
- `component`：trace 和训练标签要归因的最小字段，例如 `left_arm`、`left_hand.qpos[2]`。

一个 selection group 可以包含多个 component；一个 proposal 也可以声明只覆盖 group 中的
部分 component，但只有在 group contract 明确允许 partial ownership 时才可组合。

##### 4.5.2 归因不是一个 bool，而是多层 source mask 和 lineage

建议新增 `ActionAttributionFrame`（名称可在实现阶段调整），每个 arbitration/control tick
至少包含：

~~~text
ActionAttributionFrame
├── trace_id / episode_id / control_tick_id
├── observation_id / causal_cutoff / decision_id
├── authority_epoch / control_mode / safety_status
├── component_schema_id / component_schema_hash
├── proposal_id_by_component
├── source_kind_by_component: human / policy / replay / scripted / recovery / none
├── available_mask_by_source_and_component
├── selected_mask_by_source_and_component
├── selected_action_by_component
├── safety_modified_mask_by_component
├── safe_action_by_component
├── commanded_mask / command_id / reference_segment_id
├── executed_action_by_component / dispatch_receipt_id
├── measured_state_id / measured_valid_mask
├── hold_mask / invalid_mask / stale_mask / incompatible_mask
└── decision_reason_by_component
~~~

其中以下 mask 语义必须固定，不能继续用一个含义不明确的 `intervention_mask`：

| 字段 | 语义 | 不能替代的字段 |
|---|---|---|
| `available_mask` | 该 source 在这个 control cut 是否提供了有效 proposal | 不代表它被选中 |
| `selected_mask` | 该 source 是否成为该 component 的 authority | 不代表安全层未修改 |
| `expert_label_mask` | 该 component 是否有可用于 DAgger 的有效 human/expert label | 不等同于 human proposal 存在或数值变化 |
| `policy_behavior_mask` | 该 component 是否由 policy 成为 selected owner 并进入有效 command path | 不等同于 policy proposal 存在；safety 修改和 measured validity 另记 |
| `safety_modified_mask` | safety/planner 是否改变了该 component 或其依赖输出 | 不代表 source 发生切换 |
| `command_valid_mask` | 对应 component 是否进入有效 command/reference | 不代表 backend 已执行 |
| `measured_valid_mask` | 对应 transition 是否有有效 measured 对齐 | 不代表 action 成功 |
| `hold_mask` | 当前 component 被明确 hold/no-action | 不允许用旧 proposal 填充 |

`proposal_id_by_component` 和 `source_kind_by_component` 是身份归因；action 数值字段是各阶段
的值归因。两者必须同时保存。只保存 `human_action`、`policy_action` 和最终 action 三个向量，
但没有 per-component mask，仍然不能判断 policy 是否只控制了右臂、human 是否只接管了左手。

##### 训练信号读取边界：Arbiter 记录，reward adapter 解释

`ActionArbiter` 只负责产生可重放的 selection/ownership facts，不负责定义 task reward，也不把
这些 facts 回传给 policy。训练侧单独读取：

~~~text
ControlTrace + task outcome/measured transition + ActionComponentCatalog
  → RewardAttributionAdapter
  ├─ DAgger expert label/value/mask
  ├─ RL policy-behavior transition/mask
  ├─ safety/intervention/constraint penalty
  └─ mixed-source component credit
~~~

推荐固定以下推导关系：

- `policy_behavior_mask` 表示 policy 在 Arbiter 中成为 component owner 且进入有效 command
  path；safe action 被 clamp 不会抹掉 policy ownership，但必须同时置上
  `safety_modified_mask`，把 policy intent 与 safe value 分开保存；hold、no-action、无效
  command 则不算 policy behavior。
- `measured_valid_mask` 独立存在。训练 exporter 可以再派生
  `policy_learning_mask = policy_behavior_mask & command_valid_mask & measured_valid_mask &
  transition_valid`，但这个派生 mask 不是 Arbiter 的 authority 结果，也不能覆盖原始 mask。
- `expert_label_mask` 由有效 human/expert component proposal 和当前 mode/selection 规则派生；
  在 policy shadow 中，即使 policy proposal 没有被选中，也应保留它的 `available_mask`，而
  human selected component 才能作为对应的 DAgger label。
- task reward 从 measured transition/objective 计算，safety/intervention penalty 从
  `SafetyResult` 与 attribution 计算；两者都在 reward adapter 中合成，不让 Arbiter 依赖 task
  或 RL 算法。
- 混合 source 的 global reward 若无法可靠分摊到 component，必须记录
  `mixed_source=true` 并由 exporter 按 mask 排除或单独处理，不能把整条 transition 伪装成
  policy-only behavior。

因此 actor 的部署输入仍只有 `PolicyObservationView`，actor 输出仍只有 policy intent；
`RewardAttributionAdapter` 可以服务 actor-critic/DAgger 数据准备，但不能把
`selected_by_component`、`safety_modified_mask` 或 reward 偷渡进 actor observation。

##### 4.5.3 lineage 必须随控制结果穿过每个阶段

推荐的最小 lineage 是引用关系，不把大数组复制到每一个 150 Hz event：

~~~text
ActionProposal
  └─ proposal_id + component coverage
       ↓
ArbitrationResult
  └─ decision_id + selected proposal/source per component
       ↓
SafetyResult
  └─ selected proposal ids + modified component/dependency mask
       ↓
Reference / IK / planner output
  └─ reference_segment_id + dependency proposal ids per output component
       ↓
TeleopCommand
  └─ command_id + source/lineage metadata per effector command
       ↓
CommandDispatchReceipt
  └─ dispatch_id + sent/rejected/fault + dispatch timestamp
       ↓
Measured state
  └─ measured_state_id + interval/age/validity
~~~

这里有两个必须单独记录的情况：

1. **安全修改不改变 source identity。** 人类或 policy proposal 被 clamp 后，仍然是原 source
   的 selected action，但 `safety_modified_mask` 为 true，safe value 与 intent value 分开保存。
2. **IK/planner 可能是多输入变换。** 如果一个 coupled planner 的右臂 joint command 同时依赖
   左右臂 proposal，不能伪造一个唯一 source；应记录 `primary_owner`（用于 authority）以及
   `dependency_proposal_ids`/`dependency_component_mask`（用于可解释性）。如果未来真的加入
   blend，必须保存每个 source 的权重和 blend algorithm version，不能由训练脚本事后推断。

因此 `TeleopCommand.metadata` 可以暂时携带 `command_id` 等兼容信息，但不能把完整 lineage
塞成非结构化 JSON。后续应有 typed `ActionLineage`/`ControlTrace` contract，resolver、reference
executor、Rig 和 MuJoCo backend 都只传递或补充该 contract，不重新计算 source attribution。

##### 4.5.4 实时采集时的 control cut 与写入边界

一个正式 control cut 的顺序应固定为：

~~~text
observation snapshot
  → assign observation_id / causal cutoff
  → proposal buffer snapshot
  → ActionArbiter
  → create decision + per-component attribution
  → safety / planner / reference
  → attach lineage to TeleopCommand
  → dispatch and receipt
  → close with measured state / next observation
  → mandatory lightweight ControlTrace sink
  → asynchronous dataset/media writer
~~~

`ActionAttributionFrame` 在 Arbiter 返回时先创建 selection 部分；safe、commanded、receipt 和
measured 信息在同一个 `control_tick_id` 下按事件补齐。为了保证实时性：

- control loop 只构造 immutable 的小型 typed frame、整数 ID、mask 和必要的 canonical action
  数值，不在 150 Hz 路径做 JSON 序列化、视频编码或大对象深拷贝；
- `ControlTraceSink` 先写入进程内 bounded ring/轻量队列，异步 writer 再写 Arrow/Parquet/sidecar；
- action lineage、decision、mask 和 dispatch receipt 属于 mandatory trace，不能像视频一样静默
  drop。mandatory trace 队列失败必须暴露为 recording/trace fault，由 episode policy 决定进入
  hold、结束 episode 或标记数据无效；默认正式数采不允许无提示地继续产生不可训练数据；
- 图像、depth、tactile 等大对象仍可按现有 DropOldest/异步策略丢帧，但必须在 trace 中保留
  `frame_id`、`valid_mask`、age 和 drop reason；
- 现有 GUI/headless 的 sample-rate gate 只是存储/视频采样频率，不是 authority 或 attribution
  频率。即使 LeRobot sample 低于 reference/control rate，也必须保存每个 decision/reference
  interval 的 `decision_id`、component lineage 和 effective interval；降采样时要显式记录
  `latest/hold/interval` 对齐规则，不能用 sample timestamp 事后猜测 action 来源；
- `ControlTrace` 生命周期必须与当前 GUI/headless 共用的 `CollectionWriterRuntime` 对齐，
  不能为 policy shadow、DAgger 再建立一套独立 writer。

snapshot 之后新来的 proposal 只能进入下一个 control cut。这样同一个 tick 的
`selected_source_by_component`、`expert_label_mask` 和 `command_id` 可重放，也不会因线程调度
造成“action 数值是这一帧、source id 却是下一帧”的错配。

##### 4.5.5 当前 human 数采链路需要怎样改

第一阶段不直接改真机行为，但接入 observe-only attribution 时必须改变记录边界：

1. `TeleopSession._build_command()` 在 `merge_teleop_commands()` 之前，为每个 retargeter
   保存 source id、timestamp、覆盖的 effector/component 和原始 typed action；不能只保留 merge
   之后的 `TeleopCommand`。
2. `TeleopCommand -> RobotAction` adapter 输出 `ActionProposal` 时，带上来源和 component coverage；
   同名 effector 的覆盖关系由 Arbiter/decision 记录，不能在 merge 时静默丢弃。
3. `SessionStepResult` 需要携带 `proposal_snapshot`、`ArbitrationResult` 或等价的
   `ActionAttributionFrame` 引用；`build_collection_action_record()` 只能作为兼容 fallback，
   不能再次从最终 command 猜 source。
4. `CollectionWriterObserver`/`CollectionWriterRuntime` 同步提交 lightweight control trace
   job；`CollectorSample.action` 继续保存兼容的最终 action，但新 trace 保存 per-source action、
   selected/safe/commanded/measured 以及 masks。
5. policy shadow 时，即使 policy 没有 selected component，也必须保存其有效 proposal 和
   `available_mask`；否则后续无法构造“policy 看到了什么、human 在哪里接管”的 DAgger 样本。

这样，现有 GUI/headless 的实际发送顺序、fault/hold 和 writer 行为可以在 observe-only 阶段
保持不变，同时把丢失的 action provenance 在正确位置补回来。

#### 4.6 代码落位与现有模块复用

建议先在共享层增加小模块，不把 Arbiter 放回 `teleop_runtime/core/kernel.py`：

~~~text
robot_motion/action/
├── schema.py                 # 现有 RobotAction
├── sources.py                # 现有 RobotActionSource protocol
├── codec.py/vector_pipeline.py # 现有 vector↔RobotAction contracts
├── executor.py               # 现有 resolver/dispatcher seam；不是 Arbiter
├── proposal.py               # 后续：ActionProposal、ProposalDecision、ArbitrationResult
├── attribution.py            # 后续：component catalog、ActionLineage、AttributionFrame
└── arbiter.py                # 后续：纯 ActionArbiter、ControlMode、Authority lease

robot_motion/runtime/
├── contracts.py              # 现有 TeleopObservation/TeleopCommand
├── interfaces.py              # 现有 source/backend/retargeter/Rig protocols
├── dispatch.py               # 现有 receipt/failure contract
├── lifecycle.py              # 现有 startup/cleanup failure aggregation
├── causal_history.py         # 现有 causal history
├── rig.py                    # 现有 shared BimanualRig/CompositeRig
└── trace.py                  # 后续：ControlTrace sink/codec 与 lightweight event

teleop_runtime/action/
├── proposal_adapters.py      # human TeleopCommand、policy chunk、replay adapters
└── attribution_adapters.py   # legacy session/recorder 到 typed trace 的桥接

dataset_contracts/
└── control_trace.py          # 跨 sim/real/replay 的 trace、lineage、mask schema
~~~

可以复用现有 `robot_motion/action/sources.py` 的 source lifecycle、
`RobotActionResolver`/`ActionExecutor` 的 resolve/dispatch seam、
`CommandDispatchReceipt` 的 backend receipt、`robot_motion.deploy.canonical_action` 的
canonical filter/resolver 和 `PolicyDeployRuntime` 的 policy queue，
不要再创建第二套 vector/action/dispatcher protocol。`ActionArbiter` 和 `ActionLineage` 本身
不得 import `teleop_runtime`、hardware、MuJoCo 或 GUI；`dataset_contracts.control_trace` 也
不能依赖 Torch、writer thread 或具体 backend。

建议的测试落位：

~~~text
tests/test_action_proposal_contracts.py
tests/test_action_arbiter_selection.py
tests/test_action_arbiter_timing.py
tests/test_action_arbiter_authority.py
tests/test_action_attribution.py
tests/test_control_trace_contract.py
tests/test_human_command_robot_action_adapter.py
tests/test_action_arbiter_integration.py
~~~

其中纯逻辑测试必须覆盖：同一输入的 deterministic decision、proposal 过期、future timestamp、
clock skew、sequence rollback、source exception、显式 takeover/release、policy shadow、
human override、per-group selection、coupled group 原子性、action space/frame mismatch、
no proposal、fault veto、queue empty 和 handoff 后旧 proposal 清理。attribution 测试还必须
覆盖：human 左臂 + policy 右臂、human hand + policy arm、同一 effector 的覆盖与拒绝、
component mask 与 action layout 对齐、安全 clamp 前后 lineage 不变、coupled planner 多 proposal
依赖、hold/no-action 不复用旧 source、150 Hz command 继承正确的 decision/reference id、
mandatory trace queue failure 和异步媒体丢帧标记。集成测试使用 fake resolver、
`NullCommandDispatcher` 或 MuJoCo backend，不连接真实硬件。

## 4.7 Proposal source 与并发边界

Proposal source 可以在自己的线程/进程中做 policy inference 或设备读取，但 Arbiter 每次只
接受一个 immutable snapshot：

~~~text
source worker / frontend
  -> thread-safe ProposalBuffer.publish(proposal)
  -> control/reference tick snapshot(cutoff)
  -> ActionArbiter.arbitrate(snapshot)
  -> selected action
~~~

`ProposalBuffer` 负责容量、sequence、丢弃策略和 source transaction；`ActionArbiter` 不负责
等待慢 source，也不直接读取 queue。snapshot 后新到达的 proposal 只能进入下一控制 cut，避免
同一 tick 因线程调度不同而产生不可重放的选择结果。

policy prediction transaction、chunk replace/append、IK recovery discard 由唯一的
`PolicyDeployRuntime` 管理；Arbiter 只看到已经提交的 proposal。human source 的 VIVE/键盘/脚踏
事件也应先转成带 timestamp 的 source event，再影响下一次 authority decision。

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
reference / IK    profile-defined（当前默认约 75 Hz；兼容 profile 约 45 Hz）
command           约 150 Hz
physics           约 1000 Hz
~~~

模型推理不能放在 150 Hz command loop 中。模型只产生新 action chunk；高频 reference/command
线程消费最近一个有效、已解码且经过 Arbiter 选择的 absolute action，并继承其 proposal、decision
和 representation trace id。

---

## 四、DAgger 和真机 RL 应如何共用一套框架

### DAgger

DAgger 的样本主键应是
`(episode_id, observation_id, control_tick_id, component_schema_hash)`，而不是只有一个
`timestamp` 或一条整体 action vector。每个 policy tick 至少记录：

~~~text
observation_t
observation_id_t
policy_proposal_id_t + policy_action_t + policy_available_mask_t
expert_proposal_id_t + expert_action_t + expert_label_mask_t
selected_action_t + selected_source_by_component_t
safety_filtered_action_t + safety_modified_mask_t
commanded_action_t + command_valid_mask_t
executed_action_t + executed_valid_mask_t
behavior_source_by_component_t
policy_behavior_mask_t
expert_selected_mask_t
authority_epoch
decision_id + decision_reason_by_component
component_schema_hash
takeover_reason
policy_latency
observation_timestamp + action_effective_interval
dispatch_timestamp + measured_state_id
measured_state_{t+1} + measured_valid_mask_t
~~~

“expert label”与“实际执行 source”必须拆开：

- `expert_action_t` 保存 causally valid 的 human/expert proposal；只有对应
  `expert_label_mask_t=1` 的 component 才能作为 DAgger label。human 没有提供该 component、
  proposal 已过期、group 不完整或进入 hold 时，label 必须是 invalid/masked，而不是复用上一帧。
- `selected_action_t` 是 Arbiter 在该 tick 选择的 composition，`selected_source_by_component_t`
  是每个 component 的 authority；它可能与 expert label 不同，例如 policy 正在执行而 human
  同时提供了可用于学习的纠正 proposal。
- `safety_filtered_action_t`/`safe_action_t` 是安全层后的 canonical action，
  `safety_modified_mask_t` 用来判断安全层改变了哪些部分。
- `executed_action_t` 只表示已进入有效 dispatch/reference 的 action；必须有
  `executed_valid_mask_t` 和 receipt，不能把 command 被拒绝的 action 标成 executed。
- `expert_selected_mask_t` 表示 human 是否实际拥有 authority，不能用 action 数值差异代替；
  数值差异可以另存为 debug metric，但不能作为 DAgger intervention 的定义。

典型的 per-component 结果应能表达：

| component | policy proposal | human proposal | selected source | expert label | policy behavior |
|---|---|---|---|---:|---:|
| `left_arm` | 有 | 有 | human | 1 | 0 |
| `right_arm` | 有 | 无 | policy | 0 | 1 |
| `left_hand` | 无 | 有 | human | 1 | 0 |
| `right_hand` | 有 | 有但未接管 | policy | 按模式 | 1 |

这类样本不能通过最终的双臂/双手 action vector 反推，必须在 Arbiter 选择时直接产生
`proposal_id_by_component` 和 masks。policy shadow 即使 policy 没有被选中，也要保留它的
proposal/action/latency；human override 则同时保存 policy intent 和 human expert label，避免
训练数据只剩“最终由谁执行”的信息。

不能只记录最终执行动作，否则无法区分：

~~~text
模型主动控制
人类接管
安全层修改
人类只控制某一个末端
~~~

现有 [robot_motion/runtime/causal_history.py](/home/zc/HandMotion/Developing/DexTeleop/robot_motion/runtime/causal_history.py) 的
`source_timestamp <= cutoff` 机制非常适合作为 DAgger 的时间对齐基础，可以避免把未来的人类
动作泄漏到当前 observation。每个 label 还要保存 `action_effective_interval`：label 属于哪个
observation/control cut，而不是简单按 writer 完成时间归档。policy chunk 需要保存
`chunk_id`、keyframe index 和 proposal timestamp，不能把一个 chunk 的所有插值 command 当成
多个独立 policy decision。

#### DAgger 的 v2.3 representation boundary

如果 policy 的训练输入是 relative action view，DAgger 不能只保存一个名为 `policy_action` 的
向量。至少应同时保存以下三种语义，并且三者使用同一个 `control_tick_id` 关联：

~~~text
policy_action_model_view
  = policy 实际看到/输出的 relative 或 absolute chunk keyframe

policy_action_decoded_absolute
  = 使用该 observation 的 actual TCP anchor 解码后的 runtime-frame absolute action

expert_action_model_view
  = 用同一个 PoseActionChunkContract、同一个 frame contract、同一 causal observation
    生成的 human/expert label view
~~~

其中 `policy_action_model_view` 记录 `action_is_pad`、normalization/layout hash、
`pose_chunk_contract_hash` 和 `pose_frame_contract_hash`；`policy_action_decoded_absolute` 记录
`representation_trace_id`、`anchor_observation_id`、实测 TCP anchor 的时间戳/frame 和 decode
状态。两者不能只靠后处理重新生成，因为后处理可能已经丢失当时的 artifact、anchor 或 mode
绑定。

expert label 的转换也必须复用现有 `dataset_interface.pose_action/pose_chunk` 和
`robot_motion.relative_action`，不能把 dataset 中的 absolute storage row 直接当作 relative
label。对于每个有效 label：

1. 先按 episode 的 `PoseFrameContract` 验证 human action 与 observation frame 一致；
2. 再按 policy artifact 的 `PoseActionChunkContract` 生成模型所需的 relative/absolute view；
3. relative label 的 anchor 规则必须写入 trace，且只能使用 causally valid 的 measured TCP；
4. human 没有提供该 component、anchor 不可用、chunk 跨 episode、row 是 padding 或 action
   过期时，`expert_label_mask` 必须为 0，不能用上一条 action 补齐；
5. policy 的 raw intent 和 human expert label 即使都被解码为 absolute，也必须保留各自的
   `representation_trace_id`，避免 exporter 把两个来源误认为同一 action。

因此 DAgger 的最终读取接口应同时能返回：

~~~text
policy_model_view / policy_decoded_absolute
expert_model_view / expert_decoded_absolute
expert_label_mask_by_component
selected_source_by_component
pose_anchor_id_by_source
pose_chunk_contract_hash / pose_frame_contract_hash
~~~

这样训练器可以使用 relative label，评估器可以比较 canonical absolute intent，运行时仍只
执行安全后的 absolute action；数据表示、控制语义和执行行为不会被一个字段混在一起。

### 真机 RL

真机 RL 的 rollout 应记录：

~~~text
observation / observation_id
behavior_action = safe/executed absolute action aligned to the control interval
behavior_action_model_view (optional, generated by the same v2.3 chunk contract)
selected_action / commanded_action / executed_action
behavior_source_by_component
policy_behavior_mask / policy_logprob_mask
proposal_id_by_component / decision_id
reward
terminated
truncated
next_observation
next_observation_id
faults
latency / action_effective_interval
safety_modified_mask / command_valid_mask / measured_valid_mask
authority_epoch / control_mode
policy_artifact_id
environment_id
episode_id
~~~

真机 RL 的 `behavior_action` 默认使用与该 transition 对齐的 safe/executed canonical action，
而不是未经安全层处理的 policy intent；`commanded_action` 和 backend receipt 仍要保留，以便
区分“安全动作”和“实际送到 backend 的 transport/reference”。如果 backend 没有可靠的
canonical inverse，不能伪造一个 measured action，应使用 safe action 加
`command_valid_mask`/receipt 表示可用性。

若 policy 本身以 v2.3 relative chunk 训练，`behavior_action_model_view` 只能由同一个
`PoseActionChunkContract` 从已记录的 canonical control interval 重新组织，并保留对应的
actual TCP anchor、padding mask 和 frame hash。它是训练/回放视图，不是新的 executed action；
RL 的 `policy_logprob` 也只能与 `policy_behavior_mask` 为 1 的 model-view component 对齐。
不能把 relative vector 直接标成 `executed_action`，也不能用当前 commanded target 代替
缺失的 measured TCP anchor。

混合 human/policy rollout 也必须按 component 标记：

- policy 只在 `policy_behavior_mask=1` 的 component 上拥有 behavior policy/log-prob；human
  component 不应被当作 policy action 计算 log-prob 或 advantage。
- 一个 transition 可以使用完整的 measured reward，但训练 sampler 必须根据 source mask 决定
  是否作为 on-policy policy transition、human intervention transition 或仅作为 offline data。
- safety 修改、hold、fault、tracker loss、backend reject 和 trace loss 都要影响
  `terminated/truncated/valid_mask`，不能把故障前后的 action 无条件拼成正常 rollout。
- DAgger 的 expert label 可以来自同一 ControlTrace，但 RL 的 behavior action 必须取实际
  selected/safe/executed 层；两者不能通过一个叫 `action` 的字段混用。

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
│   ├── schema.py                 # 现有 RobotAction
│   ├── sources.py                # 现有 source protocol
│   ├── codec.py/vector_pipeline.py # 现有 action 编解码与 layout
│   ├── executor.py               # 现有 resolver/dispatcher seam
│   ├── proposal.py               # 后续 ActionProposal contract
│   ├── attribution.py            # 后续 component lineage/trace
│   └── arbiter.py                # 后续纯逻辑 authority/selection
├── runtime/
│   ├── contracts.py              # 现有 observation/command contracts
│   ├── interfaces.py              # 现有 source/backend/Rig protocols
│   ├── dispatch.py               # 现有 receipt/error contracts
│   ├── lifecycle.py              # 现有 lifecycle cleanup contracts
│   ├── causal_history.py         # 现有 causal history
│   ├── rig.py                    # 现有 shared Rig
│   └── trace.py                  # 后续 ControlTrace sink/codec
├── deploy/
│   ├── runtime.py              # 现有 PolicyDeployRuntime；旧 DeployRuntime 仅兼容 scaffold
│   ├── observation.py          # DeploymentObservation + source-neutral builders
│   ├── pose_adapter.py         # v2.3 relative/frame decode
│   ├── canonical_action.py     # shared safety → RobotAction → TeleopCommand
│   └── actions.py              # normalization/chunk-compatible action contracts
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
    ├── deploy/
    │   ├── observation.py      # MujocoDeploymentObservationSource
    │   ├── executor.py         # MujocoDeploymentActionSink / SimDispatchReceipt
    │   └── run_policy.py       # sim frontend，只组装 shared runtime
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
├── action/
│   ├── proposal_adapters.py
│   └── attribution_adapters.py
├── sources/
├── collection/
├── deploy/
└── adapters/

deployments/
├── client/
│   └── policy.py               # 无硬件 LoadedPolicy/loader（已从 teleop_runtime 提升）
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

用于描述一个控制时刻，不建议继续把所有字段都塞进现有 ActionRecord。这样可以避免破坏已有
v2.1/v2.2 数据，同时为 per-component attribution 保留结构化扩展空间。

建议将它拆成“一条 control tick 主记录 + 若干可关联的事件/lineage 记录”，而不是在每个 sample
里嵌套任意深的 JSON：

~~~text
control_ticks
├── episode/control/observation/decision ids
├── timestamps and effective interval
├── mode, safety status, authority epoch
├── policy observation view reference/contract hash (allowlisted input only)
├── policy intent action and representation/decode provenance
├── policy model-view action + decoded absolute action
├── expert model-view action + decoded absolute label
├── pose anchor ids/timestamps and v2.3 contract hashes
├── canonical policy/expert/selected/safe/commanded/executed action values
├── source/proposal/component masks
└── reward/termination/measured alignment

proposal_events
├── proposal_id, source, sequence, causal timestamps
├── typed RobotAction and component coverage
├── representation_trace_id, input/output action mode and frame
├── pose chunk/frame contract hashes and measured anchor provenance
├── validity/rejection/expiry reason
└── policy chunk/keyframe/latency metadata

component_lineage
├── control_tick_id + component_id
├── selected/proposal/source ids
├── safety/reference/command dependency ids
├── decision reason and modified/valid masks
└── dependency proposal ids for coupled transforms

execution_events
├── command_id, reference_segment_id, dispatch_receipt_id
├── sent/rejected/fault/latency
└── measured_state_id, age and validity
~~~

这里的 `control_ticks` 是 recording/training 的完整事实快照，不是 actor 的输入结构。它可以同时
保存 policy view、Arbiter decision、safe/commanded/measured 和 reward 对齐，但写入 policy view
的字段必须来自独立 allowlist；下游字段只能通过 trace/event 关联，不能因为处于同一条记录就被
`DeploymentObservation.model_input()` 或 policy wrapper 自动暴露。

其中：

- `control_ticks` 是 DAgger/RL 的主要读取入口，值和 mask 按
  `component_schema_hash` 对齐；
- `proposal_events` 保留“当时有哪些 action 候选”，包括 policy shadow 中未被选中的 policy
  action，不能只写 selected proposal；
- `component_lineage` 是 source attribution 的事实来源，至少能够从一个 component 追到
  proposal、decision、safe result、command 和 measured state；
- `execution_events` 允许同一个 reference decision 关联多个高频 command，不把每一个 150 Hz
  command 误认为一次新的 Arbiter decision；
- `ActionRecord`、`CollectorSample.action` 和已有 v2.x writer 可以继续输出兼容字段，但新
  `ControlTrace` 缺失时，新的 DAgger/RL exporter 应拒绝生成“看似完整”的训练样本。

### dataset_interface/trajectory.py

建议形成统一的时间窗口读取接口：

~~~text
TrajectoryWindow
├── observation
├── images / depth / tactile
├── robot_state
├── policy_action_model_view + policy_available_mask
├── policy_action_decoded_absolute + policy_decode_valid_mask
├── expert_action_model_view + expert_label_mask
├── expert_action_decoded_absolute + expert_decode_valid_mask
├── selected_action + selected_source_by_component
├── safe_action + safety_modified_mask
├── commanded_action + executed_action + valid masks
├── behavior_source_by_component
├── pose_anchor_id / representation_trace_id
├── pose_chunk_contract_hash / pose_frame_contract_hash
├── proposal/decision/lineage ids
├── valid_mask
├── timestamps
├── effective_intervals
├── domain
└── provenance
~~~

新的 DAgger 和 RL 数据应以此为入口。v2.1/v2.2 保留兼容读取，v2.3 数据以 absolute storage
和 `PoseActionChunkContract` model view 共存；新数据默认必须带 component schema、因果时间线、
representation trace 和 contract hashes。`intervention_mask` 如果继续对外提供，只能作为由
`expert_selected_mask` 派生的兼容字段，并明确其 component granularity；不能再作为唯一 source
标记。

---

## 七、现有代码迁移策略

第一批不重写 `run_collect.py`、`run_policy.py` 或 GUI 大入口；writer 生命周期和现有 human
dispatch seam 已先在入口内收敛，policy stream 也已迁到 `robot_motion.deploy`。`ActionArbiter`
应按以下顺序接入，避免一次迁移改变真机行为：

1. 在 `robot_motion/action/` 增加纯 contract、validation、deterministic arbiter 和 fake source，
   先不连接任何真实 Rig。
2. 在 `teleop_runtime/action/` 增加 human `TeleopCommand -> RobotAction` adapter，完成
   pose/qpos/hand/end-effector 的 round-trip golden tests。
3. 在 GUI/headless human-only 模式接入 observe-only Arbiter：只生成 decision/trace，不改变
   原有 `TeleopSession` command_filter、guard、hold 和 dispatch。
4. 将 policy chunk 和 replay 分别包装成 `ActionProposalSource`；policy prediction、queue、
   transaction 和 v2.3 decode 继续由唯一的 `PolicyDeployRuntime` 管理，Arbiter 只在
   keyframe/reference boundary 读取已经解码的候选。
5. 在 sim/dry-run 先启用 selected action，再接入 human override 和 per-group override；通过
   回归后才进入真机 `policy_shadow`。
6. 用统一 trace 写入 proposal、decision、safe result、backend receipt 和 measured state；最后
   再把入口简化成组装层。

尤其要避免让 `TeleopSession`、新的 RuntimeKernel 和 policy loop 各自保留一套独立的：

- armed/fault 状态机
- hold/recovery 逻辑
- action dispatch
- safety filter

最终应该只有一套控制权和故障状态源。

---

## 八、实现前建议冻结的几个决策

建议默认采用以下方案：

1. 模型、human、replay 的 proposal 统一使用 `RobotAction` 语义；`TeleopCommand` 只作为
   resolver/backend transport，不作为 Arbiter 的公共输入。
2. 第一版人机协同使用显式 mode/priority/authority event 和 selection group，不做任意连续
   pose blend；handoff 平滑放在 reference layer。
3. `ControlMode` 与 `SafetyStatus` 分离；fault/hold/recovery 仍由 session/backend 产生 veto，
   Arbiter 不能自行清除 fault。
4. Arbiter 在 policy/reference keyframe 边界运行，不能放进 150 Hz command worker 或 physics loop。
5. policy observation 禁止 privileged state；privileged state 只提供给 critic、reward 或 debug info。
6. 新数据以 v2.2 typed action 为主，v2.1 只作为兼容适配层。
7. sim 和真机必须共享 `RobotAction` 语义，并分别复用适合 backend 的 Safety/Planner adapter。
8. 每个 episode 必须有明确的 episode_id、domain、task profile、policy artifact 和 seed。
9. 所有动作必须尽可能区分 intent、proposal、selected、safe、commanded、executed、measured，
   且每一层都要保留 per-component source/proposal lineage 和明确 mask。
10. DAgger 的 expert label 与 RL 的 behavior action 使用不同字段和 validity mask；不能把
    人类接管 action、policy intent 或最终 command 统称为 `action`。
11. ControlTrace 是正式数采的强制旁路记录；视频/tactile 可以按策略丢帧，action attribution、
    decision、receipt 和 source lineage 不得静默丢失。
12. v2.3 relative 只作为 dataset/model-facing representation；`PoseActionChunkContract`、
    `PoseFrameContract`、`robot_motion.relative_action` 和 `robot_motion.deploy.pose_adapter`
    是唯一转换 owner。进入 Arbiter 的 proposal 必须是 absolute runtime `RobotAction`，并携带
    decode anchor、contract hash 和 representation trace 引用。

---

## 九、推荐实施顺序

### Phase 0：冻结基线

在正式接入真机模型前，整理当前工作树，形成一个可追踪的 baseline，并记录：

- 当前 v2.1/v2.2/v2.3 状态
- 哪些文件属于正在进行的迁移
- 目标 policy action profile
- deployable sim 与 task-native sim 的区别
- v2.3 的 absolute storage、model-view chunk、artifact binding 和 runtime decode 的对应关系
- 当前默认 `15→75→150` 与显式 `15→45→150` compatibility profile，禁止在新模块中硬编码频率
- policy relative anchor 必须来自 measured actual TCP；缺失/过期时的 fail-closed 行为

### Phase 1：ActionArbiter 共享契约与纯逻辑实现

先不接真机、不改变 GUI/headless 行为、不训练模型，按四个小阶段完成：

#### Phase 1A：契约与时钟

- `ActionProposal`、`ActionProposalSource`、`ArbitrationContext`、`ArbitrationResult`、
  `ProposalDecision`、`ControlMode`、`SafetyStatus`。
- `PolicyObservationView` 的 allowlist projection：明确 actor 可见字段与 runtime-private 字段，
  并用 contract test 保证 `ArbitrationResult`、`SafetyResult`、lineage、reward 和 trace metadata
  不会通过 observation `metadata` 回灌模型。
- 统一 control clock、sequence、`valid_from/valid_until`、causal cutoff 和 active effectors。
- `RobotAction` snapshot/copy/freeze 规则，以及不兼容 frame/action space 的明确错误。
- `ActionComponentCatalog`、component schema hash、selection group 与 component coverage；
  明确 effector/field/slot 的 label granularity。
- 不新增 relative 数学实现；复用现有 `PoseActionChunkContract`、`PoseFrameContract`、
  `robot_motion.relative_action` 和 `robot_motion.deploy.pose_adapter`，只定义
  `ActionRepresentationTrace`/decode receipt 的 provenance 字段。
- `ActionLineage`/`ActionAttributionFrame` 的 proposal、selected、safe、commanded、executed、
  measured id 和 source/mask 语义；明确 `expert_label_mask` 与 `policy_behavior_mask` 的区别。

#### Phase 1B：纯 Arbiter 与组合器

- deterministic freshness/authority/mode/group selection。
- 每个 `component_id` 输出 owner、selected/proposal/availability mask、decision reason 和
  selection-group 依赖；同时生成 control port 与 attribution port，reward 不写入 Arbiter。
- human-only、policy-only、policy-shadow、human-override、per-group override。
- coupled group 原子选择；不同 group 的安全组合；不做 raw pose blend。
- no-action、hold、fault-veto、stale、invalid、incompatible 的结构化 decision。
- `selected_source_by_component`、`proposal_id_by_component` 和 decision reason 必须是 Arbiter
  的正式输出，而不是由 recorder 从最终 action 猜测。

#### Phase 1C：synthetic source 和 trace

- static human/policy/replay sources、clock skew、slow source、empty queue、sequence rollback
  的 synthetic tests。
- 记录所有候选、拒绝原因、selected owner、authority epoch、handoff、每个 component 的 source
  mask 和 safety boundary。
- 用一个 mixed-source fixture 验证 human 左臂 + policy 右臂/手部的 action 值、proposal id、
  selected/safe/commanded/masked 数据能够逐层重建；验证 policy shadow 未被选中也不会丢失。
- 验证 reference/IK 输出的 dependency lineage、150 Hz command 对 decision/reference 的继承、
  mandatory trace queue failure 和媒体丢帧标记。
- 使用 `NullCommandDispatcher`/fake resolver 验证 Arbiter 不触碰硬件。

#### Phase 1D：human legacy adapter

- 将现有 retargeter 的 `TeleopCommand` 转为 `RobotAction`，完成 pose/qpos/hand/end-effector
  round-trip golden tests。
- 保留 human `delta_clutch` 自己的 tracker/home/session anchor 语义；不要把它误接成 v2.3
  policy 的 chunk-start measured-TCP anchor。两者只在 source adapter 输出的 absolute
  `RobotAction` 层汇合。
- adapter 遇到未知 command type 时返回 invalid proposal；不得绕过 Arbiter 继续发送。
- 在 `merge_teleop_commands()` 之前保留每个 retargeter 的 source/component coverage，并把
  merge 覆盖关系记录到 attribution；不能让同名 effector 的后写值覆盖掉前一候选而无 trace。
- 先以 observe-only 方式接入 `TeleopSession`，确保原有 command_filter、guard、fault/hold、
  `BimanualRig` dispatch 和 writer 输出不变。

当前已有的 `TeleopSession.dispatch_command()` 和共享 writer runtime 不重复实现；本阶段要把
它们作为集成 seam，而不是再创建一套新的 loop。

### Phase 2：统一 sim 路径

新增 MuJoCo observation/action adapter，使一个 deployable policy 可以走：

~~~text
v2.3 model view → existing pose_adapter → absolute RobotAction
  → ActionProposal → ActionArbiter
  → common pipeline → MuJoCo backend
~~~

同时保留现有 CubeLift fast/native RL 路径，避免为了 deploy parity 破坏快速实验效率。deployable
sim 必须保留与真机相同的 proposal/decision/lineage/ControlTrace 字段，即使 MuJoCo 没有真实
dispatch receipt，也要使用 backend receipt adapter 和 measured validity 标记，不能因为是 sim
就退回只记录一个 action vector。

sim 的 relative decode 必须和真机调用同一套 chunk/frame contract 与
`robot_motion.deploy.pose_adapter`；sim backend 只消费 decoded absolute `RobotAction`。在
进入真机 shadow 前至少完成：dataset absolute row → relative model view → decoded absolute
action 的 golden round-trip、frame/hash mismatch fail-closed、固定 chunk anchor、padding mask
和 `15→75/45→150` position profile 的时序回放。

### Phase 3：真机 shadow 与人工接管

在 Phase 1/2 contract 和回放验证通过后，按风险逐步运行：

- GUI `policy_shadow`：模型只预测和记录，真人实际控制，selected action 仍为 human。
- policy 的下一次输入仍只使用原有 `PolicyObservationView`/环境 measured observation；不能把
  “本次 human 取代了 policy”或 Arbiter decision 注入下一次 actor input。该事实只写入 trace，
  由训练侧 mask/label adapter 使用。
- policy shadow 同时记录 raw model-view、decoded absolute action、measured anchor 和 decode
  receipt；没有 selected 的 policy proposal 也不能从 trace 中丢掉。
- `human_override`：模型持续预测，使用显式 authority event 接管指定 group。
- `per-group override`：只接管已声明允许独立运行的 group；双臂 coupled group 不拆开。
- 真机阶段只允许记录 proposal/decision/trace，不直接从新 Arbiter 开启 policy-only；每个
  control cut 必须能够实时导出 policy/human proposal、per-component selected source 和
  safe/commanded/receipt lineage。

policy-only 先在 sim 和 dry-run 完成，真机 policy-only 需要另行通过安全评审和验收。

### Phase 4：统一 DAgger / RL 数据

实现：

- `dataset_contracts.control_trace` 和 component lineage schema
- `RewardAttributionAdapter`：从 component attribution、safe/executed/measured values 和 task
  outcome 派生 `expert_label_mask`、`policy_behavior_mask`、`policy_learning_mask` 及安全/接管
  penalty；不把这些字段加入 actor observation
- TrajectoryWindow
- DAgger expert label export
- RL rollout buffer
- v2.3 model-view/decoded-absolute 双视图 exporter；缺少 chunk/frame contract hash 或 measured
  anchor provenance 时拒绝生成训练样本
- 按 `expert_label_mask`/`policy_behavior_mask` 的 source-aware sampler
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

## 当前架构状态与最终建议（2026-07-17）

### 已确认的正式入口

当前正式真机 teleop/数采入口确定为：

```text
python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui
```

`teleop_runtime.collection` 不再作为第二个正式 GUI 控制实现，而保留为
**headless/兼容 frontend**。它可以用于无 GUI、回放、自动化检查和旧脚本过渡；新功能不应
再只在 `collection` 中实现一份、再在 GUI 主线中复制一份。

### 当前实际主链路

#### GUI 真机 teleop + 正式数采

```text
debug.main
  -> _build_components
  -> TeleopSession.run_step
       -> sources causal observation
       -> SafetyMonitor / fault / tracker hold
       -> retargeters
       -> command filter / guard
       -> TeleopSession.dispatch_command
       -> BimanualRig.send_commands       # 唯一最终硬件 dispatch
  -> recovery / gripper / preview post-process
  -> collection_step_result(T0 snapshot)
  -> PositionReferencePublisher.publish_step   # 只发布已 dispatch command 的 reference
  -> RealTeleopDatasetRecorder.record_step_async
       -> CollectionWriterRuntime
       -> CollectionWriterObserver
       -> AsyncWriterPool -> LeRobotV21Writer
```

这里的 `SessionRecorder`/`EpisodeRecorder` 是可选的 JSONL/debug sidecar；正式训练数据的
主写入是 `RealTeleopDatasetRecorder` 里的 LeRobot writer，二者不能混为一个 action source。
`PositionReferencePublisher` 也不是第二个硬件 dispatcher，它只消费当前控制 cut 的命令并
建立因果 reference timeline。

#### headless/兼容数采

```text
collection.main
  -> build_collect_app
  -> CollectionRuntime.control_step
       -> TeleopSession.run_step
       -> TeleopSession.dispatch_command
       -> BimanualRig.send_commands
  -> sample rate gate
  -> CollectionRuntime.record_step
       -> CollectionWriterRuntime
       -> CollectionWriterObserver
       -> AsyncWriterPool -> LeRobotV21Writer
```

这条路径与 GUI 主线已经共享 `TeleopSession`、`BimanualRig`、causal observation、fault/hold
语义和 writer observer；原先两处各自管理 writer/observer 生命周期，前一批已收敛到
`teleop_runtime.collection.writer_runtime.CollectionWriterRuntime`。episode 按键、暂停/恢复、
相机预览和 finalize policy 仍属于 frontend 层，暂不强行合并。

`collection` 的 trajectory replay 分支仍有一条兼容性特例：它在读取并通过
`ActionSafetyFilter` 后直接调用 rig，并走独立的 replay sample 提交逻辑。这不是当前 GUI
正式主链路；后续应把它改成 `ReplayActionProposal -> canonical safety/dispatch -> shared trace`
适配器，或将仅供历史数据回放的部分移入 `legacy`，不能继续扩展成第三套生产控制循环。

#### 真机 policy deployment

```text
teleop_runtime.deploy.cli
  -> teleop_runtime.deploy.run_policy.main
  -> build_deployment_observation (causal hardware/camera/hand snapshot)
  -> bind policy artifact pose chunk/frame contract
  -> PolicyDeployRuntime.predict_and_queue
       -> policy-rate decision / prediction transaction
       -> prepare_policy_observation
       -> policy.predict
       -> normalize_and_decode
            -> normalize_policy_output
            -> robot_motion.deploy.pose_adapter
                 -> measured actual TCP anchor
                 -> v2.3 relative/absolute + frame conversion
                 -> absolute runtime RobotAction
       -> ActionChunkBuffer replace/append
  -> PolicyDeployRuntime.prepare_next_action
       -> CanonicalDeployActionPipeline
       -> ActionSafetyFilter
       -> RobotActionResolver
       -> TeleopCommand
  -> execute_prepared_action
       -> direct dispatch (impedance/direct mode)
       -> PositionDeployExecutor (position/reference mode)
```

这里已经有可直接复用、并已由 `PolicyDeployRuntime` 接管的 v2.3 runtime decode seam；下一阶段的唯一插入点是
`absolute runtime RobotAction` 与 `CanonicalDeployActionPipeline` 之间：

~~~text
absolute runtime RobotAction
  → ActionProposal + ActionRepresentationTrace
  → ActionArbiter
  → selected absolute RobotAction
  → existing CanonicalDeployActionPipeline
~~~

这意味着 v2.3 relative action 不应让 `ActionArbiter`、`ActionSafetyFilter`、
`PositionDeployExecutor` 或 `BimanualRig` 感知 raw relative vector；它们只处理已有的绝对
runtime action。policy shadow 阶段可以暂时保留当前 direct canonical path，同时旁路生成
proposal/decision/trace；selected action 切换到 Arbiter 后，后面的 safety、reference、command
和 dispatch 链路不再另起一份。

当前 position profile 也必须从文档和实现中统一引用 `PositionControlRates`：默认是约
`15 Hz policy/sample → 75 Hz reference/IK → 150 Hz command`，另有显式的
`15 → 45 → 150 Hz` compatibility profile。relative chunk 的解码发生在 policy/keyframe
边界，不参与 75/45→150 的插值；插值和 audit 继续由现有 position-control 链路负责。

`deployments.openpi.client.real` 只是 OpenPI 参数/模型组装后委托到同一个
`run_policy.main`，没有发现第二个 policy 控制 loop。当前真正被 run_policy 使用的共享 policy
stream 是 `robot_motion.deploy.PolicyDeployRuntime`；旧 `robot_motion.deploy.DeployRuntime` 和
`deployments.client.DeploymentClientRuntime` 仍是通用 scaffold/test seam，不是正式真机部署入口。

### “已废弃不用”判断

目前没有足够证据把正式 `debug`、headless `collection` 或 `deploy.run_policy` 整体移入
`legacy`：它们仍被 README、测试和/或 OpenPI launcher 使用。当前应这样处理：

| 部分 | 当前处理 | 原因 |
|---|---|---|
| `debug.run_vive_rm65_real_teleop` | 保留为正式 GUI 主入口，继续拆 coordinator | 仍拥有真机 GUI、校准、hold/recovery 和正式数采完整生命周期 |
| `teleop_runtime.collection` | 保留为 headless/兼容 frontend | 用户确认需要保留可能性；旧脚本和测试仍大量依赖 |
| `teleop_runtime.deploy.run_policy` | 保留为正式 policy deploy loop | CLI、OpenPI launcher 和部署 README 都明确依赖 |
| `teleop_runtime.deploy.*` re-export shims | 暂不移动 | 兼容 import 仍存在，且 `legacy` 默认不打包 |
| `teleop_runtime/core/kernel.py` | 标记 experimental，暂不接入生产 | 仅 synthetic action tests 使用，不能与 `TeleopSession` 再形成第三套生产状态机 |
| `robot_motion.deploy.PolicyDeployRuntime` | 保留为唯一共享 policy stream core，不作为 CLI 入口 | 已被 `run_policy.py` 使用；不拥有真机 observation/lifecycle/dispatch |
| `robot_motion.deploy.DeployRuntime`、`deployments.client.DeploymentClientRuntime` | 保留窄 scaffold/兼容 seam，禁止扩展 | 不能再各自发展成第二、第三套 production loop |
| `collection` trajectory replay 特例 | P1 隔离/适配，暂不扩展 | 当前存在 direct-rig dispatch 和独立 episode 逻辑，是最接近“第二套 action path”的部分 |

因此本阶段没有机械搬文件到 `legacy`。只有在 `rg` 确认没有生产调用方、兼容 shim 和测试
依赖，并保留迁移说明后，才移动实现；否则先在原路径保留兼容入口并把实现收敛到共享边界。

### 本次 robot_motion 抽取后的真机 policy deployment 状态

本轮代码审计确认：共享 motion/deploy 语义已经进一步落到 `robot_motion`，但没有改变真机
backend、position executor、fault/hold 或控制频率：

- `robot_motion/deploy/runtime.py` 的 `PolicyDeployRuntime` 和
  `PolicyDeployRuntimeConfig` 现在统一封装 `DeploymentObservation` 的 policy pose-mode preparation、
  output normalization、v2.3 relative→absolute decode、policy-rate decision、prediction
  transaction、chunk queue、IK recovery discard、queue pop/lookahead 和注入的 canonical
  preparation seam。
- `robot_motion/deploy/canonical_action.py` 已提供共享的 safety-filter → `RobotAction` →
  `TeleopCommand` preparation；real `run_policy.py` 仍负责实例化配置、safety fault latch、
  position/reference execution 和硬件 dispatch。
- `robot_motion/action/` 已提供 canonical `RobotAction`、action layout/codec/source protocol、
  `RobotActionResolver` 以及通用 `ActionExecutor`/dispatcher seam；它目前服务通用/synthetic/
  replay 路径和后续 adapter，不等于已经完成 `ActionArbiter`，也没有强行替换 real deploy executor。
- `robot_motion/runtime/` 已提供 observation/command contracts、interfaces、dispatch receipt、
  lifecycle cleanup、causal history 和 shared Rig；`robot_motion.position_control`/`rm65` 已提供
  position reference/command、resampling/interpolation、trajectory policy、audit、coupled IK/
  collision/hold 等共享实现。
- `teleop_runtime/deploy/run_policy.py` 已实际调用完整的 shared policy stream boundary；真机入口仍
  持有 observation snapshot、startup/readiness/fault/hold、safety fault latch、position executor、
  debug logger 和最终 dispatch。
- `robot_motion/deploy/session_contract.py` 承载 artifact pose-mode agreement；入口只保留 CLI
  显式参数判定的兼容 wrapper。
- `deployments/client/policy.py` 承载无硬件的 `LoadedPolicy`/`HoldPolicy`/`load_policy`；
  `teleop_runtime.deploy.policy` 保留兼容 re-export，包含现有 replay 所需的 prediction
  transaction 方法。
- 本次还没有把真机 observation source、startup/recovery、safety fault latch、position sink、
  `ActionProposal`/`ActionArbiter`、per-component attribution 或 `ControlTrace` 移动；sim
  implementation 明确延期，不作为本次验收项。

增量验证：`tests/test_robot_motion_deploy_contracts.py` 及 v2.3/deploy/OpenPI 相关回归共
**109 passed, 1 skipped**；deployment `--help`、相关模块 `py_compile` 和 `git diff --check`
通过。未启动相机、机械臂、夹爪、脚踏或网络控制。

### 前一批 writer/dispatch 收敛状态与验证结果

前一批实际改动集中在以下边界，未扩散到 robot driver、真机参数或控制策略：

- `teleop_runtime/collection/writer_runtime.py`：GUI 与 headless 共用
  `CollectionWriterObserver -> AsyncWriterPool` 的 sample/job/flush/close 生命周期。
- `teleop_runtime/session.py`：正常 `run_step()`、GUI recovery 和 headless 外部 command
  共用 `TeleopSession.dispatch_command()`。
- `teleop_runtime/collection/run_collect.py`：replay、return-home、recovery 等历史外部
  command 通过统一 seam 进入；旧 fake session 仅保留窄 fallback。
- `teleop_runtime/debug/run_vive_rm65_real_teleop.py`：接入共享 writer runtime，同时保留
  旧测试和诊断代码替换 `writer_observer` 的兼容能力。

GUI 的 episode、pause/resume、异步 sample、finalize、debug metadata，以及 headless 的
sample gate、replay、键盘和 finalize 行为均保留；`TeleopSession` 的控制频率、fault/hold
规则、`BimanualRig` dispatch 和真机参数没有改变。

验证结果：

- collection、real teleop、deploy、recovery、queue、lifecycle 回归：**351 passed, 2 skipped**；
- GUI、headless collection、deploy 三个模块的 `--help` 检查通过；
- 相关模块 `py_compile` 和 `git diff --check` 通过；
- 未启动真实相机、串口、机械臂、夹爪、脚踏或网络控制，未执行 motion、servo、force-control。

### 下一阶段统一控制入口的实施顺序

前一批解决了 writer 生命周期和现有 human/headless dispatch seam；本批又落地了 policy
loader 与“模型 observation view → v2.3 decoded absolute action”的共享边界。下一阶段首先
落地 ActionArbiter 的纯逻辑和兼容适配，而不是直接重写所有入口：

1. 完成 Phase 1A/1B 的 contract、时钟、authority、group selection、component catalog 和
   deterministic decision。
2. 用 synthetic source、fake resolver、fake dispatcher 覆盖 stale、timeout、takeover、hold、
   fault、coupled group、clock skew、queue empty、action space mismatch 和 mixed-source partial
   action。
3. 完成 human `TeleopCommand -> RobotAction` adapter 的 round-trip/golden tests；在
   `merge_teleop_commands()` 前保留 source/component proposal，并以 observe-only 接入
   `TeleopSession`。
4. **[已完成第一阶段]** `PolicyDeployRuntime` 已从 `run_policy` 的 normalization/decode 外围
   继续接管 `prepare_observation_for_policy_pose_mode`、`normalize_policy_output`、
   `convert_normalized_action_chunk_to_absolute`、policy queue、prediction transaction、policy
   schedule 和注入的 canonical-preparation seam。下一步必须在不改变这些语义的前提下记录未被
   选中的 policy proposal、raw model-view、decode anchor 和 decoded absolute action。
5. **[下一步，仍不做 sim]** 先在真机 dry-run/observe-only 语义下验证
   `model view -> absolute RobotAction -> canonical preparation -> existing executor`，然后在 GUI
   真机启用 policy shadow 和显式 human override；实现 `PolicyObservationView` 的 allowlist，确保
   ActionArbiter 以下的 selected/safe/planner/dispatch/trace 状态不回流给模型。
6. 为 `ArbitrationResult` 固定 control port/attribution port；按 `ActionComponentCatalog` 输出
   每个 component 的 owner、proposal、selection group、decision reason、available/selected/
   policy/expert/safety/command/measured masks，并由独立 `RewardAttributionAdapter` 解释，不让
   reward 逻辑进入 Arbiter。
7. 扩展现有 `PositionReferenceFrame`、`PositionCommandTick`、position audit 和共享 writer
   runtime 以承载 `representation_trace_id`、proposal/decision/anchor/contract hashes；不要新建
   relative timeline 或 policy 专属 writer。最终每个 tick 能关联 proposal、component owner、
   expert/policy masks、safe action、command id、dispatch receipt 和 measured state；在此之前不
   启动真机 policy-only。

ActionArbiter 阶段的完成条件：

- 不依赖 hardware、teleop_runtime、MuJoCo 或 GUI 的纯逻辑测试全部通过；
- human-only 现有 GUI/headless 回归行为不变；
- policy-shadow 能记录 policy proposal，但实际 selected/executed action 仍由 human 控制；
- actor 的 model-facing observation 不包含 Arbiter/safety/reference/dispatch/trace/reward 内部字段；
- 每个 `component_id` 都能独立生成 policy behavior、expert label、safety modification 和
  measured validity mask，reward/exporter 不需要从最终 action 数值反推 source；
- 过期、无效、fault、未 ready 和显式 takeover 缺少有效 proposal 时均 fail closed；
- 任意 selected action 的每个 component 都能追溯到 proposal、authority epoch、causal observation、
  safety result、command/receipt 和 measured validity；
- mixed-source 数采可以在不做 post-hoc 数值猜测的情况下重建 human/policy/replay 各自的 action
  部分；DAgger 只导出 `expert_label_mask` 有效部分，RL 只把 `policy_behavior_mask` 有效部分
  当作 policy behavior；
- mandatory action trace、decision 和 lineage 不会被异步 writer 静默丢弃，媒体丢帧有明确标记；
- 未引入第二个 policy loop、第二个 writer lifecycle 或新的 direct-rig dispatch 路径。
