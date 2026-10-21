# 运行域与可拆分边界

本项目仍使用同一个仓库和同一个 Python 3.10 Conda 环境，不以拆成多个 wheel 或可安装产品为目标。
这里的“可拆分”指：每条主链有独立 owner、单向依赖和稳定装配口，可以由不同开发者并行修改，
也可以在发布时明确说明包含了哪些运行域，而不需要复制 authority、dispatch、episode 或设备生命周期。

## 五条主链

| 主链 | 领域内核 | 应用装配 | 不应吸收的责任 |
| --- | --- | --- | --- |
| 数采 | `dataset_contracts`、`dataset_interface`、`collection_runtime` | `teleop_runtime.collection` 绑定真实 source、控制会话和兼容 CLI | policy/model worker、simulator、控制组合 authority |
| 一般部署 | `robot_motion.deploy` 提供动作、队列和执行契约；`deployments` 提供模型 client、协议和 artifact | `teleop_runtime.deploy` 绑定真实 observation、backend、安全与唯一 dispatch | `control_composition` session、MuJoCo runtime、第二套执行环 |
| 混合遥操与策略部署 | `control_composition` 提供 backend-neutral 的 source admission、组合、handoff、lineage 和非权威差异 | 当前由 `control_station.composition` / Sim-HIL profile 绑定 human、policy/replay 与 target | 模型 worker、replay cursor、canonical safety、dispatch、episode commit |
| 模拟环境 | `sim.contracts`、`sim.rm65`、`sim.mujoco` 提供场景、状态、task、物理和 sim target | 仿真 CLI 或 Control Station 的显式 sim profile | 真实设备初始化、`teleop_runtime`、真实硬件 backend |
| 外部 Agent Skill | `agent_gateway.protocol/runtime/transport` 提供上层框架中立的 Skill/schema、接入、admission、reconciliation 和 operation control plane；`agent_gateway.skills` 提供目标无关 Skill 定义 | `agent_gateway.profiles` 绑定具体 application、observation、executor、Route 和 deployment | 上层 Planner/Task/Memory、caller-authored route binding、canonical safety/IK/dispatch、episode owner |

依赖方向是：

```text
dataset_contracts / robot_motion
          ^
          |-- collection_runtime <-- teleop_runtime.collection
          |-- deployments ------- <-- teleop_runtime.deploy
          |-- control_composition <-- control_station composition profile
          |-- agent_gateway core  <-- agent_gateway concrete profile
          `-- sim --------------- <-- sim CLI / control_station sim profile

application/profile layers may compose multiple lower domains
lower domains must not import application/profile layers
```

`teleop_runtime.collection` 中直接运行的 SAPIEN 数据回放是一个应用叶子适配器，不是
`collection_runtime` 的依赖。类似地，模型专用参数可以在真实部署应用层引用
`deployments.openpi`，但 `deployments` 的模型/协议实现不能持有真实 backend 或控制循环。

## 跨域装配规则

跨域功能必须落在应用/profile 边界，而不是把另一个领域的运行时并入本领域内核：

- 一般部署可以单独运行，不导入 `control_composition` 或 `sim`；
- 混合部署复用一般部署产生的 immutable policy emission，不再 pop 一次 action queue；
- 仿真 target 实现状态、物理和执行，composition 只提交 candidate，不能成为仿真 dispatch owner；
- 数采观察唯一控制链并提交给唯一 episode owner，不能为了混合控制再建 writer/lifecycle；
- `control_station` 可以构造多个 lower package，但 lower package 不得反向导入它。
- `agent_gateway` core 不导入 Control Station、sim、硬件、teleop、model deployment 或上层 Agent framework；具体 profile 可以通过 typed port 装配这些应用事实，但 lower package 不得反向导入 Gateway。

任何跨域适配器都应能回答三个问题：输入由谁推进、最终 authority/dispatch 由谁拥有、资源由谁关闭。
如果答案出现两个 owner，应先修正所有权，而不是增加同步队列或兼容状态。

真实遥操的 `RealTeleopControlInput` 直接引用 lower runtime 的实体操作，不依赖入口模块导入后
回填全局函数。Control Station 在 `run_real_teleop()` 中显式传入 progress application、输入批次
优先级和控制可用性策略，由运行时按实例绑定；缺少依赖时在资源装配前拒绝启动，同一实例不能
换绑为另一组策略。可用性策略直接使用既有纯策略模块，snapshot/event 类型仍由该模块持有，
不新增策略容器或第二套状态定义。旧 CLI 与控制器导出保留，但不再改变 lower module 的依赖。

数采的 v2.5-p native shadow 由 `teleop_runtime.collection.v25p_dual_write_shadow` 统一处理
已规划 legacy row 的追加、正式封存后的投影及失败清理；线程、子进程和 UMI 入口只保留各自的
启用条件、状态发布与生命周期所有权。共享操作不重新规划样本，不调用正式 Writer 的封存，
影子失败不改写正式保存结果；UMI 仍提交与 Writer 相同的下一帧动作对齐样本。
有状态的入口通过窄回调在清理前发布操作错误，清理失败再追加错误；新报告在读取 receipt 前
交给任务，操作未返回新报告时不清空既有报告。抽取共用逻辑不能改变这些状态发布时点。
三种入口的内置 policy-I/O shadow 也由该模块统一构造，同一组 capture profiles 同时绑定
store、producer 和投影；保留 real-teleop 与 UMI 各自的动作时序语义。启用条件、数据集身份、
参与侧和存储格式仍由调用方明确传入，不新增 episode owner 或第二套配置状态。

真实遥操的正式 writer 由 `EpisodeRecordingRuntime` 的异步准备／数据集选择事务构造，
ready 后才发布；配置未变的 ready recorder 直接复用，不重新扫描已封存数据。全局启动不构造
writer，recorder 自身不再提供关闭后原地重建的第二套生命周期。旧的
`prepare_startup_dataset_recorder()` 与 `reconfigure_for_current_args()` 已退役；现行配置校验、
候选失败回收、旧 recorder 可恢复边界和单次进程启动约束保持不变。
控制循环上下文直接引用同一个 `EpisodeRecordingRuntime`，不再逐项重新声明和装配录制能力。
倒计时与事务过渡门控由该对象的只读视图提供；逐帧调用顺序和 provider 刷新时点保持不变。

取消后的准备结果在清理线程无法启动时也不能重新进入 ready。控制面保留精确候选和清理
闭包，阻止新的接纳；poll 的清理重试间隔至少 50 ms，显式 close 可以尝试排空。准备、
启动和停止操作失败后的关闭错误同样保留资源，并保留原始业务错误；清理恢复不重复发布
原操作错误，避免重复触发停止或状态提交。实际 recorder 的关闭是可重复的 best-effort 操作。

关闭调用的 False 不能当成依赖释放屏障。后台工作仍占用资源时，通用关闭入口抛出
`CollectionShutdownPendingError`，其 `runtime_state` 保留相机、session 与原录制 owner；
主循环 watchdog 先停止，原始异常或 Ctrl+C 保留该 pending 错误而不被替换。调用方需要
对原录制 owner 重试有界 close，确认过渡已结束后再进入既有通用关闭入口；未排空不报告
关闭完成，也不自动创建新的录制/设备 owner。资源清理完成不改写先前已报告的业务失败。

数据集切换的旧 writer 回收同样保留原事务与控制面：线程无法启动、关闭报错或等待超时
均返回未完成，后续有界 close 继续消费原结果或重试，不在调用线程同步关闭。
准备失败但尚未消费的结果也须有界消费；可恢复的旧 writer 由原事务移交既有回收队列，
过渡视图同时覆盖未消费准备事务和回收 owner，完成回调仅在回收成功后发布一次。

每次有界关闭都继续消费同一 open/stop 的结果；等待超时不关闭控制面或丢弃未消费结果。
打开前配置拒绝的旧 writer 由发布方移交既有回收队列；已打开的 episode 则沿正常
stop/finalize 流程发布 saved/error，保存等待超时仍保留依赖，后续 close 继续封存。
关闭期间拒绝新打开、倒计时和数据集选择；不增加后台自动关闭 owner。

## 执行模式边界

执行模式按同时参与控制决策的 source 数量划分，而不是按是否启用 GUI、trace 或被动 shadow 划分：

| 执行模式 | source/入口 | 最终执行 owner | `control_composition` |
| --- | --- | --- | --- |
| 单源策略部署 | `teleop_runtime.deploy` | 既有 deploy safety/executor 与真实 backend | 不加载、不创建 |
| 单源真人遥操 | canonical real-teleop frontend | 既有 real-teleop session、Rig 与真实 backend | 不加载、不创建 |
| 单源 replay | `teleop_runtime.replay` | replay executor、`ReplayHardwareRig` 与真实 backend | 不加载、不创建 |
| 单源 Agent-skill | MuJoCo direct target；独立 RM65 Skill profile 的真机 TCP primitive/URDF 避障抓取接近 | target runtime 的 canonical safety、IK 与 dispatch | direct target 不加载、不创建 |
| 多源组合 | 显式 Control Station composition/Sim-HIL profile | 同一个 target runtime；composition 只产生 candidate | profile 显式加载并创建 |

被动 observer、ControlTrace、difference 或 shadow 不会因为记录了 source/output 事实就变成多源控制；
它们不得获取 authority、dispatch token，也不得让默认单源入口 eager-load `control_composition`。当两个或更多
source 同时参与动作选择、融合或 handoff 时，应用/profile 必须显式创建 composition runtime，并把结果提交给
原 target 的唯一 safety/dispatch owner，不能另建执行环。

## 当前迁移状态

- `deployments.openpi.client.real` 保留为历史命令兼容入口；实现由
  `teleop_runtime.deploy.openpi_application` 持有。前者不得重新加入参数解析、设备配置或运行循环。
- `teleop_runtime.deploy`、canonical real-teleop frontend 和 `teleop_runtime.replay` 的默认单源入口当前不会
  eager-load `control_composition`；显式 composition binding 仍按需加载。
- `tests/test_runtime_domain_boundaries.py` 在干净解释器中分别导入上述三个单源入口，并以显式 MuJoCo
  composition binding 作为正向对照；该证据证明 import/contract 边界，不代表真机运动验收。
- `control_composition` 当前不依赖 collection、model/provider、sim、hardware、teleop 或 Control Station。
- `agent_gateway` 已是顶层包：`protocol/runtime/transport` 保持上层框架与具体 backend 中立，目标无关 Skill 定义位于 `skills/`，当前 MuJoCo application、motion/policy executor、observation 和 CubeLift deployment 位于 `profiles/mujoco/`；default-off 的真实 TCP-pose 只读 attachment 位于 `profiles/real/`。历史 `control_station` import 与 CLI 只保留无状态兼容转发。
- `sim` 当前不依赖真实 teleop、hardware、collection runtime、model provider 或 Control Station；OpenPI 与
  MuJoCo observation 的接线由 `control_station.sim_hil_profile.openpi_policy_source` 持有。
- no-model Agent-skill 模拟目标由 `sim.mujoco.deploy.agent_skill_target` 持有，不再与 composition-specific
  Sim-HIL target 实现在同一文件；原 `sim_hil_target` symbol 保留为按需加载、对象 identity 不变的兼容出口。
- Agent-skill 与 composition Sim-HIL 共用的坐标身份、FK observation 和场景测量由
  `sim.mujoco.deploy.target_measurement` 持有；两个 target factory 只消费这些无状态 adapter，原
  `sim_hil_target` 与 package-root 导出保持同一对象的兼容视图。
- 两类 target 共用的 scene load/reset、actuation/timing profile 与 prepared-resource 生命周期由
  `sim.mujoco.deploy.target_preparation` 持有；`sim_hil_target` 和 `agent_skill_target` 都只在已准备资源上
  绑定各自 runtime，旧配置、prepared target 和 prepare 函数导出保持 identity 不变。
- 两类 target 共用、但不执行资源生命周期的选择 contract 由 `sim.mujoco.deploy.target_config` 单独持有；
  preparation 与两个 runtime factory 均依赖该配置，旧 `sim_hil_target`、`target_preparation` 和 package-root
  配置导出仍指向同一类型。
- deterministic async composition harness 已移到 `tests/support/mujoco_async_topology.py`，不再作为
  `sim.mujoco.deploy` 的运行时导出或发布内容。
- `sim/mujoco/deploy/composition_binding.py` 现在集中构造可选 composition runtime、retention 和 trace
  projection；`MujocoDeployRuntime.composition_port` 是带 target 生命周期门禁的窄 receiver，只暴露
  context/input/eligibility/dynamic-parameter admission 与 shadow evaluation。Control Station 已经由该 port
  调用这些能力；`policy_topology` / `composition_strategy_control` 分别保留为同一底层 topology/strategy
  对象的兼容视图。
- `MujocoDeployRuntime.authoritative_port` 是 application-facing 的 target authority 边界：只暴露不可变
  profile identity、receiver/gate/epoch/owner snapshot，以及显式 trace、proposal、producer、authority
  admission 和 canonical-safety evaluation。它不暴露 coordinator、registry、scheduler、proposal buffer 或
  dispatcher；Control Station 不再读取或传递 `MujocoDeployRuntime.core`，lower runtime 仍是 authority、
  epoch、canonical safety 和 dispatch 的唯一状态所有者。
- dynamic Sim-HIL strategy revision、alpha control 和其 retention factory 由
  `control_station.composition.mujoco_sim_hil` 构造；`sim.mujoco.deploy.sim_hil_target` 只接收 application
  提供的 callable，动态 contract 缺少该 binding 时在 target 资源构造前 fail-closed，不再在 sim 内重建
  application composition 语义。
- information plan 与 source-eligibility enablement 已从一般 `MujocoDeployConfig` 移到可选
  `MujocoCompositionRuntimeConfig`。Control Station 根据已选 contracts 显式构造该配置，Sim-HIL target 只校验
  binding 完整性并转交；具体 retention window 仍由 MuJoCo composition binding 按 target cadence 实现。
  未提供该配置的一般 MuJoCo/Agent-skill runtime 不创建 composition topology、retention 或 trace emitter；
  兼容属性为 `None`，误用 composition 兼容方法时显式失败。启用 composer authority 或注入 strategy/schema
  时必须同时显式提供该配置。一般 `sim.mujoco.deploy` 导入也不再 eager-load `control_composition`；只有请求
  composition config、execution-difference observer 或实际启用相应 runtime 时才加载这些可选集成。
- proposal promotion 的 target catalog/group/schema 与期限策略已收敛到
  `sim.mujoco.deploy.composition_promotion`；它是无状态 target policy，不做 admission、序号推进、authority、
  safety 或 dispatch。proposal 幂等缓存、admission、target state、canonical safety 和 dispatch 继续只由
  `MujocoDeployRuntime` 持有，窄 port 不暴露这些操作。旧 composition 提交/评估方法目前仅保留为薄兼容
  转发；兼容 `compose_and_submit()` 的 evaluation 也统一通过该窄 port，只有 target-owned proposal
  admission 可直接进入私有 topology。composition 依赖禁止扩散到 sim contracts、task、physics、actuation
  和基础 runtime。
- `control_station/composition` 仍是 compatibility-scoped 的应用装配区。迁移时只保留 source/target/profile
  绑定，可复用控制语义回到 `control_composition`，MuJoCo 事实与执行回到 `sim`。

## 开发与发布口径

Conda 仍是唯一受支持的环境安装方式；`pyproject.toml` 中的 extras 只表达依赖族，不代表独立产品。
一次领域发布应列出被改变的领域内核、应用装配和兼容入口，并分别运行相应的 contract/entrypoint 测试。
不要因为同仓库、同环境就允许任意反向 import，也不要为了“独立发布”复制共享动作、安全或数据 schema。

边界回归由 `tests/test_runtime_domain_boundaries.py` 以及各领域现有 boundary tests 共同保护。
本页描述当前已实施的依赖规则；未完成的拆分会明确标为迁移债务，不能据此宣称已完成运行时集成。

部署后台 worker 的获取、关闭重试、严格停止证明与后续 RTC 接入约定见
[RTC 后台资源所有权与退出屏障](../development/rtc_worker_ownership.md)。
