# 运行域与可拆分边界

本项目仍使用同一个仓库和同一个 Python 3.10 Conda 环境，不以拆成多个 wheel 或可安装产品为目标。
这里的“可拆分”指：每条主链有独立 owner、单向依赖和稳定装配口，可以由不同开发者并行修改，
也可以在发布时明确说明包含了哪些运行域，而不需要复制 authority、dispatch、episode 或设备生命周期。

## 四条主链

| 主链 | 领域内核 | 应用装配 | 不应吸收的责任 |
| --- | --- | --- | --- |
| 数采 | `dataset_contracts`、`dataset_interface`、`collection_runtime` | `teleop_runtime.collection` 绑定真实 source、控制会话和兼容 CLI | policy/model worker、simulator、控制组合 authority |
| 一般部署 | `robot_motion.deploy` 提供动作、队列和执行契约；`deployments` 提供模型 client、协议和 artifact | `teleop_runtime.deploy` 绑定真实 observation、backend、安全与唯一 dispatch | `control_composition` session、MuJoCo runtime、第二套执行环 |
| 混合遥操与策略部署 | `control_composition` 提供 backend-neutral 的 source admission、组合、handoff、lineage 和非权威差异 | 当前由 `control_station.composition` / Sim-HIL profile 绑定 human、policy/replay 与 target | 模型 worker、replay cursor、canonical safety、dispatch、episode commit |
| 模拟环境 | `sim.contracts`、`sim.rm65`、`sim.mujoco` 提供场景、状态、task、物理和 sim target | 仿真 CLI 或 Control Station 的显式 sim profile | 真实设备初始化、`teleop_runtime`、真实硬件 backend |

依赖方向是：

```text
dataset_contracts / robot_motion
          ^
          |-- collection_runtime <-- teleop_runtime.collection
          |-- deployments ------- <-- teleop_runtime.deploy
          |-- control_composition <-- control_station composition profile
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

任何跨域适配器都应能回答三个问题：输入由谁推进、最终 authority/dispatch 由谁拥有、资源由谁关闭。
如果答案出现两个 owner，应先修正所有权，而不是增加同步队列或兼容状态。

## 当前迁移状态

- `deployments.openpi.client.real` 保留为历史命令兼容入口；实现由
  `teleop_runtime.deploy.openpi_application` 持有。前者不得重新加入参数解析、设备配置或运行循环。
- `teleop_runtime.deploy` 的一般真实部署链当前不依赖 `control_composition` 或 `sim`。
- `control_composition` 当前不依赖 collection、model/provider、sim、hardware、teleop 或 Control Station。
- `sim` 当前不依赖真实 teleop、hardware、collection runtime、model provider 或 Control Station；OpenPI 与
  MuJoCo observation 的接线由 `control_station.sim_hil_profile.openpi_policy_source` 持有。
- no-model Agent-skill 模拟目标由 `sim.mujoco.deploy.agent_skill_target` 持有，不再与 composition-specific
  Sim-HIL target 实现在同一文件；原 `sim_hil_target` symbol 保留为按需加载、对象 identity 不变的兼容出口。
- deterministic async composition harness 已移到 `tests/support/mujoco_async_topology.py`，不再作为
  `sim.mujoco.deploy` 的运行时导出或发布内容。
- `sim/mujoco/deploy/composition_binding.py` 现在集中构造可选 composition runtime、retention 和 trace
  projection；`MujocoDeployRuntime.composition_port` 是带 target 生命周期门禁的窄 receiver，只暴露
  context/input/eligibility/dynamic-parameter admission 与 shadow evaluation。Control Station 已经由该 port
  调用这些能力；`policy_topology` / `composition_strategy_control` 分别保留为同一底层 topology/strategy
  对象的兼容视图。
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
