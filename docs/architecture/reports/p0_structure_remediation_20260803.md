# DexTeleop P0 项目结构治理历史实施台账（2026-08-04～08-07）

> 文档类型：历史实施台账。本文中的“当前”、文件规模、并行状态、测试数字和下一批均只描述
> 2026-08-04～08-07 的审计与实施时点，不再作为当前工作树事实。现行结构治理入口见
> [P0 项目结构治理当前门禁](../plans/dexteleop_p0_structure_remediation_plan_20260803.md)。
> 本文保留原批次顺序、证据和命令用于追溯，不授权重新执行硬件、数据变换或历史批次。

日期：2026-08-04（更新至 2026-08-07）

性质：当前工作树审计后持续维护的结构治理与实施计划

适用入口：RealTeleop GUI、headless collection、policy deploy 及共享数据/控制契约

## 0. 当前实施状态

截至 2026-08-07 的实施结果：

- **P0-G 已完成**：progress catalog 基线恢复；tactile recompression 机制改由
  `collection_runtime` 拥有，`utils` 只保留兼容导出；新增反向依赖门禁。
- **P0-DOC 已完成**：数据版本文档归入 `docs/data/v25p2/`，Pose→QPOS 离线评估归入
  `docs/real/reports/`，`showcase/` 和 reports 均有领域索引；`.gitignore` 只对
  `docs/data/**` 恢复版本控制，不放开真实数据目录。
- **P0-4B 已完成**：component schema、formal binding、compound-authority 规则已迁入
  `dataset_contracts/control_trace_validation/component_authority.py`。
- **P0-4C 四个子切片已完成**：action-adaptation request、capability、candidate、receipt
  位于 `action_adaptation.py`；publication attempt、execution directive 与 adapted
  shared-safety 绑定位于 `execution_directive.py`；target request、submission、completion、
  target admission/activation 闭合校验位于 `target_execution.py`；cancellation request、ack、
  backend HOLD permit 闭合校验位于 `execution_cancellation.py`。各子切片复用共享 context，
  没有新增平行的公共 facade、epoch、HOLD、dispatch 或执行所有权。
- **P0-4E 已完成**：session spec/rearm/epoch、transition、
  state-hash 连续性及 terminal/supersede 闭合规则已迁入
  `control_trace_validation/composition_session.py`，原实现已从 facade 删除，公共入口与
  issue tuple 不变；composition evaluation、source/proposal、replay lineage 与
  arbitration/shadow 也已分别迁入各自唯一 owner，最终 facade 收尾完成。
- **P0-4D 已完成**：reference schema bridge、receiver/segment/admission、generation
  reservation/activation、planner frame、terminal 及 scheduled closure 已统一迁入
  `control_trace_validation/reference_lifecycle.py`。这些规则共享同一索引闭环，因此保留为
  一个语义 owner，没有人为拆成 segment/activation 转发模块；原实现已从 facade 删除，
  固定 issue tuple 不变。
- **P0-4E composition-evaluation 子切片已完成**：composition input/admission、snapshot、
  numerical evaluation、activation scope/binding 与 authority disposition 校验已迁入
  `control_trace_validation/composition_evaluation.py`。session 状态仍由
  `composition_session.py` 拥有，后续 proposal/arbitration 只复用只读索引，不存在第二套
  evaluation 实现。
- `dataset_contracts/completeness.py` 当前 1,024 行，
  `evaluate_control_trace_lineage()` 当前约 130 行，只负责 context/index owner 构建、
  rule-family 调度与稳定结果汇总。
- **manifest/reference 历史版本阻塞已关闭**：2026-07-21 RM65 field evidence 按其
  `evidence_schema` 解析固定的 hash-only manifest/inventory/worker-drain identity；当前
  builder 独立验证当前合同的确定性。历史 payload 当时未归档，因此不伪造历史 builder，
  也不再要求历史 evidence 等于当前默认 builder。
- RM65 solver 相关结构调整正在另一并行工作中，本计划继续将 solver 及其直接测试/接线
  视为占用区，不触碰 P0-2。
- **P0-1B RealTeleop core 依赖反转已完成**：新审计确认 operator/runtime 只依赖
  `ProgressApplicationPort`，Station adapter 在 canonical GUI application edge 构造并注入。
  剩余 GUI/web 文件是显式兼容 shim；UMI 两条 lower→Station 依赖另列后续迁移。
- 下一阶段是 P0-1C UMI GUI 依赖审计，或在 RealTeleop 活跃功能簇稳定后进入 P0-3；当前
  RealTeleop/admittance/startup/shutdown 路径仍被并行修改，不直接启动 P0-3。

## 1. 结论

原 P0 顺序不能继续原样执行。工作树在本次审计期间仍持续变化，RealTeleop、连续轨迹、
RM65 solver、后标注、相机语义和触觉存储均有并行开发。如果现在继续按“巨型文件优先”
机械拆分，会把功能迁移、所有权迁移和行为变化混在同一批差异中。

新的执行顺序为：

1. **P0-G：先收敛并行基线和新增边界回归。**
2. **P0-4：继续拆 `dataset_contracts/completeness.py`，但一次只迁移一个纯校验族。**
3. **P0-1B：等 RealTeleop progress/application 变更稳定后，完成 Station 依赖反转。**
4. **P0-3：再按资源生命周期缩短 RealTeleop orchestration。**
5. **P0-2：最后执行 RM65 solver 模块化，并以独立 solver 方案为准。**

P0-G 未通过前不得开始新的生产结构迁移；该门禁现已通过。后续每个 P0-4 切片仍需独立
通过 characterization 和集成测试，不能把本次绿色结果外推到未测试路径。

## 2. 当前工作树快照

本节是 2026-08-04 审计时的点状证据，不是假定工作树已经冻结。

- 审计末段：122 个追踪文件有差异，61 个未追踪条目；追踪差异约
  `+9721/-1240`。
- 审计开始时只有 118 个追踪差异文件，说明并行开发在本轮检查过程中仍继续写入。
- `git diff --check` 在计划改写前通过。
- 本轮没有启动设备、机械臂、相机、踏板、触觉进程或真实控制循环。
- canonical RealTeleop `--help` 返回 0；此前
  `build_real_teleop_startup_progress_ui_payload` 的 ImportError 当前没有复现。

### 2.1 当前主要变更簇

| 变更簇 | 主要路径 | 当前判断 |
|---|---|---|
| RealTeleop progress/application | `teleop_runtime/real_teleop/*`、`control_station/real_teleop_application.py` | 高重叠，冻结 P0-1B/P0-3 |
| 连续轨迹与 replay/deploy | `robot_motion/position_control/*`、`robot_motion/replay/*`、`sim/mujoco/replay/*` | 权威执行语义活跃，禁止同时重构 |
| Chunk-relative HITL | `control_composition/trajectory/*`、`robot_motion/action/trajectory.py` | 纯契约/reducer 已有测试，正式接线尚未稳定 |
| RM65 IK/安全/位置控制 | `dual_arm_target_solver.py`、official IK governor、backend | 真机安全关键，P0-2 最后 |
| Progress/retrospective 数据 | `dataset_contracts/*progress*`、`dataset_interface/*progress*`、prepared training | 包归属总体合理，继续做边界审计 |
| 相机语义与 OpenPI 兼容 | camera semantics、OpenPI client/server/training、设备配置 | 仍在写入，不与结构迁移混合 |
| 触觉压缩/冷存储 | `collection_runtime/*compression*`、`utils/dataset_postprocess/*` | 出现新的依赖方向问题 |
| refinement annotation | `control_station/refinement_annotation/*` | 应用类继续增大，先稳定功能和草稿保护 |

## 3. 旧计划已经完成的部分

### 3.1 P0-1A 已完成

以下 lower-owned 模块不再直接导入 Station：

- `teleop_runtime/real_teleop/dataset_recorder.py`
- `teleop_runtime/real_teleop/progress_arm_control.py`
- `teleop_runtime/real_teleop/control_loop.py`
- `teleop_runtime/real_teleop/progress_adapter.py`

进度命令、Hold/Pause receipt 和 lower-facing Protocol 已进入
`teleop_runtime/real_teleop/progress_contracts.py`；Station 的 UI/application binding 位于
`control_station/real_teleop_application.py`。这解决了此前进度函数错误导入的直接故障，
但依赖反转还没有完成。

### 3.2 P0-4 基础拆分已完成

`dataset_contracts/completeness.py` 已从更大的单体降到 2,983 行，且
`evaluate_control_trace_lineage()` 仍有 2,097 行。现有目录：

```text
dataset_contracts/control_trace_validation/
  context.py
  issue_sink.py
  runner.py
  preflight.py
  station.py
  membership.py
  context_lineage.py
  source_eligibility.py
  component_authority.py
  action_adaptation.py
  execution_directive.py
  target_execution.py
  execution_cancellation.py
  composition_session.py
  composition_evaluation.py
  source_proposal.py
  replay_lineage.py
  arbitration_shadow.py
  reference_lifecycle.py
```

已经迁出的规则族：

- profile/manifest preflight；
- Station lineage；
- Episode membership；
- policy/composition context lineage；
- composition source eligibility。
- component schema、formal binding 与 compound authority。
- action-adaptation request/capability/candidate/receipt lineage。
- publication attempt、execution directive 与 adapted shared-safety lineage。
- target execution request/submission/completion、target admission/activation lineage。
- execution cancellation source/epoch/ack、backend HOLD permit lineage。
- composition session spec/rearm/epoch、transition/state 连续性及 terminal/supersede lineage。
- composition input/snapshot/evaluation、activation binding 及 authority disposition lineage。
- policy/human/composer source、proposal admission 与 profile closure lineage。
- replay contract/assessment/eligibility、candidate authorization、receipt 与 profile closure。
- arbitration manifest binding、snapshot/tick/input/computation/evaluation 与 passive shadow closure。
- reference receiver/segment/admission、generation/activation/frame/terminal 及 scheduled closure。

公共入口、`ControlTraceLineageIssue.__module__` 和最终 issue 排序仍由
`dataset_contracts.completeness` facade 保持。

## 4. 当前新增 P0-G：并行收敛与边界恢复

### 4.1 为什么它现在是最高优先级

初次重规划时不是一个可直接结构重构的稳定基线：

- 审计期间源文件仍持续变化；
- progress catalog 新增 profile 后，两个既有断言仍固定期待 6 个 profile；
- `collection_runtime/tactile_recompression.py` 反向导入
  `utils.dataset_postprocess.tactile_recompress`；
- 现有 `test_collection_runtime_boundaries.py` 未禁止 `collection_runtime -> utils`，
  因而测试全绿也无法阻止该方向继续扩散。

上述 progress catalog 和依赖方向问题已在 P0-G 处理；本节保留为问题来源和回归约束。

### 4.2 P0-G0：建立路径级稳定窗口

每个并行变更簇在进入结构治理前提供一份轻量收据：

```text
lane 名称
拥有者/目的
正在写入的路径清单
公共合同或默认行为是否变化
最小测试命令与最近结果
预计可释放给结构重构的路径
```

不要求清理、stash 或 reset 工作树，也不要求整仓提交。稳定判定使用路径级 manifest：

1. 连续两次审计之间目标路径的 diff 不再变化；
2. 目标路径对应测试通过；
3. 没有其他 lane 声明将继续修改同一路径或同一公共合同；
4. 结构批次只触碰自己的 manifest，不顺手格式化相邻文件。

### 4.3 P0-G1：恢复当前绿色基线

先由 progress/profile 变更簇处理当前两项失败：

- `test_progress_gui_is_visible_and_selectable_before_runtime_home`
- `test_progress_gui_keeps_profile_selection_available_for_any_formal_task`

需要确认新 retrospective profile 是否应进入 RealTeleop 可选 catalog：

- 如果应进入，测试改为验证稳定排序、必要 profile 集合和字段，不再硬编码总数 6；
- 如果不应进入，catalog 必须按用途/可见性合同过滤，而不是在测试里忽略。

这属于功能变更簇的合同决策，不应混入 P0 文件移动提交。

### 4.4 P0-G2：修正 tactile recompression 依赖方向

目标依赖方向：

```text
teleop_runtime real-teleop adapter
  -> collection_runtime idle lifecycle / tactile persistence mechanism
  <- utils.dataset_postprocess CLI / offline orchestration
```

不得保持：

```text
collection_runtime -> utils.dataset_postprocess
```

推荐处理：

1. 将设备无关、可取消、原子校验的 recompression 机制归入
   `collection_runtime` 的能力命名模块；
2. `utils.dataset_postprocess` 只保留 CLI、计划展示和离线调用 adapter；
3. RealTeleop 只负责 idle admission/revocation，不复制变换实现；
4. 增加边界测试，禁止 `collection_runtime` 导入 `utils`；
5. 保留新 inode、帧数/尾帧解码、hash、原子替换和取消语义。

本阶段只调整 ownership，不改变默认启用状态、压缩参数或删除条件。

### 4.5 P0-G 验收

- 目标路径在约定稳定窗口内不再变化；
- 当前 targeted suite 全绿；
- canonical RealTeleop `--help` 仍返回 0；
- lower package boundary 不再容许新增 `collection_runtime -> utils`；
- 没有设备动作或真实数据变换。

## 5. P0-4：继续拆纯 ControlTrace 校验器

这是 P0-G 后唯一可以优先推进的低冲突巨型文件治理，因为它是纯内存校验，不拥有设备、
执行队列、安全、reference 或 episode commitment。

### 5.1 不变量

- `evaluate_control_trace_lineage(facts, profile=...)` 签名不变；
- `ControlTraceLineageIssue` 的公开 import path 和类型身份不变；
- issue code、detail、root/related fact IDs、排序和重复项行为不变；
- 不在结构迁移中新增、删除或放宽 ControlTrace 规则；
- context/index 只构建一次，不让每个 rule module 重扫全量 facts；
- 每次只迁移一个闭合规则族。

### 5.2 新顺序

#### P0-4B：component schema、formal binding、compound authority（已完成）

覆盖当前约 1,249 行附近开始的关联分支：

- component action schema catalog/payload acceptance；
- formal component binding；
- conservative compound authority profile/failure；
- 与这些事实直接绑定的 composition spec/rule/strategy 引用。

迁移前先补齐 characterization fixture，冻结 issue tuple。建议模块名按真实语义命名，
例如 `component_authority.py`，不要创建 `part2.py`。

停止条件：HITL/trajectory lane 正在修改这些 event schema 或 hash domain。

#### P0-4C：adaptation、arbitration、execution handoff（四个子切片已完成）

第一子切片已迁移 request/capability/candidate/receipt；第二子切片已迁移 publication attempt、
directive 与 adapted shared-safety 绑定；第三子切片已迁移 target request、submission、
completion、target admission/activation 绑定；第四子切片已迁移 cancellation source、epoch、
required ack 和 backend HOLD permit 绑定。`robot_motion.runtime.cancellation` 继续唯一拥有
epoch/cancellation，dispatch owner 继续拥有 permit，validator 只检查证据闭合。

#### P0-4D：reference lifecycle（已完成）

reference receiver validation、segment、admission、generation reservation/activation、frame、
terminal 与 scheduled closure 已迁入唯一 `reference_lifecycle` owner。迁移前以两组
characterization 固定正常 reference lifecycle 的完整 issue tuple 和 scheduled segment
缺 admission 的精确 issue tuple；迁移后两组结果不变。该批没有修改
`control_trace.py`、runtime emitter、MuJoCo receiver、reference activation 或 dispatch 行为。

验证边界：ControlTrace 主流程 51 项、MuJoCo formal/deploy 47 项、包边界 49 项、
ControlTrace persistence/reader/reference contract 31 项通过；这是纯内存/仿真测试证据，
不代表真实设备或运动验证。

#### P0-4E：session state chain 与 facade 收尾（已完成）

composition session transition、rearm、epoch/state 连续性以及 composition input/evaluation/
activation/disposition、source/proposal、replay lineage 与 arbitration/shadow 子切片均已完成。
source/proposal 拥有通用 proposal/admission/validation 索引；replay 只读取这些共享索引并拥有
自己的 contract/eligibility/receipt；arbitration 读取 proposal 索引并拥有完整 passive shadow
状态图，没有形成新的混合巨型模块。

最终 facade 只负责 requirement profile、公共 issue 类型、context/index owner 构建、
rule-family 调度、稳定排序和 completeness report。迁移前后 characterization 20 项、
ControlTrace/MuJoCo 82 项、边界及 persistence/reference contract 80 项通过。新增
characterization 已拆入独立 tail 文件，避免继续扩大原测试单体。

### 5.3 每个切片的提交纪律

每个切片只包含：

- 一个新/扩展的 rule module；
- `completeness.py` 的等量迁出；
- 对应 characterization fixture；
- 必要的窄边界测试。

不得同时修改 `dataset_contracts/control_trace.py` event schema、运行时 emitter 或 deployment
接线。若必须改语义，结束当前结构切片，另开功能变更。

## 6. P0-1B：完成 Control Station 依赖反转（RealTeleop core 已完成）

### 6.1 当前事实

2026-08-07 再审计确认旧计划已部分过时：

- `operator_control_runtime.py` 不再导入 `control_station`，只依赖 lower-owned
  `ProgressApplicationPort`、command 与 receipt；
- `run_real_teleop(..., progress_application=...)` 接收显式注入；
- canonical GUI/debug entrypoint 作为 application edge 构造
  `StationProgressApplication`，依赖方向合理；
- `teleop_runtime/real_teleop/gui_controls.py` 只是
  `control_station.real_teleop.control_availability` 的兼容 re-export，仍有稳定 import-path
  合同测试，因此当前不能迁入 `legacy/`；
- allowlist 中另外两个真实 lower→Station 依赖位于 UMI app/controller，应作为独立 P0-1C，
  不能为了缩小 allowlist 把 UMI 与 RealTeleop runtime 混改。

### 6.2 目标结构

```text
canonical debug/GUI entrypoint
  -> 构造 StationProgressApplication
  -> 以 ProgressApplicationPort 注入 RealTeleop runtime/operator core

teleop_runtime core
  -> 只认识 ProgressApplicationPort、ProgressMutationCommand 和 lower receipts
  -X-> 不导入 control_station
```

上述目标对 RealTeleop core 已实现。兼容 shim 只有在生产、测试、文档调用者均迁移且有明确
删除条件后才能移除；不为了“零 allowlist”复制 GUI policy 或破坏历史入口。

### 6.3 后续条件

- P0-1C 先审计 UMI app/controller 的 GUI resource 生命周期与 headless 兼容要求；
- P0-3 仍要求 `operator_control_runtime.py`、`control_loop.py`、`runtime.py`、Active
  admittance、idle tactile compression 与 startup/shutdown 工作簇退出活跃写入；
- 在此之前只允许只读调用边审计和窄合同修复，不做 RealTeleop 主类拆分。

## 7. P0-3：按资源生命周期缩短 RealTeleop orchestration

当前热点：

| 文件 | 行数 | 最大定义 |
|---|---:|---:|
| `runtime.py` | 2,644 | `run_real_teleop()` 2,275 行 |
| `control_loop.py` | 3,909 | `run_real_teleop_control_loop()` 3,297 行 |
| `operator_control_runtime.py` | 5,693 | 主类 5,468 行 |
| `component_builder.py` | 2,253 | `build_real_teleop_components()` 2,044 行 |

这些文件不能按固定行数切割。目标 owner 仍是一个 runtime，但内部按资源生命周期分解：

1. startup/preflight/cleanup registry；
2. component construction 与显式 close contract；
3. operator intent admission；
4. collection/progress episode effects；
5. control-step orchestration；
6. canonical safety/Hold dispatch；
7. shutdown、flush 和 failure aggregation。

优先提取无状态 projection、窄 adapter 和成对 `start/stop` 资源；最后才缩短主循环。任何子模块
都不得获得第二套 authority、safety、reference、dispatch、writer commit 或 episode owner。

启动条件：P0-1B 完成，RealTeleop 功能 lane 绿色且目标路径稳定。

## 8. P0-2：RM65 solver 最后处理

`robot_motion/rm65/dual_arm_target_solver.py` 当前 10,974 行，主类约 9,892 行；同时仍有
`+181/-5` 的 wrist q5 boundary/安全相关并行修改。它与新增的 sample-aligned trajectory、
official IK governor、位置 profile 和 backend 变更共同影响真实动作。

因此本方案不重复设计 solver 拆分，后续以
`docs/real/plans/rm65_dual_arm_solver_modularization_plan.md` 为唯一详细方案，且必须满足：

- 连续轨迹和 wrist boundary 功能变更稳定；
- deterministic fake-solver characterization 冻结 candidate 名称、顺序、scale 和 qpos；
- latch/Hold/reset/timeout/collision query 顺序均有离线证据；
- 第一批只提取纯 geometry/diagnostics/value objects；
- facade 仍是唯一 accepted qref 和 policy-state commit owner；
- 不执行真机运动作为结构重构测试。

## 9. 新增热点观察清单

这些文件已呈现单类/单函数膨胀，但当前属于活跃功能开发，不立即升格为独立 P0：

| 文件 | 当前规模 | 处理策略 |
|---|---:|---|
| `robot_motion/position_control/sample_aligned_trajectory.py` | 1,434 行；主类 1,357 行 | 连续轨迹功能稳定后再做纯 planner/model 分界 |
| `control_station/refinement_annotation/application.py` | 1,511 行；主类 1,439 行 | 先完成草稿保护、catalog 刷新和 portable adapter，再拆 query/draft/commit projection |
| `control_station/web/calibration_web_ui.py` | 约 323 KB | 保持 application leaf；后续单独审计 server/state/static ownership |

观察清单不授权顺手拆分；任何升级为 P0 都需先有独立 characterization 和路径稳定窗口。

## 10. 验证矩阵

### 10.1 通用门禁

```bash
git status --short
git diff --check
/home/zc/anaconda3/envs/dexteleop310/bin/python -m py_compile <本批修改文件>
```

### 10.2 P0-G / P0-1 / P0-3

```bash
/home/zc/anaconda3/envs/dexteleop310/bin/python -m pytest -q \
  tests/test_control_station_boundaries.py \
  tests/test_collection_runtime_boundaries.py \
  tests/test_real_teleop_progress_application_binding.py \
  tests/test_real_teleop_progress_adapter.py \
  tests/test_real_teleop_progress_workflow.py \
  tests/test_progress_arm_control.py \
  tests/test_real_teleop_foot_pad_controls.py \
  tests/test_real_teleop_refactor_contracts.py
```

再运行 canonical RealTeleop `--help`；不启动设备。

### 10.3 P0-4

```bash
/home/zc/anaconda3/envs/dexteleop310/bin/python -m pytest -q \
  tests/test_control_trace_lineage_characterization.py \
  tests/test_dataset_v24_control_trace.py \
  tests/test_control_trace_runtime.py \
  tests/test_control_trace_episode.py \
  tests/test_control_trace_shadow_projection.py \
  tests/test_composition_context_contracts.py \
  tests/test_composition_source_eligibility.py \
  tests/test_mujoco_formal_compound_authority.py \
  tests/test_relative_component_composition.py
```

### 10.4 P0-2

采用 solver 独立计划的 deterministic/offline 测试矩阵，并至少覆盖：

- dual-arm collision-aware safety；
- official all-IK governor；
- position profile/trajectory mode；
- replay/sim fixture；
- candidate 顺序和逐帧 metadata parity。

## 11. 本次审计测试证据

最新结果只证明对应命令执行时的工作树，不替代合并前复跑：

- P0-G、RealTeleop progress、tactile compatibility 及相关边界：
  **233 passed, 4 subtests passed**。
- P0-4 ControlTrace + package boundaries 最终矩阵：**153 passed**。
- P0-4B 首轮 component/compound-authority 定向测试：**41 passed**。
- P0-4C action-adaptation 定向测试：**58 passed**。
- P0-4C directive/shared-safety characterization：**10 passed**；ControlTrace、dataset 与
  package-boundary 聚焦矩阵最新复跑：**121 passed**。
- pose-HIL 固定网格晚到策略已闭合：`robot_motion/action` 的严格
  scheduled scheduler 在 START 已落后的 tick 上记录 `skip_to_next_legal_reference_tick`，
  并生成下一合法网格；MuJoCo receiver 继续以 `reject_and_hold` 拒绝直接晚到的 segment。
- M6 capability 已区分 historical candidate v1 与 exact-runtime v2；历史 RM65 evidence
  进一步使用显式 hash-only resolver，不再调用当前 builder。manifest、reference activation、
  inventory/drain、stop/idle-worker 相关矩阵 **66 passed**；历史 JSON/hash 未改写，runtime
  dispatch、safety 和 reference activation 行为未改变。
- P0-4C target-execution 子切片完成后，字符化、target handoff、component capability、formal
  MuJoCo 与 Station Sim-HIL 聚焦矩阵：**85 passed**；ControlTrace 存储/投影/读取及四个包边界
  的扩展矩阵：**148 passed**。`py_compile` 与 `git diff --check` 均通过。
- P0-4C cancellation 子切片：迁移前基线 **64 passed**；加入精确 issue-tuple 字符化并迁移后
  **65 passed**；ControlTrace、cancellation runtime 消费方及四个包边界扩展矩阵
  **199 passed**。MuJoCo/Station 扩展矩阵 **138 passed, 4 failed**，其中两个 RTC HOLD 用例
  单独复跑 **2 passed**。
- 上述 4 个失败均位于 `tests/test_mujoco_rtc_virtual_policy.py` 的旧读取 helper：测试仍从
  `selected_action.arm_qpos_by_side` 读取，而当前选择结果为 `ComponentAction`。该问题归属
  RTC virtual-policy/ComponentAction 测试适配及正在活跃的 trajectory lane，不由 cancellation
  validator 触发，本切片不跨模块修改。
- cancellation 子切片 `py_compile`、`git diff --check` 均通过；ControlTrace schema、
  authoritative trace emitter 与 runtime cancellation 文件哈希在两次审计间保持不变。
- `teleop_runtime.real_teleop.runtime` 直接导入与 canonical RealTeleop `--help`：均返回 0，
  未复现 progress payload ImportError，也未启动设备。
- P0-DOC：`docs/` 本地 Markdown 路径检查无断链；所有一级领域目录均有 `README.md`；
  `docs/real/teleop_args.sh` 通过 `bash -n`；旧 dataset/report 路径无残留；两个
  `release/` README 的直接回链和工作目录已同步修复。未运行设备或数据变换。

## 12. 推荐的下一批实际工作

严格按以下顺序：

1. 保持 RM65 solver 及其直接测试/位置控制接线为并行占用区；
2. 保持历史 RM65 evidence resolver 与当前 builder 分离；没有现场授权时不得重新采集、
   补写历史 payload 或用当前 hash 改写历史收据；
3. 由 RTC/trajectory lane 修正 virtual-policy 测试读取器或提供稳定的 ComponentAction
   解码接口，结构治理不抢占该并行路径；
4. reference validator/schema/emitter 的路径稳定收据已取得；冻结现有完整 issue tuple 后进入
   P0-4D reference lifecycle；
5. 若上述路径不稳定，停止生产迁移并转为只读审计；RealTeleop 大量路径仍活跃，不进入
   P0-1B/P0-3，solver 未释放前不进入 P0-2。

每个阶段完成后都重新读取当前工作树；本文件提供顺序和停止条件，不把 2026-08-04 的
点状快照当作未来事实。
