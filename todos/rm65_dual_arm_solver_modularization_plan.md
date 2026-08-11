# RM65 双臂 Target Solver 模块化整理方案

日期：2026-08-04
范围：`robot_motion/rm65/dual_arm_target_solver.py` 及其直接测试/兼容 facade
性质：仅重构规划；不改变真机控制参数、候选策略、数值结果或公共入口

## 1. 结论

当前 solver 应当拆分，但不能拆成多个可独立求解或提交 qref 的 owner。

推荐结构是：

```text
real teleop / sim / replay
        |
        v
dual_arm_target_solver.py
  唯一公共 facade、唯一 cycle orchestrator、唯一最终 commit owner
        |
        +-- dual_arm_solver/geometry.py
        +-- dual_arm_solver/solve_step.py
        +-- dual_arm_solver/candidate_model.py
        +-- dual_arm_solver/evaluation.py
        +-- dual_arm_solver/selection.py
        +-- dual_arm_solver/execution.py
        +-- dual_arm_solver/diagnostics.py
        `-- dual_arm_solver/policies/
              +-- continuity.py
              +-- singularity.py
              +-- joint_limits.py
              +-- inactive_side.py
              `-- hold.py
```

这些模块是 facade 私有持有的领域组件，不是新的运行入口。任何组件都不能绕过 facade
直接提交 qref、改变 episode、调用硬件或拥有第二套 canonical safety。

## 2. 当前审计结果

2026-08-04 工作树快照：

- `dual_arm_target_solver.py`：10,974 行，约 500 KB；
- `DualArmCollisionAwareRetargeter`：约 9,892 行；
- 主类方法：125 个；
- 主类访问的 `self.*` 名称：161 个；
- `DualArmCollisionAwareRetargeterConfig`：107 个公开字段，约 600 行；
- 当前文件仍有并行中的 `+186/-5` 未提交修改，主要涉及 wrist q5 boundary hold；
- `tests/test_dual_arm_collision_aware_safety.py` 大量直接调用或 monkeypatch 私有方法和字段。

当前最大的单方法为：

| 方法 | 行数 | 主要职责混合 |
|---|---:|---|
| `_solve_bimanual_core()` | 1,168 | cycle 初始化、求解、选择、平滑、commit、metadata、result/command 构造 |
| `_select_safe_tracking_candidate()` | 966 | deadline、候选分阶段生成、认证、选择、boundary/escape fallback |
| `_evaluate_dual_arm_step_candidate()` | 853 | 关节/碰撞/pose/进度/限位/奇异/非活动侧证书 |
| `_dual_arm_singularity_escape_candidates()` | 725 | 奇异结构分析、方向状态、宽搜索、候选构造 |
| `_smooth_output_qpos_before_action()` | 298 | EMA、fast path、q5 连续性、状态更新 |
| `_dual_arm_limit_escape_candidates()` | 271 | 限位方向、escape/boundary 候选 |

真正的问题是一个类同时承担：

1. public contracts 和兼容入口；
2. Pink/collision solver 的一次物理求解；
3. 候选生成、评估、选择；
4. wrist/奇异/限位/hold 多个有状态策略；
5. execution step、平滑和 admitted velocity；
6. profiling、shadow trace 和大量重复 metadata；
7. 最终 `BimanualIKResult` / `TeleopCommand` 构造与状态提交。

## 3. 必须保持的架构合同

### 3.1 唯一权威 owner

`DualArmCollisionAwareRetargeter` 继续是：

- 唯一 solve-cycle orchestrator；
- 唯一 accepted qref commit owner；
- 唯一 policy-state commit owner；
- 唯一 `BimanualIKResult` / `TeleopCommand` 发布 owner。

子组件可以返回候选、证书、选择结果或状态迁移提案，但不能自行 dispatch、写数据集、改变
operator lifecycle，也不能形成第二条部署/仿真专用求解链。

### 3.2 保持公共入口和类型身份

首轮重构必须保留：

- `robot_motion.rm65.dual_arm_target_solver` import path；
- `DualArmCollisionAwareRetargeter` 类仍定义在该模块中，保持现有 `__module__` 合同；
- `DualArmCollisionAwareRetargeterConfig` 的平面构造参数；
- `BimanualPoseTarget*`、`BimanualIKResult*`、`DualArmTargetContext`；
- `DualArmCollisionAwareTargetSolver*` aliases；
- `solve_target_pair()`、`build_command_for_context()`、`retarget()`；
- `teleop_runtime/retargeters/dual_arm_collision_aware.py` 的兼容 adapter；
- 当前 builder/time monkeypatch seam。

### 3.3 不改变数值时序

结构提取阶段不得改变：

- 一个 outer tick 内 Pink 的物理积分次数；
- candidate 名称、生成顺序、scale 顺序；
- deadline 与 reserve 检查位置；
- collision query 次数和顺序；
- q5、singularity、joint-limit latch 的更新时点；
- smoothing、execution cap、admitted velocity 的 commit 顺序；
- Hold 进入、seed 顺序和 recovery 时序；
- metadata 字段、嵌套结构与 fallback reason。

## 4. 推荐的内部模块边界

只在真实代码被提取时创建对应文件，不为了目录形式预建空模块。

### 4.1 `candidate_model.py`

拥有内部、不可 dispatch 的值对象：

- `DualArmCandidate`；
- `EvaluatedCandidate`；
- `CandidateBatch`；
- `CandidateEvaluationReceipt`；
- `CandidateSelectionReceipt`；
- 每 cycle 的不可变 `CandidateEvaluationContext`。

当前 `_DualArmCandidate` 和 `_DualArmEvaluatedCandidate` 可先从 facade re-export，保留测试
兼容。新类型不得持有 live backend 或硬件对象。

### 4.2 `geometry.py`

只放无状态纯函数：

- full-q 与左右 6-DoF 的 layout/split/embed；
- pose/frame/TCP 变换；
- pose metrics、joint margin、distance score；
- joint limit/velocity/acceleration vector 展开；
- 不依赖 `teleop_runtime`、sim 或设备。

该模块不能 import facade，避免循环依赖；必须从具体 sibling 模块导入常量，而不是通过
`robot_motion.rm65.__init__` 反向导入。

### 4.3 `solve_step.py`

封装一次本地 coupled IK 数值步骤及其 receipt：

- target 写入；
- single-step / legacy iteration 合同；
- time-budget stop reason；
- raw q、iteration count、pose error；
- branch-continuity anchor 和 admitted-motion limit 的数值接口。

它不做候选选择，不决定 Hold，不提交 accepted qref。若底层 solver 必须为 trial 临时改变
configuration，应由 facade 管理 scoped transaction：未 commit 的 trial 必须恢复。

### 4.4 `evaluation.py`

对应当前 `_evaluate_dual_arm_step_candidate()`，输入一个不可变 context 和 candidate，输出
完整证书 receipt：

- finite/shape/joint hard limit；
- execution step；
- operating margin；
- pose progress 与 bounded worsening；
- inactive-side hold；
- collision；
- singularity/limit/posture/wrist-unload 特殊 admission；
- rejection code 和 metadata fragment。

首轮提取保持原判定顺序，不能顺便“清理”布尔条件。长期目标是 evaluation 尽量纯函数化，
不直接更新 latch。

### 4.5 `selection.py`

对应当前 `_select_safe_tracking_candidate()` 的候选阶段编排：

- deadline/reserve；
- base candidate；
- joint-limit/wrist boundary；
- singularity/posture/wrist-unload auxiliary candidate；
- collision-safe pool；
- stable selection；
- reference fallback。

它返回 `CandidateSelectionReceipt`，但最终 accepted 状态仍由 facade commit。不要把它拆成
V1/V2 两个 selector；profile 差异继续由 resolved policy config 决定。

### 4.6 `policies/`

每个有状态策略拥有自己的窄 state dataclass：

- `ContinuityState`：wrist q4/q6 canonicalization、q5 hysteresis、output alpha；
- `SingularityPolicyState`：axis/direction latch、failure count、local metadata cache；
- `JointLimitPolicyState`：escape direction latch、release guard、boundary mode；
- `InactiveSideState`：fixed/inactive-side release 和 execution cap；
- `HoldState`：hold active、anchor q/target、frames 和 recovery seeds。

第一阶段允许组件按现有顺序原位更新自己的 state，以保证 parity。完成全量回归后，再考虑把
状态更新升级成 `propose -> facade commit` 的事务式流程。不要在同一次提交中同时做物理搬移
和状态语义重写。

### 4.7 `execution.py`

管理 solver 选择完成后的、dispatch 之前的 qref 整形：

- execution max step；
- inactive-side cap；
- q5 stabilization；
- adaptive/normal-fast smoothing；
- final canonicalization；
- admitted velocity 推导。

它只能生成 `ExecutionAdmissionReceipt`。facade 在所有最终证书通过后一次性 commit。

### 4.8 `diagnostics.py`

负责不参与控制决策的输出：

- profile timing；
- candidate/evaluation metadata 合并；
- left/right 公共 metadata 模板；
- `BimanualIKResult` side metadata；
- shadow trace payload。

当前 `_solve_bimanual_core()` 中左右 metadata 大量重复，优先抽成一个 side builder，可以在
不触碰 IK 判定的前提下显著缩短核心流程。

### 4.9 内部 resolved config

公开的 107 字段 flat config 首轮保持不变。`__post_init__` 验证后，可生成内部只读分组：

```text
SolveBudgetConfig
CollisionPolicyConfig
ContinuityPolicyConfig
SingularityPolicyConfig
JointLimitPolicyConfig
ExecutionPolicyConfig
InactiveSidePolicyConfig
DiagnosticsConfig
```

CLI、profile 和现有构造调用仍使用原 flat fields；内部组件只接收自己所需的 resolved
config，避免每个组件读取整个 107-field 对象。

## 5. 目标 solve-cycle 形态

最终 facade 的核心流程应能读成一条明确流水线：

```python
cycle = self._begin_cycle(...)
raw = self._solve_step.solve(cycle.solve_input)
candidates = self._candidate_pipeline.generate(cycle, raw)
selection = self._selector.select(cycle, candidates)
execution = self._execution.admit(cycle, selection)
commit = self._commit_cycle(cycle, raw, selection, execution)
return self._result_builder.build(commit)
```

这只是目标组织形态，不允许一次性重写 `_solve_bimanual_core()`。每个阶段先由 facade 的原
私有方法委托到新组件，保持旧测试和 monkeypatch seam，再逐步缩短 facade。

## 6. 推荐迁移顺序

### 阶段 0：等待当前并行修改稳定

当前 solver 有 wrist q5 boundary-hold 并行改动。开始物理拆分前必须：

1. 重新检查 `git status` 和该文件 diff；
2. 确认并行修改 owner 已完成或划定不重叠区域；
3. 不 reset/stash/覆盖当前 `+181/-5`；
4. 固定一个可回退 commit 或补丁基线。

### 阶段 1：建立行为 characterization

冻结以下可观测量：

- candidate 名称、来源、生成顺序和 scale；
- selected q、raw q、score、collision distance；
- solver/collision 调用次数；
- status、fallback reason、iteration stop reason；
- q5/singularity/joint-limit/hold state；
- output smoothing alpha、execution cap、admitted velocity；
- metadata（排除 wall-clock/perf timing 后深比较）。

样本至少覆盖：normal、inactive side、q5 boundary、joint limit boundary、singularity escape、
collision rejection、deadline fallback、jump Hold/recovery。加入 0802 偶发 branch jump 和最近
V2 boundary-hold 数据作为离线集成样本。

### 阶段 2：先提取纯代码和 diagnostics

依次提取：

1. `geometry.py`；
2. candidate/receipt 内部类型；
3. side metadata/result builder；
4. profiling/shadow payload builder。

这一阶段不移动任何 latch，不改变 selection。

### 阶段 3：提取 candidate evaluation

将 `_evaluate_dual_arm_step_candidate()` 的逻辑原样迁移到 evaluator。facade 保留同名私有
wrapper，测试仍可调用/monkeypatch旧方法。每搬一段判定立即运行对应窄测试和 golden parity。

### 阶段 4：提取 candidate generation 与 selection

按独立策略逐个搬移：

1. base tracking/branch candidate；
2. posture/wrist-unload；
3. singularity；
4. joint limit/boundary hold；
5. deadline 与最终 selection。

不要同时迁移 singularity 与 joint-limit state；这两个分支目前互相影响 selection budget 和
fallback，必须分两次提交。

### 阶段 5：提取 continuity、execution 和 Hold state

顺序建议：

1. 无状态 wrist canonicalization helper；
2. q5/alpha continuity state；
3. execution limits/admitted velocity；
4. inactive-side state；
5. Hold/recovery。

每一步通过 facade property 暂时暴露旧字段，以兼容现有测试；待生产调用者和测试迁移完成
后再删除这些私有兼容属性。

### 阶段 6：缩短 facade 和整理配置

最后才：

- 将 `_solve_bimanual_core()` 缩成清晰 pipeline；
- 将 config 验证委托给分组 validator；
- 拆分巨型安全测试文件为按领域命名的测试文件；
- 保留一个公共 facade/API/回放集成测试。

## 7. 测试组织

建议新增/拆分为：

```text
tests/test_dual_arm_solver_contracts.py
tests/test_dual_arm_solver_geometry.py
tests/test_dual_arm_candidate_evaluation.py
tests/test_dual_arm_candidate_selection.py
tests/test_dual_arm_continuity_policy.py
tests/test_dual_arm_singularity_policy.py
tests/test_dual_arm_joint_limit_policy.py
tests/test_dual_arm_execution_admission.py
tests/test_dual_arm_hold_recovery.py
tests/test_dual_arm_solver_integration_parity.py
```

现有 `test_dual_arm_collision_aware_safety.py` 在迁移期继续运行，不能先删除或批量改写。新
组件测试应优先使用公开 receipt/state，而不是继续 monkeypatch facade 的 `self.*`。

每个阶段最小验证：

```bash
python -m pytest tests/test_dual_arm_collision_aware_safety.py -q
python -m pytest tests/test_mujoco_umi_replay.py -q
python -m pytest tests/test_real_teleop_control_profiles.py -q
python -m pytest tests/test_rm65_position_profile.py -q
python -m pytest tests/test_dual_arm_collision_aware_safety.py \
  tests/test_mujoco_umi_replay.py \
  tests/test_real_teleop_control_profiles.py \
  tests/test_rm65_position_profile.py -q
git diff --check
```

再对代表性历史数据运行离线 replay/parity，不连接真机。

## 8. 验收标准

### 行为

- public import、aliases、class `__module__` 不变；
- deterministic fake-solver 测试中 qpos 与 candidate 顺序逐帧一致；
- 除 timing 字段外 metadata 深比较一致；
- state latch、Hold/recovery、deadline fallback 时序一致；
- 历史 replay 的 status/fallback/candidate counts 不变；
- 无新增 collision query 或 IK iteration；
- p95 solve latency 不劣化超过 5%，且不增加 allocation 热点。

### 结构

- facade 只保留 contracts、公共 API、依赖组装、cycle orchestration 和最终 commit；
- safety/evaluation/selection 各有明确 owner；
- 内部模块不 import `teleop_runtime`、`control_station`、sim 或硬件；
- 内部模块不反向 import facade；
- 不新增 production entrypoint；
- 不通过 mixin、动态 method injection 或 `self` 巨型代理对象隐藏依赖；
- 新组件使用窄 typed context/receipt，而不是无结构 metadata dict 作为控制输入。

行数不是唯一目标，但建议最终：

- facade 控制在约 1,000–1,500 行；
- 单个领域模块尽量不超过约 1,500 行；
- 新增单方法原则上控制在 250 行以内；
- `_solve_bimanual_core()` 最终控制在约 150–250 行的可读 pipeline。

## 9. 不推荐的整理方式

1. **按 V1/V2/V3 复制 solver 文件**：会产生多个 safety/IK owner，后续行为继续漂移。
2. **使用多个 mixin 拼接同一个大类**：文件变短但状态依赖仍隐藏，测试和 authority 更难审计。
3. **把实现移到 `teleop_runtime`**：违反共享 motion owner 边界，sim/replay/deploy 会重新分叉。
4. **一次性移动 9,000 行再修测试**：无法区分 import、时序、状态和数值回归。
5. **重构时顺便调整阈值或修 solver 行为**：组织变化与控制变化无法独立回退。
6. **让策略组件直接更新 accepted qref**：破坏唯一 commit owner 和候选认证链。
7. **以任意 dict 传递所有状态**：只是把巨型 `self` 换成巨型字典，未形成清晰合同。

## 10. 推荐的第一批实际改动

在当前 boundary-hold 并行修改稳定后，第一批只做低风险、可独立回退的三件事：

1. 增加 deterministic cycle snapshot/golden parity helper；
2. 提取纯 `geometry.py` 和内部 candidate/receipt 类型；
3. 提取左右重复的 metadata/result builder。

第一批不移动 `_select_safe_tracking_candidate()`、任何 latch 或 execution smoothing。完成并通过
全量离线 parity 后，再进入 candidate evaluation 的拆分。

## 11. 2026-08-04 优化前基线确认

本次确认发生在大量并行修改仍存在的工作树上，因此结论仅针对下列精确文件快照，不能把
“测试通过”外推为整个工作树已经适合立即大规模搬移。

### 11.1 当前 solver 相关变化

当前 `dual_arm_target_solver.py` 相对 Git 基线为 `+186/-5`。新增行为集中在 V2
`wrist_q5_entry_boundary_hold`：

- 只在 `single_step_singularity_aware`、wrist continuity 开启且
  `wrist_q5_hysteresis_requires_escape_latch=True` 时启用；
- 当已接受 q5 尚在保护边界外、普通候选第一次向内进入或近边界跨零时，只把该侧 q5
  投影回已接受参考，保留同一候选的其他关节运动；
- 向外运动不拦截，已经位于边界内的状态不由该 entry guard 锁死；
- 该定向边界已处理时跳过 auxiliary singularity candidate expansion；
- 投影候选无法通过最终认证时返回 reference Hold；
- V1/未启用 latch-required 的配置保持旧路径，可作为原子行为回退。

这是一项控制语义变化，不应在结构迁移时顺便重写、泛化或调整阈值。尤其
`_select_safe_tracking_candidate()` 中的 early return、候选池清空、最终认证和 metadata
必须被当作当前基线的一部分冻结。

### 11.2 精确文件指纹

确认时 SHA-256：

```text
b0c300c71fe43aa3eecd6e92d92b0cc91c4a719879ee2be45666dcd47e7f7ac4  robot_motion/rm65/dual_arm_target_solver.py
24177f8acdfe556e071178d9494059533dc4d06e6c6e86d95d42c6ce1586c034  robot_motion/rm65/__init__.py
f646a6e1a4d3ab186e2ea9a6b5282ab2367a8a799a058ceefc030d1805e052e7  tests/test_dual_arm_collision_aware_safety.py
450155d8f83e10654793a27959611ab39b6b7e6cbe8c0f0f3529130a4fd12ea9  teleop_runtime/retargeters/dual_arm_collision_aware.py
```

如果任一指纹变化，开始结构修改前必须重新查看相关 diff，并至少重跑 11.3 的回归；不得
沿用本节的“已确认”结论。

### 11.3 已通过的只读/离线验证

使用现有 `dexteleop310` 环境，不连接设备：

```text
python -m py_compile：通过
新增 q5 boundary-hold 定向测试：2 passed
tests/test_dual_arm_collision_aware_safety.py：109 passed
tests/test_rm65_position_profile.py + tests/test_real_teleop_control_profiles.py：49 passed
replay shared-retargeter 类型身份测试：1 passed
```

默认 base Python 3.12 因缺少 `osqp` 无法收集测试；这是环境选择问题，不是 solver 测试
失败。后续验证固定使用项目现有 `dexteleop310` 环境，不安装或临时替换依赖。

### 11.4 允许开始结构迁移的门禁

满足以下全部条件后，才开始第一批生产代码提取：

1. 当前并行修改者不再同时编辑 `dual_arm_target_solver.py` 及其直接测试，或已明确给出一个
   可识别的冻结点；
2. 重新记录 solver、兼容 adapter 和关键测试的文件指纹；
3. 11.3 的测试仍通过，且 `git diff --check` 通过；
4. 先增加 cycle characterization/parity fixture，能比较候选名称与顺序、认证结果、最终
   qpos、状态迁移和除 timing 外的 metadata；
5. 第一批只提取纯 geometry、内部值对象或纯 metadata builder，不移动选择、q5/奇异/限位
   latch、deadline、execution smoothing、Hold/recovery；
6. 每次只移动一个 owner 明确的域，由 facade 保留同名兼容委托；每一步均可单独撤销，而不
   依赖后续步骤才能恢复；
7. 结构 patch 中禁止同时修改阈值、候选顺序、fallback、日志语义或性能参数。

若并行更新无法暂停，安全选择是继续补 characterization 测试和依赖图，不修改生产 solver
的文件布局。

## 12. 第一批安全迁移记录

日期：2026-08-04
状态：已完成第一批纯布局提取；尚未进入 candidate/evaluation/selection/state 拆分

### 12.1 迁移前冻结点

用户确认 solver 并行修改已暂停后，重新检查 11.2 的四个指纹，均与优化前基线一致。
因此本批变更以
`dual_arm_target_solver.py=b0c300c71fe43aa3eecd6e92d92b0cc91c4a719879ee2be45666dcd47e7f7ac4`
为明确回退点。

### 12.2 本批实际范围

新增私有内部包：

```text
robot_motion/rm65/dual_arm_solver/__init__.py
robot_motion/rm65/dual_arm_solver/geometry.py
```

只提取以下无状态布局操作：

- 将左右 6-DoF qpos 嵌入 12/14/24+ solver vector；
- 从 solver vector 拆出左右臂 qpos；
- 解析左右臂 full-vector joint indices；
- 构造 `(side, local_index, full_index)` entries。

`DualArmCollisionAwareRetargeter` 继续保留原 `_embed_arm_qpos()`、`_split_solution()`、
`_arm_joint_indices()`、`_arm_joint_entries()` 方法和调用点，仅改为同步纯函数委托。
没有移动或改变：

- IK/collision 调用；
- candidate 生成、顺序、认证或选择；
- q5、奇异、限位 latch；
- deadline、fallback、Hold/recovery；
- execution cap、smoothing 或 accepted-reference commit；
- public import、class identity 或 production entrypoint。

内部包只依赖 NumPy，没有导入 `teleop_runtime`、`control_station`、硬件或 sim。

### 12.3 新增 characterization

新增 `tests/test_dual_arm_solver_geometry.py`，覆盖：

- 12-DoF、14-DoF、24-DoF 和大于 24 的 `dual_arm_hand` 布局；
- embed/split round trip；
- 输入和拆分输出不共享可写内存；
- indices/entries 精确顺序；
- 不支持向量长度的 fail-closed 行为。

原有完整 solver 安全测试继续作为候选、metadata、latch 和最终输出 characterization，未被
替换或放宽。

### 12.4 验证结果

使用 `dexteleop310`，不连接设备：

```text
定向布局/q5/public-interface：12 passed
完整 solver safety + geometry：118 passed
position/profile contracts：49 passed
replay shared-retargeter identity：1 passed
py_compile：passed
git diff --check：passed
```

完整 `tests/test_mujoco_umi_replay.py` 另有一个并行 sim/viewer 失败：测试期望 viewer
`is_running()` 调用一次，当前并行实现调用两次。该失败与 solver/package import 无关，相关
shared-retargeter identity 用例单独通过；本批未修改 sim 或该测试。

迁移后文件指纹：

```text
db6b6aab7fe72b4afb0149bbcaa62bc1b4db61e5090612816f1224307073b05f  robot_motion/rm65/dual_arm_target_solver.py
99b78183ca340f266459bd18413ac73278fb7df8d89751edf96178dd79c0731b  robot_motion/rm65/dual_arm_solver/__init__.py
d94c70e87a55c0fc708261573152fe0f22dfb6cb32bd184ffda0704a366170b7  robot_motion/rm65/dual_arm_solver/geometry.py
c90745c1597025763431a692c3c67a7158a16b985f4b93e17da7042af2d3dbdb  tests/test_dual_arm_solver_geometry.py
```

### 12.5 下一批门禁

下一批优先提取内部 candidate value objects，但 facade 必须继续 re-export 私有兼容名称，
因为当前 teleop compatibility adapter 和测试仍直接导入 `_DualArmCandidate` /
`_DualArmEvaluatedCandidate`。迁移前应先增加类型兼容测试；若任何调用者依赖其
`__module__`，则暂缓移动类型定义，改为先提取纯 metadata builder。
