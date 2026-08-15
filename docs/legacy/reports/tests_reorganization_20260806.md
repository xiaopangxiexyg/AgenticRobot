# DexTeleop `tests/` 二次整理历史记录（2026-08-06）

审计日期：2026-08-06
历史状态：兼容路径恢复与 README 规则调整已完成；marker-only 修正保留；本文不再作为活动计划
工作区：`/home/zc/HandMotion/Developing/DexTeleop`
审计基准：branch `new_realman`，HEAD `3f94c77b6ea4780533132cb647840f5f1ff5004e`

当前测试放置、marker、兼容路径和硬件 gate 规则以
[`tests/README.md`](../../../tests/README.md) 为准。本文仅保留当时的审计、试点和回退记录；
其中的路径数量、测试数量和 dirty-worktree 快照不是当前事实。

## 1. 目标与原则

本次只整理测试树，不借机修改生产行为、测试断言、公共接口或依赖：

- 整理的首要目标是让维护者容易找到、调整和新增测试；目录层级本身不是目标。
- 并行开发期间，现有扁平测试路径是兼容接口；优先用清晰 basename 和现有 marker 表达 owner/risk。
- 只有在真实降低维护成本、调用方已核对且存在短期独占窗口时，才考虑按执行风险和生产 owner 迁移目录。
- HITL 及其共享 seam 全部冻结，不移动、不拆分、不改 marker、不改引用。
- 所有已修改或未跟踪测试均视为其他并行任务所有，本轮不碰。
- 文件移动、marker 治理、测试内容拆分必须分成不同批次；首批只做路径移动。
- 不添加 skip/xfail，不放宽断言，不改变默认 pytest 选择语义。
- 不运行真实设备、运动、相机、VIVE、Q6、RM65、HIL 或 hardware marker 测试。
- 每一批开始前重新核验 status、目标文件 hash、调用引用和活动计划；任一目标发生变化就停止该批次。

## 2. 当前项目与测试结构理解

生产 ownership 以仓库 `AGENTS.md` 为准：

- `control_composition` 只拥有可复用的组合、生命周期 reducer、因果 lineage 和非权威 difference/feedback；不拥有 authority、安全、dispatch 或 episode。
- `control_station` 是 GUI/application leaf；下层包不得反向依赖它。
- `robot_motion` 拥有共享 motion、Rig、Safety、Planner、position control 和 runtime contract。
- `collection_runtime` 拥有设备/UI 中立的采集生命周期与持久化机制；`teleop_runtime` 保留真机/GUI/headless frontend 和 source-bound adapter。
- `dataset_contracts` 是纯 schema/identity/hash/validation；`dataset_interface` 是 projection/reader/window/batch/validation。
- `sim` 只拥有仿真 backend/task/adapter，不得反向依赖 `teleop_runtime` 或真实硬件。

当前 `tests/README.md` 已定义目标结构：

```text
tests/
├── unit/<owner>/
├── integration/<owner-or-pipeline>/
├── performance/
├── support/
└── fixtures/
```

现有 `tests/conftest.py` 通过 basename 集合过渡性地分配 `unit`、`integration`、`optional`、`contract` marker；`pytest.ini` 从 `tests/` 递归发现 `test_*.py`。因此只移动文件且保持 basename 不变时，marker 规则仍然生效，但 node ID 的路径前缀会变化。

## 3. 2026-08-06 只读快照

- 根目录 `tests/test_*.py`：464 个模块。
- Python 测试代码：约 207,193 行。
- 默认收集：`5145 tests collected in 4.09s`，无 collection error；有一个 `hppfcl` deprecation warning。
- 当前并行状态：42 个已修改测试、24 个未跟踪测试，共 66 个直接冻结目标。
- 测试之间存在 8 处直接 helper import；被引用模块及其调用方不能分批搬迁。
- 多个测试使用 `Path(__file__).parents[1]`、相对 fixture 或仓库根路径；这些文件移动前必须先单独消除路径深度假设，不能在普通搬迁批次中顺手修改。
- `tests/` 还混有 4 个非 pytest 的现场/设备脚本；迁往 owner 的 `tools/field_validation/` 需要独立文档与安全审查，本轮首批不处理。

以上收集成功只证明 import/collection 基线，不证明全量行为正确、HITL 正确或硬件可用。

## 4. HITL 与并行修改冻结边界

冻结采用动态规则，不仅依赖文件名。每批执行前重新计算以下并集：

1. 所有 git modified/untracked test 文件。
2. 所有导入 `control_station` 或 `control_composition` 的测试。
3. 文件名或直接 import 涉及 `HITL/HIL`、`sim_hil`、human、residual、trajectory、RTC、ControlTrace 的测试。
4. `docs/HIL/README.md`、`docs/legacy/reports/hitl_research_memo_20260802.md`、
   `docs/architecture/plans/hitl_chunk_relative_migration_plan.md` 点名的测试。
5. 与上述测试互相 import helper、共享 fixture，或被活动计划/命令按精确路径引用的测试。

当前机械扫描得到 110 个 HITL/direct-or-shared 测试；与 66 个 dirty/untracked 测试合并后，冻结并集至少 154 个。剩余约 310 个文件只是“可继续审计”，不等于已经获准搬迁。

活动 HITL 文档明确点名的测试包括：

```text
tests/test_action_proposal_contracts.py
tests/test_component_action_trajectory.py
tests/test_composition_temporal_sampling.py
tests/test_control_composition_boundaries.py
tests/test_control_station_mujoco_pose_human_replay.py
tests/test_control_station_mujoco_sim_hil.py
tests/test_deploy_rtc_frontend_shadow.py
tests/test_execution_observation_contract.py
tests/test_reference_activation_contracts.py
tests/test_relative_action_schema.py
tests/test_relative_component_composition.py
tests/test_residual_anchor_contract.py
tests/test_trajectory_editing_reducer.py
tests/test_trajectory_residual_warp.py
tests/test_trajectory_session_lifecycle.py
```

此外，`tests/test_mujoco_async_topology.py` 属于已知的 async Execution Gate/base-plus-residual 共享路径，也显式冻结。上述清单不是完整白名单；动态规则优先。

## 5. 建议批准的首批试点

首批只移动两个干净、已跟踪、非 HITL、无跨测试 import、无 `__file__` 路径深度假设、无文档/脚本精确引用的单元测试：

| 旧路径 | 新路径 | 当前语义 |
|---|---|---|
| `tests/test_logical_dataset_manifest.py` | `tests/unit/dataset_interface/test_logical_dataset_manifest.py` | default `unit` |
| `tests/test_progress_exclusions.py` | `tests/unit/dataset_interface/test_progress_exclusions.py` | explicit `unit + contract` |

只读基线：

- 两文件合计：`8 passed in 0.12s`。
- `test_logical_dataset_manifest.py` SHA-256：`495a4e85d483cb60dde37762e124c67c0dfb82cb6eb3ee7e9719d64f28ce8740`。
- `test_progress_exclusions.py` SHA-256：`75cc73590c9e2b8c678b5469a84cbfd8b32fbe44c87146e08db7b3b31bc5a348`。

首批明确不做：

- 不改测试内容、basename、marker、fixture 或断言。
- 不改 `tests/conftest.py`、`pytest.ini`、`tests/README.md`。
- 不改任何生产文件或 HITL/RTC/ControlTrace/trajectory/Control Station 测试。
- 不处理现场脚本，不执行硬件测试。

## 6. 首批执行门槛与停止条件

获得二次确认后，执行前重新检查：

1. 两个源文件仍是 tracked + clean，hash 与上文一致。
2. 两个文件仍没有跨测试 import、路径深度假设或精确路径调用方。
3. HITL/并行冻结区没有被纳入本批 manifest。
4. 当前测试仍为 8 passed；若失败，不移动。

执行只创建 `tests/unit/dataset_interface/` 并完成上述两个 path move。出现以下任一情况立即停止，不尝试修复或吸收进搬迁 diff：

- 候选文件或其直接生产依赖在确认后发生并行变化；
- pre-move 测试、collection 或 import 失败；
- 发现新调用方依赖旧 node ID/path；
- move 后除已批准路径变化之外出现任何行为、marker、skip 或 test-count 差异；
- 需要修改 HITL/shared seam 才能继续。

## 7. 首批验收

移动后必须同时满足：

1. 两个文件内容 hash 与移动前完全一致。
2. 新路径精确测试仍为 8 passed。
3. 全量 `--collect-only` 与执行前实时基线保持一致；只允许这 8 个 node ID 的路径前缀按映射变化。
4. `unit/contract/integration/optional/hardware` 的选择结果不因本批变化。
5. `git diff --check` 通过。
6. scoped diff 只显示两个 rename 和新目录，不含生产代码、HITL 测试或其他测试内容修改。

这仍然不是全量测试、仿真、HITL 或硬件验证。

## 8. 首批执行结果

二次确认后重新检查时，并行任务已使全量收集从审计快照的 5145 增至 5153；候选文件、直接生产依赖、引用关系和候选测试没有变化，因此本批以移动前实时的 5153-node 集合作为比较基准。

执行结果：

- 两个文件已移动到 `tests/unit/dataset_interface/`，basename 与内容均未改变。
- 新路径聚焦测试：`8 passed in 0.19s`。
- 全量收集：移动前 5153、移动后 5153。
- 将旧路径映射到新路径后，排序 node 集合 SHA-256 均为 `9ee8f06faa3ec274ed3aff4655e27e3bec1f3441b6e981bf84aea4b0ed5b80e1`。
- marker：8 unit；其中 6 contract；integration/optional/hardware 均为 0。
- 两个测试文件与 HEAD 原内容逐字节相同，直接生产依赖 hash 在移动前后不变。
- `git diff --check` 通过。
- 未运行或修改 HITL、RTC、ControlTrace、trajectory、Control Station 或 hardware 测试。

已知且预期的变化：pytest 按路径遍历，因此两个子目录模块从原先各自在根目录 basename 顺序中的位置移动到了根目录模块之后；node 集合没有变化，但全局 collection order 发生了路径排序变化。本批没有运行全量行为测试，不能证明仓库中不存在依赖全局测试顺序的隐藏问题。

沙箱将 `.git/index` 挂载为只读，`git mv` 无法创建 `index.lock`；本批使用精确工作树移动并以内容 hash/`cmp` 验证。当前 Git 工作树会显示两个旧路径 deleted、两个新路径 untracked；后续在可写 Git index 中执行正常的 `git add -A` 时可由 Git 识别为 rename。本轮未尝试修改 index。

## 9. 后续批次（不在本次确认范围内）

首批验证稳定后，再逐批提出精确 manifest，并分别确认：

1. `dataset_contracts` / `dataset_interface` 其余 clean tests：先治理 marker，再在下一批移动。
2. `collection_runtime` 与 `teleop_runtime`：先解除 cross-test helper import 和路径深度假设，再按 owner/risk 分组。
3. `robot_motion`、mocked `hardware`、`deployments`：保持真实设备测试显式 gate，不把 mocked backend 误标为 hardware。
4. `sim`、Control Station、composition、RTC、ControlTrace、human/residual/trajectory：持续冻结，直到 HITL 并行任务明确释放 ownership window。
5. 现场脚本迁移与巨型测试拆分：作为独立项目，不与普通 path move 合并。

任何后续批次都不能把首批成功解释为对更大范围的自动授权。

## 10. 首批后的新增诊断

2026-08-06 再次只读复核时，工作区仍在快速并行变化：

- 当前测试模块已从首轮 464 增至 470，全量收集从 5153 增至 5218。
- `tests/` 当前状态包含 43 个 modified、2 个本批旧路径 deleted、31 个 untracked；不能把当前工作树当作稳定迁移窗口。
- 仍有 8 处测试模块直接 import 另一个测试模块的 helper。
- 50 个测试模块仍含 `Path(__file__).parents[...]` 或相近目录深度假设。
- 仓库 `tests/` 外共有 133 个不同测试路径被精确引用；其中 `todos/` 有 321 次引用，`docs/` 有 22 次引用。历史 `todos/` 不应批量重写，活动文档引用应随对应文件迁移逐项更新。
- 只有 47 个测试模块显式声明 marker；`tests/conftest.py` 的过渡集合当前包含 23 个 optional、30 个 integration、23 个 contract 模块。大量未登记测试会自动落入 default unit。
- 至少 17 个含 `pytest.importorskip(...)` 的模块当前没有 optional 分类；其中包含 HITL、dirty 和其他活动文件，不能一次性批量修正。

首批发现的具体语义缺口是：`test_logical_dataset_manifest.py` 为了不改变原选择语义而保留了 default unit，但它会创建目录和 JSONL 文件。按 `tests/README.md` 的“filesystem 属于 integration”规则，它最终应归入 integration，而不是长期留在 unit。

## 11. 已完成：只修 marker，不移动文件

下一步最高收益不是继续扩展目录，而是先修正一个小型、非 HITL、当前可验证的数据流水线集合。该批只修改 `tests/conftest.py`，不改测试内容或路径。

### 11.1 加入 `INTEGRATION_MODULES`

```text
test_logical_dataset_manifest.py
test_hardlink_dedupe.py
test_relocate_mask_metadata.py
```

原因：它们会在临时目录中读写 manifest、文件或真实 hardlink，属于 filesystem integration。

### 11.2 加入 `OPTIONAL_MODULES`

```text
test_dataset_media_integrity.py
test_v22_to_v23_desk_postprocess.py
test_workspace_near_boundary_postprocess.py
```

原因：它们分别依赖 OpenCV 或 PyArrow，并执行临时视频/Parquet/数据集 I/O。现有 hook 会让 optional 同时获得 integration marker，无需在两个集合中重复登记。

### 11.3 当前基线与预期变化

上述六个模块当前共 13 个测试，聚焦运行结果为 `13 passed in 0.71s`。marker-only 批次的预期变化应严格为：

- default/unfiltered 测试总数不变；
- `unit and not optional and not hardware` 减少 13 项；
- 7 项进入非 optional integration；
- 6 项进入 optional + integration；
- contract 和 hardware 选择不变；
- 测试内容 hash、node ID、收集顺序和生产文件均不变。

该变化会有意改变按 marker 选择的测试集合，因此必须作为独立批次确认，不能和路径移动合并。

### 11.4 停止条件

执行前如发生以下任一情况则停止：

- `tests/conftest.py` 或六个候选测试出现新的并行修改；
- 候选模块被新的 HITL/Control Station/composition/RTC/ControlTrace 计划引用；
- 13 项聚焦基线失败或收集数量再次在同一次 pre/post 窗口内漂移；
- marker 变化超出上文精确的 7 + 6 分组。

### 11.5 执行结果

获得确认后已仅修改 `tests/conftest.py`：

- `OPTIONAL_MODULES` 从 23 增至 26；新增 media-integrity、v22-to-v23 和 workspace migration 三个模块。
- `INTEGRATION_MODULES` 从 30 增至 33；新增 logical manifest、hardlink dedupe 和 relocate metadata 三个模块。
- 六个候选模块修改前后均为 `13 passed`。
- 全量收集修改前后均为 5218；完整 node ID 顺序与集合 SHA-256 均为 `d87e1b4c460b08242516d85e31ded0ab1d9f3f2922b46f1e5e370da50833670a`。
- marker 从 13 unit 精确变为 7 个非 optional integration 和 6 个 optional + integration；候选中 unit 为 0。
- 六个测试文件内容 hash 均未变化；未移动文件、未修改生产代码。
- `tests/conftest.py` 通过 `py_compile`，`git diff --check` 通过。
- 未运行或修改 HITL、RTC、ControlTrace、trajectory、Control Station 或 hardware 测试。

## 12. marker 稳定后的后续优先级

### N2：暂停数据流水线路径批次

不再把“把上述六个模块移动到 `tests/integration/dataset_pipeline/`”作为自然下一步。marker 已经提供执行风险分类，继续移动会改变旧路径和 collection order，却不会自动让测试更容易理解。

未来只有在一个完整 owner 已稳定、现有调用方可一次更新、且子目录能明显改善定位时，才重新提出精确迁移清单。默认保持旧路径。

### N3：解除跨测试 helper import

把现有 8 处 test-to-test import 按真实 owner 分组，先设计一个不会把整个 `tests/` 变成 Python package、不会改变 import mode 的 support contract，再逐组迁移。RM65 worker/dispatch helper 与 composition/HITL helper 必须分开；后者继续冻结。

### N4：消除目录深度假设

对 50 个依赖 `__file__` 目录层级的模块分类：

- 测试函数内部优先使用 pytest 提供的 repository root/context fixture；
- import-time AST/boundary 检查需要单独的稳定 path helper；
- fixture 相对路径应绑定 `tests/fixtures` 的显式 root，而不是 `parents[1]`。

该工作属于测试内容重构，不能和 move 合并。

### N5：稳定测试选择接口

活动 README/docs 中的 22 个精确测试路径应只在对应文件实际移动时更新；321 个历史 `todos/` 引用保留为历史证据。后续可评估 owner marker 或稳定测试组，但在 marker 语义完成审计前不新增全局选择接口。

### N6：巨型测试拆分

大文件拆分放在路径与 marker 稳定之后，且一次只处理一个非 HITL owner。拆分必须保留断言、parameter ID、fixture scope 和失败语义；不以减少行数为唯一目标。

### N7：现场脚本退出 pytest 树

`manual_camera_tactile_health_probe.py`、`rm65_force_pose_survey.py`、`rm65_force_standard_setup.py`、`rm65_safe_real_robot_test.py` 后续可迁到 owner 的 `tools/field_validation/`，但必须独立核对文档、调用方、hardware gate 和设备安全；本阶段不执行。

## 13. 目标校正：清晰、简单、兼容优先

用户进一步明确：整理是为了便于找到、调整和添加合理测试，不应人为引入困难；同时要尽量不影响并行部分并保持旧兼容。基于这一目标，后续采用以下简单规则。

### 13.1 当前兼容布局

- 并行开发期间，`tests/test_<owner-or-domain>_<feature>.py` 继续作为一般测试的默认位置和命名方式。
- `tests/fixtures/` 只放静态测试数据。
- `tests/support/` 只有在确实消除重复 helper 且 import contract 已验证时才使用；不为了目录外观提前创建抽象层。
- 不新增 owner marker、测试清单生成器、包装入口、路径别名、symlink 或第二套 pytest 配置。
- unit/integration/optional/contract/hardware 五类现有 marker 足够；新增测试应直接声明最准确的 marker，旧测试由 basename 集合渐进兼容。

### 13.2 新测试的放置原则

新增测试先回答三个问题：

1. 它验证哪个生产 owner 或跨 owner 流程？
2. 它是纯进程内行为、filesystem/worker/loopback integration、optional dependency、public contract，还是明确的 hardware test？
3. 是否已有同 feature 的测试文件可以直接追加，而无需新建近义文件？

若已有 owner/feature 文件，优先追加相邻测试；只有职责明显不同或文件已难以维护时才新建 `test_<owner>_<feature>.py`。不以“一个类/函数一个文件”为规则，也不把巨型文件拆分与目录迁移混在一起。

### 13.3 建议的下一兼容批次

首批路径试点证明内容与 node 集合可以保持一致，但也确认了旧路径失效和 collection order 变化。为了回到简单兼容布局，建议在再次确认后：

1. 将 `tests/unit/dataset_interface/test_logical_dataset_manifest.py` 恢复为 `tests/test_logical_dataset_manifest.py`。
2. 将 `tests/unit/dataset_interface/test_progress_exclusions.py` 恢复为 `tests/test_progress_exclusions.py`。
3. 保留已经验证的 marker-only 修正；`test_logical_dataset_manifest.py` 即使位于根目录，也继续正确归类为 integration。
4. 小幅修订 `tests/README.md`：把 target layout 明确为长期可选方向，把扁平 basename + marker 写成当前并行期的 canonical 添加规则。

该批不创建兼容 wrapper，因为 wrapper、symlink 或 collect-ignore 会产生重复收集、空测试或额外选择规则，反而增加维护难度。

### 13.4 兼容批次验收

- 两个文件内容 hash 不变。
- 相关 19 个不重复测试全部通过。
- marker 仍为：logical manifest 属于 integration，progress exclusions 属于 unit + contract。
- 全量 node 集合不变，仅两个 node ID 恢复旧路径；默认收集顺序恢复为根目录 basename 排序下的位置。
- 仓库内旧路径命令重新可用；不新增第二测试入口。
- scoped diff 只包含两处路径恢复、`tests/conftest.py` 已批准 marker 修正、README/计划说明；不触碰 HITL 或生产代码。

### 13.5 兼容批次执行结果

获得确认后已完成：

- 两个测试恢复到原路径 `tests/test_logical_dataset_manifest.py` 和 `tests/test_progress_exclusions.py`；试点 `tests/unit/` 目录及其 pytest 缓存已移除。
- 两个文件内容 hash 与迁移前/HEAD 完全相同，Git 不再显示这两个测试有路径或内容变化。
- 保留 `tests/conftest.py` 的 marker-only 修正：logical manifest 仍为 2 个 integration；progress exclusions 仍为 6 个 unit + contract。
- 相关 19 个不重复测试通过：`19 passed in 0.75s`。
- `tests/README.md` 已明确扁平 basename + marker 是并行期兼容规则，目录层级是有明确收益时才采用的可选长期方向。
- 验证窗口内，并行任务新增 `tests/test_composition_allocation.py` 的 6 个测试，使全量收集从 5224 增至 5230。剔除这 6 个明确的并行 node 后，旧路径映射后的 pre/post node 集合完全一致，SHA-256 均为 `571145054cbc018eb25239f889deacf85631256764967cca2d056637483dcd60`。
- `git diff --check` 通过；未修改生产代码，也未运行或修改 HITL、RTC、ControlTrace、trajectory、Control Station 或 hardware 测试。
