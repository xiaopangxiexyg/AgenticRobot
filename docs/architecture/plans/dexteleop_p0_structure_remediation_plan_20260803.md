# DexTeleop P0 项目结构治理当前门禁

状态：活动边界索引；最后复核日期为 2026-08-17。

本文只维护尚未闭合的结构治理顺序、启动条件和所有权不变量。2026-08-04～08-07 的完整
工作树快照、已完成批次、测试数字和当时命令已冻结在
[P0 历史实施台账](../reports/p0_structure_remediation_20260803.md)。早期问题规模和只读审计见
[2026-08-03 项目结构审计](../../legacy/reports/project_structure_audit_20260803.md)。

开始任何新批次前必须重新检查当前代码、测试、linked worktree 和路径 owner；本文件的日期和
历史结果不能替代当前证据，也不授权设备连接、真实运动或数据变换。

## 1. 已闭合边界

| 批次 | 当前结论 |
|---|---|
| P0-G 并行基线与边界恢复 | progress catalog 基线和 tactile recompression 依赖方向已经收口；`collection_runtime` 拥有机制，`utils` 保留离线/兼容入口 |
| P0-DOC 文档归属 | 数据、real reports、showcase 和领域索引已经归位；文档导航现由 `scripts/check_docs_navigation.py` 检查 |
| P0-4 ControlTrace 校验器拆分 | rule families 已归入 `dataset_contracts/control_trace_validation/`；`completeness.py` 保留公共 issue、context/index owner、调度和结果汇总 |
| P0-1B RealTeleop core 依赖反转 | runtime/operator core 只依赖 lower-owned `ProgressApplicationPort`；Station adapter 由 canonical application edge 构造并注入 |
| 历史 manifest/reference 分型 | 历史 hash-only evidence 与当前 builder 分开解析；不得用当前 payload/hash 回写历史收据 |

这些结论只说明对应结构切片已经落地，不表示真机 authority、安全、dispatch、episode lifecycle
或完整端到端部署已经迁移。

## 2. 仍需处理的结构治理

### 2.1 P0-1C：UMI application facade（条件性清理）

当前 `teleop_runtime/collection/umi/app.py` 是显式兼容 application facade，转发到
`control_station.umi_application`；RealTeleop 和 calibration 的两个 debug 模块同样是 canonical
application entrypoint。边界测试把这三个路径列为精确例外，未发现 UMI controller 的第二处
lower→Station 导入。

因此 P0-1C 不是立即删除兼容路径，而是先确认：

1. `python -m teleop_runtime.collection umi` 与脚本/文档消费者是否仍依赖旧路径；
2. headless parser、GUI resource lifecycle 和退出码是否由同一个 application owner 保持；
3. 能否把 canonical construction 完全留在 `control_station`，同时保留无状态兼容转发；
4. 删除条件是否已经满足。

在消费者和兼容期限未确认前，保留该 facade；不得为了清空 allowlist 复制 GUI policy、浏览器状态
或 UMI lifecycle 到 lower package。

### 2.2 P0-3：按资源生命周期缩短 RealTeleop orchestration

截至最后复核，热点仍集中在：

| 文件 | 规模信号 |
|---|---:|
| `teleop_runtime/real_teleop/runtime.py` | 2476 行 |
| `teleop_runtime/real_teleop/control_loop.py` | 4350 行 |
| `teleop_runtime/real_teleop/operator_control_runtime.py` | 5369 行 |
| `teleop_runtime/real_teleop/component_builder.py` | 2505 行 |

行数只是审计信号，不是拆分目标。推荐顺序仍是：

1. startup/preflight/cleanup registry；
2. component construction 与成对 close contract；
3. operator intent admission；
4. collection/progress episode effects；
5. control-step orchestration；
6. canonical safety/HOLD dispatch；
7. shutdown、flush 和 failure aggregation。

每个切片必须保持一个 runtime owner，不能创建第二套 authority、safety、reference、dispatch、
writer commit 或 episode owner。先提取无状态 projection、窄 adapter 和成对资源生命周期，最后
才缩短主循环。

启动门禁：

- 目标路径和公共合同已有明确 active owner，且没有语义重叠的并行写入；
- canonical GUI、headless collection 和现有兼容入口行为已字符化；
- startup failure、partial construction、HOLD、collection finalize 和 shutdown 顺序有离线测试；
- 一次只迁移一个资源生命周期，不同时调整控制参数或默认 profile。

### 2.3 P0-2：RM65 solver 模块化

`robot_motion/rm65/dual_arm_target_solver.py` 最后复核为 11402 行，且直接影响真实双臂 IK、碰撞、
continuity、latch 和 accepted qref。详细设计只由
[RM65 双臂 solver 模块化计划](../../real/plans/rm65_dual_arm_solver_modularization_plan.md)维护，
本文件不重复定义第二套拆分方案。

启动门禁：

- solver、position profile、trajectory 和 wrist boundary 路径已释放给同一结构批次；
- deterministic characterization 冻结 candidate 名称、顺序、scale、qpos 和 metadata；
- Hold/reset/timeout/collision query 及 accepted-state commit 顺序保持不变；
- 第一批只提取纯 geometry、diagnostics 或 value objects；
- facade 继续是唯一 accepted qref 和 policy-state commit owner；
- 验证只使用离线/fake/sim 证据，结构整理本身不授权真机运动。

## 3. 不再继续的旧批次

- P0-G、P0-4 和 P0-1B 已闭合；后续若出现新回归，应按新的具体 owner 建立窄任务，不能重开
  旧批次并把历史测试数字当作当前基线。
- `dataset_contracts/completeness.py` 后续增长不自动授权继续拆分；只有出现新的职责聚集且有稳定
  issue-tuple characterization 时，才建立新的 validator 计划。
- 历史 RM65 evidence 不重建、不补写缺失 payload，也不与当前 builder 强行比较。
- 巨型文件、目录平铺或 allowlist 数量本身不是迁移依据；所有权、状态机和资源生命周期才是。

## 4. 每批执行规则

1. 从最新 main 创建独立 worktree，登记 owner、base、write scope、公共语义和验证范围。
2. 先确认当前调用者、兼容路径、资源 owner 和并行 worktree，没有证据时不假定旧计划仍准确。
3. 一次只迁移一个闭合职责；功能变化、所有权迁移和行为调整分开提交。
4. 保留必要的薄兼容导出，并用对象身份、AST/数值 parity 或调用顺序测试证明行为保持。
5. 先运行最小字符化测试，再运行相关边界和入口检查；不把 shadow/sim 证据写成真实硬件验收。
6. 每个阶段通过 `git diff --check`，提交后保持 clean，并记录精确 tip 与验证结果。

## 5. 最小验证选择

所有命令从目标 worktree 根目录、在已激活的 `${DEXTELEOP_ENV_NAME:-dexteleop310}` 环境中运行。
具体测试应按实际切片选择，不机械复制历史台账中的完整矩阵。

```bash
python scripts/check_docs_navigation.py
python -m pytest -q tests/test_control_station_boundaries.py
python -m pytest -q tests/test_collection_runtime_boundaries.py
python -m pytest -q tests/test_real_teleop_refactor_contracts.py
python -m teleop_runtime.debug.run_vive_rm65_real_teleop --help
git diff --check
```

P0-2 使用 solver 独立计划的 deterministic/offline 矩阵。任何测试缺失、环境失败或 ignored asset
问题都必须单独报告，不能通过启动真实设备来替代。

## 6. 当前优先顺序

1. 把 UMI facade 作为兼容性审计，而不是默认删除任务；
2. 只有在 RealTeleop 目标路径稳定且字符化充分时，才按单一资源生命周期推进 P0-3；
3. RM65 solver 始终使用独立计划，并在安全关键并行路径稳定后最后处理；
4. 每次开始前重新读取当前工作树；若事实变化，更新本索引而不是续写历史报告。
