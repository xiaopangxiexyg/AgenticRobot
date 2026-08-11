# DexTeleop 活动工作计划

本目录只承载仍需实施或验证的工作计划，不是稳定架构、运行手册、实验报告、
证据归档或可执行工具的长期存放位置。一个文件出现在 `todos/` 中，只表示该主题
尚有明确的未完成边界，不表示其中设想已经成为当前代码事实。

## 准入规则

- 只有当前实现确实需要跨文件、跨模块或分阶段收敛时，才新增计划。
- 同一主题只保留一个当前有效计划；研究备忘和旧方案必须由当前计划明确吸收或替代。
- 计划应说明实际 owner、已经改变的责任或调用边、下一处未验证边界，以及关闭条件。
- 不要求固定证据层级。验证深度由风险和用户目标决定，并明确最远实际到达的边界。
- 新建 plan、adapter、fixture、wrapper、export 或模块本身不算进展；进展以实现、
  依赖、调用或验证边界的真实变化描述。
- 不在计划正文中持续累积逐轮命令、测试数量和聊天式实施日志。需要保留的日期化结果
  应进入所属领域的 `docs/**/reports/`。

## 退出规则

- 稳定 contract、ownership 和公共接口说明吸收到对应 package README 或 schema。
- 当前运行方式和操作安全说明进入 `docs/` 下的所属领域。
- 跨模块架构图示进入 [`docs/architecture/`](../docs/architecture/README.md)；它们只映射代码和计划，
  不反向充当实现完成或验收通过的证据。
- 日期化实验、事故、审计和验证结果进入所属领域的 reports；被替代材料只在确认
  目标路径可被版本控制后进入 `docs/legacy/`，并记录当前替代来源。
- 可复用断言进入 `tests/`；离线分析、维护和硬件诊断脚本归入实际 owner 或
  `scripts/`，不得因仍有后续工作而长期放在 `todos/probes/`。
- 当下一边界完成、取消或被其他计划吸收后，计划必须退出本目录，而不是继续追加
  “已完成”流水账。

## 当前整理边界

下列计划属于已经确认的活动结构主线，整理期间保持原路径，不做机械拆分：

- [`dexteleop_p0_structure_remediation_plan_20260803.md`](dexteleop_p0_structure_remediation_plan_20260803.md)
- [`rm65_dual_arm_solver_modularization_plan.md`](rm65_dual_arm_solver_modularization_plan.md)
- [`hitl_chunk_relative_migration_plan.md`](hitl_chunk_relative_migration_plan.md)
- [`control_station_sim_hil_decomposition_plan.md`](control_station_sim_hil_decomposition_plan.md)

这不是全部活动计划清单。当前工作树仍有并行修改和未跟踪文件；未列出的文件必须在
对应 owner、当前状态和引用关系复核后逐项归类，不能据此批量归档或删除。
