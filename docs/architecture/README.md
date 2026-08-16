# DexTeleop 架构图示

这里集中维护跨模块架构、RTC 可视化说明和需要团队共享的实施计划：

- [控制、数据与训练闭环图示](./dexteleop_architecture_visual_guide.md)
- [实时 Action Chunk（RTC）图示](./dexteleop_rtc_visual_guide.md)
- [数采、一般部署、混合控制与仿真的运行域边界](./runtime_domain_boundaries.md)
- [联邦多 Agent 开发流规划](./federated_polyrepo_development_workflow.md)
- [`lab` Task Workspace、本地 Run、draft 与远端合同（M1-M4 bootstrap）](./lab_task_workspace.md)
- [OpenPI `model/openpi` submodule 迁移说明](./openpi_submodule_migration.md)
- [M0 远端只读审计收据（2026-08-14）](./m0_remote_audit_20260814.md)

## 文档边界

- 图示和规划只归纳当前代码、测试和实施方向，不新增 runtime contract、authority、验收结论或真机
  授权；`lab_task_workspace.md` 只描述其明确标注为已实现的开发工具能力。
- 当前行为以代码和测试为准，包边界以 [`AGENTS.md`](../../AGENTS.md) 为准；[`plans/`](plans/README.md)
  只描述尚未全部落地、但需要团队共享的方向和批次。
- 实施计划仍由其原有 owner 维护：
  [主架构](plans/dexteleop_architecture_planning.md)、
  [control-core landing](plans/dexteleop_control_core_landing_plan.md)、
  [RTC landing](plans/dexteleop_rtc_landing_plan.md) 和
  [Control Station](plans/dexteleop_control_station_plan.md)。
- [`evidence/`](evidence/) 中 evidence JSON 内出现的旧 `todos/evidence/` 路径和哈希是历史收据的
  一部分；迁移版本控制位置时不回写这些记录。

图示与当前代码或上述边界冲突时，应修正图示，不能用图示反向证明实现已经完成。
