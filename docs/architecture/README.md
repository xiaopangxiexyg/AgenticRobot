# DexTeleop 架构图示

这里集中维护跨模块架构、RTC 可视化说明和需要团队共享的实施计划：

- [控制、数据与训练闭环图示](./dexteleop_architecture_visual_guide.md)
- [实时 Action Chunk（RTC）图示](./dexteleop_rtc_visual_guide.md)
- [数采、一般部署、混合控制与仿真的运行域边界](./runtime_domain_boundaries.md)
- [`lab` M1：项目无关的多仓 Task Workspace](./lab_m1_task_workspace.md)

## 文档边界

- 图示只归纳当前代码、测试和实施计划，不新增 contract、authority、验收结论或真机授权。
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
