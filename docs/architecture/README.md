# DexTeleop 架构图示

这里集中维护跨模块架构与 RTC 的可视化说明，避免将只读导航材料与 `todos/` 中的实施计划混在一起：

- [控制、数据与训练闭环图示](./dexteleop_architecture_visual_guide.md)
- [实时 Action Chunk（RTC）图示](./dexteleop_rtc_visual_guide.md)

## 文档边界

- 图示只归纳当前代码、测试和实施计划，不新增 contract、authority、验收结论或真机授权。
- 当前行为以代码和测试为准，包边界以 [`AGENTS.md`](../../AGENTS.md) 为准；`todos/` 文档只描述尚未全部落地的方向和批次。
- 实施计划仍由其原有 owner 维护：
  [主架构](../../todos/dexteleop_architecture_planning.md)、
  [control-core landing](../../todos/dexteleop_control_core_landing_plan.md)、
  [RTC landing](../../todos/dexteleop_rtc_landing_plan.md) 和
  [Control Station](../../todos/dexteleop_control_station_plan.md)。
- `todos/evidence/` 中出现的旧文档路径和哈希是历史收据的一部分；迁移图示时不回写这些记录。

图示与当前代码或上述边界冲突时，应修正图示，不能用图示反向证明实现已经完成。
