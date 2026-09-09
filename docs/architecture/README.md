# DexTeleop 架构图示

这里集中维护跨模块架构、RTC 可视化说明和需要团队共享的实施计划：

- [当前模块与运行域图示](./dexteleop_architecture_visual_guide.md)
- [实时 Action Chunk（RTC）图示](./dexteleop_rtc_visual_guide.md)
- [单源直通、多源组合及数采/部署/仿真的运行域边界](./runtime_domain_boundaries.md)
- [有界规则式数采计划与运行权限边界](./rule_based_collection.md)
- [Embodiment runtime provider 当前接口合同](./embodiment_runtime_provider_contract.md)
- [reBot B601 HITL 拓扑与厂商源码边界](./rebot_b601_hitl_topology.md)
- [ReBot B601-RS / RM65 Real2Sim 与 HITL 分阶段路线图](./plans/rebot_b601_rm65_real2sim_hitl_roadmap.md)
- [Quest 与机器人静态配对姿态采集边界](./quest_future_visualization.md)

其中运行域边界页区分 deploy、human teleop、replay 的默认单源直通模式与显式多源
`control_composition` profile。它只记录当前代码、import contract 和入口解析已经建立的边界；
被动 trace/shadow 不被算作多源控制。

## 文档边界

- 图示只归纳当前代码、测试和实施计划，不新增 contract、authority、验收结论或真机授权。
- 当前行为以代码和测试为准，包边界以 [`AGENTS.md`](../../AGENTS.md) 为准；[`plans/`](plans/README.md)
  只描述尚未全部落地、但需要团队共享的方向和批次。
- 实施计划仍由其原有 owner 维护：
  [主架构](plans/dexteleop_architecture_planning.md)、
  [control-core landing](plans/dexteleop_control_core_landing_plan.md)、
  [RTC landing](plans/dexteleop_rtc_landing_plan.md) 和
  [Control Station](plans/dexteleop_control_station_plan.md)。
- [`plans/`](plans/README.md) 是仍含未闭合工作或门禁的共享计划索引；
  [`reports/`](reports/README.md) 保存带日期的审计和已完成实施台账；
  [`evidence/`](evidence/README.md) 是按批次冻结的历史收据索引。计划、报告和 evidence
  的状态与验证范围不能相互替代。
- [`evidence/`](evidence/) 中 evidence JSON 内出现的旧 `todos/evidence/` 路径和哈希是历史收据的
  一部分；迁移版本控制位置时不回写这些记录。

图示与当前代码或上述边界冲突时，应修正图示，不能用图示反向证明实现已经完成。
