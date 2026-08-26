# 历史命令与调试资料

本目录是版本化历史说明的唯一 legacy 位置，保存已经被正式 profile、canonical entrypoint
或当前领域文档替代的命令、退役原因和审计快照。“Legacy”表示不再是当前默认入口，不表示
这些文档可以直接删除。仓库根 `legacy/` 与本目录语义不同：它严格被 Git 忽略，只能保存
开发者本地的已移除代码副本，不能包含任何 tracked path。

| 历史资料 | 内容 | 当前替代 |
|---|---|---|
| [2026-07 位控/力控 A/B](./position_control/test-7.11.md) | 15/40、15/60/120、IK、8×、inactive、singularity、数采性能演进 | [真实系统入口](../real/README.md) / [遥操数采说明](../real/TELEOP_COLLECTION.md) |
| [DMTAC 旧命令分层](./dmtac/temp_dmtac.md) | 旧 75→150 Hz、debug overlay、旧 replay/deploy 组合 | [真实系统](../real/README.md)、[硬件](../hardware/README.md) |
| [Position A/B report](./reports/position_ab_report.md) | 已有 capture 的离线分析结果 | [遥操数采说明](../real/TELEOP_COLLECTION.md) |
| [2026-07-22 命令清点](./reports/command_inventory_2026-07-22.md) | 重组前入口与散落来源快照 | [当前 docs 入口](../README.md) |
| [2026-07-17 框架复核](./reports/framework_review_20260717.md) | 当时目录、入口、契约和迁移状态的只读快照 | [当前 docs 入口](../README.md) / 各 package README |
| [2026-08-03 项目结构审计](./reports/project_structure_audit_20260803.md) | P0 实施前的规模、边界与问题快照 | [当前 P0 结构治理计划](../architecture/plans/dexteleop_p0_structure_remediation_plan_20260803.md) / [`AGENTS.md`](../../AGENTS.md) |
| [旧 old_handeye 标定树退役](./reports/old_handeye_retirement_20260825.md) | 无引用的旧手眼标定脚本、样例和调试入口退役记录 | [腕部手眼标定](../../utils/hand_eye/README.md) / [工位手眼标定](../../utils/head_eye/README.md) |
| [DexHand report helper 退役](./reports/dexhand_report_helper_retirement_20260825.md) | 无调用的私有 DexHand 报告生成 helper 退役记录 | 当前 DexHand 硬件测试入口与结果收集流程 |
| [2026-08-06 tests 整理记录](./reports/tests_reorganization_20260806.md) | marker、路径试点、兼容恢复和冻结区记录 | [当前 tests 规则](../../tests/README.md) |
| [2026-07-21 RTC 研究备忘](./reports/rtc_research_memo_20260721.md) | 异步 action chunk、时间对齐和缓冲方案的早期调研 | [当前 RTC 计划](../architecture/plans/dexteleop_rtc_landing_plan.md) / [`DEPLOY_SCHEMA.md`](../../DEPLOY_SCHEMA.md) |
| [2026-08-02 HITL 研究备忘](./reports/hitl_research_memo_20260802.md) | latched nominal chunk、residual、rebase 和 handback 的早期方案 | [当前 HITL 计划](../architecture/plans/hitl_chunk_relative_migration_plan.md) |
| [SAPIEN collection replay 退役记录](./reports/sapien_collection_replay_retirement_20260825.md) | 已删除的历史 SAPIEN 数据回放入口及替代路径 | [遥操数采说明](../real/TELEOP_COLLECTION.md) / [仿真说明](../sim/README.md) |
| [旧根 README 站点流程](./root/README_site_pipeline_2026-07-22.md) | 重组前 589 行设备、SAPIEN、真机短测、GUI 数采和回放流程 | [当前 docs 入口](../README.md) |
| [v2.3 旧站点验收](./acceptance/ACCEPTANCE_V23_site_2026-07-22.md) | 固定主机/checkpoint/测试数量及含 auto-start 的历史 A/B 表 | [当前 v2.3 验收](../data/ACCEPTANCE_V23.md) |

## 使用规则

1. 只在复现实验或追溯参数来源时使用这里的命令。
2. 真机运行前必须重新对照当前 `--help`、版本化 profile 和对应领域文档。
3. 历史文档中的 `Canonical`、`推荐` 等措辞只代表当时上下文。
4. 不直接删除历史文档；确认无引用、已有替代且迁移记录完整后，才考虑进一步归档。

## 保持在活跃路径的特殊文件

- [`docs/real/teleop_args.sh`](../real/teleop_args.sh)：当前真机遥操 Bash 参数数组；本目录内的历史
  命令也从该路径加载，避免维护两份参数。
- `README.md`、`README_ENV.md`：仍承担项目入口和环境权威，只增加 docs 导航。
- `docs/data/ACCEPTANCE_V23.md`：当前可移植验收模板；旧站点版本已归档到本目录。
- 本地 `todos/`：不进入 Git；需要团队共享的计划、历史审计、实验报告和稳定说明按实际 owner 进入 `docs/`，
  但是否完成不能只按文件名判断。
- 各 package `README.md`：仍是模块参数和技术边界的权威来源。
