# SAPIEN collection replay 退役记录

`teleop_runtime.collection.replay_dataset_sapien` 曾用于读取 LeRobot 风格采集 episode、
显示历史 SAPIEN 双臂/手部回放并打印触觉汇总。2026-08-25 审计确认它没有 production、
import、test 或发布调用者，也不属于当前受支持的采集和仿真路线，因此源码已经从版本化树
删除，不再提供对应 Python module 或 CLI。

当前替代路径：

- 数据集只读验收：`python -m utils.dataset_postprocess validate-v23 DATASET_ROOT`；
- 新仿真回放：使用 `docs/sim/` 记录的 MuJoCo 或 backend-neutral 工具；
- 旧命令和参数追溯：使用 `docs/legacy/root/README_site_pipeline_2026-07-22.md`。

仓库根 `legacy/` 只允许保存开发者自行维护、被 Git 忽略的本地副本。它不是 Python package、
共享证据或文档链接目标，不能通过 force-add、`.gitignore` 例外或 mixed tracked policy 进入
版本历史。需要共享的退役事实只记录在本文件等 `docs/legacy/` 文档中。
