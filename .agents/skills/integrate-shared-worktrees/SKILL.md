---
name: integrate-shared-worktrees
description: Coordinate standalone Git worktrees, ownership, handoffs, authorized integration and closeout. Use outside Harness; managed Tasks use their Harness contract.
---

# 轻量 Git 协作

先确定管理方式。默认 Harness；独立 Git 开发使用本 Skill。已受 Harness
管理的 Task 继续使用其 scope、lease 和证据流程，不重复写分支描述。

创建、交接、集成或收尾时读取
[Git 工作流](../../../docs/development/git-workflow.md) 的相关章节。
AgenticRobot 版本化维护这份通用流程；工具也可对其他 Git 仓库运行。

1. 用 `git worktree list --porcelain` 发现所有者；检查相关工作区和语义范围。
   用 `scripts/worktree_inventory.py <repo> --branch <branch>` 检查指定候选，
   重复 `--branch` 纳入相关同伴；无法识别所有权时再做完整盘点。
2. 独立任务通过 `scripts/claim_worktree.py` 创建隔离工作区并记录范围。
   同一任务复用归属明确的工作区；已有目录、分支、资产和其他人的修改不覆盖。
3. 审查路径和语义重叠；盘点工具只报告事实，不能授权并行或集成。
4. 交接绑定 clean commit、base、行为结果、重要评审问题处理、验证命令、环境及未验证范围。
   新改动使旧的 ready 状态失效。
5. 获得具体集成授权后，在隔离分支合并固定提交并验证；更新目标前重查
   目标和输入 SHA。只提升已验证的提交，推送仍需对应授权。
6. 开发者说明保留资料；集成者按 Git 工作流的收尾章节处理或交接工作区去向。
   合并不授权删除；保留分支不要求保留 checkout，未解决的内容不强制清理。

普通单工作区修改不需要创建虚构的角色、监督者或额外记录文件。
同一次任务复用协调决定；范围、输入或所有权发生变化时重新评估。
