---
name: integrate-shared-worktrees
description: Coordinate standalone Git worktrees, ownership, overlap review, immutable handoffs, and authorized integration. Use for independent Git development outside Harness; managed Tasks use their Harness contract.
---

# 轻量 Git 协作

先确定管理方式。默认 Harness；独立 Git 开发使用本 Skill。已受 Harness
管理的 Task 继续使用其 scope、lease 和证据流程，不重复写分支描述。

创建、交接或集成时读取
[Git 工作流](../../../docs/development/git-workflow.md) 的相关章节。
AgenticRobot 版本化维护这份通用流程；工具也可对其他 Git 仓库运行。

1. 用 `git worktree list --porcelain` 发现所有者；检查相关工作区和语义范围。
   用 `scripts/worktree_inventory.py <repo> --branch <branch>` 检查指定候选，
   重复 `--branch` 纳入相关同伴；无法识别所有权时再做完整盘点。
2. 独立任务通过 `scripts/claim_worktree.py` 创建隔离工作区并记录范围。
   一个任务一个所有者；已有目录、分支、资产和其他人的修改不覆盖。
3. 审查路径和语义重叠；盘点工具只报告事实，不能授权并行或集成。
4. 交接绑定 clean commit、base、行为结果、重要评审问题处理、验证命令、环境及未验证范围。
   新改动使旧的 ready 状态失效。
5. 获得具体集成授权后，在隔离分支合并固定提交并验证；更新目标前重查
   目标和输入 SHA。只提升已验证的提交，推送仍需对应授权。

普通单工作区修改不需要创建虚构的角色、监督者或额外记录文件。
同一次任务复用协调决定；范围、输入或所有权发生变化时重新评估。
