# AgenticRobot 开发约定

AgenticRobot 提供可独立复用的控制、运动、采集、数据、仿真、部署与下层 Gateway。
上层规划和 Memory 属于 RPent；开发协作与证据管理属于 Harness。
包的边界见 [运行域说明](docs/architecture/runtime_domain_boundaries.md)。

## 1. 选择开发入口

默认从已安装 Harness 的项目调用指南创建隔离 Task，遵守其生成的 AGENTS、scope、
writer lease 和验证流程。不要再为同一 Task 运行 claim_worktree 或维护 Git 分支描述。

独立 Git 开发可按 [轻量 Git 工作流](docs/development/git-workflow.md) 执行；
该流程由本仓库维护，可用于 RPent 等仓库。它不依赖 Harness，也不要求先下载机器人
资产。创建任务前说明选用哪种方式，保留既有工作区和所有者的修改。

## 2. 环境与资源按需准备

纯代码、测试和文档开发不要求机器人资产。只有本次任务需要仿真、模型或设备资产时，
才配置共享 `assets/` 根目录。独立 Git 工作流的认领工具会在资产可用时建立被 Git
忽略的兼容链接；缺少资产只产生警告，需要资产时使用 `--require-assets` 提前失败。

Python、依赖和锁文件以 [README_ENV.md](README_ENV.md) 为准。完整开发环境：

```bash
bash scripts/install_conda_env.sh --skip-vendor-sdk
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python scripts/dev_environment_doctor.py
```

同机 worktree 可以共享经过协调的 Conda 环境，但源码必须从自己的根目录解析；
不要让 editable 安装绑定到另一 worktree，也不要在共享环境中擅自更新依赖。
缺失机器人资产对纯文档/代码任务只是诊断警告；运行需要资产的检查前使用
`python scripts/dev_environment_doctor.py --require-development-assets`。

只有涉及本地资源、链接或迁移时才读
[资源说明](docs/setup/LOCAL_RESOURCES.md) 的相关分类，并运行
`python scripts/dev_local_resources.py audit`。共享 assets 是 read-mostly 路径兼容层，
不能通过链接修改 canonical store；mutable 采集和运行证据必须 writer 隔离。

## 3. 实现与验证

先定位职责和已有实现，保留一个清楚的实现路径。为当前复用或真实边界抽象，内部
调用者随重构更新；只有已知外部消费者需要时才保留兼容层，并说明退出条件。
验证外部输入和状态边界，避免沿内部调用链重复同一校验。

```bash
# 本地修改的行为、失败路径与相关契约：
bash scripts/run_pytest.sh -q <affected-test-paths>
git diff --check
# 环境/公共边界变更时：
bash scripts/verify_dev_base.sh
```

[测试说明](tests/README.md) 定义快速反馈、完整集成验收和显式硬件门禁。
文档编辑验证链接与命令即可；逻辑变更覆盖正常、边界、失败/取消/恢复及真实消费者。
重构检查公开行为和集合范围；禁止通过删减断言、静默 skip 或重复测试制造绿色结果。
覆盖率用于发现未覆盖路径，不能替代对正确性和安全边界的判断。

## 4. Review 与交接

检查实际 diff 是否解决当前请求、有无多余抽象或旁支清理。按可独立评审的职责组织
提交；必要的测试和文档随实现一起提交。说明固定 base/tip、主要变化、已跑检查、
环境及未验证边界。完整验收绑定待集成的组合提交，避免每个普通工作区反复跑同一
高成本套件。合并/推送另按对应授权和选定工作流执行。

本地临时推演可在 ignored `todos/`；需要共享的长期设计放在 owner 的文档目录。
不要默认产生逐步过程报告或平行的任务状态文件。
