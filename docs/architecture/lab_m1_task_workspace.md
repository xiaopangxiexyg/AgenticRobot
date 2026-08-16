# `lab` M1：项目无关的多仓 Task Workspace

状态：M1 review candidate。本文只描述本分支已经实现并由测试覆盖的能力，不代表本地 Run、远端执行、
GPU、可视化或设备控制已经启用。

## 第一部分：一般项目的多 Agent 开发框架

### 用户模型

日常开发面只暴露四个对象中的前两个：

- `Change`：一个功能变更，可聚合多个分别由不同主机或 Agent 开发的 Task；
- `Task`：一个主动开发位置，以及该位置内一个或多个仓库的可写 worktree；
- `Profile`：下一阶段加入的测试场景；
- `Run`：下一阶段加入的确定代码快照与测试证据。

M1 的命令为：

```bash
./lab change create CHG-0042 --title "cross-repository change"
./lab task create T101 --repos Controller,ComponentX --change-id CHG-0042
./lab open T101
./lab inspect T101
./lab task writer claim T101 --repo Controller --actor agent-a
./lab task writer status T101
```

底层仍使用 Git worktree，但用户不需要手工拼接 worktree 路径。每个 Task 自动生成：

```text
Worktrees/T101/
├── Controller/                  # 独立 branch/worktree
├── ComponentX/                  # 独立 branch/worktree
├── T101.code-workspace          # multi-root IDE workspace
├── task.env                     # Task 与仓库根目录
└── task.yaml                    # 分支、base commit、scope、writer epoch
```

环境在 M1 中明确为 `DEFERRED`。创建 Task 不安装依赖、不创建大型 CUDA 环境，也不下载模型、LFS、
submodule 对象或数据；后续执行 Profile 时才按需解析环境。

### 强制不变量

1. 管理 checkout 只作为稳定锚点和 worktree 管理仓；普通 Agent 在 Task 目录开发。
2. 一个 Task 只有一个 `development_host`；跨主机并行修改同一 Change 时拆成多个 Task。
3. 同一 Task 的同一 repository 同时只有一个有效 writer lease；不同 repository 可由不同 Agent 并行。
4. 创建前解析所有仓库的 exact base commit，并在写入前重新验证；任一仓库失败则不留下半个 Task。
5. `--scope` 是仓库级写入边界；`inspect` 会报告 branch drift、dirty 文件和 scope violation。
6. `--dry-run` 不创建 branch、worktree、manifest 或环境。
7. 活动 worktree、`.git`、虚拟环境、build 和日志不跨机器做文件级同步。
8. 跨机器未来只传 commit/synthetic snapshot、manifest、request、receipt 与 artifact。

writer lease 是协作授权记录，不是文件系统锁。它包含 Task、repository、development host、
`writer_epoch`、actor、lease ID 和到期时间。到期 lease 不会被静默接管；替换必须显式引用旧 lease ID。
如果同一仓库需要两个 Agent 同时修改，应创建两个 Task/branch，而不是绕过 lease。

### Git submodule 的一般规则

项目 manifest 可把一个 repository 声明为父仓中的 `pinned_gitlink`。当父仓与子仓同时被选入 Task：

- 子仓 worktree 放在父 Task worktree 的 gitlink 路径，不额外创建一个 sibling 副本；
- 子仓 base commit 必须等于父仓 exact tree 中的 gitlink；
- 不会隐式 fetch，也不会通过 checkout 下载 LFS 或大型运行资产；
- 父仓和子仓仍各有独立 branch、commit、diff、writer lease 和 PR。

这使“活动开发使用多仓 worktree”和“发布版本由父仓 gitlink 固定”可以同时成立。

### M1 明确不包含

- 自动 pull/rebase/reset/clean；
- 本地或远端测试执行、detached review snapshot 与 Run evidence；
- dirty snapshot 传输；
- SSH 调度、GPU/端口/设备 lease；
- 环境安装、服务启动、可视化、设备 observe/control；
- Task 删除或自动回收。

这些能力按 M2（本地 clean Run）、M3（远端 clean GPU-only）、M4（synthetic draft）顺序叠加，
避免一个不可审阅的大 PR 同时引入全部基础设施。

## 第二部分：当前 AgenticRobot / RPent / OpenPI 实例

项目特定的仓库名和路径只存在于 `integration/project.json`，`lab_control/` 核心没有这些标识符。
当前映射是：

| 逻辑仓库 | 管理位置 | checkout 模式 | 角色 |
| --- | --- | --- | --- |
| `AgenticRobot` | 当前 control checkout | `stable_branch` | 开发与集成 Hub |
| `RPent` | control checkout 的 sibling `RPent/` | `stable_branch` | 独立上层模块 |
| `openpi` | `AgenticRobot/model/openpi` | `pinned_gitlink` | 独立仓库的 submodule 边界 |

环境变量可覆盖物理位置：

```bash
LAB_REPOSITORY_AGENTICROBOT=/path/to/AgenticRobot
LAB_REPOSITORY_RPENT=/path/to/RPent
LAB_REPOSITORY_OPENPI=/path/to/openpi
LAB_HOME=/path/to/lab-root
```

典型本机任务：

```bash
./lab task create AR-0142 \
  --repos AgenticRobot,RPent \
  --scope 'AgenticRobot=deployments/**' \
  --scope 'RPent=integrations/**'

./lab task writer claim AR-0142 --repo AgenticRobot --actor agent-main
./lab task writer claim AR-0142 --repo RPent --actor agent-upper
./lab open AR-0142
./lab inspect AR-0142
```

涉及 OpenPI 源码时，父仓与子仓一起选中：

```bash
./lab task create AR-0143 --repos AgenticRobot,openpi
```

工具会验证 AgenticRobot base tree 的 `model/openpi` gitlink 与 OpenPI base commit 相同，并将子仓的
Task worktree放在 `AR-0143/AgenticRobot/model/openpi`。OpenPI 的 Python/CUDA/模型资产不会被合并进
AgenticRobot 环境；`model/openpi` 也不是 `model.openpi` Python package。

当前 PR 依赖顺序为：

```text
origin/main
  └── task/AR-openpi-submodule-v2       # gitlink + ownership boundary
        └── task/AR-lab-m1-workspace    # 本文描述的 M1
```

合并前必须先审阅并合并 submodule v2；随后把 M1 重基到新的 protected `main`，重新运行测试，再合并
M1。远端 `/home/user/zc/AgenticRobot` 的旧混合目录不作为这个流程的写入目标，也不会用 reset/clean
强行改造；远端 staging 与 clean source anchor 属于 M3。

## 验证

M1 的目标测试只创建临时 Git 仓库，不访问网络、GPU、设备或大型文件：

```bash
python -m pytest -q tests/test_lab_m1_workspace.py
ruff check lab_control tests/test_lab_m1_workspace.py
mypy lab_control
git diff --check
```

验收重点是：管理 checkout 不移动、多仓 Task 原子创建、multi-root workspace 正确、scope drift 可见、
writer lease 排他，以及 gitlink 子仓的路径和 revision 必须与父仓一致。
