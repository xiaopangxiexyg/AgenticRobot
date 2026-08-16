# `lab` M3a：commit-based stable sync

状态：叠加在 M1/M2 之上的 M3a review candidate。本切片只实现 Git 元数据同步和受保护的
fast-forward 原语；它不自动更新 checkout，也不接触远端旧混合目录、Task worktree、运行进程、GPU 或设备。

## 第一部分：一般项目的跨机器同步框架

跨机器的一致性定义为：

```text
host A origin/main == commit M
host B origin/main == commit M
```

而不是两个活动工作目录逐文件相同。禁止同步：

```text
.git/
活动 worktree/
.venv/
build/install/log/
Runtime/
设备状态和本机配置
```

允许传递：commit、branch/ref、Task/Run manifest、receipt 和 artifact。

### 用户入口

```bash
./lab sync inspect
./lab sync inspect --repos Controller,ComponentX
./lab sync fetch
./lab sync fetch --repos Controller --timeout-seconds 120
```

`sync inspect` 完全只读，汇总：

- canonical repository identity；
- stable branch、HEAD/tree 与 fetched `origin/<branch>`；
- baseline pin 和父子 gitlink；
- active Task writer lease；
- 哪些仓库需要 fetch；
- stable checkout 是否具备 fast-forward 条件。

`sync fetch` 只更新显式的 remote-tracking ref：

```text
refs/heads/<stable-branch>
  -> refs/remotes/origin/<stable-branch>
```

固定约束为：

- `GIT_LFS_SKIP_SMUDGE=1`；
- `--no-tags`；
- `--no-recurse-submodules`；
- `--no-write-fetch-head`；
- 禁 hooks、auto-gc 和交互式凭据提示；
- 子仓先于父仓 fetch，避免父 gitlink 指向本机尚不可解析的 child commit；
- 不执行 pull、merge、rebase、reset、clean、checkout 或文件复制。

fetch receipt 明确记录 `network_used=true`，同时记录：

```text
checkout_updated=false
submodules_fetched=false
lfs_smudge_enabled=false
large_assets_downloaded=false
task_payload_authorized=false
```

### fast-forward 原语

`plan_stable_checkout_sync` 和 `apply_stable_checkout_fast_forward` 是受保护库接口，不是当前用户命令。
apply 只接受刚生成且重新验证仍一致的计划，并要求：

```text
canonical identity matches
current branch == protected stable branch
working tree clean
local HEAD is ancestor of fetched remote ref
plan repository/HEAD/tree/ref/hash unchanged
```

满足时只执行 `merge --ff-only`；ahead/diverged/dirty/wrong branch/missing ref 全部 fail closed。

自动同步尚缺少运行时维护门禁：在执行端必须先建立短时 maintenance reservation，证明没有进程从
稳定 checkout 加载源码，并阻止新受管 runtime 在 apply 窗口启动。该组合门禁和 scheduler 属于后续 M3b；
M3a 不把可写 apply 暴露给普通 Task 或远端 shell。

## 第二部分：当前项目实例

`integration/project.json` 声明：

| repository | stable branch | 模式 |
| --- | --- | --- |
| AgenticRobot | `main` | stable management checkout |
| RPent | `main` | sibling stable management checkout |
| openpi | `dexteleop/integration` | AgenticRobot `model/openpi` pinned gitlink |

OpenPI 同步的权威关系仍是：

```text
AgenticRobot exact tree
  └── model/openpi gitlink == OpenPI exact commit
```

`lab sync` 不初始化或下载 OpenPI submodule，也不下载 checkpoint、asset、LFS、CUDA 环境或数据。

远端 `user@192.168.70.116:/home/user/zc/AgenticRobot` 当前仍是旧 ref，且
`model/openpi/.git` 是混有环境/资产历史的完整 Git 目录，因此它不满足 clean stable anchor 条件。
M3a 不会对它执行 pull/reset/clean。后续应在新路径创建 clean management/staging clone，验证后再切换；
旧目录保留为 rollback 证据。

当前 stacked review 顺序：

```text
task/AR-openpi-submodule-v3
  └── task/AR-lab-m1-workspace-v2
        └── task/AR-lab-m2-local-runs-v2
              └── task/AR-lab-m3a-stable-sync-v2
```

只有前层进入 protected `main` 并刷新 baseline 后，M3a 才能进入远端部署候选。

## 验证

```bash
./dev test lab-m3a
./lab sync inspect --repos AgenticRobot

python -m pytest -q \
  tests/test_lab_stable_sync.py \
  tests/test_lab_project_sync.py \
  tests/test_lab_m3a_trusted_fetch.py
```

测试使用本地临时 bare Git remotes 和注入式 fetch runner，不访问互联网、大文件、GPU 或设备。
