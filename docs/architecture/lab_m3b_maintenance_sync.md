# `lab` M3b：稳定 checkout 维护门禁

状态：叠加在 M3a 上的独立 review candidate。本切片只补齐**单个、无 gitlink 关系的稳定 checkout**
安全快进闭环；不把多仓/子模块切换伪装成原子操作，也不接触 SSH、GPU、模型、服务或设备。

## 第一部分：一般项目框架

M3a 已把网络 fetch 与 checkout 更新分离。M3b 在 checkout 更新前后增加：

```text
explicit fetched ref + confirmed plan hash
                    │
                    ▼
       short maintenance reservation
       （受管 runtime 启动门禁）
                    │
                    ▼
        same-owner /proc inventory
       （cwd/exe/cmdline/env/maps）
                    │
                    ▼
      revalidate exact branch/HEAD/tree/ref
                    │
                    ▼
             git merge --ff-only
                    │
                    ▼
       verify final commit/tree/clean state
                    │
                    ▼
       release reservation + persist receipt
```

维护门禁只协调采用同一控制面的 launcher。它不杀进程，不尝试接管未受管服务，也不声称具备设备安全权限。
发现 checkout owner 的任一相关进程、无法读取的同 UID 进程元数据、dirty tree、错误分支、ahead/diverged
历史或计划漂移时，更新均 fail closed。

### 用户入口

先显式 fetch，再从 JSON inspection 取得目标仓库的 exact plan hash：

```bash
./lab sync fetch --repos Component
./lab sync inspect --repos Component --json

./lab sync apply \
  --repo Component \
  --confirm-plan-sha256 <checkout_sync_plan_sha256> \
  --reservation-id MAINT-Component-001
```

`sync apply` 不访问网络，也不执行 pull、rebase、reset、clean、文件复制、submodule update 或 LFS smudge。
它一次只处理一个 `checkout_mode=stable_branch` 且不参与任何 `repository_links` 的仓库。成功 receipt 记录：

- before/after commit 与 tree；
- plan、maintenance policy、inventory 和 reservation hash；
- reservation 已释放；
- `network_used=false`；
- `task_worktrees_updated=false`；
- `submodules_updated=false`；
- `lfs_smudge_enabled=false`；
- `processes_signalled=false`、`device_access=false`。

命令是幂等的：计划已经是 `NOOP` 时返回 `UP_TO_DATE` 并留下无 mutation receipt，不申请维护窗口。

### 受信控制面约束

若 `lab` 本身从待更新 checkout import，进程盘点会检测到该依赖并拒绝自更新。这是预期行为。自动更新
AgenticRobot 这类控制仓之前，必须从受保护 tag/commit 安装独立控制面，使 executable、Python import、cwd
和环境均不指向被更新 checkout。所有受管 runtime launcher 在启动前必须调用
`assert_managed_runtime_start_allowed()`；没有接入门禁的 launcher 视为 unmanaged，M3b 不授权它并会由进程盘点阻塞。

维护 reservation 最长 300 秒，过期后 managed start gate 自动恢复；异常路径仍尝试写入 `RELEASED` 状态。
若 Git 快进已经完成但 receipt 发布失败，不做破坏性回滚；下一次 inspection 通过 commit/tree 事实恢复。

### 明确未覆盖

M3b 不更新参与 parent/child gitlink 的仓库。此类操作需要下一切片的组合维护窗口：先预留所有 checkout，
验证 parent remote gitlink 和 child object，再按可恢复顺序更新 parent/child，并在最终状态同时验证 parent clean 与
child exact pin。M3b 在此之前直接拒绝 linked repository，避免半完成状态被误报为成功。

## 第二部分：当前项目实例

当前关系为：

```text
AgenticRobot (stable_branch)
  └── model/openpi -> openpi (pinned_gitlink)

RPent (stable_branch, sibling, unlinked)
```

因此 M3b 的实际适用范围是 RPent 这类未参与 gitlink 的稳定 sibling checkout。AgenticRobot 和
`model/openpi` 必须等待多仓 gitlink transaction；远端旧的
`user@192.168.70.116:/home/user/zc/AgenticRobot` 仍不允许 pull/reset/clean。

此外，当前 stacked branches 尚未进入 protected `main`，所以本切片只作为 review candidate，不部署远端
受信控制面，也不自动更新远端 checkout。

评审顺序：

```text
task/AR-openpi-submodule-v2
  └── task/AR-lab-m1-workspace
        └── task/AR-lab-m2-local-runs
              └── task/AR-lab-m3a-stable-sync
                    └── task/AR-lab-m3b-maintenance-sync
```

## 验证

```bash
python -m pytest -q \
  tests/test_lab_runtime_inventory.py \
  tests/test_lab_stable_maintenance.py \
  tests/test_lab_project_maintenance.py

ruff check lab_control tests
mypy lab_control
git diff --check
```

测试只使用本机临时 Git checkout 和伪 `/proc` 目录，不访问网络、大文件、GPU 或设备。
