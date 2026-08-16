# `lab` M3c：parent + pinned child 组合更新

状态：叠加在 M3b 上的 review candidate。本切片解决同一主机中一个 stable parent 与其直接 pinned
gitlink children 的 checkout 联动；仍不包含 SSH、GPU、模型服务或设备执行。

## 第一部分：一般项目框架

Git 更新 parent commit 时只改变 gitlink，不保证已初始化 child checkout 自动切到该 commit。M3c 因此把
一次联动定义为可恢复的组合操作：

```text
fetch child objects, then parent ref (M3a, no checkout mutation)
       │
       ▼
plan-linked
  parent remote commit/tree
  each child target gitlink commit/tree
  exact canonical paths and identities
       │
       ▼
reserve every checkout before mutation
       │
       ▼
inventory every checkout; any process blocks all mutation
       │
       ▼
parent merge --ff-only, submodule recursion disabled
       │
       ▼
child checkout --detach <exact-gitlink>, no fetch, LFS smudge disabled
       │
       ▼
verify parent branch/HEAD/tree/clean + child HEAD/tree/clean + gitlink equality
       │
       ▼
release all reservations and publish receipt
```

用户入口：

```bash
./lab sync fetch --repos Controller
./lab sync plan-linked --repo Controller --json

./lab sync apply-linked \
  --repo Controller \
  --confirm-plan-sha256 <linked-plan-sha256> \
  --reservation-id LINK-Controller-001
```

`plan-linked` 与 `apply-linked` 首版只接受：

- 一个 `checkout_mode=stable_branch` parent；
- 一个或多个 direct `checkout_mode=pinned_gitlink` children；
- child 管理 checkout 的真实路径等于 `parent/submodule_path`；
- 所有 target Git objects 已由 M3a fetch 获得；
- 不存在 nested/overlapping gitlink graph。

直接关系限制是首版故意的 fail-closed 边界，不是框架硬编码具体仓库。后续可用图拓扑扩展多层 gitlink，
但必须先证明预留、更新顺序和恢复语义。

### 非伪原子与恢复

多个 Git checkout 不存在跨仓原子 commit。M3c 不宣称原子性：parent 快进成功、child checkout 失败时：

```text
status = QUARANTINED
recovery_required = true
parent 保持新 commit
child 保持可审计的旧 commit
所有 reservation 尽力释放
不执行 reset / clean / rebase / destructive rollback
```

receipt 保存每一步结果和最终 checkout facts。下一次 `plan-linked` 只在 parent 唯一 dirty path 集合等于已声明
gitlink paths、parent HEAD 已是目标 commit、child 仍 clean 且目标 object 可用时，给出
`HEAD_AT_TARGET_WITH_GITLINK_DRIFT` 恢复计划。恢复只完成 child exact checkout 和最终验证。

已完全一致时 `apply-linked` 幂等返回 `UP_TO_DATE`，不申请维护窗口。

### 安全与同步边界

固定约束：

- 所有 checkout 在任何 mutation 前完成组合 reservation；部分预留失败会释放已有预留；
- `/proc` 盘点任一 checkout 失败或发现相关进程时，一个 checkout 都不移动；
- parent 仅 `merge --ff-only`，明确关闭 submodule recursion；
- child 仅 `checkout --detach <exact OID>`；
- `GIT_LFS_SKIP_SMUDGE=1`，无 submodule fetch、tag fetch 或大文件下载；
- Task worktree 永不更新；
- 不 signal/kill 进程，不访问设备，不声称 motion safety；
- 从待更新 parent import 的控制面会被进程盘点阻塞，正式自动同步必须使用独立安装的受保护控制面。

## 第二部分：当前项目实例

当前项目映射为：

```text
AgenticRobot/main
  └── model/openpi @ exact OpenPI commit

RPent/main (独立 sibling)
```

因此：

- RPent 使用 M3b 单仓 guarded apply；
- AgenticRobot + `model/openpi` 使用 M3c linked apply；
- OpenPI stable checkout 必须是 `model/openpi` 的真实 submodule checkout（`.git` 为 gitfile），不能是旧的
  混合 full clone；
- M3c 不读取 checkpoint、模型配置、CUDA、数据或运行服务，gitlink target 是唯一源码联动事实。

远端 `user@192.168.70.116:/home/user/zc/AgenticRobot` 目前仍未满足受保护 clean anchor 与独立 control-plane
条件，不能直接运行 `apply-linked`。正确顺序仍是先合并候选、刷新 protected baseline，再在新路径
准备 clean clone/submodule，最后将旧目录保留为 rollback 证据。

M3c 的实现边界固定在 commit `76033197`；阶段分支已清理，当前连续交付线见
[`lab_framework_plan.md`](./lab_framework_plan.md)。

## 验证

```bash
python -m pytest -q tests/test_lab_linked_sync.py
ruff check lab_control tests/test_lab_linked_sync.py
mypy lab_control
git diff --check
```

测试覆盖成功更新、幂等 no-op、parent 已移动但 child 失败后的 quarantine/retry，以及运行进程阻塞下的零
checkout mutation。全部使用临时本地 Git remotes 和伪 `/proc`，不访问网络、大文件、GPU 或设备。
