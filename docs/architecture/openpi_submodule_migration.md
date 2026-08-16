# OpenPI `model/openpi` Submodule 迁移说明

- 状态：迁移候选已建立，尚未合并到 protected `main`
- Parent candidate：`AgenticRobot@82566f8fb73becfb9395eb2693522f81094c5ab9`
- Child pin：`Openpi@0d2d7f4e5526cca5f65e17b34f17ca8adad17407`
- Candidate branch：`task/AR-openpi-submodule`
- 远端执行主机：`user@192.168.70.116`

本文只定义 Git/source 迁移。它不改变模型、训练、推理、GPU、设备、checkpoint 或服务合同，也不授权
启动任何 runtime。

## 1. 最终边界

```text
AgenticRobot/                         # parent repository
├── .gitmodules
├── model/                            # AgenticRobot-owned Python/source namespace
│   ├── __init__.py
│   ├── arm_hold/
│   └── openpi/                       # mode 160000 child repository checkout
└── ...

RPent/                                # independent sibling repository

host-local runtime roots              # never submodule content
├── environments/
├── assets/
├── checkpoints/
├── datasets/
├── logs/
└── runs/
```

`model/openpi` 的名字表示源码放置位置，不表示把 OpenPI 变成 `model.openpi` Python 包。AgenticRobot 的
setuptools package discovery 不包含 `model*`；OpenPI 继续从自己的 `src/openpi` 和
`packages/openpi-client/src` 在独立 Python 3.11 environment 中解析。AgenticRobot 仍使用自己的 Python
3.10 environment。

## 2. Git 合同

`.gitmodules` 只声明：

```ini
[submodule "model/openpi"]
	path = model/openpi
	url = git@github.com:CYBER-QAQ/Openpi.git
```

必须保持以下不变量：

1. Parent tree 中 `model/openpi` 的 mode 是 `160000`，OID 是唯一版本事实。
2. `.gitmodules` 不声明 `branch`；`main`、`latest` 或 checkout 当前分支不能替代 gitlink。
3. OpenPI 保持独立 origin、branch、PR、writer lease 和 commit history。
4. OpenPI 内部 ALOHA/LIBERO 仍是第二层 exact gitlinks；不得 vendor 或展开进 AgenticRobot history。
5. 所有自动准备都设置 `GIT_LFS_SKIP_SMUDGE=1`，禁止 implicit submodule fetch 和大文件下载；缺 object
   时 fail closed，由受信 prepare 显式获取源码 object。
6. `.venv`、build、cache、asset、checkpoint、dataset、log 和 run output 永远不进入 gitlink，也不通过
   worktree/sync 复制。

## 3. 本机已完成的部分

- 活动 AgenticRobot working tree 已登记 `.gitmodules` 和 `model/openpi@0d2d7f4e...`；child 与两个嵌套
  submodule 均由已有本机 Git objects、`--no-fetch`、禁 LFS 初始化，状态 clean/detached。
- 为避免把落后且 dirty 的活动 `main@b7a20748...` 与框架改动混入迁移，另从
  `origin/main@ad1771da...` 创建独立 worktree/branch。
- `82566f8f...` 只包含 `.gitmodules` 和一个 gitlink，已经推送；没有源码复制或大文件。
- `integration/project.json` v2 将 child 默认路径相对 control checkout 解析为 `model/openpi`；这是通用
  `path_base` 机制，不是框架核心中的项目特例。
- `stack/baseline.json` v2 将 parent candidate gitlink 与独立 child commit 相等性绑定。其 baseline ID
  明确为 `candidate`，不得提升为 merge/release evidence。

## 4. Task / Workspace 行为

通用 allocator 根据 management repository 的实际嵌套关系和 parent base tree 的 gitlink 自动决定布局：

- Task 只修改 OpenPI：创建 OpenPI 自己的普通 task worktree/branch。
- Task 只修改 AgenticRobot：无需为了代码审视先初始化全部 child；snapshot 从 parent commit tree 读取
  gitlink，执行槽再 materialize exact child。
- Task 同时修改 AgenticRobot + OpenPI：OpenPI writable worktree 放在
  `<Task>/AgenticRobot/model/openpi`，同时作为 multi-root IDE folder；不会再创建一个版本可能不同的 sibling
  copy。
- parent base gitlink 与 child base commit 不同：Task 创建前直接拒绝。
- child 变更提交后：必须在 parent branch 更新 `model/openpi` gitlink，再生成 clean candidate snapshot。
- Profile 不需要 child 时，不强制解析无关 link；Profile 同时需要 parent/child 时必须证明三者相等：
  parent submodule OID = child commit = repository-link commit。

clean snapshot 使用 commit tree 读取 gitlinks，所以“submodule 未初始化”不等于“版本不确定”。真正创建
Run/test slot 时仍必须从受信 local anchor materialize 全部 exact submodules，并验证 clean、detached、
origin identity、tree OID 和 LFS 禁用状态。

## 5. 远端当前状态与禁止动作

2026-08-15 的只读审计显示：

- `/home/user/zc/AgenticRobot@ad1771da...` 的 index 已暂存相同 gitlink；
- 远端 `.gitmodules` 多出 `branch = main`；
- `model/openpi/.git` 仍是独立 Git directory；
- `model/openpi` 仍混有 `.venv/assets/logs` 等 host-local 内容；
- `/home/user/zc/openpi` 原工作目录仍存在且在相同 child commit；
- 没有 Git lock。

因此远端当前只是“index 已登记 gitlink”，不是完成的 clean submodule。收口前禁止：

- 在远端 `main` 直接提交当前 staged 变更；
- `git clean`、`reset --hard`、删除或覆盖任一 OpenPI 目录；
- rsync/syncthing 整个 AgenticRobot/openpi working tree；
- 把混合 checkout 当作 clean source anchor/test slot；
- 因路径已出现就启动 GPU、服务或设备测试。

## 6. 远端安全收口顺序

1. Review 并合并 `task/AR-openpi-submodule`；若 merge/rebase 改变 parent OID，记录最终 commit/tree。
2. 只读盘点远端两个 OpenPI 目录中的 tracked、ignored、runtime 路径、进程 cwd 和实际资产引用；为必须
   保留的 host-local 内容生成独立路径/校验收据。目录名不同本身不是备份证据。
3. 取得 maintenance reservation，证明没有 runtime、Agent、IDE 或测试占用将切换的 checkout。
4. 通过受信 fetch 获取最终 parent/child commits；禁止自动 pull/rebase/reset 活动 worktree。
5. 在新 staging path 从 exact parent commit 创建 clean checkout；使用现有本地 child/nested-child Git
   objects初始化 `model/openpi`，全程 no-fetch/禁 LFS。不要在旧混合目录上原地 clean。
6. 运行 source-only preflight：origin identity、parent tree、child tree、两层 gitlinks、dirty 状态、import
   roots和最大 tracked blob限制。不得加载 runtime dependency。
7. 在保留旧目录可回滚的前提下切换 canonical management path；切换后再次验证同一 identity。
8. 更新 Host/Preparation Profile receipt，使 test anchor、runtime assets 和 management checkout 三者路径
   分离；随后才允许 M3 clean GPU-only 准备继续。

这里的第 2、3、5、7 步都是远端写入/切换动作，必须由受信 wrapper/runner 执行并留 receipt；本次迁移
没有自动执行它们。

## 7. Merge 后 baseline 与证据

当前 development baseline 指向 task candidate。合并后必须原子更新：

```text
AgenticRobot commit/tree
AgenticRobot submodules[model/openpi]
openpi commit/tree/nested submodules
repository_links[AgenticRobot/model/openpi -> openpi]
baseline ID/hash
```

复用证据的最低条件是源码 tree、Profile、control plane、environment 和 assets 全部内容等价。设备或
release acceptance 默认在 merged candidate 上重跑；`82566f8f...` 的开发期证据不能自动升级。

## 8. 验收清单

- [x] 独立 migration branch/commit，不夹带业务代码或框架工作树。
- [x] `.gitmodules` 无浮动 branch。
- [x] parent gitlink 与 audited child commit 相等。
- [x] 本机 child 和嵌套 child clean、detached、no-fetch/禁 LFS 初始化。
- [x] project manifest 默认定位 `model/openpi`。
- [x] generic nested Workspace、base mismatch、uninitialized clean snapshot 有自动测试。
- [x] development baseline v2 绑定真实 parent tree/repository link。
- [ ] PR review/merge；记录 merged parent commit/tree。
- [ ] 远端 runtime/ignored content preservation receipts。
- [ ] 远端 clean submodule staging 与 canonical path 切换。
- [ ] merged baseline + remote fetch/source-anchor receipts。
- [ ] clean local/remote contract tests；GPU/device 仍属后续独立授权。

回滚时保留旧 parent checkout 和原 `/home/user/zc/openpi`，把 canonical management pointer/path 切回已知
clean parent commit；不删除 runtime assets，不改 child history，也不以文件同步“恢复”工作树。
