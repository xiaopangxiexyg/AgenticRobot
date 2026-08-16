# 多 Agent 工程框架与 AgenticRobot 联邦项目开发流

- 状态：**Proposed baseline（分阶段实施中；当前为 M1-M2 与 M3 trust-contract bootstrap working tree）**
- 适用范围：通用多 Agent 软件工程框架，以及 `AgenticRobot`、`RPent`、`openpi` 的本地/远端实例
- 当前项目最后一次只读盘点：2026-08-15（Asia/Shanghai，M0 与 submodule 迁移均部分完成）

实施账本（2026-08-14）：M1-M2 bootstrap 已在当前 working tree 实现 `lab task create/open/inspect`、
多仓 preflight、task manifest、multi-root workspace、scope drift、受信本地 R0/R1 Profile、detached
clean snapshot、Run manifest、`lab test/runs/open --run` 和 AgenticRobot `./dev doctor/test`；它尚未
提交/发布为受保护 control-plane 版本。RPent 已增加独立 `./dev doctor/test`，三仓 per-repository
environment binding、baseline 自动补齐和 openpi source-only provenance 已实现；M0 已完成主要
Git/env/import/process/symlink 重采样但仍被 legacy 活动进程和远端 asset/source
分离阻塞。M3 已有离线 GPU-only request/Profile/host-preflight/receipt/state bootstrap，以及受保护
Service policy、inert launch spec、managed health identity/ServiceReceipt 校验和 `attach-existing` 精确匹配
合同；Task/baseline 共用 clean snapshot resolver、受保护 asset lock 和 `PENDING_DISPATCH` 本地 outbox
规划、pinned SSH-stdio dispatch 及其 `status/watch/cancel` 状态观测合同也已实现；正式 transport Profile
已用旧地址历史 ED25519 key 独立确认新地址身份，但 production 专用 known_hosts 与远端受保护控制面
尚未安装，因此真实 SSH dispatch 仍闭锁。受信 GPU-only request/attempt 还固定
`loopback-only` 与 `downloads=false`；主机互联网恢复不扩大 Task payload 权限。attempt/结果/artifact 合同已在 hermetic adapter 上实现，但
真实远端 executor、真实 slot/lease 部署、服务进程启动和 M3 Exit 尚未实现。持久 GPU inventory/lease/
cleanup-gated release 及其 host-runtime receipt 绑定已在临时目录实现，尚未在真实 GPU 上 reserve。受保护 command registry 已实现
ID→adapter 映射、严格字段白名单与 registry hash 绑定；通用核心不再内置 OpenPI、模型概念、入口脚本或
制品投影路径。OpenPI production service adapter 现由受保护的
`integration/service_adapters/openpi-policy-service.json` 声明并以 manifest SHA-256 绑定 launch plan；
项目制品观察也改为“主内容 role + 任意 companion roles”的通用结构，选择身份统一为项目定义的
可选 `workload_id`；具体 adapter 可在项目层将其解释成模型、构建产物或场景，没有该概念的项目保持
null。只有 adapter 模板或 allowlist 实际需要 workload 时才强制非空。当前 request state v8 已能原子
持久化 service sandbox、inert launch plan 与 network anchor/probe evidence；plan 仍固定
`launch_authorized=false`，尚未执行真实进程。
M4 已落地 synthetic draft snapshot、增量 bundle、
无污染 verifier 和惰性 detached slot source materializer，尚未开放真实跨机传输或远端执行。另已实现
library-only clean source-anchor 构建原语：只消费现有本地 Git objects，禁 hooks/LFS smudge/network，
显式绑定 submodule source，并拒绝超限 tracked blob；尚未获准写入远端目标路径。稳定管理 clone 的
commit-based sync 也已有两阶段 library 原语：只基于已 fetch ref 规划，重验后仅允许 ff-only，绝不
rebase/reset/clean 或同步活动 worktree。受信 explicit fetch 也已实现固定 branch refspec、禁 hooks/
submodule/LFS/asset、保持 checkout 不变并出具 receipt；尚未部署远端 CLI/scheduler。
内部 `projectctl trusted-prepare status|sync|source-anchor` 现仅接受受保护 Profile 中的固定 placement，
不接受任意路径或命令；ff-only apply 还要求短时 maintenance receipt 证明 runtime dependency inventory
完整且 checkout 空闲。没有 receipt 时显式 fetch 后只返回 `WAITING_FOR_MAINTENANCE`。该入口尚未部署到
远端。只读 checkout-owner `/proc` inventory collector 已在本地实现并可通过受保护入口调用；maintenance
receipt 还必须嵌入 30 秒内的无依赖 inventory，并绑定阻止新 runtime 启动的 scheduler reservation。
本地已实现持久 `flock` reservation state、受管 launcher start gate、异常释放和组合式
`trusted-prepare auto-sync`；真实远端尚未部署，legacy/unmanaged launcher 也尚未完成迁移，因此不能仅凭
一次进程扫描宣称自动同步已启用。
远端 OpenPI 小型 registry/norm-stats 事实另以非授权 asset-observation schema 记录；唯一现场候选
`dual_rm65_dmgripper_latest` 仍缺 artifact content manifest 和显式 workload selection，不能激活正式 asset lock，
也未读取或下载大模型。
受信 control-plane 还新增了小型 package builder：只允许 clean `main == origin/main`，固定白名单且限制
单文件/总量/archive 大小，逐文件记录 blob OID 与 SHA-256；当前未提交 checkout 会被拒绝。staging、
原子激活和 rollback 合同已在临时目录测试，但尚未安装到远端，因此这不是已部署 executor。Preparation Profile 已固定版本化 install
root，inert staging primitive 只会写 `versions/<commit>` 并保存三类 receipt，不更新 `current`、不执行；
transport 的计划入口改指 `control-plane/current/projectctl`，避免从活动/稳定源码 clone 直接运行控制面。
本地还已实现 health-checked atomic activation/rollback 合同：只用 Host Profile 的独立 Python 对 staged
`projectctl --help` 做 isolated 检查，逐文件复核后切换相对 `current` symlink；rollback 复核并恢复前一
版本，不删除任何版本。它尚未部署，且不能绕过 protected commit、SSH pin 和远端环境门槛。
2026-08-15 本机受限 metadata fetch 已把 `origin/main` 前移到 `ad1771da...`；远端 management clone 也在
同一 clean parent commit 上，但 staged 了 `.gitmodules` 与 `model/openpi@0d2d7f4e...`。本机活动
working tree 仍在祖先 `b7a20748...` 并有未提交框架改动，因此不会在该目录自动 ff-only；独立分支
`task/AR-openpi-submodule@82566f8f...` 已从 `ad1771...` 形成并推送。development baseline v2 只引用该
候选；合并后必须以新 parent commit/tree 重建 Candidate/Release baseline。

2026-08-16 P0 更新：本机三仓未提交的框架入口已分别形成本地 task commits/worktrees；原
AgenticRobot、RPent、openpi 稳定目录随后仅通过显式 `pull --ff-only` 更新并保持 Git clean，当前分别为
`AgenticRobot main@0f8bd913...`、`RPent main@90ed3678...`、
`openpi dexteleop/integration@0d2d7f4e...`。OpenPI gitlink 已从最新 AgenticRobot main 重建为独立 v2
候选 `task/AR-openpi-submodule-v2@fa53dec8...`；框架 task commit 位于其上一层，development baseline
只固定 v2 gitlink parent，不形成引用自身的循环 lock。旧 `82566f8f...` 分支不 force-push，保留为历史
回滚证据。四个新 task branches 已普通 push，但尚未 PR/merge；远端执行主机状态仍以最后一次只读审计为准。
已实现命令的准确边界见 [`lab_task_workspace.md`](./lab_task_workspace.md)，不得以本规划的目标描述
代替实现证据。

本文明确分成两个主体：

1. **第一部分是通用框架要求**：适用于一般 Git 项目的多 Agent 开发、worktree、环境、测试、
   可视化、运行证据和资源锁；不依赖 AgenticRobot 的包结构或机器路径。
2. **第二部分是当前项目实例规划**：把第一部分应用到 AgenticRobot、RPent、openpi、当前本机和
   `192.168.70.116`，处理多仓分散、远端同步、GPU/硬件和既有目录迁移。

第二部分继承第一部分全部 MUST/MUST NOT；若当前项目需要收紧规则，以第二部分的显式增量为准。
任何项目特例都不能降低权限、安全、证据完整性或可复现性。

| 关注点 | 通用规范位置 | 当前项目实例位置 |
| --- | --- | --- |
| Task / Workspace / Profile / Run | A1、A4、A10 | B6-B7、B9 |
| worktree/environment 内部机制 | A3、A5、A7 | B5-B6、B9.2 |
| 用户 `lab` 门面 / 内部 `wt`、`dev`、`projectctl` | A6 | B6 |
| quick test / review snapshot | A7 | B6、B13 M2-M4 |
| namespace / GPU / device lease | A8-A9 | B9 |
| run manifest / CI / roles | A10-A12 | B10-B11、B14 |
| 多仓 lock / 自动同步 / release | 通用项目层接口见 A6.4 | B1、B7-B8 |
| 跨主机单写者 / 精确 commit-tree 测试 | A2、A4、A11 | B4、B6、B9-B10 |
| 当前路径、SHA、迁移风险 | 不属于通用规范 | B3、B5、B12-B13 |

本文不改变现有运行合同，不授权启动仿真、模型服务、真实设备或机械臂。文中的目标命令不能被
视为已实现；已落地子集及其边界必须以本文实施账本、`lab_task_workspace.md`、代码和测试为准。

---

# 第一部分：一般项目的多 Agent 开发与运行测试框架

## A1. 框架目标与基本模型

日常开发者只需要理解四个对象；Integration Owner 额外使用一个轻量 Change 聚合层：

```text
Task        要完成的开发目标及唯一开发位置
Workspace   该任务涉及的全部仓库视图
Profile     预定义的测试场景、拓扑、能力和证据要求
Run         某次不可混淆的快照 + 场景 + 资源执行记录

Change      一个功能变更下、可分布在不同开发主机的多个 Task/PR/Run 集合
```

底层 worktree、Git ref、SSH、测试槽、环境、端口、ROS domain、lease 和 artifact 搬运必须隐藏在
统一工具中。高级维护者可以诊断这些机制，但日常开发者不能被要求手工拼接它们。

### A1.1 通用核心与项目集成边界

框架代码必须保持项目无关。边界固定为：

```text
lab_control/                         通用机制与安全校验
integration/project.json            仓库目录、别名和控制仓库
integration/profiles/               项目测试场景
integration/service_adapters/       项目服务入口、逻辑 token 和制品投影
integration/host_profiles/          主机能力与受保护路径
stack/                              项目 baseline、制品锁和 impact rules
```

`lab_control/` MUST NOT 出现具体仓库名、项目包名、模型 ID、项目入口脚本、设备名或主机绝对路径。
核心只理解 repository、source、asset role、service adapter、environment、resource、lease、snapshot 和
Run receipt 等抽象。项目清单可以包含这些具体事实，但必须来自受保护版本、严格校验并以内容哈希绑定；
Task 不能传入清单路径、raw argv、shell、挂载或权限。该边界由自动测试扫描，防止项目细节重新进入核心。

通用工程单元不是“一个 Agent”或“一条分支”，而是一个可审计任务：

```text
Task
  = Task ID
  + exact base commit(s)
  + one branch/worktree per involved repository
  + allowed write scope
  + isolated development environment
  + runtime namespace and resource leases
  + acceptance commands
  + run evidence
  + PR/review/CI lifecycle
```

推荐的工程形态是：

> **worktree + 独立开发环境 + 统一运行入口 + 运行时命名空间 + 设备资源锁。**

各机制解决的问题不同：

| 机制 | 解决的问题 | 不解决的问题 |
| --- | --- | --- |
| task branch | Git 历史与提交归属 | 文件、环境和进程隔离 |
| task worktree/clone | 工作目录、HEAD、index 隔离 | 依赖、端口、GPU 和设备隔离 |
| independent environment | 安装状态和 import provenance | 共享进程/设备冲突 |
| runtime namespace | build、log、socket、port、ROS、artifact 隔离 | 稀缺硬件排他权 |
| resource lease | GPU、相机、机器人等排他使用 | 业务安全、authority 和 E-stop |
| commit/review snapshot | 长测试对应确定源码 | 测试本身是否充分 |
| run manifest | 结果和代码/环境/资源可追溯 | 自动证明人工观察正确 |
| PR、CI、merge queue | 审查和最终串行化 | 本地任务隔离 |

## A2. 通用强制不变量

1. 一个任务 MUST 有稳定 Task ID、明确目标、确定 base SHA、依赖、write scope、资源范围和
   acceptance；Agent session 只是可替换的执行者。
2. 每个任务在每个涉及的仓库 MUST 使用独立短期 branch 和独立 worktree/clone；多个 Agent
   MUST NOT 操作同一可编辑 working tree。
3. canonical/main checkout MUST 只用于 fetch、审查、worktree 编排和集成，不交给普通 Agent
   实现任务。
4. 每个 worktree MUST 是可直接打开 IDE、进入 shell、运行测试和调试的完整项目 checkout，
   不能只被当成 diff 目录。
5. 每个 worktree MUST 使用独立 environment prefix、build/install 目录和运行时 namespace；
   共享范围仅限只读/内容寻址 cache 和显式只读数据。
6. 多个 worktree MUST NOT 共享一个被反复 `pip install -e <different-checkout>` 修改的环境。
   runner MUST 证明实际 imports 来自当前任务 checkout。
7. 实时查看可以读取 Agent 的 mutable worktree；快速局部测试可以在执行锁下原地运行。长时间
   测试、可视化、仿真、GPU 或设备测试 MUST 默认使用 clean commit 的 detached review snapshot。
8. 每次测试、可视化、仿真、notebook 或设备运行 MUST 写 run manifest；若运行期间源码变化，
   结果 MUST 标记 invalid，不能成为验收证据。
9. 端口、Unix socket、ROS domain、process group、GPU 和设备 MUST 通过 namespace/lease 分配，
   不能由 Agent 自行猜测空闲值。
10. 设备 lease 只证明资源排他性，MUST NOT 替代设备网关、业务 safety、authority、HOLD、E-stop、
    断连处理或人工运动授权。
11. 普通 Agent MUST NOT 直接推送受保护主干、force push、管理其他任务 worktree、修改共享集成
    热点、部署 runtime 或取得真实设备 control 权限。
12. 自动化在 dirty、origin/base/scope/import mismatch、namespace 冲突、lease 失效或证据缺失时
    MUST fail closed；不得自动 reset、clean、stash、force checkout 或扩大权限。
13. 跨主机协作时，同一任务、同一仓库、同一分支在任一时刻 MUST 只有一个主动写入 host；其他
    host 只能在 detached slot 消费 immutable clean-commit 或受校验 draft snapshot。写入权移交必须
    显式登记和交接。
14. 主机之间 MUST NOT 同步活动 working tree、linked worktree、`.git`、environment、build 或
    runtime 目录；跨主机传递的是 Git ref/commit 或 immutable draft snapshot、task/test manifest、
    result 和 artifact。
15. 后台自动化可以 fetch 远端 refs，但 MUST NOT 自动 pull/rebase/reset 活动 worktree；正在运行
    的测试必须绑定提交时已确定的完整 commit 集合，不能随 branch 前移。
16. 一个 Task MUST 只有一个 `development_host`；该任务全部 writable repositories 共同位于
    该 host 的一个 multi-root Workspace。若必须跨 host 并行写代码，应拆成多个子任务和独立分支，
    归入同一 Change，再由 integration task 合成，而不是把同一 Workspace 拼接在两台机器上。
    整 Task handoff 只用于无法合理拆分的紧耦合调试。

## A3. 通用目录模型

一个单仓或多仓项目可以使用以下外部布局：

```text
~/project/
├── repos/
│   ├── main-project/                    # canonical management/integration checkout
│   └── component-x/                     # 多仓时的 sibling canonical checkout
│
├── tasks/
│   ├── T101/
│   │   ├── main-project/                # task branch worktree
│   │   ├── component-x/                 # 任务涉及才存在
│   │   ├── T101.code-workspace          # multi-root IDE workspace
│   │   ├── task.env                     # 按 Profile 延迟生成的 namespace；不含 secret
│   │   ├── task.yaml
│   │   └── .lab/task-state.json         # 工具状态；用户不手工编辑
│   ├── T102/
│   └── _review/
│       └── T101/<snapshot-id>/          # detached multi-repo commit snapshot
│
├── runtime/
│   ├── tasks/T101/                      # mutable quick-run records
│   └── reviews/T101/<snapshot-id>/      # stable test/viz/device records
│
├── test-slots/                           # 可选：专用测试节点的可复用 detached slots
│   ├── slot-01/
│   └── slot-02/
│
├── envs/test-slots/                      # slot 专属、lock-hash 校验的环境；不在源码内
├── shared-cache/                        # download/wheel/ccache/sccache
│
└── state/
    ├── tasks/
    ├── namespaces/
    └── resource-leases/
```

任务 worktree 内可以有任务专属 `.venv/`、CMake build、ROS install/log 等 ignored 目录，但测试
日志、视频、图片、trace 和 notebook state 应写入外部 `runtime/`，避免污染源码和 Git status。

跨仓任务的 IDE 使用 multi-root workspace，但每个仓库仍有自己的 branch、commit、diff、PR 和
owner；IDE 便利性不能模糊 Git 边界。

## A4. Task manifest 与写范围

### A4.1 Change、Task、Workspace、Profile、Run 的关系

Change 只聚合目标、依赖和证据，不拥有 working tree 或 writer：

```text
Change CHG-0042
├── Task T101-main       development_host: computer-a
├── Task T102-component  development_host: computer-b; depends_on: T101-main
└── Task T103-integrate  development_host: computer-a
                         consumes snapshots from T101/T102
```

Task manifest 增加可选 `change_id`、`depends_on` 和 `consumes_snapshots`。Integration Task 消费的是
上游 immutable snapshot/merged commit，不是另一台机器的活动 Workspace：

```yaml
change_id: CHG-0042
depends_on: [T101-main, T102-component]
consumes_snapshots:
  - {task: T101-main, snapshot: "<snapshot-id>"}
  - {task: T102-component, snapshot: "<snapshot-id>"}
```

`lab inspect CHG-0042` 汇总子任务 DAG、development hosts、branches/PRs、consumed snapshots、阻塞项
和最新 Run。常规跨机并行优先创建 child Tasks + Integration Task；whole-Task handoff 保留为例外。

```text
Task manifest
  + baseline stack
  + task repository overrides
        ↓ resolve
Effective Workspace Snapshot
  + selected Test Profile
  + allocated Runtime Resources
        ↓ execute
Run manifest + artifacts
```

Workspace 是用户看到的多仓任务目录；Effective Workspace 是 runner 实际测试的确定版本组合。
两者不能靠当前目录或 `latest` 隐式推断。

### A4.2 Task manifest

通用 task manifest 示例：

```yaml
schema: multiagent.task.v1
task_id: T101
change_id: CHG-0042
title: Add runtime contract
base_release: release-v1.2.0
development_host: computer-a
depends_on: []
consumes_snapshots: []

repositories:
  main-project:
    base_sha: "<40-hex>"
    branch: task/T101-runtime-contract
    writer:
      host: computer-a
      epoch: 1
  component-x:
    base_sha: "<40-hex>"
    branch: task/T101-component-adapter
    writer:
      host: computer-a
      epoch: 1

write_scope:
  main-project:
    - src/runtime/**
    - tests/runtime/**
  component-x:
    - src/adapter/**
    - tests/adapter/**

integration_paths:
  - main-project/src/registry.py
  - main-project/pyproject.toml
  - main-project/.github/**

runtime_namespace:
  root: ~/project/runtime/tasks/T101
  ros_domain_id: 41
  ports:
    jupyter: 18101
    gradio: 18102
    tensorboard: 18103

environments:
  main-project:
    prefix: ~/project/tasks/T101/main-project/.venv
  component-x:
    prefix: ~/project/tasks/T101/component-x/.venv
  shared_cache: ~/project/shared-cache

resources:
  network: false
  gpu: false
  simulator: false
  real_device: false

acceptance:
  - repo: main-project
    command: ./dev test unit runtime
  - repo: component-x
    command: ./dev test integration adapter-contract
```

规则：

- `allowed/write_scope` 是授权上限；新增路径、依赖、仓库、公共协议、运行时资源或设备模式必须
  重新审查任务。
- `integration_paths` 是单写集成区，普通 Implementation Agent 只读；由 Integration Owner 在各
  模块接口稳定后统一接入。
- CI 应比较 `base...HEAD` 的全部 tracked/untracked 变更与 scope，不只看最后一次 commit。
- `task.env` 由 allocator 生成，不能成为 Agent 自行申请资源的后门。
- manifest 的 `writer.host + writer.epoch` 是任务内的静态分支写入 authority，不是“某 Agent 此刻正在
  编辑”的声明。actor 开始编辑前应取得短期 repository writer lease；receipt 绑定 task/repo/host/
  epoch/branch/worktree/manifest hash，续租和释放必须匹配 actor + lease ID。过期或 stale authority
  不能自动抢占，必须引用原 lease ID 显式替换。移交前必须完成 checkpoint/push、停止旧 writer 与相关
  run，再原子递增 epoch 并更新 task manifest；stale epoch 的 publish 必须被拒绝，两个 host/actor
  不能竞争 push 同一 task branch。
- 测试 host 若发现需要修改源码，应从所测 SHA 创建独立 `fix/<task>-<host>-<topic>` 分支，经 PR 或
  由 writer cherry-pick；不能直接修改原 task branch。

`development_host` 是 Task 级不变量；各仓 `writer_host` 默认必须与它一致。远端测试不改变它。
handoff 时整个 Workspace 一起迁移；只想并行修改其中一个仓库时，应拆分子任务。

### A4.3 Baseline + task overrides

项目维护一个经过验证的 baseline stack，固定各仓 commit、contract、默认 profiles 和兼容资产
版本。用户不手工输入所有 SHA；resolver 按以下顺序生成 Effective Workspace：

```text
Effective revision(repo)
  = task override commit/draft snapshot，若该 repo 被任务修改
  = baseline pinned commit，否则
```

Profile 再从 Effective Workspace 中选择实际需要的仓库，因此只修改主仓的任务无需创建所有组件
worktree。例如 `main + component-y` 测试可以使用 task 的 main override 与 baseline 的 component-y。
解析结果必须展开为全部完整 SHA、baseline/manifest/profile hashes 后才提交 Run，禁止把浮动 branch
或“当前目录”留给执行端解释。

若主项目用 Git submodule 固定某个组件，gitlink 是版本依赖边，不能与 stack lock 中另一份独立 pin
各自漂移。通用 manifest 应显式登记 `parent repository + submodule path + child repository`；resolver
先解析父仓 commit/tree，再把 gitlink OID 作为子仓默认 revision。Task 同时覆盖父仓和子仓时，两者必须
一致：父仓 index 中的 gitlink 必须等于子仓 snapshot commit，否则 snapshot 在提交 Run 前 fail closed。
Run 同时记录父仓 commit/tree、gitlink path/OID 和子仓 commit/tree，因此 squash、只改 gitlink 或只改
子仓任一情况都会使旧证据失效。

submodule 不改变“子项目仍是独立 Git repository、独立 branch/PR/writer lease”的事实。任务只读消费
submodule 时可使用 superproject worktree 加 detached、`--no-fetch` 的 task-local submodule checkout；
任务要修改 submodule 时，allocator 应改用隔离的 task clone/backend，并在嵌套 child checkout 中建立
任务分支。不要让多个 superproject worktree 共享同一个可写 submodule 工作目录。上述差异属于
Workspace 后端，用户仍只操作 `lab task/open/inspect/test/verify`。

框架当前已实现项目无关的 `lab.baseline.v2.repository_links`，每条边固定
`parent_repository/submodule_path/child_repository`。loader 拒绝未知仓、重复 path、环和 baseline
gitlink/child pin 不一致，同时继续只读兼容无依赖边的 `lab.baseline.v1`。clean resolver 按父到子顺序
处理依赖：若子仓不是 Task override，则从子仓本地 object store 解析父 gitlink 指定的 exact commit/tree/
submodule map，标记来源为 `parent_gitlink`；若父子都是 Task override，则必须相等。snapshot/Run 记录
解析后的 links；`lab.remote-request.v2` 与 `lab.remote-outbox.v2` 也携带同一关系并再次校验父 submodule
OID、child commit 和 link commit 三者一致。整个过程不 fetch、不初始化真实项目的新 submodule。

### A4.4 Test profile 与 capability/asset resolution

Profile 声明“需要什么”，而不是硬编码在哪个目录、USB 节点或 IP 上运行：

```yaml
schema: multiagent.test-profile.v1
profile_id: component-y-device-smoke
repositories: [main-project, component-y]
requires:
  capabilities: [component-y-runtime]
  devices: [component-y-arm]
  assets: [component-y-calibration-v5]
mode: automated
command_id: integration.component-y-device-smoke
draft_policy: observe-only
acceptance:
  trials: 10
  minimum_successes: 8
  maximum_safety_aborts: 0
  reset_protocol: component-y-scene-reset-v2
artifacts: [stdout.log, trace.json, camera.mp4]
```

Host inventory 把逻辑 ID 解析到本机路径/设备，并记录 checksum/version；这些实际路径不进入 task 或
公共源码。调度器选择满足全部 capabilities 的 host，preflight 在启动前验证 asset 存在性/checksum、
权限、驱动、设备在线、端口、时钟和 lease。`command_id` 映射到受审 argv，Profile 不携带任意 shell。

### A4.5 受信控制面与被测试载荷

存放在同一源码仓库不等于属于同一信任域。任何有 GPU、内网或设备能力的 Run 都必须分离：

| 信任域 | 来源 | 包含 | 约束 |
| --- | --- | --- | --- |
| trusted control plane | 受保护 tag/commit 安装的稳定版本 | `lab`、request validator、Profile authorization policy、allocator、executor、lease client、trusted command adapters | 不能从 Task Workspace import；版本/hash 进入 Run |
| task payload | Task/merge candidate/release snapshot | 三仓源码、repo-local tests、repo-local `./dev` | 被视为未受信输入，只能在控制面授予的 sandbox/network/resource mode 中运行 |

Profile 的业务定义可以随 Task 修改并在 hermetic/mock 环境测试，但未合并 Profile 不得自行增加
GPU、网络、asset、observe/control 或 command 权限。高权限执行采用 protected control-plane 中的
authorization envelope，对 Task profile 做交集/校验；请求增加的能力一律拒绝。

`./dev` 可以属于 task payload，但不能直接取得硬件 SDK/设备节点。device observe/control 必须经过
受信 adapter/Device Gateway。控制面至少记录：

```yaml
control_plane:
  version: lab-v0.1.3
  commit: "<protected-commit>"
  profile_policy_sha256: "<64-hex>"
  executor_sha256: "<64-hex>"
```

日常 `lab` 从受保护版本安装或固定环境启动，不能从当前 Task checkout 隐式 import/执行自身修改。
修改 `lab`、Profile policy、executor 或 trusted adapter 的 Task，只能先做无特权自测，经独立审查
合并并发布新 control-plane 版本后，才可授权后续高权限 Run。

## A5. Worktree 与独立环境合同

### A5.1 Worktree 生命周期

推荐短期分支：

```text
task/T101-runtime-contract
task/T102-teleop-timeout
task/T103-visualization
```

不推荐长期 `agent-a`、`codex-work`、`develop-agent` 分支。一个 branch 同时只能由一个 worktree
checkout；编排器必须解析 `git worktree list --porcelain`，不能通过目录名猜测。

创建和删除 worktree 属于 orchestrator/Integration Owner。普通 Agent 不运行跨任务的
`git worktree remove/prune`，也不使用 `--force` 处理不干净的他人工作区。

### A5.2 Environment 与 import provenance

`lab task create` 只创建 branches/worktrees/manifest/Workspace；`lab open` 应立即可用，不同步构建
GPU/CUDA、编译工具链或大型 SDK 等重环境。第一次 `lab test/session <Profile>` 才延迟解析和准备：

```text
immutable heavy base
  keyed by dependency lock + Python/CUDA + platform + host profile
+ task/snapshot-specific first-party source overlay
+ Profile-specific extras
```

每个 worktree/run 的 first-party overlay 独立；可以共享：

- pip/uv/conda 下载 cache；
- wheel cache；
- ccache/sccache；
- 只读 base image 或 lock-addressed dependency layer；
- 明确只读且内容可校验的数据集/模型 artifact。

不能共享：

- editable install state；
- repo-local generated package metadata；
- CMake/ROS build/install/log；
- user-site 中指向任意 checkout 的 `.pth`/editable link；
- task output、socket、PID 或 service state。

统一 `./dev doctor` 必须验证：

1. 当前目录、Task ID、branch、HEAD、dirty 状态与 manifest 一致；
2. Python executable/environment prefix 属于当前 worktree 或明确允许的只读 base + 隔离 overlay；
3. 任务涉及的 first-party package `__file__` 全部来自 manifest 指定 checkout；
4. user site disabled，dependency lock/hash 正确；
5. build/output/run root、ports、sockets、ROS domain、GPU/设备 lease 无冲突；
6. host profile、config 和 resource mode 没有越过任务授权。

框架核心不得硬编码某个项目的包名、Python 版本或测试文件；这些由各仓受版本控制的
`integration/dev.json`（或等价 repo-local manifest）声明。通用入口只解析配置、证明 import
provenance 并调用已登记测试目标。项目专有名称只能出现在项目 manifest、Profile、adapter 和测试代码中。

heavy base 不得包含指向任一 checkout 的 editable install、`.pth` 或 first-party generated metadata。
正式 runner 推荐设置 `PYTHONNOUSERSITE=1`，并把 base image/env hash、overlay、executable、import
roots 和 lock hashes 写入 run manifest。

## A6. 统一开发命令分层

框架只有一个日常用户入口 `lab`；既有三层机制成为内部 implementation/admin interfaces。`lab` 不
复制状态机，而是读取同一 task/profile/run schemas 并调用下层 typed API：

```text
lab           Task / Open / Inspect / Test / Verify / Runs（唯一日常入口）
 ├─ projectctl    baseline/host/scheduler/ref-sync/release（受控管理面）
 ├─ tools/wt      branch/worktree/snapshot/env/namespace（内部任务引擎）
 └─ ./dev         当前 checkout 的仓库级 doctor/test/viz/device adapter
```

### A6.1 `lab`：面向任务和场景的用户入口

日常命令保持小而稳定：

```text
lab task create T101 --repos main-project,component-y [--dev-host computer-b]
lab open T101
lab open --run <run-id>
lab inspect T101 [--diff] [--run <run-id>]
lab inspect T101 --against-run <run-id>
lab test T101 [<profile>|--recommended] [--interactive] [--follow|--detach]
lab verify T101 [<profile>|--recommended]
lab session T101 <profile>
lab runs T101
lab runs open <run-id>
lab runs watch <run-id>
lab runs cancel <run-id>
```

- `task create` 从 baseline 建 manifest、所需仓库 worktrees 和 multi-root Workspace；不预建重环境、
  不申请 runtime namespace。未修改的 baseline repo 不必创建 writable worktree。
- `open` 根据 `development_host` 打开本地或受控 Remote SSH Workspace；用户不输入真实路径。
- `open --run` 创建/复用 read-only multi-repo review Workspace，固定 Run 实测 trees，并附带 manifest/
  artifacts；`inspect --against-run` 显示失败 Run 到当前 Task 的综合 diff。
- `inspect` 汇总 Effective Workspace、每仓 branch/revision/dirty/source（baseline 或 override）、综合
  diff、scope、recommended profiles、latest snapshot/run 和 artifact links。
- `test` 生成 draft 或 clean snapshot，解析 Profile/capabilities/assets，准备本地/远端 slot、lease、
  tunnel 和 Run；默认先展示将运行/跳过的场景与原因。
- `verify` 强制 clean；若 commit 尚未发布，可在确认后只 push manifest 允许的 task refs，但不自动
  发明 commit、不更新 main。全部 commits 可从受信 remote 取得后，执行正式验收矩阵。
- `runs` 只按 Run ID 打开聚合报告、日志、视频、trace/viz；用户不寻找远端目录或手工 rsync。
- `--follow/--detach` 不改变 Run 生命周期；SSH/客户端断线后受控 worker 继续执行。`watch/cancel` 通过
  Run ID 查询/请求取消，取消只触发反序 teardown，不声称设备已经安全停止。
- `session` 是受控交互入口：本地可进入 task shell；远端 GPU-only 进入 detached snapshot、独立 GPU
  lease、无 device group 的登记 session；device observe 只能进入受信只读 adapter；device control
  永不提供通用 shell。

用户不应为正常流程直接执行 `git worktree`、detached switch、SSH、端口分配、ROS domain 分配、
设备 lock 或 artifact copy。遇到失败时，`lab inspect` 必须显示下层诊断与可执行修复建议，而不是
隐藏错误原因。

### A6.2 `tools/wt`：内部任务/worktree 引擎

```text
tools/wt list
tools/wt create T101 --repo main-project --repo component-x
tools/wt path T101 [--repo main-project]
tools/wt open T101
tools/wt shell T101 [--repo main-project]
tools/wt diff T101
tools/wt status T101 [--json]
tools/wt env create T101
tools/wt run T101 --repo main-project -- <argv...>
tools/wt test T101 --repo main-project unit <scope>
tools/wt checkpoint T101 --publish
tools/wt review T101
tools/wt viz T101 --snapshot <id> <mode> ...
tools/wt notebook T101 --snapshot <id>
tools/wt resource acquire T101 <resource>
tools/wt device run T101 <logical-device> --mode observe -- <argv...>
tools/wt finish T101
tools/wt remove T101
```

行为约束：

- `create` 从 exact base SHA 创建各仓 worktree，生成 task manifest 和 multi-root workspace；环境与
  runtime namespace 按 Profile 延迟创建。
- `open` 使用 host profile 配置的 IDE 新窗口；VS Code 可用 `code -n`，但不是唯一实现。
- `shell` 使用带 Task ID 的 terminal/tmux session，从精确 worktree 开始。
- `diff/status` 汇总每仓 branch、HEAD、相对 base commits、tracked/untracked、scope violations、
  worktree 状态、namespace/lease 和最近 run。
- `env create` 创建独立环境并立即运行各仓 `./dev doctor`。
- `run/test` 取得 execution lock、加载 task namespace、进入精确 worktree，再调用 `./dev` 或明确
  argv；禁止 shell 字符串拼接绕过 argv 审计。
- `checkpoint --publish` 要求 scope 合规、生成明确的 WIP/checkpoint commits，并仅向 manifest 中
  各仓 task ref 推送；它不更新 main、不会触发 deployment。
- `review` 固定全部相关仓库的 checkpoint commits，生成 detached multi-repo snapshot。
- `viz/notebook/sim/device` 默认要求 snapshot；输出只写外部 runtime root。
- `finish/remove` 要求 branch 已 push、PR/receipt 存在、worktree clean、任务进程停止、lease 释放
  且精确路径匹配。

这些命令面向 orchestrator 和高级诊断，不是日常教程要求用户记忆的入口：

### A6.3 `./dev`：仓库级入口

```text
./dev doctor
./dev test unit <scope>
./dev test integration <scope>
./dev viz <registered-mode> ...
./dev notebook
./dev tensorboard
./dev sim <registered-profile> ...
./dev device <logical-device> observe
```

`./dev` 不是第二套业务 runtime。它只负责 doctor、环境装配、调用仓库现有 canonical
module/profile、输出路由和 receipt。它必须：

- 从当前 checkout 解析源码和 repo-local config；
- 执行前验证 import provenance、environment 和 namespace；
- 将 argv、cwd、有限 env、canonical entrypoint 和输出路径写入 run manifest；
- 不复制已有测试、仿真、可视化或设备实现；
- 不自动申请更高设备模式，不把 `observe` 升级为 `control`。

### A6.4 `projectctl`：项目/发布管理入口

单仓且无部署的项目可以不实现它；多仓或多主机项目用于：

```text
projectctl status [--json]
projectctl doctor
projectctl stack verify --ref <tag-or-commit>
projectctl sync-main --mode fetch-only
projectctl task handoff <task> --from <host> --to <host>
projectctl test submit <task> --target <host> --snapshot <id> --profile <profile>
projectctl test status <request-or-run-id>
projectctl test logs <run-id>
projectctl test open <run-id>
projectctl remote open <task> --host <host>
projectctl deploy --ref <release> --dry-run
projectctl deploy --ref <release>
projectctl rollback --ref <previous-release>
projectctl report --json
```

`projectctl` 不理解具体 pytest 文件名，也不允许普通 Agent 借项目工具取得 main、release、远端或
设备权限。

## A7. Mutable 快测与稳定 review snapshot

### A7.1 实时查看与 execution lock

实时查看可以直接读取 Agent 的 worktree、打开 IDE 和查看 diff，但查看者不得同时编辑同一文件，
也不应在未协调时启动 formatter、codegen、dependency install 或长测试。

锁分为三层，不能用粗粒度 Task `TESTING` 阻塞所有仓库：

```text
Repository execution lock
  单仓 quick test；只暂停该仓 writer

Workspace snapshot barrier
  生成一致的多仓 snapshot 时，短暂冻结 Profile 涉及的 writable repos

Runtime/resource lease
  snapshot 完成后用于长测试；不继续阻塞开发 worktrees
```

快速局部测试流程：

1. `wt run/test` 取得目标 repository 的 advisory execution lock；其他仓 writer 可继续工作；
2. 保存开始时 HEAD、tracked content hash、untracked manifest；
3. 调用当前 worktree 的 `./dev`；
4. 结束时再次计算源码状态；
5. 开始/结束源码不同则标记 `invalid_source_changed`；
6. 写 run manifest并释放 repository lock。

多仓 snapshot builder 取得涉及仓库的 barriers，按稳定 repo key 顺序加锁，捕获全部 trees 后立即
释放；禁止边等待边长期占锁。一个 Task 可以有多个 Agent 分别持有不同 repository writer lease，
但同一 repo/branch 仍只有一个 writer。Integration Owner 接管同一 branch 也必须递增 writer epoch。
lease 转换和 snapshot barrier 必须使用同一原子状态与稳定锁顺序；snapshot/Run 记录 lease receipt hash，
使“谁在什么 authority 下产生该快照”可审计。这些锁依赖 Agent/runner 遵循工具，不能保证时使用
detached review snapshot。

dirty synthetic snapshot 对每个实际发生修改的 repository 强制要求 ACTIVE actor lease；无 lease、过期
或 stale authority 必须在写 synthetic ref 前失败。clean detached Run 不要求 writer lease，因为 reviewer/
CI 可以只读验证不可变 commit，但仍应记录运行开始时的 lease fingerprint 作为协作审计信息。

### A7.2 Synthetic draft Git snapshot

`lab test` 可以在 execution lock 下从 dirty Workspace 生成不可变 draft snapshot，无需用户手工
commit/push：

```text
each repo current HEAD
  + temporary index (never user's index)
  + tracked working-tree state
  + explicitly allowlisted untracked source
      -> git write-tree
      -> git commit-tree (parent = exact HEAD)
      -> synthetic commit OID + tree OID

multi-repo snapshot ID
  = synthetic/clean commit OIDs + tree OIDs + task/baseline/profile hashes
```

snapshot builder 必须检查 write scope、路径穿越/symlink、secret/ignore denylist、文件类型与大小；
ignored 文件和未知大型文件默认不包含。它使用独立临时 index，不移动用户 branch/HEAD、不写用户
index，也不在用户历史创建普通 commit。Git 原生 tree 负责 binary、mode、symlink 和 rename 后内容；
submodule 必须固定 gitlink/独立仓 snapshot，LFS 必须验证 pointer 与 object 可取得，不能隐式遗漏。
构建过程由受信 control plane 在隔离环境执行，禁用 repository hooks；clean/smudge filter 只允许受保护
allowlist 中的实现，未知 filter 直接失败，避免待测 `.gitattributes` 或本地 Git 配置借 snapshot 构建执行
任意命令。

synthetic commits 通过临时 `refs/lab/snapshots/<task>/<snapshot-id>/<repo>` 或经验证的 Git bundle
传输；只能由受信 snapshot publisher 写入专用 namespace，不能触发 CI、deploy 或 release workflow。
ref 有 TTL/receipt，GC 前确认 Run 保留策略。执行端只需 detached checkout 并核对 commit/tree OID，
不再解释自定义 patch overlay。Run 记录 `type: synthetic-draft`、parent commit、commit/tree OID、
included/excluded untracked 列表和 `dirty: true`。synthetic ref 不属于受保护历史，也不能被误当签名
release commit。

draft 默认只允许 offline、sim 和隔离 GPU-only。Profile 可显式允许 H1/observe-only 的交互观察，但
仍需设备 lease、无 motion 权限和人工确认；draft 禁止 device control、motion、required check、正式
验收和 release。`lab verify` 始终要求 clean commits、可从受信 remote 获取，并重新运行测试。

### A7.3 Detached multi-repo review snapshot

长测试、可视化、notebook、仿真、GPU 和设备运行的 snapshot 结构：

```text
tasks/_review/T101/7f31c2d9/
├── main-project/                     # detached @ exact checkpoint SHA
├── component-x/                      # detached @ exact checkpoint SHA
├── snapshot.yaml                     # all commits/task/lock hashes
└── T101-review.code-workspace
```

规则：

- snapshot ID 由全部仓库 commits、task manifest hash 和 dependency lock hashes 派生；
- worktree detached、初始 clean，runner 禁止修改 tracked source；
- review environment 独立，不能复用 mutable development editable environment；
- 输出只进入 `runtime/reviews/<Task ID>/<snapshot-id>/`；
- 原 Agent 可继续开发，但新 commit 不影响已经开始的 review run；
- 多仓任务必须整体 snapshot，禁止从 PATH、父目录或 user site 随机导入另一个 checkout；
- required CI、正式验收、device control 和 release 必须在 clean commit/snapshot 上重跑。

### A7.4 专用测试节点的可复用 slot

高频远程测试可以预留少量 `test-slots/<slot-id>/`，减少重复创建目录和下载依赖，但 slot 只是
detached snapshot 的可复用容器，不能变成长期可编辑 checkout。每次使用必须按以下顺序：

1. 取得独占 slot lock，并确认上一轮登记进程已停止、资源 lease 已释放；
2. fetch 请求中的 exact commits，把各仓切到 detached 状态并验证 clean；
3. 清理 manifest 已登记的 repo-local generated outputs，禁止模糊删除或清理未知文件；
4. 选择 `slot ID + dependency lock hash` 对应的独立环境，lock 变化时重建，不复用错误 editable state；
5. 运行 `./dev doctor`，取得 namespace/GPU/device lease 后执行测试；
6. 发布 run manifest/artifacts，释放资源和 slot lock。

大型模型、只读数据、SDK、编译/下载 cache 和运行结果放在 slot 源码之外。slot 可以复用 cache，
不能在两个并发请求间共享 writable env、进程、端口或设备状态。

## A8. 运行时 namespace、进程和可视化

`task.env` 由 allocator 生成，示意：

```bash
export TASK_ID=T101
export TASK_ROOT=~/project/tasks/T101
export RUN_ROOT=~/project/runtime/tasks/T101
export ROS_DOMAIN_ID=41
export JUPYTER_PORT=18101
export GRADIO_PORT=18102
export TENSORBOARD_PORT=18103
export APP_SOCKET=~/project/runtime/tasks/T101/sockets/app.sock
export MUJOCO_GL=egl
```

- allocator 必须验证端口/ROS domain/socket registry 和实际占用；不能只对 Task ID 取模。
- 同一次 Run 中通过 loopback 通信的 service 与 client MUST 加入同一个、由受信 anchor 持有的隔离
  network namespace；分别执行两次 `--unshare-net` 会得到两个互不可见的 loopback，明确禁止。
  端口 lease 只证明框架内没有重复分配，不证明 namespace 已创建、socket 已绑定或 client 可达。
- inner sandbox 可以继承已验证的外层 network namespace，但在 anchor receipt、service/member join
  receipt 和 loopback probe 全部就绪前，其 argv 不能直接执行，更不能继承 host network。
- 首个通用 OS observation collector 只从受控 `/proc` 证明：anchor 是 runner 的直接独立 session、
  `NoNewPrivs=1`、anchor netns inode 与 runner 不同，且 PID/start-ticks/netns 在采集前后稳定；service 或
  attempt member 的 netns inode 必须与 anchor 完全一致。observation 固定 `loopback_verified=false`、
  `external_network_absence_verified=false`、`connectivity_authorized=false` 和
  `launch_authorized=false`。因此“同一 inode”仍不能替代后续 namespace 内固定探针。
- 固定 namespace probe 只能由安装版 `NetworkNamespaceProbePolicy` 提供 enter/Python executable；控制面
  以 `O_NOFOLLOW` 打开、散列并通过继承 FD 执行两者，清空 Task 环境、固定 cwd/stdin/timeout。探针只进入
  已观测 anchor netns、绑定一次临时 `127.0.0.1` socket，并读取本地 interface/IPv4/IPv6 route；不接受
  Task argv/path/endpoint，也不连接任何外部地址。receipt 证明只有 loopback interface、没有非 loopback
  route，并绑定 policy/executable hash、anchor、netns inode 与短 TTL；使用前重新验证 executable、PID
  start-ticks、netns 和 `NoNewPrivs`。它仍固定 `launch_authorized=false`，只是后续 launcher 的一个前置条件。
- GPU 先由 lease 登记 device/memory budget，再写 `CUDA_VISIBLE_DEVICES`；环境变量不是 lease。
- Jupyter、Gradio、TensorBoard、RViz、Foxglove、MuJoCo viewer 等必须显示 Task ID、branch、
  commit/dirty、snapshot ID、端口/ROS domain、profile 和 run directory。
- 进程按 Task ID 登记 PID/process group；退出只停止本任务登记的进程，禁止模糊 `pkill`。
- 可视化输出进入 run artifact 目录，不能散落在源码 worktree。

测试/运行类型：

| 类型 | 代码状态 | CI/证据 |
| --- | --- | --- |
| quick offline | mutable worktree；允许 dirty，保存 content state | development evidence；repository lock |
| remote draft offline/sim/GPU-only | immutable base + validated dirty snapshot | draft evidence；不能作为 required/release |
| stable unit/integration | clean detached snapshot/CI merge commit | 可作为 required/acceptance evidence |
| sim/viz/notebook | 默认 clean detached snapshot | 人工观察与自动断言分开记录 |
| device observe | 默认 clean snapshot；Profile 可显式允许 draft H1 | lease + 受控入口；不授予 control/motion |
| device control | reviewed clean snapshot/release | 独占 lease + 业务 safety + 人工授权 |

## A9. Resource/Device lease 与安全分级

单机非硬件资源可用带 metadata 的文件锁。多台电脑能访问同一资源时，各自 `flock` 无法互相看见，
必须使用：

- 资源连接 host 上的权威 lease service；或
- 中央 lease database/service；或
- 稳定 Device Gateway 的排他 session。

建议命令：

```text
tools/wt resource acquire T101 gpu-0
tools/wt device acquire T101 camera-front --mode observe
tools/wt device status robot-arm-01
tools/wt device run T101 robot-arm-01 --mode observe -- ./dev device robot-arm-01 observe
tools/wt device release T101 camera-front
```

lease receipt 至少包含 resource ID、Task ID、snapshot、全部相关 commits、host、mode、exclusive、
acquired/expires、heartbeat 和 owner。设备使用逻辑 ID/profile，不把 `/dev/video2`、USB 枚举顺序
或裸 IP 当稳定公共接口。

Profile 需要多个 host/GPU/device/camera/gripper/physical cell/operator 时，scheduler 必须采用组合
预留：先验证并原子 reserve 全部资源，再进入 Prepare；任一资源失败则整体释放，不能先长期占住
A 的设备再等待 B。为避免分布式死锁，资源按稳定全局 key 排序，reservation 有 TTL/幂等 key。

`physical-cell:<id>` 和 `operator:<role>` 也是资源；它们证明场地/人员被预留，不证明场景已正确
摆放。setup/reset/home 另生成 fixture receipt。GPU/file lock 也不能决定机械臂控制 owner：稳定 runtime、
test slot、remote dev session 之间的 runtime authority handoff 必须由受信 Device Gateway/authority
层完成并产生 receipt；同一时刻只允许一个 control owner。

设备模式：

| 模式 | 允许 | 额外要求 |
| --- | --- | --- |
| observe | 读状态、图像、协议、记录和可视化 | observe lease；禁止控制模式改变和动作下发 |
| control | reviewed smoke/action | exclusive lease、exact clean snapshot、业务 authority/safety、急停和现场授权 |

lease 丢失或心跳过期时，runner 停止新请求；实际设备如何 HOLD/停止仍由 Device Gateway/业务下层
合同负责。框架不能自行声称硬件已经安全停止。

## A10. Run manifest 与 artifact 证据

每次 `lab test/verify`（底层 `wt run/test/viz/notebook/sim/device`）创建唯一目录：

```text
runtime/reviews/T101/7f31c2d9/
└── 2026-08-13T20-15-33Z-rollout/
    ├── run.yaml
    ├── stdout.log
    ├── stderr.log
    ├── snapshot.yaml                 # commits/tree OIDs/synthetic refs
    ├── config.snapshot.yaml
    ├── services/                     # per-run ServiceReceipts
    ├── fixture/                      # setup/reset/recovery receipts
    └── artifacts/
```

最低 schema：

```yaml
schema: multiagent.dev-run.v1
run_id: T101-20260813T201533Z-rollout
task_id: T101
snapshot_id: 7f31c2d9
snapshot_type: clean-commit
evidence_tier: development
baseline_hash: "<sha256>"
profile_id: component-y-device-smoke
profile_hash: "<sha256>"

control_plane:
  version: lab-v0.1.3
  commit: "<protected-commit>"
  profile_policy_sha256: "<sha256>"
  executor_sha256: "<sha256>"

repositories:
  main-project:
    branch: task/T101-runtime-contract
    commit: "<40-hex>"
    tree_oid: "<git-tree-oid>"
    dirty: false
    parent_commit: null
    source: task-override
  component-x:
    branch: task/T101-component-adapter
    commit: "<40-hex>"
    tree_oid: "<git-tree-oid>"
    dirty: false
    parent_commit: null
    source: baseline

command:
  argv: ["./dev", "test", "integration", "runtime-component"]
  cwd_repo: main-project

environment:
  python: "3.11.x"
  python_executable: "<review-env>/bin/python"
  dependency_lock_hashes: {main-project: "<sha256>"}
  import_roots: {main_package: "<review-path>"}
  ros_domain_id: 41
  ports: {jupyter: 18101}
  base_environment_hash: "<sha256>"
  first_party_overlay_hash: "<sha256>"

resources:
  hosts: [computer-b]
  gpu: null
  device: null
  lease_receipt: null
  physical_cell: cell-01
  operator: operator-01
  authority_receipt: null
  assets:
    component-y-calibration-v5: "<sha256>"

topology:
  nodes: [component-y-node]
  clock_sync: {status: passed, max_offset_ms: 1.2}

services:
  - receipt: services/component-y-policy.yaml
    mode: ephemeral

fixture:
  scene_id: component-y-smoke-v3
  robot_serial: "<logical-or-redacted-id>"
  firmware: "<version>"
  initial_joint_state_sha256: "<sha256>"
  gripper_state: open
  calibration_assets: [component-y-calibration-v5]
  operator: operator-01
  setup_receipt: fixture/setup-001.yaml

attempt:
  attempt_id: A-003
  ordinal: 3
  parent_attempt_id: A-002
  retry_reason: prior_infrastructure_failure
  recovery_receipt: fixture/recovery-002.yaml

configuration:
  host_profile_hash: "<sha256>"
  files: [{path: "<path>", sha256: "<sha256>"}]

outcome:
  classification: FAILED_ASSERTION
  execution_completed: true
  device_acknowledged: true
  assertions_passed: false
  task_success: false
  evidence_complete: true
  source_stable: true
  exit_code: 0
  duration_seconds: 46.2

artifacts:
  - stdout.log
  - services/component-y-policy.yaml
  - fixture/setup-001.yaml
  - artifacts/trace.json
```

结果分类至少包含 `PASSED`、`FAILED_ASSERTION`、`BLOCKED_PRECONDITION`、`ABORTED_SAFETY`、
`INFRASTRUCTURE_FAILURE`、`INVALID_SOURCE`、`INVALID_ENVIRONMENT`、`INCONCLUSIVE_MANUAL` 和
`CANCELLED`。`execution_completed`、设备 acknowledgement、断言、任务成功和 evidence completeness
分别记录，不能折叠成一个退出码。

Run 是 snapshot/Profile/control-plane/resources 的不可变执行包络；每次 retry/trial/recovery 创建新的
append-only Attempt receipt，不覆盖旧失败。请求内容变化则创建新 Run。成功率类真机 Profile 必须定义
trials、minimum successes、maximum safety aborts 和 reset protocol；一次 exit 0 不等于验收通过。

每个被连接的服务必须先提供受信可验证的 `ServiceReceipt`，不能以“端口能连通”代替身份：

```yaml
service_id: component-y-policy-R-20260813-044
run_id: R-20260813-044
host: computer-b
mode: ephemeral
source: {commit: "<40-hex>", tree_oid: "<git-tree-oid>"}
model:
  checkpoint_sha256: "<sha256>"
  config_sha256: "<sha256>"
  norm_stats_sha256: "<sha256>"
runtime:
  environment_sha256: "<sha256>"
  gpu_uuid: "<logical-or-redacted-id>"
  endpoint: http://127.0.0.1:18427
  pid: 12345
  started_at: "<rfc3339>"
```

服务模式：`ephemeral` 由当前 Run 启动并使用独立 socket/port；`attach-existing` 只能连接受管理服务，
且 receipt 必须与请求的 source/model/config/contract/authority 完全匹配；`unmanaged` 一律拒绝。
Gateway 也要握手 Run ID、commit/tree、contract hash、authority mode 和 runtime owner。端口号、PID 或
进程名都不是身份。

交互式 notebook/viz 的人工结论必须记录 reviewer、观察项和结论，不能伪装成自动断言。每个
视频、图片和 trace 必须能反查 run manifest；孤立 artifact 不构成证据。

## A11. 通用多 Agent 交付流

### A11.1 单机与跨主机交付主线

```text
Issue/Task
  -> dependency DAG + exact base + scope + acceptance
  -> branch + worktree + environment + namespace
  -> Implementation Agent
  -> quick tests under execution lock
  -> checkpoint + detached review snapshot
  -> stable tests/viz/sim/device evidence as applicable
  -> Draft PR + self-review
  -> independent review + required CI
  -> merge queue
  -> protected main
  -> release/deploy by separate role
  -> orchestrator cleanup
```

角色：

| 角色 | 允许 | 默认禁止 |
| --- | --- | --- |
| Implementation Agent | 任务 scope 内实现、quick tests、checkpoint、任务分支 push | main、integration paths、其他 worktree、release/device control |
| Review Agent | 只读 diff/run manifests、复现 stable tests、提出 review | 修改被审 PR、批准自身最后 push、使用未授权资源 |
| Integration Owner | integration paths、跨仓协调、stack/manifest、合成测试 | 绕过 review/CI、发布未合并临时 SHA |
| Release Operator | 从受保护 ref 构建、部署、回滚、receipt | 修改源码或把 management clone 当 release |
| Resource Operator/Gateway | lease、host identity、设备模式和安全前置 | 接受 Agent 自声明权限或绕过下层安全 |

主干至少启用 required PR/checks、CODEOWNERS、非最后 push 者 review、禁止 force push 和 merge
queue。CI 必须在 merge-group 合成 ref 上运行，并调用与本地相同的 `./dev` 合同，生成相同 schema
的 run manifest。

### A11.2 跨主机传输平面

两台主机的一致性定义为“针对某个任务/请求，解析到相同的 immutable commit/tree OIDs”，而不是
两个活动目录逐文件相同：

| 平面 | 传递内容 | 允许机制 | 禁止内容 |
| --- | --- | --- | --- |
| source | commit/ref/Git objects；受校验的 immutable draft snapshot | 受保护 Git remote、snapshot store；离线时经校验的 bundle | 活动 working tree、linked worktree、`.git` |
| control | task manifest、test request、取消/审批状态 | CI、任务服务或受控 SSH 调度 | 任意 shell 字符串、隐式最新版 |
| result | run status、run manifest、人工观察结论 | CI check、result store | 只有“通过/失败”而无 commit 组合 |
| artifact | log、视频、trace、图表、报告 | artifact/object store 或显式拉取 | 把 `runtime/` 反向覆盖到源码目录 |

每台主机持有独立 canonical clone 和本机 worktrees。测试请求至少固定：Task/Change ID、全部仓库
commit/tree OID、control-plane/Profile policy、target capabilities、资源模式、assets、snapshot/config
hashes 和提交者。执行端：

1. 验证请求 schema、授权、target host、commit 可取得性和 profile；
2. fetch 所需 Git objects，在本机直接 detached checkout Profile 所需仓库的 clean 或 synthetic
   snapshot commits，并核对 tree OID；
3. 创建独立环境和 namespace，运行 `./dev doctor`；
4. 需要时取得 GPU/设备 lease，并执行已注册 argv；
5. 生成 run manifest/artifacts，将结果绑定 exact commit 组合后返回；
6. 测试 worktree 可按保留策略回收，开发 worktree不受测试期间新 push 影响。

`lab test` 默认允许 A7.2 的 draft snapshot，从而隐藏手工 commit/push/fetch；Profile policy 决定它
是否仅限 offline/sim/GPU-only，或可进入 H1 observe。`lab verify` 使用 checkpoint/clean commits，
结果才可成为 required check、正式验收或 release evidence。dirty snapshot 永不用于 device control。

### A11.3 远端 hosted development 例外路径

当修改必须在某个 host 的专有 SDK、实时内核、USB/PCIe 设备或高频硬件反馈中迭代时，可以把
writable task worktree 放在该 host；操作者从主开发机通过经批准的 Remote SSH/IDE/tmux 进入。
“操作者位于主机 A”不等于“源码位于 A”：Git writer 身份由 worktree 所在 host 决定。

该模式必须满足：

- 先对整个 Task 显式 handoff development host；原 host 的全部 writable repos checkpoint/push、
  停止写入后，新 host 才创建完整 multi-root Workspace；
- 远端 Workspace 的各仓有独立 branch、env、namespace 和 scope，不编辑 canonical clone/test slot；
- 远程入口固定 host/task/repo，使用最小权限配置，不成为任意 SSH 或设备权限后门；
- 短小现场修复优先从被测 SHA 建独立 `fix/*` 子任务，无需转移原任务；
- 修改完成后 push/PR，集成 Hub 以 detached review worktree 审查并负责最终合并。

这是一条低频例外路径。通用逻辑、协议、算法、mock/sim 和跨仓改动默认仍在主开发机完成，再把
exact commits 发送到专用节点测试。

### A11.4 Request 幂等、取消与断线恢复

远端请求使用内容寻址 `idempotency_key`，覆盖 snapshot、Profile、control plane、assets、target 和
resource mode；同 key 重复提交返回原 Run，request ID 相同但内容不同必须拒绝。状态机：

```text
SUBMITTED -> RESERVED -> PREPARING -> RUNNING -> COLLECTING -> COMPLETED|FAILED
                  \-> CANCEL_REQUESTED -> CANCELLED
                  \-> QUARANTINED
```

worker 生命周期独立于提交 SSH/client；状态和 receipts 原子持久化，重连用 Run ID 继续 watch。
cancel 是幂等请求：停止进入新步骤并反序 teardown；真实设备是否 HOLD/安全仍由本地 Gateway
receipt 证明。基础设施失败 retry 新建 Attempt，不覆盖失败证据。

### A11.5 Evidence 层级与 promotion

```text
Development Evidence   Task/synthetic snapshot；用于迭代和 PR review
Candidate Evidence     PR merge candidate/merge-group exact trees
Release Evidence       merged stack candidate/release tag
```

promotion 比较所有仓库 commit/tree OID、Profile policy/hash、assets、control plane、host environment、
service/fixture/resource receipts：

1. commit 和全部上下文完全相同，可直接引用原 evidence；
2. commit 不同但所有参与仓的 tree OID 与全部上下文相同，可生成签名/审计的
   `content-equivalent` promotion receipt，不重写原 Run；
3. 任一 tree、依赖仓、Profile/policy、asset、control plane、环境或配置变化，证据 stale，必须重跑；
4. synthetic draft 只能是 Development Evidence，不能 promotion；
5. device control 与 release acceptance 默认在 merged release candidate 上重跑，即使 tree 等价也不
   自动提升；task branch 真机结果只作为 review evidence。

### A11.6 标准 Run 生命周期

```text
Resolve -> Reserve(all resources atomically) -> Prepare -> Preflight
  -> Setup/reset/home -> Start(dependency order) -> Ready Barrier
  -> Execute Attempt -> Assert / Observe -> Teardown(reverse order)
  -> Recovery or Quarantine -> Publish
```

每步写 receipt 并可安全重入；Ready Barrier 验证全部 ServiceReceipts/authority/fixture，而非仅检查
端口。自动断言和人工观察分开。失败进入 teardown/recovery 后仍必须 Publish 不完整/失败证据。

## A12. 通用验收清单

- [ ] 每个任务有 Task ID、development host、base、overrides、scope、environment、namespace 和 acceptance。
- [ ] canonical checkout 不由普通 Agent 编辑。
- [ ] 用户只需 `lab task create|open|inspect|writer|test|verify|runs`；不手工管理 worktree/SSH/slot/port。
- [ ] `lab inspect` 展开 Effective Workspace、baseline/override 来源、diff、推荐测试和最近 Run。
- [ ] `lab open --run` 可打开实测只读源码；Run 可 watch/cancel；session 权限按 local/GPU/observe/control 分级。
- [ ] 同一 Task 的全部 writable repos 位于一个 development host；跨 host 并行开发以 Change 聚合子任务。
- [ ] manifest writer authority 与短期 actor lease 分离；同一 Task/repository 只有一个未过期 actor，
      expired/stale lease 只能按旧 lease ID 显式接管。
- [ ] 每个 worktree/review snapshot 环境独立，`./dev doctor` 证明 import provenance。
- [ ] quick test 使用 repo lock，多仓捕获使用短 snapshot barrier，长 Run 只持 runtime/resource lease。
- [ ] draft 使用不改 branch/index 的 synthetic commits/tree OIDs；`verify`/control/release 使用 clean commits。
- [ ] Baseline + task overrides 自动解析完整 commits；Profile 只 checkout/运行需要的仓库。
- [ ] Profile 通过逻辑 capability/device/asset ID 调度，preflight 校验 checksum/driver/clock/lease。
- [ ] 端口、socket、ROS、GPU 和设备有 namespace/lease；进程按任务精确停止。
- [ ] 组合资源原子预留；runtime control owner 由 Gateway authority receipt 唯一确定。
- [ ] trusted control plane 与 task payload 分离；Run 记录 lab/policy/executor 受保护版本。
- [ ] 每次 Run 记录 commits/tree、services、fixture、attempt、outcome、environment、resources 和 artifacts。
- [ ] integration paths 单写，diff scope 由 CI 执行。
- [ ] 跨主机时每个 task branch 有唯一 writer host；测试端使用 exact commit/tree 的 detached worktree。
- [ ] 活动源码、`.git`、environment/build/runtime 不做文件级同步；request/result/artifact 分层传递。
- [ ] 远程测试请求固定全部 clean/synthetic commit 与 tree OID、profile 和资源，结果可反查同一组合。
- [ ] 高频测试 slot 有独占锁、detached commit/tree、slot/lock-hash 环境和进程/resource 清场协议。
- [ ] 远端 writable worktree 仅在显式 writer handoff 后建立，不与 test slot/canonical clone 混用。
- [ ] Agent 不持有 main force-push、release、remote deploy 或 device control 权限。
- [ ] PR/CI/merge queue 和本地 runner 使用同一个 `./dev` 命令合同。
- [ ] request 有 idempotency/cancel/reconnect；证据按 Development/Candidate/Release 规则 promotion/失效。

---

# 第二部分：AgenticRobot、RPent、openpi 特殊项目规划

## B1. 项目决策与完整 stack 身份

当前项目采用 **asymmetric hub-and-spoke federated polyrepo（非对称、宿主机感知的联邦多仓库）**：

- `AgenticRobot` 是主项目和 stack 集成仓库；仓库名是 `AgenticRobot`，产品/包文档中的
  `DexTeleop` 名称继续保留，本规划不做源码或 Python 包重命名。
- `RPent` 是逻辑上层 Agent 控制面，但保持独立 sibling repository。
- `openpi` 保持独立训练/推理 fork 和独立 PR/branch 历史；其集成位置固定为 AgenticRobot
  `model/openpi` submodule，由 AgenticRobot gitlink 固定版本。本机与远端都已 staged 同一 gitlink，
  但迁移 commit、远端 clean source/runtime 分离和 baseline v2 更新尚未完成。
- GPU 运行实例当前位于 `192.168.70.116`。主机互联网现已恢复，但 Task payload 仍保持
  `loopback-only`、`downloads=false`；联网能力只能由受保护 prepare/fetch 流程显式授予。
- 远端主项目固定目标目录是 `/home/user/zc/AgenticRobot`；clone 完成后它是远端独立的 canonical
  management clone/source anchor，不是 runtime，也不是远端 Agent 编辑区。
- 三仓通过 AgenticRobot gitlink/stack lock、版本化协议、contract snapshot、policy artifact 和跨仓 CI
  联动；openpi 的 stack lock 条目只能镜像并校验 gitlink，不能成为第二个版本事实源。运行时不使用
  目录复制或隐式 `PYTHONPATH` 作为永久耦合。
- GitHub 受保护仓库是源码事实源。“便捷自动同步”表示同步审查后的 Git ref/release，不表示
  双向覆盖可编辑 working trees。

```text
                           protected Git remotes
                    AgenticRobot / RPent / openpi
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
        local-dev-control                    remote-gpu-hardware
        Development Hub                     Specialized Spoke
        三仓主要源码开发/集成                 AgenticRobot/openpi 固定 commit/tree 测试
        RPent upper 与本地仿真                 GPU、设备、远端可视化
        task/writer/test 调度                 少量硬件耦合 debug/fix 开发
        PR/review/stack release               不拥有 main 最终集成权
```

| Host | 主要角色 | 常态写入 | 例外写入 |
| --- | --- | --- | --- |
| 本机 `local-dev-control` | Development Hub、Integration Owner、Test Orchestrator | AgenticRobot、RPent、openpi 的大部分 task branches | 无；设备能力不因控制面角色而获得 |
| 远端 `remote-gpu-hardware` | GPU/device Execution Node、Test Runner、OpenPI 专用环境 | 默认无；测试使用 detached commit/tree | 硬件强耦合 openpi/AgenticRobot fix，必须独立 branch 或 writer handoff |

完整身份：

```text
Stack identity
  = AgenticRobot release tag / commit
  + 该 commit 内 components.lock 固定的 RPent commit
  + 该 commit tree 内 `model/openpi` submodule gitlink
  + components.lock 对该 gitlink 的相等性校验与兼容元数据
  + Gateway/policy/data contract hashes
  + reviewed host placement profile
```

lock 不记录 AgenticRobot 自身 commit，避免自引用；主项目 release tag 固定该 commit。

### B1.1 对新增建议的取舍

| 建议 | 本规划决定 | 原因/落点 |
| --- | --- | --- |
| 两机不实时同步活动 Git 目录 | 完整采纳，作为 MUST NOT | A2/A11、B2/B8；ref sync 与专用 clean anchor 前移均不触碰活动目录 |
| 两机各自 clone、本机 worktrees | 完整采纳 | B4-B5；远端测试用本机 detached worktrees |
| 同一 task branch 单一 writer host | 完整采纳 | A4、B6、B9；测试机修复走独立 fix branch |
| checkpoint + exact commit/tree remote testing | 完整采纳 | `tools/wt checkpoint` + `projectctl test ...`，见 B9 |
| 增加统一 `lab` 用户门面 | 完整采纳，但不复制底层状态机 | `lab` 只编排 schemas/API；`projectctl`、`tools/wt`、`./dev` 转为内部/admin 接口 |
| self-hosted runners 按能力标签路由 | 条件采纳并收紧 | 普通 PR 保持 hermetic；GPU/device runner 只接受受保护输入，见 B10.7 |
| dirty snapshot 跨机快测 | 条件采纳 | draft 可用于 offline/sim/GPU-only，Profile 可显式允许 H1 observe；禁止 control/verify/release |
| Git bundle | 仅作离线应急 | 正常路径仍是受保护 Git remote；bundle 必须 verify/hash/source 校验 |
| 主机 A clone 子项目 Y 源码 | 完整采纳 | 本机目标 source-only openpi clone；可做源码、unit/contract/mock 开发 |
| 副机固定测试槽位 | 条件采纳 | 允许少量独占 detached slots；env 按 slot + lock hash 管理，见 B9.2 |
| A 远程控制 B 的 writable worktree | 低频例外采纳 | 仅硬件强耦合任务；走 writer handoff/独立 fix branch 和受控 remote-open |
| 两机稳定 main checkout 自动前移 | 仅对专用 clean anchor 条件采纳 | refs 自动 fetch；远端 anchor 可在严格锁和 ff-only 检查下自动前移，活动/test worktree不动 |
| Change 聚合跨机 Tasks | 完整采纳 | 保持单 Task 单 host；常规并行用 child + integration Tasks，whole handoff 降为例外 |
| control plane 与 task payload 分离 | 完整采纳，列为高权限前置 | 受保护 lab/policy/executor 决定权限；Task 代码不能自我提权 |
| patch overlay draft | 不采用为主协议 | 改用临时 index + synthetic Git commit/tree OID；Git 原生传输和复现 |
| evidence/fixture/attempt/service identity | 完整采纳 | 防止 task 证据误升 release、旧端口服务串用和物理状态不可追溯 |
| 一次实现全部能力 | 明确不采用 | B13 改为 M0-M8；首个交付只覆盖本地 clean Workspace 到远端 clean GPU Run |

## B2. 当前项目增量不变量

这些规则是在 A2 基础上的项目专属收紧：

1. 三个项目 MUST 保持三个独立 Git 历史；不改 monorepo。RPent 保持 sibling repository；openpi
   位于 AgenticRobot `model/openpi` gitlink，但迁移前后都禁止文件级同步活动目录。初次 gitlink commit、
   后续 bump 或 checkout 搬迁都是独立迁移/集成变更，不与业务代码或 runtime asset 搬运混在一起。
2. 所有长期组合 MUST 固定完整 40 位 commit 与 contract/artifact hashes；runtime 不依赖浮动
   branch、`latest` 或机器偶然存在的文件。
3. 本机和远端 MUST NOT 双向同步可编辑 AgenticRobot working tree；旧排除 `.git` 的单向源码
   镜像也只能作为单写过渡，不能成为多 Agent 正式协议。
4. AgenticRobot 是下层 Gateway、执行、safety、authority 和 dispatch owner；RPent 不重建这些
   状态，openpi 不直接控制硬件。
5. 远端 management clone 与 immutable runtime MUST 分目录；ref 更新不得 restart 服务或接触设备。
6. host 配置、设备 assignment、calibration、secret、dataset、checkpoint、logs、cache、env 和
   runtime state MUST 位于 Git/sync 之外。
7. 未受信 PR CI MUST NOT 在能访问内网 GPU/相机/机械臂的 runner 上执行。
8. `/home/user/zc/AgenticRobot` 的 submodule 迁移未 commit 前不得把 staged 状态当作稳定 baseline、
   test anchor 或 release evidence；中间态不是失败证据，也不能被自动 sync 移动。
9. 每个任务中的 AgenticRobot、RPent、openpi 分支 MUST 分别登记唯一 writer host；远端测试使用
   detached clean 或 synthetic draft commits，不成为同一 task branch 的第二写入者。
10. “源码在哪台机器 checkout”不等于“仓库属于哪台机器”。三仓已用 commits MUST 可从受保护
    remote 取得；GPU、设备、驱动、校准和运行数据等 capability 才是 host-local 事实。
11. 本机 `local-dev-control` 是默认 task/writer 分配、PR、stack lock 和最终集成 Hub；远端不得因
    拥有运行能力而直接更新 main、components lock 或集成热点。
12. 远端 writable worktree 只用于显式登记的硬件耦合任务。canonical clone、detached test slot 和
    release runtime MUST NOT 被转换成开发目录。
13. 本机应能取得 openpi source 并运行不依赖远端能力的 unit/contract/mock；远端 SDK、设备、
    calibration、secret、大模型和数据不因源码可获得性而复制到本机。
14. AgenticRobot 项目中的一个 Task MUST 只有一个 development host。跨仓任务的全部 writable
    AgenticRobot/RPent/openpi worktrees 位于同一 host；需要本机/远端并行写入时拆成子任务和集成任务。
15. 用户日常只操作 `lab` 的 Task/Profile/Run；绝对路径、SSH、测试槽、端口、ROS domain、设备锁和
    artifact 传输不得成为正常测试步骤。
16. Effective Workspace MUST 从 AgenticRobot baseline release/commit + gitlinks +
    `components.lock` + task overrides 解析；openpi 迁移后以 AgenticRobot gitlink 为权威 pin，lock
    只能匹配它。未修改组件自动使用 baseline，不能要求用户逐个输入 SHA。
17. 远端 GPU/device Run MUST 使用受保护 control-plane 版本；Task 修改 `lab`、Profile、runner 或
    `./dev` 不得扩大自身能力、网络、asset 或设备权限。
18. OpenPI/Gateway 服务 MUST 以 ServiceReceipt 完成身份匹配；固定端口可达、PID 存在或 tmux 名称
    均不足以证明服务属于本 Run。
19. 真机 Run MUST 记录 physical fixture、operator/setup、firmware/calibration、Attempt/Outcome；
    stable runtime、test slot 和 remote dev 只能有一个 Gateway control owner。
20. Task evidence 不自动成为 merged/release evidence；promotion 必须比较 tree/context，真机 control
    与 release acceptance 默认在 merged stack candidate 重跑。
21. openpi submodule 的活动开发仍按独立仓 writer lease 管理。只读任务使用 task-local detached
    submodule；修改 openpi 的任务使用隔离 task clone/backend，并必须在提交 AgenticRobot 候选快照前
    把 gitlink 更新到同一 openpi commit。共享可写 submodule 目录和父/子 SHA 不一致都必须拒绝。

## B3. 2026-08-13 基线、2026-08-14 M0 与 2026-08-15 submodule 重采样

本节保留 2026-08-13 的迁移基线，并合并远端主机在 2026-08-14 01:08+08 附近报告的 M0 只读
重采样；详细命令边界与动态进程证据见
[`m0_remote_audit_20260814.md`](./m0_remote_audit_20260814.md)。SHA 仍不能被后续脚本硬编码。

| 对象 | 只读观察/用户确认 | 规划结论 |
| --- | --- | --- |
| 本机 AgenticRobot | `/home/zc/HandMotion/Developing/AgenticRobot`，origin `CYBER-QAQ/AgenticRobot.git`，`main@b7a20748...` | canonical management/integration checkout |
| 本机 remote-tracking | `origin/main@ad1771da...`；活动 main 仍在祖先 `b7a20748...` | 脏活动 checkout 不自动 ff-only；迁移应形成独立候选 commit |
| 本机工作树 | 曾有 70 个误删的 tracked calibration/profile 文件（15756 行），已按 owner 指示从 HEAD 恢复；`teleop_records` 当前无 diff | 该误删阻塞已解除；本规划文档仍为 untracked，纳入管理前需单独 review/commit |
| 本机 RPent | `/home/zc/HandMotion/Developing/RPent`，origin fork + official upstream，`main/origin-main@0e6cb9df...`，盘点时 clean | 可纳入 stack lock |
| workspace 根 | `/home/zc/HandMotion/Developing/.git` 不是有效顶层 Git 仓库 | 不能误当 superproject；task metadata 放外部任务根 |
| 远端目标 AgenticRobot | `/home/user/zc/AgenticRobot` 在 `main/origin-main@ad1771da...`；仅 `.gitmodules` 和 `model/openpi` staged | management clone 正处迁移边界，不能当 clean source/runner |
| 远端 legacy DexTeleop | `new_realman@ec95f04e...`，3955 项 dirty；存在长期数据迁移进程组及多个 shell/editor cwd | 禁止冻结/reset/clean/改 origin/删除；先由进程 owner 确认结束条件和数据收据 |
| 远端 openpi | 原 `/home/user/zc/openpi` 与新 `AgenticRobot/model/openpi` 都在 clean Git commit `0d2d7f4e...`；后者仍有独立 `.git` 且混有 `.venv/assets/logs` | gitlink SHA 正确但 checkout 形态未收口；先保留资产，再重建 clean submodule worktree |
| OpenPI source link | AgenticRobot 与 legacy 的 `deployments/openpi/src` 都指向不存在的 `/home/zc/HandMotion/openpi` | M3 preflight 必须 fail closed；通过 host profile/asset resolver 显式定位，禁止猜路径 |
| 远端设备相关进程 | `DexTeleop_0803` 有运行约 3 天 14 小时的 `hardware.foot_pad --recalibrate` | 非本轮进程；不停止、不接管，设备测试/lease 前先解决 owner 与状态 |
| 远端工具 | AgenticRobot `codex_remote/` 有 SSH wrapper 与 L0-L5 安全规则 | Codex 继续只在本机；远端是受控执行/GPU/硬件 host |

2026-08-16 本机更新不改写上述 point-in-time 远端审计：三个本机稳定 checkout 已 clean/ff-only 到各自
最新上游；框架、RPent/OpenPI `./dev` 和 v2 gitlink 分别位于隔离 task worktrees。远端没有随本机 pull
自动移动，也没有进行 working-tree 文件同步。

当前关系：

```text
本机源码：       CYBER-QAQ/AgenticRobot.git @ main
远端目标：       CYBER-QAQ/AgenticRobot.git @ main/ad1771da + staged model/openpi gitlink
远端 legacy Git：CYBER-QAQ/DexTeleop.git    @ new_realman
legacy 文件：    可能来自另一个 Git 身份的实时源码镜像
```

新 clone 已通过 origin、branch、HEAD、clean 和 import-root 初验。旧 DexTeleop 不能原地升级；M0 也
尚未完成，因为活动迁移/设备相关进程、sync owner 和 OpenPI asset receipt 仍未关闭。

## B4. 三仓所有权与依赖边界

```text
RPent core
    ↑
rpent-dexteleop adapter ── immutable Gateway contract ── AgenticRobot

AgenticRobot OpenPI client ── policy artifact + service wire ── openpi server
             │
             └── owns observation binding, safety and dispatch
```

| 仓库 | Canonical owner | 公开联动边界 | 禁止职责 |
| --- | --- | --- | --- |
| AgenticRobot | motion、composition、Agent Gateway protocol/runtime/transport、sim/real binding、OpenPI client、stack | Gateway schema/bundle、policy artifact、显式 service interface | RPent planning/Memory；OpenPI training implementation |
| RPent | task state、planning/reasoning、Program/Skill orchestration、observation selection、Experience、Memory、upper console | `rpent-dexteleop` 消费固定 Gateway snapshot | safety、IK、device init、dispatch、HOLD 真值；导入 AgenticRobot lower Python package |
| openpi | model training、checkpoint、norm stats、policy server、模型侧 wire | versioned policy artifact、metadata、发布的数据接口/schema | camera/robot/gripper owner、最终限幅/下发；直接启动 AgenticRobot runtime |

### B4.1 源码可获得性与运行能力分离

三个仓库的源码版本属于 Git 事实源；主机只提供 checkout 和运行能力。`RPent 默认在本机`、
`openpi 默认在远端` 是 placement，不是代码所有权或唯一副本：

| Host ID | 默认 checkout | Host-local capabilities | 不由该 host 自动取得的权限 |
| --- | --- | --- | --- |
| `local-dev-control` | AgenticRobot、RPent、目标 source-only openpi | 日常开发、RPent upper、IDE、离线/本地仿真、测试调度 | 远端 GPU/设备 control、远端 task branch 写入权 |
| `remote-gpu-hardware` | AgenticRobot、openpi；必要时可取 RPent commit/artifact | OpenPI runtime、GPU、相机/机械臂所在环境、AgenticRobot lower、设备侧可视化 | 任意 PR 执行、主干写入、未经 lease/审批的 motion |

RPent/openpi 的任务 checkpoint 也必须 push 到各自受保护 remote 或经批准的 task ref，不保留“只在
某台主机上的唯一源码版本”。反过来，clone 到另一台机器不代表它具有设备、secret、数据或控制权。
跨主机测试按 `host capabilities` 路由，不按某个历史目录是否存在来猜测。

### B4.2 依赖边界

现有 OpenPI 训练使用 `PYTHONPATH=$DEXTELEOP_ROOT:$OPENPI_ROOT/src` 是显式过渡依赖。目标是由
AgenticRobot CI 发布小型、版本化、无硬件副作用的 dataset-contract/interface wheel 或 schema
artifact，openpi 锁 exact version；不得永久注入整个 AgenticRobot checkout。

RPent 的 normative 细节仍以 `RPent/docs/architecture/repository_boundaries.md` 为准：Gateway 协议
由 AgenticRobot/DexTeleop 唯一生成；RPent integration snapshot 是 immutable consumer input；
RPent attach-only construction 不启动 Gateway、sim、设备或 robot runtime。

### B4.3 当前项目的三种开发模式

补充建议中的“Y”不能机械地等同于 openpi：当前项目中，OpenPI 提供 GPU 模型运行能力，而相机、
机械臂、safety 和 dispatch 仍属于 AgenticRobot lower。需要远端能力的修改可能落在 openpi，也可能
落在 AgenticRobot device/runtime adapter。

| 模式 | 适用修改 | Worktree/writer | 验证方式 |
| --- | --- | --- | --- |
| Hub 本地开发、Spoke 测试（默认） | 三仓通用逻辑、协议、状态机、配置解析、unit/contract/mock、跨仓改动 | 本机完整 Task Workspace；本机为各仓 writer | 本机 quick/review，再发 resolved clean/draft snapshot 到远端 |
| Hub 远程进入 Spoke 开发（低频） | 厂商 SDK、驱动、实时 timing、GPU/设备反馈需要分钟级闭环 | 整个 Task Workspace 位于远端；创建时选 remote dev host 或整体 handoff | `lab open` 远程打开，push 后回本机 detached review/集成 |
| Spoke 独立 fix/PR | 测试中发现的小型设备或部署问题 | 从被测 snapshot/base 建 `fix/*` 子任务，不接管原 task | 远端 evidence + 本机 review/cherry-pick/PR |

为提高默认模式覆盖率，后续重构应按“纯逻辑/协议 + mock/simulator adapter + GPU/device adapter”分层：

- AgenticRobot 的 Gateway contract、state/action mapping、状态机和 safety-independent parsing 应能在
  本机/CI 测；真实 camera/arm/hand adapter、timing 和 calibration 仅在远端 profile 启用；
- openpi 的 artifact metadata、wire schema、配置解析和 mock policy service 应能在本机/CI 测；
  CUDA/大 checkpoint/GPU serving 留在远端；
- RPent 继续只消费固定 contract，通用 upper/adapter 测试在本机完成；
- 真机配置、校准、密钥、设备枚举和大型数据始终留在远端外部 config/data roots。

具体包路径要先依据现有代码边界盘点后再定，不能为了套用示例目录批量移动模块。

## B5. 当前项目物理路径

### B5.1 本机

本机是 Development Hub，三个 source repositories 都应可取得；不能运行远端真机/GPU profile 不影响
源码开发和 mock/contract 测试：

```text
/home/zc/HandMotion/
├── Developing/
│   ├── AgenticRobot/                 # canonical management/integration checkout
│   ├── RPent/                        # canonical management/integration checkout
│   └── openpi/                       # 迁移源/object cache：非权威独立 fork clone，无 checkpoint/assets
│
├── Worktrees/
│   ├── AR-0142/
│   │   ├── AgenticRobot/             # 含 task-local model/openpi child worktree/gitlink
│   │   ├── RPent/
│   │   ├── AR-0142.code-workspace
│   │   ├── task.env
│   │   └── task.yaml
│   └── _review/AR-0142/<snapshot-id>/
│
├── Runtime/
│   ├── tasks/AR-0142/
│   ├── reviews/AR-0142/<snapshot-id>/
│   ├── releases/<stack-tag>/
│   └── current -> releases/<stack-tag>/
│
├── Config/host.local.yaml
├── SharedCache/
└── State/
    ├── namespaces/
    ├── resource-leases/
    ├── test-requests/
    ├── test-results/
    ├── task-receipts/
    └── deploy-receipts/
```

保留当前 `Developing/AgenticRobot` 和 `Developing/RPent`。普通 Agent 在 `Worktrees/<Task ID>`
开发。`Developing/AgenticRobot/model/openpi` 是 openpi 的默认 management checkout；原 sibling
source-only clone 只可继续作为本地 object source/cache，不能成为第二个权威版本。权威集成版本由
AgenticRobot gitlink 固定。默认 openpi 修改仍走本地任务分支/PR，
而不是从本机通过 SSH 修改远端 working tree。

OpenPI 自带 `aloha/libero` submodule，因此形成两层 submodule。clean snapshot 从 parent commit tree
读取 exact gitlinks，不要求只做代码审视时先初始化全部 child；执行槽再从已有本地 objects 以
`--no-fetch` materialize。AgenticRobot+openpi writable Task 会把 openpi 自己的 branch/worktree 放在
父 Task 的 `model/openpi`，`lab open` 将它作为 multi-root folder 展示；多个 Task 不共享可写 child checkout。

### B5.2 远端 192.168.70.116

远端是 Specialized Spoke。它常态运行 detached tests/runtime，仅为少量硬件耦合任务创建 writable
worktree：

```text
/home/user/zc/
├── AgenticRobot/                      # 独立 management clone；staged model/openpi gitlink 尚未提交
├── openpi/                            # 当前 source + env/assets/checkpoints/logs
├── DexTeleop/                         # legacy synced checkout；冻结/审计/保留
│
└── robot-stack/
    ├── git-cache/AgenticRobot.git/    # 可选 bare LAN cache
    ├── dev/openpi/                    # 可选受信 clean test anchor；不是第二个 integration owner
    ├── control-plane/
    │   ├── releases/<lab-version>/    # protected/tagged lab + validator/executor/policy
    │   └── current -> releases/<lab-version>/
    ├── tasks/
    │   └── AR-0201-remote-dev/         # 低频 writable；非测试槽/canonical clone
    │       ├── AgenticRobot/           # 任务涉及才存在
    │       ├── openpi/                 # 硬件/GPU 强耦合开发 worktree
    │       ├── task.yaml               # writer handoff/branch/scope
    │       └── AR-0201.code-workspace
    ├── test-slots/
    │   ├── gpu-slot-01/
    │   │   ├── AgenticRobot/           # detached @ request SHA
    │   │   ├── openpi/                 # detached @ request SHA
    │   │   └── slot.lock.yaml
    │   └── device-slot-01/              # 更高权限，和 GPU-only slot 分开
    ├── envs/test-slots/                 # slot + dependency-lock-hash 环境
    ├── reviews/AR-0142/<snapshot-id>/   # 一次性/需长期保存的正式快照
    ├── runs/AR-0142/<run-id>/         # manifests/log/video/trace
    ├── runtime/
    │   ├── releases/<stack-tag>/AgenticRobot/
    │   ├── releases/<stack-tag>/release-receipt.json
    │   └── current -> releases/<stack-tag>/
    ├── config/host.yaml
    ├── state/
    │   ├── sync/
    │   ├── deploy/
    │   ├── test-requests/
    │   ├── test-results/
    │   ├── service-receipts/
    │   ├── authority-receipts/
    │   └── resource-leases/
    ├── artifacts/                     # 显式发布/拉取，不与源码双向同步
    ├── data/
    │   ├── openpi-assets/
    │   ├── checkpoints/
    │   └── datasets/
    └── logs/
```

当前 clone 不删除重建、不并发第二个 Git 操作。旧 DexTeleop 与 openpi 过渡目录继续保留。OpenPI
大数据不在 M0-M3 首个纵向切片中直接移动；先定义配置键、独立备份收据和引用验证，再拆 source/data。

`tasks/`、`test-slots/`、`reviews/`、`.git`、env、build、`runs/` 和 `runtime/` 全部属于远端本地
状态，绝不由 Syncthing/Dropbox/实时 rsync 与本机对应目录镜像。`test-requests/`/`test-results/` 是受控执行器
的本地 inbox/outbox；内容通过 CI API、任务服务或 `codex_remote` 受控命令传输，不依赖共享目录
覆盖。跨机 artifact 只按 run ID 显式上传/下载。

## B6. AgenticRobot 管理面与项目命令实例

目标仓库结构按实际调用者分批实现：

```text
AgenticRobot/
├── stack/
│   ├── components.lock.yaml
│   ├── compatibility.yaml
│   ├── machines.yaml
│   ├── impact-rules.json                 # 已实现的 JSON-compatible protected rules
│   ├── hosts/
│   │   ├── local.example.yaml
│   │   └── remote-gpu-hardware.example.yaml
│   ├── machines.schema.json
│   ├── task.schema.json
│   ├── snapshot.schema.json
│   ├── draft-snapshot.schema.json
│   ├── effective-workspace.schema.json
│   ├── test-profile.schema.json
│   ├── test-request.schema.json
│   ├── test-result.schema.json
│   ├── dev-run.schema.json
│   ├── resource-lease.schema.json
│   └── release.schema.json
├── integration/
│   ├── profiles/                       # 跨仓/跨主机声明式场景
│   ├── scenarios/                      # 主项目拥有的组合逻辑
│   └── runners/                        # command_id 的受审实现
├── lab                                  # 唯一日常用户入口
├── tools/
│   ├── projectctl
│   ├── wt
│   ├── projectctl_lib/
│   └── wt_lib/
├── dev
├── .github/workflows/
└── docs/architecture/
```

现有 `control_station/agent_gateway/protocol/` 仍是协议唯一源码 owner；`stack/` 只锁发布 artifact/
hash，不能复制第二套 schema。RPent 与本地 openpi 各提供相同顶层动词的轻量 `./dev` adapter，
实际命令服从各自 AGENTS/README/lock；AgenticRobot `tools/wt` 只做任务组合。

本项目日常用户操作面：

```text
lab change create CHG-0042
lab inspect CHG-0042
lab task create AR-0142 --repos AgenticRobot,openpi
lab open AR-0142
lab open --run <run-id>
lab inspect AR-0142 [--diff]
lab inspect AR-0142 --against-run <run-id>
lab test AR-0142 --recommended
lab test AR-0142 agenticrobot-openpi-gpu
lab test AR-0142 openpi-rollout-viz --interactive
lab verify AR-0142 --recommended
lab session AR-0142 agenticrobot-openpi-gpu
lab runs AR-0142
lab runs open <run-id>
lab runs watch <run-id>
lab runs cancel <run-id>

# 仅硬件强耦合任务显式选择远端 development host
lab task create AR-0201 --repos AgenticRobot,openpi \
  --dev-host remote-gpu-hardware
```

跨主机常规并行按 Change 拆任务，例如本机 `AR-0142-main`、远端 `OP-0143-remote` 和本机
`INT-0144` integration task 都属于 `CHG-0042`；不把一个 Task 的 writable Workspace 横跨两机。

`lab inspect AR-0142` 的默认摘要至少包含：development host/status、各仓 branch/revision/dirty、
baseline 或 task-override 来源、综合 changed files、impact rules 推荐/跳过的 Profiles、每个 Profile
相对最新 snapshot 的通过状态，以及最近远端 Run 的 host/result/artifact links。还应给出一条明确的
Next action、静态 Profile->Host readiness（slot、asset、service、device 占用）和 evidence freshness；
例如 openpi tree 在上次 GPU Run 后变化则标记 stale。首版只做静态映射+验证，不构建通用最优调度器。
`--diff` 展开各仓 diff，`--run` 打开某次 Run；用户无需分别进入本地 worktree、远端 slot 和日志目录。

以下是 `lab` 调用的内部/admin interfaces，保留用于实现和故障诊断，不作为日常操作清单：

```text
# task/worktree
tools/wt create AR-0142 --repo AgenticRobot --repo openpi
tools/wt env create AR-0142
tools/wt open AR-0142
tools/wt diff AR-0142
tools/wt test AR-0142 --repo AgenticRobot unit agent-gateway
tools/wt checkpoint AR-0142 --publish
tools/wt review AR-0142
tools/wt viz AR-0142 --snapshot 7f31c2d9 rollout --episode 17
tools/wt device run AR-0142 camera-front --mode observe -- \
  ./dev device camera-front observe

# current checkout
./dev doctor
./dev test unit <registered-scope>
./dev test integration <registered-scope>
./dev viz <registered-mode> ...
./dev sim <registered-profile> ...
./dev device <logical-device> observe

# stack/host/release
projectctl status [--json]
projectctl stack verify --ref <tag-or-commit>
projectctl sync-main --mode fetch-only
projectctl sync-main --mode advance-anchor --ff-only
projectctl test submit AR-0142 --target remote-gpu-hardware \
  --snapshot 7f31c2d9 --profile agenticrobot-openpi-gpu
projectctl test status <request-or-run-id>
projectctl test logs <run-id>
projectctl test open <run-id>
projectctl task handoff AR-0201 \
  --from local-dev-control --to remote-gpu-hardware
projectctl remote open AR-0201 --host remote-gpu-hardware
projectctl tunnel gateway --task AR-0142
projectctl deploy --ref stack-v0.2.0 --dry-run
projectctl rollback --ref stack-v0.1.0
```

跨仓 integration test 从 task/snapshot manifest 显式设置 `LAB_ROOT`、`RPENT_ROOT`、
`OPENPI_ROOT` 为同一 snapshot 目录。禁止从父目录、PATH 或 user site “碰巧找到”另一仓库。

AgenticRobot `./dev` 只能包装现有 README/AGENTS canonical entrypoints，不新增第二套设备/safety
路径；RPent core tests 与 `integrations/dexteleop/tests` 必须显式分别运行；openpi 普通 CI 不加载大
checkpoint。

项目 task manifest 还需显式记录各仓 writer host 和测试能力，示意：

```yaml
schema: agenticrobot.task.v1
task_id: AR-0142
change_id: CHG-0042
baseline_ref: stack-v0.8.0
development_host: local-dev-control
depends_on: []
consumes_snapshots: []
repositories:
  AgenticRobot:
    branch: task/AR-0142-main
    base_sha: "<40-hex>"
    writer_host: local-dev-control
    writer_actor: agent-main-01
    writer_epoch: 1
  openpi:
    branch: task/AR-0142-openpi
    base_sha: "<40-hex>"
    writer_host: local-dev-control
    writer_actor: agent-openpi-01
    writer_epoch: 1

tests:
  - name: contract-offline
    target: any-hermetic
  - name: main-with-openpi-gpu
    target: remote-gpu-hardware
    requires: [openpi-runtime, gpu]
    profile: agenticrobot-openpi-gpu
```

一个仓库没有改动时仍应在 snapshot 中固定其 consumer commit，但可以省略 task branch/writer。
上例只为 AgenticRobot/openpi 创建 writable worktrees；RPent 由 `stack-v0.8.0` 的 lock 自动补齐，
只有 Profile 需要时才 checkout。当前默认仍是三仓主要开发和 writer 位于本机，远端只创建
detached test worktrees。

writer 仍按“仓库 × 任务”记录 actor/epoch，但同一 Task 的 `writer_host` 必须一致。硬件耦合任务
AR-0201 若从本机迁往远端，`projectctl task handoff` 一次性交接完整 AgenticRobot/openpi Workspace；
不能只把 openpi 留在远端、同时继续在本机编辑同任务 AgenticRobot。`projectctl remote open` 只是
打开这一已登记 workspace，不创建 branch、不越权申请设备。当前
`codex_remote` 模式仍是“Codex 留在本机、远端受控执行”，所以该路径只为以后受授权的人或 Agent
预留，不改变当前远端操作规则。

AgenticRobot 仓库保存 `lab`/Profile/executor 的源码，但远端高权限 Run 使用
`robot-stack/control-plane/current` 中从受保护 tag 安装的版本。Task snapshot 内修改后的 `lab`、
`integration/profiles`、`runners` 或 `./dev` 都属于被测试 payload：未合并版本只能 hermetic/mock
自测，不能改变 protected policy、command allowlist、network、GPU/device mode。设备操作最终进入
AgenticRobot 受信 Gateway，而不是把 Task Python 进程加入 device group。

远端 runner 的 `prepare -> acquire -> run -> publish -> cleanup` 是受控 worker 状态机，而不是另一套
面向 Agent 的 `tools/lab-runner` 公共接口。每一步校验 request/slot/snapshot/lease，控制面仍由本机
`projectctl test ...` 提交和查询。

首版不需要常驻调度服务：一个本机 Python/薄 shell `lab` CLI、YAML/JSON schemas、原子 receipt/
lock files、现有 `codex_remote` 受控 SSH 和按 Run ID 的 artifact 拉取即可。只有并发队列、跨用户
lease 或审计规模超过单机文件状态能力时，才升级数据库/服务；用户命令和 schemas 保持不变。

## B7. Stack lock、host profile 与 compatibility

`stack/components.lock.yaml` 示例：

```yaml
schema: agenticrobot.stack-lock.v1

control_plane:
  lab_version: lab-v0.1.3
  source_commit: "<protected-40-hex>"
  profile_policy_sha256: "<64-hex>"
  executor_bundle_sha256: "<64-hex>"

contracts:
  agent_skill:
    producer_path: control_station/agent_gateway/protocol/agent_skill_v1.schema.json
    schema_id: dexteleop.agent_skill.v1
    sha256: "<64-hex>"
  policy_artifact:
    schema: dexteleop_deploy_v2

components:
  rpent:
    repository: https://github.com/CYBER-QAQ/RPent.git
    commit: "<40-hex-merged-commit>"
    distribution: rpent-dexteleop
    contract_snapshot_path: >-
      integrations/dexteleop/src/rpent_dexteleop/gateway/contracts_snapshot/agent_skill_v1.json
    placements: [local-control-plane]

  openpi:
    repository: https://github.com/CYBER-QAQ/Openpi.git
    upstream: https://github.com/Physical-Intelligence/openpi.git
    commit: "<40-hex-merged-commit>"
    pin_source:
      type: submodule_gitlink
      parent_repository: AgenticRobot
      path: model/openpi
    placements: [remote-gpu-hardware]
    execution_mode: localhost-service

profile_assets:
  agenticrobot-openpi-gpu:
    profile_path: integration/profiles/agenticrobot-openpi-gpu.yaml
    required_asset_ids: [openpi-checkpoint-policy-v8]
  agenticrobot-device-observe:
    profile_path: integration/profiles/agenticrobot-device-observe.yaml
    required_asset_ids: [rm65-calibration-v5, camera-intrinsics-v2]
```

- URL 不依赖某台机器 SSH alias；认证由 host config 处理。
- `components.openpi.commit` 是便于审计和兼容矩阵读取的镜像字段；resolver 必须证明它等于
  AgenticRobot tree 中 `model/openpi` gitlink。两者不一致时禁止生成 snapshot、Run 或 release。
- 只接受已合并且可取得的完整 commits；tag 可作注释，commit 是权威。
- contract 同时校验 schema/version 与 canonical bytes/semantic hash。
- lock 由 Integration Owner 在独立 stack PR 更新。
- `compatibility.yaml` 表达支持窗口和弃用，不能覆盖精确 lock。

Baseline ID 解析为“AgenticRobot release tag/commit + 该 commit 内的 components lock/profile
definitions”；仍不在 lock 内自引用 AgenticRobot commit。以 AR-0142 为例：

```text
AgenticRobot = AR-0142 task override
openpi       = AR-0142 task override
RPent        = stack-v0.8.0 embedded lock
profiles     = AgenticRobot baseline definitions，Run 时记录实际 profile hash
```

`lab inspect` 和 runner 使用同一 resolver；执行前把这个符号表达式展开为完整 commits/hashes。

可提交的 `stack/machines.yaml` 只表达逻辑主机和非敏感能力，不写密码、私钥、设备地址、校准或
绝对数据路径：

```yaml
schema: agenticrobot.machines.v1
machines:
  local-dev-control:
    roles: [development-hub, integration-owner, test-orchestrator]
    runner_labels: [local-dev, rpent-upper, simulator]
    capabilities: [development, rpent-upper, offline-test, local-sim]
    default_writer_for: [AgenticRobot, RPent, openpi]
  remote-gpu-hardware:
    roles: [specialized-spoke, test-runner, execution-node]
    ssh_target: remote-gpu-hardware
    runner_group: protected-hardware
    runner_labels: [remote-gpu, openpi-runtime, device-host]
    capabilities: [gpu, openpi-runtime, agenticrobot-lower, device-observe]
    default_writer_for: []
    test_slots: [gpu-slot-01, device-slot-01]
```

labels/capabilities 只用于调度，不能作为授权证明；执行器仍须校验 request policy、protected ref、
snapshot、profile、lease 和人工审批。实际 SSH alias 的解析以及是否允许 control 由 host-local
配置和 `codex_remote` 安全层决定。

实际 host profile 不入 Git；只提交 example/schema：

```yaml
host_id: remote-gpu-hardware
roles: [agenticrobot-runtime, openpi-server, gpu, hardware]

paths:
  management_clone: /home/user/zc/AgenticRobot
  remote_dev_root: /home/user/zc/robot-stack/tasks
  test_slots_root: /home/user/zc/robot-stack/test-slots
  test_envs_root: /home/user/zc/robot-stack/envs/test-slots
  runs_root: /home/user/zc/robot-stack/runs
  runtime_root: /home/user/zc/robot-stack/runtime/current
  data_root: /home/user/zc/robot-stack/data

capabilities:
  - openpi-runtime
  - agenticrobot-lower
  - device-observe

devices:
  rm65-arm:
    gateway_id: rm65-primary
    modes: [observe, control]

assets:
  rm65-calibration-v5:
    path: /home/user/zc/robot-stack/config/calibration/rm65.yaml
    sha256: "<64-hex>"
  camera-intrinsics-v2:
    path: /home/user/zc/robot-stack/config/calibration/camera.json
    sha256: "<64-hex>"
  openpi-checkpoint-policy-v8:
    path: /home/user/zc/robot-stack/data/checkpoints/<artifact-id>
    sha256: "<64-hex>"

clock:
  require_sync: true
  max_offset_ms: 5

components:
  agenticrobot: {enabled: true}
  rpent: {enabled: false}
  openpi:
    enabled: true
    stable_service_id: openpi-stable
    endpoint_from_service_receipt: true
    reject_unmanaged_endpoint: true

network:
  expose_agent_gateway_to_lan: false
  require_ssh_forward: true

remote_development:
  enabled_by_default: false
  require_writer_handoff: true
```

公共 Python 代码不能用 `os.path.exists('/home/...')` 猜组件布局。

## B8. 自动同步、release 与远端激活

### B8.1 Source ref synchronization 与 management anchor

clone 成功并验收后，`/home/user/zc/AgenticRobot` 是远端独立管理 clone。两台主机“主项目一致”
定义为某个同步点都能解析同一 `origin/main` commit，而不是 working tree 的每个文件实时一致：

```text
task worktree -> commit -> task branch -> PR/review/CI/merge queue
  -> protected origin/main
  -> remote read-only poller fetch refs
  -> verify expected origin/protected ref
  -> sync receipt

clean checkpoint -> publish task ref + exact commit/tree
synthetic draft -> publish temporary snapshot ref/bundle + exact commit/tree
  -> remote executor fetch/verify exact Git objects
  -> local detached test worktree
```

建议未来 user-level timer 每 60 秒调用 `projectctl sync-main --mode fetch-only`，使用只读 deploy key。
`fetch-only` 实现必须：

- 固定 path、expected origin、refs 和 host identity，只更新 remote-tracking refs/Git objects；
- origin mismatch、fetch 非预期 ref、签名/CI policy 不合格即停止；
- 不 checkout/pull/rebase/reset canonical anchor，不触碰任何 task/review worktree；
- 不执行 reset/clean/stash/force checkout、dependency install、restart 或 hooks；
- 记录 old/new SHA、时间、host、结果和失败原因；
- GitHub 不可达时保持上一 SHA。

稳定 baseline checkout 可以在 fetch 之后执行独立的 `advance-anchor --ff-only`。远端
`/home/user/zc/AgenticRobot` 在确认它永久只作 clean management anchor 后，可以由 timer 条件自动
前移；实现必须取得 management lock，验证 clean、expected origin/branch、old HEAD 与上一 receipt、
目标为受保护 main 的后代、无进行中的 Git/run，并禁用 repo hooks。任一条件失败只保留新 refs，
不改变 checkout。

本机 70 个误删文件已按 owner 指示从 HEAD 恢复，`teleop_records` 当前无 diff；但本规划文档仍为
untracked，且 canonical branch/remote-tracking 尚待验收，因此本机 `advance-anchor` 继续关闭，直至
管理 checkout 完全 clean 且身份检查通过。任何 Agent task/review/test slot 都不自动前移。远程测试
和部署也不能依赖 management checkout 的当前 HEAD，只依赖请求/lock 中的 exact commit/tree。

LAN cache 必须是 bare repo；禁止 push 到 checked-out branch 或同步 `.git`。当主机无法访问共同
Git remote 时，可以把经过 `git bundle verify`、hash 和来源校验的 Git bundle 作为离线应急传输，
但它不是日常文件同步机制。

### B8.2 Stack release

正式 runtime 不跟随每次 main merge。stack PR 通过后创建 `stack-vX.Y.Z` annotated tag，并生成：

```json
{
  "schema": "agenticrobot.stack-release.v1",
  "stack_tag": "stack-v0.2.0",
  "agenticrobot_commit": "<40-hex>",
  "rpent_commit": "<40-hex>",
  "openpi_commit": "<40-hex>",
  "contract_hashes": {"agent_skill": "<64-hex>"},
  "ci_run": "<immutable-run-reference>"
}
```

远端在新 staging/release 目录验证后原子切 `runtime/current`；失败保留上一 runtime。rollback 只切回
已验证 release，不在运行目录热修、pull 或 reset，也不自动回滚 dataset/checkpoint/device state。

### B8.3 Activation 分级

| 层级 | 自动化 | 允许行为 |
| --- | --- | --- |
| source ref sync | 自动 | fetch remote refs、receipt；不改变 checkout、不运行代码 |
| management anchor advance | 专用 clean anchor 可条件自动 | 锁内 ff-only + protected ref + receipt；失败只更新 refs |
| R0/R1 release validation | 可自动 | schema/import/unit/offline/sim smoke；无设备 |
| OpenPI server | 受控 deployment job | GPU/port；不连接机械臂，显式健康检查/回退 |
| H1/device observe | 人工批准 | 只读设备，占用须 lease |
| H2/L4 | 当前操作明确授权 | 修改设备/host 状态 |
| M/L5 | 当前动作明确授权 + fresh safety receipt | motion/servo/trajectory/force control |

source sync、release 到达和 server start 都不授权真机。

## B9. 跨主机 runtime 与资源 lease

### B9.1 Effective Workspace snapshot remote test 协议

用户只运行 `lab test/verify <Task> <Profile>`。resolver 自动用 baseline + task overrides 生成
Effective Workspace；`test` 可生成 draft snapshot，`verify` 必须生成 clean-commit snapshot：

```text
Task + baseline + overrides + Profile
  -> immutable snapshot (clean commits OR synthetic draft commits)
  -> capability/asset scheduler
  -> remote validates/fetches Git objects
  -> detached AgenticRobot + openpi test slot at exact commit/tree OIDs
  -> doctor -> namespace/lease -> registered test
  -> run manifest + artifacts + status returned by run ID
```

`test-request.yaml` 的最低内容：

```yaml
schema: agenticrobot.test-request.v1
request_id: AR-0142-20260813-001
idempotency_key: "<request-content-sha256>"
task_id: AR-0142
change_id: CHG-0042
snapshot_id: "<snapshot-id>"
snapshot_type: clean-commit
baseline_ref: stack-v0.8.0
repositories:
  AgenticRobot: {commit: "<40-hex>", tree_oid: "<oid>", source: task-override}
  RPent: {commit: "<40-hex>", tree_oid: "<oid>", source: baseline}
  openpi: {commit: "<40-hex>", tree_oid: "<oid>", source: task-override}
control_plane:
  version: lab-v0.1.3
  commit: "<protected-40-hex>"
  policy_sha256: "<64-hex>"
target:
  host: remote-gpu-hardware
  requires: [openpi-runtime, gpu]
execution:
  profile: agenticrobot-openpi-gpu
  registered_command: integration-openpi
  resource_mode: gpu-only
policy:
  dirty: false
  allow_motion: false
assets:
  openpi-checkpoint-policy-v8: "<64-hex>"
```

远端执行器只接受 schema 中注册的 profile/command，不接受请求中携带的任意 shell。它必须校验
clean commit 来自预期 remote/ref policy；draft 则校验 synthetic commit/tree OID、parent、scope、
allowlisted untracked manifest 和临时 ref/bundle receipt。在远端本地直接 detached checkout 并准备
独立 environment，完成
`./dev doctor` 后才运行；结果身份是 Effective Workspace + snapshot/profile/assets/host/config/
resource receipt，而不是笼统的“AR-0142 已通过”。新 snapshot 不移动正在执行的旧 Run。

远端如发现 AgenticRobot 问题，默认只返回 evidence；若明确授权修复，则从被测 SHA 新建
`fix/AR-0142-remote-<topic>` 并单独 PR/cherry-pick，不能直接 push 本机 writer 持有的
`task/AR-0142-main`。writer host 的正式移交按 A4 执行。

draft remote test 纳入开发便利性里程碑，但分级开放：第一批只启用 offline/sim/GPU-only；H1
observe-only 必须由特定 Profile 显式允许并人工确认，不能发送 motion。`lab verify`、required check、
release、device control 和任何运动测试仍只接受 clean commits。

### B9.2 远端固定测试槽位

远端初期设置两个长期目录即可，不为每轮 checkpoint 无限创建 worktree：

| Slot | 允许 profile | 资源边界 |
| --- | --- | --- |
| `gpu-slot-01` | contract、OpenPI server、GPU-only integration/viz | GPU lease；账号不在 device group，不连接机械臂 |
| `device-slot-01` | H1 observe；未来经批准的 control acceptance | 更高权限独占 lease、safety/现场门禁；不接受普通 PR |

slot 状态为 `FREE -> PREPARING -> RUNNING -> PUBLISHING -> FREE`；异常进入 `QUARANTINED`，不能用
reset/clean/pkill 猜测恢复。`prepare` 在 slot lock 内停止本 slot 登记进程、验证两仓 detached/clean、
切换到请求中的 AgenticRobot/openpi clean/synthetic commit 并核对 tree OID。之后选择
dependency-lock-hash 对应 env，再运行 provenance doctor。RPent 若不在该测试的数据面，只验证/
记录其 SHA，不从环境中随机导入它。

env、wheel/build cache、checkpoint、dataset、SDK 和 runs 位于 slot 之外。环境可以跨请求复用，但
只在 lock hash、Python/CUDA/host profile 都匹配且 doctor 通过时复用；源码 editable 指向必须随
slot 当前 checkout 更新。未知 untracked/generated 文件会让 slot quarantine，worker 不执行宽泛
`git clean -xfd`。需要长期保留或并行运行的正式测试仍创建一次性 review snapshot。

### B9.3 当前组合的测试位置

| 变更/组合 | 默认位置 | 约束 |
| --- | --- | --- |
| AgenticRobot 通用单元/静态检查 | hosted hermetic CI 或本机 | 不需要设备/GPU |
| AgenticRobot + RPent contract/sim | 本机 `local-dev-control` | 两仓 exact snapshot；RPent attach-only |
| openpi metadata/wire/config/unit | hosted hermetic CI 或本机 | source-only clone；小 fixture/mock，不加载大 checkpoint |
| AgenticRobot + openpi mock | 本机 `local-dev-control` | 两仓 exact snapshot；mock policy service，无 CUDA/设备 |
| AgenticRobot + openpi GPU | 远端 `remote-gpu-hardware` | 两仓 detached clean/draft snapshot；GPU lease；默认无设备 |
| openpi 模型/服务 | 远端或隔离 GPU runner | checkpoint/artifact metadata 固定，不启动设备 runtime |
| 三仓静态兼容 | hosted hermetic CI | checkout lock 中全部 commits |
| AgenticRobot 跨主机基线回归 | 本机 + 远端 | 同一 exact clean commit/tree 分别运行 host-appropriate profile，结果不可互相替代 |
| RPent upper + AgenticRobot lower + openpi | 本机/远端分布式 | lower/control loop 留在远端；跨机只走版本化 Gateway 高层命令、状态、日志和测试控制 |
| 相机/机械臂 observe/control | 远端设备 host | clean release/snapshot + 权威 lease；control 另需现场和当前动作授权 |

如果某测试只是因为历史习惯才“缺仓库”，优先在目标机 fetch 源码；如果组件可稳定发布，则使用带
SHA/hash 的 wheel、image 或 bundle；只有组件本来就是网络服务边界时才做分布式测试。实时机器人
数据面和控制环不能为了凑齐三仓而随意跨机迁移。

AgenticRobot 真机 Profile 还必须固定 physical cell、robot/hand logical serial、controller/firmware、
calibration/camera assets、初始关节与 gripper 状态、场景摆放、operator 和 setup/reset/home receipt。
每次 retry/recovery 是新 Attempt。统计性 Profile 明确 trials、minimum successes、maximum safety
aborts 和 reset protocol；`FAILED_ASSERTION`、`BLOCKED_PRECONDITION`、`ABORTED_SAFETY`、
`INFRASTRUCTURE_FAILURE` 等不能合并成一个失败计数。

### B9.4 分布式 Profile 编排

RPent upper 在本机、AgenticRobot lower + openpi 在远端的全链路测试由主项目 Profile 描述，用户不
手工打开多组 SSH/tmux：

```yaml
profile_id: rpent-agenticrobot-openpi-e2e
nodes:
  upper:
    requires: [rpent-upper]
    repositories: [RPent]
    command_id: rpent.gateway-client
  lower-policy:
    requires: [agenticrobot-lower, openpi-runtime, gpu]
    repositories: [AgenticRobot, openpi]
    command_id: integration.lower-policy-e2e
dependencies:
  - lower-policy:healthy-before:upper
artifacts:
  - upper/logs
  - lower-policy/logs
  - combined-trace.json
```

`lab test AR-0142 rpent-agenticrobot-openpi-e2e` 解析同一个 Effective Workspace/Run ID，按稳定顺序
取得 host/资源 leases，准备两个节点，先启动 lower 并通过健康检查，再启动 upper；结束或失败时按
反序 teardown 并汇总证据。preflight 必须验证两机时钟同步并记录最大 offset、单调时钟起点、网络
RTT、各节点启动时间和同一 Run ID，否则跨机 trace 标记不可可靠对齐。

实时控制环、设备 watchdog、HOLD 和安全停止留在远端设备机。跨机只传版本化高层命令、状态、
测试控制和日志；网络断开时不能依赖本机 Hub 发送最后一条停止命令。

### B9.5 Remote child Task 与低频 whole-Task handoff

常规跨机并行使用同一 Change 下的 remote child Task：例如本机 `AR-0142-main` 修改 AgenticRobot，
远端 `OP-0143-remote` 只修改 openpi/GPU 专有代码并消费 AR-0142 的 immutable contract snapshot，
最后本机 `INT-0144` 消费两者 snapshots/merged commits。这样两机可以并行，同时每个 Task 仍只有
一个 development host 和完整 Workspace。

只有同一 Task 的多个仓库必须在设备反馈中紧密联调、无法用 contract/snapshot 解耦时，才整体
handoff 为 remote-hosted development：

1. 本机创建任务/scope/base，并确认远端 clone 已完成审计、目标 repo/branch clean 且没有其他
   writer；若涉及 openpi，须先按需解耦 source/data，当前混合目录在完整审计前不能直接充当
   remote-dev anchor；
2. 若任务原在本机，全部 writer checkpoint/push 后，内部 `projectctl task handoff` 原子更新 Task 的
   development host 和各仓 writer receipts；也可在创建时直接指定远端 dev host；
3. 远端从 exact clean commits 创建 `tasks/<Task ID>-remote-dev/` 完整 multi-root writable Workspace，绝不
   复用 `test-slots/`、`/home/user/zc/AgenticRobot` 或 runtime；
4. 用户仍只执行 `lab open <Task>`；工具内部 Remote SSH 打开登记的 workspace/tmux。设备运行另走
   Profile/lease/safety/现场授权；打开编辑器不等于取得设备权限；
5. 远端完成 quick test/checkpoint/push，停止进程、释放 lease，再 handoff 回本机；
6. 本机创建 detached review，运行 Hub 侧 unit/contract/mock 并通过 PR 集成。

若只是测试中发现小问题，优先从被测 snapshot/base 创建 `fix/<task>-remote-<topic>` 子任务，不
handoff 原任务。
远端开发产生的提交仍是候选修改；本机 Integration Owner 保持 main、stack lock 和最终合并权。

### B9.6 远端可视化与 artifact

Web 可视化、TensorBoard、Gradio、Jupyter 或 Foxglove 必须监听远端 localhost 和任务独立端口。
用户运行 `lab test <Task> <viz-profile> --interactive` 或 `lab runs open <run-id>`；内部通过
`projectctl`/`codex_remote` 受控层建立端口转发，并显示 Task、Effective Workspace、
host、profile、device、dirty 和 run directory。原生 RViz/MuJoCo/厂商 UI 放在远端持久桌面或登记的
task tmux/session；本机只远程查看。日志、视频和 trace 按 run ID 显式回传，不能持续镜像整个
`runs/` 目录。

### B9.7 运行服务身份与 runtime authority

固定 `127.0.0.1:8078` 只能是受管理 stable service 的逻辑入口，不能作为测试身份。GPU/test slot
默认由当前 Run 启动 ephemeral OpenPI service，分配独立 localhost port/socket 并生成 ServiceReceipt：

```yaml
service_id: openpi-R-20260813-044
run_id: R-20260813-044
mode: ephemeral
source: {commit: "<openpi-commit>", tree_oid: "<tree-oid>"}
model:
  checkpoint_sha256: "<sha256>"
  config_sha256: "<sha256>"
  norm_stats_sha256: "<sha256>"
runtime:
  environment_sha256: "<sha256>"
  cuda_driver: "<version>"
  gpu_uuid: "<logical-id>"
  endpoint: http://127.0.0.1:18427
  pid: 12345
```

如果 Profile 使用 `attach-existing` stable OpenPI/Gateway，Ready Barrier 必须取得受管理 receipt，并
逐项匹配 Run 请求的 commit/tree/checkpoint/config/norm stats/contract/authority；只有端口健康或
tmux 中存在进程一律拒绝。legacy DexTeleop、用户手工服务和无 receipt 进程为 `unmanaged`。

AgenticRobot Gateway receipt 还必须包含 Run ID、AgenticRobot commit/tree、Gateway contract hash、
authority mode 和 runtime owner。稳定 runtime、test slot、remote dev session 不能同时成为机械臂
control owner；切换由 Gateway 在本机设备侧完成 authority handoff，文件锁/GPU lease/SSH 连接均不能
代替它。

### B9.8 分布式 runtime 与资源 lease

默认 GPU、OpenPI server 和 AgenticRobot lower runtime 在远端，RPent upper runtime 可在本机。
AgenticRobot Gateway 现有 authenticated Unix transport 优先通过 SSH StreamLocalForward：

```text
local RPent
  -> local forwarded Unix socket
  -> authenticated SSH tunnel
  -> remote AgenticRobot Gateway Unix socket
  -> remote lower runtime
  -> ServiceReceipt-matched localhost OpenPI endpoint
```

未来 tunnel 必须进入 `codex_remote` 受控 wrapper 或 `projectctl` 安全公共层；普通 Agent 不手写
任意 SSH 控制命令。tunnel 不改变 RPent attach-only、Gateway admission、HOLD、safety、dispatch
所有权。

本机与远端不能用各自 `flock` 协调同一硬件。GPU/设备权威 lease 位于远端资源 host 或中央
service；receipt 额外包含 `192.168.70.116`、全部仓 SHA、snapshot、mode、TTL 和 heartbeat。

项目模式映射：

| 通用模式 | 当前项目边界 |
| --- | --- |
| observe | 通过 `codex_remote` L1/H1 wrapper 或下层 read-only Gateway；禁止 control/mode change |
| control | clean reviewed snapshot/release + exclusive lease + fresh `remote_safety_check` + 当前 turn/现场授权 |
| lease loss | 停新请求；AgenticRobot lower owner 决定 HOLD/安全状态，RPent/框架不声称已停稳 |

设备使用 host profile 的逻辑 ID，不直接依赖 `/dev/video2`、USB 枚举或散落 arm IP。

## B10. 三仓联动变更与测试

### B10.1 跨仓 contract 流

使用 **Expand -> Migrate -> Contract**：

1. AgenticRobot 添加向后兼容 contract 版本/字段，保留旧版并生成固定 bundle。
2. 同一 Change ID 创建 RPent `rpent-dexteleop` snapshot/adapter PR。
3. 仅当 policy artifact/wire/data contract 受影响时创建 openpi PR；模型内部变化不无条件影响 RPent。
4. 各仓独立合并；可用 stacked PR 提前并行，但正式 lock 只引用已合并 commit。
5. Integration Owner 更新 lock/compatibility，运行 stack CI，之后创建 stack tag。
6. 至少一个兼容 release 窗口后另开任务移除旧 contract。

### B10.2 RPent 检查

触及 Gateway protocol、Route、Skill 参数、receipt 或 transport 时，至少证明：

- AgenticRobot producer canonical bytes/hash；
- RPent snapshot 与目标 producer 完全一致；
- `rpent-dexteleop` 不导入 AgenticRobot lower package；
- attach-only construction 不启动 Gateway/sim/device/thread；
- sim/real、HOLD、reconciliation、safety 未被上层重定义。

### B10.3 OpenPI 检查

触及 client/server、policy artifact、state/action layout、camera slots、norm stats 或 dataset adapter
时，至少证明：

- openpi commit 与 checkpoint config/model metadata 可复现；
- AgenticRobot client 对 schema/layout hash/dimensions/camera slots fail closed；
- openpi server 不导入或初始化真机 SDK；
- model proposal 经过 AgenticRobot safety/authority/dispatch；
- fork/upstream 同步是独立 maintenance task，不混入普通业务 PR。

### B10.4 Test ownership、Profile 与 impact rules

| 测试/实现 | 所属位置 |
| --- | --- |
| AgenticRobot 内部 unit/device adapter tests | AgenticRobot 对应模块/tests |
| RPent core 与 `rpent-dexteleop` tests | RPent 仓库 |
| openpi model/wire/service tests | openpi 仓库 |
| AgenticRobot + RPent / AgenticRobot + openpi 集成 | AgenticRobot `integration/scenarios/` |
| 三仓跨机 E2E 拓扑 | AgenticRobot `integration/profiles/` + `scenarios/` |
| command_id 的启动/停止/采集 adapter | 代码 owner 仓库；主项目 Profile 只组合，不复制实现 |
| capability/resource/artifact/结果定义 | AgenticRobot Profile/schema；host-local 实例在外部 config |

当前 bootstrap 使用 `stack/impact-rules.json` 从所有仓库的 `base...snapshot` changed paths 推荐
Profiles；后续 schema 可保持同一语义转换为 YAML。示意：

```yaml
rules:
  - changes:
      AgenticRobot: ["control_station/agent_gateway/protocol/**"]
      openpi: ["src/**/protocol/**"]
    require: [gateway-contract, agenticrobot-openpi-gpu]
  - changes:
      AgenticRobot: ["**/device/**", "**/safety/**"]
    require: [agenticrobot-unit, agenticrobot-device-observe]
```

当前 `lab test <Task> --recommended` 已实现 clean、本机 R0/R1 纵向切片：先解析 detected changes，
对整批 Profile、baseline、clone identity、Task drift 和 dirty state 做原子式前置检查；失败时零 Run，
通过后按稳定顺序运行，每个 Profile 仍有独立 Run。无 changed path 时产生显式 no-op batch。远端、
高成本、交互式或设备场景仍需后续 request/authorization 流程，不能由本命令隐式获得权限。后续用户
界面再增加 host/跳过理由预览和高成本场景确认。规则只减少明确无关测试；Gateway schema、safety、
dispatch、stack lock、workflow 和 `codex_remote` 等高风险区域默认保守扩展矩阵，未知路径不能静默跳过。

### B10.5 Stack CI

stack PR 在临时 snapshot checkout lock 的 exact commits：

1. 验证 URLs/commits 可取得且属于预期保护历史；
2. 验证 lock/host placement；
3. byte/hash 比较 Gateway producer 与 RPent snapshot；
4. 验证 policy artifact 与 AgenticRobot client compatibility；
5. 运行 hermetic cross-repo import/contract smoke；
6. 生成通用 `dev-run` manifests 和 release candidate receipt。

测试选择按“语义组件 + 依赖图 + profile”执行，而不要求每个 PR 都跑真机：

| 变更区域 | 最低检查 | 条件检查 |
| --- | --- | --- |
| docs-only | lint/link/schema examples | 不启动 GPU/设备 |
| RPent core/adapter | RPent unit + Gateway snapshot/attach-only | 影响跨机行为时做本机 distributed mock |
| AgenticRobot 通用 core/Gateway | main unit + RPent contract + OpenPI client mock | 协议/dispatch 变化增加远端 integration；安全路径由 owner 决定 real gate |
| openpi metadata/wire/core | openpi unit + artifact/wire contract + main/openpi mock | CUDA/serving 变化增加远端 GPU-only |
| AgenticRobot device/runtime adapter | hermetic unit/contract | 远端 H1/full hardware 按风险与授权执行 |
| calibration/profile/device assignment | schema/owner review | 只能在目标设备 host 的受控 acceptance；不得由普通 PR runner读取真实 secret |

path filter 只能作为调度优化，不能作为授权或完整依赖证明。初期对 Gateway schema、safety、dispatch、
stack lock、workflow 和 remote wrapper 采用保守全量 required checks；依赖图和变更分类稳定后再减少
无关设备测试。

开发期远端 GPU-only 测试只接受 Hub 签发、已审批 writer 的同组织 task commits，并在无 device
group 的隔离账号执行；发布 GPU smoke 只接受 merged/release candidate。两者都不接受任意 PR/fork
code。real evidence 与 offline、sim、GPU、H1 分开记录，不能互相替代。

### B10.6 Task 到 merge/release 的证据转换

AgenticRobot/RPent/openpi 每仓都记录 commit + tree OID。Task Run 是 Development Evidence；PR
merge-group/stack candidate Run 是 Candidate Evidence；merged stack tag Run 是 Release Evidence。

- squash/rebase 后若全部参与仓 tree OID、protected Profile policy、control plane、assets、host env、
  service/fixture/resource receipts 完全相同，可以生成 `content-equivalent` receipt 供非设备 candidate
  check 引用；
- Integration Owner 对入口接线、components lock、compatibility、Profile 或任一仓 tree 的修改会让
  上游 Task evidence stale；
- synthetic draft 永不 promotion；
- GPU-only release gate 是否允许 content-equivalent reuse 由 protected policy 明确；默认对新的 stack
  candidate 重跑；
- device observe/control、统计性真机验收和 release acceptance 必须针对 merged stack candidate/release
  重跑，开发分支结果只进入 PR review。

因此 `components.lock` 只引用 merged commits 并不自动继承 Task 的“通过”。release receipt 必须列出
实际 Candidate/Release Run IDs，或显式的 content-equivalent promotion receipt。

### B10.7 Self-hosted runner 路由与隔离

能力标签和 runner group 可把 job 路由到正确主机，但 label 不是权限边界。建议分层：

| Job | Runner | 输入 | 权限/资源 |
| --- | --- | --- | --- |
| lint/unit/contract/merge-group | GitHub-hosted 或等价 hermetic runner | PR/merge-group | 无内网、设备、长期 secret |
| GPU-only acceptance | `protected-gpu` group + `remote-gpu` label | 已审批的同仓 commit 或 release candidate | 最小模型/制品凭据；GPU lease；无 device group |
| device observe/control | 不作为普通 PR 自动 job；受控 executor/manual environment | reviewed clean snapshot/release | request policy + exclusive lease + safety/人工门禁 |

硬件连接 runner 必须限定到受信 private repository/runner group；若仓库或 fork 模型允许不受信代码
触发 workflow，不能把该 runner 暴露给相应事件。GPU 与设备最好使用不同系统账号/runner；无法
物理隔离时，GPU job 也按硬件高权限面处理。`concurrency` 可以减少队列重叠，但不能替代资源 host
上的权威 lease。执行器还须独立拒绝 fork ref、任意 shell、dirty snapshot、capability mismatch 和
缺失审批。

主项目的 required checks 默认是 hermetic `unit-contract`/`stack-compatibility`；GPU acceptance
可以成为受保护 stack/release gate。device evidence 作为人工批准的发布证据，不让普通 PR 自动
取得真实硬件执行权。

## B11. 项目集成区、权限与供应链

当前项目 Integration Zone：

- AgenticRobot `stack/**`、host/release schema、`.github/**`、CODEOWNERS；
- Gateway 公共 schema/catalog 汇总与 RPent contract snapshot；
- OpenPI policy registry、artifact schema 和跨仓 config mapping；
- 各仓 `pyproject.toml`、environment/lock、top-level entrypoints/registry；
- `codex_remote` 安全公共层；
- real profile、device assignment、calibration 和 safety envelope。

Integration Owner 单写这些热点。业务 Agent 暴露可组合接口，由 Integration Owner 最后接入。

供应链/权限：

- protected control-plane release 与 task payload 使用不同安装根、import path 和系统权限；远端 executor
  启动前验证 lab/policy/bundle hashes，payload 不能覆盖其模块或 command registry；
- remote ref-sync 使用只读 deploy key；source push、release deploy 身份分离；
- Agent worktree 不暴露长期 token/SSH key/model key/device secret；
- secret、host profile、`.env` 位于外部 config/secret store，日志脱敏；
- release tag/receipt 建议签名，至少记录 commits、artifact SHA256、CI 和环境；
- PR CI 默认无内网/硬件访问；OpenPI server/Gateway 默认 localhost，经受控 tunnel；
- 自动 sync 用户不应有 sudo/docker/device group/service restart；必要时拆 `robot-sync`/
  `robot-deploy` 最小权限用户；
- datasets/checkpoints/media/logs 不入 Git/management clone，并遵循项目既有 inode/hardlink/frame/tail
  decode/独立备份规则。

## B12. 项目故障处理

| 故障 | 自动行为 | 人工处置 |
| --- | --- | --- |
| remote clone 尚在运行 | 不启动第二个 Git 操作、ref sync 或测试 | 只读查看进程/日志，等待正常结束 |
| clone 失败 | 保留错误/部分目录证据，不自动 reset/fetch | owner 决定重试/重建；不触碰 legacy DexTeleop |
| GitHub/网络不可达 | 保持现有 refs/runtime | 检查网络，不改文件同步 |
| management anchor dirty | fetch-only 可记录 refs；禁止 anchor advance | 审计来源、保留 diff/receipt，不 reset/clean |
| origin/branch 不符 | fail closed | 新建正确 clone 或人工修复；不原地强推 |
| 两个 host 声称同一 task branch writer | 停止 publish/submit | 核对 task manifest/最后 checkpoint，显式完成 writer handoff |
| writer handoff 未完成/旧 host 仍 active | 保留旧 writer，不创建远端 writable worktree | 停旧进程、checkpoint/push、补 handoff receipt 后重试 |
| baseline/override 无法解析完整 commit | 不生成 snapshot/Run | 修复 baseline lock、task manifest 或受信 ref；不让执行端猜版本 |
| test commit 不可取得或不合 policy | 不创建 worktree、不执行 | publish 正确 ref/重建受信 snapshot；离线时验证 bundle |
| target capability/profile 不匹配 | request rejected | 改 target/profile；不能用任意 SSH 绕过 |
| Task Profile 请求超过 protected policy | 高权限能力取交集后仍不合法则拒绝 | 先在 hermetic 环境审查/合并 Profile/control-plane 变更 |
| draft 含越 scope、忽略/疑似 secret 或未列出的 untracked | snapshot build 失败 | 明确 include/denylist，拆除 secret/大文件；不静默遗漏源码 |
| asset 缺失/checksum 或 driver 版本不符 | preflight 失败、设备不启动 | 修复 host inventory/资产部署，不能硬编码替代路径 |
| 分布式节点时钟偏移超限 | Run 不启动或 trace 标记 invalid | 恢复时钟同步并重新 preflight/run |
| test slot dirty/有未知进程或生成物 | slot 进入 `QUARANTINED`，不执行下一请求 | 按 slot receipt 审计；不得宽泛 clean/pkill |
| remote-open 指向 canonical/test slot/runtime | 请求拒绝 | 先创建并登记独立 remote-dev task worktree |
| import 来自其他 worktree | run 不启动 | 重建任务环境，清除错误 editable/user-site 影响 |
| 测试期间源码变化 | `invalid_source_changed` | checkpoint + detached review snapshot 后重跑 |
| port/ROS/socket 冲突 | run 不启动 | allocator 重新分配，不抢占未知进程 |
| GPU/device lease 失效 | 停新请求 | 保存证据、重新审批；下层按合同处理安全状态 |
| OpenPI/Gateway 端口可连但 ServiceReceipt 不匹配 | Ready Barrier 失败，停止本 Run 服务 | 查明旧/手工进程；不得 attach unmanaged service |
| stable runtime/test slot/remote dev 争夺 control owner | Gateway 拒绝 authority handoff | 明确当前 owner，完成受控 teardown/safety 后再申请 |
| fixture/setup/reset 不完整或场景漂移 | `BLOCKED_PRECONDITION`，不计行为失败 | 重新布置并生成新 setup/Attempt receipt |
| cancel/SSH 断线 | worker 持久化；cancel 反序 teardown | 按 Run ID watch；设备安全看 Gateway receipt，不看客户端连接 |
| RPent snapshot hash 不同 | stack CI 失败 | 走 additive contract/Change ID |
| OpenPI artifact/metadata 不同 | preflight 失败 | 选正确 checkpoint/config/artifact，不绕过 |
| release validation 失败 | 不切 current | 保留 staging evidence，修复后新构建 |
| activation health 失败 | 停新服务，回上一 release | 比较 receipt/log，不在 current 热修 |
| real 异常 | freeze Active path，保留 exact command/config/log/receipt | 只读诊断，关闭原因和缺失 safety gate 后再试 |

## B13. 分阶段落地

采用窄纵向切片；每个里程碑必须实际改善日常流程，未满足 Exit 时不并行建设后续高风险能力。

### M0：现场事实、legacy 依赖与风险冻结

1. **已完成（2026-08-13）**：owner 确认本机 70 个 calibration/profile 删除为误删，已从 HEAD
   恢复；`teleop_records` 当前无 diff。
2. **已完成（远端报告 2026-08-14）**：`/home/user/zc/AgenticRobot` clone clean，origin/main/HEAD/
   import root 已只读验收。
3. 在改变 `/home/user/zc/DexTeleop` sync 前，先做只读 runtime dependency inventory：systemd unit/
   WorkingDirectory、process cwd/exe/有限脱敏 environ、tmux、cron、shell startup、symlink、active
   ports、OpenPI/相机/机械臂 wrapper 和编译产物绝对路径。
4. 完整审计本机/远端 AgenticRobot、RPent、openpi 的 origin/HEAD/status/submodule/import roots、
   Python/uv/conda、CUDA/PyTorch、进程和外部 checkpoints/data/config 引用；不启动新 runtime。
5. 解释 legacy 依赖后，才把 working-tree 同步切成单写冻结/只读保留；停止本身若影响 runtime，
   作为单独迁移变更，不在盘点中直接执行。
6. 核实 protected canonical branch、本机历史是否已发布，并确认任何工具不会覆盖用户文件。

当前 M0 状态：步骤 1/2 和 Git/env/import/process/symlink 的主要重采样已完成。本机
`model/openpi@0d2d7f4e...` 已按 source-only、两层 exact gitlink、禁 LFS/no-fetch 初始化；远端
`model/openpi` 虽登记同一 gitlink，却仍混有独立 `.git`、`.venv/assets/logs`，不是合格 clean submodule
anchor。legacy owner/结束条件、远端 asset checksums、source/runtime 解耦和正式 gitlink commit 尚未完成；
不得宣称 M0 Exit 已满足，也不得停止或迁移 legacy。

Exit：clone/working trees/runtime 来源与 legacy 依赖可解释；没有新增双向活动源码同步，也没有因
“冻结”意外改变运行服务。

### M1：本机最小 clean Task Workspace（bootstrap working tree 已实现）

只实现 `lab task create|open|inspect|writer`、task schema、单 development host、writer authority/
actor lease、scope 和 multi-root Workspace；支持 AgenticRobot、AgenticRobot+RPent、
AgenticRobot+source-only openpi。
project manifest v2 可将 repository 默认路径相对 control checkout 解析；父子都 writable 时 allocator
把 child worktree 放在父仓 gitlink path。环境延迟创建，本阶段仅 clean commits。

Exit：开发者不手工使用 `git worktree` 即可创建、打开、检查多仓任务；不同 Agent 可显式 claim 不同
repository，同仓冲突 fail closed；`lab inspect` 显示 baseline/override、diff、scope、writer 和 provenance
readiness。

当前证据：dependency-free `lab_control` + 根 `./lab` 已覆盖 create/open/inspect 和 repository writer
status/claim/renew/release；租约通过原子状态、不可覆盖 receipt、TTL、显式 expired replacement、
manifest/epoch/branch/worktree 绑定与 advisory lock 验证同仓单 actor，不同仓可并行。Task inspection、
project sync writer inventory、synthetic snapshot barrier 和 clean Run 使用同一 lease fingerprint。临时 Git
仓测试还验证多仓 preflight、dry-run、canonical checkout 不变、scope drift 和 manifest path fail-closed。环境实际
provenance 和 baseline resolver 仍属于 M2，因此 M1 只能视为 bootstrap 完成，不是整体框架完成。

项目级同步观察/metadata fetch 的首个切片也已落地：project manifest v3 为每仓声明通用
`stable_branch + checkout_mode`，从而把“从哪个 ref 取得 objects”与“branch ff-only 或 exact detached
gitlink”分开；`lab sync inspect` 从 baseline repository-link DAG 自动展开依赖、按 child-before-parent
显示顺序、只汇总持有未过期 actor lease 的 active Task writers，并分别报告 fetch readiness 与 management checkout ff-only readiness；
`lab sync fetch` 仅按精确 refspec 更新 remote-tracking refs，禁 LFS/tag/递归 submodule，fetch 前后强制
核对 branch/HEAD/tree/status 不变。2026-08-16 在先形成 task checkpoint 后，三个稳定目录由用户明确
授权的显式 `pull --ff-only` 更新并保持 clean；这不是后台自动 apply，也没有移动 task worktree。该能力
尚未部署为远端定时任务，也不提供普通 Agent 的 checkout apply；已有受信
maintenance reservation + ff-only 流程仍是唯一 apply 边界。

### M2：本机 clean 测试与 Run（bootstrap working tree 部分实现）

实现最小 baseline lock/resolver、local unit/contract/mock Profiles、`lab test|runs`、三仓 `./dev
doctor|test`、repository locks、clean detached snapshot、Run manifest 和 `lab open --run`/
`inspect --against-run`。配置 protected branch/CODEOWNERS/基础 CI；此阶段仍不做 draft。

Exit：从本机 Task 一键生成 clean Run，并能打开该 Run 实测的精确多仓源码、报告和 artifacts。

当前证据：受信 JSON Profile、Python/risk/clean preflight、detached multi-repo snapshot、Profile argv
runner、进程组 timeout、commit/tree/control-plane/environment/日志哈希 Run manifest、结果分类、
`lab runs` 和 `lab open --run` 已实现；RPent 独立 `./dev doctor/test` 也已落地。最小 baseline + Task
override resolver 现已用受信 `stack/baseline.json` 固定三仓 commit/tree/submodule，并在 materialize 前
验证本机 clone identity/OID，submodule 初始化禁止 fetch；Run 记录 baseline ID/hash/来源。openpi
迁入 `model/openpi` 所需的通用 repository-link resolver 已实现：baseline v2 可声明父仓 path 到子仓
的有向无环依赖，父 gitlink 可派生未修改子仓 revision，父/子 Task overrides 冲突会在 snapshot 前
拒绝，关系继续进入远端 request/outbox identity。clean resolver 直接读取 exact commit tree 的 gitlink，
不再把“开发 worktree 未初始化 child”误判为版本不确定。当前 development baseline 已升级 v2 并固定
本机 v2 候选 `fa53dec8.../model/openpi@0d2d7f4e...`；它不是 protected main/release，合并后须重新固定
新的 parent commit/tree，不能沿用 Task evidence。
轻量 provenance Profile 已实现；openpi repo-local `./dev doctor/test` 尚在其未合并开发工作区，完整 heavy environment 尚未实现，因此
M2 仍是部分完成。三仓 per-repository environment binding 与 source overlay/provenance 已落地：受信
environment Profiles 分别解析 Conda `dexteleop310`/`rpent`/`rpent(source-only openpi)`，`lab doctor`
可独立检查，Run 在 exact snapshot overlay 上验证 import origin，记录环境 receipt，并将失败分类为
`INVALID_ENVIRONMENT`；跨仓 contract Profile 不再借用同一 Python。openpi source READY 不代表 GPU、
checkpoint、模型服务或设备 READY。2026-08-14 的真实本机 `lab doctor --json` 已再次确认三份 Profile
均为 `READY` 且 import origin 指向各自当前 checkout；三仓 management checkout 仍为 dirty，所以该
结果不能被提升为 clean/verify 或 release evidence。

轻量 Change 聚合也已落地：`lab change create/inspect` 与 `lab inspect <Change>` 从受信 Change manifest
汇总多个独立 child Tasks，不共享 branch/writer/worktree；未知 Change 不能挂接 Task，child drift 会使
Change 进入 `ATTENTION`。当前 remote-hosted Task create 仍硬关闭，避免仅修改 `development_host` 标签却
实际在本机创建 worktree；remote child/handoff 尚待后续里程碑。

Task inspection 现已增加首版静态 readiness/evidence dashboard：只读枚举 protected local Profiles，
按 protected `stack/impact-rules.json` 的 changed-path rules 推荐测试，将最新 Run 按 Profile SHA、baseline SHA、repo commit/tree 和
dirty 状态分类为 `NOT_RUN/CURRENT/STALE/FAILED`，并给出下一条命令。`inspect --against-run` 已实现同
Task Run 身份校验和 tested commit 到当前 worktree 的 per-repository changed paths。规则按每一个 changed
path 求并集；未命中路径使用仓库级保守 fallback，未知 Profile 或无 fallback 路径使 readiness 进入
`ATTENTION`，不会静默跳过。当前规则仍只是开发期推荐，不能替代完整 release matrix。
Change inspection 会分别保留各 child Task 的 writer/worktree 身份，并只聚合
`task_id + recommended profile`，不会把多个 Task 的 branch 或 evidence 合成一个虚构 Workspace。
clean 本机推荐批次也已落地：`lab test <Task> --recommended` 在创建任何 Run 前验证整批 readiness 和
全部 Task override 的 clean 状态，通过后逐 Profile 运行；批次返回 impact、Profiles、Runs、
`all_passed` 和 `no_tests_required`，其中单个 Run 仍是唯一证据单位。该批次当前不会调度远端或提升设备
权限。
`lab verify <Task> <Profile>|--recommended` 也已实现首版 Candidate 门槛：除 clean Task snapshot 外，
受信控制面必须是 clean `main` 且 `HEAD == refs/remotes/origin/main`，否则在创建 Run 前拒绝。Run manifest
明确记录 `evidence_tier=Candidate`；它不把开发分支结果自动提升为 Release evidence，也不授权 GPU/设备。

### M3：一个远端 clean GPU-only Profile（离线 trust contract bootstrap 已实现）

1. 验收远端 AgenticRobot management clone并启用只读 `fetch-only` receipt。
2. 在创建 `gpu-slot-01` 前完成 OpenPI minimal source/runtime separation：记录 HEAD/status/submodules/
   imports/env/CUDA/PyTorch/process/data paths，新建 clean source anchor；不搬大数据，只以有 checksum
   的 host-local asset ID 引用，并证明 clean source + 既有 checkpoint 可启动 GPU-only service。
3. 安装最小 protected control plane（validator/executor/Profile policy），Task payload无 device group。
4. 打通 AgenticRobot+openpi clean commits -> `gpu-slot-01` -> ephemeral per-Run OpenPI ServiceReceipt ->
   artifact 回传；禁止连接任何设备。
5. 实现 request idempotency、持久状态、`--follow|--detach`、watch/cancel、断线恢复和 slot quarantine。

资产准备的本地受信实现已进一步收窄第 2 步：artifact manifest 从受保护 Preparation Profile 指定的
现有项目 source/runtime 流式散列，绑定 Git commit/tree、registry 和 companion assets，且不联网、不下载；
workload selection 必须是单独 reviewed manifest，生成器只输出候选 lock，不能自动安装 active lock。生产
`stack/remote-assets.json` 当前仍 unresolved，真实 checkpoint 尚未散列或选择。另已依据 OpenPI 实际
`WebsocketPolicyServer` 将 GPU Profile endpoint identity 修正为动态 loopback `ws`，不再错误声明 HTTP。
真实 Run 前还必须重新散列三类 selected assets，生成绑定 observation、checkpoint manifest、selection
和 active lock 的 runtime verification receipt；非空 Host runtime assets 没有该 receipt 时 preflight
直接拒绝，因此一次性 inventory 不能被长期当作内容仍未漂移的证据。

Exit：用户一条 `lab test <Task> agenticrobot-openpi-gpu` 可运行 clean snapshot；结果记录 commits+
tree OIDs、control-plane、checkpoint、ServiceReceipt、environment 和 Run artifacts。无 fixed-port 猜服务。

当前证据：受保护 remote/host Profiles、exact clean commit/tree/submodule、项目定义的 artifact roles
logical ID + SHA-256、one-GPU/no-device、protected command IDs 和 content-addressed idempotency request 的
纯本地 builder/validator 已实现。项目 assets 通过显式 workload ID、asset-lock ID 与 lock SHA-256 原子绑定，
并贯穿 request/preflight/ServiceReceipt/attempt，防止混配。`lab remote plan|status|list` 复用本地 Run 的 clean Task+baseline
resolver，并要求每个 commit 已被 fetched remote-tracking ref 包含，再从受保护 asset lock 生成 strict
hashed `PENDING_DISPATCH` outbox；该状态不表示远端已接收或可执行。当前真实 asset lock 将三项 OpenPI
assets 明确标为 unresolved，因此远端审计前真实 planning 必须 fail closed。目标侧
`projectctl remote-inbox accept` 已实现 envelope/request/Profile 重验、inbox 原样留存、幂等
`SUBMITTED` state 和非执行 acceptance receipt；它不提供传输认证，必须由未来校验 SSH host key 的受信
transport 调用。本地只有在 ACTIVE transport Profile 的 pinned SSH host-key receipt 同时绑定 outbox/
acceptance 后才显示 `ACKNOWLEDGED`；未验证 receipt 只能显示 `ACK_RECEIVED_UNVERIFIED`。当前正式
transport Profile 已通过旧 IP 历史 known_hosts 与新 IP 公钥逐字匹配固定单一 ED25519 pin；生产仍必须使用
Lab 专用 known_hosts 文件，不复用 legacy wrapper 的 `accept-new` 作为高信任证明。
`lab remote dispatch` 已实现独立 SSH-stdio transport：专用 known_hosts 必须精确匹配单一 pin，强制
`StrictHostKeyChecking=yes`/`BatchMode=yes`，禁用 global known_hosts，固定远端 projectctl/inbox/state
argv，并只从 stdin 传 immutable envelope；默认 unresolved Profile 会在真实 SSH 前拒绝，当前只通过注入式
fake SSH 验证 argv、acceptance 和 receipt。当前 pin 已解析，但专用 known_hosts、protected control plane
与 assets 门槛尚未关闭，因此仍未执行真实 dispatch。
同一 transport 还固定实现 `projectctl remote-runner status|watch|cancel`：只有已认证 acceptance 的 request
可调用，返回状态必须完整匹配原 request 和合法 append-only history；本地按 sequence 留存 observation，
拒绝回退和同序号分叉。cancel 观测到 `CANCEL_REQUESTED` 也不等于安全停机，只有受信 teardown receipt
才能证明 runner 清理完成，且仍不替代设备侧 HOLD/E-stop。上述路径只通过 fake SSH 与临时 state store
回归；新地址只做过 wrapper 约束下的只读审计，没有发起真实 dispatch。
`projectctl remote-preflight`
可将 request 与 hostname/capability、repo
realpath/发布状态、environment import root、asset root/checksum、已预留 GPU lease/UUID 的 observation 一起
验证，并产生完整哈希绑定的 receipt；无副作用 `RequestStateStore` 实现 request 去重、slot 独占、合法
状态迁移、取消/quarantine 与 receipt 完整性校验。只读 `host-observe` collector 也已实现 repo Git
identity/status/ref reachability 与隔离 Python import-origin 探针，并把外部 asset/GPU receipts 合成
observation；本地 prepare lifecycle 可幂等推进到 `PREPARING`，预检失败记录
`BLOCKED_PRECONDITION` 并释放 slot；内部 `projectctl remote-runner prepare|stage-source|stage-readiness|status|cancel` 已提供持久
操作面。clean source staging 也已在本地临时 host/slot 上实现：从 exact clean anchors 创建 detached
commit/tree/submodule checkout，禁用 LFS，生成绑定 request/slot/Profile/control-plane 的 source receipt，
并保持 environment/assets/service/execution 全部未就绪；receipt 会附加到 `PREPARING` 状态。active cancel
先进入 `CANCEL_REQUESTED`，只有受信 teardown receipt 才能完成，且不伪造 hardware safety acknowledgement。
后续 readiness stage 也已在本机
hermetic host 上验证：受保护 source roots 注入 immutable Python、first-party imports 必须来自 detached
slot，asset 继续按已验证 logical ID/checksum/realpath 引用且不读取大型内容；其 receipt 绑定 preflight、
source、observation 和 GPU lease，但保持 service/execution/device 全关闭。通用
`PREPARING -> RUNNING` 明确 fail closed。受保护 remote Profile 还固定了 OpenPI service repository、
environment、protocol、launch mode、command adapter ID 与 loopback endpoint policy；本地
`projectctl service-identity launch-spec|verify-health|verify-existing` 与
`projectctl remote-runner stage-service-spec|stage-service-plan|stage-service-receipt` 已实现
content-addressed inert launch spec、一次性 launcher nonce、受保护 adapter manifest、service sandbox、
source/environment/assets/GPU 完整匹配、strict hashed ServiceReceipt 与 request-state 幂等绑定，以及
`unmanaged`/错版本 attach-existing 拒绝合同。request state v8 原子保存 endpoint lease、共享隔离 network
namespace plan、anchor/probe evidence、sandbox 与 service plan；
持久 lease 防止两个 Lab request 选择同一 loopback port，并支持同 owner 幂等、冲突拒绝和显式过期回收。
network plan 只声明 service/attempt 最终必须加入同一个 `shared-isolated-loopback`，并固定
`namespace_created=false`、`service_joined=false`、`attempt_joined=false`、`launch_authorized=false`；
它不是运行事实。v4 service plan 同时绑定 endpoint lease 与 network plan SHA-256，inner bubblewrap argv
不再自行 `--unshare-net`，而是固定 `outer_network_namespace_required=true`、
`outer_network_namespace_joined=false`、`direct_launcher_execution_permitted=false`。端口、network plan 或
adapter 身份漂移会拒绝覆盖。该合同测试不创建 namespace、不启动进程、不绑定或探测端口，
receipt 仍为 `execution_eligible=false`。其后的本地 trusted-execution edge 只能消费 Profile command ID 和
完整绑定的 attempt-start receipt；artifact 限于 attempt root、16 MiB、逐文件 hash，结果分解 execution/
service ack/assertions/task success/evidence，并区分 PASSED、FAILED_ASSERTION、INFRASTRUCTURE_FAILURE。
它只通过注入式 hermetic adapter 测试，未暴露任意执行 CLI、未部署远端，也始终禁止 device access。
受信 runner helper 现可在 launcher 已建立 anchor 后采集 OS observation、执行固定 in-namespace
loopback/route probe，并将二者作为同一原子状态更新；幂等重放会复验 executable hash、TTL、live PID/
start-ticks/netns/`NoNewPrivs`。它不创建 anchor、不加入 member、不启动服务，probe 仍明确
`launch_authorized=false`。trusted attempt 现进一步要求 request state v8 持久化 execution-sandbox
receipt 和上述 network evidence，并把 sandbox SHA-256 写入
attempt-start v2。受保护 sandbox probe 必须同时证明 user/network/process namespace、loopback-only、
外网不可达、filesystem isolation、`no_new_privileges` 与非 GPU 设备不可见；device allowlist 仅接受
`/dev/nvidia*` 的受限形式。固定 `bubblewrap` collector 已实现：只接受 Host Profile 中的 protected
backend path/GPU allowlist，安全读取并散列非 symlink executable，以固定 argv、清空环境、最小只读
系统树、新 namespace 和空网络执行固定探针；不接收 Task command/path/env。探针 receipt 与 Host Profile
SHA、backend SHA、实际 allowlist 绑定并按 Profile TTL 过期。远端虽只读确认存在 `bwrap`、`unshare`、
`systemd-run` 与 `prlimit`，但尚未执行 capability probe，因此真实 attempt 仍 fail closed。过期 probe
receipt 可以保留在 append-only request evidence 中供审计，但 attach/attempt launch 会重新检查 TTL 并拒绝。
active cancel 也已收紧为必须使用 strict teardown receipt 才能
从 `CANCEL_REQUESTED` 完成 `CANCELLED`，并显式拒绝把进程/GPU 清理收据解释为硬件安全证明。
只读 `remote-runner watch` 已用 append-only history sequence 提供最长 60 秒的断线恢复轮询原语，
但用户层 `lab --follow|--detach` 和真实 artifact transport 尚未实现。当前没有已部署的远端执行边，也不
提交或执行真实请求。新地址只读重采样确认 clone/env/GPU 存在，同时仍有失效 OpenPI symlink、planned
source/environment/asset/GPU lease 缺口。虽然用户随后确认远端互联网已恢复，Task payload 仍固定
`network_egress=loopback-only`、`downloads_permitted=false`；缺口只能由独立受信准备动作显式补齐，
因此当前真实观察仍必须 fail closed，executor/slot/service 启动保持关闭，M3 不能标记完成。

GPU 侧现已有严格 `nvidia-smi` identity inventory、单 GPU 选择、持久 `flock` lease、过期隔离和
cleanup-gated release 原语。lease 绑定 host Profile SHA、request、slot 与 GPU UUID；host observation
只能消费带完整 active lease 的哈希 runtime receipt，不能再用手工 lease ID 充数。slot 与 GPU 由
request-store 锁内的一次 resource reservation 组合持久化，request state v3 内嵌完整 reservation receipt；
预留前固定查询目标 UUID 的 compute process inventory，发现任何既有进程即在写 lease 前拒绝且不进行
进程控制；preflight 必须逐项匹配同一 lease。崩溃产生的 orphan/expired lease 进入 fail-closed reconciliation，不能
被下一请求隐式接管。该实现只在本地临时
目录和 fake `nvidia-smi` 上回归，尚未部署远端或占用真实 RTX 4090。
服务从未启动的 preflight failure 或“GPU lease 已写、request state 尚未写”的 crash orphan，现可通过
受信 `reconcile-nonstarted` 路径处理：request state 必须不存在任何 launch/service/attempt 证据，并重新
取得 30 秒内的 GPU 无 compute-process receipt；release receipt 会持久回写 request state v3。只要存在
可能已启动服务的证据，该路径即拒绝，必须等待完整反序 teardown。

远端 host Profile 同时固定 `trusted_prepare_egress=explicit-only`、`implicit_downloads=false` 和
`large_file_downloads=false`。这区分了“主机当前能访问互联网”和“某个 Run 被允许联网”：前者是动态
可达性，后者必须来自受保护 policy。Git fetch 或锁定依赖准备可以另立受信动作，但不能由 Task request、
repo-local `./dev` 或模型服务自行触发；Git LFS smudge 和大文件隐式下载继续禁用。

### M4：Synthetic draft Git snapshot（本地首个切片已实现）

实现临时 index + `write-tree/commit-tree`、allowlist/denylist、temporary refs/bundle TTL；只开放
offline/sim/GPU-only。验证 binary/mode/symlink/submodule/LFS/未跟踪文件、scope 和 secret 失败路径。
draft 不进入 verify、evidence promotion、device control 或 release。

Exit：dirty Workspace 可一键远测且远端直接 detached checkout synthetic commit，用户 branch/index
完全不变，Run 可复现实测 tree。

当前证据：`lab snapshot create <Task>` 已使用临时 index 生成不移动 branch/HEAD/index/worktree 的原生 Git
synthetic commits；多仓 repository barrier、前后 source fingerprint、content-derived snapshot ID、
repo-local refs、7 天 manifest expiry、默认 dry-run/显式 apply 的安全 prune、untracked 默认拒绝/显式 allowlist、secret/size/symlink/filter/
unmerged/submodule fail-closed 和幂等重放已有测试。所有 draft manifest 强制标记 remote/verify/device
均不可用。`lab snapshot export` 已生成相对 exact base 的增量 Git bundles 和 content-addressed manifest；
`projectctl draft-bundle verify` 已在 synthetic object 不存在的 base-only clone 上证明可在临时裸仓恢复
exact commit/tree/parent 且不写目标 clone，并拒绝 bundle 篡改。`projectctl draft-bundle materialize`
已在受控本地 slot root 创建 detached exact source，验证幂等复用、source drift、anchor 无污染、LFS
禁用和 exact submodule SHA；receipt 明确不授予执行/GPU/设备资格。TTL prune 已覆盖 exact old-OID
ref 删除、lab-owned state/default bundle tombstone、失败回滚和 ref 身份漂移拒绝。真实跨机传输、
request/runner slot 绑定与 GPU-only draft Profile 尚未实现，因此 M4 Exit 未满足。

### M5：分布式 RPent upper + remote lower/openpi

增加 Change/child+integration Tasks、两机 Profile、组合资源预留、受信 ServiceReceipts、时钟 offset/
RTT、Ready Barrier、反序 teardown、聚合 trace 和网络断线注入；仍不连接真实设备。

Exit：一个 Run ID 可以解释本机 RPent 与远端 AgenticRobot/openpi 的全部 trees、服务、启动/停止和
跨机证据，网络断开不影响远端本地安全边界。

### M6：设备 observe

新增 `device-slot-01` 的 observe-only policy、physical cell/operator lease、fixture/setup/reset receipt、
firmware/calibration/初始状态、Attempt/Outcome 分类、受信 observe adapter 和 cell quarantine；无 motion。

Exit：H1 observe 失败可区分前置、基础设施、安全中止、行为失败和人工不确定；retry 不覆盖旧 Attempt。

### M7：设备 control、merged evidence 与 release

仅对 reviewed merged stack candidate 增加 Gateway runtime authority handoff、fresh safety receipt、
control lease、统计性 trials/reset protocol、Candidate/Release evidence、staging/atomic current/rollback。
开发分支真机结果不自动 promotion；验证 stable runtime/test slot/remote dev 不会争夺 control owner。

Exit：release evidence 对应实际 merged trees、services、fixture/attempts 和 stack tag，并可受控回滚。

### M8：远端 hosted development

只有远端高频调试已成为可测瓶颈时，才实现 remote child Task、whole-Task handoff/handback、Remote
SSH Workspace 和受控 `lab session`。常规跨机并行仍使用 Change 下的 child Tasks；whole handoff 是
紧耦合例外。

Exit：remote development 不复用 canonical clone/test slot/runtime，不扩大 GPU/device/authority 权限。

legacy sync 的彻底停用/归档在新 runtime 已证明无依赖并稳定后执行；只有 owner 显式审批才删除旧
目录，本文不授权删除。

## B14. 当前项目首个里程碑与验收

首个交付范围只覆盖 M0-M3：**本地 multi-root clean 开发 -> 远端 exact clean GPU-only test ->
按 Run 打开精确源码与结果**。明确不包含 synthetic draft、分布式 RPent、设备 observe/control、
release/rollback、远端 writable development、大数据搬迁或删除 legacy。

验收：

- [ ] legacy runtime 依赖先只读盘点，再决定 sync 冻结；未因规划动作改变正在运行的服务。
- [ ] remote clone 成功退出；AgenticRobot/openpi Git/env/process/asset 基线可解释。
- [ ] 远端 management clone 精确为 `/home/user/zc/AgenticRobot`，本地/远端 origin 同为 AgenticRobot。
- [ ] 自动同步只 fetch protected refs；task/review/test checkout 永不自动移动。
- [ ] 本机有 source-only openpi clone，可运行 unit/contract/mock，但没有复制远端 secret/device state。
- [ ] 一个 Task 只有一个 development host；全部 writable repos 位于同一 multi-root Workspace。
- [ ] `lab task create|open|inspect` 覆盖三种本地 Workspace，创建时不安装重环境。
- [ ] `lab test|runs|open --run|inspect --against-run` 完成本地 clean unit/contract/mock 闭环。
- [ ] `.git`、活动源码/worktree、env/build/runtime/runs 不做跨机目录同步。
- [ ] legacy DexTeleop 未 reset/覆盖/删除。
- [ ] 最小 baseline + task overrides 解析完整 commits/tree OIDs；三仓 doctor 证明 import provenance。
- [ ] 远端 OpenPI 有 clean source anchor；checkpoint/config/norm stats 使用逻辑 asset ID + checksum。
- [ ] 受保护 control plane 与 Task payload 分离，Run 记录 lab/policy/executor 版本。
- [ ] `gpu-slot-01` 独占、detached/clean，environment 与 dependency lock hash 匹配；异常会 quarantine。
- [ ] GPU Run 启动 ephemeral OpenPI service；ServiceReceipt 精确匹配 source tree/checkpoint/env/endpoint。
- [ ] request 幂等、可 detach/watch/cancel，客户端断线不丢状态；取消会反序 teardown。
- [ ] GPU-only worker 无 device group，不连接相机/机械臂；本里程碑没有 H1/control/release 能力。

## B15. 当前项目明确不采用的方案

| 方案 | 原因 |
| --- | --- |
| 把本机/远端设计为完全对等开发节点 | 实际开发和集成集中在本机，远端能力/风险更高；对称设计会增加 writer、权限和状态协调成本 |
| 让日常用户手工管理 worktree/SSH/slot/lease/port | 容易测试错版本、目录、设备和结果；统一由 `lab` 以 Task/Profile/Run 编排 |
| 同一 Task 的 writable repos 分散在本机和远端 | 搜索、重构、diff、writer 和环境状态难以形成一个 Workspace；应整体迁移或拆子任务 |
| 每次测试手工输入三仓 SHA、设备路径和校准文件 | 应由 baseline + overrides + Profile/capability/asset resolver 自动展开并记录 |
| 双向实时同步可编辑主项目 | 破坏 branch/index/ownership/commit 原子性，不适合多 Agent |
| 跨机同步 `.git`、linked worktree、env/build/runs | Git 管理元数据只在本机有效，依赖/进程/结果也不能靠目录覆盖取得一致性 |
| 两台主机持续写同一 task branch | branch 无稳定 owner，测试结果和 push 顺序不可审计；测试端应 detached |
| 后台自动 pull/rebase/reset 活动 checkout | 会让 Agent、IDE 和长测试脚下源码漂移；后台只 fetch refs |
| draft snapshot 用于 device control/正式验收 | 未提交内容不具备受保护历史与最终可复现性；draft 只用于被 Profile 明确允许的开发测试 |
| 在 canonical clone、固定 test slot 或 runtime 中远端 debug | 混淆管理、测试、开发与部署身份；远端修改必须用独立 writable task worktree |
| 远端直接推 main/stack lock | Spoke 只产生 task/fix 候选提交，最终 review/集成由本机 Hub 和受保护流程完成 |
| 在 legacy DexTeleop 改 origin/reset | Git 身份与文件内容已脱钩，可能丢远端状态/历史 |
| 把 RPent 一并嵌入 AgenticRobot，或把 openpi 源码 vendor/复制进主仓 | RPent 保持 sibling；openpi 采用 gitlink 而非源码复制，host-local env/assets/data 仍在 Git 外 |
| 手工复制多个项目目录 | base、remote、cache、cleanup 和 ownership 不可审计 |
| 只有 worktree，无独立 env/namespace/lease | 只解决源码冲突，不解决实际测试/可视化/设备问题 |
| main merge 自动 restart remote runtime | 混淆 source propagation、deployment 和 device risk |
| 普通 PR 使用硬件 runner | 未受信代码可能取得内网、secret、GPU/设备权限 |

## B16. 参考依据

本规划吸收 workspace 根 `advise.md`、`multi-agents.md`、worktree 运行测试建议和后续
Task/Workspace/Profile/Run 使用体验复审，并按实际仓库/远端只读盘点收敛；这些输入不是
AgenticRobot 仓库内的现行合同。

- [Git worktree 官方文档](https://git-scm.com/docs/git-worktree.html)：多 working tree、独立
  HEAD/index，以及 submodule/multiple checkout 限制。
- [Git write-tree 官方文档](https://git-scm.com/docs/git-write-tree) 与
  [Git commit-tree 官方文档](https://git-scm.com/docs/git-commit-tree)：通过临时 index 生成不移动
  用户 branch/HEAD 的 synthetic tree/commit。
- [Git bundle 官方文档](https://git-scm.com/docs/git-bundle)：无法访问共同 remote 时，离线传递并
  验证 Git objects/refs 的应急机制。
- [GitHub protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)：required reviews/checks。
- [GitHub CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)：关键路径 owner 审查。
- [GitHub merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)：merge-group 合成检查。
- [GitHub self-hosted runner workflow](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow)：使用 labels/groups 路由 job。
- [GitHub self-hosted runner group access](https://docs.github.com/en/actions/hosting-your-own-runners/managing-access-to-self-hosted-runners-using-groups)：仓库/组织访问边界与 public fork 风险提示。
- [OpenAI Symphony](https://openai.com/index/open-source-codex-orchestration-symphony/)：Task/Issue 控制面、
  独立 workspace 与依赖 DAG。
- `RPent/docs/architecture/repository_boundaries.md`：RPent 与 AgenticRobot/DexTeleop ownership、
  dependency 和 release boundary。
- `AgenticRobot/codex_remote/AGENTS.md`：当前远端 wrapper、安全等级和真机约束。

---

总结：**第一部分以 Change / Task / Workspace / Profile / Run 作为分层模型，并把 worktree、
synthetic/clean snapshot、scheduler、lease 和 artifact 隐藏在 `lab` 内部，同时隔离受信控制面与
待测载荷；第二部分将其收敛为本机 Development Hub + 远端 Specialized Spoke：三仓主要在本机开发，
远端按 commit/tree 与 ServiceReceipt 执行可追溯 Run，设备证据还绑定 fixture、attempt 和唯一 authority
owner。实施先打通“本机 clean multi-root Workspace → 远端 clean GPU-only Run → 打开精确源码和结果”
的 M0-M3 纵向闭环，再逐步加入 draft、分布式、observe、control/release 和远端开发。**
