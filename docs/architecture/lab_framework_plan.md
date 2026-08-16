# `lab` 多 Agent 项目管理框架总规划

状态：框架设计基线与持续实施索引。具体代码行为以测试和对应里程碑文档为准。

本文只分为两部分：第一部分定义可复用于一般 Git 项目的管理框架；第二部分仅配置当前
AgenticRobot、RPent、openpi 的仓库关系和主机拓扑。项目名称、模型、设备、IP、绝对路径及运行命令不得进入
`lab_control/` 核心。

## 第一部分：一般项目的多 Agent 开发与运行框架

### 1. 用户模型

使用者只面对五个对象：

```text
Change     一个可能跨主机、跨 Task 的功能变更
Task       一个开发主机上的一个主动开发单元
Workspace  Task 涉及的全部仓库 worktree 与 IDE 视图
Profile    一种受保护的测试场景和所需能力
Run        某组确定源码、环境和资源上的一次测试；重试产生新 Attempt
```

`git worktree`、detached checkout、SSH、端口、环境变量、测试槽、artifact 路径和资源锁是实现细节，不是日常
操作界面。首选入口保持为：

```bash
lab task create
lab open
lab inspect
lab test
lab verify
lab runs
```

### 2. 不变量

1. 一个 Task 只有一个 `development_host`；同一仓库分支同一时刻只有一个 writer。
2. 同一仓库需要并行写入时必须拆成不同 Task/branch；跨主机并行功能使用 Change 聚合多个 Task。
3. 活动 working tree、linked worktree 元数据、`.git`、虚拟环境、build 和运行目录绝不做跨机文件级同步。
4. 跨机器只传递 Git object/ref、确定快照、Task/Run manifest、receipt 和 artifact。
5. 快速原地测试必须取得仓库执行锁；多仓快照使用短时 Workspace barrier；长测试在快照槽运行，不阻塞开发目录。
6. 每个 Task/测试槽具有独立的 first-party source overlay、build、日志、端口和运行时命名空间；下载缓存可共享。
7. 设备、GPU、物理 cell 和 operator 等稀缺资源使用组合 lease；真实控制权由独立受信 gateway 管理。
8. 被测试 payload 与受信控制面分离。Task 不能通过修改 Profile、runner 或命令定义扩大权限。
9. 一次 Run 必须绑定所有仓库 commit/tree、Profile、控制面、环境、asset、服务身份、主机和资源。
10. Development、Candidate、Release evidence 分级；设备控制和发布默认在 merged/release candidate 上重跑。

### 3. Git 与分支生命周期

基础 clone 是 worktree 管理锚点和稳定基线，不作为普通 Agent 的开发目录。Task 创建自己的分支和多仓
Workspace；测试使用 detached exact snapshot。

一个 Task/仓库只保留一条活动开发分支。实现里程碑用 commit、tag、Run evidence 和文档记录，不为每个连续小
切片长期保留一条 stacked branch。只有出现独立 writer、独立 review/PR 或显式 handoff 时才创建新分支。
分支合并或被替代后删除本地与远端 ref；有审计价值但不应继续开发的原型先打 `archive/*` annotated tag。

### 4. 多仓版本解析

有效测试 Workspace 按以下规则解析：

```text
Effective Workspace = protected baseline + Task overrides
```

未被 Task 修改的仓库自动使用 baseline 固定版本。父仓与 submodule 子仓同时出现时，父仓 gitlink 必须等于
子仓 commit；不能通过路径搜索或某个环境中的 editable install 猜测依赖版本。

### 5. 环境与运行入口

`task create` 只准备 branch、worktree、manifest 和 IDE Workspace。重型环境在首次执行某个 Profile 时延迟解析：

```text
immutable dependency base
+ task/snapshot first-party source overlay
+ profile-specific extras
```

共享 base 的 identity 至少覆盖 dependency lock、Python、平台和加速运行时；其中不得包含指向某个活动 checkout
的 first-party editable install。`doctor` 和 Run preflight 必须核对实际 import/source path。

### 6. 测试与执行生命周期

```text
Resolve -> Reserve -> Prepare -> Preflight -> Setup -> Start
        -> Ready Barrier -> Execute -> Assert/Observe
        -> Teardown -> Recovery -> Publish
```

- 本地快速测试可在暂停写入的开发 worktree 中执行，但结果明确标记 source stability。
- 长测试、可视化、GPU 或远端测试基于 exact clean commit；后续可增加 synthetic Git draft snapshot，但不能用于
  `verify`、设备 control 或 release。
- 远端 runner 先验证受保护 Queue Profile、Runner Profile、control-plane identity 和 slot claim；不能直接执行
  Request 中的任意 argv 或 shell。
- Run 自建服务优先使用 per-Run endpoint 并生成 ServiceReceipt；连接已有服务必须验证 receipt 完全匹配；未登记
  服务拒绝使用。
- 重试不覆盖原结果，而是产生带 `attempt_id`、`parent_run_id`、recovery receipt 和明确失败分类的新 Attempt。

### 7. 信任与证据

受信控制面来自 protected commit/tag 的外置只读安装，包含 validator、authorization policy、runner、resource
allocator、device gateway client 和 reviewed command adapter。被测试 payload 可包含仓库源码及 repo-local tests，
但只能在控制面授予的 sandbox、网络和资源模式中运行。

哈希提供内容完整性，不等于传输身份认证。跨机 transport 还必须提供对端身份、授权、重放保护和幂等请求；没有
受信 transport 时不得把“收到相同 hash”描述为“已认证”。

证据迁移规则：

1. commit 相同可直接复用；
2. commit 不同但 tree、Profile、assets、runner 与 host environment 全部相同，可显式标记 content-equivalent；
3. 任一源码树、依赖、Profile、配置或环境发生变化，旧证据失效；
4. 真机 control 和 release acceptance 默认在最终候选组合上重跑。

### 8. 分阶段落地

| 阶段 | 纵向能力 | 状态 |
| --- | --- | --- |
| M1 | 多仓 Task Workspace、writer lease、multi-root IDE | 已实现 |
| M2 | 本地 exact-clean detached Run 与 evidence | 已实现 |
| M3a-c | fetch-only sync、维护门、父仓/gitlink 子仓协调 | 已实现，尚未部署 |
| M3d-e | protected control-plane 外置安装与合并检查 | 已实现候选 |
| M3f-g | durable Request、幂等状态机、Runner claim、artifact receipt | 已实现协议 |
| M3h | transport envelope 与 exact clean snapshot staging | 下一切片 |
| M4 | synthetic draft Git snapshot，仅 offline/sim/GPU-only | 待实现 |
| M5 | 分布式 Profile、组合预留、ServiceReceipt | 待实现 |
| M6 | 设备 observe、fixture/attempt/quarantine | 待实现 |
| M7 | 设备 control、authority handoff、release evidence | 待实现 |
| M8 | 确有需要时的远端 hosted development | 待实现 |

M3h 先实现项目无关、无命令执行的协议：可移植 request envelope、完整性/重放校验，以及只从本机已有 Git objects
物化 exact clean detached slot。它不连接真实远端、不下载大文件、不运行项目命令，也不授予 GPU/设备权限。

## 第二部分：当前项目的组合与双机拓扑

### 1. 仓库角色

| 逻辑仓库 | 组织方式 | 当前职责 |
| --- | --- | --- |
| AgenticRobot | 主仓、开发与集成控制面 | `lab`、组合 Profile、跨仓测试与 baseline owner |
| RPent | sibling 独立仓库 | 独立 branch/commit/PR，通过 Task Workspace 组合 |
| openpi | AgenticRobot 的 `model/openpi` submodule | 独立历史，父仓 gitlink 固定源码版本 |

活动开发仍使用 sibling/multi-root 语义；submodule 只负责版本锁定和发布组合。Task 同时修改父仓与 openpi 时，工具在
Task Workspace 中创建两者各自的 worktree，并校验 gitlink/child revision 一致；不能把 openpi 的环境、模型、
checkpoint、数据或日志纳入父仓 Git。

### 2. 主机角色

```text
local-dev-control     主开发、集成、RPent 运行、本地测试与调度 Hub
remote-gpu-hardware   openpi 专用运行能力、GPU/设备测试与少量远端开发 Spoke
```

SSH 地址（当前为 `user@192.168.70.116`）只属于机器局部 host 配置，不进入 `lab_control/`、Task manifest 或
Queue Profile 的通用 contract。源码原则上可在主机间通过 Git 获得；GPU、驱动、校准、密钥、设备和大型资产只在
能力所在主机以逻辑 asset/capability ID 暴露。

默认闭环是：本机创建并开发 Task，形成 exact checkpoint，经 Git/受信 transport 提交 Run；远端在独占 slot 中
物化 detached snapshot、运行受保护 Profile并返回绑定代码组合的 manifest/artifacts。硬件强耦合调试才使用远端
hosted Task；远端偶发修复使用独立 child Task/fix branch，不与本机 writer 竞争同一分支。

### 3. 同步边界

两台机器保持一致的是 protected baseline 和可定位的 commit，不是活动目录：

```text
允许：Git fetch/push、exact snapshot、manifest、receipt、artifact
禁止：双向 rsync working tree、.git/worktree metadata、.venv、build、logs、runtime state
```

后台可 fetch；禁止自动 pull/rebase/reset 活动 worktree。稳定 checkout 只能由受保护控制面在 runtime inventory、
cleanliness、branch、HEAD/tree、gitlink 和 reservation 全部通过后执行确认过的 fast-forward。

### 4. 当前交付线

当前只保留两条 `task/AR-*` 开发 ref：

```text
task/AR-openpi-submodule-v3   openpi gitlink 与 ownership 基础
task/AR-lab-framework-v2      M1 以后连续的框架集成线
```

连续里程碑以 commit 固定：

| commit | 里程碑 |
| --- | --- |
| `b0748bba` | 将 openpi 固定为 `model/openpi` submodule |
| `4550984a` | 定义 submodule ownership 边界 |
| `6cfc8ddd` | M1 Task Workspace |
| `42568b5c` | M2 本地 Run |
| `4c62020e` | M3a fetch-only sync |
| `1a701571` | M3b maintenance gate |
| `76033197` | M3c linked checkout transaction |
| `894997ed` | M3d protected control-plane install |
| `67bb0b75` | M3e merge gate |
| `50293755` | M3f durable Run requests |
| `ecfa3c6c` | M3g trusted Runner claims |

已被替代但需留档的早期项目专用 prototype 固定在 annotated tag
`archive/lab-control-bootstrap-20260816`；它不是可继续开发或部署的框架分支。

当前阶段仍不对旧远端混合运行目录做 reset/clean，也不部署候选控制面。后续先在临时、本地、项目无关测试仓库中
完成 M3h，再以一个 clean source/test anchor 做无 GPU、无设备的远端预检，最后才逐步启用远端 GPU Profile。
