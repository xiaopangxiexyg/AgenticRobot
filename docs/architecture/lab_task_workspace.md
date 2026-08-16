# `lab` Task Workspace、本地 Run、draft 与远端合同（M1-M4 bootstrap）

`lab` 是项目无关的多 Agent 开发框架入口；当前仓库通过受保护 `integration/` 与 `stack/` 清单将其
实例化为 AgenticRobot 联邦开发流。当前落地的是 **M1 本机源码工作区**和
**M2 本地 clean Run bootstrap**、**M3 远端 GPU-only 离线信任合同**，以及 **M4 synthetic draft
本地传输/物化合同**：创建、打开和检查由一个
或多个独立 Git 仓库组成的 Task，通过受信本地 R0/R1 Profile 在 detached clean snapshot 中运行
测试，并可离线构造/验证远端请求、主机观察和 preflight receipt。它不会安装环境、fetch/push、连接
远端、启动模型服务、GPU、相机或机器人。唯一例外是用户显式执行的 `lab sync fetch`：它只更新
受保护 stable branch 的 remote-tracking refs，禁用 LFS、tag 和递归 submodule，不更新任何 checkout。

完整目标架构和分阶段边界见
[联邦多 Agent 开发流](./federated_polyrepo_development_workflow.md)。本文只描述已经实现并有测试覆盖的
命令，不能用于推断完整 M2-M8 已落地。

## 当前命令

先查看整个项目的同步状态，不访问网络：

```bash
./lab sync inspect
./lab sync inspect --repos AgenticRobot
./lab sync inspect --json
```

该视图从 `integration/project.json` 的通用 repository path/stable branch 和 `stack/baseline.json` 的
canonical identity/gitlink DAG 解析全部仓库。选择父仓时会自动加入其 gitlink 子仓，并以“子仓先、父仓后”
给出 fetch 顺序；同时列出每仓持有未过期 actor lease 的 active writer、baseline object/tree、已获取 remote ref、管理 checkout
dirty/branch 状态，以及父仓 remote commit 所需的 child object 是否可用。它只读 Git 元数据，不初始化
submodule，也不会把 detached gitlink checkout 切到分支。

显式刷新 remote-tracking refs：

```bash
./lab sync fetch
./lab sync fetch --repos AgenticRobot --timeout-seconds 120
```

`fetch` 使用精确 `refs/heads/<stable>:refs/remotes/origin/<stable>` refspec，并固定
`GIT_LFS_SKIP_SMUDGE=1`、`--no-tags`、`--no-recurse-submodules`、`--no-write-fetch-head`。它允许管理
checkout dirty 或子仓 detached，因为 fetch 前后会逐仓核对 branch/HEAD/tree/status 完全不变；Task
worktree 只作为 writer inventory 展示，永不移动。单仓失败会形成 `PARTIAL` receipt 并保留各仓结果，
不会谎称跨仓原子更新。聚合 receipt 以唯一 Sync ID 持久化到
`State/project-sync/<Sync ID>.json`，记录各仓 before/after remote ref 与未变化 checkout fingerprint。
该命令不等于 pull：ff-only 应用仍必须通过受信 maintenance reservation/runtime
inventory，普通 Task branch、review snapshot 和 test slot 没有自动更新入口。

项目清单 v3 显式声明每仓 `stable_branch` 和 `checkout_mode`；前者只决定 fetch 哪个受保护 ref，后者
区分可 ff-only 的 `stable_branch` checkout 与必须保持 exact SHA/detached 的 `pinned_gitlink` checkout。
两者都是项目无关的 Git 策略字段，不包含模型、设备或项目代码细节。旧 v1/v2 清单继续以
`main + stable_branch` 为兼容默认值。

跨主机或跨仓功能先创建一个轻量 Change，再让各个单 writer Task 归属其中：

```bash
./lab change create CHG-0042 --title "Main/OpenPI contract"
./lab task create AR-0142 --repos AgenticRobot --change-id CHG-0042
./lab task create OP-0143 --repos openpi --change-id CHG-0042
./lab inspect CHG-0042
./lab change inspect CHG-0042 --json
```

Change 不共享 branch、worktree 或 writer；它只聚合 child Task 状态和 development host。未知 Change ID
不能挂接 Task，任一 child branch/scope drift 会使 Change inspection 标记 `ATTENTION`。当前仅实现本机
`local-dev-control` Task 创建；传入其他 `--dev-host` 会 fail closed，避免把本机目录错误登记成远端
workspace。remote child Task/handoff 待受控远端开发流程实现后再开放。

先从 AgenticRobot management checkout 运行只读预检：

```bash
./lab task create AR-0101 \
  --repos AgenticRobot,RPent \
  --title "Gateway contract" \
  --scope 'AgenticRobot=deployments/**' \
  --scope 'RPent=integrations/dexteleop/**' \
  --dry-run
```

确认路径、base commit、branch 和 scope 后去掉 `--dry-run`：

```bash
./lab task create AR-0101 \
  --repos AgenticRobot,RPent \
  --title "Gateway contract" \
  --scope 'AgenticRobot=deployments/**' \
  --scope 'RPent=integrations/dexteleop/**'
```

默认本机布局为：

```text
/home/zc/HandMotion/Worktrees/AR-0101/
├── AgenticRobot/                 # task/AR-0101 worktree
├── RPent/                        # task/AR-0101 worktree
├── AR-0101.code-workspace
├── task.env
└── task.yaml                     # JSON-compatible YAML 1.2
```

打开或只打印 workspace：

```bash
./lab open AR-0101
./lab open AR-0101 --print
```

检查两个仓库的 branch、commit、dirty 文件和 scope：

```bash
./lab inspect AR-0101
./lab inspect AR-0101 --json
./lab inspect AR-0101 --against-run <run-id>
```

`DRIFTED` 表示至少出现 branch mismatch 或 scope violation。M1 只报告，不会 reset、clean、stash、
checkout 或自动修复。当前 inspection 还会只读枚举受信 local Profiles，按受保护
`stack/impact-rules.json` 对每个 changed path 给出推荐，并对未命中路径使用仓库级保守 fallback；
未知 Profile 或无 fallback 的路径会进入 `ATTENTION`，不会静默漏测。它还会
列出 recommended profiles，并把最近 Run 的 Profile hash、baseline hash、commit/tree 与当前 Task 对比，显示
`NOT_RUN`、`CURRENT`、`STALE` 或 `FAILED`。环境探针仍显式标为 `DEFERRED` 并建议运行 `lab doctor`，
inspect 本身不会偷偷创建环境或执行测试。`--against-run` 校验 Run 必须属于该 Task，再按每仓 tested
commit 到当前 worktree（包括未提交文件）列出 changed paths；不读取其他 Task 或猜测 `latest`。
`lab inspect <Change>` 同样展开每个 child Task 的上述 dashboard，并聚合推荐的
`Task/Profile` 对；它不会合并各 Task 的 writer、branch 或工作目录。

### 多 Agent writer lease

Task manifest 中的 `writer.host + writer.epoch` 是静态写入授权；某个 Agent 真正开始修改仓库前，必须取得
短期 actor lease。一个多仓 Task 可以由多个 Agent 并行处理不同仓库，但同一个 `Task/repository` 同时只
允许一个 actor：

```bash
./lab task writer status AR-0101
./lab task writer claim AR-0101 --repo AgenticRobot --actor agent-main-01
./lab task writer claim AR-0101 --repo RPent --actor agent-component-01
./lab task writer renew AR-0101 --repo AgenticRobot \
  --actor agent-main-01 --lease-id <lease-id>
./lab task writer release AR-0101 --repo AgenticRobot \
  --actor agent-main-01 --lease-id <lease-id>
```

租约以原子 `active.json` 和不可覆盖的 claim/renew/release receipts 保存在
`State/writer-leases/<Task>/<repository>/`；每份 receipt 绑定 Task manifest hash、host、writer epoch、branch
和 worktree path。重复 claim 同一 actor 是幂等读取，不会偷偷续期；另一个 actor 会 fail closed。过期或
authority 已变化的 lease 也不会自动被抢占，必须先检查旧 lease ID，再使用精确
`--replace-expired <lease-id>` 显式接管。这样 agent 崩溃不会永久占锁，也不会因时钟到期立即让另一个
agent 无痕覆盖现场。

`lab inspect` 与 `lab sync inspect` 消费同一个 lease 状态：前者逐仓显示 `AVAILABLE/ACTIVE/EXPIRED/
STALE_AUTHORITY`，后者只把未过期的 actor 列为 active writer。synthetic draft 的多仓屏障会短暂冻结相关
lease 转换，并把 actor lease receipt hash 写入 snapshot identity；clean Run 也记录执行开始时的 lease
fingerprint；任一有修改的仓库没有 ACTIVE lease 时，draft 会在创建 Git object/ref 前失败。lease 是协作
合同而不是操作系统强制沙箱：Agent 仍须通过 `lab` 工作，绕过工具直接编辑只能由 snapshot 前后 source
fingerprint 发现。

当前 receipt store 是单机控制面的事实源，适合主要开发集中在本机的拓扑；它没有伪装成跨主机分布式锁。
远端默认只消费 exact detached snapshot。以后若启用远端 writable Task，应通过受信 handoff/远端请求层
转移 `writer_epoch`，而不是复制本机 lease 文件或让两台主机分别 claim 同一 branch。

## 本地 clean Run

受信 Profile 位于 management checkout 的 `integration/profiles/*.json`，使用 argv 数组而不是 shell
字符串，并固定 risk、Python、repository、timeout 和命令 ID。当前自测 Profile 为：

```bash
./lab test AR-0101 lab-control-unit
./lab test AR-0101 --recommended
./lab verify AR-0101 --recommended
```

该 Profile 通过 `{python} dev test lab-control` 调用仓库注册目标，不再复制一份容易漏掉新增测试文件的
pytest 清单；`./dev` 仍只解析固定 target，Profile 不能注入 shell。

`--recommended` 只执行当前受信控制面能够本地运行的 clean R0/R1 Profiles。它先解析全部 changed
paths，再对整批推荐做一次前置检查：Task 必须 ACTIVE、所有 Task override 必须 clean，且 Profile、
baseline object/tree 和本机 clone identity 必须全部 READY；任一条件不满足时不会创建任何 Run。通过
前置检查后按稳定的 Profile ID 顺序逐个创建独立 Run。当前 diff 无需测试时返回可审计的 no-op batch，
不会为凑结果启动无关测试。远端、高成本、交互式和设备 Profile 仍不由该入口隐式授权。

`lab test` 产生 `Development` evidence，允许控制面源码 checkout 本身处于开发状态并在 Run 中诚实记录
`source_dirty=true`。`lab verify` 产生 `Candidate` evidence；它在创建 snapshot/Run 前额外要求控制面位于
clean `main`，且 `HEAD` 精确等于已获取的 `refs/remotes/origin/main`。缺少 protected ref、控制面 dirty、
本地超前或落后都会 fail closed 且零 Run。该 Candidate 仍不是 Release evidence；设备控制和发布候选仍须
在 merged/release candidate 上按独立规则重跑。

运行前必须满足：

- Profile 是 `local`、`R0/R1` 且 `requires_clean: true`；
- legacy Profile 可约束当前解释器；新 Profile 应为每条 command 绑定受信 environment ID；
- Profile 所需仓库来自 Task override 或 `stack/baseline.json`；baseline 仓库必须已有本机 clone 与精确
  commit/tree objects；
- Task 没有 branch/scope drift；
- 涉及的 Task worktree 全部 clean。

工具随后创建每 Run 独占的 review instance，避免两个本地 Run 共用 build/cache/source 状态；内容
相同的实例共享同一个 content-derived `snapshot-id`，但目录再以 Run ID 隔离：

```text
Worktrees/_review/<Task>/<snapshot-id>/<Run ID>/  # detached exact commits
Runtime/tasks/<Task>/<Run ID>/
├── run.yaml                                  # JSON-compatible YAML 1.2
└── <sequence>-<command-id>.log               # manifest 记录 SHA-256/size
```

Run 记录每仓 commit/tree OID、Profile hash、bootstrap control-plane commit/tree/executor hash、Python、
argv、cwd、exit code、duration 和 source stability。结果当前区分：

```text
PASSED
FAILED_ASSERTION
INFRASTRUCTURE_FAILURE
INVALID_SOURCE
INVALID_ENVIRONMENT
```

Profile 若修改 detached source，即使命令退出码为 0，Run 仍标记 `INVALID_SOURCE`，该 snapshot 后续
复用时也会因 dirty 而 fail closed。

本地 payload 不再继承控制进程的完整环境：runner 只注入隔离的 `HOME/TMPDIR/XDG_CACHE_HOME`、固定
`PATH`、Task/Run/repository roots 和显式 source overlay，避免把代理、token 或用户 site 意外传入测试。
每个命令日志使用 `O_EXCL/O_NOFOLLOW` 创建；前一条命令若预占后续日志路径，Run 会按
`INFRASTRUCTURE_FAILURE` 停止且不跟随 symlink。payload 通过受保护 `/usr/bin/prlimit` 获得 16 MiB
文件上限，日志采集再次以同一 file descriptor 校验 regular-file、大小、inode/mtime 和 SHA-256。这里
仍不是内核级 filesystem/network sandbox；R0/R1 Profile 只能运行已审查命令，后续高权限执行必须使用
独立受信 runner 身份。

查看历史或打开实测代码：

```bash
./lab runs AR-0101
./lab runs AR-0101 --json
./lab open --run <Run ID>
./lab open --run <Run ID> --print
```

`open --run` 会重新校验 Run/task/snapshot/workspace 路径身份，不能通过篡改 manifest 打开任意路径。

### 本地 synthetic draft snapshot（M4 首个切片）

开发期无需改动当前 branch/index 即可把 tracked working content 固化为原生 Git commit：

```bash
./lab snapshot create AR-0101
./lab snapshot create AR-0101 --include '*.py' --json
./lab snapshot export AR-0101 <snapshot-id> --output /path/to/staging
```

实现使用每仓独立临时 `GIT_INDEX_FILE` 执行 `read-tree/add/write-tree/commit-tree`；不执行 checkout、
stash、reset 或 commit 当前 branch。生成前后会比较 branch、HEAD、status、working diff、cached diff、
index entries 和已纳入 untracked 文件 hash，任一变化均 fail closed。多仓 Task 在短暂 repository barrier
内一次解析，commit 由 `refs/lab/snapshots/<snapshot-id>/<repo-hash>` 保活，manifest 写入
`State/drafts/<Task>/<snapshot-id>/snapshot.json`。

默认不纳入任何 untracked 文件；`--include` 是显式 glob allowlist。首版同时：

- 拒绝 unmerged path、非 exact/未初始化 submodule、untracked symlink；
- 拒绝 `.env`、key、token、credential/secret 路径以及超过默认 1 MiB 的 selected file；
- 拒绝任何带 clean/smudge filter（包括 LFS filter）的 selected path；
- 记录 excluded untracked path，但不读取其内容进入 snapshot；
- 标记 `remote_eligible=false`、`verify_eligible=false`、`device_eligible=false`。

因此它目前只是可审视、内容寻址的本地开发快照，不能用于正式验收、release、GPU 远测或设备测试。
manifest 有 7 天 `expires_at`。生命周期盘点默认只读：

```bash
./lab snapshot prune
./lab snapshot prune --task AR-0101 --json
./lab snapshot prune --task AR-0101 --apply
```

只有显式 `--apply` 才回收 expired snapshot。工具在 repository locks 内重新核对 inventory，要求每个
synthetic ref 仍指向 manifest 的 exact commit；随后先把 lab-owned state/default export 原子移入受控
retired tombstone，再用 old-OID 条件删除 refs。任一步失败会恢复已删 refs 和目录。成功后只清理 lab
默认 export；用户用 `--output` 创建的外部 bundle 永不被扫描或删除。

`snapshot export` 为每个 synthetic repository 生成只包含 `base..synthetic` 新对象的增量 Git bundle；
clean repository 只记录 prerequisite，不重复打包。`bundle.json` 固定源 manifest hash、base/commit/tree/ref、
submodule SHA、bundle SHA-256/size 和资格包络。导出包只能传输，仍标记 `remote_run_eligible=false`、
`verify_eligible=false`、`device_eligible=false`。manifest 上限 2 MiB、单 bundle 上限 16 MiB、整个
bundle set 上限 32 MiB；读取和散列使用不跟随 symlink 的稳定 descriptor，超过上限直接拒绝，不允许
借 draft 通道传模型、数据集或其他大文件。

受信执行侧在接收后应先运行：

```bash
./projectctl draft-bundle verify /staging/<snapshot-id>/bundle.json \
  --repository AgenticRobot=/trusted/AgenticRobot \
  --repository openpi=/trusted/openpi
```

verifier 要求目标只有准确 prerequisite base、尚不包含 synthetic commit；它在临时裸仓中通过 alternates
读取 base、从 bundle 恢复并验证 exact commit/tree/parent，随后删除临时仓。目标 management clone 不会
获得 synthetic object/ref。该 receipt 只证明 transport source。

受信侧可进一步把已验证 bundle 物化成惰性的、内容寻址的 slot source：

```bash
./projectctl draft-bundle materialize /staging/<snapshot-id>/bundle.json \
  --repository AgenticRobot=/trusted/AgenticRobot \
  --repository openpi=/trusted/openpi \
  --slot-root /trusted/test-slots \
  --slot gpu-slot-01
```

materializer 在 slot 锁内创建隔离 clone，以 detached HEAD checkout exact commit/tree；submodule 也从
clean trusted anchor 以记录的 SHA 物化。Git LFS smudge/fetch 被显式禁用，management anchor 的
branch/HEAD/status/submodule fingerprint 前后必须一致。重复请求复核并复用同一 content-addressed
目录；slot source 漂移会 fail closed。生成 receipt 仍强制 `execution_eligible=false`、
`remote_run_eligible=false`、`verify_eligible=false`、`device_eligible=false`，所以这不是 GPU、网络、
设备或任意命令执行授权。

### 独立环境与 source overlay

检查当前机器上所有受信本地环境：

```bash
./lab doctor
./lab doctor --environment agenticrobot-py310 \
             --environment rpent-py311 \
             --environment openpi-source-py311 \
             --json
```

环境 Profile 位于 `integration/environments/*.json`，声明 owning repository、Python major/minor、Conda
env 或受信绝对 Python、必须验证的 import modules，以及该仓可加入 overlay 的相对 source roots。当前：

| environment ID | repository | resolver |
| --- | --- | --- |
| `agenticrobot-py310` | AgenticRobot | Conda `dexteleop310`, Python 3.10 |
| `rpent-py311` | RPent | Conda `rpent`, Python 3.11 |
| `openpi-source-py311` | openpi | Conda `rpent`, Python 3.11；仅 source provenance，不代表 GPU runtime |

Run 不做跨 worktree editable install。它把 exact detached snapshot 的受限 source roots 组成显式
`PYTHONPATH`，为各 command 展开自己的 `{python}` launcher；在空临时 cwd 中探测解释器、prefix 和 import
origin，所有 first-party imports 必须位于对应 snapshot repository。`PYTHONNOUSERSITE=1` 且
`PYTHONDONTWRITEBYTECODE=1`，避免用户 site 或 bytecode 污染 review source。环境失败生成有 manifest 的
`INVALID_ENVIRONMENT` Run，command 不启动；若探针本身改变 source，则优先归类 `INVALID_SOURCE`。

`integration/profiles/agenticrobot-rpent-contract.json` 是第一个跨仓 Profile：AgenticRobot Gateway 合同
使用 3.10 环境，RPent repository/contract snapshot 使用 3.11 环境。未修改的 RPent 可由 baseline 自动
补齐。`integration/profiles/openpi-source-anchor.json` 只验证 baseline OpenPI 的源码布局、两个
first-party import origin 和 exact submodule gitlink；它不加载模型依赖，也不授予 GPU、网络或设备能力。

### Baseline + Task override

`stack/baseline.json` 固定当前已审计的 AgenticRobot、RPent、openpi commit/tree/submodule 组合。解析规则是：

```text
Effective snapshot = Task 中的 writable repository overrides
                   + Profile 所需但 Task 未修改的 baseline pins
```

baseline 不是源码下载器。使用某个 baseline pin 前，本机仍必须有对应 management clone，origin 的
`owner/repository` identity、commit object 和 tree OID 必须与 lock 一致；SSH alias 和 URL transport 可以
不同。解析器不会 fetch、pull 或猜测替代版本。带 submodule 的 snapshot 只使用本机已有 Git objects 做
`--no-fetch` 初始化，缺 object 会 fail closed。Run/snapshot 记录 baseline ID/hash、每仓来源和 submodule
身份。这样只修改 AgenticRobot 的 Task 可由跨仓 Profile 自动加入 RPent baseline，而无需为 RPent 创建
writable task branch。

openpi 的目标源码位置是 AgenticRobot 的 `model/openpi` submodule。baseline/Effective Workspace 不能把
AgenticRobot 中的 `model/openpi` gitlink 与独立 openpi pin 当成两个可自由选择的版本：gitlink 是权威版本边，
独立条目只补充 child tree、origin 和运行位置并必须与其相等。只读任务用 task-local detached submodule；
修改 openpi 的任务由 allocator 使用 child repository 自己的 worktree/branch；父子仓同时入 Task 时，
child worktree 位于父 Task checkout 的 gitlink path，在嵌套 child repository 中保留独立
branch/PR/writer lease，并在产生候选快照前更新父仓 gitlink。用户操作面仍是同一个 Task Workspace，
不会暴露共享可写 submodule 目录。

2026-08-15 本机 working tree 已把 `.gitmodules` 和 `model/openpi@0d2d7f4e...` 登记为 staged gitlink，
并以本机已有 objects、`GIT_LFS_SKIP_SMUDGE=1`、`--no-fetch` 初始化其两个嵌套 submodule；远端也有同一
staged gitlink，但仍需完成 source/runtime 分离和正式 commit。`lab.baseline.v2` 的通用形式是：

```json
{
  "schema": "lab.baseline.v2",
  "baseline_id": "baseline-linked-v2",
  "repositories": {
    "main-project": {
      "canonical_identity": "example/main-project",
      "commit": "1111111111111111111111111111111111111111",
      "tree_oid": "2222222222222222222222222222222222222222",
      "submodules": {
        "components/component-y": "3333333333333333333333333333333333333333"
      }
    },
    "component-y": {
      "canonical_identity": "example/component-y",
      "commit": "3333333333333333333333333333333333333333",
      "tree_oid": "4444444444444444444444444444444444444444",
      "submodules": {}
    }
  },
  "repository_links": [
    {
      "parent_repository": "main-project",
      "submodule_path": "components/component-y",
      "child_repository": "component-y"
    }
  ]
}
```

loader 要求父 baseline 的 gitlink 等于 child baseline commit，并拒绝未知仓、重复 path 和环。clean
resolver 在父仓 Task 更新 gitlink、child 未单独覆盖时，从 child management clone 的本地 objects 派生
exact commit/tree/submodule map；父子都为 Task override 时，commit 不同会直接失败。解析关系进入
snapshot/Run 以及 `lab.remote-request.v2`、`lab.remote-outbox.v2`，远端请求再次验证 parent submodule、
child commit 和 link commit 相等。clean snapshot 从 exact commit tree 读取 gitlink，不要求开发
worktree 为只做审视而预先下载/初始化所有 child；执行槽 materialize 时才要求本机已有 exact objects。
v1 baseline 继续兼容但不能声明 links。当前 development baseline 已升级为 v2，固定已推送的迁移候选
`AgenticRobot@fa53dec8... -> model/openpi@0d2d7f4e...`；其 ID 明确带 `candidate`，只用于开发期解析。
合并或 rebase 后 parent commit/tree 会改变，必须重新生成 Candidate/Release baseline 和证据，不能把这个
Task branch commit 冒充 protected main/release。

## 仓库内统一入口

AgenticRobot 当前提供：

```bash
./dev doctor
./dev doctor --json
./dev test lab-control
```

通用 `dev_cli` 不内置项目包名、Python 版本或测试清单；它从项目拥有的
`integration/dev.json` 读取这些约束。本项目配置要求 Python 3.10，并证明 `lab_control`、
`robot_motion`、`deployments` 的 import source 位于当前 checkout。配置文件及其 SHA-256 也进入
doctor 输出。`doctor` 只检查和报告，不创建环境。应从项目环境运行：

```bash
conda run -n dexteleop310 ./dev doctor
```

`lab doctor` 现在默认从 `model/openpi` 提供 source-only import provenance。原 sibling openpi 工作目录中的
repo-local `./dev`/测试辅助文件尚未进入固定的 openpi commit，因此不能假装已随 submodule 发布；在这些
文件通过 openpi 自己的 branch/PR 合并前，只能在原独立开发 worktree 使用：

```bash
conda run -n rpent ./dev doctor
conda run -n rpent ./dev test source
```

该入口只使用标准库检查 Git/submodule/source import provenance。它明确报告
`runtime_assets.resolved: false`；不得把它的 READY 或测试通过解释为 OpenPI CUDA、checkpoint、模型服务
或设备能力已经就绪。此 clone 通过 repo-local `lfs.fetchexclude=*` 保持大文件 fetch 禁用。

## 仓库解析

默认解析：

| 名称/别名 | 默认 management checkout |
| --- | --- |
| `AgenticRobot` / `main` | 当前 `lab` 所在 AgenticRobot checkout |
| `RPent` | AgenticRobot 的 sibling `RPent/` |
| `openpi` | 当前 AgenticRobot checkout 内的 `model/openpi/` submodule |

路径不同可设置：

```bash
export LAB_REPOSITORY_AGENTICROBOT=/path/to/AgenticRobot
export LAB_REPOSITORY_RPENT=/path/to/RPent
export LAB_REPOSITORY_OPENPI=/path/to/an/alternate/openpi-checkout
export LAB_HOME=/path/to/lab-home
export LAB_BASELINE_LOCK=/path/to/baseline.json
```

`LAB_HOME` 下使用 `Worktrees/<Task ID>`。也可以用
`LAB_WORKTREES_ROOT` 只覆盖 worktree 根目录。仓库名称、别名、默认路径和
可选路径环境变量由受保护的 `integration/project.json` 声明，而不是写死在框架代码中。manifest v2 的
`path_base` 只表达“相对 control checkout”或“相对其父目录”的通用寻址；框架核心没有 openpi 特例。

## 安全与事务边界

- Task ID、branch、scope 和仓库路径先校验；所有仓库全部通过 preflight 后才开始写入。
- 默认 base 是每个 management checkout 的当前 `HEAD`，manifest 固定完整 commit OID。
- 已存在同名 local branch 或已知 remote-tracking branch时拒绝创建，不擅自接管 writer。
- `git worktree add` 禁用 repository hooks；失败时只回滚本次新建的 worktree/branch。
- management checkout 可以有未提交内容，但它们不会进入新 Task；manifest 会记录创建时该 anchor
  是否 dirty。真正的 dirty/synthetic snapshot 属于 M4，不是 M1 能力。
- 默认 write scope 是仓库全量 `**`；多 Agent 或高风险任务应显式缩窄 `--scope`。
- `task.yaml` 的 workspace/worktree path、branch、base OID 和 scope 会在 inspect 时重新校验；路径被
  篡改时 fail closed。
- `lab open` 只打开代码，不赋予测试、网络、GPU 或设备权限。

## 当前未实现

以下命令和能力仍属于后续里程碑：

```text
lab session / runs watch/cancel
openpi immutable heavy environment Profile
synthetic draft 真实跨机传输、runner 绑定与 remote GPU 使用
remote observation collector / real runner / GPU lease 远端部署与真实 reservation / slot/service process lifecycle
device observe/control / authority
task handoff / remote child Task commands
```

RPent 已有自己的 `./dev doctor` 和 `./dev test unit|dexteleop|boundary|all`。baseline resolver 和
per-repository environment binding 已落地，不再让 AgenticRobot+RPent Profile 借用同一个调用者 Python。
openpi 的轻量 provenance environment 已从 `model/openpi` 验证；repo-local `./dev doctor/test source`
仍只存在于尚未合并的 openpi 开发工作区，不属于当前 gitlink commit。本机 immutable heavy environment
以及 GPU/model runtime 仍未落地。

2026-08-15 在迁移后的默认路径再次运行 `lab doctor --json`，三份受保护环境 Profile 均返回 `READY`：
AgenticRobot `lab_control` 来自当前 AgenticRobot checkout，RPent 来自当前 RPent checkout，openpi 与
`openpi_client` 分别来自 `model/openpi` 的两个受保护 source root。AgenticRobot/RPent 的 `./dev doctor`
以及 `lab doctor` 验证了 Python 版本和 import provenance；management checkout 的 dirty 状态仍需独立
记录，因此该
结果只证明本机 source/environment 可读可测，不是 clean verify、remote GPU 或 release evidence。

## M3 离线请求合同 bootstrap

`integration/remote_profiles/agenticrobot-openpi-gpu.json`、
`integration/host_profiles/remote-gpu-hardware.json` 与 `lab_control.remote_request` 已固化第一个远端
Profile 的权限包络和目标主机的受信静态布局：

```text
target: remote-gpu-hardware
risk: R1
repos: AgenticRobot + openpi exact clean commit/tree/submodules
assets: checkpoint + config + norm-stats logical ID/SHA-256
resources: exactly one GPU, device_access=false
network: task payload loopback-only, implicit/large downloads=false; trusted preparation explicit-only
commands: protected command IDs only，request 中没有 argv/shell
host: exact repo/env/asset/slot roots + gpu:cuda/openpi-runtime capabilities
```

日常 Task 侧现在先使用无网络副作用的出站计划入口：

```bash
./lab remote plan <Task ID> agenticrobot-openpi-gpu
./lab remote doctor agenticrobot-openpi-gpu
./lab remote status <request-id>
./lab remote list [Task ID]
./lab remote dispatch <request-id> \
  --transport remote-gpu-hardware-ssh \
  --dry-run
./lab remote refresh <request-id> --transport remote-gpu-hardware-ssh
./lab remote watch <request-id> --transport remote-gpu-hardware-ssh
./lab remote cancel <request-id> --transport remote-gpu-hardware-ssh
```

`remote plan` 复用本地 clean Run 的同一个 effective-workspace resolver，以 Task overrides 覆盖
`stack/baseline.json`，然后读取受保护 `stack/remote-assets.json`，生成 content-addressed request 和
strict hashed outbox envelope。每个 repo commit 必须已经被本地 fetched remote-tracking ref 包含；未
checkpoint/push 的 Task commit 会在创建 outbox 前拒绝。outbox 只写
`State/remote-outbox/<request-id>/envelope.json`，状态固定为 `PENDING_DISPATCH`、`remote_ack=null`、
`execution_eligible=false`；它不 SSH、不 push、不代表远端已接收。

asset lock 是项目定义制品角色的原子组合，而不是若干可自由拼接的散列。通用 lock 只包含
可选 `workload_id`；它表示项目定义的运行变体，可以被具体项目解释为模型、编译产物、仿真场景或其他
服务配置。没有此概念的项目保持 `null`；只有 adapter 使用 `{workload_id}` 或声明
`supported_workload_ids` 时才要求非空。lock ID、workload ID/null 和 lock 文件 SHA-256 会同时写入
request、outbox、preflight、ServiceReceipt 与 attempt-start。当前 bootstrap lock 使用
`workload_id=null` 且项目所需角色 unresolved，因此不能
生成 request；目标主机上恰好存在某个目录不能自动成为 workload 授权。

`remote doctor` 只读取 protected remote Profile、asset lock 与 transport Profiles，不访问网络；输出
`planning_ready`、`dispatch_ready` 和明确 blockers。当前 transport 已通过旧 IP 的历史 known_hosts
ED25519 公钥与新 IP 记录逐字比对形成单一 ACTIVE pin；assets 仍 unresolved，Lab 专用 known_hosts 文件和
远端 protected control plane 也尚未部署，因此真实 planning/dispatch 仍按各自门槛 fail closed。

目标节点的下一段无执行副作用入口也已实现：

```bash
./projectctl remote-inbox accept \
  --envelope received-envelope.json \
  --profile agenticrobot-openpi-gpu \
  --inbox-root /protected/runner-inbox \
  --state-root /protected/runner-state
```

`remote-inbox accept` 会重新验证 envelope hash、request content identity、protected Profile 与
control-plane version，原样持久化 envelope/request，并只把 RequestStateStore 幂等推进到
`SUBMITTED`。返回的 acceptance receipt 绑定 outbox hash、request state hash、target host 和 Profile，
且固定 `execution_eligible=false`。重复接收返回原 receipt；若同一 request 已进入 `RESERVED` 或更后状态，
接收入口拒绝伪装成一次新的 acceptance。它本身不认证文件如何抵达目标机；生产使用必须由后续受信
SSH transport/host-key receipt 调用，不能把手工复制文件视为已认证传输。

本地 dispatch 状态不会仅凭 acceptance JSON 变成 `ACKNOWLEDGED`。只有受保护 transport Profile 为
`ACTIVE`，且 transport receipt 记录的 SSH host-key fingerprint 命中 pin、同时精确绑定 outbox 与
acceptance hashes，状态才提升；否则最多是 `ACK_RECEIVED_UNVERIFIED`，仍不可执行。当前
`integration/transport_profiles/remote-gpu-hardware-ssh.json` 已固定经历史记录复核的单一 ED25519 指纹；
但专用 known_hosts 和远端 control plane 未安装、asset lock 未解析，所以真实自动传输仍保持关闭。现有
通用 `codex_remote` wrapper 的 `accept-new` 语义不能代替专用 pin 文件，也不能生成受信 transport receipt。

`remote dispatch` 的执行实现已经存在，但正式 Profile 或专用 pin 文件未就绪时会在任何 SSH 调用前拒绝。启用后它只
读取专用、非 symlink 的 known_hosts，以本地 `ssh-keygen -F` 离线验证该 host 恰好只有 Profile 固定的
一个 fingerprint；实际 argv 强制 `BatchMode=yes`、`StrictHostKeyChecking=yes`、
`GlobalKnownHostsFile=/dev/null`，并把 envelope 通过 stdin 交给固定绝对路径的远端 `projectctl
remote-inbox accept --envelope -`。Task 无法提供远端命令、路径或 shell。`--dry-run` 会完成本地 pin/
argv 检查但不使用网络；只有 SSH 返回的 acceptance 通过全量绑定后，工具才保存 authenticated transport
receipt。当前没有执行过真实 dispatch。

已核验的 pin 可显式安装到 Lab 专用 trust state，而不用手工复制 hashed known_hosts 行：

```bash
./lab remote trust \
  --transport remote-gpu-hardware-ssh \
  --source-known-hosts ~/.ssh/known_hosts
```

该命令离线调用 `ssh-keygen -F`，要求 source 对目标 host 恰好解析出一个且仅一个 Profile-pinned key，
随后以 mode `0600` 原子写入 `State/trust/known_hosts`。既有内容不同则拒绝覆盖；重复安装幂等并生成
`network_used=false`、`accept_new_used=false` 的 receipt。生产默认路径尚未写入；当前只在 `/tmp` 用真实
历史记录验证过该入口。

同一受信 SSH-stdio 通道也已实现 `refresh/watch/cancel`。这些命令只接受 outbox 中已有的 request ID，
远端 argv 固定为 `projectctl remote-runner status|watch|cancel`，不能由 Task 注入 shell 或路径；只有已有
authenticated dispatch acknowledgement 的 request 才可调用。返回状态会按原 request 的 idempotency、
Task、Profile、receipt 和完整状态历史重新验证，再作为 append-only observation 保存在 outbox。序号回退
或同序号内容分叉会 fail closed。`watch` 默认从本地最后观测序号继续；SSH 断开不会改变远端持久状态。
`cancel` 只表示取消请求已抵达：活跃请求进入 `CANCEL_REQUESTED`，只有远端受信 teardown receipt 才能
成为 `CANCELLED`，且该合同不声称机器人已经安全停止。虽然 Profile pin 已解析，这些入口仍只通过本地
fake SSH/state store 验证；远端受保护入口尚未部署，未执行真实 dispatch/watch/cancel。

当前正式 `remote-assets.json` 有意把 checkpoint/config/norm-stats 标记为 `unresolved`。在远端恢复并由
只读审计取得真实 asset ID/SHA-256 前，真实 Profile planning 必须 fail closed，不能用占位 checksum
伪造可调度请求。测试使用临时小型 asset lock 验证成功路径，不读取或下载任何大文件。

`stack/remote-asset-observations.json` 现把只读现场事实与 active lock 分离记录：绑定
`openpi@0d2d7f4e...`、registry 路径/SHA-256、norm-stats 路径/SHA-256，以及注册表引用且目录存在的
`dual_rm65_dmgripper_latest` 候选。该 schema 强制 `selection_authorized=false`、
`activation_ready=false`、主内容 `content_manifest_sha256=null`，blockers 必须同时包含
`content-manifest` 与 `explicit-selection`；它不能直接转换成可调度 asset lock。
`lab remote doctor` 会显示 `OBSERVED_NOT_AUTHORIZED`，同时仍让三个 active asset role 保持 unresolved。
本轮没有读取、散列或下载真实 checkpoint 权重。

受信控制面现提供两步、非自动激活的资产准备原语：

```bash
./projectctl assets artifact-manifest \
  --observations stack/remote-asset-observations.json \
  --workload dual_rm65_dmgripper_latest \
  --preparation-profile remote-gpu-hardware-bootstrap \
  --host-profile remote-gpu-hardware \
  --output /home/user/zc/robot-stack/data/manifests/<name>.json

./projectctl assets select \
  --observations stack/remote-asset-observations.json \
  --artifact-manifest /path/to/manifest.json \
  --selection /path/to/reviewed-selection.json
```

第一步只对 Host Profile 固定的现有 OpenPI source/runtime 中指定 checkpoint 做流式 SHA-256，拒绝
symlink、非普通文件、扫描期间变化、Git/registry/norm-stats 身份漂移，并明确记录
`network_used=false`、`large_file_downloaded=false`、`selection_authorized=false`。清单只能写入已有且
非 symlink 的受保护 asset root。第二步必须消费一份独立、reviewed 的
`lab.workload-selection.v1`，绑定 observation hash、artifact receipt 和
`gpu-only-validation/loopback-only/no-download/no-device`；它只输出候选 resolved lock，不自行覆盖
`stack/remote-assets.json`。因此“目录存在”仍不能被误升格为模型授权。

生成 active lock 以后，Run preflight 仍不能只信 lock 中的字符串。新增的 runtime asset verifier 会在
每个真实准备周期重新流式散列 checkpoint 全部普通文件，并重新散列 registry/norm-stats；只有 live 内容、
reviewed artifact manifest、workload selection 和 active lock 四者完全一致，才签发
`lab.runtime-asset-verification.v1`。Host runtime receipt 对非空 assets 强制绑定这份 receipt，
Host observation 再携带其 SHA-256；GPU preflight 缺少该 hash 时 fail closed。该过程只读本机已有内容，
不会联网或下载大文件，但可能因 checkpoint 体积产生可预期的本地顺序读取开销。

为避免在 M3 bootstrap 中复制或重新下载大型模型，远端 Host Profile 临时将现有
`/home/user/zc/openpi/{checkpoints,configs,assets}` 三个精确目录列为 legacy read-only asset roots，同时
保留 `/home/user/zc/robot-stack/data` 作为新 manifest/后续制品根。它没有把整个混合 OpenPI worktree列为
asset root；runner 只能把 selected exact paths 只读映射给单次 Run。完成数据迁移后再通过独立受保护
变更移除 legacy roots，不能自动移动或删除现有大文件。

内部维护者可以在本地从 snapshot/asset identity JSON 构造和复核 content-addressed request：

```bash
./projectctl remote-request build \
  --task-id AR-0301 \
  --snapshot-id snapshot-0301 \
  --snapshot /path/to/snapshot.json \
  --assets /path/to/assets.json \
  --profile agenticrobot-openpi-gpu

./projectctl remote-request validate request.json \
  --profile agenticrobot-openpi-gpu

# 以下 GPU 命令只允许已安装的受信控制面调用。inventory 只读；reserve
# 会持久占用 GPU；release 必须消费受信 cleanup receipt。当前开发 worktree 不得远端执行它们。
./projectctl gpu-lease inventory \
  --host-profile remote-gpu-hardware

./projectctl gpu-lease reserve \
  --request request.json \
  --profile agenticrobot-openpi-gpu \
  --host-profile remote-gpu-hardware \
  --slot gpu-slot-01 \
  --state-root /protected/runner-state \
  > gpu-reservation.json

# 只处理“服务从未启动”的 FAILED/CANCEL_REQUESTED/QUARANTINED request，
# 或 GPU 写入后 request-state 尚未落盘的 SUBMITTED orphan；有 service/attempt 证据时拒绝。
./projectctl gpu-lease reconcile-nonstarted \
  --request request.json \
  --state-root /protected/runner-state \
  --host-profile remote-gpu-hardware \
  > gpu-release.json

./projectctl host-observe \
  --host-profile remote-gpu-hardware \
  --runtime-receipts /protected/path/runtime-receipts.json \
  > observation.json

./projectctl remote-preflight \
  --request request.json \
  --observation observation.json \
  --profile agenticrobot-openpi-gpu \
  --host-profile remote-gpu-hardware

./projectctl remote-runner prepare \
  --request request.json \
  --observation observation.json \
  --profile agenticrobot-openpi-gpu \
  --host-profile remote-gpu-hardware \
  --slot gpu-slot-01 \
  --state-root /protected/runner-state

./projectctl remote-runner stage-source \
  --request request.json \
  --profile agenticrobot-openpi-gpu \
  --host-profile remote-gpu-hardware \
  --state-root /protected/runner-state

./projectctl remote-runner stage-readiness \
  --request request.json \
  --observation observation.json \
  --profile agenticrobot-openpi-gpu \
  --host-profile remote-gpu-hardware \
  --state-root /protected/runner-state

./projectctl remote-runner stage-service-spec \
  --request request.json \
  --profile agenticrobot-openpi-gpu \
  --state-root /protected/runner-state \
  --launcher-nonce <protected-one-time-nonce>

./projectctl remote-runner stage-service-plan \
  --request request.json \
  --backend-probe sandbox-backend-probe.json \
  --profile agenticrobot-openpi-gpu \
  --host-profile remote-gpu-hardware \
  --runtime-root /protected/test-slots/gpu-slot-01/runtime/<request-id> \
  --endpoint-port <runner-selected-loopback-port> \
  --state-root /protected/runner-state

./projectctl remote-runner status <request-id> \
  --state-root /protected/runner-state

./projectctl remote-runner watch <request-id> \
  --state-root /protected/runner-state \
  --after-sequence <last-seen-sequence> \
  --timeout-seconds 30

./projectctl remote-runner cancel <request-id> \
  --state-root /protected/runner-state

./projectctl remote-runner ack-cancel <request-id> \
  --receipt trusted-teardown-receipt.json \
  --state-root /protected/runner-state

./projectctl service-identity launch-spec \
  --request request.json \
  --readiness readiness.json \
  --profile agenticrobot-openpi-gpu \
  --slot gpu-slot-01 \
  --launcher-nonce <protected-one-time-nonce> \
  > launch-spec.json

./projectctl service-identity verify-health \
  --launch-spec launch-spec.json \
  --service-sandbox service-sandbox-receipt.json \
  --process-receipt managed-process-receipt.json \
  --health managed-health.json \
  --host-profile remote-gpu-hardware \
  > service-receipt.json

./projectctl service-identity verify-existing \
  --launch-spec attach-existing-spec.json \
  --receipt managed-service-receipt.json
```

`host-observe` 是预定部署在受信执行节点的只读采集器：它不 fetch/checkout，读取 host Profile 指定
repo 的 origin identity、commit/tree/submodule/status/remote-ref reachability，并通过指定环境的 Python
`-I` 探针记录模块 import origin；asset 路径与 GPU lease/UUID 来自独立受信 runtime receipts，采集器只
补充存在性、可读性和 realpath。采集成功不代表获准执行。

`remote-preflight` 要求观察值证明：目标 hostname/capabilities 正确；两个 repo 是受信 realpath 上的 clean
exact commit/tree/submodules 且 commit 已发布；两个环境的 Python/import root 正确；三个 asset 的逻辑
ID、SHA-256、存在性和允许根目录正确；恰有一份已预留 GPU lease，并记录 GPU UUID；device groups 必须
为空。输出 receipt 绑定 request、remote/host Profile、control-plane、完整 observation hash 和 GPU lease。

`RequestStateStore` 还提供无执行副作用的持久请求状态与 slot 独占模型：重复提交按完整内容哈希去重，
只允许白名单状态转换，取消与 quarantine 是显式状态；进入 `PREPARING` 前验证 receipt 结构、哈希和
请求身份。通用 `transition(PREPARING -> RUNNING)` 仍始终 fail closed；运行只能经过后述受信 attempt-start
收据。内部 `projectctl remote-runner prepare|stage-source|stage-readiness|stage-service-spec|stage-service-plan|stage-service-receipt|status|cancel`
已暴露这段持久控制面，
但不是日常 `lab` 用户命令，也不会把手工
状态转换当成测试证据。`cancel` 对 active request 只写入 `CANCEL_REQUESTED`；它不宣称进程已 teardown，
更不宣称设备已安全停止。对已经预留资源的 active request，通用状态转换不再允许直接写
`CANCELLED`；只有 `ack-cancel` 验证受信 executor 的 strict hashed teardown receipt，证明登记的进程组
已终止、managed service 已不存在且 GPU lease 已释放后才完成取消。receipt 固定
`device_access=false`、`hardware_safety_claimed=false`，因此它不能被误作机器人 HOLD/E-stop 或安全停止
证据；尚未预留任何资源的 `SUBMITTED` request 仍可直接无副作用取消。

`remote-runner watch` 以 append-only history sequence 为断线恢复游标；它只读等待状态推进、terminal
状态或最长 60 秒超时，不保持 SSH session 身份、不触发执行，也不改变 request。它是后续用户层
`--follow|--detach`/`runs watch` 的 runner 原语，当前还不是远端消息通道。

`prepare_remote_request` 已把离线 `SUBMITTED -> RESERVED -> PREPARING` 组合成可重放生命周期：成功只
停在 `PREPARING`，绝不执行 Profile command；预检失败记录 `BLOCKED_PRECONDITION` failure receipt 并
进入 terminal `FAILED`，从而释放 slot。相同 request 重放返回原状态，不产生第二次准备或覆盖失败。

`stage-source` 只接受已有 preflight receipt 的 `PREPARING` request，在 host Profile 指定 slot 中从 clean
anchors 创建 detached exact commit/tree/submodule checkout；LFS smudge/fetch 禁用，anchor fingerprint
前后不变。source receipt 与 request、slot、host/remote Profile 和 control-plane hash 绑定并写入状态，
同时明确 `environment_prepared=false`、`assets_prepared=false`、`service_started=false` 和
`execution_eligible=false`。这一步不会启动命令或把状态推进到 `RUNNING`。

`stage-readiness` 再次校验 observation/preflight 绑定，并从 host Profile 的 immutable environment 启动
`-I` 探针；first-party source roots 由 protected Profile 显式插入，所有 import origins 必须位于刚才
staged 的 detached repositories，不能来自 anchor、用户 site 或共享 editable install。asset 只接受
preflight 已验证且位于 protected asset roots 的逻辑 ID/checksum/realpath；为避免无意义读取大型 checkpoint，
本阶段不重新哈希内容，并在 receipt 中明确 `content_rehashed=false`。readiness receipt 把 source、
preflight、observation、GPU lease、environment 和 assets 绑定起来，但仍保持 `service_started=false`、
`execution_eligible=false` 和 `device_access=false`。

remote Profile 现在还包含受保护 `service` policy：service repository/environment、protocol、
launch mode、受信 command adapter ID 和 endpoint scheme 均由 protected Profile 决定，Task request
不能注入 argv、shell、固定端口或扩大模式。`service-identity launch-spec` 把 exact openpi commit/tree/
submodules、readiness environment hash、全部 asset ID/checksum、GPU lease/UUID 和一次性 launcher nonce
绑定成 content-addressed inert spec；它仍明确 `execution_eligible=false`，不会启动进程。

受信 launcher 在启动服务前还必须生成独立的 service-sandbox receipt：它把 exact launch spec、短时有效的
sandbox backend probe、staged source、immutable environment、asset realpath、GPU lease 和 per-run runtime
目录绑定到固定只读/可写挂载表；网络仅允许新的 loopback namespace，禁止下载和设备访问。launcher
启动后直接从 `/proc` 采集 managed-process receipt，证明该 PID 是 launcher 的直接子进程，PID/进程组/
session 一致，user/network/PID/mount namespace 与 launcher 隔离，`NoNewPrivs=1`，可执行文件来自 Profile
指定环境，并记录 kernel start ticks。采集前后 PID identity 必须稳定。

service-sandbox receipt 还可由受信控制面编译成 inert service-launch-plan。计划只接受安装版受保护
adapter manifest，第一项固定为 sandbox 内 environment Python，模板只能展开动态 loopback port、
staged source、逻辑 asset 和 per-run runtime 路径；shell、proxy、`PYTHONPATH/LD_PRELOAD`、Task 环境继承
和未解析 host token 全部拒绝。生成的是继承受信外层网络边界的 inner bubblewrap argv：它固定
user/PID namespace、`clearenv`、
只读 system/source/environment/assets、唯一可写 runtime、Profile GPU device allowlist 和 offline 环境。
真正 launch 前会重新逐项比较 ro-bind/bind/dev-bind/setenv 与 live sandbox receipt，因而即使攻击者重算
JSON 哈希也不能追加 `/dev/nvidia9` 或其他挂载。通用 `lab_control` 不包含 OpenPI、模型概念、入口脚本
或投影路径；当前项目的 production adapter 位于受保护
`integration/service_adapters/openpi-policy-service.json`，其原始 manifest SHA-256 会进入 plan 身份。
该项目清单固定到 openpi
`0d2d7f4e...` 中已跟踪的 `scripts/serve_dexteleop_policy.py`：Host Profile 同时只读挂载 repository root、
`src` 与 `packages/openpi-client/src`，受保护 `python -I` bootstrap 显式把后两者放到 import 前端；model、
registry、checkpoint、host 和动态 port 都来自已验证 token，不能由 Task 追加 argv。OpenPI 实际从
`checkpoint/assets/<asset_id>/norm_stats.json` 读取归一化数据，因此 adapter 还把经过 runtime receipt
重哈希的 `openpi-norm-stats` 文件，只读投影到当前唯一受支持模型
`dual_rm65_dmgripper_latest` 的固定目标 `checkpoint/assets/dual_rm65_dmgripper/norm_stats.json`。model allowlist、
投影源 role、目标 checkpoint role 和相对路径都是安装版项目 adapter 身份的一部分；真正 launch 前使用同一
受保护 adapter 重新计算额外 `ro-bind`，不能靠修改并重算 plan JSON 把另一文件伪装成 norm stats。
项目 adapter 可将通用 `workload_id` 映射为 OpenPI 的模型参数；这一解释不进入框架核心。普通服务 adapter
可以完全不声明该字段；若模板或 allowlist 需要它而 lock 为 null，则在计划生成时 fail closed。注册 adapter
仍不等于执行：runner 先通过通用持久 `EndpointLeaseStore` 为 `127.0.0.1:<port>` 建立框架内端口
互斥，再生成项目无关的 `shared-isolated-loopback` network plan。该 plan 要求 service 与 attempt 最终
加入同一个受信 anchor namespace，而不是分别 `--unshare-net`；分别创建的 loopback 互不可见。network
plan 固定 `namespace_created=false`、`service_joined=false`、`attempt_joined=false` 和
`launch_authorized=false`。service plan 同时绑定 endpoint lease 与 network plan SHA-256，并固定
`endpoint_namespace_reserved=true`、`outer_network_namespace_required=true`、
`outer_network_namespace_joined=false`、`direct_launcher_execution_permitted=false`。端口租约不绑定 socket，
network plan 也不创建 namespace 或探测网络，因此 service plan 继续固定
`process_started=false`、`launch_authorized=false`、
`endpoint_reservation_required=true`；远端 source anchor、workload selection、artifact manifest、环境和
live sandbox receipt 任一未就绪都会阻塞后续授权。

当前还实现了项目无关的 anchor/member OS observation collector，但没有把它提升为执行收据。anchor
observation 要求 runner 直接子进程、独立 process group/session、`NoNewPrivs=1`，并在两次 PID stat 与
netns inode 读取之间保持稳定；member observation 只接受 `service`/`attempt` 角色，且其 netns inode
必须与同 request/slot/host/plan 的 anchor 完全相同。二者都固定 loopback/外网隔离未验证、连接与启动
未授权。下一道边界是由受信控制面进入该 netns 执行固定 loopback/route 探针；在此之前现有
`execution_eligible=false` 的 identity-only ServiceReceipt 不能被提升为“可连接服务”证据，也不能为真实
attempt 提供运行授权（当前 attempt 路径仍只是注入式 hermetic 测试边）。

固定 in-namespace probe 原语现已补齐并接入受信 runner library staging helper。受保护 policy 提供 namespace-enter 与
Python executable 的绝对路径；collector 拒绝 symlink/非普通文件/不可执行文件，散列后通过已打开 FD
执行，防止路径在检查与执行之间被替换。命令没有 Task 参数，固定 cwd 为 `/`、stdin 为 `/dev/null`、
环境仅含最小系统变量且有 5 秒超时。探针不会访问外部目标，只在 anchor netns 内绑定临时 loopback
socket、列举接口并读取 IPv4/IPv6 route。只有 namespace inode 一致、接口仅 `lo` 且不存在非 loopback
route 才生成短时 receipt；使用前还会重新散列后端并验证 anchor PID/start-ticks/netns/`NoNewPrivs`。
receipt 的 `network_ready=true` 只代表网络前置条件，通过后仍是 `launch_authorized=false`。
helper 只消费 launcher 已建立的 anchor PID，将 anchor observation 与 probe receipt 原子写入
request state v8；相同锚点重放会对 live `/proc`、TTL 和 executable hash 重新复验，不同锚点或半份证据
会拒绝。它没有创建 namespace/member、没有启动服务，也尚未暴露为远端部署 CLI。

`verify-health` 不再仅信任服务自报的 PID 或固定端口。它会重新检查 sandbox 的 live paths，并要求服务
health 中的 endpoint、PID 和 started-at 与独立 managed-process receipt 完全一致；source/environment/
assets/GPU/nonce/command adapter 任一不符也会拒绝。通过后生成 ServiceReceipt v2，其中绑定 process
receipt SHA-256 和 kernel start ticks，但仍不授予执行或设备权限。`verify-existing` 只对 Profile 明确允许的
`attach-existing` 模式生效，并要求既有 managed receipt 的上述身份逐项一致；`unmanaged`、旧进程、
仅端口可达或 tmux/PID 存在均不能成为可连接服务。

停止侧使用对称的 managed-service teardown receipt。受信 launcher 必须先按反序停止当前 Run 登记的
节点，再由创建服务的同一个父进程执行非阻塞 `waitpid` 并扫描 `/proc`：原 PID 的 kernel start ticks
不能仍存在，且整个登记进程组必须为空；PID 已被系统复用会单独记录而不是误认成旧服务。receipt 记录
exit/signal 状态并固定 `device_access=false`、`hardware_safety_claimed=false`。当前该 collector 已通过
fake `/proc` 测试，但尚未与远端 launcher 和 cancellation/GPU release 形成真实执行闭环，因此不能把
本地 schema 通过误写为远端 teardown 已完成。

`remote-runner stage-service-spec` 将同一个 inert spec 幂等附加到 `PREPARING` request state，且必须
已经有相同 request/slot/Profile 的 readiness receipt；换 nonce 重放会被视为不同 launch intent 并拒绝
覆盖。`stage-service-plan` 从安装版受保护 adapter manifest、live sandbox probe、readiness/spec 和
runner 提供的动态端口编译 plan。它先取得持久 endpoint lease，再生成共享隔离 network plan，把 lease、
network plan、service-sandbox receipt 与 service plan 原子写入 request state v8；相同输入幂等，端口、
network plan、adapter manifest 或 receipt 漂移均拒绝。过期 lease 不会被静默抢占，只能以
`process_started=false/socket_bound=false` 收据显式回收。该 v4 service plan 尚未创建或加入真实 network
namespace、绑定 socket或取得 launcher authorization，因此不能启动服务。`stage-service-receipt` 只有在上述 plan 已持久化且实际 endpoint 完全
一致时，才能进一步附加 launcher OS attestation 与 health identity 联合生成的严格 ServiceReceipt；它
仍不执行命令，且 ServiceReceipt 自身不授予执行权。

本地 hermetic trusted-execution edge 已实现为受保护库接口而非用户 CLI。只有 Profile 白名单 command ID、
exact request/slot/Profile、ServiceReceipt、GPU lease、enforced sandbox receipt 和 executor nonce 全部绑定的 attempt-start receipt，
才能通过专用 `begin_attempt` 进入 `RUNNING`；通用 transition 不能绕过。受信 adapter 不接收 Task argv、
cwd 或环境。`integration/command_registries/agenticrobot-openpi-gpu.json` 进一步要求 command ID 解析为唯一
adapter ID，并将 registry SHA-256、adapter ID、`loopback-only` 和 `downloads=false` 绑定进 attempt；registry
采用严格字段白名单，出现 argv/cwd/env/shell 或放宽权限都会拒绝。单次 attempt 完成后先进入 `COLLECTING`，artifact 必须是 attempt root 内非 symlink 相对文件，
总量上限 16 MiB，并记录 size/SHA-256；最终明确区分 `PASSED`、`FAILED_ASSERTION` 和
`INFRASTRUCTURE_FAILURE`，将 `execution_completed`、`service_acknowledged`、`assertions_passed`、
`task_success`、`evidence_complete` 分开。全部收据保持 `device_access=false`。该边目前只用注入式本地
adapter 验证，没有远端 launcher/command registry，因此没有在真实主机执行。attempt-start v2 现在还
强制绑定 request state v8 中的 execution-sandbox receipt 和 network evidence；receipt 只能接受受保护 `bubblewrap-v1` 探针，
要求 user/network/process namespace、loopback 可用、外网不可达、filesystem isolation、`no_new_privileges`
以及 forbidden devices absent 全部成立，并且 device allowlist 只能出现 NVIDIA 节点。缺失、篡改或包含
`/dev/video*`/串口等节点时，在 adapter 调用前拒绝。受信 `projectctl sandbox-probe` 现已实现固定
`bubblewrap` capability collector：后端绝对路径与 GPU 节点来自 Host Profile，后端必须是非 symlink、
有大小上限的可执行普通文件并记录 SHA-256；探针不接收 Task argv/env/cwd，不绑定 `/home`、`/root`、
`/mnt`，以新 user/network/process namespace、空网络和最小只读系统树运行固定 Python 探针。receipt 还
要求配置的 GPU character devices 在沙箱内可见、相机/串口/input 不可见，并只在 Host Profile 规定的
短有效期内可生成 execution-sandbox receipt。`attach_execution_sandbox` 与 attempt 启动边都会重新检查
该 TTL；过期收据仍能作为历史状态读取/取消，但不能再取得 launch authority。该 collector 只通过本机
fake-backend hermetic 测试，尚未
在远端实际运行，所以它不会把“二进制存在”误作可执行授权。

这些仍是 hermetic schema/policy bootstrap，不是已部署执行入口。远端现已在新局域网地址
`user@192.168.70.116` 恢复，用户随后确认互联网连接也已恢复；测试 payload 仍由受保护 Profile 固定为
`loopback-only` 且禁止下载。本轮只读审计后仍因 protected control-plane 未部署、planned
slot/独立环境不存在、checkpoint 未形成完整 content receipt、GPU lease 尚未在远端真实预留而 fail closed。
受信 GPU inventory、持久 lease 与 cleanup-gated release 原语及内部 `projectctl gpu-lease` 已通过本机
临时目录测试；lease 与 host Profile、request、slot、GPU UUID 绑定，过期后不会被自动抢占，必须先由
受信清理流程核对。`gpu-lease reserve` 只接受已经由 inbox 持久化的 request，并在 request-store 锁内
组合预留 slot 与 GPU；request state v3 内嵌 strict resource-reservation receipt。preflight 必须消费同一
lease ID/hash/UUID，不能在 slot reservation 与 GPU reservation 之间换资源。预留前还使用固定
`nvidia-smi --query-compute-apps` 只读查询生成 GPU process receipt；目标 UUID 上存在任何既有 compute
process 时，在写 lease 之前拒绝，且不发送 signal。主机在 GPU 状态写入后、
request state 写入前崩溃时，lease 保持 fail closed，等待受信 reconciliation，绝不自动抢占。
`reconcile-nonstarted` 只在 request state 没有 launch spec、ServiceReceipt、active attempt 或历史 attempt，
且 request 已停止（或仍为 crash-orphan 的 `SUBMITTED`）时工作；它重新采集不超过 30 秒的无 compute-process
证据，释放 lease，并把 release receipt 回写 state v3。任何可能已启动服务的状态都必须走后续完整 teardown，
不能用该捷径释放。
`host-runtime-receipts` 也已改为严格哈希收据：必须嵌入完整 active lease，capability
必须与 host Profile 完全相同，asset placement 必须位于受保护 asset root，且始终声明 execution、device、
network 权限均为 false。这些能力尚未部署，也没有在真实 GPU 上 reserve。request inbox/pinned
SSH/status-watch-cancel 合同已在 fake transport 上实现；受保护 command registry 已落地，但真实 adapter/
launcher、slot/GPU lease、
artifact 回传和用户层 `lab test --follow|--detach` 仍未实现。没有启动 OpenPI、探测服务端口或访问设备。

OpenPI 当前服务实现是 WebSocket，因此受保护远端 Profile 的动态 loopback endpoint 已从错误的 `http`
修正为 `ws`；service spec/health/receipt validator 同时支持 `ws/wss`，避免“身份收据通过但客户端协议无法
连接”的假阳性。端口仍由每个 Run 动态分配，不能退回固定 `8078`。

15:40 的新地址 L1 复核确认远端实际存在 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-modeset`、
`/dev/nvidia-uvm`、`/dev/nvidia-uvm-tools` 和两个 render node。当前 compute-only Host Profile 只允许
`nvidia0`、`nvidiactl`、`nvidia-uvm`、`nvidia-uvm-tools`；不会因为主机上同时存在 camera、serial、input、
hidraw 或 USB 节点而把它们传入沙箱。该复核只列目录，没有运行 bwrap、CUDA、模型或设备。

不要用手工 SSH、rsync 或设备命令代替这些尚未落地的受控路径。

### Clean source anchor 的本地构建原语

`lab_control.source_anchor.materialize_local_source_anchor` 已以 library-only 形式实现并通过临时 Git 仓库
测试。它只从已存在、clean、canonical identity 匹配的本地 clone 取对象，创建独立 detached clone；
关闭 hooks、终端提示与 LFS smudge，不 fetch、不访问网络，并要求全部 submodule 由显式本地 source 在
exact gitlink 上初始化。checkout 前会扫描 Git tree，拒绝超过 protected 16 MiB 单文件上限的普通 tracked
blob；LFS pointer 可作为小型源码条目留下，但不会下载其内容。receipt 记录 commit/tree/submodules、
`network_used=false`、`large_file_downloaded=false` 与完整哈希。

该原语现仅通过内部受信入口
`projectctl trusted-prepare source-anchor --preparation-profile ... --host-profile ... --repository openpi --commit <OID>`
暴露：源/目标路径、submodule source 和 blob 上限全部来自仓库内受保护 Preparation/Host Profile，CLI
不能传任意路径或网络源。它仍没有部署到 `192.168.70.116`，也没有写远端目录；必须先把当前
control-plane 形成受保护 commit并核验 transport，才能建立
`/home/user/zc/robot-stack/dev/openpi`，不能借助任意远端 shell 绕过。

### 稳定管理 clone 的 commit-based 同步原语

`lab_control.stable_sync` 已实现不联网的两阶段同步：`plan_stable_checkout_sync` 只比较当前 clean checkout
与已经 fetch 到本机的 remote-tracking ref；只有 canonical identity、受保护 branch、HEAD/tree、dirty
状态和严格祖先关系全部满足时才返回 `FAST_FORWARD_READY`。`apply_stable_checkout_fast_forward` 会先重算
并逐项比对原 plan，再执行禁 hooks 的 `merge --ff-only`，最后生成带 SHA-256 的 receipt。

该原语不会自行 fetch、pull/rebase/reset/clean，也不复制 working-tree 文件；`dirty`、wrong branch、缺少 ref、
local-ahead 或 diverged 都 fail closed。fetch 是独立的受信准备动作并固定设置
`GIT_LFS_SKIP_SMUDGE=1`。它尚未接入远端 scheduler，因此当前不能声称两机已自动同步；但已经把“自动
同步”的可变部分收窄为 `explicit fetch → immutable plan → guarded ff-only apply`，Task worktree 永不参与。

`lab_control.trusted_fetch.fetch_stable_remote_ref` 现已补齐上述 explicit fetch 原语：只允许 canonical
identity 匹配、clean 且位于受保护 branch 的稳定 checkout，固定单一 `origin` branch refspec，禁 tags、
submodule recursion、FETCH_HEAD 写入、hooks、auto-gc、交互提示和 LFS smudge。fetch 后必须证明 branch、
HEAD、tree 和 worktree 完全未变，receipt 才会记录 remote ref 前后 commit。它诚实记录
`network_used=true/source_objects_fetched=true`，同时记录 `checkout_updated=false`、
`submodules_fetched=false`、`large_assets_downloaded=false` 和 `task_payload_authorized=false`。

因此互联网恢复只开放这一条未来可审计的 control-plane 准备边，不改变测试 Profile 的
`loopback-only/downloads=false`。内部入口现提供：

```bash
projectctl trusted-prepare status --preparation-profile <id> --host-profile <id> --repository AgenticRobot
projectctl trusted-prepare sync --preparation-profile <id> --host-profile <id> --repository AgenticRobot
```

`status` 完全只读且不联网；`sync` 只做受限 fetch。若结果为 `FAST_FORWARD_READY` 但没有最多 60 秒有效、
绑定 Profile/checkout/branch/当前 HEAD 的 maintenance receipt，则只返回 `WAITING_FOR_MAINTENANCE`，绝不
更新稳定目录。receipt 必须声明 runtime dependency inventory 完整、checkout 无依赖进程，且明确不授予
device 或 motion safety 权限；被篡改、过期或 HEAD 已变化都会拒绝。带合格 receipt 时仍会在 apply 前重算
完整 plan，只执行 `merge --ff-only`。maintenance inventory 的实际远端部署和 scheduler 尚未实现，CLI
也没有生成 receipt 的人工绕过选项。

`projectctl trusted-prepare runtime-inventory` 现提供只读采集器：路径只能来自 Preparation Profile，扫描
checkout owner 的 `/proc` 活动进程，检查 cwd、exe、cmdline、environ 和 maps 对 checkout 的引用；外 UID
进程只计数，不被误称为已扫描。owner metadata 不可读会使 inventory incomplete，receipt 最长只接受
30 秒，并记录扫描 scope、依赖 PID/evidence、跳过/消失计数、`processes_modified=false` 和
`device_access=false`。maintenance receipt 必须嵌入该 inventory 且 dependencies 为空，同时绑定一个
`runtime_start_blocked=true` 的 scheduler reservation ID，以堵住扫描到 apply 之间重新启动 runtime 的
竞态。现在该 reservation 不再是调用方布尔声明：受保护 control-plane root 下有带 `flock` 的持久状态，
`acquire` 会绑定 Profile/repository/checkout HEAD 和最长 60 秒期限，受管 launcher 必须调用同一
`managed-runtime-start-gate`，ACTIVE 时拒绝启动；同步成功或异常路径都会尝试写入 RELEASED receipt。

`projectctl trusted-prepare auto-sync --reservation-id <id>` 已组合为：acquire gate → `/proc` inventory →
构造短时 maintenance receipt → explicit small Git fetch → 重验 plan → 可选 `ff-only` → release。dirty、
wrong branch、活动依赖、不可读 owner PID、fetch/apply 失败都 fail closed，且异常测试证明 gate 会释放；
Task payload/device 始终不获权。真实远端仍未部署该入口，而且 legacy/unmanaged launcher 必须先完成 M0
迁移并被禁止直接从稳定 clone 启动，不能把“受管 gate 已实现”误解为可阻止任意手工进程。

### 受信 control-plane 小型部署包

`projectctl control-plane package --preparation-profile <id> --host-profile <id> --output <tar>` 已提供本地
打包合同。canonical repository identity 只能来自受保护 Preparation Profile；builder 还要求当前 checkout
为 clean `main`、`HEAD` 与 `refs/remotes/origin/main` 完全相同。包内容固定为 `projectctl`、`lab_control/`
以及 command/host/preparation/remote Profiles、小型 asset metadata 和受审
`stack/workload-selections/` manifest；tests、Task worktree、大型制品、数据、环境和其他路径不会进入。
workload selection 文件只提供版本化授权依据，不包含制品内容，也不能在打包时自动选择或激活项目
workload。单文件上限
1 MiB、总源码 8 MiB、tar 16 MiB，并记录每个 blob OID、大小和
SHA-256 及 archive SHA-256。receipt 明确 `execution_authorized=false`、`device_access=false`、未联网、
未用 working-tree 内容、未启用 LFS smudge。

package receipt 使用可移植 archive 文件名而非本机绝对路径，并有自身 SHA-256。受信 verifier 要求调用方
另行提供预期 canonical identity 和 commit，先复核 receipt，再核对 tar 大小/整体哈希和每个 regular file
的白名单路径、mode、size、SHA-256；路径穿越、symlink/hardlink、重复/额外/缺失文件都会拒绝。验证结果
仍明确 `extracted=false`、`executed=false`，因此传输完成也不等于已安装。

当前框架代码已形成并普通 push 到 task branch，但仍不位于 clean protected `main == origin/main`，因此
此命令在真实仓库会按设计拒绝，不能把开发中控制面伪装成受保护部署版本。后续必须先 review/push/merge
到 protected `origin/main`，再生成包；安全解包、远端原子
激活和 rollback 尚未实现，不能手工 scp/tar 替代。

Preparation Profile 现固定 control-plane install root 为
`/home/user/zc/robot-stack/control-plane`、`current` 链接和最多保留 3 个版本；transport 的未来
`remote_projectctl` 也已改为 `.../control-plane/current/projectctl`，不再指向可被普通源码同步改变的稳定
clone。`projectctl control-plane stage` 会先执行上述完整验证，再逐文件写入
`versions/<commit>` 的临时目录并原子改名，同时保存 package/verification/stage receipts。它拒绝 symlink
install root、既有同名版本和任何校验差异；结果固定 `active=false/current_link_updated=false/executed=false`。
`projectctl control-plane activate` 现会重新验证 staged tree，从 Host Profile 指定的
`agenticrobot-py310` 独立环境以 isolated Python 运行受信 `projectctl --help`，禁 bytecode/user-site 和
下载权限，然后用相对 symlink 原子更新 `current`。activation receipt 绑定两份 Profile、stage/tree
verification、Python、stdout/stderr hash、前一 commit，并明确只执行 control-plane health check、没有执行
Task payload 或访问设备。`rollback` 只允许从当前 activation receipt 返回其 previous commit，先对旧版本
重复完整 tree verification 和 health check，再原子切换；不会删除新旧版本。

当前尚未在远端创建 install root、传包、staging、activation 或 rollback；版本回收也刻意未实现。因此
这里是经过 hermetic 测试的部署合同，而非可调用的真实远端 executor。首次真实激活还必须等待 protected
commit/package、独立核验的 SSH pin，以及计划中的 `agenticrobot-py310` 环境存在。
