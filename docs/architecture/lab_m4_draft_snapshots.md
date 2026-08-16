# `lab` M4：Synthetic Git draft snapshot

状态：`task/AR-lab-framework-v2` 中的 review candidate。本切片实现项目无关的 dirty source capture 与 durable
Request 集成；它不传输 Git objects、不执行项目命令，也不授予网络、GPU、服务或设备权限。

## 第一部分：一般项目的管理框架

### 用户入口与授权边界

快速远端或长测试的草稿入口是：

```bash
lab test <task-id> <queue-profile> --detach --draft
```

非 ignored 的 untracked 源文件默认不进入快照，必须逐个显式声明：

```bash
lab test T101 contract-check --detach --draft \
  --include-untracked Controller=src/new_adapter.py
```

受保护 Queue Profile 必须使用 `lab.queue-profile.v2`，同时声明：

```json
{
  "snapshot_modes": ["clean", "draft"],
  "execution_class": "offline"
}
```

Draft 只允许 `offline`、`simulation` 或 `compute` execution class，只产生 Development evidence。`verify`、Candidate/
Release evidence、`observe` 和 `control` 均拒绝 draft。旧版 Queue Profile 自动保持 clean-only，不会因框架升级而隐式
扩大权限。

### Synthetic commit，而不是 patch overlay

每个 Task 仓库使用独立临时 Git index，从 Task `HEAD` 开始构造一棵准确源码树：

```text
HEAD tree
+ 当前 working tree 中所有 tracked 文件的实际内容、模式和删除
+ 显式 allowlist 的 non-ignored untracked 文件
+ 已解析的 child synthetic commit 对应父仓 gitlink
```

随后通过 Git plumbing 生成 tree 和 synthetic commit，不移动当前 branch、`HEAD`、真实 index 或 working tree。Git
原生 object 模型保留 binary blob、executable bit、symlink 和 gitlink；不自行解释 rename/copy，也不运行 checkout
filter、hook 或 LFS smudge。

只有 tree 与 `HEAD` 不同时才创建 synthetic commit；无内容变化的仓库直接使用 base commit。Commit 使用稳定身份、
parent 时间和确定消息，使同一输入得到相同 OID。快照发布到本机名字空间：

```text
refs/lab/snapshots/<snapshot-id>/<repository>
```

这些 ref 不属于用户分支，不能 push 到普通 Task branch，也不是合并候选。

### 多仓一致性

多仓 capture 采用短时 Workspace snapshot barrier，并在 barrier 前后核对每个仓库的 branch、commit、status 和 index
哈希。嵌套仓按 child-first 顺序生成快照；子仓 synthetic commit 被写入父仓临时 tree 的 gitlink。最终 Request 同时
校验：

```text
parent.submodules[path] == repository_link.commit == child.commit
```

这保证一次 draft Run 表示一组原子的多仓源码组合，而不是若干互不匹配的 dirty 目录。

### 安全与大小限制

- untracked 必须逐文件 allowlist，且必须在 Task write scope 内；ignored 文件不会被自动发现或纳入；
- `.env`、private key、credential/token 等敏感命名和常见密钥后缀 fail closed；
- 绝对或逃逸 symlink、特殊文件、冲突 index、非 UTF-8/逃逸路径拒绝；
- dirty linked repository 若未作为 Task 子仓参与同一快照则拒绝，不能静默保留旧 gitlink；
- 每个 untracked 集合、单 blob 和总捕获字节数有硬上限；
- Request 记录 base commit、synthetic marker、tree/commit、included/excluded untracked 路径和 capture hash；
- staging 仍只从 runner 已有的 Git objects 物化 detached clean slot，并再次校验 exact identity。

M4 证明了 draft 可以转化为 Git 原生、可定位且可复验的源码身份，但尚未实现跨机 object transport。若 runner 不含
synthetic objects，M3h stager 必须失败；不能隐式 fetch、复制 working tree 或退回 patch 应用。

### 明确未实现

M4 不提供：

- SSH/API/消息队列 adapter 或 synthetic object bundle 传输；
- Profile command/environment executor；
- GPU、service、device、physical cell 或 operator lease；
- ServiceReceipt、分布式 readiness barrier、fixture、Attempt 或 authority handoff；
- draft verify、设备 observe/control 或 release evidence；
- snapshot ref retention/garbage-collection policy。

因此它不能被描述为“dirty 代码已经可以在真实远端自动测试”。当前交付的是安全、幂等的本地 capture → Request →
exact staging 协议闭环。

## 第二部分：当前项目实例

当前项目只将仓库拓扑映射到通用机制：AgenticRobot 是父仓，`model/openpi` 是 gitlink child，RPent 是 sibling。
框架核心不包含模型、checkpoint、GPU、机器人、IP、绝对路径或项目命令。

当一个 Task 同时修改 AgenticRobot 与 openpi 时，M4 的 child-first capture 会先固定 openpi synthetic commit，再生成
引用该 commit 的 AgenticRobot synthetic parent。RPent 若参与同一 Task，则作为另一个独立仓库进入同一 snapshot
identity。未修改且不在 Task 中的仓库仍由 protected baseline 补齐。

真实 `user@192.168.70.116` 仍不在本切片接入。后续若实现跨机 draft 测试，应先设计经过认证、受限大小、无 LFS/
大文件的 Git object bundle 传输，再在无 GPU、无设备的临时 slot 验证；不得 rsync 任一活动 Workspace，也不得把
openpi 的模型、环境、checkpoint、数据或日志放入 synthetic source snapshot。

## 验证

```bash
python -m pytest -q tests/test_lab_m4_draft_snapshots.py
./dev test lab-m4
ruff check lab_control tests/test_lab_m4_draft_snapshots.py
mypy lab_control
git diff --check
```

测试仅使用临时 `Controller`/`Component` Git 仓库，覆盖 working-tree/index 不变、实际 tracked 内容与删除、显式
untracked allowlist、敏感路径拒绝、幂等 Request、detached exact staging，以及 child synthetic commit 与 parent
gitlink 的一致性；不访问网络、大文件、GPU 或设备。
