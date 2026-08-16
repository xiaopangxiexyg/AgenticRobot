# `lab` M6a：受认证的源码与结果包

状态：`task/AR-lab-framework-v2` 中的 review candidate。本切片补齐跨主机边界两侧的可移植数据合同；它不连接
真实远端，不实现 SSH/文件复制 daemon，也不运行真实项目命令。

## 第一部分：一般项目框架

### 边界与目标

M6a 只传递四类通用内容：

```text
immutable Run Request
exact Git objects
validated Run event/receipt
registered bounded artifact
```

活动 worktree、`.git/worktrees`、虚拟环境、build 目录和未登记输出都不进入传输包。传输包是普通目录，可以由外部
`scp`、受控 SSH adapter、共享目录或其它可靠通道搬运；`lab_control` 不选择地址，也不解释系统命令需要什么运行
条件。

### 源码包

控制端执行：

```bash
chmod 600 /path/to/transport.key

lab transfer source-export Q-... /path/to/outgoing/source-Q-... \
  --base Controller=<receiver-known-commit> \
  --base Component=<receiver-known-commit> \
  --sender-id controller-a \
  --recipient-id runner-b \
  --key-id transport-key-1 \
  --key-file /path/to/transport.key
```

每个 `--base` 是接收端已经拥有的 commit。构建器验证它是目标 commit 的祖先，只生成
`target ^base` 缺少的原生 Git objects；目标已存在时不生成 pack。默认边界为：单 blob 16 MiB、单仓 raw objects
128 MiB、pack 64 MiB、200000 objects、32 repositories。LFS smudge、checkout filter、hook 和网络全部禁用。

接收端执行：

```bash
lab transfer source-ingest /path/to/incoming/source-Q-... \
  --sender-id controller-a \
  --recipient-id runner-b \
  --key-id transport-key-1 \
  --key-file /path/to/transport.key
```

导入顺序为：验证 package HMAC 与 hash、验证内含 Request envelope、验证仓库 canonical origin 和 receiver base、校验
pack 大小/hash、`git index-pack --strict`、重新验证 commit/tree/gitlink，最后才把原始 Request 幂等写入 durable store。
导入只增加 Git objects，不移动 ref、branch、index 或 working tree。之后沿用：

```bash
lab runner claim <runner-id> <slot-id>
lab runner stage <runner-id> <request-id>
lab runner execute <runner-id> <request-id>
```

### 结果包

Runner 的 Request 进入终态后执行：

```bash
lab transfer result-export Q-... /path/to/outgoing/result-Q-... \
  --sender-id runner-b \
  --recipient-id controller-a \
  --key-id transport-key-1 \
  --key-file /path/to/transport.key
```

结果 evidence 包含 immutable Request、完整且逐条验证的 event sequence、远端终态以及已登记的 runner claim、stage、
execution、artifact receipts。只有 artifact receipt 中登记且 hash/size 仍匹配的文件会被复制；默认单文件 16 MiB、
合计 64 MiB。超出边界的文件会 fail closed，不通过本框架复制；若要保留大文件，后续应由独立 artifact store
adapter 处理并返回其自身的内容 receipt，而不是放宽代码管理 transport 的默认上限。

控制端执行：

```bash
lab transfer result-ingest /path/to/incoming/result-Q-... \
  --sender-id runner-b \
  --recipient-id controller-a \
  --key-id transport-key-1 \
  --key-file /path/to/transport.key
```

接收端先验证 HMAC、Request immutable content、事件/receipt hash 和全部 artifact，再原子保存结果并向原 Request 添加
`result-*` attachment。它不会把控制端的 `SUBMITTED` 状态伪造成远端 `COMPLETED`；远端终态是导入 evidence，调度
状态仍由各自 durable store 独立维护。重复导入相同 package 返回原 receipt，不重复复制或覆盖。

### 信任和未实现内容

- HMAC key 必须至少 32 bytes；CLI 只读取非 symlink、最大 4096 bytes、group/other 无权限的 key file。
- Request envelope 有有效期和对端身份；result package 是不可变历史 evidence，不因传输延迟过期。
- package 完整性不能代替传输可用性。M6a 不负责重试、断点续传、远端目录管理或进程常驻。
- 接收端必须事先拥有明确 receiver base。自动协商 object inventory 属于后续 host transport adapter。
- Git object 导入和结果保存均不执行 package 内代码。
- 项目命令仍是不透明系统命令；源码/结果传输 schema 不包含环境、设备、服务或项目模型语义。

## 第二部分：当前项目映射

AgenticRobot、RPent 与 `model/openpi` 只表现为通用 repository name、commit、tree 和 gitlink。父仓与 openpi 的固定关系
来自已有 snapshot `repository_links`；M6a 不为 openpi 模型、运行环境或其它内容增加专用字段。

本机和远端的实际复制暂不启用。后续 host adapter 应只完成：

```text
本机 source-export
  -> 外部受控复制
  -> 远端 source-ingest / runner execute / result-export
  -> 外部受控复制
  -> 本机 result-ingest
```

远端地址只放在 host-local adapter 配置。当前实现没有登录 `user@192.168.70.116`，没有修改远端 checkout，也没有触发
任何实际测试命令。

## 验证

```bash
./dev test lab-m6
./dev test lab-control
python -m pytest -q tests/test_lab_m6_source_transfer.py tests/test_lab_m6_result_transfer.py
ruff check lab_control tests/test_lab_m6_*.py
mypy lab_control
git diff --check
```

测试使用临时通用仓库、临时 key、临时 Runner 和无副作用 Python 命令，覆盖 exact object 导入后 staging、present 模式、
篡改拒绝、receiver base、blob/artifact 上限、终态 evidence 回收、幂等导入和错误对端/密钥。
