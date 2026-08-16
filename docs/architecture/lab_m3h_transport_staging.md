# `lab` M3h：受认证 Request envelope 与 exact snapshot staging

状态：`task/AR-lab-framework-v2` 中的 review candidate。本切片只完成项目无关的 transport/staging 管理协议，
不连接真实远端、不执行测试命令，也不分配网络、GPU、服务或设备权限。

## 第一部分：一般项目的管理框架

### 受认证、可过期、幂等的 Request envelope

`lab_control.request_transport` 将 M3f 的不可变 Run Request 封装为 `lab.request-envelope.v1`。Envelope 固定：

- sender/recipient 逻辑 ID；
- issued/expiry 时间窗；
- Request ID、idempotency key 和 payload SHA-256；
- envelope content SHA-256；
- 外部预置 key ID 与 HMAC-SHA256。

共享密钥必须由 transport adapter 的 host-local secret provider 提供，不进入 Git、Task、Request 或日志。Ingress
只信任受保护配置给出的 sender/key 映射，验证 canonical payload、接收者、时钟偏差、最长 24 小时有效期、内容哈希
和 constant-time MAC 后，才写入 runner 的持久 Request store。

同一 Envelope 重放不会产生第二个 Request 或第二条初始状态链；首次接收生成不可变
`lab.request-ingress-receipt.v1`。同一 Request ID 从不同 Envelope 再次进入会 fail closed，避免悄悄更换 sender、key
或时间窗。

HMAC 提供持有共享密钥的对端认证和内容认证，但不负责密钥分发、撤销或主机身份发现。后续 SSH/API/消息队列 adapter
仍须保护密钥文件、限制 service account，并把传输失败与测试失败分开记录。

### 无 filter 的 exact Git staging

领取有效 claim 后，受信 runner adapter 可以执行：

```bash
lab runner stage <runner-id> <request-id>
```

`ExactSnapshotStager` 只读取：

```text
validated Request + validated runner claim + trusted host repository catalog
```

它不会 fetch、clone、初始化未知 submodule 或运行仓库命令。每个仓库必须已经在本机 Git object database 中包含
Request 指定的 exact commit/tree/gitlinks，否则 slot 进入 incomplete/quarantine 状态并停止。

物化过程使用 `git worktree add --no-checkout --detach` 建立 Git 身份，再通过 plumbing API 读取原始 tree/blob 并写入
worktree。这样不会触发 Task 提供的 checkout filter、Git hook 或 LFS smudge；同时设置
`GIT_LFS_SKIP_SMUDGE=1`、禁用 terminal prompt，并对单 blob 和总源码体积设置上限。绝对/逃逸 symlink、`.git`
路径、非 UTF-8 路径和未知 tree entry 会 fail closed。

父仓/子仓 link 按拓扑顺序物化。子仓 worktree 位于父仓声明的 gitlink 路径，且两者 commit 必须与 Request link
完全一致。成功后再次校验：

```text
detached HEAD
exact commit
exact tree
exact gitlinks
clean status
```

成功产生 `lab.snapshot-stage-receipt.v1`，记录逻辑相对路径、源码 identity、claim/runner/slot、
`network_access=false` 和 `lfs_smudge=false`。Slot 已完整时调用幂等返回；已有目录但缺 receipt 时拒绝覆盖或清理，交给
后续 runner recovery/quarantine 策略处理。

### 明确未实现

M3h 不提供：

- SSH、HTTP、队列或目录轮询 daemon；
- Git fetch/object bundle 传输；
- secret 创建、分发或轮换；
- environment/build/command executor；
- artifact 字节回传；
- GPU、service、device、physical cell 或 operator lease；
- ServiceReceipt、fixture、Attempt 或 authority handoff；
- dirty/synthetic snapshot。

因此本切片不能被描述为“远端测试已部署”。它只让后续 transport adapter 和 executor 之间拥有可验证、可测试的安全
接缝。

## 第二部分：当前项目实例

当前仓库仍不登记真实 transport key、runner、slot 或远端 Queue Profile，也不联系
`user@192.168.70.116`。测试全部使用临时的 `Controller`/`Component` Git 仓库和逻辑
`controller-a`/`runner-node-b` 身份。

对当前仓库组合而言，AgenticRobot、RPent 和 openpi 仍只是受保护项目配置中的通用 repository identity。
`model/openpi` 由父仓 gitlink 和 `repository_links` 表达；M3h 的 linked staging 测试证明相同机制可将任意 child
repository 放入父仓 gitlink 路径，核心没有模型、GPU、机器人或 OpenPI 专用逻辑。

真实远端接入前还需要独立完成：

1. protected control plane 合并与外置安装；
2. 新建 clean management/source anchor，保证所需 Git objects 已存在；
3. 用 host-local secret/config 登记 transport peer、runner、slot 和 repository catalog；
4. 先做无 payload、无 GPU、无设备的 envelope + staging 预检；
5. 再实现 reviewed command/environment adapter 和 GPU-only Profile。

M3h 的实现边界以本次提交固定；连续交付线和后续 M4-M8 规划见
[`lab_framework_plan.md`](./lab_framework_plan.md)。

## 验证

```bash
python -m pytest -q tests/test_lab_m3h_transport_staging.py
./dev test lab-m3h
ruff check lab_control tests/test_lab_m3h_transport_staging.py
mypy lab_control
python -m lab_control.framework_boundary \
  --root . \
  --policy integration/policies/lab-framework-boundaries.json
git diff --check
```

覆盖 HMAC/接收者/时间窗/篡改、幂等 replay、缺失 Git objects、无 LFS smudge、detached clean identity、
incomplete slot fail-closed，以及父仓 gitlink 路径中的 linked child。测试不访问网络、大文件、GPU 或设备。
