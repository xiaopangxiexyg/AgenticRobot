# `lab` M3g：受信 Runner 领取与 artifact 收据

状态：叠加在 M3f 上的 review candidate。本切片将持久 Request 接到一个项目无关的 runner 管理协议；它仍不
执行测试载荷、不传输源码/artifact、不联系远端，也不分配 GPU、服务或设备权限。

## 第一部分：一般项目的管理框架

### Runner 只领取已授权组合

Runner Profile 是 host/control-plane 侧的受信能力声明：

```json
{
  "schema": "lab.runner-profile.v1",
  "runner_id": "runner-a",
  "description": "A managed execution adapter",
  "target_ids": ["target-a"],
  "slots": ["slot-a1"],
  "capabilities": ["python"],
  "accepted_risk_levels": ["R0"],
  "accepted_evidence_tiers": ["Development"]
}
```

这些字段都是逻辑 ID，不包含代码库、模型、设备、IP 或绝对路径。加载器拒绝 symlink、路径逃逸、未知字段、
重复值、非法 risk/evidence tier 和超限文件，并用 SHA-256 固定 Profile 身份。

领取前，runner 重新从自己的受保护 payload 加载 Queue Profile，而不是相信 Request 自报权限。以下条件必须
同时成立：

```text
request Profile hash == runner 侧受信 Queue Profile hash
request target/commands/repositories/capabilities/risk == 受信 Profile
target ∈ runner.target_ids
required capabilities ⊆ runner.capabilities
risk/evidence tier ∈ runner explicit allowlist
```

不满足的 Request 只会在 inspection 中显示具体 rejection reason，不会被领取。

### 原子槽位领取与恢复

管理入口为：

```bash
lab runner inspect <runner-id>
lab runner claim <runner-id> <slot-id>
```

这是 runner/operator 管理入口，不是日常开发者执行测试的入口；开发者仍使用 `lab test --detach`、
`lab requests` 和后续 `lab runs`。

每个 runner 的 claim loop 由 host-local `flock` 串行化；Request 的 `SUBMITTED -> RESERVED` transition 继续由
全局 request-store lock 保护，因此两个 runner 不能把同一 Request 分配到不同 slot，同一 slot 也不能同时持有
两个 active Request。

领取成功产生不可变 `lab.runner-claim.v1` 收据，绑定：

- Request ID/idempotency key；
- runner ID、Runner Profile hash 与 slot；
- target/capability；
- runner control-plane version、commit/tree、dirty 状态、executor hash 和外置安装 manifest hash；
- reservation timestamp 与收据 hash。

如果进程在 `RESERVED` 写入后、claim attachment 持久化前中断，同一 runner 可依据 exact reservation event 恢复
同一 claim；不匹配的 actor、Profile 或 capability 会 fail closed。已经存在的 active claim 会幂等返回，供未来
runner daemon 做状态 reconciliation。M3g 不会据此自动恢复或重启测试进程。

### 生命周期和 artifact 收据边界

`RunnerCoordinator` 只允许持有有效 claim 的 runner 推进：

```text
RESERVED -> PREPARING -> RUNNING -> COLLECTING
```

它不实现这些阶段内的 checkout、environment、command 或 teardown；后续 executor adapter 必须在各阶段写入自己的
受信 receipt 后再调用 transition。

COLLECTING 阶段可以记录 `lab.artifact-receipt.v1`。每个 descriptor 只包含逻辑 storage ID、安全相对路径、
media type、byte size 和 SHA-256。框架不把视频、模型、数据集或日志字节塞入 Git/request state，也不接受绝对
storage path。最多记录 256 项元数据；实际 artifact transport 必须在后续 adapter 中校验字节 hash。

只有存在 `evidence_complete=true` 且 hash/claim/runner/slot 全部匹配的收据，RunnerCoordinator 才允许标记
`COMPLETED/PASSED`。`evidence_complete` 只是 runner 对声明产物的完整性结论，不替代测试断言、设备 acknowledgement
或 Release evidence。取消仍遵循 M3f：active Request 先进入 `CANCEL_REQUESTED`，持有 claim 的 runner 完成受管
teardown 后才能确认 `CANCELLED`。

### 信任和未实现边界

从外置控制面启动时，身份取自已重新验证全部只读 payload 文件的 active install manifest，而不是被管理 checkout
当前恰好包含的源码；直接开发模式则明确记录 `install_manifest_sha256=null` 和真实 dirty 状态。
因此 M3g 新请求升级为 `lab.run-request.v2`；reader 仍验证并接受 M3f 的 v1 请求，避免升级控制面后丢失已提交
队列。v2 的 idempotency hash 原生覆盖 install manifest identity。

当前 hash receipts 能发现意外漂移和局部篡改，但普通 Unix owner 仍可能重写整个 state root；它们不是数字签名。
真实高权限 runner 仍需独立 service account、受保护只读 payload、受限文件权限和后续 transport authentication。

M3g 明确不提供：

- SSH/API/队列 transport；
- Git object 或 synthetic snapshot 传输；
- detached test slot 的源码/environment 准备；
- 任意 argv、shell、网络、GPU、service 或 device 执行；
- artifact 字节上传/下载；
- runtime ServiceReceipt、组合资源预留或 authority handoff。

## 第二部分：当前项目实例

当前仓库没有登记任何真实 `integration/runners/*.json` 或远端 Queue Profile，因此 M3g 不会发现、连接或控制任何
现有主机。实现测试全部使用临时的 `Controller` 仓库、`runner-a`、`test-target` 和 `slot-a1`，以确保核心代码
不依赖 AgenticRobot、RPent、openpi 或机器人/模型语义。

后续当前项目配置应放在受保护集成层：

```text
integration/queue-profiles/   # 测试场景的逻辑 target/command/capability
integration/runners/          # 每个 runner 的 target/slot/capability allowlist
integration/commands/         # 后续 executor 的受信 command adapter
integration/hosts/            # 后续机器局部 capability/asset 解析
```

后两项仍只是规划，M3g 不定义其 schema。`model/openpi` 的 submodule gitlink 只作为普通 repository snapshot
identity 进入 Request；runner 核心不理解其代码、模型或运行方式。

当前叠加关系追加为：

```text
task/AR-lab-m3f-run-requests-v2
  └── task/AR-lab-m3g-runner-protocol-v2
```

## 验证

```bash
python -m pytest -q tests/test_lab_runner_protocol.py
./dev test lab-m3g
ruff check lab_control tests/test_lab_runner_protocol.py
mypy lab_control
python -m lab_control.framework_boundary \
  --root . \
  --policy integration/policies/lab-framework-boundaries.json
git diff --check
```

测试覆盖 eligible/rejection 解释、slot/claim 幂等、claim 中断恢复、Queue Profile 漂移、受控 lifecycle、artifact
路径/hash/完整性、取消确认和 CLI 管理入口；不访问网络、大文件、GPU 或设备。
