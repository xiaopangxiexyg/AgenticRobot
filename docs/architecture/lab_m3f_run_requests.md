# `lab` M3f：持久化 Run 请求协议

状态：叠加在 M3e 上的 review candidate。本切片实现项目无关的请求控制面，不实现 SSH/网络传输、远端
runner、GPU/设备分配、测试执行或 artifact 回传。

## 第一部分：一般项目的管理框架

### 目标与边界

M3f 在“用户想运行一个确定测试”和“某个执行节点真正运行它”之间建立持久、可审计的协议：

```text
Task + clean snapshot + trusted Profile
                  │
                  ▼
        content-addressed Request
                  │
       durable state + event receipts
                  │
                  ▼
       future transport / runner adapter
```

用户不需要直接管理 detached worktree、测试槽、SSH 会话或远端目录。当前入口是：

```bash
lab test   <task-id> <queue-profile-id> --detach
lab verify <task-id> <queue-profile-id> --detach

lab requests list [--task-id <task-id>]
lab requests status <request-id>
lab requests watch <request-id> [--after-sequence N]
lab requests cancel <request-id> --actor <id> --reason <text>
```

`--detach` 在这里表示“持久化提交后立即返回”，不是启动后台 shell。它只接受 clean、exact Git snapshot；
dirty/synthetic snapshot 属于后续独立切片。

### 项目无关 Profile

Queue Profile 只声明通用管理语义：

```json
{
  "schema": "lab.queue-profile.v1",
  "profile_id": "contract-check",
  "description": "Run a registered contract check",
  "target_id": "runner-group-a",
  "risk_level": "R0",
  "requires_clean": true,
  "repositories": ["Controller", "Component"],
  "command_ids": ["contract-suite"],
  "capabilities": ["python"]
}
```

核心不理解具体代码库、模型、机器人、GPU 类型或主机 IP。`target_id`、repository、command 和 capability
都是由项目集成层解释的逻辑 ID。Profile 从受信目录加载，文件身份由 SHA-256 绑定；symlink、路径逃逸、
重复字段、未知 schema、非 clean Profile 和超限文件都会被拒绝。

请求中不包含 shell argv、任意环境变量、本机 worktree 路径或管理仓路径。待实现 runner 必须用
`profile_id + profile_sha256` 在自己的受保护控制面重新解析授权，不能信任 Task 载荷自行扩大命令、网络或
资源权限。

### 可移植请求身份与幂等性

请求只携带：

- Task、Profile、target、risk、capability 和 evidence tier；
- 提交请求的 control-plane version、commit/tree、dirty 状态和 executor hash；
- 每个 repository 的 commit、tree OID 和 submodule gitlink；
- 父仓 gitlink 与独立子仓 commit 的一致性关系；
- baseline ID/hash（存在时）。

所有本机绝对路径都会在提交前去除。对规范化内容计算 SHA-256，得到 `idempotency_key` 和稳定
`request_id`。同一代码组合、Profile 和 evidence tier 重复提交会返回原 Request；相同 ID 但内容不同会
fail closed。

### 状态机、槽位与取消

```text
SUBMITTED
  ├── CANCELLED / FAILED
  └── RESERVED
        └── PREPARING
              └── RUNNING
                    └── COLLECTING
                          └── COMPLETED

RESERVED/PREPARING/RUNNING/COLLECTING
  └── CANCEL_REQUESTED
        ├── CANCELLED
        └── QUARANTINED
```

`SUBMITTED` 阶段取消可以立即完成；持有运行资源后，取消只产生 `CANCEL_REQUESTED`，必须由 runner 完成
反序 teardown 后确认 `CANCELLED`。取消请求本身不证明设备或进程已经停止。

同一 store 内一个逻辑 slot 同时只能被一个 active Request 持有；slot 一旦分配不能在生命周期中更换。
这是 host-local 调度原语，不是设备 authority，也不是跨主机组合资源预留。

每次 transition 写入独立事件收据，并在 state summary 中绑定序号、时间和 SHA-256。读取时重新验证请求
内容哈希、连续 transition、事件哈希、时间单调性、slot 连续性以及终态 outcome。SHA-256 用于完整性和
错误检测；在普通用户仍可重写整个 state root 的机器上，它不替代签名、只读 mount 或独立服务账户。

终态分类保持可区分：`PASSED`、`FAILED_ASSERTION`、`BLOCKED_PRECONDITION`、`ABORTED_SAFETY`、
`INFRASTRUCTURE_FAILURE`、`INVALID_SOURCE`、`INVALID_ENVIRONMENT`、`INCONCLUSIVE_MANUAL`、
`CANCELLED` 和 `QUARANTINED`。本切片只记录分类，不声称已实现对应执行行为。

### 明确未实现

M3f 不会：

- 连接任何远端主机或自动传输 Git objects；
- 拉取、切换或修复 stable checkout；
- 创建 runner slot 或运行命令；
- 分配 GPU、端口、服务、设备、physical cell 或 operator；
- 启动/附着服务、验证 ServiceReceipt 或回传 artifact；
- 将 Development/Candidate evidence 提升为 Release evidence。

后续 transport/runner 必须消费这个协议，而不是绕过它重新接受任意 SSH 命令。

## 第二部分：当前项目实例

当前仓库只落地了通用实现和 hermetic 测试，没有注册任何真实远端 Queue Profile，也没有写入远端 IP、
OpenPI 模型、checkpoint、GPU、设备或命令细节。项目集成时应把这些映射分别放在受保护的：

```text
integration/queue-profiles/   # 逻辑 Profile 与 target/capability 要求
integration/hosts/            # 后续 host-local capability/asset 映射
integration/commands/         # 后续受信 command adapter 映射
```

其中前一目录已是默认解析位置；后两项只是后续规划，当前代码不依赖也不假定其 schema。

AgenticRobot、RPent 与作为 `model/openpi` submodule 的 openpi 仍由 `integration/project.json` 和
`stack/baseline.json` 表达 repository/gitlink 关系。M3f 的请求生成器只读取解析后的通用 repository identity，
因此同一机制也适用于完全不同的多仓项目。

当前叠加关系为：

```text
task/AR-openpi-submodule-v3
  └── task/AR-lab-m1-workspace-v2
        └── task/AR-lab-m2-local-runs-v2
              └── task/AR-lab-m3a-stable-sync-v2
                    └── task/AR-lab-m3b-maintenance-sync-v2
                          └── task/AR-lab-m3c-gitlink-sync-v2
                                └── task/AR-lab-m3d-control-install-v2
                                      └── task/AR-lab-m3e-merge-gates-v2
                                            └── task/AR-lab-m3f-run-requests-v2
```

## 验证

```bash
python -m pytest -q tests/test_lab_run_requests.py
./dev test lab-m3f
ruff check lab_control tests/test_lab_run_requests.py
mypy lab_control
python -m lab_control.framework_boundary \
  --root . \
  --policy integration/policies/lab-framework-boundaries.json
git diff --check
```

测试只创建临时、通用命名的 Git 仓库和 Task，验证请求可移植/幂等、dirty 拒绝、gitlink 绑定、Profile
symlink 拒绝、slot 排他、完整状态生命周期、取消确认、watch 和事件链完整性。它不访问网络、大文件、GPU
或设备。
