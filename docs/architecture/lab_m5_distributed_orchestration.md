# `lab` M5：分布式 Profile、组合预留与 ServiceReceipt

状态：`task/AR-lab-framework-v2` 中的 review candidate。本切片实现项目无关的分布式编排协议与本地持久状态；它不
连接主机、不启动或停止服务、不执行项目命令，也不申请真实 GPU/设备 authority。

## 第一部分：一般项目的管理框架

### Distributed topology 是不可变 Request 身份的一部分

受保护 Queue Profile 使用 `lab.queue-profile.v3` 显式声明：

```json
{
  "snapshot_modes": ["clean"],
  "execution_class": "compute",
  "execution_topology": "distributed"
}
```

首版 distributed topology 只允许 exact clean snapshot 和 `offline`、`simulation`、`compute` execution class。生成的
`lab.run-request.v4` 同时固定 Queue Profile hash、`execution_class` 和 `execution_topology=distributed`。普通单节点
RunnerCoordinator 必须将该 Request 标记为不合格；不能依靠 target 名称约定来避免误领，只有组合编排器可以处理。

旧 Queue Profile v1、v2 和 Run Request v1-v3 继续按原边界读取：旧 Profile 默认为 single topology，不会隐式获得
分布式资源权限。

### 一个 Profile 描述完整拓扑

`lab.distributed-profile.v1` 由受保护控制面加载，声明：

```text
node
  target ID
  exact repository set
  reviewed command IDs
  capabilities
  exclusive logical resource IDs
  produced service IDs + contract SHA-256

dependency
  producer node -> consumer node
  required service ID
```

加载时严格验证唯一 node/command/service/resource、服务 producer、依赖 DAG 和路径安全。Distributed Profile 必须与
Queue Profile 以及 Request 完全覆盖相同的 repository、command 和 aggregate capability 集合；少一个或多一个都
fail closed。Profile 只包含逻辑 ID，不包含 SSH 地址、设备节点、绝对 asset 路径或任意 shell。

Runner Profile v2 增加受保护的 `host_id` 和 `resources`。旧 v1 Runner 保持可读，但资源集合为空，不能承担需要显式
资源的 distributed node。

### 原子组合预留

受信控制面可以先只读规划：

```bash
lab orchestrate plan <request-id> <distributed-profile-id>
```

再一次性预留：

```bash
lab orchestrate reserve <request-id> <distributed-profile-id> \
  --orchestrator <logical-id>
```

编排器在一个全局 advisory lock 下读取全部未释放且未过期的 reservation，匹配 node 的 target/capability/resource、
risk/evidence 和 runner slot。它使用确定性候选顺序与回溯分配，只有全部 node 同时可分配时才写入一个不可变
`lab.composite-reservation.v1`：

```text
Request + Profile hashes
+ node -> runner/host/slot
+ complete exclusive resource set
+ dependency graph
+ reserved/expires time
```

任一 node 不可用时不写部分 reservation。重复相同请求幂等返回原 receipt；同一 Request 的不同编排冲突会拒绝。
释放生成单独的不可变 release receipt，不覆盖历史 reservation。TTL 只决定资源是否仍被调度器视为占用，不删除证据。

当前原子性只覆盖同一个受信 orchestration state root 中登记的 slot/resource。未登记的系统服务、手工进程或真实设备
owner 不受它控制；真实接入前必须让 host adapter 和 Device Gateway 将实际 ownership 映射到同一受信合同。

### ServiceReceipt 与 ready barrier

受信 runner adapter 在服务实际通过健康检查后调用库接口登记 `lab.service-receipt.v1`。Receipt 固定：

- Request、reservation、node、runner、host 和 slot identity；
- 服务 ID 与受保护 contract SHA-256；
- producer node 所用仓库的 exact commit/tree；
- environment/configuration SHA-256；
- endpoint（拒绝内嵌 credential）、PID 与防 PID 重用的 process start token；
- READY 时间、expiry 和内容哈希。

这避免仅因某个固定端口可连接，就把旧进程误认为当前 Run 的服务。连接已有服务也必须产生完全匹配的 receipt；未登记
服务不进入 ready barrier。

```bash
lab orchestrate barrier <request-id> <reservation-id>
```

Barrier 重新验证 reservation、当前 protected Profile/Runner hash、源码组合、每个 dependency service 的 producer、
contract、进程身份和 TTL。缺失、过期、篡改或漂移任一项都失败；全部满足后才生成不可变
`lab.ready-barrier.v1`。它是“可以进入后续 Start/Execute”的证据，不代表测试通过，也不代表服务由框架启动。

### 明确未实现

M5 不提供：

- SSH/API/daemon、跨主机两阶段提交或网络故障恢复；
- environment/build/command executor、进程组启动与反序 teardown；
- synthetic object 跨机传输或 artifact 字节回传；
- 真实 GPU/device/cell/operator backend 与 lease；
- fixture、Attempt、health renewal、service replacement 或 quarantine；
- observe/control、watchdog、authority handoff 或 release evidence。

因此本切片不能被描述为“分布式测试已经能运行”。它提供的是被执行 adapter 使用前必须满足的受保护、可审计协议。

## 第二部分：当前项目实例

当前项目可在后续配置中将本机节点表达为 AgenticRobot/RPent 的通用运行端，将远端节点表达为 AgenticRobot/openpi
组合的专用计算端；`model/openpi` 仍只通过父仓 gitlink 与 child commit 进入 exact snapshot。具体主机地址、GPU、
checkpoint、端口、设备、校准和启动命令不进入 `lab_control/`，只由受保护 host/runner/Profile/asset adapter 提供。

本切片没有创建任何当前项目的 distributed/runner 生产 Profile，也没有连接 `user@192.168.70.116`。测试使用临时
`Controller`/`Component` 仓库、`host-a`/`host-b`、逻辑 resource 和 loopback endpoint，验证通用框架而非某个模型或
机器人实现。

真实接入前至少还需要：

1. protected control plane 外置安装；
2. 双端 clean source anchor 与受认证 Request/object transport；
3. host-local inventory 将实际 slot/resource owner 映射为受保护逻辑 ID；
4. 无项目命令的 reserve/release/断线演练；
5. reviewed executor 与 per-Run service 的启动、健康检查和反序 teardown；
6. 最后才启用计算资源，设备 observe/control 留给 M6/M7。

## 验证

```bash
python -m pytest -q tests/test_lab_m5_orchestration.py
./dev test lab-m5
ruff check lab_control tests/test_lab_m5_orchestration.py
mypy lab_control
python -m lab_control.framework_boundary \
  --root . --policy integration/policies/lab-framework-boundaries.json
git diff --check
```

测试覆盖 v1/v2 Runner 兼容、Request v4/普通 runner 拒绝、Profile DAG、跨 runner 原子组合预留、资源冲突无部分写入、
释放后再预留、ServiceReceipt 源码/契约/环境/进程身份、credential endpoint 拒绝、ready barrier 和篡改检测。测试
不访问网络、大文件、GPU 或设备，也不启动服务。
