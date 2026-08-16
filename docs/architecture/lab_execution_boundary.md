# `lab` 与项目运行命令的职责边界

状态：当前实施约束。本文用于防止代码管理框架再次吸收具体项目、模型、设备或运行平台语义。

## 第一部分：一般项目的边界

`lab` 是多仓 Git 工作区与可复现测试入口，不是通用资源调度器。它只负责：

1. 解析 Task、仓库、branch、worktree、baseline 和 submodule/gitlink；
2. 生成 exact clean 或显式授权的 synthetic draft 源码快照；
3. 从受保护配置选择 `target_id`、`command_id` 和允许的源码模式；
4. 将确定快照交给目标 runner 的测试槽；
5. 调用受信 command adapter，记录开始/结束、退出码、stdout/stderr 和 artifact 索引；
6. 提供 request/run 的查询、等待、取消和幂等语义；
7. 在本机与远端之间传递 Git objects、manifest 和测试结果，不同步活动工作目录。

`lab` 不负责理解或实现：

```text
Python/Conda/CUDA 环境创建
容器、systemd、tmux 或服务生命周期
GPU、端口、设备、数据集、模型和校准文件选择
进程健康检查、设备锁、安全控制或故障恢复
某个测试场景的业务步骤和成功判定
```

这些内容全部属于受信的系统/项目命令。例如 Profile 只引用：

```json
{
  "target_id": "remote-runner",
  "command_ids": ["integration-suite"],
  "snapshot_modes": ["clean"]
}
```

runner 主机上的受保护映射再将 `integration-suite` 转换为固定命令，例如 `./dev test integration` 或
`/opt/project-tests/integration-suite`。Request 不能携带任意 shell 字符串，Task 对映射文件的修改也不能自动改变
受信 runner。命令可以自行调用系统已有的任何合法能力；`lab` 只把其输出当作不透明执行结果和 artifact。

最小执行生命周期因此收敛为：

```text
Resolve source -> Stage exact snapshot -> Invoke trusted command
               -> Collect result -> Finalize Run
```

取消表示要求 command adapter 终止本次命令并回收它创建的进程；具体程序如何安全停止仍由该命令及操作系统负责。
同理，`capabilities` 只用于匹配“某 runner 是否声明可执行这个 command”，不得在核心中展开成 GPU、设备或服务模型。

Run 的通用证据只记录：

```text
repository commit/tree/gitlink combination
Profile and command identity/hash
control-plane and runner identity
target host identity
timestamps, exit status and cancellation state
stdout/stderr and artifact content receipts
```

命令可在 artifact 中附加自己的领域报告，但 `lab_control` 不解析其模型、硬件或服务字段。

## 第二部分：当前项目的映射

AgenticRobot、RPent 和 openpi 只以 Git 仓库关系进入框架：

- AgenticRobot 是 control repository；
- RPent 是 sibling repository；
- `model/openpi` 是 AgenticRobot 固定版本的 Git submodule；
- Task 可为参与修改的仓库创建 sibling/multi-root worktree；正式组合由父仓 gitlink 和 baseline 固定。

本机是默认开发与集成节点，远端 `user@192.168.70.116` 是附加执行节点。地址只保存在 host-local transport 配置中，
不进入通用 schema。远端测试最终应表现为：

```text
lab stage exact source
  -> remote trusted adapter
  -> one configured system command
  -> exit/log/artifact receipt returned to lab
```

如果该命令需要 openpi 环境、模型、GPU、服务或真实设备，由远端现有脚本和系统配置自行准备和保护；代码管理框架不
枚举、申请或编排这些内容。新增项目也只需注册仓库、submodule 关系、runner target 和 command mapping，不需要修改
`lab_control` 核心。

## 防回归要求

- `lab_control/**/*.py` 不得出现当前项目名称、IP、绝对路径或具体运行资源名称；
- Queue Profile 不得增加 execution class、设备模式、服务拓扑或资源列表；
- Runner Profile 不得承担资源调度数据库职责；
- 任何需要解析项目专用字段的实现都应下沉到系统命令，或作为不透明 artifact 保存；
- 评审新增 schema 时先问：去掉 AgenticRobot/openpi/机器人场景后，该字段是否仍是 Git/Task/Run 管理所必需？若否，
  不应进入核心。
