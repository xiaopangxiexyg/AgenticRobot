# `lab` M5：受信系统命令执行与日志回收

状态：`task/AR-lab-framework-v2` 中的 review candidate。本切片在 M3h exact staging 与 M4 snapshot 之上增加最小
command adapter；不连接真实远端，也不解释命令内部的环境、进程或项目语义。

## 第一部分：一般项目框架

### 执行合同

一次 queued Run 已由 Queue Profile 固定 `target_id` 和一组 `command_id`。runner claim 与 exact source staging 完成
后，runner 从 host-local 受保护目录加载：

```json
{
  "schema": "lab.runner-command-map.v1",
  "runner_id": "runner-a",
  "commands": [
    {
      "command_id": "contract-suite",
      "repository": "Controller",
      "argv": ["./dev", "test", "contract"],
      "timeout_seconds": 1800
    }
  ]
}
```

Request 不能提交 argv 或 shell 字符串，只能引用已在 Queue Profile 中注册的 command ID。Command map 不属于被测试
snapshot；修改 Task 分支中的 Profile 或 runner 文件不能改变已安装 runner 的 allowlist。加载器严格拒绝额外字段，
因此不能把资源、服务拓扑或项目运行模式塞入核心合同。

### 系统命令调用

`TrustedCommandExecutor`：

1. 重新验证 runner claim、Queue Profile、stage receipt 与 materialized Git source；
2. 检查每个 command ID 在当前 runner 的 protected map 中存在，且其 repository 已进入 staged snapshot；
3. 使用 argv 数组和 `subprocess.Popen(..., shell=False)` 直接调用系统命令；
4. `cwd` 固定为命令声明的 staged repository；
5. 通过通用环境变量提供 `LAB_REQUEST_ID`、`LAB_SOURCE_ROOT`、`LAB_RUN_ROOT` 和 `LAB_EXECUTION_CONTEXT`；
6. 顺序执行并 fail-fast，监视 durable Request 的取消状态；
7. 超时或取消时只终止本次 executor 创建的进程组；
8. 执行后再次验证 commit/tree/gitlink 和 clean status，拒绝修改 staged source 的命令；
9. 将 command map hash、stage receipt hash、argv、退出码、时长和日志 hash 写入 execution receipt；
10. 将日志登记为通用 artifact receipt，再完成或失败 Request。

用户或 host runner 的底层入口是：

```bash
lab runner execute <runner-id> <request-id>
```

`execute` 可幂等读取已经完成的 receipt，不会重复启动已记录命令。若 Run 目录已出现但没有完整 receipt，则 fail
closed，不猜测上一次命令是否执行，也不自动清理或重跑。

### 明确不属于 M5 的内容

M5 不创建虚拟环境、容器或后台服务，不选择机器资源，不解释项目测试报告，也不提供任意交互 shell。系统命令内部
可以调用其主机上已经配置好的合法工具；相关准备、互斥、安全和业务断言由该命令自行负责。`lab` 仅管理源码身份、
受保护命令身份、通用进程退出和日志证据。

M5 也尚未实现：

- 网络 transport 或 SSH 调用；
- Git objects、Request 或 artifact 的跨机传输；
- runner daemon、后台队列循环或断线恢复进程；
- command 自定义 artifact 目录的收集策略；
- test slot retention、垃圾回收或自动重试。

### 状态与结果

```text
PREPARING
  -> RUNNING
  -> COLLECTING
  -> COMPLETED | FAILED

任意活动状态 --cancel--> CANCEL_REQUESTED -> CANCELLED
```

当前通用分类只有：

```text
PASSED
FAILED_ASSERTION
INFRASTRUCTURE_FAILURE
INVALID_SOURCE
CANCELLED
```

这些分类来自进程启动、超时、退出码、取消和源码完整性，不解析命令输出中的领域含义。

## 第二部分：当前项目映射

本切片没有为 AgenticRobot、RPent 或 openpi 创建生产 command map，也没有连接
`user@192.168.70.116`。当前项目后续只需在各主机受保护安装中注册 command ID 到已有系统命令的映射，例如本机
运行一个 `./dev` target、远端运行一个 host-local test adapter；具体命令内容不进入 `lab_control`。

AgenticRobot 与 `model/openpi` 的父仓/gitlink 一致性以及 RPent sibling revision 已在 snapshot/staging 层固定。
Command executor只看到通用 repository identity 和 staged path，不知道哪个仓库包含模型或其它运行实现。

下一阶段 M6 接入 transport 时保持同一边界：本机传递 exact source 与 Request，远端执行相同 command adapter，再
回传 execution/artifact receipt；不会新增项目资源调度 schema，也不会同步活动 worktree。M6a 的可移植 package
合同见 [`lab_m6_portable_transfer.md`](lab_m6_portable_transfer.md)。

## 验证

```bash
python -m pytest -q tests/test_lab_m5_command_execution.py
./dev test lab-m5
./dev test lab-control
ruff check lab_control tests/test_lab_m5_command_execution.py
mypy lab_control
git diff --check
```

测试使用临时 `Controller` 仓库和受保护 command map，覆盖成功、非零退出、日志 artifact、幂等执行、额外运行字段
拒绝与运行中取消。命令仅启动本机测试 Python 子进程，不访问网络或大文件。
