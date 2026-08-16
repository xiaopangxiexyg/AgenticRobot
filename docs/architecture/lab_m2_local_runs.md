# `lab` M2：本地 exact clean Run

状态：叠加在 M1 Task Workspace 之上的 M2 review candidate。M2 只执行本机 `R0/R1`、clean、
预注册 Profile；不包含 draft、SSH、GPU、服务或设备权限。

## 第一部分：一般项目的开发与测试合同

M2 把“看正在编辑的 worktree”和“对可复现源码运行测试”分开：

```text
Task worktree                 detached review worktree
持续开发/查看 diff     ──commit──> exact commit/tree
./dev 快速原地测试             ./lab test 的稳定测试对象
                                     │
                                     └── Run manifest + logs + hashes
```

用户入口：

```bash
./dev doctor
./dev test <repository-local-target>

./lab doctor
./lab inspect T101
./lab test T101 <profile-id>
./lab test T101 --recommended
./lab runs T101
./lab open --run <run-id>
./lab inspect T101 --against-run <run-id>
```

`./dev` 是当前 checkout 的轻量入口，适合 writer 已暂停修改时的快速测试。`./lab test` 是可复现入口：

1. 拒绝 dirty、branch drift 或 scope drift 的 Task；
2. 解析 Profile 需要的 Task overrides，并只在必要时由 baseline 补齐未修改仓库；
3. 为每个仓库创建 exact commit 的 detached review worktree；
4. submodule 只允许 `--no-fetch` 初始化，缺少本地 object 时 fail closed；
5. 解析独立 environment Profile，并验证 first-party import 来自本次 snapshot；
6. 用最小 HOME/TMP/cache、无 user site、有限日志大小的环境运行固定 argv；
7. 测试结束重新验证 commit/tree/submodule/status；
8. 生成唯一 Run，记录源码、控制面、环境、命令、结果、日志大小与 SHA-256。

Run 将以下状态分开：

- `PASSED`：命令完成、断言通过、snapshot 未漂移；
- `FAILED_ASSERTION`：测试进程正常完成但断言/退出码失败；
- `INFRASTRUCTURE_FAILURE`：启动失败或超时；
- `INVALID_SOURCE`：测试过程中 snapshot 被修改；
- `INVALID_ENVIRONMENT`：环境或 import provenance 不满足 Profile。

同一 snapshot 可产生多次 Run，但每次拥有独立目录、日志和 Run ID。长测试运行在 detached snapshot，
因此 Agent 可在原 Task worktree 继续开发；结果不会被后续编辑悄悄改变。

### Development 与 Candidate evidence

`lab test` 产生 Development evidence，允许控制面本身位于 review branch，但 manifest 必须记录控制面的
commit、tree、dirty 状态与 executor hash。

`lab verify` 产生 Candidate evidence，额外要求运行 `lab` 的 control checkout：

```text
branch == main
working tree == clean
HEAD == refs/remotes/origin/main
```

因此待测试 Task 不能通过修改 `lab`、Profile 或 runner 来扩大自己的权限。M2 尚未定义 release evidence；
真机/发布证据后续仍必须在 merged/release candidate 上重跑或按严格 tree equivalence 规则提升。

### Profile 与 impact rules

Profile 声明固定 repository、argv、timeout、environment 和风险等级。Task 不能从命令行注入 shell、cwd、
环境变量或资源权限。M2 只接受 local、clean、`R0/R1` Profile。

`lab inspect` 根据受保护 impact rules 把 changed paths 映射为推荐 Profile，并报告最近 Run 是否仍对应当前
commit/tree/Profile/baseline。未覆盖路径会显示 `ATTENTION`，不会被误称为已有测试覆盖。

## 第二部分：当前项目实例

当前 M2 仅注册一个本机 Profile：`lab-control-unit`。它验证框架自身的 M1/M2 测试，不启动
AgenticRobot runtime，不导入 OpenPI 模型，不访问 RPent runtime、GPU、网络或设备。

```bash
./dev test lab-m1
./dev test lab-m2
./dev test lab-control

./lab doctor --environment agenticrobot-py310
./lab test <Task-ID> lab-control-unit
```

项目特定内容位于：

```text
integration/project.json
integration/dev.json
integration/environments/agenticrobot-py310.json
integration/profiles/lab-control-unit.json
stack/impact-rules.json
stack/baseline.json
```

`lab_control/` 核心不含 AgenticRobot、RPent、OpenPI、GPU、模型或设备名称。OpenPI 的版本关系由
AgenticRobot `model/openpi` gitlink 与 baseline `repository_links` 表达；M2 不下载或运行 OpenPI。

当前评审依赖顺序：

```text
task/AR-openpi-submodule-v2
  └── task/AR-lab-m1-workspace
        └── task/AR-lab-m2-local-runs
```

每层合并后，下一层都必须重基到新的 protected `main` 并重新生成 baseline parent identity。当前
`stack/baseline.json` 的 ID 带 `candidate`，不能作为 release lock。

## 验证

```bash
python -m pytest -q \
  tests/test_lab_m1_workspace.py \
  tests/test_lab_writer_lease.py \
  tests/test_lab_m2_runs.py

ruff check lab_control tests/test_lab_m1_workspace.py \
  tests/test_lab_writer_lease.py tests/test_lab_m2_runs.py
mypy lab_control
git diff --check
```

测试只使用临时 Git 仓库和本机 Python，不访问网络、大文件、GPU 或设备。
