# AgenticRobot 测试耗时压缩与安全交接文档

文档版本：2026-09-08

本文记录本次 AgenticRobot 测试耗时优化的目标、当前候选、性能证据、开发约束、调试方式、集成门槛和回滚方式。文中带有日期、commit、工作区和 /tmp 路径的内容都是本次快照；集成或复测前必须重新盘点并重新固定 SHA，不能把旧快照当成当前事实。

## 1. 任务目标与背景

### 1.1 用户要求

AgenticRobot 每次合并前的测试耗时过长，尤其是完整 acceptance 或 Harness 保护的源码检查经常长时间没有终端输出。目标不是减少测试数量，而是在不降低覆盖、不放宽安全门禁、不改变受保护 Profile 语义的前提下，压缩实际墙钟时间，并让等待过程可解释。

本次上下文还观察到 RPent 的 rpent-source-check：

- 它通过 scripts/run_pytest.sh 包装并运行完整测试，不是一个很小的源码 lint。
- 单条上限为 1800 秒；子进程输出可能留在受保护 Run 日志中，外层 Harness 要等生命周期结束才返回结构化结果。
- 默认沙箱可能因为 socket 限制产生 PermissionError: [Errno 1] Operation not permitted。
- 该错误是环境能力证据，不能改写成通过，也不能靠减少测试范围来“修复”。应保留原 Run，在 socket-capable 上下文用同一提交、同一 Profile、同一输入重跑。

因此本任务同时解决两类问题：降低 AgenticRobot 本地/集成验收的墙钟时间，以及区分慢模块、环境失败、真正代码失败和缺失资源。

### 1.2 本批次安全目标

本批次只实现一个可回退的低风险优化：

1. 完整非硬件 acceptance 默认仍为 4 个 shard。
2. 资源经过人工确认的主机可以显式选择 5 或 6 个 shard。
3. 每个 shard 仍执行完整集合中的互不重复模块，并保留独立缓存、临时目录、日志和 JUnit。
4. 聚合 JUnit 仍拒绝重复 testcase；任一子进程失败仍使 wrapper 失败。
5. 不改变 marker、测试选择、节点集合、Harness Profile、远端执行、硬件边界或默认命令。
6. 不根据 nproc 自动决定 shard 数，也不把一次主机测量当成所有机器的默认保证。

## 2. 当前候选与所有权

| 项目 | 当前记录 |
| --- | --- |
| 目标仓库 | /home/zc/HandMotion/Developing/AgenticRobot-development-mainline |
| 目标线 | development |
| 目标线基准 | 4743d20a97fa07acc47d4217c39be4040c22b62e |
| 候选分支 | task/root/full-acceptance-six-shard-optin-20260908 |
| 候选提交（加入本文档前） | 04221f5e38f1ac1914085446b72e0b2088c3e424 |
| 候选工作区 | /tmp/ar-full-acceptance-six-shard-optin-20260908 |
| 候选所有者 | root |
| 目标线状态 | development 未改变，未合并、未 push |

候选原有代码范围：

```
scripts/run_full_acceptance.sh
tests/test_run_full_acceptance.py
tests/README.md
```

本文档加入后，write scope 会扩展为再包含：

```
docs/development/test-optimization-handoff-20260908.md
```

候选不拥有 README_ENV.md、Harness Profile/runner map、marker 选择逻辑、硬件、设备、远端执行、development、main 或 push。加入本文档会使原 ready 记录失效，必须重新提交、重新验证并重新固定 tip。

并行候选仍包括：

- watchdog：ff464788f980e3485c2a64219dc0d200fd558918；
- test reliability：ffe5a2cb5386b8d187bb422b8b81f4357aadfaf3。

当前盘点未发现自动精确重叠，但盘点只是事实快照，不是锁或集成授权。集成前必须重新运行 worktree_inventory.py，重新检查每个输入的完整 SHA、状态、范围和语义重叠。

## 3. 实现要点

### 3.1 Wrapper

scripts/run_full_acceptance.sh 当前行为：

```
未设置 DEXTELEOP_FULL_ACCEPTANCE_SHARDS -> 4
显式 1、2、3、4、5、6 -> 接受
其他值（包括 0、负数、7、auto） -> 退出码 2
```

已有 scripts/pytest_sharding.py 按 sparse timing profile 加稳定源文件大小 fallback 做确定性、加权的模块分配；本批不改变它。

每个子进程仍拥有独立的 pytest、Python bytecode、MuJoCo、XDG、Matplotlib、Numba、TMPDIR/basetemp、日志和 JUnit 状态。wrapper 等待全部子进程，再输出日志并合并 JUnit；不会因一个 shard 先失败而丢弃其他 shard 的证据。

### 3.2 回归测试

tests/test_run_full_acceptance.py 覆盖：

- 未设置环境变量时确实启动 4 个 shard；
- 显式 5 和 6 各启动准确数量；
- 7 和非整数值在启动子进程前拒绝；
- 六类缓存和临时目录精确隔离；
- pytest cache、JUnit、not hardware marker 和额外参数仍由 wrapper 管理；
- aggregate JUnit testcase 唯一且覆盖所有 shard；
- 第 6 个 shard 失败时退出码为 1，其他 shard 的 XML 仍保留；
- assets 临时链接的创建、校验和清理语义不变。

tests/README.md 已说明：默认仍为 4，5/6 只用于资源已验证的主机，7+ 不支持，不建议自动按 nproc 扩容。

## 4. 性能模型与证据

### 4.1 原流程为什么会久

完整 acceptance 会发现所有 tests/test_*.py 并执行整个非硬件集合。总耗时包括 pytest collection、导入初始化、少数长 MuJoCo/IK/replay/持久化模块、每个进程的缓存初始化、机器争用和 Harness artifact 生命周期。

wrapper 将子进程输出重定向到 shard 日志，通常要等所有 shard 才打印汇总。因此“十分钟没有终端错误”可能只是慢模块仍在运行或输出留在日志中；不要并发启动第二个同类受保护 Run，也不要自行删除或重置第一个 Run。

### 4.2 六分片计划

当前 full profile 计划：

- 795 个测试模块；
- 估计总权重 994.96638 秒；
- 最慢模块 tests/test_mujoco_temp_pipeline.py 约 169.954 秒，单独占一个 shard；
- 其余 shard 估计约 164.963–165.013 秒；
- 计划覆盖 795 个模块，去重后仍为 795。

计划权重不等于墙钟时间，还会受到导入、进程启动、机器负载和资源争用影响。

### 4.3 真实六分片运行

socket-capable 上下文使用同一候选和命令：

```
env DEXTELEOP_FULL_ACCEPTANCE_SHARDS=6 \
  /usr/bin/time -f '[full-acceptance-timing] elapsed=%e user=%U sys=%S max_rss_kb=%M' \
  bash scripts/run_full_acceptance.sh --durations=10
```

结果：

| 指标 | 结果 |
| --- | ---: |
| 墙钟时间 | 211.13 s |
| aggregate testcase | 9101 |
| skipped | 106 |
| failures | 4 |
| errors | 0 |
| 最长 shard | 207.313 s |
| JUnit 各 shard 时间合计 | 1078.372 s |

各 shard 实际时间约为 129.177、207.313、185.209、181.613、186.435、188.625 秒。历史四分片最大已知负载约 262 秒；按这两个快照粗略比较，六分片节省约 51 秒、约 19%。集成前必须在最终组合提交上重新测量，不能将该数字当作 SLA。

### 4.4 主机资源与默认值

本次机器约 32 CPU、61 GiB RAM、39 GiB available，但 8 GiB swap 已满。为避免普通或资源紧张主机出现 OOM、swap thrash 或 MuJoCo 初始化抖动：

- 默认值保持 4；
- 5/6 只能显式选择；
- 不自动读取 nproc；
- 使用前确认 CPU、内存、swap、其他并行任务和资产余量；
- 资源异常时回退到 4 或 1，保留 artifact 后诊断。

## 5. 已知失败及解释

socket-capable 六分片运行剩余 4 项失败：

### 5.1 reBotArm gitlink 未物化

3 项 B601 MuJoCo 测试因 hardware/reBotArm_control_py 只有固定 gitlink（5ba28acef46237eb6a7560658bbc43b06cf8a259），但 URDF 内容没有物化而失败。这是资源/工作区状态问题，不是分片逻辑通过证据。

安全处理：

1. 集成前确认 .gitmodules、gitlink SHA、允许的资源根和 runner 资源契约；
2. 在受控 runner 按项目流程初始化同一固定 gitlink；
3. 不从任意 Harness 临时 worktree 复制 vendor，不修改 gitlink 制造绿色；
4. 初始化后重跑受影响测试和完整 acceptance。

### 5.2 既有 MuJoCo 断言

tests/test_mujoco_temp_pipeline.py::test_descend_close_task_is_feasible_without_hidden_xy_actions 在六分片全量和完整模块运行中失败；完整模块结果为 1 failed, 113 passed，单独运行该节点时通过。这更像既有模块顺序、solver warm-state、资源争用或 reset 敏感性，不是本批仅增加 5/6 合法值的直接回归。

本批不修改 MuJoCo 行为、不改变断言、不添加 skip/xfail。应由 MuJoCo owner 单独分析；关闭前不能把当前全量称为绿色。

### 5.3 沙箱 socket 失败

普通沙箱运行还出现 loopback/AF_UNIX 创建失败：PermissionError: [Errno 1] Operation not permitted。socket-capable 复跑后这些失败消失。该结果只证明环境能力差异；原始 artifact 应保留为环境证据，正式 readiness 只使用控制平面和能力匹配的运行结果。

### 5.4 无硬件证据

本批没有连接相机、机械臂、VIVE、RM65 或远端设备，没有运动、归零、复位、伺服、力控或设备状态改变。离线 lifecycle 子进程打印的 protective hold 字样不能解释成真实设备动作。

## 6. 开发与调试流程

### 6.1 写入前盘点

```
git -C /home/zc/HandMotion/Developing/AgenticRobot-development-mainline worktree list --porcelain

python scripts/worktree_inventory.py \
  /home/zc/HandMotion/Developing/AgenticRobot-development-mainline \
  --mainline development \
  --branch task/root/full-acceptance-six-shard-optin-20260908
```

盘点 owner、state、base、完整 tip、dirty entries、write scope 和 semantic scope。盘点不是锁。

### 6.2 快速验证

```
bash -n scripts/run_full_acceptance.sh
ruff check tests/test_run_full_acceptance.py
bash scripts/run_pytest.sh -q \
  tests/test_run_full_acceptance.py \
  tests/test_pytest_sharding.py
git diff --check
```

本候选聚焦结果为 9 passed，连续运行 3 次，每次约 0.49 秒。

真实计划必须使用新建的空目录：

```
mkdir -- /tmp/ar-six-shard-real-plan-<fresh-id>
python scripts/pytest_sharding.py plan \
  --repository . \
  --tests-root tests \
  --output-dir /tmp/ar-six-shard-real-plan-<fresh-id> \
  --shards 6 \
  --profile full
```

### 6.3 Full acceptance

候选提交固定、资产根明确、无其他相同测试进程且主机资源可用时，才运行：

```
DEXTELEOP_ASSET_ROOT=/absolute/path/to/reviewed/assets \
DEXTELEOP_FULL_ACCEPTANCE_SHARDS=6 \
  bash scripts/run_full_acceptance.sh --durations=10
```

若 worktree 已有审核过的 assets 链接可省略资产根，否则必须使用绝对路径。每次运行使用新的 RUN_ROOT、JUNIT_DESTINATION 和 artifact root，不复用已有输出。

至少核对 shard-plan.json、aggregate JUnit、各 shard 日志、测试计数、失败分类和资源/环境信息。不要只看 wrapper 最后一行退出码。

### 6.4 Harness 受保护 Run

遵守生成的 Task/Run/lease/证据合同：

- 不在同一提交上并发启动第二个受保护 Run；
- 不因长时间无输出自行删除、重置或伪造完成；
- request、runner claim、execution/artifact receipt 和 PASSED 必须具有相同受保护 control-plane identity；
- 缺失 host-private transfer authentication 按 UNAVAILABLE；
- socket 受限失败保留并在能力匹配上下文重跑相同 Profile；
- 新提交、目标线移动、manifest/asset drift 或 runner 能力变化会使旧 readiness 失效。

## 7. 规则与限制

### 7.1 Git/worktree

- Harness Task 使用 Harness contract；独立 Git 候选才使用轻量 worktree 流程。
- 不 reset、clean、stash、删除或移动同伴 worktree 来制造干净。
- ready 必须绑定 clean commit、固定 base、验证命令、环境和已知 skips/failures。
- 新改动使旧 ready 失效，必须重新提交、盘点和声明。
- 集成在独立 integration worktree 进行；没有明确授权时不合并、不 push、不调用 protected activation。

### 7.2 测试语义

- 不减少完整 acceptance 节点，不改变 marker 语义，不用 skip/xfail 或放宽断言换速度。
- 分片按模块唯一分配；不能手工并发重叠模块。
- 不跨 shard 共享 pytest、MuJoCo、TMPDIR、Python bytecode 或 JUnit。
- 聚合报告拒绝重复 testcase，并保留失败 shard 的日志/XML。
- 默认并发度保持保守；新并发只能显式 opt-in，并以实际资源验证为依据。

### 7.3 设备、远端和资源

- 普通 pytest 与本候选不连接真实设备，不执行运动、归零、复位、伺服、力控、相机采集、VIVE 或远端 shell。
- 远端/硬件只经认可的 codex_remote 或 Harness 通道，不用 ad-hoc SSH/scp/rsync 绕过边界。
- 使用审核过的只读资产根；不把大型资源提交 Git。
- gitlink 保持提交中的精确 SHA；缺少内容时报告资源缺失，不借用其他任务临时目录制造绿色。

## 8. 集成、验收与回滚

### 8.1 集成前清单

集成者必须重新确认：

1. development 当前完整 SHA、状态和是否移动；
2. 候选 tip/tree 是否仍为预期；
3. 并行候选的 tip、状态、路径和语义重叠；
4. reBotArm 固定 gitlink 是否按允许流程物化；
5. 资产根、Python 3.10/dexteleop310、MuJoCo/EGL、CPU/RAM/swap 和 socket 能力；
6. 聚焦 9 项测试与 diff-check；
7. 最终组合提交先用默认 4 做完整 acceptance，再在资源确认后追加显式 6；
8. 保存 artifact root、每 shard 状态、aggregate JUnit、失败和未验证范围。

当前四项已知失败未关闭前，不能声称 full acceptance PASSED。

### 8.2 回滚

本批默认行为不变，回滚风险低：

- 集成前：不提升候选即可保持当前 development；
- 已集成：在固定 integration commit 上 revert 候选提交，再重跑验证；
- 主机资源不足：去掉显式 SHARDS=6，回到 4 或 1；
- 不删除原始 JUnit、日志或 Harness 证据，除非遵守正式 retention/release 流程。

## 9. 后续方向（不属于本批授权）

以下必须独立声明和验证：

1. agenticrobot-integration 固定两分片，并证明 node-ID 集合完全一致；
2. 将 deployment/RTC broad integration 从普通 task feedback 移到 integration phase，但不减少最终覆盖；
3. Harness fast feedback 在认证 runner 上显式使用 3 个 unit shard；
4. 在语义验证后研究 warning 收集、MuJoCo cache 复用或 collection 初始化优化；
5. 不跨实例共享 settled snapshot；scripted contact/pregrasp 存在 solver warm-state 风险；
6. 单独处理 descend-close 的顺序/solver/reset 敏感性；
7. 单独修复并验证 reBotArm gitlink 资源物化。

## 10. 交接呈现格式

后续汇报按以下顺序，避免混淆证据：

1. **结论**：候选是否可审查、可集成、已合入；不能时明确阻塞。
2. **固定身份**：仓库、目标线、候选分支、完整 commit/tree、base、worktree、owner、scope、artifact root。
3. **耗时与覆盖**：默认/显式 shard 数、模块数、计划/实际时间、墙钟、tests/skips/failures/errors。
4. **失败分类**：候选回归、基线失败、缺失资产/gitlink、socket/权限、超时/OOM、远端/硬件（本批为无）。
5. **命令和证据**：可复制命令、Python 环境、资产根、socket 能力、JUnit/log/plan 路径。
6. **未验证范围与下一步**：未运行的硬件/远端/Profile、下一步 owner、所需授权和重新盘点条件。

## 11. 当前结论

本批完成了低风险的 5/6 分片 opt-in 实现。聚焦验证稳定通过；六分片真实运行墙钟约 211 秒，显示资源充足主机上约 20% 的实际墙钟收益。由于完整运行仍有 gitlink 资源缺失和既有 MuJoCo 失败，且本机 swap 已满，默认值保持 4 是有意的安全决策。

本文档加入候选后必须重新提交、更新 branch description 的 write scope/tip/validation，并重新交接。除非完成集成前检查并取得明确授权，不能合并到 development、推送远端或声称受保护 acceptance 已通过。
