# AgenticRobot test suite

The current compatibility layout keeps general pytest modules at
`tests/test_<owner-or-domain>_<feature>.py` and classifies execution semantics
with markers.  The primary goal is to make tests easy to find, extend, and run;
directory symmetry is not a goal by itself.  Keep behavior changes, test
refactors, marker changes, and file moves in separate reviewable batches.

## Concurrent development mode

When production code or tests are being changed in parallel, use logical
classification instead of physical migration.  Do not move or split a test
module while another owner is editing it. Update current imports, scripts, and
documented commands together when a move is justified; an old plan is not a
permanent path freeze. Marker-only governance changes may proceed when
they are isolated from the files under active ownership.

Treat existing root-level test paths as compatibility interfaces during
parallel development.  New general tests should also use a clear root-level
basename unless an owner already has a stable, reviewed test subtree.  Do not
create a new directory merely to classify one or two files.

A physical migration batch may start only when its exact file list has a short
exclusive ownership window, every file is tracked, and the relevant pre-move
baseline is green (or has a reviewed, exact list of unrelated failures).  A
new failure must be resolved before the move continues; it must not be folded
into the migration diff.

## Default suite

From the repository root:

```bash
bash scripts/run_pytest.sh --collect-only -q
# Bounded edit feedback:
bash scripts/run_pytest.sh -q -m "unit and not optional and not hardware and not slow_sim"
bash scripts/run_pytest.sh -q -m "contract and not optional and not hardware and not slow_sim"
bash scripts/run_pytest.sh -q -m "integration and not optional and not hardware and not slow_sim"
# Parallel bounded edit feedback (unit + integration, isolated caches/reports):
bash scripts/run_fast_feedback.sh
# Complete compatibility and integration-commit acceptance:
bash scripts/run_full_acceptance.sh --durations=50
# On the validated six-shard integration host, use the same setting as Harness:
DEXTELEOP_FULL_ACCEPTANCE_SHARDS=6 bash scripts/run_full_acceptance.sh --durations=50
```

The current `pytest.ini` default discovers the root `tests/` directory.
Package-local suites under `sim/sim_real_tests/tests` and `hardware/*/tests`
remain separate until their dependency and hardware semantics have been
audited. The sim-real suite now carries the same marker vocabulary and exposes
its own fast/full commands in `sim/sim_real_tests/README.md`, but it is still
selected only by an explicit path. Do not treat a successful default run as
coverage of those suites.

## Selecting evidence for a change

| Change | Development feedback | Before integration |
| --- | --- | --- |
| Documentation or Skill wording | Links, examples, relevant contract checks | Review the diff; no new text-matching tests by default |
| Behavior or bug fix | Focused owner tests, including the demonstrated failure | Affected consumers and failure/state transitions |
| Public contract, dependency boundary, refactor | Contract and owner tests; compare collection if tests move | Relevant cross-package suites and full non-hardware acceptance on the combined commit |
| Simulation/control dynamics | Deterministic replay or simulator cases with declared inputs | Relevant complete simulation acceptance; report hardware separately |

Use coverage, when available in the test environment, to locate missed changed
branches. For a bug fix, reproduce the fault when practical and check that the
regression assertion distinguishes the faulty behavior from the intended result.
Existing behavioral tests can suffice for a refactor; mock external boundaries
without mocking away the behavior being verified. Select relevant failure
branches: invalid input, cancellation, timeout, stale state, persistence failure,
and recovery. Do not prescribe an arbitrary repository percentage or create
duplicate tests to raise it. A minimal environment's optional skips are useful
feedback but do not satisfy acceptance for changed optional behavior. Record
missing dependencies and known unrelated failures explicitly.

### 显式按变更验收：先隔离 B601 长控制测试

分片解决并行负载分配；`--changed-since` 才选择本次需要运行的测试。
默认 `bash scripts/run_full_acceptance.sh` 仍运行完整非硬件集合。第一版只对
Gateway/感知独立更改省去七个 B601 长控制模块，其他模块全部运行，包括短
B601 安全检查、混合测试、MuJoCo、架构边界及本次新增测试。它不是通用的
“文件名没有 B601 就跳过”规则，也不整体省略 MuJoCo。

在干净的已提交工作树中，选择**同依赖、资产和 vendor 环境已完整绿色的固定祖先**：

```bash
# BASE_COMMIT 由本次开发/集成者固定，不能是移动中的分支名。
python3 scripts/pytest_impact.py inspect --repository "$PWD" --base "$BASE_COMMIT"
DEXTELEOP_FULL_ACCEPTANCE_SHARDS=12 DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS=2 \
  bash scripts/run_full_acceptance.sh --changed-since "$BASE_COMMIT" --durations=50
```

先检查 `inspect` 的 `changed_paths`、`mode`、`excluded_modules`、
`fallback_reasons`。`inspect` 可分析历史提交和脏工作树，但只提供建议，不能作为
执行证据。实际 runner 要求 HEAD、已跟踪文件、未跟踪文件和子模块均干净；
拒绝无法解析或非祖先的 base，在启动前及结束时核对源与分片清单，漂移则拒绝
成功证据。工具不认证基线的绿色记录、不认证外部环境等价，也不复用基线测试
结果；调用者需要保留基线完整验收及当前环境记录。并发上限不是内存容量保证，
仍使用所在主机既有内存门禁和资源上限。

| 变更/依赖情况 | 当前策略 |
| --- | --- |
| 全部更改在 `agent_gateway/`、`deployments/perception/`、`docs/agent_gateway/`、`tests/test_agent_gateway_*.py` 或 `tests/test_sam3_service.py` 内，且依赖核查通过 | 仅省去下列七个长模块，其余测试全部重新运行 |
| 共享运动、硬件、Control Station、sim、共享 fixture、runner、依赖或锁文件、其他未知路径 | 全量回退，并列出原因 |
| 删除/重命名、文件类型/权限变化、`__init__.py`、未知格式、显式 B601 引用 | 全量回退 |
| B601 在 base 或 HEAD 中的导入/资源依赖命中更改，或存在无法解释的动态加载、全局 `pytest_plugins` 声明 | 全量回退 |
| 无差异 | 全量；不把“没有改动”当作已有通过证据 |

固定可省略集合：`test_b601_automatic_calibration.py`、
`test_b601_calibration_cycle.py`、`test_b601_coordinated_holding.py`、
`test_b601_cruise_velocity.py`、`test_b601_workspace_calibration.py`、
`test_b601_workspace_identification.py`、`test_rebot_b601_staged_assist.py`。
不会删除、减参或 mock 这些测试；全量入口仍完整运行它们。

选择器沿两端 Git 树分析这些模块和公共 `conftest.py` 的静态导入、测试根 helper、
字面量资源及全部延迟导出。少量既有动态加载实现经源码摘要固定；它们漂移后
先回退全量，重新审查才能更新摘要。gitlink 是独立的版本边界，改动 gitlink
必定全量。该检查只为上述已审查所有权边界增加保守保护，不宣称能推导任意
Python 元编程的全部依赖；新增所有者或新的可省略测试族需要独立评审。

执行产生 `selection.json`、原有 `shard-plan.json`、各分片日志和
`changed-acceptance.xml`。JUnit 标明 change-aware、实际 full/affected 模式、
固定 base/HEAD 及省略列表；省略项没有伪造的 pass/skip receipt。
按变更模式仅接受 verbosity/durations 参数，拒绝额外筛选、插件和环境
`PYTEST_ADDOPTS`/`PYTEST_PLUGINS`，避免静默缩小已声明集合。默认完整入口的参数行为不变。

**affected 成功只证明清单列出的集合，不能声称完整验收成功。** 现有明确要求
完整验收的 Harness Profile/集成门禁继续调用无此选项的完整入口。本轮只接通
仓库 runner 的显式调用，没有激活或改写外部 Harness 配置。跨包公共契约、
共享控制/安全、依赖、环境或资产变化，以及最终组合提交的完整验证，继续使用
完整验收；不得把本入口冒充必需的 full evidence。

## Maintaining the test set

When behavior changes, first find and update its existing tests. Add cases for
unrepresented behavior or failure boundaries, not one new file per fix. Exercise
the full input matrix at the responsible layer; cross-layer tests verify real
wiring and important combinations. New trust boundaries, safety conditions and
recovery paths still need their own checks.

Assert observable contracts rather than private layout or historical wording.
Reuse helpers for actual repetition; parameterize shared setup with varying
inputs, without hiding unrelated scenarios in a large table. Before retiring a
case, identify the preserved behavior coverage or the approved retired behavior;
obtain applicable deletion/safeguard approval, reusing prior authorization.
Compare collection and assertions when reorganizing tests: equal counts alone
do not establish equivalent coverage. Record this reasoning in the existing
review, with measured runtime impact when execution cost motivates the change;
neither net line reductions nor test-count quotas are acceptance criteria.

### B601 虚拟反馈驱动的重复状态读取

`AcquisitionReplay.run` 与 cycle 的 `advance` 在没有 `until` 回调时，复用本次同步
循环已经读取的 `active` 状态；前者同时复用目标偏移，避免为驱动下一 tick 反复
深拷贝完整 GUI 历史。`tick` 的默认路径仍实时读取状态。存在回调时，heartbeat
与目标偏移沿用原有的再次读取行为，因为回调即使返回 `False` 也可能修改控制器。
回归用例用真实 `owner_stopped` 验证回调之后不会依据旧状态补发 heartbeat。

量化摩擦机械模型在每次 dispatch 后仍读取完整状态，下一 tick 的纯机械计算期间
复用其中的 `active`；进入返回阶段前重新读取。控制周期、1000 Hz 机械子步、
240/180 秒循环上限、摩擦与残余负载矩阵、原生回执、驱动阻尼、全部保护校验及
原有断言保持不变。这只减少测试驱动器中重复的观察开销，不改变生产状态接口，
也不把虚拟反馈证据当作真机验证。

完整双关节批次的 6 组 `test_full_coupled_batch_and_report_keep_handoffs_distinct_from_withdrawal`
也复用上一 tick 完成后已读取的状态，供下一轮终止判断和 heartbeat 使用。两次读取
之间只有测试中的纯机械计算、断言和被动 journal 观察，没有回调或控制器状态变更；
`arm` 后先重新读取。每次控制 tick 仍重新获取完整状态并调用真实 journal。
其他 `coupled_tick` 调用默认继续实时读取；6 组组合、全部控制/机械 tick、实际回执、
逐 tick 保护断言、撤力、持久化及报告检查保留。这不减少 journal 的采样次数，
也不替换生产状态接口、记录器或 `fsync`。

### 只读快照重试的采样历史

`test_persistent_failed_state_can_retry_without_restart` 在 `_capture_history`
测试输入端提供可控历史，分别验证过大采样间隔和关节移动被真实 `stationarity`
校验器拒绝，再用正常历史验证同一设备会话可以重试成功。它不替换校验器，
不依赖共享测试进程恰好获得足够密集的线程调度，也不改变生产采样周期、时钟或门限。
原有首次失败、第二次成功、机械臂和相机不重启断言保留；其他采集、历史选择、
freshness 和静止性边界测试继续覆盖原路径。该用例中的可控历史是测试输入，
不能作为真实设备采样时序的验证证据。

### Worker 清理故障测试的等待压缩

`test_owned_worker_cleanup.py` 的启动结果未知矩阵保留五类 owner 与两种异常，
继续验证未确认退出时依赖保留、同一线程迟到启动后才允许最终释放。故障注入明确没有
调用 native allocator，因此只在该线程实例的启动事件上使用真实 `Event.wait(0)`，
不再重复等待生产级的一至两秒预算；不替换 `join_and_confirm` 或伪造完成状态。
同时直接断言该线程的退出确认仍为 False，避免其他关闭门禁掩盖错误的线程完成判定。
真实迟到启动与最终清理前恢复原等待。独立用例使用事件握手和真实等待验证迟到启动可
唤醒退出确认；已运行线程、受损 liveness、join 中断等原有场景不使用该压缩。
生产超时、线程实现、安全判断及其他测试的等待均不变，这不是生产超时性能基准。

### MuJoCo 原生集成中的重复初始化与组合采样

`mujoco_gateway_fixture.py` 只在两个已有 Gateway 测试构造器中，将场景编译端口
接到已有 `ModelArtifactCache`，前提是显式配置 `SIM_MUJOCO_CACHE_DIR`。完整验收
沿用该次 Run 的输入哈希缓存；未配置它的普通 focused wrapper 保持原构造路径，
不自动启用缓存或新增缓存根。只复用校验过的编译产物，每次构造
仍加载独立 `MjModel`、`MjData`、planner、Gateway 和生命周期资源；不共享可变模型、
物理快照、待执行计划或运行中的会话。构造结束或抛出异常后恢复原端口。
生产构造器和缓存实现不改。冷编译与命中结果的模型参数、网格、接触、执行器、
初始状态逐项比较；修改一个会话的质量和 qpos 后，其他会话与后续加载必须不受影响。
原有 `test_mujoco_fast_runtime.py` 中的冷启动、损坏恢复、跨进程发布和失效检查仍保留。

缓存会话还有函数级测试 scope。`conftest.py` 在本例 fixtures 启动前建立 scope，
缓存构造器登记本例的会话引用，观察原 `session.close()` 的真实返回，不代替它关闭；
全部本例 teardown 完成时，要求完整 close 已成功返回 accepted，且应用已关闭，
再恢复测试观察包装、释放这些引用并执行一次 `gc.collect()`，归还等待
循环回收的原生对象。它不替测试调用 `close()`；若存在未关闭应用，保留引用并
让 teardown 失败，不进行这次回收；应用的 closed 标志不能替代完整清理成功。
没有缓存会话的测试不调用回收。未进入本例
scope 的高层 fixture／独立线程构造继续走原 factory，避免无人负责的缓存对象积累。
刻意验证完整 close 被拒绝的负例显式选择 `use_model_cache=False`，保留原拒绝断言和
原构造路径；不降低缓存会话的 accepted-close 证明，也不由夹具代替测试关闭资源。
冷、热构造都使用同一构造锁，保证临时编译端口恢复后，其他构造者才进入。

加速构造后，已关闭场景的原生内存可能在 Python 自动循环回收前积累。完整验收将
`test_agent_gateway_mujoco_fast_grasp.py` 和 `test_agent_gateway_mujoco_safe_tcp_pose.py`
加入现有的独立函数分片许可表，把测试族分配到同一批数的不同进程；每个参数族仍
不可拆分。两个模块只有函数局部会话，没有共享的运行中模块 fixture；跨进程只共享
该次 Run 的校验编译产物。其函数初始估值均分已有模块估值，没有使用中断验收或
focused 结果充当完整绿色计时证据。生产 GC 参数和行为、原生堆页归还实现、分片算法、
批数、并发上限及内存停止门禁均不变；上述显式回收仅在已关闭的缓存测试 scope
结束时执行。未来增加共享运行时 fixture 时，需要重新
核对这一分片边界。

所有权事务的 7 个失败/成功分支，以及锁等待预算 1 例、refinement 锁等待 4 例、
最终场景读取后的取消/过期 2 例，使用真实 `MujocoPlannedTcpMotion` 和真实可重入锁，
配合小型内存场景。它们原本就替换了几何查询，重复编译完整机器人没有增加该事务
的断言。保留全部拒绝条件、待执行计划身份和执行状态断言，并要求精确拒绝原因。
代价是这些分支不再各自验证完整原生 Rig/场景接线；真实双臂计划、执行、锁顺序、
接触保护、Gateway 消费与资源关闭仍由其余原生集成用例验证。

源分支还提出了下列覆盖压缩；本次与新增 Skill 组合时保留主线完整参数矩阵，
仅复用编译缓存、关闭后的回收、事务测试装配及分片改进。下表是源候选的历史
取舍，不表示当前主线缩减了这些组合。以后若单独评审矩阵压缩，再按该次组合
风险与验证结果决定；本次验收逐项保留原 testcase 身份。

| 测试族 | 主线保留用例 | 源候选采样用例 | 源候选保留与放弃 |
| --- | ---: | ---: | --- |
| safe TCP policy × 左右臂 × task | 12 | 6 | 保留所有两两参数组合和每例全部真实执行断言；放弃其余三因素组合 |
| DMTAC 左右手 × 方位 × 摩擦 | 8 | 4 | 保留所有两两参数组合及真实 lift、滑移、穿透和力学断言；放弃其余三因素组合 |
| native fast selector 顺序 × 三种决策 | 6 | 3 | 每种决策一次，包含正反候选顺序；不再为每种决策重建两个原生会话。轻量 search/real-provider 策略矩阵继续保留 |
| pose task 物理 feasibility oracle | 13 | 6 | 保留随机/固定 reach、普通/各向异性 contact、接触后 lift、完整 pick；去掉部分观察表示、固定初态和中间参数预设的独立 oracle |

feasibility 的受限 z/夹爪 contact 和 positive-z lift 仍分别由真实闭环的
`test_descend_close_task_is_feasible_without_hidden_xy_actions` 与
`test_lift_short_v2_zero_policy_holds_and_positive_policy_lifts` 覆盖；各预设的配置、
动作投影、观察字段、奖励和成功条件检查保留。它们不能替代被移除组合的完整
独立物理验收，因此新增物理路径或参数交互风险时，应增加针对该风险的原生用例。
B601 摩擦/残余力矩、真实外层截止时间、replay 释放尾段、physics tick、保护阈值、
运行器、并发上限和内存门禁不在本次压缩范围。

2026-10-25 的完整验收对照以 `1e38c0045942cc83bb08e6aae9414b06f6ae7a33`
为基线，候选为 `d2b3fc046c2ec8fa1d3474b299589fb26039bbea`。这批变更的目的
是改进测试时长，不改变生产控制行为。在同一远端测试环境、12 批次、最多 2 个
外层 pytest 的设置下，`run_full_acceptance.sh` 的墙钟时间由 1000.964 秒降至
920.650 秒，减少 80.314 秒（约 8.0%）；这是一次完整对照，不是稳定性能保证，
也不是整个合并工作流的总耗时。

候选有 18,597 个普通测试和 24 个 subtest 通过，无失败或错误。18 个 skip 的
身份与原因不变；声明压缩范围以外的 testcase 身份和结果逐项一致。移除 20 个
昂贵组合样例，增加 12 个缓存一致性、恢复、生命周期和分片回归样例。
原内存限制仍为 High 6 GiB、Max 8 GiB、Swap 0；峰值由 5.22 GiB 增至
5.68 GiB，High、Max、OOM 事件均为零。保留这项内存代价与上述覆盖收缩，
不能只引用 focused 测试的加速比例。证据只覆盖离线 CPU/MuJoCo，不证明真机验收。

### 主线更新后的分片估时

完整验收的估时已按 `development@e37c6a46d` 的一次完整绿色结果更新。该次
本机验收使用 12 批次、最多 2 个外层 pytest，20,913 个普通测试和 30 个 subtest
通过，18 个原有 skip 保留，脚本墙钟时间为 1751.578 秒。它与较早的远端测量
具有不同的主线、机器和运行条件，不能直接作为优化前后的速度对照。

新的模块估时包含新增的 native checkpoint、anchor、scene suite 和 B601 cruise
测试；已注册拆分的四个模块也使用本次各函数完整参数族的累计时间。未测量或新增
函数仍分配模块剩余估时。这里不增加拆分模块，参数族仍是不可拆的调度单位，
不改测试内容、集合、断言、并发上限、运行器、物理周期或内存门禁。
有原有 skip 的模块不能用不完整的执行时间下调旧估值；本次 Sim-HIL checkpoint
模块继续保留此前的完整估值。

分片重排还必须验证原生内存峰值。`test_observed_point_motion.py` 的三个 native
测试族沿用已有的显式 Run 缓存及函数级 accepted-close scope；每例仍构建独立
模型和运行时，完整关闭后才释放测试夹具引用并执行已有的测试侧 GC。普通未配置
缓存的 focused 构造继续使用原路径，不扩大生产缓存或修改生产 GC。
B601 cruise 的完整辨识用例只消费当前 audit 和最终 result，不再在测试驱动器的
`r.audits` 中保存每个 tick 的完整 GUI audit 副本；当前 `r.audit`、所有驱动校验、
控制 tick、真实回执、`r.records`、三组参数及原有断言均保留。其他测试仍可通过
原驱动器保留完整 audit 历史。

在固定该次 JUnit 用例耗时、按现有运行器的 FIFO 等待顺序重放时，预计减少批次
失衡和空闲等待；这不重新模拟 import、收集、分片变化带来的编译缓存竞争或
主机负载变化，不能当作已经实测的墙钟加速。实际收益与正确性仍以固定候选的
完整验收为准。

### Runtime artifact 重复迁移压缩

`test_runtime_artifact_migration.py` 将已发布迁移的幂等检查接在已有完整成功场景后，
复用同一份稀疏 WAL、硬链接和符号链接目录树。保留真实 `create_plan`、`copy_verify`、
独立 `verify_plan` 以及第二次 `copy_verify`；检查源身份哈希相同且 journal 不增加事件，
不再为幂等断言额外准备并发布一次相同迁移。

`test_runtime_artifact_resolution.py` 在真实发布、移动旧 source locator 并验证成功解析后，
对同一目标文件写入等长错误内容，再验证 content hash 拒绝。成功和损坏检查都使用真实
迁移计划与落盘内容，不再重复创建发布 receipt，也能检查成功读取后的内容是否被重新验证。

这两处各减少一个独立测试结果；前置断言失败时，后续断言不会执行。其余恢复、xattr、
外部硬链接、源变化、目标替换和未发布计划拒绝场景仍独立保留。生产 writer 扫描、真实
`fsync`、复制/哈希/发布实现和验收 selector 不变。迁移测试会反复扫描主机 `/proc/*/fd`，
成本依赖可见进程与文件描述符数量；受限进程环境和主机运行的耗时不能混作优化前后证据。

### 分类逻辑与架构检查的覆盖归属

`test_rm65_grasp_trajectory.py` 的三种可达性分类和 wall-time budget 用例
原本就使用固定 FK 残差的 `_ReachabilityBuilder`，现在直接复用本文件的
无资产 planner helper，不再构建随后被替换的真实 builder 和未使用的碰撞模型。
分类、残差、求解次数和超时断言不变；真实构造、IK、碰撞规划和 MuJoCo 执行
仍由原有集成用例覆盖，不能把这些分类用例当作真实模型证据。

禁止下层反向依赖 `control_station` 的检查统一由
`test_control_station_boundaries.py::test_lower_packages_do_not_reverse_depend_on_control_station`
负责。它使用相同 import 扫描器和三个历史 shim 白名单，覆盖原 contracts 检查的
全部七个包，并额外覆盖 `collection_runtime`、`control_composition`；删除 contracts
中较窄的重复用例，不改变 canonical 检查、白名单或验收 selector。

### 同输入仿真与模型构建去重

MJCF 的 `MeshAssetRegistry` 在一次转换内为同一路径复用 OBJ/MTL 视觉材质解析结果，
避免左右臂重复解析同一组网格。注册器由每次 `convert_urdf_to_mjcf_xml` 新建，
下一次转换会重新读取 OBJ、MTL 以及此前缺失的材质输入；不会复用 MuJoCo 模型或状态。
每次视觉调用仍构建独立属性字典、检查源网格存在，碰撞网格继续独立注册原始文件。
既有材质分区测试同时覆盖重复视觉的颜色、位置、属性隔离和原始碰撞网格，
新增参数用例检查跨转换输入更新，包含缺失材质库恢复。原有物理闭环与安全矩阵保持完整。

`test_mujoco_trajectory_composition_publication.py` 在同一真实 MuJoCo session 中
先验证 `residual_effective_until == target_effective_at` 的到期拒绝，再以相同
producer sequence 0 验证有效发布。拒绝发生在输入登记前；显式检查 editor hash、
proposal 数量与 trace 数量不变，且两路 input admission 均为空。随后保留原有真实
dispatch、FK 观测、因果链、重复发布、Hold 和旧 activation 拒绝检查，不再为
无副作用的到期负例单独建场/reset。底层纯逻辑到期用例仍独立保留；这里减少一个
独立测试结果，前置负例失败时后续闭环断言不会执行。physics tick、TTL 和安全条件不变。

`test_mujoco_fast_runtime.py` 的 direct / arm-only / fully-actuated 缓存隔离
使用同一空缓存中的三个真实首次构建，复用中间的 arm-only 结果做两组比较，
不再另起空缓存重复编译 arm-only。三种配置均断言 cache miss，保留 actuator 数量、
integrator 和 manifest fingerprint 检查，并检查三种 key 两两不同。
损坏恢复、跨进程发布和运行时行为测试仍使用各自原有的隔离方式。

`test_rebot_b601_mujoco_haptics.py` 的 `pose-6d` 平移与旋转指标断言并入已有的
七种 pose 参数化矩阵；矩阵中的七种场景仍真实运行 1.5 秒，不再为相同输入单独重跑一次。
接触、反射、确定性双回放及安全负例不变。这两处各减少一个独立测试结果；
合并场景的前置断言失败时，后续断言不会执行，不代表新的性能基准或硬件证据。

`test_mujoco_fast_runtime.py` 的 clearance 配置检查并入硬超限 episode 测试，先检查真实初始化的
scene/controller/planner 安全间距，再注入原有的 hard-limit 状态。该配置不随机化
spawn，原 seed 12/13 不构成不同场景；合并用例显式检查这一前提。它不因缺少
canonical planner 而跳过，保留原来的条件式 planner 断言以及 fallback 下的
controller 检查。小超限恢复用例仍独立保留，不能将其可选依赖 skip 扩大到配置检查。

### v2.5-p shadow 的规模与持久化覆盖

`test_v25p_dual_write_shadow.py` 的 CJSON 多帧成功用例使用 12 个不同数值的样本，
覆盖首帧、零/一起始索引和 9/10 文件序号转换，并逐帧检查真实落盘后的投影。
这是对原 256 个近似重复样本的规模压缩，不再声称覆盖 256 样本的 CJSON 长 episode。
中途/末尾 mismatch 用例还验证失败 receipt、native 数据保留和下一 episode 状态清理。
文件/目录同步故障用例只在指定失败点注入异常，其他同步调用仍执行真实 `fsync`；
它们检查未完成 capture 不会发布完成 manifest，不代表真实掉电恢复验证。

Parquet 的 256 样本、4 个 chunk 长测保留，并检查逐帧读回投影；该 optional 用例
不能替代无 PyArrow 环境下的 CJSON 覆盖。native-store 的两种格式精确对照、延迟
封存及真实 producer 多帧验证仍独立保留。这里不改变生产存储、同步策略或验收 selector。

### Sim-HIL 重复建场压缩

`test_control_station_mujoco_sim_hil.py` 将两组使用相同任务配置的检查合并，
每组只执行一次真实建场和 scripted reset，不共享跨用例的可变 runtime：

- `test_sim_hil_default_target_rates_and_measured_reset_session_home` 保留原默认频率、
  planner、安全间距、执行端口与 measured-reset TCP/session-anchor 的全部断言；
  原两个用例只在无行为意义的 composer ID 标签上不同，检查都不启动 runtime。
- `test_sim_hil_residual_schema_wiring_and_relative_human_submission` 在原人类输入
  变换场景启动前检查全部角色 schema，然后验证 identity anchor、去重和相对输入。
  原独立 schema 场景仅使用不同的 graph ID 标签，不提供额外状态转换覆盖。

自定义 compose rate、非 off observation plan 的实际 start/step、V5 专属复位、
hold/rearm 和所有失败路径仍独立保留。这里减少两个重复建场用例，不截短保留场景的
physics tick 或复位动作，不减少断言或验收 selector；合并用例的前置断言失败时，
后续断言不会执行。

自定义 compose rate、residual schema/相对输入和非 off observation plan 三个接线用例
使用已有 `rm65_right_cube_lift_rl_normal_full_grasp_stage1_v1.json`，不再为这些断言执行
contact curriculum 的完整脚本抓取准备。该任务与原场景采用相同 base contract、
arm/gripper profile、home、工具坐标及 1000 Hz physics；仍真实构建 MuJoCo、安装 runtime，
并保留输入提交或观测 start/step 和全部原断言。不共享跨用例的可变 scene/runtime。

这属于接线与任务初始化场景的解耦，不是生产 reset 的加速：不再重复覆盖三种接线配置与
scripted contact reset 的交叉组合。默认频率与 measured-reset session-anchor 用例仍执行
contact curriculum 的真实 reset，V5 scripted pregrasp reset 也独立保留。测试节点、
marker、skip 条件、physics 频率、安全间距、TTL 与验收 selector 均不变。

## Cost gates in parallel worktrees

Use the fast unit and integration selectors plus the exact tests for the owner
being changed on ordinary feature worktrees. Run the relevant exact
`slow_sim` node there only when the change can affect it. Run the full
`not hardware` gate, the broad `slow_sim` selection, and the package-local
sim-real full suite once for the exact designated integration commit; do not
make every parallel worktree repeat the same high-cost acceptance behavior.

`slow_sim` is the compatibility name for the measured high-cost tier. Assign
it from node-level timing evidence while keeping the unfiltered acceptance
behavior unchanged. Simulation, shared IK replay, simulation-side model
loading, repository-wide source scans, physical closed loops, and high-volume
persistence soaks may qualify.
A module-level marker is valid only when every collected node in that module
has been measured into the same tier. A filename, loop bound, optional
dependency, or `mujoco` import alone is not sufficient evidence. Do not share
`.pytest_cache` between worktrees or invoke independent pytest processes over
overlapping modules. `scripts/run_fast_feedback.sh` is the reviewed parallel
entrypoint for the two bounded selectors. It splits both unit and integration
selectors into two duration-weighted, deterministic test-module shards by
default, runs the four children concurrently, and merges reports back to the
historical `unit.{log,xml}` and `integration.{log,xml}` evidence paths. Each
child gets its own pytest cache, bytecode/cache roots, log, and JUnit report
under `/tmp`. Set `DEXTELEOP_FAST_FEEDBACK_UNIT_SHARDS=1` or
`DEXTELEOP_FAST_FEEDBACK_INTEGRATION_SHARDS=1` to reproduce a single process
for that tier. Three or four shards per tier remain opt-in because they add
pytest processes and are not part of the measured default. Explicit test paths
or node IDs automatically disable both tiers' sharding for that targeted
invocation so a test is never selected twice. If a clean worktree does not already have the reviewed
`assets/` link, set `DEXTELEOP_ASSET_ROOT` to an absolute shared asset root
before invoking it; the wrapper creates one shared temporary link for all
children and removes only that link after the run.  Additional pytest options
may be appended, but the marker, cache, JUnit, and basetemp options remain
owned by the wrapper.

`scripts/run_full_acceptance.sh` is the complete non-hardware integration gate.
Before allocating an artifact root or starting any shard, it requires the
checkout-relative B601 URDF under the recorded `hardware/reBotArm_control_py`
gitlink to be a readable file. Some complete-suite cases use that fixed input;
`DEXTELEOP_REBOT_SOURCE_ROOT` for other cases does not replace it. Missing input
fails immediately without initializing a submodule, changing inputs, or skipping
tests. This is only an availability check, not proof of model correctness or
gitlink identity; the original model and boundary tests still run.
The wrapper also verifies that the
host can create both loopback TCP and Unix sockets required by the complete
suite. An incapable sandbox therefore fails immediately instead of running the
remaining modules and reporting predictable transport errors at the end.
如果 `PATH` 中存在 Node.js，入口还会验证浏览器契约所需的语法和 `node:assert/strict`；
不兼容时在创建证据目录和启动分片前失败。Node 缺失时仍由原有可选测试报告跳过。
远端非交互命令应显式选择已有的兼容 Node 运行时，不能假设交互 shell 的版本管理器已生效。
It assigns every discovered `tests/test_*.py` module exactly once using a
checked-in sparse timing profile plus a deterministic source-size fallback.
Modules remain the isolation unit unless the full profile explicitly registers
an oversized module for top-level-function splitting. For such a module the
planner parses the source without importing it, assigns every statically
discovered top-level `test_*` function exactly once, and fails closed on test
classes or dynamic collection hooks. Parameterized functions remain a single
target, so their complete case matrix stays together. The manifest records both
source modules and exact pytest targets. The temp pipeline, replay active
pipeline and the two independent native Gateway modules described above use
this exception to share shard capacity and bound process-local scene
accumulation. The split itself preserves each complete function parameter
family. Replay tests create their own temporary logs and
fresh runtimes; the four-rollout combined-control test and each parameterized
envelope matrix remain indivisible. Grasp trajectory stays module-local to
preserve its shared planning fixture instead of repeating that expensive plan.
The sparse full-suite weights come from testcase durations in a complete,
socket-capable green Run. New entries require at least five seconds; existing
estimates may be refreshed when their module becomes inexpensive. Refresh them
only from another exact green full-suite artifact; failed, capability-limited,
or focused runs are not representative inputs.
2026-10-25 以绿色完整提交 `ec122623b`（Run `full-20261025-b601ec`，十二批、两并发、
Node 22、Trimesh 5.1.1 / Embreex 4.4.0.post1）的数据刷新 80 个已加权模块，
并补齐 observed-point 与 observed-safe-motion 两个累计至少五秒的新模块。
这次同时更新两个既有函数拆分模块的 33 个函数估值，保持函数注册表和完整参数矩阵。
有跳过、缺少实际记录或源码已在该证据之后改变的模块继续保留既有估值；历史来源
包括 `47d7be6e`、`c3c06be6`、`298221fc`、`54edca481` 和 `dfff99b81`。
此前优化已显著缩短部分 B601、MuJoCo 和 artifact I/O 测试，继续沿用旧数值会让某些
批次实际偏轻，并把等待留在验收尾部。这些是所测环境中的调度估值，其他主机的
CPU、文件系统和可选依赖会改变耗时，仍需验证完整结果与资源余量。
这些值不是 timeout 或验收阈值；不改变测试内容、真实 `fsync`、模块隔离边界、
参数矩阵、fast Profile、并发上限或内存门禁。运行器与分片算法保持原样。
The default remains four shards. Each shard has separate pytest, Python, XDG,
Matplotlib, Numba, `TMPDIR`, basetemp, log, and JUnit state. Shards in one exact
Run share only its content-addressed MuJoCo model cache; that cache uses process
locking, atomic publication, and hash validation, and it is never reused across
Run roots. The wrapper waits for every child, rejects duplicate JUnit
identities, and writes `full-acceptance.xml` plus `shard-plan.json` under a new
artifact root. Set
`DEXTELEOP_FULL_ACCEPTANCE_SHARDS=1` for single-process reproduction. Up to twelve
shards are supported, with at most six live shard processes. More than six
shards require an explicit `DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS`; they never
implicitly increase concurrency. Validate the CPU and memory headroom for each
chosen concurrency level. Keep the deterministic default instead of deriving
this setting automatically from `nproc`. Harness callers
may set new absolute `DEXTELEOP_FULL_ACCEPTANCE_RUN_ROOT` and
`DEXTELEOP_FULL_ACCEPTANCE_JUNIT` paths so the aggregate evidence remains in
the protected Run directory. Explicit test targets are rejected because this
entrypoint represents complete-suite acceptance.

完整验收会显式加载 `scripts.pytest_native_memory`：在当前模块的测试和 teardown
完成后，使用 Linux/glibc 的 `malloc_trim` 将已经空闲的原生堆页归还操作系统，
为主堆保留 128 KiB。它不强制 Python GC、不清空模型缓存、不关闭仍存活的资源，
也不改变测试、断言、并发或内存保护。其他平台或缺少该符号时保持原有内存分配行为；
实际加载或调用异常仍会使验证失败。普通 `run_pytest.sh` 不自动加载该插件。
此优化针对长批次中已释放却仍占 RSS 的原生内存，不替代对选定并发度的完整资源验证。

在内存受限的主机上，可以把分片数量与同时运行的进程数分开配置：

```bash
DEXTELEOP_FULL_ACCEPTANCE_SHARDS=6 \
DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS=1 \
bash scripts/run_full_acceptance.sh --durations=50
```

`MAX_JOBS` 必须介于 1 与 `min(SHARDS, 6)` 之间；六片以内未设置时仍同时启动全部分片，
默认四分片行为不变。七至十二片必须显式设置 `MAX_JOBS`，配置不完整或越界时在创建证据目录前拒绝。
上例保持六分片的完整目标集合、参数矩阵和隔离边界，逐个运行 pytest 进程，避免一个长期驻留
进程积累所有模块的原生内存。失败分片不会取消后续待运行分片，最终仍合并全部 JUnit 并返回失败。
这是并发上限，不是内存硬限额；进程还可能创建子进程，调用方仍需独立的资源预算和停止条件。
中断沿用直接子进程信号机制，不能据此断言 pytest 的全部后代已退出；需要可靠整组停止时，
由调用方使用 cgroup 或等价的进程树生命周期管理并验证退出状态。
较低并发通常增加墙钟时间，不能把它作为速度提升的证据。

当长 pytest 进程的常驻内存随模块加载和测试执行累积时，可以增加 `SHARDS`，同时保持已经
验证的 `MAX_JOBS`。例如十二片、单并发会让每个进程执行更短的批次，退出后再启动下一片；
仍使用同一完整目标规划器，每个目标及其全部参数只执行一次，模块级 fixture 的隔离边界不变。
更多分片会增加收集和进程启动成本，也不能降低单个测试模块本身的峰值。应比较完整 JUnit、
墙钟时间和内存记录后选择配置，不能仅凭分片数推断加速或安全并发数。

每个分片的 `TMPDIR` 位于本次运行独有的 `/tmp/dt-full-tmp-*/shard-*` 短路径，而不是证据目录。
Python multiprocessing/forkserver 使用 `TMPDIR` 创建 Unix socket，不读取 pytest 的 `--basetemp`；
因此即使证据根目录很长，也必须同时保持这两类临时路径短且独立。正常完成（包括测试失败）后
清理临时根；中断时保留并报告路径。日志、JUnit 和各类缓存仍保存在原证据目录，生产端的路径
校验不变。

## Marker semantics

- `unit`: deterministic in-process behavior.  This is the transitional default
  for modules not yet listed in `tests/conftest.py`.
- `slow_sim`: the legacy-compatible orthogonal cost marker for measured
  long-running deterministic simulation, IK replay, repository-wide source
  scans, physical closed-loop, or high-volume soak verification. It remains in
  the complete `unit`/`integration` and `not hardware` gates and is excluded
  only by an explicit fast selector.
- `integration`: multi-component, filesystem, worker, writer, or loopback
  behavior.
- `contract`: an orthogonal public schema, dependency-boundary, action, or data
  compatibility assertion; it may accompany `unit` or `integration`.
- `optional`: requires an optional dependency such as MuJoCo, OpenCV, Torch,
  PyArrow, or OpenPI.  Import optional packages with `pytest.importorskip` before
  importing project modules that require them so collection remains usable in
  smaller environments.
- `hardware`: requires a real device, SDK service, or operator-provided hardware
  environment.

The filename sets in `tests/conftest.py` are a transitional compatibility
mechanism that preserves current pytest node IDs.  New tests should carry the
most precise marker possible; unclassified tests must not silently grow into
multi-component or optional-dependency tests.

Use the `not slow_sim` selector for local edit feedback only. Marker changes
must not reduce the complete integration-commit gate, and a slow test may move
to this tier only when the same behavior is still executed by the full suite.

## Finding and adding tests

Use the production owner or domain in the basename, followed by the behavior
under test.  Prefer adding a related case to an existing feature module; create
a new `test_<owner-or-domain>_<feature>.py` only when the responsibility is
distinct or the existing module is no longer maintainable.

Before adding a test, identify:

1. the production owner or cross-owner flow it verifies;
2. whether it is unit, integration, optional, contract, or hardware;
3. whether an existing test module already owns that behavior.

Keep static inputs under `tests/fixtures/`.  Introduce `tests/support/` helpers
only when they remove real duplication and have a reviewed import contract.
Do not add wrapper test modules, path aliases, symlinks, or a second pytest
entrypoint to simulate an organizational hierarchy.

## Real-hardware gate

Plain pytest runs must never connect to or change a real device.  Hardware tests
must have a `hardware` marker, perform no device I/O while their module is being
imported, and are skipped unless the operator supplies `--run-hardware`.

The explicit invocation shape is:

```bash
bash scripts/run_pytest.sh -q -m hardware --run-hardware <exact-test-node>
```

That flag is permission, not a safety check.  Before running it, follow the
device-specific documentation, name the exact device and test node, verify
workspace clearance and stop behavior, and obtain the required motion or
device-state authorization.  Mocked backend tests must not use the `hardware`
marker.

Operator scripts that connect to devices or can perform calibration or motion
are not pytest tests.  Keep them out of both `test_*.py` and `*_test.py` naming
and migrate them to the owning package's explicit `tools/field_validation/`
directory in a separate batch with documentation and caller checks.  The
project config intentionally limits default discovery to `test_*.py` so a
legacy operator tool cannot be imported merely because its name ends in
`_test.py`.

The tracked field-validation scripts formerly kept in `tests/` were migrated
on 2026-08-09.  RM65 state/force/configuration tools now live under
`hardware/rm65/tools/field_validation/`; the combined camera/tactile health
probe lives under `teleop_runtime/tools/field_validation/`.  Each destination
README records the exact old-to-new path map and hardware gates.  These tools
are outside pytest collection and were verified offline only during the move.

## Optional long-term layout

Execution risk is the primary directory axis; package ownership is the second
axis.  `contract` and `optional` remain orthogonal markers rather than competing
top-level directories.  This is an optional long-term direction for owners
that become stable enough to migrate as a complete, reviewable group; it is not
the required location for new tests during active parallel development.

```text
tests/
├── unit/
│   ├── control_composition/
│   ├── control_station/
│   ├── robot_motion/
│   ├── teleop_runtime/
│   ├── deployments/
│   ├── dataset_contracts/
│   ├── dataset_interface/
│   ├── sim/
│   └── hardware/          # mocked backends only
├── integration/
│   ├── control_station/
│   ├── teleop_runtime/
│   ├── policy_deploy/
│   ├── dataset_pipeline/
│   ├── sim/
│   └── sim_real/
├── performance/
├── support/
└── fixtures/
```

Real-device/HIL pytest suites, if introduced, require a separately reviewed
location and retain the `hardware` marker plus explicit execution gate.

## Safe migration protocol

If a physical migration has a concrete maintenance benefit, migrate one domain
at a time only after its files are tracked, stable, and not being edited by
another owner:

1. Record collection count, test names, parameter IDs, markers, skips, and
   errors.
2. Remove directory-depth assumptions such as
   `Path(__file__).resolve().parents[1]` before moving the file.
3. Record the exact old-to-new path map, then use `git mv`; update exact
   test-node references in documentation and scripts.
4. Compare pre/post collection while allowing only the documented path-prefix
   change.
5. Run the smallest affected tier plus `git diff --check`.
6. Never use an ordinary test migration as authorization for hardware access.

The optional layout above supersedes the narrower simulation-era example in
`sim/STRUCTURE_PLAN.md`.  Logical classification, old-path compatibility, and
reliable collection take priority over visual directory uniformity.
