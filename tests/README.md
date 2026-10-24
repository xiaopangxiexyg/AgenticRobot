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
source modules and exact pytest targets. The temp pipeline and replay active
pipeline use this exception to share shard capacity without reducing test
coverage or physics work. Replay tests create their own temporary logs and
fresh runtimes; the four-rollout combined-control test and each parameterized
envelope matrix remain indivisible. Grasp trajectory stays module-local to
preserve its shared planning fixture instead of repeating that expensive plan.
The sparse full-suite weights come from testcase durations in a complete,
socket-capable green Run and cover only modules that contributed at least five
seconds there. Refresh them only from another exact green full-suite artifact;
failed, capability-limited, or focused runs are not representative inputs.
完整验证的模块和 replay 函数权重以绿色六分片基线 `47d7be6e` 为基础；其中
artifact migration / resolution 两个 I/O 模块按 `c3c06be6` 在 2026-09-25 的
绿色全量结果更新为 49.666 / 15.847 秒，避免旧估值将耗时集中在同一分片。
2026-10-18 的绿色完整基线 `298221fc`（六分片串行、Node 22）补齐了九个此前未加权、
实际累计至少五秒的模块，包括 DMTAC 接触、Gateway 安全位姿和 worker 清理等。
既有模块与函数权重保留，避免把单一远端的 I/O 差异当作全局变化。
绿色完整基线 `54edca481`（Run `full-20261023-mainline12j1geom`，十二批串行）
补齐十三个此前未加权的慢模块。2026-10-24 的绿色完整基线 `dfff99b81`
（Run `full-20261024-visualfull2`，十二批、两并发、Node 22、Trimesh 5.1.1 /
Embreex 4.4.0.post1）再补齐十三个累计至少五秒的新模块，包含 B601 calibration、
coordinated holding、staged assist 与 Gateway fast grasp 等测试族，并按新源码刷新
safe TCP pose、gripper contact、real RM65 composition 三个模块。
该基线有 18,107 个普通用例和 24 个子测试通过、16 个跳过；单并发运行因三十分钟
上限未完成，不作为绿色权重来源。其他历史模块与函数权重继续保留。
这些证据绑定固定源码和环境；选择并发度后仍需验证内存余量及完整测试结果。
这些值只用于调度，不是 timeout 或验收阈值；不改变测试内容、真实 `fsync`、
模块隔离边界、参数矩阵或 fast Profile 的权重。
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
