# MuJoCo 批量 RL 环境架构规划

## 2026-07-15 落地状态

问题 1A 和问题 2–9 已按 `legacy/mujoco.md` 的冻结决策落地；问题 1B 的多进程批量执行器也已完成。问题 1C（native substep loop）仍按计划保持为数据驱动的后续可选项。

已新增：

- `sim/mujoco/tasks/cube_lift/kernel.py`：无字符串热路径的 scalar fast kernel；
- `sim/mujoco/tasks/cube_lift/fast_env.py`：显式单环境 Gymnasium fast adapter；
- `sim/mujoco/artifacts/cache.py`：带版本 fingerprint、内容 hash、文件锁、原子 manifest 和损坏恢复的 MJB cache；
- `sim/mujoco/runtime/cube_lift_vector.py`：共享内存多进程 VectorExecutor 和 Gymnasium VectorEnv adapter；
- `sim/mujoco/benchmarks/benchmark_cube_lift.py`：reference/fast/vector 可复现 benchmark；
- `tests/test_mujoco_fast_runtime.py`：differential、snapshot、cache、selective reset、终止帧和生命周期测试。

本机长期 debug 基线：

```text
CPU                 AMD Ryzen 9 7950X, 16C/32T
Python              3.10.20
MuJoCo Python/native 3.10.0 / 3.10.0
NumPy               1.26.4
Gymnasium           1.3.0
OMP/MKL threads     1 / 1
benchmark           500 measured vector steps, 50 warmup steps

reference scalar    137.71 policy steps/s
fast scalar         312.72 policy steps/s   (2.27× reference)
vector 1            303.83 transitions/s
vector 4           1150.96 transitions/s
vector 8           1698.60 transitions/s
vector 16          2420.48 transitions/s    (7.74× fast scalar; 17.58× reference)

16-worker sustained stress (100k transitions, 100 warmup vector steps)
                   2892.96 transitions/s    (9.25× fast scalar)
                   PSS delta +73 KiB, USS delta -12 KiB

reference reset     approximately 10.9 ms
snapshot reset mean approximately 0.077 ms
MJB cache hit load  approximately 0.115 s
```

本机不是最终采样服务器，因此这些数字只作为长期相对性能回归基线，不是最终绝对吞吐验收。16 worker 的 8×/10×继续作为观察目标；不能通过修改 collision、physics Hz、solver 或执行器语义来满足该目标。

此次 100k-transition 压力测试未发现线性内存增长。`10^6` transitions 仍作为最终采样服务器或专门长稳测试的正式门槛，不在日常 debug 回归中持续占满本机全部物理核。

## 总体结论

当前 MuJoCo 框架已经具备“可验证的单环境 RL 雏形”，但它的设计中心仍然是：

> 单场景调试、控制时钟审计、数据回放、可视化和语义正确性。

它还不是一个面向大规模 RL 采样的执行架构。

首要目标不应该是继续完善数据集执行器、相机、场景真实性或真实执行器模型，而应该先把仿真链路拆成两条：

```text
调试链路
Gymnasium Env → Scene facade → 审计对象/名称/详细 info → viewer/export

高吞吐链路
Batch actions → Fast task kernel → MjData workers → batch tensors
                                  └→ 仅 episode 结束时输出详细信息
```

建议保留现有 `CubeLiftEnv` 作为 reference/debug environment，新增独立的 fast kernel 和 vector sampler，不在现有 facade 上不断打补丁。

当前本地默认 Python 环境没有安装 `mujoco`，因此尚无可信的实测吞吐数字；基准测试应作为第一个正式交付物。

---

## 一、现有框架分析

### 1. 已经具备的基础

现有代码分层总体方向是合理的：

- `sim/contracts/`：动作、观测、时钟、场景语义。
- `sim/mujoco/modeling/`：URDF/MJCF、碰撞和对象组装。
- `sim/mujoco/scene.py`：MuJoCo 模型和数据的通用 facade。
- `sim/mujoco/control/direct_qpos.py`：临时的 150 Hz direct-qpos 执行桥。
- `sim/mujoco/tasks/cube_lift/env.py`：第一个 Gymnasium 环境。
- reward、contact、scheduler、spec 已经从 env 中拆开。
- `tests/test_mujoco_cube_lift.py` 已覆盖 Gymnasium API、确定性、reset 和 termination。

这意味着不需要推翻现有结构，主要需要补上“高吞吐 runtime 层”。

### 2. 当前环境生命周期

当前单个 policy step：

```text
7D action
  ↓
更新 right arm target / gripper target
  ↓
15 Hz policy scheduler
  ↓
Python 循环 66～67 个 1000 Hz physics tick
  ↓
约每 6～7 个 physics tick 构造一次 150 Hz DirectQposCommand
  ↓
每个 physics tick:
  写双臂和双夹爪 qpos
  清零对应 qvel
  mj_forward
  mj_step
  mj_forward
  扫描状态、位置、全部 contact
  创建审计 tick 并永久保存
  ↓
构造 observation / reward / detailed info
```

这条链路语义清晰，但不适合批量采样。

---

## 二、当前主要性能瓶颈

### 1. Python 每 physics tick 调度

`sim/mujoco/tasks/cube_lift/scheduler.py` 在 Python 中逐 tick 循环。

按照当前配置：

- policy：15 Hz；
- physics：1000 Hz；
- 每个 policy step：66 或 67 次 Python 循环；
- 150 steps episode：约 10,000 次 Python physics 调用。

RL 训练中如果并行 64 个环境，每个 policy step 就需要约 4,267 次 Python 层 tick 调度。这个成本很快会超过 MuJoCo 本身。

### 2. 重复 `mj_forward`

`sim/mujoco/scene.py` 的 `step_kinematic_direct_qpos_debug()` 每个 physics tick 当前执行：

```python
mj_forward()
mj_step()
mj_forward()
```

第一个 `mj_forward` 与 `mj_step` 内部计算存在明显重复。最后一个 `mj_forward` 是否可以删除，要根据 post-step 所需的派生状态验证，但前一个通常可以合并掉。

这是必须通过 benchmark 和数值回归验证的首批优化点。

### 3. 逐关节 Python 写入

每个 tick 都会：

- 遍历 side 字典；
- 遍历 joint name；
- 通过字典查 qpos/dof 地址；
- 单元素写 `data.qpos`；
- 构造 `controlled_dof_addrs` 列表；
- 再逐元素清零 qvel。

地址虽然已经缓存，但热路径仍然使用字符串和 Python 循环。高吞吐路径应在初始化时生成连续或索引数组：

```python
arm_qpos_indices: int32[K]
arm_dof_indices: int32[K]
gripper_qpos_indices: int32[G]
```

运行时只做数组写入，不再接触名字、字典和 dataclass。

### 4. 审计历史无界增长

`sim/mujoco/control/direct_qpos.py` 每个 physics tick 创建一个冻结 dataclass，并追加到 `_ticks`。

一个 episode 约 10,000 个对象。批量训练时这是明显的：

- Python 对象分配；
- GC 压力；
- 内存增长；
- `ticks` 属性又会转换成 tuple。

训练模式只需要累计计数和最后一个 tick。完整审计应改成显式 debug 选项或固定长度 ring buffer。

### 5. 接触检测走完整语义对象链

每次 termination 检查都会：

1. 遍历全部 `data.contact`；
2. geom ID 转 geom name；
3. geom 转 body ID；
4. body ID 转 body name；
5. 创建 `MujocoContact`；
6. 创建 tuple；
7. 再按字符串筛选 cube/table。

训练热路径只需要两个布尔值，例如：

```text
cube touches table?
cube touches right gripper?
```

初始化时缓存目标 body/geom ID，physics tick 中直接比较整数。只有 debug/info 输出时才转换为名称和对象。

### 6. 每 tick 全状态有限性检查

`state_is_finite()` 每 physics tick 扫描完整 `qpos` 和 `qvel`。

MuJoCo 自身会报告数值 warning；训练模式可采用：

- 每 policy step 检查一次；
- episode 结束时检查；
- 或只检查任务相关状态；
- debug 模式保持逐 tick 全量检查。

### 7. observation 和 info 分配过多

每一步都会：

- 多次复制 cube qpos/qvel；
- 创建 4×4 body pose；
- `np.concatenate`；
- spec `coerce()` 再 copy；
- 生成详细字典；
- contact 转 list/dict；
- numpy 转 Python list。

训练 loop 通常只需要：

```python
obs[N, D]
reward[N]
terminated[N]
truncated[N]
```

详细 `info` 应默认只在 reset、termination、truncation 或 debug 模式构造。

### 8. reset 固定执行 100 physics ticks

当前 cube spawn 固定，reset 后每次仍 settle 100 ticks。

在暂时没有 domain randomization 的情况下，可在第一次稳定后保存 settled state snapshot；后续 reset 使用状态恢复，再更新必要随机项。这样 reset 成本可以从 100 次 physics step 降到一次状态复制和一次 forward。

### 9. 模型每个 env 都重新走 URDF→MJCF→编译

`MujocoScene.from_contract()` 会：

- 解析 URDF；
- 发现相机；
- 写临时 MJCF；
- 从 XML 编译 `MjModel`；
- 创建 `MjData`。

批量环境不应重复执行这条链。MuJoCo 支持把编译模型保存为 MJB；MJB 独立于 mesh 文件且加载更快，但它是版本相关的，因此仍应保留 MJCF/contract 作为源真值。

### 10. 当前物理模型并不轻

当前生成模型大致包含：

- 47 bodies；
- 61 geoms；
- 16 joints；
- 18 meshes；
- 42 collision excludes；
- 双臂、双夹爪、相机和大量 mesh collision；
- self/cross-arm collision 开启。

对于“临时 cube lift RL plumbing”，这明显偏重。尤其只控制右臂时，左臂完整碰撞、相机碰撞和高面数 mesh 都可能带来无必要的 broadphase/narrowphase 成本。

---

## 三、建议的目标架构

### 1. 分成三个层级

```text
Task definition
  reset / action transform / reward / done / observation layout
            │
            ▼
Fast simulation kernel
  cached integer indices / preallocated buffers / MjData ownership
            │
            ▼
Batch executor
  worker scheduling / shared memory / auto-reset / seeding / profiling
            │
            ▼
Adapters
  Gymnasium Env / Gymnasium VectorEnv / future PPO-SAC trainer
```

### 2. 推荐目录

```text
sim/mujoco/
  artifacts/
    compiler.py            # contract → MJCF/MJB/cache manifest
    cache.py               # hash/version/cache invalidation

  runtime/
    model_handle.py        # immutable MjModel + precomputed IDs
    state_layout.py        # qpos/qvel/body/geom index arrays
    scalar_kernel.py       # 一个 MjData 的无分配 fast path
    vector_executor.py     # batch reset/step 接口
    worker.py              # 多进程 worker lifecycle
    shared_buffers.py      # action/obs/reward/done shared arrays
    profiling.py           # 内部分段计时

  tasks/
    base.py                # TaskKernel protocol
    cube_lift/
      kernel.py            # fast reset/action/reward/done
      env.py               # 保留现有 debug/reference env
      config.py
      reward.py

  adapters/
    gymnasium_env.py
    gymnasium_vector_env.py

  benchmarks/
    benchmark_cube_lift.py
```

不建议把这些功能塞回 `scene.py`。`scene.py` 继续服务回放、viewer、诊断和可读 API；`runtime/` 则允许直接访问 MuJoCo 数组。

### 3. 核心接口

建议 fast kernel 对外只暴露批量结构：

```python
class VectorSimulation:
    num_envs: int
    action_shape: tuple[int, ...]
    observation_shape: tuple[int, ...]

    def reset(
        self,
        env_ids: np.ndarray | None = None,
        seeds: np.ndarray | None = None,
    ) -> BatchResetResult: ...

    def step(self, actions: np.ndarray) -> BatchStepResult: ...

    def close(self) -> None: ...
```

返回值固定为：

```python
observations: float32[N, 31]
rewards: float32[N]
terminated: bool[N]
truncated: bool[N]
episode_metrics: compact structured arrays
final_observation: optional float32[N, 31]
```

不要让核心 sampler 返回 N 个 Python info dict。Gymnasium adapter 可以在边界上按需生成。

### 4. TaskKernel 应保持无训练框架依赖

任务内核负责：

- reset state；
- action transform；
- 写控制目标；
- physics substeps；
- observation；
- reward；
- termination；
- episode statistics。

它不应该依赖：

- Stable-Baselines3；
- PyTorch/JAX；
- 数据集 writer；
- viewer；
- OpenCV；
- 相机 renderer；
- teleop runtime。

这样后续无论使用 PPO、SAC、CleanRL、TorchRL 还是自研 trainer，都不需要重写 MuJoCo runtime。

---

## 四、并行策略

### 第一阶段：多进程 VectorEnv

这是当前代码最稳妥的第一版。

建议：

- 一个 worker 进程持有一个或少量 `MjData`；
- worker 启动时加载一次 MJB；
- parent 和 worker 之间使用固定 shared-memory numpy buffer；
- action 和 observation 不走 pickle；
- 每个 worker 内禁止额外 BLAS/OpenMP 线程；
- 支持显式 CPU affinity，但不要默认硬编码。

当前机器是 Ryzen 7950X：

- 16 个物理核心；
- 32 个逻辑 CPU；
- 先 benchmark `N = 1, 4, 8, 12, 16, 24, 32`；
- 大概率 12～16 worker 是首个较好区间；
- SMT 是否有收益必须实测；
- 当前 swap 几乎占满，更要记录 RSS/env 和 page fault。

多进程的优点是可以绕开 Python GIL，也最容易兼容当前带 Python callback 的 termination 逻辑。

### 第二阶段：共享 `MjModel` + 多 `MjData` 原生线程

MuJoCo 官方推荐大规模数据并行采用：

- 一个只读 `mjModel`；
- 每个线程独立一个 `mjData`；
- 不需要额外同步。

但当前每 tick 有大量 Python 任务逻辑，因此直接换 Python thread 不一定有效。只有当物理循环进入原生代码，线程共享模型才真正有优势。

### 第三阶段：评估 `mujoco.rollout`

官方 `mujoco.rollout` 可以：

- 输入 batch initial state；
- 输入 batch control sequence；
- 原生循环多个 `mj_step`；
- 使用持久线程池并行；
- 释放 GIL。

不过它不能直接替换当前链路，因为当前执行器是：

> 每 tick 直接重写 robot qpos/qvel，再做 termination callback。

而标准 rollout 更适合 actuator `ctrl` 或其他标准 user inputs。因此建议：

- 当前阶段不要强行接入 `mujoco.rollout`；
- 等临时执行器改成 ideal position actuator，或者实现原生 custom stepping kernel 后再接；
- 先用 benchmark 判断 Python fast kernel + 多进程是否已经够用。

如果第一阶段仍不够，才考虑一个很小的 pybind/C++ extension，把下面的循环移入原生层：

```text
write held qpos
zero controlled qvel
mj_step
check relevant body/contact IDs
repeat substeps
```

---

## 五、模型侧的效率方案

建议增加独立的 `rl_fast_v0` 模型 profile，不改变当前 canonical/debug profile。

### `rl_fast_v0` 可做的裁剪

- 不加载 visual meshes。
- 不创建相机和 renderer。
- 移除相机壳体碰撞。
- 用 capsule/box/低面数 convex mesh 替代高面数 collision mesh。
- 当前任务只控制右臂时：

  - 移除左臂；或
  - 将左臂替换为少量静态 proxy；或
  - 至少关闭不相关的左臂 self/cross collision。

- 只保留：

  - table；
  - right arm；
  - right gripper；
  - cube；
  - 与任务真正相关的 collision pair。

- 明确设置 solver、integrator、iterations、tolerance，不依赖隐式默认值。
- 分别 benchmark 1000/500/250 Hz，但不能只凭速度直接降频。

### 降 physics Hz 的验收条件

只有满足以下条件才允许采用更低频率：

- 相同动作轨迹下成功/失败判定一致；
- cube 最终位置和最大穿透误差在预设容差内；
- 接触保持时长语义重新以秒定义，不直接复用 tick 数；
- 不增加明显 simulation warning；
- 吞吐提升足以覆盖语义变化成本。

临时场景可以简化，但模型 metadata 必须保留：

```text
physics_profile=rl_fast_v0
execution_mode=kinematic_direct_qpos_debug / ideal_position_servo_debug
robot_dynamics=not_identified
sim2real_validated=false
```

避免以后把高速调试模型误当成物理真值。

---

## 六、分阶段实施计划

### M0：建立基准和性能剖析

交付内容：

- `benchmark_cube_lift.py`；
- 输出 JSON 和简短 Markdown report；
- 测量：

  - model compile/load 时间；
  - reset/s；
  - scalar policy steps/s；
  - physics steps/s；
  - `mj_forward`、`mj_step`、contact、observation、info 各自占比；
  - N 环境 aggregate transitions/s；
  - RSS/env；
  - CPU utilization；
  - episode determinism。

当前 M0/M1 正式测试矩阵：

```text
env_count:      1, 4, 8, 12, 16, 24, 32
info_mode:      detailed, compact/terminal
model_profile:  canonical only
physics_hz:     1000 only
collision:      unchanged
solver:         unchanged
execution_mode: kinematic_direct_qpos_debug
worker_layout:  1 env/worker
```

`rl_fast_v0`、500/250 Hz、solver/collision 简化和多 env/worker 属于后续模型/执行探索矩阵，不混入问题 1–9 的性能归因。

这是后续所有优化的依据，不能跳过。

### M1：单环境 fast kernel

优先修改热路径，但不改变任务语义：

1. 缓存所有整数 ID 和 qpos/qvel index。
2. 预分配 observation 和临时数组。
3. 移除 command/tick dataclass 分配。
4. `_ticks` 改为计数器或可选 ring buffer。
5. contact 使用 body/geom ID 判定。
6. compact info 默认开启。
7. 合并冗余 forward。
8. settled reset snapshot。
9. 详细调试仍走原有 `CubeLiftEnv`。

验收建议：

- 同 seed/action trace 的 obs/reward/done 与 reference env 一致或在明确容差内。
- scalar policy throughput 至少达到原实现 2～3 倍，才进入下一阶段。
- 不允许用删除必要 termination 检查来伪造性能提升。

### M2：批量执行器

实现：

- `VectorExecutor`；
- Linux 多进程 worker；
- shared-memory actions/observations；
- selective reset；
- configurable auto-reset；
- terminal observation；
- worker exception propagation；
- clean shutdown；
- deterministic seed assignment。

种子建议：

```python
root = np.random.SeedSequence(master_seed)
env_seeds = root.spawn(num_envs)
```

每个 env 使用稳定的 `env_id + episode_id` 派生，保证 worker 数量变化时可复现。

验收建议：

- 16 个环境最低观察目标是 scalar fast kernel 的 8 倍（50% efficiency）。
- 16 个环境目标值是 scalar fast kernel 的 10 倍（约 62.5% efficiency）。
- 连续运行不少于 10^6 policy transitions，无 worker 泄漏、死锁和无界内存增长。
- 相同 seed/action batch 在两次运行中结果一致。

### M3：Gymnasium VectorEnv 和训练框架适配

只在这一阶段接训练框架：

- Gymnasium `VectorEnv` adapter；
- PPO/SAC smoke trainer；
- observation normalization；
- episodic statistics；
- checkpoint/resume；
- evaluation environment。

训练 smoke test 不追求任务成功率，只验证：

- 能持续采样；
- loss 能更新；
- reset 正确；
- terminal observation 没被覆盖；
- 没有把 `truncated` 当成 `terminated`；
- policy inference 能批量执行。

### M4：原生批量 physics

只有 M2 数据表明 Python/IPC 仍是主瓶颈时才做：

- ideal position actuator debug profile；或
- custom native stepping extension；或
- `mujoco.rollout.Rollout` 持久线程池。

此阶段的目标是：

- 一个共享 `MjModel`；
- 多个独立 `MjData`；
- native physics substep loop；
- Python 每个 policy step 只调用一次。

### M5：相机和数据输出

不阻塞 state-based RL：

- 训练 sampler 默认不渲染；
- evaluation workers 单独渲染；
- 视频只保存少量 episode；
- dataset writer 在 episode 结束后异步消费；
- image RL 以后单独设计 renderer pool，不进入当前首要任务。

---

## 七、建议的测试体系

### 正确性测试

- Reference env 与 fast kernel differential test。
- 相同 seed/action 的 reset 和 rollout 确定性。
- 单环境与 batch 中某个 env 的一致性。
- selective reset 不影响其他 env。
- success、out-of-bounds、time-limit 三类终止。
- terminal observation 正确保存。
- worker crash 能传播到 parent。
- close 后无残留进程和 shared memory。

### 模型测试

- MJB cache key 包含：

  - contract hash；
  - URDF/MJCF hash；
  - mesh dependency hash；
  - MuJoCo version；
  - fast profile version。

- cache 失效能重新编译。
- canonical/debug 与 `rl_fast_v0` metadata 不可混用。
- collision proxy 与目标 collision pair 有显式清单。

### 性能测试

性能测试不要作为普通 CI 的硬性绝对时间断言，否则机器差异会造成波动。建议：

- CI 只检查 benchmark 可运行；
- 指定采样机保存历史 JSON；
- 用相对退化阈值，例如低于最近稳定基线 15% 才报警。

---

## 八、首轮开发的明确优先级

推荐严格按这个顺序：

1. 建 benchmark，得到真实瓶颈比例。
2. 保留现有 `CubeLiftEnv` 作为 reference。
3. 实现无字符串、无 dataclass、低分配的 scalar fast kernel。
4. 增加 settled-state reset。
5. 增加 compact info。
6. 建多进程 batch executor。
7. 建 `rl_fast_v0` 无视觉、简碰撞模型。
8. 接 Gymnasium VectorEnv。
9. 最后才选择 PPO/SAC 训练框架。
10. 只有数据证明必要时，再开发 native rollout。

最重要的架构决策是：

> 不把“可读、可审计、适合调试的 Scene facade”同时当成“RL physics inner loop”。

两者共享 contract、模型和任务定义，但使用不同 runtime。这样既保留现有工程正确性，也能真正把采样效率拉起来。

---

## 参考资料

- [MuJoCo Python bindings 与 `mujoco.rollout`](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo 多线程仿真编程模式](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)
- [MuJoCo MJCF/MJB 模型加载与保存](https://mujoco.readthedocs.io/en/stable/modeling.html)
