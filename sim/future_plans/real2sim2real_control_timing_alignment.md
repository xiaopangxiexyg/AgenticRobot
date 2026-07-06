# Real2Sim2Real Unified Control Timing Alignment

## 背景

当前 MuJoCo 离线查看工具主要是逐帧 `set_qpos`。这适合快速检查数据列和几何姿态，但它不等价于真机执行链路：

- 数据集通常是 10-15 Hz 级别的采样。
- 真机侧存在更高频的 arm control loop、限速、目标平滑、阻抗/位置控制器、硬件内部插补和机械带宽。
- 直接把 15 Hz 的离散 qpos 写进 MuJoCo，会把真实执行中被控制链路吸收的跳变视觉放大。

但这个问题不应该只在 replay/viewer 里解决。后续无论是 dataset replay、实时 sim teleop、policy/RL rollout、sim2real 部署前 dry-run，还是最终真机执行，都应该共享同一套“真机执行时序”语义：

- 低频 observation/action/tracker 信号如何映射到高频 control tick。
- 每个 tick 使用哪一个 source timestamp、插值权重、hold/stale 策略和延迟补偿。
- target smoothing、速度/加速度限制、IK、防碰撞和 backend command 的状态如何跨 tick 演化。
- 记录到 dataset/export/metadata 里的时间轴是否能还原当时的执行语义。

因此，如果目标是 real2sim2real，系统需要抽象出统一的执行时序层，而不是只给离线 replay 做一个视觉平滑补丁。

## 目标

让 sim replay、实时 sim teleop、policy/RL rollout、部署前 dry-run 和真机执行尽量使用一致的控制时间轴、插值规则和安全约束，同时保留 raw replay 用于诊断。

核心目标：

- 支持把低频 dataset/action/tracker/policy 信号转成真机控制频率附近的控制流，例如 50 Hz 或 100 Hz。
- 把真机控制 tick 的 `dt`、timestamp 选择、插值、hold/stale、delay compensation 作为统一配置，而不是散落在 replay、sim env、teleop 和部署脚本里。
- 在 sim、RL 和 real backend 中复用和真机一致的平滑、限速、IK、碰撞约束语义。
- 明确区分“诊断用 raw 结果”和“更接近真机执行的 control-like 结果”。
- 为后续 sim2real、RL 训练/评估、部署前验证和真机上线提供统一的动作时序和安全约束入口。

## 总原则

### 1. Real-Control-Clock First

真机控制频率和执行 `dt` 是 canonical clock。其他数据源只是在这个 clock 上被采样、插值或 hold：

```text
source timestamp/action/tracker/policy output
  -> UnifiedExecutionScheduler at real control dt
  -> smoothing / delay / stale handling
  -> IK / collision / limit
  -> sim, replay, RL env, or real backend
```

### 2. Replay Is One Consumer

Replay 只是统一执行时序层的一个 consumer。不能出现 replay 有一套插值逻辑、RL env 有另一套 step 逻辑、部署脚本又自己处理 delay 的情况。

### 3. Timing Metadata Must Be First-Class

任何导出的 episode、rollout、video 或 debug trace 都需要记录：

- source signal frequency and timestamps。
- execution/control frequency。
- interpolation/hold/stale policy。
- configured command latency / observation latency。
- smoothing、limit、IK、collision profile version。
- backend type: `raw_replay`、`sim_control`、`rl_env`、`real_robot` 等。

## 必须保留和统一的模式

### 1. Raw Replay

当前逐帧播放模式继续保留。

用途：

- 诊断 dataset 中的原始 qpos/action/tracker target 是否突变。
- 观察 IK 分支跳变、碰撞约束切换、tracker 卡顿等问题。
- 避免重采样或平滑把真实问题抹掉。

### 2. Qpos Resampled Replay

对已经得到的 qpos 序列做时间重采样和速度限制，再送入 MuJoCo。这个模式仍属于诊断/过渡方案，不应该成为 RL 或部署的主执行语义。

用途：

- 更接近真机“低频上层目标 + 高频底层执行”的视觉效果。
- 快速比较 15 Hz raw playback 和 50/100 Hz playback 的差异。
- 作为 dataset replay/export 的第一阶段改造，风险较低。

建议能力：

- `--execution-hz 50|100`
- 线性 qpos 插值。
- 按关节速度上限裁剪插值步长。
- 可选低通滤波，但默认优先使用显式限速，避免隐藏问题。

### 3. Control-Like Execution / Replay

对 tracker pose 或 action pose 做时间重采样，然后在 50/100 Hz 控制频率下重新进入：

```text
source target pose/action
  -> target smoothing
  -> shared RM65 IK
  -> collision-aware solve
  -> joint/pose step limiting
  -> MuJoCo qpos/control
```

用途：

- 最接近真机 real teleop 的控制链路。
- 用于判断一个 real trajectory 在 sim 中按真实控制频率执行时是否稳定。
- 后续 real2sim2real、RL policy evaluation 和部署前验证的主要评估入口。

注意：这个模式比 qpos 插值更重要，因为它模拟的是控制链路，而不是只美化已经求好的 qpos。

### 4. Online Sim / Real Execution

实时 sim backend 和真机 backend 应该消费同一个 scheduler 输出：

```text
TeleopCommand or PolicyAction
  -> UnifiedExecutionScheduler
  -> shared smoothing / IK / collision / limits
  -> MuJoCoArmBackend or RM65RealBackend
```

用途：

- 保证 sim teleop 不是“viewer-like qpos follow”，而是真机控制链路的 backend 替身。
- 部署前可以把 policy/teleop 指令先跑到 sim backend，再切换到 real backend。
- 同一段输入在 sim 和 real 之间只替换 backend，不替换时间差值逻辑。

### 5. RL / Policy Rollout

RL 环境不能直接把 policy output 当作每个 MuJoCo physics step 的 qpos 或 target。policy output 应该先进入统一执行时序层，再由执行层按照真机控制频率推进。

建议语义：

- `policy_hz` 可以低于、等于或高于 `execution_hz`，但必须显式配置。
- `env.step(action)` 产生的是上层 action，内部执行若干个 real-control ticks。
- sim physics `model.opt.timestep` 可以更高频，但控制命令只在 `execution_hz` tick 更新。
- observation latency、action latency、hold-last-action 和 stale timeout 与真机部署保持一致。

## 建议落地阶段

### Phase 1: 通用执行时序模块

核心位置建议不要只放在 `sim/` 下，而是放到 runtime/shared 层：

```text
teleop_runtime/execution/timing.py
teleop_runtime/execution/scheduler.py
sim/mujoco_real2sim/timing.py  # thin adapter / compatibility wrapper
```

职责：

- 根据原始 timestamp/frame_time/policy_step_time 生成目标控制时间轴。
- 定义 `ExecutionTimingConfig`，包含 `execution_hz`、`control_dt`、latency、stale timeout、interpolation/hold policy。
- 输出统一的 `ExecutionTick`，包含 target time、source frame 区间、插值权重、是否 stale、是否 extrapolated。
- 支持 qpos 插值。
- 支持 pose/action 插值或 hold。
- 支持关节速度/单步上限限制。
- 支持 debug trace，便于所有 consumer 复现同一条时间轴。

验收：

- 单元测试覆盖非均匀 timestamp、缺帧、重复 timestamp、短序列。
- raw qpos 在原始 frame time 上采样后能保持端点一致。
- replay、sim teleop、RL env 可以消费同一种 `ExecutionTick`/`TimingTrace` 数据结构。

### Phase 2: Dataset Qpos Replay 接入

位置建议：

```text
sim/mujoco_real2sim/replay/runner.py
```

新增模式：

```text
--playback-mode raw
--playback-mode qpos_resampled
--execution-hz 50
```

验收：

- `raw` 保持现有行为。
- `qpos_resampled` 在 viewer 和 export dataset 中都使用同一套时间轴。
- 导出的 metadata 记录 `playback_mode`、`execution_hz`、速度限制参数。
- 明确标记该模式是 diagnostic/visualization，不作为部署或 RL 的最终控制语义。

### Phase 3: Control-Like Execution 接入

位置建议：

```text
teleop_runtime/execution/control_pipeline.py
sim/mujoco_real2sim/replay/control_like.py  # replay entrypoint
```

职责：

- 从 dataset 读取 tracker-derived target pose 或 action pose。
- 在目标控制频率下重采样 target pose。
- 调用共享 RM65 IK/双臂防碰撞求解。
- 输出 backend command、sim qpos 和诊断指标。

验收：

- 能和 `robot_motion/tests` 中的离线 tracker compare 使用相同核心逻辑。
- 能同时输出 raw/action/source/control-like 的误差统计。
- 支持短窗口调试，例如 `--start-frame`、`--max-frames`。
- 可以在不改 timing/pipeline 逻辑的情况下切换 `MuJoCoArmBackend` / `RM65RealBackend`。

### Phase 4: Teleop Sim / Real Backend 统一

目标是让实时 sim teleop、真机 teleop 和离线 dataset replay 使用同一套执行语义：

```text
TeleopCommand
  -> UnifiedExecutionScheduler
  -> MuJoCoArmBackend or RM65RealBackend
```

验收：

- 真机和 sim 的默认 target smoothing、step limit、IK tuning、collision scene profile 一致。
- sim 能以 50/100 Hz 执行，而不是只跟随上层 observation/action 频率。
- 同一个 episode 可以输出 raw replay、qpos-resampled replay、control-like replay 三套可比较结果。
- 同一段 teleop 输入可以在 sim backend 和 real backend 间切换，metadata 中只变 backend 字段。

### Phase 5: RL / Deployment 接入

位置建议：

```text
teleop_runtime/execution/policy_adapter.py
sim/mujoco_real2sim/rl_env.py
```

职责：

- 将 policy output 适配成 `PolicyAction` / `TeleopCommand`。
- `env.step(action)` 内部按 `execution_hz` 推进 scheduler 和 backend。
- 将 observation/action latency、hold policy 和 stale timeout 配到与真机一致。
- 支持在 RL evaluation 中导出和 replay/control-like 完全同构的 timing trace。

验收：

- 同一个 policy action 序列在 RL env、offline rollout 和部署 dry-run 中产生一致的 execution tick 序列。
- 可以显式测试 `policy_hz != execution_hz` 的情况。
- 部署前 dry-run 与真机运行使用同一份 `ExecutionTimingConfig`。

## 评估指标

需要同时看平滑性和忠实度：

- source timestamp 到 execution tick 的分配正确性。
- policy/action/tracker latency 的配置值和实际 trace。
- qpos step max/rms。
- qpos velocity max/rms。
- TCP translation/rotation error against recorded action pose。
- collision avoidance status/fallback count。
- source target 与 final command 的延迟和误差。
- sim replay/export/RL rollout 后的视频是否与真实视频时间对齐。
- sim backend 与 real backend 在相同输入下的 command trace 差异。

重要原则：不能只用“看起来更平滑”作为成功标准。

## 风险和边界

- 过度平滑会掩盖 tracker stale、IK 分支跳变和碰撞约束突变。
- qpos 插值不能代表真实控制器，只能作为低风险的视觉和数据过渡方案。
- control-like replay 需要非常注意初始 qpos、平滑器内部状态和 collision solver 状态，否则会和真机路径漂移。
- RL env 如果直接用 physics step 频率更新 action，会训练出依赖仿真细节而不符合真机控制带宽的策略。
- 部署脚本如果绕过统一 scheduler，会重新引入 sim/real 时序不一致。
- real2sim2real 的最终判断应依赖统一控制链路，而不是单独的 viewer 美化。

## 推荐优先级

1. 先抽出 `ExecutionTimingConfig` / `UnifiedExecutionScheduler`，把真机控制 `dt`、插值、hold、stale、latency 语义固定下来。
2. 接入 `raw` / `qpos_resampled` 双模式，解决 sim 视觉播放与 dataset export 的时序问题，同时验证 timing trace。
3. 实现 `control_like`，用于复现 tracker/action/policy 进入共享 IK 后的真实控制链路。
4. 把实时 sim teleop backend、真机 backend、离线 replay 和 RL env 都接到同一套 scheduler。
5. 将部署配置和 RL evaluation 配置统一到同一份 `ExecutionTimingConfig`，确保后续训练、验证和上线使用一致的时间差值。
