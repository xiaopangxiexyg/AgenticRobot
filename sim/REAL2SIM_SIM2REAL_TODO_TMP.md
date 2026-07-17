# Real2Sim / Sim2Real 当前框架复核与 TODO（临时）

> 复核日期：2026-07-17
> 复核范围：`sim/` 当前实现、跨域 contract、MuJoCo 场景/控制/任务/runtime、数据回放与导出、仿真性能、临时 RL pipeline，以及它们与 `robot_motion/`、`dataset_contracts/`、`teleop_runtime/` 的边界。
> 本轮操作：只更新本报告文件；没有修改代码、测试、配置或依赖，没有启动 GUI viewer、真实设备或任何机器人运动。

## 0. 一句话结论

项目已经拥有一条可复用的 MuJoCo real2sim 基础链路：版本化的
`SimulationContract`、RM65 v3 场景构建、真实 qpos replay、RGB/depth/pointcloud
导出、独立的 v2.2 policy-I/O 导出、direct-qpos debug 控制、碰撞安全投影、CubeLift
reference/fast/vector runtime、MJB artifact cache，以及一个明确标为临时的 PPO smoke
pipeline。

但是，当前仍不能称为“完成了 sim2real”。现有真实含义主要是：

- MuJoCo 中的物体、桌面、地面和接触物理可推进；机械臂/夹爪大部分是 prescribed
  kinematic qpos，不是已辨识的 RM65 actuator/controller。
- `CubeLift` 的 `contact_latch_debug` 是为了让临时 RL plumbing 可运行的 task-local
  约束，不是物理抓取模型，也不能作为真实抓持稳定性证据。
- 仿真和真实已经共享部分动作、TCP、频率和布局语义，但不同入口仍使用
  `15→75→150`、`15→60→120`、`15→45→150`、`15→150→1000` 等不同 profile；当前
  `SimulationContract` 也还不是把数据、标定、代码、artifact、工具和运行 trace 全部
  绑在一起的 cross-domain run manifest。
- 真实 replay、sim replay、native v2.2 export、task rollout 和未来 policy artifact 之间
  还没有一条逐 tick、带 source/target frame、clock、validity 和 dispatch/applied state 的
  可审计 control timeline。

当前应把项目标记为：**仿真架构和 debug/RL plumbing 已明显前进，真实 actuator、物理抓取、
跨域时间线、系统辨识和 sim2real gate 尚未闭环。**

## 1. 复核证据与工作区口径

### 1.1 本轮只读检查

本轮使用了静态阅读、import/help、测试和工作区检查；没有做任何设备状态变更。

- 目标报告在本轮开始前已经是用户并行修改状态，本轮保留其余工作区改动。
- 阅读了 `sim/README.md`、`sim/mujoco/README.md`、`sim/configs/README.md`、
  `sim/SIM_ENV_TODO.md`、`sim/STRUCTURE_PLAN.md` 及其引用的关键实现。
- 阅读了 `sim/contracts/`、`sim/mujoco/control/`、`collision_safety/`、
  `tasks/cube_lift/`、`runtime/`、`artifacts/`、`adapters/`、`replay/` 和
  `dataset_v22/`。
- 相关仿真测试：`50 passed, 28 skipped in 5.95–6.02s`。
- 根项目此前的本地全量结果：`1668 passed, 49 skipped, 2 failed`；两个失败属于
  deploy execution-branch 校验和 real teleop refactor 行数冻结，不在本报告中修复。
- `python -m sim.mujoco.run_control --help` 等入口 help/import 检查通过；没有启动 viewer。

### 1.2 当前解释器与 skipped 的含义

本轮当前 shell 使用 `/home/zc/anaconda3/bin/python` 的 Python 3.12.2，而项目目标环境是
`dexteleop310` / Python 3.10。相关仿真测试的 skipped 原因已明确：

| 原因 | 影响 |
|---|---|
| 当前 shell 没有 `mujoco` | 真实 MuJoCo scene、direct qpos、collision、fast/vector、CubeLift 物理测试未在本轮实际执行 |
| 当前 shell 没有 `gymnasium` | CubeLift Gymnasium API、reset/step、100 次随机 headless 检查未在本轮实际执行 |
| 当前 shell 没有 `pyarrow` | native v2.2 parquet writer 的落盘测试跳过 |
| 本地没有指定 replay dataset | 一个 replay source dataset 测试跳过 |

因此 `50 passed` 主要证明不依赖这些可选组件的 contract、解析、CLI 和纯逻辑部分；不能写成“当前环境已完成 MuJoCo 物理验收”。目标 Python 3.10 环境仍需另行重跑。

### 1.3 历史 benchmark 的使用口径

`sim/SIM_ENV_TODO.md` 的 2026-07-15 落地段记录了一组 Python 3.10.20、MuJoCo 3.10.0、Gymnasium 1.3.0、单线程 OMP/MKL 的历史基线：

| runtime | 历史记录 |
|---|---:|
| reference scalar | 137.71 policy steps/s |
| fast scalar | 312.72 policy steps/s，约 2.27× reference |
| vector 1 | 303.83 transitions/s |
| vector 4 | 1150.96 transitions/s |
| vector 8 | 1698.60 transitions/s |
| vector 16 | 2420.48 transitions/s |
| 16-worker sustained stress | 2892.96 transitions/s / 100k transitions |
| reference reset | 约 10.9 ms |
| settled snapshot reset | 约 0.077 ms |
| MJB cache hit load | 约 0.115 s |

这些数字是文档中的历史 debug-machine baseline，本轮没有在当前 Python 3.12 shell 重跑，不能作为当前绝对吞吐验收。`SIM_ENV_TODO.md` 其他段落仍保留“当前本地没有 MuJoCo、尚无可信实测数字”的旧表述，文档内部存在时间状态冲突，后续应以带环境 fingerprint 的 benchmark JSON 为准。

## 2. 目录边界与系统地图

### 2.1 当前实际分层

```text
real collection / replay / deploy
          │
          ▼
robot_motion ───────────────┐
  action / TCP / IK / limits │
  resampling / safety        │
                             ▼
sim/contracts/ ────────> sim/mujoco/
  ActionSpec              modeling / scene / control
  ObservationSpec         replay / rendering / tasks
  TimingSpec              collision_safety / adapters
  SceneSpec               artifacts / runtime / benchmarks
  SimulationContract             │
                                 ▼
                       sim/mujoco_temp_pipeline
                       temporary simulation-only PPO

dataset_contracts <──── sim/mujoco/replay + sim/mujoco/dataset_v22
dataset_interface      (format adapters; not hardware)

sim/rm65/  ── SAPIEN renderer/preview only
```

当前责任应保持如下：

| 层 | 责任 | 当前边界 |
|---|---|---|
| `robot_motion/` | `RobotAction`、TCP/frame、RM65 coupled IK、限位、位置 resampling、trajectory policy | sim/real 共用语义；不拥有 MuJoCo scene 或训练算法 |
| `sim/contracts/` | action、observation、timing、scene、collision 和 contract version | 纯数据/语义；不初始化硬件 |
| `sim/mujoco/modeling/` | URDF/MJCF、visual/collision geom、group/mask、ground/table/object | 物理后端建模，不放 task reward |
| `sim/mujoco/scene.py` | 可读的 `MjModel`/`MjData` facade、FK、qpos、contact、camera | reference/debug API，不应同时充当高吞吐 inner loop |
| `sim/mujoco/control/` | direct qpos command、command deadline、ZOH、物理 tick audit | 只消费完整 joint command，不做 pose IK |
| `sim/mujoco/tasks/` | CubeLift task config、action/observation、reward、termination、latch | task-local，不得回流到 real/deploy contract |
| `sim/mujoco/runtime/` | fast scalar kernel、shared-memory vector executor、Gymnasium adapter | 训练采样热路径，降低对象/字符串/info 分配 |
| `dataset_contracts/` | v2.2 schema/layout/parquet/validity 等纯契约 | 不应依赖 MuJoCo 内部对象 |
| `sim/mujoco/replay/` | 真实数据 qpos/gripper replay、媒体诊断和 legacy export | 明确是 kinematic diagnostic，不是动态机器人 rollout |
| `sim/mujoco/dataset_v22/` | native v2.2 policy-I/O sim export | 已绑定 real collector layout，但仍是 sim 数据 producer |
| `sim/rm65/` | SAPIEN RM65 renderer/preview | 不是 canonical task physics 或执行后端 |
| `sim/mujoco_temp_pipeline/` | 临时 PPO smoke/evaluate/play/checkpoint/records | 不是长期 `model/` 包，也不是 real deployment policy |

`sim/STRUCTURE_PLAN.md` 规划的顶层 `model/` 当前仍不存在。PPO/BC/SAC 等长期训练域不应因为临时 pipeline 已出现就直接把 `sim/` 变成训练包。

### 2.2 后端结论

- MuJoCo 是 canonical task-physics backend，负责 real2sim、物体接触、task scene、后续 system ID 和 sim2real checks。
- SAPIEN 只负责 teleop/debug preview、tracker mapping 和 operator visualization；其 renderer 在 `sim/rm65/`，入口仍在 `teleop_runtime`，不能作为第二套 task-physics source of truth。
- `sim/mujoco/replay/` 可以直接设置 qpos，用于相机/几何/数据回放诊断；其产物必须保留 kinematic/debug provenance，不能称为 dynamic manipulation rollout。

## 3. 跨域 contract 现状

### 3.1 `SimulationContract v1` 已经做了什么

`sim/contracts/` 现在把以下内容版本化：

- `ActionSpec`：绑定现有 canonical deploy action layout、26D action、`xyz_quat_xyzw`、
  `tool_tcp` controlled frame、左右 arm base reference frame、6-slot hand 和
  `gripper_close = 0=open, 1=closed`。
- `ObservationSpec`：绑定现有 38D measured qpos/TCP/hand state、front/left wrist/right wrist
  image keys、HWC uint8 RGB 和 `latest_not_after_observation` causal timestamp policy。
- `TimingSpec`：独立声明 sample、policy、reference、IK、command、physics、camera clock；明确
  `timestamp_linear` interpolation 和 command tick 间的 ZOH，不假设频率比必须是整数。
- `SceneSpec`：绑定 MuJoCo backend、RM65 v3 URDF、RealSense registry、gravity、ground、free-body
  object 和 collision profile。
- `SimulationContract`：组合上述 spec 和 metadata，并在 JSON load 时检查 nested schema version。

这是跨域语义的必要基础，但它还不是完整的 run manifest：目前没有强制绑定 code revision、
dataset/episode hash、policy artifact、标定 snapshot、tool TCP profile、camera preprocessing、
postprocess history、随机种子、实际 dispatch/applied timeline 和 fidelity limitation 的单一 contract。

### 3.2 当前 profile 的关键事实

`sim/configs/rm65_bimanual_physics_v1.json` 显式固定了：

- MuJoCo canonical scene；
- gravity `[0, 0, -9.81]`；
- ground 和 URDF 中的一张 table collision body；
- 一个 `0.03 × 0.03 × 0.03 m`、质量 `0.05 kg` 的 `debug_cube` free body；
- cube 初始中心约为 `(-0.412, 0, 0.026)`，table top `0.01 m`，保留 `1 mm` settle clearance；
- collision safety 参数：minimum clearance `0.002 m`、hard penetration `0.001 m`、sweep max joint step `0.01 rad`、projection iterations `8`；
- physics `1000 Hz`、sim camera `30 Hz`；
- `actuator_model_pending` 和 `real_150hz_system_id_status=pending_data`。

该 JSON 没有显式写 reference/IK/command 频率，因此 `TimingSpec` 会继承项目默认的 15→75→75→150 口径；而 `sim/README.md` 仍把 canonical profile 描述为 15/45/150。这是当前必须修正/冻结的文档与 profile 漂移。

`sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json` 是 task-local config，不是第二个 `SimulationContract`。它通过 base contract SHA-256 绑定场景，并冻结：右臂控制、左臂 home、7D action、31D privileged observation、15 Hz policy、150 Hz command、1000 Hz physics、cube reset、episode limit、height threshold、reward 和 `contact_latch_debug` provenance。

## 4. Real2Sim 当前执行链路

### 4.1 Canonical physical scene / MJCF 编译

推荐入口是：

```bash
python -m sim.mujoco.run_control \
  --contract sim/configs/rm65_bimanual_physics_v1.json \
  --dump-mjcf-only \
  --mjcf-out /tmp/rm65_bimanual_physics_v1.xml
```

当前建模链路包含：

```text
SimulationContract
  → resolve URDF / camera registry
  → parse URDF
  → split visual 与 collision geom
  → material-aware v3 OBJ/MTL visual partition
  → collision group/mask + explicit excludes
  → ground/table/freejoint cube composition
  → MJCF compile
  → MujocoScene(model, data)
```

v3 CAD color 的 visual partition 只影响 visual mesh；collision mesh 和 contact behavior 不因颜色转换改变。derived OBJ 会放入 `/tmp/dexteleop_mujoco_obj_materials_v1/<source-hash>/`，属于可再生缓存，不应提交。

### 4.2 真实数据 qpos replay

`sim.mujoco.replay.runner` 当前是 kinematic diagnostic 主线：

- qpos source `auto` 的优先级是 measured/actual → current → action/replay target；
- gripper source 可从 scalar action、state/measured/commanded 或 hand vector 显式选择；
- 支持 dual/right replay，right-only 时将 inactive left arm 固定到首帧；
- 通过 timestamped action → reference → command schedule 生成 replay qpos；
- 能从 RealSense registry/source dataset 读取 intrinsics，并将 RGB render 做 real-raw distortion inspection；
- metric depth 仍是 MuJoCo render camera frame，不等价于 raw RealSense z16；
- 支持 RGB、wrist depth/mask、pointcloud 和 legacy LeRobot-style sim dataset export；
- viewer、record、export 都是诊断/数据产物路径，不是 hardware feedback loop。

旧 `sim/mujoco_real2sim` Python namespace 已移除；磁盘 task id 或数据格式中的
`mujoco_real2sim_replay` 属于稳定 metadata，不是可导入的旧包。

### 4.3 Native v2.2 sim export

`sim/mujoco/dataset_v22/` 已从 legacy replay writer 分离，直接使用
`dataset_contracts.v22.PolicyIOLayout` 和真实采集的
`dual_arm_bimanual_dmtac_gripper_v22.json`：

- RM65 双臂 DM-Tac layout 的 model-facing action/state 都是 16 个 continuous dimensions；
- 写入 layout feature metadata、layout hash、typed fixed-size Parquet columns；
- 写入 source frame index/timestamp、qpos source、gripper source、URDF/MJCF、scene path、camera metadata 和 control/reference rate provenance；
- `meta/modality_validity.json` 明确 force/torque 和 tactile unavailable，不填伪造零值；
- 当前 native v2.2 路径只接受 canonical RGB + low-dimensional contract，不接受 wrist depth/pointcloud fallback；
- legacy `--export-dataset` 路径仍负责较丰富的媒体、depth/mask/pointcloud 诊断。

这解决了旧报告中“sim writer 完全没有接入 v2.2 layout”的问题，但还没有把 `SimulationContract`、policy artifact、calibration/tool snapshot、source dataset hash 和完整 control trace 收进统一 `CrossDomainManifest`。

## 5. 控制时序与入口 profile

### 5.1 当前 profile 表

| 路径 | 上游 sample/policy | reference / IK | command | physics | 当前性质 |
|---|---:|---:|---:|---:|---|
| `TimingSpec` 默认 | 15 Hz | 75 / 75 Hz | 150 Hz | 1000 Hz | shared default；base JSON 未显式覆盖 |
| `MujocoPositionExecutionRig` 测试/real-like bridge | 通常上游 15 Hz | 60 / 60 Hz | 120 Hz | 1000 Hz | atomic dual-arm qref→qcmd simulation adapter |
| CubeLift task/reference/fast/vector | 15 Hz | 不经过 task-local IK | 150 Hz | 1000 Hz | 7D policy → direct qpos debug；左臂完整 hold |
| replay control schedule | source action rate | 默认 replay reference profile | 默认 150 Hz | timestep 可配置 | kinematic replay / media diagnostic |
| native v2.2 writer defaults | source dataset | 45 Hz metadata | 150 Hz | — | export metadata；不等价于 actuator sim |
| VIVE MuJoCo teleop debug | tracker source 15 Hz | coupled solver 60 Hz | qcmd 120 Hz | 1000 Hz | arms-only headless debug bridge |

这里的多个数字并非都错误，它们代表不同的 runtime profile；问题是 profile 名称、JSON、README、replay export metadata 和测试之间尚未完全集中管理。尤其是：

- `sim/contracts/timing.py` 默认 reference/IK 来自共享 `PositionControlRates()`，当前是 75/75；
- `sim/README.md` 仍描述 physical profile 为 15/45/150；
- `MujocoPositionExecutionRig` 的目标是 60/120；
- CubeLift task 是 15/150/1000；
- native v2.2 writer 的 reference metadata 默认是 45、control 150。

TODO：将 profile 名、实际数值、frame、interpolation、hold、camera rate 和 physics provenance 统一放进同一可序列化 registry，并在 contract hash/episode metadata 中记录，而不是依赖入口默认值或 README 文字。

### 5.2 Shared MuJoCo position adapter

`sim/mujoco/adapters/robot_motion.py` 已实现 `MujocoPositionExecutionRig`：

```text
absolute RobotAction / TeleopCommand
  → atomic dual-arm qref mailbox
  → shared acceleration-limited joint trajectory planner
  → 120 Hz qcmd proposal
  → DirectQposDebugController
  → 1000 Hz prescribed MuJoCo physics
```

它已经具备：

- active sides 必须一次性提供完整双臂 joint-vector qref；
- bounded latest-reference mailbox，不积累过期 IK frame；
- `hold_motion()` 清空待处理 reference 并 quick-stop planner；
- 每侧 planner proposal 在 direct controller 接受后一起 commit；
- planner commit 失败时重置到 MuJoCo 实际 hold qpos，避免 planner generation 混用；
- dispatch 返回 queued receipt、reference sequence 和 worker snapshot；
- `step_physics()` 可用于 deterministic/offline replay，worker 可用于 real-time-like debug。

这条路径证明了“共享 real-like position semantic”已经能进入 MuJoCo，但 metadata 仍明确是 `kinematic_direct_qpos_debug` / `robot_dynamics=not_simulated`。它不提供 actuator tracking、force control、firmware queue latency 或真实 qpos feedback。

## 6. Direct-qpos、碰撞安全与物理边界

### 6.1 Direct-qpos 已实现的边界

`DirectQposCommand` 和 `DirectQposDebugController` 当前已实现：

- 一个 command 包含完整 active side qpos、按名称的 normalized gripper close、单调递增 sequence 和 source；
- command sides 必须与 active sides 完全一致，拒绝缺 side、未知 gripper、非有限/越界值、提前 command 和非递增 sequence；
- 使用 timestamp deadline，不假设 `1000/150` 或 `1000/120` 是整数；
- 每个 physics tick 重新写 held arm/finger qpos，按配置清零 controlled qvel；
- 保留 command due/applied、active sequence、safety status、projection alpha、minimum distance 和 collision pair 的 tick audit；
- 不做 pose IK、qpos interpolation、PD、torque、gravity compensation 或 target tracking；
- 只接受上游已经形成的完整高频 qpos command。

参考 `CubeLiftEnv` 保留详细 tick objects；fast kernel 将数值索引缓存、减少字符串/对象/info 分配，默认只在 terminal 输出详细 info。

尚未完成的是可跨 real/sim/replay/dataset 消费的持久化统一 trace：当前 tick audit 没有统一外部 timestamp、source validity、task/scene hash、action lineage、measured/applied qpos 和完整 contact event schema。

### 6.2 Collision safety layer

当前 MuJoCo prescribed motion 已接入 `sim/mujoco/collision_safety/`：

- precompute forbidden geom pair oracle；
- 对 candidate qpos 做 scratch `MjData` kinematic check；
- 沿 joint-space segment sweep，发生不安全时二分投影到最后安全点；
- gripper closure 与 arm movement 分阶段投影；
- target 被修改时记录 `projected/rejected/invalid_safe_state` 等状态；
- `mj_step` 后对真实 contacts 做 hard penetration check；
- reference 和 fast CubeLift 共用同一 policy；
- task object/table 与 gripper/object contact 可以作为合法物理 task contact，robot/environment、non-gripper/object 和 self collision 按 forbidden pair 处理。

当前 profile 参数是 minimum clearance `2 mm`、hard penetration `1 mm`、最大 joint sweep step `0.01 rad`、8 次 projection iteration。相关测试覆盖 active data 不变、向桌面运动被投影、reference/fast 一致、unsafe target hold-last-safe、clearance termination 和 observation 使用 collision-approved qpos。

这仍然不是 sim2real safety 证明：

- 真实机器人不存在完全等价的 scratch `MjData` 几何检查；
- 120 Hz qcmd 连续段只在选定仿真路径有检查，真实硬件的 endpoint/segment certification 未完成；
- hand/tool 当前使用固定 proxy，不代表任意定制工具网格；
- prescribed qpos 的接触穿透和冲量不能代表 actuator 运动下的真实碰撞响应；
- 真实 arm/hand dispatch、固件队列、延迟和 measured feedback 还未接入该 safety contract。

## 7. CubeLift task 与临时 RL pipeline

### 7.1 CubeLift v0 当前语义

`rm65_right_cube_lift_rl_debug_v0` 当前是一个 task-local、privileged-state 的 Gymnasium 环境：

- action：7D `float32 Box[-1, 1]`，右臂 6 个 normalized joint target delta + 右手 gripper close delta；
- observation：31D `state_privileged`，包括右 qpos/close、cube pose/velocity、cube→grasp vector、last action 和 progress；
- left arm：固定完整 home qpos；每个 command 仍提交完整双臂 state，避免 passive left arm 下坠；
- object：固定桌面中心附近的 `0.03 m` cube，当前没有实际 reset randomization distribution；
- clocks：15 Hz policy、150 Hz command、1000 Hz physics；
- reset：重置 qpos/qvel/controller/latch，执行 100 physics ticks settle；可使用 settled snapshot；
- termination：finite state、robot/environment penetration、cube out-of-bounds、time limit；
- success：cube 高于 settled center `0.08 m` 且离开 table contact，持续 20 physics ticks，名称为 `height_only_lift_success`。

这个 success 条件**不要求 latch active、不要求 bilateral contact、不要求 relative-pose hold**。因此“高度成功”仍可能被抛起/非真实接触等行为触发，这是当前 task 明确的 loophole，不能写成 grasp success。

### 7.2 `contact_latch_debug` 的实际含义

`ContactGraspLatch` 是 task-local scaffolding：当 gripper close 达到 `0.45`、grasp center 距 cube 不超过 `0.06 m`、且左右 finger body 都与 cube contact 时，记录 grasp-to-cube transform，随后把 free cube 跟随 grasp frame；close 降到 `0.2` 以下释放。

它的价值是让临时 reach→close→lift→reward→checkpoint 链路可测试，但它不是：

- MuJoCo actuator 或 finger force model；
- friction/mass/inertia 下的自然抓持；
- real hand coupling、contact force 或 release dynamics；
- 可迁移到 `robot_motion`、`teleop_runtime` 或真实 deployment 的约束。

TODO：把三个指标分开记录：`height_only_lift_success`、`contact_latch_debug_success`、未来的 `physics_contact_grasp_success`。在 actuator/夹爪物理和 reachability gate 完成前，不能把前两个合并为 physical grasp success。

### 7.3 Reference / fast / vector runtime

当前已经形成两条运行层：

```text
reference CubeLiftEnv
  → readable scene facade
  → detailed contact / scheduler / tick info
  → viewer / diagnostic / semantic regression

FastCubeLiftEnv / CubeLiftKernel
  → cached integer qpos/dof/body indices
  → compact info and preallocated observation
  → settled-state snapshot reset
  → MJB model artifact cache

CubeLiftVectorExecutor / CubeLiftVectorEnv
  → one scalar kernel per spawned worker
  → fixed shared-memory action/observation/reward/done arrays
  → pipe only for commands and compact metadata
  → selective reset / deterministic seed / same-step auto-reset
  → terminal observation and worker error propagation
```

这部分已经覆盖了原计划中的 scalar fast kernel、MJB cache、shared-memory multiprocess vector executor、Gymnasium vector adapter、bounded shutdown 和 benchmark。下一步不应继续把 viewer/debug info 塞进热路径，而应先用目标环境 benchmark 证明瓶颈是否仍在 Python/IPC，再决定是否需要 native substep loop。

### 7.4 临时 PPO pipeline

`sim/mujoco_temp_pipeline/` 已提供：

- 7D/31D CubeLift policy adapter；
- PPO smoke trainer、normalization、checkpoint、evaluate 和 viewer play；
- staged reach/grasp/lift reward shaping；
- collision projection/rejection penalty 和 safety metrics；
- task config SHA-256 compatibility gate；
- records 下的 run config、metrics、episodes、summary、HTML report 和 checkpoint。

它明确使用 privileged state、right arm/right gripper、`contact_latch_debug`，并被文档标为 temporary simulation-only。`--allow-task-config-mismatch` 只允许 legacy collision diagnostic，不能把不匹配 checkpoint 当作正常训练/评测结果。

当前没有证据证明该临时 pipeline 已在本轮环境完成训练、稳定成功率或 sim2real。它也不应成为正式 `model/` 入口；长期 PPO/BC/SAC、rollout buffer、模型 artifact 和评测应迁移到规划中的 `model/`/`deployments/` 边界。

## 8. 数据、媒体与跨域对齐

### 8.1 已有能力

- legacy replay writer 能写 LeRobot-style Parquet、RGB videos、wrist depth raw/video/mask、pointcloud manifest 和 replay metadata；
- native v2.2 writer 能写 real-compatible typed low-dimensional fields、layout hash、source timestamps、camera provenance、episode records 和 modality validity；
- replay 使用 source intrinsics/fovy、RGB distortion model、metric depth frame 和显式 qpos/gripper source；
- MuJoCo camera pose、RGB、depth、pointcloud 和 qpos 可以作为几何/媒体诊断输出；
- `dataset_contracts.v22` 已被 native sim writer 使用，旧 v2.1-style storage 仍可保留兼容路径。

### 8.2 当前缺口

1. legacy export 与 native v2.2 export 仍是两套 writer；前者媒体丰富，后者 layout 正确但当前不写 wrist depth/pointcloud。两者还没有统一 writer contract 和 common provenance envelope。
2. `sim_provenance.json` 和 episode metadata 能记录 simulator、URDF/MJCF、source dataset、control/reference rate、camera metadata 和 modality unavailable，但还不能唯一绑定 calibration snapshot、tool TCP profile、collision profile、code revision、policy artifact 和完整 action/control trace。
3. sim image 的 metric depth、RGB distortion、RealSense intrinsics 对齐仍是几何/媒体一致性，不是 sensor noise、exposure、latency、rolling-shutter、tactile/force dynamics 的真实复现。
4. 没有 golden paired episode：同一个带 timestamp 的 source action 是否能在 real replay、sim replay、task rollout 中逐 tick 对齐 qpos/TCP/image/object/contact，尚无统一 verdict。
5. sim export 的 force/torque 和 tactile 明确 unavailable；不能为了满足训练列形状而伪造零值，也不能把 absence 当作真实传感器负例。

TODO：建立 `CrossDomainRunManifest` 或等价纯数据 contract，绑定 `SimulationContract` hash、PolicyIOLayout/PolicyArtifact hash、source episode/parquet hash、code/env fingerprint、calibration/tool/camera snapshot、timing profile、domain/fidelity limitation、seed 和 trace path。

## 9. 真实 actuator、抓取和 Sim2Real gate

### 9.1 当前没有完成的物理层

- RM65 joint actuator、servo/ideal position actuator、damping/friction/armature、latency、tracking residual 和 controller gains 未完成 system identification。
- full articulated robot 在 gravity 下直接 `mj_step` 会因无 actuator 让 passive joints collapse；这不是 valid robot rollout。
- 夹爪当前是 normalized scalar → finger qpos mapping，加上 task-local latch；没有可靠 mimic/coupling、finger pad force/contact、motor delay、closing force 和 grasp stability model。
- force control、impedance、gravity compensation、real hardware streaming 和 measured state feedback 不是当前 MuJoCo mainline 能力。
- real 150 Hz system-ID status 在 canonical profile 中仍是 pending。

### 9.2 Sim2Real gate 还缺什么

在任何 policy-only real deployment 之前，至少需要下列 gate：

| gate | 当前状态 |
|---|---|
| action layout、TCP、joint order、units、frame hash 一致 | 部分已有，需统一到 run manifest |
| source/policy/reference/command/physics clocks 可追溯 | 各入口已有局部 metadata，未统一 |
| qref→qcmd 与真实 controller deadline/latency 对齐 | 仿真 adapter 有 60→120→1000 debug path，真实认证未完成 |
| actuator tracking residual 与 measured qpos 对齐 | 未完成 system ID |
| collision segment、tool geometry、table/camera scene 对齐 | 仿真有 projection；真实连续段和任意 tool 未认证 |
| gripper contact/force/grasp/release | 当前只有 qpos/latch debug |
| real/sim paired episode 与阈值 verdict | 未建立 |
| camera RGB/depth/noise/latency/tactile fidelity | 部分几何/媒体工具存在，未形成 domain gate |
| artifact/layout/task/config compatibility | task/checkpoint hash 有局部 gate，未统一跨域 |
| fail-closed、hold、dispatch receipt、fault recovery | runtime 局部存在，未形成 sim/real 同一 trace |

因此，“MuJoCo 任务能 reset/step”“随机 action 不崩溃”“PPO loss 能更新”“sim replay 能导出视频”都不能单独作为 sim2real 通过条件。

## 10. 当前 TODO 优先级

### P0：先冻结可审计的跨域语义

- [ ] 建立统一 timing profile registry，明确每个入口使用的 sample/policy/reference/IK/command/physics/camera rate；修正 `sim/README.md`、base JSON 默认继承和 replay/native writer metadata 的不一致。
- [ ] 建立 `CrossDomainRunManifest`：绑定 `SimulationContract`、PolicyIOLayout、artifact、source dataset、frame/TCP/tool、camera preprocessing、collision profile、IK tuning、seed、code/env fingerprint 和 fidelity limitation。
- [ ] 定义 control timeline：至少区分 source action、reference pose、planned/accepted qpos、command qpos、applied qpos、measured state、contact/object event、dispatch/worker receipt，并带 timestamp/clock/frame/validity/reason。
- [ ] 为 real replay、legacy sim export、native v2.2 export 和 CubeLift rollout 做 conformance fixture，证明 reader 能区分 v2.1 compatibility、v2.2 typed data 和 task-local 7D/31D privileged data。
- [ ] 在目标 `dexteleop310` 环境安装/确认 MuJoCo、Gymnasium、PyArrow 后重跑仿真测试和 help/contract matrix；把 optional skip 分类写入可复现验收脚本。

### P1：完成物理/执行可信度边界

- [ ] 为 RM65 actuator/ideal-position-debug profile 建 system-ID 计划：joint gains、damping/friction/armature、command latency、qref/qcmd tracking、measured qpos residual 和 timing jitter。
- [ ] 完成 120 Hz command segment collision certification，明确 robot/environment、self、tool proxy、table/camera pole 和 gripper/object 的合法/禁止 pair；真实硬件需要单独台架验收。
- [ ] 将 CubeLift success 拆成 `height_only`、`contact_latch_debug` 和未来 `physics_contact` 三套 verdict；补 reachability、pregrasp/close/lift trace、bilateral contact、relative pose、hold/release 和 failure event。
- [ ] 在 actuator/夹爪物理具备前，不启动“抓取 RL 已完成”或“policy sim2real 已通过”的宣传/验收；临时 latch 只允许出现在 task-local debug pipeline。
- [ ] 明确真实 arm/hand 非原子 dispatch 与 MuJoCo atomic qref 的差异，建立 hold/fail-stop/receipt/error recovery 的共同 trace schema。

### P1：验证性能优化没有改变语义

- [ ] 在 Python 3.10 target environment 重跑 reference/fast/vector benchmark，保存完整 environment/model/cache fingerprint，不只记录 stdout 数字。
- [ ] 对 reference 与 fast 做 differential test：same seed/action、obs/reward/terminated/truncated、cube state、latch state、safety projection 和 terminal observation 必须在明确 tolerance 内一致。
- [ ] 先用 profile measurement 判断 Python tick、`mj_step`、collision oracle、contact scan、observation/info、IPC 哪个是瓶颈；只有确认需要时才开发 native substep/shared `MjModel` runtime。
- [ ] 继续保持 reference facade 可读、可审计；不要通过删除 termination、collision 或 trace 来制造 throughput。
- [ ] 把历史 benchmark 中的 16-worker 8×/10×目标当作观察指标，不当作绝对硬件能力承诺；正式采样服务器另建基线。

### P2：数据与训练域收口

- [ ] 统一 legacy media writer 与 native v2.2 writer 的 provenance/episode/manifest envelope；明确哪些媒体只属于诊断，哪些可以进入训练。
- [ ] 建 golden real/sim paired episode 和多视角 comparison report；同时比较 timestamp、qpos/TCP、RGB/depth 几何、object/contact 和 freshness，不只拼视频。
- [ ] 规定 state-privileged task 与 camera-policy observation 的边界，避免把 31D cube pose/velocity 偷喂给视觉 policy。
- [ ] 保持 `sim/mujoco_temp_pipeline` 为临时包，明确 checkpoint/records/checkpoint mismatch 的生命周期；长期 PPO/BC/SAC 放入规划的 `model/`，policy artifact 通过 `deployments`/`PolicyArtifactContract` 输出。
- [ ] 建立 Python 3.10、无硬件、可选 MuJoCo、可选媒体库、target SDK 的 CI/本地验收矩阵。

## 11. 推荐推进顺序

1. 先修正 timing/profile/manifest 口径并完成目标环境测试，不改变控制语义。
2. 用 fake scene、synthetic action、fake receipt 和 dry-run trace 验证 cross-domain timeline。
3. 在当前 direct-qpos/task 语义下完成 CubeLift reachability、latch/debug/height 三类指标分离和 deterministic paired diagnostics。
4. 在仿真中完成 real-like 60 Hz qref→120 Hz qcmd→1000 Hz physics 的 segment safety、deadline、hold 和 failure injection。
5. 再做 actuator/system ID、gripper physics、camera/tactile/force fidelity 和真实台架验证。
6. 只有以上 gate 通过后，才推进 policy artifact 的 sim evaluation、real shadow/observe-only 和低风险真实部署。

## 12. 最终判断

当前 `sim/` 已经不是旧式的“URDF 转 MJCF + 回放脚本”：它有明确的 canonical backend、versioned contracts、物理 scene、碰撞投影、real-aligned position adapter、可导出的 v2.2 数据、可重复的 CubeLift reference/fast/vector runtime 和临时 PPO pipeline。

但这些能力的可信范围必须写清楚：

> **当前交付物是可审计的 MuJoCo prescribed-motion/debug/RL plumbing，不是经过 actuator system ID、真实接触力、真实控制时序和 paired episode gate 验证的 sim2real 基线。**

在 cross-domain manifest/timeline、目标环境 optional tests、actuator/gripper model、连续段安全、真实 measured feedback 和 paired sim/real verdict 完成前，建议继续保持所有真实部署路径为 dry-run、mock、shadow 或人工批准的分阶段验证。
