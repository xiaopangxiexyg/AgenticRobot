# Real2Sim / Sim2Real 与 Dataset 底座规划（临时）

> 更新：2026-07-15。本文按当前仓库代码、测试和 README 的实际能力整理。
> 范围覆盖数据采集/后处理/训练输入、统一动作控制、MuJoCo 物理仿真、标定、system ID 与 sim2real gate；暂不展开具体 VLA/RL 网络结构。

## 0. 一句话结论

项目已经不只是“真机数据 + 可视化回放”：`robot_motion/` 已具备共享的
`RobotAction`、15/45/150 Hz position pipeline、双臂 collision-aware pose→qpos
planner；`dataset_contracts/` 已具备独立的 dataset v2.2 policy-I/O contract；
`dataset_interface/` 已具备训练时 RGB/mask/背景与光照增强；MuJoCo 已具备
collision、重力、地面、桌面和 freejoint 物体。

真正未闭环的主缺口是：**同一条带 provenance 的 action/control/state timeline
尚未贯通真实采集、离线 dataset、MuJoCo control-like replay、task rollout 与模型
artifact。** 因此下一阶段的重点不是立刻扩大随机化或训练规模，而是先把
`dataset contract ↔ control contract ↔ sim contract` 固定为可验证的一条链。

## 0.1 本阶段的明确决策：先用 `direct_qpos_debug` 跑通任务

从当前基础开始，不等待真机 150 Hz system-ID 数据，也不等待完整 MuJoCo actuator
拟合。MuJoCo 增加一个**只用于 debug / task pipeline 推进**的控制模式：

- 双臂在 `150 Hz` 接收完整的 canonical joint-position command；夹爪在同一 command
  clock 接收归一化 `close ∈ [0, 1]`，再映射到其两个 finger joint qpos。
- physics 仍为 `1000 Hz`；在 command tick 之间对 joint target 做 zero-order hold。该模式
  必须明确命名为 `kinematic_direct_qpos_debug`，不能叫 PD、torque control 或 real-equivalent
  position control。
- 已有 `robot_motion` planner 可以**离线/上游**把 pose waypoint 编译为双臂 qpos waypoint；
  MuJoCo debug controller 的职责仅是按时间执行这些 qpos，不再复制 IK 或 collision planner。
- 方块、桌面和地面继续通过 MuJoCo contact/physics 演化；机械臂和夹爪的运动则是 prescribed
  kinematic motion。因此它足以验证 reset、观测、轨迹、夹爪开合、接触诊断、数据导出和任务状态机，
  **不足以**验证真实 tracking、接触力、抓持稳定性或 sim2real transfer。

本阶段先拆成两个里程碑：**P0-RL** 先只做“右臂使桌面静止的刚体 cube 抬升到阈值高度”，作为最小
Gymnasium-compatible 物理环境；其后才做 **P0.7–P0.9** 的抓取—放置—释放流程。两者均保持左臂在
安全 home qpos。P0-RL 不等待 actuator system-ID，但也绝不把 direct-qpos 结果解释成真实抓持或
sim2real 证据。

## 1. 当前系统地图与目录职责

```text
真机采集 / replay / offline postprocess
  └── dataset_contracts/             # 无硬件依赖的数据格式、codec、layout、stats
        ├── v2.2 PolicyIOLayout
        ├── PolicyArtifactContract
        ├── Parquet fixed-list / tactile / wrist-depth codec
        └── episode stats

训练输入
  └── dataset_interface/             # video + mask + augmentation + v2.2 row reader
        ├── DexTeleopDatasetIndex / modality validity
        ├── MaskedFramePipeline
        ├── PolicyIOFrame / PolicyIOBatch
        └── TorchMaskedAugmentDataset

sim/real 共享运动语义
  └── robot_motion/                  # RobotAction、action spec、45 Hz planner、150 Hz interpolation

仿真专属实现
  └── sim/                           # SimulationContract、MuJoCo scene/modeling/replay/task env（待建）

真机专属实现
  └── teleop_runtime/                # device I/O、采集、真实 dispatch、online worker
```

依赖方向应继续保持为：

```text
teleop_runtime / robot_motion / sim / offline postprocess
                         └──> dataset_contracts <──┘

dataset_interface  ──> dataset_contracts + read-only dataset artifacts
```

`dataset_contracts/` 不得反向导入 runtime、hardware、GUI 或 postprocess；
`dataset_interface/` 不得承担离线修 mask 或改写 dataset；`sim/` 不应复制
`robot_motion/` 的动作、限位、IK 或时序规则。

## 2. 已有实现：应保留并在后续工作中复用

### 2.1 共享动作、控制与部署基础

- [x] `robot_motion.action.RobotAction` 已作为 collection、replay、deploy、sim 的
  canonical action 语义：arm pose/qpos、end-effector target、active effector、reference
  frame 与 controlled frame 都能显式表达。
- [x] 默认 arm pose 语义已经固定为 `tool_tcp` in `left_arm_base/right_arm_base`；不能把
  MuJoCo 的 `link6` pose 静默当作 TCP。
- [x] position 主线已有 `15 Hz policy/sample → 45 Hz reference/coupled collision IK →
  150 Hz qcmd interpolation` 的共享实现；双臂 planner 包含绝对关节限位、reference step
  限制、endpoint collision recheck 与 hold-last-safe。
- [x] policy artifact 可以绑定 `PolicyIOLayout` hash、输入/输出 dtype/dim 和
  `state_input_mode`；deploy 已有 layout/artifact compatibility gate。
- [x] 真实 position deploy 已记录较完整的 reference/command/dispatch audit，但这些字段
  尚未整理成所有 dataset、sim replay 都可消费的统一 episode timeline。

### 2.2 Dataset contracts：格式层已有能力

- [x] `dataset_contracts/` 已从 runtime 抽离，持有 episode/chunk/video channel 命名、
  Parquet 固定列、episode stats、wrist depth/mask codec、tactile chunk stream codec。
- [x] DexTeleop dataset v2.2 已定义 `PolicyIOLayout`：连续与离散 end-effector channel
  分开、每个 component 有名称/dim/单位/normalization/source index，且 layout 有 stable hash。
- [x] v2.1 固定 26D action / 38D state 可通过显式 layout 转成 v2.2；转换保留 legacy 列、
  不猜 q6 padding 语义、copy-on-write 写入新 dataset，并记录 source hash 与 conversion id。
- [x] `PolicyArtifactContract` 可将 checkpoint 的 tensor contract 与 dataset layout 绑定，
  避免训练时把 discrete token 当 float 或部署时错接 end-effector。
- [x] 采集 writer 已可选 `policy_io_layout`，在仍保持 LeRobot v2.1-style storage 的前提下
  写出 v2.2 canonical model columns、layout hash、episode binding、arm/hand spec metadata。
- [x] 采集端支持 parquet/video、external 或 parquet tactile、wrist raw depth/depth mask、
  calibration snapshot、device config、episode stats、storage queue/drop 统计。
- [x] 后处理已有 NaN fault tail、stutter、episode delete/split、force collision label、
  causal camera/tactile align、RealSense undistort、table/depth/final wrist keep-mask、
  v2.1→v2.2 conversion 等 copy-on-write 工具。

### 2.3 Dataset interface：训练输入层已有能力

- [x] `DexTeleopDatasetIndex` 读取 `meta/info.json`、`episodes.jsonl`、显式 mask path，
  并校验 dataset `PolicyIOLayout` binding。
- [x] `meta/modality_validity.json` 可以在 dataset / episode / video-view 三层 fail closed；
  无该文件时兼容历史数据集。
- [x] `MaskedFramePipeline` 支持 video reader cache、bitpack mask cache、frame/clip 读取、
  BGR/RGB/CHW-float 输出；wrist 使用离线 final keep-mask，front 可使用在线固定边界 mask。
- [x] 背景替换、真实背景、全图 photometric/quality augmentation 均按
  `episode + camera + epoch (+ frame)` 确定性采样并记录 audit metadata；不使用几何增强，
  因而不会破坏相机/动作几何关系。
- [x] `PolicyIOFrame/Batch` 已可从 v2.2 row 读取 typed float/token 低维输入，
  `TorchMaskedAugmentDataset` 可向基础 sample 注入增强图像。

### 2.4 MuJoCo / real2sim 已有能力

- [x] `SimulationContract` 已版本化 `ActionSpec / ObservationSpec / TimingSpec / SceneSpec`；
- [x] `sim/mujoco/dataset_v22/` 已作为独立的原生 v2.2 写盘入口接入 replay runner；它直接
  使用真机双 DM-Tac `PolicyIOLayout`，记录 layout hash、typed policy I/O、RGB 与 sim provenance，
  并在 modality validity / episode provenance 中显式声明 force/torque、tactile unavailable。
  profile 固定 15/45/150 Hz control clocks、1000 Hz physics、scene、camera 和接触参数。
- [x] URDF→MJCF 已拆出 visual/collision geom、collision group/mask、allow/deny、gravity、
  ground、URDF table collision 与 freejoint debug cube。
- [x] 独立 `ground + table + freejoint cube` GUI/drop smoke 已验证方块落在桌面；这是当前
  有效的物理接触 smoke，不代表完整被动机械臂 physics rollout。
- [x] MuJoCo Cartesian pose path 已接到 `robot_motion.rm65` coupled planner；planner 成功后
  才原子写入双臂 qpos。per-arm `MuJoCoArmBackend` 仅是 joint-vector sink，拒绝独立 Cartesian IK。
- [x] MuJoCo 与 canonical RM65 link6 FK 已有双臂一致性回归；TCP profile 必须显式提供。
- [x] measured-qpos replay、相机渲染、RGB distortion、wrist depth/mask、point cloud 与
  LeRobot-style sim replay export 已存在，适合几何/媒体诊断。

## 3. 当前不能声称“已完成”的边界

### 3.1 Dataset 与控制语义尚未完全统一

- [ ] v2.2 仍是 opt-in adapter：采集工作向量和大量历史 dataset 仍以 v2.1 26D/38D 为主。
  在 native v2.2 collector 落地前，必须保持 explicit layout conversion，禁止根据 q6 内容猜设备。
- [ ] `sim/mujoco/replay/dataset_writer.py` 目前直接写 LeRobot-style Parquet 与媒体 metadata，
  未复用 `dataset_contracts.v22.PolicyIOLayout` / `write_rows_to_parquet` / policy artifact binding；
  因此 sim export 与 real collection 还不能天然作为同一训练数据契约。
- [ ] `MaskedFramePipeline` 和 `PolicyIOFrame` 是两条成熟但独立的读取路径；尚无一个标准的
  trajectory dataset 将 row timestamp、camera/video index、tactile index、depth/mask、低维
  action/state 与 validity policy 原子地组成训练 sample/clip。
- [ ] dataset 还没有一个强制的 cross-domain manifest，将 `SimulationContract` hash、
  `PolicyIOLayout` hash、policy artifact、calibration snapshot、camera preprocessing、tool TCP、
  source dataset 和 postprocess history 绑到同一 episode/rollout。

### 3.2 MuJoCo 仍不是完整动作物理环境

- [ ] RM65 joint actuator、150 Hz target buffer、joint damping/friction/armature、控制延迟与
  `mj_step` full robot rollout 尚未建立；现在对 full robot 连续 `mj_step` 会让无 actuator 的关节坍塌。
- [ ] 完整碰撞模型已有静态/contact foundation，但未认证机械臂/夹爪在 actuator 动态过程中的
  self/table/object collision，也未认证 45 Hz endpoint 之间的 150 Hz 插值段。
- [ ] parallel gripper 当前是几何/关节开合，尚无可靠 mimic/coupling、finger pad contact、
  grasp state 或 object attachment/force 验证。
- [ ] current physical profile 的 tool/TCP、base、camera、table 标定没有成为不可缺少的
  versioned snapshot；不同真机工具 profile 不能被 sim 默认值替代。

### 3.3 对齐、训练和评测尚未闭环

- [ ] 没有 golden paired episode，能从同一 action/timestamp trace 产出 real 与 sim 的
  qpos/TCP/image/object 对比和阈值 verdict。
- [ ] 没有标准 task `reset/step`，也没有可复现的 pick-and-lift、success/failure/reward、
  object initial-state distribution。
- [ ] 没有基于真实 residual/system-ID 的 physical/camera/latency randomization ranges。
- [ ] 训练数据 split、quality gate、mask coverage、camera/tactile freshness、control fault、
  action distribution 和 sim/real domain tag 尚未被统一为 dataset card / benchmark manifest。

## 4. P0：先完成一条可审计的数据—控制—仿真链路

### P0.1 冻结 cross-domain run manifest（最高优先级）

目标：任意 real episode、derived dataset、sim replay、task rollout、checkpoint 都能回答
“用了什么动作 layout、什么 TCP/标定、什么 scene、什么媒体预处理、来自哪条 source trace”。

- [ ] 在 `dataset_contracts/` 新增纯数据的 `CrossDomainManifest`（名称可调整），不导入 runtime。
- [ ] 最小字段：
  - dataset schema / `PolicyIOLayout` hash / state-input mode；
  - `SimulationContract` schema/name/content hash（real dataset 可为空，但必须显式说明）；
  - arm action spec、hand/end-effector spec、active sides、joint order、unit、reference frame、controlled frame；
  - left/right tool profile 或 endpoint→TCP transform、base/table/head/wrist calibration snapshot id/hash；
  - camera key/order、intrinsics、undistort/crop/resize/color-space、depth convention、mask manifest；
  - sample/reference/command/physics/camera frequencies 和 timestamp policy；
  - source dataset id/episode/parquet hash、postprocess history、code revision、seed；
  - `domain = real | sim_replay | sim_rollout | synthetic` 与任何已知 fidelity limitation。
- [ ] real collection writer、v2.1→v2.2 converter、MuJoCo export、future task env 都写入同一 manifest。
- [ ] reader/deploy/sim 在 layout、TCP、camera preprocessing 或 action frame 不兼容时 fail fast。

验收：给定一个 parquet row 或 checkpoint，可以找到唯一 manifest；hash 不匹配时训练/部署/回放均拒绝继续。

### P0.2 定义统一 control timeline，而非只保存一个 `action`

目标：从 dataset 复现实验时，明确区分“策略意图、planner 输出、实际执行、测得状态”。

- [ ] 在 `dataset_contracts` 定义时间线 channel/row contract；保留 v2.1 audit 列，但增加稳定的
  canonical names 和 validity flags。
- [ ] 至少记录：
  - `source_action`：policy/teleop 原始 `RobotAction` 与 source timestamp；
  - `reference_pose`：45 Hz 插值后的 pose reference；
  - `planned_qpos`：coupled collision planner 的 raw/limited/hold result、reason、distance；
  - `command_qpos`：150 Hz interpolation qcmd、sequence、age/stale/hold；
  - `dispatch_receipt`：accepted/rejected、transport timestamp、backend modification；
  - `measured_state`：qpos、TCP、hand/gripper、camera/tactile depth/frame timestamp 与 freshness；
  - `object_state`：sim privileged state 或 real perception/marker state，必须标注来源和置信度。
- [ ] 使 causal align 的 `latest <= sample timestamp` 规则同时覆盖 video、tactile、depth 和将来的
  sim camera；禁止只靠 frame index 假设同步。
- [ ] 为每条 channel 写 source/target frame、unit、clock domain、validity/reason；故障或缺失不能被
  静默填成正常零值。

验收：同一条 trace 可驱动 deploy dry-run、control-like sim replay 和离线审计，且能逐 tick 对齐。

### P0.3 让 dataset v2.2 成为真正可消费的训练/仿真格式

- [ ] 保持 v2.1 storage compatibility，但把新采集的 `PolicyIOLayout` 设为显式默认配置；所有
  新训练 checkpoint 都必须携带 `PolicyArtifactContract`。
- [ ] 实现 native v2.2 sample assembler：直接从 typed arm/end-effector state/action 构造 row，
  不再先假装所有手都是 q6 后再派生；旧 collector 保持 adapter path。
- [ ] 给 `sim/mujoco/replay/dataset_writer.py` 接入 `dataset_contracts`：使用 `PolicyIOLayout`、
  fixed-size Parquet columns、episode layout binding、stats 和 cross-domain manifest。
- [ ] 为 v2.2→v2.2 layout migration 定义显式工具/策略（identity、增加 audit channel、不可逆
  end-effector remap 必须拒绝），不要复用 v2.1 adapter 偷换 schema。
- [ ] 建立 schema conformance fixture：同一 synthetic episode 分别经 real writer、v2.1→v2.2
  migration、sim writer 后，检查 metadata/row/media references 的契约一致性。

验收：real 和 sim 产出的 dataset 可由同一个 layout-aware reader 加载；混入不匹配 layout 的
checkpoint 会在训练启动和 deploy 启动前失败。

### P0.4 建立标准多模态 `TrajectoryDataset`

目标：把已有 `MaskedFramePipeline` 和 `PolicyIOFrame` 合成一个明确的训练 sample contract。

- [ ] 新模块建议放在 `dataset_interface/trajectory.py`，只读 dataset，不写 mask/Parquet。
- [ ] 输入：episode/index 或 timestamp window；输出：
  - typed policy action/state（continuous 与 discrete 分开）；
  - requested multi-camera frames/clip、mask 与 augmentation audit；
  - tactile/depth/pointcloud 的显式引用或 decoded tensor；
  - per-modality timestamp、age、validity、manifest/layout hash、domain tag。
- [ ] 定义 clip 对齐规则：action horizon、observation context、camera frame selection、tactile/depth
  selection均由 timestamp 表达；不允许模型 loader 各自 seek 最近 frame。
- [ ] 实现 strict / filtered 两种模式：strict 对缺失关键模态 fail；filtered 使用
  `modality_validity` 和 explicit sampler 跳过，不以重复补位掩盖数据分布变化。
- [ ] 图像 normalization/tensor transform 可由模型 adapter 注入，但不改变 dataset contract；训练
  augmentation config/hash 应进入 run manifest。

验收：一个 batch 能完整说明每个 tensor 来自哪个 row/frame/manifest；重复同 seed/epoch 结果一致。

## 5. P0：落地 `kinematic_direct_qpos_debug`，不让 actuator 缺失阻塞任务

### P0.5 控制模式、文件边界与接口冻结

现有 `sim/mujoco/controllers.py` 的 `QposController` 与 `GripperController` 是单次写入 helper；
不应继续向其中堆 scheduler、task 状态或 viewer 逻辑。新增结构建议为：

```text
sim/mujoco/control/
  __init__.py
  direct_qpos.py          # 原子 group qpos + gripper target，执行模式与边界检查
  schedule.py             # 150 Hz command / 1000 Hz physics 的 timestamp scheduler、ZOH
  trace.py                # command/applied/physics/contact audit row（sim 专属实现）

sim/mujoco/tasks/
  __init__.py
  base.py                 # reset/step/info 的最小 task interface
  cube_lift/              # P0-RL：state-privileged、pure-contact height task
    config.py
    spec.py
    env.py
    state.py
    contacts.py
    reward.py
    scheduler.py
  cube_pick_place/
    config.py             # task/object/zone/success/failure 参数与验证
    scene.py              # task-only object、target-zone、debug attachment 构造
    env.py                # reset、step、observation、reward、termination
    events.py             # grasp/lift/release/contact/timeout 判定
    scripted_policy.py    # 固定 waypoint demo；不属于训练 policy

sim/mujoco/diagnostics/
  cube_lift.py            # P0-RL GUI/headless CLI，只调用 CubeLiftEnv
  cube_pick_place.py      # GUI/headless smoke CLI，只调用 task/control 公共接口

sim/configs/
  tasks/rm65_right_cube_lift_rl_debug_v0.json
  rm65_bimanual_cube_pick_place_debug_v1.json  # 独立 task profile，不改写 cube-drop 基线
```

目录职责：`robot_motion/` 继续负责 qpos waypoint 的 canonical planning、joint layout、限位和
45→150 Hz interpolation；`sim/mujoco/control/` 只消费完整 joint command；`tasks/` 只定义 task
world、状态机和成功条件；`diagnostics/` 不包含任何 task business logic。

- [x] 已实现 `DirectQposCommand`：冻结 atomic `sequence`、完整 side→6-DoF qpos、按名称的 normalized
  gripper close 和 `source`；controller 要求 command side 与 `active_sides` 完全一致，拒绝未知 gripper、
  非有限/越界值、提前命令和非递增 sequence。它尚未带外部 `timestamp_ns`、validity/reason；这些属于
  后续 timeline/trace 层，不能假称已具备。
- [x] 已实现 `DirectQposDebugController` 及 metadata：
  `execution_mode = "kinematic_direct_qpos_debug"`、`robot_dynamics = "not_simulated"`、command/physics
  frequency、ZOH 与 `zero_controlled_qvel` 策略。`grasp_mode` 仍应由 task rollout metadata 显式补充，
  因为 controller 本身不应拥有 task grasp 语义。
- [x] controller 只消费上游完整 qpos；不做 pose IK、45→150 Hz interpolation 或 target tracking。将来
  canonical waypoint replay 必须复用 `robot_motion` 的 planner/interpolator，不能在 sim 复制一套算法。
- [x] 每一个 1000 Hz physics tick 都重施 held arm/finger qpos，并在 `mj_forward → mj_step → mj_forward`
  的固定顺序下推进；150 Hz command deadline 用浮点时间而不是假定 `1000/150` 为整数。受控 dof 可清零
  qvel，避免无 actuator 的 arm 在重力下坍塌。
- [x] 夹爪通过已有 `GripperSpec.qpos_from_close()` 映射，side/finger 缺失或 qpos 非有限时 fail fast；
  controller 留下每个 physics tick 的 command due/applied/sequence audit。
- [ ] 补充可持久化的 command/applied/physics/contact trace，包含外部 timestamp、source validity、
  task config/scene hash 与 `grasp_mode`；viewer/render loop 不得临时写 qpos。

验收：给定固定 seed 与同一 qpos command trace，headless 和 GUI 两种运行的 150 Hz command
sequence、1000 Hz tick count、arm/finger qpos、cube 初末状态与 event log 一致；所有输出均带 debug-mode
provenance，不能被 export 标为 actuator rollout。

### P0-RL：先落一个仅用于仿真 RL 的 Cube-Lift 环境（新增，优先于 pick-and-place）

**边界与结论。** 第一版 task 固定为 `rm65_right_cube_lift_rl_debug_v0`：右臂使桌面静止的
`debug_cube` 上升到阈值高度即成功；不要求放置、不要求视觉输入、不需要真机 150 Hz 数据，也不声称
sim2real。它复用现有 `kinematic_direct_qpos_debug`，因此含义是“自由 cube 的 MuJoCo 接触物理 +
处方式双臂 qpos”。机器人 actuator、真实接触力与抓持稳定性均不在此 milestone 的可信范围内。

| 审视项 | 当前已有 | RL v0 缺口与落地决定 |
| --- | --- | --- |
| 场景 | gravity、ground/table collision、`debug_cube` freejoint、v3 robot/gripper visual/collision 均已可用；base profile 现为桌面中心 `(-0.412, 0)` 的红色 3 cm cube，初始 `z=0.026 m` 留 1 mm settle clearance | 仍未证明该 center pose 可由右臂可靠 pure-contact 抓取；task 暂以它作为 fixed reset，后续 reachability gate 才能开放随机 reset。 |
| 控制/时钟 | complete dual-arm 150 Hz direct qpos、1000 Hz ZOH physics、sequence 校验，及 15 Hz policy timestamp scheduler 均已可用 | scheduler 已验证交替 66/67 physics ticks；还缺可持久化 rollout trace 和 pure-contact grasp success trace。 |
| 机器人状态 | 可读写 arm/gripper qpos，左/右 gripper spec 已推断 | 若只给右臂发命令，左臂会下坠；env 每个 command tick 必须提交完整双臂 qpos，左臂恒定 home，agent 只控制右臂。 |
| 物体/接触 | `MujocoScene` 已有命名 freejoint qpos/qvel reset、finite-state 检查和 named geom/body contact facts | 尚缺 bilateral-finger 等 grasp task 语义，以及持久化 contact trace；它们不进入 scene facade。 |
| contract | `SimulationContract.ActionSpec/ObservationSpec` 是 real/deploy 的 26D action / 38D state；RL 已有 task-local 7D action / 31D `state_privileged` observation | 不能篡改或假借为 real contract。像素 observation、dataset writer 与 policy artifact binding 后置。 |
| RL 依赖 | `gymnasium 1.3.0` 已安装，`CubeLiftEnv` 已通过 Gymnasium env checker | PyTorch/SB3 未安装且不属于当前 environment runtime dependency；待选择 GPU/CUDA 后再引入。 |

#### P0-RL.0 训练前硬门槛：先证明“可抓到”，不让 RL 替代排障

- [ ] 新增 headless `cube_lift_reachability` diagnostic：从 bench home 固定双臂和开夹爪，使用
  `robot_motion/rm65` canonical solver 或人工 qpos trace，验证 right `pregrasp → descend → close → lift`
  的 qpos。它输出 joint-limit/planner result、grasp-center/cube relative pose、所有接触 pair、cube
  高度/速度和每一段失败原因。
- [ ] 由该 diagnostic 固化 v0 的 right home、pregrasp、grasp、lift qpos 安全盒与 cube XY/yaw
  reset 小范围；记录 solver/tool/scene profile/hash。不得沿用当前 `debug_cube` drop 位或凭视觉猜测
  workspace。
- [ ] 以一条固定 pure-contact qpos trace 尝试抬升。若 cube 未能被纯接触可靠抬起，仍可继续完成
  reset/clock/Gym API 的工程 smoke，但将训练 gate 标为 **blocked: no physical lift trace**；不得暗中
  改用 weld/attachment 或把“夹爪 close”计为 lift success。若以后增加 attachment，只能是显式
  `grasp_mode=debug_attachment` 的独立 plumbing test，不能与本环境的成绩混用。
- [ ] 验证 bench home + 已静止 cube 时没有 unexpected self/table/robot contact；方块初始 center z
  由 `table_top_z + cube_half_height + clearance` 推导并在 settle 后实际核验，不能硬编码当前约
  0.03 m 的 resting center z。

#### P0-RL.1 文件边界、配置与 task-local spec

以下是**计划新建**的边界；scene 通用能力留在 `sim/mujoco/`，不得把 reward、Gym space 或 task
常量塞入 `scene.py` / `control/direct_qpos.py`：

```text
sim/mujoco/tasks/
  __init__.py
  base.py                       # typed reset/step protocol、common episode metadata
  cube_lift/
    __init__.py
    config.py                   # CubeLiftTaskConfig、JSON load、strict validation
    spec.py                     # task-local action/observation layout + Gym spaces
    env.py                      # CubeLiftEnv reset/step only
    state.py                    # task state snapshot / freejoint bridge use
    contacts.py                 # named contact classification
    reward.py                   # pure reward/success/termination functions
    scheduler.py                # policy horizon → existing 150/1000 Hz controller
    diagnostics.py              # non-GUI summary helpers
sim/mujoco/diagnostics/
  cube_lift.py                  # GUI/headless CLI；仅构建并驱动 CubeLiftEnv
  cube_lift_reachability.py     # P0-RL.0 gate
sim/configs/tasks/
  rm65_right_cube_lift_rl_debug_v0.json
tests/
  test_mujoco_cube_lift_config.py
  test_mujoco_cube_lift_env.py
  test_mujoco_cube_lift_contacts.py
```

- [x] task JSON `sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json` 引用（不复制）base
  `sim/configs/rm65_bimanual_physics_v1.json`，并冻结：task/schema
  version、base contract hash、object body/freejoint、right grasp frame 与 right gripper name、固定 left
  home、right safe qpos bound / action delta、policy/command/physics clock、episode time limit、cube reset
  pose、table height、success/reward/failure 参数、`grasp_mode=physics_contact`、render metadata。当前 safe
  qpos bound 只是 canonical absolute envelope，尚未替代 P0-RL.0 的 planner/contact verification。
- [x] `CubeLiftActionSpec` 固定为 `float32 Box(-1, 1, shape=(7,))`：前 6 维为 right joint target 的
  normalized increment，最后 1 维为 right gripper close 的 normalized increment。env 将其裁剪/映射到
  task config qpos bound 与 `[0,1]` close，生成完整 dual-arm command；
  action 不接受 TCP pose、left-arm command 或 direct MuJoCo qpos address。
- [x] `CubeLiftObservationSpec` v0 固定为 `state_privileged` 的 31-D `float32` vector，按命名 layout
  依次为 right qpos(6)、right close(1)、cube position(3)、cube quaternion(4)、cube linear/angular
  velocity(6)、cube→right-grasp-center vector(3)、last action(7)、episode progress(1)。字段顺序、dim
  与 layout hash 必须写入 config/info；它不是现有 deploy 38D state，也不得伪装成 camera policy
  observation。像素 wrapper 放到 state 环境稳定以后。

#### P0-RL.2 reset、时钟、物理和判定的最小语义

- [x] `MujocoScene` 已提供通用 helper：通过命名 freejoint 设置/read `qpos[7]` 和 `qvel[6]`、验证
  finite quaternion/velocity、枚举本 tick contact 的 geom/body 名称与 penetration distance。helper 只
  暴露物理事实；`table_contact`、`bilateral_finger_contact` 等 task 语义放在 `cube_lift/contacts.py`。
- [x] `reset(seed)` 使用独立 `np.random.Generator`：`mj_resetData` 后写入完整 dual-arm home/open
  command、freejoint cube pose/zero velocity，清除 controller/timeline/episode state，`mj_forward`，持臂
  settle N 个 physics ticks，再返回 observation。相同 seed 和 action trace 必须相同；reset 中出现 NaN、
  unexpected contact 或 cube 未静止必须 fail fast/给出原因，不能继承上一 episode 状态。
- [x] 初始 policy rate 复用 `TimingSpec.policy_hz=15`，每次 `step(action)` 保持该 action 到下一
  **policy timestamp**。内部在每个 150 Hz deadline 提交递增 sequence 的完整 dual-arm command，其余
  1000 Hz tick 只 ZOH；以累计时间/next-deadline 决定步数，绝不依赖固定整数 tick 数。每个 episode
  info/trace 要记录 policy、command、physics 三层时间与次数。
- [x] v0 成功定义为 `height_only_lift_success`：cube center z 高于 reset 后 resting z 加配置的
  `lift_delta_m`（建议先 0.08 m），持续 N 个连续 tick，且不再与 table 接触。它刻意不要求“已抓住”，
  因而可能把抛起也算成功；这是高度任务的已知 loophole，后续 `grasp_and_lift` 再加入 bilateral contact、
  relative-pose 和 hold 条件。
- [x] reward 保持可解释：归一化 lift progress + 一次性 success bonus − 小的 action-delta penalty；
  可选的 out-of-bounds/invalid-contact penalty 也全部配置化。`terminated` 仅用于 success、NaN、cube
  越界或配置化 fatal contact，`truncated` 仅用于 time limit；`info` 带终止原因、cube z、table contact、
  contact summary、controller sequence/tick 及 mode/config hash。

#### P0-RL.3 实施顺序、验收和明确后置项

1. [ ] 完成 P0-RL.0 reachability/contact gate，产出固定 qpos trace 和 v0 实际 reset 区域；这是
   `CubeLiftEnv` 的输入，不是训练后再猜的参数。
2. [x] 已实现并单测 scene freejoint/contact helper；cube-drop、direct-qpos 与 task regression 均通过。
3. [x] 已在 `README_ENV.md` 记录最小 `gymnasium` 安装，完成 task config/spec，并以
   `gymnasium.utils.env_checker.check_env` 验证 API。
4. [x] 已实现 scheduler、reset、state observation、reward/success 和 headless `CubeLiftEnv`；已覆盖
   deterministic action trace 与 100 个 random-action headless reset/step。
5. [x] 已新增同一 env 的 headless / `--viewer` CLI `python -m sim.mujoco.diagnostics.cube_lift`；viewer
   可在每个 policy tick 采样 bounded random action，但只显示/驱动 env 已产生的 state/timeline，不使用独立控制循环。

- [x] v0 最小 test matrix 已覆盖：config/action bound、same-seed reset/action-trace reproducibility、left
  hold/right action mapping、freejoint qvel 清零/settle、15/150/1000 Hz trace、synthetic success/failure、finite
  observation、100 个 headless random episode、Gym API checker 与 GUI-free headless regression。
- [ ] 训练开始 gate：`reset/step` 返回标准 Gymnasium tuple 且 env checker 通过；一条 manual/planner
  **pure-contact** trace 能稳定触发 height success；随机策略不崩溃；GUI 与 headless 对同 seed trace 给出
  相同任务状态。若 pure-contact trace 不成立，交付停留在 environment/plumbing smoke，不启动“抓取 RL”。
- [ ] 明确后置：RGB/depth policy observation、dataset writer/`CrossDomainManifest` 接入、PPO/SAC 等训练
  launcher、vectorized environment、多物体/curriculum、domain randomization、actuator dynamics 与
  sim2real calibration。它们不应阻塞本 task 的可重复物理环境，但也不能被 v0 的结果掩盖。

### P0.6 先验证 direct-qpos 运动学接触边界

direct qpos robot 与自由刚体 cube 的组合可能出现瞬移、穿透修正、非真实的接触冲量或“手指看似闭合但
方块未被可靠约束”。这不是阻塞项，但必须在开始抓取前被量化并显式分流。

- [ ] 新增 deterministic diagnostics：home/approach/lower/close/lift 的单臂 qpos 轨迹，输出每个
  physics tick 的 object pose/velocity、contact pair、penetration/contact count、gripper opening 和 cube
  relative-to-TCP pose。
- [ ] 将接触分为 `no_contact`、`one_finger_contact`、`bilateral_finger_contact`、`table_contact`、
  `arm_or_unexpected_contact`；不以“close target=1”认定抓住。
- [ ] 定义两条可切换路径：
  - `physics_contact`：不加任何约束；只有 cube 在 lift/hold 期间满足高度、相对 TCP 位移和非桌面
    接触条件时才判 grasp。此模式用于暴露模型问题，允许暂时失败。
  - `debug_attachment`：仅在连续若干 physics tick 的 bilateral finger contact、夹爪 close 阈值和
    cube 相对 TCP 距离同时满足时，创建可记录的临时 attachment/weld；release 时必须移除。它只用于
    把 reset→轨迹→观测→数据→放置的工程链路跑通，结果必须标记为 assisted grasp。
- [ ] attachment 的创建/拒绝/释放都写 event log（原因、tick、relative transform、contact evidence）；
  禁止无条件把 cube 绑到末端，也禁止将 assisted success 计入 physical-contact 成功率。

验收：在 `physics_contact` 与 `debug_attachment` 两种模式下都能得到可解释 event trace；后者在
contact precondition 不成立时必须失败而非自动吸附。

## 6. P0：Cube pick-and-place v1 的可执行流程

### P0.7 任务场景、对象和 reset

- [ ] 以现有 `rm65_bimanual_physics_v1.json` 为基线，新建而非修改 cube-drop profile：保留地面、
  桌面、cube contact 参数，并添加 `target_zone`（视觉 marker 可选、物理边界/高度区间必需）。
- [ ] 固定第一版物体为现有 `debug_cube`：尺寸、质量、惯量、friction、freejoint、初始 pose
  distribution 和 asset hash 写入 task config；先不混入多尺寸/多 mesh/randomization。
- [ ] `reset(seed)` 必须重置 simulation data、左右 arm home qpos、两个 gripper open、cube freejoint
  pose/velocity、attachment、event log、scheduler clocks 和 camera schedule；reset 后执行 `mj_forward` 并
  等待 cube 在桌面静止，不能沿用上回合接触状态。
- [ ] 冻结初始位为 right arm 可达且远离 left arm/table 的一个小范围；将 qpos home/pregrasp/grasp/
  lift/place/retreat waypoint 的来源、tool profile 和 planner result 一起版本化。
- [ ] task observation 至少给出相机图像、right/left qpos、gripper close、clock 与 fault state；cube
  world pose/contact 是 privileged observation，只能用于 scripted demo、评测和 debug，不得静默喂给
  视觉 policy。

### P0.8 固定脚本的抓取—放置状态机

第一条轨迹不依赖学习 policy。`scripted_policy.py` 按时间生成完整 dual-arm qpos + gripper target
stream，右臂逐段执行，左臂保持 home：

```text
RESET / SETTLE
  → HOME_OPEN
  → PREGRASP_ABOVE_CUBE       # 张开夹爪，安全高度
  → DESCEND_TO_GRASP
  → CLOSE_AND_VERIFY          # 等 bilateral contact / attachment precondition
  → LIFT_CLEAR_OF_TABLE       # cube 高度与无 table contact gate
  → TRANSIT_ABOVE_TARGET
  → DESCEND_TO_PLACE
  → OPEN_AND_RELEASE
  → RETREAT
  → SUCCESS / FAILURE
```

- [ ] 每段 waypoint 由 canonical RM65 solver 预检并存成 qpos trace；若 planner hold/failure、joint
  limit、left/right collision gate 或 waypoint 不可达，则不运行该段，而不是由 direct controller 绕过。
- [ ] 每段都有独立 timeout、qpos arrival tolerance、cube/table/contact precondition 与错误码；状态转移
  由观测/event 驱动，禁止只按固定 wall-clock 宣称成功。
- [ ] `LIFT_CLEAR_OF_TABLE` 要求 cube 高于 task height threshold、持续 N 个 physics tick、与 table
  无接触；`OPEN_AND_RELEASE` 后要求 cube 在 target zone 内、低速、重新与 table 接触且不再 attached。
- [ ] failure 至少覆盖：unexpected robot/table/self contact、cube 掉落/越界、attachment precondition
  不足、release 后不在 target、tick/command stale、timeout、NaN/invalid qpos。每一类写出终止原因和
  最后一个 timeline event。
- [ ] success 结果分别统计 `assisted_pick_place_success` 和 `physics_contact_pick_place_success`；在
  actuator/夹爪物理通过前，前者只代表工程流程成功。

验收：固定 seed、固定 waypoint trace、`debug_attachment` 模式连续运行 N 次均完成
`pick → lift → place → release → settle`；`physics_contact` 的结果单独报告，不因失败阻塞该工程里程碑。

### P0.9 GUI、headless replay 与最小数据产物

- [ ] 新增计划入口（实现后）：
  `python -m sim.mujoco.diagnostics.cube_pick_place --contract sim/configs/rm65_bimanual_cube_pick_place_debug_v1.json --grasp-mode debug_attachment --viewer`。
  GUI 显示当前 phase、150 Hz command sequence、gripper close、contact/event、cube/target state；GUI
  只观察同一 scheduler，不单独驱动控制。
- [ ] 同一 CLI 支持 `--headless --seed ... --trace-dir ...`；headless 是 CI 回归基线，viewer 仅用于
  排障和直观确认。
- [ ] 每次 rollout 输出 `manifest.json`、control timeline、event log、task summary、seed、task config
  hash、scene contract hash、grasp mode、cube initial/final pose 和可选 RGB/depth。先定义 sim 内部 trace，
  随后接入第 4 节的 `CrossDomainManifest` / `TrajectoryDataset`，不另造长期 schema。
- [ ] 加入 deterministic test：cube settle、reset reproducibility、scheduler 150/1000 clocks、qpos/
  gripper ZOH、attachment precondition、release removal、success/failure、GUI-free headless run。

## 7. P1：建立可量化的 Real2Sim 闭环

### P1.1 Golden paired episode 与对齐报告

- [ ] 采集一组短、低风险、动作覆盖明确的 golden episode（不要求任务成功）：双臂 qpos、TCP、
  gripper、front/wrist RGB、depth、timestamp、calibration、tool profile、control audit 完整。
- [ ] 从同一 `source_action` / control timeline 产生 sim control-like rollout，而不是从最终 qpos 直接摆姿态。
- [ ] 输出并版本化报告：
  - qpos / velocity / acceleration / delay / hold 一致性；
  - arm endpoint 与 tool TCP 的 SE(3) 误差；
  - table、robot keypoint/TCP 的 image reprojection error；
  - RGB crop/resize/undistort/color-space 检查，depth valid/scale/alignment error；
  - object pose、接触时刻、grasp/lift 事件（可测时）。
- [ ] 为每项设阈值与 fail-fast test；报告必须引用 manifest hash、dataset episode、scene profile、code revision。

### P1.2 标定与媒体 provenance

- [ ] 统一管理 left/right base、table、head/wrist cameras、RGB-depth、tool TCP 的 calibration profile；
  `utils/hand_eye` 和 `utils/head_eye` 输出应可自动写入 manifest/scene，不靠手抄矩阵。
- [ ] 修复并回归 `utils/head_eye` namespace/README 入口；确认其输出坐标系、单位、时间戳来源。
- [ ] 每个 collection、derived dataset、sim rollout 固化 calibration snapshot；后处理不得在不新增
  provenance record 的情况下替换视频、intrinsics、mask 或 depth convention。
- [ ] 建立 dataset card：source/derived lineage、episode split、quality exclusion、modality validity、
  camera preprocessing、mask coverage、background augmentation、license/PII/retention（如适用）。

### P1.3 System ID

- [ ] 无接触优先：用 150 Hz trace 拟合 target delay、qpos tracking、rate/acceleration bounds、damping、
  friction、actuator gain；train/validation trajectory 分离。
- [ ] 接触其次：table/object friction、contact solref/solimp、gripper dead-zone/pad、object mass/inertia、
  grasp/release 行为。
- [ ] 输出 versioned physics profile：参数、范围、fit metric、confidence、适用 tool/scene/robot version。
- [ ] P1.1 residual 与 system-ID confidence 成为后续 randomization 范围的唯一主要来源。

## 8. P1：训练、sim2real 与 domain randomization

### P1.4 训练/部署契约 gate

- [ ] 训练入口必须验证 dataset `PolicyIOLayout` hash、policy artifact、state-input mode、camera keys/order、
  image preprocessing、action controlled/reference frame 与 end-effector capability。
- [ ] deploy 与 sim rollout 使用同一 artifact contract；禁止模型 adapter 私自把 discrete hand token、
  q6 padding 或 absent modality 解释成别的语义。
- [ ] 同一 action trace 在 training debug decoder、sim dry-run、real deploy dry-run 中应产生同一
  layout decode 和 reference/command tick（允许并显式报告系统执行 residual）。
- [ ] 为 domain 建立明确 tag：real / sim_replay / sim_rollout / augmented_real / synthetic；训练 split 与
  benchmark 按 domain、task、scene、object、operator、time 分层，避免相邻帧泄漏。

### P1.5 有依据的随机化

- [ ] camera：外参 residual、intrinsics/crop/resize、exposure/white balance、compression、latency、drop。
- [ ] visual：复用现有 deterministic mask/background/photometric pipeline；将 augmentation config/hash
  记录到 manifest，并保留不增强 validation set。
- [ ] physics：object pose/size/mass/friction、actuator tracking/delay、joint friction/damping、
  contact parameters；范围来自 P1.3，不从任意宽 uniform 开始。
- [ ] 每个 rollout 保存 seed 和实际 sampled parameters；建立 nominal、calibration-residual、
  system-ID-confidence、held-out 四类 suite。

## 9. P2：评测与分阶段上线

- [ ] 建立固定 scene/object/seed 的 sim regression；每次变更 contract、calibration、MJCF、controller、
  postprocess 或 layout 都跑 schema + timeline + physics + media checks。
- [ ] 指标至少包含：任务成功率、qpos/TCP tracking、grasp/lift/hold、penetration/contact、command
  smoothness、latency/stale robustness、camera/mask validity、sim-real alignment residual。
- [ ] 定义 gate：`offline schema/trace → sim nominal → sim held-out → real shadow/dry-run → 小规模真机 → 扩大`。
- [ ] 真机失败 episode 必须回流到 dataset validity/label、calibration residual、system-ID validation 或
  held-out suite；不只把它当作“再多收数据”。
- [ ] 在上述 gate 稳定前，不优先投入 vectorized 大规模并行、复杂多物体 curriculum、软体/液体或
  高成本 tactile simulation。

## 10. 推荐执行顺序与每步交付物

以下把“最小 sim RL 环境”作为一条可立即开工、但不阻塞数据主干的分支；它的 task-local contract
不得替代后续 real/sim manifest。

1. **P0-RL.0 reachable physical-lift gate**：先产出 canonical-planner/manual 的 right-arm
   pregrasp/grasp/lift qpos trace、可达 cube reset 区和 contact evidence；没有纯接触 lift trace 时仅做
   environment API smoke，不开始抓取训练。
2. **P0-RL.1 scene helper + task config/spec**：freejoint/contact 查询、独立 `CubeLift*Spec`、base
   contract reference/hash、Gymnasium dependency 与 config validation。
3. **P0-RL.2 `CubeLiftEnv`**：seeded reset、15→150→1000 Hz scheduler、state-privileged observation、
   height-only reward/termination、headless deterministic/random regression。
4. **P0-RL.3 diagnostic CLI**：同一个 env 的 GUI/headless lift trace，不引入第二条 viewer 控制路径。
5. **P0.1 manifest**：schema + hash + writer/reader fail-fast tests。
6. **P0.2 timeline**：一个 real collected episode 的 source/reference/qcmd/measured trace adapter。
7. **P0.3/P0.4 dataset**：native v2.2 writer 路线、sim writer binding、标准 trajectory dataset。
8. **P0.5 direct-qpos control**：已有 group command、150/1000 Hz deadline/ZOH/headless test；下一步补
   可持久化 trace。此控制模式不依赖真机 150 Hz 数据。
9. **P0.6 direct-contact diagnostics**：将 grasp contact/attachment 的 precondition、failure trace 和
   debug-mode provenance 做实。
10. **P0.7/P0.8/P0.9 cube pick-and-place**：在 CubeLift 的 reset、clock、contact 基础上扩展 task reset、
    scripted qpos trace、GUI/headless；`debug_attachment` 的成功与纯接触结果分开统计。
11. **P1.1 paired golden episode**：real/sim 数值报告与阈值。
12. **P1.2/P1.3 calibration + system ID**：真机 150 Hz 无接触数据到位后，versioned actuator profile
    与 confidence；以此替换 direct-qpos debug 为 actuator rollout。
13. **P1.4/P1.5 training + randomization**：layout/artifact gate、基于 residual 的随机化。
14. **P2 evaluation**：held-out robustness + 分阶段真机 gate。

## 11. “可开始严肃 sim2real 训练”的最低定义

- [ ] real、sim replay、sim rollout 使用可验证的 manifest，且 `PolicyIOLayout` / frame / TCP /
  camera preprocessing 不会静默不一致。
- [ ] 同一 action/control timeline 能生成 real audit 与 sim control-like audit；qpos/TCP/delay 有
  量化残差和阈值。
- [ ] `kinematic_direct_qpos_debug` 已完成 cube pick-and-place 的工程 smoke，并且每条结果明确带有
  debug/assisted-grasp provenance；它是后续工作的推进基线，不是“严肃 sim2real”证据。
- [ ] MuJoCo 完整机器人通过 actuator-driven 150 Hz physics rollout、动态碰撞与 gripper/object
  interaction 基本验证；不再依赖直写 qpos 伪装物理控制。
- [ ] 至少一个 object task 有可重复 reset/step/success/failure、fixed-seed regression 和 paired
  real benchmark。
- [ ] training reader 对 video/mask/depth/tactile/lowdim 的 timestamp、validity、layout、domain 都
  可追溯，且 policy artifact 能在 sim/real 两端通过 compatibility gate。
- [ ] domain randomization 范围来自 calibration/system-ID residual，并在 held-out suite 上报告结果。

## 12. 当前关键入口（便于后续继续落地）

- `dataset_contracts/`：v2.2 layout/artifact、Parquet、stats、tactile、wrist depth contract。
- `dataset_interface/`：metadata validity、masked video pipeline、policy-I/O row reader、Torch adapter。
- `teleop_runtime/collection/writer_v21.py` 与 `schema_v22.py`：real collection 的 v2.1 storage + v2.2 opt-in adapter。
- `utils/dataset_postprocess/`：清理、对齐、label、v2.1→v2.2、undistort、wrist final-mask。
- `robot_motion/`：RobotAction、layout-aware deploy、45 Hz coupled planner、150 Hz interpolation。
- `sim/contracts/`：scene/action/observation/timing contract；`sim/configs/`：versioned profile。
- `sim/mujoco/`：MJCF/scene、canonical pose planner adapter、qpos sink、kinematic replay/export、isolated cube drop。
- `utils/hand_eye/`、`utils/head_eye/`、`cameras/configs/rm65/cameras/`：calibration 与 camera registry。
