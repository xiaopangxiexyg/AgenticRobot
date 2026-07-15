# 单臂 IK、双臂耦合 IK 与防碰撞规划迁移计划（临时稿）

> 目标：将单臂 IK、双臂耦合 IK、关节限制、碰撞检查、跳变保护和 last-safe hold 的运动规划职责收敛到 robot_motion。  
> 非目标：本计划不迁移 VIVE/OpenVR、tracker UDP、会话/校准、RM65 SDK/CANFD、MuJoCo scene，也不改变现有传输协议和传输时序。  
> 原则：先建立兼容 facade 和行为基线，再迁移调用方；不在迁移过程中并行维护两套“看起来都正确”的 IK。

## 1. 总体判断

这项工作不是把一份 IK 文件从 teleop_runtime 移到 robot_motion，而是把当前分散在三处的职责重新收敛：

1. robot_motion/rm65 已经拥有一部分共享实现；
2. teleop_runtime/retargeters/dual_arm_collision_aware.py 仍拥有大量实时双臂 IK、分支选择、碰撞和恢复逻辑；
3. teleop_runtime/backends/arms/rm65.py 仍拥有单臂 Cartesian→IK 和部分位置安全处理；
4. sim/mujoco/ik.py、replay 和 deploy 已经开始使用共享双臂 planner，但还需要与 real teleop 的输入和状态初始化方式统一。

因此，迁移的正确顺序是：

~~~text
建立公共契约
  → 给现有 robot_motion 实现加 facade
  → 先迁移单臂位置链
  → 再迁移双臂 target builder 与 coupled planner 的边界
  → 统一 real/sim/replay/deploy 的 planner 初始化和结果
  → 最后删除 backend/retargeter 中的重复 IK
~~~

不建议第一步就重写 DualArmCollisionAwareRetargeter，也不建议第一步把现有通用 IKSafetyResolver 强行接到 RM65 主链。前者会同时改变输入重定向、碰撞行为和输出命令；后者当前还没有被主要 runtime 调用，且其 relaxation/local-step 语义与现有 RM65 双臂 planner 不完全等价。

## 2. 当前实现清单

### 2.1 单臂 IK

当前已经存在：

- robot_motion/rm65/ik.py：
  - RM65SingleArmIKConfig；
  - build_rm65_single_arm_ik_solver；
  - solve_rm65_single_arm_ik；
  - 对 rm65_inspire_tac.ik.SingleArmIKSolver 的 lazy adapter；
  - controlled frame、hand_base、arm_link6 和 tool_tcp 的 frame 归一化；
  - iterative dry-run 与 bounded single-step 两种调用模式。
- teleop_runtime/backends/arms/rm65.py：
  - _solver_for_tcp_frame；
  - _target_pose_for_solver；
  - _solve_cartesian_command；
  - _apply_jump_gate；
  - send_command 收到 cartesian_pose 时直接解 IK。

当前问题：

- robot_motion 的单臂 helper 只返回 qpos 数组，没有统一的成功/保持/跳变/奇异/误差结果；
- backend 仍然是 Cartesian→IK 的拥有者，导致位置控制链的 planner 可能被绕过；
- backend 的单臂 IK 与双臂 planner 采用不同的状态锚点和 metadata；
- position 和 force/impedance 都可能使用 cartesian_pose，但两者的语义不同，不能只按 command_type 判断。

### 2.2 双臂耦合 IK 和碰撞规划

当前已经存在：

- robot_motion/rm65/ik.py：
  - RM65DualArmCollisionIKConfig；
  - build_rm65_dual_arm_collision_ik_solver；
  - 对 MultiTaskIKSolver 的 RM65 scene/profile 适配。
- robot_motion/rm65/position_collision.py：
  - RM65PositionCollisionPlanner；
  - 完整双臂 solver state；
  - pose frame 和 tool transform 转换；
  - 关节绝对限制；
  - 参考状态相对的 max joint step；
  - endpoint collision distance；
  - collision backoff；
  - passive side freeze；
  - last-safe hold；
  - qpos endpoint validation；
  - q_ik_raw、q_ref、realized pose、collision attempts 和 timing metadata。
- sim/mujoco/ik.py：
  - 只把 MuJoCo scene 状态提供给共享 planner；
  - 不使用 MuJoCo Jacobian；
  - 不改变 scene qpos；
  - PoseIKController 只做仿真侧结果转换。
- teleop_runtime/position_control/rm65_collision.py：
  - 从已经启动的 real backend 读取两臂 active tool TCP；
  - 建立共享 planner；
  - 要求两侧真实 qpos 和 tool profile，不猜测缺失状态。

当前问题：

- teleop_runtime/retargeters/dual_arm_collision_aware.py 仍然自己创建 MultiTaskIKSolver；
- 该 retargeter 还拥有 target smoothing、candidate branch、jump gate、局部步进、碰撞检查、posture recovery、输出 qpos smoothing 和 command metadata；
- real teleop、collection、debug、SAPIEN preview 仍直接依赖这个大 retargeter；
- robot_motion 的 planner 与旧 retargeter 的恢复策略并不完全相同，不能直接替换后假定行为一致；
- 当前 collision 主要是 endpoint certification，75→150 插值段仍明确不是 segment collision certification；
- collision planner 需要双臂初始 qpos，但部分旧入口仍使用 default initial qpos 或 backend 本地缓存，状态来源需要统一。

### 2.3 已有但尚未统一的通用 IK 栈

robot_motion 还存在：

- types.py 中的 IKResolveRequest、IKResolveResult、IKCandidate；
- interfaces.py 中的 KinematicsProvider、CollisionChecker；
- pinocchio_provider.py；
- ik_resolver.py 中的 IKSafetyResolver；
- composite_provider.py、seed_expansion.py、profiles.py。

它们提供了有价值的通用候选、FK、限位、deadline 和安全解析能力，但当前主要 runtime 并没有以它们作为 RM65 主路径。迁移时必须做一次语义对照：

| 现有栈 | 当前 RM65 双臂 planner |
|---|---|
| 单向量 IK candidate/resolver | 完整双臂 atomic qpos |
| provider 生成候选，resolver 选择 | MultiTaskIKSolver 直接生成全双臂配置 |
| profile relaxation/local step | joint step、collision backoff、jump hold |
| generic CollisionChecker | full dual-arm collision scene |
| IKResolveResult | RM65PositionCollisionPlanResult |

建议：第一阶段不强制合并这两套执行语义。先把公共字段和 capability 对齐；如果测试证明语义一致，再复用其中的纯组件。否则保留 generic resolver 作为通用/实验 API，并让 RM65 planner 通过明确 adapter 接入，而不是让两个状态机互相嵌套。

## 3. 目标边界

### 3.1 robot_motion 的目标职责

robot_motion/rm65 应拥有：

- RM65 robot model、scene profile、joint layout；
- 单臂 IK solver adapter；
- 单臂 position planner；
- 双臂 coupled IK solver adapter；
- 双臂 collision-aware position planner；
- frame/tool transform 的纯数学转换；
- joint limits、step/velocity limits；
- branch preference、jump gate、hold/recovery；
- endpoint collision validation；
- planner result 和诊断 metadata；
- 不依赖设备和传输的时间预算、deadline 和 deterministic test hook。

robot_motion 不应拥有：

- VIVE tracker discovery、字符排序、UDP；
- TeleopCalibration 的生命周期和 home recalibration；
- tracker 丢失时的操作者会话策略；
- RM65 连接、enable、Movej_CANFD、force-position API；
- 真实力传感器 tare 和硬件模式切换；
- MuJoCo scene mutation、physics step、camera/task；
- command queue 的线程和 transport retry。

### 3.2 teleop_runtime 的目标职责

teleop_runtime 继续拥有：

- VIVE/手部输入；
- tracker pose 到 arm-base target pose 的映射；
- clutch、workspace input policy、输入 smoothing；
- backend start 后读取 authoritative qpos 和 active TCP profile；
- 建立、reset、重置 epoch 和关闭 shared planner；
- real-time worker；
- 将 planner 结果转换为现有 TeleopCommand；
- final dispatch 和 transport receipt；
- legacy Cartesian compatibility facade。

注意：target builder 可以产生双臂目标，但不能在其中创建 IK solver、调用 compute_min_collision_distance 或选择 qpos branch。

### 3.3 sim 的目标职责

sim/mujoco/ik.py 保持为 adapter：

- 从 scene 读取完整两臂 qpos；
- 提供显式 tool profile；
- 调用 robot_motion planner；
- 把 planner result 转成 sim controller result；
- 最终由 scene sink 执行 qpos。

sim 不应为 real teleop 复制一套单臂或双臂 IK。sim/mujoco/diagnostics/jacobian_ik.py 可以保留为诊断工具，但不能成为 canonical control path。

## 4. 目标公共契约

不要求立刻创建所有新目录，但迁移前必须先确定下面几类对象的语义。

### 4.1 单臂请求和结果

建议新增或整理为一个 robot_motion 层的 SingleArmIKRequest/SingleArmIKResult，字段至少表达：

请求：

- side 和 robot model/profile；
- target pose；
- target reference frame：arm_base 或明确的 world；
- controlled frame：tool_tcp 或 arm_endpoint；
- measured_qpos；
- solver seed/reference qpos；
- last-safe qpos；
- max joint step、joint limits、time budget；
- tool transform/profile id；
- timestamp、sequence、source；
- 是否允许 dry-run iterative、是否允许 relaxation；
- capability 和 metadata。

结果：

- status；
- valid、hold；
- q_ik_raw；
- q_ref；
- target pose 与 realized pose；
- position/orientation error；
- joint limit/step modification reasons；
- jump/singularity/timeout/collision status；
- solver timing；
- source、sequence 和 profile metadata。

建议至少区分以下结果状态：

~~~text
tracking
limited
relaxed
hold_last_safe
invalid_input
unreachable
singularity
deadline_exceeded
solver_exception
unsupported_frame
~~~

单臂 helper 只返回 np.ndarray 的旧接口可以保留为 compatibility wrapper，但新 runtime 不应依赖它来判断安全状态。

### 4.2 双臂请求和结果

建议把当前 RM65PositionCollisionPlanner 的语义整理成 BimanualPositionPlanRequest/Result 的公共契约，RM65 实现仍可以留在 robot_motion/rm65。

请求：

- 两侧当前 authoritative qpos，必须完整；
- 两侧 active tool transform/profile，必须完整；
- active target sides；
- active side 的 pose target；
- target reference frame；
- controlled frame；
- reference timestamp/sequence；
- planner/profile id；
- max joint velocity/step；
- collision profile、minimum distance、backoff policy；
- last-safe epoch；
- source 和 metadata。

结果：

- 两侧都返回 q_ref，即使一侧是 passive hold；
- active_sides/passive_sides；
- q_ik_raw_by_side；
- q_ref_by_side；
- target/realized endpoint pose；
- planner status、hold、reason；
- endpoint collision distance 和 attempts；
- endpoint_collision_checked；
- segment_collision_checked=false；
- segment_collision_certified=false；
- solver/total timing；
- planner sequence 和 reset epoch。

双臂 planner 必须是 atomic result：不能返回 left 已经更新、right 失败但调用方仍把 left 发出去的半成功结果。最终 transport 是否原子是另一个 runtime/硬件问题，但 planner 输出必须是成对的。

### 4.3 solver 和 planner 分层

需要明确两个概念：

- solver：只负责根据目标和 seed 求 raw q，不能做 transport，也不应决定最终 hold；
- planner：负责 frame 转换、seed、限位、碰撞、jump gate、recovery、last-safe 和结果状态。

推荐的调用关系：

~~~text
SingleArmIKSolver / CoupledIKSolver
                  │
                  ▼
SingleArmPositionPlanner / BimanualPositionCollisionPlanner
                  │
                  ▼
PositionReferenceFrame / JointPositionCommand
~~~

当前 build_rm65_single_arm_ik_solver 和 build_rm65_dual_arm_collision_ik_solver 属于 solver construction；RM65PositionCollisionPlanner 已经接近 planner。下一步应避免把更多 planner policy 继续塞回 solver builder。

## 5. 迁移前必须冻结的行为基线

迁移不是先追求目录变化，而是先证明新旧路径的差异是可解释的。建议建立以下 baseline fixture：

### 5.1 单臂 fixture

每个 side 至少覆盖：

- 当前 qpos 附近的小位姿变化；
- 大位姿变化；
- tool_tcp 和 arm_endpoint；
- joint limit 附近；
- max step clipping；
- solver 无解；
- non-finite target；
- jump gate；
- dry-run iterative；
- hardware-like bounded step；
- stale/current state 不一致。

记录：

- target pose、frame、tool transform；
- current/seed qpos；
- raw qpos；
- final qpos；
- pose error；
- status、hold、reason；
- solve time；
- command metadata。

### 5.2 双臂 fixture

至少覆盖：

- 两侧同步 tracking；
- 右侧 active、左侧 passive；
- 左侧 active、右侧 passive；
- 双臂都 inactive；
- 关节限制；
- collision boundary slide；
- collision backoff；
- 无安全候选；
- initial qpos 不安全；
- solver exception；
- branch flip；
- small pose / large q jump；
- hold recovery；
- target smoothing 开/关；
- output smoothing 开/关；
- complete state 缺失；
- tool profile 缺失或不一致。

重点不是只比较最终 qpos，还要比较：

- passive side 是否保持；
- last-safe 是否更新；
- collision distance；
- hold 是否进入；
- reason/status；
- endpoint/segment certification metadata。

### 5.3 现有路径的行为分组

对 DualArmCollisionAwareRetargeter 中的逻辑逐项标记：

| 逻辑 | 目标归属 |
|---|---|
| PoseToArmRetargeter 的 tracker/desk/base 映射 | teleop_runtime |
| TeleopCalibration、clutch、输入 skipped | teleop_runtime |
| target pose smoothing | 根据来源决定；输入 smoothing 留 runtime，控制参考 smoothing 放 robot_motion/position_control |
| pose frame/tool transform | 纯转换放 robot_motion，配置加载留 runtime |
| MultiTaskIKSolver construction | robot_motion/rm65 |
| branch candidate/seed/posture preference | robot_motion |
| joint absolute/step/velocity limits | robot_motion |
| collision distance/backoff | robot_motion |
| jump gate | robot_motion |
| last-safe hold/recovery | robot_motion |
| inactive tracker jitter hold | teleop_runtime 的输入策略；planner 只接收 active/passive side |
| qpos output smoothing | robot_motion/position_control，不能藏在 transport |
| TeleopCommand metadata translation | teleop_runtime compatibility adapter |
| backend dispatch/receipt | teleop_runtime |

这张表是迁移时的“禁止误搬”清单：不应因为某段代码和 IK 在同一个方法中，就把所有代码一起移动。

## 6. 分阶段迁移方案

### 阶段 M0：建立基线和 feature path

目标：不改变默认行为，只把当前路径可观测化。

工作：

1. 为单臂和双臂路径统一记录：
   - ik_path；
   - solver/profile version；
   - tool profile id/hash；
   - scene profile；
   - qpos_state_source；
   - planner reset epoch；
   - measured/reference/last-accepted state source。
2. 为现有 real teleop、deploy、replay、MuJoCo replay 建立 golden trace 输出。
3. 增加显式路径名称：
   - legacy_backend_cartesian_ik；
   - shared_single_arm_ik；
   - shared_bimanual_collision_ik。
4. 禁止通过“solver 初始化失败后自动换另一套 solver”来隐藏差异。
5. 保持现有 VIVE transport、TeleopCommand dispatch、RM65 SDK 调用和 MuJoCo sink 不变。

验收：

- 默认命令行为和原路径一致；
- 每个 qpos 输出都能追溯到唯一的 IK path；
- 传输层的 command type、发送频率和 receipt 没有变化；
- 单臂和双臂 fixture 可以重复运行。

### 阶段 M1：建立 robot_motion facade，不先替换调用方

目标：先把公共入口稳定下来，让旧调用方可以继续运行。

建议新增的概念接口：

- SingleArmIKPlanner；
- BimanualPositionCollisionPlanner；
- CollisionModel；
- PlannerState/PlannerResetResult；
- SingleArmIKResult；
- BimanualPositionPlanResult。

实现方式：

1. SingleArmIKPlanner 初期委托 robot_motion/rm65/ik.py 的现有 solver helper；
2. BimanualPositionCollisionPlanner 初期委托 RM65PositionCollisionPlanner；
3. 旧 RM65PositionCollisionPlanResult 作为内部/兼容实现，增加到公共结果的转换；
4. 旧 TeleopCommand metadata 通过 adapter 生成，不让 planner 依赖 TeleopCommand；
5. 旧 root-level IKResolveRequest/Result 暂时保留，新增 adapter 让其可以被测试，但不把两个状态机嵌套到实时链；
6. 将 lazy import、RM65-specific solver 和 scene profile 的加载都封装在 robot_motion/rm65。

推荐的第一阶段目录可以是：

~~~text
robot_motion/
  kinematics/
    contracts.py       # 公共 request/result/status
    interfaces.py      # solver/planner/collision protocols
    adapters.py        # legacy generic IK 兼容适配
  rm65/
    ik.py              # RM65 solver construction
    single_arm.py      # RM65 single-arm planner facade
    position_collision.py  # 当前双臂 planner 的 canonical implementation
~~~

这一阶段可以只新增 facade，不搬动大段实现。这样出现行为差异时仍能快速回到原实现。

验收：

- facade 与旧 helper 在 fixture 上结果一致；
- planner 不 import teleop_runtime；
- planner 不 import hardware transport；
- planner 不接受或返回 socket/CAN/SDK 对象；
- sim 和 real runtime 都能通过 facade 创建 planner。

### 阶段 M2：迁移单臂 position IK

目标：位置控制模式的 Cartesian→IK 不再由 RM65 backend 作为主入口执行。

#### M2.1 增加单臂 planner

单臂 planner 负责：

1. 校验 target frame/tool frame；
2. 根据 measured/last-safe qpos 初始化 solver；
3. 调用 RM65 solver；
4. 应用 joint limits 和 max step；
5. 执行 jump/singularity/finite 检查；
6. 生成 q_ik_raw、q_ref 和 structured result；
7. 在失败时返回 last-safe hold；
8. 不发送任何命令。

单臂 planner 不应默认宣称拥有双臂碰撞能力。若需要 self/environment collision，必须通过显式 CollisionModel capability 提供；没有模型时返回 unsupported，而不是静默通过。

#### M2.2 在 runtime 侧增加 command resolution

real position path 改成：

~~~text
PoseToArmRetargeter
  → single-arm target intent
  → robot_motion SingleArmPositionPlanner
  → joint_vector / PositionReference
  → RM65 position sink
~~~

force/impedance path 保持：

~~~text
PoseToArmRetargeter
  → cartesian_pose target
  → RM65 impedance/force backend
~~~

这两个路径必须由 control_mode/capability 明确选择，不能仅依据 command_type=cartesian_pose 猜测。

#### M2.3 backend 兼容策略

迁移期间 RM65 backend 可以保留 cartesian_pose 分支，但应满足：

- shared position mode 下，调用方不再发送 cartesian_pose；
- backend 发现 shared position mode 收到 cartesian_pose 时 fail-closed 并提示配置错误；
- legacy_backend_cartesian_ik 模式可以保留旧行为，用于回滚和非主链兼容；
- 如果必须在兼容分支中解 IK，backend 只能调用 robot_motion 的 planner facade，不能保留另一份 solve/jump gate 实现；
- force/impedance 的 cartesian_pose 分支不受此迁移删除。

#### M2.4 单臂验收

- 同一 target/current/tool profile，新旧 q_ref 在容差内一致；
- position mode 不再触发 backend 的独立 Cartesian IK；
- force/impedance mode 仍然收到原有 cartesian target；
- backend transport command type 和发送时序保持原有行为；
- IK 失败进入 structured hold，不会把异常 qpos 发给 SDK；
- 单臂 dry-run、real fake backend、replay 都走同一 planner。

### 阶段 M3：拆分双臂 retargeter

目标：DualArmCollisionAwareRetargeter 变成“输入目标 facade”，不再是 IK/碰撞算法的拥有者。

#### M3.1 先拆 target builder

保留在 teleop_runtime 的部分：

- left_pose_retargeter/right_pose_retargeter；
- observation 和 calibration；
- VIVE/hand tracker 输入；
- desk/operator/base frame 映射；
- clutch、input skipped；
- 输入侧 target smoothing；
- 生成带 timestamp、frame、active side 的 BimanualPoseTarget。

建议把当前 DualArmTargetContext 的纯目标部分转成不依赖 TeleopCommand 的对象，例如：

~~~text
BimanualPoseTarget
  timestamp
  target_pose_by_side
  active_sides
  reference_frame
  controlled_frame
  source
  input_metadata
~~~

TeleopCalibration 不能作为 robot_motion planner 的构造依赖；它必须在 target builder 中完成转换。

#### M3.2 将纯规划逻辑迁移到 robot_motion

优先迁移的逻辑：

1. full dual-arm solver construction；
2. solver frame/task/posture configuration；
3. qpos split/embed；
4. branch candidate 和 posture preference；
5. joint limit/step/velocity constraint；
6. collision distance/check/backoff；
7. jump gate；
8. last-safe hold；
9. hold recovery；
10. q_ik_raw/q_ref/realized pose 和 timing metadata。

暂时不要迁移：

- TeleopObservation/TeleopCalibration 类型；
- tracker inactive jitter；
- 仅服务于 VIVE 输入的 smoothing；
- TeleopCommand 的 command dictionary 结构；
- backend dispatch/receipt。

#### M3.3 处理当前 planner 与旧 retargeter 的行为差异

不能直接把：

~~~text
DualArmCollisionAwareRetargeter
  → RM65PositionCollisionPlanner
~~~

当作等价替换。迁移前需建立差异矩阵：

| 行为 | 旧 retargeter | 当前 shared planner | 迁移动作 |
|---|---|---|---|
| raw coupled solve | 有 | 有 | 复用 shared |
| joint limit/step | 有多级策略 | 有硬约束 | 先对齐参数和结果 |
| collision endpoint | 有 | 有 | 复用 shared |
| collision backoff | 有局部路径 | 有 backoff scales | 用 fixture 验证差异 |
| branch selection | 有 | 需确认是否完全一致 | 先抽成 policy |
| jump gate | 有 | 需补齐/适配 | 迁移到 planner |
| hold recovery | 有多种 seed | 有 last-safe hold | 分阶段迁移 |
| target smoothing | 有 | planner 明确不做时序平滑 | 留在 input/reference layer |
| output qpos smoothing | 有 | planner 明确不做 | 放到 position_control |
| inactive jitter hold | 有 | 不属于通用 planner | 留在 teleop_runtime |

如果差异是安全策略差异，必须显式命名成不同 profile；不能通过 metadata 覆盖成“同一 IK”。

#### M3.4 保留 facade

迁移后的 DualArmCollisionAwareRetargeter 只做：

~~~text
observation + calibration
  → BimanualPoseTarget
  → shared planner
  → TeleopCommand / joint_vector
~~~

它可以继续保留 build_target_context、build_command_for_context 和 retarget 方法，以兼容现有 session、collection、debug 和 replay diagnostic，但这些方法内部不得再直接调用 MultiTaskIKSolver。

验收：

- teleop_runtime/retargeters/dual_arm_collision_aware.py 不再构造或直接调用双臂 IK solver；
- planner 可以在没有 teleop_runtime import 的情况下运行；
- 旧 retargeter 的公开方法和 command metadata 仍可用；
- 单侧目标仍保留被动侧 qpos，且被动侧不使用 guessed default；
- real/sim/replay 使用同一 planner result。

### 阶段 M4：统一 planner 初始化和状态来源

这是迁移中最关键的安全阶段。

#### M4.1 real 初始化

real backend start 后，runtime 必须按以下顺序：

~~~text
start both arm backends
  → read fresh measured qpos for both sides
  → read active tool TCP/profile for both sides
  → build planner
  → planner.reset(complete qpos pair)
  → verify initial collision safety
  → publish planner epoch
  → allow position target
~~~

不能：

- 用单侧 qpos 猜另一侧；
- 用 default initial qpos 替代 fresh hardware state；
- 用 backend cached current_qpos 掩盖读取失败；
- 用默认 tool_tcp/identity transform；
- 在 planner reset 失败后继续发送 position command。

#### M4.2 运行中状态锚点

需要明确三种状态：

- measured_qpos：硬件/仿真反馈；
- last_accepted_qref：planner 已接受的安全参考；
- last_emitted_qcmd：插值器已发出的高频命令。

建议：

- planner 的 IK seed/last-safe 以 last_accepted_qref 为主；
- measured_qpos 用于 freshness、pair skew、tracking drift 和 fault gate；
- 发生 measured/reference 漂移超过 profile 阈值时，进入 hold 或显式 re-sync；
- 不在每一帧无条件用 stale measured_qpos 覆盖 planner state；
- reset/recovery 时增加 epoch，旧 target 不得跨 epoch 继续生效。

#### M4.3 sim/replay 初始化

- sim 从 scene 读取完整两臂 qpos，然后 reset planner；
- replay 必须记录 qpos_state_source 和 tool profile；
- semantic action replay 重新执行 planner；
- executed qpos replay 只做 endpoint validation 或显式 passthrough，不重新宣称执行了 IK；
- 任意缺失状态都 fail-closed，不回退默认姿态。

验收：

- planner reset metadata 可追溯；
- 初始碰撞不安全时不会进入执行态；
- measured/reference/command 三类状态不会混用；
- planner epoch 和 sequence 在日志中单调且可检查。

### 阶段 M5：迁移所有调用方

按风险由低到高：

1. robot_motion/tests/tracker_replay.py：直接使用 shared planner/result，保留旧 retargeter 对比模式；
2. sim/mujoco/ik.py 和 pose-ik one-shot；
3. MuJoCo replay；
4. real deploy position executor；
5. teleop_runtime/replay position path；
6. SAPIEN/virtual preview；
7. collection；
8. real VIVE teleop 主链。

每迁移一个调用方，都要验证：

- target frame 和 tool transform 相同；
- qpos state source 相同；
- planner profile 相同；
- output command 只是由 adapter 转换，不重新解 IK；
- transport sink 仍接收原有最终 command 类型和频率。

### 阶段 M6：删除重复实现，保留最小兼容层

只有 M2～M5 的验收全部通过后，才做清理：

- 删除或改写 RM65 backend 内部的独立 single-arm solve 逻辑；
- 删除 DualArmCollisionAwareRetargeter 内部的 MultiTaskIKSolver、collision distance 和 branch recovery；
- 保留 legacy Cartesian facade，但明确标注 deprecated；
- 将旧 import path re-export 到 robot_motion；
- 从 teleop_runtime 移除对 rm65_inspire_tac IK tuning/pipeline 的直接依赖；
- 将纯 TCP/frame safety warning 迁到 robot_motion profile adapter；
- 更新 README、capability matrix 和诊断字段。

不应在这一步同时重命名所有目录、改 action layout 或改 transport。清理和架构迁移要分开。

## 7. 文件级迁移矩阵

| 当前文件 | 迁移后的职责 | 处理方式 |
|---|---|---|
| robot_motion/rm65/ik.py | RM65 solver construction 和 frame adapter | 保留，补充 planner-facing result/diagnostics |
| robot_motion/rm65/position_collision.py | canonical dual-arm position/collision planner | 保留为实现核心，外面增加公共 facade |
| robot_motion/rm65/scene_profiles.py | robot model/collision scene profile | 保留，增加 profile version/hash |
| robot_motion/rm65/tool_profiles.py | 纯 tool transform/profile 解析 | 保留；active hardware profile 由 runtime 注入 |
| robot_motion/types.py | generic IK compatibility types | 保留并与新 contracts 对齐，不再创建第三套状态枚举 |
| robot_motion/ik_resolver.py | generic single-arm resolver | 先做语义对照；未经验证不要直接替换 RM65 main path |
| robot_motion/pinocchio_provider.py | generic provider adapter | 可复用于单臂，但必须经过 RM65 frame/step/result adapter |
| robot_motion/position_control/contracts.py | PositionReference/command boundary | 增加 planner result、epoch、state source 的必要字段 |
| teleop_runtime/retargeters/pose_to_arm.py | input pose target builder | 保留，不放 IK |
| teleop_runtime/retargeters/dual_arm_collision_aware.py | teleop target builder + compatibility facade | 删除 solver/planner ownership，保留公开方法和 metadata adapter |
| teleop_runtime/backends/arms/rm65.py | real state/position/force sink | 位置主链删除独立 IK；保留 force 和 legacy facade |
| teleop_runtime/position_control/rm65_collision.py | real planner factory/init boundary | 保留并作为唯一 hardware-state → planner reset 入口 |
| sim/mujoco/ik.py | sim planner adapter | 保留，不复制 IK |
| sim/mujoco/controllers.py | sim command sink | 只接受 planner 输出；diagnostic IK 不进入主链 |
| teleop_runtime/replay/executor.py | real replay shell + shared planner consumer | 保留 runtime，统一 planner factory |
| robot_motion/tests/tracker_replay.py | shared planner golden/diagnostic tests | 增加 direct planner path，旧 retargeter 仅作 parity reference |

## 8. 失败、保持和回滚策略

### 8.1 planner 失败

统一规则：

- invalid target、solver exception、collision distance unavailable、no safe candidate、initial state unsafe 都不能输出新 qpos；
- 双臂返回 atomic last-safe pair；
- 结果必须带 hold=true、failure reason 和 planner sequence；
- runtime 决定是否继续发 hold command，但不能把 planner failure 当作成功 tracking；
- collision-required real position path 不得 silent fallback 到 backend 单臂 Cartesian IK。

### 8.2 迁移回滚

保留显式的 path/profile：

~~~text
shared_bimanual_collision_ik
shared_single_arm_ik
legacy_backend_cartesian_ik
~~~

回滚只允许改变 planner path，不改变：

- VIVE tracker transport；
- TeleopCommand schema；
- RM65 SDK/CANFD 调用；
- 150 Hz sink 的发送接口；
- force/impedance path。

legacy path 只用于兼容和诊断，不能因为 shared planner 初始化失败就自动切回 real motion path。共享 planner 失败时，默认是 hold/fault；只有启动参数明确选择 legacy，才允许进入 legacy。

## 9. 测试与验收门槛

### 9.1 单元测试

robot_motion 内应增加：

- frame/tool transform；
- qpos split/embed；
- single-arm result/status；
- joint bound/step intersection；
- branch selection deterministic；
- jump gate；
- hold/recovery；
- collision distance unavailable fail-closed；
- passive side freeze；
- planner epoch/sequence；
- endpoint-only certification metadata。

### 9.2 contract tests

同一组 request/profile 在不依赖 hardware/sim 的环境中，验证：

- result shape、units、frame；
- q_ik_raw 与 q_ref 的关系；
- hold 时 q_ref 等于 last-safe；
- 双臂结果始终包含两侧；
- active/passive side 语义；
- segment_collision_certified=false 在当前实现中永不被误报为 true。

### 9.3 parity tests

对旧 retargeter 和新 planner 做离线逐帧对比：

- qpos 误差；
- endpoint pose error；
- collision distance；
- status/hold；
- branch candidate；
- joint limit reason；
- solve time budget。

差异必须归类为：

- 数值误差；
- 明确 profile 改变；
- bug fix；
- 尚未迁移的 legacy policy。

不能以“最终看起来差不多”作为通过标准。

### 9.4 integration tests

- real fake backend：两臂状态读取、tool profile、planner reset、qpos dispatch；
- real fake backend：planner failure 后 hold/fault；
- sim：同一 action 在 final sink 前得到同样 q_ref/qcmd；
- replay：semantic action replay 与 deploy planner trace 一致；
- qpos replay：不重新执行 IK，且显式记录 passthrough/endpoint validation；
- force/impedance：仍接受 Cartesian target，不经过 position IK；
- backend：shared position mode 收到 Cartesian command 时明确拒绝。

### 9.5 性能和时间预算

在真实 75 Hz 目标下记录：

- solver；
- collision distance；
- branch/recovery；
- total planner；
- reference publication；
- command interpolation；
- dispatch callback；

任何超过 reference budget 的 frame 都必须在 metadata 中体现，不能通过减少日志或吞掉异常伪装成实时。

## 10. 迁移完成的判据

满足以下条件才可以说 IK 已真正归入 robot_motion：

1. real/sim/replay/deploy 的 canonical position path 都调用同一个 planner facade；
2. real teleop retargeter 不再直接构造或调用 IK/collision solver；
3. RM65 position backend 不再拥有 canonical Cartesian→IK；
4. force/impedance 的 Cartesian target 路径仍独立存在；
5. planner 初始化只接受显式完整状态和 tool profile；
6. 双臂输出是 atomic q_ref pair；
7. planner 失败不会 silent fallback 到另一套运动规划；
8. 旧路径仍可显式回滚，但不是默认隐式路径；
9. final sink 前的 golden trace 在 sim/real/replay/deploy 一致；
10. transport 逻辑、SDK 接口和发送节拍没有因本次迁移被重写。

## 11. 建议的实际执行顺序

最终建议按下面顺序落地：

1. 先实现 M0 的 trace、path 名称和 fixture；
2. 再实现 M1 facade 和公共 result，不改变 runtime；
3. 迁移单臂 shared position path，保留 force/legacy Cartesian；
4. 对双臂 retargeter 做行为差异矩阵；
5. 把 target builder 与 coupled planner 分开；
6. 让旧 DualArmCollisionAwareRetargeter 通过 facade 工作；
7. 统一 real backend-start 后的双臂 reset；
8. 依次迁移 deploy、replay、sim、collection、real teleop；
9. 删除 backend/retargeter 重复 IK；
10. 最后再考虑 segment collision、单臂环境碰撞和更通用的 IKSafetyResolver 合并。

最重要的工程判断是：**先迁移“决策权”，再迁移“文件位置”。** 当 planner 已成为唯一生成 q_ref/qpos 的决策者后，删除旧文件只是清理工作；如果先搬文件而没有冻结状态、frame、profile 和结果契约，最终仍会得到 real、sim 和 teleop 各自一套行为不同的 IK。
