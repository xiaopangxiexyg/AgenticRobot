# B601-RS 硬件开发交接说明

> 版本：2026-09-08。本文是 B601-RS 当前代码、真机 commissioning、重力采集和
> Harness/Git 交接的统一入口。它描述可复核的实现和流程，不授予新的使能、运动或
> 合并权限。任何真机动作仍须由现场操作者在确认机械支撑和物理急停后前台执行。

## 0. 先看结论

- 当前候选位于
  `/home/zc/HandMotion/Developing/HarnessLocal/agentic-robot/Worktrees/AR-20260906-b601-gravity-startup-handoff-r1/AgenticRobot`，分支为
  `task/AR-20260906-b601-gravity-startup-handoff-r1`。
- 本文撰写时实际代码提交为 `7fb67ccb611dfd40a043e5a25123e0a2537639ae`，tree 为
  `d809c202755a11cc52ba9fe201c89d9a22da901e`，工作树干净；新增本文后旧 tip 不再是
  最终交接 tip，交接前必须重新执行本文第 12 节的 pin 检查。
- 候选尚未合并到 `AgenticRobot-development-mainline`。当前 development checkout
  的现场 HEAD 为 `c0140d0fbf362bf9b3e94e840548a007b0a0039f`；使用
  `git merge-base --is-ancestor <candidate> <development>` 重新确认，不得凭分支名推断
  “已合并”。
- 外部 reBotArm 源码当前 pin 为 commit
  `5ba28acef46237eb6a7560658bbc43b06cf8a259`、tree
  `ddd34d2b400b3987d4875bd6a99bd29b74d39dac`；
  `/home/zc/HandMotion/Developing/reBotArm_control_py` 工作树干净，配置文件
  `config/rebotarm_rs.yaml` SHA-256 为
  `be51e877050cc4716188fca3b4f2a41109682513c7715c71594066a133ba5cf9`。
- 候选中的 `hardware/reBotArm_control_py` gitlink 指向上述 commit，但当前任务 checkout
  显示为未初始化（`git submodule status` 前缀为 `-`）。真机命令必须显式使用
  `--rebot-source-root /home/zc/HandMotion/Developing/reBotArm_control_py`，不能依赖一个
  未记录版本的本地模型副本。
- 已完成的代码重点是：六轴 gravity+damping protective hold、启动锁到最终控制器的
  连续 handoff、J2/J3/J4 反馈边界兼容、qvel/qacc/非有限值 guard，以及交互式多姿态重力
  残差采集和离线 candidate fit。
- 当前可以继续做正式采集和离线拟合；不能宣称重力参数、摩擦、真实力矩比例、高带宽
  触觉或最终 `HARDWARE_PASS` 已完成标定。

## 1. 任务目标与明确非目标

### 1.1 目标

B601 的第一阶段是本地有线、六自由度 arm 的可拖动重力补偿和末端阻抗 commissioning：

1. 用同一份 pinned URDF/Pinocchio 模型计算 `g(q)`、FK 和 Jacobian。
2. 用完整 `joint1..joint6` 共同实现末端平移或 6D 位姿阻抗；不能只使能 J6，不能把
   其他关节留在去使能状态后声称完成末端阻抗。
3. 解决“使能 → 锁止 → 重力补偿/阻抗”的启动力矩空窗，避免启动时明显前倾或下坠。
4. 首次 Ctrl-C、普通异常、控制时长结束和可恢复反馈异常优先保持全臂重力补偿加阻尼；
   只有无法继续发出完整安全命令或操作者明确最终停止时才全臂失能。
5. 用可复核的 JSON/log、固定 commit/tree、配置和 vendor pin 形成独立证据。

### 1.2 不属于本任务的工作

- 不把 B601 backend 变成 policy、网络遥控、leader-follower blending 或 action-chunk
  调度器；这些属于 `control_composition`/`teleop_runtime` 等上层。
- 不把 J7 夹爪伺服强行塞进六自由度 arm 的 FK、Jacobian、重力或 protective hold。
- 不把 MuJoCo ideal-effort 结果当作 RobStride 的绝对力矩、摩擦、负载或人手触感标定。
- 不自动改写源 URDF、YAML、固件参数或把 candidate overlay 直接激活。
- 不用 Harness 的离线/集成证据代替现场真机观察，也不在没有授权时启动 CAN、使能或运动。

## 2. 代码所有权和依赖边界

| 路径 | 唯一职责 | 不应放入的内容 |
| --- | --- | --- |
| `hardware/rebot_b601_rs/` | MotorBridge/SocketCAN、ID 映射、反馈、mode、enable/disable、逐关节 receipt、临时 `0x700B` torque-limit 事务 | policy、HITL blending、末端数学 |
| `robot_motion/rebot_b601_rs/` | Pinocchio 模型、FK/Jacobian、gravity、MIT torque composition、末端 wrench/pose spring、torque/slew conditioning | CAN、vendor 初始化、设备状态 |
| `sim/mujoco/rebot_b601_rs/` | MuJoCo 场景和 ideal generalized-effort 后端 | 真机 I/O 和硬件结论 |
| `scripts/hardware/rebot_b601_rs_low_gain_probe.py` | 人工运行的低带宽 probe、启动状态机、保护保持、交互采集、离线 fit CLI | 第二套生产控制入口 |
| `tests/test_rebot_b601_low_gain_probe.py` | 纯 helper、状态转换、拟合和故障路径测试；不打开 CAN | 真机操作脚本 |
| `docs/hardware/rebot_b601_rs_hardware_commissioning.md` | 频率/阶段/命令级 commissioning 说明 | moving status 或未核实结论 |
| 本文 | 面向下一位开发者的范围、规则、事实和交接模板 | 替代代码或现场授权 |

`hardware/reBotArm_control_py` 是独立 Git submodule，不是 AgenticRobot 的普通 Python 包。
更新 vendor 时先固定 child commit/tree，再更新 parent gitlink；不得复制一份生成的 URDF/mesh
绕过 pin，也不得从 mutable `main` 分支猜版本。

系统级 HITL 仍采用两个独立通道：运动通道传递受限 human residual，触觉通道将末端
wrench 通过 `J.T @ wrench` 映射为 B601 力矩。B601 的低带宽 probe 只负责本地 arm 控制，
不成为第二个 policy 或远程 authority。

## 3. 坐标、关节和数据契约

- 使用右手系和 SI 单位：`X` 向前、`Y` 向左、`Z` 向上；角度/速度/加速度为
  rad/rad·s⁻¹/rad·s⁻²，力/力矩为 N/N·m。
- arm 顺序固定为 `joint1, joint2, joint3, joint4, joint5, joint6`；ID7 是夹爪，不在
  当前 arm backend 中。
- pose 是 `base_link_T_gripper_end`；Jacobian 是 `LOCAL_WORLD_ALIGNED`、在
  `gripper_end` 原点、行顺序为线速度 XYZ 再角速度 XYZ。
- 末端 wrench 必须先在正确 frame 和原点下表达。不同 frame 不能只重排 XYZ；需要完整
  的 adjoint wrench 变换。离线检查保持功率不变量
  `wrench.T @ twist == tau.T @ qvel`。
- `MotorBridge 0.4.9` 的 `MotorState` 没有 vendor 源时间戳或接收序号；代码报告
  `freshness_verified=false`，`mechPos (0x7019)` 是显式读取/host 接收观察，不得把
  缓存值伪装为已验证的源 freshness。
- 每个完整 MIT/POS_VEL 发送都返回逐关节 receipt；任一关节失败或部分发送不能报告为
  六轴成功。`safe_stop()` 也是逐关节 receipt，`close()` 不隐式失能。

## 4. 控制数学与 torque owner

正常软件请求力矩为：

```text
tau_request = gravity_scale * g(q)
            + Kp * (q_ref - q)
            + Kd * (dq_ref - dq)
            + J(q).T @ wrench_overlay
```

然后按以下顺序处理：

1. 每关节绝对上限取 pinned URDF effort ceiling 乘现场指定的
   `--torque-limit-scale`。
2. 再按上一帧和 `--torque-rise-time-sec` 形成的 torque slew envelope 限制变化率。
3. 最后调用 upstream 同序的
   `send_mit(position, velocity, kp, kd, feedforward_tau)`。

重要 owner 规则：

- 发给 vendor 的 MIT `kp` 始终为零，避免 vendor 位置环和软件弹簧叠加。
- `pose-hold` 的位置恢复只能来自末端 6D spring 经完整 Jacobian 转置映射；软件关节 Kp
  应为零。`cartesian-hold` 同理，只产生平移 overlay。
- 软件 Kd 和 driver Kd 只能有一个实际阻尼 owner。`--driver-kd > 0` 时必须
  `--kd 0`、不使用非零 `--target-kd`，并且先完成六轴 `0x700B` 临时 torque-limit
  round-trip；driver Kd 只提供电机局部速度耗散，不提供位置锁。
- 未选择 driver Kd 时，软件 Kd 由 host 计算；20 Hz 反馈的有限差分量化可能产生阶梯感，
  这是连续性 commissioning 的观察项，不应通过叠加两个 Kd 解决。
- protective hold 去掉软件位置 Kp，以当前反馈作为每周期参考，保持可拖动；其重力在
  可信反馈下逐周期重算。
- degraded hold 使用“最后完整发送的 pose + gravity-only torque”缓存；反馈不可信时
  不伪造软件阻尼，已配置的 motor-local driver Kd 仍保持有效。这是当前明确的降级语义，
  因此要让故障保持仍有较大阻尼，现场档位应提前选择经过 round-trip 验证的 driver Kd。
- 这也是真实的当前限制：如果运行档位只有 host `--kd`、没有非零 `--driver-kd`，进入
  degraded cache 后不会继续用未经可信反馈支持的软件 Kd；此时能保证的是缓存重力，不是
  “重力+大阻尼”。下一轮若要补齐该目标，应单独设计可由缓存状态证明安全的降级阻尼并加
  针对性测试，不能在文档中把现有 gravity-only fallback 写成已经满足大阻尼。

## 5. 启动状态机和启动下坠诊断

```text
feedback warmup
  -> preflight safe_stop
  -> optional 0x700B apply/readback
  -> MIT mode prepare
  -> [DRIVE] 3/2/1/ENABLE
  -> ENABLED_UNARMED
  -> LOCKING
  -> LOCKED
  -> GRAVITY_RAMP
  -> HANDOFF
  -> ACTIVE
```

| 状态 | 实际动作 | 诊断重点 |
| --- | --- | --- |
| `ENABLED_UNARMED` | 已使能但尚未交给最终控制器 | post-enable q、qvel、ID 和 receipt |
| `LOCKING` | 用当前使能后 q 建立完整六轴临时参考；软件 startup Kp/Kd 来自 pinned YAML MIT 参数，vendor Kp 仍为零 | 首锁帧是否完整、是否出现方向反转 |
| `LOCKED` | 连续默认 4 个 feedback/control 样本重复发送当前锁参考 | qvel、dt、位置限位、torque slew |
| `GRAVITY_RAMP` | 语义上进入重力阶段；完整 `gravity_scale*g(q)` 从首个控制帧请求，不能再用第二个标量把重力削半 | 若仍下坠，先看模型重力、torque limit、slew、driver Kd 和 q 方向 |
| `HANDOFF` | 同一时长把临时 joint lock 平滑交给最终 controller；末端阻抗项与其他最终项一起交接 | startup lock 与 final controller 是否连续 |
| `ACTIVE` | 正常 `hold`/`slow-step`/Cartesian/pose 或 calibration 控制 | 逐周期 q、qvel、qacc、tau 和 receipt |

当前实现不把重力再独立衰减一次；`--torque-rise-time-sec` 是总力矩 slew 和 lock-to-final
handoff 的时间。若现场观察到刚使能轻微前倾，应从 JSON/log 对比 `gravity_target_tau_nm`、
`startup_lock_*_tau_nm`、`conditioned_tau_nm`、`slew_limited` 和 `gravity_scale`：

- `gravity_scale < 1` 或 URDF effort ceiling/scale 不足会产生真实重力残差；
- slew 在首帧限制的是总力矩，不是把某一关节单独压低；
- q 符号、base gravity 方向、payload/末端质量和关节零位误差会使某些姿态残差偏向同一侧；
- 软件/driver 双 Kd、静摩擦和 20 Hz 读数阶梯会把“能移动”误认为“已平衡”；
- 仅凭静摩擦把 arm 卡住不是重力补偿通过，必须在多个姿态、松手和小扰动下复核。

## 6. 正常模式和适用范围

| 模式 | 全部参与关节 | 参考/效果 | 备注 |
| --- | --- | --- | --- |
| `hold` | J1–J6 | 当前 post-enable q 的关节位置保持 + gravity + 单一 damping | 用于基础启动/静态诊断，不等价于末端阻抗 |
| `slow-step` | J1–J6 | 只改变指定目标关节的参考，其余仍保持；完整六轴 dispatch | 仅低幅方向和限幅检查 |
| `cartesian-hold` | J1–J6 | 当前 FK 平移锚点，平移 spring 经 `J.T` 共同调节 | 不控制姿态 |
| `pose-hold` | J1–J6 | 当前 FK 的完整 SE(3) 锚点，平移+旋转 wrench 经 `J.T` 共同调节 | 正式末端阻抗首测；软件 joint Kp=0 |
| `gravity-calibration` | J1–J6 | 可拖动 gravity+damping；R 后暂时固定 q 记录静态位置残差 | 多窗口无需重启 |
| `feedback-stability` | 不改变既有状态 | read-only 读取 `mechPos` 并统计有限差分 | 不切 mode、不使能、不失能、不发送控制帧 |
| `driver-torque-limit-roundtrip` | J1–J6 | 读原值→写 scaled effort→回读→恢复→回读 | 不使能、不发 MIT/POS_VEL；成功后才可在 active 命令使用 driver Kd |
| `gravity-calibration-fit` | 无设备 | 离线归约、配对、回归和 holdout 报告 | 只输出 candidate overlay |

`--joint` 只影响 `slow-step` 目标或单关节 gain override，不缩小 active enable/dispatch 集合。

## 7. Protective hold、degraded hold 和最终失能

### 7.1 设计原则

对承重机械臂，“停止程序”和“立即去使能”不是同义词。只要仍能发出完整六轴 MIT
gravity+damping 帧，就应尽力保持负载；不允许因普通超时、一次反馈异常或首个 Ctrl-C
把部分关节先失能而形成不对称负载路径。

```text
first Ctrl-C / ordinary handled fault / duration complete
    -> admit a verified current or last-safe snapshot
    -> all-six-joint gravity+damping protective hold (draggable)
    -> if live sample fails: cached gravity degraded hold + one retry
    -> only explicit second Ctrl-C/SIGUSR1, or live+cached full-frame failure
       -> full six-joint safe_stop
```

### 7.2 首次 Ctrl-C

- J1–J6 保持 enabled；J7 不在本 backend。
- protective hold 的软件 Kp=0，参考是每周期当前反馈，不是强行锁死在 Ctrl-C 瞬间位置；
  人可在大阻尼和重力补偿下拖动。
- 进入 hold 的首帧优先使用新鲜、合法、近似静止的 `last_safe`；若它过期但当前完整
  fault snapshot 合法，则使用当前 snapshot；若 live 读取已失效，则使用最后完整命令缓存。
- 第一次 Ctrl-C 的程序结果常为 `status=FAIL, stop_reason=operator_interrupt`，这表示
  试验被操作者中断，不等价于 hold 逻辑失败。必须同时查看
  `protective_hold.latched`、首帧 receipt、`degraded` 和 cleanup receipt。

### 7.3 普通异常和可恢复降级

以下情况只要已有合法 pose/gravity cache 且完整六轴发送仍可用，先进入或继续 degraded
protective hold：

- feedback 缺失、`valid=false`、stale、vendor status 异常或读数非有限；
- 当前位置超出模型边界但缓存仍为合法位置；
- feedback dt 不前进、`dt<=0`、qvel/qacc 非有限，或 qvel/qacc 超过当前 commissioning gate；
- FK、Jacobian、gravity、阻抗、功率一致性或总力矩计算异常；
- 单次 MIT frame 返回失败、部分关节发送或驱动器暂时拒绝命令；
- `duration-sec` 到期、校准窗口未完成、普通控制循环异常；
- 保护保持中的一次 live sample/compute 异常。

降级行为是：

1. 记录 `degraded.reason_counts`、`last_reason` 和进入次数；
2. 用最后完整发送的 q 和 gravity-only torque 组帧；不把错误样本刷新为 `last_safe`；
3. 完整六轴帧立即重试一次；成功后继续保持并等待 live feedback/control 恢复；
4. live 恢复后重新计算 gravity，记录 `recovery_count`，再回到正常 hold。

### 7.4 允许全臂失能的极端条件

全臂 `safe_stop()` 仅在以下事实成立时允许：

1. 操作者第二次 Ctrl-C 或发送 `SIGUSR1`，明确要求最终停止；
2. 没有任何通过位置、限位、年龄、速度和有限值检查的 pose/gravity cache；
3. live 完整六轴 dispatch 失败，且同一缓存完整六轴 dispatch 在一次立即重试后也失败；
4. protective-hold 首帧或后续实现本身抛出未能被 hold 内部处理的异常，无法证明下一帧
   仍可发送；
5. backend/驱动器拒绝或通信故障使“六轴完整发送”无法成立，而不是只坏一轴；
6. 现场物理急停、支撑或控制权状态已经不可确认，此时由人员按现场安全规程处置。

这些条件不是“更严格的常规测试阈值”，而是证明无法继续保持负载的最后分支。失能前后
必须记录 `full_disable_receipt.completed`、完成/失败关节列表和 `cleanup_errors`；若
receipt 不完整，不能把程序退出当作设备安全已确认。

### 7.5 保护保持中各故障的审查表

| 事件 | 首选处理 | 何时升级 |
| --- | --- | --- |
| feedback 缺失/无效/过期/vendor 异常 | degraded cached gravity + driver Kd（若已配置） | 无完整缓存或缓存帧重试仍失败 |
| 位置再次越过 URDF/有效边界 | 拒绝该样本，保留上一合法 cache | 无合法 cache 或无法完整发送 cache |
| timestamp 不前进、`dt<=0` | 不用该样本更新 q/qvel，进入降级 | 后续 cache dispatch 也失败 |
| qvel/qacc 非有限或超过 gate | 不接受为 last-safe，降级保持 | 无安全 cache 或发送失败 |
| FK/Jacobian/gravity/impedance/tau 异常 | 不发送异常计算结果，使用 gravity cache | cache 不存在/发送失败 |
| MIT 失败、部分发送、driver reject | 一次完整帧重试，再用 cache | live 与 cache 均不能完成六轴 |
| hold 首帧/后续未处理异常 | 视为 hold implementation failure | 立即 full safe_stop，并保留异常和 receipt |
| 第二次 Ctrl-C/SIGUSR1 | 明确最终停止 | 直接执行全六轴 `safe_stop` |

## 8. 反馈边界和数值异常处理

### 8.1 不改 URDF 的 tolerance + projection

URDF 的机械边界仍是权威物理/模型边界；反馈观测存在编码器量化、减速器回差和浮点误差，
因此只在进入运动学/guard 前做有界兼容投影：

- J2：`-0.006 <= raw_q < 0` 投影到 `0`；
- J3：`-0.003 <= raw_q < 0` 投影到 `0`；
- J4：运行 nominal boundary 使用精确 `[-pi/2, +pi/2]`，对
  `-pi/2-0.001 <= raw_q < -pi/2` 投影到 `-pi/2`；
- 超出上述窗口仍 fail-closed；不把任意大负值 clamp 成零；不改写源 URDF。

每个证据 JSON 同时保留 `raw_qpos_rad`、归一化后的 `qpos_rad` 和
`boundary_normalized_joint_names`。出站 reference 和 torque envelope 仍按有效 nominal limit
约束。J4 源 URDF 的 `-1.57/1.57` 是两位小数记录，不能把它误读成比 `±90°` 更窄的真实
机械边界。

### 8.2 运动 guard 的顺序

1. 先检查六轴反馈完整、状态有效、位置有限并通过边界投影/限位。
2. 用相邻 normalized q 和精确 host receive timestamp 计算 qvel。
3. 用连续 qvel 和同一 dt 计算 qacc；首个控制样本没有前一 qvel 时明确记录 unavailable。
4. 检查 `dt<=0`、非有限 qvel/qacc、qvel 越过 `--velocity-abort-rad-s`、qacc 越过
   `--acceleration-abort-rad-s2`。
5. 只有 guard 通过的样本才可计算 gravity/impedance、发送 MIT、更新 `last_safe`。

拒绝样本必须单独记录 offending joint、raw/normalized q、previous/current qvel、qacc、dt、
sequence/timestamp，并标记 `command_dispatched=false`、`eligible_for_last_safe=false`。
巨大的非有限/异常数值（例如曾经出现的 J4 超大读数）不能进入 FK、gravity 或 hold reference。

## 9. 参数政策：硬限制、协议域和现场配置要分开

这是本任务最容易被误解的地方。代码/测试不能把一个现场调参值伪装成“通用安全范围”。
当前只有一个 B601 开发者和一个现场控制 owner；不应为了假想的多开发者/多控制器场景
堆叠重复防御逻辑。唯一 CAN owner、完整六轴 dispatch、物理边界和失能收据是实际需要的
安全条件，其余调参保持在命令和现场记录中。

| 类别 | 来源/例子 | 处理规则 |
| --- | --- | --- |
| 真实物理/模型硬边界 | URDF joint position/effort；已确认的 vendor 电机额定边界；若审核确认末端速度 5 m/s、加速度 50 m/s²，则可在权威边界实现 | 必须有来源、单位和 owner；所有 profile 共享；越界 fail-closed |
| 数学/协议域 | 比例 scale 的有限性、`torque_limit_scale` 作为 `(0,1]` 的 effort fraction、非负 Kp/Kd、正持续时间 | 只防止无意义输入；不能借此偷偷收窄正常 commissioning |
| 现场 commissioning 配置 | `gravity-scale`、`torque-limit-scale`、Kp/Kd、driver Kd、slew 时间、force/torque/displacement、持续时间、qvel/qacc gate | 放在命令/profile/记录中，由硬件负责人按姿态、负载、支撑复核；不要写进常规测试断言 |
| protective-hold admission | last-safe age、stationarity velocity、fault-to-cache rebase | 是“能否用这帧作保护首帧”的门，不是 hold 激活后的拖动速度限制；异常不通过时优先尝试缓存 hold |
| 调试/证据阈值 | p50/p95/p99、重复窗口去重、拟合 pose tolerance | 只用于诊断或离线选择；不能改变出站控制边界 |

当前 `0.12 rad/s` 是根据已确认人工单轴扰动对正常 qvel gate 的 commissioning seed；
`4.0 rad/s²`、`0.02 rad/s` fault-hold stationarity、力矩/位姿/wrench 包络仍是独立门，
不能因一个门放宽而联动放宽。任何新数值先回答“它是物理上限、协议域还是本次现场配置”，
回答不了就不要进入代码或测试。

## 10. 正式重力补偿采集与拟合

### 10.1 为什么旧窗口不能直接拟合

此前一些窗口使用 `--kp 0`、没有 `P/N` 相反接近方向标签，或在窗口 5 尚未完成时被
`duration-sec` 结束。它们可以用于反馈漂移、启动和故障诊断，但静态位置残差不可观测，
也不能证明质量/质心。不要把“JSON status=OK”当成拟合输入合格。

### 10.2 现场采集协议

先做六轴 `driver-torque-limit-roundtrip`，确认 `apply_receipt`/`restore_receipt` 完整且
原值逐轴一致。正式采集使用：

```text
LOCKING → GRAVITY_RAMP → HANDOFF → ACTIVE
  ↘ FREE_DRAG
      P/N 标记接近方向 + R
      → SETTLING（默认 2 s，不计入拟合）
      → RECORDING（默认 20 s，完全不接触）
      → FREE_DRAG
```

每个窗口的现场动作：

1. 在 `FREE_DRAG` 由人拖到目标姿态；松手，确认没有接触、碰撞或持续下坠。
2. 从一侧接近时按 `P`，再按 `R`；从相反侧回到同一姿态时按 `N`，再按 `R`。按键无需
   Enter，`P/N` 是方向标签，不是关节方向或控制命令。
3. 看到 `CALIBRATION RECORDING #N` 后手离开机械臂，但手和物理急停保持在旁边可立即
   介入；不得在记录期间用手“帮它稳住”。
4. 看到 `CALIBRATION COMPLETE #N: returned to FREE_DRAG` 后再移动到下一姿态。
5. Ctrl-C 第一次仍是六轴 gravity+damping protective hold；第二次 Ctrl-C/SIGUSR1 才是
   final safe-stop。窗口中断会被标成 incomplete，不能静默算作完整样本。

推荐数据量不是“重复 100 个同类窗口”：

- training：6 个信息不同的姿态，各做正/负一对，共 12 个窗口；
- validation：3–4 个训练中未出现的新姿态，各做正/负一对，共 6–8 个窗口；
- 覆盖肘部、腕部和 J2 约 1.9–2.1 rad 的代表区域，但绝不为覆盖区域主动越过物理边界；
- 若回归矩阵秩、奇异值和姿态多样性已经足够，应停止追加同类窗口。

### 10.3 离线 fit 语义

`gravity-calibration-fit` 不连接 CAN、不切 mode、不使能、不失能：

1. 只提取完整 `RECORDING` 区间；每窗用 q、qvel、gravity 项和 position 项的中位数/MAD
   归约为一个姿态观测，不把 20 Hz 每帧当独立姿态。
2. 按姿态容差和显式 `positive/negative` 标签配对；无标签只能用
   `--calibration-allow-unpaired` 生成诊断 candidate，不能正式通过。
3. 默认先拟合 `global-scale-joint-bias`；只有独立验证显示明显姿态相关残差且确认存在
   未建模末端负载，才评估 `global-scale-payload`（非负质量 + 末端一阶矩）。
4. 用 rank、奇异值/条件数和 log-volume 选择信息量足够的 pose；保留独立 holdout。
5. 输出绑定 calibration dataset SHA、URDF SHA、config SHA、模型、配对数、rank、训练/验证
   残差，`activation` 必须为 `candidate_only_until_independent_validation`。

输出是有效重力 overlay，不是没有电流/力矩标定时的真实质量或质心测量。没有非零软件 Kp、
没有双向配对、窗口不完整、秩不足或验证残差不下降，都应保持 `CANDIDATE/FAIL`。

## 11. 真机操作和终端行为规范

### 11.1 运行前硬性前置

- 有可靠机械承托；末端不压桌、不接触本体或其他物体；运动空间清空。
- 物理急停可立即触达，并已明确“触发/恢复”流程；观察员在场。
- 同一 `can0` 上没有第二个 CAN 控制 owner；关闭官方 demo、遥操、数采和其他 servo。
- 记录 AgenticRobot commit/tree、vendor commit/tree、YAML SHA、固件、电机型号/ID、CAN
  bit-rate 和 payload/支撑状态。
- 命令在 controlling terminal 前台运行；禁止 `&`、`nohup`、daemon、吞掉 controlling
  TTY 的管道或自动发送按键。任何不可预期运动都先按物理急停，再保留日志和进程证据。

### 11.2 shell 不崩和输出分层

探针的详细 stdout/stderr 由 `--log` 写入文件，终端只保留：

```text
[B601] mode=... pid=... log=...
[B601] STOP: Ctrl-C ...
[DRIVE] 3 / 2 / 1 / ENABLE
[B601] STARTUP ...
[B601] PROTECTIVE HOLD ...
最终 JSON 摘要、结果路径、log 路径、shell remains active
```

包装命令不要使用 `set -e` 或 `set -euo pipefail`；所有失败都保存 `STATUS=$?`、打印结果
和日志路径，再让当前 shell 继续。不要为“看起来成功”吞掉非零退出码。

推荐的前台包装骨架：

以下代码是日志/TTY 包装骨架，不是可直接用于现场的控制参数；现场必须把经过评审的
`--mode` 和全部 profile 参数补齐，并再次核对 commit/tree。

```bash
WT=/absolute/path/to/pinned/AgenticRobot
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
CFG=/home/zc/HandMotion/Developing/reBotArm_control_py/config/rebotarm_rs.yaml
SRC=/home/zc/HandMotion/Developing/reBotArm_control_py
OUT_DIR=/tmp/b601-rs-commissioning/<run-id>
LOG="$OUT_DIR/command_and_output.log"
MODE=hold  # 现场替换为已评审模式，并补齐该模式所需的全部参数

mkdir -p "$OUT_DIR"
run_logged() {
  {
    printf '\n===== command start =====\n'
    date -Is
    printf 'cwd=%s\ncommand:' "$PWD"
    printf ' %q' "$@"
    printf '\n'
  } >>"$LOG"
  "$@"
  STATUS=$?
  printf 'exit_code=%s\n' "$STATUS" >>"$LOG"
  printf '[B601] probe status=%s\n' "$STATUS"
  return "$STATUS"
}

if cd "$WT"; then
  run_logged env PYTHONDONTWRITEBYTECODE=1 "$PY" \
    scripts/hardware/rebot_b601_rs_low_gain_probe.py \
    --mode "$MODE" \
    --config "$CFG" --rebot-source-root "$SRC" \
    --output "$OUT_DIR/result.json" --log "$OUT_DIR/result.log"
  STATUS=$?
else
  STATUS=86
  printf '[B601] ERROR: cannot enter worktree: %s\n' "$WT"
fi
printf '[B601] probe status=%s\n' "$STATUS"
printf '[B601] result=%s\n' "$OUT_DIR/result.json"
printf '[B601] log=%s\n' "$OUT_DIR/result.log"
printf '%s\n' '[B601] shell remains active'
```

`Ctrl-C` 第一次只请求 protective hold；第二次或 `SIGUSR1` 请求完整六轴失能。物理急停始终
是最高优先级；`safe_stop` 后仍需确认承重，不能用程序退出替代人员支撑。

## 12. Harness / Git 交接规则

### 12.1 共同原则

- 共享 worktree 先 inventory：分支、实际 HEAD/tree、dirty 状态、owner、声明 scope 和
  相关 peer；盘点不是锁，也不授予合并权。
- peer worktree 只读，禁止 reset、clean、stash、移动、删除或同步 peer 来制造干净状态。
- “ready” 必须绑定一个 clean、immutable 的完整 SHA、固定 base、实际验证命令、环境、
  已知 skips/未验证行为；任何新 commit、amend 或参与运行的未提交源码都会使旧 ready 失效。
- source code、offline/sim、remote、real hardware 证据分开呈现；沟通消息不等于 authority，
  旧 claim、旧 result、旧 hash 不能复用。

### 12.2 待合并（Harness/Task）

若本次工作由 Harness Task 管理，必须在 writer-capable canonical Harness 环境依次执行：

```text
inspect → conflicts → writer claim → renew while editing
    → scoped offline tests → diff-check → commit
    → recheck conflicts → release writer → handoff/quiesce
```

必须使用 Task 生成的 scope/lease/claim，不要再创建第二套 ownership JSON；不得直接改
`development`、不得自行 push/merge。人类真机验收必须绑定 exact claim、exact worktree、
exact commit/tree、observer、expiry、命令、输出根目录和通过条件。

### 12.3 合并后

合并完成后先重新核对 development 的实际 HEAD/tree、submodule gitlink、配置 SHA 和 clean
状态，再直接运行项目自己的 Python/pytest/field probe。Harness integration 只在仓库/发布
门禁明确要求时执行；不能因为历史上有 pre-merge Harness claim 就把同一段长 integration
test 当成每次合并后的真机必需步骤。合并后的直接 Python 运行仍须遵守第 11 节的现场安全
规则，且不能把 pre-merge claim 的证据路径当作 post-merge 证据。

### 12.4 当前分支元数据注意事项

本次核查发现分支描述仍记录旧 tip `292dbfa69dfa903455e60a825ebdfd2ab16605b9`，且状态
`awaiting-human-validation` 不被当前 inventory 识别。新增本文并提交后，必须把 branch
description 的 `tip`、`write-scope`（加入本文）、`state`、`validation` 更新为实际值；不要
把旧描述复制给下一位开发者。当前候选与依赖分支
`task/AR-20260906-b601-gravity-startup-protective-stop-r0` 在三条 B601 路径上有
inventory exact-overlap，这是因为候选有序地包含该依赖的已验证基线；它不是允许同时编辑
同一路径的授权。集成前仍须固定两个实际 SHA、检查 ancestry 和组合 diff，不能编辑或清理
依赖 worktree。

## 13. 可直接复用的验证命令

以下命令均为离线/只读，不能连接 CAN。

### 13.1 当前候选的 focused tests

```bash
WT=/home/zc/HandMotion/Developing/HarnessLocal/agentic-robot/Worktrees/AR-20260906-b601-gravity-startup-handoff-r1/AgenticRobot
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
cd "$WT"
bash scripts/run_pytest.sh tests/test_rebot_b601_low_gain_probe.py
git diff --check
git status --short --branch
```

当前核实结果：`93 passed`。测试只覆盖纯 helper、状态机、拟合和故障路径，不打开 CAN。

### 13.2 CLI/语法检查

```bash
WT=/home/zc/HandMotion/Developing/HarnessLocal/agentic-robot/Worktrees/AR-20260906-b601-gravity-startup-handoff-r1/AgenticRobot
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
cd "$WT"
env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py --help
env PYTHONDONTWRITEBYTECODE=1 "$PY" -m py_compile \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py
```

全 B601 套件可按项目 wrapper 运行；若出现 MuJoCo URDF 缺失，先检查
`git submodule status hardware/reBotArm_control_py`。当前已知历史结果是 `179 passed,
71 skipped, 3 failed`，3 个失败均源于任务 worktree 的 nested vendor submodule 未初始化，
不是代码行为通过；应记录为环境 blocker，不得改写成 PASS。

### 13.3 只读 feedback stability

```bash
WT=/home/zc/HandMotion/Developing/HarnessLocal/agentic-robot/Worktrees/AR-20260906-b601-gravity-startup-handoff-r1/AgenticRobot
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
CFG=/home/zc/HandMotion/Developing/reBotArm_control_py/config/rebotarm_rs.yaml
SRC=/home/zc/HandMotion/Developing/reBotArm_control_py
OUT_DIR=/tmp/b601-rs-commissioning/feedback-stability-<run-id>
mkdir -p "$OUT_DIR"
cd "$WT"
env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py \
  --mode feedback-stability \
  --config "$CFG" \
  --rebot-source-root "$SRC" \
  --duration-sec 20 \
  --rate-hz 20 \
  --fault-hold-max-velocity-rad-s 0.02 \
  --confirmation B601_RS_FEEDBACK_STABILITY_READ_ONLY \
  --output "$OUT_DIR/feedback.json" \
  --log "$OUT_DIR/feedback.log"
STATUS=$?
printf '[B601] feedback status=%s\n' "$STATUS"
printf '%s\n' '[B601] shell remains active'
```

`status=OK`/`acceptance_status=CAPTURED` 只表示采样完成；还要看每关节 p95/p99、漂移和
`stationarity_exceed_count`。不要仅因采样噪声提高 active probe 的 fault-hold gate。

### 13.4 正式采集（真机，现场批准后执行）

先完成 round-trip，再运行下面的两批命令；训练和验证只改变输出文件，不改变控制参数。

```bash
WT=/home/zc/HandMotion/Developing/HarnessLocal/agentic-robot/Worktrees/AR-20260906-b601-gravity-startup-handoff-r1/AgenticRobot
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
CFG=/home/zc/HandMotion/Developing/reBotArm_control_py/config/rebotarm_rs.yaml
SRC=/home/zc/HandMotion/Developing/reBotArm_control_py
OUT_DIR=/tmp/b601-rs-gravity-calibration-<run-id>
mkdir -p "$OUT_DIR"
cd "$WT"

env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py \
  --mode gravity-calibration \
  --config "$CFG" \
  --rebot-source-root "$SRC" \
  --duration-sec 900 \
  --warmup-sec 2 \
  --freeze-sec 1 \
  --rate-hz 20 \
  --calibration-settling-sec 2 \
  --calibration-duration-sec 20 \
  --kp 5.0 \
  --kd 1.0 \
  --gravity-scale 1.0 \
  --torque-limit-scale 0.40 \
  --apply-driver-torque-limits \
  --torque-rise-time-sec 0.5 \
  --velocity-abort-rad-s 5 \
  --acceleration-abort-rad-s2 40 \
  --max-error-rad 2 \
  --allow-motion \
  --auto-disable \
  --confirmation B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING \
  --fault-hold-max-age-sec 0.15 \
  --fault-hold-max-velocity-rad-s 0.02 \
  --fault-hold-max-rebase-rad 0.05 \
  --output "$OUT_DIR/train.json" \
  --log "$OUT_DIR/train.log"
STATUS=$?
printf '[B601] calibration status=%s\n' "$STATUS"
printf '[B601] shell remains active\n'
```

验证批次使用同一组参数并输出 `validation.json`/`validation.log`；不得复制训练姿态。
记录期间终端提示 `FREE_DRAG`、`CALIBRATION RECORDING`、`COMPLETE`，人手只在拖动阶段接触。

### 13.5 离线 candidate fit

```bash
cd "$WT"
env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py \
  --mode gravity-calibration-fit \
  --config "$CFG" \
  --rebot-source-root "$SRC" \
  --calibration-input "$OUT_DIR/train.json" \
  --calibration-validation-input "$OUT_DIR/validation.json" \
  --calibration-model global-scale-joint-bias \
  --calibration-pose-tolerance-rad 0.08 \
  --calibration-max-poses 12 \
  --calibration-max-velocity-rad-s 0.03 \
  --calibration-max-position-span-rad 0.02 \
  --calibration-overlay-output "$OUT_DIR/gravity_overlay_candidate.json" \
  --calibration-report-output "$OUT_DIR/gravity_fit_report.json" \
  --output "$OUT_DIR/gravity_fit.json" \
  --log "$OUT_DIR/gravity_fit.log"
STATUS=$?
printf '[B601] fit status=%s\n' "$STATUS"
printf '[B601] shell remains active\n'
```

fit 完成后必须检查：完整窗口数/不完整窗口数、方向配对数、rank/奇异值、训练和验证残差、
饱和计数、dataset/URDF/config SHA 以及 `activation=candidate_only_until_independent_validation`。
未满足条件时保留报告，不改 URDF/YAML，不把 candidate overlay 作为运行时参数自动加载。

## 14. 历史真机证据如何解读

历史结果必须区分“现场观察”和“机器可重放验收”：

- `feedback-stability` 曾稳定在约 20 Hz，J2/J3 零位附近的微小负读数通过边界层投影；这
  不是高带宽触觉闭环证明。
- no-touch qpos/Cartesian/pose hold 曾完成完整六轴发送和 cleanup；它证明低带宽链路和
  终止路径可运行，不证明真实力矩/摩擦已标定。
- 2026-09-04/06 的单轴人工扰动在旧 `0.03 rad/s` gate 下出现 `0.098028 rad/s`，原始
  JSON 应保留 `FAIL`；操作者另行确认“人工单轴扰动现象已验收”。后续正常 qvel seed 使用
  `0.12 rad/s`，不联动放宽 qacc 或 fault-hold stationarity。
- J2/J3 边界观察和 J4 精确 `±pi/2`/0.001 rad 兼容窗口只作用于反馈接纳；曾出现的 J4
  巨大非有限/异常数值必须 fail-closed，不能通过扩大 clamp 消除。
- 交互采集曾完成 1–4 号窗口，后续窗口 5 在 duration 边界前未完成，或旧版本因 J4 四舍
  五入边界中止；这类数据可诊断但不能直接作为正式重力拟合输入。
- protective hold 的人工验收重点是：首次 Ctrl-C 后 J1–J6 仍 enabled、gravity+damping
  可拖动；再次 Ctrl-C/SIGUSR1 才完成六轴 disable。程序因为操作者中断而返回 FAIL 时，
  仍须阅读 hold/cleanup 字段再判定安全行为。

建议保留的原始输出根目录示例：

```text
/tmp/b601-rs-commissioning/
/tmp/b601-rs-gravity-calibration-<run-id>/
/tmp/b601-rs-degraded-protective-hold-r1/
```

不要把临时 JSON/log 写回源码 worktree，也不要删除失败证据来制造“全绿”。

## 15. 结果呈现模板

交接/验收消息按以下顺序呈现，便于下一位开发者复核：

```text
结论：OFFLINE_PASS / HARDWARE_PASS / FAIL / NOT_RUN
阶段：<stage>
操作者/观察员/时间：<...>
AgenticRobot：HEAD=<full sha> TREE=<full sha> clean=<yes/no>
vendor：commit/tree=<...> config_sha256=<...> firmware/model/IDs=<...>
命令：<完整可复制命令>
输出：json=<absolute path> log=<absolute path>
控制：mode=<...> active_joint_names=joint1..joint6 vendor_kp=0
启动：state_history=<...> arming_settle=<...> handoff=<...>
保护：first Ctrl-C=<hold/degraded> second Ctrl-C/SIGUSR1=<safe_stop>
频率：feedback/compute/tx 的 count、observed Hz、p95/p99、deadline miss
运动：q/qvel/qacc、gravity/damping/overlay/final tau、限幅计数
异常：stop_reason、rejected_sample、degraded.reason_counts、escalation_reason
清理：cleanup_receipt.completed、full_disable_receipt、restore receipt、cleanup_errors
人工观察：方向、下坠、前倾、振荡、J6 回中、阻尼手感
未验证/阻塞：<明确列出，不用“看起来正常”替代>
```

机器字段与人工观察必须分别写出：`hardware_acceptance=NOT_CLAIMED`、离线 PASS 不等于
真机 PASS、`status=OK` 不等于重力模型已标定。

## 16. 下一位接手者的最短路线

```text
1. 重新 inventory 所有 B601 worktree，修正 branch description 的 stale tip/state
2. 固定候选 HEAD/tree、vendor commit/tree、YAML SHA，确认 clean
3. 运行 focused low-gain tests + diff-check + CLI/py_compile
4. 在唯一 CAN owner、承托和急停条件下做 driver torque-limit round-trip
5. 运行 read-only feedback-stability，先复核边界、qvel/qacc 和日志
6. 运行 training/validation 两批交互式 gravity-calibration
7. 运行 gravity-calibration-fit，只产出 candidate overlay/report
8. 独立 review 训练/验证残差和启动下坠记录，再决定是否另立接入任务
9. 如需合并，按 exact commit/tree 的 Harness/Git handoff 交接；合并后重新验证
10. overlay 接入必须有单独批准、回滚文件和再次的 no-touch/多姿态真机验收
```

关联的细节文档：

- [`rebot_b601_rs_hardware_commissioning.md`](rebot_b601_rs_hardware_commissioning.md)：阶段化
  commissioning、完整命令、频率字段和现场输出规范；
- [`rebot_b601_rs_haptic_acceptance.md`](rebot_b601_rs_haptic_acceptance.md)：离线/MuJoCo 与
  H1–H8 真机验收边界；
- [`rebot_b601_hitl_topology.md`](../architecture/rebot_b601_hitl_topology.md)：vendor gitlink、
  Pinocchio 模型和上层 HITL ownership 边界。

任何阶段遇到不可解释运动、方向反转、持续下坠、振荡、第二个 CAN owner、日志/版本身份
缺失或 cleanup 不完整，立即停止该阶段并保留原始证据；不要靠重复运行或临时放宽阈值掩盖
根因。
