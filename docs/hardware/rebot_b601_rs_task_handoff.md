# B601-RS 硬件开发交接说明

## 当前硬件开发线（2026-10-18）

- 后续候选以本地 `development@6e62032688b1c366ce87c83092758b0adfa41bcb` 为基线，
  工作区 `Developing/worktrees/b601-hardware-tuning-20261018`，分支
  `codex/b601-hardware-tuning-20261018`。下方旧版路径／状态是历史记录，不是新启动入口。
- 旧工作区 `Developing/worktrees/root-b601-staged-assist-20260923` 的 31 项未提交内容
  已完整保存在本地 `62959784f`，不删除旧分支或运行资料。新线仅移植实际缺失的硬件
  commissioning、GUI、标定 profile、轨迹 pacing 及只读碰撞审计访问器；不整支合并旧线。
- 最小共享接口为 `B601SharedControlRuntime.sample_policy(now_sec=...)`，独立提交
  `cf8dfd782`。它只读既有 command buffer 的有效性，不推进 servo 状态、不续期过期目标。
  迁移提交 `f9c98c9bb` 不包含 replay／RM65 应用改造。后续修复提交及验证结果随交付提供。
- 相对统一基线新增的 owner/probe 重力默认值沿用旧硬件线已确认组：全局 1，J2 scale
  0.96，bias `0 -0.22 -0.11 -0.08 0 0`，Coulomb 全零；不改共享控制器／仿真默认重力。
  本轮控制修复本身不调整这组数值、硬保护、碰撞模型或公开命令单位。
- 新基线只在本地完成合并；本硬件候选没有合入 `development` 或推送。
  基线中已接入的自观测过滤、夹爪等扩展保持原显式启用方式，默认 15 mm／v1／严格碰撞
  行为不随硬件迁移切换。物理验证仍由操作者运行，不把离线回归当成硬件有效性证明。

## 历史交接记录

> 版本：2026-09-22。本文是 B601-RS 当前代码、真机 commissioning、重力采集和
> Harness/Git 交接的统一入口。它描述可复核的实现和流程，不授予新的使能、运动或
> 合并权限。任何真机动作仍须由现场操作者在确认机械支撑和物理急停后前台执行。
> B601 Real2Sim、双向 ReBot/RM65 遥操、policy 传导和人工实时干预的长期阶段目标见
> [`../architecture/plans/rebot_b601_rm65_real2sim_hitl_roadmap.md`](../architecture/plans/rebot_b601_rm65_real2sim_hitl_roadmap.md)；
> 路线图不替代本文的当前候选和真机证据。

## 0. 先看结论

- 当前候选位于
  `/home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908`，
  分支为 `task/root/b601-gravity-calibration-runtime-20260908`。默认重力 profile 的代码
  checkpoint 为 `70fcb2a7ba690cbc733c586b11633738b9f1b8b1`；本文的最终交接提交和 tree 以分支
  description、交接消息及现场 `git rev-parse` 为准，不能在被提交的文档中自引用其 SHA。
- 候选尚未合并到 `AgenticRobot-development-mainline`，也未 push、未安装。截至本次交接，
  development 为 `028b6d28fb9056843959cceb074af8f0a6b3898a`，不包含上述候选；共同祖先为
  `ef54a901bb5df92d9926c7c47ef7def5c1ddfdf5`。development 在本候选六个声明路径上没有
  自共同祖先以来的改动，三方 merge preview 未见文本冲突；这只是当前只读检查，不是
  合并授权或最终语义评审。
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
  连续 handoff、J2/J3/J4 编码器边界兼容、J4/J5/J6 反馈工作余量、qvel/qacc/非有限值
  guard、原始 RobStride 帧诊断、交互式多姿态残差采集、startup replay、离线 candidate
  fit，以及逐关节重力 scale/offset/bias/Coulomb 和 driver Kd 配置。
- 操作者已确认下列正常重力补偿 profile 的真机手感合理，因此它已成为 enabled gravity
  mode 的程序默认值：软件 `kp/kd=0/0`，driver Kd=`0 0 0 0 0 0`，全局
  scale=`1.0`，joint scale=`1 0.96 1 1 1 1`，offset 全零，bias=
  `0 -0.22 -0.11 -0.08 0 0 N.m`，Coulomb 全零，速度阈值为 `0.02 rad/s`。
- 当前状态是“独立候选已准备交接”，不是“已合并/已安装/最终硬件通过”。逐关节 offset
  和 Coulomb 仍为零；没有独立力矩传感器标定，也没有高带宽触觉或系统级 `HARDWARE_PASS`。

本次默认值变更限定在 B601 硬件 owner/probe 及其 protective/degraded hold；已有
MuJoCo `B601Real2SimControllerProfile` 保持独立的仿真契约，避免未经单独验收就改变
既有仿真动力学和回归结果。

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
tau_request[i] = gravity_scale * gravity_joint_scales[i] * g_i(q + offset)
               + gravity_joint_bias_nm[i]
               + gravity_coulomb_friction_nm[i] * direction(dq_i)
               + Kp_i * (q_ref_i - q_i)
               + Kd_i * (dq_ref_i - dq_i)
               + (J(q).T @ wrench_overlay)_i
```

然后按以下顺序处理：

1. 每关节绝对上限取 pinned URDF effort ceiling 乘现场指定的
   `--torque-limit-scale`。
2. 再按上一帧和 `--torque-rise-time-sec` 形成的 torque slew envelope 限制变化率。
3. 最后调用 upstream 同序的
   `send_mit(position, velocity, kp, kd, feedforward_tau)`。

`--gravity-joint-scales J1 J2 J3 J4 J5 J6` 的操作者真机验证默认值为
`1 0.96 1 1 1 1`。它不是 URDF 修改，也没有隐式标定激活。该向量必须一致用于启动锁、
ACTIVE、protective hold 和缓存 gravity-only degraded hold，并记录到结果 profile。
`--gravity-joint-offsets-rad`、`--gravity-joint-bias-nm`、
`--gravity-coulomb-friction-nm` 的默认值分别为全零、`0 -0.22 -0.11 -0.08 0 0` 和
全零。Coulomb 方向只在有限差分速度绝对值超过默认的
`--gravity-coulomb-velocity-threshold-rad-s=0.02 rad/s` 时取正或负，否则为零；这避免静止量化噪声
驱动完整摩擦力矩反复换向。所有项贯穿启动、ACTIVE、protective hold 和缓存补偿，并记录
在结果 profile；它们组成默认 runtime profile，但不修改 URDF、不隐式改写模型文件。

canonical `rebot_b601_rs_owner.py` 默认启用 `--gravity-debug-mode`；直接调用
low-gain probe 或其他模式若要保持参数可比性，需显式使用该开关，owner 可用
`--no-gravity-debug-mode` 恢复普通保护保持语义。该模式在
使能前冻结本次 run 的 gravity scale、六轴 scale/offset/bias/friction、Coulomb
阈值和 driver Kd；ACTIVE、protective hold、恢复后的 live hold 及 degraded cache
均复用该不可变快照。它只固定参数来源，不改变任何反馈新鲜度、运动/力矩/加速度
门禁、发送重试、二次 Ctrl-C/SIGUSR1 全失能或物理急停规则。普通模式不构造该快照，
因此既有严格故障语义保持不变。结果 JSON 的 `gravity_debug` 和
`protective_hold.gravity_parameter_source` 是验收时的证据字段。

在该显式调试模式中，保护保持也不会为冻结快照中 `driver Kd=0` 的关节补入默认
`protective_hold_kd`；`protective_hold.software_kd_nm_s_per_rad` 应与本次调试的
单一阻尼 owner 一致。普通模式的默认阻尼 fallback 保持不变。

### 单关节连续助力调试界面（待真机验收）

`rebot_b601_rs_owner.py --torque-console --torque-console-confirmation
B601_RS_TORQUE_CONSOLE --allow-motion` 在原单一 CAN owner 进程内启动仅绑定
`127.0.0.1` 的浏览器界面；网页只提交短时意图，不能直接调用 MotorBridge。
独立开发 worktree 尚未编译 MotorBridge 时，用 `--motorbridge-root` 显式指定
已有 Python bindings 和已构建 ABI 的同版本 checkout，不把二进制复制进源码树。
界面默认选 J3、助力上限 `0.30 Nm`、上升斜率 `0.15 Nm/s`、撤力斜率
`2.0 Nm/s`、意图超时 `0.35 s`，这些值均可通过 `--torque-console-*` 显式调整。
所选关节助力期间的有限差分速度另设默认 `0.05 rad/s` 保护阈值；越界会在本
周期撤销助力并进入原保护保持，网页同时显示速度和阈值。它不作用于未加力的
手动摆位，也不是主动制动，零阻尼时仍可能发生惯性滑行。
该模式要求软件 Kp/Kd 为零、冻结本次重力参数、启用驱动临时力矩上限，且不与
既有单次关节/笛卡尔助力叠加。必须在启动交接完成、六轴速度都落入静止阈值后
才能从当前实测位置开始一轮；网页失联撤去助力，超出所选关节/其他关节位移
保护则退回原 protective hold，仍须第二次 Ctrl-C/SIGUSR1 才全臂失能。

此模式自动开启 `--debug-dynamics-trace`，逐周期记录反馈位置、有限差分速度、
重力、请求/限幅后的助力与最终软件力矩、完整发送回执；网页的曲线只是短时
滚动视图。所谓“已接受发送的助力”仅由完整发送回执确认，不代表独立测得的
电机实际输出力矩。该界面尚无真机验收，不能据离线测试宣称力矩标定或触觉
性能通过。

显式自动标定 profile 的两个 GUI 模式共用同一个阻尼发送适配口：固定位置采集、
完整搬运流程及其子阶段都必须通过该口解析当前控制帧的 driver Kd，再由原六轴
MIT dispatch 发送并校验回执，不能因 GUI 包装器类型不同而退回零阻尼。开始标定
时，子控制器首帧的阻尼渐升最多使用一个正常控制周期；搬运或等待期间未执行的
时间不计作渐升额度。后续继续从已确认发送的 Kd 渐升，起步扫描的 J3 Kd 仍为零，
纯重力阶段及故障保持仍使用冻结的基础配置。这里校验的是软件发送命令，不是
电机实际阻尼力矩或独立测力结果。

`b601_j3_local_identification_v1.json` 显式启用
`motion_continuous_speed_regulation`，两个 GUI 模式的局部采集阶段都复用已有连续调速。
该选项把显式配置的绝对软制动上限作为连续调速终点，不再叠加第二份速度余量：参考速度
`0.0075 / 0.015 rad/s` 对应的减力起点为 `0.01625 / 0.026 rad/s`，终点仍为
`0.039 rad/s`（本轮批准从 `0.03` 放宽 30%）。两者之间保持原 TRACK、目标和有界维持状态，
按完整 PD 加维持助力减小同向推进量；真正软超速仍制动。本轮局部 profile 的跟踪和采集
到位容差为 `0.0065 rad`，原为 `0.005 rad`；配套命令的六轴原始硬速度保护为 `0.13 rad/s`，
原为 `0.10 rad/s`。普通默认值和其他采集 profile 不变。完整搬运/回程仍使用独立的
J3 `0.05 rad/s` 软制动上限，继承本轮 `0.039 rad/s` 支撑轴软门和六轴硬门。
起步扫描期间 J3 原有的独立 `0.15 rad/s` 原始速度上限不变，其他轴使用上述硬门；
带撤力支撑确认的显式采集路径也同步接纳最高 `0.13 rad/s` 的配置，普通量化调试的接纳上限不变。
反馈、回执、人工撤销、力矩、位移绝对上限及预算保护不变。连续减力不等于
已辨识摩擦，也不证明真机能完成全部运动段，仍须检查实际采集报告和模型数据质量。

连续模式进入到位稳定阶段时，有界维持项先投影到最近完整派发的附加助力，预留当前
同方向 PD 份额，不能恢复 TRACK 调速时未实际派发的内部维持量。速度误差减力继续作用于
完整 PD 加维持输出；不再因阶段名变为 BRAKE 而失效。到位输出以实际派发量为渐变起点，
快速制动速率只允许撤掉推进量或产生有可靠运动证据的反向制动，同方向增推仍受原跟踪
渐升速率限制。到位后仍须完成停稳、渐退、零助力派发确认和撤力后停稳，不把位置进入
容差直接当作稳定；原生零回执前不得反向。所有启用 `arrival_stabilization` 的目标到位路径中，
内侧 `arrival_margin_rad` 仍用于提前进入制动/接近带；制动后停稳驻留按配置的最终位置容差判定，
不再把内侧门限重复用作停稳门。量化位置在最终容差内相邻计数间变化时，按既有位置分辨率和
停稳速度累计驻留；越出最终容差、持续超速或六轴运动仍会清零驻留。以上是软件命令链修复，
不是测力标定结果。

重要 owner 规则：

- 最终 active、protective hold 和 degraded hold 的 vendor MIT `kp` 始终为零。只有边界明确的
  `LOCKING/LOCKED/HANDOFF` 与 startup-replay `RAMP_IN/RELOCKED/HANDOFF` 临时锁允许
  使用 YAML MIT Kp；该 Kp 在电机本地闭环执行，退出临时状态时连续回到零。若接纳反馈
  位于 nominal URDF 命令包络外，对应关节的临时 Kp 必须置零，只保留重力和本地 Kd，避免
  被夹紧的编码参考成为主动回拉命令。
- `pose-hold` 的位置恢复只能来自末端 6D spring 经完整 Jacobian 转置映射；软件关节 Kp
  应为零。`cartesian-hold` 同理，只产生平移 overlay。
- 软件 Kd 和 driver Kd 只能有一个实际阻尼 owner。临时锁期间 YAML Kd 由电机本地拥有，
  主机临时 Kd 为零；交接时本地 Kd 连续回到 active driver Kd。`--driver-kd` 的已验证
  默认值为 `0 0 0 0 0 0`，也接受一个六轴共用值；任一项非零时必须
  `--kd 0`、不使用非零 `--target-kd`，并且先完成六轴 `0x700B` 临时 torque-limit
  round-trip；driver Kd 只提供电机局部速度耗散，不提供位置锁。正常重力补偿的软件
  `kp/kd` 默认均为零；需要固定参考采样的 `gravity-calibration` 必须显式选择非零 `--kp`。
- 未选择 driver Kd 时，软件 Kd 由 host 计算；20 Hz 反馈的有限差分量化可能产生阶梯感，
  这是连续性 commissioning 的观察项，不应通过叠加两个 Kd 解决。
- active path 的单帧有限差分 qvel/qacc 越限先隔离，不下发、不更新控制历史或安全缓存；
  下一帧相对最后可信样本恢复时继续运行，连续两帧越限才转入 protective hold。该策略用于
  拒绝已观测到的 J4 单帧伪零值，不提高用户配置的运动阈值，也不放宽位置/力矩边界。
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
| `LOCKING` | 用当前使能后 q 建立完整六轴临时参考；pinned YAML MIT Kp/Kd 由电机本地闭环执行，主机临时 Kp/Kd 为零 | 首锁帧是否完整、driver total-torque limit 是否已回读、是否出现方向反转 |
| `LOCKED` | 连续默认 4 个 feedback/control 样本重复发送当前锁参考 | qvel、dt、位置限位、driver Kp/Kd 与前馈力矩证据 |
| `GRAVITY_RAMP` | 语义上进入重力阶段；完整 `gravity_scale*g(q)` 从首个控制帧请求，不能再用第二个标量把重力削半 | 若仍下坠，先看模型重力、torque limit、slew、driver Kd 和 q 方向 |
| `HANDOFF` | 同一时长把临时 joint lock 平滑交给最终 controller；末端阻抗项与其他最终项一起交接 | startup lock 与 final controller 是否连续 |
| `ACTIVE` | 正常 `hold`/`slow-step`/Cartesian/pose 或 calibration 控制 | 逐周期 q、qvel、qacc、tau 和 receipt |

当前实现不把重力再独立衰减一次；`--torque-rise-time-sec` 是主机前馈 torque slew 和
driver-local lock-to-final handoff 的时间。若现场观察到刚使能轻微前倾，应从 JSON/log 对比
`gravity_target_tau_nm`、`vendor_kp_nm_per_rad`、`vendor_kd_nm_s_per_rad`、
`conditioned_tau_nm`、`slew_limited` 和 `gravity_scale`：

- `gravity_scale < 1` 或 URDF effort ceiling/scale 不足会产生真实重力残差；
- slew 在首帧限制的是总力矩，不是把某一关节单独压低；
- q 符号、base gravity 方向、payload/末端质量和关节零位误差会使某些姿态残差偏向同一侧；
- 软件/driver 双 Kd、静摩擦和 20 Hz 读数阶梯会把“能移动”误认为“已平衡”；
- 仅凭静摩擦把 arm 卡住不是重力补偿通过，必须在多个姿态、松手和小扰动下复核。

新增的 `startup_diagnostics` 直接保存使能前、使能未 armed、首个锁帧、各锁定 settle
以及首次 ACTIVE 的位置差；逐周期 `samples[].vendor_status` 保存既有 MotorBridge
operation-status 缓存速度/力矩。两者都不增加 CAN 请求、不参与控制，也没有位移 pass/fail
阈值。配套 MotorBridge `robstride-startup` 模式只被动追加 type-1/type-2 原始帧到文件；不要在
现有同步 `0x7019` 控制循环内再为六轴逐周期追加 `0x701A/0x302C`，否则会显著压低控制节拍。

## 6. 正常模式和适用范围

| 模式 | 全部参与关节 | 参考/效果 | 备注 |
| --- | --- | --- | --- |
| `hold` | J1–J6 | 当前 post-enable q 的关节位置保持 + gravity + 单一 damping | 用于基础启动/静态诊断，不等价于末端阻抗 |
| `slow-step` | J1–J6 | 只改变指定目标关节的参考，其余仍保持；完整六轴 dispatch | 仅低幅方向和限幅检查 |
| `cartesian-hold` | J1–J6 | 当前 FK 平移锚点，平移 spring 经 `J.T` 共同调节 | 不控制姿态 |
| `pose-hold` | J1–J6 | 当前 FK 的完整 SE(3) 锚点，平移+旋转 wrench 经 `J.T` 共同调节 | 正式末端阻抗首测；软件 joint Kp=0 |
| `gravity-calibration` | J1–J6 | 可拖动 gravity+damping；R 后暂时固定 q 记录静态位置残差 | 多窗口无需重启 |
| `startup-replay` | J1–J6 | 首次自然稳定后按 R；同一次使能中平滑加锁、保持并再次交接 | 区分边界释放与非边界重力偏差；不自动判稳 |
| `feedback-stability` | 不改变既有状态 | read-only 读取 `mechPos` 并统计有限差分 | 不切 mode、不使能、不失能、不发送控制帧 |
| `driver-torque-limit-roundtrip` | J1–J6 | 读原值→写 scaled effort→回读→恢复→回读 | 不使能、不发 MIT/POS_VEL；成功后才可在 active 命令使用 driver Kd |
| `gravity-calibration-fit` | 无设备 | 离线归约、姿态去重、回归和 holdout 报告 | 只输出 candidate correction |

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
| 位置再次越过反馈工作包络 | 拒绝该样本，保留上一合法 cache | 无合法 cache 或无法完整发送 cache |
| timestamp 不前进、`dt<=0` | 不用该样本更新 q/qvel，进入降级 | 后续 cache dispatch 也失败 |
| qvel/qacc 非有限或超过 gate | 不接受为 last-safe，降级保持 | 无安全 cache 或发送失败 |
| FK/Jacobian/gravity/impedance/tau 异常 | 不发送异常计算结果，使用 gravity cache | cache 不存在/发送失败 |
| MIT 失败、部分发送、driver reject | 一次完整帧重试，再用 cache | live 与 cache 均不能完成六轴 |
| hold 首帧/后续未处理异常 | 视为 hold implementation failure | 立即 full safe_stop，并保留异常和 receipt |
| 第二次 Ctrl-C/SIGUSR1 | 明确最终停止 | 直接执行全六轴 `safe_stop` |

## 8. 反馈边界和数值异常处理

### 8.1 不改 URDF 的边界投影与反馈工作包络

URDF 继续定义 nominal 模型、规划和出站命令边界，但它不是这台实物在人工拖动下每个可达
角度的完整证明。反馈观测分为编码器边界兼容和真实工作余量两层：

- J2：`-0.010 <= raw_q < 0` 投影到 `0`；该范围覆盖两次零位只读捕获和
  2026-09-08 host-id 限定反馈下观察到的 `-0.007627 rad` 启动边界瞬态；
- J3：`-0.003 <= raw_q < 0` 投影到 `0`；
- J4：运行 nominal boundary 使用精确 `[-pi/2, +pi/2]`，对
  `-pi/2-0.001 <= raw_q < -pi/2` 投影到 `-pi/2`；
- J4/J5/J6 在各自 nominal URDF 上下界外另接纳 `0.1 rad` 的真实反馈工作余量；这些值
  不做投影，可进入 FK、Jacobian 和 `g(q)`；
- 超出上述窗口/余量仍 fail-closed；不把任意大值 clamp 成边界；不改写源 URDF。

每个证据 JSON 同时保留 `raw_qpos_rad`、归一化后的 `qpos_rad` 和
`boundary_normalized_joint_names`。出站 reference 和 torque envelope 仍按有效 nominal limit
约束；slow-step 不能借用读侧余量。临时锁之外的 active probe vendor Kp 固定为零，因此位于读侧余量
时，软件位置/重力计算保留真实姿态，而编码到 MIT 帧的位置 reference 夹回 nominal
范围。临时锁必须额外取消该关节的 vendor Kp，确保读侧余量不会变成 driver 位置命令；
实际取消项进入 startup diagnostics 和逐样本证据。J4 源 URDF 的 `-1.57/1.57` 是两位小数记录，
不能把它误读成比 `±90°` 更窄的真实机械边界。

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

此前一些窗口使用 `--kp 0`，或在窗口 5 尚未完成时被 `duration-sec` 结束。它们可以用于
反馈漂移、启动和故障诊断，但 `--kp 0` 时静态位置残差不可观测，也不能据此证明质量/质心。
不要把“JSON status=OK”当成拟合输入合格。

### 10.2 现场采集协议

先做六轴 `driver-torque-limit-roundtrip`，确认 `apply_receipt`/`restore_receipt` 完整且
原值逐轴一致。正式采集使用：

```text
LOCKING → GRAVITY_RAMP → HANDOFF → ACTIVE
  ↘ FREE_DRAG
      R 固定当前位置
      → SETTLING（默认 2 s，不计入拟合）
      → RECORDING（默认 20 s，完全不接触）
      → FREE_DRAG
```

每个窗口的现场动作：

1. 在 `FREE_DRAG` 由人拖到目标姿态；松手，确认没有接触、碰撞或持续下坠。
2. 按一次 `R`（无需 Enter）；程序立即以当前六关节位置作为固定参考。无需判断接近方向，
   也不需要为了配对而再次回到同一姿态。
3. 看到 `CALIBRATION RECORDING #N` 后手离开机械臂，但手和物理急停保持在旁边可立即
   介入；不得在记录期间用手“帮它稳住”。
4. 看到 `CALIBRATION COMPLETE #N: returned to FREE_DRAG` 后再移动到下一姿态。
5. Ctrl-C 第一次仍是六轴 gravity+damping protective hold；第二次 Ctrl-C/SIGUSR1 才是
   final safe-stop。窗口中断会被标成 incomplete，不能静默算作完整样本。

推荐数据量不是“重复 100 个同类窗口”：

- training：6–8 个信息不同的姿态，每个姿态一次 `R`；
- validation：3–4 个训练中未出现的新姿态，每个姿态一次 `R`；
- 覆盖肘部、腕部和 J2 约 1.9–2.1 rad 的代表区域，但绝不为覆盖区域主动越过物理边界；
- 若回归矩阵秩、奇异值和姿态多样性已经足够，应停止追加同类窗口。

### 10.3 离线 fit 语义

`gravity-calibration-fit` 不连接 CAN、不切 mode、不使能、不失能：

1. 只提取完整 `RECORDING` 区间；每窗用 q、qvel、gravity 项和 position 项的中位数/MAD
   归约为一个姿态观测，不把 20 Hz 每帧当独立姿态。
2. 每个完整的 `R` 窗口直接形成一个静态残差观测；姿态容差只用于识别并平均近似重复姿态，
   不再要求操作者提供方向标签。旧文件里的无方向标签窗口也按相同质量条件处理。
3. 第一轮正式拟合 `global-scale`，其 `effective_gravity_scale` 可直接用于后续显式
   `--gravity-scale` 复测；只有独立验证显示明显姿态相关残差且确认存在未建模末端负载，
   才评估 `global-scale-payload`（非负质量 + 末端一阶矩）。
4. 用 rank、奇异值/条件数和 log-volume 选择信息量足够的 pose；保留独立 holdout。
5. 输出绑定 calibration dataset SHA、URDF SHA、config SHA、模型、姿态数、rank、训练/验证
   残差，`activation` 必须为 `candidate_only_until_independent_validation`。

输出是有效重力修正，不是没有电流/力矩标定时的真实质量或质心测量。没有非零软件 Kp、
窗口不完整、秩不足或验证残差不下降，都应保持 `CANDIDATE/FAIL`。

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
========================================================================
[B601] !!! PROTECTIVE HOLD LATCHED / DEGRADED / RECOVERED !!!
[B601] ARM STATE / CONTROL / DAMPING / FULL DISABLE ...
========================================================================
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

当前独立分支 owner 为 `codex-root`，声明范围是 backend、probe、两份 focused test 和两份
B601 文档；排除 development merge、push、install 和 agent 真机动作。交接文档提交后，
branch description 必须更新为 exact tip、`ready-for-integration` 和本节验证结果。候选包含
此前多个 B601 安全修复 checkpoint，不能只摘取最后一个默认值提交而漏掉其依赖链。集成前
固定完整候选 tip、development tip 和 vendor/MotorBridge 输入；不得编辑、reset、clean、
stash、删除或移动任何同伴 worktree。

## 13. 可直接复用的验证命令

13.1/13.2 是离线检查；13.3 会只读连接 CAN 但不切 mode、不使能、不发控制帧；13.4 是
明确的真机运动采集，只有现场操作者批准后才能运行。

### 13.1 当前候选的 focused tests

```bash
cd /home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908
/home/zc/anaconda3/envs/dexteleop310/bin/python3.10 -m pytest -q \
  tests/test_rebot_b601_hardware_backend.py \
  tests/test_rebot_b601_low_gain_probe.py
/home/zc/anaconda3/envs/dexteleop310/bin/python3.10 -m ruff check \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py \
  tests/test_rebot_b601_low_gain_probe.py
git diff --check
git status --short --branch
```

当前核实结果：`167 passed`、Ruff PASS、diff-check PASS。测试不打开 CAN。

### 13.2 CLI/语法检查

```bash
WT=/home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
cd "$WT"
env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py --help
env PYTHONDONTWRITEBYTECODE=1 "$PY" -m py_compile \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py
```

全 B601 套件可按项目 wrapper 运行；若出现 MuJoCo URDF 缺失，先检查
`git submodule status hardware/reBotArm_control_py`。此前较宽候选运行结果是 `206 passed,
71 skipped, 3 failed`，3 个失败均源于任务 worktree 的 nested vendor submodule 未初始化，
不是本轮 exact tip 的全量 PASS；应记录为历史环境证据，不得改写成当前全绿。

### 13.3 只读 feedback stability

```bash
WT=/home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908
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
WT=/home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908
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
  --kd 0 \
  --driver-kd 1.0 \
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
  --calibration-model global-scale \
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

fit 完成后必须检查：完整窗口数/不完整窗口数、独立姿态数、rank/奇异值、训练和验证残差、
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
- J2/J3 边界观察和 J4 精确 `±pi/2`/0.001 rad 兼容窗口只作用于反馈接纳；J4/J5/J6
  另有各 `0.1 rad` 的真实反馈工作余量，但不放宽出站命令。曾出现的 J4 巨大非有限/异常
  数值必须 fail-closed，不能通过扩大 clamp 消除。
- 2026-09-08 type-17 调试运行完成 4 个有效窗口后，J6 从 `2.774` 连续移动到
  `3.232712 rad`，其中触发旧上界的 `3.142859 rad` 来自 status=0、device=6、host=0xFD、
  parameter=0x7019 的正常回复，不是单帧通信伪值；这组证据支持上述 J6 工作余量。
- 同一次 4 窗口数据每窗 399 帧、最大窗口速度 `0.0077 rad/s`、最大位置跨度
  `0.00058 rad`，且没有 torque/slew saturation，可保留为训练候选。`global-scale` 离线拟合
  只给出 `1.0 -> 0.9999686` 和约 `0.00052 N.m` 的原始 RMS 残差，且没有独立验证批次；
  因此它不支持现在修改运行时重力比例、joint bias 或 payload 参数。
- `R` 在操作者松手后把当时位置设为 reference；按键前已发生的轻微下坠，以及被静摩擦
  挡住而没有形成位置误差的重力残差，都不会出现在 `position_tau=Kp*(qref-q)` 中。因此这
  4 点不能解释启动/自由拖动阶段的轻微前倾，后者仍应通过启动轨迹或实际电流/力矩观测定位。
- 第二批 4 个全新姿态也各完成 20 秒有效窗口，作为独立验证集时与第一批最近姿态的 L2
  距离为 `1.09..1.95 rad`。`global-scale` 只得到 `1.0 -> 0.9999686459`：训练 RMS
  `0.5248 -> 0.5173 mNm`，验证 RMS `1.1275 -> 1.1020 mNm`，绝对改善仅
  `0.0255 mNm` 且方向是略微减小补偿。joint-bias 和 payload 模型在验证集上均变差，
  payload 质量落到 `0 kg` 边界。因此这些 overlay 只保留为离线诊断，不得接入真机；
  也不应继续采集大量同类静态窗口来试图解释启动前倾。
- 更早的交互采集曾完成 1–4 号窗口，后续窗口 5 在 duration 边界前未完成，或旧版本因 J4
  四舍五入边界中止；这类 Kp=0、窗口不完整或来源不清的旧数据只可诊断，不能与上面本次
  4 个有效 Kp=5 窗口混为同一正式输入。
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
控制：mode=<...> active_joint_names=joint1..joint6 active_vendor_kp=0 temporary_lock_vendor_kp=<...>
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

### 16.1 当前交接包

| 项目 | 固定内容 |
| --- | --- |
| AgenticRobot worktree | `/home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908` |
| branch | `task/root/b601-gravity-calibration-runtime-20260908` |
| 默认 profile 代码 checkpoint | `70fcb2a7ba690cbc733c586b11633738b9f1b8b1` |
| development（本次只读快照） | `028b6d28fb9056843959cceb074af8f0a6b3898a` |
| merge base | `ef54a901bb5df92d9926c7c47ef7def5c1ddfdf5` |
| MotorBridge candidate | `0c8980afeeaf74bbd6d7329027d0af4a998b77e9`，tree `b34e138748ba4d21113a2653ed53fcce6a602571` |
| MotorBridge ABI SHA-256 | `85466a8e78532fa8194e01e59d056b36eeaea1e68c8c290cdfcb78c24dc1497f` |
| reBotArm vendor | `5ba28acef46237eb6a7560658bbc43b06cf8a259`，tree `ddd34d2b400b3987d4875bd6a99bd29b74d39dac` |
| YAML SHA-256 | `be51e877050cc4716188fca3b4f2a41109682513c7715c71594066a133ba5cf9` |

当前 worktree 除历史保留的
`.playwright-cli/page-2026-09-08T10-30-40-740Z.yml` 外没有源码修改；该文件不属于 B601，
不得删除或提交来制造“全净”。交接文档提交后应以最新 `git rev-parse HEAD`、tree 和 branch
description 替换上表的代码 checkpoint 进行最终集成 pin。

### 16.2 本轮实现完成项

1. 六轴同步 enable/dispatch/receipt，J7 明确排除；部分发送不冒充完整成功。
2. 驱动器本地临时锁完成 `ENABLED_UNARMED → LOCKING → LOCKED → GRAVITY_RAMP →
   HANDOFF → ACTIVE`，完整重力从首个锁帧参与，总力矩只经过一层 slew。
3. 第一次 Ctrl-C、普通异常和 duration 完成优先进入 J1--J6 gravity+damping protective
   hold；反馈/模型暂时不可用时使用最后完整补偿缓存和既有 motor-local Kd。只有第二次
   Ctrl-C/SIGUSR1，或 live 与 cached 完整六轴发送均失败，才允许最终全失能。
4. J2/J3 零位负零窗口、J4 精确 `+/-pi/2` 与 J4--J6 `0.1 rad` 只读工作余量已经分离；
   出站命令仍使用 nominal URDF 包络。
5. 单帧有限但不可信的 qpos 跳变先隔离，下一帧恢复则继续；非有限值、连续异常和无法
   建立可信控制状态进入 protective/degraded 路径，不能靠放宽运动阈值掩盖坏帧。
6. MotorBridge `robstride-startup`/type-17 被动原始帧记录不增加 CAN 请求；host-id 相关
   ABI、参数 ID、raw float bits 和关节诊断用于定位偶发 J4 参数响应异常。
7. `gravity-calibration`、`startup-replay` 和离线 fit 均保留；静态 R 窗口不能分离重力
   bias 与静摩擦的限制已经记录。
8. 重力运行时支持六轴 `scale/offset/bias/Coulomb`，Coulomb 只在有限差分速度超过显式
   threshold 后按方向生效；所有项贯穿 startup、ACTIVE、protective hold 和缓存补偿。
9. `--driver-kd` 兼容一个共享值或六个独立 J1--J6 值；软件 Kd 与 driver Kd 不允许同时
   成为 damping owner。
10. 操作者确认的默认正常重力 profile 已写入程序；省略相应 CLI 时使用第 0 节列出的
    数值，结果 JSON 的 `profile` 记录最终生效向量。

相关 checkpoint 顺序：

```text
6dd00466 passive startup diagnostics
55b76fe6 continuous startup replay
97473f0e stable driver-local startup lock
f803708f reversible per-joint gravity scale
ffc2bbd2 gravity offset/bias/Coulomb runtime terms
77ab9834 independent per-joint driver Kd
70fcb2a7 operator-validated gravity defaults
```

### 16.3 验证与证据边界

- 当前候选的 B601 相关离线回归得到 `476 passed`（1 个既有 hppfcl deprecation
  warning）；Ruff、`py_compile` 和 `git diff --check` 通过。该结果不包含真机动作。
- `/tmp/b601-rs-j2-gravity-scale-095-20260909-173751/`：J2 scale `0.95` 的完整 startup
  replay 状态机通过，约 20 Hz，无 torque/slew limit；最终顶层 `FAIL/operator_interrupt`
  是第二次 Ctrl-C 的记账，不抹去 replay 子结果和完整 cleanup。
  JSON/runtime/raw SHA-256 分别为 `381f3be1829ae72a07982a1611762d77a653d97c2c67470a55604263edbb4d68`、
  `7e70afccaaaf3e5ca6da8bd1977f70848429bda230ae99eeb9693dedcfd21ba4`、
  `dc8027306559cfec7b91a5d266a0bf5bd3cf5db467873fcaa72554583a3d5224`。
- `/tmp/b601-rs-j2-bias-minus010-r1/result.json`：历史机器证据确认当时实际下发的旧 profile 为
  driver Kd `0.2 0.3 0.2 0.15 0.1 0.05`、joint scale `1 0.95 1 1 1 1`、bias
  `0 -0.105 -0.1 -0.05 0 0`，protective hold 和最终六轴 cleanup 完整。该次没有按 R，
  所以 `startup_replay.status=FAIL`；它不能冒充 replay 验收，也不能作为当前默认 profile
  的验收证据。当前默认 profile 是操作者在 2026-09-22 调整后确认的配置；其 exact
  handoff commit 的真机重放仍需单独取得证据。
  JSON/runtime/raw SHA-256 分别为 `55f2d0575350e9233ff062dddf4b256b082f8005e7429b355c04b7f0c4cfd582`、
  `92a190c930de0f596d80e0e3ad28d795719f82bc182664d0cc03606c6b587f9b`、
  `4d2802bcc33b15fdedaee88c2cc985a941ad35fdafc373ba4ba8d26e3146a796`。
- 默认值提交本身只改变“省略参数时取什么值”，尚未在 exact handoff commit 上做一次
  省略所有重力 profile 参数的等价真机重放。若集成前要求严格机器可重放证据，应补这一
  次短 no-touch 运行，而不是重新标定全部参数。
- 旧静态 calibration 窗口和官方 PD sweep 可作为诊断/先验，但不能证明本机 offset、
  Coulomb、MIT Kp 力矩比例或 payload 已被独立辨识。

### 16.4 仍未完成或不得扩张的结论

- 未实现自动双向低速 sweep 来分离 `k/bias/Coulomb`；当前 Coulomb 默认全零。
- 未对新增默认 profile 做多负载、多温度、末端工具变化和全工作空间定量验收。
- 未完成高带宽末端触觉、passivity、力传感器闭环或系统级 teleoperation 验收。
- 未修改 URDF/YAML/固件持久参数；不得把 runtime bias 反写成虚假的 link mass 或 joint
  zero。
- 未合并 development、未 push、未安装，也没有授权 agent 运行真机。

### 16.5 下一位接手者的最短路线

```text
1. 读取本文、branch description 和当前 HEAD/tree；保留无关 untracked 文件
2. 复跑 167 项 focused tests、Ruff、py_compile 和 diff-check
3. 只读确认 development 与候选的共同祖先、声明路径差异和 merge preview
4. 若用户要求严格 premerge 真机证据，在承托/急停/唯一 CAN owner 下用默认参数省略方式
   做一次短 startup-replay/no-touch 等价验证；不要重复大规模静态窗口采集
5. 获得明确合并授权后才创建隔离 integration composition；合并固定提交并复跑验证
6. 合并后可直接使用目标 Python 环境运行，不再套 Harness runtime；Harness 只用于尚未合并
   composition 的受控验收
7. 新 payload、摩擦辨识或 protective-hold 独立大阻尼属于下一任务，不能夹带进本次合并
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

## 17. 启动异常恢复候选（2026-09-10，离线验证，真机暂停）

本增量基于 `1de3579c457758d14c6a9b2b7dab3362a53232f4`，分支
`codex/b601-startup-hold-recovery`；原 gravity 标定和 real2sim 候选保持不变。
具体候选 tip 与交接状态以该分支描述为准。本节不是合并或真机通过声明。

使能完成但尚无完整重力发送缓存、也没有新鲜已接纳状态时，原清理逻辑可能直接
走失能。新增恢复步骤最多读取两次反馈：整个读取窗口不得超过现有
`fault-hold-max-age-sec`；六轴必须完整、有限、在反馈限位内，有限差分速度不得超过
`fault-hold-max-velocity-rad-s`，相对使能前位置的每轴偏移不得超过
`fault-hold-max-rebase-rad`。旧位置仅用于偏移比较，不被冒充为当前位置或已发送缓存。
通过后复用原六轴零位置刚度的重力/阻尼 protective hold。已有新鲜状态或完整发送缓存
不经过这次额外采样；原默认重力参数、阻尼参数、力矩上限及出站限位不变。

明确全失能请求在恢复读取前后检查，进入 hold 时再次检查，不因恢复而额外发送首个
保持帧。正常 duration 完成和第一次停止事件仍进入保持；**如果已经因 duration 或
异常进入保持，此后的第一次 Ctrl-C 就是最终全失能请求**，不是重新开始计数。

离线验证：`tests/test_rebot_b601_low_gain_probe.py` 与
`tests/test_rebot_b601_hardware_backend.py` 共 **228 passed**，Ruff、diff-check 和
local-resource audit 通过。故障注入覆盖使能返回时的第一次 Ctrl-C、首个 RX 失败后恢复、
持续 RX 失败、坏数/越界/不完整/过快/过大偏移/读取超时、显式最终停止。
调用链测试使用 fake backend，并替换无期限 hold 循环检查入口和清理顺序；已有 hold
循环测试独立覆盖 live/cache、完整发送失败和释放语义。不能据此宣称硬件托举效果通过。

### 仍阻止无支撑使能实验的边界

- 使能前预检仍有显式六轴 `safe_stop`；不得用新进程接管正在保持的机械臂。
- 如启动后持续无法得到可信反馈且从未成功发送重力帧，本增量仍不能构造可信的
  重力保持，保留现有失能兜底；没有偷偷取消该保护，也不会盲发旧 q 或假重力。
- 部分使能失败、模型持续失败、总线发送失败和进程/供电丢失不在“始终托住”的保证内。
- MotorBridge 0.4.9 缺少设备源时间戳：本地读取时限不是硬件 freshness_verified。

因此当前修复只消除了**反馈仍可恢复时不必要的启动失能**，没有兑现“任意启动故障
绝不失能”的绝对承诺。下一次使能实验必须先确认可靠机械支撑、现场急停和唯一 CAN
控制进程；若要求无支撑并覆盖持续失联，需要另行确定驱动器本地保持/制动方案及其
硬件证据，不能仅靠放宽软件阈值实现。真机仍由操作者前台执行，详细输出写 log/JSON，
不得用 timeout、后台运行或关闭终端代替显式最终停止。

## 18. 末端六轴人工采集引导候选（2026-09-10，离线验证）

`pose-hold` 新增可选的 `--guided-pose-axis-protocol`。它只在 startup handoff 完成、状态
进入 `ACTIVE` 后开始，在前台终端依次提示：静置基线、`+/-X`、`+/-Y`、`+/-Z`、
`+/-Rx`、`+/-Ry`、`+/-Rz`，且每个动作后都有明确的松手回中采集阶段。坐标约定固定为
base/world-aligned 的 X 前、Y 左、Z 上，旋转遵循右手定则。

该协议是**操作员驱动的采集标记层**：不会生成末端增量、关节轨迹或新的 `qref`，不会
改变现有 `J.T @ wrench`、重力、阻尼、力矩调理、启动倒计时和 protective hold 路径。
每个已发送控制样本增加 guide phase、phase index 和 phase elapsed；结果 JSON 另记录完整
schedule、transition events、开始/完成时间以及 `operator_only_no_automatic_reference`
语义。引导目标必须严格小于既有平移/旋转 abort envelope，运行时长必须覆盖 startup
handoff 与完整引导，否则使能前 fail closed。

默认操作目标为每个平移方向 `0.03 m`、每个旋转方向 `8 deg`；静置 `5 s`，每个动作
`6 s`，每次回中 `4 s`，完整采集 `125 s`。这些值只用于终端提示和证据标签，并不是
位置命令或对实际位移达成的声明。当前候选没有运行真机；真机采集仍须操作者前台执行，
观察异常立即第一次 Ctrl-C 进入重力/阻尼保持，确认安全后再次 Ctrl-C 全失能。

## 19. 交互式末端六轴人工采集候选（2026-09-10，离线验证）

`pose-hold` 的 `--interactive-pose-axis-protocol` 用操作员按键取代固定时间切换。startup
handoff 进入 `ACTIVE` 后，程序先保持六轴重力补偿和 driver-local 阻尼，但不启用末端
spring，也不保存启动位置为任务空间锚点。操作员可用任意合理时长把机械臂拖到适合测试
的位置；完全松手并静止后按 `A`，程序才用该周期的新鲜六轴反馈做 FK，并把当前位置保存
为阻抗锚点。

锚点采集要求 `max(abs(dq)) <= fault-hold-max-velocity-rad-s`；不满足时只记录拒绝事件并
停留在 `POSITIONING`，不会使用旧位置或错误 q。锚点成功后，十二个方向仍按 `+/-XYZ`、
`+/-RxRyRz` 顺序进行，每项使用不同按键隔离状态：`N` 开始本项、人工缓慢移动并短暂停留、
`D` 结束动作、松手等待回到锚点附近并静止后按 `R` 进入下一项。`R` 同样经过上述六轴
静止门禁；同一批次收到多个键最多只产生一个状态转换。

协议本身不生成自动位移、旋转或关节参考。交互期间 nominal `qref` 每周期与当前新鲜 qpos
共置，且 handoff 后 vendor/software joint Kp 均为零；锚点前的 task-space overlay 为零，
锚点后唯一恢复项仍是既有有界 `J.T @ wrench`。结果 JSON 保存锚点 qpos/SE(3)、每次状态
转换和拒绝、每样本状态及相位时间。完成最后一次静止 `R` 后正常控制自动退出到既有六轴
重力/阻尼 protective hold；此时下一次 Ctrl-C 或 SIGUSR1 才请求全失能。运行总时长只是
交互超时，不驱动阶段切换。

本增量不改变重力 profile、driver torque limit、速度/加速度 abort、位姿 displacement
envelope、启动锁、异常降级或二段式退出。离线 fake-backend/状态机测试不能代替真机手感、
方向或力标定证据；真机仍只允许操作者在前台执行。
