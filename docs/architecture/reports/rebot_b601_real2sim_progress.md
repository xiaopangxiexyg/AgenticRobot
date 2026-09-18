# B601 Real2Sim 开发进度

> 总目标与阶段门禁见 [总路线图](../plans/rebot_b601_rm65_real2sim_hitl_roadmap.md)。
> 本文持续记录批次、验证和缺口；进度变化不再写入总路线图。

## 1. M0/M1 首个增量：host ZOH 与局部阻尼（2026-09-09）

本批提供可运行的控制侧诊断切片，M0/M1 尚未整体完成。开发基线为路线图候选
`618d2a366517c478a5d6b879b2a4f5aa2d2c21d3`；核对时本地 development 为 `015d44b4`，
尚未包含该候选。后续集成须显式携带重力候选和路线图依赖，不将用户推测的合并状态作为证据。

- `sim/mujoco/rebot_b601_rs/real2sim.py` 提供不可变 controller profile，集中 gravity
  scale/offset/bias/Coulomb 和逐关节 driver Kd；不改 URDF、passive damping、friction 或 armature。
- `B601MujocoEffortDriver(real2sim_profile=...)` 在 host command cadence 计算软件 torque，
  tick 间 ZOH；局部 `-Kd*dq` 每 physics tick 更新，再共同经过既有 clip/slew/clip。
  未传 profile 时保持原有 physics-rate 软件 MIT 行为。显式提供的软件 PD profile 仍有效，
  默认软件 damping 在此 opt-in 模式为零，避免默认重复阻尼。
- gravity parity 继续比较原始 plant 重力；控制补偿只影响 controller。固定打开的仿真夹爪
  仍参与同一模型计算，尚未证明与真实夹爪姿态/负载等价。
- tick 区分 `host_torque_updated`、保持的 `torque_breakdown` 和
  `driver_damping_torque_nm`；后者加到 host requested torque 后才最终限幅。metadata 记录
  URDF SHA-256、参数来源、两个 cadence 和未建模项。
- wrong-frame/invalid reference 与 watchdog 撤销会立即清空 host cache。原有 stale-sequence
  拒绝且等待 watchdog 的语义保留；第 8 节最终契约仍须闭合。
- 本地阻尼以 physics cadence 计算是显式近似；总 torque 的 physics-rate slew 是现有仿真
  保护机制，不代表 RobStride 内部 slew。启动锁、反馈坏帧隔离、cache degraded hold 和六轴
  CAN partial-send 尚未仿真，不能称为真机 runtime 完整等价。

现有入口新增可选 `--real2sim-controller --host-hz`，不创建第二个生产入口。以下命令只运行
MuJoCo，前台输出完整 JSON，stdout/stderr 可按现场规范重定向到固定 log：

```bash
# 从本增量工作树运行；Python 使用已配置的项目环境。
python -m sim.mujoco.rebot_b601_rs.haptic_demo \
  --rebot-source-root hardware/reBotArm_control_py \
  --fixture pose-6d --real2sim-controller --host-hz 20
```

首次实跑此命令在 52 个 physics tick 后触发 J6 acceleration guard（约
468.57 rad/s²，既有 fixture 限值 250 rad/s²），返回 fixture FAIL；预期命令 30 次、实际 2 次。
这一失败没有通过放宽阈值处理。它说明真机选择的补偿/Kd 与原 ideal plant、现有 fixture gains
和 20 Hz host 组合尚不通过效果验收，不能推断为真实电机失控或宣称 Real2Sim 已通过。

相同 6D fixture 的离线对照如下。这里只改变 host cadence 和是否添加局部 Kd，没有改变
速度/加速度保护阈值，也没有改变 MuJoCo plant：

| host Hz | 局部 Kd | 结果 | physics ticks |
| --- | --- | --- | --- |
| 20 | 当前逐关节配置 | J6 acceleration guard | 52 |
| 500 | 当前逐关节配置 | fixture PASS | 1500 |
| 20 | 零 | J6 velocity guard | 26 |
| 500 | 零 | fixture PASS | 1500 |

这支持继续研究低 host cadence 下的控制带宽与执行器响应，不支持仅靠放宽阈值解决。
500 Hz 是仿真对照配置，不是已证实的真机 host 能力。

接下来先离线分解 host 延迟、局部 Kd、fixture gains 和 plant 缺失项的影响，再做启动/保护
状态机等价。物理辨识需要新增真机数据时暂停开发的相关路径，并提供现场前台采集命令；
当前无需执行任何真机命令。

本批验证：B601 real2sim、ideal MIT、MuJoCo haptics/position、纯 haptics、submodule 和
control-composition 边界共 103 passed；changed-file Ruff、CLI `--help`、资源 audit 和
`git diff --check` 通过。测试使用固定 vendor gitlink 与既有 dexteleop310 环境，没有修改环境。
下一批仍在 M1：补充 startup/protective-hold 的离线状态机等价；M2 物理参数辨识未开始。

## 2. M1 后续增量：前馈/局部反馈分层与仿真生命周期

基线：`6aa74d2b982a7e0ea86f7929e28f691422ac46eb`。该基线正在集成，本增量使用独立后续
分支，不改变前批 commit/tree。本次将第 1 节从总路线图完整迁到本文；以后批次进度只维护本文。

### 实现与复核

1. 对照真机 `build_startup_lock_profile`、`resolve_driver_local_lock_command` 和
   `build_protective_hold_profile` 后修正仿真 conditioning：host feedforward 在实际 host 更新
   dt 上 clip/slew/clip，局部 Kp/Kd 在 physics tick 更新并绕过 host slew，最后总 torque
   仍受 actuator/profile 绝对限制。第 1 节“总 torque 在 physics rate 做 slew”仅描述旧基线。
   未启用 Real2Sim 的 ideal MIT 分支保持原算法。
2. `B601SimStartupProfile` 要求显式六轴 lock Kp/Kd；freeze、handoff 时长和 settling 门禁为
   仿真配置，不宣称厂商默认值。reset 从当前合法 q 建立固定 anchor，外部 reference 不能改变它。
3. 原 driver 内推进 `ENABLED_UNARMED -> LOCKING -> LOCKED -> GRAVITY_RAMP -> HANDOFF ->
   ACTIVE`。局部位置 Kp 衰减到零，局部 Kd 混合到 active 配置，host PD/overlay 随 handoff
   接入；重力从第一帧完整参与。`GRAVITY_RAMP` 保留阶段名，但不再次从零缩放重力。
4. lock 位移超界、freeze 结束未静止、startup 中 invalid/wrong-frame/watchdog，以及显式
   `request_protective_hold(reason)` 均可锁存到 live-feedback gravity+damping。software Kp、
   driver position Kp 和 overlay 请求归零；driver Kd 非零关节不重复加软件阻尼，driver Kd
   为零的关节使用真机 helper 同样的 software damping seed 1.0。
5. hold 拒绝新运动 reference。`request_disable()` 清除六轴 actuator effort，随后 hold
   请求不能反向使能；只有显式 reset 创建新仿真 epoch。前馈的有界 slew 转换可能保留短暂
   旧 torque 分量，因此“撤销 reference”不等于瞬时跳变总 torque。
6. tick 增加 host-conditioned torque、driver position torque、phase/reason；现有 demo
   增加显式 startup Kp/Kd 参数及 phase timeline。只有最终 ACTIVE 且原数值/效果条件通过，
   才报告 fixture acceptance；处于 protective hold 不冒充成功。

### 仿真结果和限制

- 原始 URDF 平衡、补偿 scale=1/bias=0 的隔离测试覆盖完整 startup 到 ACTIVE、首帧重力、
  anchor、hold 和 disable；这不是默认真机 profile 的启动手感验收。
- 当前补偿 profile、500 Hz host、六轴 Kp=2、Kd=`0.2 0.3 0.2 0.15 0.1 0.05` 的 6D demo
  完成 1500 ticks；0.1 秒因 `startup_not_settled` 进入 PROTECTIVE_HOLD，无 driver 数值故障，
  fixture exit code=3。未放宽 settling/velocity/acceleration 门禁。
- LOCKING -> LOCKED 只模拟一次完整成功的 actuator 更新；没有 CAN partial-send、enable
  ACK、接收缓存、坏帧隔离或设备源 timestamp。cached/degraded hold 尚未实现。
- 非有限状态或数值/位置/速度/加速度 guard 仍采用现有仿真 fault latch：清 torque 并冻结
  受影响 plant；不使用非法状态继续 gravity hold。
- hold/disable 是模拟 API，不是信号处理器；没有改变真机 Ctrl-C/SIGUSR1 行为，仿真代码
  不导入硬件 probe。reset 仍用既有 gravity seed，完整真机零 torque arming 时序未建模。
- M1 尚未整体完成，M2 参数辨识未开始；下一批先补充反馈故障与降级模式的明确仿真契约。

### 前台复现（仅仿真）

本批验证：110 passed（B601 Real2Sim、ideal MIT、MuJoCo haptics/position、纯 haptics、
submodule 和 control-composition 边界）；changed-file Ruff、CLI help、文档 Bash 语法、
资源 audit 和 `git diff --check` 通过。未运行真机，未改动 Python 环境。

此命令预期重现上述 protective hold，exit code=3 表示 fixture 未通过。命令和完整 JSON
写入固定日志，Python 在前台运行，shell 保持存活。路径需随最终部署位置更新。

```bash
WT=/tmp/b601-real2sim-lifecycle
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python
LOG=/tmp/b601-real2sim-command-and-output.log
ARGS=( "$PY" -m sim.mujoco.rebot_b601_rs.haptic_demo
  --rebot-source-root hardware/reBotArm_control_py
  --fixture pose-6d --real2sim-controller --host-hz 500
  --startup-lock-kp 2 2 2 2 2 2
  --startup-lock-kd 0.2 0.3 0.2 0.15 0.1 0.05 )
if (
  cd "$WT" || exit 2
  printf 'command:'
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  PYTHONDONTWRITEBYTECODE=1 "${ARGS[@]}"
) >> "$LOG" 2>&1; then
  STATUS=0
else
  STATUS=$?
fi
printf 'simulation exit code: %s\nlog: %s\n' "$STATUS" "$LOG"
printf '%s\n' 'shell remains active'
```

## M1 后继修复：驱动器本地反馈的数值积分

基线 `a295cddca0323fad7c17fe1ca911300c39ad3bd9` 保持不变，后继在
`codex/b601-real2sim-local-servo` 开发。仅修改仿真驱动、测试和本进度页；不修改总路线图、
URDF、硬件控制或操作员标定 profile。待合并，未 push、安装或运行真机。

### 原因及修复

之前将本地 Kp/Kd 在 Python 中计算成外部 `ctrl` 力矩，`implicitfast` 无法将这个外部
速度反馈的导数纳入隐式积分。因此把官方 YAML 的启动增益直接代入会造成数值不稳定，
不能据此判定官方增益不适用，更不能通过放宽速度/加速度门限解决。

现在 real2sim 路径使用 MuJoCo 原生 affine actuator bias：
`tau = host_feedforward + Kp*(q_anchor-q) - Kd*dq`。主机前馈仍按原 host cadence
执行 clip/slew/clip；本地反馈不经过主机 slew。执行器 `forcerange` 限制三项总和，
不会突破 profile 或原始执行器范围，receipt 读取实际 `actuator_force`。
disable、数值故障和 reset 同时清除 ctrl 与 native bias，避免 ctrl=0 后仍残留制动/弹簧。
原 ideal-effort 分支不变，plant 被动阻尼仍为零，不重复计入驱动器阻尼。

依据：[MuJoCo actuator force model](https://mujoco.readthedocs.io/en/latest/computation.html)
和 [affine bias / implicit integration](https://mujoco.readthedocs.io/en/latest/XMLreference.html)。
这只是理想电机局部伺服的数值表达，不代表已辨识 RobStride 带宽、摩擦、延迟或量化。

### 离线对照与尚存问题

使用 pinned vendor `5ba28ace` YAML 的 Kp=`50 150 150 50 50 50`、
Kd=`3 10 10 5 4 4`，原有 1 kHz physics、操作员重力 profile 和相同保护阈值：

- 修复前、500 Hz host：J6 加速度约 524.97 rad/s²，保护停止，未完成启动。
- 修复后、500 Hz host：1.5 s 6D fixture PASS，约 0.302 s 进入 ACTIVE。
- 修复后、20 Hz host：完成启动锁，但交接阶段 0.35 s 因
  `startup_lock_displacement` 进入 PROTECTIVE_HOLD；无数值故障，fixture 仍不通过。
- 先前 Kp=2、弱 Kd 的 `startup_not_settled` 负例继续保留，不自动提升增益或放宽门限。

因此本修复解决了高本地增益的积分缺陷，不宣称 20 Hz 主机下的 6D 阻抗已通过。
下一步需隔离交接时序、低频 fixture 前馈和 plant/profile 不匹配的影响，再决定修复，
不能把仿真 500 Hz 通过当作真机 CAN 已达到该频率。真实动态辨识仍需暂停并由操作者采集。

仅仿真复现可使用本页前台命令模板，将 WT 改为 `/tmp/b601-real2sim-local-servo`，
启动 Kp/Kd 改为上述 YAML 六轴值；保留固定命令/输出 log。500 Hz 是仿真实验设置，
该命令不能移用于真机。旧候选仍可直接用于修复前对照。

本批验证：同上七个 B601/边界测试文件共 **115 passed**（30.09 s），Ruff、资源 audit、
`git diff --check` 通过。新增覆盖官方增益完成启动、disable/fault/reset 清除 native bias、
以及本地弹簧与主机前馈叠加后的实际总力矩限幅。未修改任何硬件阈值或环境包。

## M1 交接时间边界与低频闭环隔离

后继基线 `d3d266d5bf25ed141c39ee51d2653e4ea20793db`，工作分支
`codex/b601-real2sim-handoff-timing`；上一候选保持不可变。本批仅修复仿真状态机时间比较。

原 `elapsed = 0.3 - 0.1` 得到略小于 0.2 的浮点值，使 handoff fraction 略小于 1。
此时本地锁增益已经接近零，但 phase 仍停在 HANDOFF；20 Hz 下直到 0.35 s 才再次
判定，错误延长了启动锁检查期。现复用既有 command cadence 的 `1e-12 s` 数值容差，
freeze/ramp/handoff 在对应 host deadline 正确推进。位置、速度、加速度阈值完全不变；
完成交接的 tick 仍先检查启动位移，不能凭到时跳过保护。

### 20 Hz 对照（1 kHz physics，1.5 s，无真机）

采用上一节官方 YAML 启动增益。通过进程内测试替换，仅将 demo 的末端 torque overlay
置零或把控制补偿 scale/bias 还原；不修改 plant、不改变本地 Kd、不放宽保护。
“完成”只代表数值运行完成，overlay 关闭的对照不承担末端恢复效果验收。

| 控制补偿 | 末端 overlay | 运行完成 | 峰值速度 rad/s | 峰值加速度 rad/s² |
| --- | --- | --- | --- | --- |
| 操作员 profile | 关闭 | 是 | 0.35217 | 2.24652 |
| 操作员 profile | 开启 | 否：J6 acceleration guard | 1.33080 | 637.11025 |
| 原始重力 scale=1/bias=0 | 关闭 | 是 | 0.000213 | 0.52973 |
| 原始重力 scale=1/bias=0 | 开启 | 否：J6 acceleration guard | 1.36156 | 640.49287 |

四组均在 0.300 s 进入 ACTIVE。剩余失败发生在 ACTIVE，而非 startup lock：当前
fixture 的末端弹簧/阻尼在 host tick 计算，然后 ZOH 50 ms。偏置不是这次失稳的必要条件。
同时，操作员补偿在未辨识的 ideal plant 上明显漂移，也不能称作真实重力平衡的数字孪生。

下一步需分别验证低频末端反馈的离散控制稳定域，以及用实测数据辨识 plant；不能通过
把控制偏置反向写入 URDF 来“制造平衡”，也不能把 host 20 Hz 悄悄改成仿真 1 kHz
末端反馈并声称真机等价。目前 20 Hz 6D force overlay 仍为 NOT_ACCEPTED，停止于仿真；
没有新增真机运行命令，也没有调整真机参数。总路线图未修改。

本批验证：同上七个测试文件 **118 passed**（34.35 s），新增 20/500 Hz 准时交接和
deadline 上位移保护不被跳过的回归检查；Ruff、资源 audit、`git diff --check` 通过。
交付状态为待合并，依赖上一候选；未 merge、push、安装或触发硬件。

## M1 可复现实验参数与长时验证（2026-09-10）

基线 `99a211fc92f3a9ecd46ca4bada274bfd4504ec1c`，后继分支
`codex/b601-real2sim-gain-experiments`。现有 haptic demo 增加
`--fixture-stiffness-scale` 与 `--fixture-damping-scale`，默认都为 1。
倍率同时作用于平移/旋转对应项，不改变 motor-local Kd、标定补偿、力矩限幅或
速度/加速度/效果验收门限。JSON `fixture_contract` 记录倍率及实际增益。
两者需有限非负；虚拟墙保持原有“刚度/阻尼不能同时为零”的约束。
关闭 spring/pose 增益可用于无 overlay 对照，但不冒充效果 PASS。

固定 operator gravity profile、官方启动锁增益、1 kHz physics 和 pose-6d 初始扰动，
对照 1.5 s 的结果如下（倍率实验最初通过进程内替换构造参数进行，现在可由 CLI 复现）：

| host Hz | stiffness scale | damping scale | 结果 |
| --- | --- | --- | --- |
| 20 | 1 | 1 | J6 acceleration guard，约 637.11 rad/s² |
| 20 | 1 | 0.1 | J6 acceleration guard，约 442.11 rad/s² |
| 20 | 0.25 | 0.1 | J6 velocity guard，约 2.018 rad/s |
| 20 | 0.1 | 0.1 | J6 velocity guard，约 2.005 rad/s |
| 50 | 1 | 1 | J6 acceleration guard，约 332.04 rad/s² |
| 125 | 1 | 1 | J6 acceleration guard，约 517.94 rad/s² |
| 250 | 1 | 1 | J6 acceleration guard，约 254.86 rad/s² |
| 500 | 1 | 1 | PASS |

上述有限对照不是完整稳定域，更不能将“500 Hz 通过”解释为实机最低频率。
下一步低频控制实验应分析 sampled feedback 的能量与局部阻尼作用，不能仅继续放宽阈值。

### 长时钟误差修复

500 Hz / 10 s 原先在第 7981 个物理步报 `physics_clock_desynchronized`，当时运动已基本
静止。根因是 MuJoCo 累加 dt 的舍入误差超过固定容差，而调度器使用 tick*dt。
现用浮点求和误差界 `gamma_n * abs(time)`，并将容差硬封顶为四分之一个 physics tick；
不会接受外部多推进一个完整物理步。新增 10,000 步回归，随后主动外部 step 必须拒绝。
本修复不涉及真机 timestamp、CAN freshness 或 watchdog，也不放宽运动安全阈值。

修复后同一 500 Hz / 10 s 实验完整运行 10,000 ticks，simulation/effect PASS，未触发保护。
完整 JSON：开发机 `/tmp/b601-gains-500hz-10s-corrected.json`，该文件是本机运行证据，
不是 Git 资产。可使用下列仅仿真、前台命令重现；不启动 CAN、不发送真机命令：

```bash
WT=/tmp/b601-real2sim-gain-experiments
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python
LOG=/tmp/b601-real2sim-command-and-output.log
ARGS=( "$PY" -m sim.mujoco.rebot_b601_rs.haptic_demo
  --rebot-source-root hardware/reBotArm_control_py
  --fixture pose-6d --real2sim-controller --host-hz 500 --duration-sec 10
  --fixture-stiffness-scale 1 --fixture-damping-scale 1
  --startup-lock-kp 50 150 150 50 50 50
  --startup-lock-kd 3 10 10 5 4 4 )
printf '%s\n' 'SIMULATION ONLY; foreground run'
if (
  cd "$WT" || exit 2
  printf 'command:'
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  PYTHONDONTWRITEBYTECODE=1 "${ARGS[@]}"
) >> "$LOG" 2>&1; then
  STATUS=0
else
  STATUS=$?
fi
printf 'simulation exit code: %s\nlog: %s\n' "$STATUS" "$LOG"
printf '%s\n' 'shell remains active'
```

本批仍待合并，前序候选保留。没有改 URDF、总路线图、真机实现或 Python 环境。

验证：上述七个 B601/边界测试文件 **131 passed**（35.83 s）；Ruff、资源 audit、
`git diff --check` 通过。新增倍率非法值、CLI 传递、零增益不冒充效果验收、墙约束，
以及长时正常时钟/外部一步污染回归。10 秒 6D CLI 实验为独立的离线效果证据。

## M1 采样保持功率诊断

基线 `e81dc0070fc07f6ace81e1864a7d24241209b89d`；后继分支
`codex/b601-real2sim-energy-diagnostics`。现有 demo JSON 增加 `power_diagnostics`，
不新增入口、控制增益或驱动器状态，也不改变任何命令或验收判定。

统计仅包含 ACTIVE、确实推进且未 fault 的 physics tick；末次 fault tick 排除并显式标记。
使用步前速度和该步实际 actuator force 的左矩形积分 `sum(tau · dq * dt)`，记录净功、
正功、吸收功的正幅值、峰值正功率及样本数。只读对照通道为：

- `applied_total`：实际执行器总力矩，不把其正功直接判作不稳定，因为包含重力和弹簧。
- `nominal_motor_local_damping`：本地 `-Kd*dq`，位于总 actuator saturation 之前；
  不是限幅后的可加分解。
- `held_nominal_cartesian_damping`：仅 pose fixture，采样时计算无裁剪的
  `-J(q_sample)^T D J(q_sample) dq_sample`，在 host tick 间保持。使用实际配置的 D，
  不含 startup blending、fixture clipping 或 feedforward slew；**此 torque 只用于观测，
  从不写入控制指令**，不能冒充实际末端阻尼输出。

固定原有 6D fixture、operator profile、官方 startup gains、1.5 s：

| host Hz | ACTIVE 有效样本 | 名义保持阻尼正功 J | 名义保持阻尼净功 J | 结果 |
| --- | --- | --- | --- | --- |
| 20 | 50 | 0 | -0.00068034 | 加速度保护，统计被截断 |
| 125 | 24 | 0.00027350 | +0.00003228 | 加速度保护，统计被截断 |
| 500 | 1200 | 0 | -0.00923137 | 原有 fixture PASS |

三组采样瞬间的名义阻尼均无正功率，本地阻尼均无正功。125 Hz 保持期间却出现正功，
说明“采样瞬间耗能”并不保证保持期间耗能。20 Hz 在观测窗口未出现同样证据，不能
将其失败也归因为已证实的阻尼注能。各组持续时间不同，不应直接比较总功大小。

这些通道不是完整储能函数，也未分离限幅后各力矩项，不是 energy tank、passivity observer
或稳定性证明；JSON 中 `passivity_claim=false`、`hardware_claim=false`。
下一步需把限幅/slew/重力/弹簧的影响继续分开，任何能量约束控制设计应另行验证，
不能用这份 observer 的净功直接控制或重新使能机械臂。

前序命令无需修改即可得到附加诊断。仅仿真，待合并；未改真机代码、环境或总路线图。

验证：同上七个 B601/边界测试文件 **134 passed**（33.21 s），Ruff、资源 audit、
`git diff --check` 通过。新增 signed/positive/absorbed work 恒等式、本地阻尼非正功和
125 Hz 保持期间正功回归。固定同一外部 vendor 路径的 500 Hz 前后 CLI 对照中，
移除新增 `power_diagnostics` 后整份 JSON 完全一致，确认本次观测没有改变原有输出。

## M1 主机前馈变化率隔离与 125 Hz 验证

基线 `e2e47c63fade534130b651d4fb619e6ff41090ea`，后继分支
`codex/b601-real2sim-feedforward-rate`。核实时本地 development 为 `24aa543b`，
尚不包含该基线；本批仍是有序后继候选，不声称已完成合并。

复用已有 `default_b601_mit_profile(..., torque_rise_time_sec=...)`，仅在现有 demo
开放同名 CLI/API 参数。省略时原样使用驱动器默认 0.05 s，不复制默认常量。
其语义是 `max_torque_rate = URDF effort ceiling / torque_rise_time_sec`，**不是启动锁
交接时长**。real2sim 下限制 host feedforward；motor-local Kp/Kd 不经过它。
profile 已有 metadata 与六轴实际变化率数值继续作为 JSON 证据来源。
参数必须有限且大于零；未改刚度、阻尼、URDF、力矩绝对上限和运动/效果门限。

### 1.5 s 对照

以下 rate scale 是相对原默认变化率的实验记法（不是新 CLI 参数），对应 normalization
时间分别为 0.05、0.5、5 s。实际实验通过已有 profile 构造参数完成。

| host Hz | rate scale | 结果 | 已执行 ticks | slew 受限 ticks |
| --- | --- | --- | --- | --- |
| 20 | 1 | J6 acceleration guard | 351 | 0 |
| 20 | 0.1 | 同上 | 351 | 0 |
| 20 | 0.01 | J2 velocity guard | 1150 | 16 |
| 125 | 1 | J6 acceleration guard | 329 | 8 |
| 125 | 0.1 | J6 acceleration guard | 337 | 10 |
| 125 | 0.01 | PASS | 1500 | 155 |
| 500 | 1 / 0.1 / 0.01 | 均 PASS | 各 1500 | 21 / 21 / 29 |

20 Hz 原配置在失败前没有触发 slew，不能解释为 slew 延迟导致这次失败；强行进一步
降低变化率也只是把失败换成 J2 速度保护。125 Hz 则得到一个可继续验证的有限候选。

### 10 s 方向验证

固定 operator gravity、官方启动锁参数，125 Hz、normalization=5 s：
`pose-6d/pose-tx/pose-ty/pose-tz/pose-rx/pose-rz` 六项 PASS；
`pose-ry` 完整运行且无 driver fault，但效果 FAIL。绕 Y 的最大角误差约 0.016606 rad，
初始约 0.014996 rad，比例约 1.107，超过既有 1.10 的峰值增长指标；最终误差已收敛。
原默认 500 Hz 在同一 10 s `pose-ry` 对照中也因同一峰值指标 FAIL（约 0.016593 rad）。
因此不能把此项全部归因于低频，也不能四舍五入或放宽指标来判 PASS。

这把“125 Hz 下完全不可运行”细化为“特定前馈变化率下，多数固定扰动通过，单轴效果仍有缺口”。
它不是全工作空间稳定域、真机参数建议或 HARDWARE_PASS。20 Hz 仍不通过。
下一步应隔离标定/plant 不匹配对 Y 轴瞬态的影响，并继续 M1 故障路径与 M2 数据契约，
不应围绕单个姿态继续盲目压低变化率。

仅仿真前台复现（Python 日志与命令写同一固定文件）：

```bash
WT=/tmp/b601-real2sim-feedforward-rate
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python
LOG=/tmp/b601-real2sim-command-and-output.log
ARGS=( "$PY" -m sim.mujoco.rebot_b601_rs.haptic_demo
  --rebot-source-root hardware/reBotArm_control_py
  --fixture pose-6d --real2sim-controller --host-hz 125 --duration-sec 10
  --torque-rise-time-sec 5
  --startup-lock-kp 50 150 150 50 50 50
  --startup-lock-kd 3 10 10 5 4 4 )
printf '%s\n' 'SIMULATION ONLY; foreground run'
if (
  cd "$WT" || exit 2
  printf 'command:'
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  PYTHONDONTWRITEBYTECODE=1 "${ARGS[@]}"
) >> "$LOG" 2>&1; then
  STATUS=0
else
  STATUS=$?
fi
printf 'simulation exit code: %s\nlog: %s\n' "$STATUS" "$LOG"
printf '%s\n' 'shell remains active'
```

把 fixture 改成 `pose-ry` 应返回 exit 3，保留失败供后续定位。未运行真机、未修改环境、
总路线图或先前待合并候选；本批依赖基线并等待合并。

验证：七个 B601/边界测试文件 **141 passed**（38.68 s），新增参数非法值、CLI 转发及
10 s `pose-6d` PASS / `pose-ry` 效果 FAIL 的回归；Ruff、资源 audit、文档 Bash 语法和
`git diff --check` 通过。归一化参数只控制已有变化率，不改变完整的启动阶段时长。

## M1 重力基线与故障回归、M2 证据准备

基线 `b549688ac1d0f4fa6bf8da7d89f4662d91a9050a`，后继分支
`codex/b601-real2sim-replay-evidence`。本批不把 M1 全部或 M2 定量辨识标为完成。

### Y 轴瞬态定位

新增纯 controller 工厂 `B601Real2SimControllerProfile.raw_model_baseline()` 和现有 demo
参数 `--gravity-control-baseline raw-model`，必须显式开启 real2sim。默认仍为 operator。
raw-model 仅移除重力校正，保留本地 Kd，不改变 URDF、plant 或验收门限。API 可接收
显式 controller profile，不再依赖进程内替换来做对照。

同一 10 s `pose-ry`，原始重力控制在 125 Hz（normalization=5 s）和 500 Hz（默认）均 PASS。
125 Hz 初始/最大角误差约 0.014996 rad，最终约 0.002442 rad；500 Hz 最终约 1.65e-7 rad。
这支持“控制标定与未辨识 plant 不匹配参与了超调”，不支持删除真实 bias 或反写假质量。
前节仅仿真命令加 `--gravity-control-baseline raw-model --fixture pose-ry` 可复现。

新增 ACTIVE 阶段 invalid/wrong-frame/dropout/duplicate 回归：不推进 accepted sequence，
不刷新 watchdog，最终锁存 protective hold；新合法 reference 不能自行重新使能。
这是指令通道回归，**不是**传感器坏帧、真实电机失效或 CAN partial-send 仿真。

### 当前真机证据限制（高严重度、高置信度）

现有 `real2sim.py` 的纯函数 `audit_b601_probe_trace(payload)` 仅审查既有
startup-replay/gravity-calibration JSON，不导入硬件、不打开 CAN、不拟合或回放。
检查六轴顺序、非空样本、严格递增的 sequence/反馈观察时间、完整有限的六轴数据和
非负驱动增益。返回 poll cadence、FD 速度摘要、freshness 缺口、声明身份及来源终止状态。
当前格式缺少逐样本发送时间/回执，因此格式检查通过仍是 `QUALITATIVE_TRACE_ONLY`，
绝不升级成 `quantitative_paired_replay_ready=true`。声明身份不冒充已与资产比对验证。

| 现场记录 | 样本数 | 未验证 freshness | 观察跨度 | poll 中位间隔 |
| --- | --- | --- | --- | --- |
| `/tmp/b601-rs-j2-bias-minus010-r1/result.json` | 867 | 867（100%） | 43.463 s | 0.050161 s |
| `/tmp/b601-rs-j2-gravity-scale-095-20260909-173751/b601_j2_gravity_scale_095_startup_replay.json` | 2151 | 2151（100%） | 107.875 s | 0.050157 s |

粒度是六轴主机观察＋控制记录，不是电机独立采样。顶层 FAIL 来自操作者终止，不能单独
证明样本无效；其时间/来源缺口也不能因为曾通过 startup replay 就忽略。第一组全为
WAITING_FOR_R，第二组含 RAMP_IN/RELOCKED/HANDOFF/SECOND_ACTIVE。未改写任何原始字段。

关联 raw log **有** TX/RX 证据：第一组 `motorbridge_raw.log` 跨约 1555.285 s，第二组
`motorbridge_robstride_startup_raw.log` 跨约 126.245 s。raw 使用 `unix_time_ns`，JSON
使用主机 monotonic 观察时刻；两种时钟映射和逐样本归属尚未验证，不能仅凭文件名/PID
将全部 raw 帧当成同一个可拟合窗口。这些 raw 文件保留，未复制到 Git。

定量 M2 尚需：可核验的反馈新鲜度、command/feedback 时间对齐、明确的无接触区间和
负载条件、fit/held-out 分离。旧记录可能含人手外力，不能独立将其分离为摩擦/重力误差。
按 data-quality Skill 的用途边界检查，将证据和 Python 检查保留在现有模块/Markdown，
不另建 notebook/HTML 报告体系。下面只读取旧文件，不连接设备：

```python
import json
from sim.mujoco.rebot_b601_rs.real2sim import audit_b601_probe_trace

with open("/tmp/b601-rs-j2-bias-minus010-r1/result.json") as stream:
    report = audit_b601_probe_trace(json.load(stream))
print(json.dumps(report, indent=2))
```

需现场补充：是否已有更新的带时间对齐、无接触标注的记录，以及其负载/采集条件。
若没有，应先安排采集能力验证，再准备前台真机采集命令；不得直接重放旧 torque 或
自动使能电机。此处只暂停依赖真实数据的辨识路径，不声称数字孪生或 HITL 已完成。

本批验证：七个 B601/边界测试文件 **157 passed**（46.31 s），原始重力基线 Y 轴
125 Hz / 10 s CLI PASS；Ruff、资源 audit、`git diff --check` 通过。纯数据检查覆盖
重复时间/序号、J7 混入、缺维、非有限和负驱动增益，并拒绝给旧格式伪造拟合资格。
候选待合并，未 push、安装或执行真机操作；原始现场数据仍在原路径，未纳入提交。

## M2 首轮现场采集能力验证（被动接收）

操作者确认没有更新的合格记录，转入真机采集准备。本批基于 `73e3d4ef`，分支
`codex/b601-passive-capture-clocks`；只扩展现有 frequency probe 的 read-only 模式，
不修改控制、保护保持、重力参数或总路线图。

新增可选 `--raw-output`：独占创建 JSONL，保存完整 SocketCAN 原始字节（包括无法解码的帧），
及主机 `receive_monotonic_ns / observe_realtime_ns / clock_pair_end_monotonic_ns`。
墙钟读取发生在两个单调时钟读取之间；这是 userspace recv 后观察，不是电机采样时间，
也不是内核接收时间。其他进程的本机 TX 回显可能被接收，不能冒充本 probe 的发送回执。
成功结束有 end 记录；Ctrl-C 留存部分文件但不写成功尾记录。拒绝覆盖旧 raw 文件，
拒绝 raw 与 summary 同路径，拒绝在非 read-only 模式使用该参数。

第一轮现场只被动监听 15 s，不构造 MotorBridge backend，不发送 CAN，不改变主动上报、
电机使能或现有 protective hold。操作者无需推动机械臂；保持可靠支撑，不为采集退出既有
保护程序。采集始终前台，实际命令和详细输出追加到固定 command_and_output.log，
raw/summary 每轮放入独立目录。零帧只证明该时段没有观察到流量，不能证明断连或反馈新鲜。
先核验收到的帧和时间基准，再决定是否需要显式开启遥测及受控无接触采集。
本轮不能代替配对 command/feedback 记录、动态激励或定量辨识。

离线验证：hardware backend、active report、low gain probe 三文件 **183 passed**，
全部使用假 transport；覆盖接收中断、空流、bind 失败关闭 socket、原始帧保留和 CLI 隔离。
资源 audit 和 diff 检查通过；Ruff 无新增问题，frequency probe 原有 EXE001、UP041、
RUF046 三项保持未动。候选待合并，未 push、安装或执行任何真实 CAN 操作。

## M2 静态配对采集准备

后继 `codex/b601-paired-capture` 基于 `3951ea18`，仍只修改原 probe、对应测试和本进度文档。
操作者确认首轮条件为失能静置、无人手接触、默认原装末端且无额外负载；默认末端质量
不能按零负载质量处理。15 s raw 有 4197 帧，ID 1–7 均约 40 Hz，J7 保留但不参与六轴验证。
六轴首个大端 uint16 字段均仅变化一个计数。后续 mechPos 读取与 raw 相隔约九小时，
故 ±4π 映射只有静态数值相近的初步证据，不是同步解码 PASS。

现有入口新增 `read-paired`，显式确认 `B601_RS_PAIRED_STATIC_READ_ONLY`，要求 raw 和 summary
分别写新文件；拒绝 enable/allow-motion/auto-disable。一个前台 Python 进程中，主线程经
MotorBridge 0.4.9 串行读取 ID 1–6 的 0x7019（500 ms timeout，读间隔至少 50 ms），
非 daemon 接收线程复用被动接收路径；启动接收成功后才开始请求。每次读取保存同一
monotonic 时钟域中的 start/end、序号、motor id、成功值或未完成标记。
这不是高频控制测试，也没有解码 type-24 到控制状态。SDK 读取超时不自动重试。

接收失败、非有限读取和 Ctrl-C 都停止后续请求，释放 motor/controller 句柄，停止并 join
接收线程。部分参数记录写 ERROR/INTERRUPTED summary，不能因 raw 正常关闭而冒充配对成功。
host request window 不是总线精确 TX receipt 或电机采样时间；SDK 响应关联、内核排队、
并行其他请求及源 freshness 仍需结合 raw 审查，不自动认定两条记录已配对。

重要二次复核：已安装 Python Controller.__exit__ 会 shutdown，v0.4.9 Rust CoreController
shutdown 会 disable devices。因此不使用 Controller 上下文管理器，改用 ExitStack 仅调用
Motor.close 和 Controller.close；对应 free/Drop 释放资源并停止 SDK polling，不执行 shutdown。
此前 CLI 参数读取发生在操作者已失能状态，但不能把它泛化成“退出不改变驱动状态”的入口。

现场仍须失能、有支撑、无接触、默认末端且无额外负载，先做同姿态静态对照。
不安排运动、不修改主动上报周期、SDK 安装、URDF、控制或保护保持。候选待合并。

验证：三文件 hardware backend / active report / low gain probe **194 passed**（1.18 s），
假 SDK 不提供 enable、disable 或 shutdown，覆盖六轴读取、时间窗、异常/中断留存、
非有限拒绝、接收启动失败、覆盖拒绝和 CLI 授权。Ruff 无新增问题（沿用前三项基线），
diff 检查和资源 audit 通过。真实配对采集尚未执行，未合并、push 或安装。

## M2 配对证据离线回归

后继 `codex/b601-paired-audit` 基于 `9e57409b`。现有 frequency probe 新增纯函数
`audit_paired_capture(summary, records)` 和 `audit-paired --input-summary ... --input-raw ...`。
不新建 CLI，不打开 CAN、不构造 MotorBridge、不修改采集及控制路径。
输出存在时拒绝覆盖；输入 SHA-256 绑定被审查的具体文件，不修改原始证据。

审查范围：完整 header/footer、计数和 channel 一致、经典扩展数据帧、递增接收时钟、
请求窗口不重叠、完整有限六轴样本、唯一 type-17 请求与响应、0x7019 参数及原始 float32
与 SDK 数值精确相等。缺失、重复、顺序倒置、错误主机/电机/参数、额外未归属 type-17、
RTR/error/控制帧或不完整文件均返回 FAIL。每次读取的原始帧只允许匹配一次。

type-24 只统计电机及主机分布，不参与 type-17 匹配，也不解码关节角度。本轮真实文件
暴露了先前“所有上报都发往 0xFD”的错误假设：实际 4166 帧发往 0xFD，34 帧发往 0xFE。
参数请求/响应仍全部使用 0xFD，不能将旁路上报地址差异直接解释为参数配对失败。
审查结果显式保留 mixed_stream_host_ids，不猜测其固件原因、不修改主机配置。

输入 `/tmp/b601-rs-commissioning/paired-static-oQLlXd/`：六轴各 43 次、共 258 次全部匹配，
原始 4716 帧中 type-17 为 516、type-24 为 4200。结果为 host_pairing_passed=true，
但 freshness_verified/static_decode_validation_passed/quantitative_replay_ready 均为 false。
同目录 audit_v1.json 保存了错误地址假设导致的初版 FAIL；audit_v2.json 是修正后的结果，
两份均保留用于追溯，未改写 raw.jsonl 或 summary.json。

复现（纯离线，可在当前 checkout 执行，不使用任何硬件参数）：

```bash
python scripts/hardware/rebot_b601_rs_frequency_probe.py \
  --mode audit-paired \
  --input-summary /tmp/b601-rs-commissioning/paired-static-oQLlXd/summary.json \
  --input-raw /tmp/b601-rs-commissioning/paired-static-oQLlXd/raw.jsonl
```

验证：hardware backend / active report / low gain probe 三文件 **215 passed**（1.19 s），
含 21 个新增离线用例；CLI 测试明确禁止 socket/backend 构造，并检查已有输出不被覆盖。
Ruff 无新增问题（既有 EXE001/UP041/RUF046 不动），diff 检查、资源 audit 通过。
只读检查与真实文件离线审查不等于硬件力控验收。候选待合并，未 push、安装或触发硬件。

## M2 Plant 参数入口与物理回归（非辨识结果）

后继 `codex/b601-plant-profile` 基于 `1e2d5a8e`。总路线图保持不变：B601 real2sim →
ReBot real 到 RM65 sim 遥操 → RM65 policy 到 ReBot sim → 同一 execution cut 人工 residual →
分阶段真机验收。当前仍不宣布 M1/M2 整体完成，不越过真实数据和角色决策门禁。

现有 `real2sim.py` 新增显式 `B601Real2SimPlantProfile`，包含六轴独立的
`damping_nm_s_per_rad`、`frictionloss_nm`、`armature_kg_m2`，使用已有六轴非负有限校验。
必须有非空 name/parameter_source，metadata 固定声明 provisional_not_hardware_identified，
不能仅传入参数就自称完成物理辨识。省略 profile 时维持已有默认零被动项。

参数在已有 scene 的 MJCF 编译前写入 J1–J6，禁止用于尚未验证的 position 模式；
不修改厂商 URDF/mesh、body mass/COM/inertia、J7、夹爪约束、控制增益、力矩限幅或 slew。
同一 profile 显式交给 effort driver 作为期望契约；compiled dof 系数与声明不符即拒绝，
没有 profile 时仍拒绝隐藏的非零被动项。scene 负责物理值，driver 不再次施加这些项。
重力标定 bias/scale 和 driver Kd 仍属于独立 controller profile。

MuJoCo 官方定义（经 Context7 核对）：关节 frictionloss 是有界干摩擦约束，不是简单
`-sign(dq)*C` 前馈；armature 增加广义惯量；黏性 damping 进入 passive force。
来源：https://github.com/google-deepmind/mujoco/blob/main/doc/computation/index.rst
因此不把现有 controller Coulomb 补偿项复制为 plant 的已辨识摩擦。

API 用法：在构建 scene 与 driver 时传入同一个显式 profile；以下系数保持零，仅演示入口，
不是已辨识默认值或真机调参建议：

```python
from sim.mujoco.rebot_b601_rs.real2sim import B601Real2SimPlantProfile

plant = B601Real2SimPlantProfile(parameter_source="explicit_simulation_baseline")
scene = build_b601_mujoco_scene(source_root, actuation_mode="effort", plant_profile=plant)
driver = B601MujocoEffortDriver(scene, pinocchio_model, plant_profile=plant)
```

物理回归验证：armature 只增加预期的质量矩阵对角项；黏性阻尼力等于 `-D*dq` 且耗散；
干摩擦使小外加力矩下的关节速度明显低于零摩擦 baseline；静态重力、body mass/inertia 和
driver Kd 公式保持独立。测试系数仅为可辨别效果的合成参数，不推断为这台机械臂参数。
同一 source gitlink 在隔离 worktree 中按固定 5ba28ace materialize；共享资产只读，未安装环境。

剩余实际门禁：需要带命令/力矩时间窗、无接触与负载标注的动态采集以及 held-out 数据，
才能选择非零 plant 系数；现有失能静置只支持静态及时序检查。sensor/transport delay、
量化及高带宽触觉仍未建模/验证，不因为本入口完成而改写 evidence 状态。

验证：real2sim、MuJoCo MIT/haptics/基础 scene、submodule 和 composition boundary 六文件
**150 passed**（47.08 s）；Ruff、资源 audit、diff 检查通过。早期新增测试误把已有
execution_metadata property 当作方法调用，修正测试调用后重跑完整上述集合通过，未改验收要求。
本批只修改五个声明路径，无硬件操作，候选待合并，未 push 或安装。

## P0/P1：依赖物化与主机接收观测契约（2026-09-12）

本批不修改总路线图，也不开始 P2 真机辨识。P0 在隔离 Task 工作树中按 AgenticRobot
记录的 gitlink 物化 `hardware/reBotArm_control_py`，精确提交为
`5ba28acef46237eb6a7560658bbc43b06cf8a259`。物化不产生源码差异，也不把本机外部 checkout
写进项目配置；其作用是让模型、动力学和 MuJoCo 离线测试实际加载固定的厂商资产，而不是因
空 gitlink 静默 skip。完整 B601 离线集合在该依赖上为 **469 passed**。

P1 分成两个保持旧路径不变的增量候选：

- MotorBridge 为每个被接受的 RobStride type-2 operation-status 帧记录单电机递增
  `status_sequence` 和主机单调时钟 `receive_age_ns`，通过新增 C ABI、Python 与 C++ API
  暴露；旧 `MotorState` 布局和 `get_state()` 不变。sequence 与对应 state/接收时刻在同一
  锁内提交，之后才发布给等待者，避免先看到序号、后看到旧状态。
- AgenticRobot 新增显式 `poll_feedback(require_host_freshness=True)`。该路径只消费上述已解码
  operation-status observation，不与 0x7019 参数读取混合；要求 MotorBridge 能力标记，拒绝
  空缓存、非整数/非正序号、负年龄以及超过 `feedback_timeout_sec` 的帧。默认 cache 路径和
  `request_feedback=True` 兼容路径保持原行为。
- `host_receive_freshness_verified` 与既有 `freshness_verified` 分开：前者只能证明指定帧最近被
  本机接收，后者继续为 false，因为当前协议栈没有设备采样时间戳。active-report 原始传输
  probe 仍为 `control_feedback_admission=NOT_ADMITTED`，不会因能力存在就把 raw cadence 冒充
  可控状态。
- 本批只构建、测试和检查本地隔离候选；没有打开 CAN、改变 active-report、使能、失能或发送
  任何位置/力矩命令，也没有安装到共享 Python 环境。

离线证据：MotorBridge Rust `33 passed`，Python API `3 passed`，全部 C++ example 编译通过；
MotorBridge release ABI 与 AgenticRobot 的能力协商 smoke PASS；AgenticRobot 聚焦测试
`104 passed`，物化 vendor 后的完整 B601 离线集合为 `471 passed`。精确提交、Task profile
和集成顺序由 Harness handoff 记录；两份增量在真正集成前均视为待集成，不能据此运行真机。

## P2 及后续官方能力复用清单（调研，不代表已接入）

Seeed 官方 Wiki 将 Wiki 作为 demo 使用说明的单一事实源；本地固定的官方 SDK 则作为代码
级依赖。后续按下表处理，先复用和薄适配，不复制一套平行实现：

| 能力 | 官方现有实现 | 本项目处理 |
| --- | --- | --- |
| FK、完整 pose IK、Jacobian/Pinocchio 模型 | 固定 SDK 的 `kinematics/` 与 `example/sim/fk_sim.py`、`ik_sim.py` | 继续由 `robot_motion.rebot_b601_rs` 薄封装并交叉验证，不另写求解器 |
| SE(3) 测地线、minimum-jerk、CLIK | 固定 SDK 的 `trajectory/`、`traj_sim.py`、`RebotArmEndPose` | 复用数学实现；跨机器人 residual 仍由既有 `control_composition` 和 target safety 所有者处理 |
| 重力和逆动力学 | 固定 SDK 的 `dynamics.compute_generalized_gravity` 与 `GravityCompensation` | 继续复用模型计算；操作者 scale/bias/driver-Kd 保持独立 controller profile，不反写 URDF |
| RS 电机组、MIT/POS_VEL 参数与模式准备 | 固定 SDK 的 `RebotArm` / `JointGroup` 与 RS YAML | 由现有硬件 backend 薄适配并保留逐关节 receipt/protective hold，不复制寄存器表或启动顺序 |
| RS MuJoCo 模型、physics/kinematic 模式、Fake Driver | Seeed Wiki 指向的 `ReBot_Arm_web_RS`，包含 RS MJCF、`mujoco_sync.py`、100 Hz Fake Driver 与 250 Hz MuJoCo sync | 先作为有提交身份的 upstream reference/runtime；核对 license、frame、joint order 和惯性后，只引入确有证据的资产/参数或 adapter，不把整套 ROS/Web 栈再嵌入当前包 |
| 接触、桌面物体和物理抓取 | 官方 RS MuJoCo physics scene | 后续接触阶段优先比较并复用 scene/contact 定义；kinematic 模式只做显示/接口检查，不能作为重力、增益、接触或力反馈验收 |
| 125 Hz MIT 在线参考与安全回零 | 官方 RS Web/ROS2 控制栈已有速度、加速度、jerk 限制和安全状态机 | 作为行为与参数来源核对；不得替换本项目已验证的完整六轴 gravity+damping protective hold，也不得把 Web 显示滤波带入控制环 |

调研来源：

- Seeed B601-RS Pinocchio/MeshCat：<https://wiki.seeedstudio.com/rebot_arm_b601_rs_pinocchio_meshcat/>
- Seeed B601-RS Web/ROS2/MuJoCo 开发指南：<https://wiki.seeedstudio.com/rebot_arm_b601_rs_web_simulator_developer_guide/>
- 固定官方 SDK：<https://github.com/Seeed-Projects/reBotArm_control_py>

P2 的下一步仍是测量而非实现新的控制器：先在集成后的 exact MotorBridge/AgenticRobot 组合上
执行无运动、前台、可中断的 active-report decoded observation 验收，记录 J1-J6 的 sequence、
receive-age、完整率和 cadence，并与独立 0x7019 静态基线比较。只有该证据通过后，才把接收
delay/jitter/dropout 写入 Real2Sim observation/transport profile。armature、被动阻尼、摩擦和
低通带宽仍需安全动态采集及 held-out 验证；官方场景或示例不能代替本机辨识结果。

## P2.1 主动上报传输验收（2026-09-12）

合并后的主动上报探针已在现场以前台、无运动方式完成一次 30 秒传输验收。运行时 AgenticRobot
代码为 `b7e07a840fbc82f5e50c6645fd57549458049106`，原始结果保存在
`/tmp/b601-rs-commissioning/active-report-postmerge-QyJiU5/result.json`，完整命令和终端输出保存在
同目录的 `command_and_output.log`。本次只切换并关闭 active-report 配置，没有调用电机使能、失能、
位置或力矩命令；六个目标关节的清理 receipt 均完整，`cleanup_errors=[]`。

验收结果：

- `stream_transport_acceptance=PASS`、`acceptance_passed=true`，目标 40 Hz，最小允许 36 Hz；
- J1--J6 分别观测到约 39.97--39.98 Hz，每关节 1199--1200 个样本，p99 间隔约
  25.08--25.10 ms，未出现短 active-report 帧；
- 主机反馈 ID 为配置的 `0xFD`，active-report 捕获为 8394 帧，另有 4 个 type-2
  operation-status 候选帧；后者不替代 type-24 cadence 验收；
- 原始捕获还记录到 J7 约 39.98 Hz，但 J7 是夹爪、在 B601 六轴 backend 之外，不纳入本次
  acceptance 或 cleanup 覆盖范围；该观察只保留在原始证据中。

证据边界保持不变：`motion_commands_sent=false`、`motors_enabled_by_probe=false`、
`control_feedback_admission=NOT_ADMITTED`，且 `freshness_verified=false`、
`motorbridge_state_observation_metadata_available=false`。因此本结果只证明 type-24 帧能够按
约 40 Hz 到达主机，不能证明 payload 已解码为同帧、带源序号/源时间戳的关节状态，也不能证明
高频位置、重力/阻抗或 haptic 闭环可用。下一步仍需在 MotorBridge 暴露实际 firmware 的
decoded sequence/receive-age 后，做 P2 decoded-observation 验收，再将实测 delay/jitter/dropout
写入 Real2Sim transport profile。

## P2.2 Type-24 控制帧配对记录与主机时序审计（2026-09-13）

MotorBridge Type-24 解码、失能 TX/RX、使能保持和 40 Hz 保持已分阶段完成现场验收；控制入口
现在以六轴递增 `status_sequence` 和受限 `receive_age_sec` 作为 active-report admission。一次
使用新默认人工拖动 envelope 的 40 Hz 运行由操作者在约 40.13 s 主动结束，现场确认无异常；
1604 个 compute/feedback/TX 样本均约 39.94 Hz、无 deadline miss，motion guard 没有隔离或
确认中止。因为未运行满 90 s，其自动 acceptance 为 FAIL，证据只能标记为“现场行为通过、
自动时长未完成”，不要求为本软件增量重复运动。

该运行也暴露了证据格式缺口：在线控制虽消费 Type-24 metadata，但旧 `samples[]` 未保存
逐关节 sequence/receive-age，也没有逐样本 MIT dispatch receipt，不能直接用于 transport
回放。本批保持控制路径不变，只从已经用于当前控制 tick 的 `B601HardwareState` 序列化
`b601.type24-host-observation.v1`，并把现有 `B601CommandReceipt` 附在同一样本；不增加 CAN
读取、不重复 MotorBridge 解码，也不改变 torque、gravity、driver Kd、guard 或 protective hold。

新增纯函数 `audit_b601_type24_control_trace(payload)` 严格要求：

- exact J1--J6 顺序、active-report 控制 admission 和配置内 receive-age 上限；
- 各关节正整数且逐样本严格递增的 Type-24 sequence；
- 单调反馈/control/dispatch sequence 和 `feedback <= requested <= completed`；
- 完整六轴有限 q、conditioned torque、dispatch qref，以及 accepted all-arm dispatch receipt；
- 64 位十六进制 config/URDF SHA-256 身份。

PASS 只产生 `HOST_TIMING_REPLAY_READY`：报告反馈间隔、receive age、feedback-to-dispatch 延迟，
并把 sequence 跳号记为 `unobserved_status_sequence_count`，不能把跳号直接解释成总线 packet
drop。设备没有 acquisition timestamp，所以 `device_acquisition_timing_verified=false`；负载、
接触、温度和 fit/held-out 也尚未合格，因此
`quantitative_plant_identification_ready=false`。后续 exact candidate 合并后再采集一次完整前台
Type-24 控制记录即可运行该审计；本批本身不需要、也不授权任何真机动作。

## M4 ReBot real → RM65 sim：shadow 会话生命周期（2026-09-14）

本批基于 `5a1d4b1838e811a96e547f6401a0b88777038eee`，继续复用已经合并的三层实现：
host-fresh B601 只读 pose source、anchor-relative B601→RM65 residual adapter，以及
RM65 MuJoCo composition shadow controller。新增 session 只负责把这些现有边界按正确顺序连接，
不重写 FK、SE(3) 映射、RM65 measured-pose baseline、composition、IK、安全过滤或 dispatch。

- profile 默认关闭且永久禁止 `send_actions`；启用时要求 session 与 MuJoCo runtime 使用同一
  monotonic clock domain，并以可哈希 profile 固定 operation identity 和有效时域。
- `activate()` 从同一会话时刻先取得一份当前合法 B601 pose，再捕获 RM65 当前 measured pose；
  只有两侧都成功才进入 active。source/target 拒绝或未知、未类型化的下游结果都 fail closed。
- clutch 立即清除本地 anchor 与 evaluation eligibility，不轮询 B601、不调用 MuJoCo composition；
  release/recenter 必须重新取得两侧新鲜锚点。任一侧拒绝时保持 clutched，不回用旧 anchor。
- active tick 只适配本 tick 的 B601 sample；stale/invalid source 作为本 tick withdrawal 传给既有
  shadow controller，后者不会用缓存 residual 做 evaluation。异常会把 session 置回 inactive，
  需要重新双侧激活；target time 要求位于当前时刻之后且严格递增。
- session 不启动或关闭 source/runtime，不创建 execution authority，不晋升 shadow proposal，
  不发送 sim/real command。当前产物是可审计的非执行 shadow 生命周期，尚不是操作者可运行的
  Real→Sim application，也没有进行任何真机连接、使能或运动。

本批聚焦回归覆盖默认关闭、双锚点激活、source/target 拒绝、clutch/recenter、当前样本撤销、
异常失活、时钟域约束和无 lifecycle/dispatch 调用，相关组合/边界集合 **80 passed**。
项目 fast feedback 的两个 unit 分片分别为 **4265 passed / 20 skipped**、
**4418 passed / 1 skipped**，integration 为 **741 passed / 2 skipped**；Ruff、py_compile 与
`git diff --check` 通过，临时 read-only assets 链接已由 wrapper 清除。候选完成 Harness
Profiles 前仍为开发中；
下一批应在不增加第二条控制路径的前提下，把该 session 绑定到一个前台、可视化、只驱动 RM65
MuJoCo 的 application runner，再以真实 B601 只读输入做现场验收。该后续现场步骤不得使能或
控制 B601，且需单独给出预期行为、退出语义与前台日志命令。

## M4 跨机械臂 frame/scale profile（2026-09-14）

本批在 shadow session 候选上继续，先将真实 B601 安装到 RM65 MuJoCo 坐标系所需的
两条刚体变换和三轴平移尺度收敛为一个可哈希 profile。它位于 Control Station 应用
装配边界，只构造现有 `CoordinateFrameGraph` 和 `HumanResidualPoseAdapter`，不读硬件、
不管理 clutch、不解 RM65 IK、不实现新的 safety/dispatch。

契约显式固定：

- B601 source pose 为 `T_base_link_gripper_end`，RM65 residual 为
  `right_arm_base` 参考系下的 `tool_tcp` relative delta；
- 引用变换使用 `T_target_reference_source_reference`，受控帧变换使用
  `T_source_controlled_target_controlled`，二者均经既有刚体图约束验证；
- 平移 gain 在 RM65 target reference frame 中按 XYZ 应用；轴方向必须由刚体旋转
  表达，不允许用负 gain 隐式造成反射；
- 现有 canonical adapter 只支持 1:1 旋转残差，因此非 `1.0` rotation gain 显式
  fail closed，本批没有另外复制 SE(3) 缩放逻辑；
- 启用时必须有 calibration source ID、64 位 SHA-256 和两个完整刚体变换。
  默认 `off` profile 不含 identity 矩阵，无法构造可运行 adapter。

这一增量仍不包含本机实际安装标定数值，也不代表 Real→Sim runner 或真机验收已完成。
下一步是将显式 profile 加载绑定到既有 `control_station sim-hil` 生命周期，并在未提供
有效 profile 时保持默认关闭；现场的坐标方向、尺度与 clutch 验收需在该 runner 后单独进行。

## M1 反馈故障与降级契约（当前批次）

本批不改变既有控制律或保护阈值，只把已经由驱动器实现并由回归测试覆盖的故障语义收敛为
`B601SimFaultPolicy`。它是不可变的仿真契约，不是可供调用方放宽保护的调参入口；Real2Sim
effort driver 默认附带该契约，并通过 `execution_metadata["fault_policy"]` 输出，便于每次
仿真结果审计实际采用的语义。未启用 Real2Sim 的理想 MIT 分支保持原有元数据和行为。

契约固定如下：

- 序号、source timestamp 或 receive timestamp 过旧的参考只拒绝，不抢占上一条合法参考；
  直到本地 command watchdog 到期才撤销。
- 非法参考或错误 frame 立即撤销当前参考，保留本地重力/阻尼 fallback；启动生命周期启用
  时按既有状态机进入 protective hold。
- watchdog 超时先撤销参考；只有启用 startup/lifecycle 的 Real2Sim 路径才进入 protective
  hold，否则继续当前重力/阻尼 fallback。
- 非有限状态或计算力矩锁存 simulation fault，不推进 MuJoCo physics；不从缓存的 transport
  数据恢复。
- protective hold 拒绝新的参考，重新使能只允许显式 `reset()`，不允许隐式 re-arm。

`fault_policy` 的每个动作字符串都经过严格校验，拒绝未经设计的替代语义，因此不会出现
“为了让仿真继续而把 stale/wrong-frame 当作合法参考”的隐式降级。该契约仍只说明仿真模型的
行为，不构成 RobStride 固件、CAN 新鲜度或真机安全保证。

验证：Real2Sim、MuJoCo MIT、基础 MuJoCo、动力学四个测试文件 **131 passed**（31.74 s），
新增固定契约和 metadata 回归；Ruff、`py_compile`、`git diff --check` 通过。测试前按声明
gitlink materialize 了 `hardware/reBotArm_control_py` 到 pinned commit `5ba28ace`，没有修改
其内容。没有连接 CAN、没有使能或移动真机。当前批候选待合并，未 push、安装或修改总路线图。
