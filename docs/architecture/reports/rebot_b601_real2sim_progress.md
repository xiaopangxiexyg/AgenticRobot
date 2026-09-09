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
