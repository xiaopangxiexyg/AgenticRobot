# B601-RS 力觉控制：离线与真机分阶段验收

本文把“当前无真机可验证的结果”和“机械臂接入后的物理验收”分开。任何离线、
MuJoCo 或 GUI 结果都不构成真机使能、运动或安全授权。

当前实现止于：

- `LOCAL_WORLD_ALIGNED` 末端 wrench、坐标变换与 `J.T @ wrench`；
- 平移虚拟弹簧和单侧虚拟墙；
- MuJoCo 理想广义关节力矩、重力补偿、阻尼、overlay 限幅及 watchdog；
- 可自动判定控制效果的无头仿真。

尚不存在 B601 真机 backend。电机、SocketCAN、真实反馈新鲜度、发送确认、摩擦、
绝对力矩标定、触感和物理安全均未验证。

## 1. 固定契约

内部统一使用右手系、SI 单位，`X` 向前、`Y` 向左、`Z` 向上。B601 的 Jacobian
和输入 wrench 都在 `base_link` 轴向、`gripper_end` 原点：

```text
frame_id = base_link/world_aligned@gripper_end
twist    = [vx, vy, vz, wx, wy, wz]
wrench   = [fx, fy, fz, tx, ty, tz]
tau      = J(q).T @ wrench
```

不同原点/轴向的 wrench 必须先通过完整的 `Ad(T)^-T` 变换，不能只重排 XYZ。
离线测试同时检查 `wrench.T @ twist == tau.T @ qvel` 的功率不变量。

本地有线控制的失效语义为：

- 合法但旧的 sequence/timestamp：拒绝新样本，不抢占当前合法参考，由本地 watchdog
  最终撤销；
- `valid=false` 或错误 frame：立即撤销位置/力觉 overlay，保留 backend 自己拥有的
  重力补偿和零速度阻尼；
- 反馈陈旧、缺失或非有限值：不得继续用旧姿态计算重力或力觉，必须进入经物理评审的
  fault/safe-stop 路径；
- 真机 safe-stop 不能简单等同于“立即去使能”。负载未被支撑时，去使能可能导致自由落体。

## 2. H0：无设备自动验收（当前可完整执行）

从仓库根目录执行，不打开设备、不访问 CAN，默认不启动 GUI：

```bash
bash scripts/run_pytest.sh -q \
  tests/test_rebot_b601_haptics.py \
  tests/test_rebot_b601_mujoco_haptics.py \
  tests/test_rebot_b601_dynamics.py \
  tests/test_rebot_b601_mujoco_mit.py
```

可单独查看机器可判定的 JSON 结果：

```bash
DEV_PYTHON="$(bash scripts/resolve_dev_python.sh)"
"$DEV_PYTHON" -m sim.mujoco.rebot_b601_rs.haptic_demo \
  --rebot-source-root hardware/reBotArm_control_py \
  --fixture spring
"$DEV_PYTHON" -m sim.mujoco.rebot_b601_rs.haptic_demo \
  --rebot-source-root hardware/reBotArm_control_py \
  --fixture wall
```

两次命令都必须以 `simulation_acceptance.passed=true` 退出，并满足：

- 完成全部 1 kHz physics tick 和全部 500 Hz command tick；
- 无 driver fault、非有限状态或关节位置/速度越界；
- 最终总力矩始终落在仿真 profile 的硬上限内；
- 弹簧：初始末端误差至少 20 mm，最终误差不超过初始值的 45%；
- 虚拟墙：初始穿透至少 20 mm，最终穿透不超过 2 mm；
- 输出同时记录峰值 wrench、请求 overlay、实际总力矩及所有限幅 tick。

`20 ms` 的短运行会正常完成程序，但必须因尚未形成足够闭环效果而返回非零；这用于证明
验收器没有把“运行过”误报为“效果通过”。

需要人工查看运动时，才显式增加 `--viewer`。当前 viewer 会显示机械臂的响应，不绘制墙面；
JSON 仍是效果判定依据：

```bash
"$DEV_PYTHON" -m sim.mujoco.rebot_b601_rs.haptic_demo \
  --rebot-source-root hardware/reBotArm_control_py \
  --fixture wall \
  --viewer
```

## 3. 真机测试的共同前置条件

以下 H1–H8 均为机械臂到货后的验收计划，现在未执行。进入任何使能阶段前必须同时满足：

1. 记录 AgenticRobot commit、`hardware/reBotArm_control_py` gitlink、配置文件 SHA-256、
   固件、电机型号/ID、CAN 接口和供电版本；
2. 清空工作区，机械臂与负载有可承重支撑，末端不接触桌面或本体；
3. 操作者握住物理急停，另有观察员；确认急停切断/恢复流程已实际演练；
4. 同一 CAN 与机械臂只有一个 control-loop owner；关闭官方 demo、遥操、数采和其他控制进程；
5. 每阶段使用独立日志，包含 host monotonic timestamp、cycle index、每关节 q/qdot/tau、
   命令分量、限幅、反馈年龄、发送错误和 fault transition；
6. 力矩、力矩变化率、速度、位置、Kp/Kd、wrench 和持续时间上限经过硬件负责人评审；
7. 每次只放开本阶段所需的最小权限，先单关节、后多关节，先无 overlay、后低幅 overlay；
8. 任一异常立即停止本阶段，不把“重新运行成功”当作根因闭环。

## 4. 真机 backend 上线前必须关闭的接口缺口

固定版本上游目前提供本地 `send_mit(pos, vel, kp, kd, tau)` 和 500 Hz Python 控制线程，
但不能直接视为可验收的真机 backend：

- `JointGroup.send_mit()` 逐关节发送并吞掉 `CallError`，函数返回不代表六关节完整应用；
- `get_state()` 在某个电机没有状态时填 `0.0`，无法区分真实零位与缺失反馈；
- 当前接口没有每周期发送 receipt、六关节原子性或反馈 source timestamp；
- 上游标定记录指出部分 RS 固件的 `mechVel` 不能直接当作 rad/s，应以 `mechPos`
  有限差分交叉验证；
- 已有重力补偿示例使用 Kp=2、Kd=1，这只是上游示例参数，不自动成为本项目真机验收增益。

因此真机 backend 至少要先增加：每关节有效位、反馈时间/年龄、发送错误汇总、部分发送检测、
本地 watchdog、fault latch、最终总力矩/变化率限制和明确的 safe-stop receipt。缺少其中任一项，
H2 及以后不得标记通过。

## 5. 分阶段真机验收

每阶段都必须保存“执行人、观察人、时间、commit、配置、完整日志、结论和异常”。下一阶段只在
上一阶段通过且硬件负责人签字后进行。

### H1：连接与只读反馈（电机保持去使能）

1. 仅初始化本地有线 backend，确认电机 ID 与 `joint1..joint6` 一一对应。
2. 在去使能状态缓慢手动移动一个关节，其余保持静止；逐关节验证 q 的符号、单位、零位和顺序。
3. 用 host monotonic clock 记录反馈间隔，交叉比较上游 velocity 与位置有限差分。
4. 断开一个反馈源，确认状态成为 `invalid/stale`，而不是静默变成合法的全零样本。

通过条件：六关节身份/符号正确；缺失、重复、倒序和非有限反馈都可检测；没有发送任何 MIT
目标；反馈新鲜度和速度语义已有日志证据。

### H2：急停、watchdog 与 fault 路径（机械支撑保持）

1. 在仍未施加 haptic overlay 前验证物理急停、软件 stop、进程异常退出和 CAN 断开。
2. 注入本地 command stale、错误 frame、`valid=false`、反馈 stale/nonfinite 和单关节发送失败。
3. 检查命令陈旧时撤销上层参考；frame/invalid 立即撤销；反馈故障进入预审 safe-stop；
   部分发送不得被报告为六关节成功。

通过条件：fault 在一个 watchdog 窗口内被检测并锁存；日志包含触发原因与 stop receipt；
机械臂没有无界运动。去使能后的承重由物理支撑保证，不能用机械臂“看起来没动”替代支撑证明。

### H3：单关节低幅 MIT 符号与限幅

1. 每次只使能一个已支撑关节，其他关节去使能或处于经评审的安全保持状态。
2. 使用当前实测位置作参考、零速度、零 overlay，从最小已评审 Kp/Kd 和力矩上限开始。
3. 施加极小正/负位置误差，确认运动与恢复力矩方向；再触发软件力矩和 slew limit。

通过条件：六关节正负方向均与 URDF 一致；阻尼功率 `tau_d.T @ qdot <= 0`；所有上限在
最终 actuator 边界生效；命令记录、反馈和 fault 记录可逐周期对齐。

### H4：重力补偿与阻尼

1. 先在无接触、肘部抬起的 clearance pose 验证 Pinocchio 与真机 q 的零位/符号。
2. 操作者承托机械臂；保持低 Kp/Kd，重力前馈从零按受限 slew 缓慢增加。
3. 在多个无接触姿态检查重力方向、静态漂移、峰值力矩和摩擦死区。
4. 小幅人工扰动后，确认阻尼只耗散能量，不把机械臂推向扰动方向。

通过条件：任何姿态都无反向“助推”；反馈新鲜时才能更新 `g(q)`；重力、阻尼、overlay 和
最终施加力矩可分别审计。上游约 5–11% 的模型误差与 0.2–0.5 N·m 摩擦数据只能作为历史参考，
必须在当前硬件上重新测量。

### H5：低幅虚拟弹簧

1. 在 H4 已通过的 clearance pose 锚定弹簧，先只开一个平移轴。
2. wrench 和 `J.T @ wrench` 均使用本文固定 frame；wrench/overlay 上限从最低可辨识水平开始。
3. 人工移动末端少量距离，检查 restoring force 符号、回中趋势、无持续振荡和限幅日志。
4. 重复三个平移轴，再组合三轴；暂不加入姿态弹簧。

通过条件：三个轴都只产生恢复作用，回中误差下降；功率与方向日志通过；释放、超限、错误
frame 和 watchdog 均撤销 overlay；无关节/速度/力矩越界。

### H6：低幅单侧虚拟墙

1. 在自由空间确认 wrench 为零；缓慢跨越墙面不超过经评审的小位移。
2. 检查反力始终指向允许半空间，离墙时绝不产生吸引力。
3. 重复不同法向，确认完整 wrench frame 变换没有把平移力变成错误旋转力矩。

通过条件：自由空间零反力、穿墙恢复、离墙不吸引；峰值力和变化率受限；没有持续振荡或
穿透增长。MuJoCo 的数值阈值不是实机触感阈值，H6 必须独立整定。

### H7：时序与持续运行

在低幅弹簧下至少记录 60 秒。配置目标是固定版本 YAML 的 500 Hz，但通过依据必须是实测：

- 完成周期数至少达到请求值的 90%；
- p99 cycle interval 不超过标称周期的 2 倍；
- 最大间隔必须小于实际配置的 watchdog；
- 无未解释的部分发送、反馈陈旧、限幅风暴或 fault；
- 停止后 overlay 清零且 stop receipt 完整。

这些是首轮 commissioning gate，不是硬实时证明；若 Python/SocketCAN 架构无法满足，应更换
servo owner 或降低经评审的目标频率，不能只放宽 watchdog 掩盖问题。

### H8：综合验收

仅在 H1–H7 全部通过后，才组合三轴弹簧、墙面与外部 synthetic follower wrench。仍保持
motion channel 与 haptic channel 独立，不接入 policy-human blending。通过后形成独立的
hardware acceptance receipt；该收据也不自动授权 HITL 或真实 follower 控制。

## 6. 任意阶段的立即停止条件

- 运动/力方向与预期相反，或发生不可解释的跳变、振荡、接触；
- q、qdot、tau、timestamp/age 任一非有限、缺失、倒序或超过上限；
- 单关节发送失败、部分发送、控制线程异常或其他进程争用 CAN；
- 反馈年龄达到 watchdog，或错误 frame/invalid 样本没有立即撤销 overlay；
- 限幅连续触发而操作者没有主动要求更大作用；
- 物理急停、软件 stop 或支撑装置不可用；
- 日志/commit/config 身份无法确认。

停止后先保持人员安全和机械支撑，再停止唯一控制循环、清零 overlay、执行预审 safe-stop、
确认电机状态并保存日志。没有根因和复验计划时不得直接重启同一阶段。

## 7. 验收记录最小字段

```text
stage, result, operator, observer, timestamp
AgenticRobot commit/tree, reBotArm gitlink, config path + sha256
firmware, motor model/id, CAN interface/bitrate, power/payload
loop requested/measured rate, p50/p95/p99/max interval, watchdog
per-cycle q/qdot/tau, gravity/damping/overlay/final torque
feedback valid/age, send outcome, limits, fault transitions, stop receipt
initial/final fixture metric, peak wrench/torque, human observation
known deviations, evidence path, reviewer/sign-off
```

结果必须标成 `OFFLINE_PASS`、`HARDWARE_PASS`、`FAIL` 或 `NOT_RUN`。不得用
`OFFLINE_PASS` 替代任何 H1–H8 的 `HARDWARE_PASS`。
