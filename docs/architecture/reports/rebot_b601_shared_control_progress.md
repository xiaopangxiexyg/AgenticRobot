# B601 共享控制开发进度

> 总目标和跨阶段门禁见
> [ReBot B601-RS 与 RM65 双向 Real2Sim / HITL 路线图](../plans/rebot_b601_rm65_real2sim_hitl_roadmap.md)。
> 本文只记录增量事实和证据，不声明真机验收、合并、部署或硬件安全已经完成。

## 1. 当前控制边界

目标控制链保持单一真实执行 owner：

```text
timestamped policy intent
    -> latest-value admission
    -> bounded policy-assist torque
    -> existing gravity + total-torque clip/slew
    -> existing MIT dispatch/protective hold owner
    -> measured B601 motion
    -> existing B601 -> RM65 mapping
```

Policy 不持有 CAN/backend，不直接发送 MIT 帧；人工推动也不是第二个命令源。Policy stale、
wrong-frame、无效、重放或显式 clutch 时，assist 归零，重力补偿和被动阻尼继续由原 owner
负责。J7 仍是独立通道，不进入六轴共享控制。

## 2. 已完成增量

### P0：纯控制语义

`robot_motion/rebot_b601_rs/shared_control.py` 提供：

- 带 sequence、source/receive timestamp、frame 和 validity 的 policy joint intent；
- latest-value buffer，不积压旧 action；
- target error、assist torque、assist slew 和 authority ramp 四层限制；
- 显式 policy clutch；
- override 后必须收到更高 sequence 的重新锚定指令，旧目标不能恢复；
- stale/future/invalid/replayed command 退回零 assist。

### P1：MIT overlay 适配

`robot_motion/rebot_b601_rs/shared_control_runtime.py` 将 policy intent 转换为：

```text
qref = 当前测量 q
qvel_ref = 0
vendor/software position Kp = 0（由 owner 契约保证）
joint_torque_overlay = bounded policy assist
```

该模块无 CAN、MotorBridge 或 MuJoCo 依赖。最终 gravity、driver-local damping、总 torque
clip/slew 和 dispatch 仍由实际 owner 完成。

### P2：MuJoCo 生命周期验收

现有 `sim.mujoco.rebot_b601_rs.haptic_demo` 增加 opt-in `policy-assist` fixture，没有创建
第二个 simulator/入口。Fixture 覆盖 passive、policy assist、human override、等待新锚点、
recovery 和 stale 回退。

理想 URDF plant 使用 raw-model gravity baseline，以隔离共享控制逻辑；操作者标定的
gravity scale/bias 仍属于真机 controller profile。自动验收额外要求：

- clutch 和等待重锚期间 assist 为零；
- J6 assist 不反向，避免把 fixture 新命令误看成旧目标回弹；
- J1--J5 非目标最大漂移不超过 `0.005 rad`；
- 最终状态为 passive，driver 无 fault；
- hardware claim 始终为 false。

`--output-root` 会建立唯一运行目录，将完整 JSON 写入 `result.json`，终端只显示摘要。

人工 MuJoCo 观察已经确认 J1--J5 无明显运动；第一版 recovery 反向增量造成的视觉“回弹”
已定位并移除。修复后的同向/无回拉行为仍需作为本阶段人工验收项保留。

### P3：现有真机 owner 的注入边界

`scripts/hardware/rebot_b601_rs_low_gain_probe.py` 现在允许上层以显式关键字参数注入
`B601SharedControlRuntime`。该路径：

- 只允许复用现有 `hold` owner；
- 强制 software Kp/Kd 为零；
- 禁止与 Cartesian/pose-axis controller 叠加；
- 使用当前测量 q 作为无刚度 MIT reference；
- 将 assist 放入现有 overlay，再经过原 gravity、总力矩 absolute/slew conditioner；
- assist 独立上限不得超过最终软件 torque envelope；
- 每 tick 记录 mode、authority、reason、sequence、target error 和 limit 状态；
- 普通 CLI 不构造 runtime，因此既有 hold/commissioning 行为不变。

这只是程序内接线边界，还不是可运行的真机 shared-policy application。当前没有新增 socket、
JSONL 轮询、第二个 CAN owner 或后台进程。

## 3. 当前证据边界

- 纯控制、runtime、MuJoCo 和低增益 probe 单元/契约测试通过；
- MuJoCo 只证明方向、生命周期、边界和定性运动；
- P3 尚未连接真实 policy provider，也未执行真机；
- 没有自动人工外力检测、policy replan、真实 torque sensing 或 passivity 声明；
- 当前候选不合并、不 push、不安装。

## 4. 下一步

1. 在 application leaf 中装配 policy producer、显式 policy clutch 与同进程 runtime；不得让
   producer 获得 backend/CAN handle。
2. 先用离线 scripted producer 和 MuJoCo 验证跨频率 latest-value、stale、clutch、rebase、
   无回弹和日志，再决定本地进程间传输是否真的需要。
3. 增加 shadow-only 真机观察：计算但不发送 assist，验证符号、量级、freshness 和 deadline。
4. 只有 shadow evidence 通过后，才提供前台、低权限、单关节起步的真机命令；退出仍进入原
   gravity+damping protective hold，第二次停止才允许全失能。

