# B601 真机 → RealMan 真机目标进度

## 当前基线

本任务从 `development@1c12d61a2315425dbd8161c7c4403fc94f6aef61`
（tree `df93ec405237e6568754ef2a41aac284b5828c4f`）开始。该基线已经包含：

- B601 → RealMan MuJoCo 的现有 anchor-relative residual、映射 profile 和
  唯一 MuJoCo target authority；
- RealMan → B601 MuJoCo 的只读 free-drive source、实时 push/request-response
  读取和逆向重定向；
- 现有 `RM65ArmBackend` 的 observation-only、CANFD follow、IK、速度/加速度/安全
  边界和关闭生命周期。

本任务不修改上述共享 backend，也不触碰并行 RM65 latency/IK worktree。

## P0：双端只读连通与坐标审计（本候选）

`control_station/sim_hil_profile/rebot_b601_rm65_real_connectivity.py` 是一个前台、
只读 probe：

1. 通过现有 `B601RSBackend` 读取 host-fresh 的 J1–J6 active-report，并用现有
   `B601PinocchioModel` 做 FK；
2. 通过现有 `RM65ArmBackend` 的 `observation_only=True` 读取 RealMan 当前 qpos/TCP；
3. 从现有 `b601_rm65_mapping_150_v2.json` 读取 frame transforms 和 translation gain；
4. 计算 `T_target_reference_target_controlled =
   T_target_reference_source_reference @ T_source_reference_source_controlled @
   inv(T_source_controlled_target_controlled)`，并报告映射结果与 RealMan 当前 TCP
   的位置/旋转差；
5. 记录每个样本的 B601 poll、RealMan SDK read、总耗时、反馈序号/新鲜度和失败原因。

该 probe 不调用 enable/disable/stop/Movej/Movep/MoveV，也不配置 RealMan UDP
telemetry。它只能证明连接、反馈和坐标链可观测，不能证明真机运动安全或跟随效果。

## P1：当前位姿保持（下一步）

在 P0 通过后，复用现有 `B601RM65ResidualSource` / `B601RM65MotionSession`
的 anchor/recenter 语义，新增一个 RealMan target adapter：

- source 和 frame mapping 与 MuJoCo target 完全共用；
- target 的唯一 dispatch owner 使用既有 `RM65ArmBackend`；
- 首次只发送当前 RealMan qpos 的等值保持目标，验证无跳变；
- Ctrl-C 先停止 RealMan target stream，B601 gravity/damping owner 保持运行；
- 任意 stale/错误/IK 不收敛进入 HOLD，不回退到旧的人类位姿。

## P2：极小范围动作

P1 保持稳定后，才进行 operator-authorized motion：

- 右臂、无负载、free-drive 已结束；
- 先只选 J6 或末端单轴，首轮 ±1–2°，再由操作者确认扩大到 ±5°；
- RealMan 命令使用既有 `Movej_CANFD(follow=True)`，不另造 SDK/IK；
- 目标速度/加速度沿用现有 backend 的安全边界，命令、实际 qpos、接收时间和
  dispatch receipt 全部写入独立 run 目录；
- 任何 unexpected motion 都立即冻结该 Active path，保留命令与日志，先只读排查。

## 后续顺序

`P0 只读连通 → P1 当前位姿保持 → P2 极小角度单轴 → 小范围末端位姿 →
RealMan 真机双向人工干预/policy overlay`。

不能把 MuJoCo 通过当成真机通过；每阶段分别记录 offline、sim、remote 和 hardware
evidence。所有真机命令由现场操作者在 `user` 主机前台运行。
