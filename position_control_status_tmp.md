# 位控 + 透传遥操/数采落地状态（临时）

> 范围：总结遥操、数采、sim replay、real replay/deploy 的位控/力控分链路。本文所列问题 1、2 按本轮要求暂不处理；问题 3 已落地。

## 已实现

1. **通用位控核心（sim/real 共用，位于 `robot_motion`）**
   - 定义带 `epoch/sequence`、墙钟/单调时钟、`q_ik_raw/q_ref/pose_ref` 的不可变 reference/command contract。
   - 提供因果 reference timeline，以及 75 Hz reference 到 150 Hz joint command 的插值；新目标从“最后实际下发命令”开始，避免抢占跳变，并支持 stale hold。
   - 代码：[contracts.py](robot_motion/position_control/contracts.py)、[timeline.py](robot_motion/position_control/timeline.py)、[interpolation.py](robot_motion/position_control/interpolation.py)。

2. **真机位控透传（位于 `teleop_runtime`）**
   - 默认频率拆分为：数采15 Hz、位控reference 75 Hz、CANFD joint command 150 Hz；75只由共享默认常量提供，所有运行链路读取通用`reference_hz`参数，不设45/75专属分支；力控阻抗仍保持原40 Hz backend节拍。
   - reference 只在双臂活动侧命令完整且已成功 dispatch 后发布；`pose_ref` 由最终 `q_ref` 做 FK 得到，不使用原始手柄 pose 充当标签。
   - RM65 backend 已区分 attempted / accepted / measured 状态；SDK 失败不推进 accepted anchor，后台线程有 target freshness watchdog、故障锁存和退出检查。
   - 代码：[defaults.py](teleop_runtime/defaults.py)、[reference_publisher.py](teleop_runtime/position_control/reference_publisher.py)、[rm65.py](teleop_runtime/backends/arms/rm65.py)。

3. **因果数采链路**
   - 控制更新与样本写入拆开：75 Hz 执行 `control_step()`，独立 15 Hz gate 执行 `record_step()`，不依赖固定“每三帧采一次”。
   - 15 Hz 样本严格选取 `sample_timestamp` 之前、未过期且 epoch/sequence 完整的 reference；缺失时丢弃该样本，不回退到 backend 当前命令伪造 action。
   - observation 优先使用因果硬件实测关节/FK，同时保留 commanded 状态；schema 已记录 reference、修改原因和 15 Hz transport health snapshot。
   - 代码：[runtime.py](teleop_runtime/collection/runtime.py)、[rate_gate.py](teleop_runtime/collection/rate_gate.py)、[arm_state_polling.py](teleop_runtime/collection/arm_state_polling.py)、[sample_assembler.py](teleop_runtime/collection/sample_assembler.py)、[schema_v21.py](teleop_runtime/collection/schema_v21.py)。

4. **已有安全逻辑复用与修正**
   - 保留并记录 IK 原始结果 `q_ik_raw`，最终 `q_ref` 经过碰撞、jump gate、步长限制和平滑后下发。
   - `BimanualRig` 支持 dispatch 失败后的整组 hold（当前是 fail-stop 缓解，不等于原子提交）。
   - `rm65_init_slow` 已按 `[6, 6, 7, 8, 10, 10]` 配置。
   - 代码：[dual_arm_collision_aware.py](teleop_runtime/retargeters/dual_arm_collision_aware.py)、[rig.py](robot_motion/runtime/rig.py)、[joint_velocity_limits.json](robot_motion/configs/joint_velocity_limits.json)。

5. **sim/real 通用 15→75→150 位控规划**
   - `robot_motion.position_control` 新增真实时间戳 pose/qpos 重采样：xyz 线性、quaternion 最短弧 SLERP、qpos 线性；支持抖动时间戳和非整数频率比。
   - MuJoCo replay/offline schedule 与 RM65 真机 backend 都调用同一个 `JointCommandInterpolator`；75→150 抢占锚点严格使用上一条实际 emitted/SDK accepted command，不使用未下发的理论轨迹点。75 Hz segment 起点取 IK/安全完成后的实际 qpos 发布时间，不把规划耗时错误算进插值段。
   - 代码：[resampling.py](robot_motion/position_control/resampling.py)、[interpolation.py](robot_motion/position_control/interpolation.py)、[control_schedule.py](robot_motion/replay/control_schedule.py)、[runner.py](sim/mujoco/replay/runner.py)。

6. **real replay/deploy 位控接通**
   - 正确生产链路已落为：`15 Hz target pose -> 关键帧安全限制 -> 75 Hz pose 时间插值 -> coupled dual-arm collision IK -> 绝对/每 reference 关节限位 -> 限位后 endpoint collision recheck -> 75 Hz qpos -> 公共 75→150 qcmd -> sim/real sink`。
   - 15 Hz 安全只处理 finite/quaternion、workspace、位姿/关节步长等明确限制；不会在 75 Hz 再重复应用该 gate。插值是轨迹生成，不是额外 low-pass/EMA 时序平滑。
   - offline replay 可使用已知后继关键帧，并优先保留 parquet 逐帧 timestamp；缺失、重复、倒序或非 finite 时整段显式退回 nominal action grid。online deploy 为保持因果性，明确引入一个 15 Hz action period 的延迟。手/夹爪不参与机械臂插值，仍为 15 Hz 离散值；混合 arm+hand action 共用因果 timeline，在相同的延迟后 75 Hz reference tick 逻辑调度。hand I/O 使用独立的一槽 latest-wins worker，真实拒绝会锁存 fault 并请求 arm hold；debug 记录 callback 实际时间与 backend receipt，不把“同一逻辑 tick”误写成物理同时生效。该机制是 fail-stop 而非跨设备原子事务：故障被发现时若一个 arm callback 已进入 backend，最多仍可能完成该次调用，随后记录 `accepted_before_async_abort` 并 hold；backend 必须保证有限 I/O timeout。
   - replay/deploy 不再把 75 Hz Cartesian pose 交给两个 backend 各自做单臂 IK；公共 planner 从 scene URDF fixed mount chain 取得包含 `±0.2625 m` 的 `world_T_arm_base`，并使用 backend 启动后实际 active TCP。初始化必须有完整 left+right 权威 qpos；position 真机单臂模式和缺失被动臂状态都会 fail-closed，不猜默认姿态。
   - 真机 planner reset 只接受本轮成功且新鲜的双臂 hardware measurement，不会在读取失败后退回 cached command/qpos；deploy/replay 默认限制测量年龄 0.25 s、双臂测量 skew 0.05 s。75 Hz coupled qpos 标记为 prelimited，backend 若单臂重限位、singularity hold 或改写 final target 会在 hand 入队/下发前 fault 并请求 group hold。arm queue 前后和 hand I/O 前均检查 150 Hz background health。
   - 当前已认证的是“所选 scene profile 中最终限位后的 75 Hz qpos endpoint”；75→150 插值整段仍未做连续碰撞认证。默认 fixed hand/link proxy 也不等于任意自定义工具几何已完整认证。该剩余项仍属于暂缓的问题 2。
   - impedance replay/deploy 保持现有 action-rate -> 40 Hz backend 分支。
   - 代码：[safety.py](robot_motion/position_control/safety.py)、[position_collision.py](robot_motion/rm65/position_collision.py)、[executor.py](teleop_runtime/replay/executor.py)、[deploy_executor.py](teleop_runtime/position_control/deploy_executor.py)、[run_policy.py](teleop_runtime/deploy/run_policy.py)。

7. **问题 3：完整高频审计闭环**
   - `--control-debug {off,summary,full}` 为纯观测选项，不会启用/改变 smoothing、hold、奇异保护、碰撞、安全限位或 dispatch。
   - 记录全部 75 Hz reference、150 Hz scheduler/SDK tick 与主循环阶段 timing；summary 提供 p50/p95/p99/max、deadline/skip、gap/Hz、SDK success/failure/stale/fault、queue high-water/drop。
   - writer 使用有界 `put_nowait` 队列；session 与 formal episode recorder 可动态 fanout，单一 sink 出错不会影响控制。
   - full sidecar 为 `position_reference.jsonl`、`position_command_150.jsonl`、`position_timing.jsonl`；summary/full 均写带 sanitized execution context 的 `position_summary.json`，off 不创建 recorder。reference文件/event名不编码频率；实际目标/实测 Hz 以 summary context、percentile 和事件 metadata 为准。
   - replay 的 collision debug 另有进程内 512 条有界 ring，记录 raw/final qpos、target/realized pose、distance attempts、solve/budget、reset freshness，以及 arm 前、arm queue 后和实际 hand 下发前三个 transport-health 槽；无 hand 的 tick 第三槽为 `not_applicable`，gate 异常也会先记录再 fail-closed。`/api/state` 返回 last/计数，`/api/position-collision-debug?limit=N` 返回有界事件；当前不会自动写磁盘 sidecar，也不声称 ring 已保存完整 backend receipt。未实际成功检查或 replay 已 fault 时不会报告 endpoint 已认证。
   - 代码：[audit.py](robot_motion/position_control/audit.py)、[debug_recorder.py](teleop_runtime/position_control/debug_recorder.py)、[reference_publisher.py](teleop_runtime/position_control/reference_publisher.py)、[rm65.py](teleop_runtime/backends/arms/rm65.py)。

8. **可选的速度连续 75→150 Hz 遥操规划与真机前安全审计**
   - 默认 `linear` 未改变；只有显式使用 `--position-command-planner acceleration_limited` 才启用共享在线规划器。规划状态以最后 SDK accepted `qcmd` 为锚，目标抢占继承当前速度，逐关节限制 `qd/qdd/单步/已接受端点包络`。
   - coupled IK 在该模式下从左右 backend 最后 accepted `qcmd` 同步，不再从可能领先真实执行的 preview qref 同步。反向抢占期间保留运动方向上的既有安全制动端点，避免新75 Hz目标过早收缩包络而强迫150 Hz单周期突破加速度上限。
   - manual/collision/no-valid-solution/stale 继续软件快速固定最后 accepted qpos；奇异区入口目标已由后端替换为当前qpos，规划器按异常目标10%速度受qdd约束减速。SDK/后台线程故障会锁存并由失败线程直接尝试 `Arm_Slow_Stop`；执行过运动命令的position backend在断连前也会减速停止。
   - 双臂碰撞求解器被请求但初始化失败时现在 fail-closed；只有显式 `--no-collision` 才允许无双臂防碰撞运行。速度规划同时拒绝 `command_hz < reference_hz`，防止安全reference尚未产生任何qcmd就被下一reference抢占。
   - full debug新增 braking guard、实际安全包络、endpoint velocity clamp、safety infeasible和qdd-limit-after-clamp字段。正常受控运动中后两项必须持续为false。
   - 代码：[trajectory.py](robot_motion/position_control/trajectory.py)、[rm65.py](teleop_runtime/backends/arms/rm65.py)、[run_vive_rm65_real_teleop.py](teleop_runtime/debug/run_vive_rm65_real_teleop.py)。

主要回归入口：[test_robot_motion_rm65_position_collision.py](tests/test_robot_motion_rm65_position_collision.py)、[test_robot_motion_position_control.py](tests/test_robot_motion_position_control.py)、[test_replay_execution_checks.py](tests/test_replay_execution_checks.py)、[test_position_deploy_executor.py](tests/test_position_deploy_executor.py)、[test_position_reference_publisher.py](tests/test_position_reference_publisher.py)、[test_position_collection_measured_state.py](tests/test_position_collection_measured_state.py)、[test_arm_endpoint_action.py](tests/test_arm_endpoint_action.py)、[test_rm65_impedance.py](tests/test_rm65_impedance.py)。通过数量必须以当前工作树的实际运行结果为准，本文不固化历史 passed 数；尚未执行真机运动验证。

## 本轮明确暂缓的两个问题

### 1. 双臂下发还不是真正的组级原子事务

当前 [rig.py](robot_motion/runtime/rig.py) 可并行调用左右 backend，并在任一失败后请求整组 hold；但两个 [rm65.py](teleop_runtime/backends/arms/rm65.py) backend 仍有各自的 mailbox、150 Hz worker 和 accepted 状态。因此可能出现“一侧已经接受新目标，另一侧拒绝，然后再 hold”的短暂部分提交。

online deploy 的 arm/hand 独立 worker 也属于同一类 fail-stop 边界：fault gate 能阻止新的 arm callback 进入，但不能撤销已经进入 backend 的单次调用；这一窗口已完整记录，本轮按“前两个问题暂缓”不扩展为跨设备原子 coordinator。

需要增加双臂 position transport coordinator：先统一 validate/prepare，再用共享 `group_epoch/group_sequence/start_time` commit；150 Hz 使用同一个 tick/alpha，任一侧发送失败立即锁存 group fault 并整组 hold。SDK 报文无法做到物理同一时刻，但必须做到逻辑事务一致，并记录左右实际发送时间差。

### 2. 75 Hz 安全端点不保证 150 Hz 插值线段安全

当前遥操作 [dual_arm_collision_aware.py](teleop_runtime/retargeters/dual_arm_collision_aware.py) 与 replay/deploy [position_collision.py](robot_motion/rm65/position_collision.py) 都会检查本次最终 `q_ref` endpoint；[rm65.py](teleop_runtime/backends/arms/rm65.py) 再从最后 accepted command 对 `q_ref` 做关节空间线性插值。碰撞自由空间通常非凸，所以起点、终点都安全并不代表中间路径安全；reference 抢占还会改变线段起点。

real replay/deploy 的双臂 75 Hz endpoint checker 已接通；剩余缺口仅是相邻 endpoint 之间的连续线段认证。因此不能把当前 endpoint-only contract 表述成完整连续轨迹碰撞认证。

需要在每次接受/抢占 reference 时，对“最后实际 emitted qcmd -> 新 qref”做双臂整段认证（自适应采样或连续碰撞检测）。若不安全，应限制到最大安全 `alpha`、边界滑动或整组 hold，并记录最小 clearance、首个 unsafe alpha 和修改原因。该检查必须满足75 Hz的13.33 ms预算。

## 已落地但仍需真机数据验证的实时预算项

问题 3 的观测闭环已经实现，但“是否必须进一步拆75 Hz control worker、15 Hz sample worker和低优先级 UI”必须由真实审计结果决定。当前遥操作主循环仍包含输入、IK/碰撞、dispatch 与部分 UI/采样工作；新增数据可以直接判断13.33 ms deadline是否稳定满足，但本轮未在未获得运动授权的情况下执行真机验证。

## 建议落地顺序

1. 本轮先使用 `--control-debug summary/full` 做无控制语义改变的低风险审计；依据 p95/p99、deadline miss、queue drop 决定是否拆线程。
2. 后续另行处理双臂 transport coordinator（组级事务与统一 150 Hz tick）。
3. 后续另行处理 75→150 插值整段连续碰撞认证；replay/deploy 的 75 Hz endpoint gate 已完成。
4. 本地/仿真验证通过后，仍需用户在新一轮明确授权具体 L4/L5 动作，再做低速真机验证。
