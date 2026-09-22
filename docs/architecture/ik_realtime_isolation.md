# IK 实时隔离：当前边界与探索性实现

本文记录 CPU 争用下 IK 相关路径的统一处理边界。目标不是降低既有安全标准，
而是把“计算 proposal”和“准入、reference、dispatch”拆开，使昂贵计算不能形成
FIFO 积压或恢复后的追赶突发。

## 已落地的公共合同

`robot_motion.rm65.ik_worker.ThreadedBimanualIKProposalWorker` 是一个默认不启用、
不拥有动作权威的 coupled-IK proposal worker：

- mutable solver 只由 worker 线程访问；
- 最多一个 in-flight 和一个 pending request，新的 pending 覆盖旧值；
- 每个 request 绑定 worker generation、control epoch、anchor generation、输入状态时间、
  有效截止时间和精确的双臂 reference qpos；
- reset 是高优先级生命周期屏障，旧 generation completion 不能重新进入新 epoch；
- completion 只表示计算完成，不能绕过既有 collision、joint-limit、reference、dispatch
  和硬件回执；
- completion assessment 必须同时提供当前 reference qpos 和显式允许的最大 anchor drift；
  任一侧超限都会在进入 canonical admission 前被拒绝；
- 可选 worker CPU affinity 必须实际验证，设置失败时启动 fail closed。

该 worker 已用 fake solver 覆盖 latest-only、reset/epoch、deadline、错误和防御性复制，
并用真实 Pinocchio/Pink/qpsolvers、RM65 URDF 和 coupled collision retargeter 完成一次
同姿态 proposal。它证明 solver 可以被单独所有，但不等价于真机接线已经安全。

## 全部 IK 路径清单

| 路径 | 当前 IK 所有者与时序 | 本分支处理 | 下一接入门槛 |
| --- | --- | --- | --- |
| 实时 VIVE 遥操 / collection | `DualArmCollisionAwareRetargeter.retarget()` 在 `TeleopSession._build_command()` 中同步执行；real loop 每帧又从 transport accepted qcmd 同步 solver | 公共 latest-only worker 可复用；未启用 | worker completion 必须针对**当前** accepted-qcmd 重新核验 collision/step anchor；不能仅凭“上一帧求解成功”接入 |
| real Skill / Gateway | `RealAgentMotionPort.command_for_observation()` 同步调用同一 target builder 的 FK、TCP/solver-frame 转换和 coupled solve | 明确保留 lower owner、HOLD 和 dispatch 权威 | target builder 目前同时持有 mutable FK/solver state；必须先把 worker-owned planner 与 control-thread measured-pose projection 分离，不能跨线程共享 Pinocchio/Pink data |
| position deploy / RTC | `PositionDeployExecutor` 的 reference worker 与 `OfficialAllIKGovernor` 已有独立 cadence、deadline、affinity 和实验性 precompute | 不重复修改；避免与活跃 RTC 分支语义重叠 | 统一 proposal lineage 时复用 generation/epoch/anchor/expiry；正式默认仍由现有 canonical planner 和 reference worker 决定 |
| hardware replay | `RM65PositionCollisionPlanner` 逐 reference 同步规划；impedance replay 逐 source frame 同步发送 | 两条 replay 路径均增加 `stretch_no_catchup_v1` dispatch floor；过载时拉长回放，不补发历史点 | 若异步化 planner，仍须按顺序提交 collision-limited state，不能跳过中间 admitted anchor |
| MuJoCo deploy / Skill / task | `RM65CanonicalIK`、`RM65RetargeterIK` 和 `MujocoCoupledPoseToQposResolver` 同步求解 | 保持同步，作为离线/仿真基线 | real-time HIL 才需要接 worker；普通 faster/slower-than-real-time 仿真不应伪装成硬实时验收 |
| 单臂 Cartesian backend | `RM65ArmBackend._solve_cartesian_command()` 在 backend command 路径按 TCP frame 懒加载并同步求解 | 未改 | 高频正式路径应优先接收已准入 joint reference；若必须保留 Cartesian backend IK，需要独立的单臂 proposal worker 和同等 freshness/anchor 合同 |
| debug / benchmark / trace replay | 多数直接调用 coupled solver，用于离线比较 | 保持同步 | 不得以 debug 路径的吞吐代替 control-thread deadline 证据 |

## 为什么尚未直接打开实时遥操异步 IK

当前 position 控制的关键安全事实是：coupled solver 每个 outer tick 都从两侧 transport
实际接受的 qcmd 重新同步。后台 worker 的结果天然晚于其输入；此时 120/150 Hz publisher
可能已经沿轨迹移动。如果直接复用旧结果，会出现以下任一问题：

1. 碰撞距离和步长证书相对于旧 qcmd，而不是当前 transport anchor；
2. 为避免陈旧结果而严格比较 qcmd 时，运动期间几乎所有晚一帧结果都会被拒绝；
3. 让主线程等待 worker 重新同步又恢复了原来的阻塞问题；
4. 在主线程调用 FK、worker 同时调用 Pink/Pinocchio solve，则共享 mutable data，线程安全
   没有成立。

因此正式接线需要一个新的 admission seam：worker 独占 planner；control thread 提交 immutable
状态/目标，收到 completion 后以当前 anchor 做低成本、确定性的最终证书复核，再交给现有
reference governor。复核失败必须继续旧轨迹并按现有 stale/HOLD 规则减速，不能同步补算。

## CPU 资源合同

OpenPI launcher 的 `--exclude-cpu` 继续负责从 server affinity mask 排除控制核；新增
`--native-thread-count` 在 `exec` 前统一限制 OMP、MKL、OpenBLAS、NumExpr、BLIS 和
Accelerate 线程池。两个参数均默认关闭，不改变旧启动行为。

部署时还必须遵守：

- 控制核及其 SMT sibling 成组预留；仅把两个 logical CPU 分给不同进程不代表物理隔离；
- solver、model、collection、DMTAC 和 writer 使用不同资源组；
- affinity 只说明“允许在哪些 CPU 运行”，系统级独占仍由 cpuset/cgroup 保证；
- 以 command/reference gap、solution age、deadline miss 和 run-queue 尾延迟验收，不以平均
  CPU 利用率代替。

## 真机前的剩余步骤

1. 在 no-action real frontend 中加入 current-anchor completion 复核和 proposal-only trace；
2. 证明 worker 超时、generation reset、VIVE stale、双臂 qcmd skew 时均不会进入 reference；
3. 对现有 120/150 Hz publisher 做训练/推理并发压力测试，确认不 catch up，且 stale target
   进入既有 quick-stop/HOLD；
4. shadow 证据通过后，再由操作者显式开启低速、小工作空间真机试验。

在第 1～3 步完成前，本分支没有启用任何新的硬件动作路径。
