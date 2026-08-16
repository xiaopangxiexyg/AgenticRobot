# teleop_runtime.deploy

> 本页维护 deploy 的完整模块参数和 contract。当前部署运行入口统一从
> [`docs/real/README.md`](../../docs/real/README.md#3-deploy) 复制。

`teleop_runtime.deploy` 是独立于采集主循环和左臂实验模式的策略部署入口，面向 RM65 单臂/双臂，可选 InspireTac 手、DMTac 1D gripper 和 RealSense。默认 dry-run；真实下发必须显式加 `--send-actions`，且默认要求先用数据集帧 qpos 初始化。

正式双臂 position deploy 当前默认使用：

```text
official source all-IK
→ collision-safe nearest branch
→ linear source-q reference
→ formal third-order jerk-limited source-cut governor
→ 0.05 s acceleration-to-jerk ramp
```

对应参数为
`--position-path-lift-backend official_source_all_ik_causal_governor`
（默认值）、`--position-all-ik-lookahead-source-steps 2` 和
`--position-all-ik-jerk-ramp-sec 0.05`，以及默认的
`--position-continuous-waypoint-profile formal_third_order`。真实部署 trace 显示，
`async_c2` 的后台预计算在整机负载下可与权威 60 Hz reference worker 竞争，
导致跨过两个 16.7 ms tick。因此 C2 已恢复为显式研究 profile；它只在完整
15 Hz source cut 具有同 chunk 未来覆盖、连续时间/离散动态证书和整段采样碰撞
准入时接管；worker 迟到、anchor 变化或任一证书失败时，整个 cut 回退为
third-order governor，不在 cut 内混切。它要求 `--debug`、双臂、position collision、
15 Hz policy action、120 Hz RM65 command stream 以及 same-chunk lookahead；
不满足这些条件时显式请求会 fail closed。单臂兼容运行和显式关闭 chunk
lookahead 的旧命令会保留在 Pink 路径。

RTC frontend 和带 `arm_hold_output` 的 policy artifact 没有可直接用于规划的、已完成语义
解析的 same-chunk successor，因此会在 backend/profile 解析前关闭 lookahead、回退到
`pink_pose_lift`，并把 C2 effective profile 标为 `formal_third_order`（即 C2 关闭）。不得为
获得 C2 而直接暴露未解析的未来 arm-hold action。

要显式固定当前正式 source-q third-order 执行，传：

```bash
--position-continuous-waypoint-profile formal_third_order
```

该配置不改变 official all-IK 分支选择、collision owner、15/60/120 Hz 频率或其他
deploy 参数。`async_c2`、历史
`--position-continuous-waypoint-experimental-admission`与
`--position-continuous-waypoint-async-precompute`只保留给隔离的 debug/离线研究对照，
不是当前真机推荐；它们与 `--send-actions` 的组合会在设备创建前 fail closed。

async C2 的 deadline 复核可显式把权威 reference worker 与 proposal worker 固定到
两个不同的 Linux logical CPU：

```bash
--position-reference-worker-cpu 14 \
--position-continuous-waypoint-worker-cpu 15
```

两个参数必须成对出现、CPU 必须不同且属于进程启动时可用的 affinity mask；worker
启动后还会按 native thread ID 重新读取实际 mask，不能精确验证为单 CPU 时 fail
closed。远端部署机的 2026-08-09 只读审计显示 CPU 14/30 与 15/31 分别为两组不同
物理核的 SMT siblings，因此隔离对照使用 14（60 Hz reference）和 15（C2）。这只保证
两条指定 worker 不共享同一个 logical CPU，不等价于系统级 exclusive cpuset；同机其他
进程仍需由部署环境隔离。删除这两个参数即可原子回退为原调度行为，删除
`--position-continuous-waypoint-profile async_c2` 或改回 `formal_third_order` 则同时回退
C2 路径。CPU 隔离在 no-action/full-load deadline soak 通过前不解除上述真机下发门禁。

旧 V4 的完整回退是：

```bash
--position-path-lift-backend pink_pose_lift \
--pose-lift-profile rm65_pose_lift_v4_qp_constrained
```

这里的 Pose Lift V4 是部署/replay profile，不是 live teleop 的
`POSITION_FORMAL_V1/V2/V3` profile；不要把两套版本号混用。

当前推荐的 shell 入口是 `python -m teleop_runtime.deploy`。它的薄包装
`teleop_runtime.deploy.cli.main` 会委托给同一份 `run_policy.main` 运行时；
`python -m teleop_runtime.deploy.run_policy` 和 `run_policy.main(...)` 不是“已不用”的
遗留实现，而是当前仍受支持的直接/程序化入口。

## 入口支持、责任与兼容期

| 入口 | 当前状态 | 责任边界 | 替代/迁移关系 |
| --- | --- | --- | --- |
| `python -m teleop_runtime.deploy` / `teleop_runtime.deploy.cli.main` | 推荐的 shell CLI | `teleop_runtime.deploy` 真机部署运行时 | 调用同一 `run_policy.main`，不另建控制环。 |
| `python -m teleop_runtime.deploy.run_policy` / `run_policy.main` | **受支持**的直接与程序化入口 | `teleop_runtime.deploy` 真机部署运行时 | OpenPI real launcher 当前直接依赖；未来若收敛调用面，应迁向上述 canonical CLI wrapper 或其稳定 API。 |
| `python -m deployments.openpi.client.real` | **受支持**的 OpenPI 历史命令入口 | `teleop_runtime.deploy.openpi_application` 负责真实部署装配；`deployments.openpi.client` 只负责 OpenPI transport/adapter | 兼容模块只转发到现有 `run_policy.main` 控制循环。 |

本仓库目前没有 `CODEOWNERS` 或个人维护人清单；上表的模块责任边界就是当前可审计的
owner 记录。`run_policy` 没有已宣布的结束日期，兼容期持续到所有直接调用方完成迁移、
替代 API/入口与迁移文档发布，并经过明确的弃用期为止。在此之前不得删除、降级或仅按
“兼容遗留”处理该入口。任何封存变更必须同时更新 OpenPI launcher、CLI/导入契约测试和本节。

## 当前执行链路

deploy 主线已经收敛到 canonical action path：

```text
policy output (15 Hz)
  -> NormalizedAction
  -> filter_normalized（15 Hz keyframe safety，仅一次）
  -> position: available-time-aware causal pose resampling（60 Hz）
  -> coupled dual-arm collision IK + joint/step limits + endpoint recheck（60 Hz qref）
  -> RobotAction / RobotActionResolver / joint TeleopCommand
  -> real launcher dispatch
  -> common qpos execution planner / RM65 position backend qcmd stream（120 Hz 默认）
```

代码边界：

- `robot_motion/action`: dense action vector 到 `RobotAction` 的 sim/real 通用适配层。
- `robot_motion/deploy`: policy action 过滤、canonical `RobotAction` 构造和 `TeleopCommand` 解析；`teleop_runtime.deploy` 中同名模块仅为兼容 re-export。
- `robot_motion/position_control/resampling.py`: sim/real 通用真实时间戳 pose/qpos 重采样。
- `robot_motion/rm65/position_collision.py`: sim/real 通用双臂 pose→qpos、绝对/
  reference 步长限位，以及限位后 endpoint collision recheck；不含时序平滑。
- `robot_motion/position_control/interpolation.py`: sim/real 共用 qref→qcmd
  planner；sim/real 只在最后 qcmd sink 分叉。
- `teleop_runtime/position_control/deploy_executor.py`: real-only 在线 60 Hz worker、chunk 编排、暂停/故障 hold 与 backend dispatch 边界。
- `teleop_runtime/deploy/arm_state_reader.py`: deployment-facing arm 状态刷新节拍和 bounded causal snapshot history；backend 继续持有设备状态/FK，reader 不持有控制、安全或 dispatch。
- `teleop_runtime/deploy/camera_runtime.py`: policy artifact 相机键绑定、部署侧 source 构造与启动准入适配；正式 RealSense 启动/重试和 freshness gate 继续复用 `teleop_runtime.real_teleop.camera_runtime`，本模块不另建相机生命周期所有者。
- `teleop_runtime/deploy/dispatch.py`: 当前真机 launcher 的 `TeleopCommand` 到 arm/hand backend 分发；这是 real-specific 代码，不应迁到 `robot_motion`，sim launcher 应实现自己的 dispatcher。
- `teleop_runtime/deploy/actions.py`: 仍保留旧 `ActionSafetyFilter.filter()` direct-command path 作为 legacy 封存入口；新代码应使用 `filter_normalized()`。

因此模型部署、dataset replay debug 和未来 sim deploy 都应以 `RobotAction` 作为语义边界，
以 `TeleopCommand` 作为下发边界，不再让 policy adapter 直接生成 backend command。

正式 prediction 默认启用 `--policy-observation-firewall-mode strict`：模型输入先投影为与
artifact/layout 绑定的 `PolicyObservationView`，旧 object-style policy 也只会收到由该 view 重建的净化兼容
对象，不能读取 runtime fault、authority、safety、writer/backend 或任意 frontend metadata。
`dual-validate` 仅用于无动作兼容检查，与 `--send-actions` 同时使用会直接拒绝。

### RTC-3c default-off virtual shadow

`--rtc-profile rtc_3c_virtual_shadow_15_30_60_120_v1`提供第一版增量RTC接线。它只允许builtin `hold`
虚拟policy，并强制`send_actions=false`、15 Hz observation request、30 Hz decision/action、60 Hz position
reference、120 Hz command cadence。该profile只在RTC分支关闭position chunk lookahead；未指定
`--rtc-profile`时默认仍是`off`，既有`sync_queue_v1`的replace/append、逐项pop、same-chunk lookahead与
`PolicyActionEmission`语义保持不变。

RTC coordinator在control thread上按“poll completion → matching-only result admission → 最多pop一个due
keyframe → 同一个正式`PolicyActionEmission`/canonical preparation → current-only commit-through →
coverage-triggered request”的顺序执行。commit transition绑定同一个pop receipt与emission identity，只接受已pop且
已形成正式emission的当前target；future successor无法构造合法due-pop receipt，并另有`target_not_popped`防御检查。
blocking inference只运行在capacity-1 request、bounded result、single-in-flight worker中；HOLD/FAULT先使producer
epoch/plan失效，旧worker session drain后才允许显式rearm。RTC timed plan不会写入legacy action deque，也不会把
尚未正式emission的future successor提前交给position resampler。

当前状态仅为`rtc_3c_virtual_shadow_ready_local_only`：本地virtual-policy long-run/lifecycle fixture不初始化设备；
production application只检查过`--help`，没有以RTC profile启动camera/backend。它不是OpenPI true RTC、真实
observation/transport shadow、typed preview receipt、M1C drain或motion authority。RTC-3d完成typed preview
identity/TTL/safety/downstream acceptance/reference lineage/promotion之前，RTC lookahead必须保持关闭。

现有control loop提供默认关闭、仅供application root注入的`RtcPolicyEmissionPublisher`。正式
`publish_committed_emission()`验证typed producer commit、active source binding、producer epoch、TTL和幂等publication
key后才发布`PolicyActionEmission`。observer queue满时只drop并计数；首版最多一个required control subscriber，其
queue满或异常会拒绝publication并使publisher fail closed。该commit只证明pop后的动作已固化为emission，不代表
safety/reference/backend接纳。

producer缓存stable descriptor对应的epoch binding candidate；publisher拥有active binding。rearm生成新binding和
backend-neutral activation-readiness receipt。首版publisher支持显式suspend、单required subscriber typed ACK及
原子rebind；具体application/consumer自动协调和多subscriber rotation仍是后续gate。旧
`PolicyActionEmissionFanout`只保留passive/debug兼容，`RtcFrontendStepFanout`（canonical alias
`RtcProducerDebugStepFanout`）只传producer诊断事实。它们都不拥有worker、plan、execution或motion authority。

RTC配合`--debug`时默认启用低频`summary`事件；`--rtc-debug-level trace`才记录每个RTC control tick。
`--rtc-debug-level off`可只保留普通deploy debug。RTC事件经`--rtc-debug-queue-capacity`指定的bounded observer queue
异步写入，满队列只增加drop count，不反压control loop或model worker；退出时记录
`rtc_debug_observer_status`。当前事件覆盖source binding、transport exchange、request/result/worker/plan摘要、
emission commit、publisher publication和frontend close，且只记录identity/hash/timing，不复制sensor或完整chunk。

RTC-4a另在`robot_motion.deploy.rtc_protocol`与`deployments.client.rtc`提供模型/transport/sim/real无关的
capability、envelope、conditioning receipt和local/remote backend seam。它目前只有纯本地迁移fixture，没有接到本
RTC-3c frontend；因此不改变默认同步路径、RTC-3c限制、worker/plan owner或任何设备行为。RTC-4b-A已另在
`deployments.openpi.client.rtc`增加只支持`continuation=none`的离线final-chunk adapter，并以fake transport/policy
验证artifact、server metadata、exact OpenPI payload、continuous/int64 discrete output与V22 codec。该adapter没有从本
runtime导入或注册caller，也不启动server、相机、仿真或设备；RTC-3c仍使用builtin hold。下一步RTC-4b-B必须先完成具体
模型侧prefix→suffix continuation及可验证conditioning receipt，不能把RTC-4b-A表述为true RTC。

### M3 passive ActionArbiter shadow

`--action-arbiter-mode shadow` 仅在 dry-run 或 dataset replay 调试中启用单策略、非权威的
ActionArbiter 对照。它被动消费现有 `prepare_next_action()` 唯一 queue pop 产生的同一个
`PolicyActionEmission`，生成 `would_select` evaluation；现有 canonical safety、command 和 legacy
execution 仍是唯一执行事实。shadow 不持有 action queue、不申请 dispatch owner token，也不会调用
backend。启用 `--send-actions` 时该模式会 fail closed；正式真机 shadow capture 要等 M4B durable trace
完成。配合 `--debug` 时，当前临时 JSONL 会记录 `arbitration_shadow_step` 和
`legacy_execution_observation`。非 replay policy 在成功生成 shadow evaluation 时，还会从同一个 emission、
同一个 observation 和已经由旧 owner 计算出的 post-safety result 生成
`policy_safety_shadow_comparison`：它只重算 shared arm pose/qpos numeric primitive，不再次调用旧 filter，
并把 frame/controlled-frame 等 representation 差异单独记录。该 observer 不推进 anchor、不消费 action queue、
不修改 fault/last-action/dispatch；异常只记录 `policy_safety_shadow_error`。这些 JSONL 仍不是 v2.4 durable
ControlTrace。

### v2.3 absolute/relative 虚拟等价验收

不连接设备即可运行双路部署验收：

```bash
PYTHONPATH=. python -m pytest tests/test_v23_virtual_deploy_equivalence.py -q -s
```

该测试使用双臂、3 个 chunk、每个 chunk 4 帧，并为左右臂配置不同的非恒等
runtime raw→canonical desk 旋转和平移。absolute 分支和 relative 分支都经过生产部署使用的
policy observation frame adapter、policy output normalization、chunk decoder、canonical safety、
`RobotAction`/`TeleopCommand` 和 real-runtime dispatcher；dispatcher 的终点是内存 mock backend，
不会初始化或操作硬件。验收要求两路最终 raw TCP command 的平移、旋转和 4×4 matrix 误差均
小于 `2e-6`。测试还会故意用下一帧 measured TCP 解码 relative chunk，要求产生大于 `5 mm`
的误差；该负对照用于证明测试能够发现错误 anchor，而不是只通过恒等变换。

通过时会输出一行 `V23_VIRTUAL_DEPLOY_REPORT=...`。这证明两种 pose representation 在软件
部署边界等价，但不替代真机 A/B：RM65 SDK、IK 数值解、position collision planner、通信时序
和机械跟踪误差仍需在机械臂在线后单独验收。

### v2.2 raw→v2.3 relative 真机 A/B 闭环验收

该验收只使用同一段已确认安全的 v2.2 impedance episode。两次运行必须使用相同代码、
数据集、frame range、chunk size、初始化帧、tool profile 和安全参数；唯一允许变化的是
`absolute/relative` 两个 pose-mode 参数。以下命令直接在真机所在远端仓库运行，不使用
`remote_python`。

先确认双臂在线、工作区无人/无障碍、急停可用，并让操作员全程观察。绝对分支：

```bash
cd /home/user/zc/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"

python -m teleop_runtime.deploy \
  --send-actions \
  --active-sides left right \
  --end-effector-sides none \
  --camera-mode none \
  --left-arm-ip 192.168.101.21 --right-arm-ip 192.168.1.20 \
  --arm-control-mode impedance \
  --debug-replay-dataset /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --debug-replay-episode 0 \
  --debug-replay-start-index 0 --debug-replay-end-index 20 \
  --debug-replay-chunk-size 4 \
  --init-source replay \
  --policy-action-pose-mode absolute \
  --debug-replay-output-pose-mode absolute \
  --debug --debug-dir /home/user/zc/DexTeleop/logs/v23_real_pair/absolute \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --policy-qpos-init-episode 0 --policy-qpos-init-frame 0 \
  --auto-start --no-wait-for-ready-confirm --max-steps 600
```

机械臂完全停止后，再运行 relative 分支；除下面三行外保持参数完全相同：

```bash
  --policy-action-pose-mode relative \
  --debug-replay-output-pose-mode relative \
  --debug --debug-dir /home/user/zc/DexTeleop/logs/v23_real_pair/relative
```

当前 `--init-source replay` 会把第 0 帧用于 startup target 并从 policy replay 中跳过，因此
`[0,20)` 正常应执行并对齐 source action index `1..19`，共 19 个 action。两次都完成后执行
只读比对；目录参数会自动选择其中最新的 `deploy_debug_*.jsonl`：

```bash
python -m teleop_runtime.deploy.acceptance \
  /home/user/zc/DexTeleop/logs/v23_real_pair/absolute \
  /home/user/zc/DexTeleop/logs/v23_real_pair/relative \
  --expected-actions 19 \
  --output /home/user/zc/DexTeleop/logs/v23_real_pair/acceptance.json
```

退出码 `0` 且报告 `passed=true` 才通过。验收器按 dataset source action index 对齐，不受启动
IK recovery rollback/chunk-id 变化影响，并分别检查：

- relative 解码后的 runtime-absolute target 与 absolute target：默认 `2e-6 m / 0.002°`；
- canonical command target：默认 `2 mm / 0.5°`；
- 两次 measured TCP 轨迹：首轮建议门槛 `30 mm / 5°`；
- 两次 measured qpos：首轮建议最大单关节差 `0.15 rad`；
- source index 集合完全相同，且无 policy/dispatch/cleanup fault。

前两项是 representation/command 硬门槛；后两项包含两次独立真机运行的跟踪和初始化误差，
当前只是保守首轮门槛。首轮真实分布确认后应通过 CLI 参数收紧，不能把放宽 measured 门槛
用于掩盖 decoded/command 不一致。旧日志没有 `debug_replay_source_action_index` 时 fail-closed，
必须用当前代码重新运行。

本 episode metadata 声明为 impedance，不能拿它绕过 mode binding 去验收 position。position
分支需要另选一段已经确认安全、metadata 明确为 position 的 v2.2/v2.3 episode，再以同样 A/B
方式运行；position 报告还必须没有 `position_executor_*fault`、dispatch reject、collision hold
或 debug-event drop。验收器只是日志读取器，不初始化 backend、不访问设备、不下发命令。

当前目录边界为：

- `robot_motion`：`RobotAction` schema/spec、vector adapter、canonical deploy preparation、value-level safety、时间戳 motion resampling。
- `teleop_runtime`：CLI、键盘、相机/触觉、真机 backend、在线因果 worker、`dispatch.py`、fixed-init 真机执行。
- `sim`：MuJoCo/Sapien 等仿真专属 scene/controller/dispatcher；复用 `robot_motion` 的 action/resampling/schedule。

部署频率与 action chunk 执行：

- position deploy 默认使用正式 profile `rm65_bimanual_position_v1`：15 Hz policy
  action、60 Hz pose-reference/IK、120 Hz command stream。`position` 和
  `position_45_150hz` 仅作显式兼容 profile；`impedance_40hz` 仍属于独立的力控链路。
  `--arm-control-mode`、`--position-reference-hz` 和 `--arm-control-hz` 可显式覆盖。
- chunk 内部已知下一个 action 时，通用 `robot_motion.position_control`
  resampler 使用“逻辑动作时间 + 实际可用时间”做因果线性/SLERP 插值，
  不再额外落后一个 15 Hz action period。chunk 最后一帧因为没有右端点而 hold；
  新 chunk 到达后用一个 15 Hz period 过渡到它的首帧。回退时可传
  `--no-position-chunk-lookahead` 恢复旧的固定一 action-period 延迟链路。
- 这一层只重采样 policy 给出的主 action，不重复 tracker inactive、
  tracker 低速滤波、post-IK adaptive EMA 等数采前处理，避免改变模型学到的 action 语义。
- `--position-collision-scene-profile`、`--position-collision-min-distance-m` 和
  `--position-collision-distance-epsilon-m` 配置 60 Hz endpoint checker；真实 position
  输出不能使用 `--no-position-collision-check`，该开关只保留给非运动 legacy dry-run。
- 60 Hz qref joint safety step 使用
  `--position-qref-max-angvel / --position-reference-hz`；未设置独立 qref 上限时复用
  `--arm-max-angvel`，并与
  `--max-arm-joint-step-rad` 取更严格者；限位后必须重新做 collision distance check。
- position deploy 的 coupled IK 迭代策略由 shared
  `robot_motion.rm65.RM65PositionCollisionPlannerConfig` 持有；默认
  `--position-ik-solve-min-iterations 2`、
  `--position-ik-solve-max-iterations 2`、`--position-ik-damping 0.03`。
  需要在时间预算内尝试更多迭代时，显式传
  `--position-ik-adaptive-iterations --position-ik-solve-max-iterations <N>`；
  `--position-ik-iteration-reserve-ms` 保留后续 joint limit/collision selection 时间。
  这些参数不进入 impedance 后端。
- 如果提供 `--debug-replay-dataset` / `--debug-replay-parquet`，且没有显式传 `--action-hz`，部署器会优先从数据集实际记录的 action 行间 `timestamp` / episode stats / `control_hz` 推断真实 action 执行频率，并按该频率发送 action。
- deploy debug replay 识别 v2.2 和 v2.3 typed layout。v2.3 的 `raw_absolute` 或
  `desk_absolute` action 都先统一为 policy-facing canonical desk，再由当前机器的 runtime
  pose contract 解成 runtime raw 后进入 safety/IK。canonical desk 绑定到 RM65 v3
  `workspace_near_boundary_frame`；native v2.3 policy 或 v2.3 debug replay 未显式传 contract 时，
  RM65 deploy CLI 默认加载：
  `--runtime-pose-frame-contract robot_motion/configs/pose_frames/rm65_dual_arm_workspace_near_boundary_v1_runtime_raw_absolute.json`。
  collection 的 `desk_absolute` contract 不能代替这个 runtime `raw_absolute` contract。
  policy artifact 的 `pose_training_data_contract.origin` 决定两条不可混用的路径：
  `native_v23_dataset` 使用上述 RM65 默认值，也可显式传其他当前机器的 runtime contract；
  `v22_raw_identity_compatibility` 只允许相同 RM65 `source_id` 的 identity raw→desk，
  可显式传 `rm65_real_dual_arm_v1_identity.json`，省略时生成同语义的审计 identity binding。
  后者误传 workspace-near-boundary/non-identity contract 会在 backend 创建前失败，因为旧模型的
  absolute state 数值仍绑定左右臂各自 raw base；这不影响 relative action 的 SE(3) 不变性，
  但会改变模型 state 分布。
- 也可以用 `--action-hz-dataset` 单独指定一个参考数据集来推断发送频率，而不进入 replay 模式。
- `--policy-hz` 可以省略；省略时按现有 resolver 选择 policy/action 频率。
- `--action-chunk-execution-steps X` 显式启用 receding-horizon chunk：每次收到模型
  chunk 只把前 `X` 个动作放入 real runtime queue，随后再请求/消费下一个 chunk；省略
  时保持完整 chunk 行为。该选项只改变 chunk 边界，不改变 action 的 pose、relative
  anchor、safety 或 position resampling 语义。需要严格按 X 步切换时配合
  `--action-queue-policy replace --min-action-queue 0` 使用。
- 当前第一阶段只落地统一的“前 X 步”消费边界和连续时间排程，不启动新的推理线程；
  后续异步 worker 可以复用同一配置而不改变动作语义。
- position 在线插值严格因果：只使用在当前时刻已由 chunk 产生的右端点。
  xyz 线性、quaternion 最短弧 SLERP、qpos 线性，不增加 low-pass/EMA。
- position 真机输出默认强制双臂模式：15 Hz target pose 先按时间插值为 60 Hz pose，
  再由共享 coupled planner 做双臂 collision-aware IK；绝对关节限位和每个 60 Hz
  reference 步长限位之后会重新计算所选 scene profile 的碰撞距离。通过后的 qpos 才会
  进入共享 60→120 qcmd planner；默认使用 30 ms 目标速度低通、加速度包络和
  `--position-max-command-step-deg 0.675`。real backend 不再各自执行单臂 Cartesian IK。
- 每次新 position stream（包括 clear/resume 后）只允许用本次 observation 中左右臂都
  `hardware_read_success=true`、时间戳足够新鲜且双臂时间差受限的
  `hardware_qpos/measured_qpos` 重置 coupled planner。真机下发不会回退到
  `current_qpos`、last command 或 state vector 缓存；仅显式 non-motion dry-run 可用
  observation state vector，并在 debug 中标记 `authoritative=false` 和 fallback reason。
  当前 executor 默认 freshness 上限为 0.25 s、双臂 measurement timestamp skew 上限为
  0.05 s，并会在 worker 真正调用 planner reset 时再次检查 age，而不只在 submit 时检查。
- collision planner 的 60 Hz qref 会标记为已做联合限位，RM65 backend 不再重复做独立
  单臂 step clip。每次 arm queue 后仍会核对结构化 backend receipt：任一侧 reject、
  singularity hold、target 改写或 background stream 不健康都会在 hand 入队前 fail-stop。
  由于 RM65 backend 当前没有公开的无副作用 singularity preview/双臂原子 queue API，
  两侧 SDK queue 仍是顺序边界；若第二侧才拒绝，第一侧可能已短暂入队，随后立即 group
  hold。这一 partial coupled boundary 会进入 debug receipt，不能宣称原子提交。
- 当前认证范围是“所选 scene profile 的 60 Hz 最终 qref endpoint”。相邻 endpoint 之间
  的 120 Hz 关节运动整段没有连续碰撞认证；`dual_arm_hand_proxy_open` 也是固定张开手/
  link proxy，不代表任意自定义工具外形已被完整建模。metadata/debug 会始终记录
  `endpoint_collision_checked` 与 `segment_collision_certified=false`。
- 单臂 position 真机输出暂时 fail-closed：当前没有未激活机械臂的权威 qpos/TCP 状态源，
  因此不能用默认姿态猜测被动臂。单臂 dry-run 会保留，但不宣称双臂碰撞认证。
- 同一 policy action 内的手/夹爪仍保持安全后的 15 Hz 离散值、不做插值，但与 arm
  keyframe 一起进入因果队列，在对应的 60 Hz reference tick 做逻辑调度。实际 hand I/O
  由独立有界 worker 执行，不阻塞 60 Hz arm
  reference；executor 对每个未合并 keyframe 至多调用一次 backend callback（backend 内部仍
  可能按其恢复策略重试）。调度落后时只保留最新到期离散值，不补发 burst；callback
  起止时间、backend receipt、合并/跳过和 fault/hold 都进入 deploy debug event。
  这是 fail-stop，不是跨设备原子提交：若 hand 故障发生时一个 arm callback 已越过最终
  gate 并进入 backend，该 callback 最多仍会完成一次，随后才 hold；debug 会把它标为
  `accepted_before_async_abort`。所有真实 backend callback 必须配置有限 I/O timeout，
  `shutdown_timeout_sec` 只约束 worker join，不能取消已经阻塞在 SDK 内部的调用。
  hand worker 在每次真实 hand callback 前还会重新检查全部 arm backend 的 readiness、
  emergency/send 状态与 background stream fault/thread/target health；发现故障会抛出并
  触发 executor fault/hold。该检查缩小了 arm target 入队后后台故障造成的 hand 尾发窗口，
  但健康快照与实际 hand I/O 之间仍不是跨设备事务，无法消除最后一个检查后发生故障的竞态。
- impedance 分支保持现有 action-rate -> 40 Hz backend，不进入 position executor。
- `--debug --debug-dir <dir>` 的 JSONL 会把 15 Hz `action_step` 明确记录为
  `position_async_queue`（只表示入队），并另写 `position_executor_*` 异步事件：source
  index、60 Hz parent/alpha/arm command、hand logical tick、callback 实际起止/queue lag、
  backend receipt、coalesce/skip、partial failure、fault 与 hold；不会把入队误报成已下发。
  collision summary 始终包含 status、distance、限位原因、endpoint/segment 边界；debug
  开启时另含 raw IK qpos、最终 qpos、target/realized world pose、每次 distance attempt
  和 solve/budget 时间。collision reset event 另含双臂 hardware read/measured sequence、
  measurement age/pair skew/authoritative source；arm dispatch event 含 requested/final qpos、
  rewrite、singularity gate、transport receipt 和 background health。
- RM65 策略执行的新 position profile 默认使用位控专用 8× 关节速度包络
  `5.6,5.6,7.2,12,12,16 rad/s`；显式兼容 profile `position` 仍保留
  `0.7,0.7,0.9,1.5,1.5,2.0`。`--arm-step-dt-scaling` 默认开启，
  `--arm-step-dt-max-scale` 未显式指定时，position 按
  `max(3.0, 1.1 * --arm-control-hz / --position-reference-hz)` 自动计算；
  impedance 仍按其实际 command-input rate 计算。
  joint-vector action 的 `--max-arm-joint-step-rad` 默认是 `6deg`。
  position policy 的 IK 目标步长按 60 Hz reference 间隔计算，RM65 background stream
  再按 `--arm-control-hz` 细分发送。

## Action / Observation Layout

部署器有两个加法式合同代际，旧代际不被重解释：

- `legacy_v1`：`action` 为 26 维
  `[left_arm_pose7, right_arm_pose7, left_hand_q6, right_hand_q6]`；
  `observation.state` 为 38 维
  `[left_qpos6, left_pose7, left_hand6, right_qpos6, right_pose7, right_hand6]`。
- `dexteleop_dataset_v2.2`：float action/state 都固定以
  `[left_pose7, right_pose7]` 开头，再按 `PolicyIOLayout` 分侧追加连续末端值；离散末端值放在
  独立 int64 token 字段。`observation.arm.qpos` 是独立 12D 记录/审计与初始化字段，不进入
  默认模型 `observation.state`。
- 左右末端布局可以不同，也可以缺省一侧。运行时可用
  `--left-hand-backend` / `--right-hand-backend` 覆盖全局 `--hand-backend`。
- v2.2 部署用 `--policy-artifact-json` 在创建 backend 前绑定 layout hash、输入/输出 float/token
  维度及运行时末端能力。带显式 `legacy_q6_slot_indices` 的 layout 走现有 q6 安全链；缺少该
  映射的 `v22_native` 当前 fail-closed，直到对应 native safety/dispatch adapter 被配置。
- arm pose 是 base frame 下的 `[x, y, z, qx, qy, qz, qw]`，控制对象是
  `tool_tcp`，也就是当前臂/末端 profile 绑定的软件 TCP。
- RM65 真机执行会把 `tool_tcp` 目标转换到 SDK/IK 所需的 hand_base 或 arm_link6
  内部 frame；policy action 不应直接使用这些内部 frame。
- payload、重力补偿和 impedance 参数同样由部署/真机执行 profile 解释。
- 如果部署旧版位控模型、且不希望依赖力控标定 profile，可以给对应 side 传
  `--left/right-tool-profile position_endpoint`。该 profile 将 `tool_tcp` 固定在 RM65
  机械臂末端，且不携带 payload、重力补偿或 impedance 参数。
- 部署 observation/action metadata 默认携带 `arm_action_specs_by_side`、
  `arm_controlled_frame_by_side`、`arm_reference_frame_by_side`。下游模型或 adapter
  必须按 spec 解释 arm pose。
- 旧 row/replay 数据如果缺少 canonical action metadata，会由 row replay 兼容入口按历史
  `arm_endpoint` pose 解释；新数据和 policy 输出按 `tool_tcp` spec 解释。

策略输出支持：

- `np.ndarray` shape `(26,)` 或 `(chunk, 26)`
- 单臂 active 时 shape `(7,)` 或 `(chunk, 7)`，表示该侧 arm pose
- 双臂 active 时 shape `(14,)` 或 `(chunk, 14)`，顺序 `left,right`
- hand-only shape `(12,)`，顺序 `left,right`
- dict: `{"action": ...}` 或 `{"action_chunk": ...}`
- dict: `{"left_arm_pose": pose7, "right_arm_pose": pose7, "left_hand": q6, "right_hand": q6}`
- dict: `{"left_arm_qpos": q6, "right_arm_qpos": q6}` 仍可用于 legacy/dry-run 或
  impedance/初始化等显式低层路径；collision-required 的 position 真机 policy 主链目前只
  接受双臂 target pose，避免绕过 60 Hz coupled IK 的状态提交。

注意：`DEPLOY_SCHEMA.md` 中的完整 canonical JSON 是目标契约，不是当前
`normalize_policy_output()` 可直接接收的返回格式。policy wrapper 如果内部使用该结构，
需要在返回前转换成上面的 short form、`NormalizedAction` 或 `RobotAction` 兼容路径。

## Policy 前数据集帧 QPOS 初始化

真实策略部署可以先选定一个已采集数据集 episode/frame，用该帧记录的 qpos 初始化机械臂，
然后再进入 startup init 和 policy inference。这用于让模型部署从指定数据集状态对齐启动。

```bash
python -m teleop_runtime.deploy \
  --policy /abs/path/my_policy.py:create_policy \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --policy-qpos-init-dataset /abs/path/to/dataset \
  --policy-qpos-init-episode 20 \
  --policy-qpos-init-frame 123 \
  --policy-qpos-init-source auto \
  --policy-qpos-init-arm-max-angvel 0.25,0.25,0.25,0.35,0.35,0.35 \
  --init-source current \
  --wait-for-ready-confirm \
  --send-actions
```

启动顺序：

- 启动后按 `g`，执行 dataset frame qpos init。
- 到达目标后进入 startup init；完成后如果启用 ready confirm，会显示 `ready=wait:A`。
- 再按 `A`，才开始 policy inference / action execution。

`--policy-qpos-init-source auto` 和 replay 一致，优先使用记录的
`action.replay.<side>_arm_joint_target`；v2.2 随后使用独立的
`observation.arm.qpos` 审计列，旧数据再依次回退到
`observation.arm.<side>.current_qpos`、`observation.arm.<side>.actual_qpos`。
初始化轨迹规划同样复用 replay qpos init 的慢速、双臂碰撞检查和奇异位形保护。
需要临时调整首帧 QPOS 初始化速度时，使用 `--policy-qpos-init-arm-max-angvel`；
可传 1 个 rad/s 值应用到 6 个关节，也可传 6 个逗号分隔的逐关节 rad/s 值。
该路径和旧 `--deploy-fixed-init-*` 互斥；后者保留为 legacy fixed T 轨迹入口。

安全门：

- 真实 arm 下发时，默认必须提供 `--policy-qpos-init-dataset` 或
  `--policy-qpos-init-parquet`。
- 如果确实要从当前 qpos 原地开始推理/执行，优先显式加 `--no-init-qpos`。该参数会覆盖
  deployment profile 或同一命令中注入的 `--policy-qpos-init-*` 数据集初始化输入，使
  `init-source=current` 的同位姿 Cartesian startup init 直接判定为已满足，不再进入
  dataset-frame qpos-init，也不再要求先按 `g` 发送冗余保持流；手动部署仍需按 `A` 通过
  ready confirm 后才会执行 policy。
  `--no--init-qpos` 是兼容拼写；新命令统一使用 `--no-init-qpos`。
- 旧的 `--start-init_qpos` 仍保留兼容，但它只在命令本来没有注入
  `--policy-qpos-init-*` 时允许从当前 qpos 启动，不负责覆盖 profile 默认值。
- `--no-init-qpos` 是强制安全覆盖模式，日志会明确打印被抑制的 qpos-init 输入；它不会
  自动启用 `--auto-start`，也不会关闭当前位姿 observation/reference、ready confirm、碰撞、
  受力或 observation 安全门。只应在确认当前姿态、末端、工作空间和 policy 首帧都安全时使用。
- dry-run 不受该限制，可以不提供 qpos init 数据集。

## 运行时操作和按键

脚本启动后会打印当前模式、是否真实下发、是否需要 ready confirm，以及可用按键。
按键只在运行脚本的终端处于前台且 stdin 是 TTY 时生效；大小写等价。

常用按键：

- `g`: start/resume。普通部署中启动或恢复 policy；如果配置了
  `--policy-qpos-init-*` 且尚未完成，则先执行选定数据集帧的 qpos init。
- `A`: ready confirm。`--wait-for-ready-confirm` 开启时，startup init 完成并出现
  `ready=wait:A` 后按 `A`，才开始 policy inference / action execution。`A` 不会启动
  policy qpos init；qpos init 必须用 `g` 开始。
- `h` / `H`: 普通部署中暂停 policy 并清空 action queue。policy qpos init 正在执行时不会用
  `h` 中断轨迹；如需真实急停，使用外部安全流程。
- `q`: quit。退出 deploy loop 并关闭已打开的 backend/camera。它不是急停。
- legacy fixed-init 下同一个 `h` / `H` 键有特殊含义：配置 `--deploy-fixed-init-*` 时，
  第一次按键移到 T 轨迹首帧 qpos，出现 `fixed_init=ready:H` 后第二次按键执行固定 T 轨迹。

启动状态：

- 默认不加 `--auto-start` 时，policy 初始为 paused，需要按 `g`。
- 加 `--auto-start` 且没有 qpos/fixed init gate 时，启动后自动进入 policy。
- 加 `--auto-start` 且配置了 `--policy-qpos-init-*` 时，启动后自动进入 qpos init；
  qpos init 完成后才释放 startup init / policy。
- 加 `--send-actions` 才会真实下发；真实 arm 下发还需要数据集 qpos init，或显式
  `--no-init-qpos` 从当前 qpos 启动（旧的无 profile 路径也可用 `--start-init_qpos`）。
  不加 `--send-actions` 时是 dry-run，只构造 observation、过滤动作并打印状态。

状态行常见字段：

- `policy=active|paused`: policy 主循环是否放行。
- `send=on|dry-run`: 是否真实发到硬件。
- `queue=N`: 待执行 action chunk 数量。
- `policy_qpos_init=wait:g|planning|qpos:i/N|final_hold|failed:g|timeout:g|released`: policy 前 qpos init 状态。
- `init=wait|xx%|done:<source>`: startup init 状态。
- `ready=wait:A`: startup init 已完成，正在等待 `A` 确认。
- `state_valid=a/b`: 当前 observation state 中有效元素数量。

常见 qpos init 失败：

- `qpos init planning failed before execution` / `no collision-safe interpolation step was found`：
  规划阶段没有找到满足双臂碰撞约束的插值路径，命令不会下发，policy 会保持 blocked。
  优先换一个更安全的数据集 frame，或先把机械臂手动移动到更安全的 pre-start qpos。
  如果确认是 strict proxy 过于保守，可在检查真实间隙后尝试
  `--policy-qpos-init-collision-scene-profile dual_arm_hand_proxy_open`。
- `timed out while moving to dataset frame`：轨迹已经开始执行，但实际 qpos 未在
  `--policy-qpos-init-timeout-s` 内进入 `--policy-qpos-init-qpos-tolerance`。
  日志会给出最大关节误差。先确认机械臂是否仍在运动、是否被限速/保护挡住；必要时增大
  timeout，或在确认可接受的前提下放宽 qpos tolerance。当前 policy/replay qpos init
  完成阈值默认为 `0.1rad`；显式传入的 CLI/profile 值仍优先。
- `targets={'left': 'observation_current_qpos', ...}` 表示该帧没有可用的
  `action.replay.<side>_arm_joint_target`，`--policy-qpos-init-source auto` 回退到了记录的
  observation qpos。若必须使用 replay qpos，改用
  `--policy-qpos-init-source replay_qpos`，缺失时会直接报错。

### hand16 / 空槽部署语义

后续 hand16 action layout 中，手部固定槽位用于统一夹爪、InspireTac 和高自由度
灵巧手的接口长度。固定槽位不代表所有槽都有效：

- 有效槽由 `hand_active_mask_by_side` 或本体 `HandActionSpec` 定义。
- 无效槽值约定为 `NaN`，表示 padding/unused slot。
- 部署侧不得把 `NaN` 槽解释为 0，也不得把它们下发给硬件。
- policy 输出、teleop 输入和 dataset replay 都必须在进入 safety filter/backend 前按 mask
  只取有效槽，再由具体 hand adapter 映射到本体命令。

因此，模型 checkpoint 或部署脚本如果使用 hand16，必须随同声明 hand spec；仅凭固定
action 维度不足以判断哪些 hand 槽可执行。旧模型/旧数据集应通过自己的 adapter 映射，
部署主线不为旧 state/action 维度保留额外兼容口径。

## 单臂 Dry-Run

```bash
python -m teleop_runtime.deploy \
  --policy hold \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --max-steps 200
```

未传 `--auto-start` 时启动后按 `g` 才执行策略；按 `h` 暂停并清空 action chunk，按 `q` 退出。
如果同时使用 `--wait-for-ready-confirm`，startup init 完成并出现 `ready=wait:A`
后还需要再按 `A`，才会真正开始 policy inference / action execution。

## 单臂真实下发

当前 collision-required position 主链会拒绝单臂 `--send-actions`，因为 runtime 没有
被动臂权威 qpos/TCP，不能在完整双臂模型中严谨认证。若现场实际只有一台机械臂，需要先
增加与物理场景匹配的单臂 collision scene/状态 contract；不要用默认双臂姿态绕过。
既有 impedance 40 Hz 单臂链路不经过本 position planner，仍按其已确认的 tool/payload/
force profile 与独立安全流程使用。

## 双臂 + 双手

```bash
python -m teleop_runtime.deploy \
  --policy /abs/path/my_policy.py:create_policy \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --enable-hands \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-tactile-regions all \
  --policy-qpos-init-dataset /abs/path/to/dataset \
  --policy-qpos-init-episode 20 \
  --policy-qpos-init-frame 123 \
  --send-actions
```

## 双臂 + DMTac Gripper

先按根目录 `README.md` 的“GRIPPER 和 DM TAC 分开标注”保存
`teleop_runtime/configs/hardware/dmtac/latest.json`；部署时读取同一份 latest：

```bash
python -m teleop_runtime.deploy \
  --policy /abs/path/my_policy.py:create_policy \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --hand-backend dmtac-gripper \
  --dmtac-config latest \
  --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json \
  --policy-qpos-init-dataset /abs/path/to/dataset \
  --policy-qpos-init-episode 20 \
  --policy-qpos-init-frame 123 \
  --send-actions
```

没有 DM TAC 触觉时加 `--dmtac-no-tactile`。DMTac gripper 仍占用 6 维 hand action
兼容槽，但部署 metadata 会把 `HandActionSpec` 标成 `dmtac_gripper`，只有
`--gripper-output-dim` 指定的槽是真实 gripper close command。

## 外部模型 Client

`teleop_runtime.deploy` 只负责真机侧 observation、统一动作解析、安全检查和硬件下发。
模型 server 的连接、协议校验、输入/输出适配放在 `deployments.client` 以及对应模型族目录。

OpenPI 的入口位于 `deployments/openpi/client`。例如 dry-run：

```bash
python -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8000 \
  --model dual_rm65_dmgripper_latest \
  --active-sides both \
  --hand-backend dmtac-gripper \
  --dmtac-config latest
```

该入口会把 OpenPI-specific 参数展开成普通的 `--policy
deployments.openpi.client.policy:create_policy`，因此 real runtime 不 import
也不绑定 OpenPI。真实下发仍需显式加 `--send-actions`。

## Dataset Replay Debug

可以直接 replay `teleop_runtime.collection` 采集出的 action。默认仍是 dry-run；真实下发仍需 `--send-actions`。
debug replay 在 `--action-queue-policy auto` 下会自动 append 预取的 action
chunk，避免按 `--min-action-queue` 补队列时跳过轨迹尾部；普通模型 policy 仍默认
replace pending queue。

`--debug-replay-dataset` 会在创建真机 backend 前比较 episode metadata 的
`arm_control_mode` 与所选 execution profile/mode；不一致直接拒绝。旧 episode 缺少该字段
且启用 `--send-actions` 时按 `impedance` 契约处理，避免未知旧数据误走 position；纯 dry-run
只输出 warning。该约束只作用于真正消费 dataset action 的 debug replay，不作用于仅用于
qpos 初始化或 action-rate 推断的数据集。

```bash
python -m teleop_runtime.deploy \
  --debug-replay-dataset /abs/path/to/dataset \
  --debug-replay-episode 0 \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --auto-start
```

上面命令在未显式传 `--action-hz` 时，会自动读取该 episode 的真实频率并按接近真机的节奏发送/消费 action。

## 用真机数据集校准部署频率

```bash
python -m teleop_runtime.deploy \
  --policy /abs/path/my_policy.py:Policy \
  --policy-kwargs-json '{"checkpoint": "/abs/path/model.pt"}' \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --action-hz-dataset /abs/path/to/real_dataset \
  --action-hz-episode 12 \
  --policy-qpos-init-dataset /abs/path/to/real_dataset \
  --policy-qpos-init-episode 12 \
  --policy-qpos-init-frame 0 \
  --send-actions
```

## 安全边界

- arm cartesian pose 会按 `--arm-workspace-min/--arm-workspace-max` 裁剪位置。
- arm pose 每个 action keyframe 默认限速：`--max-arm-xyz-step-m 0.10`、`--max-arm-rot-step-deg 15`。
- arm joint target 每个 action keyframe 默认限速：`--max-arm-joint-step-rad 0.10472`（约 `6deg`）。
- hand q6 当前只会 clip 到 `[0, 1]`；`--max-hand-step` 是保留参数，hand rate-limit 暂未启用。
- DMTAC 位置反馈以 `joint_timestamp` 为权威采样时间。真机部署默认拒绝年龄超过
  0.2 s 的夹爪状态；新的 telemetry 循环时间不能掩盖被跳过读取的旧位置。
- `--require-ready-observation` 会在 active state 或图像缺失时跳过策略和下发。
- `--inspire-hand-max-force` 默认继承 `teleop_runtime` 的 `800`。
