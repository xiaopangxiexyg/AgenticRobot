# teleop_runtime.deploy

`teleop_runtime.deploy` 是独立于采集主循环和左臂实验模式的策略部署入口，面向 RM65 单臂/双臂，可选 InspireTac 手、DMTac 1D gripper 和 RealSense。默认 dry-run；真实下发必须显式加 `--send-actions`，且默认要求先用数据集帧 qpos 初始化。

当前推荐的 shell 入口是 `python -m teleop_runtime.deploy`。它的薄包装
`teleop_runtime.deploy.cli.main` 会委托给同一份 `run_policy.main` 运行时；
`python -m teleop_runtime.deploy.run_policy` 和 `run_policy.main(...)` 不是“已不用”的
遗留实现，而是当前仍受支持的直接/程序化入口。

## 入口支持、责任与兼容期

| 入口 | 当前状态 | 责任边界 | 替代/迁移关系 |
| --- | --- | --- | --- |
| `python -m teleop_runtime.deploy` / `teleop_runtime.deploy.cli.main` | 推荐的 shell CLI | `teleop_runtime.deploy` 真机部署运行时 | 调用同一 `run_policy.main`，不另建控制环。 |
| `python -m teleop_runtime.deploy.run_policy` / `run_policy.main` | **受支持**的直接与程序化入口 | `teleop_runtime.deploy` 真机部署运行时 | OpenPI real launcher 当前直接依赖；未来若收敛调用面，应迁向上述 canonical CLI wrapper 或其稳定 API。 |
| `python -m deployments.openpi.client.real` | **受支持**的 OpenPI 专用薄入口 | `deployments.openpi.client` 负责 OpenPI 参数/transport；`teleop_runtime.deploy` 负责真机 I/O 和 dispatch | 直接组装 OpenPI policy 后交给 `run_policy.main`，不实现第二套控制循环。 |

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
  -> position: causal timestamp pose resampling（75 Hz）
  -> coupled dual-arm collision IK + joint/step limits + endpoint recheck（75 Hz qpos）
  -> RobotAction / RobotActionResolver / joint TeleopCommand
  -> real launcher dispatch
  -> common qpos interpolation / RM65 position backend qcmd stream（150 Hz 默认）
```

代码边界：

- `robot_motion/action`: dense action vector 到 `RobotAction` 的 sim/real 通用适配层。
- `robot_motion/deploy`: policy action 过滤、canonical `RobotAction` 构造和 `TeleopCommand` 解析；`teleop_runtime.deploy` 中同名模块仅为兼容 re-export。
- `robot_motion/position_control/resampling.py`: sim/real 通用真实时间戳 pose/qpos 重采样。
- `robot_motion/rm65/position_collision.py`: sim/real 通用双臂 75 Hz pose→qpos、绝对/
  reference 步长限位，以及限位后 endpoint collision recheck；不含时序平滑。
- `robot_motion/position_control/interpolation.py`: sim/real 共用 75 Hz qpos→150 Hz qcmd
  planner；sim/real 只在最后 qcmd sink 分叉。
- `teleop_runtime/position_control/deploy_executor.py`: real-only 在线因果 75 Hz worker、暂停/故障 hold 与 backend dispatch 边界。
- `teleop_runtime/deploy/dispatch.py`: 当前真机 launcher 的 `TeleopCommand` 到 arm/hand backend 分发；这是 real-specific 代码，不应迁到 `robot_motion`，sim launcher 应实现自己的 dispatcher。
- `teleop_runtime/deploy/actions.py`: 仍保留旧 `ActionSafetyFilter.filter()` direct-command path 作为 legacy 封存入口；新代码应使用 `filter_normalized()`。

因此模型部署、dataset replay debug 和未来 sim deploy 都应以 `RobotAction` 作为语义边界，
以 `TeleopCommand` 作为下发边界，不再让 policy adapter 直接生成 backend command。

当前目录边界为：

- `robot_motion`：`RobotAction` schema/spec、vector adapter、canonical deploy preparation、value-level safety、时间戳 motion resampling。
- `teleop_runtime`：CLI、键盘、相机/触觉、真机 backend、在线因果 worker、`dispatch.py`、fixed-init 真机执行。
- `sim`：MuJoCo/Sapien 等仿真专属 scene/controller/dispatcher；复用 `robot_motion` 的 action/resampling/schedule。

部署频率兼容旧接口：

- `--arm-execution-profile impedance_40hz|position` 复用真机遥操作已有的
  `RealArmExecutionConfig` profile resolver，避免 deployment 与 teleop 各维护一套默认值；
  历史命令可显式选择兼容profile `position_45_150hz`；`--arm-control-mode`和下面的逐阶段
  频率仍可显式覆盖。`deployments` 下所有真机命令应
  明确写 profile 或 mode，不依赖入口默认值。
- `--hz` 指定 policy/action keyframe 频率；position分支另用通用参数
  `--position-reference-hz`（默认75）和`--arm-control-hz`（默认150），算法不按45/75分支。
- `--position-collision-scene-profile`、`--position-collision-min-distance-m` 和
  `--position-collision-distance-epsilon-m` 配置 75 Hz endpoint checker；真实 position
  输出不能使用 `--no-position-collision-check`，该开关只保留给非运动 legacy dry-run。
- 75 Hz joint safety step 使用 `--arm-max-angvel / --position-reference-hz`，并与
  `--max-arm-joint-step-rad` 取更严格者；限位后必须重新做 collision distance check。
- 如果提供 `--debug-replay-dataset` / `--debug-replay-parquet`，且没有显式传 `--action-hz`，部署器会优先从数据集实际记录的 action 行间 `timestamp` / episode stats / `control_hz` 推断真实 action 执行频率，并按该频率发送 action。
- 也可以用 `--action-hz-dataset` 单独指定一个参考数据集来推断发送频率，而不进入 replay 模式。
- `--policy-hz` 可以省略；省略时按现有 resolver 选择 policy/action 频率。
- position 在线插值严格因果，显式落后一个 action period；零延迟没有未来右括号时只能 hold。xyz 线性、quaternion 最短弧 SLERP、qpos 线性，不增加 low-pass/EMA。
- position 真机输出默认强制双臂模式：15 Hz target pose 先按时间插值为 75 Hz pose，
  再由共享 coupled planner 做双臂 collision-aware IK；绝对关节限位和每 75 Hz
  reference 步长限位之后会重新计算所选 scene profile 的碰撞距离。通过后的 qpos 才会
  进入公共 75→150 qcmd planner，real backend 不再各自执行单臂 Cartesian IK。
- 每次新 position stream（包括 clear/resume 后）只允许用本次 observation 中左右臂都
  `hardware_read_success=true`、时间戳足够新鲜且双臂时间差受限的
  `hardware_qpos/measured_qpos` 重置 coupled planner。真机下发不会回退到
  `current_qpos`、last command 或 state vector 缓存；仅显式 non-motion dry-run 可用
  observation state vector，并在 debug 中标记 `authoritative=false` 和 fallback reason。
  当前 executor 默认 freshness 上限为 0.25 s、双臂 measurement timestamp skew 上限为
  0.05 s，并会在 worker 真正调用 planner reset 时再次检查 age，而不只在 submit 时检查。
- collision planner 的 75 Hz qpos 会标记为已做联合限位，RM65 backend 不再重复做独立
  单臂 step clip。每次 arm queue 后仍会核对结构化 backend receipt：任一侧 reject、
  singularity hold、target 改写或 background stream 不健康都会在 hand 入队前 fail-stop。
  由于 RM65 backend 当前没有公开的无副作用 singularity preview/双臂原子 queue API，
  两侧 SDK queue 仍是顺序边界；若第二侧才拒绝，第一侧可能已短暂入队，随后立即 group
  hold。这一 partial coupled boundary 会进入 debug receipt，不能宣称原子提交。
- 当前认证范围是“所选 scene profile 的 75 Hz 最终 qpos endpoint”。相邻 endpoint 之间
  的 150 Hz 关节插值整段没有连续碰撞认证；`dual_arm_hand_proxy_open` 也是固定张开手/
  link proxy，不代表任意自定义工具外形已被完整建模。metadata/debug 会始终记录
  `endpoint_collision_checked` 与 `segment_collision_certified=false`。
- 单臂 position 真机输出暂时 fail-closed：当前没有未激活机械臂的权威 qpos/TCP 状态源，
  因此不能用默认姿态猜测被动臂。单臂 dry-run 会保留，但不宣称双臂碰撞认证。
- 同一 policy action 内的手/夹爪仍保持安全后的 15 Hz 离散值、不做插值，但与 arm
  keyframe 一起进入因果队列，在对应的延迟后 75 Hz reference tick 做逻辑调度，避免手部
  领先机械臂一个 action period。实际 hand I/O 由独立有界 worker 执行，不阻塞 75 Hz arm
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
  index、75 Hz parent/alpha/arm command、hand logical tick、callback 实际起止/queue lag、
  backend receipt、coalesce/skip、partial failure、fault 与 hold；不会把入队误报成已下发。
  collision summary 始终包含 status、distance、限位原因、endpoint/segment 边界；debug
  开启时另含 raw IK qpos、最终 qpos、target/realized world pose、每次 distance attempt
  和 solve/budget 时间。collision reset event 另含双臂 hardware read/measured sequence、
  measurement age/pair skew/authoritative source；arm dispatch event 含 requested/final qpos、
  rewrite、singularity gate、transport receipt 和 background health。
- RM65 策略执行的关节速度/步长默认对齐
  `run_vive_rm65_real_teleop`：`--arm-max-angvel` 默认为
  `0.7,0.7,0.9,1.5,1.5,2.0`，`--arm-step-dt-scaling` 默认开启，
  `--arm-step-dt-max-scale` 未显式指定时，position 按
  `max(3.0, 1.1 * --arm-control-hz / --position-reference-hz)` 自动计算；
  impedance 仍按其实际 command-input rate 计算。
  joint-vector action 的 `--max-arm-joint-step-rad` 默认是 `6deg`。
  position policy 的 IK 目标步长按 75 Hz reference 间隔计算，RM65 background stream
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
  接受双臂 target pose，避免绕过 75 Hz coupled IK 的状态提交。

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
- 如果确实要从当前 qpos 原地开始推理/执行，必须显式加 `--start-init_qpos`。
  这是强制覆盖模式，日志会打印 override 提示；只应在确认当前姿态、末端、工作空间和
  policy 首帧都安全时使用。
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
  `--start-init_qpos` 覆盖。不加 `--send-actions` 时是 dry-run，只构造 observation、过滤动作并打印状态。

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
- `--require-ready-observation` 会在 active state 或图像缺失时跳过策略和下发。
- `--inspire-hand-max-force` 默认继承 `teleop_runtime` 的 `800`。
