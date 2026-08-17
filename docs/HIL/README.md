# Sim Human-in-the-Loop（HIL）使用说明

本文档主线对应当前 default-off 的单右臂 Sim-HIL 应用：临时 PPO 与真实或虚拟
VIVE + Q6/angle-trigger 输入经过 component-keyed composition 后，只驱动 MuJoCo。
它不会构造真实 RM65 backend，也不会改变既有 teleop、replay、deploy 或 RTC 路径。
后文仍保留历史 `CS-6b1f` replay→human takeover 说明，并明确标为历史 profile。
当前 ownership、四类公开 Sim-HIL profile 与未完成 trajectory/real gate 见
[HITL 当前边界](../architecture/plans/hitl_chunk_relative_migration_plan.md)；原 R0-R4、
R3.1a-c 的完整设计与实施记录见
[HITL 历史台账](../architecture/reports/hitl_chunk_relative_ledger_20260807.md)。

## 0. 架构边界：五个正交平面

当前 HIL 不再按“MuJoCo HITL”“OpenPI HITL”“双臂 HITL”分别复制实现，而是把变化
限制在五个彼此正交的平面：

```text
operator/device event
        │ typed intent
        ▼
Control Station application plane
  control_station/hitl/                 backend/model/topology-neutral
  ├─ exclusive lifecycle transaction
  └─ activation + fresh-emission evidence
        │ concrete lifecycle port
        ├───────────────┐
        ▼               ▼
Sim binding           Real binding               application-specific only
sim_hil_profile/      planned deploy binding
exclusive_hitl.py     before the existing single execution owner
        │               │
        └──────┬────────┘
               ▼
source plane → control_composition allocation → proposal/admission → lower execution
Human/Policy/   same-cut, component-keyed       safety/IK/reference/dispatch
Replay/...      streaming action; exclusive =
                base_only | absolute_overlay_owner

Every stage ── passive facts/receipts ──> ControlTrace projection/capture
                                         never points back into control
```

| 变化维度 | 通用契约 | 只允许出现具体知识的位置 |
|---|---|---|
| 截断接管/连续融合 | 截断切换在 `control_station/hitl`；融合规则在 `control_composition` | GUI 只选择显式 profile，不实现算法 |
| Sim/real/其它 backend | 相同 switch、source、composition、proposal 契约 | backend lifecycle port 与 target adapter |
| 虚拟、OpenPI 或其它模型 | 都先形成正式 `PolicyActionEmission` | 各模型 observation/action adapter 与 worker |
| 单臂、双臂、多 component | `ComponentActionSchema` 按 component ID/representation 描述 | 具体本体的 frame、IK、safety、dispatch capability |
| 记录/可视化 | 消费不可变 evidence/fact | trace capture、GUI、dataset projection；不得触发切换或下发 |

因此不存在“ControlTrace 控制”路径。直接接管由 typed intent 和生命周期事务完成；
ControlTrace 只记录已经发生的 HOLD、Home、selection、activation、emission 与错误。
trace admission 或持久化失败会使 capture health/finalize 失败，但不会在切换事务内部
撤销一次已经验证完成的切换或成为隐藏 authority gate。若所选采集 profile 明确要求
完整记录，Station 的独立 capture-health policy 可随后用可见原因请求 HOLD；这是显式
应用策略，不是 Trace 投影反向控制。

当前能力矩阵必须与通用契约区分：

| 边界 | 当前落地 | 尚未完成 |
|---|---|---|
| 截断式控制 | Sim 中 Policy-only ↔ Human-only 已接入通用 streaming allocation cycle；lifecycle port 只完成 HOLD/Home/activation 事务 | real deploy 对该 port 的具体 HOLD/Home/selection/activation binding |
| 连续融合 | fixed-alpha 和 Scheme-A/B 保留为独立 profile | 不作为截断式验收前置条件 |
| Policy | 虚拟 source 与当前 Sim policy source 复用同一 backend-neutral emission port；real sync/RTC deploy 可选地各投递一次被动 emission observation | 真机 Human source 的正式 composition/admission 与活动来源切换 |
| 本体 | component core 支持数据化子集；当前主线示例为右臂 pose + gripper qpos | 新本体仍需自己的 target catalog/frame/IK/safety adapter |
| 后端 | MuJoCo concrete binding 已有 | 真机不能同时启动旧 teleop loop 与 policy loop，必须复用单一下游 owner |

增加新对象时的规则是：新模型只加 source adapter，新遥操设备只加 Human emission
adapter，新 sim/真机只加 lifecycle/target adapter，新本体只声明 component schema 与
下游 capability。以上变化都不应复制 switch transaction、composer、ControlTrace schema、
policy queue 或 dispatcher。

当前优先验收的是更简单的截断式 profile：
`exclusive_truncated_hitl`。它不是另一套 composer，也不再绑定 fixed-alpha
compatibility cycle；它是通用 streaming allocation 的一个受限 profile：所有
component 仅允许原子选择 `base_only` 或 `absolute_overlay_owner`。0/1
权重只是该离散来源选择的直观解释，不是对 pose/qpos/未来动作类型的协议限制：

```text
启动前选择 Policy ---------------------------> Policy-only

启动前选择 Human -> HOLD -> 接入 Human -> 采集 Runtime Home
                 -> Human-only activation + fresh Human emission -> 继续运行

Policy-only --接管触发--> HOLD -> 接入 Human -> 采集 Runtime Home
            -> Human-only activation + fresh Human emission -> 继续运行

Human-only  --交还触发--> HOLD -> 作废旧 policy plan/chunk
            -> 以当前 sim/robot observation 重新推理
            -> Policy-only activation + fresh Policy emission -> 继续运行
```

切换事务会在 HOLD 内创建新 activation，提交全 component allocation
override，并等待精确的 post-operation allocation context 被接纳，然后才
允许下一次流式 target cycle。因此不会有“新 activation 先用默认 base-only
跑一帧”的窗口。任一中间步骤失败都停留在下游 HOLD；不会恢复旧 chunk，也不会绕过既有
composition、IK、Reference、限幅、安全或 dispatch。Human 交还后可以继续被动观测，
但其权重严格为零，不能影响 composed output。这个 profile 不启用比例混合、relative
residual、Rebase-and-clear 或自动 allocation，既有 fixed-alpha/Scheme-A/Scheme-B
行为保持不变。

本地无设备 preflight：

```bash
PYTHONPATH=. python -m control_station sim-hil \
  --dry-run --device cpu \
  --control-profile exclusive_truncated_hitl \
  --vive-coordinate-profile \
    teleop_records/control_station_vive_coordinate_profiles/vive_rm65_virtual_desk_coordinate_v1.json
```

带真实 VIVE + angle-trigger 输入、但执行仍只在 MuJoCo 的交互命令：

```bash
PYTHONPATH=. python -m control_station sim-hil --enable --control-gui \
  --control-profile exclusive_truncated_hitl \
  --compose-hz 30 \
  --vive-coordinate-profile \
    teleop_records/control_station_vive_coordinate_profiles/vive_rm65_virtual_desk_coordinate_v1.json \
  --gripper-input-source angle_trigger \
  --acceptance-root teleop_records/sim_hil_acceptance
```

页面把 Policy 和 Human 显示为两个独立输入模块。同一对来源卡贯穿启动和运行：启动前
按钮表示“以该来源启动”，启动后待命卡按钮变为“切换到该来源”，当前卡明确标为
“正在控制”。不会再出现第二套接管/交还按钮。Human 独占期间不持续轮询未被选择的
Policy。angle-trigger 仍然只提供夹爪 qpos，不能被隐式复用为切换按钮。

常用操作按四个阶段显示：准备输入、选择来源、运行控制、结束收尾。VIVE 实时状态常驻；
坐标标定和来源映射默认折叠，需要修改时再展开。运行阶段只常驻 HOLD、恢复和 Runtime
Home；封存、重建和退出位于单独的生命周期区域，并保留确认提示。按钮不可用时，原因在
按钮下方直接显示，不依赖鼠标悬停。GUI 启动页还可在 MuJoCo 构造前选择 Sim 碰撞
处理，默认启用代理安全间隙保护。启用时，下游 `collision_margin_low` 表示规划器代理
低于场景声明的预防间隙（不等于 MuJoCo 已发生物理接触），会先建立 HOLD，再终止
当前 episode；之后只能 Reset 仿真，不能 rearm、重采 Runtime Home 或切换来源继续。
此外，MuJoCo 若观测到超过硬穿透限值的实际禁碰接触，也会以独立语义终止；方块—桌面
和夹爪—方块等任务允许接触不会被误判为禁碰接触。
关闭时（CLI 等价选项为 `--no-sim-collision-stop`），pose planner 不安装碰撞 pairs，
MuJoCo 物理接触仍存在，但碰撞不会停止 Sim；IK、关节约束、非有限数值与其他执行故障
保持不变。该设置在 target 构造后锁定，且仅作用于 Sim，不改变任何真机安全链路。
现有脚踏板 F13--F17 也已有 Fix/Init/进度映射；物理接管/交还触发器必须用后续显式
profile 选择两个无冲突输入通道，而不能改写既有映射。

每次进入切换事务后的成功或失败都会写入同一个 ControlTrace：触发 intent identity、请求/完成
时间、下游 HOLD evidence、完整 Runtime Home（仅接管）、source-selection record、
新 activation、目标来源的 fresh emission、policy reset 事实与错误。它们是控制事实，
不等同于最终数据集，也不产生第二套 trace/authority/dispatcher。

需要继续研究连续 residual/allocation 时，Scheme-A 和 Scheme-B 仍作为独立、显式
profile 保留；它们与截断式 profile 共用 source、composition 和下游执行端口，但不在
同一个 activation 中热切换。

该主线不依赖 replay，也不伪造 replay cursor 或旧 Station session。它使用
component-keyed `right_arm: cartesian_pose` 与
`right_gripper: physical_joint_qpos`，只在 MuJoCo target 私有边界物化当前兼容
执行表示。当前自动验收分为：

1. fixed-alpha compatibility 保留 PPO-only、human-only 与比例混合回归；
2. 显式截断式 profile 验证 Policy-only → Human-only → Policy-only，以及每次切换的
   HOLD/Home/selection/activation/fresh-emission 证据；
3. 显式 Scheme-A profile 验证 Policy baseline、Human relative correction、Auto
   allocation 与 Rebase-and-clear；Scheme-B 在此基础上验证连续的 Human absolute
   takeover/handback；
4. episode end 先 HOLD/cancel，再完成 ControlTrace WAL、seal、commit 与回读，
   核对 source、composition、target/reference/measurement、两层 difference 和
   cancellation 事实。

最小无设备验收：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_control_station_exclusive_hitl.py::test_exclusive_hitl_virtual_policy_and_devices_switch_through_mujoco
```

它使用虚拟 policy 和本地虚拟 VIVE/gripper transport，贯通真实 component
composition、MuJoCo target、IK、ControlTrace 以及两次截断切换，不打开硬件、不联网。
真实 VIVE、Q6/angle trigger 的连接、recenter/deadman 与操作者行为仍需独立手工
验收；通过上述测试不代表真机设备已验证。交互式 Sim-HIL 使用 immediate-qref，
`--target-admission-margin-sec` 仅保留给显式 scheduled 诊断 profile；它不改变
当前交互路径，也不增加混合前数值安全或运动限制。

## 1. 当前能力与边界

| 项目 | 当前语义 |
|---|---|
| 运行目标 | MuJoCo authoritative runtime |
| policy 输入 | 临时单右臂 PPO，经正式 `PolicyActionEmission` seam 输出 pose + gripper qpos |
| human 输入 | `PassiveTeleopComponentSessionSource`；右臂 pose + 右夹爪 qpos，不拥有 dispatch |
| composition | 默认 fixed-alpha compatibility；当前优先验收的截断式 profile 与 Scheme-A/B 均须通过 `--control-profile` 显式选择 |
| 截断式操作 | 启动前选择 Policy-only 或 Human-only；运行中接管/交还；来源权重只允许 0/1 |
| 高级操作 | Scheme-A/B 保留 relative、Auto allocation、Rebase-and-clear；Scheme-B 另有 Human absolute takeover |
| arm 表示 | composition 主线保持 Cartesian pose；MuJoCo target 内部才 pose→coupled IK→qpos |
| capture | ControlTrace core WAL/seal/commit/readback；不冒充包含视觉/触觉的正式数据集 |
| 真机副作用 | 可打开真实 VIVE/gripper 输入；执行目标仍是 MuJoCo，`real_arm_motion=false` |
| application caller | `python -m control_station sim-hil`，default-off，必须显式 `--enable` |

当前主线没有第二套 policy queue、dispatcher 或 episode owner。
新的遥操设备、sim target 或模型 source 应优先复用已有 source/target/provider/emission
seam；不要按设备、checkpoint 或 simulator 新建 composition。

真机 deploy 当前新增的 `policy_emission_observer` 是仅供程序化调用方注入的
default-off、bounded、non-blocking 被动端口。sync queue 与 RTC 都只在已有唯一
`PolicyActionEmission` 形成后投递一次；观察端口异常只记录 debug event，不能阻塞或
改变既有执行。它不是 composition consumer、source-selection receipt 或真机接管授权；
未来 Active Human 接入必须另行复用正式 composition/proposal 与上述 lifecycle port。

### 1.1 Policy + Human 无命令 composition shadow

当前还提供一个 backend/model/topology-neutral 的程序化 shadow 会话：
`control_station/composition/policy_human_shadow.py` 中的
`PolicyHumanCompositionShadowSession`。它需要具体 Policy/Human emission adapter，
所以属于 composition 应用绑定，不放进保持 source-runtime-neutral 的 `hitl/` 核心。
它不是新的生产入口，也没有 CLI；未由应用根显式
构造并注入时完全关闭。启用后，现有 deploy loop 和 teleop source 只做常数开销的被动
投递，Station 线程才执行 input adaptation、receiver admission 和同一目标时刻的
composition evaluation：

```text
existing policy loop ─ PolicyActionEmission ─┐
                                             ├─ bounded passive ingress
existing teleop source ─ HumanCommandEmission┘          │
                                                        ▼ Station thread
                                             CompositionInput admission
                                                        │
                                             evaluate_composition(target time)
                                                        │
                                      non-authoritative candidate + differences

absent by construction: proposal publication / authority / HOLD / safety / IK /
                        Reference / ExecutionTarget / command / dispatch
```

程序化接线只复用已有端口：

```python
policy_kwargs["policy_emission_observer"] = shadow_session
human_fanout = PassiveHumanCommandObserverFanout(
    {"composition-shadow": shadow_session.observe_human_emission}
)
cycle = shadow_session.evaluate_at(target_effective_at=station_target_time)
```

`target_effective_at` 是 composition 的对齐时刻，不是下游
`ExecutionTarget`。每个会话只允许严格递增的目标时刻；当前没有未来 target 的
replacement/version 语义。Human 的有效区间从 receiver 实际观察时刻开始，Policy
保留正式 emission 自带的 keyframe/TTL 时序，最终由 `CompositionInputStore` 对目标
时刻选择各角色输入。两个来源频率无须相同，也不会由 TTL 代替插值；Policy 只有在
上游显式提供 guaranteed-next-step 时才可使用已有 chunk 插值接口。

队列满时只淘汰最旧的 shadow 观察并增加 `evicted_count`，绝不向 policy 或 teleop
control loop 施加背压。因此该会话适合实时可视化、差异检查和接线验证，但不是完整
source recorder；需要完整复现的数据仍由正式 ControlTrace/collection owner 保存。
Trace sink 失败会出现在每个 submission 的 `trace_failures` 中，但不会改变 composition
结果。缺少必须来源时保留通用 `no_action/invalid` 结果，不在上层擅自发 HOLD。

这条链已用单臂 pose 虚拟 Policy/Human、不同来源频率、队列淘汰、缺失 required role、
trace sink 故障和重复目标拒绝做本地无设备验证；尚未连接真实 VIVE、真机执行器或
模型服务，也不证明 Active takeover 已完成。sync deploy 中 observer 仍位于当前
`prepare_next_action()` 形成正式 emission 之后，因而只能观察，不能插入既有安全与
执行链。Active 接入仍需单独拆分“emit → composition/proposal/authority → canonical
prepare”的所有权边界并保留现有 arm-hold/lookahead 语义。

### 1.2 历史 CS-6b1f：pose takeover seam 与本体切换

本节只记录早期 replay→human pose-takeover profile，不是当前单右臂 PPO +
VIVE/gripper 应用的启动契约。

HIL 的 takeover、HOLD/cancel、readiness、emission promotion、publication barrier
和外部 recorder 都不携带“双臂”假设。`PassiveTeleopPoseSourceSpec` 现在按声明的
`reference_frame_by_effector` 与 `action_spec_hash_by_effector` 决定 pose effector
集合；默认值仍保留历史双臂 contract/hash，单臂只需建立一个新的数据化 source
contract，例如：

```python
from control_station.sources import PassiveTeleopPoseSourceSpec

human_spec = PassiveTeleopPoseSourceSpec.for_arm(
    "right_arm",
    reference_frame="right_arm_base",
    action_spec_hash="rm65-tool-tcp-pose-v1",
    source_id="human-single-arm",
    source_profile_id="single-arm-tracker-v1",
)
```

多个非双臂 effectors 也可通过 `for_effectors(...)` 显式声明。其余步骤仍复用同一
`TeleopSession(rig=None, recorder=None)`、`HumanTakeoverOperationSpec`、readiness
provider、promotion 和 recorder seam；不会为单臂新增 queue、authority、epoch 或
episode owner。spec 的 effector/frame/action-spec 集合进入 semantic hash，少一侧、
多一侧或错配都会在 source/target binding 阶段 fail closed。

当前 `CS-6b1f` MuJoCo target 是一个明确的 **RM65 coupled-bimanual capability
profile**，其 action catalog、pose→coupled-IK resolver 和 execution manifest 仍要求
`left_arm + right_arm`。因此把上面的单臂 spec 直接塞进该 profile 会被拒绝，这是
能力不兼容而不是 HIL seam 不通用。接入真正单臂本体时只需提供该本体自己的 target
catalog、pose/frame/IK（或其它 execution）adapter 与 capability/manifest evidence，
再复用同一 Station/HIL transaction；不得复制一套按单臂命名的 handoff/replay/record
实现。夹爪、6DOF/16DOF 灵巧手仍属于独立的 qpos/native component gate，不因 arm
cardinality 改成 pose。

## 2. 历史 CS-6b1f 最小验证（R1，无真实设备）

从仓库根目录、激活 `dexteleop310` 环境执行：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_control_station_mujoco_pose_human_replay.py::test_cs6b1f_sim_hil_promotes_passive_pose_and_keeps_publish_seam_unique
```

该测试验证：

1. replay 可以先成为唯一 authority；passive human poll 不产生 human proposal；
2. takeover worker 在 HOLD 下等待 takeover boundary 之后的 immutable emission；
3. virtual readiness 完成后，原 emission identity/content/timing 只 promotion/admit 一次；
4. human pose 经现有 safety、scheduled pose→qpos target 和 MuJoCo planner 执行；
5. 后续 passive emission 仍只通过已有 human publication seam 提交。

正式 capture/WAL/committed-reader 验证：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_control_station_mujoco_pose_human_replay.py::test_cs6b1f_formal_hil_capture_seals_and_rebuilds_pose_lineage
```

该测试会在 pytest 临时目录中创建 trace WAL、seal、commit 和 ActualDecisionView，
不会写入真实数据集，也不会连接设备。

## 3. 应用侧组装顺序

除下方这个 default-off 应用 profile 外，当前没有通用 HIL CLI；应用代码应复用
以下顺序，不要在 GUI 或 source 中另建控制循环。临时 PPO + 真实 VIVE/gripper
（仅驱动 MuJoCo）的具体 profile 入口是：

```bash
PYTHONPATH=. python -m control_station sim-hil --help
PYTHONPATH=. python -m control_station sim-hil \
  --dry-run --device cpu \
  --control-profile policy_human_allocation_scheme_a
```

将最后一个 profile ID 改为 `policy_human_allocation_scheme_b` 即可切换到三态方案；
不能在同一个 activation 内热换 profile，切换时应重新构造应用，从而得到新的
profile hash、allocation spec 和 continuity state。

只有显式传入 `--enable` 才会启动当前右侧 profile；`--left-q6-port` 会明确拒绝，
而不是被静默忽略。默认 `--gripper-input-source auto`：先探测显式
`--right-q6-port` 或 `--q6-config` 中的 right assignment；Q6 确实不可达时再扫描
可达设备，仍无 Q6 才回退到 calibrated angle trigger。多个 Q6 可达但没有明确
right assignment 属于配置歧义，会停止而不是猜测。可使用
`--gripper-input-source q6` 禁止回退，或用 `angle_trigger` 明确跳过 Q6 探测；
angle trigger 的配置和采样率由 `--angle-trigger-config-root`、
`--angle-trigger-rate-hz` 指定。当前右侧单臂 profile 只要求一个物理 angle
trigger；旧双侧 reader 的“两设备”要求不再泄漏到该 profile。

该 profile 使用一个 VIVE tracker（legacy packet slot 0 映射到逻辑 right）和一个
选定的 gripper input；它仍不构造或连接真实 RM65 执行 backend。Q6 与 angle
trigger 复用真机遥操的 observation source 和 `ScalarToGripperRetargeter`，Station
只选择 profile、显示 requested/resolved source 及 fallback reason。普通 `--gui`
仍是只读状态页；正式交互控制使用 `--control-gui`。

### 3.1 正式 Sim-HIL 控制 GUI

推荐从仓库根目录启动：

纯策略 + MuJoCo（无需 VIVE/夹爪/坐标 profile）可先直接启动同一 GUI：

```bash
PYTHONPATH=. python -m control_station sim-hil \
  --enable --control-gui \
  --control-profile policy_human_allocation_scheme_a \
  --acceptance-root teleop_records/sim_hil_acceptance
```

页面打开后点击“以策略基线启动动态分配”。该路径不会为了启动而探测 Q6、打开
angle-trigger 或等待 VIVE；若以后在同一 session 选择 Human 模式，才按当时的
设备选项和坐标 profile 执行接入。

```bash
# VIVE/OpenVR 主机；当前单 tracker compatibility 会发送 packet slot 0
SIM_HIL_RECEIVER_IP=127.0.0.1
python hardware/VIVE/mocap/vive_data_sender.py \
  --single --host "$SIM_HIL_RECEIVER_IP" --port 9999 --hz 300

# DexTeleop / MuJoCo 主机；单右臂 PPO + VIVE + angle-trigger Scheme-A
PYTHONPATH=. python -m control_station sim-hil \
  --enable --control-gui \
  --control-profile policy_human_allocation_scheme_a \
  --gripper-input-source angle_trigger \
  --acceptance-root teleop_records/sim_hil_acceptance
```

sender 与 receiver 不在同一台主机时，将 `SIM_HIL_RECEIVER_IP` 改为 MuJoCo 主机
的实际可达地址。该命令只把真实 VIVE 与 angle-trigger 用作 passive input，执行目标仍是
MuJoCo，`real_arm_motion=false`。命令启动后会自动在默认浏览器打开
`http://127.0.0.1:8787/`（可用 `--no-open-browser` 关闭），先只打开一个 Station 网页和一个
VIVE setup receiver；此时不会构造 MuJoCo target、PPO runtime 或 gripper reader。
网页同时显示实时左右 VIVE 来源、当前选中的坐标 profile 和按需标定控件。
Scheme-A/B 可直接点击“以策略基线启动动态分配”，此时不要求 VIVE、夹爪或坐标
profile 就绪，也不会打开这些设备；首次选择自动分配、人类相对修正、绝对接管或
rebase 时，才要求 Human 输入就绪并执行 HOLD → 接入 → Runtime Home → re-arm。
默认使用仓库内精确的
`vive_rm65_virtual_desk_coordinate_v1.json`，不依赖本机最近保存的
profile。只有显式传入 `--vive-coordinate-profile latest` 时，setup controller
才会选择当前机器的最新有效 profile；GUI 展示和随后 runtime 构造仍绑定
同一个已解析文件，不会二次解析 `latest`。若显式选择 `latest` 且目录中尚无有效
profile，则保留首次标定流程。
若 angle-trigger 使用非默认配置目录，再显式追加
`--angle-trigger-config-root <config-root>`；验收指定设备时不要使用 `auto`
代替明确的 `angle_trigger`，避免 Q6 探测或 fallback 改变本轮设备拓扑。

#### 单臂 Human demonstration 采集

需要采集可供离线 demonstration editing 使用的纯人类轨迹时，使用专用的
`human_demonstration` capture profile。它仍复用同一个 Station GUI、控制权、共享
safety、MuJoCo dispatch 和 ControlTrace episode owner，但只开放
`human_teleop`；PPO-only 控制、比例融合、动态分配、策略接管和 policy timeout
fallback 都会被服务端拒绝。该 profile 不创建第二套 recorder，也不会把 GUI 状态
当成执行证据。其启动被拆成两个明确边界：第一步只构造最终的 Human-only composition
合同并打开 passive MuJoCo viewer，Human source 仍未 attach，source/composition/
physics/ControlTrace episode 都未启动；第二步才接入 VIVE 与 angle trigger，完成
passive acceptance 和 Runtime Home，并同时开始遥操与记录。setup receiver 在第一步
可以继续显示 VIVE 就绪信息，但它没有接到 MuJoCo 控制链路。

当前仓库的 `latest` 坐标 profile 是左侧 profile，不能用于此右侧单臂命令。以下
命令显式固定到已通过离线绑定检查的右侧 profile；更换 VIVE 安装或重新标定后，应
把路径替换为新生成且 `active_sides=["right"]` 的具体文件，不要恢复成含义可漂移的
`latest`：

```bash
# 终端 1：VIVE/OpenVR sender；跨主机时改为 MuJoCo 主机的可达 IP
SIM_HIL_RECEIVER_IP=127.0.0.1
python hardware/VIVE/mocap/vive_data_sender.py \
  --single --host "$SIM_HIL_RECEIVER_IP" --port 9999 --hz 300

# 终端 2：只控制 MuJoCo 的单右臂 Human demo capture
PYTHONPATH=. python -m control_station sim-hil \
  --enable --control-gui --device cpu \
  --capture-profile human_demonstration \
  --capture-root teleop_records/sim_hil_demonstrations \
  --control-profile fixed_alpha_compatibility \
  --human-alpha 1 \
  --human-timeout-fallback hold \
  --gripper-input-source angle_trigger \
  --angle-trigger-config-root teleop_runtime/configs/hardware/angle_trigger \
  --calibration-profile teleop_records/calibration_profiles/20260627_112114_vive_rm65_virtual_desk.json \
  --vive-coordinate-profile teleop_records/control_station_vive_coordinate_profiles/20260808_220119_vive_coordinate.json \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json \
  --checkpoint sim/mujoco_temp_pipeline/records/20260728_215045_cube_lift_pregrasp_bridge_v5_center_guard/checkpoints/step_000050176_final.pt \
  --vive-port 9999
```

网页中先点击主按钮“打开静态仿真”。此时只刷新静态 viewer，可以用 MuJoCo
viewer 自带相机交互调整视角；方块和机器人不会被控制推进，也不会创建 episode。
VIVE 与坐标绑定就绪且视角确认后，再点击主按钮“开始遥操并记录”；该操作才会解析并
接入右侧 angle trigger，完成 passive input 检查和 Runtime Home，全部成功后才启动
控制与 episode 记录。
完成一次可见的抓取、抬升和短时保持后，点击“结束回合、封存并自动保存”；该操作会
先建立 terminal HOLD，再执行现有 ControlTrace WAL/seal/commit 事务。等待页面显示
“已封存并保存”或 capture 状态 `committed` 后再退出进程。中途退出或重建 runtime
会显式中止未封存的记录。当前一个进程采一条 demo；采下一条时重新运行终端 2 命令。

当前 GUI 将上述流程收敛成“本次轨迹”卡片中的一个主按钮，而不是同时暴露多个启动
按钮。主按钮会按服务端状态依次显示“打开静态仿真”→“开始遥操并记录”→“结束回合、
封存并保存”→“退出控制”；四步进度条、VIVE/坐标、angle trigger、记录计时和最终
保存路径同时给出。启动和封存期间按钮保持锁定并显示正在执行的真实边界，仍可使用
“中断当前等待”。若 Human passive preflight 失败，viewer 和已调整视角会保留，主按钮
直接变为重试；不会要求重新构造场景。运行中的“立即保持输出”保持可见，Home 重采、
重建和放弃则收在“恢复与放弃本次采集”中。记录链路故障只引导查看诊断，不会建议
绕过故障 re-arm。

封存得到的是完整 ControlTrace，不等同于已经进入训练集。离线导出仍会显式选择
decision span，并要求与该 span 绑定的任务成功证据；只有随后导出的
`safe_action` pose 和归一化 `gripper_close` 才进入通用 demonstration contract。
dispatch receipt、MuJoCo measurement 和碰撞/任务结果继续保留为独立证据。

从需要 Human 的控制模式启动时，Station 先关闭 setup receiver，再把其精确来源映射和
profile 路径交给新 application；随后解析 Q6/angle trigger 并验证 passive human
emission。纯策略/策略基线启动保留 setup owner，供之后按需接入 Human。原生
普通 acceptance 模式的 MuJoCo passive viewer 在所需来源检查通过后、realtime physics
worker 启动前构造；构造完成后会重新采集并验证一帧当前 human/policy emission，
再进入 composition preflight。这样既不会因窗口构造耗时把健康来源误判为 stale，
也不会放宽 TTL 或把旧帧改写为新帧。无 VIVE 等首次来源错误只留在网页中，不会
先弹出一个无信号且无法操作的 MuJoCo 窗口。`human_demonstration` 是显式例外：它
按上述两步流程先打开 viewer，之后才接入 Human。两种路径的 viewer 都只读取并同步
lower-owned scene，不推进 physics，也不成为新的 authority 或 safety gate。

等待输入或启动期间可点“中断启动 / 等待”。HTTP 线程只设置取消请求，实际资源
清理由 Station control thread 完成。失败或取消会关闭本次未启动成功的 application，
恢复新的 setup receiver，并允许重新标定或重试；不会复用已关闭 source，也不需要
为了可恢复的启动失败重启整个 Station 进程。

`--single` sender 将唯一物理 tracker 标记为上游 `left_wrist`。当前单右臂 profile
默认把该来源映射到逻辑输出 `right_wrist_pose`，因此无需第二个 tracker。GUI 中的
“交换左右 VIVE 来源”按钮只交换上游 role，保持下游逻辑 channel 不变：

```text
单侧：left_wrist ↔ right_wrist -> right_wrist_pose
双侧：left_wrist/right_wrist 交换 -> left/right_wrist_pose 保持不变
```

按钮只允许在 runtime 尚未启动、composition 尚未运行时使用。交换只修改
上游 role 映射，不检查来源身份、有效性或新鲜度；交换会清空旧 VIVE history，
并在对应 upstream role 已有 snapshot 时用该 snapshot 重新种入 causal history。
因此控制线程不会因为显示缓存与 causal query 不一致而制造一个额外的
`missing_channel` 周期。source mapping generation 变化也会通知下游 session
重置 pose baseline 和 retargeter smoothing；无效来源仍保持无效，不会被伪造为有效动作。
控制已经运行后不会允许直接交换，避免旧 pose 穿过映射边界或产生瞬时目标跳变。

Sim-HIL GUI 的“VIVE 接收诊断”与 real-teleop 一样显示左右 sender role 的
valid、age、seq、xyz/rpy 和 sender error，并额外标出当前哪个 role 被选入
`right_wrist_pose`。这些字段来自 mapping 之前的只读接收缓存，不参与 retarget、
composition 或执行。设备验收等待期间 status 页面仍会持续刷新：

- `waiting_for_packet`、`packet_count=0`：接收器已启动但没有 UDP 数据；检查 sender
  的 `--host/--port`、网络和端口占用，交换 role 不会解决；
- `left=有效`、`right=无效`：这是 `--single` 的预期结果，应保持
  `left_wrist -> right_wrist_pose`；按钮仍可用于显式切换来源；
- 切换到 `device '<right_wrist_unavailable>' not found` 后，输入 readiness 会
  变为无效，但可以再次点击交换恢复 `left_wrist -> right_wrist_pose`；
- `receiving` 但物理 role 无效：根据对应卡片的 sender error 排查 OpenVR/tracker，
  而不是修改逻辑映射。
- `running=true` 但 `worker_alive=false`：接收线程已经退出，检查
  `last_process_error`；这不是 sender stale，应重建 receiver。
- `packet_count` 持续增长但 `channel_update_counts` 不增长：检查显式
  `tracker_role`、logical source mapping 和 `last_process_parse_error_count`。
- GUI 的 `packet_age_sec` 使用 socket 到达端记录的 monotonic age；同时保留
  `packet_age_wall_sec` 供日志对照，系统 NTP/手工改时不会把正常包误判为 stale。
- `processed_packet_count` 与 `process_error_count` 只表示解析/发布进度；即使解析暂时
  延迟，`datagram_count` 和 packet age 仍反映真实 UDP 到达情况。
- `dropped_stale_packet_count` 是 receiver 在一次读取中清空的旧 datagram 数；这些样本
  仍进入有限历史，实时输出只发布最新样本，不等于 UDP 丢包。
- `packet_age` 新鲜但 `human_device_ready=emission_stale`：继续查看
  `last_rejection_reason`、session state、hold/fault；packet freshness 不等于
  accepted human emission freshness。

timeout 异常也会附带 `vive_receiver`、packet count/age 以及左右 role facts，便于
在无 GUI 环境中做同样判断。

网络抖动容忍与回退是 Sim-HIL 独立配置，不改变真机遥操的 tracker-loss HOLD：

```bash
PYTHONPATH=. python -m control_station sim-hil \
  --enable --control-gui \
  --human-emission-timeout 5 \
  --human-timeout-fallback policy
```

`--human-emission-timeout` 默认 `5 s`，表示 accepted human emission 的最大允许
年龄；它不会把旧位姿重新标成新位姿，也不会修改 `TeleopSession` 的自动 HOLD
门槛。`--human-timeout-fallback hold`（默认）在窗口耗尽后交给下游保持输出；
选择 `policy` 时，运行中的人类输入超时会用一帧新 PPO 输出切换到 PPO，并锁存该
选择，VIVE 恢复后不会自动夺回控制权，必须在 GUI 中重新选择人类/混合模式。
GUI 的“人类输入超时回退”与“融合目标不可执行时回退”是两个独立选项，前者只
处理网络/人类输入新鲜度。

排查 GUI 服务本身时，先不要把空响应直接送入 JSON parser：

```bash
curl -i --max-time 3 http://127.0.0.1:8787/api/vive-status
```

只有确认返回 `HTTP 200` 和 JSON 后，再使用 `python -c` 或 `jq` 提取
`vive_receiver`；连接拒绝或空响应表示 Station/GUI 没有提供状态，不能归因于 VIVE
数据包内容。

GUI 的 intent POST 只表示请求已进入 Station 队列；交换按钮会在回执后等待同一
`intent_id` 出现在 `acceptance.last_result`，再从新的 setup snapshot 刷新当前映射。

页面的完整状态和 VIVE 状态分开轮询：完整 `/api/status` 低频刷新，轻量
`/api/vive-status` 约每 250 ms 只读取 receiver snapshot，用于实时更新左右卡片，
不会重复触发 policy/composition 状态计算。

如果单个右侧 angle trigger 尚未标定，先在终端完成一次显式硬件设置：

```bash
PYTHONPATH=. python -m hardware.angle_trigger --side right
```

单侧配置保存为 `assignment/right.json` 和 `angles/right.json`，不会覆盖 UMI/双侧
遥操使用的 `latest.json`。若原有双侧配置中的 right identity 与当前唯一设备匹配，
单侧 Sim-HIL 可以直接复用其中的 right assignment/calibration。GUI 的输入设置只
加载已标定配置，不会在 Station 控制线程中阻塞等待终端 `input()`。

同一只传感器更换 USB 口后，基于 USB location 的旧 identity 可能不再匹配。此时只
更新右侧 assignment，保留原来的角度端点：

```bash
PYTHONPATH=. python -m hardware.angle_trigger --side right --reassign
```

如果换成了另一只物理传感器，或者将原 left 传感器改作 right，必须同时重新标定：

```bash
PYTHONPATH=. python -m hardware.angle_trigger \
  --side right --reassign --recalibrate-angles
```

出现 `Could not uniquely resolve the configured right sensor` 时，系统不会在 GUI 中
自动改写 identity；完成上述对应操作后，直接重新启动 runtime 即可。

页面将设备设置、正式控制和数据封存分开：

- **VIVE setup / 按需标定**：使用已有有效 profile 时无需进入标定；需要时在同一
  GUI 依次完成 `+X 后→前`、`+Y 右→左`、`+Z 下→上` 和 wrist 中立姿态，并保存后
  选用。进入标定前要求所选通道有效，过程中可以取消并返回 setup。
- **正式控制**：`human_teleop`、`ppo_only`、`proportional_blend` 是三个并列模式。
  runtime 可以直接以任一模式启动，启动后也可任意顺序、任意次数切换。前两者的
  human alpha 分别为 1 和 0；比例混合要求 alpha 严格位于 `(0,1)`。
- **运行控制**：HOLD、恢复运行、Runtime Home 和整体 runtime 重建不承担模式解锁
  职责；它们只处理对应的 lifecycle 操作。
- **数据封存**：操作者可在正式 runtime 已启动且 trace 正在记录时结束 episode 并
  封存，不要求访问或“确认通过”全部三种模式。该操作先结束运动 episode，再执行
  ControlTrace WAL flush、seal 与 commit。

模式切换仍发布 typed parameter record，并由后续 control cycle 反映到
composition/dispatch/difference 状态；这些是运行事实和诊断信息，不是进入另一模式
的解锁条件。

浏览器只向有界队列提交版本化 typed control intent；唯一 Station 主循环执行
设备启动、alpha 更新、`app.step()` 和 capture finalize。HTTP 线程没有 action
payload、authority、dispatch 或 dataset writer 所有权。普通 `--gui` 不带 POST
能力，`--control-gui` 也必须同时显式提供 `--enable`。旧
`--acceptance-gui` 仅作为命令行兼容别名。

取消当前操作与退出属于 out-of-band lifecycle intent，不依赖可能已过期的状态
revision；启动、Home 或恢复运行正在等待时，状态页继续提供缓存的只读 runtime
投影和实时 VIVE 状态。关闭 MuJoCo 原生窗口会先建立下游 HOLD，再退出 Station
循环并确认 viewer 原生资源已结束，避免把已关闭窗口留成后台运行的仿真。

当前封存内容是 **ControlTrace core**，覆盖 source、composition、target/reference、
measurement、difference、HOLD 与 commit 事实；它不是包含相机/触觉/native stream
的正式 v2.5 数据集。完整传感器数据接入仍需由 collection runtime/writer 绑定，不能
由 Station 页面自行拼装。

### 3.2 Compose 频率与下游位置控制 profile

Sim-HIL 的最终 compose/authority 频率不是写死在 Station loop 中，而是由
`--position-control-profile` 与 `--compose-hz` 共同确定：

```bash
# 默认：compose/authority 30 Hz，policy source 仍按模型配置运行，
# reference/IK 60 Hz，command 120 Hz
PYTHONPATH=. python -m control_station sim-hil --help

# 例：保留 policy source 15 Hz，compose 45 Hz；下游仍为 60/60/120
PYTHONPATH=. python -m control_station sim-hil \
  --position-control-profile position --compose-hz 45 \
  --enable
```

`--rate` 是旧参数的兼容别名，表示同一个 compose 频率；同时提供两个参数时
数值必须一致。`policy_hz` 只表示模型/source 产出频率，不会被误当成最终
compose 频率。选择的 profile 会校验 compose 频率不低于其输入 cadence、且不高于
其 reference cadence，并把 reference/IK/command clocks 原样绑定到下游 target。

Station 配置层不再额外施加 60 Hz 上限；当前 MuJoCo authoritative ActionArbiter
仍保留既有 `(0, 60]` production gate，因此超过该能力的值会在 target 构建时明确
拒绝，而不是悄悄改变执行语义。这个 gate 与 source、replay、teleop 和 RTC 的
输入/worker/reference/backend 频率无关；本次修改只作用于 default-off Sim-HIL。

Sim-HIL 的物理 VIVE jump 门限与正式遥操一致，为 `0.45 m / 160 deg`，不再误用
通用 fixture 的 `0.10 m / 45 deg` 默认值。若 latest-value consumer 超过
`0.25 s` 未轮询，下一个有效样本只重建相邻比较基线，不把跨停顿位移误报为
`translation_jump`；无效、过期和后续相邻 teleport 仍按原 safety monitor 处理。

### 3.3 坐标 profile 与启动 Home

Sim-HIL 不再把一个真机 calibration 文件同时解释成 VIVE 坐标、目标基座和
robot Home。启动绑定分成：

```text
ViveCoordinateProfile
    VIVE world/operator -> station retarget；tracker offsets

TargetFrameProfile
    exact MuJoCo URDF/scene -> profile-selected arm_base

ToolFrameProfile
    exact tool artifact -> tool_tcp semantics

SessionAnchorReceipt
    stored human Home + measured reset-state sim TCP
```

默认 coordinate profile 位于
`teleop_records/control_station_vive_coordinate_profiles/`，不包含 `T_D_B`、
workspace、qpos 或 Home。MuJoCo target 对声明 scripted reset 的任务复用
task-owned `CubeLiftKernel.reset`；dry-run 会执行这个纯 sim 初始化，但不会启动
设备、authority、runtime worker 或控制循环。随后从当前 scene 测量选中侧的
`T_arm_base_tool_tcp`，因此 human 零偏移对应当前 sim TCP，而不是历史真机 Home。
四层 ID/hash 与选中侧 pose 都进入 receipt，并通过同一个
`station_retarget_x_forward_y_left_z_up_v1` frame contract 绑定。任一来源、侧别、
tool frame 或刚体变换不一致都会在 passive source 构造前拒绝。

当前 human Home 仍来自兼容 calibration，receipt 会明确记录为 stored source；
这不是实时 VIVE recenter。未来真机接入必须用实测 base/tool/当前 TCP 生成新的
target/tool/session evidence，不能复用 sim receipt，也不能把 URDF 派生结果声明为
真机测量。

## 4. 使用 `sim/mujoco_temp_pipeline` checkpoint

### 4.1 先做独立策略基线

当前 randomized Pick checkpoint 必须与其 task config 成对传入；不要只替换
`--checkpoint` 而沿用 Sim-HIL 的默认 task：

```bash
CHECKPOINT=sim/mujoco_temp_pipeline/checkpoints/cube_pick_goal_delta_xyz_gripper_v2_s211_bootstrap.pt
TASK=sim/configs/tasks/rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json

# 不启动设备、authority、worker 或控制循环；只构造 MuJoCo task 做契约 preflight。
PYTHONPATH=. python -m control_station sim-hil \
  --dry-run --device cpu \
  --control-profile policy_human_allocation_scheme_a \
  --checkpoint "$CHECKPOINT" \
  --task-config "$TASK"

# 通过 preflight 后，才显式启动 sim-only 的交互应用。
PYTHONPATH=. python -m control_station sim-hil \
  --enable --control-gui \
  --control-profile policy_human_allocation_scheme_a \
  --checkpoint "$CHECKPOINT" \
  --task-config "$TASK" \
  --gripper-input-source angle_trigger
```

该策略是 31-D actor observation、4-D `xyz+gripper` latent、7-D task action；
adapter 将其作为 `sequential_target_delta` 物化为绝对右臂 pose 与绝对 gripper
qpos 后才进入通用 composition。它不是 absolute policy，也不是 action chunk/RTC
preview。当前本地 artifact identity 为 checkpoint SHA256
`a5d6c28a063df3969bd2044dc310f3236a75853dd599c98f05037cfe237ed57e`，配套 task
hash 为 `d52b82bd8aaa9ea8a2baa450edb7ee496ad6174d1649e0d15e17eb9db8c23431`；若文件发生
变化，应重新做 compatibility preflight，而不能沿用这里的身份结论。

临时 PPO pipeline 的默认 checkpoint 和 task config：

```bash
CHECKPOINT=sim/mujoco_temp_pipeline/checkpoints/latest.pt
TASK=sim/configs/tasks/rm65_right_cube_lift_rl_normal_v1.json

PYTHONPATH=. python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint "$CHECKPOINT" \
  --task-config "$TASK" \
  --episodes 8 --num-envs 4 --device cpu
```

有图形环境时再运行 viewer：

```bash
PYTHONPATH=. python -m sim.mujoco_temp_pipeline.play \
  --checkpoint "$CHECKPOINT" \
  --task-config "$TASK" \
  --episodes 1 --device cpu --print-safety-events
```

换 checkpoint 或 task config 时必须保持 checkpoint hash 兼容。loader 会在 hash 不一致时
fail closed；`--allow-task-config-mismatch` 只用于碰撞回归诊断，不能作为 HIL 或策略部署配置。

### 4.2 接入 state-only OpenPI action chunk

该接入分为三层：`robot_motion.deploy.ThreadedPolicyEmissionSource` 是后端无关的
非阻塞 chunk/emission source；`deployments.openpi.client` 只负责 prepared artifact、
OpenPI 协议和 v2.2 action codec；
`control_station.sim_hil_profile.openpi_policy_source` 在应用层把 MuJoCo scene
采样为通用 `DeploymentObservation` 并绑定所选 provider。composition、下游 safety、IK、reference
和 MuJoCo dispatch 均沿用已有链路，默认 `temp_ppo` 路径不变。

先在模型机从 checkpoint 与训练 profile 生成一次 prepared deployment artifact，并做
不加载模型的检查。输出 JSON 中的 `policy_artifact_json` 是客户端必须使用的文件；不要
直接使用 checkpoint 目录中的原始 `policy_artifact.json`：

```bash
cd /home/user/zc/DexTeleop
CKPT=/home/user/zc/openpi/checkpoints/<config>/<experiment>/60000
PROFILE=deployments/openpi/configs/training/writing_stage1_0804_0806_state_h16_absolute_60k_v1.json

PYTHONPATH=/home/user/zc/DexTeleop:/home/user/zc/openpi/src \
  /home/user/zc/openpi/.venv/bin/python -m deployments.openpi.server \
  --checkpoint-dir "$CKPT" \
  --prepared-training-profile "$PROFILE" \
  --openpi-root /home/user/zc/openpi \
  --host 0.0.0.0 --port 8078 \
  --prepared-config-check-only
```

将该命令报告的 prepared artifact 原样复制到 Sim-HIL 主机后，先做无 server、无设备、
无 worker、无控制循环的本地检查：

```bash
cd /path/to/DexTeleop
POLICY_ARTIFACT=/absolute/path/to/copied.prepared.policy_artifact.json

PYTHONPATH=. python -m control_station sim-hil \
  --dry-run \
  --policy-backend openpi_state \
  --openpi-policy-artifact "$POLICY_ARTIFACT" \
  --openpi-host <model-server-host> --openpi-port 8078 \
  --openpi-action-hz 15 \
  --openpi-policy-hz 15
```

模型机正式启动 server 时去掉 `--prepared-config-check-only`。本地 sim-only 交互启动为：

```bash
PYTHONPATH=. python -m control_station sim-hil \
  --enable --control-gui \
  --control-profile exclusive_truncated_hitl \
  --policy-backend openpi_state \
  --openpi-policy-artifact "$POLICY_ARTIFACT" \
  --openpi-host <model-server-host> --openpi-port 8078 \
  --openpi-action-hz 15 \
  --openpi-policy-hz 15
```

当前 state-only writing artifact 观测左右双臂的 16D policy state，并输出双臂 chunk；
现有 CubeLift Sim-HIL target 只把其中右臂和右 gripper 注册成 contributor。dry-run 会把
该投影明确报告为 `bimanual_model_to_right_arm_and_gripper_contributor`。这足以验证通用
worker、chunk 顺序、composition 和右侧下游接线，但不是 writing task 兼容性，也不是完整
双臂执行验收；完整验收仍需匹配 writing scene 和双臂 target profile。
当前工作树中的历史 `cube_lift_normal_v1.pt` 可能由旧 task JSON 生成，若上面的 evaluate
报告 mismatch，不要使用 override 冒充有效策略。先生成与当前 task config 匹配的短 smoke
checkpoint：

```bash
SMOKE_CHECKPOINT=/tmp/dexteleop_hil_cube_lift_normal_current.pt
SMOKE_RECORD_ROOT=/tmp/dexteleop_hil_temp_records

PYTHONPATH=. python -m sim.mujoco_temp_pipeline.train \
  --task-config "$TASK" \
  --checkpoint "$SMOKE_CHECKPOINT" \
  --record-root "$SMOKE_RECORD_ROOT" \
  --run-name hil_smoke \
  --num-envs 1 --rollout-steps 8 --minibatch-size 8 \
  --update-epochs 1 --total-timesteps 16 --device cpu

PYTHONPATH=. python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint "$SMOKE_CHECKPOINT" \
  --task-config "$TASK" \
  --episodes 1 --num-envs 1 --device cpu
```

该 smoke checkpoint 只验证当前 task/action/normalizer/safety 链路，不代表策略质量；
正式策略仍需足够训练步数和独立成功率验收。

`privileged_expert` checkpoint 只能作为 simulator/debug teacher，不能进入真实或通用
policy source。

### 4.2 当前动作契约与 CS-6b1f 的不兼容点

| temp pipeline 输出 | CS-6b1f 要求 |
|---|---|
| 7-D：右 TCP xyz/rotvec delta + 右夹爪 close delta | 双臂 `bimanual_ee_pose_v1` arm action |
| 右臂控制，左臂固定 home | 左右 arm 同一 coupled selection group |
| qpos/native gripper action | arm source/selected/safe pose；hand 另有 qpos component gate |
| `CubeLiftEnv` 自己的 observation/safety/task loop | 已启动的 authoritative MuJoCo runtime + immutable emission |
| normal 15-D observation；privileged 31-D observation | 目标 runtime 的已声明 observation contract |

因此 checkpoint 仍不能直接传给历史
`build_scoped_cs6b1f_mujoco_pose_human_replay_station()`。当前单右臂应用使用
`TempPPOHILPolicySource`：它验证 checkpoint/task/normalizer，读取当前 MuJoCo TCP
作为初始 anchor，把 7-D policy delta 解码为持续维护的绝对右臂 tool pose 与绝对
gripper qpos，再形成正式 `PolicyActionEmission`。它不实现第二套 IK、安全、authority
或 dispatch；这些仍由下游 MuJoCo target/runtime 所有。

当前 `control_station sim-hil` profile 是独立、default-off 的
`right_arm: cartesian_pose + right_gripper: physical_joint_qpos` component path，
不会改变历史 CS-6b1f replay/human 路径。HOLD 后仍需显式 re-arm，不能自动清门。

这条路径仍然复用 `PolicyActionEmission` / `PolicyDeployRuntime` 的 lower-owned
emission seam；模型侧不需要为 RTC 或 Station 增加专用 adapter。需要的只是应用侧
observation/action representation adapter，并且 adapter 必须保留 checkpoint schema、
task-config hash、normalizer 和 source lineage。

### 4.3 Policy + human component ControlTrace profile

`deployable_sim_immediate_component_policy_human_composer_v1` 描述当前直接
policy + human component composition 的正式 trace 拓扑。它要求真实的 policy/human
emission、component input admission、activation、composition evaluation/disposition、
component target request/completion/admission、混合前 source difference 和执行 measurement
alignment；旧 replay emission、旧 Station
intent/operation/publication barrier 以及旧 composition-session reducer facts 明确为
not applicable，不能为了 seal 而合成。

该 profile 当前标记为 `runtime_integrated`。控制 GUI 已接
`EpisodeTraceRecorderSink` 的 prepare/bind/release/seal/commit 与 committed-reader
回读；writer/journal 失败会使本次 capture 不可提交并触发下游 HOLD。这里的完成范围
仅是 ControlTrace core，不包含视觉、触觉或其它 native streams，也不等同于正式
v2.5 数据集。

## 5. 进入下一阶段前的验收

- `evaluate` 使用的 checkpoint/task hash 一致；normal policy 而非 privileged expert；
- shadow adapter 输出 finite、shape 与 contract 一致，且不调用 `submit_*` 或 authority API；
- 每个 policy output 有唯一 immutable emission identity、source validation 和 timing；
- PPO 原始 TCP delta 必须先由 `TempPPOHILPolicySource` 解码为绝对 pose；arm qpos
  不进入 composition；
- hand/gripper qpos 只有在 active-mask、joint order、units、bounds、safety 和 downstream receipt 都冻结后才可 compound；
- 默认关闭时，既有 `sync_queue_v1`、legacy replay、teleop、deploy 和 RTC 测试结果不变。

## 6. 相关入口

- [Control Station README](../../control_station/README.md)
- [Control Station architecture visual guide](../architecture/dexteleop_architecture_visual_guide.md)
- [MuJoCo temporary PPO pipeline](../../sim/mujoco_temp_pipeline/README.md)
- [MuJoCo simulation README](../sim/README.md)
