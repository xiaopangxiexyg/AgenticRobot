# MuJoCo checkpoint 组件与接入进度

2026-10-26：继续原隔离分支，已合入固定 development `6ecb2307b49c`，保留其
primitive scene v2、观察和 DMTAC 对齐改进。在 native scene、acceleration planner 和 actuated
owner 组件上新增完整 manual execution rig 的可信内部 cut；随后补齐 source admission、
execution authority、decision owners 三个控制状态组件及原 no-model target 的时钟接缝，
随后补齐完整 logical core 和标准 coupled IK/retargeter 的 native 数据恢复组件。
本轮进一步将这些 owner 与 runtime 水位通过原 no-model factory 装配成独立的
manual/immediate target；这是可信内存内分支，不是完整节点磁盘格式。Gateway 会话、
上层 agent 上下文与 RPent 联合恢复仍待接入，现有入口不自动启用恢复。

上轮组合源码 `25d5df1cbfda` 的21模块回归为 **708通过、1跳过、1失败**。
唯一失败是 `test_current_arm_runtime_binding_uses_exact_current_inputs`：V18冻结源
绑定与候选的 actuator `core.py`、`agent_skill_target.py`、`runtime.py` 三项源码不同。
原 V18 文件与其历史验证未改写，arm profile、simulation contract 和 MuJoCo executor
未更改；不能用原 V18 的验证声明替代当前候选资格。
完整 non-hardware 验收另在预检因缺少固定 B601 子模块 URDF 返回2，未启动任何 shard。
该失败回执保持原样。本轮已在独立候选中准备与主线相同 gitlink 的 B601 子模块；
不会修改主线子模块或共享源码。

观察组件阶段固定源码 `db09fe458444` 的25模块组合回归为 **679通过、1跳过、零失败**。
历史 checkpoint observer 绑定（原候选 V25，现另名保留）在保留 V24 原件的基础上登记上述三项变化、20项候选变更依赖及
39项既有直接 owner／record／编码合同；147项输入／144项源码均逐字节匹配该提交，并绑定真实消费者日志与 JUnit 回执哈希。
当前合流已保留主线模型图 V25，两份前驱证据均不会冒充合流资格；组合绑定待单独生成。
它不重新拟合 arm profile；唯一跳过项是未安装的 RPent 配对扩展，不代表 RPent 联合
验收。完整仓库验收另行记录，尚未提升 development。

## 真正开启的 Cartesian 执行观察

`sim.mujoco.deploy.observation_checkpoint` 为原标准
`MujocoExecutionDifferenceObserver` 增加完整 owner companion。runtime 仍是唯一
观察生产者，sink 仍是原被动配对 owner；没有新增日志、调度器或补偿发布路径。

- 保存源 `ExecutionObservationBinding`，其 `bound_at` 参与 binding ID/hash；恢复时
  用原 plan/capability 及源时间验证，不回拨时钟，也不以 cut 时刻生成新的身份。
- runtime 的 emitted/dropped/latest sequence、last-published 时间及本地 watermark
  原值保留。sink 单独保存它实际收到的 watermark、完整有序 pending map、最近
  association、容量和计数；拒收后两份 watermark 允许不同，不补发或重放旧帧。
- 重复 key 移尾、容量淘汰与 unmatched measurement 沿用原 observer 行为。
  `emitted <= latest` 可由原发布顺序保证，但不能用 `latest=emitted+dropped` 或
  `pending=reference_count-measurement_count` 推导保存状态。
- 原 FK builder 绑定新 scene/model，并新建 private scratch；验证实际 binding/hash，
  拒绝 scratch 别名到 live `MjData`。只复用同一个 hash helper，FK 算法与 hash domain 不变。
- capture/restore 不采样、不 publish；恢复只允许未发布的新 runtime/sink，不能原地覆盖
  活动 owner。未知 sink、projector 或配置错配明确拒绝。

首片只扩展原 manual/immediate 单臂 Cartesian plan，reference/measurement 都为
`full_frames`。这里表示每次原 runtime 的 observation cut 都采样，不表示每个1kHz
physics tick都记录。一次 step 的 reference→watermark→measurement→watermark
发布必须完整结束；caller fence 还应排除直接 sink 发布。

原生测试覆盖 pending/active/HOLD、数据拒收和 watermark 拒收后的分支继续、源绑定
时间、非空 cache、重复 key/淘汰/配对、错 schema/时间/计数、live scratch/实际FK绑定
变化，以及保存／恢复零 publish。最后评审计数反例先2失败，修复后观察及原消费者
选集31通过。它仍不证明 Gateway 的 application、operation 或 RGB-D source
已整体恢复；这些既有 owner 的准入检查不能关闭。

## TCP 安全参考与包络续跑

`robot_motion.target_safety.checkpoint` 为原 `TCPPoseEnvelopeSafetyRuntime` 保存完整
catalog/config、稳定 reference、独立 safety-reference epoch、操作内 active 记录以及
最近 projection。原 `snapshot()` 只有 hash 摘要，不能用作恢复载荷。

- 复用闭集 record codec 和原 typed 构造器，验证数据布局、缓存 hash、catalog/selection/
  constraint/Route、anchor 和 activation receipt 的关系；projection 用原纯投影函数核对。
- 新 owner 由原构造器建立新锁和索引；active entry 指回新 catalog。保存／恢复不调用
  commit、invalidate、activate 或 deactivate，不增加 reference/execution epoch。
- 稳定 reference 保留原 pose、历史采样时间和初始化事件；普通 HOLD 只清 active/projection。
  invalidate 后 `reference=None` 且 epoch 非零、absolute envelope 没有 anchor 都沿用原语义。
- 原 planning preview 闭包捕获原 owner/reference，不能复制到新分支；消费者在新 owner 上
  重新取得同一个既有查询接口。本组件不保存闭包，也不新增查询或执行 owner。

独立 owner 可验证 active 数据的纯投影续跑，但不据此恢复 Gateway operation/READY。
当前 target 只接入 **inactive** envelope；保留原 reference，通过原 factory 构造 exact
`RobotActionTCPPoseEnvelopeFilter` 并绑定新 owner，验证 clock/filter 身份。active envelope、
其他自定义 filter、外部 planned-motion owner 仍不属于此切片；坏 envelope 在加载原生分支前拒绝。
这不是新增 agent 动作策略或安全阈值，普通新建 target 及现有物理控制路径不变。

与 `6ecb2307b` 的隐私边界一致：这里的原生 checkpoint、历史 pose 和控制记录为私有恢复数据，
不注册为公共 Skill/Artifact，也不恢复已移除的公共 raw scene 字段。初始化来源认证和多组件
属于同一个有权封存的应用事务，仍由后续 application/store 接缝负责。

## Manual no-model target 的独立分支

`sim.mujoco.deploy.runtime_checkpoint` 保存完整 core 及 runtime 自有的 source sequence、
观察 adapter sequence、最新请求阶段、dispatch context、immediate target/admission
历史、reference index 和 trace 消费水位。检查缓存与原 journal 的完整、有序关系，
拒绝历史截短、遗漏已返回阶段、观察/接纳编号回退及伪造 backend receipt。
检查复用原 typed record、trace projector 和构造器，不重放旧事件来重建状态。

这里的三个“最新”不一定是同一个动作：最新请求 B 可能已拒绝，而 plant 仍执行 A；
B 也可能已经接受但等待下一 reference tick。HOLD 可以清掉物理参考而仍保留历史
index。恢复分别保留这些事实，不根据某一份缓存重新发动作或改写另一 owner。

`sim.mujoco.deploy.target_checkpoint` 组合原 rig/native、retargeter、core/runtime 和
resolver 诊断。恢复从原 factory 进入，使用源的原始 initial q 配置；不会把当前
measured q 误作新启动姿态，不执行 tracking reset、重新 grant、旧请求/trace 重放。
IK adapter 验证传入 owner 与恢复状态相同后才跳过普通初始 reset。普通新建路径
继续执行原 reset、碰撞检查、canonical safety、墙钟 adaptive IK 和唯一 dispatch。

```python
from sim.mujoco.deploy.target_checkpoint import (
    capture_motion_target, restore_motion_target,
)

# The parent fences ALL admissions, decisions, callbacks and physics ticks.
cut = capture_motion_target(source, prepared=source_prepared,
                            max_native_bytes=native_budget)
branch = restore_motion_target(
    cut,
    prepared=trusted_prepared_metadata,
    rig_config=trusted_rig_config,
    expected_model_sha256=trusted_native_model_sha256,
    binding_factory=build_fresh_component_binding,
    max_native_bytes=native_budget,
)
# branch is already running at the saved logical clock; do not call start().
```

`binding_factory` 必须针对传入的新 prepared scene 构造回调，不能复制捕获源 scene 的
闭包。原 shared-safety 回调现在由私有 typed callable 显式保存 rig/task/clock/profile
绑定，算法不变；恢复核对 safety、materializer、schema、frame 与 resolver 身份。
失败只关闭尚未发布的新 rig，源 target 和原证据保持不变。

首片仅接受原 standard coupled IK、双臂 actuated manual rig、单一 direct component
source、immediate qref、上述标准 observer 或 off、无外部 trace sink、无额外 safe-motion plan
或自定义 resolver/filter 的实际源配置。启用了这些组件的源明确拒绝，不在恢复时
关闭组件来凑通过。backend 已抛异常而无法确认完整 dispatch 返回的切点也明确拒绝；
不能把有 target 记录等同于执行已确认。

验收使用真实 CAD/MuJoCo/Pinocchio/Pink，覆盖新参考未激活、活动参考、后一请求过期、
HOLD、非零观察编号、恢复后下一次原 dispatch，以及各类回退/错绑负例。接续 physics
的 qpos/qvel 对应一致；新一轮 adaptive IK 仍受真实墙钟时间影响，不承诺两个串行
分支未来全部 telemetry 或 hash 相等。这不证明任务成功率或真实 agent 能力。

```bash
bash scripts/run_pytest.sh tests/test_mujoco_runtime_checkpoint.py tests/test_mujoco_target_checkpoint.py -q
```

后续应由 Gateway profile 组合上述 enabled observer 与 operation owner 的封存合同、
事务 fencing、磁盘 bundle 的实现/owner 身份和原子发布，再与 RPent 的 runtime／记录
前缀恢复组合验收。当前内存 API 需要 caller-trusted prepared metadata；不能将其
描述为进程重启后仅凭一个文件即可恢复完整节点。

## 完整逻辑 control core 的冷恢复

`robot_motion.action.authoritative_checkpoint` 将三个已验证组件与 authority、gate、
candidate authorization stores、active segments、previous selection、完整 trace journal
及各自水位合为同一内部 bundle。首片限定 immediate activation、完整同步 decision/
admission/取消事务结束且调用方排他的边界；不支持 active scheduled segment 或独立
异步 ACK。后端、IK、安全、adapter、observer 等外层状态仍由 target profile 负责。

原 `AuthoritativeControlRuntime` 的私有 `_checkpoint` 构造参数在普通依赖/能力检查
之后、owner 创建与 grant 之前装配恢复状态。普通初始 owner 注入仍按原 initial epoch
检查；恢复既有 live epoch 不改写 `config.initial_*`，不新发 token，不重放旧 trace。
绑定 backend 名称；stateless arbiter、trace context、compound-profile 派生索引偏离
固定实现时拒绝。回调来源及独立 backend 的证明由外层 profile/store 完成。

验证不仅检查每个 owner 自身的 hash，还通过原 projector 核对它们与 journal 同源。
按已应用的 dispatch/cancel/ACK 事实检查最终 active/previous/HOLD 状态，拒绝将另一个
合法分支的同 ID proposal 拼入当前历史，或复活已经被 HOLD 清除的 segment。该过程
只核对事实，不运行 arbiter、工具、IK 或动作。segment generation 与 execution registry
generation 是不同计数，不强迫相等；合法 READY＋hold_active、已过期但尚未处理的
segment、已 prune proposal 的旧 authorization/previous 引用均保留。

`robot_motion.action.checkpoint_codec` 提供固定 record 白名单的 data-only 字节编码，
不接受载荷指定模块/工厂，不保存 Python callable，不用 pickle。未知字段/类型、重复
JSON 键、错误数组字节数与输入 byte budget 均拒绝；解码不等于恢复或来源认证。
外层 store 负责可信来源、固定实现/profile、原子发布和磁盘配额；`max_bytes` 不等于
进程内存硬上限。此格式只承载逻辑 core，不是完整 MuJoCo 或 Gateway 节点格式。

```bash
bash scripts/run_pytest.sh tests/test_authoritative_checkpoint.py tests/test_authoritative_checkpoint_codec.py -q
```

验证包括非空 active 后再接纳新 proposal/dispatch、取消后非零 epoch、READY＋HOLD、
非空 authority 与 prune 后 replay authorization、旧状态回退、跨历史拼接、错误
embodiment、terminal 伪装及目的配置错配。新解释器从磁盘读取后恢复独立 fake backend
并继续原到期取消，事实 hash 与原进程相同。另在真实 CAD target 的 Cartesian 请求
变化后，经过原 IK/dispatch 和30个 physics tick，对完整非空 core 做编码往返。
前者证明逻辑 core 冷续跑；后者证明真实执行路径的数据可封存，均不证明恢复后的
完整 native target 或 Gateway 已可执行。原生任务成功、真实模型与硬件未在此验收。

本阶段合入固定 development `9dd06cbbb8be` 后，完整 core、三个 owner 组件、旧 runtime/
dispatch、原生 target、MuJoCo 与 domain boundary 的组合回归为 **153 passed，15.74秒**。
另保留三个先失败的反例，证实旧 active/previous 状态回退及错误 embodiment 会被拒绝。
没有以补发 grant、删历史或跳过执行回执来取得恢复通过。

## 已保存的实现

### Coupled IK 与 retargeter 的冷恢复

`robot_motion.rm65.ik_checkpoint` 保存标准 `MultiTaskIKSolver` 的 native 状态、
原任务目标、barrier cache、积分步长/频率及诊断；`retargeter_checkpoint` 保存
last-safe/raw/target、admitted velocity、jump-HOLD、奇异性/分支锁存、平滑历史、
certificate sequence 等控制历史。两个 companion 没有创建第二执行环。

恢复复用原 `DualArmCollisionAwareRetargeter` 构造器的私有入口，在首次 tracking
reset 之前装配未发布的新 owner。调用者须提供源的**原始启动配置**，不能把 cut 时的
measured q 替换为 initial q。普通构造、原安全/碰撞约束、QP 和墙钟预算保持不变。
`restore_ik_solver` 是内部的未发布候选装配接口，不可用于运行中 owner 原地恢复。

```python
from robot_motion.rm65.retargeter_checkpoint import (
    capture_retargeter, restore_retargeter,
    encode_retargeter_checkpoint, decode_retargeter_checkpoint,
)

# The caller fences complete solving and owns the persistent publication.
state = capture_retargeter(source, max_native_bytes=128 * 1024 * 1024)
encoded = encode_retargeter_checkpoint(state)
decoded = decode_retargeter_checkpoint(encoded, max_bytes=8 * 1024 * 1024)
branch = restore_retargeter(
    decoded, config=source.config, max_native_bytes=128 * 1024 * 1024
)
```

编码复用已有 canonical envelope，仅承载固定 typed data、精确数字 bits/flags 和原生
bytes。未知 mutable alias 在复制前拒绝；仅保留明确的当前 q 与 barrier q-cache alias。
父 store 仍负责来源认证、代码/profile 身份、完整事务互斥和原子提交。最大 bytes 参数
是载荷限制，原生 serializer 在检查返回尺寸之前可能已分配缓冲区，不是内存沙箱。

首片固定 Pinocchio3.7.0/Pink4.2.0/Coal3.0.1，比较实际 SDK/ABI、原生库身份、解析后
QP solver、native 模型和 collision-pair 顺序。只有原两个 FrameTask、一个 PostureTask、
SelfCollisionBarrier 和默认 Pink limits 路径可用；`limits=None` 不表示无保护。
额外 constraints/tasks、retargeter/filter、observer、smoother、未知字段与替代 owner
明确拒绝，不在恢复时清空它们。实际开启的 query timings 也不在本片资格内。

Pinocchio 官方 binary codec 保留两份互不共享的 Pin.Data，以及 solver/configuration
共同持有的一份 GeometryData；不从当前 q 重算 FK/Jacobian/collision 来覆盖合法的旧
算法阶段。GeometryData 在可信同模型上先构造，重绑 fresh functors 后加载原 request/
result，包括 cached GJK/support guesses。恢复不执行碰撞查询或 QP。

官方 codec 将 result/contact 的旧 geometry 指针置空，也不序列化 result timings。
当前 Pink/原 solver 的消费依赖模型索引、distance/nearest points 和缓存，不依赖旧指针；
这里不能推广到其他 native 消费者，也不能用 GeometryData `==` 当完整性判据。数据
bytes 往返之外，测试另比较首个真实 native query、连续原 one-QP 和独立新解释器续跑。
barrier 的 safe-displacement 在初始化和 QP 后可以分别是 `(dim,)` 与 `(nv,)`，原值原样保存。

```bash
bash scripts/run_pytest.sh tests/test_rm65_ik_checkpoint.py -q
```

本阶段标准 IK 的19项验收覆盖在最终组合中。原 owner/constructor/rig/core 的13文件
受影响回归为239通过、104.86秒；再合入固定 development 的 paired RGB-D 增量后，
checkpoint、原 pose resolver、观察/反投影和边界组合为314通过、83.55秒。
两组不能相加为独立样本。独立评审已关闭 signed-zero、alias 资格、默认 limit/tangent
身份和合法 cache phase 的缺口；保留先失败的反例。新模块和测试的 Ruff/format通过，
原 target solver 的14项既有lint告警未增加，未为此重排或修改无关算法。

这仍不是完整 MuJoCo target。新解释器用例验证的是同配置独立 IK owner，未将整个
runtime/Gateway 一起恢复。未来 adaptive outer solve 会受实际墙钟影响；另外原
`ik_frame_total_ms` 位于 resolution hash 内，因此不能把未来整条 trace hash 相同当成
恢复正确的唯一条件。本片不删除遥测，也不修改其安全或时间预算语义。

`sim/mujoco/native_checkpoint.py` 是仅支持已验证 MuJoCo 3.10.0 的原生物理组件。
调用方必须已持有物理状态的排他边界；组件自身不暂停线程、不取得 Gateway admission
或控制 authority。恢复构造独立 `MjData`/model 候选，不修改运行中的 owner，
也不调用 reset、forward 或 step。

- 复用官方 native payload，保存编译模型、积分器和求解缓存，不另写全数组序列化器。
- 补充官方 3.10 codec 漏存的 9 个 scalar 和 3 个固定 vector：约束数量、awake
  有效长度、lazy-stage 标志及 solver 诊断。原生载荷与辅助状态共同参与校验。
- 检查版本、ABI、显式载荷大小限制、有限积分状态、模型身份与原生即时字节往返。
- 拒绝 process callback、plugin、活跃 stack/threadpool；这些检查不能代替调用方互斥。
- 记录源 compilation signatures。官方反序列化后的 signatures 为零，
  **不支持恢复 `MjSpec.bind/edit/recompile` 关系**。
- 仅处理可信来源的 native bytes；SHA 不提供来源认证，原生解析器也不是安全沙箱。
  `max_native_bytes` 限制序列化载荷，不等于反序列化总内存上限。

字段依据为 [MuJoCo 3.10 官方 codec](https://github.com/google-deepmind/mujoco/blob/3.10.0/python/mujoco/structs_wrappers.cc#L816-L975)
和 [原生字段定义](https://github.com/google-deepmind/mujoco/blob/3.10.0/include/mujoco/mjxmacro.h#L998-L1040)。

## 当前验证及其含义

在 Python 3.10.20 / MuJoCo 3.10.0 上执行：

```bash
bash scripts/run_pytest.sh tests/test_mujoco_native_checkpoint.py tests/test_robot_motion_trajectory_checkpoint.py tests/test_mujoco_actuation_checkpoint.py tests/test_sim_actuation_checkpoint_clock.py -q
```

上一阶段加上原 actuator、gripper、planner 与 execution rig 的受影响回归，共 **202 passed**。
新增模块与测试的 Ruff check/format 通过；旧 `scene.py` 和 `gripper.py` 原有的
11 条 lint 告警未增加，保留基线状态。本轮未做完整 profile、渲染、远端、模型或任务验收。

立即恢复要求未改写的 native bytes 完全一致；后续相同动作续跑比较所有已分配、
公开的语义字段及模型 hash，包含辅助状态和 contact/solver/warning/timer 成员。
墙钟 `timer.duration` 不作确定性判据；稠密求解下未使用的四个 sparse-J 索引不比较；
dual/island arena 按官方 codec 的实际启用条件选择，不能读取未分配指针。
按字段比较 contact 避免把 C 结构尾部 padding 当成物理差异，不改保存物或 live state。
原先失败及有限重复测试的证据保留，没有覆盖成最终通过结果。

## 新增的 owner 接口

| 层 | 入口 | 所保留的状态与边界 |
| --- | --- | --- |
| scene facade | `MujocoScene.from_native_checkpoint` | 在新 model/data 上重建 joint/body/geom/camera 索引和 metadata，不调用 forward/reset/step；projector/scratch 从新 scene 重新创建 |
| 位置 planner | `robot_motion.position_control.trajectory_checkpoint` | 完整 q/qd/qdd、目标及安全目标历史、滤波与制动状态、时间、generation/token；必填目标逻辑时钟，要求 command transaction 已结束 |
| arm/gripper wrapper | `sim.mujoco.actuation.checkpoint` | 固定 typed component，复用原 core 恢复；保留完整 safety projection、held command、缓存及 queued intent，不调用 wrapper reset，不改 native ctrl |

Planner 恢复使用调用方给定的 limits/configuration，必须与保存值一致，不从保存物放宽限制。
目标 q/sequence/publication 和 safe-target history 成组校验，未知字段与未完成 proposal 拒绝。
该 codec 留在 `position_control` 包内，rig 只消费其接口，不从诊断 snapshot 猜回状态。

SceneSpec 使用现有 typed `to_dict/from_dict` 重建；其只读映射不能整体 deepcopy。
实际双臂 CAD 模型已验证两侧手臂、两只夹爪及 `front/left_wrist/right_wrist` 绑定，
候选 native bytes/auxiliary 不变。元数据与完整 profile 身份的绑定仍属于后续装配责任。

Actuated owner 组件保存 actual native time，恢复时逐值核对，再检验 owner 的完成时刻。
不同 clock 的浮点累加与 tick×dt 误差按有界 ULP 处理，误差界必须小于半个 tick；
不调整任何时间或 deadline。这不是互斥证明，父 rig 仍须排除中途事务与并发接纳。
faulted owner、额外 guard/oracle、未知 owner 实现不在该首阶段资格内。

真实 CAD 用例在第 193 个 physics tick 保存非稳定状态：arm 有延迟命令、gripper
有非零速率和新 queued intent。新 scene 与新 owner 装配后，对相同后续输入逐 tick
比较了 40 个 tick 的 qpos/qvel/ctrl/act/warmstart、公开 snapshot 及 queued receipt。
Native 与 planner 各自有新解释器测试；完整 owner bundle 的磁盘/fresh-exec 格式
仍待后续 store 验证，manual rig 的 fork 子进程验证见下节。

既有 `reset(initial_state=...)` 可恢复 core pending，同时将新 wrapper 的本地 receipt
初始化为 `(0,None)`；该状态原样保存，不由 core 补造历史。若已有非零本地 receipt，
对应 pending 的 sequence/target 必须一致。完整 payload、来源及生命周期绑定由后续
profile/store 负责；本层不证明任意两个组件来自同一段历史。

## Manual execution rig 的完整 cut

`sim.mujoco.adapters.checkpoint` 将上述组件接入现有 rig，提供
`capture_execution_rig`、`restore_execution_rig` 与显式 `ManualExecutionClock`。
实现位于 rig 所属 adapter 包内，没有改写控制算法、增加 scheduler 或创建第二执行环。

首片资格明确限定为标准双臂、manual-step、acceleration-limited、physical arm＋gripper，
collision projection 由原 executor 承担。realtime worker、sample-aligned QP、direct-qpos、
upstream admission、physics observer、scheduled generation guards、额外 oracle/final guard、
faulted owner、未完成 proposal、未知实例字段及未消费 bootstrap 参数都拒绝。其他原有
运行模式仍按原语义运行；checkpoint 不隐式降级它们，也不清空状态帮助通过检查。

保存内容包括完整 pending/active reference、双臂 qref 及 metadata、accepted qcmd、
reference/command 水位、逻辑 clock 当前值与 origin、rig lifecycle tick 计数、完整
planner 和 arm/gripper core/wrapper。metadata 顶层按侧保留字典，内部支持闭合的基础
数据、tuple/list 和有限 numeric ndarray；复制后与源分离，未知 live binding 明确拒绝。
即时手部 intent 已进入独立 gripper 队列，不能伪造到 reference 的 scheduled gripper 字段。
正常 HOLD 在同一锁内同步完成；非空的待处理 HOLD reason 不属于合格完整事务。

Capture 在 rig→clock 锁中取完整 cut，前后检查 ready/stop/fault。排除 observer 是为了
避免 RLock 重入取得 `mj_step` 后、planner commit 前的半 tick；仅有 pending 为空不能证明
完整边界。`close()` 在取得 rig 锁之前设置 stop event，因此结束检查会拒绝已经观察到的
并发 close。应用仍必须先排空外部 admission；本地锁不冻结尚未提交的外部命令。

恢复构造未发布的新 scene/rig/owners/planners，不调用 rig `start()`、wrapper `reset()`、
native forward/reset/step，也不重放旧目标。检查配置、scene metadata 和调用方可信 profile
提供的 `expected_model_sha256`；保留原时钟坐标，不换成接收进程的 `time.monotonic()`。
核对两侧 committed/accepted q、held/core/next command sequence、active reference 与
planner target 身份，全部通过后才返回 ready 候选。失败不改变源对象。
Standalone planner codec 允许 initialized/reset-only 状态，但 started rig 的两侧必须
已有 priming target（sequence=0 有效）；不能把无目标组件装配为 ready rig。

既有 `start()` 仍只在已启动时幂等返回；后续 close→start 仍按旧语义创建新生命周期。
这不自动重新满足 checkpoint 资格：旧 start 重置 core 时钟但不重置 native time，若两者
已经不在同一完整 tick，就继续拒绝捕获。不能为了支持该模式而忽略时间错配。
另一方面，公开 bootstrap API 在非零绝对 physics time、已有 command 水位上构造的
一致 rig 可以封存；rig lifecycle tick 计数不得被误当成 actuator 的绝对 tick。

可运行的本地仿真验证：

```bash
bash scripts/run_pytest.sh tests/test_mujoco_execution_checkpoint.py -q
```

测试包含有运动与延迟状态的第193 tick、pending mailbox、后续相同输入、HOLD 与正常
生命周期、双分支隔离、并发 close、错误 metadata/reference 身份和不支持的 bindings。
另验证合入主线后的 v2 cylinder scene 与原 metadata/native 绑定。完整 native 字段比较
复用本页说明的官方分配条件和墙钟遥测边界，不丢弃 solver/contact/auxiliary 状态。
本阶段最终将新模块与原 actuator/gripper/planner/rig、native/scene、domain boundary
相关的15个完整测试文件组合运行：**303 passed，105.91秒**；独立评审已关闭发现的
三项组合检查缺口，新模块及测试的 Ruff check/format 和 diff 检查通过。

`ExecutionRigCheckpoint` 仍是可信内部对象，**没有定义磁盘导入格式或完整 profile 来源认证**。
fork 子进程验证检查的是独立恢复与同输入续跑，不等于 fresh interpreter 从磁盘恢复。
代码/owner/profile/父分支身份和原子存储由后续统一 store/应用负责，不能用组件通过来
跳过已有源码、SDK、记录前缀或恢复 owner 检查。native ABI、MjSpec 限制仍如前述。

## 运行控制状态与实际 target 接线

控制状态保留在原 owning package 的 companion module，不另建 dispatcher、调度器
或模型调用循环。三个组件均为可信内部数据；调用方须排除完整操作期间的并发写入，
单个 store 的锁不能证明整个 runtime 已静止。

| 组件 | 入口 | 保存内容与明确边界 |
| --- | --- | --- |
| Source admission | `robot_motion.action.admission_checkpoint` | 全局 admission 水位、producer 激活历史/活跃 epoch/计数、保留 proposal 及独立 producer sequence 水位；过期记录清空后也不能重新接纳旧 sequence |
| Execution authority | `robot_motion.runtime.execution_checkpoint` | 当前逻辑 token、acquisition/live epochs、owner generation、issued/consumed permit、取消/ACK 历史与期限、fault latch；拒绝未消费 permit 和进行中的 owner transition |
| Decision owners | `robot_motion.action.decision_checkpoint` | 原调度网格、tick index、pending trigger 总数与截断证据、最后成功 publication 的两个时间；不补 tick、不重播 trigger 或 publication |

恢复创建新锁和新容器，直接恢复原 typed facts；不重新 commit、activate、admit、grant、
cancel、register 或 acknowledge。Source proposal 的 admission 必须位于所属 epoch
激活与被替换之间；合法旧 epoch 的保留事实仍保留，不能只检查当前 active epoch。
全局 admission 水位可以高于所有 source 记录，因为 authority/gate 等 store 共享它；
这些 store 的状态由上述完整 core bundle 关联装配，不能只恢复 source 就发布 runtime。

Decision pending evidence 数必须严格等于 `min(total_count, evidence_capacity)`。
否则恢复时会改变下一个 tick 的因果证据和 hash。两项先失败的反例分别覆盖截断前、
截断后的证据丢失；修复没有改写计数、补造证据或改变既有 scheduler 算法。

Dispatch token 的 acquisition epochs 可以早于当前 live epochs，正常 cancel/recovery
后不能为求相等而重授 token。旧 consumed permit 的拒绝保持；pending ACK 不伪造为
完成，仍由原 tracker 按保存的 deadline 计算。`in_process_only` 的逻辑 registry 历史
仅可在独立 native 仿真 backend 的父合同内重建；它不授权重新控制同一个 live backend，
也不提供跨进程或分布式控制锁。实际 owner 转移继续使用原 HOLD/ACK/新 generation。

现有 `build_mujoco_agent_skill_motion_target` 新增显式 `clock` 参数，默认仍为
`time.monotonic`。同一 callable 传给原 rig、runtime/control owners 与 shared safety
anchor 采样。它不替换 IK 内部的墙钟性能计时或自适应预算，不创建另一套建场入口。
非法非 callable 在 native 资源准备前拒绝；左右臂选择、原 collision/IK 与正常 lifecycle
保持原实现。

实际接线用例：

```bash
bash scripts/run_pytest.sh tests/test_mujoco_agent_control_checkpoint.py -q
```

用例通过原 factory 构建真实 CAD/no-model target，正常 start 后推进17个 physics tick，
生成一个 no-action decision，再接纳一个未 dispatch 的当前位姿 proposal。在测试独占的
相同逻辑时刻保存四个组件（含 rig），恢复到独立 native rig 的同一个新 clock，核对
proposal/epoch/permit/schedule 状态及 qpos/qvel/ctrl/warmstart，另验证真实 safety anchor
使用注入时钟。该用例没有装配完整恢复 target、执行恢复后的 IK 或请求模型，不能据此
宣称 Gateway、teacher/student 分支 rollout 或任务成功。完整 SDK/runtime/记录前缀
的原恢复门禁保持不变。

## 修正的延迟队列时间窗

原 arm/gripper 恢复校验错误要求所有待处理事件都不早于“当前 tick 结束时刻”。
实际 step 在 tick 起点消费事件，例如起点 `0.009`、终点 `0.010`、事件 `0.0095`：
事件尚未消费，应在下一个 tick 处理，不能删除或提前执行。

- Arm 用现有整数 `physics_tick` 与物理周期恢复最后处理边界。
- Gripper 支持可变 dt，snapshot 增加 `last_step={physics_time_sec,timestep_sec}`，
  记录真实处理区间；恢复检查该区间的结束时刻与 resume time 一致。
- 已处理边界及其原消费 epsilon 内的事件仍拒绝。旧 v1 缺少 `last_step` 时维持
  原保守时间界限，不猜测历史。新 typed component 要求完整 core 往返一致。
- 公共 API 反例先复现 2 失败、1 通过；修复后覆盖区间内未处理事件、已处理事件拒绝、
  非 1ms 步长、损坏区间和旧格式保守行为。`hold_targets()` 与正常控制计算没有改变。

## 尚未完成，不能据此放开上层

1. Gateway admission、物理线程和控制 owner 的联合静止边界。
2. 将本轮已装配的 manual/immediate target 分支接到真实 Gateway profile。标准
   Cartesian execution observer 与 inactive TCP envelope 已随 target 验证；活动 operation、
   外部 plan 及 application lifecycle 仍待补齐。immediate pending reference 已验证；scheduled segment 及
   异步 activation 仍未纳入资格，不能直接放行。
3. 新实例中 FK、collision、payload、observer、scratch 等消费者的统一绑定。
4. 两份历史 V25 分别保留；合流后的当前源码须使用新的独立组合绑定与完整仓库门禁。
   profile 身份、持久化/原子提交、历史产物与恢复分支的来源关系尚待接入。
5. sleep 混合状态、完整 no-model profile、渲染和远端组合验收。
6. RPent SDK/runtime/记录前缀与上述下层合同的联合恢复。

这些组件尚无生产消费者。Gateway native checkpoint 和未完成后台执行的既有拒绝不变。
后续先检查最新 development/correction 与并行改动，再继续 owner 合同；不得将当前
组件回归通过视为完整 profile 已实现，也不得用 fresh start/reset 冒充精确冷恢复。
