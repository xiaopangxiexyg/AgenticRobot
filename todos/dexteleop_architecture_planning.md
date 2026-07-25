# DexTeleop：多源动作、多速率策略与 sim/真机统一控制架构规划

更新日期：2026-07-25

## 阅读导航

本文保留完整的长期架构、契约语义和数据集规划，适合作为设计事实源；不建议从头顺序阅读全部内容。

| 想回答的问题 | 建议入口 |
|---|---|
| 最终系统要形成什么闭环 | [结论](#结论)、[目标逻辑架构](#三目标逻辑架构) |
| 当前代码已经做到哪里 | [当前代码基线与准确边界](#一当前代码基线与准确边界) |
| 哪些原则不能在迁移中改变 | [架构不变量](#二必须长期保持的架构不变量) |
| 多速率、快慢策略如何共存 | [多速率时间模型](#四多速率时间模型)、[PolicyGraph](#六policynoderuntime逻辑-policygraph-与部署绑定) |
| proposal 如何统一为实际动作 | [ActionArbiter](#七actionproposalproposalbuffer-与-actionarbiter)、[现有链路接入](#八如何接入现有-canonical-action-与-position-链路) |
| human/replay/policy 与 real/sim 如何由统一 GUI 管理 | [`Control Station` 专项规划](dexteleop_control_station_plan.md) |
| v2.4 如何支持 DAgger/RL/replay/sim2real | [v2.4 数据集格式](#九v24-数据集格式多来源多频率与训练事实层) |
| 下一步按什么顺序实施 | [最佳实施路径](#十二最佳实施路径)、[阶段验收标准](#十四阶段验收标准) |

配套文档：

- [`dexteleop_control_core_landing_plan.md`](dexteleop_control_core_landing_plan.md)：实际文件、批次、前提、退出条件和回滚；
- [`dexteleop_architecture_visual_guide.md`](dexteleop_architecture_visual_guide.md)：用流程图、状态图和缺口表快速说明当前实现；
- [`dexteleop_rtc_landing_plan.md`](dexteleop_rtc_landing_plan.md)：异步 action chunk/RTC producer 的独立实施计划；
- [`dexteleop_control_station_plan.md`](dexteleop_control_station_plan.md)：多 source/target 控制实验台与最小统一 GUI；
- 本文负责回答“目标语义是什么”，各 landing plan 负责回答“下一批怎样安全落地”，visual guide 不新增规范。

### Control Composition ownership correction（2026-07-24）

多源融合的长期 owner 冻结为根级 `control_composition/`，不再以 `HIL` 命名，也不继续把通用 composer、
strategy、session/handoff reducer 和 source/output difference 分散在 `robot_motion` 与
`control_station`。该包只依赖 `robot_motion` 和纯数据契约；不得导入具体 sim、hardware、teleop、
deployment 或 Station。它的输出边界是带 exact lineage 的 proposal candidate，不是 authority、
canonical safety 或 motion permit。

迁移必须分批保持现有 hash、数值顺序、default-off profile 和入口行为。`robot_motion` 继续唯一持有
canonical action、proposal admission、authority、epoch、cancellation、safety、reference 和 execution；
`control_station` 继续持有 application operation、publication barrier、GUI/API 与具体装配；MuJoCo、
VIVE、RM65、OpenPI、checkpoint 和 task adapter 留在各自 owner。首批 package、ports、依赖守卫和文档
已经落地；`composition`、`observability`、protocol-neutral domain events、synthetic policy-node protocol
及`ControlCompositionRuntime`均已完成唯一实现的物理迁移；composition→v2.4投影由
`dataset_interface.composition_control_trace`独立持有。旧
`robot_motion.action.{composition,mixing_strategy,source_difference,guidance_feedback}`、
`robot_motion.deploy.{PolicyTopologyRuntime,policy_node}`路径已删除；本仓库调用方及唯一仓库外调用方
远端OpenPI均已审计，因此不保留会反转依赖方向的shim。
pure handoff contract/reducer 已首次物理迁入
`control_composition.lifecycle`；本仓库已无production或仓库外调用者依赖旧路径，因此
`control_station.handoff`兼容转发已删除。
composition-session拆分也已完成：operation spec、activation state/transition、re-arm spec/receipt/validation
及backend-neutral registry均归`control_composition.lifecycle`；Station controller仅继承registry并持有
`StationOperationResult`、publication barrier和lower HOLD/cancel的应用边界。
composition-session的四类v2.4 projector也已迁入同一dataset adapter；Station bridge只负责注入
runtime/embodiment及trace sink，不再拥有重复投影。

本轮进一步冻结其“完整框架”接口，而不扩大motion authority：`CompositionContext`现为representation-neutral
不可变上下文，可承载history/state/future-action/vision/label或任意新模态的contract-bound payload；
旧`PoseMixingContext`保持同一类型别名和原hash domain。pose与hand/gripper qpos strategy均可消费同一
context cut，context必须同时满足receiver clock、admission sequence和target-time TTL，不能读取live backend。
`CompositionEvaluationRecord`把本次spec、`ControlCut`、active producer epoch snapshot、selected input record
（含source lineage/audit metadata）、composition context snapshot、result、difference、typed diagnostic及实际emit的
domain-event batch作为单一只读返回接口；proposal admission仍是后续独立边界。外部dataset adapter可将这些
领域事件投影为v2.4事实；该改动保持空context下既有pose/qpos hash与数值行为，并未改动
teleop/replay/deploy默认控制链。

`CompositionInformationPlan`作为独立、加法式信息需求契约，不进入现有activation v2/hash。它在source
acquire/copy/freeze/admit之前声明composition真正需要的context channel、payload contract、maximum age及
`metadata_only/inline/blob_ref`访问方式，并可绑定optional difference spec。无plan时保持既有行为；有plan时
undeclared channel在进入store前拒绝。数据集单独需要的视觉/触觉不得绕行composition，应直接进入capture owner；
两边都需要时共享同一immutable sample，composition只接轻量view/ref。遇到资源紧张、复制/序列化争用或source
频率不足时，应优先用该plan收紧信息需求，而不是降低控制安全、扩大TTL或让control loop等待source。

不同来源频率的统一语义也明确为target-time selection而非package内线程/插值器：每个input声明receiver-clock
`[effective_from,effective_until)`；同一cut排除late/inactive-epoch/expired事实，再按role选择确定性最新有效值。
慢源只能在TTL内复用，快源只影响后续evaluation；高频reference resampling与physical dispatch仍由
`robot_motion`持有。input/context store现已使用显式`TimeWindowRetentionSpec`：snapshot先物化同一cut结果，
再推进单调target watermark并prune窗口外full payload；producer-epoch last-sequence tombstone继续阻止duplicate/
rollback。它不再使用“构造以来4096条”的hard capacity。窗口内异常高record rate/blob size及durable trace
backpressure仍是后续production hardening gate，不能把时间窗口等同于无限资源验收。

本轮进一步冻结八项修正与实施顺序：

1. **Context admission已经收敛，旧数据identity仍保持可读。** formal runtime、
   synthetic policy node及MuJoCo adapter现统一接纳
   `CompositionContextRecord/Store/Snapshot`：channel/kind、producer
   epoch/sequence、sample/available/effective clocks、contract-bound immutable
   payload或blob reference、source/parent lineage、availability/fault，以及可选
   policy/model/artifact/cut identity。旧`robot_motion.deploy.PolicyContext*`
   runtime在本仓及已知远端OpenPI source audit后删除；v2.4旧
   `policy_context_*`只作为离线读取兼容，新事实使用
   `composition_context_*`，不得另存hidden mutable “latest context”。
2. **高级strategy当前保持最简，但输入契约必须完整。** strategy可声明current
   inputs、bounded history、future actions、robot/task/lifecycle/source-health
   state、vision/tactile/force、labels、confidence及此前difference等typed
   same-cut context window。inline strategy必须pure/no-I/O并满足profile
   deadline；初始建议p99 `<= min(2 ms, 10% composition period)`。昂贵或
   model-produced mixer只能作为异步advisory context producer；control loop不等待，
   本地deterministic strategy仍负责weight/route合法性。
3. **store现为明确时间窗口，不是4096 lifetime counter。** 每个profile声明history、
   future lead、最大有效时长和可选late-arrival；snapshot原子物化后推进closed
   target watermark并prune窗口外full payload。已返回snapshot/evaluation持有不可变
   record，不受store删除影响；每个producer epoch的last-sequence tombstone保留。
   窗口内producer rate/blob quota及durable trace backpressure仍需独立补齐，禁止
   ring overwrite或把debug持久化作为控制正确性的隐式条件。
4. **arm joint DOF不再列为composition topology限制。** mixed arm action统一为
   pose；6DoF/7DoF/其它joint count属于downstream resolver capability。
   end-effector各自声明typed dimension、names、units和mask，component catalog可将
   多个末端绑定到同一arm。当前真实限制是v1 `RobotAction` carrier与现有target：
   left/right key、每side最多一个末端、legacy q6/q16 slot及coupled MuJoCo
   capability。后续以additive component-keyed carrier和显式v1 adapter解除，
   不修改现有layout/hash。
5. **formal compound pose+hand execution继续延后。** 本轮不授权end-effector
   driver、compound safety或real motion。
6. **difference拆成两个可独立关闭的阶段。** immediate same-cut
   source-to-composed difference是低资源同步返回；downstream
   post-processing/execution difference异步比较composed、safe、IK-resolved、
   smoothed/limited、dispatched、measured stage并绑定exact receipt。两者分别支持
   `off/summary/full`；无dispatch/measurement receipt不得称为actual deviation。
7. **recording拆成core与debug。** formal profile的实际行为事实和可重放lineage
   永远持久化，并作为内存prune前置条件；heavy alternatives、pairwise matrices、
   intermediate weights/timing、preview及model internals仅在debug模式构建和保存。
   两层使用独立queue/quota，debug failure不得阻塞control/core persistence。
8. **visualization/haptic/force feedback后续解决。** 它只通过独立adapter消费第6项
   两类difference，不得隐式改变authority或dispatch。

### Control Composition增量规划修正（2026-07-25）

本次代码调整后，以下内容从“待规划”改为“已落地、待稳定”：`CompositionActivationScope` v2已经覆盖
input→snapshot→result的同activation选择与v1兼容hash；`CompositionInformationPlan`已经能在source acquisition前
声明latest-value context channel、payload contract、access mode、maximum age及optional difference；runtime已只发
protocol-neutral `CompositionDomainEvent`，v2.4投影移至`dataset_interface.composition_control_trace`。当前最小
activation/information/context/boundary契约测试为42 passed。后续不得把上述边界重新合并回Station、dataset writer或
RobotMotion。

基于当前实现，后续优先级修正为：

1. **P0纯层已落地：第一类source eligibility/liveness。** `CompositionSourceEligibilitySpec`显式冻结
   role→source slot→producer绑定；独立State/Store/Snapshot已覆盖receiver admission、status epoch、source lease、
   fault/rearm及同一`ControlCut`。资格关闭时不进入input v1/v2；开启时input snapshot使用additive v3并由独立typed
   binding连接资格snapshot。它只决定input能否参与composition，不拥有authority、HOLD、cancel或dispatch。
2. **P0验证与首个sim shadow已落地。** 已覆盖动作TTL仍有效但source为paused/fault/disconnected/rearming或
   lease-expired时立即missing并由formal gate HOLD、旧status epoch/late heartbeat不能复活来源、同epoch heartbeat只能
   延长lease不能改变状态、`rearm_required`恢复需要新status epoch及receipt、producer restart、retention tombstone、
   activation+eligibility组合，以及unscoped/default-off hash/数值不变。MuJoCo仅在显式
   `enable_composition_source_eligibility=True`时复用现有唯一`policy_topology`；虚拟policy source pause的shadow结果为
   no-action，且无proposal、command或physics motion。VIVE/Q6/replay/真实policy adapter仍未接线，它们后续只负责把设备/
   worker状态翻译成该契约，不得在通用包硬编码设备字段。
3. **P1通用化：activation向proposal后的透明lineage。** 当前formal MuJoCo re-arm会同时推进composer producer epoch，
   所以旧proposal已被现有Arbiter epoch过滤；activation并未构成当前profile的新TOCTOU缺口。但
   `ActionProposal`目前只保存composition input/spec lineage，没有activation ID/hash，跨profile审计和未来“同producer
   epoch多activation”仍不完整。实施前单独冻结additive proposal hash策略：无composition保持semantic v1，未绑定
   activation的既有composition proposal保持v2，activation-bound proposal只能新增v3或独立typed relation，禁止修改
   v1/v2 payload/hash。最终执行继续复用现有authority/execution epoch、one-shot `MotionDispatchPermit`、
   cancellation receipt和execution-target admission，不增加第二套execution lease。
4. **P1 policy emission汇入门槛：异步结果freshness。** observation/request/result generation、
   observation/result age、late completion、timed-plan pop与emission commit仍由deployment/RTC owner检查；
   composition只消费经epoch-scoped source binding验证、已经固定anchor解码的正式
   `PolicyActionEmission`所适配出的absolute `RobotAction`。完整`RtcFrontendStep`只用于producer诊断兼容，
   不进入通用composition接口；model、chunk queue、RTC plan与resampling字段也不进入通用source-state。
5. **P1资源硬化。** 时间窗口语义保持不变；per-producer/global record、rate、inline payload和byte quota作为独立
   admission circuit breaker补充，超限显式拒绝并保留诊断，不使用lossy ring overwrite，也不以debug recorder成功作为
   控制正确性的隐式条件。

当前上述source-eligibility pure store/snapshot、外部ControlTrace投影/完整性校验和default-off sim
source-loss fixture均已通过；聚焦组合、dataset contract与MuJoCo回归为119 passed。下一顺序固定为：先保持
teleop/replay/policy deploy默认链路不变并审查P0 adapter映射契约；再冻结proposal v3/typed activation relation；
最后以普通policy contributor adapter接入已绑定的formal emission。该adapter不得读取`RtcFrontendStep`或管理RTC
epoch/plan。任何一步都不改变`sync_queue_v1`、现有teleop/replay、policy deploy或真机motion authority。

### 当前一句话状态

控制 contract、单策略/回放 shadow、authoritative sim、确定性快慢 composition，以及 shared-execution/safety-anchor
纯链路已经形成；真实执行 owner 仍是 legacy。**M5.7e2-a typed reference-segment contract 已在纯层落地**：
已确认
60 Hz position worker 的 reference 可以同时依赖相邻两个 15 Hz safe keyframe，必须以 segment 记录多 parent
lineage；未来 endpoint 的 admission 不能提前覆盖当前 safety anchor，每个高频 reference 也不能伪装成新的
safety-anchor commit。M5.7e2-b1 已将默认不绑定的 raw observer scoped 接到 legacy executor，e2-b2 pure bound
projection也已落地；e2-b3 post-resolver outcome和e2-b4 local-generation terminal也已在scoped observer中落地。
e2-b5又从现有planner/resolver/command-construction显式失败分支旁路产生non-authoritative failure observation；它不改变
原fault/hold/dispatch，也不把失败冒充shared rejection。上述事实均不生成 canonical `ReferenceSegmentAdmission`，也不推进 shared anchor；production binding仍未接，只有 shared
worker 真正执行 M1C epoch/generation校验后才允许生成 authoritative admission。M5.7e2-c0 已补 permit 后的 immutable authoritative dispatch context，使 receiver 能获得 exact
decision/safety/command/target-time/epoch/generation lineage，但它本身不是 admission。当前 MuJoCo profile仍为zero
target lead，无法诚实表达END前 admission与future pending commit。目标不是替换现有行为：zero-lead保留为
`immediate_qref_sim_v1`，另增`scheduled_reference_segment_sim_v1`验证future segment；scheduled profile冻结并通过
前不进入M5.8真机dispatch gate。当前e2-c1 pure contract与e2-c2 scoped deterministic sim receiver已落地：默认
immediate runtime不变，scheduled profile按30/60 Hz fixture绑定50 ms lead；receiver二次核验live owner/epoch/generation，
形成canonical admission与pending anchor timeline，并延迟到START才激活qref。它仍不代表真实多线程worker guard或硬件ack。

主线准备工作曾增加CS-6a最小human/replay混合执行证据：exact synthetic-human q6与已有immutable qpos replay emission
共享一个scheduled MuJoCo authoritative core，通过显式`HOLD/cancel → release/takeover → READY`完成一次单向authority
交接，并在同一episode WAL中seal/commit/rebuild两个ActualDecision。authority admission同时同步唯一dispatch-owner registry
的epoch；HOLD清除已取消scheduled reference并quick-stop planner。v1明确禁止同session重新激活曾经使用过的source，直到
typed proposal invalidation或新producer epoch落地，避免旧future proposal复活。该fixture没有真实遥操设备、dataset
cursor、Station Web入口、real backend、policy或RTC依赖；policy部署接入仍等待RTC独立支线完成，不阻塞当前replay+human
sim与数采主线。

2026-07-23进一步冻结representation边界：上述CS-6a/CS-6b direct-qpos mixed实现只保留为debug/contract fixture，用于验证
authority、epoch、cancellation、readiness与capture因果，不是主线机械臂动作语义。除显式debug外，human/replay/policy的
**arm component**在source、selected和canonical-safe层均必须是pose；target侧可以在verified frame transform和coupled IK之后
生成独立arm qpos execution target，但该qpos不得回流成arm mixed source contract。该限制不适用于夹爪/灵巧手：1DoF gripper、
q6手和q16手保持各自typed qpos/native continuous representation，并在独立component strategy中做active-mask-aware插值，之后
才与arm pose decision组成compound action。direct arm-qpos mixed仅延后到规划中的同构臂profile。

CS-6b1d第一段现已于2026-07-23 scoped落地：synthetic human与replay在source/proposal/selected/canonical-safe层保持
`bimanual_ee_pose_v1`，target内部才经既有coupled IK生成resolved qpos execution target；exact frame/component/action hash、
一次formal human promotion以及原HOLD/cancel/producer epoch/authority transaction均复用既有契约。与此同时新增backend/model/
device-neutral的pure arm-pose-mixing strategy seam。输入是同一arbitration cut下的admitted arm pose candidates与可选history、future-action、
modality、label、policy-context事实；这些context必须携带source fact/hash、receiver `available_at`、control admission sequence、
clock domain与冻结payload。strategy只产出pose decision/lineage，不得推进authority/epoch、读mutable backend、消费queue、执行
safety/IK或dispatch。首版`discrete_authority_owner_pose_v1`仅以shadow运行并与现有ActionArbiter比较selection/output-pose parity，
因此启停或strategy异常都不改变production路径。未来的SE(3) residual、component routing或learned gating必须作为版本化strategy
另过proposal admission、ControlTrace、latency/fallback、canonical safety与shadow parity gate，不得把任意pose平均或GUI比例blend
直接插入执行链。当前v1 slice只覆盖arms，并显式拒绝hand/end-effector payload，避免把末端执行器静默绑定到arm owner；
hand qpos interpolation与compound component assembly需单独版本化后再接入。

CS-6b1g现已补齐一个仍为pure/shadow的固定权重SE(3) pose融合实现：
`weighted_se3_pose_fusion_v1`只读取同一receiver cut内已接纳、同action space/reference frame/controlled frame的arm
pose candidates；平移采用归一化加权Euclidean mean，旋转采用确定性reference上的shortest-path SO(3) tangent mean。
strategy spec显式记录source-slot weights、平移/旋转最大分歧阈值、missing candidate的`no_action`或
`renormalize_available`语义以及reference选择规则；frame不一致、非法quaternion、越过分歧阈值均fail closed。它不拥有
source、模型、sim、真机、authority、safety、IK或dispatch，hand/gripper仍由独立qpos/native strategy处理。当前既有
数值核心现也由generic `CompositionSpec(algebra="weighted_se3_pose_fusion")`复用，旧component-union/base+residual
spec/hash不变；`ControlCompositionRuntime.evaluate_composition()`与MuJoCo shadow facade可以形成并记录composition
snapshot/result，但绝不发布proposal。方案1的scoped sim接线现已落地：显式传入`pose_fusion_shadow_spec`时，
passive human与正式replay adapter输出都以`authority_candidate=false`进入authoritative receiver的同一causal prefix，
fusion严格复用本轮authoritative proposal cut，不另取“最新值”。human source emission、adapter receipt、
composition input/snapshot/result均进入同一ControlTrace，重复passive emission按原identity幂等，后续正式human
promotion复用同一source fact而不重复接纳。该profile默认关闭；关闭时既有proposal ID/hash、cut、decision、replay pop和
dispatch不变。开启后新增admission会按设计改变cut sequence与decision trace hash，但正式replay proposal、selected action、
safety/reference/command和dispatch仍不变。TTL缺失、pose分歧越界或shadow异常只产生`no_action/invalid/error` shadow结果，
不得改变authority、HOLD/FAULT或canonical execution。该结论只覆盖deterministic MuJoCo shadow，不是正式融合authority，
也不接真机、policy或RTC。

2026-07-24进一步把共享pose数值层从固定双臂收敛为显式component manifest：当前v1只允许
`("left_arm",)`、`("right_arm",)`或既有`("left_arm","right_arm")`非空子集。candidate先投影到exact subset，
strategy snapshot/spec、fusion循环与输出`active_effectors`必须一致；空集、hand component、隐式补齐或strategy/snapshot
不匹配均拒绝。未声明时仍是原coupled双臂，旧strategy/composition spec hash与数值顺序保持。该改动只证明共享
model/device/sim-neutral seam可服务单末端；当前CS-6b1i MuJoCo capability仍为双臂，不得绕过target resolver/safety manifest
硬接单臂。任意命名、多于左右两臂的拓扑留给后续component-schema版本，机械臂混合表示仍是pose。

同批新增第一类通用差异/反馈链，但保持非权威且未接设备。`SourceDifferenceObservation`强绑定同一
`CompositionInputSnapshot + CompositionResult + CompositionSpec`，按固定`reference - compared`方向记录source pair和
source→composed residual；pose输出平移差及`R_compared^-1 R_reference`的SO(3) rotvec，qpos只比较双方finite槽位，
并保留available/effective时间差、TTL与exact lineage。frame/action-space/representation/shape不一致显式失败，不做隐式
变换。正式接线必须提供`SourceDifferenceSpec`，绑定composition spec hash并显式列出可比较role pair与output residual
role；不能因shape/action-space相同而把`base`和`residual`等不同语义输入自动相减。
`ControlCompositionRuntime.evaluate_composition()`现以default-off参数消费该spec，在同一次exact evaluation后返回
observation并经既有trace emitter写入`source_difference_spec`及`source_difference_observation`；CS-6b1h/1i只透传
spec并在step result暴露结果。
diagnostic异常隔离为`source_difference_error`，不得改写composition result/disposition、触发HOLD或阻止proposal。
默认`None`不增加clock read、trace事实或旧binding hash字段；重复decision key只保留一次观测事实。
`GuidanceFeedbackCommand`只把选定pair映射为有界、无量纲、带TTL的`guidance_haptic_non_force` cue，
明确`authoritative=false`；当前仅`NoIoGuidanceFeedbackAdapter`做本地接纳/过期验证。三类事实已可选写入v2.4
ControlTrace，但尚未接GUI、真实VIVE触觉、HOLD自动clear、真机或production profile；不得称为力反馈。
qpos差异可观测，但首版未定义qpos→触觉cue映射，调用时必须显式拒绝而不是输出零强度假反馈。

CS-6b1h现以独立、default-off profile完成Composer-authority deterministic MuJoCo第一段，不原地升级
CS-6b1f/1g。pure `CompositionAuthorityGateSpec`冻结每个contributor的missing行为、composer producer identity、
proposal TTL、canonical-safety必需性及parallel source proposal禁令；`CompositionAuthorityDisposition`只允许ready
result发布proposal，required contributor TTL/epoch失效、all-missing、pose conflict或composer/promotion异常都进入HOLD。
Human/replay只提交`authority_candidate=false`的composition input，runtime profile关闭primary-policy/replay authority，
唯一正式arms proposal producer是pose composer；该proposal随后仍经过原Arbiter、canonical safety、pose transform、
coupled IK和scheduled MuJoCo dispatch。evaluation与promotion已分离，formal gate检查的exact result不会在proposal admission
前重新capture一个更晚cut；同一decision key及同一result promotion幂等。HOLD后新输入不得自动重开gate或提前发布恢复
proposal。新增`composition_authority_disposition` ControlTrace事实闭合result→publish/HOLD因果。contributor input
另以`composition_contributor_input`记录，并强绑定human/replay source fact ID及其semantic hash；不含source
lineage的旧`composition_input`事件与hash保持不变。该slice尚无Station
manifest/GUI operation、real readiness、formal episode profile、policy/RTC contributor或真机授权；这些必须在独立后续gate
接入，旧teleop/replay/deploy入口和CS-6b1f/1g默认行为不变。

CS-6b1i现把上述data plane接入独立、通用的typed `composition_session` operation。其逻辑session spec、activation
epoch、状态机、trace observer和publication barrier由backend-independent `CompositionSessionController`持有；
MuJoCo wrapper只提供concrete readiness、lower HOLD/cancel、adapter、input admission与sim step。Station只把composer slot提升为
唯一active authority；human/replay保持preflighted contributor，其角色、composition/gate hash、execution binding、
capture boundary及`hold_and_cancel`/显式re-arm规则由不可变`CompositionSessionOperationSpec`冻结。start只有在target、
全部contributor和composer完成preflight/candidate后才能打开publication barrier；operator HOLD、ESTOP、session stop或
composition自身missing/TTL/conflict/error会先关闭barrier并使lower runtime HOLD/cancel，再把当前activation记为`held`。
方案B的sim-only typed re-arm现已落地：逻辑session identity保持不变，但旧held activation只能`supersede→ended`，随后创建
严格递增的新activation epoch，禁止`held→active`原地恢复。`CompositionSessionRearmSpec/Receipt`精确绑定旧state hash；
MuJoCo provider只证明自身实际检查过的complete cancellation、fresh measured qpos、composer producer epoch、authority
epoch、READY gate及activation-local input barrier，不声称deadman/retargeter或真机readiness。publication barrier只在
Station成功接纳该receipt后重开；旧composer proposal因producer epoch失效，旧contributor input因新activation输入屏障不可
复用，恢复后必须提交新输入；re-arm本身不预接纳动作，首个新proposal仍完整经过canonical safety。re-arm失败保持HOLD并
再次cancel，设备/source/rig实例不重启。ControlTrace已支持记录
session/re-arm spec、receipt与逐activation transition；既有非re-arm formal profile继续记录exact contributor lineage、
composition result/disposition、唯一composer proposal、Station barrier及pose→qpos dispatch，并已通过
`EpisodeTraceRecorderSink` seal/commit；含re-arm的formal sim episode已覆盖HOLD、typed receipt、新activation
epoch、再次正式emission/dispatch及seal/commit。该profile仍是
deterministic sim/default-off，无GUI、真机、policy/RTC contributor或production入口，且不复用`human_takeover`。

2026-07-25进一步把先前隐藏在MuJoCo wrapper中的“本次activation输入”升级为通用
`CompositionActivationScope` v2。scope精确冻结operation spec ID/hash、session epoch、receiver clock、
activation/input barrier、freshness policy与required roles；它独立于producer/authority/execution epoch和episode。
正式start/re-arm输入必须携带exact activation ID/hash，snapshot只选择同scope且满足receiver
`available_at/effective_from`屏障的记录，result继续显式携带同一identity；missing或mismatch形成no-action并由formal
gate HOLD。scoped input identity与sequence tombstone均以activation为namespace，因此producer epoch可保持不变且不会发生
跨activation ID碰撞；带旧activation的late completion不能进入新snapshot。2026-07-25后freshness判定已从MuJoCo wrapper迁入
`control_composition`的统一admission/snapshot evaluator：receiver policy只比较同一receiver clock上的
`available_at/effective_from`，strict-source policy还要求producer显式声明同一clock domain并满足
`produced_at > input_barrier_at`，跨clock无映射时fail closed；MuJoCo仅保留未建立scope调用方的隔离legacy
start-time shim。未绑定input/snapshot/result继续使用原v1
ID/hash domain与数值行为。v2.4 trace不改写既有v1 record schema，而以
`composition_activation_scope`和`composition_activation_binding`把input/snapshot/result逐一关联，completeness验证
scope→operation spec、binding→typed subject及input→snapshot→result同activation闭环。

CS-6b1f现把这一pose分支推进到一个明确的sim-only、default-off human-in-the-loop formal capture slice：
`build_scoped_cs6b1f_mujoco_pose_human_replay_station()`复用CS-6b1d data plane、typed
`request_human_takeover`、Station publication barriers、SideEffect coordinator和外部
`EpisodeTraceRecorderSink`，不创建新的queue、cursor、epoch或episode owner。replay初始保持authority；
`VirtualSimHumanTakeoverReadinessProvider`只在sim worker上等待post-boundary passive emission并注入虚拟
deadman/rebase evidence，READY后同一emission identity只promotion/admit一次。profile
`deployable_sim_scheduled_pose_human_replay_station_v1`要求Station/source/authority/gate/cancellation与scheduled
pose→qpos lineage，在`hold_before_unbind`后完成WAL/seal/commit和ActualDecisionView重建。该profile不是real human
readiness、Web/production entrypoint或RTC接线；新机械臂、遥操设备、sim或模型继续通过既有typed source/target/provider
seam复用，只有生命周期/协议真正不同才新增composition。

该后续hand seam的第一版pure contract现已落地但保持未接线：`LinearHandQposInterpolationStrategy`按同一component的
admitted proposals做active-mask-aware convex interpolation，支持DMTAC 1DoF、q6和q16；q16 inactive/NaN padding不参与计算，
缺少正权重candidate默认返回`no_action`。`assemble_compound_component_action`只做严格结构join并拒绝重复hand ownership或
native end-effector target；它不拥有authority、queue、canonical safety、IK、dispatch或episode。当前已增加
`evaluate_component_mixing_shadow`，在同一arbitration cut上执行arm pose projection、hand qpos strategy和strict compound
preview，并通过`project_component_mixing_shadow`写入显式、`non_authoritative`的ControlTrace事实；sim-only fixture复用唯一
episode owner验证HOLD/no-action、abort/restart、WAL failure和terminal closure。该实现仍只作为virtual-policy/component
shadow的可替换分支，未改变任何现有sim/real/replay/deploy行为；正式compound execution仍需hand spec/joint-order/units/
active-mask/bounds/safety与downstream admission receipt gate。

2026-07-25已在上述shadow之外增加第一段正式、backend-neutral component composition协议，但仍为default-off。
`CompositionSpec`只拥有输入/required roles、component representation、权重、missing/fallback和分歧限制；
`FormalCompositionStrategySpec`只拥有实现ID/version、所需context、实现依赖和inline latency，禁止在
`parameter_payload`重复定义component语义。新`formal_component_strategy_v2`使用独立
`dexteleop.composition_spec.v2` hash域并绑定typed `ComponentCompositionRuleSet`；原三种v1 algebra的payload/hash及
`DeterministicComposer`行为不变。参考`FormalComponentCompositionStrategyV2`已在单个same-cut
`CompositionResult`中完成单臂pose与Q6/q16 active-mask qpos融合；缺正式策略、contract不匹配、策略重复定义语义、缺输入、
frame/action-space冲突或数值异常均fail closed。v2.4 ControlTrace以additive
`component_composition_rule_set`和`formal_component_composition_binding`事实记录完整规则以及
CompositionSpec→rule-set→strategy三方绑定，不修改既有spec/strategy v1 record。当前验证只使用虚拟human/policy，
未接GUI、RTC worker、replay cursor、sim authority、真机或末端驱动器；下一门是MuJoCo shadow profile兼容验证及显式
promotion scope，而不是复用旧post-Arbiter shadow作为production路径。该门随后已以真实`MujocoDeployRuntime`
完成：composer authority关闭时只产生formal result/trace，不产生proposal或physics command；MuJoCo promotion API现允许
显式component/group scope，但按当前`ActionComponentCatalog`校验catalog顺序、active effectors及coupled group完整性，
默认仍为原双臂`("left_arm","right_arm")/("arms",)`。因此单臂+hand shadow不会被隐式提升；正式compound authority仍须
单独冻结含hand的catalog、canonical hand safety、resolver/reference capability和执行receipt。

该正式门现已按保守语义落地为sim-only、default-off的
`ConservativeCompoundAuthorityProfile`。profile不绑定模型/ckpt，而是绑定
CompositionSpec→component rule set→formal strategy三方hash、唯一composer producer以及完整catalog/group scope。
启用时raw primary policy与replay authority必须关闭，AuthorityPolicy对每个required group只能授权该composer且fallback
必须为domain HOLD；backend-neutral `AuthoritativeControlRuntime.admit_proposal()`再次校验完整coverage/group/action
effectors和formal composition lineage，绕过MuJoCo wrapper提交partial proposal同样被拒绝。proposal TTL收紧为一个
decision period；缺失/过期不沿用更久的旧composition。single proposal经Arbiter、整份canonical safety、
pose→qpos+hand passthrough和完整`CompoundDispatchManifest`后才进入receiver；safety异常、safe-action缩减scope、
resolution失败或任一downstream receipt失败均生成additive v2.4 `compound_authority_failure`并执行control-domain
cancellation/HOLD，清空全部pending/active group。成功时receiver仍只在START guard一次性提升arms和required hands。
当前真实MuJoCo fixture已验证唯一proposal三组选择、partial proposal拒绝、缺proposal HOLD和partial receipt全组取消；
它仍限双RM65+双DM-Tac q6 sim，不接GUI/RTC/replay cursor/真机，也不改变现有arms-only composer和sync路径。

CS-6b2a现补齐了正式接线前的passive source边界，但没有扩大执行面。
`PassiveTeleopComponentSessionSource`与pose-only facade共享唯一non-dispatching `TeleopSession` lifecycle和observer
fanout；profile只需以`PassiveTeleopComponentSpec`声明arm pose或hand/gripper qpos component，不因VIVE、手套、串口、
sim、模型或机械臂品牌新增Station source类。arm component固定校验typed `ArmActionSpec`、reference frame与SE(3)；
hand component固定校验typed `HandActionSpec`、canonical full-slot vector、active mask及padding。hand-only emission的
whole-action space由绑定的`ActionComponentCatalog`决定，不能再根据单帧是否含arm猜测为arm-qpos profile；representation
冲突直接拒绝。旧pose-only source仍精确拒绝hand/qpos，既有入口与authority不变。该实现只证明source emission与adapter
binding，不是formal component composition/admission，也未补hand safety、anchor、reference或MuJoCo compound scheduling。
当前`RobotAction` v1仍只有left/right arm/hand和q6/q16 slot；结构复用不等于已经支持任意本体。第三effector、
非left/right component或动态末端payload必须通过EG2/EG3的generic component-action bridge与schema-visible
core落地，禁止继续在v1 source中按shape扩展。7DoF arm本身不改变pose composition contract，只要求新的
resolver/execution capability。

CS-6b2b现进一步落地正式执行前的通用边界，但仍不接末端驱动器或MuJoCo hand actuator。`ObservationSampleTracker`
为任意`ObservationChannel`分配receiver-local稳定sample ID、sequence、measured/available clock和payload hash；同一
latest sample跨多个control step复用identity，值、时间、validity或上游sequence变化才产生新sample。该sample set随
`HumanCommandEmission`进入v2.4 `human_command_emission`事实，避免只保存retarget后command而丢失本次决策实际看到的
VIVE/Q6/AngleTrigger输入。正式human接纳已抽成target-neutral `admit_human_command_emission()`，现有MuJoCo handoff复用
同一接缝；它只调用现有ProposalBuffer，重复emission由原proposal identity拒绝，不增加queue、去重器或authority owner。
canonical keyframe safety新增显式`HandQposSafetyLimits`：按绑定`HandActionSpec`只处理active slots，保留NaN/zero padding，
复用位控遥操的finite target、absolute clamp与可选step clamp语义；未配置hand profile时仍保持旧
`keyframe_safety_v1_hand_unsupported`。安全anchor现在可原子保存arm pose/qpos与hand qpos，HOLD/FAULT/epoch失效仍复用
原整组rebase-required生命周期。MuJoCo执行侧又补齐两个不改变默认authority的前置：canonical-desk transform和coupled
pose→qpos resolver可在派生contract下原值携带already-safe hand qpos；scheduled rig只在arm reference通过
epoch/generation guard正式激活时，才将同一pending reference携带的DMTAC目标送入gripper transport，HOLD或被替换的future
reference不会提前影响gripper。即时旧路径仍在publication时入队，纯arm resolver仍保留原contract hash。

CS-6b2c现按opt-in profile冻结multi-group执行语义：保留既有`arms`，新增`left_hand`/`right_hand`；每组可选择不同
proposal，但下发前必须生成immutable `CompoundDispatchManifest`，精确绑定dispatch context hash、safe-action hash、
三组selected proposal、generation、component coverage和唯一`target_effective_at`。arm仍由唯一qref
`ReferenceSegment`表达，position-like hand是同一时刻生效的held target，不伪造三条插值segment。receiver先验证完整
context/manifest/resolution，再接纳arm target；START时core先校验全部pending group/generation，随后原子提升三组，rig才将
DMTAC目标送入transport。START前HOLD会同时清除三组pending、receiver reservation和未入队hand target。MuJoCo profile以
空`compound_hand_action_specs_by_side`保持default-off；启用时要求显式canonical shared safety、scheduled pose execution和
已有DMTAC q6 profile。独立human/replay-style多proposal compound fixture仍保持原语义；composer authority仅可通过上述
额外typed conservative profile开启，replay authority和非DMTAC hand spec仍拒绝。formal sim fixture已覆盖独立VIVE-arm、
left-Q6、right-Q6 emission→proposal→Arbiter→safety→manifest→guarded activation与HOLD-before-START。

CS-6b2d现补齐sim范围内的causal input/native stream第一段。通用`ReceiverObservationCutBuilder`将任意lower source sample
保留原sample/payload/measured identity，并增加receiver-local available time、clock和admission evidence；同一未变化样本跨
cut复用identity。多个arm/hand emission可共享一个recorded `ObservationCutRecord`，各proposal只引用该cut且仍保留独立
emission/proposal/group身份。v2.4新增纯`NativeStreamContract/SampleRecord/EpisodeManifest`、append-only episode writer与
validated reader；数据按stable sample ID去重，payload content-addressed，index/contract/manifest带hash，seal后不可追加，
显式保存valid/stale/drop/gap与measured/available/written clocks。opt-in MuJoCo fixture现同时保存双VIVE pose、双Q6 input、
sim arm/hand measurement和真正进入DMTAC executor后的command；START前HOLD的command stream保持空。该writer仍是
subordinate storage primitive，尚未接现有production `CollectionWriterRuntime`/process worker、RGBD/tactile native
producer或真实deadman/rebase receipt，因此仍不能称完整production数采闭环或
真机HIL。Inspire/其他灵巧手sim/real driver继续明确后置。

CS-6b2e已把上述primitive以default-off方式接到唯一`EpisodeTraceRecorderSink`生命周期，而未增加episode owner。
factory只能使用prepared episode的同一UUID和legacy index；release在同一锁内摘取writer，先seal native manifest，再评估
ControlTrace seal。`native_streams` capability必须与writer同时启用，v2.4 capture commit显式绑定manifest hash，manifest
store和dataset reader都会复核同UUID、sealed状态及hash。native seal失败时legacy finalize仍完成，但v2.4 commit保持
`not_eligible`，不得发布半完整episode。正式MuJoCo human-only fixture已验证同UUID/seal/commit/readback；默认commit payload
与hash保持不变。该slice仍未把高频append接入现有bounded process writer，因此不能从control loop直接做同步文件I/O；
process protocol、backpressure/drop receipt及RGBD/tactile producer接线是下一独立gate。

CS-6b2f已完成该process gate的低带宽部分：现有`ProcessCollectionWorker`可选接收stream contracts，并为同一episode创建
parent-side subordinate capture proxy。pose/qpos sample只做有界、non-blocking queue submit；queue busy或inline消息超过显式
上限时返回typed rejection，proxy在seal前把drop count/reason有序提交给child manifest。child独占native writer，严格要求
episode UUID/index一致，并禁止native未seal时legacy finalize。该功能无contracts时完全关闭，现有snapshot/legacy writer路径
不变。它只批准小型pose/qpos等inline payload；RGBD/tactile等大数组不得经pickle queue，仍需复用现有shared-memory snapshot
transport并建立sample identity映射。顶层GUI/真机profile尚未启用该capability。

CS-6b2g已补齐external payload的纯契约和本地两阶段seal机制，仍保持default-off。capture阶段只登记
`NativeStreamExternalSampleIntent`，冻结后由既有specialized writer提供dataset-relative locator、logical payload hash、
artifact hash/size、encoding/compression和durability receipt；native writer逐项复核路径、文件和sample identity后，才写
sample index并seal manifest。`EpisodeTraceRecorderSink`仅对显式external profile采用
`freeze → legacy finalize → receipt bind → native seal`；原inline profile仍在release时直接seal，原payload/hash不变。
validated reader即使不加载payload也会复核external artifact，加载时返回typed locator而不伪装成已解码数组。fake
video/tactile已覆盖完整receipt、缺失receipt、越界locator与artifact篡改；任一失败只禁止v2.4 commit，不改变legacy
finalize结果。该slice尚未让现有video/tactile writer实际生成receipt，也未把shared-memory slot映射成stable intent，
因此不表示production RGBD/tactile已接通。

CS-6b2h已将上述机制接到现有process collection的首个真实specialized artifact，但仍是显式opt-in。当前只支持
`selected_rgb`：child从既有shared-memory snapshot登记stable source-frame intent，同一源帧跨多个sample复用时只登记一次；
legacy `LeRobotV21Writer`继续按原顺序写MP4，只有新artifact-report API会在close后fsync并返回文件hash/size及
`sample_frame_index → video_frame_index`映射。episode release先flush/freeze，legacy finalize成功后把逐sample receipt交给
原`EpisodeTraceRecorderSink`完成native seal；receipt投影或seal失败不回滚已完成的legacy episode，只令v2.4 commit
不合格。无contract/binding时finalize、MP4数值/顺序和旧sink调用签名不变，GUI/真机profile未打开。该流严格表示
decision/sample-selected RGB，不得宣称camera native-rate；原生RGBD与高频tactile仍需各自producer-level capture和artifact
bundle契约。

下一轮arm pose fusion规划必须单独核对以下缺口后再选实现批次：同一receiver cut上的时钟/新鲜度与history/future
窗口；source pose到统一reference frame及immutable anchor；平移与SO(3) log/exp的版本化融合规则、奇异/大分歧处理；
authority/component ownership与strategy context lineage；融合后统一canonical safety、workspace/速度/加速度限制及coupled
IK顺序；输入缺失、stale、策略异常或source冲突时退回discrete owner还是HOLD的显式gate；以及deterministic replay、
shadow parity、延迟预算和一次正式admission。hand/gripper仍走独立typed qpos interpolation，不进入arm SE(3)融合。

2026-07-22对pause/episode/human-connected corner case复核后进一步冻结三项边界。第一，source connected只表示输入资源
可读，不是candidate、authority或接管许可；policy/replay停止后默认留在HOLD，禁止自动human takeover。第二，现有teleop
`episode_stop`保持capture-only兼容语义，新的mixed formal profile另用`hold_before_unbind`，recorder/episode sink本身永不
拥有motion lifecycle。第三，human takeover必须是两阶段事务：旧source publication关闭、HOLD/cancel和producer
invalidation/drain完成后，可在HOLD下转移authority；只有post-takeover fresh human emission、fresh measured state、deadman
重新武装、retargeter rebase和canonical safety evidence齐备后才允许READY。CS-6a的immediate-READY只保留为synthetic
fixture证据，不能升级为真实human或policy→human语义；第一批实现限定为root `control_station`纯契约、虚拟source与MuJoCo
scoped验证，不改teleop/replay legacy执行和episode行为，也不接RTC producer或真机。

该第一批CS-6b0实现现已按上述边界完成scoped落地。纯receipt reducer现唯一位于
`control_composition.lifecycle.handoff`，不导入model、sim、hardware或teleop；MuJoCo bridge只增加一个显式单向
`replay→synthetic human`入口，原CS-6a immediate-READY入口与既有
human/replay submission保持不变。bridge在publication关闭后消费真实HOLD/cancellation ack并推进旧replay producer epoch，
所以历史proposal只保留为trace fact，不再进入active-epoch arbitration cut；随后在HOLD下完成release/takeover。post-takeover
human adapter/canonical-safety仅作为non-publishing preview，真实MuJoCo qpos作为fresh measurement；READY后必须由同一emission
identity/content/timing经原submission seam完成且仅完成一次formal emission/admission。当前deadman/rebase证据仍来自虚拟fixture；
其后的CS-6b1已增加Station typed operation；CS-6b1f现将device-neutral passive pose接入一个显式、虚拟readiness的
sim-only formal episode profile，但实际遥操deadman/rebase receipt、真机target和real/RTC接线仍不存在，不能从本轮测试外推为
production human takeover。

CS-6b1接线前曾有一个必须显式冻结的control-plane选择：现有`release_motion_authority(old)`完成后会把Station正确latch在
HOLD，而generic reducer在HOLD下禁止`request_motion_authority(human)`；若先请求human，old source的Station状态又仍是active。
因此不能靠两条现有single-source intent在GUI或coordinator层拼接事务，也不能简单放宽HOLD allowlist。冻结方案是新增一个
typed、multi-source、multi-phase的`request_human_takeover` operation，将old/target slot、CS-6b0 request hash、lower HOLD/
invalidation/authority evidence与最终readiness receipt绑定为一次pending operation；该schema现已落地，Web allowlist仍未开放。

该public pure contract现已冻结为CS-6b1a：`StationIntent.subject_id`绑定目标human slot，`resource_binding_id`绑定当前
non-human active slot，`operation_spec_id`绑定不可变CS-6b0 request/spec hash；`PendingStationOperation`同时冻结两个source的
先验状态。请求阶段不提前修改任一source状态，只有带aggregate control evidence且terminal status为
`human_takeover_ready`的唯一completion，才能原子提交old→preflighted、human→active并保持RUNNING。失败只能落到HOLD/FAULT；
takeover pending期间其他source/target operation被拒绝，HOLD/ESTOP会终止pending transaction，迟到success不得恢复RUNNING。
这些规则只依赖manifest source capability、target contract和typed identity，不包含MuJoCo、Vive、RM65、dataset或模型分支；
pure handoff/coordinator/barrier回归已通过。CS-6b1b scoped sim executor也已把既有CS-6b0 transaction接入现有
SideEffect coordinator与两个publication barrier：pre-request `HumanTakeoverOperationSpec`只绑定source/action/target/observer
compatibility和timeout，不携带raw action或live epoch；post-result `HumanTakeoverOperationEvidence`聚合真实request/state/receipt。
readiness provider等待在worker上，最终safety/measurement/READY通过Station pending-finalization CAS提交；若HOLD/ESTOP先赢，
finalizer不会执行lower READY。无scheduled work但backend已HOLD时使用显式`force_cancellation=True`推进execution epoch并产生真实
cancellation evidence，不伪造ack。scoped synthetic/replay fixture验证exact一次formal promotion、timeout/stale保持HOLD、late
completion不复活lower READY。加入passive human ingress后，Control Station、MuJoCo deploy与底层
authority/epoch/cancellation/owner相关回归共221 passed。
随后CS-6b1c又在root增加bounded latest-preserving human emission ingress与exception-isolated observer fanout；它只接收既有
`HumanCommandEmission`并保持exact object identity，不修改`TeleopSession`、retargeter或dispatch。该ingress刻意不实现readiness
provider：当前真实遥操没有typed post-takeover deadman-rearm和retargeter-rebase receipt，Q6手势`rearmed`不能替代前者。因此
真实human READY接线在这两个lower receipt落地前保持fail-closed；replay one-emission fanout与canonical GUI pose emission已有
通用seam，下一步应将CS-6 handoff事务接到现有arm-pose safety/transform/IK target pipeline，而不是为arm qpos mixed增加
兼容层；hand/gripper qpos component另走其typed interpolation/safety contract。Web allowlist、实际设备、component-aware mixed
formal episode与RTC仍未开放。

该接线前架构整理现已于2026-07-22完成，并把扩展边界冻结为可复用的窄seam，而不是按设备、模型或backend复制
composition。`control_station/targets/authoritative.py`只依赖共享`AuthoritativeControlRuntime`，为后续MuJoCo、其他sim与
real target复用同一Station lifecycle/evidence；`control_station/sources/{replay,policy,rtc_policy}.py`只保存lower-owned
source binding，不导入具体sim或hardware。exact MuJoCo source/target匹配仍留在现有`composition/mujoco_*.py`，因此新增
checkpoint、模型server或遥操设备在typed emission seam不变时不新增composition。manifest现统一以`AuthorityStore`作为
canonical authority epoch source，`DispatchOwnerRegistry`只作为permit enforcement mirror。RobotMotion新增typed execution-
gate transition outcome、公共gate snapshot和producer activation返回值；sim新增不携带policy/human/replay identity的raw qpos
measurement seam。CS-6b0已改为消费这些typed结果，不再切片全局trace、读取ack store或直读`rig.scene`。旧import路径、
`set_execution_gate()`返回值、teleop/replay数据面和入口均保持兼容。Control Station/RTC shadow聚焦回归133 passed，底层
authority/cancellation/MuJoCo/RTC回归78 passed；周期性边界检查记录在`control_station/README.md`。CS-6b1f sim HIL已在该窄
seam上完成；仍不得重新引入backend-specific source contract、每设备/模型一文件或第二套queue/cursor/episode owner。

2026-07-21 对并行工作树的集成复核还确认：默认 pose-frame、camera transport/placement 与 collection timing
profile 已变化，必须以 exact manifest/hash 重新做 artifact preflight。该轮发现的无硬件回归已收口：RM65
absolute limit 的静态运行时镜像与三个 RL task profile 已改为 active URDF 的 J4 ±2.44346 rad，并由解析 URDF
的漂移测试锁定；历史 right-J4≈3.10 fallback 已从共享 preview/recovery seed 移除，真实 startup/recovery仍必须
使用显式 station alignment 或 fresh measured state。M0 typed-receipt marker与foot-pad同poll安全顺序fixture也已更新。
其后并行sim actuation工作将默认RM65 profile推进到v2，曾造成task config、test expectation和real2sim artifact
SHA漂移；该漂移已由2026-07-21全量2605 passed、6 skipped关闭。CS-0/RTC/Arbiter/owner/trace聚焦回归保持全绿，
M5.5/Control Station CS-2的baseline gate已恢复。已commit v2.4 的episode compact和tactile in-place recompress现已在
任何staging/删除前fail closed，原capture保持byte-immutable；copy-on-write derived maintenance仍是后续能力，
不能通过重写原commit替代。详细evidence、责任边界和解除条件见landing plan的“并行工作树集成复核”。

状态词统一按 landing plan 使用：`contract complete`、`compatibility seam active`、
`production authoritative`、`durable capture` 和 `local only` 互不等价；文中“完成”均应结合其限定范围理解。

异步 action chunk 并行边界已完成RTC-4b-T2c live worker/timed-plan synthetic no-action shadow：RTC-1 immutable inference/plan、RTC-2
deterministic authoritative MuJoCo runner、RTC-3a bounded OS-thread carrier、RTC-3b split sync-parity seam及RTC-3b.5
single-admission snapshot均完成；显式、默认关闭的RTC-3c profile现已在现有deploy application内组合同一个timed-plan
owner、worker与正式`PolicyActionEmission` seam；commit-through只在当前popped keyframe形成正式emission后推进。六组RTC
直接fixture为60 passed，RTC-4a以42个pure/local test冻结scheduler/continuation分轴、capability/envelope/receipt、
continuous/discrete输出与两个local backend迁移；RTC-4b-A再以28个fake transport/policy test完成现有OpenPI artifact、
server metadata、exact observation payload、continuous/int64 discrete output和V22 codec的final-chunk离线往返。RTC-1～4b-A
聚焦回归130 passed；2026-07-22随当前主线复跑兼容矩阵为154 passed、1 skipped，早期evidence中的152 passed、
1 skipped保留为历史计数；
默认`run_policy.py`仍同步调用`predict_and_queue()`，RTC profile仅允许`send_actions=false`且RTC-only
lookahead disabled，`sync_queue_v1` lookahead不变；RTC-3c仍只允许builtin hold。通用RTC协议仍不导入具体
模型/transport/sim/real backend；OpenPI RTC-4b-A只是首个可替换的 raw-RTC final-chunk adapter并且只声明
`continuation=none`。T2c的caller-free live composition已保留为证据；随后新增的T3 caller只在 OpenPI 显式
raw profile 下复用同一 transport/client 接入 shared worker/coordinator，强制`send_actions=false`，不改变默认同步路径。
RTC-4b-T1/T2a又完成session cancel/close/re-arm、通用connect/metadata/inference/close预算、typed phase receipt、
fake socket压力矩阵与synthetic final-chunk localhost websocket；该范围无模型/GPU/真实observation/action send，deadline
不绑定checkpoint/config/action dimension/horizon，也不替代request TTL或worker drain。T2b随后直接复用一个现有
OpenPI observation→final-chunk loaded model server完成两次synthetic live no-action loopback：两代connection的
artifact/metadata/layout/horizon一致、metadata hash相同，返回50×16 finite chunk；它无需模型适配且仍是
`continuation=none`。该模型/checkpoint只是代表性证据，不进入通用启用条件；T2b也尚未接bounded worker/timed-plan、真实
observation或action send。T2c现已在caller-free synthetic shadow中把现有adapter→worker调用推进到普通OpenPI server：
组合探针发现immutable prepared payload的嵌套`mappingproxy`不能直接msgpack，现只在transport edge递归`deep_thaw`为
detached plain containers，内部hash事实、wire字段、模型和sync路径均不变，本地/远端adapter回归各`32 passed`。修复后
live请求已到`Policy.infer()`，但同机既有训练占用约44/49 GiB显存，首次JIT因额外约2.0 GiB申请失败；RTC按typed fault、
TTL和FAULT fail closed，30 Hz fault批次901 ticks的control-side wall最大约7.4 ms。GPU释放后又以同一artifact-derived配置
完成cold-late→warm-accepted：cold约9.85 s只因TTL拒绝，warm约0.687 s将44个future keyframes唯一接纳并丢弃6个obsolete；
320 ticks无预算超限，零emission/零legacy-pop，worker/transport完整drain且close后plan归零。
当前完整RTC聚焦回归`172 passed`，OpenPI/default deploy/position/MuJoCo兼容矩阵`160 passed, 1 skipped`；本批没有编辑
teleop/replay/position路径，RTC profile仍禁用lookahead且默认sync路径不变。
T2d随后在`deployments.client`增加纯结构化framework composition helper：任意本地runtime、HTTP、websocket或IPC adapter
只提供instance-derived capability、request spec和worker executor factory，helper在worker创建前协商并精确投影七组
identity/horizon、核对cadence，再返回未启动frontend。现有OpenPI无需shim；generic/OpenPI聚焦`42 passed`，完整RTC聚焦
更新为`176 passed`。无自定义session生命周期的backend可直接用无继承`SimpleRtcModelRuntimeAdapter`传入四件套；OpenPI
这类session-aware框架仍实现同一protocol。该helper没有模型名称registry、observation/target/dispatcher或motion authority；
OpenPI raw-RTC caller由显式 launcher factory 持有，model-RTC仍不在本批。
2026-07-23 已将该 caller 接入 `deployments.openpi.client.real` 的显式
`raw_rtc_final_chunk_15_30_60_120_v1` profile：复用同一 OpenPI transport/client，固定
`send_actions=false` 与 lookahead disabled，缺少 artifact、normalization/frame identity 或 capability 协商时 fail closed；
raw-RTC 四文件回归 `82 passed`，仓内 RTC/OpenPI 全集 `213 passed, 1 skipped`；R-R4 已补充真实 observation/action
no-action shadow，仍无 action-send authority。
随后授权的real observation no-action shadow已让双臂、两侧DM-Tac gripper和三路RealSense通过现有deploy入口形成
1200个30 Hz step，并向relative `149999/h50` OpenPI server完成30个non-authoritative final-chunk exchange；没有
position/lookahead/dispatch/action-send事件。该结果没有开放下一gate：现有10-step validity将cold与warm result全部按
TTL拒绝，completion-side control最大约`104 ms`，close为`ABANDONED_UNDRAINED`。随后增加的显式
`rtc_request_validity_steps`/`rtc_worker_stop_timeout_sec`保持10/0默认值不变且非RTC caller不能设置；50-step/2 s复跑
接纳7个warm result、形成240个唯一formal emission并完整drain，仍为零dispatch；该R-R3b历史结果的8/8 completion
step超33.3 ms、最大约`101.8 ms`，后来由model-neutral observation-cut/admission路径收敛所替代，不再是当前结论。
随后R-R4以显式`request_validity_steps=50`、`worker_stop_timeout_sec=2`完成900个30 Hz真机输入-only复跑：8次
request/response中7次merge接纳、1次cold TTL拒绝，253个唯一formal emission；control tick最大`16.80 ms`、零超预算，
close为`DRAINED/thread_exited=true`，仍零dispatch/action-step。该证据闭合raw-RTC真实输入/transport/admission/
worker-drain shadow，但不提升canonical safety/IK、target acceptance或motion authority；见
`todos/evidence/rtc/rtc_rr4_openpi_raw_rtc_real_observation_no_action_drain_gate_evidence_v1.json`。
T2e再以只读`vla-scratch`基础VLA验证第二框架：基线revision `fa3cb37a6f3ba620bd2c167a97035b5431f31d19`
保持clean，DexTeleop外置pure processed-sample binding与只依赖既有`infer(dict)`的final-chunk backend。原生ZMQ no-model
loopback已让2-camera/state输入和4x7 chunk通过generic RTC client，8项local fixture通过；未加载checkpoint/GPU或动作链。
第二框架暴露的capability/request/response样板已回收到主链`FinalChunkRtcBackend`，VLA层只保留sample校验、调用和row提取；
3项generic fixture覆盖continuous/discrete、input-hash-before-call和typed cancel。
该server尚无metadata handshake且依赖dataset推导dimension/warmup，因此只能证明接入摩擦低，真实模型仍需先形成
artifact-derived camera/state/action/frame/normalization binding，不能把LIBERO默认7D输出猜成DexTeleop action。加入T2e后
完整RTC聚焦矩阵为`187 passed`，既有兼容矩阵仍为`160 passed, 1 skipped`。
T2f现已补齐其中不依赖checkpoint的dataset/sample部分：纯`dataset_interface.vla_scratch`从v2.3 artifact派生ordered
camera、layout/chunk/frame identity与processed sample spec；完整fixture以`front/left_wrist/right_wrist`、左右pose7加双侧
连续q6末端组成26D state/action和4-step chunk，并由未修改基线`ToDataSample`接受。上述数值均来自fixture artifact，不是
adapter常量；discrete channel、未契约的state transform与padded tail因当前baseline无法无损表达而fail closed。两仓环境
依赖不重合部分保留在后续additive dataset plugin/进程边界处理，不修改基线依赖。新增5项后同口径RTC聚焦矩阵为
`192 passed`；仍未读取真实episode、加载VLA checkpoint/GPU或接入action链，下一VLA gate需兼容训练artifact、normalization
digest和decoder后才能进行loaded-model synthetic no-action。
随后T2g在未修改的vla-scratch虚拟环境中完成独立进程virtual-policy wire probe：server进程由原生
`ServePolicy`承载一个确定性的virtual `BasePolicy`模型实例，client进程使用原生`ZmqPolicyClient`；三相机/1×26
state进入server并返回4×26 finite chunk。这里没有两个模型 policy：`ServePolicy`是包装层，`ZmqPolicyClient`是传输层。
该结果确认两仓依赖不重合不阻塞server-client；combined environment只属于可选同进程训练plugin，不属于RTC gate。
caller-free RTC-4b-B1 remote source seam也已完成：独立Pi0/Pi0.5 hard-prefix sampler与
`Policy.infer_rtc()`默认关闭，原sampler/infer/server未改，相关pure/fake测试16 passed、2 deselected，V22回归10 passed、
3 skipped。带discrete token输出的model-RTC continuation在typed discrete prefix/receipt之前fail closed。其后可选门槛
RTC-4b-B2才是capability-driven模型侧prefix→suffix continuation接线与可验证conditioning receipt：能力必须由loaded model instance、artifact、
transform、norm stats和实际sampling参数生成，不得绑定某个checkpoint/config/horizon/dimension；RTC default-off，legacy
`Policy.infer()`与sampler保持独立原路径。真实observation/transport shadow仍属RTC-6。Control Station只
显示和编排该producer，不得自行predict、merge或pop，也不得把local virtual dry-run显示成real shadow或motion ready。

## 结论

当前方向不需要推翻。以下基础应继续作为唯一事实来源：

- `robot_motion.action.RobotAction` 的统一动作语义；
- `robot_motion.runtime` 的 Rig、dispatch receipt、生命周期和因果历史；
- `robot_motion.deploy` 的 policy artifact、v2.3 表示适配、chunk transaction 和 canonical deploy seam；
- `robot_motion.position_control`/`robot_motion.rm65` 的 reference、IK、trajectory、command 和 safety 语义；
- v2.3 的 absolute pose storage、relative model view 和严格 causal snapshot 契约；
- GUI/headless 已共享的 `TeleopSession.dispatch_command()` 与 writer 生命周期；
- sim/real 共用纯 contract、分别持有 backend adapter 的依赖方向。

需要修正的是此前规划中仍然存在的三个隐含假设：

1. 系统只有一个策略实例和一个 action chunk queue；
2. RGBD、policy inference、action proposal 和 Arbiter 共用一个 15 Hz 时钟；
3. 单进程运行方式可以作为逻辑 contract 的一部分。

还必须准确限定当前的本体通用性：**DexTeleop 已经具备本体无关的控制协议骨架，且arm composition以pose
隔离机械臂joint count；但当前canonical action carrier/execution v1仍是RM65左右双臂形态，尚未具备任意命名
component和任意末端schema。** 首个真机候选继续固定为 `m6_candidate_policy_arm_qpos_v1`，不在该路径上重构
`RobotAction`、26D/38D policy layout、RM65 safety 数值或公开兼容列。真机前只建立禁止新增泄漏的泛化
防火墙；component-keyed payload和manifest-driven builder的重构由独立EG批次或第二种明确本体驱动。

另有一个当前必须作为发布策略处理的工程事实：现阶段稳定的双臂 coupled IK 验收上限约为 60 Hz；当前
**DexTeleop production decision policy v1 将 `ActionArbiter` 的 authoritative 动作决策发布率限制为不超过
60 Hz**。二者需要同时写入 execution manifest，但不能混成同一个时钟或同一个代码级限制。Arbiter 上限
只属于当前 production decision profile；sensor、policy/proposal、reference、IK、command、backend/physics
和 recording clock 仍然独立，不复用这个校验。具体 control domain 是否能在某个 reference/IK/command
profile 下稳定执行，由单独版本化的 `ExecutionCapabilityProfile` 和时延证据决定。未来 hand-only、
joint-space 或经验证可接受更高 keyframe rate 的 embodiment 若需提高 Arbiter 上限，必须发布新的
decision-rate policy、execution capability、验收 fixture 和 release note，不能通过临时 CLI flag 绕过 v1。
高于 60 Hz 的 producer 结果在下一次合法 decision cut 被归并消费，更高频下游可以继续持有、重采样或
展开当前 selected action。

目标架构应调整为：

> 在统一的因果观测时间线之上，允许任意数量、不同频率的 human、policy、replay 和 scripted
> producer 产生版本化 context 或 absolute `RobotAction` proposal；所有 proposal 经过唯一的
> authority/ActionArbiter 和 canonical execution pipeline，最终进入每个 control domain 唯一的逻辑执行出口，
> 并以 component-level lineage 写入统一 ControlTrace。v2.4 数据集将这条时间线作为训练事实层，
> 同时保留 v2.3 的 absolute/relative pose 语义，并支持多来源 action、多频率 modality、reward、
> DAgger、快慢策略、真机 RL 和 sim2real 的可重放训练视图。

最终交付目标不是单向“把 action 发给机器人”，而是可重复迭代的闭环：多种遥操设备、policy、replay
产生 proposal 并控制目标 embodiment；同一运行时原位采集 observation、全部候选/选择/执行和 outcome；
sealed capture 生成可审计 TrainingView；learner 输出带完整 provenance/compatibility contract 的 policy
artifact；artifact 再经 deployment preflight、shadow 和 authority gate 回到同一控制链继续采集。更换遥操
设备或机器人 backend 时不重写 Arbiter、ControlTrace、TrainingView 和 learner 接口，只替换显式
source/embodiment adapter 并重新做 compatibility/safety 验收。

这里“允许任意数量、任意频率、任意部署位置”是 contract 能力，不代表近期要实现一个通用分布式
计算框架。最佳落地路径仍是：先冻结拓扑无关 contract，用单策略证明行为兼容，再用 synthetic
快慢策略验证多速率，之后接 deployable sim 和真机 shadow。

---

## 一、当前代码基线与准确边界

### 1. 当前正式入口

- GUI human teleop 与正式数采：
  `python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui`
- headless/兼容数采：
  `python -m teleop_runtime.collection`
- 真机 policy deployment：
  `python -m teleop_runtime.deploy`

`teleop_runtime.collection` 继续作为 headless/兼容 frontend，不发展第二套 GUI/control semantics；
`teleop_runtime/core/kernel.py` 仍为 experimental；`robot_motion.deploy.DeployRuntime` 和
`deployments.client.DeploymentClientRuntime` 仍是窄 scaffold，不是 production loop。

### 2. 已经可以直接复用或已完成 contract 落地的实现

截至 2026-07-19，需要区分“已有 production owner”“已接 production 兼容 seam”和“只有 additive
contract/test”三种状态：

| 当前模块 | 已有事实能力 | 当前接入层级 |
|---|---|---|
| `tests/fixtures/control_core/*`、`tests/test_control_core_baseline.py` | v2.3 单策略 relative decode→safe action→command、dry-run dispatch、position keyframe、入口调度与 `last_action` 基线 | M0 无硬件 golden baseline；不是硬件 acceptance |
| `dataset_contracts.canonical_encoding`、`robot_motion.runtime.{identity,control_admission,producer_epoch,observation_cut}` | canonical hash、runtime fact ID、receiver-side committed admission、producer epoch、ObservationCut record/bundle | M1A additive contract；除后续 shadow 使用外未成为全局 observation hub |
| `robot_motion.deploy.observation_view`、`PolicyDeployRuntime` | artifact-bound `PolicyObservationView`、strict actor input、sanitized object-policy compatibility | M1A.5 已接当前 policy deploy prediction；GUI/collection 不是 policy actor caller |
| `robot_motion.deploy.{inference,chunk_plan,async_producer}`、`deployments.client.{rtc,rtc_executor,rtc_transport}`、`deployments.openpi.client.{rtc,transport}`、`sim.mujoco.deploy.rtc_policy`、RTC deploy compatibility frontend | immutable async inference request/result、target-time mutable plan/merge、single-admission producer snapshot、单due-keyframe pop、emission后current-only commit、deterministic authoritative sim、session-bound bounded OS-thread carrier、model-neutral capability/executor/deadline与OpenPI final-chunk/transport映射 | RTC-1～4b-T2c scoped已落地；默认sync，显式RTC仍仅builtin-hold/no-action。T2c已完成immutable→wire thaw、live worker、cold TTL拒绝、warm 44-keyframe single admission、零emission/legacy-pop与完整drain。正式trace、真实observation、M1C execution drain、real shadow和authority未接 |
| `robot_motion.deploy.action_adapter`、`PolicyActionEmission` | 同一次正式 queue pop 产生 pre-safety absolute `RobotAction`、source validation、chunk/inference/effective-time lineage | M2 已接 deploy queue-pop seam；legacy canonical safety/command 仍是执行 owner |
| `robot_motion.action.{proposal,snapshot,authority,candidate_authorization,arbitration_input,arbiter,arbitration_runtime,persistence,safety,authoritative_control}` | immutable proposal、完整 arbitration input、target-specific candidate authorization、pure Arbiter、publication gate、per-group persistence、typed safety 与 backend-neutral authoritative orchestration | M1B contract 完成；M3 shadow 与 M5.5a–c policy/replay/cancellation deployable sim 已复用，real frontend 尚未接 authority |
| `robot_motion.runtime.cancellation` | execution epoch、group generation、worker guard、ack/deadline contract | M1C fake-worker + M5.5c MuJoCo software-queue/hold scoped integration；real reference/command/SDK worker 未接 |
| `robot_motion.runtime.{dispatch_owner,dispatch_seam,dispatch_inventory,worker_drain}` + `teleop_runtime.backends.arms.{rm65_dispatch_call,rm65_control_dispatch_call,rm65_dispatch_inventory,rm65_worker_drain}` | 同进程 owner token、motion/safety/lifecycle permit、handoff、纯 final-call gate/legacy callback adapter、不授予执行权的 profile-specific seam inventory，以及逐participant旧epoch drain证据 | M1D/M5.8-a fake-backend + M5.5a–c MuJoCo motion/safety-control seam；首个arms-only候选已有16条typed seam和三个scoped-unbound drain participant。joint-target三类语义及七条stop/start/cleanup SDK seam已有真实函数内default-off typed adapter，共10条标记`pure_wrapper_validated`；production未注入provider、其余6条仍unwrapped、worker无receipt，因此评估仍明确`legacy_mediation_ready=false`、`confers_execution_authorization=false` |
| `robot_motion.action.{execution_shadow,keyframe_safety,safety_anchor,execution_target,reference_segment,safety_migration}` + policy/human safety shadow | 纯 proposed persistence→显式 shared safety→immutable command-intent evaluation；显式 anchor/profile 的 pose/qpos keyframe safety adapter；按 group/epoch/generation 的 previous-safe-committed/measured-rebase 生命周期；ordered segment endpoint、receiver admission/guarantee、terminal 与 target-time-aware pending/effective anchor timeline；分M5.8-a/M5.8-b/M6且带证据的per-profile migration gate；policy legacy-aligned numeric 双算；human exact legacy-filter evidence/pose/qpos pure parity | M5.7a/b/e1/e2-a、M5.7c1/c2、M5.7d-a/b和M5.7f pure gate contract已接；首个arms-only profile已有绑定manifest/inventory的scoped matrix，只签署M5.8-a inventory ready，所有included rule在M5.8-b/M6仍逐条blocked。scheduled MuJoCo receiver已有canonical admission/anchor/reference lineage，legacy real worker仍仅有non-authoritative observation；无真机shared segment/permit/backend owner；measured divergence阈值、hand safety、human position-backend lineage和pose/per-tick IK reference尚未完成 |
| `teleop_runtime.deploy.arbiter_shadow` | passive single-policy proposal→snapshot→pure Arbiter→`ArbitrationEvaluation`，并与 legacy execution 分型 | M3 已接 `run_policy.py`，默认 off，且与 `--send-actions` fail closed；只写 debug event |
| `dataset_contracts.{v24,control_trace,completeness}`、`teleop_runtime.control_trace` | event registry、M3/M5 explicit record schema、六个分作用域 profile、branch/per-lineage completeness、episode membership provenance、logical envelope 与有界内存 journal | M4A.1 contract/test；M5 opt-in GUI human shadow 默认只接 subordinate runtime journal，formal capture 需显式组合 M4B.2 sink |
| `dataset_contracts.{control_trace_episode,control_trace_wal}`、`teleop_runtime.control_trace.{episode,episode_manifest,spool,health_ledger,sink}` | per-episode range/dependency/seal/commit、subordinate owner seam、atomic manifest publisher、canonical WAL、durable ack/high-water、FIFO release barrier 与 torn-tail recovery | M4B.2a/b human fixture及CS-2b-c dataset-backed replay formal fixture；seal-before-commit已验证，production builder尚未attach，任意crash点/长稳/目标机验收与policy formal profile仍未完成 |
| `dataset_interface.control_trace` | committed seal/session-WAL locator、range/dependency/membership/hash 校验、可重建 `ShadowEvaluationView` 与 authoritative variable-dt `ActualDecisionView` | M4C.1 + scoped M4C.2；deterministic与真实`MujocoScene + actuated rig` formal fixture均已seal/commit/read；seal后无commit默认拒绝、审计模式可重建；不写Parquet cache，production composition仍未验收 |
| `teleop_runtime.action.{human_proposal,human_shadow,human_shadow_frontend,human_position_shadow}`、`TeleopSession` passive hooks | strict human action adaptation、同一 emission 的 pre-filter/post-dispatch lineage、typed backend receipt preservation、position transport-target→audit pure join、profile-derived identity/component/spec、shared-Arbiter evaluation和bounded observer queue | M5/M5.7d canonical GUI arm/Inspire默认关闭wiring；pose/qpos limiter parity与position pure binding已有，production position-audit async join/ControlTrace未接；legacy owner不变，DMTAC session外分支fail closed |
| `robot_motion.deploy.context`、`control_composition.{policy_node,runtime,composition}`、`sim.mujoco.deploy.async_topology` | transport-neutral composition-node protocol、receiver-admitted context/internal composition input、同-cut context/input freshness gate、target-time snapshot、component union/qpos bounded residual、typed composite lineage与确定性受限 async completion queue | M5.2 pure harness + M5.5d 10/40/30 Hz deterministic fixture + M5.5e1 latency/jitter/deadline/cutoff/restart/burst/HOLD fixture；通用实现已迁出RobotMotion，无 production PolicyGraph、OS/GPU worker scheduler、distributed binding、dispatcher 或 action queue owner |
| `dataset_contracts.replay`、`dataset_interface.replay`、`robot_motion.replay.{proposal_source,compatibility,control_trace}`、`teleop_runtime.action.{replay_binding,replay_proposal,replay_shadow_frontend}` | replay mode/action-stage/schedule、typed source/target contract、三态 binding assessment、scoped eligibility、execution authorization、canonical replay proposal | M5.3/M5.3a complete；M5.3b GUI passive consumer 保持 legacy owner；M5.5b 仅 sim authority 已接，真机未授权 |
| `teleop_runtime.collection.CollectorSample`/`LeRobotV21Writer` | 单主表 row、外置 video/tactile/depth、v2.2 layout 与 v2.3 pose/causal 校验 | 当前正式 dataset owner 基础；尚未挂 v2.4 ControlTrace sink |
| `robot_motion.position_control`、`teleop_runtime.position_control.PositionDeployExecutor` | causal resampling、reference/IK/trajectory/command 和 real position execution | 当前 legacy execution owner 的下游能力；尚未消费 authoritative Arbiter decision/epoch permit |
| `sim.contracts.TimingSpec`、`sim.mujoco.{actuation,deploy}` | 独立 timing 字段、既有 actuated position sender/plant、per-node state ObservationCut/action adapter、policy topology、scheduled qref receiver/reference lineage 与 shared authoritative runtime binding | M5.5a–e scoped 接入；scheduled profile已有canonical segment/admission、active/pending generation、target-time anchor和120 Hz planner-command lineage，immediate profile不伪造segment；deterministic与真实MuJoCo formal fixture、START前HOLD和seal-before-commit恢复已验证；OS/GPU worker和pose/per-tick IK reference仍未验收 |

### 3. 当前仍未统一或尚未接入的事实

- `run_policy.py` 仍只构造一个 policy、一个 `PolicyDeployRuntime`、一个 action queue；真实 observation、
  startup/readiness、fault、position executor 与 legacy dispatch 仍在入口中，没有 `PolicyGraph` 或多实例调度。
- strict `PolicyObservationView` 已解决当前 deploy actor 的 control-private 泄漏；但 arm/hand/tactile 仍被组装成
  单一 `DeploymentObservation`，尚无支持多个策略按各自 cut/window 投影的 Temporal Observation Hub。
- pre-safety absolute `RobotAction` 和 policy proposal 已能在 M3 shadow 中生成；human 的 canonical GUI
  arm/Inspire 路径可在 M5 opt-in frontend 中从同一次 `TeleopCommand` emission 被动生成 proposal/evaluation，
  但仍不是 authoritative producer。DMTAC gripper 仍在 `TeleopSession` 外独立下发，启用 shadow 时 fail closed；
  replay 的 debug policy 与 canonical GUI compatibility owner 现都可从同一次 immutable emission 被动生成
  `producer_kind=replay` proposal/evaluation；CS-2b-c测试侧composition还可将该GUI worker emission接入MuJoCo
  authority并formal seal/commit/read，但production GUI与真机仍不是authoritative producer。
- `ActionArbiter`/`ArbitrationRuntime` 已有 M5.5 sim authoritative caller，但 real production frontend 仍无
  authoritative publication caller；M3 只运行 non-authoritative evaluation，真机 selected/safe/command/receipt
  事实仍由 legacy path 决定。
- M1C/M1D 的 cancellation 与 dispatch-owner guard 已在 MuJoCo authoritative seam 验证 motion/safety permit、
  execution epoch、pending reference 清除与 software-queue ack；现有 `TeleopSession`、real
  `PositionDeployExecutor`、Rig/SDK 仍没有校验这些 epoch/token，且 sim 结果不等于真实停止确认。M5.8-a 已新增
  未接 production 的纯 `DispatchSeamGate` 并完成第一轮调用点 inventory：Rig 只是 logical fan-out，RM65
  position/impedance 的后台 worker、Inspire retry 和 DMTAC direct bypass 都要求各自最终 pre-SDK gate；一个上层
  keyframe permit 不能复用于多个高频 SDK tick。
- shared runtime 不得在M6 owner切换时另建一套registry。当前runtime已支持全量注入并验证同一
  `DispatchOwnerRegistry/current token/ExecutionEpochRegistry`；部分注入、legacy token或epoch不一致均fail closed。
  这只是composition prerequisite，canonical frontend还没有production legacy owner/epoch lifecycle。
- M3 policy 的 `ArbitrationEvaluationEvent` 与 `LegacyExecutionObservationEvent` 仍只进入 debug JSONL；M5
  human frontend 已把 explicit projector/`ControlTraceRuntime` 挂到 canonical GUI 的 opt-in shadow 路径，并已
  用显式 recorder composition fixture 验证 episode membership、dataset-local WAL、FIFO release barrier 与
  seal/commit；production builder 仍未 attach。M5.3b replay GUI 的默认production路径仍只是subordinate
  RuntimeTraceSession/可选debug WAL；CS-2b-c仅在test-side application composition中复用同一loader/preflight/worker
  emission与唯一episode owner，并完成replay-specific seal/commit/read，不能据此声称production capture已激活。
- M4A.1 已对当前 M3 policy-shadow 子集检查 queue-pop emission→source validation→proposal admission、snapshot
  child、computation/evaluation 和 legacy execution 的 branch/per-lineage 引用与 hash；M5/M5.2 已增加 human、
  context/composition producer、explicit record/projection 和 fixture，但尚未把 context/composition 纳入
  branch-aware completeness。M5.3a/b 已增加 replay source/target assessment、eligibility、GUI source-frame/
  proposal/evaluation 的 lineage fixture；authoritative、DMTAC session 外 human 和 replay post-dispatch
  safe/command/receipt lineage 仍无 producer/fixture，完整因果时间、
  stream integrity也未实现。generic/policy formal profile仍显式标记`planned`；replay现只有
  `deployable_sim_scheduled_pose_to_qpos_station_v1`这一窄profile为`runtime_integrated`，不能泛化为全部replay或real
  capture。human scoped formal fixture同样尚未production激活。
- M4B 已把 logical envelope 编码移出 `ControlTraceRuntime.issue()` 的 admission lock；runtime 使用增量
  accepted/durable high-water，并在锁内只复制 diagnostic snapshot 所需的标量/集合，排序与 fact-ID 物化在锁外
  完成，避免高频 reader 长时间占用 control-facing admission lock。本地 contention fixture对 256 个小型事实
  记录 p50/p95/p99/deadline miss，并验证 spool I/O/fsync 只由 storage-worker sink 调用。但
  该证据不是 formal real profile 的长稳/大 action payload/OS worst-case acceptance；M5 只把相同 sink/worker
  接成显式 debug WAL，没有 episode owner、formal profile 或 seal。
- 当前 v2.3 `CollectorSample.action`/parquet 根 `action` 仍是合并动作；没有持久化全部 proposal、selection、
  safe/reference/command/receipt/measured lineage，也没有 ActualDecisionView、reward 或 transition overlay。
  M4C.1 已能从 committed ControlTrace facts 重建独立 `ShadowEvaluationView`，但尚未成为 production capture
  reader，也不覆盖 v2.3 根 `action`。
- `observation.last_action` 仍按入口保留不同 stage：collection 是上一条 session command 派生 action，deploy
  是 accepted canonical safe action。它们已被 M0/M1A.5 显式隔离，但尚未进入 v2.4 per-policy view contract。
- external tactile 仍由主 sample 触发落盘，不能完整保存高于 decision/sample rate 的 native stream；RGBD
  frame reuse、tactile window 和多速率 stream eligibility 尚未落地。
- sim 已在 M5.5a–e1 作为 deployable shared control-core backend 覆盖单策略、verified replay、cancellation/hold、
  deterministic 10/40/30 Hz 快慢 composition，以及受限 async completion/cutoff/restart/burst fixture；scheduled qpos
  profile已有receiver-side canonical reference admission、active/pending generation和实际120 Hz planner/qcmd lineage；
  测试侧 deterministic-scene composition 已复用真实 WAL/seal/commit/reader 完成 formal fixture，但不等于真实
  MuJoCo backend 或 production entrypoint。尚无 production PolicyGraph、真实OS/GPU worker/模型或pose/per-tick IK
  reference publication。replay proposal 已完成 typed dry-run/core、三态 binding evidence、
  debug dataset replay passive wiring 和 canonical GUI 默认关闭 consumer/RuntimeTraceSession，但 formal
  episode/post-dispatch trace 未接且 owner 未迁移；TrainingView、learner/artifact provenance 和闭环再部署
  均未开始实现。
- 当前工作树包含大量并行修改，M0–M4A 新文件仍处于本地未提交状态；本文中的“已完成”只表示当前工作树
  内实现和已运行测试，不表示已合并、发布或通过真实硬件验收。

### 4. 当前频率的真实含义

`PositionControlRates` 已明确 sample/reference/IK/command 字段相互独立。当前代码中至少存在三套 position
profile：

- library 默认：`15 sample → 75 reference、75 nominal IK → 150 command`；
- 真机遥操对齐 profile：`15 → 60 reference、60 nominal IK → 120`；
- compatibility profile：`15 → 45 reference、45 nominal IK → 150`。

`sample_hz=15` 不是全系统主时钟。但当前真机 `PositionDeployExecutor` 的 coupled IK 实际在 reference
worker 中调用；`dual_arm_ik_hz` 目前是 nominal solver budget，不是独立 scheduler。因而 library 默认的
75 Hz reference 配置不能被解释为“coupled IK 已在真机 75 Hz 稳定验收”；是否每个 reference tick 运行
coupled IK 还取决于 action space、collision profile 和 executor 路径。当前 policy deployment 的
`effective_policy_hz`、action keyframe rate、position executor `sample_hz`、reference rate 和 command rate
也不是天然同一含义。

因此后续不能再以 `15→75→150` 或 `15→60→120` 代表完整架构。它们只应是运行 profile；其中现有
`15→75→150` library default 和 `15→60→120` real-aligned profile 都继续按各自 contract 演进，不在本次
Arbiter 迁移中被静默改值。新增的 production decision policy v1 约束是
`ActionArbiter.decision_hz <= 60 Hz`，不能借此改写其他链路的频率语义。另一方面，正式启用 coupled IK 的
real profile 必须单独声明并验收实际 IK call clock；若要得到“60 Hz solve + 75 Hz joint reference
interpolation”，必须先实现对应的缓存/插值边界，不能只改配置名称假装已经解耦。

---

## 二、必须长期保持的架构不变量

1. **每个 control domain 一个执行事实源。** producer 可以很多，但每个 `control_domain_id` 只有一条
   authority、canonical safety/planner/reference 链和一个逻辑 dispatch port；任何 policy node 都不能
   直接调用 Rig、MuJoCo actuator 或硬件 SDK。vector sim 可批量承载多个 domain，但不能共享 authority 状态。
2. **proposal 统一到 absolute `RobotAction`。** raw model vector、relative action、human clutch delta、
   replay row 都在各自 source adapter 内转换；Arbiter 不做 frame/anchor/SE(3) 解码。
3. **观测、proposal 和执行全部因果。** 所有 cut 只能使用 source timestamp 与 available timestamp
   都不晚于 causal cutoff 的数据；远端晚到结果不能回写过去。
4. **模型边界位于 Arbiter 之上。** actor 只看到其 artifact 声明的 observation/context；selected owner、
   safety clamp、IK/fault、command、reward 和 trace metadata 默认不可见。
5. **安全反射不可参与普通竞争。** 触觉 policy 是 proposal producer；过力、碰撞、失联、hold 等硬安全
   是 Arbiter 后的 veto/modification 或独立高优先级安全状态。
6. **component attribution 与控制同 cut 生成。** 不能从最终 command 数值反推 human/policy 来源。
7. **v2.4 事实层使用显式 `selected_action` 表示 authoritative Arbiter 统一后的 canonical intent。** shadow
   只产生 `ArbitrationEvaluation`/`would_select_action`，不得填充 authoritative `selected_action`。safety/reference/
   command/receipt/measured 等下游事实另存并默认对 actor 不可见；裸 `action` 只允许由带
   `action_stage` 的 materialized/training view 映射生成，不得在事实层同时表示 intent 和 hardware
   behavior。事实层不指定任何默认训练 target。
8. **逻辑拓扑与部署拓扑解耦。** 单进程、IPC、共享内存、远端 RPC 只更换 transport adapter，不改变
   ID、时间戳、validity、action 和 trace contract。
9. **不要求整数频率比。** 调度、因果选择和重放基于 timestamp/deadline，不依赖“每 4 帧取 1 帧”。
10. **production decision policy v1 的 ActionArbiter 不超过 60 Hz。** 使用该发布策略的 production
    control domain 必须满足 `arbiter_decision_hz <= 60 Hz`。该约束不限制 producer、sensor、reference、
    IK、command、backend 或 recording clock；安全 veto/hold 可异步立即生效，但不得借事件旁路发布超过
    60 Hz 的普通 selected action。更高上限必须通过新的版本化 policy/capability 和安全验收发布。
11. **兼容路径先包裹、后替换。** 不一次重写 GUI、`run_policy.py`、writer 或 position executor；先
   observe-only，再切换 authority owner。
12. **动作阶段不共享一个假 component schema。** selected、reference、IK、command、measurement 分别
    绑定自己的 schema，并通过可多对多的 transformation lineage 关联。
13. **source device 与 embodiment 分别适配。** VIVE、其他 tracker、手柄或离线 replay 的 raw schema 只在
    source adapter 内存在；robot topology、kinematics、actuator/backend 只在 embodiment/execution adapter
    内存在。两侧通过 canonical observation/action/component/frame contract 连接，不能让设备字段或 RM65
    SDK 类型进入 Arbiter、TrainingView 或 policy artifact tensor contract。
14. **闭环 artifact 必须可追溯并 fail closed。** learner 产物必须引用 parent capture/derived overlay、
    TrainingView、PolicyIOLayout、observation/action/frame/component contract 和代码/配置 hash；deployment
    只有在当前 source/embodiment capability 满足 artifact compatibility spec 时才能进入 shadow/authority。
15. **“对更换不敏感”不等于任意硬件零适配。** compatible adapter 更换不得改变 control-core API 和事实
    语义；运动学、DOF、frame 或 action schema 不兼容时必须显式 retarget、重新训练或拒绝启动，不能静默
    reshape/截断动作。
16. **一次 arbitration 只由一个完整不可变输入决定。** `ActionArbiter` 只消费可 hash 的
    `ArbitrationInputSnapshot`，不得在计算中读取 ProposalBuffer、AuthorityStore、RuntimeSupervisor、frontend
    变量或“当前最新”状态。snapshot 后到达的 proposal/authority/gate/producer-epoch transition 只进入下一
    decision；但异步 safety/HOLD/FAULT 仍可使尚未发布或执行的旧 result 立即失效。
17. **执行资格由接收端 admission 线性化。** producer 只声明 `produced_at`、effective interval 和 provenance；
    本地 `available_at` 只能在 receiver 完成 validation、deep-freeze、epoch/sequence 检查并原子接纳后赋值。
    context、authority event 和 producer-epoch activation 遵守同一原则。
18. **Composition 与 Arbitration 分离。** base/residual、component union 或 SE(3) composition 由版本化、
    确定性的 Composer 在 admission 前生成新的 composite proposal；Arbiter 只做 authority/selection，不做
    数值 blend，也不全局假设权重和为 1。
19. **`no_action` 不等于无限延续旧动作。** selected segment 必须有明确 expiry；`no_action` 不创建或续期
    segment，`hold/fault` 通过 execution epoch 取消尚未执行的旧 reference/command。具体持续语义由版本化
    `ActionPersistencePolicy` 冻结，不能散落在 frontend/backend。
20. **最终 dispatch 有代码级硬互斥。** 每个 control domain 的 logical dispatch port 必须校验当前
    motion dispatch-owner generation 以及 command 的 authority/execution epoch；配置开关不能作为 legacy/new
    owner 互斥的唯一保证。human/policy/replay 是 proposal authority，不分别拥有 dispatcher token；安全与
    lifecycle 命令使用各自窄 permit，不能借 motion owner 或通用 bypass 混发。
21. **sealed/committed capture 不允许原地维护。** episode 一旦发布 v2.4 capture commit，删除 episode、重排
    legacy episode index、重压缩或替换 stream、修改 manifest/registry 都不能回写原 capture root，也不能保留
    原 commit identity 后重算内容。此类操作必须生成带 parent dataset/commit hash、derivation spec 和新 identity
    的 copy-on-write derived dataset；只有 legacy v2.2/v2.3 或尚未 commit 的 staging capture 可以在显式 profile
    下做原地维护。
22. **默认配置名不构成兼容证据。** pose/frame、camera transport、timing、component 和 artifact compatibility
    只能由实际 canonical payload/hash 与 `EpisodeExecutionManifest` 判断。历史文件名、CLI 默认值、数组长度或
    “同为 RM65”都不能替代 exact contract preflight；兼容 alias 不得被解释为旧语义仍然存在。
23. **关节位置限位由 embodiment URDF contract 决定。** 当前 execution manifest绑定的URDF是kinematic joint
    position range的唯一事实源；SDK/hardware envelope只能提供额外更窄的backend guard，不能放宽URDF。default、home、
    recovery、replay和policy seed都必须在进入执行链前通过同一URDF限位；历史fallback不得被静默clip成新的home。
24. **Profile 复用实现与证据，但不继承授权结论。** 新profile可以按hash引用相同的generic contract、URDF、
    component、backend seam和pure/sim evidence，只补自己的delta；真机motion authorization、stop/latency、
    measured safety与operator approval始终绑定具体`ExecutionManifest`，不能因共享机器人名称自动传递。
25. **GUI/Control Station 只是控制平面。** 它只能提交receiver-admitted typed operator intent和展示可重建状态，
    不得读取第二份device/dataset/model stream，不得拥有ActionArbiter、policy chunk plan、dispatcher、owner/epoch registry
    或episode lifecycle。浏览器连接、断线、重连和status polling不得改变queue pop、replay cursor、inference schedule、
    dispatch或capture事实。

泛化防火墙中的`robot_motion/action/pose_transform.py`当前登记为显式v1 compatibility surface：其
`CanonicalDeskToRuntimePoseTransformer`只服务`bimanual_ee_pose_v1`的left/right coupled pose事实。该例外不表示
generic core可以继续新增名称解析或固定payload shape；在第二本体EG2/EG3前保留现有数值与CS-2b replay证据，届时再由
component/schema驱动的transform binding替代。

---

## 三、目标逻辑架构

```text
RGBD / depth ───────────────┐
tactile / force ────────────┤
robot state ────────────────┼── Temporal Observation Hub
human device ───────────────┘      │
                                   ├── per-policy ObservationCutRecord/View
                                   │
                    ┌──────────────┴───────────────┐
                    │                              │
             slow/context producer           action policy node(s)
                    │                              │
          versioned CompositionContext             │
                    └──────────────┬───────────────┘
                                   │
                          fast/action policy
                                   │
                             ActionProposal
human / replay / scripted proposal ┤
                                   ▼
                           ProposalBuffer snapshot
                                   ▼
                             ActionArbiter
                         control port │ attribution port
                                      │──────────────→ ControlTrace
                                   ▼
                 canonical safety / IK / planner / reference
                                   ▼
                     command interpolation / scheduling
                                   ▼
                 one logical dispatch port per control domain

All lifecycle/health ─────────────→ RuntimeSupervisor
```

### Control Station 应用控制平面

human、replay和policy最终由同一个应用层实验台按source slot管理，null/MuJoCo/real由execution-domain binding
替换；但Control Station不插入proposal到dispatch的数值路径。它只负责manifest选择、连接、preflight、shadow、
authority/lifecycle请求、状态展示和capture请求。每个source继续使用唯一既有emission seam：human不重复读取遥操设备，
replay不建立第二个dataset cursor，policy不重复消费action chunk。

代码落位固定为仓库根级`control_station/`，而不是`teleop_runtime/control_station/`。它是顶层application leaf，可按
显式adapter导入human/replay/deploy/dataset/sim/real模块；所有下层模块均不得反向导入它。这样本体或real/sim替换只改变
composition binding，不会让通用控制、仿真或真机runtime天然依赖GUI平台。

```text
minimal GUI / headless client
        ↓ typed StationIntent
ControlStationRuntime（application composition only）
        ├── human source adapter  ─┐
        ├── replay source adapter ─┼→ existing proposal/Arbiter/execution core
        └── policy source adapter ─┘
                                   ├→ null target
                                   ├→ MuJoCo target
                                   └→ real target（另行manifest/authorization）
```

详细 contract、文件落位、CS-0–CS-7迁移和真机human/replay/deploy分开验收顺序见
[`dexteleop_control_station_plan.md`](dexteleop_control_station_plan.md)。正式入口在sim parity和legacy-real passive
parity前保持不变，不以新GUI为由复制owner、dispatcher或episode lifecycle。当前CS-0a pure contract与CS-0b
null/fake no-I/O composition已落地；CS-1a另有一个不自启动、仅loopback、只提供health/state GET且拒绝全部POST的
read-only server builder。CS-1b现已增加仅适用于null/no-motion manifest的authenticated typed-intent builder；Web token
不替代operator lease，HTTP response明确`adapter_executed_inline=false`，raw action和motion intent在runtime admission前拒绝。
CS-1c又增加bounded、non-authoritative UI event ring与独立no-I/O operation coordinator：capacity覆盖queued+in-flight，
并在intent admission前预留；HTTP线程不执行adapter，completion由worker回到唯一runtime result-admission seam；event eviction
显式报告gap且不改变ControlTrace/capture completeness。CS-2a1现已增加一个独立scoped application slice：无设备synthetic
human qpos经现有human emission/adapter进入同一MuJoCo authoritative core并留下source→proposal→decision lineage；它不创建
第二action queue/owner/epoch/dispatcher。CS-2a1 binding只证明authoritative data plane精确接通；CS-2a2b另以scoped fixture
证明candidate/authority必须经过typed completion，完成后才开放publication，release会关闭publication并通过同一core进入
HOLD/cancellation，且Station保持hold latch。该scoped binding仍不是production async/web authority入口。CS-2a2a现已用scheduled-qpos source-specific execution manifest
与真实WAL/episode owner/seal/commit/reader完成test-side formal capture；immediate-qref仍是fixture-only且不可commit。
source-agnostic原子publication barrier现已接到测试侧bounded side-effect worker，release在入队前关闭barrier，异常不会遗留pending；
Station manifest/intent/result/barrier事实又已通过注入式bridge进入专用scheduled-sim formal profile，并用真实
WAL/seal/commit/reader重建；旧profile仍将这些新增事件视为optional。现有CS-1c Web仍以exact no-I/O类型防火墙
拒绝该worker；另一个明确标为`cs2a2b_scoped_sim_authority_api`的loopback builder现已只对exact sim manifest/side-effect
coordinator开放typed lifecycle/candidate/authority intent，raw action、real target和production entrypoint均fail closed。
Web token不替代lease/completion/barrier/owner-epoch。CS-2b-a现已按同样边界接入qpos-compatible immutable replay。
CS-2b-b又增加exact arm-base pose和verified canonical-desk两种独立manifest：pose始终保留为proposal/selected/safe，
canonical desk到arm base的版本化变换只发生在shared safety后、coupled IK前，IK输出另建qpos execution target。现有
replay executor仍只构造一次immutable source-frame emission；lower-owned bounded fanout向GUI shadow和root Station
consumer传递同一对象，branch failure不反向影响cursor/resample/legacy dispatch。CS-2b-c现已在测试侧复用真实dataset
loader/preflight/GUI worker与同一个checked group，并用replay专用pose→resolved-qpos profile完成WAL/seal/commit/read；
该成熟度是formal fixture，不是production GUI replay、真机authority或新entrypoint。`qpos_target` GUI branch仍因缺少
lower-owned immutable emission seam而fail closed。CS-2c首个sync-policy data-plane slice也已落地：根级composition只消费
现有`PolicyDeployRuntime`唯一queue pop产生的exact `PolicyActionEmission`，并保留emission/target interval/inference/chunk
lineage进入MuJoCo proposal；它不调用predict/merge/pop，也不拥有action queue。独立exact manifest已进一步接入typed
Station lifecycle：completion后才开放publication，release先关闭barrier再由同一lower core接纳HOLD/cancellation。当前尚未
接artifact preflight；policy-specific formal fixture已用专用profile把exact emission/validation设为required，并完成
scheduled execution的WAL/seal/commit/read与ActualDecisionView重建。该结果仍不是production policy source。下一RTC
Station slice的正式data plane现已改为lower-owned
`RtcPolicyEmissionSourceBinding + PolicyActionEmission`：binding验证epoch/config/source identity，emission自身绑定
同一`source_binding_hash`，然后复用同一MuJoCo authoritative core。完整`RtcFrontendStep`只保留debug compatibility
验证，不再是Station、Control Composition或target的正式输入。异步推理期间deterministic fixture只同步推进既有target
physics clock，不新增policy scheduler。
downstream publication的Station lifecycle现已按该边界接入：connect/preflight不启动producer，release先关闭barrier再
HOLD execution core，且不改变RTC producer snapshot/worker session/plan/epoch。专用immediate-qref formal profile又将
exact emission、source validation、Station lifecycle、execution manifest、receipt和measurement纳入真实WAL/seal/commit/
committed-reader闭环，并把scheduled-reference family显式标为不适用。human/replay/sync-policy/RTC policy仍须按
单source分别验收；该fixture不是artifact/transport/production/real证据，不能接真机或替换现有GUI。
CS-3a现又把existing deploy唯一正式emission seam暴露为default-off、bounded、
committed-only `PolicyActionEmissionFanout`；root Station只接收target-neutral binding与同一个immutable emission，
不接收frontend或完整step。null-target manifest仅允许shadow，同队列ordered fence界定observer可见区间；
connect/preflight/start/stop已验证不会改变producer snapshot、worker/plan、selected-emission count或lower queue-pop。
当前binding按producer epoch冻结，rearm后旧binding fail closed；自动binding rotation尚未完成。该seam没有
artifact/transport/real-observation preflight、target execution、ControlTrace episode或motion authority，不能由
“existing frontend已接”外推为RTC-6 real shadow或production Station。
CS-3b-a现已在root application层增加纯OpenPI RTC source assessment：只比较已有artifact preflight、
capability/request negotiation、backend构造时已验证的server metadata与CS-3a source binding，并绑定到仍为null-target、
shadow-only的manifest。该assessment固定将live transport、real observation和target binding标为`not_assessed`，将execution/
motion标为denied；它不读取会主动连接的transport metadata property，也不创建websocket、加载artifact或启动producer。
CS-3b-T仍需lower-owned transport session/health与真实observation evidence，不能由CS-3b-a外推。
scoped loopback API只允许typed connect/preflight/start/stop通过bounded side-effect worker完成，浏览器不能提交action或
candidate/authority intent，且任何状态投影都固定`motion_authorized=false`；没有正式入口调用该builder。

CS-6b1e现已增加device-neutral passive pose session：application profile提供任意既有`ObservationSource`、pose retargeter和
calibration，root仅构造`TeleopSession(rig=None, recorder=None)`并拒绝record/hardware-arm eligibility。CS-6b1d可在human slot
的typed connect后启动该source和执行单次observe-only poll；emission必须精确匹配source spec声明的pose effectors、frame与
finite homogeneous pose，默认spec仍为RM65双臂tuple，`PassiveTeleopPoseSourceSpec.for_arm/for_effectors`可构造单臂或其它
明确topology；
但当前不进入proposal admission，replay仍为唯一active authority。虚拟测试已覆盖non-pose/frame/rig/observer冲突、外部ARMED
跃迁、subscriber failure和replay dispatch parity。该结果不修改canonical遥操/replay入口，不绑定Vive、具体机器人、sim或模型，
也不提供真实human READY；后者仍需lower typed deadman-rearm与retargeter-rebase receipts。

真实设备接线的配置事实源固定为当前正式真机遥操数采链路：`teleop_runtime.collection.run_collect.build_collect_app`已经统一
组合Vive tracker/clock sync、HandQ6 source、arm/hand retargeter、工具TCP、collision/safety、各后端与`BimanualRig`。现已提供
公开的`teleop_runtime.configs.build_bimanual_passive_session` projection：复用该链路的source/retargeter/calibration构造，
但强制`rig=None`、`recorder=None`且拒绝hardware-arm eligibility；构造本身不启动source。Control Station及后续composition
必须调用该公开profile，而不是复制端口默认值、tracker对齐或手部映射，也不能直接import私有`_build_*`函数。新增机械臂、
遥操设备或sim仍通过相同public profile projection和typed component contracts接入；target backend/rig继续由各自target
composition持有。既有`build_bimanual_session`默认路径和canonical遥操/数采入口保持不变。

单策略模式只是 topology 中只有一个 action policy node：

```text
Observation Hub → one PolicyDeployRuntime → ActionProposal
                → ActionArbiter → existing canonical execution
```

快慢系统则是多一个 context edge，不产生第二套控制系统。

### 1. 控制—采集—训练—再部署闭环

```text
human device(s) ─→ SourceAdapter ─┐
policy artifact ─→ PolicyNode ────┼→ ActionProposal → Arbiter → canonical execution
dataset replay ─→ ReplayAdapter ──┘                         │            │
                                                            └────┬───────┘
                                                                 ▼
                                              native streams + ControlTrace + task facts
                                                                 ▼
                                                    sealed v2.4 capture dataset
                                                                 ▼
                                      TrainingView / reward overlay / transition view
                                                                 ▼
                                                BC / DAgger / RL / fast-slow learner
                                                                 ▼
                                       TrainingRunManifest + PolicyArtifactContract
                                                                 ▼
                                          deployment compatibility preflight
                                                                 ▼
                                                   shadow → authority → new capture
```

每次迭代都必须能回答：模型看到了哪些 cut/context、target 来自哪个 source/component、实际 behavior 到了
selected/safe/commanded/measured 哪一层、reward/outcome 如何计算、checkpoint 使用了哪些 capture/view，
以及该 checkpoint 在哪个 source/embodiment binding 上通过了 shadow/authority 验收。

采集能力不能绑定 human GUI。GUI/headless、policy deployment、replay 和 sim rollout 都可以通过同一个
`EpisodeLifecycleOwner` protocol 挂接 native streams + ControlTrace sinks，生成正式 v2.4 capture；区别只在
谁创建 episode/task metadata 和哪些 completeness capability 可用。没有正式 episode 的普通 deploy debug
仍使用 `RuntimeTraceSession`，不能冒充训练数据。v2.3-compatible row 只在需要兼容 reader 时物化，不是
policy/replay capture 的前提。

这里必须区分两个终点：`runtime_trace_terminal` 终止一次完整 `RuntimeTraceSession`；dataset episode 的
finalize/seal 则以该 episode 的 bind/unbind、journal sequence range、durable high-water mark 和 profile
lineage 为依据。canonical GUI 的一个 runtime session 可以包含多个 episode，因此单个 episode finalize
不得写入 runtime terminal 或关闭后续 trace admission。episode seal assessment 是 subordinate sink 的
不可变完整性证据，不拥有或替代现有 recorder 的 commit/finalize；runtime shutdown 也不得重写已经 seal
的 episode。M4B.2 正式接线前需先冻结这一级 contract 和 crash-recovery 规则。

### 2. 遥操设备与 embodiment 的可替换边界

```text
SourceDeviceContract + calibration/retarget spec（human device）
ReplaySourceContract + action-stage/schedule spec（dataset replay）
  raw human input or typed replay record
  → SourceAdapter
  → absolute RobotAction proposal + SourceAdapterReceipt

canonical control core
  ObservationCut / ActionProposal / Arbiter / ControlTrace / TrainingView

EmbodimentContract + ExecutionBindingSpec
  selected RobotAction
  → safety/kinematics/reference/command adapter
  → backend-specific command + receipt/measurement
```

- 更换 VIVE/其他 tracker/手柄只新增或替换 `SourceDeviceContract`、calibration/retarget adapter；同一 canonical
  component/frame 语义下，Arbiter 和数据 schema 不变，raw device stream 仍可作为审计输入保存。
- 更换 RM65、sim robot 或 hand topology 时替换 `EmbodimentContract` 与 execution adapter；proposal/selected
  事实尽量保持 canonical，command/measured schema 允许不同并由 transformation lineage 连接。
- policy artifact 不直接绑定某个 USB/SDK 类，但必须声明所需 observation/action/component/frame 和
  embodiment capability。满足同一 compatibility spec 的 backend 可复用 artifact；不满足时 fail closed。
- 数据集可以联合多个 source device/domain，但一个 TrainingView 只能联合 contract-compatible 的 canonical
  字段。跨 embodiment retarget 是版本化 derived transform，不修改 sealed capture。

“不敏感”按兼容层级验收，而不是按设备名称判断：

| 变更 | 允许保持不变的核心 | 必须更新/验收 |
|---|---|---|
| 兼容 human device | Arbiter、canonical proposal、TrainingView schema | source contract、标定/retarget、adapter receipt |
| 同一 embodiment 的 backend/SDK | proposal、decision、policy artifact | execution binding、command/receipt schema、safety capability |
| canonical component/frame 兼容的 embodiment | control-core API、可兼容的 actor/action view | embodiment contract、kinematics/safety/执行适配与 artifact preflight |
| DOF/frame/component 不兼容的 embodiment | 事实层与闭环协议本身 | 显式 retarget/新 TrainingView/重新训练；不能直接复用旧 artifact |

### 3. 当前实现的本体通用性边界

当前“可替换”分为两层，不能混称为已经支持任意机器人：

| 当前层级 | 已有能力 | 尚未具备 |
|---|---|---|
| 控制协议 | control domain、clock/epoch、authority、cancellation、proposal snapshot、selection group、dispatch permit、ControlTrace、episode seal/reader | 无明显 RM65 语义依赖；新增实现仍须通过依赖防火墙 |
| composition contract | arm pose、显式component subset、typed end-effector qpos/native目标 | component-keyed carrier尚未接，因此当前只能承载v1 left/right名称 |
| action/execution v1 | 左右 component、RM65 direct arm qpos6、xyz+quat pose、q6/q16 hand、RM65 sim/IK/safety profile | 非left/right component、一个arm多个末端、任意末端schema和payload-neutral Arbiter；7DoF只需新增resolver/execution profile |

同构 backend 替换也必须重新验证 joint order/方向/单位/零位、frame/state schema、worker/retry、HOLD/ESTOP/
home/rearm、receipt/cancellation acknowledgement 和每个最终设备调用的 permit；统一 Python 接口不构成
execution compatibility evidence。

真机前的泛化防火墙冻结为：

- `robot_motion.action/runtime` 的 generic 模块不得新增 RM65/Inspire/DMTAC import；现有 profile-specific
  `robot_motion.rm65` 和 v1 compatibility surface 允许保留；
- generic Arbiter 不得新增 `left/right` 名称解析、q6/q16 shape 分支或按数组长度推断 action space；
- `dataset_contracts`/`dataset_interface` 不得依赖 hardware、teleop runtime、sim 或 vendor SDK；offline
  postprocess 的 runtime bridge 必须显式隔离，普通包 import 不得加载这些依赖；
- 新 contract 必须以 typed schema/hash 声明 DOF、frame、units 和 component，不得根据 payload shape 猜测；
- RM65 profile-specific、v1 compatibility、generic-core leak、legacy materialized view 四类债务分别登记，
  不把合法的 embodiment 实现误判为核心泄漏。

当前防火墙已 scoped 落地：shared action/runtime 的禁止依赖和 v1 hardcode 文件采用 allowlist 测试；离线
postprocess 普通 import 已不再加载 hardware/robot runtime；DM-Tac calibration 文件契约已与 hardware 默认
provider 分离。这里的“通过”只证明依赖方向和已知债务不再扩散，不证明新的7DoF
resolver/execution profile、第三component或动态payload已经可执行。

第二本体前只规划三个批次，不进入当前 RM65 真机变更面：

1. **EG1 contract/binding**：冻结 `EmbodimentContract`、`EmbodimentBindingAssessment`、
   `ControlComponentSpec`、`ActionPayloadSchema`、`EffectorActionCapabilitySpec` 和 topology binding；contract
   compatibility、validated execution capability、motion authorization 始终独立。
2. **EG2 v1 bridge/fixture**：建立唯一 `RobotActionV1 → generic component-action` 无损投影；用单 7DoF、
   6+7DoF coupled、1DoF gripper、q16 hand 和第三 manipulator component 做无硬件正向 fixture。迁移期仍只有
   一个 canonical 事实源。
3. **EG3 second embodiment**：由明确的第二本体驱动 payload-opaque/schema-visible Arbiter、按 schema/profile
   注册的 safety/execution、manifest-driven builder 和 v2.4 native per-component facts；旧 left/right 字段只作
   compatibility materialized view。

Arbiter 的目标是数值不透明而非语义不透明：它仍必须验证 component ID、payload schema hash、coverage、
selection group、coupled 原子性和输出 action schema，但不解释 6/7DoF、pose 数组、q6/q16 或数值 safety。
不同 action schema 的 safety state 也不能统一成一种 previous-safe：position、velocity/twist、torque、trajectory
和 discrete mode 分别需要自己的版本化状态与 handler。Policy artifact 仍保持固定 `PolicyIOLayout`，控制 core
通用不意味着模型 tensor 动态化。

---

## 四、多速率时间模型

### 1. Clock 与 timestamp taxonomy

系统仍需分别配置 sensor、observation cut、inference、context、proposal、decision、reference/IK、
command、backend/physics 和 trace/media clock；`observation_hz != inference_hz != proposal_hz !=
decision_hz` 是合法配置。但频率配置不能替代事件时间语义。

v2.4 不给所有对象复用含义模糊的 `source_timestamp`，而是按事件角色冻结：

```text
SensorSampleRecord
  measured_at / available_at

ObservationCutRecord
  cutoff_at / built_at

PolicyInferenceEvent
  triggered_at / started_at / completed_at / received_at

CompositionContextRecord / ActionProposal
  produced_at / available_at / effective_from / effective_until

ActionKeyframe
  target_effective_at

ProposalSnapshot / ArbitrationDecision
  snapshot_cutoff_at / snapshot_created_at
  decision_cutoff_at / decided_at / target_effective_at

CommandEvent
  scheduled_for / sent_at / result_received_at

MeasurementSample
  measured_at / available_at
```

每个时间值都携带 `clock_domain`；控制 eligibility 只使用同一 `control_domain_id` 下的本地 control
clock，目标是 local monotonic：

```text
proposal.available_at <= decision.decision_cutoff_at
proposal.effective_from <= decision.target_effective_at < proposal.effective_until
```

模型输入对应的历史时刻由 `observation_cut_id` 及其 sample refs 表达，不用 proposal 的
`produced_at` 猜测。远端 producer 的原始 wall/monotonic time 只作为 clock evidence；真正执行资格由
control 进程的 `received_at/available_at` 决定。

`available_at` 的生成过程固定为 receiver-side admission 线性化点：

```text
ActionProposalDraft
  produced_at / effective interval / producer identity+epoch+sequence / payload
  → receiver validation + semantic-field validation
  → detached deep-freeze
  → ProducerEpochRegistry / sequence check
  → atomic ProposalBuffer admission
  → receiver stamps available_at + admission_sequence
  → immutable ActionProposal + ProposalAdmissionReceipt
```

draft 不包含 authoritative `available_at`；transport receive time 只是证据，不能替代成功 admission。
`ProposalAdmissionReceipt` 明确 accepted/rejected、proposal ID、可空的 receiver `available_at`/admission
sequence 和 typed rejection reason；只有 accepted receipt 拥有执行资格时间/序号。`CompositionContextDraft`、
`AuthorityEventDraft` 与 producer-epoch activation 使用同样规则，
避免远端或 frontend 自报的“已可用”状态进入本地因果选择。

控制域维护 `ProducerEpochRegistry`，记录每个 producer 的 active epoch、activation event ID、`activated_at`、
previous epoch 与 activation reason。Arbiter 不读取可变 registry；每次 decision 将其不可变投影放入
`ArbitrationInputSnapshot`。旧 epoch proposal 保留审计事实，但在新 epoch activation 后不得进入后续候选集。

多个 Store 还必须共享同一个因果 prefix。每个 control domain 持有 `ControlAdmissionCoordinator`；所有会
影响 arbitration 的 proposal、authority/mode event、execution-gate transition 和 producer-epoch activation
在各自 validation/deep-freeze 完成后，通过 coordinator 的同一 commit 临界区写入目标 Store、获得递增
`control_admission_sequence`，最后才原子推进 `committed_high_watermark`。仅“预分配 sequence 后异步写
Store”不算线性化，因为 snapshot 可能看到 watermark 却看不到对应事实。

`DecisionScheduleTick` 在 coordinator 下冻结 `(decision_cutoff_at, cutoff_admission_sequence=
committed_high_watermark)`；各 Store 只返回同时满足 `available_at<=cutoff_at` 和
`control_admission_sequence<=cutoff_admission_sequence` 的不可变记录。这样完整 snapshot 是某个已提交
prefix，而不是分别读取多个 Store 拼出的、从未同时存在过的混合状态。validation 可在临界区外完成；commit
区只负责短时的 store insertion、receiver `available_at`/sequence stamping 和 watermark publish。

v1 的 `capture_cut()` 在同一 coordinator read barrier 内要求各注册 Store 只做内存中的 immutable
state/candidate projection，全部 child snapshot 建成后再释放 barrier；不得在 barrier 内做 policy、hashing、
I/O 或 Arbiter 计算。这样也避免 bounded Store 在 cut 与分次查询之间淘汰所需记录。若实测 admission pause
超出预算，再演进为 versioned/MVCC read lease，但不能退回无 pin 的跨 Store 分次“最新值”读取。

`clock_domain` 不能只是自由字符串。v2.4 Core 同时冻结 `ClockDomainDescriptor`：clock kind、session/boot
identity、resolution、monotonic guarantee、reset epoch，以及可选的 wall/remote mapping evidence、offset
uncertainty 和采样区间。跨进程传输只能比较共享本地 control clock，或使用接收端重新赋予的
`available_at`；clock reset/进程重启必须产生新 epoch，不能延续旧 sequence 假装连续。

当前 v2.3 `CausalSnapshotContract` 已冻结
`source_timestamp<=T0_and_available_timestamp<=T0` 且主要使用 local/receiver wall time，不能原地改 hash。
v2.4 adapter 保留这些字段作为兼容 evidence，并在在线运行可获得时增加 local-monotonic time；历史
v2.3 数据缺少 monotonic evidence 时明确标记 unavailable，不伪造换算。

### 1.1 Canonical encoding、ID 与 hash

Phase 1A 必须与时间 contract 同时冻结 `CanonicalEncodingSpec v1`，不能只在各模块中约定“canonical hash”。
它至少规定：map key/string 编码、enum/tuple/optional/bytes 表示、float 精度与 `-0.0`、NaN/Inf policy、
numpy dtype/shape/endianness、timestamp 单位、四元数是否做符号 canonicalization、hash 算法及 event/schema
domain separator。运行时 proposal/snapshot、dataset record、manifest 和 replay 必须使用同一纯实现或通过
同一组跨层 conformance fixture，禁止 Python `repr()`、对象地址、dict 插入顺序和 Parquet row offset 进入
semantic hash。

stable ID 与 content hash 分工明确：ID 表示一次事实对象的 identity，hash 表示其 canonical content；同一
内容的两次 producer emission 可以有不同 ID，但不能因 serialize/deserialize、transport binding 或输入
map 顺序不同而改变 content hash。含 numpy/mapping 的 frozen dataclass 在 proposal 边界深复制并转为只读
canonical payload；仅 `dataclass(frozen=True)` 不视为深不可变。

v1 的具体默认值已经冻结：唯一实现位于 `dataset_contracts.canonical_encoding`，使用 SHA-256 与显式
schema/domain separator；ndarray 编码 dtype、shape 和 little-endian bytes；NaN 使用固定 bit pattern，Inf
拒绝，`-0.0` 归一为 `+0.0`。generic encoder 不改变 quaternion 符号，几何 canonicalization 仍由
action/pose contract 负责。每类 record 显式投影 canonical payload，不自动 hash 任意 metadata/dataclass。

这里明确允许 `robot_motion → dataset_contracts.canonical_encoding`：`dataset_contracts` 是无 I/O、无 Torch、
无硬件/runtime 反向依赖的底层纯 contract package。`dataset_contracts` 不得导入 `robot_motion`；若未来纯
contract 明显扩展为独立顶层包，再整体迁移，M1A 不为命名问题复制第二份 encoder。

事实 ID 由 `runtime_session_id/control_domain_id/producer_id/producer_epoch/sequence` 组成，和 content hash
分工；可空 `episode_uuid` 只表达 capture membership。
一个 policy chunk 的每个 keyframe 形成一个 proposal，共享 inference/output/chunk lineage。proposal 边界
冻结 detached `RobotAction` snapshot；M1A 不全局改变现有 `RobotAction` caller 的可变性。

例如 RGBD 只有 15 Hz，但存在两种合法的 30 Hz action 生成方式：

1. 30 Hz 快策略每次使用最新 RGBD latent、最新 tactile/state 推理；
2. 15 Hz 推理一次产生两个 30 Hz keyframe，每个 keyframe 记录 `target_effective_at`。

当前 `PolicyDeployRuntime` 的 chunk 只有 chunk ID/index 和 queue schedule；迁移期若 keyframe 时间由
profile 推导，必须记录 `ActionEmissionScheduleSpec`、推导起点和 receipt，不能把 queue pop 时刻当成模型
产生时刻。这两种方式在 trace 中均需区分 `inference_id`、`policy_output_id` 和 `proposal_id`。

### 2. 推荐 profile，而不是全局固定频率

无论当前采用哪种 profile，只要绑定 production decision policy v1，`ActionArbiter` 都必须满足：

```text
0 < arbiter_decision_hz <= 60 Hz
```

这是 Arbiter 自身的发布率上限，不是系统统一时钟，也不限制下列其他 clock。

当前单策略 profile 可以继续是：

```text
RGBD/policy proposal   15 Hz
Arbiter decision       15 Hz
pose reference         60 或 75 Hz（按 profile）
coupled IK solve       当前 real executor 随 reference tick；必须受 capability 验收
joint reference        可与 pose reference/IK 同频；未来可显式插值解耦
command                120 或 150 Hz
```

快慢策略 profile 可以是：

```text
RGBD                    15 Hz
slow vision policy      10–15 Hz
tactile/state           30–100+ Hz
fast action policy      30 Hz
Arbiter decision        30 Hz
pose reference          60 Hz
coupled IK solve        由 action space/execution capability 决定
joint reference         60 Hz
command                 120 Hz
```

如果 human authority 必须以 60 Hz 响应，可以选择：

- normal Arbiter 仍为 30 Hz，takeover/hold/fault 作为 60 Hz 或异步 authority/safety event；或
- Arbiter 为 60 Hz，30 Hz fast proposal 保持两个 tick、15 Hz slow proposal 保持四个 tick。

选择由 profile 和延迟预算决定；v1 不得超过 60 Hz。硬安全 veto 永远不能等待低频 policy tick；它可以
立即使当前 selected action 失效并进入 HOLD/FAULT，但新的普通 selected action 仍只在受限 Arbiter cut
发布。

### 3. 对当前字段的兼容处理

- `PositionControlRates.sample_hz` 暂不破坏性改名；在 position executor 中继续表示输入 keyframe rate。
- 系统层新增明确的 `decision_rate_policy_id/version`、`decision_hz`、per-policy
  `inference_hz/proposal_hz` 和 per-stream sensor rate。
- `decision_hz` 在 production arbitration runtime 的配置校验、实际 decision publication gate 和运行审计
  三处执行 `<=60 Hz`；不能只依赖 frontend CLI 默认值。其他 clock 不复用这个上限。
- execution manifest 另行绑定 `ExecutionCapabilityProfile`：至少声明 validated/configured selected-keyframe、
  coupled-IK、reference-resample、command/backend rate、适用 action space/selection group 及验收证据。
  `validated_ceiling_hz` 与留有余量的 `configured_operating_hz` 必须分开，不能默认把 ceiling 当工作点。
- 进入 `PositionDeployExecutor` 时，把 selected action keyframe stream 的实际 rate 映射为兼容
  `sample_hz`；不要再从 RGBD 或单个 policy rate 推导它。
- 当前 real executor 的 `dual_arm_ik_hz` 仅是 nominal budget；在独立 IK scheduler 真正实现前，manifest
  还必须记录 `actual_ik_call_clock=reference_worker`，不能声称 reference 与 IK 已物理解耦。
- `sim.contracts.TimingSpec.policy_hz` 暂时保留单策略兼容；多策略频率放入新的 PolicyGraph/execution profile
  contract，待 sim frontend 使用后再考虑 schema 升级。
- 所有 scheduler 使用 injected clock 和 deadline；测试应包含非整数频率组合，避免隐含取模逻辑。

---

## 五、Temporal Observation Hub：复用因果历史，不重造全模态同步帧

### 1. 演进方式

不新建另一套 frame buffer。建议把 `TemporalObservationHub` 实现为一个很薄的 registry/coordinator：

```text
TemporalObservationHub
├── arm_state        -> CausalSnapshotHistory
├── hand_telemetry   -> CausalSnapshotHistory
├── tactile          -> CausalSnapshotHistory
├── human_input      -> CausalSnapshotHistory
├── camera streams   -> existing causal frame buffers / immutable frame refs
└── external streams -> CausalSnapshotHistory or adapter
```

`CausalSnapshotHistory` 已经提供 immutable copy、sequence、source timestamp、available timestamp 和
`latest_at_or_before(T0)`，应直接复用。`dataset_contracts.CausalSnapshotContract` v1 已固定 arm、
hand、tactile、camera 的语义；新增通用 stream/cut contract 时应版本化扩展，不要静默改变 v1 hash。

当前 `CausalSnapshotHistory` 会深拷贝 payload，适合低维 state/tactile snapshot，但不应要求高分辨率
RGBD 在每个 policy cut 中重复复制。必须把可持久化记录与运行时 handle 分开，不能让一个
`payload_ref` 同时表示 Python object、共享内存地址和 dataset URI。

### 2. ObservationCutRecord 与 ObservationBundle

每个 policy trigger 先形成可序列化、可 hash 的 `ObservationCutRecord`：

```text
ObservationCutRecord
├── observation_cut_id
├── control_domain_id / policy_node_id
├── cutoff_at / built_at
├── selected_sample_by_stream
│   ├── stream_id / frame_id / sequence
│   ├── measured_at / available_at
│   ├── measured_age_sec / available_age_sec（v2.3 可保留 source alias）
│   ├── valid / stale / missing
│   └── stable payload object ID / content metadata
├── context_dependencies
├── contract/layout hashes
└── trigger_reason
```

本地模型调用使用运行时对象：

```text
ObservationBundle
├── record: ObservationCutRecord
└── payload_handles_by_stream   # 仅当前进程/transport 生命周期有效
```

dataset/ControlTrace 只持久化 record 和稳定 object ID；in-process、shared-memory、RPC adapter 分别构造
自己的 bundle/envelope。RPC 的临时 blob/共享内存 descriptor 属于 transport envelope，不进入 record，
也不得被 writer 序列化。replay 先由稳定 ID 解析 payload，再重建新的 bundle。

各 policy 的 observation builder 决定如何投影和聚合：

- slow vision policy：最新 RGBD、与图像时间对齐的 state、tactile summary；
- fast tactile policy：最新 tactile/state、指定 `CompositionContext`、最近 RGBD 及其 age；
- replay policy：按记录的 observation cut ID 重建，不重新查询“当前最新值”。

同一 RGBD 被 30 Hz policy 连续使用时，`frame_id` 和真实采集时间保持不变，只更新当前 cut 下的 age。
不得复制成不同 frame ID，也不得把没有更新的图像伪装成新帧。

### 3. PolicyObservationView

`DeploymentObservation` 暂时继续作为当前单策略和 v2.3 adapter 的兼容容器；在它与 model wrapper
之间新增 allowlist projection：

```text
ObservationBundle
  → PolicyObservationBuilder
  → PolicyObservationView
  → existing pose-mode preparation
  → policy.predict
```

`PolicyObservationView` 只包含 artifact/layout 声明的：

- image/depth/tactile/state/history；
- per-modality validity、age 和 frame ID；
- 由 `LastActionContract` 明确声明的 last-action stage、producer scope、pose mode 和 anchor；
- 上游 `CompositionContext` 的公开字段和 context ID。

allowlist 不是简单删除 `faults`/`metadata` 两个顶层 key，而是 artifact-bound typed field contract；每个
field path 声明 dtype/shape、required/optional、access role、missing policy 和（若适用）last-action stage。
projection 输出也必须与 runtime payload 隔离，避免模型原地修改共享 numpy/mapping。测试必须向私有
metadata 注入 selected owner、fault、receipt、reward、writer/backend 状态并证明 actor view 无法读取。

默认禁止进入 actor view：

- observation/control faults 的内部原因；
- authority epoch、selected owner、proposal/decision ID；
- safety/IK/planner/dispatch 状态；
- ControlTrace、reward、expert/policy mask；
- writer queue、backend health 和任意 frontend 私有 metadata。

M0 已冻结 `DeploymentObservation.model_input()` 的 legacy 泄漏负面基线；M1A.5 后当前 deploy
`policy.predict()` 不再把该 unrestricted projection 作为事实源，而是只使用 strict `PolicyObservationView`，
object-style compatibility 也由 strict view 重建 sanitized observation。legacy `model_input()` 方法仍存在供兼容
代码使用，因此新增 actor caller 仍必须经过 firewall，不能直接调用它。训练侧 critic 或 reward 若需要
privileged data，应通过独立 training-only view 读取 ControlTrace，不能污染部署 actor contract。

### 4. 分布式时钟规则

- sensor 的 measured/source time 表示数据测量时间；所有对象的 local `available_at` 表示控制进程可安全
  消费的时间。proposal/context 不再借用 sensor `source_timestamp` 一词。
- 不同步的远端时钟不能直接与本地 control clock 比较；transport 必须提供 clock-domain/offset
  evidence，或者只以本地 receive/available time 参与执行 eligibility。
- 远端 policy 返回结果必须引用原 `observation_cut_id`；晚到结果只能从当前时刻起成为候选，不能
  retroactively 改写旧 decision。
- replay 使用记录的 causal IDs 和 logical timestamps，不重新采用 wall-clock “最新值”。

---

## 六、PolicyNodeRuntime、逻辑 PolicyGraph 与部署绑定

### 1. 共同节点接口与现有 `PolicyDeployRuntime`

当前 `PolicyDeployRuntime` 明确拥有 normalize/decode、action chunk queue、lookahead 和 schedule reset，
所以它只适合 action-producing policy，不能作为所有 context node 的通用基类。目标接口应更窄：

```text
PolicyNodeRuntime
  consume(ObservationBundle) -> PolicyNodeOutput

ActionPolicyNodeRuntime
  wraps existing PolicyDeployRuntime
  output = ActionProposal(s)

ContextPolicyNodeRuntime
  output = CompositionContextRecord/Bundle
  no action queue / v2.3 decode / RobotAction conversion

DeterministicComposerNode
  input = context/base/residual
  output = ActionProposal(s) + composition lineage
```

现有 `PolicyDeployRuntime` 继续唯一持有一个 action policy 的 observation pose preparation、prediction
transaction、v2.3 relative→absolute decode、chunk replace/append/pop/lookahead 和 recovery discard。每个
action node 有独立实例/queue；context node 不复制这些行为。所有 node 都不拥有 observation acquisition、
authority、safety、reference、dispatch、writer 或设备 lifecycle。

M2/M3 迁移期固定 `admission_mode=on_emission_pop_v1`：inference 完成时只记录 PolicyOutput 和完整
`ActionEmissionScheduleSpec`，未来 keyframe 仍留在现有 ActionChunkBuffer；唯一一次
`prepare_next_action()` pop 产生 immutable `PolicyActionEmission`，随后才构造 Draft 并 admission。时间语义为：

```text
proposal.produced_at        = 对应 inference/output completed time
proposal.available_at       = 本 keyframe pop 后 receiver admission commit time
proposal.target_effective_at = schedule spec 预先推导的逻辑 keyframe time
```

因此 future keyframe 不会绕过 legacy queue 提前参与 authority，shadow 也不二次 pop。proposal 是否 selected
不反向控制 policy queue：shadow、human takeover 或 proposal rejection 时 stream 仍按其 schedule 前进，被 pop
但未 selected 的 proposal 保留 trace，恢复 authority 时不能倒退使用旧 keyframe。未来若需要 inference 完成
时整 chunk admission，必须发布新的 producer schedule/admission mode、buffer/replay contract 和 parity fixture，
不能静默改变 v1。

### 2. 逻辑 PolicyNodeSpec / PolicyGraphSpec

逻辑 contract 只描述算法语义：

```text
PolicyNodeSpec
├── policy_node_id / output_kind
├── artifact/layout/frame/context schema hashes
├── input ObservationContract
├── trigger_kind / inference_hz / proposal_hz
├── deadline budget / result validity policy
├── context dependencies
└── action component coverage（仅 action/composer node）

PolicyGraphSpec
├── nodes
├── versioned CompositionContextEdgeSpec
├── deterministic composer edges
└── graph validation/hash
```

`inference_hz` 和 `proposal_hz` 可以不同；每个 action keyframe 使用 `target_effective_at`。artifact 仍使用
per-policy `PolicyIOLayout`，不能把所有模型塞进一个全局 layout。`AuthorityPolicySpec` 属于 control
configuration，不属于 policy graph；graph 只发布结果，不能决定最终 owner。

### 3. PolicyOrchestrator 与 RuntimeSupervisor 的边界

使用 `PolicyOrchestrator` 名称避免与全局监督混淆。它负责：

- 验证/实例化受限 policy DAG；
- inference trigger、deadline 和 context 路由；
- policy timeout/late-result 检测；
- 发布 context/proposal/health event。

`RuntimeSupervisor` 负责 sensor/backend/writer/process health、restart/producer epoch、mode transition 和
degradation request。两者都不能直接 fallback 或 dispatch。fast policy timeout 后只能发布 health event、
mode-transition request 或 fallback-producer-enable event；实际 owner 变化必须由
`ControlMode + AuthorityPolicySpec + ActionArbiter` 在后续 decision cut 中确定。

M2–M6 继续使用现有单个 `PolicyDeployRuntime` + 窄 `arbiter_shadow` wiring，不提前引入 Orchestrator。到 M7C
production multi-policy shadow 阶段，`run_policy.py` 才演进为注入 `PolicyOrchestrator`：先注册一个 action
node 并证明 timing/queue/decode/pause-reset parity，再接 context/composer node 和真实 fast/slow topology。

### 4. `CompositionContextRecord` 与 runtime payload

慢策略可以输出 task/skill intent、subgoal/waypoint、contact phase、constraint、trajectory plan 或 latent，
但持久化 record 与运行时 payload 同样必须分开：

```text
CompositionContextRecord
├── context_id / producer_id / producer_epoch / sequence
├── optional producer_node_id / observation_cut_id / policy identity
├── produced_at / available_at / effective_from / effective_until
├── context_channel / context_kind / availability / fault
├── payload_contract_hash / optional producer_artifact_hash
├── source fact/hash / parent context lineage
└── immutable semantic payload 或 stable blob refs

CompositionContext transport envelope
├── record
└── runtime tensor/blob handles
```

`context_kind`是开放的版本化名称；`artifact_bound_latent`必须绑定producer artifact、payload contract及允许的
consumer contract，不能作为跨模型/跨版本的通用任务事实。快策略的cut/input必须引用context ID；乱序旧版本
不得覆盖新版本。policy字段只是可选lineage，human/replay/task/state/vision producer不需要伪造policy identity。

只有可执行 control action 进入 `ActionProposal`。slow base + fast residual 由显式、版本化
`DeterministicComposerNode` 在 Arbiter 前完成，并记录 base/residual/composed lineage；Arbiter 第一版不做
pose blend。

Composer 自身也必须是 snapshot-driven 的纯节点：

```text
base/residual/context draft
  → receiver-side internal admission
  → immutable CompositionInputRecord(authority_candidate=false)
  → CompositionInputStore
  → CompositionInputSnapshot

CompositionInputSnapshot
  base/context/residual IDs + content hashes + effective intervals
  internal admission sequence/cutoff
  frame/component schema hashes + CompositionSpec hash
  → DeterministicComposerNode
  → CompositeActionProposalDraft
  → normal receiver admission
```

Composer 不消费未 admission 的临时 Python 对象。internal input 复用 control-domain admission coordinator、
stable ID/content hash 和 receiver `available_at`，但默认不写入 global ProposalBuffer，也不能被 Arbiter 单独
选择；只有成功 admission 的 CompositeActionProposal 是 authority candidate。若某个上游输出确实同时需要
独立 authority 和 composition input，必须显式发布两个有 lineage 的 typed fact，不能因共享对象引用而隐式
双重候选。

第一版只支持可证明确定性的 component union、base + bounded residual、fixed semantic transform 和显式
SE(3) composition。base/context 缺失、residual 缺失、frame 不兼容和 late input 的行为由
`CompositionSpec` 明确为 invalid/no-action/zero-residual 等结果；snapshot 后晚到输入只能进入下一次 composition。
近期不实现 confidence blend、freshness 自动权重、latency-driven 连续调权、Arbiter 内 pose averaging 或未版本化
renormalization。未来新增比例混合时，由具体 composition algebra 定义 source role、temporal validity、fallback
和 weight policy，系统不强制所有 algebra 的权重和为 1。

### 5. DeploymentBindingSpec

placement/transport 不进入 `PolicyNodeSpec` 或其 semantic hash，而由独立部署 manifest 描述：

```text
DeploymentBindingSpec
├── policy_node_id
├── process/machine placement
├── transport kind / endpoint
├── queue/QoS
└── timeout/reconnect policy
```

同一个 PolicyGraph 可映射为单进程、单机多进程或多机；更换 in-process/shared-memory/RPC 只改变
deployment hash，不改变 policy semantic hash。第一阶段只实现 in-process binding，核心 contract 不依赖
Python 对象地址、共享内存名或具体 RPC 框架。

---

## 七、ActionProposal、ProposalBuffer 与 ActionArbiter

### 1. ActionProposal

proposal 必须在 absolute runtime `RobotAction` 层生成：

```text
ActionProposal
├── proposal_id
├── control_domain_id / embodiment_instance_id
├── producer_kind: human / policy / replay / scripted / recovery
├── producer_id / policy_node_id / producer_epoch
├── action: immutable absolute RobotAction
├── produced_at / available_at
├── control_admission_sequence
├── effective_from / effective_until / target_effective_at
├── sequence
├── observation_cut_id
├── policy_output_id / inference_id
├── context_dependency_ids
├── representation_trace_id
├── component_schema_hash / component_coverage
├── selection_group requirements
├── confidence                 # 第一版只记录，不用于隐式抢权
├── validity/status
├── semantic_content_hash
└── audit_metadata             # 不参与任何控制判断
```

`RobotAction` 虽为 frozen dataclass，但内部 numpy/mapping 仍需在 proposal 边界复制并冻结。raw relative
vector、anchor、normalization、frame conversion 写入 `ActionRepresentationTrace`，不交给 Arbiter 计算。
proposal 不再使用 sensor 风格的 `source_timestamp`；输入历史完全由 `observation_cut_id` 表达。

proposal 的有效期是硬边界；旧 action 只能在 `effective_until` 前继续使用。queue empty、timeout 和 stale
都必须显式产生状态，不能无限 hold 最近 proposal。

proposal 的 priority、authority request、validity、fallback、component coverage、frame/action space、selection
group、timing/context 和 safety 语义必须是 typed field 并进入 `semantic_content_hash`，禁止藏入
`audit_metadata`。Arbiter、Composer、safety 和 persistence 不得读取 audit metadata；目标持久化 record 另用
storage envelope 的 `record_integrity_hash` 防篡改。该 hash 由 M4B/M4C 的 dataset/trace encoder 在加入
episode membership、journal sequence、schema registry refs 等持久化字段后计算，不放在 runtime
`ActionProposal` 本体中，也不与控制 semantic hash 混用。当前 M4A 同名 logical envelope 只用于内存
conformance，不代表 storage encoder 已实现。

现有 `RobotAction.metadata` 不能在 M1A 被粗暴改名或全部排除：当前 resolver 会读取其中的
`arm_action_specs_by_side`/`hand_action_specs_by_side`。proposal admission 必须对这些已知 execution-relevant key
做 typed allowlist/canonical projection 并纳入 semantic hash，未知且会影响 resolver 的字段 fail closed；待
兼容 caller 迁移后再提升为显式 typed action spec。纯 debug/source note 才进入 audit metadata。

#### 1.1 Admission contract

producer 先提交 `ActionProposalDraft`，receiver 仅在完整 validation/deep-freeze/epoch-sequence 检查成功后
原子写入 buffer，并返回 `ProposalAdmissionReceipt`。rejected draft 产生 admission fact，但绝不能获得可执行
`available_at` 或进入 snapshot。buffer capacity failure、duplicate、sequence rollback、inactive epoch、schema/
frame mismatch 和 non-finite payload 都使用 typed rejection reason；不能静默 drop 或靠日志文本解释。

### 2. ProposalBuffer

`ProposalBuffer` 是 bounded、multi-producer、只保存 immutable proposal 的因果缓冲区：

- 按 producer ID/sequence 拒绝 rollback 和重复发布；
- 在 `decision_cutoff_at` 获取“截至此时已 available 的 proposal snapshot”；
- snapshot 后 producer 的写入不能改变本次 arbitration；
- expired/invalid proposal 仍可产生 trace event，但不进入有效候选集；
- buffer 不运行 policy、不做 authority、不做 safety。

每次提交 Arbiter 的对象必须是可重放的 `ProposalSnapshot`：

```text
ProposalSnapshot
├── proposal_snapshot_id / control_domain_id
├── snapshot_cutoff_at / snapshot_created_at
├── control_cutoff_admission_sequence / included sequence range
├── reduction_policy_id/hash
├── candidate proposal IDs + proposal content hashes
├── producer_epoch_summary
└── canonical snapshot hash
```

snapshot hash 同时覆盖 `proposal_id` 与对应 `proposal_content_hash`，防止 ID 错误复用或持久化内容损坏只靠
ID 列表无法发现。

`ProposalSnapshot` 只冻结候选集，不足以单独决定 authority。每个 scheduler tick 必须继续冻结完整输入：

```text
ArbitrationInputSnapshot
├── arbitration_input_snapshot_id / control_domain_id / embodiment_instance_id
├── control_cutoff_admission_sequence
├── proposal_snapshot_id / hash
├── AuthorityStateSnapshot
│   └── policy hash / mode / leases / causal event IDs / authority_epoch
├── ExecutionGateSnapshot
│   └── status / gate_epoch / reason / source event IDs
├── ProducerEpochRegistrySnapshot
├── previous_authoritative_decision_id / previous_selection_by_group
├── active selection groups / action+component schema hashes
├── DecisionScheduleTick
├── per-store included admission sequence ranges
└── canonical input hash
```

pure Arbiter 的唯一逻辑输入是该对象：

```text
ActionArbiter.arbitrate(snapshot: ArbitrationInputSnapshot) -> ArbitrationResult
```

不得在调用时另外传入 `frontend.mode`、`runtime.fault` 或 `authority_store.current()`。snapshot 后状态变化不
修改本次纯计算结果；若 gate/fault/execution epoch 在 publication 前变化，publication/dispatch gate 必须拒绝
旧 result 并触发 cancellation，下一 tick 再反映新状态。

decision scheduler 使用版本化 `DecisionScheduleSpec`：

```text
DecisionScheduleSpec
  schedule_mode: fixed / event_limited
  decision_period / cutoff_policy / target_lead_time
  publication_deadline_offset / max_target_lateness
  late_decision_policy / missed_tick_policy / trigger_coalescing_policy

DecisionScheduleTick
  schedule_tick_id / scheduled_decision_at / decision_cutoff_at
  cutoff_admission_sequence
  target_effective_at / publication_deadline_at
  trigger IDs+reasons / schedule_spec_hash
```

`decided_at` 表示 pure calculation 完成，`published_at` 表示 authoritative result 成功通过 publication gate 并
交给执行链。二者与 scheduler 晚醒、下游 IK/reference deadline 分开审计，不能共用一个 timestamp。

第一版 candidate reduction 必须以本次 `target_effective_at` 为中心，而不是笼统选择“最后发布的一条”：

```text
eligible =
  proposal.available_at <= snapshot_cutoff_at
  and proposal.control_admission_sequence <= control_cutoff_admission_sequence
  and proposal.effective_from <= target_effective_at < proposal.effective_until
  and proposal.producer_epoch == active_producer_epoch

per-producer/group winner ordering =
  max(effective_from)
  then max(sequence)
  then lexical-min(proposal_id) deterministic tie-break
```

effective interval 固定为半开区间 `[effective_from, effective_until)`，相邻 keyframe 在边界处不能同时
eligible。chunk keyframe interval 由版本化 `ActionEmissionScheduleSpec` 的推导起点、keyframe cadence、
artifact horizon/maximum hold 或显式 validity duration 给出；最后一个 interval 不能由实际 queue pop 时间
临时决定。

同一 producer/group 的普通 keyframe effective interval 默认不得重叠；确需重叠时必须由版本化 overlap
policy 明确选择，不能让较大的 chunk index 或较晚 arrival 隐式抢占当前 target。所有 raw proposal 继续留在
trace；snapshot 只冻结实际提交 Arbiter 的 reduced candidate set。replay 直接校验 snapshot hash，不依赖
未来版本的 buffer 实现重新推断候选集。测试覆盖同一 chunk 的多个未来 keyframe、replace/append chunk、
late chunk、overlapping interval、相同 cutoff 和无序输入。

decision runtime 还需声明 `max_target_lateness` 与 `late_decision_policy`。若 `decided_at-target_effective_at`
超过预算，不能把已经过期的目标照常发布；v1 由 profile 显式选择 `no_action` 或 `hold`，并记录 deadline
miss。candidate reduction 产生的 `superseded_before_snapshot`、decision gate 合并导致的
`coalesced_before_decision`、以及没有任何合法 decision target 覆盖的 `effective_interval_missed` 是 proposal
disposition/审计原因，不得改写原始 proposal validity，也不能统称为 stale。

固定频率 Arbiter 第一版直接在每个 decision tick 获取 snapshot，配置和实际 decision publication 均不得
超过 60 Hz。事件触发加 rate limit 可以作为后续模式，但事件 burst 必须 coalesce 到同一个受限 decision
gate；contract 从一开始保留 trigger reason、cutoff、decided 和 target-effective time。

### 3. ActionArbiter

`ActionArbiter` 的纯选择函数本身没有 wall-clock 频率概念；production frontend 不得因此直接无限频调用并
发布 decision。共享 `ArbitrationRuntime`/decision gate 必须是唯一生产调用入口，并同时保证：

- configured `decision_hz <= 60 Hz`，大于 60 的配置 fail closed；
- 相邻普通 `ArbitrationDecision` 的发布时间满足 60 Hz 最小周期，事件 burst 合并后只消费最新合法
  `ProposalSnapshot`；
- 记录 configured/effective/measured decision rate 和被 coalesce 的 trigger 数；
- safety veto、FAULT、HOLD 可立即生效，但只产生 safety/mode event，不伪造一个超频 selected action；
- 该 gate 不限制 proposal、reference、IK、command 或 backend tick。

production decision policy v1 支持最高 60 Hz 的 authoritative learned-action 更新。100/500 Hz producer
表示更高频候选生成，不保证每个 proposal 成为 selected action；普通 learned correction 若确实要求
超过 60 Hz，必须发布新的 decision-rate policy，或定义一条单独受安全/authority/trace 约束的下游局部
controller contract，不能偷偷塞进 command worker。另一种兼容方式是由不超过 60 Hz 的 policy 输出带显式
horizon 的高频 trajectory/reference，由现有执行层展开。

rate gate 只证明“没有超频”，不证明 deadline 可满足。execution manifest 还需记录
`decision_runtime_budget`、`ik_runtime_budget`、`end_to_end_control_budget` 和 `late_decision_policy`；synthetic
测试覆盖 deadline/miss 语义，真实 profile 在获得明确硬件测试授权后验收 proposal available→snapshot→
arbitration→shared safety→IK/planner→reference publish→first command send 的 p50/p95/p99 与 miss count。

Arbiter 只负责 authority 与确定性选择：

```text
ArbitrationResult
├── computation_id / result_id / control_domain_id / embodiment_instance_id
├── arbitration_input_snapshot_id / input_hash / proposal_snapshot_id
├── arbiter_contract_id/hash / arbiter_implementation_hash
├── authority_policy_hash / component_selection_policy_hash
├── decision_schedule_spec_hash / canonical_encoding_version
├── schedule_tick_id / decision_cutoff_at / decided_at / target_effective_at
├── authority_epoch / control_mode / execution_gate_status
├── selected_by_group
├── selected_by_component
├── proposal_id/source_id by component
├── composed_action: RobotAction | None
├── rejected/expired/incompatible decisions
├── output_status: selected / no_action / hold / fault
└── attribution
```

它不负责 sensor polling、policy inference、relative decode、IK、workspace/collision safety、trajectory、
dispatch、reward 或 writer lifecycle。

选择规则不能埋在 frontend priority 常量中。M1B 同时冻结最小 `AuthorityPolicySpec`、`AuthorityEvent` 和
`AuthorityLease` contract，明确 source/group priority、takeover/release、lease expiry、human proposal 失效时
hold/fallback、recovery producer 权限和 shadow 固定 owner。完整多策略 policy 可以后置，但 M5 human shadow
不得再发明一套硬编码 authority 规则。

每次 arbitration 按固定顺序执行：

1. 验证 action space、frame、shape、finite、component coverage、sequence 和时间字段；
2. 以 available/effective interval 检查因果性和 freshness；
3. 应用外部 `ExecutionGateStatus`/readiness veto；
4. 按 `ControlMode` 和显式 authority lease/takeover 过滤候选；
5. 按 selection group、priority、sequence 和 timestamp 做确定性选择；
6. 检查 coupled group 完整性与 component compatibility；
7. 输出 control port 与 attribution port。

第一版不根据 confidence、动作幅度或“人是否移动”隐式抢权，不做 raw pose blend。handoff 平滑由
selected action 以下的 reference/trajectory 层完成。

#### 3.1 Action persistence 与 publication

Arbiter 的 `selected/no_action/hold/fault` 不能直接交给各 backend 自行解释。共享纯转换固定为：

```text
ArbitrationResult
  + ActiveExecutionSnapshot
  + ActionPersistencePolicy
  → ExecutionDirective

ExecutionDirective:
  global_status: normal / hold_all / fault_all
  directive_by_selection_group:
    group_id
    publish_selected / keep_existing_until_expiry / cancel_group_and_hold /
    keep_backend_hold
    selected_action_ref / existing_segment_id / segment_expiry
    group_segment_generation
  persistence_policy_hash
  publication_fallback: no_action / hold / null
```

`ActionPersistencePolicy` 至少声明 selected segment maximum lifetime、no-action、proposal starvation、segment
expiry、hold/fault 和 backend-hold behavior，均能按 selection group 配置。v1 语义是：selected 为对应 group
建立带 expiry 的新 segment；普通 no-action 只让该 group 的旧 segment 运行到原 expiry，不阻止其他独立
group 发布；coupled group 始终原子处理。全局 safety HOLD/FAULT 覆盖全部 group，递增 domain
`execution_epoch` 并取消旧 command；局部 cancel 递增该 group 的 `segment_generation`。若 backend/
ExecutionCapabilityProfile 不支持某 group 独立停止，局部 cancel 必须显式升级为 coupled-group 或 hold-all，
不能假装已局部生效。fault 额外 latch，只有显式 recovery 才进入可执行新 epoch。

authoritative publication barrier 校验 live gate、deadline/rate 和 target-specific candidate authorization；最终
motion/safety permit 在 backend 调用前校验 dispatch-owner generation、authority epoch 和 execution epoch。这些
检查不会回写或改变已经生成的纯 `ArbitrationResult`。只有 accepted selected result 产生
`ArbitrationDecisionEvent` 并可激活 `publish_selected`；`no_action/hold/fault` 或 rejected-selected 的显式
publication fallback 仍产生不含 selected action 的 `ExecutionDirective`，允许旧 segment 保持至原 expiry，或
触发 cancellation/backend hold。此类 directive 的 `decision_id=null`，必须引用 publication attempt，不能进入
`ActualDecisionView`。已经处于 backend hold 时使用 `keep_backend_hold`，不得每个 decision tick 重复递增 epoch。

### 4. ControlMode、ExecutionGateStatus 与三层 safety

建议保留两套正交状态：

```text
ControlMode:
  HUMAN_ONLY / POLICY_SHADOW / HUMAN_OVERRIDE /
  PER_GROUP_OVERRIDE / POLICY_ONLY

ExecutionGateStatus:
  NOT_READY / READY / HOLD / FAULT / RECOVERING
```

`ExecutionGateStatus` 表示是否允许选择/继续执行，来自现有 `TeleopSession`、deploy lifecycle、backend
health 或 safety/reflex guard；它与 Arbiter 后可能修改动作的 `SharedCanonicalSafetyResult` 不是同一层。
普通
proposal 不能清除 fault。显式 human takeover 可以更新 authority epoch；若要求接管但 human proposal
已过期，默认 HOLD，不静默回退 policy。

迁移时不能把当前 policy `ActionSafetyFilter` 原样套到所有 human action。现有代码中 human 路径已有
`SafetyMonitor`、`command_filter/command_guard`，policy 路径有 `ActionSafetyFilter`，position 层另有
keyframe/joint/collision constraints，backend 还会返回 SDK receipt。目标语义明确分为：

```text
SourceValidationResult
  source-specific frame/shape/finite/representation/coverage validation

SharedCanonicalSafetyResult
  Arbiter 后唯一的 workspace/joint/collision/force constraint modification/veto

BackendCommandGuardResult
  SDK ready、servo/fault、command range、transport reject/hold
```

source validation 可以按 human/policy/replay 使用不同 adapter，但不改变 authority；shared canonical
safety 只有一套；backend guard 只能 accept/reject/hold，不能产生另一套 action。三层结果和修改原因都写入
lineage。近期迁移先把现有 filter 分类、包成 typed result 并做 parity，不提前合并语义不同的 guard。

shared safety 还必须把四类常被混称为“anchor”的语义分开：representation/decode anchor、artifact-specific
model last-action、canonical keyframe safety anchor，以及 planner/reference segment anchor。目标 keyframe safety
的正常 continuity 使用 per-selection-group `previous_safe_committed`；startup、HOLD/FAULT recovery 和 execution
epoch 变化后必须用 fresh measured state 显式 rebase。`previous_selected` 只用于 intent audit/独立 limiter，
`previous_commanded` 只用于 backend slew 或 legacy-aligned comparison，不能作为跨 action/reference/command schema
的全局 anchor。

`committed` 不是 safety 计算完成，而是 safe action 经显式 resolved execution target admission，或 direct backend
command receipt accepted 后的事实。shadow、publication/queue/planner/backend rejection 不更新 authoritative
anchor。anchor state 的 scope 是 control domain + selection group + execution epoch + segment generation，而不是
producer，因此 compatible source handoff 可保持连续性；hold/fault/group cancel/epoch change 则产生 typed
invalidation 并要求 measured rebase。正常 previous-safe continuity 之外，fresh measured-state divergence 是独立
guard：它限制 target 与真实状态漂移，但不改变 actor `LastActionContract`，也不把 noisy/stale measured state变成
每 tick 唯一 continuity anchor。

当前代码并不统一遵守上述目标语义：policy pose 优先 deploy `last_action`、qpos 每 tick 使用 measured qpos；
direct dry-run 和 position queue admission 都可能更新 deploy `last_action`；human temporary limiter 在 guard/dispatch
前推进内部 previous command；只有 scoped authoritative sim 在 permit 与全部 receipt accepted 后才建立 active
segment。因此迁移必须同时记录 `LEGACY_ALIGNED`（复现旧 anchor，验证 numeric primitive）和
`COUNTERFACTUAL_SHARED`（独立 shadow anchor state，预测 owner 切换后行为），不能把 anchor-policy difference
误判为 primitive mismatch。

触觉用途必须区分：

- learned tactile correction：30 Hz+ `ActionProposal` producer；
- over-force/collision/lost-communication：高频 safety veto/modification，不参与普通 priority 竞争。

异步 safety/HOLD/FAULT 不能只写一条 event；它必须通过正式 control seam 取消已进入 reference/command
worker 的旧执行段：

```text
ExecutionCancellation
├── cancellation_id / control_domain_id / execution_epoch
├── invalidated decision/reference segment IDs
├── effective_at / reason / latch policy
└── authority_epoch evidence

ExecutionCancellationReceipt
├── software_queue_ack
├── sdk_request_ack
├── backend_state_ack（若 backend 可提供）
└── measured_motion_stopped_ack（若测量可判定）
```

`authority_epoch` 与 `execution_epoch` 分开：owner 切换不必等同于 fault cancellation，fault cancellation 也
不一定改变 owner。恢复产生新的 execution epoch；旧 reference segment、queued command 和 lookahead 因
epoch 不匹配自动失效，不能只依赖某个 frontend 尽力调用 `clear()`。

取消的线性化点固定为 control domain 内 `execution_epoch` 的原子递增。`ReferenceSegment`、
`PositionCommandTick` 和 backend command 都携带 epoch；worker 至少在 dequeue 和真正调用 backend/SDK 前
各检查一次，不匹配则拒绝发送。顺序是先递增 epoch，再发布 cancellation、停止旧 reference/command、
请求 backend hold 并收集分层 acknowledgement。SDK 调用成功只表示 request ack，不能冒充机械臂已停止。
profile 声明 `cancellation_ack_deadline` 和必需 ack level；超时升级为 FAULT。已经发送的旧命令仍保留原
receipt，不宣称可被撤回。

### 5. Component catalog 与 selection group

`ActionComponentCatalog` 应版本化描述 `RobotAction`、PolicyIO 和 dataset 中的稳定字段，例如：

```text
left_arm.pose
right_arm.pose
left_hand.qpos[0..N]
right_hand.target
```

Arbiter 的默认 authority 粒度是 effector 或 coupled selection group；trace 可以保存到 field/slot。
四元数 scalar、双臂 coupled IK 等不能被错误拆成独立 owner。

每个 decision 同时生成：

- available mask；
- selected owner/mask；
- rejected/expired/invalid reason；
- proposal ID 和 context dependency；
- component/group compatibility 结果。

`ActionComponentCatalog` 只描述 proposal/selected canonical `RobotAction`。selected 以下分别绑定
`ReferenceSchema`、`CommandSchema` 和 `MeasurementSchema`：

```text
selected Cartesian component(s)
  → reference pose/qpos component(s)
  → actuator command component(s)
  → measured joint/TCP component(s)
```

每次转换记录 transform contract hash、input/output component set 和 dependency proposal IDs。双臂 coupled
IK 允许多对多依赖；command 的 `primary_authority_owner` 只是审计摘要，不能替代完整 dependency set。

---

## 八、如何接入现有 canonical action 与 position 链路

### 1. 目标边界

```text
source-specific representation
  → source validation + existing decode/frame/anchor adapter
  → absolute RobotAction proposal
  → ActionProposal / ActionArbiter
  → selected RobotAction
  → shared canonical safety
  → planner / resolver / reference / IK
  → TeleopCommand
  → command schema / backend command guard
  → logical dispatch port / receipt / measurement
```

#### 1.1 最终 dispatch ownership gate

当前 `TeleopSession.dispatch_command()` 是 human/recovery 共用 seam，`PositionDeployExecutor` 另有 generation、
dispatch lock 和 callback 前检查；这些能阻止各自 executor 内的旧任务，却不能证明 legacy frontend 与未来
shared arbitration runtime 不会同时调用最终 backend。因此 M6 切换前增加 control-domain 级硬互斥：

```text
DispatchOwnerToken
  token_id / control_domain_id / owner_runtime_id
  owner_kind: legacy_control_path | shared_arbitration_runtime
  owner_generation / acquired_at / revoked_at

DispatchPermit
  kind: MOTION | SAFETY_CONTROL | LIFECYCLE_CONTROL
  token_id + owner_generation
  authority_epoch + execution_epoch
  command/directive lineage ID
```

token 只表示哪个 runtime 拥有 logical dispatch port；human/policy/replay 仍是 proposal/authority source，不
创建各自 token。`MOTION` 用于 pose/qpos/hand/trajectory target，最终 backend seam 在每次实际 SDK/sim call
前校验 current token generation、control domain、current authority/execution epoch 和 command permit，失败产生
typed reject receipt，不能静默 drop。

`SAFETY_CONTROL` 用于 emergency stop、hold、disable motion、collision stop 和 cancellation request，不依赖
普通 motion owner token，但必须引用已 admission 的 safety/cancellation event、目标 execution epoch、授权的
safety source 和 control domain；它不是通用 bypass。`LIFECYCLE_CONTROL` 用于 startup/home/servo enable/
rearm，只能由显式 lifecycle state machine 和 operator/safety policy 授权，普通 human/policy/replay authority
不能构造。三类 permit 使用不同 capability/validation path 和 typed receipt。

owner 切换固定为 gate→HOLD/TRANSITIONING、execution epoch 递增并取消旧队列、revoke old token、等待 profile
要求的 cancellation acknowledgement、原子 grant new token、更新 authority state、再允许新 segment。短暂
无 owner/HOLD 合法，双 token 同时有效不合法。第一版 registry 是同进程原子对象；跨进程/多机 owner lease
必须等 DeploymentBinding 实现共享一致性机制，不能把进程内 token 当作分布式锁。

### 2. 当前 policy 路径的必要小重构

当前 `CanonicalDeployActionPipeline.prepare()` 是：

```text
NormalizedAction
  → ActionSafetyFilter
  → robot_action_from_vector_action
  → RobotActionResolver
```

这与 proposal 需要在 pre-safety absolute `RobotAction` 层生成存在顺序差异。最佳迁移方式不是复制一套
safety，而是拆出并复用现有转换：

1. 从 `CanonicalDeployActionPipeline` 提取纯 `DeployRobotActionAdapter` 和 `SourceValidationResult`，内部
   继续调用现有 `robot_action_from_vector_action()` 和同一 layout/spec；
2. `PolicyDeployRuntime` 的 decoded `NormalizedAction` 先经 adapter 生成 absolute `RobotAction` 和
   representation receipt，再形成 proposal；
3. 为 canonical pipeline 增加 `prepare_robot_action()`/typed safety seam；policy parity 期可通过现有
   codec 把 selected `RobotAction` 投影到兼容 filter 输入，但只允许一次 safety pass，且不宣称当前
   policy filter 已经等价于 human shared safety；
4. 现有 `prepare(NormalizedAction, ...)` 保留为兼容 wrapper，并委托新内部步骤，保证当前
   `run_policy.py` 行为和测试不变；
5. Arbiter observe-only 稳定后，才把 selected `RobotAction` 接入该 typed seam。

不得在 proposal 前过滤一次、selected 后再过滤一次，也不得让不同 policy node 各自拥有 safety。

### 3. human 路径

human retargeter 的 `TeleopCommand` 暂不一次改写。增加窄 adapter：

```text
TeleopCommand before merge
  → RobotAction snapshot
  → HumanActionProposal
```

adapter 支持 cartesian pose、arm qpos、hand qpos 和 native end-effector target；保留 reference frame、
controlled frame、hand mask 和 human `delta_clutch` anchor provenance。未知 command type 产生 invalid
proposal 并 fail closed。每次 mapping 产生 `SourceAdapterReceipt`，按 component 标记 `lossless`、
`lossy_but_explicit` 或 `unsupported`；默认补 frame、丢弃字段、猜 backend mode 或无法恢复 partial hand mask
都不能静默成为 authoritative proposal。

第一阶段只 observe-only：现有 `TeleopSession.command_filter/guard/fault/hold/dispatch` 完全不变；记录
proposal/decision 与实际 command 的对照。切换 owner 后，human/policy/replay 都通过同一 selected
`RobotAction` safety/resolver seam，旧 merge 不再承担 authority。

### 4. replay 路径：从现有兼容实现收敛为 proposal source

当前已有两套可复用能力，但尚未形成统一 owner：

- `teleop_runtime.deploy.replay.DatasetReplayPolicy` 把 dataset action 伪装成 policy output，复用
  `PolicyDeployRuntime` queue/decode；
- `teleop_runtime.replay`/`robot_motion.replay` 已有 dataset preflight、轨迹分组、position resampling 和真机
  replay executor。

它们当前都不能直接被称为最终 replay architecture。目标是保留 reader/preflight/resampling 数学，将
replay action 在 dispatch 前收敛为 `producer_kind=replay` 的 `ActionProposal`，经过相同
AuthorityPolicySpec、snapshot、Arbiter、shared safety、execution cancellation 和 ControlTrace。现有 frontend
先作为 compatibility owner 保留，完成 shadow parity 后才迁移，不因本规划直接归档。

replay 分成四种明确能力，不能混名：

```text
decision replay     proposals/snapshot → pure Arbiter，验证相同 decision
policy replay       recorded ObservationCut/View → artifact inference，比较新旧 proposal
action replay       recorded canonical proposal/selected/safe stage → replay proposal → normal control
command audit       读取 backend command/receipt 作审计；默认不得跨 embodiment 再发送
```

`dataset_interface.replay` 只读取 sealed facts/materialized view，`robot_motion.replay` 持有无 I/O 的 schedule/
canonical replay primitive，`teleop_runtime.action.replay_proposal` 负责 real frontend wiring。replay 必须显式
选择 source action stage、resampling policy、pose/frame/action schema 和 target embodiment compatibility；禁止
把旧 qpos/servo packet 当成跨本体可移植的 canonical action。真机 action replay 仍是高风险控制行为，必须
单独 preflight、安全授权和互斥 authority owner；“数据可 replay”不等于自动允许运动。

若 replay source 选择历史 `safe` stage，它只表示来源动作曾通过旧 safety profile；转换出的 replay proposal
仍必须经过目标运行时当前 shared safety，不能携带 bypass。新 safe result 可以因 embodiment、calibration、
state 或 safety profile 不同而变化，并同时记录 old-safe provenance 与 new safety lineage。

当前落地边界（2026-07-19）：上述 contract 已在 typed dry-run 与默认关闭 canonical GUI passive shadow 层实现。debug `DatasetReplayPolicy` 不再在
shadow facts 中冒充普通 policy source，而是从既有唯一 queue-pop emission 被动生成内容寻址的
`ReplayActionRecord`、`ReplaySourceAdapterReceipt` 和 `producer_kind=replay` proposal；source/local frame
index、source stage、capture-relative time、runtime effective interval、frame/component binding 均显式保留。
`replay_shadow_dry_run_v1` 已能从 replay record 追到 proposal/snapshot/evaluation/legacy execution，并用同一
frozen snapshot 重放 pure Arbiter。`robot_motion.replay` 的既有 trajectory action 已有无 I/O parity adapter，
`ReplayExecutionController` 已可选向 bounded nonblocking queue 发布该 immutable source-frame emission：
impedance 在 action tick 发布，position 在 causal left source keyframe 到期时发布，不把 offline interpolation
使用的未来 right lookahead 提前暴露；qpos/SDK packet 仍不进入 canonical proposal。默认无 queue 时不构造
emission，queue full/consumer error 不改变 init、resampling、dispatch 或 completion。source-relative time
作为 capture provenance 原样保留；position effective interval 跟随 source-time resampling，而 impedance
effective interval 跟随原有固定 `action_hz` tick，带抖动的 capture frame time 不改变 legacy impedance
schedule。canonical GUI 现可显式启用 bounded consumer：成功 preflight 后绑定 source context，第一次同一
immutable source-frame emission 才惰性创建 replay-specific Arbiter shadow、RuntimeTraceSession 和可选 debug
WAL；它不申请 DispatchOwnerToken，也不调用 backend。target frame/capability evidence 不充分时只产生带完整
assessment lineage 的 conditional evaluation；source frame/component contract 不充分时只记录 rejection。由于
tap 位于 legacy dispatch 前，当前不能宣称已有 post-dispatch safe/command/receipt 或 formal episode capture，
也不得把这一状态称为真机 replay authority complete。

#### replay compatibility 不是一个布尔值

M5.3a 已移除 typed core 中曾经混合多种语义的 `ReplayTargetBinding.compatible: bool`。canonical GUI 与未来
authoritative 路径统一使用四层事实：

```text
source interpretation
  → ReplaySourceActionContract / LegacyReplayContractDerivationReceipt
target relation
  → ReplayTargetBindingSpec + ReplayBindingAssessment
candidate use
  → CandidateBindingEligibility
execution authority
  → ReplayExecutionAuthorization
```

`ReplayBindingAssessment.verdict` 使用 `VERIFIED_COMPATIBLE`、`VERIFIED_INCOMPATIBLE`、`INDETERMINATE`；每个
action-stage/action-space/frame/transform/unit/component/spec/selection-group/execution-capability 维度分别记录
EXACT/DERIVED/TRANSFORM/SUBSET/UNKNOWN/MISMATCH/NOT_APPLICABLE。source contract 必须来自 explicit dataset
contract 或版本化 deterministic legacy derivation；字段不足保持 unknown，禁止按数组宽度、当前设备默认值或
旧 executor 能否运行来猜 source frame。

assessment 只描述 source 与一个 immutable `TargetRuntimeActionContract` 的关系，不授予 motion authority。
同一 source 换 target contract 必须得到不同 assessment；target profile 变化后旧 assessment 不可复用。
`VERIFIED_COMPATIBLE` 可成为后续 execution candidate，但仍需 current safety/capability、operator authorization、
execution gate 与 DispatchOwnerToken。`VERIFIED_INCOMPATIBLE` 只能 record/rejected。

为不阻塞 non-authoritative GUI shadow，M5.3a/b 实现了一个严格限定的 conditional 分支：source interpretation 已
verified、没有已知 mismatch、且输入来自当前 legacy replay 的同一次 immutable emission 时，indeterminate
assessment 可以得到 `PASSIVE_SHADOW` eligibility；它必须记录 `evaluation_allowed=true`、
`authoritative_allowed=false`，语义是“would select if target binding is valid”，不能写成 verified compatibility。
若 source canonical frame/spec 本身不确定，则不能进入 Arbiter candidate，只能保留 source/derivation rejection。

M5.3a/b 的 replay 专属 eligibility gate 继续只支持 passive evaluation。M5.5b 为 authoritative 路径新增独立、
receiver-admitted 的 `CandidateAuthorizationRecord/Snapshot`：它同时绑定已 admission proposal 的 content hash、
binding eligibility、typed execution authorization 和 exact target contract，并随完整 `ArbitrationInputSnapshot`
冻结。pure Arbiter 不读取 live verifier/store；publication barrier 只对被选中的 replay proposal检查冻结证据。
因此晚到证据只能影响下一 decision，缺失/denied/target mismatch 在 backend 前 fail closed，且不会把 conditional
shadow 升格为 compatible。

### 5. position/reference/command

在 production decision policy v1 下，selected action 是不超过 60 Hz 的 Arbiter decision/keyframe，
不是 60/120/150 Hz 每个执行 tick 的新 policy action；Arbiter 上限不限制下游 reference、IK 或 command
clock：

```text
selected RobotAction keyframes
  → ResolvedExecutionTarget
      ├─ direct command intent
      │    → backend acceptance
      │    → eligible anchor commit
      └─ pose/qpos/trajectory reference
           → ReferenceSegmentAdmission
           → derived PositionReferenceFrame / per-tick IK result
           → PositionCommandTick
           → backend dispatch receipt
  → measured state alignment
```

继续复用 `ReferenceTimeline`、`PositionDeployExecutor`、`JointCommandInterpolator` 和 position audit。
只扩展 lineage：

```text
decision_id
proposal_id_by_component
policy_node/context IDs
representation_trace_id
reference_segment_id
command_id
dispatch_receipt_id
measurement_alignment_id
```

每条 edge 还记录 input/output schema hash、transform contract hash 和 component dependency set；不能把
Cartesian selected action、qpos reference、actuator command 和 measured TCP 都解释为同一个 action vector。

`PositionReferencePublisher` 仍是现有 real teleop 接受并发布已 dispatch target 的边界，不新增 policy
专属 timeline。60/120 Hz execution event 继承所属 decision，不被记录为新的 source proposal。direct command
与 reference target 必须分支表达：direct command 只有 required backend receipt 全部 accepted 后才能形成 anchor
commit evidence；reference target 必须先成为 typed segment，不能把 frontend queue submit、单个 reference emission
或 command receipt混名为 segment admission。

当前 `CausalMotionResampler` 已证明一个 reference 可同时引用时间上较早和较晚的两个 source keyframe，并保存
parent index/timestamp/alpha；这里的 earlier/later 不是 left/right arm。M5.7e2 将这层关系冻结为有序、带角色的
endpoint parent：`START` 可以来自 fresh measured rebase 或已 committed safe endpoint，`END` 来自当前 safe
target。每个 parent 内部仍可包含 coupled 双臂 component projection。

segment admission 与 anchor 生效是两个时间事实：worker 可以提前接纳未来 endpoint，但该 endpoint 只能在
`effective_from <= next_safety_target_effective_at`、commit evidence 已在 safety cutoff 前 admission、且 segment
未 supersede/cancel、epoch/generation 仍匹配时参与 anchor 选择。实现不维护一个会被未来 admission 立即覆盖的
mutable `current_anchor`；admission、commit candidate 和 terminal/invalidation 都是不可变事实，某个 target time 下的
`PENDING_EFFECTIVE/ACTIVE/SUPERSEDED/INVALIDATED` 由窄 per-group timeline resolver 派生。
segment 还必须显式区分 `legacy_execution_observation`、`counterfactual_shadow` 和
`authoritative_shared`；前两者不能因数值/时间 eligible 而推进真实 anchor。

`ReferenceSegmentAdmission` 只说明 worker 原子接纳了一个 segment，不自动证明 endpoint 可执行、backend 已接受或
连续轨迹安全。profile 必须声明 `admission_guarantee`/`required_anchor_commit_guarantee`；当前 collision-aware pose
worker 在每个高频 reference tick 才运行 coupled IK/planner，因此 segment admission 默认最多声明
`ENDPOINTS_ONLY`，每个 `ReferenceFrameEmission` 还要引用该 tick 实际使用的 measured/solver input 和 result。
只有显式连续认证才能声明 `CONTINUOUS_CERTIFIED`。

legacy 迁移期间必须再分三层成熟度：`LegacyPositionReferenceSegmentObservation` 只记录 worker 实际选择 parent pair
并由 observer 首次看到其派生 reference 的事实；`BoundLegacyReferenceSegmentObservation` 可绑定 typed domain/group/parent 和调用方
execution context，但仍是 non-authoritative observation；只有真正执行 shared M1C guard 的 worker 才能产生
`ReferenceSegmentAdmission`。当前 position worker 没有独立 segment queue/admission 点，因此第一层采用
`first_observed_reference_derived_from_exact_parent_pair` 观察语义，不能把 `_accept_submission()`、`submit()` 或 debug callback
改称 admission。executor-local generation 使用独立 namespace；调用方附带 execution epoch/generation 只证明上下文
绑定，除非同时有 creation/dequeue/pre-backend guard evidence，否则不能声称这些语义已被 worker enforce。

legacy parent provenance 允许 `LEGACY_POST_SAFETY_KEYFRAME`、`LEGACY_POSITION_KEYFRAME` 和
`MEASURED_REBASE` 等真实 stage；projection 不得将其升级成不存在的 shared safety result。raw/bound observation 均
固定不能产生 anchor commit、authority、permit 或 backend side effect。插值 frame 引用其 raw segment；exact/hold
frame 单独记录 derivation，不伪造双端点 segment。public ControlTrace schema 等 runtime producer 稳定后再冻结。

当前 scoped e2-b1 已在 `PositionDeployExecutor` 增加默认不绑定的非阻塞 observer：它在 resampler 选定真实 parent 后、
per-tick resolver/IK/planner 前生成 `resampled_pre_resolver` fact，并按 modality/generation/parent pair去重 segment。
这证明 reference derivation lineage，但不证明 solver input/result、segment terminal、backend acceptance或M1C enforcement；
bound projection目前只有 pure fixture，无 production binding caller。上述缺口补齐前仍不得生成 canonical admission。

legacy pre-resolver frame、post-resolver outcome和backend receipt是三个不同事实。post-resolver observation只能在
resolver/planner成功、command intent已构造且 executor-local generation/fault二次检查通过后产生；它记录 actual
planner input/output和 command-intent hash，但固定声明 enforcement namespace仍为 legacy local generation。它不能
因“pre-backend checked”而升级成 shared M1C admission，也不能将 backend accepted置真。planner内部 measured state
尚无 typed ID时必须显式标记不可完整重放，不从 debug metadata猜测。

legacy local-generation terminal同样只是观察事实：clear/close可产生 canceled，fault可产生 faulted；它只证明旧
generation失效，不证明backend停止。当前worker没有正常 completed或同-generation supersession原子事件，禁止离线
猜测。production bound caller还必须等待shared execution epoch/group generation owner和正式sink，不能只复用
Arbiter shadow的control-domain ID补齐缺失语义。

authoritative receiver还需要一个与普通command payload分离的不可变dispatch context。该context在最终motion permit
通过后携带decision/directive/shared-safety/command lineage、target effective time、authority/execution epoch和
per-group generation；receiver必须在实际mailbox/timeline接纳点再次校验live scope，才能生成canonical admission。
context被传入不等于worker已enforce。当前MuJoCo e2-c0已提供该seam，但现有zero target lead使qref admission与END
effective time重合，不能形成future pending segment；必须用版本化nonzero target-lead/activation profile验证，或将
zero-lead路径明确保持为immediate qref admission而非reference segment。两种语义使用独立版本化
`ReferenceActivationSpec`，不得根据lead是否为零或payload字段是否存在临时推断mode。

scheduled profile v1要求receiver在segment START前完成接纳，late admission固定reject-and-hold，不实现
mid-segment catch-up。target lead至少覆盖一个selected-keyframe period、validated publication→receiver budget和
scheduling guard margin；deterministic 30/60 Hz sim fixture先使用`1/30 + 1/60 = 50 ms`，其中一个reference period
只是fixture margin，不是real默认值。startup/recovery的START来自同schema fresh measured rebase并对齐下一reference
tick，steady-state START来自previous committed endpoint，END来自current shared-safe target。

terminal同样是因果事实而非事后标签：`available_at`决定某次safety cutoff能否看到terminal，`effective_at`决定目标
时刻是否已失效，同时间由receiver admission sequence排序。只有在cutoff前可用的cancel/supersede/fault才能影响该次
在线timeline resolution；离线reader不得把晚到terminal倒灌进过去decision。

### 6. real policy frontend

`teleop_runtime/deploy/run_policy.py` 继续持有：

- real CLI/config/artifact binding；
- arm/hand/camera construction；
- startup/readiness/recovery/keyboard；
- real fault latch、hold 和 manual rearm；
- position executor、hardware dispatch 和 real debug observer。

当前 `PolicyDeployRuntime.prepare_next_action()` 是 action queue 唯一 pop 点，并已返回 raw action、
lookahead 和 pop 前后 queue length。迁移时将其结果扩展为 typed、immutable `PolicyActionEmission`，但不
新增第二个消费 API：

```text
PolicyDeployRuntime.prepare_next_action()   # 唯一一次 pop
  → PolicyActionEmission(chunk/keyframe/inference/output/target-time lineage)
      ├─→ legacy canonical execution owner
      └─→ passive arbiter_shadow adapter
```

shadow coordinator 不持有 queue、不调用 prediction transaction、不调用 `pop_action()`/`prepare_next_action()`，
只消费 frontend 已取得的同一个 emission。lookahead 的 peek/validation 不能被误记成已消费 keyframe。启停
shadow 必须保持 queue pop count、chunk index、queue length、prediction schedule、replace/append、
`last_action` 和 dispatch count 完全一致。

它先通过窄 `arbiter_shadow` 组合现有“一个 policy 变量 + 一个 runtime”，不把真机对象移入
`robot_motion`。单实例 parity 完成后，只增加用于 synthetic 的最小 context/action node protocol；完整
production `PolicyOrchestrator` 仍等确有多节点需求时再引入。

### 7. deployable sim

未来 `sim/mujoco/deploy/` 只提供：

- deterministic sim clock；
- scene state/RGBD/tactile observation producer；
- `RobotAction`/qpos sink；
- `SimDispatchReceipt` 和 measured state。

它复用同一 ObservationCutRecord/Bundle、policy node、proposal/arbiter、canonical action、position 和 trace
contract。现有 `sim.mujoco.actuation.MujocoActuatedPositionExecutor` 已验证 15/60/120 的 sender/plant
边界，可作为 command/physics backend；CubeLift task-native RL 路径继续保留，不冒充 deployable sim。

---

## 九、v2.4 数据集格式：多来源、多频率与训练事实层

### 1. v2.4 与 v2.3 的关系

建议正式使用：

```text
dexteleop_dataset_v2.4
```

v2.4 不是替换 v2.3 pose contract，而是在其上增加 control/learning trace：

| 版本 | 核心增量 |
|---|---|
| v2.1-style | episode parquet、video 和外置 tactile/depth 的基础物理布局 |
| v2.2 | 可配置 `PolicyIOLayout`、连续/离散 action/state 和 layout hash |
| v2.3 | absolute pose storage、relative model/deploy view、pose/frame contract、严格 causal snapshot |
| v2.4 | 可扩展的 control/learning facts：多来源 proposal→selected action、multi-rate stream、context、execution 与派生训练层 |

v2.4 必须继承以下 v2.3 规则：

- pose 落盘仍为 explicit raw absolute 或 canonical desk absolute；默认 native v2.4 使用 desk absolute；
- relative action 仍是 reader/deployment representation，不另建相互漂移的 relative parquet 真相；
- relative chunk anchor 仍来自 causal measured actual TCP，并固定引用 anchor observation ID；
- 每个 sensor/action/control event 都保留其角色对应的 timestamp 和 clock domain；sensor 兼容保留
  v2.3 source/available evidence，proposal/context 使用 produced/available/effective taxonomy；
- 所有 model-facing view 仍绑定 `PolicyIOLayout`、pose/frame、chunk 和 artifact hash。

v2.4 新增的是“谁提出了哪些 action、如何变成统一 action，以及统一 action 执行后的真实结果”，而不是
修改 v2.3 的几何定义。`dexteleop_dataset_v2.4` 通过 capability 渐进实现，不要求第一版同时完成全部
learning/sim2real schema：

```text
control_trace_core       # 第一批 mandatory
native_streams           # 可按 stream 声明
composition_context      # generic topology 使用时 mandatory
policy_context           # legacy v2.4 trace read compatibility only
learning_overlay         # reward/labels/training views
domain_manifest          # sim2real extension
```

reader 必须按 capability 和 completeness matrix 判断可用性，不能仅凭版本号假定所有扩展都存在。

### 2. 四层事实模型与三类存储

v2.4 先区分语义层级：

1. **原始事实：** SensorSample、HumanInput、PolicyInference/Output、CompositionContext、ActionProposal、
   task/termination fact；这些不能由最终 command 反推。
2. **在线控制事实：** ProposalSnapshot、ArbitrationEvaluation/ArbitrationDecision、
   LegacyExecutionObservation、SafetyResult、ReferenceSegment、Command、
   DispatchReceipt、MeasurementAlignment；这些由当时在线控制过程产生。
3. **派生学习事实：** expert/policy/learning mask、reward、transition、credit window 和 training target；
   必须携带 derivation spec/version，可离线重算。
4. **物化视图：** actual-decision、fixed-rate compatibility、BC、DAgger、fast-policy、RL view；它们是可
   重建缓存，不改变前两层原始含义。

物理存储仍采用三类组织：

```text
native streams       # sensor/state/reference/command 等按真实频率存储
typed control trace  # 第一、二层事实事件
materialized views   # actual-decision / fixed-rate / training compatibility cache
```

各层职责如下：

1. **native stream 是原始时间事实。** RGBD、tactile、state、force、human input、reference、command
   各自按 retention profile 保存，不因某个 decision row 没有产生就自动丢失。
2. **ControlTrace 是关系事实。** capture 保存 ObservationCutRecord、policy output、proposal、Arbiter
   decision、safety、dispatch 等第一/二层事件；reward overlay 通过稳定 ID 引用它们。
3. **materialized view 是兼容与快速读取层。** 不代表所有 modality 都产生了一帧，也不承载所有高频
   execution sample；任何 view 都必须声明 source IDs 和 materialization spec。

不建议继续给 `CollectorSample`/单 parquet row 增加任意数量的 `policy_0.*`、`policy_1.*`、tactile
window 或 120 Hz command 列。那会导致 schema 随策略数变化、重复大 payload，并使频率语义再次退化为
nominal 15 Hz。

### 3. 逻辑表与可演进物理布局

冻结逻辑 table family，不冻结“每 episode、每 family 一个 Parquet 文件”。capture root 保持
LeRobot-v2.1-style 兼容外形，并通过 manifest 定位实际分区：

```text
datasets/<capture_name>/
  meta/
    info.json
    capture_manifest.json       # live/sealed、capture hash
    episodes.jsonl
    episodes_stats.jsonl
    tasks.jsonl
    logical_table_layout.json   # family -> partition/fragment locator
    contracts/
      action_component_catalog.json
      reference_command_measurement_schemas.json
      stream_catalog.json
      control_trace_contract.json
      policy_graphs.jsonl
      deployment_bindings.jsonl
      execution_manifests.jsonl
      policy_io_layouts.jsonl
      domain_contracts.jsonl

  data/
    chunk-000/episode_000000.parquet
      # manifest-selected materialization；authoritative 时可为 ActualDecisionView，
      # shadow 期继续保留 legacy row，不让 would_select 覆盖实际执行事实

  trace/
    <logical_family>/<manifest-selected fragments or chunk parts>.parquet

  streams/
    <stream_id>/chunk-000/episode_000000/manifest.json + payload

  videos/...
  tactile/...
  wrist_depth/...
```

typed family 保持为 `observation_cuts`、`policy_events`、`action_events`、`execution_events` 和
`task_events`；派生层另有 `reward_events`/`annotation_events`。高频 reference/command 数值放 native
stream 时，execution event 只保存 segment/range ID 和语义事件。

路径和 partition strategy 由 `logical_table_layout.json` 声明。现有 writer 每个 episode 原子 finalize，
所以 v2.4 Core 第一版可以先写 episode fragment；数据量增大后可将 sealed fragments copy-on-write compact
为 `trace/<family>/chunk-xxx.parquet`，以 `episode_uuid + row group + event_sequence` 索引。reader 只依赖
logical table locator，不依赖文件名。这样既避免早期改写 writer 原子性，也不会把大量小文件固化为 contract。

现有 `videos/`、`tactile/` 和 `wrist_depth/` 可以继续作为特化 stream encoding；`StreamCatalog` 为它们
提供统一逻辑索引，不要求第一版搬文件。

capture root 在采集期间可以按 episode append；一旦 `capture_manifest.status=sealed`，streams、第一/二层
trace、contracts 和 checksum 全部不可变。离线 reward、annotation 和新 training view 写入独立 derived
overlay：

```text
derived/<derivation_id>/
  meta/derivation_manifest.json   # parent_capture_hash + spec/code/model hashes
  trace/reward_events/...
  trace/annotation_events/...
  views/...
```

多个 overlay 由 `DatasetDerivationManifest`/collection manifest 组合，不回写 sealed capture。在线环境
提供的 success/contact/termination 先作为 capture task fact；具体 reward 仍按 spec 进入 overlay。

### 4. ID 与时间主键

v2.4 不能只靠 `episode_index + frame_index` 关联数据。建议所有对象使用稳定 ID：

```text
episode_uuid
runtime_session_id
control_domain_id / embodiment_instance_id
stream_id / stream_sample_id
observation_cut_id
policy_inference_id / policy_output_id
context_id
representation_trace_id
proposal_id
proposal_snapshot_id
decision_id
safety_decision_id
reference_segment_id
command_id
dispatch_receipt_id
measurement_id
task_event_id / reward_event_id / transition_id
```

runtime fact ID 统一由
`runtime_session_id + control_domain_id + producer_id + producer_epoch + monotonic sequence` 组成；不能依赖
物理 row offset，也不能因为进入/离开一个 dataset episode 而改变。dataset record 另带可空
`episode_uuid` 作为 capture membership；policy deploy 在没有数采 episode 时仍可产生稳定 fact ID，同一
runtime session 中多个 episode 也不重编号 producer。所有 ControlTrace Core event 必须显式携带
`control_domain_id`；存在多 embodiment 时还要携带 `embodiment_instance_id`，不能只靠文件路径推断。所有
事件至少包含：

- 该 event type 规定的 role-specific timestamps；
- 每个 timestamp 的 clock domain 和可选 clock evidence；
- sequence 和 producer epoch；
- valid/invalid/drop 状态；
- contract/schema hash。

必须区分三类 materialization：

```text
ActualDecisionView
  一行一个 authoritative ArbitrationDecision
  decision_index / decision_cutoff_at / decided_at
  effective_interval_start/end / dt_sec
  variable dt；不声明真实 decision 等间隔

FixedRateCompatibilityView
  15/30 Hz 等显式 materialization profile
  is_resampled = true
  source_decision_ids
  hold/skip/interpolate policy + hash

ShadowEvaluationView
  一行一个非权威 ArbitrationEvaluation
  would_select_action / would_select_owner_by_component
  legacy execution comparison IDs
```

native v2.4 的 authoritative capture 才可将 `data/...episode.parquet` 物化为 ActualDecisionView；其
`frame_index` 仅兼容映射到
`decision_index`，`frame_count` 是真实 decision row 数。`nominal_fps` 只允许出现在 fixed-rate view 或
作为旧 reader 的 target-profile metadata，不能用来断言真实 decision 均匀。每个 stream 的 sample count、
measured rate 和 drop count 单独写入 manifest/episode summary。

M3–M5 shadow capture 不生成 ActualDecisionView：旧 v2.3 row 继续表示 legacy execution fact，v2.4 使用
`ShadowEvaluationView` 保存反事实结果，并用独立 `LegacyExecutionObservationEvent` 保存实际进入旧
safety/command/dispatch 的 action 与 receipt。`legacy_control_path` 是 execution lineage origin，不伪装成
`ArbitrationDecisionEvent`。

### 5. 多来源 action 如何形成统一 action

v2.4 固定 action lineage：

```text
raw policy/human/replay representation
  → absolute RobotAction proposal(s)
  → ActionArbiter selected canonical RobotAction
  → safety modification / planner / reference
  → command / dispatch receipt
  → measured state and task outcome
```

#### 5.1 proposal 事实

`action_events` 中每条 `ActionProposalEvent` 保存：

- control domain、producer kind/id、policy node/artifact/producer epoch；
- produced/available/effective interval 和 keyframe target effective time；
- observation cut、policy output、context dependency IDs；
- absolute canonical `RobotAction` 值；
- `ActionComponentCatalog` 下的 coverage/valid mask；
- representation mode、v2.3 anchor 和 frame/contract receipt；
- raw validity/status，以及与具体 snapshot/decision 关联的 superseded/coalesced/interval-missed disposition；
  shadow 是 consumer/evaluation mode，不改写 proposal 自身 validity。

多个 proposal 使用同一个 dataset-level `ActionComponentCatalog` 投影。continuous float 与 discrete
token 分开保存；partial proposal 使用 mask，不用零或 NaN 猜 source coverage。raw relative output 可作为
representation payload/sidecar 保存，但 Arbiter 和统一 action 只引用 decoded absolute proposal。

#### 5.2 computation、shadow evaluation 与 authoritative publication 事实

pure calculation、shadow 解释和 authoritative publication 分型，避免“算出了动作”被误记为“系统正式发布
了动作”：

```text
ArbitrationComputationEvent
  input_snapshot_id / arbiter contract+implementation hashes
  result ID/hash / decided_at

ArbitrationEvaluationEvent
  non_authoritative = true
  computation/result ID
  would_select_action / would_select_owner_by_component
  evaluation_reason

PublicationAttemptEvent
  computation/result ID / ExecutionDirective ID
  live gate evidence / dispatch owner generation
  authority epoch / execution epoch / attempted_at

ArbitrationDecisionEvent
  authoritative = true
  publication_attempt_id / published_at
  authority_epoch / execution_epoch
  selected_action / ExecutionDirective
  actual control lineage

ArbitrationPublicationRejectedEvent
  publication_attempt_id / computation/result ID
  rejection reason + conflicting gate/epoch/token evidence
  cancellation ID | null
```

shadow 只生成 computation + evaluation；authoritative runtime 生成 computation 后必须经过 publication attempt。
barrier 成功才生成 `ArbitrationDecisionEvent`，失败只生成 rejection，不能创建 selected/reference segment。

每条 `ArbitrationDecisionEvent` 保存：

- control domain、schedule tick/cutoff/decided/published/target-effective time、trigger、authority epoch；
- `arbitration_input_snapshot_id/hash`、`proposal_snapshot_id/hash`、authority/gate/producer-epoch snapshot IDs；
- 每个 selection group/component 的 selected proposal/source；
- available/selected/rejected/expired/incompatible masks 和 reason；
- 组合后的 selected canonical `RobotAction`；
- handoff、hold/no-action/fault 状态。

只有 authoritative event 能生成事实层 `selected_action` 和 ActualDecisionView。shadow evaluation 只能
生成 `would_select_action`/ShadowEvaluationView；M6 owner 切换前不得覆盖 v2.3 compatibility row 的实际
动作。computation 是确定性计算事实但不是控制决定；proposal/snapshot 本身仍是事实，evaluation 是明确
标记的反事实计算结果，publication rejection 也必须可审计。

#### 5.3 事实层 `selected_action` 与 materialized `action`

native v2.4 的 `data/...episode.parquet` 中：

```text
selected_action                         = authoritative Arbiter selected canonical intent（pre-safety）
selected_action.discrete                = selected canonical discrete token
selected_action.component_valid_mask    = 本次 selected action 的有效 component
selected_action.decision_id             = 对应 ArbitrationDecisionEvent
decision.previous_selected_action_id = 上一个有效 selected decision/action ID
```

`selected_action` 就是事实层的“统一动作”。它可以由 human 左臂、policy 右臂和另一个 policy 的 hand
component 组成，但每个
component 的 owner/proposal ID 必须可查。

事实层不提供含义未声明的裸 `action`。只有 `ActualDecisionView`、`FixedRateCompatibilityView` 或
`TrainingViewContract` 可以将某个显式 stage 映射为兼容列 `action`，并强制写入例如
`action_stage=selected_pre_safety`、source schema hash 和 materialization spec hash。这样旧/第三方 reader
不能在未读取 contract 时把 intent 自动当成 hardware behavior 或默认训练 target。

以下内容不得复用 `selected_action` 的名字：

```text
action.proposal.<source>
action.safe
action.reference
action.commanded
action.executed
action.measured_effect
```

它们分别存在 proposal/execution trace 中。HOLD、FAULT 或无有效 selected action 时，decision view 使用显式
status/valid mask；不能把零向量或无限期重复的 last action 伪装成新统一动作。

native v2.4 不在 dataset-level 定义 `observation.last_action`。事实层通过前序 event ID 保存
previous selected/safe/commanded/measurement 关系，policy event 另存每个 node 的 raw output 和 decoded
proposal。`PolicyObservationView` 只有在 artifact 的 `LastActionContract` 声明后才投影：

```text
last_action_semantics:
  previous_policy_output
  previous_decoded_proposal
  previous_selected
  previous_safe
  previous_commanded
  previous_executed_feedback
```

contract 同时绑定 producer scope、action/command schema、pose mode、anchor 和 unavailable policy。v2.3
compatibility row 继续保留其现有 `observation.last_action`，不能在 dual-write 时改含义。

长序列 policy 不能把整个 action history 继续压缩成一个 `last_action`。artifact 必须另行声明
`HistoryActionContract`，至少区分 policy/raw proposal history 与实际 behavior history，并规定 human override
期间使用 counterfactual policy、human behavior、二者带 source token，还是 mask。它与 relative decode anchor、
`LastActionContract`、`previous_safe_committed` 和 backend slew anchor 是五个独立 contract。

### 6. 统一动作以下对模型不可见

“对模型不可见”指默认 actor input/target contract 不可见，不代表数据集不记录。v2.4 为字段和 stream
定义 access role：

```text
actor_observation
actor_target
composition_context
critic_only
reward_only
audit_only
control_private
```

默认规则：

- actor observation 只读取 `PolicyObservationView` allowlist；
- 事实层不定义默认 actor target；每个 target 必须由版本化 `TrainingViewContract` 显式选择；
- safety result、IK/planner、reference、command、receipt、hardware fault detail、reward 和 attribution
  不能作为下一步 actor input；
- critic/reward/offline analysis 可以通过显式 training-only view 读取执行事实；
- 如果算法确实需要执行反馈，必须新增版本化 observation field，例如
  `previous_executed_action_feedback`，不能偷换 `observation.last_action` 语义。

### 6.1 Learning observability：只前移不可恢复事实

控制事实与 learner 实现之间增加一个窄的、可选的 learning-observability 层。它不进入 Arbiter，不改变 actor
allowlist，也不拥有训练线程。以下证据只有 policy adapter 在 inference 时实际提供，才可成为 capture fact：

```text
PolicyInferenceObservability
├── observation cut / policy output / model-side chunk
├── artifact-bound adaptation state before/after
└── generative sampler/noise evidence
```

`PolicyAdaptationStateRecord` 表示 RNN/KV/fast-weight/test-time-adapter 等模型内部顺序状态；它不是可被其他 node
通用消费的 `CompositionContext`。state transition 绑定 inference，而不是 chunk queue 中之后发生的每次 keyframe pop。
普通 capture 不要求逐步保存完整权重；profile 可以只要求 state identity/checksum 和周期 checkpoint，也必须声明
其只能 semantic replay、不能 bitwise replay。若模型是 stateless，显式 `not_applicable`，不能制造空 state。

生成式 policy 的 evidence 同样由 capability/profile 驱动。需要 exact-noise 或 FlowDAgger 类训练的 profile 应保存
实际 initial-noise tensor/blob、sampler/solver/action-head/artifact hash 和生成出的 model-side chunk；仅有 seed/RNG
receipt 只能声明较弱的重建能力，不能冒充实际 noise。当前通用 `PolicyLike.predict()`/OpenPI client 只返回最终
output，因此在具体 server/client 增加 typed hook 前，这些字段必须 unavailable/not-applicable，不能由 output
反推。

`PolicyActionCodecContract` 连接 model action chunk 与 canonical trajectory：decode 是现有 v2.3 执行事实，encode
是 human correction 回到模型原生 chunk 的独立能力。encode 必须绑定 observation/anchor、frame、horizon/cadence、
rotation convention、normalization、padding、component mask 和 effective-time schedule；没有 round-trip fixture 时
不得声明 bidirectional。通用框架不实现一个猜测性的 inverse。

这些事实加入 formal trace 前必须有具体 producer 和最小消费 fixture。ControlTrace admission 失败只使声明依赖
它的 learning profile/view 不完整，不改变 motion；runtime object、CUDA pointer、共享内存 handle 不可持久化。

`dataset_interface` 不应向模型返回完整 parquet row。每个模型使用 `TrainingViewContract`：

```text
TrainingViewContract
├── view_id / policy_node_role
├── PolicyIOLayout / ActionComponentCatalog hashes
├── observation cut/window specification
├── context dependencies
├── action target kind: human_expert_proposal / selected_human_components /
│                      selected_action / policy_proposal / safe_action / behavior_action
├── target component mask / action stage / provenance
├── behavior action stage: selected / safe / commanded（仅 RL/critic）
├── reward/task spec
├── field/stream allowlist by access role
└── resampling/window/padding rules
```

这样 BC、DAgger 和 RL 可以基于同一份事实选择不同 target，而不改变原始数据含义；缺少显式 target
provenance/mask 的 view 必须 fail closed。

### 7. 多频率 modality 的 native stream contract

`StreamCatalog` 为每个 stream 定义：

```text
stream_id / modality / producer_id
value schema、shape、dtype、units、frame
measured/available clock domain（兼容 v2.3 source alias）
nominal/min/max expected rate
encoding/compression/payload location
retention policy
validity/freshness policy
access role
stream contract hash
```

每个 `SensorSampleRecord`/stream index 保存 sequence、measured/available timestamp、stable payload object
ID、validity 和 drop/gap reason；runtime handle 只存在于 bundle。ObservationCutRecord 只引用：

- point selection：`latest_at_or_before(T0)` 的 sample ID；
- aligned selection：与 RGBD measured time 对齐的 state sample ID；
- window selection：tactile/state 的 `[start, end]` sample range 和 aggregation contract；
- held context：同一 frame/context ID 及其当前 age。

不得在主表中复制高频样本，也不得用简单的 frame number 比例对齐。重复使用同一 15 Hz RGBD 时，30 Hz
fast-policy cut 引用相同 frame ID；tactile 仍可引用新的 sample/window。

现有 writer 的 external tactile dedup 是很好的兼容基础，但 native v2.4 还需要独立 stream ingestion：

- v2.3 兼容模式继续只写被主 sample 选中的 tactile snapshot；
- v2.4 native 模式允许 tactile producer 按 retention profile 独立写入 30–1000 Hz stream；
- raw tactile、derived force/slip feature 可作为两个 stream，以 derivation contract 关联；
- 若因带宽只保留 decimated/window feature，必须保存明确 decimation/filter contract，不能声称是 raw；
- video/RGBD 每个真实 sensor frame 只编码一次，decision/cut 通过 frame ID 引用；
- episode summary 为每个 stream 保存 produced/written/dropped/gap/rate/latency 统计。

### 8. Reward、task outcome 与 transition

reward 不应直接写进 ActionArbiter，也不应只有一个不可解释的 `reward` float。v2.4 先记录 task facts：

```text
TaskEvent
├── task/phase/skill/contact-state
├── source: env / online operator event / online detector
├── event interval / confidence / validity
├── success / failure / termination / truncation / reason
└── referenced observation/measurement IDs
```

再允许一个或多个版本化 reward 定义产生 `RewardEvent`：

```text
RewardEvent
├── reward_event_id
├── parent_capture_hash / derivation_id
├── reward_spec_id / code-or-config hash / version
├── interval [decision_i, decision_j) / dt_sec
├── total reward
├── named reward terms
├── source: online_env / offline_recompute / human_label / learned_reward
├── referenced decisions/actions/measurements/task events
├── terminated / truncated / bootstrap_allowed
└── validity/confidence
```

同一个 episode 可以并存多个 reward spec，但这些 event 写入 immutable derived overlay；sealed capture
不追加文件、不改 checksum。human/offline annotation 使用独立 `AnnotationEvent` 进入带 parent hash 的
overlay。RL transition 由
`TransitionViewContract` 构造：

```text
(observation_cut_t,
 actor_action_t,
 environment_behavior_action_t,
 selected/proposal attribution,
 reward over [t, t_next),
 next_observation_cut,
 dt_sec,
 terminated,
 truncated,
 bootstrap mask)
```

`actor_action_t` 保留 policy 原生 action/chunk schema；`environment_behavior_action_t` 必须显式选择 selected、
safe、reference 或 commanded stage。真机 RL 通常需要 safe/commanded
及 dispatch/measured validity 来描述真实环境行为，但这些字段只进入 learner/critic/reward view，不成为
部署 actor 的在线输入。每个 behavior stage 还必须绑定自己的 action/command schema；joint command 不能
因为被选作 behavior fact，就伪装成与 actor `RobotAction` 同空间的训练 target。只有有显式 codec/transform
证据时，critic/Q learner 才能把两个 action space 关联。

离线 reward/annotation 的 available time 可以晚于 episode；它们只能作为版本化训练派生事实，不能被
回填成当时 actor 已知的 observation/context。是否允许某个 learner view 使用它，由 access role 和
`TrainingViewContract` 决定。

### 9. 面向训练任务的 v2.4 视图

| 用途 | observation/target | 必需 lineage |
|---|---|---|
| BC | 指定 policy view + selected human action 或 human proposal | observation cut、proposal/decision、component mask |
| DAgger/intervention | failure 前 policy proposal/history + human correction sequence；按 component/phase 派生 label | policy/human proposal、authority/clutch/release、selected/safe/commanded/measured、derived segmentation/mask |
| 快慢系统 | slow observation→context；fast observation+context→action | context version、fast cut、proposal、后续 outcome window |
| 真机 RL | actor view、真实 behavior stage、reward、next cut | policy proposal、selected/safe/commanded、receipt/measured、termination |
| sim RL | 与真机相同 canonical actor/action；可增加 privileged critic/reward-only stream | domain、randomization、physics state、同一 action/trace contract |
| sim2real | 多 dataset root 的同构 training view | pose/action/layout/view hashes、domain contract、sensor/control profile |

component mask 分为两类，不能继续只用一个 `intervention_mask`：

- 第一/二层事实 mask：`proposal_available_mask`、`selected_mask`、`safety_modified_mask`、
  `command_valid_mask`、`receipt_valid_mask`、`measurement_valid_mask`；
- 第三层派生 mask：`expert_label_mask`、`policy_behavior_mask`、`policy_learning_mask`。

派生 mask 必须绑定 `training_view_id + derivation_spec_hash`；尤其 `policy_learning_mask` 取决于算法、target
stage、reward 和 transition validity，不能作为一个无版本的 capture 原始列。

capture 只保存可观察的 takeover/release、human raw/canonical proposal、policy counterfactual proposal、实际
behavior 和 safety projection。`PRE_FAILURE/TAKEOVER_ALIGNMENT/STABLE_CORRECTION/RECOVERY_AND_RELEASE` 以及
context/action/inversion/preference/risk/recovery/value mask 由 immutable overlay 版本化派生，不写死在 sealed
capture。post-safety action 也不是默认 expert target：projection 足够小时可由 view 选择 safe target；大幅 clamp
的数据保留 raw intent 与 constraint evidence，但普通 BC/inversion view 应 fail closed 或降级到 risk/preference
用途。

快慢 credit 规则必须显式：slow context 的 reward/outcome window 由 slow-view contract 定义；fast policy
使用真实 30 Hz action transition。若窗口中发生 context switch、human takeover、safety modification 或
episode boundary，必须拆分或标记，不能静默归到单一 source。

### 9.1 训练产物如何回到部署

现有 `dataset_contracts.policy_artifact.PolicyArtifactContract` v1 继续保持兼容。v2.4 learner 通过 additive、
content-addressed artifact bundle 绑定 base checkpoint 与 adaptation modules，不在 v1 dataclass 中追加会改变旧 hash
的可选字段。learner 输出至少包含：

```text
TrainingRunManifest
├── parent capture/derived overlay hashes
├── DatasetCollectionManifest / TrainingViewContract hashes
├── reward/transition spec hashes（若适用）
├── learner code/config/dependency/seed hashes
├── train/validation split 与 metrics
└── produced checkpoint/artifact digest

PolicyArtifactBundle / v2.4 extension
├── exact base policy/model/action-head digest
├── adaptation module digests + composition order
├── adaptation-state init/reset contract（若适用）
├── generative sampler/noise adapter contract（若适用）
├── PolicyIOLayout / PolicyObservationContract / LastActionContract
├── HistoryActionContract / PolicyActionCodecContract（若适用）
├── action representation/chunk/frame/component contract hashes
├── required context contracts
├── EmbodimentCompatibilitySpec
├── training_run_manifest_hash
└── model/checkpoint digest
```

`deployments` 继续只负责 client/protocol/artifact transport；`robot_motion.deploy` 做纯 compatibility/session
contract 解析；`teleop_runtime.deploy` 在构造设备前或进入 readiness 前执行 deployment preflight。preflight
校验 observation streams、frames/calibration、action component coverage、end-effector capability、timing/
decision policy 和 embodiment adapter。通过后仍先 shadow，不能因为 artifact 来自本仓库训练就自动获得
authority。

闭环验收不是“checkpoint 文件生成成功”，而是同一个 artifact 能：

1. 从声明的 TrainingView fixture 重建模型输入并离线推理；
2. 在 deterministic replay/sim 中生成 contract-compatible proposal；
3. 在目标 embodiment binding 上通过 preflight 和 shadow parity；
4. 产生新 capture，并让新 trace 反向引用 artifact/training-run hash。

DAgger 的迭代闭环因此是
`policy+human proposal capture → expert/component TrainingView → updated artifact → shadow/deploy → new capture`；
RL 是 `behavior/reward/next-cut capture → transition view → updated artifact → sim/real rollout → new capture`。
二者共用事实层和 artifact gate，不各建一套数据/部署协议。

### 10. sim2real 与 domain contract

为避免把“机器人本体”“遥操输入设备”和“本次运行环境”混成一个 hash，先分型：

```text
SourceDeviceContract
  physical human-input schema / clock / calibration / retarget capability

ReplaySourceContract
  source capture/event / action stage / schedule / resampling / retarget contract

EmbodimentContract
  robot topology / effectors / canonical component+frame capability
  kinematics / actuator interface requirements / safety capability

DomainContract
  domain: real / sim
  embodiment_contract_hash / station_id
  pose-frame/calibration binding
  sensor/backend/physics/latency/noise/randomization contract
```

`DomainContract` 不包含本次跑了哪个 policy，也不把当前使用的 VIVE 当作 embodiment 属性：

```text
domain_id / domain_hash
domain: real / sim
embodiment_contract_hash / station_id
pose-frame/calibration contract
sensor/actuator capability profiles
backend/physics/simulator versions
actuator/sensor latency and noise model
domain randomization distribution/range contract
```

每个 episode 另保存 `EpisodeExecutionManifest`：

```text
episode_uuid / control_domain_ids / embodiment_instance_ids
source_device or replay_source contracts / source_adapter_receipts
embodiment_contract_hash / execution_binding_hash
policy_graph_hash / deployment_binding_hash
policy artifact versions / producer epochs
authority and safety profiles
trace requirement / action persistence profiles
decision/reference/command/physics timing profiles
task contract / active streams / randomization sample and seed
```

因此同一 station 更换 policy 或兼容遥操设备不改变 embodiment hash，同一 sim scene 更换快慢 topology 也
不创建新 domain；更换机器人 topology 才产生新的 EmbodimentContract。randomization sample/seed 是 episode
domain binding，可引用稳定的 base domain contract。

sim privileged state 必须标记 `critic_only`、`reward_only` 或 `audit_only`；它不能因为存在于同一 row 就被
actor reader 读取。real/sim 或不同 embodiment 的 actor-facing ActionComponentCatalog、PolicyIOLayout、
canonical frame 和 TrainingViewContract 必须满足 artifact 声明的 compatibility spec；DomainContract 与
EpisodeExecutionManifest 可以不同。若需要跨 embodiment action retarget，结果写入 derived overlay 并绑定
transform hash，不能修改原 capture 或假装原 action 天然同构。

由于 v2.3 pose source/frame contract 当前是 dataset-level immutable，初期不建议把不同 station 的 real
数据和不同 sim source 强行写进同一个物理 root。更稳妥的 sim2real 方式是：

```text
DatasetCollectionManifest
├── real dataset roots
├── sim dataset roots
├── capture/derived overlay hashes
├── required compatible contract hashes
├── domain sampling/weight policy
└── allowed missing/privileged modalities
```

各 root 保持内部同质，训练时由 collection manifest 联合采样。这也便于对 real/sim 分别做质量验收和
数据版本冻结。

### 11. dataset/episode metadata 与 contract registry

`meta/info.json` 至少增加：

```text
dexteleop_dataset_schema = dexteleop_dataset_v2.4
base_pose_contract = v2.3
capabilities
capture_status / capture_hash
action_component_catalog_hash
reference/command/measurement schema hashes
stream_catalog_hash
control_trace_contract_hash
policy_io/graph registry hashes
deployment/execution/domain registry hashes
source-device/embodiment registry hashes
training-run/policy-artifact registry hashes（capture 只引用当次 deployed artifact）
logical table layout hash
completeness matrix schema
```

dataset-level 固定 action component、pose 和基础 stream schema；episode metadata 绑定本次实际启用的：

- `DomainContract` binding 和 `EpisodeExecutionManifest`；
- policy graph、deployment binding、policy artifacts 和 producer epochs；
- source device/adapter receipt、control domain/embodiment contract+instance、station 和 task spec；
- decision/reference/command/physics profiles；
- active stream IDs 和 per-stream integrity；
- authority mode 与 session/termination summary。

允许 episode 使用 registry 中不同的 policy graph/artifact，但不得改变共享 action/pose/component 语义。
live capture append 只能新增 registry entry，不能原地修改已有 ID 对应的内容；seal 后 registry 也不可变。
training view、reward 和 annotation registry 位于 derived overlay。所有 entry 使用 canonical hash。

### 12. v2.3 兼容与迁移

v2.4 reader 可以为 v2.3 提供明确标记的 compatibility view，但不能伪造不存在的事实：

- 每个 v2.3 row 可映射为一个 compatibility step/observation cut；除非有证据，否则不声称存在真实
  `ArbitrationDecision`；
- 现有根 `action` 只有在 human-only/command lineage 证据充分时，映射为一个 human proposal 和
  selected action；
- 没有保存的 policy proposal、slow context、Arbiter candidate、safety modification、reward 或
  dispatch receipt 一律标记 unavailable，不能从最终数值猜测；
- existing causal camera/arm/hand/tactile snapshot columns 可以转换为 stream sample refs；
- compatibility view 保存 completeness matrix、`compatibility_origin=v2.3_single_action_row` 和
  `arbitration_unavailable=true`。

如需物理转换，必须 copy-on-write。native v2.4 writer 在过渡期可同时生成：

1. v2.3-compatible decision main row；
2. v2.4 stream/trace events。

这是同一个 `CollectionWriterRuntime` 的多 sink 输出，不是第二个控制 loop，也不为每个 policy 新建 writer
lifecycle。不要继续把 v2.4 typed event 全部塞进 `schema_v21.py`；新增纯 contract/encoder，并保留旧
row encoder 作为 compatibility materializer。

### 13. writer、完整性与验收

持久化层统一包裹：

```text
PersistedRecordEnvelope
  fact_id / semantic_content_hash
  encoded record + schema/registry refs
  episode membership / journal sequence
  record_integrity_hash
```

`record_integrity_hash` 覆盖最终 encoded envelope；runtime fact 不预先声称自己拥有 dataset row 的完整性 hash。

mandatory 不能用“信息存在时”或一个全局列表表达。每个 runtime/capture 绑定版本化
`TraceRequirementProfile`：

```text
TraceRequirementProfile
  profile_id / runtime_mode
  required / optional / not-applicable event types
  required streams / receipt+measurement policy
  completeness and writer-failure behavior

current profiles:
  plain_dry_run_v1 / policy_shadow_dry_run_v1 / replay_shadow_dry_run_v1
  formal_real_shadow_v1 / authoritative_real_v1 / deployable_sim_v1
```

例如 authoritative real 要求 proposal admission、ProposalSnapshot、ArbitrationInputSnapshot、computation、
publication attempt、successful decision 或 typed rejection、proposed/published ExecutionDirective、shared safety、
reference/command lineage、dispatch owner/receipt 和 terminal；policy/replay shadow 要求 computation/evaluation、
legacy execution observation 及 legacy safe/command/receipt，而不允许把 `would_select` 当 authoritative
decision。not-applicable 与 missing 必须分开。

- profile-required facts 无法 admission/durable 时必须留下 sequence gap/health evidence并使 capture
  uncommitted；这里的“required”不承诺任意故障下零丢失；
- RGBD/depth/raw tactile payload 可使用明确 drop/retention policy，但 index 必须记录 gap/drop；
- finalize 前验证所有 ID 引用存在、producer sequence 单调；所有声明为 causal dependency 的
  measured/available/produced/effective timestamp 满足对应 consumer cutoff；
- selected action 的每个有效 component 必须能追溯 proposal，safe/command/receipt 必须追溯 decision；
- derived overlay finalize 时，reward/transition 必须追溯 task fact、measurement、parent capture 和 reward
  spec hash；
- stream manifest 的 count/range/checksum 必须与 payload 和 ObservationCutRecord refs 一致；
- 不使用单一 `training_valid`。capture episode finalize 生成事实完整性，derived overlay 再按 view 生成
  eligibility；reader 合并为完整性矩阵：

  ```text
  episode_commit_status
  control_trace_core_complete
  event_family_complete
  stream_integrity_by_stream_id
  replay_capability: decision / control / sensor / full_environment
  training_view_eligibility_by_view_id   # derived overlay/index
  ```

- mandatory writer 失败时 `episode_commit_status` 不能为 committed；某个可选 stream 缺失只使依赖它的
  view/replay level invalid，不能自动废弃仍完整的 BC view；反之 receipt 缺失的数据不能进入 real-RL view；
- 是否因 writer failure 触发真机 HOLD 由 runtime profile 决定，writer 不直接控制机器人；
- control loop 不得阻塞于 Parquet/media I/O，但 bounded async journal 也不能承诺“任意故障下零丢失”。
  required ControlTrace 使用与 media 分离的有界 event journal/episode spool，并分别审计 producer event
  sequence、journal `issued_sequence`、sink `accepted_sequence`、sink `durable_sequence`、durable high-water mark 和明确的
  dropped/gap range：`issued>accepted` 表示 admission/overflow 缺口，`accepted>durable` 表示持久化缺口；
  queue overflow/disk failure 发布 typed writer-health event，不能伪装为普通 media drop；
- shadow profile 可以在 mandatory sink 失败后继续控制但 episode 必须 uncommitted；authoritative profile
  是否进入 HOLD 由显式 runtime policy 决定。finalize 只能 seal 到 durable high-water mark，进程崩溃恢复
  必须截断未完成 fragment 并保留 gap/failure evidence；
- 现有 `modality_validity` 已有 dataset/episode/video 的 fail-closed 基础，v2.4 将其版本化扩展到
  stream/event-family/view/replay 粒度，reader 在 payload I/O 前检查 eligibility；
- dataset episode 的 begin/finalize/abort/commit authority 仍属于现有 frontend/recorder；
  `CollectionWriterRuntime` 按当前代码只拥有 sample→jobs→flush/close，不拥有 episode。ControlTrace journal、
  legacy row、stream/media 都是该 episode owner 下的 subordinate sink，不得各自建立 episode 状态机。
  policy deploy 没有 dataset episode 时使用独立命名的 `RuntimeTraceSession`，不能冒充 committed capture。

进程崩溃后，纯内存 `issued_sequence` 无法证明所有 issued-but-not-accepted 范围。ControlTrace 分级落地：

```text
M4A  deterministic in-memory admission/completeness semantics
M4B  preallocated local spool/WAL 与 crash recovery
M4C  Parquet/materialized views
```

只有 M4B 后才能声明跨进程崩溃的 durable high-water recovery；M4A 只能保证进程存活期间有证据地标记
overflow/failure，并使 capture uncommitted。

当前落地状态（2026-07-19，M4A.1）：在原 M4A generic journal 之上，当前 M3 policy-shadow 事件已改为
explicit top-level allowlist projection；runtime dataclass 新增字段不会静默进入 record/hash，未知字段 fail
closed。trace 新增独立 `policy_action_emission`/`source_validation`，保留唯一 queue-pop lineage；M3
completeness 已从 event presence 提升到 branch/per-lineage 引用与 hash 检查。profile 已拆分 plain dry-run、
policy/replay shadow dry-run、formal real shadow、authoritative real 和 deployable sim；尚无 fixture 的 formal
profile 标记 `planned` 并强制 incomplete。episode envelope 记录 lifecycle owner、binding event、membership
epoch/bound time，bind/unbind 是 typed transition event；authority/gate admission event 也已有 explicit
projection。durable high-water 仍固定为 0，未接 frontend、WAL、writer-health reserve、latency acceptance、
Parquet 或真机 capture；context/composition 和 authoritative event schema 仍须在对应 producer 落地时补齐。
实现仍位于 `teleop_runtime/control_trace/`，以避开现有生产模块 `teleop_runtime/recording.py` 的命名冲突。

随后 scoped M4B 已在无 frontend fixture 中落地：`canonical_loads` 与
`dataset_contracts.control_trace_wal` 定义 64-byte versioned frame header、payload SHA-256、header CRC、sequence
trailer 和 canonical decode；`ControlTraceSpool` 使用固定最大文件、`pwrite+fsync`、严格递增 journal sequence
和 deterministic torn/corrupt-tail scan/recovery；独立 fixed-slot health ledger 在主 spool 满或恢复丢弃 tail 时
保存 durable evidence；`ControlTraceSpoolSink` 只有在 fsync 后才能向 runtime 回报 durable sequence/high-water。
独立 worker 只管理 notify/flush/stop，不拥有 episode；`issue()` 只 admission immutable
fact/sequence/membership snapshot，storage envelope/hash 在锁外由 sink materialize。
这仍不是正式 capture：没有 frontend wiring、episode seal、formal profile fixture、后台 worker lifecycle、长稳/
power-loss 测试或 Parquet reader，`capture_commit_eligible` 继续为 false。

---

## 十、Lifecycle 与监督

`RuntimeSupervisor` 统一观察但不吞并现有 backend 状态机：

```text
sensor health
policy node ready/alive/timeout
context stale/rollback
transport partition
proposal starvation
Arbiter status
backend fault/hold/recovery
writer health
mode/authority transition request
```

第一阶段 real startup/recovery 仍由 `TeleopSession` 和 `run_policy.py` 持有，Supervisor 只接收 typed
health event。后续收敛时必须显式定义降级规则，但 Supervisor 只请求状态变化：

- slow context stale：fast policy fallback、hold 或禁用，由 profile 决定；
- fast policy timeout：使用仍有效 proposal、fallback producer 或 hold；
- remote policy partition：旧 proposal 到 `effective_until` 后失效；
- policy restart：新 producer epoch，旧 queue/context 不得跨 epoch 使用；
- safety FAULT：清空/隔离旧 selected segment，恢复必须由现有安全流程产生新 authority epoch。

execution cancellation 只能使尚未 dispatch 的 queue item、旧 reference segment 和后续 command generation
失效；已经发送给 SDK/backend 的 command 不声明“撤回”。其原 receipt 保留，并另行记录 hold request、
hold result 和 backend state acknowledgement。该 contract 属于 `robot_motion.runtime`，不属于 action
selection 语义。

不能让各 policy worker 自行 dispatch、清 fault 或偷偷选择 fallback。
所有 health/mode/fallback event 携带 `control_domain_id`；一个 sim env 的 timeout 不能污染另一个 env 的
authority epoch。

---

## 十一、建议目录调整

本节只描述长期 ownership；结合当前文件和调用链的最小落地批次、前置门槛与回滚点，见
[`dexteleop_control_core_landing_plan.md`](dexteleop_control_core_landing_plan.md)。实现时不应一次创建
下列所有文件。

当前owner结构及仍可增量扩展的位置如下，不大规模移动入口：

```text
control_composition/
├── activation.py           # formal start/re-arm activation identity + receiver barrier
├── composition/
│   ├── core.py              # immutable input/snapshot/spec/result + deterministic composer
│   ├── strategies.py        # replaceable pose and hand/gripper-qpos strategies
│   ├── pose.py              # arm-pose public surface
│   └── qpos.py              # end-effector qpos public surface
├── observability/
│   ├── difference.py        # source/output difference facts
│   └── feedback.py          # bounded non-authoritative guidance
├── lifecycle/               # handoff/session/rearm reducers and registry
├── events.py                # protocol-neutral composition domain events
├── information.py           # predeclared composition-only information demand
├── policy_node.py           # composition-aware synthetic node protocol
└── runtime.py               # receiver-side evaluation and proposal promotion

robot_motion/
├── action/
│   ├── proposal.py          # ActionProposal / ProposalBuffer contracts
│   ├── snapshot.py          # ProposalSnapshot / reduction policy
│   ├── components.py        # ActionComponentCatalog / selection groups
│   ├── authority.py         # AuthorityPolicySpec / Event / Lease
│   ├── arbitration_input.py # authority/gate/epoch + proposal 的完整 immutable snapshot
│   ├── arbiter.py           # pure deterministic ActionArbiter
│   ├── arbitration_runtime.py # production decision gate；唯一落实 <=60 Hz 的位置
│   ├── persistence.py       # ActionPersistencePolicy / ExecutionDirective
│   ├── safety.py            # typed source/shared/backend safety results
│   └── attribution.py       # decision/control lineage contracts
├── runtime/
│   ├── causal_history.py    # 已有，继续复用
│   ├── observation_cut.py   # ObservationCutRecord / runtime Bundle
│   ├── control_admission.py # per-domain commit sequence/high-watermark/cut
│   ├── producer_epoch.py    # receiver-side activation registry/snapshot
│   ├── execution_gate.py    # immutable gate snapshot/publication barrier
│   ├── dispatch_owner.py    # per-domain token/generation/permit hard gate
│   └── cancellation.py      # execution epoch、异步取消及 acknowledgement contract
├── position_control/
│   └── capability.py        # ExecutionCapabilityProfile；与 Arbiter rate policy 分型
└── deploy/
    ├── observation_view.py  # per-policy allowlist projection
    ├── policy_graph.py      # PolicyNodeSpec / PolicyGraphSpec
    ├── orchestrator.py      # DAG scheduling/context routing，不做 authority
    ├── deployment.py        # DeploymentBindingSpec
    ├── action_adapter.py    # pre-safety RobotAction + PolicyActionEmission seam
    └── canonical_action.py  # 保留兼容 wrapper/typed safety seam

teleop_runtime/
├── action/
│   ├── human_proposal.py    # TeleopCommand -> RobotAction/proposal
│   ├── replay_proposal.py   # replay record -> ActionProposal real wiring
│   └── real_adapters.py     # real observation/health/action sink adapters
├── collection/
│   ├── v24_encoder.py       # decision view + typed event encoding
│   ├── control_trace_sink.py # mandatory journal/durable high-water mark
│   └── stream_sink.py       # native-rate stream sink；复用现有 writer runtime
└── control_trace/
    ├── runtime.py          # deploy/collection 共用 subordinate journal/spool
    ├── lifecycle.py       # EpisodeLifecycleOwner adapter / RuntimeTraceSession
    └── projection.py      # 当前 generic；M4B 前改 explicit event projection

dataset_interface/
└── composition_control_trace.py # composition domain event -> v2.4 projection

control_composition/
├── composition/context.py # generic CompositionContext admission/store/snapshot
├── retention.py           # semantic time-window retention
├── policy_node.py         # transport-neutral typed node protocol
└── runtime.py             # same-cut composition owner；只可发布proposal draft

dataset_contracts/
├── canonical_encoding.py    # ID/hash 的跨 runtime/dataset canonical encoding v1
├── v24.py                   # capability + ControlTrace Core root contract
├── time.py                  # role-specific timestamp/clock evidence
├── control_trace.py         # 当前 generic envelope；M4B 前补 explicit event/lineage schema
├── stream_catalog.py        # native stream、sample ref、retention/integrity
├── stage_schemas.py         # action/reference/command/measurement schemas
├── completeness.py         # commit/completeness/view/replay matrix
├── execution_manifest.py    # execution/deployment/domain bindings
├── source_device.py         # physical teleop device + calibration/adapter contract
├── replay.py                # ReplaySourceContract/action-stage/schedule identity
├── embodiment.py            # topology/component/frame/capability contract
├── training_run.py          # capture/view/learner/checkpoint provenance
├── domain.py                # real/sim domain 与 privileged-field contract
├── policy_graph.py          # serializable logical graph manifest
├── training_view.py         # Learning Extension；actor/critic/DAgger/RL view
├── task_reward.py           # Learning Extension；reward/transition overlay
└── dataset_collection.py    # Domain Extension；capture/overlay/sim2real manifest

dataset_interface/
├── control_trace.py         # readers/index
├── replay.py                # sealed facts/view -> typed replay records
├── streams.py               # native stream lookup/window/alignment
├── training_view.py         # role-aware projection，禁止整 row 直通模型
└── trajectory.py            # variable-dt、15/30 Hz 等训练 view/window

sim/mujoco/deploy/
├── observation.py
├── action_sink.py
└── runner.py

robot_motion/replay/
└── proposal_source.py       # 无 I/O replay schedule/record -> proposal primitive
```

目录名可在实现时微调，但 ownership 不变：

- `robot_motion` 不依赖真机 SDK、MuJoCo、Torch 或 RPC 框架；
- `robot_motion` 可以依赖 `dataset_contracts` 中无 I/O 的 canonical encoding primitive；
- `dataset_contracts` 不得反向导入 `robot_motion`；
- `dataset_contracts` 只放可序列化纯 contract；
- `teleop_runtime` 只实现 real composition/adapters；
- `sim` 只实现 sim adapter；
- `deployments` 继续持有模型 client/protocol/artifact transport；
- `model/` 等 trace/rollout contract 稳定后再加入 learner，不作为当前 runtime 入口。

---

## 十二、最佳实施路径

### Phase 0：冻结当前行为基线

当前状态：已由 landing M0 的无硬件 golden fixture/固定 SHA-256 覆盖；不包含真实硬件时延或安全验收。

- 固定 GUI/headless/deploy 三条入口及当前 real fault/hold/dispatch 行为；
- 生成版本化 baseline manifest，记录 fixture/hash、单 policy output、decoded absolute action、safety result、
  `RobotAction`、`TeleopCommand`、position keyframe、dispatch intent、timestamp/anchor evidence 和当前
  `last_action` stage；不能只依赖会随实现一起修改的测试期望；
- 固定 `DeploymentObservation`、pose/frame/anchor 和 causal snapshot contract hash；
- 为 `DeploymentObservation.model_input()` 冻结 control-private 泄漏负面测试；
- 选取最小 v2.3 fixture，冻结当前主 row、external tactile 去重/引用及 relative training view 行为；
- 明确旧数据中根 `action` 的实际生成位置和已知 lineage 缺口，作为 v2.4 compatibility completeness 基线；
- 不移动当前仍有 caller 的代码到 `legacy`。
- baseline 固定在 `tests/fixtures/control_core/v23_single_policy_baseline_v1.json`，普通测试只验证而不自动
  重写；变更必须显式维护并说明行为差异。

### Phase 1A：在线时间与动作 contract

当前状态：M1A/M1A.5/M1B/M1C/M1D 的 contract 与 deterministic tests 已落地；其中只有 strict actor view
已成为当前 deploy prediction 输入，pure Arbiter 只接 M3 shadow，cancellation/dispatch-owner 尚未接
production worker/backend。因此本 Phase 是“contract 完成、执行迁移未完成”。

只冻结跑通控制闭环不可缺少的语义：

本节是架构层的一组能力；实施时先完成 M0 baseline，再按 landing plan 拆为 M1A
identity/admission/proposal、M1A.5 observation-firewall activation、M1B complete arbitration input/schedule/
persistence、M1C cancellation/epoch 和 M1D dispatch ownership gate，不能一次大批量落地。

- `control_domain_id`/`embodiment_instance_id` 和 role-specific timestamp taxonomy；
- `ClockDomainDescriptor`、reset epoch 及 `CanonicalEncodingSpec v1`/跨层 hash conformance fixture；
- `StreamSampleRef`、`ObservationCutRecord`/`ObservationBundle`；
- `PolicyNodeSpec` identity、`LastActionContract` 和 `PolicyObservationView` firewall；
- `ActionProposalDraft`/receiver admission receipt、per-domain ControlAdmissionCoordinator/committed prefix、
  ProducerEpochRegistry、immutable `ActionProposal`；
- target-time-aware `ProposalSnapshot` 与完整 `ArbitrationInputSnapshot`、DecisionScheduleSpec、
  ActionPersistencePolicy/ExecutionDirective；
- reduction/input hash、
  `ArbitrationEvaluation`/authoritative `ArbitrationDecision` 分型；
- `ActionComponentCatalog`、authority/attribution ports 和 independent decision clock；
- production Arbiter `decision_hz<=60 Hz` 的配置、发布 gate 与审计契约；不把该上限传播到其他 clock；
- source/shared/backend 三层 safety result，以及 action/reference/command/measurement schema 边界。
- `ExecutionGateStatus`、`ExecutionCancellation`/receipt 与独立 execution epoch。
- final dispatch-owner token/generation 与 per-command authority/execution epoch permit。

M1A 先实现 artifact-bound typed `PolicyObservationView` 而不接 production；M1A.5 通过 legacy/strict 双生成
和 tensor/path/shape/value parity 后，才让正式 actor 只消费 strict view，修正
`DeploymentObservation.model_input()` 的 metadata/fault 泄漏。暂不冻结 reward、训练 mask、完整
sim2real sampler 或通用 RPC。

### Phase 1B：最小 v2.4 ControlTrace Core

当前状态：M4A.1 已完成当前 M3/M5 子集的 explicit record schema、policy/human-shadow branch/per-lineage
completeness、profile scope 拆分、authority/gate event projection 和 episode membership provenance；M5 又补入
`HumanCommandEmission`、独立 `SourceAdapterReceipt`、source validation 及 `human_shadow_dry_run_v1` lineage
fixture；M5.2 首先补入旧PolicyContext/context-admission与composition input/snapshot/result，当前已由
generic `composition_context_*` schema/projection/completeness取代运行时写入；旧事件仍可离线读取。
仍无 replay/authoritative lineage schema。scoped M4B 已完成当前 explicit schema 的 WAL codec、bounded
spool、fsync durable ack、tail recovery、writer-health reserve 和小型 admission contention fixture；M5 opt-in
GUI frontend 可启用同一 scoped debug spool worker，但没有 formal dataset episode lifecycle、formal profile、
production episode 自动接线或 ActualDecisionView，不能把本 Phase 称为完整 authoritative capture。
M4B 只能持久化已显式冻结的 schema，不能为尚无 producer/fixture 的 planned event 猜测 payload。

实现 `control_trace_core` capability：

- observation/sample selection event；
- policy inference/context reference event；
- proposal admission/producer epoch、ProposalSnapshot/DecisionScheduleTick/authority+gate snapshot、
  ArbitrationInputSnapshot/computation/evaluation/publication-attempt/authoritative-decision/publication-rejection/
  proposed-or-published ExecutionDirective/legacy-execution event；
- source validation、shared safety、reference/command/dispatch/measurement event；
- execution cancellation/acknowledgement 和 writer issued/accepted/durable high-water mark；
- episode task termination/truncation fact；
- TraceRequirementProfile、ShadowEvaluationView、authoritative ActualDecisionView、logical table locator 和
  completeness matrix。

这一完整阶段只定义第一/二层控制事实和 capture seal；learner/Domain Extension 不进入实现关键路径。后续前移的
Phase 5L 只增加不可恢复 policy evidence 的 pure contract/fixture，不改变这里的 ControlTrace owner 或完成口径。
capture seal 属于 M4B/M4C，不属于已完成的 M4A。

### Phase 2：单策略按新边界运行，保持行为一致

当前状态：M2 pre-safety `PolicyActionEmission` 和 M3 deploy shadow 已落地；legacy safety/command/dispatch 仍是
唯一 owner。M5 已完成 strict human adapter、不可变 pre-merge provenance、默认关闭的 `TeleopSession`
pre-filter/post-dispatch passive hooks，以及复用同一 ProposalBuffer/input snapshot/pure Arbiter 的 human dry-run
fixture；`HumanProposalShadowRuntime` 已由 `HumanShadowFrontend` 接入 canonical GUI 的 arm/Inspire 路径，按实际
profile 构造 runtime identity、component catalog、arm/hand spec，并通过 bounded non-blocking observer queue
生成 component parity/legacy trace。默认仍为 off，双臂 position collision-aware 路径按现有 IK 后 seam 如实
记录 absolute qpos，其他已覆盖路径记录 absolute pose；该过渡边界不代表 Arbiter 已移动到 IK 之前。
DMTAC gripper 仍由 `TeleopSession` 外的独立分支下发，启用 human shadow 时 fail closed。默认 trace 是无
episode membership 的 `RuntimeTraceSession`；可选 WAL 也只是 scoped debug durability，不是 formal capture。
shared execution shadow 尚未开始。

- 拆出 `DeployRobotActionAdapter` 和 typed canonical safety seam，保留旧 `prepare()` wrapper；
- 先用窄 `teleop_runtime.deploy.arbiter_shadow` 包住当前单个 `PolicyDeployRuntime`；完整
  `PolicyOrchestrator` 等单策略 shadow parity 后、确有多节点需求时再引入；
- policy 输出生成 proposal/snapshot/`ArbitrationEvaluation` trace，但先 observe-only，现有执行仍是 owner；
  只有后续 authoritative owner 才发布 `ArbitrationDecision`；human proposal 按
  landing plan 的后续独立门槛接入，不与 policy shadow 同批切换；
- 对比 decoded absolute action、safe action、command 和 position keyframe golden output，并验证当前
  deploy/collection 各自的 `last_action` 语义没有被 v2.4 静默改写；
- 先用 subordinate `ControlTraceJournal` 承担 deploy/collection 共用 event admission/spool；dataset episode
  仍由现有 frontend/recorder owner 管理，`CollectionWriterRuntime` 通过 composition 接入，而不是把其
  `SessionStepResult/CollectorComponents` 依赖带入 policy deploy；同时保留 v2.3-compatible row 作为对照；
- 第一版只引用现有已采集的 RGBD/state/tactile，不要求高频 native ingestion；completeness matrix 必须
  准确说明哪些 stream/replay/view 尚不可用。

M5 余项必须保持被动语义：DMTAC 若纳入 shadow，只能在现有唯一 gripper action 消费/下发位置旁增加
immutable emission 与 legacy-execution tap，不能重新读取输入、再次计算 target 或创建第二个 send loop。
正式 capture 则由既有 dataset episode owner bind subordinate ControlTrace journal；普通 GUI debug session
不得自行升级为 episode owner。

这一阶段不接第二个真实策略、不改变真机 authority、不为当前 real control domain 增加第二个 dispatch port。

### Phase 3：deterministic 多速率/快慢 synthetic 验证

当前状态：M5.2 pure contract/harness 已完成并已继续泛化。当前工作树已有transport-neutral `PolicyNodeRuntime`、
receiver-admitted `CompositionContextStore`/snapshot、internal-only `CompositionInputStore`/snapshot、显式
`TimeWindowRetentionSpec`、版本化`CompositionSpec`与`DeterministicComposer`；
`CompositionInformationPlan`已能预声明latest-value context demand，composition runtime只发protocol-neutral domain
events，ControlTrace projection由外部dataset adapter持有；activation v2已冻结input→snapshot→result的同activation
identity及v1兼容hash。新增的opt-in formal strategy v2已冻结strategy spec、精确action/context/information-plan
input、v4 result和result binding；framework统一生成result identity，strategy只返回status/reason/action及实际
使用的context IDs。MuJoCo runtime现已提供不进入config/hash的显式strategy注入端口，CS6b1h formal pose
fixture已验证同一input snapshot下v1/v2左右臂pose/action hash一致、v2 application binding、v4
trace/proposal lineage及既有authority→safety→reference→dispatch链不变。默认仍走deterministic兼容分支；
下一迁移门是选定一个命名的default-off sim运行profile做持续parity，而不是直接切换现有入口；最终仍要废弃v1，
不长期维护两套production composer。
fixture 验证 15 Hz RGBD stable-ID reuse、100 Hz tactile latest sample、10 Hz slow context、40 Hz fast residual、
context sequence rollback/stale/restart/cut isolation，以及 30/45/60 Hz Arbiter。v1 composer 只支持 component
union、qpos base-plus-bounded-residual及weighted SE(3) pose fusion；当前唯一具体runtime binding在MuJoCo，
composer authority默认关闭，尚无真机或OpenPI正式接线，也没有PolicyOrchestrator、通用DAG/RPC或真实第二模型。
普通 proposal 的 `semantic.v1` payload/hash 保持不变，带composition lineage的proposal使用显式
`semantic.v2` domain；activation尚未作为proposal独立字段存在，但已通过v2/v4 result identity间接进入
`policy_output_id` hash lineage。source eligibility/liveness纯层和
default-off MuJoCo shadow已经落地，但真实设备、replay与policy worker的状态adapter尚未接线；source/embodiment
可替换性仍需后续多backend conformance，不是正式production binding结论。

使用 injected fake clock 和 fake producers：

```text
15 Hz fake RGBD
100 Hz fake tactile/state
10 Hz fake slow context policy
30 或 40 Hz fake fast action policy
30/45/60 Hz configurable Arbiter
60/75 Hz reference
120/150 Hz command
```

该阶段只实现最小 synthetic topology contract/harness：`PolicyNodeRuntime` protocol、
`CompositionContextRecord`/time-window store、`FakeContextPolicyNode`、`FakeActionPolicyNode`、一个 bounded-residual
`DeterministicComposer` fixture 和确定性 context edge；
不接 `run_policy.py`，不实现通用 DAG、RPC 或 production `PolicyOrchestrator`。

必须验证：

- 同一 RGBD 重用时 frame ID 不变、age 正确；
- fast proposal 引用准确的 observation cut 和 slow context version；
- action/context node 使用不同 runtime，context node 不出现 action queue/decode；
- proposal produced/available/effective time 和 chunk `target_effective_at` 正确；
- snapshot reduction、proposal snapshot ID/hash 和 replay candidate set 一致；
- 相同 `ArbitrationInputSnapshot` + arbiter contract/implementation/selection-policy hash 得到完全相同 result
  hash；snapshot 后到达的 proposal、authority event、
  gate/producer epoch transition 只影响下一计算，而 live fault/cancellation 可阻止旧 result 发布；
- base/residual composition input snapshot、缺失/late fallback 和 lineage 可重放，不使用隐式 freshness weight；
- 同一 chunk 多未来 keyframe、重叠 effective interval、replace/append 和 late chunk 按 target-time policy
  确定性选择；
- 不使用未来 sample/context，旧 context 乱序到达不覆盖新 context；
- late inference、empty queue、stale context、producer restart 和 authority takeover 结果确定；
- 非整数频率组合不依赖 frame-count modulo；
- `decision_hz>60` 配置 fail closed；高频 proposal/event burst 被确定性 coalesce，且 reference/IK/command
  仍可独立使用高于或不同于 Arbiter 的频率；
- 单进程运行和 envelope serialize/deserialize replay 得到相同 decision；
- mixed-source component attribution 可完整重建；
- ActualDecisionView 保留 variable `dt`，FixedRateCompatibilityView 显式记录 resampling；
- 多个 `control_domain_id` 的 proposal/authority/producer epoch/replay 互不污染；
- 15 Hz RGBD sample 被 30/40 Hz cut 通过同一 stable ID 正确复用；
- 事实层 `selected_action`、proposal、safe/commanded/measured 阶段不混名，裸 `action` 只由声明 stage 的
  view 生成，actor view 无法读取 control-private 字段；
- 不同缺失事件/stream 产生正确的 view-specific completeness，而不是一个全局 training-valid。
- `no_action` 不刷新旧 segment expiry，hold/fault 产生 cancellation directive；

实际 IPC/RPC 可以延期，先证明 contract 与 transport 无关。

### Phase 4：可选 native stream capability（对应 landing M7A）

该能力可在 landing M5–M7A 之间独立推进，不是所有单策略 M6 authority 迁移的全局前提。若目标
ExecutionManifest、PolicyObservationContract 或 TraceRequirementProfile 声明依赖某个 native stream，则该
stream capability 必须在对应 shadow/authority gate 前完成；不依赖高频 tactile 的当前单策略 profile 不因
此被阻塞。

- 在同一 `CollectionWriterRuntime` 中增加独立 stream sink，不让主 decision row 触发高频数据是否存在；
- 先接 state/tactile 的可控频率 fixture，再接真实 30 Hz+ tactile/state retention profile；
- 验证一个真实 RGBD frame 只编码一次、window/cut 只引用 stable sample IDs；
- 保存 produced/written/dropped/gap/rate/latency 和 decimation/filter contract；
- 不把“支持最高 1000 Hz”误写为第一版必须无条件全量保存 1000 Hz raw tactile。

### Phase 5：deployable sim

- 新增 `sim/mujoco/deploy` observation/action adapter；
- 先跑单策略 profile，再跑 fake slow/fast profile；
- 将 v2.3 compatibility action 和 v2.4 canonical action replay 都转换为 replay proposal，通过同一
  Arbiter/safety/execution 链在 sim authoritative 运行；command/servo packet 只做 audit，不跨 embodiment
  直接发送；
- 复用现有 actuated position sender、position rates、Arbiter 和 ControlTrace；
- 优先验证 timestamp taxonomy、proposal snapshot、stage transformation、ControlTrace Core 和
  control-domain isolation；
- sim 输出同一 v2.4 trace/stream core，并记录最小 backend/physics/randomization execution facts；完整
  DomainContract、privileged training view 和 sim2real sampling 延后到 Phase 7；
- 用 deterministic golden replay 比较 sim/real shared boundary，不要求 backend qcmd/measured 数值相同；
- 保留 CubeLift native RL 路径，不强行统一其高性能内部 loop。

M5.5d 已完成测试驱动的 10 Hz context/base、40 Hz residual、30 Hz Arbiter fixture；M5.5e1 已进一步完成确定性
受限 async harness（latency/jitter/deadline/stale/cutoff/restart/burst/cancellation competition），所有输出仍经既有
receiver admission，且 context freshness 与 composition input 在同一 causal cut 冻结；它不等同于真实线程、GPU
inference、generic DAG、RPC 或 production multi-policy。
M5.7e2-c 的 sim segment producer 完成后，M5.5e2 才做 formal sim capture。第一阶段使用测试侧composition fixture，
但只负责wiring，必须复用真实MuJoCo runtime、ControlTrace/WAL、唯一episode owner、seal/commit与committed reader；
hash不得包含测试路径/case/tmpdir/object identity。其成熟度只称`formal_sim_capture_fixture_complete`。`sim`不能import
`teleop_runtime`，也不能复制journal/lifecycle；production application root在fixture证明依赖关系后再决定。

当前该测试侧composition已在`dexteleop310`中同时通过deterministic facade和真实MuJoCo actuated backend。scheduled
completeness按分支验证：START前取消不要求也不允许伪造activation/frame；accepted segment仍必须闭合
admission/candidate/reservation/terminal。seal后、commit前进程终止留下明确uncommitted episode，默认reader拒绝训练消费，
审计模式可从seal/WAL恢复。该结论仍不代表production sim entrypoint、长稳或目标机性能验收。

首个M6 release candidate冻结为`m6_candidate_policy_arm_qpos_v1`：policy是唯一motion authority；operator/runtime
safety supervisor通过独立`SAFETY_CONTROL`执行HOLD/ESTOP/manual stop，operator-authorized recovery state machine通过
`LIFECYCLE_CONTROL`执行recovery/home/rearm，human motion proposal不适用。双臂coupled qpos、30 Hz decision、scheduled
segment；planner/qcmd cadence profile化，当前sim证据为120 Hz。60 Hz只可作为已有activation/velocity/capability timing
语义，不得记成不存在的reference producer。hand、DMTAC、replay、pose impedance、fix-arm human与human takeover均以
manifest/component catalog hash显式排除。

当前工作树已将该候选实现为backend-neutral `ControlExecutionManifest`。它作为scheduled formal episode的WAL依赖事实
持久化，并校验component catalog/action schema lineage；它不携带dispatch token，也不能替代SafetyMigrationMatrix或
ExecutionAuthorization。policy safety override已另行收敛为启动期static typed admission，RM65 quick-stop已具有bounded
operation与per-tick one-shot safety permit的pure/fake fixture；两者均尚未接production caller或真机SDK seam。

M5.8-a 的arms-only调用点现具有分层成熟度事实：generic `DispatchSeamInventoryManifest`冻结profile、实现路径、调用
阶段、permit/operation要求和集成成熟度；RM65-specific builder则按当前代码登记`send_command` target admission、
background worker dequeue、最终`_send_joint_target()`、qpos-init home、local rearm、force-position startup、quick-stop
operation/tick、hold/disengage/fault/ESTOP/rollback/close stop与disconnect。相同`_send_joint_target()`分别登记普通MOTION、
quick-stop `SAFETY_CONTROL` tick和home `LIFECYCLE_CONTROL`三种语义，避免按Python方法名错误合并权限。三条共享
joint-target调用及七条hold/disengage/fault/rollback/startup/ESTOP/close-stop SDK seam已在真实函数内加入default-off typed
adapter，并经逐调用one-shot fake gate验证，10条状态为`pure_wrapper_validated`；其余6条entry仍为
`inventoried_unwrapped`。production未注入provider，worker epoch也未执行，故全部required seam仍属于unmediated，结构
完整不表示硬件已授权或M5.8-b/M6已放行。首个manifest显式排除的Inspire/DMTAC不会被静默加入该清单，其各自profile
仍需独立inventory。

在此之上已有一个严格default-off的`LegacyControlDomainExecutionContext` composition prerequisite。它将一个manifest、
对应inventory/assessment、唯一`DispatchOwnerRegistry + legacy token`、唯一`ExecutionEpochRegistry`和
`ExecutionEpochGuard`绑定到同一session/domain/clock/group作用域；当前唯一模式是`fake_only_unbound`，明确没有backend
binding或hardware authorization。fake final-call preparation先用同一epoch registry核验`ExecutionWorkItem`的domain、
execution epoch和group generation，再签发一次性motion permit，由`DispatchSeamGate`复查owner/authority/execution。
logical target admission不能被当作final motion seam。该顺序只证明composition可行：group check与owner gate分属两个锁，
尚未成为真实worker内紧邻SDK调用的原子线性化点，因此仍不能升级任何inventory entry为`production_mediated`。

M1C 的`software_queue` acknowledgement现另有pure worker-drain证据层：每个
`ExecutionCancellationParticipantSpec`声明worker种类、影响group与可提供证据；`WorkerEpochDrainReceipt`必须证明worker
已观察新execution epoch、旧epoch in-flight计数为零且不会再产生旧epoch调用，单纯`queue.clear()`不能通过。一个
`WorkerEpochDrainAggregate`只有收到本profile全部required participant的accepted receipt后，才能投影成现有
`CancellationAcknowledgementTracker`中的单个`software_queue` ack。该contract本身不会停止线程。M5.8-b1已在
`PositionDeployExecutor`和`RM65ArmBackend`中增加严格default-off的typed work carrier/checkpoint/drain方法，并以真实
worker代码路径的fake阻塞SDK fixture验证：reference submission绑定epoch后才入队，position dequeue/callback、backend
target admission/background dequeue/pre-SDK均保留并复查同一个work envelope；ordered coordinator必须先drain上游，再
drain左右sender，旧SDK调用未退出时不得签发receipt。默认legacy构造不注入guard，因而数值和调用路径不变。

这仍不是production participant evidence：fake-only composition已能将唯一owner/execution registry注入三个真实worker
class，canonical policy builder也已有无CLI开关的default-off seam，但真实IP会fail closed且硬件没有运行该模式。
普通motion item不得被quick-stop沿用；当前pure/default-off实现已增加独立
`safety_backend_command`，强制引用新execution epoch中的`RM65_QUICK_STOP` operation和逐tick evidence，fault latch只对
这种显式safety work放行而继续阻止普通motion。晚到ack不能消除deadline FAULT；operation/tick、worker receipt/aggregate
及fault escalation已有explicit ControlTrace schema/projection，但尚无canonical真机producer/WAL evidence。完成production
现场production binding与验证前，drain plan继续
标记`scoped_unbound`，不能投影真实software-queue acknowledgement；当前RM65构造器对
`connect_hardware=True + partial epoch/drain binding`直接fail closed。

ControlTrace持久化故障与安全取消已明确解耦：worker drain先完成，再投影receipt/aggregate；sink异常形成显式partial
admission report并使相应capture/eligibility不完整，不得静默忽略，但也不得阻止HOLD、FAULT或旧epoch drain。这个隔离
只证明错误处理contract，现场runtime尚未产生durable WAL evidence。

首个arms-only候选还具有确定性的scoped `SafetyMigrationProfileMatrix`：matrix强绑定同一execution manifest、exact RM65
inventory及其结构评估；9条included rule冻结目标disposition并引用typed contract/inventory evidence，在
`m5_8a_inventory`为ready，但在`m5_8b_legacy_mediation`和`m6_shared_authority`分别保留production caller、worker guard、
real receipt/ack、hardware threshold/parity等具体blocking reason。hand、DMTAC、replay、pose impedance、fix-arm human与
human takeover由manifest exclusion在三个gate均明确not-applicable。该matrix是“可审计地完成盘点”的证据，不是profile
owner硬件签署，不改变全部RM65 seam的`inventoried_unwrapped`状态。

### Phase 5L：Learning Observability（与 M5.7/5.5e 并行）

只实现 transport-neutral pure contract 和一个端到端消费 fixture：policy-observability capability、inference-bound
adaptation state、generative sample evidence、HistoryActionContract 和 decode/optional-encode codec identity。随后
为一个明确的 model adapter 增加 opt-in evidence hook，并用 policy+human raw event 构造双历史/sequence fixture。
没有 concrete adapter 时停在 pure contract，不猜 noise/fast weights；没有 encoder round-trip 时不声明 correction
chunk 可用于 inversion。该阶段不依赖 M6 authority，也不引入 learner/Torch。

### Phase 6：真机 shadow 与单策略 authority 迁移

按风险逐步进行：

1. M3 仅做 dry-run/replay/non-capture single-policy shadow；M4B durable spool 完成后才允许正式 real shadow
   capture；
2. explicit human override/per-group override shadow；
3. M5.7b 先冻结 per-group/epoch 的 previous-safe-committed/measured-rebase anchor lifecycle；M5.7c1 已用
   policy 唯一 emission、实际 observation 和旧 post-safety result 完成 legacy-aligned arm pose/qpos numeric 双算；
   旧 anchor 的 whole-vector finite 条件和 qpos frame representation 差异均显式记录；
4. M5.7e1 已冻结 direct-command/pose/qpos/trajectory `ResolvedExecutionTarget` admission 与 per-group safe-action
   projection；其“reference target admission 即刻 commit”的 pure fixture 只适用于明确声明足够 guarantee 的窄
   profile，不能用于 future/lookahead segment。M5.7c2 已证明 measured rebase→safe→typed shadow admission 的基础
   连续状态，但 M5.7e2 必须将其升级为 target-time-aware commit timeline；M5.7d-a 已接 human temporary limiter 的
   exact legacy anchor/filter evidence 与 pose pure parity，仍需 qpos/position profile；
5. M5.7e2-a 冻结 ordered endpoint parent、segment admission/terminal、admission guarantee 和 pending/effective commit；
   e2-b 只对 legacy position worker 做 passive instrumentation，e2-c 接 sim authoritative，e2-d 在真实 producer
   存在后再冻结 ControlTrace/completeness；随后 M5.7f 按 execution profile 关闭 SafetyMigrationMatrix；
6. M5.8-a 先做所有 backend seam inventory 和无行为 permit wrapper；M5.8-b 取得硬件授权后才做 legacy-owner real
   mediation、epoch 二次检查和 latency/ack 验收。整个 M5.8 中 legacy 仍是唯一 motion owner；未验收的 measured
   divergence 可在该 legacy-owner gate profile 中显式 disabled，但 M6 前必须具备有效阈值和硬件 evidence；
7. single-policy + human authority policy 的 dry-run/sim 互斥 owner 切换；
8. 通过独立安全验收后，M6 只切已验证 control domain 的 dispatch token owner，不在同批首次接线；
9. fast policy、slow context→fast proposal 的 production shadow 和 authority 切换作为后续独立门槛。

replay 在本阶段先做 proposal/selected/safe/command parity shadow；真机 authoritative replay 是独立高风险
授权，不由 single-policy authority 自动覆盖。它必须使用相同 embodiment/frame/action preflight、互斥 owner
和 cancellation contract。

shadow 中未选中的 proposal、context、decode anchor 和 per-component evaluation 也必须写入 trace；actor
observation 仍不得包含 Arbiter/safety/reward 内部状态。episode finalize 必须验证 mandatory lineage 和
stream manifest；completeness matrix 必须按用途拒绝不完整 view。

### Phase 7：Learner/Artifact/Domain Extension 与 model package

- sealed capture 之外实现 immutable reward/annotation/training-view overlay；
- `dataset_interface` 从统一 ControlTrace/native streams 生成 slow/fast/DAgger/RL views，不向模型返回
  未投影的完整 row；
- 实现版本化 reward recompute、`RewardAttributionAdapter`、expert/policy masks 和 variable-`dt` rollout buffer；
- 冻结完整 `DomainContract`、`DatasetDerivationManifest` 和跨 homogeneous real/sim roots 的
  `DatasetCollectionManifest` sampler；
- 检查 action/pose/layout/view/domain hash，并隔离 sim privileged critic/reward streams；
- 某个 view 所依赖的 observation/context/proposal/safety/receipt lineage 缺失时，只拒绝该 view，不以
  单一全局布尔值误杀其他仍完整用途；
- 消费 Phase 5L 的 inference/history/codec evidence，离线生成 intervention segmentation、action inversion、support
  assessment 和 safety-projection eligibility；这些都是带 parent/spec hash 的 derived overlay；
- 最后再增加 BC/DAgger/RL learner 和 sim/real rollout collector；Torch/RL 依赖保持 optional。
- learner 必须输出 `TrainingRunManifest + PolicyArtifactContract`，把 capture/overlay/view、代码配置、
  checkpoint digest 和 embodiment compatibility 绑定起来；
- 新 artifact 必须完成 offline view reconstruction、deterministic replay/sim proposal parity、目标 binding
  preflight 和 shadow，随后产生引用该 artifact/training-run hash 的新 capture，才算闭环验收完成。

---

## 十三、近期明确不做的事情

- 不把所有 sensor 或 policy 强制 resample 成一个 15/30 Hz tensor；
- 不让 proposal/context 复用含义不清的 `source_timestamp`，也不把远端 producer clock 直接用于本地 eligibility；
- 不在 native v2.4 全局定义 `observation.last_action` 或默认 actor target；
- 不把 `ActionArbiter` 绑定到 RGBD、某个 policy 或 reference clock；
- 不为 fast/slow policy 各建一套 safety、position executor、dispatcher 或 writer；
- 不在 Arbiter 中重新解码 relative action、查 measured anchor 或做任意 pose blend；
- 不让 Arbiter 在调用期间读取 mutable store/frontend 状态，也不把 snapshot 后的 fault 解释为旧 result 仍可
  执行；纯 result 可重放，publication/dispatch eligibility 必须重新校验；
- 不把 priority、authority、validity、frame/component、persistence 或 safety 语义藏入 audit metadata；
- 不把硬安全反射包装成普通 policy proposal；
- 不把 proposal、selected、safe、commanded 和 measured action 压成一个含义随 reader 改变的 `action`；
- 不把所有 policy/event/高频 modality 展开进单个超宽 parquet row，也不因主 row 未触发就声称高频流不存在；
- 不从 v2.3 最终 action 数值反推没有记录的 proposal、Arbiter、safety、reward 或 dispatch lineage；
- 不让 actor reader 读取完整 dataset row，或把 sim privileged state、reward、fault/dispatch detail 意外泄漏给模型；
- 不把 placement/transport 写入 logical PolicyGraph hash，也不把 policy artifact 写入 DomainContract；
- 不修改 sealed capture 来追加 reward/annotation，不把派生 mask 冒充原始事实；
- 不把 trace logical family 固化为每 episode 六个物理文件；
- 初期不把 pose/source contract 不同的 real station 和 sim domain 强行 append 到同一 dataset root；
- 不立即实现通用分布式 DAG/RPC 框架；只先冻结 transport-neutral envelope；
- 不直接让 sim import `teleop_runtime.deploy.run_policy`；
- 不把 recorded SDK/servo command 当成跨 embodiment replay action；replay 默认回到 canonical proposal stage；
- 不以“设备无关”为由忽略 SourceDevice/Embodiment compatibility preflight，或对不兼容 DOF/frame 静默
  reshape/截断；
- 不把当前仍有 production/test/docs caller 的 collection、deploy 或 compatibility shim 移入 `legacy`；
- 不在 ControlTrace Core 稳定前启动真机 policy-only；不在 Learning Extension/对应 view eligibility 完成前
  启动真机 RL 训练。

---

## 十四、阶段验收标准

### Contract/单策略兼容完成（当前：contract 基本完成，production 迁移部分完成）

M0–M3 已覆盖当前单 policy deploy 的 baseline、strict observation、emission 和 passive arbitration；M1C/M1D
仍无 production worker/backend caller，collection/human/replay 也未迁入 proposal core。因此以下列表是本阶段
最终验收，不应全部勾为当前已满足。

- baseline manifest 可重复生成；canonical encoding 在 runtime record、dataset record 和 transport round-trip
  后得到相同 content hash；
- 当前单 policy 的 model view、v2.3 decode、chunk queue 和 canonical command golden 行为不变；
- v2.3 wall-time causal hash 不变，v2.4 role-specific/local-monotonic time 通过 adapter 增量提供；
- deploy 与 collection 当前不同的 `last_action` stage 均被显式记录，没有被统一成错误的 previous selected；
- actor observation 不含 Arbiter/safety/trace/reward 私有字段；
- producer 不能自报 authoritative available time；accepted proposal 有 receiver admission receipt，rejected draft
  永不进入 snapshot；proposal 是 immutable absolute `RobotAction`，可追溯 cut 和 representation receipt；
- proposal/authority/gate/epoch event 共享 per-domain committed admission sequence；完整 snapshot 只覆盖一个
  time+sequence committed prefix，不包含 in-flight Store write；
- 当前 real control domain 没有新增第二个 production loop/dispatch port 或 writer lifecycle。

### 多速率/多策略 contract 完成（当前：M5.2 pure synthetic 完成，production 未开始）

- action/context node 使用共同 `PolicyNodeRuntime` 协议，但只有 action node 拥有独立 action queue/decode；
- logical PolicyGraph hash 不因 in-process/shared-memory/RPC binding 改变；
- RGBD 15 Hz、tactile/action 30 Hz+ 可以因果组合，不伪造 frame；
- Arbiter 在独立 decision clock 上确定性运行，proposal timeout/stale 时 fail closed 或按显式 profile 降级；
- production Arbiter 的配置值和实测 decision publication rate 均不超过 60 Hz；大于 60 的配置拒绝启动，
  event burst 不绕过 gate，其他链路没有被错误套用这一上限；
- fast proposal 能精确引用 slow context version，重放不依赖隐式“当前最新值”。
- Composer 使用 immutable input snapshot；base/residual 缺失与 late fallback 由版本化 spec 决定，Arbiter 不
  做 blend。

### ActionArbiter/trace 完成（当前：纯 contract + policy/human/replay scoped shadow，未 authoritative）

当前已满足 immutable input/pure result/60 Hz gate 的单元测试、policy `would_select` shadow，以及 canonical
GUI arm/Inspire 的默认关闭 human `would_select`/legacy-execution trace。M5.3/M5.3a typed replay dry-run 已接
三态 source/target assessment 与 `would_select` lineage；M5.3b 已在 canonical GUI 接入默认关闭、无 authority
的 source-frame consumer/RuntimeTraceSession，可选 debug WAL 仍不是 formal capture。DMTAC human branch、shared
execution、production backend token、cancellation acknowledgement 和 formal durable capture 均未接入。

- human、policy、replay 的每个 component 都能追溯 proposal、owner、decision reason 和 authority epoch；
- decision 引用完整 `ArbitrationInputSnapshot` ID/hash 和稳定 ProposalSnapshot；相同 input、arbiter contract/
  implementation 与 selection-policy hash 得到相同 result hash，snapshot 后状态只影响下一计算，live
  safety/cancellation 可阻止旧 result 发布；
- no-action 不延长旧 segment，hold/fault 产生带 execution epoch 的 cancellation directive；
- per-group no-action/segment expiry 不阻塞独立 group，coupled group 原子；不支持局部 stop 的 backend 显式
  升级 cancellation scope；
- selected、safe、reference、commanded、receipt、measured 使用各自 schema 并可多对多关联；
- 每个 control domain 的 authority/epoch/dispatch lineage 隔离；
- 最终 backend seam 校验唯一 motion dispatch-owner token/generation 与 command authority/execution epoch，legacy/new
  owner 不可能同时有效；
- motion、安全、lifecycle permit 分型；motion token revoke 后安全 hold 仍可执行，普通 authority 不能借安全/
  lifecycle permit 绕过 gate；
- HOLD/FAULT 的 execution cancellation 可使旧 reference/command segment 因 execution epoch 失效，并能取得
  reference、command 和 backend acknowledgement；
- TraceRequirementProfile 按 runtime mode 区分 required/optional/not-applicable；required trace 不静默丢失，
  媒体 drop 有明确记录。
- computation、publication attempt、successful decision 和 rejection 分型；只有 successful publication 产生
  selected_action、ActualDecisionView 和新 reference segment。

### v2.4 ControlTrace Core 完成（当前：M4A.1 explicit schema + scoped M4B durable-spool milestone）

当前已具备 M3 policy-shadow 与 M5 human-shadow 子集的 explicit payload allowlist、queue-pop/source/proposal/
decision/legacy 逐 lineage 检查、profile scope、membership provenance、WAL/spool/fsync/tail recovery 和 health
evidence fixture，并有 M5.2 context/composition explicit record 与 deterministic fixture、M5.3 replay
source-frame/record/receipt/proposal 以及 M5.3a/b binding/eligibility/GUI conditional-evaluation branch-aware
fixture。M5.5a 又由实际 deployable-sim producer 冻结 authoritative publication/decision/directive/shared-safety/
command/dispatch/measurement schema，并可重建 ActualDecisionView；但 DMTAC human、replay authoritative/formal
episode、context/composition branch-aware completeness、formal sim capture/seal 和 native stream 验收仍未满足。

- v2.4 继承 v2.3 absolute storage/relative view/causal anchor；事实层 `selected_action` 只表示 authoritative selected
  canonical pre-safety intent，裸 `action` 只由声明 `action_stage` 的 materialized/training view 生成，所有
  下游 action stage 使用独立名称和 ID；
- native v2.4 不全局定义 `observation.last_action`，也不定义默认训练 target；
- 每个有效 selected component 都可回溯 producer/proposal/observation cut，未选择 proposal 也能用于
  DAgger、intervention 和 reward attribution；
- ActualDecisionView 只来自 authoritative decision 并保留真实 variable `dt`；shadow 使用独立
  ShadowEvaluationView，fixed-rate view 显式记录 resampling 和 source decision IDs；
- 声明 `native_streams` capability 后，RGBD、tactile、state、human input、reference/command 等按 native
  stream 存储和引用；同一真实 frame 被多个 policy cut 重用时只存一次，不伪造新帧；
- v2.3 compatibility view 明确标记 unavailable lineage 和 completeness matrix，不制造不存在的 Arbiter；
- episode finalize 检测悬空 ID、因果越界、stream gap/checksum 和缺失 mandatory trace，并分别给出 commit、
  core、stream、view 和 replay eligibility；
- mandatory journal 的 issued/accepted/durable sequence、overflow/crash recovery 和 seal high-water mark 可审计，
  writer failure 不阻塞 control loop，也不静默提交不完整 episode；
- capture seal 后不可变，trace logical family 的物理 partition 可由 manifest 演进。

### Learning/Domain Extension 完成（当前：M4L.1 pure observability 已开始，learner/domain 未开始）

- `dataset_contracts.learning_observability` 已冻结 artifact-bound state/sample/inference join、history 与 codec
  identity，并有 deterministic consumer fixture；尚无真实 model adapter producer、ControlTrace event 或 blob store；
- 事实层不定义默认 target；BC/DAgger/fast/slow/RL target 和派生 mask 均绑定 TrainingView/derivation hash；
- reward/annotation 位于 parent-hash immutable overlay，可重算 variable-`dt` transition，不修改 capture；
- DomainContract 不含 policy topology/artifact；EpisodeExecutionManifest 单独记录本次运行配置；
- sim privileged data 只进入显式 critic/reward/audit view，DatasetCollectionManifest 校验 real/sim 兼容 hash；
- TrainingRunManifest 可从 artifact 追溯 capture/overlay/view/code/config/checkpoint digest；部署产生的新 capture
  反向记录 deployed artifact/training-run hash，BC/DAgger/RL 至少各有一个 deterministic round-trip fixture。

### source/replay/embodiment 可替换闭环完成（当前：typed replay dry-run 起步，闭环未完成）

- human、policy、replay 都只通过 ActionProposal 进入 authority；现有 replay reader/scheduler 被复用，但最终
  authoritative replay 不保留独立 dispatch owner；
- 替换两个 contract-compatible source-device adapter 时，canonical proposal/decision 语义不变，raw source
  provenance/adapter receipt 可区分；
- 替换两个 compatible real/sim embodiment adapter 时，proposal/decision/TrainingView contract 不变，
  command/measurement lineage 按各自 schema 变化；不兼容 binding 在 dispatch 前 fail closed；
- sealed capture 可生成 replay record 和 TrainingView；训练 artifact 经 offline→sim/replay→shadow→authority
  gate 回到同一控制链，并产生下一代可追溯 capture。

### sim/real 共享完成（当前：M5.5a/b 单策略与 verified replay sim 已接 shared core，real/multi-policy 未完成）

- real/sim 使用同一 ObservationCutRecord/Bundle、PolicyNode、ActionProposal、Arbiter、absolute `RobotAction`、
  position 和 ControlTrace contract；
- backend 只替换 observation producer、clock、action sink 和 receipt；
- task-native sim 与 deployable sim 目标清晰，不互相伪装；
- 真机 shadow 之前所有 shared contract 都可在 deterministic sim/dry-run 重放验证。

---

## 最终判断

DexTeleop 下一阶段不应围绕“把 15 Hz policy 插入 60/120 Hz 控制 loop”继续扩展，而应把现有基础
提升为一个多流、异步、因果、可追溯的统一执行系统：

> RGBD 15 Hz 只是一个 sensor stream；slow/fast policy 是 topology 中不同的 producer；
> `PolicyDeployRuntime` 是 action policy node 的现有 stream core；ActionArbiter 在独立 decision clock 上
> 按 production decision policy v1 以不超过 60 Hz 消费 versioned proposal；该限制不约束其余链路，
> 60/120/150 Hz 仍可作为 selected
> action 以下的执行展开；所有 source、
> context、decision、safety 和 execution 通过稳定 ID 写入 ControlTrace。

这套系统最终形成的不是一次性数据导出，而是：

> source/embodiment adapter → unified control + capture → replay/training view → versioned artifact →
> compatibility preflight + shadow/authority → next capture。

真机 preflight 的证据等级必须保持分层：ping/TCP port 可达只属于 network evidence；SDK session/state read、
backend ready、safety/lifecycle state、worker drain、stop acknowledgement 与 motion authorization 是后续独立事实。
某一层通过不得自动提升下一层。站点配置若仍含 TEST-NET/placeholder 地址，真机运行必须 fail closed 或要求显式
站点 IP；不得为了让 preflight 通过而把占位配置静默改为可运动地址。

2026-07-21 的远端现场证据已将 `SDK session/state read` 子层关闭：严格只读入口在两侧各完成 5 次
`Get_Current_Arm_State/Get_Current_Tool_Frame`，返回码均为 0，5 轮 `arm_err/sys_err` 也均为 0；未开启 background streaming/force-position，且未发送
运动命令；两侧均加载明确的 `DMTAC_M` TCP profile。读取 qpos 均在当前 active URDF 限位内，最小余量为左 J3
0.069 rad、右 J3 0.342 rad。该记录不包含 backend motion-ready、急停/使能链、drain、stop acknowledgement 或
motion authorization；尤其左 J3 余量较小，任何后续运动批次都必须重新读取 fresh state 并在调用前完成 URDF 与
目标方向检查。

在上述只读层之后，首个获授权的状态改变被进一步缩窄为`M5.8-b0 stationary stop-only`：每侧最多一次
`Arm_Slow_Stop`，由exact safety-control one-shot permit保护；同一组合中的joint-target adapter无条件拒绝，lifecycle、
force-position、background streaming和其他stop seam也不在授权范围。该批只验证真实final-call gate、SDK返回和
调用前后状态，不提供motion authority，不证明worker drain或机器人已经物理停止，也不能作为M6放行证据。
现场执行已验证两侧one-shot permit与真实`Arm_Slow_Stop` final call：dispatch均accepted、SDK均返回0，时延为
10.758/8.582 ms，调用后qpos变化不超过`5.24e-5 rad`，且binding继续声明`motion_authorized=false`。该结果只作为
scoped final-call evidence。首轮控制台payload没有展开authorization有效期和permit/validation receipt ID，故标记为
partial lineage；输出schema随后已补齐，但不会为了元数据重复调用硬件。本证据不得提升为完整M5.8-b、drain、
measured-stop或M6 authority。

后续无硬件M5.8-b1已关闭“旧work identity在异步边界丢失”的代码缺口：`ExecutionWorkItem`作为typed字段随
reference/command传递，RM65 mailbox保存派生的per-side backend work，而不是在最终调用时读取当前epoch后回填。worker
drain取得其自身发送锁后才可声明old-epoch in-flight为零，并清除旧target；三个participant的聚合顺序固定为上游
reference worker后左右backend sender。该能力默认关闭，尚未由production builder启用，且没有替代stop、SDK ack或
measured-stop证据。

M5.8-b1-real-A 进一步建立了一个独立的硬件诊断 composition，但没有改变上述 production 结论。其 profile 为
`m5_8b1_real_a_idle_worker_drain_stop_only_v1`：只允许无target的真实background worker、同一execution epoch下的
三participant drain，以及drain完成后每侧一次`Arm_Slow_Stop`；MOTION与LIFECYCLE_CONTROL始终未授权。
该profile的authorization/registry/token/WAL均为进程局部且运行结束即失效，不写入正式deploy配置，也不改变
`m6_candidate_policy_arm_qpos_v1`的manifest/hash、默认cadence或safety参数。其trace属于runtime diagnostic，明确
不可作为公开v2.4 capture commit或M6 authority evidence。后续真实操作必须重新构造相应execution profile和授权，
不得继承这个窄批次为排除风险而采用的保守配置。

该profile的现场运行现已完成：两侧真实background worker分别运行103/93个idle tick，均未激活target且
joint-target send count为0；execution epoch由0推进至1，三个required participant均生成accepted drain receipt，
aggregate与software-queue acknowledgement完整。drain后两侧独立Slow-Stop permit和SDK调用均accepted/result 0，
trace的10个issued fact全部durable，且capture仍不可commit。由于本次没有target、discarded work或in-flight旧调用，
证据等级严格限定为`scoped_idle_worker_drain_stop_field_evidence`：它证明真实idle worker执行了同一epoch barrier，
不证明target-bearing worker、pre-SDK旧调用竞争、physical measured-stop、production composition或M6 authority。

遥操设备和本体可以替换，核心闭环不改；只有 source calibration/retarget、embodiment execution binding 和
compatibility evidence 随硬件变化。若 canonical action/frame/component 不兼容，系统明确拒绝复用，而不是
以“设备无关”掩盖需要 retarget 或重新训练的事实。

截至本次复盘，前期 identity/admission/observation/emission/Arbiter/cancellation/dispatch-owner 的纯 contract
已经建立，最重要的近期工作应改为依次完成七件事：

1. M5.3/M5.3a typed replay core、source-frame queue、三态 assessment 与 scoped eligibility 已完成；M5.3b
   默认关闭 canonical GUI consumer/RuntimeTraceSession/可选 debug WAL 也已接入，继续保持 legacy owner，
   不把 conditional evaluation 或 pre-dispatch tap 冒充 motion authority/actual receipt；
2. M4B.2a 已补 per-episode bind/unbind、dependency closure、durable range seal、trace clock domain、commit
   candidate 及 recorder 可选 seam；M4B.2b 已补 human/policy/replay source-specific formal profile 和原子
   seal/commit publisher，并保持 legacy finalize 失败隔离。human scoped runtime fixture 已完成 session WAL 共址、
   observer FIFO release barrier、多 episode session 和后台 finalize 验证；legacy metadata 只保存 immutable
   sidecar refs；seal-before-commit process-kill 已证明“有 seal 无 commit 即 uncommitted”。下一步覆盖更多
   crash 点、长稳、大 payload、目标机器 latency/power-loss acceptance。CS-2b-c现只将
   `deployable_sim_scheduled_pose_to_qpos_station_v1`窄replay profile接成runtime-integrated，尚未接入的policy、generic/
   real replay formal profile继续不得标为完成；
3. M4C.1 logical reader/ShadowEvaluationView 已完成；M5.5a 已由真实 authoritative sim producer 冻结
   publication→decision→directive→safety→command→receipt/measured schema，并据此完成 scoped variable-dt
   ActualDecisionView；deterministic facade与真实MuJoCo的测试侧formal episode seal/commit/read已完成，仍缺
   production sim entrypoint、长稳/power-loss/目标机验收与Parquet cache；
4. 为 DMTAC 的 session 外既有下发补 passive human emission seam；若 formal profile 不启用 DMTAC 则明确
   `not_applicable`，不能声称已覆盖；
5. M5.2 deterministic slow/fast/context/composer harness 已通过；M5.5a 单策略、M5.5b verified replay、
   M5.5c cancellation/hold 与 M5.5d minimal slow/fast composer
   deployable sim 已复用 shared authoritative core，且 replay eligibility/execution authorization 已冻结进
   authoritative input/publication barrier；gate/safety cancellation 独立于 Arbiter tick，当前只证明 MuJoCo
   software-queue ack。`ControlCompositionRuntime` 位于 authority 之上，只发布 composite proposal；10 Hz context/base、
   40 Hz residual、30 Hz Arbiter 的确定性 fixture 已验证高频 producer 不提升 authority publication rate。下一步补
   真实异步 scheduling 与 formal capture；M5.7a 已提供无副作用 shared-execution comparison seam、
   SafetyMigrationMatrix v1 和显式 anchor/profile 的 arm pose/qpos keyframe-safety adapter；M5.7b 已增加
   previous-safe-committed/measured-rebase 纯 lifecycle、typed commit evidence 和 safety input snapshot；M5.7c1
   已在 policy dry-run shadow 中接同一 emission 的 legacy-aligned arm pose/qpos 双算，observer error 不进入旧 fault
   或执行状态，numeric parity 与 representation mismatch 分开。M5.7c2 又以 pure fixture 验证 measured rebase→
   shared safe→typed shadow target admission→previous-safe-committed 的两步连续性；rejected/direct-command admission
   不提交。measured-divergence 数值阈值、frontend counterfactual binding、hand 与 human/position parity尚未冻结。
   M5.7e1 已有 pure resolved-target/admission contract，但尚无 worker caller；
   typed reference 必须等 shared resolver/planner publication seam 明确后再记录，
   不能从 `TeleopCommand` 或 backend receipt 反推，也不能复制 Arbiter/composer；
   public execution 选择已冻结：目标 shared profile 在 typed `POSE_REFERENCE`/`QPOS_REFERENCE` admission
   提交 safe anchor；现有 direct command path 是独立 compatibility profile，且只能在 backend/command acceptance
   后提交。`PositionDeployExecutor.submit()` 不是 reference admission，direct-command shadow 不能产生 committed
   anchor。ordered reference-segment/target-time contract、legacy scoped lineage、双activation profile与authoritative
   qpos segment receiver与实际120 Hz planner proposal/follow/accepted qcmd lineage已按deterministic sim范围落地；现有sim
   没有、也不新增独立60 Hz reference producer。scheduled successor必须将active generation与future reservation分开：
   admission只reserve，START才在predecessor END candidate提交后原子activate；正常handoff不是supersession或measured
   completion。v1 supersession仅允许START前显式替换pending segment，START后必须cancel/rebase。e2-d explicit
   ControlTrace projection与conditional lineage已按scoped sim落地；当前下一门槛是generation reservation/successor竞争
   fixture与formal sim capture；
6. 明确各 frontend 的 previous selected/safe/commanded/measured anchor、freshness 与 coupled-group 策略后，在
   sim/无副作用 shadow 中完成 persistence→shared safety→reference/command intent parity，然后在 legacy owner
   下把 M1C/M1D guard 接到所有最终 backend seam；
7. 最后才迁移单策略/replay authority，随后按 profile 需求推进 native streams、Learning/Domain Extension 和
   production multi-policy；所有真机步骤仍需独立安全授权。

这条路径最大化复用当前已经抽到 `robot_motion` 的链路，也避免未来加入 30 Hz 触觉策略、慢 VLA、
远端 GPU 或多模型协作时重新改写控制语义。
