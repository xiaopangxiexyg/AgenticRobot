# DexTeleop Control Core 落地与前提迁移计划

更新日期：2026-07-22

## 快速入口

本文是执行台账，不重复定义长期架构。第一次阅读建议按以下顺序：

1. 先看本节的当前状态和下一批；
2. 在[复盘状态口径](#复盘状态口径2026-07-19)中确认某项“完成”属于 contract、shadow、sim 还是真实 authority；
3. 到[迁移批次](#四前提依赖与迁移批次)查看该批的前提、交付和退出条件；
4. 实施前回看[当前实施边界](#七当前实施边界)，不得用计划中的对象冒充已有事实。

配套文档：

- [`dexteleop_architecture_planning.md`](dexteleop_architecture_planning.md)：长期语义、控制/数据闭环和不变量；
- [`dexteleop_architecture_visual_guide.md`](../dexteleop_architecture_visual_guide.md)：当前实现、数据流、阶段依赖和缺口图；
- [`dexteleop_rtc_landing_plan.md`](dexteleop_rtc_landing_plan.md)：异步 action chunk/RTC 的独立producer计划；
- [`dexteleop_control_station_plan.md`](dexteleop_control_station_plan.md)：human/replay/policy与null/sim/real统一实验台；
- 本文中的批次状态与退出条件优先于 visual guide 的摘要；visual guide 不作为新增授权或验收证据。

### 当前执行面板

| 项目 | 当前结论 |
|---|---|
| 真实运动 owner | 仍为 legacy GUI/deploy/replay 路径；shared Arbiter 没有真机 motion token |
| 已打通的 authoritative 路径 | 仅 scoped MuJoCo sim，含 policy、verified replay、cancellation、deterministic slow/fast composer 与受限 async scheduling fixture |
| 已接真实 frontend 的新链 | policy/human/replay 均为默认关闭的 passive shadow/compatibility seam，不改变 queue pop、dispatch 或 owner |
| 数据持久化 | human、scheduled-qpos sim与CS-2b-c dataset-backed replay均已有test-side formal fixture，可seal/commit/read；`dexteleop310` 中真实`MujocoScene + actuated rig`已通过，START前HOLD可在不伪造activation/frame时提交，seal后进程终止保持uncommitted但可审计重建；production builder与policy formal profile仍未完成 |
| 当前主批次 | M5.8-b1-real-A 独立 idle-worker drain/stop-only 现场批次已通过：无target真实worker、同一epoch registry、三participant drain、诊断WAL及每侧一次Slow-Stop均取得完整profile证据；motion/lifecycle authority仍为false，真实运动owner未变 |
| 当前本体通用性 | control/authority/trace 协议可复用；canonical action/execution v1 仍固定左右双6DoF RM65/q6-q16形态。准确表述为“对 contract-compatible embodiment 可替换” |
| 异步 action chunk | RTC-1/2/3a/3b/3b.5与RTC-3c default-off caller已落地；只有显式RTC profile进入bounded worker/timed-plan路径，默认仍为sync。唯一`rtc_frontend.step()`返回点新增注入式bounded passive fanout，CS-3a null-target Station只观察同一immutable step；正式artifact/transport/real-observation、ControlTrace capture与M1C drain仍未接 |
| 统一控制实验台 | 根级`control_station/` application leaf已冻结；CS-0a/0b、CS-1a/1b/1c已落地。CS-2a1以exact human-only manifest接通无设备synthetic qpos emission→现有HumanProposalAdapter→同一MuJoCo authoritative core/trace；CS-2a2a又以scheduled-qpos source-specific execution manifest完成真实WAL/seal/commit/reader的test-side formal capture；CS-2a2b scoped fixture现已要求candidate/authority typed completion后才开放publication，release会在同一core产生HOLD/cancellation并保持Station hold latch；测试侧bounded side-effect worker已验证release-before-enqueue与typed failure closure，注入式Station trace bridge又已将manifest/intent/result/barrier事实纳入专用scheduled-sim formal profile并完成seal/commit/reader重建。现有CS-1c Web exact-type继续拒绝该worker；独立scoped-sim loopback builder现已只对exact sim manifest开放typed lifecycle/candidate/authority API，且拒绝raw action、real target和production caller。仍无真实adapter或真机motion authorization |
| 已记录的后续目标 | 在完成本文新增的并行改动复核门槛后，进入 `m6_candidate_policy_arm_qpos_v1` 的 M5.8-b legacy-owner real mediation；只接最终 SDK seam、真实 worker epoch/drain、stop acknowledgement 与 latency evidence，不在同批切换 shared authority |
| 当前复核状态 | Control Station/RTC/Arbiter/owner/trace聚焦133 passed；2026-07-21在`dexteleop310`重跑全量为2605 passed、6 skipped、3 warnings、7 subtests，先前6项RM65 MuJoCo actuation profile v1→v2名称/hash漂移已收口 |
| 下一批 | 控制主线继续保持真机窄批次隔离；Control Station四条single-source sim formal fixture已scoped闭合，CS-3a existing-deploy RTC passive source也已通过null-target/no-authority隔离。下一步只可补production artifact/transport/real-observation preflight与startup composition；Station仍不得predict/merge/pop、驱动RTC producer lifecycle、修改真实owner或继承real-A授权 |
| 禁止提前进入 | M5.8-b 真机 legacy mediation、M6 authority 切换、真机 replay authority |

### 已确认的下一批语义

position worker 在 60 Hz 对相邻两个 15 Hz safe keyframe 做因果插值。代码核查确认 resampler 已保存两个时间
parent 的 source index/timestamp/alpha，collision-aware IK/planner 则在每个 reference tick 才运行。下一批按以下
已经确认的边界实施：

- `ReferenceSegment` 使用有序 `START/END` endpoint parent；parent source 可分别是 measured rebase、已 committed
  safe endpoint 或当前 safe target，“earlier/later”不得写成容易与左右机械臂混淆的 left/right；
- `ReferenceSegmentAdmission` 显式引用 parent、插值/规划 contract、admission guarantee 和半开有效区间；
- segment admission 只产生 pending commit candidate；只有其 `effective_from` 不晚于下一 safety target time、
  evidence 在 safety cutoff 前可用、且 segment 未 supersede/cancel、epoch/generation 仍匹配时，才成为 active anchor；
- 60 Hz 派生 reference frame 只记录 segment lineage，不产生 60 Hz Arbiter decision，也不重复提交 safety anchor；
- 后续 resolver/IK/planner/backend 失败通过 HOLD、execution epoch 和 typed rejection 失效，不能把 admission
  解释为连续碰撞安全或实际执行成功；
- `PositionDeployExecutor.submit()` 仍只是旧 frontend queue admission，不是 execution-target/reference admission。

实现保持事件事实不可变：不在一个 `SafetyAnchorCommit` 上原地修改 `PENDING/ACTIVE/SUPERSEDED/INVALIDATED`；
这些状态由 segment admission、commit candidate、terminal/invalidation evidence 和目标时刻确定性派生。

M5.7e2-a 没有需要产品选择的开放问题，已经按上述语义完成纯 contract。e2-b 的边界也已确认：允许新增默认
关闭、仅 observe-only 的 legacy parent/scope binding，但事实成熟度必须分为 raw legacy observation、bound legacy
observation 和 authoritative shared admission 三层。e2-b 只实现前两层；调用方附带 typed epoch/generation 不等于
legacy worker 已执行 M1C 校验。当前 worker 没有独立的 segment queue/admission 原子点，因此 raw segment 的形成
语义固定为“observer 首次看到由这一 ordered parent pair 实际派生的 reference”，不得改称 queue admission。所有实现仍以当前
worker 的实际插值、local generation/fault 检查和 callback 顺序为准，不通过猜测补齐。

e2-c 代码核查又发现一个不能由 adapter 猜测的时间前提：当前 MuJoCo authoritative profile 的
`target_lead_time_sec=0`，permit 通过后 qref 会在同一 decision 时刻立即写入 latest-reference mailbox；而 e2-a
segment contract 要求 receiver 在 END endpoint effective time 之前完成 admission，才能形成可审计的
`PENDING_EFFECTIVE → ACTIVE`。因此不能把当前 one-shot qref receipt 事后包装成 future segment admission，也不能
仅为通过 contract 放宽“admission 不得晚于 segment END”的校验。推荐的 e2-c1 路径是发布显式 nonzero target-lead
sim profile，并让 receiver 以同一 schedule 激活/展开 reference；具体 lead 与迟到策略必须成为
`DecisionScheduleSpec/ExecutionCapabilityProfile` 的版本化语义。最终选择不是替换现有 profile，而是并存：
`immediate_qref_sim_v1` 保持 zero lead 与既有 M5.5a–d 数值行为，不生成 future segment；
`scheduled_reference_segment_sim_v1` 使用显式 nonzero lead，专门验证 pending/effective、replacement 和
cancellation。两者不能在 reader/runtime 中按字段是否存在隐式切换。

## 目的与范围

本文是
[`dexteleop_architecture_planning.md`](dexteleop_architecture_planning.md)
的实施附录，只回答三个问题：

1. 新的 `ActionProposal`、`ActionArbiter`、ControlTrace Core 应落在哪些实际文件；
2. 在不改变现有真机 dispatch owner 的前提下，应该按什么顺序迁移；
3. 每个批次开始前需要满足什么前提，失败时如何回退。

本计划不授权真机运动、远端控制或硬件测试，也不表示要一次创建全部目标文件。第一轮实现只覆盖单
policy、单 `control_domain_id`、observe-only arbitration；多策略、native stream 和 learner 仍在事实链稳定后
再接入。例外是无法从 sealed capture 事后恢复的 learning-observability evidence：其纯 contract 与 fixture
可以和 M5.7 并行前移，但不得引入 Torch、训练线程、在线参数更新或新的执行 owner。

长期完成条件是闭环而非单次 authority 切换：human/policy/replay 通过同一 proposal/control core 控制并采集
v2.4 facts；dataset 可重新生成 replay record 与 TrainingView；learner 输出带 provenance/compatibility 的
artifact；artifact 经 offline/sim/shadow/authority gate 回到同一控制链并生成下一代 capture。更换遥操设备
或 contract-compatible embodiment 只替换 source/embodiment adapter 和验收 binding，不复制 Arbiter、writer、
dataset reader 或 learner API；若 DOF/frame/component 不兼容，则保留同一闭环协议，但必须显式 retarget、
生成新 TrainingView/重新训练或 fail closed，不能把“设备不敏感”解释为旧 artifact 无条件可移植。

首个真机放行单位继续是完全不变的 `m6_candidate_policy_arm_qpos_v1`：policy-only coupled 双臂 6DoF qpos、
30 Hz authoritative decision、scheduled segment 和当前 120 Hz planner/qcmd evidence；operator 只拥有独立
SAFETY_CONTROL/LIFECYCLE_CONTROL。hand、DMTAC、replay、pose impedance、fix-arm human 和 human motion
takeover 均由 manifest 显式 `not_applicable`。本批次不修改 `RobotAction` v1、26D/38D layout、RM65 safety
数值、qpos/reference 语义或 public compatibility columns。

### 复盘状态口径（2026-07-19）

后文不再用一个“完成”混合不同成熟度。状态统一解释为：

- **contract complete**：纯类型/算法及 deterministic fixture 已实现；不代表有 production caller；
- **compatibility seam active**：已接现有入口，但旧 owner/行为保持不变；
- **production authoritative**：已成为真实 selected/execution owner，并通过对应安全验收；
- **durable capture**：required facts 已进入可恢复存储并可按 high-water mark seal；
- **local only**：只存在于当前 dirty worktree，尚不表示 merged/released/hardware-accepted。

当前事实台账：

| 批次 | contract | 入口/执行接入 | 数据持久化 | 准确结论 |
|---|---|---|---|---|
| M0 | historical fixture complete；current source-marker revalidation **red** | `TeleopSession` 新增 observe-only receipt/observer seam 后需重审控制顺序 | golden JSON+SHA-256 未自动更新 | 数值/dispatch行为尚无失败证据，但当前两个 baseline 测试失败；重审前不能称当前工作树 M0 green |
| M1A | complete | proposal/admission 本身未全局接入 | 无 | identity/admission/immutable proposal 可复用 |
| M1A.5 | complete | current policy prediction 已使用 strict actor view | manifest metadata only | deploy actor firewall 已激活 |
| M1B | complete | M3 shadow 与 M5.5 sim authoritative caller 已复用；real frontend 无 authority caller | sim explicit trace | schedule/snapshot/Arbiter/persistence contract 已进入 sim，未进入真机 owner |
| M1C | complete，sim scoped integration | M5.5c 已接 MuJoCo in-process reference queue/hold；real reference/command/SDK worker 未接 | cancellation/ack explicit trace | atomic epoch、旧 permit/reference 失效与 software-queue ack 已在 sim 验证；不是硬件 stop acceptance |
| M1D | complete，sim scoped integration | M5.5a–c motion/safety-control permit 已位于 MuJoCo backend seam；real Rig/SDK 未接 | dispatch-owner validation explicit trace | sim backend 前互斥 gate 生效；真机 legacy/new owner 仍无硬互斥接线 |
| M2 | complete | 已接 `PolicyDeployRuntime.prepare_next_action()` 的唯一 queue-pop seam | debug lineage only | pre-safety absolute `RobotAction` emission 已激活，legacy owner 不变 |
| M3 | complete for scoped dry-run/replay shadow | `run_policy.py --action-arbiter-mode shadow`，默认 off，禁止 `--send-actions` | debug JSONL only | non-authoritative evaluation 已接入，无 selected/dispatch authority |
| M4A.1 | contract complete for current M3/M5 subset | M5 opt-in GUI shadow 已接 subordinate in-memory journal/projector | 默认仅内存，durable HWM=0 | explicit projection、policy/human-shadow lineage、profile scope 和 membership provenance 已完成；不能 commit capture |
| scoped M4B | contract complete for current explicit schema | M5 human 与 M5.3b replay opt-in GUI shadow 可接后台 spool worker | 可选 debug WAL+fsync+recovery，未 bind/seal capture | bounded spool、health ledger、durable ack 已验证；RuntimeTraceSession 不是 formal capture |
| M5 human shadow | partial frontend integration | canonical GUI arm/Inspire passive wiring 已完成，默认 off；DMTAC session 外分支 fail closed | in-memory RuntimeTraceSession；可选 scoped debug WAL | profile-derived identity/catalog/spec、qpos/pose seam、typed lineage 已验证；legacy owner 不变 |
| M5.2 synthetic topology | pure contract/harness complete | 无 production caller、PolicyOrchestrator 或第二 action queue | explicit in-memory context/composition trace fixture | 15/100/10/40 Hz 多速率、30/45/60 Hz Arbiter、composer/source/embodiment conformance 已验证 |
| M5.3 replay typed core | complete | debug dataset replay 已复用唯一 queue-pop emission；executor observer queue 由 M5.3b 默认关闭 consumer 复用 | replay record/source-frame emission/adapter receipt 有 explicit fixture | `producer_kind=replay`、两种既有调度的 timing/parity/failure isolation 已验证；legacy owner 不变 |
| M5.3a replay binding evidence | contract complete | replay-specific eligibility gate 已接 passive adapter；sim authority 由 M5.5b 复用 | source/target/derivation/assessment/eligibility explicit schema 与 lineage 已接 | 已移除旧 `compatible: bool`；三态 verifier、component/coupled-group/frame/transform/unknown target 矩阵通过；typed execution authorization 已有 sim caller，真机仍无授权 |
| M5.3b replay GUI passive shadow | compatibility seam active | canonical GUI 默认关闭 consumer 已接同一 executor emission queue；legacy replay owner 不变 | in-memory RuntimeTraceSession；可选 debug WAL，非 formal capture | target contract 不充分时只允许同 emission conditional evaluation；source contract 不充分时 record/reject；无 motion token/backend caller |
| M4B.2a episode seal/owner seam | contract + optional recorder seam complete | `RealTeleopDatasetRecorder` 可显式 attach subordinate sink；当前 production builder 不 attach | per-episode range/seal/commit contract；持久化由 M4B.2b publisher 承担 | per-episode bind/unbind、dependency closure、durable range、clock domain、legacy-finalize failure isolation 已验证；不终止 runtime session |
| M4B.2b formal profile/publisher | human与CS-2b-c replay scoped runtime fixture complete，production activation pending | human observer FIFO barrier及dataset-backed replay exact-emission composition均已验证；production builder未attach，generic `formal_real_shadow_v1`仍planned | dataset-local session WAL + atomic per-episode seal/commit manifest fixture可用 | replay专用pose→resolved-qpos lineage可commit/read；尚缺任意crash点/长稳/大payload/目标机验收，policy formal profile仍planned |
| M4C.1 logical reader/shadow view | scoped implementation complete | 无 runtime/control caller | committed seal + canonical session WAL 可重建；不落冗余物化文件 | 校验 commit/seal/range/dependency/membership/hash；区分 conditional replay evaluation；只输出 `would_select`，不生成 `selected_action`/裸 `action` |
| M5.5a single-policy deployable sim / M4C.2 | scoped implementation complete | `sim.mujoco.deploy` 已接 shared authoritative core 与现有 MuJoCo position rig；无 real frontend caller | authoritative explicit facts 与 variable-dt ActualDecisionView 可由内存 fixture 重建；尚未绑定 formal sim episode seal | observation cut→policy proposal→snapshot/publication→directive→safety→command→one-shot permit→MuJoCo receipt/measured 已贯通；slow/fast 和 formal profile 仍未完成 |
| M5.5b replay authoritative sim | scoped qpos + pose + dataset-backed GUI formal fixture complete | `sim.mujoco.deploy`仅在显式replay authority下接exact replay；CS-2b-c复用production loader/preflight/worker的同一emission，但legacy Rig为device-free fake；不接真机 | replay-specific profile要求source/binding/authorization与pose→resolved-qpos→segment/command lineage，WAL/seal/commit/read已接 | qpos pose-frame为N/A；arm-base/desk证据不互用；`qpos_target` GUI分支无lower-owned emission seam而fail closed；仍无production GUI composition或真机authority |
| M5.5c cancellation/hold sim | scoped implementation complete | gate hold/fault 可独立于 Arbiter tick 立即 cancellation；no-action expiry/publication fallback/shared-safety fault 已接 MuJoCo hold | non-selected directive、cancellation、safety-control permit 与 ack explicit trace 已接；formal sim seal 未接 | software-queue ack、epoch 失效、重复 hold 去重、fault latch 已验证；真实 SDK/backend-state/measured-stop ack、slow/fast、reference-segment fact 与 formal sim profile 尚未完成 |
| M5.5d minimal slow/fast sim | scoped deterministic multirate authority fixture complete | 独立 `PolicyTopologyRuntime` 共享 admission/epoch，10 Hz context/base、40 Hz residual 经 composer 形成 30 Hz proposal 后进入同一 sim Arbiter/dispatcher；默认关闭 composer authority | context/admission/snapshot、composition input/snapshot/result、composite proposal 与 authoritative lineage 已接 in-memory trace | 已证明 per-node observation cut、context dependency、40→30 Hz candidate coalescing 和 deterministic composition 可落到同一执行链；不是生产 PolicyGraph/worker scheduler/distributed runtime，native tactile/RGBD 与 formal sim capture 未接 |
| M5.7a shared execution shadow seam | pure contract + explicit shadow trace complete | `evaluate_shared_execution_shadow()` 只计算 proposed persistence→显式注入 safety→immutable command intent；无 frontend caller、segment owner、permit、Rig 或 backend | 独立 `shared_execution_evaluation` 事件引用 `ArbitrationEvaluation`；authoritative profile 明确 not-applicable | safe policy fixture 已比对 command 数值，hold fixture 证明不调用 safety/backend；legacy normalized safety 尚未迁移，SafetyMigrationMatrix/reference parity 未完成 |
| M5.7b safety anchor lifecycle | pure contract complete | 无 frontend/registry/segment/backend owner | 暂无 runtime ControlTrace producer | 已冻结 per-group/epoch/generation 的 previous-safe-committed、measured rebase、typed commit/reject evidence 与 immutable safety input；divergence 阈值未激活 |
| M5.7c1 policy legacy-aligned safety | scoped dry-run shadow active | `run_policy.py --action-arbiter-mode shadow` 对同一 immutable emission、observation 和旧 post-safety result 做一次被动双算；默认 off，replay 分支不误用 policy semantics | debug JSONL `policy_safety_shadow_comparison`，尚非 durable ControlTrace | arm pose/qpos numeric primitive parity、旧 anchor 选择、override/fault 和 observer failure isolation 已验证；representation 差异单列；不推进真实/counterfactual anchor |
| M5.7c2 counterfactual shared anchor | pure contract + deterministic chain fixture complete | 无 `run_policy.py` caller；target resolver/receiver 必须注入，无 segment/permit/backend | 不注册 ControlTrace event | measured startup rebase→shared safety→typed shadow target admission→previous-safe-committed 已连续两步验证；rejected/direct-command admission 不提交，enabled divergence guard fail closed；frontend binding 待 target profile |
| M5.7e1 resolved execution-target admission | pure contract complete | 无 position worker/direct backend caller；`PositionDeployExecutor.submit()` 明确不是该 admission | 暂无 runtime ControlTrace producer | 已分型 direct-command/pose/qpos/trajectory target、半开 schedule interval、receiver scope/epoch/generation/expiry rejection、per-group safe-action projection；direct command 必须等 backend acceptance，reference admission evidence 现显式区分 available/effective time |
| M5.7e2-a reference segment/anchor timeline | pure contract complete | 无 position worker/sim runtime caller；不改变 legacy queue/reference/IK/dispatch | 暂无 runtime ControlTrace producer | ordered START/END typed parent、admission guarantee、terminal、pending/effective/superseded/invalidation assessment与 ambiguous fail-closed 已落地；`SafetyAnchorState` 已分离 `available_at`/effective `committed_at` |
| M5.7e2-b legacy reference observation | b1 raw、b3 resolution、b4 terminal scoped active；b2 binding pure complete | executor可选 observer分别在resampler parent选定后、local final check后和generation失效时非阻塞提交；无production builder caller；legacy owner/行为不变 | producer稳定前不冻结public ControlTrace schema | pre/post-resolver分型；collision input/output hash；clear/fault terminal；`CONTEXT_ONLY`/local generation/M1C分型；failure outcome与production binding pending |
| M5.7e2-c0 authoritative dispatch context | typed seam complete，sim scoped active | permit 通过后可选 contextual dispatcher收到 immutable decision/safety/command/target-time/owner/epoch/group-generation；旧 dispatcher仍兼容 | 不新增 public trace event | context只提供可信lineage；receiver仍须读取live owner/epoch/generation并产生独立validation evidence |
| M5.7e2-c1 reference activation | pure contract complete；immediate与scheduled sim binding active | 默认`immediate_qref_sim_v1`保持zero lead；scheduled spec按30/60 Hz得到50 ms lead | 不新增public trace event | exact schedule/capability hash绑定、START前deadline、late reject-and-hold、measured first START和endpoint-time commit已冻结 |
| M5.7e2-c2 scheduled qpos receiver | scoped deterministic sim complete | 仅coupled bimanual qpos；receiver在自己的临界区二次读取live owner/authority/execution/next-group-generation；future admission只保留pending reservation，到START才切active generation | generation reservation/activation已成为explicit trace | measured first START、steady successor、START时handoff、旧generation callback拒绝、HOLD取消和双pending fail-closed已验证；仅START前显式supersede仍无production caller |
| M5.7e2-c3 planner/qcmd lineage | scoped deterministic sim complete | 不新增60 Hz resampler；仅scheduled segment记录START activation与实际command-due tick，immediate profile不伪造lineage | target activation显式引用generation activation；frame经activation间接闭合generation lineage | 每帧记录segment/reference/command ID、worker time、planner proposal/follow/accepted qcmd、epoch/generation和limit flags；一条decision可派生多帧且不推进anchor |
| M5.7e2-d / M5.5e2 scheduled formal capture | test-side formal fixture complete for current scope | deterministic facade与真实`MujocoScene + actuated rig`均已完成WAL/seal/commit/reader；START前HOLD分支不产生activation/frame；尚无production sim entrypoint | scheduled event family改为branch-conditional lineage；seal后进程终止无commit时默认reader拒绝，`require_commit=False`仅供审计重建 | 正常与pre-START cancel均可重建；真实MuJoCo focused回归通过；长稳、power-loss、目标机性能仍未验收 |
| M4L/M5L learning observability | M4L.1 pure contract + consumer fixture complete | 当前 `PolicyLike.predict()`/OpenPI client 只暴露最终 output；无 noise/fast-weight runtime hook | 不进入当前 ControlTrace profile | 已冻结 capability、inference-state/sample join、history/codec identity；普通 policy 可显式 not-applicable，actual noise 与 RNG-only 分型；M4L.2 等 concrete adapter API |

M5.2 完成时的聚焦回归覆盖 M0–M5.2 当前控制 contract、现有 session/deploy seam：376 passed、2 skipped；
其中 M5 覆盖 strict human mapping、不可变 pre-merge provenance、共享 Arbiter dry-run、canonical GUI
默认关闭 wiring、非阻塞 observer queue、typed trace lineage 和 failure isolation。该结果不是全仓测试、
process-kill/power-loss、长稳、
目标机性能或硬件验收。当前工作树还包含
大量并行修改及未跟踪的新 control-core 文件，后续变更必须继续保持 scoped，不能把本地状态描述为已发布。
M5.3 当前实现后的两组无硬件 focused 回归合计为 410 passed、3 skipped：其中 control-core/trace/replay
组为 229 passed、2 skipped，deploy/queue/replay compatibility 组为 181 passed、1 skipped。它们覆盖
action/Arbiter/ControlTrace、policy/human shadow、synthetic topology、dataset deploy、debug/hardware replay
compatibility、新 replay lineage，以及 replay observer failure isolation；同样不是全仓、真机或长稳验收。
M5.3a/b 收敛后最新 replay/ControlTrace/GUI/session 聚焦回归为 149 passed、3 skipped；其中新增 fixture 覆盖
三态 compatibility、typed evidence hash、conditional same-emission evaluation、source unknown fail-closed、
默认关闭 GUI wiring、无 dispatch token 和可选 debug WAL durability。该数字与前述 410 的测试集合不同，不能
相加，也不表示 formal episode integration、真实 legacy receipt 或硬件验收。

---

## 一、必须先冻结的决定

### 1. production decision policy v1 将 ActionArbiter 限制为 60 Hz

当前工程事实包括“稳定双臂 coupled IK 验收上限约 60 Hz”，但它不等于 Arbiter 调度器。DexTeleop 将
`ActionArbiter<=60 Hz` 收敛为独立、版本化的 production decision policy v1，不把 IK clock 与 decision
clock 合并，也不把“IK<=60”写成所有未来 embodiment 的普适架构定律。使用该 policy 的 authoritative
普通 decision 发布必须
满足：

```text
0 < configured_arbiter_hz <= 60 Hz
actual normal decision publication rate <= 60 Hz
```

这个限制只属于 `ActionArbiter`，不得扩散到其他链路：

- sensor、human input、policy inference 和 proposal producer 可以高于 60 Hz；
- reference、IK、trajectory、command、backend、physics 和 recording clock 保持独立；
- 120/150 Hz command 或更高频 physics 不需要因 Arbiter 降频；
- event-triggered arbitration 仍要经过同一个 decision gate，burst 只合并 trigger，不超频发布 decision；
- safety veto、HOLD、FAULT 可以立即生效，但它们产生 safety/mode event，不伪装成一个超频的普通
  selected action。

`ActionArbiter` 的纯函数可在测试和 replay 中任意调用；受限的是 production
`ArbitrationDecision` 的发布入口。所有正式 frontend 必须经过共享 `ArbitrationRuntime`，不能直接在各自
loop 中调用 pure Arbiter 后发布 decision。

60 Hz 常量应由 `robot_motion/action/arbitration_runtime.py` 持有，例如
`MAX_PRODUCTION_ARBITER_HZ = 60.0`，并绑定 policy version identity。不要把它加入
`PositionControlRates` 的全局校验，也不要借此次迁移修改现有 `15→75→150`、`15→60→120` 或
compatibility profile。

同时引入独立的 `ExecutionCapabilityProfile`/manifest binding，分别记录 validated ceiling 与留有余量的
configured operating rate、适用 action space/selection group、selected-keyframe、pose-reference、coupled-IK、
joint-reference、command/backend rate，以及 latency/CPU acceptance evidence。当前代码虽有独立 `ik_hz`
字段，但真机 coupled IK 仍随 reference worker 调用，`dual_arm_ik_hz` 只是 nominal budget；不能把 75 Hz
reference profile 当作 75 Hz coupled-IK 已验收，也不能为修正规划而把所有 reference/IK 强制降到 60 Hz。

未来 hand-only、joint-space 或经验证可接受更高 selected-keyframe rate 的 profile 若需提高上限，必须发布新的 decision-rate policy、execution capability、golden fixture、安全验收和 release note；不能用普通 CLI参数或 debug flag 绕过 v1。

### 2. 第一位迁移 owner 是 policy shadow，不是 GUI 主循环

当前最窄且最清楚的接入点在：

```text
robot_motion.deploy.PolicyDeployRuntime
  → absolute NormalizedAction
  → CanonicalDeployActionPipeline
  → TeleopCommand
  → PositionDeployExecutor / dispatch
```

因此先让 `teleop_runtime/deploy/run_policy.py` 产生 observe-only proposal、snapshot 和 evaluation；旧
`CanonicalDeployActionPipeline` 与 dispatch 继续是唯一执行 owner。等 policy 数值、时间和 fault parity
稳定后，再接 human `TeleopCommand` adapter。这样不会同时重构两条最复杂入口。

这里的 observe-only 结果准确名称是 `ArbitrationEvaluation`，不是 authoritative
`ArbitrationDecision`。M6 前 shadow 只记录 `would_select`；旧路径实际 action/command/receipt 使用独立
`LegacyExecutionObservation` lineage，不能被 shadow `selected_action` 覆盖。

### 3. runtime contract 与 dataset schema 不互相反向依赖

在线控制对象放在 `robot_motion`，持久化 schema 放在 `dataset_contracts`，编码 adapter 放在
`teleop_runtime`：

```text
robot_motion runtime event ──→ teleop_runtime encoder ──→ dataset_contracts record
```

- `robot_motion/action` 不导入 writer、Parquet、Torch、MuJoCo 或硬件 SDK；
- `dataset_contracts` 不导入 runtime object、frontend 或硬件模块；
- 明确允许 `robot_motion` 依赖 `dataset_contracts.canonical_encoding` 这一纯 primitive；后者保持无 I/O、
  无 Torch、无硬件且不反向导入 `robot_motion`，M1A 不复制第二份 encoder；
- `teleop_runtime/collection` 负责把 runtime event 编码成 v2.4 record 并交给现有 writer 生命周期；
- dataset replay 不通过反序列化 Python runtime handle 恢复对象，而是从稳定 ID、schema 和 primitive
  payload 重建。

runtime 与 dataset 对同一事实的 hash 必须服从一个纯 `CanonicalEncodingSpec v1`。若两层保留独立类型，
必须通过共享 conformance fixture 证明 runtime event、dataset record、transport round-trip 和无序 map 输入
产生相同 canonical content hash；不能各自用 dataclass/JSON 默认行为计算“canonical hash”。

M1A 固定以下默认实现，不再留给 caller 自选：

- 唯一实现位于 `dataset_contracts/canonical_encoding.py`，使用 SHA-256 和显式 schema/domain separator；
- ndarray 规范化为 little-endian bytes，并编码 dtype 与 shape；NaN 使用固定 canonical bit pattern，Inf 拒绝，
  `-0.0` 归一为 `+0.0`；
- generic encoder 不修改 quaternion 符号；几何 quaternion canonicalization 只由 action/pose contract 负责；
- 每类 record 提供显式 canonical payload projection，禁止把任意 metadata、Python `repr()` 或整个 dataclass
  自动纳入 semantic hash。

### 4. 真机前只建立本体泛化防火墙，不重构 action v1

当前准确能力是“对 contract-compatible embodiment 可替换”，不是“任意本体只换 backend”。M5.8/M6 的
首个放行单位固定为 `m6_candidate_policy_arm_qpos_v1`；本轮只阻止新的 RM65 假设继续进入 generic core，
不清除已有 v1 compatibility surface。

#### 泛化防火墙

新增静态测试必须执行以下规则：

```text
robot_motion/action 和 robot_motion/runtime 的 generic 模块
  不得新增 robot_motion.rm65、teleop_runtime、hardware、sim 或 vendor import

dataset_contracts / dataset_interface
  不得导入 robot_motion、teleop_runtime、hardware、sim 或 postprocess implementation

utils.dataset_postprocess 的普通 package import
  不得 eager-load robot_motion、teleop_runtime、hardware 或 vendor SDK
```

现有 v1 action/schema 中的 `left/right`、q6/q16 与固定 6DoF 只通过精确 allowlist 登记；测试比较当前基线，
允许删除但不允许新增。profile-specific `robot_motion.rm65`、real backend 和 RM65 sim adapter 不进入该 allowlist，
因为这些位置合法拥有本体语义。

2026-07-21 scoped 落地证据：

- `tests/test_embodiment_generalization_firewall.py` 已冻结 shared action/runtime 的设备依赖边界、现有 v1
  hardcode 文件 allowlist，以及 Arbiter 唯一的 `_copy_selected_component()` 名称投影 seam；该测试只阻止
  债务扩散，不构成 7DoF 或任意 component 的正向支持证据；
- DM-Tac calibration 的文件格式、checksum 和 median-base 算法已下沉到纯
  `dataset_contracts.tactile_calibration`；hardware wrapper 保留既有 import/API 和默认目录，offline force
  processor 显式选择兼容目录，不再静态依赖 hardware package；
- v2.2→v2.3 desk migration 直接消费纯 `PoseFrameContract` 和 checked-in profile JSON，不再导入
  `robot_motion.pose_frame_profiles`；`utils.dataset_postprocess` 对重型兼容工具使用 lazy public export，普通
  package import 不再加载 runtime/hardware；
- 定向证据为 80 passed、1 skipped；`GD-DATA-DEPS-001` 的当前静态/eager 依赖缺口已关闭，但后续新增
  postprocess 模块仍受同一 boundary test 约束。

#### Generalization debt inventory v1

| evidence ID | 分类 | 当前位置/事实 | 处理时机 |
|---|---|---|---|
| `GD-RM65-PROFILE-001` | RM65 profile-specific | `robot_motion.rm65`、RM65 backend/IK/collision/permit inventory | 合法保留；每个 execution manifest 独立验收 |
| `GD-ACTION-V1-001` | v1 compatibility surface | `RobotAction` 的 left/right、arm/hand、6DoF、q6/q16 | 当前冻结；EG2 由唯一 lossless bridge 包裹 |
| `GD-ARBITER-CORE-001` | generic-core leak | Arbiter 通过 effector name 解析 side/kind并重组固定 payload | 第二本体前 EG3 移除；当前禁止继续增加分支 |
| `GD-SAFETY-V1-001` | v1 compatibility surface | keyframe safety 的 pose7/qpos6 与 hand unsupported | 当前 RM65 profile 保留；EG3 改为 schema/profile handler |
| `GD-POLICY-IO-V1-001` | v1 compatibility surface | 26D action、38D state、dual-arm `PolicyIOLayout` | artifact 固定布局继续合法；跨本体必须新 layout/preflight |
| `GD-SIM-RM65-001` | RM65 profile-specific | sim action/scene/limits 与 RM65 asset | 合法保留；第二本体新增独立 sim contract |
| `GD-DATA-VIEW-001` | legacy materialized view | Parquet left/right 字段 | 保留兼容输出；EG3 才新增 native per-component facts |
| `GD-BUILDER-001` | composition leak | canonical builder 直接构造 RM65/Inspire/DMTAC，registry 尚非 manifest owner | 当前不重构；第二本体 EG3 以 manifest-driven builder 替换 |
| `GD-DATA-DEPS-001` | dependency leak | offline postprocess 曾静态/eager 导入 hardware 或 robot runtime | scoped 已修复并由 import-boundary test 锁定；新增模块继续 fail closed |

该台账只描述依赖与可移植性，不授予 motion authority，也不把任一 `BLOCKED` profile升级为 ready。

### 5. 并行工作树集成复核（2026-07-21）

本次在不启动设备的前提下，复核了 UMI、camera、DMTAC、dataset maintenance/postprocess、OpenPI、MuJoCo/RL
和 real-teleop refactor 等并行改动。结论不是回滚这些功能，而是把“代码已存在”与“已进入
`m6_candidate_policy_arm_qpos_v1` 证据链”严格分开。

| evidence ID | 并行变更事实 | 对当前主线的影响 | 处置/门槛 |
|---|---|---|---|
| `PAR-M0-001` | `TeleopSession.dispatch_command()` 现在接收 typed receipts，并增加 human proposal/filter/execution passive observer；旧 M0 source marker 不再存在 | 已人工复核实际 typed-receipt dispatch seam并版本化更新M0 marker；未改变queue pop或dispatch owner | **closed/no-hardware**；M0及全量suite通过，真机数值/时序仍归M5.8-b acceptance |
| `PAR-RM65-LIMIT-001` | 历史默认right J4=3.10 rad，而active URDF J4上限为2.44346 rad；共享transport常量和RL task profile此前仍写3.10 | 用户已冻结“当前manifest绑定URDF是关节位置限位唯一事实源”；继续保留3.10会让backend/sim接受URDF外动作 | **closed/no-hardware**：共享limit镜像、backend/sim消费者和三个RL task profile已与active URDF一致，并有URDF解析漂移测试；历史fallback已从默认preview/recovery seed移除。真实home仍不由默认值授权，必须来自显式alignment/fresh measured state并在真机preflight复核 |
| `PAR-GUI-FIXTURE-001` | foot-pad test引入同poll UI采样与缺省属性路径 | `__new__` contract fixture曾失败；UI必须先采样test-start，才能阻止同poll物理踏板触发fix/init | **closed/no-hardware**：缺省属性fail-safe、online edge基线和安全采样顺序已有回归；全量suite通过 |
| `PAR-POSE-001` | 默认 collection/deploy pose contract 已切到 `rm65_real_dual_arm_workspace_near_boundary_v1`；历史 table-center 文件名当前承载相同新 payload/alias | 改变默认数据/模型 frame hash；旧路径名不能证明仍是旧 table-center 语义 | M6 manifest、dataset和artifact只接受 exact source ID/payload/hash；历史 artifact若绑定旧contract必须保留独立旧payload或显式迁移，禁止借alias静默通过 |
| `PAR-RATE-001` | collection/position默认 profile 从 15/75/150 调整为 15/60/120 | 与当前 scoped RM65 evidence一致，但可能被误读成全局频率限制 | 只作为 execution profile default；唯一全局 production-v1 gate仍是 `ActionArbiter <=60 Hz`，reference/command不受该上限 |
| `PAR-CAMERA-001` | RealSense native transport改为YUYV并在source adapter内确定性转换为配置的BGR/RGB；registry收敛到v3 wrist/head placement | actor像素语义有测试保护，但transport/profile/URDF binding已变化；旧v1/v2 placement不再是默认支持事实 | artifact/session manifest绑定native format、output color space、camera registry和v3 placement hash；真机 camera startup/latency另验收 |
| `PAR-DMTAC-001` | DMTAC增加USB assignment、显式tactile calibration、telemetry grace和async visualizer；默认不再自动校准solved fields | 不阻塞arms-only候选；会改变包含DMTAC的formal capture完整性和calibration provenance | `m6_candidate_policy_arm_qpos_v1`继续显式`NOT_APPLICABLE`；DMTAC profile单独绑定assignment/calibration/permit/trace evidence |
| `PAR-DATA-MUTATE-001` | episode compact和tactile recompress原设计会原地修改finalized内容 | 与immutable sealed capture/derived overlay原则冲突，可能使相同commit identity指向不同内容 | **fail-closed已落地**：检测到任一v2.4 `capture_commit.cjson`即在staging/backup/manifest写入前拒绝，且内部maintenance helper有二次保护；legacy/pre-commit staging保持兼容。copy-on-write derived dataset尚未实现，因此committed v2.4目前是“可读、不可原地维护”，不是功能回退 |
| `PAR-OPENPI-001` | 新 `rm65_bimanual_dmtac` OpenPI profile具有artifact preflight/contract-check-only路径，使用impedance/hand配置且默认不send actions | 是独立部署工具链，不等同scheduled-qpos arms-only M6候选 | 不继承M5.5/M5.8/M6证据；若未来启用动作，发布独立ExecutionManifest、safety/permit和hardware acceptance |
| `PAR-SIM-RL-001` | actuated MuJoCo/RL/UMI replay能力扩展 | 可作为domain/task证据，但不是real servo twin或M5.8 hardware evidence | 只有绑定同一execution/trace manifest的formal fixture才能进入候选证据；普通RL成功不能替代authority/safety/receipt验收 |

依赖变更仅新增 dataset extra 的 `zstandard==0.25.0`，`uv.lock` 和 conda install check 已同步；本次 full suite
未显示新的 import/dependency failure。`zstandard` 的存在不改变上表的 capture immutability gate。

#### 真机前新增回归门槛

1. `PAR-M0-001`、`PAR-GUI-FIXTURE-001` 和 `PAR-RM65-LIMIT-001` 已关闭；其后并行sim actuation v2迁移曾产生6项profile name/hash漂移失败，2026-07-21全量2605 passed、6 skipped已证明该项关闭；
2. `m6_candidate_policy_arm_qpos_v1` 绑定 exact pose-frame、camera（如启用）、timing、component、URDF/solver-limit
   和 execution capability hash，不消费路径名/default值作为兼容证据；
3. OpenPI impedance/DMTAC、UMI、human position、replay和RL profile保持独立，不因共享RM65名称进入候选manifest；
4. 已commit v2.4 capture 的episode compact与tactile in-place recompress已fail closed，不能修改原seal/commit事实；
   copy-on-write derived maintenance仍未实现。该限制不阻塞正式capture、读取、训练或关闭maintenance的纯控制
   M5.8-b，只阻塞“对committed root原地维护”这一操作；公开派生版本必须另建dataset identity、parent hash、
   derivation manifest和新commit；
5. 上述控制收口只恢复进入M5.8-b的前提，不构成真机授权。后续目标仍是legacy-owner mediation，M6 owner切换另批执行。

#### Profile evidence 的组合复用规则

“不继承arms-only M6候选的authority evidence”只禁止整包转移执行授权，不表示新profile从零开始：

- generic proposal/Arbiter/ControlTrace/owner/epoch contract及其pure fixture可直接作为parent evidence复用；
- exact相同的URDF、component catalog、backend final seam、timing和safety contract可按hash引用已有evidence；
- profile只需补自己的delta，例如impedance worker、hand/DMTAC permit、不同camera/pose binding或不同cancellation ack；
- `motion_authorized`、硬件stop/latency、measured-divergence和operator approval不得跨不同ExecutionManifest自动继承；
- 新profile的evidence bundle记录parent evidence IDs + delta evidence + exclusion，避免代码分叉和重复验收。

因此profile不会在功能上“落后”：它可以立即消费共享的新contract和实现；只是在未补完自身delta证据前保持
shadow/blocked，不能冒充已获得真机motion authority。

---

## 二、当前可复用文件与目标职责

| 当前文件 | 保留能力 | 此次迁移方式 |
|---|---|---|
| `robot_motion/action/schema.py` | `RobotAction`、`EndEffectorTarget`、action schema | 不移动；proposal 直接引用其不可变快照 |
| `robot_motion/action/vector_pipeline.py` | vector→`RobotAction` | 由新 deploy adapter 复用，不再复制 vector decode |
| `robot_motion/action/executor.py` | `RobotActionResolver`、dispatch adapter 基础；当前读取 metadata 中 arm/hand spec | 保留在 Arbiter 下游；execution-relevant metadata 进入 typed semantic projection，不误归 audit metadata |
| `robot_motion/deploy/runtime.py` | 单 policy prediction、relative→absolute、chunk queue | 继续作为一个 action policy instance 的 stream core |
| `robot_motion/deploy/pose_adapter.py` | v2.3 frame/anchor/relative decode | proposal 之前的唯一表示转换来源 |
| `robot_motion/deploy/canonical_action.py` | 当前 deploy safety→`RobotAction`→command 兼容链 | 保留 public wrapper；先增加 pre-safety shadow seam，后续才调整 owner |
| `robot_motion/runtime/causal_history.py` | bounded causal history | ObservationCut 复用，不再新建通用 ring buffer |
| `robot_motion/position_control/*` | resampling、reference、trajectory、audit | 完全位于 selected action 下游；不套用 Arbiter 60 Hz 配置 |
| `teleop_runtime/deploy/run_policy.py` | 真机 policy composition、lifecycle、position executor、dispatch | 只做 wiring；新增逻辑下沉到窄模块，避免继续膨胀 |
| `teleop_runtime/position_control/deploy_executor.py` | real position keyframe execution；已有 executor 内 generation/dispatch lock/callback 前复查 | 复用内部取消防线；M6 前再在最终 seam 增加跨 legacy/new runtime owner token |
| `teleop_runtime.deploy.replay.DatasetReplayPolicy` | dataset action→policy queue 的 debug/compat replay | 保留 reader/layout/frame 校验；后续被 replay proposal adapter 包裹，不作为最终独立 authority 语义 |
| `teleop_runtime/replay`、`robot_motion/replay` | dataset preflight、轨迹分组、resampling、真机 replay executor | 复用 reader/schedule/safety 数学；最终 action 在 dispatch 前收敛为 replay proposal，现有 executor 先保持兼容 owner |
| `teleop_runtime/session.py` | human observation/fault/hold/guard/dispatch seam | policy shadow 稳定后再增加 observe-only proposal hook |
| `teleop_runtime/collection/writer_runtime.py` | sample/job/flush/close；代码明确不拥有 episode | ControlTrace 作为 subordinate sink composition 接入；dataset episode 仍由现有 frontend/recorder 管理 |
| `dataset_contracts/v23.py`、`causal_snapshot.py` | v2.3 absolute/relative 与因果兼容事实 | hash 和字段语义不原地修改；v2.4 只增量引用 |
| `dataset_contracts/policy_artifact.py`、`deployments/*` | PolicyIO/layout/frame-bound artifact 与模型 transport | 扩展 provenance/embodiment compatibility；不创建第二套 checkpoint sidecar/protocol |

明确不以 `teleop_runtime/core/kernel.py` 为落地点；它仍是 experimental，不能借迁移变成第二套生产状态机。

---

## 三、实际文件放置

### 1. M1A–M1D 已创建与 M5.2 计划创建的共享控制文件

```text
robot_motion/action/
├── proposal.py              # ActionProposal、ProposalBuffer、producer sequence/epoch
├── snapshot.py              # ProposalSnapshot、candidate reduction、canonical hash
├── components.py            # component catalog、selection group、coverage
├── authority.py             # 最小 AuthorityPolicySpec/Event/Lease
├── arbitration_input.py     # authority/gate/producer epoch + proposal 完整快照
├── arbiter.py               # pure deterministic ActionArbiter
├── arbitration_runtime.py   # DecisionScheduleSpec/tick、<=60 Hz publication gate
├── persistence.py           # ActionPersistencePolicy / ExecutionDirective
└── composition.py           # 尚未创建；M5.2 确定性 composer contract/fixture

robot_motion/runtime/
├── observation_cut.py       # ObservationCutRecord 与进程内 ObservationBundle
├── control_admission.py     # per-domain commit sequence/high-watermark/cut
├── producer_epoch.py        # receiver-side activation registry/snapshot
├── execution_gate.py        # ExecutionGateSnapshot/publication barrier
├── cancellation.py          # execution epoch、异步取消与 acknowledgement
└── dispatch_owner.py        # per-domain owner token/generation/permit gate

robot_motion/position_control/
└── capability.py            # ExecutionCapabilityProfile；不复用 Arbiter 60 Hz 校验
```

职责边界：

- `proposal.py` 持有 Draft/admitted Proposal/AdmissionReceipt/Buffer，不负责 relative decode、safety、IK 或 dispatch；
- `snapshot.py` 按本次 `target_effective_at` 固定第一版 reduction，不能用最新 sequence 提前消费未来
  keyframe；
- `arbiter.py` 只消费完整 immutable `ArbitrationInputSnapshot`；不得另读 store/clock/frontend state；
- `arbitration_runtime.py` 是 production decision 唯一发布口，使用 injected monotonic clock，错过 deadline
  时跳过/合并，禁止 catch-up burst；
- `observation_cut.py` 中持久化友好的 record 与本地 payload handle 的 bundle 分型，runtime handle 不进入
  dataset。
- `ActionProposal` v1 每个 keyframe 一个 proposal；同一 chunk 的 keyframe 共享
  `inference_id/policy_output_id/chunk_id`，分别声明自己的 effective interval；
- proposal ID 使用
  `runtime_session_id/control_domain_id/producer_id/producer_epoch/sequence` 组成的稳定 identity，content hash
  不承担对象 identity；dataset 的可空 `episode_uuid` 只表达 capture membership，不改变 runtime fact ID；
- snapshot canonical hash 同时包含 candidate `proposal_id + proposal_content_hash`；
- proposal 边界生成 detached `RobotAction` snapshot：ndarray 只读、mapping/metadata 递归冻结；M1A 不
  全局改变现有 `RobotAction` caller 的可变性。
- `ActionProposal.audit_metadata` 不参与控制或 semantic hash；priority/authority/validity/frame/component/
  timing/context 等均为 typed semantic field。当前 resolver 读取的 RobotAction arm/hand specs 必须经 allowlist
  canonical projection 进入 semantic hash，不能误归为 audit metadata；完整 record 另有 integrity hash。

`robot_motion/action/__init__.py` 只在 contract 稳定后导出必要 public type；开发中的内部 helper 不应全部
提升为顶层 API。

### 2. policy 接入文件

```text
robot_motion/deploy/
├── action_adapter.py        # absolute NormalizedAction→pre-safety RobotAction + representation receipt
└── observation_view.py      # per-policy allowlist、LastActionContract、control-private firewall

teleop_runtime/deploy/
└── arbiter_shadow.py        # passive proposal/evaluation coordinator，不操作 action queue
```

需要小范围修改：

- `robot_motion/deploy/canonical_action.py`：保留现有 `prepare()` 行为，内部复用或旁路调用
  `DeployRobotActionAdapter`；第一批不改变 safety/dispatch 数值顺序；
- `robot_motion/deploy/runtime.py`：`prepare_next_action()` 继续是唯一 queue pop；只在其返回结果中补齐 typed、
  immutable `PolicyActionEmission`（raw action、chunk/keyframe、inference/output、target-time lineage），不加入
  硬件或 Arbiter；
- `teleop_runtime/deploy/run_policy.py`：把同一次 `prepare_next_action()` 得到的 emission 同时交给 legacy
  execution 与 passive shadow coordinator；shadow 不得调用 queue/prediction API；不在大文件中实现 buffer、
  Arbiter 或 trace schema。

不要在第一批创建完整 `PolicyGraph`/`PolicyOrchestrator`。单 policy shadow 证明接口后，再新增：

```text
robot_motion/deploy/policy_node.py
robot_motion/deploy/policy_graph.py
robot_motion/deploy/context.py
robot_motion/deploy/orchestrator.py
robot_motion/deploy/deployment.py
```

### 3. v2.4 ControlTrace Core 文件

M4 全阶段只实现控制事实持久化，不实现 reward、训练 target 或完整 Domain Extension；以下为目标布局，
M4A 实际子集见后文执行状态：

```text
dataset_contracts/
├── canonical_encoding.py    # canonical payload、content hash 与 conformance identity
├── v24.py                   # schema version、capability、capture root
├── time.py                  # role-specific timestamp/clock evidence
├── control_trace.py         # observation/policy/proposal/decision/execution records
├── stage_schemas.py         # action/reference/command/measurement schema identity
├── completeness.py         # commit/core/stream/view/replay completeness
└── execution_manifest.py   # decision policy + execution capability + runtime/episode binding

teleop_runtime/collection/
├── v24_encoder.py           # runtime event→pure v2.4 record
└── control_trace_sink.py    # mandatory journal、durable high-water mark、flush/finalize/integrity

teleop_runtime/control_trace/
├── runtime.py               # deploy/collection 共用的 subordinate event journal/spool
├── lifecycle.py             # EpisodeLifecycleOwner adapter / RuntimeTraceSession 分型
└── projection.py            # M3 runtime fact→generic ControlTraceFact；M4B 前改显式 projection

dataset_interface/
└── control_trace.py         # ID lookup、shadow/actual views、lineage reader
```

`control_trace.py` 冻结逻辑 event family，不冻结“一 episode 六个 Parquet 文件”。物理 chunk/partition 由
manifest 决定，避免大量小文件。M4A 只支持 frozen Python/NumPy logical mapping 和 deterministic hash，并非
JSON-like storage encoding；M4B 冻结 WAL bytes，M4C 再落 Parquet/materialized layout。

`ControlTraceRuntime` 是 deploy/collection 共用的 subordinate event journal，不拥有 dataset episode 的
begin/finalize/abort/commit。现有 frontend/recorder 继续作为 `EpisodeLifecycleOwner`，legacy row、
ControlTrace、stream/media 都是其 sinks；`CollectionWriterRuntime` 通过 composition 接入，而不是把
  `SessionStepResult`/`CollectorComponents` 依赖带入 `run_policy.py`。policy deploy 没有 dataset episode 时
使用 `RuntimeTraceSession`。不能让 ControlTrace 的产生依赖 15 Hz `SessionStepResult.handle_step()`，也
不能为不同 frontend 新建相互竞争的 episode owner。

profile-required event 与 media payload 使用分离队列。control loop 不阻塞于 Parquet/media I/O，但 bounded
async journal 不承诺任意故障下零丢失；sink 维护 `issued_sequence`、`accepted_sequence`、
`durable_sequence`、durable high-water mark 和 gap/drop evidence。queue overflow、磁盘失败或 crash
recovery 产生 typed writer-health event：shadow profile 可继续但 capture uncommitted，authoritative
profile 是否 HOLD 由 runtime policy 决定，writer 自身不直接改变 authority。只有接入 preallocated local
spool/WAL 后，才声明跨进程 crash recovery；内存阶段只保证存活期间可见的 completeness evidence。

### 4. human 接入文件

policy shadow 验收后再创建：

```text
teleop_runtime/action/
├── __init__.py
└── human_proposal.py        # pre-merge TeleopCommand→RobotAction proposal adapter
```

`teleop_runtime/session.py` 只增加一个 observe-only event hook，不改变现有
`command_filter → command_guard → dispatch_command()` 顺序。GUI 主文件只做 hook wiring。未知 command type
必须产出 invalid proposal 并 fail closed，不能为记录方便而猜测 pose/frame/component。

### 5. replay 与 artifact 闭环文件

replay shadow 阶段按现有实现分层复用：

```text
dataset_interface/replay.py                 # sealed facts/view → typed replay record
dataset_contracts/replay.py                 # ReplaySourceContract/action-stage/schedule
robot_motion/replay/proposal_source.py       # 无 I/O schedule/record → ActionProposal
teleop_runtime/action/replay_proposal.py     # real frontend/preflight wiring，不 dispatch
```

`teleop_runtime.deploy.replay.DatasetReplayPolicy` 和 `teleop_runtime.replay` 暂保留兼容入口；先证明相同
action/frame/schedule 经 replay proposal 得到 parity，再迁移 owner。recorded backend command 只用于 audit，
默认 replay stage 是 canonical proposal/selected/safe 中显式选择的一层。

Learning Extension 的目标落位如下，等 M7 再创建或扩展：

```text
dataset_contracts/{source_device,embodiment,training_run}.py
dataset_contracts/policy_artifact.py          # v2.4 provenance/compatibility extension
dataset_interface/training_view.py
model/                                        # learner，仍不是 production control entrypoint
```

### 6. 暂不创建的文件

以下内容有明确设计，但不是 Control Core 首轮前提：

- `sim/mujoco/deploy/*`；
- `dataset_contracts/training_view.py`、`task_reward.py`、`domain.py`；
- `dataset_interface/training_view.py`、native stream window reader；
- `model/` learner、DAgger/RL loop；
- 通用 RPC/共享内存 transport；
- 新的顶层 control kernel 或第二个 dispatcher。

---

## 四、前提依赖与迁移批次

整体依赖顺序为：

```text
M0 行为基线与 observation firewall 负面基线
  → M1A 时间/ID/admission/canonical encoding/immutable proposal
  → M1A.5 strict observation firewall activation
  → M1B complete arbitration input/schedule/persistence/pure Arbiter/60 Hz gate
  → M1C execution cancellation/epoch contract
  → M1D dispatch ownership gate
  → M2 pre-safety RobotAction adapter parity
  → M3 policy ActionArbiter shadow
  → M4A v2.4 generic ControlTrace envelope + in-memory completeness
  → M4A.1 explicit policy-shadow schema/lineage/profile/membership correction
  → M4B typed record/schema 修正 + durable spool/WAL
  → M5 human proposal shadow
  → M5.2 minimal slow/fast synthetic topology
  → M5.3 replay typed core
  → M5.3a replay source/target binding evidence
  → M5.3b canonical GUI passive replay shadow
  → M4B.2a episode seal/owner seam
  → M4B.2b source-specific formal profile/durable publisher acceptance
  → M4C.1 logical reader/ShadowEvaluationView
  → M5.5 deployable sim shadow
  → M4C.2 authoritative/sim materialized view
  → M5.7 shared execution path shadow
  → M5.8 dispatch gate integration under legacy owner
  → M6 单策略 authority owner 切换
  → M6.5 replay authority 迁移（sim 必须；真机独立授权）
  → M7A native streams
  → M7B learning/artifact 闭环
  → M7C production 多策略 shadow
  → M8 多策略 authority owner 切换（按需、独立验收）
```

后一个批次不得用“代码已经写完”替代前一个批次的验收证据。

这里的依赖是当前选择的落地顺序，不是说 human adapter 的纯单元测试在理论上依赖磁盘 WAL。M4B 被安排在
M5 前，是为了先修正并冻结 typed event/admission/lifecycle schema，再让 human/replay 新 source 复用同一
durable journal；M4C 刻意放在 M5.3 后，避免只按单 policy event 过早冻结 materialized view。正式 real
shadow capture 仍硬依赖 M4B；M5/M5.2/M5.3 的纯 contract fixture 不构成真机 capture 授权。

### M0：冻结现有行为基线

执行状态（2026-07-21 复核）：M0 历史无硬件验收和 golden fixture 仍保留。single-policy fixture覆盖 relative decode/chunk、
canonical safe action/command、dry-run dispatch、position keyframe admission、position/non-position scheduling、
TeleopSession control order、collection/deploy last-action stage 和 legacy private-field leakage；fixture 已人工审查，
固定 SHA-256 与内容一致，普通测试无自动更新路径。但并行修改将 `TeleopSession` 的 dispatch 调用改为接收 typed
receipt 并增加 observe-only hooks，旧 source marker 不再匹配，导致两个 M0 baseline test 当前失败。现有代码顺序
看起来仍保持 filter/guard/dispatch，但必须重新人工核查 marker 与行为后发布新的 baseline evidence；禁止只为绿测
机械更新字符串或 golden。M1A 已有实现不因此回滚，但当前工作树在 M0 revalidation 完成前不得进入真机接线。

前提：只读审计当前 dirty worktree，确认不覆盖并行改动。

工作：

- 生成版本化 baseline manifest：fixture/hash、`PolicyDeployRuntime` output/chunk/relative decode、decoded
  absolute action、`CanonicalDeployActionPipeline` safety result、`RobotAction`、`TeleopCommand`、position
  keyframe、dispatch intent、timestamp/anchor evidence 和 `last_action` stage；
- 固定 `run_policy.py` position/non-position dry-run 调度行为；
- 记录 GUI `TeleopSession` 的 filter/guard/hold/dispatch 顺序；
- 记录当前 collection/deploy 各自真实 `last_action` stage；
- 注入 selected owner、fault、receipt、reward、writer/backend 状态，冻结当前 model-input 泄漏负面 fixture，
  作为 M1A typed allowlist 的验收输入；
- 保持 v2.3 causal contract hash 不变。

baseline 固定写入：

```text
tests/fixtures/control_core/v23_single_policy_baseline_v1.json
tests/fixtures/control_core/v23_single_policy_baseline_v1.sha256
tests/test_control_core_baseline.py
```

fixture 只能通过显式维护操作更新，普通测试只验证、不自动重写；变更必须同时给出行为差异说明。

优先复用现有测试：

```text
tests/test_robot_motion_action_contracts.py
tests/test_robot_motion_deploy_contracts.py
tests/test_v23_virtual_deploy_equivalence.py
tests/test_v23_real_deploy_acceptance.py
tests/test_teleop_frequency_defaults.py
tests/test_robot_motion_position_control.py
tests/test_mujoco_actuated_position.py
```

退出条件：baseline manifest 可重复生成；能够用无硬件 fixture 重放 policy output 到现有 canonical
command，并知道每一处时间、fault、private metadata 和 `last_action` 的当前语义。

### M1A：时间、ID、receiver admission、canonical encoding 与 immutable proposal

执行状态（2026-07-18）：M1A 已完成无硬件、additive contract 落地。已实现 versioned canonical encoding/golden
hash、clock-domain/reset/session identity、per-domain committed-prefix coordinator、receiver-admitted producer
epoch、bounded immutable proposal/receipt、ObservationCutRecord/Bundle 和 typed PolicyObservationView；M1A 完成时
尚未接 production caller，随后 M1A.5 已按下节激活。M1A 本身未修改 GUI、writer、position profile 或 dispatch。

前提：M0 golden fixture 可重复通过。

创建/修改：

```text
dataset_contracts/canonical_encoding.py
robot_motion/action/proposal.py
robot_motion/runtime/identity.py
robot_motion/runtime/immutable.py
robot_motion/runtime/control_admission.py
robot_motion/runtime/producer_epoch.py
robot_motion/runtime/observation_cut.py
robot_motion/deploy/observation_view.py
robot_motion/action/__init__.py
tests/test_action_proposal_contracts.py
tests/test_observation_cut_contracts.py
tests/test_control_admission.py
tests/test_canonical_encoding.py
tests/test_policy_observation_view.py
```

必须验证：

- `ClockDomainDescriptor`、session/boot identity、reset epoch 和 receiver-side availability 规则明确；
- producer 只提交不含 authoritative `available_at` 的 `ActionProposalDraft`；validation、deep-freeze、active
  epoch/sequence 检查和 atomic buffer admission 成功后，receiver 才赋予 `available_at/admission_sequence` 并
  返回 typed `ProposalAdmissionReceipt`；received-but-not-admitted 永不 eligible；
- 每个 control domain 的 `ControlAdmissionCoordinator` 只在 target Store insertion 成功的同一短临界区内
  分配 `control_admission_sequence`、stamp receiver `available_at` 并推进 committed high-watermark；禁止先
  分配 sequence 再异步写 Store。decision cut 原子读取 `(cutoff_at, committed_high_watermark)`，所有 Store
  以 time+sequence 双条件返回同一 committed prefix；
- proposal/action 深复制后不能被 producer 侧 numpy/mapping 修改；
- publication buffer 对 sequence rollback、duplicate 和容量耗尽 fail closed；proposal 的 effective interval
  在 M1A 自洽校验，基于 decision cutoff 的 expired/future-available candidate reduction 留给 M1B snapshot；
- map 顺序、numpy round-trip、runtime→dataset record 和 transport envelope 不改变 canonical content hash；
- float、NaN/Inf、`-0.0`、dtype/shape/endianness 和 timestamp unit 遵守本文件已冻结的 v1 规则；generic
  encoder 不自行 quaternion canonicalization；
- stable fact ID 使用 runtime-session/domain/producer/epoch/sequence；dataset 的 `episode_uuid | null` 只做
  capture membership，和 content hash 的职责不混淆；
- 每个 action keyframe 形成独立 proposal，并保留共享 chunk/inference/output lineage；
- `ProducerEpochRegistry` 的 activation 有稳定 ID/time/reason；新 epoch 激活后旧 epoch proposal 不进入后续
  snapshot，旧事实仍可审计；context/authority admission 复用同一 receiver-side 原则；
- concurrent same-timestamp proposal/authority/gate/epoch admissions 在 injected scheduler 下形成唯一 committed
  prefix；cut 不包含 in-flight/uncommitted store write，且 crash/rejection 不留下已推进但无对应 fact 的 sequence；
- audit metadata 变化不改变 arbitration semantic hash，且 Arbiter/safety 不可读取；现有 resolver 依赖的
  RobotAction arm/hand spec 必须进入 typed/allowlisted semantic projection，未知 execution-relevant metadata
  fail closed；
- typed `PolicyObservationView` 拒绝 M0 注入的 control-private 字段，并且 projection payload 与 runtime
  mutable object 隔离。

退出条件：时间/ID/hash/immutability conformance fixture 通过，production 代码尚无 caller，当前 model
input 行为由兼容 wrapper 保持，现有 runtime 行为零变化。

### M1A.5：strict PolicyObservationView activation

执行状态（2026-07-18）：已完成无硬件激活并暂停于 M1B 前。`PolicyDeployRuntime` 现在先生成 artifact/layout-bound
strict view，并在同一次 prediction transaction 内完成 legacy allowlisted field parity；正式预测以 strict view
为唯一事实源。现有 object-style policy 只获得由 strict view 重建的
`sanitized_deployment_observation_v1`，function-style policy 只获得 strict `model_input()`；两者均不能看到 runtime
fault、authority、safety、writer/backend 或任意 frontend metadata。OpenPI debug envelope 同步移除了 fault。
`run_policy.py` 默认 `strict`，`dual-validate` 只允许无动作兼容检查，`--send-actions` 会 fail closed；manifest
记录 contract hash、兼容 adapter、owner、删除条件和 M6 前期限。未改变 queue pop、chunk replacement、prediction
transaction、relative decode、safety、IK 或 dispatch owner。

前提：M1A contract 通过，M0 已固定 legacy model input 的 tensor/path/shape/value。

先在 runtime 内部让 legacy unrestricted projection 与 strict allowlist view 双生成并逐字段比较，但不把
unrestricted object 交给模型；parity 通过后，正式 `policy.predict()` 只接收 artifact-bound strict view，legacy
API shape 仅保留显式 sanitized compatibility/debug 对照且不能
携带 fault、authority、writer/backend 或任意 frontend metadata。activation 由独立配置/contract gate 控制，
失败可在 shadow-only compatibility window 回退并留下差异 evidence；manifest 必须记录 compatibility adapter、
owner、删除条件和最迟在 M6 authoritative gate 前结束的期限。authoritative mode 禁止 unrestricted legacy
model input。

退出条件：M3 shadow 前 production actor 已只有 strict input；现有模型输入数值、shape、last-action contract
和推理调度不变，control-private 注入负面测试通过。

本批实际修改/新增：

```text
robot_motion/deploy/{observation_view,runtime}.py
deployments/client/policy.py
deployments/openpi/client/observation_adapter.py
teleop_runtime/deploy/run_policy.py
tests/test_policy_observation_firewall_activation.py
```

### M1B：complete arbitration input、schedule、persistence 与 pure Arbiter

执行状态（2026-07-18）：M1B 已完成 additive、无硬件 contract 落地，并暂停于 M1C 前。已实现 target-time-aware
`ProposalSnapshot`、同一 committed cut 下的完整 `ArbitrationInputSnapshot`、最小 receiver-admitted
authority/lease 与 execution-gate store、component catalog、pure `ActionArbiter`、per-group persistence、三层 safety
result、`ExecutionCapabilityProfile`，以及 fixed/event-limited `DecisionScheduler` 和 production
`ArbitrationRuntime`。普通 authoritative selected decision 的配置与发布均 fail closed 于 60 Hz；该限制没有进入
sensor、proposal、reference、IK、command、backend 或 recording 配置。所有 M1B 模块尚无 production frontend
caller，未接 `run_policy.py`、GUI、writer、position executor 或 dispatch。

本批额外冻结：event burst 只保留 bounded trigger evidence 并记录 coalesced count；late/gate-changed/schedule-
mismatch result 只能形成 publication rejection 和显式 fallback，只有 successful publication 能形成
`AuthoritativeArbitrationDecision`。500 条高频 proposal fixture 只留下 target cut 的确定性候选，其余保存
`superseded_before_snapshot` evidence，不承诺逐条 selected。decision runtime budget 作为 computation metrics 记录，
不进入 pure result hash。

前提：M1A canonical identity 和 immutable proposal contract 稳定。

创建/修改：

```text
robot_motion/action/{snapshot,components,authority,arbitration_input,arbiter,arbitration_runtime,persistence,safety}.py
robot_motion/runtime/execution_gate.py
robot_motion/position_control/capability.py
tests/test_action_snapshot.py
tests/test_action_arbiter.py
tests/test_action_arbitration_runtime.py
tests/test_action_persistence.py
tests/test_arbitration_publication.py
tests/test_action_safety_contracts.py
tests/test_position_execution_capability.py
```

必须验证：

- snapshot 按本次 `target_effective_at` 选择 interval 覆盖 target 的 proposal，不以最新 sequence 提前消费
  未来 keyframe；
- effective interval 固定为 `[effective_from,effective_until)`；chunk keyframe interval 由
  `ActionEmissionScheduleSpec` 的起点/cadence/horizon/maximum-hold 推导，不能用 queue pop 时刻决定；
- 同一 chunk 多 keyframe、replace/append、late chunk、overlapping interval 和无序输入结果确定；
- winner 顺序固定为 `max(effective_from) → max(sequence) → lexical-min(proposal_id)`；
- snapshot reduction/hash 与输入顺序无关，hash 同时覆盖 proposal ID 与 proposal content hash；
- `ArbitrationInputSnapshot` 完整冻结 ProposalSnapshot、AuthorityStateSnapshot、ExecutionGateSnapshot、
  ProducerEpochRegistrySnapshot、previous selection、active groups/schema 和 DecisionScheduleTick；pure Arbiter
  只接受该对象，不读取 store、clock、环境变量或 frontend；
- DecisionScheduleTick 冻结 `cutoff_admission_sequence`，每个子 snapshot 保存 included sequence range；所有
  Store 以 `available_at<=cutoff_at && control_admission_sequence<=cutoff_sequence` 查询同一 committed prefix；
- v1 `capture_cut()` 在 coordinator read barrier 内只完成各 Store 的 immutable in-memory projection，全部 child
  snapshot 建成后释放；barrier 内禁止 hashing/I/O/Arbiter 计算，并测试 bounded Store 不在 cut/query 间淘汰
  所需记录及 admission pause budget；
- snapshot 后到达的 proposal、takeover、gate 或 producer restart 不改变本次 result hash，只进入下一
  input snapshot；但 live HOLD/FAULT/execution-epoch transition 必须在 publication barrier 取消/拒绝尚未执行
  的旧 result；
- `DecisionScheduleSpec/Tick` 分开 scheduled-decision、cutoff、target-effective、publication-deadline、
  decided 和 published time，并冻结 fixed/event-limited、missed-tick、late 和 trigger-coalescing policy；
- `max_target_lateness`/`late_decision_policy` 明确；超预算时按 profile no-action/hold，不发布过期 target；
- proposal disposition 区分 superseded、gate-coalesced、effective-interval-missed，不改写原始 validity；
- 最小 `AuthorityPolicySpec/Event/Lease` 明确 source/group priority、takeover/release、expiry、fallback/hold 和
  recovery producer 权限，不把规则硬编码在 frontend；
- `ExecutionGateStatus` 与 Arbiter 后 `SharedCanonicalSafetyResult`、backend guard 的类型和 owner 分离；
- component selection、coupled group 和 attribution 一次生成；
- `ActionPersistencePolicy + ActiveExecutionSnapshot → ExecutionDirective` 的结果按 selection group 确定；
  no-action 只不续期对应 group，独立 group 可继续 publish，coupled group 原子处理；global HOLD/FAULT 覆盖
  全部 group，局部 cancel 使用 group segment generation，backend 不支持局部停止时显式升级 hold scope；
- computation、shadow evaluation、publication attempt、successful authoritative decision 和 publication rejection
  使用不同 event；只有成功 publication 能产生 selected_action/ActualDecisionView/reference segment；
- computation/result 绑定 arbiter contract ID/hash、implementation hash、authority/selection policy hash、
  decision schedule hash 和 canonical encoding version；ExecutionDirective 另绑定 persistence policy hash；
- `decision_hz=60` 合法，`>60` fail closed；
- injected clock 下 event burst 不产生 catch-up decision；
- 100/500 Hz proposal 能被 60 Hz 或更低 Arbiter 因果消费；
- 100/500 Hz proposal 不保证逐条成为 selected action，未消费原因可审计；
- reference/IK/command rate 不属于该配置，测试不得为它们强加 60 Hz；
- synthetic budget 测试记录 decision runtime/deadline miss；真实 IK/end-to-end p50/p95/p99 留到获得明确
  硬件测试授权后的 execution capability 验收。

退出条件：相同 ArbitrationInputSnapshot + arbiter contract/implementation/selection-policy hash 得到相同
ArbitrationResult hash；schedule/group persistence/publication、pure Arbiter、snapshot 和 production gate 单元
测试通过，production 代码尚无 caller，现有 runtime 行为零变化。

### M1C：execution cancellation 与 epoch contract

执行状态（2026-07-18）：M1C 已完成 additive、无硬件 contract 落地，并暂停于 M1D 前。已新增
`ExecutionEpochRegistry`，以 per-control-domain lock 内的 epoch 更新作为 domain cancellation 线性化点；局部
selection-group cancellation 使用同一锁内的 segment generation 更新，backend capability 不支持局部 stop 或
fault 请求局部 scope 时会显式升级为 domain cancellation 并记录原因。`ExecutionWorkItem`/`ExecutionEpochGuard`
覆盖 reference segment、position command 和 backend command 的 dequeue/pre-backend 双重检查；现有
`PositionReferenceFrame.epoch`/`PositionCommandTick.epoch` 增加了明确的 `execution_epoch` 兼容别名，未改变
production 数据布局或行为。四层 acknowledgement、deadline timeout→typed FAULT escalation request、显式
recovery 新 epoch、authority/execution epoch 转移矩阵和并发原子性已有 deterministic tests；尚未连接现有
position worker、Rig、SDK、dispatch owner 或真机。

前提：M1B decision/attribution identity 稳定。

创建/修改：

```text
robot_motion/runtime/cancellation.py
tests/test_execution_cancellation.py
```

必须验证：

- authority epoch 与 execution epoch 独立；
- domain execution epoch 用于 hold-all/fault/owner transition；selection-group local cancellation 使用独立
  segment generation。coupled group 和 backend cancellation capability 由 ExecutionCapabilityProfile 决定，
  不支持局部 stop 时必须升级 scope 并记录原因；
- cancellation 的唯一线性化点是 control domain 内 `execution_epoch` 原子递增；reference segment、command
  tick 和 backend command 全部携带 epoch；
- reference/command worker 在 dequeue 与真正 backend/SDK call 前都检查 epoch，不匹配则拒绝发送；
- HOLD/FAULT cancellation 能引用并使旧 decision/reference segment/尚未 dispatch 的 queued command 失效；
- cancellation reason 至少分型为 fault、hold、authority_handoff、owner_rollback、backend_not_ready 和
  manual_stop；authority/execution epoch 的合法转移有显式矩阵测试；
- acknowledgement 分为 software queue、SDK request、backend state、measured motion stopped；profile 声明
  必需层级和 `cancellation_ack_deadline`，超时升级为 FAULT；
- 已发送 SDK command 不声明“撤回”；保留其 receipt，另行记录 hold request/result 和 backend state；
- recovery 产生新 execution epoch，旧 lookahead/segment 不能复活；
- cancellation event 不被计为普通 selected decision，也不能绕过 decision-rate gate。

退出条件：fake reference/command/backend worker 能确定性演示 cancel→ack→new epoch；尚不连接真机 dispatch。

### M1D：dispatch ownership gate

执行状态（2026-07-18）：M1D 已完成 additive、无硬件 contract 落地，并暂停于 M2 前。已新增同进程、
per-control-domain `DispatchOwnerRegistry`；token owner 仅允许 `legacy_control_path` 与
`shared_arbitration_runtime`，human/policy/replay 不能成为 dispatch owner。`MOTION`、`SAFETY_CONTROL` 与
`LIFECYCLE_CONTROL` 使用不同 permit 类型和 validation path，所有 permit 均由 registry 签发、一次性消费，
在 logical backend-call 线性化点产生 typed accept/reject receipt。owner handoff/rollback 必须在 HOLD 或
TRANSITIONING 下引用 M1C domain cancellation，原子 revoke 旧 token，等待完整 cancellation acknowledgement，
再以更高 authority epoch grant 新 generation；中间无 owner 合法，双 owner 不可达。普通 safety cancellation/
recovery 和 authority epoch 更新也可在不改变 runtime owner 的情况下推进 live permit gate。当前仅有 fake
backend fixture，尚未把 registry/permit 接到现有 `TeleopSession`、deploy executor、Rig、sim 或真机 SDK；
`process_scope=in_process_only`，不宣称跨进程锁能力。

前提：M1C execution epoch/cancellation 语义稳定。

创建：

```text
robot_motion/runtime/dispatch_owner.py
tests/test_dispatch_owner.py
```

第一版是同进程、per-control-domain 的原子 owner registry。`DispatchOwnerToken` 只区分
`legacy_control_path` 与 `shared_arbitration_runtime`；human/policy/replay 不是 dispatch owner。每条待发 command
使用分型 `DispatchPermit`：`MOTION` 携带 token generation、authority epoch、execution epoch 和 directive
lineage；`SAFETY_CONTROL` 只授权引用已 admission safety/cancellation event 的 stop/hold/disable；
`LIFECYCLE_CONTROL` 只授权 lifecycle state machine 的 startup/home/enable/rearm。后两者不依赖 motion owner，
但必须通过各自窄 capability/source/domain/epoch 检查，不能实现成通用 bypass。

必须验证：同一 domain 最多一个 current token；revoked/wrong-domain/wrong-generation token、旧 authority/
execution epoch command 全部在 backend call 前被拒绝并产生 typed receipt；rollback 先取消新 execution epoch；
owner transition 期间为 HOLD；无 owner 合法但双 owner 永不合法。M1D 只验证 pure/in-memory gate，不连接
现有 Rig 或真机 SDK；motion token revoke 后，合法 safety hold 仍能通过而普通 motion 不能；policy/human
不能构造 lifecycle permit；跨进程 token 不在本批假装成分布式锁。

退出条件：owner switch/rollback fixture 可确定性重放，任一时刻最多一个 token 有效。

### M2：拆出 pre-safety RobotAction adapter

前提：M1A–M1D contract 稳定；v2.3 pose/frame/anchor golden 已锁定。M2 仍不把 token 接到真机 seam。

创建/修改：

```text
robot_motion/deploy/action_adapter.py
robot_motion/deploy/canonical_action.py
robot_motion/deploy/runtime.py             # 只补 lineage 时才改
tests/test_deploy_action_emission.py
tests/test_robot_motion_deploy_contracts.py
tests/test_v23_virtual_deploy_equivalence.py
```

迁移方法：

1. `PolicyDeployRuntime` 继续输出已按 v2.3 解码为 absolute 的 `NormalizedAction`；
2. adapter 复用 `robot_action_from_vector_action()` 生成 pre-safety `RobotAction` 和 representation receipt；
3. `prepare_next_action()` 的唯一 pop 同时形成 immutable `PolicyActionEmission`；legacy execution 与后续
   shadow 都消费该对象，lookahead peek 不算 emission，也不产生第二次 pop；
4. M2/M3 固定 `admission_mode=on_emission_pop_v1`：inference 时只记录 PolicyOutput/keyframe schedule，future
   keyframe 留在旧 chunk queue；pop 后才形成 Draft/admitted Proposal。`produced_at=output completed_at`、
   `available_at=pop 后 admission commit time`、`target_effective_at=ActionEmissionScheduleSpec logical time`；
5. proposal selected 与否不影响 queue 继续按逻辑时间消费；被 pop 但未 selected 的 proposal 进入 trace，
   human takeover/shadow 结束后不能倒退使用旧 keyframe；
6. 当前 `ActionSafetyFilter` 与 `CanonicalDeployActionPipeline.prepare()` 仍是执行 owner；
7. 同时比较新 pre-safety action 经兼容 wrapper 后的 safe action/command 与 M0 golden；
8. parity 未完成前不调整 safety 顺序，不删除现有 wrapper。整 chunk 提前 admission 属于未来新 schedule mode，
   不在迁移中静默启用。

退出条件：所有有效 action 数值和 fault 行为一致，relative anchor/frame receipt 可完整追溯，dispatch 尚未
接触 Arbiter 输出。

执行状态（2026-07-18）：M2 已完成无硬件、兼容 owner 不变的落地，并暂停于 M3 前。新增纯
`DeployRobotActionAdapter`，复用现有 `robot_action_from_vector_action()` 将 v2.3 已解码 absolute
`NormalizedAction` 提升为 immutable pre-safety `RobotAction`，同时保留 legacy metadata 完整的
compatibility action；`CanonicalDeployActionPipeline` 仍按原顺序执行现有 policy safety 和 resolver。
`PolicyDeployRuntime` 现在在唯一 `prepare_next_action()` pop 上形成 immutable `PolicyActionEmission`，记录
inference/output、chunk/keyframe、representation receipt 和半开 effective interval；lookahead 仍只 peek，
不会产生 emission 或第二次消费。正式 deploy caller 仅补入已解析 `effective_action_hz` 的 keyframe period，
没有引入 proposal admission、Arbiter、DispatchOwnerToken 或 backend wiring。emission/adapter tap 异常通过
`emission_error` 暴露但不改变 legacy preparation。absolute/relative anchor、pre-/post-safety 数值、immutable
payload、single-pop、fault/command parity、M0 golden 和 deploy `--help` 已通过无硬件验证。

### M3：单 policy ActionArbiter shadow

前提：M2 parity 通过；M1A.5 strict actor view 已激活，M1B decision gate 已证明不超过 60 Hz，M1C/M1D
cancellation/ownership contract 可用。

创建/修改：

```text
teleop_runtime/deploy/arbiter_shadow.py
teleop_runtime/deploy/run_policy.py         # 只做构造与 event wiring
tests/test_policy_arbiter_shadow.py
```

运行模式建议为内部三态：

```text
off            # 完全不构造新链
shadow         # 产生 proposal/snapshot/evaluation，旧链执行
authoritative  # M6 之前禁止在真机启用
```

其中 shadow 只基于完整 `ArbitrationInputSnapshot` 产生 `ArbitrationEvaluationEvent`，不得产生 authoritative `ArbitrationDecisionEvent` 或事实层
`selected_action`。它记录 `would_select`，实际旧链 action/safe/command/receipt 通过独立
`LegacyExecutionObservationEvent` 对照；不能调用 Rig、position executor 或 backend。没有 proposal、queue
empty、policy timeout、readiness false 时也记录明确 evaluation/status，不能用上一条 action 无限填充。

shadow 是严格 passive tap：只消费 `run_policy.py` 已经从唯一 `prepare_next_action()` 调用取得的
`PolicyActionEmission`，不得持有 queue、调用 `pop_action()`/`prepare_next_action()`、发起 prediction
transaction 或清理 queue。

shadow 不申请 `DispatchOwnerToken`。冻结 input snapshot 后修改 frontend mode/current gate 不得改变已有
evaluation；相同 input snapshot + arbiter contract/implementation/selection-policy hash 重放必须得到相同
`would_select`。若 live gate 在 evaluation 后转为
HOLD/FAULT，只记录 publication rejection/cancellation，不允许 shadow 或旧 result 获得执行资格。

M3 的验收范围是 dry-run、deterministic replay 和 non-capture shadow wiring；M4B durable spool/profile-required
trace 完成前，不开展正式 real policy shadow capture。

退出条件：单 policy dry-run/replay 中，新 `would_select` pre-safety action 与旧路径对应 action 逐 component
一致；打开/关闭 shadow 对 queue pop count、chunk index、queue length、prediction schedule、chunk
replace/append、旧 dispatch count、fault/hold、`last_action` 和 loop rate 无影响。

执行状态（2026-07-18）：M3 已完成无硬件、默认关闭的单 policy dry-run/replay shadow 落地，并暂停于
M4 前。新增 `PassivePolicyArbiterShadow`，只接受同一次正式 queue pop 返回的 immutable
`PolicyActionEmission`，经 receiver-side proposal admission、完整 `ArbitrationInputSnapshot` 和 shared pure
`ActionArbiter` 生成显式 `ArbitrationEvaluationEvent`/`would_select_action`；没有 authoritative decision、
publication、`DispatchOwnerToken`、queue/prediction API 或 backend capability。legacy safe/command/execution
通过独立 `LegacyExecutionObservationEvent` 对照，并验证 pre-safety action 逐 component/frame parity。
frontend 新增默认 `off` 的 `--action-arbiter-mode {off,shadow}`；M3 `shadow` 与 `--send-actions` 组合 fail
closed，正式 real shadow capture 仍等待 M4B。policy prediction/emission schedule 的 control `now` 已从 wall
clock 修正为同一 local monotonic clock，observation/debug wall timestamp 未改；新 replace chunk 的 logical
effective start 不早于 inference completion。ProposalBuffer 增加只删除已过期 record、保留 producer sequence
high-water mark 的 bounded retention seam。当前 debug JSONL 只用于 M3 对照，不冒充 v2.4 durable trace。

### M4：v2.4 ControlTrace Core shadow

前提：M3 已有稳定 event，不再依靠 debug 字符串解析 decision。

执行状态（2026-07-19，M4A.1）：原 M4A 的 generic envelope + deterministic in-memory journal 已完成；复盘后
又完成当前 M3 policy-shadow 子集的 explicit schema 修正。现有纯 `ControlTraceFact`、逻辑 event registry、
`PersistedRecordEnvelope`、六个分作用域 `TraceRequirementProfile`、`RuntimeTraceSession`/外部
`EpisodeTraceMembership` typed transition 和有界 `ControlTraceRuntime`。journal 以同一 issued sequence domain
保存 accepted/rejected gap evidence；M4A.1 的
durable high-water mark 固定为 0，即使 in-memory completeness 满足也绝不声明 capture 可 commit。M3
`PolicyActionEmission`、source validation、producer epoch、proposal/snapshot/input/computation/evaluation 和
legacy execution 已有无 I/O explicit projection，shadow profile 中 authoritative decision 为 not-applicable，
`would_select_action` 不会变成 `selected_action`。由于仓库已有生产导入的 `teleop_runtime/recording.py`，实际新增位置采用不冲突的
`teleop_runtime/control_trace/`，未迁移或改写现有 `SessionRecorder`。M4A.1 本身尚无 durability；下述 scoped
M4B 也仍未接 frontend、现有 episode owner、Parquet 或正式真机 shadow capture。

scoped M4B 随后已完成当前 explicit schema 的无 frontend durable fixture：

- `control_trace_wal.py` 使用 64-byte versioned header、payload SHA-256、header CRC、journal sequence trailer 和
  strict canonical decode；
- `ControlTraceSpool` 使用固定最大文件、`pwrite+fsync`、strict-increasing sequence 和 torn/corrupt-tail recovery；
- 独立 1024-byte fixed-slot health ledger 在主 spool 满、append 失败或 recovery 丢弃 tail 时保存 durable evidence；
- `ControlTraceSpoolSink` 只在 fsync 成功后回报 runtime durable sequence/high-water，伪造 sink/hash 会拒绝；
- `ControlTraceSpoolWorker` 提供独立 start/notify/flush/stop lifecycle，但不拥有 episode begin/commit；
- `ControlTraceRuntime.issue()` 不再在 admission lock 内 materialize envelope/hash，storage worker 在锁外编码；
- contention fixture 对 256 个小事实记录 p50/p95/p99 和 50 ms deadline miss，当前验收 p99<20 ms、零 miss。

这些是本地 deterministic/torn-write fixture，不是 process-kill、power-loss、长稳、大 action payload 或目标真机
计算机 acceptance；没有后台 worker lifecycle、frontend/episode owner wiring 或 capture seal，因此
`capture_commit_eligible` 仍为 false。

M4 全阶段目标文件（不能视为 M4A 已全部创建）：

```text
dataset_contracts/{v24,time,control_trace,control_trace_wal,stage_schemas,completeness,execution_manifest}.py
dataset_contracts/__init__.py
teleop_runtime/collection/{v24_encoder,control_trace_sink}.py
teleop_runtime/control_trace/runtime.py
teleop_runtime/control_trace/lifecycle.py
teleop_runtime/control_trace/{spool,health_ledger,sink,worker}.py
teleop_runtime/collection/writer_runtime.py
dataset_interface/control_trace.py
tests/test_dataset_v24_control_trace.py
tests/test_control_trace_sink.py
tests/test_control_trace_wal.py
tests/test_control_trace_spool.py
tests/test_recording_lifecycle_contract.py
```

M4A.1 + scoped M4B 实际新增/修改为：

```text
dataset_contracts/{v24,control_trace,control_trace_wal,completeness}.py
dataset_contracts/canonical_encoding.py       # 增加严格 canonical decode
dataset_contracts/{__init__,README}.py
teleop_runtime/control_trace/{__init__,lifecycle,runtime,projection,spool,health_ledger,sink,worker}.py
teleop_runtime/deploy/arbiter_shadow.py       # 仅暴露已 admission 的 startup epoch fact
tests/test_dataset_v24_control_trace.py
tests/test_control_trace_runtime.py
tests/test_control_trace_shadow_projection.py
tests/test_control_trace_wal.py
tests/test_control_trace_spool.py
tests/test_control_trace_sink.py
```

`dataset_contracts/time.py`、`stage_schemas.py`、`execution_manifest.py`、collection encoder/sink、
`dataset_interface.control_trace` 和现有 writer/frontend wiring 均未实现。当前 `PersistedRecordEnvelope` 是
logical in-memory envelope，名称不表示已经写入 persistent storage。

M4A.1 将 `TraceRequirementProfile` 收紧为 `plain_dry_run_v1`、`policy_shadow_dry_run_v1`、
`replay_shadow_dry_run_v1`、`formal_real_shadow_v1`、`authoritative_real_v1` 和 `deployable_sim_v1`。profile
除 required/optional/not-applicable 外还声明 lineage policy、runtime-diagnostic/formal-capture scope 和
fixture-only/planned 状态。尚无 runtime fixture 的 formal profile 强制产生 `profile_not_implemented` lineage
issue，不能由“一种事件各出现一次”伪装 complete。

M4 全阶段目标逻辑 event family 包括；M4A.1 当前只为单 policy shadow fixture 直接相关的子集实现 explicit
schema，其余已注册的 planned event 必须在对应 producer/runtime 存在后补齐，当前构造会 fail closed：

```text
ObservationCutRecord / PolicyInference reference
ProposalAdmission / ProducerEpochActivation
ActionProposal / ProposalSnapshot / DecisionScheduleTick
AuthorityStateSnapshot / ExecutionGateSnapshot / ArbitrationInputSnapshot
ArbitrationComputation / ArbitrationEvaluation（shadow）
PublicationAttempt / ArbitrationDecision / ArbitrationPublicationRejected
ExecutionDirective / LegacyExecutionObservation（shadow 阶段）
source/shared/backend safety result
reference/command/DispatchOwner/DispatchReceipt/measurement alignment
ExecutionCancellation / acknowledgement
episode termination/truncation
```

哪些 family 必须出现只由当前 TraceRequirementProfile 决定。例如 authoritative real 必须有 decision→
directive→safety→command→owner/receipt→terminal lineage；shadow 必须有 evaluation 与 legacy execution
lineage，但不得伪造 authoritative decision。

runtime fact 持有 source semantic hash 和 explicit record schema identity。M4B/M4C 的 dataset encoder 必须使用版本化 storage envelope 包裹
fact ID、encoded record、schema/registry refs、episode membership 和 journal sequence 后再计算
`record_integrity_hash`；当前 M4A logical `PersistedRecordEnvelope` 仅用于内存 conformance，不能被描述为最终
storage envelope，也不能在 runtime `ActionProposal` 中预先声称其 hash 覆盖最终 dataset row。

按三个子门槛落地：M4A deterministic in-memory admission/completeness；M4B preallocated local spool/WAL 与
crash recovery；M4C Parquet/materialized view。正式 real policy shadow capture 前至少通过 M4B，并绑定
`formal_real_shadow_v1`；dry-run/replay diagnostics 使用各自 profile。v2.3 row 和 hash
保持不变。v2.4 capture finalize 通过 completeness matrix 描述缺失能力，不设置一个粗粒度
`training_valid`。writer failure 不得静默丢 profile-required trace，也不得由 writer 自行改变 authority。

只有 authoritative `ArbitrationDecision` 使用事实层 `selected_action` 并生成 ActualDecisionView。M3–M5
生成 ShadowEvaluationView，v2.3 row 继续表示 legacy execution；`would_select` 不得占用 selected/action 列。
只有声明 `action_stage` 和 materialization spec hash 的 decision/training view 才生成兼容裸列 `action`。
mandatory sink 维护 issued/accepted/durable sequence 和 durable high-water mark；overflow/crash fixture 必须
证明 control loop 不阻塞、capture 不被错误 seal、gap evidence 可审计。M4A 不宣称进程崩溃后可恢复内存中
issued-but-not-accepted 范围。

sequence 分层保存：producer sequence 属于源事实；journal issued sequence 表示 runtime 已发出；sink accepted/
durable sequence 表示接纳和可恢复持久化。它们不能压成一个“event sequence”。

dataset episode begin/finalize/abort/commit 仍由现有 frontend/recorder owner；`ControlTraceRuntime` 只是
subordinate journal。`EpisodeLifecycleOwner` protocol 后续允许 GUI/headless、policy deploy、replay 和 sim
rollout frontend 创建正式 v2.4 capture，但每次运行只能有一个 owner；policy deploy 无 dataset episode 时
使用 `RuntimeTraceSession`。

`runtime_trace_terminal` 只终止整个 `RuntimeTraceSession`，不得被复用为单个 dataset episode 的终点。
canonical GUI 的一个 runtime session 可以依次包含多个 episode，因此 M4B.2 必须在正式接线前冻结两级
完整性：runtime-session 级完整性要求唯一的 runtime terminal；episode-capture 级完整性以同一 owner 的
bind/unbind、episode 内 journal range、durable high-water mark 和 profile lineage 为依据，并生成独立、
不可变的 episode seal assessment。单个 episode finalize 不得关闭 ControlTrace admission，session shutdown
也不得反向改变已 seal episode 的事实。seal 只是 ControlTrace sink 的完整性证据，dataset commit/finalize
仍由现有 recorder owner 决定。

M4A.1/M5 当前 completeness 的准确范围是：除 required/not-applicable/terminal/admission gap 外，已对 M3
policy-shadow 检查唯一 queue-pop emission、source validation、proposal admission accepted/rejected 分支、
snapshot candidate/child、computation/evaluation、legacy execution 引用和关键 source hash。它尚不验证
authoritative/replay/context/composition lineage、完整因果 timestamp、stream、跨 schema transform 或
per-view eligibility。`writer_failure_behavior` 目前只是 profile 字段，尚未驱动
RuntimeSupervisor/HOLD；overflow 通过 rejection receipt/gap 留证，但还不会自动写出可持久化 writer-health
fact。只有 `plain_dry_run_v1`/`policy_shadow_dry_run_v1`/`human_shadow_dry_run_v1` 是当前 fixture-only
diagnostic profile；其余 profile
显式为 planned，completeness 会产生 `profile_not_implemented`，不能声称对应运行模式已完整。

当前 synchronous `issue()` 没有文件 I/O且容量满时立即拒绝；logical/storage envelope 编码与 hash 已移到
admission lock 外，accepted/durable high-water 使用增量维护。`snapshot()` 在锁内只取得标量和集合副本，排序与
fact-ID 物化在锁外完成，避免 diagnostic reader 长时间占用同一 control-facing lock。本地 contention fixture
已经覆盖该边界，但仍不能代替 real control loop 的长稳、大 payload、目标机器 OS worst-case 验收。M4B 继续
保持 control-facing admission、spool durability 和独立 failure evidence 分层。

#### M4A 复盘后、M4B 写盘前必须修正的 contract

1. **已完成当前 M3 子集的 explicit record projection。** 每个已实现 event 绑定 top-level allowlist/schema
   version，projector 不再递归展开 dataclass；audit metadata、runtime handle 和新增字段不会静默进入 payload/hash。
   尚无 producer/fixture 的 planned event 没有 record spec，构造会 fail closed，等对应阶段再冻结。
2. **已完成 policy-shadow lineage，其他模式随 producer 落地。** 当前已按 correlation/root ID 表达唯一
   queue-pop emission→source validation→proposal admission、accepted proposal、snapshot child、decision
   tick→input→computation→evaluation 以及 legacy execution 分支。authoritative publication 的 decision/rejection、
   selected 后的 directive/safety/command/receipt 和全程 HOLD 等合法分支尚未实现；authoritative profile
   保持 planned/incomplete，不能提前写入 WAL registry。
3. **已修正 profile scope/name。** plain dry-run、policy-shadow dry-run、replay-shadow dry-run、formal-real
   shadow、authoritative-real 和 deployable-sim 已分型，并携带 lineage/scope/implementation status。
4. **已增加 human source 的显式可重放 events。** `PolicyActionEmission`、`HumanCommandEmission`、独立
   `SourceAdapterReceipt`、source validation、`AuthorityEvent` 和 `ExecutionGateEvent` 已有 explicit
   record/projection；human receipt 不再塞入宽泛 metadata。M5.2 前还需
   `PolicyContext`/context admission 和 composition input/snapshot/output。新增 event 必须在 durable
   schema 冻结前进入 registry/profile migration，而不是塞进 audit metadata。
5. **已完成 logical episode membership provenance。** envelope 覆盖 lifecycle owner、binding event ID、
   membership epoch/bound time；bind/unbind 产生 typed transition，runtime fact ID 不因 episode 绑定而改变。
   M4B 仍需证明这些 transition 与 WAL append/recovery 的原子边界。
6. **scoped M4B 已区分 logical payload 与 WAL storage encoding。** versioned frame/header/length/CRC/SHA/trailer、
   strict canonical decode、`pwrite+fsync` 和 torn-tail recovery fixture 已完成；Parquet 仍属于 M4C。
7. **scoped M4B 已增加独立 writer-health 证据通道。** health ledger 使用预分配 fixed slot，与主 spool 容量
   分离；capacity、append failure 和 recovered tail 可留下 fsync evidence。它仍不能承诺同盘硬故障或 health
   ledger 自身损坏时零丢失，目标机 power-loss acceptance 尚未做。
8. **scoped M4B 已把 envelope 编码和 diagnostic snapshot 的排序/ID 物化移出 admission lock，并加入小型
   latency fixture。** accepted/durable high-water 不再由每次 receipt/snapshot 全表扫描；当前只证明 256 个小
   fact 在本地 injected lock contention 下的 p50/p95/p99/50 ms deadline；正式 real shadow 前仍需大 action
   payload、后台 worker、长稳和目标机验收，不能用该单测替代 production SLO。

M4A.1 已满足的退出条件：event registry/profile scope 确定；同一 M3 shadow fixture 可显式投影为不含
`ArbitrationDecision`/`selected_action` 的 in-memory facts；logical envelope hash 可重复；admission overflow、
not-applicable、terminal-after-close、membership transition、缺失 proposal/child/hash 的 lineage issue 和 durable
HWM=0 可审计；现有 recorder/import 不受影响。

M4C.1 已实现 committed seal + canonical session WAL 的 logical reader 与可重建 `ShadowEvaluationView`：reader
验证 sidecar semantic hash、commit→seal 引用、episode range/dependency hash、lifecycle membership 和 WAL
完整性；seal-only 数据默认拒绝，只能显式 `require_commit=False` 做 audit read。view 保留 conditional replay
assessment，且不产生 `selected_action`/裸 `action`。当前不写 Parquet/materialized cache，也没有
ActualDecisionView。

M4 全阶段尚未满足的退出条件：ActualDecisionView/authoritative lineage、任意 crash 点与目标机 power-loss/
性能 acceptance、Parquet 物化缓存和 v2.3/v2.4 正式双写。M4B.2 已完成本地 seal-before-commit process-kill
fixture，但不能替代目标机掉电证据；其余分别属于 M4C.2/后续 integration，
不得计入 M4A 完成度。

### M4L/M5L：Learning Observability（与 M5.7 并行，不实现 learner）

该支线只冻结大规模 capture 后无法可靠补算的 policy-side 证据，不改变 proposal、Arbiter、safety、dispatch 或
episode owner。当前 `PolicyDeployRuntime` 能稳定提供 inference/output/chunk/emission lineage，但通用
`PolicyLike.predict()` 与 OpenPI client 不提供 sampler noise、fast weights 或反向 action codec，因此第一批必须
是 capability/profile 驱动且 fail explicit：未声明能力时为 `not_applicable`；声明 required 而 adapter 未提交
证据时只使对应 training view/profile 不完整，不能伪造记录，也不能影响控制 loop。

分批落地：

```text
M4L.1 pure learning-observability contracts + deterministic consumer fixture
M4L.2 opt-in PolicyInference evidence hook + explicit ControlTrace projection
M5L.1 raw intervention/dual-history join fixture
M5L.2 one concrete v2.3 PolicyActionCodec round-trip fixture
M5L.3 formal policy+human sealed capture profile
M7B   learner / derived inversion / reward / artifact closure
```

M4L.1 已落地且只包含：

- `PolicyLearningObservabilitySpec`：按 policy node/artifact 声明 adaptation、generative sample、model-side chunk
  的 evidence level；
- `PolicyAdaptationStateRecord` 与 inference before/after linkage；state 属于 artifact-bound 顺序状态，不复用
  `PolicyContext`，也不绑定未来每个 queue-pop keyframe；
- `GenerativeActionSampleRecord`：只记录 adapter 实际提供的 noise/RNG evidence、sampler contract 和 model-side
  chunk；seed/RNG-only 不得声称 exact-noise replay；
- `HistoryActionContract`：显式区分 proposal history、behavior history 和 human override policy，不复用
  `LastActionContract` 或 safety anchor；
- `PolicyActionCodecContract`：先声明 decode/optional encode 能力和精确 layout/frame/chunk/schedule/
  normalization hash，不在通用框架里猜一个 inverse。

实现位于 `dataset_contracts.learning_observability`，对应 deterministic join fixture 已覆盖 stateless N/A、
state-before/after parent、actual-noise/RNG-only 分型、model-side chunk、history override 和 codec direction；不向
`ControlTrace` registry 添加尚无 runtime producer 的 event。M4L.2 只有在具体 policy adapter 能返回 typed evidence
后才冻结 event schema；实际 tensor/
checkpoint 使用 content-addressed blob ref，runtime Python object、CUDA pointer 或共享内存 handle 不入数据集。

M5L 原始 capture 只保存 authority/takeover/release、human raw/canonical proposal、同 cut 的 policy proposal、
selected/safe/command/measurement、clutch/reset 和 safety projection。`PRE_FAILURE/TAKEOVER_ALIGNMENT/
STABLE_CORRECTION/RECOVERY_AND_RELEASE`、监督 mask、inversion target 和 support 判定均为 parent-hash derived
overlay，不回写 sealed capture。

退出条件：普通 stateless/non-generative policy 可在全 not-applicable profile 下保持现有行为；stateful/generative
fixture 能从同一 inference ID 联结 observation、state-before/after、实际 sample evidence 与 model action chunk；
required evidence 缺失时只拒绝对应 TrainingView eligibility。进入 M4L.2 前必须先选定一个真实 model/server
adapter 及其可导出的 evidence API；在此之前不修改 `PolicyLike.predict()` 的现有返回语义。

### M5：human proposal shadow

当前状态（2026-07-19）：已在现有 `teleop_runtime.action` 上增量加入 strict human adapter，而非复制原有
`adapters.py`；支持显式 cartesian pose、arm qpos、带 `HandActionSpec` 的 hand qpos 和带 adapter identity 的
native end-effector target。缺失 arm frame/spec、缺失 hand spec、未知 command type 均产生 `unsupported`
receipt，不猜设备/本体语义。`TeleopSession` 已有默认关闭的 passive pre-filter emission hook 和
post-guard/post-dispatch legacy observation hook；hook 使用 detached immutable snapshot，异常不会改变
filter/guard/dispatch。human draft 已通过与 M3 policy 相同的 ProposalBuffer、完整 input snapshot 和 pure
Arbiter 得到 `would_select`；`HumanProposalShadowRuntime` 可将两个 session hook 关联到同一 source emission，
把最终 legacy command 重新适配后生成 component parity 与 `LegacyExecutionObservation`，并形成完整
`human_shadow_dry_run_v1` fixture。

canonical GUI 现已通过 `HumanShadowFrontend` 完成默认关闭的 arm/Inspire passive wiring：每个进程生成稳定
`runtime_session_id`，`control_domain_id`/`embodiment_instance_id` 可显式配置并有 profile-derived 默认值，
component catalog、arm frame、hand spec 来自实际启用的 session profile。observer 使用独立有界 FIFO 和后台
线程，control loop 只执行 non-blocking admission；overflow/处理失败会使 trace completeness 失败，但不改变
legacy dispatch。session emission/legacy observation 的接收时刻统一使用本地 monotonic clock。现有双臂
position collision-aware seam 已经是 IK 后 qpos，因此 shadow 如实记录 absolute `bimanual_qpos_v1`；其余
已覆盖路径记录 absolute pose。这是迁移期事实，不代表 Arbiter 已经移动到 IK 之前。

当前持久化边界是普通 GUI debug 默认使用无 episode membership 的 `RuntimeTraceSession`；显式
`--human-action-shadow-trace-root` 只启用 scoped WAL/spool/health worker，仍不拥有 dataset episode、commit 或
seal。DMTAC gripper 目前仍由 canonical GUI 的 `_dispatch_q6_gripper_control()` 在 `TeleopSession` 外独立下发，
现有 passive hook 无法完整覆盖；启用 human shadow 且存在 DMTAC side 时会在启动设备前 fail closed，不能
静默漏记 hand action。故 M5 仍是 partial frontend integration，不能称为 formal capture、全 effector coverage
或 authority migration。

前提：M3/M4A 已证明 policy proposal/evaluation 与 legacy execution 可分型投影，M4B 已把显式 typed
record、lineage rule、episode membership 和 durable spool 冻结，M1B 最小 `AuthorityPolicySpec` 已冻结，且
GUI 当前 dispatch 顺序有 M0 基线。若只先开发 human adapter 的纯 contract fixture，可不依赖磁盘 durability；
但不得在 M4B 前把它接成 formal capture 或宣称 profile-required trace 完整。

创建/修改：

```text
teleop_runtime/action/{__init__,human_proposal}.py
teleop_runtime/action/{human_shadow,human_shadow_frontend}.py
teleop_runtime/session.py
teleop_runtime/debug/run_vive_rm65_real_teleop.py  # 只做 hook wiring
tests/{test_human_action_proposal,test_human_shadow_frontend}.py
tests/test_real_teleop_refactor_contracts.py
```

先在 command merge/共享 safety 之前捕获 producer/component provenance，随后由 shadow Arbiter 计算
`would_select`。现有 `TeleopSession.command_filter/guard/fault/hold/dispatch_command()` 完全保留 owner。

每种 human command mapping 必须生成 `SourceAdapterReceipt` 并按 component 标记 `lossless`、
`lossy_but_explicit` 或 `unsupported`。默认补 frame、丢弃字段、猜 backend mode、无法恢复 partial hand mask
时不得生成 authoritative-valid proposal；unsupported/损失原因进入 typed receipt，不靠 metadata 文本。

已满足的无硬件退出条件：human command 与 proposal 的 pose/frame/component 对应及 adapter loss status 可重放；
启用 shadow 不二次消费动作，也不申请 dispatch token；默认关闭时不分配 runtime identity；observer queue
overflow 不阻塞控制并留下 incomplete evidence。尚未满足：DMTAC passive emission seam、正式 episode
bind/finalize/seal、目标机长稳与 latency 验收，以及真实 GUI 数采 action/fault/hold/dispatch parity。

### M5.2：minimal slow/fast synthetic topology

当前状态（2026-07-19）：pure contract/harness 已完成，未接任何 production caller。新增 transport-neutral
`PolicyNodeRuntime`/typed request-output、receiver-admitted `PolicyContextStore` 与 target-time-aware context
snapshot、internal-only `CompositionInputStore`/snapshot、`CompositionSpec` 和 `DeterministicComposer`。v1 composer
只实现 component union 与 qpos base-plus-bounded-residual；pose/SE(3) residual、confidence/freshness blend、通用
DAG/RPC 均未实现。composite proposal 使用 typed `composition_input_ids/hashes/spec_hash`，raw input 不能进入
ProposalBuffer；普通 proposal 保持原 `semantic.v1` payload/hash，不含 composition 的空字段，只有带 typed
composition lineage 的 proposal 使用显式 `semantic.v2` hash domain。context/composition admission、record、
snapshot 和 result 已有 explicit ControlTrace record/projection，
但尚未纳入 formal profile 的完整 lineage/seal 验收。

deterministic harness 已验证 15 Hz RGBD stable-ID reuse、100 Hz tactile latest sample、10 Hz slow context、40 Hz
fast residual、context sequence rollback/stale/restart/cut isolation、30/45/60 Hz Arbiter 和 composite lineage。
另有 test-local conformance fixture 证明两个 raw source schema 可得到相同 canonical action，compatible real/sim
sink 可保留 proposal 事实而使用不同 command schema，不兼容 DOF 在 dispatch 前 fail closed。这里的 fake sink
不是 production embodiment contract 或 sim backend。

前提：M1A–M1B 拓扑无关 contract 稳定，且 M4 event registry 已先补入显式
PolicyContext/context-admission/composition lineage record；不要求接真实第二个模型。

只创建最小 shared contract 与测试 harness：

```text
robot_motion/deploy/policy_node.py       # PolicyNodeRuntime protocol
robot_motion/deploy/context.py           # PolicyContextRecord + minimal ContextStore
robot_motion/action/composition.py       # internal InputRecord/Store/Snapshot/Spec + composer fixture
tests/test_policy_synthetic_topology.py  # FakeContextPolicyNode/FakeActionPolicyNode
tests/test_action_composition.py
tests/test_source_embodiment_adapter_contracts.py
```

使用 15 Hz fake RGBD、100 Hz fake state/tactile、10 Hz context node、30/40 Hz action node 和
30/45/60 Hz Arbiter，验证 context version/乱序/stale、producer restart、非整数频率、frame reuse、
observation cut 及 proposal dependency。该 harness 不接 `run_policy.py`，不创建 production
`PolicyOrchestrator`、通用 DAG 或 RPC。

base/residual/context draft 先经 receiver-side internal admission 成为 immutable `CompositionInputRecord` 并进入
`CompositionInputStore`，默认 `authority_candidate=false`；未 admission 的 Python object 不得进入 snapshot，
raw residual/base 也不得同时落入 global ProposalBuffer。随后最小 fixture 为：fake slow base + fake fast
bounded residual → CompositionInputSnapshot →
CompositeActionProposalDraft → normal admission → Arbiter。只测试 component union/base-plus-bounded-residual/
显式 transform；base/context 缺失、residual 缺失、旧 context 和 late input 按 CompositionSpec 得到
no-action/invalid/zero-residual 等显式结果。snapshot 后输入只进入下一 composition，lineage 可重放；不实现
confidence/freshness 自动权重、通用 blend 或 Arbiter 内 pose averaging，也不全局要求系数和为 1。

测试还必须证明 internal input 不可被 Arbiter 单独选择、composite 可追溯全部 input ID/hash、snapshot 后 late
residual 只进入下一 composition；确需某 output 同时独立 authority 时，必须显式发布第二个有 lineage 的 fact。

另用两个 contract-compatible fake human-source adapter 验证 raw schema 可替换但 canonical proposal 不变；
用两个 compatible fake embodiment sink 验证 proposal/decision 不变而 command/receipt schema 可不同；对
DOF/frame/component 不兼容 binding 在执行前 fail closed。

退出条件：single-process deterministic replay 可复现 slow-context→fast-proposal lineage；该测试没有新增
production control loop、dispatcher 或 action queue owner。

### M5.3：replay proposal shadow

前提：M4 能区分 evaluation/authoritative/legacy execution，v2.3 pose/frame/action compatibility baseline 已
冻结；不授权真机 replay 运动。

创建/修改：

```text
dataset_interface/replay.py
dataset_contracts/replay.py
robot_motion/replay/proposal_source.py
teleop_runtime/action/replay_proposal.py
tests/test_replay_action_proposal.py
tests/test_replay_control_core_parity.py
```

先支持显式 replay mode：decision replay、policy-observation replay、canonical action replay 和 command audit。
v2.3 row 通过 compatibility contract 选择 action stage；v2.4 直接读取 proposal/selected/safe facts。所有
action replay 都形成 `producer_kind=replay` proposal，绑定 source dataset/capture、episode/event ID、原
action stage、frame/component schema、schedule/resampling spec 和 target embodiment compatibility。

选择 recorded `safe` stage 只记录 old safety provenance，不授予 bypass：replay proposal 仍经过目标运行时
当前 shared safety，新 safe result 允许因 embodiment/state/calibration/safety profile 不同而变化，并必须同时
保留 old-safe source 与 new-safety lineage。

现有 `DatasetReplayPolicy`/`teleop_runtime.replay` 仍是实际 owner；新链只做 passive evaluation，并证明启停
shadow 不改变 frame/action index、queue/schedule、init、dispatch 和 completion。recorded command/SDK packet
不得作为跨 embodiment canonical replay 输入。

退出条件：同一 fixture 的现有 replay action 与新 replay proposal 在 pre-safety canonical stage 数值和时间
一致；decision replay 可重建 snapshot/Arbiter；不兼容 frame/action/embodiment binding 在任何 dispatch 前
fail closed。

执行状态（2026-07-19）：M5.3 的 typed dry-run/core、M5.3a binding evidence 与 M5.3b canonical GUI passive
consumer 已完成当前 scoped batch。新增四层复用 seam：`dataset_contracts.replay` 冻结 replay mode、source action stage、
schedule 和 target binding；`dataset_interface.replay` 只按显式 dataset contract 选择 stage，v2.2/v2.3
仅将根 `action` 解释为 `selected_pre_safety`，未知列 fail closed；`robot_motion.replay.proposal_source`
把 canonical action 及 provenance 转为 `producer_kind=replay` draft；`teleop_runtime.action.replay_proposal`
只做 frontend 适配。recorded command 被限制为 audit-only；recorded safe 只保留 old-safety provenance，
draft 没有 safety bypass。

`run_policy.py` 的 debug dataset replay shadow 已接同一次 `PolicyDeployRuntime.prepare_next_action()` 返回的
emission 和 raw-action source index，不调用第二次 predict/pop/cursor；typed contract 不可确认时只禁用新
replay shadow，不把 replay 错记为 policy，也不改变 legacy execution。fixture 已验证 cursor/queue pop、
chunk/source index、effective interval、canonical component value 和 frozen snapshot→pure Arbiter replay；
`replay_source_frame_emission`/`replay_action_record`/`replay_source_adapter_receipt` 已有 explicit schema、projection 和
`replay_shadow_dry_run_v1` branch-aware completeness。`robot_motion.replay` 既有 trajectory action 也通过纯
adapter 验证了 source index/relative time/target-effective parity。`ReplayExecutionController` 现可选接收 bounded
`put_nowait` observer queue：impedance 在既有 action tick 发一次，position 只在 causal left source keyframe
到期时发一次，不把未来 interpolation lookahead 提前公开；qpos/SDK command 路径保持 audit-only。默认无 queue
时不构造 emission；queue full/consumer error 只计 typed diagnostics，不改变 init、resampling、dispatch 或
completion。source dataset 的相对时间始终作为 provenance 保留；position source-frame 的 effective interval
跟随现有 source-time resampling，impedance source-frame 的 effective interval 则跟随既有固定 `action_hz`
tick，即使原始 frame time 有抖动也不得偷偷改变旧调度。fake-rig 两种 schedule 均已验证。尚未完成的是由
`ReplayGuiSession` 现可在显式 CLI/session opt-in 时创建并消费该 queue，成功 preflight 后只绑定 dataset/source
context，第一次 immutable emission 才惰性创建 replay-specific Arbiter shadow 与 RuntimeTraceSession；可选
debug WAL 不拥有 dataset episode lifecycle。target frame/capability 尚无 immutable runtime evidence 时 assessment
保持 `INDETERMINATE`，仅同 emission conditional evaluation；source frame/component contract 不足时只记录
adapter rejection，不生成 proposal。该接入仍不能称为 formal real shadow/capture complete，也不授权运动。

### M5.3a：replay source/target binding typed evidence

M5.3 typed core 曾使用的 `ReplayTargetBinding.compatible: bool` 同时混合了 source 是否可解释、target 是否兼容和
是否允许执行，现已不再沿用到 canonical GUI。M5.3a 将其拆为：

```text
ReplaySourceActionContract / LegacyReplayContractDerivationReceipt
ReplayTargetBindingSpec / TargetRuntimeActionContract
ReplayBindingAssessment（VERIFIED_COMPATIBLE / VERIFIED_INCOMPATIBLE / INDETERMINATE）
CandidateBindingEligibility（record/evaluation/authoritative candidate scope）
ReplayExecutionAuthorization（后续 execution preflight，独立于 compatibility）
```

source interpretation 必须来自 v2.4 explicit contract、v2.3 explicit metadata，或输入充分且版本化的 legacy
derivation；不能按数组宽度、当前机器人默认值或“旧 executor 能运行”猜 source frame/spec。target assessment
至少逐维检查 action stage/space、pose frame/transform、unit/convention、component/spec、selection-group 原子性和
execution capability。`INDETERMINATE` 与 mismatch/compatible 严格分型。

M5.3a 的 scoped 实现允许：source 已 VERIFIED、无已知 mismatch、且 proposal 来自当前 legacy replay 同一次
immutable emission 时，`INDETERMINATE` 可得到 `PASSIVE_SHADOW` eligibility；该 evaluation 必须通过 assessment/
eligibility lineage 标记为 conditional，`authoritative_allowed=false`。known mismatch 只能 record/rejected；source
interpretation 失败不得生成 canonical proposal。authoritative candidate 必须 verified compatible，但 motion
authorization 仍由后续独立 preflight/token/operator gate 决定。

M5.3a/b 阶段没有为 passive shadow 扩张 authority；M5.5b 现以 additive
`CandidateAuthorizationStore/Snapshot` 将 target-specific eligibility 与 execution authorization 固化到通用
`ArbitrationInputSnapshot`。pure Arbiter 仍只做确定性 authority selection，publication barrier 才检查被选中的
replay proposal 是否具有同一 cut 内的 receiver-admitted exact-target 证据；human/policy 无需伪造 replay binding。

代码落位：

```text
dataset_contracts/replay.py                 # source/target/assessment/use schema
dataset_interface/replay.py                 # explicit/legacy source derivation
robot_motion/replay/compatibility.py         # pure verifier
teleop_runtime/action/replay_binding.py      # target builder + scoped eligibility
teleop_runtime/action/replay_proposal.py     # consume assessment，不读 compatible bool
```

退出条件：unknown 不冒充 compatible；same source + different target 得到不同 assessment；verified mismatch 在
backend 前 fail closed；recorded safe 仍重新经过当前 safety；没有 shadow caller 读取旧 compatible bool。

执行状态：上述 scoped contract、pure verifier、explicit ControlTrace projection/lineage 和测试矩阵已完成。
M5.5b 已新增 sim-only authoritative caller：receiver 在 proposal admission 后校验
`VERIFIED_COMPATIBLE` assessment、execution-candidate eligibility、typed `ReplayExecutionAuthorization`、exact
target hash 和当前 dispatch-owner token，再将该 relation 独立 admission 并冻结进
`ArbitrationInputSnapshot`；publication barrier 对 replay 缺证据/拒绝/target mismatch fail closed。该事实只证明
sim authority，不授权真机 replay，也不改变 M5.3b conditional shadow 语义。

### M5.3b：canonical GUI replay passive shadow consumer

复用现有 `ReplayExecutionController` 同一次 source-frame emission queue，在 `ReplayGuiSession` 上层增加默认关闭、
bounded/nonblocking consumer：

```text
same immutable source-frame emission
  → source interpretation / binding assessment / eligibility
  → ReplayActionRecord / ActionProposal
  → passive ArbitrationEvaluation
  → RuntimeTraceSession / optional debug WAL
```

consumer 不再次读取 dataset、不 resample、不拥有 cursor/action queue，不申请 DispatchOwnerToken，不调用 backend。
queue overflow、assessment rejection 或 consumer failure 只使 shadow evidence incomplete，不改变 init、schedule、
dispatch、completion。runtime diagnostic 与 formal episode capture 分开；M4B.2 完成前不得称 formal real shadow。
现有 source emission 位于 dispatch 前，因此 M5.3b 第一批只能准确记录 legacy source/emission lineage；若要声明
legacy safe/command/receipt 完整，必须在既有执行结果 seam 增加 post-dispatch typed observation，不能从 source
action反推实际执行。

退出条件：开关前后 source index、调度、命令值/顺序/dispatch count 完全一致；conditional evaluation 可从
assessment 追溯；known mismatch 无 selectable candidate；没有 motion authority。

M5.5b 本批无硬件聚焦回归为 157 passed、2 skipped；另有 31 项 legacy finalize、WAL/spool/sink、episode seal 与
logical reader 兼容测试通过。包含 `dataset_episode_maintenance` 的补充组因当前解释器缺少可选 `pyarrow` 在收集期
未运行，未安装依赖、未改环境元数据；该缺失不影响本批纯 contract/sim 结果，也不能被记为通过。所有测试均未
启动 GUI、真机、远端设备或 motion。

执行状态：默认关闭 consumer、bounded queue、lazy runtime、三态 assessment、conditional/rejected branch、
RuntimeTraceSession 和可选 debug WAL 已接入 `ReplayGuiSession`；CLI 仅增加显式 opt-in 参数。当前 emission 位于
legacy dispatch 前，因此 trace 只声明 source/proposal/evaluation lineage，不伪造 post-dispatch legacy
safe/command/receipt。formal episode bind/seal 与真实 execution observation 仍归 M4B.2/M5.7。

### M5.5：deployable sim shadow

前提：M3–M5.3b 的 shared contract、trace、human/policy/replay shadow 与 minimal slow/fast synthetic 均已
通过无硬件验收；authoritative replay 还必须把 eligibility 固化进 input/publication barrier。

- 新增 `sim/mujoco/deploy` observation/action adapter；
- 复用同一 proposal/snapshot/Arbiter/cancellation/ControlTrace，不复制 `run_policy.py`；
- 先跑单 policy deterministic profile，再把 M5.2 fake slow/fast topology 接到 sim observation/action adapter，
  验证 owner isolation、context lineage、fault/hold 和 execution epoch；
- 在 sim 中让 M5.3 canonical replay proposal 成为 authoritative owner，证明 replay 不需要独立 dispatcher；
- M5.7/M6 前至少有一个 deployable sim golden replay 通过。

执行状态（2026-07-19，M5.5a）：已新增 backend-neutral
`robot_motion.action.AuthoritativeControlRuntime` 与 `sim.mujoco.deploy` 薄绑定。当前 deterministic 单策略 qpos
fixture 由 MuJoCo state `ObservationCutRecord/Bundle` 产生 proposal，复用 receiver admission、完整
`ArbitrationInputSnapshot`、不超过 60 Hz 的 publication gate、per-group persistence、一次 shared safety、
`RobotActionResolver`、M1D one-shot motion permit 和现有 `MujocoPositionExecutionRig` 的 60→120→1000 执行；
`sim` 未导入 `teleop_runtime`，也没有复制 `run_policy.py`。最终 permit 在 backend 调用前线性化校验，backend
exception/missing receipt 生成失败 receipt，不会建立成功 active segment。

同批次只为实际 producer 冻结了 authoritative publication/decision/directive/safety/command/dispatch/measurement
explicit schema，并实现不含裸 `action`、只以 `selected_action_stage=selected_pre_safety` 表意的 variable-dt
`ActualDecisionView`。计算完成时刻来自 publication fact 的 computation metrics，不再把 schedule time 冒充
compute completion。当前这些是 in-memory deterministic fixture；`deployable_sim_v1` formal episode profile、WAL
seal/commit 和 Parquet cache 尚未激活。

执行状态（2026-07-19，M5.5b）：已新增 receiver-admitted `CandidateAuthorizationRecord/Snapshot`，并把 snapshot
ID/hash 纳入完整 arbitration input。replay authorization 的公开 core 入口不接受调用方自报的
`authoritative_allowed=True`，而是核验 typed assessment、eligibility、`ReplayExecutionAuthorization`、exact target
contract 和当前 dispatch owner；晚到 admission 只能进入下一 cut。publication 对缺失、denied 或 target mismatch
证据拒绝发布。MuJoCo canonical qpos replay golden 已证明 source facts→replay proposal→同一 Arbiter→当前 shared
safety→同一 one-shot permit/backend→receipt 的闭环；recorded safe 仍不会绕过当前 safety。joint-space qpos 的
pose-frame check 仅在双方均无 pose frame 时标为 `NOT_APPLICABLE`，EE pose 的 unknown/mismatch 规则未放宽。

M5.5b 本身不等于完整 M5.5；其后 M5.5c 已补 MuJoCo scoped hold/fault cancellation，但 fake slow/fast topology
接入、reference-segment typed fact、formal sim episode seal/commit 和 validated sim2real dynamics 仍属于后续；
真机 replay authority 继续属于 M6.5 的独立高风险授权。

执行状态（2026-07-19，M5.5c cancellation scoped）：`AuthoritativeControlRuntime` 现对 accepted selected、
non-selected `no_action/hold/fault` 和 rejected-publication fallback 统一生成显式 `ExecutionDirective`；只有 accepted
selected 产生 `ArbitrationDecision`，无 decision 的 directive 以 nullable `decision_id` 和必需
`publication_attempt_id` 保留事实分型。`no_action` 不续期旧 segment；到期后原子递增 execution epoch、使旧 permit/
pending reference 失效，经 `SAFETY_CONTROL hold` permit 调用 MuJoCo `hold_motion()`，并记录 cancellation/ack。
已处于 backend hold 时后续 tick 不重复递增 epoch；成功 dispatch 新 selected segment 后才解除该运行时 hold 状态。

外部 gate hold/fault 不等待 30/60 Hz Arbiter tick，而是在 gate admission 时立即走同一 cancellation 线性化路径；
下一 decision cut 只观察已 admission 的 gate。shared safety fault 同样先 cancellation，再把 fault gate 留给下一 cut；
fault latch 不允许直接写回 READY，显式 recovery 尚待后续接入。当前 MuJoCo capability 只要求并证明
`software_queue` acknowledgement，不能解释为 SDK request、backend state 或 measured motion stopped 已验证。
`ActualDecisionView` 忽略 `decision_id=null` 的 non-selected directive，因而不会把 hold/no-action 或 publication
fallback 物化成 selected action。该批 117 项 control/trace/replay/sim 聚焦回归通过，未启动 GUI、真机或远端设备。

因此 M5.5c 只完成 cancellation/hold 子批次；slow/fast sim、typed reference-segment、formal sim episode seal/commit
和 dynamics/sim2real acceptance 仍属于后续，不得据此进入 M6 真机 owner 切换。

执行状态（2026-07-19，M5.5d minimal topology scoped）：新增
`robot_motion.deploy.PolicyTopologyRuntime`，其只拥有 receiver-admitted `PolicyContextStore`、
`CompositionInputStore`、context/composition snapshot、`DeterministicComposer` 和 composite proposal publication；
它共享 authoritative core 的 coordinator/producer-epoch prefix，但不拥有 authority policy、execution state、
action queue、dispatcher 或 backend。该边界避免把策略图塞进 `AuthoritativeControlRuntime`，也没有引入
`action → deploy → action` 的模块加载环。

MuJoCo 增加默认关闭的 `enable_composer_authority` 与 per-node observation adapter。synthetic fixture 使用独立
slow context、slow base、fast residual 和 composer producer identity，context/base/residual raw input 从不直接进入
ProposalBuffer；只有带完整 input/context/spec lineage 的 `producer_kind=composer` proposal 能进入既有
ProposalBuffer→Arbiter→shared safety→one-shot motion permit→MuJoCo 链。trace completeness 现校验 context 的
observation/parent、composition input 的 context/observation、snapshot ID/hash、result input lineage，以及 composite
proposal 与 composition result/spec 一致性。初始 scoped 总回归为 211 passed；补入确定性多速率 fixture 后，当前
control/topology/trace/replay/sim 相关集合为 164 passed、2 skipped，两组测试集合不同，不能相加解释为全仓结果。

新增 multirate fixture 在同一 control domain 内按确定性时间线产生 10 Hz slow context/base、40 Hz fast residual 和
30 Hz decision。9 个 fast residual 与 3 个 slow base/context 只形成 7 个 composite authority proposal 和 7 次
authoritative decision；raw composition input 从未进入 `ProposalBuffer`，decision publication 间隔不小于 1/30 s。
因此该 fixture 证明了 producer 可高于 Arbiter 频率、但 authoritative learned-action 更新仍受独立 decision gate
约束；它不等于已经实现真实线程、GPU inference scheduler 或 native RGBD/tactile stream。

M5.5d 仍只是由测试显式触发的 topology authority fixture：没有实现通用 DAG scheduler、真实异步 10/40 Hz worker、
RGBD/tactile native stream、RPC/distributed binding、模型 artifact 或 fallback supervisor。真实 slow context 的业务
payload及其到 base action 的 adapter 仍由版本化 policy/artifact contract 决定，框架没有根据字段名猜测。

后续 M5.5e 拆成两个独立子门槛，避免把 scheduler 和 dataset ownership 混成一个改动：

- M5.5e1 limited async harness（scoped 完成）：新增确定性的有界 completion queue，不启动 OS thread，也不建立
  第二个 control loop；已覆盖 latency/jitter、deadline miss、cutoff 后结果、stale context、producer restart、burst、
  composition 与 HOLD 竞争和 queue-full fail-closed。context/input 仍经过既有 receiver admission，只有 composite
  proposal 进入 ProposalBuffer，30 Hz decision gate 未被高频 producer 绕过。声明 `required_context_channels` 的
  composition 现在从同一个 causal cut 同时冻结 context/input snapshot；context missing 或 dependency mismatch
  产生 typed `no_action` result，不 admission proposal。该 fixture 不是 production worker、GPU scheduler、通用 DAG、
  RPC 或 distributed partition 验收。
- M5.5e2 formal sim capture：第一阶段采用测试侧 composition fixture，绑定真实 authoritative MuJoCo runtime模块、真实
  ControlTrace runtime/WAL/spool、唯一 EpisodeLifecycleOwner、seal/commit publisher和committed reader。fixture只负责
  construct/inject/run/crash/recover/assert，不得实现测试专属lifecycle、WAL、completeness、receiver或reader。semantic
  identity/hash禁止包含测试路径、pytest case、临时目录绝对路径或fixture对象identity。该阶段成熟度固定为
  `formal_sim_capture_fixture_complete`，不等于production sim entrypoint。`sim`仍不得依赖`teleop_runtime`；正式
  composition root在fixture证明依赖关系后另行决定。

formal profile拆为`deployable_sim_immediate_qref_v1`与`deployable_sim_scheduled_qpos_v1`。scheduled profile要求
decision/directive/shared safety、receipt/measurement和execution manifest；segment event family按实际分支条件闭合：每个
safe scheduled result必须有一次segment/admission attempt，accepted segment必须有candidate/reservation/terminal，只有实际
到达START才允许出现成对的generation/target activation，planner frame必须引用activation。START前取消不得为了满足presence
检查而伪造activation/frame；发生cancellation时还要求request/ack。immediate profile把segment family明确
标为`not_applicable`，不能按missing处理。M5.5e2最低覆盖normal/no-action/late admission/mailbox failure、START前后
HOLD、replacement、terminal因果可用性、process-kill关键点、WAL tail recovery和committed reader重建；任一悬空
segment/frame/candidate均不得commit。Parquet cache继续后置。

执行状态（2026-07-21）：测试侧composition已用deterministic scene facade贯通真实runtime/WAL/subordinate lifecycle/
seal/commit/committed reader，并重建一条decision到多条planner frame的lineage。该fixture发现并修复了
`ControlTraceFact`在hash后才把list冻结成tuple导致WAL重读hash不一致的问题；同一helper在项目
`dexteleop310`环境中已通过真实`MujocoScene.from_contract + rm65_bimanual_udp_feedback_profile` composition。
另有START前HOLD fixture证明合法取消不需要伪造activation/frame；formal scheduled seal后、capture commit前终止进程时，
公开reader默认拒绝该episode，而审计模式可从durable seal/WAL恢复同一decision lineage。当前成熟度为
`formal_sim_capture_fixture_complete`，不等于production sim entrypoint；长稳、任意power-loss点与目标机性能仍未验收。

`typed reference_segment` 当前只在 scheduled MuJoCo receiver 内形成 canonical admission；该 receiver 对 live
owner/authority/execution/group generation 做第二次检查，并在线性化插入 timeline 后才出 receipt。legacy real
position worker 只产生 raw/bound legacy observation，仍不能升级成 canonical admission。`immediate_qref_sim_v1`
也不伪造 future segment；实际高频下游是 120 Hz planner/qcmd lineage，而不是人为新增的 60 Hz reference producer。

### M5.7：shared execution path shadow

前提：M5.5 deployable sim 通过；M1B per-group persistence/publication taxonomy、M1C cancellation 和 shared
safety typed seam 已稳定。

将 computation/evaluation 的 `would_select` 送入完全无 backend side effect 的执行影子链：

```text
ArbitrationComputation/Evaluation
  → group-level ActionPersistencePolicy / ExecutionDirective
  → shared safety seam
  → resolver/planner/reference
  → command intent + execution lineage
```

它不得申请 motion token、不得调用 Rig/position executor/backend，也不得改变 legacy queue/segment。分别与
legacy actual safe/reference/command 对比，验证 shared safety migration matrix、component/coupled-group
directive、segment expiry、no-action、local/global cancellation 和 transform lineage。无法等价的现有 safety
规则必须显式保留在目标 shared/backend 层或阻塞迁移，不能只记录数值差异后继续。

该链生成的 directive/safety/reference/command 均标记 `shadow/proposed`，不得写成 authoritative
ExecutionDirective/ArbitrationDecision 或创建真实 segment。

退出条件：policy/human fixture 的 shared safety/reference/command intent parity 可解释且可重放；shadow
directive 不产生 dispatch/hold/lifecycle side effect。

执行状态（2026-07-19，M5.7a pure seam）：新增 `robot_motion.action.execution_shadow`。该纯 evaluator 接受冻结的
`ArbitrationResult`、调用方提供的 `ActiveExecutionSnapshot`/`ActionPersistencePolicy` 和显式 shared-safety
函数，输出 non-authoritative `SharedExecutionShadowEvaluation`。command 只以 deep-frozen semantic payload/hash
存在，不返回可供 dispatcher 直接消费的 mutable command，也没有 segment/cancellation/permit/backend owner。
selected fixture 已证明其 command value 与当前 legacy canonical resolver 对齐；hold fixture 已证明不会调用 safety
或产生 command intent。

ControlTrace 增加独立 `shared_execution_evaluation` shadow event，显式引用 source `ArbitrationEvaluation` 与同一
computation result，内嵌 proposed directive/safety/command-intent 及各自 hash；它不生成 decision ID，也不复用
authoritative `execution_directive`、`shared_safety_result` 或 `command` event。`authoritative_real_v1` 与
`deployable_sim_v1` 将该事件列为 not-applicable，lineage 会拒绝缺失/mismatch 的 source evaluation 或被标成
authoritative 的记录。

随后新增 `robot_motion.action.keyframe_safety` 的 additive canonical adapter：调用方必须显式提供版本化
`CanonicalKeyframeSafetyProfile` 与 `CanonicalActionSafetyAnchor`，anchor kind 显式区分 previous selected/safe、
`previous_safe_committed`、previous commanded 或 measured state，且 ID/kind/hash/frame 都进入 typed `SharedCanonicalSafetyResult`，包括 fail-closed
结果。数值约束直接复用 `robot_motion.position_control.safety` 的 pose/qpos keyframe primitives，不把 provenance
写进 `RobotAction.metadata`，也不自行决定模型的 `last_action`。v1 只覆盖 frame 完全一致的 arm pose 或 arm qpos；
hand、mixed modality、component/frame mismatch 均 fail closed。它当前只在无副作用 execution-shadow fixture
中使用，并不表示 legacy safety owner 已迁移。M5.7c1/M5.7e1 后当前
control/topology/trace/replay/sim scoped 回归为 252 passed；本机基础 Python 缺少可选 `pyarrow`，因此
`test_dataset_episode_maintenance.py` 未进入该组，不能把此数字解释为全仓或 Arrow integration 验收。

这不代表 M5.7 完成：现有 policy `ActionSafetyFilter` 仍以 normalized action + observation 为输入；新增 adapter
没有选择 production anchor/staleness/coupled-group profile，也未覆盖 hand、human/position/backend safety。
在逐入口完成旧/新双算和 SafetyMigrationMatrix 前，不能用该 adapter 替换当前 safety owner。下一步只有在明确
每个 frontend 的 anchor policy 后，才能接默认关闭的 policy/human comparison；并在 resolver/planner 真正发布
reference 时补 typed reference fact。

#### M5.7b：Safety anchor lifecycle contract（当前 pure contract 已落地）

目标 shared profile 不为所有 frontend 复用一个“last action”。当前冻结的目标语义是：

```text
正常 canonical continuity  = previous_safe_committed
初始化/恢复 rebase          = fresh measured_state
真实 tracking divergence    = 独立 measured-state guard
previous commanded          = backend/legacy comparison anchor
previous selected           = intent audit/rate-limit evidence，不是 shared safety anchor
```

这里 `committed` 不等于 safety 函数返回：必须有显式、accepted 的 execution-target admission 或 direct backend
command receipt；shadow safety、rejected publication/target、未成功 queue admission 都不能更新 target anchor。
anchor scope 是 `control_domain + selection_group + execution_epoch + group_segment_generation`，不是 producer；
同 epoch 的 human/policy/replay handoff 不因 source ID 改变自动清空连续性，HOLD/FAULT/epoch change/group cancel
则通过显式 transition 使对应 group 进入 `REBASE_REQUIRED`。

代码事实说明为什么 M5.7 必须同时保留 legacy-aligned 与 target-chain 两种 comparison：

| 当前路径 | 当前 reference/update 事实 | 与目标 `previous_safe_committed` 的差异 |
|---|---|---|
| policy pose `ActionSafetyFilter` | 仅当整个 deploy `observation.last_action` vector 均 finite 时使用其 active pose slice；常见稀疏 NaN vector 回退 measured pose | `last_action` 由 frontend 更新，且旧 whole-vector gate 不等于目标 per-group previous-safe；尚无 target-admission receipt |
| policy qpos `ActionSafetyFilter` | 每 tick 使用 observation measured qpos，不使用 pose 路径的 last-action helper | 当前是 measured-step guard，不是 previous-safe continuity |
| `run_policy.py` direct/dry-run | `_PreparedActionExecution.accepted` 后更新 `last_action`；dry-run 也 accepted | accepted 不等于 backend 执行 |
| `PositionDeployExecutor.submit()` | keyframe 入异步 queue 后即使尚未 reference/backend dispatch，也使上层更新 `last_action` | queue admission、reference admission、backend receipt 尚未分型 |
| human `TemporaryDualArmSafetyLimiter` | filter 阶段就推进内部 previous output，早于 command guard/dispatch | legacy parity 必须复现；目标链不能把 guard-rejected output commit |
| shared authoritative sim | permit 通过且所有 command receipt accepted 后才建立 active segment | 最接近目标 commit seam，但仍缺独立 typed resolved-target/reference fact |

新增 `robot_motion.action.safety_anchor`，将 lifecycle 与 numeric `CanonicalKeyframeSafetyProfile` 分开：

- `SafetyAnchorPolicy`：固定 v1 normal/recovery kind、freshness 与 source-handoff 规则；
- `SafetyAnchorState/Transition`：按 group/epoch/generation 表达 uninitialized/active/rebase-required 和
  measured rebase、safe-target commit、hold/fault/backend-reject/epoch/group invalidation；
- `SafetyAnchorCommitEvidence`：只接受 typed `execution_target_admitted` 或
  `direct_backend_command_accepted`，rejected evidence 不能推进 state；
- `SafetyEvaluationInputSnapshot`：冻结 selected action、anchor state、measured evidence、cutoff、numeric profile/
  lifecycle policy/divergence guard/schema hash，仍位于 Arbiter 之后；
- `MeasuredDivergenceGuardSpec`：只冻结 enable/threshold/freshness/violation contract；当前没有 validated real
  threshold，也没有接 frontend，因此不激活或冒充已完成 divergence safety；
- 所有 state transition 都是纯函数；没有 registry、worker、segment、permit、backend 或真实 anchor owner。

当前 pure fixture 已验证 fresh measured startup rebase、selected 与 safe 不同只 commit safe、rejected evidence
不更新、execution epoch invalidation、下一 segment generation、snapshot hash determinism 与 immutable action。
M5.7b 尚未增加 ControlTrace event/profile requirement。M5.7c1 已先以 debug JSONL 冻结实际 policy producer
payload；后续 formal shadow trace 接线必须为 safety comparison 定义独立 typed event，不能把它写成 authoritative
`SharedSafetyResult`，也不能由 debug event 反推 anchor transition。

M5.7c/d 双算必须显式区分：

```text
LEGACY_ALIGNED
  新 numeric primitive 使用旧路径实际 anchor，判断数值/故障是否等价

COUNTERFACTUAL_SHARED
  使用独立 shadow SafetyAnchorState，只在 shadow target-admission evidence 上演进，预测 owner 切换后的长期行为
```

两种结果不能混为一个 parity 布尔值；anchor-policy difference、numeric-primitive difference 和 representation-stage
difference 必须分型。M5.7c1 已完成 policy `LEGACY_ALIGNED` arm pose/qpos 双算：它复用唯一 queue-pop emission，
不重跑旧 filter，不申请 permit，也不改变 fault/last-action/dispatch；qpos 中旧 post-safety adapter 补 frame 而
pre-safety/shared safety 不补的事实作为 representation mismatch 单列，不冒充 numeric mismatch。旧 pose filter
只有在整个 deploy `last_action` vector 均 finite 时才使用 previous action，稀疏 NaN vector 会回退 measured pose，
comparison 已按该实际条件复现，而不是按单侧 slice 猜测。

近期顺序调整为：M5.7e1 先冻结 typed `ResolvedExecutionTarget`/admission seam 和 commit linearization point →
M5.7c2 运行 `COUNTERFACTUAL_SHARED` anchor state → M5.7d human direct/position 分 profile 双算 → M5.7e2
reference publication → M5.7f SafetyMigrationMatrix closure。position typed reference 未出现前，
不得用高频 command tick 代替 reference anchor；replay/composer/sim 均绑定 target execution profile，不继承 source
dataset 或 raw residual 的旧 anchor。hand/DMTAC 继续独立 `not-covered/not-applicable`，不复制 arm policy。

#### M5.7c2：Counterfactual shared anchor chain（当前 pure fixture 已落地）

`robot_motion.action.execution_shadow` 现可在显式 active shadow anchor 上构造 immutable
`SafetyEvaluationInputSnapshot`，运行 shared keyframe safety，并只在调用方注入的 typed
`ResolvedExecutionTarget + accepted ExecutionTargetAdmissionReceipt` 后推进 shadow
`previous_safe_committed`。fixture 已证明第一步从 fresh measured qpos rebase，第二步从上一次 committed safe
target 连续限幅；只计算 safety、rejected target 或 direct-command target admission 都不会推进 anchor。

该批故意没有接 `run_policy.py`。当前 direct 与 position frontend 尚未共享一个已冻结的 target binding：pose/qpos
究竟发布 direct command intent、reference 还是 trajectory，必须由 frontend profile/resolver 声明；pure evaluator
不替它选择。底层 target admission 仍是中性 contract，counterfactual wrapper 单独标记
`non_authoritative=true`，不得投影为真实 execution-target/anchor event。measured-divergence evaluator 和真实阈值尚未
落地，enabled guard 当前 fail closed。

#### M5.7d-a：Human direct legacy safety evidence（当前已落地）

canonical GUI 的 `TemporaryDualArmSafetyLimiter` 现在额外生成 immutable、content-hashed 的
`TemporaryDualArmSafetyEvaluation`：逐 arm 保存本次 input、旧 limiter 实际 previous anchor、filter output、shape、
limit/initialize 结果及 limiter profile。该 receipt 只是旧 filter-stage evidence；旧实现仍会在 guard/dispatch 之前推进
自己的 compatibility state，这一行为没有被修改，也明确不能作为目标 `previous_safe_committed`。

`TeleopSession` 新增默认不绑定的 post-filter/pre-guard observe-only hook；canonical GUI 仅在 human shadow 与 temporary
limiter 同时启用时，将同一 `HumanCommandEmission`、immutable filtered-command snapshot 与 limiter receipt 放入原有
bounded shadow observer queue。observer 异常继续与 guard/dispatch 隔离。`HumanLegacySafetyBindingEvidence` 显式验证
limiter input 是否仍等于 human emission、limiter output 是否又被后置 init/fix filter 改写、previous anchor 是否存在、
`command_scale` 是否可由当前 shared primitive 表达。只有全部成立才允许进行 `LEGACY_ALIGNED` numeric comparison。

pure pose fixture 已用 `workspace_enabled=false` 的显式 legacy comparison profile 证明 shared translation/rotation step
primitive 与旧 limiter 输出一致；新增开关默认仍为 `true`，默认 profile 沿用原 v1 hash payload，只有显式关闭时
进入新 hash 字段，因此没有改变既有 canonical safety profile identity。当前尚未将
comparison 注册为 ControlTrace event，也尚未覆盖 qpos、human position reference、counterfactual target-chain、
hand/DMTAC；这些状态必须继续标记 partial/not-covered。

#### M5.7d-b：Human qpos legacy limiter parity（当前 scoped 落地）

新增纯 `build_human_legacy_aligned_safety_profile()`，只从同一次
`TemporaryDualArmSafetyEvaluation` 复制旧 limiter 实际拥有的 numeric bound：pose profile 使用 xyz/rotation step并
显式关闭旧 limiter 不拥有的workspace envelope；qpos profile只使用`max_joint_step_deg→rad`。helper不包含collision、
resolver、reference planner、backend、measured divergence或anchor commit语义。

qpos fixture复用canonical GUI现有`bimanual_qpos_v1 + arm_joint_policy=qpos` adapter、同一human emission、旧limiter
previous-command evidence和post-filter action，验证shared joint-step primitive数值一致。结果仍是
`non_authoritative` filter-stage comparison，不生成`ReferenceSegmentAdmission`、execution-target admission或真实decision。
代码核查同时确认canonical GUI position链并不使用`PositionDeployExecutor`：其qpos command进入
`RM65ArmBackend`自身的background planner/stream；因此policy deploy的legacy reference observer不能冒充human-position
lineage。后续human position closure必须复用backend已有`PositionAuditEvent`/transport-target sequence，或新增等价typed
binding，不能把本批qpos limiter parity写成完整position/reference parity。

#### M5.7d-c：Human position backend receipt/audit binding（当前 scoped 落地）

`TeleopSession`此前调用`BimanualRig.send_commands()`后丢弃返回值，使human legacy observation无法恢复实际backend
queue receipt。本批只做additive passive修复：`HumanLegacyCommandObservation`新增per-effector
`HumanDispatchReceiptSnapshot`，typed`CommandDispatchReceipt`保留accepted/operation/status/time/result/metadata；`None`
和backend-specific对象只记录receipt kind/type，不猜accepted语义。该字段只在既有human execution observer启用时产生，
不改变dispatch顺序、backend输入或owner。

新增纯`bind_human_position_backend_evidence()`，以command与receipt共同保存的
`position_transport_target_sequence`连接既有`PositionCommandAuditEvent.target_sequence`，并分型为
`target_admitted`、`planner_observed`、`backend_accepted_observed`。qpos或sequence不一致fail closed；所有结果固定
`non_authoritative=true, anchor_commit_eligible=false`。它不声明M1C enforcement、canonical reference admission、
segment collision certification或measured motion完成。当前production human-shadow runtime尚未异步订阅position audit
fanout，ControlTrace也未投影该binding，因此本批只是可信pure join与receipt-preservation seam，而非formal capture闭环。

#### M5.7e1：Resolved execution-target admission（当前 pure contract 已落地）

新增 `robot_motion.action.execution_target`，在 shared-safe action 与 worker/reference/backend 之间冻结以下事实：

```text
ResolvedExecutionTarget
  source full safe-action hash
  per-selection-group projected safe action/hash
  direct-command / pose-reference / qpos-reference / trajectory-segment kind
  input action schema / output target schema
  resolver / transform / planner contract hash
  component dependencies
  [scheduled_from, scheduled_until)
  immutable target payload/hash

ExecutionTargetAdmissionReceipt
  receiver + local admitted_at
  execution epoch + group segment generation
  accepted / typed rejection reason
```

receiver admission 对 stale execution epoch、错误 generation、尚未 resolved 与过期 interval 自动 fail closed。
M5.7e1 的 pure fixture 当前允许 pose/qpos/reference/trajectory admission 直接转换成
`SafetyAnchorCommitEvidence`，但代码核查确认该语义只适合 admission guarantee 已足够且没有 future endpoint 的窄
profile；不能直接用于当前 position lookahead/segment worker。direct command intent 固定为
`on_backend_acceptance`，仅有 target admission不能提交 anchor。完整 shared-safe action hash用于证明 source lineage；
anchor state 只提交按 `component_dependencies` 投影的 group action，避免局部 group 携带其他 group 状态。

该批仍没有接 `PositionDeployExecutor`：`submit()` 只是 frontend/source queue admission，不能被包装成 target
admission。后续 worker 必须在 resolver/planner 成功、generation/fault 二次检查通过且 reference/execution target
真正被接纳时产生 receipt；direct path 则等待 backend accepted receipt。当前也没有 runtime target registry、segment
owner、ControlTrace event 或 anchor state owner。

#### M5.7e2：Reference segment、target-time anchor 与 runtime lineage

代码依据：`CausalMotionResampler.sample_at()` 已返回时间上较早/较晚的 source index、timestamp 和 interpolation
alpha；`PositionDeployExecutor._emit_reference()` 已把这些 parent 写入 debug metadata，并在每个高频 tick 后续才运行
collision-aware IK/planner、canonical resolve 和 backend dispatch。因而 segment 的 source parent 与每 tick solver/
measured input必须分别记录，不能从最终 command 反推。

##### M5.7e2-a：纯 segment contract（当前已落地）

只在 `robot_motion` 冻结，不接 frontend、worker、ControlTrace 或 backend：

```text
ReferenceEndpointParent
  role = START | END
  source_kind = MEASURED_REBASE | COMMITTED_SAFE_ENDPOINT | CURRENT_SAFE_TARGET
  source fact ID / projected group-action hash / frame+schema hash
  target_effective_at

ReferenceSegment
  control domain / selection group / execution epoch / group generation
  authority kind = legacy execution observation | counterfactual shadow | authoritative shared
  exactly one ordered START and one ordered END parent in v1
  interpolation / resolver / planner contracts
  input/output schema + component dependencies
  [scheduled_from, scheduled_until)
  admission guarantee + segment safety evidence level

ReferenceSegmentAdmissionReceipt
  receiver-side admitted_at / admission sequence
  accepted or typed rejection
  replaced/superseded segment relation

SafetyAnchorCommitCandidate
  source segment/admission + END parent safe action
  evidence_created_at + effective_from
  required admission guarantee

ReferenceSegmentTerminal
  COMPLETED | SUPERSEDED | CANCELED | EXPIRED | FAULTED
  effective_at + last emitted reference/evidence
```

这些均为不可变事实。`PENDING_EFFECTIVE/ACTIVE/SUPERSEDED/INVALIDATED` 是 timeline resolver 在给定
`safety_cutoff_at + next_safety_target_effective_at + current epoch/generation` 下的派生结果，不是被原地改写的字段。
candidate 只有满足以下条件才可转成当前 safety evaluation 的 `SafetyAnchorCommitEvidence`：admission accepted；
`evidence_created_at <= safety_cutoff_at`；`effective_from <= next_safety_target_effective_at`；所需 guarantee 已满足；
在该目标时刻之前没有 matching supersession/cancel/fault/epoch invalidation。future endpoint 提前 admission 不得覆盖
当前 active anchor。只有 `authoritative_shared` candidate 可转换成真实 anchor commit；legacy passive observation
和 counterfactual shadow 即使数值 eligible，也不能推进 authoritative state。

v1 endpoint parent 固定为两个有角色的端点，但不固定为“两条 safe action”：startup/recovery 的 START 可以是 fresh
measured rebase，连续运行 START 是 previous committed safe endpoint，END 是 current safe target。每个 endpoint
内部可以是 coupled 双臂 projection；禁止用 left/right 命名时间 parent。

`admission_guarantee` 至少区分 queue-only、interpolation-segment accepted、endpoint resolved、trajectory validated、
backend accepted；`SafetyAnchorPolicy`/binding 声明所需等级，不能只按 target kind 猜。segment safety evidence 至少
区分 `UNVERIFIED`、`ENDPOINTS_ONLY`、`DISCRETE_REFERENCE_CHECKED`、`CONTINUOUS_CERTIFIED`；当前 RM65 路径不能
默认声称 continuous certification。

退出条件：canonical hash/immutable payload、parent role/cardinality、scope/interval、guarantee ordering、future
pending selection、supersession/cancel/epoch invalidation和 deterministic replay fixtures 全部通过。

scheduled v1 进一步区分正常`SUCCESSOR`与异常`SUPERSEDES`。successor要求same domain/group/execution epoch、generation
严格+1，且new START的fact/hash/time与predecessor END完全一致；提前admission不终止predecessor、不取消其END anchor
candidate，也不得提前切换active group generation。receiver只建立`PendingGroupGenerationReservation`，到START按顺序先
提交predecessor END candidate，再原子激活reserved generation并退休旧generation，terminal reason为
`retired_by_successor_activation`而不是`completed/superseded`。anchor commit语义明确为
`execution_target_accepted_at_effective_time`，不表示measured target reached。

`SUPERSEDES` v1只支持`replace_pending_before_start`：必须显式引用pending old segment；START前使旧candidate与generation
reservation失效，START后replacement拒绝并要求cancel/HOLD→new epoch或显式generation transition→fresh measured rebase。
无显式relation的重叠pending segment fail closed；同一timestamp以receiver admission sequence定序，不实现mid-segment
replan。`scheduled interval ended`、`execution lineage terminal`与`measured target reached`继续是三个不同事实。

执行状态（2026-07-20）：新增 `robot_motion.action.reference_segment`，实现上述 endpoint、segment、receiver
admission、terminal、commit candidate/assessment/timeline resolution，并在多个同 generation 候选缺显式
supersession 时 fail closed。`SafetyAnchorCommitEvidence/State/SafetyEvaluationInputSnapshot` 同步分离 evidence
`available_at` 与 target `effective_from/target_effective_at`；旧 e1 caller 未提供新字段时仍退化为相同时间，不改变
当前 frontend。新增测试覆盖 measured START、future pending、causal cutoff、guarantee、terminal、ambiguity 和
anchor resolution 和 legacy-observation authority firewall；相关 position/action/sim scoped 回归 157 passed。该结果无 worker caller、ControlTrace producer、
真机或硬件 acceptance。

##### M5.7e2-b：legacy position worker passive instrumentation

本批只建立 legacy 执行事实，不生产 canonical admission。成熟度固定为：

```text
1. LegacyPositionReferenceSegmentObservation
   worker 实际选中 ordered parent pair，observer 第一次看到由其派生 reference 的原始事实

2. BoundLegacyReferenceSegmentObservation
   将原始事实绑定到 typed domain/group/parent/execution context，仍为 non-authoritative legacy observation

3. ReferenceSegmentAdmission
   shared worker 真正执行 M1C epoch/generation 校验后的 canonical admission；不属于 e2-b
```

当前 `PositionDeployExecutor` 没有独立 segment queue，也没有“前置 resolver/planner 检查完成后原子写入 segment”的
单一点。`_accept_submission()` 只把 keyframe 推入 resampler；`_emit_reference()` 才得到真实 earlier/later parent、
timestamp 和 alpha，并随后逐 tick 执行 collision-aware IK/planner。因此 e2-b 的 raw segment observation 线性化语义
固定为 `first_observed_reference_derived_from_exact_parent_pair`，不是 queue/timeline admission；exact-keyframe 和 causal-hold
reference 作为 frame derivation 单独记录，不伪造一个双端点插值 segment。

第一层至少记录 worker/executor identity、executor-local generation 及固定 namespace、ordered START/END source
index/timestamp/stage/payload hash、插值 contract、区间、首次观察时间和 legacy authority。local generation 固定命名为
`position_deploy_executor_local_generation_v1`，不得写入 M1C `execution_epoch` 或 shared group generation 字段。

第二层使用纯 projection，并显式携带 `ExecutionContextBindingEvidence.enforcement_level`：

```text
CONTEXT_ONLY       caller 提供 typed context，worker 未校验
CREATION_CHECKED   worker 在形成/入队时校验
DEQUEUE_CHECKED    worker 在消费时校验
PRE_BACKEND_CHECKED backend 调用前再次校验
```

e2-b 现有 worker 只能声明其实际证据；仅提供 ID/epoch 数值时固定为 `CONTEXT_ONLY`。projection parent 可使用
`LEGACY_POST_SAFETY_KEYFRAME`、`LEGACY_POSITION_KEYFRAME`、`MEASURED_REBASE` 等真实 source kind，不得制造
`SHARED_SAFE_RESULT`。known domain/group/frame/schema mismatch fail closed；无 typed parent provenance 时只保留 raw
observation。

高频 `LegacyPositionReferenceFrameObservation` 引用 raw segment（仅插值时）、scheduled/emitted time、alpha、reference
payload/hash、local generation 和实际可得的 resolver/planner evidence。若每 tick IK 使用新 measured/solver input，则
这些输入/result必须另行引用，不能只凭 endpoint 声称可重放 qpos。observer 默认为关闭，使用非阻塞、故障隔离的
passive sink；开启/关闭不得改变 keyframe queue、parent选择、reference值/数量/时序、IK/planner调用、fault/clear、
command顺序或 dispatch count。

两层 observation 代码级固定 `non_authoritative=true`、`anchor_commit_eligible=false`、无 admission receipt；不得进入
`SafetyAnchorState.advance()`、authoritative segment registry、permit 或 backend。producer 稳定前只做 runtime typed
fact/fixture，不在 `dataset_contracts.control_trace` 提前冻结 public record。

退出条件还包括：每个实际插值 parent pair只产生一个 raw segment observation；每个 frame绑定正确 parent pair；
replacement/cancel只能按真实可观测事实表达；observer overflow/error不进入 legacy fault；projection失败不影响执行；
typed IDs supplied 明确不等于 epoch semantics enforced。

执行状态（2026-07-20）：新增 `teleop_runtime.position_control.reference_observation`，实现 raw endpoint/segment/frame、
`ExecutionContextBindingEvidence`、legacy parent provenance、pure bound projection 和 bounded background observer。
`PositionDeployExecutor` 新增默认 `None` 的可选 observer；仅在 observer 存在时保存实际 resampler source stage/payload
hash，并在 parent pair 已由 resampler 选定后、resolver/IK/planner 前调用 `submit_nowait()`。同一 generation/modality/
parent pair只产生一次 segment observation；exact/hold frame无 segment ID；observer exception/queue backpressure不进入
legacy fault。当前没有 production frontend builder、typed binding caller、完整 post-resolver measured/solver identity、
same-generation supersession producer或 ControlTrace schema，因此本批仍是 scoped compatibility seam，不是 canonical
admission。
新增及较宽 position/control-core/sim 回归 128 passed，无硬件运行。

##### M5.7e2-b3：post-resolver legacy outcome（当前 scoped 已落地）

pre-resolver frame 与 resolver/planner/backend 结果必须保持为两个不可变事实，不能回写或升级
`LegacyPositionReferenceFrameObservation`。当前 worker 存在一个可忠实观察的成功点：collision planner（若启用）和
canonical resolver 已成功、immutable command intent 已形成、executor-local generation/fault 最终二次检查已通过，
但 backend callback 尚未调用。该点生成：

```text
LegacyPositionReferenceResolutionObservation
  source frame observation ID/hash
  resolved_at
  resolver evidence hash
  planner status/input/output/evidence hash
  measured-state evidence status/IDs
  command-intent hash
  local pre-backend generation check evidence
  enforcement namespace = executor-local only
  non_authoritative = true
  admission/anchor/backend accepted = false
```

它只证明旧 worker 对该 frame 形成了什么 command intent以及本地 `_generation` 检查已经通过，不证明 shared M1C
execution epoch、dispatch permit、SDK acceptance或机械臂执行。non-collision pose/qpos path显式记录 planner
`not_applicable`；collision path记录实际 reference input和 planned qpos output hash，但当前 planner内部 measured state
没有 typed ID，因此必须标记 `legacy_planner_state_not_typed`，不能宣称完整 solver replay。

observer继续使用同一 bounded FIFO。resolution job若发现对应 pre-resolver frame因 observer overflow未被记录，可以先
补发同一 deterministic frame/segment fact，再发 outcome；任何 drop/error只影响 completeness，不进入 legacy fault。
planner/resolver/command-construction failure 使用独立 immutable observation，不能覆盖成功 outcome；segment terminal、
same-generation supersession和production binding caller分别由后续子批处理。失败事实只能从现有显式异常/fail-closed
分支被动产生，不从 debug log 反推。

退出条件：direct pose/qpos和collision planner success均引用同一 frame；local pre-backend check与M1C enforcement
分型；command/planner hash确定性；observer开关不改变planner调用、command、dispatch或fault；无 canonical admission、
anchor commit、permit或backend receipt。

执行状态（2026-07-20）：observer FIFO现支持独立 resolution job；若对应 frame job未被接收，会先重建同一
deterministic pre-resolver frame/segment，再输出 `LegacyPositionReferenceResolutionObservation`。executor在实际
local generation/fault最终检查通过后、backend callback前提交 outcome。direct pose/qpos显式 planner N/A；collision
path记录pose-reference input、planned qpos output和endpoint evidence hash，并将内部 measured state标为 untyped。
resolution observer异常继续不进入legacy fault。较宽 position/control-core/sim 回归 131 passed，无硬件运行。

下一子批只处理明确的 legacy generation invalidation terminal：clear/fault/close 可分别产生 canceled/faulted
observation；当前 worker没有正常 segment completed或同-generation supersession原子事件，因此不得推断这两类 terminal。

##### M5.7e2-b4：executor-local generation terminal

`_generation` 原子递增是旧 worker真正使旧 resampler/queue/callback失效的本地线性化点。observer可以在同一临界区
非阻塞提交 `LegacyReferenceGenerationInvalidation`，由 FIFO 对该 generation中已经观察到的 segment派生 immutable
`LegacyPositionReferenceSegmentTerminalObservation`：

- `clear`/`close` → `canceled`；fault latch → `faulted`；
- terminal保留 executor-local generation namespace，固定 non-authoritative、无 shared execution epoch/anchor effect；
- manual rearm通常在 fault已清空旧stream后发生，只记录generation transition，不重复终止不存在的segment；
- timestamp是local generation invalidation的monotonic audit evidence，不能解释为backend已停止；
- terminal job丢失/observer错误只降低trace completeness，不影响hold、fault或worker reset。

当前没有事实依据产生 `completed`，也没有同-generation segment registry可证明 `superseded`；两者继续留空。backend
stop/hold acknowledgement属于 M1C/M5.8，不由 legacy terminal代替。

执行状态（2026-07-20）：observer FIFO按 generation保存已真实观察到的插值 segment；clear/close/manual-rearm/fault
在旧 `_generation` 递增线性化点非阻塞提交 invalidation。clear/close得到 `canceled`，fault得到 `faulted`；无 active
segment时不制造terminal。terminal固定 `executor_local_generation_only`、`backend_stop_acknowledged=false`、
non-authoritative。正常 completed和同-generation superseded仍不推断。较宽回归 132 passed，无硬件运行。

##### M5.7e2-b5：resolver/planner failure observation（当前 scoped 已落地）

`LegacyPositionReferenceResolutionFailureObservation` 只记录旧 worker 已真实进入的失败分支：source frame/hash、
executor-local generation namespace、`planner/resolver/command_construction/final_local_guard` stage、原异常类型/消息和
deterministic evidence hash。它固定 `non_authoritative=true`、`anchor_commit_eligible=false`、无 admission receipt，
也不声明 backend 已拒绝或机械臂已停止。

`PositionDeployExecutor` 当前在 collision planner exception、显式 endpoint fail-closed、canonical resolver exception和
command construction exception旁路提交该事实；observer失败仍不改变原异常、fault latch、hold、arm/hand dispatch数量。
集成 fixture 已验证 collision hold 在任何 arm/hand dispatch 前继续 fault，同时产生 planner-stage failure observation。
`final_local_guard` 已冻结为合法 typed stage，但只有发现代码中存在独立、可观测的该类失败点后才允许生产，不能为了
覆盖枚举而复制其他异常。该子批仍不是 shared rejection/admission、M1C cancellation ack或 ControlTrace public schema。

production binding核查结论：`run_policy.py` 仅在 Arbiter shadow模式已有稳定 runtime session/control-domain identity，
但没有可复用的shared execution epoch、group generation或formal reference trace sink。当前不得只填前两个ID后将
legacy observation包装成“bound production fact”；该缺口等待M1C/shared execution context owner明确。下一主线先在
已有完整shared context的MuJoCo authoritative runtime完成e2-c。

##### M5.7e2-c：sim authoritative segment integration

在 shared MuJoCo authoritative runtime 中让 segment/admission/terminal 成为真实执行事实，验证 future pending
commit 只在 target time 生效、replacement/cancel使其失效、no-action不创建 segment、hold/fault终止 segment，且
reference frame不会增加 Arbiter decision 或反复推进 anchor。pose per-tick IK 必须保留 solver/state lineage。

代码核查将本批拆成以下前置门槛：

```text
M5.7e2-c0  immutable authoritative dispatch context seam       ✅ scoped
M5.7e2-c1  dual profile + receiver activation semantics        ✅ pure/default-compatible
M5.7e2-c2  qpos segment admission/terminal/anchor timeline      ✅ scoped deterministic sim
M5.7e2-c3  actual 120 Hz planner/qcmd lineage                    ✅ scoped deterministic sim
M5.7e2-c4  active generation + pending reservation handoff       ✅ scoped deterministic sim
```

e2-c0 新增 backend-neutral `AuthoritativeCommandDispatchContext`：它在最终 motion permit 校验通过后，将 exact
decision/directive/shared-safety/command ID、safe action/hash、target effective time、authority/execution epoch、
per-group generation和selected proposal ref交给可选 contextual dispatcher。普通单参数 dispatcher继续可用；sim
contextual seam仍转发到当前 `core.dispatcher`，因此现有backend fault注入、receipt和调用顺序不变。context本身明确
不是 admission evidence；receiver仍必须在自己的原子接纳点检查live epoch/generation。

e2-c1 不修改现有zero-lead行为。MuJoCo immediate profile中qref在decision时立即进入mailbox，而
`ReferenceSegment`的END target time也等于该decision时刻，因而合法receiver admission窗口已经结束。两条语义必须
以独立版本化profile并存：

1. `immediate_qref_sim_v1`：`target_lead=0`，`activation_mode=IMMEDIATE_TARGET`；只记录 immediate target/queue/
   backend evidence，不生成 future `ReferenceSegment`；
2. `scheduled_reference_segment_sim_v1`：`activation_mode=SCHEDULED_SEGMENT`；第一版要求receiver在segment START
   前接纳，late admission固定reject-and-hold，不实现mid-segment catch-up；END只在target effective time后可提交anchor。

scheduled sim fixture 的lead不使用任意正数。v1绑定要求：

```text
target_lead_time
  >= selected_keyframe_period
   + validated publication_to_receiver_budget
   + scheduling_guard_margin
```

当前deterministic 30/60 Hz fixture先将“预算+余量”保守物化为一个reference period，因此target lead为
`1/30 + 1/60 = 50 ms`；第一个segment的START对齐到下一reference tick并来自fresh measured rebase，END位于decision
target time。该数值不是real production默认值，真实profile必须由p95/p99/deadline evidence发布。

新增窄 `ReferenceActivationSpec`，至少冻结activation mode、schedule hash、admission deadline、minimum margin、
first-segment、late-admission、reference-tick alignment、anchor-commit和replacement policy。schedule/capability/spec
不一致启动即fail closed。该spec不拥有scheduler、worker或anchor state。

禁止通过把 `admitted_at == scheduled_until` 当作未过期、回填更早时间或在 `step()` 返回后补造receipt来绕开该前提。

`ReferenceSegmentTerminal` 同步明确两个时间和receiver顺序：terminal事实只有在`available_at <= safety_cutoff_at`
时在线可见，再由`effective_at <= target_effective_at`判断目标时刻是否失效；同时间事实按receiver admission sequence
排序。现有`recorded_at`在迁移期只作为`available_at`兼容别名，不再承担模糊语义。

执行状态（2026-07-20）：新增`robot_motion.action.reference_activation`，冻结上述immediate/scheduled spec与exact
schedule/capability hash绑定。MuJoCo默认仍为`immediate_qref_sim_v1`且zero lead/dispatch行为不变；scheduled
profile按30/60 Hz生成50 ms lead，并已由c2 scoped receiver承接。
`ReferenceSegmentTerminal`已增加receiver-side `available_at + terminal_admission_sequence`，timeline只消费cutoff前可见
terminal；`recorded_at`仅保留builder/property兼容入口。新增fixture覆盖不足lead/margin、binding mismatch、同时间sequence
和晚到terminal，相关聚焦回归117 passed，无硬件运行。

##### M5.7e2-c2：scheduled qpos receiver timeline（scoped 已落地）

第一版仅覆盖MuJoCo coupled bimanual qpos，不扩pose/IK。落位与线性化点固定为：

```text
AuthoritativeCommandDispatchContext
  → receiver second live check
      domain / authority epoch / execution epoch / group generation
  → build ordered START/END qpos segment
  → receiver timeline admission
  → ReferenceSegmentAdmissionReceipt
  → downstream future qref mailbox（失败则生成fault terminal）

worker clock reaches START
  → activate admitted qref in acceleration-limited planner
  → later high-rate frame lineage belongs to c3
```

`ReferenceSegmentTimeline`保存不可变admission/terminal/candidate，不使用future admission覆盖mutable current slot。第一次
START为admission时fresh measured qpos的typed rebase，effective time为下一reference tick；steady START必须精确复用前一
END parent/hash/time。END来自context中的shared-safe action与target time。receiver admission sequence独立于command ID；
segment identity由ordered endpoint、scope/epoch/generation、interval和activation/interpolation contract决定。

第二次live check必须是receiver真实执行的检查：context generation应等于core当前active group generation的下一代，
execution/authority epoch必须仍匹配；检查失败不得写timeline/mailbox，并返回typed rejected receipt。upstream motion permit
通过不替代这一步。scheduled qref在admission后先保持pending，worker不得在START前交给planner；mailbox receipt只证明
timeline/queue接纳，planner activation与后续backend command仍是下游事实。

replacement/cancel/fault必须生成terminal并使尚未生效candidate失效；no-action不得调用receiver或创建segment。c2只验证
pending→active timeline与取消/替换，不冻结public ControlTrace event；c3稳定frame producer后再进入e2-d。

实际落位增加了receiver侧第二次校验contract：dispatch context现携带permit/token/owner generation；receiver在自己的
接纳锁内重新捕获live owner、authority epoch、execution epoch和下一group generation。校验通过后才建立ordered
segment、receiver admission、pending anchor candidate并写入future qref mailbox；planner在worker time到达START前不消费
该qref。第一次START来自admission时的sim measured qpos，连续segment START复用前一END；HOLD/cancellation生成terminal并
使pending candidate不可提交。当前保证严格限定为single-thread deterministic MuJoCo，不能外推为M5.8真实worker原子锁、
dequeue/pre-backend二次检查或硬件停止ack。receiver timeline admission是c2线性化点；future qref mailbox是下游，queue
失败会生成fault terminal，不能把两者描述成跨对象原子提交。

c4进一步消除了“future successor admission提前切换generation”的错误：authoritative core和receiver均分离active与pending，
successor admission只预留`active+1`；Rig在START接受qref前调用receiver activation guard，随后每个planner/qcmd tick再次
校验active segment/epoch/generation。正常successor到START才退休predecessor，且使用
`retired_by_successor_activation`而非`completed/superseded`；无显式关系的第二个pending segment fail closed。
`supersedes_pending`仅允许START前替换并使旧candidate/reservation失效，当前尚无production runtime caller，不能隐式启用。

c3代码核查确认：当前`MujocoPositionExecutionRig`没有独立60 Hz插值reference producer。
scheduled qref在START进入`AccelerationLimitedJointTrajectoryPlanner`后，planner按120 Hz command cadence产生
`JointTrajectoryProposal/qcmd`；`reference_hz=60`目前只进入planner的reference period/target-velocity语义。因此c3不能
直接补造60 Hz `ReferenceFrameEmission`。已确认并采用A：按真实链路记录120 Hz planner proposal/follow qcmd/backend
accepted qcmd并引用segment，不新增60 Hz执行层。

c3 scoped producer在worker实际START激活点生成target-activation observation，在每个真实command-due且backend接受后生成
planner-command-frame observation。它记录planner token/generation、qref、planner proposal、follow output、accepted qcmd、
limit flags和segment/epoch/generation。一次decision可对应多个frame；frame不产生decision或anchor commit。若END target time后
planner仍在向endpoint收敛，后续qcmd仍可引用最近segment，这表示执行派生关系，不声称机器人已在END时刻到位。immediate
profile没有canonical segment，因此不会生成这些scheduled lineage observations。

##### M5.7e2-d：ControlTrace 与 completeness

只在 e2-b/c 有真实 producer 后冻结 explicit record/projection：`reference_segment_admission`、
`reference_frame_emission`（或连续 range）、`reference_segment_terminal`、`safety_anchor_commit/transition`。formal
profile 校验 safe→resolved target→segment→reference range→command/receipt 的 ID/hash/interval/epoch lineage，而非只
检查 event presence。`ActualDecisionView` 仍一行一个 authoritative Arbiter decision，不按 reference frame扩行。

执行状态（2026-07-21）：scoped scheduled producer已具备explicit projection。当前logical records为
`reference_receiver_validation`、`reference_segment`、`reference_segment_admission`、
`group_generation_reservation`、`group_generation_activation`、`safety_anchor_commit_candidate`、
`reference_target_activation`、`planner_command_frame`和
`reference_segment_terminal`。lineage检查验证END parent对应shared-safety fact、admission对应exact receiver validation与
segment hash、candidate对应accepted admission、planner frame对应唯一activation，以及terminal的last frame归属。
scheduled event family现按分支闭合，START前HOLD不要求伪造activation/frame。该范围已由deterministic和真实MuJoCo
composition验证WAL/seal/commit/reader，并验证seal后、commit前进程终止保持uncommitted但可审计恢复；仍不等于长稳、
production sim entrypoint或real worker profile。

M5.5e 依赖同步调整：e2-a 后可并行 M5.5e1 async harness；e2-c1–c3 后才进入 M5.5e2 formal sim capture。

#### M5.7 SafetyMigrationMatrix（代码事实盘点 v1）

本表只确认当前 owner 与目标层，不表示规则已经迁移。`preserve-current` 表示 M6 前必须继续由旧 owner 执行；
`pending-contract` 表示缺少明确语义时禁止替换为 identity safety。

matrix从现在起按execution profile和gate阶段逐步收口，不等待e2全部结束后一次填写。每个profile同时标记
`covered/blocked/not_applicable`以及`required_before_M5.8a/M5.8b/M6`；M5.8a只做无行为permit wrapper inventory，
M5.8b要求真实legacy mediation，M6才要求validated measured-divergence阈值。暂未覆盖的human qpos/position、Inspire
hand和DMTAC不能用arm pose evidence代替。

| 当前规则/证据位置 | 当前语义 | 目标层 | 当前迁移状态与前置证据 |
|---|---|---|---|
| `robot_motion.safety.SafetyMonitor` + `TeleopSession.run_step()` | required/missing/invalid/stale channel、pose validity、translation/rotation jump、minimum control rate；fault 后 freeze/recovery | sensor/source validation + supervisor/gate；不是 action blend/safety modification | `preserve-current`；需把 violation→gate/hold/fault 映射做 typed shadow parity，低控制率仍按现有 non-freezing profile 处理 |
| `TeleopSession` tracker-loss state machine | grace→automatic hold→manual hold、`Rig.hold_motion()`、长时间丢失要求重新 home | frontend health detector → execution gate/cancellation + lifecycle authorization | `preserve-current`；需证明两个 threshold、baseline reset、recovery healthy frames、home-recalibration latch 完整保留 |
| `TeleopSession.command_guard` | 通用seam任意异常均fail closed；canonical GUI当前只在`fix_arm_controller + dual_arm_retargeter`同时存在时绑定`fixed_arm_final_collision_fault()`，其余profile为`None` | fix-arm source rewrite后的coupled final-pair planner guard；失败进入execution gate/cancellation | `profile-enumerated/preserve-current`；已排除“所有GUI都有未知guard”的误解，但fix-arm safe/reject/exception→typed gate parity仍未完成，相关profile阻塞M6 |
| canonical GUI `TemporaryDualArmSafetyLimiter` | filter 阶段以自身 previous output 做 pose/qpos step/scale，并在 guard/backend acceptance 前推进内部 state | legacy comparison evidence；目标 normal anchor 仍为 accepted reference/direct backend receipt 后的 previous-safe | `human-arm-legacy-evidence-active`；exact receipt、同-emission hook及pose/qpos numeric parity已完成；counterfactual runtime caller和typed trace尚未完成 |
| `ActionSafetyFilter` representation checks | action shape/finite、pose/quaternion、arm/hand spec、active side 与 frame 可解释性 | source validation / representation adapter | `preserve-current`；迁移后 rejection 不能成为 selectable proposal |
| `ActionSafetyFilter._resolve_safety_overrides()` | policy metadata 只能收紧已启用的 xyz/rotation/qpos/hand step bound，非法 override 使整条 action fault | typed safety profile admission；不得继续读任意 audit metadata | `pending-contract`；需把可收紧字段移入 semantic contract/hash，并证明不能放宽 baseline |
| `ActionSafetyFilter` pose workspace/xyz/rotation step | workspace soft bound、相对 reference 的 translation/rotation keyframe limit、quat normalize；旧 helper 要求整个 deploy `last_action` finite，否则回退 measured pose | shared canonical `RobotAction` safety | `legacy-aligned-shadow-active + counterfactual-pure`；M5.7c1 已验证 numeric primitive/override/fault；M5.7c2 已证明 typed target admission 后 previous-safe continuity，但 pose frontend target binding/runtime caller 尚未接 |
| `ActionSafetyFilter` qpos/hand | qpos 当前每 tick 相对 measured qpos 做 hard envelope/step；hand `[0,1]` clip；任一 arm fault时 all-or-nothing command clear | shared canonical safety + measured divergence + component/coupled-group policy | `arm-legacy-aligned-shadow-active + qpos-counterfactual-pure`；两步 previous-safe fixture 已通过；measured divergence、hand、coupled fault/clip 与 frontend target binding 仍未完成 |
| `PositionDeployExecutor` keyframe/anchor/rearm | 15 Hz keyframe safety、reference anchor domain、fresh measured qpos/pose 才允许 manual rearm；fault latch 后 clear queue + hold | shared safety result + execution gate/cancellation + lifecycle recovery | `preserve-current`；M1C epoch 必须替代 generation clear 后再迁 owner，rearm 不能由普通 READY 事件绕过 |
| `RM65PositionCollisionPlanner` | coupled IK、joint envelope/step/velocity、collision endpoint/backoff/hold、solve budget | Arbiter 下 shared planner/IK safety | `preserve-current`；当前仅 endpoint collision evidence，`segment_collision_certified=False`，不得在 trace/验收中升级为连续轨迹认证 |
| `classify_position_trajectory_policy()` + qref planner | collision/jump hard-stop，singularity/time-budget/step-limit degraded speed，quick-stop 保持最后 accepted qpos | shared reference/trajectory safety + cancellation/hold | `preserve-current`；需将 quick-stop/degraded reason 与 reference segment lineage 分型后再做 parity |
| `require_arm_backends_ready_for_position_dispatch()`、Rig/SDK receipts | SDK ready、background target/servo/ESTOP/follow-state、missing/failed receipt | backend command guard | `preserve-current + human-position-pure-binding`；human observation现保留typed queue receipt并可按transport-target sequence关联planner/SDK audit；production async join/ControlTrace和M1C enforcement未接。M5.8仍需在最终backend seam接窄permit |
| DMTAC/hand 独立 worker | actuator mapping、range、USB/worker health、可能在 `TeleopSession` 外发送 | source/end-effector adapter + backend guard + 同一 domain dispatch owner | `not-covered`；未完成 passive emission/最终 seam 前，相关 profile 必须 `not_applicable` 或阻塞，不得声称 hand safety parity |

基于该盘点，下一代码批不能直接把 legacy `ActionSafetyFilter` 整体包装成 shared safety：它同时混合 source
interpretation、模型专属 normalized layout、基于 `last_action` 的 temporal bound 与 canonical modification。最小正确
拆分顺序是：先提取不依赖 normalized layout 的 qpos hard-envelope/finite 纯函数及 typed anchor contract；再以旧/新
双算 fixture 验证 pose workspace/step；最后才把 fault→hold/cancellation 与 position planner 串到默认关闭 frontend。

M5.7f 的退出产物按 execution profile 组织，而不是只维护一张无法判定可启动范围的总表。第一批至少分别覆盖
`policy_direct_pose_v1`、`policy_arm_qpos_v1`、`human_direct_pose_v1`、`human_direct_qpos_v1`、
`human_position_collision_aware_v1`、`replay_to_position_v1`、`hand_inspire_v1` 和 `hand_dmtac_v1`。每条规则只能
落到 `PRESERVED_SOURCE_VALIDATION`、`MOVED_SHARED_SAFETY`、`MOVED_REFERENCE_PLANNER`、
`MOVED_BACKEND_GUARD`、`REIMPLEMENTED`、`INTENTIONALLY_REMOVED` 或 `NOT_APPLICABLE`，并引用 evidence。
仍含 `pending-contract`/`partial-shadow-contract`/`not-covered` 的 profile 必须禁止进入后续 gate，不能用其他 profile
的完成状态代替。

执行状态（2026-07-20）：`robot_motion.action.safety_migration` 已落地纯、可hash的profile gate contract。每条规则
必须分别声明 `m5_8a_inventory`、`m5_8b_legacy_mediation`、`m6_shared_authority` 的
`ready/blocked/not_applicable`；`ready`必须引用evidence，`blocked`必须给出原因，`not_applicable`必须引用profile排除
证据。M6-ready规则还必须具有上述最终disposition及其证据。纯评估明确验证“M5.8-a可盘点”不会自动放行M5.8-b或
M6。首个arms-only候选现已将下表相关项收敛为绑定manifest/exact inventory的scoped matrix；它只签署
`m5_8a_inventory=ready`，全部included rule在M5.8-b/M6仍blocked。其余profile仍是人工盘点；production profile owner、
真实mediation和硬件验收未齐前不得提升状态。

首个待签署release candidate冻结为`m6_candidate_policy_arm_qpos_v1`：motion authority仅policy；controlled components为
coupled left/right arm qpos；operator/runtime safety supervisor只通过`SAFETY_CONTROL`执行HOLD/ESTOP/manual stop，
operator-authorized recovery state machine只通过`LIFECYCLE_CONTROL`执行recovery/home/rearm，human motion proposal
authority明确`not_applicable`。authoritative decision为30 Hz；scheduled reference segment启用，实际planner/qcmd cadence
由profile配置，当前sim evidence为120 Hz。`reference_timing_semantics_hz=60`只有确实参与activation/velocity/capability
预算时才声明，`actual_reference_frame_producer=not_applicable`，不得虚构60 Hz emission。Inspire/DMTAC/replay authority、
pose impedance、fix-arm human motion与human takeover均用manifest/component-catalog hash作为显式exclusion evidence。
放行单位是该ExecutionManifest，不是整个DexTeleop。

执行状态（2026-07-20）：`robot_motion.action.execution_manifest` 已将上述候选固化为纯、可hash的
`ControlExecutionManifest`，并把不存在的60 Hz producer明确记为`not_applicable`。scheduled formal fixture在episode
bind前将该manifest投影为durable ControlTrace dependency，seal/reader会校验其component catalog与ArbitrationInput、
ReferenceSegment input schema一致；manifest存在不等于SafetyMigration gate通过或motion已获授权。

policy safety override也已增加`SafetyConstraintOverrideSpec/AdmissionReceipt`纯契约：v1只允许
`deployment_session_static`，启动preflight一次性绑定artifact、baseline profile、action space与selection group，并逐字段
证明requested bound不大于且不禁用已启用baseline。production `ActionSafetyFilter`尚未改为消费receipt，旧per-action
metadata行为仍保持；在正式迁移前不得把pure admission fixture标为runtime closure。

当前按代码事实形成的profile gate台账如下。“可进入M5.8-a inventory”只允许枚举并包裹调用点，不表示可运行真机
mediation，更不表示可切shared authority：

| Execution profile | 当前可用evidence | M5.8-a inventory | M5.8-b legacy mediation | M6/shared authority |
|---|---|---|---|---|
| `policy_direct_pose_v1` | M3 passive emission；M5.7c1 legacy-aligned pose；M5.7c2 pure anchor chain | `READY_FOR_INVENTORY`：direct arm/impedance final seam | `BLOCKED`：real token/epoch wrapper、cancellation ack和trace profile未接 | `BLOCKED`：target binding、measured divergence、safety matrix未闭合 |
| `policy_arm_qpos_v1` | M3 passive emission；qpos legacy-aligned/counterfactual pure；legacy position reference observation | `READY_FOR_INVENTORY`：PositionDeployExecutor→Rig/backend seam | `BLOCKED`：worker M1C guard和real receipt/ack未接 | `BLOCKED`：real canonical segment owner、divergence和rearm parity |
| `human_direct_pose_v1` | same-emission filter receipt；pose numeric parity；legacy execution observation | `READY_FOR_INVENTORY`：BimanualRig→RM65 impedance/direct seam | `BLOCKED`：typed final permit与receipt completeness未接 | `BLOCKED`：accepted-target anchor runtime、fix-arm profile parity |
| `human_direct_qpos_v1` | qpos numeric parity；typed queue receipt preservation；transport-target audit pure join | `READY_FOR_INVENTORY`：RM65ArmBackend queue与background SDK seam | `BLOCKED`：production audit join、dequeue/pre-backend epoch check未接 | `BLOCKED`：accepted-target anchor、divergence、stop acknowledgement |
| `human_position_collision_aware_v1` | dual-arm retargeter/planner旧证据 + human qpos/receipt pure join | `READY_FOR_INVENTORY`：coupled planner、queue、background SDK seam均需列入 | `BLOCKED`：fix-arm final collision parity、quick-stop/degraded lineage和real guard未闭合 | `BLOCKED`：coupled planner/shared segment ownership未迁移 |
| `replay_to_position_v1` | typed source/target assessment、GUI passive shadow、verified sim authority | `SCOPED_SEPARATELY`：可盘点但不得继承policy/human授权 | `BLOCKED`：real replay execution authorization不存在 | `NOT_M6`：真机replay属于独立M6.5高风险gate |
| `hand_inspire_v1` | in-session human proposal/legacy dispatch基础事实 | `READY_FOR_INVENTORY`：hand backend final seam单独盘点 | `BLOCKED`：hand safety/receipt/token profile未闭合 | `BLOCKED` |
| `hand_dmtac_v1` | 当前在`TeleopSession`外，human shadow显式fail closed | `BLOCKED`：先建立同一domain passive emission与最终seam inventory | `BLOCKED` | `BLOCKED` |

因此M5.7f当前是“首个arms-only profile的M5.8-a scoped matrix已形成，real mediation/shared authority closure未完成”。
不能把该matrix或表中`READY_FOR_INVENTORY`解释成M5.8-b/M6已放行。其余profile仍只能引用稳定typed evidence继续
收敛，尤其不能让hold/home/rearm共用motion token，也不能把mock结果写成real acceptance。

measured-divergence 的真实阈值不是 M5.8-a 无行为 wrapper 的前提：legacy-owner gate profile 可以显式声明 shared
guard disabled，由旧 safety 继续拥有运动语义。但 M6 shared authority 前必须提供 per-action-space threshold、
measurement freshness、violation scope/policy 和硬件 evidence，不能为了进入 M5.8 临时猜数值。

### M5.8：legacy owner 下接入 dispatch gate

前提：M5.7 parity 通过；M1D 三类 permit contract 稳定。

本批拆为两个门槛，整个阶段只向现有 legacy control path 发放 current motion token；shared arbitration runtime
始终无 token。现有 hold/stop/disable 使用受限 `SAFETY_CONTROL` permit，startup/home/servo-enable/rearm 使用
lifecycle-state-machine 授权的 `LIFECYCLE_CONTROL` permit，禁止通过一个宽泛 bypass 绕开 gate。

#### M5.8-a：dispatch seam inventory 与无行为 wrapper

- 枚举 direct arm、position worker、hand、hold/stop、home/rearm 的全部实际 backend call；
- 为每个启用 profile生成完整 seam inventory，任一遗漏都 fail closed；
- 接 typed permit wrapper，但 legacy 仍是唯一 token holder；
- 只用 fake/dry-run 验证 command 数值、次数和顺序不变，以及 token/generation/authority/execution epoch mismatch
  在 backend call 前得到 typed rejection；
- 不运行真机、不把 wrapper fixture 记为 real mediation。

M5.8-a1 已完成纯 wrapper 与第一轮代码 inventory。`robot_motion.runtime.dispatch_seam` 现在提供
`DispatchSeamSpec/DispatchSeamGate/DispatchSeamValidationReceipt`：它在 fake callback 前验证 exact permit kind、
exact safety/lifecycle operation、owner generation 与 authority/execution epoch；拒绝不调用 callback，接受后立即调用
一次并保留原返回值/异常。该模块没有 production builder，也没有安装到 Rig、worker 或 SDK，不能记为真实 gate
接线。

M5.8-a2 已补充 `DispatchPermitInvocation/PermittedDispatchAdapter` scoped fake fixture：legacy 单参数 callback 可在不
改变 payload、返回类型、调用次数和顺序的前提下被包裹；permit provider 必须针对每个具体 final call 返回新 permit
与该次读取的 live epoch，adapter不缓存上层 permit。direct callback、三次异步 tick、epoch mismatch、one-shot reuse、
exact safety/lifecycle operation和原 backend exception均已有无硬件测试。它仍不是 production接线，也不提供真实
legacy epoch owner。

M5.8-a3 已新增纯 `DispatchSeamInventoryEntry/Manifest/Assessment`，并由
`teleop_runtime.backends.arms.rm65_dispatch_inventory`为`m6_candidate_policy_arm_qpos_v1`按真实代码构造16条清单：
`send_command` logical target、background dequeue、最终`_send_joint_target()`、qpos-init home、local rearm、
force-position startup、quick-stop operation/tick，以及hold/disengage/background-fault/ESTOP/startup-rollback/close
的stop与disconnect。清单显式区分同一`_send_joint_target()`在普通motion、quick-stop tick和home中的三种permit语义；
fallback loop标明每个实际尝试仍需新one-shot permit。纯评估当前返回`inventory_structurally_complete=true`，但所有
entry均为`inventoried_unwrapped`，因此`legacy_mediation_ready=false`且永不授予execution authorization。该结论只完成
arms-only候选的代码审计边界，不表示production wrapper、M1C worker guard、drain ack、硬件接线或M5.8-b已完成。

M5.8-a4 已新增default-off `LegacyControlDomainExecutionContext` pure composition：同一个arms-only manifest/inventory
绑定唯一legacy owner registry/token、execution epoch registry、selection-group generation和epoch guard；构造后snapshot
固定显示`composition_mode=fake_only_unbound`、`production_binding_enabled=false`、无hardware authorization。fake final
motion path要求`ExecutionWorkItem`先通过`before_backend_call`的domain/execution/group-generation检查，随后才针对同一个
inventoried pre-SDK seam签发one-shot motion permit并由owner gate复查authority/execution。stale group、非final logical
admission、不同inventory assessment和permit reuse均fail closed。当前两个检查仍未在真实worker同一临界区内线性化，
所以这只是composition/fake wiring证据，所有RM65 entry仍为`inventoried_unwrapped`。

M5.8-a5 已把抽象`software_queue` ack收紧为pure worker drain contract。每个required reference/position/backend/
controller/follow worker先产生`WorkerEpochDrainReceipt`，至少证明已观察新execution epoch、old-epoch in-flight为零且
`no_future_old_epoch_calls=true`；仅清空queue、仍有在途调用或未观察新epoch均拒绝。`WorkerEpochDrainAggregate`要求
profile声明的全部participant都有accepted receipt，之后才可向既有`CancellationAcknowledgementTracker`投影一个
`software_queue` ack。首个arms-only profile现另有scoped `RM65ArmsOnlyWorkerDrainPlan`，显式列出
`PositionDeployExecutor._run`以及左右两个`RM65ArmBackend._background_stream_loop_impl`为三个独立participant，并绑定
exact inventory seam；其plan状态仍是`scoped_unbound`，不声称production worker已经发出receipt。M5.8-a5本身只定义
证据contract；后续b1虽然加入default-off代码检查点，但仍没有SDK、backend-state或measured-stopped acknowledgement，
因此不能用于真实handoff。

M5.8-b1 的第一批无硬件接线随后完成，但不改变上述production结论。`EffectorCommand/TeleopCommand`新增独立于
audit metadata 的可选typed `ExecutionWorkItem` carrier；未配置guard时为`None`且legacy行为保持不变。显式配置后，
`PositionDeployExecutor`要求keyframe在queue admission时携带`reference_segment` work，在submission dequeue和reference
callback前复查同一`ExecutionEpochRegistry`，再派生typed `position_command` work；`RM65ArmBackend.send_command()`在target
admission复查并按side/transport sequence派生`backend_command` work，background worker在取当前target和真正
`_send_joint_target()`前再次检查。取消后的software drain固定按reference worker→left stream→right stream顺序；每个
participant等待已经跨过检查的调用退出、清除旧epoch work并返回现有`WorkerEpochDrainReceipt`，全量齐备后才形成aggregate。
阻塞fake SDK fixture已证明drain不会在旧调用完成前返回，返回后旧target不再尾随发送且旧producer无法重新入队；相关
相关回归通过。

第二批无硬件收口已将取消后的 safety work 与普通 motion work 严格分型。`ExecutionWorkItem`新增
`safety_backend_command`，必须同时引用 bounded `SafetyControlOperation` 和具体 tick evidence，且不得继承
decision/reference lineage；普通 motion 在 fault-latched epoch 继续被拒绝，只有该显式 safety work 可执行。
`LegacyControlDomainExecutionContext`现可在自己的`ExecutionEpochRegistry`先线性化 cancellation，再让同一个
`DispatchOwnerRegistry`观察新epoch，随后 admission RM65 quick-stop operation，并为每个tick生成新的work、evidence和
one-shot `SAFETY_CONTROL` permit。HOLD/fault fixture均验证通过。RM65 backend若在epoch-mediated模式下仍试图让
legacy software quick-stop继承普通motion item，会在修改mailbox和SDK调用前fail closed；production必须走独立的新epoch
safety operation路径。

deadline/trace语义也已落地到pure/default-off范围：晚于`acknowledgement_deadline_at`才到达的ack会保留为事实，但不能
把cancellation标记complete或消除FAULT escalation；`safety_control_operation`、`safety_control_tick`、
`execution_worker_drain`、`execution_worker_drain_aggregate`和`execution_fault_escalation`已有explicit ControlTrace
record/projection与consumer fixture。它们尚未由canonical真机frontend写入WAL，因此只是可持久化contract，不是现场
durable evidence。

第三批无硬件composition已增加`RM65LegacyWorkerExecutionBinding`：它从不可分割的manifest/inventory/drain/matrix
evidence bundle创建唯一`LegacyControlDomainExecutionContext`，并向position reference、left RM65 stream、right RM65
stream三个真实worker class提供同一个`ExecutionEpochGuard`和各自participant spec。canonical policy
`build_arm_backends()`及position executor构造点已有仅程序化、无CLI开关的default-off seam；fake fixture验证三者共享
同一registry。该binding固定`fake_only_unbound`，检测到真实arm IP立即拒绝，不能被误当成production authorization。
worker drain trace在控制动作完成后单独projection；任一sink失败会产生显式partial admission report并使证据不完整，
但不会反向阻塞或撤销已完成的安全取消。

该批仍是`default-off / production-unbound`：canonical deploy builder已有fake-only唯一context注入seam，但正式入口没有
CLI/production activation，真实线程没有执行该模式，新增execution facts也没有由现场runtime写入ControlTrace或调用stop。quick-stop的
新epoch work/permit contract已完成，但尚未接入真实background quick-stop producer。在production composition和现场
验证闭合前，plan继续保留`scoped_unbound`，matrix中的
worker drain与M5.8-b/M6 gate仍为blocked；代码也显式拒绝把当前partial epoch/drain binding构造到
`connect_hardware=True`的RM65 backend，避免pure/fake能力被误接成真机授权。

与上述证据同步，`m6_candidate_policy_arm_qpos_v1`已有scoped SafetyMigration matrix：9条included rule在M5.8-a均
ready并引用manifest/inventory/typed contract hash；M5.8-b仍因production caller、真实worker guard、drain/SDK receipt等
逐条blocked，M6仍因real safety parity、measured-divergence阈值、恢复/停止硬件evidence等逐条blocked。6项manifest
exclusion在三个gate均为not-applicable。matrix不授予permit，也不改变inventory integration status。

M5.8-a6 已增加默认`evidence_only_unbound`的候选 composition bundle，将同一 execution manifest、16条exact seam、
inventory assessment、三participant drain plan与SafetyMigrationMatrix的三个gate结果绑定为一个确定性hash。builder只允许
`m5_8a_inventory=ready`且`m5_8b_legacy_mediation/m6_shared_authority=blocked`，不创建owner/token/worker/backend或hardware
authorization；它是未来production preflight的输入，不是production binding。

M5.8-a7 已在真实 `RM65ArmBackend._send_joint_target()` 最终调用点加入默认关闭的 typed adapter 注入 seam。普通
motion、qpos-init/home 与 background quick-stop tick 不再要求未来 gate 从 SDK 方法或线程状态猜测语义：内部真实调用点
分别携带 exact seam ID、permit kind/operation、source、call path、target sequence 与实际 qpos hash。无 adapter 时仍直接
执行原 legacy callback；显式 adapter 缺 source/context 时在 SDK 前 fail closed。`RM65PermittedPreSdkDispatchAdapter`
要求 invocation provider 对每次实际调用返回 one-shot permit，并以 exact seam/operation 通过既有 owner gate；无硬件 fixture
已验证 motion/home/quick-stop 分型、拒绝不调用 SDK、四次调用使用四个不同 permit、原返回值及默认路径不变。因此三条
共享 `_send_joint_target()` inventory entry 现在是 `pure_wrapper_validated`，但 production frontend 没有注入 adapter，
provider 也没有绑定真实 worker 的 `ExecutionWorkItem`/epoch 与 safety/lifecycle state owner；它们仍计入
`unmediated_seam_ids`，不能记为 `production_mediated` 或硬件授权。

M5.8-a8 将同样的边界扩展到当前arms-only inventory中的真实 safety/lifecycle SDK call：hold、position-stream
disengage、background-fault stop fallback、startup rollback force-stop、force-position startup、ESTOP和close-before-disconnect
stop均通过独立的default-off `RM65ControlDispatchContext` 注入点。一次方法内的多个设备调用不会共享授权：
`Clear_Force_Data`与`Start_Force_Position_Move`、ESTOP的force-stop与arm-stop、close的force-stop与arm-stop，以及fault
fallback的每次尝试都生成不同call context并要求provider给出新one-shot permit。无adapter时原调用顺序/返回值不变；fake
adapter拒绝时SDK callback不执行。对应七条inventory entry现为`pure_wrapper_validated`。连同a7共10/16条entry达到
pure-wrapper成熟度；logical target admission、worker dequeue、home target admission、rearm state transition、quick-stop
operation admission和disconnect仍为`inventoried_unwrapped`，且全部16条在未达到`production_mediated`前继续属于
`unmediated_seam_ids`。

同时补齐了 M6 所需但此前缺失的 composition 前提：`AuthoritativeControlRuntime` 现在可以显式采用外部同一组
`DispatchOwnerRegistry + current shared token + ExecutionEpochRegistry`，并严格核验 session/domain、owner kind/runtime、
clock domain、authority/execution epoch和active selection group；三者必须全量注入，禁止部分注入。默认 MuJoCo行为仍
由runtime自建事实源。fake handoff fixture已验证 legacy owner → HOLD/cancel/ack → revoke legacy token → grant shared
token后，新 runtime采用的是同一 registry/epoch事实源，而不是悄悄创建第二套 owner。production frontend目前尚未创建
和持有这组 legacy lifecycle事实，因此这只是 M5.8/M6 composition prerequisite，不是real wiring。

第一轮 inventory 的准确边界如下：

| 实际调用层 | 当前事实 | M5.8-a/b 要求 |
|---|---|---|
| `BimanualRig.send_commands()` | 只将 command 分发到各 effector backend；对 RM65 position/impedance 来说通常只是 target admission | 可做 logical admission guard，但不能替代 worker/SDK 前 gate |
| `PositionDeployExecutor` arm/discrete callback | 真实 legacy policy worker 分别在 reference 与 latest-wins discrete worker 调用 callback | dequeue 前检查 M1C epoch；callback 下游仍须在最终 backend seam 检查 permit |
| `RM65ArmBackend.send_command()` direct | 非 background 模式在 `_send_joint_target()` 直接调用 `Movej_CANFD/Force_Position_Move_Joint` | 每次 SDK motion call 前 one-shot motion permit |
| `RM65ArmBackend` background stream | `send_command()` 只写 target；后台 tick 经 local target-sequence检查后反复 `_send_joint_target()` | queue admission、worker dequeue/current-target、每个 SDK tick 三层分开；上层 keyframe permit不得跨 tick复用 |
| `RM65PoseImpedanceBackend` | frontend `send_command()` 只更新 latest target；backend loop 调 controller，controller 内还可能有 CANFD follow worker | outer worker与controller/follow worker均需 epoch lineage；真正 `Movej_CANFD` 前每 tick独立 permit |
| RM65 hold/ESTOP/close | quick-stop可能先写 planner，`Arm_Stop/Arm_Slow_Stop/Stop_Force_Position_Move`才是实际 safety call | `RM65_QUICK_STOP` operation级授权 + 每个实际planner/SDK tick独立one-shot safety permit；operation/permit均不能替代stop ack |
| RM65 start/tool/force-enable | connect/tool-frame更新、`Start_Force_Position_Move`、enable/warm-start属于 lifecycle | lifecycle state-machine permit；读状态不需要 motion permit |
| Inspire hand | `send_command()` 内直接 `set_joint_pos`，I/O失败后可能重连并重试一次 | 每次实际写（含retry）必须有当前 motion permit；telemetry read不进入 motion gate |
| DMTAC gripper | `_move_gripper_locked()`为最终 move；`send_fixed_joint_pos()`可绕普通send开关；`set_torque()`会立即写设备 | `_move_gripper_locked/send_fixed_joint_pos/set_torque`均为MOTION final seam且每次write独立permit；首个arms-only manifest显式not-applicable |
| lifecycle cleanup | close中可能先stop再断连；单纯socket/uninit不是motion，但stop具有安全语义 | stop与disconnect分型，不能用一个宽泛lifecycle bypass覆盖两者 |

由此冻结两个 M5.8-a 不变量：

1. logical command/keyframe 的 permit 不能复用于一个异步 worker 派生出的多个 SDK tick；每个实际 motion SDK call
   需要独立 one-shot permit，并引用其 command/reference lineage；
2. 上层 Rig gate 不是最终防线。存在后台 worker 的 profile 必须同时覆盖 admission、dequeue/current epoch 和
   pre-SDK seam，任一缺失都不能进入 M5.8-b。

permit语义现已完成pure contract/fake gate fixture但尚未实现real mediation：
`SafetyControlOperation(kind=RM65_QUICK_STOP)`只允许触发时冻结的stop
trajectory/last accepted state，并限定duration/tick/allowed calls/terminal/ack；每tick permit引用operation ID、tick sequence、
epoch和command lineage，operation超时、epoch变化或terminal后全部拒绝。禁止reusable safety permit。DMTAC通用
`set_torque()`即使某次降低力矩也仍为MOTION；未来只有新增并验证单调`REDUCE_TORQUE_LIMIT`才可另作safety operation。

#### M5.8-b：legacy-owner real mediation

只有取得明确硬件测试和安全范围后才执行：所有真实 motion call 经过 current legacy token；worker dequeue 和
backend call 前都复查 epoch；revoke 后旧 callback不能尾随发送；记录 gate latency、SDK request、backend state 和
measured-stop 可提供的分层 acknowledgement。安全/lifecycle permit保持独立。

无硬件阶段先用 fake backend/dry-run 验证 gate 开关前后 command 数值、dispatch count/order 不变，revoke/
generation/authority/execution epoch mismatch 在 backend call 前产生 typed rejection，旧 position/discrete
worker 都无法尾随发送。若需验证真实 backend latency/receipt parity，必须另行取得明确硬件范围与安全授权；
未授权不能把 mock 结果标记为 real acceptance。

退出条件：legacy owner 是唯一 motion token holder；安全和 lifecycle 命令各自使用窄 permit；shared path
仍 observe-only。M6 后续只切 token owner，不再同时完成首次 gate 接线和 authority 切换。

##### M5.8-b0：stationary stop-only 授权批次

2026-07-21 用户已明确授权一个更窄的现场批次：双臂静止且现场急停可用时，每侧只允许一次由
`SAFETY_CONTROL` one-shot permit保护的`Arm_Slow_Stop`。本批显式禁止enable、home、rearm、`Movej_CANFD`、
force-position、background streaming和authority切换。

实现使用独立`RM65StopOnlyHardwareAuthorization`与`RM65StopOnlyLegacyBinding`：绑定首个candidate manifest、
exact inventory和唯一legacy owner registry；joint-target adapter为deny-all，control provider仅接受
`rm65.safety.hold.direct_stop + operation=hold + sdk_method=Arm_Slow_Stop`，每侧预算一次且授权过期fail closed。
专用现场入口在调用前做两次fresh-state/error/URDF/静止性检查，输出permit validation、SDK result和调用前后
qpos/latency。它不允许fallback到`Arm_Stop`，也不把结果提升为worker drain、measured-stop ack、完整M5.8-b或M6
evidence。

stop-only实施当时本地全量无硬件suite为2409 passed、6 skipped，远端`dexteleop310`新增stop-only纯测试为6 passed；
M5.8-b1-real-A现场证据收口后本机`dexteleop310`最新全量为2491 passed、6 skipped。远端
`--help`入口可加载。现场命令仍必须由能够观察工作区和急停状态的人员执行；自动化agent不能替代
`--confirm-arms-stationary/--confirm-estop-ready`物理确认。

现场结果为`status=stop_only_completed`：左右两侧的dispatch validation均`accepted`、SDK result均为0，调用时延
分别为10.758 ms与8.582 ms；调用前静止漂移及调用后最大qpos变化均仅为`3.49e-5–5.24e-5 rad`。manifest hash
为`964ea0f5...b46bc4`，inventory hash为`62c603cf...c1de53`，binding仍明确
`motion_authorized=false`。原始payload已保存为
`tests/fixtures/control_core/rm65_m5_8b0_stop_only_field_evidence_2026-07-21.json`并由测试绑定当前manifest/inventory。
首轮CLI只展开了context/validation hash，没有保存完整authorization有效期、exact context字段和permit/validation
receipt ID，因此证据成熟度只标记为`scoped_final_call_field_evidence / partial_v1 lineage`。后续输出格式已修正，
但不为补元数据重复发送stop；本次不能关闭worker drain、measured-stop acknowledgement或完整M5.8-b gate。

##### M5.8-b1：default-off real-worker epoch/drain 接线

当前仅完成上述代码级carrier、三个真实worker中的可选检查点、software drain方法和ordered coordinator；没有在
`run_policy.py`或canonical GUI中启用。production接线的剩余硬前提为：

1. 由一个`LegacyControlDomainExecutionContext`创建并持有owner/authority/execution/group事实源，禁止每个backend自建；
2. reference submission在首次queue admission时绑定旧/新epoch，不能在callback时读取“当前epoch”后回填；
3. quick-stop operation使用取消后的目标epoch和逐tick safety permit，不得继承已失效motion item；pure/default-off已完成，production caller待接；
4. cancellation coordinator先原子推进registry，再按上游到下游drain，并在deadline失败时FAULT；contract已完成，production coordinator待接；
5. accepted/rejected epoch check、participant receipt、aggregate、stop request和后续ack进入formal trace；explicit schema/projection已完成，现场producer/WAL待接；
6. fake/default-off composition通过后，真实worker drain与stop acknowledgement仍需新的现场授权。

###### M5.8-b1-real-A：隔离的 idle-worker drain/stop-only 现场批次

本批使用独立 profile `m5_8b1_real_a_idle_worker_drain_stop_only_v1`，不是
`m6_candidate_policy_arm_qpos_v1` 的替代配置，也不是 canonical deploy/GUI 的新默认值。实现只存在于专用
debug composition 中：真实 RM65 background worker 可以启动，但没有任何 target source；joint-target adapter
始终 deny-all；只有在同一 `ExecutionEpochRegistry` 下完成 position-reference、left sender、right sender 三个
participant 的完整 drain 后，才可为每侧签发一次 `Arm_Slow_Stop` permit。

窄批次的保守设置必须满足以下隔离不变量：

1. authorization、owner token、epoch registry、worker 与 WAL 都由本次进程创建并随进程结束失效；后续真实操作
   必须重新建立自己的 execution profile、authorization 和 composition，禁止复用本批对象；
2. 不修改 `m6_candidate_policy_arm_qpos_v1` 的 manifest/hash、正式 backend 默认值、canonical deploy CLI 参数、
   command cadence、safety threshold 或真实 motion profile；
3. `production_binding_enabled=true` 只表示这个专用 diagnostic composition 确实绑定了真实 worker，不表示
   production deploy 已启用，也不授予 MOTION/LIFECYCLE_CONTROL；
4. 输出写入独立 runtime-diagnostic WAL 与 `evidence.json`，明确
   `capture_commit_eligible=false`；它不能 seal/commit 为公开 v2.4 capture，也不能被提升为 M6 authority evidence；
5. 每次运行都重新读取 fresh state、检查 controller error、URDF 限位与静止漂移；任一 worker 出现target、
   send count非零、drain不完整、trace不完整或stop调用失败均 fail closed，不自动进入后续 motion；
6. 本批只关闭“真实无target worker是否观察新epoch并完成software drain”与“drain后final Slow-Stop是否受同一事实源
   保护”的窄证据，不证明带target streaming、motion permit、backend/measured-stop ack、owner handoff或M6 ready。

当前代码入口为
`teleop_runtime.debug.rm65_real_debug.run_rm65_idle_worker_drain_preflight`，已用真实 worker thread + mock SDK
fixture 验证：worker确实运行、`Movej_CANFD`调用数为0、三participant drain完整、每侧Slow-Stop最多一次、
WAL durable complete且公开capture不可提交。现场执行仍必须由能够观察工作区和急停状态的人员显式确认。

2026-07-21 现场执行结果为`idle_worker_drain_stop_only_completed`。左右background worker在drain前分别运行
103/93 tick，`send_count=0`且`target_active=false`；execution epoch从0推进到1，position-reference、left sender、
right sender三个receipt均`accepted/drained`，aggregate为complete，software-queue acknowledgement为
`all_required_workers_drained`。本次没有target或旧SDK invocation，所以三个receipt的discarded work均为空、
in-flight均为0；这证明真实idle worker可以完成epoch barrier，不证明带target/in-flight worker的drain。

drain后左右Slow-Stop permit分别为独立sequence 1/2，dispatch均accepted、SDK result均为0，时延为
6.477/6.420 ms；调用前静止漂移小于`3.50e-5 rad`，调用后变化小于`1.76e-5 rad`。ControlTrace
`issued_sequence=durable_high_watermark=10`、in-memory/durable均complete，同时
`capture_commit_eligible=false`。现场原始`evidence.json`的SHA-256为
`32e4850e9cd0aafdd748226929fd0079c139125ef314378869fb44b18f3d6b32`；bounded field fixture位于
`tests/fixtures/control_core/rm65_m5_8b1_real_a_idle_worker_drain_field_evidence_2026-07-21.json`。
该结果关闭real-A窄profile，不关闭target-bearing production mediation、SDK/backend/measured-stop ack或M6 gate。

### M6：单策略 authority owner 切换

前提：

- M3 与 M5 shadow parity 均通过；
- M4B/M4C 的 branch-aware per-lineage ControlTrace 在目标 profile 下 durable complete；M4A 的 event-type
  presence 不能满足此条件；
- M5.2/M5.5 已证明拓扑 contract 和 deployable sim，M5.7/M5.8 已证明 shared execution parity 和 legacy-owner
  gate integration；本批只授权一个 action policy instance；
- source/shared/backend 三层 safety 的实际 owner 已逐项确认；
- `SafetyMigrationMatrix` 已把每一条现有 human/policy/position/backend safety 标为 preserved、moved、
  reimplemented 或 intentionally_removed，并有 parity evidence；
- strict `PolicyObservationView` 已是 production actor 唯一输入；
- `authoritative_real_v1` TraceRequirementProfile 的 required lineage 在 dry-run/deployable sim 完整；
- `DispatchOwnerToken/DispatchPermit` 已接到最终 shared backend seam，而不是只存在于配置或上层 frontend；
- legacy mediation 与 shared runtime 使用同一个 `DispatchOwnerRegistry`、`ExecutionEpochRegistry` 和 current-token
  generation；owner切换时禁止shared runtime重新创建或隐式grant第二套token；
- execution cancellation 已以 atomic epoch increment 线性化，旧 reference/command 在 backend call 前失效，
  并在 deadline 内取得 profile 要求的 queue/SDK/backend/measured acknowledgement；
- 60 Hz Arbiter gate 在长期 synthetic burst 测试中没有超频；
- 目标 `ExecutionCapabilityProfile` 明确 configured operating rate，不默认使用 60 Hz ceiling；在另行获得
  硬件测试授权后，decision/IK/end-to-end latency p50/p95/p99 和 deadline miss 满足验收预算；
- dry-run/deployable sim 先通过；
- 真机切换另行获得明确范围和安全授权。

切换原则：

- 只在现有 canonical seam 替换 action owner，不再创建外层 main loop；
- `run_policy.py` 和 GUI frontend 仍负责 composition/lifecycle，不各自实现 Arbiter；
- `ArbitrationRuntime` 输出 selected `RobotAction` 后只进入一条 shared safety/resolver/reference/dispatch 链；
- 旧执行 owner 在迁移窗口保留为显式 rollback mode，不能与新 owner 同时 dispatch；
- authority 切换产生新 epoch，清理旧 selected segment/proposal 的规则进入 trace。
- fault/HOLD 使用独立 execution epoch/cancellation；不能只清 proposal buffer 而让旧 command segment 继续。

切换 fixture 固定为：legacy executing → gate HOLD/TRANSITIONING → increment execution epoch/cancel old queue →
revoke legacy token → 等待 profile-required ack → grant shared-arbitration token → 更新 authority state → 发布第一
个新 segment。任何时刻最多一个 dispatch token 有效；publication 后 gate/epoch 改变时，旧 directive 在实际
backend call 前仍必须被 permit check 拒绝。

退出条件：每个 `control_domain_id` 同一时刻只有一个 dispatch owner；切换前后相同输入的 selected、safe、
command 和 receipt lineage 均可比较；回退不会破坏 v2.3 数据或遗留悬空 epoch。

### M6.5：replay authority 迁移

前提：M5.3 parity、M5.5 sim authoritative replay 和 M6 互斥 owner/cancellation 均通过。

sim replay 必须使用与 policy/human 相同的 proposal→Arbiter→safety→execution 链。真机 replay 则按
source dataset/action stage/target embodiment 单独申请安全授权和 preflight，不由 M6 policy authority 自动
授权；现有 `teleop_runtime.replay` frontend 可以保留 UX、init 和 dataset selection，但最终 dispatch owner
迁到 shared control core，禁止旧 executor 与新 owner 并发发送。

退出条件：sim/获授权真机各自只有一个 shared-arbitration dispatch owner，replay 只作为当前 authority source；
replay 产生新的 capture，并能追溯 source
capture/event、retarget/resampling、decision 和执行 receipt。未获真机授权时，M6.5 以 sim 完成状态结束。

### M7A：native streams

M6 与 M6.5 sim replay 完成后，先独立落地 tactile/state native-rate ingestion、StreamCatalog、window/cut
reader、gap/drop/integrity 和 15 Hz RGBD 多 cut 复用。它不引入 PolicyGraph、learner 或分布式 transport。

这是当前不依赖高频 tactile 的单策略 profile 的默认顺序，不是全局硬依赖。若某个更早的
ExecutionManifest/PolicyObservationContract/TraceRequirementProfile 显式要求 native stream，则只把所需
stream capability 前移为该 profile 的 shadow/authority 前置验收，不要求提前实现完整 M7A。

退出条件：高频 stream 不由 decision row 触发；同一 RGBD payload 只存一次并被稳定 ID 重用；不同
TraceRequirementProfile/TrainingView 能准确判断 stream eligibility。

### M7B：learning/artifact closure

M7B 不再首次定义运行时不可恢复事实。它消费 M4L/M5L 已冻结且通过 profile completeness 的 capture，在 sealed
capture/native reader 稳定后实现 TrainingView、intervention segmentation/reward/annotation/inversion overlay、
BC/DAgger/RL learner fixture、`TrainingRunManifest` 和 additive artifact bundle/v2.4 compatibility extension。
现有 `PolicyArtifactContract` v1 保持兼容，不在原类中强塞 fast-weight/noise adapter 字段。

退出条件：至少一条无硬件 BC/DAgger fixture 和一条 sim RL/rollout fixture 能从 sealed capture 生成
TrainingView，产出引用 capture/overlay/view/code/config/checkpoint digest 的 artifact；artifact 经 offline
reconstruction→deterministic replay/sim→preflight/shadow 并产生反向记录 artifact/training-run hash 的新
capture。生成式 inversion/support assessment、intervention phase/mask 和 safety-projection target eligibility 都是
版本化派生事实；`actor_action` 与 `environment_behavior_action` 必须分别绑定 schema/stage。替换 compatible fake
source/sim embodiment 不改闭环接口，不兼容 binding fail closed。

### M7C：production multi-policy shadow

只实现 production `PolicyNodeRuntime`/PolicyGraph、slow-context routing、partition/restart/stale、确定性 Composer
和 fast/slow shadow。先完成 in-process binding；shared-memory/RPC 等 distributed binding 只做独立 transport
parity，不作为 M7A/M7B 或首个 M7C shadow 的阻塞前提。

这些扩展继续复用不超过 60 Hz 的同一个 `ArbitrationRuntime`；其他 clock 不降频。100 Hz learned producer
可以记录/归并 proposal，但 v1 不承诺 100 Hz selected-action 更新。M7C 不自动取得多策略真机 authority。

### M8：多策略 authority owner 切换（按需）

只有 M7C 的 fast/slow dependency shadow、stale/partition/restart、component authority，M7A/M7B 的
ControlTrace/training eligibility 和 deployable sim 验收完成后，才对指定 control domain 另行申请多策略
authority 切换。M6 的单策略授权
不能外推为 M8 授权；仍要求互斥 dispatch owner、版本化 AuthorityPolicySpec/ExecutionCapabilityProfile 和
独立真机安全验收。

### CS-0–CS-7：Control Station 并行应用层迁移

该支线不替代M5.8/M6，也不把GUI变成新的motion owner。详细对象、文件树、页面和验收矩阵见
[`dexteleop_control_station_plan.md`](dexteleop_control_station_plan.md)。本执行台账只冻结依赖关系：

实现位于仓库根级`control_station/`，作为只入不出的application composition leaf。`robot_motion`、`teleop_runtime`、
`deployments`、`dataset_contracts/interface`、`sim`和`hardware`均不得导入它；现有frontend通过lower-owned native seam由
根级adapter主动组合。

```text
CS-0a pure manifest/intent/state ✅
→ CS-0b null/fake composition ✅
→ CS-1a loopback read-only GUI builder ✅
→ CS-1b authenticated typed intent GUI（null/no-motion）✅
→ CS-1c bounded events / no-I/O operation coordinator ✅
→ CS-2a1 synthetic-human authoritative data plane ✅ scoped
→ CS-2a2a scheduled-qpos formal capture ✅ fixture
→ CS-2a2b Station authority lifecycle/status/trace ✅ scoped
→ CS-2a2b dedicated loopback Web authority API ✅ scoped；无production caller/real target/action payload
→ CS-2b-a qpos-compatible immutable replay emission authority ✅ scoped
→ CS-2b-b exact pose + verified canonical-desk transform + one-emission fanout ✅ scoped
→ CS-2b-c dataset-backed GUI replay composition + formal capture ✅ scoped fixture
→ CS-2c sync policy-only MuJoCo data plane ✅ scoped（只消费既有PolicyActionEmission）
→ CS-2c Station lifecycle ✅ scoped
→ CS-2c policy-specific formal capture ✅ scoped fixture
→ CS-2d RTC virtual policy-only（不拥有predict/plan/merge/pop）
→ CS-3 existing real frontend passive composition
→ CS-4 M5.8受限真机微动操作台
→ CS-5 human/replay/deploy分别真机验收
→ CS-6 authority handoff / deterministic composer
→ CS-7 production entrypoint migration
```

必须保持：

- human adapter只消费现有`TeleopSession` emission，不重复读取device；
- replay adapter只消费现有source-frame emission，不创建第二cursor/resampler；
- qpos-only target不得把legacy pose replay在source adapter中隐式转换；pose proposal必须由pose-capable shared execution
  在safety之后显式解析为qpos reference；
- policy adapter只消费唯一`PolicyActionEmission`，RTC模式也不调用predict/merge/pop；
- null/MuJoCo/real binding互不继承motion authorization；
- Control Station只向唯一`EpisodeLifecycleOwner`请求capture，不创建第二episode；
- CS-1c event ring只是可丢弃UI cache，eviction/gap不能被解释为ControlTrace丢失或capture不完整；
- CS-1c operation capacity必须在intent receiver admission前预留，HTTP线程不得执行adapter，null/fake completion只从独立
  coordinator worker回到runtime result-admission seam；
- CS-2a1只绑定既有MuJoCo authoritative data plane，不得把binding receipt解释成Station已取得motion authority；
- synthetic human仅为无设备Python fixture，浏览器仍不能提交raw qpos；CS-2a2a的episode owner只存在于测试侧
  composition，production lifecycle仍未接；
- CS-4之前浏览器不能提交任意qpos/pose；微动也必须是server-side typed bounded operation；
- T2 human、T3 replay、T4 deploy必须分开验收，通过后才允许T5 handoff和T6混合；
- 在sim parity、legacy-real passive parity和README迁移前不新增canonical production entrypoint。

RTC依赖保持独立：CS-0/1不等待RTC；CS-2可先接RTC-2 virtual runner；RTC-3b split parity与RTC-3b.5
timed-plan唯一admission/transition-first/worker session已local-only关闭。正式异步deploy source下一步仍必须等待RTC-3c
virtual-policy default-off frontend wiring。RTC-3c profile禁用position lookahead，`sync_queue_v1`保持原行为；pure state或
RTC-3a worker存在都不等于production source ready，也不继承M6 motion authorization。

### EG1–EG3：第二本体驱动的泛化批次（RM65 真机验收后）

这些批次不是 M5.8/M6 前提，也不得混入首个 RM65 owner 切换。

**EG1：pure contract/binding**

冻结 `EmbodimentContract`、`EmbodimentBindingAssessment`、`ControlComponentSpec`、
`ActionPayloadSchema`、`EffectorActionCapabilitySpec` 和 `ControlDomainTopologyBinding`。binding verdict 至少
区分 exact canonical compatible、execution-adapter compatible、explicit retarget required、training-view only、
indeterminate 和 incompatible；contract compatibility、validated execution capability、motion authorization
仍是三个独立 gate。

**EG2：RobotAction v1 bridge 与正向 fixture**

只建立一条 lossless `RobotActionV1 → generic component-action projection`，不允许 v1 action 与 generic
envelope 同时成为两份权威动作。无硬件 fixture 覆盖单7DoF、异构6+7DoF coupled、1DoF gripper、q16 hand、
第三 manipulator component，以及同一 canonical fact 到两个 command schema 的映射；不接 production builder。

**EG3：第二本体 production migration**

在第二本体已经选定后，才把 Arbiter 收敛为 numeric-payload opaque/schema visible，把 safety/execution 按
action schema/profile 注册，把 builder 改为 manifest preflight + accepted binding，并增加 v2.4 native
per-component facts。推荐第二本体为单7DoF arm + 1DoF gripper；另一套双6DoF不足以暴露现有拓扑假设。

---

## 五、60 Hz 限制的实现与测试细节

### 1. 配置校验

建议 `ArbitrationRuntimeConfig` 至少包含：

```text
decision_hz
schedule_mode: fixed | event_limited
clock_domain
max_pending_triggers
cutoff_policy / target_lead_time / publication_deadline_offset
decision_runtime_budget
max_target_lateness
late_decision_policy: no_action | hold
missed_tick_policy / trigger_coalescing_policy
```

`max_hz=60` 是 production decision policy v1 的代码级 invariant，不作为可被普通 CLI 提高的参数。若未来有充分证据要提高上限，应发布新的 policy/capability，修改版本化 contract、验收 fixture 和发布说明，而不是通过临时 flag 绕过。

### 2. 调度语义

- fixed mode 使用 monotonic deadline，晚醒时跳到下一个未来 deadline，不补发历史 decision；
- event-limited mode 在 gate 关闭期间累计 trigger reason/最新 cutoff，开 gate 后只构造一个 snapshot；
- producer publish 与 proposal buffer 写入不受 60 Hz 限制；
- production decision policy v1 只保证最高 60 Hz authoritative learned-action 更新；高于 60 Hz producer
  的 proposal 会被 reduction/gate 合并或错过 effective interval，不保证逐条 selected；
- snapshot cutoff 必须取本次 decision 的真实 causal cutoff，不能回填成最早 trigger time；
- 每个 tick 固定 scheduled-decision/cutoff/target-effective/publication-deadline；`decided_at` 与 authoritative
  `published_at` 分开，publication 前 live gate/epoch 改变会拒绝旧 result；
- `no_action/hold/fault` 普通 decision 同样计入 decision rate；独立 safety event 不计为普通 selected decision；
- replay 可以无实时等待地调用 pure Arbiter，但重建出的 logical decision timestamps 必须符合 capture
  contract。

### 3. 审计字段

Execution manifest 和 runtime audit 至少记录：

```text
decision_rate_policy_id/version
configured_decision_hz
max_production_arbiter_hz = 60
schedule_mode
decision_count
coalesced_trigger_count
missed_deadline_count
min/mean/p95 decision interval
observed_max_windowed_decision_hz
clock_domain
decision_runtime_p50/p95/p99
target_lateness_p50/p95/p99
deadline_miss_count
execution_capability_profile_id/hash
```

验收不能只看平均值；必须验证最小相邻发布时间和滑动窗口内 decision count，防止 deadline catch-up 造成瞬时超过 60 Hz。
真实 execution capability 另行审计 coupled-IK runtime 与 proposal available→first command send 的
p50/p95/p99；这些指标不把 Arbiter 上限传播成 reference/IK/command 的全局配置校验。

---

## 六、每批次的通用验证与回滚

每批次至少运行：

```text
git diff --check
targeted pytest
python -m py_compile <本批次新增/修改模块>
相关 entrypoint --help（仅在 CLI 变化时）
```

不因普通单元测试授权运行真机、远端进程或 hardware test。

回滚策略：

- M1A–M1D/M2 是 additive contract/wrapper，回滚只需移除未接入的 caller；
- M3/M4 默认 `off`，shadow fault 不得进入 backend dispatch；
- M5 的 observer hook 可独立关闭，不改变 `TeleopSession` owner；
- M5.3 replay proposal 默认 shadow；M6.5 切换前现有 replay owner 保持唯一，回滚不得形成双 executor；
- M5.7 可整体关闭且没有 backend side effect；M5.8 回滚只能在 HOLD/cancel 后恢复无 gate 的 legacy seam，
  不得留下 shared token 或 safety/lifecycle bypass；
- M6 使用互斥 owner mode，回滚到 legacy owner 前必须终止新 epoch、清空 selected segment 并记录 mode
  transition，不能热切换成双 dispatch；
- v2.4 capture 与 v2.3 row 双写期间任何一方失败都按各自 completeness/commit contract 处理，不覆盖已
  finalize 的 capture。

---

## 七、当前实施边界

M0 历史baseline、M1A、M1A.5、M1B、M1C、M1D、M2、M3、scoped M4A、M4A.1、scoped M4B、M5.2 pure
synthetic contract，以及 M5.3/M5.3a typed replay contract 已在当前本地工作树完成；
M5 已完成 strict adapter/passive session seam/shared-Arbiter/legacy parity fixture，并已把 canonical GUI 的
arm/Inspire 路径接成默认关闭的 non-authoritative frontend；DMTAC 的 session 外独立下发尚未覆盖。
这里 M1C/M1D 是 additive fake-worker/backend contract，M3 是默认关闭且禁止发动作的 deploy shadow，M4A 是
未接 frontend 的 in-memory envelope/journal；M4A.1 只把当前 policy-shadow 子集收紧为 explicit schema/lineage，
scoped M4B 的 WAL/spool/health/sink/worker 已可由 M5 human 与 M5.3b replay GUI 显式 debug 配置接入；
M4B.2a 已增加 per-episode range/seal/commit-candidate 纯契约、subordinate lifecycle 和
`RealTeleopDatasetRecorder` 默认不 attach 的可选组合 seam；M4B.2b 已提供 dataset-local WAL、原子 seal/commit
publisher 和 human source-specific formal runtime fixture。production builder 仍未 attach，因此它还不是公开
formal capture。M5/M5.3b 都没有申请 dispatch token，也不是 production authority。

2026-07-21 并行改动复核后的原五项失败已关闭：M0 marker 已按当前 typed receipt/observer seam 复核，GUI foot-pad
fixture 已消除测试自身的脆弱生命周期假设，active URDF 已成为当前 manifest 的关节位置限位唯一事实源；当时全量
达到2491 passed、6 skipped。随后并行sim actuation profile v2迁移曾造成task config/test/real2sim SHA漂移，
该漂移已在2026-07-21的2605 passed、6 skipped全量回归中关闭；旧现场证据继续按原hash解释。默认pose/camera/timing变化以及OpenPI/DMTAC/UMI/RL支线仍只通过各自exact manifest进入
证据链；committed v2.4的原地compact/recompress继续禁止用于正式capture。

2026-07-21 远端只读 preflight 使用项目规定的 `codex_remote` wrapper 完成：远端项目位于
`~/zc/DexTeleop`，实际环境为 `~/miniconda3/envs/dexteleop310`；左右控制器
`192.168.101.21:8080`、`192.168.1.20:8080` 均可达。远端关键 M5.8 文件与本地 sync-check 通过，pure
inventory/pre-SDK/worker-drain 测试为 16 passed、1 skipped。上述只证明代码、网络与端口前提，不证明 SDK ready、
急停/使能状态、worker drain、stop acknowledgement 或 motion authorization。远端与本地
`teleop_runtime/configs/hardware/rm65_arm_ips/latest.json` 仍含左臂 TEST-NET 地址 `192.0.2.10`；在现场重新确认并
显式保存配置前，任何真机命令必须显式传入两台站点 IP，禁止隐式使用 `latest`。该 fail-safe 配置不应为了测试方便
被静默改成可运动地址。

同一轮审计发现原“只读” RM65 state-viz 会经共享 backend 默认启动流程调用
`Algo_Kin_Set_Singularity_Thresholds`，因此此前不能作为严格 read-only SDK preflight。入口现已显式设置
`configure_singularity_thresholds=false`，同时保持 force-position 与 background streaming 关闭；新增测试固定这三个
条件，并在虚拟模式通过。现场已按显式站点 IP 运行 5 Hz、5 step、无 renderer 的有限步采样；两侧
`Get_Current_Arm_State/Get_Current_Tool_Frame` 均连续返回 0，且 5 轮日志中的 `arm_err/sys_err` 均为 0；逻辑映射为
`192.168.101.21 -> left_arm`、`192.168.1.20 -> right_arm`，两侧均显式加载 `DMTAC_M` TCP profile，endpoint delta
为 0.1400 m/0 deg。观测关节状态稳定；相对 active URDF 限位的最小余量分别为左 J3 0.069 rad、右 J3
0.342 rad，均在限位内，但左 J3 已接近上限，后续 motion preflight 必须继续使用 fresh measured state 与 URDF
限位逐次检查。该证据只关闭 read-only SDK state 与已暴露控制器错误码门槛，不证明关节使能/急停物理链、stop acknowledgement、worker drain、
pre-SDK permit 或 motion authorization。

当前 scoped M4B.2b/M4C.1 已完成后，主线已推进到 scoped M5.5d deterministic multirate slow/fast sim 与
M5.7a pure shared-execution shadow seam；独立 proposed/shadow trace schema、SafetyMigrationMatrix v1 和显式
anchor/profile 的 arm pose/qpos shared keyframe-safety adapter 已完成；M5.7b 又完成 per-group/epoch/generation 的
previous-safe-committed/measured-rebase 纯 lifecycle、显式 target-admission evidence 与 safety input snapshot。
M5.7c1 已接 policy `LEGACY_ALIGNED` numeric 双算；M5.7c2 pure fixture 已证明 counterfactual state 只在 typed
shadow target admission 后演进。M5.7e2-a 已修正 target binding 的时间语义：typed
`POSE_REFERENCE`/`QPOS_REFERENCE` segment admission 先形成 pending commit，只有达到 endpoint target time、
evidence 已因果可用且 segment 仍有效时才可提交；direct command 只保留独立 compatibility profile，且只能在
backend/command acceptance 后提交；`PositionDeployExecutor.submit()` 仍不是 target admission。M5.7d-a 已为
canonical GUI temporary limiter 增加 exact pre-filter anchor/input/output receipt，并在同一 human emission 上接入默认
关闭的 post-filter/pre-guard passive evidence；pure pose fixture 已完成 legacy-aligned numeric parity。它不产生
ControlTrace execution fact，也不会把 filter-stage state 当成 committed anchor。下一步仍需 human position profile；
formal sim capture继续遵守`sim`不得依赖`teleop_runtime`的边界。

M5.7e2-b1–b5目前已完成scoped legacy observer：raw parent/segment/frame、post-resolver success outcome、
executor-local generation terminal和显式planner/resolver/command-construction failure均已接入，bound projection有pure
fixture。它们继续固定为non-authoritative，不推进shared anchor；`final_local_guard`只有typed stage、没有伪造producer，
完整measured/solver identity、same-generation supersession、public ControlTrace schema与production binding仍待补齐。
production binding必须等待shared execution context owner，不能使用Arbiter shadow ID伪造M1C epoch/group generation。

M5.7e2-c1已保留`immediate_qref_sim_v1`兼容行为并冻结
`scheduled_reference_segment_sim_v1 + ReferenceActivationSpec`。scheduled spec按30/60 Hz deterministic fixture绑定
50 ms lead。e2-c2现已在deterministic MuJoCo内接入真正的receiver-side second check、canonical admission、pending anchor
timeline和START延迟激活；该结论不覆盖真实多线程worker。e2-c3也已按实际120 Hz planner/qcmd链路生成scoped immutable
observations，没有新增60 Hz resampler；e2-c0 context被传入仍不等于receiver已enforce，只有c2 validation+admission receipt
才构成scoped sim接纳事实。c4已分离active generation与pending reservation，successor仅在START切代；e2-d explicit
schema、conditional lineage及deterministic-facade/真实MuJoCo formal seal/commit/read fixture也已落地；START前HOLD和
seal-before-commit进程终止已验证branch/recovery语义。M5.5e1随后增加了确定性受限
async completion queue，并验证deadline、cutoff、stale context、producer restart、burst和HOLD竞争；它没有启动真实
线程或实现PolicyGraph。当前主线已到首个arms-only M5.8-a无行为composition收口；production sim application root仍未冻结。
这里的下一实现门槛仍是 safety ownership/migration evidence，不能把当前 pose-last-action、qpos-measured、
human pre-guard previous-command 三种事实伪装成已经统一。在 M5.7 shared execution seam
上明确 typed reference publication 前，
不能从现有 command/receipt 反推 reference，也不能跳到
真机 owner 切换。M4B.2a 已确认并落实：`RealTeleopDatasetRecorder` 是 canonical
GUI 的唯一 dataset episode owner，`ControlTraceRuntime` 只接收 owner 提供的 bind/unbind；episode seal 使用
独立 journal range、durable HWM、dependency refs 与 trace clock domain，单个 episode 不写
`runtime_trace_terminal`；legacy finalize 成功后只生成 v2.4 commit candidate，trace post-finalize 失败不改写
legacy success。production builder 目前不 attach 该 sink，因此既有 GUI 行为不变。

M4B.2b 已进一步冻结 human/policy/replay 三个 source-specific formal profile 的最低事件族；generic
`formal_real_shadow_v1` 继续为 planned，不能用于 commit。seal assessment 与 capture commit 使用
`meta/v24/control_trace/episodes/<episode_uuid>/{seal_assessment,capture_commit}.cjson` 原子发布：先持久化 seal，
legacy finalize 成功后才允许发布 commit，临时文件或缺失 commit 一律视为 uncommitted；M4C 后续只增加聚合
索引，不改变这两个事实。human formal fixture 已验证同一 observer FIFO 上的 release barrier：barrier 前已接收
的 emission/legacy-execution 属于当前 episode，barrier 后的 event 不跨入该 episode；`stop_episode()` 不等待
barrier，legacy writer 仍先 finalize，v2.4 commit 只在后台确认 trace seal 后发布。legacy episode metadata 只写
上述 immutable sidecar refs，不写会永久残留的 `pending` 状态。当前 production builder 仍未 attach，正式激活
前尚需任意 crash 点 recovery、长稳、大 payload 和目标机 latency/power-loss acceptance；policy formal profile与generic/
real replay profile仍为planned。CS-2b-c只关闭`deployable_sim_scheduled_pose_to_qpos_station_v1`窄profile，不能以human或该
sim fixture冒充全部公开formal capture。M5.2已有producer/fixture的context/composition record可继续做completeness/
durability验收，尚无producer的planned event仍不得凭规划猜测并持久化。上述integration验收前不得接正式real shadow capture；
M4C.1 logical reader/ShadowEvaluationView 已在 human committed fixture 与 replay conditional fixture 上完成；
它从 core facts 重建，不冻结 Parquet 物理布局。M5.5a 已基于真实 authoritative sim producer 完成 scoped
M4C.2 `ActualDecisionView`；scheduled deterministic-facade与真实MuJoCo formal episode现均可seal/commit/read，
seal-before-commit终止的公开/审计reader语义也已验证；长稳、power-loss全矩阵、Parquet cache和production sim entrypoint
仍未验收，真机authority仍远未授权。

DMTAC session 外 gripper passive seam 是可并行支线：若 formal profile 声明启用 DMTAC，必须在对应 capture
前补同一次既有动作消费的 passive emission/legacy-execution seam，并证明不改变 send 顺序或频率；若 profile
明确禁用则标记 `not_applicable`。不得用第二次读取输入或第二个 dispatcher 补齐。M5.2 的 context/composition
records 在进入 formal capture 前仍需补 branch-aware completeness、episode membership 和 durable recovery fixture。

M4A/M4A.1 不能恢复 crash 前的 issued-but-not-accepted 内存事实；scoped M4B 能恢复已形成合法 WAL frame 的
accepted record 和检测/丢弃损坏 tail，但尚不能恢复未进入 spool 的 issued fact，也不能 seal capture。M3 debug JSONL 不是 durable
ControlTrace。现有 production reference/command/backend worker 和 final dispatch seam 尚未接入 M1C/M1D
guard，不能提前声称真机已经具备新 cancellation 或 legacy/new owner 硬互斥保护。

M5.8-a pure gate/inventory之外，`AuthoritativeControlRuntime`现可采用调用方提供的同一owner/token/epoch事实源，并有
完整legacy→shared handoff fake fixture；这只消除了M6 composition会创建第二套registry的实现风险。RM65 joint-target
真实函数已有default-off typed adapter注入点，主要RM65 safety/lifecycle SDK调用也已有逐调用typed seam；但canonical
GUI/policy frontend仍没有production legacy lifecycle owner，真实worker也未提供 invocation/drain receipt，logical
admission/dequeue/operation owner尚未绑定。RM65 quick-stop operation owner、disconnect drain条件和DMTAC仍未完成real
mediation。因此M5.8-b继续fail closed且需要独立硬件授权。
