# DexTeleop 实时 Action Chunk 推理（RTC）落地计划

状态：`raw-RTC producer implemented / OpenPI final-chunk connected / R-R4 real observation no-action + drain passed / downstream target and HITL explicitly out of RTC scope / no-hardware-authorization`

最后审计日期：2026-07-25

本文只规划 action-producing policy 的异步 chunk execution 与 flow/diffusion RTC。严格区分：raw-RTC 保持模型
既有 `observation -> final action chunk` 接口，由客户端执行异步 timed-plan；model-RTC 才允许模型消费未来
动作 prefix/mask 并返回 conditioning receipt。本文不创建新的
production entrypoint，不改变当前真机 dispatch owner，不授权任何设备动作，也不把 RTC 变成 M5.8/M6
已有 gate 的隐含前提。

### 2026-07-25 ownership correction

raw-RTC 的正式终点冻结为 `PolicyActionEmission`。RTC 只拥有 request、worker、result、producer-plan
admission、timed mutable plan、due-pop、emission commit、TTL/epoch/session 与 producer lifecycle；它不拥有
proposal、Control Composition、Control Station、canonical safety、IK、reference、target admission、HOLD物理执行、
dispatch或设备receipt。

```text
raw-RTC -> PolicyActionEmission
             |-> ordinary deploy consumer
             |-> optional control_composition contributor adapter
             `-> passive trace/status observer
```

文中历史 `RTC-TX*` 名称只记录曾完成的下游兼容性探索，不再属于 raw-RTC gate。后续实现不得为了接入普通部署或HITL
把target/composition receipt反向加入producer commit-through。

### 本轮模块调整后的规划冻结

本轮已完成的结构调整：

- canonical RTC coordinator、frontend config/step及source binding统一归
  `robot_motion.deploy.rtc_runtime`；`teleop_runtime.deploy.rtc_frontend`只保留import-compatible re-export；
- 正式producer输出统一为`PolicyActionEmissionResult.emission`。`RtcFrontendStep.action_preparation`只是无lookahead、
  无target preparation的兼容视图，完整step不再是下游必需契约；
- `RtcPolicyEmissionSourceBinding`是唯一binding实现，既有Station/step binding名称只保持同一class identity的alias。
  为避免无意义迁移，历史deploy v1 hash domain保持不变，不新增仅用于改名的v2；
- RTC emission把exact epoch-scoped `source_binding_hash`纳入自身semantic hash；同步queue emission该字段保持
  `None`，因此`sync_queue_v1`既有hash、顺序、pop count与lookahead行为不变；
- production seam改为committed-only `RtcPolicyEmissionPublisher`；旧
  `PolicyActionEmissionFanout`仅保留passive/debug兼容。完整
  `RtcFrontendStepFanout`只允许debug/producer observability，不得重新进入Station、Composition或target正式数据面。

据此，下一步只保留以下可落地gate：

1. **R-B0 compatibility/evidence refresh：**固定旧import alias、历史hash domain、sync emission hash及
   default-off/no-action矩阵；不改teleop/replay链路；
2. **R-B1 cross-rearm binding rotation：**当前stable descriptor与epoch binding已分层，producer缓存candidate，
   publisher拥有initial active binding；rearm后旧consumer会正确fail closed，并产生backend-neutral activation-readiness
   receipt。publisher侧显式suspend、单required subscriber typed ACK、原子rebind/resume已实现；具体consumer自动协调、
   timeout/restart与多subscriber ordered rotation仍未实现。
   production fanout前需由lower RTC owner显式发布“旧binding→新epoch binding”的typed transition，并保证新binding
   在该epoch首条emission前可见；Station/consumer不得自己递增epoch、复制config或猜hash；
3. **D-P1 ordinary policy contributor adapter：**若主线需要policy参与多源composition，只在
   `PolicyActionEmission`终点之后适配为普通composition input，保留emission/source lineage；不得把
   `RtcFrontendStep`、worker、timed plan或model字段加入`control_composition`；
4. **D-TX2/D-TX3/D-LA：**typed trace、target receiver、scheduled preview/lookahead继续是下游部署gate，不属于
   RTC producer完成条件；immediate/current-only consumer不被D-LA阻塞；
5. **RTC-4b-B：**model continuation仍是可选独立分支，不阻塞任何final-chunk raw-RTC provider。

## 当前实施状态（2026-07-25）

RTC-1 已按 deployment-only 边界形成第一版 local-only 实现：

- `robot_motion/deploy/inference.py`：版本化 inference mode、delay estimator spec/state、当前兼容性model-space prefix、
  request 与 worker result immutable contract；其中prefix/模型协议归属将在MRTC-D1/R1中渐进拆分，raw-RTC不改义；
- `robot_motion/deploy/chunk_plan.py`：target-time keyframe/plan、single in-flight binding、producer committed-through、
  unique due pop、`replace_mutable_future_v1` merge 与 receipt；
- `robot_motion/deploy/__init__.py`：只增加上述 pure deployment contract 的公开导出；
- `tests/test_policy_chunk_plan.py`：覆盖 qpos、zero/late timing、non-integer cadence、epoch/request/representation
  mismatch、duplicate/out-of-order、worker fault、empty future、execution horizon、committed prefix、immutable hash
  与 fixed-anchor relative codec round-trip。

RTC-2 又增加一个 deployment sim 专用的 deterministic fixture：

- `sim/mujoco/deploy/rtc_policy.py`：从 causal MuJoCo qpos observation 生成绝对双臂 qpos chunk 的虚拟 policy，
  以及 bounded result mailbox、decision-tick result admission、单帧 due pop 和 existing authoritative sim sink组合；
- `tests/test_mujoco_rtc_virtual_policy.py`：覆盖 zero/fixed/under-forecast latency、jitter replay、duplicate、fault、
  HOLD同tick、错误epoch、scheduled-reference generation 1→2 successor、committed endpoint、qcmd/measured和
  cancellation trace；另覆盖d=1..4、over-forecast、result cutoff前后、zero execution suffix、29 Hz cadence和
  missed-tick no-burst；每个decision最多形成一个`ActionProposal`，不调用whole-chunk proposal admission；
- 虚拟runner新增显式terminal lifecycle invalidation：HOLD/FAULT/TRANSITIONING先走已有execution gate/cancel，
  再递增policy producer epoch并清空producer-local plan。已排队的旧completion不从mailbox中伪删除，而是在新epoch
  merge边界以`producer_epoch_mismatch + request_not_in_flight`拒绝；该fixture不提供静默恢复。

RTC-3a 的隔离OS-thread carrier现已归属通用deploy core：

- `robot_motion/deploy/policy_worker.py`：只接收绑定`PolicyInferenceRequest`的immutable
  `PolicyObservationView`，request mailbox固定capacity 1且全程只允许一个active request，completion mailbox有界；
- `teleop_runtime/deploy/policy_worker.py`只保留同一class identity的兼容re-export，不再拥有worker实现；
- submit/poll均为non-blocking；executor异常转换成typed fault `PolicyChunkResult`；stop timeout如果仍有阻塞调用，
  明确返回`executing_call_not_drained`，且`execution_drain_evidence=false`；
- stop对可选`request_cancel()`只做一次隔离式dispatch，typed receipt区分
  `not_supported/pending/requested/declined/failed`；cancel hook本身阻塞也不能越过配置的stop timeout，且
  cancellation accepted绝不冒充worker thread drain；
- `tests/test_deploy_rtc_policy_worker.py`：使用同一个MuJoCo虚拟qpos policy contract验证actor-view detach、single
  in-flight、duplicate、typed fault、blocked stop和old-epoch late completion fail-closed。

RTC-3b 在不切换正式同步caller的前提下增加prediction/admission拆分兼容面：

- `robot_motion/deploy/runtime.py`：新增control-thread inference ID reservation、worker-side
  `predict_chunk()`与control-thread `admit_prediction_result()`；prediction candidate冻结decoded absolute prefix，
  admission才分配chunk ID、更新replace/append queue、lineage、schedule和prediction-rate timestamp；
- 既有`predict_and_queue()`、`ActionChunkBuffer` replace/append、唯一`prepare_next_action()` pop及
  `on_emission_pop_v1`均保留；本批没有把legacy queue误当RTC timed plan，也没有连接worker/frontend；
- 带`begin/commit/rollback_prediction_transaction` hook的debug/replay policy会在split prediction前显式拒绝，继续
  使用原同步transaction路径；
- `tests/test_deploy_rtc_prediction_admission.py`：以虚拟policy覆盖full-prefix selection、replace、部分消费后的append、
  same-chunk lookahead边界、逐项pop顺序及旧/新chunk-lineage和emission semantic hash严格parity。candidate冻结时
  保留canonical encoding中的list/tuple类型标签，避免“数值相同但事实hash变化”。

RTC-3b.5 已在仍无frontend caller的条件下关闭两套admission truth：

- `robot_motion/deploy/async_producer.py`：新增control-thread-owned immutable producer snapshot；完整active request、
  worker session与timed plan只能经pure transition一起建立/settle/invalidate，matching result才可清active request；
- result admission唯一调用`merge_policy_chunk_result()`；old request、duplicate与old worker session result不进入legacy
  deque，也不能误清新的in-flight request；HOLD/FAULT/rearm/reconnect采用transition-first并递增producer epoch；
- 新增版本化RTC-3c virtual-shadow cadence profile，固定`send_actions=false`和RTC-only lookahead disabled；复用已有
  `PositionControlRates`，未修改position executor或sync lookahead；
- `robot_motion/deploy/policy_worker.py`为work item/submission/completion/stop补不可复用`worker_session_id`与
  ACTIVE/RETIRED/ABANDONED_UNDRAINED/DRAINED状态；blocked thread仍不冒充M1C execution drain；
- `PolicyActionPlanPopOutcome`补decision/selected target/oldest skipped target/lateness/hash；latest-valid/no-catchup
  pop语义保持不变；
- `tests/test_deploy_rtc_producer_state.py`新增pure state/race/profile fixture，并有bounded worker completion→唯一timed
  plan admission交叉fixture；scoped evidence见
  [`docs/architecture/evidence/rtc/rtc_3b5_local_evidence_v1.json`](../evidence/rtc/rtc_3b5_local_evidence_v1.json)。

RTC-3c 已把上述边界以显式、默认关闭的profile接入现有deploy application，但仍只完成本地虚拟policy fixture：

- `robot_motion/deploy/rtc_runtime.py`：control-thread coordinator拥有唯一producer snapshot与一个bounded worker；
  每tick按completion admission→最多一个due pop→正式`PolicyActionEmission`→coverage request顺序运行，不持有设备、
  dispatcher或position reference authority，也不写legacy deque；该模块不导入teleop、replay、Control Station、sim、
  hardware或任何model family；
- `teleop_runtime/deploy/rtc_frontend.py`只保留兼容re-export，现有`run_policy`调用方和测试无需同时迁移；
- `robot_motion/deploy/runtime.py`：already-popped timed keyframe先经target-neutral emission seam形成现有
  `PolicyActionEmission`；embedding application在RTC `step()`返回后才可选调用canonical preparation，不产生RTC专用
  第二种正式emission，也不让RTC调用下游safety/target；
- 正式emission成功后才以绑定同一pop receipt/emission identity的独立producer transition推进
  `producer_plan_committed_through`到当前已pop keyframe；future successor无法形成合法due-pop receipt（并保留
  `target_not_popped`防御检查），未emission successor持续保持mutable；
- `teleop_runtime/deploy/run_policy.py`：新增默认`off`的`--rtc-profile`。唯一首批profile强制builtin hold、
  `send_actions=false`、15 Hz observation request、30 Hz decision/action、60 Hz reference、120 Hz command，并仅在RTC
  profile关闭position lookahead；默认`sync_queue_v1`仍走原`predict_and_queue()`与legacy queue；
- HOLD/FAULT先invalidate producer epoch/plan再停止worker；只有观察到旧session `DRAINED`后才允许显式rearm并生成新
  worker session。阻塞worker不会阻塞control step，也不会冒充M1C drain；旧session late completion只能拒绝并计数；
- `tests/test_deploy_rtc_frontend_shadow.py`用virtual chunk/hold policy覆盖240 tick long-run、唯一emission、bounded
  single-in-flight、TTL/underrun、blocked worker、late completion、FAULT与explicit rearm；全程不初始化设备。

RTC-4a 已增加独立于具体模型、transport、sim/real backend的通用协议与本地迁移fixture；RTC-4b-T0又将其接入共享worker：

- `robot_motion/deploy/rtc_protocol.py`：将scheduler与continuation显式分轴；保留
  `sync_queue_v1/async_aligned_chunk_v1/rtc_flow_inpaint_v1`无损映射，并提供immutable capability、request spec、
  inference envelope、conditioning receipt、model response与validation receipt；
- handshake精确绑定artifact、model action space/dim、pose mode、component scope、codec/frame/layout/normalization、
  prediction/execution horizon、model-input hash、prefix mask和continuation implementation/parameter hash；任何不匹配均
  fail closed；
- `deployments/client/rtc.py`：只定义local/remote共用的`RtcModelBackend`、model-space decoder与
  `ModelAgnosticRtcClient`；返回`PolicyChunkResult`事实但不拥有worker、timed plan、merge/pop、safety或dispatch；
- `deployments/client/rtc_executor.py`：把immutable actor observation经显式prepared-input adapter交给上述client，作为
  caller-owned bounded worker中的blocking callable；它不创建thread，不接plan或target；
- `tests/test_rtc_model_agnostic_protocol.py`：两个不同的deterministic local backend通过同一request spec与decoder；
  capability identity可以变化，但request/plan事实及decoded canonical action数值不变；另覆盖v1映射、mask/receipt/hash/
  horizon拒绝、capability drift、deterministic replay与import firewall；
- RTC-4a模块不导入Torch/JAX/websocket、OpenPI、`teleop_runtime`、sim或hardware；fake backend只存在于测试，未把
  测试模型写入production runtime。加入optional typed cancellation转发检查后当前通用协议fixture为`44 passed`。

RTC-4b-A 已完成首个concrete的OpenPI raw-RTC final-chunk adapter；随后增加了显式、default-off的
OpenPI raw-RTC caller，但没有改变默认同步部署，也没有授予真实 motion authority：

- `deployments/openpi/client/rtc.py`复用`PolicyArtifactContract`、`RemoteModelPolicy` metadata validation与
  `V22DeploymentAdapter`，只声明`async_timed_plan + continuation=none`；完整artifact、server metadata、layout、codec、
  frame、normalization、model input及continuous/discrete output都进入hash；
- artifact必须显式声明model ID、checkpoint和camera contract，server必须声明exact action horizon；RTC-4b-A只允许
  artifact一致的absolute action。relative artifact、缺字段、metadata drift、短chunk、NaN、非法token、input hash mismatch
  与clock regression均fail closed；
- continuous与int64 discrete action使用独立矩阵，token不cast为float；当前typed prefix没有discrete通道，因此带token
  continuation继续由通用handshake拒绝；
- adapter现提供`create_worker_executor()`，只组合通用client与OpenPI prepared-input，不创建worker或target authority；现有
  websocket/server payload、legacy `Policy.infer()`和默认部署路径不变；
- `tests/test_openpi_rtc_final_chunk_adapter.py`使用无网络fake transport覆盖continuous/discrete round-trip、2/4/8 horizon、
  0～0.9 s fake latency、transport fault、reset、array detach和deterministic replay。adapter fixture为`28 passed`；加上
  通用协议与RTC-1～3c。新增model-neutral 2/4/8 horizon及blocked-backend fixture为`6 passed`，OpenPI adapter为
  `29 passed`，本轮聚焦交叉矩阵为`107 passed`。

RTC-4b-T3（raw-RTC caller）已完成最小部署接线，但只开放 shadow：

- `deployments/openpi/client/real.py`新增显式 `--rtc-profile raw_rtc_final_chunk_15_30_60_120_v1`、必填
  `--rtc-normalization-hash`，并把 resolved runtime frame contract（或显式 legacy frame hash）传给 OpenPI policy 的
  `create_raw_rtc_adapter()`；normalization/frame identity 不按 checkpoint、config 或 model name 猜测；
- `teleop_runtime/deploy/run_policy.py`只在显式 raw profile 且有 model-package adapter factory 时创建共享
  `build_rtc_model_shadow_frontend()`；profile 强制 15/30/60/120 Hz、lookahead disabled 和 `send_actions=false`，
  默认 `off` 与 RTC-3c builtin-hold 分支不变；无 factory、artifact 或 capability drift 会 fail closed；
- OpenPI policy 复用已创建 transport；adapter wrapper 不加载第二个模型、不创建第二个 server session，不拥有 worker、plan
  或 dispatch；本地 fake adapter→shared frontend 已覆盖一次正式 emission，尚无真实 observation/action-send 证据；
- 新增显式 `--rtc-real-shadow` 只读 guard：仅 raw-RTC 可用，live arm/hand endpoint 未显式开启时 fail-closed；RM65
  跳过奇异点/force/CANFD/tool-frame 写入，阻断 command/stop，阻断 DMTAC 启动配置与 tactile calibration 写入，并用
  直接 `Get_Current_Arm_State` 作为 impedance observation；`--debug` 还会记录统一的
  `rtc_transport_exchange` identity/latency 事实，但它仍是 non-authoritative observability。R-R3/R-R3b/R-R4 已在显式
  `send_actions=false` 边界下取得真实 observation/transport no-action 证据，R-R4 的显式 validity/drain 复跑已闭合
  worker close；这些证据仍不能替代 canonical safety/IK、target acceptance 或 RTC-7 motion authority；
- model-RTC 仍不在本批：任何 prefix/mask/conditioning receipt/identity promotion 继续归 RTC-4b-B 后续 gate。

2026-07-23 受控回归：raw-RTC/OpenPI 四文件为 `82 passed`，仓内 RTC/OpenPI 全集为 `213 passed, 1 skipped`；只包含
fake/local transport、synthetic frontend 和参数/契约检查，不启动真实设备、不读取真实 observation、不发送 action。
边界、source hash 与 non-claims 记录于
[`docs/architecture/evidence/rtc/rtc_4b_t3_openpi_raw_rtc_local_evidence_v1.json`](../evidence/rtc/rtc_4b_t3_openpi_raw_rtc_local_evidence_v1.json)。

RTC-4b-T1 已关闭existing-server transport前的本地session lifecycle契约，但仍未连接server：

- 通用client/executor只增加可选、快速返回的`request_cancel()`转发；没有该hook的local/remote backend继续返回
  unsupported，既有blocking-call与late-completion语义不变；
- `OpenPIWebsocketTransport`增加session cancellation latch、幂等异步socket close、显式`begin_session()` re-arm和
  单调connection/session generation；begin/cancel和close completion分别形成typed receipt；close仍pending或close失败时
  re-arm fail closed，旧连接不被新session复用；
- OpenPI RTC backend在创建新worker executor时才re-arm，并在每次response后复核server metadata hash，reconnect drift
  fail closed；普通OpenPI policy、websocket payload、server和`sync_queue_v1`均未修改；
- cancel receipt只证明请求已接受；只有同一`PolicyWorkerStopReceipt.thread_exited=true`才证明该inference worker已drain，
  仍不是M1C/M1D execution drain；
- 伪policy/transport覆盖无hook、非法hook、hook自身阻塞、blocked infer被cancel解除、fault completion、重复cancel、异步
  close、pending/failed-close re-arm拒绝、同transport新session重连和metadata drift。本批四文件聚焦fixture为`51 passed`，
  无网络、server、设备或
  action send；scoped evidence见
  [`docs/architecture/evidence/rtc/rtc_4b_t1_transport_lifecycle_local_evidence_v1.json`](../evidence/rtc/rtc_4b_t1_transport_lifecycle_local_evidence_v1.json)。

RTC-4b-T2a 又完成不依赖具体模型的client transport deadline与no-model localhost协议loopback：

- `deployments.client.rtc_transport`只定义可选`connect/metadata/inference/close`相对monotonic预算、typed phase receipt和
  deadline exception；字段不包含model ID、checkpoint、config、action dimension或horizon，也不进入model capability；
- OpenPI websocket只做具体映射。未传deadline spec时不新增`open_timeout/close_timeout/recv(timeout)`参数，普通
  `OpenPIClientPolicy`、无限ConnectionRefused重试、payload与server均保持原行为；显式超时才返回typed failure并退役可能
  含late response的socket，不生成fallback chunk；
- bounded phase history区分`completed/timed_out/cancelled/failed`，但不冒充request TTL、worker drain或execution drain；
  reconnect仍必须建立fresh connection generation，OpenPI RTC backend继续复核metadata hash；
- fake websocket覆盖默认调用形状、connect重试到期、metadata/inference timeout、cancel-vs-timeout、close与fresh reconnect；
  实际`websockets==16.0` localhost用synthetic final chunk验证两次连接和metadata continuity。该server不加载模型/GPU，不使用
  真实observation且无action send；
- 该contract可由HTTP、IPC或本地模型runtime另行映射；更换模型只替换backend/input adapter/decoder/transport映射，不改变
  RTC request/result/timed-plan/emission与`sync_queue_v1`边界。scoped evidence见
  [`docs/architecture/evidence/rtc/rtc_4b_t2a_transport_deadline_local_evidence_v1.json`](../evidence/rtc/rtc_4b_t2a_transport_deadline_local_evidence_v1.json)。

RTC-4b-T2b 已在不改变server/model源码的前提下完成existing OpenPI model server synthetic/no-action loopback：

- 一个代表性loaded instance精确匹配registry与artifact：16D state/action、horizon 50、双wrist RGB与固定layout hash；
  该实例只作兼容性证据，通用启用条件不得引用其model/config/checkpoint；
- 两代fresh websocket connection各发送一次zero state + two zero RGB wrist image，均只接收和hash final chunk；两个
  `actions[50,16]`均为float32且finite，`actions_discrete[50,0]`与layout hash一致；
- metadata semantic hash跨cancel/close/re-arm/reconnect保持一致，8个connect/metadata/inference/close phase receipt全部
  completed；首轮含模型编译约9.79 s，第二轮约0.80 s，这些只是真实server transport样本，不是control-loop/p99门槛；
- probe只在远端`/tmp`运行，未导入RTC producer、teleop/replay/position/sim/hardware/Control Station，未读取真实相机、
  未调用planner/safety/executor、未发送action；server结束后精确清理并复核loopback端口关闭；
- T2c已复用现有adapter、bounded worker与唯一timed-plan producer完成caller-free synthetic shadow；cold TTL拒绝、warm
  single admission、control-side non-blocking和完整drain均已闭合；随后T3只增加显式 raw-RTC caller 接线，真实
  observation 与任何 motion authority 仍另行授权。

scoped evidence见
[`docs/architecture/evidence/rtc/rtc_4b_t2b_openpi_live_no_action_evidence_v1.json`](../evidence/rtc/rtc_4b_t2b_openpi_live_no_action_evidence_v1.json)。

RTC-4b-T2c 已按 caller-free synthetic/no-action 边界完成现有 adapter、bounded worker 与唯一 timed-plan producer组合：

- 开始前重新审计远端 OpenPI 当前 checkout。existing websocket handler 仍只调用普通 `Policy.infer(obs)`，final-chunk
  wire payload与T2b一致；新增HTTP metadata和caller-free `infer_rtc()`修复不在本轮`continuation=none`调用链。相关
  policy/Pi0/V22 pure/fake回归为`26 passed, 3 skipped, 2 deselected`；
- 首次组合探针发现 `OpenPIRtcPreparedInput` 为 hash/immutability 递归冻结的嵌套 `mappingproxy` 被直接交给 msgpack。
  修复只位于外部 transport adapter 边界：prepared fact继续保持immutable并使用原semantic hash，
  `_prepared_input_payload()`在调用普通`RemoteModelPolicy`前以`deep_thaw()`生成detached plain containers。没有新增
  OpenPI server字段、模型adapter、checkpoint/config分支，也没有改`sync_queue_v1`；fake transport明确锁定嵌套
  `dexteleop`容器已转为`dict`，本地与远端定向回归均为`32 passed`；
- 修复后的完整RTC聚焦矩阵为`172 passed`，OpenPI/default deploy/position/MuJoCo兼容矩阵为
  `160 passed, 1 skipped`，scoped `py_compile`与空白检查通过。并行主线的teleop/replay/position改动仅作回读，T2c没有
  编辑这些路径；RTC profile继续禁用lookahead，sync queue的lookahead/顺序/pop与emission语义不变；
- 修复后同一探针已越过worker/serialization并到达现有OpenPI `Policy.infer()`。当时远端另一个既有训练进程占用约
  `44 GiB / 49 GiB`显存，server首次JIT还需约`2.0 GiB`而按typed fault返回`RESOURCE_EXHAUSTED`。RTC control-side
  仍以30 Hz运行，fault批次观测到901 ticks、单tick wall最大约`7.4 ms`，result以
  `request_deadline_expired + worker_result_fault`拒绝并进入FAULT；这只验证fault/non-blocking路径，不冒充healthy plan
  admission、late-success或性能证据；
- GPU空闲后使用同一artifact-derived配置重新启动同一server，没有更换checkpoint、硬编码horizon/dimension或放宽TTL。
  第一个cold ready result耗时约`9.85 s`，只以`request_deadline_expired`拒绝且不改plan；第二个fresh request warm result
  耗时约`0.687 s`，从50帧中丢弃6个obsolete rows并将其余44个future keyframes一次性接纳到generation 322的唯一plan；
- 320个30 Hz control ticks中318个覆盖in-flight inference，control duration最大约`13.81 ms`、wall最大约`13.84 ms`、
  inference-active tick最大约`6.97 ms`，无budget exceed或>=50 ms阻塞；formal emission与legacy queue pop均为0，
  `producer_plan_committed_through=null`，符合RTC-3c A的no-lookahead/no-hidden-commit边界；
- worker stop为`stopped/thread_exited=true`，transport phases为
  `connect→metadata→inference→inference→close`且全部completed；close前plan hash唯一，close后plan归零并进入
  `TRANSITIONING`。server进程已精确清理且loopback `8078`关闭。

scoped evidence见
[`docs/architecture/evidence/rtc/rtc_4b_t2c_openpi_live_worker_plan_no_action_evidence_v1.json`](../evidence/rtc/rtc_4b_t2c_openpi_live_worker_plan_no_action_evidence_v1.json)。

RTC-4b-T2d 又关闭不同模型框架重复组装 frontend 的接入摩擦，但仍没有新增production caller：

- `deployments/client/rtc_frontend.py`新增结构化`RtcModelRuntimeAdapter`和
  `build_rtc_model_shadow_frontend()`；模型框架只需提供instance-derived capability、request spec以及返回blocking executor
  的factory，executor内部继续自由组合本地runtime/HTTP/websocket/IPC、input preparer与decoder；
- 无自定义session生命周期的backend可直接用`SimpleRtcModelRuntimeAdapter`传入backend/decoder/request spec/input
  preparer，无需继承或registry；OpenPI这类需要fresh worker session/reconnect的框架仍直接实现同一structural protocol；
- helper在worker创建前完成capability negotiation，将artifact/codec/frame/layout/normalization/prediction/execution horizon
  从request spec精确投影到`RtcPolicyShadowConfig`，并核对deploy runtime与profile action cadence；不再由每个框架手抄七组
  identity/horizon字段；
- helper只返回未启动frontend，仍由application显式拥有`start/step/close`。它不创建observation source、target、
  safety/reference/dispatcher，不注册模型名称，不持有worker/plan以外的新authority；
- generic fake adapter覆盖2/4/8 horizon、blocked backend、identity projection、capability drift和cadence mismatch；现有
  `OpenPIFinalChunkRtcBackend`无需shim即可满足同一structural protocol。两文件聚焦回归`42 passed`，完整RTC聚焦矩阵
  `176 passed`，OpenPI/default deploy/position/MuJoCo兼容矩阵保持`160 passed, 1 skipped`。

T2d scoped evidence见
[`docs/architecture/evidence/rtc/rtc_4b_t2d_framework_composition_local_evidence_v1.json`](../evidence/rtc/rtc_4b_t2d_framework_composition_local_evidence_v1.json)。

RTC-4b-T2e以第二个模型框架验证T2d接入面，采用`/home/zc/HandMotion/vla-scratch`基础VLA作为只读基线：

- 基线保持revision `fa3cb37a6f3ba620bd2c167a97035b5431f31d19`且工作树clean，本批没有在该仓库增加文件、修改server、
  policy、dataset或配置；
- `dataset_contracts/vla_scratch.py`新增纯processed-sample binding，冻结其现有四个输入key、ordered camera、
  state history/dimension、action horizon/dimension和representation ID，供未来dataset adapter与deployment共用；
- `deployments/vla_scratch/client/rtc.py`只接受现有`infer(dict)->dict`transport，通过
  `SimpleRtcModelRuntimeAdapter`组合input preparer/backend/decoder；不导入`vla_scratch`、ZMQ、Torch或任何target owner；
- 第二框架首先暴露出每个final-chunk backend仍需重复capability/request/response envelope的主链摩擦；已回到
  `deployments.client`抽出`FinalChunkRtcBinding/FinalChunkRtcBackend/FinalChunkModelOutput`，统一input-hash、连续/离散
  shape/dtype、timing和response构造。VLA层现在只剩processed-input校验、现有`infer()`调用与action-key提取；
- 8项local fixture覆盖输入顺序/shape、无继承composition、4x7 final chunk和缺key/wrong shape/NaN/integer输出拒绝；
  原始vla-scratch ZMQ client/server另完成localhost no-model loopback，实际server收到`[2,3,8,10]` images和`[1,7]`
  state并返回4x7 chunk；没有加载checkpoint/GPU、真实observation或action send；
- 当前server没有metadata/capability handshake，serve还依赖dataset样本推导dimension并warmup；因此本批只证明外置接入
  足够轻量，不证明现有LIBERO checkpoint与DexTeleop action/frame兼容。真实VLA gate必须先生成artifact-derived binding；
- 3项generic final-chunk伪策略另覆盖continuous+discrete rows、model call前input-hash拒绝和typed optional cancel；
  加入T2e后的完整RTC聚焦矩阵为`187 passed`，OpenPI/default deploy/position/MuJoCo兼容矩阵仍为
  `160 passed, 1 skipped`。

T2e scoped evidence见
[`docs/architecture/evidence/rtc/rtc_4b_t2e_vla_scratch_additive_local_evidence_v1.json`](../evidence/rtc/rtc_4b_t2e_vla_scratch_additive_local_evidence_v1.json)。

RTC-4b-T2f已按“完整双臂、多相机canonical样本”完成artifact/dataset-view preflight，仍保持外部基线零修改：

- 新增纯`dataset_interface.vla_scratch`，从v2.3 `PolicyArtifactContract`派生sample spec和binding identity；camera顺序、
  state/action维度、horizon、pose mode与layout/chunk/frame hash均来自既有contract，不读取checkpoint/config名称分支；
- 完整fixture使用artifact显式声明`front/left_wrist/right_wrist`，构造左右pose7加双侧连续q6末端的26D canonical-desk
  state/action、1帧state history和4步chunk；输入mapping故意逆序，输出仍严格按artifact顺序形成`[3,3,H,W]`；
- 同一`VlaScratchProcessedObservation`继续直接供RTC部署adapter使用，训练视图只增加既有
  `action_chunk.actions`，没有复制模型/transport/scheduler语义；基线自己的`ToDataSample`已在其原虚拟环境接受
  3-camera/1x26-state/4x26-action dtype/shape；
- 当前baseline没有processed discrete channel与action padding loss mask，因此binding拒绝离散layout，sample拒绝
  `action_is_pad=true`；masked/transformed state也必须先有训练/部署共享transform contract，不能在adapter中猜测；
- 两仓虚拟环境当前不是合并环境（基线侧缺DexTeleop SciPy，DexTeleop侧缺基线`jaxtyping`）；这是可选同进程
  dataset plugin 的环境观察项，不是server-client RTC gate。正式部署保持DexTeleop client与vla-scratch server进程隔离，
  只通过既有serialized inference payload互通；不构成修改任一基线依赖的理由；
- 5项新fixture与既有pose/artifact/VLA RTC矩阵合计`64 passed`。未读取真实episode、未加载checkpoint/GPU，未连接真实
  observation、worker/plan、safety/target或action send。

T2f之后的loaded-model synthetic no-action gate不再缺processed sample shape，但仍必须先有与该artifact一致的训练产物、
normalization digest和decoder/codec binding；当前LIBERO checkpoint仍不满足该条件。

T2f scoped evidence见
[`docs/architecture/evidence/rtc/rtc_4b_t2f_vla_scratch_dataset_view_local_evidence_v1.json`](../evidence/rtc/rtc_4b_t2f_vla_scratch_dataset_view_local_evidence_v1.json)。

RTC-4b-T2g又完成了server-client进程边界的完整样本virtual-policy probe：在未修改的vla-scratch环境中，server进程以原生
`ServePolicy`承载一个仅用于验证的确定性virtual `BasePolicy`模型实例，独立进程的原生`ZmqPolicyClient`发送三相机、1×26 state，
收到finite 4×26 final chunk；没有checkpoint/GPU、DexTeleop import、真实observation或action send。该probe的意义是确认
环境依赖隔离与wire contract可以同时成立，不是loaded checkpoint compatibility；其中只有一个模型实例，`ServePolicy`只是
server wrapper，client与DexTeleop RTC adapter都不加载第二个模型。DexTeleop侧的RTC adapter、identity/decoder
和worker contract仍由T2f/192项本地回归覆盖；正式部署由DexTeleop侧选择已安装的ZMQ/HTTP/IPC transport实现，或通过外部
transport relay连接这两个进程，不要求把两仓依赖合并。

T2g scoped evidence见
[`docs/architecture/evidence/rtc/rtc_4b_t2g_vla_scratch_process_boundary_virtual_policy_evidence_v1.json`](../evidence/rtc/rtc_4b_t2g_vla_scratch_process_boundary_virtual_policy_evidence_v1.json)。

RTC-4b-T2h 把第二框架验证中剩余的 final-chunk 接线重复收敛为公共 composition kit：

- 新增 `create_final_chunk_rtc_runtime_adapter()`，由一个 instance-derived `FinalChunkRtcBinding` 唯一派生
  capability/request spec，并组合 input preparation、blocking model call、精确 input hash、output mapping 与 decoder；
  helper 不拥有 worker、timed plan、safety、target、reference 或 dispatch；
- 新增精确 key 的 `MappingFinalChunkOutputAdapter`，允许 server 保留额外 diagnostics，同时分别校验 continuous/discrete
  matrix 和 optional source-output identity；binding 声明离散维度但 mapping 未配置离散 key（或反向不一致）时在构造期拒绝；
- vla-scratch 默认入口已直接使用公共 helper 和原生 `FinalChunkRtcBackend`，保留原命名 backend 作为兼容薄封装；
  外部 `/home/zc/HandMotion/vla-scratch` 仍零修改。OpenPI 因拥有 metadata/session/reconnect 生命周期，继续直接实现
  `RtcModelRuntimeAdapter`，本批不强迫其降级到 sessionless helper；
- 聚焦跨适配器回归为 `87 passed`，覆盖公共 final-chunk、vla-scratch、通用 deploy runtime、OpenPI final-chunk 和
  absolute canonicalizer；没有新增 production caller、依赖或动作权限，也没有修改 teleop/replay、Control Station、
  `sync_queue_v1` 或 position lookahead。

T2h 只提升“不同 final-chunk 模型框架接入 raw-RTC”的通用性，不宣称任意本体、任意 sim target 或 model-RTC 已自动适配。
下一通用化 gate 应收敛 launcher 向 adapter factory 传递的 typed context，替代各 launcher 的大段 opaque kwargs；由于
`teleop_runtime.deploy.run_policy` 与 OpenPI launcher 正在并行主线中变化，该 gate 必须先做接口审计和兼容 characterization，
不能在本批直接改写 production caller。

T2h scoped evidence见
[`docs/architecture/evidence/rtc/rtc_4b_t2h_raw_rtc_final_chunk_composition_local_evidence_v1.json`](../evidence/rtc/rtc_4b_t2h_raw_rtc_final_chunk_composition_local_evidence_v1.json)。

### 下游部署附录：不属于 RTC 完成门槛

2026-07-24 对“OpenPI raw-RTC → canonical pose/hand → MuJoCo scheduled target”建议进行代码级复核后，保留其
raw-RTC 定义、canonical pre-safety 边界、target acceptance 独立性、latency-aware coverage和跨 provider/target
验收方向仍有参考价值，但target intersection是独立下游部署工作，不得再作为RTC阶段或完成度。当前代码事实分为：

```text
S: model/source portability
   final-chunk composition kit
   -> typed launcher adapter-factory context

T: target intersection
   emission proof/catalog preflight
   -> typed ControlTrace
   -> early target admission
   -> scheduled pose+DMTAC target

S -> raw-RTC producer portability
T -> independent downstream deploy integration
```

以下已有主线事实不得重复实现：

- `PolicyDeployRuntime` 是 deployment source/emission seam，不是 MuJoCo/real target；RTC frontend 依赖它不等于依赖
  real target。先保留该 owner，只有第二个非 `PolicyDeployRuntime` producer fixture 证明其接口过宽时才抽 protocol；
- `ActionComponentCatalog`、`TargetRuntimeActionContract`、`ReferenceActivationSpec`、minimum activation margin 和
  `ExecutionTargetAdmissionReceipt` 已存在；RTC 不再定义同名 catalog/target/receipt；
- cadence 与 shadow gate 目前继续由 default-off `PolicyAsyncExecutionProfile` 表达。target authorization 使用既有
  execution manifest/target contract；在首个 sim authority gate 前不先做三套 profile 的大范围重构；
- Control Station 只组合已存在的 RTC source 与 target，不拥有 worker、plan、pop、canonicalizer 或 target safety。

当前下游target接线存在两个必须先闭合的真实阻塞；它们不阻塞raw-RTC producer：

1. typed `PolicyActionEmission` 已以 canonicalization receipt 表达 source representation，但现有 MuJoCo emission
   admission仍要求 legacy receipt；更早的阻塞是 `policy_action_emission` ControlTrace v1 使用精确字段集，typed payload
   会因 canonical action/receipt 字段而 fail closed。typed trace 必须以新增 schema/version 落地，不能改写 v1 或影响
   teleop/replay 已提交记录；
2. 当前 keyframe 在 due-pop 时才形成唯一正式 emission，而 scheduled pose target 要求在生效前提交并满足 activation
   lead。仅增大 `first_target_effective_at` 无效，因为 submission 仍发生在 due 时刻；提前使用 future successor 又违反
   RTC-3c A 的 no-lookahead/no-hidden-reference 决定。因此 scheduled pose+DMTAC 必须等待下游D-LA typed early-admission
   gate，或一个等价、唯一且可 promotion 的正式提前 emission 协议。

据此重新分类后续工作：

1. **RTC-4b-T2i source factory context：** 在 `run_policy`/OpenPI launcher 主线接口稳定后，把 opaque kwargs 收敛为
   typed、向后兼容的 adapter-factory context；不改变 sync caller、server payload 或模型；
2. **下游D-TX1 target compatibility pure gate（已完成）：** `robot_motion/deploy/emission_target.py`只复用既有 target
   contract/catalog/activation spec，评估 action space、clock、component scope、coupled group、pose frame/transform、
   activation lead和 legacy/typed representation proof；assessment 只允许进入后续 target preflight，不调用
   target、trace、IK、安全或 dispatch。assessment现同时绑定exact emission ID/hash；legacy pose frame contract/hash
   必须与独立binding hash成对提供；
3. **下游D-TX3a proposal preparation（已完成）：** accepted assessment可形成绑定emission、
   assessment、target/catalog/activation、representation proof和producer identity的`ActionProposalDraft`；target
   control-domain/embodiment只能来自`TargetRuntimeActionContract`，固定`authority_request=none`。rejected assessment
   不产生draft。真实`ProposalBuffer`伪target已验证首次admission和重复identity拒绝；本层不写ControlTrace、不调用
   Arbiter/safety/IK/target/dispatch，也不接Control Station/HITL；
4. **数据投影D-TX2 typed emission trace：** 新增 typed `policy_action_emission` schema/version及 committed-reader
   characterization；legacy v1 hash/字段/reader保持不变。该项需等待并行 ControlTrace 主线稳定；
5. **下游D-TX3b target receiver admission：** 待shared target seam稳定后，MuJoCo消费TX3a preparation，使用catalog
   而非双臂字面量，并接受legacy representation receipt或typed canonicalization receipt；两者都只证明source
   representation，不等于target safety receipt。不得在此步骤旁路TX2 formal trace要求；
6. **下游D-LA scheduled lookahead：** typed identity、valid-until、canonical safety verdict、target acceptance receipt、
   reference generation/source index与later promotion属于target-side preview协议；不得控制raw-RTC producer commit；
7. **下游D-TX4 OpenPI+MuJoCo shadow：** 先用 `protocol_smoke`、`send_actions=false`，记录 target preflight、seam jump、
   p50/p95/p99 latency、coverage/underrun、target rejection 和 drain；不授权 MuJoCo dispatch；
8. **下游D-TX5 sim authority：** warm-up、startup coverage、latency-derived trigger、zero duplicate/catch-up、HOLD/rearm、
   target safety/IK/reference receipts 和 DRAINED 全部满足后，仅授权 MuJoCo；
9. **下游D-TX6 visual/generalization：** 独立 `visual_alignment` profile；再接一个非 OpenPI final-chunk provider和一个
   非 MuJoCo deterministic target。sim authority 不推导 real authority。

coverage trigger 的初始配置不得继续假设 `2 steps` 可覆盖 OpenPI 延迟；启用 TX4 前必须按同一 transport/profile
实测值满足
`ceil(p95 observation-to-admission / action_period) + guard_steps`，同时分别记录 prediction horizon、request TTL、
admitted coverage、committed prefix 与 mutable future。冷启动结果只用于 warm-up，target保持 HOLD。

本次审计曾尝试验证“typed canonical + immediate MuJoCo pose”，但现有安全契约明确禁止 production typed qpos，且
MuJoCo pose profile明确只允许 scheduled reference；随后 target-only probe又被 typed ControlTrace v1字段检查拒绝。
相关未闭合代码已撤回，没有在 runtime 留下 bypass。该 negative characterization 是上述D-TX2/D-LA排序的依据，
不得表述为 target admission 已完成。

TX1 scoped evidence见
[`docs/architecture/evidence/rtc/rtc_tx1_policy_emission_target_compatibility_local_evidence_v1.json`](../evidence/rtc/rtc_tx1_policy_emission_target_compatibility_local_evidence_v1.json)。
TX3a scoped evidence见
[`docs/architecture/evidence/rtc/rtc_tx3a_policy_emission_target_proposal_preparation_local_evidence_v1.json`](../evidence/rtc/rtc_tx3a_policy_emission_target_proposal_preparation_local_evidence_v1.json)。

RTC ownership与Control Station现明确分离：RTC core只位于`robot_motion/deploy`和`deployments/client`；Control Station
不是RTC scheduler、worker、plan owner或必要caller。既有可选passive consumer属于外部application composition，不进入
RTC-4b-T0/T1/T2a/T2b/T2c/T2d/T2e/T2f/T2g/T2h实现与测试依赖闭包。

选择了第7.2节的窄 harness方案：RTC completion、plan merge与due pop需要同一个plan owner和control-loop顺序，
不把这些职责塞进测试专用`DeterministicAsyncTopologyHarness`。两个fixture都使用显式ManualClock，但RTC runner是
RTC completion唯一事实源，不会同时把同一个policy result交给通用async harness调度。

production默认caller仍是同步路径；只有显式RTC profile进入RTC-3c本地shadow coordinator。它未增加
process/socket/Torch/OpenPI server，未获得真实observation/transport shadow或SDK authority。按实施期新增范围约束，
本批没有修改遥操、采集、replay或position executor链路。当前六组RTC-1/2/3a/3b/3b.5/3c直接fixture为
`60 passed in 2.07s`；deployment/OpenPI/position/MuJoCo兼容回归另为`152 passed, 1 skipped in 3.55s`（RTC-3c
最终精确复跑见本批evidence）。此前shared deployment、RTC、MuJoCo async和authoritative
deploy聚焦回归`103 passed`及sync兼容矩阵`94 passed, 1 skipped`保留为历史台账，不替代本次精确命令。
RTC-3c machine-readable manifest见
[`docs/architecture/evidence/rtc/rtc_3c_local_evidence_v1.json`](../evidence/rtc/rtc_3c_local_evidence_v1.json)。

Control Station的CS-2d现已作为独立application-leaf fixture消费正式
`PolicyActionEmission + RtcPolicyEmissionSourceBinding`，并将同一emission送入既有immediate-qref MuJoCo authority。
完整immutable `RtcFrontendStep`只保留为debug compatibility入口，用于核验lower-owned due-pop与commit后提取同一emission，
不是下游数据面要求。
Station不调用frontend step/predict/merge/pop，且其lifecycle只开关downstream publication；release不改变RTC producer
snapshot、worker session、plan或epoch。RTC专用formal profile已完成WAL/seal/commit/read，但仍无真实observation、
OpenPI transport、production caller或SDK authority，因此不提升RTC-3c本身的real/production成熟度。

CS-3a历史上为现有deploy control loop增加了默认关闭、注入式`PolicyActionEmissionFanout`；当前正式路径已收敛为
`RtcPolicyEmissionPublisher.publish_committed_emission()`，由publisher验证typed commit/binding/TTL/idempotency。
observer full/error不反压lower loop，required control subscriber失败则拒绝publication并保持fail closed。
target-neutral source binding与null-target/shadow-only Station fixture已验证producer/session/epoch/config及emission lineage；
同队列ordered fence保证pre-start backlog不会在start后冒充active observation。Station不接收frontend对象，也没有
predict/merge/pop、producer lifecycle、candidate/authority、dispatcher、episode owner或target。该scope仍不提供OpenPI
transport、真实observation、production startup owner、正式capture或RTC-6 real shadow evidence。
application root从producer candidate激活target-neutral binding，并把bounded queue和binding显式注入passive Station；
Control Station不再import或构造teleop publisher。root bootstrap可预先分配queue并等待binding，但不启动deploy、设备或
producer；未收到binding时Station构造fail closed。
`RtcFrontendStepFanout`继续作为debug-only兼容观察面，不得重新成为Station、Composition或target正式输入。
RTC timed emission现将epoch-scoped source binding hash纳入自身semantic hash；sync queue emission保持`None`，因此不改变
既有同步顺序或hash。producer rearm会产生新binding hash，旧fanout/consumer fail closed；自动binding rotation尚未实现，
不得把单epochpassive fixture表述为跨rearm production lifecycle完成。

本轮实施中主线再次推进：M5.8-b1无硬件范围已完成typed worker、唯一fake composition、quick-stop新epoch、
deadline/FAULT、trace projection及canonical deploy default-off注入seam；M5.8-b1-real-A的独立idle-worker
drain/stop-only现场批次也已取得窄profile证据。RTC实施批次曾记录
`2491 passed, 6 skipped, 3 warnings, 7 subtests passed`，real-A field-evidence相关回归为`27 passed`；此后主线
先前全量台账的6项并行RM65 MuJoCo actuation profile v1→v2漂移已于2026-07-21收口；最新全量为
`2605 passed, 6 skipped, 3 warnings, 7 subtests passed`，M5.5/CS-2不再受该baseline阻塞。
该现场结果只有无target真实worker、三participant epoch drain、诊断WAL和每侧一次Slow-Stop证据，不提供
motion/lifecycle authority，也不证明带target/in-flight work。RTC不调用该入口，也不继承其授权。RTC timed plan继续
定义为producer-local preparation state，不是`ExecutionWorkItem`、`ReferenceSegmentAdmission`、worker drain或
stop acknowledgement。RTC-0对moving production frontend的指标接线本批延后，不与M5.8-b1现场gate合批。
本批期间主架构与control-core landing plan又更新；RTC-3b predict/admit拆分现已完成，Control Station也推进到
CS-1c null-target bounded no-I/O coordinator。二者均未修改真实owner、未继承real-A授权，且GUI/coordinator没有调用
predict、merge或pop；CS-1c UI event ring也不是RTC completion或ControlTrace事实。
RTC-3c local virtual/no-action wiring、RTC-4a generic capability、RTC-4b-A final-chunk adapter、RTC-4b-T1 local
transport lifecycle、RTC-4b-T2a model-neutral deadline/no-model localhost loopback与RTC-4b-T2b existing OpenPI
model server synthetic/no-action loopback与T2c caller-free live worker→timed-plan healthy admission均已满足；下一门槛是
独立授权的T3真实observation no-action shadow，或并行的
可选RTC-4b-B真实continuation model-side prefix→suffix round-trip与RTC-6真实observation/transport shadow。下游D-LA
typed preview/lookahead仍是独立target gate，不阻塞RTC-4b，且RTC profile在
该gate完成前持续禁用lookahead。

## 0. 文档权威与冲突处理

本文是以下主线文档的附属 landing plan：

1. [`dexteleop_architecture_planning.md`](./dexteleop_architecture_planning.md) 负责目标架构、时间/authority/
   safety/trace 不变量；
2. [`dexteleop_control_core_landing_plan.md`](./dexteleop_control_core_landing_plan.md) 负责当前 M0–M8
   控制主线、完成状态与真机 gate；
3. 本文只负责 RTC-0–RTC-7 的局部顺序、contract、文件边界和验收；
4. [`dexteleop_rtc_visual_guide.md`](../dexteleop_rtc_visual_guide.md) 只负责把当前实现、兼容链和阶段gate可视化，
   不新增规范或授权；
5. [`dexteleop_control_station_plan.md`](./dexteleop_control_station_plan.md) 负责统一GUI如何观察和编排RTC
   policy source，但不拥有worker/plan/pop；
6. [`RTC 异步 Action Chunk 研究备忘`](../../legacy/reports/rtc_research_memo_20260721.md)
   不是规范来源。其内容与项目 action/frame/time/authority
   contract 冲突时，以前三份文档和代码事实为准。

如果本文与主架构或 control-core landing plan 冲突，主架构和 control-core landing plan 优先。RTC 的
实施不得通过修改本文来绕过 M5.8/M6 safety、owner、epoch、worker-drain 或硬件授权条件。

## 1. 结论与冻结决定

### 1.1 严格区分 raw-RTC 与 model-RTC

DexTeleop 按两个可独立验收的能力落地：

1. **raw-RTC / `async_aligned_chunk_v1`：** 模型保持既有 `observation -> final action chunk` 接口；deploy侧负责
   non-blocking worker、late crop、timed-plan replacement、epoch/TTL/underrun 与 one-emission。它不要求模型看到未来
   动作，也不声称解决生成分布的跨chunk连续性。
2. **model-RTC / `rtc_flow_inpaint_v1`：** 模型显式消费带identity的future prefix/mask，并返回conditioning receipt；
   flow/diffusion guidance只是候选实现之一。它建立在raw-RTC scheduler之上，但作为独立后续能力验收。

本文允许把第一类准确称为raw-RTC，把第二类称为model-RTC；不得省略限定词把两者混为一个完成状态。model-RTC
capability、prefix codec或guided-inference receipt不阻塞raw-RTC。

### 1.2 保持现有 `on_emission_pop_v1`

RTC 的整 chunk result 先进入 policy producer 内部的 timed plan；未来 keyframe 仍只在唯一一次
`prepare_next_action()` 消费时生成 `PolicyActionEmission`，然后才进入 proposal receiver/Arbiter/shadow。

因此 RTC-0–RTC-6 保持：

```text
action_emission_admission_mode = on_emission_pop_v1
```

内部 plan merge 不是 whole-chunk proposal admission。若未来要在 inference completion 时把整 chunk 提前
admission 到 Arbiter，必须另发 producer schedule/admission mode、overlap/supersession contract、replay parity
和 migration fixture；该能力不在本文范围内。

### 1.3 RTC 不提前引入 PolicyGraph

M2–M6 继续使用一个 `PolicyDeployRuntime` + 窄 frontend wiring。RTC worker 是单 action policy 的执行绑定，
不是 `PolicyOrchestrator`、通用 DAG 或多策略 scheduler。Production multi-policy 仍属于 M7C。

### 1.4 正式 action 主线是 arm pose + typed end-effector target

RTC 的 production-facing action contract 不以 MuJoCo direct-qpos fixture 为中心：

```text
arm:
  absolute pose
  或 v2.3 fixed-observation-anchor relative pose

end effector:
  typed position/qpos target
  例如 gripper 或 Inspire Hand
```

双臂 pose 与两侧 end-effector target 作为同一 coupled keyframe 原子接纳。relative/absolute 只描述 arm pose 的
source representation；end-effector target 不参与 pose re-anchor，也不能因 pose codec 被隐式 cast、插值或重排。

`sim/mujoco/deploy/rtc_policy.py` 和 CS-2d 的 arm qpos 路径只保留为 deterministic scheduler/authority fixture，
用于验证 latency、merge、pop、epoch 和 trace；它不是 production RTC action schema，也不得决定 OpenPI、其他模型
或真机 profile 的形状。低层 arm qpos 只能在 canonical pose 经 safety/IK 后出现，或由单独声明的 debug/test
profile 使用。

未来真机 RTC authority 仍必须发布独立 exact manifest；本文不预先冻结某个 qpos、checkpoint、模型 family 或
end-effector 型号的 profile ID。

### 1.5 默认不做 post-hoc weighted blend

`async_aligned_chunk_v1` 的第一版 merge 固定为：

```text
preserve_executed_and_committed_prefix
+ replace_mutable_future_suffix
```

不在线性 action vector、四元数或多模态 policy output 上做通用 weighted average。轨迹连续性由：

1. 已有 position reference/trajectory profile；
2. model-RTC 的 model-side future-action guidance；
3. 独立、版本化且经过 safety 验收的 trajectory bridge；

分别负责，不能让 Arbiter 或 generic chunk buffer 隐式做 blend。

### 1.6 不新增 production entrypoint

真机最终仍通过：

```bash
python -m teleop_runtime.deploy
```

接入。RTC 在验收前必须是 default-off profile/config。MuJoCo RTC 复用现有 `sim.mujoco.deploy` shared-core
runtime 和测试 fixture，不以本文为由新增第二个正式 sim/real dispatcher。

### 1.7 RTC-3c 仅在 RTC profile 禁用 position lookahead

首轮 RTC-3c 冻结为：

```text
RTC-3c virtual-policy / send_actions=false shadow:
  position_lookahead = disabled
  scope = RTC profile only

sync_queue_v1:
  position_lookahead behavior unchanged
```

RTC-3c producer内部的唯一合法顺序是：

```text
TimedPolicyActionPlan
  -> select/pop exactly one due keyframe
  -> PolicyActionEmission
  -> accepted producer emission commit
  -> return formal emission

optional downstream
  -> source-binding validation
  -> proposal/admission/canonical safety
  -> optional position reference processing
```

本阶段不得把尚未正式 emission 的 future successor 传给 position resampler。开关 RTC profile 不得改变
`sync_queue_v1` 的 lookahead 数值、顺序、pop count、chunk lineage 或 emission hash。

RTC lookahead 不永久取消；它被移到独立下游 `D-LA / pre-motion-lookahead gate`。只有 typed preview identity、
`valid_until`、canonical safety verdict、downstream acceptance receipt、reference generation/source index及
later identity promotion without duplicate admission全部落地并验收后，RTC profile才可重新启用lookahead。
preview acceptance不得反向推进或定义raw-RTC producer commit-through。

### 1.8 RTC-4 core 必须独立于模型、transport、sim/real backend

RTC-4 冻结为“通用协议先行，模型 adapter 后接”，不能把 OpenPI、websocket、JAX/Torch、checkpoint registry或真机
资源写进 RTC core：

```text
robot_motion.deploy.rtc_protocol
  = pure immutable capability/request/response/receipt
  = scheduler_mode × continuation_mode
  = v1 compatibility mapping
  = hash/mask/horizon/artifact/codec validation
  != model invocation / transport / thread / sim / hardware

deployments.client.rtc
  = model-family-neutral adapter protocol + local/remote backend seam
  = response validation + model-space decoder seam
  != OpenPI field names / websocket / device owner

deployments.<model>.client
  = capability metadata mapping + concrete transport/model adapter
```

同一个RTC v2 envelope必须能被deterministic local fake、离线dataset adapter、OpenPI或其他模型实现消费；迁移模型时
只替换backend与model-space codec，不改变producer epoch、timed plan、due-pop、formal emission、safety或dispatch链。
`sim`只依赖`robot_motion`的pure contract；real frontend也不成为协议定义来源。

## 2. 当前链路审计基线

### 2.1 真机正式链路

当前 [`teleop_runtime/deploy/run_policy.py`](../../../teleop_runtime/deploy/run_policy.py) 的 action loop 顺序是：

```text
capture observation
  -> PolicyDeployRuntime.should_predict()
  -> PolicyDeployRuntime.predict_and_queue()  # synchronous/blocking
  -> prepare_next_action()                    # unique queue pop
  -> passive Arbiter/safety shadow
  -> canonical safety/command
  -> direct dispatch OR PositionDeployExecutor.submit()
```

`policy.predict()`、OpenPI websocket send/recv、normalize/decode 和 chunk queue mutation 都发生在
`prepare_next_action()` 之前。一次慢 inference 会延长整个 action loop；position worker 可以继续消费已接纳的
reference，但 frontend 不会在阻塞期间提交新的 sample-rate keyframe。

### 2.2 `PolicyDeployRuntime` 与 `ActionChunkBuffer`

当前 [`robot_motion/deploy/runtime.py`](../../../robot_motion/deploy/runtime.py) 同时拥有：

- policy observation pose preparation 与 strict actor view；
- optional begin/commit/rollback prediction transaction；
- synchronous `policy.predict()`；
- output normalize、v2.3 relative→absolute decode；
- `ActionChunkExecutionPolicy` prefix selection；
- chunk lineage 与 `ActionEmissionScheduleSpec`；
- replace/append/pop/peek、prediction schedule 和 recovery discard。

当前 [`robot_motion/deploy/actions.py`](../../../robot_motion/deploy/actions.py) 的 `ActionChunkBuffer` 是普通 deque：

- `replace()` 清空并写入完整 chunk；
- `append()` 追加完整 chunk；
- `pop()` 消费队首；
- `peek()` 只提供一个 successor。

它没有 target-time keyed mutable plan、committed boundary、request/epoch、late-result admission、TTL、atomic
plan generation 或 merge receipt，不能直接作为 RTC plan buffer。

### 2.3 position execution 的 committed boundary

[`teleop_runtime/position_control/deploy_executor.py`](../../../teleop_runtime/position_control/deploy_executor.py) 当前典型
profile 是 sample 15 Hz、reference 60 Hz、command 120 Hz。启用 chunk lookahead 时，当前 keyframe submit 会同时：

- 验证同 chunk successor；
- 为 successor 分配 source index 和 logical timestamp；
- 将 successor 放入 pose/qpos resampler；
- 将其保存为 staged lookahead，下一次 submit 通过 value match 复用。

因此在现有 `sync_queue_v1` 且 lookahead 启用时，producer/queue 的不可改写边界至少包含：

```text
already_popped_current
+ successfully_staged_same_chunk_lookahead
```

这里定义的 `producer_plan_committed_through` 只表示“policy producer 不得重写”，不等同于 shared
`ReferenceSegmentAdmission`、safe-anchor commit、backend accepted 或 measured execution。Trace 和命名必须保持这几类
receipt 分离。

RTC-3c 不沿用上述 compatibility lookahead。其 producer boundary 只覆盖已经 pop、形成唯一正式emission并完成producer
commit的当前keyframe；它不包含尚未emission的future successor：

```text
RTC-3c producer_plan_committed_through
  = popped/current formally emitted-and-producer-committed keyframe only
  != passive future successor
```

因此新 result 可以替换所有尚未 pop 的 mutable future，不会撞上 hidden staged successor。D-LA只定义下游
preview/reference acceptance与identity promotion，不扩大或重定义raw-RTC producer committed boundary。

### 2.4 OpenPI remote policy

[`deployments/client/remote_policy.py`](../../../deployments/client/remote_policy.py) 将 observation 转成 request，调用
blocking `transport.infer()`，再把最终 dense `[H,D]` response 转成 `UnifiedActionChunk`。

[`deployments/openpi/client/transport.py`](../../../deployments/openpi/client/transport.py) 使用同步 websocket
`send()`/`recv()`。把它放入 worker 可以实现异步 runtime，但当前协议只返回最终 action chunk，不能在客户端对
已经完成的样本补做 paper-style flow guidance。True RTC 必须在模型服务端 denoising loop 中实现。

本仓库 [`deployments/openpi/server.py`](../../../deployments/openpi/server.py) 只是根据 artifact 生成临时 registry 并启动
OpenPI checkout 内的 `scripts/serve_dexteleop_policy.py`。服务端 RTC 改动属于跨仓库/checkout 工作，必须单独版本化、
测试和记录 server capability；不能只修改本仓库客户端便声称完成 RTC。

### 2.5 MuJoCo authoritative 与 async harness

[`sim/mujoco/deploy/runtime.py`](../../../sim/mujoco/deploy/runtime.py) 已能：

- capture policy observation cut；
- 接收 bimanual qpos `RobotAction`；
- 让 chunk prefix 的多个 keyframe 具有连续 target time；
- 经 shared proposal/Arbiter/safety/permit/reference receiver 到 MuJoCo；
- 记录 scheduled reference/qcmd/measured lineage。

但 `submit_policy_action_chunk()` 当前把选中 prefix 的所有 future keyframe 提前 admission，并用
`_next_policy_chunk_target_at` 做 append 型连续排程；它没有推理请求、late completion、旧计划 mutable suffix replacement
和 latency crop。RTC fixture 不应让旧/新整个 chunk 同时提前进入 Arbiter，再依赖 proposal overlap 偶然解决替换。

[`tests/support/mujoco_async_topology.py`](../../../tests/support/mujoco_async_topology.py) 是受限、确定性的 completion harness，
目前覆盖 context/composition/decision/restart/gate，不是 OS/GPU scheduler。RTC 应复用它的 causal completion 模式，
增加 typed policy-chunk-result job 或建立同目录下同性质的窄 harness；二者只能选择一个，不能复制两套 async truth。

### 2.6 非 RTC 链路

- `robot_motion.deploy.DeployRuntime` 和 `deployments.client.DeploymentClientRuntime` 是窄 scaffold，不是 production
  scheduling owner；只要求 contract parity，不在其上建立真实 worker lifecycle。
- replay chunk 是已存在的数据源，不是 model inference。Debug replay 第一版保持 sync transaction 语义；真机
  replay authority 仍属于 M6.5 独立授权。
- GUI collection/human teleop 不因 RTC 改为 policy producer，也不改变 `TeleopSession`/writer owner。

## 3. 术语与时间语义

### 3.1 角色

```text
Observation Cut
  -> PolicyInferenceRequest
  -> Policy Worker / Remote Server
  -> PolicyChunkResult
  -> receiver-side result validation
  -> TimedPolicyActionPlan merge
  -> one due keyframe pop
  -> PolicyActionEmission
  -> Proposal/Arbiter/Safety/Execution
```

### 3.2 必须区分的时间

```text
request_triggered_at
inference_started_at
inference_completed_at       # model/server completion
result_received_at           # local transport receive
result_admitted_at           # local plan merge commit

observation.cutoff_at
keyframe.target_effective_at
keyframe.emitted_at           # unique pop
proposal.available_at         # receiver admission, not producer claim

command.scheduled_at
command.sent_at
command.receipt_at
measurement.sampled_at
```

所有 runtime scheduling 真值使用 local monotonic clock 和显式 clock-domain ID。Wall time 只用于日志/跨进程关联，
不能参与同进程 causal eligibility。非整数频率和 variable `dt` 不得依赖 frame-count modulo。

### 3.3 delay 不是一个字段

至少记录三种量：

```text
d_forecast_steps       # request 时保守预测，供 trigger/model guidance
d_elapsed_steps        # ceil((received - observation_cut) / action_period)
s_consumed_steps       # request cursor 到当前 cursor 的实际消费数
```

merge eligibility 以实际 plan cursor、target time 和 committed receipt 为准，`d_elapsed_steps` 只作为时间估计/审计。
如果二者因 loop jitter、missed tick、pause 或 lookahead staging 不一致，不能用一个四舍五入的 `d` 覆盖实际事实。

### 3.4 trigger

只允许一个有效 in-flight request。触发阈值由测量结果计算，不冻结一个拍脑袋比例：

```text
required_future_coverage
  = delay_budget_steps
  + minimum_execution_or_guidance_horizon
  + control_jitter_guard_steps
```

当 mutable future coverage 下降到该阈值且当前无 in-flight 时，capture 最新 causal observation 并提交请求。若
prediction horizon 无法覆盖当前 delay budget 和必要 fresh suffix，RTC fail closed 或回到启动时明确选择的非 RTC
profile；运行中不得在 control thread 隐式同步 fallback。

### 3.5 result 到达与裁剪

`async_aligned_chunk_v1`：

1. 校验 request ID、session/domain、producer epoch、observation cut、artifact/codec/frame hash；
2. 将结果按原 request observation anchor 解码到 canonical absolute；
3. 删除 target time 已在过去或已越过 committed boundary 的结果前缀；
4. 保留旧 plan 的 executed/committed 部分；
5. 用新结果替换 mutable future suffix；
6. 原子发布新 immutable plan snapshot 和 merge receipt。

`rtc_flow_inpaint_v1` 额外要求：

1. request 时将旧计划 future suffix 编码到模型原生 action space；
2. server 使用 `d_forecast_steps`、execution horizon 和 suffix 做 guided sampling；
3. result 到达后仍按实际 cursor/committed boundary 再裁剪；
4. 如果实际延迟超过 forecast，已经承诺/过期的额外前缀只能丢弃，不能重新锚定或回放。

### 3.6 冻结 cadence lattice，不新增重复的全局 rate contract

RTC 必须同时记录 observation、decision、action target、request、due-pop、position sample/reference/command 与
measurement 的 cadence/clock domain。首个验证 profile 使用现有 rate contract 的组合，而不是再造一个与
`PositionDeployExecutorConfig`/`robot_motion.position_control` 竞争的通用 cadence 类型：

```text
observation sample      15 Hz example profile
policy decision/action  30 Hz
RTC request             coverage-triggered, single in-flight
due pop                 decision cut, at most one keyframe
position reference      60 Hz when position processing is enabled
backend command         120 Hz when command processing is enabled
measurement             native backend/sim timestamp
```

具体数值由 profile/version/hash 绑定；上表不是全局默认。非整数 rate 必须由 target time 处理，不能用 frame modulo。
当前 `pop_due_policy_action()` 已实现 missed tick 时丢弃陈旧 backlog、只选 latest still-valid due keyframe，并已有
29 Hz no-catch-up-burst fixture；下一批只补 selected target、oldest skipped target、lateness 和 skipped count
observability，不重写已经成立的 pop 语义。

### 3.7 区分 last action、generation prefix 与 execution future

以下三类状态不得再共用一个“previous/future action”字段：

```text
last_action_for_observation       # 当前既有 contract：downstream accepted 后更新
continuation_prefix_for_generation # model-space guidance/continuation input
future_plan_for_execution          # canonical absolute timed plan
```

现有 `LastActionContract(stage=previous_safe_action, scope=policy_instance)` 与 frontend 更新时机保持不变；result receive、
plan merge、future staging 都不能提前更新它。`PolicyInferencePrefix` 继续独立绑定 source plan ID/generation/hash。
model-RTC 后续若需要 hard/soft prefix，应把 downstream-accepted committed prefix 与 producer-local mutable overlap 分开，
不从 `last_action` 反推。

## 4. Action representation：absolute 与 relative 两条分支

### 4.1 两条 raw-RTC 分支，共用一个 scheduler

RTC 不为 absolute/relative 分别创建 worker、timed plan、merge 或 emission。分支只存在于 result
canonicalization 之前：

```text
absolute raw-RTC:
  absolute arm pose + typed end-effector target
    -> declared-frame conversion and validation
    -> canonical absolute action

relative raw-RTC:
  fixed-anchor relative arm pose + typed end-effector target
    -> decode with the immutable original request anchor
    -> canonical absolute action

shared:
  PolicyChunkResult
    -> TimedPolicyActionPlan
    -> one due pop
    -> PolicyActionEmission
    -> component safety / IK / reference / target
```

因此“absolute RTC / relative RTC”在本文中默认指 raw-RTC 的 source-representation branch；model-RTC 是与其正交的
future-action conditioning 能力，不能用是否 relative 来代替 raw/model 两级定义。

### 4.2 plan 的唯一真值是 typed canonical absolute action

进入 timed plan 前，每个 chunk row 都必须成为显式 typed canonical action：

```text
arm target:
  absolute pose in declared canonical/runtime execution frame

end-effector target:
  typed position/qpos target with adapter/layout identity
```

Safety、Arbiter、reference 和 backend 不消费带隐式 observation anchor 的 relative action。arm pose 的
absolute/relative source mode 不得改变 end-effector target 的数值、时序或 identity。

当前代码已允许 result/keyframe **二选一**保存：

```text
legacy_normalized_absolute_v1
或
typed_canonical_policy_action_v1
```

typed 分支以 `CanonicalPolicyAction.action: RobotAction` 为唯一动作真值；result 会校验每行 receipt 都绑定同一
observation cut、codec/frame/layout/norm、source pose mode、conversion contract 与 relative anchor identity。merge/pop
保持 typed payload，不生成 26D 副本；同一非空 plan 禁止混合两类 payload。legacy constructor、hash domain、同步
queue 与逐项 emission 均保持原值。

正式emission也已按相同二选一语义扩展：typed keyframe pop后先形成唯一`PolicyActionEmission`，其中pre-safety
`RobotAction`与canonicalization receipt identity保持不变；legacy emission继续使用原hash domain与golden hash。
缺少typed preparer/safety capability时，emission仍成立，但preparation保持no-action并返回显式error，不做legacy
fallback。

### 4.2.1 typed component safety 与preparation

`apply_canonical_keyframe_safety()`现已在原owner内扩展native `end_effector_targets_by_side`，覆盖：

```text
per adapter + value_type capability
continuous absolute bounds + max-step + bounded elapsed-time rate
discrete allowlist + transition policy
explicit measured/previous-safe-committed anchor
atomic arm-pose + end-effector verdict
safe | modified | fault
```

`RuntimeEndEffectorCapability.safety`是runtime session内的唯一配置入口；真实`v22_native`输出缺少该能力时启动即拒绝。
模型artifact只声明channel，不能提供或覆盖runtime bounds。`CanonicalPolicyActionPipeline`提供无backend authority的
typed safety→resolver preparation seam；真正proposal/Arbiter/reference/backend caller仍须由后续独立integration gate
接线。

因果顺序冻结为：

```text
pop -> emission -> proposal/admission -> canonical safety
    -> safe RobotAction -> resolver/reference/backend -> downstream receipt
```

downstream receipt不阻塞current emission；只有下游D-LA future preview/lookahead需要pre-emission acceptance。不得把native
end-effector target改写为hand qpos；RTC lookahead仍保持disabled。

### 4.3 absolute raw-RTC

absolute artifact 必须显式声明 action frame、controlled/tool frame、layout、normalization 和 component mask。adapter
只进行一次版本化 frame/codec conversion：

```text
T_model_absolute[i]
  -> declared model-absolute-frame conversion
  -> T_canonical_absolute[i]
```

即使 conversion 数值上是 identity，也必须形成 receipt 并绑定 artifact、frame/layout/normalization hash、source
chunk hash 与 canonical chunk hash。不得因为某个 checkpoint 当前恰好输出 runtime raw pose 而省略 identity。

### 4.4 relative raw-RTC：fixed request anchor

当前 [`robot_motion/relative_action.py`](../../../robot_motion/relative_action.py) 的定义是：整个 chunk 的每一行共享该
request observation cut 的 actual TCP anchor。双臂 anchor 必须来自同一个原子 observation cut：

```text
T_absolute[side, i] = T_request_anchor[side] @ T_relative[side, i]
```

它不是 chained delta：

```text
T_anchor @ product(delta[0:i])
```

也不是在 result completion、plan merge 或 emission 时按新的 measured TCP 重锚定。模型返回较晚时，只能使用原
request anchor 解码后裁掉过期 row；延迟不能改变同一输出的几何含义。

raw-RTC relative decode 只要求 model-space→canonical absolute 的 decoder，不要求模型 server 接收 RTC 字段，也不要求
absolute→relative prefix encoder。request 必须冻结并校验至少：

```text
observation_cut_id / anchor_timestamp
left/right actual TCP anchor
source + canonical frame identities
controlled/tool-frame identities
relative pose convention / rotation representation
layout / normalization / artifact hashes
```

任一侧 anchor、frame、codec 或 atomic group 不匹配时，整个 coupled keyframe/chunk fail closed。end-effector qpos/
position target 按原 component identity 透传，不参加 SE(3) 变换。

### 4.5 canonicalization receipt 与 admission 边界

absolute 与 relative 两条分支都应在首次进入 `PolicyChunkResult` 前形成同一版本的 canonicalization receipt；建议字段
包括 source pose mode、observation/anchor identity、frame/layout/normalization hash、source chunk hash、canonical
chunk hash 与每个 canonical row hash。absolute branch 使用显式 identity/no-anchor receipt，不能靠字段缺失表示。

worker 可以完成 heavy decode，但 control thread 在 plan admission 时仍须复核 request、epoch、artifact、
representation receipt 与 immutable payload hash。timed plan 中不得保存“以后再找 anchor”的裸 relative ndarray，
也不得同时保存可互相漂移的 relative/absolute 两份执行真值。

### 4.6 model-RTC 的表示扩展是后续独立 gate

model-RTC 需要从 canonical future 生成模型 continuation prefix：

| 模型动作表示 | future prefix 规则 |
|---|---|
| absolute pose | canonical future 按 artifact frame/layout/normalization 编回 model space |
| fixed-anchor relative pose | `T_prefix_rel_new[i] = inverse(T_new_anchor) @ T_old_abs[i]`，再编码回 model space |

只有 pose component 参与 re-anchor；end-effector qpos/position target 和 discrete token 必须使用各自 typed
prefix/mask。hard prefix 应来自 downstream-accepted reference，producer-local mutable overlap 只能是 soft prefix。
这要求 encode/decode round-trip、conditioning receipt 和后续 identity promotion，分别属于model-RTC与下游D-LA
独立验收，不阻塞absolute或relative raw-RTC。

#### 4.6.1 model-RTC 代码组织与首批时间契约

RTC是跨层feature，不新增跨`robot_motion`、`deployments`、`teleop_runtime`、`sim`或`control_station`的顶级
`rtc/` package，也不让`robot_motion`完整承载model-RTC。在线计划因果进入`robot_motion.deploy.rtc`内部子包；
版本化continuation训练/artifact schema进入`dataset_contracts`，离线派生view进入`dataset_interface`；
model-neutral encoding/protocol进入`deployments.client`，具体模型provider进入`deployments/<model>/client`；
learner/sampler/loss最终属于 `policy_learning/`。现有`rtc_protocol.py`、`rtc_runtime.py`等flat module保持兼容并渐进迁移，
不得一次性搬迁、建立新production entrypoint或让下层包反向依赖`deployments`。

`PolicyAsyncExecutionProfile`历史上同时绑定producer cadence与position/send/lookahead application设置，其字段和hash
保持不变。新producer代码只消费target-neutral `RtcProducerProfile`投影；position sample/reference/IK/command rate、
lookahead和send authorization继续由embedding application拥有。

model-RTC首批冻结唯一generation/execution lattice：

```text
generation_start_at
    + forecast_delay_steps * action_period_sec
    = first_target_effective_at

model generates full_chunk[0:H]
client returns executable suffix full_chunk[delay:H]
```

runtime与model adapter不得各自再次应用delay crop。首批prefix source只允许
`producer_local_overlap_v1`；sampler的clamped/weighted数值约束强度不构成downstream target acceptance、reference commit或
motion authority。相关pure contract位于`robot_motion.deploy.rtc.continuation`。

首批pure timed-plan selector已落地：只读取immutable plan snapshot，不修改plan、不插值/重采样、不调用模型；只选择
generation lattice上typed canonical、同cadence且内部连续的old-plan keyframe。prefix exclusive horizon不得超过
`forecast_delay_steps`，避免producer-local overlap侵入本轮可执行suffix。已经pop的前导generation index可以缺失，但后续
encoder必须在versioned prefix mask中显式标为未选中；选中区间内部有洞、identity/clock/lattice/representation不匹配均
fail closed。empty/no-overlap/不足最小步数只产生显式bootstrap outcome，不伪造prefix。下一批只接
跨层data/artifact contract与兼容迁移，不直接在`robot_motion`内接canonical-to-model-space encoder；仍不接
target/reference/safety/dispatch。

#### 4.6.2 model-RTC 跨层契约与迁移计划（2026-07-26 冻结）

model-RTC必须把“同一语义”与“同一owner”分开：训练、在线计划和模型调用共享versioned identity，但各层只实现自己
拥有的转换或生命周期。首阶段不修改原始v2.3/v2.4 parquet、episode、media、collection、replay、HITL或ControlTrace。

| 契约/行为 | canonical owner | 明确不负责 |
|---|---|---|
| producer cadence、generation/execution timeline、old-plan overlap slice、canonical conditioning bundle、bootstrap、epoch/plan identity | `robot_motion.deploy.rtc` | model tensor、dataset sample、sampler、loss、target |
| continuation mode、training source kind、runtime source role、mask/loss语义、支持的delay/prefix范围、codec/layout/frame/normalization binding | `dataset_contracts` pure schema | projection、runtime state、模型调用、持久化owner |
| episode/window→artifact-action training prefix/target/mask view、teacher/recorded source provenance | `dataset_interface` | normalization、framework tensor、learner、runtime plan、episode writer |
| injected codec编排、capability negotiation、model envelope、encoding/consumption receipt validation | `deployments.client` | model-specific数值codec、plan advancement、worker ownership、target/safety/dispatch |
| OpenPI/其它模型的artifact-bound双向codec、normalization、输入字段、loaded-instance capability、continuation sampler和transport | `deployments/<model>/client`或对应模型工程 | 通用RTC调度、dataset schema、robot execution |
| optimizer、loss、noise/corruption schedule、训练loop | `policy_learning/`或仓库外模型工程 | deployment runtime、motion authority |

首个pure data contract暂定`ModelContinuationDataContractV1`，至少冻结：

```text
schema/version
continuation mode + conditioning representation
prediction horizon
supported forecast-delay/prefix range
prefix-mask semantics
target-valid/pad mask semantics
loss-mask semantics
training prefix source kinds
runtime prefix source roles
component/action pose/discrete support
action codec + layout + frame + normalization hashes
training projection implementation/parameters hash
```

该contract不得包含runtime request、producer epoch、plan id、target receipt或motion authority。它通过optional
artifact field及hash绑定checkpoint/server；旧artifact缺失该字段时只能声明`continuation=none`，解析和raw-RTC hash/
行为保持兼容。

首个派生训练view至少区分：

```text
trajectory_overlap_teacher_v1
  从同一示教轨迹构造理想prefix，只证明训练projection

recorded_policy_output_v1
  来自有exact model-output/codec lineage的历史policy chunk

synthetic_corruption_v1
  可选、由明确参数hash约束的训练扰动
```

普通future action target不得自动标记为`recorded_policy_output_v1`。训练sample分别携带conditioning prefix/mask、
full target chunk、target-valid/pad mask和loss mask；conditioned位置是否参与loss由contract声明，禁止用一个`mask`
同时表达padding、conditioning和loss。动态runtime latency不能被单一checkpoint白名单固定；artifact声明支持范围，
训练projection记录实际采样delay，runtime request选择值必须落在该范围内。上述三项是training provenance；
在线首批只接受独立的runtime source role `producer_local_overlap_v1`。artifact可分别声明训练覆盖与runtime支持，
但不得用training source identity伪造plan/epoch lineage。

数值转换只允许一条语义真值链：

```text
dataset_interface
  canonical dataset rows -> artifact-action training view
                         （不做model normalization/framework tensor）

model/provider supplied bidirectional codec
  artifact-action/canonical action <-> exact model space
  bound by codec/layout/frame/normalization/implementation hashes

deployments.client
  只注入调用该codec并验证hash/round-trip/receipt，不猜转换
```

训练工程与deployment provider应复用同一codec实现；若跨仓无法共享代码，必须通过同一golden fixture、contract hash及
双向round-trip conformance证明数值等价。不得让`dataset_interface`导入`deployments`，也不得让`deployments`为了
复用训练pipeline而取得dataset reader/window ownership。

现有实现的渐进归位如下：

| 当前符号 | 目标 |
|---|---|
| `RtcProducerProfile`、`RtcContinuationTimeline`、`RtcPlanContinuationSlice/Outcome` | 保留在`robot_motion.deploy.rtc` |
| `PolicyInferencePrefix`中的model-space ndarray | 旧路径仅保留兼容导出/构造语义；新canonical conditioning bundle由`robot_motion.deploy.rtc`组合已有typed timeline/slice/context/data-contract identity |
| `RtcContinuationEncodingReceipt` | 旧receipt保持兼容；新的model-space encoding/consumption receipt及校验进入`deployments.client` |
| `RtcContinuationCapability`、`RtcProtocolRequestSpec`、`RtcInferenceEnvelopeV2`、`RtcConditioningReceiptV2` | 新canonical protocol owner迁入`deployments.client`；旧`robot_motion.deploy.rtc_protocol`按characterization保持兼容，禁止bulk move |
| `PolicyInferenceRequest` | 保留scheduler/causal request；model payload逐步改为versioned pure binding，不在`robot_motion`实现codec |

依赖方向禁止`robot_motion`导入`deployments`，因此旧v2 protocol不能通过反向import成为新实现shim。MRTC-R1/C1采用：

```text
robot_motion
  frozen v1/v2 compatibility request/protocol
  + additive canonical conditioning bundle
  + causal binding（request hash domain v3；只含bundle hash，不含model ndarray）

deployments.client
  model-RTC envelope v3
  = causal request + encoded model prefix/mask + artifact/capability binding

upper-layer parity adapter
  只做旧fixture/兼容映射，不成为第二个producer或plan owner
```

旧class/hash domain保持冻结；新v3使用新type和hash domain，不能让两个同名实现并行演化。raw-RTC继续使用现有
`PolicyInferenceRequest`/`continuation=none`路径。model-RTC切到v3后，`robot_motion`只验证binding hash与result
回显，不解析model-space内容。

迁移批次与exit gate：

1. **MRTC-ORG0（本文档，完成）：**冻结上述owner、依赖方向和“不改raw dataset/控制链”范围。
2. **MRTC-D1 pure data/artifact contract（完成）：**已新增
   `ModelContinuationDataContract`，冻结continuation mode、delay/prefix range、continuous prefix、training/runtime
   source role、prefix/padding/loss mask语义及artifact identity；`PolicyArtifactContract`采用optional/default-none
   binding。旧OpenPI raw-RTC artifact hash有固定golden，当前D1定向兼容回归157项通过；未增加raw dataset字段、
   runtime/model import或控制权。
   **MRTC-D1b（完成）：**冻结v1序列化/hash不变，将其只在内存中规范化为`legacy_v1_compat`；新增v2
   `inference_only_projection_v1`（`training=null`）与`trained_prefix_conditioning_v1`（training required）。
   artifact/wire始终绑定source contract hash，legacy origin不能写入新v2。D2对inference-only fail closed，generic
   conditioning和OpenPI同时接受v1/v2 source contract；相关定向回归96项通过。
   **MRTC-B1/B2（完成）：**OpenPI conditioned backend只从base artifact读取source contract并拒绝missing/mismatch；
   capability由artifact runtime、loaded codec与server loaded-instance metadata交集形成。模型声明versioned sampler
   算法描述，`Policy`绑定实际sampling参数，server绑定external/internal dims与horizon并生成provider capability hash；
   call-site receipt独立回传该hash、implementation identity和sampler constraint。客户端envelope/context回显不再充当
   provider assertion。missing/tampered/drift负例、跨仓portable golden及default-off legacy旁路均已覆盖。
3. **MRTC-D2 offline projection（完成）：**已新增确定性
   `ModelContinuationTrainingSample`及episode-local projection；首批只支持continuous absolute
   `trajectory_overlap_teacher_v1`，显式接收delay/prefix而不在data layer引入sampler。prefix、target-padding、loss
   mask独立；episode tail必须保留真实executable suffix；relative、discrete、NaN/Inf、错误binding及无typed lineage的
   recorded-policy/synthetic source均fail closed。D1+D2定向测试24项、RTC/artifact兼容回归168项、dataset-interface
   相关回归121项通过；未修改raw dataset、writer、normalization、framework tensor或runtime。
4. **MRTC-R1 runtime decoupling（R1a完成，R1b随C1）：**现有timeline/selector及旧
   `PolicyInferencePrefix`/v2 envelope保持冻结；`PolicyInferenceRequest`已增加optional
   `conditioning_binding_hash`，使新model-RTC causal request无需携带model-space ndarray。新binding与旧prefix互斥，
   raw-RTC禁止携带，并使用独立v3 request hash domain；raw request、旧prefix、旧flow request三个固定golden未变，
   runtime/protocol/plan回归78项通过。canonical bundle、encoder及v3 envelope归后续C1；其中bundle由
   `robot_motion.deploy.rtc`持有，model-space encoder/envelope归`deployments.client`。D1/D2/R1a合并后的RTC相关回归
   321项通过（loopback测试需绕过环境HTTP代理）。
5. **MRTC-C1 generic online conditioning（C1a-f本地闭环已实现）：**
   `robot_motion.deploy.rtc`已组合immutable canonical bundle并随work item传递；只有bound work item走typed executor，
   raw/legacy executor与hash domain保持原样。`deployments.client`已有injected bidirectional codec、full-horizon
   prefix/mask、v3 capability/envelope、encoding/consumption receipt、full-output验证后exact-once delay crop，以及
   `PolicyChunkResult`/plan-merge binding校验。external/internal transform ownership、portable array hash、provider-transform
   fake、tamper rejection与frontend explicit bootstrap均已覆盖，不启动target。P0 hardening进一步将该crop命名为
   `generation_prefix_crop`并与plan-owned `admission_obsolete_filter`分离，显式投影generation/conditioning/selected/
   executable-candidate四种horizon；`clamped_v1`在client crop前逐row精确回验，conditioned executor在frontend构造期
   检查`execute_work_item()`。sampler receipt/plan admission均不得推进execution commitment，mask仍由selected indices
   与schedule唯一表达，不增加重复schema字段。P0 protocol cleanup进一步冻结旧v2 model-space prefix与pre-C1
   `RtcContinuationEncodingReceipt`：raw/final-chunk的canonical v2用法仅为`continuation=none`，新conditioned接入只走
   causal binding与deployments-owned v3；C1 bundle/runtime在构造期拒绝非`flow_inpaint`contract，work-item executor
   使用包含legacy callable面的专用typed protocol。结构收敛后，v3 evidence bundle只由envelope持有capability与
   encoded conditioning，bundle访问器不再保存第二份可漂移对象；OpenPI runtime config也不再复制artifact-bound
   continuation contract。通用纯校验函数统一artifact/runtime、codec、sampler、bootstrap与selected schedule绑定，
   provider边界仍保持fail closed。
6. **MRTC-P1 provider接入（OpenPI additive接线与B1/B2证据闭环完成，真实模型sampler shadow待执行）：**OpenPI provider按loaded instance的metadata
   声明external/internal dims与bootstrap，DexTeleop `rtc_conditioned.py`复用既有transport、prepared observation和typed
   decoder；server内继续执行normalize/pad，并在真实sampler调用后产生双prefix receipt。普通payload仍走旧`infer()`，
   `serve_policy.py`与实际DexTeleop入口`serve_dexteleop_policy.py`均由
   `--enable-rtc-continuation`显式开启新路径，不使用model/checkpoint/config白名单。远端provider/wire单元测试
   15项及Pi0纯projection测试2项通过；本地跨层契约123项、扩大RTC/OpenPI/VLA迁移回归290项通过。已只读加载真实
   checkpoint并导出loaded-instance capability，但尚未调用其conditioned sampler，也未完成真实observation shadow。
7. **MRTC-S1 shadow（S1a artifact binding、S1b0本地coordinator、S1b1 direct GPU sampler及S1b2 wire/frontend shadow完成）：**
   `deployments.openpi.client.rtc_artifact_binding`已能将原artifact、真实loaded-instance provider capability与显式
   delay/prefix范围收敛为v2 `inference_only_projection_v1` artifact及可重放receipt；不修改source artifact，也不使用
   model/checkpoint/config白名单。`0712-wrist-cube@92599`只读加载得到的capability已形成仓内派生artifact与receipt，
   source/bound artifact、provider、codec/layout/frame/normalization和continuation hash均可离线重放。
   `RtcConditionedModelRuntimeAdapterV3`及`build_rtc_conditioned_model_shadow_frontend_v3()`现已将v3 backend直接接入
   既有bounded worker/timed plan；不经过raw/v2 capability降格。fake provider闭环已验证worker线程不阻塞control
   loop、首轮bootstrap result进入唯一plan并正式emission、后续request从同一plan取得canonical overlap prefix，
   同时保持`send_actions=false`、lookahead disabled和production queue pop为0。聚焦兼容矩阵120项通过。
   `0712-wrist-cube@92599`真实JAX policy又以合成双腕图像/16D state直接完成一次bootstrap和一次三帧hard-prefix
   `infer_rtc()`：两次输出均为finite `50x16`；第二次receipt确认消费generation indices `[0,1,2]`，最终external
   output对应三帧与输入prefix最大绝对误差为0。该证据只证明loaded sampler数值路径，不包含server-level provider
   assertions、DexTeleop envelope、timed-plan admission或control-loop时序。
   S1b2已通过localhost OpenPI wire把同一loaded provider接入v3 frontend：首轮bootstrap request的forecast delay为2，
   第二轮动态升至16并从唯一旧计划选择generation indices `[0..14]`；两轮server consumption receipt、v3 response和
   plan admission均通过，分别接纳27帧。30 Hz control budget为33.33 ms，观测到的最大step为13.88 ms；27次正式
   emission、legacy queue pop为0、无late rejection，且`send_actions=false`/lookahead disabled。该实跑同时暴露并关闭
   三个通用缺口：worker校验先normalize v2 continuation wrapper、conditioned re-request coverage默认绑定contract
   maximum delay、已有plan时generation start锚定下一枚尚未pop的keyframe而非任意control-tick相位。
   下一步才进入真实observation input-only shadow，仍不接target或讨论motion效果。scoped evidence见
   `docs/architecture/evidence/rtc/mrtc_s1b1_openpi_gpu_conditioned_sampler_synthetic_evidence_v1.json`及
   `docs/architecture/evidence/rtc/mrtc_s1b2_openpi_gpu_wire_frontend_shadow_evidence_v1.json`。
   **MRTC-H1 debug/lease hardening（完成）：**复用既有`--debug + --rtc-debug-level`和bounded observer，不新增
   RTC debug状态机。默认不注入observer；已有应用gated callback在debug off时也必须在payload构造前返回，因此不计算
   masked-output误差、不累计额外计数或保存bundle；
   显式debug才从已验证的envelope/response/receipt惰性投影bootstrap/conditioned mode、
   generation/conditioning/selected/encoded/consumed/executable计数、source-plan binding、latency及masked-output误差。
   raw/model-RTC共用一个debug payload dispatch入口，所有记录均为`debug_only`、`non_authoritative`且无action-send
   authority。lease测试覆盖in-flight期间cursor推进后的obsolete filter、epoch rotation late-result拒绝，以及debug
   开关不改变conditioned result hash。
   **MRTC-T0 trained preflight semantics（完成）：**首个trained profile明确使用v2
   `trained_prefix_conditioning_v1` provenance，但runtime继续声明`flow_inpaint`并复用现有v3
   envelope/receipt/crop；`trained_prefix_conditioning` runtime mode继续保留为未来不同sampler语义的独立gate。
   pure validator同时冻结continuous absolute、teacher-overlap和显式training contract，拒绝legacy/inference-only提升。
   `dataset_interface`新增framework-neutral training window：observation与generation start绑定同一episode/index，
   first executable index严格为`start + delay`；delay/prefix从artifact范围内的显式pair集合，以
   spec/contract/episode/start/draw identity做stateless hash selection，tail只保留存在真实executable suffix的pair。
   sampling spec、causal binding及合并后的training window均有独立hash，chosen pair继续由既有sample hash绑定，
   不新增重复selection receipt；既有D2 sample/hash与v2 artifact
   serialization不变。本批不实现normalization、framework tensor、model/loss、trained artifact writer、provider改动或
   真实训练。
   **MRTC-T1 OpenPI additive training/deploy adapter（已接线，GPU smoke待空闲）：**新增独立training source/run/
   checkpoint evidence、v2.2 absolute compatibility artifact template、stateless sampling transform和可选
   `model_rtc.action_loss_mask`。OpenPI普通二元batch/mean loss不变；只有显式mask才走suffix reduction。外部launcher
   包装任意通过artifact/horizon/layout/camera/dataset-root gate的base config，base checkpoint只读，输出目录独立，
   不注册第二个默认config。远端真实`0712-wrist-cube + 92599 + 503 episodes + wrist pair` no-train dry-run已通过并形成
   stable training-run manifest；OpenPI mask seam 4项CPU测试通过。部署CLI新增显式
   `model_rtc_flow_inpaint_shadow_15_30_60_120_v1`，复用现有conditioned v3 backend与同一个producer，仍强制
   `send_actions=false`、lookahead off；trained checkpoint在连接前验证artifact/run/receipt。当前远端另一个训练进程
   占用GPU，未启动第二进程，首个10-step GPU smoke及其sidecar验证仍待资源空闲。

现有OpenPI checkpoint是普通full-chunk flow训练、RTC为推理期hard inpainting，因此其新binding必须使用
`inference_only_projection_v1`且`training=null`。只有未来有真实prefix-conditioned训练证据的artifact才可使用
`trained_prefix_conditioning_v1`；v1 `legacy_v1_compat`不自动证明任一来源。

RTC-4b-B provider工作在MRTC-D1/D2/R1/C1通过前暂停扩大。任何批次均不得：

- 修改teleop/replay/collection/HITL/control_composition或创建第二个source/plan/worker owner；
- 将teacher prefix称为真实old plan，或从decoded action反推缺失的model-output provenance；
- 让dataset projection、debug observer或model receipt取得target/reference/safety/dispatch authority；
- 以model-RTC需求反向改变raw-RTC `continuation=none`或`sync_queue_v1`默认行为；
- 把本地 artifact 根 `model/` 创建成 Python runtime entrypoint。

### 4.7 表示 gate

absolute 与 relative raw-RTC 共用以下 gate：

- artifact-derived component/layout/frame/normalization/horizon/cadence identity；
- canonical action 与 canonicalization receipt 可重放；
- wrong/stale anchor、frame、layout、normalization、NaN/Inf fail closed；
- quaternion sign/canonicalization 与 SO(3) error tolerance；
- 双臂同 observation cut、同 request、atomic admission；
- arm qpos production profile默认拒绝；direct-qpos只允许显式 test/debug capability；
- RTC profile开关不改变 `sync_queue_v1` 的 decode、顺序、pop count 或 emission hash。

relative raw-RTC 额外要求 original-request-anchor negative control：如果错误地改用 completion/emission 时 anchor，测试必须
产生超过容差的差异并拒绝。model-RTC relative prefix 再额外要求 absolute→relative→absolute round-trip；不能反向把该
要求误设为 raw-RTC 的模型端前置条件。

## 5. 目标 pure contract

建议在 `robot_motion.deploy` 增加纯 Python/NumPy、无 Torch、无 websocket、无 OS thread 的 contract。正式文件名在
RTC-1 开始前冻结；暂定职责如下。

### 5.1 `PolicyInferenceRequest`

必需字段：

```text
request_id / inference_id
runtime_session_id / control_domain_id
producer_id / producer_epoch
policy_node_id / artifact_id+hash
observation_cut_id / cutoff_at / step_index
request_triggered_at / request_admitted_at
action_clock_domain_id / action_period_sec
prediction_horizon / minimum_execution_horizon
plan_generation / plan_snapshot_hash / request_cursor
producer_plan_committed_through
d_forecast_steps / delay_estimator_spec_id+hash
previous_suffix_descriptor
action_codec/frame/layout/normalization hashes
valid_until / deadline_at
```

Observation payload本身仍使用现有 strict `PolicyObservationView`/transport adapter；request contract 引用 cut 和 payload
identity，不复制一套 unrestricted observation schema。

### 5.2 `PolicyChunkResult`

```text
request/inference identity
producer epoch
inference started/completed
result received
source model chunk size
model-space output descriptor/hash
decoded absolute actions
server capability/protocol version
conditioning prefix/hash
effective RTC parameters
server timing/transport timing
error/fault descriptor
```

Result 是 immutable worker output。Worker 不持有 plan queue，不分配 proposal sequence，不提交 safety/reference，不调用
SDK。

### 5.3 `TimedPolicyActionKeyframe`

```text
target_effective_at / effective interval
action period / clock domain
decoded absolute action
request/inference/output/chunk/keyframe lineage
observation cut and representation receipt
plan generation
state = mutable | producer_committed
valid_until
```

Executed/popped keyframe不继续留在 future plan 中；其事实由 emission/proposal/execution trace承担。

### 5.4 `TimedPolicyActionPlanSnapshot`

- immutable keyframe tuple；
- monotonic target-time order；
- 一个 plan generation 和 canonical content hash；
- cursor/remaining mutable horizon；
- producer committed-through；
- current in-flight request identity；
- 不包含 lock、thread、socket、policy object 或 backend handle。

### 5.5 `PolicyChunkMergeReceipt`

至少包含：

```text
old/new plan generation and hash
request/result identity
admitted/rejected verdict and reasons
actual cursor/committed boundary
preserved committed keyframes
discarded_obsolete_count
discarded_beyond_horizon_count
replaced_mutable_range
new future coverage
late/stale/epoch/deadline facts
merge policy spec id/hash
```

该 receipt 是 producer plan admission 事实，不是 proposal admission 或 backend receipt。

### 5.6 `InferenceDelayEstimatorSpec/State`

Estimator 必须版本化：window size、quantile/max policy、warmup default、network/control guard 和 clamp 都进入 spec/hash。
RTC-0 结束前不冻结生产数值。Runtime state 记录样本但不进入 logical spec hash。

### 5.7 RTC-3b.5：唯一 producer admission truth

当前已经存在两条各自正确、但不能直接串联的局部语义：

```text
RTC-1: PolicyInferenceRequest -> PolicyChunkResult -> TimedPolicyActionPlan merge
RTC-3b: PolicyChunkPredictionCandidate -> admit_prediction_result() -> legacy ActionChunkBuffer
```

RTC-3b 只证明 split 后与 `sync_queue_v1` 的 queue 行为一致；它不是 RTC result admission。RTC-3c 禁止采用：

```text
worker -> PolicyChunkPredictionCandidate
       -> admit_prediction_result()
       -> legacy deque
```

RTC-3b.5 必须新增一个 control-thread-owned、immutable 的 async producer state（暂定
`PolicyAsyncProducerSnapshot`），把 active request、worker session/lifecycle 与
`TimedPolicyActionPlanSnapshot` 放在同一个状态转换边界中。plan 内已有的 in-flight identity/generation/cursor/
committed-through 不得在外层复制成第二份可独立更新的真值。

目标链路冻结为：

```text
PolicyInferenceRequest
  -> worker.predict_chunk / virtual executor
  -> immutable PolicyChunkResult
  -> control-thread transition-first admission
  -> merge_policy_chunk_result(current plan, current request, result)
  -> new TimedPolicyActionPlanSnapshot + PolicyChunkMergeReceipt
  -> pop_due_policy_action()
  -> one PolicyActionEmission
```

control thread 的最终 admission 仍须基于 current session/domain/producer epoch/request/plan generation/cursor/
committed-through/artifact/codec/frame/deadline 做 receiver-side 检查。已有 `merge_policy_chunk_result()` 是唯一 RTC plan
merge 语义；不得在 frontend、worker 或 legacy queue 再实现第二套 crop/replace 规则。

producer transition 表冻结为：

| event | precondition | atomic outcome |
|---|---|---|
| submit request | producer eligible；无active request；plan无in-flight | `mark_policy_action_plan_in_flight()`与保存完整immutable request在同一snapshot transition完成；外层不复制可独立修改的request identity标量 |
| matching result/fault | result匹配current session和active request | 调用唯一merge；accepted或matching terminal rejection均按merge outcome settle plan并清active request |
| non-matching/duplicate/old-session result | 不匹配current active request/session | 记录typed rejection；current plan和active request均不变，旧result不得误清新request |
| HOLD/FAULT/rearm/reconnect/profile change | transition事实先到达 | 先切epoch/session并invalidate active request/plan，再处理本tick mailbox completion |
| decision due pop | post-transition plan有效 | `pop_due_policy_action()`最多选一个latest still-valid keyframe，并形成唯一正式emission |
| RTC-3c current admission | current emission被下游正式接纳/记录 | 只推进current/popped事实；不commit或stage任何future successor |

完整 `PolicyInferenceRequest` 可作为snapshot中的active payload；plan的in-flight字段是其原子identity projection。两者只由
上述transition一起建立/清除，不允许frontend分别赋值。

建议文档提出的 `PreparedPolicyPlanPatch` 暂不直接引入：当前 decode 已在 worker/executor 侧完成，plan merge 对首批小
chunk 是 pure、有限循环。先测量 decode/merge/control-loop p99；只有 merge 超出冻结预算时，才增加非实时 preparation
stage，并在 control thread 保留 current epoch/generation/commit 的最终 CAS 校验。

## 6. Runtime ownership 与线程模型

### 6.1 线程职责

```text
existing frontend/control loop
  owns observation cut selection
  owns request admission and epoch snapshot
  owns result validation + plan merge
  owns unique action pop
  owns existing Arbiter/safety/execution calls

policy worker
  owns blocking policy/transport inference
  owns model preprocessing required by policy adapter
  publishes immutable result/error
  never mutates PolicyDeployRuntime queue/plan
  never calls safety/reference/backend/SDK

existing position workers
  unchanged downstream owner under current profile
  never call policy inference
```

RTC 不要求新增 sensor thread；继续使用当前 causal observation cut 和已有相机/state acquisition。也不要求 Python
`async/await`；第一版可以使用一个 OS thread 包装现有 blocking transport。

### 6.2 bounded mailbox

- request mailbox capacity 1；
- single valid in-flight request；
- result mailbox bounded；
- 不允许 GPU 忙时排队执行一串旧 observation；
- duplicate/late/out-of-order result 由 request/epoch/result admission拒绝并记录；
- lock 内只做短时 identity/state check 与 immutable snapshot swap，不做 decode、network、GPU 或全量 trajectory fitting。

### 6.3 prediction transaction

当前 debug/replay policy 可能实现 `begin_prediction_transaction()`、`commit_prediction_transaction()`、
`rollback_prediction_transaction()`。第一版 RTC worker 不跨线程接管该 transaction：

- production model policy 可进入 async worker；
- transaction-capable debug/replay policy 默认保持 `sync_queue_v1`；
- 若以后需要异步 replay，必须增加 worker-owned transaction + control-thread admission ACK/rollback handshake，并证明
  shutdown/cancellation/retry exactly-once；
- 不得让 worker commit 已被 control thread 因 stale epoch 拒绝的 chunk。

### 6.4 lifecycle/epoch

下列事件必须递增或切换 RTC request/plan generation，使旧 result 无效：

- pause/hold/fault；
- human takeover/authority transition；
- policy reset/restart；
- IK recovery discard gate；
- manual rearm/home/recovery；
- action profile、artifact、frame/layout、active side 变化；
- worker/transport reconnect 后无法证明 session continuity。

`queue.clear()` 本身不是 worker drain 或 cancellation acknowledgement。RTC worker 必须进入现有 M1C/M1D 的
participant/epoch 证据模型后，才能参与真机 authority；在此之前只允许 sim/dry-run/shadow。

RTC-3b.5 还必须给 worker instance 分配不可复用的 `worker_session_id`，并明确以下状态：

```text
ACTIVE
RETIRED_WAITING_FOR_COMPLETION
ABANDONED_UNDRAINED
DRAINED
```

Python thread 无法强制终止已经阻塞的 infer call；`stop(timeout)` 返回不代表该调用已消失。旧 session 的 completion
即使稍后到达，也只能形成 rejected/observed fact，不能进入新 session 的 plan。RTC-3c virtual-policy fixture验证状态与
late completion；RTC-4b-T1已补local cancel/close/reconnect receipt，RTC-4b-T2a/T2b现已分别闭合no-model localhost与
existing model server synthetic/no-action transport验收；真实模型调用
尚未与bounded worker、epoch/TTL和唯一timed-plan admission组合，因此不形成control loop不阻塞的live证据。
是否将 local model 隔离到 process，必须依据具体 backend、资源回收和测量结果决定，本阶段不拍板。

### 6.5 failure policy

Runtime failure不能隐式转为 control-thread synchronous inference。每个 execution profile显式选择：

```text
reject_new_result
expire_remaining_plan_at_TTL
request_hold_or_pause
fault_latch_if_required
```

是否允许继续执行已经 producer-committed 的短前缀，必须由 profile、安全和 TTL 决定。不能无限重复最后一个
policy action，也不能用当前 measured pose 在每个 gap 静默重锚定。

## 7. Sim 集成策略

### 7.0 sim observation 由 sim2real 并行主线负责

本计划不新增、修改或迁移 MuJoCo→Real observation adapter、相机/状态拼装、frame alignment、modality parity 或
sim2real 数据契约；这些工作由 sim2real 方向并行负责。RTC 只消费其发布的 immutable deployment observation cut，
并将以下内容视为外部前置事实：

```text
observation_cut_id + monotonic timestamp
artifact-declared model inputs
actual TCP anchors for relative action decode
frame/tool/calibration identities
observation contract/hash
```

RTC 测试可使用 synthetic/fake observation fixture 验证 scheduler 和 representation contract，但不得据此实现第二套
production sim observation，也不得把 fake qpos observation 提升为通用模型输入。sim2real contract 尚未发布或 identity
不一致时，相关集成 gate 保持 blocked；RTC 不进入其文件修复或猜测缺失字段。

### 7.1 contract-first deterministic runner

先建立测试侧 `DeterministicPolicyChunkRunner`（名称暂定），只组合：

- ManualClock；
- 由现有 fixture 或 sim2real contract 提供的 immutable observation cut；
- pure inference request/result/plan merge；
- deterministic completion schedule；
- 每个 decision time 唯一 pop 一个 due keyframe；
- existing `MujocoDeployRuntime.submit_policy_action()`；
- existing authoritative step/trace/reference receiver。

不要直接使用 `submit_policy_action_chunk()` 同时提前 admission 旧/新整 chunk。兼容 API 和现有测试保留，但 RTC
runner必须证明每个 target time 只有 producer merge 后的唯一 emission进入 proposal链。

### 7.2 async harness 选择

RTC-2 开始前在以下二选一并记录理由：

1. 给现有 `DeterministicAsyncTopologyHarness` 增加 immutable `policy_chunk_result` operation；或
2. 新增一个只处理 policy inference completion 的窄 harness，并复用相同 release/receipt primitives。

不得同时形成两套 clock、completion ordering、deadline 或 producer-restart truth。无论选择哪种，都不得让
`sim` import `teleop_runtime`。

### 7.3 sim 场景矩阵

- zero latency parity；
- fixed latency `d=1..H`；
- jitter/latency forecast under/over-estimation；
- result恰好在decision cutoff前/后；
- out-of-order/duplicate response；
- result after producer restart；
- pause/HOLD 与 completion 同时发生；
- buffer underrun、empty chunk、invalid action；
- non-integer policy/action/decision rates；
- action burst 不绕过 Arbiter ≤60 Hz；
- scheduled reference pending/START/cancellation competition；
- trace replay 得到相同 merge/emission/decision hash。

### 7.4 model-guided sim

True RTC 先用明确支持 guidance 的 model adapter或 deterministic fake flow policy验证 contract，再接真实 OpenPI。
比较：

```text
sync_queue_v1
naive_async_replace_fixture_only
async_aligned_chunk_v1
rtc_flow_inpaint_v1
```

`naive_async_replace` 只作为评估 baseline，不进入 production config。

## 8. Real frontend 集成策略

### 8.1 不把现有 `predict_and_queue()` 直接丢进线程

需要把当前行为拆为两个清晰阶段，同时保持 sync parity：

```text
predict_chunk(request) -> immutable PolicyChunkResult
admit_prediction_result(result, current_state) -> merge receipt
```

Sync mode 在同一线程依次调用二者；async mode由worker调用第一步、control thread调用第二步。Queue counters、chunk ID、
lineage retention、prediction schedule、discard gate 和 transaction behavior 必须有 parity fixture，不能用大规模 rewrite
替换当前 `PolicyDeployRuntime`。

### 8.2 control-loop 顺序

目标顺序：

```text
sample current gate/authority/profile inputs
apply lifecycle transition first; invalidate epoch/request/plan when required
capture current observation cut only for a still-eligible producer
poll worker result without blocking
validate/admit result against the post-transition state and atomically swap plan
optionally submit one new inference request
pop one due action
existing emission/shadow/canonical safety/execution
sleep to current action-rate schedule
```

同一个 tick 同时出现 completion 与 HOLD/FAULT/pause/rearm/reconnect/profile change 时，transition 必须先线性化；旧
completion 随后按新 epoch/session fail closed。不得先 merge 再依赖后续 `clear()` 撤销已经发布的 plan。

结果 decode/merge如果不能稳定落入 control-loop budget，应在非实时 preparation thread生成 ready-to-admit immutable
candidate，control thread只做 cheap validation/snapshot swap。任何这种优化都必须保留 receiver-side current epoch/commit
二次检查。

### 8.3 RTC-3c lookahead disabled；下游D-LA才做typed preview

RTC-3c profile 不形成或不消费 position lookahead；传给 position processing 的 successor 必须是 `None`。因此当前批次
不修改 `PositionDeployExecutor`、canonical safety 的 preview 分支或 resampler staging contract，也不需要用兼容
handshake 推进 future successor 的 `commit_through`。

必须验证：

1. RTC profile 下 `prepare_next_action()`/RTC emission adapter 不返回或不消费 position lookahead；
2. 每个 keyframe 首次可能影响 position reference 前，已经存在唯一 `PolicyActionEmission`；
3. 新 result 可替换全部尚未 pop 的 mutable future，不撞上 hidden staged successor；
4. 开关 RTC profile 不改变 `sync_queue_v1` 的 lookahead 数值、顺序、pop count 和 emission hash。

D-LA再引入typed preview receipt。其receipt至少绑定preview identity/`valid_until`、canonical safety verdict、
downstream acceptance、reference generation/source index与later identity promotion；没有这些字段时，RTC lookahead
保持关闭。该receipt不得推进raw-RTC producer commit-through；现有`PositionDeployExecutor.submit()`仍不能被改称
shared target-time reference admission。

### 8.4 dry-run/shadow 限制

RTC-3/RTC-6 真机 frontend 验收必须满足：

- `send_actions=false`；
- 不打开或改变设备状态作为测试前提；
- 不连接第二个 dispatcher；
- worker error不改变 legacy send顺序/次数；
- 开/关 async/RTC 不改变 observation firewall、decode、canonical safety 或 shadow authority语义；
- 正式 capture 接入前遵守现有 ControlTrace profile/seal要求。

## 9. 通用 RTC v2 协议与模型 adapter 计划

RTC v2 contract先在`robot_motion.deploy`冻结；OpenPI只是后续一个实现，不是协议命名空间或权威来源。

### 9.1 capability handshake

Server metadata 必须显式声明，客户端不能通过 model 名称猜测：

```text
rtc_protocol_version
rtc_modes_supported
rtc_guidance_implementation_id
supported_action_pose_modes
prefix_action_space/layout/normalization
prediction_horizon
min/max execution horizon
delay units/cadence
codec/frame/artifact hashes
supports_conditioning_receipt
```

Metadata 缺失、mode不匹配或 hash不一致时，`rtc_flow_inpaint_v1` fail closed；可按启动配置选择
`async_aligned_chunk_v1`，不能静默声称 model-RTC；它在没有 prefix/receipt 时仍是 raw-RTC。

### 9.2 request

在现有 observation payload 外增加版本化 RTC envelope：

```text
request_id / producer_epoch
rtc_mode
d_forecast_steps
minimum_execution_horizon
previous_chunk_suffix_model_space
suffix valid length/mask
request action cadence/horizon
codec/frame/layout/normalization hashes
```

### 9.3 response

```text
echoed request/epoch/protocol
guided model-space action chunk
effective d/horizon/mask schedule
conditioning prefix hash
guidance/clipping/denoise implementation identity
server inference start/end
model/artifact identity
```

### 9.4 模型 backend / 服务端职责

Paper-style RTC 的 gradient guidance、soft mask 和 denoising integration 在声明该continuation capability的具体
model backend内完成。客户端
不对最终 chunk 伪造等价 guidance。服务端改动必须：

- 位于明确的model adapter/checkpoint commit；OpenPI实现另绑定其checkout commit；
- 有 offline deterministic/model fixture；
- 记录 denoising steps、guidance weight clipping、prefix schedule；
- 测量 vanilla/RTC GPU latency和内存；
- 对 NaN/shape/mask/horizon mismatch fail closed；
- 与本仓库 protocol fixture 做 serialize/deserialize parity。

具体算法超参数不从论文/LeRobot示例直接复制；RTC-4b 依据目标 model/checkpoint、action cadence和实测 latency单独冻结。

### 9.5 协议演进：scheduler 与 continuation algorithm 分轴，但不破坏 v1

现有 `sync_queue_v1`、`async_aligned_chunk_v1`、`rtc_flow_inpaint_v1` 已进入 request hash、测试与文档，RTC-3c 不重命名
或删除这些 mode。RTC-4 的 protocol v2/capability 才将两个维度显式分开：

```text
scheduler_mode
  = sync_queue | async_timed_plan

continuation_mode
  = none | flow_inpaint | trained_prefix_conditioning | native_continuation
```

v1 mode 通过显式 compatibility mapping 投影到两轴，不能让 server/model name 隐式选择算法。training-time action
conditioning、native continuation 等新方法只作为可选 capability；在目标 checkpoint、协议 receipt、codec 和效果/延迟
fixture完成前，不升级为 RTC-3c production 要求。

同理，现有 `minimum_execution_horizon_steps` 等 v1 字段保持 wire/hash 兼容。后续 protocol v2 可把 prediction horizon、
fresh mutable suffix、guidance prefix、coverage guard 与 committed hard prefix 分型；本批只在文档和内部 trigger spec中使用
清晰术语，不做破坏式批量 rename。

## 10. 分阶段实施

### RTC-0：基线观测，不改变行为

目标：先知道当前真实 latency、loop overrun 和 buffer coverage，避免在没有数据时冻结阈值。

计划修改面：

- `robot_motion/deploy/runtime.py`：补全 triggered/started/completed/received 可观测字段，保持 sync；
- `teleop_runtime/deploy/run_policy.py`：记录 action-loop duration/overrun、queue depth、prediction latency；
- OpenPI client debug metadata：区分 serialization/network/server/model（只有服务端提供时记录）；
- 不修改 queue policy、pop、dispatch、last_action、safety、owner。

最低指标：

```text
inference latency p50/p95/p99/max
transport/server/model split when available
action loop jitter/overrun
queue depth in steps and seconds
chunk source/executed horizon
empty queue/hold duration
position lookahead staged/accepted count
policy error/reconnect/reset
```

退出条件：

- sync golden、queue pop count、chunk index、canonical command不变；
- 指标有明确 clock domain和单位；
- 目标模型/网络条件有可重复的无硬件 measurement profile；
- 不把 debug log 冒充 formal ControlTrace。

### RTC-1：pure contract 与 timed plan

目标：实现第5节 pure contract 和确定性 merge，不加 worker、不接 frontend。

建议文件职责（正式命名前复核现有目录）：

```text
robot_motion/deploy/inference.py     # request/result/delay contracts
robot_motion/deploy/chunk_plan.py    # timed keyframe/plan/merge receipt
```

也可合并为一个窄文件，但不得把 OS thread/websocket/Torch 放进 `robot_motion`。

必须测试：

- zero-latency replace/append compatibility projection；
- target-time order、non-integer cadence；
- committed prefix不可覆盖；
- obsolete crop、empty future、deadline/TTL；
- request/epoch/artifact/frame mismatch；
- duplicate/out-of-order result；
- immutable snapshot/hash/serialization；
- qpos 与 fixed-anchor relative pose codec fixture；
- plan merge receipt可重放。

退出条件：pure tests全部通过，现有 production caller为零，`sync_queue_v1`无行为变化。

### RTC-2：deterministic MuJoCo async aligned chunk

目标：在无 OS thread、无真实模型条件下贯通：

```text
observation cut -> request -> delayed result -> plan merge
-> one-keyframe emission -> authoritative sim -> reference/qcmd/measured
```

该阶段的 deterministic fixture只用 bimanual qpos，以隔离验证scheduler/authority；它不是通用 RTC action
contract。完成第7.3矩阵，并验证 result completion不会绕过 receiver admission、Arbiter
rate、safety或scheduled-reference generation。

退出条件：相同 seed/schedule 的 merge/emission/decision/trace hash可重放；async harness仍明确不是 production
scheduler。

当前实施状态为`RTC-2 deterministic local-only gate complete`：虚拟qpos policy已贯通causal observation、
delayed immutable result、plan
merge、单帧pop、`submit_policy_action()`与现有authoritative MuJoCo decision/dispatch。zero/fixed/under-forecast、
jitter replay、duplicate、fault、HOLD和epoch mismatch已覆盖；scheduled-reference profile已验证generation 1→2
successor、前一段committed endpoint、pending/active切换、旧generation retirement、planner qcmd、measured qpos和
cancellation trace。HOLD lifecycle会显式清plan/切producer epoch，旧completion必须在merge边界fail closed。
result cutoff边界、d=1..4、over/under-forecast、invalid decoded action、empty execution suffix、29 Hz cadence和
missed-tick no-burst均已覆盖；相同schedule的完整trace hash可重放，因此满足RTC-2的deterministic退出条件。
本结论不包含OS worker、真实model或frontend wiring；这些分别属于RTC-3及以后阶段。

### RTC-3：real deploy frontend 的无硬件 async aligned shadow

RTC-3 的目标是证明异步 producer 接线，而不是在同一批重构 position preview 或实现 OpenPI model-RTC。首个 RTC-3c
只使用可控虚拟 policy、`send_actions=false` 和 RTC-only lookahead-disabled profile；blocking OpenPI transport 的
deadline/reconnect/loopback 在该 gate 后另批接入。

当前实施状态为`RTC-3c virtual-policy default-off frontend local dry-run complete`：bounded OS worker、唯一timed-plan
producer snapshot及同一个正式`PolicyActionEmission` seam已由现有`run_policy.py`在显式RTC profile下组合；默认profile
仍调用原`predict_and_queue()`，transaction-capable debug/replay仍固定同步。`PolicyDeployRuntime`的
`predict_chunk()`/`admit_prediction_result()`继续只作为legacy queue parity seam，不进入RTC timed plan。现有worker stop
receipt刻意不是M1C/M1D `WorkerEpochDrainReceipt`；本状态也不包含OpenPI transport、真实observation shadow或SDK。

#### RTC-3b.5：single admission truth closure（local-only complete）

已按以下顺序落地，每一项仍保持无 production caller：

1. **A—baseline/evidence：** 锁定 RTC 聚焦命令、当前 commit/dirty fingerprint、RTC profile/spec/hash 和主线外部失败，
   生成 machine-readable、non-formal evidence manifest；并行 MuJoCo v1→v2 失败不得被 RTC patch 顺手修改；
2. **B—unique producer state：** 在 `robot_motion/deploy/async_producer.py`（暂定名）增加 immutable
   `PolicyAsyncProducerSnapshot` 与 pure transitions；唯一 merge 调用现有 `merge_policy_chunk_result()`，不经过
   `PolicyChunkPredictionCandidate`/legacy deque；
3. **C—transition-first/session：** 扩展 `robot_motion/deploy/policy_worker.py` 的 worker session/lifecycle receipt，覆盖
   completion 与 HOLD/FAULT/rearm/reconnect/profile change 同 tick、blocked old-session late completion；不冒充 M1C drain；
4. **D—cadence/lookahead profile：** 复用现有 rate contract冻结15/30/60/120 Hz交叉fixture；RTC profile强制
   lookahead disabled，`sync_queue_v1` 数值/hash golden不变；
5. **E—due-pop observability：** 在不改变“latest still-valid due、no catch-up burst”语义的前提下补 selected target、
   oldest skipped target、lateness 与 skipped count receipt。

实际文件边界：

```text
robot_motion/deploy/async_producer.py       # pure owner state + transitions；无 thread/I/O
robot_motion/deploy/chunk_plan.py           # 仅补 due-pop observability；保留 merge truth
robot_motion/deploy/policy_worker.py        # OS worker session/lifecycle；无 plan/SDK
tests/test_deploy_rtc_producer_state.py     # race/admission/state-machine matrix
tests/test_deploy_rtc_policy_worker.py      # session/blocked/late completion
tests/test_mujoco_rtc_virtual_policy.py     # cadence + stale due observability
```

退出证据：五组RTC直接fixture`53 passed`，deployment/OpenPI/position/MuJoCo兼容回归`150 passed, 1 skipped`；
machine-readable manifest为
[`rtc_3b5_local_evidence_v1.json`](../evidence/rtc/rtc_3b5_local_evidence_v1.json)。该证据明确是
`non_formal_local_only`，没有调用production入口或硬件，也不提供M1C drain、real shadow或motion authority。

若现有 `PolicyActionEmission` builder 不能直接承载 timed keyframe，应提取共享 builder，但继续使用同一个 emission schema；
不得新增 RTC 专用的第二种“正式 emission”。`robot_motion/deploy/runtime.py` 的 RTC-3b legacy split seam保持原样，仅以
golden test证明未漂移。

#### RTC-3c：virtual-policy default-off frontend shadow

RTC-3b.5通过后，已在existing deploy application内增加窄、default-off wiring；没有新增entrypoint：

```text
robot_motion/deploy/rtc_runtime.py           # control-thread coordinator，无设备owner
teleop_runtime/deploy/rtc_frontend.py        # import兼容shim，无RTC ownership
teleop_runtime/deploy/run_policy.py         # startup mode/profile选择与现有loop单点调用
tests/test_deploy_rtc_frontend_shadow.py    # virtual policy, send_actions=false, long-run
```

必须验证：

- worker infer 阻塞/延迟不阻塞 control loop，mailbox始终bounded且single in-flight；
- request/result/session/epoch/plan generation/committed-through只有一个 admission truth；
- 每个 selected keyframe恰好一次 `PolicyActionEmission`，无 whole-chunk proposal admission；
- TTL、underrun、HOLD/FAULT、late completion 和显式 rearm 行为可重放；
- RTC profile下future successor不进入position resampler；new result可替换全部un-popped mutable future；
- `send_actions=false` 时不新增 SDK/backend/dispatcher call；
- startup default仍是`sync_queue_v1`，RTC关闭时 production行为完全不变；
- transaction-capable debug/replay继续同步；不修改遥操、采集或replay链路。

本地退出条件已满足：virtual-policy long-run与race矩阵全绿，control-loop/decode/merge budget有non-formal
fixture evidence，默认关闭且测试没有硬件动作。状态冻结为`rtc_3c_virtual_shadow_ready_local_only`，不等于OpenPI
transport ready、real shadow或motion authority。production application的`--help`只做解析检查；本批没有用真实入口
启动camera/backend，因此“无新增SDK/backend/dispatcher call”由isolated frontend fixture与`send_actions=false`
fail-closed验证，不宣称完整应用资源初始化已经过无硬件现场审计。

#### 下游D-LA：pre-motion lookahead gate（后续独立批次）

只在 typed preview identity/TTL、canonical safety verdict、downstream acceptance、reference generation/source index、
later identity promotion/no-duplicate fixture全部完成后，允许RTC profile重新启用position lookahead。该批可以修改
position deployment协议，但不得反向改变RTC producer commit、RTC-3c或`sync_queue_v1`基线。

### RTC-4：通用 capability/protocol 与模型 adapter

目标：先冻结独立于模型和真机的版本化协议/capability，再由具体模型实现model-side continuation/guidance。

顺序：

1. **RTC-4a generic core（已完成）：** pure capability、scheduler/continuation双轴、v1 mapping、request/response/
   conditioning receipt与wrong capability/hash/mask/horizon fail-closed；
2. **RTC-4a local adapter（已完成）：** 两个deterministic fake backend + canonical decoder loopback，证明替换backend
   不改变RTC request/producer/control contract；fake实现只在测试中；
3. **RTC-4b-A final-chunk protocol adapter（已完成）：** 先补continuous+discrete完整输出contract，再以现有artifact、fake
   transport和V22 codec完成`continuation=none` offline round-trip；只宣称raw-RTC，不宣称model-RTC；
4. **RTC-4b-T existing-server transport：** T1已完成local cancel/close/drain/reconnect typed lifecycle；T2a已完成
   model-neutral deadline、fake socket压力测试和no-model localhost websocket loopback；T2b已直接复用现有OpenPI
   `observation -> final chunk` model server完成两次synthetic live no-action loopback与fresh reconnect。T2c已把现有
   final-chunk adapter、bounded worker和唯一timed-plan producer组合完成cold-late→warm-accepted healthy admission；T3才接真实
   observation no-action shadow。上述gate都不修改model/server sampling语义；
5. **RTC-4b-B optional continuation provider：** offline dataset/model unit与model-space prefix→canonical absolute suffix
   round-trip；带discrete output者必须先有typed discrete prefix/receipt；
6. **RTC-4b profile：** latency/memory与代表性兼容模型实例的效果门槛；实例只提供证据，不成为启用白名单；
7. OpenPI只作为首个候选adapter/provider，不阻塞其他本地或模型实现复用RTC-4a。

RTC-4a退出条件已满足：两个不同的deterministic local backend通过同一capability/envelope/decoder测试；v1映射、
wrong capability/hash/mask/horizon全部fail closed；无Torch/JAX/websocket/OpenPI/sim/teleop/hardware import。该结论
只证明协议和adapter迁移面，不证明任何具体模型支持model-RTC。

RTC-4b-A 在接现有artifact/client前增加一个通用输出完整性gate。`PolicyIOLayout`允许float32 continuous action与
int64 discrete token action并存，因此RTC response、capability和decoder必须分别绑定两者的维度、dtype、horizon与hash，
不得把token cast成float或静默丢弃。首批只允许：

```text
continuation=none + continuous output [+ discrete token output]
```

当前`PolicyInferencePrefix`只保存continuous model-space prefix；在新增typed discrete prefix/mask/conditioning receipt前，
任何`model_action_discrete_dim > 0`的flow/trained/native continuation必须fail closed。该限制不影响continuous-only model-RTC
研究，但防止离散夹爪模型被错误标记为已完整conditioning。

RTC-4b-A concrete adapter冻结为离线、无网络fixture优先：复用现有`PolicyArtifactContract`、server metadata validation、
   `RemoteModelPolicy` final chunk与`V22DeploymentAdapter`显式codec；只声明 raw-RTC 的`continuation=none`，不接RTC-3c worker，不启动
websocket，也不从model name推断能力。artifact完整内容、layout、codec/frame/normalization与server metadata摘要都必须进入
capability/implementation hash；任何缺失或漂移在inference前拒绝。

RTC-4b-A退出条件已满足：offline fake transport通过artifact/server/input/output/codec完整round-trip，只声明
`continuation=none`。完整RTC-4b退出条件仍未满足：具体continuation backend必须返回可验证conditioning receipt，客户端
不得靠model name猜RTC支持。

RTC-4b-T 不需要OpenPI模型侧适配。现有server继续只接收observation并返回final chunk；DexTeleop worker负责不阻塞control
loop，RTC protocol adapter负责identity/validation，timed-plan负责late result和mutable future。该路径可以进入真实transport
的`send_actions=false` shadow，但只能声明 raw-RTC 的`async_timed_plan + continuation=none`，不能宣称 prefix-conditioned model-RTC。
T1只完成本地session lifecycle：cancel hook被独立隔离且最多调用一次，OpenPI transport close不阻塞control stop；
cancel、worker drain与新session begin各自有独立事实。T2a已用通用client contract补connect/metadata/infer/close预算、
typed phase receipt、late-socket退役，并在无模型localhost websocket验证close后重连和metadata continuity。T2b已在已有
OpenPI model server上复核exact loaded-instance artifact/metadata/layout/horizon，并经两代connection得到finite final chunk；
该实例只提供兼容性证据，不成为checkpoint/config白名单。T2b本身未经过bounded worker或timed-plan；T2c只用
synthetic observation和`send_actions=false`组合这三层，已验证wire修复、fault/non-blocking、cold TTL拒绝、warm single
admission与完整drain；T3才允许真实observation no-action shadow。transport预算不进入model capability，也不替代request
`valid_until`和plan TTL。
Control Station在该gate之前最多投影已经存在的artifact/capability/source-binding一致性证据；它不得读取
`OpenPIWebsocketTransport.metadata`来探测状态，因为该property会主动连接并可能持续重试。`server metadata validated at
backend construction`、`live transport ready`、`real observation ready`、`target compatible`和`motion authorized`必须保持为
五个不同结论。

#### RTC-4b-B：checkpoint/config 无关的 OpenPI continuation 落地

RTC-4b-B 不以某个 `model_id`、`config_name`、checkpoint step、固定 horizon 或固定 action dimension 作为接入条件。
首个可实现 provider 可以是 JAX Pi0/Pi0.5 flow sampler，但这是明确的**算法/后端能力边界**，不是 checkpoint 白名单；
同一 provider 应适用于所有满足下列 runtime predicate 的已加载模型实例：

```text
loaded model exposes the supported flow-sampling seam
artifact/layout/codec/frame/action-pose semantics are complete
external continuous prefix can pass the model's existing action transforms
internal action horizon/dimension and external action dimension can be derived exactly
normalization identity can be computed from the actually loaded norm statistics
sampling parameters and action cadence are explicit and hash-bound
discrete continuation is absent until typed discrete prefix/receipt exists
```

不满足 predicate 的模型在普通 OpenPI 部署下仍必须照常工作；只有用户显式选择 RTC profile 时才 fail closed并报告具体
unsupported reason。不得根据名称猜测能力，也不得为了让某个 checkpoint 通过而写 horizon=50、external=16D、internal=32D、
15 Hz、absolute-only 等隐藏常量。首轮可以用现有可运行 checkpoint 作为验证样本，但样本不是生产判断条件。

兼容性边界冻结如下：

1. **默认 server 完全不变：** 未传显式 RTC enable 时，不创建RTC wrapper、不编译RTC sampler、不增加RTC metadata，
   原 `Policy.infer()`、`model.sample_actions()`、websocket request/response和registry语义保持原样；
2. **legacy request完全旁路：** 即使server显式启用RTC，未携带版本化RTC envelope的普通observation仍只调用一次原
   `Policy.infer()`；不得提前消费RNG、transform observation、构造prefix或改变返回字段；
3. **新 sampler独立：** 保留原 `sample_actions()` 函数体与JIT入口，新增RTC专用sampling seam；RTC clamped-prefix
   数值projection不能以optional branch塞进legacy compiled path；这里的clamped只描述sampler，不表示downstream hard prefix；
4. **server core保持通用：** 优先以一个可选policy wrapper区分legacy payload和RTC payload，继续复用现有
   `WebsocketPolicyServer -> policy.infer()`；若wrapper不能在不复制/访问私有transform状态的前提下实现，才在
   `Policy`增加独立`infer_rtc()`，但不得重写legacy `infer()`；
5. **capability按实例生成：** startup从已加载model/config、artifact、input/output transforms、norm stats和真实
   `sample_kwargs`构造immutable capability；registry只负责选择现有模型，不新增per-checkpoint RTC allowlist；
6. **artifact/checkpoint只读：** 不原地修改checkpoint。RTC runtime capability绑定完整artifact hash和由实际norm stats
   计算的normalization hash；缺少可重建identity时，显式RTC启动失败，不接受手填占位hash；
7. **profile不做隐式fallback：** RTC协商失败、运行期receipt不匹配或transport异常时进入既有RTC HOLD/pause/fault
   语义；不得在control loop中偷偷切回blocking sync inference。用户下一次启动仍可显式选择原sync profile。

当前artifact/training config没有通用、typed的model action cadence字段，因此不得从已有15 Hz部署默认反推。兼容性优先的
首轮方案是：显式启用RTC时额外要求一个无默认值的RTC-only `model_action_period_sec` startup/profile字段；provider在模型调用前
与`PolicyInferenceRequest.action_period_sec`精确核对，并把该值同时绑定进request/envelope和
`implementation_parameters_hash`。若未来artifact正式声明同一字段，两者必须一致。该方案不修改dataset schema、checkpoint或
legacy server配置；在该字段来源未确认前不进入模型侧代码实施。

推荐的最小模型侧算法是独立的clamped flow-inpaint provider，而不是通用post-hoc blend：

```text
external model-space prefix [Hp, Dexternal]
  -> existing input/action transforms
  -> normalized + padded internal prefix [Hmodel, Dinternal]

x_known(t) = t * prefix_noise + (1 - t) * prefix_action

for each existing Euler denoise step:
  evaluate the original velocity field once
  take the original Euler update
  project only masked prefix rows to x_known(next_t)

final internal output
  -> existing output transforms
  -> validate/re-project masked external rows to the exact requested prefix
  -> RTC response + conditioning receipt
```

该路径不增加transformer forward、不训练模型，也不改未mask suffix的采样方程。`implementation_id`必须标识实际
model family/backend/算法版本；`implementation_parameters_hash`至少绑定denoise steps、integrator、时间方向、prefix-noise
来源、projection时机、mask语义、internal padding规则和final projection。不得把“Pi0.5”或某个checkpoint名字当作算法
identity。

OpenPI最小改动面按以下顺序实施：

```text
RTC-4b-B1 offline model seam (remote source complete)
  - additive Pi0/Pi0.5 RTC-only sampler
  - additive Policy.infer_rtc() with no server caller
  - original sample_actions / Policy.infer / serve script untouched
  - pure/fixed-noise tests only; no checkpoint load

RTC-4b-B2 capability + wire integration (planned)
new RTC capability/provider module
  - runtime support predicate
  - immutable capability + rejection reasons
  - RTC request/response/receipt validation

optional request wrapper
  - legacy request delegates byte-for-byte
  - RTC envelope alone calls the existing infer_rtc seam

serve_dexteleop_policy.py
  - explicit default-off enable flag
  - enabled startup publishes instance-derived capability
  - existing websocket server remains unchanged if wrapper is sufficient
```

RTC-4b-B1已按该边界完成：远端OpenPI只修改`models/pi0.py`、`models/pi0_test.py`、`policies/policy.py`和
`policies/policy_test.py`。原`sample_actions()`和`Policy.infer()`函数体未改；新增sampler/`infer_rtc()`默认关闭且没有server
caller。fake policy覆盖legacy fixed-noise parity、hard mask、invalid prefix、不同external/internal dimension的
normalize→pad→sample→crop→unnormalize往返；相关测试`16 passed, 2 deselected`，V22 adapter回归`10 passed, 3 skipped`，
Ruff通过。`policy_config.py`、`serve_dexteleop_policy.py`和`websocket_policy_server.py`回读hash保持基线不变。证据见
[`rtc_4b_b1_openpi_remote_source_evidence_v1.json`](../evidence/rtc/rtc_4b_b1_openpi_remote_source_evidence_v1.json)。

B1只证明模型推理库能够在独立入口消费prefix/mask，不声明server、transport、完整checkpoint/JIT性能或DexTeleop
continuation已经接通；下一步仍先做RTC-4b-T existing-server final-chunk transport。B1失败时删除新增caller-free seam即可
回滚，legacy deploy不受影响。

B1落地后的DexTeleop主线复核仍为RTC聚焦`130 passed`；相同兼容矩阵随主线新增两项测试后为
`154 passed, 1 skipped`。RTC-3c/4a evidence中的`152 passed, 1 skipped`继续作为当时测试集的历史记录，不回写旧证据。

B1批次没有修改DexTeleop的`sync_queue_v1`、RTC-3c coordinator、position executor、遥操、采集或replay。后续B2才在
DexTeleop侧新增`OpenPIFlowContinuationRtcBackend`，并与现有`OpenPIFinalChunkRtcBackend`并存；前者只在capability协商
通过后使用，后者仍是`continuation=none`的兼容/离线适配器，不能被原地改义。

兼容性验收必须先于效果和性能验收：

- RTC关闭时，代表性legacy request的metadata、输入transform结果、固定noise输出、返回键、RNG推进序列与现状一致；
- RTC开启但收到legacy request时，wrapper只delegate一次，结果与未wrap policy一致，且不会触发RTC JIT；
- 测试用synthetic model/transform矩阵覆盖不同horizon、external/internal dimension、camera set和action pose codec，证明实现
  没有checkpoint/config常量；真实checkpoint只作为样本追加，不作为判断逻辑；
- capability由loaded instance生成，任意artifact/layout/norm/sampling parameter/action period漂移都会改变hash并使旧请求拒绝；
- `continuation=none`固定noise与旧final-chunk路径保持golden parity；partial/full/empty-invalid prefix、mask边界、NaN/Inf、
  horizon/dimension/hash错误和unsupported backend全部fail closed；
- masked prefix在首次可能进入producer plan前已有有效conditioning receipt，suffix保持有限；固定observation/noise/request的
  response与receipt hash可重放；
- 启动后先prewarm vanilla与RTC两个独立JIT，再分别测p50/p95/p99 latency、显存和compile count；测量前不冻结性能阈值；
- default-off OpenPI server/client与DexTeleop `sync_queue_v1`兼容回归完全不变。

实现前仍有一个工程性前置条件：当前远端OpenPI工作树的已检查源码不能仅由其HEAD commit复现。必须先冻结远端
`git status`与目标文件hash，并确认一个可审计的本地OpenPI checkout/realtime-sync编辑面；不得通过SSH直接修改远端源码。

### RTC-5：pose representation 收敛与 target-side sim 验证

RTC-5 不实现 sim observation。它等待 sim2real 并行方向提供满足第7.0节的 observation contract，只负责将模型 final
chunk 经 absolute/relative raw-RTC 分支收敛为同一 canonical action，再接既有 sim target/authority。

按以下子 gate 顺序执行：

1. **RTC-5A typed canonical action：** 在不删除 `NormalizedAction` sync compatibility 的前提下，使
   `PolicyChunkResult`/`TimedPolicyActionKeyframe` 能携带 arm absolute pose + typed end-effector target；
2. **RTC-5B absolute raw-RTC：** 以不同 horizon/layout/model backend 的 final chunk 验证 declared-frame
   canonicalization receipt、timed plan、one-pop/emission 与 target preparation；
3. **RTC-5C relative raw-RTC：** 使用 original request observation cut 的双臂 fixed anchor 解码；结果完成后机器人/
   fixture 已移动时仍保持相同 absolute target，并覆盖 wrong-current-anchor negative control；
4. **RTC-5D sim target integration：** 仅消费 sim2real 已提供的 observation contract，把 canonical arm pose 经
   canonical safety/IK 送入 arm target，把 end-effector qpos/position target送入对应 executor；不走 direct arm qpos
   shortcut。

最低验收：

- absolute 与 relative branch 在构造等价目标时产生等价 canonical pose、end-effector target 和 emission lineage；
- end-effector component 在 pose frame conversion/re-anchor 前后数值与 identity 不变；
- mixed/stale anchor、wrong frame/tool/layout/norm、单侧双臂失败和 arm-qpos production payload fail closed；
- zero/fixed/jitter latency下的 TTL、underrun、epoch、late completion、mutable-future replacement 与 one-emission；
- target-side planned pose、IK qpos、end-effector target、reference/receipt lineage可追溯；
- `sync_queue_v1` 与 direct-qpos deterministic fixture的既有顺序/hash不变；
- sim observation 的实现、字段完整性与 real parity只引用 sim2real evidence，不由RTC测试代签。

在上述 raw-RTC gate 之后，才可选接 model-RTC continuation，比较 absolute-model prefix 与 relative-model re-anchored
prefix 的 round-trip、delay sweep、chunk-boundary位置/旋转/速度/加速度和 fresh-observation响应性。task success/
throughput只作为效果指标，不替代 causal/safety/representation验收。

退出条件分开签署：`absolute_raw_rtc_sim_target_ready` 与 `relative_raw_rtc_sim_target_ready` 可独立完成；
`model_rtc_sim_ready` 必须另有有效 conditioning receipt。任何一个都不等于 real shadow 或 motion authorization。

### RTC-6：真机无动作 RTC shadow

目标：在真实 observation/transport 条件下生成和评估 RTC plan，但不发动作。

比较方式优先使用同一录制 observation cut 的 offline/loopback replay，避免为了比较同时运行两份模型并改变实时资源。
若需要在线双算，必须明确 GPU contention 对 latency 的污染，且两份结果都只进入 shadow。

最低验收：

- artifact/frame/layout/codec exact preflight；
- 实测 p99 latency下有足够 horizon；
- no stale/old-epoch plan admission；
- source pose mode、canonicalization receipt、planned canonical pose、post-IK qpos 与 end-effector target envelope；
- shared safety counterfactual无新增 fault类别或明确解释差异；
- pause/HOLD/recovery期间无旧 plan复活；
- worker/transport failure请求本地 hold/pause，不隐式 sync fallback。

退出条件：只得到 `rtc_real_shadow_ready` 类型证据，不得到 motion authorization。

### RTC-7：独立 real authority gate

默认顺序：与目标 canonical pose + typed end-effector action component相同的非 RTC execution profile先完成其
M5.8/M6 owner、安全和硬件验收，再申请 RTC-specific real authority。direct-qpos fixture或其他 action profile的
验收不能跨 representation 继承。任何提前运动的例外都需要主 landing plan 显式修订和新的用户硬件授权；本文不提供该授权。

RTC-specific manifest至少重新绑定：

- policy artifact + RTC server/model commit；
- RTC protocol/capability/codec hashes；
- delay estimator和trigger/TTL/underrun policy；
- action horizon/cadence/merge mode；
- worker participant/epoch/drain；
- execution manifest/safety matrix/dispatch inventory；
- real latency、stop/hold acknowledgement和motion acceptance。

真机 rollout 从 stationary/低动态/短时/低速范围开始，逐级扩大；每一级都需要现场安全范围和授权，不在计划阶段
预先授权。

## 11. 验收矩阵

### 11.1 行为兼容

- `sync_queue_v1` 的 model view、decode、queue、pop、chunk lineage、last_action、canonical command golden不变；
- `action_chunk_execution_steps` 的 prefix语义不变；
- `replace/append` 继续只描述 legacy sync queue；RTC plan merge使用独立 versioned spec；
- passive shadow开关不改变 legacy执行；
- RTC profile强制lookahead disabled，开关RTC不改变`sync_queue_v1`的lookahead数值、顺序、pop count或emission hash；
- 不新增第二个 production loop、writer或dispatcher。

### 11.2 causal/time

- observation cut之后的sample不进入该request；
- result不能自报 authoritative available time；
- stale/late/out-of-order response不能改写过去或 committed future；
- target time、emission、proposal admission、command/receipt/measurement分型；
- non-integer rates、jitter和missed tick可重放；
- missed tick只选择latest still-valid due keyframe且不追赶burst；receipt记录selected/skipped target与lateness；
- delay estimator warmup/clamp有版本化默认。

### 11.3 representation

- model-space和canonical absolute均有独立identity/hash；
- absolute/relative只在canonicalization前分支，之后共用同一typed timed plan、merge、pop与emission；
- arm action是pose；end-effector是独立typed qpos/position target，pose变换不得修改后者；
- fixed-anchor relative语义不变；
- relative result只使用original request observation anchor；completion/emission时anchor negative control必须失败；
- absolute branch即使为identity conversion也形成canonicalization receipt；
- wrong frame/anchor/layout/normalization fail closed；
- quaternion不直接线性平均；
- direct arm qpos只允许显式deterministic test/debug capability；production pose payload、NaN/Inf、inactive component检查；
- 双臂anchor来自同一observation cut且coupled keyframe atomic admission；
- discrete/hand未声明capability时拒绝 RTC prefix。

### 11.4 concurrency/lifecycle

- single in-flight；
- bounded mailbox，无旧 observation backlog；
- result admission在current epoch/generation二次检查；
- shutdown、pause、restart、rearm、fault、transport reconnect不会复活旧 result；
- worker不调用 safety/reference/backend/SDK；
- blocking policy call不会阻塞 action/control loop；
- lifecycle/authority/profile transition先于同tick result admission；
- worker session状态区分ACTIVE/RETIRED/ABANDONED_UNDRAINED/DRAINED，旧session completion不能进入新plan；
- worker drain是 typed receipt，不是 `queue.clear()`，blocked thread的stop timeout不冒充drained。

### 11.5 safety/authority

- 所有 emitted action继续走同一 canonical safety；
- future chunk不提前参与 Arbiter authority；
- RTC-3c每个keyframe首次影响position reference前已有唯一正式emission，future successor不被passive staged；
- D-LA之前RTC lookahead与shared reference preview admission保持未实现/关闭；
- no action/TTL/underrun有显式 hold/pause/fault语义；
- RTC不能取得或携带 dispatch token；
- real authority需要新的 exact execution manifest和硬件验收。

### 11.6 性能

至少测量：

```text
control/action loop jitter p50/p95/p99/max
request queue wait
preprocess/network/server/model/postprocess latency
d_forecast error and deadline misses
future coverage in steps/sec
obsolete/rejected/merged chunk counts
selected due target / oldest skipped target / target lateness / skipped count
buffer underrun and hold duration
chunk-boundary position/velocity/acceleration
SE(3) translation/rotation continuity when pose profile is enabled
GPU memory/utilization and RTC guidance overhead
```

性能达标不能替代安全/authority验收；安全通过也不能掩盖 buffer经常 underrun或控制循环超时。

### 11.7 无 RTC gold 对照顺序

表示与scheduler验收采用一条冻结的gold ladder，不把三个独立变化同时引入：

```text
G0: production sync_queue_v1 / no RTC / absolute final chunk
    = 唯一gold

G1: absolute raw-RTC
    = 同一observation cut + 同一captured final chunk
    = 只改变worker/timed-plan调度

G2: relative sync / no RTC
    = 将G0 target按G0 observation anchor编码为relative
    = 仍走原sync逐项执行

G3: relative raw-RTC
    = 与G2相同relative chunk和original request anchor
    = 只改变worker/timed-plan调度
```

首选从正常无RTC调用捕获一次真实model final chunk、metadata与observation cut，再离线重放G0–G3；不要在线同时运行
两份随机policy并把采样差异或GPU contention误判为RTC差异。若模型不能固定RNG，captured output是唯一比较输入。

逐级比较：

- G0↔G1：canonical action row/hash、顺序、effective target time、pop count、emission payload/hash；
- G0↔G2：relative encode→original-anchor decode后的canonical pose与typed end-effector target；
- G2↔G3：decoded canonical action、timed-plan crop、one-emission、TTL/epoch/late-result行为；
- G0↔G3：最终canonical execution payload与安全输入，不要求representation receipt相同；
- moved-current-anchor negative control：用completion/emission时pose解码必须偏离gold并被测试识别。

G1失败时不得继续把问题归因于relative codec；G2失败时不得进入relative raw-RTC；G3只有在G0/G1/G2均通过后才可
签署。local伪策略先覆盖确定性矩阵，真实policy capture/replay再提供模型/transport证据，两者不能互相代签。

## 12. Trace 与调试

RTC-0 可先使用明确标记为 non-formal 的 debug metrics。需要进入 formal ControlTrace 时，先冻结 explicit payload
allowlist和 completeness requirement，建议事件语义至少覆盖：

```text
policy_inference_request
policy_inference_result_received
policy_chunk_result_admission
policy_action_plan_merge
policy_action_plan_invalidation
policy_worker_health
policy_worker_session_transition
```

这些事件不能替代现有：

```text
observation cut
PolicyActionEmission
proposal admission/snapshot/decision
safety/reference/command/receipt/measurement
cancellation/epoch/owner transition
```

Formal event ID/hash、episode membership、WAL/seal/commit必须按现有 M4B/M4C contract接入，不在 RTC worker中建立第二个
writer lifecycle。

## 13. Rollback 与配置

### 13.1 启动期模式选择

运行模式是 startup/profile配置：

```text
sync_queue_v1
async_aligned_chunk_v1
rtc_flow_inpaint_v1
```

每个模式绑定 exact spec/hash。运行中不能因一次网络错误静默从 RTC切到 sync并阻塞control loop；故障处理按第6.5节。

### 13.2 default-off

在各阶段退出前：

- production默认仍是现有 sync行为；
- async/RTC flag在 `--help` 中明确 experimental/dry-run范围；
- artifact/server不支持时启动失败或按用户显式选择的非 RTC mode启动；
- 不写入 pyproject/environment/lock，除非实际新增依赖。

### 13.3 回滚

- RTC-0/1 可以通过不注册caller完全回滚；
- RTC-2 sim runner独立于 real frontend；
- RTC-3 frontend可关闭worker回到原 sync path，但必须在启动前选择；
- RTC-4 protocol capability不影响旧 server/client v1；
- RTC-7 authority回滚必须按 M1C/M1D：HOLD/cancel、终止新epoch、drain worker/segments、切回唯一legacy/shared owner，
  不能热切成双dispatch。

## 14. 当前延后项与不得延后的决定

以下问题依据 RTC-0 数据和 RTC-2/4实验再冻结，不应阻塞 pure contract：

- delay estimator使用 rolling max、p99还是bounded hybrid；
- request/result worker最终用 thread还是独立process；
- OpenPI首个continuation provider覆盖哪些可检测的model/backend能力；不得退化为checkpoint/config白名单；
- prefix payload是否需要压缩/shared memory；
- exact trigger guard、TTL和minimum execution horizon；
- async result的heavy decode是否需要第二个preparation stage；
- model-RTC continuation何时进入；它不得阻塞absolute/relative raw-RTC final-chunk部署；
- production sim application root。

以下问题不阻塞 RTC-3c，但已有明确后续 gate：

- RTC position lookahead 的 typed preview receipt：移到下游D-LA；
- downstream-accepted hard prefix 与 producer-local soft overlap：后续model-RTC typed component capability；
- scheduler/continuation algorithm 两轴协议：RTC-4 protocol v2，保留v1映射；
- async-aligned chunk boundary continuity gate：模型/轨迹profile实测后再冻结，不自动取得motion authority；
- property-based framework：先用确定性race table/exhaustive小状态测试，不为RTC-3b.5直接新增Hypothesis依赖。

不允许待决：

- whole-chunk admission是否静默替代 on-pop：不允许；
- worker是否可直接发SDK：不允许；
-是否把 weighted blend放进Arbiter：不允许；
-是否复用原M6硬件授权：不允许；
-是否让 `sim` import `teleop_runtime`：不允许。

## 15. 本轮执行结果与推荐下一批

### 15.1 对新增建议的采纳决策

| 建议 | 决策 | 当前代码依据或调整理由 |
|---|---|---|
| RTC-1 plan与RTC-3b queue收敛为唯一admission truth | 直接采纳 | 两条局部语义均已实现，但candidate→legacy deque不能表达epoch/deadline/committed future；RTC-3b.5改走timed plan唯一merge |
| lifecycle/gate transition先于同tick completion admission | 直接采纳 | RTC-2已按该语义失效旧plan；real frontend文档顺序原先相反，现已更正并列race fixture |
| RTC-3c禁用position lookahead | 直接采纳（用户冻结A） | 当前lookahead会在current submit时把successor staged进resampler；首批只允许正式emission后影响reference，sync行为不变 |
| worker session与retired/abandoned/drained状态 | 直接采纳 | 现有blocked stop能诚实报告not drained，但线程调用可继续存在；session identity用于拒绝迟到completion |
| freeze cadence lattice | 调整后采纳 | 复用已有position rate/capability contract，增加RTC profile绑定与cross-rate fixture，不再造全局重复类型 |
| missed tick只取latest valid target | 只采纳observability | 该语义及29 Hz no-burst测试已存在；只补selected/skipped/lateness receipt，不把已完成语义列为P0重写 |
| last_action/prefix/execution plan分离 | 调整后采纳 | 现有`LastActionContract`及accepted后更新时机正确；保留v1，只在async state/协议中禁止三者混用 |
| `PreparedPolicyPlanPatch` + cheap CAS | 条件采纳 | worker已完成decode，当前merge是pure小循环；先测p99，超预算才拆preparation stage，最终current-state检查不能移走 |
| hard/soft continuation prefix | 延后采纳 | 真正hard prefix需要downstream accepted reference事实；RTC-3c no-action不应伪造，留给后续model-RTC typed component capability |
| scheduler与continuation algorithm分轴 | 延后并保持兼容 | 保留已进入hash/测试的v1 mode ID；RTC-4 protocol v2再增加两轴与显式映射 |
| process worker、transport half-open/reconnect | 部分延后 | RTC-3c先验证virtual policy与session状态；OpenPI接线前必须有transport deadline，local process取决于backend测量 |
| property/soak/evidence | 分阶段采纳 | 先用无新依赖的race table、virtual long-run与machine-readable non-formal manifest；formal trace复用既有WAL owner |
| 直接把candidate接入deque作为RTC-3c | 拒绝 | 会保留两套request/admission truth并绕开timed plan committed boundary |
| RTC-3c立即实现typed preview receipt | 拒绝为RTC内容 | 会扩大为position reference预接纳协议重构；保留为下游D-LA必要gate |
| 立即批量rename delay/horizon/mode字段 | 拒绝破坏式修改 | v1字段已进入semantic hash与fixture；只允许v2 additive contract和compatibility mapping |
| `async_aligned_chunk_v1`自动具备motion authority | 拒绝 | 它只证明调度/因果；real authority仍需独立manifest、连续性、安全、worker drain与现场证据 |
| RTC按absolute/relative source pose明确分支 | 调整后采纳 | 两条分支只负责plan前canonicalization，共用worker/timed-plan/one-pop/emission；不复制scheduler |
| relative result绑定original request anchor | 直接采纳 | fixed chunk anchor已是v2.3语义；completion/emission时重新锚定会使网络延迟改变动作几何意义 |
| plan保存typed canonical absolute action | 分阶段采纳 | `RobotAction`已有通用component schema，但result/keyframe当前仍是`NormalizedAction`；先加版本化payload与单向compat mapping，不破坏sync |
| relative raw-RTC要求模型future prefix | 拒绝 | raw-RTC server只返回final chunk；absolute→relative prefix re-encode与conditioning receipt属于后续model-RTC |
| 新增独立`RelativePolicyInferenceRequest/Result`调度类型 | 拒绝 | anchor/pose-mode进入通用request representation receipt即可；另起类型会复制epoch、TTL、merge与admission truth |
| UMI噪声建模、task-frame augmentation与跨本体训练评测 | 不纳入RTC计划 | 内容可能有研究价值，但属于dataset/model/sim2real主线，不能扩大deployment scheduler范围 |
| RTC实现MuJoCo→Real observation adapter | 拒绝并划出边界 | sim observation由sim2real并行负责；RTC只消费immutable observation cut及其contract/hash |

RTC-3b.5 evidence manifest 只记录 `commit`、dirty-path fingerprint、RTC profile/spec/hash、精确测试命令与结果、
Python/environment identity、采集时间和`non_formal_local_only` scope。它不写入ControlTrace、不声称全量green，也不把
并行主线失败归因或吸收到RTC patch。

### 15.2 已完成范围与下一批入口

本轮完成 RTC-1 pure contract、RTC-2 deterministic virtual-policy sim gate、RTC-3a隔离worker carrier、
RTC-3b split predict/admit sync-parity seam、RTC-3b.5 single-admission closure、RTC-3c default-off local frontend、
RTC-4a model-agnostic protocol/local migration fixture、RTC-4b-A OpenPI final-chunk offline adapter及RTC-4b-T3显式
OpenPI raw-RTC caller接线；不新增或修改OpenPI server进程，不接model-RTC continuation、不做真实observation shadow、
不发送动作、不接真机：

1. 已冻结 RTC contract 使用的 local monotonic clock、target time、cursor、deadline 与 producer committed-through
   字段；RTC-0 production metrics 数值仍待后续实测冻结；
2. 已形成 `PolicyInferenceRequest/PolicyChunkResult/TimedPolicyActionPlanSnapshot/PolicyChunkMergeReceipt` pure
   dataclass；
3. 已完成 qpos、target time、epoch、late result、producer-committed prefix、representation hash、fault 和
   out-of-order deterministic merge测试；
4. fixed-anchor relative pose已从纯codec推进到versioned request context、typed result/keyframe、通用
   adapter/frontend与control-thread anchor identity复核；具体relative artifact和target caller仍不得由pure fixture代签；
5. 已复核文件命名和`robot_motion.deploy`/`teleop_runtime.deploy` export；默认production caller仍是同步路径，RTC-3c
   继续仅允许builtin hold dry-run；OpenPI raw profile必须显式开启且固定`send_actions=false`；
6. 已完成 deployment-only targeted tests与 whitespace检查；
7. 已实现deployment sim范围的窄RTC runner；虚拟policy结果只在decision tick admission，且每tick最多向现有
   authoritative sink提交一个keyframe；
8. 已补`scheduled_reference_segment_sim_v1`下的generation 1→2 successor、committed endpoint、qcmd/measured、
   HOLD cancel、producer epoch plan invalidation和晚到result fail-closed；
9. 已补decision-cutoff前/后边界、invalid/empty result、sim non-integer cadence、d=1..4、over-forecast和
   burst-rate矩阵，RTC-2 deterministic退出审计完成；
10. 已完成RTC-3a隔离、bounded、无production caller的OS worker carrier与虚拟policy lifecycle测试；stop timeout
    不冒充execution drain，晚到completion仍由producer epoch/plan merge拒绝；
11. 已冻结`predict_chunk()`与control-thread `admit_prediction_result()`拆分及sync parity，relative decode继续使用
    原request observation的fixed chunk-start anchor，transaction-capable debug/replay policy继续留在sync路径；
12. 已完成RTC-3b.5唯一producer snapshot、matching-only settlement、transition-first/session、固定
    `send_actions=false`/lookahead-disabled cadence profile与due-pop observability；split candidate不进入timed plan；
13. 已完成RTC-3c coordinator、existing-entrypoint default-off选择、same-emission preparation与virtual-policy 240-tick
    long-run；正式emission后只commit当前popped keyframe，RTC分支不读写legacy deque、不暴露future lookahead，
    HOLD/FAULT/late completion/rearm均有fixture；
14. 已生成`non_formal_local_only` evidence manifest；六组RTC直接fixture`60 passed`，兼容回归
    `152 passed, 1 skipped`，full suite未重跑；
15. 已完成RTC-4a scheduler/continuation分轴、capability/envelope/conditioning receipt、model-neutral client seam、
    两个local backend迁移fixture与continuous/discrete输出gate；加入T1 optional cancellation检查后当前通用协议测试
    `44 passed`；
16. 已完成RTC-4b-A OpenPI final-chunk `continuation=none` offline adapter与28项伪transport/policy鲁棒fixture；RTC聚焦
    回归`130 passed`，兼容回归`152 passed, 1 skipped`；
17. 已完成caller-free RTC-4b-B1 OpenPI offline model seam：独立Pi0/Pi0.5 clamped-prefix sampler与`Policy.infer_rtc()`默认
    关闭，原sampler/infer/server均不改；远端pure/fake测试16 passed、2 deselected，V22回归10 passed、3 skipped；
18. RTC-4b-T1已完成session cancel-close-drain-reconnect local contract；RTC-4b-T2a已完成model-neutral deadline、
    fake socket鲁棒矩阵和no-model localhost final-chunk loopback；RTC-4b-T2b已用现有OpenPI
    observation→final-chunk model server完成两次synthetic live no-action loopback，fresh reconnect保持metadata hash，两个
    50×16 chunk均finite。该T2b结果仍不接worker/timed-plan、真实observation或action send，也不要求模型适配；T2c随后完成
    caller-free live synthetic worker→timed-plan shadow、wire-thaw、cold TTL拒绝、warm 44-keyframe single admission与完整drain，之后才做可选RTC-4b-B capability-driven prefix→canonical suffix
    continuation provider。后者按loaded instance
    的model/artifact/transform/norm/sampling facts协商，不绑定特定checkpoint/config/horizon/dimension；typed preview独立
    留在下游D-LA，正式trace和真实shadow分别复核，继续不修改遥操/replay链路。T3只在显式
    `raw_rtc_final_chunk_15_30_60_120_v1`下复用现有OpenPI transport/client创建shared frontend，仍不改变默认同步调用；
    model-RTC prefix/mask/receipt/identity promotion继续留在后续RTC-4b-B gate；
19. Control Station的CS-0/CS-1c pure contract、null/fake composition、loopback状态、typed intent与独立no-I/O operation
    coordinator已完成；HTTP不执行adapter，bounded UI event cache不作为ControlTrace；其未来policy adapter只允许观察已有request、
    completion、plan/queue depth和唯一`PolicyActionEmission`；不得为了GUI提前把RTC-3a接入production frontend，
    也不得调用`predict()`、merge plan或再次pop action chunk。
20. R-A1/R-R1现已在plan前分别完成absolute与relative typed canonicalization。R-R1 generic virtual-policy使用原
    request observation的fixed measured-TCP anchor形成typed absolute emission，wrong cut/time/contract、legacy
    relative result和tampered anchor receipt均fail closed；模型/server payload、OpenPI absolute artifact、sync queue与
    teleop/replay不变。本轮representation/generic/OpenPI聚焦`65 passed`，扩展RTC/OpenPI/deploy矩阵
    `246 passed, 1 skipped`，另有canonical safety/trace/position兼容`85 passed`，均为no-hardware。

### 15.3 代码落地批次与当前状态

以下批次按顺序、小步落地；每批保持 default-off，并先用通用伪策略覆盖，不以 OpenPI checkpoint 或 MuJoCo direct-qpos
fixture决定 contract：

1. **R-A0 typed payload pure contract**
   - 在 `robot_motion.deploy` 增加 immutable、可hash的 canonical policy action/receipt；
   - payload以 `RobotAction` component为权威，覆盖双臂absolute pose与typed end-effector target；
   - 为现有 `NormalizedAction` 提供单向compatibility view/property，保持旧constructor、sync queue与emission golden；
   - 先改 pure payload/hash/fixture，再接result/keyframe；不接frontend、position、sim target或SDK。
   - 当前已完成pure payload：`CanonicalPolicyAction`复用既有`freeze_robot_action()`与
     `robot_action_semantic_payload()`作为唯一freeze/hash owner；legacy 26D view按需派生，native typed
     end-effector不可无损表示时显式拒绝；
   - `PolicyChunkResult`/`TimedPolicyActionKeyframe`已增加legacy/typed互斥 payload，typed result使用新hash domain，
     legacy result/keyframe/plan golden hash保持不变；
   - typed receipt必须与result的cut/codec/frame/layout/norm一致，且一个chunk不得混合absolute/relative、
     conversion contract或relative anchor；merge/pop保持typed identity；
   - 非空plan拒绝payload kind切换；frontend preparation对typed payload显式fail closed，不做隐式legacy fallback。
   - 当前伪策略fixture已覆盖G0→G1→G2→G3 canonical output、timed-plan逐项pop与wrong-current-anchor负例；
     真实policy capture/replay仍属于R-A1/R-R1 evidence。
   - 2026-07-23首轮payload/result/keyframe/emission fixture为`32 passed`、RTC/OpenPI/VLA矩阵为`225 passed`；
     R-A0b/c/d落地后，typed emission/safety/session定向矩阵为`80 passed`，扩展RTC/OpenPI/VLA/deploy/safety/
     ControlTrace no-hardware矩阵为`303 passed`。均不读取真机observation、不启动模型server、不发送动作。
2. **R-A0c typed formal emission（pure已完成）**
   - due-pop后先形成typed emission，再进入proposal/safety；missing safety只阻止preparation；
   - legacy emission payload/hash与sync queue保持不变；
   - typed ControlTrace记录canonical action/receipt hash，不生成26D副本。
3. **R-A0b native component safety（pure已完成，real profile待配置）**
   - arm pose继续复用explicit-anchor canonical keyframe safety；
   - native EE capability覆盖continuous bounds/step/bounded-rate和discrete token/transition；
   - `previous_safe_committed`与`measured_state`由caller提供，plan/worker不选anchor；
   - arm/EE形成一个atomic safe/modified/fault verdict，不把EE降级为legacy hand qpos；
   - `v22_native + hardware_output`缺safety capability时session resolution启动即拒绝。
4. **R-A0d typed preparation（pure/local已完成）**
   - `CanonicalPolicyActionPipeline`只组合shared safety与`RobotActionResolver`，不持有backend/dispatch authority；
   - missing preparer/capability维持formal emission但不生成command；
   - proposal/Arbiter/reference/backend正式caller仍未接入，不构成sim/real execution evidence。
5. **R-A1 absolute raw-RTC（pure/generic/OpenPI fake已完成）**
   - generic model adapter以artifact声明的absolute frame/layout/norm生成canonicalization receipt；
   - fake local、generic final-chunk与OpenPI absolute adapter复用同一入口；
   - 覆盖identity conversion、frame conversion、2/4/8及非固定horizon、双臂+不同end-effector shape、late result；
   - OpenPI server、payload和默认sync caller保持不变。
   - `AbsolutePolicyActionChunkCanonicalizer`现已把model-space continuous/discrete chunk转换为typed
     `CanonicalPolicyAction`，receipt绑定source/canonical chunk hash与连续source index；
   - `ModelAgnosticRtcClient`只允许decoder返回全legacy或全typed一种representation，混装和pose-mode drift
     fail closed；
   - OpenPI通过显式`typed_canonical_v1`选择native `RobotAction` codec；未配置时继续使用原legacy decoder/hash，
     不修改server payload、transport、worker或同步policy调用；
   - fake回归覆盖2/4/8 horizon、声明frame conversion、双臂、2D continuous hand + 1D discrete gripper、
     chunk identity篡改与late completion。正式proposal/target caller和真实profile仍不属于本批。
   - 本批直接表示/OpenPI/safety矩阵为`147 passed`；RTC/OpenPI/timed-plan/emission/deploy扩展无硬件矩阵为
     `278 passed`。
6. **R-R1 relative raw-RTC（pure/generic frontend已完成）**
   - generic canonicalizer只从immutable `PolicyInferenceRequest`/observation view读取original cut双臂anchor；
   - relative final chunk在worker侧解为canonical absolute，control thread复核receipt后进入同一timed plan；
   - 用移动后的fake measured pose做negative control，证明completion/emission时重新锚定必然被发现；
   - 不传future prefix，不修改模型server，不把`last_action`与future plan合并。
   - `PolicyActionRepresentationContext`现已版本化冻结cut/contract/timestamp/step、active arm group、
     reference/controlled frame、SE(3)约定与每侧measured actual-TCP anchor；context进入request hash，
     `RtcPolicyShadowConfig`保留`request_spec.action_pose_mode`；
   - `RelativePolicyActionChunkCanonicalizer`只转换arm pose，continuous/discrete end-effector component保持原值；
     client与control-thread merge分别复核pose mode/anchor identity，relative legacy ndarray result fail closed；
   - generic virtual-policy frontend已完成relative final chunk→original request anchor→typed absolute
     `PolicyActionEmission`，wrong cut/time/contract与missing builder均在模型调用前FAULT/reject；legacy queue pop为零；
   - OpenPI adapter现已按artifact解析absolute/relative：absolute默认legacy result不变，relative强制typed result并绑定
     structured pose chunk、pose-frame hash、layout-derived arm slice与artifact/server horizon；launcher只对relative raw
     profile自动选择typed codec，server payload不变；
   - OpenPI fake transport/shared frontend已产生fixed-request-anchor relative typed emission并保持`send_actions=false`；
   - R-R2已用registry中的relative `149999/h50`启动未修改的OpenPI final-chunk server。一次真实response被同时送入
     现有sync relative decoder与RTC typed canonical decoder，双臂pose和typed末端target最大误差均为`0.0`；
     随后的真实blocking inference位于bounded worker，50帧中按当前时刻丢弃38个obsolete、接纳12个future，
     并仅形成1个formal emission。该emission绑定original request anchor、source index 38、
     `canonical_desk`双臂frame、`send_actions=false`和lookahead-disabled；legacy queue pop为零；
   - R-R2只使用deterministic synthetic三相机observation，未初始化camera/target/safety/position/dispatcher，
     因此下一gate已收窄为单独授权的real observation/transport no-action，不是新增RTC scheduler或模型适配。
   - R-R3已完成一次授权范围内的real observation/transport no-action shadow：1200个30 Hz loop step后双臂、两侧
     DM-Tac gripper和三路RealSense均形成真实有效observation，OpenPI relative `149999/h50`完成30个唯一
     `continuation=none` exchange；全程`send_actions=false`、gripper `send_commands=false`、lookahead disabled，
     debug事件中没有position/dispatch/action-send入口。该gate尚未通过：当前10-step request validity使cold
     `10.17 s`及warm median `0.705 s`结果全部按`request_deadline_expired`拒绝，formal emission为零；30个completion
     step均出现control-budget超限（最大`103.99 ms`），正常结束时worker为`ABANDONED_UNDRAINED`。随后只增加
     default-preserving、model-neutral的`rtc_request_validity_steps`和`rtc_worker_stop_timeout_sec`显式参数；50-step/2 s
     复跑中cold result仍唯一TTL拒绝，7个warm result接纳256个future keyframe并形成240个唯一formal emission，
     `send_actions=false`且零dispatch，close得到`DRAINED/thread_exited=true`。对原始900行JSONL按
     `completion_hash/request_transition_hash/emission_id`重新分类后修正了最初解释：7个不伴随新request的warm
     completion/admission均低于预算（median `12.20 ms`、max `13.68 ms`），234个不伴随新request的ordinary
     emission也均低于预算（median `7.82 ms`、max `15.91 ms`）；全部8个budget exceedance都发生在构造并提交
     三相机新request的tick（median `98.82 ms`、max `101.80 ms`），其中仅1个同时包含cold TTL completion。
	     根因位于control-thread request observation-cut路径：当前`_observation_cut_id()`对包含图像的完整
	     `PolicyObservationView.payload`使用通用canonical JSON/ndarray hex哈希，而不是50-keyframe completion merge。
	     随后的R-R3c本地无设备修复将`observation_cut_id`版本化为session/domain/producer epoch/request sequence/
	     observation contract/time/step的轻量因果身份，并将work-item hash收窄为同一cut与actor contract binding；
	     精确model payload仍由worker-side`model_input_hash`生成、后端核对并由response回显。observation firewall同时
	     保留immutable detach与字段级parity，但复用`image`/`images`alias且移除重复deep-thaw/uint8比较。三路
	     `720x1280x3 uint8`、horizon 50虚拟OpenPI回归中，absolute request tick median/max为`4.44/8.03 ms`，
	     relative为`11.70/12.26 ms`；completion+due emission max为`26.61 ms`，均低于30 Hz的`33.33 ms`预算。
	     同一源码经remote sync check后在`dex-remote`空闲机无设备复跑，absolute/relative request max为
	     `3.39/5.13 ms`、completion+due emission max为`21.96 ms`，同样零超限。该结果只证明本地/远端
	     无设备request path，不替代真实observation/transport gate。
	     R-R3c real no-action复跑已按当前主线修正两处部署接线：server launcher在exec后显式保留DexTeleop
	     `PYTHONPATH`；显式`--rtc-real-shadow`固定headless auto-start/no-ready-confirm并禁用本批未授权的DMTAC
	     tactile，ordinary production profile不变。复跑仍在RTC loop之前被当前left-wrist RealSense
	     `261122273924`拒绝`640x480@15 yuyv`阻塞；相同失败重复两次，三个frontend均close为
	     `DRAINED/thread_exited=true`且只有startup/close事件。配置文件自7月19日未变，但设备于成功R-R3b之后
	     重新枚举，因此该问题归属real camera/device profile，不得由RTC改相机或sim observation绕过。
	     因此仍不得进入action-send；下一批只需以当前实现复跑同一no-action real-observation gate并要求零budget
	     exceedance，但前提是相机owner恢复并验证left-wrist可用profile。证据见
	     `docs/architecture/evidence/rtc/rtc_rr3_openpi_relative_real_observation_no_action_evidence_v1.json`和
	     `docs/architecture/evidence/rtc/rtc_rr3b_openpi_relative_real_observation_admission_no_action_evidence_v1.json`以及
	     `docs/architecture/evidence/rtc/rtc_rr3c_local_request_path_regression_v1.json`、
	     `docs/architecture/evidence/rtc/rtc_rr3c_real_rerun_blocked_camera_v1.json`。
	     2026-07-23追加deploy-only相机启动编排：通用deploy CLI新增default-off的
	     `--camera-startup-hardware-reset`与`--camera-start-stagger-sec`；显式RTC real-shadow profile才固定启用
	     serial-targeted hardware reset，并与当前真机遥操一致地按camera config插入顺序逐台启动、相邻启动间隔
	     `0.5 s`。ordinary OpenPI profile、相机配置默认值和遥操/replay均不变。
	     `teleop_runtime/configs/devices/realsense.json`是本gate唯一的逻辑角色来源：
	     `front/left_wrist/right_wrist`只能绑定其中各自的librealsense serial，不按`/dev/video*`或V4L2展示的
	     `Serial`字段重映射。2026-07-24用远端`pyrealsense2`只读枚举确认三台设备精确为
	     `213222079798(D435)/261122273924(D405)/260522278055(D405)`，与权威配置完整匹配；此前依据V4L2字段作出的
	     “配置设备缺失”判断已废止。
	     随后按no-action边界尝试R-R3c gate，但两次均在camera start/reset之前被DMTAC assignment fail-closed：
	     第一次缺`/dev/ttyUSB0`，第二次缺`/dev/ttyUSB1`。只读核对进一步确认，shared `dmtac/latest.json`保存的
	     left/right `by-path`均不再匹配当前USB拓扑；两只CH340也没有可分别绑定的唯一USB serial，当前`by-id`只有
	     一个非唯一别名。因此不能由RTC按`ttyUSB`序号推断左右，现有assignment的fail-closed对依赖共享assignment的
	     production路径仍然合理。本批没有camera
	     hardware reset、RTC frontend step、transport exchange、formal emission或dispatch。不得通过禁用末端、
	     伪造hand state或重写assignment绕过。对于严格限定为`send_actions=false`、双侧observation-only且只验
	     raw-RTC时序/identity/admission/emission/drain的mechanics shadow，可以使用现有deploy CLI的显式manual port
	     override，并把左右物理身份未复核写入证据；该豁免不适用于policy side-semantics、canonical target或任何
	     motion gate。证据见
	     `docs/architecture/evidence/rtc/rtc_rr3d_real_rerun_blocked_dmtac_v1.json`。
	     随后以当前默认参数完成R-R4 deploy-only回归：真实双臂/双DM-Tac/三相机观测在600个30 Hz step内进入同一
	     OpenPI raw-RTC worker，完成19次request、18次response，10次merge接纳并形成9个唯一formal emission；
	     `request_deadline_expired` 1次、`no_eligible_future` 7次，control budget exceedance为0，且全程
	     `send_actions=false`、lookahead disabled、dispatch/action-step均为0。该默认profile的`request_validity_steps=10`
	     与`worker_stop_timeout_sec=0`保持兼容，但最后一个in-flight request使close为`ABANDONED_UNDRAINED`，所以本批只
	     证明真实输入→transport→admission/emission的shadow接线，不证明drained close或action readiness。证据见
	     `docs/architecture/evidence/rtc/rtc_rr4_openpi_raw_rtc_real_observation_no_action_default_profile_evidence_v1.json`；
	     下一步只需在仍然`send_actions=false`的边界下单独收敛raw-RTC drain/cancellation gate，不改变teleop/replay或
	     `sync_queue_v1`。
	     紧接着用显式`request_validity_steps=50`与`worker_stop_timeout_sec=2`完成R-R4 drain-gate复跑：900个30 Hz
	     step内8次request/response，7次merge接纳、1次cold TTL拒绝，273个keyframe接纳、253个唯一formal emission；
	     control tick最大`16.80 ms`、超预算为0，close为`DRAINED/thread_exited=true`，仍全程零dispatch/action-step。
	     这闭合了真实输入→transport→admission/emission→bounded worker close的shadow gate，但不提升canonical safety/IK、
	     target acceptance或motion authority。证据见
	     `docs/architecture/evidence/rtc/rtc_rr4_openpi_raw_rtc_real_observation_no_action_drain_gate_evidence_v1.json`。
	     同参数独立复跑再次得到900个30 Hz step、8次完成exchange、7次merge接纳、1次cold TTL拒绝；
	     290个admitted keyframe全部形成290个唯一formal emission，零重复、零budget exceedance，control tick
	     median/p95/max为`0.58/13.17/17.58 ms`，close仍为`DRAINED/thread_exited=true`。日志中的442个underrun
	     step分布在16段无可执行plan的窗口，未触发动作或隐藏lookahead；它们是本shadow下显式可观察的hold窗口，
	     不是side identity的证明。三路相机按配置顺序reset/start，腕部少量约`0.11–0.13 s`的stale提示均自恢复。
	     本次manual `ttyUSB0/1`绑定沿用共享遥操assignment中的历史side alias，但没有重新确认物理左右，因此只增强
	     raw-RTC mechanics的可重复性证据，不增加model input语义或motion readiness结论。
   - 本批OpenPI adapter/launcher定向回归`69 passed`，RTC/OpenPI无硬件扩展矩阵`246 passed, 1 skipped`，
     canonical/position/deploy兼容矩阵`83 passed`；远端实际absolute `92599`与relative h16 `150000` artifact已拉取到
     临时目录并由本地新config解析；R-R2 live证据见
     `docs/architecture/evidence/rtc/rtc_rr2_openpi_relative_live_no_action_evidence_v1.json`。
7. **R-S1 target-side sim integration**
   - 等待sim2real发布第7.0节observation contract后再开始；
   - 只接其immutable observation source，验证canonical pose→component safety→IK→arm target及typed
     end-effector target链；
   - 不修改sim observation实现，不走direct arm qpos shortcut，不让`sim`反向依赖`teleop_runtime`。
8. **R-M1 optional model-RTC**
   - 先完成MRTC-D1/D2/R1/C1，再分别验收absolute model-space prefix与relative new-anchor re-encode；
   - producer-local overlap只需要plan lineage、encode/decode round-trip和conditioning receipt，不要求target
     preview/downstream acceptance；
   - downstream-accepted hard prefix与position lookahead仍是独立D-LA gate，不进入首个model-RTC producer；
   - 该批失败不回退或降级R-A1/R-R1 raw-RTC。

R-A0/R-A0b/R-A0c/R-A0d的pure/local seam已冻结。进入真实native profile前仍须为具体gripper/hand确认units、bounds、
step/rate、discrete transition和measured-state可用性；不得填写猜测值。进入正式proposal/target caller前还须补typed
representation lineage消费与anchor lifecycle evidence。R-S1开始前唯一外部阻塞是sim2real observation contract；
不得由RTC侧临时补一套adapter绕过。

这一批不得修改：

- real dispatch/safety/SDK seam；
- M5.8-b/M6 owner状态；
- production entrypoint默认行为；
- OpenPI外部checkout；
- sim observation、MuJoCo→Real adapter或sim2real并行目录；
- teleop、collection或replay链；
- PolicyGraph/M7C；
- raw dataset磁盘schema、collection writer或lockfile（additive derived-view/pure contract按MRTC-D1/D2单独验收）。

## 16. 参考与证据

外部算法参考：

- Kevin Black, Manuel Y. Galliker, Sergey Levine, “Real-Time Execution of Action Chunking Flow Policies”,
  <https://arxiv.org/abs/2506.07339>；
- Physical Intelligence RTC project，<https://www.pi.website/research/real_time_chunking>；
- LeRobot RTC documentation，<https://huggingface.co/docs/lerobot/rtc>；
- LeRobot real-robot RTC example，
  <https://github.com/huggingface/lerobot/blob/main/examples/rtc/eval_with_real_robot.py>；
- Physical Intelligence simulated implementation，
  <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>；
- OpenPI remote inference文档，
  <https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md>；
- Zhaoyang Lyu等，“Training-Time Action Conditioning for Efficient Real-Time Chunking”，
  <https://arxiv.org/abs/2512.05964>；
- Min Ho等，“Learning Native Continuation for Action Chunking Flow Policies”，
  <https://arxiv.org/abs/2602.12978>；
- OpenPI issue #717，<https://github.com/Physical-Intelligence/openpi/issues/717>（仅作为同步handler可能阻塞的待测风险，
  不是已解决的官方capability或稳定保证）。

采用范围：论文/LeRobot用于 RTC guidance、delay、prefix和queue实现参考；training-time/native continuation工作只用于
protocol扩展方向，不是RTC-3c要求；OpenPI示例说明当前client获得整段action chunk，但不提供DexTeleop所需的
session/epoch/plan admission语义。DexTeleop的 action/frame/anchor、
causal time、receiver admission、Arbiter、safety、epoch、dispatch owner和trace语义始终以本项目 contract为准。

本计划形成前的当前工作树聚焦基线为：chunk emission、legacy queue policy、MuJoCo deploy/async、position executor、
OpenPI common/π0.5 deploy相关测试 `110 passed, 1 skipped`。该结果只证明当前无硬件基线可用，不证明 RTC 已实现。
