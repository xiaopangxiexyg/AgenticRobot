# DexTeleop RTC 当前落地边界

状态：活动 producer/协议计划；最后复核日期为 2026-08-17。

本文维护实时 Action Chunk（RTC）的当前 owner、时序/表示不变量、现有 profile 成熟度和剩余 gate。
原 RTC-0～RTC-7、R-R/MRTC 批次、历史真机 no-action 收据、测试数字和完整算法讨论已冻结在
[RTC 历史落地台账](../reports/rtc_landing_ledger_20260725.md)。历史 evidence 可以证明其记录时点的
边界，不能替代当前 transport、设备、server、artifact 或 action authority 复核。

RTC 是 policy producer feature，不是新的顶级产品或控制核心。普通 deploy owner 见
[`teleop_runtime/deploy/README.md`](../../../teleop_runtime/deploy/README.md)，共享 control/authority gate 见
[Control Core 当前边界](dexteleop_control_core_landing_plan.md)，下游多源装配见
[Control Station 当前计划](dexteleop_control_station_plan.md)。

## 1. 当前结论

RTC 的正式输出边界是一个已经 committed 的 `PolicyActionEmission`：

```text
causal observation cut
  → PolicyInferenceRequest
  → bounded policy worker / model transport
  → immutable PolicyChunkResult
  → control-thread validation and timed-plan merge
  → at most one due keyframe pop per decision cut
  → PolicyActionEmission
  → ordinary deploy or optional composition consumer
  → downstream safety / target / reference / dispatch
```

RTC 负责 request、worker、result、producer plan、merge、due-pop、emission commit、TTL、producer epoch/session
和 producer lifecycle。RTC 不负责 proposal/Control Station、canonical safety、IK、reference、target admission、
HOLD 的物理执行、dispatch 或设备 receipt。

当前 `python -m teleop_runtime.deploy` 仍默认 `--rtc-profile off`，走既有 `sync_queue_v1`。仓库中的三个
RTC application profile 都是显式、default-off、no-action profile：

```text
rtc_3c_virtual_shadow_15_30_60_120_v1
raw_rtc_final_chunk_15_30_60_120_v1
model_rtc_flow_inpaint_shadow_15_30_60_120_v1
```

这些 profile 禁用 position lookahead 并拒绝 `--send-actions`。因此当前 RTC 代码、OpenPI adapter、真实输入
no-action evidence 或 worker drain 都不构成 RTC motion authorization。

## 2. Raw RTC 与 model RTC

### 2.1 Raw RTC

Raw RTC 保持模型原有 `observation → final action chunk` 接口。客户端只做异步推理、结果校验、delay-aware
裁剪、timed-plan replacement 和唯一 due-pop；模型不消费 future action prefix。

Raw RTC 的 absolute/relative 只是 source representation 分支，进入 plan 前都必须成为 typed canonical
absolute action：

```text
absolute model action ─┐
                       ├─ canonicalization receipt → canonical absolute plan
fixed-anchor relative ─┘
```

Relative action 必须使用原 request observation cut 的 immutable anchor，不能在 result receive、merge 或 emission
时按新 measured pose 重新锚定。过期 row 只能裁掉，不能改变其几何含义。

### 2.2 Model RTC

Model RTC 允许模型消费 versioned future-prefix/mask 并返回 conditioning receipt。它与 raw RTC scheduler 正交，
不能通过“relative action”或“有 overlap”隐式启用。

首个 continuation 时间关系为：

```text
generation_start_at
  + forecast_delay_steps * action_period_sec
  = first_target_effective_at
```

runtime 与 model adapter 只能有一处执行 delay crop。producer-local overlap 不等于 downstream accepted prefix，
也不代表 target preview、safety/reference acceptance 或 motion authority。

## 3. 当前 package ownership

| 责任 | owner |
|---|---|
| producer cadence、timeline、plan overlap selection、epoch/plan lineage | `robot_motion.deploy.rtc` 与现有 flat compatibility modules |
| coordinator、bounded worker、timed plan、due-pop | `robot_motion.deploy.rtc_runtime`、`async_producer`、`chunk_plan`、`policy_worker` |
| application CLI、observation/device assembly、publisher lifecycle | `teleop_runtime.deploy` |
| model-neutral capability/envelope/conditioning/transport contract | `deployments.client` |
| OpenPI artifact-bound codec、metadata、transport adapter | `deployments.openpi.client` |
| continuation schema/training semantics | `dataset_contracts` |
| offline continuation projection/training view | `dataset_interface` |
| 非正式 learner、loss、training experiment | `experiments/policy_learning` 或独立模型工程 |
| target safety、IK、reference、dispatch | target runtime / `robot_motion`，不属于 RTC |

`teleop_runtime.deploy.rtc_frontend` 是 `robot_motion.deploy.rtc_runtime` 的 import-compatible re-export，不拥有
第二个 coordinator、worker 或 plan。现有 flat RTC modules 继续作为兼容 surface，不能一次性 bulk move 到子包。

## 4. Producer 时序不变量

### 4.1 单一 request 和 bounded carrier

- 同一 producer/session 最多一个有效 in-flight request；
- request mailbox capacity 固定有界，submit/poll 不阻塞 control step；
- blocking model/transport 只在 worker 中运行；
- completion 由 control thread 校验并 admission，worker callback 不直接 publish action；
- worker cancel accepted、thread exited 和 execution drain 是不同 receipt；stop timeout 不得冒充 drained。

### 4.2 必须保留的时间

```text
request_triggered_at
inference_started_at
inference_completed_at
result_received_at
result_admitted_at
observation.cutoff_at
keyframe.target_effective_at
keyframe.emitted_at
```

调度真值使用 local monotonic clock和显式 clock-domain ID。Wall time只用于日志关联。非整数频率和 variable
`dt` 通过 target time 处理，不能使用 frame modulo。

Delay 至少区分：

```text
d_forecast_steps   # request-time budget/guidance
d_elapsed_steps    # receive relative to observation cut
s_consumed_steps   # actual producer cursor movement
```

merge eligibility 以实际 plan cursor、target time、epoch 和 committed boundary 为准，不能用一个四舍五入的 `d`
覆盖 pause、jitter、missed tick 或 staging 差异。

### 4.3 Plan 与 emission

- executed/committed prefix immutable，新 result 只替换 mutable future suffix；
- old request、duplicate、old worker session 或 producer epoch mismatch fail closed；
- 每个 decision cut最多选择一个仍有效的 due keyframe，不补发过期 backlog；
- due-pop 成功后先形成正式 `PolicyActionEmission`，再推进 producer committed-through；
- emission commit 只证明 producer action 已固化，不证明 safety、reference、target 或 backend接纳。

HOLD/FAULT/transition 必须先使 producer epoch/plan 失效，再处理 worker retirement/rearm。late completion 保留为
可审计 rejection，不能删除后宣称没有竞态。

## 5. Action 和 component representation

Plan 的唯一动作真值是：

- legacy normalized absolute row；或
- typed canonical `RobotAction` 与其 canonicalization receipt。

同一个非空 plan 不能混装两类 payload，也不能同时保存会漂移的 relative/absolute 执行副本。Arm pose 与 typed
end-effector component分别绑定 frame/layout/normalization/component identity；qpos、continuous gripper 和 discrete
token 不参与 SE(3) 重锚定。

因果顺序保持：

```text
pop → emission → proposal/admission → canonical safety
    → safe action → resolver/reference/backend → downstream receipt
```

RTC profile 当前没有 future lookahead authority。任何 future preview/promotion 必须是下游 D-LA gate，包含 exact
identity、TTL、canonical safety、reference admission 和 rollback；不得把 current emission commit 扩大解释为 future
target acceptance。

## 6. 当前已具备的边界

- target-neutral RTC coordinator 和 profile 投影位于 `robot_motion.deploy`；
- default-off deploy CLI 可选择 virtual、OpenPI final-chunk raw RTC 和 model-RTC shadow profile；
- OpenPI final-chunk adapter 与 model-neutral client/protocol分层，provider adapter不拥有 worker/plan/dispatch；
- continuation schema、training view、prepared OpenPI model-RTC profile 和 runtime conditioning modules 已分 owner落位；
- `RtcPolicyEmissionPublisher` 只发布 committed emission，支持显式 suspend、单 required subscriber ACK 和原子
  rebind/resume；旧 passive/debug fanout 保持兼容；
- producer epoch binding 和 stable descriptor 已分离，旧 consumer 在 rearm 后 fail closed；
- bounded debug observer 只记录 identity/hash/timing，queue full 不反压 producer。

这些是代码/contract 边界，不自动证明 current server、artifact、real observation、transport session 或 hardware
状态。历史 no-action evidence 已移入 report/evidence，新的运行结论必须绑定新日期和 exact artifact/profile。

## 7. 剩余 gate

### R-B0：compatibility 与默认行为

- 固定 `sync_queue_v1` hash、queue pop、last-action、lookahead 和 CLI 默认行为；
- 保持 legacy import alias 和 source-binding hash domain；
- RTC 开关前后的普通 deploy dry-run/`--help` 行为一致；
- RTC profile继续 fail closed拒绝 `--send-actions`。

### R-B1：cross-rearm binding rotation

当前单 required subscriber 的 suspend/ACK/rebind/resume seam 已存在。production fanout 前仍需明确：

- application 如何触发并等待旧 binding quiescence；
- timeout、consumer restart 和 late ACK 的状态机；
- 多 required subscriber 的 deterministic ordered ACK/rollback；
- 新 binding 在新 epoch 第一条 emission 前可见；
- Station/consumer 不自行递增 epoch、复制 config 或猜 binding hash。

### R-B2：普通 deploy 与 composition consumer

- ordinary deploy 只消费同一 `PolicyActionEmission`，不读取 `RtcFrontendStep` 或 timed plan；
- composition contributor adapter位于 emission 之后，保留 source lineage，不把 worker/model字段带入
  `control_composition`；
- publisher/consumer failure保持 producer 与 target owner可审计，不能产生第二 action queue。

### R-B3：real observation/transport no-action refresh

在 exact host、camera/hand mapping、artifact、server metadata和 profile上复跑时，必须证明：

- `send_actions=false`、零 target/dispatch/action-step；
- request/result/admission/emission唯一性与 control budget；
- worker close 的 typed drain/undrained结论；
- transport/server失败不启动 backend，也不以旧 evidence冒充当前在线。

该 gate 仍是 no-action；通过后也不能进入 action-send。

### MRTC：model continuation

- training/data contract、artifact capability和runtime conditioning identity精确一致；
- model/provider返回可验证 consumption receipt，不把普通 final chunk标成 inpaint；
- prefix encode/decode round-trip、mask、forecast range、bootstrap和late crop可重放；
- raw RTC仍可独立运行，model-RTC失败不降低raw contract或隐式fallback同步推理。

### D-LA / RTC action authority（下游独立安全批次）

Future preview、target admission、canonical safety、reference、dispatch、HOLD、real worker drain和motion permit均由
下游部署/Control Core owner验收。Human/replay/HITL/Control Station 不成为 RTC producer gate，RTC no-action证据也
不继承普通 sync deploy 的 motion authorization。

## 8. 每批验证

在已激活的 `${DEXTELEOP_ENV_NAME:-dexteleop310}` 环境中选择相关测试：

```bash
python scripts/check_docs_navigation.py
python -m pytest -q tests/test_policy_chunk_plan.py
python -m pytest -q tests/test_deploy_rtc_policy_worker.py
python -m pytest -q tests/test_deploy_rtc_producer_state.py
python -m pytest -q tests/test_deploy_rtc_frontend_shadow.py
python -m pytest -q tests/test_rtc_package_architecture.py
python -m pytest -q tests/test_openpi_rtc_final_chunk_adapter.py
python -m pytest -q tests/test_model_continuation_data_contract.py
python -m teleop_runtime.deploy --help
git diff --check
```

按变更范围补充 OpenPI transport、model continuation、real-shadow no-action 或 downstream target测试。只有明确的
真实运行任务才使用远端/设备 wrapper；本计划的文档与离线验证不启动 server、camera、robot、gripper 或 motion。

## 9. 当前优先顺序

1. 锁住 R-B0 default-off/no-action 和 legacy hash/import parity；
2. 完成 R-B1 application-level cross-rearm binding coordination；
3. 让 ordinary deploy/composition只消费 committed `PolicyActionEmission`，不泄漏 producer内部对象；
4. 需要现场证据时先刷新 R-B3 no-action，随后另立 D-LA/real authority安全任务；
5. model-RTC按独立 MRTC gate演进，不阻塞 raw final-chunk RTC，也不反向改变默认 sync deploy。
