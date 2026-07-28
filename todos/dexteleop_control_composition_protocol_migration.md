# Control Composition 协议迁移与后续计划

更新日期：2026-07-28

本文记录当前开发阶段的破坏性协议调整。它描述代码现状和迁移边界，不是稳定版
兼容承诺。

## 1. 已冻结的边界

```text
Action value
  -> SourceActionSample
  -> CompositionEvaluation(composed_candidate)
  -> proposal admission
  -> downstream target/execution capability
```

`control_composition` 只负责同一 causal cut 上的来源选择、数值组合、候选结果、差异和
lineage。它不产生 `SafeAction`，不运行 IK、平滑、限幅、reference generation 或
dispatch，也不声明 motion safety。

输入边界默认保留有限的原始数值。未来可增加 default-off
`CandidateActionSpaceCheck` 作为静态动作空间早期拒绝，但该检查不能改写候选或替代
下游执行安全。

## 2. 本轮破坏性迁移

旧名称已从活动 Python caller 中移除，不提供兼容 alias：

| 旧协议 | 新协议 |
|---|---|
| `CompositionResult` | `CompositionEvaluation` |
| `composition_result` | `composition_evaluation` |
| `composition_result_id/hash` | `composition_evaluation_id/hash` |
| `composed_action` | `composed_candidate` |
| `composed_action_hash` | `composed_candidate_hash` |
| `dexteleop.composition_result.*` | `dexteleop.composition_evaluation.*` |

因此新 evaluation 的 ID/hash 与旧 result 不兼容。仓库内的
`control_composition`、`control_station`、`dataset_interface`、`sim`、
`teleop_runtime` 和测试 caller 已同步迁移。历史 artifact 若需要读取，应在其
所属 reader/dataset migration 中显式投影；不得把旧字段重新加入新运行时协议。

## 3. Reference 与 measurement 观测

正式通用契约位于
`robot_motion.action.execution_observation`：

- `ExecutionObservationPlan`：default-off 的下游观测需求，支持
  reference 的 `off/decimated_frames/full_frames` 以及独立的
  measurement `off/decimated_frames/full_frames`；
- `ExecutionObservationCapabilitySpec/Binding`：target 构建时对 non-off plan
  整体接纳；不支持的 mode/component 必须构建失败，不允许 silent degrade；
- `ExecutionObservationWatermark`：以 stream progress 暴露 sink rejection/drop，
  而不是为每一帧再增加一层重复 receipt；
- `ExecutionReferenceFrameObservation`：由 reference owner 从真实下游对象导出的
  component-keyed immutable frame；
- `ExecutionMeasurementObservation`：backend measurement 的 component-keyed 投影；
- `ExecutionFactReference`：下游父事实的 opaque identity/hash/time。

request/submission/completion/admission/activation 生命周期继续只由
`TargetExecutionStatusSnapshot` 及其原始 receipt 表达，不在 observation sink
内重复发布。

Reference 必须带 `reference_stage_id`，并明确时间语义：

- `point_sample`：未绑定的普通点值不声称处于激活状态；由同一个下游
  observation binding 在 measurement cut 产生的 point sample 则是 owner 明确
  配对的实际 reference；
- `active_interval`：owner 明确声明实际半开区间
  `[effective_from,effective_until)`，才允许计算 actual reference-to-measurement
  difference。

`control_composition.observability` 不再从 reference 列表中选择“最新值”，也不重建
插值 timeline。调用者必须提供唯一的下游配对 frame；通用层只验证
control domain、selection group、clock、stage、有效区间、schema 和 component，
然后计算已声明 metric。

MuJoCo adapter 已将内部 `execution_epoch/group_segment_generation` 映射为公共
`execution_fence_sequence/reference_revision`。保留的 interval projector 服务
已有离线/显式终止事实；新的 Sim-HIL observation binding 在 measurement cut
发布 `mujoco.active_qref_at_measurement + point_sample`，避免后续 supersession
反向修改旧 interval。

现有真机 position observer 位于 resampler 后、resolver/planner 前。其薄投影保留
同一个 `ResampledMotionReference` 的精确值，但标为
`position.pre_resolver_reference + point_sample`；它不能冒充 backend active qref，
也不能直接形成 actual execution deviation。

## 4. 仍未完成

1. **其他 backend 的真实 active-reference/command receipt**：MuJoCo Sim-HIL
   已具备 build-time optional、绑定后 required 的 qref/measurement 发布 seam；
   其他 MuJoCo/真机 owner 仍需提供准确 interval 或同周期
   reference/measurement pairing，不得由 `control_composition` 猜测。
2. **ExecutionTarget production migration**：薄协议现已分别表达 authoritative/
   adaptation/candidate identity、requested effective time、target-private processing
   facts、resolved/scheduled/admitted/available time与 clock domain；accepted
   submission 还必须以 `TargetExecutionCompletionReceipt` 终止为
   resolved/rejected/failed/cancelled。target capability 需声明 activation 是
   `required` 还是 `omitted`；required 时由 owner 发布独立 activation receipt。
   MuJoCo Sim-HIL 已由实际 scheduled-reference owner 接入，并发布 exact
   request/submission/completion、point target、target admission 与原子
   activation；target 只描述目标时刻的 endpoint，连续插值仍是独立 Reference。
   其他 MuJoCo/真机 owner 尚需分别迁移；协议不会代替 owner 生成这些时间。
3. **legacy 数值算法边缘**：composition input、formal strategy decision、
   `CompositionEvaluation.composed_candidate`、promotion 和
   `ActionProposal` 现均原生使用 `ComponentAction`。正式 pose/qpos component
   strategy 直接读取 payload schema，不再解析 `left/right arm/hand` 命名。
   旧 deterministic-v1 算法只在其明确 compatibility implementation 内短暂物化
   `RobotAction`，随后立即投影回同一个 runtime action schema；新来源不得依赖这条
   边缘。
4. **可选 action-space check**：尚未实现，保持 default pass-through；不应在其完成
   前新增任何隐式 clamp、HOLD 或 safety verdict。

完整 activation action schema 已进入 `ControlExecutionManifest` 和
`execution_manifest` ControlTrace v2 记录；完整性检查同时核对 manifest 与
arbitration-input 的 `action_schema_hash`。保留的 `component_catalog_hash`
只服务于现有 v1 target/reference bridge，不再承担 generic carrier identity。

下游结构接纳使用 `ComponentTargetCapabilitySpec`，以 exact active component
sets 声明共享 resource 的合法组合，并由 `TargetCapabilityAssessment` 返回纯
accepted/rejected 结果。它不属于 `control_composition`，也不会自行触发 HOLD、
安全检查或 dispatch。方案 B 已将 typed representation adaptation 接入唯一
production seam：authoritative publication 先固定 Arbiter 决策，之后产生
`ActionAdaptationRequest`、assessment、non-safe `AdaptedActionCandidate` 与
`ActionAdaptationReceipt`，再形成 `ExecutionDirective` 并进入既有 canonical
safety/resolver/target 链路。适配拒绝不会调用这些下游阶段，也不会自行签发 HOLD。

现有 MuJoCo pose/qpos profile 在该 lower compatibility edge 精确物化
`RobotAction v1`；policy、RTC、replay、human 和设备来源不需要知道该 adapter，
因此这一接线不依赖模型结构、checkpoint、sim 或遥操设备。带准确 requested/
processing/resolved/scheduled/admitted/available 时间及 receiver identity 的薄
协议已由 `TargetExecutionRequest`、submission/completion receipt、
`ComponentExecutionTarget`、独立 admission/activation receipt 表达，不能与
`ActionAdaptationReceipt` 混用。Sim-HIL 已不再使用
`ResolvedExecutionTarget`；旧 carrier 仍是其他 MuJoCo profile 的
SafeAction/RobotAction-v1 compatibility 边缘，只有 caller 清零后才可归档。
真实 VIVE 仍只是 human source，未因本次下游迁移获得任何真机 motion authority。

当前 temporal policy 已实现并进入 snapshot/ControlTrace：
`CompositionTemporalSelectionSpec` 声明 same-target interval selection、overlap
behavior、target lead/lag。旧 `RobotAction.timestamp` 仅是 compatibility
lineage，不参与 selection，也不再用于生成 proposal target time。覆盖同一 role 的
事实按 `effective_from -> receiver admission sequence -> fact ID` 排序；runtime
profile 的 lead/lag 由 retention window 推导。

`CompositionInputSnapshot` 与 `CompositionEvaluation` 现在显式携带
`action_schema_hash`、`target_effective_at` 和 `target_clock_domain_id`。
composer proposal 另外携带独立的 `composition_evaluation_id/hash`；它不再冒充
`policy_output_id`。authoritative publication 同样自包含 arbitration result hash、
component action/schema、target time/clock 和 selected proposal mapping，因此后续
adaptation 不再回读 computation side object 来补语义。

Context binding 已分成两种明确模式：

- `same_cut_global`（默认）：context 与 action 共享 receiver `ControlCut`，但不要求
  每个来源声明它为祖先，适用于 task/vision/state/advisory；
- `selected_input_ancestor`：仅对声明的 `bound_input_roles`（空表示所有已选 role）
  验证 context ID 出现在对应 input lineage 中。

这消除了“全局视觉上下文必须复制到每个 human/policy 输入”的错误约束。

## 5. 验证边界

本轮只进行纯 contract、adapter 和 sim 测试；没有启动真机、遥操设备、policy
worker、replay、servo 或 dispatch。新增 observer/adapter 均为 passive/default-off，
未接入 production control loop。
