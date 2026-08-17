# HITL 与 Chunk-Aware Relative Control 当前边界

状态：活动边界与后续 gate；最后复核日期为 2026-08-17。

本文只维护 Human-in-the-Loop（HITL/HIL）的当前 ownership、公开 Sim-HIL profile、
chunk-aware trajectory seam 和尚未完成的 real/application gate。原 R0-R4、R3.1a-c、
算法推导、阶段测试数字和早期工作树记录已冻结在
[HITL 历史实施台账](../reports/hitl_chunk_relative_ledger_20260807.md)。历史台账证明其
记录时点的设计与收据，不能替代当前 profile、设备、模型、target 或 motion authority
复核。

当前 Sim-HIL 操作方法见 [HIL 使用说明](../../HIL/README.md)；Control Station 的 GUI
与 application 边界见 [Control Station 当前计划](dexteleop_control_station_plan.md)；
proposal、authority、safety、target、reference 与 dispatch 边界见
[Control Core 当前边界](dexteleop_control_core_landing_plan.md)。本文不调整 Skill、
`agent_gateway` 或上层 RPent。

## 1. 当前结论

HITL 不是一个新的控制核心。当前必须分开理解三个正交问题：

1. **应用生命周期与来源切换**：谁接纳 operator intent，何时 HOLD，何时捕获 Home，
   如何建立新 activation 并等待 fresh emission；
2. **多来源 composition**：同一 causal cut 上 Policy/Human 各 component 如何选择、
   相对叠加或切换 owner；
3. **未来 trajectory editing**：source-owned timed plan 如何形成 locked base、preview、
   residual-session、rebase、future handback 或 base refresh。

当前公开 Sim-HIL 已完成前两类的 MuJoCo-only application profile。第三类已有 reusable
carrier、reducer、preview、rebase 和一个薄的 MuJoCo proposal-publication seam，但尚未成为
`python -m control_station sim-hil` 的正式 trajectory profile。

```text
operator/device event
  -> Control Station typed intent and application lifecycle
  -> source-owned PolicyActionEmission / HumanCommandEmission
  -> control_composition same-cut evaluation
  -> one composed candidate
  -> proposal / authority / target / safety / IK / reference / dispatch

source-owned timed plan
  -> ComponentActionTrajectory
  -> control_composition.trajectory
  -> preview/shadow or explicit proposal-publication adapter
  -> the same downstream chain when a future profile selects it
```

无论哪条路径，HITL 都不新增第二个 policy queue、human device loop、authority、canonical
safety、IK、dispatcher 或 episode owner。

## 2. 当前公开 Sim-HIL profile

入口 `python -m control_station sim-hil` 默认不启动执行，必须显式传入 `--enable`。
`--control-profile` 当前只选择一个完整的 application-session/control-cycle pair，不能在
同一个 activation 内热换算法。

| profile | 选择方式 | 当前语义 |
|---|---|---|
| `fixed_alpha_compatibility` | 默认 profile | 保留既有 Policy/Human fixed-alpha 数值与生命周期兼容行为 |
| `exclusive_truncated_hitl` | 显式选择 | 全 component 原子选择 Policy-only `base_only` 或 Human-only `absolute_overlay_owner` |
| `policy_human_allocation_scheme_a` | 显式选择 | Policy baseline 与 Human `relative_overlay`；支持显式/自动 allocation 与手动 rebase-and-clear |
| `policy_human_allocation_scheme_b` | 显式选择 | 在 Scheme-A 基础上允许 `absolute_overlay_owner`，并带 target-side handoff continuity |

这四个 profile 共用 source、composition、proposal 和 MuJoCo 下游，不按模型、设备或
checkpoint 复制实现。Scheme-A/B 的 streaming allocation 是当前 action emission 的
same-cut/per-target composition，不等同于未来整段 `ComponentActionTrajectory` 的 locked-base
编辑。

### 2.1 Exclusive truncated HITL

截断切换是 application lifecycle transaction，不是另一套 composer：

```text
Policy-only -> HOLD -> attach/prime Human -> capture Runtime Home
            -> new activation -> fresh Human emission -> Human-only

Human-only  -> HOLD -> invalidate old policy plan/chunk -> reset Policy
            -> new activation -> fresh Policy emission -> Policy-only
```

`control_station.hitl` 只验证 HOLD、source selection、activation、fresh emission 和下游
receipt。具体 MuJoCo effect 由 `control_station.sim_hil_profile.exclusive_hitl` 提供。失败
保持 lower HOLD；ControlTrace observer 只能记录结果，不能改变事务结论。

### 2.2 Scheme-A 与 Scheme-B

Scheme-A/B 通过 component-keyed allocation context 选择已声明的 role：

```text
base_only
relative_overlay
absolute_overlay_owner  # Scheme-B only
```

`relative_overlay` 是相对 active anchor 的 state-valued correction，不按 control tick
积分。`absolute_overlay_owner` 仍只是 composition 中的 component owner 选择，不创建新的
RobotMotion authority。缺失、过期或不满足 evidence window 的 Human 输入不能被解释成
“Human 静止”；rebase-and-clear 必须经过 typed intent、fresh sample、anchor receipt 和
下一未提交 target boundary。

Scheme-B 从 absolute owner 回到 Policy 的 target continuity 已在当前 per-cut allocation
profile 中存在；它不能被扩大解释为 chunk trajectory 的 nominal-base handback 已经完成。

## 3. Package ownership

| 责任 | 当前 owner |
|---|---|
| operator intent、可重建 application state、profile/lifecycle coordination | `control_station` |
| GUI、浏览器生命周期、页面资源、operator interaction | 目标统一为 `control_station.gui` 子集 |
| backend/model/topology-neutral exclusive switch contract | `control_station.hitl` |
| Policy/Human emission 的 application source adapter | `control_station.sources` |
| MuJoCo allocation、trajectory publication、rebase 等具体应用绑定 | `control_station.composition` |
| Sim-HIL session/control-cycle/profile assembly | `control_station.sim_hil_profile` |
| 多来源 causal cut、allocation、relative math、trajectory reducer/preview | `control_composition` |
| canonical action/trajectory carrier 与下游 proposal/execution contract | `robot_motion.action` |
| policy worker、chunk queue/cursor 与正式 emission | 对应 policy/deploy source owner |
| human device production、retarget、deadman/recenter readiness | lower teleop/device owner |
| MuJoCo target、IK、reference、safety 与 dispatch | `sim.mujoco.deploy` 和 `robot_motion` lower owner |
| ControlTrace capture/seal/readback | application trace/capture owner；永远是被动 evidence |

现有 `control_station/sim_hil_gui.py` 与 `control_station/web/` 是 GUI 迁移期间的兼容路径。
后续不得再向这些散落模块增加控制算法、source state、HOLD/re-arm 状态机或浏览器外的
资源 owner；GUI/browser/operator 代码最终归入 `control_station.gui`，迁移顺序由 Control
Station 当前计划统一管理。

`control_station` 可以构造 lower implementations 并通过 typed port 连接它们，但不能复制
`control_composition` 的 reducer、policy/human producer、target safety、dispatch 或 episode
commit。lower package 也不能反向导入 `control_station`。

## 4. Chunk-aware trajectory seam

### 4.1 已存在的 reusable core

- `robot_motion.action.trajectory.ComponentActionTrajectory` 是 component-keyed、不可变、
  model/backend-neutral 的 timed carrier；
- `source_immutable_through` 只表示 producer 不再改写的范围，不表示 proposal、target、
  Reference 或 execution 已接纳；
- `control_composition.trajectory` 已拥有 pass-through/residual-active reducer、
  `ResidualSessionActivationBinding`、`NominalBaseBinding`、active/shadow trajectory、
  anchor revision、rebase transaction、transition assessment、continuity correction、
  intent/corrected preview、freshness 与 exact-target preparation；
- relative translation 和 joint-position/qpos temporal warp 已有；未声明的 representation
  和未实现的 full-pose rotation warp 保持 fail closed；
- preview、shadow 与 difference 都是 non-authoritative，不得冒充 future Reference 或
  executable target。

### 4.2 当前 Station seam

`control_station.composition` 已提供 default-off 的：

- trajectory intent/corrected preview controller；
- trajectory composition shadow controller；
- residual rebase command port；
- `MujocoTrajectoryPublicationController`。

这些 controller 只做 application gate、typed intent/port 翻译或一次 exact proposal
publication。`MujocoTrajectoryPublicationController` 不调用 lower `runtime.step()`；target、
reference 与 dispatch 仍由既有 MuJoCo loop 在自己的边界完成。

当前公开 Sim-HIL policy source 即使消费 action chunk，也通过 source-owned worker/queue 逐步
形成正式 `PolicyActionEmission`，再进入 streaming allocation。它没有把内部 queue 暴露给
trajectory reducer，也没有选择上述 trajectory publication controller。因此不能把
“OpenPI action chunk 可用于 Sim-HIL”写成“chunk-aware locked-base trajectory profile 已上线”。

## 5. 必须保持的因果与安全不变量

### 5.1 单一 owner

- policy/human source production 与 cursor 由各自 lower owner 推进；
- Control Station 只消费 emission/receipt，不在 GUI 或 profile 中重新推理、轮询或重建
  source queue；
- composition 每个 target 只捕获一个 receiver cut，并输出一个 candidate；
- proposal、authority、target admission、reference、command 和 dispatch 各有独立 owner 与
  receipt，不能压成一个 `committed_until`；
- trace、status、preview 和 GUI cache 只能观察，不得反向驱动控制。

### 5.2 Session 与 anchor

- `ResidualSessionActivationBinding` 保持一次连续交互的初始 activation/cut/base；
- session 内 rebase 使用新的 immutable active-anchor revision，不能静默替换旧 receipt；
- pending rebase 从声明的未提交 target boundary 起 fence 旧 anchor；新 fresh receipt 到达前
  返回 typed `no_action`；
- rebase 只清除 relative residual，不自动改变 authority 或 absolute owner；
- 已发布 target 不重写，物理连续性由 downstream owner 处理。

### 5.3 Default-off 与真实设备

- `control_station sim-hil` 未传 `--enable` 时不启动 source、worker 或 MuJoCo execution；
- exclusive、Scheme-A/B 必须显式选择，不能改变 default fixed-alpha compatibility；
- 当前 profile 可以读取真实 VIVE/Q6/angle-trigger，但执行 target 仍是 MuJoCo；
- 仓库当前没有 active real-deploy HITL lifecycle binding；被动 policy/human observer 不是
  composition consumer，也不是 motion authorization；
- 任何 real shadow 或 real action gate 都必须另行绑定 exact device、target、safety、
  deadman/recenter、HOLD、worker drain 和 dispatch owner。

## 6. 当前已完成与未完成

### 已完成

- 四类显式 Sim-HIL application profile 及原子 lifecycle/control-cycle 选择；
- MuJoCo-only Policy/Human source、component composition、proposal 与唯一 lower execution；
- exclusive Policy-only -> Human-only -> Policy-only 的 typed transaction 和虚拟设备闭环；
- Scheme-A/B allocation、manual/automatic mode selection、rebase-and-clear 与 Scheme-B target
  continuity；
- ControlTrace core WAL/seal/commit/readback 接入当前 component composition profile；
- reusable trajectory carrier/reducer/preview/rebase 与薄 MuJoCo proposal-publication contract；
- default-off/no-action shadow、preview 和 passive observer seams。

### 未完成

- `control_station.gui` 的目录收口；现有 `sim_hil_gui.py`/`web` 仍是明确迁移债务；
- real deploy 的 concrete exclusive/continuous HITL lifecycle 与 target binding；
- 把 source-owned `ComponentActionTrajectory`、trajectory session controller、operator
  clutch/deadman/release 和 durable trace 组装成公开 Sim-HIL trajectory profile；
- chunk trajectory 的 release handback、active-session base refresh 与对应 future-boundary
  Reference/Measurement revalidation；
- detector-driven automatic rebase、partial-component rebase、rotation temporal warp 和更多
  representation/topology；
- 真实 VIVE/trigger 行为、模型 transport、real observation、hardware target 与 motion
  evidence 的当前复核。

## 7. 后续 gate

### H-B0：兼容性与命名

- 默认 profile、CLI 选择、source emission 顺序与 fixed-alpha 数值保持不变；
- exclusive、Scheme-A/B 与 trajectory editing 使用不同名词和测试，不再互相借用完成状态；
- legacy CS-6/replay 与旧研究备忘继续明确标历史，不成为当前启动说明。

### H-B1：GUI/browser 收口

- 将 Sim-HIL 页面、浏览器开启/关闭、server handle 和 operator view 移入
  `control_station.gui`；
- `sim_hil.py` 只装配 GUI port 与 application intent，不保留 HTML/JS、浏览器或页面状态；
- 迁移保持 CLI、DOM/API compatibility，并由 Control Station 边界测试约束；
- 本 gate 不改变 composition、HITL transaction、target 或 device semantics。

### H-B2：公开 trajectory Sim-HIL profile

- source owner 显式输出 immutable `ComponentActionTrajectory`，不暴露 raw queue/cursor；
- application 把 typed clutch/deadman/release 与 residual session、publication barrier、
  episode lifecycle 和 trace profile 绑定；
- 每次只发布一个 exact due target，proposal receipt 后才推进 trajectory emission barrier；
- pause、HOLD、re-arm、late completion、support expiry 和 teardown 不能复活旧 state；
- profile 先 virtual/no-action，再 MuJoCo；不直接进入 real motion。

### H-B3：trajectory handback/base refresh

- handback 与 base refresh 保持两个事务；
- future switch 必须绑定 compatible base、fresh Reference/Measurement、tracking validity、
  assessment/revalidation 和 continuity correction；
- 不实现任意 newest-shadow replacement，不把 corrected preview 当 executable Reference；
- support expiry 返回 no_action/request，不隐式升级 Human authority。

### H-B4：real binding

- 复用现有单一下游 deploy owner，不同时运行旧 teleop loop 与 policy loop；
- concrete port 必须证明 HOLD/Home/source reset/fresh emission/target admission 与 worker drain；
- 先做 passive/no-action shadow，再单独审查 canonical safety、deadman、recenter、fallback、
  dispatch 和 motion permit；
- 本文及历史 Sim-HIL 测试都不授权真机动作。

## 8. 验证建议

在已激活的 `${DEXTELEOP_ENV_NAME:-dexteleop310}` 环境中按变更范围选择：

```bash
python scripts/check_docs_navigation.py
python -m pytest -q tests/test_control_station_hitl_contracts.py
python -m pytest -q tests/test_control_station_sim_hil_profile_selection.py
python -m pytest -q tests/test_control_station_sim_hil_allocation_profile.py
python -m pytest -q tests/test_control_station_exclusive_hitl.py
python -m pytest -q tests/test_component_action_trajectory.py
python -m pytest -q tests/test_trajectory_editing_reducer.py
python -m pytest -q tests/test_trajectory_session_lifecycle.py
python -m control_station sim-hil --help
git diff --check
```

涉及 GUI 时补充 `tests/test_control_station_sim_hil_gui.py` 和 browser/server lifecycle 测试；
涉及 active Sim-HIL 时补充 allocation control-cycle、proposal、MuJoCo target 和 ControlTrace
矩阵。只有用户明确授权真实设备或 motion 的任务才能运行对应现场验证。

## 9. 当前优先顺序

1. 保持四类当前 Sim-HIL profile 与 trajectory editing 术语、完成状态分离；
2. 按 Control Station 计划完成 GUI/browser/operator subset 收口；
3. 需要 chunk-aware continuous trajectory 时，先完成 H-B2 的公开 Sim-only profile；
4. 再实现 H-B3 future handback/base refresh，不让 preview 越权；
5. real HITL 作为独立 H-B4 安全批次，从 no-action evidence 开始。
