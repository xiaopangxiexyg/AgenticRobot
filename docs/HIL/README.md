# Human-in-the-Loop 文档入口

本目录描述 DexTeleop 当前的 Human-in-the-Loop（HIL/HITL）边界和 Sim-HIL 操作。
当前可运行目标是显式启用的 MuJoCo application；真实 VIVE、Q6 或 angle-trigger 可以
作为 Human 输入，但不会因此构造真实 RM65 backend。仓库当前没有 active real-deploy
HITL binding，也没有由本目录授权的真机 motion 流程。

## 从哪里开始

| 目的 | 文档 |
|---|---|
| 理解 current ownership、四类 profile、trajectory 与 real gate | [HITL 当前边界](../architecture/plans/hitl_chunk_relative_migration_plan.md) |
| 启动、操作、排障、capture、配置输入设备或 policy source | [Sim-HIL 综合操作指南](./guides/sim_hil_operator_guide.md) |
| 追溯 R0-R4、R3.1a-c、ResidualSession/preview/rebase/handback 设计 | [HITL 历史实施台账](../architecture/reports/hitl_chunk_relative_ledger_20260807.md) |
| 追溯 Sim-HIL VIVE/Q6 wrapper 退役原因和保留边界 | [Sim-HIL VIVE/Q6 wrapper 退役报告](../legacy/reports/sim_hil_vive_q6_wrapper_retirement_20260825.md) |
| 理解 GUI、browser、application profile 的收口方向 | [Control Station 当前计划](../architecture/plans/dexteleop_control_station_plan.md) |
| 理解 proposal、authority、safety、target、reference 和 dispatch | [Control Core 当前边界](../architecture/plans/dexteleop_control_core_landing_plan.md) |

综合操作指南是当前命令与排障事实源。历史台账中的命令、测试数字和“当前”结论只对应
原记录时点，不能代替新的 `--help`、artifact、设备或 runtime preflight。

## 当前运行边界

唯一 HIL 应用入口是：

```bash
python -m control_station sim-hil
```

入口本身 default-off。只有显式 `--enable` 才启动 source、worker 和 MuJoCo execution；
`--dry-run` 只构造并验证 policy/task/calibration/target contract，不启动设备、authority 或
control loop。

当前 profile：

| `--control-profile` | 语义 | 选择方式 |
|---|---|---|
| `fixed_alpha_compatibility` | 既有 Policy/Human fixed-alpha compatibility | 默认 |
| `exclusive_truncated_hitl` | Policy-only 与 Human-only 的事务式截断切换 | 显式 |
| `policy_human_allocation_scheme_a` | Policy baseline + Human relative overlay + allocation/rebase | 显式 |
| `policy_human_allocation_scheme_b` | Scheme-A + Human absolute owner + target continuity | 显式 |

profile 绑定 application session 与 control-cycle port，不能在同一个 activation 内热换。
`exclusive_truncated_hitl` 的 0/1 来源切换、Scheme-A/B 的连续 allocation，以及尚未公开
接入的 chunk-aware trajectory editing 是三个不同边界，不能借用彼此的完成状态。

## 安全 preflight

在已激活的 `${DEXTELEOP_ENV_NAME:-dexteleop310}` 环境和仓库根目录执行：

```bash
python -m control_station sim-hil --help

python -m control_station sim-hil \
  --dry-run --device cpu \
  --control-profile policy_human_allocation_scheme_a
```

`--dry-run` 通过只证明本地配置和 MuJoCo task 可以构造，不证明 policy 质量、真实设备、
网络 source、operator behavior、ControlTrace finalize 或真机 readiness。交互启动、VIVE
sender、Human demonstration、OpenPI server 与完整参数见综合操作指南，不要从历史报告中
复制旧命令。

## Owner 地图

```text
operator/browser
  -> Control Station typed intent + application lifecycle
  -> lower-owned Policy/Human emissions
  -> control_composition same-cut evaluation
  -> RobotMotion proposal/authority contracts
  -> MuJoCo target safety/IK/reference/dispatch
  -> passive ControlTrace facts
```

| 责任 | owner |
|---|---|
| operator intent、profile/lifecycle coordination、rebuildable UI state | `control_station` |
| GUI、浏览器生命周期、页面资源与 operator interaction | 目标统一为 `control_station.gui` |
| backend-neutral exclusive switch transaction | `control_station.hitl` |
| Policy/Human application source adapter | `control_station.sources` |
| MuJoCo allocation、rebase、trajectory publication binding | `control_station.composition` |
| Sim-HIL session/control-cycle/profile assembly | `control_station.sim_hil_profile` |
| causal cut、relative math、allocation、trajectory reducer/preview | `control_composition` |
| action/trajectory carrier、proposal/execution contracts | `robot_motion.action` |
| target safety、IK、reference、command、dispatch | `sim.mujoco.deploy` 与 lower `robot_motion` owner |
| policy worker/queue/cursor、human device production | 各自 lower source owner |
| trace WAL/seal/commit/readback | application capture owner；只观察，不控制 |

现有 `control_station/sim_hil_gui.py` 与 `control_station/web/` 是 GUI 收口前的兼容路径。
它们描述当前 caller，并不改变目标边界：新增 GUI/browser/operator 逻辑必须进入
`control_station.gui` 子集；控制算法、source state、HOLD/re-arm、safety 和 dispatch 不得
继续进入页面或 HTTP 模块。

## 操作面与证据面

- `--gui` 是只读状态页；
- `--control-gui` 是 loopback typed-intent operator UI，必须与 `--enable` 一起使用；
- 浏览器线程只提交有界、版本化 intent；Station control thread 独占资源启动、控制 step
  和 capture finalize；
- closing/refreshing browser 不等于结束 control runtime，browser 生命周期必须由
  application-owned handle 管理；
- ControlTrace 记录 source、composition、target/reference、measurement、difference、HOLD
  与 commit 事实，但不反向授予 authority；
- 当前 ControlTrace core 不等同于包含相机、触觉和 native streams 的正式训练数据集；
- Human demonstration capture 复用同一个 execution/capture owner，离线导出仍需选择
  decision span 并绑定任务成功证据。

## 三类控制语义

### 截断切换

`exclusive_truncated_hitl` 在 HOLD 内完成 source selection、Runtime Home（Human 接管）、
new activation 和 fresh emission。Policy 恢复会先作废旧 plan/chunk。任一步失败都保持
lower HOLD，不恢复旧 chunk，也不跳过 composition、safety、IK 或 dispatch。

### 连续 allocation

Scheme-A/B 对 component 使用 `base_only`、`relative_overlay` 和（Scheme-B）
`absolute_overlay_owner`。relative correction 是相对 active anchor 的 state，不按 control
tick 积分。rebase-and-clear 需要 typed intent、fresh Human sample、anchor receipt 和下一
未提交 target boundary；缺失/过期输入不能解释成“Human 静止”。

### Chunk-aware trajectory

`ComponentActionTrajectory`、trajectory reducer、preview、anchor revision、rebase 与薄 MuJoCo
proposal-publication seam 已存在，但当前公开 Sim-HIL 仍消费 source-owned
`PolicyActionEmission` 并做 streaming allocation。它没有公开的 locked-base trajectory
profile；future handback/base refresh 和 real binding 仍是独立 gate。

## 当前不应声称的能力

- 真实 RM65 上的 Policy/Human active switch 或 continuous blending；
- Sim-HIL no-action/virtual tests 已经授权真机 motion；
- OpenPI action chunk 已经接入整段 locked-base trajectory editing；
- corrected preview、ControlTrace 或 GUI status 是 executable Reference/authority；
- Scheme-B per-cut target continuity 等同于 trajectory-session future handback；
- 读取真实 Human 设备意味着执行 target 也是真机；
- ControlTrace core 已经是可训练的完整多模态 episode。

## 最小文档与契约验证

```bash
python scripts/check_docs_navigation.py
python -m pytest -q tests/test_control_station_hitl_contracts.py
python -m pytest -q tests/test_control_station_sim_hil_profile_selection.py
python -m pytest -q tests/test_control_station_sim_hil_allocation_profile.py
python -m pytest -q tests/test_control_station_exclusive_hitl.py
python -m pytest -q tests/test_control_station_sim_hil_gui.py
python -m control_station sim-hil --help
git diff --check
```

GUI server 测试只应绑定本机 loopback；沙箱禁止 AF_INET 时，需要在仍不联网、不启动真实
浏览器和设备的测试环境重跑。真实 VIVE、模型 server、viewer、capture 或 hardware 只在
对应任务明确授权后验证。
