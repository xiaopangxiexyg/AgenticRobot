# DexTeleop Control Station 当前边界与整理计划

状态：活动结构计划；最后复核日期为 2026-08-17。

本文只维护 Control Station 仍然生效的职责边界、当前结构债务、迁移顺序和验收门禁。此前
CS-0～CS-6b2h 的逐批状态、历史测试数字和完整合同讨论已冻结在
[Control Station 历史实施台账](../reports/control_station_implementation_ledger_20260724.md)。历史台账
用于追溯，不能替代当前代码、测试、[`AGENTS.md`](../../../AGENTS.md) 或新的 worktree 审计。

本阶段不调整 `agent_gateway/**`、Skill 定义/执行器或上层 RPent 通信语义。历史
`python -m control_station agent-gateway-mujoco` 仅按现有兼容转发边界保留；后续统一调整必须另立
任务和 owner。

## 1. 当前结论

Control Station 是 **GUI-facing application control plane**。它可以组织 human、replay、policy、
simulation 和 real-target 的应用接线，但不是第二套 action data plane。

它应负责：

- typed operator intent、接纳 receipt 和 operator-control lease；
- 可重建的 application lifecycle、UI status 和有界 event projection；
- application operation coordination、profile 选择及 source/target 的薄绑定；
- GUI server、浏览器生命周期、静态资源和操作员交互；
- 把下层已经产生的 emission、receipt、health 和 lifecycle port 组合成应用视图。

它不应负责：

- 设备、模型、dataset cursor 或 source worker 的生产；
- composition 数值策略、source admission、canonical safety、IK、reference、dispatch 或 authority；
- ControlTrace schema、WAL、Episode lifecycle、dataset writer 或 artifact persistence；
- 通过浏览器直接提交任意 qpos、pose、SDK packet 或未版本化 blend 参数。

现有正式入口保持不变：

```text
GUI human teleop / formal collection
  python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui

headless / compatibility collection
  python -m teleop_runtime.collection

policy deployment
  python -m teleop_runtime.deploy
```

`python -m control_station` 已是辅助 application-profile dispatcher，但不是替代上述入口的统一
production runtime。正式入口迁移只能在对应 profile 行为、退出、资源释放和安全证据逐项等价后再决定。

## 2. 当前 ownership

| 责任 | 唯一 owner | Control Station 的角色 |
|---|---|---|
| composition cut、strategy、session、handoff、difference | `control_composition` | 选择版本化 profile，绑定现有 port |
| human source admission、passive session、emission fanout | `teleop_runtime.action` | 绑定 application readiness/status |
| replay cursor、binding、fanout | `teleop_runtime.replay` / `robot_motion.replay` | 发 typed load/start/pause/seek 请求，不读第二遍 dataset |
| sync/RTC emission 与 step validation | `teleop_runtime.deploy` / `robot_motion.deploy` | 观察 producer 状态并绑定正式 emission |
| OpenPI artifact/protocol assessment | `deployments.openpi` | application preflight projection，不启动第二个 client |
| MuJoCo measurement、readiness、resolution、execution evidence | `sim.mujoco` | concrete sim profile assembly |
| authority、safety、IK、reference、dispatch | `robot_motion` | 请求和展示 receipt，不重建事实 |
| schema、projection、capture、persistence | `dataset_contracts`、`dataset_interface`、`collection_runtime` 与现有 capture owner | 发 begin/finalize 请求和显示状态 |
| GUI、browser、operator interaction | `control_station.gui` 目标子集 | 唯一物理 owner；其他 Station 模块只经 typed GUI port 交互 |

一个逻辑资源仍只能有一个 producer、cursor/plan owner、authority、canonical safety、dispatcher 和
Episode owner。Station UI cache、observer queue 和 status projection 可以被丢弃重建，不能推进或重建这些
权威状态。

## 3. 当前代码事实

### 3.1 已具备的稳定 application core

下列根级模块已经形成 backend-neutral 的 application core，并由边界测试约束：

```text
control_station/contracts.py
control_station/state_machine.py
control_station/runtime.py
control_station/operation_coordinator.py
control_station/publication_barrier.py
control_station/status.py
control_station/events.py
```

这些模块拥有 typed contract、deterministic reducer、operation coordination 和 rebuildable projection，
不应导入具体 teleop、replay、policy、sim、hardware 或 dataset writer 实现。

当前 dispatcher 暴露以下 application command：

```text
sim-hil
vive-calibration
vive-deflection
refinement-annotation
agent-gateway-mujoco    # historic forwarding command; excluded from this phase
```

其中 command 存在只证明入口和 application profile 存在，不表示 production、真机或 motion-ready。

### 3.2 GUI/browser 物理归属仍分散

当前直接拥有 `webbrowser` 或 HTTP server 生命周期的代码至少分布在：

```text
control_station/sim_hil_gui.py
control_station/vive_calibration_gui.py
control_station/vive_deflection_gui.py
control_station/web/server.py
control_station/web/calibration_web_ui.py
control_station/refinement_annotation/server.py
```

此外，若干 root application、WebGL view 和 static assets 与这些模块直接配对。这是当前最明确的
结构债务：职责在语义上属于 Station，但没有严格位于统一的 `control_station/gui/` 子集。

### 3.3 过渡性 concrete profiles

`control_station/composition/` 和 `control_station/sim_hil_profile/` 仍包含具体 MuJoCo、policy、replay、
human 和 capture 接线。它们在迁移期可作为 application leaf 导入多个下层包，但不能把可复用数值、
source/target runtime、dataset/model 实现固化为 Station ownership。最终保留的薄 assembly 应归入
声明式 `control_station/profiles/`。

边界测试当前只允许三个 lower→Station application-facade 例外：

```text
teleop_runtime/debug/run_vive_rm65_real_teleop.py
teleop_runtime/debug/run_vive_rm65_calibration_debug.py
teleop_runtime/collection/umi/app.py
```

它们是 canonical/compatibility entrypoint，不授权其他 lower package 新增反向依赖。迁移时先保留原
import path 为无状态薄转发，确认所有调用者和删除条件后再移除。

## 4. 目标物理结构

```text
control_station/
├── contracts.py / state_machine.py / runtime.py   # stable application core
├── operation_coordinator.py / status.py / events.py
├── gui/
│   ├── web/                                       # HTTP/API/static serving
│   ├── lifecycle/                                 # browser open/close/session resources
│   ├── real_teleop/                               # operator-facing real-teleop UI binding
│   ├── sim_hil/                                   # operator-facing sim-HIL UI binding
│   ├── calibration/                               # VIVE/RM65 calibration UI
│   └── refinement_annotation/                     # post-capture operator UI
├── profiles/                                      # thin declarative application assemblies
├── sources/ / targets/                            # typed application bindings only
└── <legacy public modules>                        # temporary compatibility re-exports
```

目录名只是目标布局，迁移的判断依据仍是资源生命周期和 ownership：

- 任何导入 `webbrowser`、HTTP server、浏览器 session、static asset 或 operator form 的实现必须位于
  `gui/`；
- `gui/` 可以依赖 stable core 和 profile-facing port，stable core 不得反向依赖 `gui/`；
- application coordinator 可以请求 GUI start/stop，但不能自己打开浏览器、持有 server thread 或解释
  browser payload；
- profile 可以组合多个 lower package，但只能绑定已有 typed port，不创建第二 queue、worker、cursor、
  authority/epoch owner、safety、dispatcher 或 Episode lifecycle；
- 旧 public import/CLI 路径在迁移期保留薄 shim。兼容导出必须指向同一个实现对象，不能复制逻辑或状态。

## 5. GUI 边界整理批次

### CS-G0：冻结当前 GUI debt

- 精确列出允许出现在 `gui/` 之外的 browser/server/static/operator-interaction 文件；
- 增加 AST 边界检查，禁止 debt allowlist 增长；
- 字符化每个入口的 parser、默认 bind、browser-open policy、start/close 幂等、thread join、退出码和
  public imports；
- import、`--help` 和 unit test 不得打开浏览器、绑定端口或启动设备。

### CS-G1：迁移被动 VIVE GUI

先处理 `vive-calibration` 与 `vive-deflection`：它们的 motion/episode 权限最窄，适合建立
`control_station.gui` 的 server/browser lifecycle 约定。移动实现时保留旧模块为薄 re-export，验证
对象身份、CLI help、bind/close 和静态资源 parity。

### CS-G2：迁移 refinement annotation GUI

把 server、browser lifecycle、operator selection 和静态资源放入 `gui/refinement_annotation/`；dataset
发现/读取仍由 `dataset_interface`，draft/persistence 仍由 `collection_runtime.progress`。迁移不得打开数据集
写锁的第二个 owner，也不得改变 draft/finalize 语义。

### CS-G3：迁移 Sim-HIL GUI

将 `sim_hil_gui.py` 与 status/operator view 的 GUI 部分归入 `gui/sim_hil/`。`sim_hil_profile/` 保留
application lifecycle 与 typed control-cycle binding；MuJoCo execution、composition strategy 和 source
worker 继续由下层 owner 提供。默认关闭和 `--enable` gate 不变。

### CS-G4：迁移 RealTeleop/calibration GUI

最后处理 `web/calibration_web_ui.py`、real-teleop operator controls 和 RM65 calibration application。
这是与设备、collection、HOLD 和 shutdown 顺序耦合最强的一批，必须逐个资源迁移，不能同时重写控制循环。
canonical RealTeleop 入口与参数保持不变；GUI 断连不得改变 dispatch、capture 或安全 watchdog。

每个批次只减少 debt allowlist，不顺手统一框架、重写页面或改变交互语义。共享 browser/server helper 只在
至少两个已迁移应用具有相同 lifecycle contract 时提取。

## 6. Profile 与 lower ownership 整理批次

GUI 归属稳定后，再继续原 CS-B 主线：

1. **CS-B1 source 去重：**确认 human/replay/policy 的生产、queue/cursor 和 admission 已由下层唯一拥有；
   Station 只保留 readiness、status 和薄 binding。保留有调用者的兼容导出。
2. **CS-B2 data/model 分离：**把 ControlTrace/OpenPI 具体 projection 约束到外部 adapter；Station core 只消费
   protocol-neutral receipt/event，不拥有 schema、WAL 或 artifact client。
3. **CS-B3 concrete profile 减薄：**可复用 composition 回到 `control_composition`，MuJoCo readiness/execution
   回到 `sim.mujoco`，Station profile 只装配 typed ports 和 manifest。
4. **CS-B4 package 收口：**将剩余薄 assembly 从 `composition/` 迁为 `profiles/`，删除无调用者 shim，最后才
   评估统一 production entrypoint。

`agent_gateway/**`、Skill primitive 和上层 RPent 连接不属于这些批次；即使与 Station profile 相邻，也不能在
本计划中顺手迁移。

## 7. 行为保持不变量

1. 浏览器打开、关闭、重连或 event 丢弃不改变 source queue、replay cursor、inference schedule、dispatch、
   capture 或 authority state。
2. HTTP token、operator lease、candidate、motion authority 和 backend permit 是不同事实，不能互相替代。
3. 浏览器只提交 typed intent draft；receiver admission 后才形成 intent，operation completion 必须带下层
   evidence。
4. GUI status 是可重建视图，不是 ControlTrace、source of truth 或 safety watchdog。
5. GUI/server 资源必须成对 start/close；partial construction、端口占用、browser-open 失败和 shutdown 异常均有
   确定的释放顺序。
6. 一个 formal capture 只有一个 Episode owner；GUI 的 begin/finalize 只是请求。
7. sim、shadow、fixture 或 `--help` 证据不授权 real motion。
8. 任何 target/profile/manifest 变化使旧 preflight 和 authorization 失效。

## 8. 每批验证

最小通用检查从已激活的 `${DEXTELEOP_ENV_NAME:-dexteleop310}` 环境运行：

```bash
python scripts/check_docs_navigation.py
python -m pytest -q tests/test_control_station_boundaries.py
python -m pytest -q tests/test_runtime_domain_boundaries.py
python -m control_station --help
git diff --check
```

GUI 物理迁移还必须按应用增加：

- old/new import object identity 或明确的反射/序列化 parity；
- parser 和 `--help` 输出/退出码；
- HTTP route、method、payload limit、same-origin/token、status schema 与 static asset；
- server start/close、重复 close、port-in-use、browser-open failure 和 thread join；
- 对应 application 的既有 focused tests。

测试只能证明执行到的离线边界。没有设备授权时不启动 VIVE、相机、机械臂、手、真实 policy transport 或
真实 collection writer。

## 9. 当前优先顺序

1. 先建立 CS-G0 精确 GUI debt gate，禁止 browser/operator lifecycle 继续散落；
2. 按 CS-G1 → G2 → G3 → G4 从低风险被动 GUI 到真机 GUI 逐批迁移，每批独立提交；
3. GUI 归属稳定后推进 CS-B1～B4，只迁移唯一 ownership，不重写数值或安全语义；
4. 每批开始前重新核对 linked worktree、实际调用者和 current main；历史台账不作为当前通过证明；
5. 完成批次的测试数字和现场细节进入带日期 report，不再回填到本活动计划。
