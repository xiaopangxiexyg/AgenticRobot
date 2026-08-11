# Control Station Sim-HIL 功能拆解与整理计划

更新日期：2026-08-07

状态：已完成 live-code ownership 审计及 P1--P5 的无行为迁移；D5a typed
control-cycle seam 已完成，allocation profile binding 尚未开始
接线。后续仍只按独立 owner 逐段迁移，不在结构调整中改变 public CLI、fixed-alpha
数值、设备生命周期或下游执行。

当前整理结果（2026-08-07）：

- `sim_hil_profile/status.py`：唯一 GUI status schema 与纯只读投影；
- `sim_hil_profile/config.py`：默认值、fallback 集合和 profile 配置验证；
- `sim_hil_profile/human.py`：VIVE/夹爪 source contract 与被动 source 选择/绑定；
- `sim_hil_profile/factory.py`：MuJoCo reset-state anchor、readiness 与一次性 graph assembly；
- `sim_hil_profile/application.py`：start/step/HOLD/re-arm/close 核心生命周期；
- `sim_hil_profile/application_human.py`：human attachment、acceptance 与 runtime Home
  子生命周期；
- `sim_hil_profile/application_status.py`：对 application 私有状态的只读投影；
- `sim_hil_profile/validation.py`：多个 profile 模块共享的两条纯基础校验；
- `sim_hil.py`：仍是兼容 facade 与 canonical CLI/GUI runner，当前约 1067 行；
- CLI runner 尚未机械迁入 `cli.py`；当前文件已经收敛为入口职责，不为了行数继续拆；
- 原 `control_station.sim_hil` 导入面继续由显式导入转发，没有在 package
  `__init__.py` 做宽泛 re-export。

## 1. 目标与冻结边界

`control_station/sim_hil.py` 整理前约 4970 行，同时包含配置、坐标/设备绑定、MuJoCo
target factory、状态投影、应用生命周期、CLI 和 acceptance GUI 接线。整理目标不是按
行数机械切文件，而是让每个模块对应一个真实 owner、状态机或资源生命周期。

本轮冻结：

- `python -m control_station sim-hil` 与 `control_station.sim_hil` 既有公开导入不变；
- default-off、`--enable`、dry-run 和 acceptance-gui 语义不变；
- fixed-alpha composition、fallback、HOLD/re-arm、episode end 与 target step 顺序不变；
- 不启动 VIVE/Q6、模型、MuJoCo viewer、physics worker 或真机；
- 不修改 `control_composition` 数值算法、MuJoCo target、IK、安全、authority 或 dispatch；
- 不借结构整理接入 allocation/D4；D5 在本计划完成后另行进行。

## 2. 当前功能分区

### 2.1 MuJoCo/Sim 必须拥有的具体绑定

以下内容不能泛化成 sim/real 共用算法：

- MuJoCo task/checkpoint/target 配置与 `prepare/build_mujoco_sim_hil_target()`；
- reset-state tool TCP 捕获与 Sim target session anchor；
- MuJoCo component materializer、ingress projector、FK execution observation；
- MuJoCo viewer 生命周期与 physics step profile；
- MuJoCo gripper/reference/measurement/target execution 的状态展示适配。

这些代码可从巨型入口迁到明确的 Sim-HIL profile factory/target binding，但仍属于
Control Station application leaf 或 `sim` 提供的下游端口，不能下沉到通用
`control_composition`。

### 2.2 Sim/Real/其他 application 可共享的语义

可共享的是接口和时序规则，不是当前整个 `MujocoSimHILApplication` 类：

- typed operator intent 与 application lifecycle coordination；
- publication-before-release、HOLD/re-arm、episode terminal ordering；
- 非 burst catch-up 的周期 deadline 推进；
- source readiness/availability 的只读快照接口；
- source publication、proposal、target、Reference/Measurement 的只读状态投影；
- source/target factory 通过 typed port 注入，而不是由 GUI 状态对象重建资源。

当前实现仍直接访问 MuJoCo runtime、fixed-alpha composition 和 VIVE session，因此首轮
不把整个类改名成“通用 application runtime”。只有得到第二个真实调用者并冻结 typed
port 后，才提升为跨 sim/real 公共模块。

### 2.3 可复用 utilities

- `_advance_control_deadline()` 是纯 application scheduling 规则，但目前只有 Sim-HIL
  两个调用点；先保留兼容位置，待出现第二个 profile 后再迁为公共 scheduling helper，
  避免为单函数新增杂物模块。
- `_finite()`/`_required()` 在 P1--P5 拆分后出现五个真实调用模块，因此收敛到
  profile-local `validation.py`；它不是跨项目 `utils.py`，不承载业务语义。
- 文件 SHA-256 是 profile/文件身份工具；当前与 VIVE coordinate profile 绑定紧密，后续
  应由 coordinate-profile binding owner 统一，而不是放入泛化 `utils.py`。
- GUI-safe status projection 是一组有独立输入/输出和失败策略的纯函数，已有多个调用者，
  应首先迁移到专门的 `sim_hil_profile/status.py`。

### 2.4 应归还其他模块边界的内容

- VIVE/Q6/angle-trigger 设备构造与关闭：继续由 `teleop_runtime` builder/session 所有；
  Station 只选择 profile、绑定 target contract、持有 application attachment。
- VIVE coordinate profile 的存储/验证：`control_station/vive_calibration_profiles.py`；
  Sim target anchor 仍是 Sim-HIL concrete binding。
- 通用 composition、allocation、trajectory、difference：`control_composition`；Station
  只做 profile/port 组合。
- MuJoCo target capability、execution、FK、Reference/Measurement：`sim/mujoco` 与
  `robot_motion`；Station 只读 receipt/snapshot。
- capture/seal/commit：既有 `sim_hil_acceptance.py` 和 lower capture owner；不能迁回
  application class 或 status projector。
- GUI HTML/HTTP：`sim_hil_gui.py`；它只接收状态与 typed intent，不启动资源。

## 3. 建议组织

保留 `control_station/sim_hil.py` 作为兼容 facade 和 canonical CLI module。新增语义明确的
内部 package：

```text
control_station/
  sim_hil.py                         # compatibility exports + main facade
  sim_hil_profile/
    __init__.py                      # 不做宽泛 re-export
    status.py                        # 纯、只读、GUI-safe projection
    config.py                        # Sim-HIL profile config/defaults
    human.py                         # concrete VIVE/gripper selection/binding
    application.py                   # core control-cycle/application lifecycle
    control_cycle.py                 # replaceable typed cycle seam + compatibility adapter
    application_human.py             # human attachment/Home sub-lifecycle
    application_status.py            # application read-only projection
    factory.py                       # MuJoCo target/source/composition assembly
    validation.py                    # profile-local pure scalar/text checks
    cli.py                           # deferred: argparse/run/acceptance GUI wiring
```

这不是要求一次创建全部文件。只有对应切片迁移时才创建模块；避免空 scaffold 和过度
拆分。`sim_hil_acceptance.py`、`sim_hil_gui.py`、`vive_calibration*.py` 保持现有 owner。

## 4. 分步实施

### P1：只读 status projection——已完成

- 迁移 emission/input/composition/human channel/gripper/VIVE/runtime contract/target/
  execution-difference/mix projection；
- 统一唯一 `SIM_HIL_STATUS_SCHEMA_V1` 声明；
- `sim_hil.py` 保留私有兼容导入，现有测试与仓库外调用不需要立即修改；
- `sim_hil_acceptance.py` 改为从新 owner 导入 `_vive_receiver_status`，消除跨巨型入口的
  私有反向依赖；
- characterization：状态 payload、异常抑制、VIVE wall/monotonic freshness、GUI schema
  完全不变。

### P2：profile config 与 CLI 参数映射——已完成

- 迁移 defaults、fallback sets、`MujocoSimHILApplicationConfig`；
- `build_arg_parser()`/`_config_from_args()` 暂留 CLI owner，避免配置与命令行同时迁移；
- 保留 facade re-export 和 dataclass identity/import path 兼容测试。

### P3：human/calibration concrete binding——已完成

- 将 human attachment、VIVE/gripper source spec 与 passive source builder 迁到
  `human.py`；
- coordinate profile merge 尽量复用 `vive_calibration_profiles`，Sim target reset anchor
  保持明确的 concrete callback；
- 不把设备实例、session cursor 或 recorder 放进 generic Station source adapter。

实施说明：VIVE coordinate merge 与 source binding 已迁移；MuJoCo reset-state TCP
capture/session anchor 刻意留在 concrete sim assembly，避免 human 模块反向拥有 sim。
设备构造仍由 `teleop_runtime` 的 lazy builder 完成，导入新模块不会打开设备。

### P4：MuJoCo factory/target assembly——已完成

- 迁移 `_build_application()` 中纯 assembly；
- factory 只构造一次 target/source/composition/application graph，不拥有长期 control loop；
- dry-run 继续只做 contract preflight，不加载 checkpoint、设备或 simulator worker。

实施说明：facade wrapper 显式注入 human source builder，既保持既有测试/profile
替换能力，又避免 factory 反向导入 CLI。新增的 `sim` 依赖位置已作为 concrete profile
写入 boundary allowlist，未向通用 Station 模块扩散。

### P5：application lifecycle class——已完成

- 将 `MujocoSimHILApplication` 迁到 `application.py`；
- 先保持现有方法与状态字段原样，只把 `status()` 依赖改为 P1 projector；
- 第二阶段才引入 typed control-cycle port，使 fixed-alpha 与 allocation profile 可替换；
  不在机械迁移时改 HOLD/fallback 数值语义。

实施说明：application 按真实 concern 进一步分成 core lifecycle、human attachment/Home
子生命周期和 read-only status mixin；没有复制 source cursor、authority、safety、IK、
dispatch 或 capture owner。

### P5b：typed control-cycle seam——已完成

- `application.step()` 不再写死调用 `composition.step()`，而是调用一个 typed
  `SimHILControlCyclePort`；端口只协调一个 cycle 并返回 lower-owned execution result；
- 默认 factory 显式选择 `FixedAlphaCompatibilityControlCyclePort`，参数、返回对象、数值顺序
  与现有 fixed-alpha cycle 保持一致；旧 profile 的 blend-failure/human-timeout fallback 只由
  该 port spec 显式保留；
- port 在 composition activation 完成后 activate，在 HOLD/re-arm boundary 前 deactivate，
  并在 lower composition/runtime 关闭前 close；这样未来 allocation worker 与 publication
  barrier 可以被先 fence/join，而不复制 application episode 状态机；
- allocation profile 尚未构造。特别是本切片没有新增 allocation source role、residual
  anchor、activity basis supplier、context worker、proposal producer 或执行语义。

### P6：CLI/acceptance runner——本轮不执行

- 最后迁移 argparse/run/main/GUI bootstrap；
- `sim_hil.py` 收敛为稳定 facade；
- 入口 help、dry-run JSON、read-only GUI、acceptance GUI 和 cleanup 顺序必须字符级或结构级
  等价。

复核决定：当前 facade 约 1067 行，剩余内容是 argparse、dry-run、GUI bootstrap、run/main
和兼容 re-export，已经是同一应用入口职责。现在迁到 `cli.py` 只会移动行数并增加一层
转发，不形成新的 owner；待 CLI runner 出现第二个调用者或 acceptance bootstrap 再形成
独立状态机时再拆。

## 5. 验收

每一阶段必须满足：

1. `control_station.sim_hil` 现有导出仍可导入；
2. sim-hil GUI/acceptance/MuJoCo contract 测试通过；
3. package boundary allowlist 不扩散；
4. `--help` 与 default-off dry-run 不构造资源；
5. 无 proposal/hash/ControlTrace schema 或状态 payload 漂移；
6. `git diff --check`、`py_compile` 通过；
7. 只迁移当前切片，不顺手修改 allocation、RTC、真机遥操或 dataset。

## 6. 停止条件

- 移动要求改变公开 CLI、GUI JSON、fallback/HOLD 或 episode lifecycle；
- 当前文件出现新的并行重叠修改；
- 需要选择是否兼容仓库外私有符号，而已知唯一远端 OpenPI 调用面无法确认；
- 测试表明现有状态函数带有隐藏控制副作用；
- 需要启动设备、模型、viewer、physics 或 motion 才能验证。

## 7. 本轮验证回执

- `py_compile`：facade、config、human、status、GUI 与 acceptance 模块通过；
- 聚焦回归：结构拆分后，Sim-HIL GUI、acceptance、MuJoCo contract 与 Station boundary
  的 114 个非 socket 用例通过；5 个 loopback HTTP 用例在允许本机 socket 后通过，更新
  concrete boundary allowlist 后对应测试通过，覆盖集合共 120 个用例；
- 去重基础校验后，另行复验 config、deadline、re-arm、status 与 boundary 共 `8 passed`；
- canonical CLI：`python -m control_station sim-hil --help` 正常，参数面未变化；
- facade identity：配置类、坐标绑定函数与 status schema 均指向新唯一 owner；
- lazy import：仅导入 facade/config/human/status 不加载 `sim.mujoco.deploy` 或
  concrete passive VIVE/Q6 builder；
- D5a control-cycle seam：专项 `3 passed`，GUI/acceptance `78 passed`，MuJoCo Sim-HIL
  contract `34 passed`，allocation/publication `20 passed`，Station boundary `8 passed`；
  全部为本地 virtual/contract 验证，未启动设备、模型 server、viewer、physics 或 motion；
- `git diff --check` 通过；验证期间未启动设备、模型、viewer、physics 或 motion。
