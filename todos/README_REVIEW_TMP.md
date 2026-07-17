# DexTeleop 当前框架复核（临时）

> 复核日期：2026-07-17
> 复核范围：当前工作区的目录边界、入口链路、运行时、数据契约、部署、仿真、硬件适配、迁移状态、文档和测试证据。
> 本轮操作：只更新本报告文件；没有修改代码、测试、配置或依赖文件，没有启动相机、VIVE、RM65、灵巧手或任何真实运动。

## 1. 结论先行

DexTeleop 已经从“多个功能脚本并列”推进到较清晰的分层框架：`robot_motion` 负责共享运动语义和控制安全，`hardware` 负责设备后端，`teleop_runtime` 负责真实遥操作/采集/部署入口，`dataset_contracts` 与 `dataset_interface` 负责纯数据契约，`sim` 负责仿真，`deployments` 负责策略客户端和协议整合，`legacy` 负责确认归档内容。

当前最重要的架构收敛点已经出现：

- GUI 人遥操作、headless 兼容采集和策略部署都开始共享“观测准备—动作解析—安全处理—最终派发”附近的语义。
- GUI 采集已经有 `TeleopSession`、`BimanualRig`、因果观测、故障/hold、位置参考发布和统一 writer runtime 等较明确的主链路。
- 部署侧已经有策略频率、预测事务、动作 chunk、v2.3 解码、IK 恢复丢弃和 canonical action preparation；但它还没有成为完整的真实设备生命周期/故障/hold/位置执行控制器。
- 数据层已经覆盖 v2.1/v2.2 兼容、v2.2 typed action、v2.3 pose frame、固定 horizon pose/action chunk、policy artifact/layout hash、视频/遮挡/缺失视图和 UMI arm-free 采集等新契约。
- RM65 双臂表中心坐标、双臂 IK、工具安全、MuJoCo 位置执行和碰撞安全投影已经形成新的能力层，但真实控制链路与仿真链路仍存在不同的频率、安全认证和执行原子性边界。

因此，当前状态适合继续做契约收敛、测试修复和 observe-only 验证，不宜把工作区视为可直接发布的稳定基线。主要原因是：

1. 工作区仍有大量并行修改，当前统计为 137 个已修改、17 个已删除、45 个未跟踪路径；这些变化不能简单视为单一提交的完整迁移结果。
2. 在当前本地环境执行全量 pytest 得到 `1668 passed, 49 skipped, 2 failed`，两个失败分别涉及真实 CLI 执行分支校验和 GUI 入口行数冻结基线。
3. 项目目标 Python 是 3.10，而本轮运行测试的解释器是 Python 3.12.2；当前结果是本地无硬件环境证据，不是目标 `dexteleop310` 环境的发布验收。
4. 位置控制频率已经按 profile 分化，但代码、入口默认值和仿真/README 仍同时出现 `15→75→150`、`15→60→120`、`15→45→150` 等口径；必须先把 profile、manifest 和验收条件冻结，不能只看一个全局数字。
5. 真实位置执行仍主要是端点碰撞检查，尚未认证 120 Hz 命令层的连续段碰撞；真实双臂/手部派发也不是单事务原子提交。
6. 策略与人遥操作之间的共享 seam 已经明确，但 `ActionArbiter`、统一 `ControlTrace`/lineage 和完整 sim deploy 对接仍是下一阶段工作，而不是当前已完成能力。

## 2. 复核证据与当前工作区基线

### 2.1 只读核对项

本次复核使用了以下低风险检查：

- 阅读根目录 README、`AGENTS.md`、架构规划、各运行时 README、数据契约 README、仿真 README 和关键入口/实现。
- 查看 `git status --short`、目标报告的原始 diff、包发现配置和迁移测试；没有重置、清理或覆盖并行修改。
- `env PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --collect-only -q`：收集到 1716 个测试。
- `env PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`：1668 通过、49 跳过、2 失败。
- 入口 `--help` 检查：GUI 正式入口、headless collection、deploy 和 MuJoCo control 均正常退出。
- `uv lock --check`：通过；这只证明 lock 文件可解析，不能证明依赖已安装或目标环境已验证。
- `git diff --check`：报告更新后再次执行，用于检查本次文档差异中的空白/补丁问题。

### 2.2 工作区与环境事实

| 项目 | 当前事实 | 解释 |
|---|---|---|
| 版本库状态 | 137 modified、17 deleted、45 untracked | 统计包含已有并行工作，不能据此推断本轮修改范围 |
| Python | `/home/zc/anaconda3/bin/python`，Python 3.12.2 | 与项目目标 Python 3.10 不一致 |
| 项目目标环境 | `environment.yml` 的 `dexteleop310` | 还需要在目标环境重跑关键测试 |
| 依赖元数据 | `pyproject.toml`、`environment.yml`、`uv.lock`、`README_ENV.md` 已存在 | 旧报告中“没有统一环境元数据”的结论已经过时 |
| lock 检查 | `uv lock --check` 通过 | 未进行依赖安装和硬件 SDK 验证 |
| CI | 当前工作区未见明确的 CI workflow | 仍缺少自动化的目标 Python/无硬件验收门槛 |
| 本轮外部状态 | 未初始化真实设备，未发送运动命令 | 结论只来自静态阅读、导入/help、测试和契约检查 |

`pyproject.toml` 当前要求 `>=3.10,<3.11`，包发现范围覆盖 `dataset_contracts`、`dataset_interface`、`deployments`、`hardware`、`robot_motion`、`sim`、`teleop_runtime` 和 `utils` 等目录，排除了 `tests`、`legacy` 和资产目录。当前仍需注意：包发现配置中还保留 `rm65_inspire_tac*` 模式，而旧的根级 `rm65_inspire_tac` Python 包已经迁移/删除；这更像是需要清理和安装验证的元数据漂移，不等同于当前生产代码仍在导入旧包。

## 3. 当前目录分层与所有权

当前主目录可以概括为：

```text
                         ┌─────────────────────────────┐
                         │ teleop_runtime               │
                         │ GUI / headless / deploy      │
                         └──────────────┬──────────────┘
                                        │
             ┌──────────────────────────┼─────────────────────────┐
             │                          │                         │
   ┌─────────▼─────────┐      ┌─────────▼─────────┐      ┌────────▼─────────┐
   │ robot_motion       │      │ dataset_contracts │      │ deployments       │
   │ Rig/Safety/IK/     │      │ schema/frame/     │      │ policy/protocol/  │
   │ planner/control    │      │ artifact/layout  │      │ artifact client   │
   └─────────┬─────────┘      └─────────┬─────────┘      └───────────────────┘
             │                          │
   ┌─────────▼─────────┐      ┌─────────▼─────────┐
   │ hardware           │      │ dataset_interface │
   │ vendor/device      │      │ window/batch/     │
   │ backends           │      │ video/tactile     │
   └────────────────────┘      └───────────────────┘

   ┌────────────────────┐      ┌───────────────────┐
   │ sim                │      │ legacy            │
   │ MuJoCo/SAPIEN/     │      │ confirmed archive │
   │ adapters/tasks     │      │ only              │
   └────────────────────┘      └───────────────────┘
```

按 `AGENTS.md` 和现有代码，目录责任如下：

| 目录 | 当前责任 | 边界判断 |
|---|---|---|
| `robot_motion/` | 共享运动语义、Rig、Safety、Planner、位置控制、部署动作准备和运行时契约 | 不应初始化 vendor 设备，也不应放训练实现 |
| `hardware/` | VIVE、相机、RM65、灵巧手、触觉等设备后端 | 不应拥有数据集 schema 或 learner |
| `sim/` | MuJoCo/SAPIEN 后端、任务、仿真 adapter 和 safety | 应依赖 `robot_motion`，不能反向依赖 `teleop_runtime` 或真实硬件 |
| `teleop_runtime/` | 真实遥操作、GUI/headless 采集、策略部署前端和生命周期编排 | 是 canonical CLI 所在层 |
| `deployments/` | 模型客户端、协议和 artifact 整合 | 当前部分 runtime 仍是 scaffold |
| `dataset_contracts/` | 纯 schema、layout、frame、artifact、stats 和兼容契约 | 不应依赖硬件、训练或 UI |
| `dataset_interface/` | trajectory/window/batch、视频、触觉和 pose/action adapter | 保持离线/纯数据方向 |
| `model/` | 规划中的 learner/DAgger/RL 包 | 当前根目录没有正式 `model/` 包 |
| `legacy/` | 已确认归档的代码/文档 | 生产代码不应导入；当前包发现已排除它 |
| `tests/`、`todos/` | 验证和规划 | 不属于运行时生产入口 |

`sim/mujoco_temp_pipeline/` 是明确标为临时的 simulation-only PPO/快速 vector pipeline，使用 privileged state 和 debug contact latch，不应被误读为正式 `model/` 或 sim-to-real 策略栈。

## 4. Canonical 入口与实际执行链

### 4.1 入口表

| 用途 | Canonical 命令 | 当前定位 |
|---|---|---|
| GUI 人遥操作与正式采集 | `python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui` | 正式 GUI 主入口 |
| headless/兼容采集 | `python -m teleop_runtime.collection` | headless/compatibility；不是第二套 GUI 主实现 |
| 策略部署 | `python -m teleop_runtime.deploy` | 正式部署前端 |
| MuJoCo 控制检查 | `python -m sim.mujoco.run_control` | 仿真控制/诊断入口，不是硬件入口 |
| 通用 kernel | `teleop_runtime/core/kernel.py` | experimental，不是正式替代入口 |

旧的直接调用仍可能被 OpenPI real launcher 或兼容测试使用，但应视为兼容 API/迁移过渡，不能因此再增加新的平行生产入口。

### 4.2 GUI 正式链路

当前架构规划中给出的主链可以还原为：

```text
debug.main
  → _build_components
  → TeleopSession.run_step
  → causal observation sources
  → SafetyMonitor / fault tracker / hold
  → retargeters
  → command filter / guard
  → TeleopSession.dispatch_command
  → BimanualRig.send_commands
       （当前最终 hardware dispatch seam）
  → recovery / gripper / preview
  → collection_step_result(T0)
  → PositionReferencePublisher.publish_step
       （唯一 reference timeline）
  → RealTeleopDatasetRecorder.record_step_async
  → CollectionWriterRuntime
  → observer
  → AsyncWriterPool
  → LeRobotV21Writer
```

这里有两个重要收敛结果：

- GUI 的动作派发不应再让 camera/手部/数据写入代码各自重复实现主控制语义。
- 位置参考发布和数据记录都从同一 step 结果派生，目标是避免“屏幕上看到的动作”和“落盘动作”使用不同时间线。

GUI 当前主入口仍较大：`teleop_runtime/debug/run_vive_rm65_real_teleop.py` 约 15188 行。相关逻辑已经抽出多个模块，但入口仍同时承担参数、设备组装、GUI、主循环、采集和兼容行为，后续仍有继续拆分的结构性收益。

### 4.3 Headless/兼容采集链路

`teleop_runtime.collection` 复用观测、fault/hold、writer runtime 和部分 Rig/控制语义，但其定位是 headless/兼容。当前文档明确区分：

- 正式 GUI 采集走 debug GUI 主入口。
- headless 入口主要用于兼容、无 GUI 或受限环境。
- collection replay/直接 Rig 的特殊路径仍存在，后续应收敛成 `ReplayActionProposal` 等显式动作来源，或明确降为 legacy compatibility，而不应成为第三套控制语义。

正式 GUI 的位置采集默认 `--collection-worker-mode auto` 会解析为 process worker；相机数组和嵌套触觉数组使用 shared-memory slot。队列过载时暂停 episode 写入而不是继续积压或改变机器人运动；thread fallback 主要用于不适合 process 的 force/impedance 路径和兼容场景。

### 4.4 策略部署链路

当前部署主链为：

```text
teleop_runtime.deploy.cli
  → run_policy.main
  → build_deployment_observation
  → artifact pose/chunk/frame binding
  → PolicyDeployRuntime.predict_and_queue
  → policy-rate decision
  → prediction transaction
  → observation preparation
  → policy predict
  → normalize + v2.3 decode
       （measured TCP anchor、relative/absolute/frame 转换）
  → ActionChunkBuffer
  → prepare_next_action
  → CanonicalDeployActionPipeline
  → ActionSafetyFilter
  → RobotActionResolver
  → TeleopCommand
  → direct dispatch 或 PositionDeployExecutor
```

当前已经实现的部署侧责任包括 observation preparation、policy-rate 决策、预测事务、action chunk、IK recovery 时丢弃无效 chunk、v2.3 解码和 canonical preparation。它还没有统一拥有以下真实控制责任：

- hardware lifecycle 和设备初始化；
- 全局 fault/hold/recovery 状态机；
- 真实位置 sink 的全部时序语义；
- GUI/部署统一的 `ControlTrace`、动作来源和 lineage；
- 人/策略/回放之间的 `ActionArbiter`；
- 完整的仿真部署对接。

目前最合理的共享 seam 是“模型视图 → absolute `RobotAction` → canonical preparation”。下一阶段应在 absolute `RobotAction` 与 `CanonicalDeployActionPipeline` 之间插入纯逻辑 `ActionProposal`/`ActionArbiter`，让人、策略、回放能复用安全和派发边界，而不是各自复制一套执行器。

## 5. 数据契约、动作布局与采集输出

### 5.1 Dataset contracts

`dataset_contracts/` 当前保持硬件无关，主要包含：

- parquet/schema、stats、tactile、wrist depth、v2.1 compatibility；
- v2.2 typed action 和 layout metadata；
- v2.3 pose frame source contract；
- pose/action fixed-horizon chunk contract；
- policy artifact、normalization 和 layout hash。

`dataset_interface/` 当前承担视频/遮挡/缺失视图、确定性 photometric augmentation、归一化 state noise/dropout/masked concat、Torch adapter、pose/action window 和 batch adapter。它不应承担离线 mask 生成、硬件初始化或 learner 实现。

### 5.2 v2.3 pose frame 与 UMI

v2.3 的关键变化不是简单增加一个数组，而是把源坐标系写进契约：包含 `source_id`、原始 frame 和变换信息，消费方必须显式绑定 source frame，不能凭上下文猜测坐标系。当前运行时区分 collection desk frame 与 RM65 runtime raw/table-center 语义，需要在数据元信息和部署 artifact 中保留这条边界。

固定 horizon 的 `PoseActionChunkContract` 使用 episode-local start，短 chunk 必须明确选择 `drop` 或 `pad_last`，不能隐式重复或跨 episode 拼接。

UMI 当前是正式的 arm-free v2.3 采集分支：

- `utils/umi_hand_eye` 负责 RealSense + VIVE tracker 到 board-defined canonical desk 的刚体标定；标定有相机序列号、tracker 绑定和过期时间校验。
- `teleop_runtime.collection.umi` 复用 VIVE、RealSense、因果 frame、AngleTrigger、scalar gripper、DMTac、HandStateHistoryPoller、writer 和四踏板 GUI。
- UMI 路径不会构造 RM65 arm backend，也不会把标注动作当作机器人控制。
- `--validate-only` 只校验配置、layout、标定和设备绑定，不初始化设备；实际采集要求新鲜标定。
- 该标定解决的是 UMI 相机/追踪器/桌面坐标关系，不是 RM65 base/TCP 标定。

### 5.3 Action layout 与 artifact gate

当前仍需同时兼容 legacy v1 action layout（常见 26D/38D）和 v2.2 typed action。v2.2 的 float/int token 分离、layout hash 和 artifact gate 是必要的，不应再用“数组维度相同”作为兼容判据。

`v22_q6_bridge` 是显式 legacy q6 桥接路径；`v22_native` 如果没有完整的 native safety 和 dispatch adapter，应 fail-closed，而不是自动猜测 q6 语义。

相对动作、绝对动作、measured TCP anchor 和 source frame 转换应在部署前完成且留下可追踪元信息。当前动作 chunk 和 v2.3 解码已经具备基础，但统一的动作来源 lineage 仍待 `ActionArbiter`/`ControlTrace` 收口。

## 6. 位置控制、时序 profile 与安全边界

### 6.1 当前频率不是一个全局值

当前源码和 README 实际使用多个明确 profile：

| 使用场景 | 上游 policy/sample | reference / IK | command | physics | 当前含义 |
|---|---:|---:|---:|---:|---|
| GUI/real collection position 默认 | 15 Hz | 75 Hz | 150 Hz | — | `PositionControlRates` 默认 profile |
| deploy position 自动 profile | 15 Hz | 60 Hz | 120 Hz | — | `position_15_60_120hz`，当前部署默认 |
| deploy 显式 `position` | 15 Hz | 75 Hz | 150 Hz | — | 与共享默认 profile 对齐 |
| deploy 显式兼容 `position_45_150hz` | 15 Hz | 45 Hz | 150 Hz | — | 旧/兼容低 reference profile |
| MuJoCo CubeLift direct-qpos debug | 15 Hz | — | 150 Hz qpos | 1000 Hz | 上游完整 qpos + ZOH；不是 actuator dynamics |
| impedance 分支 | 入口相关 | — | — | 40 Hz control | 与 position 分支独立，不能套用 position 结论 |

其中 `robot_motion/position_control/interpolation.py` 的共享默认是 15→75→150，`config.py` 另外提供 15→60→120 和 15→45→150 profile。`teleop_runtime.deploy` 在没有显式选择时使用 `position_15_60_120hz`；GUI/collection 和显式 position 仍是 75/150 口径。仿真 README、真实部署 README 和历史规划文档曾分别描述不同 profile，这不是可以忽略的排版差异，而是需要写进 profile 名、运行 manifest 和验收测试的控制契约。

### 6.2 Position deploy 的已知时序语义

当前 position deploy 文档和实现已经明确了以下规则：

- 默认启用 position chunk lookahead；当下一动作已知时消除一个周期的等待，`--no-position-chunk-lookahead` 保留延迟兼容路径。
- chunk 结束后保持最后一帧，不跨 episode 借帧。
- 真实 send 前要求新鲜的双臂 measured qpos，年龄不超过 0.25 s、双臂 skew 不超过 0.05 s；不能以旧缓存或当前猜测状态替代权威 qpos。
- 单臂 position send 在缺少被动权威 qpos 时 fail-closed。
- 60 Hz qref endpoint collision 是已选择 profile 的检查结果；120 Hz 命令层的 inter-endpoint segment collision 尚未认证。
- hand collision 当前使用固定 open hand proxy，不代表任意定制工具的真实几何。
- arm queue 和 hand I/O 是分开、按序、部分可提交的路径，不是双臂加手部的原子事务；故障处理需要 fail-stop/hold 和 receipt 证据。

### 6.3 真实与仿真的碰撞安全并不等价

RM65 scene profile 当前有 12 组预期 active collision pairs，覆盖双臂 link proxy、手部 proxy、桌面和相机杆等对象。真实位置侧仍以 endpoint/配置化几何检查为主；缺少 120 Hz 连续命令段的完整认证，也没有任意工具网格的通用表示。

仿真侧的 `sim/mujoco/collision_safety` 已有更强的参考实现：

- forbidden geom pair distance oracle；
- joint sweep、投影到最后安全 qpos；
- hard penetration 检查；
- 约束 clearance、penetration、joint step 和 projection iteration 的策略参数；
- shared safety projection 可供 reference/fast CubeLift 路径使用。

这不能直接证明真实硬件安全。真实两侧 SDK 顺序发送、手部异步 worker、设备状态和实际工具几何仍需要单独的硬件/台架验收，而本轮没有执行这些动作。

## 7. 仿真现状

MuJoCo 是当前 canonical task physics；`sim/rm65` 的 SAPIEN 路径属于 preview/渲染适配，不是第二套已验收的控制后端。

当前 MuJoCo 主线包括：

- RM65 双臂/桌面/相机杆等场景配置和 geometry collision safety；
- `MujocoPositionExecutionRig`，语义为原子双臂 60 Hz qref → 120 Hz qcmd → 1000 Hz physics；使用 bounded latest-reference mailbox，不积压旧 reference，并拒绝部分双臂 reference；
- Fast/vector CubeLift 环境、executor、MJB cache 和 benchmark 基础设施；
- direct-qpos debug controller，把上游完整 150 Hz qpos 以 ZOH 送入 1000 Hz physics；
- CubeLift 的右臂/右手 debug task、归一化关节/夹爪 delta、privileged observation 和 contact latch debug 语义。

当前仍明确未完成或不能过度解读的部分：

- 没有 RM65 完整 actuator dynamics、力控/重力补偿/阻抗或真实硬件 streaming 仿真；
- `robot_dynamics=not_simulated` 的 direct-qpos 结果不能作为物理抓取、跟踪质量或 sim-to-real 证据；
- `sim/mujoco_temp_pipeline` 的 PPO checkpoint 和 benchmark 是临时、simulation-only、privileged-state pipeline，不是正式 learner 包；
- `sim/configs/rm65_bimanual_physics_v1.json` 仍把 actuator model 和真实 150 Hz system identification 标为 pending data；
- 仿真 60/120/1000 与 GUI 75/150、deploy 60/120 的组合必须通过同名 profile 和 manifest 对齐，不能仅凭“仿真跑通”推断真实链路等价。

## 8. 设备输入、采集并发与新增能力

### 8.1 VIVE

VIVE sender/source 当前已向 300 Hz 语义收敛：单个 OpenVR pose snapshot 可复用到一帧 payload，支持 velocity payload/fallback、sequence/loss/duplicate diagnostics、基于时间的 history、motion intent 和可选独立 clock-sync control port。旧 sender v1 的 shape/clock 语义不能与新协议混用。

这部分仍需要在实际设备和网络条件下验证丢包、时钟、追踪丢失和 hold 行为；本轮只做代码/测试阅读和无硬件检查。

### 8.2 采集 writer runtime

新的 process writer 路径把相机和嵌套触觉数组放入 shared-memory slot，以有界队列和 overload pause 保护采集 episode。其价值是把写盘压力与运动主循环解耦，同时保留 episode 边界和状态观测。

需要继续关注：

- writer overload 时数据状态、episode end 和恢复动作是否在所有 worker mode 一致；
- process/thread fallback 是否保持相同的 ordering、failure 和 receipt 语义；
- 相机、触觉、状态、动作的 T0 时间线是否能在落盘数据中重建。

### 8.3 RM65 frame/IK/tool safety

新的 RM65 相关能力包括 table-center frame 配置、双臂 IK、单臂 IK、IK tuning、target solver 和 tool safety。现有测试要求活跃 RM65 URDF 具有 table-center frame，并共享目标 profile；这有助于消除 collection desk frame 与 runtime raw frame 的隐式混淆。

后续仍需把 frame profile、tool proxy、IK tuning、collision scene profile 和实际机器人固件/工具版本一起写入可复现的运行 manifest，否则同一个数据集或策略 artifact 仍可能在不同坐标/工具假设下运行。

## 9. 迁移与兼容状态

当前已经能看到的主要迁移方向：

| 旧路径/语义 | 当前路径/语义 | 状态判断 |
|---|---|---|
| `dataset_postprocess` | `utils/dataset_postprocess` | 旧路径已移除，迁移测试覆盖 |
| `sim/mujoco_real2sim` | `sim/mujoco` | canonical 仿真路径已建立 |
| `utils/wrist_cam_fix_mask` | `utils/wrist_cam_fixed_mask` | 命名迁移已建立 |
| 根级 `rm65_inspire_tac` | `legacy/rm65_inspire_tac` 归档方向 + 新 `robot_motion`/`hardware` 拆分 | 生产导入已基本切断，但包发现和文档仍有残留 |
| 旧 action/deploy imports | `robot_motion` canonical 语义 + `teleop_runtime` compatibility shim | 不应突然删除兼容入口 |
| 多套 writer/dispatch 实现 | `CollectionWriterRuntime`、`TeleopSession.dispatch_command` seam | 已有收敛证据，仍需统一 trace/arbiter |

`tests/test_package_migrations.py` 和依赖边界测试支持“旧根包无 Python 实现、canonical 包存在、生产依赖不应反向进入 legacy”的方向。当前仍建议清理：

- `teleop_runtime/README.md` 中仍出现旧的 `rm65_inspire_tac.ik.SingleArmIKSolver` 示例；
- `pyproject.toml` 中的 `rm65_inspire_tac*` package include 与实际目录状态不一致；
- 规划文档、历史 README 和临时计划中仍有旧路径/旧频率描述，需要标注历史或改成 canonical profile；
- 不要因为迁移残留而直接删除仍被 OpenPI launcher、兼容测试或用户脚本使用的 active shim。

## 10. 当前验证结果

### 10.1 已通过的只读验证

- 测试收集：1716 个测试成功收集。
- `python -m teleop_runtime.debug.run_vive_rm65_real_teleop --help`：通过。
- `python -m teleop_runtime.collection --help`：通过。
- `python -m teleop_runtime.deploy --help`：通过。
- `python -m sim.mujoco.run_control --help`：通过。
- `uv lock --check`：通过。
- 本报告更新后 `git diff --check`：应以最终复核结果为准；本次不会把其他工作区差异混入报告修改。

上述 help/import 检查没有初始化真实设备，也没有执行 homing、replay、servo、force-control、reboot 或其他 device-state change。

### 10.2 全量 pytest 当前结果

在当前 Python 3.12.2、禁用用户 site 和第三方 pytest plugin 的环境中：

```text
1668 passed, 49 skipped, 2 failed
```

两个失败为：

1. `tests/test_deploy_policy_qpos_init.py::DeployPolicyQposInitTest::test_real_arm_send_requires_explicit_execution_branch_in_real_cli`

   测试期望真实 CLI 在 `--send-actions` 且没有显式 execution branch 时抛出 `ValueError("explicit execution branch")`，当前实际没有抛出。相邻测试又覆盖了显式 `--start_init_qpos` 的允许路径，说明这里是 execution branch 参数解析/验证契约需要重新对齐的问题，不能简单把所有 qpos 初始化参数都禁掉。

2. `tests/test_real_teleop_refactor_contracts.py::test_real_teleop_light_dedup_measurements_are_frozen`

   冻结测试期望 GUI 主入口基线为 14536 行，当前实际约 15188 行。代码已经有抽取模块，但入口仍继续变化；这是 refactor line-count baseline 与当前树不一致的信号，不应通过修改测试基线来掩盖尚未复核的结构变化。

这两个失败都没有在本轮修复，因为本轮明确只做分析和报告。

### 10.3 验证边界

49 个 skipped 的具体原因需要在目标环境中按测试名分类；不能把 skip 自动算作通过。当前测试运行器也不是项目声明的 Python 3.10 环境，因此至少以下结论仍未形成发布证据：

- `dexteleop310` 中的完整测试结果；
- 真实/可选 SDK 安装后的导入和 ABI 兼容；
- VIVE、相机、触觉、RM65、灵巧手的真实时序与故障处理；
- 真实位置 60/75/120/150 profile 的端到端 deadline、碰撞和 dispatch receipt；
- 真实策略部署、策略/人遥操作仲裁和数据落盘的完整闭环；
- MuJoCo 结果与真实执行之间的 system identification 或 sim-to-real 证明。

## 11. 风险与待办优先级

### P0：当前验收阻塞

- 先解释并修复/重新定义真实 CLI 的“显式 execution branch”校验，再决定 `--start_init_qpos` 等参数的允许组合；必须保留 fail-closed 原则。
- 重新审核 GUI refactor 冻结测试：确认 15188 行变化对应的模块边界、重复逻辑和入口责任，再更新合理的结构契约；不要只改数字。
- 在项目目标 Python 3.10 环境重跑收集、全量和关键入口检查，给出与本地 Python 3.12 分开的结果。

### P1：控制与契约风险

- 冻结并集中声明 position profiles：GUI/collection 的 15→75→150、deploy 默认的 15→60→120、兼容的 15→45→150，以及 MuJoCo 的 1000 Hz physics 语义；将 profile 名、frame、tool proxy、collision scene、IK tuning 和 artifact/layout hash 写入运行 manifest。
- 补齐 120 Hz 命令层连续段碰撞认证，明确真实硬件的 segment sweep、工具几何和双臂同步策略；在此之前不能声称真实位置链路已完成全频率碰撞安全。
- 明确 arm/hand 非原子派发在异常时的状态机、hold、fail-stop 和 receipt；不要让 writer 成功被误解为 hardware dispatch 成功。
- 建立纯逻辑 `ActionProposal`/`ActionArbiter`，统一人、策略、回放的优先级、来源、版本、时间戳和安全入口。
- 增加统一 `ControlTrace`/lineage，使 observation、policy prediction、decoded action、safety decision、resolved command、dispatch receipt 和 dataset row 可以关联。
- 对 `v22_native` 继续保持 fail-closed，只有 native safety/dispatch adapter 和 artifact gate 齐全后才允许真实发送。

### P2：工程与可维护性风险

- 清理 `pyproject.toml` 的旧包发现模式和 `teleop_runtime/README.md` 的旧 import 示例，并给历史规划文档加明确日期/状态。
- 明确 `sim/mujoco_temp_pipeline` 的临时生命周期、产物目录和删除条件，避免临时 PPO 逐渐成为未声明的正式模型入口。
- 继续拆分超大 GUI/deploy 入口，但每一步都以行为契约、入口 smoke test 和 line-count/模块责任审查为依据。
- 为 Python 3.10、无硬件、可选仿真、SDK 存在/缺失等矩阵建立 CI 或至少可复现的本地验收脚本；当前工作区未见明确 CI workflow。
- 分类处理 `logs/`、`datasets/`、`teleop_records/`、`sim/outputs`、`remote_data/` 等生成物和未跟踪临时计划；清理动作必须另行确认，不能在本次报告任务中代做。

## 12. 建议的后续推进顺序

建议按下面顺序推进，且每一阶段先保持 read-only/mock/observe-only：

1. 固定 Python 3.10 环境和测试命令，分类 49 个 skipped，处理当前两个失败测试对应的真实契约。
2. 形成单一的 control profile registry：频率、lookahead、qref freshness、frame、tool proxy、collision profile、IK tuning、action layout 和 artifact hash 必须可打印、可落盘、可复现。
3. 用 fake hardware、synthetic observation、fake resolver 和 dry-run dispatch 验证 `ActionProposal`/`ActionArbiter`/`ControlTrace`，先不连接真实机器人。
4. 在仿真中完成 60 Hz qref、120 Hz qcmd 的连续段安全测试和故障注入，再单独制定真实硬件台架验收范围。
5. 只在显式安全范围、设备状态检查和人工确认都具备后，做真实设备的单臂/低风险/hold-first 验证；策略-only real deployment 应排在这些门槛之后。
6. 最后再处理 legacy shim 的删除条件、文档收口、CI 和临时 pipeline 归档。

## 13. 最终判断

项目的方向是正确的：共享运动层、纯数据契约、真实运行时、仿真和部署前端的边界已经比旧版清晰，GUI 采集和策略部署也不再完全各自为政。当前瓶颈不在“缺少更多入口”，而在于把已经存在的多个 profile、frame、动作布局和执行安全语义冻结成可验证的统一契约。

在两个测试失败、目标 Python 环境未验证、频率文档未完全收口、真实连续段碰撞和非原子派发仍未完成之前，本项目应标记为“架构迁移进行中、可继续开发和 dry-run 验证”，而不是“真实机器人部署基线已完成”。
