# 真机遥操入口重构临时协作说明

审查更新：2026-07-15。状态：进行中。本文件是重构期间的事实记录、边界清单和验收门槛；不要把它当成新的运行入口。整轮重构完成并经过真机负责人确认后，再归档或删除。

## 为什么现在需要更新计划

在 GUI coordinator 文件迁移前，`teleop_runtime/debug/run_vive_rm65_real_teleop.py` 为 **18,836 行**；原记录中的“17,144 行”已经失真。GUI coordinator 迁移后入口为 **16,694 行**；完成 1A/1B 后曾为 **15,834 行**，前一批纯模块迁移后为 **15,300 行**，timing payload 迁移后为 **15,213 行**，共享 writer lifecycle 接线后为 **15,217 行**，兼容 writer observer seam 后为 **15,230 行**，原地轻量去重后为 **15,172 行**，dataset/task layout adapter 迁移后为 **15,145 行**，dataset 实现本体迁移后为 **15,162 行**，本批低风险 debug serializer 的迁移快照为 **14,431 行**。验证期间工作区另有 position-IK/TrackerMotionIntent 相关并发改动，当前入口实测为 **14,536 行**；该增量不属于本批迁移。当前新增/扩展的迁出文件包括 `teleop_runtime/gui/real_teleop_control.py`（2,297 行）、`teleop_runtime/gui/status_payloads.py`（871 行）和 `teleop_runtime/real_teleop/action_debug_payload.py`（556 行），以及 `components.py`（57 行）、`cli_value_parsers.py`（90 行）、`dataset_finalize.py`（60 行）、`dataset_layout.py`（420 行）、`debug_payload_primitives.py`（176 行）、`debug_timing_payload.py`（179 行）和 `debug_ik_payload.py`（815 行）。`teleop_runtime/real_teleop/` 目前为 25 个模块、5,648 行；入口仍保留 CLI 组装、硬件组装、运行时控制、数采副作用和清理逻辑。

这不是简单的“大文件格式化”问题。入口同时承载以下不同风险等级的职责：

- 纯配置/格式化/状态投影：可以在协议固定后低风险迁移；
- source、backend、线程和 writer 生命周期：需要明确所有权后才能迁移；
- command filter、Hold、runtime Home、校准和夹爪门控：属于控制语义，不能和结构重构混批；
- `main()` 的启动、异常路径和反向清理：是系统级编排，必须最后收拢。

本轮目标是让入口最终成为“解析参数 → 创建 runtime → 运行循环 → 统一清理”的薄编排层，同时保持旧入口和旧导入路径兼容。行数只作为风险信号，不作为单独的完成标准。

## 当前事实快照

### 已经迁出的模块

这些模块已经由旧入口导入或通过兼容桥接使用；后续不要把相同实现复制回入口：

| 模块 | 所有权 | 备注 |
| --- | --- | --- |
| `wrist_depth_freshness.py` | 只读的新鲜度状态 | 不启动相机、不读设备 |
| `q6_hold_gesture.py` | Q6 Hold/recalibration 状态机 | 不发送机械臂命令 |
| `debug_timing.py`、`debug_recorders.py` | 观测 timing 和 JSONL debug writer | 不改变控制命令 |
| `main_loop_watchdog.py` | 主循环停滞观察 | 不代替安全停机 |
| `temporary_dual_arm_safety.py`、`fix_arm_collision_guard.py` | 真机专用 command/fault guard | 限制顺序和语义必须保持 |
| `control_profile.py`、`workspace.py`、`side_mapping.py` | 纯参数解析/映射策略 | 不接触硬件 |
| `collection_bridge.py` | 真机状态到 collection contract 的 adapter | 不拥有 writer 生命周期 |
| `foot_pad_controls.py` | 脚踏特殊模式输入 | 设备打开仅发生在显式 runtime 生命周期中 |
| `gui_controls.py`、`gui_tracker_status.py`、`gui_special_arm_status.py`、`gui_vive_calibration.py` | GUI 可用性和只读 payload | 不应在这些模块中发送运动命令 |
| `teleop_runtime/gui/real_teleop_control.py` | `RealTeleopControlInput` 文件本体 | 由旧入口绑定显式 support；暂时仍拥有 GUI 副作用协调 |
| `teleop_runtime/gui/status_payloads.py` | 1A 只读状态/GUI payload 投影 | 只接收 mapping/snapshot；不刷新 source/backend |
| `teleop_runtime/real_teleop/action_debug_payload.py` | 1B action-debug 序列化与聚合 | backend `get_state()` reader 仍归旧入口 |
| `teleop_runtime/real_teleop/debug_timing_payload.py` | 已读取 snapshot 的 timing/fault/channel payload | 不读取 backend、camera 或 source |
| `teleop_runtime/real_teleop/debug_ik_payload.py` | IK/backend/impedance/observation 纯 debug serializer | 不读取 backend、camera 或 source；入口保留 frame coordinator |

入口中的兼容桥仍然是有意保留的：测试、外部脚本和并行修改者目前大量从
`teleop_runtime.debug.run_vive_rm65_real_teleop` 导入旧名字。迁移完成前，旧名字只能 alias 或使用参数转发 wrapper，不能悄悄改变签名或默认值。

### 入口中剩余的主要热点

以下行号是本次审查时的近似边界；每批迁移后必须重新测量，不能机械依赖行号。

| 区域 | 当前范围 | 风险判断 |
| --- | ---: | --- |
| CLI、硬件/任务 profile 合并 | `parse_args()` 起始约 1,662 行 | 纯 value parser 已迁出；Namespace 合并和默认值仍留在入口 |
| VIVE calibration、相机 source helper | 约 3,945–5,349 | capture 有 GUI/设备时序，不应与纯几何函数一起整体搬动 |
| episode layout、action debug reader、dataset recorder | 约 5,364–6,727 | writer、poller、pause/resume 共享生命周期；finalize task 仅迁出状态类 |
| `_build_components()` | 约 6,738–8,052 | 打开/启动硬件、创建 source/backend、失败回滚；高风险 |
| arm/gripper runtime、剩余状态和 command debug reader | 约 8,055–10,238 | 混合只读状态与命令发送，必须继续按副作用分层 |
| IK/backend/impedance 纯 debug serializer 与 frame coordinator | `debug_ik_payload.py` 815 行；入口约 10,239–10,515 | serializer 只接收已读取 mapping/对象；入口保留 command/backend/timing frame 组装 |
| GUI/status/action-debug 兼容导入 | 约 11,329–11,464 | 旧入口保留名字、reader seam 和 support bridge |
| startup self-check、alignment、`main()` | 约 10,532–14,536 | 启动顺序、控制循环和 reverse cleanup；最后拆 |

## 不可破坏的安全和生命周期不变量

任何拆分都必须先在旧入口中写 characterization test 或等价的直接组件测试，并证明下列行为未变：

1. **单向依赖**：兼容入口可以导入 `teleop_runtime.real_teleop`；新模块不得反向导入 `run_vive_rm65_real_teleop`，也不得为了取一个常量导入完整入口。
2. **导入无副作用**：模块 import 不打开串口、相机、脚踏、网络，不创建后台线程，不发送 stop/enable/trajectory/servo 命令。
3. **控制过滤顺序**：当前 command filter 的顺序（init-arm → temporary dual-arm safety → fix-arm，以及必要的同步）必须保持；不能因为拆成多个对象而重复过滤或漏过滤。
4. **Hold 和 Home 门控**：runtime Home 未完成、Hold 生效、校准/episode finalize 期间的 arm/gripper 控制开关必须保持原语义；状态 payload 可以迁移，门控决策不能先拆成无状态 formatter。
5. **启动事务**：设备 preflight、source/backend 创建、state reader、preview 和 writer 的启动顺序不变；中途失败必须释放已经成功创建的资源。
6. **清理事务**：正常退出、键盘中断、异常和部分初始化失败都使用同一套 reverse cleanup；不得通过 `__del__`、import-time thread 或隐式全局对象补救。
7. **数采一致性**：pause/resume、countdown、warmup、writer queue、finalize task 和 metadata 的时间语义必须保持；不能把 recorder 拆成只“看起来独立”但共享未声明可变状态的模块。
8. **硬件访问边界**：纯 payload/策略模块只接收 snapshot、mapping 或 protocol，不主动调用 backend 的读写方法；任何会产生运动或设备状态变化的操作必须留在显式 runtime 层。

## 推荐的谨慎拆分顺序

每一批只处理一个边界。批次之间先跑回归，确认没有行为漂移后再继续。

### 第 0 批：冻结契约和测量基线

- 记录入口行数、已迁出模块行数、兼容 alias 数量和直接测试清单。
- 用 `rg` 建立旧入口公开/半公开导入名清单；任何待删除函数都先确认没有测试、脚本或 notebook 依赖。
- 为 `_build_components()`、`RealTeleopControlInput.poll()`、`RealTeleopDatasetRecorder` 和 `main()` 增加最小的 fake-object/seam 测试；测试不打开真实设备。
- 不在此批改变安全阈值、控制频率、夹爪策略、网络地址或设备 preflight。

#### 第 0 批冻结结果（2026-07-15）

- 入口基线：18,836 行。
- `teleop_runtime/real_teleop/` 基线：17 个 Python 模块、3,289 行。
- GUI coordinator 迁移基线：`teleop_runtime/gui/real_teleop_control.py` 2,297 行；迁移后旧入口 16,694 行。
- 入口到 extracted module 的直接 import site：23；迁移/兼容标记：17。
- 已有直接组件测试：13 个文件；本批新增 `tests/test_real_teleop_refactor_contracts.py`。
- 新增 contract 覆盖：入口和组件关键签名、recorder 生命周期方法集合、`poll()` 的输入/发布顺序。测试只使用 fake terminal/UI/foot-pad，不调用 `_build_components()` 或 `main()`。
- 本批没有移动生产实现，没有改变控制参数和硬件生命周期。上述数值由测试锁定；下一批开始前若有意改变，必须先更新基线和本文件。

#### GUI 文件本体迁移结果（2026-07-15）

- `RealTeleopControlInput` 完整类体已迁到 `teleop_runtime/gui/real_teleop_control.py`，旧入口在原兼容位置导入同名类。
- 旧入口执行显式 support binding；新 GUI 文件不反向导入旧入口，避免循环依赖。
- support 只允许白名单名称；函数通过动态 proxy 读取旧入口当前属性，因此既有 `patch.object(legacy_entrypoint, ...)` 和旧 monkeypatch 点继续生效。
- `RealTeleopControlInput.__module__` 保留旧入口路径，降低 pickle、日志和外部 introspection 变化风险。
- 本迁移没有拆 command filter、硬件组装、`main()`、Hold/校准/夹爪副作用，只改变类体文件归属。

#### 1A/1B 文件本体迁移结果（2026-07-15）

- **1A**：`teleop_runtime/gui/status_payloads.py` 承载 JSON 安全转换、camera self-check/tactile 状态投影、pose-quality/debug 文本等纯只读函数。旧入口保留同名绑定；唯一的旧支持依赖是 `_command_backend_status_text`，通过动态 proxy 保持旧 patch 点有效。
- 1A 没有迁移 `_build_camera_status`、`_build_gripper_status` 或任何 `source.device_info()`、`frame_buffer()`、backend refresh；这些读取边界仍在入口。
- **1B**：`teleop_runtime/real_teleop/action_debug_payload.py` 承载 active-side 推断、已读取 state 的 backend snapshot 序列化、jitter focus 和 `dexteleop_action_debug_v1` 聚合。旧入口仍保留 `_action_debug_backend_snapshot()` 与 `_preloaded_action_debug_backend_snapshot()`，因此 backend `get_state()` 调用次数和时序没有迁移。
- 1B 通过显式 support binding 使用旧入口的 compact serializer、channel/fault payload 和 reader；新模块不反向导入旧入口。
- 1A/1B 后旧入口从 16,694 行降至 **15,834 行**，减少 **860 行**；新增模块行数为 525 和 556。旧名字、recorder 接口和 GUI coordinator 兼容桥继续保留。

#### 低风险纯模块迁移（本批，2026-07-15）

- `teleop_runtime/real_teleop/components.py` 现在承载 `RealTeleopComponents` dataclass。它只描述已创建对象，不负责构造、启动、关闭或发送命令；旧入口通过 alias 保留类身份和 `__module__`。
- `teleop_runtime/real_teleop/cli_value_parsers.py` 承载 9 个无副作用 `argparse` value parser。会修改 `Namespace`、写日志的 `_resolve_movement_multiple_args()` 明确保留在入口。
- `teleop_runtime/gui/status_payloads.py` 扩展承载 IK warning/timing 的 metadata → payload 纯函数；没有把 backend reader 或 `_build_pose_quality_display_payload()` 的 collection 读取边界迁出。
- `teleop_runtime/real_teleop/debug_payload_primitives.py` 承载 pose/qpos/compact metadata 序列化原语；action-debug 模块仍通过 support binding 使用旧入口 reader seam。
- `teleop_runtime/real_teleop/dataset_finalize.py` 承载 `RealDatasetFinalizeTask` 的锁保护生命周期字段和状态方法；writer、线程创建、目录 finalize 和 recorder owner 保留在入口。
- 本批旧入口从 **15,834** 行降至 **15,300** 行，减少 **534** 行；extracted workspace 为 22 个模块、4,228 行。旧入口到 `real_teleop` 的直接 import site 为 23，compatibility marker 为 17。
- 所有新模块均不反向导入 `run_vive_rm65_real_teleop`，import-only/`--help` 不启动设备或线程；`RealDatasetFinalizeTask` 和 `RealTeleopComponents` 的旧路径 introspection 由 bridge 显式保留。

#### Debug timing payload 迁移（本批，2026-07-15）

- `teleop_runtime/real_teleop/debug_timing_payload.py` 承载 `_fault_signature`、phase/latency/fault/channel snapshot payload；输入均为已读取对象，不读取 backend、相机或 source。
- `_camera_status_log_payload`、`_backend_timing_log_payload` 和 `_build_debug_timing_log_frame` 仍保留在旧入口，因为它们分别连接 camera status、backend compact helper 和总 debug frame coordinator。
- 新模块对 `_float_or_none`、`_compact_debug_value` 使用动态 support binding；旧入口 monkeypatch 点和 action-debug 的 support 链继续有效。
- 旧入口从 **15,300** 行降至 **15,213** 行，减少 **87** 行；extracted workspace 为 23 个模块、4,407 行，直接 import site 为 24，compatibility marker 为 17。

#### Shared writer lifecycle boundary（本批，2026-07-15）

- 新增 `teleop_runtime.collection.writer_runtime.CollectionWriterRuntime`，统一封装
  `CollectionWriterObserver` 的 sample/job/flush/close 生命周期。
- GUI `RealTeleopDatasetRecorder` 和 headless `CollectionRuntime` 共用该 writer runtime；
  episode、pause/resume、异步 sample、finalize、GUI 状态和 headless replay 仍由各自 frontend
  持有，避免把控制副作用混入共享 writer 层。
- 旧的 `writer_observer` 属性保留为兼容 alias；没有改变 writer queue/backpressure、T0
  timestamp、ActionRecord 生成或任何真机 dispatch 参数。
- 这不是完整 `DatasetRecorder` 文件迁移；下一步仍需先冻结 recorder lifecycle contract，
  再决定是否把 `RealTeleopDatasetRecorder` 本体迁到 `teleop_runtime.real_teleop`。
- 本批入口行数为 **15,230**，extracted workspace 仍为 23 个模块、4,407 行；新增共享模块
  位于 `teleop_runtime.collection`，不计入 real_teleop extracted workspace。

#### 原地轻量去重（本批，2026-07-15）

- `_impedance_record_compact_debug()` 和 `_impedance_record_debug()` 原先各自维护一份大字段目录，合计 253 个目录条目，其中 116 个字段在两份目录中重复出现。此次只把无副作用的字段目录拆成有序 tuple 片段，再按两个旧函数各自原来的顺序拼接。
- compact/full 两套目录中有意保留不同的 backend target、joint-rate limiter 和 `ik_metadata` 位置；compact 的 `ik_metadata` 仍由原来的专用压缩分支追加，未改变输出顺序、字段缺失规则或值压缩上限。
- 两个函数的签名、`Mapping` 边界、循环和 debug payload 读取逻辑均未改动；未触碰 CLI、主循环、recorder、夹爪或任何硬件控制路径。
- 入口从 **15,230** 行降至 **15,172** 行，减少 **58** 行；`teleop_runtime/real_teleop/` 仍为 23 个模块、4,407 行，未新增依赖边界。
- 新增 contract 覆盖两套目录的长度、关键分叉顺序和 `ik_metadata` 特殊处理，并用完整 synthetic record 验证输出 key order。

#### CLI/config helper 原位重排（本批，2026-07-15）

- 将 `control_profile` compatibility bridge 及 `_impedance_mode_arg()`、工具 profile/TCP、初始对齐摘要、dataset layout 和 round-trip UI helper 整段从旧位置（约 5,218–5,359）原样挪到 `parse_args()` 之前（当前约 1,528–1,669）。
- 函数体、签名、旧导入名和 bridge 文本没有改写；这里只改变同一 Python 模块内的定义位置，使后续 parser/config 模块可以连续截取。
- 保留旧 camera source 与 GUI status bridge 的相邻边界；未跨越 `globals()` support binding、recorder、component factory 或 `main()` 启动清理逻辑。
- 入口仍为 **15,172** 行，extracted workspace 仍为 23 个模块、4,407 行；这是纯布局变更，不计作行数瘦身。
- AST 顶层定义唯一性、`parse_args()` 前置位置、静态编译、`--help`、contract 和全量回归均通过。

#### Dataset/task layout adapter 文件迁移（本批，2026-07-15）

- 将 `_expand_cli_path`、任务注册表查询、正式 dataset layout、round-trip phase 和 collection task UI state 的实现迁到 `teleop_runtime/real_teleop/dataset_layout.py`。
- 入口保留原有 `_...` 名称：无日期的路径/registry/UI helper 使用同一函数对象 alias；需要日期的四个 layout wrapper 继续在旧入口读取 `datetime` 后把 `date_prefix` 显式传入，保留现有 `patch.object(legacy_entrypoint, "datetime", ...)` seam。
- 新模块只委托 `teleop_runtime/gui/collection_tasks.py`，不导入旧入口，不打开设备、不创建线程、不访问 backend/source。旧入口仍保留 `gui_collection_tasks` 全局名，因为 GUI coordinator 的白名单 support contract 仍直接使用该命名空间。
- 入口从 **15,172** 行降至 **15,145** 行，减少 **27** 行；`teleop_runtime/real_teleop/` 变为 24 个模块、4,510 行；直接 import site 为 25，compatibility marker 为 17。
- 新增 contract 检查旧入口 alias 身份、路径/任务 ID 输出及新模块无反向入口依赖；GUI collection visualization 回归保持通过。

#### Dataset/task layout 实现本体归属迁移（本批，2026-07-15）

- 将原 `teleop_runtime/gui/collection_tasks.py` 中的 `FormalDatasetLayout`、任务 registry 解析、general/round-trip layout 计算、namespace 更新和 UI state 生成全部迁入 `teleop_runtime/real_teleop/dataset_layout.py`；该文件现在承接真实实现，不再只是 adapter。
- `teleop_runtime/gui/collection_tasks.py` 缩为 62 行兼容导出，保留旧模块路径、常量、dataclass 和公开函数身份；GUI coordinator 不需要改调用面，也没有第二份布局实现。
- 新实现支持原有 `expand_path` 注入；旧入口的 `_load_task_registry`、`_task_definition_for_key`、`_collection_task_ui_state` 保留原参数签名，`_resolve_formal_dataset_layout` 继续返回旧 tuple，日期仍由入口显式注入。
- 入口因保留 tuple/signature compatibility wrapper 为 **15,162** 行；`teleop_runtime/real_teleop/` 为 24 个模块、4,827 行；直接 import site 25，compatibility marker 17。
- 这次只改变实现所有权和 import 方向：新模块不导入 GUI 或旧入口；不打开设备、不创建线程、不访问 backend/source。

#### 低风险 IK/backend/impedance debug serializer 迁移（本批，2026-07-15）

- 将入口中连续的纯 payload 区域原样迁到
  `teleop_runtime/real_teleop/debug_ik_payload.py`：IK profile、execution step、candidate
  selection、command metadata、backend/impedance snapshot 和 observation pose 的 compact/full
  serializer，以及它们使用的有序字段目录。
- 迁移只改变文件归属，没有重写函数体、签名、字段顺序或缺失值规则。入口通过显式 alias
  保留全部旧名字和常量身份；`_command_for_side`、`_build_ik_debug_log_frame`、camera/backend
  timing reader 及总 frame builder 仍留在入口，继续承接运行时读取和日志帧组装。
- 新模块只接收已捕获的 mapping、`TeleopCommand` 或 observation 对象，并调用
  `debug_payload_primitives`；不读取硬件/backend/source，不发布 GUI 状态，不创建线程，也不
  触碰 command filter、Hold、校准、episode 或夹爪门控。
- 入口从 **15,162** 行降至 **14,431** 行，减少 **731** 行；新模块为 815 行，
  `teleop_runtime/real_teleop/` 变为 25 个模块、5,648 行；直接 import site 为 26，
  compatibility marker 仍为 17。
- 新增 contract 检查入口与新模块的函数/常量 identity、无反向入口依赖，并复用现有
  impedance catalog synthetic record 验证 compact/full 输出 key order。该批不执行真实设备或运动。

#### Shared dispatch seam（本批，2026-07-15）

- `TeleopSession.dispatch_command()` 现在是 `TeleopSession -> Rig` 的显式最终 dispatch
  seam；普通 `run_step()`、GUI recovery 和 headless external command 都通过它进入。
- headless 旧 fake session 仍由窄 fallback 支持，真实 `CollectApp` 不再在 replay、return-home
  或 recovery helper 中直接访问 `rig.send_commands`。
- 该 seam 不重复执行 safety/fault/hold，也不自动改变 state；调用方现有的 guard、recovery
  和 replay safety 顺序保持不变。

### 第 1 批：纯只读 payload 和配置投影

#### 1A 已完成（2026-07-15）

优先抽取不产生副作用、输入输出可完整描述的函数，建议拆成 `status_payloads.py`、`calibration_profile.py` 或同等小模块：

- camera/gripper/tactile/hand 状态格式化；
- pose quality、IK warning/timing 的 compact payload；
- task/dataset layout 的纯路径和 metadata 解析；
- VIVE calibration profile 的 JSON schema 和只读 UI state。

这些函数必须改为显式接收 snapshot/mapping；如果为了生成状态而调用 `get_state()`、`read_*()` 或 backend refresh，就不属于本批。旧入口保留同名 wrapper，直接测试新模块结果与旧路径逐项相等。

### 第 2 批：action debug 与 recorder 的生命周期边界

#### 1B 已完成（2026-07-15）

action-debug 的纯 snapshot/payload 部分已迁到
`teleop_runtime/real_teleop/action_debug_payload.py`。下一步只处理 recorder 的生命周期，不再重复移动 action-debug 实现。

本批已先完成 recorder 之前的最小边界：`RealDatasetFinalizeTask` 已独立成
`dataset_finalize.py`，但 `RealTeleopDatasetRecorder` 仍是唯一的 writer/thread
owner。下一批若迁 recorder，只允许围绕下面冻结的接口和 finalize task 做文件本体
迁移，不得顺手移动 source/backend 创建、夹爪门控或 command filter。

- 设计并冻结 `DatasetRecorder` 接口：`start`、`start_episode`、`record_step(_async)`、`pause/resume`、`stop_episode`、`poll_finalize`、`close`。
- 只在接口和 finalize 状态机有直接测试后，才把 recorder 实现迁到 `teleop_runtime.real_teleop.dataset_recorder`。writer、arm/hand poller 的所有权必须明确归 recorder，不能由入口和新模块同时关闭。
- 每批保留 metadata 中的 `source_entrypoint` 和 episode/finalize 字段，避免“文件变薄但数据语义变了”。

### 第 3 批：GUI 事件分发与 `RealTeleopControlInput`

`RealTeleopControlInput` 的文件本体现在位于 `teleop_runtime/gui/real_teleop_control.py`（约 2,297 行，包含 support binding）。它仍是最大的状态耦合点之一；下一步不要再按方法长度直接切文件，建议分三步：

1. 先定义不可变 `GuiControlState`、`GuiControlEvents` 和 event-consumption contract；已有 `gui_controls.py` 只负责 controls projection。
2. 抽取无副作用的事件分类、active-controls 计算和状态转换 reducer；旧类继续拥有设备/episode side effect。
3. 仅当 reducer 的转换矩阵覆盖 Hold、runtime Home、VIVE calibration、fix/init-arm、episode countdown、finalize 和 gripper gating 后，再把副作用协调器迁出。

校准采样、脚踏设备、gripper warmup 和 episode recorder 不能在同一批移动。它们都应由控制协调器调用，而不是让 GUI 模块直接持有 backend。

#### GUI 部分的具体规划（先设计，后迁移）

现有 `gui_controls.py` 只表达“给定快照后哪些按钮可用”；它不等同于完整的
`RealTeleopControlInput` 状态，也不应直接扩张成一个新的万能类。GUI 相关拆分应保持四层：

1. **输入适配层**：把 terminal、Web UI、foot-pad 和 Q6 gesture 统一成带
   `source/name/payload/timestamp` 的规范事件。适配层不访问 backend，不调用 recorder，不修改 `args`。
2. **状态/reducer 层**：只维护 `should_quit`、Hold（含 reason）、runtime-home/calibration 请求、episode start/stop、visualization/debug toggle、tracker lock 等 GUI 可观察状态。状态转换必须是可测的，并能拒绝 busy/运行中不允许的事件。
3. **副作用协调层**：暂时仍由 `RealTeleopControlInput` 持有；它按显式 effect 调用 fix/init-arm controller、VIVE calibration、dataset recorder、gripper policy、safety baseline reset 和 publish。任何会影响机械臂、夹爪、writer 或 source 的操作都留在这一层。
4. **只读发布层**：复用现有 `gui_controls.py`、`gui_*_status.py` 和状态 payload；只把 snapshot 转成 UI JSON，不从 backend 主动刷新硬件。

第 0 批已冻结的 `poll()` 顺序是：

```text
terminal.poll
→ foot_pad.consume_events / route
→ terminal.should_quit
→ terminal.consume_calibration_request
→ ui.consume_events / dispatch
→ q6 gesture update
→ raw tracker publish
→ gripper status publish
```

后续 GUI 第 1 小批只抽“输入适配 + 纯 reducer”，保留上述顺序的 characterization test；不能先把
`request_hold()`、`_handle_fix_arm_c_event()`、`_handle_init_arm_g_event()`、校准采样或夹爪力度写入迁出。特别注意：

- `request_hold()` 同时改变 Hold、gripper enable、Q6 gesture state 和 UI；它不是普通 UI setter；
- `episode_busy()` 是 gripper force、movement multiple、VIVE calibration 和 task 切换的共同门控；
- fix/init-arm、foot-pad 和 tracker swap 有互斥和 runtime Home 前置条件；
- UI 发布节流（raw pose、IK、camera、gripper）属于可观察契约，不能因拆分而改变频率或首次强制发布行为。

只有当 event/reducer 矩阵覆盖 pre-home、running、hold、episode countdown、recording、paused、finalizing、fix-arm、init-arm 和 calibration 等状态后，才允许把副作用协调器拆成小型 effect handlers。每个 handler 必须有明确的 owner 和 close 顺序，不能让 GUI 层和 runtime 层重复清理同一个对象。

### 第 4 批：组件组装和 runtime loop

- 先抽出只描述依赖的 `RealTeleopRuntimeConfig`/组件工厂接口，再迁移 `_build_components()`；工厂必须提供成功初始化列表和失败回滚路径。
- 新 runtime 的启动顺序固定为：参数/配置 → preflight → source/backend → session/retargeter → state readers/preview → recorder/debug → GUI；具体以当前行为测试为准，不能凭感觉重排。
- `main()` 最后才变成薄 orchestrator。它只应负责解析、创建 runtime、进入 loop、处理退出原因和调用一次统一 cleanup；command filter 和设备细节不得重新塞回 `main.py`。
- 如果需要兼容旧的 monkeypatch/import 点，保留极薄的 legacy bridge，并在文档中登记删除条件。

## 每批验收门槛

纯结构迁移也必须执行以下门槛；涉及真实硬件的测试不在默认验收内：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  teleop_runtime/debug/run_vive_rm65_real_teleop.py \
  teleop_runtime/real_teleop/*.py
python -m teleop_runtime.debug.run_vive_rm65_real_teleop --help
git diff --check
```

另外，每批需要：

- 新模块直接单测；旧入口导入名和代表性行为回归；
- import-only 检查，确认没有启动硬件、线程或 socket；
- 对 command/filter/lifecycle 变更执行 fake backend 测试；
- 用 `wc -l` 更新本文件的快照，记录迁出模块和未迁移热点；
- 一批一个提交/可回滚边界，避免把格式化、重命名、安全策略和协议改动混在一起。

全量测试若受环境依赖（例如 `pyarrow`、UDP、相机驱动）影响，必须把失败原因和实际执行的测试范围记录下来，不能把“未运行”写成“通过”。

### 本次审查验证（2026-07-15）

- 本批契约与纯模块回归：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_real_teleop_refactor_contracts.py`：**13 passed**。
- real-teleop broad 回归：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_real_teleop*.py tests/test_dmtac_gripper_bimanual_config.py`：**226 passed**。
- 新增 contract 覆盖：components/parser/finalize/debug primitive/debug timing payload/dataset layout 的旧符号身份、snapshot 等价性、无反向入口依赖、finalize task 状态生命周期、纯 parser 边界，以及阻抗 debug 两套字段目录的完整顺序和 `ik_metadata` 特殊处理。
- 入口及 `teleop_runtime.real_teleop/*.py` 静态编译通过。
- `python -m teleop_runtime.debug.run_vive_rm65_real_teleop --help` 通过。
- `git diff --check` 通过。
- 未启动真实相机、串口、机械臂、夹爪、脚踏或网络控制；未执行 motion/servo/force-control。

#### Dataset/task layout adapter verification（2026-07-15）

- GUI collection visualization：**103 passed**。
- 契约额外验证 `gui.collection_tasks` 与 `real_teleop.dataset_layout` 的 dataclass/函数身份、旧入口 helper 签名和无反向入口依赖。
- 排除已知无关的 `tests/test_mujoco_temp_pipeline.py` 后工作区全量回归：**1531 passed, 47 skipped**。
- 未排除的全量回归：**1534 passed, 47 skipped, 1 failed**；唯一失败为
  `tests/test_mujoco_temp_pipeline.py::test_temp_policy_checkpoint_round_trip`，是当前 PyTorch
  `torch.load(weights_only=True)` 对 `torch.torch_version.TorchVersion` 未 allowlist 的环境/既有
  checkpoint 兼容问题，与本次 teleop dataset layout 迁移无调用路径关系，未在本批顺手修改。

#### Shared writer lifecycle boundary verification（2026-07-15）

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_teleop_collection_runtime_action_record.py tests/test_real_teleop_refactor_contracts.py tests/test_collection_storage_queue.py tests/test_lifecycle_cleanup.py`：**29 passed**。
- 合并执行真实 teleop、collection、deploy、recovery、queue 和 lifecycle 回归：**351 passed, 2 skipped**。
- GUI、headless、deploy 三个 `--help` import/argument checks 均通过。
- `teleop_runtime/collection/writer_runtime.py`、`collection/runtime.py` 和 GUI entrypoint 静态编译通过；`git diff --check` 通过。
- 当前工作区全量回归：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`：**1530 passed, 47 skipped**。
- 未启动真实相机、串口、机械臂、夹爪、脚踏或网络控制；未执行 motion/servo/force-control。

## 暂不拆分或暂不改动的内容

- `run_vive_sapien_virtual` 的共享 helper：先确认依赖方向和 virtual/real 两条入口的共同契约，不能为瘦身直接复制或反向导入。
- `_dispatch_q6_gripper_control`、arm command backend、emergency/stop/force-control 路径：这是运动副作用边界，除非有专门的行为测试和安全审查，不做纯结构顺手移动。
- DMTAC tactile process、camera startup/self-check、USB/preflight、CANFD/impedance 参数：设备生命周期和性能/安全语义尚未冻结。
- 任何“顺便”修改默认端口、速度、力矩、控制频率、Hold 阈值、workspace/collision 限制或 cleanup timeout 的提交。

## 完成标准

重构完成不以“删除了多少行”判定，而以以下条件判定：

1. 旧 CLI 和旧导入路径在兼容期仍可用，且所有 bridge 都有明确删除条件；
2. 入口主要剩下 parser、runtime factory、loop 和 cleanup，纯策略/格式化/serializer 不再驻留其中；
3. 新模块没有反向依赖入口，没有 import-time 硬件副作用；
4. command、Hold/Home、数采 finalize、启动事务和 reverse cleanup 的 characterization tests 通过；
5. 每个迁移批次都有可回滚边界、测试记录和更新后的行数快照；
6. 真机验证单独进行，并由操作者显式确认，不以静态测试替代硬件安全验证。
